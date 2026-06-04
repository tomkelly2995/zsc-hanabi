#!/usr/bin/env python3
"""
scripts/evaluate_ll_inference.py

Log-likelihood partner inference evaluation.

For each BC profile in a run's pool, we compute how well it explains the
observed behaviour of a new partner (agent B from a different run).  We then
deploy the best-response checkpoint (BR_k) that was trained specifically
against the best-fitting profile, instead of the generic final oracle.

This measures whether diverse BC profiles (created via temperature scheduling)
allow an agent to identify and adapt to a new partner's convention at test time.

Usage
─────
  # Compare generic oracle vs LL-adaptive oracle (run1 A meets run2 B):
  python scripts/evaluate_ll_inference.py \\
      --run1 runs/alpha03_seed42 --run2 runs/alpha03_seed123 \\
      --n_probe 2 --n_eval 500

  # Sweep over different probe budgets:
  python scripts/evaluate_ll_inference.py \\
      --run1 runs/alpha03_seed42 --run2 runs/alpha03_seed123 \\
      --probe_sweep --n_eval 500
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import collections
import glob
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from src.jax_agents.simulation import _bc_net
from src.jax_agents.rppo_agent_jax import (
    HIDDEN_DIM, run_K_rppo_probe_episodes,
)
from src.jax_networks.rppo_actor_critic import RPPOActorCritic
from src.population.rppo_policy_jax import RPPOPolicy
from src.utils.checkpointing import load_agent, load_metadata
from src.jax_env.tiny_hanabi import (
    reset      as _env_reset,
    step       as _env_step,
    get_obs    as _env_get_obs,
    legal_mask as _env_legal_mask,
)
from main_jax import run_joint_rppo_episodes

_rppo_model = RPPOActorCritic()


# ---------------------------------------------------------------------------
# GRU Adapter module
# ---------------------------------------------------------------------------

class GRUAdapter(nn.Module):
    """Residual bottleneck adapter placed between the GRU output and trunk.

    Architecture: h(64) → Dense(adapter_dim, ReLU) → Dense(64) → h + delta

    The up-projection is zero-initialised, so the adapter starts as an exact
    identity mapping.  Only adapter params are updated at test time; all BR
    params remain frozen.  Gradient does not flow through time (GRU hidden
    states are stop_gradient'd before the adapter sees them).

    Intuition: the GRU's 64-dim carry encodes a mixture of all conventions
    learned during training.  The adapter learns a low-rank (adapter_dim-wide)
    re-weighting of that carry that emphasises the current partner's convention,
    before the frozen trunk projects it to the 128-dim action-prediction space.
    """
    adapter_dim: int = 8
    gru_dim:     int = HIDDEN_DIM   # 64

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
        down  = nn.relu(nn.Dense(self.adapter_dim, name="down")(h))
        # Small non-zero init (std=0.01) so gradient flows to the down projection
        # immediately.  The residual connection still keeps the adapter close to
        # identity at game 0 (delta ≈ 0.01 scale), while allowing both projections
        # to receive gradient from the start.
        delta = nn.Dense(self.gru_dim, name="up",
                         kernel_init=nn.initializers.normal(stddev=0.01),
                         bias_init=nn.initializers.zeros)(down)
        return h + delta


_adapter_model = GRUAdapter()


def init_adapter(adapter_dim: int = 8, seed: int = 0) -> dict:
    """Initialise adapter variables.  Returns {'params': {'down': …, 'up': …}}.

    Up-projection uses small normal init (std=0.01) so gradient reaches both
    projections from game 1.  Output is still near-identity at initialisation.
    """
    key      = jax.random.PRNGKey(seed)
    dummy_h  = jnp.zeros(HIDDEN_DIM)
    adapter  = GRUAdapter(adapter_dim=adapter_dim)
    return adapter.init(key, dummy_h)


def fine_tune_adapter(ac_params: dict, adapter_vars: dict,
                      episode: dict, player_id: int,
                      lr: float = 1e-4, grad_clip: float = 0.1,
                      adapter_dim: int = 8,
                      _diag: bool = False) -> dict:
    """One REINFORCE step on adapter params only.

    Frozen:  all BR network params (obs_proj, gru, trunk, actor, critic).
    Updated: GRUAdapter (down + up projections, ~1096 params for adapter_dim=8).

    Data flow:
        obs → [obs_proj] → [gru] → h_stopped → [ADAPTER] → [trunk] → logits

    The GRU hidden state is stop_gradient'd after each step (same as last-layer
    fine-tuning), so gradients only flow through the adapter and frozen trunk/actor
    weights (the latter via the closure — no grad computed for them).
    """
    obs_list    = []
    action_list = []
    for step in episode["steps"]:
        if step["current_player"] == player_id:
            obs_list.append(jnp.array(step["player_obs"][player_id], jnp.float32))
            action_list.append(step["action"])

    if not obs_list:
        return adapter_vars

    score  = float(episode["score"])
    inner  = ac_params["params"]          # frozen BR params (closure)
    adapter = GRUAdapter(adapter_dim=adapter_dim)

    def loss_fn(adap_vars):
        h     = jnp.zeros(HIDDEN_DIM)
        total = jnp.array(0.0)
        for obs, action in zip(obs_list, action_list):
            # Run frozen BR model to get new GRU hidden state
            _, _, new_h, _ = _rppo_model.apply({"params": inner}, obs, h)
            new_h = jax.lax.stop_gradient(new_h)
            # Adapter transforms the stopped GRU output (trainable)
            adapted = adapter.apply(adap_vars, new_h)
            # Apply frozen trunk and actor manually (same params, no grad)
            trunk  = jax.nn.relu(adapted @ inner["trunk"]["kernel"]
                                 + inner["trunk"]["bias"])
            logits = trunk @ inner["actor"]["kernel"] + inner["actor"]["bias"]
            total  = total + jax.nn.log_softmax(logits)[action]
            h = new_h
        return -(score / 5.0) * total / len(action_list)

    grads = jax.grad(loss_fn)(adapter_vars)

    if _diag:
        flat  = jax.tree_util.tree_leaves(grads)
        gnorm = float(sum(jnp.sum(g ** 2) for g in flat) ** 0.5)
        npar  = sum(g.size for g in flat)
        print(f"    [adapter diag] score={score:.0f}  steps={len(obs_list)}"
              f"  grad_norm={gnorm:.4f}  n_params={npar}")

    grads       = jax.tree_util.tree_map(lambda g: jnp.clip(g, -grad_clip, grad_clip), grads)
    new_adap    = jax.tree_util.tree_map(lambda p, g: p - lr * g, adapter_vars, grads)
    return new_adap


def _run_adapter_episode(key, ac_params_a: dict, adapter_vars_a: dict,
                         ac_params_b: dict, adapter_vars_b: dict | None,
                         adapter_dim: int = 8) -> dict:
    """Run one episode where A (and optionally B) uses a GRU adapter.

    A always uses its adapter.  B uses its adapter only if adapter_vars_b is not None;
    otherwise B uses its BR params unmodified.
    """
    inner_a  = ac_params_a["params"]
    inner_b  = ac_params_b["params"]
    adapter  = GRUAdapter(adapter_dim=adapter_dim)

    key, k_reset, k_steps = jax.random.split(key, 3)
    state      = _env_reset(k_reset)
    prev_state = state
    prev_act   = jnp.array(0, dtype=jnp.int32)

    h_a = jnp.zeros(HIDDEN_DIM)
    h_b = jnp.zeros(HIDDEN_DIM)

    steps    = []
    step_key = k_steps
    done     = False

    while not done:
        step_key, k_act = jax.random.split(step_key)
        cur   = int(jnp.argmax(state.cur_player_idx))
        obs0  = jnp.array(_env_get_obs(state, prev_state, prev_act, 0))
        obs1  = jnp.array(_env_get_obs(state, prev_state, prev_act, 1))
        legal = jnp.array(_env_legal_mask(state), dtype=jnp.float32)

        if cur == 0:   # A — always with adapter
            _, _, new_h, _ = _rppo_model.apply({"params": inner_a}, obs0, h_a)
            adapted = adapter.apply(adapter_vars_a, new_h)
            trunk   = jax.nn.relu(adapted @ inner_a["trunk"]["kernel"]
                                  + inner_a["trunk"]["bias"])
            logits  = trunk @ inner_a["actor"]["kernel"] + inner_a["actor"]["bias"]
            h_a     = new_h
        else:          # B — with adapter only if adapter_vars_b is provided
            if adapter_vars_b is not None:
                _, _, new_h, _ = _rppo_model.apply({"params": inner_b}, obs1, h_b)
                adapted = adapter.apply(adapter_vars_b, new_h)
                trunk   = jax.nn.relu(adapted @ inner_b["trunk"]["kernel"]
                                      + inner_b["trunk"]["bias"])
                logits  = trunk @ inner_b["actor"]["kernel"] + inner_b["actor"]["bias"]
                h_b     = new_h
            else:
                logits, _, h_b, _ = _rppo_model.apply({"params": inner_b}, obs1, h_b)

        masked = jnp.where(legal > 0, logits, -1e9)
        probs  = jax.nn.softmax(masked)
        action = int(jax.random.choice(k_act, 8, p=probs))

        steps.append({
            "current_player": cur,
            "player_obs": {0: np.array(obs0), 1: np.array(obs1)},
            "action": action,
        })

        act_jnp    = jnp.array(action, dtype=jnp.int32)
        new_state, _, done_jnp = _env_step(state, act_jnp)
        done       = bool(done_jnp)
        prev_state = state
        prev_act   = act_jnp
        state      = new_state

    return {"steps": steps, "score": int(state.score)}


# ---------------------------------------------------------------------------
# Action decoding  (tiny Hanabi: 2 colors, 2 ranks, 2 cards per hand)
#   0-1  : Discard card 0/1
#   2-3  : Play   card 0/1
#   4-5  : Hint   color 0/1  (color 0 = Red, color 1 = White)
#   6-7  : Hint   rank  1/2
# ---------------------------------------------------------------------------

_COLOR_NAMES = ["Red", "White"]
_RANK_NAMES  = ["1",   "2"]

def decode_action(a: int) -> str:
    if 0 <= a <= 1: return f"Discard card {a}"
    if 2 <= a <= 3: return f"Play card {a - 2}"
    if 4 <= a <= 5: return f"Hint color {_COLOR_NAMES[a - 4]}"
    if 6 <= a <= 7: return f"Hint rank {_RANK_NAMES[a - 6]}"
    return f"Unknown({a})"


def print_game(ep: dict, game_idx: int, pol_a_label: str, pol_b_label: str) -> None:
    """Print a human-readable move-by-move transcript of one episode."""
    player_labels = {0: f"P0({pol_a_label})", 1: f"P1({pol_b_label})"}
    print(f"  ┌─ Game {game_idx+1}  score={int(ep['score'])} "
          f"  [{len(ep['steps'])} steps] ─────────────────────")
    for t, step in enumerate(ep["steps"]):
        p   = step["current_player"]
        act = decode_action(step["action"])
        print(f"  │  step {t+1:>2}  {player_labels[p]:<20} {act}")
    print(f"  └{'─'*55}")


def action_summary(episodes: list) -> dict:
    """
    Summarise action-type frequencies across a list of episodes.
    Returns counts for: discard, play, hint_color, hint_rank (per player).
    """
    counts = {p: {"discard": 0, "play": 0, "hint_color": 0, "hint_rank": 0}
              for p in [0, 1]}
    for ep in episodes:
        for step in ep["steps"]:
            p = step["current_player"]
            a = step["action"]
            if 0 <= a <= 1:   counts[p]["discard"]    += 1
            elif 2 <= a <= 3: counts[p]["play"]        += 1
            elif 4 <= a <= 5: counts[p]["hint_color"]  += 1
            elif 6 <= a <= 7: counts[p]["hint_rank"]   += 1
    return counts


def print_action_summary(episodes: list, pol_a_label: str, pol_b_label: str,
                         header: str = "") -> None:
    """Print per-player action-type breakdown and mean score for a batch."""
    counts = action_summary(episodes)
    scores = [ep["score"] for ep in episodes]
    labels = {0: f"P0({pol_a_label})", 1: f"P1({pol_b_label})"}
    if header:
        print(f"  ── {header} ──")
    print(f"  Mean score: {np.mean(scores):.2f} ± {np.std(scores):.2f}  "
          f"({len(episodes)} games)")
    for p in [0, 1]:
        c   = counts[p]
        tot = sum(c.values()) or 1
        print(f"  {labels[p]:<22}  "
              f"play={c['play']:>3} ({100*c['play']/tot:.0f}%)  "
              f"discard={c['discard']:>3} ({100*c['discard']/tot:.0f}%)  "
              f"hint_color={c['hint_color']:>3} ({100*c['hint_color']/tot:.0f}%)  "
              f"hint_rank={c['hint_rank']:>3} ({100*c['hint_rank']/tot:.0f}%)")


# ---------------------------------------------------------------------------
# Log-likelihood computation
# ---------------------------------------------------------------------------

def _partner_obs_actions(episodes: list, partner_id: int) -> list[tuple]:
    """
    Extract (obs, action) pairs for the partner player from episode dicts.

    Returns a list of per-episode sequences: each element is a list of
    (obs_array, action_int) tuples for that episode.
    """
    seqs = []
    for ep in episodes:
        seq = []
        for step in ep["steps"]:
            if step["current_player"] == partner_id:
                seq.append((step["player_obs"][partner_id], step["action"]))
        if seq:
            seqs.append(seq)
    return seqs


def compute_log_likelihood(bc_params, episode_seqs: list) -> float:
    """
    Compute total log-likelihood of observed partner (obs, action) pairs
    under a BC model.

    The BC model is a GRU: hidden state is maintained across timesteps within
    an episode and reset to zero between episodes.

    Args:
        bc_params    : BC model param dict (numpy, will be moved to device).
        episode_seqs : list of per-episode (obs, action) sequences from
                       _partner_obs_actions().

    Returns:
        Total log-likelihood (sum over all observed actions, all episodes).
        More negative = worse fit; less negative = better fit.
    """
    bc_params_j = jax.tree_util.tree_map(jnp.array, bc_params)
    total_ll = 0.0

    for seq in episode_seqs:
        h = jnp.zeros(HIDDEN_DIM)
        for obs, action in seq:
            obs_j = jnp.array(obs)
            logits = _bc_net.apply({"params": bc_params_j}, obs_j, h)
            log_probs = jax.nn.log_softmax(logits)
            total_ll += float(log_probs[int(action)])
            h = _bc_net.apply(
                {"params": bc_params_j}, h, jnp.array(action, jnp.int32),
                method=_bc_net.gru_step,
            )

    return total_ll


def infer_profile(profile_pool: list, episode_seqs: list,
                  available_ks: set | None = None) -> tuple[int, list]:
    """
    Select the BC profile that best explains the observed partner behaviour.

    Args:
        profile_pool : list of bc_model_params dicts (from checkpoint payload).
        episode_seqs : per-episode (obs, action) sequences from _partner_obs_actions().
        available_ks : if provided, restrict argmax to these indices only.
                       This is important because profile_pool always has one more
                       entry than there are BR checkpoints: the final profile was
                       extracted after the last training iteration but no BR was
                       ever trained against it.  Passing {k for k,_ in brs}
                       prevents the argmax from selecting that orphaned profile
                       and falling back to the numerically-nearest BR (which may
                       not be the true second-best match).

    Returns:
        (best_idx, ll_scores) where best_idx is the argmax profile index and
        ll_scores is the list of per-profile log-likelihoods.
    """
    ll_scores = [
        compute_log_likelihood(bc_params, episode_seqs)
        for bc_params in profile_pool
    ]
    if available_ks:
        best_idx = max(
            (k for k in range(len(ll_scores)) if k in available_ks),
            key=lambda k: ll_scores[k],
        )
    else:
        best_idx = int(np.argmax(ll_scores))
    return best_idx, ll_scores


# ---------------------------------------------------------------------------
# BR checkpoint loading
# ---------------------------------------------------------------------------

def _find_br(run_dir: str, agent_id: str, k: int) -> str | None:
    """
    Find the BR checkpoint for agent_id at profile index k.

    Checks run_dir directly first, then run_dir/brs/ (the default layout
    produced by training with --br_save_dir set to a brs/ subdirectory).
    """
    for search_dir in [run_dir, os.path.join(run_dir, "brs")]:
        path = os.path.join(search_dir, f"{agent_id}_br{k:03d}.pkl")
        if os.path.exists(path):
            return path
    return None


def load_br(path: str, device=None) -> RPPOPolicy:
    """Load a BR_k checkpoint as an RPPOPolicy."""
    device = device or jax.devices()[0]
    with open(path, "rb") as f:
        payload = pickle.load(f)
    params = jax.device_put(
        jax.tree_util.tree_map(jnp.array, payload["params"]), device
    )
    return RPPOPolicy(ac_params=params, player_id=payload["player_id"])


def _list_brs(run_dir: str, agent_id: str) -> list[tuple[int, str]]:
    """
    Return (k, path) pairs for all available BR checkpoints, sorted by k.

    Searches both run_dir and run_dir/brs/.
    """
    patterns = [
        os.path.join(run_dir, f"{agent_id}_br*.pkl"),
        os.path.join(run_dir, "brs", f"{agent_id}_br*.pkl"),
    ]
    paths = sorted(set(p for pat in patterns for p in glob.glob(pat)))
    result = []
    for p in paths:
        name = os.path.basename(p)
        k = int(name[len(agent_id) + 3 : -4])
        result.append((k, p))
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _eval_policy_vs_partner(policy_a, policy_b, n_episodes: int,
                             seed: int, device) -> dict:
    """Run n_episodes and return mean, std, and raw scores."""
    key      = jax.random.PRNGKey(seed)
    episodes = run_joint_rppo_episodes(policy_a, policy_b, n_episodes,
                                       key=key, device=device)
    scores   = np.array([ep["score"] for ep in episodes], dtype=np.float32)
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores)),
            "scores": scores}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_br(run_dir: str, agent_id: str, best_k: int,
                brs: list, device, quiet: bool = False) -> tuple[RPPOPolicy | None, int]:
    """Load BR_k, falling back to nearest available k if exact match missing."""
    br_path = _find_br(run_dir, agent_id, best_k)
    if br_path is None and brs:
        available_ks = [k for k, _ in brs]
        best_k       = min(available_ks, key=lambda k: abs(k - best_k))
        br_path      = _find_br(run_dir, agent_id, best_k)
        if not quiet:
            print(f"    (exact BR not found; using BR_{best_k:03d})")
    if br_path is None:
        return None, best_k
    return load_br(br_path, device=device), best_k


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_ll_eval(run1_dir: str, run2_dir: str, n_probe: int, n_eval: int,
                seed: int, device) -> None:
    sep = "=" * 65
    print(sep)
    print("Log-likelihood inference evaluation  (fully symmetric)")
    print(f"  Run 1: {run1_dir}")
    print(f"  Run 2: {run2_dir}")
    print(f"  Probe episodes: {n_probe}   Eval episodes: {n_eval}")
    print(sep)

    # ── Load both agents ──────────────────────────────────────────────────────
    path_a1 = os.path.join(run1_dir, "agent_A_final.pkl")
    path_b2 = os.path.join(run2_dir, "agent_B_final.pkl")
    for p in (path_a1, path_b2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    pol_a1 = load_agent(path_a1, device=device)
    pol_b2 = load_agent(path_b2, device=device)

    meta_a1 = load_metadata(path_a1)
    meta_b2 = load_metadata(path_b2)
    pool_a  = meta_a1.get("profile_pool", [])
    pool_b  = meta_b2.get("profile_pool", [])
    brs_a1  = _list_brs(run1_dir, "agent_A")
    brs_b2  = _list_brs(run2_dir, "agent_B")

    print(f"\n  agent_A (run1): iter={meta_a1['iteration']}  "
          f"profiles={len(pool_a)}  BRs={len(brs_a1)}")
    print(f"  agent_B (run2): iter={meta_b2['iteration']}  "
          f"profiles={len(pool_b)}  BRs={len(brs_b2)}")

    if not pool_a or not pool_b:
        print("\n  ERROR: profile_pool missing from one or both checkpoints.")
        return

    # ── Probe episodes: both agents play, we observe both sides ──────────────
    print(f"\n  Running {n_probe} probe episode(s) …")
    probe_eps   = run_joint_rppo_episodes(
        pol_a1, pol_b2, n_probe, key=jax.random.PRNGKey(seed), device=device
    )
    probe_score = float(np.mean([ep["score"] for ep in probe_eps]))

    # A observes B's actions to infer B's profile from A's pool
    seqs_b_seen_by_a = _partner_obs_actions(probe_eps, pol_b2.player_id)   # player 1
    # B observes A's actions to infer A's profile from B's pool
    seqs_a_seen_by_b = _partner_obs_actions(probe_eps, pol_a1.player_id)   # player 0

    n_obs_a = sum(len(s) for s in seqs_b_seen_by_a)
    n_obs_b = sum(len(s) for s in seqs_a_seen_by_b)
    print(f"  Probe score: {probe_score:.2f}  |  "
          f"A observed {n_obs_a} B-actions, B observed {n_obs_b} A-actions")

    # ── Independent LL inference ──────────────────────────────────────────────
    avail_a = {k for k, _ in brs_a1}
    avail_b = {k for k, _ in brs_b2}
    best_k, ll_scores_a = infer_profile(pool_a, seqs_b_seen_by_a, available_ks=avail_a)
    best_j, ll_scores_b = infer_profile(pool_b, seqs_a_seen_by_b, available_ks=avail_b)

    print(f"\n  A's LL scores (observing B's actions under A's BC profiles):")
    for k, ll in enumerate(ll_scores_a):
        marker = " ← A selects BR" if k == best_k else ""
        print(f"    profile {k:3d}: {ll:8.3f}{marker}")

    print(f"\n  B's LL scores (observing A's actions under B's BC profiles):")
    for j, ll in enumerate(ll_scores_b):
        marker = " ← B selects BR" if j == best_j else ""
        print(f"    profile {j:3d}: {ll:8.3f}{marker}")

    # ── Load adaptive policies ────────────────────────────────────────────────
    pol_a1_adaptive, best_k = _resolve_br(run1_dir, "agent_A", best_k, brs_a1, device)
    pol_b2_adaptive, best_j = _resolve_br(run2_dir, "agent_B", best_j, brs_b2, device)

    has_a_br = pol_a1_adaptive is not None
    has_b_br = pol_b2_adaptive is not None

    # ── Evaluate all combinations ─────────────────────────────────────────────
    print(f"\n  {'Condition':<30}  {'Score':>7}  {'±':>7}")
    print("  " + "-" * 46)

    def _run(label, pa, pb, s):
        r = _eval_policy_vs_partner(pa, pb, n_eval, s, device)
        print(f"  {label:<30}  {r['mean']:>7.3f}  {r['std']:>7.3f}")
        return r

    res_both_generic   = _run("A_generic   vs B_generic",   pol_a1, pol_b2, seed+1)

    if has_a_br:
        res_a_adaptive = _run(f"A_BR{best_k:03d}    vs B_generic",
                              pol_a1_adaptive, pol_b2, seed+2)
    if has_b_br:
        res_b_adaptive = _run(f"A_generic   vs B_BR{best_j:03d}",
                              pol_a1, pol_b2_adaptive, seed+3)
    if has_a_br and has_b_br:
        res_both_adaptive = _run(f"A_BR{best_k:03d}    vs B_BR{best_j:03d}",
                                 pol_a1_adaptive, pol_b2_adaptive, seed+4)

    # ── Self-play ceiling ─────────────────────────────────────────────────────
    path_b1 = os.path.join(run1_dir, "agent_B_final.pkl")
    if os.path.exists(path_b1):
        pol_b1 = load_agent(path_b1, device=device)
        _run("A_generic   vs B1 (self-play)", pol_a1, pol_b1, seed+5)

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    baseline = res_both_generic["mean"]
    if has_a_br:
        print(f"  A-only inference gain:   {res_a_adaptive['mean'] - baseline:+.3f}")
    if has_b_br:
        print(f"  B-only inference gain:   {res_b_adaptive['mean'] - baseline:+.3f}")
    if has_a_br and has_b_br:
        print(f"  Both-adaptive gain:      {res_both_adaptive['mean'] - baseline:+.3f}")


def run_probe_sweep(run1_dir: str, run2_dir: str, n_eval: int,
                    seed: int, device) -> None:
    """Evaluate LL inference (both agents adaptive) across different probe budgets."""
    sep = "=" * 65
    print(sep)
    print("Probe budget sweep  (both agents adaptive)")
    print(f"  Run 1: {run1_dir}   Run 2: {run2_dir}")
    print(sep)

    path_a1 = os.path.join(run1_dir, "agent_A_final.pkl")
    path_b2 = os.path.join(run2_dir, "agent_B_final.pkl")
    pol_a1  = load_agent(path_a1, device=device)
    pol_b2  = load_agent(path_b2, device=device)
    meta_a1 = load_metadata(path_a1)
    meta_b2 = load_metadata(path_b2)
    pool_a  = meta_a1.get("profile_pool", [])
    pool_b  = meta_b2.get("profile_pool", [])
    brs_a1  = _list_brs(run1_dir, "agent_A")
    brs_b2  = _list_brs(run2_dir, "agent_B")

    res_generic = _eval_policy_vs_partner(pol_a1, pol_b2, n_eval, seed, device)
    print(f"\n  Generic baseline (both): {res_generic['mean']:.3f} ± {res_generic['std']:.3f}")
    print(f"\n  {'n_probe':>8}  {'A→k':>6}  {'B→j':>6}  {'score':>8}  {'delta':>8}")
    print("  " + "-" * 46)

    for n_probe in [1, 2, 4, 8, 16]:
        probe_eps = run_joint_rppo_episodes(
            pol_a1, pol_b2, n_probe,
            key=jax.random.PRNGKey(seed + n_probe), device=device
        )
        seqs_b = _partner_obs_actions(probe_eps, pol_b2.player_id)
        seqs_a = _partner_obs_actions(probe_eps, pol_a1.player_id)

        best_k, _ = infer_profile(pool_a, seqs_b, available_ks={k for k, _ in brs_a1})
        best_j, _ = infer_profile(pool_b, seqs_a, available_ks={k for k, _ in brs_b2})

        pol_a_br, best_k = _resolve_br(run1_dir, "agent_A", best_k, brs_a1, device)
        pol_b_br, best_j = _resolve_br(run2_dir, "agent_B", best_j, brs_b2, device)

        if pol_a_br and pol_b_br:
            res = _eval_policy_vs_partner(
                pol_a_br, pol_b_br, n_eval, seed + 100 + n_probe, device
            )
            delta = res["mean"] - res_generic["mean"]
            print(f"  {n_probe:>8}  {best_k:>6}  {best_j:>6}  "
                  f"{res['mean']:>8.3f}  {delta:>+8.3f}")
        else:
            print(f"  {n_probe:>8}  {best_k:>6}  {best_j:>6}  {'no BRs':>8}  {'n/a':>8}")


# ---------------------------------------------------------------------------
# Online incremental LL inference
# ---------------------------------------------------------------------------

def run_online_ll_eval(run1_dir: str, run2_dir: str, n_eval: int,
                       seed: int, device,
                       score_weighted: bool = False,
                       window: int = 50,
                       print_games: bool = False) -> None:
    """
    Online incremental windowed LL inference — no separate probe phase.

    After every game, both agents update their running per-profile LL score
    over a sliding window of recent games and re-select their best-fit BR.
    Only the last `window` games count toward the LL estimate, preventing
    early evidence from permanently anchoring the selection.

    score_weighted=True weights each game's LL contribution by score/5 with
    a 0.1 floor, so high-scoring games carry more weight.

    print_games=True prints a move-by-move transcript of every episode plus
    a 50-game action summary, for qualitative analysis.
    """
    sep = "=" * 65
    mode = "score-weighted" if score_weighted else "unweighted"
    print(sep)
    print(f"Online windowed LL inference  [{mode}, window={window}]")
    print(f"  Run 1: {run1_dir}")
    print(f"  Run 2: {run2_dir}")
    print(f"  Eval episodes: {n_eval}")
    print(sep)

    # ── Load agents and pools ─────────────────────────────────────────────────
    path_a1 = os.path.join(run1_dir, "agent_A_final.pkl")
    path_b2 = os.path.join(run2_dir, "agent_B_final.pkl")
    for p in (path_a1, path_b2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    pol_a_generic = load_agent(path_a1, device=device)
    pol_b_generic = load_agent(path_b2, device=device)

    meta_a = load_metadata(path_a1)
    meta_b = load_metadata(path_b2)
    pool_a = meta_a.get("profile_pool", [])
    pool_b = meta_b.get("profile_pool", [])
    brs_a  = _list_brs(run1_dir, "agent_A")
    brs_b  = _list_brs(run2_dir, "agent_B")

    print(f"\n  agent_A (run1): iter={meta_a['iteration']}  "
          f"profiles={len(pool_a)}  BRs={len(brs_a)}")
    print(f"  agent_B (run2): iter={meta_b['iteration']}  "
          f"profiles={len(pool_b)}  BRs={len(brs_b)}")

    if not pool_a or not pool_b:
        print("\n  ERROR: profile_pool missing from one or both checkpoints.")
        return

    # ── Generic baseline (batch) for comparison ───────────────────────────────
    print(f"\n  Running generic baseline ({n_eval} episodes) …", end="", flush=True)
    res_generic = _eval_policy_vs_partner(
        pol_a_generic, pol_b_generic, n_eval, seed, device
    )
    print(f"  {res_generic['mean']:.3f} ± {res_generic['std']:.3f}")

    # ── Windowed LL state ─────────────────────────────────────────────────────
    # Per profile: deque of recent per-game LL contributions (length ≤ window)
    ll_hist_a = [collections.deque(maxlen=window) for _ in pool_a]
    ll_hist_b = [collections.deque(maxlen=window) for _ in pool_b]

    def _running(hist): return sum(hist)

    cur_k: int | None = None
    cur_j: int | None = None
    br_cache_a: dict[int, RPPOPolicy] = {}
    br_cache_b: dict[int, RPPOPolicy] = {}

    scores       = []
    selections_a = []
    selections_b = []
    window_eps   = []   # episodes in the current 50-game display window

    key = jax.random.PRNGKey(seed + 1)

    print(f"\n  Running {n_eval} online inference episodes …\n")

    for game_idx in range(n_eval):
        key, subkey = jax.random.split(key)

        # Labels for printing
        a_label = f"generic" if cur_k is None else f"BR{cur_k:03d}"
        b_label = f"generic" if cur_j is None else f"BR{cur_j:03d}"

        # Deploy current best estimate
        pol_a = br_cache_a[cur_k] if cur_k is not None and cur_k in br_cache_a \
                else pol_a_generic
        pol_b = br_cache_b[cur_j] if cur_j is not None and cur_j in br_cache_b \
                else pol_b_generic

        selections_a.append(cur_k)
        selections_b.append(cur_j)

        # Run one episode
        [ep] = run_joint_rppo_episodes(pol_a, pol_b, 1, key=subkey, device=device)
        score = float(ep["score"])
        scores.append(score)
        window_eps.append(ep)

        if print_games:
            print_game(ep, game_idx, a_label, b_label)

        # Score weight
        w = max(score / 5.0, 0.1) if score_weighted else 1.0

        # Extract partner sequences
        seqs_b = _partner_obs_actions([ep], pol_b_generic.player_id)
        seqs_a = _partner_obs_actions([ep], pol_a_generic.player_id)

        # Update windowed LL: push new contribution, oldest auto-drops via maxlen
        for k, bc in enumerate(pool_a):
            ll_hist_a[k].append(w * compute_log_likelihood(bc, seqs_b))
        for j, bc in enumerate(pool_b):
            ll_hist_b[j].append(w * compute_log_likelihood(bc, seqs_a))

        # Re-select best profile — restrict to profiles that have a BR checkpoint
        avail_a_set = {k for k, _ in brs_a}
        avail_b_set = {k for k, _ in brs_b}
        ll_now_a    = [_running(h) for h in ll_hist_a]
        ll_now_b    = [_running(h) for h in ll_hist_b]
        new_k = max(avail_a_set, key=lambda k: ll_now_a[k])
        new_j = max(avail_b_set, key=lambda j: ll_now_b[j])

        # Load BR if changed (cached)
        if new_k != cur_k and new_k not in br_cache_a:
            pol, actual_k = _resolve_br(run1_dir, "agent_A", new_k, brs_a, device,
                                        quiet=True)
            if pol is not None:
                br_cache_a[new_k] = pol
                br_cache_a[actual_k] = pol
        if new_k in br_cache_a:
            cur_k = new_k

        if new_j != cur_j and new_j not in br_cache_b:
            pol, actual_j = _resolve_br(run2_dir, "agent_B", new_j, brs_b, device,
                                        quiet=True)
            if pol is not None:
                br_cache_b[new_j] = pol
                br_cache_b[actual_j] = pol
        if new_j in br_cache_b:
            cur_j = new_j

        # Every 50 games: print summary
        if (game_idx + 1) % 50 == 0:
            w50 = np.array([ep["score"] for ep in window_eps])
            print(f"\n  ── 50-game summary  (games {game_idx-48}–{game_idx+1}) ──")
            print(f"  A→k={cur_k}  B→j={cur_j}  "
                  f"mean={w50.mean():.3f} ± {w50.std():.2f}")
            if print_games:
                print_action_summary(window_eps, a_label, b_label)
            # Top-3 profiles by current windowed LL (restricted to profiles with BRs)
            top_a = sorted(avail_a_set, key=lambda k: -ll_now_a[k])[:3]
            top_b = sorted(avail_b_set, key=lambda j: -ll_now_b[j])[:3]
            print(f"  A LL top-3: " +
                  "  ".join(f"k={k}({ll_now_a[k]:.1f})" for k in top_a))
            print(f"  B LL top-3: " +
                  "  ".join(f"j={j}({ll_now_b[j]:.1f})" for j in top_b))
            window_eps = []

    # ── Final results ─────────────────────────────────────────────────────────
    scores_arr = np.array(scores, dtype=np.float32)
    disp_w     = max(1, n_eval // 10)

    print(f"\n{'='*65}")
    print(f"  {'Metric':<42}  {'Score':>7}")
    print("  " + "-" * 52)
    print(f"  {'Generic baseline (batch)':<42}  {res_generic['mean']:>7.3f}")
    print(f"  {'Online windowed LL (all games)':<42}  {scores_arr.mean():>7.3f}")
    print(f"  {f'  first {disp_w} games':<42}  {scores_arr[:disp_w].mean():>7.3f}")
    print(f"  {f'  last  {disp_w} games':<42}  {scores_arr[-disp_w:].mean():>7.3f}")

    gain_all  = scores_arr.mean() - res_generic["mean"]
    gain_late = scores_arr[-disp_w:].mean() - res_generic["mean"]
    print(f"\n  Inference gain (all games):           {gain_all:+.3f}")
    print(f"  Inference gain (last {disp_w} games):      {gain_late:+.3f}")

    print(f"\n  Score trajectory ({disp_w}-game windows):")
    for i in range(0, n_eval, disp_w):
        chunk = scores_arr[i : i + disp_w]
        a_sel = [s for s in selections_a[i : i + disp_w] if s is not None]
        b_sel = [s for s in selections_b[i : i + disp_w] if s is not None]
        a_mode = int(np.bincount(a_sel).argmax()) if a_sel else "generic"
        b_mode = int(np.bincount(b_sel).argmax()) if b_sel else "generic"
        print(f"    games {i+1:>4}–{i+len(chunk):>4}:  "
              f"mean={chunk.mean():.3f}  A→{a_mode}  B→{b_mode}")

    ll_final_a = [_running(h) for h in ll_hist_a]
    ll_final_b = [_running(h) for h in ll_hist_b]
    top_k = sorted(avail_a_set, key=lambda k: -ll_final_a[k])[:3]
    top_j = sorted(avail_b_set, key=lambda j: -ll_final_b[j])[:3]
    print(f"\n  A final LL top-3 (window={window}): " +
          "  ".join(f"k={k}({ll_final_a[k]:.1f})" for k in top_k))
    print(f"  B final LL top-3 (window={window}): " +
          "  ".join(f"j={j}({ll_final_b[j]:.1f})" for j in top_j))
    print(f"  Final selection: A→k={cur_k}  B→j={cur_j}")


# ---------------------------------------------------------------------------
# Last-layer fine-tuning
# ---------------------------------------------------------------------------

def fine_tune_last_layer(ac_params: dict, episode: dict, player_id: int,
                         lr: float = 1e-4, grad_clip: float = 0.1,
                         original_actor: dict | None = None,
                         constraint_alpha: float = 0.0,
                         _diag: bool = False) -> dict:
    """
    One REINFORCE gradient step on the actor head only (128×8 + 8 = 1032 params).

    Frozen:  obs_proj, gru, trunk, critic.
    Updated: actor kernel + bias.

    The gradient is: ∂/∂actor [ -(score/5) * mean(log π(aₜ|oₜ)) ]

    jax.lax.stop_gradient on the GRU hidden state prevents backprop
    through time so only the single-step actor head receives gradient.
    Score-0 games contribute zero gradient naturally (no explicit skip needed).

    If original_actor and constraint_alpha > 0 are supplied, a soft parameter
    constraint is applied after the gradient step:
        actor = (1 - α) * actor_updated + α * original_actor
    This pulls the adapted policy back toward its BR initialisation after
    every game, preventing long-term drift without blocking short-term adaptation.
    """
    obs_list    = []
    action_list = []
    for step in episode["steps"]:
        if step["current_player"] == player_id:
            obs_list.append(jnp.array(step["player_obs"][player_id], jnp.float32))
            action_list.append(step["action"])

    if not obs_list:
        return ac_params

    score        = float(episode["score"])
    inner        = ac_params["params"]
    actor_params = inner["actor"]
    frozen       = {k: v for k, v in inner.items() if k != "actor"}

    def loss_fn(actor_p):
        full  = {**frozen, "actor": actor_p}
        h     = jnp.zeros(HIDDEN_DIM)
        total = jnp.array(0.0)
        for obs, action in zip(obs_list, action_list):
            logits, _, h, _ = _rppo_model.apply({"params": full}, obs, h)
            h     = jax.lax.stop_gradient(h)
            total = total + jax.nn.log_softmax(logits)[action]
        return -(score / 5.0) * total / len(action_list)

    grads     = jax.grad(loss_fn)(actor_params)
    if _diag:
        grad_norm = float(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)) ** 0.5)
        print(f"    [diag] score={score:.0f}  steps={len(obs_list)}"
              f"  grad_norm={grad_norm:.4f}  lr={lr}  effective_step={lr*min(grad_norm,grad_clip):.2e}")
    grads     = jax.tree_util.tree_map(lambda g: jnp.clip(g, -grad_clip, grad_clip), grads)
    new_actor = jax.tree_util.tree_map(lambda p, g: p - lr * g, actor_params, grads)
    if _diag:
        delta = jax.tree_util.tree_map(lambda n, o: n - o, new_actor, actor_params)
        delta_norm = float(sum(jnp.sum(d**2) for d in jax.tree_util.tree_leaves(delta)) ** 0.5)
        print(f"    [diag] param_delta_norm={delta_norm:.2e}  (actor has {sum(p.size for p in jax.tree_util.tree_leaves(actor_params))} params)")

    # Soft constraint: blend back toward the original BR params
    if original_actor is not None and constraint_alpha > 0.0:
        new_actor = jax.tree_util.tree_map(
            lambda updated, orig: (1 - constraint_alpha) * updated + constraint_alpha * orig,
            new_actor, original_actor,
        )

    return {"params": {**frozen, "actor": new_actor}}


def run_probe_finetune_eval(run1_dir: str, run2_dir: str,
                            n_probe: int, n_eval: int,
                            seed: int, device,
                            ft_lr: float = 1e-4,
                            ft_clip: float = 0.1,
                            ft_alpha: float = 0.0,
                            symmetric_ll: bool = False,
                            symmetric_ft: bool = False,
                            symmetric_cb: bool = False,
                            cb_beta: float = 0.0,
                            cb_warmup: int = 0,
                            ft_diag: bool = False) -> None:
    """
    Fixed probe → LL inference → BR selection → last-layer fine-tuning.

    Three orthogonal symmetry flags (each can be used independently):

      symmetric_ll : B also runs LL inference to pick its own BR_j.
                     Without this B stays as the generic oracle.
      symmetric_ft : B also fine-tunes its actor head after every game,
                     exactly like A does.  Requires symmetric_ll so B
                     starts from a BR rather than the generic policy.
      symmetric_cb : B also blends its logits with its own BC-profile-j
                     at weight β.  Requires symmetric_ll.

    cb_beta > 0 enables convention-conditioned belief blending for A.
    cb_warmup linearly ramps β from 0 → cb_beta over the first cb_warmup games.
    """
    ll_tag = "LL✓" if symmetric_ll else "LL✗"
    ft_tag = "FT✓" if symmetric_ft else "FT✗"
    cb_tag = f"CB✓β={cb_beta}" if (cb_beta > 0) else "CB✗"
    sym_tag = f"A+B [{ll_tag} {ft_tag} {cb_tag}]" if (symmetric_ll or symmetric_ft) \
              else f"A-only [{cb_tag}]"
    cb_str = f"  cb_beta={cb_beta}  cb_warmup={cb_warmup}" if cb_beta > 0 else ""
    sep = "=" * 65
    print(sep)
    print(f"Probe → LL inference → BR + last-layer fine-tuning  [{sym_tag}]")
    print(f"  Run 1: {run1_dir}   Run 2: {run2_dir}")
    print(f"  Probe: {n_probe}   Eval: {n_eval}   lr={ft_lr}   clip={ft_clip}{cb_str}")
    print(sep)

    # ── Load ──────────────────────────────────────────────────────────────────
    path_a1 = os.path.join(run1_dir, "agent_A_final.pkl")
    path_b2 = os.path.join(run2_dir, "agent_B_final.pkl")
    for p in (path_a1, path_b2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    pol_a_generic = load_agent(path_a1, device=device)
    pol_b_generic = load_agent(path_b2, device=device)
    meta_a        = load_metadata(path_a1)
    meta_b        = load_metadata(path_b2)
    pool_a        = meta_a.get("profile_pool", [])
    pool_b        = meta_b.get("profile_pool", []) if symmetric_ll else []
    need_b_pool   = symmetric_ll or symmetric_ft or symmetric_cb
    brs_a         = _list_brs(run1_dir, "agent_A")
    brs_b         = _list_brs(run2_dir, "agent_B") if need_b_pool else []

    print(f"\n  agent_A (run1): iter={meta_a['iteration']}  "
          f"profiles={len(pool_a)}  BRs={len(brs_a)}")
    if need_b_pool:
        print(f"  agent_B (run2): iter={meta_b['iteration']}  "
              f"profiles={len(pool_b)}  BRs={len(brs_b)}")
    if not pool_a:
        print("\n  ERROR: profile_pool missing for A.")
        return

    # ── Generic baseline ──────────────────────────────────────────────────────
    print(f"\n  Running generic baseline ({n_eval} episodes) …", end="", flush=True)
    res_generic = _eval_policy_vs_partner(
        pol_a_generic, pol_b_generic, n_eval, seed, device
    )
    print(f"  {res_generic['mean']:.3f} ± {res_generic['std']:.3f}")

    # ── Phase 1: probe → LL inference ─────────────────────────────────────────
    print(f"\n  Phase 1 — running {n_probe} probe episodes …")
    probe_eps   = run_joint_rppo_episodes(
        pol_a_generic, pol_b_generic, n_probe,
        key=jax.random.PRNGKey(seed + 1), device=device
    )
    probe_score = float(np.mean([ep["score"] for ep in probe_eps]))

    # A infers from B's observed actions
    seqs_b        = _partner_obs_actions(probe_eps, pol_b_generic.player_id)
    best_k, ll_a  = infer_profile(pool_a, seqs_b, available_ks={k for k, _ in brs_a})
    top3_a        = sorted((k for k, _ in brs_a), key=lambda k: -ll_a[k])[:3]
    print(f"  Probe score: {probe_score:.2f}  →  A selects BR_{best_k:03d}")
    print("  A top-3 LL: " + "  ".join(f"k={k}({ll_a[k]:.1f})" for k in top3_a))

    pol_br_a, best_k = _resolve_br(run1_dir, "agent_A", best_k, brs_a, device)
    if pol_br_a is None:
        print(f"  ERROR: no BR found for A k={best_k}")
        print(f"{run1_dir}, agent_A, {best_k}, {brs_a}, {device}")
        return

    # B infers its own BR (symmetric_ll) or stays generic
    if symmetric_ll and pool_b:
        seqs_a       = _partner_obs_actions(probe_eps, pol_a_generic.player_id)
        best_j, ll_b = infer_profile(pool_b, seqs_a, available_ks={k for k, _ in brs_b})
        top3_b       = sorted((j for j, _ in brs_b), key=lambda j: -ll_b[j])[:3]
        print(f"                              B selects BR_{best_j:03d}")
        print("  B top-3 LL: " + "  ".join(f"j={j}({ll_b[j]:.1f})" for j in top3_b))
        pol_br_b, best_j = _resolve_br(run2_dir, "agent_B", best_j, brs_b, device)
        if pol_br_b is None:
            print(f"  ERROR: no BR found for B j={best_j}")
            return
        b_label = f"BR{best_j:03d}"
    else:
        pol_br_b = pol_b_generic
        best_j   = None
        b_label  = "generic"

    # ── Both BRs fixed baseline (no fine-tuning) ──────────────────────────────
    br_label = f"A_BR{best_k:03d} vs B_{b_label}"
    print(f"\n  Running {br_label} fixed baseline ({n_eval} episodes) …",
          end="", flush=True)
    res_br_fixed = _eval_policy_vs_partner(
        pol_br_a, pol_br_b, n_eval, seed + 2, device
    )
    print(f"  {res_br_fixed['mean']:.3f} ± {res_br_fixed['std']:.3f}")

    # ── Phase 2: fine-tuning loop ─────────────────────────────────────────────
    cb_mode  = f" + CB β={cb_beta} warmup={cb_warmup}" if cb_beta > 0 else ""
    ft_who   = "A+B fine-tune" if symmetric_ft else "A fine-tunes"
    b_status = f"{b_label} {'fine-tunes' if symmetric_ft else 'fixed'}"
    print(f"\n  Phase 2 — {n_eval} episodes  ({ft_who}{cb_mode}, B={b_status}) …")
    print(f"  (first JAX grad call compiles; subsequent calls are fast)\n")

    # A's mutable state
    ac_params_a    = dict(pol_br_a.ac_params)
    orig_actor_a   = ac_params_a["params"]["actor"]
    player_id_a    = pol_br_a.player_id

    # B's mutable state (only needed if B fine-tunes)
    if symmetric_ft:
        ac_params_b  = dict(pol_br_b.ac_params)
        orig_actor_b = ac_params_b["params"]["actor"]
        player_id_b  = pol_br_b.player_id
    else:
        ac_params_b  = pol_br_b.ac_params   # fixed reference, never updated

    # BC profiles for blending
    bc_params_k = pool_a[best_k] if cb_beta > 0 else None
    bc_params_j = (pool_b[best_j] if (cb_beta > 0 and symmetric_cb
                                       and best_j is not None and pool_b)
                   else None)

    key        = jax.random.PRNGKey(seed + 3)
    scores_ft  = []
    window_eps = []

    for game_idx in range(n_eval):
        key, subkey = jax.random.split(key)

        # Current β (with optional warmup ramp)
        beta_t = (cb_beta * min(1.0, game_idx / cb_warmup)
                  if (cb_beta > 0 and cb_warmup > 0) else cb_beta)
        beta_b = beta_t if (symmetric_cb and bc_params_j is not None) else 0.0

        if cb_beta > 0:
            # Step-by-step episode with BC logit blending
            ep = _run_conv_belief_episode(
                subkey, ac_params_a, bc_params_k, ac_params_b, beta_t,
                bc_params_j=bc_params_j, beta_b=beta_b,
            )
        else:
            # Vectorised episode runner
            pol_a_cur = RPPOPolicy(
                ac_params=jax.device_put(ac_params_a, device),
                player_id=player_id_a,
            )
            if symmetric_ft:
                pol_b_cur = RPPOPolicy(
                    ac_params=jax.device_put(ac_params_b, device),
                    player_id=player_id_b,
                )
                [ep] = run_joint_rppo_episodes(
                    pol_a_cur, pol_b_cur, 1, key=subkey, device=device
                )
            else:
                [ep] = run_joint_rppo_episodes(
                    pol_a_cur, pol_br_b, 1, key=subkey, device=device
                )

        scores_ft.append(float(ep["score"]))
        window_eps.append(ep)

        # Fine-tune A
        diag_this = ft_diag and game_idx < 5
        ac_params_a = fine_tune_last_layer(
            ac_params_a, ep, player_id_a,
            lr=ft_lr, grad_clip=ft_clip,
            original_actor=orig_actor_a,
            constraint_alpha=ft_alpha,
            _diag=diag_this,
        )

        # Fine-tune B (symmetric_ft only)
        if symmetric_ft:
            ac_params_b = fine_tune_last_layer(
                ac_params_b, ep, player_id_b,
                lr=ft_lr, grad_clip=ft_clip,
                original_actor=orig_actor_b,
                constraint_alpha=ft_alpha,
            )

        if (game_idx + 1) % 50 == 0:
            w      = np.array([e["score"] for e in window_eps])
            cb_info = f"  β_now={beta_t:.3f}" if cb_beta > 0 else ""
            sym_info = " (A+B ft)" if symmetric_ft else ""
            print(f"  ── 50-game summary (games {game_idx-48}–{game_idx+1}) ──")
            print(f"  mean={w.mean():.3f} ± {w.std():.2f}  "
                  f"lr={ft_lr}  α={ft_alpha}{cb_info}{sym_info}")
            a_lbl = f"BR{best_k:03d}+ft"
            b_lbl = f"{b_label}+ft" if symmetric_ft else b_label
            print_action_summary(window_eps, a_lbl, b_lbl)
            window_eps = []

    # ── Results ───────────────────────────────────────────────────────────────
    ft_arr = np.array(scores_ft, dtype=np.float32)
    disp_w = max(1, n_eval // 10)

    print(f"\n{'='*65}")
    print(f"  {'Condition':<42}  {'Score':>7}  {'vs generic':>10}")
    print("  " + "-" * 62)

    def _row(label, mean, baseline=res_generic["mean"]):
        print(f"  {label:<42}  {mean:>7.3f}  {mean-baseline:>+10.3f}")

    b_ft_label = f"{b_label}+ft" if symmetric_ft else b_label
    _row("Generic (both)",                          res_generic["mean"])
    _row(br_label + " fixed",                       res_br_fixed["mean"])
    _row(f"A_BR{best_k:03d}+ft vs B_{b_ft_label} (all)", ft_arr.mean())
    _row(f"  first {disp_w} games",                ft_arr[:disp_w].mean())
    _row(f"  last  {disp_w} games",                ft_arr[-disp_w:].mean())

    sp_path = os.path.join(run1_dir, "agent_B_final.pkl")
    if os.path.exists(sp_path):
        pol_b1 = load_agent(sp_path, device=device)
        r_sp   = _eval_policy_vs_partner(pol_a_generic, pol_b1, n_eval, seed+4, device)
        _row("Self-play ceiling (A_generic+B1)", r_sp["mean"])

    print(f"\n  Fine-tune trajectory ({disp_w}-game windows):")
    for i in range(0, n_eval, disp_w):
        chunk = ft_arr[i : i + disp_w]
        print(f"    games {i+1:>4}–{i+len(chunk):>4}:  mean={chunk.mean():.3f}")


# ---------------------------------------------------------------------------
# Convention-conditioned belief  (BC logit blending at test time)
# ---------------------------------------------------------------------------

def _run_conv_belief_episode(key, ac_params_a: dict, bc_params_k,
                              ac_params_b: dict, beta_a: float,
                              bc_params_j=None, beta_b: float = 0.0) -> dict:
    """
    Run one episode with optional BC logit blending for A and/or B.

        A's logits: (1 - β_a) * br_logits_a + β_a * bc_logits_k
        B's logits: (1 - β_b) * br_logits_b + β_b * bc_logits_j  (if bc_params_j given)

    Each agent maintains separate GRU hidden states for its BR model and its
    BC model, so the BC model's convention signal conditions on full history.

    Returns an episode dict: {"steps": [...], "score": int}.
    """
    bc_k    = jax.tree_util.tree_map(jnp.array, bc_params_k)
    bc_j    = jax.tree_util.tree_map(jnp.array, bc_params_j) \
              if bc_params_j is not None else None
    inner_a = ac_params_a["params"]
    inner_b = ac_params_b["params"]

    key, k_reset, k_steps = jax.random.split(key, 3)
    state      = _env_reset(k_reset)
    prev_state = state
    prev_act   = jnp.array(0, dtype=jnp.int32)

    h_br_a = jnp.zeros(HIDDEN_DIM)   # A's BR GRU
    h_bc_a = jnp.zeros(HIDDEN_DIM)   # A's BC-k GRU
    h_br_b = jnp.zeros(HIDDEN_DIM)   # B's BR GRU
    h_bc_b = jnp.zeros(HIDDEN_DIM)   # B's BC-j GRU (only used if bc_j is not None)

    steps    = []
    partner_pred_max_probs = []  # Track partner prediction confidence
    partner_pred_entropy = []    # Track partner prediction entropy
    step_key = k_steps
    done     = False

    while not done:
        step_key, k_act = jax.random.split(step_key)
        cur   = int(jnp.argmax(state.cur_player_idx))
        obs0  = jnp.array(_env_get_obs(state, prev_state, prev_act, 0))
        obs1  = jnp.array(_env_get_obs(state, prev_state, prev_act, 1))
        legal = jnp.array(_env_legal_mask(state), dtype=jnp.float32)

        if cur == 0:   # A's turn
            br_logits, _, h_br_a, partner_logits_a = _rppo_model.apply({"params": inner_a}, obs0, h_br_a)
            # Track partner (B) prediction stats
            partner_probs_a = jax.nn.softmax(partner_logits_a)
            partner_pred_max_probs.append(float(jnp.max(partner_probs_a)))
            partner_entropy_a = -jnp.sum(partner_probs_a * jnp.log(partner_probs_a + 1e-8))
            partner_pred_entropy.append(float(partner_entropy_a))

            bc_logits             = _bc_net.apply({"params": bc_k}, obs0, h_bc_a)
            final_logits          = (1.0 - beta_a) * br_logits + beta_a * bc_logits
            masked_logits         = jnp.where(legal > 0, final_logits, -1e9)
            probs                 = jax.nn.softmax(masked_logits)
            action                = int(jax.random.choice(k_act, 8, p=probs))
            h_bc_a = _bc_net.apply(
                {"params": bc_k}, h_bc_a,
                jnp.array(action, jnp.int32), method=_bc_net.gru_step,
            )
        else:          # B's turn
            br_logits, _, h_br_b, partner_logits_b = _rppo_model.apply({"params": inner_b}, obs1, h_br_b)
            # Track partner (A) prediction stats
            partner_probs_b = jax.nn.softmax(partner_logits_b)
            partner_pred_max_probs.append(float(jnp.max(partner_probs_b)))
            partner_entropy_b = -jnp.sum(partner_probs_b * jnp.log(partner_probs_b + 1e-8))
            partner_pred_entropy.append(float(partner_entropy_b))

            if bc_j is not None and beta_b > 0.0:
                bc_logits_b  = _bc_net.apply({"params": bc_j}, obs1, h_bc_b)
                final_logits = (1.0 - beta_b) * br_logits + beta_b * bc_logits_b
            else:
                final_logits = br_logits
            masked_logits = jnp.where(legal > 0, final_logits, -1e9)
            probs         = jax.nn.softmax(masked_logits)
            action        = int(jax.random.choice(k_act, 8, p=probs))
            if bc_j is not None and beta_b > 0.0:
                h_bc_b = _bc_net.apply(
                    {"params": bc_j}, h_bc_b,
                    jnp.array(action, jnp.int32), method=_bc_net.gru_step,
                )

        steps.append({
            "current_player": cur,
            "player_obs": {0: np.array(obs0), 1: np.array(obs1)},
            "action": action,
        })

        act_jnp    = jnp.array(action, dtype=jnp.int32)
        new_state, _, done_jnp = _env_step(state, act_jnp)
        done       = bool(done_jnp)
        prev_state = state
        prev_act   = act_jnp
        state      = new_state

    return {
        "steps": steps,
        "score": int(state.score),
        "partner_pred_max_probs": partner_pred_max_probs,
        "partner_pred_entropy": partner_pred_entropy,
    }


def run_conv_belief_eval(run1_dir: str, run2_dir: str,
                         n_probe: int, n_eval: int,
                         seed: int, device,
                         cb_beta: float = 0.3,
                         cb_warmup: int = 0,
                         symmetric: bool = True) -> None:
    """
    Convention-conditioned belief evaluation.

    Phase 1 (n_probe games): generic probe → LL inference → select BR_k for A
    (and optionally BR_j for B if symmetric=True).

    Phase 2 (n_eval games): A's logits are blended with BC-profile-k logits
    using weight β.  B uses its (optionally inferred) BR unchanged.

    The BC model encodes "what would a convention-k player do at this state?"
    Blending it into A's logits nudges A toward convention-k consistent actions
    without any gradient updates — pure test-time adaptation.

    cb_beta=0    → pure BR (identical to fixed probe eval)
    cb_beta=1    → pure BC (acts like the convention follower)
    cb_beta=0.3  → recommended: 30% BC signal, 70% BR
    cb_warmup>0  → ramp β linearly from 0 to cb_beta over the first cb_warmup
                   games, giving the BC GRU time to build up meaningful history
                   before its signal is trusted
    """
    sep  = "=" * 65
    mode = "symmetric" if symmetric else "one-sided"
    warmup_str = f", warmup={cb_warmup}" if cb_warmup > 0 else ""
    print(sep)
    print(f"Convention-conditioned belief  [β={cb_beta}{warmup_str}, {mode}]")
    print(f"  Run 1: {run1_dir}")
    print(f"  Run 2: {run2_dir}")
    print(f"  Probe: {n_probe}   Eval: {n_eval}   β={cb_beta}   warmup={cb_warmup}")
    print(sep)

    # ── Load ──────────────────────────────────────────────────────────────────
    path_a1 = os.path.join(run1_dir, "agent_A_final.pkl")
    path_b2 = os.path.join(run2_dir, "agent_B_final.pkl")
    for p in (path_a1, path_b2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    pol_a_generic = load_agent(path_a1, device=device)
    pol_b_generic = load_agent(path_b2, device=device)
    meta_a        = load_metadata(path_a1)
    meta_b        = load_metadata(path_b2)
    pool_a        = meta_a.get("profile_pool", [])
    pool_b        = meta_b.get("profile_pool", []) if symmetric else []
    brs_a         = _list_brs(run1_dir, "agent_A")
    brs_b         = _list_brs(run2_dir, "agent_B") if symmetric else []

    print(f"\n  agent_A (run1): iter={meta_a['iteration']}  "
          f"profiles={len(pool_a)}  BRs={len(brs_a)}")
    if symmetric:
        print(f"  agent_B (run2): iter={meta_b['iteration']}  "
              f"profiles={len(pool_b)}  BRs={len(brs_b)}")

    if not pool_a:
        print("\n  ERROR: profile_pool missing for A.")
        return

    # ── Generic baseline ──────────────────────────────────────────────────────
    print(f"\n  Running generic baseline ({n_eval} episodes) …", end="", flush=True)
    res_generic = _eval_policy_vs_partner(
        pol_a_generic, pol_b_generic, n_eval, seed, device
    )
    print(f"  {res_generic['mean']:.3f} ± {res_generic['std']:.3f}")

    # ── Phase 1: probe → LL inference ─────────────────────────────────────────
    print(f"\n  Phase 1 — running {n_probe} probe episodes …")
    probe_eps   = run_joint_rppo_episodes(
        pol_a_generic, pol_b_generic, n_probe,
        key=jax.random.PRNGKey(seed + 1), device=device,
    )
    probe_score = float(np.mean([ep["score"] for ep in probe_eps]))

    seqs_b       = _partner_obs_actions(probe_eps, pol_b_generic.player_id)
    best_k, ll_a = infer_profile(pool_a, seqs_b, available_ks={k for k, _ in brs_a})
    top3_a       = sorted((k for k, _ in brs_a), key=lambda k: -ll_a[k])[:3]
    print(f"  Probe score: {probe_score:.2f}  →  A selects BR_{best_k:03d}  "
          f"(BC profile k={best_k})")
    print("  A top-3 LL: " + "  ".join(f"k={k}({ll_a[k]:.1f})" for k in top3_a))

    pol_br_a, best_k = _resolve_br(run1_dir, "agent_A", best_k, brs_a, device)
    if pol_br_a is None:
        print(f"  ERROR: no BR found for A k={best_k}")
        print(f"{run1_dir}, agent_A, {best_k}, {brs_a}, {device}")
        return
    bc_params_k = pool_a[best_k]   # BC profile used to condition A's logits

    if symmetric and pool_b:
        seqs_a       = _partner_obs_actions(probe_eps, pol_a_generic.player_id)
        best_j, ll_b = infer_profile(pool_b, seqs_a, available_ks={k for k, _ in brs_b})
        top3_b       = sorted((j for j, _ in brs_b), key=lambda j: -ll_b[j])[:3]
        print(f"                              B selects BR_{best_j:03d}")
        print("  B top-3 LL: " + "  ".join(f"j={j}({ll_b[j]:.1f})" for j in top3_b))
        pol_br_b, best_j = _resolve_br(run2_dir, "agent_B", best_j, brs_b, device)
        if pol_br_b is None:
            print(f"  ERROR: no BR found for B j={best_j}")
            return
        b_label = f"BR{best_j:03d}"
    else:
        pol_br_b = pol_b_generic
        best_j   = None
        b_label  = "generic"

    # ── Fixed BR baseline (β=0, no blending) ─────────────────────────────────
    br_label = f"A_BR{best_k:03d} vs B_{b_label}"
    print(f"\n  Running fixed BR baseline ({n_eval} episodes) …", end="", flush=True)
    res_br_fixed = _eval_policy_vs_partner(
        pol_br_a, pol_br_b, n_eval, seed + 2, device
    )
    print(f"  {res_br_fixed['mean']:.3f} ± {res_br_fixed['std']:.3f}")

    # ── Phase 2: convention-conditioned belief (β > 0) ────────────────────────
    print(f"\n  Phase 2 — {n_eval} episodes  "
          f"(A blends BR{best_k:03d}+BC{best_k} β={cb_beta}, B={b_label}) …\n")

    ac_params_a = pol_br_a.ac_params
    ac_params_b = pol_br_b.ac_params
    key         = jax.random.PRNGKey(seed + 3)
    scores_cb   = []
    partner_pred_confidences = []  # Collect partner prediction max probs
    partner_pred_entropies = []    # Collect partner prediction entropies
    window_eps  = []

    for game_idx in range(n_eval):
        key, subkey = jax.random.split(key)
        # Linear β warm-up: ramp from 0 to cb_beta over first cb_warmup games
        if cb_warmup > 0:
            beta_t = cb_beta * min(1.0, game_idx / cb_warmup)
        else:
            beta_t = cb_beta
        ep = _run_conv_belief_episode(
            subkey, ac_params_a, bc_params_k, ac_params_b, beta_t,
        )
        scores_cb.append(float(ep["score"]))
        # Collect partner prediction diagnostics
        partner_pred_confidences.extend(ep.get("partner_pred_max_probs", []))
        partner_pred_entropies.extend(ep.get("partner_pred_entropy", []))
        window_eps.append(ep)

        if (game_idx + 1) % 50 == 0:
            w = np.array([e["score"] for e in window_eps])
            beta_now = cb_beta * min(1.0, game_idx / cb_warmup) if cb_warmup > 0 else cb_beta
            print(f"  ── 50-game summary (games {game_idx-48}–{game_idx+1}) ──")
            print(f"  mean={w.mean():.3f} ± {w.std():.2f}  β_now={beta_now:.3f}")
            print_action_summary(window_eps, f"BR{best_k:03d}+BC{best_k}", b_label)
            window_eps = []

    # ── Results ───────────────────────────────────────────────────────────────
    cb_arr = np.array(scores_cb, dtype=np.float32)
    disp_w = max(1, n_eval // 10)

    print(f"\n{'='*65}")
    print(f"  {'Condition':<44}  {'Score':>7}  {'vs generic':>10}")
    print("  " + "-" * 64)

    def _row(label, mean, baseline=res_generic["mean"]):
        print(f"  {label:<44}  {mean:>7.3f}  {mean-baseline:>+10.3f}")

    _row("Generic (both)",                         res_generic["mean"])
    _row(f"{br_label} fixed (β=0)",               res_br_fixed["mean"])
    _row(f"A_BR{best_k:03d}+BC{best_k} β={cb_beta} vs B_{b_label} (all)",
         cb_arr.mean())
    _row(f"  first {disp_w} games",               cb_arr[:disp_w].mean())
    _row(f"  last  {disp_w} games",               cb_arr[-disp_w:].mean())

    sp_path = os.path.join(run1_dir, "agent_B_final.pkl")
    if os.path.exists(sp_path):
        pol_b1 = load_agent(sp_path, device=device)
        r_sp   = _eval_policy_vs_partner(pol_a_generic, pol_b1, n_eval, seed + 4, device)
        _row("Self-play ceiling (A_generic+B1)",   r_sp["mean"])

    gain_all  = cb_arr.mean()       - res_generic["mean"]
    gain_vs_br = cb_arr.mean()      - res_br_fixed["mean"]
    print(f"\n  Gain vs generic:   {gain_all:+.3f}")
    print(f"  Gain vs fixed BR:  {gain_vs_br:+.3f}")


    print(f"\n  Score trajectory ({disp_w}-game windows):")
    for i in range(0, n_eval, disp_w):
        chunk = cb_arr[i : i + disp_w]
        print(f"    games {i+1:>4}–{i+len(chunk):>4}:  mean={chunk.mean():.3f}")


# ---------------------------------------------------------------------------
# Adapter fine-tuning evaluation
# ---------------------------------------------------------------------------

def run_adapter_finetune_eval(run1_dir: str, run2_dir: str,
                              n_probe: int, n_eval: int,
                              seed: int, device,
                              adapter_dim: int = 8,
                              ft_lr: float = 1e-4,
                              ft_clip: float = 0.1,
                              symmetric: bool = False,
                              symmetric_ll: bool = False,
                              symmetric_ft: bool = False,
                              ft_diag: bool = False) -> None:
    """Probe → LL inference → BR selection → GRU adapter fine-tuning.

    Sequence:
      1. n_probe games with generic agents → compute per-profile LL.
      2. Select BR_k (A) and optionally BR_j (B) via LL argmax.
      3. Initialise a GRUAdapter (identity) for A and optionally B.
      4. For each eval game:
           a. Run episode with current adapter(s).
           b. Fine-tune adapter(s) via REINFORCE on the game outcome.

    The adapter learns to re-orient the frozen GRU representation toward the
    current partner's convention without touching any BR weights.

    symmetric (deprecated): if True, enables both symmetric_ll and symmetric_ft
    symmetric_ll: B also runs LL inference to pick BR_j
    symmetric_ft: B's adapter also fine-tunes after every game (requires symmetric_ll)
    """
    # Backward compatibility: if symmetric=True but neither ll/ft specified, enable both
    if symmetric and not symmetric_ll and not symmetric_ft:
        symmetric_ll = True
        symmetric_ft = True

    ll_tag = "LL✓" if symmetric_ll else "LL✗"
    ft_tag = "FT✓" if symmetric_ft else "FT✗"
    sym_tag = f"A+B [{ll_tag} {ft_tag}]" if symmetric_ll else "A-only"
    sep = "=" * 65
    print(sep)
    print(f"Probe → LL inference → GRU adapter fine-tuning  [{sym_tag}]")
    print(f"  Run 1: {run1_dir}   Run 2: {run2_dir}")
    print(f"  Probe: {n_probe}   Eval: {n_eval}   "
          f"adapter_dim={adapter_dim}   lr={ft_lr}   clip={ft_clip}")
    print(sep)

    # ── Load ──────────────────────────────────────────────────────────────────
    path_a1 = os.path.join(run1_dir, "agent_A_final.pkl")
    path_b2 = os.path.join(run2_dir, "agent_B_final.pkl")
    for p in (path_a1, path_b2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    pol_a_generic = load_agent(path_a1, device=device)
    pol_b_generic = load_agent(path_b2, device=device)
    meta_a        = load_metadata(path_a1)
    meta_b        = load_metadata(path_b2)
    pool_a        = meta_a.get("profile_pool", [])
    pool_b        = meta_b.get("profile_pool", []) if symmetric_ll else []
    brs_a         = _list_brs(run1_dir, "agent_A")
    brs_b         = _list_brs(run2_dir, "agent_B") if symmetric_ll else []

    print(f"\n  agent_A (run1): iter={meta_a['iteration']}  "
          f"profiles={len(pool_a)}  BRs={len(brs_a)}")
    if symmetric_ll:
        print(f"  agent_B (run2): iter={meta_b['iteration']}  "
              f"profiles={len(pool_b)}  BRs={len(brs_b)}")
    if not pool_a:
        print("\n  ERROR: profile_pool missing for A.")
        return

    # ── Generic baseline ──────────────────────────────────────────────────────
    print(f"\n  Running generic baseline ({n_eval} episodes) …", end="", flush=True)
    res_generic = _eval_policy_vs_partner(
        pol_a_generic, pol_b_generic, n_eval, seed, device
    )
    print(f"  {res_generic['mean']:.3f} ± {res_generic['std']:.3f}")

    # ── Phase 1: probe → LL inference ─────────────────────────────────────────
    print(f"\n  Phase 1 — running {n_probe} probe episodes …")
    probe_eps   = run_joint_rppo_episodes(
        pol_a_generic, pol_b_generic, n_probe,
        key=jax.random.PRNGKey(seed + 1), device=device,
    )
    probe_score = float(np.mean([ep["score"] for ep in probe_eps]))

    seqs_b       = _partner_obs_actions(probe_eps, pol_b_generic.player_id)
    best_k, ll_a = infer_profile(pool_a, seqs_b, available_ks={k for k, _ in brs_a})
    top3_a       = sorted((k for k, _ in brs_a), key=lambda k: -ll_a[k])[:3]
    print(f"  Probe score: {probe_score:.2f}  →  A selects BR_{best_k:03d}")
    print("  A top-3 LL: " + "  ".join(f"k={k}({ll_a[k]:.1f})" for k in top3_a))

    pol_br_a, best_k = _resolve_br(run1_dir, "agent_A", best_k, brs_a, device)
    if pol_br_a is None:
        print(f"  ERROR: no BR found for A k={best_k}")
        print(f"{run1_dir}, agent_A, {best_k}, {brs_a}, {device}")
        return

    if symmetric_ll and pool_b:
        seqs_a       = _partner_obs_actions(probe_eps, pol_a_generic.player_id)
        best_j, ll_b = infer_profile(pool_b, seqs_a, available_ks={k for k, _ in brs_b})
        top3_b       = sorted((j for j, _ in brs_b), key=lambda j: -ll_b[j])[:3]
        print(f"                              B selects BR_{best_j:03d}")
        print("  B top-3 LL: " + "  ".join(f"j={j}({ll_b[j]:.1f})" for j in top3_b))
        pol_br_b, best_j = _resolve_br(run2_dir, "agent_B", best_j, brs_b, device)
        if pol_br_b is None:
            print(f"  ERROR: no BR found for B j={best_j}")
            return
        b_label = f"BR{best_j:03d}"
    else:
        pol_br_b = pol_b_generic
        best_j   = None
        b_label  = "generic"

    # ── Fixed BR baseline (no adapter) ────────────────────────────────────────
    br_label = f"A_BR{best_k:03d} vs B_{b_label}"
    print(f"\n  Running {br_label} fixed baseline ({n_eval} episodes) …",
          end="", flush=True)
    res_br_fixed = _eval_policy_vs_partner(
        pol_br_a, pol_br_b, n_eval, seed + 2, device
    )
    print(f"  {res_br_fixed['mean']:.3f} ± {res_br_fixed['std']:.3f}")

    # ── Phase 2: adapter fine-tuning ──────────────────────────────────────────
    n_adap = GRUAdapter(adapter_dim=adapter_dim)
    n_adap_params = sum(
        x.size for x in jax.tree_util.tree_leaves(
            init_adapter(adapter_dim=adapter_dim)
        )
    )
    who = "A+B" if symmetric_ft else "A"
    print(f"\n  Phase 2 — {n_eval} episodes  "
          f"(GRU adapter, dim={adapter_dim}, {n_adap_params} params, {who} updates) …")
    print(f"  (first JAX grad call compiles; subsequent calls are fast)\n")

    adap_vars_a = init_adapter(adapter_dim=adapter_dim, seed=seed)
    adap_vars_b = init_adapter(adapter_dim=adapter_dim, seed=seed + 1) if symmetric_ft else None

    ac_params_a = pol_br_a.ac_params
    ac_params_b = pol_br_b.ac_params

    key        = jax.random.PRNGKey(seed + 3)
    scores_ft  = []
    window_eps = []

    for game_idx in range(n_eval):
        key, subkey = jax.random.split(key)

        ep = _run_adapter_episode(
            subkey, ac_params_a, adap_vars_a,
            ac_params_b, adap_vars_b, adapter_dim=adapter_dim,
        )
        scores_ft.append(float(ep["score"]))
        window_eps.append(ep)

        # Fine-tune A's adapter
        diag_this = ft_diag and game_idx < 5
        adap_vars_a = fine_tune_adapter(
            ac_params_a, adap_vars_a, ep, pol_br_a.player_id,
            lr=ft_lr, grad_clip=ft_clip, adapter_dim=adapter_dim, _diag=diag_this,
        )

        # Fine-tune B's adapter (symmetric_ft only)
        if symmetric_ft and adap_vars_b is not None:
            adap_vars_b = fine_tune_adapter(
                ac_params_b, adap_vars_b, ep, pol_br_b.player_id,
                lr=ft_lr, grad_clip=ft_clip, adapter_dim=adapter_dim,
            )

        if (game_idx + 1) % 50 == 0:
            w = np.array([e["score"] for e in window_eps])
            print(f"  ── 50-game summary (games {game_idx - 48}–{game_idx + 1}) ──")
            print(f"  mean={w.mean():.3f} ± {w.std():.2f}  "
                  f"lr={ft_lr}  adapter_dim={adapter_dim}")
            a_lbl = f"BR{best_k:03d}+adap"
            b_lbl = f"{b_label}+adap" if symmetric_ft else b_label
            print_action_summary(window_eps, a_lbl, b_lbl)
            window_eps = []

    # ── Results ───────────────────────────────────────────────────────────────
    ft_arr = np.array(scores_ft, dtype=np.float32)
    disp_w = max(1, n_eval // 10)

    print(f"\n{'='*65}")
    print(f"  {'Condition':<44}  {'Score':>7}  {'vs generic':>10}")
    print("  " + "-" * 64)

    def _row(label, mean, baseline=res_generic["mean"]):
        print(f"  {label:<44}  {mean:>7.3f}  {mean - baseline:>+10.3f}")

    b_ft_label = f"{b_label}+adap" if symmetric else b_label
    _row("Generic (both)",                          res_generic["mean"])
    _row(f"{br_label} fixed (no adapter)",          res_br_fixed["mean"])
    _row(f"A_BR{best_k:03d}+adap vs B_{b_ft_label} (all)",  ft_arr.mean())
    _row(f"  first {disp_w} games",                ft_arr[:disp_w].mean())
    _row(f"  last  {disp_w} games",                ft_arr[-disp_w:].mean())

    sp_path = os.path.join(run1_dir, "agent_B_final.pkl")
    if os.path.exists(sp_path):
        pol_b1 = load_agent(sp_path, device=device)
        r_sp   = _eval_policy_vs_partner(pol_a_generic, pol_b1, n_eval, seed + 4, device)
        _row("Self-play ceiling (A_generic+B1)",    r_sp["mean"])

    print(f"\n  Fine-tune trajectory ({disp_w}-game windows):")
    for i in range(0, n_eval, disp_w):
        chunk = ft_arr[i : i + disp_w]
        print(f"    games {i + 1:>4}–{i + len(chunk):>4}:  mean={chunk.mean():.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Log-likelihood partner inference evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run1",           type=str, required=True,
                   help="Run directory containing agent_A checkpoints + BR_k files")
    p.add_argument("--run2",           type=str, required=True,
                   help="Run directory containing agent_B_final.pkl (the new partner)")
    p.add_argument("--n_probe",        type=int, default=2,
                   help="Probe episodes for fixed-probe mode")
    p.add_argument("--n_eval",         type=int, default=500,
                   help="Evaluation episodes")
    p.add_argument("--probe_sweep",    action="store_true",
                   help="Sweep over n_probe in [1,2,4,8,16]")
    p.add_argument("--online",         action="store_true",
                   help="Online windowed LL inference (game-by-game update)")
    p.add_argument("--score_weighted", action="store_true",
                   help="Weight each game's LL contribution by its score (use with --online)")
    p.add_argument("--window",         type=int, default=50,
                   help="Sliding window size for windowed LL (use with --online)")
    p.add_argument("--print_games",    action="store_true",
                   help="Print move-by-move transcripts and 50-game action summaries")
    p.add_argument("--probe_finetune",  action="store_true",
                   help="Fixed probe → LL inference → BR + last-layer fine-tuning")
    p.add_argument("--symmetric",       action="store_true",
                   help="Alias for --symmetric_ll (backward-compatible)")
    p.add_argument("--symmetric_ll",    action="store_true",
                   help="Both agents run LL inference to pick their own BR (use with --probe_finetune)")
    p.add_argument("--symmetric_ft",    action="store_true",
                   help="Both agents fine-tune their actor head after each game (use with --probe_finetune)")
    p.add_argument("--symmetric_cb",    action="store_true",
                   help="Both agents blend their logits with their own BC profile (use with --probe_finetune and --cb_beta)")
    p.add_argument("--ft_lr",          type=float, default=1e-4,
                   help="Learning rate for last-layer fine-tuning (use with --probe_finetune)")
    p.add_argument("--ft_clip",        type=float, default=0.1,
                   help="Gradient clip norm for fine-tuning (use with --probe_finetune)")
    p.add_argument("--ft_alpha",       type=float, default=0.0,
                   help="Soft constraint: blend α toward original BR params after each update (0=off, 0.1=light)")
    p.add_argument("--conv_belief",    action="store_true",
                   help="Convention-conditioned belief: blend BR logits with BC-profile logits at test time")
    p.add_argument("--cb_beta",        type=float, default=0,
                   help="BC logit blend weight β for --conv_belief (0=pure BR, 1=pure BC, 0.3=recommended)")
    p.add_argument("--cb_warmup",      type=int, default=0,
                   help="Ramp β linearly from 0 to cb_beta over this many games (0=no warmup)")
    p.add_argument("--ft_diag",        action="store_true",
                   help="Print gradient norm and param-change norm for first 5 fine-tuning steps")
    p.add_argument("--adapter_ft",     action="store_true",
                   help="Probe → LL inference → BR + GRU adapter fine-tuning")
    p.add_argument("--adapter_dim",    type=int, default=8,
                   help="Bottleneck dimension of the GRU adapter (use with --adapter_ft)")
    p.add_argument("--seed",           type=int, default=0)
    args = p.parse_args()

    device = jax.devices()[0]
    print(f"JAX device: {device}")

    if args.adapter_ft:
        run_adapter_finetune_eval(args.run1, args.run2, args.n_probe, args.n_eval,
                                  args.seed, device,
                                  adapter_dim=args.adapter_dim,
                                  ft_lr=args.ft_lr, ft_clip=args.ft_clip,
                                  symmetric_ll=args.symmetric_ll or args.symmetric,
                                  symmetric_ft=args.symmetric_ft,
                                  ft_diag=args.ft_diag)
    elif args.conv_belief:
        run_conv_belief_eval(args.run1, args.run2, args.n_probe, args.n_eval,
                             args.seed, device,
                             cb_beta=args.cb_beta,
                             cb_warmup=args.cb_warmup,
                             symmetric=args.symmetric)
    elif args.probe_finetune:
        run_probe_finetune_eval(args.run1, args.run2, args.n_probe, args.n_eval,
                                args.seed, device,
                                ft_lr=args.ft_lr, ft_clip=args.ft_clip,
                                ft_alpha=args.ft_alpha,
                                symmetric_ll=args.symmetric_ll or args.symmetric,
                                symmetric_ft=args.symmetric_ft,
                                symmetric_cb=args.symmetric_cb,
                                cb_beta=args.cb_beta, cb_warmup=args.cb_warmup,
                                ft_diag=args.ft_diag)
    elif args.online:
        run_online_ll_eval(args.run1, args.run2, args.n_eval, args.seed, device,
                           score_weighted=args.score_weighted,
                           window=args.window,
                           print_games=args.print_games)
    elif args.probe_sweep:
        run_probe_sweep(args.run1, args.run2, args.n_eval, args.seed, device)
    else:
        run_ll_eval(args.run1, args.run2, args.n_probe, args.n_eval,
                    args.seed, device)
