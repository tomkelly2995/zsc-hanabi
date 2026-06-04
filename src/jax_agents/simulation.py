# src/jax_agents/simulation.py
# Pure-JAX episode simulation for Cooperative ODCFR.
#
# run_K_episodes    — vmapped over K random keys → batch of EpisodeData.
# collect_to_buffers — converts JAX arrays to numpy and populates buffers.
#                     Adv buffer stores raw scores (no IS correction).
#                     Non-sampled action targets are computed on-the-fly at
#                     training time from the current network (no staleness).
#
# Design notes
# ─────────────
# • lax.scan over MAX_EPISODE_STEPS — JIT-safe, vmappable.
# • "done" carry flag: once the game ends, subsequent scan iterations are
#   no-ops (carry returned unchanged), so padding cost is minimal.
# • Observations for both players are computed at each step (both use static
#   aidx=0 and aidx=1); the current player's obs is selected dynamically.
# • GRU.step uses the static Python loop inside GRUEncoder.__call__ — no
#   nn.scan, so Flax 0.10.4 + JAX 0.4.38 compatibility is maintained.
# • All shared params (adv_params, gru_params, bc_params) are closed over
#   in the vmapped lambda, not passed as batched vmap arguments.

from __future__ import annotations

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp

from src.jax_env.tiny_hanabi import (
    reset    as _env_reset,
    step     as _env_step,
    get_obs  as _env_get_obs,
    legal_mask as _env_legal_mask,
    NUM_MOVES as NUM_ACTIONS,
    OBS_SIZE  as OBS_DIM,
)
from src.jax_networks.advantage_net import AdvantageNet
from src.jax_networks.gru_encoder   import GRUEncoder, HIDDEN_DIM
from src.jax_networks.bc_partner_model import _BCNet
from src.jax_networks.policy_net    import PolicyNet
from flax import linen as nn

# Shared stateless module instances — all state lives in the params dicts.
_adv_net = AdvantageNet()
_gru_enc = GRUEncoder()
_bc_net  = _BCNet()
_pol_net = PolicyNet()


class _NextActionHead(nn.Module):
    """Linear head: HIDDEN_DIM → NUM_ACTIONS for auxiliary next-action prediction."""
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, h):
        return nn.Dense(self.num_actions)(h)


_aux_head = _NextActionHead()

MAX_EPISODE_STEPS  = 20
MAX_SELF_TURNS     = 10   # ≤ ceil(MAX_EPISODE_STEPS / 2)
MAX_PARTNER_TURNS  = 10

# ---------------------------------------------------------------------------
# Training constants — prevent extreme Q-values from IS weighting
# ---------------------------------------------------------------------------
Q_CLIP = 50.0     # clip Q-outputs to prevent explosion in MSE loss


# ---------------------------------------------------------------------------
# Regret matching
# ---------------------------------------------------------------------------

def regret_matching_jax(q_vals: jnp.ndarray, legal: jnp.ndarray) -> jnp.ndarray:
    """
    Pure-JAX regret matching.

    Baseline V = mean(Q_legal) — the uniform-strategy expected value.  This is
    the standard single-pass regret matching used in vanilla CFR and produces
    strategies that are proportional to positive advantages, which is healthy
    for exploration and gives moderate strategy entropy.

    A two-pass approach (using V(σ) instead of mean) was tested but consistently
    drove strategies to σ(a*)≈1 (entropy < 0.1 nats), making them too greedy:
    the corrected baseline equals Q(best) whenever σ₀ concentrates on one action,
    zeroing all other advantages and collapsing back to pass-1 result.  This
    also worsened IS-correction instability because near-zero σ values for
    non-dominant actions triggered the PI_A_MIN floor.

    Args:
        q_vals : (num_actions,) float32 — raw Q estimates from advantage net
        legal  : (num_actions,) bool    — True where action is legal

    Returns:
        (num_actions,) float32 — probability distribution (sums to 1)
    """
    n_legal  = jnp.maximum(1.0, legal.sum())
    legal_q  = jnp.where(legal, q_vals, 0.0)
    v_sigma  = legal_q.sum() / n_legal                # mean over legal actions

    # Advantage: max(0, Q_a - V) for legal actions; 0 for illegal.
    adv = jnp.where(legal, jnp.maximum(0.0, q_vals - v_sigma), 0.0)
    total = adv.sum()

    safe_total    = jnp.where(total > 0, total, 1.0)  # avoid 0-div
    sigma_adv     = adv / safe_total
    sigma_uniform = legal.astype(jnp.float32) / n_legal
    return jnp.where(total > 0, sigma_adv, sigma_uniform)


# ---------------------------------------------------------------------------
# Single episode
# ---------------------------------------------------------------------------

def _run_episode(key, adv_params, gru_params, bc_params, player_id: int,
                 partner_temp: float, explore_eps: float):
    """
    Simulate one complete episode.

    player_id is a static Python int so get_obs can use literal aidx values.

    Returns a dict with fixed-shape JAX arrays (all dims static for vmap).
    """
    partner_id = 1 - player_id  # Python int, resolved at trace time

    k_reset, k_steps = jax.random.split(key)
    step_keys = jax.random.split(k_steps, MAX_EPISODE_STEPS)  # (MAX_STEPS, 2)

    init_state = _env_reset(k_reset)

    init_carry = {
        "state":      init_state,
        "prev_state": init_state,
        "prev_act":   jnp.array(0, dtype=jnp.int32),
        "h_oppo":     jnp.zeros(HIDDEN_DIM, dtype=jnp.float32),
        "h_bc":       jnp.zeros(HIDDEN_DIM, dtype=jnp.float32),
        "done":       jnp.array(False),
        # Self-turn accumulation (pre-allocated, filled dynamically)
        "self_obs_h":          jnp.zeros((MAX_SELF_TURNS, OBS_DIM + HIDDEN_DIM), jnp.float32),
        "self_actions":        jnp.zeros(MAX_SELF_TURNS, jnp.int32),
        "self_sigmas":         jnp.zeros((MAX_SELF_TURNS, NUM_ACTIONS), jnp.float32),
        "n_self":              jnp.array(0, jnp.int32),
        # Partner-turn accumulation
        "partner_actions":     jnp.zeros(MAX_PARTNER_TURNS, jnp.int32),
        "partner_prefix_lens": jnp.zeros(MAX_SELF_TURNS, jnp.int32),
        "n_partner":           jnp.array(0, jnp.int32),
    }

    def episode_step(carry, step_key):

        def live(carry):
            state      = carry["state"]
            prev_state = carry["prev_state"]
            prev_act   = carry["prev_act"]

            # Observations for both players (static aidx — resolved at trace time).
            obs_0 = _env_get_obs(state, prev_state, prev_act, 0)  # (OBS_DIM,)
            obs_1 = _env_get_obs(state, prev_state, prev_act, 1)  # (OBS_DIM,)

            # Select self/partner obs using the static player_id.
            # (Python if — resolved at compile time, not a JAX conditional.)
            if player_id == 0:
                obs_self = obs_0
                obs_part = obs_1
            else:
                obs_self = obs_1
                obs_part = obs_0

            legal = _env_legal_mask(state)  # (8,) bool for cur_player_idx

            # cur_player_idx is (NA,) one-hot float; index by static player_id.
            is_self = state.cur_player_idx[player_id] > 0.5   # scalar bool

            # ---- Self turn ------------------------------------------------
            def self_fn(args):
                step_key, carry = args
                obs_h = jnp.concatenate([obs_self, carry["h_oppo"]])    # (148,)
                q     = _adv_net.apply(adv_params, obs_h)                # (8,)
                q = jnp.clip(q, -Q_CLIP, Q_CLIP)               # bound Q-values
                sigma = regret_matching_jax(q, legal)
                sigma = sigma / sigma.sum()  # safety normalise

                # ε-exploration: mix regret-matched sigma with uniform over legal
                # actions.  Ensures every action type (including hint_rank) gets
                # sampled, preventing the self-reinforcing Q-value collapse where
                # never-selected actions can never get real score targets.
                #
                # Pure sigma is stored in the buffer for correct policy distillation
                # and regret accounting.  Only the action selection uses the mixed
                # distribution — this is the standard ε-on-policy MCCFR approach.
                n_legal_f     = jnp.maximum(1.0, legal.astype(jnp.float32).sum())
                uniform       = legal.astype(jnp.float32) / n_legal_f
                sigma_explore = (1.0 - explore_eps) * sigma + explore_eps * uniform
                sigma_explore = sigma_explore / sigma_explore.sum()  # safety renorm
                action = jax.random.choice(step_key, NUM_ACTIONS, p=sigma_explore)

                i = carry["n_self"]
                new_carry = {
                    **carry,
                    "self_obs_h":   carry["self_obs_h"].at[i].set(obs_h),
                    "self_actions": carry["self_actions"].at[i].set(action),
                    "self_sigmas":  carry["self_sigmas"].at[i].set(sigma),   # pure σ
                    "partner_prefix_lens": carry["partner_prefix_lens"].at[i].set(
                        carry["n_partner"]
                    ),
                    "n_self": i + 1,
                }
                return action.astype(jnp.int32), new_carry

            # ---- Partner turn ---------------------------------------------
            def partner_fn(args):
                step_key, carry = args
                # Forward pass: obs + BC's own GRU state → action logits
                action_logits = _bc_net.apply(
                    {"params": bc_params}, obs_part, carry["h_bc"]
                )
                masked  = jnp.where(legal, action_logits / partner_temp,
                                    jnp.finfo(jnp.float32).min)
                probs   = jax.nn.softmax(masked, axis=-1)
                action  = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                # Update oracle's GRU (oracle tracks partner actions for obs_h)
                new_h_oppo = _gru_enc.step(gru_params, carry["h_oppo"], action)
                # Update BC's own GRU (BC tracks its own past actions)
                new_h_bc = _bc_net.apply(
                    {"params": bc_params},
                    carry["h_bc"],
                    action,
                    method=_bc_net.gru_step,
                )
                j = carry["n_partner"]
                new_carry = {
                    **carry,
                    "partner_actions": carry["partner_actions"].at[j].set(action),
                    "n_partner":       j + 1,
                    "h_oppo":          new_h_oppo,
                    "h_bc":            new_h_bc,
                }
                return action.astype(jnp.int32), new_carry

            action, new_carry = jax.lax.cond(
                is_self, self_fn, partner_fn, (step_key, carry)
            )

            new_state, _, new_done = _env_step(new_carry["state"], action)
            new_carry = {
                **new_carry,
                "state":      new_state,
                "prev_state": state,
                "prev_act":   action,
                "done":       new_done,
            }
            return new_carry

        # No-op when episode already finished.
        def skip(carry):
            return carry

        new_carry = jax.lax.cond(carry["done"], skip, live, carry)
        return new_carry, None

    final_carry, _ = jax.lax.scan(episode_step, init_carry, step_keys)

    return {
        "score":               final_carry["state"].score.astype(jnp.float32),
        "self_obs_h":          final_carry["self_obs_h"],
        "self_actions":        final_carry["self_actions"],
        "self_sigmas":         final_carry["self_sigmas"],
        "n_self":              final_carry["n_self"],
        "partner_actions":     final_carry["partner_actions"],
        "partner_prefix_lens": final_carry["partner_prefix_lens"],
        "n_partner":           final_carry["n_partner"],
    }


# ---------------------------------------------------------------------------
# K-episode batch
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnums=(4, 5, 6))
def run_K_episodes(
    keys,
    adv_params,
    gru_params,
    bc_params,
    player_id: int,
    partner_temp: float = 0.3,
    explore_eps: float = 0.06,
):
    """
    Simulate K episodes in parallel via vmap.

    Args:
        keys        : (K, 2) JAX key array — one key per episode
        adv_params  : Flax param dict for AdvantageNet
        gru_params  : Flax param dict for GRUEncoder
        bc_params   : Flax param dict for _BCNet (partner model)
        player_id   : 0 or 1 (static — determines self/partner obs selection)
        partner_temp: BC model softmax temperature (static)
        explore_eps : ε for ε-on-policy exploration (static). Mixes regret-matched
                      sigma with a uniform-over-legal distribution at self-turns.
                      Prevents action collapse (e.g. hint_rank never selected).
                      Set to 0.0 to disable. Default 0.06.

    Returns:
        dict of JAX arrays, each with leading dimension K.
    """
    return jax.vmap(
        lambda k: _run_episode(k, adv_params, gru_params, bc_params,
                               player_id, partner_temp, explore_eps)
    )(keys)


# ---------------------------------------------------------------------------
# Buffer population
# ---------------------------------------------------------------------------

def collect_to_buffers(
    results,
    adv_buffer,
    pol_buffer,
    iter_t: float,
) -> None:
    """
    Convert JAX episode results to numpy and populate adv/pol buffers.

    Advantage buffer format — 7-tuple per (episode, self-turn):
      (obs_raw, partner_acts, prefix_len, taken_action, raw_score, iter_t, sigma_a)

      • obs_raw      : (OBS_DIM,) float32 — raw observation (no GRU baked in)
      • partner_acts : (MAX_PARTNER_TURNS,) int32 — full episode partner sequence
      • prefix_len   : int — partner actions taken before this self-turn
      • taken_action : int — action index sampled from regret-matched σ
      • raw_score    : float — raw team score (IS correction applied at training time)
      • iter_t       : float — CFR iteration (used for linear weighting in training)
      • sigma_a      : float — σ(a*) at collection time, used for clipped IS correction
                       in _update_adv / _update_gru.  Clipped IS target =
                       min(raw_score / σ(a*), IS_CLIP) where IS_CLIP=10.  Storing
                       raw_score and σ(a*) separately keeps the buffer valid if
                       IS_CLIP is tuned without re-collecting data.

    Storing obs_raw (not obs_h) means the advantage net always trains on
    GRU encodings consistent with the CURRENT gru_params — the re-encoding
    happens inside _update_adv / _update_gru at training time.

    Targets for non-sampled actions are NOT pre-computed here.  Instead,
    _update_adv and _update_gru compute them on-the-fly from the current
    network using stop_gradient, so non-sampled Q predictions are always
    fresh (no staleness from historical GRU checkpoints).

    Policy buffer format — 5-tuple per (episode, self-turn):
      (obs_raw, partner_acts, prefix_len, sigma, iter_t)

      • iter_t : float — CFR iteration (used for linear recency weighting in
                 pol_net training, same as adv_net)

    Operates on CPU — all JAX arrays are materialised once via np.array().

    Args:
        results    : dict from run_K_episodes (JAX arrays, leading dim K)
        adv_buffer : ReservoirBuffer for advantage data
        pol_buffer : ReservoirBuffer for policy data
        iter_t     : CFR iteration index (stored with each adv and pol entry)
    """
    # Single device→host copy.
    scores       = np.array(results["score"],               dtype=np.float32)   # (K,)
    obs_hs       = np.array(results["self_obs_h"],          dtype=np.float32)   # (K, S, OBS_DIM+HIDDEN)
    sel_actions  = np.array(results["self_actions"],        dtype=np.int32)     # (K, S)
    sigmas       = np.array(results["self_sigmas"],         dtype=np.float32)   # (K, S, 8)
    n_selfs      = np.array(results["n_self"],              dtype=np.int32)     # (K,)
    partner_acts = np.array(results["partner_actions"],     dtype=np.int32)     # (K, P)
    prefix_lens  = np.array(results["partner_prefix_lens"], dtype=np.int32)     # (K, S)

    K = scores.shape[0]
    for k in range(K):
        n_s = int(n_selfs[k])
        for i in range(n_s):
            a_star         = int(sel_actions[k, i])
            weighted_score = float(scores[k])   # raw score — no IS correction

            sigma_a = float(sigmas[k, i, a_star])   # σ(a*) — for clipped IS at train time
            adv_buffer.add((
                obs_hs[k, i, :OBS_DIM],  # (OBS_DIM,) raw obs — no stale GRU
                partner_acts[k],          # (MAX_PARTNER_TURNS,) int32
                int(prefix_lens[k, i]),   # number of valid partner acts at this turn
                a_star,                   # int — taken action index
                weighted_score,           # float — raw team score
                float(iter_t),            # CFR iteration — for linear recency weighting
                sigma_a,                  # float — σ(a*) for clipped IS correction
            ))
            pol_buffer.add((
                obs_hs[k, i, :OBS_DIM],  # (OBS_DIM,) raw obs — no stale GRU
                partner_acts[k],          # (MAX_PARTNER_TURNS,) full episode seq
                int(prefix_lens[k, i]),   # partner actions before this self-turn
                sigmas[k, i],             # (8,) regret-matched strategy
                float(iter_t),            # CFR iteration — for linear recency weighting
            ))


# ---------------------------------------------------------------------------
# Lightweight probe episode — policy net vs BC partner, scores only
# ---------------------------------------------------------------------------

def _run_probe_episode(key, pol_params, gru_params, bc_params, player_id: int,
                       partner_temp: float):
    """
    Simulate one episode using the policy net for self turns and the BC model
    for partner turns.  Returns only the final team score.

    Used by XDOHanabiSolverJax._probe_profile_score() to generate an
    agent-specific score for each newly added BehaviourProfile, so that the
    two agents' meta-strategies can diverge independently.

    Unlike _run_episode (which uses the advantage net + regret matching and
    accumulates replay data), this function:
      • Uses pol_params (PolicyNet) for self turns — the averaged policy.
      • Needs no replay buffers — only the scalar score is returned.
      • Is therefore much cheaper per episode.
    """
    partner_id = 1 - player_id  # resolved at trace time

    k_reset, k_steps = jax.random.split(key)
    step_keys = jax.random.split(k_steps, MAX_EPISODE_STEPS)

    init_state = _env_reset(k_reset)
    init_carry = {
        "state":      init_state,
        "prev_state": init_state,
        "prev_act":   jnp.array(0, dtype=jnp.int32),
        "h_oppo":     jnp.zeros(HIDDEN_DIM, dtype=jnp.float32),
        "h_bc":       jnp.zeros(HIDDEN_DIM, dtype=jnp.float32),
        "done":       jnp.array(False),
    }

    def episode_step(carry, step_key):
        def live(carry):
            state      = carry["state"]
            prev_state = carry["prev_state"]
            prev_act   = carry["prev_act"]

            obs_0 = _env_get_obs(state, prev_state, prev_act, 0)
            obs_1 = _env_get_obs(state, prev_state, prev_act, 1)

            if player_id == 0:
                obs_self = obs_0
                obs_part = obs_1
            else:
                obs_self = obs_1
                obs_part = obs_0

            legal    = _env_legal_mask(state)
            is_self  = state.cur_player_idx[player_id] > 0.5

            # ---- Self turn: policy net --------------------------------------
            def self_fn(args):
                step_key, carry = args
                obs_h  = jnp.concatenate([obs_self, carry["h_oppo"]])
                logits = _pol_net.apply(pol_params, obs_h)
                masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
                probs  = jax.nn.softmax(masked)
                action = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                return action.astype(jnp.int32), carry

            # ---- Partner turn: BC model + GRU update -----------------------
            def partner_fn(args):
                step_key, carry = args
                action_logits = _bc_net.apply(
                    {"params": bc_params}, obs_part, carry["h_bc"]
                )
                masked  = jnp.where(legal, action_logits / partner_temp,
                                    jnp.finfo(jnp.float32).min)
                probs   = jax.nn.softmax(masked)
                action  = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                new_h_oppo = _gru_enc.step(gru_params, carry["h_oppo"], action)
                new_h_bc   = _bc_net.apply(
                    {"params": bc_params},
                    carry["h_bc"],
                    action,
                    method=_bc_net.gru_step,
                )
                return action.astype(jnp.int32), {
                    **carry,
                    "h_oppo": new_h_oppo,
                    "h_bc":   new_h_bc,
                }

            action, new_carry = jax.lax.cond(
                is_self, self_fn, partner_fn, (step_key, carry)
            )

            new_state, _, new_done = _env_step(new_carry["state"], action)
            new_carry = {
                **new_carry,
                "state":      new_state,
                "prev_state": state,
                "prev_act":   action,
                "done":       new_done,
            }
            return new_carry

        def skip(carry):
            return carry

        new_carry = jax.lax.cond(carry["done"], skip, live, carry)
        return new_carry, None

    final_carry, _ = jax.lax.scan(episode_step, init_carry, step_keys)
    return final_carry["state"].score.astype(jnp.float32)


@partial(jax.jit, static_argnums=(4, 5))
def run_K_probe_episodes(
    keys,
    pol_params,
    gru_params,
    bc_params,
    player_id: int,
    partner_temp: float = 0.3,
):
    """
    Run K probe episodes in parallel via vmap.

    Args:
        keys        : (K, 2) JAX key array
        pol_params  : Flax param dict for PolicyNet (self turns)
        gru_params  : Flax param dict for GRUEncoder (h_oppo tracking)
        bc_params   : Flax param dict for _BCNet (partner turns)
        player_id   : 0 or 1 (static)
        partner_temp: BC softmax temperature (static)

    Returns:
        (K,) float32 — team score for each probe episode
    """
    return jax.vmap(
        lambda k: _run_probe_episode(k, pol_params, gru_params, bc_params,
                                     player_id, partner_temp)
    )(keys)


# ---------------------------------------------------------------------------
# Vectorised joint episode collection (policy_a vs policy_b)
# ---------------------------------------------------------------------------

def _run_joint_episode(key, pol_params_a, gru_params_a, pol_params_b, gru_params_b):
    """
    Simulate one episode with policy_a (player 0) vs policy_b (player 1).

    Both agents track the partner via their own GRU encoder, updated online
    after each partner action — identical to the Python run_joint_episodes
    but fully JAX-traced for vmap.

    All params are closed over (not batched), so vmap maps only over keys.

    Returns a dict of fixed-shape arrays (no Python control flow at runtime).
    """
    k_reset, k_steps = jax.random.split(key)
    step_keys = jax.random.split(k_steps, MAX_EPISODE_STEPS)

    init_state = _env_reset(k_reset)

    init_carry = {
        "state":       init_state,
        "prev_state":  init_state,
        "prev_act":    jnp.array(0, jnp.int32),
        "h_a":         jnp.zeros(HIDDEN_DIM, jnp.float32),  # A's GRU tracking B
        "h_b":         jnp.zeros(HIDDEN_DIM, jnp.float32),  # B's GRU tracking A
        "done":        jnp.array(False),
        "obs0":        jnp.zeros((MAX_EPISODE_STEPS, OBS_DIM), jnp.float32),
        "obs1":        jnp.zeros((MAX_EPISODE_STEPS, OBS_DIM), jnp.float32),
        "cur_players": jnp.zeros(MAX_EPISODE_STEPS, jnp.int32),
        "actions":     jnp.zeros(MAX_EPISODE_STEPS, jnp.int32),
        "n_steps":     jnp.array(0, jnp.int32),
    }

    def episode_step(carry, step_key):

        def live(carry):
            state      = carry["state"]
            prev_state = carry["prev_state"]
            prev_act   = carry["prev_act"]

            obs_0 = _env_get_obs(state, prev_state, prev_act, 0)
            obs_1 = _env_get_obs(state, prev_state, prev_act, 1)
            legal = _env_legal_mask(state)

            is_a = state.cur_player_idx[0] > 0.5   # scalar bool

            # ---- Player 0 (agent A) turn ------------------------------------
            def a_turn(args):
                step_key, carry = args
                obs_h  = jnp.concatenate([obs_0, carry["h_a"]])
                logits = _pol_net.apply(pol_params_a, obs_h)
                masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
                probs  = jax.nn.softmax(masked)
                action = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                # B observes A's action → update h_b using B's GRU
                new_h_b = _gru_enc.step(gru_params_b, carry["h_b"], action)
                return action.astype(jnp.int32), {**carry, "h_b": new_h_b}

            # ---- Player 1 (agent B) turn ------------------------------------
            def b_turn(args):
                step_key, carry = args
                obs_h  = jnp.concatenate([obs_1, carry["h_b"]])
                logits = _pol_net.apply(pol_params_b, obs_h)
                masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
                probs  = jax.nn.softmax(masked)
                action = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                # A observes B's action → update h_a using A's GRU
                new_h_a = _gru_enc.step(gru_params_a, carry["h_a"], action)
                return action.astype(jnp.int32), {**carry, "h_a": new_h_a}

            action, new_carry = jax.lax.cond(
                is_a, a_turn, b_turn, (step_key, carry)
            )

            new_state, _, new_done = _env_step(new_carry["state"], action)
            i = carry["n_steps"]
            cur_player = jnp.argmax(state.cur_player_idx).astype(jnp.int32)
            new_carry = {
                **new_carry,
                "state":       new_state,
                "prev_state":  state,
                "prev_act":    action,
                "done":        new_done,
                "obs0":        carry["obs0"].at[i].set(obs_0),
                "obs1":        carry["obs1"].at[i].set(obs_1),
                "cur_players": carry["cur_players"].at[i].set(cur_player),
                "actions":     carry["actions"].at[i].set(action),
                "n_steps":     i + 1,
            }
            return new_carry

        def skip(carry):
            return carry

        new_carry = jax.lax.cond(carry["done"], skip, live, carry)
        return new_carry, None

    final_carry, _ = jax.lax.scan(episode_step, init_carry, step_keys)

    return {
        "score":       final_carry["state"].score.astype(jnp.float32),
        "obs0":        final_carry["obs0"],         # (MAX_EPISODE_STEPS, OBS_DIM)
        "obs1":        final_carry["obs1"],
        "cur_players": final_carry["cur_players"],  # (MAX_EPISODE_STEPS,)
        "actions":     final_carry["actions"],
        "n_steps":     final_carry["n_steps"],
    }


@jax.jit
def run_N_joint_episodes(keys, pol_params_a, gru_params_a, pol_params_b, gru_params_b):
    """
    Simulate N joint episodes in parallel via vmap.

    Args:
        keys        : (N, 2) JAX key array — one per episode
        pol_params_a: Flax param dict for agent A's PolicyNet
        gru_params_a: Flax param dict for agent A's GRUEncoder
        pol_params_b: Flax param dict for agent B's PolicyNet
        gru_params_b: Flax param dict for agent B's GRUEncoder

    Returns:
        dict of JAX arrays each with leading dimension N.
    """
    return jax.vmap(
        lambda k: _run_joint_episode(k, pol_params_a, gru_params_a,
                                        pol_params_b, gru_params_b)
    )(keys)


def joint_results_to_episode_dicts(results) -> list:
    """
    Convert vmapped joint episode results to the episode-dict format expected
    by XDOHanabiSolverJax.extract_behaviour_profile().

    Single device→host copy, then pure numpy/Python — no JAX dispatch.

    Args:
        results : dict output of run_N_joint_episodes (leading dim N)

    Returns:
        list of {"steps": [...], "score": float} dicts
    """
    scores      = np.array(results["score"],       dtype=np.float32)  # (N,)
    all_obs0    = np.array(results["obs0"],         dtype=np.float32)  # (N, T, OBS_DIM)
    all_obs1    = np.array(results["obs1"],         dtype=np.float32)
    all_cur     = np.array(results["cur_players"],  dtype=np.int32)    # (N, T)
    all_acts    = np.array(results["actions"],      dtype=np.int32)    # (N, T)
    all_n       = np.array(results["n_steps"],      dtype=np.int32)    # (N,)

    episodes = []
    for i in range(len(scores)):
        n = int(all_n[i])
        steps = [
            {
                "current_player": int(all_cur[i, t]),
                "player_obs":     {0: all_obs0[i, t], 1: all_obs1[i, t]},
                "action":         int(all_acts[i, t]),
            }
            for t in range(n)
        ]
        episodes.append({"steps": steps, "score": float(scores[i])})
    return episodes
