# src/jax_agents/rppo_agent_jax.py
# Recurrent PPO inner-loop solver for cooperative Tiny Hanabi.
#
# Replaces CooperativeODCFRAgentJax as the oracle.
# The RPPOActorCritic carries its own GRU hidden state across self-turns,
# so no separate h_oppo encoder is needed.
#
# Training interface
# ──────────────────
#   agent.collect_episodes(profiles, meta_strategy, n_episodes, key)
#       → list of trajectory dicts (one per episode)
#   agent.update(trajectories)
#       → loss dict; updates self.params in-place
#
# Episode simulation
# ──────────────────
#   _run_rppo_episode  — single episode via lax.scan (fully JAX-traceable)
#   run_K_rppo_episodes — K episodes in parallel via vmap + JIT
#   run_K_rppo_probe_episodes — score-only probe variant
#
#   These mirror run_K_episodes / run_K_probe_episodes in simulation.py.
#   No Python loops in the hot path: episode collection is ~10-50× faster
#   than the Python-loop version.
#
# Shaped reward
# ─────────────
#   r_t = Σ score_increments between self-turn t and self-turn t+1.
#   Includes both the agent's own card plays and the partner's.
#   This is the cooperative credit-assignment signal: the agent receives
#   the full team score improvement that happens "on its watch."
#
# Profile interface
# ─────────────────
#   Identical to CooperativeODCFRAgentJax: accepts a list of
#   BehaviourProfileJax objects and a meta_strategy weight vector.
#   Rollouts are split proportionally across profiles (largest-remainder).

from __future__ import annotations

from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.jax_networks.rppo_actor_critic import (
    RPPOActorCritic, OBS_DIM, HIDDEN_DIM, NUM_ACTIONS,
)
from src.jax_env.tiny_hanabi import (
    reset      as _env_reset,
    step       as _env_step,
    get_obs    as _env_get_obs,
    legal_mask as _env_legal_mask,
)
from src.jax_agents.simulation import _bc_net   # shared stateless _BCNet instance

_ac_net = RPPOActorCritic()

_PLAYER_IDS = {"agent_A": 0, "agent_B": 1}

# Episode step bounds — match simulation.py so vmap shapes are consistent.
MAX_EPISODE_STEPS = 20
MAX_SELF_TURNS    = 10   # ≤ ceil(MAX_EPISODE_STEPS / 2)


# ---------------------------------------------------------------------------
# Module-level JIT helpers (used by diagnose_rppo.py and debug code)
# ---------------------------------------------------------------------------

@jax.jit
def _bc_forward(bc_params, obs, h_bc):
    """BC model action logits (pre-temperature, pre-mask)."""
    return _bc_net.apply({"params": bc_params}, obs, h_bc)


@jax.jit
def _bc_gru_step(bc_params, h_bc, action):
    """Advance BC model's own GRU carry by one action token."""
    return _bc_net.apply(
        {"params": bc_params}, h_bc, action, method=_bc_net.gru_step
    )


@jax.jit
def _ac_forward(params, obs, h, legal):
    """Actor-critic forward: returns (masked_logits, value, new_h)."""
    logits, value, new_h = _ac_net.apply(params, obs, h)
    masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
    return masked, value, new_h


# ---------------------------------------------------------------------------
# Rollout allocation
# ---------------------------------------------------------------------------

def _split_K(weights: np.ndarray, K_total: int) -> np.ndarray:
    """Distribute K_total rollouts across profiles proportional to weights."""
    raw      = weights * K_total
    K_ks     = np.floor(raw).astype(int)
    fracs    = raw - K_ks
    for idx in np.argsort(-fracs)[:int(K_total - K_ks.sum())]:
        K_ks[idx] += 1
    return K_ks


# ---------------------------------------------------------------------------
# GAE computation (numpy, runs on CPU)
# ---------------------------------------------------------------------------

def _compute_gae(
    rewards:    np.ndarray,
    values:     np.ndarray,
    dones:      np.ndarray,
    last_value: float,
    gamma:      float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation over a single episode's self-turns.

    Args:
        rewards    : (T,) shaped reward per self-turn
        values     : (T,) critic estimate per self-turn
        dones      : (T,) 1.0 if this self-turn is terminal, else 0.0
        last_value : bootstrap value after the last self-turn (0 if terminal)
        gamma      : discount factor
        gae_lambda : GAE λ

    Returns:
        advantages : (T,) GAE advantages
        returns    : (T,) targets for the value function (advantages + values)
    """
    T   = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val  = last_value if t == T - 1 else float(values[t + 1])
        next_done = float(dones[t])
        delta = rewards[t] + gamma * next_val * (1.0 - next_done) - values[t]
        gae   = delta + gamma * gae_lambda * (1.0 - next_done) * gae
        adv[t] = gae
    return adv, adv + values.astype(np.float32)


# ---------------------------------------------------------------------------
# JAX-native episode simulation  (lax.scan + vmap)
# ---------------------------------------------------------------------------

def _run_rppo_episode(
    key,
    ac_params,
    bc_params,
    player_id:   int,
    partner_temp: float,
):
    """
    Simulate one RPPO episode using lax.scan — fully JAX-traceable and vmappable.

    The RPPO actor-critic GRU (h_ac) is updated on every SELF-turn, exactly as
    in the Python-loop version.  The BC partner model GRU (h_bc) is updated on
    every PARTNER-turn.  Shaped rewards (score deltas) are accumulated into the
    last self-turn's reward slot in the carry.

    player_id and partner_temp are Python-static values resolved at trace time.

    Returns a dict of fixed-shape JAX arrays (all dims static for vmap):
        score         ()             float32
        out_of_lives  ()             bool
        n_self        ()             int32   — number of valid self-turns
        n_steps       ()             int32   — total episode length
        self_obs      (S, OBS_DIM)   float32 — observation at each self-turn
        self_h_prev   (S, HIDDEN)    float32 — GRU carry before each self-turn
        self_h_after  (S, HIDDEN)    float32 — GRU carry after each self-turn (for partner aux loss)
        self_legal    (S, NUM_ACT)   bool    — legal mask at each self-turn
        self_actions  (S,)           int32
        self_log_probs(S,)           float32
        self_values   (S,)           float32
        self_rewards  (S,)           float32 — shaped (cooperative credit)
        self_dones    (S,)           float32 — 1.0 at terminal self-turn
        partner_actions (S,)         int32   — partner's action after each self-turn
    where S = MAX_SELF_TURNS (padding beyond n_self is zero).
    """
    k_reset, k_steps = jax.random.split(key)
    step_keys = jax.random.split(k_steps, MAX_EPISODE_STEPS)

    init_state = _env_reset(k_reset)

    init_carry = {
        "state":          init_state,
        "prev_state":     init_state,
        "prev_act":       jnp.array(0, jnp.int32),
        "h_ac":           jnp.zeros(HIDDEN_DIM, jnp.float32),
        "h_bc":           jnp.zeros(HIDDEN_DIM, jnp.float32),
        "done":           jnp.array(False),
        "prev_score":     jnp.array(0.0, jnp.float32),
        # Trajectory arrays (pre-allocated, indexed as self-turns occur)
        "self_obs":       jnp.zeros((MAX_SELF_TURNS, OBS_DIM),     jnp.float32),
        "self_h_prev":    jnp.zeros((MAX_SELF_TURNS, HIDDEN_DIM),  jnp.float32),
        "self_h_after":   jnp.zeros((MAX_SELF_TURNS, HIDDEN_DIM),  jnp.float32),
        "self_legal":     jnp.zeros((MAX_SELF_TURNS, NUM_ACTIONS), jnp.bool_),
        "self_actions":   jnp.zeros(MAX_SELF_TURNS,                jnp.int32),
        "self_log_probs": jnp.zeros(MAX_SELF_TURNS,                jnp.float32),
        "self_values":    jnp.zeros(MAX_SELF_TURNS,                jnp.float32),
        "self_rewards":   jnp.zeros(MAX_SELF_TURNS,                jnp.float32),
        "self_dones":     jnp.zeros(MAX_SELF_TURNS,                jnp.float32),
        "partner_actions":jnp.zeros(MAX_SELF_TURNS,                jnp.int32),
        "n_self":         jnp.array(0,                             jnp.int32),
        "n_steps":        jnp.array(0,                             jnp.int32),
    }

    def episode_step(carry, step_key):

        def live(carry):
            state      = carry["state"]
            prev_state = carry["prev_state"]
            prev_act   = carry["prev_act"]

            obs_0 = _env_get_obs(state, prev_state, prev_act, 0)
            obs_1 = _env_get_obs(state, prev_state, prev_act, 1)
            legal = _env_legal_mask(state)

            # Python if — resolved at trace time, not a JAX conditional.
            if player_id == 0:
                obs_self = obs_0
                obs_part = obs_1
            else:
                obs_self = obs_1
                obs_part = obs_0

            is_self = state.cur_player_idx[player_id] > 0.5

            # ── Self turn ────────────────────────────────────────────────────
            def self_fn(args):
                step_key, carry = args
                h_prev = carry["h_ac"]
                logits, value, new_h, _ = _ac_net.apply(ac_params, obs_self, h_prev)
                masked   = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
                probs    = jax.nn.softmax(masked)
                action   = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                log_prob = jax.nn.log_softmax(masked)[action]

                i = carry["n_self"]
                new_carry = {
                    **carry,
                    "h_ac":           new_h,
                    "self_obs":       carry["self_obs"].at[i].set(obs_self),
                    "self_h_prev":    carry["self_h_prev"].at[i].set(h_prev),
                    "self_h_after":   carry["self_h_after"].at[i].set(new_h),
                    "self_legal":     carry["self_legal"].at[i].set(legal),
                    "self_actions":   carry["self_actions"].at[i].set(action),
                    "self_log_probs": carry["self_log_probs"].at[i].set(log_prob),
                    "self_values":    carry["self_values"].at[i].set(value),
                    "n_self":         i + 1,
                }
                return action.astype(jnp.int32), new_carry

            # ── Partner turn ─────────────────────────────────────────────────
            def partner_fn(args):
                step_key, carry = args
                bc_logits = _bc_net.apply(
                    {"params": bc_params}, obs_part, carry["h_bc"]
                )
                masked = jnp.where(
                    legal, bc_logits / partner_temp, jnp.finfo(jnp.float32).min
                )
                probs    = jax.nn.softmax(masked)
                action   = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                new_h_bc = _bc_net.apply(
                    {"params": bc_params}, carry["h_bc"], action,
                    method=_bc_net.gru_step,
                )
                # Store partner action indexed by the last self-turn
                last_idx = jnp.maximum(0, carry["n_self"] - 1)
                new_partner_actions = carry["partner_actions"].at[last_idx].set(action.astype(jnp.int32))
                return action.astype(jnp.int32), {
                    **carry,
                    "h_bc": new_h_bc,
                    "partner_actions": new_partner_actions,
                }

            action, new_carry = jax.lax.cond(
                is_self, self_fn, partner_fn, (step_key, carry)
            )

            new_state, _, new_done = _env_step(new_carry["state"], action)

            # Shaped reward: attribute score delta to the last self-turn.
            # Works for both self-turn deltas (current entry) and partner-turn
            # deltas (previous entry) because last_idx = max(0, n_self - 1)
            # always points at the most recently recorded self-turn.
            delta    = new_state.score.astype(jnp.float32) - carry["prev_score"]
            last_idx = jnp.maximum(0, new_carry["n_self"] - 1)
            new_rewards = new_carry["self_rewards"].at[last_idx].add(
                jnp.where(new_carry["n_self"] > 0, delta, 0.0)
            )

            # Done flag: mark last self-turn terminal when episode ends.
            new_dones = new_carry["self_dones"].at[last_idx].add(
                jnp.where(new_done & (new_carry["n_self"] > 0), 1.0, 0.0)
            )

            return {
                **new_carry,
                "state":        new_state,
                "prev_state":   state,
                "prev_act":     action,
                "done":         new_done,
                "prev_score":   new_state.score.astype(jnp.float32),
                "self_rewards": new_rewards,
                "self_dones":   new_dones,
                "n_steps":      carry["n_steps"] + 1,
            }

        # No-op when episode already finished.
        def skip(carry):
            return carry

        new_carry = jax.lax.cond(carry["done"], skip, live, carry)
        return new_carry, None

    final_carry, _ = jax.lax.scan(episode_step, init_carry, step_keys)

    return {
        "score":          final_carry["state"].score.astype(jnp.float32),
        "out_of_lives":   final_carry["state"].out_of_lives,
        "n_self":         final_carry["n_self"],
        "n_steps":        final_carry["n_steps"],
        "self_obs":       final_carry["self_obs"],
        "self_h_prev":    final_carry["self_h_prev"],
        "self_h_after":   final_carry["self_h_after"],
        "self_legal":     final_carry["self_legal"],
        "self_actions":   final_carry["self_actions"],
        "self_log_probs": final_carry["self_log_probs"],
        "self_values":    final_carry["self_values"],
        "self_rewards":   final_carry["self_rewards"],
        "self_dones":     final_carry["self_dones"],
        "partner_actions": final_carry["partner_actions"],
    }


@partial(jax.jit, static_argnums=(3, 4))
def run_K_rppo_episodes(
    keys,
    ac_params,
    bc_params,
    player_id:    int,
    partner_temp: float,
):
    """
    Simulate K RPPO episodes in parallel via vmap.

    Args:
        keys        : (K, 2) JAX key array — one key per episode
        ac_params   : RPPOActorCritic Flax variable dict
        bc_params   : _BCNet Flax param dict (partner model)
        player_id   : 0 or 1  (static — selects self/partner obs at trace time)
        partner_temp: BC softmax temperature (static)

    Returns:
        dict of JAX arrays with leading dimension K.
    """
    return jax.vmap(
        lambda k: _run_rppo_episode(k, ac_params, bc_params, player_id, partner_temp)
    )(keys)


@partial(jax.jit, static_argnums=(3, 4))
def run_K_rppo_probe_episodes(
    keys,
    ac_params,
    bc_params,
    player_id:    int,
    partner_temp: float,
):
    """
    Vmapped probe: run K episodes and return only the team scores (K,).

    Cheaper than run_K_rppo_episodes when only the mean score is needed
    (e.g. eval probes in diagnose_rppo.py and XDO meta-game scoring).
    """
    return jax.vmap(
        lambda k: _run_rppo_episode(
            k, ac_params, bc_params, player_id, partner_temp
        )["score"]
    )(keys)


def _results_to_trajs(results) -> list[dict]:
    """
    Convert run_K_rppo_episodes output to a list of trajectory dicts
    compatible with RPPOAgentJax.update().

    Single device→host copy (np.array calls), then pure numpy slicing.
    """
    scores          = np.array(results["score"],          dtype=np.float32)
    self_obs        = np.array(results["self_obs"],       dtype=np.float32)
    self_h_prev     = np.array(results["self_h_prev"],    dtype=np.float32)
    self_legal      = np.array(results["self_legal"],     dtype=bool)
    self_actions    = np.array(results["self_actions"],   dtype=np.int32)
    self_lps        = np.array(results["self_log_probs"], dtype=np.float32)
    self_values     = np.array(results["self_values"],    dtype=np.float32)
    self_rewards    = np.array(results["self_rewards"],   dtype=np.float32)
    self_dones      = np.array(results["self_dones"],     dtype=np.float32)
    partner_actions = np.array(results.get("partner_actions", np.zeros_like(self_actions)), dtype=np.int32)
    n_selfs         = np.array(results["n_self"],         dtype=np.int32)

    trajs = []
    for k in range(len(scores)):
        T = int(n_selfs[k])
        if T == 0:
            trajs.append(
                {kk: np.zeros((0,)) for kk in (
                    "obs", "h_prev", "legal", "actions",
                    "log_probs", "values", "rewards", "dones", "partner_actions",
                )} | {"score": float(scores[k])}
            )
            continue
        trajs.append({
            "obs":              self_obs[k, :T],
            "h_prev":           self_h_prev[k, :T],
            "legal":            self_legal[k, :T],
            "actions":          self_actions[k, :T],
            "log_probs":        self_lps[k, :T],
            "values":           self_values[k, :T],
            "rewards":          self_rewards[k, :T],
            "dones":            self_dones[k, :T],
            "partner_actions":  partner_actions[k, :T],
            "score":            float(scores[k]),
        })
    return trajs


# ---------------------------------------------------------------------------
# JIT-compiled PPO gradient step
# ---------------------------------------------------------------------------

def _make_ppo_step(
    optimizer:  optax.GradientTransformation,
    clip_eps:   float,
    ent_coef:   float,
    vf_coef:    float,
    aux_loss_weight: float = 0.0,
):
    """
    Build a JIT-compiled PPO minibatch update function.

    Each sample in the minibatch is processed independently using the stored
    h_prev (hidden state at the time the sample was collected).  This avoids
    full BPTT through the episode while still giving the GRU's parameters
    a gradient signal (one step of unrolling per sample).

    The PPO clip ratio limits how much the policy can change per update,
    keeping the stored h_prev a valid approximation.

    Optional auxiliary loss: if aux_loss_weight > 0, adds cross-entropy loss
    for partner action prediction to encourage GRU convention encoding.
    """
    @jax.jit
    def step(
        params, opt_state,
        obs,        # (B, 84)
        h_prev,     # (B, 64)
        legal,      # (B, 8)  bool
        actions,    # (B,)    int32
        old_lps,    # (B,)    float32 — log probs from rollout
        advantages, # (B,)    float32
        returns,    # (B,)    float32
        partner_actions=None,  # (B,) — partner's action for aux loss
    ):
        def loss_fn(p):
            def fwd(ob, h, lg):
                logits, value, _, partner_logits = _ac_net.apply(p, ob, h)
                masked = jnp.where(lg, logits, jnp.finfo(jnp.float32).min)
                return masked, value, partner_logits

            masked_b, values_b, partner_logits_b = jax.vmap(fwd)(obs, h_prev, legal)

            log_probs = jax.nn.log_softmax(masked_b, axis=-1)          # (B, 8)
            B         = obs.shape[0]
            new_lps   = log_probs[jnp.arange(B), actions]              # (B,)

            # Clipped surrogate objective.
            # advantages are already batch-normalised before this call.
            ratio = jnp.exp(new_lps - old_lps)
            pg1   = -ratio * advantages
            pg2   = -jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = jnp.mean(jnp.maximum(pg1, pg2))

            # Value function loss
            value_loss = jnp.mean((values_b - returns) ** 2)

            # Entropy bonus — encourages exploration
            probs   = jax.nn.softmax(masked_b, axis=-1)
            entropy = -jnp.mean(jnp.sum(probs * log_probs, axis=-1))

            # Auxiliary loss: partner action prediction
            aux_loss = 0.0
            if aux_loss_weight > 0.0 and partner_actions is not None:
                partner_log_probs = jax.nn.log_softmax(partner_logits_b, axis=-1)
                aux_loss = -jnp.mean(partner_log_probs[jnp.arange(B), partner_actions])

            total = actor_loss + vf_coef * value_loss - ent_coef * entropy + aux_loss_weight * aux_loss
            return total, (actor_loss, value_loss, entropy, aux_loss, ratio)

        (_, (al, vl, ent, aux_l, ratio)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        clip_frac  = jnp.mean(
            (jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32)
        )
        return new_params, new_opt_state, al, vl, ent, clip_frac

    return step


# ---------------------------------------------------------------------------
# JAX-native GAE + fused update (GAE + all PPO epochs in one GPU dispatch)
# ---------------------------------------------------------------------------

def _gae_single_jax(
    rewards:    jax.Array,   # (S,)
    values:     jax.Array,   # (S,)
    dones:      jax.Array,   # (S,)
    gamma:      float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Reverse-scan GAE for one episode's padded (S,) arrays.

    Padding entries (beyond n_self) have reward=0, value=0, done=0.  The
    done=1.0 at the final valid self-turn terminates the backward pass before
    the padded region, so padding contributes zero advantage.
    """
    next_vals = jnp.concatenate([values[1:], jnp.zeros(1)])
    deltas    = rewards + gamma * next_vals * (1.0 - dones) - values

    # Compute gae[t] = delta[t] + gamma*λ*(1-done[t]) * gae[t+1]
    # via a forward scan over the time-reversed sequence.
    flip_d    = jnp.flip(deltas)
    flip_coef = jnp.flip(gamma * gae_lambda * (1.0 - dones))

    def _rev_step(gae_next, x):
        delta, coef = x
        gae = delta + coef * gae_next
        return gae, gae

    _, flip_adv = jax.lax.scan(_rev_step, jnp.array(0.0), (flip_d, flip_coef))
    advantages  = jnp.flip(flip_adv)
    return advantages, advantages + values


def _make_fused_update(
    optimizer:      optax.GradientTransformation,
    clip_eps:       float,
    ent_coef:       float,
    vf_coef:        float,
    ppo_epochs:     int,
    minibatch_size: int,
    gamma:          float,
    gae_lambda:     float,
    K:              int,   # episodes per batch — compile-time constant
):
    """
    Build a JIT-compiled function that fuses GAE + all PPO epochs into one
    GPU dispatch.  Eliminates Python GAE loops and Python minibatch loops.

    Call signature:
        fused_update(params, opt_state, key,
                     self_obs, self_h_prev, self_legal, self_actions,
                     self_log_probs, self_values, self_rewards, self_dones,
                     n_self)
        → (new_params, new_opt_state, actor_loss, value_loss, entropy, clip_frac)

    All array inputs must already be on the target device (shape (K, S, ...)).
    """
    ppo_step_fn = _make_ppo_step(optimizer, clip_eps, ent_coef, vf_coef)
    N    = K * MAX_SELF_TURNS
    n_mb = N // minibatch_size   # complete minibatches; partial last batch dropped

    @jax.jit
    def fused_update(
        params, opt_state, key,
        self_obs,       # (K, S, OBS_DIM)
        self_h_prev,    # (K, S, HIDDEN_DIM)
        self_legal,     # (K, S, NUM_ACTIONS)
        self_actions,   # (K, S)
        self_log_probs, # (K, S)
        self_values,    # (K, S)
        self_rewards,   # (K, S)
        self_dones,     # (K, S)
        n_self,         # (K,)
    ):
        # ── JAX GAE (vectorised over K episodes) ───────────────────────────
        advantages, returns = jax.vmap(
            lambda r, v, d: _gae_single_jax(r, v, d, gamma, gae_lambda)
        )(self_rewards, self_values, self_dones)

        # ── Flatten (K, S, ...) → (N, ...) ────────────────────────────────
        obs_f   = self_obs.reshape(N, self_obs.shape[-1])
        h_f     = self_h_prev.reshape(N, self_h_prev.shape[-1])
        legal_f = self_legal.reshape(N, self_legal.shape[-1])
        act_f   = self_actions.reshape(N)
        lps_f   = self_log_probs.reshape(N)
        adv_f   = advantages.reshape(N)
        ret_f   = returns.reshape(N)

        # ── Validity mask: True for real self-turns, False for padding ──────
        step_idx = jnp.arange(MAX_SELF_TURNS)[None, :]        # (1, S)
        valid    = (step_idx < n_self[:, None]).reshape(N)     # (N,)

        # ── Advantage normalisation over valid transitions only ─────────────
        n_valid  = jnp.maximum(jnp.sum(valid).astype(jnp.float32), 1.0)
        adv_mean = jnp.sum(jnp.where(valid, adv_f, 0.0)) / n_valid
        adv_var  = (
            jnp.sum(jnp.where(valid, (adv_f - adv_mean) ** 2, 0.0)) / n_valid
        )
        adv_norm = (adv_f - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)
        adv_norm = jnp.where(valid, adv_norm, 0.0)   # zero out padding

        # ── PPO epochs via nested lax.scan ──────────────────────────────────
        def epoch_fn(carry, perm_key):
            params, opt_state = carry
            perm = jax.random.permutation(perm_key, N)

            def mb_fn(carry, mb_idx):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice_in_dim(
                    perm, mb_idx * minibatch_size, minibatch_size
                )
                params, opt_state, al, vl, ent, cf = ppo_step_fn(
                    params, opt_state,
                    obs_f[idx], h_f[idx], legal_f[idx],
                    act_f[idx], lps_f[idx], adv_norm[idx], ret_f[idx],
                )
                return (params, opt_state), (al, vl, ent, cf)

            (params, opt_state), mb_losses = jax.lax.scan(
                mb_fn, (params, opt_state), jnp.arange(n_mb)
            )
            return (params, opt_state), mb_losses

        epoch_keys = jax.random.split(key, ppo_epochs)
        (params, opt_state), all_losses = jax.lax.scan(
            epoch_fn, (params, opt_state), epoch_keys
        )

        return (
            params, opt_state,
            jnp.mean(all_losses[0]),   # actor_loss
            jnp.mean(all_losses[1]),   # value_loss
            jnp.mean(all_losses[2]),   # entropy
            jnp.mean(all_losses[3]),   # clip_frac
        )

    return fused_update


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RPPOAgentJax:
    """
    Recurrent PPO oracle for cooperative Tiny Hanabi.

    Drop-in replacement for CooperativeODCFRAgentJax as the XDO inner solver.
    Accepts the same profile pool / meta_strategy interface.

    Episode collection uses run_K_rppo_episodes (lax.scan + vmap) — no Python
    loop in the hot path.  On GPU this gives ~10-50× speedup over the
    sequential Python-loop version.

    Public attributes (for diagnostic scripts):
        params            : RPPOActorCritic Flax param dict
        player_id         : 0 or 1
        _partner_temp     : BC softmax temperature
        last_actor_loss   : mean actor loss from last update()
        last_value_loss   : mean value loss from last update()
        last_entropy      : mean policy entropy from last update()
        last_clip_frac    : fraction of ratios clipped from last update()
        last_mean_return  : mean GAE return from last update()
        last_mean_score   : mean episode score from last collect_episodes()
    """

    def __init__(
        self,
        agent_id:            str,
        lr:                  float = 3e-4,
        gamma:               float = 0.99,
        gae_lambda:          float = 0.95,
        clip_eps:            float = 0.2,
        ent_coef:            float = 0.01,
        vf_coef:             float = 0.5,
        ppo_epochs:          int   = 4,
        minibatch_size:      int   = 256,
        max_grad_norm:       float = 0.5,
        partner_temperature: float = 0.3,
        aux_loss_weight:     float = 0.0,
        seed:                int   = 0,
        device                     = None,   # jax.Device to pin to; None → default
    ):
        self.agent_id      = agent_id
        self.player_id     = _PLAYER_IDS[agent_id]
        self._partner_temp = partner_temperature
        self._gamma        = gamma
        self._gae_lambda   = gae_lambda
        self._ppo_epochs   = ppo_epochs
        self._mbs          = minibatch_size
        self._aux_loss_weight = aux_loss_weight
        self._device       = device if device is not None else jax.devices()[0]

        # Per-instance numpy RNG — avoids global numpy state races when two
        # agents run update() concurrently in different threads.
        self._np_rng = np.random.default_rng(seed)

        key       = jax.random.PRNGKey(seed)
        dummy_obs = jnp.zeros((1, OBS_DIM))
        dummy_h   = jnp.zeros((1, HIDDEN_DIM))
        # Pin params to the agent's device immediately after initialisation.
        self.params = jax.device_put(
            _ac_net.init(key, dummy_obs, dummy_h), self._device
        )

        optimizer       = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(lr),
        )
        self._opt       = optimizer
        self._opt_state = jax.device_put(
            optimizer.init(self.params), self._device
        )
        self._ppo_step  = _make_ppo_step(optimizer, clip_eps, ent_coef, vf_coef, aux_loss_weight)

        # Store hyperparams needed to build the fused update lazily.
        self._clip_eps = clip_eps
        self._ent_coef = ent_coef
        self._vf_coef  = vf_coef
        # Lazy cache for _make_fused_update — rebuilt if K changes.
        self._fused_update_K:  int   = -1
        self._fused_update_fn        = None

        # Logging
        self.last_actor_loss:  float = float("nan")
        self.last_value_loss:  float = float("nan")
        self.last_entropy:     float = float("nan")
        self.last_clip_frac:   float = float("nan")
        self.last_mean_return: float = float("nan")
        self.last_mean_score:  float = float("nan")

    # ------------------------------------------------------------------
    # Episode collection
    # ------------------------------------------------------------------

    def collect_episodes(
        self,
        profiles:      list,
        meta_strategy: np.ndarray,
        n_episodes:    int,
        key:           jax.Array,
    ) -> list:
        """
        Run n_episodes against BC profiles weighted by meta_strategy.

        Uses run_K_rppo_episodes (lax.scan + vmap) — all K episodes for each
        profile are dispatched to the accelerator in a single JIT call.

        Rollouts are split across profiles proportionally using the
        largest-remainder method (same as CooperativeODCFRAgentJax).

        Returns:
            list of trajectory dicts, one per episode.
        """
        K_ks      = _split_K(meta_strategy, n_episodes)
        all_trajs: list = []

        for profile, K_k in zip(profiles, K_ks):
            K_k = int(K_k)
            if K_k == 0:
                continue

            # Round K_k UP to the nearest power of 2 so that
            # run_K_rppo_episodes is only ever JIT-compiled for
            # ≤9 distinct batch sizes {1,2,4,8,16,32,64,128,256}.
            # Without this, every unique K_k value produced by
            # _split_K over a growing profile pool triggers a new
            # XLA compilation; after 40+ iterations that cache
            # grows large enough to OOM the Kaggle kernel.
            K_padded = 1
            while K_padded < K_k:
                K_padded <<= 1

            key, k_ep  = jax.random.split(key)
            # Pin episode keys and BC params to this agent's device so the JIT
            # kernel dispatches there even when called from a background thread.
            ep_keys    = jax.device_put(jax.random.split(k_ep, K_padded), self._device)
            bc_params  = jax.device_put(profile.bc_model_params, self._device)
            results    = run_K_rppo_episodes(
                ep_keys, self.params, bc_params,
                self.player_id, self._partner_temp,
            )
            # Block until JAX computation finishes before the host-side
            # _results_to_trajs call to avoid hidden async latency.
            results["score"].block_until_ready()
            # Trim the K_padded results back to K_k — the extra episodes are
            # discarded on the CPU side (no re-tracing, numpy slicing only).
            if K_padded > K_k:
                results = {k: v[:K_k] for k, v in results.items()}
            all_trajs.extend(_results_to_trajs(results))

        self.last_mean_score = (
            float(np.mean([t["score"] for t in all_trajs]))
            if all_trajs else 0.0
        )
        return all_trajs

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(self, trajectories: list) -> dict:
        """
        Compute GAE on collected trajectories then run PPO epochs.

        Args:
            trajectories : list of dicts from collect_episodes() / _results_to_trajs()

        Returns:
            dict of mean losses: actor_loss, value_loss, entropy, clip_frac,
            mean_return.  Also updates self.last_* attributes.
        """
        all_obs, all_h, all_legal = [], [], []
        all_actions, all_log_probs = [], []
        all_advantages, all_returns = [], []
        all_partner_actions = []

        for traj in trajectories:
            if traj["obs"].shape[0] == 0:
                continue
            adv, ret = _compute_gae(
                traj["rewards"], traj["values"], traj["dones"],
                last_value=0.0,   # terminal → bootstrap = 0
                gamma=self._gamma,
                gae_lambda=self._gae_lambda,
            )
            all_obs.append(traj["obs"])
            all_h.append(traj["h_prev"])
            all_legal.append(traj["legal"])
            all_actions.append(traj["actions"])
            all_log_probs.append(traj["log_probs"])
            all_partner_actions.append(traj.get("partner_actions", jnp.zeros(traj["obs"].shape[0], jnp.int32)))
            all_advantages.append(adv)
            all_returns.append(ret)

        if not all_obs:
            return {}

        # Pin all training arrays to this agent's device before any JIT dispatch.
        # When two agents run update() concurrently in separate threads each set
        # of arrays lands on its own GPU, avoiding cross-device transfers.
        obs        = jax.device_put(jnp.array(np.concatenate(all_obs)),       self._device)
        h_prev     = jax.device_put(jnp.array(np.concatenate(all_h)),         self._device)
        legal      = jax.device_put(jnp.array(np.concatenate(all_legal)),      self._device)
        actions    = jax.device_put(jnp.array(np.concatenate(all_actions)),    self._device)
        old_lps    = jax.device_put(jnp.array(np.concatenate(all_log_probs)), self._device)
        partner_actions = jax.device_put(jnp.array(np.concatenate(all_partner_actions)), self._device) if all_partner_actions else None
        returns    = jax.device_put(jnp.array(np.concatenate(all_returns)),    self._device)

        # Normalise advantages over the full batch before splitting into
        # minibatches.  Per-minibatch normalisation gives each minibatch a
        # different effective scale, producing inconsistent gradient magnitudes
        # across the epoch.  Batch-level normalisation ensures every minibatch
        # shares the same advantage scale, giving the actor a stable signal.
        adv_raw    = np.concatenate(all_advantages)
        advantages = jax.device_put(
            jnp.array((adv_raw - adv_raw.mean()) / (adv_raw.std() + 1e-8)),
            self._device,
        )

        N = obs.shape[0]
        actor_ls, value_ls, entropies, clip_fracs = [], [], [], []

        for _ in range(self._ppo_epochs):
            # Use per-instance RNG (not global np.random) for thread safety.
            perm = self._np_rng.permutation(N)
            for start in range(0, N, self._mbs):
                idx = jax.device_put(
                    jnp.array(perm[start: start + self._mbs]), self._device
                )
                (self.params, self._opt_state,
                 al, vl, ent, cf) = self._ppo_step(
                    self.params, self._opt_state,
                    obs[idx], h_prev[idx], legal[idx],
                    actions[idx], old_lps[idx],
                    advantages[idx], returns[idx],
                    partner_actions[idx] if partner_actions is not None else None,
                )
                actor_ls.append(float(al))
                value_ls.append(float(vl))
                entropies.append(float(ent))
                clip_fracs.append(float(cf))

        self.last_actor_loss  = float(np.mean(actor_ls))
        self.last_value_loss  = float(np.mean(value_ls))
        self.last_entropy     = float(np.mean(entropies))
        self.last_clip_frac   = float(np.mean(clip_fracs))
        self.last_mean_return = float(returns.mean())

        return {
            "actor_loss":  self.last_actor_loss,
            "value_loss":  self.last_value_loss,
            "entropy":     self.last_entropy,
            "clip_frac":   self.last_clip_frac,
            "mean_return": self.last_mean_return,
        }

    # ------------------------------------------------------------------
    # Fused update — GAE + all PPO epochs in one GPU dispatch
    # ------------------------------------------------------------------

    def _ensure_fused_update(self, K: int) -> None:
        """Lazily build (or rebuild) the fused update function for batch size K."""
        if K != self._fused_update_K:
            self._fused_update_fn = _make_fused_update(
                self._opt,
                self._clip_eps,
                self._ent_coef,
                self._vf_coef,
                self._ppo_epochs,
                self._mbs,
                self._gamma,
                self._gae_lambda,
                K,
            )
            self._fused_update_K = K

    def update_fused(
        self,
        results: dict,
        key:     jax.Array,
    ) -> dict:
        """
        Fused GAE + all PPO epochs in a single GPU dispatch.

        Use in place of update() when episode data is already on-device as
        padded JAX arrays (i.e. from run_K_rppo_episodes directly, before
        _results_to_trajs strips the padding).

        Reduces per-update GPU dispatches from
            1 (collect) + ppo_epochs × n_minibatches (update)
        to
            1 (collect) + 1 (fused GAE+update).

        Args:
            results : dict from run_K_rppo_episodes — (K, MAX_SELF_TURNS, ...)
                      arrays already on self._device.
            key     : PRNG key consumed for minibatch permutations.

        Returns:
            loss dict with actor_loss, value_loss, entropy, clip_frac.
        """
        K = int(results["self_obs"].shape[0])
        self._ensure_fused_update(K)

        (self.params, self._opt_state,
         al, vl, ent, cf) = self._fused_update_fn(
            self.params, self._opt_state, key,
            results["self_obs"],
            results["self_h_prev"],
            results["self_legal"],
            results["self_actions"],
            results["self_log_probs"],
            results["self_values"],
            results["self_rewards"],
            results["self_dones"],
            results["n_self"],
        )

        self.last_actor_loss = float(al)
        self.last_value_loss = float(vl)
        self.last_entropy    = float(ent)
        self.last_clip_frac  = float(cf)

        return {
            "actor_loss": self.last_actor_loss,
            "value_loss": self.last_value_loss,
            "entropy":    self.last_entropy,
            "clip_frac":  self.last_clip_frac,
        }


# ---------------------------------------------------------------------------
# Joint episode simulation — both players use RPPOActorCritic
# (for the XDO outer loop; mirrors _run_joint_episode / run_N_joint_episodes
#  in simulation.py so that joint_results_to_episode_dicts can be reused)
# ---------------------------------------------------------------------------

def _run_joint_rppo_episode(key, ac_params_0, ac_params_1):
    """
    Simulate one episode where player 0 uses ac_params_0 and player 1 uses
    ac_params_1 (both RPPOActorCritic).  Each player's GRU carry is updated
    only on their own turns.

    Returns a dict with the same keys as _run_joint_episode in simulation.py:
        score       ()                    float32
        obs0        (MAX_EPISODE_STEPS, OBS_DIM)  float32
        obs1        (MAX_EPISODE_STEPS, OBS_DIM)  float32
        cur_players (MAX_EPISODE_STEPS,)  int32
        actions     (MAX_EPISODE_STEPS,)  int32
        n_steps     ()                    int32

    This identical key set means joint_results_to_episode_dicts() works
    unchanged on the output of run_N_joint_rppo_episodes().
    """
    k_reset, k_steps = jax.random.split(key)
    step_keys = jax.random.split(k_steps, MAX_EPISODE_STEPS)

    init_state = _env_reset(k_reset)

    init_carry = {
        "state":       init_state,
        "prev_state":  init_state,
        "prev_act":    jnp.array(0, jnp.int32),
        "h_0":         jnp.zeros(HIDDEN_DIM, jnp.float32),  # player 0 GRU carry
        "h_1":         jnp.zeros(HIDDEN_DIM, jnp.float32),  # player 1 GRU carry
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

            is_0 = state.cur_player_idx[0] > 0.5   # scalar bool

            # ---- Player 0 turn ----------------------------------------------
            def p0_turn(args):
                step_key, carry = args
                logits, _, new_h, _ = _ac_net.apply(ac_params_0, obs_0, carry["h_0"])
                masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
                probs  = jax.nn.softmax(masked)
                action = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                return action.astype(jnp.int32), {**carry, "h_0": new_h}

            # ---- Player 1 turn ----------------------------------------------
            def p1_turn(args):
                step_key, carry = args
                logits, _, new_h, _ = _ac_net.apply(ac_params_1, obs_1, carry["h_1"])
                masked = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
                probs  = jax.nn.softmax(masked)
                action = jax.random.choice(step_key, NUM_ACTIONS, p=probs)
                return action.astype(jnp.int32), {**carry, "h_1": new_h}

            action, new_carry = jax.lax.cond(
                is_0, p0_turn, p1_turn, (step_key, carry)
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
        "obs0":        final_carry["obs0"],
        "obs1":        final_carry["obs1"],
        "cur_players": final_carry["cur_players"],
        "actions":     final_carry["actions"],
        "n_steps":     final_carry["n_steps"],
    }


@jax.jit
def run_N_joint_rppo_episodes(keys, ac_params_0, ac_params_1):
    """
    Simulate N joint RPPO episodes in parallel via vmap.

    Args:
        keys       : (N, 2) JAX key array — one per episode
        ac_params_0: RPPOActorCritic Flax variable dict for player 0
        ac_params_1: RPPOActorCritic Flax variable dict for player 1

    Returns:
        dict of JAX arrays each with leading dimension N.
        Keys: score, obs0, obs1, cur_players, actions, n_steps.
        Compatible with joint_results_to_episode_dicts() from simulation.py.
    """
    return jax.vmap(
        lambda k: _run_joint_rppo_episode(k, ac_params_0, ac_params_1)
    )(keys)
