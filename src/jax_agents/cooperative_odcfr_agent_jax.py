# src/jax_agents/cooperative_odcfr_agent_jax.py
# JAX port of CooperativeODCFRAgent.
#
# Key differences from the PyTorch version
# ─────────────────────────────────────────
# • Simulation  : run_K_episodes() via jax.vmap — all K rollouts in one JIT call.
# • GRU replay  : jax.vmap over the batch rather than pack_padded_sequence.
# • Training    : JIT-compiled via closures built once at __init__.
# • No aux loss : next-action prediction auxiliary loss is omitted for
#                 simplicity; the advantage loss alone gives the GRU adequate
#                 gradient signal once the buffer has enough samples.
# • Buffers     : numpy-backed ReservoirBuffer (same API as PyTorch version).
#
# Optimizer structure
# ───────────────────
# • adv_opt  : updates adv_params only (fast path — stored obs_h, no GRU).
# • gru_opt  : updates (adv_params, gru_params) jointly (GRU re-encoding with
#              gradients, same signal as the PyTorch _update_gru).
# • pol_opt  : updates pol_params only (cross-entropy vs regret-matched sigma).
#
# Public interface
# ────────────────
#   train(profile, cfr_iterations, ...) → pol_params (trained PolicyNet params)

from __future__ import annotations

import time
from functools import partial
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from src.jax_networks.gru_encoder   import HIDDEN_DIM, NUM_ACTIONS
from src.jax_agents.buffer          import ReservoirBuffer
from src.jax_agents.simulation      import (
    run_K_episodes,
    collect_to_buffers,
    MAX_PARTNER_TURNS,
    OBS_DIM,
    _adv_net,
    _gru_enc,
    _pol_net,
    _aux_head,
)

_PLAYER_IDS = {"agent_A": 0, "agent_B": 1}
IN_DIM = OBS_DIM + HIDDEN_DIM  # 148

# Clipped importance-sampling cap for outcome-sampling MCCFR.
# Targets for the sampled action are min(raw_score / σ(a*), IS_CLIP) rather
# than raw_score.  This corrects the bias that chronically underestimates
# Q-values for rarely-sampled actions (σ(a*) near the ε-floor ≈ 0.0075 gives
# raw amplification of ~133×; capping at 10 limits variance while preserving
# most of the correction signal).  σ(a*) is stored in adv_buf entry index 6.
IS_CLIP = 10.0


def _split_K(weights: np.ndarray, K_total: int) -> np.ndarray:
    """
    Split K_total rollouts across N profiles proportional to weights using the
    largest-remainder method, guaranteeing the result sums exactly to K_total.

    Profiles with weight rounded to 0 receive 0 rollouts and are skipped in the
    simulation loop, so K_total is effectively concentrated on the profiles that
    actually carry meta-strategy weight.

    Args:
        weights : (N,) float array — already normalised (sums to 1.0)
        K_total : total number of rollouts to distribute

    Returns:
        (N,) int array summing to K_total
    """
    raw   = weights * K_total
    K_ks  = np.floor(raw).astype(int)
    remainder = K_total - K_ks.sum()
    # Assign remainder slots to profiles with the largest fractional parts.
    fracs = raw - K_ks
    for idx in np.argsort(-fracs)[:int(remainder)]:
        K_ks[idx] += 1
    return K_ks


# ---------------------------------------------------------------------------
# GRU re-encoding helper — fixes stale obs_h in replay buffers
# ---------------------------------------------------------------------------

@jax.jit
def _reenc_obs_h_batches(gru_params, all_obs_raw, all_p_acts, all_pfx_lens):
    """
    Re-encode h_oppo from stored partner action sequences using the CURRENT
    gru_params, then concatenate with raw obs to produce fresh obs_h.

    This fixes the stale-obs_h problem: buffers store obs_h computed with
    historical GRU checkpoints; after a GRU update those encodings no longer
    match the current GRU, causing the advantage/policy nets to train on an
    inconsistent representation.  Re-encoding here ensures _update_adv and
    _update_pol always see obs_h that is consistent with the current GRU.

    Args:
        gru_params   : current GRU Flax param dict (traced, not recompiled)
        all_obs_raw  : (n_steps, B, OBS_DIM)         float32
        all_p_acts   : (n_steps, B, MAX_PARTNER_TURNS) int32
        all_pfx_lens : (n_steps, B)                  int32

    Returns:
        (n_steps, B, OBS_DIM + HIDDEN_DIM) float32 — fresh obs_h
    """
    def encode_one_batch(obs_raw, p_acts, pfx_lens):
        def encode_one(acts, pfx_len):
            return _gru_enc.apply(gru_params, acts, pfx_len)   # (HIDDEN_DIM,)
        h_oppo = jax.vmap(encode_one)(p_acts, pfx_lens)        # (B, HIDDEN_DIM)
        return jnp.concatenate([obs_raw, h_oppo], axis=-1)     # (B, IN_DIM)

    return jax.vmap(encode_one_batch)(all_obs_raw, all_p_acts, all_pfx_lens)


# ---------------------------------------------------------------------------
# JIT-compiled step factories — built once at agent init, reused every call
# ---------------------------------------------------------------------------

def _make_adv_step(optimizer: optax.GradientTransformation):
    L2_REG = 1e-5  # L2 regularization to prevent Q-value explosion

    @jax.jit
    def step(params, opt_state, all_obs_h, all_actions, all_w_scores, all_weights):
        """
        Multi-step weighted MSE update via lax.scan — one JIT call for all steps.

        Targets are built on-the-fly from the current network's Q predictions,
        with the sampled action's Q overridden by the importance-weighted score.
        Non-sampled actions use stop_gradient(Q_net) as their target — they
        contribute zero gradient, which is correct for outcome-sampling MCCFR
        (we have no unbiased estimate for actions we didn't sample).

        Args:
            all_obs_h    : (n_steps, B, IN_DIM) — fresh obs_h (re-encoded with
                           current GRU; computed by _reenc_obs_h_batches before
                           this call)
            all_actions  : (n_steps, B)         — int32 taken action indices
            all_w_scores : (n_steps, B)         — float32 clipped IS-corrected scores
                           = min(raw_score / σ(a*), IS_CLIP) per (episode, self-turn)
                           computed in _update_adv from adv_buf entries b[4] and b[6]
            all_weights  : (n_steps, B)         — float32 recency weights (iter_t / Σ)
        """
        def body(carry, batch):
            p, s = carry
            obs_h_b, actions_b, w_scores_b, weights_b = batch

            def loss_fn(p):
                preds = _adv_net.apply(p, obs_h_b)           # (B, 8)
                preds = jnp.clip(preds, -50.0, 50.0)        # clip to prevent explosion
                # Targets: frozen current-Q for non-sampled actions (→ zero
                # gradient there), importance-weighted score for sampled action.
                # stop_gradient ensures non-sampled actions see no gradient.
                targets = jax.lax.stop_gradient(preds).at[
                    jnp.arange(preds.shape[0]), actions_b
                ].set(w_scores_b)                             # (B, 8)
                mse = (weights_b * ((preds - targets) ** 2).mean(-1)).sum()
                l2 = L2_REG * (preds ** 2).mean()  # L2 regularization
                return mse + l2

            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, new_s = optimizer.update(grads, s)
            return (optax.apply_updates(p, updates), new_s), loss

        (new_params, new_state), losses = jax.lax.scan(
            body, (params, opt_state),
            (all_obs_h, all_actions, all_w_scores, all_weights),
        )
        return new_params, new_state, losses.mean()

    return step


def _make_gru_step(optimizer: optax.GradientTransformation, aux_loss_weight: float = 0.5):
    L2_REG = 1e-5  # L2 regularization to prevent Q-value explosion

    @jax.jit
    def step(adv_params, gru_params, aux_params, opt_state,
             all_obs_b, all_p_acts, all_pfx_lens, all_actions, all_w_scores, all_weights):
        """
        Multi-step joint update of adv_params + gru_params + aux_params via lax.scan.
        GRU re-encodes partner sequences WITH gradients; aux next-action prediction
        loss provides additional gradient signal to the GRU.

        Targets for the adv loss are built on-the-fly (same as _make_adv_step):
        stop_gradient(Q_net) for non-sampled actions, IS-weighted score for the
        sampled action.  This ensures GRU gradients come from fresh Q predictions,
        not from stale buffer targets.

        Args:
            all_obs_b    : (n_steps, B, OBS_DIM)
            all_p_acts   : (n_steps, B, MAX_PARTNER_TURNS)
            all_pfx_lens : (n_steps, B)
            all_actions  : (n_steps, B)  — int32 taken action indices
            all_w_scores : (n_steps, B)  — float32 IS-weighted scores
            all_weights  : (n_steps, B)  — float32 IS weights
        """
        def body(carry, batch):
            p_triple, opt_s = carry
            obs_b, p_acts_b, pfx_lens_b, actions_b, w_scores_b, weights_b = batch

            def loss_fn(pt):
                adv_p, gru_p, aux_p = pt

                # Encode each partner sequence, collecting all intermediate states.
                def encode_one(acts, pfx_len):
                    _, states = _gru_enc.apply(gru_p, acts, pfx_len,
                                               return_states=True)
                    return states  # (MAX_PARTNER_TURNS, HIDDEN_DIM)

                all_states = jax.vmap(encode_one)(p_acts_b, pfx_lens_b)
                # (B, MAX_PARTNER_TURNS, HIDDEN_DIM)

                # h_oppo for advantage: hidden state after pfx_len valid steps.
                def final_h(states, pfx_len):
                    idx = jnp.maximum(pfx_len - 1, 0)
                    return jnp.where(pfx_len > 0, states[idx], jnp.zeros(HIDDEN_DIM))

                h_oppo = jax.vmap(final_h)(all_states, pfx_lens_b)  # (B, HIDDEN_DIM)
                obs_h  = jnp.concatenate([obs_b, h_oppo], -1)        # (B, IN_DIM)
                preds  = _adv_net.apply(adv_p, obs_h)                # (B, 8)
                preds = jnp.clip(preds, -50.0, 50.0)            # clip to prevent explosion

                # On-the-fly targets: frozen Q for non-sampled, IS-score for sampled.
                targets = jax.lax.stop_gradient(preds).at[
                    jnp.arange(preds.shape[0]), actions_b
                ].set(w_scores_b)                                     # (B, 8)
                mse = (weights_b * ((preds - targets) ** 2).mean(-1)).sum()
                l2 = L2_REG * (preds ** 2).mean()
                adv_loss = mse + l2

                # Aux: predict next partner action from each intermediate GRU state.
                def aux_loss_one(states, acts, pfx_len):
                    # states : (MAX_PARTNER_TURNS, HIDDEN_DIM)
                    # valid prediction positions: t in 0 .. pfx_len-2
                    logits   = jax.vmap(
                        lambda h: _aux_head.apply(aux_p, h)
                    )(states[:-1])                              # (MAX_PARTNER_TURNS-1, 8)
                    next_acts = acts[1:]                        # (MAX_PARTNER_TURNS-1,)
                    xe = optax.softmax_cross_entropy_with_integer_labels(
                        logits, next_acts
                    )                                           # (MAX_PARTNER_TURNS-1,)
                    t_idx   = jnp.arange(MAX_PARTNER_TURNS - 1)
                    mask    = (t_idx < (pfx_len - 1)).astype(jnp.float32)
                    n_valid = jnp.maximum(1.0, (pfx_len - 1).astype(jnp.float32))
                    return (mask * xe).sum() / n_valid

                aux_loss = jax.vmap(aux_loss_one)(
                    all_states, p_acts_b, pfx_lens_b
                ).mean()

                return adv_loss + aux_loss_weight * aux_loss

            loss, grads = jax.value_and_grad(loss_fn)(p_triple)
            updates, new_opt_s = optimizer.update(grads, opt_s)
            new_p_triple = optax.apply_updates(p_triple, updates)
            return (new_p_triple, new_opt_s), loss

        init_carry = ((adv_params, gru_params, aux_params), opt_state)
        (new_p_triple, new_opt_state), losses = jax.lax.scan(
            body, init_carry,
            (all_obs_b, all_p_acts, all_pfx_lens, all_actions, all_w_scores, all_weights),
        )
        new_adv, new_gru, new_aux = new_p_triple
        return new_adv, new_gru, new_aux, new_opt_state, losses.mean()

    return step


@jax.jit
def _compute_mean_regret(adv_params: dict, obs_h: jnp.ndarray) -> jnp.ndarray:
    """
    Compute mean positive regret from the advantage network's current Q-estimates.

    For each observation in obs_h:
      1. Get Q(a) for all actions via the adv network.
      2. Apply regret matching: σ(a) ∝ max(0, Q(a)).
      3. Compute V(σ) = Σ_a σ(a) · Q(a)  (value under the regret-matched strategy).
      4. Instantaneous regret r(a) = Q(a) − V(σ).
      5. Mean positive regret = mean over observations and actions of max(0, r(a)).

    A value near zero means the current policy is close to a Nash equilibrium for
    the observed partner.  A large value means CFR still has significant gains to
    find.  In a 4-point game this typically starts around 1–2 and should trend
    toward 0.1–0.3 as the CFR loop converges.

    Args:
        adv_params : current advantage-net Flax params
        obs_h      : (N, IN_DIM) re-encoded observations

    Returns:
        scalar — mean positive regret across all observations and actions
    """
    q_vals  = _adv_net.apply(adv_params, obs_h)          # (N, 8)
    pos_q   = jnp.maximum(0.0, q_vals)                   # regret-matching numerators
    denom   = pos_q.sum(axis=-1, keepdims=True) + 1e-8
    sigma   = pos_q / denom                               # (N, 8) — regret-matched strategy
    v_sigma = (sigma * q_vals).sum(axis=-1, keepdims=True)  # (N, 1) — value under strategy
    regret  = q_vals - v_sigma                            # (N, 8) — per-action regret
    return jnp.mean(jnp.maximum(0.0, regret))            # scalar


def _make_pol_step(optimizer: optax.GradientTransformation):
    @jax.jit
    def step(params, opt_state, all_obs_h, all_sigma, all_weights):
        """
        Multi-step weighted cross-entropy update via lax.scan.

        Later CFR iterations produce better regret-matched sigmas (more
        traversals have accumulated), so their entries are up-weighted by
        iter_t / sum(iter_ts) — identical to the adv_net linear weighting.

        Args:
            all_obs_h   : (n_steps, B, 148)
            all_sigma   : (n_steps, B, 8)
            all_weights : (n_steps, B) float32 — per-sample weights, each
                          batch pre-normalised so weights.sum() == 1.0
        """
        def body(carry, batch):
            p, s = carry
            obs_h_b, sigma_b, weights_b = batch
            def loss_fn(p):
                logits = _pol_net.apply(p, obs_h_b)
                ce = -(sigma_b * jax.nn.log_softmax(logits, -1)).sum(-1)  # (B,)
                return (weights_b * ce).sum()   # weighted sum (weights sum to 1)
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, new_s = optimizer.update(grads, s)
            return (optax.apply_updates(p, updates), new_s), loss

        (new_params, new_state), losses = jax.lax.scan(
            body, (params, opt_state), (all_obs_h, all_sigma, all_weights)
        )
        return new_params, new_state, losses.mean()

    return step


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CooperativeODCFRAgentJax:
    """
    Cooperative ODCFR oracle — JAX/Flax port of CooperativeODCFRAgent.
    Manages AdvantageNet, PolicyNet, and GRUEncoder params via optax.
    """

    def __init__(
        self,
        agent_id: str,
        K_simulations: int = 16,
        adv_buffer_size: int = 1_000_000,
        adv_train_steps: int = 100,
        gru_train_steps: int = 10,
        pol_train_steps: int = 200,
        batch_size: int = 512,
        adv_lr: float = 1e-3,
        pol_lr: float = 1e-3,
        gru_lr: float = 5e-5,        # lowered from 1e-4 — prevents GRU representation
                                      # oscillation; with gru_train_steps=10 this is stable
        gru_grad_clip: float = 0.3,  # global-norm clip applied before Adam update
        partner_temperature: float = 0.3,
        aux_loss_weight: float = 0.2,
        explore_eps: float = 0.06,   # ε-on-policy exploration to prevent action collapse
        seed: int = 0,
    ):
        self.agent_id         = agent_id
        self.player_id        = _PLAYER_IDS[agent_id]
        self.K                = K_simulations
        self._adv_steps       = adv_train_steps
        self._gru_steps       = gru_train_steps
        self._pol_steps       = pol_train_steps
        self._batch           = batch_size
        self._partner_temp    = partner_temperature
        self._aux_loss_weight = aux_loss_weight
        self._explore_eps     = explore_eps

        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # ---- Network params -----------------------------------------------
        dummy_in   = jnp.zeros((1, IN_DIM))
        dummy_acts = jnp.zeros((MAX_PARTNER_TURNS,), dtype=jnp.int32)
        dummy_h    = jnp.zeros((1, HIDDEN_DIM))

        self.adv_params = _adv_net.init(k1, dummy_in)
        self.pol_params = _pol_net.init(k2, dummy_in)
        self.gru_params = _gru_enc.init(k3, dummy_acts, jnp.array(1))
        self.aux_params = _aux_head.init(k4, dummy_h)

        # ---- Optimisers ---------------------------------------------------
        adv_opt = optax.adam(adv_lr)
        # GRU optimiser: gradient clipping BEFORE Adam to prevent the large
        # parameter updates that caused cos_sim=0.73 in diagnostics.
        # clip_by_global_norm rescales the entire gradient vector so its L2
        # norm ≤ gru_grad_clip, then Adam applies the adaptive LR on top.
        gru_opt = optax.chain(
            optax.clip_by_global_norm(gru_grad_clip),
            optax.adam(gru_lr),
        )
        pol_opt = optax.adam(pol_lr)

        self._adv_opt_state = adv_opt.init(self.adv_params)
        self._gru_opt_state = gru_opt.init(
            (self.adv_params, self.gru_params, self.aux_params)
        )
        self._pol_opt_state = pol_opt.init(self.pol_params)

        # ---- JIT steps (compiled once) ------------------------------------
        self._adv_step = _make_adv_step(adv_opt)
        self._gru_step = _make_gru_step(gru_opt, aux_loss_weight)
        self._pol_step = _make_pol_step(pol_opt)

        # ---- Replay buffers -----------------------------------------------
        self._adv_buf = ReservoirBuffer(adv_buffer_size)
        self._pol_buf = ReservoirBuffer(adv_buffer_size)

        # ---- Timing / logging --------------------------------------------
        self.last_train_sim_secs:  float = 0.0
        self.last_train_adv_secs:  float = 0.0
        self.last_train_gru_secs:  float = 0.0
        self.last_train_pol_secs:  float = 0.0
        self.last_cfr_iterations_run: int = 0
        self.last_probe_delta: float = float("nan")
        self.last_adv_loss:    float = float("nan")  # D1 — advantage net training loss
        self.last_gru_loss:    float = float("nan")  # D2b — GRU joint-update training loss
        self.last_mean_regret: float = float("nan")  # mean positive regret from adv net

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def train(
        self,
        profile=None,
        cfr_iterations: int = 100,
        prewarm_iters: int = 12,
        k_simulations: Optional[int] = None,
        early_stop_patience: int = 20,
        early_stop_delta: float = 1e-2,
        early_stop_min_iters: int = 250,
        key: Optional[jax.Array] = None,
        # Soft-aggregate oracle arguments
        profiles: Optional[list] = None,
        meta_strategy: Optional[np.ndarray] = None,
        # Pol-net update frequency
        pol_update_every: int = 50,
    ):
        """
        Run Cooperative ODCFR for cfr_iterations iterations.

        Supports two modes:

        Single-profile (original):
            train(profile=p, ...)
            All K rollouts per CFR iteration run against profile p.
            Used by diagnose_oracle.py and for backward compatibility.

        Soft-aggregate oracle:
            train(profiles=[p0, p1, ...], meta_strategy=[w0, w1, ...], ...)
            K rollouts per CFR iteration are split across profiles proportional
            to meta_strategy weights (largest-remainder method, sums to K).
            The advantage buffer receives data from the full convention mixture,
            so the oracle trains a best-response to the meta-strategy distribution
            rather than a single sampled convention.  This directly implements the
            XDO oracle subgame: best-respond to the current meta-strategy.

            Profiles with very low weight receive 0 rollouts (floor of w*K == 0)
            and are skipped — compute concentrates on the conventions that
            actually matter for the meta-strategy.

        Args:
            profile        : single BehaviourProfileJax (single-profile mode)
            cfr_iterations : CFR traversal budget
            prewarm_iters  : warm-up rollouts before the main CFR loop (see below)
            k_simulations  : total rollouts per CFR iteration (default: self.K)
            key            : JAX PRNG key (auto-generated if None)
            profiles       : list of BehaviourProfileJax (soft-oracle mode)
            meta_strategy  : weight array aligned with profiles (soft-oracle mode)

        Returns:
            pol_params — trained PolicyNet params (snapshot → HanabiPolicyJax)
        """
        if key is None:
            key = jax.random.PRNGKey(0)

        K = k_simulations if k_simulations is not None else self.K

        # --- Resolve profile list and per-profile rollout counts ---------------
        if profiles is not None:
            # Soft-aggregate oracle mode
            _profiles = list(profiles)
            _weights  = np.asarray(meta_strategy, dtype=np.float64)
            _weights  = _weights / _weights.sum()   # normalise defensively
        elif profile is not None:
            # Single-profile mode (backward compat / diagnose_oracle.py)
            _profiles = [profile]
            _weights  = np.array([1.0], dtype=np.float64)
        else:
            raise ValueError("train() requires either `profile` or `profiles`.")

        # K_ks[k] = rollouts allocated to profile k this training call.
        # Fixed for the entire call — no within-call JIT recompilation from
        # varying batch size.  Recompiles at most once per XDO iteration when
        # meta_strategy shifts enough to change any K_k.
        K_ks = _split_K(_weights, K)   # (N,) ints summing to K

        # Always start both buffers clean — prewarm (below) replaces the old
        # adv_retention carryover and avoids cross-profile contamination.
        self._adv_buf.partial_reset(retention=0.0)
        self._pol_buf.partial_reset(retention=0.0)

        # --- Pre-warm phase --------------------------------------------------
        # Run prewarm_iters rollouts against the mixture so the advantage net
        # has a stable prior before iteration 1.  Uses the same K_ks split so
        # the warm-up distribution matches the training distribution.
        # Data is discarded after one adv update — never enters the CFR average.
        if prewarm_iters > 0:
            _pw_pol_buf = ReservoirBuffer(1)   # throwaway — never read back
            for t_pw in range(1, prewarm_iters + 1):
                for prof, K_k in zip(_profiles, K_ks):
                    if K_k == 0:
                        continue
                    key, k_sim = jax.random.split(key)
                    sim_keys = jax.random.split(k_sim, K_k)
                    pw_results = run_K_episodes(
                        sim_keys,
                        self.adv_params,
                        self.gru_params,
                        prof.bc_model_params,
                        player_id=self.player_id,
                        partner_temp=self._partner_temp,
                        explore_eps=self._explore_eps,
                    )
                    collect_to_buffers(pw_results, self._adv_buf, _pw_pol_buf, float(t_pw))
                pw_results["score"].block_until_ready()   # sync after last profile
            # One adv update on prewarm data — initialises net without cold start.
            self._update_adv(prewarm_iters)
            # Discard: prewarm data must not bias the formal CFR average.
            self._adv_buf.partial_reset(retention=0.0)
            del _pw_pol_buf

        # Early-stopping probe: store raw buffer data, re-encode with current GRU
        # each check so the probe obs_h stays consistent with the current GRU.
        probe_raw:    Optional[tuple]        = None   # (obs_raw, p_acts, pfx_lens)
        probe_prev_q: Optional[jnp.ndarray] = None
        stable_count: int = 0
        self.last_cfr_iterations_run = cfr_iterations

        sim_secs = adv_secs = 0.0

        for t in range(1, cfr_iterations + 1):

            # -- Simulation --------------------------------------------------
            # Run K_k episodes against each profile and merge into shared buffers.
            # Profiles with K_k == 0 (weight too low to earn a rollout) are skipped.
            t0 = time.perf_counter()
            last_results = None
            for prof, K_k in zip(_profiles, K_ks):
                if K_k == 0:
                    continue
                key, k_sim = jax.random.split(key)
                sim_keys = jax.random.split(k_sim, K_k)
                last_results = run_K_episodes(
                    sim_keys,
                    self.adv_params,
                    self.gru_params,
                    prof.bc_model_params,
                    player_id=self.player_id,
                    partner_temp=self._partner_temp,
                    explore_eps=self._explore_eps,
                )
                collect_to_buffers(last_results, self._adv_buf, self._pol_buf, float(t))
            if last_results is not None:
                last_results["score"].block_until_ready()   # sync after last profile
            sim_secs += time.perf_counter() - t0

            # -- Advantage net update ----------------------------------------
            t0 = time.perf_counter()
            self._update_adv(t)
            adv_secs += time.perf_counter() - t0

            # -- Intermediate pol-net update ----------------------------------
            # pol_net is not used during CFR (adv_net drives simulation), so
            # updating it mid-loop is safe and has zero effect on the CFR signal.
            # Frequent updates let pol_net track the improving sigma incrementally
            # rather than fitting the entire history in one shot at the end.
            if pol_update_every > 0 and t % pol_update_every == 0:
                self._update_pol()

            # -- Early stopping (Q_adv stability) ----------------------------
            if t >= early_stop_min_iters and len(self._adv_buf) >= self._batch:
                if probe_raw is None:
                    batch = self._adv_buf.sample(self._batch)
                    # b[0]=obs_raw  b[1]=partner_acts  b[2]=prefix_len
                    probe_raw = (
                        jnp.array(np.stack([b[0] for b in batch], dtype=np.float32)),
                        jnp.array(np.stack([b[1] for b in batch], dtype=np.int32)),
                        jnp.array(np.array([b[2] for b in batch], dtype=np.int32)),
                    )
                # Re-encode with current GRU so probe stays GRU-consistent.
                obs_raw_p, p_acts_p, pfx_lens_p = probe_raw
                h_p = jax.vmap(
                    lambda a, l: _gru_enc.apply(self.gru_params, a, l)
                )(p_acts_p, pfx_lens_p)
                probe_obs_h = jnp.concatenate([obs_raw_p, h_p], axis=-1)
                preds = _adv_net.apply(self.adv_params, probe_obs_h)
                if probe_prev_q is not None:
                    delta = float(jnp.abs(preds - probe_prev_q).mean())
                    self.last_probe_delta = delta
                    if delta < early_stop_delta:
                        stable_count += 1
                        if stable_count >= early_stop_patience:
                            self.last_cfr_iterations_run = t
                            break
                    else:
                        stable_count = 0
                probe_prev_q = preds

        # -- GRU update: single pass after all CFR iterations (GRU frozen during loop)
        # Updating GRU inside the loop creates a non-stationary training target:
        # old buffer entries re-encode differently as the GRU drifts, causing D1
        # loss to slowly rise.  A single end-of-loop update is consistent with
        # published Deep CFR: the representation updates between outer XDO calls.
        t0 = time.perf_counter()
        self._update_gru()
        self.last_train_gru_secs = time.perf_counter() - t0

        # -- Final policy net update -----------------------------------------
        # Always do one final update after the loop regardless of pol_update_every
        # so the pol_net reflects the very last sigma (handles cases where CFR
        # stopped early or the last iteration wasn't a multiple of pol_update_every).
        t0 = time.perf_counter()
        self._update_pol()
        self.last_train_pol_secs = time.perf_counter() - t0

        self.last_train_sim_secs = sim_secs
        self.last_train_adv_secs = adv_secs

        # --- Mean positive regret (convergence diagnostic) -------------------
        # Sample a fresh batch from the adv buffer, re-encode with current GRU,
        # and compute mean positive regret from the final adv network state.
        # Near-zero → policy close to Nash for this partner; large → still
        # finding gains.  One forward pass — negligible overhead.
        if len(self._adv_buf) >= self._batch:
            reg_batch   = self._adv_buf.sample(self._batch)
            reg_obs_raw = jnp.array(np.stack([b[0] for b in reg_batch]), dtype=jnp.float32)
            reg_p_acts  = jnp.array(np.stack([b[1] for b in reg_batch]), dtype=jnp.int32)
            reg_pfx_len = jnp.array(np.array([b[2] for b in reg_batch]), dtype=jnp.int32)
            reg_h = jax.vmap(
                lambda a, l: _gru_enc.apply(self.gru_params, a, l)
            )(reg_p_acts, reg_pfx_len)
            reg_obs_h = jnp.concatenate([reg_obs_raw, reg_h], axis=-1)
            self.last_mean_regret = float(_compute_mean_regret(self.adv_params, reg_obs_h))

        return self.pol_params

    # ------------------------------------------------------------------
    # Private update helpers
    # ------------------------------------------------------------------

    def _update_adv(self, iteration_t: int) -> None:
        if len(self._adv_buf) < self._batch:
            return
        # Buffer format (7-tuple):
        #   b[0]=obs_raw  b[1]=partner_acts  b[2]=prefix_len
        #   b[3]=taken_action  b[4]=raw_score  b[5]=iter_t  b[6]=sigma_a
        all_obs_raw, all_p_acts, all_pfx_lens = [], [], []
        all_actions, all_w_scores, all_weights = [], [], []
        for _ in range(self._adv_steps):
            batch = self._adv_buf.sample(self._batch)
            all_obs_raw.append( np.stack([b[0] for b in batch]).astype(np.float32))
            all_p_acts.append(  np.stack([b[1] for b in batch]).astype(np.int32))
            all_pfx_lens.append(np.array([b[2] for b in batch], dtype=np.int32))
            all_actions.append( np.array([b[3] for b in batch], dtype=np.int32))
            raw_scores  = np.array([b[4] for b in batch], dtype=np.float32)
            iter_ts     = np.array([b[5] for b in batch], dtype=np.float32)
            sigma_a     = np.array([b[6] for b in batch], dtype=np.float32)
            # Clipped IS correction: min(score / σ(a*), IS_CLIP).
            # Corrects the bias that underestimates Q for rarely-sampled actions
            # without the extreme variance of uncapped IS weights.
            is_scores   = np.minimum(raw_scores / np.maximum(sigma_a, 1e-6), IS_CLIP)
            all_w_scores.append(is_scores)
            all_weights.append(iter_ts / iter_ts.sum())

        # Re-encode h_oppo with current GRU in one fused JIT call — fixes staleness.
        all_obs_h = _reenc_obs_h_batches(
            self.gru_params,
            jnp.array(np.stack(all_obs_raw)),    # (adv_steps, B, OBS_DIM)
            jnp.array(np.stack(all_p_acts)),     # (adv_steps, B, MAX_PARTNER_TURNS)
            jnp.array(np.stack(all_pfx_lens)),   # (adv_steps, B)
        )  # (adv_steps, B, IN_DIM)

        self.adv_params, self._adv_opt_state, adv_loss = self._adv_step(
            self.adv_params, self._adv_opt_state,
            all_obs_h,
            jnp.array(np.stack(all_actions)),    # (adv_steps, B)
            jnp.array(np.stack(all_w_scores)),   # (adv_steps, B)
            jnp.array(np.stack(all_weights)),    # (adv_steps, B)
        )
        self.last_adv_loss = float(adv_loss)

    def _update_gru(self) -> None:
        if len(self._adv_buf) < self._batch:
            return
        # Buffer format (7-tuple):
        #   b[0]=obs_raw  b[1]=partner_acts  b[2]=prefix_len
        #   b[3]=taken_action  b[4]=raw_score  b[5]=iter_t  b[6]=sigma_a
        all_obs_b, all_p_acts, all_pfx_lens = [], [], []
        all_actions, all_w_scores, all_weights = [], [], []
        for _ in range(self._gru_steps):
            batch = self._adv_buf.sample(self._batch)
            all_obs_b.append(   np.stack([b[0] for b in batch]).astype(np.float32))
            all_p_acts.append(  np.stack([b[1] for b in batch]).astype(np.int32))
            all_pfx_lens.append(np.array([b[2] for b in batch], dtype=np.int32))
            all_actions.append( np.array([b[3] for b in batch], dtype=np.int32))
            raw_scores  = np.array([b[4] for b in batch], dtype=np.float32)
            iter_ts     = np.array([b[5] for b in batch], dtype=np.float32)
            sigma_a     = np.array([b[6] for b in batch], dtype=np.float32)
            is_scores   = np.minimum(raw_scores / np.maximum(sigma_a, 1e-6), IS_CLIP)
            all_w_scores.append(is_scores)
            all_weights.append(iter_ts / iter_ts.sum())
        (self.adv_params, self.gru_params, self.aux_params,
         self._gru_opt_state, gru_loss) = self._gru_step(
            self.adv_params, self.gru_params, self.aux_params, self._gru_opt_state,
            jnp.array(np.stack(all_obs_b)),
            jnp.array(np.stack(all_p_acts)),
            jnp.array(np.stack(all_pfx_lens)),
            jnp.array(np.stack(all_actions)),
            jnp.array(np.stack(all_w_scores)),
            jnp.array(np.stack(all_weights)),
        )
        self.last_gru_loss = float(gru_loss)

    def _update_pol(self) -> None:
        if len(self._pol_buf) < self._batch:
            return
        # pol_buf stores 5-tuple: (obs_raw, partner_acts, prefix_len, sigma, iter_t).
        # Re-encode h_oppo with current GRU params so pol_net trains on representations
        # consistent with what it will see at inference time.
        all_obs_raw, all_p_acts, all_pfx_lens, all_sigma, all_weights = [], [], [], [], []
        for _ in range(self._pol_steps):
            batch = self._pol_buf.sample(self._batch)
            # b[0]=obs_raw  b[1]=partner_acts  b[2]=prefix_len  b[3]=sigma  b[4]=iter_t
            all_obs_raw.append( np.stack([b[0] for b in batch]).astype(np.float32))
            all_p_acts.append(  np.stack([b[1] for b in batch]).astype(np.int32))
            all_pfx_lens.append(np.array([b[2] for b in batch], dtype=np.int32))
            all_sigma.append(   np.stack([b[3] for b in batch]).astype(np.float32))
            iter_ts = np.array([b[4] for b in batch], dtype=np.float32)
            all_weights.append(iter_ts / iter_ts.sum())   # normalised linear weights

        all_obs_h = _reenc_obs_h_batches(
            self.gru_params,
            jnp.array(np.stack(all_obs_raw)),
            jnp.array(np.stack(all_p_acts)),
            jnp.array(np.stack(all_pfx_lens)),
        )  # (pol_steps, B, IN_DIM)

        self.pol_params, self._pol_opt_state, _ = self._pol_step(
            self.pol_params, self._pol_opt_state,
            all_obs_h,
            jnp.array(np.stack(all_sigma)),
            jnp.array(np.stack(all_weights)),   # (pol_steps, B)
        )
