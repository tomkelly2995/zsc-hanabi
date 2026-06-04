# src/xdo/xdo_rppo_solver_jax.py
# XDO outer-loop solver using RPPOAgentJax as the inner oracle.
#
# XDORPPOHanabiSolverJax mirrors XDOHanabiSolverJax but replaces the
# CooperativeODCFRAgentJax oracle with RPPOAgentJax.
#
# Verbatim copies from XDOHanabiSolverJax:
#   - __init__ structure (pools, meta_strategy, log_weights, probe_key, logging attrs)
#   - solve_meta_game()
#   - extract_behaviour_profile()
#   - Exploration floor logic
#
# RPPO-specific adaptations:
#   - __init__: instantiates RPPOAgentJax, stores RPPO hyperparams
#   - _probe_profile_score: uses run_K_rppo_probe_episodes + self.oracle.params
#   - request_new_policy: fresh RPPOAgentJax each call, PPO loop, returns RPPOPolicy

from __future__ import annotations

import time
import numpy as np
import jax
import jax.numpy as jnp

from src.jax_agents.rppo_agent_jax import (
    RPPOAgentJax,
    run_K_rppo_episodes,
    run_K_rppo_probe_episodes,
)
from src.population.behaviour_profile_jax import BehaviourProfileJax, _build_bc_pairs
from src.population.rppo_policy_jax import RPPOPolicy

_PROBE_EPISODES = 512   # episodes per probe — same as XDOHanabiSolverJax

_PLAYER_IDS      = {"agent_A": 0, "agent_B": 1}
_PARTNER_SOURCES = {"agent_A": "agent_B", "agent_B": "agent_A"}


class XDORPPOHanabiSolverJax:
    """
    Cooperative XDO outer-loop solver for one agent using RPPO as the oracle.

    One instance per agent.  The two agents never share data directly —
    the only cross-agent signal is the shared episode batch passed to
    extract_behaviour_profile() each iteration.

    Typical outer-loop usage:
        solver = XDORPPOHanabiSolverJax("agent_A")
        solver.extract_behaviour_profile(initial_batch)
        for each iteration:
            solver.solve_meta_game()
            policy = solver.request_new_policy(ppo_updates=1000)
            solver.extract_behaviour_profile(shared_batch)
    """

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 84,
        num_actions: int = 8,
        exploration_gamma: float = 0.15,
        log_weight_decay: float = 0.9575,
        partner_temperature: float = 0.3,
        # RPPO hyperparams
        lr: float = 3e-4,
        ent_coef: float = 0.05,
        vf_coef: float = 0.5,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 128,
        max_grad_norm: float = 0.5,
        n_episodes: int = 256,
        ppo_updates: int = 1000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        aux_loss_weight: float = 0.0,
        # Meta-game hyperparams
        rescore_topk: int = 3,
        rescore_full_every: int = 5,
        coupling_alpha: float = 0.3,
        seed: int = 0,
        device=None,   # jax.Device to pin this agent to; None → jax.devices()[0]
        # Latest-only oracle training
        temp_schedule: list | None = None,
        # None → constant partner_temperature; list cycles per iteration
        br_save_dir: str | None = None,
        # Directory to save per-iteration BR_k checkpoints; None → skip
    ):
        self.agent_id    = agent_id
        self.player_id   = _PLAYER_IDS[agent_id]
        self.partner_id  = 1 - self.player_id
        self.obs_dim     = obs_dim
        self.num_actions = num_actions
        self._device     = device if device is not None else jax.devices()[0]

        # Hyperparameters used in solve_meta_game / extract_behaviour_profile
        self._exploration_gamma  = exploration_gamma
        self._log_weight_decay   = log_weight_decay
        self._rescore_topk       = rescore_topk
        self._rescore_full_every = rescore_full_every
        self._coupling_alpha     = coupling_alpha

        # RPPO training hyperparams — stored for request_new_policy
        self._lr             = lr
        self._ent_coef       = ent_coef
        self._vf_coef        = vf_coef
        self._clip_eps       = clip_eps
        self._ppo_epochs     = ppo_epochs
        self._minibatch_size = minibatch_size
        self._max_grad_norm  = max_grad_norm
        self._n_episodes     = n_episodes
        self._ppo_updates    = ppo_updates
        self._gamma          = gamma
        self._gae_lambda     = gae_lambda
        self._aux_loss_weight = aux_loss_weight
        self._partner_temp   = partner_temperature

        # Temperature schedule: cycles across oracle calls to diversify BC profiles.
        # Each oracle trains against a partner with a different softmax temperature,
        # making profiles behaviourally distinct (useful for log-likelihood inference).
        self._temp_schedule = list(temp_schedule) if temp_schedule else [partner_temperature]
        self._br_save_dir   = br_save_dir

        # Profile / policy pools and meta-game state
        self.profile_pool:    list       = []
        self.policy_pool:     list       = []
        self._profile_scores: list       = []
        self._log_weights:    np.ndarray = np.array([], dtype=np.float64)
        self.meta_strategy:   np.ndarray = np.array([], dtype=np.float64)

        # Build initial oracle — will be replaced fresh on each request_new_policy call.
        # Pin to self._device so probe calls always dispatch on the right GPU.
        self.oracle = RPPOAgentJax(
            agent_id=agent_id,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            ppo_epochs=ppo_epochs,
            minibatch_size=minibatch_size,
            max_grad_norm=max_grad_norm,
            partner_temperature=partner_temperature,
            aux_loss_weight=aux_loss_weight,
            seed=seed,
            device=self._device,
        )

        # Iteration counters
        self._iteration: int = 0
        self._meta_game_count: int = 0

        # Probe key pinned to this agent's device so probe dispatches there.
        self._probe_key = jax.device_put(
            jax.random.PRNGKey(seed + 99999), self._device
        )

        # RNG for generating fresh seeds each request_new_policy call
        self._rng = np.random.default_rng(seed)

        # Logging outputs (mirrors XDOHanabiSolverJax)
        self.last_bc_loss:     float = 0.0
        self.last_bc_accuracy: float = 0.0
        self.last_probe_score: float = 0.0
        # RPPO-specific training metrics
        self.last_train_secs:  float = float("nan")
        self.last_actor_loss:  float = float("nan")
        self.last_value_loss:  float = float("nan")
        self.last_entropy:     float = float("nan")
        self.last_clip_frac:   float = float("nan")
        self.last_mean_score:  float = float("nan")

    # ------------------------------------------------------------------
    # 1. Meta-game solver — copied verbatim from XDOHanabiSolverJax
    # ------------------------------------------------------------------

    def solve_meta_game(self, partner_scores=None) -> None:
        """
        Update MetaStrategy via multiplicative-weights swap-regret minimiser.
        η = sqrt(log(N) / T).  Does nothing for pools of size 0 or 1.

        Args:
            partner_scores : optional list/array of the *other* agent's raw probe
                scores (one per profile, same index order).  When provided, each
                agent's own score for profile k is blended with the partner's
                score before the meta-game update:

                    blended[k] = (1 - α) * own[k] + α * partner[k]

                where α = self._coupling_alpha (default 0.3).

        Re-scoring strategy (two complementary mechanisms):

        1. Periodic full rescore (rescore_full_every > 0):
           Every rescore_full_every outer iterations, ALL profiles are re-scored.

        2. Top-k partial rescore (rescore_topk > 0):
           On non-full-rescore iterations, re-score the top-k highest-weight profiles.
        """
        N = len(self.profile_pool)
        if N == 0:
            return

        do_full_rescore = (
            self._rescore_full_every > 0
            and self._iteration > 0
            and self._iteration % self._rescore_full_every == 0
        )

        if do_full_rescore:
            for idx in range(N):
                self._profile_scores[int(idx)] = self._probe_profile_score(
                    self.profile_pool[int(idx)]
                )
        elif self._rescore_topk > 0:
            k = min(self._rescore_topk, N)
            if len(self.meta_strategy) == N and N > 1:
                top_indices = np.argsort(self.meta_strategy)[-k:]
            else:
                top_indices = np.arange(N)
            for idx in top_indices:
                self._profile_scores[int(idx)] = self._probe_profile_score(
                    self.profile_pool[int(idx)]
                )

        if N == 1:
            self._log_weights = np.array([0.0])
            self.meta_strategy = np.array([1.0])
            self._meta_game_count += 1
            return

        self._meta_game_count += 1
        T = self._meta_game_count
        eta = np.sqrt(np.log(N) / T)

        scores = np.array(self._profile_scores, dtype=np.float64)

        if (partner_scores is not None
                and self._coupling_alpha > 0.0
                and len(partner_scores) > 0):
            n_overlap = min(N, len(partner_scores))
            p = np.array(partner_scores[:n_overlap], dtype=np.float64)
            scores[:n_overlap] = (
                (1.0 - self._coupling_alpha) * scores[:n_overlap]
                + self._coupling_alpha * p
            )

        G    = np.maximum(0.0, scores[np.newaxis, :] - scores[:, np.newaxis])
        gain = G.sum(axis=0)

        if self._log_weight_decay < 1.0:
            self._log_weights *= self._log_weight_decay

        self._log_weights += eta * gain

        log_w_shifted = self._log_weights - self._log_weights.max()
        w = np.exp(log_w_shifted)
        self.meta_strategy = w / w.sum()

        floor = self._exploration_gamma / N
        for _ in range(10):
            self.meta_strategy = np.maximum(self.meta_strategy, floor)
            self.meta_strategy /= self.meta_strategy.sum()
            if self.meta_strategy.min() >= floor - 1e-10:
                break

    # ------------------------------------------------------------------
    # 1b. Probe evaluation — RPPO-specific
    # ------------------------------------------------------------------

    def _probe_profile_score(self, profile: BehaviourProfileJax) -> float:
        """
        Estimate how well the current oracle policy coordinates with the
        given partner BehaviourProfile, by running _PROBE_EPISODES episodes.

        Uses run_K_rppo_probe_episodes with self.oracle.params (RPPOActorCritic).
        """
        self._probe_key, k = jax.random.split(self._probe_key)
        keys      = jax.device_put(jax.random.split(k, _PROBE_EPISODES), self._device)
        bc_params = jax.device_put(profile.bc_model_params, self._device)
        scores = run_K_rppo_probe_episodes(
            keys,
            self.oracle.params,
            bc_params,
            self.player_id,
            self._partner_temp,
        )
        return float(jnp.mean(scores))

    # ------------------------------------------------------------------
    # 2. Request new policy from RPPO oracle
    # ------------------------------------------------------------------

    def request_new_policy(
        self,
        ppo_updates: int | None = None,
        n_episodes: int | None = None,
    ) -> RPPOPolicy:
        """
        Instantiate a fresh RPPOAgentJax, run the PPO training loop, and
        return a frozen RPPOPolicy snapshot.

        Fresh start every call: weights are randomly re-initialised so each
        XDO iteration's oracle is unbiased by previous training.

        Args:
            ppo_updates : number of collect+update iterations (default: self._ppo_updates)
            n_episodes  : episodes collected per update (default: self._n_episodes)

        Returns:
            RPPOPolicy — frozen snapshot of ac_params for this agent's player_id.

        Raises:
            RuntimeError: if profile_pool is empty.
        """
        if not self.profile_pool:
            raise RuntimeError(
                "profile_pool is empty — call extract_behaviour_profile() first."
            )

        ppo_updates = ppo_updates if ppo_updates is not None else self._ppo_updates
        n_episodes  = n_episodes  if n_episodes  is not None else self._n_episodes

        # Fresh seed from the stored RNG so each call is independent.
        fresh_seed = int(self._rng.integers(0, 2**31))

        agent = RPPOAgentJax(
            agent_id=self.agent_id,
            lr=self._lr,
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            clip_eps=self._clip_eps,
            ent_coef=self._ent_coef,
            vf_coef=self._vf_coef,
            ppo_epochs=self._ppo_epochs,
            minibatch_size=self._minibatch_size,
            max_grad_norm=self._max_grad_norm,
            partner_temperature=self._partner_temp,
            seed=fresh_seed,
            device=self._device,
        )

        key = jax.device_put(
            jax.random.PRNGKey(fresh_seed ^ 0xDEADBEEF), self._device
        )

        # ── Latest-only profile selection ────────────────────────────────────
        # Always train the oracle against the most recently extracted BC profile
        # rather than sampling from the meta-strategy mixture.  This gives a
        # cleaner, progressive training signal: each oracle best-responds to the
        # strongest observed convention so far.  The meta-strategy is still
        # updated each iteration for logging and coupling_alpha alignment.
        profile_idx = len(self.profile_pool) - 1
        profile     = self.profile_pool[profile_idx]

        # ── Temperature schedule ─────────────────────────────────────────────
        # Cycle through configured temperatures so consecutive BC profiles see
        # different levels of partner stochasticity.  Low temperature → sharp,
        # deterministic partner; high temperature → exploratory partner.
        # Different temperatures produce behaviourally distinct BC profiles,
        # which is required for log-likelihood inference to be discriminative.
        temp = self._temp_schedule[profile_idx % len(self._temp_schedule)]

        bc_params = jax.device_put(profile.bc_model_params, self._device)

        # Pre-compute the padded batch size (power-of-2 rounding used by
        # collect_episodes).  This stays constant throughout training so
        # _fused_update_fn is compiled exactly once.
        K_padded = 1
        while K_padded < n_episodes:
            K_padded <<= 1
        agent._ensure_fused_update(K_padded)

        t0 = time.perf_counter()
        for _ in range(ppo_updates):
            key, k_ep, k_upd = jax.random.split(key, 3)

            # ── Collect K_padded episodes in one JIT call ─────────────────
            ep_keys = jax.device_put(
                jax.random.split(k_ep, K_padded), self._device
            )
            results = run_K_rppo_episodes(
                ep_keys, agent.params, bc_params,
                self.player_id, temp,
            )
            # Trim padding back to n_episodes on the host before the fused
            # update so that the fused function sees its compiled batch size.
            if K_padded > n_episodes:
                results = {k: v[:n_episodes] for k, v in results.items()}
                agent._ensure_fused_update(n_episodes)

            agent.last_mean_score = float(jnp.mean(results["score"]))

            # ── Fused GAE + all PPO epochs in one JIT call ─────────────────
            agent.update_fused(results, k_upd)

        self.last_train_secs = time.perf_counter() - t0

        # Update oracle so _probe_profile_score uses fresh weights
        self.oracle = agent

        # Store final training stats
        self.last_actor_loss = agent.last_actor_loss
        self.last_value_loss = agent.last_value_loss
        self.last_entropy    = agent.last_entropy
        self.last_clip_frac  = agent.last_clip_frac
        self.last_mean_score = agent.last_mean_score

        # ── Save BR_k checkpoint ─────────────────────────────────────────────
        # BR_k = oracle trained against profile_k.  Saved before
        # extract_behaviour_profile() appends the next profile so that the
        # index matches: br{k:03d} was the best-response to profile_pool[k].
        if self._br_save_dir is not None:
            from src.utils.checkpointing import save_br
            br_path = save_br(self, self._br_save_dir, self.agent_id, profile_idx)
            print(f"  [br] saved → {br_path}  (temp={temp:.2f})")

        snapshot = RPPOPolicy(ac_params=agent.params, player_id=self.player_id)
        self.policy_pool.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # 3. Extract BehaviourProfile — copied verbatim from XDOHanabiSolverJax
    # ------------------------------------------------------------------

    def extract_behaviour_profile(
        self,
        episodes: list,
        bc_epochs: int = 100,
        bc_rounds: int = 3,
        seed: int = 0,
    ) -> BehaviourProfileJax:
        """
        Build a BehaviourProfileJax from a shared episode batch.

        Reads only the *partner's* actions and public observations.
        Never reads this agent's observations or policy weights.

        Args:
            episodes  : list of episode dicts with keys:
                          "steps" : list of {"current_player": int,
                                             "player_obs": {0: array, 1: array},
                                             "action": int}
                          "score" : int/float — final team score
            bc_epochs : supervised training epochs for the BC partner model.
            bc_rounds : iterative teacher-forcing rounds (default 3).
            seed      : RNG seed for BC model initialisation.

        Returns:
            The newly created BehaviourProfileJax (also appended to profile_pool).
        """
        raw_sequences:     list = []
        raw_obs_sequences: list = []
        raw_oracle_nexts:  list = []

        for episode in episodes:
            ep_actions:      list = []
            ep_obs:          list = []
            ep_oracle_nexts: list = []
            steps = episode["steps"]
            for i, step in enumerate(steps):
                if step["current_player"] == self.partner_id:
                    ep_actions.append(step["action"])
                    ep_obs.append(step["player_obs"][self.partner_id])
                    oracle_next = -1
                    for j in range(i + 1, len(steps)):
                        if steps[j]["current_player"] == self.player_id:
                            oracle_next = steps[j]["action"]
                            break
                    ep_oracle_nexts.append(oracle_next)
            if ep_actions:
                raw_sequences.append(ep_actions)
                raw_obs_sequences.append(ep_obs)
                raw_oracle_nexts.append(ep_oracle_nexts)

        if not raw_sequences:
            raw_sequences     = [[0]]
            raw_obs_sequences = [[np.zeros(self.obs_dim, dtype=np.float32)]]
            raw_oracle_nexts  = [[-1]]

        profile = BehaviourProfileJax.from_episodes(
            raw_sequences=raw_sequences,
            raw_obs_sequences=raw_obs_sequences,
            raw_oracle_nexts=raw_oracle_nexts,
            iteration_created=self._iteration,
            source=_PARTNER_SOURCES[self.agent_id],
            bc_epochs=bc_epochs,
            bc_rounds=bc_rounds,
            seed=seed,
        )

        eval_pairs = _build_bc_pairs(
            raw_sequences, raw_obs_sequences, raw_oracle_nexts,
            profile.bc_model.net, profile.bc_model.params,
        )
        self.last_bc_loss = float(profile.bc_train_loss)
        acc, _ = profile.bc_model.evaluate(eval_pairs)
        self.last_bc_accuracy = float(acc)

        probe_score = self._probe_profile_score(profile)
        self.last_probe_score = probe_score

        self.profile_pool.append(profile)
        self._profile_scores.append(probe_score)

        N = len(self.profile_pool)
        self._log_weights = np.append(self._log_weights, 0.0)
        log_w_shifted = self._log_weights - self._log_weights.max()
        w = np.exp(log_w_shifted)
        self.meta_strategy = w / w.sum()

        floor = self._exploration_gamma / N
        for _ in range(10):
            self.meta_strategy = np.maximum(self.meta_strategy, floor)
            self.meta_strategy /= self.meta_strategy.sum()
            if self.meta_strategy.min() >= floor - 1e-10:
                break

        self._iteration += 1
        return profile
