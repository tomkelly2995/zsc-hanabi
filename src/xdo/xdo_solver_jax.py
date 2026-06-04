# src/xdo/xdo_solver_jax.py
# JAX port of XDOHanabiSolver — cooperative XDO outer loop.
#
# Differences from the PyTorch version:
#   • BehaviourProfileJax  : no cached_h_oppo (GRU encodes on-the-fly in sim)
#   • HanabiPolicyJax      : holds Flax param dicts instead of state_dicts
#   • request_new_policy() : returns HanabiPolicyJax; oracle.train() returns pol_params
#   • No lazy GRU re-encoding (JAX simulation re-encodes partner actions each CFR iter)
#   • No PyTorch dependency anywhere in this file
#
# No-info-sharing guarantee: extract_behaviour_profile() reads only the
# partner's actions and public observations. Policy weights are never passed
# across agents.

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.jax_agents.cooperative_odcfr_agent_jax import CooperativeODCFRAgentJax
from src.jax_agents.simulation import run_K_probe_episodes
from src.population.behaviour_profile_jax import BehaviourProfileJax, _build_bc_pairs
from src.population.hanabi_policy_jax import HanabiPolicyJax

_PROBE_EPISODES = 512   # episodes per probe — enough signal, negligible cost

_PLAYER_IDS = {"agent_A": 0, "agent_B": 1}
_PARTNER_SOURCES = {"agent_A": "agent_B", "agent_B": "agent_A"}


class XDOHanabiSolverJax:
    """
    Cooperative XDO outer-loop solver for one agent (JAX/Flax version).

    One instance per agent. The two agents never share data directly —
    the only cross-agent signal is the shared episode batch passed to
    extract_behaviour_profile() each iteration.

    Typical outer-loop usage:
        solver = XDOHanabiSolverJax("agent_A")
        solver.extract_behaviour_profile(initial_batch)
        for each iteration:
            solver.solve_meta_game()
            policy = solver.request_new_policy(cfr_iterations=T)
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
        adv_train_steps: int = 100,
        gru_train_steps: int = 10,
        pol_train_steps: int = 200,
        K_simulations: int = 16,
        aux_loss_weight: float = 0.2,
        gru_lr: float = 5e-5,        # forwarded to CooperativeODCFRAgentJax
        gru_grad_clip: float = 0.3,  # forwarded to CooperativeODCFRAgentJax
        explore_eps: float = 0.06,   # forwarded to CooperativeODCFRAgentJax
        rescore_topk: int = 3,       # profiles to re-score per solve_meta_game call
        rescore_full_every: int = 5, # re-score ALL profiles every N outer iterations
        coupling_alpha: float = 0.3, # blend weight for partner scores in meta-game
        seed: int = 0,
    ):
        self.agent_id   = agent_id
        self.player_id  = _PLAYER_IDS[agent_id]
        self.partner_id = 1 - self.player_id
        self.obs_dim    = obs_dim
        self.num_actions = num_actions

        # Hyperparameters used in solve_meta_game / extract_behaviour_profile
        self._exploration_gamma   = exploration_gamma
        self._log_weight_decay    = log_weight_decay
        self._rescore_topk        = rescore_topk
        self._rescore_full_every  = rescore_full_every
        self._coupling_alpha      = coupling_alpha

        # Profile / policy pools and meta-game state
        self.profile_pool:    list            = []
        self.policy_pool:     list            = []
        self._profile_scores: list            = []   # one probe score per profile
        self._log_weights:    np.ndarray      = np.array([], dtype=np.float64)
        self.meta_strategy:   np.ndarray      = np.array([], dtype=np.float64)

        self.oracle = CooperativeODCFRAgentJax(
            agent_id=agent_id,
            K_simulations=K_simulations,
            partner_temperature=partner_temperature,
            adv_train_steps=adv_train_steps,
            gru_train_steps=gru_train_steps,
            pol_train_steps=pol_train_steps,
            aux_loss_weight=aux_loss_weight,
            gru_lr=gru_lr,
            gru_grad_clip=gru_grad_clip,
            explore_eps=explore_eps,
            seed=seed,
        )

        # ----- Iteration counters --------------------------------------------
        self._iteration: int = 0
        self._meta_game_count: int = 0

        # ----- Probe key (separate stream from oracle seed) ------------------
        self._probe_key = jax.random.PRNGKey(seed + 99999)

        # ----- Logging outputs -----------------------------------------------
        self.last_bc_loss: float = 0.0
        self.last_bc_accuracy: float = 0.0
        self.last_probe_score: float = 0.0

    # ------------------------------------------------------------------
    # 1. Meta-game solver (identical algorithm to PyTorch version)
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

                where α = self._coupling_alpha (default 0.5).

                This bounds L1 divergence between the two agents' meta-strategies
                while still letting them differ based on role asymmetry
                (player 0 vs player 1 have genuinely different probe scores).

                If len(partner_scores) < N (e.g. partner just added a new profile
                after its own solve_meta_game call), blending is applied only for
                the overlapping prefix; remaining own scores are used as-is.

                Set coupling_alpha=0.0 or pass partner_scores=None to disable.

        Re-scoring strategy (two complementary mechanisms):

        1. Periodic full rescore (rescore_full_every > 0):
           Every rescore_full_every outer iterations, ALL profiles are re-scored
           with the oracle's current policy.  This ensures low-weight profiles
           that the oracle has since learned to coordinate with can rise back into
           the meta-strategy mixture, countering convention pool stagnation.

        2. Top-k partial rescore (rescore_topk > 0):
           On non-full-rescore iterations, re-score the top-k highest-weight
           profiles.  High-weight profiles drive the training distribution, so
           keeping their scores fresh directly improves CFR training quality.
           Profiles at the exploration floor (γ/N) are skipped on partial
           rescores — they contribute negligible rollouts regardless of score.

        The two mechanisms are mutually exclusive per call: a full-rescore
        iteration skips the top-k pass (all scores are already fresh).
        """
        N = len(self.profile_pool)
        if N == 0:
            return

        # --- Determine rescore mode for this iteration -----------------------
        do_full_rescore = (
            self._rescore_full_every > 0
            and self._iteration > 0
            and self._iteration % self._rescore_full_every == 0
        )

        if do_full_rescore:
            # Full rescore: refresh every profile score with current pol_params.
            # Allows low-weight / stale profiles to resurface if the oracle has
            # matured enough to coordinate with them.
            for idx in range(N):
                self._profile_scores[int(idx)] = self._probe_profile_score(
                    self.profile_pool[int(idx)]
                )

        elif self._rescore_topk > 0:
            # Partial rescore: top-k by current meta-strategy weight.
            #   N == 1 : only one profile, always re-score it.
            #   N  > 1 : pick the top-k — these drive the training distribution.
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

        # Own probe scores — never modified in-place; blending is local only.
        scores = np.array(self._profile_scores, dtype=np.float64)   # (N,)

        # Partner-score blending: mix own and partner scores before the
        # meta-game update so divergence is bounded by role asymmetry, not
        # by unconstrained multiplicative-weights amplification.
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
        gain = G.sum(axis=0)                       # column sums

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
    # 1b. Probe evaluation — agent-specific profile scoring
    # ------------------------------------------------------------------

    def _probe_profile_score(self, profile: BehaviourProfileJax) -> float:
        """
        Estimate how well this agent's current policy coordinates with the
        given partner BehaviourProfile, by running _PROBE_EPISODES episodes
        with the oracle's pol_params vs the profile's BC model.

        This gives each agent an independent score signal so their
        meta-strategies can diverge — unlike the shared mean team score from
        the episode batch, which is identical for both agents and forces
        L1(meta_A, meta_B) = 0 forever.

        Called immediately after BehaviourProfileJax is built in
        extract_behaviour_profile().  The oracle's pol_params reflect the
        most recently trained policy (from the preceding request_new_policy
        call); on the very first call they are randomly initialised, giving a
        low but valid baseline score.

        Returns:
            float — mean team score over _PROBE_EPISODES episodes.
        """
        self._probe_key, k = jax.random.split(self._probe_key)
        keys = jax.random.split(k, _PROBE_EPISODES)
        scores = run_K_probe_episodes(
            keys,
            self.oracle.pol_params,
            self.oracle.gru_params,
            profile.bc_model_params,
            self.player_id,
            self.oracle._partner_temp,
        )
        return float(jnp.mean(scores))

    # ------------------------------------------------------------------
    # 2. Request new policy from oracle
    # ------------------------------------------------------------------

    def request_new_policy(
        self,
        cfr_iterations: int = 100,
        k_simulations: int | None = None,
        early_stop_min_iters: int = 250,
        early_stop_delta: float = 1e-2,
        partner_meta_strategy: np.ndarray | None = None,
    ) -> HanabiPolicyJax:
        """
        Sample a BehaviourProfile, run ODCFR oracle, snapshot into HanabiPolicyJax.

        Args:
            cfr_iterations       : CFR traversal budget for the oracle.
            k_simulations        : rollouts per CFR iteration (uses oracle default if None).
            early_stop_min_iters : minimum CFR iters before early stopping is checked.
            early_stop_delta     : Q-value convergence threshold for early stopping.
            partner_meta_strategy: if provided, use the partner's meta-strategy weights
                                   to split K rollouts across profile_pool rather than
                                   self.meta_strategy.  This trains the oracle to
                                   best-respond to what the partner is *actually playing*
                                   rather than what this agent thinks is optimal.
                                   Must have the same length as self.profile_pool.
                                   Falls back to self.meta_strategy if None.

        Returns:
            HanabiPolicyJax — frozen snapshot of the newly trained policy.

        Raises:
            RuntimeError: if profile_pool is empty.
        """
        if not self.profile_pool:
            raise RuntimeError(
                "profile_pool is empty — call extract_behaviour_profile() first."
            )

        # Determine which weights to use for splitting K rollouts.
        # partner_meta_strategy: oracle trains to best-respond to the partner's
        # actual current mixture — more correct than using own meta-strategy.
        # Falls back to self.meta_strategy if partner weights not provided.
        if partner_meta_strategy is not None:
            n = len(self.profile_pool)
            w = np.asarray(partner_meta_strategy[:n], dtype=np.float64)
            train_weights = w / w.sum()
        else:
            train_weights = self.meta_strategy

        pol_params = self.oracle.train(
            profiles=self.profile_pool,
            meta_strategy=train_weights,
            cfr_iterations=cfr_iterations,
            k_simulations=k_simulations,
            early_stop_min_iters=early_stop_min_iters,
            early_stop_delta=early_stop_delta,
        )

        # Snapshot pol_params + gru_params into a frozen HanabiPolicyJax
        snapshot = HanabiPolicyJax(
            pol_params=pol_params,
            gru_params=self.oracle.gru_params,
        )
        self.policy_pool.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # 3. Extract BehaviourProfile from shared episode batch
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
        raw_sequences:    list = []
        raw_obs_sequences: list = []
        raw_oracle_nexts:  list = []   # oracle's next action after each BC turn

        for episode in episodes:
            ep_actions:      list = []
            ep_obs:          list = []
            ep_oracle_nexts: list = []
            steps = episode["steps"]
            for i, step in enumerate(steps):
                if step["current_player"] == self.partner_id:
                    ep_actions.append(step["action"])
                    ep_obs.append(step["player_obs"][self.partner_id])
                    # Find oracle's next action after this BC turn
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

        # BC quality metrics — teacher-forced h_bc so the GRU memory reflects
        # actual episode history at each decision point, not a zero baseline.
        eval_pairs = _build_bc_pairs(
            raw_sequences, raw_obs_sequences, raw_oracle_nexts,
            profile.bc_model.net, profile.bc_model.params,
        )
        self.last_bc_loss = float(profile.bc_train_loss)
        acc, _ = profile.bc_model.evaluate(eval_pairs)
        self.last_bc_accuracy = float(acc)

        # Probe score: how well does THIS agent's current policy coordinate
        # with the new profile?  Agent-specific — breaks the symmetry that
        # caused both agents' meta-strategies to be always identical.
        probe_score = self._probe_profile_score(profile)
        self.last_probe_score = probe_score

        # Update pool and meta-strategy weights
        self.profile_pool.append(profile)
        self._profile_scores.append(probe_score)

        N = len(self.profile_pool)
        self._log_weights = np.append(self._log_weights, 0.0)
        log_w_shifted = self._log_weights - self._log_weights.max()
        w = np.exp(log_w_shifted)
        self.meta_strategy = w / w.sum()

        # Apply exploration floor — keeps meta_strategy consistent with the
        # floored version produced by solve_meta_game().  Without this,
        # meta_strategy is overwritten with an unfloored distribution every
        # iteration, allowing it to concentrate to [0,…,1,…,0] and making
        # the L1 divergence metric meaningless (reaches 2.0 exactly).
        floor = self._exploration_gamma / N
        for _ in range(10):
            self.meta_strategy = np.maximum(self.meta_strategy, floor)
            self.meta_strategy /= self.meta_strategy.sum()
            if self.meta_strategy.min() >= floor - 1e-10:
                break

        self._iteration += 1
        return profile
