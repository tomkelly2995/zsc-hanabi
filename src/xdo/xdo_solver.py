# src/xdo/xdo_solver.py
# XDOHanabiSolver — cooperative XDO outer loop, one instance per agent.
# Maintains BehaviourProfilePool, MetaStrategy (swap regret), ScoreMatrix,
# and owns the CooperativeODCFRAgent oracle.
#
# Episode batch format expected by extract_behaviour_profile():
#   List of episode dicts, each with:
#     {
#       'steps': [
#           {
#             'current_player': int,
#             'player_obs': {0: np.ndarray(84,), 1: np.ndarray(84,)},
#             'action': int,
#           },
#           ...
#       ],
#       'score': int,   # final team score for this episode
#     }
#
# No-info-sharing guarantee: extract_behaviour_profile() reads only the
# partner's actions and public observations from the episode batch.
# request_new_policy() never passes policy weights to the oracle.

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.cooperative_odcfr_agent import CooperativeODCFRAgent
from src.population.behaviour_profile import BehaviourProfile, _build_obs_action_pairs
from src.population.hanabi_policy import HanabiPolicy

_PLAYER_IDS = {"agent_A": 0, "agent_B": 1}
_PARTNER_SOURCES = {"agent_A": "agent_B", "agent_B": "agent_A"}


class XDOHanabiSolver:
    """
    Cooperative XDO outer-loop solver for one agent.

    One instance is created for Agent A and one for Agent B. They never
    share data directly — the only cross-agent signal is the shared episode
    batch passed into extract_behaviour_profile() each iteration.

    Typical outer-loop usage (see Section 6.2):
        solver = XDOHanabiSolver("agent_A")
        # --- initialisation ---
        solver.extract_behaviour_profile(initial_batch)
        # --- per-iteration ---
        for each iteration:
            solver.solve_meta_game()
            policy = solver.request_new_policy(cfr_iterations=T)
            drift  = solver.last_embedding_drift          # log to Tensorboard
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
        device: torch.device | None = None,
    ):
        self.agent_id = agent_id
        self.player_id = _PLAYER_IDS[agent_id]
        self.partner_id = 1 - self.player_id
        self.obs_dim = obs_dim
        self.num_actions = num_actions

        # ----- Core data structures ----------------------------------------
        # BehaviourProfiles observed from the partner, one per XDO iteration
        self.profile_pool: list = []
        # Frozen HanabiPolicy snapshots, one per XDO iteration
        self.policy_pool: list = []
        # Per-profile mean team score (diagonal of the conceptual ScoreMatrix)
        self._profile_scores: list = []
        # Log-space weights for the multiplicative-weights update.
        # Stored as log(w) to prevent float64 overflow/underflow after many
        # iterations. Updated by addition: log_w += eta * gain.
        # Normalised via log-sum-exp when converting to meta_strategy.
        self._log_weights: np.ndarray = np.array([], dtype=np.float64)
        # Normalised MetaStrategy distribution
        self.meta_strategy: np.ndarray = np.array([], dtype=np.float64)

        # Minimum per-profile sampling probability for the PRD exploration floor.
        # Each profile is guaranteed at least γ/N weight after every MetaStrategy
        # update, preventing complete concentration on a single convention and
        # ensuring late-discovered profiles can always compete.
        # Based on PSRO projected replicator dynamics (Lanctot et al., 2017).
        self._exploration_gamma: float = exploration_gamma
        # Multiplicative-weights decay factor applied to log-weights each
        # iteration before adding new gains.  Values < 1.0 cause old evidence
        # to fade, making the MetaStrategy more responsive to newly discovered
        # conventions and reducing inertia. 1.0 = no decay (original behaviour).
        self._log_weight_decay: float = log_weight_decay

        # ----- Oracle ---------------------------------------------------------
        self.oracle = CooperativeODCFRAgent(
            agent_id=agent_id,
            partner_temperature=partner_temperature,
            adv_train_steps=adv_train_steps,
            device=device,
        )

        # ----- Iteration counters --------------------------------------------
        self._iteration: int = 0           # number of profiles added
        self._meta_game_count: int = 0     # number of solve_meta_game() calls (= T)
        self._gru_train_count: int = 0     # number of oracle.train() calls

        # ----- Logging outputs (read by main loop) ---------------------------
        self.last_embedding_drift: float = 0.0
        # BC model quality for the most recently added profile
        self.last_bc_loss: float = 0.0
        self.last_bc_accuracy: float = 0.0
        # Mean pairwise cosine distance between all profile embeddings.
        # 0.0 = all embeddings identical; 1.0 = maximally spread.
        self.last_embedding_spread: float = 0.0

    # ------------------------------------------------------------------
    # 1. Meta-game solver
    # ------------------------------------------------------------------

    def solve_meta_game(self) -> None:
        """
        Update MetaStrategy via multiplicative-weights swap-regret minimiser.

        Constructs the swap gains matrix G over current profile scores, then
        updates unnormalised weights multiplicatively and renormalises.

        η = sqrt(log(N) / T)  — standard no-regret learning-rate schedule.

        Does nothing if the pool has fewer than 2 profiles.
        """
        N = len(self.profile_pool)
        if N == 0:
            return
        if N == 1:
            self._log_weights = np.array([0.0])
            self.meta_strategy = np.array([1.0])
            self._meta_game_count += 1
            return

        self._meta_game_count += 1
        T = self._meta_game_count
        eta = np.sqrt(np.log(N) / T)

        scores = np.array(self._profile_scores)           # (N,)
        # G[i, j] = max(0, score_j - score_i)
        # G[i, j] = max(0, Score(j) - Score(i))  — spec Section 2.2.
        # gain[j] = Σ_i G[i, j]: total advantage of profile j over all others.
        # Column sums → high gain for BETTER profiles → their weights increase.
        # (Row sums would measure "how easy is it to leave profile i", giving
        # higher gain to worse profiles and reversing the convergence direction.)
        G = np.maximum(0.0, scores[np.newaxis, :] - scores[:, np.newaxis])  # (N, N)
        gain = G.sum(axis=0)                              # (N,) — column sums

        # Decay old log-weights before adding new gains.  This fades past
        # evidence so recently discovered conventions can overcome accumulated
        # weight from earlier (possibly worse) profiles, reducing inertia.
        if self._log_weight_decay < 1.0:
            self._log_weights *= self._log_weight_decay

        # Update in log-space: equivalent to w *= exp(eta * gain) but safe
        # against float64 overflow/underflow after many iterations.
        self._log_weights += eta * gain

        # Normalise via log-sum-exp for numerical stability.
        log_w_shifted = self._log_weights - self._log_weights.max()
        w = np.exp(log_w_shifted)
        self.meta_strategy = w / w.sum()

        # PRD exploration floor: every profile retains at least γ/N weight so
        # late-discovered conventions are never fully excluded from sampling.
        # A single clamp+renorm isn't sufficient — renormalising redistributes
        # weight and can push clamped values back below the floor. Iterating
        # converges in 2–3 steps in practice.
        floor = self._exploration_gamma / N
        for _ in range(10):
            self.meta_strategy = np.maximum(self.meta_strategy, floor)
            self.meta_strategy /= self.meta_strategy.sum()
            if self.meta_strategy.min() >= floor - 1e-10:
                break

    # ------------------------------------------------------------------
    # 2. Request new policy from oracle
    # ------------------------------------------------------------------

    def request_new_policy(
        self,
        cfr_iterations: int = 100,
        k_simulations: int | None = None,
        gru_update_every: int = 5,
        early_stop_min_iters: int = 250,
        early_stop_delta: float = 1e-2,
    ) -> HanabiPolicy:
        """
        Sample a BehaviourProfile, optionally reencode (lazy), run the
        ODCFR oracle, snapshot the result into a HanabiPolicy.

        Steps (matching Section 6.2, steps 2–3–4–6):
          2. Sample profile ~ MetaStrategy. Lazy-reencode if GRU has been
             updated since this profile was last encoded.
          3. oracle.train(profile, cfr_iterations) — GRU updated inside.
          4. Compute and store embedding drift for Tensorboard logging.
          6. Deepcopy pi_avg + GRU into a new HanabiPolicy; add to pool.

        Args:
            cfr_iterations: CFR traversal budget passed to the oracle.
            k_simulations : rollouts per CFR iteration. If None, uses the
                            oracle's default self.K. Pass an increasing value
                            each XDO iteration to give the oracle more
                            exploration budget as the MetaStrategy grows.

        Returns:
            HanabiPolicy — frozen snapshot of the newly trained policy.

        Raises:
            RuntimeError: if profile_pool is empty.
        """
        if len(self.profile_pool) == 0:
            raise RuntimeError(
                "profile_pool is empty — call extract_behaviour_profile() first."
            )

        # --- Sample profile --------------------------------------------------
        profile_idx = np.random.choice(
            len(self.profile_pool), p=self.meta_strategy
        )
        profile = self.profile_pool[profile_idx]

        # --- Lazy re-encoding ------------------------------------------------
        encoded_at = getattr(profile, "_encoded_at_gru_count", 0)
        if encoded_at < self._gru_train_count:
            old_h = profile.reencode(self.oracle.gru_encoder)
            profile._encoded_at_gru_count = self._gru_train_count
            self.last_embedding_drift = float(max(
                0.0,
                1.0 - F.cosine_similarity(
                    old_h.unsqueeze(0),
                    profile.cached_h_oppo.unsqueeze(0),
                ).item(),
            ))
        else:
            self.last_embedding_drift = 0.0

        # --- Train oracle (GRU weights are updated inside) -------------------
        policy_net = self.oracle.train(profile, cfr_iterations,
                                       k_simulations=k_simulations,
                                       gru_update_every=gru_update_every,
                                       early_stop_min_iters=early_stop_min_iters,
                                       early_stop_delta=early_stop_delta)
        self._gru_train_count += 1

        # --- Snapshot --------------------------------------------------------
        snapshot = HanabiPolicy(
            pi_avg_state_dict=policy_net.state_dict(),
            gru_state_dict=self.oracle.gru_encoder.state_dict(),
        )
        self.policy_pool.append(snapshot)
        return snapshot

    # ------------------------------------------------------------------
    # 3. Extract BehaviourProfile from shared episode batch
    # ------------------------------------------------------------------

    def _compute_embedding_spread(self) -> float:
        """
        Mean pairwise cosine distance between all cached_h_oppo embeddings.

        Returns a value in [0, 1]:
          ~0.0 — all profiles encoded to the same embedding (GRU not
                  differentiating partners, or pool has only one profile).
          >0.0  — GRU is producing distinct representations for different
                  partner behaviours (desired).

        Uses cosine *distance* = 1 − cosine_similarity, so two identical
        vectors give 0.0 and two orthogonal vectors give 1.0.
        Computed over all N*(N-1)/2 unique pairs; returns 0.0 for N < 2.
        """
        N = len(self.profile_pool)
        if N < 2:
            return 0.0

        # Stack all embeddings: (N, 64)
        H = torch.stack([p.cached_h_oppo for p in self.profile_pool])  # (N, 64)
        # Normalise rows to unit vectors for efficient dot-product similarity
        H_norm = torch.nn.functional.normalize(H, dim=1)               # (N, 64)
        # Similarity matrix: (N, N) — dot products of unit vectors
        sim = H_norm @ H_norm.t()                                       # (N, N)
        # Extract upper triangle (i < j) — N*(N-1)/2 values
        idx = torch.triu_indices(N, N, offset=1)
        pairwise_sim = sim[idx[0], idx[1]]                              # (N*(N-1)/2,)
        mean_dist = float((1.0 - pairwise_sim).clamp(min=0.0).mean().item())
        return mean_dist

    def extract_behaviour_profile(
        self,
        episodes: list,
        bc_epochs: int = 40,
    ) -> BehaviourProfile:
        """
        Build a BehaviourProfile from a shared episode batch.

        Extracts only the *partner's* actions and public observations from
        each episode. Never reads this agent's observations or actions.
        The mean team score is stored as the per-profile score entry used
        by solve_meta_game().

        Args:
            episodes : list of episode dicts (see module-level docstring for
                       the expected format).
            bc_epochs: supervised training epochs for the BC partner model.

        Returns:
            The newly created BehaviourProfile (also appended to profile_pool).
        """
        raw_sequences: list = []
        raw_obs_sequences: list = []
        scores: list = []

        for episode in episodes:
            ep_actions: list = []
            ep_obs: list = []
            for step in episode["steps"]:
                if step["current_player"] == self.partner_id:
                    ep_actions.append(step["action"])
                    ep_obs.append(step["player_obs"][self.partner_id])
            if ep_actions:
                raw_sequences.append(ep_actions)
                raw_obs_sequences.append(ep_obs)
            scores.append(float(episode["score"]))

        mean_score = float(np.mean(scores)) if scores else 0.0

        # Safety fallback: partner may not have acted (very unlikely in practice)
        if not raw_sequences:
            raw_sequences = [[0]]
            raw_obs_sequences = [[np.zeros(self.obs_dim, dtype=np.float32)]]

        profile = BehaviourProfile.from_episodes(
            raw_sequences=raw_sequences,
            raw_obs_sequences=raw_obs_sequences,
            gru_encoder=self.oracle.gru_encoder,
            iteration_created=self._iteration,
            source=_PARTNER_SOURCES[self.agent_id],
            bc_epochs=bc_epochs,
        )
        # Mark encoded at the current GRU version so lazy re-encoding is skipped
        profile._encoded_at_gru_count = self._gru_train_count

        # --- BC model quality metrics ----------------------------------------
        # Evaluate the freshly trained BC model on the same pairs it was
        # trained on (in-sample accuracy gives an upper bound; what matters
        # is whether it's well above the 1/8 = 12.5% random baseline).
        eval_pairs = _build_obs_action_pairs(raw_sequences, raw_obs_sequences)
        self.last_bc_loss = profile.bc_train_loss
        self.last_bc_accuracy, _ = profile.bc_model.evaluate(eval_pairs)

        # --- Update data structures -----------------------------------------
        self.profile_pool.append(profile)
        self._profile_scores.append(mean_score)

        # --- Embedding spread (computed after pool updated) ------------------
        self.last_embedding_spread = self._compute_embedding_spread()

        # New profile starts with log-weight 0 (= linear weight 1.0).
        # Normalise via log-sum-exp consistent with solve_meta_game().
        self._log_weights = np.append(self._log_weights, 0.0)
        log_w_shifted = self._log_weights - self._log_weights.max()
        w = np.exp(log_w_shifted)
        self.meta_strategy = w / w.sum()

        self._iteration += 1
        return profile
