# src/agents/cooperative_odcfr_agent.py
# CooperativeODCFRAgent — inner-loop oracle, one instance per agent.
# Implements outcome-sampling Deep CFR with team counterfactual regret.
# Partner turns are simulated via the BehaviourProfile's BCPartnerModel;
# h_oppo is maintained by stepping the GRU on observed partner actions.
# No policy weights from the partner are accessed at any point.
# OBS_DIM=84, NUM_ACTIONS=8, H_DIM=64  (verified against tiny Hanabi config)

import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.networks.advantage_net import AdvantageNet
from src.networks.policy_net import PolicyNet
from src.networks.gru_encoder import GRUEncoder
from src.env.hle_wrapper import HLEWrapper


# ------------------------------------------------------------------
# Reservoir buffer
# ------------------------------------------------------------------

class ReservoirBuffer:
    """
    Fixed-capacity reservoir sampling buffer.
    Maintains a uniform random sample of all items ever added.
    """

    def __init__(self, max_size: int):
        self._max = max_size
        self._data: list = []
        self._count: int = 0

    def add(self, item) -> None:
        self._count += 1
        if len(self._data) < self._max:
            self._data.append(item)
        else:
            idx = random.randint(0, self._count - 1)
            if idx < self._max:
                self._data[idx] = item

    def sample(self, batch_size: int) -> list:
        n = min(batch_size, len(self._data))
        return random.choices(self._data, k=n)

    def __len__(self) -> int:
        return len(self._data)


# ------------------------------------------------------------------
# Regret matching
# ------------------------------------------------------------------

def _regret_matching(
    q_vals: torch.Tensor,
    legal_actions: list,
    num_actions: int,
) -> torch.Tensor:
    """
    Convert Q values (team counterfactual regret estimates) into a
    current-strategy distribution via regret matching.

    Regret matching weights actions by their *advantage* over the current
    strategy value — i.e. how much better each action is than average —
    not by their raw Q values.  Using raw Q values is incorrect because
    Hanabi scores are always >= 0, so all legal actions would always
    receive positive weight and the strategy would never concentrate on
    the genuinely best action.

    V_σ = mean Q over legal actions (uniform baseline for first call;
          refining to the sigma-weighted average would require a two-pass
          update but the uniform mean is standard and sufficient here).

    Illegal actions receive zero probability. Falls back to uniform over
    legal actions if all advantages are <= 0.

    Args:
        q_vals       : tensor of shape (num_actions,) — Q estimates
        legal_actions: list of int — currently legal action UIDs
        num_actions  : int — total action space size

    Returns:
        tensor of shape (num_actions,) — probability distribution,
        zero for all illegal actions.
    """
    sigma = torch.zeros(num_actions, device=q_vals.device)
    legal_q = q_vals[legal_actions]

    # Baseline: current strategy value (uniform mean over legal actions)
    v_sigma = legal_q.mean()

    # Advantage: how much better than average; clip negatives to zero
    advantages = torch.clamp(legal_q - v_sigma, min=0.0)
    total = advantages.sum().item()
    if total > 0:
        sigma[legal_actions] = advantages / total
    else:
        # All actions equally good (or equally bad) — play uniform
        sigma[legal_actions] = 1.0 / len(legal_actions)
    return sigma


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

_PLAYER_IDS = {"agent_A": 0, "agent_B": 1}


class CooperativeODCFRAgent:
    """
    Cooperative ODCFR oracle.

    train(profile, cfr_iterations) runs cfr_iterations of outcome-sampling
    Deep CFR, then returns the trained pi_avg (PolicyNet).

    Each CFR iteration:
      1. Run K episode simulations, collecting (obs_h_oppo, regret_target,
         iteration_t) into AdvantageBuffer and (obs_h_oppo, sigma) into
         PolicyBuffer.
      2. Re-train Q_adv on the full AdvantageBuffer with iteration weights.

    After all iterations:
      3. Train pi_avg on the full PolicyBuffer.
      4. Return pi_avg.
    """

    def __init__(
        self,
        agent_id: str,
        obs_dim: int = 84,
        num_actions: int = 8,
        K_simulations: int = 16,
        adv_buffer_size: int = 1_000_000,
        adv_train_steps: int = 100,
        pol_train_steps: int = 200,
        batch_size: int = 512,
        adv_lr: float = 1e-3,
        pol_lr: float = 1e-3,
        aux_weight: float = 0.2,
        partner_temperature: float = 0.3,
        device: torch.device | None = None,
    ):
        self.agent_id = agent_id
        self.player_id = _PLAYER_IDS[agent_id]
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.K = K_simulations
        self._adv_train_steps = adv_train_steps
        self._pol_train_steps = pol_train_steps
        self._batch_size = batch_size
        self._aux_weight = aux_weight
        self._partner_temperature = partner_temperature

        # Device — default to CUDA if available, fall back to CPU
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Networks — one set per agent, no weight sharing; all moved to device
        self.advantage_net = AdvantageNet(input_dim=obs_dim + 64, num_actions=num_actions).to(self.device)
        self.policy_net = PolicyNet(input_dim=obs_dim + 64, num_actions=num_actions).to(self.device)
        self.gru_encoder = GRUEncoder(num_actions=num_actions).to(self.device)

        # Auxiliary next-action prediction head.
        # Predicts the partner's next action from the GRU hidden state at each
        # step — gives the GRU a self-supervised signal that is independent of
        # the advantage loss, preventing embedding collapse in early training
        # when Q-value estimates are too noisy to drive useful representations.
        self.next_action_head = nn.Linear(64, num_actions).to(self.device)

        # Replay buffers
        self._adv_buffer = ReservoirBuffer(adv_buffer_size)
        self._pol_buffer = ReservoirBuffer(adv_buffer_size)

        # Optimisers
        # GRU and next_action_head are included in _adv_opt so all three are
        # updated together in _update_gru() via a combined advantage +
        # next-action-prediction loss.
        self._adv_opt = torch.optim.Adam(
            list(self.advantage_net.parameters()) +
            list(self.gru_encoder.parameters()) +
            list(self.next_action_head.parameters()),
            lr=adv_lr,
        )
        self._pol_opt = torch.optim.Adam(self.policy_net.parameters(), lr=pol_lr)

        # Shared environment instance (reset before each simulation)
        self._env = HLEWrapper()

        # Timing accumulators — populated by train(), read by main.py
        self.last_train_sim_secs: float = 0.0
        self.last_train_adv_secs: float = 0.0
        self.last_train_gru_secs: float = 0.0
        self.last_train_pol_secs: float = 0.0
        # Actual CFR iterations executed (may be < cfr_iterations if early stopped)
        self.last_cfr_iterations_run: int = 0
        # Mean absolute Q_adv change on the probe set at the final CFR iteration.
        # Use this to calibrate early_stop_delta: if this value is well above delta
        # at iter 500 the net genuinely needs more iterations; if it's near or below
        # delta the net has converged and early stopping is just miscalibrated.
        self.last_probe_delta: float = float("nan")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(
        self,
        profile,
        cfr_iterations: int,
        adv_retention: float = 0.2,
        k_simulations: int | None = None,
        early_stop_patience: int = 20,
        early_stop_delta: float = 1e-2,
        early_stop_min_iters: int = 250,
        gru_update_every: int = 5,
    ) -> PolicyNet:
        """
        Run Cooperative ODCFR for cfr_iterations iterations against profile.

        At the start of each call the policy buffer is cleared entirely
        (sigma targets were computed against the previous profile, so they're
        always stale).  The advantage buffer is partially retained: a random
        20% of entries are carried over with their iteration weights reset to
        1.0 so the advantage net has enough samples to begin training
        immediately (avoids the first ~32 CFR iterations being wasted while
        the buffer fills past batch_size).  The remaining 80% is discarded to
        flush stale cross-profile data.

        Args:
            profile              : BehaviourProfile — sampled from this agent's pool
            cfr_iterations       : int — number of CFR iterations (outer count)
            adv_retention        : float in [0, 1] — fraction of advantage buffer
                                   entries to carry over (default 0.2)
            k_simulations        : int | None — rollouts per CFR iteration. If None,
                                   uses self.K (the value set at construction time).
                                   Pass a value > self.K to increase exploration as
                                   the MetaStrategy grows more complex.
            early_stop_patience  : int — number of consecutive stable iterations
                                   required before stopping early (default 20).
            early_stop_delta     : float — mean absolute Q_adv change threshold
                                   below which an iteration is considered stable
                                   (default 5e-2).
            early_stop_min_iters : int — earliest CFR iteration at which early
                                   stopping may fire (default 50). Prevents the
                                   check from triggering before the advantage net
                                   has had enough iterations to learn anything
                                   meaningful (otherwise retained-buffer warmth
                                   causes the probe to look stable from iter 1).
            gru_update_every     : int — run _update_gru() every N CFR iterations
                                   instead of every iteration (default 5). The
                                   partner profile is fixed within a train() call
                                   so the GRU representation only needs to converge,
                                   not track changes — every 5 iters is sufficient.

        Returns:
            PolicyNet — the trained pi_avg, ready to be snapshotted into a
            HanabiPolicy by XDOHanabiSolver.
        """
        # --- Partial advantage buffer retention ----------------------------
        # Carry over a random subset of old entries, resetting iteration_t
        # to 1.0 so they count as low-weight warmup data relative to the
        # new samples (t=1..cfr_iterations) that will be added below.
        retained = []
        if adv_retention > 0 and len(self._adv_buffer) > 0:
            n_keep = max(1, int(len(self._adv_buffer) * adv_retention))
            kept = random.sample(self._adv_buffer._data,
                                 min(n_keep, len(self._adv_buffer._data)))
            retained = [(obs_h, pa, target, 1.0) for obs_h, pa, target, _ in kept]

        self._adv_buffer = ReservoirBuffer(self._adv_buffer._max)
        self._pol_buffer = ReservoirBuffer(self._pol_buffer._max)

        for item in retained:
            self._adv_buffer.add(item)

        K = k_simulations if k_simulations is not None else self.K

        # Early-stopping state
        probe_obs_h: torch.Tensor | None = None       # fixed probe batch, set once
        probe_prev_preds: torch.Tensor | None = None  # Q_adv predictions last iter
        stable_count: int = 0
        self.last_cfr_iterations_run = cfr_iterations  # overwritten if we stop early

        sim_secs = adv_secs = gru_secs = 0.0
        for t in range(1, cfr_iterations + 1):
            t0 = time.perf_counter()
            for _ in range(K):
                self._run_simulation(profile, iteration_t=t)
            sim_secs += time.perf_counter() - t0

            t0 = time.perf_counter()
            self._update_advantage_net(iteration_t=t)
            adv_secs += time.perf_counter() - t0

            # ----------------------------------------------------------
            # Early stopping: Q_adv stability on a fixed probe set.
            # Once the buffer has enough samples, we lock in a probe batch
            # and check whether the advantage net's predictions are still
            # changing meaningfully.  If they haven't moved by more than
            # early_stop_delta (mean absolute change) for early_stop_patience
            # consecutive iterations, the sub-game is considered solved and
            # we skip the remaining iterations.
            # ----------------------------------------------------------
            if t >= early_stop_min_iters and len(self._adv_buffer) >= self._batch_size:
                if probe_obs_h is None:
                    # Fix the probe set once and reuse it every iteration so
                    # the stability signal is comparable across iterations.
                    # Only sampled once early_stop_min_iters is reached so the
                    # probe reflects a net that has already done real learning.
                    # Move to device immediately — kept there for all iterations.
                    probe_batch = self._adv_buffer.sample(self._batch_size)
                    probe_obs_h = torch.stack([b[0] for b in probe_batch]).to(self.device)
                with torch.no_grad():
                    preds = self.advantage_net(probe_obs_h)
                if probe_prev_preds is not None:
                    mean_abs_change = (preds - probe_prev_preds).abs().mean().item()
                    self.last_probe_delta = mean_abs_change
                    if mean_abs_change < early_stop_delta:
                        stable_count += 1
                        if stable_count >= early_stop_patience:
                            self.last_cfr_iterations_run = t
                            break
                    else:
                        stable_count = 0
                probe_prev_preds = preds

            if t % gru_update_every == 0:
                t0 = time.perf_counter()
                self._update_gru()
                gru_secs += time.perf_counter() - t0

        t0 = time.perf_counter()
        self._update_policy_net()
        pol_secs = time.perf_counter() - t0

        self.last_train_sim_secs = sim_secs
        self.last_train_adv_secs = adv_secs
        self.last_train_gru_secs = gru_secs
        self.last_train_pol_secs = pol_secs
        return self.policy_net

    # ------------------------------------------------------------------
    # Traversal (single episode simulation)
    # ------------------------------------------------------------------

    def _run_simulation(self, profile, iteration_t: int) -> None:
        """
        Simulate one complete Hanabi episode.

        Self turns: Q_adv → regret matching → sample action.
                    Record (obs_h_oppo, action, sigma) for buffer.
        Partner turns: BC model → sample action → step h_oppo via GRU.

        At terminal: use team score as the CF value for the taken action.
        Populate AdvantageBuffer and PolicyBuffer.
        """
        obs_step = self._env.reset()
        # h_oppo starts from zeros — consistent with HanabiPolicy.initial_h_oppo().
        # Maintained under no_grad during traversal for efficiency; the GRU is
        # trained end-to-end in _update_gru by recomputing h_oppo from the stored
        # raw partner-action sequences WITH gradients.
        h_oppo = self.gru_encoder.initial_hidden()
        partner_id = 1 - self.player_id
        done = False
        info = {"score": 0}
        self_records: list = []
        # Running list of partner actions observed so far this episode.
        # Snapshot is stored with each self-turn record so _update_gru can replay
        # the exact GRU prefix that produced h_oppo at that moment.
        partner_actions_so_far: list = []

        while not done:
            current_player = obs_step.current_player

            if current_player == self.player_id:
                obs = obs_step.player_obs[self.player_id]
                legal = obs_step.legal_moves[self.player_id]

                obs_t = torch.from_numpy(obs).float().to(self.device)
                obs_h = torch.cat([obs_t, h_oppo]).unsqueeze(0)   # (1, 148) on device

                with torch.no_grad():
                    q_vals = self.advantage_net(obs_h).squeeze(0)  # (8,) on device

                sigma = _regret_matching(q_vals, legal, self.num_actions)
                action = torch.multinomial(sigma, 1).item()

                # Store buffer items as CPU tensors — keeps the replay buffer
                # off GPU memory and avoids device mismatches when sampling.
                self_records.append({
                    "partner_actions": list(partner_actions_so_far),
                    "obs_h":           obs_h.squeeze(0).detach().cpu(),
                    "action":          action,
                    "sigma":           sigma.detach().cpu(),
                })

            else:
                # Partner turn — BC model only, no policy weights accessed
                partner_obs = obs_step.player_obs[partner_id]
                partner_legal = obs_step.legal_moves[partner_id]

                probs = profile.bc_model.action_distribution(
                    partner_obs, partner_legal,
                    temperature=self._partner_temperature,
                )
                action = torch.multinomial(probs, 1).item()

                # Track for GRU replay in _update_gru
                partner_actions_so_far.append(action)
                with torch.no_grad():
                    h_oppo = self.gru_encoder.step(h_oppo, action)

            obs_step, _, done, info = self._env.step(action)

        final_score = float(info["score"])

        # Populate buffers using terminal team score.
        # Move obs_h to device for inference, store target back on CPU.
        for rec in self_records:
            obs_h_dev = rec["obs_h"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_vals = self.advantage_net(obs_h_dev).squeeze(0)
            target = q_vals.clone()
            target[rec["action"]] = final_score

            self._adv_buffer.add((
                rec["obs_h"],                   # CPU
                rec["partner_actions"],
                target.detach().cpu(),          # CPU
                float(iteration_t),
            ))
            self._pol_buffer.add((rec["obs_h"], rec["sigma"]))  # both CPU

    # ------------------------------------------------------------------
    # Network updates
    # ------------------------------------------------------------------

    def _update_advantage_net(self, iteration_t: int) -> None:
        """
        Train Q_adv on the AdvantageBuffer using stored obs_h (fast path).
        Samples are weighted by their iteration index (recency bias).
        Called once per CFR iteration.
        """
        if len(self._adv_buffer) < self._batch_size:
            return

        self.advantage_net.train()
        for _ in range(self._adv_train_steps):
            batch   = self._adv_buffer.sample(self._batch_size)
            obs_h   = torch.stack([b[0] for b in batch]).to(self.device)   # (B, 148)
            targets = torch.stack([b[2] for b in batch]).to(self.device)   # (B, 8)
            weights = torch.tensor(
                [b[3] for b in batch], dtype=torch.float32, device=self.device
            )
            weights = weights / weights.sum()

            preds      = self.advantage_net(obs_h)             # (B, 8)
            per_sample = ((preds - targets) ** 2).mean(dim=1) # (B,)
            loss       = (weights * per_sample).sum()

            self._adv_opt.zero_grad()
            loss.backward()
            self._adv_opt.step()

        self.advantage_net.eval()

    def _update_gru(self) -> None:
        """
        Train the GRU encoder via a combined loss:

          loss = advantage_loss + aux_weight * next_action_prediction_loss

        advantage_loss: MSE between predicted Q-values (using recomputed h_oppo)
            and stored targets — same signal as before.

        next_action_prediction_loss: cross-entropy over per-step GRU outputs.
            For each sequence [a0, a1, ..., aL-1] of length >= 2, the GRU
            hidden state after processing a0..a_{t-1} is used to predict a_t.
            This gives the GRU a self-supervised objective independent of Q-value
            quality, preventing embedding collapse in early training when the
            advantage signal is too noisy to drive useful representations.
            Only sequences of length >= 2 contribute (length 0 has no actions;
            length 1 has a single action with nothing to predict from).

        Called ONCE per CFR iteration. All non-empty sequences are processed in
        a single batched GRU forward pass via pack_padded_sequence, replacing
        the previous per-length-group loop.
        """
        if len(self._adv_buffer) < self._batch_size:
            return

        batch = self._adv_buffer.sample(self._batch_size)
        seqs    = [b[1] for b in batch]                        # list of list[int]
        targets = torch.stack([b[2] for b in batch]).to(self.device)    # (B, 8)
        weights = torch.tensor(
            [b[3] for b in batch], dtype=torch.float32, device=self.device
        )
        weights = weights / weights.sum()

        # Extract obs from stored obs_h (first obs_dim dims), move to device
        obs_stacked = torch.stack([b[0][:self.obs_dim] for b in batch]).to(self.device)  # (B, obs_dim)

        B = len(seqs)
        # h_oppos_tensor: final hidden states on device; rows for empty seqs stay zero.
        h_oppos_tensor = torch.zeros(B, self.gru_encoder.hidden_dim, device=self.device)

        aux_loss_terms: list[torch.Tensor] = []

        self.gru_encoder.train()
        self.next_action_head.train()

        # Collect indices and sequences for non-empty entries only.
        nz_info = [(i, s) for i, s in enumerate(seqs) if len(s) > 0]
        if nz_info:
            nz_indices, nz_seqs = zip(*nz_info)
            nz_indices = list(nz_indices)
            nz_seqs    = list(nz_seqs)
            nz_lengths = [len(s) for s in nz_seqs]
            max_len    = max(nz_lengths)
            n          = len(nz_seqs)

            # Pad all non-empty sequences to max_len in one tensor (on device).
            padded_acts = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
            for j, seq in enumerate(nz_seqs):
                padded_acts[j, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

            # Single batched GRU forward pass for all non-empty sequences.
            embedded = self.gru_encoder.embedding(padded_acts)  # (n, max_len, embed_dim)
            packed   = pack_padded_sequence(
                embedded, nz_lengths, batch_first=True, enforce_sorted=False
            )
            output_packed, h_n = self.gru_encoder.gru(packed)
            h_finals = h_n.squeeze(0)                           # (n, hidden_dim)

            # Scatter final hidden states back into h_oppos_tensor.
            nz_idx_t = torch.tensor(nz_indices, dtype=torch.long, device=self.device)
            h_oppos_tensor = h_oppos_tensor.index_copy(0, nz_idx_t, h_finals)

            # Next-action prediction aux loss over all valid (non-padded) positions.
            if max_len >= 2:
                output_padded, _ = pad_packed_sequence(
                    output_packed, batch_first=True, total_length=max_len
                )                                               # (n, max_len, hidden_dim)

                # valid_mask[j, t] = True iff position t has a next action to predict,
                # i.e. t < nz_lengths[j] - 1.
                nz_lens_t  = torch.tensor(nz_lengths)
                positions  = torch.arange(max_len - 1).unsqueeze(0)         # (1, max_len-1)
                valid_mask = positions < (nz_lens_t.unsqueeze(1) - 1)       # (n, max_len-1)

                if valid_mask.any():
                    h_pred_all    = output_padded[:, :-1, :]   # (n, max_len-1, hidden_dim)
                    next_acts_all = padded_acts[:, 1:]          # (n, max_len-1)
                    logits = self.next_action_head(h_pred_all[valid_mask])
                    aux_loss_terms.append(
                        F.cross_entropy(logits, next_acts_all[valid_mask])
                    )

        # Concatenate obs and h_oppo in one vectorised op instead of per-sample cats.
        obs_h = torch.cat([obs_stacked, h_oppos_tensor], dim=1)  # (B, obs_dim + hidden_dim)

        self.advantage_net.train()
        preds      = self.advantage_net(obs_h)
        per_sample = ((preds - targets) ** 2).mean(dim=1)
        adv_loss   = (weights * per_sample).sum()

        if aux_loss_terms:
            aux_loss = torch.stack(aux_loss_terms).mean()
            loss = adv_loss + self._aux_weight * aux_loss
        else:
            loss = adv_loss

        self._adv_opt.zero_grad()
        loss.backward()
        self._adv_opt.step()

        self.advantage_net.eval()
        self.gru_encoder.eval()
        self.next_action_head.eval()

    def _update_policy_net(self) -> None:
        """
        Train pi_avg on the PolicyBuffer via cross-entropy against the
        regret-matched strategies collected during traversal.
        Called once at the end of train(), after all CFR iterations.
        """
        if len(self._pol_buffer) < self._batch_size:
            return

        self.policy_net.train()
        for _ in range(self._pol_train_steps):
            batch = self._pol_buffer.sample(self._batch_size)
            obs_h = torch.stack([b[0] for b in batch]).to(self.device)    # (B, 148)
            sigma = torch.stack([b[1] for b in batch]).to(self.device)    # (B, 8)

            logits = self.policy_net(obs_h)
            loss = -(sigma * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

            self._pol_opt.zero_grad()
            loss.backward()
            self._pol_opt.step()

        self.policy_net.eval()
