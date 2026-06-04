# src/agents/bc_partner_model.py
# BCPartnerModel — trained per BehaviourProfile.
# BC network: partner_obs (84-dim) -> action distribution (8-dim).
# Architecture: Dense(84,128,ReLU) -> Dropout(0.2) -> Dense(128,64,ReLU) -> Dense(64,8)
# Training: CrossEntropy with Adam weight_decay=1e-4 (logit squeezing).
# weight_decay penalises large logit magnitudes, preventing overconfidence
# without fighting against genuinely certain predictions (unlike label smoothing).
# Constant learning rate is used — cosine annealing is poorly suited to XDO's
# moving-target setting where each new profile requires full adaptability.
# Replaces i.i.d. action sampling during ODCFR traversal with a proper
# game-state-conditioned partner policy. No access to partner internals.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCPartnerModel(nn.Module):
    """
    Behavioural cloning model for the partner.

    Trained on (partner_obs, partner_action) pairs extracted from joint
    episodes. Used during ODCFR traversal to sample a plausible partner
    action given the partner's current public observation.

    No-info-sharing: trained entirely from observed behaviour. Never
    accesses partner network weights or private state.
    """

    def __init__(self, obs_dim: int = 84, num_actions: int = 8):
        super().__init__()
        self._num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: float tensor of shape (batch, obs_dim)
        Returns:
            float tensor of shape (batch, num_actions) — raw logits
        """
        return self.net(obs)

    def action_distribution(
        self,
        partner_obs: np.ndarray,
        legal_actions: list,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Return a probability distribution over legal actions for partner_obs.

        Illegal actions are masked to zero probability. Used during ODCFR
        traversal: sample an action from the returned distribution, then
        step h_oppo forward via the GRU on the sampled action.

        Args:
            partner_obs  : np.ndarray or torch.Tensor of shape (obs_dim,)
            legal_actions: list of int — currently legal action UIDs
            temperature  : softmax temperature applied to masked logits.
                           1.0 = standard softmax (full stochasticity).
                           <1.0 = sharpened distribution (e.g. 0.3 concentrates
                           mass on the most likely legal action, reducing CFR
                           trajectory variance while retaining some stochasticity).
                           0.0 = one-hot argmax (fully deterministic).

        Returns:
            torch.Tensor of shape (num_actions,) — probabilities.
            Illegal action indices are exactly 0.0.
        """
        if isinstance(partner_obs, np.ndarray):
            obs_t = torch.from_numpy(partner_obs).float()
        else:
            obs_t = partner_obs.float()

        with torch.no_grad():
            logits = self.forward(obs_t.unsqueeze(0)).squeeze(0)   # (num_actions,)

        mask = torch.full((self._num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        masked_logits = logits + mask

        if temperature == 0.0:
            probs = torch.zeros(self._num_actions)
            probs[masked_logits.argmax()] = 1.0
            return probs

        return F.softmax(masked_logits / temperature, dim=0)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_supervised(
        self,
        obs_action_pairs: list,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
    ) -> float:
        """
        Train on (partner_obs, action_id) pairs via cross-entropy loss.

        Called once at profile creation and again on lazy re-encoding.
        A fresh optimizer is created each call so retraining starts clean.

        Args:
            obs_action_pairs: list of (obs_vector, action_id) tuples.
                              obs_vector: array-like of shape (obs_dim,)
                              action_id : int
            epochs          : number of full passes over the dataset
            lr              : Adam learning rate (constant — no scheduler,
                              since XDO presents a moving target and a
                              decayed lr would prevent adaptation to new profiles)
            batch_size      : mini-batch size; clamped to dataset size if smaller
            weight_decay    : L2 penalty on all parameters (logit squeezing).
                              Penalises large logit magnitudes to prevent
                              overconfidence without suppressing genuinely
                              certain predictions the way label_smoothing does.

        Returns:
            float — mean cross-entropy loss over the final epoch.
                    Returns 0.0 if obs_action_pairs is empty.
        """
        if len(obs_action_pairs) == 0:
            return 0.0

        # Build tensors once up front
        obs_list = [
            torch.from_numpy(o).float() if isinstance(o, np.ndarray)
            else o.float()
            for o, _ in obs_action_pairs
        ]
        obs_tensor = torch.stack(obs_list)                                    # (N, obs_dim)
        act_tensor = torch.tensor([a for _, a in obs_action_pairs], dtype=torch.long)  # (N,)

        n = len(obs_action_pairs)
        effective_batch = min(batch_size, n)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        final_loss = 0.0
        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            num_batches = 0
            for start in range(0, n, effective_batch):
                idx = perm[start: start + effective_batch]
                logits = self.forward(obs_tensor[idx])        # (b, num_actions)
                loss = criterion(logits, act_tensor[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                num_batches += 1
            if epoch == epochs - 1:
                final_loss = epoch_loss / max(num_batches, 1)
        self.eval()
        return final_loss

    def evaluate(self, obs_action_pairs: list) -> tuple:
        """
        Evaluate the BC model on a set of (obs, action) pairs.

        Args:
            obs_action_pairs: list of (obs_vector, action_id) tuples.

        Returns:
            (accuracy, mean_entropy) where:
              accuracy    : float — fraction of actions where argmax(logits) == true action.
                            Random baseline = 1/num_actions = 0.125 for 8 actions.
              mean_entropy: float — mean Shannon entropy of the predicted softmax
                            distributions. Low = confident predictions; high = uncertain.
                            Max = log(num_actions) ≈ 2.08 for 8 actions (uniform).
        """
        if len(obs_action_pairs) == 0:
            return 0.0, 0.0

        obs_list = [
            torch.from_numpy(o).float() if isinstance(o, np.ndarray)
            else o.float()
            for o, _ in obs_action_pairs
        ]
        obs_tensor = torch.stack(obs_list)                                    # (N, obs_dim)
        act_tensor = torch.tensor([a for _, a in obs_action_pairs], dtype=torch.long)  # (N,)

        self.eval()
        with torch.no_grad():
            logits = self.forward(obs_tensor)                                 # (N, num_actions)
            probs  = F.softmax(logits, dim=1)                                 # (N, num_actions)
            preds  = logits.argmax(dim=1)                                     # (N,)

        accuracy = (preds == act_tensor).float().mean().item()
        # Shannon entropy per sample: -Σ p log p
        log_probs    = torch.log(probs.clamp(min=1e-12))
        entropies    = -(probs * log_probs).sum(dim=1)                        # (N,)
        mean_entropy = entropies.mean().item()

        return accuracy, mean_entropy
