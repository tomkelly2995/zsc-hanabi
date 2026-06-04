# src/networks/policy_net.py
# pi_avg: average strategy network.
# Input : obs (84-dim) || h_oppo (64-dim) = 148-dim concatenated before passing in.
# Output: 8 raw action logits (softmax + legal-action mask applied externally or via masked_probs()).
# Architecture (spec Section 4.2):
#   Dense(148, 256, ReLU) -> Dense(256, 128, ReLU) -> Dense(128, 128, ReLU) -> Dense(128, 8)
# Updated via reservoir sampling during CFR traversal.
# One instance per agent — no weight sharing.

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    """
    Average policy network (pi_avg). Maps concatenated [obs || h_oppo] to
    raw action logits.

    forward() returns raw logits. Use masked_probs() for the legal-action
    masked probability distribution needed at inference time.
    """

    def __init__(self, input_dim: int = 148, num_actions: int = 8):
        super().__init__()
        self._num_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs_h_oppo: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_h_oppo: float tensor of shape (batch, 148)
        Returns:
            float tensor of shape (batch, 8) — raw action logits
        """
        return self.net(obs_h_oppo)

    def masked_probs(self, obs_h_oppo: torch.Tensor, legal_actions: list) -> torch.Tensor:
        """
        Convenience method: forward pass + legal-action mask + softmax.

        Illegal actions are set to -inf before softmax so they receive
        zero probability. Used by HanabiPolicy and the traversal loop.

        Args:
            obs_h_oppo  : float tensor of shape (1, 148) — single observation.
            legal_actions: list of int — action UIDs that are currently legal.

        Returns:
            float tensor of shape (num_actions,) — probability distribution.
            Illegal action indices will be exactly 0.0.
        """
        logits = self.forward(obs_h_oppo).squeeze(0)          # (num_actions,)
        mask = torch.full((self._num_actions,), float("-inf"))
        mask[legal_actions] = 0.0
        return F.softmax(logits + mask, dim=0)
