# src/networks/advantage_net.py
# Q_adv: predicts team counterfactual regret values per action.
# Input : obs (84-dim) || h_oppo (64-dim) = 148-dim concatenated before passing in.
# Output: 8 raw regret values (no output activation).
# Architecture (spec Section 4.2):
#   Dense(148, 256, ReLU) -> Dense(256, 128, ReLU) -> Dense(128, 128, ReLU) -> Dense(128, 8)
# Retrained from scratch each CFR iteration using AdvantageBuffer.
# One instance per agent — no weight sharing.

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdvantageNet(nn.Module):
    """
    Advantage network (Q_adv). Maps concatenated [obs || h_oppo] to a
    vector of team counterfactual regret values, one per action.

    Output is raw (no activation) — regret matching is applied by the caller.
    """

    def __init__(self, input_dim: int = 148, num_actions: int = 8):
        super().__init__()
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
                        = torch.cat([obs, h_oppo], dim=-1)
        Returns:
            float tensor of shape (batch, 8) — raw team counterfactual regret values
        """
        return self.net(obs_h_oppo)
