# src/jax_networks/advantage_net.py
# Flax port of AdvantageNet: 148→256→128→128→8, raw Q-values, no output activation.
# Input: concat([obs(84), h_oppo(64)]) = 148-dim vector.

from flax import linen as nn


class AdvantageNet(nn.Module):
    """148→256→128→128→8 advantage Q-value network."""

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (..., 148) float32 — concat of obs and partner embedding
        Returns:
            (..., 8) float32 — raw Q-values for each action
        """
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return nn.Dense(8)(x)
