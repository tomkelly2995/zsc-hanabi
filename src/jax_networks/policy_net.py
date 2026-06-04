# src/jax_networks/policy_net.py
# Flax port of PolicyNet: 148→256→128→128→8.
# Input: concat([obs(84), h_oppo(64)]) = 148-dim vector.

import jax.numpy as jnp
from flax import linen as nn


class PolicyNet(nn.Module):
    """148→256→128→128→8 policy network."""

    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: (..., 148) float32
        Returns:
            (..., 8) float32 — raw logits
        """
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        return nn.Dense(8)(x)

    def masked_probs(self, params, obs, h_oppo, legal_mask):
        """
        Softmax over legal actions; illegal actions receive probability 0.

        Args:
            params     : Flax param dict
            obs        : (..., 84) float32
            h_oppo     : (..., 64) float32
            legal_mask : (..., 8) bool — True where action is legal

        Returns:
            (..., 8) float32 — probability distribution over actions
        """
        x = jnp.concatenate([obs, h_oppo], axis=-1)
        logits = self.apply(params, x)
        # Replace illegal action logits with -inf so softmax gives 0.
        masked = jnp.where(legal_mask, logits, jnp.finfo(jnp.float32).min)
        return nn.softmax(masked, axis=-1)
