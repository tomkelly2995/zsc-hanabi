# src/jax_networks/rppo_actor_critic.py
# Recurrent Actor-Critic for cooperative Tiny Hanabi.
#
# Replaces the separate AdvantageNet + GRUEncoder (h_oppo) + PolicyNet stack
# used by ODCFR with a single module that carries its own GRU hidden state
# across self-turns.  The hidden state implicitly learns to track partner
# behaviour via the last_action field (22-dim) baked into every observation.
#
# Architecture
# ─────────────
#   obs (84) → Dense(128, ReLU) → GRUCell(64) → Dense(128, ReLU)
#                                                 ├─ actor head → logits (8)
#                                                 └─ critic head → value (1)
#
# The GRU carry is reset to zeros at the start of every episode and advances
# one step per self-turn.  Partner turns are not direct GRU inputs — their
# effect is captured through the observation received on the agent's next turn.

from __future__ import annotations
import jax.numpy as jnp
from flax import linen as nn

OBS_DIM     = 84
HIDDEN_DIM  = 64
NUM_ACTIONS = 8


class RPPOActorCritic(nn.Module):
    """
    Recurrent Actor-Critic for cooperative Tiny Hanabi.

    Attributes:
        hidden_dim  : GRU hidden size (default 64, matches ODCFR GRUEncoder)
        num_actions : action space size (default 8)
    """
    hidden_dim:  int = HIDDEN_DIM
    num_actions: int = NUM_ACTIONS

    @nn.compact
    def __call__(self, obs: jnp.ndarray, h: jnp.ndarray):
        """
        Single self-turn forward pass.

        Args:
            obs : (..., 84) float32 — raw observation
            h   : (..., 64) float32 — GRU carry from previous self-turn
                  (zeros at episode start)

        Returns:
            logits : (..., 8)  float32 — raw action logits (pre-mask)
            value  : (...,)    float32 — state value estimate
            new_h  : (..., 64) float32 — updated GRU carry
            partner_logits : (..., 8) float32 — predicted partner action logits
        """
        x     = nn.relu(nn.Dense(128, name="obs_proj")(obs))
        new_h, _ = nn.GRUCell(features=self.hidden_dim, name="gru")(h, x)
        trunk  = nn.relu(nn.Dense(128, name="trunk")(new_h))
        logits = nn.Dense(self.num_actions, name="actor")(trunk)
        value  = nn.Dense(1, name="critic")(trunk).squeeze(-1)
        partner_logits = nn.Dense(self.num_actions, name="partner_head")(new_h)
        return logits, value, new_h, partner_logits
