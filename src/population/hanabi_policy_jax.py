# src/population/hanabi_policy_jax.py
# JAX/Flax port of HanabiPolicy.
#
# Frozen snapshot of trained (pol_params, gru_params). All state is
# immutable after construction — only h_oppo advances during play.
# Stateless Flax modules are used for inference (no PyTorch eval/no_grad).

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.jax_networks.policy_net import PolicyNet
from src.jax_networks.gru_encoder import GRUEncoder, HIDDEN_DIM

_pol_net = PolicyNet()
_gru_enc = GRUEncoder()


class HanabiPolicyJax:
    """
    Frozen policy snapshot used during meta-game evaluation and deployment.

    Holds immutable Flax param dicts for PolicyNet and GRUEncoder. The only
    thing that changes at execution time is h_oppo, updated via the GRU after
    each observed partner action.

    Attributes:
        pol_params : Flax param dict for PolicyNet
        gru_params : Flax param dict for GRUEncoder
    """

    def __init__(self, pol_params, gru_params):
        # Params are JAX pytrees — frozen by convention (never mutated).
        self.pol_params = pol_params
        self.gru_params = gru_params

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        h_oppo: jnp.ndarray,
        legal_actions: list,
    ) -> dict:
        """
        Return a probability distribution over legal actions.

        Args:
            obs          : (obs_dim,) float array — current observation
            h_oppo       : (HIDDEN_DIM,) float32 — current partner embedding
            legal_actions: list[int] — currently legal action indices

        Returns:
            dict {action_id: probability} — only legal actions, sums to 1.
        """
        obs_arr = jnp.array(obs, dtype=jnp.float32)
        legal_mask = jnp.zeros(8, dtype=jnp.bool_).at[jnp.array(legal_actions)].set(True)
        probs = _pol_net.masked_probs(self.pol_params, obs_arr, h_oppo, legal_mask)
        return {a: float(probs[a]) for a in legal_actions}

    # ------------------------------------------------------------------
    # Online h_oppo update
    # ------------------------------------------------------------------

    def update_h_oppo(self, h_oppo: jnp.ndarray, partner_action: int) -> jnp.ndarray:
        """
        Advance h_oppo by one observed partner action.

        Args:
            h_oppo         : (HIDDEN_DIM,) float32
            partner_action : int — observed action index

        Returns:
            (HIDDEN_DIM,) float32 — updated carry
        """
        return _gru_enc.step(self.gru_params, h_oppo, partner_action)

    def initial_h_oppo(self) -> jnp.ndarray:
        """Return zero hidden state for the start of a new episode."""
        return jnp.zeros(HIDDEN_DIM, dtype=jnp.float32)
