# src/population/rppo_policy_jax.py
# Frozen RPPO policy snapshot — analogous to HanabiPolicyJax but holds
# RPPOActorCritic params instead of PolicyNet + GRUEncoder param dicts.

from __future__ import annotations


class RPPOPolicy:
    """
    Frozen snapshot of a trained RPPOActorCritic policy.

    Used during meta-game evaluation and joint episode collection in the
    RPPO XDO outer loop.  Immutable after construction.

    Attributes:
        ac_params : RPPOActorCritic Flax variable dict ({"params": ...})
        player_id : 0 (agent_A) or 1 (agent_B)
    """

    def __init__(self, ac_params, player_id: int):
        self.ac_params = ac_params
        self.player_id = player_id
