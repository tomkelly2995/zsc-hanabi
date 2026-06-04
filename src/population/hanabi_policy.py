# src/population/hanabi_policy.py
# HanabiPolicy: static, frozen snapshot of a trained CooperativeODCFRAgent.
# Constructed from state_dicts — never holds references to live training objects.
# Weights are frozen (requires_grad=False) and networks are in eval mode.
# Online adaptation at execution time: h_oppo stepped forward via GRU only.

import copy
import numpy as np
import torch

from src.networks.policy_net import PolicyNet
from src.networks.gru_encoder import GRUEncoder


class HanabiPolicy:
    """
    Frozen policy snapshot used during meta-game evaluation and deployment.

    Encapsulates a pi_avg network and a GRU encoder, both loaded from
    state_dicts and permanently frozen. The only thing that changes at
    execution time is h_oppo, which is stepped forward via the GRU after
    each observed partner action.
    """

    def __init__(
        self,
        pi_avg_state_dict: dict,
        gru_state_dict: dict,
        obs_dim: int = 84,
        num_actions: int = 8,
    ):
        # Build fresh network instances and load the frozen weights.
        # load_state_dict copies tensors, so these instances are independent
        # of whatever network produced the state_dicts.
        self._policy_net = PolicyNet(input_dim=obs_dim + 64, num_actions=num_actions)
        self._policy_net.load_state_dict(copy.deepcopy(pi_avg_state_dict))
        self._policy_net.eval()
        self._policy_net.requires_grad_(False)

        self._gru_encoder = GRUEncoder(num_actions=num_actions)
        self._gru_encoder.load_state_dict(copy.deepcopy(gru_state_dict))
        self._gru_encoder.eval()
        self._gru_encoder.requires_grad_(False)

        self._num_actions = num_actions

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        h_oppo: torch.Tensor,
        legal_actions: list,
    ) -> dict:
        """
        Return a probability distribution over legal_actions.

        Args:
            obs          : np.ndarray or torch.Tensor of shape (obs_dim,)
            h_oppo       : torch.Tensor of shape (64,) — current partner embedding
            legal_actions: list of int — currently legal action UIDs

        Returns:
            dict {action_id: probability} — only legal actions are included.
            Probabilities sum to 1.0.
        """
        if isinstance(obs, np.ndarray):
            obs_t = torch.from_numpy(obs).float()
        else:
            obs_t = obs.float()

        obs_h = torch.cat([obs_t, h_oppo]).unsqueeze(0)   # (1, obs_dim + 64)

        with torch.no_grad():
            probs = self._policy_net.masked_probs(obs_h, legal_actions)  # (num_actions,)

        return {a: probs[a].item() for a in legal_actions}

    # ------------------------------------------------------------------
    # Online h_oppo update
    # ------------------------------------------------------------------

    def update_h_oppo(self, h_oppo: torch.Tensor, partner_action: int) -> torch.Tensor:
        """
        Step h_oppo forward by one observed partner action.

        Called after every partner move during both evaluation and live play.
        Does not mutate the input tensor.

        Args:
            h_oppo        : torch.Tensor of shape (64,)
            partner_action: int — observed partner action UID

        Returns:
            New torch.Tensor of shape (64,).
        """
        with torch.no_grad():
            return self._gru_encoder.step(h_oppo, partner_action)

    def initial_h_oppo(self) -> torch.Tensor:
        """
        Return a zero hidden state for the start of a new episode.

        Returns:
            torch.Tensor of shape (64,), all zeros.
        """
        return self._gru_encoder.initial_hidden()
