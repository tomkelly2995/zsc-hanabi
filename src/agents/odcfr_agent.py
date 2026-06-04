"""
src/agents/odcfr_agent.py
-------------------------
Stub for the Opponent-Modelling Deep CFR Agent (ODCFR).

This file defines the interface that XDOSolver expects.  All methods raise
NotImplementedError so that missing implementations surface clearly at
runtime rather than silently doing nothing.

Integration contract with XDOSolver
-------------------------------------
XDOSolver calls:

    avg_policy_net = agent.train(
        iterations       = <int>,
        opponent_context = {
            "policy_pool":   list[NeuralNetworkPolicy],   # opponent's pool
            "meta_strategy": np.ndarray,                  # weights over pool
        },
    )

train() must return a PyTorch nn.Module (the Average Policy Network) whose
forward(obs) maps a (1, obs_dim) observation tensor to (1, num_actions) logits.

XDOSolver then wraps the returned module in a NeuralNetworkPolicy snapshot
(deep-copying state_dict()), so train() does NOT need to freeze the network
itself – just return it after training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Any


# ---------------------------------------------------------------------------
# Placeholder network architectures
# ---------------------------------------------------------------------------

class AdvantageNetwork(nn.Module):
    """
    Maps (observation) → advantage values for each action.
    Used during CFR traversals to compute regrets.

    Replace the architecture below with your real design.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) float tensor.
        Returns:
            (batch, num_actions) advantage logits.
        """
        return self.net(obs)


class AveragePolicyNetwork(nn.Module):
    """
    Maps (observation) → action logits representing the average policy.
    This is the network whose weights are snapshotted at the end of each
    XDO iteration and stored as a NeuralNetworkPolicy.

    Replace the architecture below with your real design.
    """

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch, obs_dim) float tensor.
        Returns:
            (batch, num_actions) raw logits (apply softmax externally).
        """
        return self.net(obs)


# ---------------------------------------------------------------------------
# ODCFR Agent
# ---------------------------------------------------------------------------

class ODCFRAgent:
    """
    Opponent-Modelling Deep CFR Agent.

    One instance is created per player by XDOSolver.  Each instance owns its
    own AdvantageNetwork, AveragePolicyNetwork, replay buffers, and Belief
    Engine so that the two players train completely independently.

    Args:
        player_id:   Index of the player this agent acts for (0 or 1).
        num_actions: Total number of actions in Hanabi.
        obs_dim:     Dimensionality of the OpenSpiel observation tensor.
        hidden_dim:  Hidden layer width for both networks.
        lr:          Learning rate for both network optimisers.
    """

    def __init__(
        self,
        player_id: int,
        num_actions: int,
        obs_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
    ):
        self.player_id = player_id
        self.num_actions = num_actions
        self.obs_dim = obs_dim

        # ------------------------------------------------------------------ #
        # Networks                                                             #
        # ------------------------------------------------------------------ #
        self.advantage_net = AdvantageNetwork(obs_dim, num_actions, hidden_dim)
        self.average_policy_net = AveragePolicyNetwork(obs_dim, num_actions, hidden_dim)

        # ------------------------------------------------------------------ #
        # Optimisers                                                           #
        # ------------------------------------------------------------------ #
        self.advantage_optimiser = torch.optim.Adam(
            self.advantage_net.parameters(), lr=lr
        )
        self.average_optimiser = torch.optim.Adam(
            self.average_policy_net.parameters(), lr=lr
        )

        # ------------------------------------------------------------------ #
        # Replay buffers                                                       #
        # Advantage buffer stores (obs, action, advantage_target) tuples.     #
        # Average buffer stores (obs, action, reach_weight) tuples.           #
        # ------------------------------------------------------------------ #
        self._advantage_buffer: list[tuple] = []
        self._average_buffer: list[tuple] = []

        # ------------------------------------------------------------------ #
        # Belief Engine / Opponent Model                                       #
        # Tracks a belief distribution over hidden state given observed        #
        # opponent actions.  Replace with your real implementation.           #
        # ------------------------------------------------------------------ #
        self._belief_engine: Any = None   # TODO: instantiate BeliefEngine here

    # ---------------------------------------------------------------------- #
    # Public interface expected by XDOSolver                                  #
    # ---------------------------------------------------------------------- #

    def train(
        self,
        iterations: int,
        opponent_context: dict,
    ) -> nn.Module:
        """
        Run `iterations` DeepCFR traversals to compute a Best Response against
        the opponent's current meta-strategy, then return the trained Average
        Policy Network.

        This method is called by XDOSolver._get_best_response() and entirely
        replaces the tabular TabularBestResponse from OpenSpiel.

        Args:
            iterations:
                Number of DeepCFR tree-traversal steps to perform.

            opponent_context:
                A dict with two keys, provided by XDOSolver:

                "policy_pool"  – list[NeuralNetworkPolicy]
                    The opponent's current pool of frozen policy snapshots.
                    During traversals, when it is the opponent's turn, you must
                    sample one of these policies (see `meta_strategy` below)
                    and call policy.action_probabilities(state) to pick their
                    action.  This is how the opponent's behaviour is modelled
                    during trajectory generation.

                "meta_strategy" – np.ndarray, shape (pool_size,), sums to 1
                    Probability weights over the opponent's policy pool.
                    Sample index k ~ Categorical(meta_strategy) to decide
                    which opponent policy governs each traversal.

        Returns:
            nn.Module – the AveragePolicyNetwork after training.
            XDOSolver will immediately deep-copy its state_dict() into a
            NeuralNetworkPolicy snapshot, so this module can continue to be
            mutated in future iterations without affecting the snapshot.

        Implementation outline (fill in each step)
        -------------------------------------------
        1.  TRAVERSAL LOOP  (repeat `iterations` times)
            a.  Reset the environment to a new Hanabi episode.
            b.  At each decision node for self.player_id:
                  - Build the observation tensor from state.observation_tensor().
                  - Query self.advantage_net to get current action advantages.
                  - Compute regret-matching probabilities.
                  - Store (obs, action, instantaneous_regret) in
                    self._advantage_buffer.
                  - Store (obs, action, reach_weight) in self._average_buffer.
            c.  At each decision node for the opponent:
                  - Sample k ~ Categorical(opponent_context["meta_strategy"]).
                  - Retrieve policy = opponent_context["policy_pool"][k].
                  - Sample action ~ policy.action_probabilities(state).
                  - Pass the observed action to self._belief_engine.update()
                    so the belief distribution over hidden cards is updated.

        2.  NETWORK UPDATES  (after every N traversals, or at the end)
            a.  Sample a mini-batch from self._advantage_buffer.
            b.  Minimise MSE loss between advantage_net(obs) and targets.
            c.  Sample a mini-batch from self._average_buffer.
            d.  Minimise cross-entropy loss on average_policy_net(obs).

        3.  RETURN
            return self.average_policy_net
            (XDOSolver wraps it in a NeuralNetworkPolicy snapshot immediately.)
        """
        raise NotImplementedError(
            "ODCFRAgent.train() is not yet implemented.  "
            "Follow the outline in the docstring above."
        )

    # ---------------------------------------------------------------------- #
    # Internal helpers (stubs)                                                 #
    # ---------------------------------------------------------------------- #

    def _traverse(self, state, reach_prob: float, opponent_context: dict):
        """
        Recursive DeepCFR tree traversal for self.player_id.

        At self.player_id nodes: collect advantage targets, update regret sums.
        At opponent nodes: sample action from opponent_context, update belief.
        At chance nodes: sample according to the game's transition dynamics.

        Args:
            state:            Current OpenSpiel Hanabi state.
            reach_prob:       Product of all action probabilities on the path
                              to this node (for reach-weighted averaging).
            opponent_context: Forwarded from train(); contains pool + weights.
        """
        raise NotImplementedError

    def _sample_opponent_action(self, state, opponent_context: dict) -> int:
        """
        Sample an action for the opponent at `state`.

        Steps:
          1.  Draw k ~ Categorical(opponent_context["meta_strategy"]).
          2.  policy = opponent_context["policy_pool"][k]
          3.  probs  = policy.action_probabilities(state)
          4.  Return an action sampled from probs.

        Args:
            state:            Current OpenSpiel Hanabi state.
            opponent_context: Contains "policy_pool" and "meta_strategy".

        Returns:
            int – sampled action index.
        """
        policy_pool   = opponent_context["policy_pool"]
        meta_strategy = opponent_context["meta_strategy"]

        # Sample which policy to use this traversal.
        k = int(np.random.choice(len(policy_pool), p=meta_strategy))
        policy = policy_pool[k]

        # Get the probability distribution over legal actions.
        action_probs = policy.action_probabilities(state)          # dict[int, float]
        actions      = list(action_probs.keys())
        probs        = np.array([action_probs[a] for a in actions])
        probs        = probs / probs.sum()                         # re-normalise for safety

        return int(np.random.choice(actions, p=probs))

    def _update_advantage_network(self):
        """
        Sample from self._advantage_buffer and perform a gradient step on
        self.advantage_net to minimise MSE against the stored advantage targets.
        """
        raise NotImplementedError

    def _update_average_network(self):
        """
        Sample from self._average_buffer and perform a gradient step on
        self.average_policy_net to minimise cross-entropy against the stored
        reach-weighted action targets.
        """
        raise NotImplementedError