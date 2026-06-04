# src/env/hle_wrapper.py
# HLEWrapper: thin wrapper around hanabi_learning_environment.rl_env.HanabiEnv.
# Verified: observation_size()=84, num_moves()=8 for tiny Hanabi config.
#
# Key design decisions:
#   - reset() and step() return a clean StepObs namedtuple instead of the raw
#     HLE dict, which contains all players' hands and must never reach agents.
#   - obs vectors are numpy float32 arrays, ready for torch.from_numpy().
#   - legal_moves are lists of ints (action UIDs in [0, num_moves())).
#   - reward is the incremental score delta from the last step.
#   - info always contains 'score' (current total score).

import json
import os
import numpy as np
from collections import namedtuple
from hanabi_learning_environment import rl_env

# Default config path — resolved relative to this file so it works regardless
# of where the process is launched from.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "configs", "tiny_hanabi.json"
)


def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

# Returned by reset() and step(). Safe to pass to individual agents:
# access obs.player_obs[player_id] for that agent's observation.
StepObs = namedtuple("StepObs", [
    "current_player",   # int: which player must act next
    "player_obs",       # list[np.ndarray]: obs vector per player, shape (84,)
    "legal_moves",      # list[list[int]]: legal action UIDs per player
])


class HLEWrapper:
    """
    Thin wrapper around HanabiEnv for the tiny 2-player, 2-colour config.

    Hides the raw HLE observation dict (which contains all players' hidden
    hands) and exposes only what agents are permitted to see under the
    no-information-sharing constraint.
    """

    def __init__(self, config: dict = None, config_path: str = None):
        if config is not None:
            cfg = config
        elif config_path is not None:
            cfg = _load_config(config_path)
        else:
            cfg = _load_config(_DEFAULT_CONFIG_PATH)
        self._env = rl_env.HanabiEnv(cfg)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> StepObs:
        """
        Reset for a new episode.

        Returns:
            StepObs with current_player, per-player obs vectors, and
            per-player legal move lists.
        """
        raw = self._env.reset()
        return self._extract(raw)

    def step(self, action: int):
        """
        Apply action (int UID) and advance the game.

        Args:
            action: int in [0, num_moves()) — action UID for the current player.

        Returns:
            obs   : StepObs
            reward: float — incremental score delta (>=0 on successful play)
            done  : bool
            info  : dict with key 'score' (cumulative team score so far)
        """
        raw_obs, reward, done, _ = self._env.step(action)
        obs = self._extract(raw_obs)
        info = {"score": self._env.state.score()}
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Dimension accessors (verified: 84, 8 for tiny Hanabi config)
    # ------------------------------------------------------------------

    def observation_size(self) -> int:
        """Length of the vectorised observation for one player."""
        return self._env.vectorized_observation_shape()[0]

    def num_moves(self) -> int:
        """Total number of possible moves (legal + illegal)."""
        return self._env.num_moves()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract(self, raw: dict) -> StepObs:
        """Convert raw HLE observation dict to a StepObs."""
        player_obs = []
        legal_moves = []
        for p_obs in raw["player_observations"]:
            player_obs.append(np.array(p_obs["vectorized"], dtype=np.float32))
            legal_moves.append(p_obs["legal_moves_as_int"])
        return StepObs(
            current_player=raw["current_player"],
            player_obs=player_obs,
            legal_moves=legal_moves,
        )
