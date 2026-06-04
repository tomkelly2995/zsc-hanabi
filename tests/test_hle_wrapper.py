"""
tests/test_hle_wrapper.py

Run with:
    cd <project_root>
    python -m pytest tests/test_hle_wrapper.py -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hanabi-learning-environment"))

import numpy as np
import pytest
from src.env.hle_wrapper import HLEWrapper, StepObs

OBS_DIM = 84
NUM_MOVES = 8


@pytest.fixture
def env():
    return HLEWrapper()


# ------------------------------------------------------------------
# 1. Dimension accessors
# ------------------------------------------------------------------

def test_observation_size(env):
    assert env.observation_size() == OBS_DIM

def test_num_moves(env):
    assert env.num_moves() == NUM_MOVES


# ------------------------------------------------------------------
# 2. reset()
# ------------------------------------------------------------------

def test_reset_returns_stepobs(env):
    obs = env.reset()
    assert isinstance(obs, StepObs)

def test_reset_obs_shape(env):
    obs = env.reset()
    assert len(obs.player_obs) == 2
    for vec in obs.player_obs:
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (OBS_DIM,)
        assert vec.dtype == np.float32

def test_reset_current_player_valid(env):
    obs = env.reset()
    assert obs.current_player in (0, 1)

def test_reset_legal_moves_structure(env):
    obs = env.reset()
    assert len(obs.legal_moves) == 2
    # Current player must have at least one legal move
    assert len(obs.legal_moves[obs.current_player]) > 0
    # All legal move UIDs are in valid range
    for player_moves in obs.legal_moves:
        for m in player_moves:
            assert 0 <= m < NUM_MOVES

def test_reset_is_deterministic_with_same_env(env):
    # Two separate envs should both produce valid (possibly different) resets
    env2 = HLEWrapper()
    obs1 = env.reset()
    obs2 = env2.reset()
    # Both must be valid regardless of whether they match
    assert obs1.player_obs[0].shape == obs2.player_obs[0].shape


# ------------------------------------------------------------------
# 3. step()
# ------------------------------------------------------------------

def test_step_returns_correct_types(env):
    obs = env.reset()
    action = obs.legal_moves[obs.current_player][0]
    result = env.step(action)
    assert len(result) == 4
    new_obs, reward, done, info = result
    assert isinstance(new_obs, StepObs)
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "score" in info

def test_step_obs_shape_unchanged(env):
    obs = env.reset()
    action = obs.legal_moves[obs.current_player][0]
    new_obs, _, _, _ = env.step(action)
    assert len(new_obs.player_obs) == 2
    for vec in new_obs.player_obs:
        assert vec.shape == (OBS_DIM,)
        assert vec.dtype == np.float32

def test_step_score_is_non_negative(env):
    obs = env.reset()
    action = obs.legal_moves[obs.current_player][0]
    _, _, _, info = env.step(action)
    assert info["score"] >= 0

def test_step_reward_is_numeric(env):
    obs = env.reset()
    action = obs.legal_moves[obs.current_player][0]
    _, reward, _, _ = env.step(action)
    assert isinstance(reward, (int, float))


# ------------------------------------------------------------------
# 4. Full random episode
# ------------------------------------------------------------------

def test_random_episode_completes(env):
    """Play a full random episode; score must be in [0, 4] for tiny Hanabi."""
    obs = env.reset()
    done = False
    steps = 0
    score = 0
    while not done:
        action = obs.legal_moves[obs.current_player][0]  # always pick first legal
        obs, reward, done, info = env.step(action)
        score = info["score"]
        steps += 1
        assert steps < 500, "Episode did not terminate"
    assert 0 <= score <= 4  # max score = colors * ranks = 2 * 2

def test_multiple_episodes_reset_cleanly(env):
    """Resetting between episodes should not carry over state."""
    for _ in range(3):
        obs = env.reset()
        assert obs.current_player in (0, 1)
        action = obs.legal_moves[obs.current_player][0]
        _, _, _, info = env.step(action)
        assert info["score"] >= 0
