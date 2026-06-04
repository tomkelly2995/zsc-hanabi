"""
tests/test_hanabi_policy.py

Run with:
    cd <project_root>
    python -m pytest tests/test_hanabi_policy.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import torch
import pytest

from src.population.hanabi_policy import HanabiPolicy
from src.networks.policy_net import PolicyNet
from src.networks.gru_encoder import GRUEncoder

OBS_DIM = 84
NUM_ACTIONS = 8
H_DIM = 64


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

@pytest.fixture
def state_dicts():
    pol = PolicyNet(input_dim=OBS_DIM + H_DIM, num_actions=NUM_ACTIONS)
    gru = GRUEncoder(num_actions=NUM_ACTIONS)
    return pol.state_dict(), gru.state_dict()

@pytest.fixture
def policy(state_dicts):
    pi_sd, gru_sd = state_dicts
    return HanabiPolicy(pi_sd, gru_sd)

def random_obs():
    return np.random.rand(OBS_DIM).astype(np.float32)

def random_h():
    return torch.randn(H_DIM)


# ------------------------------------------------------------------
# 1. Construction and freezing
# ------------------------------------------------------------------

def test_policy_net_frozen(policy):
    for p in policy._policy_net.parameters():
        assert not p.requires_grad

def test_gru_frozen(policy):
    for p in policy._gru_encoder.parameters():
        assert not p.requires_grad

def test_policy_net_eval_mode(policy):
    assert not policy._policy_net.training

def test_gru_eval_mode(policy):
    assert not policy._gru_encoder.training

def test_independent_of_source_networks(state_dicts):
    """Mutating the source networks after construction must not affect the policy."""
    pi_sd, gru_sd = state_dicts
    pol_src = PolicyNet(input_dim=OBS_DIM + H_DIM, num_actions=NUM_ACTIONS)
    gru_src = GRUEncoder(num_actions=NUM_ACTIONS)
    policy = HanabiPolicy(pol_src.state_dict(), gru_src.state_dict())

    obs = random_obs()
    h = random_h()
    legal = [0, 1, 2]
    probs_before = policy.act(obs, h, legal)

    # Trash the source networks
    with torch.no_grad():
        for p in pol_src.parameters():
            p.fill_(999.0)

    probs_after = policy.act(obs, h, legal)
    assert probs_before == probs_after


# ------------------------------------------------------------------
# 2. initial_h_oppo
# ------------------------------------------------------------------

def test_initial_h_oppo_shape(policy):
    h = policy.initial_h_oppo()
    assert h.shape == (H_DIM,)

def test_initial_h_oppo_is_zero(policy):
    h = policy.initial_h_oppo()
    assert torch.all(h == 0)


# ------------------------------------------------------------------
# 3. act()
# ------------------------------------------------------------------

def test_act_returns_dict(policy):
    result = policy.act(random_obs(), random_h(), [0, 1, 2])
    assert isinstance(result, dict)

def test_act_keys_are_legal_actions(policy):
    legal = [1, 3, 5]
    result = policy.act(random_obs(), random_h(), legal)
    assert set(result.keys()) == set(legal)

def test_act_probs_sum_to_one(policy):
    legal = [0, 2, 4, 6]
    result = policy.act(random_obs(), random_h(), legal)
    assert abs(sum(result.values()) - 1.0) < 1e-5

def test_act_probs_non_negative(policy):
    legal = [0, 1, 7]
    result = policy.act(random_obs(), random_h(), legal)
    for p in result.values():
        assert p >= 0.0

def test_act_single_legal_action(policy):
    result = policy.act(random_obs(), random_h(), [4])
    assert abs(result[4] - 1.0) < 1e-5

def test_act_all_legal_actions(policy):
    legal = list(range(NUM_ACTIONS))
    result = policy.act(random_obs(), random_h(), legal)
    assert len(result) == NUM_ACTIONS
    assert abs(sum(result.values()) - 1.0) < 1e-5

def test_act_accepts_numpy_obs(policy):
    result = policy.act(random_obs(), random_h(), [0, 1])
    assert isinstance(result, dict)

def test_act_accepts_tensor_obs(policy):
    obs_t = torch.randn(OBS_DIM)
    result = policy.act(obs_t, random_h(), [0, 1])
    assert isinstance(result, dict)

def test_act_is_deterministic(policy):
    """Same inputs must give same output — no stochasticity in the network."""
    obs = random_obs()
    h = random_h()
    legal = [0, 2, 5]
    r1 = policy.act(obs, h, legal)
    r2 = policy.act(obs, h, legal)
    assert r1 == r2

def test_act_h_oppo_affects_output(policy):
    """Different h_oppo values must produce different action distributions."""
    obs = random_obs()
    legal = [0, 1, 2, 3]
    r1 = policy.act(obs, torch.zeros(H_DIM), legal)
    r2 = policy.act(obs, torch.ones(H_DIM), legal)
    assert r1 != r2

def test_act_does_not_mutate_h_oppo(policy):
    h = random_h()
    h_copy = h.clone()
    policy.act(random_obs(), h, [0, 1, 2])
    assert torch.allclose(h, h_copy)


# ------------------------------------------------------------------
# 4. update_h_oppo()
# ------------------------------------------------------------------

def test_update_h_oppo_shape(policy):
    h = policy.initial_h_oppo()
    h_new = policy.update_h_oppo(h, 0)
    assert h_new.shape == (H_DIM,)

def test_update_h_oppo_changes_state(policy):
    h = policy.initial_h_oppo()
    h_new = policy.update_h_oppo(h, 3)
    assert not torch.allclose(h, h_new)

def test_update_h_oppo_does_not_mutate_input(policy):
    h = random_h()
    h_copy = h.clone()
    policy.update_h_oppo(h, 2)
    assert torch.allclose(h, h_copy)

def test_update_h_oppo_all_actions_valid(policy):
    for action in range(NUM_ACTIONS):
        h = policy.initial_h_oppo()
        h_new = policy.update_h_oppo(h, action)
        assert h_new.shape == (H_DIM,)

def test_update_h_oppo_sequence_affects_act(policy):
    """h_oppo updated through a sequence must influence act() output."""
    obs = random_obs()
    legal = [0, 1, 2]
    h0 = policy.initial_h_oppo()
    h1 = policy.update_h_oppo(h0, 5)
    h2 = policy.update_h_oppo(h1, 3)
    r0 = policy.act(obs, h0, legal)
    r2 = policy.act(obs, h2, legal)
    assert r0 != r2
