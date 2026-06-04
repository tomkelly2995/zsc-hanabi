"""
tests/test_bc_partner_model.py

Run with:
    cd <project_root>
    python -m pytest tests/test_bc_partner_model.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import pytest
from src.agents.bc_partner_model import BCPartnerModel

OBS_DIM = 84
NUM_ACTIONS = 8


def random_obs():
    return np.random.rand(OBS_DIM).astype(np.float32)

def random_pairs(n: int):
    return [(random_obs(), np.random.randint(0, NUM_ACTIONS)) for _ in range(n)]


@pytest.fixture
def model():
    return BCPartnerModel(obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)


# ------------------------------------------------------------------
# 1. Module structure
# ------------------------------------------------------------------

def test_is_nn_module(model):
    assert isinstance(model, nn.Module)

def test_has_trainable_parameters(model):
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)


# ------------------------------------------------------------------
# 2. forward()
# ------------------------------------------------------------------

def test_forward_shape_single(model):
    x = torch.randn(1, OBS_DIM)
    out = model(x)
    assert out.shape == (1, NUM_ACTIONS)

def test_forward_shape_batched(model):
    x = torch.randn(16, OBS_DIM)
    out = model(x)
    assert out.shape == (16, NUM_ACTIONS)

def test_forward_is_raw_logits(model):
    """forward() output should NOT sum to 1 — it's raw logits."""
    x = torch.randn(1, OBS_DIM)
    out = model(x).squeeze(0)
    assert not torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-3)


# ------------------------------------------------------------------
# 3. action_distribution()
# ------------------------------------------------------------------

def test_action_distribution_shape(model):
    obs = random_obs()
    probs = model.action_distribution(obs, legal_actions=[0, 1, 2])
    assert probs.shape == (NUM_ACTIONS,)

def test_action_distribution_sums_to_one(model):
    obs = random_obs()
    probs = model.action_distribution(obs, legal_actions=[1, 3, 5])
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

def test_action_distribution_illegal_are_zero(model):
    legal = [0, 2, 7]
    illegal = [i for i in range(NUM_ACTIONS) if i not in legal]
    probs = model.action_distribution(random_obs(), legal_actions=legal)
    for i in illegal:
        assert probs[i].item() == 0.0

def test_action_distribution_legal_are_positive(model):
    legal = [1, 4, 6]
    probs = model.action_distribution(random_obs(), legal_actions=legal)
    for i in legal:
        assert probs[i].item() > 0.0

def test_action_distribution_single_legal_action(model):
    probs = model.action_distribution(random_obs(), legal_actions=[5])
    assert torch.isclose(probs[5], torch.tensor(1.0), atol=1e-5)
    for i in range(NUM_ACTIONS):
        if i != 5:
            assert probs[i].item() == 0.0

def test_action_distribution_all_legal(model):
    probs = model.action_distribution(random_obs(), legal_actions=list(range(NUM_ACTIONS)))
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs > 0).all()

def test_action_distribution_accepts_tensor_obs(model):
    obs_t = torch.randn(OBS_DIM)
    probs = model.action_distribution(obs_t, legal_actions=[0, 1])
    assert probs.shape == (NUM_ACTIONS,)

def test_action_distribution_accepts_numpy_obs(model):
    obs_np = random_obs()
    probs = model.action_distribution(obs_np, legal_actions=[0, 1])
    assert probs.shape == (NUM_ACTIONS,)


# ------------------------------------------------------------------
# 4. train_supervised()
# ------------------------------------------------------------------

def test_train_supervised_runs_without_error(model):
    pairs = random_pairs(50)
    model.train_supervised(pairs, epochs=2)

def test_train_supervised_empty_pairs_no_crash(model):
    model.train_supervised([], epochs=5)

def test_train_supervised_single_pair(model):
    model.train_supervised([(random_obs(), 3)], epochs=2)

def test_train_supervised_smaller_than_batch(model):
    """Dataset of 10 pairs with batch_size=64 should not crash."""
    pairs = random_pairs(10)
    model.train_supervised(pairs, epochs=3, batch_size=64)

def test_train_supervised_model_in_eval_after_training(model):
    """train_supervised should leave the model in eval mode."""
    pairs = random_pairs(30)
    model.train_supervised(pairs, epochs=2)
    assert not model.training

def test_train_supervised_reduces_loss_on_deterministic_data(model):
    """
    Train on a dataset where obs always maps to action 0.
    After training, action 0 should have the highest probability.
    """
    fixed_obs = random_obs()
    pairs = [(fixed_obs, 0)] * 200
    model.train_supervised(pairs, epochs=30, lr=1e-2)
    probs = model.action_distribution(fixed_obs, legal_actions=list(range(NUM_ACTIONS)))
    assert probs.argmax().item() == 0

def test_train_supervised_weights_change(model):
    """Weights must actually update during training."""
    params_before = [p.clone() for p in model.parameters()]
    pairs = random_pairs(50)
    model.train_supervised(pairs, epochs=5)
    for before, after in zip(params_before, model.parameters()):
        if before.numel() > 1:   # skip any trivially small tensors
            assert not torch.allclose(before, after)
            break

def test_train_supervised_fresh_each_call(model):
    """
    Each call to train_supervised is independent: training twice on the
    same data should produce the same result as training once (since the
    optimizer resets and weights are the starting point each time).
    Specifically: two separate models trained identically should agree.
    """
    torch.manual_seed(0)
    model_a = BCPartnerModel()
    torch.manual_seed(0)
    model_b = BCPartnerModel()

    pairs = random_pairs(40)
    torch.manual_seed(1)
    model_a.train_supervised(pairs, epochs=5)
    torch.manual_seed(1)
    model_b.train_supervised(pairs, epochs=5)

    obs = torch.randn(1, OBS_DIM)
    assert torch.allclose(model_a(obs), model_b(obs), atol=1e-5)
