"""
tests/test_networks.py
Tests for AdvantageNet and PolicyNet.

Run with:
    cd <project_root>
    python -m pytest tests/test_networks.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytest
from src.networks.advantage_net import AdvantageNet
from src.networks.policy_net import PolicyNet

INPUT_DIM = 148   # 84 obs + 64 h_oppo
NUM_ACTIONS = 8


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def make_input(batch: int = 1) -> torch.Tensor:
    return torch.randn(batch, INPUT_DIM)


# ==================================================================
# AdvantageNet
# ==================================================================

class TestAdvantageNet:

    @pytest.fixture
    def net(self):
        return AdvantageNet(input_dim=INPUT_DIM, num_actions=NUM_ACTIONS)

    def test_is_nn_module(self, net):
        assert isinstance(net, nn.Module)

    def test_output_shape_single(self, net):
        out = net(make_input(1))
        assert out.shape == (1, NUM_ACTIONS)

    def test_output_shape_batched(self, net):
        out = net(make_input(32))
        assert out.shape == (32, NUM_ACTIONS)

    def test_output_is_raw_no_softmax(self, net):
        """Values should NOT sum to 1 — output is raw, not a distribution."""
        out = net(make_input(1)).squeeze(0)
        assert not torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-3)

    def test_output_can_be_negative(self, net):
        """Regret values are unbounded — negative values are valid."""
        # Run enough batches that at least one negative value appears
        found_negative = False
        for _ in range(20):
            out = net(make_input(16))
            if (out < 0).any():
                found_negative = True
                break
        assert found_negative

    def test_gradients_flow(self, net):
        x = make_input(4)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_has_trainable_parameters(self, net):
        params = list(net.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_two_instances_are_independent(self):
        net1 = AdvantageNet()
        net2 = AdvantageNet()
        with torch.no_grad():
            net2.net[0].weight.fill_(0.0)
            net2.net[0].bias.fill_(0.0)
        x = make_input(1)
        assert not torch.allclose(net1(x), net2(x))

    def test_obs_and_h_oppo_concatenated_correctly(self, net):
        """Changing obs part should change output; so should changing h_oppo part."""
        base = make_input(1)
        obs_changed = base.clone()
        obs_changed[0, :84] += 10.0
        h_changed = base.clone()
        h_changed[0, 84:] += 10.0
        out_base = net(base)
        assert not torch.allclose(net(obs_changed), out_base)
        assert not torch.allclose(net(h_changed), out_base)


# ==================================================================
# PolicyNet
# ==================================================================

class TestPolicyNet:

    @pytest.fixture
    def net(self):
        return PolicyNet(input_dim=INPUT_DIM, num_actions=NUM_ACTIONS)

    def test_is_nn_module(self, net):
        assert isinstance(net, nn.Module)

    def test_output_shape_single(self, net):
        out = net(make_input(1))
        assert out.shape == (1, NUM_ACTIONS)

    def test_output_shape_batched(self, net):
        out = net(make_input(32))
        assert out.shape == (32, NUM_ACTIONS)

    def test_forward_is_raw_logits(self, net):
        """forward() should NOT produce a probability distribution."""
        out = net(make_input(1)).squeeze(0)
        assert not torch.isclose(out.sum(), torch.tensor(1.0), atol=1e-3)

    def test_gradients_flow(self, net):
        x = make_input(4)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for p in net.parameters():
            assert p.grad is not None

    def test_has_trainable_parameters(self, net):
        params = list(net.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_two_instances_are_independent(self):
        net1 = PolicyNet()
        net2 = PolicyNet()
        with torch.no_grad():
            net2.net[0].weight.fill_(0.0)
            net2.net[0].bias.fill_(0.0)
        x = make_input(1)
        assert not torch.allclose(net1(x), net2(x))

    # masked_probs tests

    def test_masked_probs_shape(self, net):
        x = make_input(1)
        probs = net.masked_probs(x, legal_actions=[0, 2, 5])
        assert probs.shape == (NUM_ACTIONS,)

    def test_masked_probs_sum_to_one(self, net):
        x = make_input(1)
        probs = net.masked_probs(x, legal_actions=[1, 3, 6])
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)

    def test_masked_probs_illegal_are_zero(self, net):
        legal = [0, 2, 5]
        illegal = [i for i in range(NUM_ACTIONS) if i not in legal]
        x = make_input(1)
        probs = net.masked_probs(x, legal_actions=legal)
        for i in illegal:
            assert probs[i].item() == 0.0

    def test_masked_probs_legal_are_nonzero(self, net):
        legal = [1, 4, 7]
        x = make_input(1)
        probs = net.masked_probs(x, legal_actions=legal)
        for i in legal:
            assert probs[i].item() > 0.0

    def test_masked_probs_single_legal_action(self, net):
        """When only one action is legal it must get probability 1.0."""
        x = make_input(1)
        probs = net.masked_probs(x, legal_actions=[3])
        assert torch.isclose(probs[3], torch.tensor(1.0), atol=1e-5)
        for i in range(NUM_ACTIONS):
            if i != 3:
                assert probs[i].item() == 0.0

    def test_masked_probs_all_actions_legal(self, net):
        """All-legal mask should still sum to 1 and all probs > 0."""
        x = make_input(1)
        probs = net.masked_probs(x, legal_actions=list(range(NUM_ACTIONS)))
        assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        assert (probs > 0).all()


# ==================================================================
# Cross-network: independence
# ==================================================================

def test_advantage_and_policy_weights_independent():
    """Updating AdvantageNet weights must not affect PolicyNet and vice versa."""
    adv = AdvantageNet()
    pol = PolicyNet()
    x = make_input(1)
    pol_out_before = pol(x).detach().clone()

    # Train adv for a step
    opt = torch.optim.SGD(adv.parameters(), lr=0.1)
    loss = adv(x).sum()
    loss.backward()
    opt.step()

    pol_out_after = pol(x).detach().clone()
    assert torch.allclose(pol_out_before, pol_out_after)
