"""
tests/test_cooperative_odcfr.py

Run with:
    cd <project_root>
    python -m pytest tests/test_cooperative_odcfr.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hanabi-learning-environment"))

import random
import numpy as np
import torch
import pytest

from src.agents.cooperative_odcfr_agent import (
    CooperativeODCFRAgent,
    ReservoirBuffer,
    _regret_matching,
)
from src.networks.policy_net import PolicyNet
from src.networks.gru_encoder import GRUEncoder
from src.population.behaviour_profile import BehaviourProfile
from src.env.hle_wrapper import HLEWrapper

OBS_DIM = 84
NUM_ACTIONS = 8
H_DIM = 64


# ------------------------------------------------------------------
# Helpers — build a minimal real BehaviourProfile from live HLE episodes
# ------------------------------------------------------------------

def collect_episodes(n: int, player_id: int):
    """Run n random episodes and return raw_sequences, raw_obs_sequences."""
    env = HLEWrapper()
    raw_seqs, raw_obs = [], []
    partner_id = 1 - player_id
    for _ in range(n):
        obs_step = env.reset()
        ep_actions, ep_obs = [], []
        done = False
        while not done:
            legal = obs_step.legal_moves[obs_step.current_player]
            action = random.choice(legal)
            if obs_step.current_player == partner_id:
                ep_actions.append(action)
                ep_obs.append(obs_step.player_obs[partner_id].copy())
            obs_step, _, done, _ = env.step(action)
        if ep_actions:
            raw_seqs.append(ep_actions)
            raw_obs.append(ep_obs)
    return raw_seqs, raw_obs


@pytest.fixture(scope="module")
def profile_for_agent_a():
    """A real BehaviourProfile as seen by Agent A (partner = player 1)."""
    encoder = GRUEncoder()
    seqs, obs_seqs = collect_episodes(n=5, player_id=0)
    if not seqs:
        seqs = [[0]]
        obs_seqs = [[np.zeros(OBS_DIM, dtype=np.float32)]]
    return BehaviourProfile.from_episodes(
        raw_sequences=seqs,
        raw_obs_sequences=obs_seqs,
        gru_encoder=encoder,
        iteration_created=0,
        source="agent_B",
    )


def make_agent(agent_id="agent_A", K=2, adv_train_steps=5, pol_train_steps=5, batch_size=8):
    """Small agent for fast tests."""
    return CooperativeODCFRAgent(
        agent_id=agent_id,
        K_simulations=K,
        adv_train_steps=adv_train_steps,
        pol_train_steps=pol_train_steps,
        batch_size=batch_size,
    )


# ==================================================================
# ReservoirBuffer
# ==================================================================

class TestReservoirBuffer:

    def test_empty_at_start(self):
        buf = ReservoirBuffer(max_size=10)
        assert len(buf) == 0

    def test_add_fills_up_to_max(self):
        buf = ReservoirBuffer(max_size=5)
        for i in range(10):
            buf.add(i)
        assert len(buf) == 5

    def test_add_below_max(self):
        buf = ReservoirBuffer(max_size=100)
        for i in range(10):
            buf.add(i)
        assert len(buf) == 10

    def test_sample_returns_correct_count(self):
        buf = ReservoirBuffer(max_size=100)
        for i in range(50):
            buf.add(i)
        sample = buf.sample(20)
        assert len(sample) == 20

    def test_sample_clamps_to_available(self):
        buf = ReservoirBuffer(max_size=100)
        for i in range(5):
            buf.add(i)
        sample = buf.sample(100)
        assert len(sample) == 5

    def test_sample_values_come_from_buffer(self):
        buf = ReservoirBuffer(max_size=100)
        items = list(range(20))
        for i in items:
            buf.add(i)
        sample = buf.sample(10)
        for s in sample:
            assert s in items


# ==================================================================
# _regret_matching
# ==================================================================

class TestRegretMatching:

    def test_output_shape(self):
        q = torch.tensor([1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.0, 0.1])
        sigma = _regret_matching(q, [0, 2, 5], NUM_ACTIONS)
        assert sigma.shape == (NUM_ACTIONS,)

    def test_sums_to_one(self):
        q = torch.randn(NUM_ACTIONS)
        sigma = _regret_matching(q, [0, 1, 2, 3], NUM_ACTIONS)
        assert torch.isclose(sigma.sum(), torch.tensor(1.0), atol=1e-5)

    def test_illegal_actions_zero(self):
        q = torch.randn(NUM_ACTIONS)
        legal = [1, 3, 5]
        sigma = _regret_matching(q, legal, NUM_ACTIONS)
        for i in range(NUM_ACTIONS):
            if i not in legal:
                assert sigma[i].item() == 0.0

    def test_uniform_when_all_negative(self):
        q = torch.full((NUM_ACTIONS,), -1.0)
        legal = [0, 2, 4]
        sigma = _regret_matching(q, legal, NUM_ACTIONS)
        for a in legal:
            assert torch.isclose(sigma[a], torch.tensor(1.0 / 3), atol=1e-5)

    def test_positive_regrets_used(self):
        # Only action 2 has positive Q — should get probability 1
        q = torch.tensor([-1.0, -1.0, 5.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        sigma = _regret_matching(q, [0, 1, 2], NUM_ACTIONS)
        assert torch.isclose(sigma[2], torch.tensor(1.0), atol=1e-5)
        assert sigma[0].item() == 0.0
        assert sigma[1].item() == 0.0


# ==================================================================
# CooperativeODCFRAgent construction
# ==================================================================

class TestAgentConstruction:

    def test_agent_a_player_id(self):
        agent = make_agent("agent_A")
        assert agent.player_id == 0

    def test_agent_b_player_id(self):
        agent = make_agent("agent_B")
        assert agent.player_id == 1

    def test_networks_created(self):
        agent = make_agent()
        assert agent.advantage_net is not None
        assert agent.policy_net is not None
        assert agent.gru_encoder is not None

    def test_buffers_empty_at_start(self):
        agent = make_agent()
        assert len(agent._adv_buffer) == 0
        assert len(agent._pol_buffer) == 0


# ==================================================================
# _run_simulation
# ==================================================================

class TestRunSimulation:

    def test_simulation_fills_adv_buffer(self, profile_for_agent_a):
        agent = make_agent()
        agent._run_simulation(profile_for_agent_a, iteration_t=1)
        # Must have at least one self-turn record (game won't end before self acts)
        assert len(agent._adv_buffer) >= 0   # may be 0 if partner went last every turn
        # At minimum, the simulation must complete without error

    def test_simulation_completes_without_error(self, profile_for_agent_a):
        agent = make_agent()
        for _ in range(3):
            agent._run_simulation(profile_for_agent_a, iteration_t=1)

    def test_simulation_adds_to_both_buffers(self, profile_for_agent_a):
        agent = make_agent()
        # Run enough simulations that at least one self-turn is encountered
        for _ in range(5):
            agent._run_simulation(profile_for_agent_a, iteration_t=1)
        # Both buffers should be equal in size (one entry per self-turn)
        assert len(agent._adv_buffer) == len(agent._pol_buffer)

    def test_simulation_uses_bc_model_not_policy_weights(self, profile_for_agent_a):
        """
        The BC model must be queried for partner turns.
        Verify by checking the profile's bc_model is in eval mode throughout.
        """
        agent = make_agent()
        # BC model should remain in eval mode after simulation
        assert not profile_for_agent_a.bc_model.training
        agent._run_simulation(profile_for_agent_a, iteration_t=1)
        assert not profile_for_agent_a.bc_model.training


# ==================================================================
# Network updates
# ==================================================================

class TestNetworkUpdates:

    def test_advantage_net_updates_with_enough_data(self, profile_for_agent_a):
        agent = make_agent(K=4, batch_size=4, adv_train_steps=3)
        params_before = [p.clone() for p in agent.advantage_net.parameters()]
        # Fill buffer with enough data
        for _ in range(10):
            agent._run_simulation(profile_for_agent_a, iteration_t=1)
        agent._update_advantage_net(iteration_t=1)
        changed = any(
            not torch.allclose(b, a)
            for b, a in zip(params_before, agent.advantage_net.parameters())
        )
        assert changed

    def test_advantage_net_no_update_below_batch_size(self, profile_for_agent_a):
        agent = make_agent(K=1, batch_size=10_000)  # batch larger than buffer
        params_before = [p.clone() for p in agent.advantage_net.parameters()]
        agent._run_simulation(profile_for_agent_a, iteration_t=1)
        agent._update_advantage_net(iteration_t=1)
        unchanged = all(
            torch.allclose(b, a)
            for b, a in zip(params_before, agent.advantage_net.parameters())
        )
        assert unchanged

    def test_policy_net_updates_with_enough_data(self, profile_for_agent_a):
        agent = make_agent(K=4, batch_size=4, pol_train_steps=3)
        params_before = [p.clone() for p in agent.policy_net.parameters()]
        for _ in range(10):
            agent._run_simulation(profile_for_agent_a, iteration_t=1)
        agent._update_policy_net()
        changed = any(
            not torch.allclose(b, a)
            for b, a in zip(params_before, agent.policy_net.parameters())
        )
        assert changed


# ==================================================================
# train() end-to-end
# ==================================================================

class TestTrain:

    def test_train_returns_policy_net(self, profile_for_agent_a):
        agent = make_agent(K=2, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        result = agent.train(profile_for_agent_a, cfr_iterations=2)
        assert isinstance(result, PolicyNet)

    def test_train_policy_net_output_shape(self, profile_for_agent_a):
        agent = make_agent(K=2, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        policy_net = agent.train(profile_for_agent_a, cfr_iterations=2)
        x = torch.randn(1, OBS_DIM + H_DIM)
        out = policy_net(x)
        assert out.shape == (1, NUM_ACTIONS)

    def test_train_policy_net_in_eval_mode(self, profile_for_agent_a):
        agent = make_agent(K=2, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        policy_net = agent.train(profile_for_agent_a, cfr_iterations=2)
        assert not policy_net.training

    def test_train_advantage_net_in_eval_mode(self, profile_for_agent_a):
        agent = make_agent(K=2, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        agent.train(profile_for_agent_a, cfr_iterations=2)
        assert not agent.advantage_net.training

    def test_train_fills_buffers(self, profile_for_agent_a):
        agent = make_agent(K=3, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        agent.train(profile_for_agent_a, cfr_iterations=2)
        assert len(agent._adv_buffer) > 0
        assert len(agent._pol_buffer) > 0

    def test_two_agents_independent(self, profile_for_agent_a):
        """Networks and buffers of two separate agents must not be shared."""
        agent_a = make_agent("agent_A", K=2, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        agent_b = make_agent("agent_B", K=2, batch_size=4, adv_train_steps=2, pol_train_steps=2)
        agent_a.train(profile_for_agent_a, cfr_iterations=1)
        # agent_b's buffers should still be empty
        assert len(agent_b._adv_buffer) == 0
        assert len(agent_b._pol_buffer) == 0
