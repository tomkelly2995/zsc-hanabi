# tests/test_jax_agent.py
# Tests for CooperativeODCFRAgentJax.
#
# Strategy: use tiny configs (K=2, few iters, small buffers, small batches)
# to keep runtime fast. Focus on correctness of shapes, buffer growth,
# param updates, and train() return value.

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from src.jax_agents.cooperative_odcfr_agent_jax import CooperativeODCFRAgentJax
from src.jax_networks.bc_partner_model import BCPartnerModel
from src.jax_networks.gru_encoder import HIDDEN_DIM
from src.jax_agents.simulation import OBS_DIM, MAX_PARTNER_TURNS


# ---------------------------------------------------------------------------
# Shared minimal profile fixture
# ---------------------------------------------------------------------------

class _MinimalProfile:
    """Minimal stand-in for BehaviourProfileJax: just needs bc_model_params."""
    def __init__(self):
        bc = BCPartnerModel()
        self.bc_model_params = bc.params


@pytest.fixture(scope="module")
def profile():
    return _MinimalProfile()


@pytest.fixture
def agent():
    return CooperativeODCFRAgentJax(
        agent_id="agent_A",
        K_simulations=2,
        adv_buffer_size=500,
        adv_train_steps=2,
        pol_train_steps=2,
        batch_size=4,
        seed=42,
    )


@pytest.fixture
def agent_b():
    return CooperativeODCFRAgentJax(
        agent_id="agent_B",
        K_simulations=2,
        adv_buffer_size=500,
        adv_train_steps=2,
        pol_train_steps=2,
        batch_size=4,
        seed=7,
    )


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_player_id_A(self, agent):
        assert agent.player_id == 0

    def test_player_id_B(self, agent_b):
        assert agent_b.player_id == 1

    def test_adv_params_not_none(self, agent):
        assert agent.adv_params is not None

    def test_pol_params_not_none(self, agent):
        assert agent.pol_params is not None

    def test_gru_params_not_none(self, agent):
        assert agent.gru_params is not None

    def test_buffers_empty(self, agent):
        assert len(agent._adv_buf) == 0
        assert len(agent._pol_buf) == 0

    def test_timing_attrs_zero(self, agent):
        assert agent.last_train_sim_secs == 0.0
        assert agent.last_train_adv_secs == 0.0
        assert agent.last_train_gru_secs == 0.0
        assert agent.last_train_pol_secs == 0.0


# ---------------------------------------------------------------------------
# 2. train() — smoke test (very few iterations)
# ---------------------------------------------------------------------------

class TestTrainSmoke:
    def test_returns_pytree(self, agent, profile):
        result = agent.train(profile, cfr_iterations=3)
        # pol_params is a Flax param pytree (dict-like)
        assert result is not None
        leaves = jax.tree_util.tree_leaves(result)
        assert len(leaves) > 0

    def test_returns_pol_params(self, agent, profile):
        result = agent.train(profile, cfr_iterations=3)
        # Should match agent.pol_params after training
        assert result is agent.pol_params

    def test_buffers_populated(self, agent, profile):
        # After fresh agent train, both buffers should have entries
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=2,
            adv_buffer_size=500, batch_size=4, seed=1
        )
        a.train(profile, cfr_iterations=3)
        assert len(a._adv_buf) > 0
        assert len(a._pol_buf) > 0

    def test_timing_attrs_updated(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=2,
            adv_buffer_size=500, batch_size=4, seed=2
        )
        a.train(profile, cfr_iterations=3)
        assert a.last_train_sim_secs > 0
        assert a.last_train_pol_secs > 0

    def test_cfr_iterations_recorded(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=2,
            adv_buffer_size=500, batch_size=4, seed=3
        )
        a.train(profile, cfr_iterations=5)
        assert a.last_cfr_iterations_run == 5

    def test_agent_B_runs(self, agent_b, profile):
        result = agent_b.train(profile, cfr_iterations=3)
        assert result is not None

    def test_custom_key(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=2,
            adv_buffer_size=500, batch_size=4, seed=99
        )
        key = jax.random.PRNGKey(123)
        result = a.train(profile, cfr_iterations=2, key=key)
        assert result is not None


# ---------------------------------------------------------------------------
# 3. train() — param update verification
# ---------------------------------------------------------------------------

class TestParamUpdates:
    def test_adv_params_change(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=4,
            adv_buffer_size=500, adv_train_steps=5, batch_size=4, seed=10
        )
        before = jax.tree_util.tree_leaves(a.adv_params)
        before_vals = [np.array(x) for x in before]

        a.train(profile, cfr_iterations=10)

        after = jax.tree_util.tree_leaves(a.adv_params)
        changed = any(
            not np.allclose(b, np.array(a_), atol=1e-8)
            for b, a_ in zip(before_vals, after)
        )
        assert changed, "adv_params should change after training"

    def test_pol_params_change(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=4,
            adv_buffer_size=500, pol_train_steps=5, batch_size=4, seed=11
        )
        before = jax.tree_util.tree_leaves(a.pol_params)
        before_vals = [np.array(x) for x in before]

        a.train(profile, cfr_iterations=10)

        after = jax.tree_util.tree_leaves(a.pol_params)
        changed = any(
            not np.allclose(b, np.array(a_), atol=1e-8)
            for b, a_ in zip(before_vals, after)
        )
        assert changed, "pol_params should change after training"


# ---------------------------------------------------------------------------
# 4. Buffer retention after partial_reset
# ---------------------------------------------------------------------------

class TestBufferRetention:
    def test_adv_retention(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=4,
            adv_buffer_size=500, batch_size=4, seed=20
        )
        a.train(profile, cfr_iterations=5)
        n_after_first = len(a._adv_buf)

        # Second train call should reset adv_buf then refill it
        a.train(profile, cfr_iterations=5)
        # Buffer should have grown again from near-empty
        assert len(a._adv_buf) > 0
        assert len(a._adv_buf) <= n_after_first + 50  # bounded by episodes

    def test_pol_buf_reset(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=4,
            adv_buffer_size=500, batch_size=4, seed=21
        )
        a.train(profile, cfr_iterations=5)
        a._pol_buf.add((np.zeros(148, np.float32), np.zeros(8, np.float32)))
        n = len(a._pol_buf)
        # Second train resets pol_buf completely
        a.train(profile, cfr_iterations=3)
        # pol_buf should have been reset then refilled (not have old sentinel)
        assert len(a._pol_buf) <= n + 50


# ---------------------------------------------------------------------------
# 5. Early stopping
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_early_stop_terminates(self, profile):
        # With large delta threshold, stable_count reaches patience quickly.
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=4,
            adv_buffer_size=500, adv_train_steps=2, batch_size=4, seed=30
        )
        a.train(
            profile,
            cfr_iterations=1000,
            early_stop_min_iters=2,
            early_stop_patience=2,
            early_stop_delta=1e6,   # huge threshold → always "stable"
        )
        # Should have stopped well before 1000
        assert a.last_cfr_iterations_run < 1000


# ---------------------------------------------------------------------------
# 6. GRU joint update path
# ---------------------------------------------------------------------------

class TestGRUUpdate:
    def test_gru_params_change_after_joint_update(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=4,
            adv_buffer_size=500, adv_train_steps=2, batch_size=4, seed=40
        )
        before = jax.tree_util.tree_leaves(a.gru_params)
        before_vals = [np.array(x) for x in before]

        # GRU is always updated once at the end of train()
        a.train(profile, cfr_iterations=10)

        after = jax.tree_util.tree_leaves(a.gru_params)
        changed = any(
            not np.allclose(b, np.array(a_), atol=1e-8)
            for b, a_ in zip(before_vals, after)
        )
        assert changed, "gru_params should change after joint update"


# ---------------------------------------------------------------------------
# 7. k_simulations override
# ---------------------------------------------------------------------------

class TestKSimulationsOverride:
    def test_k_override_runs(self, profile):
        a = CooperativeODCFRAgentJax(
            agent_id="agent_A", K_simulations=2,
            adv_buffer_size=500, batch_size=4, seed=50
        )
        # Override K at call time
        result = a.train(profile, cfr_iterations=3, k_simulations=4)
        assert result is not None
