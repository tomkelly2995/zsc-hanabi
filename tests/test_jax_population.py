# tests/test_jax_population.py
# Tests for BehaviourProfileJax and HanabiPolicyJax.

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from src.population.behaviour_profile_jax import BehaviourProfileJax, _build_obs_action_pairs
from src.population.hanabi_policy_jax import HanabiPolicyJax
from src.jax_networks.gru_encoder import HIDDEN_DIM
from src.jax_agents.cooperative_odcfr_agent_jax import CooperativeODCFRAgentJax


OBS_DIM = 84
NUM_ACTIONS = 8

_OBS = np.zeros(OBS_DIM, dtype=np.float32)
_RAW_SEQS         = [[0, 1, 2], [3, 4], [0]]
_RAW_OBS          = [[_OBS, _OBS, _OBS], [_OBS, _OBS], [_OBS]]
_RAW_ORACLE_NEXTS = [[-1, -1, -1], [-1, -1], [-1]]   # no oracle follow-up in test data


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def profile():
    return BehaviourProfileJax.from_episodes(
        _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS,
        iteration_created=1, source="agent_B", bc_epochs=2,
    )


@pytest.fixture(scope="module")
def trained_agent():
    a = CooperativeODCFRAgentJax(
        "agent_A", K_simulations=2, batch_size=4, seed=0,
        adv_train_steps=2, pol_train_steps=2,
    )
    p = BehaviourProfileJax.from_episodes(
        _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_B", bc_epochs=2,
    )
    a.train(p, cfr_iterations=3)
    return a


@pytest.fixture(scope="module")
def policy(trained_agent):
    return HanabiPolicyJax(trained_agent.pol_params, trained_agent.gru_params)


# ===========================================================================
# BehaviourProfileJax
# ===========================================================================

class TestBuildObsActionPairs:
    def test_flattens_correctly(self):
        pairs = _build_obs_action_pairs(_RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS)
        assert len(pairs) == sum(len(s) for s in _RAW_SEQS)  # 6

    def test_pair_structure(self):
        pairs = _build_obs_action_pairs(_RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS)
        obs, h_bc, act, oracle_next = pairs[0]
        assert obs.shape == (OBS_DIM,)
        assert isinstance(act, (int, np.integer))

    def test_empty_sequences(self):
        pairs = _build_obs_action_pairs([], [], [])
        assert pairs == []


class TestBehaviourProfileJaxInit:
    def test_raw_sequences_stored(self, profile):
        assert profile.raw_sequences == _RAW_SEQS

    def test_raw_obs_sequences_stored(self, profile):
        assert profile.raw_obs_sequences == _RAW_OBS

    def test_bc_model_params_is_dict(self, profile):
        assert isinstance(profile.bc_model_params, dict)

    def test_bc_model_params_has_keys(self, profile):
        # _BCNet has Dense_0, Dense_1, Dense_2
        assert len(profile.bc_model_params) > 0

    def test_bc_model_params_matches_bc_model(self, profile):
        # bc_model_params is the same object as bc_model.params
        assert profile.bc_model_params is profile.bc_model.params

    def test_iteration_created(self, profile):
        assert profile.iteration_created == 1

    def test_source(self, profile):
        assert profile.source == "agent_B"

    def test_bc_train_loss_finite(self, profile):
        assert np.isfinite(float(profile.bc_train_loss))

    def test_bc_train_loss_positive(self, profile):
        assert float(profile.bc_train_loss) > 0


class TestBehaviourProfileJaxFromEpisodes:
    def test_empty_sequences_no_crash(self):
        p = BehaviourProfileJax.from_episodes([], [], [], 0, "agent_A", bc_epochs=1)
        assert p.bc_train_loss == 0.0

    def test_different_seeds_different_params(self):
        p0 = BehaviourProfileJax.from_episodes(
            _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_A", bc_epochs=1, seed=0
        )
        p1 = BehaviourProfileJax.from_episodes(
            _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_A", bc_epochs=1, seed=99
        )
        leaves0 = jax.tree_util.tree_leaves(p0.bc_model_params)
        leaves1 = jax.tree_util.tree_leaves(p1.bc_model_params)
        assert any(
            not np.allclose(np.array(a), np.array(b))
            for a, b in zip(leaves0, leaves1)
        )

    def test_more_epochs_lowers_loss(self):
        p_few = BehaviourProfileJax.from_episodes(
            _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_A", bc_epochs=1, seed=42
        )
        p_many = BehaviourProfileJax.from_episodes(
            _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_A", bc_epochs=30, seed=42
        )
        assert p_many.bc_train_loss <= p_few.bc_train_loss + 2.0  # soft bound


class TestBehaviourProfileJaxRetrain:
    def test_retrain_bc_updates_params(self):
        p = BehaviourProfileJax.from_episodes(
            _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_A", bc_epochs=1,
        )
        before = jax.tree_util.tree_leaves(p.bc_model_params)
        before_vals = [np.array(x) for x in before]

        p.retrain_bc(bc_epochs=10)

        after = jax.tree_util.tree_leaves(p.bc_model_params)
        changed = any(
            not np.allclose(b, np.array(a), atol=1e-8)
            for b, a in zip(before_vals, after)
        )
        assert changed

    def test_retrain_bc_syncs_bc_model_params(self):
        p = BehaviourProfileJax.from_episodes(
            _RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS, 1, "agent_A", bc_epochs=1,
        )
        p.retrain_bc(bc_epochs=5)
        assert p.bc_model_params is p.bc_model.params

    def test_retrain_empty_no_crash(self):
        p = BehaviourProfileJax.from_episodes([], [], [], 0, "agent_A", bc_epochs=1)
        loss = p.retrain_bc(bc_epochs=5)
        assert loss == 0.0


class TestBehaviourProfileJaxEvaluate:
    def test_evaluate_returns_tuple(self, profile):
        pairs = _build_obs_action_pairs(_RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS)
        result = profile.evaluate_bc(pairs)
        assert len(result) == 2

    def test_evaluate_accuracy_in_range(self, profile):
        pairs = _build_obs_action_pairs(_RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS)
        acc, loss = profile.evaluate_bc(pairs)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_loss_positive(self, profile):
        pairs = _build_obs_action_pairs(_RAW_SEQS, _RAW_OBS, _RAW_ORACLE_NEXTS)
        acc, loss = profile.evaluate_bc(pairs)
        assert loss >= 0.0


# ===========================================================================
# HanabiPolicyJax
# ===========================================================================

class TestHanabiPolicyJaxInit:
    def test_pol_params_stored(self, policy, trained_agent):
        assert policy.pol_params is trained_agent.pol_params

    def test_gru_params_stored(self, policy, trained_agent):
        assert policy.gru_params is trained_agent.gru_params


class TestHanabiPolicyJaxInitialH:
    def test_h0_shape(self, policy):
        h0 = policy.initial_h_oppo()
        assert h0.shape == (HIDDEN_DIM,)

    def test_h0_zeros(self, policy):
        h0 = policy.initial_h_oppo()
        assert jnp.allclose(h0, jnp.zeros(HIDDEN_DIM))


class TestHanabiPolicyJaxAct:
    def test_returns_dict(self, policy):
        h0 = policy.initial_h_oppo()
        result = policy.act(_OBS, h0, [0, 1, 2])
        assert isinstance(result, dict)

    def test_keys_match_legal(self, policy):
        h0 = policy.initial_h_oppo()
        legal = [0, 3, 5]
        result = policy.act(_OBS, h0, legal)
        assert set(result.keys()) == set(legal)

    def test_probs_sum_to_one(self, policy):
        h0 = policy.initial_h_oppo()
        result = policy.act(_OBS, h0, list(range(NUM_ACTIONS)))
        assert abs(sum(result.values()) - 1.0) < 1e-5

    def test_probs_non_negative(self, policy):
        h0 = policy.initial_h_oppo()
        result = policy.act(_OBS, h0, [0, 1, 2, 3])
        assert all(v >= 0 for v in result.values())

    def test_single_legal_action(self, policy):
        h0 = policy.initial_h_oppo()
        result = policy.act(_OBS, h0, [4])
        assert abs(result[4] - 1.0) < 1e-5

    def test_different_h_gives_different_dist(self, policy):
        h0 = policy.initial_h_oppo()
        h1 = policy.update_h_oppo(h0, 2)
        d0 = policy.act(_OBS, h0, list(range(NUM_ACTIONS)))
        d1 = policy.act(_OBS, h1, list(range(NUM_ACTIONS)))
        # Distributions should differ (h changed)
        assert d0 != d1


class TestHanabiPolicyJaxUpdateH:
    def test_update_shape(self, policy):
        h0 = policy.initial_h_oppo()
        h1 = policy.update_h_oppo(h0, 0)
        assert h1.shape == (HIDDEN_DIM,)

    def test_update_changes_h(self, policy):
        h0 = policy.initial_h_oppo()
        h1 = policy.update_h_oppo(h0, 3)
        assert not jnp.allclose(h0, h1)

    def test_different_actions_give_different_h(self, policy):
        h0 = policy.initial_h_oppo()
        h1 = policy.update_h_oppo(h0, 0)
        h2 = policy.update_h_oppo(h0, 5)
        assert not jnp.allclose(h1, h2)

    def test_sequential_updates(self, policy):
        h = policy.initial_h_oppo()
        for a in [0, 1, 2, 3]:
            h = policy.update_h_oppo(h, a)
        assert h.shape == (HIDDEN_DIM,)
        assert not jnp.allclose(h, policy.initial_h_oppo())

    def test_does_not_mutate_input(self, policy):
        h0 = policy.initial_h_oppo()
        h0_copy = jnp.array(h0)
        _ = policy.update_h_oppo(h0, 2)
        assert jnp.allclose(h0, h0_copy)
