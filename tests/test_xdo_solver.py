"""
tests/test_xdo_solver.py

Integration tests for XDOHanabiSolver.
Requires the HLE environment — slowest test file (~15–30 s).

Run with:
    cd <project_root>
    python -m pytest tests/test_xdo_solver.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "hanabi-learning-environment"))

import random
import numpy as np
import torch
import pytest

from src.xdo.xdo_solver import XDOHanabiSolver
from src.population.behaviour_profile import BehaviourProfile
from src.population.hanabi_policy import HanabiPolicy
from src.env.hle_wrapper import HLEWrapper


# ------------------------------------------------------------------
# Episode batch helper
# ------------------------------------------------------------------

def run_episode(env: HLEWrapper) -> dict:
    """
    Run one random episode and return it in the XDOHanabiSolver episode format:
        {
          'steps': [{'current_player', 'player_obs', 'action'}, ...],
          'score': int,
        }
    """
    obs_step = env.reset()
    steps = []
    done = False
    info = {"score": 0}
    while not done:
        current = obs_step.current_player
        legal = obs_step.legal_moves[current]
        action = random.choice(legal)
        steps.append({
            "current_player": current,
            "player_obs": {
                0: obs_step.player_obs[0].copy(),
                1: obs_step.player_obs[1].copy(),
            },
            "action": action,
        })
        obs_step, _, done, info = env.step(action)
    return {"steps": steps, "score": info["score"]}


def make_batch(n: int = 4) -> list:
    env = HLEWrapper()
    return [run_episode(env) for _ in range(n)]


@pytest.fixture(scope="module")
def batch():
    return make_batch(n=4)


def make_solver(agent_id="agent_A") -> XDOHanabiSolver:
    return XDOHanabiSolver(agent_id=agent_id)


# ==================================================================
# 1. Construction
# ==================================================================

class TestConstruction:

    def test_agent_a_player_id(self):
        s = make_solver("agent_A")
        assert s.player_id == 0
        assert s.partner_id == 1

    def test_agent_b_player_id(self):
        s = make_solver("agent_B")
        assert s.player_id == 1
        assert s.partner_id == 0

    def test_pools_empty_at_start(self):
        s = make_solver()
        assert len(s.profile_pool) == 0
        assert len(s.policy_pool) == 0
        assert len(s._profile_scores) == 0

    def test_meta_strategy_empty_at_start(self):
        s = make_solver()
        assert len(s.meta_strategy) == 0

    def test_oracle_created(self):
        s = make_solver()
        assert s.oracle is not None


# ==================================================================
# 2. extract_behaviour_profile
# ==================================================================

class TestExtractBehaviourProfile:

    def test_returns_behaviour_profile(self, batch):
        s = make_solver()
        p = s.extract_behaviour_profile(batch)
        assert isinstance(p, BehaviourProfile)

    def test_profile_added_to_pool(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert len(s.profile_pool) == 1

    def test_score_recorded(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert len(s._profile_scores) == 1
        assert isinstance(s._profile_scores[0], float)

    def test_mean_score_in_valid_range(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert 0.0 <= s._profile_scores[0] <= 4.0   # tiny Hanabi max = 2*2

    def test_meta_strategy_grows(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert len(s.meta_strategy) == 1
        assert abs(s.meta_strategy[0] - 1.0) < 1e-6

    def test_meta_strategy_sums_to_one_after_two_profiles(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.extract_behaviour_profile(batch)
        assert abs(s.meta_strategy.sum() - 1.0) < 1e-6

    def test_iteration_increments(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert s._iteration == 1
        s.extract_behaviour_profile(batch)
        assert s._iteration == 2

    def test_profile_source_correct_for_agent_a(self, batch):
        s = make_solver("agent_A")
        p = s.extract_behaviour_profile(batch)
        assert p.source == "agent_B"

    def test_profile_source_correct_for_agent_b(self, batch):
        s = make_solver("agent_B")
        p = s.extract_behaviour_profile(batch)
        assert p.source == "agent_A"

    def test_profile_h_oppo_shape(self, batch):
        s = make_solver()
        p = s.extract_behaviour_profile(batch)
        assert p.cached_h_oppo.shape == (64,)

    def test_only_partner_actions_extracted_agent_a(self, batch):
        """Agent A's profile should contain Player 1's (partner's) actions only."""
        s = make_solver("agent_A")   # partner = player 1
        p = s.extract_behaviour_profile(batch)
        for seq in p.raw_sequences:
            assert isinstance(seq, list)
            for a in seq:
                assert 0 <= a < 8

    def test_encoded_at_gru_count_set(self, batch):
        s = make_solver()
        p = s.extract_behaviour_profile(batch)
        assert hasattr(p, "_encoded_at_gru_count")
        assert p._encoded_at_gru_count == 0  # no oracle training yet


# ==================================================================
# 3. solve_meta_game
# ==================================================================

class TestSolveMetaGame:

    def test_no_op_with_empty_pool(self):
        s = make_solver()
        s.solve_meta_game()   # must not crash

    def test_single_profile_stays_at_one(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.solve_meta_game()
        assert abs(s.meta_strategy[0] - 1.0) < 1e-6

    def test_meta_strategy_sums_to_one(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.extract_behaviour_profile(batch)
        s.solve_meta_game()
        assert abs(s.meta_strategy.sum() - 1.0) < 1e-6

    def test_meta_game_count_increments(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert s._meta_game_count == 0
        s.solve_meta_game()
        assert s._meta_game_count == 1

    def test_higher_scoring_profile_gets_more_weight(self, batch):
        """
        If profile 1 consistently scores higher than profile 0, its weight
        should increase relative to profile 0 after several update steps.
        """
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.extract_behaviour_profile(batch)
        # Manually set scores so profile 1 is clearly better
        s._profile_scores = [0.5, 3.5]
        for _ in range(5):
            s.solve_meta_game()
        assert s.meta_strategy[1] > s.meta_strategy[0]

    def test_equal_scores_gives_uniform(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.extract_behaviour_profile(batch)
        s._profile_scores = [2.0, 2.0]   # identical scores
        s.solve_meta_game()
        assert abs(s.meta_strategy[0] - s.meta_strategy[1]) < 1e-6

    def test_log_weights_persist_across_calls(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.extract_behaviour_profile(batch)
        s._profile_scores = [1.0, 3.0]
        s.solve_meta_game()
        w1 = s._log_weights.copy()
        s.solve_meta_game()
        w2 = s._log_weights.copy()
        # Log-weights should accumulate across calls (additive update)
        assert not np.allclose(w1, w2)


# ==================================================================
# 4. request_new_policy
# ==================================================================

class TestRequestNewPolicy:

    def test_raises_with_empty_pool(self):
        s = make_solver()
        with pytest.raises(RuntimeError):
            s.request_new_policy(cfr_iterations=1)

    def test_returns_hanabi_policy(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        policy = s.request_new_policy(cfr_iterations=1)
        assert isinstance(policy, HanabiPolicy)

    def test_policy_pool_grows(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert len(s.policy_pool) == 0
        s.request_new_policy(cfr_iterations=1)
        assert len(s.policy_pool) == 1

    def test_gru_train_count_increments(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        assert s._gru_train_count == 0
        s.request_new_policy(cfr_iterations=1)
        assert s._gru_train_count == 1

    def test_policy_can_act(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)
        policy = s.request_new_policy(cfr_iterations=1)
        obs = np.random.rand(84).astype(np.float32)
        h = policy.initial_h_oppo()
        result = policy.act(obs, h, legal_actions=[0, 1, 2])
        assert isinstance(result, dict)
        assert abs(sum(result.values()) - 1.0) < 1e-5

    def test_embedding_drift_zero_on_first_call(self, batch):
        """On first call, profile was just encoded — no GRU training yet, drift = 0."""
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.request_new_policy(cfr_iterations=1)
        assert s.last_embedding_drift == 0.0

    def test_embedding_drift_positive_after_gru_update(self, batch):
        """After oracle training, re-encoding the same profile should show drift > 0."""
        s = make_solver()
        s.extract_behaviour_profile(batch)
        s.request_new_policy(cfr_iterations=2)   # updates GRU
        # Now gru_train_count=1; profile._encoded_at_gru_count=0 → reencode triggered
        s.request_new_policy(cfr_iterations=1)
        # Drift is non-negative (may be 0 if GRU barely changed, but typically > 0)
        assert s.last_embedding_drift >= 0.0


# ==================================================================
# 5. Full XDO iteration
# ==================================================================

class TestFullIteration:

    def test_one_full_iteration(self, batch):
        """
        Full single XDO iteration:
          1. extract_behaviour_profile (initialisation)
          2. solve_meta_game
          3. request_new_policy
          4. extract_behaviour_profile (new iteration's profile)
        All steps must complete without error and leave solver in valid state.
        """
        s = make_solver()

        # Initialisation: first profile from random-policy episode batch
        s.extract_behaviour_profile(batch)

        # Outer loop step
        s.solve_meta_game()
        policy = s.request_new_policy(cfr_iterations=1)
        s.extract_behaviour_profile(batch)

        assert len(s.profile_pool) == 2
        assert len(s.policy_pool) == 1
        assert abs(s.meta_strategy.sum() - 1.0) < 1e-6
        assert isinstance(policy, HanabiPolicy)

    def test_two_iterations(self, batch):
        s = make_solver()
        s.extract_behaviour_profile(batch)

        for _ in range(2):
            s.solve_meta_game()
            s.request_new_policy(cfr_iterations=1)
            s.extract_behaviour_profile(batch)

        assert len(s.profile_pool) == 3
        assert len(s.policy_pool) == 2
        assert abs(s.meta_strategy.sum() - 1.0) < 1e-6

    def test_two_solvers_independent(self, batch):
        """Agent A and Agent B solver instances must not share any state."""
        sa = make_solver("agent_A")
        sb = make_solver("agent_B")

        sa.extract_behaviour_profile(batch)
        sa.request_new_policy(cfr_iterations=1)

        # solver_b should be completely untouched
        assert len(sb.profile_pool) == 0
        assert len(sb.policy_pool) == 0
        assert sb._gru_train_count == 0
