# tests/test_jax_xdo_solver.py
# Tests for XDOHanabiSolverJax.
#
# Focuses on correctness of the outer-loop logic (meta-strategy, profile pool,
# policy pool) using tiny configs to keep runtime manageable.

import pytest
import numpy as np

from src.xdo.xdo_solver_jax import XDOHanabiSolverJax
from src.population.hanabi_policy_jax import HanabiPolicyJax
from src.population.behaviour_profile_jax import BehaviourProfileJax

OBS_DIM = 84

# ---------------------------------------------------------------------------
# Episode batch helpers
# ---------------------------------------------------------------------------

def _make_episode(partner_id: int, n_steps: int = 4, score: int = 3) -> dict:
    steps = []
    for i in range(n_steps):
        player = (i + partner_id) % 2   # alternate turns, partner goes first
        steps.append({
            "current_player": player,
            "player_obs": {
                0: np.zeros(OBS_DIM, dtype=np.float32),
                1: np.zeros(OBS_DIM, dtype=np.float32),
            },
            "action": i % 8,
        })
    return {"steps": steps, "score": score}


def _make_batch(partner_id: int, n_episodes: int = 4, score: int = 3) -> list:
    return [_make_episode(partner_id, score=score) for _ in range(n_episodes)]


# ---------------------------------------------------------------------------
# Fixture: fast solver
# ---------------------------------------------------------------------------

@pytest.fixture
def solver():
    return XDOHanabiSolverJax(
        "agent_A",
        K_simulations=2,
        adv_train_steps=2,
        pol_train_steps=2,
        seed=0,
    )


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------

class TestInit:
    def test_agent_id(self, solver):
        assert solver.agent_id == "agent_A"

    def test_player_id(self, solver):
        assert solver.player_id == 0

    def test_partner_id(self, solver):
        assert solver.partner_id == 1

    def test_pools_empty(self, solver):
        assert solver.profile_pool == []
        assert solver.policy_pool == []

    def test_meta_strategy_empty(self, solver):
        assert len(solver.meta_strategy) == 0

    def test_oracle_created(self, solver):
        assert solver.oracle is not None


# ---------------------------------------------------------------------------
# 2. extract_behaviour_profile
# ---------------------------------------------------------------------------

class TestExtractBehaviourProfile:
    def test_returns_profile(self, solver):
        batch = _make_batch(partner_id=1)
        p = solver.extract_behaviour_profile(batch, bc_epochs=2)
        assert isinstance(p, BehaviourProfileJax)

    def test_profile_appended(self, solver):
        solver2 = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1)
        solver2.extract_behaviour_profile(batch, bc_epochs=2)
        assert len(solver2.profile_pool) == 1

    def test_score_stored(self, solver):
        # Score is now from probe evaluation (not batch mean), so check it is a
        # valid finite float in the game's score range rather than == batch score.
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1, score=5)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        probe_score = s._profile_scores[0]
        assert isinstance(probe_score, float)
        assert 0.0 <= probe_score <= 10.0   # valid finite score range
        assert s.last_probe_score == pytest.approx(probe_score)

    def test_meta_strategy_updated(self, solver):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        assert len(s.meta_strategy) == 1
        assert abs(s.meta_strategy[0] - 1.0) < 1e-6

    def test_two_profiles_meta_strategy_sums_to_one(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        assert abs(s.meta_strategy.sum() - 1.0) < 1e-6

    def test_iteration_counter_increments(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        assert s._iteration == 1
        s.extract_behaviour_profile(batch, bc_epochs=2)
        assert s._iteration == 2

    def test_bc_loss_recorded(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        assert np.isfinite(s.last_bc_loss)

    def test_bc_accuracy_in_range(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        batch = _make_batch(partner_id=1)
        s.extract_behaviour_profile(batch, bc_epochs=2)
        assert 0.0 <= s.last_bc_accuracy <= 1.0

    def test_empty_episode_batch_no_crash(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile([], bc_epochs=1)
        assert len(s.profile_pool) == 1

    def test_episode_no_partner_action_no_crash(self):
        # All steps belong to self (player 0); partner (1) never acts.
        batch = [{"steps": [{"current_player": 0, "player_obs": {0: np.zeros(OBS_DIM), 1: np.zeros(OBS_DIM)}, "action": 0}], "score": 1}]
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(batch, bc_epochs=1)
        assert len(s.profile_pool) == 1


# ---------------------------------------------------------------------------
# 3. solve_meta_game
# ---------------------------------------------------------------------------

class TestSolveMetaGame:
    def test_single_profile_weight_one(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(_make_batch(1), bc_epochs=2)
        s.solve_meta_game()
        assert abs(s.meta_strategy[0] - 1.0) < 1e-6

    def test_two_profiles_sums_to_one(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(_make_batch(1, score=2), bc_epochs=2)
        s.extract_behaviour_profile(_make_batch(1, score=5), bc_epochs=2)
        s.solve_meta_game()
        assert abs(s.meta_strategy.sum() - 1.0) < 1e-6

    def test_better_profile_gets_more_weight(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(_make_batch(1, score=1), bc_epochs=2)
        s.extract_behaviour_profile(_make_batch(1, score=10), bc_epochs=2)
        # Run many meta-game iterations to let weights converge
        for _ in range(50):
            s.solve_meta_game()
        # Profile 1 (score=10) should dominate
        assert s.meta_strategy[1] > s.meta_strategy[0]

    def test_floor_respected(self):
        s = XDOHanabiSolverJax(
            "agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2,
            exploration_gamma=0.3, seed=0
        )
        s.extract_behaviour_profile(_make_batch(1, score=0), bc_epochs=2)
        s.extract_behaviour_profile(_make_batch(1, score=100), bc_epochs=2)
        for _ in range(100):
            s.solve_meta_game()
        floor = 0.3 / 2
        assert s.meta_strategy.min() >= floor - 1e-6

    def test_noop_when_empty(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.solve_meta_game()   # should not raise
        assert len(s.meta_strategy) == 0


# ---------------------------------------------------------------------------
# 4. request_new_policy
# ---------------------------------------------------------------------------

class TestRequestNewPolicy:
    def test_raises_when_pool_empty(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        with pytest.raises(RuntimeError):
            s.request_new_policy(cfr_iterations=2)

    def test_returns_hanabi_policy(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(_make_batch(1), bc_epochs=2)
        s.solve_meta_game()
        policy = s.request_new_policy(cfr_iterations=3)
        assert isinstance(policy, HanabiPolicyJax)

    def test_policy_appended_to_pool(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(_make_batch(1), bc_epochs=2)
        s.solve_meta_game()
        s.request_new_policy(cfr_iterations=3)
        assert len(s.policy_pool) == 1

    def test_policy_can_act(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        s.extract_behaviour_profile(_make_batch(1), bc_epochs=2)
        s.solve_meta_game()
        policy = s.request_new_policy(cfr_iterations=3)
        h0 = policy.initial_h_oppo()
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        dist = policy.act(obs, h0, [0, 1, 2])
        assert abs(sum(dist.values()) - 1.0) < 1e-5

    def test_two_iterations_two_policies(self):
        s = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        for _ in range(2):
            s.extract_behaviour_profile(_make_batch(1), bc_epochs=2)
            s.solve_meta_game()
            s.request_new_policy(cfr_iterations=3)
        assert len(s.policy_pool) == 2


# ---------------------------------------------------------------------------
# 5. Full mini outer-loop smoke test
# ---------------------------------------------------------------------------

class TestOuterLoop:
    def test_three_iterations_no_crash(self):
        solver_a = XDOHanabiSolverJax("agent_A", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=0)
        solver_b = XDOHanabiSolverJax("agent_B", K_simulations=2, adv_train_steps=2, pol_train_steps=2, seed=1)

        batch = _make_batch(partner_id=1, n_episodes=4)

        # Init
        solver_a.extract_behaviour_profile(batch, bc_epochs=2)
        solver_b.extract_behaviour_profile(batch, bc_epochs=2)

        for _ in range(3):
            solver_a.solve_meta_game()
            solver_b.solve_meta_game()
            solver_a.request_new_policy(cfr_iterations=3)
            solver_b.request_new_policy(cfr_iterations=3)
            solver_a.extract_behaviour_profile(batch, bc_epochs=2)
            solver_b.extract_behaviour_profile(batch, bc_epochs=2)

        assert len(solver_a.policy_pool) == 3
        assert len(solver_b.policy_pool) == 3
        assert len(solver_a.profile_pool) == 4  # 1 init + 3 iter
        assert len(solver_b.profile_pool) == 4
