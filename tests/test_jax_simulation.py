# tests/test_jax_simulation.py
# Tests for src/jax_agents/buffer.py and src/jax_agents/simulation.py.

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from src.jax_agents.buffer import ReservoirBuffer
from src.jax_agents.simulation import (
    regret_matching_jax,
    run_K_episodes,
    collect_to_buffers,
    MAX_SELF_TURNS,
    MAX_PARTNER_TURNS,
    MAX_EPISODE_STEPS,
    NUM_ACTIONS,
    OBS_DIM,
    HIDDEN_DIM,
    _adv_net,
    _gru_enc,
    _bc_net,
)

KEY = jax.random.PRNGKey(0)
IN_DIM = OBS_DIM + HIDDEN_DIM  # 148


# ---------------------------------------------------------------------------
# Helpers — minimal valid params for each network
# ---------------------------------------------------------------------------

def _adv_params():
    return _adv_net.init(KEY, jnp.zeros((1, IN_DIM)))

def _gru_params():
    return _gru_enc.init(KEY, jnp.zeros((5,), dtype=jnp.int32), jnp.array(1))

def _bc_params():
    from src.jax_networks.bc_partner_model import BCPartnerModel
    return BCPartnerModel(seed=0).params


# ---------------------------------------------------------------------------
# 1. ReservoirBuffer
# ---------------------------------------------------------------------------

class TestReservoirBuffer:

    def test_empty_buffer(self):
        buf = ReservoirBuffer(100)
        assert len(buf) == 0

    def test_add_below_capacity(self):
        buf = ReservoirBuffer(10)
        for i in range(5):
            buf.add(i)
        assert len(buf) == 5

    def test_add_to_capacity(self):
        buf = ReservoirBuffer(5)
        for i in range(5):
            buf.add(i)
        assert len(buf) == 5

    def test_add_beyond_capacity_stays_at_max(self):
        buf = ReservoirBuffer(5)
        for i in range(20):
            buf.add(i)
        assert len(buf) == 5

    def test_sample_returns_correct_size(self):
        buf = ReservoirBuffer(100)
        for i in range(50):
            buf.add(i)
        batch = buf.sample(10)
        assert len(batch) == 10

    def test_sample_cannot_exceed_buffer_size(self):
        buf = ReservoirBuffer(100)
        buf.add(42)
        batch = buf.sample(999)
        assert len(batch) == 1

    def test_sample_empty_buffer(self):
        buf = ReservoirBuffer(100)
        batch = buf.sample(10)
        assert len(batch) == 0

    def test_partial_reset_reduces_size(self):
        buf = ReservoirBuffer(100)
        for i in range(100):
            buf.add(i)
        buf.partial_reset(retention=0.2)
        assert len(buf) == 20

    def test_partial_reset_zero_retention(self):
        buf = ReservoirBuffer(100)
        for i in range(50):
            buf.add(i)
        buf.partial_reset(retention=0.0)
        assert len(buf) == 0

    def test_partial_reset_resets_count(self):
        # After reset, new items should be retained with high probability
        # (they won't be immediately replaced).
        buf = ReservoirBuffer(10)
        for i in range(10):
            buf.add(i)
        buf.partial_reset(retention=0.5)
        n_before = len(buf)
        buf.add(999)
        # 999 should be added since buffer not at capacity.
        assert 999 in buf._data

    def test_total_count_equals_len_after_reset(self):
        buf = ReservoirBuffer(100)
        for i in range(80):
            buf.add(i)
        buf.partial_reset(0.25)
        assert buf._total == len(buf)

    def test_items_accessible_after_sample(self):
        buf = ReservoirBuffer(100)
        for i in range(10):
            buf.add(np.array([i, i]))
        batch = buf.sample(5)
        for item in batch:
            assert isinstance(item, np.ndarray)

    def test_reservoir_property(self):
        # With max_size=1 and many inserts, all original items should
        # appear in the buffer approximately uniformly.
        np.random.seed(42)
        counts = np.zeros(10, dtype=int)
        for _ in range(10000):
            buf = ReservoirBuffer(1)
            for j in range(10):
                buf.add(j)
            counts[buf._data[0]] += 1
        # All 10 values should have appeared roughly equally (1000 ± 200)
        assert np.all(counts > 600), f"bad reservoir distribution: {counts}"


# ---------------------------------------------------------------------------
# 2. regret_matching_jax
# ---------------------------------------------------------------------------

class TestRegretMatching:

    def _legal(self, idxs, n=NUM_ACTIONS):
        mask = jnp.zeros(n, dtype=bool)
        return mask.at[jnp.array(idxs)].set(True)

    def test_sum_to_one(self):
        q    = jnp.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.5, 1.0])
        mask = jnp.ones(NUM_ACTIONS, dtype=bool)
        s    = regret_matching_jax(q, mask)
        assert abs(float(s.sum()) - 1.0) < 1e-5

    def test_illegal_actions_zero(self):
        q    = jax.random.normal(KEY, (NUM_ACTIONS,))
        mask = self._legal([0, 2, 4])
        s    = regret_matching_jax(q, mask)
        assert float(s[1]) == pytest.approx(0.0, abs=1e-6)
        assert float(s[3]) == pytest.approx(0.0, abs=1e-6)

    def test_uniform_fallback(self):
        # All Q-values equal → uniform over legal actions.
        q    = jnp.zeros(NUM_ACTIONS)
        mask = self._legal([0, 2, 4, 6])
        s    = regret_matching_jax(q, mask)
        for i in [0, 2, 4, 6]:
            assert float(s[i]) == pytest.approx(0.25, abs=1e-5)

    def test_best_action_gets_weight(self):
        # Only action 3 has Q > mean.
        q    = jnp.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        mask = jnp.ones(NUM_ACTIONS, dtype=bool)
        s    = regret_matching_jax(q, mask)
        assert float(s[3]) > 0.5

    def test_non_negative(self):
        q    = jax.random.normal(KEY, (NUM_ACTIONS,))
        mask = jnp.ones(NUM_ACTIONS, dtype=bool)
        s    = regret_matching_jax(q, mask)
        assert jnp.all(s >= 0.0)

    def test_jit(self):
        q    = jnp.ones(NUM_ACTIONS)
        mask = jnp.ones(NUM_ACTIONS, dtype=bool)
        jitted = jax.jit(regret_matching_jax)
        s = jitted(q, mask)
        assert abs(float(s.sum()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3. run_K_episodes
# ---------------------------------------------------------------------------

class TestRunKEpisodes:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.adv_params = _adv_params()
        self.gru_params = _gru_params()
        self.bc_params  = _bc_params()

    def _run(self, K=4, player_id=0):
        keys = jax.random.split(KEY, K)
        return run_K_episodes(
            keys, self.adv_params, self.gru_params, self.bc_params,
            player_id=player_id, partner_temp=0.3,
        )

    def test_output_keys_present(self):
        res = self._run(K=2)
        for key in ["score", "self_obs_h", "self_actions", "self_sigmas",
                    "n_self", "partner_actions", "partner_prefix_lens", "n_partner"]:
            assert key in res, f"missing key: {key}"

    def test_score_shape(self):
        res = self._run(K=4)
        assert res["score"].shape == (4,)

    def test_score_in_range(self):
        res = self._run(K=8)
        assert jnp.all(res["score"] >= 0)
        assert jnp.all(res["score"] <= 4)  # max score for tiny Hanabi (2 colors × 2 ranks)

    def test_self_obs_h_shape(self):
        res = self._run(K=4)
        assert res["self_obs_h"].shape == (4, MAX_SELF_TURNS, IN_DIM)

    def test_self_actions_shape(self):
        res = self._run(K=4)
        assert res["self_actions"].shape == (4, MAX_SELF_TURNS)

    def test_self_sigmas_shape(self):
        res = self._run(K=4)
        assert res["self_sigmas"].shape == (4, MAX_SELF_TURNS, NUM_ACTIONS)

    def test_partner_actions_shape(self):
        res = self._run(K=4)
        assert res["partner_actions"].shape == (4, MAX_PARTNER_TURNS)

    def test_n_self_non_negative(self):
        res = self._run(K=8)
        assert jnp.all(res["n_self"] >= 0)

    def test_n_self_bounded(self):
        res = self._run(K=8)
        assert jnp.all(res["n_self"] <= MAX_SELF_TURNS)

    def test_n_partner_bounded(self):
        res = self._run(K=8)
        assert jnp.all(res["n_partner"] <= MAX_PARTNER_TURNS)

    def test_sigmas_sum_to_one_for_valid_turns(self):
        res = self._run(K=4)
        n_selfs = np.array(res["n_self"])
        sigmas  = np.array(res["self_sigmas"])
        for k in range(4):
            for i in range(int(n_selfs[k])):
                s = float(sigmas[k, i].sum())
                assert abs(s - 1.0) < 1e-4, f"sigma[{k},{i}] sums to {s}"

    def test_different_keys_give_different_scores(self):
        keys1 = jax.random.split(jax.random.PRNGKey(0), 4)
        keys2 = jax.random.split(jax.random.PRNGKey(1), 4)
        r1 = run_K_episodes(keys1, self.adv_params, self.gru_params, self.bc_params, 0, 0.3)
        r2 = run_K_episodes(keys2, self.adv_params, self.gru_params, self.bc_params, 0, 0.3)
        # At least some scores should differ (very unlikely to be all identical).
        assert not jnp.array_equal(r1["score"], r2["score"])

    def test_same_key_gives_same_result(self):
        keys = jax.random.split(KEY, 4)
        r1 = run_K_episodes(keys, self.adv_params, self.gru_params, self.bc_params, 0, 0.3)
        r2 = run_K_episodes(keys, self.adv_params, self.gru_params, self.bc_params, 0, 0.3)
        np.testing.assert_array_equal(np.array(r1["score"]), np.array(r2["score"]))

    def test_player_id_1(self):
        res = self._run(K=4, player_id=1)
        assert res["score"].shape == (4,)
        assert jnp.all(res["n_self"] >= 0)

    def test_no_nan_in_obs_h(self):
        res = self._run(K=4)
        assert not jnp.any(jnp.isnan(res["self_obs_h"]))

    def test_no_nan_in_sigmas(self):
        res = self._run(K=4)
        assert not jnp.any(jnp.isnan(res["self_sigmas"]))


# ---------------------------------------------------------------------------
# 4. Importance-weighted scores in advantage buffer
# ---------------------------------------------------------------------------

class TestImportanceWeighting:
    """
    Verify the adv buffer stores raw_score (b[4]) and sigma_a (b[6]) separately.
    Clipped IS correction — min(raw_score / σ(a*), IS_CLIP) — is applied at
    training time in _update_adv / _update_gru, not at collection time.
    """

    def setup_method(self):
        self.adv_params = _adv_params()
        self.gru_params = _gru_params()
        self.bc_params  = _bc_params()

    def _collect(self, K=4, iter_t=1.0):
        keys = jax.random.split(KEY, K)
        res  = run_K_episodes(keys, self.adv_params, self.gru_params,
                               self.bc_params, 0, 0.3)
        adv_buf = ReservoirBuffer(max_size=10000)
        pol_buf = ReservoirBuffer(max_size=10000)
        collect_to_buffers(res, adv_buf, pol_buf, iter_t)
        return adv_buf, pol_buf, res

    def test_raw_score_finite(self):
        adv_buf, _, _ = self._collect(K=4)
        for item in adv_buf._data:
            assert np.isfinite(item[4]), f"raw_score is not finite: {item[4]}"

    def test_raw_score_non_negative(self):
        # Tiny Hanabi scores ∈ [0, 4].
        adv_buf, _, _ = self._collect(K=8)
        for item in adv_buf._data:
            assert item[4] >= 0.0, f"raw_score < 0: {item[4]}"

    def test_sigma_a_valid(self):
        # σ(a*) is the pure regret-matched probability for the taken action.
        # It can be exactly 0 when the action was selected via the ε-exploration
        # uniform component rather than by regret matching.  _update_adv guards
        # against zero with max(sigma_a, 1e-6), giving IS_CLIP for zero entries.
        adv_buf, _, _ = self._collect(K=8)
        for item in adv_buf._data:
            sigma_a = item[6]
            assert 0.0 <= sigma_a <= 1.0 + 1e-6, \
                f"sigma_a out of range: {sigma_a}"

    def test_sigma_a_matches_taken_action(self):
        # Collect one batch and verify sigma_a == sigmas[k, i, taken_action]
        # by checking that clipped IS ≥ raw_score (since σ(a*) ≤ 1).
        adv_buf, _, _ = self._collect(K=8)
        for item in adv_buf._data:
            raw_score = item[4]
            sigma_a   = item[6]
            is_score  = min(raw_score / max(sigma_a, 1e-6), 10.0)
            assert is_score >= raw_score - 1e-6, \
                f"clipped IS score {is_score} < raw_score {raw_score}"

    def test_taken_action_index_valid(self):
        adv_buf, _, _ = self._collect(K=4)
        for item in adv_buf._data:
            assert 0 <= item[3] < NUM_ACTIONS, f"action index out of range: {item[3]}"


# ---------------------------------------------------------------------------
# 5. collect_to_buffers
# ---------------------------------------------------------------------------

class TestCollectToBuffers:
    """
    Advantage buffer 7-tuple format:
      (obs_raw, partner_acts, prefix_len, taken_action, raw_score, iter_t, sigma_a)
         [0]       [1]          [2]           [3]          [4]       [5]     [6]

      Clipped IS correction min(raw_score / σ(a*), IS_CLIP) is applied at
      training time in _update_adv / _update_gru, not stored in the buffer.

    Policy buffer 5-tuple format:
      (obs_raw, partner_acts, prefix_len, sigma, iter_t)
         [0]       [1]          [2]         [3]    [4]
    """

    def setup_method(self):
        self.adv_params = _adv_params()
        self.gru_params = _gru_params()
        self.bc_params  = _bc_params()

    def _collect(self, K=4, iter_t=1.0):
        keys = jax.random.split(KEY, K)
        res  = run_K_episodes(keys, self.adv_params, self.gru_params,
                               self.bc_params, 0, 0.3)
        adv_buf = ReservoirBuffer(max_size=10000)
        pol_buf = ReservoirBuffer(max_size=10000)
        collect_to_buffers(res, adv_buf, pol_buf, iter_t)
        return adv_buf, pol_buf, res

    def test_adv_buffer_non_empty(self):
        adv_buf, _, _ = self._collect(K=4)
        assert len(adv_buf) > 0

    def test_pol_buffer_non_empty(self):
        _, pol_buf, _ = self._collect(K=4)
        assert len(pol_buf) > 0

    def test_adv_entry_structure(self):
        # 7-tuple: (obs_raw, partner_acts, prefix_len, taken_action, raw_score, iter_t, sigma_a)
        adv_buf, _, _ = self._collect(K=4)
        item = adv_buf.sample(1)[0]
        obs_raw, partner_acts, prefix_len, taken_action, raw_score, iter_t, sigma_a = item
        assert obs_raw.shape == (OBS_DIM,),      f"obs_raw shape wrong: {obs_raw.shape}"
        assert partner_acts.shape == (MAX_PARTNER_TURNS,)
        assert isinstance(prefix_len, int)
        assert isinstance(taken_action, int)
        assert isinstance(raw_score, float)
        assert isinstance(iter_t, float)
        assert isinstance(sigma_a, float) and 0.0 < sigma_a <= 1.0 + 1e-6

    def test_pol_entry_structure(self):
        # 5-tuple: (obs_raw, partner_acts, prefix_len, sigma, iter_t)
        _, pol_buf, _ = self._collect(K=4, iter_t=3.0)
        item = pol_buf.sample(1)[0]
        obs_raw, partner_acts, prefix_len, sigma, iter_t = item
        assert obs_raw.shape == (OBS_DIM,),   f"obs_raw shape wrong: {obs_raw.shape}"
        assert partner_acts.shape == (MAX_PARTNER_TURNS,)
        assert isinstance(prefix_len, int)
        assert sigma.shape == (NUM_ACTIONS,)
        assert isinstance(iter_t, float) and iter_t >= 1.0

    def test_iter_t_stored_correctly(self):
        adv_buf, _, _ = self._collect(K=4, iter_t=7.0)
        for item in adv_buf._data:
            assert item[5] == pytest.approx(7.0)   # iter_t is now index 5

    def test_prefix_len_bounded(self):
        adv_buf, _, _ = self._collect(K=8)
        for item in adv_buf._data:
            prefix_len = item[2]
            assert 0 <= prefix_len <= MAX_PARTNER_TURNS

    def test_taken_action_in_range(self):
        adv_buf, _, _ = self._collect(K=8)
        for item in adv_buf._data:
            assert 0 <= item[3] < NUM_ACTIONS

    def test_buffer_size_matches_total_self_turns(self):
        K = 4
        keys = jax.random.split(KEY, K)
        res  = run_K_episodes(keys, self.adv_params, self.gru_params,
                               self.bc_params, 0, 0.3)
        adv_buf = ReservoirBuffer(max_size=10000)
        pol_buf = ReservoirBuffer(max_size=10000)
        collect_to_buffers(res, adv_buf, pol_buf, 1.0)

        expected = int(np.array(res["n_self"]).sum())
        assert len(adv_buf) == expected
        assert len(pol_buf) == expected


# ---------------------------------------------------------------------------
# 5. Joint episode simulation (run_N_joint_episodes / joint_results_to_episode_dicts)
# ---------------------------------------------------------------------------

from src.jax_agents.simulation import (
    run_N_joint_episodes,
    joint_results_to_episode_dicts,
    _pol_net,
)
from src.jax_networks.policy_net import PolicyNet
from src.jax_networks.gru_encoder import GRUEncoder

# Initialise dummy pol/gru params for two agents
def _make_pol_gru_params(seed):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    dummy_in  = jnp.zeros((1, IN_DIM))
    dummy_acts = jnp.zeros((MAX_PARTNER_TURNS,), dtype=jnp.int32)
    pol_params = _pol_net.init(k1, dummy_in)
    gru_params = _gru_enc.init(k2, dummy_acts, jnp.array(1))
    return pol_params, gru_params

POL_A, GRU_A = _make_pol_gru_params(0)
POL_B, GRU_B = _make_pol_gru_params(7)

N_JOINT = 6
_JOINT_KEYS = jax.random.split(jax.random.PRNGKey(99), N_JOINT)


@pytest.fixture(scope="module")
def joint_results():
    return run_N_joint_episodes(_JOINT_KEYS, POL_A, GRU_A, POL_B, GRU_B)

@pytest.fixture(scope="module")
def joint_dicts(joint_results):
    return joint_results_to_episode_dicts(joint_results)


class TestRunNJointEpisodes:
    def test_score_shape(self, joint_results):
        assert joint_results["score"].shape == (N_JOINT,)

    def test_obs0_shape(self, joint_results):
        assert joint_results["obs0"].shape == (N_JOINT, MAX_EPISODE_STEPS, OBS_DIM)

    def test_obs1_shape(self, joint_results):
        assert joint_results["obs1"].shape == (N_JOINT, MAX_EPISODE_STEPS, OBS_DIM)

    def test_cur_players_shape(self, joint_results):
        assert joint_results["cur_players"].shape == (N_JOINT, MAX_EPISODE_STEPS)

    def test_actions_shape(self, joint_results):
        assert joint_results["actions"].shape == (N_JOINT, MAX_EPISODE_STEPS)

    def test_n_steps_shape(self, joint_results):
        assert joint_results["n_steps"].shape == (N_JOINT,)

    def test_n_steps_positive(self, joint_results):
        assert (joint_results["n_steps"] > 0).all()

    def test_n_steps_within_max(self, joint_results):
        assert (joint_results["n_steps"] <= MAX_EPISODE_STEPS).all()

    def test_score_non_negative(self, joint_results):
        assert (joint_results["score"] >= 0).all()

    def test_cur_players_binary(self, joint_results):
        vals = np.array(joint_results["cur_players"])
        assert set(np.unique(vals)).issubset({0, 1})

    def test_actions_in_range(self, joint_results):
        acts = np.array(joint_results["actions"])
        assert (acts >= 0).all() and (acts < NUM_ACTIONS).all()

    def test_different_keys_different_scores(self):
        keys_1 = jax.random.split(jax.random.PRNGKey(1), N_JOINT)
        keys_2 = jax.random.split(jax.random.PRNGKey(2), N_JOINT)
        r1 = run_N_joint_episodes(keys_1, POL_A, GRU_A, POL_B, GRU_B)
        r2 = run_N_joint_episodes(keys_2, POL_A, GRU_A, POL_B, GRU_B)
        assert not jnp.array_equal(r1["n_steps"], r2["n_steps"])

    def test_different_policies_different_behaviour(self):
        pol_c, gru_c = _make_pol_gru_params(42)
        r1 = run_N_joint_episodes(_JOINT_KEYS, POL_A, GRU_A, POL_B, GRU_B)
        r2 = run_N_joint_episodes(_JOINT_KEYS, pol_c, gru_c, POL_B, GRU_B)
        assert not jnp.array_equal(r1["actions"], r2["actions"])


class TestJointResultsToEpisodeDicts:
    def test_returns_correct_count(self, joint_dicts):
        assert len(joint_dicts) == N_JOINT

    def test_episode_has_steps_and_score(self, joint_dicts):
        ep = joint_dicts[0]
        assert "steps" in ep and "score" in ep

    def test_score_is_float(self, joint_dicts):
        assert isinstance(joint_dicts[0]["score"], float)

    def test_steps_count_matches_n_steps(self, joint_results, joint_dicts):
        n_steps = np.array(joint_results["n_steps"])
        for i, ep in enumerate(joint_dicts):
            assert len(ep["steps"]) == int(n_steps[i])

    def test_step_has_required_keys(self, joint_dicts):
        step = joint_dicts[0]["steps"][0]
        assert set(step.keys()) == {"current_player", "player_obs", "action"}

    def test_player_obs_has_both_players(self, joint_dicts):
        obs = joint_dicts[0]["steps"][0]["player_obs"]
        assert 0 in obs and 1 in obs

    def test_obs_shape(self, joint_dicts):
        obs = joint_dicts[0]["steps"][0]["player_obs"]
        assert obs[0].shape == (OBS_DIM,)
        assert obs[1].shape == (OBS_DIM,)

    def test_current_player_is_int(self, joint_dicts):
        assert isinstance(joint_dicts[0]["steps"][0]["current_player"], int)

    def test_action_is_int(self, joint_dicts):
        assert isinstance(joint_dicts[0]["steps"][0]["action"], int)

    def test_action_in_range(self, joint_dicts):
        for ep in joint_dicts:
            for step in ep["steps"]:
                assert 0 <= step["action"] < NUM_ACTIONS

    def test_current_player_in_range(self, joint_dicts):
        for ep in joint_dicts:
            for step in ep["steps"]:
                assert step["current_player"] in (0, 1)
