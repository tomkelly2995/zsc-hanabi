# tests/test_jax_networks.py
# Unit tests for src/jax_networks/{advantage_net,policy_net,gru_encoder,bc_partner_model}.

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from src.jax_networks.advantage_net import AdvantageNet
from src.jax_networks.policy_net import PolicyNet
from src.jax_networks.gru_encoder import GRUEncoder, NUM_ACTIONS, HIDDEN_DIM
from src.jax_networks.bc_partner_model import BCPartnerModel

KEY = jax.random.PRNGKey(42)
OBS_DIM = 84
H_DIM = 64
IN_DIM = OBS_DIM + H_DIM  # 148


# ---------------------------------------------------------------------------
# 1. AdvantageNet
# ---------------------------------------------------------------------------

class TestAdvantageNet:

    def setup_method(self):
        self.net = AdvantageNet()
        self.params = self.net.init(KEY, jnp.zeros((1, IN_DIM)))

    def test_output_shape_single(self):
        x = jnp.ones((IN_DIM,))
        out = self.net.apply(self.params, x)
        assert out.shape == (8,), f"expected (8,), got {out.shape}"

    def test_output_shape_batch(self):
        x = jnp.ones((16, IN_DIM))
        out = self.net.apply(self.params, x)
        assert out.shape == (16, 8)

    def test_output_dtype(self):
        x = jnp.ones((IN_DIM,))
        out = self.net.apply(self.params, x)
        assert out.dtype == jnp.float32

    def test_no_nan(self):
        x = jax.random.normal(KEY, (32, IN_DIM))
        out = self.net.apply(self.params, x)
        assert not jnp.any(jnp.isnan(out))

    def test_no_output_activation(self):
        # Raw Q-values can be negative — verify range is unconstrained.
        x = jax.random.normal(KEY, (256, IN_DIM)) * 10
        out = self.net.apply(self.params, x)
        assert jnp.any(out < 0), "expected some negative Q-values with large inputs"

    def test_jit(self):
        x = jnp.ones((IN_DIM,))
        jitted = jax.jit(self.net.apply)
        out = jitted(self.params, x)
        assert out.shape == (8,)

    def test_vmap(self):
        xs = jax.random.normal(KEY, (8, IN_DIM))
        out = jax.vmap(lambda x: self.net.apply(self.params, x))(xs)
        assert out.shape == (8, 8)


# ---------------------------------------------------------------------------
# 2. PolicyNet
# ---------------------------------------------------------------------------

class TestPolicyNet:

    def setup_method(self):
        self.net = PolicyNet()
        dummy = jnp.zeros((1, IN_DIM))
        self.params = self.net.init(KEY, dummy)

    def test_output_shape(self):
        x = jnp.ones((IN_DIM,))
        out = self.net.apply(self.params, x)
        assert out.shape == (8,)

    def test_masked_probs_shape(self):
        obs = jnp.ones((OBS_DIM,))
        h = jnp.ones((H_DIM,))
        mask = jnp.ones(8, dtype=bool)
        probs = self.net.masked_probs(self.params, obs, h, mask)
        assert probs.shape == (8,)

    def test_masked_probs_sum_to_one(self):
        obs = jax.random.normal(KEY, (OBS_DIM,))
        h = jax.random.normal(KEY, (H_DIM,))
        mask = jnp.array([True, True, True, True, False, False, True, True])
        probs = self.net.masked_probs(self.params, obs, h, mask)
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_masked_probs_zeros_on_illegal(self):
        obs = jax.random.normal(KEY, (OBS_DIM,))
        h = jax.random.normal(KEY, (H_DIM,))
        mask = jnp.array([True, True, False, False, True, True, False, False])
        probs = self.net.masked_probs(self.params, obs, h, mask)
        assert float(probs[2]) == pytest.approx(0.0, abs=1e-6)
        assert float(probs[3]) == pytest.approx(0.0, abs=1e-6)
        assert float(probs[6]) == pytest.approx(0.0, abs=1e-6)
        assert float(probs[7]) == pytest.approx(0.0, abs=1e-6)

    def test_masked_probs_non_negative(self):
        obs = jax.random.normal(KEY, (OBS_DIM,))
        h = jax.random.normal(KEY, (H_DIM,))
        mask = jnp.ones(8, dtype=bool)
        probs = self.net.masked_probs(self.params, obs, h, mask)
        assert jnp.all(probs >= 0.0)

    def test_all_illegal_except_one(self):
        obs = jax.random.normal(KEY, (OBS_DIM,))
        h = jax.random.normal(KEY, (H_DIM,))
        mask = jnp.array([False, False, False, True, False, False, False, False])
        probs = self.net.masked_probs(self.params, obs, h, mask)
        assert float(probs[3]) == pytest.approx(1.0, abs=1e-5)

    def test_jit(self):
        obs = jax.random.normal(KEY, (OBS_DIM,))
        h = jax.random.normal(KEY, (H_DIM,))
        mask = jnp.ones(8, dtype=bool)
        jitted = jax.jit(lambda p, o, hh, m: self.net.masked_probs(p, o, hh, m))
        probs = jitted(self.params, obs, h, mask)
        assert abs(float(probs.sum()) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3. GRUEncoder
# ---------------------------------------------------------------------------

class TestGRUEncoder:

    MAX_LEN = 10

    def setup_method(self):
        self.enc = GRUEncoder()
        dummy_actions = jnp.zeros((self.MAX_LEN,), dtype=jnp.int32)
        self.params = self.enc.init(KEY, dummy_actions, jnp.array(1))

    def _make_seq(self, actions, max_len=None):
        if max_len is None:
            max_len = self.MAX_LEN
        pad = max_len - len(actions)
        return jnp.array(actions + [0] * pad, dtype=jnp.int32), jnp.array(len(actions))

    def test_output_shape(self):
        actions, seq_len = self._make_seq([1, 2, 3])
        h = self.enc.apply(self.params, actions, seq_len)
        assert h.shape == (HIDDEN_DIM,)

    def test_output_dtype(self):
        actions, seq_len = self._make_seq([0])
        h = self.enc.apply(self.params, actions, seq_len)
        assert h.dtype == jnp.float32

    def test_padding_does_not_affect_result(self):
        # Same sequence, different padding — output should be identical.
        seq = [3, 1, 4, 1]
        a1 = jnp.array(seq + [0] * 6, dtype=jnp.int32)
        a2 = jnp.array(seq + [7] * 6, dtype=jnp.int32)  # different padding values
        h1 = self.enc.apply(self.params, a1, jnp.array(4))
        h2 = self.enc.apply(self.params, a2, jnp.array(4))
        np.testing.assert_allclose(np.array(h1), np.array(h2), atol=1e-6)

    def test_different_sequences_give_different_h(self):
        a1, l1 = self._make_seq([0, 1, 2])
        a2, l2 = self._make_seq([3, 4, 5])
        h1 = self.enc.apply(self.params, a1, l1)
        h2 = self.enc.apply(self.params, a2, l2)
        assert not jnp.allclose(h1, h2), "different sequences gave same embedding"

    def test_step_updates_h(self):
        h0 = jnp.zeros(HIDDEN_DIM)
        h1 = self.enc.step(self.params, h0, 3)
        assert h1.shape == (HIDDEN_DIM,)
        assert not jnp.allclose(h0, h1)

    def test_step_matches_encode_length1(self):
        # step() on action a from zero state == encode_sequence of length 1.
        h0 = jnp.zeros(HIDDEN_DIM)
        action = 5
        h_step = self.enc.step(self.params, h0, action)
        a, l = self._make_seq([action])
        h_enc = self.enc.apply(self.params, a, l)
        np.testing.assert_allclose(np.array(h_step), np.array(h_enc), atol=1e-6)

    def test_encode_batch_shape(self):
        N = 6
        actions_batch = jnp.zeros((N, self.MAX_LEN), dtype=jnp.int32)
        seq_lens = jnp.ones(N, dtype=jnp.int32) * 3
        h_mean = self.enc.encode_batch(self.params, actions_batch, seq_lens)
        assert h_mean.shape == (HIDDEN_DIM,)

    def test_encode_batch_mean(self):
        # Batch of identical sequences → same as single encode.
        a, l = self._make_seq([1, 2])
        h_single = self.enc.apply(self.params, a, l)
        actions_batch = jnp.stack([a, a, a])
        seq_lens = jnp.array([l, l, l])
        h_batch = self.enc.encode_batch(self.params, actions_batch, seq_lens)
        np.testing.assert_allclose(np.array(h_single), np.array(h_batch), atol=1e-6)

    def test_jit_encode(self):
        a, l = self._make_seq([2, 3, 4])
        jitted = jax.jit(self.enc.apply)
        h = jitted(self.params, a, l)
        assert h.shape == (HIDDEN_DIM,)

    def test_vmap_encode(self):
        N = 8
        keys = jax.random.split(KEY, N)
        actions_batch = jax.vmap(
            lambda k: jax.random.randint(k, (self.MAX_LEN,), 0, NUM_ACTIONS)
        )(keys)
        seq_lens = jnp.ones(N, dtype=jnp.int32) * 5
        hs = jax.vmap(lambda a, l: self.enc.apply(self.params, a, l))(
            actions_batch, seq_lens
        )
        assert hs.shape == (N, HIDDEN_DIM)

    def test_longer_sequence_changes_h(self):
        # seq [1,2] vs [1,2,3] — h must differ.
        a1, l1 = self._make_seq([1, 2])
        a2, l2 = self._make_seq([1, 2, 3])
        h1 = self.enc.apply(self.params, a1, l1)
        h2 = self.enc.apply(self.params, a2, l2)
        assert not jnp.allclose(h1, h2)


# ---------------------------------------------------------------------------
# 4. BCPartnerModel
# ---------------------------------------------------------------------------

class TestBCPartnerModel:

    def setup_method(self):
        self.model = BCPartnerModel(lr=1e-3, seed=0)

    def _make_pairs(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        obs = rng.random((n, OBS_DIM)).astype(np.float32)
        acts = rng.integers(0, NUM_ACTIONS, size=n)
        return [(obs[i], int(acts[i])) for i in range(n)]

    def test_action_distribution_shape(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        probs = self.model.action_distribution(jnp.array(obs))
        assert probs.shape == (NUM_ACTIONS,)

    def test_action_distribution_sums_to_one(self):
        obs = np.ones(OBS_DIM, dtype=np.float32)
        probs = self.model.action_distribution(jnp.array(obs))
        assert abs(float(probs.sum()) - 1.0) < 1e-5

    def test_action_distribution_non_negative(self):
        obs = np.ones(OBS_DIM, dtype=np.float32)
        probs = self.model.action_distribution(jnp.array(obs))
        assert jnp.all(probs >= 0.0)

    def test_action_distribution_batch(self):
        obs = np.ones((16, OBS_DIM), dtype=np.float32)
        probs = self.model.action_distribution(jnp.array(obs))
        assert probs.shape == (16, NUM_ACTIONS)

    def test_train_supervised_runs(self):
        pairs = self._make_pairs(100)
        self.model.train_supervised(pairs, epochs=2)
        assert self.model.train_loss >= 0.0

    def test_train_supervised_reduces_loss(self):
        # Use a deterministic dataset with a clear signal (all label=0).
        rng = np.random.default_rng(99)
        obs = rng.random((400, OBS_DIM)).astype(np.float32)
        pairs = [(obs[i], 0) for i in range(400)]
        _, loss_before = self.model.evaluate(pairs)
        self.model.train_supervised(pairs, epochs=20)
        _, loss_after = self.model.evaluate(pairs)
        assert loss_after < loss_before, (
            f"training did not reduce loss: before={loss_before:.4f}, after={loss_after:.4f}"
        )

    def test_evaluate_returns_floats(self):
        pairs = self._make_pairs(50)
        acc, loss = self.model.evaluate(pairs)
        assert isinstance(acc, float)
        assert isinstance(loss, float)

    def test_evaluate_accuracy_in_range(self):
        pairs = self._make_pairs(100)
        acc, _ = self.model.evaluate(pairs)
        assert 0.0 <= acc <= 1.0

    def test_evaluate_empty_pairs(self):
        acc, loss = self.model.evaluate([])
        assert acc == 0.0
        assert loss == 0.0

    def test_train_supervised_empty_pairs(self):
        # Should not raise.
        self.model.train_supervised([], epochs=5)

    def test_accuracy_improves_with_training(self):
        # Dataset: obs with first bit = action class (4 classes).
        rng = np.random.default_rng(7)
        n = 400
        obs = rng.random((n, OBS_DIM)).astype(np.float32)
        acts = (obs[:, 0] * NUM_ACTIONS).astype(int).clip(0, NUM_ACTIONS - 1)
        pairs = [(obs[i], int(acts[i])) for i in range(n)]
        acc_before, _ = self.model.evaluate(pairs)
        self.model.train_supervised(pairs, epochs=30)
        acc_after, _ = self.model.evaluate(pairs)
        assert acc_after > acc_before, (
            f"accuracy did not improve: before={acc_before:.3f}, after={acc_after:.3f}"
        )
