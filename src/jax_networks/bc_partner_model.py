# src/jax_networks/bc_partner_model.py
# BC partner model: MLP with GRU over own action history.
#
# Architecture (_BCNet):
#   • bc_embed   : Embed(8→16)   — action embedding for BC's own GRU
#   • bc_gru     : GRUCell(64)   — encodes BC's own past actions → h_bc
#   • dense1,2   : 128→64        — shared trunk: concat(obs_raw, h_bc) → rep
#   • action_head: 64→8          — action prediction (cross-entropy vs observed)
#
# The 84-dim observation already contains a 32-dim analytically-computed
# Bayesian belief about the partner's own hand (card_knowledge normalised by
# remaining card counts, plus colour/rank hint flags).  No auxiliary belief
# head is needed — the belief information enters as a direct input feature.
#
# Using setup() rather than @nn.compact so that gru_step() shares the same
# submodule instances (and therefore the same params) as __call__().
#
# BCPartnerModel (stateful wrapper)
#   • train_supervised(pairs, epochs) — pairs: (obs, h_bc, action, oracle_next)
#   • evaluate(pairs)                 — accuracy / action loss only
#   • gru_step(h_bc, action)          → updated h_bc
#   • zero_state()                    → zeros h_bc

import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

OBS_DIM      = 84
NUM_ACTIONS  = 8
EMBED_DIM    = 16
HIDDEN_DIM   = 64


class _BCNet(nn.Module):
    """
    BC partner model with GRU memory over its own action history.

    setup() is used so that gru_step() can call self.bc_embed and self.bc_gru
    through the same bound instances as __call__(), sharing parameters.
    """
    hidden_dim:  int = HIDDEN_DIM
    num_actions: int = NUM_ACTIONS
    embed_dim:   int = EMBED_DIM

    def setup(self):
        self.bc_embed    = nn.Embed(self.num_actions, self.embed_dim)
        self.bc_gru      = nn.GRUCell(features=self.hidden_dim)
        self.dense1      = nn.Dense(128)
        self.dense2      = nn.Dense(64)
        self.action_head = nn.Dense(self.num_actions)

    def __call__(self, obs: jnp.ndarray, h_bc: jnp.ndarray):
        """
        Forward pass: (obs, h_bc) → action_logits.

        The 84-dim obs already contains a 32-dim Bayesian belief over the
        partner's own hand (card_knowledge × remaining counts, normalised).
        No auxiliary belief head is needed.

        Args:
            obs   : (OBS_DIM,) raw observation for BC's current turn
            h_bc  : (HIDDEN_DIM,) BC's own GRU hidden state

        Returns:
            action_logits : (NUM_ACTIONS,) — action prediction logits
        """
        x      = jnp.concatenate([obs, h_bc], axis=-1)  # (OBS_DIM + HIDDEN_DIM,)
        x      = nn.relu(self.dense1(x))             # (128,)
        shared = nn.relu(self.dense2(x))             # (64,)
        return self.action_head(shared)

    def gru_step(self, h: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        Advance BC's own GRU by one action token.

        Args:
            h      : (HIDDEN_DIM,) current hidden state
            action : scalar int32 — the action BC just took

        Returns:
            (HIDDEN_DIM,) — updated hidden state
        """
        emb     = self.bc_embed(action)              # (EMBED_DIM,)
        h_new, _ = self.bc_gru(h, emb)              # (HIDDEN_DIM,)
        return h_new


class BCPartnerModel:
    """
    Stateful wrapper: _BCNet + optax AdamW.

    Public API
    ──────────
        action_distribution(obs, h_bc)          → (NUM_ACTIONS,) softmax probs
        gru_step(h_bc, action)                  → (HIDDEN_DIM,) updated h_bc
        zero_state()                            → zeros h_bc
        train_supervised(pairs, epochs)         → updates params in-place
        evaluate(pairs)                         → (accuracy, mean_action_loss)
    """

    def __init__(
        self,
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        seed:         int   = 0,
    ):
        self.net       = _BCNet()
        self.optimizer = optax.adamw(lr, weight_decay=weight_decay)

        k1, k2    = jax.random.split(jax.random.PRNGKey(seed))
        dummy_obs = jnp.zeros((OBS_DIM,))
        dummy_h   = jnp.zeros((HIDDEN_DIM,))
        dummy_act = jnp.array(0, dtype=jnp.int32)

        # __call__ uses: dense1, dense2, action_head, belief_head
        vars_call = self.net.init(k1, dummy_obs, dummy_h)
        # gru_step uses: bc_embed, bc_gru  (not touched by __call__)
        vars_gru  = self.net.init(k2, dummy_h, dummy_act,
                                  method=self.net.gru_step)
        # Merge — key sets are disjoint, so this is safe.
        self.params    = {**vars_call["params"], **vars_gru["params"]}
        self.opt_state = self.optimizer.init(self.params)
        self.train_loss: float = 0.0

        self._train_step = _make_train_step(self.net, self.optimizer)

    # -----------------------------------------------------------------------

    def zero_state(self) -> jnp.ndarray:
        """Return initial (all-zeros) h_bc state."""
        return jnp.zeros(HIDDEN_DIM, dtype=jnp.float32)

    def action_distribution(
        self, obs: jnp.ndarray, h_bc: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Args:
            obs   : (OBS_DIM,) or (N, OBS_DIM) float32
            h_bc  : (HIDDEN_DIM,) or (N, HIDDEN_DIM) float32
        Returns:
            (NUM_ACTIONS,) or (N, NUM_ACTIONS) float32 softmax probabilities
        """
        action_logits = self.net.apply({"params": self.params}, obs, h_bc)
        return jax.nn.softmax(action_logits, axis=-1)

    def gru_step(self, h_bc: jnp.ndarray, action) -> jnp.ndarray:
        """Advance BC GRU state by one action token."""
        return self.net.apply(
            {"params": self.params},
            h_bc,
            jnp.asarray(action, dtype=jnp.int32),
            method=self.net.gru_step,
        )

    # -----------------------------------------------------------------------

    def train_supervised(
        self,
        pairs:      list,
        epochs:     int = 40,
        batch_size: int = 64,
    ) -> None:
        """
        Supervised training on BC pairs.

        Args:
            pairs : list of (obs, h_bc, action, oracle_next_action)
                    • obs               : (OBS_DIM,) float32
                    • h_bc              : (HIDDEN_DIM,) float32 — teacher-forced GRU state
                    • action            : int — BC's own action at this turn
                    • oracle_next_action: int — oracle's action on the immediately
                                          following oracle turn; -1 if none (last BC
                                          turn with no following oracle step).
        """
        if not pairs:
            return

        obs_arr    = jnp.array([p[0] for p in pairs], dtype=jnp.float32)  # (N, OBS_DIM)
        h_bc_arr   = jnp.array([p[1] for p in pairs], dtype=jnp.float32)  # (N, HIDDEN_DIM)
        act_arr    = jnp.array([p[2] for p in pairs], dtype=jnp.int32)    # (N,)
        # oracle_next: clamp -1 → 0 for safe cross-entropy; valid_arr masks those out
        oracle_arr = jnp.array(
            [p[3] if p[3] >= 0 else 0 for p in pairs], dtype=jnp.int32
        )                                                                   # (N,)
        valid_arr  = jnp.array(
            [1.0 if p[3] >= 0 else 0.0 for p in pairs], dtype=jnp.float32
        )                                                                   # (N,)

        N   = obs_arr.shape[0]
        key = jax.random.PRNGKey(0)
        total_loss, n_batches = 0.0, 0

        for _ in range(epochs):
            key, sk = jax.random.split(key)
            perm        = jax.random.permutation(sk, N)
            obs_s       = obs_arr[perm]
            h_s         = h_bc_arr[perm]
            act_s       = act_arr[perm]
            oracle_s    = oracle_arr[perm]
            valid_s     = valid_arr[perm]

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                self.params, self.opt_state, loss = self._train_step(
                    self.params, self.opt_state,
                    obs_s[start:end], h_s[start:end], act_s[start:end],
                    oracle_s[start:end], valid_s[start:end],
                )
                total_loss += float(loss)
                n_batches  += 1

        if n_batches > 0:
            self.train_loss = total_loss / n_batches

    def evaluate(self, pairs: list) -> tuple:
        """
        Args:
            pairs : list of (obs, h_bc, action, oracle_next_action)
        Returns:
            (accuracy: float, mean_action_loss: float)
        """
        if not pairs:
            return 0.0, 0.0

        obs_arr  = jnp.array([p[0] for p in pairs], dtype=jnp.float32)
        h_bc_arr = jnp.array([p[1] for p in pairs], dtype=jnp.float32)
        act_arr  = jnp.array([p[2] for p in pairs], dtype=jnp.int32)

        action_logits = self.net.apply({"params": self.params}, obs_arr, h_bc_arr)
        preds    = jnp.argmax(action_logits, axis=-1)
        accuracy = float(jnp.mean(preds == act_arr))
        loss     = float(
            optax.softmax_cross_entropy_with_integer_labels(action_logits, act_arr).mean()
        )
        return accuracy, loss


# ---------------------------------------------------------------------------
# JIT-compiled training step
# ---------------------------------------------------------------------------

def _make_train_step(
    net:       _BCNet,
    optimizer: optax.GradientTransformation,
):
    @jax.jit
    def train_step(params, opt_state, obs_b, h_bc_b, act_b, oracle_b, valid_b):
        def loss_fn(p):
            action_logits = net.apply({"params": p}, obs_b, h_bc_b)
            return optax.softmax_cross_entropy_with_integer_labels(
                action_logits, act_b
            ).mean()

        loss, grads   = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), new_opt_state, loss

    return train_step
