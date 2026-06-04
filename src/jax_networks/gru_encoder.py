# src/jax_networks/gru_encoder.py
# Flax port of GRUEncoder: Embed(8→16) → GRU(hidden=64).
#
# Variable-length sequences are handled with a static Python loop that is
# unrolled by JAX during JIT tracing.  This avoids the nn.scan / jax.lax.scan
# compatibility issues with bound Flax modules across JAX versions.
# For tiny Hanabi max_len ≤ 20 the unrolled graph is compact and fast.
#
# Masking: at each step t, the carry is updated only when t < seq_len;
# otherwise the previous carry is kept.  This matches pack_padded_sequence
# semantics from the PyTorch implementation.
#
# Three public entry-points:
#   __call__(actions, seq_len, init_h)  — encode one padded sequence
#   encode_batch(params, actions_batch, seq_lens) — mean-pool over N sequences
#   step(params, h, action)             — single online step

import jax
import jax.numpy as jnp
from flax import linen as nn

NUM_ACTIONS = 8
EMBED_DIM = 16
HIDDEN_DIM = 64


class GRUEncoder(nn.Module):
    num_actions: int = NUM_ACTIONS
    embed_dim: int = EMBED_DIM
    hidden_dim: int = HIDDEN_DIM

    @nn.compact
    def __call__(self, actions, seq_len, init_h=None, return_states=False):
        """
        Encode a single padded action sequence.

        Args:
            actions       : (max_len,) int32  — padded action indices [0, num_actions)
            seq_len       : scalar int32      — true sequence length (≤ max_len)
            init_h        : (hidden_dim,) or None — initial carry (zeros if None)
            return_states : bool — if True, also return all intermediate hidden
                            states as (max_len, hidden_dim). Resolved at trace
                            time so it does not add JAX control-flow overhead.

        Returns:
            h      : (hidden_dim,) float32 — GRU carry after seq_len valid steps
            states : (max_len, hidden_dim) float32 — only when return_states=True
        """
        embed_layer = nn.Embed(num_embeddings=self.num_actions, features=self.embed_dim)
        gru_cell = nn.GRUCell(features=self.hidden_dim)

        if init_h is None:
            init_h = jnp.zeros(self.hidden_dim)

        h = init_h
        max_len = actions.shape[0]  # static integer — loop unrolls at JIT time
        all_h = []

        for t in range(max_len):
            emb = embed_layer(actions[t])          # (embed_dim,)
            new_h, _ = gru_cell(h, emb)            # (hidden_dim,)
            # Keep carry frozen at padding positions (t >= seq_len).
            h = jnp.where(t < seq_len, new_h, h)
            if return_states:
                all_h.append(h)

        if return_states:
            return h, jnp.stack(all_h)             # (hidden_dim,), (max_len, hidden_dim)
        return h                                   # (hidden_dim,)

    def step(self, params, h, action):
        """
        Single online step: advance GRU by one action token.

        Args:
            params : Flax param dict
            h      : (hidden_dim,) float32
            action : scalar int

        Returns:
            (hidden_dim,) float32 — updated carry
        """
        return self.apply(
            params,
            jnp.array([action], dtype=jnp.int32),
            jnp.array(1, dtype=jnp.int32),
            h,
        )

    def encode_batch(self, params, actions_batch, seq_lens):
        """
        Encode N sequences and mean-pool their final hidden states.

        Args:
            params        : Flax param dict
            actions_batch : (N, max_len) int32
            seq_lens      : (N,) int32

        Returns:
            (hidden_dim,) float32 — mean hidden state across N sequences
        """
        hs = jax.vmap(lambda a, l: self.apply(params, a, l))(
            actions_batch, seq_lens
        )  # (N, hidden_dim)
        return hs.mean(axis=0)
