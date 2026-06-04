"""
tests/test_gru_encoder.py

Run with:
    cd <project_root>
    python -m pytest tests/test_gru_encoder.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from src.networks.gru_encoder import GRUEncoder

HIDDEN_DIM = 64
NUM_ACTIONS = 8


@pytest.fixture
def encoder():
    return GRUEncoder(num_actions=NUM_ACTIONS, embed_dim=16, hidden_dim=HIDDEN_DIM)


# ------------------------------------------------------------------
# 1. initial_hidden
# ------------------------------------------------------------------

def test_initial_hidden_shape(encoder):
    h = encoder.initial_hidden()
    assert h.shape == (HIDDEN_DIM,)

def test_initial_hidden_is_zero(encoder):
    h = encoder.initial_hidden()
    assert torch.all(h == 0)


# ------------------------------------------------------------------
# 2. encode_sequences — shape and type
# ------------------------------------------------------------------

def test_encode_single_sequence(encoder):
    h = encoder.encode_sequences([[0, 1, 2]])
    assert h.shape == (HIDDEN_DIM,)

def test_encode_multiple_sequences(encoder):
    h = encoder.encode_sequences([[0, 1], [3, 2, 1], [5]])
    assert h.shape == (HIDDEN_DIM,)

def test_encode_single_action_sequence(encoder):
    h = encoder.encode_sequences([[4]])
    assert h.shape == (HIDDEN_DIM,)

def test_encode_long_sequence(encoder):
    h = encoder.encode_sequences([list(range(NUM_ACTIONS)) * 10])
    assert h.shape == (HIDDEN_DIM,)

def test_encode_variable_length_sequences(encoder):
    """All variable lengths should still produce a (HIDDEN_DIM,) result."""
    seqs = [[0], [0, 1, 2, 3], [7, 6, 5, 4, 3, 2, 1, 0]]
    h = encoder.encode_sequences(seqs)
    assert h.shape == (HIDDEN_DIM,)

def test_encode_returns_tensor(encoder):
    h = encoder.encode_sequences([[0, 1, 2]])
    assert isinstance(h, torch.Tensor)

def test_encode_empty_sequence_handled(encoder):
    """Empty episode should not crash; uses zero hidden state."""
    h = encoder.encode_sequences([[], [0, 1]])
    assert h.shape == (HIDDEN_DIM,)


# ------------------------------------------------------------------
# 3. encode_sequences — mean pooling behaviour
# ------------------------------------------------------------------

def test_encode_two_identical_sequences_equals_one(encoder):
    """Mean of two identical sequences should equal encoding one."""
    seq = [0, 1, 2]
    h_one = encoder.encode_sequences([seq])
    h_two = encoder.encode_sequences([seq, seq])
    assert torch.allclose(h_one, h_two, atol=1e-6)

def test_encode_different_sequences_differ(encoder):
    """Sequences with different actions should produce different embeddings."""
    h1 = encoder.encode_sequences([[0, 0, 0, 0]])
    h2 = encoder.encode_sequences([[7, 7, 7, 7]])
    assert not torch.allclose(h1, h2)


# ------------------------------------------------------------------
# 4. step — shape and type
# ------------------------------------------------------------------

def test_step_shape(encoder):
    h = encoder.initial_hidden()
    h_new = encoder.step(h, 0)
    assert h_new.shape == (HIDDEN_DIM,)

def test_step_returns_tensor(encoder):
    h = encoder.initial_hidden()
    h_new = encoder.step(h, 3)
    assert isinstance(h_new, torch.Tensor)

def test_step_changes_hidden_state(encoder):
    """Stepping forward must produce a different hidden state than the input."""
    h = encoder.initial_hidden()
    h_new = encoder.step(h, 1)
    assert not torch.allclose(h, h_new)

def test_step_all_actions_valid(encoder):
    """All action UIDs 0–7 must be valid inputs to step."""
    h = encoder.initial_hidden()
    for action in range(NUM_ACTIONS):
        h = encoder.step(h, action)
        assert h.shape == (HIDDEN_DIM,)

def test_step_sequence_matches_encode(encoder):
    """
    Stepping through a sequence one action at a time should produce the same
    final hidden state as encoding the full sequence in one call.
    """
    seq = [2, 5, 1, 3]
    h_step = encoder.initial_hidden()
    for action in seq:
        h_step = encoder.step(h_step, action)

    # encode_sequences mean-pools over episodes. With a single episode the
    # mean equals the only episode's final hidden state.
    with torch.no_grad():
        h_encode = encoder.encode_sequences([seq])

    assert torch.allclose(h_step, h_encode, atol=1e-6)

def test_step_is_stateless(encoder):
    """step() must not mutate the input hidden state."""
    h = encoder.initial_hidden()
    h_copy = h.clone()
    encoder.step(h, 2)
    assert torch.allclose(h, h_copy)


# ------------------------------------------------------------------
# 5. Module properties
# ------------------------------------------------------------------

def test_is_nn_module(encoder):
    import torch.nn as nn
    assert isinstance(encoder, nn.Module)

def test_has_trainable_parameters(encoder):
    params = list(encoder.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)

def test_two_encoders_independent(encoder):
    """Modifying one encoder's weights must not affect another."""
    encoder2 = GRUEncoder()
    # Zero out encoder2's embedding weights
    with torch.no_grad():
        encoder2.embedding.weight.fill_(0.0)
    h1 = encoder.encode_sequences([[1, 2, 3]])
    h2 = encoder2.encode_sequences([[1, 2, 3]])
    assert not torch.allclose(h1, h2)
