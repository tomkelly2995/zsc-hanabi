"""
tests/test_behaviour_profile.py

Run with:
    cd <project_root>
    python -m pytest tests/test_behaviour_profile.py -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from src.population.behaviour_profile import BehaviourProfile, _build_obs_action_pairs
from src.networks.gru_encoder import GRUEncoder
from src.agents.bc_partner_model import BCPartnerModel

OBS_DIM = 84
NUM_ACTIONS = 8
HIDDEN_DIM = 64


# ------------------------------------------------------------------
# Shared fixtures and helpers
# ------------------------------------------------------------------

def make_raw_sequences(n_episodes=3, ep_len=4):
    return [[np.random.randint(0, NUM_ACTIONS) for _ in range(ep_len)]
            for _ in range(n_episodes)]

def make_raw_obs_sequences(n_episodes=3, ep_len=4):
    return [[np.random.rand(OBS_DIM).astype(np.float32) for _ in range(ep_len)]
            for _ in range(n_episodes)]

@pytest.fixture
def encoder():
    return GRUEncoder()

@pytest.fixture
def raw_data():
    return make_raw_sequences(), make_raw_obs_sequences()

@pytest.fixture
def profile(encoder, raw_data):
    seqs, obs_seqs = raw_data
    return BehaviourProfile.from_episodes(
        raw_sequences=seqs,
        raw_obs_sequences=obs_seqs,
        gru_encoder=encoder,
        iteration_created=0,
        source="agent_B",
    )


# ------------------------------------------------------------------
# 1. _build_obs_action_pairs helper
# ------------------------------------------------------------------

def test_build_pairs_length():
    seqs = [[0, 1, 2], [3, 4]]
    obs_seqs = [[np.zeros(OBS_DIM)] * 3, [np.zeros(OBS_DIM)] * 2]
    pairs = _build_obs_action_pairs(seqs, obs_seqs)
    assert len(pairs) == 5   # 3 + 2

def test_build_pairs_contents():
    seqs = [[7]]
    obs_seqs = [[np.ones(OBS_DIM, dtype=np.float32)]]
    pairs = _build_obs_action_pairs(seqs, obs_seqs)
    assert len(pairs) == 1
    obs, act = pairs[0]
    assert act == 7
    assert obs.shape == (OBS_DIM,)

def test_build_pairs_empty():
    pairs = _build_obs_action_pairs([], [])
    assert pairs == []


# ------------------------------------------------------------------
# 2. from_episodes — fields stored correctly
# ------------------------------------------------------------------

def test_from_episodes_stores_raw_sequences(profile, raw_data):
    seqs, _ = raw_data
    assert profile.raw_sequences == seqs

def test_from_episodes_stores_raw_obs_sequences(profile, raw_data):
    _, obs_seqs = raw_data
    assert profile.raw_obs_sequences is obs_seqs

def test_from_episodes_cached_h_oppo_shape(profile):
    assert profile.cached_h_oppo.shape == (HIDDEN_DIM,)

def test_from_episodes_cached_h_oppo_is_tensor(profile):
    assert isinstance(profile.cached_h_oppo, torch.Tensor)

def test_from_episodes_bc_model_is_bc_partner_model(profile):
    assert isinstance(profile.bc_model, BCPartnerModel)

def test_from_episodes_bc_model_in_eval_mode(profile):
    assert not profile.bc_model.training

def test_from_episodes_iteration_created(encoder):
    seqs = make_raw_sequences()
    obs_seqs = make_raw_obs_sequences()
    p = BehaviourProfile.from_episodes(seqs, obs_seqs, encoder, iteration_created=5, source="agent_A")
    assert p.iteration_created == 5

def test_from_episodes_source(encoder):
    seqs = make_raw_sequences()
    obs_seqs = make_raw_obs_sequences()
    p = BehaviourProfile.from_episodes(seqs, obs_seqs, encoder, iteration_created=0, source="agent_A")
    assert p.source == "agent_A"


# ------------------------------------------------------------------
# 3. from_episodes — encoding is consistent with GRUEncoder directly
# ------------------------------------------------------------------

def test_from_episodes_h_oppo_matches_encoder(encoder):
    seqs = make_raw_sequences()
    obs_seqs = make_raw_obs_sequences()
    p = BehaviourProfile.from_episodes(seqs, obs_seqs, encoder, iteration_created=0, source="agent_B")
    with torch.no_grad():
        expected = encoder.encode_sequences(seqs)
    assert torch.allclose(p.cached_h_oppo, expected, atol=1e-6)


# ------------------------------------------------------------------
# 4. from_episodes — bc_model is functional after training
# ------------------------------------------------------------------

def test_from_episodes_bc_model_produces_valid_distribution(profile):
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    probs = profile.bc_model.action_distribution(obs, legal_actions=[0, 1, 2])
    assert probs.shape == (NUM_ACTIONS,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


# ------------------------------------------------------------------
# 5. __init__ direct construction
# ------------------------------------------------------------------

def test_direct_init_stores_fields(encoder):
    seqs = make_raw_sequences()
    obs_seqs = make_raw_obs_sequences()
    h = encoder.encode_sequences(seqs)
    bc = BCPartnerModel()
    p = BehaviourProfile(
        raw_sequences=seqs,
        raw_obs_sequences=obs_seqs,
        cached_h_oppo=h,
        bc_model=bc,
        bc_train_loss=0.42,
        iteration_created=3,
        source="agent_A",
    )
    assert p.iteration_created == 3
    assert p.source == "agent_A"
    assert torch.allclose(p.cached_h_oppo, h)


# ------------------------------------------------------------------
# 6. reencode — updates cached_h_oppo
# ------------------------------------------------------------------

def test_reencode_returns_old_h_oppo(profile, encoder):
    old_h_before = profile.cached_h_oppo.clone()
    # Perturb GRU weights so re-encoding produces a different value
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    returned_old_h = profile.reencode(encoder)
    assert torch.allclose(returned_old_h, old_h_before, atol=1e-6)

def test_reencode_updates_cached_h_oppo(profile, encoder):
    old_h = profile.cached_h_oppo.clone()
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    profile.reencode(encoder)
    assert not torch.allclose(profile.cached_h_oppo, old_h)

def test_reencode_new_h_matches_encoder(profile, encoder):
    with torch.no_grad():
        for p in encoder.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    profile.reencode(encoder)
    with torch.no_grad():
        expected = encoder.encode_sequences(profile.raw_sequences)
    assert torch.allclose(profile.cached_h_oppo, expected, atol=1e-6)

def test_reencode_retrains_bc_model(profile, encoder):
    """BC model weights must change after reencode (retrained on same data)."""
    params_before = [p.clone() for p in profile.bc_model.parameters()]
    # Zero out bc_model weights so we know retraining changes them
    with torch.no_grad():
        for p in profile.bc_model.parameters():
            p.zero_()
    profile.reencode(encoder)
    params_after = list(profile.bc_model.parameters())
    changed = any(
        not torch.allclose(b, a)
        for b, a in zip(params_before, params_after)
    )
    assert changed

def test_reencode_bc_model_still_valid_after_reencode(profile, encoder):
    profile.reencode(encoder)
    obs = np.random.rand(OBS_DIM).astype(np.float32)
    probs = profile.bc_model.action_distribution(obs, legal_actions=[0, 3, 5])
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


# ------------------------------------------------------------------
# 7. Drift computation (cosine distance) — used by XDO solver
# ------------------------------------------------------------------

def test_drift_is_zero_when_gru_unchanged(encoder):
    """If GRU weights haven't changed, re-encoding should give zero drift."""
    seqs = make_raw_sequences()
    obs_seqs = make_raw_obs_sequences()
    p = BehaviourProfile.from_episodes(seqs, obs_seqs, encoder, 0, "agent_B")
    old_h = p.reencode(encoder)
    drift = 1.0 - F.cosine_similarity(old_h.unsqueeze(0), p.cached_h_oppo.unsqueeze(0)).item()
    assert abs(drift) < 1e-5

def test_drift_is_positive_when_gru_changed(encoder):
    """After perturbing GRU weights, cosine drift must be > 0."""
    seqs = make_raw_sequences()
    obs_seqs = make_raw_obs_sequences()
    p = BehaviourProfile.from_episodes(seqs, obs_seqs, encoder, 0, "agent_B")
    with torch.no_grad():
        for param in encoder.parameters():
            param.add_(torch.randn_like(param) * 1.0)
    old_h = p.reencode(encoder)
    drift = 1.0 - F.cosine_similarity(old_h.unsqueeze(0), p.cached_h_oppo.unsqueeze(0)).item()
    assert drift > 0.0
