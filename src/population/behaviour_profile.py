# src/population/behaviour_profile.py
# BehaviourProfile: observed partner behaviour, stored per XDO iteration.
# Built from publicly observable partner actions and observations only —
# no policy weights accessed (no-information-sharing constraint enforced).
#
# Normal construction path: BehaviourProfile.from_episodes(...)
#   Encodes cached_h_oppo via GRUEncoder and trains BCPartnerModel inline.
#
# Lazy re-encoding path: profile.reencode(gru_encoder)
#   Called by XDOHanabiSolver when a profile is sampled after a GRU update.
#   Returns old cached_h_oppo so the caller can log embedding drift.

import torch
from src.agents.bc_partner_model import BCPartnerModel


class BehaviourProfile:
    """
    Stores observed partner behaviour for use by the ODCFR oracle.

    Attributes:
        raw_sequences      : list[list[int]]  — per-episode partner action IDs
        raw_obs_sequences  : list[list[array]] — per-episode partner public obs
        cached_h_oppo      : torch.Tensor (64,) — GRU embedding of raw_sequences
        bc_model           : BCPartnerModel — trained on (obs, action) pairs
        bc_train_loss      : float — cross-entropy loss from final BC training epoch
        iteration_created  : int — XDO iteration that produced this profile
        source             : str — "agent_A" or "agent_B"
    """

    def __init__(
        self,
        raw_sequences: list,
        raw_obs_sequences: list,
        cached_h_oppo: torch.Tensor,
        bc_model: BCPartnerModel,
        bc_train_loss: float,
        iteration_created: int,
        source: str,
    ):
        self.raw_sequences = raw_sequences
        self.raw_obs_sequences = raw_obs_sequences
        self.cached_h_oppo = cached_h_oppo        # (64,) tensor
        self.bc_model = bc_model
        self.bc_train_loss = bc_train_loss
        self.iteration_created = iteration_created
        self.source = source
        # Tracks which GRU training version encoded cached_h_oppo.
        # Set by XDOHanabiSolver after construction; used to decide whether
        # lazy re-encoding is needed before sampling.
        self._encoded_at_gru_count: int = 0

    # ------------------------------------------------------------------
    # Factory constructor (normal creation path)
    # ------------------------------------------------------------------

    @classmethod
    def from_episodes(
        cls,
        raw_sequences: list,
        raw_obs_sequences: list,
        gru_encoder,
        iteration_created: int,
        source: str,
        bc_epochs: int = 40,
    ) -> "BehaviourProfile":
        """
        Build a BehaviourProfile from raw episode data.

        Encodes cached_h_oppo immediately using the current GRU, trains a
        fresh BCPartnerModel on the extracted (obs, action) pairs, then
        returns the completed profile.

        Called by XDOHanabiSolver.extract_behaviour_profile().

        Args:
            raw_sequences     : list[list[int]] — partner action sequences
            raw_obs_sequences : list[list[array]] — partner public obs sequences
            gru_encoder       : GRUEncoder — current agent's encoder
            iteration_created : int — current XDO iteration index
            source            : str — "agent_A" or "agent_B"
            bc_epochs         : int — supervised training epochs for BC model
        """
        with torch.no_grad():
            cached_h_oppo = gru_encoder.encode_sequences(raw_sequences)

        bc_model = BCPartnerModel()
        pairs = _build_obs_action_pairs(raw_sequences, raw_obs_sequences)
        bc_train_loss = bc_model.train_supervised(pairs, epochs=bc_epochs)

        return cls(
            raw_sequences=raw_sequences,
            raw_obs_sequences=raw_obs_sequences,
            cached_h_oppo=cached_h_oppo,
            bc_model=bc_model,
            bc_train_loss=bc_train_loss,
            iteration_created=iteration_created,
            source=source,
        )

    # ------------------------------------------------------------------
    # Lazy re-encoding
    # ------------------------------------------------------------------

    def reencode(self, gru_encoder, bc_epochs: int = 20) -> torch.Tensor:
        """
        Recompute cached_h_oppo with the current GRU weights and retrain
        the BC model. Called by XDOHanabiSolver when this profile is sampled
        after a GRU update (lazy re-encoding strategy, spec Section 3.2).

        Args:
            gru_encoder: GRUEncoder — the agent's current (updated) encoder
            bc_epochs  : int — epochs for BC model retraining

        Returns:
            torch.Tensor (64,) — the *old* cached_h_oppo, so the caller can
            compute embedding drift:
                drift = 1 - F.cosine_similarity(old_h, new_h, dim=0)
        """
        old_h = self.cached_h_oppo.clone()

        with torch.no_grad():
            self.cached_h_oppo = gru_encoder.encode_sequences(self.raw_sequences)

        pairs = _build_obs_action_pairs(self.raw_sequences, self.raw_obs_sequences)
        self.bc_train_loss = self.bc_model.train_supervised(pairs, epochs=bc_epochs)

        return old_h


# ------------------------------------------------------------------
# Module-level helper (used by both from_episodes and reencode)
# ------------------------------------------------------------------

def _build_obs_action_pairs(raw_sequences: list, raw_obs_sequences: list) -> list:
    """
    Flatten per-episode action and observation sequences into a flat list of
    (obs_vector, action_id) tuples for BC model training.

    Args:
        raw_sequences     : list[list[int]]   — action IDs per episode
        raw_obs_sequences : list[list[array]] — public obs vectors per episode

    Returns:
        list of (obs_vector, int) tuples
    """
    pairs = []
    for obs_seq, act_seq in zip(raw_obs_sequences, raw_sequences):
        for obs, act in zip(obs_seq, act_seq):
            pairs.append((obs, act))
    return pairs
