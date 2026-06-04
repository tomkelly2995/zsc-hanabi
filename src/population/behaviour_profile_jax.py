# src/population/behaviour_profile_jax.py
# JAX port of BehaviourProfile.
#
# Changes vs original:
#   • _build_obs_action_pairs → now returns 4-tuples (obs, h_bc, action, oracle_next)
#     where h_bc=zeros (cheap evaluation approximation).
#   • _build_bc_pairs         → NEW: returns 4-tuples with teacher-forced h_bc.
#   • BehaviourProfileJax     → stores raw_oracle_nexts alongside raw_sequences.
#
# No-information-sharing constraint: only the partner's publicly observable
# actions and observations are used to build the profile.

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from src.jax_networks.bc_partner_model import BCPartnerModel, _BCNet, HIDDEN_DIM


def _build_obs_action_pairs(
    raw_sequences:     list,
    raw_obs_sequences: list,
    raw_oracle_nexts:  list,
) -> list:
    """
    Flatten per-episode partner (obs, h_bc, action, oracle_next) tuples for BC
    evaluation.  Uses h_bc=zeros (zero-init approximation) — fast but slightly
    pessimistic on accuracy for the GRU-augmented model.

    Args:
        raw_sequences     : list[list[int]]   — partner action sequences
        raw_obs_sequences : list[list[array]] — partner public obs sequences
        raw_oracle_nexts  : list[list[int]]   — oracle's next action after each
                            BC turn (-1 if no oracle turn follows)
    Returns:
        list of (obs, h_bc_zeros, action, oracle_next) tuples
    """
    zeros_h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    pairs   = []
    for obs_seq, act_seq, oracle_seq in zip(
        raw_obs_sequences, raw_sequences, raw_oracle_nexts
    ):
        for obs, act, oracle_next in zip(obs_seq, act_seq, oracle_seq):
            pairs.append((
                np.array(obs, dtype=np.float32),
                zeros_h,
                int(act),
                int(oracle_next),
            ))
    return pairs


def _build_bc_pairs(
    raw_sequences:     list,
    raw_obs_sequences: list,
    raw_oracle_nexts:  list,
    net:               _BCNet,
    params:            dict,
) -> list:
    """
    Build training pairs with teacher-forced h_bc computed using the BC model's
    current params.  For each episode, h_bc starts at zeros and advances one
    step after each BC turn using the actual action taken (teacher forcing).

    Args:
        raw_sequences     : list[list[int]]   — partner action sequences
        raw_obs_sequences : list[list[array]] — partner public obs sequences
        raw_oracle_nexts  : list[list[int]]   — oracle's next action after each
                            BC turn (-1 if no oracle turn follows)
        net    : _BCNet module instance
        params : current _BCNet params dict

    Returns:
        list of (obs, h_bc, action, oracle_next) tuples
    """
    pairs = []
    for obs_seq, act_seq, oracle_seq in zip(
        raw_obs_sequences, raw_sequences, raw_oracle_nexts
    ):
        h_bc = np.zeros(HIDDEN_DIM, dtype=np.float32)
        for obs, act, oracle_next in zip(obs_seq, act_seq, oracle_seq):
            pairs.append((
                np.array(obs, dtype=np.float32),
                h_bc.copy(),
                int(act),
                int(oracle_next),
            ))
            # Teacher-force: advance h_bc with the actual action taken
            h_bc = np.array(
                net.apply(
                    {"params": params},
                    jnp.array(h_bc, dtype=jnp.float32),
                    jnp.array(act, dtype=jnp.int32),
                    method=net.gru_step,
                ),
                dtype=np.float32,
            )
    return pairs


class BehaviourProfileJax:
    """
    Stores observed partner behaviour for use by CooperativeODCFRAgentJax.

    Attributes:
        raw_sequences      : list[list[int]]    — per-episode partner action IDs
        raw_obs_sequences  : list[list[array]]  — per-episode partner public obs
        raw_oracle_nexts   : list[list[int]]    — oracle's next action after each
                             BC turn (-1 if no oracle turn follows)
        bc_model_params    : Flax param dict for _BCNet — used directly by agent
        bc_model           : BCPartnerModel wrapper (for retraining / evaluate)
        bc_train_loss      : float
        iteration_created  : int
        source             : str — "agent_A" or "agent_B"
    """

    def __init__(
        self,
        raw_sequences:     list,
        raw_obs_sequences: list,
        raw_oracle_nexts:  list,
        bc_model:          BCPartnerModel,
        bc_train_loss:     float,
        iteration_created: int,
        source:            str,
    ):
        self.raw_sequences     = raw_sequences
        self.raw_obs_sequences = raw_obs_sequences
        self.raw_oracle_nexts  = raw_oracle_nexts
        self.bc_model          = bc_model
        self.bc_model_params   = bc_model.params   # Flax dict read by agent
        self.bc_train_loss     = bc_train_loss
        self.iteration_created = iteration_created
        self.source            = source

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_episodes(
        cls,
        raw_sequences:     list,
        raw_obs_sequences: list,
        raw_oracle_nexts:  list,
        iteration_created: int,
        source:            str,
        bc_epochs:         int   = 40,
        bc_lr:             float = 1e-3,
        bc_rounds:         int   = 3,
        seed:              int   = 0,
    ) -> "BehaviourProfileJax":
        """
        Build a BehaviourProfileJax from raw episode data.

        Uses iterated teacher forcing: bc_rounds rounds of
        (_build_bc_pairs with current params → partial train).
        Each round rebuilds teacher-forced h_bc using the GRU weights from
        the previous round, progressively closing the lag between training
        pairs and the trained model.

        bc_rounds=1 reproduces the old single-pass behaviour exactly.
        epochs_per_round = max(1, bc_epochs // bc_rounds), so total epochs
        ≈ bc_epochs regardless of bc_rounds.
        """
        bc_model = BCPartnerModel(lr=bc_lr, seed=seed)
        epochs_per_round = max(1, bc_epochs // bc_rounds)

        if raw_sequences:
            for _ in range(bc_rounds):
                pairs = _build_bc_pairs(
                    raw_sequences, raw_obs_sequences, raw_oracle_nexts,
                    bc_model.net, bc_model.params,
                )
                if pairs:
                    bc_model.train_supervised(pairs, epochs=epochs_per_round)

        bc_train_loss = bc_model.train_loss

        return cls(
            raw_sequences=raw_sequences,
            raw_obs_sequences=raw_obs_sequences,
            raw_oracle_nexts=raw_oracle_nexts,
            bc_model=bc_model,
            bc_train_loss=bc_train_loss,
            iteration_created=iteration_created,
            source=source,
        )

    # ------------------------------------------------------------------
    # Retraining
    # ------------------------------------------------------------------

    def retrain_bc(self, bc_epochs: int = 20) -> float:
        """Retrain BC model in-place on stored raw sequences."""
        pairs = _build_bc_pairs(
            self.raw_sequences, self.raw_obs_sequences, self.raw_oracle_nexts,
            self.bc_model.net, self.bc_model.params,
        )
        if not pairs:
            return 0.0
        self.bc_model.train_supervised(pairs, epochs=bc_epochs)
        self.bc_train_loss   = self.bc_model.train_loss
        self.bc_model_params = self.bc_model.params
        return self.bc_train_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_bc(self, pairs: list) -> tuple:
        """
        Evaluate BC model accuracy and mean action loss.

        Args:
            pairs : list of (obs, h_bc, action, oracle_next) tuples
                    (from _build_obs_action_pairs — uses zero h_bc approximation)
        Returns:
            (accuracy, mean_action_loss) floats
        """
        return self.bc_model.evaluate(pairs)
