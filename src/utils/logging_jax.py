# src/utils/logging_jax.py
# TensorboardLoggerJax — JAX-compatible logger using tensorboardX.
# Drop-in equivalent of logging.py but with no PyTorch dependency.
#
# Metrics written per XDO iteration
# ───────────────────────────────────────────────────────────────────
#   agent/{id}/mean_joint_score        mean team score from shared batch
#   agent/{id}/swap_regret             per-agent swap regret
#   agent/{id}/metastrategy_entropy    Shannon entropy of MetaStrategy
#   agent/{id}/top_convention_score    score of the highest-weight profile
#   agent/{id}/probe_score             pol_net probe score vs new profile
#   agent/{id}/pool_size               number of profiles in pool
#
#   inner_loop/{id}/bc_loss            BC model training loss
#   inner_loop/{id}/bc_accuracy        BC model action-prediction accuracy
#   inner_loop/{id}/cfr_iters_run      actual CFR iterations (early-stop aware)
#   inner_loop/{id}/probe_delta        Q-value convergence delta at stop
#   inner_loop/{id}/adv_loss           advantage net training loss
#   inner_loop/{id}/gru_loss           GRU joint-update training loss
#
#   timing/{id}/episodes_secs          joint episode collection time
#   timing/{id}/policy_train_secs      oracle train() total time
#   timing/{id}/profile_build_secs     extract_behaviour_profile() time
#   timing/{id}/sim_secs               CFR simulation time inside train()
#   timing/{id}/adv_secs               advantage net update time inside train()
#   timing/{id}/gru_secs               GRU update time inside train()
#   timing/{id}/pol_secs               policy net update time inside train()
#
#   metastrategy/l1_divergence         L1(meta_A, meta_B)
#   eval/joint_score                   final 200-episode evaluation score

from __future__ import annotations

import os


class TensorboardLoggerJax:
    """
    Thin wrapper around tensorboardX.SummaryWriter for structured
    per-iteration logging of JAX XDO training metrics.

    Usage:
        logger = TensorboardLoggerJax("runs/my_run")
        logger.log_iteration(...)
        logger.log_profile_metrics(...)
        logger.log_timings(...)
        logger.log_metastrategy_divergence(...)
        logger.log_eval(...)
        logger.close()
    """

    def __init__(self, log_dir: str):
        try:
            from tensorboardX import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "tensorboardX is required for TensorboardLoggerJax. "
                "Install it with: pip install tensorboardX"
            ) from e
        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir

    def _w(self, tag: str, value: float, step: int) -> None:
        """Write a scalar only if value is finite — silently skips NaN/Inf."""
        import math
        if math.isfinite(value):
            self._writer.add_scalar(tag, value, step)

    # ------------------------------------------------------------------
    # Per-iteration outer-loop metrics
    # ------------------------------------------------------------------

    def log_iteration(
        self,
        iteration: int,
        agent_id: str,
        mean_joint_score: float,
        swap_regret: float,
        metastrategy_entropy: float,
        top_convention_score: float,
        probe_score: float,
        pool_size: int,
    ) -> None:
        """
        Log outer-loop quality metrics for one XDO iteration.

        Args:
            iteration            : XDO iteration index (1-based).
            agent_id             : "agent_A" or "agent_B".
            mean_joint_score     : Mean team score over the shared episode batch.
            swap_regret          : SR = Σ_i π[i] * max(0, max_j score_j − score_i).
                                   Trends toward 0 as MetaStrategy converges.
            metastrategy_entropy : Shannon entropy H(π). High = spread; low = focused.
            top_convention_score : Score of the profile with highest MetaStrategy weight.
            probe_score          : Mean team score of pol_net vs newest BC profile
                                   (agent-specific, from _probe_profile_score).
            pool_size            : Number of profiles in the pool.
        """
        p = f"agent/{agent_id}"
        self._w(f"{p}/mean_joint_score",     mean_joint_score,     iteration)
        self._w(f"{p}/swap_regret",          swap_regret,          iteration)
        self._w(f"{p}/metastrategy_entropy", metastrategy_entropy, iteration)
        self._w(f"{p}/top_convention_score", top_convention_score, iteration)
        self._w(f"{p}/probe_score",          probe_score,          iteration)
        self._w(f"{p}/pool_size",            float(pool_size),     iteration)

    # ------------------------------------------------------------------
    # Inner-loop / oracle quality metrics
    # ------------------------------------------------------------------

    def log_profile_metrics(
        self,
        iteration: int,
        agent_id: str,
        bc_loss: float,
        bc_accuracy: float,
        cfr_iters_run: int,
        probe_delta: float,
        adv_loss: float,
        gru_loss: float,
        mean_regret: float = float("nan"),
    ) -> None:
        """
        Log inner-loop quality metrics for the oracle and newly added profile.

        Args:
            iteration    : XDO iteration index (1-based).
            agent_id     : "agent_A" or "agent_B".
            bc_loss      : BC model training loss (lower = better fit to partner).
            bc_accuracy  : BC model action-prediction accuracy (random baseline = 0.125).
            cfr_iters_run: Actual CFR iterations run (may be < budget via early stopping).
            probe_delta  : Mean Q-value change at early-stop point (convergence indicator).
            adv_loss     : Advantage net training loss (last reported value).
            gru_loss     : GRU joint-update training loss (last reported value).
            mean_regret  : Mean positive regret from adv net (near-zero = near Nash).
        """
        p = f"inner_loop/{agent_id}"
        self._w(f"{p}/bc_loss",       bc_loss,              iteration)
        self._w(f"{p}/bc_accuracy",   bc_accuracy,          iteration)
        self._w(f"{p}/cfr_iters_run", float(cfr_iters_run), iteration)
        self._w(f"{p}/probe_delta",   probe_delta,          iteration)
        self._w(f"{p}/adv_loss",      adv_loss,             iteration)
        self._w(f"{p}/gru_loss",      gru_loss,             iteration)
        self._w(f"{p}/mean_regret",   mean_regret,          iteration)

    # ------------------------------------------------------------------
    # Timing breakdown
    # ------------------------------------------------------------------

    def log_timings(
        self,
        iteration: int,
        agent_id: str,
        episodes_secs: float,
        policy_train_secs: float,
        profile_build_secs: float,
        sim_secs: float,
        adv_secs: float,
        gru_secs: float,
        pol_secs: float,
    ) -> None:
        """Log per-agent timing breakdown for one XDO iteration (seconds)."""
        p = f"timing/{agent_id}"
        self._w(f"{p}/episodes_secs",      episodes_secs,      iteration)
        self._w(f"{p}/policy_train_secs",  policy_train_secs,  iteration)
        self._w(f"{p}/profile_build_secs", profile_build_secs, iteration)
        self._w(f"{p}/sim_secs",           sim_secs,           iteration)
        self._w(f"{p}/adv_secs",           adv_secs,           iteration)
        self._w(f"{p}/gru_secs",           gru_secs,           iteration)
        self._w(f"{p}/pol_secs",           pol_secs,           iteration)

    # ------------------------------------------------------------------
    # Cross-agent metrics
    # ------------------------------------------------------------------

    def log_metastrategy_divergence(self, iteration: int, l1_divergence: float) -> None:
        """
        Log L1(meta_A, meta_B). Only meaningful when both pools are the same size.
        Trends toward 0 when both agents converge to the same convention mixture,
        and toward 2 when they concentrate on completely different profiles.
        """
        self._w("metastrategy/l1_divergence", l1_divergence, iteration)

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------

    def log_eval(self, joint_score: float) -> None:
        """Log the final 200-episode evaluation score (logged once at step 0)."""
        self._w("eval/joint_score", joint_score, 0)

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush and close the SummaryWriter."""
        self._writer.flush()
        self._writer.close()
