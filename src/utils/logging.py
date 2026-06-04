# src/utils/logging.py
# TensorboardLogger — thin wrapper around SummaryWriter for structured
# per-iteration logging of XDO training metrics.
#
# Metrics logged per iteration:
#   agent/{agent_id}/mean_joint_score     — mean team score from shared batch
#   agent/{agent_id}/swap_regret          — per-agent swap regret estimate
#   agent/{agent_id}/team_cf_regret       — gap between best profile and mixture
#   agent/{agent_id}/embedding_drift      — cosine distance after GRU re-encoding
#   metastrategy/l1_divergence            — L1 distance between A's and B's MetaStrategies

from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """Thin wrapper around SummaryWriter for structured per-iteration logging."""

    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir=log_dir)

    def log_iteration(
        self,
        iteration: int,
        agent_id: str,
        mean_joint_score: float,
        swap_regret: float,
        metastrategy_entropy: float,
        top_convention_score: float,
        embedding_drift: float,
    ) -> None:
        """
        Log all per-agent metrics for one XDO iteration.

        Args:
            iteration            : XDO iteration index (1-based).
            agent_id             : "agent_A" or "agent_B".
            mean_joint_score     : Mean team score across the shared episode batch.
            swap_regret          : SR^T = Σ_i π[i] * max_j(score_j − score_i).
                                   Measures gain left on the table by the mixture.
            metastrategy_entropy : Shannon entropy of the MetaStrategy distribution.
                                   High = uniform across profiles; low = concentrated.
            top_convention_score : The stored score of whichever profile currently
                                   holds the most MetaStrategy weight — i.e. the score
                                   of the convention the solver has converged to.
                                   More reliable than tracking the argmax of raw scores
                                   because it reflects what the MetaStrategy actually
                                   chose, not a noisy single-batch outlier.
            embedding_drift      : Cosine distance between old and re-encoded h_oppo.
        """
        prefix = f"agent/{agent_id}"
        self._writer.add_scalar(f"{prefix}/mean_joint_score",     mean_joint_score,     iteration)
        self._writer.add_scalar(f"{prefix}/swap_regret",          swap_regret,          iteration)
        self._writer.add_scalar(f"{prefix}/metastrategy_entropy", metastrategy_entropy, iteration)
        self._writer.add_scalar(f"{prefix}/top_convention_score", top_convention_score, iteration)
        self._writer.add_scalar(f"{prefix}/embedding_drift",      embedding_drift,      iteration)

    def log_profile_metrics(
        self,
        iteration: int,
        agent_id: str,
        bc_loss: float,
        bc_accuracy: float,
        embedding_spread: float,
    ) -> None:
        """
        Log inner-loop quality metrics for the newly added BehaviourProfile.

        Args:
            iteration       : XDO iteration index (1-based).
            agent_id        : "agent_A" or "agent_B".
            bc_loss         : Final-epoch mean cross-entropy of the BC model.
                              Falling loss = BC model learning the partner's policy.
                              Plateau = either converged or data too noisy/sparse.
            bc_accuracy     : Fraction of partner actions correctly predicted by
                              the BC model (in-sample).  Random baseline = 0.125
                              (1/8 actions).  >0.3 suggests meaningful learning.
            embedding_spread: Mean pairwise cosine distance between all profile
                              cached_h_oppo embeddings.  Near 0 = GRU maps all
                              partners to same point (bad).  Growing = GRU learning
                              to distinguish different partner behaviours (good).
        """
        prefix = f"inner_loop/{agent_id}"
        self._writer.add_scalar(f"{prefix}/bc_loss",           bc_loss,           iteration)
        self._writer.add_scalar(f"{prefix}/bc_accuracy",       bc_accuracy,       iteration)
        self._writer.add_scalar(f"{prefix}/embedding_spread",  embedding_spread,  iteration)

    def log_metastrategy_divergence(self, iteration: int, l1_divergence: float) -> None:
        """
        Log the L1 distance between Agent A's and Agent B's MetaStrategies.
        Only valid when both pools have the same number of profiles.

        Args:
            iteration     : XDO iteration index (1-based).
            l1_divergence : sum(|π_A[i] − π_B[i]|) over all profile indices.
        """
        self._writer.add_scalar("metastrategy/l1_divergence", l1_divergence, iteration)

    def close(self) -> None:
        """Flush and close the underlying SummaryWriter."""
        self._writer.close()
