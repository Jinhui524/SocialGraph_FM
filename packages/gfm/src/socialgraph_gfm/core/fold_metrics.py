"""Equal-fold metric derivation from live core prediction records."""

from __future__ import annotations

import math
from dataclasses import dataclass

from socialgraph_gfm.canonical import canonical_sha256

from .fold_evaluation import FoldPredictionRecord
from .metrics import (
    BinaryThreshold,
    TaskMetricSet,
    binary_auprc,
    binary_auroc,
    binary_brier,
    binary_ece,
    binary_metrics_at_threshold,
    filtered_ranking_metrics,
    mean_absolute_error,
    negative_class_auprc,
    recall_at_fixed_fpr,
    spearman_correlation,
)


@dataclass(frozen=True)
class FoldMetricInput:
    prediction: FoldPredictionRecord
    evaluation_scores: tuple[float, ...]
    evaluation_probabilities: tuple[float, ...]
    threshold: BinaryThreshold | None


def _binary_metrics(
    task_id: str,
    fold: FoldMetricInput,
) -> dict[str, float]:
    prediction = fold.prediction
    if (
        fold.threshold is None
        or len(fold.evaluation_scores) != len(prediction.targets)
        or fold.evaluation_scores != fold.evaluation_probabilities
        or not all(0.0 <= value <= 1.0 for value in fold.evaluation_probabilities)
    ):
        raise ValueError("binary fold metrics require aligned calibrated probabilities")
    point = binary_metrics_at_threshold(
        fold.evaluation_scores,
        prediction.targets,
        threshold=fold.threshold,
    )
    if task_id == "tolokers.risk":
        return {
            "auprc": binary_auprc(fold.evaluation_scores, prediction.targets),
            "auroc": binary_auroc(fold.evaluation_scores, prediction.targets),
            "brier": binary_brier(fold.evaluation_probabilities, prediction.targets),
            "ece": binary_ece(fold.evaluation_probabilities, prediction.targets),
            "macroF1": point["macroF1"],
            "recallAtFpr": recall_at_fixed_fpr(
                fold.evaluation_scores, prediction.targets, max_fpr=0.10
            ),
        }
    return {
        "auroc": binary_auroc(fold.evaluation_scores, prediction.targets),
        "macroF1": point["macroF1"],
        "mcc": point["mcc"],
        "negativeAuprc": negative_class_auprc(fold.evaluation_scores, prediction.targets),
    }


def _fold_metrics(task_id: str, fold: FoldMetricInput) -> dict[str, float]:
    prediction = fold.prediction
    if task_id in {"tolokers.risk", "wiki-rfa.vote-sign"}:
        expected_kind = "node-binary" if task_id == "tolokers.risk" else "signed-edge"
        if prediction.task_kind != expected_kind:
            raise ValueError("fold prediction task kind differs from the governance task")
        return _binary_metrics(task_id, fold)
    if task_id == "github.relation-completion":
        if (
            prediction.task_kind != "edge-binary"
            or fold.threshold is not None
            or fold.evaluation_probabilities
            or fold.evaluation_scores != prediction.scores
        ):
            raise ValueError("link fold metrics require exact live raw scores")
        return filtered_ranking_metrics(
            positive_scores=prediction.scores,
            filtered_negative_scores=tuple(
                item.negative_scores for item in prediction.link_candidates
            ),
            hits_at=(10,),
        )
    if task_id == "penn94.community-resilience":
        if (
            prediction.task_kind != "resilience-regression"
            or fold.threshold is not None
            or fold.evaluation_probabilities
            or fold.evaluation_scores != prediction.scores
        ):
            raise ValueError("resilience fold metrics require exact live regression scores")
        return {
            "mae": mean_absolute_error(prediction.scores, prediction.targets),
            "spearman": spearman_correlation(prediction.scores, prediction.targets),
        }
    raise ValueError("equal-fold metrics are unavailable for this task")


def derive_equal_weight_fold_metrics(
    *,
    task_id: str,
    folds: tuple[FoldMetricInput, ...],
) -> TaskMetricSet:
    """Compute each fold independently, then average each metric with equal fold weight."""

    if not folds or any(type(item) is not FoldMetricInput for item in folds):
        raise ValueError("fold metric inventory must contain exact live inputs")
    fold_ids = tuple(item.prediction.fold_id for item in folds)
    if (
        fold_ids != tuple(sorted(set(fold_ids)))
        or len({item.prediction.prediction_hash for item in folds}) != len(folds)
        or any(
            len(item.evaluation_scores) != len(item.prediction.targets)
            or not all(math.isfinite(value) for value in item.evaluation_scores)
            for item in folds
        )
    ):
        raise ValueError("fold inventory must be unique, sorted, finite, and aligned")
    per_fold = tuple(_fold_metrics(task_id, item) for item in folds)
    names = tuple(sorted(per_fold[0]))
    if any(tuple(sorted(item)) != names for item in per_fold):
        raise ValueError("fold metric inventories differ")
    metrics = {name: math.fsum(item[name] for item in per_fold) / len(per_fold) for name in names}
    prediction_hash = canonical_sha256(
        [
            {
                "foldId": item.prediction.fold_id,
                "livePredictionHash": item.prediction.prediction_hash,
                "evaluationScores": item.evaluation_scores,
                "evaluationProbabilities": item.evaluation_probabilities,
            }
            for item in folds
        ]
    )
    target_hash = canonical_sha256(
        [
            {
                "foldId": item.prediction.fold_id,
                "bindingHash": item.prediction.binding_hash,
                "targets": item.prediction.targets,
            }
            for item in folds
        ]
    )
    thresholds = tuple(
        item.threshold.threshold_hash for item in folds if item.threshold is not None
    )
    if thresholds and len(thresholds) != len(folds):
        raise ValueError("threshold inventory must cover every classification fold")
    threshold_hash = (
        None
        if not thresholds
        else canonical_sha256(
            [
                {"foldId": item.prediction.fold_id, "thresholdHash": item.threshold.threshold_hash}
                for item in folds
                if item.threshold is not None
            ]
        )
    )
    return TaskMetricSet.create(
        task_id=task_id,
        metrics=metrics,
        prediction_hash=prediction_hash,
        target_hash=target_hash,
        threshold_hash=threshold_hash,
    )


__all__ = ["FoldMetricInput", "derive_equal_weight_fold_metrics"]
