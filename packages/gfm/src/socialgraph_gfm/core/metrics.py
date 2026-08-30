"""Deterministic finite metrics and validation-only threshold records."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256


_HASH = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _finite_values(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    normalized_values: list[float] = []
    for value in values:
        dimension = getattr(value, "ndim", None)
        if isinstance(value, (bool, str, bytes, list, tuple)) or (
            dimension is not None and dimension != 0
        ):
            raise ValueError(f"{name} must be rank one and contain numeric scalars")
        try:
            normalized_values.append(float(value))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be rank one and contain numeric scalars") from exc
    normalized = tuple(normalized_values)
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError(f"{name} must contain finite values")
    return normalized


def _binary_inputs(
    scores: Sequence[float], labels: Sequence[int | float], *, require_both: bool = True
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    normalized_scores = _finite_values(scores, name="binary scores")
    normalized_label_values = _finite_values(labels, name="binary labels")
    if len(normalized_scores) != len(normalized_label_values):
        raise ValueError("binary scores and labels must align")
    normalized_labels: list[int] = []
    for value in normalized_label_values:
        if value not in {0.0, 1.0}:
            raise ValueError("binary labels must contain only zero or one")
        normalized_labels.append(int(value))
    result = tuple(normalized_labels)
    if require_both and set(result) != {0, 1}:
        raise ValueError("binary labels must contain both classes")
    return normalized_scores, result


def binary_auroc(scores: Sequence[float], labels: Sequence[int | float]) -> float:
    values, targets = _binary_inputs(scores, labels)
    ordered = sorted(zip(values, targets, strict=True), key=lambda item: item[0])
    positives = sum(targets)
    negatives = len(targets) - positives
    wins = 0.0
    negatives_below = 0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group = ordered[index:end]
        group_positive = sum(label for _score, label in group)
        group_negative = len(group) - group_positive
        wins += group_positive * (negatives_below + 0.5 * group_negative)
        negatives_below += group_negative
        index = end
    return wins / (positives * negatives)


def binary_auprc(scores: Sequence[float], labels: Sequence[int | float]) -> float:
    values, targets = _binary_inputs(scores, labels)
    ordered = sorted(zip(values, targets, strict=True), key=lambda item: item[0], reverse=True)
    positive_count = sum(targets)
    seen = 0
    seen_positive = 0
    average_precision = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        group_positive = sum(label for _score, label in ordered[index:end])
        seen += end - index
        seen_positive += group_positive
        average_precision += (group_positive / positive_count) * (seen_positive / seen)
        index = end
    return average_precision


def negative_class_auprc(scores: Sequence[float], labels: Sequence[int | float]) -> float:
    """Average precision when lower scores denote the signed negative class."""

    values, targets = _binary_inputs(scores, labels)
    return binary_auprc(tuple(-value for value in values), tuple(1 - target for target in targets))


def _confusion(
    scores: Sequence[float],
    labels: Sequence[int | float],
    *,
    threshold: float,
    comparison: Literal["greater-than", "greater-than-or-equal"],
) -> tuple[int, int, int, int]:
    if not math.isfinite(float(threshold)):
        raise ValueError("binary threshold must be finite")
    values, targets = _binary_inputs(scores, labels)
    tp = fp = tn = fn = 0
    for score, target in zip(values, targets, strict=True):
        predicted = int(score > threshold if comparison == "greater-than" else score >= threshold)
        if predicted and target:
            tp += 1
        elif predicted:
            fp += 1
        elif target:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _binary_point_metrics(
    scores: Sequence[float],
    labels: Sequence[int | float],
    *,
    threshold: float,
    comparison: Literal["greater-than", "greater-than-or-equal"],
) -> dict[str, float]:
    tp, fp, tn, fn = _confusion(scores, labels, threshold=threshold, comparison=comparison)
    total = tp + fp + tn + fn
    positive_f1 = _f1(tp, fp, fn)
    negative_f1 = _f1(tn, fn, fp)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = 0.0 if denominator == 0 else (tp * tn - fp * fn) / denominator
    return {
        "accuracy": (tp + tn) / total,
        "macroF1": (positive_f1 + negative_f1) / 2,
        "mcc": mcc,
    }


def binary_metrics_at_threshold(
    scores: Sequence[float],
    labels: Sequence[int | float],
    *,
    threshold: BinaryThreshold,
) -> dict[str, float]:
    """Apply an immutable validation-selected threshold to held-out scores."""

    if type(threshold) is not BinaryThreshold:
        raise TypeError("test metrics require an exact BinaryThreshold record")
    return _binary_point_metrics(
        scores,
        labels,
        threshold=threshold.threshold,
        comparison=threshold.comparison,
    )


def recall_at_fixed_fpr(
    scores: Sequence[float], labels: Sequence[int | float], *, max_fpr: float
) -> float:
    if not 0.0 <= max_fpr <= 1.0 or not math.isfinite(max_fpr):
        raise ValueError("max_fpr must be finite and between zero and one")
    values, targets = _binary_inputs(scores, labels)
    ordered = sorted(zip(values, targets, strict=True), key=lambda item: item[0], reverse=True)
    positives = sum(targets)
    negatives = len(targets) - positives
    tp = fp = 0
    best = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        tp += sum(label for _score, label in ordered[index:end])
        fp += sum(1 - label for _score, label in ordered[index:end])
        if fp / negatives <= max_fpr:
            best = max(best, tp / positives)
        index = end
    return best


def _probability_inputs(
    probabilities: Sequence[float], labels: Sequence[int | float]
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    values, targets = _binary_inputs(probabilities, labels, require_both=False)
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must remain between zero and one")
    return values, targets


def binary_brier(probabilities: Sequence[float], labels: Sequence[int | float]) -> float:
    values, targets = _probability_inputs(probabilities, labels)
    return math.fsum(
        (probability - target) ** 2 for probability, target in zip(values, targets, strict=True)
    ) / len(values)


def binary_ece(
    probabilities: Sequence[float],
    labels: Sequence[int | float],
    *,
    bin_count: int = 10,
) -> float:
    if type(bin_count) is not int or not 2 <= bin_count <= 100:
        raise ValueError("bin_count must be an integer between 2 and 100")
    values, targets = _probability_inputs(probabilities, labels)
    bins: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for probability, target in zip(values, targets, strict=True):
        index = min(int(probability * bin_count), bin_count - 1)
        bins[index].append((probability, target))
    ece = 0.0
    for rows in bins:
        if not rows:
            continue
        confidence = math.fsum(row[0] for row in rows) / len(rows)
        frequency = math.fsum(row[1] for row in rows) / len(rows)
        ece += len(rows) / len(values) * abs(confidence - frequency)
    return ece


class BinaryThreshold(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-binary-threshold/1.0"] = Field(alias="schemaVersion")
    selection_role: Literal["validation"] = Field(alias="selectionRole")
    objective: Literal["macro-f1"]
    threshold: float
    comparison: Literal["greater-than", "greater-than-or-equal"]
    validation_score: float = Field(alias="validationScore", ge=0.0, le=1.0)
    validation_partition_hash: str = Field(alias="validationPartitionHash", pattern=_HASH)
    validation_scores_hash: str = Field(alias="validationScoresHash", pattern=_HASH)
    validation_targets_hash: str = Field(alias="validationTargetsHash", pattern=_HASH)
    threshold_hash: str = Field(alias="thresholdHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_record(self):
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"threshold_hash"})
        )
        if self.threshold_hash != expected:
            raise ValueError("thresholdHash does not match threshold evidence")
        return self


def select_binary_threshold(
    validation_scores: Sequence[float],
    validation_labels: Sequence[int | float],
    *,
    validation_partition_hash: str,
    objective: Literal["macro-f1"],
) -> BinaryThreshold:
    if not isinstance(validation_partition_hash, str) or not re.fullmatch(
        _HASH, validation_partition_hash
    ):
        raise ValueError("validation_partition_hash must be a SHA-256 digest")
    scores, labels = _binary_inputs(validation_scores, validation_labels)
    best_threshold = max(scores)
    best_comparison: Literal["greater-than", "greater-than-or-equal"] = "greater-than"
    best_score = float("-inf")
    candidates: tuple[tuple[float, Literal["greater-than", "greater-than-or-equal"]], ...] = (
        (max(scores), "greater-than"),
    ) + tuple(
        (threshold, "greater-than-or-equal") for threshold in sorted(set(scores), reverse=True)
    )
    for threshold, comparison in candidates:
        metric = _binary_point_metrics(
            scores,
            labels,
            threshold=threshold,
            comparison=comparison,
        )["macroF1"]
        if metric > best_score:
            best_threshold, best_comparison, best_score = threshold, comparison, metric
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-binary-threshold/1.0",
        "selectionRole": "validation",
        "objective": objective,
        "threshold": best_threshold,
        "comparison": best_comparison,
        "validationScore": best_score,
        "validationPartitionHash": validation_partition_hash,
        "validationScoresHash": canonical_sha256(list(scores)),
        "validationTargetsHash": canonical_sha256(list(labels)),
    }
    payload["thresholdHash"] = canonical_sha256(payload)
    return BinaryThreshold.model_validate(payload)


def filtered_ranking_metrics(
    *,
    positive_scores: Sequence[float],
    filtered_negative_scores: Sequence[Sequence[float]],
    hits_at: Sequence[int] = (1, 3, 10),
) -> dict[str, float]:
    positives = _finite_values(positive_scores, name="positive scores")
    if len(positives) != len(filtered_negative_scores):
        raise ValueError("positive and filtered-negative queries must align")
    if not hits_at or any(type(value) is not int or value < 1 for value in hits_at):
        raise ValueError("hits_at values must be positive integers")
    if len(set(hits_at)) != len(hits_at):
        raise ValueError("hits_at values must be unique")
    ranks: list[int] = []
    flattened_scores: list[float] = []
    flattened_labels: list[int] = []
    for positive, raw_negatives in zip(positives, filtered_negative_scores, strict=True):
        raw_negative_values = tuple(raw_negatives)
        if not raw_negative_values:
            raise ValueError("each filtered ranking query requires at least one negative candidate")
        negatives = _finite_values(raw_negative_values, name="filtered negative scores")
        ranks.append(1 + sum(value >= positive for value in negatives))
        flattened_scores.append(positive)
        flattened_labels.append(1)
        flattened_scores.extend(negatives)
        flattened_labels.extend(0 for _ in negatives)
    if not any(label == 0 for label in flattened_labels):
        raise ValueError("filtered ranking requires at least one negative candidate")
    result = {
        "filteredMrr": math.fsum(1.0 / rank for rank in ranks) / len(ranks),
        "auprc": binary_auprc(flattened_scores, flattened_labels),
    }
    for cutoff in sorted(hits_at):
        result[f"hitsAt{cutoff}"] = sum(rank <= cutoff for rank in ranks) / len(ranks)
    return result


def mean_absolute_error(predictions: Sequence[float], targets: Sequence[float]) -> float:
    values = _finite_values(predictions, name="regression predictions")
    expected = _finite_values(targets, name="regression targets")
    if len(values) != len(expected):
        raise ValueError("regression predictions and targets must align")
    return math.fsum(abs(left - right) for left, right in zip(values, expected, strict=True)) / len(
        values
    )


def _average_ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + 1 + end) / 2
        for original in order[index:end]:
            ranks[original] = average
        index = end
    return tuple(ranks)


def spearman_correlation(predictions: Sequence[float], targets: Sequence[float]) -> float:
    values = _finite_values(predictions, name="rank predictions")
    expected = _finite_values(targets, name="rank targets")
    if len(values) != len(expected):
        raise ValueError("rank predictions and targets must align")
    left = _average_ranks(values)
    right = _average_ranks(expected)
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    covariance = math.fsum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right))
    if left_scale == 0.0 or right_scale == 0.0:
        raise ValueError("Spearman correlation requires non-constant ranks")
    return covariance / (left_scale * right_scale)


class MetricValue(_StrictModel):
    name: str = Field(min_length=1)
    value: float

    @model_validator(mode="after")
    def validate_finite(self):
        if not math.isfinite(self.value):
            raise ValueError("metric values must be finite")
        return self


class TaskMetricSet(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-task-metric-set/1.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", min_length=1)
    metrics: tuple[MetricValue, ...] = Field(strict=False, min_length=1)
    prediction_hash: str = Field(alias="predictionHash", pattern=_HASH)
    target_hash: str = Field(alias="targetHash", pattern=_HASH)
    threshold_hash: str | None = Field(default=None, alias="thresholdHash", pattern=_HASH)
    metric_set_hash: str = Field(alias="metricSetHash", pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        task_id: str,
        metrics: dict[str, float],
        prediction_hash: str,
        target_hash: str,
        threshold_hash: str | None,
    ) -> TaskMetricSet:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-task-metric-set/1.0",
            "taskId": task_id,
            "metrics": [{"name": name, "value": value} for name, value in sorted(metrics.items())],
            "predictionHash": prediction_hash,
            "targetHash": target_hash,
            "thresholdHash": threshold_hash,
        }
        payload["metricSetHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_metric_set(self):
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("metric names must be unique and sorted")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"metric_set_hash"})
        )
        if self.metric_set_hash != expected:
            raise ValueError("metricSetHash does not match metric evidence")
        return self


__all__ = [
    "BinaryThreshold",
    "MetricValue",
    "TaskMetricSet",
    "binary_auprc",
    "binary_auroc",
    "binary_brier",
    "binary_ece",
    "binary_metrics_at_threshold",
    "filtered_ranking_metrics",
    "mean_absolute_error",
    "negative_class_auprc",
    "recall_at_fixed_fpr",
    "select_binary_threshold",
    "spearman_correlation",
]
