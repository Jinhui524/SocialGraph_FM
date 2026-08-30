from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from socialgraph_gfm.core.metrics import (
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
    select_binary_threshold,
    spearman_correlation,
)


def test_literal_binary_metrics_and_validation_threshold() -> None:
    scores = (0.9, 0.8, 0.2, 0.1)
    labels = (1, 0, 1, 0)
    assert binary_auroc(scores, labels) == pytest.approx(0.75)
    assert binary_auprc(scores, labels) == pytest.approx(5 / 6)
    threshold = select_binary_threshold(
        (0.5, 0.4),
        (1, 0),
        validation_partition_hash="d" * 64,
        objective="macro-f1",
    )
    point = binary_metrics_at_threshold(scores, labels, threshold=threshold)
    assert point == {
        "accuracy": pytest.approx(0.5),
        "macroF1": pytest.approx(0.5),
        "mcc": pytest.approx(0.0),
    }
    assert binary_brier(scores, labels) == pytest.approx(0.325)
    assert binary_ece(scores, labels, bin_count=2) == pytest.approx(0.35)
    assert recall_at_fixed_fpr(scores, labels, max_fpr=0.0) == pytest.approx(0.5)

    selected = select_binary_threshold(
        scores,
        labels,
        validation_partition_hash="e" * 64,
        objective="macro-f1",
    )
    assert selected.selection_role == "validation"
    assert selected.validation_partition_hash == "e" * 64
    assert selected.threshold == pytest.approx(0.9)
    assert selected.comparison == "greater-than-or-equal"
    assert selected.validation_score == pytest.approx(11 / 15)
    assert selected.threshold_hash


def test_threshold_selection_can_choose_the_all_negative_frontier() -> None:
    selected = select_binary_threshold(
        (0.9, 0.8, 0.7, 0.1),
        (0, 0, 0, 1),
        validation_partition_hash="f" * 64,
        objective="macro-f1",
    )
    assert selected.threshold == pytest.approx(0.9)
    assert selected.comparison == "greater-than"
    assert selected.validation_score == pytest.approx(3 / 7)
    point = binary_metrics_at_threshold(
        (0.9, 0.8, 0.7, 0.1),
        (0, 0, 0, 1),
        threshold=selected,
    )
    assert point["accuracy"] == pytest.approx(0.75)
    with pytest.raises(TypeError, match="BinaryThreshold"):
        binary_metrics_at_threshold(
            (0.9, 0.1),
            (1, 0),
            threshold=0.5,  # type: ignore[arg-type]
        )


def test_ties_use_group_auc_ap_and_pessimistic_filtered_rank() -> None:
    assert binary_auroc((0.5, 0.5), (1, 0)) == pytest.approx(0.5)
    assert binary_auprc((0.5, 0.5), (1, 0)) == pytest.approx(0.5)
    ranking = filtered_ranking_metrics(
        positive_scores=(0.9, 0.5),
        filtered_negative_scores=((0.8, 0.1), (0.6, 0.5)),
        hits_at=(1, 3),
    )
    assert ranking["filteredMrr"] == pytest.approx(2 / 3)
    assert ranking["hitsAt1"] == pytest.approx(0.5)
    assert ranking["hitsAt3"] == pytest.approx(1.0)
    assert ranking["auprc"] == pytest.approx(0.7)


def test_negative_class_auprc_uses_inverted_labels_and_scores() -> None:
    assert negative_class_auprc((0.9, 0.8, 0.7, 0.6, 0.5), (1, 0, 1, 0, 0)) == pytest.approx(
        11 / 12
    )


def test_regression_metrics_handle_ties_deterministically() -> None:
    assert mean_absolute_error((1.0, 4.0), (2.0, 2.0)) == pytest.approx(1.5)
    assert spearman_correlation((1.0, 2.0, 2.0, 4.0), (1.0, 3.0, 2.0, 4.0)) == pytest.approx(
        0.9486832980505138
    )


@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        ((), (), "nonempty"),
        ((0.1,), (0,), "both classes"),
        ((0.1, math.nan), (0, 1), "finite"),
        ((0.1, 0.2), (0,), "align"),
        ((0.1, 0.2), (0, 2), "binary"),
    ],
)
def test_invalid_binary_inputs_fail_closed(scores, labels, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        binary_auroc(scores, labels)


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: binary_auroc(
                ((0.1,), (0.2,)),  # type: ignore[arg-type]
                (0, 1),
            ),
            "rank one",
        ),
        (
            lambda: binary_auroc(
                (0.1, 0.2),
                ((0,), (1,)),  # type: ignore[arg-type]
            ),
            "rank one",
        ),
        (
            lambda: filtered_ranking_metrics(
                positive_scores=(0.8,), filtered_negative_scores=((),)
            ),
            "at least one negative",
        ),
        (
            lambda: filtered_ranking_metrics(
                positive_scores=(0.8,),
                filtered_negative_scores=(((0.1,),),),  # type: ignore[arg-type]
            ),
            "rank one",
        ),
        (lambda: mean_absolute_error((1.0, math.inf), (1.0, 2.0)), "finite"),
        (lambda: spearman_correlation((1.0, 1.0), (1.0, 2.0)), "non-constant"),
    ],
)
def test_metric_shape_finiteness_and_degenerate_inputs_fail_closed(operation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        operation()


def test_threshold_record_is_partition_and_content_hash_bound() -> None:
    threshold = select_binary_threshold(
        (0.8, 0.2),
        (1, 0),
        validation_partition_hash="1" * 64,
        objective="macro-f1",
    )
    payload = threshold.model_dump(mode="json", by_alias=True)
    payload["validationPartitionHash"] = "2" * 64
    with pytest.raises(ValidationError, match="thresholdHash"):
        BinaryThreshold.model_validate(payload)


def test_metric_set_is_sorted_finite_and_hash_bound() -> None:
    record = TaskMetricSet.create(
        task_id="tolokers",
        metrics={"auroc": 0.75, "auprc": 0.5},
        prediction_hash="a" * 64,
        target_hash="b" * 64,
        threshold_hash="c" * 64,
    )
    assert tuple(item.name for item in record.metrics) == ("auprc", "auroc")
    assert record.metric_set_hash
    payload = record.model_dump(mode="json", by_alias=True)
    payload["metrics"][1]["value"] = float("nan")
    with pytest.raises(ValidationError, match="finite"):
        TaskMetricSet.model_validate(payload)
