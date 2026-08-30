from __future__ import annotations

from dataclasses import replace

import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.fold_evaluation import FoldPredictionRecord
from socialgraph_gfm.core.fold_metrics import FoldMetricInput, derive_equal_weight_fold_metrics
from socialgraph_gfm.core.metrics import select_binary_threshold


def _binary_prediction(
    fold_id: str,
    *,
    scores: tuple[float, ...],
    targets: tuple[int, ...],
) -> FoldPredictionRecord:
    payload = {
        "schemaVersion": "socialgraph-fm.core-live-fold-predictions/1.0",
        "foldId": fold_id,
        "taskKind": "node-binary",
        "headName": "node_head",
        "graphVersionHash": "1" * 64,
        "bindingHash": canonical_sha256({"fold": fold_id, "targets": targets}),
        "modelIdentityHash": "2" * 64,
        "headStateHash": "3" * 64,
        "adapterSchemaHash": "4" * 64,
        "adapterStateHash": "5" * 64,
        "entityIds": [f"{fold_id}-{index}" for index in range(len(targets))],
        "targets": list(targets),
        "scores": list(scores),
        "probabilities": list(scores),
        "candidateProtocol": None,
        "endpointIds": [],
        "endpointInventoryHash": None,
        "knownPositivePairsHash": None,
        "linkCandidates": [],
    }
    payload["predictionHash"] = canonical_sha256(payload)
    return FoldPredictionRecord.model_validate(payload)


def _input(prediction: FoldPredictionRecord) -> FoldMetricInput:
    threshold = select_binary_threshold(
        prediction.probabilities,
        prediction.targets,
        validation_partition_hash=canonical_sha256(prediction.fold_id),
        objective="macro-f1",
    )
    return FoldMetricInput(
        prediction=prediction,
        evaluation_scores=prediction.probabilities,
        evaluation_probabilities=prediction.probabilities,
        threshold=threshold,
    )


def test_fold_metrics_are_equal_weighted_not_flattened_by_entity_count() -> None:
    small = _binary_prediction(
        "official-00",
        scores=(0.1, 0.9),
        targets=(0, 1),
    )
    large = _binary_prediction(
        "official-01",
        scores=(0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1),
        targets=(0, 0, 0, 0, 1, 1, 1, 1),
    )

    result = derive_equal_weight_fold_metrics(
        task_id="tolokers.risk",
        folds=(_input(small), _input(large)),
    )

    # Fold 0 AUC=1 and fold 1 AUC=0: equal-fold aggregation is exactly 0.5.
    values = {item.name: item.value for item in result.metrics}
    assert values["auroc"] == pytest.approx(0.5)


def test_fold_metric_hashes_bind_every_live_prediction_target_and_threshold() -> None:
    first = _input(_binary_prediction("official-00", scores=(0.1, 0.9), targets=(0, 1)))
    second = _input(_binary_prediction("official-01", scores=(0.2, 0.8), targets=(0, 1)))
    original = derive_equal_weight_fold_metrics(task_id="tolokers.risk", folds=(first, second))

    changed = replace(second, evaluation_scores=(0.8, 0.2), evaluation_probabilities=(0.8, 0.2))
    mutated = derive_equal_weight_fold_metrics(task_id="tolokers.risk", folds=(first, changed))

    assert original.prediction_hash != mutated.prediction_hash
    assert original.target_hash == mutated.target_hash
    assert original.threshold_hash == mutated.threshold_hash


def test_fold_inventory_must_be_unique_sorted_and_task_consistent() -> None:
    first = _input(_binary_prediction("official-00", scores=(0.1, 0.9), targets=(0, 1)))
    with pytest.raises(ValueError, match="fold inventory"):
        derive_equal_weight_fold_metrics(task_id="tolokers.risk", folds=(first, first))
