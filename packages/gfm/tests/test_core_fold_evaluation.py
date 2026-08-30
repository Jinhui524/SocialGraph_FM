from __future__ import annotations

from typing import Literal

import pytest
import torch

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.bundle import (
    CoreGraphBundle,
    calculate_graph_version_hash,
)
from socialgraph_gfm.core.fold_evaluation import (
    LinkCandidateScores,
    VerifiedFoldPredictions,
    bind_authoritative_fold_test,
    infer_core_gfm_fold,
    prepare_authoritative_fold,
    verify_core_gfm_fold_predictions,
)
from socialgraph_gfm.core.formal_preflight import (
    ExperimentLabels,
    ExperimentSplitFold,
)
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.structure_features import STRUCTURE_FEATURE_NAMES


TaskKind = Literal[
    "node-binary",
    "signed-edge",
    "resilience-regression",
    "edge-binary",
]


def _bundle(*, assignment_kind: Literal["node", "edge"], directed: bool = False):
    nodes = tuple("abcde")
    pairs = (
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
        ("d", "e"),
        ("b", "e"),
    )
    if assignment_kind == "node":
        assignments = [{"entityId": node_id, "role": "train"} for node_id in nodes]
        strategy = "all-visible-training"
    else:
        roles = ("train", "train", "validation", "validation", "test", "test")
        assignments = [
            {"entityId": f"edge:{left}:{right}", "role": role}
            for (left, right), role in zip(pairs, roles, strict=True)
        ]
        strategy = "signed-pair-stratified-70-15-15" if directed else "spanning-forest-80-10-10"
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": node_id, "index": index} for index, node_id in enumerate(nodes)],
        "edges": [
            {
                "sourceId": left,
                "targetId": right,
                "edgeType": ("support" if not directed or index % 2 == 0 else "oppose"),
                "weight": 1.0,
            }
            for index, (left, right) in enumerate(pairs)
        ],
        "nodeFeatures": [
            {
                "kind": "numeric",
                "name": "activity",
                "values": [0.0, 1.0, 2.0, 3.0, 4.0],
            }
        ],
        # Deliberately stale values: fold preparation must not copy them.
        "structuralFeatures": {
            "names": ["degree"],
            "values": [[99.0], [99.0], [99.0], [99.0], [99.0]],
        },
        "source": {
            "sourceName": "fold-evaluation-fixture",
            "sourceSha256": "1" * 64,
        },
        "splitManifest": {
            "strategy": strategy,
            "assignments": assignments,
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _fold(
    bundle: CoreGraphBundle,
    *,
    assignment_kind: Literal["node", "edge"],
) -> ExperimentSplitFold:
    if assignment_kind == "node":
        assignments = [
            {"entityId": node_id, "role": role}
            for node_id, role in zip(
                tuple("abcde"),
                ("train", "train", "validation", "test", "test"),
                strict=True,
            )
        ]
        strategy = "official"
    else:
        roles = ("train", "train", "validation", "validation", "test", "test")
        assignments = [
            {"entityId": f"edge:{edge.source_id}:{edge.target_id}", "role": role}
            for edge, role in zip(bundle.edges, roles, strict=True)
        ]
        strategy = bundle.split_manifest.strategy
    manifest = {"strategy": strategy, "assignments": assignments}
    return ExperimentSplitFold(
        foldId="official-00",
        splitManifest=manifest,
        splitManifestHash=canonical_sha256(manifest),
    )


def _labels(
    prepared,
    *,
    target_name: str,
    task_kind: TaskKind,
    omit: str | None = None,
    extra: bool = False,
) -> ExperimentLabels:
    assignments = prepared.bundle.split_manifest.assignments
    values = []
    for index, assignment in enumerate(sorted(assignments, key=lambda item: item.entity_id)):
        if assignment.entity_id == omit:
            continue
        if task_kind == "node-binary":
            value: int | float = index % 2
        elif task_kind == "resilience-regression":
            value = float(index) / 10.0
        elif task_kind == "signed-edge":
            edge = next(
                edge
                for edge in prepared.bundle.edges
                if f"edge:{edge.source_id}:{edge.target_id}" == assignment.entity_id
            )
            value = 1 if edge.edge_type == "support" else -1
        else:
            value = 1
        values.append({"entityId": assignment.entity_id, "value": value})
    if extra:
        values.append({"entityId": "not-in-the-fold", "value": 0})
    values.sort(key=lambda item: str(item["entityId"]))
    targets = [{"name": target_name, "values": values}]
    payload = {
        "schemaVersion": "socialgraph-fm.core-experiment-labels/1.0",
        "requirementId": "fixture",
        "targets": targets,
        "labelsHash": canonical_sha256(targets),
    }
    return ExperimentLabels.model_validate(payload)


def _prepared(task_kind: TaskKind):
    edge_task = task_kind in {"signed-edge", "edge-binary"}
    base = _bundle(
        assignment_kind="edge" if edge_task else "node",
        directed=task_kind == "signed-edge",
    )
    return prepare_authoritative_fold(
        base,
        _fold(base, assignment_kind="edge" if edge_task else "node"),
    )


def test_fold_preparation_replaces_split_and_recomputes_train_visible_structure():
    base = _bundle(assignment_kind="node")
    fold = _fold(base, assignment_kind="node")

    prepared = prepare_authoritative_fold(base, fold)

    assert prepared.bundle.split_manifest == fold.split_manifest
    assert prepared.bundle.graph_version_hash != base.graph_version_hash
    assert prepared.bundle.graph_version_hash == calculate_graph_version_hash(prepared.bundle)
    assert prepared.bundle.structural_features is not None
    assert prepared.bundle.structural_features.names == STRUCTURE_FEATURE_NAMES
    degree = prepared.bundle.structural_features.names.index("degree")
    assert tuple(row[degree] for row in prepared.bundle.structural_features.values) == (
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    assert prepared.record.assignment_kind == "node"
    assert prepared.record.test_entity_ids == ("d", "e")


def test_fold_preparation_rejects_an_incomplete_authoritative_inventory():
    base = _bundle(assignment_kind="node")
    fold = _fold(base, assignment_kind="node")
    manifest = fold.split_manifest.model_dump(mode="python", by_alias=True)
    manifest["assignments"] = manifest["assignments"][:-1]
    broken = ExperimentSplitFold(
        foldId="official-00",
        splitManifest=manifest,
        splitManifestHash=canonical_sha256(manifest),
    )

    with pytest.raises(ValueError, match="exact node or edge inventory"):
        prepare_authoritative_fold(base, broken)


@pytest.mark.parametrize(
    ("task_kind", "target_name", "expected_test_targets"),
    (
        ("node-binary", "banned", (1, 0)),
        ("resilience-regression", "pressure", (0.3, 0.4)),
        ("signed-edge", "voteSign", (0, 1)),
        ("edge-binary", "relationCompletion", (1, 1)),
    ),
)
def test_fold_label_binding_uses_exact_authoritative_test_entities(
    task_kind: TaskKind,
    target_name: str,
    expected_test_targets: tuple[int | float, ...],
):
    prepared = _prepared(task_kind)
    labels = _labels(
        prepared,
        target_name=target_name,
        task_kind=task_kind,
    )

    bound = bind_authoritative_fold_test(
        prepared,
        labels,
        target_name=target_name,
        task_kind=task_kind,
    )

    assert bound.record.entity_ids == prepared.record.test_entity_ids
    assert bound.record.targets == expected_test_targets
    assert bound.record.labels_hash == labels.labels_hash
    if task_kind in {"node-binary", "resilience-regression"}:
        assert bound.record.node_indices
        assert not bound.record.edge_pairs
    else:
        assert bound.record.edge_pairs
        assert not bound.record.node_indices


@pytest.mark.parametrize(("omit", "extra"), (("a", False), (None, True)))
def test_fold_label_binding_rejects_missing_or_extra_label_entities(omit: str | None, extra: bool):
    prepared = _prepared("node-binary")
    labels = _labels(
        prepared,
        target_name="banned",
        task_kind="node-binary",
        omit=omit,
        extra=extra,
    )

    with pytest.raises(ValueError, match="exact authoritative assignment inventory"):
        bind_authoritative_fold_test(
            prepared,
            labels,
            target_name="banned",
            task_kind="node-binary",
        )


def test_fold_label_binding_rejects_locator_role_mismatch():
    prepared = _prepared("node-binary")
    labels = _labels(
        prepared,
        target_name="banned",
        task_kind="node-binary",
    )

    with pytest.raises(ValueError, match="task locator kind"):
        bind_authoritative_fold_test(
            prepared,
            labels,
            target_name="banned",
            task_kind="signed-edge",
        )


@pytest.mark.parametrize(
    ("task_kind", "target_name"),
    (
        ("node-binary", "banned"),
        ("resilience-regression", "pressure"),
        ("signed-edge", "voteSign"),
        ("edge-binary", "relationCompletion"),
    ),
)
def test_verified_fold_predictions_are_derived_from_the_live_model_and_adapter(
    task_kind: TaskKind,
    target_name: str,
):
    torch.manual_seed(7)
    prepared = _prepared(task_kind)
    labels = _labels(
        prepared,
        target_name=target_name,
        task_kind=task_kind,
    )
    bound = bind_authoritative_fold_test(
        prepared,
        labels,
        target_name=target_name,
        task_kind=task_kind,
    )
    adapter = BundleInputAdapter(prepared.bundle, mode="training")
    model = CoreGFM(node_classes=2)

    predictions = infer_core_gfm_fold(model, adapter, bound)

    verify_core_gfm_fold_predictions(model, adapter, bound, predictions)
    assert predictions.record.entity_ids == bound.record.entity_ids
    assert len(predictions.record.scores) == len(bound.record.entity_ids)
    if task_kind == "resilience-regression":
        assert predictions.record.probabilities == ()
    else:
        assert all(0.0 <= value <= 1.0 for value in predictions.record.probabilities)

    head = getattr(model, predictions.record.head_name)
    with torch.no_grad():
        next(head.parameters()).add_(0.25)
    with pytest.raises(ValueError, match="live model and adapter"):
        verify_core_gfm_fold_predictions(model, adapter, bound, predictions)


def test_link_completion_enumerates_every_filtered_endpoint_corruption():
    torch.manual_seed(11)
    prepared = _prepared("edge-binary")
    labels = _labels(
        prepared,
        target_name="relationCompletion",
        task_kind="edge-binary",
    )
    bound = bind_authoritative_fold_test(
        prepared,
        labels,
        target_name="relationCompletion",
        task_kind="edge-binary",
    )
    adapter = BundleInputAdapter(prepared.bundle, mode="training")
    model = CoreGFM(node_classes=2)

    predictions = infer_core_gfm_fold(model, adapter, bound)

    assert predictions.record.endpoint_ids == tuple("abcde")
    assert predictions.record.endpoint_inventory_hash == canonical_sha256(list("abcde"))
    candidates = {
        item.query_entity_id: item.negative_endpoint_pairs
        for item in predictions.record.link_candidates
    }
    assert candidates == {
        "edge:b:e": (("a", "e"), ("b", "d"), ("c", "e")),
        "edge:d:e": (("a", "d"), ("a", "e"), ("b", "d"), ("c", "e")),
    }
    known = {(edge.source_id, edge.target_id) for edge in prepared.bundle.edges}
    assert all(
        pair not in known and pair[0] != pair[1]
        for candidate in predictions.record.link_candidates
        for pair in candidate.negative_endpoint_pairs
    )
    assert all(
        len(candidate.negative_endpoint_pairs) == len(candidate.negative_scores)
        for candidate in predictions.record.link_candidates
    )
    forged = predictions.record.link_candidates[0].model_dump(mode="python", by_alias=True)
    forged["candidateInventoryHash"] = "0" * 64
    with pytest.raises(ValueError, match="candidateInventoryHash"):
        LinkCandidateScores.model_validate(forged)


def test_verified_prediction_artifact_cannot_be_publicly_constructed_or_tampered():
    with pytest.raises(TypeError):
        VerifiedFoldPredictions()  # type: ignore[call-arg]

    torch.manual_seed(13)
    prepared = _prepared("node-binary")
    labels = _labels(
        prepared,
        target_name="banned",
        task_kind="node-binary",
    )
    bound = bind_authoritative_fold_test(
        prepared,
        labels,
        target_name="banned",
        task_kind="node-binary",
    )
    adapter = BundleInputAdapter(prepared.bundle, mode="training")
    model = CoreGFM(node_classes=2)
    predictions = infer_core_gfm_fold(model, adapter, bound)
    object.__setattr__(
        predictions,
        "record",
        predictions.record.model_copy(update={"scores": (123.0, 456.0)}),
    )

    with pytest.raises(ValueError, match="runtime seal|live model and adapter"):
        verify_core_gfm_fold_predictions(model, adapter, bound, predictions)
