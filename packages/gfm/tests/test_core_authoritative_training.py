from __future__ import annotations

from typing import Literal

import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.authoritative_training import (
    AuthoritativeFoldTrainValidation,
    AuthoritativeFoldTrainValidationRecord,
    bind_authoritative_supervised_train_validation,
    derive_authoritative_fold_train_validation,
    verify_authoritative_fold_train_validation,
)
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.fold_evaluation import prepare_authoritative_fold
from socialgraph_gfm.core.formal_preflight import ExperimentLabels, ExperimentSplitFold
from socialgraph_gfm.core.supervised import EncodedGraphProvenance, SupervisedPartition


TaskKind = Literal[
    "node-binary",
    "signed-edge",
    "resilience-regression",
    "edge-binary",
]


def _fixture(task_kind: TaskKind, *, invert: bool = False):
    node_ids = tuple(f"n{index:03d}" for index in range(50))
    edge_pairs = tuple(zip(node_ids[:-1], node_ids[1:], strict=True))
    directed = task_kind == "signed-edge"
    assignment_kind = "edge" if task_kind in {"signed-edge", "edge-binary"} else "node"
    inventory = (
        tuple(f"edge:{left}:{right}" for left, right in edge_pairs)
        if assignment_kind == "edge"
        else node_ids
    )
    base = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": node_id, "index": index} for index, node_id in enumerate(node_ids)],
        "edges": [
            {
                "sourceId": left,
                "targetId": right,
                "edgeType": "relation",
                "weight": 1.0,
            }
            for left, right in edge_pairs
        ],
        "nodeFeatures": [
            {
                "kind": "numeric",
                "name": "activity",
                "values": [float(index) for index in range(len(node_ids))],
            }
        ],
        "structuralFeatures": None,
        "source": {"sourceName": "authoritative-training-fixture", "sourceSha256": "1" * 64},
        "splitManifest": {
            "strategy": "all-visible-training",
            "assignments": [{"entityId": entity_id, "role": "train"} for entity_id in inventory],
        },
    }
    base["graphVersionHash"] = calculate_graph_version_hash(base)
    bundle = CoreGraphBundle.model_validate(base)

    fold_inventory = inventory[:49] if assignment_kind == "node" else inventory
    assignments = []
    for index, entity_id in enumerate(fold_inventory):
        role = "train" if index < 44 else "validation" if index < 48 else "test"
        assignments.append({"entityId": entity_id, "role": role})
    if assignment_kind == "node":
        assignments.append({"entityId": inventory[-1], "role": "test"})
    manifest = {
        "strategy": (
            "official"
            if assignment_kind == "node"
            else "signed-pair-stratified-70-15-15"
            if directed
            else "spanning-forest-80-10-10"
        ),
        "assignments": assignments,
    }
    fold = ExperimentSplitFold(
        foldId="official-00",
        splitManifest=manifest,
        splitManifestHash=canonical_sha256(manifest),
    )
    prepared = prepare_authoritative_fold(bundle, fold)

    values = []
    for index, assignment in enumerate(sorted(assignments, key=lambda item: item["entityId"])):
        if task_kind == "node-binary":
            value: int | float = (index + int(invert)) % 2
        elif task_kind == "signed-edge":
            value = 1 if (index + int(invert)) % 2 == 0 else -1
        elif task_kind == "edge-binary":
            value = 1
        else:
            value = float(index) / 10.0 + (1.0 if invert else 0.0)
        values.append({"entityId": assignment["entityId"], "value": value})
    target_name = {
        "node-binary": "banned",
        "signed-edge": "voteSign",
        "edge-binary": "relationCompletion",
        "resilience-regression": "pressure",
    }[task_kind]
    targets = [{"name": target_name, "values": values}]
    labels = ExperimentLabels.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-experiment-labels/1.0",
            "requirementId": "fixture",
            "targets": targets,
            "labelsHash": canonical_sha256(targets),
        }
    )
    return prepared, labels, target_name


def _provenance(graph_version_hash: str, num_nodes: int) -> EncodedGraphProvenance:
    payload = {
        "schemaVersion": "socialgraph-fm.core-encoded-graph/1.0",
        "graphVersionHash": graph_version_hash,
        "modelIdentityHash": "2" * 64,
        "modelIdentityScope": "core-frozen-encoder",
        "adapterSchemaHash": "3" * 64,
        "adapterStateHash": "4" * 64,
        "topologyHash": "5" * 64,
        "topologyTensorHash": "6" * 64,
        "inputTensorHash": "7" * 64,
        "encodedTensorHash": "8" * 64,
        "numNodes": num_nodes,
    }
    payload["artifactHash"] = canonical_sha256(payload)
    return EncodedGraphProvenance.model_validate(payload)


@pytest.mark.parametrize(
    ("task_kind", "expected_values", "uses_nodes"),
    (
        ("node-binary", {0, 1}, True),
        ("signed-edge", {0, 1}, False),
        ("edge-binary", {1}, False),
        ("resilience-regression", None, True),
    ),
)
def test_full_budget_uses_exact_train_and_validation_roles_with_stable_locators(
    task_kind: TaskKind,
    expected_values: set[int] | None,
    uses_nodes: bool,
):
    prepared, labels, target_name = _fixture(task_kind)

    evidence = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind=task_kind,
        label_budget="full",
        experiment_seed=3,
    )

    roles = {item.entity_id: item.role for item in prepared.fold.split_manifest.assignments}
    assert evidence.record.train.entity_ids == tuple(
        sorted(entity_id for entity_id, role in roles.items() if role == "train")
    )
    assert evidence.record.validation.entity_ids == tuple(
        sorted(entity_id for entity_id, role in roles.items() if role == "validation")
    )
    assert bool(evidence.record.train.node_indices) is uses_nodes
    assert bool(evidence.record.train.edge_pairs) is not uses_nodes
    if expected_values is None:
        assert all(type(value) is float for value in evidence.record.train.targets)
    else:
        assert set(evidence.record.train.targets) == expected_values
    assert verify_authoritative_fold_train_validation(evidence) is evidence.record


@pytest.mark.parametrize(
    ("task_kind", "expected_five", "expected_twenty"),
    (
        ("node-binary", 10, 40),
        ("signed-edge", 10, 40),
        ("edge-binary", 5, 20),
        ("resilience-regression", 5, 20),
    ),
)
def test_scarce_budgets_select_exact_canonical_counts_per_task(
    task_kind: TaskKind,
    expected_five: int,
    expected_twenty: int,
):
    prepared, labels, target_name = _fixture(task_kind)

    five = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind=task_kind,
        label_budget="5",
        experiment_seed=2,
    )
    twenty = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind=task_kind,
        label_budget="20",
        experiment_seed=2,
    )

    assert len(five.record.train.entity_ids) == expected_five
    assert len(twenty.record.train.entity_ids) == expected_twenty
    assert five.record.validation == twenty.record.validation


def test_hash_ranked_sampling_is_reproducible_and_seed_bound():
    prepared, labels, target_name = _fixture("node-binary")

    first = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="node-binary",
        label_budget="5",
        experiment_seed=1,
    )
    replay = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="node-binary",
        label_budget="5",
        experiment_seed=1,
    )
    other_seed = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="node-binary",
        label_budget="5",
        experiment_seed=2,
    )

    assert first.record == replay.record
    assert first.record.train.entity_ids != other_seed.record.train.entity_ids
    assert first.record.record_hash != other_seed.record.record_hash


def test_per_class_budget_fails_closed_when_a_class_is_short():
    prepared, labels, target_name = _fixture("node-binary")
    raw = labels.model_dump(mode="python", by_alias=True)
    raw["targets"][0]["values"] = [{**item, "value": 0} for item in raw["targets"][0]["values"]]
    raw["labelsHash"] = canonical_sha256(raw["targets"])
    one_class = ExperimentLabels.model_validate(raw)

    with pytest.raises(ValueError, match="class 1.*20"):
        derive_authoritative_fold_train_validation(
            prepared,
            one_class,
            target_name=target_name,
            task_kind="node-binary",
            label_budget="20",
            experiment_seed=0,
        )


def test_verifier_rejects_caller_target_replacement_even_when_labels_rehash():
    prepared, labels, target_name = _fixture("node-binary")
    evidence = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="node-binary",
        label_budget="5",
        experiment_seed=0,
    )
    _prepared, replacement, _target = _fixture("node-binary", invert=True)
    object.__setattr__(evidence, "labels", replacement)

    with pytest.raises(ValueError, match="changed|derivation"):
        verify_authoritative_fold_train_validation(evidence)


def test_verifier_rejects_validation_role_mixed_into_scarce_training():
    prepared, labels, target_name = _fixture("node-binary")
    evidence = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="node-binary",
        label_budget="5",
        experiment_seed=0,
    )
    record = evidence.record
    injected_id = record.validation.entity_ids[0]
    injected_index = record.validation.node_indices[0]
    rows = list(zip(record.train.entity_ids, record.train.node_indices, record.train.targets))
    rows[-1] = (injected_id, injected_index, rows[-1][2])
    rows.sort(key=lambda item: item[0])
    forged_train = SupervisedPartition(
        entityIds=tuple(item[0] for item in rows),
        nodeIndices=tuple(item[1] for item in rows),
        targets=tuple(item[2] for item in rows),
    )
    forged = record.model_copy()
    object.__setattr__(forged, "train", forged_train)
    raw = forged.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
    object.__setattr__(forged, "record_hash", canonical_sha256(raw))
    object.__setattr__(evidence, "record", forged)

    with pytest.raises(ValueError, match="changed|derivation"):
        verify_authoritative_fold_train_validation(evidence)


def test_verifier_rejects_rehashed_seed_and_target_changes():
    prepared, labels, target_name = _fixture("node-binary")
    evidence = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="node-binary",
        label_budget="5",
        experiment_seed=0,
    )
    raw = evidence.record.model_dump(mode="python", by_alias=True)
    raw["experimentSeed"] = 4
    raw["samplingContextHash"] = canonical_sha256({"forged": True, "experimentSeed": 4})
    raw["recordHash"] = canonical_sha256(
        {key: value for key, value in raw.items() if key != "recordHash"}
    )
    with pytest.raises(ValueError):
        forged = AuthoritativeFoldTrainValidationRecord.model_validate(raw)
        object.__setattr__(evidence, "record", forged)
        verify_authoritative_fold_train_validation(evidence)


def test_supervised_materialization_is_bound_to_the_verified_fold_record():
    prepared, labels, target_name = _fixture("signed-edge")
    evidence = derive_authoritative_fold_train_validation(
        prepared,
        labels,
        target_name=target_name,
        task_kind="signed-edge",
        label_budget="5",
        experiment_seed=0,
    )
    provenance = _provenance(prepared.bundle.graph_version_hash, len(prepared.bundle.nodes))

    data = bind_authoritative_supervised_train_validation(evidence, provenance)

    assert data.train == evidence.record.train
    assert data.validation == evidence.record.validation
    assert data.graph_version_hash == prepared.bundle.graph_version_hash

    wrong = _provenance("f" * 64, len(prepared.bundle.nodes))
    with pytest.raises(ValueError, match="graph identity"):
        bind_authoritative_supervised_train_validation(evidence, wrong)


def test_authoritative_training_evidence_cannot_be_constructed_by_callers():
    with pytest.raises(TypeError, match="emitted only"):
        AuthoritativeFoldTrainValidation()
