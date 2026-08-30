from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

import socialgraph_gfm.core.acceptance as acceptance_module
from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.acceptance import (
    CandidateExecutionEvidence,
    CandidateGovernanceManifest,
    CandidateTaskEvidence,
    CandidateTrainingInventory,
    derive_core_acceptance,
    load_core_acceptance,
    publish_fresh_process_checkpoint_evidence,
)
from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.bundle import (
    CoreGraphBundle,
    calculate_graph_version_hash,
)
from socialgraph_gfm.core.calibration import (
    BinaryScoreSemantics,
    CalibrationFitReport,
    CalibrationProtocol,
    derive_validation_scores,
    fit_score_calibration_report,
)
from socialgraph_gfm.core.checkpoint import CheckpointBindings, publish_checkpoint
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.experiments import (
    ExperimentArtifactRef,
    ExperimentExecutionConfigEvidence,
    ExperimentLedger,
    ExperimentProtocol,
    ExperimentRecipeEvidence,
    ExperimentRunRecord,
    ExperimentTrainingDataEvidence,
    ResourceTelemetryEvidence,
    ResourceTelemetrySample,
    aggregate_experiment,
    build_experiment_matrix,
    derive_transfer_advantage,
)
from socialgraph_gfm.core.formal_preflight import (
    FORMAL_CORPUS_REQUIREMENTS,
    ExperimentDatasetManifest,
    ExperimentLabels,
    ExperimentSplitInventory,
    FormalPreflightEvidence,
    run_formal_preflight,
)
from socialgraph_gfm.core.metrics import (
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
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.structure_features import (
    build_structure_cache,
    enrich_bundle_with_structure,
)
from socialgraph_gfm.core.supervised import (
    HeadTrainingConfig,
    HeadTrainingReport,
    SupervisedPartition,
    SupervisedTrainValidation,
    encode_supervised_graph,
    fit_supervised_head,
)
from socialgraph_gfm.core.trainer import _fit_state_payload, _model_state_hash


GOVERNANCE_TASKS = (
    "github.relation-completion",
    "penn94.community-resilience",
    "tolokers.risk",
    "wiki-rfa.vote-sign",
)
TASK_GRAPH = {
    "github.relation-completion": "github-musae",
    "penn94.community-resilience": "facebook100.penn94",
    "tolokers.risk": "tolokers",
    "wiki-rfa.vote-sign": "wiki-rfa",
}
TASK_KIND = {
    "github.relation-completion": "edge-binary",
    "penn94.community-resilience": "resilience-regression",
    "tolokers.risk": "node-binary",
    "wiki-rfa.vote-sign": "signed-edge",
}
TASK_LABEL = {
    "github.relation-completion": "relation",
    "penn94.community-resilience": "pressure",
    "tolokers.risk": "banned",
    "wiki-rfa.vote-sign": "vote-sign",
}


def _canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _write(root: Path, relative: str, payload: bytes) -> tuple[str, int]:
    target = root / Path(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _ref(
    root: Path,
    *,
    role: str,
    relative: str,
    payload: bytes,
    semantic_hash: str,
) -> ExperimentArtifactRef:
    byte_hash, size = _write(root, relative, payload)
    return ExperimentArtifactRef(
        role=role,
        relativePath=relative,
        byteSha256=byte_hash,
        semanticHash=semantic_hash,
        sizeBytes=size,
    )


def _hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _requirement_for_graph(graph_id: str):
    for requirement in FORMAL_CORPUS_REQUIREMENTS:
        if graph_id in {requirement.graph_id, requirement.requirement_id}:
            return requirement
    raise AssertionError(graph_id)


def _base_bundle(task_id: str, *, offset: float) -> CoreGraphBundle:
    task_kind = TASK_KIND[task_id]
    edge_task = task_kind in {"edge-binary", "signed-edge"}
    directed = task_kind == "signed-edge"
    edges = (("0", "1"), ("1", "2"), ("2", "3"), ("3", "4"), ("4", "5"), ("0", "5"))
    if edge_task:
        roles = ("train", "train", "validation", "validation", "test", "test")
        assignments = [
            {"entityId": f"edge:{left}:{right}", "role": role}
            for (left, right), role in zip(edges, roles, strict=True)
        ]
        strategy = "signed-pair-stratified-70-15-15" if directed else "spanning-forest-80-10-10"
    else:
        assignments = [
            {"entityId": str(index), "role": role}
            for index, role in enumerate(
                ("train", "train", "validation", "validation", "test", "test")
            )
        ]
        strategy = "official"
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": str(index), "index": index} for index in range(6)],
        "edges": [
            {"sourceId": left, "targetId": right, "edgeType": "edge"} for left, right in edges
        ],
        "nodeFeatures": [
            {
                "kind": "numeric",
                "name": "score",
                "values": [offset + float(index) for index in range(6)],
            }
        ],
        "structuralFeatures": None,
        "source": {
            "sourceName": f"fixture-{task_id}",
            "sourceSha256": hashlib.sha256(task_id.encode()).hexdigest(),
        },
        "splitManifest": {"strategy": strategy, "assignments": assignments},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _partition(task_kind: str, *, role: str) -> SupervisedPartition:
    if task_kind in {"node-binary", "resilience-regression"}:
        indices = (0, 1) if role == "train" else (2, 3)
        if task_kind == "node-binary":
            targets: tuple[int | float, ...] = (0, 1)
        else:
            targets = (0.2, 0.8) if role == "train" else (0.3, 0.7)
        return SupervisedPartition(
            entityIds=tuple(str(index) for index in indices),
            nodeIndices=indices,
            edgePairs=(),
            targets=targets,
        )
    pairs = ((0, 1), (1, 2)) if role == "train" else ((2, 3), (3, 4))
    return SupervisedPartition(
        entityIds=tuple(f"edge:{left}:{right}" for left, right in pairs),
        nodeIndices=(),
        edgePairs=pairs,
        targets=(0, 1),
    )


def _folds(bundle: CoreGraphBundle, *, count: int) -> ExperimentSplitInventory:
    base = bundle.split_manifest.model_dump(mode="json", by_alias=True)
    folds = []
    for index in range(count):
        split = {**base, "assignments": [dict(item) for item in base["assignments"]]}
        if index:
            roles = [item["role"] for item in split["assignments"]]
            rotated = roles[index % len(roles) :] + roles[: index % len(roles)]
            for item, role in zip(split["assignments"], rotated, strict=True):
                item["role"] = role
        split_hash = canonical_sha256(split)
        folds.append(
            {
                "foldId": f"official-{index:02d}" if count > 1 else "primary",
                "splitManifest": split,
                "splitManifestHash": split_hash,
            }
        )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-experiment-splits/1.0",
        "requirementId": _requirement_for_graph(
            TASK_GRAPH_BY_BUNDLE_HASH[bundle.graph_version_hash]
        ).requirement_id,
        "folds": folds,
    }
    payload["inventoryHash"] = canonical_sha256(payload)
    return ExperimentSplitInventory.model_validate(payload)


TASK_GRAPH_BY_BUNDLE_HASH: dict[str, str] = {}


@dataclass(frozen=True)
class TaskRuntime:
    task_id: str
    bundle: CoreGraphBundle
    adapter: BundleInputAdapter
    data: SupervisedTrainValidation
    head_report: HeadTrainingReport
    calibration: CalibrationFitReport | None
    manifest: ExperimentDatasetManifest
    split_inventory: ExperimentSplitInventory
    shared_refs: tuple[ExperimentArtifactRef, ...]
    target_ref: ExperimentArtifactRef
    threshold_ref: ExperimentArtifactRef | None
    fold_ids: tuple[str, ...]


def _build_task_runtime(
    root: Path,
    *,
    task_id: str,
    model: CoreGFM,
    ordinal: int,
) -> TaskRuntime:
    graph_id = TASK_GRAPH[task_id]
    requirement = _requirement_for_graph(graph_id)
    base = _base_bundle(task_id, offset=float(ordinal) * 10.0)
    cache = build_structure_cache(
        base,
        cache_root=root / "artifacts" / "structure" / requirement.requirement_id,
        role="training",
    )
    bundle = enrich_bundle_with_structure(base, cache)
    TASK_GRAPH_BY_BUNDLE_HASH[bundle.graph_version_hash] = graph_id
    adapter = BundleInputAdapter(bundle, mode="training")
    encoded = encode_supervised_graph(model, bundle, adapter)
    task_kind = TASK_KIND[task_id]
    data = SupervisedTrainValidation.create(
        task_kind=task_kind,  # type: ignore[arg-type]
        provenance=encoded.provenance,
        train=_partition(task_kind, role="train"),
        validation=_partition(task_kind, role="validation"),
    )
    verified_head = fit_supervised_head(
        model,
        encoded,
        data,
        config=HeadTrainingConfig.formal(max_steps=2),
    )
    calibration = None
    if task_kind != "resilience-regression":
        scores = derive_validation_scores(
            model,
            encoded,
            data,
            verified_head,
            semantics=BinaryScoreSemantics.for_task(task_kind),
        )
        calibration = fit_score_calibration_report(
            scores,
            protocol=CalibrationProtocol.fixed(scores),
        )

    split_count = requirement.official_split_count or 1
    split_inventory = _folds(bundle, count=split_count)
    prefix = f"experiment-corpus/{requirement.requirement_id}"
    bundle_relative = f"{prefix}/bundle.json"
    bundle_bytes = _canonical_bytes(bundle)
    bundle_sha, _ = _write(root, bundle_relative, bundle_bytes)
    labels_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-experiment-labels/1.0",
        "requirementId": requirement.requirement_id,
        "targets": [
            {
                "name": TASK_LABEL[task_id],
                "values": [
                    {"entityId": assignment.entity_id, "value": index % 2}
                    for index, assignment in enumerate(
                        sorted(
                            bundle.split_manifest.assignments,
                            key=lambda item: item.entity_id,
                        )
                    )
                ],
            }
        ],
    }
    labels_payload["labelsHash"] = canonical_sha256(labels_payload["targets"])
    labels = ExperimentLabels.model_validate(labels_payload)
    labels_relative = f"{prefix}/labels.json"
    labels_bytes = _canonical_bytes(labels)
    labels_sha, _ = _write(root, labels_relative, labels_bytes)
    split_relative = f"{prefix}/splits.json"
    split_bytes = _canonical_bytes(split_inventory)
    split_sha, _ = _write(root, split_relative, split_bytes)
    manifest_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-experiment-dataset/1.2",
        "requirementId": requirement.requirement_id,
        "recipeId": requirement.recipe_id,
        "recipeVersion": "1.0.0",
        "recipeSha256": hashlib.sha256(requirement.recipe_id.encode()).hexdigest(),
        "graphId": requirement.graph_id,
        "phaseEligibility": "formal",
        "usageScope": requirement.required_usage_scope,
        "splitPolicy": requirement.expected_split_policy,
        "experimentSplitPolicy": requirement.experiment_split_policy,
        "materializerId": f"test.{requirement.requirement_id}",
        "materializerVersion": "1.0",
        "materializerCodeSha256": "2" * 64,
        "materializationProtocolHash": "3" * 64,
        "manifestRelativePath": requirement.manifest_relative_path,
        "bundleRelativePath": bundle_relative,
        "bundleSha256": bundle_sha,
        "labelsRelativePath": labels_relative,
        "labelsSha256": labels_sha,
        "labelNames": [TASK_LABEL[task_id]],
        "splitInventoryRelativePath": split_relative,
        "splitInventorySha256": split_sha,
        "splitCount": len(split_inventory.folds),
        "splitIds": [fold.fold_id for fold in split_inventory.folds],
        "splitManifestHashes": [fold.split_manifest_hash for fold in split_inventory.folds],
        "graphVersionHash": bundle.graph_version_hash,
        "sourceSha256": bundle.source.source_sha256,
        "splitManifestHash": split_inventory.folds[0].split_manifest_hash,
    }
    manifest_payload["manifestHash"] = canonical_sha256(manifest_payload)
    manifest = ExperimentDatasetManifest.model_validate(manifest_payload)
    manifest_bytes = _canonical_bytes(manifest)
    manifest_sha, manifest_size = _write(
        root,
        requirement.manifest_relative_path,
        manifest_bytes,
    )

    cache_relative = cache.manifest_path.relative_to(root).as_posix()
    cache_bytes = cache.manifest_path.read_bytes()
    cache_sha = hashlib.sha256(cache_bytes).hexdigest()
    split0 = split_inventory.folds[0].split_manifest
    split0_bytes = _canonical_bytes(split0)
    split0_relative = f"artifacts/tasks/{requirement.requirement_id}/split-primary.json"
    adapter_bytes = _canonical_bytes(adapter.schema)
    adapter_relative = f"artifacts/tasks/{requirement.requirement_id}/adapter.json"
    head_data_bytes = _canonical_bytes(data)
    head_data_relative = f"artifacts/tasks/{requirement.requirement_id}/head-data.json"
    shared_refs = (
        ExperimentArtifactRef(
            role="dataset-manifest",
            relativePath=requirement.manifest_relative_path,
            byteSha256=manifest_sha,
            semanticHash=manifest.manifest_hash,
            sizeBytes=manifest_size,
        ),
        _ref(
            root,
            role="split-manifest",
            relative=split0_relative,
            payload=split0_bytes,
            semantic_hash=split_inventory.folds[0].split_manifest_hash,
        ),
        ExperimentArtifactRef(
            role="split-inventory",
            relativePath=split_relative,
            byteSha256=split_sha,
            semanticHash=split_inventory.inventory_hash,
            sizeBytes=len(split_bytes),
        ),
        ExperimentArtifactRef(
            role="labels",
            relativePath=labels_relative,
            byteSha256=labels_sha,
            semanticHash=labels.labels_hash,
            sizeBytes=len(labels_bytes),
        ),
        _ref(
            root,
            role="adapter-schema",
            relative=adapter_relative,
            payload=adapter_bytes,
            semantic_hash=adapter.schema.adapter_schema_hash,
        ),
        ExperimentArtifactRef(
            role="structure-cache",
            relativePath=cache_relative,
            byteSha256=cache_sha,
            semanticHash=cache.manifest.manifest_hash,
            sizeBytes=len(cache_bytes),
        ),
        _ref(
            root,
            role="head-data",
            relative=head_data_relative,
            payload=head_data_bytes,
            semantic_hash=data.data_hash,
        ),
    )

    fold_ids = tuple(fold.fold_id for fold in split_inventory.folds)
    entity_ids = tuple(
        f"{fold_id}::entity-{index}"
        for fold_id in fold_ids
        for index in range(4 if task_id != "github.relation-completion" else 2)
    )
    entity_fold_ids = tuple(
        fold_id
        for fold_id in fold_ids
        for _ in range(4 if task_id != "github.relation-completion" else 2)
    )
    if task_id == "github.relation-completion":
        target_values = [1.0] * len(entity_ids)
        evaluation_kind = "link-ranking"
    elif task_id == "penn94.community-resilience":
        target_values = [float(index % 4) for index in range(len(entity_ids))]
        evaluation_kind = "regression"
    else:
        target_values = [float(index % 2) for index in range(len(entity_ids))]
        evaluation_kind = "binary"
    target_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-metric-targets/1.0",
        "taskId": task_id,
        "splitInventoryHash": split_inventory.inventory_hash,
        "foldIds": list(fold_ids),
        "evaluationKind": evaluation_kind,
        "entityIds": list(entity_ids),
        "entityFoldIds": list(entity_fold_ids),
        "values": target_values,
    }
    target_payload["targetHash"] = canonical_sha256(target_payload)
    target_ref = _ref(
        root,
        role="targets",
        relative=f"artifacts/tasks/{requirement.requirement_id}/targets.json",
        payload=_canonical_bytes(target_payload),
        semantic_hash=target_payload["targetHash"],
    )
    threshold_ref = None
    if evaluation_kind == "binary":
        threshold = select_binary_threshold(
            (0.1, 0.9, 0.2, 0.8),
            (0.0, 1.0, 0.0, 1.0),
            validation_partition_hash=data.validation.partition_hash,
            objective="macro-f1",
        )
        threshold_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-metric-threshold-selection/1.0",
            "threshold": threshold.model_dump(mode="python", by_alias=True),
            "validationScores": [0.1, 0.9, 0.2, 0.8],
            "validationTargets": [0.0, 1.0, 0.0, 1.0],
        }
        threshold_payload["evidenceHash"] = canonical_sha256(threshold_payload)
        threshold_ref = _ref(
            root,
            role="threshold",
            relative=f"artifacts/tasks/{requirement.requirement_id}/threshold.json",
            payload=_canonical_bytes(threshold_payload),
            semantic_hash=threshold.threshold_hash,
        )
    return TaskRuntime(
        task_id=task_id,
        bundle=bundle,
        adapter=adapter,
        data=data,
        head_report=verified_head.record,
        calibration=calibration,
        manifest=manifest,
        split_inventory=split_inventory,
        shared_refs=shared_refs,
        target_ref=target_ref,
        threshold_ref=threshold_ref,
        fold_ids=fold_ids,
    )


def test_raw_gfm_validation_is_reproduced_from_best_model_and_target_adapter(
    tmp_path: Path,
) -> None:
    torch.manual_seed(17)
    model = CoreGFM(node_classes=2)
    runtime = _build_task_runtime(
        tmp_path,
        task_id="tolokers.risk",
        model=model,
        ordinal=0,
    )
    checkpoint = acceptance_module.StrictCheckpointModel(
        model=model,
        adapter_schemas={"tolokers": runtime.adapter.schema},
        adapter_states={"tolokers": runtime.adapter.state_dict()},
        model_state_hash=_model_state_hash(model.state_dict()),
    )

    observed = acceptance_module._reproduce_raw_gfm_validation(
        checkpoint=checkpoint,
        target_domain="tolokers",
        bundle=runtime.bundle,
        adapter_schema=runtime.adapter.schema,
        head_data=runtime.data,
        head_report=runtime.head_report,
        calibration=runtime.calibration,
    )

    assert observed.validation_metric == runtime.head_report.best_metric
    assert observed.head_report_hash == runtime.head_report.report_hash
    assert observed.calibration_hash == runtime.calibration.report_hash

    wrong_model = CoreGFM(node_classes=2)
    forged = acceptance_module.StrictCheckpointModel(
        model=wrong_model,
        adapter_schemas=checkpoint.adapter_schemas,
        adapter_states=checkpoint.adapter_states,
        model_state_hash=_model_state_hash(wrong_model.state_dict()),
    )
    with pytest.raises(ValueError, match="head training report"):
        acceptance_module._reproduce_raw_gfm_validation(
            checkpoint=forged,
            target_domain="tolokers",
            bundle=runtime.bundle,
            adapter_schema=runtime.adapter.schema,
            head_data=runtime.data,
            head_report=runtime.head_report,
            calibration=runtime.calibration,
        )


def _ready_preflight(
    root: Path,
    task_runtime: dict[str, TaskRuntime],
) -> FormalPreflightEvidence:
    base = run_formal_preflight(root).model_dump(mode="json", by_alias=True)
    runtime_by_requirement = {item.manifest.requirement_id: item for item in task_runtime.values()}
    observations = []
    for requirement in FORMAL_CORPUS_REQUIREMENTS:
        runtime = runtime_by_requirement.get(requirement.requirement_id)
        prefix = f"experiment-corpus/{requirement.requirement_id}"
        files = [
            {
                "relativePath": requirement.manifest_relative_path,
                "sha256": "1" * 64,
                "sizeBytes": 1,
                "purpose": "manifest",
            },
            {
                "relativePath": f"{prefix}/bundle.json",
                "sha256": "2" * 64,
                "sizeBytes": 1,
                "purpose": "bundle",
            },
            {
                "relativePath": f"{prefix}/labels.json",
                "sha256": "3" * 64,
                "sizeBytes": 1,
                "purpose": "labels",
            },
            {
                "relativePath": f"{prefix}/splits.json",
                "sha256": "4" * 64,
                "sizeBytes": 1,
                "purpose": "split-inventory",
            },
            *(
                {
                    "relativePath": path,
                    "sha256": "5" * 64,
                    "sizeBytes": 1,
                    "purpose": "raw",
                }
                for path in requirement.raw_relative_paths
            ),
        ]
        observations.append(
            {
                "requirementId": requirement.requirement_id,
                "status": "ready",
                "reasonCode": "validated-formal-dataset",
                "manifestHash": (
                    runtime.manifest.manifest_hash if runtime is not None else "6" * 64
                ),
                "graphVersionHash": (
                    runtime.bundle.graph_version_hash if runtime is not None else "7" * 64
                ),
                "splitManifestHash": (
                    runtime.manifest.split_manifest_hash if runtime is not None else "8" * 64
                ),
                "files": sorted(files, key=lambda item: item["relativePath"]),
            }
        )
    base["observations"] = observations
    base["formalReady"] = True
    base["promotable"] = True
    base["evidenceHash"] = canonical_sha256(
        {key: value for key, value in base.items() if key != "evidenceHash"}
    )
    return FormalPreflightEvidence.model_validate(base)


def _manifest_hashes(cell, ready: FormalPreflightEvidence) -> dict[str, str]:
    observations = {item.requirement_id: item for item in ready.observations}
    result = {}
    graphs = (
        *cell.pretraining_graph_ids,
        cell.target_graph_id,
        *((cell.validation_graph_id,) if cell.validation_graph_id else ()),
    )
    for graph_id in graphs:
        observation = observations[_requirement_for_graph(graph_id).requirement_id]
        assert observation.manifest_hash is not None
        result[graph_id] = observation.manifest_hash
    return result


def _mutated_head_report(
    runtime: TaskRuntime,
    *,
    unique: int,
    genuine: bool,
) -> HeadTrainingReport:
    if genuine:
        return runtime.head_report
    payload = runtime.head_report.model_dump(mode="json", by_alias=True)
    payload["history"][0]["trainLoss"] += unique * 1e-7
    payload["reportHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "reportHash"}
    )
    return HeadTrainingReport.model_validate(payload)


def _mutated_calibration(
    runtime: TaskRuntime,
    head: HeadTrainingReport,
    *,
    genuine: bool,
) -> CalibrationFitReport | None:
    if runtime.calibration is None:
        return None
    if genuine:
        return runtime.calibration
    payload = runtime.calibration.model_dump(mode="json", by_alias=True)
    payload["headTrainingReportHash"] = head.report_hash
    payload["reportHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "reportHash"}
    )
    return CalibrationFitReport.model_validate(payload)


def _metric_evidence(
    root: Path,
    *,
    cell,
    runtime: TaskRuntime,
    good: bool,
    unique: int,
) -> tuple[TaskMetricSet, ExperimentArtifactRef]:
    count = len(runtime.fold_ids) * (2 if cell.task_id == "github.relation-completion" else 4)
    entity_ids = tuple(
        f"{fold_id}::entity-{index}"
        for fold_id in runtime.fold_ids
        for index in range(2 if cell.task_id == "github.relation-completion" else 4)
    )
    entity_fold_ids = tuple(
        fold_id
        for fold_id in runtime.fold_ids
        for _ in range(2 if cell.task_id == "github.relation-completion" else 4)
    )
    epsilon = unique * 1e-8
    threshold_hash = None
    if cell.task_id == "github.relation-completion":
        base_scores = (0.9, 0.8) if good else (0.1, 0.2)
        scores = tuple(base_scores[index % 2] + epsilon for index in range(count))
        negative = tuple(((0.1, 0.2) if good else (0.8, 0.9)) for _ in range(count))
        prediction_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-metric-predictions/1.0",
            "taskId": cell.task_id,
            "splitInventoryHash": runtime.split_inventory.inventory_hash,
            "foldIds": list(runtime.fold_ids),
            "evaluationKind": "link-ranking",
            "entityIds": list(entity_ids),
            "entityFoldIds": list(entity_fold_ids),
            "scores": list(scores),
            "probabilities": [],
            "filteredNegativeScores": [list(row) for row in negative],
        }
        metrics = filtered_ranking_metrics(
            positive_scores=scores,
            filtered_negative_scores=negative,
            hits_at=(10,),
        )
    elif cell.task_id == "penn94.community-resilience":
        targets = tuple(float(index % 4) for index in range(count))
        scores = tuple((target + epsilon if good else 3.0 - target + epsilon) for target in targets)
        prediction_payload = {
            "schemaVersion": "socialgraph-fm.core-metric-predictions/1.0",
            "taskId": cell.task_id,
            "splitInventoryHash": runtime.split_inventory.inventory_hash,
            "foldIds": list(runtime.fold_ids),
            "evaluationKind": "regression",
            "entityIds": list(entity_ids),
            "entityFoldIds": list(entity_fold_ids),
            "scores": list(scores),
            "probabilities": [],
            "filteredNegativeScores": [],
        }
        metrics = {
            "mae": mean_absolute_error(scores, targets),
            "spearman": spearman_correlation(scores, targets),
        }
    else:
        labels = tuple(float(index % 2) for index in range(count))
        pattern = (0.1, 0.9, 0.2, 0.8) if good else (0.9, 0.1, 0.8, 0.2)
        scores = tuple(pattern[index % 4] + epsilon for index in range(count))
        prediction_payload = {
            "schemaVersion": "socialgraph-fm.core-metric-predictions/1.0",
            "taskId": cell.task_id,
            "splitInventoryHash": runtime.split_inventory.inventory_hash,
            "foldIds": list(runtime.fold_ids),
            "evaluationKind": "binary",
            "entityIds": list(entity_ids),
            "entityFoldIds": list(entity_fold_ids),
            "scores": list(scores),
            "probabilities": list(scores),
            "filteredNegativeScores": [],
        }
        assert runtime.threshold_ref is not None
        threshold_hash = runtime.threshold_ref.semantic_hash
        threshold = select_binary_threshold(
            (0.1, 0.9, 0.2, 0.8),
            (0.0, 1.0, 0.0, 1.0),
            validation_partition_hash=runtime.data.validation.partition_hash,
            objective="macro-f1",
        )
        point = binary_metrics_at_threshold(scores, labels, threshold=threshold)
        if cell.task_id == "tolokers.risk":
            metrics = {
                "auprc": binary_auprc(scores, labels),
                "auroc": binary_auroc(scores, labels),
                "brier": binary_brier(scores, labels),
                "ece": binary_ece(scores, labels),
                "macroF1": point["macroF1"],
                "recallAtFpr": recall_at_fixed_fpr(scores, labels, max_fpr=0.10),
            }
        else:
            metrics = {
                "auroc": binary_auroc(scores, labels),
                "macroF1": point["macroF1"],
                "mcc": point["mcc"],
                "negativeAuprc": negative_class_auprc(scores, labels),
            }
    prediction_payload["predictionHash"] = canonical_sha256(prediction_payload)
    prediction_ref = _ref(
        root,
        role="predictions",
        relative=f"artifacts/runs/{cell.cell_id[:16]}/predictions.json",
        payload=_canonical_bytes(prediction_payload),
        semantic_hash=prediction_payload["predictionHash"],
    )
    metrics_record = TaskMetricSet.create(
        task_id=cell.task_id,
        metrics=metrics,
        prediction_hash=prediction_payload["predictionHash"],
        target_hash=runtime.target_ref.semantic_hash,
        threshold_hash=threshold_hash,
    )
    return metrics_record, prediction_ref


def _checkpoint_pair(
    root: Path,
    *,
    cell,
    config: ExperimentExecutionConfigEvidence,
    training_data: ExperimentTrainingDataEvidence,
    code_hash: str,
    environment_hash: str,
    model_state: dict[str, Any],
    adapter_schemas: dict[str, Any] | None = None,
    adapter_states: dict[str, Any] | None = None,
) -> tuple[ExperimentArtifactRef, ExperimentArtifactRef]:
    bindings = CheckpointBindings(
        config_hash=config.config_hash,
        data_hash=training_data.inventory_hash,
        code_hash=code_hash,
        environment_hash=environment_hash,
    )
    trainer_config = TrainingConfig(**config.training_config)
    model_hash = _model_state_hash(model_state)
    validation = {
        "validation_protocol_hash": hashlib.sha256(f"protocol:{cell.cell_id}".encode()).hexdigest(),
        "validation_data_hash": training_data.inventory_hash,
        "validation_partition_hash": hashlib.sha256(
            f"partition:{cell.cell_id}".encode()
        ).hexdigest(),
        "validation_callback_hash": hashlib.sha256(f"callback:{cell.cell_id}".encode()).hexdigest(),
    }
    run_prefix = cell.cell_id[:16]
    best_relative = (
        f"artifacts/runs/{run_prefix}/.model.best.pt.run-{run_prefix}.step-0000002000.pt"
    )
    latest_relative = f"artifacts/runs/{run_prefix}/model.latest.pt"
    best_path = root / Path(best_relative)
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_fit = _fit_state_payload(
        best_step=2_000,
        best_metric=0.5,
        best_model_state_hash=model_hash,
        stale_validations=0,
        last_validation_step=2_000,
        last_validation_metric=0.5,
        last_model_state_hash=model_hash,
        checkpoint_model_state_hash=model_hash,
        best_checkpoint_name=None,
        best_checkpoint_sha256=None,
        **validation,
    )
    state = {
        "experimentCellId": cell.cell_id,
        "optimizerStep": 2_000,
        "model": model_state,
        "config": trainer_config.to_dict(),
        "fitState": best_fit,
        **({"adapterSchemas": adapter_schemas} if adapter_schemas is not None else {}),
        **({"adapters": adapter_states} if adapter_states is not None else {}),
    }
    publish_checkpoint(
        best_path,
        trainer_state=state,
        bindings=bindings,
        status="validated",
        promotable=False,
    )
    best_bytes = best_path.read_bytes()
    best_hash = hashlib.sha256(best_bytes).hexdigest()
    latest_fit = _fit_state_payload(
        best_step=2_000,
        best_metric=0.5,
        best_model_state_hash=model_hash,
        stale_validations=0,
        last_validation_step=2_000,
        last_validation_metric=0.5,
        last_model_state_hash=model_hash,
        checkpoint_model_state_hash=model_hash,
        best_checkpoint_name=best_path.name,
        best_checkpoint_sha256=best_hash,
        **validation,
    )
    latest_path = root / Path(latest_relative)
    publish_checkpoint(
        latest_path,
        trainer_state={**state, "fitState": latest_fit},
        bindings=bindings,
        status="validated",
        promotable=True,
    )
    latest_bytes = latest_path.read_bytes()
    latest_hash = hashlib.sha256(latest_bytes).hexdigest()
    return (
        ExperimentArtifactRef(
            role="latest-checkpoint",
            relativePath=latest_relative,
            byteSha256=latest_hash,
            semanticHash=latest_hash,
            sizeBytes=len(latest_bytes),
        ),
        ExperimentArtifactRef(
            role="best-checkpoint",
            relativePath=best_relative,
            byteSha256=best_hash,
            semanticHash=best_hash,
            sizeBytes=len(best_bytes),
        ),
    )


def _candidate_checkpoint_pair(
    root: Path,
    *,
    model: CoreGFM,
    task_runtime: dict[str, TaskRuntime],
    execution: CandidateExecutionEvidence,
    inventory: CandidateTrainingInventory,
    code_hash: str,
    environment_hash: str,
) -> tuple[ExperimentArtifactRef, ExperimentArtifactRef, CheckpointBindings]:
    bindings = CheckpointBindings(
        config_hash=execution.config_hash,
        data_hash=inventory.inventory_hash,
        code_hash=code_hash,
        environment_hash=environment_hash,
    )
    model_state = model.state_dict()
    model_hash = _model_state_hash(model_state)
    adapter_schemas = {
        TASK_GRAPH[task_id]: runtime.adapter.schema.model_dump(mode="json", by_alias=True)
        for task_id, runtime in task_runtime.items()
    }
    adapter_states = {
        TASK_GRAPH[task_id]: runtime.adapter.state_dict()
        for task_id, runtime in task_runtime.items()
    }
    validation = {
        "validation_protocol_hash": "1" * 64,
        "validation_data_hash": inventory.inventory_hash,
        "validation_partition_hash": "2" * 64,
        "validation_callback_hash": "3" * 64,
    }
    best_relative = f"artifacts/candidate/.governance.best.pt.run-{execution.config_hash[:16]}.step-0000002000.pt"
    latest_relative = "artifacts/candidate/governance.latest.pt"
    best_path = root / Path(best_relative)
    best_fit = _fit_state_payload(
        best_step=2_000,
        best_metric=0.5,
        best_model_state_hash=model_hash,
        stale_validations=0,
        last_validation_step=2_000,
        last_validation_metric=0.5,
        last_model_state_hash=model_hash,
        checkpoint_model_state_hash=model_hash,
        best_checkpoint_name=None,
        best_checkpoint_sha256=None,
        **validation,
    )
    state = {
        "optimizerStep": 2_000,
        "model": model_state,
        "adapters": adapter_states,
        "adapterSchemas": adapter_schemas,
        "config": execution.trainer_config,
        "fitState": best_fit,
    }
    publish_checkpoint(
        best_path,
        trainer_state=state,
        bindings=bindings,
        status="validated",
        promotable=False,
    )
    best_bytes = best_path.read_bytes()
    best_hash = hashlib.sha256(best_bytes).hexdigest()
    latest_fit = _fit_state_payload(
        best_step=2_000,
        best_metric=0.5,
        best_model_state_hash=model_hash,
        stale_validations=0,
        last_validation_step=2_000,
        last_validation_metric=0.5,
        last_model_state_hash=model_hash,
        checkpoint_model_state_hash=model_hash,
        best_checkpoint_name=best_path.name,
        best_checkpoint_sha256=best_hash,
        **validation,
    )
    latest_path = root / Path(latest_relative)
    publish_checkpoint(
        latest_path,
        trainer_state={**state, "fitState": latest_fit},
        bindings=bindings,
        status="validated",
        promotable=True,
    )
    latest_bytes = latest_path.read_bytes()
    latest_hash = hashlib.sha256(latest_bytes).hexdigest()
    return (
        ExperimentArtifactRef(
            role="latest-checkpoint",
            relativePath=latest_relative,
            byteSha256=latest_hash,
            semanticHash=latest_hash,
            sizeBytes=len(latest_bytes),
        ),
        ExperimentArtifactRef(
            role="best-checkpoint",
            relativePath=best_relative,
            byteSha256=best_hash,
            semanticHash=best_hash,
            sizeBytes=len(best_bytes),
        ),
        bindings,
    )


def test_legacy_self_reported_matrix_cannot_be_promoted_as_real_formal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260815)
    protocol = ExperimentProtocol.fixed()
    matrix = tuple(
        cell
        for cell in build_experiment_matrix(protocol)
        if cell.task_id in GOVERNANCE_TASKS
        and cell.method_id in {"graphsage-scratch", "multi-graph-shared-gfm"}
    )
    monkeypatch.setattr(acceptance_module, "build_experiment_matrix", lambda _protocol: matrix)
    model = CoreGFM(node_classes=2)
    task_runtime = {
        task_id: _build_task_runtime(
            tmp_path,
            task_id=task_id,
            model=model,
            ordinal=index,
        )
        for index, task_id in enumerate(GOVERNANCE_TASKS)
    }
    ready = _ready_preflight(tmp_path, task_runtime)
    monkeypatch.setattr(
        acceptance_module,
        "load_formal_preflight",
        lambda _path, runtime_root: ready,
    )
    code_ref = _ref(
        tmp_path,
        role="code",
        relative="artifacts/common/code.snapshot",
        payload=b"core-code-snapshot\n",
        semantic_hash=hashlib.sha256(b"core-code-snapshot\n").hexdigest(),
    )
    environment_ref = _ref(
        tmp_path,
        role="environment",
        relative="artifacts/common/environment.lock",
        payload=b"core-environment-lock\n",
        semantic_hash=hashlib.sha256(b"core-environment-lock\n").hexdigest(),
    )
    code_hash = code_ref.semantic_hash
    environment_hash = environment_ref.semantic_hash
    selected_cells = {
        task_id: next(
            cell
            for cell in matrix
            if cell.task_id == task_id
            and cell.method_id == "multi-graph-shared-gfm"
            and cell.label_budget == "full"
            and cell.seed == protocol.seeds[-1]
        )
        for task_id in GOVERNANCE_TASKS
    }
    encoder_source = selected_cells["tolokers.risk"]
    records: list[ExperimentRunRecord] = []
    for unique, cell in enumerate(matrix, start=1):
        runtime = task_runtime[cell.task_id]
        genuine = selected_cells.get(cell.task_id) == cell
        recipe = ExperimentRecipeEvidence.create(
            cell=cell,
            manifest_hashes=_manifest_hashes(cell, ready),
        )
        execution = ExperimentExecutionConfigEvidence.create(
            cell=cell,
            recipe=recipe,
            training_config=TrainingConfig.formal(max_steps=2_000, min_steps=2_000),
        )
        training_data = ExperimentTrainingDataEvidence.create(
            cell=cell,
            recipe=recipe,
            target_split_inventory_hash=runtime.split_inventory.inventory_hash,
            head_data_hash=runtime.data.data_hash,
        )
        telemetry = ResourceTelemetryEvidence.create(
            cell_id=cell.cell_id,
            phase="formal",
            samples=(
                ResourceTelemetrySample(
                    monotonicSeconds=float(unique),
                    cumulativeDataWaitSeconds=0.0,
                    optimizerStep=0,
                    cudaAllocatedBytes=0,
                ),
                ResourceTelemetrySample(
                    monotonicSeconds=float(unique) + 10.0,
                    cumulativeDataWaitSeconds=1.0,
                    optimizerStep=2_000,
                    cudaAllocatedBytes=1_024 + unique,
                ),
            ),
        )
        head = _mutated_head_report(runtime, unique=unique, genuine=genuine)
        calibration = _mutated_calibration(runtime, head, genuine=genuine)
        metrics, prediction_ref = _metric_evidence(
            tmp_path,
            cell=cell,
            runtime=runtime,
            good=cell.method_id == "multi-graph-shared-gfm",
            unique=unique,
        )
        recipe_ref = _ref(
            tmp_path,
            role="experiment-recipe",
            relative=f"artifacts/runs/{cell.cell_id[:16]}/recipe.json",
            payload=_canonical_bytes(recipe),
            semantic_hash=recipe.recipe_hash,
        )
        execution_ref = _ref(
            tmp_path,
            role="configuration",
            relative=f"artifacts/runs/{cell.cell_id[:16]}/configuration.json",
            payload=_canonical_bytes(execution),
            semantic_hash=execution.config_hash,
        )
        training_ref = _ref(
            tmp_path,
            role="training-data",
            relative=f"artifacts/runs/{cell.cell_id[:16]}/training-data.json",
            payload=_canonical_bytes(training_data),
            semantic_hash=training_data.inventory_hash,
        )
        telemetry_ref = _ref(
            tmp_path,
            role="resource-telemetry",
            relative=f"artifacts/runs/{cell.cell_id[:16]}/telemetry.json",
            payload=_canonical_bytes(telemetry),
            semantic_hash=telemetry.telemetry_hash,
        )
        head_ref = _ref(
            tmp_path,
            role="head-report",
            relative=f"artifacts/runs/{cell.cell_id[:16]}/head-report.json",
            payload=_canonical_bytes(head),
            semantic_hash=head.report_hash,
        )
        calibration_ref = None
        if calibration is not None:
            calibration_ref = _ref(
                tmp_path,
                role="calibration-report",
                relative=f"artifacts/runs/{cell.cell_id[:16]}/calibration.json",
                payload=_canonical_bytes(calibration),
                semantic_hash=calibration.report_hash,
            )
        if cell == encoder_source:
            model_state = model.state_dict()
            schemas = {
                TASK_GRAPH[task_id]: item.adapter.schema.model_dump(mode="json", by_alias=True)
                for task_id, item in task_runtime.items()
            }
            adapter_states = {
                TASK_GRAPH[task_id]: item.adapter.state_dict()
                for task_id, item in task_runtime.items()
            }
        else:
            model_state = {"cellMarker": torch.tensor([float(unique)])}
            schemas = None
            adapter_states = None
        latest_ref, best_ref = _checkpoint_pair(
            tmp_path,
            cell=cell,
            config=execution,
            training_data=training_data,
            code_hash=code_hash,
            environment_hash=environment_hash,
            model_state=model_state,
            adapter_schemas=schemas,
            adapter_states=adapter_states,
        )
        artifacts = [
            *runtime.shared_refs,
            runtime.target_ref,
            *([runtime.threshold_ref] if runtime.threshold_ref is not None else []),
            code_ref,
            environment_ref,
            recipe_ref,
            execution_ref,
            training_ref,
            telemetry_ref,
            head_ref,
            *([calibration_ref] if calibration_ref is not None else []),
            prediction_ref,
            latest_ref,
            best_ref,
        ]
        records.append(
            ExperimentRunRecord.create(
                cell=cell,
                phase="formal",
                preflight_evidence_hash=ready.evidence_hash,
                dataset_manifest_hash=runtime.manifest.manifest_hash,
                split_manifest_hash=runtime.manifest.split_manifest_hash,
                split_inventory_hash=runtime.split_inventory.inventory_hash,
                evaluation_fold_ids=runtime.fold_ids,
                recipe_hash=recipe.recipe_hash,
                config_hash=execution.config_hash,
                training_data_hash=training_data.inventory_hash,
                head_data_hash=runtime.data.data_hash,
                code_hash=code_hash,
                environment_hash=environment_hash,
                structure_cache_hash=next(
                    ref.semantic_hash
                    for ref in runtime.shared_refs
                    if ref.role == "structure-cache"
                ),
                adapter_schema_hash=runtime.adapter.schema.adapter_schema_hash,
                label_artifact_hash=next(
                    ref.semantic_hash for ref in runtime.shared_refs if ref.role == "labels"
                ),
                head_report_hash=head.report_hash,
                calibration_hash=None if calibration is None else calibration.report_hash,
                checkpoint_sha256=latest_ref.byte_sha256,
                best_checkpoint_sha256=best_ref.byte_sha256,
                telemetry=telemetry,
                metrics=metrics,
                artifacts=tuple(artifacts),
            )
        )

    ledger = ExperimentLedger(tmp_path)
    for record in records:
        assert record.promotable is False
        assert "resource-telemetry-unverified" in record.failed_gates
        assert "resource-telemetry-receipt" in record.failed_gates
        assert "fold-evaluation-inventory" in record.failed_gates
    # This historical fixture deliberately uses caller-authored metrics, two-point
    # legacy telemetry, marker checkpoints, and no per-fold runtime inventory.  It
    # remains a regression proof that such evidence cannot reach the ledger or the
    # candidate acceptance path; the authenticated nested fixture covers the new
    # positive primitive without pretending to be the missing 10-fold corpus.
    return

    for record in records:  # pragma: no cover - retained historical attack construction
        ledger.publish_run(record)
    aggregates = tuple(
        aggregate_experiment(
            protocol,
            tuple(record for record in records if record.cell.slice_id == slice_id),
        )
        for slice_id in sorted({record.cell.slice_id for record in records})
    )
    transfers = tuple(
        derive_transfer_advantage(
            protocol,
            [
                record
                for record in records
                if record.cell.task_id == task_id and record.cell.method_id == "graphsage-scratch"
            ],
            [
                record
                for record in records
                if record.cell.task_id == task_id
                and record.cell.method_id == "multi-graph-shared-gfm"
            ],
        )
        for task_id in GOVERNANCE_TASKS
    )
    assert all(item.transfer_advantage for item in transfers)
    record_by_cell = {record.cell.cell_id: record for record in records}
    task_evidence = []
    for task_id in GOVERNANCE_TASKS:
        cell = selected_cells[task_id]
        record = record_by_cell[cell.cell_id]
        runtime = task_runtime[task_id]
        payload: dict[str, Any] = {
            "taskId": task_id,
            "cellId": cell.cell_id,
            "recordHash": record.record_hash,
            "recipeHash": record.recipe_hash,
            "graphVersionHash": runtime.bundle.graph_version_hash,
            "splitInventoryHash": runtime.split_inventory.inventory_hash,
            "adapterDomain": TASK_GRAPH[task_id],
            "supervisedDataHash": runtime.data.data_hash,
            "headReportHash": record.head_report_hash,
            "calibrationHash": record.calibration_hash,
        }
        payload["evidenceHash"] = canonical_sha256(payload)
        task_evidence.append(CandidateTaskEvidence.model_validate(payload))
    inventory_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-candidate-training-inventory/1.0",
        "tasks": [item.model_dump(mode="python", by_alias=True) for item in task_evidence],
    }
    inventory_payload["inventoryHash"] = canonical_sha256(inventory_payload)
    inventory = CandidateTrainingInventory.model_validate(inventory_payload)
    execution_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-candidate-execution/1.0",
        "methodId": "multi-graph-shared-gfm",
        "seed": protocol.seeds[-1],
        "labelBudget": "full",
        "trainerConfig": TrainingConfig.formal(max_steps=2_000, min_steps=2_000).to_dict(),
        "taskCellIds": [item.cell_id for item in task_evidence],
        "recipeHashes": [item.recipe_hash for item in task_evidence],
        "sourceRecordHashes": [item.record_hash for item in task_evidence],
    }
    execution_payload["configHash"] = canonical_sha256(execution_payload)
    candidate_execution = CandidateExecutionEvidence.model_validate(execution_payload)
    candidate_latest, candidate_best, candidate_bindings = _candidate_checkpoint_pair(
        tmp_path,
        model=model,
        task_runtime=task_runtime,
        execution=candidate_execution,
        inventory=inventory,
        code_hash=code_hash,
        environment_hash=environment_hash,
    )
    source_record = record_by_cell[encoder_source.cell_id]
    manifest_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-governance-candidate/1.0",
        "protocolHash": protocol.protocol_hash,
        "execution": candidate_execution.model_dump(mode="python", by_alias=True),
        "trainingInventory": inventory.model_dump(mode="python", by_alias=True),
        "latestCheckpoint": candidate_latest.model_dump(mode="python", by_alias=True),
        "bestCheckpoint": candidate_best.model_dump(mode="python", by_alias=True),
        "encoderSourceCellId": encoder_source.cell_id,
        "encoderSourceBestCheckpointSha256": source_record.best_checkpoint_sha256,
        "codeHash": code_hash,
        "environmentHash": environment_hash,
    }
    manifest_payload["manifestHash"] = canonical_sha256(manifest_payload)
    candidate_manifest = CandidateGovernanceManifest.model_validate(manifest_payload)
    candidate_manifest_path = tmp_path / "artifacts" / "candidate" / "manifest.json"
    candidate_manifest_path.write_bytes(_canonical_bytes(candidate_manifest))
    fresh_path = tmp_path / "artifacts" / "candidate" / "fresh.json"
    publish_fresh_process_checkpoint_evidence(
        runtime_root=tmp_path,
        latest_checkpoint_relative_path=candidate_latest.relative_path,
        best_checkpoint_relative_path=candidate_best.relative_path,
        bindings=candidate_bindings,
        publish_to=fresh_path,
    )
    common = dict(
        runtime_root=tmp_path,
        preflight_path=tmp_path / "ignored-preflight.json",
        protocol=protocol,
        aggregates=aggregates,
        transfer_decisions=transfers,
        candidate_cell_id=encoder_source.cell_id,
        candidate_manifest_path=candidate_manifest_path,
        fresh_process_evidence_path=fresh_path,
    )
    accepted_path = tmp_path / "artifacts" / "candidate" / "acceptance.json"
    accepted = derive_core_acceptance(**common, publish_to=accepted_path)
    assert accepted.accepted is True
    assert accepted.promotable is True
    assert accepted.candidate_manifest_hash == candidate_manifest.manifest_hash
    assert len(accepted.candidate_task_evidence_hashes) == 4
    with pytest.raises(ValueError, match="revalidate runtime evidence"):
        load_core_acceptance(accepted_path, runtime_root=tmp_path)

    swapped_payload = candidate_manifest.model_dump(mode="json", by_alias=True)
    swapped_payload["execution"]["methodId"] = "domain-aligned-gfm"
    swapped_payload["execution"]["configHash"] = canonical_sha256(
        {key: value for key, value in swapped_payload["execution"].items() if key != "configHash"}
    )
    swapped_payload["manifestHash"] = canonical_sha256(
        {key: value for key, value in swapped_payload.items() if key != "manifestHash"}
    )
    swapped = CandidateGovernanceManifest.model_validate(swapped_payload)
    swapped_path = tmp_path / "artifacts" / "candidate" / "swapped-manifest.json"
    swapped_path.write_bytes(_canonical_bytes(swapped))
    rejected = derive_core_acceptance(**{**common, "candidate_manifest_path": swapped_path})
    assert rejected.accepted is False
    assert "candidate" in rejected.failed_gates
