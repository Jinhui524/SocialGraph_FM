from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.acceptance import (
    _calibrated_probabilities,
    _verify_cell_gfm_fold_evaluations,
)
from socialgraph_gfm.core.adapters import BundleInputAdapter, derive_training_selection
from socialgraph_gfm.core.calibration import (
    BinaryScoreSemantics,
    CalibrationProtocol,
    derive_validation_scores,
    fit_score_calibration_report,
)
from socialgraph_gfm.core.checkpoint import (
    CheckpointBindings,
    load_checkpoint,
    publish_checkpoint,
)
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.experiments import (
    ExperimentExecutionConfigEvidence,
    ExperimentProtocol,
    ExperimentRecipeEvidence,
    ThresholdSelectionEvidence,
    build_experiment_matrix,
)
from socialgraph_gfm.core.fold_evaluation import (
    infer_core_gfm_fold,
    prepare_authoritative_fold,
)
from socialgraph_gfm.core.fold_inventory import (
    CellFoldEvaluationInventory,
    FoldEvaluationBinding,
    FoldRuntimeArtifactRef,
)
from socialgraph_gfm.core.fold_recovery import verify_fold_recovery_state
from socialgraph_gfm.core.formal_preflight import (
    ExperimentDatasetManifest,
    ExperimentLabels,
    ExperimentSplitInventory,
)
from socialgraph_gfm.core.metrics import select_binary_threshold
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.resource_telemetry import (
    ResourceTelemetryRecorder,
    verify_resource_telemetry,
)
from socialgraph_gfm.core.telemetry_receipt import (
    OperatorTelemetryCapability,
    TelemetryReceiptSigner,
    TrustedTelemetryPolicy,
)
from socialgraph_gfm.core.supervised import (
    HeadTrainingConfig,
    SupervisedPartition,
    SupervisedTrainValidation,
    encode_supervised_graph,
    fit_supervised_head,
)
from socialgraph_gfm.core.trainer import CoreTrainer, TrainingGraph, _model_state_hash
from socialgraph_gfm.core.training_data import PreparedGraph

from test_core_acceptance_real import _build_task_runtime
from test_core_experiments import _record


_NESTED_CAPABILITY = OperatorTelemetryCapability.from_secret(
    key_id="nested-test-runner",
    secret=bytes(range(32)),
)
_NESTED_SIGNER = TelemetryReceiptSigner(_NESTED_CAPABILITY)


def _canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _fold_ref(
    root: Path,
    *,
    role: str,
    relative: str,
    payload: bytes,
    semantic_hash: str,
) -> FoldRuntimeArtifactRef:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return FoldRuntimeArtifactRef(
        role=role,
        relativePath=relative,
        byteSha256=hashlib.sha256(payload).hexdigest(),
        semanticHash=semantic_hash,
        sizeBytes=len(payload),
    )


def _replace_fold_artifact(
    inventory: CellFoldEvaluationInventory,
    *,
    fold_index: int,
    changed_ref: FoldRuntimeArtifactRef,
    fold_data_hash: str | None = None,
) -> CellFoldEvaluationInventory:
    binding = inventory.folds[fold_index]
    binding_raw = binding.model_dump(mode="python", by_alias=True)
    replaced = False
    artifacts = []
    for artifact in binding_raw["artifacts"]:
        if artifact["role"] == changed_ref.role:
            artifacts.append(changed_ref.model_dump(mode="python", by_alias=True))
            replaced = True
        else:
            artifacts.append(artifact)
    assert replaced
    binding_raw["artifacts"] = artifacts
    if fold_data_hash is not None:
        binding_raw["foldDataHash"] = fold_data_hash
    binding_raw["bindingHash"] = canonical_sha256(
        {key: value for key, value in binding_raw.items() if key != "bindingHash"}
    )
    changed_binding = FoldEvaluationBinding.model_validate(binding_raw)
    inventory_raw = inventory.model_dump(mode="python", by_alias=True)
    inventory_raw["folds"] = list(inventory_raw["folds"])
    inventory_raw["folds"][fold_index] = changed_binding.model_dump(mode="python", by_alias=True)
    inventory_raw["inventoryHash"] = canonical_sha256(
        {key: value for key, value in inventory_raw.items() if key != "inventoryHash"}
    )
    return CellFoldEvaluationInventory.model_validate(inventory_raw)


def _partition(
    prepared,
    labels: ExperimentLabels,
    *,
    role: str,
) -> SupervisedPartition:
    target = next(item for item in labels.targets if item.name == "banned")
    target_by_id = {item.entity_id: item.value for item in target.values}
    node_by_id = {item.id: item.index for item in prepared.bundle.nodes}
    entity_ids = tuple(
        sorted(
            item.entity_id
            for item in prepared.bundle.split_manifest.assignments
            if item.role == role
        )
    )
    return SupervisedPartition(
        entityIds=entity_ids,
        nodeIndices=tuple(node_by_id[item] for item in entity_ids),
        edgePairs=(),
        targets=tuple(int(target_by_id[item]) for item in entity_ids),
    )


def _simple_fit_state(step: int, model_hash: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "optimizerStep": step,
        "checkpointModelStateHash": model_hash,
    }
    payload["stateHash"] = canonical_sha256(payload)
    return payload


def _drifted_validation_context(fit_state: dict[str, Any]) -> dict[str, Any]:
    changed = dict(fit_state)
    changed.update(
        {
            "validationProtocolHash": canonical_sha256({"drift": "protocol"}),
            "validationDataHash": canonical_sha256({"drift": "data"}),
            "validationPartitionHash": canonical_sha256({"drift": "partition"}),
            "validationCallbackHash": canonical_sha256({"drift": "callback"}),
        }
    )
    context_hash = canonical_sha256(
        {
            "protocolHash": changed["validationProtocolHash"],
            "dataHash": changed["validationDataHash"],
            "partitionHash": changed["validationPartitionHash"],
            "callbackHash": changed["validationCallbackHash"],
        }
    )
    changed["validationContextHash"] = context_hash
    for prefix in ("best", "last"):
        title = prefix.title()
        changed[f"{prefix}ValidationHash"] = canonical_sha256(
            {
                "optimizerStep": changed[
                    f"{prefix}Step" if prefix == "best" else "lastValidationStep"
                ],
                "validationMetric": changed[
                    f"{prefix}Metric" if prefix == "best" else "lastValidationMetric"
                ],
                "modelStateHash": changed[
                    f"{prefix}ModelStateHash" if prefix == "best" else "lastModelStateHash"
                ],
                "validationContextHash": context_hash,
            }
        )
        assert changed[f"{prefix}ValidationHash"] is not None, title
    changed["stateHash"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "stateHash"}
    )
    return changed


def _checkpoint_pair(
    root: Path,
    *,
    cell,
    fold_id: str,
    fold_data_hash: str,
    execution: ExperimentExecutionConfigEvidence,
    record,
    model: CoreGFM,
    adapter: BundleInputAdapter,
    bundle,
    validation_metric: float,
    validation_protocol_hash: str,
    validation_data_hash: str,
    validation_partition_hash: str,
    validation_callback_hash: str,
) -> tuple[
    FoldRuntimeArtifactRef,
    FoldRuntimeArtifactRef,
    dict[str, Any],
    str,
    str,
]:
    domain = f"{cell.target_graph_id}::{fold_id}"
    config = TrainingConfig(**execution.training_config)
    selection = derive_training_selection(bundle)
    node_index = {item.id: item.index for item in bundle.nodes}
    pairs: list[tuple[int, int]] = []
    for edge_ordinal in selection.visible_edge_indices:
        edge = bundle.edges[edge_ordinal]
        left = node_index[edge.source_id]
        right = node_index[edge.target_id]
        pairs.append((left, right))
        if not bundle.directed:
            pairs.append((right, left))
    edge_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    graph = PreparedGraph.from_edge_index(
        num_nodes=len(bundle.nodes), edge_index=edge_index, directed=bundle.directed
    )
    trainer = CoreTrainer(
        model,
        {domain: TrainingGraph.from_bundle(adapter=adapter, graph=graph)},
        config=config,
        seed=cell.seed,
    )
    trainer.optimizer_step = 2_000
    trainer.scheduler.last_epoch = 2_000
    model_state = model.state_dict()
    model_hash = _model_state_hash(model_state)
    run_id = canonical_sha256({"cellId": cell.cell_id, "foldId": fold_id})[:16]
    prefix = f"artifacts/nested/{cell.cell_id[:12]}/{fold_id}"
    best_relative = f"{prefix}/.model.best.pt.run-{run_id}.step-0000002000.pt"
    latest_relative = f"{prefix}/model.latest.pt"
    trainer.fit_best_step = 2_000
    trainer.fit_best_metric = validation_metric
    trainer.fit_best_model_state_hash = model_hash
    trainer.fit_last_validation_step = 2_000
    trainer.fit_last_validation_metric = validation_metric
    trainer.fit_last_model_state_hash = model_hash
    trainer.fit_validation_protocol_hash = validation_protocol_hash
    trainer.fit_validation_data_hash = validation_data_hash
    trainer.fit_validation_partition_hash = validation_partition_hash
    trainer.fit_validation_callback_hash = validation_callback_hash
    trainer.fit_best_checkpoint_name = None
    trainer.fit_best_checkpoint_sha256 = None
    best_state = trainer.state_dict()
    identity = {"experimentCellId": cell.cell_id, "evaluationFoldId": fold_id}
    bindings = CheckpointBindings(
        config_hash=record.config_hash,
        data_hash=fold_data_hash,
        code_hash=record.code_hash,
        environment_hash=record.environment_hash,
    )
    best_path = root / best_relative
    best_path.parent.mkdir(parents=True, exist_ok=True)
    publish_checkpoint(
        best_path,
        trainer_state={**best_state, **identity},
        bindings=bindings,
        status="validated",
        promotable=False,
    )
    best_bytes = best_path.read_bytes()
    best_hash = hashlib.sha256(best_bytes).hexdigest()
    trainer.fit_best_checkpoint_name = best_path.name
    trainer.fit_best_checkpoint_sha256 = best_hash
    latest_state = trainer.state_dict()
    latest_fit = latest_state["fitState"]
    latest_path = root / latest_relative
    publish_checkpoint(
        latest_path,
        trainer_state={**latest_state, **identity},
        bindings=bindings,
        status="validated",
        promotable=True,
    )
    latest_bytes = latest_path.read_bytes()
    best_recovery = verify_fold_recovery_state(
        {**best_state, **identity},
        bundle=bundle,
        adapter_domain=domain,
        config=config,
        expected_seed=cell.seed,
        expected_cell_id=cell.cell_id,
        expected_fold_id=fold_id,
    )
    latest_recovery = verify_fold_recovery_state(
        {**latest_state, **identity},
        bundle=bundle,
        adapter_domain=domain,
        config=config,
        expected_seed=cell.seed,
        expected_cell_id=cell.cell_id,
        expected_fold_id=fold_id,
    )
    return (
        FoldRuntimeArtifactRef(
            role="latest-checkpoint",
            relativePath=latest_relative,
            byteSha256=hashlib.sha256(latest_bytes).hexdigest(),
            semanticHash=hashlib.sha256(latest_bytes).hexdigest(),
            sizeBytes=len(latest_bytes),
        ),
        FoldRuntimeArtifactRef(
            role="best-checkpoint",
            relativePath=best_relative,
            byteSha256=best_hash,
            semanticHash=best_hash,
            sizeBytes=len(best_bytes),
        ),
        latest_fit,
        canonical_sha256(
            {
                "latest": latest_recovery.composite_state_hash,
                "best": best_recovery.composite_state_hash,
            }
        ),
        canonical_sha256(
            {
                "latest": latest_recovery.recovery_state_hash,
                "best": best_recovery.recovery_state_hash,
            }
        ),
    )


def _two_fold_runtime(tmp_path: Path):
    torch.manual_seed(20260815)
    protocol = ExperimentProtocol.fixed()
    cell = next(
        item
        for item in build_experiment_matrix(protocol)
        if item.task_id == "tolokers.risk"
        and item.method_id == "graphsage-scratch"
        and item.label_budget == "full"
        and item.seed == protocol.seeds[0]
    )
    record = _record(cell, value=0.5)
    source_runtime = _build_task_runtime(
        tmp_path,
        task_id="tolokers.risk",
        model=CoreGFM(node_classes=2),
        ordinal=0,
    )
    split_raw = source_runtime.split_inventory.model_dump(mode="python", by_alias=True)
    split_raw["folds"] = split_raw["folds"][:2]
    split_raw["inventoryHash"] = canonical_sha256(
        {key: value for key, value in split_raw.items() if key != "inventoryHash"}
    )
    split_inventory = ExperimentSplitInventory.model_validate(split_raw)
    labels_ref = next(item for item in source_runtime.shared_refs if item.role == "labels")
    labels = ExperimentLabels.model_validate_json(
        (tmp_path / labels_ref.relative_path).read_bytes()
    )
    manifest_raw = source_runtime.manifest.model_dump(mode="python", by_alias=True)
    manifest_raw["splitCount"] = 2
    manifest_raw["splitIds"] = [item.fold_id for item in split_inventory.folds]
    manifest_raw["splitManifestHashes"] = [
        item.split_manifest_hash for item in split_inventory.folds
    ]
    split_bytes = _canonical_bytes(split_inventory)
    manifest_raw["splitInventorySha256"] = hashlib.sha256(split_bytes).hexdigest()
    manifest_raw["manifestHash"] = canonical_sha256(
        {key: value for key, value in manifest_raw.items() if key != "manifestHash"}
    )
    manifest = ExperimentDatasetManifest.model_validate(manifest_raw)
    manifest_hashes = {
        graph_id: hashlib.sha256(graph_id.encode()).hexdigest()
        for graph_id in (
            *cell.pretraining_graph_ids,
            cell.target_graph_id,
            *((cell.validation_graph_id,) if cell.validation_graph_id else ()),
        )
    }
    recipe = ExperimentRecipeEvidence.create(cell=cell, manifest_hashes=manifest_hashes)
    execution = ExperimentExecutionConfigEvidence.create(
        cell=cell,
        recipe=recipe,
        training_config=TrainingConfig.formal(max_steps=2_000, min_steps=2_000),
    )
    assert execution.config_hash == record.config_hash
    signer = _NESTED_SIGNER
    telemetry_policy = TrustedTelemetryPolicy(_NESTED_CAPABILITY)

    bindings: list[FoldEvaluationBinding] = []
    for ordinal, fold in enumerate(split_inventory.folds):
        torch.manual_seed(100 + ordinal)
        prepared = prepare_authoritative_fold(source_runtime.bundle, fold)
        adapter = BundleInputAdapter(prepared.bundle, mode="training")
        model = CoreGFM(node_classes=2)
        encoded = encode_supervised_graph(model, prepared.bundle, adapter)
        data = SupervisedTrainValidation.create(
            task_kind="node-binary",
            provenance=encoded.provenance,
            train=_partition(prepared, labels, role="train"),
            validation=_partition(prepared, labels, role="validation"),
        )
        verified_head = fit_supervised_head(
            model,
            encoded,
            data,
            config=HeadTrainingConfig.formal(max_steps=2),
        )
        validation_scores = derive_validation_scores(
            model,
            encoded,
            data,
            verified_head,
            semantics=BinaryScoreSemantics.for_task("node-binary"),
        )
        calibration = fit_score_calibration_report(
            validation_scores,
            protocol=CalibrationProtocol.fixed(validation_scores),
        )
        probabilities = _calibrated_probabilities(
            tuple(float(value) for value in validation_scores.logits.tolist()),
            calibration,
        )
        targets = tuple(float(value) for value in validation_scores.targets.tolist())
        threshold = select_binary_threshold(
            probabilities,
            targets,
            validation_partition_hash=data.validation.partition_hash,
            objective="macro-f1",
        )
        threshold_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-metric-threshold-selection/1.0",
            "threshold": threshold.model_dump(mode="python", by_alias=True),
            "validationScores": probabilities,
            "validationTargets": targets,
        }
        threshold_payload["evidenceHash"] = canonical_sha256(threshold_payload)
        selection = ThresholdSelectionEvidence.model_validate(threshold_payload)
        from socialgraph_gfm.core.fold_evaluation import bind_authoritative_fold_test

        bound = bind_authoritative_fold_test(
            prepared,
            labels,
            target_name="banned",
            task_kind="node-binary",
        )
        predictions = infer_core_gfm_fold(model, adapter, bound).record
        fold_data_hash = canonical_sha256(
            {
                "cellId": cell.cell_id,
                "foldId": fold.fold_id,
                "preparedGraphVersionHash": prepared.bundle.graph_version_hash,
                "headDataHash": data.data_hash,
            }
        )
        latest_ref, best_ref, latest_fit, composite_hash, recovery_hash = _checkpoint_pair(
            tmp_path,
            cell=cell,
            fold_id=fold.fold_id,
            fold_data_hash=fold_data_hash,
            execution=execution,
            record=record,
            model=model,
            adapter=adapter,
            bundle=prepared.bundle,
            validation_metric=verified_head.best_metric,
            validation_protocol_hash=verified_head.record.config_hash,
            validation_data_hash=verified_head.record.data_hash,
            validation_partition_hash=verified_head.record.validation_partition_hash,
            validation_callback_hash=canonical_sha256(
                {
                    "callback": "socialgraph-gfm.core.supervised-loss/1.0",
                    "taskKind": data.task_kind,
                    "splitEvidenceHash": verified_head.record.split_evidence_hash,
                }
            ),
        )
        recorder = ResourceTelemetryRecorder(
            cell_id=cell.cell_id,
            run_id=f"fold-{cell.cell_id[:12]}-{ordinal}",
            phase="formal",
            config_hash=record.config_hash,
            data_hash=fold_data_hash,
            code_hash=record.code_hash,
            environment_hash=record.environment_hash,
        )
        model_state = model.state_dict()
        model_hash = _model_state_hash(model_state)
        recorder.record_start(
            model_state=model_state,
            fit_state=_simple_fit_state(0, model_hash),
        )
        for step in range(250, 2_000, 250):
            recorder.record_checkpoint(
                optimizer_step=step,
                model_state=model_state,
                fit_state=_simple_fit_state(step, model_hash),
            )
        telemetry = recorder.finish(
            final_optimizer_step=2_000,
            model_state=model_state,
            fit_state=latest_fit,
            latest_checkpoint_semantic_hash=latest_ref.semantic_hash,
            best_checkpoint_semantic_hash=best_ref.semantic_hash,
        )
        telemetry_record = verify_resource_telemetry(telemetry)
        receipt = signer.issue(
            telemetry=telemetry,
            fold_id=fold.fold_id,
            composite_state_hash=composite_hash,
            recovery_state_hash=recovery_hash,
        )
        prefix = f"artifacts/nested/{cell.cell_id[:12]}/{fold.fold_id}"
        refs = [
            best_ref,
            _fold_ref(
                tmp_path,
                role="calibration-report",
                relative=f"{prefix}/calibration.json",
                payload=_canonical_bytes(calibration),
                semantic_hash=calibration.report_hash,
            ),
            _fold_ref(
                tmp_path,
                role="head-data",
                relative=f"{prefix}/head-data.json",
                payload=_canonical_bytes(data),
                semantic_hash=data.data_hash,
            ),
            _fold_ref(
                tmp_path,
                role="head-report",
                relative=f"{prefix}/head-report.json",
                payload=_canonical_bytes(verified_head.record),
                semantic_hash=verified_head.report_hash,
            ),
            latest_ref,
            _fold_ref(
                tmp_path,
                role="predictions",
                relative=f"{prefix}/predictions.json",
                payload=_canonical_bytes(predictions),
                semantic_hash=predictions.prediction_hash,
            ),
            _fold_ref(
                tmp_path,
                role="resource-telemetry",
                relative=f"{prefix}/telemetry.json",
                payload=_canonical_bytes(telemetry_record),
                semantic_hash=telemetry_record.telemetry_hash,
            ),
            _fold_ref(
                tmp_path,
                role="telemetry-receipt",
                relative=f"{prefix}/telemetry-receipt.json",
                payload=_canonical_bytes(receipt),
                semantic_hash=receipt.receipt_hash,
            ),
            _fold_ref(
                tmp_path,
                role="threshold",
                relative=f"{prefix}/threshold.json",
                payload=_canonical_bytes(selection),
                semantic_hash=selection.evidence_hash,
            ),
        ]
        refs.sort(key=lambda item: item.role)
        binding_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-fold-evaluation-binding/1.0",
            "foldId": fold.fold_id,
            "runtimeKind": "core-gfm",
            "taskKind": "node-binary",
            "splitManifestHash": fold.split_manifest_hash,
            "preparedGraphVersionHash": prepared.bundle.graph_version_hash,
            "foldDataHash": fold_data_hash,
            "adapterDomain": f"{cell.target_graph_id}::{fold.fold_id}",
            "artifacts": [item.model_dump(mode="python", by_alias=True) for item in refs],
        }
        binding_payload["bindingHash"] = canonical_sha256(binding_payload)
        bindings.append(FoldEvaluationBinding.model_validate(binding_payload))
    inventory_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-cell-fold-evaluation/1.0",
        "cellId": cell.cell_id,
        "taskId": cell.task_id,
        "datasetManifestHash": manifest.manifest_hash,
        "splitInventoryHash": split_inventory.inventory_hash,
        "labelsHash": labels.labels_hash,
        "targetName": "banned",
        "foldIds": [item.fold_id for item in bindings],
        "folds": [item.model_dump(mode="python", by_alias=True) for item in bindings],
    }
    inventory_payload["inventoryHash"] = canonical_sha256(inventory_payload)
    inventory = CellFoldEvaluationInventory.model_validate(inventory_payload)
    return (
        record,
        inventory,
        manifest,
        split_inventory,
        labels,
        source_runtime.bundle,
        recipe,
        execution,
        telemetry_policy,
    )


def _verify_runtime(
    root: Path,
    runtime,
    *,
    inventory: CellFoldEvaluationInventory,
):
    (
        record,
        _original_inventory,
        manifest,
        split_inventory,
        labels,
        bundle,
        recipe,
        execution,
        telemetry_policy,
    ) = runtime
    return _verify_cell_gfm_fold_evaluations(
        root=root,
        record=record,
        inventory=inventory,
        manifest=manifest,
        split_inventory=split_inventory,
        labels=labels,
        bundle=bundle,
        recipe=recipe,
        execution_config=execution,
        telemetry_policy=telemetry_policy,
    )


def test_real_two_fold_runtime_is_strict_loaded_and_live_recomputed(tmp_path: Path) -> None:
    (
        record,
        inventory,
        manifest,
        split_inventory,
        labels,
        bundle,
        recipe,
        execution,
        telemetry_policy,
    ) = _two_fold_runtime(tmp_path)

    observed = _verify_cell_gfm_fold_evaluations(
        root=tmp_path,
        record=record,
        inventory=inventory,
        manifest=manifest,
        split_inventory=split_inventory,
        labels=labels,
        bundle=bundle,
        recipe=recipe,
        execution_config=execution,
        telemetry_policy=telemetry_policy,
    )

    assert tuple(item.name for item in observed.metrics.metrics) == record.cell.required_metrics
    assert math.isfinite(observed.validation_metric)


def test_self_hashed_fold_prediction_cannot_replace_live_checkpoint_inference(
    tmp_path: Path,
) -> None:
    (
        record,
        inventory,
        manifest,
        split_inventory,
        labels,
        bundle,
        recipe,
        execution,
        telemetry_policy,
    ) = _two_fold_runtime(tmp_path)
    first = inventory.folds[0]
    prediction_ref = next(item for item in first.artifacts if item.role == "predictions")
    prediction_path = tmp_path / prediction_ref.relative_path
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["scores"][0] = float(prediction["scores"][0]) + 5.0
    prediction["predictionHash"] = canonical_sha256(
        {key: value for key, value in prediction.items() if key != "predictionHash"}
    )
    prediction_bytes = _canonical_bytes(prediction)
    prediction_path.write_bytes(prediction_bytes)
    changed_ref = FoldRuntimeArtifactRef(
        role="predictions",
        relativePath=prediction_ref.relative_path,
        byteSha256=hashlib.sha256(prediction_bytes).hexdigest(),
        semanticHash=prediction["predictionHash"],
        sizeBytes=len(prediction_bytes),
    )
    first_raw = first.model_dump(mode="python", by_alias=True)
    first_raw["artifacts"] = [
        (
            changed_ref.model_dump(mode="python", by_alias=True)
            if item["role"] == "predictions"
            else item
        )
        for item in first_raw["artifacts"]
    ]
    first_raw["bindingHash"] = canonical_sha256(
        {key: value for key, value in first_raw.items() if key != "bindingHash"}
    )
    changed_binding = FoldEvaluationBinding.model_validate(first_raw)
    inventory_raw = inventory.model_dump(mode="python", by_alias=True)
    inventory_raw["folds"] = list(inventory_raw["folds"])
    inventory_raw["folds"][0] = changed_binding.model_dump(mode="python", by_alias=True)
    inventory_raw["inventoryHash"] = canonical_sha256(
        {key: value for key, value in inventory_raw.items() if key != "inventoryHash"}
    )
    changed_inventory = CellFoldEvaluationInventory.model_validate(inventory_raw)

    with pytest.raises(ValueError, match="persisted fold predictions differ"):
        _verify_cell_gfm_fold_evaluations(
            root=tmp_path,
            record=record,
            inventory=changed_inventory,
            manifest=manifest,
            split_inventory=split_inventory,
            labels=labels,
            bundle=bundle,
            recipe=recipe,
            execution_config=execution,
            telemetry_policy=telemetry_policy,
        )


def test_second_fold_rehashed_head_target_cannot_replace_authoritative_labels(
    tmp_path: Path,
) -> None:
    runtime = _two_fold_runtime(tmp_path)
    record, inventory, *_rest = runtime
    second = inventory.folds[1]
    head_ref = next(item for item in second.artifacts if item.role == "head-data")
    head_path = tmp_path / head_ref.relative_path
    head_data = json.loads(head_path.read_text(encoding="utf-8"))
    original = int(head_data["train"]["targets"][0])
    head_data["train"]["targets"][0] = 1 - original
    head_data["dataHash"] = canonical_sha256(
        {key: value for key, value in head_data.items() if key != "dataHash"}
    )
    changed_bytes = _canonical_bytes(head_data)
    head_path.write_bytes(changed_bytes)
    changed_ref = FoldRuntimeArtifactRef(
        role="head-data",
        relativePath=head_ref.relative_path,
        byteSha256=hashlib.sha256(changed_bytes).hexdigest(),
        semanticHash=head_data["dataHash"],
        sizeBytes=len(changed_bytes),
    )
    changed_fold_data_hash = canonical_sha256(
        {
            "cellId": record.cell.cell_id,
            "foldId": second.fold_id,
            "preparedGraphVersionHash": second.prepared_graph_version_hash,
            "headDataHash": head_data["dataHash"],
        }
    )
    changed_inventory = _replace_fold_artifact(
        inventory,
        fold_index=1,
        changed_ref=changed_ref,
        fold_data_hash=changed_fold_data_hash,
    )

    with pytest.raises(ValueError, match="training data identity"):
        _verify_runtime(tmp_path, runtime, inventory=changed_inventory)


def test_second_fold_self_consistent_receipt_hash_cannot_replace_operator_mac(
    tmp_path: Path,
) -> None:
    runtime = _two_fold_runtime(tmp_path)
    _record, inventory, *_rest = runtime
    second = inventory.folds[1]
    receipt_ref = next(item for item in second.artifacts if item.role == "telemetry-receipt")
    receipt_path = tmp_path / receipt_ref.relative_path
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["mac"] = "0" * 64 if receipt["mac"] != "0" * 64 else "1" * 64
    receipt["receiptHash"] = canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receiptHash"}
    )
    changed_bytes = _canonical_bytes(receipt)
    receipt_path.write_bytes(changed_bytes)
    changed_ref = FoldRuntimeArtifactRef(
        role="telemetry-receipt",
        relativePath=receipt_ref.relative_path,
        byteSha256=hashlib.sha256(changed_bytes).hexdigest(),
        semanticHash=receipt["receiptHash"],
        sizeBytes=len(changed_bytes),
    )
    changed_inventory = _replace_fold_artifact(
        inventory,
        fold_index=1,
        changed_ref=changed_ref,
    )

    with pytest.raises(ValueError, match="operator MAC"):
        _verify_runtime(tmp_path, runtime, inventory=changed_inventory)


def test_second_fold_republished_checkpoint_cannot_change_training_seed(
    tmp_path: Path,
) -> None:
    runtime = _two_fold_runtime(tmp_path)
    record, inventory, *_rest = runtime
    second = inventory.folds[1]
    latest_ref = next(item for item in second.artifacts if item.role == "latest-checkpoint")
    checkpoint_bindings = CheckpointBindings(
        config_hash=record.config_hash,
        data_hash=second.fold_data_hash,
        code_hash=record.code_hash,
        environment_hash=record.environment_hash,
    )
    original = load_checkpoint(
        (tmp_path / latest_ref.relative_path).read_bytes(),
        expected_bindings=checkpoint_bindings,
    )
    changed_trainer = dict(original["trainer"])
    changed_trainer["trainingSeed"] = record.cell.seed + 1
    changed_relative = str(
        Path(latest_ref.relative_path).with_name("model.republished.latest.pt")
    ).replace("\\", "/")
    changed_path = tmp_path / changed_relative
    publish_checkpoint(
        changed_path,
        trainer_state=changed_trainer,
        bindings=checkpoint_bindings,
        status="validated",
        promotable=True,
    )
    changed_bytes = changed_path.read_bytes()
    changed_hash = hashlib.sha256(changed_bytes).hexdigest()
    changed_ref = FoldRuntimeArtifactRef(
        role="latest-checkpoint",
        relativePath=changed_relative,
        byteSha256=changed_hash,
        semanticHash=changed_hash,
        sizeBytes=len(changed_bytes),
    )
    changed_inventory = _replace_fold_artifact(
        inventory,
        fold_index=1,
        changed_ref=changed_ref,
    )

    with pytest.raises(ValueError, match="training seed"):
        _verify_runtime(tmp_path, runtime, inventory=changed_inventory)


def test_second_fold_latest_validation_context_cannot_drift_from_best_and_live(
    tmp_path: Path,
) -> None:
    runtime = _two_fold_runtime(tmp_path)
    (
        record,
        inventory,
        _manifest,
        split_inventory,
        _labels,
        bundle,
        _recipe,
        execution,
        _telemetry_policy,
    ) = runtime
    second = inventory.folds[1]
    refs = {item.role: item for item in second.artifacts}
    checkpoint_bindings = CheckpointBindings(
        config_hash=record.config_hash,
        data_hash=second.fold_data_hash,
        code_hash=record.code_hash,
        environment_hash=record.environment_hash,
    )
    latest = load_checkpoint(
        (tmp_path / refs["latest-checkpoint"].relative_path).read_bytes(),
        expected_bindings=checkpoint_bindings,
    )
    best = load_checkpoint(
        (tmp_path / refs["best-checkpoint"].relative_path).read_bytes(),
        expected_bindings=checkpoint_bindings,
    )
    changed_trainer = dict(latest["trainer"])
    changed_trainer["fitState"] = _drifted_validation_context(dict(changed_trainer["fitState"]))
    prefix = str(Path(refs["latest-checkpoint"].relative_path).parent).replace("\\", "/")
    changed_latest_relative = f"{prefix}/model.validation-context-drift.latest.pt"
    changed_latest_path = tmp_path / changed_latest_relative
    publish_checkpoint(
        changed_latest_path,
        trainer_state=changed_trainer,
        bindings=checkpoint_bindings,
        status="validated",
        promotable=True,
    )
    changed_latest_bytes = changed_latest_path.read_bytes()
    changed_latest_hash = hashlib.sha256(changed_latest_bytes).hexdigest()
    changed_latest_ref = FoldRuntimeArtifactRef(
        role="latest-checkpoint",
        relativePath=changed_latest_relative,
        byteSha256=changed_latest_hash,
        semanticHash=changed_latest_hash,
        sizeBytes=len(changed_latest_bytes),
    )

    prepared = prepare_authoritative_fold(bundle, split_inventory.folds[1])
    assert execution.training_config is not None
    config = TrainingConfig(**execution.training_config)
    assert second.adapter_domain is not None
    latest_recovery = verify_fold_recovery_state(
        changed_trainer,
        bundle=prepared.bundle,
        adapter_domain=second.adapter_domain,
        config=config,
        expected_seed=record.cell.seed,
        expected_cell_id=record.cell.cell_id,
        expected_fold_id=second.fold_id,
    )
    best_recovery = verify_fold_recovery_state(
        best["trainer"],
        bundle=prepared.bundle,
        adapter_domain=second.adapter_domain,
        config=config,
        expected_seed=record.cell.seed,
        expected_cell_id=record.cell.cell_id,
        expected_fold_id=second.fold_id,
    )
    pair_composite_hash = canonical_sha256(
        {
            "latest": latest_recovery.composite_state_hash,
            "best": best_recovery.composite_state_hash,
        }
    )
    pair_recovery_hash = canonical_sha256(
        {
            "latest": latest_recovery.recovery_state_hash,
            "best": best_recovery.recovery_state_hash,
        }
    )

    model_state = changed_trainer["model"]
    model_hash = _model_state_hash(model_state)
    recorder = ResourceTelemetryRecorder(
        cell_id=record.cell.cell_id,
        run_id=f"validation-context-drift-{second.fold_id}",
        phase="formal",
        config_hash=record.config_hash,
        data_hash=second.fold_data_hash,
        code_hash=record.code_hash,
        environment_hash=record.environment_hash,
    )
    recorder.record_start(
        model_state=model_state,
        fit_state=_simple_fit_state(0, model_hash),
    )
    for step in range(250, 2_000, 250):
        recorder.record_checkpoint(
            optimizer_step=step,
            model_state=model_state,
            fit_state=_simple_fit_state(step, model_hash),
        )
    verified_telemetry = recorder.finish(
        final_optimizer_step=2_000,
        model_state=model_state,
        fit_state=changed_trainer["fitState"],
        latest_checkpoint_semantic_hash=changed_latest_hash,
        best_checkpoint_semantic_hash=refs["best-checkpoint"].semantic_hash,
    )
    telemetry = verify_resource_telemetry(verified_telemetry)
    receipt = _NESTED_SIGNER.issue(
        telemetry=verified_telemetry,
        fold_id=second.fold_id,
        composite_state_hash=pair_composite_hash,
        recovery_state_hash=pair_recovery_hash,
    )
    changed_telemetry_ref = _fold_ref(
        tmp_path,
        role="resource-telemetry",
        relative=f"{prefix}/telemetry.validation-context-drift.json",
        payload=_canonical_bytes(telemetry),
        semantic_hash=telemetry.telemetry_hash,
    )
    changed_receipt_ref = _fold_ref(
        tmp_path,
        role="telemetry-receipt",
        relative=f"{prefix}/telemetry-receipt.validation-context-drift.json",
        payload=_canonical_bytes(receipt),
        semantic_hash=receipt.receipt_hash,
    )
    changed_inventory = _replace_fold_artifact(
        inventory,
        fold_index=1,
        changed_ref=changed_latest_ref,
    )
    changed_inventory = _replace_fold_artifact(
        changed_inventory,
        fold_index=1,
        changed_ref=changed_telemetry_ref,
    )
    changed_inventory = _replace_fold_artifact(
        changed_inventory,
        fold_index=1,
        changed_ref=changed_receipt_ref,
    )

    with pytest.raises(ValueError, match="fold latest/best checkpoint"):
        _verify_runtime(tmp_path, runtime, inventory=changed_inventory)
