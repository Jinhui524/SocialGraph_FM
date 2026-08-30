from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

import socialgraph_gfm.core.acceptance as acceptance_module
from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.acceptance import (
    _strict_core_gfm_from_checkpoint,
    derive_core_acceptance,
    load_core_acceptance,
)
from socialgraph_gfm.core.checkpoint import CheckpointBindings, publish_checkpoint
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core.experiments import (
    ExperimentArtifactRef,
    ExperimentProtocol,
    ExperimentRunRecord,
)
from socialgraph_gfm.core.formal_preflight import (
    FORMAL_CORPUS_REQUIREMENTS,
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
    recall_at_fixed_fpr,
    select_binary_threshold,
)
from socialgraph_gfm.core.supervised import HeadTrainingConfig
from socialgraph_gfm.core.telemetry_receipt import (
    OperatorTelemetryCapability,
    TrustedTelemetryPolicy,
)
from socialgraph_gfm.core.trainer import _fit_state_payload, _model_state_hash
from socialgraph_gfm.tensor_digest import canonical_tensor_digest


def _canonical_bytes(value) -> bytes:
    return (canonical_json(value) + "\n").encode()


def _telemetry_policy() -> TrustedTelemetryPolicy:
    capability = OperatorTelemetryCapability.from_secret(
        key_id="acceptance-test-key",
        secret=bytes(range(32)),
    )
    return TrustedTelemetryPolicy(capability)


def test_candidate_checkpoint_requires_complete_core_gfm_and_adapter_inventory() -> None:
    with pytest.raises(ValueError, match="complete CoreGFM"):
        _strict_core_gfm_from_checkpoint(
            {
                "trainer": {
                    "model": {
                        "node_head.weight": torch.zeros((2, 128)),
                        "node_head.bias": torch.zeros(2),
                    },
                    "adapters": {},
                    "adapterSchemas": {},
                }
            }
        )


def test_formal_raw_gfm_checkpoint_rejects_a_marker_model() -> None:
    with pytest.raises(ValueError, match="formal raw GFM checkpoint"):
        acceptance_module._strict_raw_gfm_checkpoint(
            {
                "trainer": {
                    "model": {"cellMarker": torch.tensor([1.0])},
                    "adapterSchemas": {},
                    "adapters": {},
                }
            },
            expected_domains=("tolokers",),
        )


def _write(root: Path, relative: str, payload: bytes) -> tuple[str, int]:
    path = root / Path(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _self_hashed(payload: dict, field: str) -> dict:
    payload[field] = canonical_sha256(payload)
    return payload


def _ready_preflight(
    tmp_path: Path,
    *,
    tolokers_manifest_hash: str,
    split_hash: str,
) -> FormalPreflightEvidence:
    base = run_formal_preflight(tmp_path).model_dump(mode="json", by_alias=True)
    observations = []
    for requirement in FORMAL_CORPUS_REQUIREMENTS:
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
        is_tolokers = requirement.requirement_id == "tolokers"
        observations.append(
            {
                "requirementId": requirement.requirement_id,
                "status": "ready",
                "reasonCode": "validated-formal-dataset",
                "manifestHash": (tolokers_manifest_hash if is_tolokers else "6" * 64),
                "graphVersionHash": "7" * 64,
                "splitManifestHash": split_hash if is_tolokers else "8" * 64,
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


def _fixture_artifacts(tmp_path: Path):
    split = {
        "strategy": "official",
        "assignments": [
            {"entityId": "node:a", "role": "train"},
            {"entityId": "node:b", "role": "validation"},
            {"entityId": "node:c", "role": "test"},
        ],
    }
    split_hash = canonical_sha256(split)
    split_ids = [f"official-{index:02d}" for index in range(10)]
    split_hashes = [split_hash, *(f"{index + 1:064x}" for index in range(9))]
    manifest = {
        "schemaVersion": "socialgraph-fm.core-experiment-dataset/1.2",
        "requirementId": "tolokers",
        "recipeId": "tolokers",
        "recipeVersion": "1.0.0",
        "recipeSha256": "1" * 64,
        "graphId": "tolokers",
        "phaseEligibility": "formal",
        "usageScope": "public-serving-eligible",
        "splitPolicy": "official",
        "experimentSplitPolicy": "official",
        "materializerId": "test.tolokers",
        "materializerVersion": "1.0",
        "materializerCodeSha256": "2" * 64,
        "materializationProtocolHash": "3" * 64,
        "manifestRelativePath": "experiment-corpus/tolokers/dataset-manifest.json",
        "bundleRelativePath": "experiment-corpus/tolokers/bundle.json",
        "bundleSha256": "4" * 64,
        "labelsRelativePath": "experiment-corpus/tolokers/labels.json",
        "labelsSha256": "5" * 64,
        "labelNames": ["banned"],
        "splitInventoryRelativePath": "experiment-corpus/tolokers/splits.json",
        "splitInventorySha256": "6" * 64,
        "splitCount": 10,
        "splitIds": split_ids,
        "splitManifestHashes": split_hashes,
        "graphVersionHash": "7" * 64,
        "sourceSha256": "8" * 64,
        "splitManifestHash": split_hash,
    }
    _self_hashed(manifest, "manifestHash")
    labels = {
        "schemaVersion": "socialgraph-fm.core-experiment-labels/1.0",
        "requirementId": "tolokers",
        "targets": [
            {
                "name": "banned",
                "values": [
                    {"entityId": "node:a", "value": 0},
                    {"entityId": "node:b", "value": 1},
                ],
            }
        ],
    }
    labels["labelsHash"] = canonical_sha256(labels["targets"])
    adapter = {
        "schemaVersion": "socialgraph-fm.core-adapter-schema/1.1",
        "sourceGraphVersionHash": "7" * 64,
        "fitRowIdsHash": "9" * 64,
        "fitRowCount": 2,
        "visibleTopologyHash": "a" * 64,
        "visibleTopologyEdgeCount": 1,
        "fields": [{"kind": "numeric", "name": "degree", "mean": 1.0, "scale": 1.0}],
    }
    _self_hashed(adapter, "adapterSchemaHash")

    training_data_hash = "f" * 64
    config_hash = "b" * 64
    code_hash = "c" * 64
    environment_hash = "d" * 64
    model_state = {
        "node_head.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "node_head.bias": torch.tensor([-0.5, 0.5]),
    }
    head_state_hash = canonical_sha256(
        {
            name.removeprefix("node_head."): canonical_tensor_digest(value)
            for name, value in sorted(model_state.items())
        }
    )
    head_config = HeadTrainingConfig.formal(max_steps=2)
    head = {
        "schemaVersion": "socialgraph-fm.core-head-training-report/1.0",
        "taskKind": "node-binary",
        "headName": "node_head",
        "graphVersionHash": "7" * 64,
        "modelIdentityHash": "1" * 64,
        "modelIdentityScope": "core-frozen-encoder",
        "encodingArtifactHash": "2" * 64,
        "adapterSchemaHash": adapter["adapterSchemaHash"],
        "adapterStateHash": "3" * 64,
        "topologyHash": "4" * 64,
        "dataHash": training_data_hash,
        "trainPartitionHash": "5" * 64,
        "validationPartitionHash": "6" * 64,
        "numNodes": 3,
        "splitEvidenceScope": "authoritative",
        "splitEvidenceHash": "7" * 64,
        "trainingConfig": head_config.model_dump(mode="python", by_alias=True),
        "configHash": head_config.config_hash,
        "encodedTensorHash": "8" * 64,
        "history": [{"step": 1, "trainLoss": 0.8, "validationLoss": 0.7, "validationMetric": 0.7}],
        "bestStep": 1,
        "bestMetric": 0.7,
        "headStateHash": head_state_hash,
        "promotionEligible": True,
    }
    _self_hashed(head, "reportHash")
    calibration_artifact = {
        "schemaVersion": "socialgraph-fm.core-score-calibration/2.0",
        "calibrationVersion": "test-calibration",
        "method": "sigmoid",
        "temperature": 1.0,
        "bias": 0.0,
        "protocolHash": "9" * 64,
    }
    _self_hashed(calibration_artifact, "artifactHash")
    calibration = {
        "schemaVersion": "socialgraph-fm.core-calibration-fit/1.0",
        "calibration": calibration_artifact,
        "validationScoreBatchHash": "a" * 64,
        "headTrainingReportHash": head["reportHash"],
        "validationPartitionHash": head["validationPartitionHash"],
        "headStateHash": head_state_hash,
        "scoreSemanticsHash": "b" * 64,
        "validationLogitsHash": "c" * 64,
        "validationTargetsHash": "d" * 64,
        "beforeNll": 0.7,
        "afterNll": 0.7,
        "beforeEce": 0.1,
        "afterEce": 0.1,
        "beforeBrier": 0.2,
        "afterBrier": 0.2,
        "promotionEligible": True,
    }
    _self_hashed(calibration, "reportHash")

    artifact_payloads = {
        "dataset-manifest": _canonical_bytes(manifest),
        "split-manifest": _canonical_bytes(split),
        "labels": _canonical_bytes(labels),
        "adapter-schema": _canonical_bytes(adapter),
        "head-report": _canonical_bytes(head),
        "calibration-report": _canonical_bytes(calibration),
        "configuration": b"formal-config\n",
        "training-data": b"training-data\n",
        "code": b"code-snapshot\n",
        "environment": b"environment-lock\n",
        "structure-cache": b"structure-cache\n",
    }
    semantics = {
        "dataset-manifest": manifest["manifestHash"],
        "split-manifest": split_hash,
        "labels": labels["labelsHash"],
        "adapter-schema": adapter["adapterSchemaHash"],
        "head-report": head["reportHash"],
        "calibration-report": calibration["reportHash"],
        "configuration": config_hash,
        "training-data": training_data_hash,
        "code": code_hash,
        "environment": environment_hash,
        "structure-cache": "e" * 64,
    }
    refs = []
    for role, payload in artifact_payloads.items():
        relative = f"artifacts/common/{role}.json"
        byte_hash, size = _write(tmp_path, relative, payload)
        refs.append(
            ExperimentArtifactRef(
                role=role,
                relativePath=relative,
                byteSha256=byte_hash,
                semanticHash=semantics[role],
                sizeBytes=size,
            )
        )

    bindings = CheckpointBindings(
        config_hash=config_hash,
        data_hash=training_data_hash,
        code_hash=code_hash,
        environment_hash=environment_hash,
    )
    best_relative = "artifacts/common/model.best.pt"
    latest_relative = "artifacts/common/model.latest.pt"
    best_path = tmp_path / best_relative
    training_config = TrainingConfig.formal(max_steps=2_000, min_steps=2_000)
    model_state_hash = _model_state_hash(model_state)
    fit_context = {
        "validation_protocol_hash": "1" * 64,
        "validation_data_hash": "2" * 64,
        "validation_partition_hash": "3" * 64,
        "validation_callback_hash": "4" * 64,
    }
    best_fit_state = _fit_state_payload(
        best_step=2_000,
        best_metric=0.7,
        best_model_state_hash=model_state_hash,
        stale_validations=0,
        last_validation_step=2_000,
        last_validation_metric=0.7,
        last_model_state_hash=model_state_hash,
        checkpoint_model_state_hash=model_state_hash,
        best_checkpoint_name=None,
        best_checkpoint_sha256=None,
        **fit_context,
    )
    publish_checkpoint(
        best_path,
        trainer_state={
            "optimizerStep": 2_000,
            "model": model_state,
            "config": training_config.to_dict(),
            "fitState": best_fit_state,
        },
        bindings=bindings,
        status="validated",
        promotable=False,
    )
    best_bytes = best_path.read_bytes()
    best_hash = hashlib.sha256(best_bytes).hexdigest()
    latest_path = tmp_path / latest_relative
    latest_fit_state = _fit_state_payload(
        best_step=2_000,
        best_metric=0.7,
        best_model_state_hash=model_state_hash,
        stale_validations=0,
        last_validation_step=2_000,
        last_validation_metric=0.7,
        last_model_state_hash=model_state_hash,
        checkpoint_model_state_hash=model_state_hash,
        best_checkpoint_name=best_path.name,
        best_checkpoint_sha256=best_hash,
        **fit_context,
    )
    publish_checkpoint(
        latest_path,
        trainer_state={
            "optimizerStep": 2_000,
            "model": model_state,
            "config": training_config.to_dict(),
            "fitState": latest_fit_state,
        },
        bindings=bindings,
        status="validated",
        promotable=True,
    )
    latest_bytes = latest_path.read_bytes()
    latest_hash = hashlib.sha256(latest_bytes).hexdigest()
    refs.extend(
        (
            ExperimentArtifactRef(
                role="best-checkpoint",
                relativePath=best_relative,
                byteSha256=best_hash,
                semanticHash=best_hash,
                sizeBytes=len(best_bytes),
            ),
            ExperimentArtifactRef(
                role="latest-checkpoint",
                relativePath=latest_relative,
                byteSha256=latest_hash,
                semanticHash=latest_hash,
                sizeBytes=len(latest_bytes),
            ),
        )
    )
    entity_ids = ["node:a", "node:b", "node:c", "node:d"]
    target_values = [0.0, 1.0, 0.0, 1.0]
    target = {
        "schemaVersion": "socialgraph-fm.core-metric-targets/1.0",
        "taskId": "tolokers.risk",
        "evaluationKind": "binary",
        "entityIds": entity_ids,
        "values": target_values,
    }
    _self_hashed(target, "targetHash")
    threshold = select_binary_threshold(
        [0.1, 0.9, 0.2, 0.8],
        target_values,
        validation_partition_hash="4" * 64,
        objective="macro-f1",
    )
    threshold_evidence = {
        "schemaVersion": "socialgraph-fm.core-metric-threshold-selection/1.0",
        "threshold": threshold.model_dump(mode="python", by_alias=True),
        "validationScores": [0.1, 0.9, 0.2, 0.8],
        "validationTargets": target_values,
    }
    _self_hashed(threshold_evidence, "evidenceHash")
    shared_metric_refs = []
    for role, relative, payload, semantic in (
        (
            "targets",
            "artifacts/common/targets.json",
            _canonical_bytes(target),
            target["targetHash"],
        ),
        (
            "threshold",
            "artifacts/common/threshold.json",
            _canonical_bytes(threshold_evidence),
            threshold.threshold_hash,
        ),
    ):
        byte_hash, size = _write(tmp_path, relative, payload)
        shared_metric_refs.append(
            ExperimentArtifactRef(
                role=role,
                relativePath=relative,
                byteSha256=byte_hash,
                semanticHash=semantic,
                sizeBytes=size,
            )
        )

    metric_variants = {}
    for name, scores in {
        "good": [0.1, 0.9, 0.2, 0.8],
        "bad": [0.9, 0.1, 0.8, 0.2],
    }.items():
        predictions = {
            "schemaVersion": "socialgraph-fm.core-metric-predictions/1.0",
            "taskId": "tolokers.risk",
            "evaluationKind": "binary",
            "entityIds": entity_ids,
            "scores": scores,
            "probabilities": scores,
            "filteredNegativeScores": [],
        }
        _self_hashed(predictions, "predictionHash")
        prediction_payload = _canonical_bytes(predictions)
        prediction_relative = f"artifacts/common/predictions-{name}.json"
        prediction_byte_hash, prediction_size = _write(
            tmp_path, prediction_relative, prediction_payload
        )
        prediction_ref = ExperimentArtifactRef(
            role="predictions",
            relativePath=prediction_relative,
            byteSha256=prediction_byte_hash,
            semanticHash=predictions["predictionHash"],
            sizeBytes=prediction_size,
        )
        point = binary_metrics_at_threshold(scores, target_values, threshold=threshold)
        metrics = TaskMetricSet.create(
            task_id="tolokers.risk",
            metrics={
                "auprc": binary_auprc(scores, target_values),
                "auroc": binary_auroc(scores, target_values),
                "brier": binary_brier(scores, target_values),
                "ece": binary_ece(scores, target_values),
                "macroF1": point["macroF1"],
                "recallAtFpr": recall_at_fixed_fpr(scores, target_values, max_fpr=0.10),
            },
            prediction_hash=predictions["predictionHash"],
            target_hash=target["targetHash"],
            threshold_hash=threshold.threshold_hash,
        )
        metric_variants[name] = {
            "metrics": metrics,
            "refs": tuple(
                sorted(
                    (*refs, *shared_metric_refs, prediction_ref),
                    key=lambda item: item.role,
                )
            ),
        }
    return {
        "metric_variants": metric_variants,
        "manifest_hash": manifest["manifestHash"],
        "split_hash": split_hash,
        "adapter_hash": adapter["adapterSchemaHash"],
        "labels_hash": labels["labelsHash"],
        "head_hash": head["reportHash"],
        "calibration_hash": calibration["reportHash"],
        "config_hash": config_hash,
        "training_data_hash": training_data_hash,
        "code_hash": code_hash,
        "environment_hash": environment_hash,
        "latest_hash": latest_hash,
        "best_hash": best_hash,
        "latest_relative": latest_relative,
        "best_relative": best_relative,
        "bindings": bindings,
    }


def _record(cell, *, good: bool, preflight_hash: str, fixture) -> ExperimentRunRecord:
    variant = fixture["metric_variants"]["good" if good else "bad"]
    return ExperimentRunRecord.create(
        cell=cell,
        phase="formal",
        preflight_evidence_hash=preflight_hash,
        dataset_manifest_hash=fixture["manifest_hash"],
        split_manifest_hash=fixture["split_hash"],
        config_hash=fixture["config_hash"],
        training_data_hash=fixture["training_data_hash"],
        code_hash=fixture["code_hash"],
        environment_hash=fixture["environment_hash"],
        structure_cache_hash="e" * 64,
        adapter_schema_hash=fixture["adapter_hash"],
        label_artifact_hash=fixture["labels_hash"],
        head_report_hash=fixture["head_hash"],
        calibration_hash=fixture["calibration_hash"],
        checkpoint_sha256=fixture["latest_hash"],
        best_checkpoint_sha256=fixture["best_hash"],
        optimizer_steps=2_000,
        elapsed_seconds=10.0,
        data_wait_seconds=1.0,
        peak_cuda_bytes=1024,
        metrics=variant["metrics"],
        artifacts=variant["refs"],
    )


def test_current_missing_corpus_yields_hash_bound_rejected_report(tmp_path: Path) -> None:
    preflight_path = tmp_path / "preflight.json"
    run_formal_preflight(tmp_path, publish_to=preflight_path)
    report_path = tmp_path / "acceptance.json"
    report = derive_core_acceptance(
        runtime_root=tmp_path,
        preflight_path=preflight_path,
        protocol=ExperimentProtocol.fixed(),
        aggregates=(),
        transfer_decisions=(),
        candidate_cell_id=None,
        fresh_process_evidence_path=None,
        telemetry_policy=_telemetry_policy(),
        publish_to=report_path,
    )
    assert report.status == "rejected"
    assert report.accepted is False
    assert report.promotable is False
    assert "formal-preflight" in report.failed_gates
    assert "matrix-completeness" in report.failed_gates
    assert load_core_acceptance(report_path, runtime_root=tmp_path) == report


def test_publication_rederives_inputs_and_refuses_a_stale_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight_path = tmp_path / "preflight.json"
    run_formal_preflight(tmp_path, publish_to=preflight_path)
    arguments = dict(
        runtime_root=tmp_path,
        preflight_path=preflight_path,
        protocol=ExperimentProtocol.fixed(),
        aggregates=(),
        transfer_decisions=(),
        candidate_cell_id=None,
        fresh_process_evidence_path=None,
        telemetry_policy=_telemetry_policy(),
    )
    observed = derive_core_acceptance(**arguments)
    changed_payload = observed.model_dump(mode="json", by_alias=True)
    changed_payload["preflightEvidenceHash"] = "f" * 64
    changed_payload["acceptanceHash"] = canonical_sha256(
        {key: value for key, value in changed_payload.items() if key != "acceptanceHash"}
    )
    changed = acceptance_module.CoreAcceptance.model_validate(changed_payload)
    derivations = iter((observed, changed))
    monkeypatch.setattr(
        acceptance_module,
        "_derive_core_acceptance_once",
        lambda **_arguments: next(derivations),
    )
    seam_calls: list[Path] = []
    monkeypatch.setattr(
        acceptance_module,
        "_ACCEPTANCE_PUBLICATION_SEAM",
        seam_calls.append,
    )
    report_path = tmp_path / "stale-acceptance.json"
    with pytest.raises(RuntimeError, match="inputs changed during publication"):
        derive_core_acceptance(**arguments, publish_to=report_path)
    assert seam_calls == [report_path]
    assert load_core_acceptance(report_path, runtime_root=tmp_path) == observed
