"""Fail-closed core candidate, serving-smoke, and live promotion evidence."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import os
import platform
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal

import torch

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .acceptance import (
    CandidateGovernanceManifest,
    CoreAcceptance,
    derive_core_acceptance,
)
from .artifact_catalog import (
    ArtifactCatalogDocument,
    ArtifactEntry,
    feature_contract_for_bundle,
)
from .bundle import CoreGraphBundle, load_core_graph_bundle_json
from .checkpoint import CheckpointBindings, load_checkpoint
from .experiments import (
    ExperimentAggregate,
    ExperimentLedger,
    ExperimentProtocol,
    TransferDecision,
)
from .formal_preflight import (
    _OwnedFileLease,
    _PublicationParentLease,
    _PublisherLock,
    _path_identity,
    _publish_exact as _hardened_publish_exact,
)
from .inference_contracts import GfmRunRequest
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root
from .serving_control import (
    CapturedServingControl,
    ServingControlDocument,
    ServingControlStore,
    ServingHighWater,
)
from .serving_head import CoreServingHead
from .serving_registry import (
    CalibrationBinding,
    RegistryDocument,
    RegressionConfidenceArtifact,
    ScoreCalibration,
    ServingAdapterBinding,
    ServingCheckpointManifest,
    ServingModel,
    ServingTaskHead,
    _tensor_state_hash,
    _validate_captured_calibrations,
    _validate_captured_checkpoint,
)
from .telemetry_receipt import TrustedTelemetryPolicy


_HASH = r"^[0-9a-f]{64}$"
_TASK_ENTITY_ORDER = (
    ("core.community_resilience_review", "community"),
    ("core.risk_and_trust_review", "node"),
    ("core.risk_and_trust_review", "edge"),
    ("core.collaboration_completion", "node-pair"),
)
_ACCEPTED_TASK_BINDINGS = {
    ("core.community_resilience_review", "community"): (
        "penn94.community-resilience",
        "facebook100.penn94",
    ),
    ("core.risk_and_trust_review", "node"): ("tolokers.risk", "tolokers"),
    ("core.risk_and_trust_review", "edge"): (
        "wiki-rfa.vote-sign",
        "wiki-rfa",
    ),
    ("core.collaboration_completion", "node-pair"): (
        "github.relation-completion",
        "github-musae",
    ),
}
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
_MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


@dataclass(frozen=True)
class AcceptanceDerivationInputs:
    """Complete non-serializable inputs required to rederive formal acceptance."""

    runtime_root: Path
    preflight_path: Path
    protocol: ExperimentProtocol
    aggregates: tuple[ExperimentAggregate, ...]
    transfer_decisions: tuple[TransferDecision, ...]
    candidate_cell_id: str
    candidate_manifest_path: Path
    fresh_process_evidence_path: Path
    telemetry_policy: TrustedTelemetryPolicy

    def derive(self) -> CoreAcceptance:
        return _ACCEPTANCE_DERIVER(
            runtime_root=self.runtime_root,
            preflight_path=self.preflight_path,
            protocol=self.protocol,
            aggregates=self.aggregates,
            transfer_decisions=self.transfer_decisions,
            candidate_cell_id=self.candidate_cell_id,
            candidate_manifest_path=self.candidate_manifest_path,
            fresh_process_evidence_path=self.fresh_process_evidence_path,
            telemetry_policy=self.telemetry_policy,
        )


_ACCEPTANCE_DERIVER = derive_core_acceptance


class CandidateServingDefinition(_StrictModel):
    """Requested serving surface; every task/entity byte identity remains explicit."""

    schema_version: Literal["socialgraph-fm.core-serving-definition/1.0"] = Field(
        alias="schemaVersion"
    )
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    task_heads: tuple[ServingTaskHead, ...] = Field(alias="taskHeads", strict=False)
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)
    definition_hash: str = Field(alias="definitionHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_definition(self):
        expected_tasks = (
            "core.community_resilience_review",
            "core.risk_and_trust_review",
            "core.collaboration_completion",
        )
        if tuple(head.task_id for head in self.task_heads) != expected_tasks:
            raise ValueError("serving definition requires the exact three public tasks")
        observed = tuple(
            (head.task_id, binding.entity_type)
            for head in self.task_heads
            for binding in head.calibrations
        )
        if observed != _TASK_ENTITY_ORDER:
            raise ValueError("serving definition requires the exact task/entity inventory")
        for head in self.task_heads:
            for binding in head.calibrations:
                _source_task, expected_domain = _ACCEPTED_TASK_BINDINGS[
                    (head.task_id, binding.entity_type)
                ]
                if binding.adapter_domain != expected_domain:
                    raise ValueError(
                        "serving definition semantic task/domain binding differs from "
                        "the fixed public task contract"
                    )
        if self.definition_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"definition_hash"})
        ):
            raise ValueError("definitionHash does not bind the serving definition")
        return self

    @property
    def task_binding_inventory_hash(self) -> str:
        return canonical_sha256(
            [
                {
                    "taskId": head.task_id,
                    **binding.model_dump(mode="python", by_alias=True),
                }
                for head in self.task_heads
                for binding in head.calibrations
            ]
        )


class StageArtifact(_StrictModel):
    role: str = Field(min_length=1, max_length=200)
    task_id: str | None = Field(default=None, alias="taskId", min_length=1, max_length=200)
    entity_type: str | None = Field(default=None, alias="entityType", min_length=1, max_length=50)
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    sha256: str = Field(pattern=_HASH)
    size_bytes: int = Field(alias="sizeBytes", gt=0)

    @model_validator(mode="after")
    def validate_artifact(self):
        parsed = PurePosixPath(self.relative_path.replace("\\", "/"))
        if parsed.is_absolute() or ".." in parsed.parts or ":" in self.relative_path:
            raise ValueError("stage artifact path must be safe and relative")
        if (self.task_id is None) != (self.entity_type is None):
            raise ValueError("task-scoped stage artifacts require taskId and entityType together")
        return self


class CandidateStage(_StrictModel):
    """Non-live content-addressed serving candidate."""

    schema_version: Literal["socialgraph-fm.core-candidate-stage/1.0"] = Field(
        alias="schemaVersion"
    )
    acceptance_hash: str = Field(alias="acceptanceHash", pattern=_HASH)
    candidate_manifest_hash: str = Field(alias="candidateManifestHash", pattern=_HASH)
    experiment_summary_hash: str = Field(alias="experimentSummaryHash", pattern=_HASH)
    source_checkpoint_sha256: str = Field(alias="sourceCheckpointSha256", pattern=_HASH)
    serving_checkpoint_sha256: str = Field(alias="servingCheckpointSha256", pattern=_HASH)
    serving_model: ServingModel = Field(alias="servingModel")
    definition_hash: str = Field(alias="definitionHash", pattern=_HASH)
    task_binding_inventory_hash: str = Field(alias="taskBindingInventoryHash", pattern=_HASH)
    artifacts: tuple[StageArtifact, ...] = Field(min_length=6, strict=False)
    artifact_inventory_hash: str = Field(alias="artifactInventoryHash", pattern=_HASH)
    stage_hash: str = Field(alias="stageHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_stage(self):
        identities = tuple((item.role, item.task_id, item.entity_type) for item in self.artifacts)
        if identities != tuple(sorted(set(identities))):
            raise ValueError("stage artifact inventory must be unique and sorted")
        if self.serving_model.state != "accepted":
            raise ValueError("candidate stage model must remain accepted, not live")
        if self.serving_model.checkpoint.sha256 != self.serving_checkpoint_sha256:
            raise ValueError("candidate stage checkpoint differs from serving model")
        if self.artifact_inventory_hash != canonical_sha256(
            [item.model_dump(mode="python", by_alias=True) for item in self.artifacts]
        ):
            raise ValueError("artifactInventoryHash does not bind candidate bytes")
        if self.stage_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"stage_hash"})
        ):
            raise ValueError("stageHash does not bind the candidate stage")
        return self

    def artifact(
        self, role: str, *, task_id: str | None = None, entity_type: str | None = None
    ) -> StageArtifact:
        matches = [
            item
            for item in self.artifacts
            if (item.role, item.task_id, item.entity_type) == (role, task_id, entity_type)
        ]
        if len(matches) != 1:
            raise LookupError("candidate stage artifact is unavailable or ambiguous")
        return matches[0]


@dataclass(frozen=True)
class _CandidateBytes:
    acceptance: bytes
    candidate_manifest: bytes
    source_checkpoint: bytes
    serving_checkpoint: bytes
    serving_manifest: bytes
    serving_model: ServingModel
    task_binding_inventory_hash: str
    confidence: tuple[tuple[str, str, bytes], ...]


def _canonical_bytes(value: BaseModel) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _read_inside(root: Path, path: Path, *, max_bytes: int) -> bytes:
    lexical = reject_link_components(path)
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("promotion input escapes the authorized runtime root") from error
    return read_confined_snapshot(root, relative, max_bytes=max_bytes)


def _read_ref(root: Path, relative_path: str, *, max_bytes: int) -> bytes:
    return read_confined_snapshot(root, relative_path, max_bytes=max_bytes)


def _atomic_immutable(root: Path, path: Path, payload: bytes) -> None:
    """Reuse the held-parent, no-follow, post-flush exact publisher."""

    _hardened_publish_exact(
        root,
        path,
        payload,
        conflict_message="conflicting immutable promotion artifact already exists",
    )


def _serialize_checkpoint(trainer: dict[str, Any], *, bindings: CheckpointBindings) -> bytes:
    payload = {
        "schemaVersion": "socialgraph-fm.core-checkpoint/1.0",
        "bindings": {
            "config_hash": bindings.config_hash,
            "data_hash": bindings.data_hash,
            "code_hash": bindings.code_hash,
            "environment_hash": bindings.environment_hash,
        },
        "status": "accepted",
        "promotable": True,
        "trainer": trainer,
    }
    stream = BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def _accepted_task_binding_inventory_hash(
    definition: CandidateServingDefinition,
    manifest: CandidateGovernanceManifest,
) -> str:
    """Bind each public task/entity to its exact accepted source evidence."""

    evidence_by_task = {item.task_id: item for item in manifest.training_inventory.tasks}
    if len(evidence_by_task) != 4:
        raise ValueError("formal candidate must bind four distinct accepted tasks")
    inventory: list[dict[str, Any]] = []
    for head in definition.task_heads:
        for binding in head.calibrations:
            source_task, expected_domain = _ACCEPTED_TASK_BINDINGS[
                (head.task_id, binding.entity_type)
            ]
            evidence = evidence_by_task.get(source_task)
            if evidence is None:
                raise ValueError("serving semantic task is absent from accepted task evidence")
            if (
                evidence.adapter_domain != expected_domain
                or binding.adapter_domain != evidence.adapter_domain
            ):
                raise ValueError("serving semantic task/domain differs from accepted task evidence")
            expects_calibration = binding.confidence_kind == "binary-calibration"
            if expects_calibration != (evidence.calibration_hash is not None):
                raise ValueError(
                    "serving confidence kind differs from accepted task calibration evidence"
                )
            inventory.append(
                {
                    "publicTaskId": head.task_id,
                    "entityType": binding.entity_type,
                    "servingBinding": binding.model_dump(mode="python", by_alias=True),
                    "acceptedTaskEvidence": evidence.model_dump(mode="python", by_alias=True),
                }
            )
    if (
        tuple((item["publicTaskId"], item["entityType"]) for item in inventory)
        != _TASK_ENTITY_ORDER
    ):
        raise ValueError("accepted task binding inventory is incomplete or out of order")
    return canonical_sha256(inventory)


def _source_confidence_artifacts(
    *,
    root: Path,
    manifest: CandidateGovernanceManifest,
    definition: CandidateServingDefinition,
) -> dict[tuple[str, str], bytes]:
    evidence_by_task = {item.task_id: item for item in manifest.training_inventory.tasks}
    if len(evidence_by_task) != 4:
        raise ValueError("formal candidate must bind four distinct accepted tasks")
    result: dict[tuple[str, str], bytes] = {}
    ledger = ExperimentLedger(root)
    for head in definition.task_heads:
        for binding in head.calibrations:
            source_task, expected_domain = _ACCEPTED_TASK_BINDINGS[
                (head.task_id, binding.entity_type)
            ]
            task_evidence = evidence_by_task.get(source_task)
            if task_evidence is None:
                raise ValueError("serving semantic task is absent from accepted evidence")
            if (
                task_evidence.adapter_domain != expected_domain
                or binding.adapter_domain != task_evidence.adapter_domain
            ):
                raise ValueError("serving semantic task/domain differs from accepted task evidence")
            snapshot = _read_ref(
                root, binding.calibration_relative_path, max_bytes=_MAX_EVIDENCE_BYTES
            )
            if hashlib.sha256(snapshot).hexdigest() != binding.calibration_sha256:
                raise ValueError("confidence source byte hash differs from serving definition")
            artifact: ScoreCalibration | RegressionConfidenceArtifact
            if binding.confidence_kind == "regression-interval":
                artifact = RegressionConfidenceArtifact.model_validate_json(snapshot)
                if artifact.validation_head_report_hash != task_evidence.head_report_hash:
                    raise ValueError(
                        "regression confidence is not bound to accepted validation head"
                    )
            else:
                artifact = ScoreCalibration.model_validate_json(snapshot)
                record = ledger.load_run(task_evidence.cell_id)
                refs = [item for item in record.artifacts if item.role == "calibration-report"]
                if len(refs) != 1 or task_evidence.calibration_hash is None:
                    raise ValueError("accepted binary task lacks one calibration report")
                from .calibration import CalibrationFitReport

                report_snapshot = _read_ref(
                    root, refs[0].relative_path, max_bytes=_MAX_EVIDENCE_BYTES
                )
                fit_report = CalibrationFitReport.model_validate_json(report_snapshot)
                if (
                    report_snapshot != _canonical_bytes(fit_report)
                    or fit_report.report_hash != task_evidence.calibration_hash
                    or fit_report.calibration != artifact
                ):
                    raise ValueError(
                        "serving calibration is not the accepted validation calibration"
                    )
            version = (
                artifact.calibration_version
                if isinstance(artifact, ScoreCalibration)
                else artifact.confidence_version
            )
            if (
                version != binding.calibration_version
                or artifact.method != binding.calibration_method
                or artifact.artifact_hash != binding.calibration_artifact_hash
                or artifact.protocol_hash != binding.calibration_protocol_hash
            ):
                raise ValueError("confidence artifact differs from task/entity binding")
            result[(head.task_id, binding.entity_type)] = snapshot
    if tuple(result) != _TASK_ENTITY_ORDER:
        raise ValueError("confidence source inventory is incomplete or out of order")
    return result


def _materialize_formal_candidate(
    report: CoreAcceptance,
    derivation: AcceptanceDerivationInputs,
    definition: CandidateServingDefinition,
) -> _CandidateBytes:
    root = secure_existing_root(derivation.runtime_root)
    manifest_snapshot = _read_inside(
        root, derivation.candidate_manifest_path, max_bytes=_MAX_EVIDENCE_BYTES
    )
    manifest = CandidateGovernanceManifest.model_validate_json(manifest_snapshot)
    if (
        manifest_snapshot != _canonical_bytes(manifest)
        or manifest.manifest_hash != report.candidate_manifest_hash
    ):
        raise ValueError("candidate manifest differs from accepted formal evidence")
    task_binding_inventory_hash = _accepted_task_binding_inventory_hash(definition, manifest)
    source_checkpoint = _read_ref(
        root, manifest.best_checkpoint.relative_path, max_bytes=_MAX_CHECKPOINT_BYTES
    )
    if (
        hashlib.sha256(source_checkpoint).hexdigest() != report.candidate_checkpoint_sha256
        or manifest.best_checkpoint.byte_sha256 != report.candidate_checkpoint_sha256
    ):
        raise ValueError("candidate checkpoint differs from accepted formal evidence")
    bindings = CheckpointBindings(
        config_hash=manifest.execution.config_hash,
        data_hash=manifest.training_inventory.inventory_hash,
        code_hash=manifest.code_hash,
        environment_hash=manifest.environment_hash,
    )
    source_payload = load_checkpoint(source_checkpoint, expected_bindings=bindings)
    trainer = source_payload["trainer"]
    model_state = trainer.get("model")
    schemas = trainer.get("adapterSchemas")
    states = trainer.get("adapters")
    domains = tuple(sorted(item.adapter_domain for item in manifest.training_inventory.tasks))
    if (
        not isinstance(model_state, dict)
        or not isinstance(schemas, dict)
        or not isinstance(states, dict)
        or len(domains) != 4
        or len(set(domains)) != 4
        or any(domain not in schemas or domain not in states for domain in domains)
    ):
        raise ValueError("accepted checkpoint lacks the four exact serving adapters")
    adapter_bindings: list[ServingAdapterBinding] = []
    selected_schemas: dict[str, Any] = {}
    selected_states: dict[str, Any] = {}
    for domain in domains:
        from .adapters import AdapterSchema

        schema = AdapterSchema.model_validate(schemas[domain])
        state = states[domain]
        if not isinstance(state, dict):
            raise ValueError("accepted adapter state is invalid")
        bucket_counts = {
            field.bucket_count for field in schema.fields if hasattr(field, "bucket_count")
        }
        if len(bucket_counts) > 1:
            raise ValueError("one adapter schema cannot mix multi-hot bucket inventories")
        adapter_binding = ServingAdapterBinding(
            adapterDomain=domain,
            adapterSchemaHash=schema.adapter_schema_hash,
            adapterStateHash=_tensor_state_hash(state),
            multiHotBuckets=next(iter(bucket_counts), 256),
        )
        adapter_bindings.append(adapter_binding)
        selected_schemas[domain] = schema.model_dump(mode="python", by_alias=True)
        selected_states[domain] = state
    definition_bindings = {
        binding.adapter_domain: binding
        for head in definition.task_heads
        for binding in head.calibrations
    }
    if set(definition_bindings) != set(domains):
        raise ValueError("serving definition does not use the four accepted adapters")
    for adapter in adapter_bindings:
        task_binding = definition_bindings[adapter.adapter_domain]
        if (
            task_binding.adapter_schema_hash != adapter.adapter_schema_hash
            or task_binding.adapter_state_hash != adapter.adapter_state_hash
        ):
            raise ValueError("serving definition adapter identity differs from checkpoint")
    confidence = _source_confidence_artifacts(root=root, manifest=manifest, definition=definition)
    serving_checkpoint = _serialize_checkpoint(
        {
            "model": model_state,
            "adapterSchemas": selected_schemas,
            "adapters": selected_states,
        },
        bindings=bindings,
    )
    serving_sha = hashlib.sha256(serving_checkpoint).hexdigest()
    confidence_paths = {
        key: f"artifacts/confidence-{hashlib.sha256(value).hexdigest()}.json"
        for key, value in confidence.items()
    }
    task_heads: list[ServingTaskHead] = []
    for head in definition.task_heads:
        updated: list[CalibrationBinding] = []
        for confidence_binding in head.calibrations:
            key = (head.task_id, confidence_binding.entity_type)
            snapshot = confidence[key]
            updated.append(
                confidence_binding.model_copy(
                    update={
                        "calibration_relative_path": confidence_paths[key],
                        "calibration_sha256": hashlib.sha256(snapshot).hexdigest(),
                    }
                )
            )
        task_heads.append(head.model_copy(update={"calibrations": tuple(updated)}))
    primary = adapter_bindings[0]
    node_weight = model_state.get("node_head.weight")
    node_classes = int(getattr(node_weight, "shape", (0,))[0])
    serving_manifest_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-checkpoint-manifest/1.1",
        "task4CheckpointSha256": serving_sha,
        "accepted": True,
        "promotable": True,
        "modelStateHash": _tensor_state_hash(model_state),
        "adapterStateHash": primary.adapter_state_hash,
        "adapterSchemaHash": primary.adapter_schema_hash,
        "adapterDomain": primary.adapter_domain,
        "nodeClasses": node_classes,
        "multiHotBuckets": primary.multi_hot_buckets,
        "adapterBindings": [
            item.model_dump(mode="python", by_alias=True) for item in adapter_bindings
        ],
        "taskHeads": [item.model_dump(mode="python", by_alias=True) for item in task_heads],
    }
    serving_manifest = ServingCheckpointManifest.model_validate(serving_manifest_payload)
    serving_manifest_snapshot = _canonical_bytes(serving_manifest)
    manifest_sha = hashlib.sha256(serving_manifest_snapshot).hexdigest()
    feature_inventory = [
        {
            "taskId": head.task_id,
            "entityType": binding.entity_type,
            "featureContractHash": binding.graph_feature_contract_hash,
        }
        for head in task_heads
        for binding in head.calibrations
    ]
    model_payload: dict[str, Any] = {
        "modelVersionId": definition.model_version_id,
        "state": "accepted",
        "checkpoint": {
            "relativePath": f"artifacts/checkpoint-{serving_sha}.pt",
            "sha256": serving_sha,
            "servingManifestRelativePath": f"artifacts/manifest-{manifest_sha}.json",
            "servingManifestSha256": manifest_sha,
            "bindings": {
                "configHash": bindings.config_hash,
                "dataHash": bindings.data_hash,
                "codeHash": bindings.code_hash,
                "environmentHash": bindings.environment_hash,
            },
            "adapterDomain": primary.adapter_domain,
            "nodeClasses": node_classes,
            "multiHotBuckets": primary.multi_hot_buckets,
        },
        "taskHeads": [item.model_dump(mode="python", by_alias=True) for item in task_heads],
        "tasks": [item.task_id for item in task_heads],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(feature_inventory),
        "maxNodes": definition.max_nodes,
        "maxEdges": definition.max_edges,
    }
    model_payload["modelVersionHash"] = canonical_sha256(
        {key: value for key, value in model_payload.items() if key != "state"}
    )
    serving_model = ServingModel.model_validate(model_payload)
    _validate_captured_checkpoint(serving_model, serving_manifest_snapshot, serving_checkpoint)
    for head in serving_model.task_heads:
        _validate_captured_calibrations(
            serving_model,
            head.task_id,
            {
                binding.entity_type: confidence[(head.task_id, binding.entity_type)]
                for binding in head.calibrations
            },
        )
    return _CandidateBytes(
        acceptance=_canonical_bytes(report),
        candidate_manifest=manifest_snapshot,
        source_checkpoint=source_checkpoint,
        serving_checkpoint=serving_checkpoint,
        serving_manifest=serving_manifest_snapshot,
        serving_model=serving_model,
        task_binding_inventory_hash=task_binding_inventory_hash,
        confidence=tuple(
            (task_id, entity_type, confidence[(task_id, entity_type)])
            for task_id, entity_type in _TASK_ENTITY_ORDER
        ),
    )


_CANDIDATE_MATERIALIZER = _materialize_formal_candidate


def stage_candidate(
    *,
    report: CoreAcceptance,
    derivation: AcceptanceDerivationInputs,
    definition: CandidateServingDefinition,
    stage_root: Path,
) -> CandidateStage:
    """Stage only a fully derived formal report; rejected reports write no bytes."""

    if type(report) is not CoreAcceptance or not (
        report.accepted and report.promotable and report.status == "accepted"
    ):
        raise ValueError("candidate staging requires a complete accepted formal report")
    if type(derivation) is not AcceptanceDerivationInputs:
        raise TypeError("candidate staging requires exact acceptance derivation inputs")
    if type(definition) is not CandidateServingDefinition:
        raise TypeError("candidate staging requires an exact serving definition")
    rederived = derivation.derive()
    if type(rederived) is not CoreAcceptance or rederived != report:
        raise ValueError("formal acceptance changed during candidate staging")
    materialized = _CANDIDATE_MATERIALIZER(report, derivation, definition)
    artifact_values: list[tuple[str, str | None, str | None, str, bytes]] = [
        ("acceptance-report", None, None, "json", materialized.acceptance),
        ("candidate-manifest", None, None, "json", materialized.candidate_manifest),
        ("source-checkpoint", None, None, "pt", materialized.source_checkpoint),
        ("serving-checkpoint", None, None, "pt", materialized.serving_checkpoint),
        ("serving-manifest", None, None, "json", materialized.serving_manifest),
    ]
    artifact_values.extend(
        ("confidence", task_id, entity_type, "json", snapshot)
        for task_id, entity_type, snapshot in materialized.confidence
    )
    model_snapshot = _canonical_bytes(materialized.serving_model)
    artifact_values.append(("serving-model", None, None, "json", model_snapshot))
    artifacts: list[StageArtifact] = []
    writes: list[tuple[str, bytes]] = []
    for role, task_id, entity_type, suffix, snapshot in artifact_values:
        digest = hashlib.sha256(snapshot).hexdigest()
        relative = f"artifacts/{role}-{digest}.{suffix}"
        artifacts.append(
            StageArtifact(
                role=role,
                taskId=task_id,
                entityType=entity_type,
                relativePath=relative,
                sha256=digest,
                sizeBytes=len(snapshot),
            )
        )
        writes.append((relative, snapshot))
    artifacts.sort(key=lambda item: (item.role, item.task_id or "", item.entity_type or ""))
    artifact_hash = canonical_sha256(
        [item.model_dump(mode="python", by_alias=True) for item in artifacts]
    )
    stage_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-candidate-stage/1.0",
        "acceptanceHash": report.acceptance_hash,
        "candidateManifestHash": report.candidate_manifest_hash,
        "experimentSummaryHash": canonical_sha256(
            {
                "aggregateHashes": report.aggregate_hashes,
                "transferDecisionHashes": report.transfer_decision_hashes,
                "winnerSelectionHash": report.winner_selection_hash,
            }
        ),
        "sourceCheckpointSha256": report.candidate_checkpoint_sha256,
        "servingCheckpointSha256": hashlib.sha256(materialized.serving_checkpoint).hexdigest(),
        "servingModel": materialized.serving_model.model_dump(mode="python", by_alias=True),
        "definitionHash": definition.definition_hash,
        "taskBindingInventoryHash": materialized.task_binding_inventory_hash,
        "artifacts": [item.model_dump(mode="python", by_alias=True) for item in artifacts],
        "artifactInventoryHash": artifact_hash,
    }
    stage_payload["stageHash"] = canonical_sha256(stage_payload)
    stage = CandidateStage.model_validate(stage_payload)
    stage_snapshot = _canonical_bytes(stage)
    target_root = Path(stage_root)
    for relative, snapshot in writes:
        _atomic_immutable(
            derivation.runtime_root,
            target_root.joinpath(*PurePosixPath(relative).parts),
            snapshot,
        )
    stage_path = target_root / "candidates" / f"{stage.stage_hash}.json"
    _atomic_immutable(derivation.runtime_root, stage_path, stage_snapshot)
    if stage_path.read_bytes() != stage_snapshot:
        raise RuntimeError("candidate stage changed after exact publication")
    return stage


class AcceptedCandidate(_StrictModel):
    """Append-only proof that a staged candidate was independently rederived."""

    schema_version: Literal["socialgraph-fm.core-accepted-candidate/1.0"] = Field(
        alias="schemaVersion"
    )
    status: Literal["accepted"]
    accepted: Literal[True]
    candidate_stage_hash: str = Field(alias="candidateStageHash", pattern=_HASH)
    acceptance_hash: str = Field(alias="acceptanceHash", pattern=_HASH)
    candidate_manifest_hash: str = Field(alias="candidateManifestHash", pattern=_HASH)
    experiment_summary_hash: str = Field(alias="experimentSummaryHash", pattern=_HASH)
    source_checkpoint_sha256: str = Field(alias="sourceCheckpointSha256", pattern=_HASH)
    serving_checkpoint_sha256: str = Field(alias="servingCheckpointSha256", pattern=_HASH)
    serving_model_version_id: str = Field(
        alias="servingModelVersionId", min_length=1, max_length=300
    )
    serving_model_hash: str = Field(alias="servingModelHash", pattern=_HASH)
    task_binding_inventory_hash: str = Field(alias="taskBindingInventoryHash", pattern=_HASH)
    artifact_inventory_hash: str = Field(alias="artifactInventoryHash", pattern=_HASH)
    acceptance_revalidation_hash: str = Field(alias="acceptanceRevalidationHash", pattern=_HASH)
    accepted_hash: str = Field(alias="acceptedHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"accepted_hash"})
        )
        if self.accepted_hash != expected:
            raise ValueError("acceptedHash does not bind the independently accepted candidate")
        return self


def _reopen_stage(stage: CandidateStage, stage_root: Path) -> _CandidateBytes:
    if type(stage) is not CandidateStage:
        raise TypeError("candidate acceptance requires an exact CandidateStage")
    root = secure_existing_root(stage_root)
    snapshots: dict[tuple[str, str | None, str | None], bytes] = {}
    for artifact in stage.artifacts:
        snapshot = _read_ref(root, artifact.relative_path, max_bytes=_MAX_CHECKPOINT_BYTES)
        if (
            len(snapshot) != artifact.size_bytes
            or hashlib.sha256(snapshot).hexdigest() != artifact.sha256
        ):
            raise ValueError("candidate stage artifact bytes changed")
        snapshots[(artifact.role, artifact.task_id, artifact.entity_type)] = snapshot
    acceptance = CoreAcceptance.model_validate_json(
        snapshots[("acceptance-report", None, None)]
    )
    manifest_snapshot = snapshots[("candidate-manifest", None, None)]
    source_checkpoint = snapshots[("source-checkpoint", None, None)]
    serving_checkpoint = snapshots[("serving-checkpoint", None, None)]
    serving_manifest = snapshots[("serving-manifest", None, None)]
    serving_model_snapshot = snapshots[("serving-model", None, None)]
    serving_model = ServingModel.model_validate_json(serving_model_snapshot)
    if (
        _canonical_bytes(acceptance) != snapshots[("acceptance-report", None, None)]
        or _canonical_bytes(serving_model) != serving_model_snapshot
        or acceptance.acceptance_hash != stage.acceptance_hash
        or hashlib.sha256(source_checkpoint).hexdigest() != stage.source_checkpoint_sha256
        or hashlib.sha256(serving_checkpoint).hexdigest() != stage.serving_checkpoint_sha256
        or serving_model != stage.serving_model
    ):
        raise ValueError("candidate stage semantic binding changed")
    _validate_captured_checkpoint(serving_model, serving_manifest, serving_checkpoint)
    confidence: list[tuple[str, str, bytes]] = []
    for head in serving_model.task_heads:
        task_snapshots: dict[str, bytes] = {
            binding.entity_type: snapshots[("confidence", head.task_id, binding.entity_type)]
            for binding in head.calibrations
        }
        _validate_captured_calibrations(serving_model, head.task_id, task_snapshots)
        confidence.extend(
            (head.task_id, binding.entity_type, task_snapshots[binding.entity_type])
            for binding in head.calibrations
        )
    if tuple((task, entity) for task, entity, _snapshot in confidence) != _TASK_ENTITY_ORDER:
        raise ValueError("candidate stage confidence inventory changed")
    return _CandidateBytes(
        acceptance=snapshots[("acceptance-report", None, None)],
        candidate_manifest=manifest_snapshot,
        source_checkpoint=source_checkpoint,
        serving_checkpoint=serving_checkpoint,
        serving_manifest=serving_manifest,
        serving_model=serving_model,
        task_binding_inventory_hash=stage.task_binding_inventory_hash,
        confidence=tuple(confidence),
    )


def accept_candidate(
    *,
    stage: CandidateStage,
    report: CoreAcceptance,
    derivation: AcceptanceDerivationInputs,
    definition: CandidateServingDefinition,
    stage_root: Path,
    accepted_root: Path,
) -> AcceptedCandidate:
    """Independently rederive and append an accepted record without touching live state."""

    if type(report) is not CoreAcceptance or not (
        report.accepted and report.promotable and report.status == "accepted"
    ):
        raise ValueError("candidate acceptance requires a complete accepted formal report")
    if type(derivation) is not AcceptanceDerivationInputs:
        raise TypeError("candidate acceptance requires exact derivation inputs")
    if type(definition) is not CandidateServingDefinition:
        raise TypeError("candidate acceptance requires exact serving definition")
    reopened = _reopen_stage(stage, stage_root)
    staged_manifest = CandidateGovernanceManifest.model_validate_json(reopened.candidate_manifest)
    if (
        reopened.candidate_manifest != _canonical_bytes(staged_manifest)
        or staged_manifest.manifest_hash != stage.candidate_manifest_hash
    ):
        raise ValueError("candidate manifest changed during independent acceptance")
    rederived = derivation.derive()
    if type(rederived) is not CoreAcceptance or rederived != report:
        raise ValueError("formal acceptance changed during independent candidate acceptance")
    staged_report = CoreAcceptance.model_validate_json(reopened.acceptance)
    expected_binding_inventory_hash = _accepted_task_binding_inventory_hash(
        definition, staged_manifest
    )
    if (
        staged_report != report
        or stage.definition_hash != definition.definition_hash
        or stage.task_binding_inventory_hash != expected_binding_inventory_hash
    ):
        raise ValueError("candidate stage differs from independently accepted inputs")
    revalidation_hash = canonical_sha256(
        {
            "acceptanceHash": rederived.acceptance_hash,
            "candidateStageHash": stage.stage_hash,
            "artifactInventoryHash": stage.artifact_inventory_hash,
            "servingCheckpointSha256": stage.serving_checkpoint_sha256,
            "servingModelHash": stage.serving_model.model_version_hash,
        }
    )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-accepted-candidate/1.0",
        "status": "accepted",
        "accepted": True,
        "candidateStageHash": stage.stage_hash,
        "acceptanceHash": report.acceptance_hash,
        "candidateManifestHash": stage.candidate_manifest_hash,
        "experimentSummaryHash": stage.experiment_summary_hash,
        "sourceCheckpointSha256": stage.source_checkpoint_sha256,
        "servingCheckpointSha256": stage.serving_checkpoint_sha256,
        "servingModelVersionId": stage.serving_model.model_version_id,
        "servingModelHash": stage.serving_model.model_version_hash,
        "taskBindingInventoryHash": stage.task_binding_inventory_hash,
        "artifactInventoryHash": stage.artifact_inventory_hash,
        "acceptanceRevalidationHash": revalidation_hash,
    }
    payload["acceptedHash"] = canonical_sha256(payload)
    accepted = AcceptedCandidate.model_validate(payload)
    snapshot = _canonical_bytes(accepted)
    target = Path(accepted_root) / "accepted" / f"{accepted.accepted_hash}.json"
    _atomic_immutable(derivation.runtime_root, target, snapshot)
    if target.read_bytes() != snapshot:
        raise RuntimeError("accepted candidate changed after append-only publication")
    return accepted


class ServingSmokeTaskResult(_StrictModel):
    """One fresh-process observation for one public task output entity."""

    schema_version: Literal["socialgraph-fm.core-serving-smoke-task/1.0"] = Field(
        alias="schemaVersion"
    )
    task_id: Literal[
        "core.community_resilience_review",
        "core.risk_and_trust_review",
        "core.collaboration_completion",
    ] = Field(alias="taskId")
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(alias="entityType")
    fixture_artifact_hash: str = Field(alias="fixtureArtifactHash", pattern=_HASH)
    fixture_bundle_sha256: str = Field(alias="fixtureBundleSha256", pattern=_HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH)
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH)
    confidence_artifact_hash: str = Field(alias="confidenceArtifactHash", pattern=_HASH)
    confidence_protocol_hash: str = Field(alias="confidenceProtocolHash", pattern=_HASH)
    request_hash: str = Field(alias="requestHash", pattern=_HASH)
    result_hash: str = Field(alias="resultHash", pattern=_HASH)
    finding_hashes: tuple[str, ...] = Field(alias="findingHashes", min_length=1, strict=False)
    all_pending_human_review: Literal[True] = Field(alias="allPendingHumanReview")
    task_result_hash: str = Field(alias="taskResultHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_binding(self):
        if (self.task_id, self.entity_type) not in _TASK_ENTITY_ORDER:
            raise ValueError("serving smoke task/entity pairing is not declared")
        if self.finding_hashes != tuple(sorted(set(self.finding_hashes))):
            raise ValueError("serving smoke findings must be unique and sorted")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"task_result_hash"})
        )
        if self.task_result_hash != expected:
            raise ValueError("taskResultHash does not bind the fresh-process task result")
        return self


class ServingSmokeReport(_StrictModel):
    """Independent fresh-process proof; success is derived, never caller-controlled."""

    schema_version: Literal["socialgraph-fm.core-serving-smoke/1.0"] = Field(
        alias="schemaVersion"
    )
    accepted_candidate_hash: str = Field(alias="acceptedCandidateHash", pattern=_HASH)
    acceptance_hash: str = Field(alias="acceptanceHash", pattern=_HASH)
    serving_model_version_id: str = Field(
        alias="servingModelVersionId", min_length=1, max_length=300
    )
    serving_model_hash: str = Field(alias="servingModelHash", pattern=_HASH)
    source_checkpoint_sha256: str = Field(alias="sourceCheckpointSha256", pattern=_HASH)
    serving_checkpoint_sha256: str = Field(alias="servingCheckpointSha256", pattern=_HASH)
    task_binding_inventory_hash: str = Field(alias="taskBindingInventoryHash", pattern=_HASH)
    process_interpreter_sha256: str = Field(alias="processInterpreterSha256", pattern=_HASH)
    process_environment_hash: str = Field(alias="processEnvironmentHash", pattern=_HASH)
    task_results: tuple[ServingSmokeTaskResult, ...] = Field(alias="taskResults", strict=False)
    failed_gates: tuple[str, ...] = Field(alias="failedGates", strict=False)
    succeeded: bool
    smoke_hash: str = Field(alias="smokeHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_report(self):
        observed = tuple((item.task_id, item.entity_type) for item in self.task_results)
        if observed != _TASK_ENTITY_ORDER:
            raise ValueError("serving smoke must contain the exact task/entity inventory")
        if self.failed_gates != tuple(sorted(set(self.failed_gates))):
            raise ValueError("serving smoke failed gates must be unique and sorted")
        expected_success = not self.failed_gates
        if self.succeeded != expected_success:
            raise ValueError("serving smoke succeeded state must be derived from failed gates")
        expected_hash = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"smoke_hash"})
        )
        if self.smoke_hash != expected_hash:
            raise ValueError("smokeHash does not bind the fresh-process serving evidence")
        return self


class ServingSmokeFixture(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-smoke-fixture/1.0"] = Field(
        alias="schemaVersion"
    )
    task_id: Literal[
        "core.community_resilience_review",
        "core.risk_and_trust_review",
        "core.collaboration_completion",
    ] = Field(alias="taskId")
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(alias="entityType")
    artifact: ArtifactEntry
    request: GfmRunRequest
    fixture_hash: str = Field(alias="fixtureHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_fixture(self):
        if (self.task_id, self.entity_type) not in _TASK_ENTITY_ORDER:
            raise ValueError("serving smoke fixture task/entity pairing is not declared")
        if self.request.task_id != self.task_id:
            raise ValueError("serving smoke request task differs from fixture")
        requested_entity = CoreServingHead._requested_entity_type(self.request)
        if requested_entity != self.entity_type:
            raise ValueError("serving smoke request entity differs from fixture")
        if self.request.graph_version_id != self.artifact.graph_version_id:
            raise ValueError("serving smoke request graph differs from fixture")
        if self.fixture_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"fixture_hash"})
        ):
            raise ValueError("fixtureHash does not bind fixture graph and request")
        return self


class _SmokeFixtureInventory(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-smoke-inventory/1.0"] = Field(
        alias="schemaVersion"
    )
    fixtures: tuple[ServingSmokeFixture, ...] = Field(strict=False)
    inventory_hash: str = Field(alias="inventoryHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_inventory(self):
        if tuple((item.task_id, item.entity_type) for item in self.fixtures) != _TASK_ENTITY_ORDER:
            raise ValueError("serving smoke requires the exact four fixture bindings")
        if self.inventory_hash != canonical_sha256(
            [item.model_dump(mode="python", by_alias=True) for item in self.fixtures]
        ):
            raise ValueError("serving smoke fixture inventory hash mismatch")
        return self


_SMOKE_FACTORY_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedServingSmoke:
    """Process-local capability proving the report came from the fresh worker."""

    report: ServingSmokeReport
    fixture_inventory_hash: str
    _factory_seal: object

    def verify(self) -> None:
        if (
            type(self) is not VerifiedServingSmoke
            or self._factory_seal is not _SMOKE_FACTORY_SEAL
            or type(self.report) is not ServingSmokeReport
            or not self.report.succeeded
        ):
            raise TypeError("serving promotion requires sealed fresh-process smoke evidence")


def _new_verified_smoke(
    report: ServingSmokeReport, fixture_inventory_hash: str
) -> VerifiedServingSmoke:
    verified = object.__new__(VerifiedServingSmoke)
    object.__setattr__(verified, "report", report)
    object.__setattr__(verified, "fixture_inventory_hash", fixture_inventory_hash)
    object.__setattr__(verified, "_factory_seal", _SMOKE_FACTORY_SEAL)
    return verified


def _process_identity() -> tuple[str, str]:
    executable = Path(sys.executable).resolve(strict=True)
    interpreter_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    try:
        import torch_geometric

        pyg_version = torch_geometric.__version__
    except Exception:  # pragma: no cover - production GFM environment includes PyG
        pyg_version = "unavailable"
    environment_hash = canonical_sha256(
        {
            "implementation": platform.python_implementation(),
            "pythonVersion": platform.python_version(),
            "platform": platform.platform(),
            "torchVersion": torch.__version__,
            "pygVersion": pyg_version,
            "interpreterSha256": interpreter_hash,
        }
    )
    return interpreter_hash, environment_hash


def _materialize_fixture(
    root: Path, fixture: ServingSmokeFixture
) -> tuple[bytes, CoreGraphBundle]:
    snapshot = _read_ref(root, fixture.artifact.relative_path, max_bytes=_MAX_CHECKPOINT_BYTES)
    if hashlib.sha256(snapshot).hexdigest() != fixture.artifact.bundle_sha256:
        raise ValueError("serving smoke fixture bundle hash mismatch")
    bundle = load_core_graph_bundle_json(snapshot)
    if snapshot != _canonical_bytes(bundle):
        raise ValueError("serving smoke fixture bundle is not canonical")
    feature_contract = feature_contract_for_bundle(bundle)
    if (
        bundle.graph_version_hash != fixture.artifact.graph_version_hash
        or bundle.schema_version != fixture.artifact.graph_schema_version
        or len(bundle.nodes) != fixture.artifact.node_count
        or len(bundle.edges) != fixture.artifact.edge_count
        or feature_contract != fixture.artifact.feature_contract
    ):
        raise ValueError("serving smoke fixture differs from its artifact binding")
    return snapshot, bundle


def _smoke_worker(
    *,
    accepted_path: Path,
    stage_path: Path,
    stage_root: Path,
    fixture_root: Path,
    fixture_inventory_path: Path,
) -> ServingSmokeReport:
    accepted_snapshot = accepted_path.read_bytes()
    accepted = AcceptedCandidate.model_validate_json(accepted_snapshot)
    stage_snapshot = stage_path.read_bytes()
    stage = CandidateStage.model_validate_json(stage_snapshot)
    fixture_snapshot = fixture_inventory_path.read_bytes()
    inventory = _SmokeFixtureInventory.model_validate_json(fixture_snapshot)
    if (
        accepted_snapshot != _canonical_bytes(accepted)
        or stage_snapshot != _canonical_bytes(stage)
        or fixture_snapshot != _canonical_bytes(inventory)
        or accepted.candidate_stage_hash != stage.stage_hash
        or accepted.acceptance_hash != stage.acceptance_hash
        or accepted.serving_model_hash != stage.serving_model.model_version_hash
        or accepted.serving_checkpoint_sha256 != stage.serving_checkpoint_sha256
    ):
        raise ValueError("fresh-process inputs are not the exact accepted candidate")
    candidate = _reopen_stage(stage, stage_root)
    model = candidate.serving_model
    manifest, checkpoint = _validate_captured_checkpoint(
        model, candidate.serving_manifest, candidate.serving_checkpoint
    )
    if manifest.task4_checkpoint_sha256 != accepted.serving_checkpoint_sha256:
        raise ValueError("fresh-process checkpoint differs from accepted candidate")
    confidence_by_key = {
        (task, entity): snapshot for task, entity, snapshot in candidate.confidence
    }
    results: list[ServingSmokeTaskResult] = []
    root = secure_existing_root(fixture_root)
    for fixture in inventory.fixtures:
        bundle_snapshot, bundle = _materialize_fixture(root, fixture)
        if fixture.request.model_version_id != model.model_version_id:
            raise ValueError("fresh-process request model differs from accepted candidate")
        binding = model.task_head(fixture.task_id).calibration(fixture.entity_type)
        if binding.graph_feature_contract_hash != fixture.artifact.feature_contract_hash:
            raise ValueError("fresh-process fixture feature contract differs from task binding")
        calibrations = _validate_captured_calibrations(
            model,
            fixture.task_id,
            {
                item.entity_type: confidence_by_key[(fixture.task_id, item.entity_type)]
                for item in model.task_head(fixture.task_id).calibrations
            },
        )
        findings = CoreServingHead().execute(
            fixture.request, bundle, model, checkpoint, calibrations
        )
        if not findings or any(item.review_status != "pending-human-review" for item in findings):
            raise ValueError("fresh-process task produced no pending-human-review finding")
        result_hash = canonical_sha256(
            [item.model_dump(mode="python", by_alias=True) for item in findings]
        )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-serving-smoke-task/1.0",
            "taskId": fixture.task_id,
            "entityType": fixture.entity_type,
            "fixtureArtifactHash": fixture.artifact.artifact_hash,
            "fixtureBundleSha256": hashlib.sha256(bundle_snapshot).hexdigest(),
            "graphVersionHash": bundle.graph_version_hash,
            "featureContractHash": fixture.artifact.feature_contract_hash,
            "adapterDomain": binding.adapter_domain,
            "adapterSchemaHash": binding.adapter_schema_hash,
            "adapterStateHash": binding.adapter_state_hash,
            "confidenceArtifactHash": binding.calibration_artifact_hash,
            "confidenceProtocolHash": binding.calibration_protocol_hash,
            "requestHash": canonical_sha256(
                fixture.request.model_dump(mode="python", by_alias=True)
            ),
            "resultHash": result_hash,
            "findingHashes": sorted(item.finding_hash for item in findings),
            "allPendingHumanReview": True,
        }
        payload["taskResultHash"] = canonical_sha256(payload)
        results.append(ServingSmokeTaskResult.model_validate(payload))
    interpreter_hash, environment_hash = _process_identity()
    report_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke/1.0",
        "acceptedCandidateHash": accepted.accepted_hash,
        "acceptanceHash": accepted.acceptance_hash,
        "servingModelVersionId": accepted.serving_model_version_id,
        "servingModelHash": accepted.serving_model_hash,
        "sourceCheckpointSha256": accepted.source_checkpoint_sha256,
        "servingCheckpointSha256": accepted.serving_checkpoint_sha256,
        "taskBindingInventoryHash": accepted.task_binding_inventory_hash,
        "processInterpreterSha256": interpreter_hash,
        "processEnvironmentHash": environment_hash,
        "taskResults": [item.model_dump(mode="python", by_alias=True) for item in results],
        "failedGates": [],
        "succeeded": True,
    }
    report_payload["smokeHash"] = canonical_sha256(report_payload)
    return ServingSmokeReport.model_validate(report_payload)


def run_fresh_process_serving_smoke(
    *,
    accepted: AcceptedCandidate,
    stage: CandidateStage,
    stage_root: Path,
    fixture_root: Path,
    fixtures: tuple[ServingSmokeFixture, ...],
    python_executable: Path | None = None,
    timeout_seconds: float = 300.0,
    publish_to: Path | None = None,
) -> VerifiedServingSmoke:
    """Execute the exact accepted bytes in a separate canonical GFM process."""

    if type(accepted) is not AcceptedCandidate or type(stage) is not CandidateStage:
        raise TypeError("fresh serving smoke requires exact accepted/stage evidence")
    if accepted.candidate_stage_hash != stage.stage_hash:
        raise ValueError("accepted candidate and candidate stage differ")
    inventory_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke-inventory/1.0",
        "fixtures": [item.model_dump(mode="python", by_alias=True) for item in fixtures],
    }
    inventory_payload["inventoryHash"] = canonical_sha256(inventory_payload["fixtures"])
    inventory = _SmokeFixtureInventory.model_validate(inventory_payload)
    executable = Path(sys.executable if python_executable is None else python_executable)
    executable = executable.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="socialgraph-gfm-smoke-") as temporary_name:
        temporary = Path(temporary_name)
        accepted_path = temporary / "accepted.json"
        stage_path = temporary / "stage.json"
        inventory_path = temporary / "fixtures.json"
        accepted_path.write_bytes(_canonical_bytes(accepted))
        stage_path.write_bytes(_canonical_bytes(stage))
        inventory_path.write_bytes(_canonical_bytes(inventory))
        command = [
            str(executable),
            "-m",
            "socialgraph_gfm.core.promotion",
            "--fresh-smoke-worker",
            str(accepted_path),
            str(stage_path),
            str(Path(stage_root).resolve(strict=True)),
            str(Path(fixture_root).resolve(strict=True)),
            str(inventory_path),
        ]
        environment = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_root if not existing else os.pathsep.join((source_root, existing))
        )
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
            env=environment,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"fresh-process serving smoke failed closed: {detail}")
    report = ServingSmokeReport.model_validate_json(completed.stdout)
    if completed.stdout != _canonical_bytes(report):
        raise ValueError("fresh-process serving smoke output is not canonical")
    expected_interpreter_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    if (
        report.accepted_candidate_hash != accepted.accepted_hash
        or report.acceptance_hash != accepted.acceptance_hash
        or report.serving_model_version_id != accepted.serving_model_version_id
        or report.serving_model_hash != accepted.serving_model_hash
        or report.source_checkpoint_sha256 != accepted.source_checkpoint_sha256
        or report.serving_checkpoint_sha256 != accepted.serving_checkpoint_sha256
        or report.task_binding_inventory_hash != accepted.task_binding_inventory_hash
        or report.process_interpreter_sha256 != expected_interpreter_hash
    ):
        raise ValueError("fresh-process serving smoke identity differs from accepted bytes")
    verified = _new_verified_smoke(report, inventory.inventory_hash)
    verified.verify()
    if publish_to is not None:
        _atomic_immutable(
            secure_existing_root(stage_root),
            Path(publish_to),
            _canonical_bytes(report),
        )
    return verified


def _fresh_smoke_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fresh-smoke-worker", action="store_true")
    parser.add_argument("accepted_path")
    parser.add_argument("stage_path")
    parser.add_argument("stage_root")
    parser.add_argument("fixture_root")
    parser.add_argument("fixture_inventory_path")
    parsed = parser.parse_args(arguments)
    if not parsed.fresh_smoke_worker:
        return 2
    report = _smoke_worker(
        accepted_path=Path(parsed.accepted_path),
        stage_path=Path(parsed.stage_path),
        stage_root=Path(parsed.stage_root),
        fixture_root=Path(parsed.fixture_root),
        fixture_inventory_path=Path(parsed.fixture_inventory_path),
    )
    sys.stdout.buffer.write(_canonical_bytes(report))
    return 0


class PromotionReceipt(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-promotion-receipt/1.0"] = Field(
        alias="schemaVersion"
    )
    accepted_candidate_hash: str = Field(alias="acceptedCandidateHash", pattern=_HASH)
    serving_smoke_hash: str = Field(alias="servingSmokeHash", pattern=_HASH)
    serving_model_version_id: str = Field(alias="servingModelVersionId", min_length=1)
    serving_model_hash: str = Field(alias="servingModelHash", pattern=_HASH)
    control_generation: int = Field(alias="controlGeneration", ge=1)
    control_hash: str = Field(alias="controlHash", pattern=_HASH)
    registry_generation: int = Field(alias="registryGeneration", ge=1)
    registry_hash: str = Field(alias="registryHash", pattern=_HASH)
    catalog_generation: int = Field(alias="catalogGeneration", ge=1)
    catalog_hash: str = Field(alias="catalogHash", pattern=_HASH)
    receipt_hash: str = Field(alias="receiptHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_receipt(self):
        if not (self.control_generation == self.registry_generation == self.catalog_generation):
            raise ValueError("promotion generations must advance together")
        if self.receipt_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"receipt_hash"})
        ):
            raise ValueError("promotion receipt hash mismatch")
        return self


def _promotion_receipt(
    *,
    accepted: AcceptedCandidate,
    smoke: ServingSmokeReport,
    control: ServingControlDocument,
) -> PromotionReceipt:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-promotion-receipt/1.0",
        "acceptedCandidateHash": accepted.accepted_hash,
        "servingSmokeHash": smoke.smoke_hash,
        "servingModelVersionId": accepted.serving_model_version_id,
        "servingModelHash": accepted.serving_model_hash,
        "controlGeneration": control.generation,
        "controlHash": control.control_hash,
        "registryGeneration": control.registry.generation,
        "registryHash": control.registry.semantic_hash,
        "catalogGeneration": control.catalog.generation,
        "catalogHash": control.catalog.semantic_hash,
    }
    payload["receiptHash"] = canonical_sha256(payload)
    return PromotionReceipt.model_validate(payload)


def _promotion_receipt_relative(control: ServingControlDocument) -> str:
    return f"versions/promotion-receipt-g{control.generation}-{control.control_hash}.json"


def _load_high_water(serving_control: ServingControlStore) -> ServingHighWater | None:
    if not serving_control.high_water_path.exists():
        return None
    snapshot = _read_ref(
        serving_control.high_water_root,
        serving_control.high_water_path.name,
        max_bytes=_MAX_EVIDENCE_BYTES,
    )
    record = ServingHighWater.model_validate_json(snapshot)
    if snapshot != _canonical_bytes(record):
        raise ValueError("serving high-water evidence is not canonical")
    return record


def _high_water_matches(
    captured: CapturedServingControl,
    high_water: ServingHighWater,
) -> bool:
    return (
        high_water.control_generation == captured.document.generation
        and high_water.control_hash == captured.document.control_hash
        and high_water.registry_generation == captured.registry_document.generation
        and high_water.registry_hash == captured.registry_hash
        and high_water.catalog_generation == captured.catalog_document.generation
        and high_water.catalog_hash == captured.catalog_hash
    )


def _recover_published_promotion(
    *,
    accepted: AcceptedCandidate,
    smoke: ServingSmokeReport,
    stage: CandidateStage,
    reopened: _CandidateBytes,
    fixtures: tuple[tuple[ArtifactEntry, bytes], ...],
    captured: CapturedServingControl,
    control_root: Path,
) -> PromotionReceipt | None:
    """Reopen the prepublished receipt and every live byte it authorizes."""

    receipt_relative = _promotion_receipt_relative(captured.document)
    receipt_path = control_root.joinpath(*PurePosixPath(receipt_relative).parts)
    if not receipt_path.exists():
        return None
    snapshot = _read_ref(control_root, receipt_relative, max_bytes=_MAX_EVIDENCE_BYTES)
    receipt = PromotionReceipt.model_validate_json(snapshot)
    if snapshot != _canonical_bytes(receipt):
        raise ValueError("live promotion receipt is not canonical")
    expected_receipt = _promotion_receipt(
        accepted=accepted,
        smoke=smoke,
        control=captured.document,
    )
    live_ready = tuple(
        item for item in captured.registry_document.models if item.state == "servingReady"
    )
    expected_model = reopened.serving_model.model_copy(update={"state": "servingReady"})
    if receipt != expected_receipt:
        raise ValueError("live promotion recovery receipt differs from candidate or smoke")
    if live_ready != (expected_model,):
        raise ValueError("live promotion recovery model differs from accepted candidate")
    expected_artifacts = (
        (stage.artifact("serving-checkpoint"), reopened.serving_checkpoint),
        (stage.artifact("serving-manifest"), reopened.serving_manifest),
        *(
            (
                stage.artifact("confidence", task_id=task, entity_type=entity),
                snapshot,
            )
            for task, entity, snapshot in reopened.confidence
        ),
    )
    for artifact, artifact_snapshot in expected_artifacts:
        if (
            _read_ref(
                control_root,
                artifact.relative_path,
                max_bytes=_MAX_CHECKPOINT_BYTES,
            )
            != artifact_snapshot
        ):
            raise ValueError("live promotion recovery artifact differs from staged bytes")
    catalog_by_id = {item.artifact_id: item for item in captured.catalog_document.artifacts}
    for entry, fixture_snapshot in fixtures:
        relative = f"artifacts/fixture-{entry.bundle_sha256}.json"
        expected_entry = entry.model_copy(update={"relative_path": relative})
        if catalog_by_id.get(entry.artifact_id) != expected_entry:
            raise ValueError("live promotion recovery catalog differs from smoke fixture")
        if _read_ref(control_root, relative, max_bytes=_MAX_CHECKPOINT_BYTES) != fixture_snapshot:
            raise ValueError("live promotion recovery fixture bytes changed")
    return receipt


def _PROMOTION_FAILURE_SEAM(_stage: str) -> None:
    return


def _CONTROL_SWAP_SEAM(_stage: str, _path: Path) -> None:
    return


def _rename_exchange(parent: _PublicationParentLease, left: str, right: str) -> None:
    """Atomically exchange two held-parent basenames on Linux."""

    if not sys.platform.startswith("linux"):
        raise RuntimeError("atomic control exchange is unavailable on this platform")
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic control exchange is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            parent.descriptor,
            os.fsencode(left),
            parent.descriptor,
            os.fsencode(right),
            2,
        )
        != 0
    ):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _windows_replace_with_backup(target: Path, replacement: Path, backup: Path) -> None:
    kernel32 = getattr(ctypes, "windll").kernel32
    replace_file = kernel32.ReplaceFileW
    replace_file.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    replace_file.restype = ctypes.c_int
    if not replace_file(str(target), str(replacement), str(backup), 1, None, None):
        raise ctypes.WinError()


@dataclass
class _OwnedControlSwap:
    parent: _PublicationParentLease
    target: Path
    backup: Path
    expected: bytes
    replacement: bytes
    new_identity: tuple[int, int]
    backup_identity: tuple[int, int] | None = None
    new_lease: _OwnedFileLease | None = None
    backup_lease: _OwnedFileLease | None = None
    previous: bytes | None = None
    state: Literal["active", "rolled-back", "ownership-lost", "committed"] = "active"

    def acquire_leases(self) -> None:
        """Acquire post-swap proofs only after the caller owns this state."""

        if self.state != "active":
            raise RuntimeError("owned serving control swap is no longer active")
        if self.new_lease is None:
            self.new_lease = _OwnedFileLease(
                self.target,
                self.new_identity,
                deletable=False,
                parent_lease=self.parent,
            )
        if self.backup_identity is None:
            self.backup_identity = _path_identity(self.backup)
        if self.backup_lease is None:
            self.backup_lease = _OwnedFileLease(
                self.backup,
                self.backup_identity,
                deletable=False,
                parent_lease=self.parent,
            )

    def verify(self) -> None:
        self.acquire_leases()
        new_lease = self.new_lease
        backup_lease = self.backup_lease
        if new_lease is None or backup_lease is None:  # pragma: no cover - internal invariant
            raise RuntimeError("owned serving control leases were not acquired")
        new_lease.assert_visible_binding()
        backup_lease.assert_visible_binding()
        current = new_lease.read(max_bytes=_MAX_EVIDENCE_BYTES)
        previous = backup_lease.read(max_bytes=_MAX_EVIDENCE_BYTES)
        if current != self.replacement:
            raise ValueError("owned serving control swap bytes changed")
        if self.previous is None:
            self.previous = previous
        elif previous != self.previous:
            raise ValueError("owned serving control backup bytes changed")

    @property
    def expected_matched(self) -> bool:
        if self.previous is None:
            raise RuntimeError("owned serving control predecessor was not verified")
        return self.previous == self.expected

    def _rollback_windows_without_backup_lease(self) -> bool:
        """Reverse the OS backup when its random name cannot be leased."""

        new_lease = self.new_lease
        if new_lease is None:
            return False
        try:
            new_lease.assert_visible_binding()
            if new_lease.read(max_bytes=_MAX_EVIDENCE_BYTES) != self.replacement:
                return False
        except (OSError, ValueError):
            return False
        replacement_identity = new_lease.identity
        new_lease.close()
        if self.backup_lease is not None:
            self.backup_lease.close()
        rollback_backup = self.target.with_name(
            f".{self.target.name}.{uuid.uuid4().hex}.rolled-back"
        )
        current: _OwnedFileLease | None = None
        displaced: _OwnedFileLease | None = None
        try:
            _windows_replace_with_backup(self.target, self.backup, rollback_backup)
            self.parent.flush()
            current = _OwnedFileLease(
                self.target,
                _path_identity(self.target),
                deletable=False,
                parent_lease=self.parent,
            )
            displaced = _OwnedFileLease(
                rollback_backup,
                _path_identity(rollback_backup),
                deletable=False,
                parent_lease=self.parent,
            )
        except (OSError, ValueError):
            if current is not None:
                current.close()
            if displaced is not None:
                displaced.close()
            return False
        if current is None or displaced is None:  # pragma: no cover - internal invariant
            return False
        try:
            predecessor_snapshot = current.read(max_bytes=_MAX_EVIDENCE_BYTES)
            displaced_snapshot = displaced.read(max_bytes=_MAX_EVIDENCE_BYTES)
            current.assert_visible_binding()
            displaced.assert_visible_binding()
            predecessor_identity = current.identity
            displaced_identity = displaced.identity
        finally:
            current.close()
            displaced.close()
        if displaced_identity == replacement_identity and displaced_snapshot == self.replacement:
            self.previous = predecessor_snapshot
            self.backup_identity = predecessor_identity
            self.state = "rolled-back"
            return True
        correction_backup = self.target.with_name(
            f".{self.target.name}.{uuid.uuid4().hex}.rollback-correction"
        )
        competitor: _OwnedFileLease | None = None
        retained_predecessor: _OwnedFileLease | None = None
        try:
            _windows_replace_with_backup(
                self.target,
                rollback_backup,
                correction_backup,
            )
            self.parent.flush()
            competitor = _OwnedFileLease(
                self.target,
                displaced_identity,
                deletable=False,
                parent_lease=self.parent,
            )
            retained_predecessor = _OwnedFileLease(
                correction_backup,
                predecessor_identity,
                deletable=False,
                parent_lease=self.parent,
            )
        except (OSError, ValueError):
            if competitor is not None:
                competitor.close()
            if retained_predecessor is not None:
                retained_predecessor.close()
            return False
        if competitor is None or retained_predecessor is None:  # pragma: no cover
            return False
        try:
            if (
                competitor.read(max_bytes=_MAX_EVIDENCE_BYTES) != displaced_snapshot
                or retained_predecessor.read(max_bytes=_MAX_EVIDENCE_BYTES) != predecessor_snapshot
            ):
                raise ValueError("fallback rollback correction bytes changed")
            competitor.assert_visible_binding()
            retained_predecessor.assert_visible_binding()
        finally:
            competitor.close()
            retained_predecessor.close()
        self.state = "ownership-lost"
        return False

    def rollback_owned(self) -> bool:
        """Restore the exact displaced predecessor while both names remain owned."""

        if self.state == "rolled-back":
            return True
        if self.state != "active":
            return False

        try:
            self.verify()
        except (OSError, ValueError):
            if os.name == "nt":
                return self._rollback_windows_without_backup_lease()
            return False
        if self.previous is None:  # pragma: no cover - established by verify
            return False
        previous = self.previous
        new_lease = self.new_lease
        backup_lease = self.backup_lease
        if new_lease is None or backup_lease is None:  # pragma: no cover - internal invariant
            return False
        if os.name == "nt":
            # ReplaceFileW cannot operate while either proof handle is open.
            # It atomically retains the displaced publisher bytes in a fresh
            # backup, so the predecessor is never restored by delete+rename.
            previous_identity = backup_lease.identity
            replacement_identity = new_lease.identity
            new_lease.close()
            backup_lease.close()
            rollback_backup = self.target.with_name(
                f".{self.target.name}.{uuid.uuid4().hex}.rolled-back"
            )
            current: _OwnedFileLease | None = None
            displaced: _OwnedFileLease | None = None
            try:
                _windows_replace_with_backup(self.target, self.backup, rollback_backup)
                self.parent.flush()
                current_identity = _path_identity(self.target)
                displaced_identity = _path_identity(rollback_backup)
                current = _OwnedFileLease(
                    self.target,
                    current_identity,
                    deletable=False,
                    parent_lease=self.parent,
                )
                displaced = _OwnedFileLease(
                    rollback_backup,
                    displaced_identity,
                    deletable=False,
                    parent_lease=self.parent,
                )
            except (OSError, ValueError):
                if current is not None:
                    current.close()
                if displaced is not None:
                    displaced.close()
                return False
            if current is None or displaced is None:  # pragma: no cover - internal invariant
                return False
            try:
                current_snapshot = current.read(max_bytes=_MAX_EVIDENCE_BYTES)
                displaced_snapshot = displaced.read(max_bytes=_MAX_EVIDENCE_BYTES)
                current.assert_visible_binding()
                displaced.assert_visible_binding()
                predecessor_restored = (
                    current.identity == previous_identity and current_snapshot == previous
                )
                publisher_displaced = (
                    displaced.identity == replacement_identity
                    and displaced_snapshot == self.replacement
                )
            finally:
                current.close()
                displaced.close()
            if not predecessor_restored:
                self.state = "ownership-lost"
                return False
            if not publisher_displaced:
                # A competitor replaced the visible control after verify and
                # before reverse ReplaceFileW.  The first reverse retained it
                # in rollback_backup; atomically exchange it back into view.
                correction_backup = self.target.with_name(
                    f".{self.target.name}.{uuid.uuid4().hex}.rollback-correction"
                )
                competitor: _OwnedFileLease | None = None
                retained_previous: _OwnedFileLease | None = None
                try:
                    _windows_replace_with_backup(
                        self.target,
                        rollback_backup,
                        correction_backup,
                    )
                    self.parent.flush()
                    competitor = _OwnedFileLease(
                        self.target,
                        displaced_identity,
                        deletable=False,
                        parent_lease=self.parent,
                    )
                    retained_previous = _OwnedFileLease(
                        correction_backup,
                        previous_identity,
                        deletable=False,
                        parent_lease=self.parent,
                    )
                except (OSError, ValueError):
                    if competitor is not None:
                        competitor.close()
                    if retained_previous is not None:
                        retained_previous.close()
                    return False
                if competitor is None or retained_previous is None:  # pragma: no cover
                    return False
                try:
                    if (
                        competitor.read(max_bytes=_MAX_EVIDENCE_BYTES) != displaced_snapshot
                        or retained_previous.read(max_bytes=_MAX_EVIDENCE_BYTES) != previous
                    ):
                        raise ValueError("rollback correction bytes changed")
                    competitor.assert_visible_binding()
                    retained_previous.assert_visible_binding()
                finally:
                    competitor.close()
                    retained_previous.close()
                self.state = "ownership-lost"
                return False
        else:
            _rename_exchange(self.parent, self.target.name, self.backup.name)
            self.parent.flush()
            new_lease.close()
            backup_lease.close()
        self.state = "rolled-back"
        restored = _OwnedFileLease(
            self.target,
            _path_identity(self.target),
            deletable=False,
            parent_lease=self.parent,
        )
        try:
            return restored.read(max_bytes=_MAX_EVIDENCE_BYTES) == previous
        finally:
            restored.close()

    def commit(self) -> None:
        # The prior control is content-addressed by neither the registry nor the
        # active ServingControl name.  Preserve its random-name backup on every
        # platform: deleting it by pathname would reopen the same substitution
        # race this lease is designed to avoid, while retaining it is harmless
        # and gives operators exact recovery evidence.
        if self.new_lease is not None:
            self.new_lease.close()
        if self.backup_lease is not None:
            self.backup_lease.close()
        self.state = "committed"

    def close(self) -> None:
        if self.new_lease is not None:
            self.new_lease.close()
        if self.backup_lease is not None:
            self.backup_lease.close()


def _swap_control_exact(
    parent: _PublicationParentLease,
    target: Path,
    expected: bytes,
    replacement: bytes,
) -> _OwnedControlSwap:
    if target.parent != parent.parent:
        raise ValueError("serving control is outside its held publication parent")
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.swap")
    backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.previous")
    descriptor = parent.open_file(temporary.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(replacement)
        stream.flush()
        os.fsync(stream.fileno())
    temporary_identity = _path_identity(temporary)
    parent.flush()
    _CONTROL_SWAP_SEAM("before-final-read", target)
    current_identity = _path_identity(target)
    current = _OwnedFileLease(
        target,
        current_identity,
        deletable=False,
        parent_lease=parent,
    )
    try:
        if current.read(max_bytes=_MAX_EVIDENCE_BYTES) != expected:
            raise ValueError("serving control changed during exact CAS")
    finally:
        # Windows ReplaceFileW needs delete sharing on the replaced name.  The
        # atomic backup is re-opened and checked against this exact identity.
        current.close()
    if os.name == "nt":
        _windows_replace_with_backup(target, temporary, backup)
    else:
        backup = temporary
        _rename_exchange(parent, target.name, temporary.name)
    return _OwnedControlSwap(
        parent=parent,
        target=target,
        backup=backup,
        expected=expected,
        replacement=replacement,
        new_identity=temporary_identity,
    )


def _held_control_snapshot(
    parent: _PublicationParentLease,
    target: Path,
) -> bytes:
    """Read one exact visible control inode through its held publication parent."""

    identity = _path_identity(target)
    lease = _OwnedFileLease(
        target,
        identity,
        deletable=False,
        parent_lease=parent,
    )
    try:
        lease.assert_visible_binding()
        snapshot = lease.read(max_bytes=_MAX_EVIDENCE_BYTES)
        lease.assert_visible_binding()
        return snapshot
    finally:
        lease.close()


def _accept_exact_captured_control(
    serving_control: ServingControlStore,
    captured: CapturedServingControl,
    parent: _PublicationParentLease,
) -> None:
    """Accept high-water only while the exact captured control remains held."""

    identity = _path_identity(serving_control.path)
    lease = _OwnedFileLease(
        serving_control.path,
        identity,
        deletable=False,
        parent_lease=parent,
    )
    try:
        lease.assert_visible_binding()
        if lease.read(max_bytes=_MAX_EVIDENCE_BYTES) != captured.control_snapshot:
            raise ValueError("serving control changed before high-water acceptance")
        serving_control.accept(captured)
        lease.assert_visible_binding()
        if lease.read(max_bytes=_MAX_EVIDENCE_BYTES) != captured.control_snapshot:
            raise ValueError("serving control changed during high-water acceptance")
    finally:
        lease.close()


def _verified_fixture_inventory(
    fixture_root: Path, fixtures: tuple[ServingSmokeFixture, ...]
) -> tuple[_SmokeFixtureInventory, tuple[tuple[ArtifactEntry, bytes], ...]]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke-inventory/1.0",
        "fixtures": [item.model_dump(mode="python", by_alias=True) for item in fixtures],
    }
    payload["inventoryHash"] = canonical_sha256(payload["fixtures"])
    inventory = _SmokeFixtureInventory.model_validate(payload)
    root = secure_existing_root(fixture_root)
    materialized = tuple(
        (fixture.artifact, _materialize_fixture(root, fixture)[0]) for fixture in inventory.fixtures
    )
    return inventory, materialized


def promote_serving_ready(
    *,
    accepted: AcceptedCandidate,
    stage: CandidateStage,
    verified_smoke: VerifiedServingSmoke,
    stage_root: Path,
    fixture_root: Path,
    fixtures: tuple[ServingSmokeFixture, ...],
    serving_control: ServingControlStore,
    failure_injector: Callable[[str], None] | None = None,
) -> PromotionReceipt:
    """Publish immutable artifacts first and atomically replace ServingControl last."""

    if type(accepted) is not AcceptedCandidate or type(stage) is not CandidateStage:
        raise TypeError("live promotion requires exact accepted candidate evidence")
    if type(serving_control) is not ServingControlStore:
        raise TypeError("live promotion requires an exact ServingControlStore")
    if type(verified_smoke) is not VerifiedServingSmoke:
        raise TypeError("self-hashed smoke reports cannot authorize live promotion")
    verified_smoke.verify()
    smoke = verified_smoke.report
    if (
        accepted.candidate_stage_hash != stage.stage_hash
        or accepted.acceptance_hash != stage.acceptance_hash
        or accepted.serving_checkpoint_sha256 != stage.serving_checkpoint_sha256
        or accepted.serving_model_hash != stage.serving_model.model_version_hash
        or smoke.accepted_candidate_hash != accepted.accepted_hash
        or smoke.acceptance_hash != accepted.acceptance_hash
        or smoke.serving_model_version_id != accepted.serving_model_version_id
        or smoke.serving_model_hash != accepted.serving_model_hash
        or smoke.source_checkpoint_sha256 != accepted.source_checkpoint_sha256
        or smoke.serving_checkpoint_sha256 != accepted.serving_checkpoint_sha256
        or smoke.task_binding_inventory_hash != accepted.task_binding_inventory_hash
        or not smoke.succeeded
    ):
        raise ValueError("accepted candidate and fresh serving smoke differ")
    reopened = _reopen_stage(stage, stage_root)
    inventory, fixture_bytes = _verified_fixture_inventory(fixture_root, fixtures)
    if inventory.inventory_hash != verified_smoke.fixture_inventory_hash:
        raise ValueError("promotion fixture inventory differs from fresh-process smoke")
    for fixture, result in zip(inventory.fixtures, smoke.task_results, strict=True):
        binding = reopened.serving_model.task_head(fixture.task_id).calibration(fixture.entity_type)
        if (
            result.fixture_artifact_hash != fixture.artifact.artifact_hash
            or result.fixture_bundle_sha256 != fixture.artifact.bundle_sha256
            or result.graph_version_hash != fixture.artifact.graph_version_hash
            or result.feature_contract_hash != fixture.artifact.feature_contract_hash
            or result.adapter_domain != binding.adapter_domain
            or result.adapter_schema_hash != binding.adapter_schema_hash
            or result.adapter_state_hash != binding.adapter_state_hash
            or result.confidence_artifact_hash != binding.calibration_artifact_hash
            or result.confidence_protocol_hash != binding.calibration_protocol_hash
        ):
            raise ValueError("fresh-process smoke result differs from promotion fixture/model")
    inject = _PROMOTION_FAILURE_SEAM if failure_injector is None else failure_injector
    control_path = serving_control.path
    control_root = serving_control.control_root
    control_parent = _PublicationParentLease(
        control_root,
        control_path.parent,
        create=False,
    )
    try:
        promotion_lock = _PublisherLock(
            control_parent,
            f".{control_path.name}.promotion.lock",
            active_message="another serving promotion is active",
        )
    except Exception:
        control_parent.close()
        raise
    publication: _OwnedControlSwap | None = None
    high_water_accept_started = False
    try:
        original = serving_control.capture()
        if not (
            original.document.generation
            == original.registry_document.generation
            == original.catalog_document.generation
        ):
            raise ValueError("live serving generations are inconsistent")
        control_parent.assert_confined()
        if _held_control_snapshot(control_parent, control_path) != original.control_snapshot:
            raise ValueError("serving control changed before promotion")
        recovered = _recover_published_promotion(
            accepted=accepted,
            smoke=smoke,
            stage=stage,
            reopened=reopened,
            fixtures=fixture_bytes,
            captured=original,
            control_root=control_root,
        )
        high_water = _load_high_water(serving_control)
        if recovered is not None:
            _accept_exact_captured_control(serving_control, original, control_parent)
            return recovered
        if high_water is None:
            if original.document.generation != 0:
                raise ValueError("live serving control has an incomplete unaccepted promotion")
            _accept_exact_captured_control(serving_control, original, control_parent)
        elif not _high_water_matches(original, high_water):
            raise ValueError("live serving control has an incomplete unaccepted promotion")
        try:
            # Publish exact accepted checkpoint, manifest, and confidence bytes.
            stage_artifacts = (
                stage.artifact("serving-checkpoint"),
                stage.artifact("serving-manifest"),
                *(
                    stage.artifact("confidence", task_id=task, entity_type=entity)
                    for task, entity in _TASK_ENTITY_ORDER
                ),
            )
            for index, artifact in enumerate(stage_artifacts):
                snapshot = _read_ref(
                    secure_existing_root(stage_root),
                    artifact.relative_path,
                    max_bytes=_MAX_CHECKPOINT_BYTES,
                )
                _atomic_immutable(
                    control_root,
                    control_root.joinpath(*PurePosixPath(artifact.relative_path).parts),
                    snapshot,
                )
                inject(f"artifact-{index}")
            published_entries: list[ArtifactEntry] = []
            for index, (entry, snapshot) in enumerate(fixture_bytes):
                relative = f"artifacts/fixture-{entry.bundle_sha256}.json"
                _atomic_immutable(
                    control_root, control_root.joinpath(*PurePosixPath(relative).parts), snapshot
                )
                published_entries.append(entry.model_copy(update={"relative_path": relative}))
                inject(f"fixture-{index}")
            generation = original.document.generation + 1
            live_model = reopened.serving_model.model_copy(update={"state": "servingReady"})
            existing_models = [
                model.model_copy(update={"state": "accepted"})
                if model.state == "servingReady"
                else model
                for model in original.registry_document.models
                if model.model_version_id != live_model.model_version_id
            ]
            registry = RegistryDocument(
                schemaVersion="socialgraph-fm.core-serving-registry/2.0",
                generation=generation,
                models=tuple(
                    sorted((*existing_models, live_model), key=lambda item: item.model_version_id)
                ),
            )
            if len([item for item in registry.models if item.state == "servingReady"]) != 1:
                raise ValueError("promotion must produce exactly one servingReady model")
            existing_entries = {
                item.artifact_id: item for item in original.catalog_document.artifacts
            }
            for entry in published_entries:
                previous = existing_entries.get(entry.artifact_id)
                if previous is not None and (
                    previous.artifact_hash != entry.artifact_hash
                    or previous.bundle_sha256 != entry.bundle_sha256
                    or previous.graph_version_hash != entry.graph_version_hash
                ):
                    raise ValueError("promotion fixture conflicts with live artifact identity")
                existing_entries[entry.artifact_id] = entry
            catalog = ArtifactCatalogDocument(
                schemaVersion="socialgraph-fm.core-serving-graph-catalog/1.0",
                generation=generation,
                artifacts=tuple(existing_entries[key] for key in sorted(existing_entries)),
            )
            registry_snapshot = _canonical_bytes(registry)
            catalog_snapshot = _canonical_bytes(catalog)
            registry_sha = hashlib.sha256(registry_snapshot).hexdigest()
            catalog_sha = hashlib.sha256(catalog_snapshot).hexdigest()
            registry_relative = f"versions/registry-g{generation}-{registry_sha}.json"
            catalog_relative = f"versions/catalog-g{generation}-{catalog_sha}.json"
            _atomic_immutable(
                control_root,
                control_root.joinpath(*PurePosixPath(registry_relative).parts),
                registry_snapshot,
            )
            inject("registry")
            _atomic_immutable(
                control_root,
                control_root.joinpath(*PurePosixPath(catalog_relative).parts),
                catalog_snapshot,
            )
            inject("catalog")
            control_payload: dict[str, Any] = {
                "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
                "generation": generation,
                "registry": {
                    "relativePath": registry_relative,
                    "sha256": registry_sha,
                    "semanticHash": canonical_sha256(
                        registry.model_dump(mode="python", by_alias=True)
                    ),
                    "generation": generation,
                },
                "catalog": {
                    "relativePath": catalog_relative,
                    "sha256": catalog_sha,
                    "semanticHash": canonical_sha256(
                        catalog.model_dump(mode="python", by_alias=True)
                    ),
                    "generation": generation,
                },
            }
            control_payload["controlHash"] = canonical_sha256(control_payload)
            control_document = ServingControlDocument.model_validate(control_payload)
            new_control_snapshot = _canonical_bytes(control_document)
            receipt = _promotion_receipt(
                accepted=accepted,
                smoke=smoke,
                control=control_document,
            )
            _atomic_immutable(
                control_root,
                control_root.joinpath(
                    *PurePosixPath(_promotion_receipt_relative(control_document)).parts
                ),
                _canonical_bytes(receipt),
            )
            inject("receipt")
            inject("before-control-cas")
            control_parent.assert_confined()
            if _held_control_snapshot(control_parent, control_path) != original.control_snapshot:
                raise ValueError("serving control changed during promotion")
            publication = _swap_control_exact(
                control_parent,
                control_path,
                original.control_snapshot,
                new_control_snapshot,
            )
            publication.acquire_leases()
            control_parent.flush()
            _CONTROL_SWAP_SEAM("after-atomic-swap", control_path)
            publication.verify()
            if not publication.expected_matched:
                if not publication.rollback_owned():
                    raise RuntimeError(
                        "serving control changed during exact CAS and its exact predecessor "
                        "could not be restored"
                    )
                raise ValueError("serving control changed during exact CAS")
            inject("after-control-replace")
            publication.verify()
            if _held_control_snapshot(control_parent, control_path) != new_control_snapshot:
                raise ValueError("promoted serving control failed final reread")
            winning = serving_control.capture()
            if (
                winning.document != control_document
                or winning.registry_document != registry
                or winning.catalog_document != catalog
            ):
                raise ValueError("promoted serving control reopened to different bytes")
            live_registry_path = control_root.joinpath(
                *PurePosixPath(winning.document.registry.relative_path).parts
            )
            from .serving_registry import ServingRegistry

            live_registry = ServingRegistry.load(live_registry_path, runtime_root=control_root)
            capabilities = live_registry.capabilities(registry_snapshot=winning.registry_snapshot)
            ready = [
                item for item in winning.registry_document.models if item.state == "servingReady"
            ]
            if (
                len(ready) != 1
                or ready[0].model_version_hash != accepted.serving_model_hash
                or capabilities.get("servingReady") is not True
            ):
                raise ValueError("promoted registry did not reload one accepted serving model")
            publication.verify()
            inject("before-high-water-accept")
            high_water_accept_started = True
            serving_control.accept(winning)
            publication.commit()
            return receipt
        except Exception as error:
            restored = False
            if publication is not None:
                if high_water_accept_started:
                    try:
                        publication.commit()
                    except Exception as cleanup_error:
                        error.add_note(
                            f"commit-uncertain serving-control lease close failed: {cleanup_error}"
                        )
                    error.add_note(
                        "serving control was retained because high-water acceptance outcome "
                        "is uncertain"
                    )
                else:
                    try:
                        restored = publication.rollback_owned()
                    except Exception as cleanup_error:
                        error.add_note(
                            f"owned serving-control rollback failed closed: {cleanup_error}"
                        )
                    if not restored:
                        error.add_note(
                            "serving control was preserved because the visible name no longer "
                            "proved publisher ownership"
                        )
            if (
                _read_ref(
                    control_root,
                    original.document.registry.relative_path,
                    max_bytes=_MAX_EVIDENCE_BYTES,
                )
                != original.registry_snapshot
                or _read_ref(
                    control_root,
                    original.document.catalog.relative_path,
                    max_bytes=_MAX_EVIDENCE_BYTES,
                )
                != original.catalog_snapshot
            ):
                raise RuntimeError("failed promotion changed original live bytes")
            restored_snapshot = (
                publication.previous
                if publication is not None and publication.previous is not None
                else original.control_snapshot
            )
            if restored and (
                _held_control_snapshot(control_parent, control_path) != restored_snapshot
            ):
                raise RuntimeError("owned rollback did not restore the exact displaced control")
            raise
    finally:
        try:
            if publication is not None:
                publication.close()
        finally:
            try:
                promotion_lock.close()
            finally:
                control_parent.close()


__all__ = [
    "AcceptanceDerivationInputs",
    "AcceptedCandidate",
    "CandidateStage",
    "CandidateServingDefinition",
    "PromotionReceipt",
    "ServingSmokeReport",
    "ServingSmokeFixture",
    "ServingSmokeTaskResult",
    "VerifiedServingSmoke",
    "accept_candidate",
    "promote_serving_ready",
    "run_fresh_process_serving_smoke",
    "stage_candidate",
]


if __name__ == "__main__":  # pragma: no cover - exercised through a fresh process
    raise SystemExit(_fresh_smoke_main(sys.argv[1:]))
