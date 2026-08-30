"""Validated serving registry: the sole source of core readiness."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .adapters import AdapterParameterModule, AdapterSchema
from .checkpoint import CheckpointBindings, load_checkpoint
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root
from .model import CoreGFM


_HASH = r"^[0-9a-f]{64}$"
MAX_CHECKPOINT_BYTES = 1024 * 1024 * 1024
TaskId = Literal[
    "core.community_resilience_review",
    "core.risk_and_trust_review",
    "core.collaboration_completion",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True, frozen=True)


class CheckpointBindingsRecord(_StrictModel):
    config_hash: str = Field(alias="configHash", pattern=_HASH)
    data_hash: str = Field(alias="dataHash", pattern=_HASH)
    code_hash: str = Field(alias="codeHash", pattern=_HASH)
    environment_hash: str = Field(alias="environmentHash", pattern=_HASH)

    def as_checkpoint_bindings(self) -> CheckpointBindings:
        return CheckpointBindings(
            config_hash=self.config_hash,
            data_hash=self.data_hash,
            code_hash=self.code_hash,
            environment_hash=self.environment_hash,
        )


class ServingCheckpoint(_StrictModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    sha256: str = Field(pattern=_HASH)
    serving_manifest_relative_path: str = Field(
        alias="servingManifestRelativePath", min_length=1, max_length=500
    )
    serving_manifest_sha256: str = Field(alias="servingManifestSha256", pattern=_HASH)
    bindings: CheckpointBindingsRecord
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    node_classes: int = Field(alias="nodeClasses", ge=1, le=100_000)
    multi_hot_buckets: int = Field(alias="multiHotBuckets", ge=1, le=65_536)

    @field_validator("relative_path", "serving_manifest_relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value.replace("\\", "/"))
        if parsed.is_absolute() or ".." in parsed.parts or ":" in value:
            raise ValueError("checkpoint path must be a safe relative path")
        return parsed.as_posix()


class CalibrationBinding(_StrictModel):
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(alias="entityType")
    confidence_kind: Literal["binary-calibration", "regression-interval"] = Field(
        alias="confidenceKind"
    )
    calibration_version: str = Field(alias="calibrationVersion", min_length=1, max_length=300)
    calibration_method: Literal["sigmoid", "validation-residual-interval"] = Field(
        alias="calibrationMethod"
    )
    calibration_artifact_hash: str = Field(alias="calibrationArtifactHash", pattern=_HASH)
    calibration_relative_path: str = Field(
        alias="calibrationRelativePath", min_length=1, max_length=500
    )
    calibration_sha256: str = Field(alias="calibrationSha256", pattern=_HASH)
    calibration_protocol_hash: str = Field(alias="calibrationProtocolHash", pattern=_HASH)
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH)
    graph_feature_contract_hash: str = Field(alias="graphFeatureContractHash", pattern=_HASH)

    @field_validator("calibration_relative_path")
    @classmethod
    def validate_calibration_path(cls, value: str) -> str:
        parsed = PurePosixPath(value.replace("\\", "/"))
        if parsed.is_absolute() or ".." in parsed.parts or ":" in value:
            raise ValueError("calibration path must be safe and relative")
        return parsed.as_posix()

    @model_validator(mode="after")
    def validate_confidence_kind(self):
        regression = self.entity_type == "community"
        expected = (
            ("regression-interval", "validation-residual-interval")
            if regression
            else ("binary-calibration", "sigmoid")
        )
        if (self.confidence_kind, self.calibration_method) != expected:
            raise ValueError(
                "community outputs require regression intervals and other outputs require "
                "binary calibration"
            )
        return self


class ServingTaskHead(_StrictModel):
    task_id: TaskId = Field(alias="taskId")
    kind: Literal["community-resilience", "risk-and-trust", "collaboration-completion"]
    node_output_index: int | None = Field(default=None, alias="nodeOutputIndex", ge=0)
    calibrations: tuple[CalibrationBinding, ...] = Field(min_length=1, strict=False)

    @model_validator(mode="after")
    def validate_task_kind(self):
        expected = {
            "core.community_resilience_review": "community-resilience",
            "core.risk_and_trust_review": "risk-and-trust",
            "core.collaboration_completion": "collaboration-completion",
        }
        if self.kind != expected[self.task_id]:
            raise ValueError("task head kind does not match taskId")
        if (self.task_id == "core.risk_and_trust_review") != (
            self.node_output_index is not None
        ):
            raise ValueError("only the risk task head requires nodeOutputIndex")
        if self.task_id == "core.risk_and_trust_review" and self.node_output_index != 1:
            raise ValueError("risk task nodeOutputIndex must bind positive class 1")
        expected_entities = {
            "core.community_resilience_review": ("community",),
            "core.risk_and_trust_review": ("node", "edge"),
            "core.collaboration_completion": ("node-pair",),
        }[self.task_id]
        entities = tuple(binding.entity_type for binding in self.calibrations)
        if entities != expected_entities:
            raise ValueError("task head calibrations must bind every output entity exactly")
        return self

    def calibration(self, entity_type: str) -> CalibrationBinding:
        binding = next(
            (item for item in self.calibrations if item.entity_type == entity_type), None
        )
        if binding is None:
            raise LookupError("calibration is unavailable for task output entity")
        return binding


class ScoreCalibration(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-score-calibration/2.0"] = Field(alias="schemaVersion")
    calibration_version: str = Field(alias="calibrationVersion", min_length=1, max_length=300)
    method: Literal["sigmoid"]
    temperature: float = Field(gt=0.0, le=1_000_000.0)
    bias: float = Field(ge=-1_000_000.0, le=1_000_000.0)
    protocol_hash: str = Field(alias="protocolHash", pattern=_HASH)
    artifact_hash: str = Field(alias="artifactHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_artifact_hash(self):
        if self.artifact_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"artifact_hash"})
        ):
            raise ValueError("calibration artifact hash mismatch")
        return self


class RegressionConfidenceArtifact(_StrictModel):
    """Validation-derived residual interval; never represented as a probability."""

    schema_version: Literal["socialgraph-fm.core-regression-confidence-artifact/1.0"] = Field(
        alias="schemaVersion"
    )
    confidence_version: str = Field(alias="confidenceVersion", min_length=1, max_length=300)
    method: Literal["validation-residual-interval"]
    coverage: float = Field(gt=0.0, lt=1.0)
    residual_quantile: float = Field(alias="residualQuantile", ge=0.0)
    validation_count: int = Field(alias="validationCount", ge=2)
    validation_head_report_hash: str = Field(alias="validationHeadReportHash", pattern=_HASH)
    validation_partition_hash: str = Field(alias="validationPartitionHash", pattern=_HASH)
    validation_prediction_hash: str = Field(alias="validationPredictionHash", pattern=_HASH)
    validation_target_hash: str = Field(alias="validationTargetHash", pattern=_HASH)
    absolute_residuals: tuple[float, ...] = Field(
        alias="absoluteResiduals", min_length=2, strict=False
    )
    protocol_hash: str = Field(alias="protocolHash", pattern=_HASH)
    artifact_hash: str = Field(alias="artifactHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_artifact_hash(self):
        if (
            self.validation_count != len(self.absolute_residuals)
            or self.absolute_residuals != tuple(sorted(self.absolute_residuals))
            or any(not math.isfinite(value) or value < 0.0 for value in self.absolute_residuals)
        ):
            raise ValueError("regression confidence requires sorted finite validation residuals")
        quantile_index = min(
            self.validation_count,
            math.ceil((self.validation_count + 1) * self.coverage),
        )
        if self.residual_quantile != self.absolute_residuals[quantile_index - 1]:
            raise ValueError(
                "residualQuantile is not derived from validation residuals and coverage"
            )
        expected_protocol_hash = canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.core-regression-confidence-protocol/1.0",
                "method": self.method,
                "coverage": self.coverage,
                "validationCount": self.validation_count,
                "validationHeadReportHash": self.validation_head_report_hash,
                "validationPartitionHash": self.validation_partition_hash,
                "validationPredictionHash": self.validation_prediction_hash,
                "validationTargetHash": self.validation_target_hash,
            }
        )
        if self.protocol_hash != expected_protocol_hash:
            raise ValueError("regression confidence protocol is not validation-derived")
        if self.artifact_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"artifact_hash"})
        ):
            raise ValueError("regression confidence artifact hash mismatch")
        return self


ConfidenceArtifact: TypeAlias = ScoreCalibration | RegressionConfidenceArtifact


class ServingAdapterBinding(_StrictModel):
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH)
    multi_hot_buckets: int = Field(alias="multiHotBuckets", ge=1, le=65_536)


class ServingCheckpointManifest(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-checkpoint-manifest/1.1"] = Field(
        alias="schemaVersion"
    )
    task4_checkpoint_sha256: str = Field(alias="task4CheckpointSha256", pattern=_HASH)
    accepted: Literal[True]
    promotable: Literal[True]
    model_state_hash: str = Field(alias="modelStateHash", pattern=_HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH)
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    node_classes: int = Field(alias="nodeClasses", ge=1, le=100_000)
    multi_hot_buckets: int = Field(alias="multiHotBuckets", ge=1, le=65_536)
    adapter_bindings: tuple[ServingAdapterBinding, ...] = Field(
        alias="adapterBindings", min_length=1, strict=False
    )
    task_heads: tuple[ServingTaskHead, ...] = Field(alias="taskHeads", min_length=1, strict=False)

    @model_validator(mode="after")
    def validate_adapter_inventory(self):
        domains = tuple(item.adapter_domain for item in self.adapter_bindings)
        if domains != tuple(sorted(set(domains))):
            raise ValueError("serving adapter bindings must be unique and sorted")
        primary = self.adapter_bindings[0]
        if (
            self.adapter_domain != primary.adapter_domain
            or self.adapter_schema_hash != primary.adapter_schema_hash
            or self.adapter_state_hash != primary.adapter_state_hash
            or self.multi_hot_buckets != primary.multi_hot_buckets
        ):
            raise ValueError("serving manifest primary adapter differs from its inventory")
        by_domain = {item.adapter_domain: item for item in self.adapter_bindings}
        for head in self.task_heads:
            for entity in head.calibrations:
                adapter = by_domain.get(entity.adapter_domain)
                if (
                    adapter is None
                    or adapter.adapter_schema_hash != entity.adapter_schema_hash
                    or adapter.adapter_state_hash != entity.adapter_state_hash
                ):
                    raise ValueError("task/entity adapter binding differs from manifest inventory")
        if set(by_domain) != {
            entity.adapter_domain for head in self.task_heads for entity in head.calibrations
        }:
            raise ValueError("manifest adapter inventory must be used by a task/entity binding")
        return self


class ServingModel(_StrictModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH)
    state: Literal["accepted", "servingReady"]
    checkpoint: ServingCheckpoint
    task_heads: tuple[ServingTaskHead, ...] = Field(alias="taskHeads", min_length=1, strict=False)
    tasks: tuple[TaskId, ...] = Field(min_length=1, strict=False)
    graph_schema_versions: tuple[Literal["socialgraph-fm.core-graph-bundle/2.0"], ...] = Field(
        alias="graphSchemaVersions", min_length=1, strict=False
    )
    graph_feature_contract_hash: str = Field(alias="graphFeatureContractHash", pattern=_HASH)
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)

    @model_validator(mode="after")
    def validate_model_binding(self):
        head_tasks = tuple(head.task_id for head in self.task_heads)
        if len(set(head_tasks)) != len(head_tasks) or set(head_tasks) != set(self.tasks):
            raise ValueError("taskHeads must match tasks exactly")
        feature_inventory = [
            {
                "taskId": head.task_id,
                "entityType": binding.entity_type,
                "featureContractHash": binding.graph_feature_contract_hash,
            }
            for head in self.task_heads
            for binding in head.calibrations
        ]
        if self.graph_feature_contract_hash != canonical_sha256(feature_inventory):
            raise ValueError(
                "graphFeatureContractHash must bind the ordered task/entity feature inventory"
            )
        payload = self.model_dump(
            mode="python", by_alias=True, exclude={"model_version_hash", "state"}
        )
        if self.model_version_hash != canonical_sha256(payload):
            raise ValueError("modelVersionHash does not bind the serving model definition")
        return self

    def task_head(self, task_id: str) -> ServingTaskHead:
        head = next((item for item in self.task_heads if item.task_id == task_id), None)
        if head is None:
            raise LookupError("registered task head is unavailable")
        return head


class RegistryDocument(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-registry/2.0"] = Field(
        alias="schemaVersion"
    )
    generation: int = Field(ge=0)
    models: tuple[ServingModel, ...] = Field(strict=False)


@dataclass(frozen=True)
class VerifiedCheckpoint:
    sha256: str
    payload: dict[str, Any]
    snapshot: bytes


@dataclass(frozen=True)
class CapturedModelLease:
    """Hash-bound serving bytes captured under one stable registry generation."""

    registry_snapshot: bytes
    registry_source_sha256: str
    registry_hash: str
    registry_generation: int
    model_version_id: str
    task_id: str
    manifest_snapshot: bytes
    checkpoint_snapshot: bytes
    calibration_snapshots: tuple[tuple[str, bytes], ...]

    def materialize(
        self,
    ) -> tuple[ServingModel, VerifiedCheckpoint, dict[str, ConfidenceArtifact], str]:
        if hashlib.sha256(self.registry_snapshot).hexdigest() != self.registry_source_sha256:
            raise ValueError("captured registry bytes changed")
        document = RegistryDocument.model_validate_json(self.registry_snapshot)
        if document.generation != self.registry_generation:
            raise ValueError("captured registry generation mismatch")
        if (
            canonical_sha256(document.model_dump(mode="python", by_alias=True))
            != self.registry_hash
        ):
            raise ValueError("captured registry semantic hash mismatch")
        model = next(
            (item for item in document.models if item.model_version_id == self.model_version_id),
            None,
        )
        if model is None or model.state != "servingReady":
            raise ValueError("captured model is not serving ready")
        manifest, verified = _validate_captured_checkpoint(
            model, self.manifest_snapshot, self.checkpoint_snapshot
        )
        calibrations = _validate_captured_calibrations(
            model, self.task_id, dict(self.calibration_snapshots)
        )
        return model, verified, calibrations, manifest.adapter_schema_hash


def _tensor_state_hash(state: dict[str, Any]) -> str:
    records: list[dict[str, object]] = []
    for name, value in sorted(state.items()):
        if not hasattr(value, "detach"):
            raise ValueError("serving state dictionaries may contain only tensors")
        records.append({"name": name, **canonical_tensor_digest(value)})
    return canonical_sha256(records)


def _validate_captured_checkpoint(
    model: ServingModel, manifest_bytes: bytes, checkpoint_bytes: bytes
) -> tuple[ServingCheckpointManifest, VerifiedCheckpoint]:
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    if not hmac.compare_digest(manifest_hash, model.checkpoint.serving_manifest_sha256):
        raise ValueError("serving checkpoint manifest hash does not match registry")
    manifest = ServingCheckpointManifest.model_validate_json(manifest_bytes)
    if (
        manifest.task4_checkpoint_sha256 != model.checkpoint.sha256
        or manifest.adapter_domain != model.checkpoint.adapter_domain
        or manifest.node_classes != model.checkpoint.node_classes
        or manifest.multi_hot_buckets != model.checkpoint.multi_hot_buckets
        or manifest.task_heads != model.task_heads
    ):
        raise ValueError("serving checkpoint manifest does not match registry")
    observed = hashlib.sha256(checkpoint_bytes).hexdigest()
    if not hmac.compare_digest(observed, model.checkpoint.sha256):
        raise ValueError("checkpoint hash does not match registry")
    payload = load_checkpoint(
        checkpoint_bytes,
        expected_bindings=model.checkpoint.bindings.as_checkpoint_bindings(),
    )
    if payload.get("status") != "accepted" or payload.get("promotable") is not True:
        raise ValueError("serving checkpoint must be accepted and promotable")
    trainer = payload["trainer"]
    model_state = trainer.get("model")
    adapters = trainer.get("adapters")
    adapter_schemas = trainer.get("adapterSchemas")
    if (
        not isinstance(model_state, dict)
        or not isinstance(adapters, dict)
        or not isinstance(adapter_schemas, dict)
    ):
        raise ValueError("serving checkpoint requires model, adapter state, and fitted schema")
    if _tensor_state_hash(model_state) != manifest.model_state_hash:
        raise ValueError("serving checkpoint model state hash mismatch")
    if set(adapters) != set(adapter_schemas) or set(adapters) != {
        binding.adapter_domain for binding in manifest.adapter_bindings
    }:
        raise ValueError("serving checkpoint adapter inventory does not match manifest")
    for binding in manifest.adapter_bindings:
        try:
            adapter_schema = AdapterSchema.model_validate_json(
                json.dumps(
                    adapter_schemas[binding.adapter_domain],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        except (KeyError, ValueError) as error:
            raise ValueError("serving checkpoint fitted adapter schema is invalid") from error
        if adapter_schema.adapter_schema_hash != binding.adapter_schema_hash:
            raise ValueError("serving checkpoint adapter schema hash mismatch")
        adapter_state = adapters[binding.adapter_domain]
        if (
            not isinstance(adapter_state, dict)
            or _tensor_state_hash(adapter_state) != binding.adapter_state_hash
        ):
            raise ValueError("serving checkpoint adapter state hash mismatch")
        names = tuple(str(name) for name in adapter_state)
        if any(name.startswith("_field_") or "._field_" in name for name in names):
            raise ValueError("serving checkpoint adapter state contains legacy graph row tensors")
        try:
            adapter_module = AdapterParameterModule(adapter_schema)
            adapter_module.load_state_dict(adapter_state, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise ValueError(
                "serving checkpoint adapter state cannot be strictly loaded"
            ) from error
    try:
        serving_model = CoreGFM(node_classes=model.checkpoint.node_classes)
        serving_model.load_state_dict(model_state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("serving checkpoint model state cannot be strictly loaded") from error
    required = {
        "encoder.",
        "node_head.",
        "binary_link_head.",
        "signed_edge_head.",
        "resilience_head.",
    }
    names = tuple(str(name) for name in model_state)
    if any(not any(name.startswith(prefix) for name in names) for prefix in required):
        raise ValueError("serving checkpoint task head state is incomplete")
    node_weight = model_state.get("node_head.weight")
    if getattr(node_weight, "shape", (None,))[0] != model.checkpoint.node_classes:
        raise ValueError("serving checkpoint node class binding mismatch")
    if any(
        head.node_output_index is not None
        and head.node_output_index >= model.checkpoint.node_classes
        for head in model.task_heads
    ):
        raise ValueError("serving task head output index is outside node classes")
    return manifest, VerifiedCheckpoint(observed, payload, checkpoint_bytes)


def _validate_captured_calibrations(
    model: ServingModel, task_id: str, snapshots: dict[str, bytes]
) -> dict[str, ConfidenceArtifact]:
    expected = {binding.entity_type: binding for binding in model.task_head(task_id).calibrations}
    if set(snapshots) != set(expected):
        raise ValueError("captured calibrations do not match model task heads")
    result: dict[str, ConfidenceArtifact] = {}
    for entity_type, binding in expected.items():
        snapshot = snapshots[entity_type]
        observed = hashlib.sha256(snapshot).hexdigest()
        if not hmac.compare_digest(observed, binding.calibration_sha256):
            raise ValueError("calibration file hash does not match registry")
        artifact_type: type[ScoreCalibration] | type[RegressionConfidenceArtifact] = (
            ScoreCalibration
            if binding.confidence_kind == "binary-calibration"
            else RegressionConfidenceArtifact
        )
        calibration = artifact_type.model_validate_json(snapshot)
        version = (
            calibration.calibration_version
            if isinstance(calibration, ScoreCalibration)
            else calibration.confidence_version
        )
        if (
            version != binding.calibration_version
            or calibration.method != binding.calibration_method
            or calibration.artifact_hash != binding.calibration_artifact_hash
            or calibration.protocol_hash != binding.calibration_protocol_hash
        ):
            raise ValueError("loaded calibration does not match serving task head")
        result[entity_type] = calibration
    return result


class ServingRegistry:
    """A checked registry snapshot that refreshes atomically on file replacement."""

    def __init__(
        self,
        path: Path,
        runtime_root: Path,
        document: RegistryDocument,
        source_sha256: str,
    ) -> None:
        self.path = path
        self.runtime_root = runtime_root
        self._document = document
        self._source_sha256 = source_sha256
        self._lock = threading.RLock()
        self._validated: dict[tuple[str, str, str, int, int, int], VerifiedCheckpoint] = {}
        self._validate_unique(document)

    @classmethod
    def load(cls, path: str | Path, *, runtime_root: str | Path) -> ServingRegistry:
        root = secure_existing_root(runtime_root)
        registry_lexical = reject_link_components(path)
        if not registry_lexical.is_file():
            raise ValueError("registry must be an existing regular file")
        registry_path = registry_lexical.resolve(strict=True)
        try:
            registry_path.relative_to(root)
        except ValueError as error:
            raise ValueError("registry must be inside the authorized runtime root") from error
        relative = registry_path.relative_to(root).as_posix()
        snapshot = read_confined_snapshot(root, relative, max_bytes=16 * 1024 * 1024)
        document = RegistryDocument.model_validate_json(snapshot)
        return cls(registry_path, root, document, hashlib.sha256(snapshot).hexdigest())

    def _registry_snapshot(self) -> bytes:
        relative = self.path.relative_to(self.runtime_root).as_posix()
        return read_confined_snapshot(
            self.runtime_root,
            relative,
            max_bytes=16 * 1024 * 1024,
        )

    @staticmethod
    def _validate_unique(document: RegistryDocument) -> None:
        identifiers = [model.model_version_id for model in document.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("registry modelVersionId values must be unique")

    def refresh(self) -> None:
        snapshot = self._registry_snapshot()
        observed = hashlib.sha256(snapshot).hexdigest()
        if observed == self._source_sha256:
            return
        document = RegistryDocument.model_validate_json(snapshot)
        self._validate_unique(document)
        if document.generation < self._document.generation:
            raise ValueError("registry generation cannot move backwards")
        self._document = document
        self._source_sha256 = observed
        self._validated.clear()

    @property
    def registry_hash(self) -> str:
        self.refresh()
        return canonical_sha256(self._document.model_dump(mode="python", by_alias=True))

    @property
    def generation(self) -> int:
        self.refresh()
        return self._document.generation

    def checkpoint_snapshot(self, model: ServingModel) -> VerifiedCheckpoint:
        relative = model.checkpoint.relative_path
        manifest_bytes = read_confined_snapshot(
            self.runtime_root,
            model.checkpoint.serving_manifest_relative_path,
            max_bytes=1024 * 1024,
        )
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

        # Inspect every lexical component before reading even file metadata.
        path = reject_link_components(self.runtime_root.joinpath(*PurePosixPath(relative).parts))
        details = path.lstat()
        key = (
            model.model_version_hash,
            model.checkpoint.sha256,
            manifest_hash,
            details.st_dev,
            details.st_ino,
            details.st_mtime_ns,
        )
        cached = self._validated.get(key)
        if cached is not None and details.st_size == len(cached.snapshot):
            for head in model.task_heads:
                for binding in head.calibrations:
                    self.calibration(model, head.task_id, binding.entity_type)
            return cached
        snapshot = read_confined_snapshot(
            self.runtime_root, relative, max_bytes=MAX_CHECKPOINT_BYTES
        )
        _manifest, verified = _validate_captured_checkpoint(model, manifest_bytes, snapshot)
        for head in model.task_heads:
            for binding in head.calibrations:
                self.calibration(model, head.task_id, binding.entity_type)
        self._validated = {key: verified}
        return verified

    def calibration(
        self, model: ServingModel, task_id: str, entity_type: str
    ) -> ConfidenceArtifact:
        head = model.task_head(task_id)
        binding = head.calibration(entity_type)
        snapshot = read_confined_snapshot(
            self.runtime_root,
            binding.calibration_relative_path,
            max_bytes=1024 * 1024,
        )
        observed = hashlib.sha256(snapshot).hexdigest()
        if not hmac.compare_digest(observed, binding.calibration_sha256):
            raise ValueError("calibration file hash does not match registry")
        artifact_type: type[ScoreCalibration] | type[RegressionConfidenceArtifact] = (
            ScoreCalibration
            if binding.confidence_kind == "binary-calibration"
            else RegressionConfidenceArtifact
        )
        calibration = artifact_type.model_validate_json(snapshot)
        version = (
            calibration.calibration_version
            if isinstance(calibration, ScoreCalibration)
            else calibration.confidence_version
        )
        if (
            version != binding.calibration_version
            or calibration.method != binding.calibration_method
            or calibration.artifact_hash != binding.calibration_artifact_hash
            or calibration.protocol_hash != binding.calibration_protocol_hash
        ):
            raise ValueError("loaded calibration does not match serving task head")
        return calibration

    def resolve_model(self, model_version_id: str, *, require_serving: bool = True) -> ServingModel:
        self.refresh()
        model = next(
            (item for item in self._document.models if item.model_version_id == model_version_id),
            None,
        )
        if model is None:
            raise LookupError("model version is not registered")
        if require_serving and model.state != "servingReady":
            raise LookupError("model version is not serving ready")
        self.checkpoint_snapshot(model)
        return model

    def acquire_model_lease(
        self,
        model_version_id: str,
        task_id: str,
        *,
        registry_snapshot: bytes | None = None,
    ) -> CapturedModelLease:
        """Capture one model and all referenced bytes under a stable control snapshot."""

        for _attempt in range(1 if registry_snapshot is not None else 3):
            control = (
                registry_snapshot if registry_snapshot is not None else self._registry_snapshot()
            )
            source_hash = hashlib.sha256(control).hexdigest()
            document = RegistryDocument.model_validate_json(control)
            self._validate_unique(document)
            model = next(
                (item for item in document.models if item.model_version_id == model_version_id),
                None,
            )
            if model is None or model.state != "servingReady":
                raise LookupError("model version is not serving ready")
            head = model.task_head(task_id)
            manifest = read_confined_snapshot(
                self.runtime_root,
                model.checkpoint.serving_manifest_relative_path,
                max_bytes=1024 * 1024,
            )
            checkpoint = read_confined_snapshot(
                self.runtime_root,
                model.checkpoint.relative_path,
                max_bytes=MAX_CHECKPOINT_BYTES,
            )
            calibrations = tuple(
                (
                    binding.entity_type,
                    read_confined_snapshot(
                        self.runtime_root,
                        binding.calibration_relative_path,
                        max_bytes=1024 * 1024,
                    ),
                )
                for binding in head.calibrations
            )
            if (
                registry_snapshot is None
                and hashlib.sha256(self._registry_snapshot()).hexdigest() != source_hash
            ):
                continue
            lease = CapturedModelLease(
                registry_snapshot=control,
                registry_source_sha256=source_hash,
                registry_hash=canonical_sha256(document.model_dump(mode="python", by_alias=True)),
                registry_generation=document.generation,
                model_version_id=model_version_id,
                task_id=task_id,
                manifest_snapshot=manifest,
                checkpoint_snapshot=checkpoint,
                calibration_snapshots=calibrations,
            )
            lease.materialize()
            return lease
        raise ValueError("registry changed during bounded serving lease acquisition")

    def control_matches(self, lease: CapturedModelLease) -> bool:
        return hmac.compare_digest(
            hashlib.sha256(self._registry_snapshot()).hexdigest(),
            lease.registry_source_sha256,
        )

    def capabilities(self, *, registry_snapshot: bytes | None = None) -> dict[str, object]:
        for _attempt in range(1 if registry_snapshot is not None else 3):
            control = (
                registry_snapshot if registry_snapshot is not None else self._registry_snapshot()
            )
            document = RegistryDocument.model_validate_json(control)
            self._validate_unique(document)
            accepted = list(document.models)
            for model in accepted:
                manifest = read_confined_snapshot(
                    self.runtime_root,
                    model.checkpoint.serving_manifest_relative_path,
                    max_bytes=1024 * 1024,
                )
                checkpoint = read_confined_snapshot(
                    self.runtime_root,
                    model.checkpoint.relative_path,
                    max_bytes=MAX_CHECKPOINT_BYTES,
                )
                _validate_captured_checkpoint(model, manifest, checkpoint)
                for task_id in model.tasks:
                    snapshots: dict[str, bytes] = {
                        binding.entity_type: read_confined_snapshot(
                            self.runtime_root,
                            binding.calibration_relative_path,
                            max_bytes=1024 * 1024,
                        )
                        for binding in model.task_head(task_id).calibrations
                    }
                    _validate_captured_calibrations(model, task_id, snapshots)
            if registry_snapshot is not None or self._registry_snapshot() == control:
                break
        else:
            raise ValueError("registry changed during bounded capability acquisition")
        serving = [model for model in accepted if model.state == "servingReady"]
        tasks = sorted({task for model in accepted for task in model.tasks})
        return {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "registryHash": canonical_sha256(document.model_dump(mode="python", by_alias=True)),
            "registryGeneration": document.generation,
            "servingReady": bool(serving),
            "models": [
                {
                    "modelVersionId": model.model_version_id,
                    "modelVersionHash": model.model_version_hash,
                    "state": model.state,
                    "tasks": list(model.tasks),
                    "graphSchemaVersions": list(model.graph_schema_versions),
                    "graphFeatureContractHash": model.graph_feature_contract_hash,
                    "taskBindings": [
                        {
                            "taskId": head.task_id,
                            "entityType": binding.entity_type,
                            "confidenceKind": binding.confidence_kind,
                            "calibrationVersion": binding.calibration_version,
                            "method": binding.calibration_method,
                            "calibrationArtifactHash": binding.calibration_artifact_hash,
                            "calibrationProtocolHash": binding.calibration_protocol_hash,
                            "adapterDomain": binding.adapter_domain,
                            "adapterSchemaHash": binding.adapter_schema_hash,
                            "adapterStateHash": binding.adapter_state_hash,
                            "featureContractHash": binding.graph_feature_contract_hash,
                        }
                        for head in model.task_heads
                        for binding in head.calibrations
                    ],
                    "maxNodes": model.max_nodes,
                    "maxEdges": model.max_edges,
                }
                for model in accepted
            ],
            "tasks": tasks,
            "readiness": {
                "modelValidated": bool(accepted),
                "coreServingReady": bool(serving),
            },
        }


def write_empty_registry(path: Path) -> None:
    """Publish the default no-product-model registry atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
        "generation": 0,
        "models": [],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


__all__ = [
    "CalibrationBinding",
    "ConfidenceArtifact",
    "RegressionConfidenceArtifact",
    "ServingModel",
    "ServingAdapterBinding",
    "ServingRegistry",
    "ScoreCalibration",
    "ServingTaskHead",
    "VerifiedCheckpoint",
    "write_empty_registry",
]
