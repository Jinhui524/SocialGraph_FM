"""Versioned contracts shared by materialization, runs, registry and future serving."""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .canonical import canonical_sha256

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class FeatureTarget(str, Enum):
    NODE = "node"
    EDGE = "edge"
    GRAPH = "graph"


class FeatureModality(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"


class MissingPolicy(str, Enum):
    ERROR = "error"
    ZERO_WITH_MASK = "zero_with_mask"
    CATEGORY_WITH_MASK = "category_with_mask"


class PrivacyLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    PROHIBITED = "prohibited"


class FitScope(str, Enum):
    NONE = "none"
    TRAIN_SPLIT_ONLY = "train_split_only"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InternalTimeRange(ContractModel):
    start: datetime
    end: datetime

    @model_validator(mode="after")
    def validate_range(self) -> InternalTimeRange:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("time range values must be timezone-aware")
        if self.end < self.start:
            raise ValueError("time range end must not precede start")
        return self


class InternalFeatureManifest(ContractModel):
    schema_version: Literal["gfm.feature/1.0"] = Field("gfm.feature/1.0", alias="schemaVersion")
    name: str = Field(min_length=1, max_length=128, pattern=r"^[^.]+$")
    target: FeatureTarget
    owner_type: str | None = Field(None, alias="ownerType", max_length=128, pattern=r"^[^.]+$")
    modality: FeatureModality
    dtype: str = Field(min_length=1, max_length=64)
    dimensions: int = Field(1, ge=1, le=65_536)
    missing_policy: MissingPolicy = Field(alias="missingPolicy")
    privacy_level: PrivacyLevel = Field(alias="privacyLevel")
    inference_allowed: bool = Field(alias="inferenceAllowed")
    fit_scope: FitScope = Field(FitScope.NONE, alias="fitScope")
    available_at: datetime | None = Field(None, alias="availableAt")
    embedding_model_ref: str | None = Field(None, alias="embeddingModelRef", max_length=512)

    @model_validator(mode="after")
    def validate_semantics(self) -> InternalFeatureManifest:
        if self.available_at is not None and (
            self.available_at.tzinfo is None or self.available_at.utcoffset() is None
        ):
            raise ValueError("availableAt must be timezone-aware")
        if self.privacy_level == PrivacyLevel.PROHIBITED and self.inference_allowed:
            raise ValueError("prohibited features cannot be enabled for inference")
        if self.modality == FeatureModality.TEXT and self.embedding_model_ref:
            raise ValueError("raw text features cannot claim an embedding model reference")
        if self.modality == FeatureModality.EMBEDDING and not self.embedding_model_ref:
            raise ValueError("embedding features require embeddingModelRef")
        if self.modality not in (FeatureModality.EMBEDDING,) and self.dimensions != 1:
            raise ValueError("only embedding features may have dimensions greater than one")
        if (
            self.modality in (FeatureModality.NUMERIC, FeatureModality.CATEGORICAL)
            and self.fit_scope != FitScope.TRAIN_SPLIT_ONLY
        ):
            raise ValueError("numeric/categorical features must fit on the train split only")
        return self


class InternalGraphSnapshotRef(ContractModel):
    schema_version: Literal["gfm.graph-snapshot-ref/1.0"] = Field(
        "gfm.graph-snapshot-ref/1.0", alias="schemaVersion"
    )
    graph_version: str = Field(alias="graphVersion", min_length=1, max_length=128)
    fact_hash: str = Field(alias="factHash", pattern=SHA256_PATTERN)
    content_hash: str = Field(alias="contentHash", pattern=SHA256_PATTERN)
    profile: Literal[
        "collaboration.actor-interaction/1.0",
        "collaboration.activity-hetero/1.0",
    ]
    time_range: InternalTimeRange | None = Field(None, alias="timeRange")
    feature_manifest: tuple[InternalFeatureManifest, ...] = Field(default=(), alias="featureManifest")
    inference_property_allowlist: tuple[str, ...] = Field(
        default=(), alias="inferencePropertyAllowlist"
    )
    deidentification: Literal["none", "pseudonymized", "aggregated"] = "pseudonymized"
    user_data_training_opt_in: Literal[False] = Field(False, alias="userDataTrainingOptIn")

    @model_validator(mode="after")
    def validate_features(self) -> InternalGraphSnapshotRef:
        keys = [feature_key(feature) for feature in self.feature_manifest]
        if len(keys) != len(set(keys)):
            raise ValueError("owner-qualified feature keys must be unique")
        by_name: dict[str, list[InternalFeatureManifest]] = {}
        by_key = {feature_key(feature): feature for feature in self.feature_manifest}
        for feature in self.feature_manifest:
            by_name.setdefault(feature.name, []).append(feature)
        resolved: list[InternalFeatureManifest] = []
        for item in self.inference_property_allowlist:
            if item in by_key:
                resolved.append(by_key[item])
                continue
            matches = by_name.get(item, [])
            if len(matches) > 1:
                raise ValueError(f"allowlist feature {item!r} is ambiguous; use owner.name")
            if not matches:
                raise ValueError(f"allowlist contains undeclared feature: {item!r}")
            resolved.extend(matches)
        if len(resolved) != len({feature_key(feature) for feature in resolved}):
            raise ValueError("allowlist resolves to duplicate owner-qualified features")
        if any(
            not feature.inference_allowed or feature.privacy_level == PrivacyLevel.PROHIBITED
            for feature in resolved
        ):
            raise ValueError("allowlist contains a prohibited or inference-disabled feature")
        return self

    def resolved_inference_feature_keys(self) -> tuple[str, ...]:
        by_key = {feature_key(feature): feature for feature in self.feature_manifest}
        by_name: dict[str, list[InternalFeatureManifest]] = {}
        for feature in self.feature_manifest:
            by_name.setdefault(feature.name, []).append(feature)
        return tuple(
            item if item in by_key else feature_key(by_name[item][0])
            for item in self.inference_property_allowlist
        )


def feature_key(feature: InternalFeatureManifest) -> str:
    """Return the unambiguous internal feature identity used by fitting and inference."""

    return f"{feature.owner_type}.{feature.name}" if feature.owner_type else feature.name


TRANSFORM_RECIPE_VERSION: Literal["gfm.feature-transform-recipe/1.0"] = (
    "gfm.feature-transform-recipe/1.0"
)


class FeatureTransformState(ContractModel):
    schema_version: Literal["gfm.feature-transform-state/1.0"] = Field(
        "gfm.feature-transform-state/1.0", alias="schemaVersion"
    )
    recipe_version: Literal["gfm.feature-transform-recipe/1.0"] = Field(
        TRANSFORM_RECIPE_VERSION, alias="recipeVersion"
    )
    feature_key: str = Field(alias="featureKey", min_length=1)
    owner_type: str = Field(alias="ownerType", min_length=1)
    modality: Literal[FeatureModality.NUMERIC, FeatureModality.CATEGORICAL]
    dtype: str = Field(min_length=1)
    dimensions: int = Field(ge=1)
    missing_policy: MissingPolicy = Field(alias="missingPolicy")
    kind: Literal["numeric_standardizer", "category_vocabulary"]
    mean: float | None = None
    scale: float | None = None
    categories: tuple[str, ...] | None = None
    categories_hash: str | None = Field(None, alias="categoriesHash", pattern=SHA256_PATTERN)
    state_hash: str = Field(alias="stateHash", pattern=SHA256_PATTERN)

    def logical_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "recipeVersion": self.recipe_version,
            "featureKey": self.feature_key,
            "ownerType": self.owner_type,
            "modality": self.modality,
            "dtype": self.dtype,
            "dimensions": self.dimensions,
            "missingPolicy": self.missing_policy,
            "kind": self.kind,
            "mean": self.mean,
            "scale": self.scale,
            "categories": self.categories,
            "categoriesHash": self.categories_hash,
        }

    @model_validator(mode="after")
    def validate_state(self) -> FeatureTransformState:
        if self.feature_key != f"{self.owner_type}.{self.feature_key.rsplit('.', 1)[-1]}":
            raise ValueError("featureKey must be owner-qualified")
        if self.kind == "numeric_standardizer":
            if self.modality != FeatureModality.NUMERIC:
                raise ValueError("numeric transform requires numeric modality")
            if self.mean is None or self.scale is None or not math.isfinite(self.mean):
                raise ValueError("numeric transform requires a finite mean and scale")
            if not math.isfinite(self.scale) or self.scale <= 0:
                raise ValueError("numeric transform scale must be finite and positive")
            if self.categories is not None or self.categories_hash is not None:
                raise ValueError("numeric transform cannot contain categories")
        else:
            if self.modality != FeatureModality.CATEGORICAL:
                raise ValueError("category transform requires categorical modality")
            if self.mean is not None or self.scale is not None:
                raise ValueError("category transform cannot contain numeric state")
            if not self.categories or tuple(sorted(set(self.categories))) != self.categories:
                raise ValueError("categories must be a nonempty sorted unique tuple")
            if self.categories_hash != canonical_sha256(self.categories):
                raise ValueError("categoriesHash does not match categories")
        if self.state_hash != canonical_sha256(self.logical_payload()):
            raise ValueError("stateHash does not match transform state")
        return self


class FeatureTransformArtifact(ContractModel):
    schema_version: Literal["gfm.feature-transform-artifact/1.0"] = Field(
        "gfm.feature-transform-artifact/1.0", alias="schemaVersion"
    )
    recipe_version: Literal["gfm.feature-transform-recipe/1.0"] = Field(
        TRANSFORM_RECIPE_VERSION, alias="recipeVersion"
    )
    source_snapshot_payload_hash: str = Field(
        alias="sourceSnapshotPayloadHash", pattern=SHA256_PATTERN
    )
    source_snapshot_contract_hash: str = Field(
        alias="sourceSnapshotContractHash", pattern=SHA256_PATTERN
    )
    fit_selection_hash: str = Field(alias="fitSelectionHash", pattern=SHA256_PATTERN)
    states: tuple[FeatureTransformState, ...]
    artifact_hash: str = Field(alias="artifactHash", pattern=SHA256_PATTERN)

    def logical_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "recipeVersion": self.recipe_version,
            "sourceSnapshotPayloadHash": self.source_snapshot_payload_hash,
            "sourceSnapshotContractHash": self.source_snapshot_contract_hash,
            "fitSelectionHash": self.fit_selection_hash,
            "states": self.states,
        }

    @model_validator(mode="after")
    def validate_artifact(self) -> FeatureTransformArtifact:
        keys = [state.feature_key for state in self.states]
        if len(keys) != len(set(keys)):
            raise ValueError("transform states must have unique featureKey values")
        if self.artifact_hash != canonical_sha256(self.logical_payload()):
            raise ValueError("artifactHash does not match transform artifact")
        return self


class NodeRecord(ContractModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=512)
    node_type: str = Field(alias="nodeType", min_length=1, max_length=128)
    features: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_finite(self) -> NodeRecord:
        _assert_finite(self.features)
        return self


class EdgeRecord(ContractModel):
    edge_id: str = Field(alias="edgeId", min_length=1, max_length=512)
    source: str = Field(min_length=1, max_length=512)
    target: str = Field(min_length=1, max_length=512)
    relation: str = Field(min_length=1, max_length=128)
    timestamp: datetime | None = None
    valid_from: datetime | None = Field(None, alias="validFrom")
    valid_to: datetime | None = Field(None, alias="validTo")
    weight: float = 1.0
    original_relation_type: str | None = Field(None, alias="originalRelationType")
    features: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_temporal(self) -> EdgeRecord:
        for value in (self.timestamp, self.valid_from, self.valid_to):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("edge temporal fields must be timezone-aware")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("validTo must not precede validFrom")
        if not math.isfinite(self.weight):
            raise ValueError("edge weight must be finite")
        _assert_finite(self.features)
        return self


class GraphSnapshot(ContractModel):
    schema_version: Literal["gfm.graph-snapshot/1.0"] = Field(
        "gfm.graph-snapshot/1.0", alias="schemaVersion"
    )
    ref: InternalGraphSnapshotRef
    nodes: tuple[NodeRecord, ...]
    edges: tuple[EdgeRecord, ...]

    @model_validator(mode="after")
    def validate_integrity(self) -> GraphSnapshot:
        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("edge IDs must be unique")
        known = set(node_ids)
        dangling = [edge.edge_id for edge in self.edges if edge.source not in known or edge.target not in known]
        if dangling:
            raise ValueError(f"edges reference unknown nodes: {dangling[:5]}")
        expected_content_hash = self.payload_hash()
        if self.ref.content_hash != expected_content_hash:
            raise ValueError(
                "ref.contentHash does not match the canonical snapshot payload: "
                f"expected {expected_content_hash}"
            )
        expected_fact_hash = self.fact_payload_hash()
        if self.ref.fact_hash != expected_fact_hash:
            raise ValueError(
                "ref.factHash does not match the canonical node/edge facts: "
                f"expected {expected_fact_hash}"
            )
        return self

    def payload_hash(self) -> str:
        return canonical_sha256(
            {
                "profile": self.ref.profile,
                "featureManifest": self.ref.feature_manifest,
                "nodes": self.nodes,
                "edges": self.edges,
            }
        )

    def fact_payload_hash(self) -> str:
        return canonical_sha256({"nodes": self.nodes, "edges": self.edges})


class SmokeCorpusManifest(ContractModel):
    schema_version: Literal["gfm.corpus/1.0"] = Field("gfm.corpus/1.0", alias="schemaVersion")
    corpus_id: str = Field(alias="corpusId", min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    purpose: Literal["synthetic_test_only"] = "synthetic_test_only"
    license_id: Literal["INTERNAL-SYNTHETIC-NONDATA"] = Field(
        "INTERNAL-SYNTHETIC-NONDATA", alias="licenseId"
    )
    adapter: str = Field(min_length=1, max_length=256)
    split: Literal["synthetic_smoke"] = "synthetic_smoke"
    source_hash: str = Field(alias="sourceHash", pattern=SHA256_PATTERN)
    snapshot_refs: tuple[InternalGraphSnapshotRef, ...] = Field(alias="snapshotRefs", min_length=1)


class InternalGovernanceTaskManifest(ContractModel):
    schema_version: Literal["gfm.governance-task/1.0"] = Field(
        "gfm.governance-task/1.0", alias="schemaVersion"
    )
    task_id: Literal[
        "core.community_health_observation",
        "core.newcomer_support",
        "core.coordination_review",
    ] = Field(alias="taskId")
    enabled: Literal[False] = False
    required_profiles: tuple[str, ...] = Field(alias="requiredProfiles", min_length=1)
    required_modalities: tuple[FeatureModality, ...] = Field(default=(), alias="requiredModalities")
    output_kind: str = Field(alias="outputKind", min_length=1, max_length=128)
    human_review_required: Literal[True] = Field(True, alias="humanReviewRequired")
    refusal_conditions: tuple[str, ...] = Field(alias="refusalConditions", min_length=1)


class SmokeRunMetrics(ContractModel):
    schema_version: Literal["gfm.smoke-metrics/1.0"] = Field(
        "gfm.smoke-metrics/1.0", alias="schemaVersion"
    )
    device: Literal["cpu", "cuda"]
    elapsed_seconds: float = Field(alias="elapsedSeconds", ge=0)
    max_memory_mb: float = Field(alias="maxMemoryMb", ge=0)
    fresh_process_verified: Literal[True] = Field(True, alias="freshProcessVerified")
    optimizer_restored: Literal[True] = Field(True, alias="optimizerRestored")
    checkpoint_state_hash: str = Field(alias="checkpointStateHash", pattern=SHA256_PATTERN)
    checkpoint_artifact_sha256: str = Field(
        alias="checkpointArtifactSha256", pattern=SHA256_PATTERN
    )


class SmokeTrainingRunManifest(ContractModel):
    schema_version: Literal["gfm.training-run/1.0"] = Field(
        "gfm.training-run/1.0", alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=128)
    run_kind: Literal["smoke"] = Field("smoke", alias="runKind")
    status: RunStatus
    seed: int = Field(ge=0, le=2**32 - 1)
    code_hash: str = Field(alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str = Field(alias="environmentHash", pattern=SHA256_PATTERN)
    corpus: SmokeCorpusManifest
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime | None = Field(None, alias="finishedAt")
    failure_code: str | None = Field(None, alias="failureCode")
    artifacts: tuple[str, ...] = ()
    smoke_metrics: SmokeRunMetrics | None = Field(None, alias="smokeMetrics")

    @model_validator(mode="after")
    def validate_status(self) -> SmokeTrainingRunManifest:
        if self.started_at.tzinfo is None:
            raise ValueError("startedAt must be timezone-aware")
        if self.finished_at is not None and self.finished_at.tzinfo is None:
            raise ValueError("finishedAt must be timezone-aware")
        if (
            self.status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED)
            and self.finished_at is None
        ):
            raise ValueError("terminal runs require finishedAt")
        if self.status == RunStatus.FAILED and not self.failure_code:
            raise ValueError("failed runs require failureCode")
        if self.status == RunStatus.SUCCEEDED and self.smoke_metrics is None:
            raise ValueError("succeeded smoke runs require smokeMetrics attestation")
        return self


class SmokeCheckpointManifest(ContractModel):
    schema_version: Literal["gfm.checkpoint/1.0"] = Field(
        "gfm.checkpoint/1.0", alias="schemaVersion"
    )
    checkpoint_id: str = Field(alias="checkpointId", min_length=1, max_length=128)
    run_id: str = Field(alias="runId", min_length=1, max_length=128)
    step: int = Field(ge=0)
    smoke_only: bool = Field(alias="smokeOnly")
    state_hash: str = Field(alias="stateHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    artifact_sha256: str = Field(alias="artifactSha256", pattern=SHA256_PATTERN)
    artifact_path: str = Field(alias="artifactPath", min_length=1)
    created_at: datetime = Field(alias="createdAt")


class CorpusArrayManifest(ContractModel):
    name: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=SHA256_PATTERN)
    dtype: str = Field(min_length=1, max_length=64)
    shape: tuple[int, ...] = Field(min_length=0, max_length=8)
    byte_count: int = Field(alias="byteCount", ge=0)

    @model_validator(mode="after")
    def validate_shape(self) -> CorpusArrayManifest:
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("array shape dimensions must be non-negative")
        return self


class TemporalLinkProtocolManifest(ContractModel):
    schema_version: Literal["gfm.temporal-link-protocol/1.0"] = Field(
        "gfm.temporal-link-protocol/1.0", alias="schemaVersion"
    )
    protocol_id: Literal["ogbl-collab-official-v1"] = Field(alias="protocolId")
    train_year_max: Literal[2017] = Field(2017, alias="trainYearMax")
    validation_year: Literal[2018] = Field(2018, alias="validationYear")
    test_year: Literal[2019] = Field(2019, alias="testYear")
    evaluator: Literal["ogb.linkproppred.Evaluator(ogbl-collab)"]
    primary_metric: Literal["Hits@50"] = Field("Hits@50", alias="primaryMetric")
    validation_negative_source: Literal["ogb_official_stored"] = Field(
        "ogb_official_stored", alias="validationNegativeSource"
    )
    test_negative_source: Literal["ogb_official_stored"] = Field(
        "ogb_official_stored", alias="testNegativeSource"
    )
    undirected_canonicalization: Literal["min_max"] = Field(
        "min_max", alias="undirectedCanonicalization"
    )
    feature_point_in_time_verified: Literal[False] = Field(
        False, alias="featurePointInTimeVerified"
    )


class FormalCorpusManifest(ContractModel):
    schema_version: Literal["gfm.formal-corpus/1.0"] = Field(
        "gfm.formal-corpus/1.0", alias="schemaVersion"
    )
    corpus_id: Literal["ogbl-collab"] = Field("ogbl-collab", alias="corpusId")
    purpose: Literal["formal_benchmark"] = "formal_benchmark"
    dataset_role: Literal["benchmark"] = Field("benchmark", alias="datasetRole")
    ogb_version: Literal["1.3.6"] = Field("1.3.6", alias="ogbVersion")
    license_id: Literal["ODC-BY-1.0"] = Field("ODC-BY-1.0", alias="licenseId")
    license_source_url: str = Field(alias="licenseSourceUrl", min_length=1, max_length=4096)
    license_accepted: Literal[True] = Field(True, alias="licenseAccepted")
    attribution: str = Field(min_length=1, max_length=1024)
    source_fingerprint: str = Field(alias="sourceFingerprint", pattern=SHA256_PATTERN)
    package_sha256: str = Field(alias="packageSha256", pattern=SHA256_PATTERN)
    adapter: str = Field(min_length=1, max_length=256)
    adapter_version: str = Field(alias="adapterVersion", min_length=1, max_length=64)
    node_count: Literal[235868] = Field(235868, alias="nodeCount")
    feature_shape: tuple[Literal[235868], Literal[128]] = Field(alias="featureShape")
    message_edge_count: int = Field(alias="messageEdgeCount", ge=1)
    split_sizes: dict[str, int] = Field(alias="splitSizes")
    arrays: tuple[CorpusArrayManifest, ...] = Field(min_length=1)
    temporal_protocol: TemporalLinkProtocolManifest = Field(alias="temporalProtocol")
    warnings: tuple[str, ...] = ()
    logical_hash: str = Field(alias="logicalHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")
    artifact_path: str = Field(alias="artifactPath", min_length=1)

    def logical_payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "corpusId": self.corpus_id,
            "purpose": self.purpose,
            "datasetRole": self.dataset_role,
            "ogbVersion": self.ogb_version,
            "licenseId": self.license_id,
            "licenseSourceUrl": self.license_source_url,
            "licenseAccepted": self.license_accepted,
            "attribution": self.attribution,
            "sourceFingerprint": self.source_fingerprint,
            "packageSha256": self.package_sha256,
            "adapter": self.adapter,
            "adapterVersion": self.adapter_version,
            "nodeCount": self.node_count,
            "featureShape": self.feature_shape,
            "messageEdgeCount": self.message_edge_count,
            "splitSizes": self.split_sizes,
            "arrays": self.arrays,
            "temporalProtocol": self.temporal_protocol,
            "warnings": self.warnings,
        }

    @model_validator(mode="after")
    def validate_corpus(self) -> FormalCorpusManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("createdAt must be timezone-aware")
        required_splits = {"train", "validation", "test"}
        if set(self.split_sizes) != required_splits:
            raise ValueError("splitSizes must contain exactly train/validation/test")
        if any(count <= 0 for count in self.split_sizes.values()):
            raise ValueError("all formal split sizes must be positive")
        names = [array.name for array in self.arrays]
        if len(names) != len(set(names)):
            raise ValueError("formal corpus array names must be unique")
        if self.logical_hash != canonical_sha256(self.logical_payload()):
            raise ValueError("logicalHash does not match formal corpus payload")
        return self


class BaselineConfig(ContractModel):
    schema_version: Literal["gfm.baseline-config/1.0"] = Field(
        "gfm.baseline-config/1.0", alias="schemaVersion"
    )
    config_id: Literal["ogbl-collab-baseline"] = Field(alias="configId")
    tracks: tuple[Literal["ogb_official", "strict_edge_time"], ...]
    models: tuple[Literal["cn", "aa", "ra", "mlp", "graphsage"], ...]
    dev_seed: Literal[20260811] = Field(20260811, alias="devSeed")
    formal_seeds: tuple[Literal[20260812, 20260813, 20260814], ...] = Field(
        alias="formalSeeds"
    )
    hidden_channels: Literal[128] = Field(128, alias="hiddenChannels")
    dropout: float = 0.2
    learning_rate: float = Field(0.001, alias="learningRate")
    weight_decay: float = Field(0.0, alias="weightDecay")
    gradient_clip: float = Field(1.0, alias="gradientClip")
    negative_ratio: float = Field(1.0, alias="negativeRatio")
    neighbor_fanout: tuple[Literal[15], Literal[10]] = Field(alias="neighborFanout")
    candidate_batch_sizes: tuple[Literal[4096, 2048, 1024], ...] = Field(
        alias="candidateBatchSizes"
    )
    cuda_memory_limit_mib: Literal[7168] = Field(7168, alias="cudaMemoryLimitMiB")
    dev_epochs: Literal[5] = Field(5, alias="devEpochs")
    dev_positive_limit: Literal[50000] = Field(50000, alias="devPositiveLimit")
    formal_max_epochs: Literal[50] = Field(50, alias="formalMaxEpochs")
    formal_min_epochs: Literal[10] = Field(10, alias="formalMinEpochs")
    eval_every: Literal[2] = Field(2, alias="evalEvery")
    patience: Literal[8] = 8
    train_positive_limit: Literal[262144] = Field(262144, alias="trainPositiveLimit")
    score_batch_size: Literal[65536] = Field(65536, alias="scoreBatchSize")
    inference_batch_size: Literal[8192] = Field(8192, alias="inferenceBatchSize")
    max_checkpoints_per_run: Literal[3] = Field(3, alias="maxCheckpointsPerRun")
    official_min_validation_hits50: float = Field(
        0.4, alias="officialMinValidationHits50"
    )
    official_min_test_hits50: float = Field(
        0.35, alias="officialMinTestHits50"
    )
    official_min_test_gain_over_mlp: float = Field(
        0.05, alias="officialMinTestGainOverMlp"
    )

    @model_validator(mode="after")
    def validate_fixed_suite(self) -> BaselineConfig:
        if self.tracks != ("ogb_official", "strict_edge_time"):
            raise ValueError("baseline v1 requires both fixed tracks")
        if self.models != ("cn", "aa", "ra", "mlp", "graphsage"):
            raise ValueError("baseline v1 requires the fixed model suite")
        if self.formal_seeds != (20260812, 20260813, 20260814):
            raise ValueError("baseline v1 requires the three fixed formal seeds")
        if self.candidate_batch_sizes != (4096, 2048, 1024):
            raise ValueError("baseline v1 batch probe order is immutable")
        fixed_floats = (
            ("dropout", self.dropout, 0.2),
            ("learningRate", self.learning_rate, 0.001),
            ("weightDecay", self.weight_decay, 0.0),
            ("gradientClip", self.gradient_clip, 1.0),
            ("negativeRatio", self.negative_ratio, 1.0),
            ("officialMinValidationHits50", self.official_min_validation_hits50, 0.4),
            ("officialMinTestHits50", self.official_min_test_hits50, 0.35),
            ("officialMinTestGainOverMlp", self.official_min_test_gain_over_mlp, 0.05),
        )
        for name, actual, expected in fixed_floats:
            if actual != expected:
                raise ValueError(f"baseline v1 requires {name}={expected}")
        return self


class BaselineCheckpointManifest(ContractModel):
    schema_version: Literal["gfm.baseline-checkpoint/1.0"] = Field(
        "gfm.baseline-checkpoint/1.0", alias="schemaVersion"
    )
    checkpoint_id: str = Field(alias="checkpointId", min_length=1, max_length=128)
    run_id: str = Field(alias="runId", min_length=1, max_length=128)
    epoch: int = Field(ge=0)
    track: Literal["ogb_official", "strict_edge_time"]
    model: Literal["mlp", "graphsage"]
    checkpoint_kind: Literal["baseline"] = Field("baseline", alias="checkpointKind")
    registrable: Literal[False] = False
    state_hash: str = Field(alias="stateHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    corpus_hash: str = Field(alias="corpusHash", pattern=SHA256_PATTERN)
    artifact_sha256: str = Field(alias="artifactSha256", pattern=SHA256_PATTERN)
    artifact_path: str = Field(alias="artifactPath", min_length=1)
    verification_digest: str | None = Field(
        None, alias="verificationDigest", pattern=SHA256_PATTERN
    )
    created_at: datetime = Field(alias="createdAt")

    @model_validator(mode="after")
    def validate_checkpoint_time(self) -> BaselineCheckpointManifest:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("createdAt must be timezone-aware")
        return self


class BaselineRunManifest(ContractModel):
    schema_version: Literal["gfm.baseline-run/1.0"] = Field(
        "gfm.baseline-run/1.0", alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=128)
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=128)
    run_kind: Literal["baseline", "exploratory"] = Field(alias="runKind")
    phase: Literal["dev", "formal"]
    track: Literal["ogb_official", "strict_edge_time"]
    model: Literal["cn", "aa", "ra", "mlp", "graphsage"]
    status: RunStatus
    seed: int = Field(ge=0, le=2**32 - 1)
    code_hash: str = Field(alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str = Field(alias="environmentHash", pattern=SHA256_PATTERN)
    corpus_hash: str = Field(alias="corpusHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime | None = Field(None, alias="finishedAt")
    best_epoch: int | None = Field(None, alias="bestEpoch", ge=0)
    best_validation_hits50: float | None = Field(None, alias="bestValidationHits50")
    peak_cuda_memory_mib: float = Field(0.0, alias="peakCudaMemoryMiB", ge=0)
    duration_seconds: float | None = Field(None, alias="durationSeconds", ge=0)
    failure_code: str | None = Field(None, alias="failureCode")
    artifacts: tuple[str, ...] = ()
    registrable: Literal[False] = False

    @model_validator(mode="after")
    def validate_baseline_run(self) -> BaselineRunManifest:
        for field_name, value in (("startedAt", self.started_at), ("finishedAt", self.finished_at)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{field_name} must be timezone-aware")
        terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
        if self.status in terminal and self.finished_at is None:
            raise ValueError("terminal baseline runs require finishedAt")
        if self.status == RunStatus.FAILED and not self.failure_code:
            raise ValueError("failed baseline runs require failureCode")
        if self.best_validation_hits50 is not None and not math.isfinite(
            self.best_validation_hits50
        ):
            raise ValueError("bestValidationHits50 must be finite")
        return self


class BaselineEvaluationReport(ContractModel):
    schema_version: Literal["gfm.baseline-evaluation/1.0"] = Field(
        "gfm.baseline-evaluation/1.0", alias="schemaVersion"
    )
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=128)
    run_id: str = Field(alias="runId", min_length=1, max_length=128)
    phase: Literal["dev", "formal"]
    track: Literal["ogb_official", "strict_edge_time"]
    model: Literal["cn", "aa", "ra", "mlp", "graphsage"]
    seed: int = Field(ge=0, le=2**32 - 1)
    validation_metrics: dict[str, float] = Field(alias="validationMetrics")
    test_metrics: dict[str, float] | None = Field(None, alias="testMetrics")
    strata: dict[str, dict[str, float]] = Field(default_factory=dict)
    score_counts: dict[str, int] = Field(alias="scoreCounts")
    test_read_after_selection: bool = Field(False, alias="testReadAfterSelection")
    report_hash: str = Field(alias="reportHash", pattern=SHA256_PATTERN)

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python", by_alias=True, exclude={"report_hash"}
        )

    @model_validator(mode="after")
    def validate_evaluation(self) -> BaselineEvaluationReport:
        _assert_finite(self.validation_metrics)
        _assert_finite(self.test_metrics)
        _assert_finite(self.strata)
        if self.phase == "dev" and self.test_metrics is not None:
            raise ValueError("development runs must not read test metrics")
        if self.test_metrics is not None and not self.test_read_after_selection:
            raise ValueError("test metrics require testReadAfterSelection=true")
        if self.report_hash != canonical_sha256(self.logical_payload()):
            raise ValueError("reportHash does not match baseline evaluation")
        return self


class BaselineAcceptanceReport(ContractModel):
    schema_version: Literal["gfm.baseline-acceptance/1.0"] = Field(
        "gfm.baseline-acceptance/1.0", alias="schemaVersion"
    )
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=128)
    accepted: bool
    corpus_hash: str = Field(alias="corpusHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    required_learning_runs: Literal[12] = Field(12, alias="requiredLearningRuns")
    completed_learning_runs: int = Field(alias="completedLearningRuns", ge=0, le=12)
    completed_heuristic_runs: int = Field(alias="completedHeuristicRuns", ge=0, le=6)
    peak_cuda_memory_mib: float = Field(alias="peakCudaMemoryMiB", ge=0)
    metric_summary: dict[str, dict[str, float]] = Field(alias="metricSummary")
    code_hash: str | None = Field(None, alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str | None = Field(
        None, alias="environmentHash", pattern=SHA256_PATTERN
    )
    duration_seconds: float | None = Field(None, alias="durationSeconds", ge=0)
    run_results: tuple[dict[str, Any], ...] | None = Field(None, alias="runResults")
    gates: dict[str, bool]
    warnings: tuple[str, ...] = ()
    report_hash: str = Field(alias="reportHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python",
            by_alias=True,
            exclude={"report_hash", "created_at"},
            exclude_none=True,
        )

    @model_validator(mode="after")
    def validate_acceptance(self) -> BaselineAcceptanceReport:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("createdAt must be timezone-aware")
        _assert_finite(self.metric_summary)
        _assert_finite(self.run_results)
        if self.accepted and (
            self.completed_learning_runs != self.required_learning_runs
            or self.completed_heuristic_runs != 6
            or not self.gates
            or not all(self.gates.values())
        ):
            raise ValueError("accepted baseline reports require all fixed gates")
        if self.report_hash != canonical_sha256(self.logical_payload()):
            raise ValueError("reportHash does not match baseline acceptance")
        return self


class InternalModelCapability(ContractModel):
    schema_version: Literal["gfm.model-capability/1.0"] = Field(
        "gfm.model-capability/1.0", alias="schemaVersion"
    )
    model_id: str = Field(alias="modelId", min_length=1, max_length=128)
    validated: bool = False
    serving_ready: bool = Field(False, alias="servingReady")
    supported_profiles: tuple[str, ...] = Field(default=(), alias="supportedProfiles")
    supported_modalities: tuple[FeatureModality, ...] = Field(default=(), alias="supportedModalities")
    temporal: bool = False
    tasks: tuple[str, ...] = ()
    max_nodes: int = Field(0, alias="maxNodes", ge=0)
    max_edges: int = Field(0, alias="maxEdges", ge=0)

    @model_validator(mode="after")
    def serving_requires_validation(self) -> InternalModelCapability:
        if self.serving_ready and not self.validated:
            raise ValueError("servingReady requires a validated model")
        return self


class InternalFindingEvidence(ContractModel):
    evidence_id: str = Field(alias="evidenceId", min_length=1)
    kind: str = Field(min_length=1)
    ref: str = Field(min_length=1)


class InternalGovernanceFinding(ContractModel):
    schema_version: Literal["gfm.governance-finding/1.0"] = Field(
        "gfm.governance-finding/1.0", alias="schemaVersion"
    )
    finding_id: str = Field(alias="findingId", min_length=1)
    task_id: str = Field(alias="taskId", min_length=1)
    target_ref: str = Field(alias="targetRef", min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    time_range: InternalTimeRange = Field(alias="timeRange")
    reason_codes: tuple[str, ...] = Field(alias="reasonCodes", min_length=1)
    evidence: tuple[InternalFindingEvidence, ...] = Field(min_length=1)
    graph_ref: InternalGraphSnapshotRef = Field(alias="graphRef")
    model_id: str = Field(alias="modelId", min_length=1)
    checkpoint_id: str = Field(alias="checkpointId", min_length=1)
    human_review_required: Literal[True] = Field(True, alias="humanReviewRequired")


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and Infinity are forbidden")
    if isinstance(value, dict):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_finite(nested)
