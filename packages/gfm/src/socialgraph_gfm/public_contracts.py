"""Canonical external wire contracts shared with the loopback API.

This module deliberately mirrors ``socialgraph-fm-api/app/gfm_schemas.py``. Offline
materialization and smoke-only records use explicitly named internal contracts elsewhere;
adapters are the only supported bridge between the two boundaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CollaborationProfile = Literal[
    "collaboration.actor-interaction/1.0",
    "collaboration.activity-hetero/1.0",
]
CoreTaskId = Literal[
    "core.community_health_observation",
    "core.newcomer_support",
    "core.coordination_review",
]


class GfmModel(BaseModel):
    """Strict, JSON-only contracts shared by the workbench and GFM runtime."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TimeRange(GfmModel):
    start: datetime | None = None
    end: datetime | None = None

    @model_validator(mode="after")
    def validate_order(self) -> TimeRange:
        for value in (self.start, self.end):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("timeRange 必须使用带时区的时间")
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("timeRange.start 不能晚于 end")
        return self


class FeatureManifest(GfmModel):
    id: str = Field(min_length=1, max_length=200)
    attribute: str = Field(min_length=1, max_length=1000)
    target: Literal["node", "edge", "graph"]
    modality: Literal["numeric", "categorical", "text", "boolean", "timestamp", "embedding"]
    dtype: str = Field(min_length=1, max_length=100)
    missing_policy: Literal["mask", "zero", "unknown_token", "reject"] = Field(
        alias="missingPolicy"
    )
    privacy_level: Literal["public", "project", "sensitive", "prohibited"] = Field(
        alias="privacyLevel"
    )
    fit_scope: Literal["none", "train_only", "all_nodes_transductive"] = Field(
        alias="fitScope", default="none"
    )
    available_at_field: str | None = Field(
        alias="availableAtField", default=None, max_length=1000
    )
    embedding_ref: str | None = Field(alias="embeddingRef", default=None, max_length=1000)
    inference_allowed: bool = Field(alias="inferenceAllowed", default=False)

    @model_validator(mode="after")
    def validate_inference_boundary(self) -> FeatureManifest:
        if self.privacy_level == "prohibited" and self.inference_allowed:
            raise ValueError("prohibited 特征不得进入推理 allowlist")
        if self.modality == "embedding" and self.embedding_ref is None:
            raise ValueError("embedding 特征必须引用不可变 embeddingRef")
        return self


class GraphSnapshotRef(GfmModel):
    schema_version: Literal["gfm-graph-snapshot-ref/1.0"] = Field(
        alias="schemaVersion", default="gfm-graph-snapshot-ref/1.0"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_fact_hash: str = Field(alias="graphFactHash", pattern=r"^[0-9a-f]{64}$")
    content_hash: str = Field(alias="contentHash", pattern=r"^[0-9a-f]{64}$")
    profile: CollaborationProfile
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    time_range: TimeRange | None = Field(alias="timeRange", default=None)
    feature_manifest: list[FeatureManifest] = Field(
        alias="featureManifest", default_factory=list, max_length=4096
    )


class CorpusManifest(GfmModel):
    schema_version: Literal["gfm-corpus-manifest/1.0"] = Field(
        alias="schemaVersion", default="gfm-corpus-manifest/1.0"
    )
    id: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=100)
    source_hash: str = Field(alias="sourceHash", pattern=r"^[0-9a-f]{64}$")
    license_id: str = Field(alias="licenseId", min_length=1, max_length=200)
    intended_use: Literal["synthetic_test_only", "pretraining", "adaptation", "evaluation"] = Field(
        alias="intendedUse"
    )
    split: Literal["synthetic", "temporal", "official", "held_out_domain"]
    adapter: str = Field(min_length=1, max_length=500)


class CoreTaskManifest(GfmModel):
    schema_version: Literal["socialgraph-fm.core-task/1.0"] = Field(
        alias="schemaVersion", default="socialgraph-fm.core-task/1.0"
    )
    task_id: CoreTaskId = Field(alias="taskId")
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2000)
    family: Literal["collaboration_governance"] = "collaboration_governance"
    enabled: Literal[False] = False
    required_profiles: list[CollaborationProfile] = Field(alias="requiredProfiles")
    requires_temporal_edges: bool = Field(alias="requiresTemporalEdges")
    output_kind: Literal["community_observation", "member_support", "review_queue"] = Field(
        alias="outputKind"
    )
    human_review_required: Literal[True] = Field(alias="humanReviewRequired", default=True)
    refusal_conditions: list[str] = Field(alias="refusalConditions", min_length=1)


class TrainingRunManifest(GfmModel):
    schema_version: Literal["gfm-training-run/1.0"] = Field(
        alias="schemaVersion", default="gfm-training-run/1.0"
    )
    id: str = Field(min_length=1, max_length=200)
    task_id: CoreTaskId = Field(alias="taskId")
    graph_snapshot: GraphSnapshotRef = Field(alias="graphSnapshot")
    corpus: list[CorpusManifest] = Field(default_factory=list)
    seed: int = Field(ge=0, le=2**32 - 1)
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "smoke"]
    code_hash: str = Field(alias="codeHash", pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(alias="environmentHash", pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    artifact_hashes: list[str] = Field(alias="artifactHashes", default_factory=list)
    created_at: datetime = Field(alias="createdAt")


class CheckpointManifest(GfmModel):
    schema_version: Literal["gfm-checkpoint/1.0"] = Field(
        alias="schemaVersion", default="gfm-checkpoint/1.0"
    )
    id: str = Field(min_length=1, max_length=200)
    run_id: str = Field(alias="runId", min_length=1, max_length=200)
    step: int = Field(ge=0)
    weights_hash: str = Field(alias="weightsHash", pattern=r"^[0-9a-f]{64}$")
    optimizer_hash: str = Field(alias="optimizerHash", pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    integrity_hash: str = Field(alias="integrityHash", pattern=r"^[0-9a-f]{64}$")
    registrable: Literal[False] = False


class ModelCapability(GfmModel):
    model_id: str = Field(alias="modelId", min_length=1, max_length=200)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=100)
    profiles: list[CollaborationProfile]
    tasks: list[CoreTaskId]
    modalities: list[Literal["numeric", "categorical", "text", "temporal", "structural"]]
    temporal: bool
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)
    validated: bool


class FindingEvidence(GfmModel):
    kind: Literal["node", "edge", "subgraph", "document", "metric"]
    ref: str = Field(min_length=1, max_length=2000)
    summary: str | None = Field(default=None, max_length=4000)


class CoreFinding(GfmModel):
    schema_version: Literal["socialgraph-fm.core-finding/1.0"] = Field(
        alias="schemaVersion", default="socialgraph-fm.core-finding/1.0"
    )
    id: str = Field(min_length=1, max_length=200)
    task_id: CoreTaskId = Field(alias="taskId")
    target_id: str = Field(alias="targetId", min_length=1, max_length=1000)
    score: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    time_range: TimeRange | None = Field(alias="timeRange", default=None)
    reason_codes: list[str] = Field(alias="reasonCodes", min_length=1, max_length=100)
    evidence: list[FindingEvidence] = Field(default_factory=list, max_length=1000)
    provenance: dict[str, Any]
    human_review_required: Literal[True] = Field(alias="humanReviewRequired", default=True)


class CompatibilityMapping(GfmModel):
    profile: CollaborationProfile
    node_type_map: dict[str, Literal["actor", "artifact", "community", "topic"]] = Field(
        alias="nodeTypeMap", default_factory=dict
    )
    edge_type_map: dict[
        str,
        Literal[
            "collaborates",
            "replies",
            "mentions",
            "reviews",
            "endorses",
            "actor_interacts_actor",
            "actor_contributes_artifact",
            "actor_joins_community",
            "artifact_belongs_community",
            "artifact_has_topic",
        ],
    ] = Field(alias="edgeTypeMap", default_factory=dict)
    timestamp_field: str | None = Field(alias="timestampField", default=None)
    inference_attribute_allowlist: list[str] = Field(
        alias="inferenceAttributeAllowlist", default_factory=list
    )
    privacy_level: Literal["public", "project", "sensitive"] = Field(alias="privacyLevel")
    deidentify: bool = True
    user_data_training_opt_in: Literal[False] = Field(
        alias="userDataTrainingOptIn", default=False
    )

    @model_validator(mode="after")
    def validate_profile_vocabulary(self) -> CompatibilityMapping:
        actor_relations = {"collaborates", "replies", "mentions", "reviews", "endorses"}
        hetero_relations = {
            "actor_interacts_actor",
            "actor_contributes_artifact",
            "actor_joins_community",
            "artifact_belongs_community",
            "artifact_has_topic",
        }
        mapped_relations = set(self.edge_type_map.values())
        mapped_nodes = set(self.node_type_map.values())
        if self.profile == "collaboration.actor-interaction/1.0":
            if mapped_nodes.difference({"actor"}):
                raise ValueError("actor-interaction profile 只允许 actor 节点")
            if mapped_relations.difference(actor_relations):
                raise ValueError("actor-interaction profile 包含不兼容的关系")
        elif mapped_relations.difference(hetero_relations):
            raise ValueError("activity-hetero profile 包含不兼容的关系")
        return self


class CompatibilityIssue(GfmModel):
    code: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)
    field: str | None = Field(default=None, max_length=1000)


class GfmCompatibilityReport(GfmModel):
    schema_version: Literal["gfm-compatibility-report/1.0"] = Field(
        alias="schemaVersion", default="gfm-compatibility-report/1.0"
    )
    compatible: bool
    mapping: CompatibilityMapping | None = None
    features: list[FeatureManifest] = Field(default_factory=list)
    blockers: list[CompatibilityIssue] = Field(default_factory=list)
    warnings: list[CompatibilityIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_outcome(self) -> GfmCompatibilityReport:
        if self.compatible and (self.mapping is None or self.blockers):
            raise ValueError("compatible report 必须有 mapping 且不能有 blocker")
        if not self.compatible and not self.blockers:
            raise ValueError("incompatible report 必须说明 blocker")
        return self


class GfmReadiness(GfmModel):
    workbench_input_ready: bool = Field(alias="workbenchInputReady")
    gfm_infrastructure_ready: bool = Field(alias="gfmInfrastructureReady")
    corpus_ready: bool = Field(alias="corpusReady")
    model_is_validated: bool = Field(alias="modelValidated")
    core_serving_ready: bool = Field(alias="coreServingReady")
    large_graph_ui_release_ready: bool = Field(alias="largeGraphUiReleaseReady")


class CoreCapabilitiesResponse(GfmModel):
    schema_version: Literal["gfm-capabilities/1.0"] = Field(
        alias="schemaVersion", default="gfm-capabilities/1.0"
    )
    serving_ready: Literal[False] = Field(alias="servingReady", default=False)
    supported_profiles: list[CollaborationProfile] = Field(alias="supportedProfiles")
    models: list[ModelCapability] = Field(default_factory=list)
    readiness: GfmReadiness
    user_data_training_opt_in: Literal[False] = Field(
        alias="userDataTrainingOptIn", default=False
    )


class GfmTasksResponse(GfmModel):
    schema_version: Literal["gfm-tasks/1.0"] = Field(
        alias="schemaVersion", default="gfm-tasks/1.0"
    )
    tasks: list[CoreTaskManifest]


class CoreRunRequest(GfmModel):
    """Reserved public request; validation never implies a model is installed."""

    schema_version: Literal["gfm-run-request/1.0"] = Field(
        alias="schemaVersion", default="gfm-run-request/1.0"
    )
    task_id: CoreTaskId | None = Field(alias="taskId", default=None)
    graph_snapshot: GraphSnapshotRef | None = Field(alias="graphSnapshot", default=None)


PUBLIC_CONTRACTS: tuple[type[GfmModel], ...] = (
    TimeRange,
    FeatureManifest,
    GraphSnapshotRef,
    CorpusManifest,
    CoreTaskManifest,
    TrainingRunManifest,
    CheckpointManifest,
    ModelCapability,
    FindingEvidence,
    CoreFinding,
    CompatibilityMapping,
    CompatibilityIssue,
    GfmCompatibilityReport,
    GfmReadiness,
    CoreCapabilitiesResponse,
    GfmTasksResponse,
    CoreRunRequest,
)


def public_contract_schemas() -> dict[str, Any]:
    return {
        model.__name__: model.model_json_schema(by_alias=True)
        for model in PUBLIC_CONTRACTS
    }
