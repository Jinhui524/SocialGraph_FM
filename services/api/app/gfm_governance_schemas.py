"""Public contracts for SocialGraph-FM Governance online inference."""

from __future__ import annotations

import math
import hashlib
import json
from datetime import datetime
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .gfm_hashing import canonical_sha256
from .governance_skill_runtime.catalog import load_product_skill_catalog

GOVERNANCE_SCHEMA_VERSION = "socialgraph-fm.gfm-governance/2.0"
GOVERNANCE_INPUT_SCHEMA_VERSION = "socialgraph-fm.governance-input/2.0"
GOVERNANCE_CHANNEL = "governance"
GOVERNANCE_PROTOCOL = "global"
GOVERNANCE_MODALITIES = ("coRT", "coURL", "hashSeq", "fastRT", "tweetSim")
GOVERNANCE_PUBLIC_SKILLS = load_product_skill_catalog().names

HASH_PATTERN = r"^[0-9a-f]{64}$"
ARTIFACT_PATTERN = r"^governance-artifact-[0-9a-f]{32}$"
RUN_PATTERN = r"^governance-[0-9a-f]{32}$"
CASE_PATTERN = r"^case-[0-9a-f]{32}$"

GovernanceModalityV2 = Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"]
GovernancePublicSkillV2 = Literal[
    "inspect_graph",
    "run_governance_analysis",
    "get_evidence_subgraph",
    "discover_coordination_groups",
    "rank_coordination_relations",
    "retrieve_similar_cases",
    "get_model_dataset_cards",
    "draft_review_report",
]
if tuple(get_args(GovernancePublicSkillV2)) != GOVERNANCE_PUBLIC_SKILLS:
    raise RuntimeError("SocialGraph-FM Governance capability skills do not match the canonical catalog")
RunStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
RunStage = Literal[
    "queued",
    "validating",
    "preprocessing",
    "inferencing",
    "deriving",
    "freezing",
    "completed",
]
CaseState = Literal["draft", "active", "concluded", "archived"]
TargetType = Literal["node", "relation", "group"]
ReviewDecision = Literal["confirmed", "rejected", "pending"]
NonNegativeInt = Annotated[int, Field(ge=0)]
GovernanceExpertV2 = Literal[
    "shared",
    "domain:china",
    "domain:cuba",
    "domain:iran",
    "domain:russia",
    "domain:UAE",
    "domain:venezuela",
    "null",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
    )


class ManifestFile(FrozenModel):
    sha256: str = Field(pattern=HASH_PATTERN)
    bytes: int = Field(ge=1, le=512 * 1024 * 1024)


class GovernanceInputManifest(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-input/2.0"] = Field(
        alias="schemaVersion"
    )
    dataset_id: str = Field(alias="datasetId", pattern=r"^[A-Za-z0-9._:-]{1,100}$")
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)
    node_count: int = Field(alias="nodeCount", ge=1, le=10_000)
    relation_row_count: int = Field(alias="relationRowCount", ge=1, le=500_000)
    feature_dimension: Literal[768] = Field(alias="featureDimension")
    modalities: tuple[GovernanceModalityV2, ...] = Field(min_length=1, max_length=5)
    files: dict[str, ManifestFile]
    license: str | None = Field(default=None, min_length=1, max_length=200)
    source_uri: str | None = Field(
        default=None, alias="sourceUri", min_length=1, max_length=2_048
    )

    @model_validator(mode="after")
    def validate_files_and_modalities(self) -> GovernanceInputManifest:
        if set(self.files) != {"nodes.csv", "relations.csv", "features.npz"}:
            raise ValueError("manifest files must name the three contract files exactly")
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("manifest modalities must be unique")
        positions = [GOVERNANCE_MODALITIES.index(value) for value in self.modalities]
        if positions != sorted(positions):
            raise ValueError("manifest modalities must use canonical order")
        if any(ord(character) < 32 for character in self.display_name):
            raise ValueError("displayName contains a control character")
        return self


class GovernanceLimits(FrozenModel):
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_relation_rows: int = Field(alias="maxRelationRows", ge=1)
    max_evidence_nodes: int = Field(alias="maxEvidenceNodes", ge=1)
    max_evidence_edges: int = Field(alias="maxEvidenceEdges", ge=1)
    max_preview_nodes: int = Field(alias="maxPreviewNodes", ge=1)
    max_preview_edges: int = Field(alias="maxPreviewEdges", ge=1)


class GovernanceCapabilities(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    channel: Literal["governance"]
    task_id: Literal["coordination_risk"] = Field(alias="taskId")
    serving_ready: bool = Field(alias="servingReady")
    online_forward_ready: bool = Field(alias="onlineForwardReady")
    unavailable_reason: str | None = Field(default=None, alias="unavailableReason")
    model_version_id: str | None = Field(default=None, alias="modelVersionId")
    model_version_hash: str | None = Field(
        default=None, alias="modelVersionHash", pattern=HASH_PATTERN
    )
    model_state_hash: str | None = Field(
        default=None, alias="modelStateHash", pattern=HASH_PATTERN
    )
    supported_protocols: tuple[Literal["global"], ...] = Field(
        alias="supportedProtocols"
    )
    skills: tuple[GovernancePublicSkillV2, ...]
    input_schema_version: Literal["socialgraph-fm.governance-input/2.0"] = Field(
        alias="inputSchemaVersion"
    )
    modalities: tuple[GovernanceModalityV2, ...]
    sample_artifact_id: str | None = Field(
        default=None, alias="sampleArtifactId", pattern=ARTIFACT_PATTERN
    )
    limits: GovernanceLimits
    capability_hash: str = Field(alias="capabilityHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self) -> GovernanceCapabilities:
        if self.supported_protocols != ("global",):
            raise ValueError("only Global online inference is supported")
        if self.skills != GOVERNANCE_PUBLIC_SKILLS:
            raise ValueError("capabilities must expose the exact public Skills registry")
        if self.modalities != GOVERNANCE_MODALITIES:
            raise ValueError("capabilities must expose the complete modality contract")
        if self.serving_ready != self.online_forward_ready:
            raise ValueError("servingReady must reflect online forward readiness")
        if self.serving_ready and not all(
            (self.model_version_id, self.model_version_hash, self.model_state_hash)
        ):
            raise ValueError("ready capabilities require the complete model identity")
        _validate_hash(self, "capability_hash")
        return self


class GovernanceHealth(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    service_identity: str = Field(alias="serviceIdentity", pattern=HASH_PATTERN)
    serving_ready: bool = Field(alias="servingReady")
    online_forward_ready: bool = Field(alias="onlineForwardReady")
    model_version_id: str | None = Field(default=None, alias="modelVersionId")
    model_version_hash: str | None = Field(
        default=None, alias="modelVersionHash", pattern=HASH_PATTERN
    )
    model_state_hash: str | None = Field(
        default=None, alias="modelStateHash", pattern=HASH_PATTERN
    )
    device: Literal["cpu", "cuda"]
    dtype: Literal["float32", "float16", "bfloat16"]
    loaded_at: datetime | None = Field(default=None, alias="loadedAt")
    queue_depth: int = Field(alias="queueDepth", ge=0)
    active_run_id: str | None = Field(default=None, alias="activeRunId", pattern=RUN_PATTERN)
    runtime_recipe_hash: str = Field(alias="runtimeRecipeHash", pattern=HASH_PATTERN)
    health_hash: str = Field(alias="healthHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self) -> GovernanceHealth:
        if self.serving_ready != self.online_forward_ready:
            raise ValueError("servingReady must reflect online forward readiness")
        _validate_hash(self, "health_hash")
        return self


class GovernanceArtifactReceipt(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_id: str = Field(alias="datasetId")
    display_name: str = Field(alias="displayName")
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=HASH_PATTERN)
    manifest_sha256: str = Field(alias="manifestSha256", pattern=HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1, le=10_000)
    relation_row_count: int = Field(alias="relationRowCount", ge=1, le=500_000)
    self_loops_removed: int = Field(alias="selfLoopsRemoved", ge=0)
    clean_self_loops: bool = Field(alias="cleanSelfLoops")
    modalities: tuple[GovernanceModalityV2, ...]
    compatibility: Literal["compatible"]
    created_at: datetime = Field(alias="createdAt")
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> GovernanceArtifactReceipt:
        _validate_hash(self, "artifact_hash")
        return self


class GovernanceArtifactCompatibility(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    input_schema_version: Literal["socialgraph-fm.governance-input/2.0"] = Field(
        alias="inputSchemaVersion"
    )
    compatible: bool
    requires_self_loop_cleaning: bool = Field(alias="requiresSelfLoopCleaning")
    prospective_artifact_id: str = Field(
        alias="prospectiveArtifactId", pattern=ARTIFACT_PATTERN
    )
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1, le=10_000)
    relation_row_count: int = Field(alias="relationRowCount", ge=1, le=500_000)
    self_loops_detected: int = Field(alias="selfLoopsDetected", ge=0)
    modalities: tuple[GovernanceModalityV2, ...]
    issues: tuple[str, ...]
    compatibility_hash: str = Field(alias="compatibilityHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> GovernanceArtifactCompatibility:
        if self.requires_self_loop_cleaning != (self.self_loops_detected > 0):
            raise ValueError("self-loop compatibility state is inconsistent")
        _validate_hash(self, "compatibility_hash")
        return self


class GovernanceArtifact(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1, le=10_000)
    relation_row_count: int = Field(alias="relationRowCount", ge=1, le=500_000)
    self_loops_removed: int = Field(alias="selfLoopsRemoved", ge=0)
    modalities: tuple[GovernanceModalityV2, ...]
    created_at: datetime = Field(alias="createdAt")
    compatibility: Literal["compatible"]
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> GovernanceArtifact:
        _validate_hash(self, "artifact_hash")
        return self


class GovernanceArtifactList(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    items: tuple[GovernanceArtifactReceipt, ...]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=100)


class GovernancePreviewQuery(FrozenModel):
    preset: Literal["overview", "relation", "evidence", "groups"]
    node_budget: int | None = Field(default=None, alias="nodeBudget", ge=1, le=3_000)
    edge_budget: int | None = Field(default=None, alias="edgeBudget", ge=0, le=12_000)
    relation: GovernanceModalityV2 | None = None
    anchor_node_ids: tuple[str, ...] = Field(
        default=(), alias="anchorNodeIds", max_length=8
    )
    group_budget: int | None = Field(default=None, alias="groupBudget", ge=1, le=24)

    @model_validator(mode="after")
    def validate_preset(self) -> GovernancePreviewQuery:
        limits = {"overview": (3_000, 12_000), "relation": (80, 160), "evidence": (60, 120)}
        if len(set(self.anchor_node_ids)) != len(self.anchor_node_ids):
            raise ValueError("anchorNodeIds must be unique")
        if self.preset == "groups":
            if (
                self.node_budget is not None
                or self.edge_budget is not None
                or self.relation is not None
                or self.anchor_node_ids
            ):
                raise ValueError("groups accepts groupBudget only")
            return self
        max_nodes, max_edges = limits[self.preset]
        if (self.node_budget or 1) > max_nodes or (
            self.edge_budget is not None and self.edge_budget > max_edges
        ):
            raise ValueError("preview budget exceeds preset limits")
        if self.group_budget is not None:
            raise ValueError("groupBudget is valid only for groups")
        if (self.preset == "relation") != (self.relation is not None):
            raise ValueError("relation is required only for relation preset")
        if self.preset != "evidence" and self.anchor_node_ids:
            raise ValueError("anchorNodeIds are valid only for evidence")
        return self


class GovernancePreviewNode(FrozenModel):
    id: str = Field(min_length=1, max_length=128)
    label: str = Field(min_length=1, max_length=256)
    degree: int = Field(ge=0)
    structure_missing: bool = Field(alias="structureMissing")
    score: float | None = Field(default=None, ge=0, le=1)
    risk_band: Literal["low", "review", "high"] | None = Field(
        default=None, alias="riskBand"
    )
    group_id: str | None = Field(default=None, alias="groupId")


class GovernancePreviewEdge(FrozenModel):
    id: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)
    modalities: tuple[GovernanceModalityV2, ...]
    factual: bool = True


class GovernanceGroupSupernode(GovernancePreviewNode):
    aggregate: Literal[True]
    member_count: int = Field(alias="memberCount", ge=1, le=10_000)
    risk_p90: float = Field(alias="riskP90", ge=0, le=1)
    mean_risk: float = Field(alias="meanRisk", ge=0, le=1)


class GovernanceAggregateEdge(GovernancePreviewEdge):
    aggregate: Literal[True]
    count: int = Field(ge=1)
    weight: int = Field(ge=1)


class GovernancePreviewRelationCounts(FrozenModel):
    co_rt: int = Field(alias="coRT", ge=0)
    co_url: int = Field(alias="coURL", ge=0)
    hash_seq: int = Field(alias="hashSeq", ge=0)
    fast_rt: int = Field(alias="fastRT", ge=0)
    tweet_sim: int = Field(alias="tweetSim", ge=0)


class GovernancePreviewGroup(FrozenModel):
    group_id: str = Field(alias="groupId", min_length=1, max_length=128)
    member_count: int = Field(alias="memberCount", ge=2, le=10_000)
    member_node_ids: tuple[str, ...] = Field(alias="memberNodeIds", max_length=10_000)
    average_risk: float = Field(alias="averageRisk", ge=0, le=1)
    p90_risk: float = Field(alias="p90Risk", ge=0, le=1)
    priority: float = Field(ge=0, le=1)
    relation_counts: GovernancePreviewRelationCounts = Field(alias="relationCounts")
    derivation: str = Field(min_length=1, max_length=500)
    rank: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_members(self) -> GovernancePreviewGroup:
        if self.member_count != len(self.member_node_ids) or len(set(self.member_node_ids)) != len(
            self.member_node_ids
        ):
            raise ValueError("group member inventory mismatch")
        return self


class GovernancePreviewNodeEdgeBudgets(FrozenModel):
    nodes: int = Field(ge=1, le=3_000)
    edges: int = Field(ge=0, le=12_000)


class GovernancePreviewGroupBudgets(FrozenModel):
    groups: int = Field(ge=1, le=24)
    max_groups: Literal[24] = Field(alias="maxGroups")


class GovernancePreviewInventoryCounts(FrozenModel):
    nodes: int = Field(ge=1, le=10_000)
    edges: int = Field(ge=0)
    groups: int = Field(ge=0)


class GovernancePreviewOverviewSources(FrozenModel):
    group_representatives: int = Field(alias="groupRepresentatives", ge=0)
    component_representatives: int = Field(alias="componentRepresentatives", ge=0)
    bridge_endpoints: int = Field(alias="bridgeEndpoints", ge=0)
    isolates: int = Field(ge=0)
    high_risk: int = Field(alias="highRisk", ge=0)
    review_risk: int = Field(alias="reviewRisk", ge=0)
    low_risk: int = Field(alias="lowRisk", ge=0)
    high_degree: int = Field(alias="highDegree", ge=0)
    mid_degree: int = Field(alias="midDegree", ge=0)
    low_degree: int = Field(alias="lowDegree", ge=0)
    ranked_fill: int = Field(alias="rankedFill", ge=0)


class GovernancePreviewRelationSources(FrozenModel):
    relation_endpoints: int = Field(alias="relationEndpoints", ge=0)
    relation_edges: int = Field(alias="relationEdges", ge=0)


class GovernancePreviewEvidenceSources(FrozenModel):
    anchors: int = Field(ge=0)
    neighbors: int = Field(ge=0)
    induced_edges: int = Field(alias="inducedEdges", ge=0)


class GovernancePreviewGroupSources(FrozenModel):
    group_supernodes: int = Field(alias="groupSupernodes", ge=0)
    inter_group_edges: int = Field(alias="interGroupEdges", ge=0)


class GovernanceGraphPreview(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    run_id: str | None = Field(default=None, alias="runId", pattern=RUN_PATTERN)
    result_hash: str | None = Field(default=None, alias="resultHash", pattern=HASH_PATTERN)
    nodes: tuple[GovernancePreviewNode | GovernanceGroupSupernode, ...]
    edges: tuple[GovernancePreviewEdge | GovernanceAggregateEdge, ...]
    node_count: int = Field(alias="nodeCount", ge=1)
    edge_count: int = Field(alias="edgeCount", ge=1)
    partial_preview: bool = Field(alias="partialPreview")
    preset: Literal["overview", "relation", "evidence", "groups"] | None = None
    budgets: GovernancePreviewNodeEdgeBudgets | GovernancePreviewGroupBudgets | None = None
    selection_recipe_id: str | None = Field(
        default=None, alias="selectionRecipeId", min_length=1, max_length=200
    )
    is_partial: bool | None = Field(default=None, alias="isPartial")
    groups: tuple[GovernancePreviewGroup, ...] | None = None
    source_counts: (
        GovernancePreviewOverviewSources
        | GovernancePreviewRelationSources
        | GovernancePreviewEvidenceSources
        | GovernancePreviewGroupSources
        | None
    ) = Field(default=None, alias="sourceCounts")
    inventory_counts: GovernancePreviewInventoryCounts | None = Field(
        default=None, alias="inventoryCounts"
    )
    preview_hash: str = Field(alias="previewHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> GovernanceGraphPreview:
        if (self.run_id is None) != (self.result_hash is None):
            raise ValueError("run preview identity is incomplete")
        projection_values = (
            self.budgets,
            self.selection_recipe_id,
            self.is_partial,
            self.groups,
            self.source_counts,
            self.inventory_counts,
        )
        if self.preset is None:
            if any(value is not None for value in projection_values) or any(
                isinstance(node, GovernanceGroupSupernode) for node in self.nodes
            ) or any(isinstance(edge, GovernanceAggregateEdge) for edge in self.edges):
                raise ValueError("projection metadata requires preset")
        else:
            if any(value is None for value in projection_values):
                raise ValueError("projection metadata is incomplete")
            assert self.inventory_counts is not None
            if (
                self.inventory_counts.nodes != self.node_count
                or self.inventory_counts.edges != self.edge_count
                or self.is_partial != self.partial_preview
            ):
                raise ValueError("projection inventory mismatch")
            expected_source = {
                "overview": GovernancePreviewOverviewSources,
                "relation": GovernancePreviewRelationSources,
                "evidence": GovernancePreviewEvidenceSources,
                "groups": GovernancePreviewGroupSources,
            }[self.preset]
            if not isinstance(self.source_counts, expected_source):
                raise ValueError("sourceCounts do not match preset")
            group_projection = self.preset == "groups"
            if group_projection != isinstance(self.budgets, GovernancePreviewGroupBudgets):
                raise ValueError("budgets do not match preset")
            invalid_types = (
                not all(isinstance(node, GovernanceGroupSupernode) for node in self.nodes)
                or not all(isinstance(edge, GovernanceAggregateEdge) for edge in self.edges)
                if group_projection
                else any(isinstance(node, GovernanceGroupSupernode) for node in self.nodes)
                or any(isinstance(edge, GovernanceAggregateEdge) for edge in self.edges)
            )
            if invalid_types:
                raise ValueError("projection node or edge type mismatch")
        payload = self.model_dump(mode="json", by_alias=True, exclude={"preview_hash"})
        if self.preset is None:
            for field in (
                "preset",
                "budgets",
                "selectionRecipeId",
                "isPartial",
                "groups",
                "sourceCounts",
                "inventoryCounts",
            ):
                payload.pop(field, None)
        if canonical_sha256(payload) != self.preview_hash:
            raise ValueError("previewHash mismatch")
        return self


class OnlineRunRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    protocol: Literal["global"]
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)
    top_k: int = Field(default=100, alias="topK", ge=1, le=10_000)

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="json", by_alias=True))


class OnlineRunStatus(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)
    status: RunStatus
    stage: RunStage
    progress: int = Field(ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    error_code: str | None = Field(default=None, alias="errorCode")
    cancel_requested: bool = Field(default=False, alias="cancelRequested")
    status_hash: str = Field(alias="statusHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> OnlineRunStatus:
        _validate_hash(self, "status_hash")
        return self


class GovernanceRouteWeight(FrozenModel):
    expert: GovernanceExpertV2
    weight: float = Field(ge=0, le=1)


class GovernanceFindingV2(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    label: str | None = Field(default=None, max_length=256)
    score: float = Field(ge=0, le=1)
    logit: float
    rank: int = Field(ge=1)
    risk_band: Literal["low", "review", "high"] = Field(alias="riskBand")
    predicted_positive: bool = Field(alias="predictedPositive")
    structure_missing: bool = Field(alias="structureMissing")
    routes: tuple[GovernanceRouteWeight, ...] = Field(min_length=3, max_length=3)
    modality_contribution: GovernanceModalityContribution = Field(
        alias="modalityContribution"
    )
    modality_evidence: dict[GovernanceModalityV2, NonNegativeInt] = Field(
        alias="modalityEvidence"
    )
    community_id: str | None = Field(default=None, alias="communityId")

    @field_validator("logit")
    @classmethod
    def finite_logit(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("logit must be finite")
        return value

    @model_validator(mode="after")
    def validate_modalities(self) -> GovernanceFindingV2:
        if set(self.modality_evidence) != set(GOVERNANCE_MODALITIES):
            raise ValueError("modalityEvidence must cover all five modalities")
        if (
            self.routes[0].expert != "shared"
            or self.routes[0].weight != 1
            or len({route.expert for route in self.routes}) != 3
            or not math.isclose(
                self.routes[1].weight + self.routes[2].weight,
                1.0,
                rel_tol=0,
                abs_tol=1e-5,
            )
            or (self.risk_band == "high") != self.predicted_positive
        ):
            raise ValueError("finding route or risk-band contract is inconsistent")
        return self


class GovernanceModalityContribution(FrozenModel):
    text: float = Field(ge=0, le=1)
    structure: float = Field(ge=0, le=1)


class GovernanceCalibration(FrozenModel):
    temperature: float = Field(gt=0)
    bias: float
    reference_threshold: float = Field(alias="referenceThreshold", ge=0, le=1)
    applicability: Literal["reference_replay", "out_of_domain_unverified"]


class GovernanceDistribution(FrozenModel):
    low: int = Field(ge=0)
    review: int = Field(ge=0)
    high: int = Field(ge=0)
    predicted_positive: int = Field(alias="predictedPositive", ge=0)
    total: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_total(self) -> GovernanceDistribution:
        if self.low + self.review + self.high != self.total:
            raise ValueError("risk distribution does not sum to total")
        if self.predicted_positive > self.total:
            raise ValueError("predictedPositive exceeds total")
        if self.predicted_positive != self.high:
            raise ValueError("only high-band nodes may be predicted positive")
        return self


class OnlineRunResult(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId")
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)
    threshold: float = Field(ge=0, le=1)
    calibration: GovernanceCalibration
    reference_metrics: dict[str, Any] = Field(alias="referenceMetrics")
    dataset_metrics: None = Field(alias="datasetMetrics")
    distribution: GovernanceDistribution
    findings: tuple[GovernanceFindingV2, ...]
    total_findings: int = Field(alias="totalFindings", ge=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    completed_at: datetime = Field(alias="completedAt")
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> OnlineRunResult:
        if (
            len(self.findings) > self.total_findings
            or len({item.node_id for item in self.findings}) != len(self.findings)
            or tuple(item.rank for item in self.findings)
            != tuple(range(1, len(self.findings) + 1))
        ):
            raise ValueError("topK findings exceed totalFindings")
        _validate_hash(self, "result_hash")
        return self


class PageInfo(FrozenModel):
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=10_000)


class FindingsPage(PageInfo):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    items: tuple[GovernanceFindingV2, ...]
    page_hash: str = Field(alias="pageHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> FindingsPage:
        if (
            len({item.node_id for item in self.items}) != len(self.items)
            or tuple(item.rank for item in self.items)
            != tuple(range(self.offset + 1, self.offset + len(self.items) + 1))
        ):
            raise ValueError("findings page rank inventory is invalid")
        _validate_hash(self, "page_hash")
        return self


class DerivationItem(FrozenModel):
    id: str = Field(min_length=1, max_length=300)
    kind: Literal["group", "factual_relation", "potential_link"]
    priority: float = Field(ge=0, le=1)
    node_ids: tuple[str, ...] = Field(alias="nodeIds", min_length=1)
    source: str | None = Field(default=None, max_length=128)
    target: str | None = Field(default=None, max_length=128)
    modalities: tuple[GovernanceModalityV2, ...] = ()
    member_count: int | None = Field(default=None, alias="memberCount", ge=1)
    mean_score: float | None = Field(default=None, alias="meanScore", ge=0, le=1)
    p90_score: float | None = Field(default=None, alias="p90Score", ge=0, le=1)
    score_components: dict[str, float] = Field(default_factory=dict, alias="scoreComponents")
    factual: bool
    limitation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_kind(self) -> DerivationItem:
        expected_factual = self.kind == "factual_relation"
        if self.factual != expected_factual:
            raise ValueError("derivation factual flag disagrees with kind")
        if self.kind == "group":
            if (
                self.source is not None
                or self.target is not None
                or self.member_count != len(self.node_ids)
                or self.member_count < 2
                or self.mean_score is None
                or self.p90_score is None
            ):
                raise ValueError("group derivation inventory is invalid")
        elif (
            self.source is None
            or self.target is None
            or self.source == self.target
            or self.node_ids != (self.source, self.target)
            or self.member_count is not None
            or self.mean_score is not None
            or self.p90_score is not None
            or (self.kind == "factual_relation" and not self.modalities)
        ):
            raise ValueError("relation/link derivation inventory is invalid")
        if any(not math.isfinite(value) for value in self.score_components.values()):
            raise ValueError("scoreComponents must be finite")
        return self


class DerivationPage(PageInfo):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    items: tuple[DerivationItem, ...]
    page_hash: str = Field(alias="pageHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> DerivationPage:
        _validate_hash(self, "page_hash")
        return self


class EvidenceRelationV2(FrozenModel):
    modality: GovernanceModalityV2
    raw_weight: float = Field(alias="rawWeight", ge=0, allow_inf_nan=False)


class EvidenceNeighborV2(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    score: float = Field(ge=0, le=1)
    hop: Literal[1]
    risk_band: Literal["low", "review", "high"] = Field(alias="riskBand")
    predicted_positive: bool = Field(alias="predictedPositive")
    structure_missing: bool = Field(alias="structureMissing")
    modalities: tuple[GovernanceModalityV2, ...] = Field(min_length=1, max_length=5)
    relations: tuple[EvidenceRelationV2, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_relations(self) -> EvidenceNeighborV2:
        if self.modalities != tuple(item.modality for item in self.relations):
            raise ValueError("neighbor modalities do not match relation evidence")
        if len(set(self.modalities)) != len(self.modalities) or tuple(
            sorted(self.modalities, key=GOVERNANCE_MODALITIES.index)
        ) != self.modalities:
            raise ValueError("neighbor modalities must be unique and canonical")
        if (self.risk_band == "high") != self.predicted_positive:
            raise ValueError("neighbor risk band disagrees with predictedPositive")
        return self


class EvidenceStructuralSignalsV2(FrozenModel):
    fused_degree: int = Field(alias="fusedDegree", ge=0, le=9_999)
    structure_missing: bool = Field(alias="structureMissing")
    relation_neighbor_counts: dict[GovernanceModalityV2, NonNegativeInt] = Field(
        alias="relationNeighborCounts"
    )
    two_hop_node_count: int = Field(alias="twoHopNodeCount", ge=0, le=9_999)
    relation_evidence_role: Literal["explanationOnly"] = Field(
        alias="relationEvidenceRole"
    )

    @model_validator(mode="after")
    def validate_modalities(self) -> EvidenceStructuralSignalsV2:
        if set(self.relation_neighbor_counts) != set(GOVERNANCE_MODALITIES):
            raise ValueError("relationNeighborCounts must cover all five modalities")
        return self


class EvidenceSubgraphNodeV2(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    score: float = Field(ge=0, le=1)
    hop: Literal[0, 1, 2]
    risk_band: Literal["low", "review", "high"] = Field(alias="riskBand")
    predicted_positive: bool = Field(alias="predictedPositive")
    structure_missing: bool = Field(alias="structureMissing")

    @model_validator(mode="after")
    def validate_band(self) -> EvidenceSubgraphNodeV2:
        if (self.risk_band == "high") != self.predicted_positive:
            raise ValueError("subgraph risk band disagrees with predictedPositive")
        return self


class EvidenceSubgraphEdgeV2(FrozenModel):
    id: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=128)
    target: str = Field(min_length=1, max_length=128)
    relations: tuple[EvidenceRelationV2, ...] = Field(min_length=1, max_length=5)
    evidence_role: Literal["explanationOnly"] = Field(alias="evidenceRole")

    @model_validator(mode="after")
    def validate_relations(self) -> EvidenceSubgraphEdgeV2:
        modalities = tuple(item.modality for item in self.relations)
        if len(set(modalities)) != len(modalities) or tuple(
            sorted(modalities, key=GOVERNANCE_MODALITIES.index)
        ) != modalities:
            raise ValueError("subgraph relations must be unique and canonical")
        return self


class EvidenceSubgraphV2(FrozenModel):
    depth: Literal[2]
    node_count: int = Field(alias="nodeCount", ge=1, le=300)
    edge_count: int = Field(alias="edgeCount", ge=0, le=1_000)
    truncated: bool
    nodes: tuple[EvidenceSubgraphNodeV2, ...] = Field(min_length=1, max_length=300)
    edges: tuple[EvidenceSubgraphEdgeV2, ...] = Field(max_length=1_000)

    @model_validator(mode="after")
    def validate_inventory(self) -> EvidenceSubgraphV2:
        node_ids = {item.node_id for item in self.nodes}
        if (
            self.node_count != len(self.nodes)
            or self.edge_count != len(self.edges)
            or len(node_ids) != len(self.nodes)
            or sum(item.hop == 0 for item in self.nodes) != 1
            or any(edge.source not in node_ids or edge.target not in node_ids for edge in self.edges)
            or len({edge.id for edge in self.edges}) != len(self.edges)
        ):
            raise ValueError("evidence subgraph inventory is inconsistent")
        return self


class NodeEvidenceV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)
    threshold: float = Field(ge=0, le=1)
    node: GovernanceFindingV2
    neighbors: tuple[EvidenceNeighborV2, ...]
    structural_signals: EvidenceStructuralSignalsV2 = Field(alias="structuralSignals")
    evidence_subgraph: EvidenceSubgraphV2 = Field(alias="evidenceSubgraph")
    truncated: bool
    limitation: str = Field(min_length=1, max_length=1_000)
    evidence_hash: str = Field(alias="evidenceHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> NodeEvidenceV2:
        roots = [item for item in self.evidence_subgraph.nodes if item.hop == 0]
        if (
            roots[0].node_id != self.node.node_id
            or roots[0].score != self.node.score
            or self.truncated != self.evidence_subgraph.truncated
            or self.structural_signals.structure_missing != self.node.structure_missing
            or self.structural_signals.fused_degree < len(self.neighbors)
            or any(
                neighbor.node_id not in {item.node_id for item in self.evidence_subgraph.nodes}
                for neighbor in self.neighbors
            )
        ):
            raise ValueError("evidence payload is internally inconsistent")
        _validate_hash(self, "evidence_hash")
        return self


class RunList(PageInfo):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    items: tuple[OnlineRunStatus, ...]


class RunComparisonNode(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    left_score: float = Field(alias="leftScore", ge=0, le=1)
    right_score: float = Field(alias="rightScore", ge=0, le=1)
    score_delta: float = Field(alias="scoreDelta", ge=-1, le=1)
    left_rank: int = Field(alias="leftRank", ge=1)
    right_rank: int = Field(alias="rightRank", ge=1)
    rank_delta: int = Field(alias="rankDelta")
    risk_band_changed: bool = Field(alias="riskBandChanged")


class RunComparison(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    left_run_id: str = Field(alias="leftRunId", pattern=RUN_PATTERN)
    right_run_id: str = Field(alias="rightRunId", pattern=RUN_PATTERN)
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    compared_nodes: int = Field(alias="comparedNodes", ge=1, le=10_000)
    changes: tuple[RunComparisonNode, ...]
    group_summary: dict[str, int] = Field(alias="groupSummary")
    review_summary: dict[str, int] = Field(alias="reviewSummary")
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self) -> RunComparison:
        _validate_hash(self, "comparison_hash")
        return self


class CaseCreateRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)


class CaseTransitionRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    state: CaseState
    reason: str = Field(min_length=1, max_length=1_000)


class CaseItemRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    target_type: TargetType = Field(alias="targetType")
    target_id: str = Field(alias="targetId", min_length=1, max_length=300)
    note: str = Field(default="", max_length=2_000)


class ReviewEventRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    target_type: TargetType = Field(alias="targetType")
    target_id: str = Field(alias="targetId", min_length=1, max_length=300)
    decision: ReviewDecision
    reason: str = Field(min_length=1, max_length=2_000)
    actor: str = Field(default="local-analyst", min_length=1, max_length=100)


class CaseItem(FrozenModel):
    item_id: str = Field(alias="itemId", pattern=r"^item-[0-9a-f]{32}$")
    target_type: TargetType = Field(alias="targetType")
    target_id: str = Field(alias="targetId")
    note: str
    created_at: datetime = Field(alias="createdAt")
    item_hash: str = Field(alias="itemHash", pattern=HASH_PATTERN)


class ReviewEvent(FrozenModel):
    event_id: str = Field(alias="eventId", pattern=r"^event-[0-9a-f]{32}$")
    target_type: TargetType = Field(alias="targetType")
    target_id: str = Field(alias="targetId")
    decision: ReviewDecision
    reason: str
    actor: str
    sequence: int = Field(ge=1)
    created_at: datetime = Field(alias="createdAt")
    previous_event_hash: str | None = Field(
        alias="previousEventHash", pattern=HASH_PATTERN
    )
    event_hash: str = Field(alias="eventHash", pattern=HASH_PATTERN)


class GovernanceCase(FrozenModel):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    case_id: str = Field(alias="caseId", pattern=CASE_PATTERN)
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    title: str
    description: str
    state: CaseState
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    items: tuple[CaseItem, ...]
    review_events: tuple[ReviewEvent, ...] = Field(alias="reviewEvents")
    current_decisions: dict[str, ReviewDecision] = Field(alias="currentDecisions")
    case_hash: str = Field(alias="caseHash", pattern=HASH_PATTERN)


class CaseList(PageInfo):
    schema_version: Literal["socialgraph-fm.gfm-governance/2.0"] = Field(
        alias="schemaVersion"
    )
    items: tuple[GovernanceCase, ...]


class AdaptationBinding(FrozenModel):
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    run_artifact_hash: str = Field(alias="runArtifactHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)
    recipe_hash: str = Field(alias="recipeHash", pattern=HASH_PATTERN)
    code_hash: str = Field(alias="codeHash", pattern=HASH_PATTERN)
    seed: int = Field(ge=0, le=2**63 - 1)


class TargetLabelSelectionRecipe(FrozenModel):
    version: Literal["graph-fused-degree-quartile-stable-hash-v2"]
    stratification: Literal["graph-fused-degree-rank-quartile"]
    structural_strata: Literal[4] = Field(alias="structuralStrata")
    labels_per_class: Literal[8] = Field(alias="labelsPerClass")
    labels_per_class_per_stratum: Literal[2] = Field(alias="labelsPerClassPerStratum")
    score_inputs: tuple[()] = Field(alias="scoreInputs")


class TargetPackageReceipt(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-package-receipt/1.1"] = Field(alias="schemaVersion")
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=200)
    source_schema_version: Literal["socialgraph-fm.anonymized-posts/1.0"] = Field(alias="sourceSchemaVersion")
    source_sha256: str = Field(alias="sourceSha256", pattern=HASH_PATTERN)
    authorization_reference: str = Field(alias="authorizationReference", min_length=1, max_length=300)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=HASH_PATTERN)
    labels_sha256: str = Field(alias="labelsSha256", pattern=HASH_PATTERN)
    encoder: dict[str, Any]
    selection_recipe: dict[str, Any] = Field(alias="selectionRecipe")
    label_selection_recipe: TargetLabelSelectionRecipe = Field(alias="labelSelectionRecipe")
    coverage: dict[str, Any]
    receipt_hash: str = Field(alias="receiptHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self) -> TargetPackageReceipt:
        if set(self.encoder) != {"modelId", "revision", "cacheSha256", "compatibility", "dimension"} or self.encoder.get("compatibility") != "dimension-only-unverified" or self.encoder.get("dimension") != 768:
            raise ValueError("target receipt encoder provenance is invalid")
        if not isinstance(self.encoder.get("modelId"), str) or not isinstance(self.encoder.get("revision"), str) or not isinstance(self.encoder.get("cacheSha256"), str):
            raise ValueError("target receipt encoder provenance is invalid")
        if self.selection_recipe != {
            "version": "connected-structural-hash-v2",
            "nodeCount": 128,
            "requiredIo": 16,
            "requiredControls": 64,
            "minimumNonemptyModalities": 4,
            "scoreInputs": [],
            "groupRelations": {"maxGroupAccounts": 256, "totalPotentialPairBudget": 50_000},
            "fastRT": {"windowSeconds": 10, "pairBudget": 50_000, "algorithm": "sorted-sliding-window-v1"},
            "tweetSim": {"mutualTopK": 5, "cosineThreshold": 0.8, "pairBudget": 10_000},
        }:
            raise ValueError("target receipt selection recipe is invalid")
        modalities = self.coverage.get("nonemptyModalities")
        if self.coverage.get("nodeCount") != 128 or not isinstance(self.coverage.get("ioCount"), int) or not isinstance(self.coverage.get("controlCount"), int) or self.coverage["ioCount"] + self.coverage["controlCount"] != 128 or self.coverage["ioCount"] < 16 or self.coverage["controlCount"] < 64 or self.coverage.get("connected") is not True or not isinstance(modalities, list) or len(modalities) < 4 or len(set(modalities)) != len(modalities) or any(item not in GOVERNANCE_MODALITIES for item in modalities):
            raise ValueError("target receipt coverage is invalid")
        if self.receipt_hash != canonical_sha256(self.model_dump(mode="json", by_alias=True, exclude={"receipt_hash"})):
            raise ValueError("receiptHash mismatch")
        return self


def _target_label_file_sha256(
    receipt: TargetPackageReceipt, sources: tuple[ImportedSidecarLabelSource, ...]
) -> str:
    rows = sorted(
        (
            {
                "nodeId": source.node_id,
                "label": source.cohort,
                "structuralStratum": source.structural_stratum,
                "fusedDegree": source.fused_degree,
            }
            for source in sources
        ),
        key=lambda row: str(row["nodeId"]),
    )
    document = {
        "schemaVersion": "socialgraph-fm.governance-target-label-recipe/1.1",
        "datasetId": receipt.dataset_id,
        "bundleSha256": receipt.bundle_sha256,
        "selectionRecipe": receipt.label_selection_recipe.model_dump(mode="json", by_alias=True),
        "labels": rows,
    }
    payload = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


class AdaptationLabelEvidence(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    label: Literal["positive", "negative"]
    source_type: Literal["concluded_review", "imported_sidecar"] = Field(
        alias="sourceType"
    )
    source_record_id: str = Field(alias="sourceRecordId", min_length=1, max_length=200)
    source_record_hash: str = Field(alias="sourceRecordHash", pattern=HASH_PATTERN)
    review_event_hash: str | None = Field(
        default=None, alias="reviewEventHash", pattern=HASH_PATTERN
    )
    binding: AdaptationBinding
    structural_stratum: int | None = Field(default=None, alias="structuralStratum", ge=0, le=3)
    fused_degree: int | None = Field(default=None, alias="fusedDegree", ge=0, le=9_999)
    labels_sha256: str | None = Field(default=None, alias="labelsSha256", pattern=HASH_PATTERN)
    receipt_hash: str | None = Field(default=None, alias="receiptHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_source(self) -> AdaptationLabelEvidence:
        if (self.source_type == "concluded_review") != (
            self.review_event_hash is not None
        ):
            raise ValueError("concluded reviews require a reviewEventHash")
        sidecar_values = (
                self.structural_stratum,
                self.fused_degree,
                self.labels_sha256,
                self.receipt_hash,
        )
        if self.source_type == "imported_sidecar" and any(
            value is not None for value in sidecar_values
        ) != all(value is not None for value in sidecar_values):
            raise ValueError("imported sidecar provenance cannot be partial")
        return self


class AdaptationSourceRecord(FrozenModel):
    source_type: Literal["concluded_review", "imported_sidecar"] = Field(
        alias="sourceType"
    )
    source_record_id: str = Field(alias="sourceRecordId", min_length=1, max_length=200)
    source_record_hash: str = Field(alias="sourceRecordHash", pattern=HASH_PATTERN)
    review_event_hash: str | None = Field(
        default=None, alias="reviewEventHash", pattern=HASH_PATTERN
    )


class TargetLabelSet(FrozenModel):
    schema_version: Literal[
        "socialgraph-fm.governance-target-label-set/1.0",
        "socialgraph-fm.governance-target-label-set/1.1",
    ] = Field(
        alias="schemaVersion"
    )
    binding: AdaptationBinding
    sidecar_receipt: TargetPackageReceipt | None = Field(default=None, alias="sidecarReceipt")
    source_records: tuple[AdaptationSourceRecord, ...] = Field(
        alias="sourceRecords", max_length=256
    )
    review_event_hashes: tuple[str, ...] = Field(
        alias="reviewEventHashes", max_length=256
    )
    labels: tuple[AdaptationLabelEvidence, ...] = Field(max_length=256)
    conflicts: tuple[str, ...]
    positive_count: int = Field(alias="positiveCount", ge=4)
    negative_count: int = Field(alias="negativeCount", ge=4)
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_inventory(self) -> TargetLabelSet:
        if self.conflicts:
            raise ValueError("label set contains conflicts")
        if len(self.labels) < 8 or len({item.node_id for item in self.labels}) != len(
            self.labels
        ):
            raise ValueError("label set requires eight unique eligible nodes")
        if any(item.binding != self.binding for item in self.labels):
            raise ValueError("label binding mismatch")
        positive = sum(item.label == "positive" for item in self.labels)
        negative = sum(item.label == "negative" for item in self.labels)
        if (positive, negative) != (self.positive_count, self.negative_count):
            raise ValueError("label class counts are inconsistent")
        expected_sources = tuple(
            AdaptationSourceRecord(
                sourceType=item.source_type,
                sourceRecordId=item.source_record_id,
                sourceRecordHash=item.source_record_hash,
                reviewEventHash=item.review_event_hash,
            )
            for item in self.labels
        )
        if self.source_records != expected_sources:
            raise ValueError("source record inventory is inconsistent")
        if self.review_event_hashes != tuple(
            item.review_event_hash
            for item in self.labels
            if item.review_event_hash is not None
        ):
            raise ValueError("review-event hash inventory is inconsistent")
        imported = tuple(
            item for item in self.labels if item.source_type == "imported_sidecar"
        )
        if imported:
            if self.schema_version == "socialgraph-fm.governance-target-label-set/1.0":
                if self.sidecar_receipt is not None or any(item.receipt_hash is not None for item in imported):
                    raise ValueError("legacy sidecars cannot claim 1.1 provenance")
                imported = ()
            elif self.sidecar_receipt is None or len(imported) != 16 or len(imported) != len(self.labels):
                raise ValueError("imported sidecar label set requires the 1.1 receipt contract")
            for cohort in (("positive", "negative") if imported else ()):
                for stratum in range(4):
                    if sum(item.label == cohort and item.structural_stratum == stratum for item in imported) != 2:
                        raise ValueError("structural stratum label quota mismatch")
            if imported:
                receipt = self.sidecar_receipt
                if receipt is None:
                    raise ValueError("sidecar label receipt is missing")
                if any(item.labels_sha256 != receipt.labels_sha256 or item.receipt_hash != receipt.receipt_hash for item in imported):
                    raise ValueError("sidecar label receipt binding mismatch")
        elif self.schema_version != "socialgraph-fm.governance-target-label-set/1.0" or self.sidecar_receipt is not None:
            raise ValueError("legacy label set cannot claim sidecar provenance")
        logical = self.model_dump(mode="json", by_alias=True, exclude={"label_set_hash"})
        if logical.get("sidecarReceipt") is None:
            logical.pop("sidecarReceipt", None)
        for label in logical["labels"]:
            for field in ("structuralStratum", "fusedDegree", "labelsSha256", "receiptHash"):
                if label.get(field) is None:
                    label.pop(field, None)
        expected_hash = canonical_sha256(logical)
        if self.label_set_hash != expected_hash:
            raise ValueError("labelSetHash mismatch")
        return self


class ImportedSidecarLabelSource(FrozenModel):
    source_type: Literal["imported_sidecar"] = Field(alias="sourceType")
    source_record_id: str = Field(alias="sourceRecordId", min_length=1, max_length=200)
    source_record_hash: str = Field(alias="sourceRecordHash", pattern=HASH_PATTERN)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    cohort: Literal["io", "control"]
    structural_stratum: int = Field(alias="structuralStratum", ge=0, le=3)
    fused_degree: int = Field(alias="fusedDegree", ge=0, le=9_999)
    labels_sha256: str = Field(alias="labelsSha256", pattern=HASH_PATTERN)
    receipt_hash: str = Field(alias="receiptHash", pattern=HASH_PATTERN)


class ConcludedReviewLabelSource(FrozenModel):
    source_type: Literal["concluded_review"] = Field(alias="sourceType")
    case_id: str = Field(alias="caseId", pattern=CASE_PATTERN)
    event_hash: str = Field(alias="eventHash", pattern=HASH_PATTERN)


AdaptationLabelSource = Annotated[
    ImportedSidecarLabelSource | ConcludedReviewLabelSource,
    Field(discriminator="source_type"),
]


class TargetLabelSetCreateRequest(FrozenModel):
    schema_version: Literal[
        "socialgraph-fm.governance-target-label-set/1.0",
        "socialgraph-fm.governance-target-label-set/1.1",
    ] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    sources: tuple[AdaptationLabelSource, ...] = Field(min_length=1, max_length=256)
    sidecar_receipt: TargetPackageReceipt | None = Field(default=None, alias="sidecarReceipt")

    @model_validator(mode="after")
    def reject_direct_source_disagreement(self) -> TargetLabelSetCreateRequest:
        imported: dict[str, str] = {}
        for source in self.sources:
            if not isinstance(source, ImportedSidecarLabelSource):
                continue
            label = "positive" if source.cohort == "io" else "negative"
            previous = imported.get(source.node_id)
            if previous is not None:
                if previous != label:
                    raise ValueError("same-node source disagreement is blocking")
                raise ValueError("duplicate imported node label")
            imported[source.node_id] = label
        imported_sources = tuple(
            source
            for source in self.sources
            if isinstance(source, ImportedSidecarLabelSource)
        )
        if imported_sources:
            if self.schema_version != "socialgraph-fm.governance-target-label-set/1.1" or self.sidecar_receipt is None or len(imported_sources) != 16 or len(imported_sources) != len(self.sources):
                raise ValueError("imported sidecars require the 1.1 receipt-bound contract")
            if _target_label_file_sha256(self.sidecar_receipt, imported_sources) != self.sidecar_receipt.labels_sha256:
                raise ValueError("labelsSha256 mismatch")
            for cohort in ("io", "control"):
                for stratum in range(4):
                    if sum(source.cohort == cohort and source.structural_stratum == stratum for source in imported_sources) != 2:
                        raise ValueError("structural stratum label quota mismatch")
            for source in imported_sources:
                expected_hash = canonical_sha256(
                    {
                        "schemaVersion": "socialgraph-fm.governance-target-label-recipe/1.1",
                        "datasetId": self.sidecar_receipt.dataset_id,
                        "bundleSha256": self.sidecar_receipt.bundle_sha256,
                        "labelsSha256": self.sidecar_receipt.labels_sha256,
                        "receiptHash": self.sidecar_receipt.receipt_hash,
                        "nodeId": source.node_id,
                        "label": source.cohort,
                        "structuralStratum": source.structural_stratum,
                        "fusedDegree": source.fused_degree,
                    }
                )
                if source.source_record_hash != expected_hash or source.labels_sha256 != self.sidecar_receipt.labels_sha256 or source.receipt_hash != self.sidecar_receipt.receipt_hash:
                    raise ValueError("imported source provenance hash mismatch")
        elif self.schema_version != "socialgraph-fm.governance-target-label-set/1.0" or self.sidecar_receipt is not None:
            raise ValueError("legacy concluded reviews cannot claim sidecar provenance")
        return self


class TargetReviewPolicy(FrozenModel):
    schema_version: Literal[
        "socialgraph-fm.governance-target-review-policy/1.0"
    ] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)
    status: Literal["collecting_reviews", "ready", "insufficient_signal", "invalid"]
    selected_lambda: float = Field(alias="selectedLambda")
    lambda_candidates: tuple[float, ...] = Field(alias="lambdaCandidates")
    validation_losses: dict[str, float] = Field(alias="validationLosses")
    eligible_label_count: int = Field(alias="eligibleLabelCount", ge=8)
    positive_count: int = Field(alias="positiveCount", ge=4)
    negative_count: int = Field(alias="negativeCount", ge=4)
    embedding_dimension: Literal[256] = Field(alias="embeddingDimension")
    positive_centroid_hash: str = Field(alias="positiveCentroidHash", pattern=HASH_PATTERN)
    negative_centroid_hash: str = Field(alias="negativeCentroidHash", pattern=HASH_PATTERN)
    normalization_epsilon: float = Field(alias="normalizationEpsilon", gt=0)
    fitting_recipe: Literal[
        "l2-centroids+run-zscore+loo-balanced-log-loss-v1"
    ] = Field(alias="fittingRecipe")
    ready_policy_hash: str | None = Field(
        default=None, alias="readyPolicyHash", pattern=HASH_PATTERN
    )
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_policy(self) -> TargetReviewPolicy:
        candidates = (0.0, 0.25, 0.5, 1.0)
        if self.lambda_candidates != candidates or self.selected_lambda not in candidates:
            raise ValueError("policy lambda inventory is invalid")
        if self.normalization_epsilon != 1e-8:
            raise ValueError("policy normalization epsilon is invalid")
        if tuple(self.validation_losses) != ("0", "0.25", "0.5", "1"):
            raise ValueError("policy validation-loss inventory is invalid")
        if (self.status == "ready") != (self.selected_lambda != 0.0):
            raise ValueError("policy readiness is inconsistent")
        if (self.status == "ready") != (self.ready_policy_hash is not None):
            raise ValueError("ready policy publication state is inconsistent")
        _validate_hash(self, "policy_hash")
        return self


class AdaptationComparisonRow(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    base_score: float = Field(alias="baseScore", ge=0, le=1)
    base_rank: int = Field(alias="baseRank", ge=1)
    adapted_review_priority: float = Field(alias="adaptedReviewPriority", ge=0, le=1)
    adapted_rank: int = Field(alias="adaptedRank", ge=1)
    rank_delta: int = Field(alias="rankDelta")

    @model_validator(mode="after")
    def validate_delta(self) -> AdaptationComparisonRow:
        if self.rank_delta != self.adapted_rank - self.base_rank:
            raise ValueError("rankDelta is inconsistent")
        return self


class AdaptationComparisonPage(FrozenModel):
    schema_version: Literal[
        "socialgraph-fm.governance-adaptation-comparison/1.0"
    ] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    total: int = Field(ge=1, le=10_000)
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    rows: tuple[AdaptationComparisonRow, ...] = Field(max_length=500)
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)
    page_hash: str = Field(alias="pageHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_page(self) -> AdaptationComparisonPage:
        if self.offset + len(self.rows) > self.total or len(self.rows) > self.limit:
            raise ValueError("comparison page bounds are invalid")
        _validate_hash(self, "page_hash")
        return self


TARGET_TASK_PATTERN = r"^target-task-[0-9a-f]{32}$"


class TargetTaskFileDescriptor(FrozenModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9.-]{0,63}$")
    sha256: str = Field(pattern=HASH_PATTERN)
    bytes: int = Field(ge=1, le=96 * 1024 * 1024)


class TargetTaskDocument(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-task-bundle/1.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)
    mode: Literal["zero_shot", "few_shot"]
    node_count: int = Field(alias="nodeCount", ge=1, le=10_000)
    fused_edge_count: int = Field(alias="fusedEdgeCount", ge=1, le=500_000)
    modalities: tuple[GovernanceModalityV2, ...]
    inference: TargetTaskFileDescriptor
    target_receipt: TargetTaskFileDescriptor = Field(alias="targetReceipt")
    labels: TargetTaskFileDescriptor | None = None
    label_receipt: TargetTaskFileDescriptor | None = Field(default=None, alias="labelReceipt")

    @model_validator(mode="after")
    def validate_inventory(self) -> TargetTaskDocument:
        if not self.modalities or len(set(self.modalities)) != len(self.modalities):
            raise ValueError("task modalities must be nonempty and unique")
        if any(ord(character) < 32 for character in self.display_name):
            raise ValueError("displayName contains a control character")
        if tuple(sorted(self.modalities, key=GOVERNANCE_MODALITIES.index)) != self.modalities:
            raise ValueError("task modalities must use canonical order")
        if self.inference.name != "inference.zip" or self.target_receipt.name != "target-receipt.json":
            raise ValueError("task file descriptors have invalid names")
        detached = self.labels is not None and self.label_receipt is not None
        if (self.mode == "few_shot") != detached:
            raise ValueError("few-shot tasks require detached labels and receipt")
        if self.labels is not None and self.labels.name != "labels.json":
            raise ValueError("labels descriptor has an invalid name")
        if self.label_receipt is not None and self.label_receipt.name != "label-receipt.json":
            raise ValueError("label receipt descriptor has an invalid name")
        return self


class TargetDomainReceipt(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-domain-receipt/2.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    country_id: str = Field(alias="countryId", min_length=1, max_length=200)
    source_content_hash: str = Field(alias="sourceContentHash", pattern=HASH_PATTERN)
    source_manifest_sha256: str = Field(alias="sourceManifestSha256", pattern=HASH_PATTERN)
    graph_population: str = Field(alias="graphPopulation", min_length=1, max_length=200)
    graph_population_mask_sha256: str | None = Field(default=None, alias="graphPopulationMaskSha256", pattern=HASH_PATTERN)
    label_eligibility: str = Field(alias="labelEligibility", min_length=1, max_length=200)
    label_eligibility_mask_sha256: str | None = Field(default=None, alias="labelEligibilityMaskSha256", pattern=HASH_PATTERN)
    inference_sha256: str = Field(alias="inferenceSha256", pattern=HASH_PATTERN)
    node_set_sha256: str = Field(alias="nodeSetSha256", pattern=HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1, le=10_000)
    fused_edge_count: int = Field(alias="fusedEdgeCount", ge=1, le=500_000)
    modalities: tuple[GovernanceModalityV2, ...]
    connected: bool
    selection_recipe: dict[str, Any] = Field(alias="selectionRecipe")
    receipt_hash: str = Field(alias="receiptHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self) -> TargetDomainReceipt:
        if (
            (self.graph_population != "full")
            != (self.graph_population_mask_sha256 is not None)
            or (self.label_eligibility != "none")
            != (self.label_eligibility_mask_sha256 is not None)
            or not self.modalities
            or len(set(self.modalities)) != len(self.modalities)
            or tuple(sorted(self.modalities, key=GOVERNANCE_MODALITIES.index))
            != self.modalities
            or self.selection_recipe.get("scoreInputs") != []
        ):
            raise ValueError("target selection must not use model outputs")
        _validate_hash(self, "receipt_hash")
        return self


class TargetLabelV2(FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    label: Literal["positive", "negative"]
    structural_stratum: int = Field(alias="structuralStratum", ge=0, le=3)
    fused_degree: int = Field(alias="fusedDegree", ge=1)


class TargetLabelSetV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-label-set/2.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    inference_sha256: str = Field(alias="inferenceSha256", pattern=HASH_PATTERN)
    labels: tuple[TargetLabelV2, ...] = Field(min_length=8, max_length=256)
    positive_count: int = Field(alias="positiveCount", ge=0, le=256)
    negative_count: int = Field(alias="negativeCount", ge=0, le=256)
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_label_set(self) -> TargetLabelSetV2:
        if len({row.node_id for row in self.labels}) != len(self.labels):
            raise ValueError("label nodes must be unique")
        positive = sum(row.label == "positive" for row in self.labels)
        negative = len(self.labels) - positive
        if min(positive, negative) < 4 or (positive, negative) != (self.positive_count, self.negative_count):
            raise ValueError("label class inventory is invalid")
        _validate_hash(self, "label_set_hash")
        return self


class TargetLabelReceiptV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-label-receipt/2.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId")
    target_receipt_hash: str = Field(alias="targetReceiptHash", pattern=HASH_PATTERN)
    labels_sha256: str = Field(alias="labelsSha256", pattern=HASH_PATTERN)
    source_labels_sha256: str = Field(alias="sourceLabelsSha256", pattern=HASH_PATTERN)
    eligibility_mask_sha256: str = Field(alias="eligibilityMaskSha256", pattern=HASH_PATTERN)
    eligible_node_ids: tuple[str, ...] = Field(alias="eligibleNodeIds", min_length=8, max_length=10_000)
    selection_recipe: dict[str, Any] = Field(alias="selectionRecipe")
    receipt_hash: str = Field(alias="receiptHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_label_receipt(self) -> TargetLabelReceiptV2:
        if len(set(self.eligible_node_ids)) != len(self.eligible_node_ids) or self.selection_recipe.get("scoreInputs") != []:
            raise ValueError("label receipt inventory is invalid")
        _validate_hash(self, "receipt_hash")
        return self


class TargetTaskRegistration(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-task-registration/1.0"] = Field(alias="schemaVersion")
    registration_id: str = Field(alias="registrationId", pattern=TARGET_TASK_PATTERN)
    outer_bundle_sha256: str = Field(alias="outerBundleSha256", pattern=HASH_PATTERN)
    task: TargetTaskDocument
    target_receipt: TargetDomainReceipt = Field(alias="targetReceipt")
    labels: TargetLabelSetV2 | None = None
    label_receipt: TargetLabelReceiptV2 | None = Field(default=None, alias="labelReceipt")
    artifact: GovernanceArtifactReceipt
    created_at: datetime = Field(alias="createdAt")
    registration_hash: str = Field(alias="registrationHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_registration(self) -> TargetTaskRegistration:
        if self.task.task_id != self.target_receipt.task_id or self.task.inference.sha256 != self.artifact.bundle_sha256:
            raise ValueError("target registration identity mismatch")
        if (self.labels is None) != (self.label_receipt is None):
            raise ValueError("detached label registration is incomplete")
        _validate_hash(self, "registration_hash")
        return self


class ConcludedReviewReference(FrozenModel):
    case_id: str = Field(alias="caseId", pattern=CASE_PATTERN)
    event_hash: str = Field(alias="eventHash", pattern=HASH_PATTERN)


class ImportedSidecarLabelSetCreateRequestV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-label-set/2.0"] = Field(alias="schemaVersion")
    source_type: Literal["imported_sidecar"] = Field(alias="sourceType")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)


class ConcludedReviewLabelSetCreateRequestV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-label-set/2.0"] = Field(alias="schemaVersion")
    source_type: Literal["concluded_review"] = Field(alias="sourceType")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    reviews: tuple[ConcludedReviewReference, ...] = Field(min_length=8, max_length=256)


AdaptationLabelSetCreateRequestV2 = Annotated[
    ImportedSidecarLabelSetCreateRequestV2 | ConcludedReviewLabelSetCreateRequestV2,
    Field(discriminator="source_type"),
]


class TargetReviewPolicyFitRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-review-policy-fit-request/1.0"] = Field(
        alias="schemaVersion"
    )
    target_task_registration_id: str = Field(
        alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN
    )
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)


def target_label_binding_hash(
    label_set_hash: str,
    target_task_registration_id: str,
    run_id: str,
    result_hash: str,
) -> str:
    """Return the immutable identity of one label-content/run binding."""

    return canonical_sha256(
        {
            "labelSetHash": label_set_hash,
            "targetTaskRegistrationId": target_task_registration_id,
            "runId": run_id,
            "resultHash": result_hash,
        }
    )


class ReviewCollectionItemRequest(FrozenModel):
    target_type: TargetType = Field(alias="targetType")
    target_id: str = Field(alias="targetId", min_length=1, max_length=300)
    note: str = Field(default="", max_length=2_000)


class ReviewCollectionCreateRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-review-collection/1.0"] = Field(alias="schemaVersion")
    idempotency_key: str = Field(alias="idempotencyKey", pattern=r"^[A-Za-z0-9._:-]{1,200}$")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2_000)
    items: tuple[ReviewCollectionItemRequest, ...] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_items(self) -> ReviewCollectionCreateRequest:
        identities = {(item.target_type, item.target_id) for item in self.items}
        if len(identities) != len(self.items):
            raise ValueError("review collection items must be unique")
        return self


class ReviewCollection(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-review-collection/1.0"] = Field(alias="schemaVersion")
    idempotency_key: str = Field(alias="idempotencyKey")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    case: GovernanceCase
    collection_hash: str = Field(alias="collectionHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_collection(self) -> ReviewCollection:
        _validate_hash(self, "collection_hash")
        return self


class TargetReviewPolicyV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-review-policy/2.0"] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)
    status: Literal["collecting_reviews", "ready", "insufficient_signal", "invalid"]
    selected_lambda: float = Field(alias="selectedLambda")
    eligible_label_count: int = Field(alias="eligibleLabelCount", ge=8, le=256)
    positive_count: int = Field(alias="positiveCount", ge=4)
    negative_count: int = Field(alias="negativeCount", ge=4)
    fitting_recipe: Literal["l2-centroids+run-zscore+loo-balanced-log-loss-v1"] = Field(alias="fittingRecipe")
    base_outputs_immutable: Literal[True] = Field(alias="baseOutputsImmutable")
    adapted_output_fields: tuple[Literal["adaptedReviewPriority"], Literal["adaptedRank"]] = Field(alias="adaptedOutputFields")
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_v2_policy(self) -> TargetReviewPolicyV2:
        if self.positive_count + self.negative_count != self.eligible_label_count:
            raise ValueError("policy label counts are inconsistent")
        candidates = (0.0, 0.25, 0.5, 1.0)
        if self.selected_lambda not in candidates:
            raise ValueError("selected lambda is invalid")
        expected_status = "insufficient_signal" if self.selected_lambda == 0.0 else "ready"
        if self.status != expected_status:
            raise ValueError("policy readiness disagrees with the selected lambda")
        _validate_hash(self, "policy_hash")
        return self


class AdaptationComparisonV2(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-adaptation-comparison/2.0"] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    total: int = Field(ge=1, le=10_000)
    base_outputs_immutable: Literal[True] = Field(alias="baseOutputsImmutable")
    rows: tuple[AdaptationComparisonRow, ...]
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_v2_comparison(self) -> AdaptationComparisonV2:
        if self.total != len(self.rows) or len({row.node_id for row in self.rows}) != self.total:
            raise ValueError("comparison inventory is invalid")
        _validate_hash(self, "comparison_hash")
        return self


class AdaptationHandoffCreateRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-adaptation-handoff/1.0"] = Field(alias="schemaVersion")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    decision: Literal["pending_governance_review", "approved", "rejected", "superseded"] = "pending_governance_review"


class AdaptationGovernanceHandoff(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-adaptation-handoff/1.0"] = Field(alias="schemaVersion")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    target_receipt_hash: str = Field(alias="targetReceiptHash", pattern=HASH_PATTERN)
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)
    decision: Literal["pending_governance_review", "approved", "rejected", "superseded"]
    base_model_mutation: Literal[False] = Field(alias="baseModelMutation")
    handoff_hash: str = Field(alias="handoffHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_handoff(self) -> AdaptationGovernanceHandoff:
        _validate_hash(self, "handoff_hash")
        return self


class AdaptationOverlayActivationRequest(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-adaptation-overlay/1.0"] = Field(alias="schemaVersion")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)


class AdaptationOverlayActivation(FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-adaptation-overlay/1.0"] = Field(alias="schemaVersion")
    target_task_registration_id: str = Field(alias="targetTaskRegistrationId", pattern=TARGET_TASK_PATTERN)
    target_receipt_hash: str = Field(alias="targetReceiptHash", pattern=HASH_PATTERN)
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)
    active: Literal[True]
    base_model_mutation: Literal[False] = Field(alias="baseModelMutation")
    activation_hash: str = Field(alias="activationHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_activation(self) -> AdaptationOverlayActivation:
        _validate_hash(self, "activation_hash")
        return self


def _validate_hash(model: BaseModel, field_name: str) -> None:
    actual = getattr(model, field_name)
    expected = canonical_sha256(
        model.model_dump(mode="json", by_alias=True, exclude={field_name})
    )
    if actual != expected:
            raise ValueError(f"{type(model).model_fields[field_name].alias} mismatch")


__all__ = [name for name in globals() if name.startswith("Governance") or name in {
    "ARTIFACT_PATTERN", "AdaptationBinding", "AdaptationComparisonPage",
    "AdaptationComparisonRow", "AdaptationLabelEvidence", "AdaptationLabelSource",
    "AdaptationSourceRecord", "CASE_PATTERN", "CaseCreateRequest", "CaseItemRequest",
    "CaseList", "CaseState", "CaseTransitionRequest", "DerivationPage",
    "FindingsPage", "FrozenModel", "GovernanceCase", "GOVERNANCE_INPUT_SCHEMA_VERSION",
    "ImportedSidecarLabelSource", "ConcludedReviewLabelSource",
    "GOVERNANCE_MODALITIES", "GOVERNANCE_SCHEMA_VERSION", "NodeEvidenceV2",
    "OnlineRunRequest", "OnlineRunResult", "OnlineRunStatus", "ReviewEventRequest",
    "RUN_PATTERN", "RunList", "TargetLabelSet", "TargetLabelSetCreateRequest",
    "TargetReviewPolicy",
    "RunComparison",
}]
