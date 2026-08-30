"""Public contracts for the isolated SocialGraph-FM Research channel.

These models intentionally do not reuse core's formal serving readiness.
SocialGraph-FM Research is a separately labelled, single-seed preview and every inference
response remains hash-bound to the graph and model that produced it.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gfm_hashing import canonical_sha256

RESEARCH_SCHEMA_VERSION = "socialgraph-fm.research/1.0"
RESEARCH_RELEASE_LABEL = "SocialGraph-FM Research"
RESEARCH_SEED = 1729
HASH = r"^[0-9a-f]{64}$"

ResearchSchemaVersion = Literal["socialgraph-fm.research/1.0"]
ResearchReleaseLabel = Literal["SocialGraph-FM Research"]
ResearchSeed = Literal[1729]

ResearchTaskId = Literal[
    "research.content_policy_review",
    "research.account_risk_review",
    "research.signed_relation_review",
    "core.collaboration_completion",
]
RESEARCH_TASK_IDS: tuple[ResearchTaskId, ...] = (
    "research.content_policy_review",
    "research.account_risk_review",
    "research.signed_relation_review",
    "core.collaboration_completion",
)


class ResearchModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        frozen=True,
        protected_namespaces=("model_dump",),
    )


class ResearchModelCapability(ResearchModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    task_ids: tuple[ResearchTaskId, ...] = Field(alias="taskIds", strict=False)
    graph_schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="graphSchemaVersion"
    )
    max_nodes: int = Field(alias="maxNodes", ge=20, le=50_000)
    max_edges: int = Field(alias="maxEdges", ge=1, le=1_500_000)
    claim_status: Literal["observed_transfer_gain", "not_demonstrated"] = Field(
        alias="claimStatus"
    )

    @model_validator(mode="after")
    def exact_task_inventory(self) -> ResearchModelCapability:
        if self.task_ids != RESEARCH_TASK_IDS:
            raise ValueError("SocialGraph-FM Research model must expose the four ordered tasks")
        return self


class ResearchUploadCapability(ResearchModel):
    compatible_task_ids: tuple[
        Literal["core.collaboration_completion"], ...
    ] = Field(alias="compatibleTaskIds", strict=False)
    auxiliary_capabilities: tuple[Literal["similar-nodes"], ...] = Field(
        alias="auxiliaryCapabilities", strict=False
    )
    min_nodes: Literal[5] = Field(alias="minNodes")
    max_nodes: Literal[50_000] = Field(alias="maxNodes")
    max_edges: Literal[1_500_000] = Field(alias="maxEdges")


class ResearchCapabilities(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    channel: Literal["research"]
    release_label: ResearchReleaseLabel = Field(alias="releaseLabel")
    seed: ResearchSeed
    preliminary: Literal[True]
    research_serving_ready: bool = Field(alias="researchServingReady")
    unavailable_reason: str | None = Field(
        alias="unavailableReason", default=None, max_length=100
    )
    model: ResearchModelCapability | None = None
    task_ids: tuple[ResearchTaskId, ...] = Field(alias="taskIds", strict=False)
    upload: ResearchUploadCapability
    capability_hash: str = Field(alias="capabilityHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_capability(self) -> ResearchCapabilities:
        if self.task_ids != RESEARCH_TASK_IDS:
            raise ValueError("SocialGraph-FM Research capability task inventory is not canonical")
        if self.research_serving_ready != (self.model is not None):
            raise ValueError("researchServingReady must derive from the validated model")
        if self.research_serving_ready == (self.unavailable_reason is not None):
            raise ValueError("unavailableReason must be present exactly when serving is unavailable")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"capability_hash"})
        )
        if self.capability_hash != expected:
            raise ValueError("capabilityHash mismatch")
        return self


class NodeReviewScope(ResearchModel):
    kind: Literal["nodes"]
    node_ids: tuple[str, ...] = Field(
        alias="nodeIds", min_length=1, max_length=10_000, strict=False
    )

    @model_validator(mode="after")
    def unique_nodes(self) -> NodeReviewScope:
        if any(not item or len(item) > 300 for item in self.node_ids):
            raise ValueError("nodeIds contains an invalid identifier")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("nodeIds must be unique")
        return self


class DirectedPairsScope(ResearchModel):
    kind: Literal["directed-node-pairs"]
    pairs: tuple[tuple[str, str], ...] = Field(
        min_length=1, max_length=10_000, strict=False
    )

    @model_validator(mode="after")
    def valid_pairs(self) -> DirectedPairsScope:
        if any(
            not source
            or not target
            or source == target
            or len(source) > 300
            or len(target) > 300
            for source, target in self.pairs
        ):
            raise ValueError("directed pairs require bounded, distinct endpoints")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("directed pairs must be unique")
        return self


class CollaborationCandidatesScope(ResearchModel):
    kind: Literal["collaboration-candidates"]
    anchor_node_id: str = Field(alias="anchorNodeId", min_length=1, max_length=300)
    top_k: int = Field(alias="topK", ge=1, le=100)


ResearchTargetScope = Annotated[
    NodeReviewScope | DirectedPairsScope | CollaborationCandidatesScope,
    Field(discriminator="kind"),
]


class ResearchMetric(ResearchModel):
    name: str = Field(min_length=1, max_length=100)
    value: float

    @model_validator(mode="after")
    def finite_value(self) -> ResearchMetric:
        if not math.isfinite(self.value):
            raise ValueError("metric value must be finite")
        return self


class ResearchScenario(ResearchModel):
    scenario_id: Literal[
        "twitch-content-policy",
        "tolokers-account-risk",
        "wiki-rfa-signed-relation",
        "email-eu-collaboration",
    ] = Field(alias="scenarioId")
    dataset_id: Literal["twitch-language", "tolokers", "wiki-rfa", "email-eu-core"] = (
        Field(alias="datasetId")
    )
    title: str = Field(min_length=1, max_length=100)
    task_id: ResearchTaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str | None = Field(
        alias="graphVersionHash", default=None, pattern=HASH
    )
    model_version_id: str | None = Field(
        alias="modelVersionId", default=None, max_length=300
    )
    enabled: bool
    unavailable_reason: str | None = Field(
        alias="unavailableReason", default=None, max_length=100
    )
    default_target_scope: ResearchTargetScope = Field(alias="defaultTargetScope")
    primary_metric: ResearchMetric | None = Field(alias="primaryMetric", default=None)
    scratch_delta: float | None = Field(alias="scratchDelta", default=None)

    @model_validator(mode="after")
    def validate_scenario(self) -> ResearchScenario:
        expected = {
            "twitch-content-policy": (
                "twitch-language",
                "research.content_policy_review",
                "nodes",
            ),
            "tolokers-account-risk": (
                "tolokers",
                "research.account_risk_review",
                "nodes",
            ),
            "wiki-rfa-signed-relation": (
                "wiki-rfa",
                "research.signed_relation_review",
                "directed-node-pairs",
            ),
            "email-eu-collaboration": (
                "email-eu-core",
                "core.collaboration_completion",
                "collaboration-candidates",
            ),
        }[self.scenario_id]
        observed = (self.dataset_id, self.task_id, self.default_target_scope.kind)
        if observed != expected:
            raise ValueError("scenario identity, task, and target scope do not match")
        if self.enabled != (
            self.model_version_id is not None and self.graph_version_hash is not None
        ):
            raise ValueError("enabled scenarios require model and graph identities")
        if self.enabled == (self.unavailable_reason is not None):
            raise ValueError("unavailableReason must be present exactly when disabled")
        if self.scratch_delta is not None and not math.isfinite(self.scratch_delta):
            raise ValueError("scratchDelta must be finite")
        return self


class ResearchScenariosResponse(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    release_label: ResearchReleaseLabel = Field(alias="releaseLabel")
    seed: ResearchSeed
    preliminary: Literal[True]
    scenarios: tuple[ResearchScenario, ...] = Field(strict=False)
    scenarios_hash: str = Field(alias="scenariosHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_scenarios(self) -> ResearchScenariosResponse:
        expected = (
            "twitch-content-policy",
            "tolokers-account-risk",
            "wiki-rfa-signed-relation",
            "email-eu-collaboration",
        )
        if tuple(item.scenario_id for item in self.scenarios) != expected:
            raise ValueError("SocialGraph-FM Research must expose the four ordered scenarios")
        if self.scenarios_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"scenarios_hash"})
        ):
            raise ValueError("scenariosHash mismatch")
        return self


class ResearchPreviewNode(ResearchModel):
    id: str = Field(min_length=1, max_length=300)
    label: str = Field(min_length=1, max_length=500)


class ResearchPreviewEdge(ResearchModel):
    id: str = Field(min_length=1, max_length=700)
    source: str = Field(min_length=1, max_length=300)
    target: str = Field(min_length=1, max_length=300)
    directed: bool


class ResearchScenarioGraphPreview(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    scenario_id: Literal[
        "twitch-content-policy",
        "tolokers-account-risk",
        "wiki-rfa-signed-relation",
        "email-eu-collaboration",
    ] = Field(alias="scenarioId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    nodes: tuple[ResearchPreviewNode, ...] = Field(max_length=800, strict=False)
    edges: tuple[ResearchPreviewEdge, ...] = Field(max_length=2_500, strict=False)
    partial_preview: bool = Field(alias="partialPreview")
    node_count: int = Field(alias="nodeCount", ge=1, le=50_000)
    edge_count: int = Field(alias="edgeCount", ge=0, le=1_500_000)
    preview_hash: str = Field(alias="previewHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_preview(self) -> ResearchScenarioGraphPreview:
        node_ids = tuple(item.id for item in self.nodes)
        edge_ids = tuple(item.id for item in self.edges)
        if len(set(node_ids)) != len(node_ids) or len(set(edge_ids)) != len(edge_ids):
            raise ValueError("preview node and edge identifiers must be unique")
        available = set(node_ids)
        if any(item.source not in available or item.target not in available for item in self.edges):
            raise ValueError("preview edges must reference preview nodes")
        expected_partial = len(self.nodes) < self.node_count or len(self.edges) < self.edge_count
        if self.partial_preview != expected_partial:
            raise ValueError("partialPreview does not match preview inventory")
        if self.preview_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"preview_hash"})
        ):
            raise ValueError("previewHash mismatch")
        return self


class ResearchRunParameters(ResearchModel):
    candidate_limit: int = Field(alias="candidateLimit", ge=1, le=1_000)


class ResearchRunRequest(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    task_id: ResearchTaskId = Field(alias="taskId")
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    target_scope: ResearchTargetScope = Field(alias="targetScope")
    scenario_id: str | None = Field(alias="scenarioId", default=None, max_length=100)
    parameters: ResearchRunParameters

    @model_validator(mode="after")
    def validate_task_scope(self) -> ResearchRunRequest:
        expected = {
            "research.content_policy_review": "nodes",
            "research.account_risk_review": "nodes",
            "research.signed_relation_review": "directed-node-pairs",
            "core.collaboration_completion": "collaboration-candidates",
        }[self.task_id]
        if self.target_scope.kind != expected:
            raise ValueError("targetScope does not match taskId")
        return self

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class ResearchRunStatus(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = Field(ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    error_code: str | None = Field(alias="errorCode", default=None, max_length=100)
    state_hash: str = Field(alias="stateHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_status(self) -> ResearchRunStatus:
        expected_progress = {
            "queued": 0,
            "running": 10,
            "succeeded": 100,
            "failed": 100,
        }
        if self.progress != expected_progress[self.status]:
            raise ValueError("progress does not match status")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("only failed runs carry errorCode")
        if self.state_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"state_hash"})
        ):
            raise ValueError("stateHash mismatch")
        return self


class ResearchFinding(ResearchModel):
    id: str = Field(min_length=1, max_length=200)
    rank: int = Field(ge=1, le=10_000)
    entity_type: Literal["node", "directed-edge", "node-pair"] = Field(
        alias="entityType"
    )
    entity_ids: tuple[str, ...] = Field(alias="entityIds", strict=False)
    score: float = Field(ge=0, le=1)
    score_kind: Literal["probability", "ranking-score"] = Field(alias="scoreKind")
    calibrated: bool
    reason_codes: tuple[str, ...] = Field(
        alias="reasonCodes", max_length=20, strict=False
    )
    limitations: tuple[str, ...] = Field(min_length=1, max_length=20, strict=False)
    review_required: Literal[True] = Field(alias="reviewRequired")

    @model_validator(mode="after")
    def validate_finding(self) -> ResearchFinding:
        expected_size = 1 if self.entity_type == "node" else 2
        if len(self.entity_ids) != expected_size or any(
            not item or len(item) > 300 for item in self.entity_ids
        ):
            raise ValueError("finding entityIds do not match entityType")
        if not math.isfinite(self.score):
            raise ValueError("finding score must be finite")
        if self.calibrated != (self.score_kind == "probability"):
            raise ValueError("only calibrated findings may expose probability")
        if any(not code or len(code) > 100 for code in self.reason_codes):
            raise ValueError("reasonCodes contain an invalid value")
        if any(not item or len(item) > 500 for item in self.limitations):
            raise ValueError("limitations contain an invalid value")
        return self


class ResearchRunResult(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    task_id: ResearchTaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    seed: ResearchSeed
    preliminary: Literal[True]
    calibration_status: Literal["calibrated", "ranking_only"] = Field(
        alias="calibrationStatus"
    )
    findings: tuple[ResearchFinding, ...] = Field(strict=False)
    completed_at: datetime = Field(alias="completedAt")
    result_hash: str = Field(alias="resultHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_result(self) -> ResearchRunResult:
        entity_type = {
            "research.content_policy_review": "node",
            "research.account_risk_review": "node",
            "research.signed_relation_review": "directed-edge",
            "core.collaboration_completion": "node-pair",
        }[self.task_id]
        if any(item.entity_type != entity_type for item in self.findings):
            raise ValueError("finding entity type does not match taskId")
        if tuple(item.rank for item in self.findings) != tuple(
            range(1, len(self.findings) + 1)
        ):
            raise ValueError("finding ranks must be contiguous and ordered")
        if not self.findings:
            if self.calibration_status != "ranking_only":
                raise ValueError("empty findings must be ranking_only")
        elif self.calibration_status == "calibrated":
            if not all(item.calibrated for item in self.findings):
                raise ValueError("calibrated results require probability findings")
        elif any(item.calibrated for item in self.findings):
            raise ValueError("ranking_only results cannot expose probabilities")
        if self.result_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"result_hash"})
        ):
            raise ValueError("resultHash mismatch")
        return self


class SimilarNodesRequest(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=300)
    top_k: int = Field(alias="topK", ge=1, le=50)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)


class StructuralFacts(ResearchModel):
    degree: int = Field(ge=0)
    in_degree: int = Field(alias="inDegree", ge=0)
    out_degree: int = Field(alias="outDegree", ge=0)
    pagerank: float = Field(ge=0, le=1)
    clustering: float = Field(ge=0, le=1)
    core_number: int = Field(alias="coreNumber", ge=0)

    @model_validator(mode="after")
    def finite_facts(self) -> StructuralFacts:
        if not math.isfinite(self.pagerank) or not math.isfinite(self.clustering):
            raise ValueError("structural facts must be finite")
        return self


class SimilarNodeMatch(ResearchModel):
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=300)
    dataset_id: str | None = Field(alias="datasetId", default=None, max_length=100)
    similarity: float = Field(ge=-1, le=1)
    structural_facts: StructuralFacts = Field(alias="structuralFacts")

    @model_validator(mode="after")
    def finite_similarity(self) -> SimilarNodeMatch:
        if not math.isfinite(self.similarity):
            raise ValueError("similarity must be finite")
        return self


class SimilarNodesResponse(ResearchModel):
    schema_version: ResearchSchemaVersion = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=300)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    matches: tuple[SimilarNodeMatch, ...] = Field(max_length=50, strict=False)
    result_hash: str = Field(alias="resultHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_response(self) -> SimilarNodesResponse:
        if self.result_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"result_hash"})
        ):
            raise ValueError("resultHash mismatch")
        return self


class ResearchCompatibilityBlocker(ResearchModel):
    code: str = Field(pattern=r"^[A-Z0-9_]{1,100}$")
    message: str = Field(min_length=1, max_length=500)


class ResearchGraphCompatibility(ResearchModel):
    intended_use: Literal["gfm_research"] = Field(alias="intendedUse")
    status: Literal["compatible", "blocked"]
    compatible_task_ids: tuple[
        Literal["core.collaboration_completion"], ...
    ] = Field(alias="compatibleTaskIds", strict=False)
    auxiliary_capabilities: tuple[Literal["similar-nodes"], ...] = Field(
        alias="auxiliaryCapabilities", strict=False
    )
    blockers: tuple[ResearchCompatibilityBlocker, ...] = Field(strict=False)
    adapter_status: Literal["pending_registration", "ready"] = Field(
        alias="adapterStatus"
    )

    @model_validator(mode="after")
    def validate_compatibility(self) -> ResearchGraphCompatibility:
        has_capability = bool(
            self.compatible_task_ids or self.auxiliary_capabilities
        )
        if self.status != ("compatible" if has_capability else "blocked"):
            raise ValueError("compatibility status must derive from available capabilities")
        if self.status == "blocked" and not self.blockers:
            raise ValueError("blocked graph must explain its blockers")
        return self


def build_unavailable_capabilities(
    reason: str = "RESEARCH_MODEL_NOT_INSTALLED",
) -> ResearchCapabilities:
    payload: dict[str, Any] = {
        "schemaVersion": RESEARCH_SCHEMA_VERSION,
        "channel": "research",
        "releaseLabel": RESEARCH_RELEASE_LABEL,
        "seed": RESEARCH_SEED,
        "preliminary": True,
        "researchServingReady": False,
        "unavailableReason": reason,
        "model": None,
        "taskIds": list(RESEARCH_TASK_IDS),
        "upload": {
            "compatibleTaskIds": ["core.collaboration_completion"],
            "auxiliaryCapabilities": ["similar-nodes"],
            "minNodes": 5,
            "maxNodes": 50_000,
            "maxEdges": 1_500_000,
        },
    }
    payload["capabilityHash"] = canonical_sha256(payload)
    return ResearchCapabilities.model_validate(payload)


__all__ = [
    "RESEARCH_RELEASE_LABEL",
    "RESEARCH_SCHEMA_VERSION",
    "RESEARCH_SEED",
    "RESEARCH_TASK_IDS",
    "ResearchCapabilities",
    "ResearchFinding",
    "ResearchGraphCompatibility",
    "ResearchRunRequest",
    "ResearchRunResult",
    "ResearchRunStatus",
    "ResearchScenario",
    "ResearchScenariosResponse",
    "ResearchTaskId",
    "SimilarNodesRequest",
    "SimilarNodesResponse",
    "build_unavailable_capabilities",
]
