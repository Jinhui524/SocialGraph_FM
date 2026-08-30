"""Exact loopback wire contracts shared with the API SocialGraph-FM Research gateway."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256

from .contracts import RESEARCH_TASK_IDS, ResearchTaskId
from .workflow import load_registry

WIRE_SCHEMA = "socialgraph-fm.research/1.0"
HASH = r"^[0-9a-f]{64}$"


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, protected_namespaces=("model_dump",)
    )


class WireModelCapability(WireModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    task_ids: tuple[ResearchTaskId, ...] = Field(alias="taskIds", strict=False)
    graph_schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="graphSchemaVersion"
    )
    max_nodes: int = Field(alias="maxNodes", ge=20, le=50_000)
    max_edges: int = Field(alias="maxEdges", ge=1, le=1_500_000)
    claim_status: Literal["observed_transfer_gain", "not_demonstrated"] = Field(alias="claimStatus")

    @model_validator(mode="after")
    def exact_tasks(self):
        if self.task_ids != RESEARCH_TASK_IDS:
            raise ValueError("SocialGraph-FM Research model task inventory is not canonical")
        return self


class WireGraphReference(WireModel):
    kind: Literal["registered-scenario", "uploaded-artifact"]
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    graph_fact_hash: str | None = Field(default=None, alias="graphFactHash", pattern=HASH)
    artifact_id: str | None = Field(default=None, alias="artifactId", max_length=300)
    artifact_hash: str | None = Field(default=None, alias="artifactHash", pattern=HASH)
    node_count: int = Field(alias="nodeCount", ge=0, le=50_000)
    edge_count: int = Field(alias="edgeCount", ge=0, le=1_500_000)

    @model_validator(mode="after")
    def uploaded_identity(self):
        required = (self.graph_fact_hash, self.artifact_id, self.artifact_hash)
        if (self.kind == "uploaded-artifact") != all(item is not None for item in required):
            raise ValueError("uploaded graph reference identity is incomplete")
        return self


class WireNodeScope(WireModel):
    kind: Literal["nodes"]
    node_ids: tuple[str, ...] = Field(alias="nodeIds", min_length=1, max_length=10_000)


class WireDirectedScope(WireModel):
    kind: Literal["directed-node-pairs"]
    pairs: tuple[tuple[str, str], ...] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def distinct(self):
        if len(set(self.pairs)) != len(self.pairs) or any(
            left == right for left, right in self.pairs
        ):
            raise ValueError("directed pair targets must be unique with distinct endpoints")
        return self


class WireCollaborationScope(WireModel):
    kind: Literal["collaboration-candidates"]
    anchor_node_id: str = Field(alias="anchorNodeId", min_length=1, max_length=300)
    top_k: int = Field(alias="topK", ge=1, le=100)


WireTargetScope = Annotated[
    WireNodeScope | WireDirectedScope | WireCollaborationScope,
    Field(discriminator="kind"),
]


class WireRunParameters(WireModel):
    candidate_limit: int = Field(alias="candidateLimit", ge=1, le=1_000)


class WireRunRequest(WireModel):
    schema_version: Literal["socialgraph-fm.research/1.0"] = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    task_id: ResearchTaskId = Field(alias="taskId")
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    target_scope: WireTargetScope = Field(alias="targetScope")
    scenario_id: str | None = Field(default=None, alias="scenarioId", max_length=100)
    parameters: WireRunParameters

    @model_validator(mode="after")
    def task_scope(self):
        expected = {
            "research.content_policy_review": "nodes",
            "research.account_risk_review": "nodes",
            "research.signed_relation_review": "directed-node-pairs",
            "core.collaboration_completion": "collaboration-candidates",
        }[self.task_id]
        if self.target_scope.kind != expected:
            raise ValueError("research task and target scope disagree")
        return self

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class WireRunEnvelope(WireModel):
    schema_version: Literal["socialgraph-fm.research/1.0"] = Field(alias="schemaVersion")
    request: WireRunRequest
    graph_reference: WireGraphReference = Field(alias="graphReference")
    expected_model: WireModelCapability = Field(alias="expectedModel")


class WireSimilarRequest(WireModel):
    schema_version: Literal["socialgraph-fm.research/1.0"] = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=300)
    top_k: int = Field(alias="topK", ge=1, le=50)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)


class WireSimilarEnvelope(WireModel):
    schema_version: Literal["socialgraph-fm.research/1.0"] = Field(alias="schemaVersion")
    request: WireSimilarRequest
    graph_reference: WireGraphReference = Field(alias="graphReference")
    expected_model: WireModelCapability = Field(alias="expectedModel")


class WireGraphRegistrationEnvelope(WireModel):
    schema_version: Literal["socialgraph-fm.research/1.0"] = Field(alias="schemaVersion")
    graph_reference: WireGraphReference = Field(alias="graphReference")
    compatible_task_ids: tuple[Literal["core.collaboration_completion"], ...] = Field(
        alias="compatibleTaskIds", strict=False
    )
    auxiliary_capabilities: tuple[Literal["similar-nodes"], ...] = Field(
        alias="auxiliaryCapabilities", strict=False
    )
    expected_model: WireModelCapability = Field(alias="expectedModel")

    @model_validator(mode="after")
    def uploaded_only(self):
        if self.graph_reference.kind != "uploaded-artifact":
            raise ValueError("SocialGraph-FM Research registration accepts uploaded artifacts only")
        return self


def model_capability(registry: dict[str, Any]) -> dict[str, Any]:
    return WireModelCapability(
        modelVersionId=registry["modelVersionId"],
        modelVersionHash=registry["modelVersionHash"],
        artifactHash=registry["artifactHash"],
        taskIds=registry["taskIds"],
        graphSchemaVersion=registry["graphSchemaVersion"],
        maxNodes=registry["maxNodes"],
        maxEdges=registry["maxEdges"],
        claimStatus=(
            "observed_transfer_gain"
            if registry["claimStatus"] in {"observed", "observed_transfer_gain"}
            else "not_demonstrated"
        ),
    ).model_dump(mode="json", by_alias=True)


def capabilities_payload(research_root) -> dict[str, Any]:
    try:
        registry = load_registry(research_root)
    except FileNotFoundError:
        registry = None
    payload: dict[str, Any] = {
        "schemaVersion": WIRE_SCHEMA,
        "channel": "research",
        "releaseLabel": "SocialGraph-FM Research",
        "seed": 1729,
        "preliminary": True,
        "researchServingReady": registry is not None,
        "unavailableReason": None if registry is not None else "RESEARCH_MODEL_NOT_INSTALLED",
        "model": None if registry is None else model_capability(registry),
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
    return payload


def scenarios_payload(research_root) -> dict[str, Any]:
    try:
        registry = load_registry(research_root)
    except FileNotFoundError:
        registry = None
    defaults = (
        (
            "twitch-content-policy",
            "twitch-language",
            "Content policy review",
            "research.content_policy_review",
            "research:twitch-language",
            {"kind": "nodes", "nodeIds": ["0"]},
        ),
        (
            "tolokers-account-risk",
            "tolokers",
            "Historical account status review",
            "research.account_risk_review",
            "research:tolokers",
            {"kind": "nodes", "nodeIds": ["0"]},
        ),
        (
            "wiki-rfa-signed-relation",
            "wiki-rfa",
            "Governance relation stance review",
            "research.signed_relation_review",
            "research:wiki-rfa",
            {"kind": "directed-node-pairs", "pairs": [["0", "1"]]},
        ),
        (
            "email-eu-collaboration",
            "email-eu-core",
            "Collaboration relation candidates",
            "core.collaboration_completion",
            "research:email-eu-core",
            {"kind": "collaboration-candidates", "anchorNodeId": "0", "topK": 10},
        ),
    )
    by_id = {} if registry is None else {item["scenarioId"]: item for item in registry["scenarios"]}
    rows = []
    for scenario_id, dataset_id, title, task_id, graph_id, target in defaults:
        registered = by_id.get(scenario_id)
        rows.append(
            {
                "scenarioId": scenario_id,
                "datasetId": dataset_id,
                "title": title,
                "taskId": task_id,
                "graphVersionId": graph_id,
                "graphVersionHash": None if registered is None else registered["graphVersionHash"],
                "modelVersionId": None if registry is None else registry["modelVersionId"],
                "enabled": registered is not None,
                "unavailableReason": None
                if registered is not None
                else "RESEARCH_MODEL_NOT_INSTALLED",
                "defaultTargetScope": target
                if registered is None
                else registered["defaultTargetScope"],
                "primaryMetric": None if registered is None else registered["primaryMetric"],
                "scratchDelta": None if registered is None else registered["scratchDelta"],
            }
        )
    payload = {
        "schemaVersion": WIRE_SCHEMA,
        "releaseLabel": "SocialGraph-FM Research",
        "seed": 1729,
        "preliminary": True,
        "scenarios": rows,
    }
    payload["scenariosHash"] = canonical_sha256(payload)
    return payload


__all__ = [
    "WIRE_SCHEMA",
    "WireGraphReference",
    "WireGraphRegistrationEnvelope",
    "WireModelCapability",
    "WireRunEnvelope",
    "WireRunRequest",
    "WireSimilarEnvelope",
    "WireSimilarRequest",
    "capabilities_payload",
    "model_capability",
    "scenarios_payload",
]
