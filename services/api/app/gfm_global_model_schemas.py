"""Public API contracts for the isolated SocialGraph-FM Global governance channel."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gfm_hashing import canonical_sha256

GLOBAL_MODEL_SCHEMA_VERSION = "socialgraph-fm.gfm-global-model/1.0"
GLOBAL_MODEL_HEALTH_SCHEMA_VERSION = "socialgraph-fm.global-model-health/1.0"
GLOBAL_MODEL_CARD_SCHEMA_VERSION = "socialgraph-fm.global-model-card/1.0"
GLOBAL_MODEL_TASK_ID = "coordination_risk"
GLOBAL_MODEL_DATASET_VERSION_ID = "socialgraph-fm:russia"
GLOBAL_MODEL_SEED = 12121995
HASH = r"^[0-9a-f]{64}$"

GlobalModelProtocol = Literal["in_domain", "low_label", "cross_domain", "global"]
GLOBAL_MODEL_PROTOCOLS: tuple[GlobalModelProtocol, ...] = ("in_domain", "low_label", "cross_domain", "global")
GlobalModelModality = Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"]


def _service_identity(
    model_version_id: str | None,
    model_version_hash: str | None,
    corpus_hash: str | None,
) -> str:
    return canonical_sha256(
        {
            "service": "socialgraph-fm-gfm/global-model",
            "datasetVersionId": GLOBAL_MODEL_DATASET_VERSION_ID,
            "modelVersionId": model_version_id,
            "modelVersionHash": model_version_hash,
            "corpusHash": corpus_hash,
        }
    )


class GlobalModelContract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
    )


class GlobalModelMetric(GlobalModelContract):
    macro_f1: float = Field(alias="macroF1", ge=0.0, le=1.0)
    pr_auc: float = Field(alias="prAuc", ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    labelled_train_nodes: int = Field(alias="labelledTrainNodes", ge=0)


class GlobalModelProtocolModel(GlobalModelContract):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH)
    state: Literal["frozenDemo", "servingReady"]


class GlobalModelCapability(GlobalModelContract):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    corpus_hash: str = Field(alias="corpusHash", pattern=HASH)
    source_code_hash: str = Field(alias="sourceCodeHash", pattern=HASH)
    task_id: Literal["coordination_risk"] = Field(alias="taskId")
    protocols: tuple[GlobalModelProtocol, ...]
    protocol_models: dict[GlobalModelProtocol, GlobalModelProtocolModel] = Field(
        alias="protocolModels"
    )
    state: Literal["preliminary", "servingReady"]

    @model_validator(mode="after")
    def exact_protocols(self) -> GlobalModelCapability:
        if self.protocols != GLOBAL_MODEL_PROTOCOLS:
            raise ValueError("GlobalModel protocol inventory is not canonical")
        if set(self.protocol_models) != set(GLOBAL_MODEL_PROTOCOLS):
            raise ValueError("GlobalModel protocol model inventory is not canonical")
        if any(
            (protocol == "global") != (model.state == "servingReady")
            for protocol, model in self.protocol_models.items()
        ):
            raise ValueError("only the global model may be servingReady")
        if len({model.model_version_id for model in self.protocol_models.values()}) != len(
            GLOBAL_MODEL_PROTOCOLS
        ) or len(
            {model.model_version_hash for model in self.protocol_models.values()}
        ) != len(GLOBAL_MODEL_PROTOCOLS):
            raise ValueError("GlobalModel protocol model identities must be unique")
        global_model = self.protocol_models["global"]
        if (
            global_model.model_version_id != self.model_version_id
            or global_model.model_version_hash != self.model_version_hash
        ):
            raise ValueError("top-level GlobalModel identity must describe the Global model")
        return self


class GlobalModelCapabilities(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(
        alias="schemaVersion"
    )
    channel: Literal["socialgraph-global"]
    release_label: Literal["SocialGraph-FM Global"] = Field(alias="releaseLabel")
    seed: Literal[12121995]
    serving_ready: bool = Field(alias="servingReady")
    unavailable_reason: str | None = Field(alias="unavailableReason")
    task_id: Literal["coordination_risk"] = Field(alias="taskId")
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    model: GlobalModelCapability | None
    capability_hash: str = Field(alias="capabilityHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_identity(self) -> GlobalModelCapabilities:
        if self.serving_ready != (self.model is not None):
            raise ValueError("servingReady and model presence disagree")
        if self.capability_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"capability_hash"})
        ):
            raise ValueError("capabilityHash mismatch")
        return self


class GlobalModelHealth(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.global-model-health/1.0"] = Field(
        alias="schemaVersion"
    )
    service_identity: str = Field(alias="serviceIdentity", pattern=HASH)
    serving_ready: bool = Field(alias="servingReady")
    model_version_id: str | None = Field(alias="modelVersionId", max_length=300)
    model_version_hash: str | None = Field(alias="modelVersionHash", pattern=HASH)
    corpus_hash: str | None = Field(alias="corpusHash", pattern=HASH)
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    health_hash: str = Field(alias="healthHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_health(self) -> GlobalModelHealth:
        ready_identity = (
            self.model_version_id is not None
            and self.model_version_hash is not None
            and self.corpus_hash is not None
        )
        if self.serving_ready != ready_identity:
            raise ValueError("GlobalModel health readiness identity is incomplete")
        if self.service_identity != _service_identity(
            self.model_version_id, self.model_version_hash, self.corpus_hash
        ):
            raise ValueError("GlobalModel serviceIdentity mismatch")
        if self.health_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"health_hash"})
        ):
            raise ValueError("healthHash mismatch")
        return self


class GlobalModelArchitecture(GlobalModelContract):
    name: str = Field(min_length=1, max_length=500)
    text_features: str = Field(alias="textFeatures", min_length=1, max_length=500)
    structural_features: str = Field(
        alias="structuralFeatures", min_length=1, max_length=500
    )
    gnn_layers: Literal[2] = Field(alias="gnnLayers")
    hidden_dim: Literal[256] = Field(alias="hiddenDim")
    router: str = Field(min_length=1, max_length=500)


class GlobalModelTrainingData(GlobalModelContract):
    countries: tuple[str, ...]
    node_count: int = Field(alias="nodeCount", ge=1)
    node_count_by_country: dict[str, int] = Field(alias="nodeCountByCountry")
    content: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_countries(self) -> GlobalModelTrainingData:
        countries = ("china", "cuba", "iran", "russia", "UAE", "venezuela")
        if self.countries != countries or set(self.node_count_by_country) != set(countries):
            raise ValueError("GlobalModel model-card country inventory is not canonical")
        if any(value < 1 for value in self.node_count_by_country.values()):
            raise ValueError("GlobalModel model-card country node counts must be positive")
        if sum(self.node_count_by_country.values()) != self.node_count:
            raise ValueError("GlobalModel model-card node count does not add up")
        return self


class GlobalModelLicense(GlobalModelContract):
    name: str = Field(min_length=1, max_length=300)
    license: Literal["CC-BY-4.0", "MIT"]
    url: str = Field(min_length=1, max_length=2048)


class GlobalModelSourceAttribution(GlobalModelContract):
    kind: Literal["inspired"]
    paper_url: str = Field(alias="paperUrl", min_length=1, max_length=2048)
    complete_reproduction: Literal[False] = Field(alias="completeReproduction")


class GlobalModelCard(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.global-model-card/1.0"] = Field(
        alias="schemaVersion"
    )
    release_id: Literal["socialgraph-fm"] = Field(alias="releaseId")
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    task_id: Literal["coordination_risk"] = Field(alias="taskId")
    architecture: GlobalModelArchitecture
    protocols: dict[GlobalModelProtocol, GlobalModelProtocolModel]
    training_data: GlobalModelTrainingData = Field(alias="trainingData")
    intended_use: tuple[str, ...] = Field(alias="intendedUse", min_length=1)
    out_of_scope: tuple[str, ...] = Field(alias="outOfScope", min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)
    ethics: tuple[str, ...] = Field(min_length=1)
    licenses: tuple[GlobalModelLicense, ...] = Field(min_length=2, max_length=2)
    source_attribution: GlobalModelSourceAttribution = Field(alias="sourceAttribution")
    metrics: dict[GlobalModelProtocol, dict[str, Any]]
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    model_card_hash: str = Field(alias="modelCardHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_card(self) -> GlobalModelCard:
        if set(self.protocols) != set(GLOBAL_MODEL_PROTOCOLS) or set(self.metrics) != set(GLOBAL_MODEL_PROTOCOLS):
            raise ValueError("GlobalModel model-card protocol inventory is not canonical")
        global_model = self.protocols["global"]
        if (
            global_model.model_version_id != self.model_version_id
            or global_model.model_version_hash != self.model_version_hash
        ):
            raise ValueError("GlobalModel model-card top-level identity is not Global")
        if {item.license for item in self.licenses} != {"CC-BY-4.0", "MIT"}:
            raise ValueError("GlobalModel model-card license inventory is incomplete")
        if self.model_card_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"model_card_hash"})
        ):
            raise ValueError("modelCardHash mismatch")
        return self


class GlobalModelScenario(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(
        alias="schemaVersion"
    )
    scenario_id: Literal["russia-coordination-risk"] = Field(alias="scenarioId")
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    graph_version_hash: str | None = Field(alias="graphVersionHash", pattern=HASH)
    model_version_id: str | None = Field(alias="modelVersionId", max_length=300)
    enabled: bool
    unavailable_reason: str | None = Field(alias="unavailableReason")
    node_count: Literal[716] = Field(alias="nodeCount")
    edge_count: int = Field(alias="edgeCount", ge=0)
    protocols: tuple[GlobalModelProtocol, ...]
    metrics: dict[GlobalModelProtocol, GlobalModelMetric | None]
    limitations: tuple[str, ...]
    scenario_hash: str = Field(alias="scenarioHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_scenario(self) -> GlobalModelScenario:
        if self.protocols != GLOBAL_MODEL_PROTOCOLS or set(self.metrics) != set(GLOBAL_MODEL_PROTOCOLS):
            raise ValueError("scenario protocol metrics are not canonical")
        if self.enabled != (self.graph_version_hash is not None and self.model_version_id is not None):
            raise ValueError("scenario readiness identity is incomplete")
        if self.scenario_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"scenario_hash"})
        ):
            raise ValueError("scenarioHash mismatch")
        return self


class GlobalModelPreviewNode(GlobalModelContract):
    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=100)
    degree: int = Field(ge=0)
    structure_missing: bool = Field(alias="structureMissing")


class GlobalModelPreviewEdge(GlobalModelContract):
    id: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=100)
    target: str = Field(min_length=1, max_length=100)
    modality: Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim", "fused"]


class GlobalModelScenarioPreview(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    nodes: tuple[GlobalModelPreviewNode, ...]
    edges: tuple[GlobalModelPreviewEdge, ...]
    node_count: Literal[716] = Field(alias="nodeCount")
    edge_count: int = Field(alias="edgeCount", ge=0)
    partial_preview: bool = Field(alias="partialPreview")
    preview_hash: str = Field(alias="previewHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self) -> GlobalModelScenarioPreview:
        if self.preview_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"preview_hash"})
        ):
            raise ValueError("previewHash mismatch")
        return self


class GlobalModelRunRequest(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    task_id: Literal["coordination_risk"] = Field(alias="taskId")
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    protocol: GlobalModelProtocol
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    top_k: int = Field(default=50, alias="topK", ge=1, le=500)

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class GlobalModelRunStatus(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", pattern=r"^global-model-[0-9a-f]{32}$")
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = Field(ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    error_code: str | None = Field(default=None, alias="errorCode", max_length=100)


class GlobalModelRoute(GlobalModelContract):
    expert: str = Field(min_length=1, max_length=50)
    weight: float = Field(ge=0.0, le=1.0)


class GlobalModelModalityEvidence(GlobalModelContract):
    co_rt: int = Field(alias="coRT", ge=0)
    co_url: int = Field(alias="coURL", ge=0)
    hash_seq: int = Field(alias="hashSeq", ge=0)
    fast_rt: int = Field(alias="fastRT", ge=0)
    tweet_sim: int = Field(alias="tweetSim", ge=0)


class GlobalModelNodeFinding(GlobalModelContract):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    risk_band: Literal["high", "review", "low"] = Field(alias="riskBand")
    predicted_positive: bool = Field(alias="predictedPositive")
    structure_missing: bool = Field(alias="structureMissing")
    routes: tuple[GlobalModelRoute, ...] = Field(min_length=1, max_length=3)
    modality_evidence: GlobalModelModalityEvidence = Field(alias="modalityEvidence")


class GlobalModelRunResult(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", pattern=r"^global-model-[0-9a-f]{32}$")
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    task_id: Literal["coordination_risk"] = Field(alias="taskId")
    protocol: GlobalModelProtocol
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    corpus_hash: str = Field(alias="corpusHash", pattern=HASH)
    split_hash: str = Field(alias="splitHash", pattern=HASH)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    threshold: float = Field(ge=0.0, le=1.0)
    metrics: GlobalModelMetric
    findings: tuple[GlobalModelNodeFinding, ...]
    limitations: tuple[str, ...]
    completed_at: datetime = Field(alias="completedAt")
    result_hash: str = Field(alias="resultHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_result(self) -> GlobalModelRunResult:
        if len({item.node_id for item in self.findings}) != len(self.findings):
            raise ValueError("node findings must be unique")
        if tuple(item.rank for item in self.findings) != tuple(range(1, len(self.findings) + 1)):
            raise ValueError("node finding ranks must be contiguous")
        if self.result_hash != canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"result_hash"})
        ):
            raise ValueError("resultHash mismatch")
        return self


class GlobalModelRelationEvidence(GlobalModelContract):
    modality: GlobalModelModality
    raw_weight: float = Field(alias="rawWeight", allow_inf_nan=False)


class GlobalModelEvidenceNeighbor(GlobalModelContract):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=1.0)
    hop: Literal[1]
    risk_band: Literal["high", "review", "low"] = Field(alias="riskBand")
    predicted_positive: bool = Field(alias="predictedPositive")
    structure_missing: bool = Field(alias="structureMissing")
    modalities: tuple[GlobalModelModality, ...]
    relations: tuple[GlobalModelRelationEvidence, ...]

    @model_validator(mode="after")
    def validate_relations(self) -> GlobalModelEvidenceNeighbor:
        if self.modalities != tuple(item.modality for item in self.relations):
            raise ValueError("GlobalModel neighbor modalities do not match relation evidence")
        return self


class GlobalModelStructuralSignals(GlobalModelContract):
    fused_degree: int = Field(alias="fusedDegree", ge=0)
    structure_missing: bool = Field(alias="structureMissing")
    relation_neighbor_counts: GlobalModelModalityEvidence = Field(
        alias="relationNeighborCounts"
    )
    two_hop_node_count: int = Field(alias="twoHopNodeCount", ge=0, le=715)
    relation_evidence_role: Literal["explanationOnly"] = Field(
        alias="relationEvidenceRole"
    )


class GlobalModelEvidenceSubgraphNode(GlobalModelContract):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=100)
    score: float = Field(ge=0.0, le=1.0)
    hop: Literal[0, 1, 2]
    risk_band: Literal["high", "review", "low"] = Field(alias="riskBand")
    predicted_positive: bool = Field(alias="predictedPositive")
    structure_missing: bool = Field(alias="structureMissing")


class GlobalModelEvidenceSubgraphEdge(GlobalModelContract):
    id: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=100)
    target: str = Field(min_length=1, max_length=100)
    relations: tuple[GlobalModelRelationEvidence, ...]
    evidence_role: Literal["explanationOnly"] = Field(alias="evidenceRole")


class GlobalModelEvidenceSubgraph(GlobalModelContract):
    depth: Literal[2]
    node_count: int = Field(alias="nodeCount", ge=1, le=716)
    edge_count: int = Field(alias="edgeCount", ge=0)
    truncated: bool
    nodes: tuple[GlobalModelEvidenceSubgraphNode, ...]
    edges: tuple[GlobalModelEvidenceSubgraphEdge, ...]

    @model_validator(mode="after")
    def validate_graph(self) -> GlobalModelEvidenceSubgraph:
        node_ids = {item.node_id for item in self.nodes}
        if (
            self.node_count != len(self.nodes)
            or self.edge_count != len(self.edges)
            or len(node_ids) != len(self.nodes)
            or not any(item.hop == 0 for item in self.nodes)
            or any(edge.source not in node_ids or edge.target not in node_ids for edge in self.edges)
            or len({edge.id for edge in self.edges}) != len(self.edges)
        ):
            raise ValueError("GlobalModel evidence subgraph inventory is invalid")
        return self


class GlobalModelNodeEvidence(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", pattern=r"^global-model-[0-9a-f]{32}$")
    result_hash: str = Field(alias="resultHash", pattern=HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    threshold: float = Field(ge=0.0, le=1.0)
    node: GlobalModelNodeFinding
    neighbors: tuple[GlobalModelEvidenceNeighbor, ...]
    structural_signals: GlobalModelStructuralSignals = Field(alias="structuralSignals")
    evidence_subgraph: GlobalModelEvidenceSubgraph = Field(alias="evidenceSubgraph")
    limitation: str = Field(min_length=1, max_length=500)
    evidence_hash: str = Field(alias="evidenceHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self) -> GlobalModelNodeEvidence:
        subgraph_nodes = {item.node_id: item for item in self.evidence_subgraph.nodes}
        root = subgraph_nodes.get(self.node.node_id)
        if (
            root is None
            or root.hop != 0
            or root.score != self.node.score
            or self.structural_signals.structure_missing != self.node.structure_missing
            or self.structural_signals.fused_degree != len(self.neighbors)
            or any(item.node_id not in subgraph_nodes for item in self.neighbors)
        ):
            raise ValueError("GlobalModel evidence payload is internally inconsistent")
        if self.evidence_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        ):
            raise ValueError("evidenceHash mismatch")
        return self


class GlobalModelReviewRequest(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    node_id: str = Field(alias="nodeId", min_length=1, max_length=100)
    decision: Literal["confirmed", "rejected", "pending"]
    reason: str = Field(min_length=1, max_length=1000)


class GlobalModelReviewRecord(GlobalModelContract):
    schema_version: Literal["socialgraph-fm.gfm-global-model/1.0"] = Field(alias="schemaVersion")
    review_id: str = Field(alias="reviewId", pattern=r"^review-[0-9a-f]{32}$")
    run_id: str = Field(alias="runId", pattern=r"^global-model-[0-9a-f]{32}$")
    node_id: str = Field(alias="nodeId", min_length=1, max_length=100)
    decision: Literal["confirmed", "rejected", "pending"]
    reason: str = Field(min_length=1, max_length=1000)
    created_at: datetime = Field(alias="createdAt")
    review_hash: str = Field(alias="reviewHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self) -> GlobalModelReviewRecord:
        if self.review_hash != canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"review_hash"})
        ):
            raise ValueError("reviewHash mismatch")
        return self


def build_unavailable_capabilities() -> GlobalModelCapabilities:
    payload = {
        "schemaVersion": GLOBAL_MODEL_SCHEMA_VERSION,
        "channel": "socialgraph-global",
        "releaseLabel": "SocialGraph-FM Global",
        "seed": GLOBAL_MODEL_SEED,
        "servingReady": False,
        "unavailableReason": "GFM_GLOBAL_MODEL_NOT_INSTALLED",
        "taskId": GLOBAL_MODEL_TASK_ID,
        "datasetVersionId": GLOBAL_MODEL_DATASET_VERSION_ID,
        "model": None,
    }
    payload["capabilityHash"] = canonical_sha256(payload)
    return GlobalModelCapabilities.model_validate(payload)


__all__ = [
    "GLOBAL_MODEL_DATASET_VERSION_ID",
    "GLOBAL_MODEL_HEALTH_SCHEMA_VERSION",
    "GLOBAL_MODEL_CARD_SCHEMA_VERSION",
    "GLOBAL_MODEL_PROTOCOLS",
    "GLOBAL_MODEL_SCHEMA_VERSION",
    "GLOBAL_MODEL_SEED",
    "GLOBAL_MODEL_TASK_ID",
    "GlobalModelCapabilities",
    "GlobalModelHealth",
    "GlobalModelCard",
    "GlobalModelCapability",
    "GlobalModelProtocolModel",
    "GlobalModelNodeEvidence",
    "GlobalModelReviewRecord",
    "GlobalModelReviewRequest",
    "GlobalModelRunRequest",
    "GlobalModelRunResult",
    "GlobalModelRunStatus",
    "GlobalModelScenario",
    "GlobalModelScenarioPreview",
    "build_unavailable_capabilities",
]
