from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

AnalysisTask = Literal[
    "overview",
    "centrality",
    "bridge_detection",
    "community",
    "link_prediction",
    "node_role",
    "similar_structure",
]
IntentSource = Literal["llm"]
GraphViewMode = Literal["global", "local", "path"]
LayoutPreset = Literal["balanced", "compact", "spread"]
AnalysisOverlay = Literal["degree", "articulation", "components", "community"]
FilterValue: TypeAlias = StrictStr | StrictInt | StrictFloat | StrictBool
ColumnDataType = Literal["string", "integer", "float", "boolean", "datetime", "unknown"]
GraphDirectedness = Literal["directed", "undirected", "unspecified"]
GraphBuildSource = Literal["llm"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class TimeRange(ApiModel):
    start: str | None = Field(default=None, min_length=1, max_length=64)
    end: str | None = Field(default=None, min_length=1, max_length=64)


class GraphContextSummary(ApiModel):
    """The complete allowlist of graph information that may reach the LLM."""

    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    density: float = Field(ge=0)
    connected_components: int = Field(alias="connectedComponents", ge=0)
    node_types: list[str] = Field(alias="nodeTypes", default_factory=list, max_length=50)
    edge_types: list[str] = Field(alias="edgeTypes", default_factory=list, max_length=50)
    has_weight: bool = Field(alias="hasWeight")
    has_timestamp: bool = Field(alias="hasTimestamp")
    time_range: TimeRange | None = Field(alias="timeRange", default=None)

    @field_validator("node_types", "edge_types")
    @classmethod
    def normalize_type_vocabularies(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            cleaned = value.strip()
            if cleaned and cleaned not in result:
                result.append(cleaned[:100])
        return result


class NormalizeIntentRequest(ApiModel):
    text: str = Field(min_length=1, max_length=4_000)
    graph_context: GraphContextSummary | None = Field(alias="graphContext", default=None)

    @field_validator("text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class GraphColumnProfile(ApiModel):
    """Aggregate-only column metadata; values and source rows are deliberately forbidden."""

    name: str = Field(min_length=1, max_length=200)
    inferred_type: ColumnDataType = Field(alias="inferredType")
    non_null_count: int = Field(alias="nonNullCount", ge=0)
    null_count: int = Field(alias="nullCount", ge=0)
    unique_count: int = Field(alias="uniqueCount", ge=0)

    @field_validator("name")
    @classmethod
    def normalize_column_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("column name must not be blank")
        return cleaned


class GraphBuildFileProfile(ApiModel):
    role: Literal["nodes", "edges"]
    column_profiles: list[GraphColumnProfile] = Field(
        alias="columnProfiles", min_length=1, max_length=200
    )

    @field_validator("column_profiles")
    @classmethod
    def unique_column_names(cls, values: list[GraphColumnProfile]) -> list[GraphColumnProfile]:
        seen: set[str] = set()
        for profile in values:
            key = profile.name.casefold()
            if key in seen:
                raise ValueError(f"duplicate column profile: {profile.name}")
            seen.add(key)
        return values


class NormalizeGraphBuildIntentRequest(ApiModel):
    description: str = Field(min_length=1, max_length=4_000)
    column_profiles: list[GraphColumnProfile] | None = Field(
        alias="columnProfiles",
        default=None,
        min_length=1,
        max_length=200,
    )
    files: list[GraphBuildFileProfile] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )

    @field_validator("description")
    @classmethod
    def normalize_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("description must not be blank")
        return cleaned

    @field_validator("column_profiles")
    @classmethod
    def unique_column_names(cls, values: list[GraphColumnProfile] | None) -> list[GraphColumnProfile] | None:
        if values is None:
            return None
        seen: set[str] = set()
        for profile in values:
            key = profile.name.casefold()
            if key in seen:
                raise ValueError(f"duplicate column profile: {profile.name}")
            seen.add(key)
        return values

    @model_validator(mode="after")
    def validate_profile_shape(self) -> NormalizeGraphBuildIntentRequest:
        if (self.column_profiles is None) == (self.files is None):
            raise ValueError("provide exactly one of columnProfiles or files")
        if self.files is not None:
            roles = [file.role for file in self.files]
            if sorted(roles) != ["edges", "nodes"]:
                raise ValueError("files must contain exactly one nodes and one edges profile")
        return self


class GraphBuildColumnMapping(ApiModel):
    source_column: str | None = Field(alias="sourceColumn", default=None, max_length=200)
    target_column: str | None = Field(alias="targetColumn", default=None, max_length=200)
    edge_type_column: str | None = Field(alias="edgeTypeColumn", default=None, max_length=200)
    weight_column: str | None = Field(alias="weightColumn", default=None, max_length=200)
    timestamp_column: str | None = Field(alias="timestampColumn", default=None, max_length=200)


class GraphBuildNodeMapping(ApiModel):
    id_column: str | None = Field(alias="idColumn", default=None, max_length=200)
    label_column: str | None = Field(alias="labelColumn", default=None, max_length=200)
    type_column: str | None = Field(alias="typeColumn", default=None, max_length=200)


class GraphBuildIntentMeta(ApiModel):
    schema_version: Literal["1.0", "1.1"] = Field(alias="schemaVersion", default="1.0")
    source: GraphBuildSource
    request_id: str = Field(alias="requestId", min_length=1, max_length=128)
    model: str | None = Field(default=None, max_length=200)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class GraphBuildIntentResponse(ApiModel):
    kind: Literal["graph_build_intent"] = "graph_build_intent"
    mapping: GraphBuildColumnMapping
    node_mapping: GraphBuildNodeMapping | None = Field(alias="nodeMapping", default=None)
    directedness: GraphDirectedness = "unspecified"
    confidence: float = Field(ge=0, le=1)
    requires_mapping: bool = Field(alias="requiresMapping")
    meta: GraphBuildIntentMeta


class ModelGraphBuildIntentOutput(ApiModel):
    source_column: str | None = Field(alias="sourceColumn", default=None, max_length=200)
    target_column: str | None = Field(alias="targetColumn", default=None, max_length=200)
    edge_type_column: str | None = Field(alias="edgeTypeColumn", default=None, max_length=200)
    weight_column: str | None = Field(alias="weightColumn", default=None, max_length=200)
    timestamp_column: str | None = Field(alias="timestampColumn", default=None, max_length=200)
    node_id_column: str | None = Field(alias="nodeIdColumn", default=None, max_length=200)
    node_label_column: str | None = Field(alias="nodeLabelColumn", default=None, max_length=200)
    node_type_column: str | None = Field(alias="nodeTypeColumn", default=None, max_length=200)
    directedness: GraphDirectedness = "unspecified"
    confidence: float = Field(ge=0, le=1)


class ViewCommand(ApiModel):
    """A bounded presentation command; it never contains graph facts or node IDs."""

    mode: GraphViewMode | None = None
    focus_terms: list[str] = Field(alias="focusTerms", default_factory=list, max_length=20)
    depth: Literal[1, 2, 3] | None = None
    node_type_terms: list[str] = Field(alias="nodeTypeTerms", default_factory=list, max_length=20)
    edge_type_terms: list[str] = Field(alias="edgeTypeTerms", default_factory=list, max_length=20)
    layout_preset: LayoutPreset | None = Field(alias="layoutPreset", default=None)
    overlay: AnalysisOverlay | None = None

    @field_validator("focus_terms", "node_type_terms", "edge_type_terms")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for value in values:
            cleaned = value.strip()[:80]
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                result.append(cleaned)
        return result


class IntentMeta(ApiModel):
    schema_version: Literal["1.1"] = Field(alias="schemaVersion", default="1.1")
    source: IntentSource
    request_id: str = Field(alias="requestId", min_length=1, max_length=128)
    model: str | None = Field(default=None, max_length=200)
    warnings: list[str] = Field(default_factory=list, max_length=20)


class ChatIntentResponse(ApiModel):
    kind: Literal["chat"] = "chat"
    reply: str = Field(min_length=1, max_length=1_000)
    meta: IntentMeta


class AnalysisIntentResponse(ApiModel):
    kind: Literal["analysis_request"] = "analysis_request"
    normalized_text: str = Field(alias="normalizedText", min_length=1, max_length=1_000)
    task: AnalysisTask
    targets: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)
    time_range: TimeRange | None = Field(alias="timeRange", default=None)
    filters: dict[str, FilterValue] = Field(default_factory=dict, max_length=20)
    view: ViewCommand | None = None
    meta: IntentMeta


IntentNormalizationResponse = Annotated[
    ChatIntentResponse | AnalysisIntentResponse,
    Field(discriminator="kind"),
]


class ModelChatOutput(ApiModel):
    kind: Literal["chat"]
    reply: str = Field(min_length=1, max_length=1_000)


class ModelAnalysisOutput(ApiModel):
    kind: Literal["analysis_request"]
    normalized_text: str = Field(alias="normalizedText", min_length=1, max_length=1_000)
    task: AnalysisTask
    targets: list[str] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)
    time_range: TimeRange | None = Field(alias="timeRange", default=None)
    filters: dict[str, FilterValue] = Field(default_factory=dict, max_length=20)
    view: ViewCommand | None = None


ModelIntentOutput = Annotated[ModelChatOutput | ModelAnalysisOutput, Field(discriminator="kind")]


class HealthResponse(ApiModel):
    status: Literal["ok"] = "ok"
    service: Literal["socialgraph-fm-api"] = "socialgraph-fm-api"
    version: Literal["0.1.0"] = "0.1.0"


class IntentNormalizationCapability(ApiModel):
    configured: bool
    mode: Literal["llm_required"] = "llm_required"
    provider: Literal["openai_compatible"] | None = None
    model: str | None = None
    api_mode: Literal["chat_completions"] = Field(
        alias="apiMode", default="chat_completions"
    )
    connection_status: Literal[
        "not_configured", "configured_unverified", "call_succeeded", "error"
    ] = Field(alias="connectionStatus")


class AnalysisCapabilities(ApiModel):
    local_tasks: list[AnalysisTask] = Field(alias="localTasks")
    gfm_tasks: list[AnalysisTask] = Field(alias="gfmTasks")
    gfm_connected: bool = Field(alias="gfmConnected", default=False)


class DataBoundaryCapability(ApiModel):
    sends_raw_graph: Literal[False] = Field(alias="sendsRawGraph", default=False)
    allowed_graph_fields: list[str] = Field(alias="allowedGraphFields")


class ResearchDatasetCapability(ApiModel):
    persistent_artifacts: Literal[True] = Field(alias="persistentArtifacts", default=True)
    trusted_local_enabled: bool = Field(alias="trustedLocalEnabled")
    loopback_only: Literal[True] = Field(alias="loopbackOnly", default=True)
    safe_upload_formats: list[str] = Field(alias="safeUploadFormats")


class RuntimeContractCapability(ApiModel):
    """Version handshake used to fail closed on stale frontend/backend pairs."""

    build_id: str = Field(alias="buildId")
    api_contract: Literal["socialgraph-fm-api/1.1"] = Field(
        alias="apiContract", default="socialgraph-fm-api/1.1"
    )
    storage_schema: Literal["dataset-store/2"] = Field(
        alias="storageSchema", default="dataset-store/2"
    )
    dataset_artifact_schemas: list[Literal["1.0", "2.0", "2.1", "2.2"]] = Field(
        alias="datasetArtifactSchemas"
    )
    training_ref_schemas: list[Literal["1.0", "1.1"]] = Field(
        alias="trainingRefSchemas"
    )
    graph_handoff_schemas: list[
        Literal["socialgraph-fm-graph/1.0", "socialgraph-fm-graph/1.1"]
    ] = Field(alias="graphHandoffSchemas")
    graph_fact_hash: Literal["graph-fact-hash/1"] = Field(
        alias="graphFactHash", default="graph-fact-hash/1"
    )
    converter_environment_fingerprint: str = Field(alias="converterEnvironmentFingerprint")
    converter_environment: dict[str, object] = Field(alias="converterEnvironment")


class CapabilitiesResponse(ApiModel):
    intent_normalization: IntentNormalizationCapability = Field(alias="intentNormalization")
    analysis: AnalysisCapabilities
    data_boundary: DataBoundaryCapability = Field(alias="dataBoundary")
    research_datasets: ResearchDatasetCapability = Field(alias="researchDatasets")
    runtime: RuntimeContractCapability
