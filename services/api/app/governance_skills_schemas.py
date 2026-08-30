"""Strict public and internal contracts for SocialGraph-FM Governance skills."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, get_args

from pydantic import ConfigDict, Field, field_validator, model_validator

from .gfm_governance_schemas import (
    ARTIFACT_PATTERN,
    CASE_PATTERN,
    HASH_PATTERN,
    RUN_PATTERN,
    FrozenModel,
)
from .governance_skill_runtime.catalog import load_product_skill_catalog

SKILL_SCHEMA_VERSION: Literal["socialgraph-fm.governance-skills/1.0"] = (
    "socialgraph-fm.governance-skills/1.0"
)
ASSISTANT_SCHEMA_VERSION: Literal["socialgraph-fm.governance-assistant/1.0"] = (
    "socialgraph-fm.governance-assistant/1.0"
)
DISPATCH_SCHEMA_VERSION: Literal["socialgraph-fm.governance-assistant-dispatch/1.0"] = (
    "socialgraph-fm.governance-assistant-dispatch/1.0"
)
GOVERNANCE_COMMAND_SCHEMA_VERSION: Literal["socialgraph-fm.governance-command/1.0"] = (
    "socialgraph-fm.governance-command/1.0"
)
GOVERNANCE_RESULT_SCHEMA_VERSION: Literal["socialgraph-fm.governance-result/1.0"] = (
    "socialgraph-fm.governance-result/1.0"
)

SkillName = Literal[
    "inspect_graph",
    "run_governance_analysis",
    "get_evidence_subgraph",
    "discover_coordination_groups",
    "rank_coordination_relations",
    "retrieve_similar_cases",
    "get_model_dataset_cards",
    "draft_review_report",
]
ReadOnlySkillName = Literal[
    "inspect_graph",
    "get_evidence_subgraph",
    "discover_coordination_groups",
    "rank_coordination_relations",
    "retrieve_similar_cases",
    "get_model_dataset_cards",
]
ConfirmationAction = Literal["run_governance_analysis", "save_draft_report", "submit_review"]
DispatchIntent = Literal[
    "answer",
    "start_analysis",
    "open_review",
    "submit_review",
    "draft_report",
]
AnswerMode = Literal[
    "overview",
    "analysis_summary",
    "coordination_summary",
    "evidence_requirements",
    "review_guidance",
    "method_scope",
    "knowledge",
    "case_draft",
]
GenerationMode = Literal["llm_assisted", "deterministic_report"]
NarrationMode = Literal["auto", "deterministic_only"]
FallbackPhase = Literal["intent", "planning", "skill_execution", "narration"]
EvidenceSourceKind = Literal["graph", "skill", "knowledge", "case"]
InternalCommand = Literal[
    "inspect_graph",
    "run_governance_analysis",
    "get_evidence_subgraph",
    "discover_coordination_groups",
    "rank_coordination_relations",
    "retrieve_similar_cases",
    "get_model_dataset_cards",
    "draft_review_report",
    "trace_evidence",
    "summarize_groups",
    "inspect_relations",
    "find_similar_cases",
    "get_model_card",
    "draft_report",
    "search_knowledge",
    "index_case",
]


class SkillsModel(FrozenModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=False,
    )


class GraphIdentity(SkillsModel):
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)


class ModelIdentity(SkillsModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)


class InspectGraphParams(SkillsModel):
    scope_node_ids: tuple[str, ...] = Field(
        default=(), alias="scopeNodeIds", max_length=100
    )
    run_id: str | None = Field(default=None, alias="runId", pattern=RUN_PATTERN)
    candidate_limit: int = Field(default=5, alias="candidateLimit", ge=1, le=5)

    @model_validator(mode="after")
    def validate_scope(self) -> InspectGraphParams:
        if tuple(sorted(set(self.scope_node_ids))) != self.scope_node_ids:
            raise ValueError("scopeNodeIds must be unique and sorted")
        return self


class TraceEvidenceParams(SkillsModel):
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)


class SummarizeGroupsParams(SkillsModel):
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class InspectRelationsParams(SkillsModel):
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)
    relation_kind: Literal["factual", "potential"] = Field(
        default="factual", alias="relationKind"
    )
    modalities: tuple[
        Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"], ...
    ] = Field(default=(), max_length=5)

    @model_validator(mode="after")
    def validate_modalities(self) -> InspectRelationsParams:
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique")
        if self.relation_kind == "potential" and self.modalities:
            raise ValueError("potential relations do not accept factual modalities")
        return self


class CaseKindEntry(SkillsModel):
    kind: Literal["node", "relation", "group"]
    target_ids: tuple[str, ...] = Field(alias="targetIds", min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_targets(self) -> CaseKindEntry:
        if tuple(sorted(set(self.target_ids))) != self.target_ids:
            raise ValueError("targetIds must be unique and sorted")
        return self


class FindSimilarCasesParams(SkillsModel):
    case_id: str | None = Field(default=None, alias="caseId", pattern=CASE_PATTERN)
    run_id: str | None = Field(default=None, alias="runId", pattern=RUN_PATTERN)
    kind_entries: tuple[CaseKindEntry, ...] = Field(
        default=(), alias="kindEntries", max_length=3
    )
    limit: int = Field(default=10, ge=1, le=25)

    @model_validator(mode="after")
    def validate_query(self) -> FindSimilarCasesParams:
        by_case = self.case_id is not None
        by_targets = self.run_id is not None or bool(self.kind_entries)
        if by_case == by_targets:
            raise ValueError("provide either caseId or runId/kindEntries")
        if by_targets and (self.run_id is None or not self.kind_entries):
            raise ValueError("runId and kindEntries are required together")
        order = {"node": 0, "relation": 1, "group": 2}
        kinds = tuple(entry.kind for entry in self.kind_entries)
        if len(set(kinds)) != len(kinds) or tuple(sorted(kinds, key=order.__getitem__)) != kinds:
            raise ValueError("kindEntries must be unique and ordered node/relation/group")
        return self


class GetModelCardParams(SkillsModel):
    pass


class DraftReportParams(SkillsModel):
    case_id: str = Field(alias="caseId", pattern=CASE_PATTERN)
    format: Literal["markdown", "json"]


class RunGovernanceAnalysisParams(SkillsModel):
    top_k: int = Field(default=100, alias="topK", ge=1, le=10_000)
    protocol: Literal["global"]


SkillParams = (
    InspectGraphParams
    | TraceEvidenceParams
    | SummarizeGroupsParams
    | InspectRelationsParams
    | FindSimilarCasesParams
    | GetModelCardParams
    | DraftReportParams
    | RunGovernanceAnalysisParams
)

_PARAM_MODELS: dict[str, type[SkillParams]] = {
    "inspect_graph": InspectGraphParams,
    "run_governance_analysis": RunGovernanceAnalysisParams,
    "get_evidence_subgraph": TraceEvidenceParams,
    "discover_coordination_groups": SummarizeGroupsParams,
    "rank_coordination_relations": InspectRelationsParams,
    "retrieve_similar_cases": FindSimilarCasesParams,
    "get_model_dataset_cards": GetModelCardParams,
    "draft_review_report": DraftReportParams,
}

_CANONICAL_PRODUCT_SKILLS = load_product_skill_catalog()
if tuple(get_args(SkillName)) != _CANONICAL_PRODUCT_SKILLS.names:
    raise RuntimeError("SkillName does not match the canonical SocialGraph-FM Governance catalog")
if tuple(get_args(ReadOnlySkillName)) != tuple(
    item.name for item in _CANONICAL_PRODUCT_SKILLS.items if item.read_only
):
    raise RuntimeError("ReadOnlySkillName does not match the canonical SocialGraph-FM Governance catalog")
if tuple(_PARAM_MODELS) != _CANONICAL_PRODUCT_SKILLS.names:
    raise RuntimeError("public parameter models do not match the canonical SocialGraph-FM Governance catalog")


class SkillExecuteRequest(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    skill: SkillName
    graph: GraphIdentity
    model: ModelIdentity
    params: dict[str, Any]

    @model_validator(mode="after")
    def validate_params(self) -> SkillExecuteRequest:
        _PARAM_MODELS[self.skill].model_validate(self.params)
        return self

    def parsed_params(self) -> SkillParams:
        return _PARAM_MODELS[self.skill].model_validate(self.params)


class SkillInvocationRequest(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    graph: GraphIdentity
    model: ModelIdentity
    params: dict[str, Any]

    def bind(self, skill: SkillName) -> SkillExecuteRequest:
        return SkillExecuteRequest(
            schemaVersion=self.schema_version,
            skill=skill,
            graph=self.graph,
            model=self.model,
            params=self.params,
        )


class ConfirmationTicket(SkillsModel):
    token: str = Field(pattern=r"^governance-confirm-[0-9a-f]{64}$")
    action: ConfirmationAction
    request_digest: str = Field(alias="requestDigest", pattern=HASH_PATTERN)
    expires_at: datetime = Field(alias="expiresAt")


class SkillExecutionResponse(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    execution_id: str = Field(alias="executionId", pattern=r"^governance-exec-[0-9a-f]{32}$")
    skill: SkillName
    status: Literal["completed", "confirmation_required"]
    result: dict[str, Any]
    confirmation: ConfirmationTicket | None = None
    provenance: dict[str, Any]
    audit_hash: str = Field(alias="auditHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_confirmation(self) -> SkillExecutionResponse:
        if (self.status == "confirmation_required") != (self.confirmation is not None):
            raise ValueError("confirmation must match status")
        return self


class SkillConfirmationRequest(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    token: str = Field(pattern=r"^governance-confirm-[0-9a-f]{64}$")


class SkillConfirmationResponse(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    action: ConfirmationAction
    status: Literal["completed"]
    result: dict[str, Any]
    audit_hash: str = Field(alias="auditHash", pattern=HASH_PATTERN)


class SkillDescriptor(SkillsModel):
    name: SkillName
    read_only: bool = Field(alias="readOnly")
    confirmation_required: bool = Field(alias="confirmationRequired")
    description: str
    parameter_schema: dict[str, Any] = Field(alias="parameterSchema")


class SkillCatalog(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    items: tuple[SkillDescriptor, ...] = Field(min_length=8, max_length=8)
    catalog_hash: str = Field(alias="catalogHash", pattern=HASH_PATTERN)


class KnowledgeSearchRequest(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    graph: GraphIdentity
    model: ModelIdentity
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=5, ge=1, le=10)


class KnowledgeItem(SkillsModel):
    source_label: str = Field(alias="sourceLabel", min_length=1, max_length=200)
    source_uri: str = Field(alias="sourceUri", min_length=1, max_length=500)
    content_hash: str = Field(alias="contentHash", pattern=HASH_PATTERN)
    chunk_hash: str = Field(alias="chunkHash", pattern=HASH_PATTERN)
    text: str = Field(max_length=2_000)
    rank: int = Field(ge=1, le=10)

    @field_validator("source_uri")
    @classmethod
    def reject_local_path(cls, value: str) -> str:
        lowered = value.lower()
        if (
            lowered.startswith("file:")
            or value.startswith(("/", "\\"))
            or "\\" in value
            or (len(value) > 2 and value[1:3] == ":/")
        ):
            raise ValueError("sourceUri must not expose a local path")
        return value


class KnowledgeSearchResponse(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    items: tuple[KnowledgeItem, ...] = Field(max_length=10)
    index_hash: str = Field(alias="indexHash", pattern=HASH_PATTERN)
    audit_hash: str = Field(alias="auditHash", pattern=HASH_PATTERN)


class SimilarCasesSearchRequest(FindSimilarCasesParams):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    graph: GraphIdentity
    model: ModelIdentity


class SimilarCasesSearchResponse(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-skills/1.0"] = Field(
        alias="schemaVersion"
    )
    query: dict[str, Any]
    items: tuple[dict[str, Any], ...] = Field(max_length=25)
    index_hash: str = Field(alias="indexHash", pattern=HASH_PATTERN)
    backfill: dict[str, int]
    audit_hash: str = Field(alias="auditHash", pattern=HASH_PATTERN)


class AssistantContext(SkillsModel):
    run_id: str | None = Field(default=None, alias="runId", pattern=RUN_PATTERN)
    case_id: str | None = Field(default=None, alias="caseId", pattern=CASE_PATTERN)
    selected_node_ids: tuple[str, ...] = Field(
        default=(), alias="selectedNodeIds", max_length=25
    )


class AssistantTurnRequest(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-assistant/1.0"] = Field(
        alias="schemaVersion"
    )
    graph: GraphIdentity
    model: ModelIdentity
    message: str = Field(min_length=1, max_length=2_000)
    context: AssistantContext = Field(default_factory=AssistantContext)


class AssistantSkillTrace(SkillsModel):
    skill: ReadOnlySkillName
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)


class AssistantTurnResponse(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-assistant/1.0"] = Field(
        alias="schemaVersion"
    )
    turn_id: str = Field(alias="turnId", pattern=r"^governance-turn-[0-9a-f]{32}$")
    answer: str = Field(min_length=1, max_length=8_000)
    deterministic_fallback: bool = Field(alias="deterministicFallback")
    skill_calls: tuple[AssistantSkillTrace, ...] = Field(alias="skillCalls", max_length=4)
    cited_hashes: tuple[str, ...] = Field(alias="citedHashes", max_length=50)
    audit_hash: str = Field(alias="auditHash", pattern=HASH_PATTERN)


class AssistantDispatchTarget(SkillsModel):
    target_type: Literal["node", "relation", "group"] = Field(alias="targetType")
    target_id: str = Field(alias="targetId", min_length=1, max_length=300)


class AssistantDispatchContext(SkillsModel):
    run_id: str | None = Field(default=None, alias="runId", pattern=RUN_PATTERN)
    case_id: str | None = Field(default=None, alias="caseId", pattern=CASE_PATTERN)
    case_hash: str | None = Field(default=None, alias="caseHash", pattern=HASH_PATTERN)
    selected_target: AssistantDispatchTarget | None = Field(default=None, alias="selectedTarget")
    top_k: int = Field(default=100, alias="topK", ge=1, le=10_000)
    review_decision: Literal["confirmed", "rejected", "pending"] | None = Field(
        default=None, alias="reviewDecision"
    )
    review_reason: str | None = Field(default=None, alias="reviewReason", min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def validate_case_hash(self) -> AssistantDispatchContext:
        if self.case_hash is not None and self.case_id is None:
            raise ValueError("caseHash requires caseId")
        return self


class AssistantDispatchRequest(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-assistant-dispatch/1.0"] = Field(
        alias="schemaVersion"
    )
    graph: GraphIdentity
    model: ModelIdentity
    message: str = Field(min_length=1, max_length=2_000)
    intent: Literal["answer"] | None = None
    answer_mode: AnswerMode | None = Field(default=None, alias="answerMode")
    narration_mode: NarrationMode = Field(default="auto", alias="narrationMode")
    context: AssistantDispatchContext = Field(default_factory=AssistantDispatchContext)

    @model_validator(mode="after")
    def validate_answer_mode(self) -> AssistantDispatchRequest:
        if self.answer_mode is not None and self.intent != "answer":
            raise ValueError("answerMode requires the explicit answer intent")
        if self.answer_mode == "case_draft" and self.context.case_id is None:
            raise ValueError("case_draft requires caseId")
        return self


class AssistantDispatchNavigation(SkillsModel):
    view: Literal["governance_review"]
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    case_id: str | None = Field(default=None, alias="caseId", pattern=CASE_PATTERN)
    target: AssistantDispatchTarget | None = None


class AssistantEvidenceRef(SkillsModel):
    label: str = Field(min_length=1, max_length=200)
    source_kind: EvidenceSourceKind = Field(alias="sourceKind")
    hash: str = Field(pattern=HASH_PATTERN)


class AssistantDispatchResponse(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-assistant-dispatch/1.0"] = Field(
        alias="schemaVersion"
    )
    dispatch_id: str = Field(alias="dispatchId", pattern=r"^governance-dispatch-[0-9a-f]{32}$")
    intent: DispatchIntent
    answer_mode: AnswerMode | None = Field(default=None, alias="answerMode")
    status: Literal["completed", "confirmation_required", "blocked"]
    answer: str = Field(min_length=1, max_length=4_000)
    result: dict[str, Any] = Field(default_factory=dict)
    deterministic_fallback: bool = Field(alias="deterministicFallback")
    generation_mode: GenerationMode | None = Field(default=None, alias="generationMode")
    fallback_phase: FallbackPhase | None = Field(default=None, alias="fallbackPhase")
    reason_code: str | None = Field(
        default=None, alias="reasonCode", pattern=r"^[A-Z][A-Z0-9_]{1,99}$"
    )
    evidence_refs: tuple[AssistantEvidenceRef, ...] = Field(
        default=(), alias="evidenceRefs", max_length=50
    )
    confirmation: ConfirmationTicket | None = None
    navigation: AssistantDispatchNavigation | None = None
    skill_calls: tuple[AssistantSkillTrace, ...] = Field(alias="skillCalls", max_length=4)
    cited_hashes: tuple[str, ...] = Field(alias="citedHashes", max_length=50)
    audit_hash: str = Field(alias="auditHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_effect(self) -> AssistantDispatchResponse:
        if (self.intent == "answer") != (self.answer_mode is not None):
            raise ValueError("answerMode must match an answer intent")
        if self.intent != "answer" and self.skill_calls:
            raise ValueError("skillCalls are only valid for answer intents")
        expected_action = {
            "start_analysis": "run_governance_analysis",
            "submit_review": "submit_review",
            "draft_report": "save_draft_report",
        }.get(self.intent)
        if self.status == "confirmation_required":
            if self.confirmation is None or self.confirmation.action != expected_action:
                raise ValueError("confirmation action does not match dispatch intent")
        elif self.confirmation is not None:
            raise ValueError("confirmation is only valid for confirmation_required status")
        if (self.intent == "open_review" and self.status == "completed") != (
            self.navigation is not None
        ):
            raise ValueError("navigation must match a completed open_review intent")
        return self


class GovernanceCommandEnvelope(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-command/1.0"] = Field(
        alias="schemaVersion"
    )
    command_id: str = Field(alias="commandId", pattern=r"^governance-command-[0-9a-f]{32}$")
    command: InternalCommand
    graph: GraphIdentity
    model: ModelIdentity
    params: dict[str, Any]


class GfmResultProvenance(SkillsModel):
    generated_at: datetime = Field(alias="generatedAt")
    implementation_version: str = Field(
        alias="implementationVersion", min_length=1, max_length=200
    )
    input_hash: str = Field(alias="inputHash", pattern=HASH_PATTERN)


class GovernanceResultEnvelope(SkillsModel):
    schema_version: Literal["socialgraph-fm.governance-result/1.0"] = Field(
        alias="schemaVersion"
    )
    command_id: str = Field(alias="commandId", pattern=r"^governance-command-[0-9a-f]{32}$")
    command: InternalCommand
    status: Literal["completed", "confirmation_required"]
    graph: GraphIdentity
    model: ModelIdentity
    result: dict[str, Any]
    provenance: GfmResultProvenance
    warnings: tuple[str, ...] = Field(max_length=20)


class IndexCaseReceipt(SkillsModel):
    case_id: str = Field(alias="caseId", pattern=CASE_PATTERN)
    record_hash: str = Field(alias="recordHash", pattern=HASH_PATTERN)
    index_hash: str = Field(alias="indexHash", pattern=HASH_PATTERN)
    indexed_at: datetime = Field(alias="indexedAt")
    idempotent: bool


__all__ = [name for name in globals() if not name.startswith("_")]
