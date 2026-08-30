"""Strict no-LLM tool contracts over registered Task-5 records only."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from socialgraph_gfm.canonical import canonical_json

from .bundle import CoreGraphBundle
from .governance import (
    CalibratedConfidence,
    GovernanceFinding,
    MANUAL_REVIEW_LIMITATION,
    NON_CAUSAL_LIMITATION,
    TaskId,
    validate_similar_case_provenance,
)
from .knowledge import KnowledgeSearchResult, KnowledgeStore
from .retrieval import StructuralIndex, StructuralSearchResult


_HASH_PATTERN = r"^[0-9a-f]{64}$"


def _markdown_data_identifier(value: str) -> str:
    encoded = json.dumps(value, ensure_ascii=False)
    encoded = (
        encoded.replace("`", r"\u0060")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("&", r"\u0026")
        .replace("[", r"\u005b")
        .replace("]", r"\u005d")
        .replace("(", r"\u0028")
        .replace(")", r"\u0029")
    )
    return f"`{encoded}`"


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class _StrictRequest(_StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=False,
        validate_by_alias=True,
        validate_by_name=False,
        protected_namespaces=("model_dump",),
        strict=True,
    )


def _canonical_string_array(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError("value must be a canonical JSON string array")
    return tuple(value)


class InspectGraphRequest(_StrictRequest):
    schema_version: Literal["socialgraph-fm.core-skill.inspect-graph.request/2.0"] = Field(
        alias="schemaVersion"
    )
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    scope_node_ids: tuple[str, ...] = Field(default=(), alias="scopeNodeIds")

    _canonical_scope = field_validator("scope_node_ids", mode="before")(_canonical_string_array)


class InspectGraphResponse(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-skill.inspect-graph.response/2.0"] = Field(
        default="socialgraph-fm.core-skill.inspect-graph.response/2.0", alias="schemaVersion"
    )
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    directed: bool
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    scope_node_ids: tuple[str, ...] = Field(alias="scopeNodeIds")
    limitations: tuple[str, ...]


class RunCoreTaskRequest(_StrictRequest):
    schema_version: Literal["socialgraph-fm.core-skill.run-core-task.request/2.0"] = Field(
        alias="schemaVersion"
    )
    task_id: TaskId = Field(alias="taskId")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    scope_node_ids: tuple[str, ...] = Field(default=(), alias="scopeNodeIds")

    _canonical_scope = field_validator("scope_node_ids", mode="before")(_canonical_string_array)


class RunCoreTaskResponse(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-skill.run-core-task.response/2.0"] = Field(
        default="socialgraph-fm.core-skill.run-core-task.response/2.0", alias="schemaVersion"
    )
    task_id: TaskId = Field(alias="taskId")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    finding_hashes: tuple[str, ...] = Field(alias="findingHashes")
    manual_review_required: Literal[True] = Field(default=True, alias="manualReviewRequired")
    limitations: tuple[str, ...]


class RetrieveEvidenceRequest(_StrictRequest):
    schema_version: Literal["socialgraph-fm.core-skill.retrieve-evidence.request/2.0"] = Field(
        alias="schemaVersion"
    )
    query: str = Field(min_length=1, max_length=2000)
    structural_reference_hash: str | None = Field(
        default=None, alias="structuralReferenceHash", pattern=_HASH_PATTERN
    )
    limit: int = Field(default=10, ge=1, le=100)


class RetrieveEvidenceResponse(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-skill.retrieve-evidence.response/2.0"] = Field(
        default="socialgraph-fm.core-skill.retrieve-evidence.response/2.0", alias="schemaVersion"
    )
    knowledge_results: tuple[KnowledgeSearchResult, ...] = Field(alias="knowledgeResults")
    structural_results: tuple[StructuralSearchResult, ...] = Field(alias="structuralResults")
    limitations: tuple[str, ...]


class GenerateReportRequest(_StrictRequest):
    schema_version: Literal["socialgraph-fm.core-skill.generate-report.request/2.0"] = Field(
        alias="schemaVersion"
    )
    finding_hashes: tuple[str, ...] = Field(alias="findingHashes", min_length=1)
    format: Literal["markdown", "json"] = "markdown"

    _canonical_findings = field_validator("finding_hashes", mode="before")(_canonical_string_array)


class GenerateReportResponse(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-skill.generate-report.response/2.0"] = Field(
        default="socialgraph-fm.core-skill.generate-report.response/2.0", alias="schemaVersion"
    )
    format: Literal["markdown", "json"]
    content: str
    cited_finding_hashes: tuple[str, ...] = Field(alias="citedFindingHashes")
    cited_evidence_hashes: tuple[str, ...] = Field(alias="citedEvidenceHashes")
    generated_without_llm: Literal[True] = Field(default=True, alias="generatedWithoutLlm")
    limitations: tuple[str, ...]


_SKILL_NAMES = (
    "generate_report",
    "inspect_graph",
    "retrieve_evidence",
    "run_core_task",
)


class CoreSkillRegistry:
    """A closed tool registry with no path for caller-supplied facts or scores."""

    def __init__(
        self,
        *,
        graphs: Sequence[CoreGraphBundle],
        findings: Sequence[GovernanceFinding],
        structural_index: StructuralIndex,
        knowledge_store: KnowledgeStore,
    ) -> None:
        self._graphs: dict[str, CoreGraphBundle] = {}
        for graph in graphs:
            validated = CoreGraphBundle.model_validate(
                graph.model_dump(mode="python", by_alias=True)
            )
            if validated.graph_version_hash in self._graphs:
                raise ValueError("duplicate registered graph version")
            self._graphs[validated.graph_version_hash] = validated
        self._findings: dict[str, GovernanceFinding] = {}
        for finding in findings:
            validated_finding = GovernanceFinding.model_validate(
                finding.model_dump(mode="python", by_alias=True)
            )
            if validated_finding.graph_version_hash not in self._graphs:
                raise ValueError("finding references an unregistered graph")
            for similar_case in validated_finding.similar_cases:
                validate_similar_case_provenance(similar_case, structural_index)
            if validated_finding.finding_hash in self._findings:
                raise ValueError("duplicate registered finding hash")
            self._findings[validated_finding.finding_hash] = validated_finding
        self._structural_index = structural_index
        self._knowledge_store = knowledge_store

    @property
    def skill_names(self) -> tuple[str, ...]:
        return _SKILL_NAMES

    def execute(
        self, skill_name: str, payload: Mapping[str, Any]
    ) -> (
        InspectGraphResponse
        | RunCoreTaskResponse
        | RetrieveEvidenceResponse
        | GenerateReportResponse
    ):
        if skill_name == "inspect_graph":
            inspect_request = InspectGraphRequest.model_validate(payload)
            return self._inspect_graph(inspect_request)
        if skill_name == "run_core_task":
            task_request = RunCoreTaskRequest.model_validate(payload)
            return self._run_core_task(task_request)
        if skill_name == "retrieve_evidence":
            retrieval_request = RetrieveEvidenceRequest.model_validate(payload)
            return self._retrieve_evidence(retrieval_request)
        if skill_name == "generate_report":
            report_request = GenerateReportRequest.model_validate(payload)
            return self._generate_report(report_request)
        raise ValueError("unsupported skill; only the four core skills are registered")

    def execute_plan(self, plan: str) -> None:
        if not isinstance(plan, str):
            raise TypeError("plan must be text")
        raise ValueError(
            "natural-language plans are unsupported; invoke one strict registered skill request"
        )

    def _graph(self, graph_hash: str) -> CoreGraphBundle:
        try:
            return self._graphs[graph_hash]
        except KeyError as error:
            raise ValueError("unknown registered graph version") from error

    @staticmethod
    def _scope(graph: CoreGraphBundle, requested: tuple[str, ...]) -> tuple[str, ...]:
        all_nodes = {node.id for node in graph.nodes}
        scope = tuple(sorted(requested)) if requested else tuple(sorted(all_nodes))
        if len(set(scope)) != len(scope) or not set(scope) <= all_nodes:
            raise ValueError("scope contains duplicate or unknown graph nodes")
        return scope

    def _inspect_graph(self, request: InspectGraphRequest) -> InspectGraphResponse:
        graph = self._graph(request.graph_version_hash)
        scope = self._scope(graph, request.scope_node_ids)
        scope_set = set(scope)
        edge_count = sum(
            edge.source_id in scope_set and edge.target_id in scope_set for edge in graph.edges
        )
        return InspectGraphResponse(
            graph_version_hash=graph.graph_version_hash,
            directed=graph.directed,
            node_count=len(scope),
            edge_count=edge_count,
            scope_node_ids=scope,
            limitations=(
                "Counts describe registered static graph facts only.",
                NON_CAUSAL_LIMITATION,
            ),
        )

    def _run_core_task(self, request: RunCoreTaskRequest) -> RunCoreTaskResponse:
        graph = self._graph(request.graph_version_hash)
        scope = set(self._scope(graph, request.scope_node_ids))
        hashes = tuple(
            sorted(
                finding.finding_hash
                for finding in self._findings.values()
                if finding.graph_version_hash == graph.graph_version_hash
                and finding.task_id == request.task_id
                and set(finding.subject_ids) <= scope
            )
        )
        return RunCoreTaskResponse(
            task_id=request.task_id,
            graph_version_hash=graph.graph_version_hash,
            finding_hashes=hashes,
            limitations=(MANUAL_REVIEW_LIMITATION, NON_CAUSAL_LIMITATION),
        )

    def _retrieve_evidence(self, request: RetrieveEvidenceRequest) -> RetrieveEvidenceResponse:
        knowledge = self._knowledge_store.search(request.query, limit=request.limit)
        structural = (
            self._structural_index.query_by_record(
                request.structural_reference_hash, limit=request.limit
            )
            if request.structural_reference_hash is not None
            else ()
        )
        return RetrieveEvidenceResponse(
            knowledge_results=knowledge,
            structural_results=structural,
            limitations=(
                "Text knowledge uses SQLite FTS5/BM25 and does not enter the GFM encoder.",
                "Structural similarities are non-causal retrieval scores, not labels or future predictions.",
            ),
        )

    def _generate_report(self, request: GenerateReportRequest) -> GenerateReportResponse:
        if len(set(request.finding_hashes)) != len(request.finding_hashes):
            raise ValueError("findingHashes must be unique")
        try:
            findings = tuple(self._findings[value] for value in request.finding_hashes)
        except KeyError as error:
            raise ValueError("report references an unknown registered finding") from error
        evidence_hashes = tuple(
            sorted({item.evidence_hash for finding in findings for item in finding.evidence})
        )
        if request.format == "json":
            content = canonical_json(
                {
                    "findings": [
                        {
                            "findingHash": finding.finding_hash,
                            "taskId": finding.task_id,
                            "findingType": finding.finding_type,
                            "subjectIds": finding.subject_ids,
                            "scoreHash": finding.score.score_hash,
                            "calibratedConfidence": finding.calibrated_confidence,
                            "evidence": [
                                {
                                    "evidenceHash": item.evidence_hash,
                                    "metric": item.metric,
                                    "limitations": item.limitations,
                                }
                                for item in finding.evidence
                            ],
                            "limitations": finding.limitations,
                        }
                        for finding in findings
                    ],
                    "limitations": [MANUAL_REVIEW_LIMITATION, NON_CAUSAL_LIMITATION],
                    "generatedWithoutLlm": True,
                }
            )
        else:
            lines = ["# Core Governance Evidence Report", "", "Deterministic no-LLM fallback."]
            for finding in findings:
                confidence = finding.calibrated_confidence
                if isinstance(confidence, CalibratedConfidence):
                    confidence_line = f"- Calibrated confidence: {confidence.value}"
                else:
                    confidence_line = (
                        "- Regression interval (not a probability): "
                        f"point estimate {confidence.point_estimate}; "
                        f"[{confidence.lower_bound}, {confidence.upper_bound}]; "
                        f"coverage {confidence.coverage}; method {confidence.method}"
                    )
                lines.extend(
                    [
                        "",
                        f"## {finding.finding_type}",
                        "",
                        f"- Finding hash: `{finding.finding_hash}`",
                        f"- Task: `{finding.task_id}`",
                        "- Subjects: "
                        + ", ".join(
                            _markdown_data_identifier(identifier)
                            for identifier in finding.subject_ids
                        ),
                        f"- Registered score hash: `{finding.score.score_hash}`",
                        confidence_line,
                        "- Evidence hashes: "
                        + ", ".join(f"`{item.evidence_hash}`" for item in finding.evidence),
                        "- Evidence limitations: "
                        + " ".join(
                            limitation
                            for item in finding.evidence
                            for limitation in item.limitations
                        ),
                        "- Limitations: " + " ".join(finding.limitations),
                    ]
                )
            lines.extend(
                [
                    "",
                    "## Required limitations",
                    "",
                    f"- {MANUAL_REVIEW_LIMITATION}",
                    f"- {NON_CAUSAL_LIMITATION}",
                ]
            )
            content = "\n".join(lines) + "\n"
        return GenerateReportResponse(
            format=request.format,
            content=content,
            cited_finding_hashes=request.finding_hashes,
            cited_evidence_hashes=evidence_hashes,
            limitations=(MANUAL_REVIEW_LIMITATION, NON_CAUSAL_LIMITATION),
        )


__all__ = [
    "GenerateReportRequest",
    "GenerateReportResponse",
    "InspectGraphRequest",
    "InspectGraphResponse",
    "RetrieveEvidenceRequest",
    "RetrieveEvidenceResponse",
    "RunCoreTaskRequest",
    "RunCoreTaskResponse",
    "CoreSkillRegistry",
]
