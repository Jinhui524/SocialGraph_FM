"""Strict internal command contracts for SocialGraph-FM Governance product skills."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, cast, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import MAX_NODES
from .materialize import OnlineInferenceData
from .reviewed_cases import CaseKindEntry
from .skill_catalog import load_runtime_skill_catalog

COMMAND_SCHEMA_VERSION = "socialgraph-fm.governance-command/1.0"
RESULT_SCHEMA_VERSION: Literal["socialgraph-fm.governance-result/1.0"] = (
    "socialgraph-fm.governance-result/1.0"
)
IMPLEMENTATION_VERSION: Literal["socialgraph-fm-gfm/governance-skills/1.0"] = (
    "socialgraph-fm-gfm/governance-skills/1.0"
)
_PRODUCT_SKILL_CATALOG = load_runtime_skill_catalog()
if _PRODUCT_SKILL_CATALOG.implementation_version != IMPLEMENTATION_VERSION:
    raise RuntimeError("Governance skill implementation version does not match the catalog")
PUBLIC_SKILLS = _PRODUCT_SKILL_CATALOG.names
_INTERNAL_COMMANDS = ("index_case", "search_knowledge")
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_TIME_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"

InternalCommand = Literal[
    "inspect_graph",
    "run_governance_analysis",
    "get_evidence_subgraph",
    "discover_coordination_groups",
    "rank_coordination_relations",
    "retrieve_similar_cases",
    "get_model_dataset_cards",
    "draft_review_report",
    "index_case",
    "search_knowledge",
]
if tuple(get_args(InternalCommand)) != PUBLIC_SKILLS + _INTERNAL_COMMANDS:
    raise RuntimeError("internal GFM commands do not match the canonical SocialGraph-FM Governance catalog")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class GraphBinding(_StrictModel):
    artifact_id: str = Field(alias="artifactId", pattern=r"^governance-artifact-[0-9a-f]{32}$")
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=_HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)


class ModelBinding(_StrictModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_state_hash: str = Field(alias="modelStateHash", pattern=_HASH_PATTERN)


class CommandRequest(_StrictModel):
    schema_version: Literal["socialgraph-fm.governance-command/1.0"] = Field(alias="schemaVersion")
    command_id: str = Field(alias="commandId", pattern=r"^governance-command-[0-9a-f]{32}$")
    command: InternalCommand
    graph: GraphBinding
    model: ModelBinding
    params: dict[str, Any]


class CommandProvenance(_StrictModel):
    generated_at: str = Field(alias="generatedAt", pattern=_TIME_PATTERN)
    implementation_version: Literal["socialgraph-fm-gfm/governance-skills/1.0"] = Field(
        default=IMPLEMENTATION_VERSION, alias="implementationVersion"
    )
    input_hash: str = Field(alias="inputHash", pattern=_HASH_PATTERN)


class CommandResponse(_StrictModel):
    schema_version: Literal["socialgraph-fm.governance-result/1.0"] = Field(
        default=RESULT_SCHEMA_VERSION, alias="schemaVersion"
    )
    command_id: str = Field(alias="commandId", pattern=r"^governance-command-[0-9a-f]{32}$")
    command: str
    status: Literal["completed", "confirmation_required"]
    graph: GraphBinding
    model: ModelBinding
    result: dict[str, Any]
    provenance: CommandProvenance
    warnings: tuple[str, ...] = ()


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError("value must be a JSON string array")
    return tuple(value)


def _entry_tuple(value: Any) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("kindEntries must be a JSON array")
    return tuple(value)


def _validate_entries(entries: tuple[CaseKindEntry, ...]) -> None:
    order = {"node": 0, "relation": 1, "group": 2}
    if (
        not entries
        or len(entries) > 3
        or tuple(sorted(entries, key=lambda item: order[item.kind])) != entries
        or len({item.kind for item in entries}) != len(entries)
    ):
        raise ValueError("kindEntries must have unique kinds in canonical order")


def _kind_key(
    entries: Sequence[CaseKindEntry],
) -> Literal[
    "node",
    "relation",
    "group",
    "node+relation",
    "node+group",
    "relation+group",
    "node+relation+group",
]:
    return cast(Any, "+".join(item.kind for item in entries))


class InspectGraphParams(_StrictModel):
    scope_node_ids: tuple[str, ...] = Field(default=(), alias="scopeNodeIds")
    run_id: str | None = Field(
        default=None, alias="runId", pattern=r"^governance-[0-9a-f]{32}$"
    )
    candidate_limit: int = Field(default=5, alias="candidateLimit", ge=1, le=5)

    _scope = field_validator("scope_node_ids", mode="before")(_string_tuple)


class RunGovernanceAnalysisParams(_StrictModel):
    protocol: Literal["global"]
    top_k: int = Field(alias="topK", ge=1, le=MAX_NODES)


class EvidenceParams(_StrictModel):
    run_id: str = Field(alias="runId", pattern=r"^governance-[0-9a-f]{32}$")
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)


class PageParams(_StrictModel):
    run_id: str = Field(alias="runId", pattern=r"^governance-[0-9a-f]{32}$")
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1_000)


class RelationParams(PageParams):
    relation_kind: Literal["factual", "potential"] = Field(
        default="factual", alias="relationKind"
    )
    modalities: tuple[Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"], ...] = ()

    _modalities = field_validator("modalities", mode="before")(_string_tuple)

    @model_validator(mode="after")
    def validate_modalities(self) -> RelationParams:
        if len(set(self.modalities)) != len(self.modalities):
            raise ValueError("modalities must be unique")
        if self.relation_kind == "potential" and self.modalities:
            raise ValueError("potential relations do not accept factual modalities")
        return self


class SimilarCaseParams(_StrictModel):
    case_id: str | None = Field(default=None, alias="caseId", pattern=r"^[A-Za-z0-9._:-]{1,200}$")
    run_id: str | None = Field(default=None, alias="runId", pattern=r"^governance-[0-9a-f]{32}$")
    kind_entries: tuple[CaseKindEntry, ...] = Field(default=(), alias="kindEntries")
    limit: int = Field(default=10, ge=1, le=100)

    _entries = field_validator("kind_entries", mode="before")(_entry_tuple)

    @model_validator(mode="after")
    def validate_query(self) -> SimilarCaseParams:
        by_case = self.case_id is not None
        by_run = self.run_id is not None and bool(self.kind_entries)
        if by_case == by_run or (self.case_id is not None and (self.run_id or self.kind_entries)):
            raise ValueError("similar cases require caseId or runId+kindEntries")
        if self.kind_entries:
            _validate_entries(self.kind_entries)
        return self


class EmptyParams(_StrictModel):
    pass


class DraftReportParams(_StrictModel):
    case_id: str = Field(alias="caseId", pattern=r"^[A-Za-z0-9._:-]{1,200}$")
    case_hash: str = Field(alias="caseHash", pattern=_HASH_PATTERN)
    run_id: str = Field(alias="runId", pattern=r"^governance-[0-9a-f]{32}$")
    result_hash: str = Field(alias="resultHash", pattern=_HASH_PATTERN)
    kind_entries: tuple[CaseKindEntry, ...] = Field(alias="kindEntries", min_length=1, max_length=3)
    format: Literal["markdown", "json"]
    review_hash: str | None = Field(default=None, alias="reviewHash", pattern=_HASH_PATTERN)

    _entries = field_validator("kind_entries", mode="before")(_entry_tuple)

    @model_validator(mode="after")
    def validate_targets(self) -> DraftReportParams:
        _validate_entries(self.kind_entries)
        return self


class IndexCaseParams(_StrictModel):
    case_id: str = Field(alias="caseId", pattern=r"^[A-Za-z0-9._:-]{1,200}$")
    case_hash: str = Field(alias="caseHash", pattern=_HASH_PATTERN)
    run_id: str = Field(alias="runId", pattern=r"^governance-[0-9a-f]{32}$")
    result_hash: str = Field(alias="resultHash", pattern=_HASH_PATTERN)
    kind_entries: tuple[CaseKindEntry, ...] = Field(alias="kindEntries", min_length=1, max_length=3)
    concluded_at: str = Field(alias="concludedAt", pattern=_TIME_PATTERN)
    review_hash: str = Field(alias="reviewHash", pattern=_HASH_PATTERN)
    review_status: Literal["concluded", "reviewed"] = Field(alias="reviewStatus")

    _entries = field_validator("kind_entries", mode="before")(_entry_tuple)

    @model_validator(mode="after")
    def validate_targets(self) -> IndexCaseParams:
        _validate_entries(self.kind_entries)
        return self


class KnowledgeSearchParams(_StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    limit: int = Field(default=10, ge=1, le=50)


class SkillRuntimeProtocol(Protocol):
    root: Path
    global_model_root: Path
    _model: Any

    def _artifact(self, artifact_id: str) -> OnlineInferenceData: ...

    def result(self, run_id: str) -> dict[str, Any]: ...

    def evidence(self, run_id: str, node_id: str) -> dict[str, Any]: ...

    def derivations(self, run_id: str, kind: str, *, offset: int, limit: int) -> dict[str, Any]: ...

    def _outputs(self, run_id: str) -> dict[str, np.ndarray]: ...

    def _analytics_arrays(self, run_id: str) -> dict[str, np.ndarray]: ...

    def _analytics_document(self, run_id: str) -> dict[str, Any]: ...

    def _relation_derivation(
        self, data: OnlineInferenceData, arrays: Mapping[str, np.ndarray], index: int
    ) -> dict[str, Any]: ...

    def _finding(
        self,
        data: OnlineInferenceData,
        arrays: Mapping[str, np.ndarray],
        index: int,
        *,
        rank: int | None = None,
        community_sizes: np.ndarray | None = None,
    ) -> dict[str, Any]: ...


def _components(node_count: int, edges: Sequence[tuple[int, int]]) -> int:
    parents = list(range(node_count))

    def find(value: int) -> int:
        while parents[value] != value:
            parents[value] = parents[parents[value]]
            value = parents[value]
        return value

    for source, target in edges:
        left, right = find(source), find(target)
        if left != right:
            parents[right] = left
    return len({find(value) for value in range(node_count)})




__all__ = [
    "COMMAND_SCHEMA_VERSION",
    "IMPLEMENTATION_VERSION",
    "PUBLIC_SKILLS",
    "RESULT_SCHEMA_VERSION",
    "CommandProvenance",
    "CommandRequest",
    "CommandResponse",
    "DraftReportParams",
    "EmptyParams",
    "EvidenceParams",
    "GraphBinding",
    "IndexCaseParams",
    "InspectGraphParams",
    "InternalCommand",
    "KnowledgeSearchParams",
    "ModelBinding",
    "PageParams",
    "RelationParams",
    "RunGovernanceAnalysisParams",
    "SimilarCaseParams",
    "SkillRuntimeProtocol",
]
