"""Fail-closed uploaded-graph compatibility and cross-graph structural retrieval."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256

from .contracts import (
    COLLABORATION_TASK,
    HASH_PATTERN,
    ResearchSimilarityHit,
    ResearchSimilarityRequest,
    ResearchSimilarityResult,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class UploadedGraphDescriptor(_StrictModel):
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    directed: bool
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    simple: bool
    self_loop_count: int = Field(alias="selfLoopCount", ge=0)
    parallel_edge_count: int = Field(alias="parallelEdgeCount", ge=0)


class UploadCompatibility(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-upload-compatibility/1.0"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    compatible_task_ids: tuple[Literal["core.collaboration_completion"], ...] = Field(
        alias="compatibleTaskIds", strict=False
    )
    structural_similarity_eligible: bool = Field(alias="structuralSimilarityEligible")
    blockers: tuple[str, ...] = Field(strict=False)
    compatibility_hash: str = Field(alias="compatibilityHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.compatibility_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"compatibility_hash"})
        ):
            raise ValueError("compatibilityHash mismatch")
        return self


def inspect_uploaded_graph(descriptor: UploadedGraphDescriptor) -> UploadCompatibility:
    blockers: list[str] = []
    if descriptor.directed:
        blockers.append("COLLABORATION_REQUIRES_UNDIRECTED_GRAPH")
    if not descriptor.simple or descriptor.self_loop_count or descriptor.parallel_edge_count:
        blockers.append("COLLABORATION_REQUIRES_SIMPLE_GRAPH")
    if descriptor.node_count < 20:
        blockers.append("COLLABORATION_REQUIRES_AT_LEAST_20_NODES")
    if descriptor.node_count > 50_000:
        blockers.append("GRAPH_EXCEEDS_50000_NODE_LIMIT")
    if descriptor.edge_count > 1_500_000:
        blockers.append("GRAPH_EXCEEDS_1500000_EDGE_LIMIT")
    possible_pairs = descriptor.node_count * max(0, descriptor.node_count - 1) // 2
    if possible_pairs - descriptor.edge_count < 10:
        blockers.append("COLLABORATION_REQUIRES_10_NON_EDGE_CANDIDATES")

    similarity_blocked = descriptor.node_count < 5 or descriptor.edge_count < 4
    if similarity_blocked:
        blockers.append("STRUCTURAL_SIMILARITY_REQUIRES_5_NODES_AND_4_EDGES")
    task_blockers = tuple(item for item in blockers if not item.startswith("STRUCTURAL_"))
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.research-upload-compatibility/1.0",
        "graphVersionId": descriptor.graph_version_id,
        "graphVersionHash": descriptor.graph_version_hash,
        "compatibleTaskIds": () if task_blockers else (COLLABORATION_TASK,),
        "structuralSimilarityEligible": not similarity_blocked,
        "blockers": tuple(blockers),
    }
    payload["compatibilityHash"] = canonical_sha256(payload)
    return UploadCompatibility.model_validate(payload)


class ResearchStructuralRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-structural-record/1.0"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=500)
    vector: tuple[float, ...] = Field(min_length=1, strict=False)
    deterministic_facts: dict[str, float] = Field(alias="deterministicFacts")
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    record_hash: str = Field(alias="recordHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_record(self):
        if not all(math.isfinite(value) for value in self.vector) or not any(self.vector):
            raise ValueError("research structural vector must be finite and non-zero")
        if not all(math.isfinite(value) for value in self.deterministic_facts.values()):
            raise ValueError("deterministic structural facts must be finite")
        if self.record_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        ):
            raise ValueError("recordHash mismatch")
        return self

    @classmethod
    def create(cls, **values: Any) -> ResearchStructuralRecord:
        payload = {"schemaVersion": "socialgraph-fm.research-structural-record/1.0", **values}
        payload["recordHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class ResearchStructuralIndex:
    """Deterministic cosine retrieval across graph versions for one model."""

    def __init__(self, records: tuple[ResearchStructuralRecord, ...]) -> None:
        if not records:
            raise ValueError("research structural index requires records")
        model_ids = {(item.model_version_id, item.model_version_hash) for item in records}
        dimensions = {len(item.vector) for item in records}
        identities = {(item.graph_version_hash, item.node_id) for item in records}
        if len(model_ids) != 1 or len(dimensions) != 1:
            raise ValueError("research structural records require one model and vector width")
        if len(identities) != len(records):
            raise ValueError("research structural node identities must be unique")
        self._records = tuple(
            sorted(records, key=lambda item: (item.graph_version_id, item.node_id, item.record_hash))
        )
        self._by_node = {
            (record.graph_version_hash, record.node_id): record for record in self._records
        }

    def query(self, request: ResearchSimilarityRequest) -> ResearchSimilarityResult:
        reference = self._by_node.get((request.graph_version_hash, request.node_id))
        if reference is None or reference.graph_version_id != request.graph_version_id:
            raise ValueError("similarity source node is not registered for the requested graph")
        if (
            reference.model_version_id != request.model_version_id
            or reference.model_version_hash != request.model_version_hash
        ):
            raise ValueError("similarity request model identity is incompatible")
        source_norm = math.sqrt(sum(value * value for value in reference.vector))
        scored: list[tuple[float, ResearchStructuralRecord]] = []
        for candidate in self._records:
            if candidate.record_hash == reference.record_hash:
                continue
            candidate_norm = math.sqrt(sum(value * value for value in candidate.vector))
            score = sum(
                left * right
                for left, right in zip(reference.vector, candidate.vector, strict=True)
            ) / (source_norm * candidate_norm)
            scored.append((max(-1.0, min(1.0, score)), candidate))
        selected = sorted(
            scored,
            key=lambda item: (-item[0], item[1].graph_version_id, item[1].node_id),
        )[: request.top_k]
        hits = tuple(
            ResearchSimilarityHit(
                graph_version_id=record.graph_version_id,
                graph_version_hash=record.graph_version_hash,
                node_id=record.node_id,
                score=score,
                deterministic_facts=record.deterministic_facts,
                record_hash=record.record_hash,
            )
            for score, record in selected
        )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-similarity-result/1.0",
            "requestHash": request.request_hash,
            "modelVersionId": request.model_version_id,
            "modelVersionHash": request.model_version_hash,
            "hits": [item.model_dump(mode="python", by_alias=True) for item in hits],
        }
        payload["resultHash"] = canonical_sha256(payload)
        return ResearchSimilarityResult.model_validate(payload)


__all__ = [
    "ResearchStructuralIndex",
    "ResearchStructuralRecord",
    "UploadCompatibility",
    "UploadedGraphDescriptor",
    "inspect_uploaded_graph",
]
