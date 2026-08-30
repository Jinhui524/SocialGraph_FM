"""Strict, canonical SocialGraph-FM Core data contract."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class StaticNode(_StrictModel):
    id: str = Field(min_length=1, max_length=500)
    index: int = Field(ge=0)


class StaticEdge(_StrictModel):
    source_id: str = Field(alias="sourceId", min_length=1, max_length=500)
    target_id: str = Field(alias="targetId", min_length=1, max_length=500)
    edge_type: str = Field(alias="edgeType", min_length=1, max_length=200)
    weight: float = 1.0

    @model_validator(mode="after")
    def validate_weight(self):
        if not math.isfinite(self.weight):
            raise ValueError("edge weight must be finite")
        return self


class NumericFeature(_StrictModel):
    kind: Literal["numeric"]
    name: str = Field(min_length=1, max_length=200)
    values: tuple[float, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_values(self):
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("numeric feature values must be finite")
        return self


class CategoricalFeature(_StrictModel):
    kind: Literal["categorical"]
    name: str = Field(min_length=1, max_length=200)
    values: tuple[str | None, ...] = Field(strict=False)


class MultiHotFeature(_StrictModel):
    kind: Literal["multiHot"]
    name: str = Field(min_length=1, max_length=200)
    row_offsets: tuple[int, ...] = Field(alias="rowOffsets", strict=False)
    values: tuple[str, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_sparse_rows(self):
        if not self.row_offsets or self.row_offsets[0] != 0:
            raise ValueError("multi-hot rowOffsets must start at zero")
        if any(left > right for left, right in zip(self.row_offsets, self.row_offsets[1:])):
            raise ValueError("multi-hot rowOffsets must be monotonic")
        if self.row_offsets[-1] != len(self.values):
            raise ValueError("multi-hot final row offset must equal the sparse value count")
        return self


NodeFeature = Annotated[
    NumericFeature | CategoricalFeature | MultiHotFeature,
    Field(discriminator="kind"),
]


StructuralFeatureRow = Annotated[tuple[float, ...], Field(strict=False)]


class StructuralFeatures(_StrictModel):
    names: tuple[str, ...] = Field(strict=False)
    values: tuple[StructuralFeatureRow, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_matrix(self):
        if len(set(self.names)) != len(self.names) or any(not name for name in self.names):
            raise ValueError("structural feature names must be unique and nonempty")
        if any(len(row) != len(self.names) for row in self.values):
            raise ValueError("structural feature row width does not match names")
        if any(not math.isfinite(value) for row in self.values for value in row):
            raise ValueError("structural feature values must be finite")
        return self


class SourceProvenance(_StrictModel):
    source_name: str = Field(alias="sourceName", min_length=1, max_length=500)
    source_uri: str | None = Field(default=None, alias="sourceUri", min_length=1, max_length=4000)
    citation: str | None = Field(default=None, min_length=1, max_length=4000)
    source_sha256: str = Field(alias="sourceSha256", pattern=r"^[0-9a-f]{64}$")


class SplitAssignment(_StrictModel):
    entity_id: str = Field(alias="entityId", min_length=1, max_length=500)
    role: Literal["train", "validation", "test", "unlabeled"]


class SplitManifest(_StrictModel):
    strategy: Literal[
        "official",
        "all-visible-training",
        "graph-disjoint",
        "leave-one-domain-out",
        "spanning-forest-80-10-10",
        "signed-pair-stratified-70-15-15",
        "stratified-node-70-15-15/1.0",
        "official-10-splits/1.0",
        "candidate-grouped-signed-70-15-15/1.0",
        "spanning-forest-80-10-10/1.0",
    ]
    assignments: tuple[SplitAssignment, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_assignments(self):
        identifiers = [assignment.entity_id for assignment in self.assignments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("split assignment entity IDs must be unique")
        return self


class CoreGraphBundle(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="schemaVersion"
    )
    directed: bool
    nodes: tuple[StaticNode, ...] = Field(strict=False)
    edges: tuple[StaticEdge, ...] = Field(strict=False)
    node_features: tuple[NodeFeature, ...] = Field(alias="nodeFeatures", strict=False)
    structural_features: StructuralFeatures | None = Field(
        default=None, alias="structuralFeatures"
    )
    source: SourceProvenance
    split_manifest: SplitManifest = Field(alias="splitManifest")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_graph(self):
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node IDs must be unique")
        expected_nodes = [(identifier, index) for index, identifier in enumerate(sorted(node_ids))]
        if [(node.id, node.index) for node in self.nodes] != expected_nodes:
            raise ValueError("node indices must be gap-free and derived from sorted stable IDs")

        known = set(node_ids)
        edge_keys: set[tuple[str, str, str, float]] = set()
        for edge in self.edges:
            if edge.source_id not in known or edge.target_id not in known:
                raise ValueError("every edge endpoint must reference a declared node ID")
            if edge.source_id == edge.target_id:
                raise ValueError("self-loop direction semantics are not supported")
            if not self.directed and edge.source_id > edge.target_id:
                raise ValueError("undirected edge endpoints must use canonical stable-ID order")
            key = (edge.source_id, edge.target_id, edge.edge_type, edge.weight)
            if key in edge_keys:
                raise ValueError("duplicate semantic edges are forbidden")
            edge_keys.add(key)

        feature_names = [feature.name for feature in self.node_features]
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("node feature names must be unique")
        for feature in self.node_features:
            row_count = (
                len(feature.row_offsets) - 1
                if isinstance(feature, MultiHotFeature)
                else len(feature.values)
            )
            if row_count != len(self.nodes):
                raise ValueError("node feature row count must equal node count")
        if self.structural_features is not None and len(self.structural_features.values) != len(
            self.nodes
        ):
            raise ValueError("structural feature row count must equal node count")

        expected_hash = calculate_graph_version_hash(
            self.model_dump(mode="python", by_alias=True)
        )
        if self.graph_version_hash != expected_hash:
            raise ValueError("graphVersionHash does not match canonical semantic graph payload")
        return self


def _semantic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract and order graph semantics; raw-source provenance is intentionally separate."""

    edges = []
    for edge in payload.get("edges", ()):
        if isinstance(edge, dict):
            normalized_edge = dict(edge)
            normalized_edge.setdefault("weight", 1.0)
            edges.append(normalized_edge)
        else:
            edges.append(edge)
    features = list(payload.get("nodeFeatures", ()))
    split = payload.get("splitManifest")
    semantic: dict[str, Any] = {
        "schemaVersion": payload.get("schemaVersion"),
        "directed": payload.get("directed"),
        "nodes": sorted(list(payload.get("nodes", ())), key=canonical_json),
        "edges": sorted(edges, key=canonical_json),
        "nodeFeatures": sorted(features, key=canonical_json),
        "structuralFeatures": payload.get("structuralFeatures"),
        "splitManifest": split,
    }
    if isinstance(split, dict):
        semantic["splitManifest"] = {
            **split,
            "assignments": sorted(list(split.get("assignments", ())), key=canonical_json),
        }
    return semantic


def calculate_graph_version_hash(payload: dict[str, Any] | CoreGraphBundle) -> str:
    """Hash canonical graph semantics, excluding graphVersionHash and source provenance."""

    if isinstance(payload, CoreGraphBundle):
        raw = payload.model_dump(mode="python", by_alias=True)
    elif isinstance(payload, dict):
        raw = payload
    else:
        raise TypeError("CoreGraphBundle hashing accepts only a mapping or validated bundle")
    return canonical_sha256(_semantic_payload(raw))


def load_core_graph_bundle_json(serialized: str | bytes) -> CoreGraphBundle:
    """Load JSON only; pickle and arbitrary object deserialization are deliberately unsupported."""

    if not isinstance(serialized, (str, bytes)):
        raise TypeError("CoreGraphBundle input must be UTF-8 JSON text or bytes")
    return CoreGraphBundle.model_validate_json(serialized)


__all__ = [
    "CategoricalFeature",
    "MultiHotFeature",
    "NumericFeature",
    "SourceProvenance",
    "SplitManifest",
    "StaticEdge",
    "CoreGraphBundle",
    "StaticNode",
    "StructuralFeatures",
    "calculate_graph_version_hash",
    "load_core_graph_bundle_json",
]
