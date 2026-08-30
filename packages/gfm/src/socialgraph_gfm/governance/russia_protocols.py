"""Deterministic display-only Russia projections from frozen SocialGraph-FM Global results."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_bytes, canonical_sha256, file_sha256
from socialgraph_gfm.gfm.corpus.common import resolve_within
from socialgraph_gfm.global_model.contracts import TRACE_NAMES
from socialgraph_gfm.global_model.corpus import GlobalCountryCorpus, load_corpus_index

PROJECTION_SCHEMA_VERSION = "socialgraph-fm.governance-russia-protocol-focus/1.0"
MANIFEST_SCHEMA_VERSION = "socialgraph-fm.governance-russia-protocol-focus-manifest/1.0"
SELECTION_RECIPE_ID = "frozen-score-top40-plus-first3-fused-neighbors/1.0"
Protocol: TypeAlias = Literal["in_domain", "low_label", "cross_domain", "global"]  # noqa: UP040
DirectoryProtocol: TypeAlias = Literal["in_domain", "low_label", "cross_domain", "global"]  # noqa: UP040
ModalityName: TypeAlias = Literal[  # noqa: UP040
    "coRT", "coURL", "hashSeq", "fastRT", "tweetSim"
]
PROTOCOL_DIRECTORIES: tuple[tuple[DirectoryProtocol, Protocol], ...] = (
    ("in_domain", "in_domain"),
    ("low_label", "low_label"),
    ("cross_domain", "cross_domain"),
    ("global", "global"),
)
EXPECTED_FOCUS_COUNTS: Mapping[Protocol, tuple[int, int]] = {
    "in_domain": (48, 826),
    "low_label": (53, 530),
    "cross_domain": (51, 183),
    "global": (57, 629),
}
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_RESULT_ARRAYS = {
    "node_ids",
    "scores",
    "logits",
    "structure_missing",
    "router_indices",
    "router_weights",
    "modality_counts",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
        allow_inf_nan=False,
    )


class ProtocolFocusProvenance(_StrictModel):
    registry_hash: str = Field(alias="registryHash", pattern=_HASH_PATTERN)
    registry_file_sha256: str = Field(alias="registryFileSha256", pattern=_HASH_PATTERN)
    corpus_hash: str = Field(alias="corpusHash", pattern=_HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=_HASH_PATTERN)
    result_json_sha256: str = Field(alias="resultJsonSha256", pattern=_HASH_PATTERN)
    result_npz_sha256: str = Field(alias="resultNpzSha256", pattern=_HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    model_state_hash: str = Field(alias="modelStateHash", pattern=_HASH_PATTERN)
    evaluation_split_hash: str = Field(alias="evaluationSplitHash", pattern=_HASH_PATTERN)
    training_hash: str = Field(alias="trainingHash", pattern=_HASH_PATTERN)
    labelled_train_nodes: int = Field(alias="labelledTrainNodes", ge=0)
    evaluation_hash: str = Field(alias="evaluationHash", pattern=_HASH_PATTERN)


class ProtocolFocusModality(_StrictModel):
    modality: ModalityName
    weight: Annotated[float, Field(ge=0)]


class ProtocolFocusNode(_StrictModel):
    id: str = Field(pattern=r"^russia:(?:0|[1-9][0-9]{0,2})$")
    source_node_index: int = Field(alias="sourceNodeIndex", ge=0, le=715)
    label: str = Field(min_length=1, max_length=100)
    score: Annotated[float, Field(ge=0, le=1)]
    score_rank: int = Field(alias="scoreRank", ge=1, le=716)
    above_threshold: bool = Field(alias="aboveThreshold")
    is_anchor: bool = Field(alias="isAnchor")
    anchor_rank: int | None = Field(alias="anchorRank", default=None, ge=1, le=40)
    selected_by_anchors: tuple[str, ...] = Field(alias="selectedByAnchors")
    fused_degree: int = Field(alias="fusedDegree", ge=0)
    structure_missing: bool = Field(alias="structureMissing")
    modality_counts: dict[str, int] = Field(alias="modalityCounts")

    @model_validator(mode="after")
    def validate_node(self) -> ProtocolFocusNode:
        if self.id != f"russia:{self.source_node_index}":
            raise ValueError("node id does not match sourceNodeIndex")
        if self.is_anchor != (self.anchor_rank is not None):
            raise ValueError("anchorRank must be present exactly for anchor nodes")
        if not self.is_anchor and not self.selected_by_anchors:
            raise ValueError("non-anchor focus nodes must identify a selecting anchor")
        if set(self.modality_counts) != set(TRACE_NAMES):
            raise ValueError("modalityCounts must use the fixed five-modality inventory")
        if any(value < 0 for value in self.modality_counts.values()):
            raise ValueError("modalityCounts cannot be negative")
        return self


class ProtocolFocusEdge(_StrictModel):
    id: str = Field(pattern=r"^russia:edge:[0-9]+:[0-9]+$")
    source: str = Field(pattern=r"^russia:(?:0|[1-9][0-9]{0,2})$")
    target: str = Field(pattern=r"^russia:(?:0|[1-9][0-9]{0,2})$")
    modalities: tuple[ProtocolFocusModality, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_edge(self) -> ProtocolFocusEdge:
        left = int(self.source.partition(":")[2])
        right = int(self.target.partition(":")[2])
        if left >= right or self.id != f"russia:edge:{left}:{right}":
            raise ValueError("focus edges must use canonical undirected endpoint order")
        observed = tuple(item.modality for item in self.modalities)
        expected = tuple(name for name in TRACE_NAMES if name in observed)
        if observed != expected or len(set(observed)) != len(observed):
            raise ValueError("edge modalities must be unique and use canonical trace order")
        return self


class RussiaProtocolFocusProjection(_StrictModel):
    schema_version: Literal["socialgraph-fm.governance-russia-protocol-focus/1.0"] = Field(
        alias="schemaVersion"
    )
    directory_protocol: DirectoryProtocol = Field(alias="directoryProtocol")
    protocol: Protocol
    country: Literal["russia"]
    dataset_version_id: Literal["socialgraph-fm:russia"] = Field(alias="datasetVersionId")
    projection_only: Literal[True] = Field(alias="projectionOnly")
    uploadable: Literal[False]
    inference_required: Literal[False] = Field(alias="inferenceRequired")
    metric_scope: Literal["none-projection-is-not-an-evaluation-sample"] = Field(
        alias="metricScope"
    )
    score_semantics: Literal["frozen-global-full-graph-node-score"] = Field(
        alias="scoreSemantics"
    )
    selection_uses_labels: Literal[False] = Field(alias="selectionUsesLabels")
    selection_recipe_id: Literal[
        "frozen-score-top40-plus-first3-fused-neighbors/1.0"
    ] = Field(alias="selectionRecipeId")
    anchor_count: Literal[40] = Field(alias="anchorCount")
    neighbor_limit_per_anchor: Literal[3] = Field(alias="neighborLimitPerAnchor")
    source_graph_node_count: Literal[716] = Field(alias="sourceGraphNodeCount")
    source_graph_edge_count: Literal[9715] = Field(alias="sourceGraphEdgeCount")
    node_count: int = Field(alias="nodeCount", ge=40, le=160)
    edge_count: int = Field(alias="edgeCount", ge=0, le=9715)
    threshold: Annotated[float, Field(ge=0, le=1)]
    provenance: ProtocolFocusProvenance
    nodes: tuple[ProtocolFocusNode, ...]
    edges: tuple[ProtocolFocusEdge, ...]
    projection_hash: str = Field(alias="projectionHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_projection(self) -> RussiaProtocolFocusProjection:
        mapping = dict(PROTOCOL_DIRECTORIES)
        if mapping[self.directory_protocol] != self.protocol:
            raise ValueError("directoryProtocol does not map to protocol")
        if self.node_count != len(self.nodes) or self.edge_count != len(self.edges):
            raise ValueError("focus counts do not match node/edge inventories")
        indices = tuple(item.source_node_index for item in self.nodes)
        if indices != tuple(sorted(set(indices))):
            raise ValueError("focus nodes must be unique and in source index order")
        anchor_ranks = sorted(
            item.anchor_rank for item in self.nodes if item.anchor_rank is not None
        )
        if anchor_ranks != list(range(1, 41)):
            raise ValueError("focus projection must contain exactly the ranked top 40 anchors")
        node_ids = {item.id for item in self.nodes}
        edge_ids = tuple(item.id for item in self.edges)
        if edge_ids != tuple(sorted(set(edge_ids), key=_edge_sort_key)):
            raise ValueError("focus edges must be unique and in canonical endpoint order")
        if any(item.source not in node_ids or item.target not in node_ids for item in self.edges):
            raise ValueError("focus edge endpoint is outside the projection")
        logical = self.model_dump(mode="json", by_alias=True, exclude={"projection_hash"})
        if self.projection_hash != canonical_sha256(logical):
            raise ValueError("projectionHash is invalid")
        return self


class RussiaProtocolFocusManifest(_StrictModel):
    schema_version: Literal[
        "socialgraph-fm.governance-russia-protocol-focus-manifest/1.0"
    ] = Field(alias="schemaVersion")
    directory_protocol: DirectoryProtocol = Field(alias="directoryProtocol")
    protocol: Protocol
    projection_only: Literal[True] = Field(alias="projectionOnly")
    uploadable: Literal[False]
    projection_file: Literal["projection.json"] = Field(alias="projectionFile")
    projection_sha256: str = Field(alias="projectionSha256", pattern=_HASH_PATTERN)
    projection_bytes: int = Field(alias="projectionBytes", ge=1, le=16 * 1024 * 1024)
    projection_hash: str = Field(alias="projectionHash", pattern=_HASH_PATTERN)
    selection_recipe_id: Literal[
        "frozen-score-top40-plus-first3-fused-neighbors/1.0"
    ] = Field(alias="selectionRecipeId")
    node_count: int = Field(alias="nodeCount", ge=40, le=160)
    edge_count: int = Field(alias="edgeCount", ge=0, le=9715)
    provenance: ProtocolFocusProvenance
    manifest_hash: str = Field(alias="manifestHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_manifest(self) -> RussiaProtocolFocusManifest:
        mapping = dict(PROTOCOL_DIRECTORIES)
        if mapping[self.directory_protocol] != self.protocol:
            raise ValueError("directoryProtocol does not map to protocol")
        logical = self.model_dump(mode="json", by_alias=True, exclude={"manifest_hash"})
        if self.manifest_hash != canonical_sha256(logical):
            raise ValueError("manifestHash is invalid")
        return self


def _edge_sort_key(edge_id: str) -> tuple[int, int]:
    _prefix, _edge, left, right = edge_id.split(":")
    return int(left), int(right)


def _read_json_object(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > max_bytes:
        raise ValueError(f"JSON artifact is absent, unsafe, or too large: {path.name}")
    try:
        value = json.loads(
            path.read_bytes(),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"forbidden JSON constant {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise TypeError(f"JSON artifact must contain an object: {path.name}")
    return value


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _required_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _load_registry(root: Path) -> tuple[Path, dict[str, Any]]:
    registry_path = root / "registry" / "socialgraph-global.json"
    registry = _read_json_object(registry_path)
    registry_hash = _required_string(registry.get("registryHash"), "registryHash")
    logical = {key: value for key, value in registry.items() if key != "registryHash"}
    if (
        registry.get("schemaVersion") != "socialgraph-fm.global-model-registry/1.0"
        or registry.get("releaseId") != "socialgraph-fm"
        or registry_hash != canonical_sha256(logical)
    ):
        raise ValueError("SocialGraph-FM Global registry identity is invalid")
    return registry_path, registry


def _load_result_arrays(
    path: Path,
    *,
    expected_sha256: str,
    corpus: GlobalCountryCorpus,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError("frozen result NPZ identity is invalid")
    before = path.stat()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != _RESULT_ARRAYS:
                raise ValueError("frozen result NPZ inventory is invalid")
            node_ids = np.asarray(archive["node_ids"])
            scores = np.asarray(archive["scores"])
            structure_missing = np.asarray(archive["structure_missing"])
            modality_counts = np.asarray(archive["modality_counts"])
    except (OSError, ValueError) as error:
        raise ValueError("frozen result NPZ is not a safe numeric artifact") from error
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("frozen result NPZ changed while being read")
    expected_modality_counts = np.column_stack(
        [np.diff(corpus.relation(name).indptr) for name in TRACE_NAMES]
    )
    if (
        node_ids.dtype != np.dtype(np.int64)
        or scores.dtype != np.dtype(np.float32)
        or structure_missing.dtype != np.dtype(np.bool_)
        or modality_counts.dtype != np.dtype(np.int32)
        or node_ids.shape != (716,)
        or scores.shape != (716,)
        or structure_missing.shape != (716,)
        or modality_counts.shape != (716, 5)
        or not np.array_equal(node_ids, np.arange(716, dtype=np.int64))
        or not bool(np.isfinite(scores).all())
        or bool(((scores < 0) | (scores > 1)).any())
        or not np.array_equal(structure_missing, corpus.structure_missing)
        or not np.array_equal(modality_counts, expected_modality_counts)
    ):
        raise ValueError("frozen result arrays do not match the Russia corpus contract")
    return scores, structure_missing, modality_counts


def _protocol_projection(
    root: Path,
    *,
    registry_path: Path,
    registry: Mapping[str, Any],
    corpus: GlobalCountryCorpus,
    directory_protocol: DirectoryProtocol,
    protocol: Protocol,
) -> RussiaProtocolFocusProjection:
    protocol_artifacts = _required_mapping(registry.get("protocolArtifacts"), "protocolArtifacts")
    artifact = _required_mapping(protocol_artifacts.get(protocol), f"protocolArtifacts.{protocol}")
    result_paths = _required_mapping(artifact.get("resultPaths"), "resultPaths")
    result_descriptor = _required_mapping(result_paths.get("russia"), "resultPaths.russia")
    json_relative = _required_string(result_descriptor.get("jsonPath"), "jsonPath")
    npz_relative = _required_string(result_descriptor.get("npzPath"), "npzPath")
    json_sha256 = _required_string(result_descriptor.get("jsonSha256"), "jsonSha256")
    npz_sha256 = _required_string(result_descriptor.get("npzSha256"), "npzSha256")
    result_json_path = resolve_within(root, json_relative)
    result_npz_path = resolve_within(root, npz_relative)
    if file_sha256(result_json_path) != json_sha256:
        raise ValueError("frozen result JSON identity is invalid")
    result = _read_json_object(result_json_path)
    result_hash = _required_string(result.get("resultHash"), "resultHash")
    if result_hash != canonical_sha256(
        {key: value for key, value in result.items() if key != "resultHash"}
    ):
        raise ValueError("frozen resultHash is invalid")

    graph_version_hash = _required_string(registry.get("graphVersionHash"), "graphVersionHash")
    corpus_hash = _required_string(registry.get("corpusHash"), "corpusHash")
    model_version_id = _required_string(
        artifact.get("protocolModelVersionId"), "protocolModelVersionId"
    )
    model_version_hash = _required_string(
        artifact.get("protocolModelVersionHash"), "protocolModelVersionHash"
    )
    model_state_hash = _required_string(artifact.get("modelStateHash"), "modelStateHash")
    result_split_hash = _required_string(result.get("splitHash"), "result.splitHash")
    expected_result_bindings = {
        "schemaVersion": "socialgraph-fm.global-model-result/1.0",
        "releaseId": "socialgraph-fm",
        "country": "russia",
        "protocol": protocol,
        "nodeCount": 716,
        "graphVersionHash": graph_version_hash,
        "corpusHash": corpus_hash,
        "modelVersionId": model_version_id,
        "modelVersionHash": model_version_hash,
        "modelStateHash": model_state_hash,
        "npzPath": npz_relative,
        "npzSha256": npz_sha256,
    }
    if any(result.get(key) != value for key, value in expected_result_bindings.items()):
        raise ValueError("frozen result does not match its registry model/corpus bindings")
    if result_split_hash != corpus.split("full-fold-0").descriptor.split_hash:
        raise ValueError("frozen result splitHash is not Russia full-fold-0")
    threshold = artifact.get("threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or result.get("threshold") != threshold
    ):
        raise ValueError("frozen result threshold is invalid")
    scores, structure_missing, modality_counts = _load_result_arrays(
        result_npz_path,
        expected_sha256=npz_sha256,
        corpus=corpus,
    )

    ranked = sorted(range(716), key=lambda index: (-float(scores[index]), index))
    anchors = tuple(ranked[:40])
    anchor_rank = {node: position for position, node in enumerate(anchors, start=1)}
    selected = set(anchors)
    selected_by: dict[int, set[int]] = {}
    fused = corpus.fused_csr
    for anchor in anchors:
        start = int(fused.indptr[anchor])
        stop = int(fused.indptr[anchor + 1])
        for raw_neighbor in fused.indices[start:stop][:3]:
            neighbor = int(raw_neighbor)
            selected.add(neighbor)
            selected_by.setdefault(neighbor, set()).add(anchor)
    selected_ids = tuple(sorted(selected))
    selected_set = set(selected_ids)
    score_rank = {node: position for position, node in enumerate(ranked, start=1)}
    degrees = np.diff(fused.indptr)

    nodes = tuple(
        ProtocolFocusNode(
            id=f"russia:{node}",
            sourceNodeIndex=node,
            label=f"Account {node}",
            score=float(scores[node]),
            scoreRank=score_rank[node],
            aboveThreshold=bool(float(scores[node]) >= float(threshold)),
            isAnchor=node in anchor_rank,
            anchorRank=anchor_rank.get(node),
            selectedByAnchors=tuple(
                f"russia:{anchor}" for anchor in sorted(selected_by.get(node, set()))
            ),
            fusedDegree=int(degrees[node]),
            structureMissing=bool(structure_missing[node]),
            modalityCounts={
                name: int(modality_counts[node, position])
                for position, name in enumerate(TRACE_NAMES)
            },
        )
        for node in selected_ids
    )
    modalities_by_pair: dict[tuple[int, int], list[ProtocolFocusModality]] = {}
    for trace_name in TRACE_NAMES:
        relation = corpus.relation(trace_name)
        for left in selected_ids:
            start = int(relation.indptr[left])
            stop = int(relation.indptr[left + 1])
            for raw_right, raw_weight in zip(
                relation.indices[start:stop], relation.weights[start:stop], strict=True
            ):
                right = int(raw_right)
                if right in selected_set and left < right:
                    modalities_by_pair.setdefault((left, right), []).append(
                        ProtocolFocusModality(
                            modality=cast(ModalityName, trace_name), weight=float(raw_weight)
                        )
                    )
    fused_pairs = {
        (left, int(raw_right))
        for left in selected_ids
        for raw_right in fused.indices[int(fused.indptr[left]) : int(fused.indptr[left + 1])]
        if left < int(raw_right) and int(raw_right) in selected_set
    }
    if set(modalities_by_pair) != fused_pairs:
        raise ValueError("focus relation modalities do not exactly cover the induced fused graph")
    edges = tuple(
        ProtocolFocusEdge(
            id=f"russia:edge:{left}:{right}",
            source=f"russia:{left}",
            target=f"russia:{right}",
            modalities=tuple(modalities_by_pair[(left, right)]),
        )
        for left, right in sorted(fused_pairs)
    )
    expected_counts = EXPECTED_FOCUS_COUNTS[protocol]
    if (len(nodes), len(edges)) != expected_counts:
        raise ValueError("frozen Russia focus projection counts changed")

    provenance = ProtocolFocusProvenance(
        registryHash=_required_string(registry.get("registryHash"), "registryHash"),
        registryFileSha256=file_sha256(registry_path),
        corpusHash=corpus_hash,
        graphVersionHash=graph_version_hash,
        resultHash=result_hash,
        resultJsonSha256=json_sha256,
        resultNpzSha256=npz_sha256,
        modelVersionId=model_version_id,
        modelVersionHash=model_version_hash,
        modelStateHash=model_state_hash,
        evaluationSplitHash=result_split_hash,
        trainingHash=_required_string(artifact.get("trainingHash"), "trainingHash"),
        labelledTrainNodes=_required_nonnegative_int(
            artifact.get("labelledTrainNodes"), "labelledTrainNodes"
        ),
        evaluationHash=_required_string(artifact.get("evaluationHash"), "evaluationHash"),
    )
    logical: dict[str, Any] = {
        "schemaVersion": PROJECTION_SCHEMA_VERSION,
        "directoryProtocol": directory_protocol,
        "protocol": protocol,
        "country": "russia",
        "datasetVersionId": "socialgraph-fm:russia",
        "projectionOnly": True,
        "uploadable": False,
        "inferenceRequired": False,
        "metricScope": "none-projection-is-not-an-evaluation-sample",
        "scoreSemantics": "frozen-global-full-graph-node-score",
        "selectionUsesLabels": False,
        "selectionRecipeId": SELECTION_RECIPE_ID,
        "anchorCount": 40,
        "neighborLimitPerAnchor": 3,
        "sourceGraphNodeCount": 716,
        "sourceGraphEdgeCount": 9715,
        "nodeCount": len(nodes),
        "edgeCount": len(edges),
        "threshold": float(threshold),
        "provenance": provenance.model_dump(mode="json", by_alias=True),
        "nodes": [item.model_dump(mode="json", by_alias=True) for item in nodes],
        "edges": [item.model_dump(mode="json", by_alias=True) for item in edges],
    }
    return RussiaProtocolFocusProjection.model_validate(
        {
            **logical,
            "provenance": provenance,
            "nodes": nodes,
            "edges": edges,
            "projectionHash": canonical_sha256(logical),
        },
        strict=True,
    )


def _load_source(root: str | Path) -> tuple[Path, Path, dict[str, Any], GlobalCountryCorpus]:
    source_root = Path(root).expanduser().resolve(strict=True)
    if not source_root.is_dir():
        raise ValueError("SocialGraph-FM Global root is not a directory")
    registry_path, registry = _load_registry(source_root)
    corpus_index = load_corpus_index(source_root / "corpus", verify_manifests=True)
    corpus = corpus_index.load_country(
        "russia", verify_hashes=True, verify_values=True, mmap_mode="r"
    )
    if (
        corpus_index.manifest.content_hash != registry.get("corpusHash")
        or corpus.manifest.content_hash != registry.get("graphVersionHash")
        or corpus.manifest.node_count != 716
        or corpus.manifest.edge_count != 19_430
        or tuple(corpus.manifest.trace_names) != TRACE_NAMES
    ):
        raise ValueError("Russia corpus does not match the frozen registry")
    return source_root, registry_path, registry, corpus


def _manifest_for(
    projection: RussiaProtocolFocusProjection, projection_bytes: bytes
) -> RussiaProtocolFocusManifest:
    logical: dict[str, Any] = {
        "schemaVersion": MANIFEST_SCHEMA_VERSION,
        "directoryProtocol": projection.directory_protocol,
        "protocol": projection.protocol,
        "projectionOnly": True,
        "uploadable": False,
        "projectionFile": "projection.json",
        "projectionSha256": hashlib.sha256(projection_bytes).hexdigest(),
        "projectionBytes": len(projection_bytes),
        "projectionHash": projection.projection_hash,
        "selectionRecipeId": SELECTION_RECIPE_ID,
        "nodeCount": projection.node_count,
        "edgeCount": projection.edge_count,
        "provenance": projection.provenance.model_dump(mode="json", by_alias=True),
    }
    return RussiaProtocolFocusManifest.model_validate(
        {
            **logical,
            "provenance": projection.provenance,
            "manifestHash": canonical_sha256(logical),
        },
        strict=True,
    )


def generate_russia_protocol_focus(
    global_model_root: str | Path, output_directory: str | Path
) -> Path:
    """Generate four local display projections without emitting upload bundles."""

    root, registry_path, registry, corpus = _load_source(global_model_root)
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    projections = tuple(
        _protocol_projection(
            root,
            registry_path=registry_path,
            registry=registry,
            corpus=corpus,
            directory_protocol=directory_protocol,
            protocol=protocol,
        )
        for directory_protocol, protocol in PROTOCOL_DIRECTORIES
    )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for projection in projections:
            projection_bytes = canonical_bytes(
                projection.model_dump(mode="json", by_alias=True)
            )
            manifest = _manifest_for(projection, projection_bytes)
            protocol_root = destination / projection.directory_protocol
            _atomic_bytes(protocol_root / "projection.json", projection_bytes)
            _atomic_bytes(
                protocol_root / "manifest.json",
                canonical_bytes(manifest.model_dump(mode="json", by_alias=True)),
            )
        verify_russia_protocol_focus(root, destination)
        return destination
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_russia_protocol_focus(
    global_model_root: str | Path, output_directory: str | Path
) -> tuple[RussiaProtocolFocusManifest, ...]:
    """Re-derive and verify every projection and all frozen provenance bindings."""

    root, registry_path, registry, corpus = _load_source(global_model_root)
    destination = Path(output_directory).expanduser().resolve(strict=True)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("protocol focus root is unsafe")
    if {item.name for item in destination.iterdir()} != {
        item[0] for item in PROTOCOL_DIRECTORIES
    }:
        raise ValueError("protocol focus directory inventory is invalid")
    manifests: list[RussiaProtocolFocusManifest] = []
    for directory_protocol, protocol in PROTOCOL_DIRECTORIES:
        protocol_root = destination / directory_protocol
        if protocol_root.is_symlink() or not protocol_root.is_dir():
            raise ValueError("protocol focus directory is unsafe")
        if {item.name for item in protocol_root.iterdir()} != {"projection.json", "manifest.json"}:
            raise ValueError("protocol focus artifact inventory is invalid")
        projection_path = protocol_root / "projection.json"
        manifest_path = protocol_root / "manifest.json"
        if projection_path.is_symlink() or manifest_path.is_symlink():
            raise ValueError("protocol focus artifact is unsafe")
        projection_bytes = projection_path.read_bytes()
        projection = RussiaProtocolFocusProjection.model_validate_json(
            projection_bytes, strict=True
        )
        manifest = RussiaProtocolFocusManifest.model_validate_json(
            manifest_path.read_bytes(), strict=True
        )
        expected_projection = _protocol_projection(
            root,
            registry_path=registry_path,
            registry=registry,
            corpus=corpus,
            directory_protocol=directory_protocol,
            protocol=protocol,
        )
        expected_bytes = canonical_bytes(
            expected_projection.model_dump(mode="json", by_alias=True)
        )
        expected_manifest = _manifest_for(expected_projection, expected_bytes)
        if (
            projection_bytes != expected_bytes
            or projection != expected_projection
            or manifest != expected_manifest
        ):
            raise ValueError("protocol focus artifacts do not match frozen source evidence")
        manifests.append(manifest)
    return tuple(manifests)


__all__ = [
    "EXPECTED_FOCUS_COUNTS",
    "MANIFEST_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "PROTOCOL_DIRECTORIES",
    "SELECTION_RECIPE_ID",
    "RussiaProtocolFocusManifest",
    "RussiaProtocolFocusProjection",
    "generate_russia_protocol_focus",
    "verify_russia_protocol_focus",
]
