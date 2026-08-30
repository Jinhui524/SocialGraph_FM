"""Deterministic train-visible structural features and immutable NPZ caches."""

from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .bundle import CoreGraphBundle, calculate_graph_version_hash
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root

if TYPE_CHECKING:
    from .adapters import AdapterSchema, TrainingSelection


STRUCTURE_FEATURE_NAMES = (
    "degree",
    "in-degree",
    "out-degree",
    "pagerank",
    "clustering",
    "triangle-count",
    "ego-density",
    "k-core",
    "reciprocity",
    "component-size-fraction",
    "two-hop-size",
    "mean-neighbor-degree",
    "rwse-1",
    "rwse-2",
    "rwse-4",
    "rwse-8",
)
_ALGORITHM_VERSION: Literal[
    "socialgraph-fm.core-visible-topology-structure/1.0"
] = "socialgraph-fm.core-visible-topology-structure/1.0"
_CACHE_SCHEMA = "socialgraph-fm.core-structure-cache/1.0"
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_NPZ_BYTES = 512 * 1024 * 1024
_MASK_64 = (1 << 64) - 1


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, strict=True
    )


class StructureAlgorithmConfig(_StrictModel):
    schema_version: Literal[
        "socialgraph-fm.core-structure-algorithm/1.0"
    ] = Field(alias="schemaVersion")
    algorithm_version: Literal[
        "socialgraph-fm.core-visible-topology-structure/1.0"
    ] = Field(alias="algorithmVersion")
    pagerank_damping: float = Field(alias="pagerankDamping", gt=0.0, lt=1.0)
    pagerank_tolerance: float = Field(alias="pagerankTolerance", gt=0.0)
    pagerank_max_iterations: int = Field(alias="pagerankMaxIterations", ge=1, le=1000)
    rwse_steps: tuple[int, ...] = Field(alias="rwseSteps", strict=False)
    rwse_walk_count: int = Field(alias="rwseWalkCount", ge=1, le=4096)
    rwse_seed: int = Field(alias="rwseSeed", ge=0, le=_MASK_64)

    @classmethod
    def fixed(cls) -> StructureAlgorithmConfig:
        return cls(
            schemaVersion="socialgraph-fm.core-structure-algorithm/1.0",
            algorithmVersion=_ALGORITHM_VERSION,
            pagerankDamping=0.85,
            pagerankTolerance=1e-12,
            pagerankMaxIterations=200,
            rwseSteps=(1, 2, 4, 8),
            rwseWalkCount=32,
            rwseSeed=20260815,
        )

    @model_validator(mode="after")
    def validate_fixed_view(self):
        if self.rwse_steps != (1, 2, 4, 8):
            raise ValueError("core RWSE steps must be exactly 1, 2, 4, and 8")
        return self

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class StructureCacheManifest(_StrictModel):
    schema_version: Literal[
        "socialgraph-fm.core-structure-cache/1.0"
    ] = Field(alias="schemaVersion")
    artifact_id: str = Field(alias="artifactId", pattern=r"^[0-9a-f]{64}$")
    role: Literal["training", "inference"]
    base_graph_version_hash: str = Field(
        alias="baseGraphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    enriched_graph_version_hash: str = Field(
        alias="enrichedGraphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    source_sha256: str = Field(alias="sourceSha256", pattern=r"^[0-9a-f]{64}$")
    split_manifest_hash: str = Field(
        alias="splitManifestHash", pattern=r"^[0-9a-f]{64}$"
    )
    fit_row_ids_hash: str = Field(alias="fitRowIdsHash", pattern=r"^[0-9a-f]{64}$")
    fit_row_count: int = Field(alias="fitRowCount", ge=1)
    visible_topology_hash: str = Field(
        alias="visibleTopologyHash", pattern=r"^[0-9a-f]{64}$"
    )
    visible_topology_edge_count: int = Field(alias="visibleTopologyEdgeCount", ge=0)
    algorithm_version: Literal[
        "socialgraph-fm.core-visible-topology-structure/1.0"
    ] = Field(alias="algorithmVersion")
    algorithm_hash: str = Field(alias="algorithmHash", pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    feature_names: tuple[str, ...] = Field(alias="featureNames", strict=False)
    train_means: tuple[float, ...] = Field(alias="trainMeans", strict=False)
    train_scales: tuple[float, ...] = Field(alias="trainScales", strict=False)
    transform_hash: str = Field(alias="transformHash", pattern=r"^[0-9a-f]{64}$")
    tensor_digest: str = Field(alias="tensorDigest", pattern=r"^[0-9a-f]{64}$")
    npz_file_name: Literal["structure.npz"] = Field(alias="npzFileName")
    npz_sha256: str = Field(alias="npzSha256", pattern=r"^[0-9a-f]{64}$")
    manifest_hash: str = Field(alias="manifestHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_manifest(self):
        width = len(self.feature_names)
        if self.feature_names != STRUCTURE_FEATURE_NAMES:
            raise ValueError("structure cache feature inventory is not the fixed view")
        if len(self.train_means) != width or len(self.train_scales) != width:
            raise ValueError("structure cache transform width does not match features")
        if not all(math.isfinite(value) for value in (*self.train_means, *self.train_scales)):
            raise ValueError("structure cache transform must be finite")
        if not all(value > 0 for value in self.train_scales):
            raise ValueError("structure cache scales must be positive")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"manifest_hash"})
        )
        if self.manifest_hash != expected:
            raise ValueError("manifestHash does not match canonical structure cache")
        return self


@dataclass(frozen=True)
class StructureCacheArtifact:
    manifest: StructureCacheManifest
    rows: np.ndarray
    manifest_path: Path
    npz_path: Path


@dataclass(frozen=True)
class _Topology:
    weak: tuple[tuple[int, ...], ...]
    incoming: tuple[tuple[int, ...], ...]
    outgoing: tuple[tuple[int, ...], ...]
    directed_pairs: frozenset[tuple[int, int]]


def _validate_visible_indices(
    bundle: CoreGraphBundle, visible_edge_indices: Sequence[int]
) -> tuple[int, ...]:
    selected = tuple(visible_edge_indices)
    if len(selected) != len(set(selected)):
        raise ValueError("visible edge indices must be unique")
    if any(type(index) is not int or index < 0 or index >= len(bundle.edges) for index in selected):
        raise ValueError("visible edge indices must be integer positions in range")
    return selected


def _build_topology(bundle: CoreGraphBundle, selected: tuple[int, ...]) -> _Topology:
    count = len(bundle.nodes)
    weak: list[set[int]] = [set() for _ in range(count)]
    incoming: list[set[int]] = [set() for _ in range(count)]
    outgoing: list[set[int]] = [set() for _ in range(count)]
    node_index = {node.id: node.index for node in bundle.nodes}
    directed_pairs: set[tuple[int, int]] = set()
    for edge_index in selected:
        edge = bundle.edges[edge_index]
        left = node_index[edge.source_id]
        right = node_index[edge.target_id]
        weak[left].add(right)
        weak[right].add(left)
        outgoing[left].add(right)
        incoming[right].add(left)
        directed_pairs.add((left, right))
        if not bundle.directed:
            outgoing[right].add(left)
            incoming[left].add(right)
            directed_pairs.add((right, left))
    return _Topology(
        weak=tuple(tuple(sorted(neighbors)) for neighbors in weak),
        incoming=tuple(tuple(sorted(neighbors)) for neighbors in incoming),
        outgoing=tuple(tuple(sorted(neighbors)) for neighbors in outgoing),
        directed_pairs=frozenset(directed_pairs),
    )


def _pagerank(topology: _Topology, config: StructureAlgorithmConfig) -> np.ndarray:
    count = len(topology.outgoing)
    if count == 0:
        return np.empty((0,), dtype=np.float64)
    rank = np.full(count, 1.0 / count, dtype=np.float64)
    damping = config.pagerank_damping
    for _ in range(config.pagerank_max_iterations):
        dangling = sum(
            rank[index] for index, neighbors in enumerate(topology.outgoing) if not neighbors
        )
        updated = np.full(count, (1.0 - damping + damping * dangling) / count)
        for source, neighbors in enumerate(topology.outgoing):
            if neighbors:
                share = damping * rank[source] / len(neighbors)
                updated[np.asarray(neighbors, dtype=np.int64)] += share
        if float(np.abs(updated - rank).sum()) <= config.pagerank_tolerance:
            rank = updated
            break
        rank = updated
    total = float(rank.sum())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("PageRank failed to produce a finite probability vector")
    return rank / total


def _triangles(topology: _Topology) -> np.ndarray:
    weak_sets = tuple(set(neighbors) for neighbors in topology.weak)
    values = np.zeros(len(topology.weak), dtype=np.float64)
    for left, neighbors in enumerate(topology.weak):
        for right in neighbors:
            if right <= left:
                continue
            for third in weak_sets[left] & weak_sets[right]:
                if third > right:
                    values[left] += 1
                    values[right] += 1
                    values[third] += 1
    return values


def _core_numbers(topology: _Topology) -> np.ndarray:
    import heapq

    degrees = [len(neighbors) for neighbors in topology.weak]
    heap = [(degree, node) for node, degree in enumerate(degrees)]
    heapq.heapify(heap)
    removed = [False] * len(degrees)
    cores = np.zeros(len(degrees), dtype=np.float64)
    current_core = 0
    while heap:
        degree, node = heapq.heappop(heap)
        if removed[node] or degree != degrees[node]:
            continue
        removed[node] = True
        current_core = max(current_core, degree)
        cores[node] = current_core
        for neighbor in topology.weak[node]:
            if not removed[neighbor]:
                degrees[neighbor] -= 1
                heapq.heappush(heap, (degrees[neighbor], neighbor))
    return cores


def _component_fractions(topology: _Topology) -> np.ndarray:
    count = len(topology.weak)
    result = np.zeros(count, dtype=np.float64)
    visited: set[int] = set()
    for start in range(count):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        visited.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in topology.weak[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        fraction = len(component) / count if count else 0.0
        result[np.asarray(component, dtype=np.int64)] = fraction
    return result


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK_64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
    return (value ^ (value >> 31)) & _MASK_64


def _rwse(topology: _Topology, config: StructureAlgorithmConfig) -> np.ndarray:
    count = len(topology.outgoing)
    result = np.zeros((count, len(config.rwse_steps)), dtype=np.float64)
    step_positions = {step: index for index, step in enumerate(config.rwse_steps)}
    maximum = max(config.rwse_steps)
    for start in range(count):
        returns = [0] * len(config.rwse_steps)
        for walk in range(config.rwse_walk_count):
            current = start
            for step in range(1, maximum + 1):
                neighbors = topology.outgoing[current]
                if neighbors:
                    counter = (
                        config.rwse_seed
                        ^ (start * 0xD6E8FEB86659FD93)
                        ^ (walk * 0xA5A3564E27F886A7)
                        ^ (step * 0x9E3779B185EBCA87)
                        ^ (current * 0xC2B2AE3D27D4EB4F)
                    ) & _MASK_64
                    current = neighbors[_splitmix64(counter) % len(neighbors)]
                position = step_positions.get(step)
                if position is not None and current == start:
                    returns[position] += 1
        result[start] = np.asarray(returns, dtype=np.float64) / config.rwse_walk_count
    return result


def compute_structure_rows(
    bundle: CoreGraphBundle,
    *,
    visible_edge_indices: Sequence[int],
    config: StructureAlgorithmConfig,
) -> np.ndarray:
    """Compute the fixed non-text structural view from only authorized edges."""

    if not isinstance(config, StructureAlgorithmConfig):
        raise TypeError("structure config must be a validated StructureAlgorithmConfig")
    selected = _validate_visible_indices(bundle, visible_edge_indices)
    topology = _build_topology(bundle, selected)
    degree = np.asarray([len(neighbors) for neighbors in topology.weak], dtype=np.float64)
    incoming = np.asarray(
        [len(neighbors) for neighbors in topology.incoming], dtype=np.float64
    )
    outgoing = np.asarray(
        [len(neighbors) for neighbors in topology.outgoing], dtype=np.float64
    )
    triangles = _triangles(topology)
    possible_triangles = degree * (degree - 1) / 2
    clustering = np.divide(
        triangles,
        possible_triangles,
        out=np.zeros_like(triangles),
        where=possible_triangles > 0,
    )
    possible_ego = degree * (degree + 1) / 2
    ego_density = np.divide(
        degree + triangles,
        possible_ego,
        out=np.zeros_like(degree),
        where=possible_ego > 0,
    )
    reciprocity = np.zeros(len(bundle.nodes), dtype=np.float64)
    for node, neighbors in enumerate(topology.weak):
        if neighbors:
            reciprocity[node] = sum(
                (node, neighbor) in topology.directed_pairs
                and (neighbor, node) in topology.directed_pairs
                for neighbor in neighbors
            ) / len(neighbors)
    weak_sets = tuple(set(neighbors) for neighbors in topology.weak)
    two_hop = np.zeros(len(bundle.nodes), dtype=np.float64)
    mean_neighbor_degree = np.zeros(len(bundle.nodes), dtype=np.float64)
    for node, neighbors in enumerate(topology.weak):
        if neighbors:
            reached = set().union(*(weak_sets[neighbor] for neighbor in neighbors))
            reached.discard(node)
            reached.difference_update(neighbors)
            two_hop[node] = len(reached)
            mean_neighbor_degree[node] = sum(degree[neighbor] for neighbor in neighbors) / len(
                neighbors
            )
    rwse = _rwse(topology, config)
    columns = (
        degree,
        incoming,
        outgoing,
        _pagerank(topology, config),
        clustering,
        triangles,
        ego_density,
        _core_numbers(topology),
        reciprocity,
        _component_fractions(topology),
        two_hop,
        mean_neighbor_degree,
        *(rwse[:, index] for index in range(rwse.shape[1])),
    )
    rows = np.asarray(np.column_stack(columns), dtype="<f4", order="C")
    if rows.shape != (len(bundle.nodes), len(STRUCTURE_FEATURE_NAMES)):
        raise ValueError("structure kernel emitted an unexpected tensor shape")
    if not np.all(np.isfinite(rows)):
        raise ValueError("structure kernel emitted non-finite values")
    rows.setflags(write=False)
    return rows


def _semantic_edge_payload(bundle: CoreGraphBundle, edge_index: int) -> dict[str, object]:
    edge = bundle.edges[edge_index]
    return {
        "sourceId": edge.source_id,
        "targetId": edge.target_id,
        "edgeType": edge.edge_type,
        "weight": edge.weight,
    }


def _topology_hash(bundle: CoreGraphBundle, indices: tuple[int, ...]) -> str:
    return canonical_sha256(
        sorted(
            (_semantic_edge_payload(bundle, index) for index in indices),
            key=canonical_json,
        )
    )


def _selection(
    bundle: CoreGraphBundle, role: Literal["training", "inference"]
) -> tuple[tuple[str, ...], tuple[int, ...], str, str]:
    if role not in {"training", "inference"}:
        raise ValueError("structure cache role must be training or inference")
    if role == "training":
        from .adapters import derive_training_selection

        selected: TrainingSelection = derive_training_selection(bundle)
        return (
            selected.fit_row_ids,
            selected.visible_edge_indices,
            selected.fit_row_ids_hash,
            selected.visible_topology_hash,
        )
    fit_ids = tuple(node.id for node in bundle.nodes)
    if not fit_ids:
        raise ValueError("structure cache requires at least one node")
    visible = tuple(range(len(bundle.edges)))
    return fit_ids, visible, canonical_sha256(list(fit_ids)), _topology_hash(bundle, visible)


def _algorithm_hash(config: StructureAlgorithmConfig) -> str:
    semantics = {
        "version": config.algorithm_version,
        "weakProjection": [
            "degree",
            "clustering",
            "triangle-count",
            "ego-density",
            "k-core",
            "component-size-fraction",
            "two-hop-size",
            "mean-neighbor-degree",
        ],
        "directed": ["in-degree", "out-degree", "pagerank", "reciprocity", "rwse"],
        "rwseCounter": "splitmix64-counter/1.0",
        "featureNames": STRUCTURE_FEATURE_NAMES,
    }
    return canonical_sha256(semantics)


def _artifact_binding(
    bundle: CoreGraphBundle,
    role: Literal["training", "inference"],
    config: StructureAlgorithmConfig,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[int, ...]]:
    fit_ids, visible, fit_hash, topology_hash = _selection(bundle, role)
    split_hash = canonical_sha256(
        bundle.split_manifest.model_dump(mode="python", by_alias=True)
    )
    binding: dict[str, Any] = {
        "role": role,
        "baseGraphVersionHash": bundle.graph_version_hash,
        "sourceSha256": bundle.source.source_sha256,
        "splitManifestHash": split_hash,
        "fitRowIdsHash": fit_hash,
        "fitRowCount": len(fit_ids),
        "visibleTopologyHash": topology_hash,
        "visibleTopologyEdgeCount": len(visible),
        "algorithmVersion": config.algorithm_version,
        "algorithmHash": _algorithm_hash(config),
        "configHash": config.config_hash,
        "featureNames": STRUCTURE_FEATURE_NAMES,
    }
    return binding, fit_ids, visible


def _tensor_digest(rows: np.ndarray) -> str:
    return canonical_sha256(
        {
            "dtype": "float32-le",
            "shape": list(rows.shape),
            "sha256": hashlib.sha256(rows.tobytes(order="C")).hexdigest(),
        }
    )


def _enriched_payload(bundle: CoreGraphBundle, rows: np.ndarray) -> dict[str, Any]:
    payload = bundle.model_dump(mode="json", by_alias=True)
    payload["structuralFeatures"] = {
        "names": list(STRUCTURE_FEATURE_NAMES),
        "values": rows.astype(float).tolist(),
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return payload


def _array_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, allow_pickle=False)
    return output.getvalue()


def _npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in sorted(arrays.items()):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _array_bytes(array))
    return output.getvalue()


def _normalization(rows: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    selected = rows[fit_indices]
    means_values: list[float] = []
    scales_values: list[float] = []
    for column in range(rows.shape[1]):
        values = tuple(float(value) for value in selected[:, column])
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means_values.append(mean)
        scales_values.append(scale if scale > 0 else 1.0)
    means = np.asarray(means_values, dtype="<f8")
    scales = np.asarray(scales_values, dtype="<f8")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
        raise ValueError("structure normalization is not finite")
    return np.asarray(means, dtype="<f8"), np.asarray(scales, dtype="<f8")


def _cache_location(
    cache_root: Path, binding: dict[str, Any]
) -> tuple[str, Path, Path, Path]:
    artifact_id = canonical_sha256(binding)
    directory = cache_root / artifact_id
    return artifact_id, directory, directory / "manifest.json", directory / "structure.npz"


def _read_npz(
    serialized: bytes,
    *,
    node_count: int,
    fit_count: int,
    visible_count: int,
) -> dict[str, np.ndarray]:
    expected_sizes = {
        "rows.npy": node_count * len(STRUCTURE_FEATURE_NAMES) * 4 + 1024,
        "means.npy": len(STRUCTURE_FEATURE_NAMES) * 8 + 1024,
        "scales.npy": len(STRUCTURE_FEATURE_NAMES) * 8 + 1024,
        "fit_rows.npy": fit_count * 8 + 1024,
        "visible_edges.npy": visible_count * 8 + 1024,
    }
    try:
        with zipfile.ZipFile(io.BytesIO(serialized), "r") as archive:
            members = archive.infolist()
            if (
                len(members) != len(expected_sizes)
                or {member.filename for member in members} != set(expected_sizes)
            ):
                raise ValueError("structure NPZ member inventory is invalid")
            if any(
                member.compress_type != zipfile.ZIP_STORED
                or member.flag_bits & 0x1
                or member.file_size < 1
                or member.file_size > expected_sizes[member.filename]
                for member in members
            ):
                raise ValueError("structure NPZ member encoding or size is invalid")
        with np.load(io.BytesIO(serialized), allow_pickle=False) as archive:
            if set(archive.files) != {"fit_rows", "means", "rows", "scales", "visible_edges"}:
                raise ValueError("structure NPZ member inventory is invalid")
            return {name: np.array(archive[name], copy=True) for name in archive.files}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("structure NPZ cannot be safely loaded") from error


def _expected_arrays(
    bundle: CoreGraphBundle, fit_ids: tuple[str, ...], visible: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    row_by_id = {node.id: node.index for node in bundle.nodes}
    fit_rows = np.asarray([row_by_id[identifier] for identifier in fit_ids], dtype="<i8")
    visible_edges = np.asarray(visible, dtype="<i8")
    return fit_rows, visible_edges


def load_structure_cache(
    bundle: CoreGraphBundle,
    *,
    cache_root: Path,
    role: Literal["training", "inference"],
    config: StructureAlgorithmConfig | None = None,
) -> StructureCacheArtifact:
    """Load and revalidate every cache binding and primitive array."""

    actual_config = config or StructureAlgorithmConfig.fixed()
    binding, fit_ids, visible = _artifact_binding(bundle, role, actual_config)
    root = secure_existing_root(cache_root)
    artifact_id, directory, manifest_path, npz_path = _cache_location(root, binding)
    if not directory.is_dir():
        raise ValueError("structure cache artifact does not exist")
    manifest_bytes = read_confined_snapshot(
        root, f"{artifact_id}/manifest.json", max_bytes=_MAX_MANIFEST_BYTES
    )
    manifest = StructureCacheManifest.model_validate_json(manifest_bytes)
    for key, expected in binding.items():
        field_name = {
            "baseGraphVersionHash": "base_graph_version_hash",
            "sourceSha256": "source_sha256",
            "splitManifestHash": "split_manifest_hash",
            "fitRowIdsHash": "fit_row_ids_hash",
            "fitRowCount": "fit_row_count",
            "visibleTopologyHash": "visible_topology_hash",
            "visibleTopologyEdgeCount": "visible_topology_edge_count",
            "algorithmVersion": "algorithm_version",
            "algorithmHash": "algorithm_hash",
            "configHash": "config_hash",
            "featureNames": "feature_names",
            "role": "role",
        }[key]
        if getattr(manifest, field_name) != expected:
            raise ValueError("structure cache binding does not match bundle/config")
    if manifest.artifact_id != artifact_id:
        raise ValueError("structure cache binding artifact ID mismatch")
    npz_bytes = read_confined_snapshot(
        root, f"{artifact_id}/structure.npz", max_bytes=_MAX_NPZ_BYTES
    )
    if hashlib.sha256(npz_bytes).hexdigest() != manifest.npz_sha256:
        raise ValueError("structure cache NPZ byte hash mismatch")
    arrays = _read_npz(
        npz_bytes,
        node_count=len(bundle.nodes),
        fit_count=len(fit_ids),
        visible_count=len(visible),
    )
    expected_dtypes = {
        "rows": np.dtype("<f4"),
        "means": np.dtype("<f8"),
        "scales": np.dtype("<f8"),
        "fit_rows": np.dtype("<i8"),
        "visible_edges": np.dtype("<i8"),
    }
    if any(arrays[name].dtype != dtype for name, dtype in expected_dtypes.items()):
        raise ValueError("structure NPZ primitive dtype is invalid")
    rows = np.asarray(arrays["rows"], dtype="<f4", order="C")
    if rows.shape != (len(bundle.nodes), len(STRUCTURE_FEATURE_NAMES)):
        raise ValueError("structure NPZ row shape is invalid")
    fit_rows, visible_edges = _expected_arrays(bundle, fit_ids, visible)
    if not np.array_equal(arrays["fit_rows"], fit_rows) or not np.array_equal(
        arrays["visible_edges"], visible_edges
    ):
        raise ValueError("structure cache binding arrays do not match topology")
    if arrays["means"].shape != (len(STRUCTURE_FEATURE_NAMES),) or arrays[
        "scales"
    ].shape != (len(STRUCTURE_FEATURE_NAMES),):
        raise ValueError("structure NPZ transform shape is invalid")
    if not np.array_equal(arrays["means"], np.asarray(manifest.train_means)) or not np.array_equal(
        arrays["scales"], np.asarray(manifest.train_scales)
    ):
        raise ValueError("structure cache transform arrays do not match manifest")
    if _tensor_digest(rows) != manifest.tensor_digest:
        raise ValueError("structure cache tensor digest mismatch")
    transform_hash = canonical_sha256(
        {
            "featureNames": STRUCTURE_FEATURE_NAMES,
            "fitRowIdsHash": manifest.fit_row_ids_hash,
            "means": list(manifest.train_means),
            "scales": list(manifest.train_scales),
        }
    )
    if transform_hash != manifest.transform_hash:
        raise ValueError("structure cache transform hash mismatch")
    enriched_hash = _enriched_payload(bundle, rows)["graphVersionHash"]
    if enriched_hash != manifest.enriched_graph_version_hash:
        raise ValueError("structure cache enriched graph hash mismatch")
    rows.setflags(write=False)
    return StructureCacheArtifact(
        manifest=manifest,
        rows=rows,
        manifest_path=manifest_path,
        npz_path=npz_path,
    )


def build_structure_cache(
    bundle: CoreGraphBundle,
    *,
    cache_root: Path,
    role: Literal["training", "inference"],
    config: StructureAlgorithmConfig | None = None,
) -> StructureCacheArtifact:
    """Compute once, atomically publish, and exact-reload one structure cache."""

    actual_config = config or StructureAlgorithmConfig.fixed()
    root_path = reject_link_components(cache_root)
    root_path.mkdir(parents=True, exist_ok=True)
    root = secure_existing_root(root_path)
    binding, fit_ids, visible = _artifact_binding(bundle, role, actual_config)
    artifact_id, directory, _manifest_path, _npz_path = _cache_location(root, binding)
    lock = root / f".{artifact_id}.publisher.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("structure cache already has an active publisher") from error
    staging = root / f".{artifact_id}.{uuid.uuid4().hex}.staging"
    try:
        os.close(descriptor)
        if directory.exists() or directory.is_symlink():
            return load_structure_cache(
                bundle, cache_root=root, role=role, config=actual_config
            )
        rows = compute_structure_rows(
            bundle, visible_edge_indices=visible, config=actual_config
        )
        fit_rows, visible_edges = _expected_arrays(bundle, fit_ids, visible)
        means, scales = _normalization(rows, fit_rows)
        transform_payload = {
            "featureNames": STRUCTURE_FEATURE_NAMES,
            "fitRowIdsHash": binding["fitRowIdsHash"],
            "means": means.tolist(),
            "scales": scales.tolist(),
        }
        arrays = {
            "rows": rows,
            "means": means,
            "scales": scales,
            "fit_rows": fit_rows,
            "visible_edges": visible_edges,
        }
        npz_bytes = _npz_bytes(arrays)
        enriched_hash = _enriched_payload(bundle, rows)["graphVersionHash"]
        manifest_payload: dict[str, Any] = {
            "schemaVersion": _CACHE_SCHEMA,
            "artifactId": artifact_id,
            **binding,
            "enrichedGraphVersionHash": enriched_hash,
            "trainMeans": means.tolist(),
            "trainScales": scales.tolist(),
            "transformHash": canonical_sha256(transform_payload),
            "tensorDigest": _tensor_digest(rows),
            "npzFileName": "structure.npz",
            "npzSha256": hashlib.sha256(npz_bytes).hexdigest(),
        }
        manifest_payload["manifestHash"] = canonical_sha256(manifest_payload)
        manifest = StructureCacheManifest.model_validate(manifest_payload)
        manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
        staging.mkdir()
        for destination, content in (
            (staging / "structure.npz", npz_bytes),
            (staging / "manifest.json", manifest_bytes),
        ):
            file_descriptor = os.open(
                destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(file_descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(staging, directory)
        return load_structure_cache(
            bundle, cache_root=root, role=role, config=actual_config
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        lock.unlink(missing_ok=True)


def enrich_bundle_with_structure(
    bundle: CoreGraphBundle, cache: StructureCacheArtifact
) -> CoreGraphBundle:
    """Return a derived immutable bundle carrying the cache's fixed raw rows."""

    manifest = cache.manifest
    split_hash = canonical_sha256(
        bundle.split_manifest.model_dump(mode="python", by_alias=True)
    )
    if (
        manifest.base_graph_version_hash != bundle.graph_version_hash
        or manifest.source_sha256 != bundle.source.source_sha256
        or manifest.split_manifest_hash != split_hash
    ):
        raise ValueError("structure cache does not bind the supplied base bundle")
    enriched = CoreGraphBundle.model_validate(_enriched_payload(bundle, cache.rows))
    if enriched.graph_version_hash != manifest.enriched_graph_version_hash:
        raise ValueError("enriched bundle hash does not match structure cache")
    return enriched


def verify_adapter_structure_binding(
    cache: StructureCacheArtifact, schema: AdapterSchema
) -> None:
    """Prove a fitted adapter uses the cache's exact train-only transform."""

    from .adapters import StructureFieldSchema

    structure = [field for field in schema.fields if isinstance(field, StructureFieldSchema)]
    if len(structure) != 1:
        raise ValueError("adapter schema must contain exactly one structure field")
    field = structure[0]
    manifest = cache.manifest
    if (
        manifest.role != "training"
        or schema.source_graph_version_hash != manifest.enriched_graph_version_hash
        or schema.fit_row_ids_hash != manifest.fit_row_ids_hash
        or schema.fit_row_count != manifest.fit_row_count
        or schema.visible_topology_hash != manifest.visible_topology_hash
        or schema.visible_topology_edge_count != manifest.visible_topology_edge_count
        or field.names != manifest.feature_names
        or field.means != manifest.train_means
        or field.scales != manifest.train_scales
        or field.algorithm_version != manifest.algorithm_version
    ):
        raise ValueError("adapter structure transform does not match training cache")


__all__ = [
    "STRUCTURE_FEATURE_NAMES",
    "StructureAlgorithmConfig",
    "StructureCacheArtifact",
    "StructureCacheManifest",
    "build_structure_cache",
    "compute_structure_rows",
    "enrich_bundle_with_structure",
    "load_structure_cache",
    "verify_adapter_structure_binding",
]
