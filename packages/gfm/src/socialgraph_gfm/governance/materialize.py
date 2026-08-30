"""Safe, deterministic materialization of Governance online inference bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import struct
import tempfile
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from .contracts import (
    ARTIFACT_ID_PATTERN,
    INPUT_SCHEMA_VERSION,
    MAX_NODES,
    MAX_RELATION_ROWS,
    MODALITIES,
    SCHEMA_VERSION,
    GovernanceInputManifest,
)

_BUNDLE_MEMBERS = ("manifest.json", "nodes.csv", "relations.csv", "features.npz")
_MAX_BUNDLE_BYTES = 256 * 1024 * 1024
_MAX_EXPANDED_BYTES = 1024 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
_NODE_ID_MAX_BYTES = 512
_NODE_ID_MAX_CHARS = 128


class BundleValidationError(ValueError):
    """The submitted archive does not satisfy the governed inference contract."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bounded_json(path: Path, *, maximum: int = 16 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= maximum:
        raise BundleValidationError(f"invalid JSON artifact: {path.name}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleValidationError(f"invalid JSON artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise BundleValidationError(f"JSON artifact must be an object: {path.name}")
    return payload


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_node_id(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or len(value) > _NODE_ID_MAX_CHARS
        or len(value.encode("utf-8")) > _NODE_ID_MAX_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise BundleValidationError("node_id must be a trimmed, non-control UTF-8 string")
    return value


def _safe_label(value: str, node_id: str) -> str:
    label = value or node_id
    if len(label) > 256 or any(ord(character) < 32 or ord(character) == 127 for character in label):
        raise BundleValidationError("display_name is invalid")
    return label


def _validate_zip(path: Path) -> dict[str, zipfile.ZipInfo]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_BUNDLE_BYTES:
        raise BundleValidationError("bundle is absent, linked, or oversized")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) != len(_BUNDLE_MEMBERS):
                raise BundleValidationError("bundle must contain exactly four root files")
            inventory: dict[str, zipfile.ZipInfo] = {}
            expanded = 0
            for info in infos:
                if (
                    info.filename not in _BUNDLE_MEMBERS
                    or info.filename in inventory
                    or info.is_dir()
                    or "/" in info.filename
                    or "\\" in info.filename
                    or info.flag_bits & 0x1
                    or ((info.external_attr >> 16) & 0o170000) == 0o120000
                ):
                    raise BundleValidationError("bundle contains an unsafe or unexpected member")
                if info.file_size < 1:
                    raise BundleValidationError("bundle members must be nonempty")
                expanded += info.file_size
                if info.compress_size == 0 or info.file_size > info.compress_size * _MAX_COMPRESSION_RATIO:
                    raise BundleValidationError("bundle member compression ratio is unsafe")
                inventory[info.filename] = info
            if expanded > _MAX_EXPANDED_BYTES:
                raise BundleValidationError("bundle expands beyond the configured limit")
            if set(inventory) != set(_BUNDLE_MEMBERS):
                raise BundleValidationError("bundle member inventory is incomplete")
            return inventory
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise BundleValidationError("bundle is not a valid ZIP archive") from exc


def _read_member(archive: zipfile.ZipFile, name: str, maximum: int) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > maximum:
        raise BundleValidationError(f"{name} exceeds its safety limit")
    data = archive.read(info)
    if len(data) != info.file_size:
        raise BundleValidationError(f"{name} changed while it was read")
    return data


def _parse_manifest(raw: bytes) -> GovernanceInputManifest:
    try:
        payload = json.loads(raw.decode("utf-8"))
        return GovernanceInputManifest.model_validate(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        raise BundleValidationError("manifest.json does not match the Governance input contract") from exc


def _parse_nodes(raw: bytes, expected_count: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError as exc:
        raise BundleValidationError("nodes.csv must be UTF-8") from exc
    reader = csv.DictReader(stream, strict=True)
    if reader.fieldnames not in (["node_id"], ["node_id", "display_name"]):
        raise BundleValidationError("nodes.csv header must be node_id[,display_name]")
    node_ids: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    try:
        for row in reader:
            if None in row or len(node_ids) >= MAX_NODES:
                raise BundleValidationError("nodes.csv contains malformed or excessive rows")
            node_id = _safe_node_id(row["node_id"])
            if node_id in seen:
                raise BundleValidationError("nodes.csv contains a duplicate node_id")
            seen.add(node_id)
            node_ids.append(node_id)
            labels.append(_safe_label(row.get("display_name", ""), node_id))
    except csv.Error as exc:
        raise BundleValidationError("nodes.csv is malformed") from exc
    if len(node_ids) != expected_count:
        raise BundleValidationError("nodes.csv row count does not match manifest.nodeCount")
    return tuple(node_ids), tuple(labels)


def _load_features(raw: bytes, node_ids: Sequence[str]) -> np.ndarray:
    if len(raw) > 128 * 1024 * 1024:
        raise BundleValidationError("features.npz exceeds its safety limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            expected = {"node_ids.npy", "text_features.npy"}
            if (
                {item.filename for item in infos} != expected
                or len(infos) != 2
                or sum(item.file_size for item in infos) > 64 * 1024 * 1024
                or any(item.is_dir() or item.file_size < 1 for item in infos)
            ):
                raise BundleValidationError("features.npz has an unsafe array inventory")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {"node_ids", "text_features"}:
                raise BundleValidationError("features.npz must contain node_ids and text_features")
            feature_ids = np.asarray(archive["node_ids"])
            text_features = np.asarray(archive["text_features"])
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise BundleValidationError("features.npz is not a safe NumPy archive") from exc
    if feature_ids.dtype.hasobject or feature_ids.dtype.kind not in {"U", "S"}:
        raise BundleValidationError("features.npz node_ids must be a fixed-width string array")
    if feature_ids.ndim != 1 or feature_ids.shape[0] != len(node_ids):
        raise BundleValidationError("features.npz node_ids shape is invalid")
    if feature_ids.dtype.kind == "S":
        try:
            aligned_ids = tuple(value.decode("utf-8") for value in feature_ids.tolist())
        except UnicodeDecodeError as exc:
            raise BundleValidationError("features.npz node_ids are not UTF-8") from exc
    else:
        aligned_ids = tuple(str(value) for value in feature_ids.tolist())
    if aligned_ids != tuple(node_ids):
        raise BundleValidationError("features.npz node_ids do not exactly align with nodes.csv")
    if (
        text_features.dtype != np.dtype(np.float32)
        or text_features.shape != (len(node_ids), 768)
        or not bool(np.isfinite(text_features).all())
    ):
        raise BundleValidationError("text_features must be finite float32 [N,768]")
    return np.ascontiguousarray(text_features)


def _parse_relations(
    raw: bytes,
    *,
    node_ids: Sequence[str],
    expected_rows: int,
    clean_self_loops: bool,
) -> tuple[dict[str, list[tuple[int, int, float]]], int, tuple[str, ...]]:
    try:
        stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError as exc:
        raise BundleValidationError("relations.csv must be UTF-8") from exc
    reader = csv.DictReader(stream, strict=True)
    if reader.fieldnames != ["source", "target", "modality", "weight"]:
        raise BundleValidationError(
            "relations.csv header must be source,target,modality,weight"
        )
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    relations: dict[str, list[tuple[int, int, float]]] = {name: [] for name in MODALITIES}
    seen: set[tuple[str, int, int]] = set()
    observed_modalities: set[str] = set()
    raw_rows = 0
    self_loops = 0
    try:
        for row in reader:
            raw_rows += 1
            if None in row or raw_rows > MAX_RELATION_ROWS:
                raise BundleValidationError("relations.csv contains malformed or excessive rows")
            source_id = row["source"]
            target_id = row["target"]
            if source_id not in index or target_id not in index:
                raise BundleValidationError("relations.csv contains a dangling node reference")
            modality = row["modality"]
            if modality not in relations:
                raise BundleValidationError("relations.csv contains an unknown modality")
            try:
                weight = float(row["weight"])
            except ValueError as exc:
                raise BundleValidationError("relation weights must be finite numbers") from exc
            if not math.isfinite(weight) or weight < 0:
                raise BundleValidationError("relation weights must be finite and nonnegative")
            source, target = index[source_id], index[target_id]
            pair = (modality, min(source, target), max(source, target))
            if pair in seen:
                raise BundleValidationError("relations.csv contains a same-modality duplicate")
            seen.add(pair)
            if source == target:
                self_loops += 1
                if not clean_self_loops:
                    raise BundleValidationError(
                        "self-loops require explicit cleanSelfLoops=true materialization"
                    )
                continue
            observed_modalities.add(modality)
            relations[modality].append((pair[1], pair[2], weight))
    except csv.Error as exc:
        raise BundleValidationError("relations.csv is malformed") from exc
    if raw_rows != expected_rows:
        raise BundleValidationError(
            "relations.csv row count does not match manifest.relationRowCount"
        )
    if raw_rows - self_loops < 1:
        raise BundleValidationError("at least one non-self relation is required")
    for values in relations.values():
        values.sort(key=lambda item: (item[0], item[1]))
    return relations, self_loops, tuple(name for name in MODALITIES if name in observed_modalities)


def _directed_csr(
    rows: Sequence[tuple[int, int, float]], node_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directed = [(source, target, weight) for source, target, weight in rows]
    directed.extend((target, source, weight) for source, target, weight in rows)
    directed.sort(key=lambda item: (item[0], item[1]))
    sources = np.fromiter((item[0] for item in directed), dtype=np.int64, count=len(directed))
    indices = np.fromiter((item[1] for item in directed), dtype=np.int64, count=len(directed))
    weights = np.fromiter((item[2] for item in directed), dtype=np.float64, count=len(directed))
    counts = np.bincount(sources, minlength=node_count)
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    return indptr, indices, weights


def _graph_arrays(
    relations: Mapping[str, Sequence[tuple[int, int, float]]], node_count: int
) -> tuple[
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    undirected = tuple(
        sorted({(source, target) for rows in relations.values() for source, target, _ in rows})
    )
    directed = sorted(
        [(source, target) for source, target in undirected]
        + [(target, source) for source, target in undirected]
    )
    sources = np.fromiter((item[0] for item in directed), dtype=np.int64, count=len(directed))
    indices = np.fromiter((item[1] for item in directed), dtype=np.int64, count=len(directed))
    counts = np.bincount(sources, minlength=node_count)
    indptr = np.empty(node_count + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(counts, out=indptr[1:])
    edge_index = np.vstack((sources, indices)).astype(np.int64, copy=False)
    degrees = np.diff(indptr)
    structure_missing = np.ascontiguousarray(degrees == 0, dtype=np.bool_)
    percentiles = np.percentile(degrees, np.linspace(0, 100, 128))
    degree_bucket = np.searchsorted(percentiles, degrees, side="right") - 1
    degree_bucket = np.clip(degree_bucket, 0, 127).astype(np.uint8)
    return undirected, edge_index, indptr, indices, degree_bucket, structure_missing


def _component_count(node_count: int, edges: Sequence[tuple[int, int]]) -> int:
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
    return len({find(node) for node in range(node_count)})


def _graph_stats(
    *,
    node_count: int,
    fused_indptr: np.ndarray,
    undirected_edges: Sequence[tuple[int, int]],
    relation_counts: Mapping[str, int],
) -> np.ndarray:
    degrees = np.diff(fused_indptr).astype(np.float64, copy=False)
    density = (
        (2 * len(undirected_edges)) / (node_count * (node_count - 1))
        if node_count > 1
        else 0.0
    )
    components = _component_count(node_count, undirected_edges)
    quantiles = np.percentile(degrees, (25, 50, 90))
    if node_count > 1:
        _, counts = np.unique(degrees, return_counts=True)
        probabilities = counts.astype(np.float64) / node_count
        entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(node_count))
    else:
        entropy = 0.0
    total = sum(relation_counts.values())
    proportions = (
        tuple(relation_counts[name] / total for name in MODALITIES)
        if total
        else (0.0,) * len(MODALITIES)
    )
    return np.asarray(
        (
            np.log1p(node_count),
            density,
            components / node_count,
            float(np.mean(degrees == 0)),
            *(float(np.log1p(value)) for value in quantiles),
            entropy,
            *proportions,
        ),
        dtype=np.float32,
    )


def _dataset_content_hash(
    *, manifest_hash: str, file_digests: Mapping[str, str], clean: bool, removed: int
) -> str:
    return canonical_sha256(
        {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "manifestHash": manifest_hash,
            "fileDigests": dict(file_digests),
            "cleanSelfLoops": clean,
            "selfLoopsRemoved": removed,
        }
    )


def _graph_version_hash(
    node_ids: Sequence[str], undirected_edges: Sequence[tuple[int, int]]
) -> str:
    digest = hashlib.sha256(b"socialgraph-fm.governance-graph-v2\0")
    for node_id in node_ids:
        encoded = node_id.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    for source, target in undirected_edges:
        digest.update(struct.pack(">QQ", source, target))
    return digest.hexdigest()


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / f"{name}.npy"
    with path.open("xb") as stream:
        np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "path": path.name,
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }


@dataclass(frozen=True)
class MaterializedArtifact:
    root: Path
    document: Mapping[str, Any]

    @property
    def artifact_id(self) -> str:
        return str(self.document["artifactId"])

    @property
    def dataset_content_hash(self) -> str:
        return str(self.document["datasetContentHash"])

    @property
    def graph_version_hash(self) -> str:
        return str(self.document["graphVersionHash"])


def _load_existing(
    path: Path, *, expected_dataset_hash: str, expected_graph_hash: str
) -> MaterializedArtifact:
    document = _bounded_json(path / "artifact.json")
    logical = {key: value for key, value in document.items() if key != "artifactHash"}
    if (
        document.get("schemaVersion") != SCHEMA_VERSION
        or document.get("artifactHash") != canonical_sha256(logical)
        or document.get("datasetContentHash") != expected_dataset_hash
        or document.get("graphVersionHash") != expected_graph_hash
    ):
        raise BundleValidationError("existing materialized artifact identity is invalid")
    return MaterializedArtifact(root=path, document=document)


def materialize_bundle(
    root: str | Path,
    artifact_id: str,
    *,
    expected_dataset_content_hash: str,
    expected_graph_version_hash: str,
    clean_self_loops: bool,
) -> MaterializedArtifact:
    """Revalidate an incoming archive and atomically freeze its numeric graph artifacts."""

    if ARTIFACT_ID_PATTERN.fullmatch(artifact_id) is None:
        raise BundleValidationError("artifactId is invalid")
    selected = Path(root).expanduser().resolve()
    incoming_root = selected / "incoming"
    artifact_root = selected / "artifacts"
    incoming_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    bundle = (incoming_root / artifact_id / "bundle.zip").resolve()
    if not bundle.is_relative_to(incoming_root.resolve()) or bundle.is_symlink():
        raise BundleValidationError("incoming bundle path escapes the configured root")
    target = artifact_root / artifact_id
    if target.is_dir():
        return _load_existing(
            target,
            expected_dataset_hash=expected_dataset_content_hash,
            expected_graph_hash=expected_graph_version_hash,
        )
    _validate_zip(bundle)
    with zipfile.ZipFile(bundle) as archive:
        raw_manifest = _read_member(archive, "manifest.json", 2 * 1024 * 1024)
        raw_nodes = _read_member(archive, "nodes.csv", 16 * 1024 * 1024)
        raw_relations = _read_member(archive, "relations.csv", 256 * 1024 * 1024)
        raw_features = _read_member(archive, "features.npz", 128 * 1024 * 1024)
    manifest = _parse_manifest(raw_manifest)
    raw_files = {
        "nodes.csv": raw_nodes,
        "relations.csv": raw_relations,
        "features.npz": raw_features,
    }
    file_digests = {name: _hash_bytes(value) for name, value in raw_files.items()}
    for name, value in raw_files.items():
        descriptor = manifest.files[name]
        if descriptor.sha256 != file_digests[name] or descriptor.bytes != len(value):
            raise BundleValidationError(f"{name} does not match its manifest descriptor")
    node_ids, labels = _parse_nodes(raw_nodes, manifest.nodeCount)
    text_features = _load_features(raw_features, node_ids)
    relations, removed, observed_modalities = _parse_relations(
        raw_relations,
        node_ids=node_ids,
        expected_rows=manifest.relationRowCount,
        clean_self_loops=clean_self_loops,
    )
    if observed_modalities != manifest.modalities:
        raise BundleValidationError("manifest.modalities does not match relations.csv")
    dataset_hash = _dataset_content_hash(
        manifest_hash=_hash_bytes(raw_manifest),
        file_digests=file_digests,
        clean=clean_self_loops,
        removed=removed,
    )
    if dataset_hash != expected_dataset_content_hash:
        raise BundleValidationError("datasetContentHash does not match bundle content")
    if artifact_id != f"governance-artifact-{dataset_hash[:32]}":
        raise BundleValidationError("artifactId is not derived from datasetContentHash")
    (
        undirected_edges,
        edge_index,
        fused_indptr,
        fused_indices,
        degree_bucket,
        structure_missing,
    ) = _graph_arrays(relations, len(node_ids))
    graph_hash = _graph_version_hash(node_ids, undirected_edges)
    if graph_hash != expected_graph_version_hash:
        raise BundleValidationError("graphVersionHash does not match materialized topology")
    relation_counts = {name: len(relations[name]) * 2 for name in MODALITIES}
    graph_stats = _graph_stats(
        node_count=len(node_ids),
        fused_indptr=fused_indptr,
        undirected_edges=undirected_edges,
        relation_counts=relation_counts,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{artifact_id}.", suffix=".staging", dir=artifact_root))
    try:
        arrays: dict[str, dict[str, Any]] = {}
        arrays["text_features"] = _save_array(staging, "text_features", text_features)
        arrays["edge_index"] = _save_array(staging, "edge_index", edge_index)
        arrays["fused_indptr"] = _save_array(staging, "fused_indptr", fused_indptr)
        arrays["fused_indices"] = _save_array(staging, "fused_indices", fused_indices)
        arrays["degree_bucket"] = _save_array(staging, "degree_bucket", degree_bucket)
        arrays["structure_missing"] = _save_array(
            staging, "structure_missing", structure_missing
        )
        arrays["graph_stats"] = _save_array(staging, "graph_stats", graph_stats)
        for modality in MODALITIES:
            indptr, indices, weights = _directed_csr(relations[modality], len(node_ids))
            token = modality.lower()
            arrays[f"relation_{token}_indptr"] = _save_array(
                staging, f"relation_{token}_indptr", indptr
            )
            arrays[f"relation_{token}_indices"] = _save_array(
                staging, f"relation_{token}_indices", indices
            )
            arrays[f"relation_{token}_weights"] = _save_array(
                staging, f"relation_{token}_weights", weights
            )
        _atomic_json(
            staging / "nodes.json",
            {
                "schemaVersion": SCHEMA_VERSION,
                "nodes": [
                    {"nodeId": node_id, "label": label}
                    for node_id, label in zip(node_ids, labels, strict=True)
                ],
            },
        )
        document: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "inputSchemaVersion": INPUT_SCHEMA_VERSION,
            "artifactId": artifact_id,
            "datasetId": manifest.datasetId,
            "displayName": manifest.displayName,
            "datasetContentHash": dataset_hash,
            "graphVersionHash": graph_hash,
            "bundleSha256": file_sha256(bundle),
            "manifestHash": _hash_bytes(raw_manifest),
            "fileDigests": file_digests,
            "nodeCount": len(node_ids),
            "rawRelationRowCount": manifest.relationRowCount,
            "relationRowCount": manifest.relationRowCount - removed,
            "selfLoopsRemoved": removed,
            "cleanSelfLoops": clean_self_loops,
            "modalities": list(observed_modalities),
            "fusedUndirectedEdgeCount": len(undirected_edges),
            "relationEdgeCounts": {
                name: len(relations[name]) for name in MODALITIES
            },
            "license": manifest.license,
            "sourceUri": manifest.sourceUri,
            "createdAt": _utc_now(),
            "arrays": arrays,
            "nodesSha256": file_sha256(staging / "nodes.json"),
        }
        document["artifactHash"] = canonical_sha256(document)
        _atomic_json(staging / "artifact.json", document)
        try:
            os.replace(staging, target)
        except FileExistsError:
            return _load_existing(
                target,
                expected_dataset_hash=dataset_hash,
                expected_graph_hash=graph_hash,
            )
        return MaterializedArtifact(root=target, document=document)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class OnlineInferenceData:
    artifact: MaterializedArtifact
    node_ids: tuple[str, ...]
    labels: tuple[str, ...]
    arrays: Mapping[str, np.ndarray]

    @property
    def edge_index(self) -> np.ndarray:
        return self.arrays["edge_index"]

    @property
    def text_features(self) -> np.ndarray:
        return self.arrays["text_features"]

    @property
    def degree_bucket(self) -> np.ndarray:
        return self.arrays["degree_bucket"]

    @property
    def structure_missing(self) -> np.ndarray:
        return self.arrays["structure_missing"]

    @property
    def graph_stats(self) -> np.ndarray:
        return self.arrays["graph_stats"]


def load_materialized_artifact(path: str | Path) -> OnlineInferenceData:
    """Open and independently hash-check a frozen online inference artifact."""

    root = Path(path).resolve(strict=True)
    document = _bounded_json(root / "artifact.json")
    logical = {key: value for key, value in document.items() if key != "artifactHash"}
    if document.get("schemaVersion") != SCHEMA_VERSION or document.get(
        "artifactHash"
    ) != canonical_sha256(logical):
        raise BundleValidationError("materialized artifact hash is invalid")
    descriptor_map = document.get("arrays")
    if not isinstance(descriptor_map, dict):
        raise BundleValidationError("materialized array inventory is invalid")
    arrays: dict[str, np.ndarray] = {}
    for name, raw_descriptor in descriptor_map.items():
        if not isinstance(name, str) or not isinstance(raw_descriptor, dict):
            raise BundleValidationError("materialized array descriptor is invalid")
        relative = raw_descriptor.get("path")
        if not isinstance(relative, str) or Path(relative).name != relative:
            raise BundleValidationError("materialized array path is invalid")
        array_path = (root / relative).resolve(strict=True)
        if not array_path.is_relative_to(root) or array_path.is_symlink():
            raise BundleValidationError("materialized array path escapes artifact root")
        if (
            array_path.stat().st_size != raw_descriptor.get("bytes")
            or file_sha256(array_path) != raw_descriptor.get("sha256")
        ):
            raise BundleValidationError("materialized array hash or length mismatch")
        try:
            array = np.load(array_path, allow_pickle=False, mmap_mode="r")
        except (OSError, ValueError) as exc:
            raise BundleValidationError("materialized array is not safe NPY") from exc
        if (
            array.dtype.hasobject
            or array.dtype.str != raw_descriptor.get("dtype")
            or list(array.shape) != raw_descriptor.get("shape")
        ):
            raise BundleValidationError("materialized array dtype or shape mismatch")
        arrays[name] = array
    nodes_document = _bounded_json(root / "nodes.json")
    if file_sha256(root / "nodes.json") != document.get("nodesSha256"):
        raise BundleValidationError("materialized node inventory hash mismatch")
    raw_nodes = nodes_document.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) != document.get("nodeCount"):
        raise BundleValidationError("materialized node inventory is invalid")
    node_ids: list[str] = []
    labels: list[str] = []
    for item in raw_nodes:
        if not isinstance(item, dict) or set(item) != {"nodeId", "label"}:
            raise BundleValidationError("materialized node entry is invalid")
        node_ids.append(_safe_node_id(str(item["nodeId"])))
        labels.append(_safe_label(str(item["label"]), node_ids[-1]))
    node_count = len(node_ids)
    required_shapes = {
        "text_features": (node_count, 768),
        "edge_index": (2, int(document["fusedUndirectedEdgeCount"]) * 2),
        "fused_indptr": (node_count + 1,),
        "degree_bucket": (node_count,),
        "structure_missing": (node_count,),
        "graph_stats": (13,),
    }
    for name, shape in required_shapes.items():
        if name not in arrays or arrays[name].shape != shape:
            raise BundleValidationError(f"materialized {name} shape is invalid")
    if not bool(np.isfinite(arrays["text_features"]).all()) or not bool(
        np.isfinite(arrays["graph_stats"]).all()
    ):
        raise BundleValidationError("materialized model inputs are not finite")
    artifact = MaterializedArtifact(root=root, document=document)
    return OnlineInferenceData(
        artifact=artifact,
        node_ids=tuple(node_ids),
        labels=tuple(labels),
        arrays=arrays,
    )


__all__ = [
    "BundleValidationError",
    "MaterializedArtifact",
    "OnlineInferenceData",
    "load_materialized_artifact",
    "materialize_bundle",
]
