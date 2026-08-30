"""Deterministic, lossless Russia replay bundle partitioning."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from .contracts import INPUT_SCHEMA_VERSION, MODALITIES, GovernanceInputManifest
from .materialize import (
    BundleValidationError,
    _dataset_content_hash,
    _graph_version_hash,
    _load_features,
    _parse_manifest,
    _parse_nodes,
    _parse_relations,
    _read_member,
    _validate_zip,
)

CATALOG_SCHEMA_VERSION = "socialgraph-fm.governance-russia-shard-catalog/1.0"
PARTITION_RECIPE_ID = "russia-fused-components-lpt-tail/1.0"
EXPECTED_NODE_COUNTS = (491, 75, 75, 75)
EXPECTED_FUSED_EDGE_COUNTS = (9_529, 77, 48, 61)
_MEMBERS = ("manifest.json", "nodes.csv", "relations.csv", "features.npz")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RussiaBundleDescriptor(_StrictModel):
    file_name: str = Field(alias="fileName", pattern=r"^[A-Za-z0-9._-]+\.zip$")
    sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    bytes: Annotated[int, Field(ge=1)]
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=100)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=_SHA256_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_SHA256_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1)
    relation_row_count: int = Field(alias="relationRowCount", ge=1)
    fused_undirected_edge_count: int = Field(alias="fusedUndirectedEdgeCount", ge=1)
    modalities: tuple[str, ...]
    relation_edge_counts: dict[str, int] = Field(alias="relationEdgeCounts")
    component_ids: tuple[str, ...] = Field(default=(), alias="componentIds")

    @model_validator(mode="after")
    def validate_modalities(self) -> RussiaBundleDescriptor:
        if set(self.relation_edge_counts) != set(MODALITIES):
            raise ValueError("relationEdgeCounts must include all five Governance modalities")
        if any(value < 0 for value in self.relation_edge_counts.values()):
            raise ValueError("relationEdgeCounts cannot be negative")
        if tuple(name for name in MODALITIES if self.relation_edge_counts[name]) != self.modalities:
            raise ValueError("modalities must exactly describe nonempty relation inventories")
        if sum(self.relation_edge_counts.values()) != self.relation_row_count:
            raise ValueError("relationEdgeCounts do not sum to relationRowCount")
        return self


class RussiaCoverage(_StrictModel):
    node_count: Literal[716] = Field(alias="nodeCount")
    relation_row_count: Literal[10968] = Field(alias="relationRowCount")
    fused_undirected_edge_count: Literal[9715] = Field(alias="fusedUndirectedEdgeCount")
    cross_shard_fused_edge_count: Literal[0] = Field(alias="crossShardFusedEdgeCount")
    node_inventory_hash: str = Field(alias="nodeInventoryHash", pattern=_SHA256_PATTERN)
    relation_inventory_hash: str = Field(alias="relationInventoryHash", pattern=_SHA256_PATTERN)
    feature_inventory_hash: str = Field(alias="featureInventoryHash", pattern=_SHA256_PATTERN)
    coverage_hash: str = Field(alias="coverageHash", pattern=_SHA256_PATTERN)


class RussiaShardCatalog(_StrictModel):
    schema_version: Literal["socialgraph-fm.governance-russia-shard-catalog/1.0"] = Field(
        alias="schemaVersion"
    )
    source_label: Literal["russia-replay.zip"] = Field(alias="sourceLabel")
    source_sha256: str = Field(alias="sourceSha256", pattern=_SHA256_PATTERN)
    partition_recipe_id: Literal["russia-fused-components-lpt-tail/1.0"] = Field(
        alias="partitionRecipeId"
    )
    supported_modalities: tuple[str, ...] = Field(alias="supportedModalities")
    full: RussiaBundleDescriptor
    shards: tuple[RussiaBundleDescriptor, ...]
    coverage: RussiaCoverage
    catalog_hash: str = Field(alias="catalogHash", pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_catalog(self) -> RussiaShardCatalog:
        if self.supported_modalities != MODALITIES:
            raise ValueError("supportedModalities must preserve the fixed five-modality contract")
        if len(self.shards) != 4:
            raise ValueError("the Russia catalog must contain exactly four shards")
        if tuple(item.node_count for item in self.shards) != EXPECTED_NODE_COUNTS:
            raise ValueError("Russia shard node counts do not match the frozen partition")
        if tuple(item.fused_undirected_edge_count for item in self.shards) != (
            EXPECTED_FUSED_EDGE_COUNTS
        ):
            raise ValueError("Russia shard edge counts do not match the frozen partition")
        logical = self.model_dump(mode="json", by_alias=True, exclude={"catalog_hash"})
        if self.catalog_hash != canonical_sha256(logical):
            raise ValueError("catalogHash is invalid")
        return self


class _SourceBundle:
    def __init__(
        self,
        *,
        path: Path,
        manifest: GovernanceInputManifest,
        raw_manifest: bytes,
        node_ids: tuple[str, ...],
        labels: tuple[str, ...],
        features: np.ndarray,
        rows: tuple[tuple[int, int, str, str], ...],
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.raw_manifest = raw_manifest
        self.node_ids = node_ids
        self.labels = labels
        self.features = features
        self.rows = rows


def _numeric_node_id(node_id: str) -> int:
    prefix, separator, suffix = node_id.partition(":")
    if prefix != "russia" or separator != ":" or not suffix.isascii() or not suffix.isdigit():
        raise BundleValidationError("Russia shard source has a noncanonical node id")
    return int(suffix)


def _load_source(path: str | Path) -> _SourceBundle:
    source = Path(path).expanduser().resolve(strict=True)
    _validate_zip(source)
    with zipfile.ZipFile(source) as archive:
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
    for name, raw in raw_files.items():
        descriptor = manifest.files[name]
        if descriptor.bytes != len(raw) or descriptor.sha256 != hashlib.sha256(raw).hexdigest():
            raise BundleValidationError(f"{name} does not match its manifest descriptor")
    node_ids, labels = _parse_nodes(raw_nodes, manifest.nodeCount)
    if tuple(sorted(node_ids, key=_numeric_node_id)) != node_ids:
        raise BundleValidationError("Russia source node rows are not in canonical numeric order")
    features = _load_features(raw_features, node_ids)
    parsed, removed, observed = _parse_relations(
        raw_relations,
        node_ids=node_ids,
        expected_rows=manifest.relationRowCount,
        clean_self_loops=False,
    )
    if removed or observed != manifest.modalities:
        raise BundleValidationError("Russia source relation inventory is inconsistent")
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    reader = csv.DictReader(io.StringIO(raw_relations.decode("utf-8-sig"), newline=""))
    rows = tuple(
        (
            min(index[row["source"]], index[row["target"]]),
            max(index[row["source"]], index[row["target"]]),
            row["modality"],
            row["weight"],
        )
        for row in reader
    )
    if len(rows) != sum(len(values) for values in parsed.values()):
        raise BundleValidationError("Russia source relation rows changed during parsing")
    return _SourceBundle(
        path=source,
        manifest=manifest,
        raw_manifest=raw_manifest,
        node_ids=node_ids,
        labels=labels,
        features=features,
        rows=rows,
    )


def _components(
    node_count: int, rows: Sequence[tuple[int, int, str, str]]
) -> tuple[tuple[int, ...], ...]:
    neighbors = [set[int]() for _ in range(node_count)]
    for source, target, _modality, _weight in rows:
        neighbors[source].add(target)
        neighbors[target].add(source)
    unseen = set(range(node_count))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        members: list[int] = []
        pending = [root]
        while pending:
            current = pending.pop()
            members.append(current)
            following = sorted(neighbors[current] & unseen, reverse=True)
            unseen.difference_update(following)
            pending.extend(following)
        components.append(tuple(sorted(members)))
    components.sort(key=lambda item: (-len(item), item[0]))
    return tuple(components)


def partition_russia_components(
    node_count: int, rows: Sequence[tuple[int, int, str, str]]
) -> tuple[tuple[int, ...], ...]:
    """Return the frozen four-way component partition in deterministic node order."""

    components = _components(node_count, rows)
    if len(components) < 4:
        raise BundleValidationError("Russia graph has fewer than four connected components")
    bins: list[list[int]] = [list(components[0]), [], [], []]
    for component in components[1:]:
        target = min(range(1, 4), key=lambda index: (len(bins[index]), index))
        bins[target].extend(component)
    return tuple(tuple(sorted(values)) for values in bins)


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _fixed_archive(entries: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", allowZip64=False) as archive:
        for name in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, entries[name], compresslevel=6)
    return stream.getvalue()


def _features_bytes(node_ids: Sequence[str], features: np.ndarray) -> bytes:
    return _fixed_archive(
        {
            "node_ids.npy": _npy_bytes(np.asarray(node_ids)),
            "text_features.npy": _npy_bytes(np.asarray(features, dtype=np.float32)),
        }
    )


def _csv_bytes(rows: Sequence[Sequence[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


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


def _bundle_descriptor(
    path: Path,
    *,
    dataset_id: str,
    raw_manifest: bytes,
    raw_files: Mapping[str, bytes],
    node_ids: Sequence[str],
    rows: Sequence[tuple[int, int, str, str]],
    component_ids: Sequence[str] = (),
) -> RussiaBundleDescriptor:
    undirected = tuple(sorted({(row[0], row[1]) for row in rows}))
    relation_counts = {name: sum(row[2] == name for row in rows) for name in MODALITIES}
    file_digests = {name: hashlib.sha256(value).hexdigest() for name, value in raw_files.items()}
    return RussiaBundleDescriptor(
        fileName=path.name,
        sha256=file_sha256(path),
        bytes=path.stat().st_size,
        datasetId=dataset_id,
        datasetContentHash=_dataset_content_hash(
            manifest_hash=hashlib.sha256(raw_manifest).hexdigest(),
            file_digests=file_digests,
            clean=False,
            removed=0,
        ),
        graphVersionHash=_graph_version_hash(node_ids, undirected),
        nodeCount=len(node_ids),
        relationRowCount=len(rows),
        fusedUndirectedEdgeCount=len(undirected),
        modalities=tuple(name for name in MODALITIES if relation_counts[name]),
        relationEdgeCounts=relation_counts,
        componentIds=tuple(component_ids),
    )


def _write_shard(
    destination: Path,
    source: _SourceBundle,
    members: Sequence[int],
    *,
    shard_number: int,
    component_ids: Sequence[str],
) -> RussiaBundleDescriptor:
    old_to_new = {old: new for new, old in enumerate(members)}
    node_ids = tuple(source.node_ids[index] for index in members)
    labels = tuple(source.labels[index] for index in members)
    rows = tuple(
        sorted(
            (
                old_to_new[source_index],
                old_to_new[target_index],
                modality,
                weight,
            )
            for source_index, target_index, modality, weight in source.rows
            if source_index in old_to_new and target_index in old_to_new
        )
    )
    node_bytes = _csv_bytes([("node_id", "display_name"), *zip(node_ids, labels, strict=True)])
    relation_bytes = _csv_bytes(
        [
            ("source", "target", "modality", "weight"),
            *(
                (node_ids[left], node_ids[right], modality, weight)
                for left, right, modality, weight in rows
            ),
        ]
    )
    feature_bytes = _features_bytes(node_ids, source.features[np.asarray(members, dtype=np.int64)])
    raw_files = {
        "nodes.csv": node_bytes,
        "relations.csv": relation_bytes,
        "features.npz": feature_bytes,
    }
    relation_counts = {name: sum(row[2] == name for row in rows) for name in MODALITIES}
    dataset_id = f"socialgraph-fm:russia:component-shard-{shard_number:02d}-of-04"
    manifest = GovernanceInputManifest.model_validate(
        {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "datasetId": dataset_id,
            "displayName": f"Governance Russia component shard {shard_number:02d} of 04",
            "nodeCount": len(node_ids),
            "relationRowCount": len(rows),
            "featureDimension": 768,
            "modalities": [name for name in MODALITIES if relation_counts[name]],
            "files": {
                name: {"sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)}
                for name, value in raw_files.items()
            },
            "license": source.manifest.license,
            "sourceUri": source.manifest.sourceUri,
        }
    )
    raw_manifest = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    _atomic_bytes(
        destination,
        _fixed_archive(
            {
                "manifest.json": raw_manifest,
                "nodes.csv": node_bytes,
                "relations.csv": relation_bytes,
                "features.npz": feature_bytes,
            }
        ),
    )
    return _bundle_descriptor(
        destination,
        dataset_id=dataset_id,
        raw_manifest=raw_manifest,
        raw_files=raw_files,
        node_ids=node_ids,
        rows=rows,
        component_ids=component_ids,
    )


def _component_id(source: _SourceBundle, members: Sequence[int]) -> str:
    return canonical_sha256({"nodeIds": [source.node_ids[index] for index in members]})


def generate_russia_shards(source_bundle: str | Path, output_directory: str | Path) -> Path:
    """Copy the canonical full bundle and emit four lossless component shards plus catalog."""

    source = _load_source(source_bundle)
    destination = Path(output_directory).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    if len(source.node_ids) != 716 or len(source.rows) != 10_968:
        raise BundleValidationError("the source is not the frozen 716-node Russia replay")
    partitions = partition_russia_components(len(source.node_ids), source.rows)
    fused_counts = tuple(
        len({(row[0], row[1]) for row in source.rows if row[0] in set(part)}) for part in partitions
    )
    if tuple(map(len, partitions)) != EXPECTED_NODE_COUNTS or fused_counts != (
        EXPECTED_FUSED_EDGE_COUNTS
    ):
        raise BundleValidationError("Russia graph does not match the frozen component partition")
    components = _components(len(source.node_ids), source.rows)
    component_lookup = {
        member: _component_id(source, component) for component in components for member in component
    }
    destination.mkdir(parents=True, exist_ok=False)
    try:
        full_path = destination / "russia-full.zip"
        _atomic_bytes(full_path, source.path.read_bytes())
        with zipfile.ZipFile(source.path) as archive:
            raw_files = {
                name: archive.read(name) for name in ("nodes.csv", "relations.csv", "features.npz")
            }
        full = _bundle_descriptor(
            full_path,
            dataset_id=source.manifest.datasetId,
            raw_manifest=source.raw_manifest,
            raw_files=raw_files,
            node_ids=source.node_ids,
            rows=source.rows,
        )
        shard_descriptors: list[RussiaBundleDescriptor] = []
        for position, members in enumerate(partitions, start=1):
            suffix = "main" if position == 1 else "tail"
            file_name = f"russia-shard-{position:02d}-of-04-{suffix}-{len(members):03d}n.zip"
            component_ids = tuple(sorted({component_lookup[index] for index in members}))
            shard_descriptors.append(
                _write_shard(
                    destination / file_name,
                    source,
                    members,
                    shard_number=position,
                    component_ids=component_ids,
                )
            )
        relation_inventory = [
            {
                "source": source.node_ids[left],
                "target": source.node_ids[right],
                "modality": modality,
                "weight": weight,
            }
            for left, right, modality, weight in sorted(source.rows)
        ]
        coverage_logical = {
            "nodeCount": 716,
            "relationRowCount": 10_968,
            "fusedUndirectedEdgeCount": 9_715,
            "crossShardFusedEdgeCount": 0,
            "nodeInventoryHash": canonical_sha256({"nodeIds": source.node_ids}),
            "relationInventoryHash": canonical_sha256({"relations": relation_inventory}),
            "featureInventoryHash": hashlib.sha256(
                np.ascontiguousarray(source.features).tobytes(order="C")
            ).hexdigest(),
        }
        coverage = {**coverage_logical, "coverageHash": canonical_sha256(coverage_logical)}
        logical_catalog: dict[str, Any] = {
            "schemaVersion": CATALOG_SCHEMA_VERSION,
            "sourceLabel": "russia-replay.zip",
            "sourceSha256": file_sha256(source.path),
            "partitionRecipeId": PARTITION_RECIPE_ID,
            "supportedModalities": list(MODALITIES),
            "full": full.model_dump(mode="json", by_alias=True),
            "shards": [item.model_dump(mode="json", by_alias=True) for item in shard_descriptors],
            "coverage": coverage,
        }
        payload = {**logical_catalog, "catalogHash": canonical_sha256(logical_catalog)}
        RussiaShardCatalog.model_validate(payload)
        _atomic_bytes(
            destination / "catalog.json",
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
        )
        verify_russia_shard_catalog(source.path, destination / "catalog.json")
        return destination / "catalog.json"
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def verify_russia_shard_catalog(
    source_bundle: str | Path, catalog_path: str | Path
) -> RussiaShardCatalog:
    """Fail closed unless shard inventories are an exact disjoint cover of the source facts."""

    source = _load_source(source_bundle)
    path = Path(catalog_path).expanduser().resolve(strict=True)
    catalog = RussiaShardCatalog.model_validate_json(path.read_bytes())
    if catalog.source_sha256 != file_sha256(source.path):
        raise BundleValidationError("catalog sourceSha256 does not match the supplied source")
    root = path.parent
    seen_nodes: set[str] = set()
    seen_relations: set[tuple[str, str, str, str]] = set()
    fused_edges: set[tuple[str, str]] = set()
    feature_rows: dict[str, bytes] = {}
    for descriptor in catalog.shards:
        bundle_path = (root / descriptor.file_name).resolve(strict=True)
        if not bundle_path.is_relative_to(root) or bundle_path.is_symlink():
            raise BundleValidationError("catalog shard path is unsafe")
        if (
            file_sha256(bundle_path) != descriptor.sha256
            or bundle_path.stat().st_size != descriptor.bytes
        ):
            raise BundleValidationError("catalog shard bytes do not match their descriptor")
        shard = _load_source(bundle_path)
        if set(shard.node_ids) & seen_nodes:
            raise BundleValidationError("Russia shard node inventories overlap")
        seen_nodes.update(shard.node_ids)
        for index, node_id in enumerate(shard.node_ids):
            feature_rows[node_id] = np.ascontiguousarray(shard.features[index]).tobytes()
        for left, right, modality, weight in shard.rows:
            relation = (shard.node_ids[left], shard.node_ids[right], modality, weight)
            if relation in seen_relations:
                raise BundleValidationError("Russia shard relation inventories overlap")
            seen_relations.add(relation)
            fused_edges.add((relation[0], relation[1]))
    source_relations = {
        (source.node_ids[left], source.node_ids[right], modality, weight)
        for left, right, modality, weight in source.rows
    }
    if seen_nodes != set(source.node_ids) or seen_relations != source_relations:
        raise BundleValidationError("Russia shards are not a lossless source cover")
    if len(fused_edges) != 9_715:
        raise BundleValidationError("Russia shard fused-edge union is incomplete")
    for index, node_id in enumerate(source.node_ids):
        if feature_rows.get(node_id) != np.ascontiguousarray(source.features[index]).tobytes():
            raise BundleValidationError("Russia shard feature rows differ from the source")
    return catalog


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "EXPECTED_FUSED_EDGE_COUNTS",
    "EXPECTED_NODE_COUNTS",
    "PARTITION_RECIPE_ID",
    "RussiaShardCatalog",
    "generate_russia_shards",
    "partition_russia_components",
    "verify_russia_shard_catalog",
]
