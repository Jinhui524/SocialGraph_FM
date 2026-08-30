"""Deterministic, presentation-sized Russia answer-pack generation.

The canonical Russia replay is intentionally much denser than a useful
defence-screen graph.  This module derives small induced neighbourhoods from
that replay while retaining the original node order, 768-dimensional feature
rows and every relation row whose endpoints are selected.  The resulting
archives are ordinary ``socialgraph-fm.governance-input/2.0`` bundles: they can
be uploaded through the same Global path as any user-provided graph.

Answer packs are presentation inputs, not replacement datasets.  No labels,
split masks or frozen result arrays are copied into an archive.  A catalog
records the deterministic recipe, source hash and optional offline validation
summary separately.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import os
import tempfile
import zipfile
from collections import Counter
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

ANSWER_PACK_CATALOG_SCHEMA_VERSION = "socialgraph-fm.governance-russia-answer-pack-catalog/1.0"
ANSWER_PACK_RECIPE_ID = "russia-risk-balanced-neighborhood/1.0"
ANSWER_PACK_FILENAMES = ("russia-01.zip", "russia-02.zip", "russia-03.zip", "russia-04.zip")
ANSWER_PACK_NODE_RANGE = (72, 144)
ANSWER_PACK_MAX_FUSED_EDGES = 240
_MEMBERS = ("manifest.json", "nodes.csv", "relations.csv", "features.npz")
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

# The seeds and target sizes are part of the recipe.  They were selected from
# the low-degree perimeter of the replay graph, producing readable connected
# neighbourhoods instead of the dense central core.
ANSWER_PACK_SELECTIONS: tuple[tuple[int, int], ...] = (
    (5, 128),
    (44, 120),
    (170, 96),
    (98, 120),
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class RussiaAnswerPackDescriptor(_StrictModel):
    file_name: str = Field(alias="fileName", pattern=r"^russia-0[1-4]\.zip$")
    sha256: Annotated[str, Field(pattern=_SHA256_PATTERN)]
    bytes: Annotated[int, Field(ge=1)]
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=100)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=_SHA256_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_SHA256_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=ANSWER_PACK_NODE_RANGE[0], le=ANSWER_PACK_NODE_RANGE[1])
    relation_row_count: int = Field(alias="relationRowCount", ge=1)
    fused_undirected_edge_count: int = Field(alias="fusedUndirectedEdgeCount", ge=1, le=ANSWER_PACK_MAX_FUSED_EDGES)
    modalities: tuple[str, ...]
    relation_edge_counts: dict[str, int] = Field(alias="relationEdgeCounts")
    selection: Mapping[str, Any]
    validation: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def validate_inventory(self) -> RussiaAnswerPackDescriptor:
        if set(self.relation_edge_counts) != set(MODALITIES):
            raise ValueError("relationEdgeCounts must include all five Governance modalities")
        if any(value < 0 for value in self.relation_edge_counts.values()):
            raise ValueError("relationEdgeCounts cannot be negative")
        expected_modalities = tuple(name for name in MODALITIES if self.relation_edge_counts[name])
        if expected_modalities != self.modalities:
            raise ValueError("modalities must describe nonempty relation inventories")
        if sum(self.relation_edge_counts.values()) != self.relation_row_count:
            raise ValueError("relationEdgeCounts do not sum to relationRowCount")
        return self


class RussiaAnswerPackCatalog(_StrictModel):
    schema_version: Literal[
        "socialgraph-fm.governance-russia-answer-pack-catalog/1.0"
    ] = Field(alias="schemaVersion")
    source_file: str = Field(alias="sourceFile", min_length=1, max_length=512)
    source_sha256: str = Field(alias="sourceSha256", pattern=_SHA256_PATTERN)
    recipe_id: Literal["russia-risk-balanced-neighborhood/1.0"] = Field(alias="recipeId")
    supported_modalities: tuple[str, ...] = Field(alias="supportedModalities")
    packs: tuple[RussiaAnswerPackDescriptor, ...]
    selection_recipe: Mapping[str, Any] = Field(alias="selectionRecipe")
    catalog_hash: str = Field(alias="catalogHash", pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_catalog(self) -> RussiaAnswerPackCatalog:
        if self.supported_modalities != MODALITIES:
            raise ValueError("supportedModalities must preserve the fixed five-modality contract")
        if tuple(item.file_name for item in self.packs) != ANSWER_PACK_FILENAMES:
            raise ValueError("answer-pack filenames must be russia-01.zip through russia-04.zip")
        if len(self.packs) != len(ANSWER_PACK_SELECTIONS):
            raise ValueError("exactly four answer packs are required")
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
        raise BundleValidationError("Russia source has a noncanonical node id")
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
    raw_files = {"nodes.csv": raw_nodes, "relations.csv": raw_relations, "features.npz": raw_features}
    for name, raw in raw_files.items():
        descriptor = manifest.files[name]
        if descriptor.bytes != len(raw) or descriptor.sha256 != hashlib.sha256(raw).hexdigest():
            raise BundleValidationError(f"{name} does not match its source manifest")
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


def _adjacency(source: _SourceBundle) -> list[set[int]]:
    neighbors = [set[int]() for _ in source.node_ids]
    for left, right, _modality, _weight in source.rows:
        neighbors[left].add(right)
        neighbors[right].add(left)
    return neighbors


def _connected_components(neighbors: Sequence[set[int]]) -> tuple[tuple[int, ...], ...]:
    unseen = set(range(len(neighbors)))
    components: list[tuple[int, ...]] = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        pending = [root]
        members: list[int] = []
        while pending:
            current = pending.pop()
            members.append(current)
            following = sorted(neighbors[current] & unseen, reverse=True)
            unseen.difference_update(following)
            pending.extend(following)
        components.append(tuple(sorted(members)))
    components.sort(key=lambda item: (-len(item), item[0]))
    return tuple(components)


def _expand_neighborhood(
    neighbors: Sequence[set[int]],
    *,
    seed: int,
    target_size: int,
    risk_flags: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Expand a connected set, preferring low induced edge growth.

    A heap stores the number of selected neighbours for every frontier node;
    this avoids the order-dependent ring/rail partitions used by the old
    sample generator and makes the result reproducible across Python runs.
    """

    if not 0 <= seed < len(neighbors):
        raise BundleValidationError("answer-pack seed is outside the source graph")
    selected = {seed}
    frontier_counts: dict[int, int] = {
        value: 1 for value in neighbors[seed] if value != seed
    }
    heap: list[tuple[int, int, int, int]] = [
        (count, len(neighbors[node]), int(risk_flags[node]) if risk_flags is not None else 0, node)
        for node, count in frontier_counts.items()
    ]
    heapq.heapify(heap)
    while len(selected) < target_size:
        candidate: int | None = None
        while heap:
            count, _degree, _risk, node = heapq.heappop(heap)
            if node not in selected and frontier_counts.get(node) == count:
                candidate = node
                break
        if candidate is None:
            raise BundleValidationError(
                f"answer-pack seed {seed} cannot reach target size {target_size}"
            )
        selected.add(candidate)
        for neighbor in neighbors[candidate]:
            if neighbor in selected:
                continue
            count = frontier_counts.get(neighbor, 0) + 1
            frontier_counts[neighbor] = count
            heapq.heappush(
                heap,
                (
                    count,
                    len(neighbors[neighbor]),
                    int(risk_flags[neighbor]) if risk_flags is not None else 0,
                    neighbor,
                ),
            )
    return tuple(sorted(selected))


def select_answer_pack_members(
    source_bundle: str | Path,
    *,
    seed: int,
    target_size: int,
    frozen_scores: Sequence[float] | np.ndarray | None = None,
) -> tuple[int, ...]:
    """Select one deterministic connected answer-pack neighbourhood."""

    if not ANSWER_PACK_NODE_RANGE[0] <= target_size <= ANSWER_PACK_NODE_RANGE[1]:
        raise BundleValidationError("answer-pack node count is outside the presentation range")
    source = _load_source(source_bundle)
    risk_flags: np.ndarray | None = None
    if frozen_scores is not None:
        risk_flags = np.asarray(frozen_scores, dtype=np.float32) >= 0.64
        if risk_flags.shape != (len(source.node_ids),):
            raise BundleValidationError("frozen score inventory does not align to Russia nodes")
    members = _expand_neighborhood(
        _adjacency(source), seed=seed, target_size=target_size, risk_flags=risk_flags
    )
    return members


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _fixed_archive(entries: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", allowZip64=False) as archive:
        for name, value in entries.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value, compresslevel=6)
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
    csv.writer(stream, lineterminator="\n").writerows(rows)
    return stream.getvalue().encode("utf-8")


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
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
) -> RussiaAnswerPackDescriptor:
    undirected = tuple(sorted({(left, right) for left, right, _modality, _weight in rows}))
    relation_counts = {name: sum(row[2] == name for row in rows) for name in MODALITIES}
    file_digests = {name: hashlib.sha256(value).hexdigest() for name, value in raw_files.items()}
    return RussiaAnswerPackDescriptor(
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
        selection={},
    )


def _write_answer_pack(
    destination: Path,
    source: _SourceBundle,
    members: Sequence[int],
    *,
    pack_number: int,
    selection: Mapping[str, Any],
) -> RussiaAnswerPackDescriptor:
    old_to_new = {old: new for new, old in enumerate(members)}
    node_ids = tuple(source.node_ids[index] for index in members)
    labels = tuple(source.labels[index] for index in members)
    rows = tuple(
        sorted(
            (
                old_to_new[left],
                old_to_new[right],
                modality,
                weight,
            )
            for left, right, modality, weight in source.rows
            if left in old_to_new and right in old_to_new
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
    raw_files = {"nodes.csv": node_bytes, "relations.csv": relation_bytes, "features.npz": feature_bytes}
    relation_counts = {name: sum(row[2] == name for row in rows) for name in MODALITIES}
    dataset_id = f"socialgraph-fm:russia:answer-pack-{pack_number:02d}"
    manifest = GovernanceInputManifest.model_validate(
        {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "datasetId": dataset_id,
            "displayName": f"Russia answer pack {pack_number:02d}",
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
        manifest.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    archive_bytes = _fixed_archive(
        {
            "manifest.json": raw_manifest,
            "nodes.csv": node_bytes,
            "relations.csv": relation_bytes,
            "features.npz": feature_bytes,
        }
    )
    _atomic_bytes(destination, archive_bytes)
    descriptor = _bundle_descriptor(
        destination,
        dataset_id=dataset_id,
        raw_manifest=raw_manifest,
        raw_files=raw_files,
        node_ids=node_ids,
        rows=rows,
    )
    payload = descriptor.model_dump(mode="json", by_alias=True)
    payload["selection"] = dict(selection)
    return RussiaAnswerPackDescriptor.model_validate(payload)


def generate_russia_answer_packs(
    source_bundle: str | Path,
    output_directory: str | Path,
    *,
    frozen_scores: Sequence[float] | np.ndarray | None = None,
    validation: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Generate four neutral, deterministic answer-pack ZIPs and a catalog.

    Existing files with the four reserved names are atomically replaced.  No
    other file in ``output_directory`` is touched, which keeps prior samples
    and protocol artifacts intact.
    """

    source = _load_source(source_bundle)
    if len(source.node_ids) != 716 or len(source.rows) != 10_968:
        raise BundleValidationError("the source is not the frozen 716-node Russia replay")
    neighbors = _adjacency(source)
    components = _connected_components(neighbors)
    giant = set(components[0])
    if len(giant) < max(target for _seed, target in ANSWER_PACK_SELECTIONS):
        raise BundleValidationError("Russia replay giant component is too small for answer packs")
    risk_array: np.ndarray | None = None
    if frozen_scores is not None:
        risk_array = np.asarray(frozen_scores, dtype=np.float32)
        if risk_array.shape != (len(source.node_ids),) or not bool(np.isfinite(risk_array).all()):
            raise BundleValidationError("frozen scores do not align to the source graph")
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    descriptors: list[RussiaAnswerPackDescriptor] = []
    for number, (seed, target_size) in enumerate(ANSWER_PACK_SELECTIONS, start=1):
        members = _expand_neighborhood(
            neighbors, seed=seed, target_size=target_size, risk_flags=(risk_array >= 0.64) if risk_array is not None else None
        )
        if not set(members).issubset(giant):
            raise BundleValidationError("answer-pack selection escaped the source giant component")
        fused_edges = {
            (left, right)
            for left, right, _modality, _weight in source.rows
            if left in set(members) and right in set(members)
        }
        if len(fused_edges) > ANSWER_PACK_MAX_FUSED_EDGES:
            raise BundleValidationError("answer-pack fused edge budget exceeded")
        risk_summary: dict[str, Any] = {}
        if risk_array is not None:
            selected_scores = risk_array[np.asarray(members, dtype=np.int64)]
            high = int(np.count_nonzero(selected_scores >= 0.64))
            risk_summary = {
                "frozenHighRiskCount": high,
                "frozenHighRiskRate": round(high / len(members), 6),
            }
        isolated = sum(not (neighbors[index] & set(members)) for index in members)
        selection = {
            "seedNodeId": source.node_ids[seed],
            "seedIndex": seed,
            "targetNodeCount": target_size,
            "algorithm": "connected-frontier-min-induced-edge-growth",
            "componentCount": 1,
            "isolatedNodeCount": isolated,
            **risk_summary,
        }
        descriptor = _write_answer_pack(
            destination / ANSWER_PACK_FILENAMES[number - 1],
            source,
            members,
            pack_number=number,
            selection=selection,
        )
        payload = descriptor.model_dump(mode="json", by_alias=True)
        payload["selection"] = selection
        if validation is not None:
            payload["validation"] = dict(validation.get(ANSWER_PACK_FILENAMES[number - 1], {})) or None
        descriptors.append(RussiaAnswerPackDescriptor.model_validate(payload))
    recipe = {
        "recipeId": ANSWER_PACK_RECIPE_ID,
        "seeds": [seed for seed, _target in ANSWER_PACK_SELECTIONS],
        "targetNodeCounts": [target for _seed, target in ANSWER_PACK_SELECTIONS],
        "frontierOrdering": ["inducedEdgeGrowth", "nodeDegree", "riskFlag", "nodeIndex"],
        "riskThreshold": 0.64,
        "fusedEdgeBudget": ANSWER_PACK_MAX_FUSED_EDGES,
        "sourceNodeCount": len(source.node_ids),
        "sourceRelationRowCount": len(source.rows),
    }
    logical: dict[str, Any] = {
        "schemaVersion": ANSWER_PACK_CATALOG_SCHEMA_VERSION,
        "sourceFile": source.path.name,
        "sourceSha256": file_sha256(source.path),
        "recipeId": ANSWER_PACK_RECIPE_ID,
        "supportedModalities": list(MODALITIES),
        "packs": [item.model_dump(mode="json", by_alias=True) for item in descriptors],
        "selectionRecipe": recipe,
    }
    catalog_payload = {**logical, "catalogHash": canonical_sha256(logical)}
    RussiaAnswerPackCatalog.model_validate(catalog_payload)
    _atomic_bytes(
        destination / "catalog.json",
        json.dumps(catalog_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return destination / "catalog.json"


def verify_russia_answer_pack_catalog(
    source_bundle: str | Path, catalog_path: str | Path
) -> RussiaAnswerPackCatalog:
    """Verify catalog hashes and each answer-pack input contract."""

    source = _load_source(source_bundle)
    path = Path(catalog_path).expanduser().resolve(strict=True)
    catalog = RussiaAnswerPackCatalog.model_validate_json(path.read_bytes())
    if catalog.source_sha256 != file_sha256(source.path):
        raise BundleValidationError("answer-pack catalog source hash does not match source")
    root = path.parent
    for descriptor in catalog.packs:
        bundle_path = (root / descriptor.file_name).resolve(strict=True)
        if not bundle_path.is_relative_to(root) or bundle_path.is_symlink():
            raise BundleValidationError("answer-pack path is unsafe")
        if file_sha256(bundle_path) != descriptor.sha256 or bundle_path.stat().st_size != descriptor.bytes:
            raise BundleValidationError("answer-pack bytes do not match catalog")
        with zipfile.ZipFile(bundle_path) as archive:
            if tuple(archive.namelist()) != _MEMBERS:
                raise BundleValidationError("answer-pack members are not the fixed input contract")
            raw_files = {
                name: archive.read(name)
                for name in ("nodes.csv", "relations.csv", "features.npz")
            }
        pack = _load_source(bundle_path)
        if len(pack.node_ids) != descriptor.node_count or len(pack.rows) != descriptor.relation_row_count:
            raise BundleValidationError("answer-pack inventory disagrees with catalog")
        undirected = {(left, right) for left, right, _modality, _weight in pack.rows}
        if len(undirected) != descriptor.fused_undirected_edge_count:
            raise BundleValidationError("answer-pack fused edge count disagrees with catalog")
        if len(undirected) > ANSWER_PACK_MAX_FUSED_EDGES:
            raise BundleValidationError("answer-pack fused edge budget exceeded")
        actual_counts = {name: sum(row[2] == name for row in pack.rows) for name in MODALITIES}
        if actual_counts != descriptor.relation_edge_counts:
            raise BundleValidationError("answer-pack relation inventory disagrees with catalog")
        source_index = {node_id: index for index, node_id in enumerate(source.node_ids)}
        if any(node_id not in source_index for node_id in pack.node_ids):
            raise BundleValidationError("answer-pack contains a node outside the Russia source")
        selected = {source_index[node_id] for node_id in pack.node_ids}
        expected_rows = Counter(
            (source.node_ids[left], source.node_ids[right], modality, weight)
            for left, right, modality, weight in source.rows
            if left in selected and right in selected
        )
        actual_rows = Counter(
            (pack.node_ids[left], pack.node_ids[right], modality, weight)
            for left, right, modality, weight in pack.rows
        )
        if actual_rows != expected_rows:
            raise BundleValidationError("answer-pack relations are not an induced source subgraph")
        for position, node_id in enumerate(pack.node_ids):
            source_position = source_index[node_id]
            if not np.array_equal(pack.features[position], source.features[source_position]):
                raise BundleValidationError("answer-pack feature rows differ from the source")
        expected_dataset_hash = _dataset_content_hash(
            manifest_hash=hashlib.sha256(pack.raw_manifest).hexdigest(),
            file_digests={name: hashlib.sha256(value).hexdigest() for name, value in raw_files.items()},
            clean=False,
            removed=0,
        )
        expected_graph_hash = _graph_version_hash(
            pack.node_ids,
            tuple(sorted({(left, right) for left, right, _modality, _weight in pack.rows})),
        )
        if descriptor.dataset_content_hash != expected_dataset_hash:
            raise BundleValidationError("answer-pack dataset content hash disagrees with catalog")
        if descriptor.graph_version_hash != expected_graph_hash:
            raise BundleValidationError("answer-pack graph hash disagrees with catalog")
    return catalog


__all__ = [
    "ANSWER_PACK_CATALOG_SCHEMA_VERSION",
    "ANSWER_PACK_FILENAMES",
    "ANSWER_PACK_MAX_FUSED_EDGES",
    "ANSWER_PACK_NODE_RANGE",
    "ANSWER_PACK_RECIPE_ID",
    "ANSWER_PACK_SELECTIONS",
    "RussiaAnswerPackCatalog",
    "RussiaAnswerPackDescriptor",
    "generate_russia_answer_packs",
    "select_answer_pack_members",
    "verify_russia_answer_pack_catalog",
]
