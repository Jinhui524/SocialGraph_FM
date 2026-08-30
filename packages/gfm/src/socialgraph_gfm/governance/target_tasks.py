"""Deterministic operator-only target-task bundles from trusted NPY corpora."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import zipfile
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.global_model.contracts import CountryId
from socialgraph_gfm.global_model.corpus import GlobalCountryCorpus, load_corpus_index

from .contracts import (
    INPUT_SCHEMA_VERSION,
    MAX_NODES,
    MAX_RELATION_ROWS,
    MODALITIES,
    GovernanceInputManifest,
)
from .materialize import _graph_arrays, _load_features, _parse_nodes, _parse_relations

TASK_BUNDLE_SCHEMA_VERSION = "socialgraph-fm.governance-target-task-bundle/1.0"
TARGET_RECEIPT_SCHEMA_VERSION = "socialgraph-fm.governance-target-domain-receipt/2.0"
LABEL_SET_SCHEMA_VERSION = "socialgraph-fm.governance-target-label-set/2.0"
LABEL_RECEIPT_SCHEMA_VERSION = "socialgraph-fm.governance-target-label-receipt/2.0"
GOVERNANCE_TARGET_CATALOG_SCHEMA_VERSION = "socialgraph-fm.governance-target-catalog/1.0"
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_INNER_MEMBERS = ("manifest.json", "nodes.csv", "relations.csv", "features.npz")
_ZERO_SHOT_MEMBERS = ("task.json", "inference.zip", "target-receipt.json")
_FEW_SHOT_MEMBERS = (*_ZERO_SHOT_MEMBERS, "labels.json", "label-receipt.json")
_MAX_OUTER_BYTES = 128 * 1024 * 1024
_MAX_MEMBER_BYTES = 96 * 1024 * 1024
_MAX_COMPRESSION_RATIO = 200
_TARGET_NODE_COUNT = 108
_MIN_FUSED_EDGES = 180
_MAX_FUSED_EDGES = 260
_GOVERNANCE_CATALOG_NAME = "governance-target-tasks.catalog.json"
_GOVERNANCE_CATALOG_ROOT = ".governance-target-catalog"
_GOVERNANCE_PACKAGE_NAMES = (
    "target-domain-a-zero.sgtask.zip",
    "target-domain-b-few.sgtask.zip",
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class TaskFileDescriptor(_FrozenModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9.-]{0,63}$")
    sha256: str = Field(pattern=_HASH_PATTERN)
    bytes: Annotated[int, Field(ge=1, le=_MAX_MEMBER_BYTES)]


class TargetTaskDocument(_FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-task-bundle/1.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)
    mode: Literal["zero_shot", "few_shot"]
    node_count: int = Field(alias="nodeCount", ge=1, le=MAX_NODES)
    fused_edge_count: int = Field(alias="fusedEdgeCount", ge=1, le=MAX_RELATION_ROWS)
    modalities: tuple[Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"], ...]
    inference: TaskFileDescriptor
    target_receipt: TaskFileDescriptor = Field(alias="targetReceipt")
    labels: TaskFileDescriptor | None = None
    label_receipt: TaskFileDescriptor | None = Field(default=None, alias="labelReceipt")

    @model_validator(mode="after")
    def validate_inventory(self) -> TargetTaskDocument:
        if (
            not self.modalities
            or len(set(self.modalities)) != len(self.modalities)
            or tuple(sorted(self.modalities, key=MODALITIES.index)) != self.modalities
        ):
            raise ValueError("task modalities must be a nonempty ordered unique subset")
        if any(ord(character) < 32 for character in self.display_name):
            raise ValueError("displayName contains a control character")
        if (
            self.inference.name != "inference.zip"
            or self.target_receipt.name != "target-receipt.json"
        ):
            raise ValueError("task file descriptors have invalid names")
        detached = self.labels is not None and self.label_receipt is not None
        if (self.mode == "few_shot") != detached:
            raise ValueError("few-shot tasks require both detached label documents")
        if self.labels is not None and self.labels.name != "labels.json":
            raise ValueError("labels descriptor has an invalid name")
        if self.label_receipt is not None and self.label_receipt.name != "label-receipt.json":
            raise ValueError("label receipt descriptor has an invalid name")
        return self


class TargetDomainReceipt(_FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-domain-receipt/2.0"] = Field(
        alias="schemaVersion"
    )
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    country_id: str = Field(alias="countryId", min_length=1, max_length=200)
    source_content_hash: str = Field(alias="sourceContentHash", pattern=_HASH_PATTERN)
    source_manifest_sha256: str = Field(alias="sourceManifestSha256", pattern=_HASH_PATTERN)
    graph_population: str = Field(alias="graphPopulation", min_length=1, max_length=200)
    graph_population_mask_sha256: str | None = Field(
        default=None, alias="graphPopulationMaskSha256", pattern=_HASH_PATTERN
    )
    label_eligibility: str = Field(alias="labelEligibility", min_length=1, max_length=200)
    label_eligibility_mask_sha256: str | None = Field(
        default=None, alias="labelEligibilityMaskSha256", pattern=_HASH_PATTERN
    )
    inference_sha256: str = Field(alias="inferenceSha256", pattern=_HASH_PATTERN)
    node_set_sha256: str = Field(alias="nodeSetSha256", pattern=_HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1, le=MAX_NODES)
    fused_edge_count: int = Field(alias="fusedEdgeCount", ge=1, le=MAX_RELATION_ROWS)
    modalities: tuple[Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"], ...]
    connected: bool
    selection_recipe: Mapping[str, object] = Field(alias="selectionRecipe")
    receipt_hash: str = Field(alias="receiptHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self) -> TargetDomainReceipt:
        if (
            not self.modalities
            or len(set(self.modalities)) != len(self.modalities)
            or tuple(sorted(self.modalities, key=MODALITIES.index)) != self.modalities
        ):
            raise ValueError("receipt modalities must be a nonempty ordered unique subset")
        if (self.graph_population != "full") != (self.graph_population_mask_sha256 is not None):
            raise ValueError("graph population mask binding is inconsistent")
        if (self.label_eligibility != "none") != (self.label_eligibility_mask_sha256 is not None):
            raise ValueError("label eligibility mask binding is inconsistent")
        if self.selection_recipe.get("scoreInputs") != []:
            raise ValueError("target selection must not use scores, logits, or ranks")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("receiptHash mismatch")
        return self


class TargetLabel(_FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    label: Literal["positive", "negative"]
    structural_stratum: int = Field(alias="structuralStratum", ge=0, le=3)
    fused_degree: int = Field(alias="fusedDegree", ge=1)


class TargetLabelSetV2(_FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-label-set/2.0"] = Field(
        alias="schemaVersion"
    )
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    inference_sha256: str = Field(alias="inferenceSha256", pattern=_HASH_PATTERN)
    labels: tuple[TargetLabel, ...] = Field(min_length=8, max_length=256)
    positive_count: int = Field(alias="positiveCount", ge=0, le=256)
    negative_count: int = Field(alias="negativeCount", ge=0, le=256)
    label_set_hash: str = Field(alias="labelSetHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_labels(self) -> TargetLabelSetV2:
        by_node: dict[str, str] = {}
        for row in self.labels:
            previous = by_node.get(row.node_id)
            if previous is not None:
                if previous != row.label:
                    raise ValueError("label set contains conflicting labels")
                raise ValueError("label set contains a duplicate node")
            by_node[row.node_id] = row.label
        positive = sum(row.label == "positive" for row in self.labels)
        negative = len(self.labels) - positive
        if min(positive, negative) < 4:
            raise ValueError("label set requires at least four labels per class")
        if (positive, negative) != (self.positive_count, self.negative_count):
            raise ValueError("label class counts are inconsistent")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"label_set_hash"})
        )
        if self.label_set_hash != expected:
            raise ValueError("labelSetHash mismatch")
        return self


class TargetLabelReceipt(_FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-label-receipt/2.0"] = Field(
        alias="schemaVersion"
    )
    task_id: str = Field(alias="taskId", pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$")
    target_receipt_hash: str = Field(alias="targetReceiptHash", pattern=_HASH_PATTERN)
    labels_sha256: str = Field(alias="labelsSha256", pattern=_HASH_PATTERN)
    source_labels_sha256: str = Field(alias="sourceLabelsSha256", pattern=_HASH_PATTERN)
    eligibility_mask_sha256: str = Field(alias="eligibilityMaskSha256", pattern=_HASH_PATTERN)
    eligible_node_ids: tuple[str, ...] = Field(
        alias="eligibleNodeIds", min_length=8, max_length=MAX_NODES
    )
    selection_recipe: Mapping[str, object] = Field(alias="selectionRecipe")
    receipt_hash: str = Field(alias="receiptHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self) -> TargetLabelReceipt:
        if len(set(self.eligible_node_ids)) != len(self.eligible_node_ids):
            raise ValueError("eligible node inventory contains duplicates")
        if self.selection_recipe.get("scoreInputs") != []:
            raise ValueError("label selection recipe is invalid")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"receipt_hash"})
        )
        if self.receipt_hash != expected:
            raise ValueError("label receiptHash mismatch")
        return self


@dataclass(frozen=True)
class GeneratedGovernanceTasks:
    zero_shot: Path
    few_shot: Path


@dataclass(frozen=True)
class VerifiedTargetTask:
    path: Path
    task: TargetTaskDocument
    receipt: TargetDomainReceipt
    labels: TargetLabelSetV2 | None
    label_receipt: TargetLabelReceipt | None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _governance_package_declarations(
    packages: Sequence[tuple[str, bytes]],
) -> list[dict[str, object]]:
    """Return the one canonical package inventory used by publish and verify."""

    if len(packages) != 2:
        raise ValueError("governance target package inventory must contain two packages")
    return [
        {
            "role": role,
            "name": name,
            "sha256": _sha256_bytes(value),
            "bytes": len(value),
        }
        for role, (name, value) in zip(("zero_shot", "few_shot"), packages, strict=True)
    ]


def _governance_generation_id(declarations: Sequence[Mapping[str, object]]) -> str:
    """Derive a generation identity from exact role/name/digest/size declarations."""

    return canonical_sha256({"packages": list(declarations)})


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability; Windows does not expose openable dirs."""

    try:
        descriptor = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _fixed_zip_bytes(entries: Sequence[tuple[str, bytes]], *, compresslevel: int = 6) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", allowZip64=False) as archive:
        for name, value in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, value, compresslevel=compresslevel)
    return stream.getvalue()


def _features_npz(node_ids: Sequence[str], features: np.ndarray) -> bytes:
    return _fixed_zip_bytes(
        (
            ("node_ids.npy", _npy_bytes(np.asarray(node_ids))),
            ("text_features.npy", _npy_bytes(np.asarray(features, dtype=np.float32))),
        )
    )


def _csv_bytes(header: Sequence[str], rows: Iterable[Sequence[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _stable_digest(country: str, salt: int, *values: object) -> bytes:
    logical = ":".join((country, str(salt), *(str(value) for value in values)))
    return hashlib.sha256(logical.encode("ascii")).digest()


def _neighbors(corpus: GlobalCountryCorpus, node: int) -> tuple[int, ...]:
    fused = corpus.fused_csr
    return tuple(map(int, fused.indices[fused.indptr[node] : fused.indptr[node + 1]]))


def _induced_edge_count(corpus: GlobalCountryCorpus, selected: set[int]) -> int:
    return sum(
        1
        for source in selected
        for target in _neighbors(corpus, source)
        if source < target and target in selected
    )


def _relation_edges(
    corpus: GlobalCountryCorpus, selected: set[int], modality: str
) -> tuple[tuple[int, int, float], ...]:
    relation = corpus.relation(modality)
    rows: list[tuple[int, int, float]] = []
    for source in selected:
        start, stop = int(relation.indptr[source]), int(relation.indptr[source + 1])
        for position in range(start, stop):
            target = int(relation.indices[position])
            if source < target and target in selected:
                rows.append((source, target, float(relation.weights[position])))
    return tuple(sorted(rows))


def _shortest_path(
    corpus: GlobalCountryCorpus,
    selected: set[int],
    target: int,
    allowed: set[int],
    *,
    country: str,
) -> tuple[int, ...]:
    if target in selected:
        return ()
    queue = deque([target])
    parents: dict[int, int | None] = {target: None}
    meeting: int | None = None
    while queue and meeting is None:
        node = queue.popleft()
        for neighbor in sorted(
            _neighbors(corpus, node), key=lambda value: _stable_digest(country, 0, value)
        ):
            if neighbor not in allowed or neighbor in parents:
                continue
            parents[neighbor] = node
            if neighbor in selected:
                meeting = neighbor
                break
            queue.append(neighbor)
    if meeting is None:
        raise ValueError("required target seeds are not in one connected source component")
    path: list[int] = []
    cursor = meeting
    while parents[cursor] is not None:
        cursor = cast(int, parents[cursor])
        path.append(cursor)
    return tuple(path)


def _modality_anchor_edges(
    corpus: GlobalCountryCorpus, allowed: set[int], country: str
) -> tuple[tuple[int, int], ...]:
    anchors: list[tuple[int, int]] = []
    for modality in MODALITIES:
        relation = corpus.relation(modality)
        edges = [
            (source, int(relation.indices[position]))
            for source in allowed
            for position in range(int(relation.indptr[source]), int(relation.indptr[source + 1]))
            if source < int(relation.indices[position])
            and int(relation.indices[position]) in allowed
        ]
        if not edges:
            raise ValueError(f"{country} source scope has no {modality} edge")
        anchors.append(min(edges, key=lambda edge: _stable_digest(country, 0, modality, *edge)))
    return tuple(anchors)


def _select_nodes(
    corpus: GlobalCountryCorpus,
    allowed: set[int],
    country: str,
    *,
    required_nodes: Sequence[int] = (),
) -> tuple[int, ...]:
    anchors = _modality_anchor_edges(corpus, allowed, country)
    selected: set[int] = set(anchors[0])
    seeds = [node for edge in anchors[1:] for node in edge] + list(required_nodes)
    for node in seeds:
        path = _shortest_path(corpus, selected, node, allowed, country=country)
        selected.update(path)
        if len(selected) > _TARGET_NODE_COUNT:
            raise ValueError("required structural seeds exceed the target node budget")
    frontier_counts: dict[int, int] = {}
    for node in selected:
        for neighbor in _neighbors(corpus, node):
            if neighbor in allowed and neighbor not in selected:
                frontier_counts[neighbor] = frontier_counts.get(neighbor, 0) + 1
    current_edges = _induced_edge_count(corpus, selected)
    while len(selected) < _TARGET_NODE_COUNT:
        candidates: list[tuple[float, int, bytes, int]] = []
        remaining_nodes = _TARGET_NODE_COUNT - len(selected)
        desired_increment = max(1.0, (220 - current_edges) / remaining_nodes)
        for node, increment in frontier_counts.items():
            if increment and current_edges + increment <= _MAX_FUSED_EDGES:
                candidates.append(
                    (
                        abs(increment - desired_increment),
                        increment,
                        _stable_digest(country, 0, node),
                        node,
                    )
                )
        if not candidates:
            raise ValueError("target sampler cannot meet the connected edge budget")
        node = min(candidates)[-1]
        current_edges += frontier_counts.pop(node)
        selected.add(node)
        for neighbor in _neighbors(corpus, node):
            if neighbor in allowed and neighbor not in selected:
                frontier_counts[neighbor] = frontier_counts.get(neighbor, 0) + 1
    fused_edges = current_edges
    if not _MIN_FUSED_EDGES <= fused_edges <= _MAX_FUSED_EDGES:
        raise ValueError("target sampler did not meet the fused edge budget")
    if any(not _relation_edges(corpus, selected, modality) for modality in MODALITIES):
        raise ValueError("target sampler did not retain all five modalities")
    return tuple(sorted(selected))


def _anonymous_node_id(country: str, source_index: int) -> str:
    value = hashlib.sha256(f"target-task-v1:{country}:{source_index}".encode("ascii")).hexdigest()
    return f"anonymous:{value[:20]}"


def _target_graph_facts(
    corpus: GlobalCountryCorpus,
    country: str,
    selected_nodes: Sequence[int],
) -> tuple[dict[int, int], dict[int, int]]:
    selected = set(selected_nodes)
    degrees = {node: len(set(_neighbors(corpus, node)) & selected) for node in selected_nodes}
    ordered = sorted(
        selected_nodes,
        key=lambda node: (degrees[node], _anonymous_node_id(country, node)),
    )
    strata = {node: min(3, position * 4 // len(ordered)) for position, node in enumerate(ordered)}
    return degrees, strata


def _uae_provisional_label_requirements(
    corpus: GlobalCountryCorpus,
    eligible: set[int],
    base_nodes: Sequence[int],
) -> tuple[int, ...]:
    """Choose score-blind seeds predicted to cover induced target quartiles."""

    base = set(base_nodes)
    base_degrees, _ = _target_graph_facts(corpus, "UAE", base_nodes)
    base_keys = sorted((base_degrees[node], _anonymous_node_id("UAE", node)) for node in base_nodes)
    required: list[int] = []
    for label in (1, 0):
        for stratum in range(4):
            candidates: list[int] = []
            for node in eligible:
                if int(corpus.labels[node]) != label:
                    continue
                predicted_degree = (
                    base_degrees[node]
                    if node in base
                    else max(1, len(set(_neighbors(corpus, node)) & base))
                )
                key = (predicted_degree, _anonymous_node_id("UAE", node))
                predicted_position = sum(existing < key for existing in base_keys)
                predicted_stratum = min(3, predicted_position * 4 // len(base_keys))
                if predicted_stratum == stratum:
                    candidates.append(node)
            candidates.sort(
                key=lambda node: _stable_digest("UAE-target-label", 0, label, stratum, node)
            )
            if len(candidates) < 2:
                raise ValueError("UAE fold-0 labels cannot seed all target structural strata")
            required.extend(candidates[:2])
    return tuple(required)


def _uae_target_label_seeds(
    corpus: GlobalCountryCorpus,
    eligible: set[int],
    target_nodes: Sequence[int],
) -> tuple[tuple[int, int, int, int], ...]:
    degrees, strata = _target_graph_facts(corpus, "UAE", target_nodes)
    target = set(target_nodes)
    rows: list[tuple[int, int, int, int]] = []
    for label in (1, 0):
        class_nodes = sorted(
            node for node in eligible & target if int(corpus.labels[node]) == label
        )
        vectors = np.asarray(corpus.text_features[class_nodes], dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != 768 or not np.isfinite(vectors).all():
            raise ValueError("UAE label-selection input features are invalid")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0):
            raise ValueError("UAE label-selection input features contain a zero vector")
        normalized = vectors / norms[:, None]
        centroid = normalized.mean(axis=0, dtype=np.float64)
        centroid_norm = float(np.linalg.norm(centroid))
        if not np.isfinite(centroid_norm) or centroid_norm <= 0:
            raise ValueError("UAE label-selection class centroid is invalid")
        centroid /= centroid_norm
        similarities = {
            node: float(vector @ centroid)
            for node, vector in zip(class_nodes, normalized, strict=True)
        }
        for stratum in range(4):
            candidates = sorted(
                (node for node in class_nodes if strata[node] == stratum),
                key=lambda node: (
                    -round(similarities[node], 12),
                    _anonymous_node_id("UAE", node),
                ),
            )
            if len(candidates) < 2:
                raise ValueError("UAE target graph labels do not cover structural strata")
            rows.extend((node, label, stratum, degrees[node]) for node in candidates[:2])
    return tuple(rows)


def _inner_bundle(
    corpus: GlobalCountryCorpus,
    country: str,
    selected_nodes: Sequence[int],
    display_name: str,
) -> tuple[bytes, dict[int, str], int]:
    selected = set(selected_nodes)
    identifiers = {node: _anonymous_node_id(country, node) for node in selected_nodes}
    relation_rows: list[tuple[object, ...]] = []
    observed: list[str] = []
    for modality in MODALITIES:
        edges = _relation_edges(corpus, selected, modality)
        if edges:
            observed.append(modality)
        relation_rows.extend(
            (identifiers[source], identifiers[target], modality, format(weight, ".17g"))
            for source, target, weight in edges
        )
    nodes_bytes = _csv_bytes(
        ("node_id", "display_name"),
        (
            (identifiers[node], f"Anonymous account {index}")
            for index, node in enumerate(selected_nodes)
        ),
    )
    relations_bytes = _csv_bytes(("source", "target", "modality", "weight"), relation_rows)
    features_bytes = _features_npz(
        tuple(identifiers[node] for node in selected_nodes),
        np.asarray(corpus.text_features[np.asarray(selected_nodes)], dtype=np.float32),
    )
    files = {
        name: {"sha256": _sha256_bytes(value), "bytes": len(value)}
        for name, value in (
            ("nodes.csv", nodes_bytes),
            ("relations.csv", relations_bytes),
            ("features.npz", features_bytes),
        )
    }
    manifest = GovernanceInputManifest.model_validate(
        {
            "schemaVersion": INPUT_SCHEMA_VERSION,
            "datasetId": f"socialgraph-fm:governance:{country.lower()}:target",
            "displayName": display_name,
            "nodeCount": len(selected_nodes),
            "relationRowCount": len(relation_rows),
            "featureDimension": 768,
            "modalities": observed,
            "files": files,
            "license": "Internal authorized governance material",
        }
    )
    inference = _fixed_zip_bytes(
        (
            ("manifest.json", _canonical_bytes(manifest.model_dump(mode="json"))),
            ("nodes.csv", nodes_bytes),
            ("relations.csv", relations_bytes),
            ("features.npz", features_bytes),
        )
    )
    return inference, identifiers, _induced_edge_count(corpus, selected)


def _descriptor(name: str, value: bytes) -> dict[str, object]:
    return {"name": name, "sha256": _sha256_bytes(value), "bytes": len(value)}


def _receipt_bytes(
    *,
    task_id: str,
    country: str,
    corpus: GlobalCountryCorpus,
    inference: bytes,
    identifiers: Mapping[int, str],
    fused_edges: int,
    graph_population: str,
    graph_mask_hash: str | None,
    label_eligibility: str,
    label_mask_hash: str | None,
) -> bytes:
    payload: dict[str, object] = {
        "schemaVersion": TARGET_RECEIPT_SCHEMA_VERSION,
        "taskId": task_id,
        "countryId": country,
        "sourceContentHash": corpus.manifest.content_hash,
        "sourceManifestSha256": file_sha256(corpus.root / "manifest.json"),
        "graphPopulation": graph_population,
        "graphPopulationMaskSha256": graph_mask_hash,
        "labelEligibility": label_eligibility,
        "labelEligibilityMaskSha256": label_mask_hash,
        "inferenceSha256": _sha256_bytes(inference),
        "nodeSetSha256": canonical_sha256(sorted(identifiers.values())),
        "nodeCount": len(identifiers),
        "fusedEdgeCount": fused_edges,
        "modalities": list(MODALITIES),
        "connected": True,
        "selectionRecipe": {
            "version": "connected-source-csr-stable-hash-v1",
            "nodeCount": _TARGET_NODE_COUNT,
            "minimumFusedEdges": _MIN_FUSED_EDGES,
            "maximumFusedEdges": _MAX_FUSED_EDGES,
            "requiredModalities": list(MODALITIES),
            "scoreInputs": [],
        },
    }
    payload["receiptHash"] = canonical_sha256(payload)
    TargetDomainReceipt.model_validate(payload)
    return _canonical_bytes(payload)


def _outer_bundle(
    task_id: str,
    display_name: str,
    inference: bytes,
    receipt: bytes,
    fused_edges: int,
    *,
    labels: bytes | None = None,
    label_receipt: bytes | None = None,
) -> bytes:
    mode = "few_shot" if labels is not None else "zero_shot"
    payload: dict[str, object] = {
        "schemaVersion": TASK_BUNDLE_SCHEMA_VERSION,
        "taskId": task_id,
        "displayName": display_name,
        "mode": mode,
        "nodeCount": _TARGET_NODE_COUNT,
        "fusedEdgeCount": fused_edges,
        "modalities": list(MODALITIES),
        "inference": _descriptor("inference.zip", inference),
        "targetReceipt": _descriptor("target-receipt.json", receipt),
    }
    entries: list[tuple[str, bytes]] = [
        ("task.json", b""),
        ("inference.zip", inference),
        ("target-receipt.json", receipt),
    ]
    if labels is not None and label_receipt is not None:
        payload["labels"] = _descriptor("labels.json", labels)
        payload["labelReceipt"] = _descriptor("label-receipt.json", label_receipt)
        entries.extend((("labels.json", labels), ("label-receipt.json", label_receipt)))
    TargetTaskDocument.model_validate(payload)
    entries[0] = ("task.json", _canonical_bytes(payload))
    return _fixed_zip_bytes(entries)


def _generate_bytes(corpus_root: Path) -> tuple[tuple[str, bytes], tuple[str, bytes]]:
    index = load_corpus_index(corpus_root, verify_manifests=True)
    cuba = index.load_country(cast(CountryId, "cuba"), verify_hashes=True, verify_values=True)
    uae = index.load_country(cast(CountryId, "UAE"), verify_hashes=True, verify_values=True)
    cuba_mask_path = cuba.root / "arrays" / "split_full_0_test.npy"
    uae_mask_path = uae.root / "arrays" / "split_full_0_test.npy"
    cuba_test = set(map(int, np.flatnonzero(cuba.split("full-fold-0").test_mask)))
    uae_test = set(map(int, np.flatnonzero(uae.split("full-fold-0").test_mask)))

    cuba_nodes = _select_nodes(cuba, cuba_test, "cuba")
    uae_base_nodes = _select_nodes(uae, set(range(uae.manifest.node_count)), "UAE")
    provisional_label_nodes = _uae_provisional_label_requirements(uae, uae_test, uae_base_nodes)
    uae_nodes = _select_nodes(
        uae,
        set(range(uae.manifest.node_count)),
        "UAE",
        required_nodes=provisional_label_nodes,
    )
    label_seeds = _uae_target_label_seeds(uae, uae_test, uae_nodes)
    inference_a, ids_a, fused_a = _inner_bundle(cuba, "cuba", cuba_nodes, "匿名目标数据源 A")
    inference_b, ids_b, fused_b = _inner_bundle(uae, "UAE", uae_nodes, "匿名目标数据源 B")
    receipt_a = _receipt_bytes(
        task_id="target-a",
        country="cuba",
        corpus=cuba,
        inference=inference_a,
        identifiers=ids_a,
        fused_edges=fused_a,
        graph_population="fold0_test",
        graph_mask_hash=file_sha256(cuba_mask_path),
        label_eligibility="none",
        label_mask_hash=None,
    )
    receipt_b = _receipt_bytes(
        task_id="target-b",
        country="UAE",
        corpus=uae,
        inference=inference_b,
        identifiers=ids_b,
        fused_edges=fused_b,
        graph_population="full",
        graph_mask_hash=None,
        label_eligibility="fold0_test",
        label_mask_hash=file_sha256(uae_mask_path),
    )
    label_rows = [
        {
            "nodeId": ids_b[node],
            "label": "positive" if label == 1 else "negative",
            "structuralStratum": stratum,
            "fusedDegree": degree,
        }
        for node, label, stratum, degree in label_seeds
    ]
    label_rows.sort(key=lambda row: str(row["nodeId"]))
    labels_payload: dict[str, object] = {
        "schemaVersion": LABEL_SET_SCHEMA_VERSION,
        "taskId": "target-b",
        "inferenceSha256": _sha256_bytes(inference_b),
        "labels": label_rows,
        "positiveCount": 8,
        "negativeCount": 8,
    }
    labels_payload["labelSetHash"] = canonical_sha256(labels_payload)
    TargetLabelSetV2.model_validate(labels_payload)
    labels_bytes = _canonical_bytes(labels_payload)
    receipt_b_model = TargetDomainReceipt.model_validate_json(receipt_b)
    label_receipt_payload: dict[str, object] = {
        "schemaVersion": LABEL_RECEIPT_SCHEMA_VERSION,
        "taskId": "target-b",
        "targetReceiptHash": receipt_b_model.receipt_hash,
        "labelsSha256": _sha256_bytes(labels_bytes),
        "sourceLabelsSha256": file_sha256(uae.root / "arrays" / "labels.npy"),
        "eligibilityMaskSha256": file_sha256(uae_mask_path),
        "eligibleNodeIds": sorted(_anonymous_node_id("UAE", node) for node in uae_test),
        "selectionRecipe": {
            "version": "fold0-test-target-class-centroid-cosine-v3",
            "stratification": "target-induced-fused-degree-node-id-quartile",
            "structuralStrata": 4,
            "labelsPerClass": 8,
            "labelsPerClassPerStratum": 2,
            "requiredSeedRecipe": ("base-target-predicted-induced-quartile-stable-hash-v1"),
            "featureBasis": "authenticated-target-input-text-features-768-float32",
            "representativeness": ("per-class-unit-centroid-cosine-within-target-stratum"),
            "similarityQuantizationDecimals": 12,
            "tieBreak": "anonymous-node-id-ascending",
            "scoreInputs": [],
        },
    }
    label_receipt_payload["receiptHash"] = canonical_sha256(label_receipt_payload)
    TargetLabelReceipt.model_validate(label_receipt_payload)
    label_receipt_bytes = _canonical_bytes(label_receipt_payload)
    return (
        (
            "target-domain-a-zero.sgtask.zip",
            _outer_bundle("target-a", "目标域网络 A", inference_a, receipt_a, fused_a),
        ),
        (
            "target-domain-b-few.sgtask.zip",
            _outer_bundle(
                "target-b",
                "目标域网络 B",
                inference_b,
                receipt_b,
                fused_b,
                labels=labels_bytes,
                label_receipt=label_receipt_bytes,
            ),
        ),
    )


def generate_governance_target_tasks(
    corpus_root: str | Path, output_directory: str | Path
) -> GeneratedGovernanceTasks:
    """Publish both task bundles behind one atomic catalog pointer.

    An existing valid pointer is retained for identical content and replaced only
    after a complete new generation has been materialized and verified.  Previous
    generations are intentionally retained so a failed pointer replacement cannot
    invalidate the active catalog.
    """

    corpus = Path(corpus_root).expanduser().resolve(strict=True)
    requested = Path(output_directory).expanduser().absolute()
    if requested.exists() and (requested.is_symlink() or requested.is_junction()):
        raise ValueError("governance adaptation output directory cannot be a reparse point")
    destination = requested.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    generated = _generate_bytes(corpus)
    if tuple(name for name, _ in generated) != _GOVERNANCE_PACKAGE_NAMES:
        raise ValueError("governance generator returned an invalid package inventory")
    catalog_path = destination / _GOVERNANCE_CATALOG_NAME
    catalog_root = destination / _GOVERNANCE_CATALOG_ROOT
    root_package_paths = tuple(destination / name for name in _GOVERNANCE_PACKAGE_NAMES)
    if catalog_path.exists() and (catalog_path.is_symlink() or not catalog_path.is_file()):
        raise ValueError("governance target catalog pointer is unsafe")
    if catalog_root.exists() and (
        catalog_root.is_symlink() or catalog_root.is_junction() or not catalog_root.is_dir()
    ):
        raise ValueError("governance target catalog root is unsafe")
    if any(path.exists() for path in root_package_paths):
        raise FileExistsError("root-level governance packages are reserved; run reset first")

    package_declarations = _governance_package_declarations(generated)
    generation_id = _governance_generation_id(package_declarations)
    catalog_targets = [
        {
            "role": declaration["role"],
            "path": f"{_GOVERNANCE_CATALOG_ROOT}/{generation_id}/{declaration['name']}",
            "sha256": declaration["sha256"],
            "bytes": declaration["bytes"],
        }
        for declaration in package_declarations
    ]
    catalog: dict[str, object] = {
        "schemaVersion": GOVERNANCE_TARGET_CATALOG_SCHEMA_VERSION,
        "generationId": generation_id,
        "targets": catalog_targets,
    }
    catalog["catalogHash"] = canonical_sha256(catalog)
    catalog_bytes = _canonical_bytes(catalog)

    active_paths: tuple[Path, Path] | None = None
    if catalog_path.exists():
        active_paths = _governance_catalog_paths(destination)
        active_values = tuple(path.read_bytes() for path in active_paths)
        active_declarations = _governance_package_declarations(
            tuple(
                (name, value)
                for name, value in zip(_GOVERNANCE_PACKAGE_NAMES, active_values, strict=True)
            )
        )
        if active_declarations == package_declarations:
            return GeneratedGovernanceTasks(zero_shot=active_paths[0], few_shot=active_paths[1])

    catalog_root.mkdir(parents=True, exist_ok=True)
    _fsync_directory(destination)
    staging_directory = catalog_root / f".{generation_id}.{os.getpid()}.tmp"
    generation_directory = catalog_root / generation_id
    generation_created = False
    committed = False
    catalog_temporary = destination / f".{_GOVERNANCE_CATALOG_NAME}.{os.getpid()}.tmp"
    try:
        staging_directory.mkdir()
        for name, value in generated:
            with (staging_directory / name).open("xb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        _fsync_directory(staging_directory)
        _verify_generation_directory(staging_directory, generated)
        if generation_directory.exists():
            _verify_generation_directory(generation_directory, generated)
            staging_directory.rmdir()
        else:
            os.replace(staging_directory, generation_directory)
            generation_created = True
            _fsync_directory(catalog_root)
            _verify_generation_directory(generation_directory, generated)
        with catalog_temporary.open("xb") as stream:
            stream.write(catalog_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        # This is the sole replacement of the active catalog pointer.
        os.replace(catalog_temporary, catalog_path)
        committed = True
        _fsync_directory(destination)
    finally:
        catalog_temporary.unlink(missing_ok=True)
        _remove_staged_generation(staging_directory)
        if (
            not committed
            and generation_created
            and generation_directory.is_dir()
            and not generation_directory.is_symlink()
        ):
            _remove_staged_generation(generation_directory)
            _fsync_directory(catalog_root)
    paths = tuple(generation_directory / name for name in _GOVERNANCE_PACKAGE_NAMES)
    return GeneratedGovernanceTasks(zero_shot=paths[0], few_shot=paths[1])


def _verify_generation_directory(
    directory: Path,
    generated: Sequence[tuple[str, bytes]],
) -> None:
    """Verify a staged/existing generation without trusting its directory name."""

    if not directory.exists():
        raise ValueError("governance target generation is unavailable")
    if directory.is_symlink() or directory.is_junction() or not directory.is_dir():
        raise ValueError("governance target generation is unsafe")
    expected_names = tuple(name for name, _ in generated)
    children = tuple(directory.iterdir())
    if tuple(sorted(child.name for child in children)) != tuple(sorted(expected_names)):
        raise ValueError("governance target generation inventory is invalid")
    for name, expected in generated:
        candidate = directory / name
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError("governance target generation package is unavailable")
        if candidate.read_bytes() != expected:
            raise ValueError("governance target generation package bytes changed")


def _remove_staged_generation(directory: Path) -> None:
    """Remove only a private, unpublished generation staged by this process."""

    if not directory.is_dir() or directory.is_symlink() or directory.is_junction():
        return
    for name in _GOVERNANCE_PACKAGE_NAMES:
        candidate = directory / name
        if candidate.is_file() and not candidate.is_symlink():
            candidate.unlink()
    try:
        directory.rmdir()
    except OSError:
        # Unexpected content is left for explicit operator reset instead of deleted.
        pass


def _governance_catalog_paths(root: Path) -> tuple[Path, Path]:
    catalog_path = root / _GOVERNANCE_CATALOG_NAME
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise FileNotFoundError("committed governance target catalog is unavailable")
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("governance target catalog is invalid") from error
    if not isinstance(catalog, dict) or set(catalog) != {
        "schemaVersion",
        "generationId",
        "targets",
        "catalogHash",
    }:
        raise ValueError("governance target catalog inventory is invalid")
    logical = {key: value for key, value in catalog.items() if key != "catalogHash"}
    generation_id = catalog.get("generationId")
    if (
        catalog.get("schemaVersion") != GOVERNANCE_TARGET_CATALOG_SCHEMA_VERSION
        or not isinstance(generation_id, str)
        or len(generation_id) != 64
        or any(character not in "0123456789abcdef" for character in generation_id)
        or catalog.get("catalogHash") != canonical_sha256(logical)
    ):
        raise ValueError("governance target catalog identity is invalid")
    targets = catalog.get("targets")
    if not isinstance(targets, list) or len(targets) != 2:
        raise ValueError("governance target catalog target inventory is invalid")
    paths: list[Path] = []
    package_values: list[tuple[str, bytes]] = []
    declared_declarations: list[dict[str, object]] = []
    for role, name, entry in zip(
        ("zero_shot", "few_shot"), _GOVERNANCE_PACKAGE_NAMES, targets, strict=True
    ):
        if not isinstance(entry, dict) or set(entry) != {"role", "path", "sha256", "bytes"}:
            raise ValueError("governance target catalog entry is invalid")
        expected_relative = PurePosixPath(_GOVERNANCE_CATALOG_ROOT, generation_id, name)
        if entry.get("role") != role or entry.get("path") != expected_relative.as_posix():
            raise ValueError("governance target catalog role binding is invalid")
        candidate = root.joinpath(*expected_relative.parts)
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError("governance target catalog package is unavailable")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise ValueError("governance target catalog package path is unsafe")
        value = candidate.read_bytes()
        if entry.get("bytes") != len(value) or entry.get("sha256") != _sha256_bytes(value):
            raise ValueError("governance target catalog package digest is invalid")
        package_values.append((name, value))
        declared_declarations.append(
            {
                "role": role,
                "name": name,
                "sha256": entry.get("sha256"),
                "bytes": entry.get("bytes"),
            }
        )
        paths.append(candidate)
    actual_declarations = _governance_package_declarations(package_values)
    if declared_declarations != actual_declarations:
        raise ValueError("governance target catalog package declarations are invalid")
    if generation_id != _governance_generation_id(actual_declarations):
        raise ValueError("governance target catalog generation identity is invalid")
    return cast(tuple[Path, Path], tuple(paths))


def _safe_zip_entries(raw: bytes, expected: tuple[str, ...], *, label: str) -> dict[str, bytes]:
    if len(raw) > _MAX_OUTER_BYTES:
        raise ValueError(f"{label} exceeds the byte limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            names = tuple(archive.namelist())
            if names != expected or len(set(names)) != len(names):
                raise ValueError(f"{label} member inventory is invalid")
            values: dict[str, bytes] = {}
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                mode = (info.external_attr >> 16) & 0o170000
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in info.filename
                    or len(path.parts) != 1
                    or mode == 0o120000
                ):
                    raise ValueError(f"{label} member path is unsafe")
                if info.file_size > _MAX_MEMBER_BYTES:
                    raise ValueError(f"{label} member exceeds the byte limit")
                if (
                    info.compress_size == 0
                    or info.file_size > info.compress_size * _MAX_COMPRESSION_RATIO
                ):
                    raise ValueError(f"{label} member compression ratio is unsafe")
                value = archive.read(info)
                if len(value) != info.file_size:
                    raise ValueError(f"{label} member size changed while reading")
                values[info.filename] = value
            return values
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ValueError(f"{label} is not a valid bounded ZIP") from error


def _verify_inner(
    raw: bytes,
) -> tuple[GovernanceInputManifest, set[str], int, dict[str, int], dict[str, int]]:
    entries = _safe_zip_entries(raw, _INNER_MEMBERS, label="inference ZIP")
    manifest = GovernanceInputManifest.model_validate_json(entries["manifest.json"])
    for name, descriptor in manifest.files.items():
        if (
            len(entries[name]) != descriptor.bytes
            or _sha256_bytes(entries[name]) != descriptor.sha256
        ):
            raise ValueError("inference member digest or size mismatch")
    ordered_node_ids, _ = _parse_nodes(entries["nodes.csv"], manifest.nodeCount)
    _load_features(entries["features.npz"], ordered_node_ids)
    relations, _, observed_modalities = _parse_relations(
        entries["relations.csv"],
        node_ids=ordered_node_ids,
        expected_rows=manifest.relationRowCount,
        clean_self_loops=False,
    )
    if observed_modalities != manifest.modalities:
        raise ValueError("manifest.modalities does not match observed relations")
    fused, _, _, _, _, _ = _graph_arrays(relations, len(ordered_node_ids))
    node_ids = set(ordered_node_ids)
    adjacency: dict[str, set[str]] = {node: set() for node in ordered_node_ids}
    for source_index, target_index in fused:
        source = ordered_node_ids[source_index]
        target = ordered_node_ids[target_index]
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen: set[str] = set()
    pending = [next(iter(node_ids))]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        pending.extend(adjacency[node] - seen)
    if seen != node_ids:
        raise ValueError("inference fused graph is not connected")
    degrees = {node: len(adjacency[node]) for node in ordered_node_ids}
    ordered_by_degree = sorted(ordered_node_ids, key=lambda node: (degrees[node], node))
    strata = {
        node: min(3, position * 4 // len(ordered_by_degree))
        for position, node in enumerate(ordered_by_degree)
    }
    return manifest, node_ids, len(fused), degrees, strata


def verify_target_task_bundle(path: str | Path) -> VerifiedTargetTask:
    source = Path(path).expanduser().resolve(strict=True)
    raw = source.read_bytes()
    if source.stat().st_size != len(raw):
        raise ValueError("target task changed while reading")
    with zipfile.ZipFile(io.BytesIO(raw)) as probe:
        names = tuple(probe.namelist())
    if names not in (_ZERO_SHOT_MEMBERS, _FEW_SHOT_MEMBERS):
        raise ValueError("target task member inventory is invalid")
    entries = _safe_zip_entries(raw, names, label="target task ZIP")
    task = TargetTaskDocument.model_validate_json(entries["task.json"])
    expected_names = _FEW_SHOT_MEMBERS if task.mode == "few_shot" else _ZERO_SHOT_MEMBERS
    if names != expected_names:
        raise ValueError("target task mode disagrees with member inventory")
    descriptors = (task.inference, task.target_receipt, task.labels, task.label_receipt)
    for descriptor in (item for item in descriptors if item is not None):
        value = entries[descriptor.name]
        if len(value) != descriptor.bytes or _sha256_bytes(value) != descriptor.sha256:
            raise ValueError("target task member digest or size mismatch")
    manifest, node_ids, fused_edges, degrees, strata = _verify_inner(entries["inference.zip"])
    receipt = TargetDomainReceipt.model_validate_json(entries["target-receipt.json"])
    if (
        receipt.task_id != task.task_id
        or receipt.inference_sha256 != task.inference.sha256
        or receipt.node_count != task.node_count
        or manifest.nodeCount != task.node_count
        or receipt.fused_edge_count != task.fused_edge_count
        or fused_edges != task.fused_edge_count
        or receipt.modalities != task.modalities
        or tuple(manifest.modalities) != task.modalities
        or receipt.node_set_sha256 != canonical_sha256(sorted(node_ids))
    ):
        raise ValueError("target task receipt does not bind the inference graph")
    labels = None
    label_receipt = None
    if task.mode == "few_shot":
        labels = TargetLabelSetV2.model_validate_json(entries["labels.json"])
        label_receipt = TargetLabelReceipt.model_validate_json(entries["label-receipt.json"])
        if (
            labels.task_id != task.task_id
            or labels.inference_sha256 != task.inference.sha256
            or any(row.node_id not in node_ids for row in labels.labels)
            or any(
                row.fused_degree != degrees[row.node_id]
                or row.structural_stratum != strata[row.node_id]
                for row in labels.labels
            )
            or label_receipt.labels_sha256 != cast(TaskFileDescriptor, task.labels).sha256
            or label_receipt.target_receipt_hash != receipt.receipt_hash
            or label_receipt.eligibility_mask_sha256 != receipt.label_eligibility_mask_sha256
            or any(row.node_id not in label_receipt.eligible_node_ids for row in labels.labels)
        ):
            raise ValueError("detached label degree/stratum facts do not bind the target graph")
    return VerifiedTargetTask(source, task, receipt, labels, label_receipt)


def verify_governance_target_tasks(
    output_directory: str | Path,
    *,
    corpus_root: str | Path,
) -> tuple[VerifiedTargetTask, VerifiedTargetTask]:
    root = Path(output_directory).expanduser().resolve(strict=True)
    paths = _governance_catalog_paths(root)
    verified = cast(
        tuple[VerifiedTargetTask, VerifiedTargetTask],
        tuple(verify_target_task_bundle(path) for path in paths),
    )
    target_a, target_b = verified
    corpus = Path(corpus_root).expanduser().resolve(strict=True)
    corpus_index = load_corpus_index(corpus, verify_manifests=True)
    cuba = corpus_index.load_country(
        cast(CountryId, "cuba"), verify_hashes=False, verify_values=False
    )
    uae = corpus_index.load_country(
        cast(CountryId, "UAE"), verify_hashes=False, verify_values=False
    )
    cuba_mask_path = cuba.root / "arrays" / "split_full_0_test.npy"
    uae_mask_path = uae.root / "arrays" / "split_full_0_test.npy"
    uae_labels_path = uae.root / "arrays" / "labels.npy"
    expected_eligible_ids = tuple(
        sorted(
            _anonymous_node_id("UAE", int(node))
            for node in np.flatnonzero(uae.split("full-fold-0").test_mask)
        )
    )
    if (
        target_a.path.name != paths[0].name
        or target_a.task.task_id != "target-a"
        or target_a.task.mode != "zero_shot"
        or target_a.task.node_count != _TARGET_NODE_COUNT
        or not _MIN_FUSED_EDGES <= target_a.task.fused_edge_count <= _MAX_FUSED_EDGES
        or target_a.task.modalities != MODALITIES
        or target_a.receipt.country_id != "cuba"
        or target_a.receipt.source_content_hash != cuba.manifest.content_hash
        or target_a.receipt.source_manifest_sha256 != file_sha256(cuba.root / "manifest.json")
        or target_a.receipt.graph_population != "fold0_test"
        or target_a.receipt.graph_population_mask_sha256 != file_sha256(cuba_mask_path)
        or target_a.receipt.label_eligibility != "none"
        or target_a.receipt.connected is not True
        or target_a.labels is not None
        or target_a.label_receipt is not None
    ):
        raise ValueError("target-a does not satisfy the Cuba zero-shot governance role")
    labels = target_b.labels
    label_receipt = target_b.label_receipt
    if (
        target_b.path.name != paths[1].name
        or target_b.task.task_id != "target-b"
        or target_b.task.mode != "few_shot"
        or target_b.task.node_count != _TARGET_NODE_COUNT
        or not _MIN_FUSED_EDGES <= target_b.task.fused_edge_count <= _MAX_FUSED_EDGES
        or target_b.task.modalities != MODALITIES
        or target_b.receipt.country_id != "UAE"
        or target_b.receipt.source_content_hash != uae.manifest.content_hash
        or target_b.receipt.source_manifest_sha256 != file_sha256(uae.root / "manifest.json")
        or target_b.receipt.graph_population != "full"
        or target_b.receipt.label_eligibility != "fold0_test"
        or target_b.receipt.label_eligibility_mask_sha256 != file_sha256(uae_mask_path)
        or target_b.receipt.connected is not True
        or labels is None
        or label_receipt is None
        or label_receipt.source_labels_sha256 != file_sha256(uae_labels_path)
        or label_receipt.eligibility_mask_sha256 != file_sha256(uae_mask_path)
        or label_receipt.eligible_node_ids != expected_eligible_ids
        or len(labels.labels) != 16
        or (labels.positive_count, labels.negative_count) != (8, 8)
        or any(row.node_id not in label_receipt.eligible_node_ids for row in labels.labels)
        or any(
            sum(row.label == label and row.structural_stratum == stratum for row in labels.labels)
            != 2
            for label in ("positive", "negative")
            for stratum in range(4)
        )
    ):
        raise ValueError("target-b does not satisfy the UAE 16-label 8/8 governance role")
    expected_packages = dict(_generate_bytes(corpus))
    for item in verified:
        if item.path.read_bytes() != expected_packages[item.path.name]:
            raise ValueError(
                f"governance catalog {item.task.task_id} does not match the expected deterministic package"
            )
    return verified


def reset_governance_target_tasks(output_directory: str | Path) -> None:
    """Remove only the canonical catalog, packages, and exact staging remnants."""

    requested = Path(output_directory).expanduser().absolute()
    if not requested.exists():
        return
    if requested.is_symlink() or requested.is_junction():
        raise ValueError("governance adaptation output directory cannot be a reparse point")
    root = requested.resolve(strict=True)
    package_paths: list[Path] = []
    for name in _GOVERNANCE_PACKAGE_NAMES:
        path = root / name
        if path.exists():
            if not path.is_file() or path.is_symlink():
                raise ValueError("governance adaptation package inventory is unsafe")
            package_paths.append(path)
    catalog_path = root / _GOVERNANCE_CATALOG_NAME
    if catalog_path.exists() and (not catalog_path.is_file() or catalog_path.is_symlink()):
        raise ValueError("governance target catalog is unsafe")
    if catalog_path.exists():
        _governance_catalog_paths(root)
    staging_files: list[Path] = []
    for name in _GOVERNANCE_PACKAGE_NAMES:
        for path in root.glob(f".{name}.*.tmp"):
            if not path.is_file() or path.is_symlink():
                raise ValueError("governance adaptation staging artifact is unsafe")
            staging_files.append(path)
    for path in root.glob(f".{_GOVERNANCE_CATALOG_NAME}.*.tmp"):
        if not path.is_file() or path.is_symlink():
            raise ValueError("governance catalog staging artifact is unsafe")
        staging_files.append(path)

    catalog_root = root / _GOVERNANCE_CATALOG_ROOT
    generated_directories: list[Path] = []
    generated_files: list[Path] = []
    if catalog_root.exists():
        if not catalog_root.is_dir() or catalog_root.is_symlink() or catalog_root.is_junction():
            raise ValueError("governance target catalog root is unsafe")
        for directory in catalog_root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or directory.is_junction():
                raise ValueError("governance target catalog generation is unsafe")
            if re.fullmatch(r"[0-9a-f]{64}|\.[0-9a-f]{64}\.\d+\.tmp", directory.name) is None:
                raise ValueError("governance target catalog generation name is unsafe")
            children = tuple(directory.iterdir())
            if (
                len(children) > len(_GOVERNANCE_PACKAGE_NAMES)
                or any(
                    child.name not in _GOVERNANCE_PACKAGE_NAMES
                    or not child.is_file()
                    or child.is_symlink()
                    for child in children
                )
                or not {child.name for child in children}.issubset(_GOVERNANCE_PACKAGE_NAMES)
            ):
                raise ValueError("governance target catalog generation inventory is unsafe")
            generated_directories.append(directory)
            generated_files.extend(children)

    # Nothing is removed until the complete reserved inventory above is known safe.
    catalog_path.unlink(missing_ok=True)
    for path in (*package_paths, *staging_files, *generated_files):
        path.unlink()
    for path in generated_files:
        if path.exists():
            raise OSError(f"governance target package could not be removed: {path}")
    for directory in reversed(generated_directories):
        directory.rmdir()
    if catalog_root.exists():
        catalog_root.rmdir()


__all__ = [
    "GOVERNANCE_TARGET_CATALOG_SCHEMA_VERSION",
    "LABEL_SET_SCHEMA_VERSION",
    "TASK_BUNDLE_SCHEMA_VERSION",
    "GeneratedGovernanceTasks",
    "TargetDomainReceipt",
    "TargetLabelSetV2",
    "TargetTaskDocument",
    "VerifiedTargetTask",
    "generate_governance_target_tasks",
    "reset_governance_target_tasks",
    "verify_governance_target_tasks",
    "verify_target_task_bundle",
]
