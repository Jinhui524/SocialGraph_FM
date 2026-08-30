"""Shared limits, in-memory records, and canonical identity helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

import numpy as np

from ..dataset_schemas import (
    DatasetInspection,
    DatasetIssue,
    DatasetPreparationSpec,
    DatasetProfile,
    GraphVersionTargetDomainEnvelope,
    SourceFileDigest,
)



MAX_NODES = 2_000_000
MAX_EDGES = 5_000_000
MAX_ARRAY_ELEMENTS = 20_000_000
MAX_TRUSTED_ARRAY_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARRAYS = 64
PREVIEW_NODES = 3_000
PREVIEW_EDGES = 12_000
MAX_STORED_INSPECTIONS = 128
MAX_STORED_ARTIFACTS = 128
_SPLIT_KEYS = ("train_mask", "val_mask", "test_mask", "train_idx", "val_idx", "test_idx")
ArrayRole = Literal[
    "edge_index",
    "node_id_map",
    "feature",
    "label",
    "split",
    "variant",
    "auxiliary",
]


@dataclass(frozen=True)
class UploadedEntry:
    name: str
    data: bytes


@dataclass
class GraphPayload:
    node_count: int
    edge_index: np.ndarray
    features: np.ndarray | None = None
    feature_dimension: int | None = None
    labels: np.ndarray | None = None
    split_names: list[str] = field(default_factory=list)
    splits: dict[str, np.ndarray] = field(default_factory=dict)
    variant_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    node_ids: np.ndarray | None = None
    node_labels: np.ndarray | None = None
    node_types: np.ndarray | None = None
    node_attributes: np.ndarray | None = None
    edge_ids: np.ndarray | None = None
    edge_types: np.ndarray | None = None
    edge_weights: np.ndarray | None = None
    edge_timestamps: np.ndarray | None = None
    edge_directed: np.ndarray | None = None
    edge_attributes: np.ndarray | None = None
    node_identity_kind: Literal["source", "row_index"] = "row_index"
    directed: bool = False
    directedness: Literal["directed", "undirected", "mixed", "unspecified"] | None = None


@dataclass
class AdapterResult:
    detected_format: str
    status: Literal["accepted", "mapping_required", "conversion_required", "rejected"]
    profile: DatasetProfile | None
    issues: list[DatasetIssue]
    payload: GraphPayload | None = None
    dataset_candidates: list[str] = field(default_factory=list)
    dataset_name: str | None = None
    raw_manifest: dict[str, object] | None = None
    derived_manifest: dict[str, object] | None = None
    attachments: dict[str, bytes] = field(default_factory=dict)


@dataclass
class InspectionRecord:
    project_id: str
    response: DatasetInspection
    payload: GraphPayload | None
    checksum: str
    source_files: list[str]
    source_file_digests: list[SourceFileDigest]
    dataset_name: str | None = None
    raw_manifest: dict[str, object] | None = None
    derived_manifest: dict[str, object] | None = None
    attachments: dict[str, bytes] = field(default_factory=dict)
    retained_bytes: int = 0
    last_accessed_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class DatasetAdapter(Protocol):
    def matches(self, entries: dict[str, UploadedEntry]) -> bool: ...

    def inspect(self, entries: dict[str, UploadedEntry]) -> AdapterResult: ...


def _ecmascript_graph_number(value: float) -> str:
    """Render a finite binary64 value like ECMAScript ``JSON.stringify``."""

    if not math.isfinite(value):
        raise ValueError("canonical graph JSON forbids NaN and Infinity")
    if value == 0:
        return "0"

    negative = value < 0
    mantissa, marker, exponent_text = repr(abs(value)).lower().partition("e")
    exponent = int(exponent_text) if marker else 0
    whole, dot, fraction = mantissa.partition(".")
    raw_digits = whole + (fraction if dot else "")
    leading_zero_count = len(raw_digits) - len(raw_digits.lstrip("0"))
    digits = raw_digits[leading_zero_count:] or "0"
    decimal_position = len(whole) + exponent - leading_zero_count

    # ECMAScript selects fixed notation for [1e-6, 1e21), unlike Python repr.
    magnitude = abs(value)
    if 1e-6 <= magnitude < 1e21:
        if decimal_position <= 0:
            rendered = "0." + "0" * (-decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + "0" * (decimal_position - len(digits))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
    else:
        digits = digits.rstrip("0") or "0"
        coefficient = digits[0] + (("." + digits[1:]) if len(digits) > 1 else "")
        scientific_exponent = decimal_position - 1
        exponent_sign = "+" if scientific_exponent >= 0 else "-"
        rendered = f"{coefficient}e{exponent_sign}{abs(scientific_exponent)}"
    return ("-" if negative else "") + rendered


def _canonical_graph_json(value: object) -> str:
    """Serialize graph facts with code-point keys and ECMAScript number spelling."""

    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _ecmascript_graph_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_canonical_graph_json(item) for item in value) + "]"
    if isinstance(value, dict):
        entries = sorted(((str(key), item) for key, item in value.items()), key=lambda item: item[0])
        return "{" + ",".join(
            f"{_canonical_graph_json(key)}:{_canonical_graph_json(item)}"
            for key, item in entries
        ) + "}"
    raise TypeError(f"unsupported canonical graph value: {type(value).__name__}")


def _inspection_record_retained_bytes(record: InspectionRecord) -> int:
    """Conservative retained-memory budget for the process-local inspection cache."""

    size = len(record.project_id.encode("utf-8"))
    size += len(record.response.model_dump_json(by_alias=True).encode("utf-8"))
    size += len(record.checksum.encode("utf-8"))
    size += sum(len(value.encode("utf-8")) for value in record.source_files)
    size += sum(
        len(value.model_dump_json(by_alias=True).encode("utf-8"))
        for value in record.source_file_digests
    )
    if record.dataset_name:
        size += len(record.dataset_name.encode("utf-8"))
    for manifest in (record.raw_manifest, record.derived_manifest):
        if manifest is not None:
            size += len(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )
    size += sum(len(name.encode("utf-8")) + len(value) for name, value in record.attachments.items())
    if record.payload is not None:
        arrays: list[np.ndarray] = []
        for value in vars(record.payload).values():
            if isinstance(value, np.ndarray):
                arrays.append(value)
            elif isinstance(value, dict):
                arrays.extend(item for item in value.values() if isinstance(item, np.ndarray))
        seen: set[int] = set()
        for value in arrays:
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            size += int(value.nbytes)
    return size


def graph_fact_hash_v1(envelope: GraphVersionTargetDomainEnvelope) -> str:
    """Canonical hash of browser graph facts, independent from row ordering.

    The matching JavaScript implementation must UTF-8 encode the same canonical
    JSON and prefix it with ``socialgraph-fm-graph-fact-v1\0``.
    """

    facts = {
        "directedness": envelope.directedness,
        "nodes": sorted(
            (
                node.model_dump(mode="json", by_alias=True)
                for node in envelope.nodes
            ),
            key=lambda item: str(item["id"]),
        ),
        "edges": sorted(
            (
                edge.model_dump(mode="json", by_alias=True)
                for edge in envelope.edges
            ),
            key=lambda item: (
                str(item["id"]),
                str(item["source"]),
                str(item["target"]),
            ),
        ),
    }
    serialized = _canonical_graph_json(facts).encode("utf-8")
    return hashlib.sha256(b"socialgraph-fm-graph-fact-v1\x00" + serialized).hexdigest()


def dataset_preparation_hash_v1(spec: DatasetPreparationSpec) -> str:
    value = spec.model_dump(mode="json", by_alias=True)
    for key in ("featureAttributes", "excludedAttributes"):
        value[key] = sorted(set(value[key]))
    governance = value["governance"]
    governance["attributeAllowlist"] = sorted(set(governance["attributeAllowlist"]))
    governance["excludedAttributes"] = sorted(set(governance["excludedAttributes"]))
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(b"socialgraph-fm-dataset-preparation-v1\x00" + encoded).hexdigest()


def _issue(
    code: str,
    message: str,
    *,
    file: str | None = None,
    severity: Literal["warning", "error"] = "error",
) -> DatasetIssue:
    return DatasetIssue(severity=severity, code=code, message=message, file=file)
