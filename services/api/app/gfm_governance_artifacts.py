"""Strict, torch-free intake for SocialGraph-FM Governance inference bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
import threading
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from pydantic import ValidationError

from .gfm_client import GfmProxyError, _reject_link_components
from .gfm_hashing import canonical_sha256
from .gfm_governance_schemas import (
    GOVERNANCE_MODALITIES,
    GOVERNANCE_SCHEMA_VERSION,
    GovernanceArtifactList,
    GovernanceArtifactReceipt,
    GovernanceInputManifest,
)

_EXPECTED_MEMBERS = frozenset(
    {"manifest.json", "nodes.csv", "relations.csv", "features.npz"}
)
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_NPZ_EXPANDED_BYTES = 64 * 1024 * 1024
_MAX_TEXT_FIELD_BYTES = 1_024
_LOCK = threading.RLock()


class GovernanceBundleError(GfmProxyError):
    pass


def _fail(status_code: int, code: str) -> None:
    raise GovernanceBundleError(status_code, code)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_zip_members(payload: bytes, *, max_expanded_bytes: int) -> dict[str, bytes]:
    if not payload.startswith(b"PK"):
        _fail(400, "GOVERNANCE_ZIP_REQUIRED")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise GovernanceBundleError(400, "GOVERNANCE_ZIP_INVALID") from error
    result: dict[str, bytes] = {}
    expanded = 0
    with archive:
        members = archive.infolist()
        if len(members) != len(_EXPECTED_MEMBERS):
            _fail(400, "GOVERNANCE_ZIP_MEMBERS_INVALID")
        for member in members:
            path = PurePosixPath(member.filename.replace("\\", "/"))
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 1
                or member.is_dir()
                or "\x00" in member.filename
                or member.flag_bits & 0x1
                or member.file_size < 1
            ):
                _fail(400, "GOVERNANCE_ZIP_PATH_UNSAFE")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                _fail(400, "GOVERNANCE_ZIP_LINK_REJECTED")
            name = path.as_posix()
            if name in result or name not in _EXPECTED_MEMBERS:
                _fail(400, "GOVERNANCE_ZIP_MEMBERS_INVALID")
            expanded += member.file_size
            if expanded > max_expanded_bytes:
                _fail(413, "GOVERNANCE_ZIP_EXPANDED_TOO_LARGE")
            if member.compress_size == 0 or member.file_size > member.compress_size * 200:
                _fail(413, "GOVERNANCE_ZIP_RATIO_INVALID")
            try:
                result[name] = archive.read(member)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise GovernanceBundleError(400, "GOVERNANCE_ZIP_INVALID") from error
    if set(result) != _EXPECTED_MEMBERS:
        _fail(400, "GOVERNANCE_ZIP_MEMBERS_INVALID")
    return result


def _parse_manifest(data: bytes, entries: dict[str, bytes]) -> GovernanceInputManifest:
    if not data or len(data) > _MAX_MANIFEST_BYTES:
        _fail(400, "GOVERNANCE_MANIFEST_SIZE_INVALID")
    try:
        raw = json.loads(data.decode("utf-8"))
        manifest = GovernanceInputManifest.model_validate(raw)
    except (UnicodeDecodeError, ValueError, ValidationError) as error:
        raise GovernanceBundleError(400, "GOVERNANCE_MANIFEST_INVALID") from error
    for name, descriptor in manifest.files.items():
        if descriptor.bytes != len(entries[name]) or descriptor.sha256 != _digest(entries[name]):
            _fail(409, "GOVERNANCE_MANIFEST_FILE_HASH_MISMATCH")
    return manifest


def _read_csv(
    data: bytes,
    *,
    expected_headers: set[tuple[str, ...]],
    max_rows: int,
    code: str,
) -> list[dict[str, str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise GovernanceBundleError(400, code) from error
    if "\x00" in text:
        _fail(400, code)
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    headers = tuple(reader.fieldnames or ())
    if headers not in expected_headers or len(set(headers)) != len(headers):
        _fail(400, code)
    rows: list[dict[str, str]] = []
    try:
        for raw in reader:
            if len(rows) >= max_rows:
                _fail(413, code)
            if None in raw or any(value is None for value in raw.values()):
                _fail(400, code)
            row = {key: value for key, value in raw.items() if key is not None and value is not None}
            if any(len(value.encode("utf-8")) > _MAX_TEXT_FIELD_BYTES for value in row.values()):
                _fail(400, code)
            rows.append(row)
    except csv.Error as error:
        raise GovernanceBundleError(400, code) from error
    return rows


def _validate_node_id(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or len(value) > 128
        or len(value.encode("utf-8")) > 512
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        _fail(400, "GOVERNANCE_NODE_ID_INVALID")
    return value


def _parse_nodes(data: bytes, manifest: GovernanceInputManifest) -> tuple[list[str], list[str]]:
    rows = _read_csv(
        data,
        expected_headers={("node_id",), ("node_id", "display_name")},
        max_rows=10_000,
        code="GOVERNANCE_NODES_CSV_INVALID",
    )
    if len(rows) != manifest.node_count:
        _fail(409, "GOVERNANCE_NODE_COUNT_MISMATCH")
    ids: list[str] = []
    labels: list[str] = []
    seen: set[str] = set()
    for row in rows:
        node_id = _validate_node_id(row["node_id"])
        if node_id in seen:
            _fail(409, "GOVERNANCE_NODE_ID_DUPLICATE")
        seen.add(node_id)
        label = row.get("display_name", "") or node_id
        if len(label) > 256 or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in label
        ):
            _fail(400, "GOVERNANCE_NODE_LABEL_INVALID")
        ids.append(node_id)
        labels.append(label)
    return ids, labels


def _validate_npz_container(data: bytes) -> None:
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise GovernanceBundleError(400, "GOVERNANCE_FEATURES_NPZ_INVALID") from error
    total = 0
    with archive:
        members = archive.infolist()
        if {member.filename for member in members} != {
            "node_ids.npy",
            "text_features.npy",
        } or len(members) != 2:
            _fail(400, "GOVERNANCE_FEATURES_ARRAYS_INVALID")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
                _fail(400, "GOVERNANCE_FEATURES_NPZ_INVALID")
            if member.flag_bits & 0x1 or member.file_size < 1:
                _fail(400, "GOVERNANCE_FEATURES_NPZ_INVALID")
            total += member.file_size
            if total > _MAX_NPZ_EXPANDED_BYTES:
                _fail(413, "GOVERNANCE_FEATURES_EXPANDED_TOO_LARGE")


def _parse_features(data: bytes, node_ids: list[str]) -> None:
    _validate_npz_container(data)
    try:
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            if set(arrays.files) != {"node_ids", "text_features"}:
                _fail(400, "GOVERNANCE_FEATURES_ARRAYS_INVALID")
            array_ids = np.asarray(arrays["node_ids"])
            features = np.asarray(arrays["text_features"])
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise GovernanceBundleError(400, "GOVERNANCE_FEATURES_NPZ_INVALID") from error
    if array_ids.ndim != 1 or array_ids.shape[0] != len(node_ids):
        _fail(409, "GOVERNANCE_FEATURE_NODE_ALIGNMENT_MISMATCH")
    if array_ids.dtype.kind not in {"U", "S"} or array_ids.dtype.hasobject:
        _fail(400, "GOVERNANCE_FEATURE_NODE_IDS_DTYPE_INVALID")
    try:
        decoded = [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in array_ids.tolist()
        ]
    except UnicodeDecodeError as error:
        raise GovernanceBundleError(
            400, "GOVERNANCE_FEATURE_NODE_IDS_UTF8_INVALID"
        ) from error
    if decoded != node_ids:
        _fail(409, "GOVERNANCE_FEATURE_NODE_ALIGNMENT_MISMATCH")
    if features.dtype != np.dtype("float32") or features.shape != (len(node_ids), 768):
        _fail(400, "GOVERNANCE_FEATURE_SHAPE_INVALID")
    if not np.isfinite(features).all():
        _fail(400, "GOVERNANCE_FEATURE_NONFINITE")


def _parse_relations(
    data: bytes,
    manifest: GovernanceInputManifest,
    node_ids: list[str],
    *,
    clean_self_loops: bool,
) -> tuple[list[tuple[int, int]], int, int, tuple[str, ...]]:
    rows = _read_csv(
        data,
        expected_headers={("source", "target", "modality", "weight")},
        max_rows=500_000,
        code="GOVERNANCE_RELATIONS_CSV_INVALID",
    )
    if len(rows) != manifest.relation_row_count:
        _fail(409, "GOVERNANCE_RELATION_COUNT_MISMATCH")
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    seen: set[tuple[int, int, str]] = set()
    fused: set[tuple[int, int]] = set()
    modalities: set[str] = set()
    self_loops = 0
    for row in rows:
        source_id = row["source"]
        target_id = row["target"]
        if source_id not in index or target_id not in index:
            _fail(409, "GOVERNANCE_RELATION_NODE_MISSING")
        source = index[source_id]
        target = index[target_id]
        modality = row["modality"]
        if modality not in GOVERNANCE_MODALITIES:
            _fail(400, "GOVERNANCE_RELATION_MODALITY_INVALID")
        try:
            weight = float(row["weight"])
        except ValueError as error:
            raise GovernanceBundleError(400, "GOVERNANCE_RELATION_WEIGHT_INVALID") from error
        if not math.isfinite(weight) or weight < 0:
            _fail(400, "GOVERNANCE_RELATION_WEIGHT_INVALID")
        left, right = sorted((source, target))
        key = (left, right, modality)
        if key in seen:
            _fail(409, "GOVERNANCE_RELATION_DUPLICATE")
        seen.add(key)
        if source == target:
            self_loops += 1
            continue
        fused.add((left, right))
        modalities.add(modality)
    if self_loops and not clean_self_loops:
        _fail(409, "GOVERNANCE_SELF_LOOP_CONFIRMATION_REQUIRED")
    if not fused:
        _fail(400, "GOVERNANCE_RELATIONS_EMPTY_AFTER_CLEANING")
    if set(manifest.modalities) != modalities:
        _fail(409, "GOVERNANCE_MODALITIES_MISMATCH")
    return sorted(fused), len(seen) - self_loops, self_loops, tuple(
        value for value in GOVERNANCE_MODALITIES if value in modalities
    )


def _graph_version_hash(node_ids: list[str], fused_edges: list[tuple[int, int]]) -> str:
    digest = hashlib.sha256(b"socialgraph-fm.governance-graph-v2\x00")
    for node_id in node_ids:
        encoded = node_id.encode("utf-8")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
    for source, target in fused_edges:
        digest.update(struct.pack(">QQ", source, target))
    return digest.hexdigest()


def inspect_governance_bundle(
    payload: bytes,
    *,
    clean_self_loops: bool,
    max_expanded_bytes: int,
) -> tuple[GovernanceInputManifest, dict[str, Any]]:
    entries = _safe_zip_members(payload, max_expanded_bytes=max_expanded_bytes)
    manifest = _parse_manifest(entries["manifest.json"], entries)
    node_ids, _ = _parse_nodes(entries["nodes.csv"], manifest)
    _parse_features(entries["features.npz"], node_ids)
    fused_edges, relation_count, self_loops, modalities = _parse_relations(
        entries["relations.csv"],
        manifest,
        node_ids,
        clean_self_loops=clean_self_loops,
    )
    file_digests = {name: _digest(entries[name]) for name in sorted(manifest.files)}
    content_identity = {
        "schemaVersion": manifest.schema_version,
        "manifestHash": _digest(entries["manifest.json"]),
        "fileDigests": file_digests,
        "cleanSelfLoops": clean_self_loops,
        "selfLoopsRemoved": self_loops,
    }
    dataset_content_hash = canonical_sha256(content_identity)
    return manifest, {
        "artifactId": f"governance-artifact-{dataset_content_hash[:32]}",
        "datasetContentHash": dataset_content_hash,
        "graphVersionHash": _graph_version_hash(node_ids, fused_edges),
        "bundleSha256": _digest(payload),
        "manifestSha256": _digest(entries["manifest.json"]),
        "nodeCount": len(node_ids),
        "relationRowCount": relation_count,
        "selfLoopsRemoved": self_loops,
        "modalities": list(modalities),
    }


class GovernanceArtifactInbox:
    """Immutable shared inbox; only GFM turns receipts into serving artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)
        self.incoming_root = self.root / "incoming"
        self.incoming_root.mkdir(parents=True, exist_ok=True)
        if self.incoming_root.is_symlink():
            raise ValueError("SocialGraph-FM Governance incoming root cannot be a link")

    def _directory(self, artifact_id: str) -> Path:
        if re.fullmatch(r"governance-artifact-[0-9a-f]{32}", artifact_id) is None:
            _fail(404, "GOVERNANCE_ARTIFACT_NOT_FOUND")
        return self.incoming_root / artifact_id

    def commit(
        self,
        payload: bytes,
        *,
        clean_self_loops: bool,
        max_expanded_bytes: int,
    ) -> GovernanceArtifactReceipt:
        manifest, inspected = inspect_governance_bundle(
            payload,
            clean_self_loops=clean_self_loops,
            max_expanded_bytes=max_expanded_bytes,
        )
        created_at = datetime.now(UTC).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )
        receipt_payload: dict[str, Any] = {
            "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
            "artifactId": inspected["artifactId"],
            "datasetId": manifest.dataset_id,
            "displayName": manifest.display_name,
            "datasetContentHash": inspected["datasetContentHash"],
            "graphVersionHash": inspected["graphVersionHash"],
            "bundleSha256": inspected["bundleSha256"],
            "manifestSha256": inspected["manifestSha256"],
            "nodeCount": inspected["nodeCount"],
            "relationRowCount": inspected["relationRowCount"],
            "selfLoopsRemoved": inspected["selfLoopsRemoved"],
            "cleanSelfLoops": clean_self_loops,
            "modalities": inspected["modalities"],
            "compatibility": "compatible",
            "createdAt": created_at,
        }
        receipt_payload["artifactHash"] = canonical_sha256(receipt_payload)
        receipt = GovernanceArtifactReceipt.model_validate(receipt_payload)
        destination = self._directory(receipt.artifact_id)
        with _LOCK:
            if destination.exists():
                existing = self.get(receipt.artifact_id)
                bundle_path = destination / "bundle.zip"
                if (
                    existing.dataset_content_hash != receipt.dataset_content_hash
                    or not bundle_path.is_file()
                    or _digest(bundle_path.read_bytes()) != existing.bundle_sha256
                ):
                    _fail(409, "GOVERNANCE_ARTIFACT_ID_CONFLICT")
                return existing
            staging = self.incoming_root / f".stage-{uuid.uuid4().hex}"
            try:
                staging.mkdir(parents=False, exist_ok=False)
                self._write_file(staging / "bundle.zip", payload)
                self._write_file(
                    staging / "receipt.json",
                    receipt.model_dump_json(by_alias=True, indent=2).encode("utf-8"),
                )
                os.replace(staging, destination)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return receipt

    @staticmethod
    def _write_file(path: Path, payload: bytes) -> None:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

    def get(self, artifact_id: str) -> GovernanceArtifactReceipt:
        directory = self._directory(artifact_id)
        try:
            _reject_link_components(directory)
        except ValueError as error:
            raise GovernanceBundleError(404, "GOVERNANCE_ARTIFACT_NOT_FOUND") from error
        path = directory / "receipt.json"
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_MANIFEST_BYTES:
                raise FileNotFoundError
            receipt = GovernanceArtifactReceipt.model_validate_json(path.read_bytes())
        except FileNotFoundError as error:
            raise GovernanceBundleError(404, "GOVERNANCE_ARTIFACT_NOT_FOUND") from error
        except (OSError, ValidationError, ValueError) as error:
            raise GovernanceBundleError(502, "GOVERNANCE_ARTIFACT_RECEIPT_INVALID") from error
        if receipt.artifact_id != artifact_id:
            _fail(502, "GOVERNANCE_ARTIFACT_RECEIPT_INVALID")
        return receipt

    def list(self, *, offset: int, limit: int) -> GovernanceArtifactList:
        receipts: list[GovernanceArtifactReceipt] = []
        with _LOCK:
            for path in self.incoming_root.iterdir():
                if not path.is_dir() or path.is_symlink() or not path.name.startswith("governance-artifact-"):
                    continue
                try:
                    receipts.append(self.get(path.name))
                except GfmProxyError:
                    continue
        receipts.sort(key=lambda item: (item.created_at, item.artifact_id), reverse=True)
        return GovernanceArtifactList.model_validate(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "items": receipts[offset : offset + limit],
                "total": len(receipts),
                "offset": offset,
                "limit": limit,
            }
        )


__all__ = [
    "GovernanceArtifactInbox",
    "GovernanceBundleError",
    "inspect_governance_bundle",
]
