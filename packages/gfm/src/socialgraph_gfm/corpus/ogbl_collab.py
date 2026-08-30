"""Trusted fetch, safe package ingestion and formal validation for ``ogbl-collab``.

Only :func:`fetch_ogbl_collab` enters the legacy OGB/PyG pickle boundary.  The
resulting package and all later artifacts contain JSON and NumPy arrays only;
every later load uses ``allow_pickle=False`` and revalidates the complete corpus.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
import sys
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping

import numpy as np
from pydantic import ValidationError

from ..canonical import canonical_json, canonical_sha256, file_sha256
from ..contracts import (
    CorpusArrayManifest,
    FormalCorpusManifest,
    TemporalLinkProtocolManifest,
)
from ..errors import ContractViolation
from ..runtime import artifact_root, prepare_runtime_layout, require_storage_reserve

CORPUS_ID = "ogbl-collab"
LICENSE_ID = "ODC-BY-1.0"
LICENSE_URL = "https://ogb.stanford.edu/docs/linkprop/#ogbl-collab"
ATTRIBUTION = "Open Graph Benchmark: ogbl-collab"
OGB_VERSION = "1.3.6"
PACKAGE_RELATIVE_PATH = Path("datasets/packages/ogbl-collab.sgfm.zip")
ARTIFACT_RELATIVE_PATH = Path("datasets/processed/ogbl-collab-v1.npz")
MANIFEST_RELATIVE_PATH = Path("datasets/manifests/ogbl-collab.json")
LICENSE_RECEIPT_RELATIVE_PATH = Path(
    "datasets/manifests/ogbl-collab-license-acceptance.json"
)
PACKAGE_GRAPH_PATH = "datasets/ogbl-collab/graph.npz"
PACKAGE_ENTRIES = frozenset({"manifest.json", PACKAGE_GRAPH_PATH})

MAX_OUTER_ENTRIES = 2
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_PACKAGE_UNCOMPRESSED_BYTES = 768 * 1024 * 1024
MAX_NPZ_UNCOMPRESSED_BYTES = 640 * 1024 * 1024
MAX_COMPRESSION_RATIO = 2_000
COPY_CHUNK_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True)
class CorpusExpectation:
    node_count: int
    feature_dim: int
    message_edge_count: int
    train_size: int
    validation_size: int
    test_size: int
    validation_negative_size: int
    test_negative_size: int


OFFICIAL_EXPECTATION = CorpusExpectation(
    node_count=235_868,
    feature_dim=128,
    message_edge_count=2_358_104,
    train_size=1_179_052,
    validation_size=60_084,
    test_size=46_329,
    validation_negative_size=100_000,
    test_negative_size=100_000,
)

ENHANCED_SOURCE_ARRAY_KEYS = frozenset(
    {
        "directed",
        "edge_index",
        "edge_timestamp",
        "edge_weight",
        "node_id_map",
        "num_nodes",
        "variant_test_negative",
        "variant_test_positive",
        "variant_test_weight",
        "variant_test_year",
        "variant_train_positive",
        "variant_train_weight",
        "variant_train_year",
        "variant_validation_negative",
        "variant_validation_positive",
        "variant_validation_weight",
        "variant_validation_year",
        "x",
    }
)
CORE_SOURCE_ARRAY_KEYS = ENHANCED_SOURCE_ARRAY_KEYS - {
    "variant_test_weight",
    "variant_test_year",
    "variant_train_weight",
    "variant_train_year",
    "variant_validation_weight",
    "variant_validation_year",
}
SOURCE_ARRAY_SCHEMAS = (CORE_SOURCE_ARRAY_KEYS, ENHANCED_SOURCE_ARRAY_KEYS)
DERIVED_ARRAY_KEYS = frozenset(
    {
        "strict_test_message_edge_index",
        "strict_test_message_edge_timestamp",
        "strict_train_message_edge_index",
        "strict_train_message_edge_timestamp",
        "strict_train_positive",
        "strict_validation_message_edge_index",
        "strict_validation_message_edge_timestamp",
    }
)
CORE_PROCESSED_ARRAY_KEYS = CORE_SOURCE_ARRAY_KEYS | DERIVED_ARRAY_KEYS
ENHANCED_PROCESSED_ARRAY_KEYS = ENHANCED_SOURCE_ARRAY_KEYS | DERIVED_ARRAY_KEYS
PROCESSED_ARRAY_SCHEMAS = (CORE_PROCESSED_ARRAY_KEYS, ENHANCED_PROCESSED_ARRAY_KEYS)


def corpus_manifest_path(
    root: str | Path | None,
    corpus_id: str = CORPUS_ID,
) -> Path:
    """Return the fixed manifest path without creating filesystem state."""

    if corpus_id != CORPUS_ID:
        raise ContractViolation(f"Unsupported formal corpus: {corpus_id}")
    return artifact_root(root) / MANIFEST_RELATIVE_PATH


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"{CORPUS_ID}: {message}")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_no_duplicates(payload: bytes, *, label: str) -> dict[str, Any]:
    if len(payload) > MAX_MANIFEST_BYTES:
        raise _fail(f"{label} exceeds the {MAX_MANIFEST_BYTES}-byte limit")

    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise _fail(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise _fail(f"{label} contains forbidden JSON constant {value}")

    try:
        decoded = payload.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except UnicodeDecodeError as exc:
        raise _fail(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise _fail(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise _fail(f"{label} must be a JSON object")
    return value


def _validate_member_name(name: str, *, label: str) -> None:
    if not name or "\\" in name or "\x00" in name or ":" in name:
        raise _fail(f"{label} contains unsafe archive member {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise _fail(f"{label} contains path traversal member {name!r}")
    if path.as_posix() != name:
        raise _fail(f"{label} contains non-canonical archive member {name!r}")


def _validate_zip_infos(
    archive: zipfile.ZipFile,
    *,
    expected_names: frozenset[str],
    max_entries: int,
    max_uncompressed_bytes: int,
    label: str,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > max_entries:
        raise _fail(f"{label} contains too many entries")
    by_name: dict[str, zipfile.ZipInfo] = {}
    total = 0
    for info in infos:
        _validate_member_name(info.filename, label=label)
        if info.filename in by_name:
            raise _fail(f"{label} contains duplicate member {info.filename!r}")
        if info.is_dir():
            raise _fail(f"{label} contains an unexpected directory entry")
        unix_mode = (info.external_attr >> 16) & 0o170000
        if unix_mode == stat.S_IFLNK:
            raise _fail(f"{label} contains a symbolic-link entry")
        if info.flag_bits & 0x1:
            raise _fail(f"{label} contains an encrypted entry")
        total += info.file_size
        if total > max_uncompressed_bytes:
            raise _fail(f"{label} exceeds the bounded uncompressed size")
        if info.file_size and info.file_size / max(info.compress_size, 1) > MAX_COMPRESSION_RATIO:
            raise _fail(f"{label} contains a suspicious compression ratio")
        by_name[info.filename] = info
    names = frozenset(by_name)
    if names != expected_names:
        missing = sorted(expected_names - names)
        extra = sorted(names - expected_names)
        raise _fail(f"{label} entry whitelist mismatch; missing={missing}, extra={extra}")
    return by_name


def _copy_bounded(source: Any, destination: Path, *, limit: int) -> None:
    copied = 0
    with destination.open("xb") as output:
        while True:
            chunk = source.read(COPY_CHUNK_BYTES)
            if not chunk:
                break
            copied += len(chunk)
            if copied > limit:
                raise _fail("archive member exceeded its declared safety limit")
            output.write(chunk)


def _npz_member_names(keys: frozenset[str]) -> frozenset[str]:
    return frozenset(f"{name}.npy" for name in keys)


def _load_npz_safely(path: Path, *, expected_keys: frozenset[str]) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(path) as archive:
            _validate_zip_infos(
                archive,
                expected_names=_npz_member_names(expected_keys),
                max_entries=len(expected_keys),
                max_uncompressed_bytes=MAX_NPZ_UNCOMPRESSED_BYTES,
                label="NPZ artifact",
            )
    except zipfile.BadZipFile as exc:
        raise _fail("graph artifact is not a valid NPZ/ZIP") from exc

    try:
        with np.load(path, allow_pickle=False) as loaded:
            if frozenset(loaded.files) != expected_keys:
                raise _fail("NPZ array whitelist changed between inspection and load")
            arrays = {
                name: np.array(loaded[name], copy=True, order="C", subok=False)
                for name in sorted(expected_keys)
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise _fail("NPZ contains an unreadable or pickle/object array") from exc
    for name, array in arrays.items():
        if array.dtype.hasobject:
            raise _fail(f"array {name} uses the forbidden object dtype")
    return arrays


def _load_package_npz_safely(path: Path) -> dict[str, np.ndarray]:
    """Accept the API v1 core package or the lossless v2 package, and nothing else."""

    enhanced_error: ContractViolation | None = None
    try:
        return _load_npz_safely(path, expected_keys=ENHANCED_SOURCE_ARRAY_KEYS)
    except ContractViolation as exc:
        enhanced_error = exc
    try:
        return _load_npz_safely(path, expected_keys=CORE_SOURCE_ARRAY_KEYS)
    except ContractViolation as exc:
        raise _fail(
            "NPZ does not match the core or lossless ogbl-collab array whitelist; "
            f"enhanced={enhanced_error}; core={exc}"
        ) from exc


def _validate_package_manifest(value: Mapping[str, Any]) -> str:
    if value.get("schemaVersion") != "socialgraph-fm-dataset-package/1.0":
        raise _fail("package manifest schemaVersion is unsupported")
    fingerprint = value.get("sourceFingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise _fail("package sourceFingerprint is missing or invalid")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise _fail("package sourceFingerprint is not hexadecimal") from exc
    if value.get("skipped") != []:
        raise _fail("formal package must not contain skipped datasets")
    datasets = value.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise _fail("formal package must contain exactly one dataset")
    item = datasets[0]
    if not isinstance(item, dict):
        raise _fail("package dataset descriptor must be an object")
    required = {
        "name": CORPUS_ID,
        "path": PACKAGE_GRAPH_PATH,
        "sourceFormat": "trusted_local_ogb",
        "datasetRole": "benchmark",
        "splitKind": "official",
        "directed": False,
    }
    for key, expected in required.items():
        if item.get(key) != expected:
            raise _fail(f"package dataset field {key} must be {expected!r}")
    policy = item.get("licensePolicy")
    if not isinstance(policy, dict):
        raise _fail("package licensePolicy is absent")
    if policy.get("status") != "verified" or policy.get("identifier") != LICENSE_ID:
        raise _fail("package license is not verified ODC-BY-1.0")
    if policy.get("sourceUrl") != LICENSE_URL or policy.get("attribution") != ATTRIBUTION:
        raise _fail("package license evidence does not match the official source")
    protocol = item.get("linkPredictionProtocol")
    if not isinstance(protocol, dict):
        raise _fail("package linkPredictionProtocol is absent")
    protocol_values = {
        "messagePassingEdgeArray": "edge_index",
        "trainPositiveArray": "variant_train_positive",
        "validationPositiveArray": "variant_validation_positive",
        "testPositiveArray": "variant_test_positive",
        "validationNegativeArray": "variant_validation_negative",
        "testNegativeArray": "variant_test_negative",
        "edgeYearArray": "edge_timestamp",
        "edgeWeightArray": "edge_weight",
        "trainYearMax": 2017,
        "validationYear": 2018,
        "testYear": 2019,
        "negativeSampler": "stored",
        "evaluator": "ogb.linkproppred.Evaluator(ogbl-collab)",
        "evaluatorVersion": OGB_VERSION,
    }
    for key, expected in protocol_values.items():
        if protocol.get(key) != expected:
            raise _fail(f"package temporal protocol field {key} is invalid")
    return fingerprint


def _load_safe_package(
    package: Path,
    *,
    temporary_directory: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any], str]:
    package = package.expanduser().resolve(strict=True)
    if not package.is_file() or package.is_symlink():
        raise _fail("package must be a regular, non-symlink file")
    try:
        with zipfile.ZipFile(package) as archive:
            infos = _validate_zip_infos(
                archive,
                expected_names=PACKAGE_ENTRIES,
                max_entries=MAX_OUTER_ENTRIES,
                max_uncompressed_bytes=MAX_PACKAGE_UNCOMPRESSED_BYTES,
                label="SGFM package",
            )
            manifest_payload = archive.read(infos["manifest.json"])
            package_manifest = _json_no_duplicates(
                manifest_payload,
                label="package manifest",
            )
            _validate_package_manifest(package_manifest)
            temporary_directory.mkdir(parents=True, exist_ok=True)
            graph_path = temporary_directory / f"ogbl-collab-{uuid.uuid4().hex}.npz"
            try:
                with archive.open(infos[PACKAGE_GRAPH_PATH]) as source:
                    _copy_bounded(
                        source,
                        graph_path,
                        limit=MAX_NPZ_UNCOMPRESSED_BYTES,
                    )
                arrays = _load_package_npz_safely(graph_path)
            finally:
                graph_path.unlink(missing_ok=True)
    except zipfile.BadZipFile as exc:
        raise _fail("package is not a valid ZIP archive") from exc
    return arrays, package_manifest, file_sha256(package)


def _shape(array: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if array.shape != expected:
        raise _fail(f"array {name} shape is {array.shape}, expected {expected}")


def _dtype(array: np.ndarray, expected: np.dtype[Any], name: str) -> None:
    if array.dtype != expected:
        raise _fail(f"array {name} dtype is {array.dtype}, expected {expected}")


def _validate_edge_array(
    array: np.ndarray,
    *,
    size: int,
    node_count: int,
    name: str,
    canonical: bool,
    allow_self_loops: bool = False,
) -> None:
    _shape(array, (2, size), name)
    _dtype(array, np.dtype(np.int64), name)
    if array.size and (int(array.min()) < 0 or int(array.max()) >= node_count):
        raise _fail(f"array {name} contains an out-of-bounds node id")
    if not allow_self_loops and np.any(array[0] == array[1]):
        raise _fail(f"array {name} contains a self-loop")
    comparison = array[0] > array[1] if allow_self_loops else array[0] >= array[1]
    if canonical and np.any(comparison):
        raise _fail(f"array {name} is not min/max canonicalized")


def _canonical_pairs(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack((np.minimum(array[0], array[1]), np.maximum(array[0], array[1])))
    )


def _pair_keys(array: np.ndarray, node_count: int) -> np.ndarray:
    canonical = _canonical_pairs(array)
    return canonical[0] * np.int64(node_count) + canonical[1]


def _validate_unique_pairs(array: np.ndarray, *, node_count: int, name: str) -> None:
    keys = np.sort(_pair_keys(array, node_count))
    if keys.size > 1 and np.any(keys[1:] == keys[:-1]):
        raise _fail(f"array {name} contains duplicate undirected pairs")


def _validate_symmetric_message_graph(arrays: Mapping[str, np.ndarray]) -> None:
    edges = arrays["edge_index"]
    years = arrays["edge_timestamp"]
    weights = arrays["edge_weight"]
    source, target = edges
    if np.any(source == target):
        raise _fail("message graph contains a self-loop")
    forward = source < target
    reverse = source > target
    if int(forward.sum()) != int(reverse.sum()):
        raise _fail("message graph is not bidirectionally symmetric")

    def records(mask: np.ndarray, *, swap: bool) -> np.ndarray:
        selected_source = target[mask] if swap else source[mask]
        selected_target = source[mask] if swap else target[mask]
        dtype = np.dtype(
            [("source", "<i8"), ("target", "<i8"), ("year", "<i2"), ("weight", "<f4")]
        )
        result = np.empty(selected_source.size, dtype=dtype)
        result["source"] = selected_source
        result["target"] = selected_target
        result["year"] = years[mask]
        result["weight"] = weights[mask]
        result.sort(order=("source", "target", "year", "weight"))
        return result

    if not np.array_equal(records(forward, swap=False), records(reverse, swap=True)):
        raise _fail("message graph reverse edges do not preserve year/weight multiplicity")


def _validate_train_alignment(arrays: Mapping[str, np.ndarray], node_count: int) -> None:
    source, target = arrays["edge_index"]
    forward = source < target
    message_pairs = np.stack((source[forward], target[forward]))
    train_pairs = arrays["variant_train_positive"]
    if not np.array_equal(
        np.sort(_pair_keys(message_pairs, node_count)),
        np.sort(_pair_keys(train_pairs, node_count)),
    ):
        raise _fail("train positives do not exactly match the undirected message graph")

    if "variant_train_year" not in arrays:
        return
    message_dtype = np.dtype(
        [("pair", "<i8"), ("year", "<i2"), ("weight", "<f4")]
    )
    message = np.empty(message_pairs.shape[1], dtype=message_dtype)
    message["pair"] = _pair_keys(message_pairs, node_count)
    message["year"] = arrays["edge_timestamp"][forward]
    message["weight"] = arrays["edge_weight"][forward]
    train = np.empty(train_pairs.shape[1], dtype=message_dtype)
    train["pair"] = _pair_keys(train_pairs, node_count)
    train["year"] = arrays["variant_train_year"]
    train["weight"] = arrays["variant_train_weight"]
    message.sort(order=("pair", "year", "weight"))
    train.sort(order=("pair", "year", "weight"))
    if not np.array_equal(message, train):
        raise _fail("message graph year/weight records do not match the train split")


def _validate_source_arrays(
    arrays: Mapping[str, np.ndarray],
    expectation: CorpusExpectation,
    *,
    canonical_supervision: bool,
) -> None:
    source_keys = frozenset(arrays)
    if source_keys not in SOURCE_ARRAY_SCHEMAS:
        raise _fail("source array whitelist is incomplete")
    has_split_metadata = source_keys == ENHANCED_SOURCE_ARRAY_KEYS
    n = expectation.node_count
    _shape(arrays["x"], (n, expectation.feature_dim), "x")
    _dtype(arrays["x"], np.dtype(np.float32), "x")
    if not bool(np.isfinite(arrays["x"]).all()):
        raise _fail("node features contain NaN or Infinity")

    _shape(arrays["num_nodes"], (), "num_nodes")
    _dtype(arrays["num_nodes"], np.dtype(np.int64), "num_nodes")
    if int(arrays["num_nodes"]) != n:
        raise _fail("num_nodes disagrees with the official dataset size")
    _shape(arrays["directed"], (), "directed")
    _dtype(arrays["directed"], np.dtype(np.bool_), "directed")
    if bool(arrays["directed"]):
        raise _fail("ogbl-collab must be undirected")
    _shape(arrays["node_id_map"], (n,), "node_id_map")
    if arrays["node_id_map"].dtype.kind != "U" or arrays["node_id_map"].dtype.hasobject:
        raise _fail("node_id_map must be a fixed-width Unicode array")
    width = 50_000
    for start in range(0, n, width):
        stop = min(start + width, n)
        expected_ids = np.arange(start, stop).astype(str)
        if not np.array_equal(arrays["node_id_map"][start:stop], expected_ids):
            raise _fail("node_id_map is not the canonical 0..N-1 identity map")

    _validate_edge_array(
        arrays["edge_index"],
        size=expectation.message_edge_count,
        node_count=n,
        name="edge_index",
        canonical=False,
    )
    _shape(arrays["edge_timestamp"], (expectation.message_edge_count,), "edge_timestamp")
    _dtype(arrays["edge_timestamp"], np.dtype(np.int16), "edge_timestamp")
    if arrays["edge_timestamp"].size and (
        int(arrays["edge_timestamp"].min()) < 1800
        or int(arrays["edge_timestamp"].max()) > 2017
    ):
        raise _fail("message graph contains an invalid or future edge year")
    _shape(arrays["edge_weight"], (expectation.message_edge_count,), "edge_weight")
    _dtype(arrays["edge_weight"], np.dtype(np.float32), "edge_weight")
    if not bool(np.isfinite(arrays["edge_weight"]).all()) or np.any(
        arrays["edge_weight"] <= 0
    ):
        raise _fail("message graph edge weights must be finite and positive")

    split_specs = (
        ("train", expectation.train_size, None),
        ("validation", expectation.validation_size, 2018),
        ("test", expectation.test_size, 2019),
    )
    for split, size, exact_year in split_specs:
        edge_name = f"variant_{split}_positive"
        year_name = f"variant_{split}_year"
        weight_name = f"variant_{split}_weight"
        _validate_edge_array(
            arrays[edge_name],
            size=size,
            node_count=n,
            name=edge_name,
            canonical=canonical_supervision,
        )
        if has_split_metadata:
            _shape(arrays[year_name], (size,), year_name)
            _dtype(arrays[year_name], np.dtype(np.int16), year_name)
            if exact_year is None:
                if arrays[year_name].size and int(arrays[year_name].max()) > 2017:
                    raise _fail("train supervision contains a future edge year")
            elif not bool(np.all(arrays[year_name] == exact_year)):
                raise _fail(f"{split} supervision is not confined to {exact_year}")
            _shape(arrays[weight_name], (size,), weight_name)
            _dtype(arrays[weight_name], np.dtype(np.float32), weight_name)
            if not bool(np.isfinite(arrays[weight_name]).all()) or np.any(
                arrays[weight_name] <= 0
            ):
                raise _fail(f"{split} weights must be finite and positive")

    for split, size in (
        ("validation", expectation.validation_negative_size),
        ("test", expectation.test_negative_size),
    ):
        name = f"variant_{split}_negative"
        _validate_edge_array(
            arrays[name],
            size=size,
            node_count=n,
            name=name,
            canonical=canonical_supervision,
            # These are immutable official OGB evaluator candidates.  OGB 1.3.6
            # contains a handful of self-pairs; causal training samplers enforce
            # the stronger no-self-loop rule separately.
            allow_self_loops=True,
        )
        negative = np.sort(_pair_keys(arrays[name], n))
        positive = np.sort(_pair_keys(arrays[f"variant_{split}_positive"], n))
        if np.intersect1d(negative, positive, assume_unique=False).size:
            raise _fail(f"{split} stored negatives overlap target positives")

    _validate_symmetric_message_graph(arrays)
    _validate_train_alignment(arrays, n)


def _normalise_source_arrays(
    arrays: Mapping[str, np.ndarray],
    expectation: CorpusExpectation,
) -> dict[str, np.ndarray]:
    _validate_source_arrays(arrays, expectation, canonical_supervision=False)
    normalized = {
        name: np.array(value, copy=True, order="C", subok=False)
        for name, value in arrays.items()
    }
    for name in (
        "variant_train_positive",
        "variant_validation_positive",
        "variant_test_positive",
        "variant_validation_negative",
        "variant_test_negative",
    ):
        normalized[name] = _canonical_pairs(normalized[name])
    _validate_source_arrays(normalized, expectation, canonical_supervision=True)
    return normalized


def _derive_strict_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    edges = arrays["edge_index"]
    years = arrays["edge_timestamp"]
    before_2017 = years <= 2016
    train_2017 = (years == 2017) & (edges[0] < edges[1])
    validation_edges = arrays["variant_validation_positive"]
    validation_both_directions = np.concatenate(
        (validation_edges, validation_edges[::-1]),
        axis=1,
    )
    return {
        "strict_train_message_edge_index": np.ascontiguousarray(edges[:, before_2017]),
        "strict_train_message_edge_timestamp": np.ascontiguousarray(years[before_2017]),
        "strict_train_positive": np.ascontiguousarray(edges[:, train_2017]),
        "strict_validation_message_edge_index": np.ascontiguousarray(edges),
        "strict_validation_message_edge_timestamp": np.ascontiguousarray(years),
        "strict_test_message_edge_index": np.ascontiguousarray(
            np.concatenate((edges, validation_both_directions), axis=1)
        ),
        "strict_test_message_edge_timestamp": np.ascontiguousarray(
            np.concatenate(
                (
                    years,
                    np.full(validation_both_directions.shape[1], 2018, dtype=np.int16),
                )
            )
        ),
    }


def _validate_processed_arrays(
    arrays: Mapping[str, np.ndarray],
    expectation: CorpusExpectation,
) -> None:
    processed_keys = frozenset(arrays)
    if processed_keys not in PROCESSED_ARRAY_SCHEMAS:
        raise _fail("processed array whitelist is incomplete")
    source_keys = processed_keys - DERIVED_ARRAY_KEYS
    source = {name: arrays[name] for name in source_keys}
    _validate_source_arrays(source, expectation, canonical_supervision=True)
    expected = _derive_strict_arrays(source)
    for name, value in expected.items():
        actual = arrays[name]
        if actual.dtype != value.dtype or actual.shape != value.shape or not np.array_equal(
            actual, value
        ):
            raise _fail(f"derived strict temporal array {name} is invalid or tampered")
    if arrays["strict_train_message_edge_timestamp"].size and int(
        arrays["strict_train_message_edge_timestamp"].max()
    ) > 2016:
        raise _fail("strict train message graph contains an edge after 2016")
    if arrays["strict_validation_message_edge_timestamp"].size and int(
        arrays["strict_validation_message_edge_timestamp"].max()
    ) > 2017:
        raise _fail("strict validation message graph contains an edge after 2017")
    if arrays["strict_test_message_edge_timestamp"].size and int(
        arrays["strict_test_message_edge_timestamp"].max()
    ) > 2018:
        raise _fail("strict test message graph contains an edge after 2018")


def _processed_schema(names: frozenset[str]) -> frozenset[str]:
    if names not in PROCESSED_ARRAY_SCHEMAS:
        raise _fail("processed artifact array inventory is not an approved schema")
    return names


def _little_endian_contiguous(array: np.ndarray) -> np.ndarray:
    value = np.ascontiguousarray(array)
    if value.dtype.itemsize <= 1 or value.dtype.byteorder == "|":
        return value
    if value.dtype.byteorder == ">" or (
        value.dtype.byteorder == "=" and sys.byteorder == "big"
    ):
        return value.byteswap().view(value.dtype.newbyteorder("<"))
    if value.dtype.byteorder != "<":
        return value.view(value.dtype.newbyteorder("<"))
    return value


def _array_record(name: str, array: np.ndarray) -> CorpusArrayManifest:
    value = _little_endian_contiguous(array)
    digest = hashlib.sha256(memoryview(value).cast("B")).hexdigest()
    return CorpusArrayManifest(
        name=name,
        sha256=digest,
        dtype=str(value.dtype),
        shape=tuple(value.shape),
        byteCount=value.nbytes,
    )


def _array_records(arrays: Mapping[str, np.ndarray]) -> tuple[CorpusArrayManifest, ...]:
    return tuple(_array_record(name, arrays[name]) for name in sorted(arrays))


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        writer: Any = np.savez_compressed
        writer(temporary, **{name: arrays[name] for name in sorted(arrays)})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _license_receipt_path(root: Path) -> Path:
    return root / LICENSE_RECEIPT_RELATIVE_PATH


def _write_license_receipt(root: Path, package_sha256: str) -> None:
    receipt = {
        "schemaVersion": "gfm.license-acceptance/1.0",
        "corpusId": CORPUS_ID,
        "licenseId": LICENSE_ID,
        "accepted": True,
        "packageSha256": package_sha256,
        "acceptedAt": datetime.now(UTC),
    }
    _atomic_text(_license_receipt_path(root), canonical_json(receipt))


def _check_license_receipt(root: Path, package_sha256: str) -> dict[str, Any]:
    path = _license_receipt_path(root)
    if not path.is_file() or path.is_symlink():
        raise _fail("explicit ODC-BY-1.0 license acceptance receipt is absent")
    receipt = _json_no_duplicates(path.read_bytes(), label="license acceptance receipt")
    expected_keys = {
        "schemaVersion",
        "corpusId",
        "licenseId",
        "accepted",
        "packageSha256",
        "acceptedAt",
    }
    if set(receipt) != expected_keys:
        raise _fail("license acceptance receipt fields are invalid")
    required = {
        "schemaVersion": "gfm.license-acceptance/1.0",
        "corpusId": CORPUS_ID,
        "licenseId": LICENSE_ID,
        "accepted": True,
        "packageSha256": package_sha256,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise _fail(f"license acceptance receipt field {key} is invalid")
    try:
        accepted_at = datetime.fromisoformat(str(receipt["acceptedAt"]).replace("Z", "+00:00"))
    except ValueError as exc:
        raise _fail("license acceptance timestamp is invalid") from exc
    if accepted_at.tzinfo is None or accepted_at.utcoffset() is None:
        raise _fail("license acceptance timestamp must be timezone-aware")
    return receipt


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@contextlib.contextmanager
def _trusted_ogb_pickle_scope() -> Iterator[None]:
    """Temporarily opt the pinned OGB loader into legacy trusted pickle loads."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by runtime profile
        raise _fail("PyTorch is required to load the official OGB cache") from exc
    original = torch.load

    def trusted_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original


def _ogb_arrays(dataset: Any, split: Mapping[str, Mapping[str, Any]]) -> dict[str, np.ndarray]:
    data = dataset[0]

    def positive(part: str) -> np.ndarray:
        value = _to_numpy(split[part]["edge"])
        if value.ndim != 2 or value.shape[1] != 2:
            raise _fail(f"official {part} positive edges are not [E,2]")
        return value.T.astype(np.int64, copy=False)

    def negative(part: str) -> np.ndarray:
        value = _to_numpy(split[part]["edge_neg"])
        if value.ndim != 2 or value.shape[1] != 2:
            raise _fail(f"official {part} negative edges are not [E,2]")
        return value.T.astype(np.int64, copy=False)

    def split_vector(part: str, name: str, dtype: np.dtype[Any]) -> np.ndarray:
        if name not in split[part]:
            raise _fail(f"official {part} split is missing {name}")
        return _to_numpy(split[part][name]).reshape(-1).astype(dtype, copy=False)

    features = _to_numpy(data.x).astype(np.float32, copy=False)
    node_count = int(features.shape[0])
    return {
        "edge_index": _to_numpy(data.edge_index).astype(np.int64, copy=False),
        "x": features,
        "num_nodes": np.asarray(node_count, dtype=np.int64),
        "node_id_map": np.asarray([str(index) for index in range(node_count)]),
        "directed": np.asarray(False, dtype=np.bool_),
        "edge_weight": _to_numpy(data.edge_weight).reshape(-1).astype(np.float32, copy=False),
        "edge_timestamp": _to_numpy(data.edge_year).reshape(-1).astype(np.int16, copy=False),
        "variant_train_positive": positive("train"),
        "variant_train_weight": split_vector("train", "weight", np.dtype(np.float32)),
        "variant_train_year": split_vector("train", "year", np.dtype(np.int16)),
        "variant_validation_positive": positive("valid"),
        "variant_validation_negative": negative("valid"),
        "variant_validation_weight": split_vector("valid", "weight", np.dtype(np.float32)),
        "variant_validation_year": split_vector("valid", "year", np.dtype(np.int16)),
        "variant_test_positive": positive("test"),
        "variant_test_negative": negative("test"),
        "variant_test_weight": split_vector("test", "weight", np.dtype(np.float32)),
        "variant_test_year": split_vector("test", "year", np.dtype(np.int16)),
    }


def _directory_fingerprint(source: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (path for path in source.glob("**/*") if path.is_file() and not path.is_symlink()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    if not files:
        raise _fail("official OGB cache does not contain any files")
    for path in files:
        relative = path.relative_to(source).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
        digest.update(b"\x00")
    return digest.hexdigest()


def _package_manifest(source_fingerprint: str) -> dict[str, Any]:
    recorded_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    item = {
        "name": CORPUS_ID,
        "path": PACKAGE_GRAPH_PATH,
        "sourceFormat": "trusted_local_ogb",
        "sourceFiles": ["OGB 1.3.6 official cache"],
        "datasetRole": "benchmark",
        "splitKind": "official",
        "directed": False,
        "licensePolicy": {
            "status": "verified",
            "identifier": LICENSE_ID,
            "sourceUrl": LICENSE_URL,
            "allowedUses": ["evaluation", "adaptation", "inference", "pretraining"],
            "attribution": ATTRIBUTION,
            "evidenceIds": ["ogbl-collab-official-metadata"],
        },
        "licenseEvidence": [
            {
                "id": "ogbl-collab-official-metadata",
                "kind": "official_metadata",
                "sourceUrl": LICENSE_URL,
                "recordedAt": recorded_at,
                "recordedBy": "socialgraph-fm-ogb-adapter-v2",
            }
        ],
        "dataGovernance": {
            "containsPersonalData": False,
            "deidentified": True,
            "attributeAllowlist": [],
            "excludedAttributes": [],
            "retention": "research_archive",
            "userDataTrainingOptIn": False,
        },
        "transformRecipes": [
            {
                "id": "ogb-identity-v1",
                "graphVariant": "raw",
                "inputArray": "x",
                "outputArray": "x",
                "featureTransform": "identity",
                "fitScope": "none",
                "parameters": {"ogbDataset": CORPUS_ID},
            }
        ],
        "linkPredictionProtocol": {
            "messagePassingEdgeArray": "edge_index",
            "trainPositiveArray": "variant_train_positive",
            "validationPositiveArray": "variant_validation_positive",
            "testPositiveArray": "variant_test_positive",
            "validationNegativeArray": "variant_validation_negative",
            "testNegativeArray": "variant_test_negative",
            "edgeYearArray": "edge_timestamp",
            "edgeWeightArray": "edge_weight",
            "trainYearMax": 2017,
            "validationYear": 2018,
            "testYear": 2019,
            "negativeSampler": "stored",
            "undirectedCanonicalization": "min_max",
            "reverseEdgeLeakagePolicy": "reject",
            "positiveOverlapPolicy": "allow_temporal_recurrence",
            "evaluator": "ogb.linkproppred.Evaluator(ogbl-collab)",
            "evaluatorVersion": OGB_VERSION,
        },
        "transforms": [
            "ogb_official_cache_to_safe_npz",
            "preserve_official_temporal_split_metadata",
        ],
    }
    return {
        "schemaVersion": "socialgraph-fm-dataset-package/1.0",
        "trustedSource": "local official OGB cache (path intentionally excluded)",
        "sourceFingerprint": source_fingerprint,
        "datasets": [item],
        "skipped": [],
    }


def _write_package_atomic(
    destination: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    temporary_directory: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    graph = temporary_directory / f"ogbl-collab-source-{uuid.uuid4().hex}.npz"
    package = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        writer: Any = np.savez_compressed
        writer(graph, **{name: arrays[name] for name in sorted(arrays)})
        with zipfile.ZipFile(package, "x", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, allow_nan=False, sort_keys=True),
            )
            archive.write(graph, PACKAGE_GRAPH_PATH)
        os.replace(package, destination)
    finally:
        graph.unlink(missing_ok=True)
        package.unlink(missing_ok=True)


def fetch_ogbl_collab(
    root: str | Path | None,
    *,
    accept_license: str,
) -> dict[str, Any]:
    """Fetch/process OGB in the explicit runtime root and emit a safe SGFM package."""

    if accept_license != LICENSE_ID:
        raise _fail(f"--accept-license must be exactly {LICENSE_ID}")
    layout = prepare_runtime_layout(root, operation="fetch")
    selected_root = layout.root
    package_path = selected_root / PACKAGE_RELATIVE_PATH
    if package_path.exists():
        arrays, package_manifest, package_hash = _load_safe_package(
            package_path,
            temporary_directory=layout.temporary,
        )
        _validate_source_arrays(
            arrays,
            OFFICIAL_EXPECTATION,
            canonical_supervision=False,
        )
        _write_license_receipt(selected_root, package_hash)
        return {
            "schemaVersion": "gfm.corpus-fetch/1.0",
            "corpusId": CORPUS_ID,
            "packagePath": str(package_path),
            "packageSha256": package_hash,
            "sourceFingerprint": package_manifest["sourceFingerprint"],
            "licenseId": LICENSE_ID,
            "reused": True,
        }

    # Acquisition and the legacy trusted-pickle boundary stay in the API
    # package's canonical converter.  The GFM side consumes only its JSON+NPZ
    # output and never accepts uploaded Pickle.
    from ..corpus_fetch import fetch_ogbl_collab as bootstrap_safe_package

    bootstrap_safe_package(selected_root, accept_license=accept_license)
    arrays, package_manifest, package_hash = _load_safe_package(
        package_path,
        temporary_directory=layout.temporary,
    )
    _validate_source_arrays(arrays, OFFICIAL_EXPECTATION, canonical_supervision=False)
    require_storage_reserve(selected_root, operation="run")
    _write_license_receipt(selected_root, package_hash)
    return {
        "schemaVersion": "gfm.corpus-fetch/1.0",
        "corpusId": CORPUS_ID,
        "packagePath": str(package_path),
        "packageSha256": package_hash,
        "sourceFingerprint": package_manifest["sourceFingerprint"],
        "licenseId": LICENSE_ID,
        "reused": False,
    }


def _copy_package_into_root(source: Path, destination: Path, package_hash: str) -> None:
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        if not destination.is_file() or file_sha256(destination) != package_hash:
            raise _fail("an immutable package with different content already exists")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, COPY_CHUNK_BYTES)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if file_sha256(temporary) != package_hash:
            raise _fail("package changed while it was copied into the runtime root")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _formal_manifest(
    *,
    package_sha256: str,
    source_fingerprint: str,
    arrays: Mapping[str, np.ndarray],
) -> FormalCorpusManifest:
    records = _array_records(arrays)
    protocol = TemporalLinkProtocolManifest(
        protocolId="ogbl-collab-official-v1",
        evaluator="ogb.linkproppred.Evaluator(ogbl-collab)",
    )
    warnings = [
        "OGB author features are aggregate paper embeddings; their point-in-time provenance "
        "cannot be independently verified.",
        "Official transductive train positives are also present in the train message graph.",
    ]
    if "variant_validation_year" not in arrays:
        warnings.append(
            "The API v1 safe package omits split year/weight vectors; official validation/test "
            "years are enforced by the signed protocol while message-edge years/weights remain "
            "fully preserved."
        )
    values: dict[str, Any] = {
        "schemaVersion": "gfm.formal-corpus/1.0",
        "corpusId": CORPUS_ID,
        "purpose": "formal_benchmark",
        "datasetRole": "benchmark",
        "ogbVersion": OGB_VERSION,
        "licenseId": LICENSE_ID,
        "licenseSourceUrl": LICENSE_URL,
        "licenseAccepted": True,
        "attribution": ATTRIBUTION,
        "sourceFingerprint": source_fingerprint,
        "packageSha256": package_sha256,
        "adapter": "socialgraph_gfm.corpus.ogbl_collab",
        "adapterVersion": "1.0",
        "nodeCount": OFFICIAL_EXPECTATION.node_count,
        "featureShape": (
            OFFICIAL_EXPECTATION.node_count,
            OFFICIAL_EXPECTATION.feature_dim,
        ),
        "messageEdgeCount": OFFICIAL_EXPECTATION.message_edge_count,
        "splitSizes": {
            "train": OFFICIAL_EXPECTATION.train_size,
            "validation": OFFICIAL_EXPECTATION.validation_size,
            "test": OFFICIAL_EXPECTATION.test_size,
        },
        "arrays": records,
        "temporalProtocol": protocol,
        "warnings": tuple(warnings),
    }
    values["logicalHash"] = canonical_sha256(values)
    values["createdAt"] = datetime.now(UTC)
    values["artifactPath"] = ARTIFACT_RELATIVE_PATH.as_posix()
    try:
        return FormalCorpusManifest.model_validate(values)
    except ValidationError as exc:
        raise _fail(f"formal corpus manifest construction failed: {exc}") from exc


def prepare_ogbl_collab_corpus(
    package: str | Path,
    root: str | Path | None,
) -> dict[str, Any]:
    """Validate the trusted package, materialize immutable NPZ, and publish its manifest."""

    layout = prepare_runtime_layout(root, operation="run")
    selected_root = layout.root
    package_path = Path(package).expanduser().resolve(strict=True)
    arrays, package_manifest, package_hash = _load_safe_package(
        package_path,
        temporary_directory=layout.temporary,
    )
    normalized = _normalise_source_arrays(arrays, OFFICIAL_EXPECTATION)
    processed = {**normalized, **_derive_strict_arrays(normalized)}
    _validate_processed_arrays(processed, OFFICIAL_EXPECTATION)

    canonical_package = selected_root / PACKAGE_RELATIVE_PATH
    _copy_package_into_root(package_path, canonical_package, package_hash)
    _check_license_receipt(selected_root, package_hash)
    artifact = selected_root / ARTIFACT_RELATIVE_PATH
    manifest_path = selected_root / MANIFEST_RELATIVE_PATH
    if artifact.exists() or manifest_path.exists():
        checked = check_ogbl_collab_corpus(selected_root)
        if checked["packageSha256"] != package_hash:
            raise _fail("a different immutable formal corpus is already materialized")
        return checked

    _write_npz_atomic(artifact, processed)
    reloaded = _load_npz_safely(
        artifact,
        expected_keys=_processed_schema(frozenset(processed)),
    )
    _validate_processed_arrays(reloaded, OFFICIAL_EXPECTATION)
    manifest = _formal_manifest(
        package_sha256=package_hash,
        source_fingerprint=str(package_manifest["sourceFingerprint"]),
        arrays=reloaded,
    )
    _atomic_text(
        manifest_path,
        canonical_json(manifest.model_dump(mode="python", by_alias=True, exclude_none=False)),
    )
    return check_ogbl_collab_corpus(selected_root)


def _resolve_artifact(root: Path, relative: str) -> Path:
    if "\\" in relative or not relative:
        raise _fail("artifactPath must be a canonical POSIX relative path")
    posix = PurePosixPath(relative)
    if posix.is_absolute() or any(part in ("", ".", "..") for part in posix.parts):
        raise _fail("artifactPath contains path traversal")
    if posix.as_posix() != ARTIFACT_RELATIVE_PATH.as_posix():
        raise _fail("artifactPath is not the fixed ogbl-collab processed location")
    candidate = root.joinpath(*posix.parts)
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise _fail("artifactPath escapes the runtime root") from exc
    if not resolved.is_file() or candidate.is_symlink():
        raise _fail("processed artifact must be a regular, non-symlink file")
    return resolved


def _validate_array_records(
    claimed: tuple[CorpusArrayManifest, ...],
    arrays: Mapping[str, np.ndarray],
) -> None:
    actual = _array_records(arrays)
    claimed_by_name = {record.name: record for record in claimed}
    actual_by_name = {record.name: record for record in actual}
    if set(claimed_by_name) != set(actual_by_name):
        raise _fail("manifest array inventory does not match the processed NPZ")
    for name in sorted(actual_by_name):
        if claimed_by_name[name] != actual_by_name[name]:
            raise _fail(f"processed array {name} failed hash/shape/dtype/size verification")


def check_ogbl_collab_corpus(root: str | Path | None) -> dict[str, Any]:
    """Re-read every artifact and return a validated formal manifest alias dictionary."""

    selected_root = artifact_root(root)
    require_storage_reserve(selected_root, operation="run")
    manifest_path = selected_root / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise _fail(f"formal manifest is absent: {manifest_path}")
    payload = _json_no_duplicates(manifest_path.read_bytes(), label="formal corpus manifest")
    try:
        manifest = FormalCorpusManifest.model_validate(payload)
    except ValidationError as exc:
        raise _fail(f"formal corpus manifest contract is invalid: {exc}") from exc

    package_path = selected_root / PACKAGE_RELATIVE_PATH
    if not package_path.is_file() or package_path.is_symlink():
        raise _fail("canonical source package is absent")
    package_hash = file_sha256(package_path)
    if package_hash != manifest.package_sha256:
        raise _fail("canonical source package SHA-256 does not match the manifest")
    _check_license_receipt(selected_root, package_hash)
    artifact = _resolve_artifact(selected_root, manifest.artifact_path)
    claimed_names = frozenset(record.name for record in manifest.arrays)
    arrays = _load_npz_safely(
        artifact,
        expected_keys=_processed_schema(claimed_names),
    )
    _validate_processed_arrays(arrays, OFFICIAL_EXPECTATION)
    _validate_array_records(manifest.arrays, arrays)
    if manifest.message_edge_count != arrays["edge_index"].shape[1]:
        raise _fail("manifest messageEdgeCount does not match the artifact")
    split_sizes = {
        "train": arrays["variant_train_positive"].shape[1],
        "validation": arrays["variant_validation_positive"].shape[1],
        "test": arrays["variant_test_positive"].shape[1],
    }
    if manifest.split_sizes != split_sizes:
        raise _fail("manifest splitSizes do not match the artifact")
    return manifest.model_dump(mode="json", by_alias=True, exclude_none=False)


def load_ogbl_collab_arrays(root: str | Path | None) -> dict[str, np.ndarray]:
    """Load a fully revalidated corpus for baseline code; arrays never contain pickle data."""

    manifest = check_ogbl_collab_corpus(root)
    selected_root = artifact_root(root)
    artifact = _resolve_artifact(selected_root, str(manifest["artifactPath"]))
    claimed_names = frozenset(str(record["name"]) for record in manifest["arrays"])
    arrays = _load_npz_safely(
        artifact,
        expected_keys=_processed_schema(claimed_names),
    )
    _validate_processed_arrays(arrays, OFFICIAL_EXPECTATION)
    return arrays
