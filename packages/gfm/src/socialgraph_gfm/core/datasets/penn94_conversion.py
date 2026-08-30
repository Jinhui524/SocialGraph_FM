"""One-time fixed-asset conversion for LINKX's pickle-backed Penn94 splits.

This module does not expose a generic pickle conversion interface. The only deserialization path
is an isolated worker bound to one commit, URL, filename, raw SHA-256, and runtime location.
Production loading of the derived asset always uses ``numpy.load(..., allow_pickle=False)``.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import pickletools
import shutil
import subprocess
import sys
import uuid
import zipfile
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from socialgraph_gfm.runtime import core_runtime_root

PENN94_LINKX_COMMIT = "82f8f05c5c3ec16bd5b505cc7ad62ab5e09051e6"
PENN94_RAW_SPLIT_URL = (
    "https://raw.githubusercontent.com/CUAI/Non-Homophily-Large-Scale/"
    f"{PENN94_LINKX_COMMIT}/data/splits/fb100-Penn94-splits.npy"
)
PENN94_RAW_SPLIT_SHA256 = "88a1060358482d8e25b978ab59c4ff71771388cd5ffac3dd775a3cd9dc85b032"
PENN94_RAW_SPLIT_MAX_BYTES = 1_000_000
PENN94_DATA_SHA256 = "85a9b5b6aa908695d0c154648ddb88977480c58ba3000c70bb973e64fa7bc69a"
PENN94_LABELED_NODE_COUNT = 38_815
PENN94_SPLIT_COUNTS = {"train": 19_407, "valid": 9_703, "test": 9_705}
PENN94_CONVERTER_VERSION = "socialgraph-fm.core-penn94-fixed-converter/1.0"

_WORKER_FLAG = "--fixed-hash-locked-penn94-worker"


@dataclass(frozen=True)
class _Penn94RuntimePaths:
    root: Path
    raw_split: Path
    raw_penn: Path
    staging: Path
    worker_output: Path
    published_target: Path


def _runtime_paths() -> _Penn94RuntimePaths:
    root = core_runtime_root()
    raw_root = root / "raw" / "facebook100" / "1.0.0"
    staging = root / ".penn94-fixed-conversion-staging"
    return _Penn94RuntimePaths(
        root=root,
        raw_split=raw_root / "fb100-Penn94-splits.npy",
        raw_penn=raw_root / "Penn94.mat",
        staging=staging,
        worker_output=staging / "penn94-official-splits-safe.npz",
        published_target=(
            root / "derived" / "facebook100" / "penn94-official-splits" / "1.0.0"
        ),
    )

_ALLOWED_PICKLE_OPCODES = {
    "APPENDS",
    "BINGET",
    "BININT",
    "BININT1",
    "BININT2",
    "BINPUT",
    "BINUNICODE",
    "BUILD",
    "EMPTY_DICT",
    "EMPTY_LIST",
    "GLOBAL",
    "MARK",
    "NEWFALSE",
    "NEWTRUE",
    "NONE",
    "PROTO",
    "REDUCE",
    "SETITEMS",
    "SHORT_BINBYTES",
    "STOP",
    "TUPLE",
    "TUPLE1",
    "TUPLE3",
}
_ALLOWED_PICKLE_GLOBALS = {
    "numpy.core.multiarray _reconstruct",
    "numpy ndarray",
    "numpy dtype",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_penn94_raw_split(path: Path) -> str:
    """Verify bytes against the sole authorized pickle-backed source."""

    observed = hashlib.sha256(_read_fixed_raw_bytes(path)).hexdigest()
    if observed != PENN94_RAW_SPLIT_SHA256:
        raise ValueError("Penn94 split does not match the fixed raw SHA-256")
    return observed


def _read_fixed_raw_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw_bytes = stream.read(PENN94_RAW_SPLIT_MAX_BYTES + 1)
    if len(raw_bytes) > PENN94_RAW_SPLIT_MAX_BYTES:
        raise ValueError("Penn94 split exceeds the fixed catalog maximum")
    return raw_bytes


def validate_penn94_safe_splits(
    arrays: Mapping[str, Any], *, labeled_node_indices: Collection[int] | np.ndarray
) -> None:
    """Validate five exact integer splits against the Penn94 labeled-node mask."""

    if set(arrays) != set(PENN94_SPLIT_COUNTS):
        raise ValueError("Penn94 safe split inventory is invalid")
    labeled = np.asarray(labeled_node_indices)
    if labeled.ndim != 1 or labeled.dtype.kind not in "iu" or labeled.dtype.kind == "b":
        raise ValueError("Penn94 labeled-node indices must be a one-dimensional integer array")
    labeled_values = tuple(int(value) for value in labeled)
    labeled_set = set(labeled_values)
    if (
        len(labeled_values) != PENN94_LABELED_NODE_COUNT
        or len(labeled_set) != PENN94_LABELED_NODE_COUNT
        or min(labeled_set, default=-1) < 0
    ):
        raise ValueError("Penn94 labeled-node mask must contain exactly 38,815 unique indices")

    normalized: dict[str, np.ndarray] = {}
    for role, expected_count in PENN94_SPLIT_COUNTS.items():
        array = np.asarray(arrays[role])
        if array.shape != (5, expected_count):
            raise ValueError(f"Penn94 {role} array has the wrong exact shape")
        if array.dtype.kind not in "iu" or array.dtype.kind == "b":
            raise ValueError(f"Penn94 {role} array must contain integers only")
        normalized[role] = array.astype(np.int64, copy=False)

    for split_index in range(5):
        role_sets = {
            role: {int(value) for value in array[split_index]}
            for role, array in normalized.items()
        }
        if any(len(role_sets[role]) != PENN94_SPLIT_COUNTS[role] for role in role_sets):
            raise ValueError("Penn94 split roles must not contain duplicate indices")
        if (
            role_sets["train"] & role_sets["valid"]
            or role_sets["train"] & role_sets["test"]
            or role_sets["valid"] & role_sets["test"]
        ):
            raise ValueError("Penn94 split roles must be disjoint")
        if set().union(*role_sets.values()) != labeled_set:
            raise ValueError("Penn94 split indices must exactly match the labeled-node mask")


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array, dtype=np.dtype("<i8")),
        version=(1, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def write_deterministic_safe_splits(path: Path, arrays: Mapping[str, Any]) -> str:
    """Write deterministic compressed NPY members containing primitive integers only."""

    if set(arrays) != set(PENN94_SPLIT_COUNTS):
        raise ValueError("Penn94 safe output inventory is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for role in ("train", "valid", "test"):
            array = np.asarray(arrays[role])
            if array.dtype.kind not in "iu" or array.dtype.kind == "b":
                raise ValueError("Penn94 safe output permits primitive integer arrays only")
            member = zipfile.ZipInfo(f"{role}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100600 << 16
            archive.writestr(member, _npy_bytes(array), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return _sha256(path)


def load_penn94_safe_splits(
    path: Path, *, labeled_node_indices: Collection[int] | np.ndarray
):
    """Production loader for the integer-only derived asset; pickle is always disabled."""

    from .acquire import safe_load_numpy_arrays
    from ..splits import IndexSplit

    arrays = safe_load_numpy_arrays(
        path,
        expected_keys=set(PENN94_SPLIT_COUNTS),
        max_array_elements=100_000,
        max_total_array_bytes=2 * 1024 * 1024,
    )
    validate_penn94_safe_splits(arrays, labeled_node_indices=labeled_node_indices)
    return tuple(
        IndexSplit(
            train=tuple(int(value) for value in arrays["train"][split_index]),
            validation=tuple(int(value) for value in arrays["valid"][split_index]),
            test=tuple(int(value) for value in arrays["test"][split_index]),
        )
        for split_index in range(5)
    )


class _Penn94RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        qualified = f"{module} {name}"
        if qualified == "numpy.core.multiarray _reconstruct":
            return getattr(np, "_core").multiarray._reconstruct
        if qualified == "numpy ndarray":
            return np.ndarray
        if qualified == "numpy dtype":
            return np.dtype
        raise pickle.UnpicklingError(f"forbidden Penn94 pickle global: {qualified}")


def _read_verified_object_payload(path: Path) -> bytes:
    raw_bytes = _read_fixed_raw_bytes(path)
    if hashlib.sha256(raw_bytes).hexdigest() != PENN94_RAW_SPLIT_SHA256:
        raise ValueError("Penn94 split does not match the fixed raw SHA-256")
    with io.BytesIO(raw_bytes) as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError("Penn94 raw split must use NPY version 1.0")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        if shape != (5,) or fortran_order or dtype != np.dtype("O"):
            raise ValueError("Penn94 raw split has an unexpected object-array header")
        payload = stream.read()
    operations = tuple(pickletools.genops(payload))
    opcode_names = {operation.name for operation, _argument, _position in operations}
    if not opcode_names <= _ALLOWED_PICKLE_OPCODES:
        raise ValueError("Penn94 raw split contains a forbidden pickle opcode")
    globals_used = {
        str(argument)
        for operation, argument, _position in operations
        if operation.name == "GLOBAL"
    }
    if globals_used != _ALLOWED_PICKLE_GLOBALS:
        raise ValueError("Penn94 raw split contains a forbidden pickle global")
    return payload


def _convert_in_isolated_worker() -> None:
    def deny_network(event: str, _args: tuple[Any, ...]) -> None:
        if event.startswith("socket."):
            raise RuntimeError("network access is forbidden in the Penn94 deserialization worker")

    sys.addaudithook(deny_network)
    paths = _runtime_paths()
    payload = _read_verified_object_payload(paths.raw_split)
    loaded = _Penn94RestrictedUnpickler(io.BytesIO(payload)).load()
    if not isinstance(loaded, np.ndarray) or loaded.shape != (5,) or loaded.dtype != np.dtype("O"):
        raise ValueError("Penn94 restricted result has an unexpected outer structure")
    rows: dict[str, list[np.ndarray]] = {role: [] for role in PENN94_SPLIT_COUNTS}
    for split in loaded:
        if not isinstance(split, dict) or set(split) != set(PENN94_SPLIT_COUNTS):
            raise ValueError("Penn94 restricted result has an unexpected split structure")
        for role in rows:
            values = np.asarray(split[role])
            if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
                raise ValueError("Penn94 restricted result contains non-integer split values")
            rows[role].append(values.astype(np.int64, copy=False))
    arrays = {role: np.stack(values) for role, values in rows.items()}
    write_deterministic_safe_splits(paths.worker_output, arrays)


def _converter_code_sha256() -> str:
    return _sha256(Path(__file__).resolve())


def _write_canonical_json(path: Path, payload: Mapping[str, Any]) -> None:
    from socialgraph_gfm.canonical import canonical_json

    path.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")


def _validate_existing_publication(target: Path, labeled_indices: np.ndarray) -> Path:
    from socialgraph_gfm.canonical import canonical_sha256
    from .recipes import load_dataset_recipes

    recipe = load_dataset_recipes()["facebook100"]
    sources = {source.source_id: source for source in recipe.sources}
    manifest_path = target / "conversion-manifest.json"
    asset_path = target / "penn94-official-splits-safe.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    without_hash = {key: value for key, value in manifest.items() if key != "manifestSha256"}
    expected_keys = {
        "schemaVersion", "sourceCommit", "sourceUrl", "sourceSha256", "penn94DataUrl",
        "penn94DataObservedSha256", "derivedFormat", "derivedSha256", "converterVersion",
        "converterCodeSha256", "splitCount", "labeledNodeCount", "roleCounts",
        "recipeSha256", "manifestSha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("Penn94 conversion manifest inventory mismatch")
    if manifest.get("manifestSha256") != canonical_sha256(without_hash):
        raise ValueError("Penn94 conversion manifest hash mismatch")
    if manifest.get("derivedSha256") != _sha256(asset_path):
        raise ValueError("Penn94 derived asset hash mismatch")
    if manifest.get("converterCodeSha256") != _converter_code_sha256():
        raise ValueError("Penn94 converter code hash mismatch")
    expected_fields = {
        "schemaVersion": "socialgraph-fm.core-penn94-split-conversion/1.0",
        "sourceCommit": PENN94_LINKX_COMMIT,
        "sourceUrl": PENN94_RAW_SPLIT_URL,
        "sourceSha256": PENN94_RAW_SPLIT_SHA256,
        "penn94DataUrl": sources["Penn94"].url,
        "penn94DataObservedSha256": PENN94_DATA_SHA256,
        "derivedFormat": "npz with primitive little-endian int64 NPY members",
        "converterVersion": PENN94_CONVERTER_VERSION,
        "splitCount": 5,
        "labeledNodeCount": PENN94_LABELED_NODE_COUNT,
        "roleCounts": PENN94_SPLIT_COUNTS,
        "recipeSha256": recipe.recipe_sha256,
    }
    if any(manifest.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("Penn94 conversion manifest fixed provenance mismatch")
    load_penn94_safe_splits(asset_path, labeled_node_indices=labeled_indices)
    return target


def convert_penn94_official_splits() -> Path:
    """Convert the fixed Penn94 split below the configured core runtime root."""

    from socialgraph_gfm.canonical import canonical_sha256

    from .acquire import download_source, safe_load_mat_arrays
    from .recipes import load_dataset_recipes

    paths = _runtime_paths()
    recipe = load_dataset_recipes()["facebook100"]
    sources = {source.source_id: source for source in recipe.sources}
    split_source = sources["Penn94-official-splits"]
    if (
        split_source.url != PENN94_RAW_SPLIT_URL
        or split_source.expected_sha256 != PENN94_RAW_SPLIT_SHA256
        or split_source.max_bytes != PENN94_RAW_SPLIT_MAX_BYTES
    ):
        raise ValueError("Penn94 recipe is not bound to the authorized split asset")
    if paths.raw_split.exists():
        verify_penn94_raw_split(paths.raw_split)
    else:
        download_source(
            recipe_id=recipe.recipe_id,
            source_id=split_source.source_id,
            runtime_root=paths.root,
        )
        verify_penn94_raw_split(paths.raw_split)
    if paths.raw_penn.exists():
        penn_observed = _sha256(paths.raw_penn)
        if penn_observed != PENN94_DATA_SHA256:
            raise ValueError("Penn94 data does not match the fixed observed SHA-256")
    else:
        penn_result = download_source(
            recipe_id=recipe.recipe_id,
            source_id=sources["Penn94"].source_id,
            runtime_root=paths.root,
        )
        penn_observed = penn_result.observed_sha256
        if penn_observed != PENN94_DATA_SHA256:
            raise ValueError("Penn94 data does not match the fixed observed SHA-256")
    arrays = safe_load_mat_arrays(
        paths.raw_penn,
        expected_keys={"A", "local_info"},
        max_array_elements=5_000_000,
        max_worker_memory_bytes=2 * 1024 * 1024 * 1024,
        timeout_seconds=60,
    )
    profile = np.asarray(arrays["local_info"])
    labeled_indices = np.flatnonzero(profile[:, 1] != 0).astype(np.int64)
    if labeled_indices.size != PENN94_LABELED_NODE_COUNT:
        raise ValueError("Penn94 data does not contain exactly 38,815 labeled gender nodes")

    if paths.published_target.exists():
        return _validate_existing_publication(paths.published_target, labeled_indices)
    if paths.staging.exists():
        resolved = paths.staging.resolve()
        if resolved != (paths.root / ".penn94-fixed-conversion-staging").resolve():
            raise ValueError("refusing to clean an unexpected Penn94 staging path")
        shutil.rmtree(resolved)
    paths.staging.mkdir(parents=True)
    try:
        environment = os.environ.copy()
        for key in tuple(environment):
            upper = key.upper()
            if "PROXY" in upper or "TOKEN" in upper or "SECRET" in upper or "PASSWORD" in upper:
                environment.pop(key, None)
        environment["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), _WORKER_FLAG],
            cwd=paths.staging,
            env=environment,
            check=True,
            timeout=120,
        )
        load_penn94_safe_splits(paths.worker_output, labeled_node_indices=labeled_indices)
        derived_sha = _sha256(paths.worker_output)
        publication = paths.staging / f"publication-{uuid.uuid4().hex}"
        publication.mkdir()
        asset_path = publication / "penn94-official-splits-safe.npz"
        os.replace(paths.worker_output, asset_path)
        manifest_without_hash = {
            "schemaVersion": "socialgraph-fm.core-penn94-split-conversion/1.0",
            "sourceCommit": PENN94_LINKX_COMMIT,
            "sourceUrl": PENN94_RAW_SPLIT_URL,
            "sourceSha256": PENN94_RAW_SPLIT_SHA256,
            "penn94DataUrl": sources["Penn94"].url,
            "penn94DataObservedSha256": penn_observed,
            "derivedFormat": "npz with primitive little-endian int64 NPY members",
            "derivedSha256": derived_sha,
            "converterVersion": PENN94_CONVERTER_VERSION,
            "converterCodeSha256": _converter_code_sha256(),
            "splitCount": 5,
            "labeledNodeCount": PENN94_LABELED_NODE_COUNT,
            "roleCounts": PENN94_SPLIT_COUNTS,
            "recipeSha256": recipe.recipe_sha256,
        }
        manifest = {
            **manifest_without_hash,
            "manifestSha256": canonical_sha256(manifest_without_hash),
        }
        _write_canonical_json(publication / "conversion-manifest.json", manifest)
        paths.published_target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(publication, paths.published_target)
        return _validate_existing_publication(paths.published_target, labeled_indices)
    finally:
        if paths.staging.exists():
            shutil.rmtree(paths.staging)


if __name__ == "__main__":
    if sys.argv != [str(Path(__file__)), _WORKER_FLAG]:
        raise SystemExit("This fixed converter has no public command-line interface")
    _convert_in_isolated_worker()


__all__ = [
    "PENN94_CONVERTER_VERSION",
    "PENN94_DATA_SHA256",
    "PENN94_LABELED_NODE_COUNT",
    "PENN94_LINKX_COMMIT",
    "PENN94_RAW_SPLIT_SHA256",
    "PENN94_RAW_SPLIT_MAX_BYTES",
    "PENN94_RAW_SPLIT_URL",
    "PENN94_SPLIT_COUNTS",
    "convert_penn94_official_splits",
    "load_penn94_safe_splits",
    "validate_penn94_safe_splits",
    "verify_penn94_raw_split",
    "write_deterministic_safe_splits",
]
