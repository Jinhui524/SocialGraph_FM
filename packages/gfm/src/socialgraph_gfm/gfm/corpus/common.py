"""Small, fail-closed artifact primitives shared by GFM corpus adapters.

The helpers deliberately support only JSON/JSONL and numeric NumPy arrays.
They are kept separate from the legacy OGB corpus boundary, which is allowed
to consume trusted local pickle data during its one-time conversion.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ...canonical import canonical_json, canonical_sha256, file_sha256
from ...errors import ContractViolation

MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_NPZ_BYTES = 8 * 1024 * 1024 * 1024
MAX_NPZ_ENTRIES = 64
PORTABLE_ID_HASH_ALGORITHM = "canonical-sha256-first-8-bytes-big-endian-uint64"
WINDOWS_SHARING_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
WINDOWS_SHARING_RETRY_ERRORS = frozenset({32, 33, 1224})
WINDOWS_ACCESS_DENIED_RETRY_DELAYS = (0.05, 0.1)


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"GFM corpus: {message}")


def portable_id_hash(value: str) -> np.uint64:
    """Return the stable uint64 identifier used to join text and graph rows."""

    if not isinstance(value, str) or not value:
        raise _fail("portable identifier must be a nonempty string")
    digest = bytes.fromhex(canonical_sha256(value))
    return np.uint64(int.from_bytes(digest[:8], "big", signed=False))


def safe_relative_path(value: str) -> PurePosixPath:
    """Validate one portable artifact path and reject traversal or aliases."""

    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise _fail(f"unsafe artifact path {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _fail(f"artifact path traverses its root: {value!r}")
    if path.as_posix() != value:
        raise _fail(f"artifact path is not canonical POSIX: {value!r}")
    return path


def resolve_within(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    """Resolve a validated relative path without allowing a symlink escape."""

    root = root.expanduser().resolve()
    path = safe_relative_path(relative)
    candidate = root.joinpath(*path.parts)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _fail(f"artifact path escapes its root: {relative!r}") from exc
    if must_exist and (candidate.is_symlink() or not resolved.is_file()):
        raise _fail(f"artifact must be a regular non-symlink file: {relative!r}")
    return resolved


def _temporary(path: Path) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")


def _replace_with_windows_retry(source: Path, destination: Path) -> None:
    """Replace one artifact, tolerating only short Windows sharing races.

    Antivirus scanners, editors, and readers opened without ``FILE_SHARE_DELETE``
    can make an otherwise valid atomic replace fail transiently with sharing or
    lock violations (WinError 32, 33, or 1224).  Access denied (WinError 5) is
    ambiguous, so it receives only two micro-retries and only when the target was
    already an ordinary writable file.  Every other case fails immediately.  The
    fully fsynced temporary file remains the replacement source throughout.
    """

    sharing_attempt = 0
    access_denied_attempt = 0
    access_denied_retryable = (
        destination.exists()
        and destination.is_file()
        and not destination.is_symlink()
        and os.access(destination, os.W_OK)
    )
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            code = getattr(exc, "winerror", None)
            if code in WINDOWS_SHARING_RETRY_ERRORS and sharing_attempt < len(
                WINDOWS_SHARING_RETRY_DELAYS
            ):
                delay = WINDOWS_SHARING_RETRY_DELAYS[sharing_attempt]
                sharing_attempt += 1
                time.sleep(delay)
                continue
            if (
                code == 5
                and access_denied_retryable
                and access_denied_attempt < len(WINDOWS_ACCESS_DENIED_RETRY_DELAYS)
            ):
                delay = WINDOWS_ACCESS_DENIED_RETRY_DELAYS[access_denied_attempt]
                access_denied_attempt += 1
                time.sleep(delay)
                continue
            raise


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Hold one non-blocking cross-process lock for an artifact operation.

    The lock file is intentionally persistent: deleting it after unlock would
    permit two processes to lock different inodes during a remove/create race.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl_module: Any = __import__("fcntl")
                fcntl_module.flock(stream.fileno(), fcntl_module.LOCK_EX | fcntl_module.LOCK_NB)
            locked = True
        except OSError as exc:
            raise _fail(f"exclusive operation is already running: {path.name}") from exc
        try:
            yield
        finally:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl_unlock_module: Any = __import__("fcntl")
                    fcntl_unlock_module.flock(stream.fileno(), fcntl_unlock_module.LOCK_UN)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_windows_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    count = 0
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(canonical_json(row) + "\n")
                count += 1
            stream.flush()
            os.fsync(stream.fileno())
        _replace_with_windows_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def append_jsonl_fsync(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    """Append one already validated API page and durably record it for resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return len(rows)


def read_json_object(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > max_bytes:
        raise _fail(f"JSON artifact is absent, unsafe or too large: {path.name}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"forbidden JSON constant {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise _fail(f"invalid JSON artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(f"JSON artifact must be an object: {path.name}")
    return value


def read_jsonl(path: Path, *, max_line_bytes: int = 8 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise _fail(f"JSONL artifact is absent or unsafe: {path}")
    with path.open("rb") as stream:
        for number, payload in enumerate(stream, start=1):
            if len(payload) > max_line_bytes:
                raise _fail(f"JSONL line {number} exceeds the safety limit")
            try:
                value = json.loads(payload)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise _fail(f"invalid JSONL line {number}") from exc
            if not isinstance(value, dict):
                raise _fail(f"JSONL line {number} must be an object")
            yield value


def _validate_array(name: str, value: np.ndarray) -> np.ndarray:
    if not name or "/" in name or "\\" in name or "." in name:
        raise _fail(f"unsafe NumPy array name {name!r}")
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind in {"O", "S", "U", "V"}:
        raise _fail(f"array {name!r} is not a fixed numeric/bool array")
    if array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
        raise _fail(f"array {name!r} contains NaN or Infinity")
    return np.ascontiguousarray(array)


def atomic_write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if not arrays or len(arrays) > MAX_NPZ_ENTRIES:
        raise _fail("NPZ array inventory is empty or exceeds its limit")
    validated = {name: _validate_array(name, value) for name, value in sorted(arrays.items())}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary(path)
    try:
        with temporary.open("xb") as stream:
            writer: Any = np.savez_compressed
            writer(stream, **validated)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_npz_safe(
    path: Path,
    *,
    expected: Mapping[str, tuple[str, int | None]],
) -> dict[str, np.ndarray]:
    """Inspect archive names first, then load with ``allow_pickle=False``."""

    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_NPZ_BYTES:
        raise _fail("NPZ artifact is absent, unsafe or too large")
    expected_names = {f"{name}.npy" for name in expected}
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) != len(expected_names):
                raise _fail("NPZ entry count does not match its schema")
            seen: set[str] = set()
            total = 0
            for info in infos:
                name = info.filename
                if name in seen or name not in expected_names:
                    raise _fail(f"NPZ contains a duplicate or unexpected member {name!r}")
                safe_relative_path(name)
                if info.is_dir() or ((info.external_attr >> 16) & 0o170000) == stat.S_IFLNK:
                    raise _fail("NPZ contains a directory or symbolic link")
                total += info.file_size
                if total > MAX_NPZ_BYTES:
                    raise _fail("NPZ exceeds its uncompressed safety limit")
                seen.add(name)
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(expected):
                raise _fail("NPZ array inventory changed while loading")
            arrays = {
                name: np.array(loaded[name], copy=True, order="C", subok=False)
                for name in sorted(expected)
            }
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise _fail("NPZ is unreadable or contains a pickle/object array") from exc
    for name, (dtype, dimensions) in expected.items():
        array = _validate_array(name, arrays[name])
        if array.dtype.str != dtype or (dimensions is not None and array.ndim != dimensions):
            raise _fail(f"array {name!r} does not match dtype/rank contract")
    return arrays


def array_inventory(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    records = []
    for name, value in sorted(arrays.items()):
        array = _validate_array(name, value)
        records.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
                "byteLength": int(array.nbytes),
            }
        )
    return records


@dataclass(frozen=True)
class ShardRecord:
    path: str
    sha256: str
    rows: int
    arrays: tuple[dict[str, Any], ...]


class NumericShardWriter:
    """Write deterministic, immutable numeric NPZ shards."""

    def __init__(self, directory: Path, *, prefix: str, rows_per_shard: int) -> None:
        if rows_per_shard < 1 or not prefix or any(char in prefix for char in "/\\:"):
            raise _fail("invalid shard writer configuration")
        self.directory = directory
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self._index = 0

    def write(self, arrays: Mapping[str, np.ndarray]) -> ShardRecord:
        validated = {name: _validate_array(name, value) for name, value in arrays.items()}
        rows = {int(value.shape[0]) for value in validated.values() if value.ndim > 0}
        if len(rows) != 1:
            raise _fail("all shard arrays must share their first dimension")
        row_count = rows.pop()
        if row_count > self.rows_per_shard:
            raise _fail("shard exceeds configured row capacity")
        relative = f"{self.prefix}-{self._index:05d}.npz"
        path = self.directory / relative
        if path.exists():
            raise _fail(f"immutable shard already exists: {relative}")
        atomic_write_npz(path, validated)
        self._index += 1
        return ShardRecord(
            path=relative,
            sha256=file_sha256(path),
            rows=row_count,
            arrays=tuple(array_inventory(validated)),
        )


def build_manifest(
    *,
    schema_version: str,
    corpus_id: str,
    license_id: str,
    source: Mapping[str, Any],
    shards: Sequence[ShardRecord],
    splits: Mapping[str, Any],
    privacy: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": schema_version,
        "corpusId": corpus_id,
        "licenseId": license_id,
        "source": dict(source),
        "shards": [record.__dict__ for record in shards],
        "splits": dict(splits),
        "privacy": dict(privacy),
    }
    if extra:
        payload.update(extra)
    payload["logicalHash"] = canonical_sha256(payload)
    payload["createdAt"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return payload


def verify_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    logical_hash = manifest.get("logicalHash")
    payload = {
        key: value for key, value in manifest.items() if key not in {"logicalHash", "createdAt"}
    }
    if not isinstance(logical_hash, str) or logical_hash != canonical_sha256(payload):
        raise _fail("manifest logical hash does not match")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise _fail("manifest has no shards")
    expected_paths = {"manifest.json"}
    for record in shards:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "rows", "arrays"}:
            raise _fail("manifest shard descriptor is invalid")
        path = resolve_within(directory, str(record["path"]))
        expected_paths.add(str(record["path"]))
        if file_sha256(path) != record["sha256"]:
            raise _fail(f"artifact hash mismatch: {record['path']}")
        raw_arrays = record["arrays"]
        if not isinstance(raw_arrays, (list, tuple)):
            raise _fail("manifest shard array inventory is invalid")
        arrays = list(raw_arrays)
        if path.suffix == ".npz":
            if not arrays:
                raise _fail("numeric shard is missing its array inventory")
            expected = {
                str(item["name"]): (str(item["dtype"]), len(item["shape"]))
                for item in arrays
                if isinstance(item, dict)
            }
            if len(expected) != len(arrays):
                raise _fail("numeric shard array inventory is malformed")
            loaded = load_npz_safe(path, expected=expected)
            if array_inventory(loaded) != arrays:
                raise _fail(f"numeric shard array metadata mismatch: {record['path']}")
            rows = {int(value.shape[0]) for value in loaded.values() if value.ndim > 0}
            if len(rows) != 1 or rows.pop() != int(record["rows"]):
                raise _fail(f"numeric shard row count mismatch: {record['path']}")
    expected_directories: set[str] = set()
    for expected_path in expected_paths:
        parent = PurePosixPath(expected_path).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_paths: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise _fail(f"artifact directory contains a symbolic link: {relative}")
        if path.is_dir():
            if relative not in expected_directories:
                raise _fail(f"artifact directory contains an undeclared directory: {relative}")
            continue
        if not path.is_file():
            raise _fail(f"artifact directory contains a non-regular entry: {relative}")
        actual_paths.add(relative)
    if actual_paths != expected_paths:
        raise _fail(
            "artifact file whitelist mismatch; "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
