"""Persistent, hash-bound structural retrieval for reviewed Governance cases."""

from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import stat
import tempfile
import threading
import time
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_sha256

if os.name == "nt":  # pragma: win32 cover
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _CreateFileW = _kernel32.CreateFileW
    _CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    _CreateFileW.restype = wintypes.HANDLE
    _FlushFileBuffers = _kernel32.FlushFileBuffers
    _FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _FlushFileBuffers.restype = wintypes.BOOL
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

    _INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
    _GENERIC_WRITE = 0x40000000
    _FILE_SHARE_READ = 0x0001
    _FILE_SHARE_WRITE = 0x0002
    _FILE_SHARE_DELETE = 0x0004
    _OPEN_EXISTING = 3
    _FILE_FLAG_BACKUP_SEMANTICS = 0x02000000

INDEX_SCHEMA_VERSION = "socialgraph-fm.governance-reviewed-case-index/1.0"
TRANSITION_SCHEMA_VERSION = "socialgraph-fm.governance-reviewed-case-transition/1.0"
CASE_SCHEMA_VERSION: Literal["socialgraph-fm.governance-reviewed-case/1.0"] = (
    "socialgraph-fm.governance-reviewed-case/1.0"
)
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_CASE_PATTERN = r"^[A-Za-z0-9._:-]{1,200}$"
_RUN_PATTERN = r"^governance-[0-9a-f]{32}$"
_TIME_PATTERN = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
_TRANSITION_FILE = ".pending-transition.json"
_TRANSITION_STAGE = ".pending-transition"
_MAX_CONTROL_BYTES = 64 * 1024 * 1024
_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}
_ROOT_LOCK_DEPTHS = threading.local()


def _root_lock_key(root: Path) -> str:
    return os.path.normcase(str(root))


def _root_thread_lock(root: Path) -> threading.RLock:
    key = _root_lock_key(root)
    with _ROOT_LOCKS_GUARD:
        return _ROOT_LOCKS.setdefault(key, threading.RLock())


def _assert_same_open_file(path: Path, opened: os.stat_result) -> None:
    try:
        current = os.lstat(path)
    except OSError as error:
        raise ValueError("reviewed-case file identity changed while open") from error
    if not os.path.samestat(opened, current):
        raise ValueError("reviewed-case file identity changed while open")


def _open_regular_file(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("reviewed-case file is missing, linked, or unreadable") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("reviewed-case file is not regular")
        _assert_same_open_file(path, opened)
        return descriptor, opened
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_bytes(path: Path, *, max_bytes: int) -> bytes:
    descriptor, opened = _open_regular_file(path)
    try:
        if opened.st_size < 1 or opened.st_size > max_bytes:
            raise ValueError("reviewed-case file size is outside its safety bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("reviewed-case file changed while being read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ValueError("reviewed-case file grew while being read")
        _assert_same_open_file(path, opened)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _regular_descriptor(path: Path) -> dict[str, Any]:
    descriptor, opened = _open_regular_file(path)
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
        _assert_same_open_file(path, opened)
    finally:
        os.close(descriptor)
    if total != opened.st_size:
        raise ValueError("reviewed-case file changed while being hashed")
    return {"bytes": total, "sha256": digest.hexdigest()}


def _copy_verified_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    descriptor, opened = _open_regular_file(source)
    digest = hashlib.sha256()
    total = 0
    try:
        if opened.st_size != expected_bytes:
            raise ValueError("reviewed-case SQLite identity is invalid")
        with destination.open("xb") as target:
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
                total += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        _assert_same_open_file(source, opened)
    finally:
        os.close(descriptor)
    if total != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError("reviewed-case SQLite identity is invalid")


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("reviewed-case JSON contains a duplicate key")
        value[key] = item
    return value


def _read_canonical_json_object(path: Path, *, label: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, max_bytes=_MAX_CONTROL_BYTES)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if raw != _canonical_json_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _acquire_process_lock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        while True:
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                return
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                time.sleep(0.05)
    else:
        import fcntl

        attributes = vars(fcntl)
        attributes["flock"](descriptor, attributes["LOCK_EX"])


def _release_process_lock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        attributes = vars(fcntl)
        attributes["flock"](descriptor, attributes["LOCK_UN"])


@contextmanager
def _filesystem_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise ValueError("reviewed-case root lock is missing, linked, or unreadable") from error
    acquired = False
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("reviewed-case root lock is not regular")
        if opened.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        elif opened.st_size != 1:
            raise ValueError("reviewed-case root lock identity is invalid")
        _acquire_process_lock(descriptor)
        acquired = True
        opened = os.fstat(descriptor)
        _assert_same_open_file(path, opened)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, 2) != b"\0":
            raise ValueError("reviewed-case root lock identity is invalid")
        yield
    finally:
        try:
            if acquired:
                _release_process_lock(descriptor)
        finally:
            os.close(descriptor)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
    )


class CaseKindEntry(_StrictModel):
    kind: Literal["node", "relation", "group"]
    target_ids: tuple[str, ...] = Field(alias="targetIds", min_length=1, max_length=100)

    @field_validator("target_ids", mode="before")
    @classmethod
    def validate_target_ids(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise ValueError("targetIds must be a JSON string array")
        return tuple(value)

    @model_validator(mode="after")
    def validate_canonical_ids(self) -> CaseKindEntry:
        if tuple(sorted(set(self.target_ids))) != self.target_ids:
            raise ValueError("targetIds must be unique and canonically sorted")
        return self


class ReviewedCaseRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.governance-reviewed-case/1.0"] = Field(
        default=CASE_SCHEMA_VERSION, alias="schemaVersion"
    )
    case_id: str = Field(alias="caseId", pattern=_CASE_PATTERN)
    case_hash: str = Field(alias="caseHash", pattern=_HASH_PATTERN)
    run_id: str = Field(alias="runId", pattern=_RUN_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=_HASH_PATTERN)
    kind_key: Literal[
        "node",
        "relation",
        "group",
        "node+relation",
        "node+group",
        "relation+group",
        "node+relation+group",
    ] = Field(alias="kindKey")
    kind_entries: tuple[CaseKindEntry, ...] = Field(alias="kindEntries", min_length=1, max_length=3)
    concluded_at: str = Field(alias="concludedAt", pattern=_TIME_PATTERN)
    review_hash: str = Field(alias="reviewHash", pattern=_HASH_PATTERN)
    review_status: Literal["concluded", "reviewed"] = Field(alias="reviewStatus")
    artifact_id: str = Field(alias="artifactId", pattern=r"^governance-artifact-[0-9a-f]{32}$")
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=_HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_state_hash: str = Field(alias="modelStateHash", pattern=_HASH_PATTERN)
    vector_file: str = Field(alias="vectorFile", pattern=r"^[0-9a-f]{64}\.npz$")
    vector_sha256: str = Field(alias="vectorSha256", pattern=_HASH_PATTERN)
    vector_bytes: int = Field(alias="vectorBytes", ge=1, le=16 * 1024 * 1024)
    vector_content_hash: str = Field(alias="vectorContentHash", pattern=_HASH_PATTERN)
    indexed_at: str = Field(alias="indexedAt", pattern=_TIME_PATTERN)
    source_request_hash: str = Field(alias="sourceRequestHash", pattern=_HASH_PATTERN)
    record_hash: str = Field(alias="recordHash", pattern=_HASH_PATTERN)

    @field_validator("kind_entries", mode="before")
    @classmethod
    def validate_kind_entries(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("kindEntries must be a JSON array")
        return tuple(value)

    @model_validator(mode="after")
    def validate_record_hash(self) -> ReviewedCaseRecord:
        kind_order = {"node": 0, "relation": 1, "group": 2}
        canonical = tuple(sorted(self.kind_entries, key=lambda item: kind_order[item.kind]))
        if canonical != self.kind_entries or len({item.kind for item in canonical}) != len(
            canonical
        ):
            raise ValueError("kindEntries must have unique kinds in canonical order")
        expected_key = "+".join(item.kind for item in canonical)
        if self.kind_key != expected_key:
            raise ValueError("kindKey must be derived from kindEntries")
        logical = self.model_dump(mode="json", by_alias=True, exclude={"record_hash"})
        if self.record_hash != canonical_sha256(logical):
            raise ValueError("recordHash is invalid")
        return self


@dataclass(frozen=True)
class CaseVectors:
    embedding: np.ndarray
    structure: np.ndarray
    modality: np.ndarray


@dataclass(frozen=True)
class SimilarCase:
    record: ReviewedCaseRecord
    score: float
    embedding_score: float
    structure_score: float
    modality_score: float


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
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        # FlushFileBuffers requires GENERIC_WRITE. Sharing read, write, and delete keeps the
        # short-lived barrier compatible with cooperating readers, writers, and renames.
        handle = _CreateFileW(
            str(path),
            _GENERIC_WRITE,
            _FILE_SHARE_READ | _FILE_SHARE_WRITE | _FILE_SHARE_DELETE,
            None,
            _OPEN_EXISTING,
            _FILE_FLAG_BACKUP_SEMANTICS,
            None,
        )
        if handle == _INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), "failed to open directory durability handle")
        try:
            if not _FlushFileBuffers(handle):
                raise OSError(ctypes.get_last_error(), "failed to flush directory metadata")
        finally:
            if not _CloseHandle(handle):
                raise OSError(ctypes.get_last_error(), "failed to close directory durability handle")
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_replace(source: Path, destination: Path) -> None:
    source_parent = source.parent
    os.replace(source, destination)
    _fsync_directory(destination.parent)
    if source_parent != destination.parent:
        _fsync_directory(source_parent)


def _durable_unlink(path: Path) -> None:
    path.unlink()
    _fsync_directory(path.parent)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, _canonical_json_bytes(value))


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
    return stream.getvalue()


def _vector_bytes(vectors: CaseVectors) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=False) as archive:
        for name, value in (
            ("embedding.npy", vectors.embedding),
            ("structure.npy", vectors.structure),
            ("modality.npy", vectors.modality),
        ):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, _npy_bytes(value), compresslevel=6)
    return stream.getvalue()


def _validated_vectors(vectors: CaseVectors) -> CaseVectors:
    embedding = np.ascontiguousarray(vectors.embedding, dtype=np.float32)
    structure = np.ascontiguousarray(vectors.structure, dtype=np.float32)
    modality = np.ascontiguousarray(vectors.modality, dtype=np.float32)
    if (
        embedding.shape != (256,)
        or structure.shape != (6,)
        or modality.shape != (5,)
        or not all(bool(np.isfinite(value).all()) for value in (embedding, structure, modality))
    ):
        raise ValueError("reviewed-case vectors have an invalid shape, dtype, or value")
    return CaseVectors(embedding=embedding, structure=structure, modality=modality)


def _vector_content_hash(vectors: CaseVectors) -> str:
    return canonical_sha256(
        {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
            }
            for name, value in (
                ("embedding", vectors.embedding),
                ("structure", vectors.structure),
                ("modality", vectors.modality),
            )
        }
    )


def _load_vector(path: Path, record: ReviewedCaseRecord) -> CaseVectors:
    try:
        raw = _read_regular_bytes(path, max_bytes=16 * 1024 * 1024)
    except ValueError as error:
        raise ValueError("reviewed-case vector artifact identity is invalid") from error
    if len(raw) != record.vector_bytes or hashlib.sha256(raw).hexdigest() != record.vector_sha256:
        raise ValueError("reviewed-case vector artifact identity is invalid")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            expected = {"embedding.npy", "structure.npy", "modality.npy"}
            if (
                len(infos) != 3
                or {item.filename for item in infos} != expected
                or any(item.is_dir() or item.file_size < 1 for item in infos)
            ):
                raise ValueError("reviewed-case vector inventory is invalid")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != {"embedding", "structure", "modality"}:
                raise ValueError("reviewed-case vector keys are invalid")
            vectors = _validated_vectors(
                CaseVectors(
                    embedding=np.asarray(archive["embedding"]),
                    structure=np.asarray(archive["structure"]),
                    modality=np.asarray(archive["modality"]),
                )
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError("reviewed-case vector artifact is unsafe") from error
    if _vector_content_hash(vectors) != record.vector_content_hash:
        raise ValueError("reviewed-case vector content hash is invalid")
    return vectors


def _similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0 and right_norm == 0:
        return 1.0
    if left_norm == 0 or right_norm == 0:
        return 0.0
    cosine = float(np.dot(left.astype(np.float64), right.astype(np.float64))) / (
        left_norm * right_norm
    )
    return min(1.0, max(0.0, (min(1.0, max(-1.0, cosine)) + 1.0) / 2.0))


class ReviewedCaseIndex:
    """SQLite metadata plus safe numeric vectors, verified before every read or write."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.database = self.root / "index.sqlite3"
        self.vector_root = self.root / "vectors"
        self.manifest = self.root / "manifest.json"
        self._transition = self.root / _TRANSITION_FILE
        self._staging = self.root / _TRANSITION_STAGE
        self._thread_lock = _root_thread_lock(self.root)
        self._lock_key = _root_lock_key(self.root)
        self._lock_path = self.root.parent / f".{self.root.name}.reviewed-case.lock"
        with self._locked():
            if not self.root.exists():
                self._initialize()
            self._ready_records()

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self._thread_lock:
            depths = getattr(_ROOT_LOCK_DEPTHS, "values", None)
            if depths is None:
                depths = {}
                _ROOT_LOCK_DEPTHS.values = depths
            depth = depths.get(self._lock_key, 0)
            if depth:
                depths[self._lock_key] = depth + 1
                try:
                    yield
                finally:
                    depths[self._lock_key] -= 1
                return
            with _filesystem_lock(self._lock_path):
                depths[self._lock_key] = 1
                try:
                    yield
                finally:
                    del depths[self._lock_key]

    def _initialize(self) -> None:
        self.root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self.root.name}.initialize.",
                suffix=".tmp",
                dir=self.root.parent,
            )
        )
        try:
            staging_vectors = staging / "vectors"
            staging_database = staging / "index.sqlite3"
            staging_vectors.mkdir()
            self._create_database(staging_database)
            document = self._manifest_document((), database=staging_database)
            _atomic_json(staging / "manifest.json", document)
            _durable_replace(staging, self.root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    @staticmethod
    def _create_database(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute(
                "CREATE TABLE cases ("
                "case_id TEXT PRIMARY KEY, model_state_hash TEXT NOT NULL, kind TEXT NOT NULL, "
                "graph_version_hash TEXT NOT NULL, concluded_at TEXT NOT NULL, "
                "record_hash TEXT NOT NULL UNIQUE, record_json TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX cases_compatibility ON cases(model_state_hash, kind, case_id)"
            )
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES('schemaVersion', ?)",
                (INDEX_SCHEMA_VERSION,),
            )
            connection.commit()
        finally:
            connection.close()
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())

    @staticmethod
    def _descriptor(path: Path) -> dict[str, Any]:
        return _regular_descriptor(path)

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @classmethod
    def _valid_descriptor(cls, value: Any) -> bool:
        return (
            isinstance(value, dict)
            and set(value) == {"bytes", "sha256"}
            and type(value.get("bytes")) is int
            and value["bytes"] > 0
            and cls._valid_sha256(value.get("sha256"))
        )

    @classmethod
    def _validate_manifest_document(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("reviewed-case index manifest must be an object")
        expected_keys = {
            "schemaVersion",
            "database",
            "vectors",
            "recordHashes",
            "indexHash",
            "manifestHash",
        }
        logical = {key: item for key, item in value.items() if key != "manifestHash"}
        vectors = value.get("vectors")
        record_hashes = value.get("recordHashes")
        if (
            set(value) != expected_keys
            or value.get("schemaVersion") != INDEX_SCHEMA_VERSION
            or value.get("manifestHash") != canonical_sha256(logical)
            or not cls._valid_descriptor(value.get("database"))
            or not isinstance(vectors, dict)
            or any(
                not isinstance(name, str)
                or len(name) != 68
                or not name.endswith(".npz")
                or not cls._valid_sha256(name[:-4])
                or not cls._valid_descriptor(descriptor)
                for name, descriptor in vectors.items()
            )
            or not isinstance(record_hashes, list)
            or len(set(record_hashes)) != len(record_hashes)
            or any(not cls._valid_sha256(item) for item in record_hashes)
        ):
            raise ValueError("reviewed-case index manifest hash or shape is invalid")
        index_hash = canonical_sha256(
            {"schemaVersion": INDEX_SCHEMA_VERSION, "recordHashes": record_hashes}
        )
        if value.get("indexHash") != index_hash:
            raise ValueError("reviewed-case logical index hash is invalid")
        return value

    def _read_manifest(self) -> dict[str, Any]:
        value = _read_canonical_json_object(
            self.manifest,
            label="reviewed-case index manifest",
        )
        return self._validate_manifest_document(value)

    def _connection(self, *, read_only: bool, database: Path | None = None) -> sqlite3.Connection:
        database = self.database if database is None else database
        if read_only:
            uri = database.as_uri() + "?mode=ro&immutable=1"
            connection = sqlite3.connect(uri, uri=True)
        else:
            connection = sqlite3.connect(database)
            connection.execute("PRAGMA synchronous=FULL")
        connection.row_factory = sqlite3.Row
        return connection

    def _records(self, database: Path | None = None) -> tuple[ReviewedCaseRecord, ...]:
        connection = self._connection(read_only=True, database=database)
        try:
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            metadata = connection.execute("SELECT key, value FROM metadata ORDER BY key").fetchall()
            if [tuple(row) for row in integrity] != [("ok",)] or [
                tuple(row) for row in metadata
            ] != [("schemaVersion", INDEX_SCHEMA_VERSION)]:
                raise ValueError("reviewed-case SQLite schema is invalid")
            rows = connection.execute(
                "SELECT case_id, model_state_hash, kind, graph_version_hash, concluded_at, "
                "record_hash, record_json FROM cases ORDER BY case_id"
            ).fetchall()
        finally:
            connection.close()
        records = tuple(ReviewedCaseRecord.model_validate_json(row["record_json"]) for row in rows)
        for row, record in zip(rows, records, strict=True):
            columns = (
                row["case_id"],
                row["model_state_hash"],
                row["kind"],
                row["graph_version_hash"],
                row["concluded_at"],
                row["record_hash"],
            )
            expected = (
                record.case_id,
                record.model_state_hash,
                record.kind_key,
                record.graph_version_hash,
                record.concluded_at,
                record.record_hash,
            )
            canonical_record = json.dumps(
                record.model_dump(mode="json", by_alias=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if columns != expected or row["record_json"] != canonical_record:
                raise ValueError("reviewed-case SQLite row binding is invalid")
        return records

    @classmethod
    def _matches_descriptor(cls, path: Path, descriptor: Any) -> bool:
        if not cls._valid_descriptor(descriptor):
            return False
        try:
            return _regular_descriptor(path) == descriptor
        except ValueError:
            return False

    def _records_for_manifest(
        self, manifest: Mapping[str, Any], database: Path
    ) -> tuple[ReviewedCaseRecord, ...]:
        descriptor = manifest.get("database")
        if not self._valid_descriptor(descriptor):
            raise ValueError("reviewed-case SQLite identity is invalid")
        assert isinstance(descriptor, dict)
        with tempfile.TemporaryDirectory(prefix="socialgraph-reviewed-case-snapshot-") as raw:
            snapshot = Path(raw) / "index.sqlite3"
            _copy_verified_regular_file(
                database,
                snapshot,
                expected_bytes=descriptor["bytes"],
                expected_sha256=descriptor["sha256"],
            )
            records = self._records(snapshot)
        record_hashes = [record.record_hash for record in records]
        if manifest.get("recordHashes") != record_hashes:
            raise ValueError("reviewed-case logical index hash is invalid")
        return records

    def _verify_vector_files(
        self,
        records: Sequence[ReviewedCaseRecord],
        vector_manifest: Mapping[str, Any],
    ) -> None:
        expected_vectors = {record.vector_file for record in records}
        if set(vector_manifest) != expected_vectors:
            raise ValueError("reviewed-case manifest vector inventory is invalid")
        for record in records:
            descriptor = vector_manifest[record.vector_file]
            if descriptor != {
                "bytes": record.vector_bytes,
                "sha256": record.vector_sha256,
            }:
                raise ValueError("reviewed-case vector descriptor is invalid")
            _load_vector(self.vector_root / record.vector_file, record)

    def _vector_inventory(self) -> set[str]:
        if self.vector_root.is_symlink() or not self.vector_root.is_dir():
            raise ValueError("reviewed-case vector directory is missing or linked")
        entries = tuple(self.vector_root.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError("reviewed-case vector directory contains an unsafe entry")
        return {path.name for path in entries}

    def _verify_snapshot(
        self,
        manifest: Mapping[str, Any],
    ) -> tuple[ReviewedCaseRecord, ...]:
        validated = self._validate_manifest_document(dict(manifest))
        records = self._records_for_manifest(validated, self.database)
        expected_vectors = {record.vector_file for record in records}
        if self._vector_inventory() != expected_vectors:
            raise ValueError("reviewed-case vector inventory is invalid")
        vectors = validated["vectors"]
        assert isinstance(vectors, dict)
        self._verify_vector_files(records, vectors)
        return records

    def _verify(self) -> tuple[ReviewedCaseRecord, ...]:
        return self._verify_snapshot(self._read_manifest())

    def _manifest_document(
        self,
        records: Sequence[ReviewedCaseRecord],
        *,
        database: Path,
    ) -> dict[str, Any]:
        record_hashes = [record.record_hash for record in records]
        index_hash = canonical_sha256(
            {"schemaVersion": INDEX_SCHEMA_VERSION, "recordHashes": record_hashes}
        )
        logical: dict[str, Any] = {
            "schemaVersion": INDEX_SCHEMA_VERSION,
            "database": self._descriptor(database),
            "vectors": {
                record.vector_file: {
                    "bytes": record.vector_bytes,
                    "sha256": record.vector_sha256,
                }
                for record in records
            },
            "recordHashes": record_hashes,
            "indexHash": index_hash,
        }
        return {**logical, "manifestHash": canonical_sha256(logical)}

    @classmethod
    def _validate_transition_document(
        cls, value: Any
    ) -> tuple[dict[str, Any], dict[str, Any], ReviewedCaseRecord]:
        if not isinstance(value, dict):
            raise TypeError("reviewed-case pending transition must be an object")
        expected_keys = {
            "schemaVersion",
            "previousManifest",
            "nextManifest",
            "record",
            "transitionHash",
        }
        logical = {key: item for key, item in value.items() if key != "transitionHash"}
        if (
            set(value) != expected_keys
            or value.get("schemaVersion") != TRANSITION_SCHEMA_VERSION
            or value.get("transitionHash") != canonical_sha256(logical)
        ):
            raise ValueError("reviewed-case pending transition hash or shape is invalid")
        previous = cls._validate_manifest_document(value.get("previousManifest"))
        following = cls._validate_manifest_document(value.get("nextManifest"))
        record = ReviewedCaseRecord.model_validate(value.get("record"))
        previous_hashes = previous["recordHashes"]
        following_hashes = following["recordHashes"]
        previous_vectors = previous["vectors"]
        following_vectors = following["vectors"]
        assert isinstance(previous_hashes, list)
        assert isinstance(following_hashes, list)
        assert isinstance(previous_vectors, dict)
        assert isinstance(following_vectors, dict)
        if (
            previous["manifestHash"] == following["manifestHash"]
            or previous["database"] == following["database"]
            or len(following_hashes) != len(previous_hashes) + 1
            or set(following_hashes) != {*previous_hashes, record.record_hash}
            or record.record_hash in previous_hashes
            or record.vector_file in previous_vectors
            or set(following_vectors) != {*previous_vectors, record.vector_file}
            or following_vectors.get(record.vector_file)
            != {"bytes": record.vector_bytes, "sha256": record.vector_sha256}
            or any(
                following_vectors.get(name) != descriptor
                for name, descriptor in previous_vectors.items()
            )
        ):
            raise ValueError("reviewed-case pending transition is not one exact append")
        return previous, following, record

    def _read_transition(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], ReviewedCaseRecord]:
        value = _read_canonical_json_object(
            self._transition,
            label="reviewed-case pending transition",
        )
        previous, following, record = self._validate_transition_document(value)
        return value, previous, following, record

    def _validate_staging_root(self) -> None:
        if not os.path.lexists(self._staging):
            return
        if self._staging.is_symlink() or self._staging.is_junction() or not self._staging.is_dir():
            raise ValueError("reviewed-case transition staging path is unsafe")

    def _remove_staging(self) -> None:
        self._validate_staging_root()
        if self._staging.exists():
            shutil.rmtree(self._staging)
            _fsync_directory(self.root)

    def _staging_inventory(self) -> set[str]:
        self._validate_staging_root()
        if not self._staging.exists():
            return set()
        entries = tuple(self._staging.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError("reviewed-case transition staging contains an unsafe entry")
        return {path.name for path in entries}

    def _clear_transition(self, expected: Mapping[str, Any]) -> None:
        current, _, _, _ = self._read_transition()
        if current != expected:
            raise ValueError("reviewed-case pending transition changed during recovery")
        self._remove_staging()
        current, _, _, _ = self._read_transition()
        if current != expected:
            raise ValueError("reviewed-case pending transition changed during cleanup")
        _durable_unlink(self._transition)

    @staticmethod
    def _contains_record(
        records: Sequence[ReviewedCaseRecord], expected: ReviewedCaseRecord
    ) -> bool:
        return any(record == expected for record in records)

    def _recover_transition(self) -> tuple[ReviewedCaseRecord, ...]:
        transition, previous, following, pending_record = self._read_transition()
        current_manifest = self._read_manifest()
        if current_manifest not in (previous, following):
            raise ValueError("reviewed-case manifest is outside the pending transition")
        previous_database = previous["database"]
        following_database = following["database"]
        matches_previous = self._matches_descriptor(self.database, previous_database)
        matches_following = self._matches_descriptor(self.database, following_database)
        if matches_previous == matches_following:
            raise ValueError("reviewed-case SQLite is outside the pending transition")

        self._validate_staging_root()
        staged_database = self._staging / "index.sqlite3"
        staged_vector = self._staging / pending_record.vector_file
        staged_names = self._staging_inventory()
        allowed_staged_names = {"index.sqlite3", pending_record.vector_file}
        if not staged_names <= allowed_staged_names:
            raise ValueError("reviewed-case transition staging inventory is invalid")
        if matches_previous:
            if current_manifest != previous:
                raise ValueError("reviewed-case pending transition orders manifest before SQLite")
            previous_records = self._records_for_manifest(previous, self.database)
            previous_vectors = previous["vectors"]
            following_vectors = following["vectors"]
            assert isinstance(previous_vectors, dict)
            assert isinstance(following_vectors, dict)
            previous_names = set(previous_vectors)
            following_names = set(following_vectors)
            actual_names = self._vector_inventory()
            if actual_names not in (previous_names, following_names):
                raise ValueError("reviewed-case vectors are outside the pending transition")
            self._verify_vector_files(previous_records, previous_vectors)

            final_vector_present = pending_record.vector_file in actual_names
            staged_vector_present = pending_record.vector_file in staged_names
            if final_vector_present:
                _load_vector(self.vector_root / pending_record.vector_file, pending_record)
            if staged_vector_present:
                _load_vector(staged_vector, pending_record)
            if final_vector_present and staged_vector_present:
                raise ValueError("reviewed-case pending vector exists in two states")

            if "index.sqlite3" in staged_names:
                staged_records = self._records_for_manifest(following, staged_database)
                if not self._contains_record(staged_records, pending_record):
                    raise ValueError("reviewed-case staged SQLite omits the pending record")
            if final_vector_present:
                _durable_unlink(self.vector_root / pending_record.vector_file)
            self._clear_transition(transition)
            return self._verify_snapshot(previous)

        if not self._contains_record(
            self._records_for_manifest(following, self.database), pending_record
        ):
            raise ValueError("reviewed-case committed SQLite omits the pending record")
        if self._vector_inventory() != set(following["vectors"]):
            raise ValueError("reviewed-case committed vectors are outside the pending transition")
        self._verify_snapshot(following)
        if staged_names:
            raise ValueError("reviewed-case committed transition retained staged payloads")
        if current_manifest == previous:
            _atomic_json(self.manifest, following)
            if self._read_manifest() != following:
                raise ValueError("reviewed-case recovered manifest did not publish exactly")
        self._clear_transition(transition)
        return self._verify_snapshot(following)

    def _ready_records(self) -> tuple[ReviewedCaseRecord, ...]:
        if os.path.lexists(self._transition):
            return self._recover_transition()
        records = self._verify()
        self._remove_staging()
        return records

    @staticmethod
    def _insert_record(connection: sqlite3.Connection, record: ReviewedCaseRecord) -> None:
        connection.execute(
            "INSERT INTO cases(case_id, model_state_hash, kind, graph_version_hash, "
            "concluded_at, record_hash, record_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record.case_id,
                record.model_state_hash,
                record.kind_key,
                record.graph_version_hash,
                record.concluded_at,
                record.record_hash,
                json.dumps(
                    record.model_dump(mode="json", by_alias=True),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )

    def _stage_transition(
        self,
        previous: Mapping[str, Any],
        updated: Sequence[ReviewedCaseRecord],
        record: ReviewedCaseRecord,
        raw_vectors: bytes,
    ) -> tuple[dict[str, Any], Path, Path]:
        self._remove_staging()
        self._staging.mkdir()
        staged_database = self._staging / "index.sqlite3"
        staged_vector = self._staging / record.vector_file
        try:
            previous_database = previous.get("database")
            if not self._valid_descriptor(previous_database):
                raise ValueError("reviewed-case previous SQLite identity is invalid")
            assert isinstance(previous_database, dict)
            _copy_verified_regular_file(
                self.database,
                staged_database,
                expected_bytes=previous_database["bytes"],
                expected_sha256=previous_database["sha256"],
            )
            connection = self._connection(read_only=False, database=staged_database)
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._insert_record(connection, record)
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()
            with staged_database.open("r+b") as stream:
                os.fsync(stream.fileno())
            _atomic_bytes(staged_vector, raw_vectors)
            following = self._manifest_document(updated, database=staged_database)
            staged_records = self._records_for_manifest(following, staged_database)
            if tuple(staged_records) != tuple(updated):
                raise ValueError("reviewed-case staged SQLite is not the exact next snapshot")
            _load_vector(staged_vector, record)
            logical: dict[str, Any] = {
                "schemaVersion": TRANSITION_SCHEMA_VERSION,
                "previousManifest": dict(previous),
                "nextManifest": following,
                "record": record.model_dump(mode="json", by_alias=True),
            }
            transition = {**logical, "transitionHash": canonical_sha256(logical)}
            self._validate_transition_document(transition)
            _atomic_json(self._transition, transition)
            written, _, _, _ = self._read_transition()
            if written != transition:
                raise ValueError("reviewed-case pending transition did not publish exactly")
            return transition, staged_database, staged_vector
        except BaseException:
            if not os.path.lexists(self._transition):
                self._remove_staging()
            raise

    @property
    def index_hash(self) -> str:
        with self._locked():
            self._ready_records()
            return str(self._read_manifest()["indexHash"])

    def index(
        self,
        metadata: Mapping[str, Any],
        vectors: CaseVectors,
        *,
        indexed_at: str,
        source_request_hash: str,
    ) -> tuple[ReviewedCaseRecord, bool, str]:
        with self._locked():
            records = self._ready_records()
            previous_manifest = self._read_manifest()
            case_id = str(metadata["caseId"])
            existing = next((record for record in records if record.case_id == case_id), None)
            normalized = _validated_vectors(vectors)
            vector_content_hash = _vector_content_hash(normalized)
            if existing is not None:
                if (
                    existing.source_request_hash != source_request_hash
                    or existing.vector_content_hash != vector_content_hash
                ):
                    raise ValueError("caseId is already bound to different reviewed evidence")
                return existing, True, str(previous_manifest["indexHash"])
            file_name = f"{hashlib.sha256(case_id.encode('utf-8')).hexdigest()}.npz"
            raw_vectors = _vector_bytes(normalized)
            vector_path = self.vector_root / file_name
            logical = {
                "schemaVersion": CASE_SCHEMA_VERSION,
                **dict(metadata),
                "vectorFile": file_name,
                "vectorSha256": hashlib.sha256(raw_vectors).hexdigest(),
                "vectorBytes": len(raw_vectors),
                "vectorContentHash": vector_content_hash,
                "indexedAt": indexed_at,
                "sourceRequestHash": source_request_hash,
            }
            record = ReviewedCaseRecord.model_validate(
                {**logical, "recordHash": canonical_sha256(logical)}
            )
            updated = tuple(sorted((*records, record), key=lambda item: item.case_id))
            transition, staged_database, staged_vector = self._stage_transition(
                previous_manifest,
                updated,
                record,
                raw_vectors,
            )
            following = transition["nextManifest"]
            assert isinstance(following, dict)
            _durable_replace(staged_vector, vector_path)
            _durable_replace(staged_database, self.database)
            _atomic_json(self.manifest, following)
            self._verify_snapshot(following)
            self._clear_transition(transition)
            return record, False, str(following["indexHash"])

    def record(self, case_id: str) -> ReviewedCaseRecord:
        with self._locked():
            records = self._ready_records()
            try:
                return next(record for record in records if record.case_id == case_id)
            except StopIteration as error:
                raise KeyError(case_id) from error

    def vectors(self, record: ReviewedCaseRecord) -> CaseVectors:
        with self._locked():
            self._ready_records()
            return _load_vector(self.vector_root / record.vector_file, record)

    def query(
        self,
        vectors: CaseVectors,
        *,
        model_state_hash: str,
        kind_key: str,
        limit: int,
        exclude_case_id: str | None = None,
    ) -> tuple[SimilarCase, ...]:
        with self._locked():
            if not 1 <= limit <= 100:
                raise ValueError("similar-case limit must be between 1 and 100")
            query = _validated_vectors(vectors)
            records = self._ready_records()
            values: list[SimilarCase] = []
            for record in records:
                if (
                    record.model_state_hash != model_state_hash
                    or record.kind_key != kind_key
                    or record.case_id == exclude_case_id
                ):
                    continue
                candidate = _load_vector(self.vector_root / record.vector_file, record)
                embedding = _similarity(query.embedding, candidate.embedding)
                structure = _similarity(query.structure, candidate.structure)
                modality = _similarity(query.modality, candidate.modality)
                score = 0.7 * embedding + 0.2 * structure + 0.1 * modality
                if not math.isfinite(score):
                    raise ValueError("similar-case score is non-finite")
                values.append(
                    SimilarCase(
                        record=record,
                        score=score,
                        embedding_score=embedding,
                        structure_score=structure,
                        modality_score=modality,
                    )
                )
            values.sort(key=lambda item: (-item.score, item.record.case_id))
            return tuple(values[:limit])


__all__ = [
    "CASE_SCHEMA_VERSION",
    "INDEX_SCHEMA_VERSION",
    "CaseKindEntry",
    "CaseVectors",
    "ReviewedCaseIndex",
    "ReviewedCaseRecord",
    "SimilarCase",
]
