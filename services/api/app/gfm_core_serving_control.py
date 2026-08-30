"""Torch-free API authority for strict serving metadata and monotonic acceptance."""

from __future__ import annotations

import hashlib
import os
import stat
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field, model_validator

from .gfm_hashing import canonical_json, canonical_sha256
from .gfm_core_schemas import (
    CoreServingCheckpointManifest,
    CoreServingModel,
    CoreServingRegistry,
    CoreServingControl,
    CoreServingGraphCatalog,
    StrictModel,
)

MAX_CONTROL_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_MODELS = 128
MAX_TOTAL_MANIFEST_BYTES = 16 * 1024 * 1024
_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
_CORE_HIGH_WATER_LOCK = threading.RLock()


def _CORE_CONTROL_CAPTURE_SEAM(_stage: str) -> None:
    return


def _CORE_PATH_WALK_SEAM(_stage: str, _path: Path) -> None:
    return


def _CORE_HIGH_WATER_SEAM(_stage: str, _path: Path) -> None:
    return


def _reject_link_components(path: str | Path) -> Path:
    candidate = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current /= part
        if current.exists() or current.is_symlink():
            details = current.lstat()
            if stat.S_ISLNK(details.st_mode) or bool(
                getattr(details, "st_file_attributes", 0) & _REPARSE_POINT
            ):
                raise ValueError(
                    "API serving metadata path contains a link or reparse point"
                )
    return candidate


def _safe_relative(value: str) -> str:
    parsed = PurePosixPath(value.replace("\\", "/"))
    if (
        not parsed.parts
        or parsed.is_absolute()
        or ".." in parsed.parts
        or ":" in value
    ):
        raise ValueError("API serving metadata reference must be safe and relative")
    return parsed.as_posix()


@dataclass(frozen=True)
class _CoreFileSnapshot:
    payload: bytes
    token: tuple[int, ...]


def _bounded_read(descriptor: int, *, size: int, max_bytes: int) -> bytes:
    if size < 1 or size > max_bytes:
        raise ValueError("API serving metadata is not a bounded regular file")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise ValueError("API serving metadata changed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ValueError("API serving metadata grew while reading")
    return b"".join(chunks)


def _identity(path: Path) -> tuple[int, int]:
    details = path.lstat()
    return int(details.st_dev), int(details.st_ino)


def _read_posix(root: Path, parts: tuple[str, ...], *, max_bytes: int) -> _CoreFileSnapshot:
    paths = [
        root,
        *(root.joinpath(*parts[:index]) for index in range(1, len(parts) + 1)),
    ]
    before = tuple(_identity(path) for path in paths)
    _CORE_PATH_WALK_SEAM("before-open", paths[-1])
    descriptors: list[int] = []
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | no_follow
    try:
        descriptor = os.open(root, directory_flags)
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != before[0]:
            raise ValueError("API serving root changed during held walk")
        for index, part in enumerate(parts[:-1], start=1):
            descriptor = os.open(part, directory_flags, dir_fd=descriptors[-1])
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != before[index]:
                raise ValueError("API serving path component changed during held walk")
        final = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | no_follow,
            dir_fd=descriptors[-1],
        )
        descriptors.append(final)
        opened = os.fstat(final)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != before[-1]
        ):
            raise ValueError("API serving final file changed during held walk")
        _CORE_PATH_WALK_SEAM("after-open", paths[-1])
        if tuple(_identity(path) for path in paths) != before:
            raise ValueError("API serving path changed during held walk")
        payload = _bounded_read(final, size=opened.st_size, max_bytes=max_bytes)
        after = tuple(os.fstat(descriptor) for descriptor in descriptors)
        if tuple((item.st_dev, item.st_ino) for item in after) != before:
            raise ValueError("API serving held path changed while reading")
        final_after = after[-1]
        if (
            final_after.st_size,
            final_after.st_mtime_ns,
            final_after.st_ctime_ns,
        ) != (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns):
            raise ValueError("API serving final file changed while reading")
        if tuple(_identity(path) for path in paths) != before:
            raise ValueError("API serving lexical path changed while reading")
        return _CoreFileSnapshot(
            payload,
            (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            ),
        )
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


if os.name == "nt":  # pragma: win32 cover
    import ctypes
    from ctypes import wintypes

    class _FILE_INFO(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("created", wintypes.FILETIME),
            ("accessed", wintypes.FILETIME),
            ("written", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

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
    _GetInfo = _kernel32.GetFileInformationByHandle
    _GetInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_FILE_INFO)]
    _GetInfo.restype = wintypes.BOOL
    _GetFinalPath = _kernel32.GetFinalPathNameByHandleW
    _GetFinalPath.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _GetFinalPath.restype = wintypes.DWORD
    _ReadFile = _kernel32.ReadFile
    _ReadFile.argtypes = [
        wintypes.HANDLE,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.LPVOID,
    ]
    _ReadFile.restype = wintypes.BOOL
    _FlushFileBuffers = _kernel32.FlushFileBuffers
    _FlushFileBuffers.argtypes = [wintypes.HANDLE]
    _FlushFileBuffers.restype = wintypes.BOOL
    _CloseHandle = _kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL
    _INVALID = ctypes.c_void_p(-1).value

    def _win_open(path: Path, *, final: bool, share: int = 1) -> int:
        handle = _CreateFileW(
            str(path),
            0x80 | (0x80000000 if final else 0),
            share,
            None,
            3,
            0x00200000 | 0x02000000,
            None,
        )
        if handle == _INVALID:
            raise OSError(ctypes.get_last_error(), "failed to open held API path")
        return handle

    def _win_info(handle: int) -> _FILE_INFO:
        info = _FILE_INFO()
        if not _GetInfo(handle, ctypes.byref(info)):
            raise OSError(ctypes.get_last_error(), "failed to inspect held API path")
        return info

    def _win_id(handle: int) -> tuple[int, int]:
        info = _win_info(handle)
        return info.volume, (info.index_high << 32) | info.index_low

    def _win_path_id(path: Path) -> tuple[int, int]:
        handle = _win_open(path, final=path.is_file(), share=7)
        try:
            return _win_id(handle)
        finally:
            _CloseHandle(handle)

    def _win_final_path(handle: int) -> Path:
        needed = _GetFinalPath(handle, None, 0, 0)
        buffer = ctypes.create_unicode_buffer(needed + 1)
        if not needed or not _GetFinalPath(handle, buffer, len(buffer), 0):
            raise OSError(ctypes.get_last_error(), "failed to resolve held API path")
        value = buffer.value
        return Path(
            "\\\\" + value[8:]
            if value.startswith("\\\\?\\UNC\\")
            else value.removeprefix("\\\\?\\")
        )

    def _read_windows(
        root: Path, parts: tuple[str, ...], *, max_bytes: int
    ) -> _CoreFileSnapshot:
        paths = [
            root,
            *(root.joinpath(*parts[:index]) for index in range(1, len(parts) + 1)),
        ]
        before = tuple(_win_path_id(path) for path in paths)
        lexical_before = tuple(_identity(path) for path in paths)
        _CORE_PATH_WALK_SEAM("before-open", paths[-1])
        handles: list[int] = []
        try:
            for index, path in enumerate(paths):
                handle = _win_open(path, final=index == len(paths) - 1)
                handles.append(handle)
                info = _win_info(handle)
                if info.attributes & _REPARSE_POINT or _win_id(handle) != before[index]:
                    raise ValueError("API serving path changed or became reparse point")
                if index < len(paths) - 1 and not info.attributes & 0x10:
                    raise ValueError("API serving parent is not a directory")
            final = handles[-1]
            info = _win_info(final)
            observed = os.path.normcase(os.path.abspath(_win_final_path(final)))
            trusted = os.path.normcase(os.path.abspath(root))
            expected = os.path.normcase(os.path.abspath(paths[-1]))
            if (
                observed != expected
                or os.path.commonpath((observed, trusted)) != trusted
            ):
                raise ValueError("held API path escaped trusted root")
            _CORE_PATH_WALK_SEAM("after-open", paths[-1])
            if tuple(_identity(path) for path in paths) != lexical_before:
                raise ValueError("API serving path changed during held walk")
            size = (info.size_high << 32) | info.size_low
            if size < 1 or size > max_bytes:
                raise ValueError("API serving metadata is not a bounded regular file")
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                length = min(1024 * 1024, remaining)
                buffer = ctypes.create_string_buffer(length)
                read = wintypes.DWORD()
                if (
                    not _ReadFile(final, buffer, length, ctypes.byref(read), None)
                    or not read.value
                ):
                    raise ValueError("API serving metadata changed while reading")
                chunks.append(buffer.raw[: read.value])
                remaining -= read.value
            after = tuple(_win_info(handle) for handle in handles)
            if tuple(_win_id(handle) for handle in handles) != before:
                raise ValueError("API serving held path changed while reading")
            final_after = after[-1]
            if (
                final_after.size_high,
                final_after.size_low,
                final_after.written.dwHighDateTime,
                final_after.written.dwLowDateTime,
            ) != (
                info.size_high,
                info.size_low,
                info.written.dwHighDateTime,
                info.written.dwLowDateTime,
            ):
                raise ValueError("API serving final file changed while reading")
            if tuple(_identity(path) for path in paths) != lexical_before:
                raise ValueError("API serving lexical path changed while reading")
            return _CoreFileSnapshot(
                b"".join(chunks),
                (
                    *_win_id(final),
                    size,
                    info.written.dwLowDateTime,
                    info.written.dwHighDateTime,
                ),
            )
        finally:
            for handle in reversed(handles):
                _CloseHandle(handle)

    def _flush_parent_directory(path: Path) -> None:
        handle = _CreateFileW(
            str(path),
            0x40000000 | 0x80,
            7,
            None,
            3,
            0x00200000 | 0x02000000,
            None,
        )
        if handle == _INVALID:
            raise OSError(
                ctypes.get_last_error(), "failed to open API high-water directory"
            )
        try:
            info = _win_info(handle)
            if info.attributes & _REPARSE_POINT or not info.attributes & 0x10:
                raise ValueError("API high-water parent must be a non-reparse directory")
            if not _FlushFileBuffers(handle):
                error = ctypes.get_last_error()
                if error not in {1, 50}:
                    raise OSError(error, "failed to flush API high-water directory")
        finally:
            _CloseHandle(handle)


else:

    def _flush_parent_directory(path: Path) -> None:
        directory = os.open(
            path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _read_confined_snapshot(
    root: Path, relative: str, *, max_bytes: int
) -> _CoreFileSnapshot:
    parts = PurePosixPath(_safe_relative(relative)).parts
    lexical_root = _reject_link_components(root)
    paths = [
        lexical_root,
        *(lexical_root.joinpath(*parts[:index]) for index in range(1, len(parts) + 1)),
    ]
    if any(not path.exists() or path.is_symlink() for path in paths):
        raise ValueError("API serving path must be existing and non-link")
    _reject_link_components(paths[-1])
    if os.name == "nt":
        return _read_windows(lexical_root, parts, max_bytes=max_bytes)
    return _read_posix(lexical_root, parts, max_bytes=max_bytes)


class CoreServingHighWater(StrictModel):
    schema_version: Literal["socialgraph-fm.core-api-serving-control-high-water/1.0"] = (
        Field(alias="schemaVersion")
    )
    control_generation: int = Field(alias="controlGeneration", ge=0)
    control_hash: str = Field(alias="controlHash", pattern=r"^[0-9a-f]{64}$")
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=r"^[0-9a-f]{64}$")
    catalog_generation: int = Field(alias="catalogGeneration", ge=0)
    catalog_hash: str = Field(alias="catalogHash", pattern=r"^[0-9a-f]{64}$")
    record_hash: str = Field(alias="recordHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_record_hash(self):
        if self.record_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        ):
            raise ValueError("API serving high-water record hash mismatch")
        return self


@dataclass(frozen=True)
class CoreServingManifestSnapshot:
    model: CoreServingModel
    source_bytes: bytes
    source_sha256: str
    manifest: CoreServingCheckpointManifest


@dataclass(frozen=True)
class CoreServingSnapshot:
    control_source_bytes: bytes
    control: CoreServingControl
    registry_source_bytes: bytes
    registry_source_sha256: str
    registry: CoreServingRegistry
    catalog_source_bytes: bytes
    catalog_source_sha256: str
    catalog: CoreServingGraphCatalog
    manifests: tuple[CoreServingManifestSnapshot, ...]

    @property
    def registry_hash(self) -> str:
        return self.control.registry.semantic_hash

    @property
    def catalog_hash(self) -> str:
        return self.control.catalog.semantic_hash

    def model(self, model_version_id: str) -> CoreServingModel:
        model = next(
            (
                item
                for item in self.registry.models
                if item.model_version_id == model_version_id
            ),
            None,
        )
        if model is None:
            raise LookupError("model version is not present in accepted API metadata")
        return model

    def manifest(self, model_version_id: str) -> CoreServingManifestSnapshot:
        snapshot = next(
            (
                item
                for item in self.manifests
                if item.model.model_version_id == model_version_id
            ),
            None,
        )
        if snapshot is None:
            raise LookupError("model manifest is not present in accepted API metadata")
        return snapshot


def _high_water(snapshot: CoreServingSnapshot) -> CoreServingHighWater:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-api-serving-control-high-water/1.0",
        "controlGeneration": snapshot.control.generation,
        "controlHash": snapshot.control.control_hash,
        "registryGeneration": snapshot.registry.generation,
        "registryHash": snapshot.registry_hash,
        "catalogGeneration": snapshot.catalog.generation,
        "catalogHash": snapshot.catalog_hash,
    }
    payload["recordHash"] = canonical_sha256(payload)
    return CoreServingHighWater.model_validate(payload)


def _atomic_private_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = (canonical_json(payload) + "\n").encode("utf-8")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _flush_parent_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class CoreServingControlStore:
    """Capture and durably accept strict dependency-neutral serving metadata."""

    def __init__(self, path: str | Path, *, high_water_root: str | Path) -> None:
        control_path = _reject_link_components(path)
        if not control_path.is_file():
            raise ValueError("API serving control must be an existing regular file")
        self.control_root = _reject_link_components(control_path.parent).resolve(
            strict=True
        )
        self.control_path = self.control_root / control_path.name
        root = _reject_link_components(high_water_root)
        root.mkdir(parents=True, exist_ok=True)
        self.high_water_root = _reject_link_components(root).resolve(strict=True)
        self.high_water_path = (
            self.high_water_root / "api-serving-control-high-water.json"
        )

    def _control(self) -> _CoreFileSnapshot:
        return _read_confined_snapshot(
            self.control_root, self.control_path.name, max_bytes=MAX_CONTROL_BYTES
        )

    def _capture(self, required_model_id: str | None) -> CoreServingSnapshot:
        control_a = self._control()
        control = CoreServingControl.model_validate_json(control_a.payload)
        registry_capture = _read_confined_snapshot(
            self.control_root,
            control.registry.relative_path,
            max_bytes=MAX_METADATA_BYTES,
        )
        catalog_capture = _read_confined_snapshot(
            self.control_root,
            control.catalog.relative_path,
            max_bytes=MAX_METADATA_BYTES,
        )
        if (
            hashlib.sha256(registry_capture.payload).hexdigest()
            != control.registry.sha256
        ):
            raise ValueError("API serving registry byte hash mismatch")
        if (
            hashlib.sha256(catalog_capture.payload).hexdigest()
            != control.catalog.sha256
        ):
            raise ValueError("API serving catalog byte hash mismatch")
        registry = CoreServingRegistry.model_validate_json(registry_capture.payload)
        catalog = CoreServingGraphCatalog.model_validate_json(catalog_capture.payload)
        if len(registry.models) > MAX_MODELS:
            raise ValueError("API serving registry exceeds the model-count limit")
        if (
            registry.generation != control.registry.generation
            or canonical_sha256(registry.model_dump(mode="python", by_alias=True))
            != control.registry.semantic_hash
        ):
            raise ValueError("API serving registry semantic binding mismatch")
        if (
            catalog.generation != control.catalog.generation
            or canonical_sha256(catalog.model_dump(mode="python", by_alias=True))
            != control.catalog.semantic_hash
        ):
            raise ValueError("API serving catalog semantic binding mismatch")
        manifests: list[CoreServingManifestSnapshot] = []
        total_manifest_bytes = 0
        for model in registry.models:
            remaining_manifest_bytes = MAX_TOTAL_MANIFEST_BYTES - total_manifest_bytes
            captured = _read_confined_snapshot(
                self.control_root,
                model.checkpoint.serving_manifest_relative_path,
                max_bytes=min(MAX_MANIFEST_BYTES, remaining_manifest_bytes),
            )
            source_hash = hashlib.sha256(captured.payload).hexdigest()
            total_manifest_bytes += len(captured.payload)
            if total_manifest_bytes > MAX_TOTAL_MANIFEST_BYTES:
                raise ValueError(
                    "API serving manifests exceed the aggregate byte limit"
                )
            if source_hash != model.checkpoint.serving_manifest_sha256:
                raise ValueError("API serving manifest byte hash mismatch")
            manifest = CoreServingCheckpointManifest.model_validate_json(
                captured.payload
            )
            if (
                manifest.task4_checkpoint_sha256 != model.checkpoint.sha256
                or manifest.adapter_domain != model.checkpoint.adapter_domain
                or manifest.node_classes != model.checkpoint.node_classes
                or manifest.multi_hot_buckets != model.checkpoint.multi_hot_buckets
                or manifest.task_heads != model.task_heads
                or any(
                    head.node_output_index is not None
                    and head.node_output_index >= manifest.node_classes
                    for head in manifest.task_heads
                )
            ):
                raise ValueError("API serving manifest does not match model descriptor")
            manifests.append(
                CoreServingManifestSnapshot(
                    model=model,
                    source_bytes=captured.payload,
                    source_sha256=source_hash,
                    manifest=manifest,
                )
            )
        if required_model_id is not None and all(
            model.model_version_id != required_model_id for model in registry.models
        ):
            raise LookupError("required model is not present in API serving metadata")
        _CORE_CONTROL_CAPTURE_SEAM("after-references")
        control_b = self._control()
        if control_b != control_a:
            raise ValueError("API serving control changed during bounded capture")
        return CoreServingSnapshot(
            control_source_bytes=control_a.payload,
            control=control,
            registry_source_bytes=registry_capture.payload,
            registry_source_sha256=hashlib.sha256(registry_capture.payload).hexdigest(),
            registry=registry,
            catalog_source_bytes=catalog_capture.payload,
            catalog_source_sha256=hashlib.sha256(catalog_capture.payload).hexdigest(),
            catalog=catalog,
            manifests=tuple(manifests),
        )

    def _accept(self, snapshot: CoreServingSnapshot) -> None:
        candidate = _high_water(snapshot)
        payload = candidate.model_dump(mode="python", by_alias=True)
        expected_bytes = (canonical_json(payload) + "\n").encode("utf-8")

        def verify_persisted() -> None:
            _CORE_HIGH_WATER_SEAM("before-persisted-reread", self.high_water_path)
            persisted_capture = _read_confined_snapshot(
                self.high_water_root,
                self.high_water_path.name,
                max_bytes=MAX_CONTROL_BYTES,
            )
            persisted = CoreServingHighWater.model_validate_json(
                persisted_capture.payload
            )
            if persisted != candidate or persisted_capture.payload != expected_bytes:
                raise ValueError(
                    "API serving high-water publication verification failed"
                )

        current: CoreServingHighWater | None = None
        current_capture: _CoreFileSnapshot | None = None
        if self.high_water_path.exists():
            current_capture = _read_confined_snapshot(
                self.high_water_root,
                self.high_water_path.name,
                max_bytes=MAX_CONTROL_BYTES,
            )
            current = CoreServingHighWater.model_validate_json(current_capture.payload)
        if current is not None:
            identities = (
                (
                    candidate.control_generation,
                    candidate.control_hash,
                    current.control_generation,
                    current.control_hash,
                ),
                (
                    candidate.registry_generation,
                    candidate.registry_hash,
                    current.registry_generation,
                    current.registry_hash,
                ),
                (
                    candidate.catalog_generation,
                    candidate.catalog_hash,
                    current.catalog_generation,
                    current.catalog_hash,
                ),
            )
            for (
                generation,
                value_hash,
                previous_generation,
                previous_hash,
            ) in identities:
                if generation < previous_generation:
                    raise ValueError(
                        "API serving metadata generation rollback rejected"
                    )
                if generation == previous_generation and value_hash != previous_hash:
                    raise ValueError(
                        "API serving metadata same-generation fork rejected"
                    )
            if candidate == current:
                if current_capture is None or current_capture.payload != expected_bytes:
                    raise ValueError(
                        "API serving high-water publication verification failed"
                    )
                _flush_parent_directory(self.high_water_root)
                verify_persisted()
                return
        _atomic_private_json(self.high_water_path, payload)
        verify_persisted()

    def acquire(self, required_model_id: str | None = None) -> CoreServingSnapshot:
        with _CORE_HIGH_WATER_LOCK:
            snapshot = self._capture(required_model_id)
            self._accept(snapshot)
            return snapshot


__all__ = [
    "CoreServingControlStore",
    "CoreServingManifestSnapshot",
    "CoreServingSnapshot",
]
