"""Bounded HTTPS acquisition and safe atomic source extraction."""

from __future__ import annotations

import gzip
import hashlib
import os
import json
import math
import subprocess
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections.abc import Callable, Collection, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Protocol

from .recipes import SourceRecipe, load_dataset_recipes


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    observed_sha256: str
    byte_count: int


ResponseContext = AbstractContextManager[BinaryIO]
OpenUrl = Callable[[urllib.request.Request, float], ResponseContext]


class _Readable(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _Writable(Protocol):
    def write(self, data: bytes) -> int: ...


class _AllowlistedRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, allowed_urls: Collection[str]) -> None:
        self._allowed_urls = frozenset(allowed_urls)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        normalized = urllib.parse.urljoin(req.full_url, newurl)
        if urllib.parse.urlsplit(normalized).scheme != "https":
            raise ValueError("redirect target must use HTTPS")
        if normalized not in self._allowed_urls:
            raise ValueError("redirect target is outside the explicit recipe allowlist")
        return super().redirect_request(req, fp, code, msg, headers, normalized)


def _default_open_url(allowed_urls: Collection[str]) -> OpenUrl:
    opener = urllib.request.build_opener(_AllowlistedRedirectHandler(allowed_urls))

    def open_url(request: urllib.request.Request, timeout: float) -> ResponseContext:
        return opener.open(request, timeout=timeout)  # type: ignore[return-value]

    return open_url


def _safe_runtime_root(runtime_root: Path) -> Path:
    root = runtime_root.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("runtime root must not be a filesystem root")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256_bounded(path: Path, maximum: int) -> str:
    if path.stat().st_size > maximum:
        raise FileExistsError("conflicting raw source already exists")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(
    *,
    recipe_id: str,
    source_id: str,
    runtime_root: Path,
    timeout_seconds: float = 30.0,
    open_url: OpenUrl | None = None,
) -> DownloadResult:
    """Download one explicitly recipe-allowlisted HTTPS source via bounded streaming."""

    if not _safe_identifier(recipe_id) or not _safe_identifier(source_id):
        raise ValueError("recipe and source identifiers must be safe catalog identifiers")
    recipes = load_dataset_recipes()
    if recipe_id not in recipes:
        raise ValueError("recipe identifier is not in the packaged catalog")
    recipe = recipes[recipe_id]
    by_id = {item.source_id: item for item in recipe.sources}
    if source_id not in by_id:
        raise ValueError("source identifier is not in the packaged recipe catalog")
    source = by_id[source_id]
    if urllib.parse.urlsplit(source.url).scheme != "https":
        raise ValueError("dataset downloads require HTTPS")
    root = _safe_runtime_root(runtime_root)
    target_directory = (root / "raw" / recipe_id / recipe.recipe_version).resolve()
    if not target_directory.is_relative_to(root):
        raise ValueError("download target escapes runtime root")
    target_directory.mkdir(parents=True, exist_ok=True)
    filename = Path(urllib.parse.unquote(urllib.parse.urlsplit(source.url).path)).name
    if not filename:
        raise ValueError("source URL must end in a filename")
    target = (target_directory / filename).resolve()
    if not target.is_relative_to(target_directory):
        raise ValueError("download filename escapes runtime target")
    staging = target_directory / f".{filename}.{uuid.uuid4().hex}.part"
    request = urllib.request.Request(source.url, headers={"User-Agent": "socialgraph-gfm/0.1"})
    transport = open_url or _default_open_url({source.url})
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with transport(request, timeout_seconds) as response:
            response_url = response.geturl()  # type: ignore[attr-defined]
            if urllib.parse.urlsplit(response_url).scheme != "https":
                raise ValueError("download response must use HTTPS")
            if response_url != source.url:
                raise ValueError("download response URL is outside the explicit recipe allowlist")
            content_length = response.headers.get("Content-Length")  # type: ignore[attr-defined]
            if content_length is not None and int(content_length) > source.max_bytes:
                raise ValueError("response Content-Length exceeds recipe maximum")
            with staging.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    byte_count += len(chunk)
                    if byte_count > source.max_bytes:
                        raise ValueError("response body exceeds recipe maximum")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        observed = digest.hexdigest()
        if source.expected_sha256 is not None and observed != source.expected_sha256:
            raise ValueError("download SHA-256 does not match recipe")
        if target.exists():
            existing = _sha256_bounded(target, source.max_bytes)
            if existing != observed:
                raise FileExistsError("conflicting raw source already exists")
            staging.unlink()
            return DownloadResult(path=target, observed_sha256=observed, byte_count=byte_count)
        os.replace(staging, target)
        return DownloadResult(path=target, observed_sha256=observed, byte_count=byte_count)
    except Exception:
        staging.unlink(missing_ok=True)
        raise


def _safe_identifier(value: str) -> bool:
    return (
        bool(value)
        and value not in {".", ".."}
        and not any(character in value for character in ("/", "\\", "\0"))
        and all(character.isalnum() or character in {"-", "_", " "} for character in value)
    )


def _validated_member_path(name: str) -> PurePosixPath:
    if "\\" in name or name.startswith("/"):
        raise ValueError("archive member path is unsafe")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("archive member path is unsafe")
    return path


def _copy_bounded(source: _Readable, target: _Writable, maximum: int) -> int:
    total = 0
    while chunk := source.read(1024 * 1024):
        total += len(chunk)
        if total > maximum:
            raise ValueError("expanded source exceeds configured maximum")
        target.write(chunk)
    return total


def _extract_zip(
    source_path: Path, staging: Path, expected: set[str], max_expanded_bytes: int
) -> None:
    with zipfile.ZipFile(source_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise ValueError("archive contains duplicate members")
        normalized = {str(_validated_member_path(name)) for name in names}
        if normalized != expected:
            raise ValueError("archive inventory does not match recipe")
        if sum(item.file_size for item in members) > max_expanded_bytes:
            raise ValueError("expanded source exceeds configured maximum")
        total = 0
        for member in members:
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("archive symlinks are forbidden")
            if member.flag_bits & 0x1:
                raise ValueError("encrypted archive members are forbidden")
            target = staging.joinpath(*PurePosixPath(member.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as stream, target.open("xb") as output:
                total += _copy_bounded(stream, output, max_expanded_bytes - total)


def _read_npy_header(stream: BinaryIO) -> tuple[tuple[int, ...], Any]:
    import numpy as np

    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError("unsupported NumPy array format")
    return tuple(int(value) for value in shape), dtype


def _validate_array_header(
    *, shape: tuple[int, ...], dtype: Any, max_array_elements: int, max_total_array_bytes: int
) -> int:
    if dtype.hasobject:
        raise ValueError("pickle/object NumPy arrays are forbidden")
    elements = math.prod(shape)
    if elements > max_array_elements:
        raise ValueError("NumPy array element limit exceeded")
    allocation = elements * int(dtype.itemsize)
    if allocation > max_total_array_bytes:
        raise ValueError("NumPy array allocation limit exceeded")
    return allocation


def safe_load_numpy_arrays(
    path: Path,
    *,
    expected_keys: Collection[str] | None,
    max_array_elements: int,
    max_total_array_bytes: int,
) -> Mapping[str, Any]:
    """Load numeric NumPy sources without pickle/object arrays or surprise keys."""

    import numpy as np

    if max_array_elements <= 0 or max_total_array_bytes <= 0:
        raise ValueError("NumPy limits must be positive")
    expected = None if expected_keys is None else set(expected_keys)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            names = [item.filename for item in members]
            if len(names) != len(set(names)):
                raise ValueError("NumPy archive contains duplicate inventory")
            if any("/" in name or "\\" in name or not name.endswith(".npy") for name in names):
                raise ValueError("NumPy archive inventory is unsafe")
            keys = {name[:-4] for name in names}
            if expected is not None and keys != expected:
                raise ValueError("NumPy archive inventory does not match expected keys")
            if sum(item.file_size for item in members) > max_total_array_bytes:
                raise ValueError("NumPy archive uncompressed size exceeds allocation limit")
            total = 0
            for member in members:
                with archive.open(member) as stream:
                    shape, dtype = _read_npy_header(stream)  # type: ignore[arg-type]
                total += _validate_array_header(
                    shape=shape,
                    dtype=dtype,
                    max_array_elements=max_array_elements,
                    max_total_array_bytes=max_total_array_bytes,
                )
            if total > max_total_array_bytes:
                raise ValueError("NumPy archive total allocation limit exceeded")
    else:
        with path.open("rb") as stream:
            shape, dtype = _read_npy_header(stream)
        _validate_array_header(
            shape=shape,
            dtype=dtype,
            max_array_elements=max_array_elements,
            max_total_array_bytes=max_total_array_bytes,
        )
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                keys = set(loaded.files)
                if len(keys) != len(loaded.files):
                    raise ValueError("NumPy archive contains duplicate inventory")
                if expected is not None and keys != expected:
                    raise ValueError("NumPy archive inventory does not match expected keys")
                arrays = {key: loaded[key] for key in sorted(keys)}
            finally:
                loaded.close()
        else:
            if expected_keys not in (None, {"array"}):
                raise ValueError("NumPy array inventory does not match expected keys")
            arrays = {"array": loaded}
    except ValueError as error:
        if "Object arrays cannot be loaded" in str(error):
            raise ValueError("pickle/object NumPy arrays are forbidden") from error
        raise
    if any(getattr(array.dtype, "hasobject", False) for array in arrays.values()):
        raise ValueError("pickle/object NumPy arrays are forbidden")
    return arrays


def safe_load_mat_arrays(
    path: Path,
    *,
    expected_keys: Collection[str],
    max_array_elements: int,
    max_worker_memory_bytes: int,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Parse compressed MAT data in a memory-bounded worker and load safe numeric output."""

    import numpy as np
    from scipy import sparse

    if set(expected_keys) != {"A", "local_info"}:
        raise ValueError("MAT worker supports the declared Facebook100 inventory only")
    with tempfile.TemporaryDirectory(prefix="socialgraph-mat-") as temporary:
        directory = Path(temporary)
        contract = directory / "contract.json"
        output = directory / "arrays.npz"
        contract.write_text(
            json.dumps(
                {
                    "input": str(path.resolve()),
                    "output": str(output),
                    "expectedKeys": sorted(expected_keys),
                    "maxArrayElements": max_array_elements,
                    "maxMemoryBytes": max_worker_memory_bytes,
                }
            ),
            encoding="utf-8",
        )
        process = subprocess.run(
            [sys.executable, "-m", "socialgraph_gfm.core.datasets.mat_worker", str(contract)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()
            raise ValueError(f"MAT worker rejected source: {detail}")
        arrays = safe_load_numpy_arrays(
            output,
            expected_keys={"a_col", "a_data", "a_row", "a_shape", "local_info"},
            max_array_elements=max_array_elements,
            max_total_array_bytes=max_worker_memory_bytes,
        )
        shape = tuple(int(value) for value in arrays["a_shape"])
        adjacency = sparse.coo_matrix(
            (arrays["a_data"], (arrays["a_row"], arrays["a_col"])), shape=shape
        )
        return {"A": adjacency, "local_info": np.asarray(arrays["local_info"])}


def extract_source_atomic(
    *,
    source_path: Path,
    source: SourceRecipe,
    target_directory: Path,
    max_expanded_bytes: int,
) -> Path:
    """Validate and extract a source to staging, then atomically publish it."""

    if max_expanded_bytes <= 0:
        raise ValueError("expanded-size maximum must be positive")
    target = target_directory.resolve()
    if target.exists():
        raise FileExistsError(f"materialized source target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    expected = set(source.inventory)
    try:
        if source.archive_type == "zip":
            _extract_zip(source_path, staging, expected, max_expanded_bytes)
        elif source.archive_type == "gzip":
            if len(source.inventory) != 1:
                raise ValueError("gzip source inventory must contain one output")
            output_path = staging.joinpath(*_validated_member_path(source.inventory[0]).parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(source_path, "rb") as stream, output_path.open("xb") as output:
                _copy_bounded(stream, output, max_expanded_bytes)
        else:
            if len(source.inventory) != 1:
                raise ValueError("single-file source inventory must contain one output")
            output_path = staging.joinpath(*_validated_member_path(source.inventory[0]).parts)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with source_path.open("rb") as stream, output_path.open("xb") as output:
                _copy_bounded(stream, output, max_expanded_bytes)
            if source.archive_type in {"npy", "npz"}:
                safe_load_numpy_arrays(
                    output_path,
                    expected_keys=None,
                    max_array_elements=max_expanded_bytes,
                    max_total_array_bytes=max_expanded_bytes,
                )
            elif source.archive_type == "mat":
                if source_path.read_bytes()[:20] != b"MATLAB 5.0 MAT-file ":
                    raise ValueError("MAT source does not have a supported MATLAB 5 header")
        os.replace(staging, target)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


__all__ = [
    "DownloadResult",
    "download_source",
    "extract_source_atomic",
    "safe_load_mat_arrays",
    "safe_load_numpy_arrays",
]
