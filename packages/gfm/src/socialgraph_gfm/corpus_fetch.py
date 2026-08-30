"""Trusted, local-only acquisition for the fixed ogbl-collab baseline corpus.

The public GFM corpus reader never loads Pickle.  This module is the narrow
bootstrap boundary that downloads the pinned official archive, creates the
local OGB cache, and invokes the API package's existing trusted converter in a
fresh process.  The resulting ``.sgfm.zip`` contains JSON and NPZ only.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .canonical import file_sha256
from .runtime import prepare_runtime_layout, require_storage_reserve

OGBL_COLLAB_LICENSE = "ODC-BY-1.0"
OGBL_COLLAB_URL = "https://snap.stanford.edu/ogb/data/linkproppred/collab.zip"
# OGB 1.3.6's release-v1 archive, observed from the official HTTPS endpoint.
OGBL_COLLAB_ARCHIVE_SHA256 = (
    "c5563198e041c338f0a78e11322bb2eb2de76b68f0e9ae3e3b6d6af2d8ca64cc"
)
OGBL_COLLAB_ARCHIVE_BYTES = 121_625_147


def _download_pinned_archive(destination: Path) -> None:
    if destination.is_file():
        if (
            destination.stat().st_size == OGBL_COLLAB_ARCHIVE_BYTES
            and file_sha256(destination) == OGBL_COLLAB_ARCHIVE_SHA256
        ):
            return
        raise ValueError(
            f"existing ogbl-collab archive does not match the pinned release: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        OGBL_COLLAB_URL,
        headers={"User-Agent": "SocialGraph-FM-ogbl-collab-baseline/1.0"},
    )
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".download", dir=destination.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(handle, "wb") as output, urllib.request.urlopen(
            request, timeout=60
        ) as response:
            if response.geturl().lower().split("?", 1)[0] != OGBL_COLLAB_URL.lower():
                raise ValueError("ogbl-collab download was redirected away from the pinned URL")
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        if byte_count != OGBL_COLLAB_ARCHIVE_BYTES:
            raise ValueError(
                f"ogbl-collab archive size mismatch: {byte_count} != "
                f"{OGBL_COLLAB_ARCHIVE_BYTES}"
            )
        if digest.hexdigest() != OGBL_COLLAB_ARCHIVE_SHA256:
            raise ValueError("ogbl-collab archive SHA-256 mismatch")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_extract_archive(archive_path: Path, cache_root: Path) -> Path:
    """Extract one verified archive into a cache root that must be empty.

    The caller creates a fresh, private conversion directory.  Refusing an
    existing OGB dataset directory is intentional: PyG loads ``processed/*.pt``
    with Pickle semantics, so reusing a pre-populated cache would let local
    files bypass the pinned archive hash.
    """

    target = cache_root / "ogbl_collab"
    if target.exists():
        raise ValueError(
            "refusing to reuse an existing ogbl-collab conversion cache"
        )
    cache_root_resolved = cache_root.resolve()
    with tempfile.TemporaryDirectory(prefix=".ogbl-collab-", dir=cache_root) as temporary_name:
        temporary = Path(temporary_name)
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members:
                raise ValueError("ogbl-collab archive is empty")
            total_size = 0
            for member in members:
                logical = PurePosixPath(member.filename)
                if logical.is_absolute() or ".." in logical.parts:
                    raise ValueError("ogbl-collab archive contains a path traversal entry")
                # Unix symlinks must not be followed during trusted extraction.
                if ((member.external_attr >> 16) & 0o170000) == 0o120000:
                    raise ValueError("ogbl-collab archive contains a symbolic link")
                total_size += member.file_size
                if total_size > 2 * 1024**3:
                    raise ValueError("ogbl-collab archive exceeds the extraction safety limit")
            archive.extractall(temporary)
        extracted = temporary / "collab"
        if not extracted.is_dir():
            raise ValueError("ogbl-collab archive does not contain the expected collab directory")
        required = (
            extracted / "raw" / "edge.csv.gz",
            extracted / "raw" / "node-feat.csv.gz",
            extracted / "split" / "time" / "train.pt",
            extracted / "RELEASE_v1.txt",
        )
        if any(not path.is_file() for path in required):
            raise ValueError("ogbl-collab archive is missing required release-v1 files")
        if extracted.resolve().parent != temporary.resolve():
            raise ValueError("ogbl-collab extraction escaped its temporary directory")
        if target.parent.resolve() != cache_root_resolved:
            raise ValueError("ogbl-collab cache target escaped the configured runtime root")
        shutil.move(str(extracted), str(target))
    return target


def _api_project_root() -> Path:
    gfm_project = Path(__file__).resolve().parents[2]
    candidate = gfm_project.parents[1] / "services" / "api"
    if not (candidate / "app" / "dataset_tools.py").is_file():
        raise FileNotFoundError(
            "the trusted API converter was not found at services/api"
        )
    return candidate


def _convert_with_api(cache_root: Path, package_path: Path) -> dict[str, Any]:
    if package_path.is_file():
        return {
            "converted": False,
            "reason": "safe package already exists",
            "package": str(package_path),
        }
    api_root = _api_project_root()
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(api_root) + (
        os.pathsep + existing_python_path if existing_python_path else ""
    )
    command = (
        sys.executable,
        "-m",
        "app.dataset_tools",
        "convert-ogbl-collab",
        "--input",
        str(cache_root),
        "--output",
        str(package_path),
        "--trust-pickle",
    )
    result = subprocess.run(
        command,
        cwd=api_root,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "trusted ogbl-collab conversion failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("trusted API converter returned invalid JSON") from error
    return {"converted": True, **payload}


def fetch_ogbl_collab(
    root: str | Path | None,
    *,
    accept_license: str,
) -> dict[str, Any]:
    """Acquire and locally convert the only accepted baseline corpus."""

    if accept_license != OGBL_COLLAB_LICENSE:
        raise ValueError(
            f"explicit --accept-license {OGBL_COLLAB_LICENSE} is required"
        )
    layout = prepare_runtime_layout(root, operation="fetch")
    archive_path = layout.raw_ogb / "collab.zip"
    package_path = layout.packages / "ogbl-collab.sgfm.zip"
    _download_pinned_archive(archive_path)
    # Never pass the persistent raw-data directory to the legacy Pickle
    # converter.  Build its cache from the hash-verified archive in a fresh
    # isolated directory and discard the cache after the safe JSON+NPZ package
    # has been written.  Any pre-existing raw/ogb/ogbl_collab directory is
    # therefore ignored and cannot cross the trust boundary.
    with tempfile.TemporaryDirectory(
        prefix="ogbl-collab-verified-", dir=layout.temporary
    ) as isolated_name:
        isolated_root = Path(isolated_name)
        cache_path = _safe_extract_archive(archive_path, isolated_root)
        conversion = _convert_with_api(isolated_root, package_path)
        cache_was_isolated = cache_path.parent == isolated_root
    run_reserve = require_storage_reserve(layout.root, operation="run")
    return {
        "ok": True,
        "corpusId": "ogbl-collab",
        "licenseAccepted": accept_license,
        "sourceUrl": OGBL_COLLAB_URL,
        "archive": {
            "path": str(archive_path),
            "sha256": OGBL_COLLAB_ARCHIVE_SHA256,
            "byteCount": OGBL_COLLAB_ARCHIVE_BYTES,
        },
        "conversionCache": {
            "isolated": cache_was_isolated,
            "retained": False,
            "provenance": "pinned-archive-sha256",
        },
        "package": {
            "path": str(package_path),
            "sha256": file_sha256(package_path),
            "byteCount": package_path.stat().st_size,
        },
        "conversion": conversion,
        "runStorageReserve": run_reserve,
    }
