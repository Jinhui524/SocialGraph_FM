"""Fail-closed installation of the deterministic prebuilt Web client."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .layout import RuntimeLayout


SCHEMA_VERSION = "socialgraph-fm.web-bundle/1.0"
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_EXPANDED_BYTES = 40 * 1024 * 1024
MAX_FILES = 2_000


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _safe_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("Web bundle contains an invalid member path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"Web bundle contains an unsafe member path: {value}")
    if ":" in value or value.startswith("/"):
        raise RuntimeError(f"Web bundle contains an unsafe member path: {value}")
    return path


def _load_manifest(layout: RuntimeLayout, archive: bytes) -> dict[str, Any]:
    try:
        manifest = json.loads(layout.web_bundle_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Prebuilt Web bundle manifest is missing or invalid") from error
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        raise RuntimeError("Prebuilt Web bundle manifest schema is unsupported")
    declaration = manifest.get("archive")
    files = manifest.get("files")
    if not isinstance(declaration, dict) or not isinstance(files, list):
        raise RuntimeError("Prebuilt Web bundle manifest structure is invalid")
    if (
        declaration.get("path") != "bundles/web/client.zip"
        or declaration.get("bytes") != len(archive)
        or declaration.get("sha256") != _sha256(archive)
        or len(archive) > MAX_ARCHIVE_BYTES
    ):
        raise RuntimeError("Prebuilt Web bundle archive failed integrity validation")
    if (
        manifest.get("fileCount") != len(files)
        or not 0 < len(files) <= MAX_FILES
        or manifest.get("inventoryHash") != _sha256(_canonical_json(files))
    ):
        raise RuntimeError("Prebuilt Web bundle inventory failed integrity validation")
    return manifest


def install_web_bundle(layout: RuntimeLayout) -> dict[str, Any]:
    if layout.web_bundle_archive.is_symlink() or layout.web_bundle_manifest.is_symlink():
        raise RuntimeError("Prebuilt Web bundle files cannot be symbolic links")
    try:
        archive_bytes = layout.web_bundle_archive.read_bytes()
    except OSError as error:
        raise RuntimeError("Prebuilt Web bundle is missing") from error
    manifest = _load_manifest(layout, archive_bytes)
    declared = manifest["files"]
    seen: set[str] = set()
    actual: list[dict[str, Any]] = []
    expanded = 0
    layout.assert_safe_var_path(layout.web_client_root)
    layout.assert_safe_var_path(layout.temp_root)
    layout.temp_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="web-", suffix=".stage", dir=layout.temp_root))
    backup = Path(
        tempfile.mkdtemp(prefix="web-backup-", suffix=".stage", dir=layout.temp_root)
    )
    backup.rmdir()
    layout.assert_safe_var_path(backup)
    switched = False
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            members = archive.infolist()
            if len(members) != len(declared):
                raise RuntimeError("Prebuilt Web bundle member count differs from its manifest")
            for member, expected in zip(members, declared, strict=True):
                path = _safe_path(member.filename)
                key = path.as_posix().casefold()
                if key in seen:
                    raise RuntimeError(f"Prebuilt Web bundle contains a duplicate path: {path}")
                seen.add(key)
                if member.is_dir():
                    raise RuntimeError(f"Prebuilt Web bundle contains an undeclared directory: {path}")
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RuntimeError(f"Prebuilt Web bundle contains a symbolic link: {path}")
                payload = archive.read(member)
                expanded += len(payload)
                if expanded > MAX_EXPANDED_BYTES:
                    raise RuntimeError("Prebuilt Web bundle exceeds its expanded byte limit")
                record = {
                    "path": path.as_posix(),
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
                if record != expected:
                    raise RuntimeError(f"Prebuilt Web bundle member failed validation: {path}")
                destination = staging.joinpath(*path.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)
                actual.append(record)
        if (
            manifest.get("totalBytes") != expanded
            or manifest.get("inventoryHash") != _sha256(_canonical_json(actual))
            or not (staging / "index.html").is_file()
        ):
            raise RuntimeError("Prebuilt Web bundle expanded inventory is invalid")
        if layout.web_client_root.exists():
            os.replace(layout.web_client_root, backup)
        try:
            layout.web_client_root.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, layout.web_client_root)
            switched = True
        except Exception:
            if backup.exists() and not layout.web_client_root.exists():
                os.replace(backup, layout.web_client_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return {
            "schemaVersion": SCHEMA_VERSION,
            "fileCount": len(actual),
            "totalBytes": expanded,
            "archiveSha256": manifest["archive"]["sha256"],
            "destination": str(layout.web_client_root),
        }
    finally:
        if not switched and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
