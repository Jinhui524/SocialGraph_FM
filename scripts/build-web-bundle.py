#!/usr/bin/env python3
"""Build or verify the deterministic prebuilt SocialGraph-FM Web bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "socialgraph-fm.web-bundle/1.0"
ARCHIVE_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SOURCE_FILES = (
    "index.html",
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "vite.config.mjs",
)
SOURCE_DIRECTORIES = ("public", "src")
TEXT_SOURCE_SUFFIXES = frozenset(
    {
        ".css",
        ".csv",
        ".gexf",
        ".graphml",
        ".html",
        ".js",
        ".json",
        ".md",
        ".mjs",
        ".svg",
        ".ts",
        ".tsv",
        ".tsx",
        ".txt",
    }
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _files_below(root: Path) -> tuple[Path, ...]:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"Expected a real directory: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeError(f"Web bundle input cannot contain links: {candidate}")
        if candidate.is_file():
            files.append(candidate)
    return tuple(sorted(files, key=lambda value: value.relative_to(root).as_posix()))


def _source_inventory(web_root: Path) -> tuple[dict[str, Any], ...]:
    paths: list[Path] = []
    for name in SOURCE_FILES:
        candidate = web_root / name
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(f"Required Web build input is missing or linked: {candidate}")
        paths.append(candidate)
    for name in SOURCE_DIRECTORIES:
        paths.extend(_files_below(web_root / name))
    inventory: list[dict[str, Any]] = []
    for path in sorted(set(paths), key=lambda value: value.relative_to(web_root).as_posix()):
        payload = path.read_bytes()
        if path.suffix.lower() in TEXT_SOURCE_SUFFIXES:
            payload = payload.replace(b"\r\n", b"\n")
        inventory.append(
            {
                "path": path.relative_to(web_root).as_posix(),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    return tuple(inventory)


def build_bundle(repository: Path) -> tuple[bytes, bytes]:
    web_root = repository / "apps" / "web"
    distribution = web_root / "dist" / "client"
    output_files = _files_below(distribution)
    if not output_files or not (distribution / "index.html").is_file():
        raise RuntimeError("apps/web/dist is missing; run npm --prefix apps/web run build first")

    archive_buffer = io.BytesIO()
    output_inventory: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for path in output_files:
            relative = path.relative_to(distribution).as_posix()
            payload = path.read_bytes()
            info = zipfile.ZipInfo(relative, date_time=ARCHIVE_TIMESTAMP)
            info.create_system = 0
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            output_inventory.append(
                {"path": relative, "bytes": len(payload), "sha256": _sha256(payload)}
            )

    archive_bytes = archive_buffer.getvalue()
    source_inventory = _source_inventory(web_root)
    source_hash = _sha256(_canonical_json(source_inventory))
    inventory_hash = _sha256(_canonical_json(output_inventory))
    manifest: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "archive": {
            "path": "bundles/web/client.zip",
            "bytes": len(archive_bytes),
            "sha256": _sha256(archive_bytes),
        },
        "sourceFileCount": len(source_inventory),
        "sourceHash": source_hash,
        "fileCount": len(output_inventory),
        "totalBytes": sum(int(item["bytes"]) for item in output_inventory),
        "inventoryHash": inventory_hash,
        "files": output_inventory,
    }
    return archive_bytes, _canonical_json(manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    repository = arguments.repository.expanduser().resolve(strict=True)
    archive_bytes, manifest_bytes = build_bundle(repository)
    destination = repository / "bundles" / "web"
    archive_path = destination / "client.zip"
    manifest_path = destination / "manifest.json"
    if arguments.check:
        mismatches = []
        for path, expected in (
            (archive_path, archive_bytes),
            (manifest_path, manifest_bytes),
        ):
            try:
                actual = path.read_bytes()
            except FileNotFoundError:
                mismatches.append(path.relative_to(repository).as_posix())
                continue
            if actual != expected:
                mismatches.append(path.relative_to(repository).as_posix())
        if mismatches:
            raise RuntimeError("Stale Web bundle: " + ", ".join(mismatches))
        print("Web bundle is current")
        return 0

    destination.mkdir(parents=True, exist_ok=True)
    temporary_archive = archive_path.with_suffix(".zip.tmp")
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    try:
        temporary_archive.write_bytes(archive_bytes)
        temporary_manifest.write_bytes(manifest_bytes)
        os.replace(temporary_archive, archive_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_archive.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    print(f"Wrote {archive_path.relative_to(repository)} ({len(archive_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
