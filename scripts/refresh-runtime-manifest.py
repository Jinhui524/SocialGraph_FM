#!/usr/bin/env python3
"""Regenerate the canonical runtime bundle inventory from tracked content roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "socialgraph-fm.runtime-bundle/1.0"
CONTENT_ROOTS = (
    "bundles/models/socialgraph-global",
    "bundles/governance",
    "bundles/web",
    "examples/governance",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _role(relative: str) -> str:
    mappings = (
        ("bundles/models/socialgraph-global/", "model"),
        ("bundles/governance/knowledge/", "knowledge"),
        ("bundles/governance/reviewed-cases/", "reviewed_cases"),
        ("bundles/web/", "web"),
        ("examples/governance/russia/", "russia_example"),
        ("examples/governance/target-domain/", "target_domain_example"),
    )
    for prefix, role in mappings:
        if relative.startswith(prefix):
            return role
    raise RuntimeError(f"Runtime asset has no canonical role: {relative}")


def _model_identity(repository: Path) -> dict[str, str]:
    path = (
        repository
        / "bundles"
        / "models"
        / "socialgraph-global"
        / "exports"
        / "socialgraph-global"
        / "export-manifest.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    fields = ("releaseId", "modelVersionId", "modelVersionHash", "artifactHash", "corpusHash")
    if any(not isinstance(document.get(field), str) for field in fields):
        raise RuntimeError("Global export manifest has an invalid model identity")
    return {field: document[field] for field in fields}


def build_manifest(repository: Path) -> bytes:
    assets: list[dict[str, Any]] = []
    for root_name in CONTENT_ROOTS:
        root = repository / root_name
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"Runtime content root is missing or linked: {root_name}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise RuntimeError(f"Runtime content cannot contain links: {path}")
            if not path.is_file():
                continue
            relative = path.relative_to(repository).as_posix()
            assets.append(
                {
                    "path": relative,
                    "role": _role(relative),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    assets.sort(key=lambda item: str(item["path"]))
    if len({item["path"] for item in assets}) != len(assets):
        raise RuntimeError("Runtime content roots produced duplicate assets")
    inventory = "".join(
        f'{item["path"]}\t{item["bytes"]}\t{item["sha256"]}\t{item["role"]}\n'
        for item in assets
    )
    document = {
        "schemaVersion": SCHEMA_VERSION,
        "bundleVersion": "1.0.0",
        "contentRoots": list(CONTENT_ROOTS),
        "assets": assets,
        "fileCount": len(assets),
        "totalBytes": sum(int(item["bytes"]) for item in assets),
        "inventoryHash": hashlib.sha256(inventory.encode("utf-8")).hexdigest(),
        "model": _model_identity(repository),
    }
    return (
        json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    repository = arguments.repository.expanduser().resolve(strict=True)
    destination = repository / "bundles" / "runtime-manifest.json"
    expected = build_manifest(repository)
    if arguments.check:
        if destination.read_bytes() != expected:
            raise RuntimeError("bundles/runtime-manifest.json is stale")
        print("Runtime manifest is current")
        return 0
    temporary = destination.with_suffix(".json.tmp")
    try:
        temporary.write_bytes(expected)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {destination.relative_to(repository)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
