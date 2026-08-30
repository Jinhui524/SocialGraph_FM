from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import uuid
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "socialgraph-fm-dataset-store-audit/1.0"
BACKUP_SCHEMA = "socialgraph-fm-dataset-store-backup/1.0"
SUPPORTED_SCHEMAS = {"2.1", "2.2"}
LEGACY_SCHEMAS = {"1.0", "2.0"}
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_COLUMNS = {
    "id",
    "dataset_name",
    "checksum",
    "canonical_graph_hash",
    "scope",
    "created_at",
    "artifact_json",
    "tensor_path",
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_read_only(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True, timeout=30)
    connection.execute("PRAGMA query_only=ON")
    return connection


def _schema_version(payload: dict[str, Any]) -> tuple[str | None, list[str]]:
    direct = payload.get("schemaVersion")
    if isinstance(direct, str) and direct:
        return direct, []
    raw_manifest = payload.get("rawManifest")
    if isinstance(raw_manifest, dict):
        inferred = raw_manifest.get("schemaVersion")
        if isinstance(inferred, str) and inferred:
            return inferred, ["SCHEMA_INFERRED_FROM_RAW_MANIFEST"]
    return None, []


def _safe_store_path(root: Path, stored_path: str) -> Path | None:
    candidate_path = Path(stored_path)
    candidate = (
        candidate_path.resolve()
        if candidate_path.is_absolute()
        else (root / candidate_path).resolve()
    )
    if candidate == root or root in candidate.parents:
        return candidate
    return None


def _validate_current_contract(payload: dict[str, Any]) -> str | None:
    try:
        project_root = str(Path(__file__).resolve().parents[1])
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from app.dataset_schemas import DatasetArtifact

        DatasetArtifact.model_validate(payload)
    except (ImportError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {str(exc)[:500]}"
    return None


def _duplicate_identity(payload: dict[str, Any]) -> str:
    raw_manifest = payload.get("rawManifest")
    raw_manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
    handoff = raw_manifest.get("graphVersionHandoff")
    handoff = handoff if isinstance(handoff, dict) else {}
    return _canonical_hash(
        {
            "datasetName": payload.get("datasetName"),
            "sourceFormat": payload.get("sourceFormat"),
            "checksum": payload.get("checksum"),
            "canonicalGraphHash": payload.get("canonicalGraphHash"),
            "contentHash": payload.get("contentHash"),
            "sourceChecksum": raw_manifest.get("sourceChecksum"),
            "graphVersionId": handoff.get("graphVersionId"),
            "graphContentHash": handoff.get("contentHash"),
            "buildSpecHash": handoff.get("buildSpecHash"),
            "selectedDatasetManifest": raw_manifest.get("selectedDatasetManifest"),
        }
    )


def _is_repeated_character_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and len(set(value.lower())) == 1


def _test_candidate_reasons(
    payload: dict[str, Any],
    *,
    duplicate_count: int,
) -> tuple[list[str], str | None]:
    reasons: list[str] = []
    confidence: str | None = None
    raw_manifest = payload.get("rawManifest")
    raw_manifest = raw_manifest if isinstance(raw_manifest, dict) else {}
    provenance_text = json.dumps(
        {
            "sourcePath": raw_manifest.get("sourcePath"),
            "sourceFiles": raw_manifest.get("sourceFiles"),
        },
        ensure_ascii=False,
    ).lower()
    if any(
        marker in provenance_text
        for marker in ("pytest-of-", "pytest-", "testserver", "\\tmp\\", "/tmp/")
    ):
        reasons.append("EXPLICIT_TEST_PROVENANCE")
        confidence = "high"

    handoff = raw_manifest.get("graphVersionHandoff")
    if isinstance(handoff, dict) and any(
        _is_repeated_character_hash(handoff.get(field))
        for field in ("contentHash", "buildSpecHash")
    ):
        reasons.append("SYNTHETIC_REPEATED_CHARACTER_HASH")
        confidence = "high"

    package_manifest = raw_manifest.get("packageManifest")
    if isinstance(package_manifest, dict):
        datasets = package_manifest.get("datasets")
        if isinstance(datasets, list):
            names = {
                str(item.get("name", "")).lower()
                for item in datasets
                if isinstance(item, dict)
            }
            if names == {"alpha", "beta"}:
                reasons.append("SYNTHETIC_ALPHA_BETA_PACKAGE")
                confidence = "high"

    source_format = str(payload.get("sourceFormat") or raw_manifest.get("sourceFormat") or "")
    if duplicate_count >= 3 and source_format in {
        "graph_npz",
        "socialgraph_dataset_package",
        "graph_version_target_domain",
    }:
        reasons.append("REPEATED_FIXTURE_LIKE_IDENTITY")
        confidence = confidence or "medium"
    return reasons, confidence


def _table_names(connection: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]


def _table_count(connection: sqlite3.Connection, table: str) -> int:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
        raise ValueError("unsafe SQLite table name")
    return int(connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def audit_store(root: Path) -> dict[str, Any]:
    """Inspect a DatasetStore without instantiating it or opening SQLite for writes."""

    started_at = datetime.now(UTC).isoformat()
    root = root.expanduser().resolve()
    database_path = root / "datasets.sqlite3"
    if not database_path.is_file():
        raise FileNotFoundError(f"DatasetStore database not found: {database_path}")

    store_issues: list[dict[str, str]] = []
    items: list[dict[str, Any]] = []
    with _connect_read_only(database_path) as connection:
        tables = _table_names(connection)
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        table_counts = {table: _table_count(connection, table) for table in tables}
        if "dataset_artifacts" not in tables:
            store_issues.append(
                {
                    "severity": "error",
                    "code": "ARTIFACT_TABLE_MISSING",
                    "message": "dataset_artifacts table does not exist",
                }
            )
            rows: list[sqlite3.Row] = []
        else:
            columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(dataset_artifacts)")
            }
            missing_columns = sorted(REQUIRED_COLUMNS - columns)
            if missing_columns:
                store_issues.append(
                    {
                        "severity": "error",
                        "code": "ARTIFACT_COLUMNS_MISSING",
                        "message": ", ".join(missing_columns),
                    }
                )
                rows = []
            else:
                connection.row_factory = sqlite3.Row
                rows = list(
                    connection.execute(
                        """
                        SELECT id, dataset_name, checksum, canonical_graph_hash,
                               scope, created_at, artifact_json, tensor_path
                        FROM dataset_artifacts
                        ORDER BY created_at, id
                        """
                    )
                )

    parsed_rows: list[tuple[sqlite3.Row, dict[str, Any] | None, str | None]] = []
    identities: Counter[str] = Counter()
    for row in rows:
        try:
            payload = json.loads(row["artifact_json"])
            if not isinstance(payload, dict):
                raise TypeError("artifact_json root is not an object")
            identity = _duplicate_identity(payload)
            identities[identity] += 1
            parsed_rows.append((row, payload, identity))
        except (json.JSONDecodeError, TypeError) as exc:
            parsed_rows.append((row, None, f"PARSE_ERROR:{type(exc).__name__}"))

    referenced_artifact_ids: set[str] = set()
    for row, payload, parsed_identity in parsed_rows:
        artifact_id = str(row["id"])
        referenced_artifact_ids.add(artifact_id)
        reasons: list[str] = []
        status = "quarantined"
        schema_version: str | None = None
        dataset_name = row["dataset_name"]
        test_reasons: list[str] = []
        test_confidence: str | None = None

        tensor = _safe_store_path(root, str(row["tensor_path"]))
        if tensor is None:
            reasons.append("TENSOR_PATH_ESCAPES_STORE")
        elif not tensor.is_file():
            reasons.append("TENSOR_FILE_MISSING")
        elif not zipfile.is_zipfile(tensor):
            reasons.append("TENSOR_FILE_NOT_SAFE_NPZ_CONTAINER")

        if payload is None:
            reasons.append("ARTIFACT_JSON_INVALID")
        else:
            schema_version, schema_reasons = _schema_version(payload)
            reasons.extend(schema_reasons)
            payload_id = payload.get("id")
            if payload_id != artifact_id:
                reasons.append("ARTIFACT_ID_MISMATCH")
            if payload.get("checksum") != row["checksum"]:
                reasons.append("SOURCE_CHECKSUM_DB_MISMATCH")
            if payload.get("canonicalGraphHash") != row["canonical_graph_hash"]:
                reasons.append("CANONICAL_GRAPH_HASH_DB_MISMATCH")

            artifact_file = root / "artifacts" / artifact_id / "artifact.json"
            if artifact_file.is_file():
                try:
                    persisted_payload = json.loads(artifact_file.read_text(encoding="utf-8"))
                    if persisted_payload != payload:
                        reasons.append("ARTIFACT_JSON_FILE_DB_MISMATCH")
                except (OSError, json.JSONDecodeError):
                    reasons.append("ARTIFACT_JSON_FILE_INVALID")
            elif schema_version in SUPPORTED_SCHEMAS:
                reasons.append("ARTIFACT_JSON_FILE_MISSING")

            fatal_integrity = any(
                code
                in {
                    "TENSOR_PATH_ESCAPES_STORE",
                    "TENSOR_FILE_MISSING",
                    "TENSOR_FILE_NOT_SAFE_NPZ_CONTAINER",
                    "ARTIFACT_ID_MISMATCH",
                    "SOURCE_CHECKSUM_DB_MISMATCH",
                    "CANONICAL_GRAPH_HASH_DB_MISMATCH",
                    "ARTIFACT_JSON_FILE_DB_MISMATCH",
                    "ARTIFACT_JSON_FILE_INVALID",
                    "ARTIFACT_JSON_FILE_MISSING",
                }
                for code in reasons
            )
            if not fatal_integrity and schema_version in LEGACY_SCHEMAS:
                status = "needs-reimport"
                reasons.append("LEGACY_SCHEMA_REQUIRES_NEW_ARTIFACT_ID")
            elif not fatal_integrity and schema_version in SUPPORTED_SCHEMAS:
                contract_error = _validate_current_contract(payload)
                if contract_error is None:
                    status = "compatible"
                else:
                    reasons.append("CURRENT_SCHEMA_CONTRACT_INVALID")
                    reasons.append(contract_error)
            elif not fatal_integrity:
                reasons.append("SCHEMA_UNSUPPORTED_OR_MISSING")

            duplicate_count = identities.get(parsed_identity or "", 1)
            test_reasons, test_confidence = _test_candidate_reasons(
                payload,
                duplicate_count=duplicate_count,
            )

        item = {
            "artifactId": artifact_id,
            "datasetName": dataset_name,
            "schemaVersion": schema_version,
            "structuralStatus": status,
            "testCandidate": bool(test_reasons),
            "testCandidateConfidence": test_confidence,
            "reasons": reasons,
            "testCandidateReasons": test_reasons,
            "tensorPath": str(row["tensor_path"]),
        }
        items.append(item)

    artifacts_root = root / "artifacts"
    directory_ids = {
        path.name for path in artifacts_root.iterdir() if path.is_dir()
    } if artifacts_root.is_dir() else set()
    orphan_directories = sorted(directory_ids - referenced_artifact_ids)
    missing_directories = sorted(referenced_artifact_ids - directory_ids)
    if orphan_directories:
        store_issues.append(
            {
                "severity": "warning",
                "code": "ORPHAN_ARTIFACT_DIRECTORIES",
                "message": f"{len(orphan_directories)} unreferenced artifact directories require review",
            }
        )
    if missing_directories:
        store_issues.append(
            {
                "severity": "error",
                "code": "REFERENCED_ARTIFACT_DIRECTORIES_MISSING",
                "message": f"{len(missing_directories)} referenced artifact directories are missing",
            }
        )

    categories = {
        "compatible": [item for item in items if item["structuralStatus"] == "compatible"],
        "needs-reimport": [
            item for item in items if item["structuralStatus"] == "needs-reimport"
        ],
        "quarantined": [item for item in items if item["structuralStatus"] == "quarantined"],
        # This is deliberately an overlapping review list; it never authorizes deletion.
        "test-candidate": [item for item in items if item["testCandidate"]],
    }
    counts = {name: len(values) for name, values in categories.items()}
    audit_identity = {
        "databaseUserVersion": user_version,
        "items": [
            {
                "artifactId": item["artifactId"],
                "structuralStatus": item["structuralStatus"],
                "schemaVersion": item["schemaVersion"],
                "testCandidate": item["testCandidate"],
                "reasons": item["reasons"],
            }
            for item in items
        ],
        "orphanDirectories": orphan_directories,
        "missingDirectories": missing_directories,
    }
    return {
        "schemaVersion": REPORT_SCHEMA,
        "generatedAt": started_at,
        "mode": "read-only",
        "mutationsPerformed": False,
        "storeRoot": str(root),
        "database": {
            "path": str(database_path),
            "bytes": database_path.stat().st_size,
            "userVersion": user_version,
            "walPresent": database_path.with_name(database_path.name + "-wal").exists(),
            "tables": tables,
            "tableCounts": table_counts,
        },
        "status": "review-required"
        if counts["needs-reimport"] or counts["quarantined"] or store_issues
        else "compatible",
        "counts": counts,
        "categories": categories,
        "storeIssues": store_issues,
        "orphanArtifactDirectories": orphan_directories,
        "missingArtifactDirectories": missing_directories,
        "auditHash": _canonical_hash(audit_identity),
        "safety": {
            "testCandidateIsAdvisoryOnly": True,
            "automaticDeletionAllowed": False,
            "legacyArtifactsMustNotBeOverwritten": True,
        },
    }


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _copy_sqlite_snapshot(source_path: Path, destination_path: Path) -> None:
    with (
        _connect_read_only(source_path) as source,
        sqlite3.connect(destination_path) as destination,
    ):
        source.backup(destination)


def backup_store(root: Path, destination_parent: Path) -> dict[str, Any]:
    """Create a new, non-overwriting snapshot; this is never called by audit mode."""

    root = root.expanduser().resolve()
    destination_parent = destination_parent.expanduser().resolve()
    if destination_parent == root or root in destination_parent.parents:
        raise ValueError("backup destination must not be inside the source DatasetStore")
    database_path = root / "datasets.sqlite3"
    if not database_path.is_file():
        raise FileNotFoundError(f"DatasetStore database not found: {database_path}")

    destination_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = destination_parent / f"dataset-store-backup-{stamp}-{uuid.uuid4().hex[:8]}"
    destination.mkdir(exist_ok=False)
    try:
        _copy_sqlite_snapshot(database_path, destination / "datasets.sqlite3")
        skipped_database_files = {
            database_path.name,
            database_path.name + "-wal",
            database_path.name + "-shm",
        }
        for source in root.iterdir():
            if source.name in skipped_database_files:
                continue
            target = destination / source.name
            if source.is_dir():
                shutil.copytree(source, target)
            elif source.is_file():
                shutil.copy2(source, target)

        audit = audit_store(destination)
        files: list[dict[str, str | int]] = []
        for path in sorted(value for value in destination.rglob("*") if value.is_file()):
            relative = path.relative_to(destination).as_posix()
            files.append(
                {
                    "path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        manifest = {
            "schemaVersion": BACKUP_SCHEMA,
            "createdAt": datetime.now(UTC).isoformat(),
            "sourceStore": str(root),
            "destination": str(destination),
            "databaseSnapshotMethod": "sqlite-backup-api",
            "fileCount": len(files),
            "totalBytes": sum(int(item["bytes"]) for item in files),
            "files": files,
            "auditHash": audit["auditHash"],
            "auditCounts": audit["counts"],
            "sourceMutated": False,
            "deletionsPerformed": False,
        }
        _write_json_atomic(destination / "backup-manifest.json", manifest)
        return {"backup": manifest, "audit": audit}
    except Exception as exc:
        _write_json_atomic(
            destination / "backup-status.json",
            {
                "schemaVersion": BACKUP_SCHEMA,
                "status": "incomplete",
                "errorType": type(exc).__name__,
                "message": str(exc)[:500],
                "deletionsPerformed": False,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only DatasetStore migration preview. Passing --backup-to explicitly "
            "creates a new, non-overwriting backup; it never modifies the source store."
        )
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--backup-to",
        type=Path,
        help="Explicitly create a full snapshot under this parent directory",
    )
    parser.add_argument(
        "--fail-on-quarantined",
        action="store_true",
        help="Return exit code 2 when quarantined rows exist",
    )
    args = parser.parse_args(argv)

    try:
        report = (
            backup_store(args.store, args.backup_to)
            if args.backup_to is not None
            else audit_store(args.store)
        )
    except (OSError, ValueError, sqlite3.Error) as exc:
        report = {
            "schemaVersion": REPORT_SCHEMA,
            "generatedAt": datetime.now(UTC).isoformat(),
            "status": "failed",
            "mode": "backup" if args.backup_to is not None else "read-only",
            "mutationsPerformed": False if args.backup_to is None else None,
            "error": {"type": type(exc).__name__, "message": str(exc)[:500]},
        }
        exit_code = 1
    else:
        audit = report.get("audit", report)
        exit_code = (
            2
            if args.fail_on_quarantined and audit["counts"]["quarantined"]
            else 0
        )

    if args.output:
        _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
