from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import numpy as np

from .dataset_schemas import (
    ArtifactReference,
    DatasetArtifact,
    DatasetArtifactDeletionImpact,
    DatasetArtifactRef,
    DatasetIssue,
    GraphDatasetBinding,
    OrphanArtifactDirectory,
    ResourceLifecycle,
    TrustedConversionJob,
)

logger = logging.getLogger(__name__)


class DatasetArtifactStore:
    """SQLite metadata + filesystem tensor storage for durable research artifacts."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if os.environ.get("PYTEST_CURRENT_TEST"):
            temporary_root = Path(tempfile.gettempdir()).resolve()
            if self.root != temporary_root and temporary_root not in self.root.parents:
                raise RuntimeError(
                    "测试 DatasetStore 必须位于系统临时目录；拒绝访问正式数据目录"
                )
        self.artifacts_root = self.root / "artifacts"
        self.staging_root = self.root / "staging"
        self.purge_recovery_root = self.root / "purge-recovery"
        self.jobs_root = self.root / "jobs"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.purge_recovery_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "datasets.sqlite3"
        self._lock = threading.RLock()
        self.last_list_issues: list[dict[str, str]] = []
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS dataset_artifacts (
                    id TEXT PRIMARY KEY,
                    dataset_name TEXT,
                    checksum TEXT NOT NULL,
                    canonical_graph_hash TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,
                    tensor_path TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trusted_conversion_jobs (
                    id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL,
                    job_json TEXT NOT NULL,
                    authorization_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS graph_handoff_tokens (
                    token_hash TEXT PRIMARY KEY,
                    graph_version_id TEXT NOT NULL,
                    graph_fact_hash TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS graph_dataset_bindings (
                    id TEXT PRIMARY KEY,
                    graph_version_id TEXT NOT NULL,
                    graph_fact_hash TEXT NOT NULL,
                    preparation_hash TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(graph_version_id, graph_fact_hash, preparation_hash),
                    FOREIGN KEY(artifact_id) REFERENCES dataset_artifacts(id)
                );
                CREATE TABLE IF NOT EXISTS artifact_lifecycle (
                    artifact_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK(status IN ('active', 'trashed')),
                    updated_at TEXT NOT NULL,
                    trashed_at TEXT,
                    FOREIGN KEY(artifact_id) REFERENCES dataset_artifacts(id) ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _safe_artifact_name(artifact_id: str) -> str:
        if (
            not artifact_id
            or artifact_id in {".", ".."}
            or Path(artifact_id).name != artifact_id
            or "/" in artifact_id
            or "\\" in artifact_id
        ):
            raise ValueError("ARTIFACT_ID_UNSAFE")
        return artifact_id

    def _artifact_path(self, artifact_id: str) -> Path:
        return self.artifacts_root / self._safe_artifact_name(artifact_id)

    def _purge_recovery_path(self, artifact_id: str) -> Path:
        return self.purge_recovery_root / self._safe_artifact_name(artifact_id)

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _write_artifact_directory(
        self,
        artifact_dir: Path,
        artifact: DatasetArtifact,
        arrays: dict[str, np.ndarray],
        *,
        attachments: dict[str, bytes] | None = None,
    ) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=False)
        try:
            tensor_path = artifact_dir / "graph.npz"
            temporary_tensor = artifact_dir / "graph.npz.tmp"
            with temporary_tensor.open("wb") as handle:
                np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            os.replace(temporary_tensor, tensor_path)

            self._write_json_atomic(
                artifact_dir / "artifact.json",
                artifact.model_dump(mode="json", by_alias=True),
            )
            self._write_json_atomic(
                artifact_dir / "raw-manifest.json", artifact.raw_manifest
            )
            self._write_json_atomic(
                artifact_dir / "derived-manifest.json",
                artifact.derived_manifest,
            )
            for relative_name, payload in (attachments or {}).items():
                normalized = PurePosixPath(relative_name.replace("\\", "/"))
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise ValueError("artifact attachment path is unsafe")
                destination = artifact_dir.joinpath(*normalized.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(payload)
                os.replace(temporary, destination)
        except Exception:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            raise

    @staticmethod
    def _insert_artifact_rows(
        connection: sqlite3.Connection,
        artifact: DatasetArtifact,
        *,
        tensor_path: Path,
        root: Path,
        now: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO dataset_artifacts
            (id, dataset_name, checksum, canonical_graph_hash, scope,
             created_at, artifact_json, tensor_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                artifact.dataset_name,
                artifact.checksum,
                artifact.canonical_graph_hash,
                artifact.scope,
                artifact.created_at.isoformat(),
                artifact.model_dump_json(by_alias=True),
                str(tensor_path.relative_to(root)),
            ),
        )
        connection.execute(
            """
            INSERT INTO artifact_lifecycle
            (artifact_id, status, updated_at, trashed_at)
            VALUES (?, 'active', ?, NULL)
            """,
            (artifact.id, now.isoformat()),
        )

    def save_artifact(
        self,
        artifact: DatasetArtifact,
        arrays: dict[str, np.ndarray],
        *,
        attachments: dict[str, bytes] | None = None,
    ) -> None:
        artifact_dir = self._artifact_path(artifact.id)
        with self._lock, self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM dataset_artifacts WHERE id = ?", (artifact.id,)
            ).fetchone():
                raise ValueError("DatasetArtifact 不可覆盖")
        self._write_artifact_directory(
            artifact_dir,
            artifact,
            arrays,
            attachments=attachments,
        )
        try:
            with self._lock, self._connect() as connection:
                self._insert_artifact_rows(
                    connection,
                    artifact,
                    tensor_path=artifact_dir / "graph.npz",
                    root=self.root,
                    now=datetime.now(UTC),
                )
        except Exception:
            shutil.rmtree(artifact_dir, ignore_errors=True)
            raise

    def stage_artifact(
        self,
        artifact: DatasetArtifact,
        arrays: dict[str, np.ndarray],
        *,
        attachments: dict[str, bytes] | None = None,
    ) -> None:
        staged_dir = self.staging_root / self._safe_artifact_name(artifact.id)
        with self._lock, self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM dataset_artifacts WHERE id = ?", (artifact.id,)
            ).fetchone():
                raise ValueError("DatasetArtifact 不可覆盖")
            if self._artifact_path(artifact.id).exists() or staged_dir.exists():
                raise ValueError("DatasetArtifact 暂存路径不可覆盖")
        self._write_artifact_directory(
            staged_dir,
            artifact,
            arrays,
            attachments=attachments,
        )

    def discard_staged_artifact(self, artifact_id: str) -> None:
        staged_dir = self.staging_root / self._safe_artifact_name(artifact_id)
        shutil.rmtree(staged_dir, ignore_errors=True)

    def get_artifact(
        self,
        artifact_id: str,
        *,
        include_trashed: bool = False,
    ) -> DatasetArtifact | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifacts.artifact_json
                FROM dataset_artifacts AS artifacts
                LEFT JOIN artifact_lifecycle AS lifecycle
                  ON lifecycle.artifact_id = artifacts.id
                WHERE artifacts.id = ?
                  AND (? OR COALESCE(lifecycle.status, 'active') = 'active')
                """,
                (artifact_id, include_trashed),
            ).fetchone()
        return DatasetArtifact.model_validate_json(row[0]) if row else None

    def list_artifacts(self, *, include_trashed: bool = False) -> list[DatasetArtifactRef]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT artifacts.artifact_json,
                       COALESCE(lifecycle.status, 'active')
                FROM dataset_artifacts AS artifacts
                LEFT JOIN artifact_lifecycle AS lifecycle
                  ON lifecycle.artifact_id = artifacts.id
                WHERE ? OR COALESCE(lifecycle.status, 'active') = 'active'
                ORDER BY artifacts.created_at DESC
                """,
                (include_trashed,),
            ).fetchall()
        result: list[DatasetArtifactRef] = []
        issues: list[dict[str, str]] = []
        for row in rows:
            try:
                artifact = DatasetArtifact.model_validate_json(row[0])
                result.append(
                    DatasetArtifactRef.model_validate(
                        {
                            "schemaVersion": artifact.schema_version,
                            "id": artifact.id,
                            "datasetName": artifact.dataset_name,
                            "checksum": artifact.checksum,
                            "canonicalGraphHash": artifact.canonical_graph_hash,
                            "contentHash": artifact.content_hash,
                            "manifestHash": artifact.manifest_hash,
                            "datasetRole": artifact.dataset_role,
                            "readinessStatus": "unchecked"
                            if artifact.schema_version in {"2.1", "2.2"}
                            else "legacy",
                            "scope": artifact.scope,
                            "profile": artifact.profile.model_dump(by_alias=True),
                            "createdAt": artifact.created_at,
                            "lifecycle": row[1],
                        }
                    )
                )
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raw_id = "unknown"
                try:
                    decoded = json.loads(row[0])
                    raw_id = str(decoded.get("id", "unknown")) if isinstance(decoded, dict) else "unknown"
                except (TypeError, json.JSONDecodeError):
                    pass
                issues.append({"artifactId": raw_id, "code": "ARTIFACT_ROW_INVALID"})
                logger.error("isolated invalid DatasetArtifact row id=%s error=%s", raw_id, exc)
        self.last_list_issues = issues
        return result

    @staticmethod
    def _lifecycle_from_row(artifact_id: str, row: tuple[str, str, str | None]) -> ResourceLifecycle:
        return ResourceLifecycle(
            artifactId=artifact_id,
            status=cast(Literal["active", "trashed"], row[0]),
            updatedAt=datetime.fromisoformat(row[1]),
            trashedAt=datetime.fromisoformat(row[2]) if row[2] else None,
        )

    def _get_lifecycle(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> ResourceLifecycle:
        artifact = connection.execute(
            "SELECT 1 FROM dataset_artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
        if artifact is None:
            raise ValueError("ARTIFACT_NOT_FOUND")
        row = connection.execute(
            """
            SELECT status, updated_at, trashed_at
            FROM artifact_lifecycle WHERE artifact_id = ?
            """,
            (artifact_id,),
        ).fetchone()
        if row is None:
            created_at = connection.execute(
                "SELECT created_at FROM dataset_artifacts WHERE id = ?",
                (artifact_id,),
            ).fetchone()[0]
            return ResourceLifecycle(
                artifactId=artifact_id,
                status="active",
                updatedAt=datetime.fromisoformat(created_at),
            )
        return self._lifecycle_from_row(artifact_id, row)

    def get_lifecycle(self, artifact_id: str) -> ResourceLifecycle:
        with self._lock, self._connect() as connection:
            return self._get_lifecycle(connection, artifact_id)

    def list_trashed_artifact_ids(self) -> list[str]:
        with self._lock, self._connect() as connection:
            return [
                row[0]
                for row in connection.execute(
                    """
                    SELECT artifact_id FROM artifact_lifecycle
                    WHERE status = 'trashed' ORDER BY artifact_id
                    """
                ).fetchall()
            ]

    @staticmethod
    def _canonical_impact_payload(
        *,
        artifact_id: str,
        lifecycle: str,
        blockers: list[ArtifactReference],
        dependents: list[ArtifactReference],
        preserved: list[str],
    ) -> bytes:
        value = {
            "artifactId": artifact_id,
            "lifecycle": lifecycle,
            "blockers": [
                item.model_dump(mode="json", by_alias=True)
                for item in sorted(blockers, key=lambda item: (item.kind, item.id))
            ],
            "dependents": [
                item.model_dump(mode="json", by_alias=True)
                for item in sorted(dependents, key=lambda item: (item.kind, item.id))
            ],
            "preserved": sorted(preserved),
        }
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _deletion_impact(
        self,
        connection: sqlite3.Connection,
        artifact_id: str,
    ) -> DatasetArtifactDeletionImpact:
        lifecycle = self._get_lifecycle(connection, artifact_id)
        artifact_row = connection.execute(
            "SELECT artifact_json FROM dataset_artifacts WHERE id = ?",
            (artifact_id,),
        ).fetchone()
        try:
            artifact = DatasetArtifact.model_validate_json(artifact_row[0])
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("ARTIFACT_ROW_INVALID") from exc
        blockers: list[ArtifactReference] = []
        dependents: list[ArtifactReference] = []
        for row in connection.execute(
            """
            SELECT id, graph_version_id, graph_fact_hash, preparation_hash
            FROM graph_dataset_bindings WHERE artifact_id = ? ORDER BY id
            """,
            (artifact_id,),
        ).fetchall():
            reference = ArtifactReference(
                kind="graph_dataset_binding",
                id=row[0],
                blocking=True,
                detail={
                    "graphVersionId": row[1],
                    "graphFactHash": row[2],
                    "preparationHash": row[3],
                },
            )
            blockers.append(reference)
            dependents.append(reference)
        embedded: dict[str, Any] = {}
        for index, training_reference in enumerate(
            [*artifact.training_refs, *([artifact.training_ref] if artifact.training_ref else [])]
        ):
            identity = training_reference.ref_hash or f"embedded-{index}"
            embedded.setdefault(identity, training_reference)
        for identity, training_reference in sorted(embedded.items()):
            dependents.append(
                ArtifactReference(
                    kind="embedded_training_ref",
                    id=identity,
                    blocking=False,
                    detail={
                        "schemaVersion": training_reference.schema_version,
                        "intendedUse": training_reference.intended_use,
                    },
                )
            )
        artifact_path = self._artifact_path(artifact_id)
        preserved = [
            f"artifact-directory:{artifact_id}"
            if artifact_path.is_dir()
            else f"missing-artifact-directory:{artifact_id}",
            *[f"embedded-training-ref:{item.id}" for item in dependents if not item.blocking],
        ]
        digest = hashlib.sha256(
            self._canonical_impact_payload(
                artifact_id=artifact_id,
                lifecycle=lifecycle.status,
                blockers=blockers,
                dependents=dependents,
                preserved=preserved,
            )
        ).hexdigest()
        return DatasetArtifactDeletionImpact(
            artifactId=artifact_id,
            lifecycle=lifecycle.status,
            blockers=blockers,
            dependents=dependents,
            preserved=preserved,
            impactHash=digest,
        )

    def deletion_impact(self, artifact_id: str) -> DatasetArtifactDeletionImpact:
        with self._lock, self._connect() as connection:
            return self._deletion_impact(connection, artifact_id)

    def set_lifecycle(self, artifact_id: str, status: str) -> ResourceLifecycle:
        if status not in {"active", "trashed"}:
            raise ValueError("LIFECYCLE_STATUS_INVALID")
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = self._get_lifecycle(connection, artifact_id)
            if current.status == status:
                return current
            now = datetime.now(UTC)
            trashed_at = now.isoformat() if status == "trashed" else None
            connection.execute(
                """
                INSERT INTO artifact_lifecycle
                (artifact_id, status, updated_at, trashed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    trashed_at = excluded.trashed_at
                """,
                (artifact_id, status, now.isoformat(), trashed_at),
            )
            return ResourceLifecycle(
                artifactId=artifact_id,
                status=cast(Literal["active", "trashed"], status),
                updatedAt=now,
                trashedAt=now if status == "trashed" else None,
            )

    def purge_artifact(
        self,
        artifact_id: str,
        *,
        expected_impact_hash: str,
        confirmation: str,
    ) -> bool:
        """Purge metadata and files; return True only if cleanup remains recoverable."""

        if confirmation != artifact_id[-8:]:
            raise ValueError("CONFIRMATION_MISMATCH")
        artifact_path = self._artifact_path(artifact_id)
        recovery_path = self._purge_recovery_path(artifact_id)
        moved = False
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    impact = self._deletion_impact(connection, artifact_id)
                    if impact.impact_hash != expected_impact_hash:
                        raise ValueError("REFERENCE_SET_CHANGED")
                    if impact.lifecycle != "trashed":
                        raise ValueError("ARTIFACT_NOT_TRASHED")
                    if impact.blockers:
                        raise ValueError("ARTIFACT_REFERENCED")
                    if recovery_path.exists():
                        raise ValueError("PURGE_RECOVERY_COLLISION")
                    if artifact_path.exists():
                        os.replace(artifact_path, recovery_path)
                        moved = True
                    cursor = connection.execute(
                        "DELETE FROM dataset_artifacts WHERE id = ?",
                        (artifact_id,),
                    )
                    if cursor.rowcount != 1:
                        raise ValueError("ARTIFACT_NOT_FOUND")
            except Exception:
                if moved and recovery_path.exists() and not artifact_path.exists():
                    try:
                        os.replace(recovery_path, artifact_path)
                    except OSError:
                        logger.exception(
                            "failed to restore artifact directory after purge rollback id=%s",
                            artifact_id,
                        )
                raise
        if not moved:
            return False
        try:
            shutil.rmtree(recovery_path)
        except OSError:
            logger.exception("purged metadata but file cleanup is pending id=%s", artifact_id)
            return True
        return False

    def list_orphan_artifacts(self) -> list[OrphanArtifactDirectory]:
        with self._lock, self._connect() as connection:
            registered = {
                row[0]
                for row in connection.execute("SELECT id FROM dataset_artifacts").fetchall()
            }
        result: list[OrphanArtifactDirectory] = []
        for source, root in (
            ("artifacts", self.artifacts_root),
            ("purge_recovery", self.purge_recovery_root),
        ):
            for path in sorted(root.iterdir(), key=lambda item: item.name):
                if not path.is_dir() or path.name in registered:
                    continue
                artifact_json = path / "artifact.json"
                tensor = path / "graph.npz"
                reason: str | None = None
                recoverable = artifact_json.is_file() and tensor.is_file()
                if not recoverable:
                    reason = "ARTIFACT_FILES_INCOMPLETE"
                else:
                    try:
                        artifact = DatasetArtifact.model_validate_json(
                            artifact_json.read_text(encoding="utf-8")
                        )
                        if artifact.id != path.name:
                            recoverable = False
                            reason = "ARTIFACT_ID_MISMATCH"
                    except (OSError, ValueError, TypeError, json.JSONDecodeError):
                        recoverable = False
                        reason = "ARTIFACT_MANIFEST_INVALID"
                result.append(
                    OrphanArtifactDirectory(
                        artifactId=path.name,
                        source=cast(Literal["artifacts", "purge_recovery"], source),
                        relativePath=str(path.relative_to(self.root)).replace("\\", "/"),
                        recoverable=recoverable,
                        reason=reason,
                    )
                )
        return result

    def recover_orphan_artifact(
        self,
        artifact_id: str,
    ) -> tuple[DatasetArtifact, ResourceLifecycle]:
        self._safe_artifact_name(artifact_id)
        destinations = [
            path
            for path in (
                self._artifact_path(artifact_id),
                self._purge_recovery_path(artifact_id),
            )
            if path.is_dir()
        ]
        if not destinations:
            raise ValueError("ORPHAN_NOT_FOUND")
        if len(destinations) != 1:
            raise ValueError("ORPHAN_AMBIGUOUS")
        source = destinations[0]
        try:
            artifact = DatasetArtifact.model_validate_json(
                (source / "artifact.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("ORPHAN_NOT_RECOVERABLE") from exc
        if artifact.id != artifact_id or not (source / "graph.npz").is_file():
            raise ValueError("ORPHAN_NOT_RECOVERABLE")
        target = self._artifact_path(artifact_id)
        moved = False
        with self._lock:
            try:
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    if connection.execute(
                        "SELECT 1 FROM dataset_artifacts WHERE id = ?", (artifact_id,)
                    ).fetchone():
                        raise ValueError("ARTIFACT_ALREADY_REGISTERED")
                    if source != target:
                        if target.exists():
                            raise ValueError("ORPHAN_TARGET_COLLISION")
                        os.replace(source, target)
                        moved = True
                    now = datetime.now(UTC)
                    connection.execute(
                        """
                        INSERT INTO dataset_artifacts
                        (id, dataset_name, checksum, canonical_graph_hash, scope,
                         created_at, artifact_json, tensor_path)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            artifact.id,
                            artifact.dataset_name,
                            artifact.checksum,
                            artifact.canonical_graph_hash,
                            artifact.scope,
                            artifact.created_at.isoformat(),
                            artifact.model_dump_json(by_alias=True),
                            str((target / "graph.npz").relative_to(self.root)),
                        ),
                    )
                    connection.execute(
                        """
                        INSERT INTO artifact_lifecycle
                        (artifact_id, status, updated_at, trashed_at)
                        VALUES (?, 'active', ?, NULL)
                        ON CONFLICT(artifact_id) DO UPDATE SET
                            status = 'active',
                            updated_at = excluded.updated_at,
                            trashed_at = NULL
                        """,
                        (artifact_id, now.isoformat()),
                    )
                    lifecycle = ResourceLifecycle(
                        artifactId=artifact_id,
                        status="active",
                        updatedAt=now,
                    )
            except Exception:
                if moved and target.exists() and not source.exists():
                    try:
                        os.replace(target, source)
                    except OSError:
                        logger.exception(
                            "failed to restore orphan directory after recovery rollback id=%s",
                            artifact_id,
                        )
                raise
        return artifact, lifecycle

    def create_handoff_reservation(
        self,
        *,
        token_hash: str,
        graph_version_id: str,
        graph_fact_hash: str,
        expires_at: datetime,
    ) -> None:
        with self._lock, self._connect() as connection:
            now = datetime.now(UTC).isoformat()
            connection.execute(
                "DELETE FROM graph_handoff_tokens WHERE expires_at <= ? AND consumed_at IS NULL",
                (now,),
            )
            connection.execute(
                """
                INSERT INTO graph_handoff_tokens
                (token_hash, graph_version_id, graph_fact_hash, expires_at, consumed_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (token_hash, graph_version_id, graph_fact_hash, expires_at.isoformat()),
            )

    def validate_handoff_token(
        self,
        *,
        token_hash: str,
        graph_version_id: str,
        graph_fact_hash: str,
        now: datetime,
        allow_consumed: bool = False,
    ) -> None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT graph_version_id, graph_fact_hash, expires_at, consumed_at
                FROM graph_handoff_tokens WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("HANDOFF_TOKEN_INVALID")
        if row[3] is not None and not allow_consumed:
            raise ValueError("HANDOFF_TOKEN_CONSUMED")
        if row[3] is None and datetime.fromisoformat(row[2]) <= now:
            raise ValueError("HANDOFF_TOKEN_EXPIRED")
        if row[0] != graph_version_id or row[1] != graph_fact_hash:
            raise ValueError("HANDOFF_TOKEN_IDENTITY_MISMATCH")

    def cancel_handoff_token(self, *, token_hash: str, now: datetime) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE graph_handoff_tokens SET consumed_at = ?
                WHERE token_hash = ? AND consumed_at IS NULL
                """,
                (now.isoformat(), token_hash),
            )
            if cursor.rowcount != 1:
                raise ValueError("HANDOFF_TOKEN_CONSUMED_OR_INVALID")

    def find_binding(
        self,
        *,
        graph_version_id: str,
        graph_fact_hash: str,
        preparation_hash: str,
    ) -> GraphDatasetBinding | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, graph_version_id, graph_fact_hash, artifact_id,
                       preparation_hash, created_at
                FROM graph_dataset_bindings
                WHERE graph_version_id = ? AND graph_fact_hash = ? AND preparation_hash = ?
                """,
                (graph_version_id, graph_fact_hash, preparation_hash),
            ).fetchone()
        if row is None:
            return None
        return GraphDatasetBinding(
            id=row[0], graphVersionId=row[1], graphFactHash=row[2], artifactId=row[3],
            preparationHash=row[4], createdAt=datetime.fromisoformat(row[5])
        )

    def resolve_graph_version_binding(self, graph_version_id: str) -> GraphDatasetBinding | None:
        """Resolve one immutable active graph identity without accepting caller hashes or paths."""

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT binding.id, binding.graph_version_id, binding.graph_fact_hash,
                       binding.artifact_id, binding.preparation_hash, binding.created_at
                FROM graph_dataset_bindings AS binding
                JOIN dataset_artifacts AS artifact ON artifact.id = binding.artifact_id
                LEFT JOIN artifact_lifecycle AS lifecycle
                  ON lifecycle.artifact_id = artifact.id
                WHERE binding.graph_version_id = ?
                  AND COALESCE(lifecycle.status, 'active') = 'active'
                ORDER BY binding.created_at, binding.id
                """,
                (graph_version_id,),
            ).fetchall()
        if not rows:
            return None
        if len({row[2] for row in rows}) != 1:
            raise ValueError("GRAPH_VERSION_IDENTITY_CONFLICT")
        row = rows[0]
        return GraphDatasetBinding(
            id=row[0],
            graphVersionId=row[1],
            graphFactHash=row[2],
            artifactId=row[3],
            preparationHash=row[4],
            createdAt=datetime.fromisoformat(row[5]),
        )

    def commit_binding(
        self,
        *,
        binding: GraphDatasetBinding,
        token_hash: str,
        now: datetime,
        allow_consumed: bool = False,
    ) -> GraphDatasetBinding:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT graph_version_id, graph_fact_hash, expires_at, consumed_at
                FROM graph_handoff_tokens WHERE token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None:
                raise ValueError("HANDOFF_TOKEN_INVALID")
            if row[3] is not None and not allow_consumed:
                raise ValueError("HANDOFF_TOKEN_CONSUMED")
            if row[3] is None and datetime.fromisoformat(row[2]) <= now:
                raise ValueError("HANDOFF_TOKEN_EXPIRED")
            if row[0] != binding.graph_version_id or row[1] != binding.graph_fact_hash:
                raise ValueError("HANDOFF_TOKEN_IDENTITY_MISMATCH")
            existing = connection.execute(
                """
                SELECT id, graph_version_id, graph_fact_hash, artifact_id,
                       preparation_hash, created_at
                FROM graph_dataset_bindings
                WHERE graph_version_id = ? AND graph_fact_hash = ? AND preparation_hash = ?
                """,
                (
                    binding.graph_version_id,
                    binding.graph_fact_hash,
                    binding.preparation_hash,
                ),
            ).fetchone()
            if existing is None:
                connection.execute(
                    """
                    INSERT INTO graph_dataset_bindings
                    (id, graph_version_id, graph_fact_hash, preparation_hash, artifact_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        binding.id,
                        binding.graph_version_id,
                        binding.graph_fact_hash,
                        binding.preparation_hash,
                        binding.artifact_id,
                        binding.created_at.isoformat(),
                    ),
                )
                resolved = binding
            else:
                resolved = GraphDatasetBinding(
                    id=existing[0], graphVersionId=existing[1], graphFactHash=existing[2],
                    artifactId=existing[3], preparationHash=existing[4],
                    createdAt=datetime.fromisoformat(existing[5])
                )
            connection.execute(
                """
                UPDATE graph_handoff_tokens SET consumed_at = ?
                WHERE token_hash = ? AND consumed_at IS NULL
                """,
                (now.isoformat(), token_hash),
            )
        return resolved

    def activate_staged_handoff(
        self,
        *,
        artifact: DatasetArtifact,
        binding: GraphDatasetBinding,
        token_hash: str,
        now: datetime,
    ) -> GraphDatasetBinding:
        """Atomically publish staged files, metadata, binding and token consumption.

        The filesystem cannot participate in SQLite's transaction, so files are
        moved first but are not considered active until the metadata and binding
        rows commit together. Any ordinary failure removes the moved directory;
        a process crash can leave only a recoverable, unregistered orphan.
        """

        staged_dir = self.staging_root / self._safe_artifact_name(artifact.id)
        artifact_dir = self._artifact_path(artifact.id)
        moved = False
        connection: sqlite3.Connection | None = None
        with self._lock:
            try:
                connection = self._connect()
                connection.execute("BEGIN IMMEDIATE")
                token_row = connection.execute(
                    """
                    SELECT graph_version_id, graph_fact_hash, expires_at, consumed_at
                    FROM graph_handoff_tokens WHERE token_hash = ?
                    """,
                    (token_hash,),
                ).fetchone()
                if token_row is None:
                    raise ValueError("HANDOFF_TOKEN_INVALID")
                if (
                    token_row[0] != binding.graph_version_id
                    or token_row[1] != binding.graph_fact_hash
                ):
                    raise ValueError("HANDOFF_TOKEN_IDENTITY_MISMATCH")

                identity_rows = connection.execute(
                    "SELECT DISTINCT graph_fact_hash FROM graph_dataset_bindings "
                    "WHERE graph_version_id = ?",
                    (binding.graph_version_id,),
                ).fetchall()
                if any(row[0] != binding.graph_fact_hash for row in identity_rows):
                    raise ValueError("GRAPH_VERSION_IDENTITY_CONFLICT")

                existing = connection.execute(
                    """
                    SELECT id, graph_version_id, graph_fact_hash, artifact_id,
                           preparation_hash, created_at
                    FROM graph_dataset_bindings
                    WHERE graph_version_id = ?
                      AND graph_fact_hash = ?
                      AND preparation_hash = ?
                    """,
                    (
                        binding.graph_version_id,
                        binding.graph_fact_hash,
                        binding.preparation_hash,
                    ),
                ).fetchone()
                if existing is not None:
                    if token_row[3] is None:
                        if datetime.fromisoformat(token_row[2]) <= now:
                            raise ValueError("HANDOFF_TOKEN_EXPIRED")
                        connection.execute(
                            "UPDATE graph_handoff_tokens SET consumed_at = ? WHERE token_hash = ?",
                            (now.isoformat(), token_hash),
                        )
                    connection.commit()
                    self.discard_staged_artifact(artifact.id)
                    return GraphDatasetBinding(
                        id=existing[0],
                        graphVersionId=existing[1],
                        graphFactHash=existing[2],
                        artifactId=existing[3],
                        preparationHash=existing[4],
                        createdAt=datetime.fromisoformat(existing[5]),
                    )

                if token_row[3] is not None:
                    raise ValueError("HANDOFF_TOKEN_CONSUMED_OR_INVALID")
                if datetime.fromisoformat(token_row[2]) <= now:
                    raise ValueError("HANDOFF_TOKEN_EXPIRED")
                if not staged_dir.is_dir():
                    raise ValueError("HANDOFF_STAGED_ARTIFACT_MISSING")
                if artifact_dir.exists():
                    raise ValueError("HANDOFF_ARTIFACT_PATH_CONFLICT")

                os.replace(staged_dir, artifact_dir)
                moved = True
                self._insert_artifact_rows(
                    connection,
                    artifact,
                    tensor_path=artifact_dir / "graph.npz",
                    root=self.root,
                    now=now,
                )
                connection.execute(
                    """
                    INSERT INTO graph_dataset_bindings
                    (id, graph_version_id, graph_fact_hash, preparation_hash,
                     artifact_id, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        binding.id,
                        binding.graph_version_id,
                        binding.graph_fact_hash,
                        binding.preparation_hash,
                        binding.artifact_id,
                        binding.created_at.isoformat(),
                    ),
                )
                consumed = connection.execute(
                    """
                    UPDATE graph_handoff_tokens SET consumed_at = ?
                    WHERE token_hash = ? AND consumed_at IS NULL
                    """,
                    (now.isoformat(), token_hash),
                )
                if consumed.rowcount != 1:
                    raise ValueError("HANDOFF_TOKEN_CONSUMED_OR_INVALID")
                connection.commit()
                return binding
            except Exception:
                if connection is not None:
                    try:
                        connection.rollback()
                    except sqlite3.Error:
                        logger.exception("failed to rollback staged graph handoff transaction")
                if moved:
                    shutil.rmtree(artifact_dir, ignore_errors=True)
                else:
                    self.discard_staged_artifact(artifact.id)
                raise
            finally:
                if connection is not None:
                    connection.close()

    def load_arrays(self, artifact_id: str) -> dict[str, np.ndarray]:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT artifacts.tensor_path
                FROM dataset_artifacts AS artifacts
                LEFT JOIN artifact_lifecycle AS lifecycle
                  ON lifecycle.artifact_id = artifacts.id
                WHERE artifacts.id = ?
                  AND COALESCE(lifecycle.status, 'active') = 'active'
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise FileNotFoundError("DatasetArtifact 不存在")
        path = (self.root / row[0]).resolve(strict=True)
        if self.root not in path.parents:
            raise ValueError("Artifact 张量路径越界")
        arrays: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                value = np.asarray(archive[name])
                if value.dtype.hasobject:
                    raise ValueError(f"数组 {name} 使用 object dtype")
                arrays[name] = value
        return arrays

    def load_attachment(self, artifact_id: str, relative_name: str) -> bytes:
        relative = PurePosixPath(relative_name.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact attachment path is unsafe")
        artifact_dir = (self.artifacts_root / artifact_id).resolve(strict=True)
        path = artifact_dir.joinpath(*relative.parts).resolve(strict=True)
        if artifact_dir not in path.parents:
            raise ValueError("artifact attachment path is unsafe")
        return path.read_bytes()

    def artifact_directory(self, artifact_id: str) -> Path:
        return self._artifact_path(artifact_id)

    def create_job(self, job: TrustedConversionJob, authorization_hash: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trusted_conversion_jobs
                (id, updated_at, job_json, authorization_hash) VALUES (?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.updated_at.isoformat(),
                    job.model_dump_json(by_alias=True),
                    authorization_hash,
                ),
            )

    def update_job(self, job: TrustedConversionJob, *, clear_authorization: bool = False) -> None:
        with self._lock, self._connect() as connection:
            if clear_authorization:
                connection.execute(
                    """
                    UPDATE trusted_conversion_jobs
                    SET updated_at = ?, job_json = ?, authorization_hash = NULL
                    WHERE id = ?
                    """,
                    (job.updated_at.isoformat(), job.model_dump_json(by_alias=True), job.id),
                )
            else:
                connection.execute(
                    """
                    UPDATE trusted_conversion_jobs
                    SET updated_at = ?, job_json = ? WHERE id = ?
                    """,
                    (job.updated_at.isoformat(), job.model_dump_json(by_alias=True), job.id),
                )

    def get_job(self, job_id: str) -> tuple[TrustedConversionJob, str | None] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT job_json, authorization_hash
                FROM trusted_conversion_jobs WHERE id = ?
                """,
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return TrustedConversionJob.model_validate_json(row[0]), row[1]

    def mark_interrupted_jobs(self) -> None:
        """A process restart cannot resume a child safely; make that state explicit."""

        now = datetime.now(UTC)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT id, job_json FROM trusted_conversion_jobs"
            ).fetchall()
            for job_id, payload in rows:
                job = TrustedConversionJob.model_validate_json(payload)
                if job.status not in {"queued", "running"}:
                    continue
                job.status = "failed"
                job.progress = 0
                job.updated_at = now
                job.issues.append(
                    DatasetIssue(
                        severity="error",
                        code="CONVERTER_PROCESS_INTERRUPTED",
                        message="API 服务重启，中断的转换进程不会自动恢复。",
                    )
                )
                connection.execute(
                    """
                    UPDATE trusted_conversion_jobs
                    SET updated_at = ?, job_json = ?, authorization_hash = NULL
                    WHERE id = ?
                    """,
                    (now.isoformat(), job.model_dump_json(by_alias=True), job_id),
                )
