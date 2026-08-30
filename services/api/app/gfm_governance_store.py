"""Append-only local governance store for SocialGraph-FM Governance analyst workflows."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NoReturn

from .gfm_client import GfmProxyError, _reject_link_components
from .gfm_hashing import canonical_sha256
from .gfm_governance_schemas import (
    CASE_PATTERN,
    GOVERNANCE_SCHEMA_VERSION,
    CaseCreateRequest,
    CaseItem,
    CaseItemRequest,
    CaseList,
    CaseTransitionRequest,
    GovernanceCase,
    ReviewEvent,
    ReviewEventRequest,
    ReviewCollection,
    ReviewCollectionCreateRequest,
    TargetLabelSet,
    TargetLabelSetV2,
    TargetReviewPolicy,
    TargetReviewPolicyV2,
    TargetTaskRegistration,
    target_label_binding_hash,
)

_TRANSITIONS = {
    "draft": frozenset({"active"}),
    "active": frozenset({"concluded"}),
    "concluded": frozenset({"archived"}),
    "archived": frozenset({"active"}),
}

_AUDIT_INVALID = "GOVERNANCE_AUDIT_INVALID"
_STORE_SCHEMA_VERSION = 4


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


class GovernanceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "governance.sqlite3"
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        _reject_link_components(self.database_path)
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS governance_cases (
                    case_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS governance_store_metadata (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS online_run_bindings (
                    run_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    dataset_content_hash TEXT NOT NULL,
                    graph_version_hash TEXT NOT NULL,
                    model_version_id TEXT NOT NULL,
                    model_version_hash TEXT NOT NULL,
                    model_state_hash TEXT,
                    created_at TEXT NOT NULL,
                    binding_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS case_state_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('draft','active','concluded','archived')),
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(case_id) REFERENCES governance_cases(case_id)
                );
                CREATE TABLE IF NOT EXISTS case_items (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    target_type TEXT NOT NULL CHECK(target_type IN ('node','relation','group')),
                    target_id TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    item_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(case_id) REFERENCES governance_cases(case_id),
                    UNIQUE(case_id, target_type, target_id)
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    target_type TEXT NOT NULL CHECK(target_type IN ('node','relation','group')),
                    target_id TEXT NOT NULL,
                    decision TEXT NOT NULL CHECK(decision IN ('confirmed','rejected','pending')),
                    reason TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(case_id) REFERENCES governance_cases(case_id)
                );
                CREATE TABLE IF NOT EXISTS adaptation_metadata (
                    kind TEXT NOT NULL CHECK(kind IN ('label_set','policy')),
                    record_hash TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    audit_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(kind, record_hash)
                );
                CREATE TABLE IF NOT EXISTS review_collections (
                    idempotency_key TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    result_hash TEXT,
                    target_task_registration_id TEXT NOT NULL,
                    case_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(case_id) REFERENCES governance_cases(case_id)
                );
                CREATE TABLE IF NOT EXISTS target_task_registrations (
                    registration_id TEXT PRIMARY KEY,
                    outer_bundle_sha256 TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    audit_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS target_adaptation_metadata (
                    kind TEXT NOT NULL,
                    record_hash TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    audit_hash TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(kind, record_hash)
                );
                """
            )
            connection.execute("BEGIN IMMEDIATE")
            try:
                self._migrate_schema(connection)
                self._create_triggers(connection)
                case_rows = connection.execute(
                    "SELECT case_id FROM governance_cases ORDER BY case_id"
                ).fetchall()
                for row in case_rows:
                    self._verified_case_rows(connection, str(row["case_id"]))
            except Exception:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _create_triggers(connection: sqlite3.Connection) -> None:
        statements = (
            """CREATE TRIGGER IF NOT EXISTS no_review_event_update
               BEFORE UPDATE ON review_events
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_review_event_delete
               BEFORE DELETE ON review_events
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_state_event_update
               BEFORE UPDATE ON case_state_events
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_state_event_delete
               BEFORE DELETE ON case_state_events
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_case_item_update
               BEFORE UPDATE ON case_items
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_case_item_delete
               BEFORE DELETE ON case_items
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_governance_case_update
               BEFORE UPDATE ON governance_cases
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_governance_case_delete
               BEFORE DELETE ON governance_cases
               BEGIN SELECT RAISE(ABORT, 'append-only'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_run_binding_update
               BEFORE UPDATE ON online_run_bindings
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_run_binding_delete
               BEFORE DELETE ON online_run_bindings
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_governance_metadata_update
               BEFORE UPDATE ON governance_store_metadata
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_governance_metadata_delete
               BEFORE DELETE ON governance_store_metadata
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_adaptation_metadata_update
               BEFORE UPDATE ON adaptation_metadata
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_adaptation_metadata_delete
               BEFORE DELETE ON adaptation_metadata
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_review_collection_update
               BEFORE UPDATE ON review_collections
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_review_collection_delete
               BEFORE DELETE ON review_collections
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_target_task_registration_update
               BEFORE UPDATE ON target_task_registrations
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_target_task_registration_delete
               BEFORE DELETE ON target_task_registrations
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_target_adaptation_metadata_update
               BEFORE UPDATE ON target_adaptation_metadata
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
            """CREATE TRIGGER IF NOT EXISTS no_target_adaptation_metadata_delete
               BEFORE DELETE ON target_adaptation_metadata
               BEGIN SELECT RAISE(ABORT, 'immutable'); END""",
        )
        for statement in statements:
            connection.execute(statement)

    def _migrate_schema(self, connection: sqlite3.Connection) -> None:
        case_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(governance_cases)")
        }
        binding_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(online_run_bindings)")
        }
        collection_columns = {
            str(row["name"])
            for row in connection.execute("PRAGMA table_info(review_collections)")
        }
        marker = connection.execute(
            "SELECT schema_version FROM governance_store_metadata WHERE singleton = 1"
        ).fetchone()
        if marker is not None:
            version = int(marker["schema_version"])
            if version == _STORE_SCHEMA_VERSION:
                if (
                    "root_hash" not in case_columns
                    or "model_state_hash" not in binding_columns
                    or "result_hash" not in collection_columns
                ):
                    self._audit_invalid()
                return
            if version in {2, 3} and "root_hash" in case_columns:
                connection.execute("DROP TRIGGER IF EXISTS no_governance_metadata_update")
                if "model_state_hash" not in binding_columns:
                    connection.execute(
                        "ALTER TABLE online_run_bindings ADD COLUMN model_state_hash TEXT"
                    )
                if "result_hash" not in collection_columns:
                    connection.execute(
                        "ALTER TABLE review_collections ADD COLUMN result_hash TEXT"
                    )
                connection.execute(
                    "UPDATE governance_store_metadata SET schema_version = ? WHERE singleton = 1",
                    (_STORE_SCHEMA_VERSION,),
                )
                return
            self._audit_invalid()
        if "root_hash" in case_columns:
            self._audit_invalid()

        legacy_items = connection.execute(
            "SELECT * FROM case_items ORDER BY sequence"
        ).fetchall()
        for row in legacy_items:
            legacy_payload = {
                "itemId": str(row["item_id"]),
                "targetType": str(row["target_type"]),
                "targetId": str(row["target_id"]),
                "note": str(row["note"]),
                "createdAt": str(row["created_at"]),
            }
            if canonical_sha256(legacy_payload) != str(row["item_hash"]):
                self._audit_invalid()

        connection.execute("DROP TRIGGER IF EXISTS no_case_item_update")
        connection.execute("DROP TRIGGER IF EXISTS no_governance_case_update")
        connection.execute("ALTER TABLE governance_cases ADD COLUMN root_hash TEXT")
        if "model_state_hash" not in binding_columns:
            connection.execute(
                "ALTER TABLE online_run_bindings ADD COLUMN model_state_hash TEXT"
            )
        if "result_hash" not in collection_columns:
            connection.execute(
                "ALTER TABLE review_collections ADD COLUMN result_hash TEXT"
            )
        case_rows = connection.execute(
            "SELECT * FROM governance_cases ORDER BY case_id"
        ).fetchall()
        for row in case_rows:
            root_payload = self._case_root_payload(row)
            connection.execute(
                "UPDATE governance_cases SET root_hash = ? WHERE case_id = ?",
                (canonical_sha256(root_payload), str(row["case_id"])),
            )
        for row in legacy_items:
            item_payload = self._case_item_payload(row)
            connection.execute(
                "UPDATE case_items SET item_hash = ? WHERE sequence = ?",
                (canonical_sha256(item_payload), int(row["sequence"])),
            )
        connection.execute(
            """INSERT INTO governance_store_metadata (singleton, schema_version)
               VALUES (1, ?)""",
            (_STORE_SCHEMA_VERSION,),
        )

    @staticmethod
    def _case_root_payload(row: sqlite3.Row | dict[str, str]) -> dict[str, str]:
        return {
            "caseId": str(row["case_id"]),
            "runId": str(row["run_id"]),
            "title": str(row["title"]),
            "description": str(row["description"]),
            "createdAt": str(row["created_at"]),
        }

    @staticmethod
    def _case_item_payload(row: sqlite3.Row | dict[str, str]) -> dict[str, str]:
        return {
            "itemId": str(row["item_id"]),
            "caseId": str(row["case_id"]),
            "targetType": str(row["target_type"]),
            "targetId": str(row["target_id"]),
            "note": str(row["note"]),
            "createdAt": str(row["created_at"]),
        }

    def put_run_binding(self, value: dict[str, str]) -> None:
        required = {
            "runId",
            "requestHash",
            "artifactId",
            "datasetContentHash",
            "graphVersionHash",
            "modelVersionId",
            "modelVersionHash",
            "modelStateHash",
            "createdAt",
        }
        if set(value) != required:
            raise ValueError("run binding fields are invalid")
        binding_hash = canonical_sha256(value)
        try:
            with self._lock, self._connect() as connection:
                existing = connection.execute(
                    "SELECT * FROM online_run_bindings WHERE run_id = ?", (value["runId"],)
                ).fetchone()
                if existing is not None:
                    if str(existing["binding_hash"]) != binding_hash:
                        raise GfmProxyError(409, "GOVERNANCE_RUN_BINDING_CONFLICT")
                    return
                connection.execute(
                    """INSERT INTO online_run_bindings
                       (run_id, request_hash, artifact_id, dataset_content_hash,
                        graph_version_hash, model_version_id, model_version_hash,
                        model_state_hash, created_at, binding_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        value["runId"],
                        value["requestHash"],
                        value["artifactId"],
                        value["datasetContentHash"],
                        value["graphVersionHash"],
                        value["modelVersionId"],
                        value["modelVersionHash"],
                        value["modelStateHash"],
                        value["createdAt"],
                        binding_hash,
                    ),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_RUN_BINDING_PERSIST_FAILED") from error

    def get_run_binding(self, run_id: str) -> dict[str, str]:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM online_run_bindings WHERE run_id = ?", (run_id,)
                ).fetchone()
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_RUN_BINDING_READ_FAILED") from error
        if row is None:
            raise GfmProxyError(404, "GOVERNANCE_RUN_NOT_FOUND")
        if row["model_state_hash"] is None:
            raise GfmProxyError(409, "GOVERNANCE_RUN_BINDING_MODEL_STATE_MISSING")
        value = {
            "runId": str(row["run_id"]),
            "requestHash": str(row["request_hash"]),
            "artifactId": str(row["artifact_id"]),
            "datasetContentHash": str(row["dataset_content_hash"]),
            "graphVersionHash": str(row["graph_version_hash"]),
            "modelVersionId": str(row["model_version_id"]),
            "modelVersionHash": str(row["model_version_hash"]),
            "modelStateHash": str(row["model_state_hash"]),
            "createdAt": str(row["created_at"]),
        }
        if canonical_sha256(value) != str(row["binding_hash"]):
            raise GfmProxyError(502, "GOVERNANCE_RUN_BINDING_INVALID")
        return value

    @staticmethod
    def _check_case_id(case_id: str) -> None:
        import re

        if re.fullmatch(CASE_PATTERN, case_id) is None:
            raise GfmProxyError(404, "GOVERNANCE_CASE_NOT_FOUND")

    @staticmethod
    def _audit_invalid() -> NoReturn:
        raise GfmProxyError(502, _AUDIT_INVALID)

    def _verified_case_rows(
        self, connection: sqlite3.Connection, case_id: str
    ) -> tuple[
        sqlite3.Row,
        tuple[sqlite3.Row, ...],
        tuple[sqlite3.Row, ...],
        tuple[sqlite3.Row, ...],
    ]:
        case_row = connection.execute(
            "SELECT * FROM governance_cases WHERE case_id = ?", (case_id,)
        ).fetchone()
        if case_row is None:
            raise GfmProxyError(404, "GOVERNANCE_CASE_NOT_FOUND")
        if canonical_sha256(self._case_root_payload(case_row)) != str(
            case_row["root_hash"]
        ):
            self._audit_invalid()

        state_rows = tuple(
            connection.execute(
                """SELECT * FROM case_state_events
                   WHERE case_id = ? ORDER BY sequence""",
                (case_id,),
            ).fetchall()
        )
        if not state_rows:
            self._audit_invalid()
        previous_state: str | None = None
        for index, row in enumerate(state_rows):
            state = str(row["state"])
            reason = str(row["reason"])
            created_at = str(row["created_at"])
            payload = {
                "caseId": case_id,
                "state": state,
                "reason": reason,
                "createdAt": created_at,
            }
            valid_transition = (
                state == "draft"
                and reason == "case-created"
                and created_at == str(case_row["created_at"])
                if index == 0
                else previous_state is not None
                and state in _TRANSITIONS.get(previous_state, frozenset())
            )
            if (
                not valid_transition
                or canonical_sha256(payload) != str(row["event_hash"])
            ):
                self._audit_invalid()
            previous_state = state

        item_rows = tuple(
            connection.execute(
                "SELECT * FROM case_items WHERE case_id = ? ORDER BY sequence",
                (case_id,),
            ).fetchall()
        )
        for row in item_rows:
            if canonical_sha256(self._case_item_payload(row)) != str(row["item_hash"]):
                self._audit_invalid()

        event_rows = tuple(
            connection.execute(
                "SELECT * FROM review_events WHERE case_id = ? ORDER BY sequence",
                (case_id,),
            ).fetchall()
        )
        previous_event_hash: str | None = None
        for sequence, row in enumerate(event_rows, start=1):
            stored_previous = (
                str(row["previous_event_hash"])
                if row["previous_event_hash"] is not None
                else None
            )
            review_payload: dict[str, Any] = {
                "eventId": str(row["event_id"]),
                "caseId": case_id,
                "targetType": str(row["target_type"]),
                "targetId": str(row["target_id"]),
                "decision": str(row["decision"]),
                "reason": str(row["reason"]),
                "actor": str(row["actor"]),
                "sequence": sequence,
                "createdAt": str(row["created_at"]),
                "previousEventHash": stored_previous,
            }
            event_hash = str(row["event_hash"])
            if (
                stored_previous != previous_event_hash
                or canonical_sha256(review_payload) != event_hash
            ):
                self._audit_invalid()
            previous_event_hash = event_hash

        return case_row, state_rows, item_rows, event_rows

    def create_case(self, request: CaseCreateRequest) -> GovernanceCase:
        case_id = f"case-{uuid.uuid4().hex}"
        created_at = _now()
        root_record = {
            "case_id": case_id,
            "run_id": request.run_id,
            "title": request.title,
            "description": request.description,
            "created_at": created_at,
        }
        root_hash = canonical_sha256(self._case_root_payload(root_record))
        initial = {
            "caseId": case_id,
            "state": "draft",
            "reason": "case-created",
            "createdAt": created_at,
        }
        state_hash = canonical_sha256(initial)
        try:
            with self._lock, self._connect() as connection:
                connection.execute(
                    """INSERT INTO governance_cases
                       (case_id, run_id, title, description, created_at, root_hash)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        case_id,
                        request.run_id,
                        request.title,
                        request.description,
                        created_at,
                        root_hash,
                    ),
                )
                connection.execute(
                    """INSERT INTO case_state_events
                       (case_id, state, reason, created_at, event_hash)
                       VALUES (?, 'draft', 'case-created', ?, ?)""",
                    (case_id, created_at, state_hash),
                )
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_PERSIST_FAILED") from error
        return self.get_case(case_id)

    def create_review_collection(
        self, request: ReviewCollectionCreateRequest
    ) -> ReviewCollection:
        request_payload = request.model_dump(mode="json", by_alias=True)
        request_hash = canonical_sha256(request_payload)
        case_id = f"case-{uuid.uuid4().hex}"
        created_at = _now()
        active_at = _now()
        root_record = {
            "case_id": case_id,
            "run_id": request.run_id,
            "title": request.title,
            "description": request.description,
            "created_at": created_at,
        }
        initial = {
            "caseId": case_id,
            "state": "draft",
            "reason": "case-created",
            "createdAt": created_at,
        }
        active = {
            "caseId": case_id,
            "state": "active",
            "reason": "review-collection-activated",
            "createdAt": active_at,
        }
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM review_collections WHERE idempotency_key = ?",
                    (request.idempotency_key,),
                ).fetchone()
                if existing is not None:
                    if str(existing["request_hash"]) != request_hash:
                        raise GfmProxyError(
                            409, "GOVERNANCE_REVIEW_COLLECTION_IDEMPOTENCY_CONFLICT"
                        )
                    if existing["result_hash"] is not None and str(
                        existing["result_hash"]
                    ) != request.result_hash:
                        raise GfmProxyError(
                            409, "GOVERNANCE_REVIEW_COLLECTION_IDEMPOTENCY_CONFLICT"
                        )
                    existing_case_id = str(existing["case_id"])
                    connection.commit()
                    case = self.get_case(existing_case_id)
                    return self._review_collection_model(
                        request.idempotency_key,
                        str(existing["target_task_registration_id"]),
                        request_hash,
                        str(existing["result_hash"] or request.result_hash),
                        case,
                    )
                connection.execute(
                    """INSERT INTO governance_cases
                       (case_id, run_id, title, description, created_at, root_hash)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        case_id,
                        request.run_id,
                        request.title,
                        request.description,
                        created_at,
                        canonical_sha256(self._case_root_payload(root_record)),
                    ),
                )
                connection.execute(
                    """INSERT INTO case_state_events
                       (case_id, state, reason, created_at, event_hash)
                       VALUES (?, 'draft', 'case-created', ?, ?)""",
                    (case_id, created_at, canonical_sha256(initial)),
                )
                for item in request.items:
                    item_id = f"item-{uuid.uuid4().hex}"
                    item_record = {
                        "item_id": item_id,
                        "case_id": case_id,
                        "target_type": item.target_type,
                        "target_id": item.target_id,
                        "note": item.note,
                        "created_at": created_at,
                    }
                    connection.execute(
                        """INSERT INTO case_items
                           (item_id, case_id, target_type, target_id, note, created_at, item_hash)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item_id,
                            case_id,
                            item.target_type,
                            item.target_id,
                            item.note,
                            created_at,
                            canonical_sha256(self._case_item_payload(item_record)),
                        ),
                    )
                connection.execute(
                    """INSERT INTO case_state_events
                       (case_id, state, reason, created_at, event_hash)
                       VALUES (?, 'active', 'review-collection-activated', ?, ?)""",
                    (case_id, active_at, canonical_sha256(active)),
                )
                connection.execute(
                    """INSERT INTO review_collections
                       (idempotency_key, request_hash, result_hash,
                        target_task_registration_id, case_id, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        request.idempotency_key,
                        request_hash,
                        request.result_hash,
                        request.target_task_registration_id,
                        case_id,
                        created_at,
                    ),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_REVIEW_COLLECTION_PERSIST_FAILED"
            ) from error
        return self._review_collection_model(
            request.idempotency_key,
            request.target_task_registration_id,
            request_hash,
            request.result_hash,
            self.get_case(case_id),
        )

    def put_target_task_registration(
        self, registration: TargetTaskRegistration, bundle: bytes
    ) -> None:
        payload = registration.model_dump(mode="json", by_alias=True)
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        audit_hash = canonical_sha256(
            {
                "kind": "target_task_registration",
                "recordHash": registration.registration_hash,
                "payload": payload,
            }
        )
        bundle_root = self.root / "target-tasks"
        bundle_root.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_root / f"{registration.registration_id}.sgtask.zip"
        temporary = bundle_root / f".{registration.registration_id}.{uuid.uuid4().hex}.tmp"
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT * FROM target_task_registrations WHERE registration_id = ?",
                    (registration.registration_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["payload_json"]) != encoded
                        or str(existing["audit_hash"]) != audit_hash
                        or not bundle_path.is_file()
                        or bundle_path.read_bytes() != bundle
                    ):
                        raise GfmProxyError(
                            409, "GOVERNANCE_TARGET_TASK_REGISTRATION_CONFLICT"
                        )
                    return
                with temporary.open("xb") as stream:
                    stream.write(bundle)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, bundle_path)
                connection.execute(
                    """INSERT INTO target_task_registrations
                       (registration_id, outer_bundle_sha256, payload_json, audit_hash, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        registration.registration_id,
                        registration.outer_bundle_sha256,
                        encoded,
                        audit_hash,
                        registration.created_at.isoformat(),
                    ),
                )
        except GfmProxyError:
            raise
        except (OSError, sqlite3.Error) as error:
            if bundle_path.exists():
                bundle_path.unlink(missing_ok=True)
            raise GfmProxyError(
                502, "GOVERNANCE_TARGET_TASK_REGISTRATION_PERSIST_FAILED"
            ) from error
        finally:
            temporary.unlink(missing_ok=True)

    def get_target_task_registration(
        self, registration_id: str
    ) -> TargetTaskRegistration:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM target_task_registrations WHERE registration_id = ?",
                    (registration_id,),
                ).fetchone()
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_TARGET_TASK_REGISTRATION_READ_FAILED"
            ) from error
        if row is None:
            raise GfmProxyError(404, "GOVERNANCE_TARGET_TASK_NOT_FOUND")
        try:
            payload = json.loads(str(row["payload_json"]))
            expected_audit = canonical_sha256(
                {
                    "kind": "target_task_registration",
                    "recordHash": payload["registrationHash"],
                    "payload": payload,
                }
            )
            if expected_audit != str(row["audit_hash"]):
                raise ValueError("target task audit mismatch")
            return TargetTaskRegistration.model_validate(payload)
        except (KeyError, TypeError, ValueError) as error:
            raise GfmProxyError(
                502, "GOVERNANCE_TARGET_TASK_REGISTRATION_INVALID"
            ) from error

    def target_task_bundle_path(self, registration_id: str) -> Path:
        self.get_target_task_registration(registration_id)
        return self.root / "target-tasks" / f"{registration_id}.sgtask.zip"

    @staticmethod
    def _review_collection_model(
        idempotency_key: str,
        target_task_registration_id: str,
        request_hash: str,
        result_hash: str,
        case: GovernanceCase,
    ) -> ReviewCollection:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.governance-review-collection/1.0",
            "idempotencyKey": idempotency_key,
            "targetTaskRegistrationId": target_task_registration_id,
            "requestHash": request_hash,
            "resultHash": result_hash,
            "case": case.model_dump(mode="json", by_alias=True),
        }
        payload["collectionHash"] = canonical_sha256(payload)
        return ReviewCollection.model_validate(payload)

    def _case_state(self, connection: sqlite3.Connection, case_id: str) -> tuple[str, str]:
        _, state_rows, _, _ = self._verified_case_rows(connection, case_id)
        row = state_rows[-1]
        return str(row["state"]), str(row["created_at"])

    def transition(self, case_id: str, request: CaseTransitionRequest) -> GovernanceCase:
        self._check_case_id(case_id)
        created_at = _now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current, _ = self._case_state(connection, case_id)
                if request.state not in _TRANSITIONS[current]:
                    raise GfmProxyError(409, "GOVERNANCE_CASE_TRANSITION_INVALID")
                payload = {
                    "caseId": case_id,
                    "state": request.state,
                    "reason": request.reason,
                    "createdAt": created_at,
                }
                connection.execute(
                    """INSERT INTO case_state_events
                       (case_id, state, reason, created_at, event_hash)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        case_id,
                        request.state,
                        request.reason,
                        created_at,
                        canonical_sha256(payload),
                    ),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_PERSIST_FAILED") from error
        return self.get_case(case_id)

    def add_item(self, case_id: str, request: CaseItemRequest) -> GovernanceCase:
        self._check_case_id(case_id)
        item_id = f"item-{uuid.uuid4().hex}"
        created_at = _now()
        payload = {
            "itemId": item_id,
            "caseId": case_id,
            "targetType": request.target_type,
            "targetId": request.target_id,
            "note": request.note,
            "createdAt": created_at,
        }
        item_hash = canonical_sha256(payload)
        try:
            with self._lock, self._connect() as connection:
                state, _ = self._case_state(connection, case_id)
                if state not in {"draft", "active"}:
                    raise GfmProxyError(409, "GOVERNANCE_CASE_READ_ONLY")
                connection.execute(
                    """INSERT INTO case_items
                       (item_id, case_id, target_type, target_id, note, created_at, item_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item_id,
                        case_id,
                        request.target_type,
                        request.target_id,
                        request.note,
                        created_at,
                        item_hash,
                    ),
                )
        except GfmProxyError:
            raise
        except sqlite3.IntegrityError as error:
            raise GfmProxyError(409, "GOVERNANCE_CASE_ITEM_EXISTS") from error
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_PERSIST_FAILED") from error
        return self.get_case(case_id)

    def add_review(self, case_id: str, request: ReviewEventRequest) -> GovernanceCase:
        self._check_case_id(case_id)
        event_id = f"event-{uuid.uuid4().hex}"
        created_at = _now()
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                state, _ = self._case_state(connection, case_id)
                if state != "active":
                    raise GfmProxyError(409, "GOVERNANCE_CASE_NOT_ACTIVE")
                item = connection.execute(
                    """SELECT 1 FROM case_items WHERE case_id = ?
                       AND target_type = ? AND target_id = ?""",
                    (case_id, request.target_type, request.target_id),
                ).fetchone()
                if item is None:
                    raise GfmProxyError(404, "GOVERNANCE_CASE_ITEM_NOT_FOUND")
                previous = connection.execute(
                    """SELECT event_hash FROM review_events WHERE case_id = ?
                       ORDER BY sequence DESC LIMIT 1""",
                    (case_id,),
                ).fetchone()
                previous_hash = str(previous["event_hash"]) if previous else None
                sequence_row = connection.execute(
                    "SELECT COUNT(*) AS count FROM review_events WHERE case_id = ?",
                    (case_id,),
                ).fetchone()
                sequence = int(sequence_row["count"]) + 1
                payload = {
                    "eventId": event_id,
                    "caseId": case_id,
                    "targetType": request.target_type,
                    "targetId": request.target_id,
                    "decision": request.decision,
                    "reason": request.reason,
                    "actor": request.actor,
                    "sequence": sequence,
                    "createdAt": created_at,
                    "previousEventHash": previous_hash,
                }
                connection.execute(
                    """INSERT INTO review_events
                       (event_id, case_id, target_type, target_id, decision, reason,
                        actor, created_at, previous_event_hash, event_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id,
                        case_id,
                        request.target_type,
                        request.target_id,
                        request.decision,
                        request.reason,
                        request.actor,
                        created_at,
                        previous_hash,
                        canonical_sha256(payload),
                    ),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_REVIEW_PERSIST_FAILED") from error
        return self.get_case(case_id)

    def get_case(self, case_id: str) -> GovernanceCase:
        self._check_case_id(case_id)
        try:
            with self._lock, self._connect() as connection:
                row, state_rows, item_rows, event_rows = self._verified_case_rows(
                    connection, case_id
                )
                state = str(state_rows[-1]["state"])
                updated_at = str(state_rows[-1]["created_at"])
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_READ_FAILED") from error
        items = tuple(
            CaseItem(
                itemId=item["item_id"],
                targetType=item["target_type"],
                targetId=item["target_id"],
                note=item["note"],
                createdAt=item["created_at"],
                itemHash=item["item_hash"],
            )
            for item in item_rows
        )
        events = tuple(
            ReviewEvent(
                eventId=event["event_id"],
                targetType=event["target_type"],
                targetId=event["target_id"],
                decision=event["decision"],
                reason=event["reason"],
                actor=event["actor"],
                sequence=index,
                createdAt=event["created_at"],
                previousEventHash=event["previous_event_hash"],
                eventHash=event["event_hash"],
            )
            for index, event in enumerate(event_rows, start=1)
        )
        decisions: dict[str, str] = {}
        for event in events:
            decisions[f"{event.target_type}:{event.target_id}"] = event.decision
        payload: dict[str, Any] = {
            "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
            "caseId": row["case_id"],
            "runId": row["run_id"],
            "title": row["title"],
            "description": row["description"],
            "state": state,
            "createdAt": row["created_at"],
            "updatedAt": updated_at,
            "items": [item.model_dump(mode="json", by_alias=True) for item in items],
            "reviewEvents": [event.model_dump(mode="json", by_alias=True) for event in events],
            "currentDecisions": decisions,
        }
        payload["caseHash"] = canonical_sha256(payload)
        return GovernanceCase.model_validate(payload)

    def list_cases(self, *, offset: int, limit: int) -> CaseList:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT case_id FROM governance_cases ORDER BY created_at DESC, case_id"
            ).fetchall()
        items = tuple(self.get_case(str(row["case_id"])) for row in rows[offset : offset + limit])
        return CaseList.model_validate(
            {
                "schemaVersion": GOVERNANCE_SCHEMA_VERSION,
                "items": items,
                "total": len(rows),
                "offset": offset,
                "limit": limit,
            }
        )

    def case_state_timeline(self, case_id: str) -> list[dict[str, Any]]:
        self._check_case_id(case_id)
        try:
            with self._lock, self._connect() as connection:
                _, rows, _, _ = self._verified_case_rows(connection, case_id)
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_READ_FAILED") from error
        return [
            {
                "sequence": index,
                "state": str(row["state"]),
                "reason": str(row["reason"]),
                "createdAt": str(row["created_at"]),
                "eventHash": str(row["event_hash"]),
            }
            for index, row in enumerate(rows, start=1)
        ]

    def run_review_summary(self, run_id: str) -> dict[str, int]:
        with self._lock, self._connect() as connection:
            case_row = connection.execute(
                "SELECT COUNT(*) AS count FROM governance_cases WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            event_row = connection.execute(
                """SELECT COUNT(*) AS count FROM review_events AS reviews
                   JOIN governance_cases AS cases ON cases.case_id = reviews.case_id
                   WHERE cases.run_id = ?""",
                (run_id,),
            ).fetchone()
        return {
            "caseCount": int(case_row["count"]),
            "reviewEventCount": int(event_row["count"]),
        }

    def _put_adaptation_metadata(
        self,
        *,
        kind: str,
        record_hash: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if len(encoded.encode("utf-8")) > 2 * 1024 * 1024:
            raise GfmProxyError(502, "GOVERNANCE_ADAPTATION_METADATA_TOO_LARGE")
        audit_hash = canonical_sha256(
            {"kind": kind, "recordHash": record_hash, "payload": payload}
        )
        try:
            with self._lock, self._connect() as connection:
                existing = connection.execute(
                    """SELECT payload_json, audit_hash FROM adaptation_metadata
                       WHERE kind = ? AND record_hash = ?""",
                    (kind, record_hash),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["payload_json"]) != encoded
                        or str(existing["audit_hash"]) != audit_hash
                    ):
                        raise GfmProxyError(
                            409, "GOVERNANCE_ADAPTATION_METADATA_CONFLICT"
                        )
                    return
                connection.execute(
                    """INSERT INTO adaptation_metadata
                       (kind, record_hash, run_id, payload_json, audit_hash, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (kind, record_hash, run_id, encoded, audit_hash, _now()),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_PERSIST_FAILED"
            ) from error

    def put_adaptation_label_set(self, value: TargetLabelSet) -> None:
        self._put_adaptation_metadata(
            kind="label_set",
            record_hash=value.label_set_hash,
            run_id=value.binding.run_id,
            payload=value.model_dump(mode="json", by_alias=True),
        )

    def put_adaptation_policy(self, value: TargetReviewPolicy) -> None:
        self._put_adaptation_metadata(
            kind="policy",
            record_hash=value.policy_hash,
            run_id=value.binding.run_id,
            payload=value.model_dump(mode="json", by_alias=True),
        )

    def put_target_adaptation_metadata(
        self,
        *,
        kind: str,
        record_hash: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        audit_hash = canonical_sha256(
            {"kind": kind, "recordHash": record_hash, "payload": payload}
        )
        try:
            with self._lock, self._connect() as connection:
                existing = connection.execute(
                    """SELECT payload_json, audit_hash FROM target_adaptation_metadata
                       WHERE kind = ? AND record_hash = ?""",
                    (kind, record_hash),
                ).fetchone()
                if existing is not None:
                    if str(existing["payload_json"]) != encoded or str(existing["audit_hash"]) != audit_hash:
                        raise GfmProxyError(
                            409, "GOVERNANCE_ADAPTATION_METADATA_CONFLICT"
                        )
                    return
                connection.execute(
                    """INSERT INTO target_adaptation_metadata
                       (kind, record_hash, run_id, payload_json, audit_hash, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (kind, record_hash, run_id, encoded, audit_hash, _now()),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_PERSIST_FAILED"
            ) from error

    def get_target_adaptation_metadata(
        self, *, kind: str, record_hash: str
    ) -> dict[str, Any]:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    """SELECT payload_json, audit_hash FROM target_adaptation_metadata
                       WHERE kind = ? AND record_hash = ?""",
                    (kind, record_hash),
                ).fetchone()
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_READ_FAILED"
            ) from error
        if row is None:
            raise GfmProxyError(404, "GOVERNANCE_ADAPTATION_NOT_FOUND")
        try:
            payload = json.loads(str(row["payload_json"]))
            if not isinstance(payload, dict) or canonical_sha256(
                {"kind": kind, "recordHash": record_hash, "payload": payload}
            ) != str(row["audit_hash"]):
                raise ValueError("target adaptation audit mismatch")
            return payload
        except (TypeError, ValueError) as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
            ) from error

    def put_target_label_set(
        self,
        value: TargetLabelSetV2,
        *,
        run_id: str,
        source: dict[str, Any],
    ) -> None:
        registration_id = source.get("targetTaskRegistrationId")
        result_hash = source.get("resultHash")
        if (
            not isinstance(registration_id, str)
            or not isinstance(result_hash, str)
            or source.get("runId") != run_id
        ):
            raise GfmProxyError(
                409, "GOVERNANCE_ADAPTATION_PROVENANCE_MISMATCH"
            )
        binding_hash = target_label_binding_hash(
            value.label_set_hash, registration_id, run_id, result_hash
        )
        self.put_target_adaptation_metadata(
            kind="label_set_content",
            record_hash=value.label_set_hash,
            run_id=run_id,
            payload=value.model_dump(mode="json", by_alias=True),
        )
        self.put_target_adaptation_metadata(
            kind="label_set_binding",
            record_hash=binding_hash,
            run_id=run_id,
            payload={
                "schemaVersion": "TargetLabelSetBinding/1.0",
                "bindingHash": binding_hash,
                "labelSetHash": value.label_set_hash,
                "targetTaskRegistrationId": registration_id,
                "runId": run_id,
                "resultHash": result_hash,
                "source": source,
            },
        )

    def get_target_label_set(
        self,
        label_set_hash: str,
        *,
        target_task_registration_id: str | None = None,
        run_id: str | None = None,
        result_hash: str | None = None,
    ) -> tuple[TargetLabelSetV2, dict[str, Any]]:
        identity_values = (target_task_registration_id, run_id, result_hash)
        if any(value is not None for value in identity_values) and (
            run_id is None or result_hash is None
        ):
            raise GfmProxyError(422, "GOVERNANCE_ADAPTATION_BINDING_INVALID")

        legacy_payload: dict[str, Any] | None = None
        try:
            content = self.get_target_adaptation_metadata(
                kind="label_set_content", record_hash=label_set_hash
            )
        except GfmProxyError as error:
            if error.status_code != 404:
                raise
            try:
                legacy_payload = self.get_target_adaptation_metadata(
                    kind="label_set", record_hash=label_set_hash
                )
                content = legacy_payload["record"]
            except GfmProxyError:
                raise
            except (KeyError, TypeError) as legacy_error:
                raise GfmProxyError(
                    502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
                ) from legacy_error
        else:
            try:
                legacy_payload = self.get_target_adaptation_metadata(
                    kind="label_set", record_hash=label_set_hash
                )
            except GfmProxyError as error:
                if error.status_code != 404:
                    raise

        bindings = self._target_label_bindings(label_set_hash)
        if legacy_payload is not None:
            try:
                legacy_source = legacy_payload["source"]
                if not isinstance(legacy_source, dict):
                    raise TypeError
                bindings.append(legacy_source)
            except (KeyError, TypeError) as error:
                raise GfmProxyError(
                    502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
                ) from error

        matches = [
            source
            for source in bindings
            if (target_task_registration_id is None or source.get("targetTaskRegistrationId") == target_task_registration_id)
            and (run_id is None or source.get("runId") == run_id)
            and (result_hash is None or source.get("resultHash") == result_hash)
        ]
        unique_matches = {
            canonical_sha256(source): source
            for source in matches
        }
        if not unique_matches:
            raise GfmProxyError(404, "GOVERNANCE_ADAPTATION_NOT_FOUND")
        if len(unique_matches) != 1:
            raise GfmProxyError(409, "GOVERNANCE_ADAPTATION_BINDING_AMBIGUOUS")
        try:
            return TargetLabelSetV2.model_validate(content), next(
                iter(unique_matches.values())
            )
        except (TypeError, ValueError) as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
            ) from error

    def _target_label_bindings(self, label_set_hash: str) -> list[dict[str, Any]]:
        try:
            with self._lock, self._connect() as connection:
                rows = connection.execute(
                    """SELECT record_hash, payload_json, audit_hash
                       FROM target_adaptation_metadata
                       WHERE kind = 'label_set_binding'
                       ORDER BY record_hash"""
                ).fetchall()
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_READ_FAILED"
            ) from error
        sources: list[dict[str, Any]] = []
        for row in rows:
            try:
                record_hash = str(row["record_hash"])
                payload = json.loads(str(row["payload_json"]))
                if (
                    not isinstance(payload, dict)
                    or canonical_sha256(
                        {
                            "kind": "label_set_binding",
                            "recordHash": record_hash,
                            "payload": payload,
                        }
                    )
                    != str(row["audit_hash"])
                    or payload.get("bindingHash") != record_hash
                ):
                    raise ValueError("target label binding audit mismatch")
                if payload.get("labelSetHash") != label_set_hash:
                    continue
                if payload.get("schemaVersion") != "TargetLabelSetBinding/1.0":
                    raise ValueError("target label binding schema mismatch")
                expected_hash = target_label_binding_hash(
                    label_set_hash,
                    str(payload["targetTaskRegistrationId"]),
                    str(payload["runId"]),
                    str(payload["resultHash"]),
                )
                source = payload["source"]
                if expected_hash != record_hash or not isinstance(source, dict):
                    raise ValueError("target label binding identity mismatch")
                if (
                    source.get("targetTaskRegistrationId")
                    != payload["targetTaskRegistrationId"]
                    or source.get("runId") != payload["runId"]
                    or source.get("resultHash") != payload["resultHash"]
                ):
                    raise ValueError("target label source identity mismatch")
                sources.append(source)
            except (KeyError, TypeError, ValueError) as error:
                raise GfmProxyError(
                    502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
                ) from error
        return sources

    def put_target_policy(self, value: TargetReviewPolicyV2) -> None:
        self.put_target_adaptation_metadata(
            kind="policy",
            record_hash=value.policy_hash,
            run_id=value.binding.run_id,
            payload=value.model_dump(mode="json", by_alias=True),
        )

    def get_target_policy(self, policy_hash: str) -> TargetReviewPolicyV2:
        return TargetReviewPolicyV2.model_validate(
            self.get_target_adaptation_metadata(kind="policy", record_hash=policy_hash)
        )

    def _get_adaptation_metadata(
        self, *, kind: str, record_hash: str
    ) -> dict[str, Any]:
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute(
                    """SELECT payload_json, audit_hash FROM adaptation_metadata
                       WHERE kind = ? AND record_hash = ?""",
                    (kind, record_hash),
                ).fetchone()
        except sqlite3.Error as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_READ_FAILED"
            ) from error
        if row is None:
            raise GfmProxyError(404, "GOVERNANCE_ADAPTATION_NOT_FOUND")
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, ValueError) as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
            ) from error
        if not isinstance(payload, dict) or canonical_sha256(
            {"kind": kind, "recordHash": record_hash, "payload": payload}
        ) != str(row["audit_hash"]):
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
            )
        return payload

    def get_adaptation_label_set(self, label_set_hash: str) -> TargetLabelSet:
        try:
            return TargetLabelSet.model_validate(
                self._get_adaptation_metadata(
                    kind="label_set", record_hash=label_set_hash
                )
            )
        except GfmProxyError:
            raise
        except ValueError as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
            ) from error

    def get_adaptation_policy(self, policy_hash: str) -> TargetReviewPolicy:
        try:
            return TargetReviewPolicy.model_validate(
                self._get_adaptation_metadata(kind="policy", record_hash=policy_hash)
            )
        except GfmProxyError:
            raise
        except ValueError as error:
            raise GfmProxyError(
                502, "GOVERNANCE_ADAPTATION_METADATA_AUDIT_INVALID"
            ) from error


__all__ = ["GovernanceStore"]
