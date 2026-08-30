"""Append-only API-owned audit, confirmation, and case-index metadata."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

from ..gfm_client import GfmProxyError, _reject_link_components
from ..gfm_hashing import canonical_json, canonical_sha256

_AUDIT_INVALID = "GOVERNANCE_SKILL_AUDIT_INVALID"
_INDEX_STATE_CONFLICT = "GOVERNANCE_CASE_INDEX_STATE_CONFLICT"


def _now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("ascii")).hexdigest()


class GovernanceSkillsStore:
    """Small durable store whose mutable facts are represented by hash-chained events."""

    def __init__(self, root: str | Path) -> None:
        self.root = _reject_link_components(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "skills-governance.sqlite3"
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
            index_integrity_table_exists = connection.execute(
                """SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'case_index_integrity_events'"""
            ).fetchone() is not None
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS skill_audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS confirmation_grants (
                    grant_id TEXT PRIMARY KEY,
                    token_hash TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL CHECK(action IN (
                        'run_governance_analysis','save_draft_report','submit_review'
                    )),
                    request_digest TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    grant_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS confirmation_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    grant_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('issued','consumed')),
                    created_at TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE,
                    FOREIGN KEY(grant_id) REFERENCES confirmation_grants(grant_id)
                );
                CREATE TABLE IF NOT EXISTS confirmed_reports (
                    report_id TEXT PRIMARY KEY,
                    draft_hash TEXT NOT NULL UNIQUE,
                    format TEXT NOT NULL CHECK(format IN ('markdown','json')),
                    content_json TEXT NOT NULL,
                    cited_hashes_json TEXT NOT NULL,
                    draft_payload_json TEXT,
                    draft_payload_hash TEXT,
                    confirmed_at TEXT NOT NULL,
                    record_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS case_index_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    case_id TEXT NOT NULL,
                    case_hash TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending','succeeded','failed')),
                    index_hash TEXT,
                    error_code TEXT,
                    created_at TEXT NOT NULL,
                    previous_event_hash TEXT,
                    event_hash TEXT NOT NULL UNIQUE
                );
                CREATE TABLE IF NOT EXISTS case_index_claims (
                    case_id TEXT PRIMARY KEY,
                    case_hash TEXT NOT NULL,
                    pending_event_hash TEXT NOT NULL UNIQUE,
                    owner_id TEXT NOT NULL,
                    lease_expires_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS case_index_integrity_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_count INTEGER NOT NULL UNIQUE CHECK(event_count > 0),
                    case_event_hash TEXT NOT NULL UNIQUE,
                    previous_integrity_hash TEXT,
                    integrity_hash TEXT NOT NULL UNIQUE
                );
                """
            )
            report_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(confirmed_reports)")
            }
            if "draft_payload_json" not in report_columns:
                connection.execute(
                    "ALTER TABLE confirmed_reports ADD COLUMN draft_payload_json TEXT"
                )
            if "draft_payload_hash" not in report_columns:
                connection.execute(
                    "ALTER TABLE confirmed_reports ADD COLUMN draft_payload_hash TEXT"
                )
            if not index_integrity_table_exists:
                self._bootstrap_index_integrity(connection)
            grants_sql_row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'confirmation_grants'"
            ).fetchone()
            if grants_sql_row is not None and "submit_review" not in str(grants_sql_row["sql"]):
                # Validate the append-only history before replacing the legacy two-action CHECK.
                self._validate(connection)
                for table in ("confirmation_grants", "confirmation_events"):
                    connection.execute(f"DROP TRIGGER IF EXISTS no_{table}_update")
                    connection.execute(f"DROP TRIGGER IF EXISTS no_{table}_delete")
                connection.executescript(
                    """
                    ALTER TABLE confirmation_events RENAME TO confirmation_events_legacy;
                    ALTER TABLE confirmation_grants RENAME TO confirmation_grants_legacy;
                    CREATE TABLE confirmation_grants (
                        grant_id TEXT PRIMARY KEY,
                        token_hash TEXT NOT NULL UNIQUE,
                        action TEXT NOT NULL CHECK(action IN (
                            'run_governance_analysis','save_draft_report','submit_review'
                        )),
                        request_digest TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT NOT NULL,
                        grant_hash TEXT NOT NULL UNIQUE
                    );
                    CREATE TABLE confirmation_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        grant_id TEXT NOT NULL,
                        event_type TEXT NOT NULL CHECK(event_type IN ('issued','consumed')),
                        created_at TEXT NOT NULL,
                        previous_event_hash TEXT,
                        event_hash TEXT NOT NULL UNIQUE,
                        FOREIGN KEY(grant_id) REFERENCES confirmation_grants(grant_id)
                    );
                    INSERT INTO confirmation_grants SELECT * FROM confirmation_grants_legacy;
                    INSERT INTO confirmation_events SELECT * FROM confirmation_events_legacy;
                    DROP TABLE confirmation_events_legacy;
                    DROP TABLE confirmation_grants_legacy;
                    """
                )
            report_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(confirmed_reports)")
            }
            if "draft_payload_json" not in report_columns:
                connection.execute(
                    "ALTER TABLE confirmed_reports ADD COLUMN draft_payload_json TEXT"
                )
            if "draft_payload_hash" not in report_columns:
                connection.execute(
                    "ALTER TABLE confirmed_reports ADD COLUMN draft_payload_hash TEXT"
                )
            for table in (
                "skill_audit_events",
                "confirmation_grants",
                "confirmation_events",
                "confirmed_reports",
                "case_index_events",
                "case_index_integrity_events",
            ):
                connection.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS no_{table}_update
                    BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT, 'append-only'); END"""
                )
                connection.execute(
                    f"""CREATE TRIGGER IF NOT EXISTS no_{table}_delete
                    BEFORE DELETE ON {table} BEGIN SELECT RAISE(ABORT, 'append-only'); END"""
                )
            self._validate(connection)

    @staticmethod
    def _invalid() -> None:
        raise GfmProxyError(502, _AUDIT_INVALID)

    @staticmethod
    def _index_state_conflict() -> NoReturn:
        raise GfmProxyError(409, _INDEX_STATE_CONFLICT)

    def _validate(self, connection: sqlite3.Connection) -> None:
        previous: str | None = None
        rows = connection.execute(
            "SELECT * FROM skill_audit_events ORDER BY sequence"
        ).fetchall()
        for row in rows:
            payload = {
                "eventId": str(row["event_id"]),
                "kind": str(row["kind"]),
                "subjectId": str(row["subject_id"]),
                "requestHash": str(row["request_hash"]),
                "responseHash": str(row["response_hash"]),
                "status": str(row["status"]),
                "createdAt": str(row["created_at"]),
                "previousEventHash": row["previous_event_hash"],
            }
            if row["previous_event_hash"] != previous or canonical_sha256(payload) != row[
                "event_hash"
            ]:
                self._invalid()
            previous = str(row["event_hash"])

        grants = connection.execute(
            "SELECT * FROM confirmation_grants ORDER BY grant_id"
        ).fetchall()
        for row in grants:
            try:
                payload = json.loads(str(row["payload_json"]))
            except ValueError:
                self._invalid()
            payload_hash = canonical_sha256(payload)
            grant_payload = {
                "grantId": str(row["grant_id"]),
                "tokenHash": str(row["token_hash"]),
                "action": str(row["action"]),
                "requestDigest": str(row["request_digest"]),
                "payloadHash": payload_hash,
                "createdAt": str(row["created_at"]),
                "expiresAt": str(row["expires_at"]),
            }
            if payload_hash != row["payload_hash"] or canonical_sha256(grant_payload) != row[
                "grant_hash"
            ]:
                self._invalid()
            self._validate_confirmation_events(connection, str(row["grant_id"]))

        reports = connection.execute("SELECT * FROM confirmed_reports").fetchall()
        for row in reports:
            try:
                content = json.loads(str(row["content_json"]))
                cited_hashes = json.loads(str(row["cited_hashes_json"]))
            except ValueError:
                self._invalid()
            payload = {
                "reportId": str(row["report_id"]),
                "draftHash": str(row["draft_hash"]),
                "format": str(row["format"]),
                "content": content,
                "citedHashes": cited_hashes,
                "confirmedAt": str(row["confirmed_at"]),
            }
            if (row["draft_payload_json"] is None) != (
                row["draft_payload_hash"] is None
            ):
                self._invalid()
            if row["draft_payload_json"] is not None:
                try:
                    draft_payload = json.loads(str(row["draft_payload_json"]))
                except ValueError:
                    self._invalid()
                payload_hash = canonical_sha256(draft_payload)
                if payload_hash != row["draft_payload_hash"]:
                    self._invalid()
                payload["draftPayloadHash"] = payload_hash
            if canonical_sha256(payload) != row["record_hash"]:
                self._invalid()

        case_ids = connection.execute(
            "SELECT DISTINCT case_id FROM case_index_events ORDER BY case_id"
        ).fetchall()
        for row in case_ids:
            self._validate_index_events(connection, str(row["case_id"]))
        self._validate_index_integrity(connection)
        self._validate_index_claims(connection)

    def _validate_confirmation_events(
        self, connection: sqlite3.Connection, grant_id: str
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            connection.execute(
                "SELECT * FROM confirmation_events WHERE grant_id = ? ORDER BY sequence",
                (grant_id,),
            ).fetchall()
        )
        previous: str | None = None
        for index, row in enumerate(rows):
            payload = {
                "grantId": grant_id,
                "eventType": str(row["event_type"]),
                "createdAt": str(row["created_at"]),
                "previousEventHash": row["previous_event_hash"],
            }
            valid_type = row["event_type"] == ("issued" if index == 0 else "consumed")
            if (
                not valid_type
                or index > 1
                or row["previous_event_hash"] != previous
                or canonical_sha256(payload) != row["event_hash"]
            ):
                self._invalid()
            previous = str(row["event_hash"])
        if not rows:
            self._invalid()
        return rows

    def _validate_index_events(
        self, connection: sqlite3.Connection, case_id: str
    ) -> tuple[sqlite3.Row, ...]:
        rows = tuple(
            connection.execute(
                "SELECT * FROM case_index_events WHERE case_id = ? ORDER BY sequence",
                (case_id,),
            ).fetchall()
        )
        previous: str | None = None
        pending_open = False
        pending_hash: str | None = None
        last_success_hash: str | None = None
        for row in rows:
            status = str(row["status"])
            if status == "pending":
                if pending_open or str(row["case_hash"]) == last_success_hash:
                    self._invalid()
                pending_open = True
                pending_hash = str(row["case_hash"])
            else:
                if not pending_open or str(row["case_hash"]) != pending_hash:
                    self._invalid()
                pending_open = False
                pending_hash = None
                if status == "succeeded":
                    last_success_hash = str(row["case_hash"])
            if (status == "succeeded") != (row["index_hash"] is not None):
                self._invalid()
            if (status == "failed") != (row["error_code"] is not None):
                self._invalid()
            payload = {
                "eventId": str(row["event_id"]),
                "caseId": case_id,
                "caseHash": str(row["case_hash"]),
                "status": status,
                "indexHash": row["index_hash"],
                "errorCode": row["error_code"],
                "createdAt": str(row["created_at"]),
                "previousEventHash": row["previous_event_hash"],
            }
            if row["previous_event_hash"] != previous or canonical_sha256(payload) != row[
                "event_hash"
            ]:
                self._invalid()
            previous = str(row["event_hash"])
        return rows

    def _validate_index_claims(self, connection: sqlite3.Connection) -> None:
        claims = connection.execute("SELECT * FROM case_index_claims").fetchall()
        for claim in claims:
            current = connection.execute(
                "SELECT * FROM case_index_events WHERE case_id = ? ORDER BY sequence DESC LIMIT 1",
                (str(claim["case_id"]),),
            ).fetchone()
            try:
                lease_expires_at = datetime.fromisoformat(
                    str(claim["lease_expires_at"]).replace("Z", "+00:00")
                )
            except ValueError:
                self._invalid()
            if (
                current is None
                or current["status"] != "pending"
                or current["case_hash"] != claim["case_hash"]
                or current["event_hash"] != claim["pending_event_hash"]
                or not str(claim["owner_id"])
                or lease_expires_at.tzinfo is None
            ):
                self._invalid()

    def _append_index_integrity(
        self, connection: sqlite3.Connection, case_event_hash: str
    ) -> str:
        previous = connection.execute(
            """SELECT event_count, integrity_hash FROM case_index_integrity_events
            ORDER BY event_count DESC LIMIT 1"""
        ).fetchone()
        event_count = int(previous["event_count"]) + 1 if previous else 1
        previous_hash = str(previous["integrity_hash"]) if previous else None
        payload = {
            "eventCount": event_count,
            "caseEventHash": case_event_hash,
            "previousIntegrityHash": previous_hash,
        }
        integrity_hash = canonical_sha256(payload)
        connection.execute(
            """INSERT INTO case_index_integrity_events
            (event_count, case_event_hash, previous_integrity_hash, integrity_hash)
            VALUES (?, ?, ?, ?)""",
            (event_count, case_event_hash, previous_hash, integrity_hash),
        )
        return integrity_hash

    def _bootstrap_index_integrity(self, connection: sqlite3.Connection) -> None:
        if connection.execute(
            "SELECT 1 FROM case_index_integrity_events LIMIT 1"
        ).fetchone() is not None:
            return
        events = connection.execute(
            "SELECT event_hash FROM case_index_events ORDER BY sequence"
        ).fetchall()
        for event in events:
            self._append_index_integrity(connection, str(event["event_hash"]))

    def _validate_index_integrity(self, connection: sqlite3.Connection) -> None:
        case_events = connection.execute(
            "SELECT event_hash FROM case_index_events ORDER BY sequence"
        ).fetchall()
        integrity_events = connection.execute(
            "SELECT * FROM case_index_integrity_events ORDER BY event_count"
        ).fetchall()
        if len(case_events) != len(integrity_events):
            self._invalid()
        previous: str | None = None
        for event_count, (case_event, integrity_event) in enumerate(
            zip(case_events, integrity_events, strict=True),
            start=1,
        ):
            payload = {
                "eventCount": event_count,
                "caseEventHash": str(case_event["event_hash"]),
                "previousIntegrityHash": previous,
            }
            if (
                integrity_event["event_count"] != event_count
                or integrity_event["case_event_hash"] != case_event["event_hash"]
                or integrity_event["previous_integrity_hash"] != previous
                or integrity_event["integrity_hash"] != canonical_sha256(payload)
            ):
                self._invalid()
            previous = str(integrity_event["integrity_hash"])

    def append_audit(
        self,
        *,
        kind: str,
        subject_id: str,
        request_hash: str,
        response_hash: str,
        status: str,
    ) -> str:
        created_at = _timestamp(_now())
        event_id = f"audit-{uuid.uuid4().hex}"
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                row = connection.execute(
                    "SELECT event_hash FROM skill_audit_events ORDER BY sequence DESC LIMIT 1"
                ).fetchone()
                previous = str(row["event_hash"]) if row else None
                payload = {
                    "eventId": event_id,
                    "kind": kind,
                    "subjectId": subject_id,
                    "requestHash": request_hash,
                    "responseHash": response_hash,
                    "status": status,
                    "createdAt": created_at,
                    "previousEventHash": previous,
                }
                event_hash = canonical_sha256(payload)
                connection.execute(
                    """INSERT INTO skill_audit_events
                    (event_id, kind, subject_id, request_hash, response_hash, status,
                     created_at, previous_event_hash, event_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id,
                        kind,
                        subject_id,
                        request_hash,
                        response_hash,
                        status,
                        created_at,
                        previous,
                        event_hash,
                    ),
                )
                return event_hash
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_SKILL_AUDIT_PERSIST_FAILED") from error

    def issue_confirmation(
        self,
        *,
        action: str,
        request_digest: str,
        payload: dict[str, Any],
        ttl_seconds: int,
    ) -> tuple[str, datetime]:
        token = f"governance-confirm-{secrets.token_hex(32)}"
        token_digest = _token_hash(token)
        grant_id = f"grant-{uuid.uuid4().hex}"
        created_at = _now()
        expires_at = created_at + timedelta(seconds=ttl_seconds)
        created_text = _timestamp(created_at)
        expires_text = _timestamp(expires_at)
        payload_json = canonical_json(payload)
        payload_hash = canonical_sha256(payload)
        grant_payload = {
            "grantId": grant_id,
            "tokenHash": token_digest,
            "action": action,
            "requestDigest": request_digest,
            "payloadHash": payload_hash,
            "createdAt": created_text,
            "expiresAt": expires_text,
        }
        grant_hash = canonical_sha256(grant_payload)
        issued_payload = {
            "grantId": grant_id,
            "eventType": "issued",
            "createdAt": created_text,
            "previousEventHash": None,
        }
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                connection.execute(
                    """INSERT INTO confirmation_grants
                    (grant_id, token_hash, action, request_digest, payload_json, payload_hash,
                     created_at, expires_at, grant_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        grant_id,
                        token_digest,
                        action,
                        request_digest,
                        payload_json,
                        payload_hash,
                        created_text,
                        expires_text,
                        grant_hash,
                    ),
                )
                connection.execute(
                    """INSERT INTO confirmation_events
                    (grant_id, event_type, created_at, previous_event_hash, event_hash)
                    VALUES (?, 'issued', ?, NULL, ?)""",
                    (grant_id, created_text, canonical_sha256(issued_payload)),
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CONFIRMATION_PERSIST_FAILED") from error
        return token, expires_at

    def consume_confirmation(self, token: str) -> tuple[str, str, dict[str, Any]]:
        token_digest = _token_hash(token)
        consumed_at = _timestamp(_now())
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                row = connection.execute(
                    "SELECT * FROM confirmation_grants WHERE token_hash = ?", (token_digest,)
                ).fetchone()
                if row is None:
                    raise GfmProxyError(404, "GOVERNANCE_CONFIRMATION_INVALID")
                events = self._validate_confirmation_events(connection, str(row["grant_id"]))
                if len(events) > 1:
                    raise GfmProxyError(409, "GOVERNANCE_CONFIRMATION_ALREADY_USED")
                expires_at = datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00"))
                if expires_at <= _now():
                    raise GfmProxyError(410, "GOVERNANCE_CONFIRMATION_EXPIRED")
                previous = str(events[-1]["event_hash"])
                consumed_payload = {
                    "grantId": str(row["grant_id"]),
                    "eventType": "consumed",
                    "createdAt": consumed_at,
                    "previousEventHash": previous,
                }
                connection.execute(
                    """INSERT INTO confirmation_events
                    (grant_id, event_type, created_at, previous_event_hash, event_hash)
                    VALUES (?, 'consumed', ?, ?, ?)""",
                    (
                        str(row["grant_id"]),
                        consumed_at,
                        previous,
                        canonical_sha256(consumed_payload),
                    ),
                )
                payload = json.loads(str(row["payload_json"]))
                return str(row["action"]), str(row["request_digest"]), payload
        except GfmProxyError:
            raise
        except (sqlite3.Error, ValueError) as error:
            raise GfmProxyError(502, "GOVERNANCE_CONFIRMATION_READ_FAILED") from error

    def save_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        report_id = f"report-{uuid.uuid4().hex}"
        confirmed_at = _timestamp(_now())
        record = {
            "reportId": report_id,
            "draftHash": payload["draftHash"],
            "format": payload["format"],
            "content": payload["content"],
            "citedHashes": payload["citedHashes"],
            "confirmedAt": confirmed_at,
        }
        draft_payload_hash = canonical_sha256(payload)
        record["draftPayloadHash"] = draft_payload_hash
        record_hash = canonical_sha256(record)
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                connection.execute(
                    """INSERT INTO confirmed_reports
                    (report_id, draft_hash, format, content_json, cited_hashes_json,
                     draft_payload_json, draft_payload_hash, confirmed_at, record_hash)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        report_id,
                        payload["draftHash"],
                        payload["format"],
                        canonical_json(payload["content"]),
                        canonical_json(payload["citedHashes"]),
                        canonical_json(payload),
                        draft_payload_hash,
                        confirmed_at,
                        record_hash,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise GfmProxyError(409, "GOVERNANCE_REPORT_ALREADY_SAVED") from error
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_REPORT_PERSIST_FAILED") from error
        return {**record, "draft": payload, "recordHash": record_hash}

    def _append_index_event(
        self,
        connection: sqlite3.Connection,
        *,
        case_id: str,
        case_hash: str,
        status: str,
        index_hash: str | None,
        error_code: str | None,
        expected_pending_hash: str | None,
    ) -> str:
        created_at = _timestamp(_now())
        event_id = f"index-event-{uuid.uuid4().hex}"
        rows = self._validate_index_events(connection, case_id)
        previous = str(rows[-1]["event_hash"]) if rows else None
        previous_status = str(rows[-1]["status"]) if rows else None
        previous_case_hash = str(rows[-1]["case_hash"]) if rows else None
        last_success_hash = next(
            (
                str(row["case_hash"])
                for row in reversed(rows)
                if row["status"] == "succeeded"
            ),
            None,
        )
        valid_fields = (
            (status == "pending" and index_hash is None and error_code is None)
            or (status == "succeeded" and index_hash is not None and error_code is None)
            or (status == "failed" and index_hash is None and error_code is not None)
        )
        if status == "pending":
            valid_transition = (
                expected_pending_hash is None
                and previous_status != "pending"
                and case_hash != last_success_hash
            )
        else:
            valid_transition = (
                status in {"succeeded", "failed"}
                and previous_status == "pending"
                and previous_case_hash == case_hash
                and (
                    expected_pending_hash is None
                    or expected_pending_hash == previous
                )
            )
        if not valid_fields or not valid_transition:
            self._index_state_conflict()
        payload = {
            "eventId": event_id,
            "caseId": case_id,
            "caseHash": case_hash,
            "status": status,
            "indexHash": index_hash,
            "errorCode": error_code,
            "createdAt": created_at,
            "previousEventHash": previous,
        }
        event_hash = canonical_sha256(payload)
        connection.execute(
            """INSERT INTO case_index_events
            (event_id, case_id, case_hash, status, index_hash, error_code,
             created_at, previous_event_hash, event_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                case_id,
                case_hash,
                status,
                index_hash,
                error_code,
                created_at,
                previous,
                event_hash,
            ),
        )
        self._append_index_integrity(connection, event_hash)
        return event_hash

    def append_index_event(
        self,
        *,
        case_id: str,
        case_hash: str,
        status: str,
        index_hash: str | None = None,
        error_code: str | None = None,
        expected_pending_hash: str | None = None,
    ) -> str:
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                return self._append_index_event(
                    connection,
                    case_id=case_id,
                    case_hash=case_hash,
                    status=status,
                    index_hash=index_hash,
                    error_code=error_code,
                    expected_pending_hash=expected_pending_hash,
                )
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_INDEX_AUDIT_FAILED") from error

    @staticmethod
    def _index_terminal_state(row: sqlite3.Row, case_hash: str) -> dict[str, Any]:
        status = str(row["status"])
        return {
            "state": (
                "succeeded"
                if status == "succeeded" and str(row["case_hash"]) == case_hash
                else "failed"
            ),
            "pendingEventHash": None,
        }

    def claim_index_attempt(
        self,
        *,
        case_id: str,
        case_hash: str,
        owner_id: str,
        lease_seconds: float,
        expected_pending_hash: str | None = None,
    ) -> dict[str, Any]:
        now = _now()
        lease_expires_at = _timestamp(now + timedelta(seconds=lease_seconds))
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                rows = self._validate_index_events(connection, case_id)
                current = rows[-1] if rows else None
                if expected_pending_hash is not None:
                    pending_index = next(
                        (
                            index
                            for index, row in enumerate(rows)
                            if row["event_hash"] == expected_pending_hash
                            and row["status"] == "pending"
                        ),
                        None,
                    )
                    if pending_index is None:
                        self._index_state_conflict()
                    if pending_index + 1 < len(rows):
                        return self._index_terminal_state(rows[pending_index + 1], case_hash)
                elif current is not None and current["status"] == "succeeded":
                    if str(current["case_hash"]) == case_hash:
                        return self._index_terminal_state(current, case_hash)
                if current is not None and current["status"] == "pending":
                    pending_hash = str(current["event_hash"])
                    claim = connection.execute(
                        "SELECT * FROM case_index_claims WHERE case_id = ?",
                        (case_id,),
                    ).fetchone()
                    if claim is not None:
                        expires_at = datetime.fromisoformat(
                            str(claim["lease_expires_at"]).replace("Z", "+00:00")
                        )
                        if expires_at > now:
                            return {
                                "state": "waiting",
                                "pendingEventHash": pending_hash,
                            }
                    self._append_index_event(
                        connection,
                        case_id=case_id,
                        case_hash=str(current["case_hash"]),
                        status="failed",
                        index_hash=None,
                        error_code="GOVERNANCE_CASE_INDEX_INTERRUPTED",
                        expected_pending_hash=pending_hash,
                    )
                    connection.execute("DELETE FROM case_index_claims WHERE case_id = ?", (case_id,))
                pending_hash = self._append_index_event(
                    connection,
                    case_id=case_id,
                    case_hash=case_hash,
                    status="pending",
                    index_hash=None,
                    error_code=None,
                    expected_pending_hash=None,
                )
                connection.execute(
                    """INSERT INTO case_index_claims
                    (case_id, case_hash, pending_event_hash, owner_id, lease_expires_at)
                    VALUES (?, ?, ?, ?, ?)""",
                    (case_id, case_hash, pending_hash, owner_id, lease_expires_at),
                )
                return {"state": "claimed", "pendingEventHash": pending_hash}
        except GfmProxyError:
            raise
        except (sqlite3.Error, ValueError) as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_INDEX_AUDIT_FAILED") from error

    def renew_index_lease(
        self,
        *,
        case_id: str,
        pending_event_hash: str,
        owner_id: str,
        lease_seconds: float,
    ) -> bool:
        lease_expires_at = _timestamp(_now() + timedelta(seconds=lease_seconds))
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                updated = connection.execute(
                    """UPDATE case_index_claims SET lease_expires_at = ?
                    WHERE case_id = ? AND pending_event_hash = ? AND owner_id = ?""",
                    (lease_expires_at, case_id, pending_event_hash, owner_id),
                )
                return updated.rowcount == 1
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_INDEX_AUDIT_FAILED") from error

    def finish_index_attempt(
        self,
        *,
        case_id: str,
        case_hash: str,
        pending_event_hash: str,
        owner_id: str,
        status: str,
        index_hash: str | None = None,
        error_code: str | None = None,
    ) -> str:
        try:
            with self._lock, self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                self._validate(connection)
                claim = connection.execute(
                    """SELECT 1 FROM case_index_claims
                    WHERE case_id = ? AND case_hash = ?
                      AND pending_event_hash = ? AND owner_id = ?""",
                    (case_id, case_hash, pending_event_hash, owner_id),
                ).fetchone()
                if claim is None:
                    self._index_state_conflict()
                event_hash = self._append_index_event(
                    connection,
                    case_id=case_id,
                    case_hash=case_hash,
                    status=status,
                    index_hash=index_hash,
                    error_code=error_code,
                    expected_pending_hash=pending_event_hash,
                )
                connection.execute(
                    """DELETE FROM case_index_claims
                    WHERE case_id = ? AND pending_event_hash = ? AND owner_id = ?""",
                    (case_id, pending_event_hash, owner_id),
                )
                return event_hash
        except GfmProxyError:
            raise
        except sqlite3.Error as error:
            raise GfmProxyError(502, "GOVERNANCE_CASE_INDEX_AUDIT_FAILED") from error

    def index_status(self, case_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            self._validate(connection)
            rows = self._validate_index_events(connection, case_id)
            if not rows:
                return None
            row = rows[-1]
            return {
                "caseId": case_id,
                "caseHash": str(row["case_hash"]),
                "status": str(row["status"]),
                "indexHash": row["index_hash"],
                "errorCode": row["error_code"],
                "updatedAt": str(row["created_at"]),
                "eventHash": str(row["event_hash"]),
            }

    def validate(self) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            self._validate(connection)
            row = connection.execute(
                "SELECT COUNT(*) AS count, MAX(event_hash) AS ignored FROM skill_audit_events"
            ).fetchone()
            latest = connection.execute(
                "SELECT event_hash FROM skill_audit_events ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            index_latest = connection.execute(
                """SELECT event_count, integrity_hash FROM case_index_integrity_events
                ORDER BY event_count DESC LIMIT 1"""
            ).fetchone()
        return {
            "valid": True,
            "eventCount": int(row["count"]),
            "headHash": str(latest["event_hash"]) if latest else None,
            "caseIndexEventCount": int(index_latest["event_count"]) if index_latest else 0,
            "caseIndexHeadHash": (
                str(index_latest["integrity_hash"]) if index_latest else None
            ),
        }


__all__ = ["GovernanceSkillsStore"]
