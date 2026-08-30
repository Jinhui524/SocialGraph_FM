"""Versioned SQLite document knowledge and append-only human project memory."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .governance import (
    EvidenceItem,
    GovernanceFinding,
    validate_similar_case_provenance,
)
from .retrieval import StructuralIndex


SCHEMA_VERSION = "socialgraph-fm.core-knowledge-sqlite/2.2"
_HASH_PATTERN = r"^[0-9a-f]{64}$"
_CANONICAL_SCHEMA_DDL = """
CREATE TABLE schema_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version TEXT NOT NULL,
    schema_fingerprint TEXT NOT NULL
);
CREATE TABLE knowledge_documents (
    document_hash TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    source_uri TEXT,
    payload_json TEXT NOT NULL
);
CREATE VIRTUAL TABLE knowledge_fts USING fts5(
    document_hash UNINDEXED,
    title,
    body,
    category,
    tokenize = 'unicode61'
);
CREATE TABLE registered_findings (
    finding_hash TEXT PRIMARY KEY,
    graph_version_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE registered_adaptation_evidence (
    finding_hash TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY (finding_hash, evidence_hash),
    FOREIGN KEY (finding_hash) REFERENCES registered_findings(finding_hash)
);
CREATE TABLE project_memory (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    record_hash TEXT NOT NULL UNIQUE,
    finding_hash TEXT NOT NULL,
    review_status TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (finding_hash) REFERENCES registered_findings(finding_hash)
);
CREATE TRIGGER project_memory_no_update
BEFORE UPDATE ON project_memory
BEGIN SELECT RAISE(ABORT, 'project memory is append-only'); END;
CREATE TRIGGER project_memory_no_delete
BEFORE DELETE ON project_memory
BEGIN SELECT RAISE(ABORT, 'project memory is append-only'); END;
CREATE TRIGGER registered_findings_no_update
BEFORE UPDATE ON registered_findings
BEGIN SELECT RAISE(ABORT, 'registered finding provenance is immutable'); END;
CREATE TRIGGER registered_findings_no_delete
BEFORE DELETE ON registered_findings
BEGIN SELECT RAISE(ABORT, 'registered finding provenance is immutable'); END;
CREATE TRIGGER registered_adaptation_no_update
BEFORE UPDATE ON registered_adaptation_evidence
BEGIN SELECT RAISE(ABORT, 'registered adaptation provenance is immutable'); END;
CREATE TRIGGER registered_adaptation_no_delete
BEFORE DELETE ON registered_adaptation_evidence
BEGIN SELECT RAISE(ABORT, 'registered adaptation provenance is immutable'); END;
"""


def _quote_sqlite_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _normalize_schema_sql(sql: str | None) -> str | None:
    return None if sql is None else re.sub(r"\s+", " ", sql).strip()


def _describe_layout(connection: sqlite3.Connection) -> dict[str, Any]:
    objects = tuple(
        {
            "type": row[0],
            "name": row[1],
            "table": row[2],
            "sql": _normalize_schema_sql(row[3]),
        }
        for row in connection.execute(
            """
            SELECT type, name, tbl_name, sql
            FROM sqlite_master
            ORDER BY type, name, tbl_name
            """
        )
    )
    table_names = tuple(item["name"] for item in objects if item["type"] == "table")
    tables: dict[str, Any] = {}
    for table_name in table_names:
        quoted_table = _quote_sqlite_identifier(table_name)
        table_info = tuple(connection.execute(f"PRAGMA table_info({quoted_table})"))
        foreign_keys = tuple(connection.execute(f"PRAGMA foreign_key_list({quoted_table})"))
        index_list = tuple(connection.execute(f"PRAGMA index_list({quoted_table})"))
        indexes: dict[str, Any] = {}
        for index_row in index_list:
            index_name = index_row[1]
            quoted_index = _quote_sqlite_identifier(index_name)
            indexes[index_name] = {
                "indexInfo": tuple(connection.execute(f"PRAGMA index_info({quoted_index})")),
                "indexXInfo": tuple(
                    connection.execute(f"PRAGMA index_xinfo({quoted_index})")
                ),
            }
        tables[table_name] = {
            "tableInfo": table_info,
            "foreignKeyList": foreign_keys,
            "indexList": index_list,
            "indexes": indexes,
        }
    return {
        "objects": objects,
        "tables": tables,
        "sqliteSequenceExpected": "sqlite_sequence" in table_names,
        "ftsShadowInventory": tuple(
            name for name in table_names if name.startswith("knowledge_fts_")
        ),
    }


def _expected_layout() -> dict[str, Any]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_CANONICAL_SCHEMA_DDL)
        return _describe_layout(connection)
    finally:
        connection.close()


_EXPECTED_LAYOUT = _expected_layout()
SCHEMA_FINGERPRINT = canonical_sha256(_EXPECTED_LAYOUT)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class KnowledgeDocument(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-knowledge-document/2.0"] = Field(
        alias="schemaVersion"
    )
    category: Literal[
        "data-card", "model-card", "paper-summary", "governance-rule", "limitation"
    ]
    title: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=100_000)
    source_uri: str | None = Field(default=None, alias="sourceUri", min_length=1, max_length=4000)
    document_hash: str = Field(alias="documentHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"document_hash"})
        )
        if self.document_hash != expected:
            raise ValueError("documentHash does not match canonical document content")
        return self

    @classmethod
    def create(
        cls,
        *,
        category: Literal[
            "data-card", "model-card", "paper-summary", "governance-rule", "limitation"
        ],
        title: str,
        body: str,
        source_uri: str | None,
    ) -> KnowledgeDocument:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-knowledge-document/2.0",
            "category": category,
            "title": title,
            "body": body,
            "sourceUri": source_uri,
        }
        payload["documentHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class KnowledgeSearchResult(_StrictModel):
    document_hash: str = Field(alias="documentHash", pattern=_HASH_PATTERN)
    title: str
    category: str
    source_uri: str | None = Field(alias="sourceUri")
    bm25_score: float = Field(alias="bm25Score")
    document: KnowledgeDocument


class ProjectReviewRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-project-review/2.0"] = Field(
        alias="schemaVersion"
    )
    finding_hash: str = Field(alias="findingHash", pattern=_HASH_PATTERN)
    review_status: Literal["pending", "confirmed", "rejected"] = Field(alias="reviewStatus")
    reviewer_id: str = Field(alias="reviewerId", min_length=1, max_length=500)
    annotation: str = Field(min_length=1, max_length=20_000)
    adaptation_evidence_hashes: tuple[str, ...] = Field(
        default=(), alias="adaptationEvidenceHashes", strict=False
    )
    created_at: str = Field(alias="createdAt", min_length=1, max_length=100)
    record_hash: str = Field(alias="recordHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_record(self):
        try:
            parsed = datetime.fromisoformat(self.created_at.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError("createdAt must be an ISO-8601 datetime") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("createdAt must include a timezone")
        if any(not re.fullmatch(_HASH_PATTERN, item) for item in self.adaptation_evidence_hashes):
            raise ValueError("adaptation evidence hashes must be lowercase SHA-256 values")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        )
        if self.record_hash != expected:
            raise ValueError("recordHash does not match canonical project review content")
        return self

    @classmethod
    def create(
        cls,
        *,
        finding_hash: str,
        review_status: Literal["pending", "confirmed", "rejected"],
        reviewer_id: str,
        annotation: str,
        created_at: str,
        adaptation_evidence_hashes: tuple[str, ...] = (),
    ) -> ProjectReviewRecord:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-project-review/2.0",
            "findingHash": finding_hash,
            "reviewStatus": review_status,
            "reviewerId": reviewer_id,
            "annotation": annotation,
            "adaptationEvidenceHashes": adaptation_evidence_hashes,
            "createdAt": created_at,
        }
        payload["recordHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class KnowledgeStore:
    """Each operation validates canonical records at the database boundary."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            objects = connection.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
            ).fetchall()
            if objects:
                try:
                    version_row = connection.execute(
                        "SELECT schema_version FROM schema_metadata WHERE singleton = ?",
                        (1,),
                    ).fetchone()
                except sqlite3.DatabaseError as error:
                    raise ValueError("knowledge SQLite layout metadata is missing or malformed") from error
                if version_row is None or version_row[0] != SCHEMA_VERSION:
                    raise ValueError("unsupported knowledge SQLite schema version")
                try:
                    fingerprint_row = connection.execute(
                        "SELECT schema_fingerprint FROM schema_metadata WHERE singleton = ?",
                        (1,),
                    ).fetchone()
                except sqlite3.DatabaseError as error:
                    raise ValueError("knowledge SQLite schema fingerprint is missing") from error
                if fingerprint_row is None or fingerprint_row[0] != SCHEMA_FINGERPRINT:
                    raise ValueError("knowledge SQLite schema fingerprint mismatch")
                self._validate_layout(connection)
                return
            connection.executescript(
                f"""
                BEGIN IMMEDIATE;
                {_CANONICAL_SCHEMA_DDL}
                INSERT INTO schema_metadata(
                    singleton, schema_version, schema_fingerprint
                ) VALUES (1, '{SCHEMA_VERSION}', '{SCHEMA_FINGERPRINT}');
                COMMIT;
                """
            )
            self._validate_layout(connection)

    @staticmethod
    def _validate_layout(connection: sqlite3.Connection) -> None:
        try:
            observed_layout = _describe_layout(connection)
        except sqlite3.DatabaseError as error:
            raise ValueError("knowledge SQLite layout cannot be described exactly") from error
        if observed_layout != _EXPECTED_LAYOUT:
            raise ValueError("knowledge SQLite layout descriptor mismatch")
        if canonical_sha256(observed_layout) != SCHEMA_FINGERPRINT:
            raise ValueError("knowledge SQLite actual layout fingerprint mismatch")

    def add_document(self, document: KnowledgeDocument) -> None:
        validated = KnowledgeDocument.model_validate(
            document.model_dump(mode="python", by_alias=True)
        )
        payload_json = canonical_json(validated)
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO knowledge_documents(
                        document_hash, category, title, body, source_uri, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        validated.document_hash,
                        validated.category,
                        validated.title,
                        validated.body,
                        validated.source_uri,
                        payload_json,
                    ),
                )
                connection.execute(
                    "INSERT INTO knowledge_fts(document_hash, title, body, category) VALUES (?, ?, ?, ?)",
                    (
                        validated.document_hash,
                        validated.title,
                        validated.body,
                        validated.category,
                    ),
                )
            except sqlite3.IntegrityError as error:
                existing = self.get_document(validated.document_hash)
                if existing != validated:
                    raise ValueError("knowledge document hash collision") from error

    def get_document(self, document_hash: str) -> KnowledgeDocument:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM knowledge_documents WHERE document_hash = ?",
                (document_hash,),
            ).fetchone()
        if row is None:
            raise ValueError("unknown knowledge document hash")
        try:
            document = KnowledgeDocument.model_validate_json(row[0])
        except Exception as error:
            raise ValueError("invalid stored knowledge document") from error
        if document.document_hash != document_hash:
            raise ValueError("invalid stored knowledge document hash binding")
        return document

    def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeSearchResult, ...]:
        if limit < 1 or limit > 100:
            raise ValueError("knowledge search limit must be between 1 and 100")
        if not isinstance(query, str):
            raise TypeError("knowledge query must be text")
        # Quoted token conjunction preserves FTS5/BM25 while disabling operators and syntax.
        tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)
        if not tokens:
            return ()
        match_expression = " ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)
        with self._connect() as connection:
            canonical_rows = connection.execute(
                "SELECT document_hash, payload_json FROM knowledge_documents ORDER BY document_hash"
            ).fetchall()
            indexed_rows = connection.execute(
                "SELECT document_hash, title, body, category FROM knowledge_fts ORDER BY document_hash"
            ).fetchall()
            canonical_documents: dict[str, KnowledgeDocument] = {}
            for document_hash, payload_json in canonical_rows:
                try:
                    document = KnowledgeDocument.model_validate_json(payload_json)
                except Exception as error:
                    raise ValueError("invalid stored knowledge document") from error
                if document.document_hash != document_hash:
                    raise ValueError("invalid stored knowledge document hash binding")
                canonical_documents[document_hash] = document
            expected_index = [
                (item.document_hash, item.title, item.body, item.category)
                for item in canonical_documents.values()
            ]
            if indexed_rows != expected_index:
                raise ValueError("invalid FTS index does not match canonical knowledge")
            rows = connection.execute(
                """
                SELECT document_hash, title, body, category, bm25(knowledge_fts) AS rank
                FROM knowledge_fts
                WHERE knowledge_fts MATCH ?
                ORDER BY rank ASC, document_hash ASC
                LIMIT ?
                """,
                (match_expression, limit),
            ).fetchall()
        results: list[KnowledgeSearchResult] = []
        for document_hash, indexed_title, indexed_body, indexed_category, rank in rows:
            document = canonical_documents[document_hash]
            if (
                indexed_title != document.title
                or indexed_body != document.body
                or indexed_category != document.category
            ):
                raise ValueError("invalid FTS index does not match canonical knowledge")
            results.append(
                KnowledgeSearchResult(
                    document_hash=document.document_hash,
                    title=document.title,
                    category=document.category,
                    source_uri=document.source_uri,
                    bm25_score=float(rank),
                    document=document,
                )
            )
        return tuple(results)

    def append_review(self, review: ProjectReviewRecord) -> None:
        validated = ProjectReviewRecord.model_validate(
            review.model_dump(mode="python", by_alias=True)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            finding_row = connection.execute(
                "SELECT payload_json FROM registered_findings WHERE finding_hash = ?",
                (validated.finding_hash,),
            ).fetchone()
            if finding_row is None:
                raise ValueError("project review requires a registered finding")
            try:
                registered_finding = GovernanceFinding.model_validate_json(finding_row[0])
            except Exception as error:
                raise ValueError("registered finding provenance is invalid") from error
            if registered_finding.finding_hash != validated.finding_hash:
                raise ValueError("registered finding provenance hash mismatch")
            for evidence_hash in validated.adaptation_evidence_hashes:
                evidence_row = connection.execute(
                    """
                    SELECT payload_json FROM registered_adaptation_evidence
                    WHERE finding_hash = ? AND evidence_hash = ?
                    """,
                    (validated.finding_hash, evidence_hash),
                ).fetchone()
                if evidence_row is None:
                    raise ValueError("project review references unknown adaptation evidence")
                try:
                    evidence = EvidenceItem.model_validate_json(evidence_row[0])
                except Exception as error:
                    raise ValueError("registered adaptation evidence is invalid") from error
                if evidence.evidence_hash != evidence_hash:
                    raise ValueError("registered adaptation evidence hash mismatch")
            connection.execute(
                """
                INSERT INTO project_memory(
                    record_hash, finding_hash, review_status, reviewer_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    validated.record_hash,
                    validated.finding_hash,
                    validated.review_status,
                    validated.reviewer_id,
                    validated.created_at,
                    canonical_json(validated),
                ),
            )

    def register_finding(
        self,
        finding: GovernanceFinding,
        *,
        structural_index: StructuralIndex | None = None,
    ) -> None:
        validated = GovernanceFinding.model_validate(
            finding.model_dump(mode="python", by_alias=True)
        )
        if validated.similar_cases and structural_index is None:
            raise ValueError("similar cases require a registered structural query/result resolver")
        for similar_case in validated.similar_cases:
            validate_similar_case_provenance(similar_case, structural_index)
        payload = canonical_json(validated)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM registered_findings WHERE finding_hash = ?",
                (validated.finding_hash,),
            ).fetchone()
            if row is not None:
                try:
                    existing = GovernanceFinding.model_validate_json(row[0])
                except Exception as error:
                    raise ValueError("registered finding provenance is invalid") from error
                if existing != validated:
                    raise ValueError("registered finding hash collision")
                return
            connection.execute(
                """
                INSERT INTO registered_findings(finding_hash, graph_version_hash, payload_json)
                VALUES (?, ?, ?)
                """,
                (validated.finding_hash, validated.graph_version_hash, payload),
            )

    def register_adaptation_evidence(
        self, finding_hash: str, evidence: EvidenceItem
    ) -> None:
        validated = EvidenceItem.model_validate(
            evidence.model_dump(mode="python", by_alias=True)
        )
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT graph_version_hash, payload_json FROM registered_findings WHERE finding_hash = ?",
                (finding_hash,),
            ).fetchone()
            if row is None:
                raise ValueError("adaptation evidence requires a registered finding")
            try:
                finding = GovernanceFinding.model_validate_json(row[1])
            except Exception as error:
                raise ValueError("registered finding provenance is invalid") from error
            if finding.finding_hash != finding_hash or row[0] != finding.graph_version_hash:
                raise ValueError("registered finding provenance hash mismatch")
            if validated.graph_version_hash != finding.graph_version_hash:
                raise ValueError("adaptation evidence graph version does not match finding")
            existing = connection.execute(
                """
                SELECT payload_json FROM registered_adaptation_evidence
                WHERE finding_hash = ? AND evidence_hash = ?
                """,
                (finding_hash, validated.evidence_hash),
            ).fetchone()
            if existing is not None:
                try:
                    observed = EvidenceItem.model_validate_json(existing[0])
                except Exception as error:
                    raise ValueError("registered adaptation evidence is invalid") from error
                if observed != validated:
                    raise ValueError("registered adaptation evidence hash collision")
                return
            connection.execute(
                """
                INSERT INTO registered_adaptation_evidence(
                    finding_hash, evidence_hash, payload_json
                ) VALUES (?, ?, ?)
                """,
                (finding_hash, validated.evidence_hash, canonical_json(validated)),
            )

    def reviews_for_finding(self, finding_hash: str) -> tuple[ProjectReviewRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_hash, payload_json
                FROM project_memory
                WHERE finding_hash = ?
                ORDER BY sequence ASC
                """,
                (finding_hash,),
            ).fetchall()
        reviews: list[ProjectReviewRecord] = []
        for record_hash, payload_json in rows:
            try:
                review = ProjectReviewRecord.model_validate_json(payload_json)
            except Exception as error:
                raise ValueError("invalid stored project review record") from error
            if review.record_hash != record_hash or review.finding_hash != finding_hash:
                raise ValueError("invalid stored project review hash binding")
            reviews.append(review)
        return tuple(reviews)


__all__ = [
    "KnowledgeDocument",
    "KnowledgeSearchResult",
    "KnowledgeStore",
    "ProjectReviewRecord",
    "SCHEMA_VERSION",
]
