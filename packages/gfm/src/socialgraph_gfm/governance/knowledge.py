"""Trusted-file ingestion and tamper-evident FTS retrieval for Governance evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256

KNOWLEDGE_SCHEMA_VERSION = "socialgraph-fm.governance-knowledge-index/1.0"
_ALLOWED_SUFFIXES = {".md", ".txt", ".json"}
_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_MAX_CHUNK_CHARS = 1_200
_MAX_QUERY_CHARS = 2_000
_CHINESE_QUERY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("模型", ("model", "method")),
    ("方法", ("method", "model")),
    ("适用", ("scope", "applicability")),
    ("范围", ("scope",)),
    ("限制", ("limitation", "limit")),
    ("局限", ("limitation",)),
    ("证据", ("evidence",)),
    ("关系", ("relation",)),
    ("协同", ("coordination",)),
    ("群组", ("group",)),
    ("数据", ("dataset", "data")),
    ("输入", ("input",)),
    ("复核", ("review",)),
    ("研判", ("review", "governance")),
    ("治理", ("governance",)),
    ("风险", ("risk",)),
    ("图谱", ("graph",)),
    ("案例", ("case",)),
    ("报告", ("report",)),
    ("知识", ("knowledge",)),
    ("少样本", ("few", "shot")),
)


def _search_expression(query: str) -> str | None:
    """Build a bounded FTS expression with minimal Chinese domain expansion."""

    terms: list[tuple[str, bool]] = []
    for token in re.findall(r"[A-Za-z0-9_]+", query):
        terms.append((token, False))
    for chinese, aliases in _CHINESE_QUERY_ALIASES:
        if chinese in query:
            terms.append((chinese, True))
            terms.extend((alias, False) for alias in aliases)
    cjk_runs = re.findall(r"[\u3400-\u9fff]+", query)
    if not any(chinese in query for chinese, _aliases in _CHINESE_QUERY_ALIASES):
        terms.extend((run, True) for run in cjk_runs if 1 < len(run) <= 12)

    unique: list[tuple[str, bool]] = []
    seen: set[tuple[str, bool]] = set()
    for token, prefix in terms:
        key = (token.casefold(), prefix)
        if key not in seen:
            seen.add(key)
            unique.append((token, prefix))
        if len(unique) == 32:
            break
    if not unique:
        return None
    parts = []
    for token, prefix in unique:
        escaped = token.replace('"', '""')
        parts.append(f'"{escaped}"*' if prefix else f'"{escaped}"')
    return " OR ".join(parts)


@dataclass(frozen=True)
class KnowledgeSource:
    label: str
    path: Path
    uri: str


@dataclass(frozen=True)
class KnowledgeResult:
    source_label: str
    source_uri: str
    content_hash: str
    chunk_hash: str
    text: str
    rank: int


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        ),
    )


def _normalize_source(path: Path) -> tuple[bytes, str]:
    resolved = path.expanduser().resolve(strict=True)
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or resolved.suffix.lower() not in _ALLOWED_SUFFIXES
        or not 1 <= resolved.stat().st_size <= _MAX_SOURCE_BYTES
    ):
        raise ValueError("knowledge source must be an explicit .md, .txt, or .json file")
    raw = resolved.read_bytes()
    if b"\x00" in raw:
        raise ValueError("knowledge source appears to be binary")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("knowledge source must be UTF-8 text") from error
    if resolved.suffix.lower() == ".json":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError("knowledge JSON source is invalid") from error
        text = canonical_json(value)
    normalized = "\n".join(
        line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    )
    normalized = normalized.strip()
    if not normalized:
        raise ValueError("knowledge source is empty after normalization")
    return raw, normalized


def _chunks(text: str) -> tuple[str, ...]:
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= _MAX_CHUNK_CHARS:
            pieces.append(paragraph)
            continue
        for start in range(0, len(paragraph), _MAX_CHUNK_CHARS):
            pieces.append(paragraph[start : start + _MAX_CHUNK_CHARS])
    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + "\n\n" + piece
        if len(candidate) <= _MAX_CHUNK_CHARS:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return tuple(packed)


def _validate_source(source: KnowledgeSource) -> KnowledgeSource:
    if (
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ._-]{0,99}", source.label)
        or not source.uri
        or len(source.uri) > 2_048
        or any(ord(character) < 32 for character in source.uri)
    ):
        raise ValueError("knowledge source label or URI is invalid")
    return source


def build_knowledge_index(root: str | Path, sources: Sequence[KnowledgeSource]) -> Path:
    """Build one immutable logical index from an explicit, non-recursive source list."""

    destination = Path(root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(destination)
    validated = tuple(
        sorted((_validate_source(value) for value in sources), key=lambda item: item.label)
    )
    if not validated or len({item.label for item in validated}) != len(validated):
        raise ValueError("knowledge source labels must be nonempty and unique")
    destination.mkdir(parents=True, exist_ok=False)
    database = destination / "knowledge.sqlite3"
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "CREATE TABLE documents (source_label TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE chunks (chunk_hash TEXT PRIMARY KEY, source_label TEXT NOT NULL, "
            "ordinal INTEGER NOT NULL, record_json TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_hash UNINDEXED, source_label UNINDEXED, text)"
        )
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES('schemaVersion', ?)",
            (KNOWLEDGE_SCHEMA_VERSION,),
        )
        for source in validated:
            raw, normalized = _normalize_source(source.path)
            content_hash = hashlib.sha256(raw).hexdigest()
            normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            chunk_hashes: list[str] = []
            for ordinal, text in enumerate(_chunks(normalized)):
                chunk_logical = {
                    "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                    "sourceLabel": source.label,
                    "sourceUri": source.uri,
                    "contentHash": content_hash,
                    "ordinal": ordinal,
                    "text": text,
                }
                chunk_hash = canonical_sha256(chunk_logical)
                chunk = {**chunk_logical, "chunkHash": chunk_hash}
                chunk_hashes.append(chunk_hash)
                chunks.append(chunk)
                encoded = json.dumps(
                    chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                connection.execute(
                    "INSERT INTO chunks(chunk_hash, source_label, ordinal, record_json) "
                    "VALUES (?, ?, ?, ?)",
                    (chunk_hash, source.label, ordinal, encoded),
                )
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_hash, source_label, text) VALUES (?, ?, ?)",
                    (chunk_hash, source.label, text),
                )
            document_logical = {
                "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                "sourceLabel": source.label,
                "sourceUri": source.uri,
                "contentHash": content_hash,
                "normalizedContentHash": normalized_hash,
                "bytes": len(raw),
                "chunkHashes": chunk_hashes,
            }
            document = {
                **document_logical,
                "documentHash": canonical_sha256(document_logical),
            }
            documents.append(document)
            connection.execute(
                "INSERT INTO documents(source_label, record_json) VALUES (?, ?)",
                (
                    source.label,
                    json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                ),
            )
        document_hashes = [value["documentHash"] for value in documents]
        chunk_hashes = [value["chunkHash"] for value in chunks]
        index_hash = canonical_sha256(
            {
                "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                "documentHashes": document_hashes,
                "chunkHashes": chunk_hashes,
            }
        )
        connection.execute("INSERT INTO metadata(key, value) VALUES('indexHash', ?)", (index_hash,))
        connection.commit()
    except BaseException:
        connection.rollback()
        connection.close()
        for path in destination.iterdir():
            path.unlink(missing_ok=True)
        destination.rmdir()
        raise
    finally:
        if connection:
            connection.close()
    logical_manifest: dict[str, Any] = {
        "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
        "database": {"bytes": database.stat().st_size, "sha256": file_sha256(database)},
        "documentHashes": [value["documentHash"] for value in documents],
        "chunkHashes": [value["chunkHash"] for value in chunks],
        "indexHash": index_hash,
    }
    _atomic_json(
        destination / "manifest.json",
        {**logical_manifest, "manifestHash": canonical_sha256(logical_manifest)},
    )
    KnowledgeIndex(destination).verify()
    return destination / "manifest.json"


class KnowledgeIndex:
    """Read-only verified FTS index; local ingestion paths are never stored or returned."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.database = self.root / "knowledge.sqlite3"
        self.manifest = self.root / "manifest.json"

    def _manifest(self) -> dict[str, Any]:
        if self.manifest.is_symlink() or not self.manifest.is_file():
            raise ValueError("knowledge manifest is missing or linked")
        value = json.loads(self.manifest.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("knowledge manifest must be an object")
        logical = {key: item for key, item in value.items() if key != "manifestHash"}
        if value.get("schemaVersion") != KNOWLEDGE_SCHEMA_VERSION or value.get(
            "manifestHash"
        ) != canonical_sha256(logical):
            raise ValueError("knowledge manifest hash is invalid")
        return value

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database.as_uri() + "?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def verify(self) -> str:
        manifest = self._manifest()
        descriptor = manifest.get("database")
        if (
            not isinstance(descriptor, dict)
            or self.database.is_symlink()
            or not self.database.is_file()
            or self.database.stat().st_size != descriptor.get("bytes")
            or file_sha256(self.database) != descriptor.get("sha256")
        ):
            raise ValueError("knowledge SQLite identity is invalid")
        connection = self._connection()
        try:
            metadata = dict(connection.execute("SELECT key, value FROM metadata").fetchall())
            document_rows = connection.execute(
                "SELECT record_json FROM documents ORDER BY source_label"
            ).fetchall()
            chunk_rows = connection.execute(
                "SELECT record_json FROM chunks ORDER BY source_label, ordinal"
            ).fetchall()
            fts_count = int(connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
        finally:
            connection.close()
        if metadata.get("schemaVersion") != KNOWLEDGE_SCHEMA_VERSION:
            raise ValueError("knowledge SQLite schema is invalid")
        documents = [json.loads(row["record_json"]) for row in document_rows]
        chunks = [json.loads(row["record_json"]) for row in chunk_rows]
        if any(not isinstance(item, dict) for item in (*documents, *chunks)):
            raise TypeError("knowledge record must be an object")
        for document in documents:
            logical = {key: value for key, value in document.items() if key != "documentHash"}
            if document.get("documentHash") != canonical_sha256(logical):
                raise ValueError("knowledge document hash is invalid")
        for chunk in chunks:
            logical = {key: value for key, value in chunk.items() if key != "chunkHash"}
            if chunk.get("chunkHash") != canonical_sha256(logical):
                raise ValueError("knowledge chunk hash is invalid")
        document_hashes = [value["documentHash"] for value in documents]
        chunk_hashes = [value["chunkHash"] for value in chunks]
        index_hash = canonical_sha256(
            {
                "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                "documentHashes": document_hashes,
                "chunkHashes": chunk_hashes,
            }
        )
        if (
            fts_count != len(chunks)
            or manifest.get("documentHashes") != document_hashes
            or manifest.get("chunkHashes") != chunk_hashes
            or manifest.get("indexHash") != index_hash
            or metadata.get("indexHash") != index_hash
        ):
            raise ValueError("knowledge logical index identity is invalid")
        return index_hash

    def search(self, query: str, *, limit: int = 10) -> tuple[KnowledgeResult, ...]:
        index_hash = self.verify()
        del index_hash
        if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_CHARS:
            raise ValueError("knowledge query is empty or oversized")
        if not 1 <= limit <= 50:
            raise ValueError("knowledge result limit must be between 1 and 50")
        expression = _search_expression(query)
        if expression is None:
            return ()
        connection = self._connection()
        try:
            rows = connection.execute(
                "SELECT c.record_json, bm25(chunks_fts) AS relevance "
                "FROM chunks_fts JOIN chunks c ON c.chunk_hash = chunks_fts.chunk_hash "
                "WHERE chunks_fts MATCH ? "
                "ORDER BY relevance ASC, c.chunk_hash ASC LIMIT ?",
                (expression, limit),
            ).fetchall()
        finally:
            connection.close()
        results: list[KnowledgeResult] = []
        for rank, row in enumerate(rows, start=1):
            value = json.loads(row["record_json"])
            results.append(
                KnowledgeResult(
                    source_label=str(value["sourceLabel"]),
                    source_uri=str(value["sourceUri"]),
                    content_hash=str(value["contentHash"]),
                    chunk_hash=str(value["chunkHash"]),
                    text=str(value["text"]),
                    rank=rank,
                )
            )
        return tuple(results)


def default_source_uri(label: str) -> str:
    return "local-label://" + quote(label, safe="")


__all__ = [
    "KNOWLEDGE_SCHEMA_VERSION",
    "KnowledgeIndex",
    "KnowledgeResult",
    "KnowledgeSource",
    "build_knowledge_index",
    "default_source_uri",
]
