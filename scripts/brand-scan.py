from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

LONG_TOKENS = tuple(
    value.casefold()
    for value in (
        "io" + "hunter",
        "info" + "opsgfm",
        "py" + "gfm",
        "generic" + "agent",
        "md" + "gfm",
        "static" + "-v2",
        "static" + "_v2",
        "research" + "-v1",
        "research" + "_v1",
        "ioh" + "2",
        "bridge" + "top2router",
        "bridge" + "-inspired",
    )
)
SHORT_PATTERN = re.compile(r"(?<![a-z0-9])r" + r"q[123](?![a-z0-9])", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:rq|RQ)[123](?=[A-Z_]|$)")
TEXT_SUFFIXES = {
    ".css",
    ".csv",
    ".html",
    ".js",
    ".json",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
ARCHIVE_SUFFIXES = {".zip", ".npz"}
RAW_BINARY_SUFFIXES = {".pt", ".sqlite", ".sqlite3"}
CONTENT_EXCEPTIONS = {
    "THIRD_PARTY_NOTICES.md",
}
LEGACY_SKILL_NAME = "run_" + "io" + "hunter"
LEGACY_SKILL_MIGRATION_PATH = "skills/README.md"
LEGACY_SKILL_MIGRATION_NOTE = (
    "Migration note: the private predecessor capability formerly named "
    f"`{LEGACY_SKILL_NAME}` maps to the sole public canonical name "
    "`run_governance_analysis`; no compatibility alias is exposed."
).encode()
MODEL_CARD = "bundles/models/socialgraph-global/exports/socialgraph-global/model-card.json"
KNOWLEDGE_DATABASE = "bundles/governance/knowledge/knowledge.sqlite3"
KNOWLEDGE_SCHEMA_VERSION = "socialgraph-fm.governance-knowledge-index/1.0"
KNOWLEDGE_SOURCES = {
    "model-card": (MODEL_CARD, "model://socialgraph-global/card"),
    "project-readme": ("README.md", "repo://README.md"),
    "technical-reference": ("docs/REFERENCE.md", "repo://docs/REFERENCE.md"),
}
KNOWLEDGE_CHUNK_CHAR_LIMIT = 1_200
MAX_ARCHIVE_DEPTH = 3
MAX_MEMBER_BYTES = 64 * 1024 * 1024


def _matches(value: str) -> list[str]:
    folded = value.casefold()
    found = [token for token in LONG_TOKENS if token in folded]
    if SHORT_PATTERN.search(value):
        found.append("research-question protocol label")
    if IDENTIFIER_PATTERN.search(value):
        found.append("research-question identifier")
    return found


def _portable_member(value: str) -> str:
    if not value or "\\" in value or value.startswith("/") or ":" in value:
        raise ValueError(f"unsafe archive member path: {value!r}")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe archive member path: {value!r}")
    return value


def _scan_bytes(label: str, value: bytes, *, short_tokens: bool) -> list[str]:
    folded = value.lower()
    findings = [token for token in LONG_TOKENS if token.encode("ascii") in folded]
    if short_tokens:
        text = value.decode("utf-8", "ignore")
        if SHORT_PATTERN.search(text):
            findings.append("research-question protocol label")
        if IDENTIFIER_PATTERN.search(text):
            findings.append("research-question identifier")
    return [f"{label}: {item}" for item in findings]


def _scan_archive(label: str, value: bytes, *, depth: int) -> list[str]:
    if depth > MAX_ARCHIVE_DEPTH:
        raise ValueError(f"archive nesting exceeds {MAX_ARCHIVE_DEPTH}: {label}")
    findings: list[str] = []
    from io import BytesIO

    with zipfile.ZipFile(BytesIO(value), "r") as archive:
        for member in archive.infolist():
            name = _portable_member(member.filename)
            findings.extend(f"{label} -> {name}: {item}" for item in _matches(name))
            if member.is_dir():
                continue
            if member.file_size > MAX_MEMBER_BYTES:
                continue
            content = archive.read(member)
            suffix = PurePosixPath(name).suffix.casefold()
            if suffix in ARCHIVE_SUFFIXES and zipfile.is_zipfile(BytesIO(content)):
                findings.extend(_scan_archive(f"{label} -> {name}", content, depth=depth + 1))
            elif suffix in TEXT_SUFFIXES:
                findings.extend(_scan_bytes(f"{label} -> {name}", content, short_tokens=True))
            elif suffix in RAW_BINARY_SUFFIXES:
                findings.extend(_scan_bytes(f"{label} -> {name}", content, short_tokens=False))
    return findings


def _tracked(root: Path) -> list[str]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return sorted(
            value.decode("utf-8")
            for value in completed.stdout.split(b"\0")
            if value
        )
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(root).parts
    )


def _scan_model_card(label: str, value: bytes) -> list[str]:
    try:
        candidate = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return _scan_bytes(label, value, short_tokens=True)
    if not isinstance(candidate, dict):
        return _scan_bytes(label, value, short_tokens=True)
    public_content = dict(candidate)
    public_content.pop("licenses", None)
    public_content.pop("sourceAttribution", None)
    return _scan_bytes(
        label,
        json.dumps(public_content, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        short_tokens=True,
    )


def _without_approved_skill_migration(label: str, value: bytes) -> bytes:
    """Remove exactly one reviewed legacy-name note before generic brand checks."""

    if (
        label == LEGACY_SKILL_MIGRATION_PATH
        and value.count(LEGACY_SKILL_MIGRATION_NOTE) == 1
    ):
        return value.replace(LEGACY_SKILL_MIGRATION_NOTE, b"", 1)
    return value


def _normalized_knowledge_source(path: Path) -> tuple[bytes, str]:
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig")
    if path.suffix.casefold() == ".json":
        text = json.dumps(
            json.loads(text), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    normalized = "\n".join(
        line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()
    return raw, normalized


def _knowledge_chunks(text: str) -> tuple[str, ...]:
    paragraphs = [value.strip() for value in re.split(r"\n\s*\n", text) if value.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= KNOWLEDGE_CHUNK_CHAR_LIMIT:
            pieces.append(paragraph)
            continue
        pieces.extend(
            paragraph[start : start + KNOWLEDGE_CHUNK_CHAR_LIMIT]
            for start in range(0, len(paragraph), KNOWLEDGE_CHUNK_CHAR_LIMIT)
        )
    packed: list[str] = []
    current = ""
    for piece in pieces:
        candidate = piece if not current else current + "\n\n" + piece
        if len(candidate) <= KNOWLEDGE_CHUNK_CHAR_LIMIT:
            current = candidate
        else:
            if current:
                packed.append(current)
            current = piece
    if current:
        packed.append(current)
    return tuple(packed)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _scan_knowledge_database(root: Path, database: Path) -> list[str]:
    """Verify that the embedded index is an exact projection of tracked sources."""

    findings: list[str] = []
    connection: sqlite3.Connection | None = None
    try:
        expected_documents: dict[str, dict[str, object]] = {}
        expected_chunks: list[dict[str, object]] = []
        source_bytes: dict[str, bytes] = {}
        for label in sorted(KNOWLEDGE_SOURCES):
            relative, uri = KNOWLEDGE_SOURCES[label]
            source = root.joinpath(*PurePosixPath(relative).parts)
            if not source.is_file():
                return [f"{KNOWLEDGE_DATABASE}: missing source {relative}"]
            raw, normalized = _normalized_knowledge_source(source)
            source_bytes[label] = raw
            content_hash = hashlib.sha256(raw).hexdigest()
            normalized_hash = hashlib.sha256(normalized.encode()).hexdigest()
            chunk_hashes: list[str] = []
            for ordinal, text in enumerate(_knowledge_chunks(normalized)):
                logical_chunk: dict[str, object] = {
                    "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                    "sourceLabel": label,
                    "sourceUri": uri,
                    "contentHash": content_hash,
                    "ordinal": ordinal,
                    "text": text,
                }
                chunk_hash = _canonical_sha256(logical_chunk)
                chunk_hashes.append(chunk_hash)
                expected_chunks.append({**logical_chunk, "chunkHash": chunk_hash})
            logical_document: dict[str, object] = {
                "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                "sourceLabel": label,
                "sourceUri": uri,
                "contentHash": content_hash,
                "normalizedContentHash": normalized_hash,
                "bytes": len(raw),
                "chunkHashes": chunk_hashes,
            }
            expected_documents[label] = {
                **logical_document,
                "documentHash": _canonical_sha256(logical_document),
            }
        expected_index_hash = _canonical_sha256(
            {
                "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
                "documentHashes": [
                    expected_documents[label]["documentHash"]
                    for label in sorted(expected_documents)
                ],
                "chunkHashes": [chunk["chunkHash"] for chunk in expected_chunks],
            }
        )

        connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        schema_inventory = {
            (str(kind), str(name), str(table), None if sql is None else str(sql))
            for kind, name, table, sql in connection.execute(
                "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
            )
        }
        expected_schema_inventory = {
            ("index", "sqlite_autoindex_chunks_1", "chunks", None),
            ("index", "sqlite_autoindex_documents_1", "documents", None),
            ("index", "sqlite_autoindex_metadata_1", "metadata", None),
            (
                "table",
                "chunks",
                "chunks",
                (
                    "CREATE TABLE chunks (chunk_hash TEXT PRIMARY KEY, "
                    "source_label TEXT NOT NULL, ordinal INTEGER NOT NULL, "
                    "record_json TEXT NOT NULL)"
                ),
            ),
            (
                "table",
                "chunks_fts",
                "chunks_fts",
                (
                    "CREATE VIRTUAL TABLE chunks_fts USING "
                    "fts5(chunk_hash UNINDEXED, source_label UNINDEXED, text)"
                ),
            ),
            (
                "table",
                "chunks_fts_config",
                "chunks_fts_config",
                "CREATE TABLE 'chunks_fts_config'(k PRIMARY KEY, v) WITHOUT ROWID",
            ),
            (
                "table",
                "chunks_fts_content",
                "chunks_fts_content",
                "CREATE TABLE 'chunks_fts_content'(id INTEGER PRIMARY KEY, c0, c1, c2)",
            ),
            (
                "table",
                "chunks_fts_data",
                "chunks_fts_data",
                "CREATE TABLE 'chunks_fts_data'(id INTEGER PRIMARY KEY, block BLOB)",
            ),
            (
                "table",
                "chunks_fts_docsize",
                "chunks_fts_docsize",
                "CREATE TABLE 'chunks_fts_docsize'(id INTEGER PRIMARY KEY, sz BLOB)",
            ),
            (
                "table",
                "chunks_fts_idx",
                "chunks_fts_idx",
                (
                    "CREATE TABLE 'chunks_fts_idx'"
                    "(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID"
                ),
            ),
            (
                "table",
                "documents",
                "documents",
                (
                    "CREATE TABLE documents "
                    "(source_label TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
                ),
            ),
            (
                "table",
                "metadata",
                "metadata",
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
            ),
        }
        if schema_inventory != expected_schema_inventory:
            findings.append(f"{KNOWLEDGE_DATABASE}: schema inventory is not canonical")

        metadata = {
            str(key): str(value)
            for key, value in connection.execute("SELECT key, value FROM metadata")
        }
        if metadata != {
            "schemaVersion": KNOWLEDGE_SCHEMA_VERSION,
            "indexHash": expected_index_hash,
        }:
            findings.append(f"{KNOWLEDGE_DATABASE}: metadata is not canonical")

        document_rows = connection.execute(
            "SELECT source_label, record_json FROM documents ORDER BY source_label"
        ).fetchall()
        chunk_rows = connection.execute(
            "SELECT chunk_hash, source_label, ordinal, record_json FROM chunks "
            "ORDER BY source_label, ordinal"
        ).fetchall()
        expected_document_rows = [
            (label, _canonical_json(expected_documents[label]))
            for label in sorted(expected_documents)
        ]
        if document_rows != expected_document_rows:
            findings.append(f"{KNOWLEDGE_DATABASE}: document records are not canonical")

        expected_chunk_rows = [
            (
                str(chunk["chunkHash"]),
                str(chunk["sourceLabel"]),
                int(chunk["ordinal"]),
                _canonical_json(chunk),
            )
            for chunk in expected_chunks
        ]
        if chunk_rows != expected_chunk_rows:
            findings.append(f"{KNOWLEDGE_DATABASE}: chunk records are not canonical")

        expected_fts_rows = [
            (
                index,
                str(chunk["chunkHash"]),
                str(chunk["sourceLabel"]),
                str(chunk["text"]),
            )
            for index, chunk in enumerate(expected_chunks, start=1)
        ]
        fts_rows = connection.execute(
            "SELECT rowid, chunk_hash, source_label, text FROM chunks_fts ORDER BY rowid"
        ).fetchall()
        shadow_content_rows = connection.execute(
            "SELECT id, c0, c1, c2 FROM chunks_fts_content ORDER BY id"
        ).fetchall()
        if fts_rows != expected_fts_rows or shadow_content_rows != expected_fts_rows:
            findings.append(f"{KNOWLEDGE_DATABASE}: FTS projection is not canonical")
        if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
            findings.append(f"{KNOWLEDGE_DATABASE}: SQLite integrity check failed")
        fts_copy = sqlite3.connect(":memory:")
        try:
            connection.backup(fts_copy)
            fts_copy.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('integrity-check')")
        except sqlite3.DatabaseError:
            findings.append(f"{KNOWLEDGE_DATABASE}: FTS integrity check failed")
        finally:
            fts_copy.close()

        for label, (relative, _uri) in KNOWLEDGE_SOURCES.items():
            raw = source_bytes[label]
            if label == "model-card":
                findings.extend(_scan_model_card(relative, raw))
            else:
                content = _without_approved_skill_migration(relative, raw)
                findings.extend(_scan_bytes(relative, content, short_tokens=True))
    except (OSError, sqlite3.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        findings.append(f"{KNOWLEDGE_DATABASE}: cannot verify source projection ({error})")
    finally:
        if connection is not None:
            connection.close()
    return findings


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    for relative in _tracked(root):
        findings.extend(f"path {relative}: {item}" for item in _matches(relative))
        if relative in CONTENT_EXCEPTIONS:
            continue
        path = root.joinpath(*PurePosixPath(relative).parts)
        if not path.is_file():
            findings.append(f"missing tracked file: {relative}")
            continue
        suffix = path.suffix.casefold()
        if relative == MODEL_CARD:
            findings.extend(_scan_model_card(relative, path.read_bytes()))
        elif relative == KNOWLEDGE_DATABASE:
            findings.extend(_scan_knowledge_database(root, path))
        elif suffix in ARCHIVE_SUFFIXES:
            findings.extend(_scan_archive(relative, path.read_bytes(), depth=0))
        elif suffix in TEXT_SUFFIXES:
            content = _without_approved_skill_migration(relative, path.read_bytes())
            findings.extend(_scan_bytes(relative, content, short_tokens=True))
        elif suffix in RAW_BINARY_SUFFIXES:
            findings.extend(_scan_bytes(relative, path.read_bytes(), short_tokens=False))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    arguments = parser.parse_args()
    root = arguments.repository_root.resolve(strict=True)
    findings = scan(root)
    if findings:
        print("Brand scan rejected the publication candidate:")
        for finding in findings[:100]:
            print(f"- {finding}")
        if len(findings) > 100:
            print(f"- ... {len(findings) - 100} additional findings")
        return 1
    print(f"Brand scan passed for {len(_tracked(root))} publication candidate files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
