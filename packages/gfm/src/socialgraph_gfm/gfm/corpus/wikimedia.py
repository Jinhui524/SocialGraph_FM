"""Privacy-preserving Wikipedia Talk corpus sampler for 2011--2015."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import os
import sqlite3
import shutil
import tarfile
import tempfile
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ...canonical import file_sha256
from ...errors import ContractViolation
from ...runtime import RuntimeLayout
from .common import (
    NumericShardWriter,
    ShardRecord,
    atomic_write_json,
    atomic_write_jsonl,
    build_manifest,
    load_npz_safe,
    read_json_object,
    read_jsonl,
    resolve_within,
    verify_manifest,
)

CORPUS_ID = "wikimedia-talk-article-2011-2015"
DOMAIN_ID = CORPUS_ID
ARTICLE_ID = 4_264_973
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
LICENSE_ID = "CC0-1.0"
LICENSE_ACCEPT_VALUES = frozenset({"CC0", "CC0-1.0"})
LICENSE_URL = "https://meta.wikimedia.org/wiki/Research:Detox/Data_Release"
MAX_COMMENTS = 1_500_000
EVENT_ROWS_PER_SHARD = 50_000
SQLITE_BATCH_ROWS = 2_048
YEARS = (2011, 2012, 2013, 2014, 2015)
TRAIN_END_INCLUSIVE = int(
    datetime(2013, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()
)
VALIDATION_END_INCLUSIVE = int(
    datetime(2014, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()
)
SPLIT_NAMES = ("train", "validation", "test")
ACCESS_ROLES = ("train", "validation", "test", "shadow")
ACCESS_ROLE_TAGS = {"train": "tr", "validation": "va", "test": "te", "shadow": "sh"}
PHYSICAL_ACCESS_SCHEMA = "gfm.physical-role-views/1.0"
EVENT_DIGEST_DTYPE = np.dtype(
    [
        ("src", "<i8"),
        ("dst", "<i8"),
        ("timestamp", "<i8"),
        ("relation", "<i2"),
        ("revision_pseudonym", "<u8"),
        ("split", "i1"),
    ]
)
EXPECTED_FILES: dict[int, dict[str, Any]] = {
    2011: {"id": 7_383_256, "size": 485_250_027, "md5": "cf7b92785303fe4f6da9b1ef7c5cb2c2"},
    2012: {"id": 7_383_259, "size": 412_907_264, "md5": "8f6d0c66b66120d7fd2e877d542a3bc5"},
    2013: {"id": 7_383_262, "size": 365_379_391, "md5": "fa0fe445e89db6c2cb5fef69527c249f"},
    2014: {"id": 7_383_271, "size": 345_430_723, "md5": "2c119e3da9f29235c8a1c10f1c446063"},
    2015: {"id": 7_383_289, "size": 338_140_119, "md5": "b41660f044970fb8aad6d80350d5d448"},
}
EXPECTED_COLUMNS = frozenset(
    {"rev_id", "comment", "raw_comment", "timestamp", "page_id", "page_title", "user_id", "user_text", "bot", "admin"}
)
MetadataClient = Callable[[], Mapping[str, Any]]
Download = Callable[[str, Path], None]


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"Wikimedia Talk: {message}")


def _update_event_digest(
    digest: Any, arrays: Mapping[str, np.ndarray], mask: np.ndarray | None = None
) -> None:
    rows = arrays["src"].shape[0] if mask is None else int(np.count_nonzero(mask))
    structured = np.empty(rows, dtype=EVENT_DIGEST_DTYPE)
    for name in EVENT_DIGEST_DTYPE.names or ():
        structured[name] = arrays[name] if mask is None else arrays[name][mask]
    digest.update(structured.tobytes(order="C"))


def _json_url(url: str) -> Mapping[str, Any]:
    import json

    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "SocialGraph-FM/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as stream:  # noqa: S310
            value = json.load(stream)
    except (OSError, ValueError) as exc:
        raise _fail("Figshare metadata request failed") from exc
    if not isinstance(value, dict):
        raise _fail("Figshare metadata is not an object")
    return value


def _download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "SocialGraph-FM/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as source, path.open("xb") as target:  # noqa: S310
            shutil.copyfileobj(source, target, 4 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
    except OSError as exc:
        path.unlink(missing_ok=True)
        raise _fail("Figshare file download failed") from exc


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)  # noqa: S324 - official Figshare checksum
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_wikimedia(
    root: str | Path,
    *,
    accept_license: str,
    years: Sequence[int] = YEARS,
    metadata_client: MetadataClient | None = None,
    downloader: Download | None = None,
    enforce_fixed_metadata: bool = True,
) -> dict[str, Any]:
    """Fetch fixed Figshare files after validating article metadata and MD5."""

    if accept_license not in LICENSE_ACCEPT_VALUES:
        raise _fail("accept_license must be CC0 (canonical identifier CC0-1.0)")
    selected_years = tuple(years)
    if not selected_years or len(set(selected_years)) != len(selected_years) or not set(selected_years).issubset(YEARS):
        raise _fail("years must be a unique subset of 2011--2015")
    formal_source = (
        metadata_client is None and downloader is None and enforce_fixed_metadata
    )
    metadata = (metadata_client or (lambda: _json_url(API_URL)))()
    license_value = metadata.get("license")
    license_name = license_value.get("name") if isinstance(license_value, dict) else None
    if metadata.get("id") != ARTICLE_ID or license_name != "CC0":
        raise _fail("Figshare article identity or license changed")
    files = metadata.get("files")
    if not isinstance(files, list):
        raise _fail("Figshare metadata lacks files")
    by_id = {item.get("id"): item for item in files if isinstance(item, dict)}
    layout = RuntimeLayout.from_root(root)
    raw = layout.raw_wikimedia_talk
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for year in selected_years:
        expected = EXPECTED_FILES[year]
        item = by_id.get(expected["id"])
        if not isinstance(item, dict):
            raise _fail(f"Figshare metadata lacks fixed article-talk file for {year}")
        expected_name = f"comments_article_{year}.tar.gz"
        if enforce_fixed_metadata and (
            item.get("name") != expected_name
            or item.get("size") != expected["size"]
            or item.get("computed_md5") != expected["md5"]
        ):
            raise _fail(f"Figshare metadata drifted for {year}")
        url = item.get("download_url")
        if not isinstance(url, str) or url != f"https://ndownloader.figshare.com/files/{expected['id']}":
            raise _fail(f"Figshare download URL is not the fixed official endpoint for {year}")
        destination = raw / expected_name
        if not destination.exists():
            temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
            try:
                (downloader or _download)(url, temporary)
                if temporary.is_symlink() or not temporary.is_file():
                    raise _fail("downloaded source is not a regular file")
                temporary_digest = _file_md5(temporary)
                if enforce_fixed_metadata and (
                    temporary.stat().st_size != expected["size"]
                    or temporary_digest != expected["md5"]
                ):
                    raise _fail(
                        f"downloaded source checksum/size mismatch for {year}"
                    )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        if destination.is_symlink() or not destination.is_file():
            raise _fail("downloaded source is not a regular file")
        digest = _file_md5(destination)
        if enforce_fixed_metadata and (destination.stat().st_size != expected["size"] or digest != expected["md5"]):
            raise _fail(f"downloaded source checksum/size mismatch for {year}")
        records.append(
            {
                "year": year,
                "fileId": expected["id"],
                "name": destination.name,
                "size": destination.stat().st_size,
                "figshareMd5": digest,
                "sha256": file_sha256(destination),
                "downloadUrl": url,
            }
        )
    receipt = {
        "schemaVersion": "gfm.wikimedia-fetch/1.0",
        "articleId": ARTICLE_ID,
        "articleApi": API_URL,
        "licenseId": LICENSE_ID,
        "licenseEvidence": LICENSE_URL,
        "formalEligible": formal_source,
        "files": records,
    }
    atomic_write_json(raw / "fetch-receipt.json", receipt)
    return receipt


def _is_true(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "t"}


def _parse_user_id(value: str | None) -> int:
    """Parse Detox's nullable, float-rendered integral user identifier.

    The official article-talk TSV stores this nullable column using an empty
    string for anonymous/IP edits and ``<digits>.0`` for registered users.  It
    must not pass through ``float`` because large identifiers could lose
    precision.  No other decimal or exponent spelling is accepted.
    """

    raw = (value or "").strip()
    if not raw:
        return 0
    integer = raw[:-2] if raw.endswith(".0") else raw
    if not integer.isascii() or not integer.isdigit():
        raise _fail("source TSV contains an invalid user ID")
    return int(integer)


def _rows_from_tar(path: Path) -> Iterator[dict[str, str]]:
    try:
        archive = tarfile.open(path, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        raise _fail(f"cannot read source archive {path.name}") from exc
    with archive:
        members = archive.getmembers()
        if not members or len(members) > 10_000:
            raise _fail("source TAR entry inventory is empty or excessive")
        for member in members:
            name = member.name
            if member.issym() or member.islnk() or name.startswith("/") or "\\" in name or ".." in Path(name).parts:
                raise _fail("source TAR contains an unsafe member")
            if not member.isfile() or not name.endswith((".tsv", ".tsv.gz")):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise _fail("source TAR member cannot be read")
            binary: Any = gzip.GzipFile(fileobj=extracted) if name.endswith(".gz") else extracted
            with io.TextIOWrapper(binary, encoding="utf-8", errors="strict", newline="") as text:
                reader = csv.DictReader(text, delimiter="\t")
                if reader.fieldnames is None or not EXPECTED_COLUMNS.issubset(reader.fieldnames):
                    raise _fail("source TSV schema changed")
                for row in reader:
                    yield {key: row.get(key, "") for key in EXPECTED_COLUMNS}


def _valid_row(row: Mapping[str, str], year: int) -> tuple[int, int, int, str] | None:
    user_id = _parse_user_id(row.get("user_id"))
    try:
        page_id = int(row["page_id"])
        rev_id = int(row["rev_id"])
        timestamp = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise _fail("source TSV contains an invalid numeric/time field") from exc
    if timestamp.tzinfo is None or timestamp.astimezone(UTC).year != year:
        raise _fail("source TSV timestamp does not match its fixed year archive")
    if user_id <= 0 or _is_true(row["bot"]):
        return None
    text = row["comment"].replace("NEWLINE_TOKEN", "\n").replace("TAB_TOKEN", "\t").strip()
    if not text:
        return None
    return rev_id, page_id, user_id, text


def _pseudonymous_id(kind: str, source_id: int, salt: str) -> np.uint64:
    digest = hashlib.sha256(f"{salt}\0{kind}\0{source_id}".encode()).digest()
    return np.uint64(int.from_bytes(digest[:8], "little", signed=False))


def _opaque_source_key(kind: str, source_id: int, salt: str) -> str:
    """Return a full-width, temporary join key without persisting a source ID."""

    return hashlib.sha256(f"{salt}\0{kind}\0{source_id}".encode()).hexdigest()


def _configure_spool(connection: sqlite3.Connection) -> None:
    """Keep SQLite's working set bounded and all large ordering work on disk."""

    connection.execute("PRAGMA trusted_schema=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")


def _flush_page_counts(
    connection: sqlite3.Connection, rows: list[tuple[str, str, str, int]]
) -> None:
    if not rows:
        return
    connection.executemany(
        """
        INSERT INTO page_counts(
            page_key, score, tie_key, event_count, first_timestamp, last_timestamp
        )
        VALUES (?, ?, ?, 1, ?, ?)
        ON CONFLICT(page_key) DO UPDATE SET
            event_count=event_count + 1,
            first_timestamp=MIN(first_timestamp, excluded.first_timestamp),
            last_timestamp=MAX(last_timestamp, excluded.last_timestamp)
        """,
        tuple((*row, row[3]) for row in rows),
    )
    connection.commit()
    rows.clear()


def _split_for_last_timestamp(timestamp: int) -> int:
    """Assign a complete page history from the page's final eligible event."""

    if timestamp <= TRAIN_END_INCLUSIVE:
        return 0
    if timestamp <= VALIDATION_END_INCLUSIVE:
        return 1
    return 2


def _flush_selected_events(
    connection: sqlite3.Connection,
    rows: list[tuple[int, str, str, str, str, str, str]],
) -> None:
    if not rows:
        return
    try:
        connection.executemany(
            """
            INSERT INTO events(
                timestamp, revision_key, page_key, user_key, pseudonym, text
            )
            SELECT ?, ?, ?, ?, ?, ?
            WHERE EXISTS (
                SELECT 1 FROM page_counts
                WHERE page_key=? AND selected=1
            )
            """,
            rows,
        )
        connection.commit()
    except sqlite3.IntegrityError as exc:
        raise _fail(
            "selected source contains a duplicate revision or pseudonym collision"
        ) from exc
    finally:
        rows.clear()


def _populate_local_map(
    connection: sqlite3.Connection, *, table: str, column: str
) -> int:
    if table not in {"user_map", "page_map"} or column not in {"user_key", "page_key"}:
        raise AssertionError("unsafe internal SQLite identifier")
    connection.execute(
        f"CREATE TABLE {table}(source_key TEXT PRIMARY KEY, local_id INTEGER UNIQUE NOT NULL) WITHOUT ROWID"
    )
    batch: list[tuple[str, int]] = []
    count = 0
    for (source_key,) in connection.execute(
        f"SELECT DISTINCT {column} FROM events ORDER BY {column}"
    ):
        batch.append((str(source_key), count))
        count += 1
        if len(batch) == SQLITE_BATCH_ROWS:
            connection.executemany(
                f"INSERT INTO {table}(source_key, local_id) VALUES (?, ?)", batch
            )
            batch.clear()
    if batch:
        connection.executemany(
            f"INSERT INTO {table}(source_key, local_id) VALUES (?, ?)", batch
        )
    connection.commit()
    return count


def _node_shards(
    directory: Path, *, user_count: int, page_count: int, rows_per_shard: int
) -> list[ShardRecord]:
    writer = NumericShardWriter(
        directory, prefix="nodes", rows_per_shard=rows_per_shard
    )
    node_count = user_count + page_count
    records: list[ShardRecord] = []
    for start in range(0, node_count, rows_per_shard):
        stop = min(start + rows_per_shard, node_count)
        positions = np.arange(start, stop, dtype=np.int64)
        records.append(
            writer.write(
                {"node_type": (positions >= user_count).astype(np.int16)}
            )
        )
    return records


def prepare_wikimedia(
    raw_files: Sequence[str | Path],
    root: str | Path,
    *,
    max_comments: int = MAX_COMMENTS,
    row_source: Callable[[Path], Iterable[dict[str, str]]] | None = None,
    event_rows_per_shard: int = EVENT_ROWS_PER_SHARD,
) -> dict[str, Any]:
    """Create a bounded-memory, whole-page sample in an atomic directory.

    Direct source identifiers are converted to salted opaque join keys before a
    private, transient SQLite spool is written.  Selected cleaned text streams
    from that spool into JSONL; the spool is removed before staged verification.
    """

    if max_comments < 1 or max_comments > MAX_COMMENTS:
        raise _fail(f"max_comments must be in 1..{MAX_COMMENTS}")
    if event_rows_per_shard < 1 or event_rows_per_shard > EVENT_ROWS_PER_SHARD:
        raise _fail(
            f"event_rows_per_shard must be in 1..{EVENT_ROWS_PER_SHARD}"
        )
    layout = RuntimeLayout.from_root(root)
    raw = layout.raw_wikimedia_talk.resolve()
    receipt = read_json_object(raw / "fetch-receipt.json")
    if (
        receipt.get("schemaVersion") != "gfm.wikimedia-fetch/1.0"
        or receipt.get("articleId") != ARTICLE_ID
        or receipt.get("licenseId") != LICENSE_ID
    ):
        raise _fail("fetch receipt identity or license is invalid")
    received = {str(item["name"]): item for item in receipt.get("files", []) if isinstance(item, dict)}
    files: list[tuple[int, Path]] = []
    for source_path in raw_files:
        path = Path(source_path).expanduser().resolve(strict=True)
        try:
            path.relative_to(raw)
        except ValueError as exc:
            raise _fail("source file is outside the accepted raw directory") from exc
        item = received.get(path.name)
        if not isinstance(item, dict) or file_sha256(path) != item.get("sha256"):
            raise _fail("source file does not match the accepted fetch receipt")
        year = int(item["year"])
        files.append((year, path))
    if not files:
        raise _fail("no accepted source files were provided")

    output = layout.processed_gfm / CORPUS_ID
    if output.exists():
        if output.is_symlink() or not output.is_dir() or not (output / "manifest.json").is_file():
            raise _fail("processed corpus path exists without a complete manifest")
        return check_wikimedia(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    salt = os.environ.get("SOCIALGRAPH_GFM_PSEUDONYM_SALT", "").strip()
    if not salt:
        raise _fail("SOCIALGRAPH_GFM_PSEUDONYM_SALT is required and is never persisted")

    source = row_source or _rows_from_tar
    staging = output.with_name(f".{CORPUS_ID}.{uuid.uuid4().hex}.staging")
    staging.mkdir(mode=0o700)
    spool_path = staging / "spool.sqlite3"
    connection: sqlite3.Connection | None = None
    excluded_anonymous = 0
    excluded_bot = 0
    try:
        connection = sqlite3.connect(spool_path)
        _configure_spool(connection)
        connection.executescript(
            """
            CREATE TABLE page_counts(
                page_key TEXT PRIMARY KEY,
                score TEXT NOT NULL,
                tie_key TEXT NOT NULL,
                event_count INTEGER NOT NULL CHECK(event_count > 0),
                first_timestamp INTEGER NOT NULL,
                last_timestamp INTEGER NOT NULL,
                split INTEGER CHECK(split IN (0, 1, 2)),
                selected INTEGER NOT NULL DEFAULT 0 CHECK(selected IN (0, 1))
            ) WITHOUT ROWID;
            CREATE TABLE events(
                timestamp INTEGER NOT NULL,
                revision_key TEXT PRIMARY KEY,
                page_key TEXT NOT NULL,
                user_key TEXT NOT NULL,
                pseudonym TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )

        # Pass one counts valid events per page.  Only a fixed-size write batch
        # is resident; page cardinality is held by SQLite.
        count_batch: list[tuple[str, str, str, int]] = []
        for year, path in files:
            for row in source(path):
                source_user_id = _parse_user_id(row.get("user_id"))
                if source_user_id <= 0:
                    excluded_anonymous += 1
                    continue
                if _is_true(row.get("bot", "")):
                    excluded_bot += 1
                    continue
                parsed = _valid_row(row, year)
                if parsed is None:
                    continue
                page_id = parsed[1]
                page_key = _opaque_source_key("page", page_id, salt)
                timestamp = int(
                    datetime.fromisoformat(
                        row["timestamp"].replace("Z", "+00:00")
                    ).timestamp()
                )
                count_batch.append(
                    (
                        page_key,
                        hashlib.sha256(
                            f"wikimedia-page-v1\0{page_id}".encode()
                        ).hexdigest(),
                        page_key,
                        timestamp,
                    )
                )
                if len(count_batch) == SQLITE_BATCH_ROWS:
                    _flush_page_counts(connection, count_batch)
        _flush_page_counts(connection, count_batch)

        # A page is an indivisible split unit.  Its last eligible event chooses
        # the cohort: pages ending by 2013 train, ending in 2014 validation,
        # and ending in 2015 test.  Earlier history stays with that same page.
        connection.execute(
            """
            UPDATE page_counts
            SET split=CASE
                WHEN last_timestamp <= ? THEN 0
                WHEN last_timestamp <= ? THEN 1
                ELSE 2
            END
            """,
            (TRAIN_END_INCLUSIVE, VALIDATION_END_INCLUSIVE),
        )

        # Select complete pages in the same stable-hash order as the former
        # in-memory implementation, but never materialize the page inventory.
        connection.execute(
            "CREATE INDEX page_selection_order "
            "ON page_counts(score, tie_key, page_key)"
        )
        selected_total = 0
        selection_batch: list[tuple[str]] = []
        for page_key, event_count in connection.execute(
            "SELECT page_key, event_count FROM page_counts "
            "ORDER BY score, tie_key, page_key"
        ):
            count = int(event_count)
            if selected_total + count > max_comments:
                continue
            selection_batch.append((str(page_key),))
            selected_total += count
            if len(selection_batch) == SQLITE_BATCH_ROWS:
                connection.executemany(
                    "UPDATE page_counts SET selected=1 WHERE page_key=?",
                    selection_batch,
                )
                selection_batch.clear()
        if selection_batch:
            connection.executemany(
                "UPDATE page_counts SET selected=1 WHERE page_key=?",
                selection_batch,
            )
        connection.commit()
        if selected_total < 1:
            raise _fail("whole-page sampler selected no eligible events")

        # Pass two inserts only events whose complete page was selected.  The
        # SELECT predicate lives in SQLite, avoiding an in-memory selected set.
        event_batch: list[tuple[int, str, str, str, str, str, str]] = []
        for year, path in files:
            for row in source(path):
                parsed = _valid_row(row, year)
                if parsed is None:
                    continue
                revision_id, page_id, user_id, text_value = parsed
                page_key = _opaque_source_key("page", page_id, salt)
                timestamp = int(
                    datetime.fromisoformat(
                        row["timestamp"].replace("Z", "+00:00")
                    ).timestamp()
                )
                event_batch.append(
                    (
                        timestamp,
                        _opaque_source_key("revision", revision_id, salt),
                        page_key,
                        _opaque_source_key("user", user_id, salt),
                        str(int(_pseudonymous_id("revision", revision_id, salt))),
                        text_value,
                        page_key,
                    )
                )
                if len(event_batch) == SQLITE_BATCH_ROWS:
                    _flush_selected_events(connection, event_batch)
        _flush_selected_events(connection, event_batch)

        connection.execute("CREATE INDEX events_page_key ON events(page_key)")
        connection.execute("CREATE INDEX events_user_key ON events(user_key)")
        mismatch = connection.execute(
            """
            SELECT p.page_key
            FROM page_counts AS p
            LEFT JOIN events AS e ON e.page_key=p.page_key
            WHERE p.selected=1
            GROUP BY p.page_key, p.event_count
            HAVING COUNT(e.revision_key) != p.event_count
            LIMIT 1
            """
        ).fetchone()
        if mismatch is not None:
            raise _fail("second pass changed a selected page history")
        event_count = int(
            connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        )
        if event_count != selected_total or event_count > max_comments:
            raise _fail("whole-page sampler count changed between source passes")

        event_split_counts = tuple(
            int(
                connection.execute(
                    """
                    SELECT COALESCE(SUM(event_count), 0)
                    FROM page_counts WHERE selected=1 AND split=?
                    """,
                    (split,),
                ).fetchone()[0]
            )
            for split in range(3)
        )
        page_split_counts = tuple(
            int(
                connection.execute(
                    "SELECT COUNT(*) FROM page_counts WHERE selected=1 AND split=?",
                    (split,),
                ).fetchone()[0]
            )
            for split in range(3)
        )
        if sum(event_split_counts) != event_count:
            raise _fail("page split assignment does not cover every selected event")

        connection.execute(
            "CREATE INDEX events_order ON events(timestamp, revision_key)"
        )
        user_count = _populate_local_map(
            connection, table="user_map", column="user_key"
        )
        page_count = _populate_local_map(
            connection, table="page_map", column="page_key"
        )
        if user_count < 1 or page_count < 1:
            raise _fail("selected sample has no registered users or pages")

        writer = NumericShardWriter(
            staging,
            prefix="events",
            rows_per_shard=event_rows_per_shard,
        )
        event_records: list[ShardRecord] = []
        access_writers = {
            role: NumericShardWriter(
                staging,
                prefix=f"rv-e-{ACCESS_ROLE_TAGS[role]}",
                rows_per_shard=event_rows_per_shard,
            )
            for role in ACCESS_ROLES
        }
        access_event_records: dict[str, list[ShardRecord]] = {
            role: [] for role in ACCESS_ROLES
        }
        spool_connection = connection

        def text_rows() -> Iterator[dict[str, Any]]:
            src_values: list[int] = []
            dst_values: list[int] = []
            timestamp_values: list[int] = []
            pseudonym_values: list[int] = []
            split_values: list[int] = []
            role_values: dict[str, dict[str, list[int]]] = {
                role: {
                    "src": [],
                    "dst": [],
                    "timestamp": [],
                    "revision_pseudonym": [],
                    "split": [],
                }
                for role in ACCESS_ROLES
            }

            def flush_role(role: str) -> None:
                values = role_values[role]
                if not values["src"]:
                    return
                access_event_records[role].append(
                    access_writers[role].write(
                        {
                            "src": np.asarray(values["src"], dtype=np.int64),
                            "dst": np.asarray(values["dst"], dtype=np.int64),
                            "timestamp": np.asarray(values["timestamp"], dtype=np.int64),
                            "relation": np.zeros(len(values["src"]), dtype=np.int16),
                            "revision_pseudonym": np.asarray(
                                values["revision_pseudonym"], dtype=np.uint64
                            ),
                            "split": np.asarray(values["split"], dtype=np.int8),
                        }
                    )
                )
                for value in values.values():
                    value.clear()

            def flush_numeric() -> None:
                if not src_values:
                    return
                event_records.append(
                    writer.write(
                        {
                            "src": np.asarray(src_values, dtype=np.int64),
                            "dst": np.asarray(dst_values, dtype=np.int64),
                            "timestamp": np.asarray(
                                timestamp_values, dtype=np.int64
                            ),
                            "relation": np.zeros(
                                len(src_values), dtype=np.int16
                            ),
                            "revision_pseudonym": np.asarray(
                                pseudonym_values, dtype=np.uint64
                            ),
                            "split": np.asarray(split_values, dtype=np.int8),
                        }
                    )
                )
                src_values.clear()
                dst_values.clear()
                timestamp_values.clear()
                pseudonym_values.clear()
                split_values.clear()

            cursor = spool_connection.execute(
                """
                SELECT e.timestamp, u.local_id, p.local_id,
                       e.pseudonym, e.text, c.split
                FROM events AS e
                JOIN user_map AS u ON u.source_key=e.user_key
                JOIN page_map AS p ON p.source_key=e.page_key
                JOIN page_counts AS c ON c.page_key=e.page_key
                ORDER BY e.timestamp, e.revision_key
                """
            )
            for (
                timestamp,
                user_id,
                page_id,
                pseudonym,
                text_value,
                split,
            ) in cursor:
                src_values.append(int(user_id))
                dst_values.append(user_count + int(page_id))
                timestamp_values.append(int(timestamp))
                pseudonym_values.append(int(pseudonym))
                split_values.append(int(split))
                role = SPLIT_NAMES[int(split)]
                role_values[role]["src"].append(int(user_id))
                role_values[role]["dst"].append(user_count + int(page_id))
                role_values[role]["timestamp"].append(int(timestamp))
                role_values[role]["revision_pseudonym"].append(int(pseudonym))
                role_values[role]["split"].append(int(split))
                if len(src_values) == event_rows_per_shard:
                    flush_numeric()
                if len(role_values[role]["src"]) == event_rows_per_shard:
                    flush_role(role)
                yield {
                    "id": str(pseudonym),
                    "text": str(text_value),
                    "timestamp": int(timestamp),
                }
            flush_numeric()
            for role in ACCESS_ROLES:
                flush_role(role)

        written_text_rows = atomic_write_jsonl(staging / "text.jsonl", text_rows())
        if written_text_rows != event_count or sum(
            item.rows for item in event_records
        ) != event_count:
            raise _fail("streamed numeric/text event counts differ from the spool")
        text_record = ShardRecord(
            path="text.jsonl",
            sha256=file_sha256(staging / "text.jsonl"),
            rows=written_text_rows,
            arrays=(),
        )
        node_records = _node_shards(
            staging,
            user_count=user_count,
            page_count=page_count,
            rows_per_shard=event_rows_per_shard,
        )

        # No spool, source join key or plaintext source ID is allowed in the
        # immutable corpus.  Close and unlink it before manifest construction.
        connection.close()
        connection = None
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(f"{spool_path}{suffix}").unlink(missing_ok=True)

        manifest = build_manifest(
            schema_version="gfm.wikimedia-corpus/1.0",
            corpus_id=CORPUS_ID,
            license_id=LICENSE_ID,
            source={
                "articleId": ARTICLE_ID,
                "articleApi": API_URL,
                "licenseEvidence": LICENSE_URL,
                "formalEligible": receipt.get("formalEligible") is True
                and row_source is None,
                "files": receipt["files"],
            },
            shards=tuple(
                event_records
                + node_records
                + [
                    record
                    for role in ACCESS_ROLES
                    for record in access_event_records[role]
                ]
                + [text_record]
            ),
            splits={
                "strategy": "page-last-event-time",
                "assignmentUnit": "article-talk-page",
                "assignmentTimestamp": "last-eligible-event",
                "historyPolicy": "complete-page-history-kept-in-assigned-split",
                "pageDisjoint": True,
                "eventSplitArray": "split",
                "codes": {"0": "train", "1": "validation", "2": "test"},
                "train": "page-last-event-in-2011-2013",
                "validation": "page-last-event-in-2014",
                "test": "page-last-event-in-2015",
                "trainEndInclusive": TRAIN_END_INCLUSIVE,
                "validationEndInclusive": VALIDATION_END_INCLUSIVE,
                "counts": {
                    "events": dict(zip(SPLIT_NAMES, event_split_counts, strict=True)),
                    "pages": dict(zip(SPLIT_NAMES, page_split_counts, strict=True)),
                },
            },
            privacy={
                "anonymousExcluded": excluded_anonymous,
                "botsExcluded": excluded_bot,
                "usernamesPersisted": False,
                "ipAddressesPersisted": False,
                "pageTitlesPersisted": False,
                "sourceUserIdsPersisted": False,
                "sourcePageIdsPersisted": False,
                "sourceRevisionIdsPersisted": False,
                "identifiers": (
                    "contiguous local node indices and salted revision pseudonyms"
                ),
                "pseudonymSaltPersisted": False,
                "publicCheckpointEligible": receipt.get("formalEligible") is True
                and row_source is None,
            },
            extra={
                "domainId": DOMAIN_ID,
                "userCount": user_count,
                "pageCount": page_count,
                "nodeCount": user_count + page_count,
                "nodeOffsets": {"user": 0, "page": user_count},
                "nodeTypes": {
                    "0": "registered-user",
                    "1": "article-talk-page",
                },
                "eventCount": event_count,
                "maximumComments": max_comments,
                "eventShardRows": event_rows_per_shard,
                "numericShardRowLimit": EVENT_ROWS_PER_SHARD,
                "eventOrdering": "timestamp-then-salted-revision-digest",
                "sampling": "two-pass-whole-page-stable-sha256",
                "fullPageHistories": True,
                "physicalAccess": {
                    "schemaVersion": PHYSICAL_ACCESS_SCHEMA,
                    "roles": list(ACCESS_ROLES),
                    "roleFamilies": {
                        "events": {
                            role: [record.path for record in access_event_records[role]]
                            for role in ACCESS_ROLES
                        }
                    },
                    "sharedFamilies": {
                        "nodes": [record.path for record in node_records]
                    },
                    "mergeOrder": {"events": "timestamp-pseudonym"},
                    "historySemantics": (
                        "each validation/test role file contains the complete history "
                        "of pages assigned to that role"
                    ),
                },
            },
        )
        # Manifest is always the final staged artifact.  A full streaming check
        # must succeed before the directory becomes visible at its final name.
        atomic_write_json(staging / "manifest.json", manifest)
        _check_wikimedia_directory(staging)
        os.replace(staging, output)
        return check_wikimedia(root)
    finally:
        if connection is not None:
            connection.close()
        if staging.exists():
            shutil.rmtree(staging)


def _check_wikimedia_directory(output: Path) -> dict[str, Any]:
    """Stream-check a staged or published corpus with bounded working memory."""

    if output.is_symlink() or not output.is_dir():
        raise _fail("processed corpus directory is absent or unsafe")
    manifest = read_json_object(output / "manifest.json")
    if manifest.get("schemaVersion") != "gfm.wikimedia-corpus/1.0":
        raise _fail("processed manifest schema is unsupported")
    if (
        manifest.get("corpusId") != CORPUS_ID
        or manifest.get("domainId") != DOMAIN_ID
        or manifest.get("licenseId") != LICENSE_ID
    ):
        raise _fail("processed manifest identity or license is invalid")
    source = manifest.get("source")
    if (
        not isinstance(source, dict)
        or source.get("articleId") != ARTICLE_ID
        or source.get("articleApi") != API_URL
        or source.get("licenseEvidence") != LICENSE_URL
        or not isinstance(source.get("formalEligible"), bool)
    ):
        raise _fail("processed source/license evidence is invalid")
    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict) or privacy.get(
        "publicCheckpointEligible"
    ) is not source.get("formalEligible"):
        raise _fail("processed formal-eligibility evidence is inconsistent")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or any(
        splits.get(key) != value
        for key, value in {
            "strategy": "page-last-event-time",
            "assignmentUnit": "article-talk-page",
            "assignmentTimestamp": "last-eligible-event",
            "historyPolicy": "complete-page-history-kept-in-assigned-split",
            "pageDisjoint": True,
            "eventSplitArray": "split",
            "codes": {"0": "train", "1": "validation", "2": "test"},
            "train": "page-last-event-in-2011-2013",
            "validation": "page-last-event-in-2014",
            "test": "page-last-event-in-2015",
            "trainEndInclusive": TRAIN_END_INCLUSIVE,
            "validationEndInclusive": VALIDATION_END_INCLUSIVE,
        }.items()
    ):
        raise _fail("processed page-disjoint split contract is invalid")
    if (
        manifest.get("sampling") != "two-pass-whole-page-stable-sha256"
        or manifest.get("fullPageHistories") is not True
        or manifest.get("eventOrdering")
        != "timestamp-then-salted-revision-digest"
    ):
        raise _fail("processed sampling/order contract is invalid")
    verify_manifest(output, manifest)
    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict) or any(
        privacy.get(field) is not False
        for field in (
            "usernamesPersisted",
            "ipAddressesPersisted",
            "pageTitlesPersisted",
            "sourceUserIdsPersisted",
            "sourcePageIdsPersisted",
            "sourceRevisionIdsPersisted",
            "pseudonymSaltPersisted",
        )
    ):
        raise _fail("privacy manifest does not prohibit direct identifiers")

    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise _fail("processed manifest shard inventory is invalid")
    event_records: list[dict[str, Any]] = []
    access_event_records: list[dict[str, Any]] = []
    node_records: list[dict[str, Any]] = []
    text_records: list[dict[str, Any]] = []
    event_names = {
        "src",
        "dst",
        "timestamp",
        "relation",
        "revision_pseudonym",
        "split",
    }
    for item in shards:
        if not isinstance(item, dict):
            raise _fail("processed manifest shard descriptor is invalid")
        arrays = item.get("arrays")
        names = {
            value.get("name")
            for value in arrays
            if isinstance(arrays, list) and isinstance(value, dict)
        } if isinstance(arrays, list) else set()
        if names == event_names:
            if str(item.get("path", "")).startswith("rv-e-"):
                access_event_records.append(item)
            else:
                event_records.append(item)
        elif names == {"node_type"}:
            node_records.append(item)
        elif not names and item.get("path") == "text.jsonl":
            text_records.append(item)
        else:
            raise _fail("processed manifest declares an unknown shard family")
    if not event_records or not node_records or len(text_records) != 1:
        raise _fail("manifest lacks a complete Wikimedia shard family")
    physical_access = manifest.get("physicalAccess")
    if (
        not isinstance(physical_access, dict)
        or physical_access.get("schemaVersion") != PHYSICAL_ACCESS_SCHEMA
        or physical_access.get("roles") != list(ACCESS_ROLES)
        or physical_access.get("sharedFamilies")
        != {"nodes": [str(record["path"]) for record in node_records]}
        or physical_access.get("mergeOrder")
        != {"events": "timestamp-pseudonym"}
    ):
        raise _fail("Wikimedia physical role-view contract is invalid")
    role_family = physical_access.get("roleFamilies")
    if not isinstance(role_family, dict) or set(role_family) != {"events"}:
        raise _fail("Wikimedia physical role-view family is invalid")
    role_paths = role_family["events"]
    if not isinstance(role_paths, dict) or set(role_paths) != set(ACCESS_ROLES):
        raise _fail("Wikimedia physical role paths are invalid")
    expected_access_paths = {
        str(path)
        for role in ACCESS_ROLES
        for path in role_paths[role]
        if isinstance(role_paths[role], list) and isinstance(path, str)
    }
    if expected_access_paths != {
        str(record["path"]) for record in access_event_records
    }:
        raise _fail("Wikimedia physical role paths differ from declared shards")
    access_role_digests = [hashlib.sha256() for _ in SPLIT_NAMES]
    for split_code, role in enumerate(SPLIT_NAMES):
        paths = role_paths.get(role)
        if not isinstance(paths, list):
            raise _fail("Wikimedia physical role path inventory is malformed")
        role_rows = 0
        for index, path in enumerate(paths):
            if path != f"rv-e-{ACCESS_ROLE_TAGS[role]}-{index:05d}.npz":
                raise _fail("Wikimedia physical role shards are not sequential")
            record = next(
                (item for item in access_event_records if item["path"] == path), None
            )
            if record is None:
                raise _fail("Wikimedia physical role shard is absent")
            loaded = load_npz_safe(
                resolve_within(output, path),
                expected={
                    str(item["name"]): (str(item["dtype"]), len(item["shape"]))
                    for item in record["arrays"]
                },
            )
            if bool(np.any(loaded["split"] != split_code)):
                raise _fail("Wikimedia physical role shard contains another split")
            _update_event_digest(access_role_digests[split_code], loaded)
            role_rows += int(record["rows"])
        if role_rows != int(splits["counts"]["events"][role]):
            raise _fail("Wikimedia physical role row count differs from split contract")
    if role_paths.get("shadow") != []:
        raise _fail("Wikimedia 2011-2015 corpus cannot declare shadow events")

    shard_rows = manifest.get("eventShardRows")
    numeric_limit = manifest.get("numericShardRowLimit")
    if (
        not isinstance(shard_rows, int)
        or isinstance(shard_rows, bool)
        or not 1 <= shard_rows <= EVENT_ROWS_PER_SHARD
        or numeric_limit != EVENT_ROWS_PER_SHARD
    ):
        raise _fail("Wikimedia numeric shard limit is invalid")
    for index, record in enumerate(event_records):
        if record.get("path") != f"events-{index:05d}.npz":
            raise _fail("Wikimedia event shards are not sequential")
        rows = record.get("rows")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or not 1 <= rows <= shard_rows
            or (index < len(event_records) - 1 and rows != shard_rows)
        ):
            raise _fail("Wikimedia event shard rows violate the fixed bound")
    for index, record in enumerate(node_records):
        if record.get("path") != f"nodes-{index:05d}.npz":
            raise _fail("Wikimedia node shards are not sequential")
        rows = record.get("rows")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or not 1 <= rows <= shard_rows
            or (index < len(node_records) - 1 and rows != shard_rows)
        ):
            raise _fail("Wikimedia node shard rows violate the fixed bound")

    count = int(manifest.get("eventCount", -1))
    maximum = int(manifest.get("maximumComments", -1))
    if not 1 <= count <= maximum <= MAX_COMMENTS:
        raise _fail("Wikimedia event count or sampling cap is invalid")
    if sum(int(item["rows"]) for item in event_records) != count:
        raise _fail("Wikimedia event shard rows do not sum to eventCount")
    if text_records[0].get("rows") != count:
        raise _fail("Wikimedia text shard rows do not match eventCount")

    offsets = manifest.get("nodeOffsets")
    node_count = int(manifest.get("nodeCount", -1))
    user_count = int(manifest.get("userCount", -1))
    page_count = int(manifest.get("pageCount", -1))
    if (
        not isinstance(offsets, dict)
        or offsets.get("user") != 0
        or offsets.get("page") != user_count
        or user_count < 1
        or page_count < 1
        or node_count != user_count + page_count
    ):
        raise _fail("Wikimedia node offsets are invalid")
    if sum(int(item["rows"]) for item in node_records) != node_count:
        raise _fail("Wikimedia node shard rows do not sum to nodeCount")

    split_counts_contract = splits.get("counts")
    if not isinstance(split_counts_contract, dict):
        raise _fail("Wikimedia split counts are absent")
    event_counts_contract = split_counts_contract.get("events")
    page_counts_contract = split_counts_contract.get("pages")
    if not isinstance(event_counts_contract, dict) or not isinstance(
        page_counts_contract, dict
    ):
        raise _fail("Wikimedia split counts are malformed")
    for values, total in (
        (event_counts_contract, count),
        (page_counts_contract, page_count),
    ):
        if set(values) != set(SPLIT_NAMES) or any(
            not isinstance(values[name], int)
            or isinstance(values[name], bool)
            or int(values[name]) < 0
            for name in SPLIT_NAMES
        ) or sum(int(values[name]) for name in SPLIT_NAMES) != total:
            raise _fail("Wikimedia split counts do not cover the corpus")

    node_position = 0
    for record in node_records:
        inventory = record["arrays"]
        nodes = load_npz_safe(
            resolve_within(output, str(record["path"])),
            expected={
                str(item["name"]): (str(item["dtype"]), len(item["shape"]))
                for item in inventory
                if isinstance(item, dict)
            },
        )["node_type"]
        if nodes.dtype != np.dtype(np.int16) or nodes.shape != (int(record["rows"]),):
            raise _fail("Wikimedia node shard dtype or shape is invalid")
        positions = np.arange(
            node_position, node_position + nodes.size, dtype=np.int64
        )
        if not np.array_equal(nodes, (positions >= user_count).astype(np.int16)):
            raise _fail("Wikimedia node types violate typed node offsets")
        node_position += nodes.size

    text_iterator = read_jsonl(resolve_within(output, "text.jsonl"))
    earliest = int(datetime(2011, 1, 1, tzinfo=UTC).timestamp())
    latest = int(datetime(2015, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp())
    previous_order: tuple[int, bytes] | None = None
    checked_rows = 0
    with tempfile.TemporaryDirectory(
        prefix=".wikimedia-check-", dir=output.parent
    ) as temporary_directory:
        seen = sqlite3.connect(Path(temporary_directory) / "seen.sqlite3")
        try:
            _configure_spool(seen)
            seen.execute(
                "CREATE TABLE pseudonyms(value TEXT PRIMARY KEY) WITHOUT ROWID"
            )
            seen.execute(
                """
                CREATE TABLE page_splits(
                    page_node INTEGER PRIMARY KEY,
                    split INTEGER NOT NULL CHECK(split IN (0, 1, 2)),
                    last_timestamp INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    conflict INTEGER NOT NULL CHECK(conflict IN (0, 1))
                ) WITHOUT ROWID
                """
            )
            seen_batch: list[tuple[str]] = []
            page_batch: list[tuple[int, int, int]] = []
            actual_event_split_counts = [0, 0, 0]
            canonical_role_digests = [hashlib.sha256() for _ in SPLIT_NAMES]

            def flush_audit() -> None:
                if not seen_batch:
                    return
                try:
                    seen.executemany(
                        "INSERT INTO pseudonyms(value) VALUES (?)", seen_batch
                    )
                except sqlite3.IntegrityError as exc:
                    raise _fail(
                        "Wikimedia revision pseudonyms are not unique"
                    ) from exc
                seen.executemany(
                    """
                    INSERT INTO page_splits(
                        page_node, split, last_timestamp, event_count, conflict
                    ) VALUES (?, ?, ?, 1, 0)
                    ON CONFLICT(page_node) DO UPDATE SET
                        last_timestamp=MAX(
                            page_splits.last_timestamp, excluded.last_timestamp
                        ),
                        event_count=page_splits.event_count + 1,
                        conflict=CASE
                            WHEN page_splits.split != excluded.split THEN 1
                            ELSE page_splits.conflict
                        END
                    """,
                    page_batch,
                )
                seen.commit()
                seen_batch.clear()
                page_batch.clear()

            for record in event_records:
                inventory = record["arrays"]
                events = load_npz_safe(
                    resolve_within(output, str(record["path"])),
                    expected={
                        str(item["name"]): (
                            str(item["dtype"]),
                            len(item["shape"]),
                        )
                        for item in inventory
                        if isinstance(item, dict)
                    },
                )
                rows = int(record["rows"])
                expected_dtypes = {
                    "src": np.dtype(np.int64),
                    "dst": np.dtype(np.int64),
                    "timestamp": np.dtype(np.int64),
                    "relation": np.dtype(np.int16),
                    "revision_pseudonym": np.dtype(np.uint64),
                    "split": np.dtype(np.int8),
                }
                if any(
                    events[name].dtype != dtype
                    or events[name].shape != (rows,)
                    for name, dtype in expected_dtypes.items()
                ):
                    raise _fail("Wikimedia event arrays are misaligned")
                if np.any(events["relation"] != 0):
                    raise _fail("Wikimedia event relation type is invalid")
                if np.any((events["split"] < 0) | (events["split"] > 2)):
                    raise _fail("Wikimedia event split code is invalid")
                for split_code in range(3):
                    _update_event_digest(
                        canonical_role_digests[split_code],
                        events,
                        events["split"] == split_code,
                    )
                if (
                    int(events["src"].min()) < 0
                    or int(events["src"].max()) >= user_count
                    or int(events["dst"].min()) < user_count
                    or int(events["dst"].max()) >= node_count
                ):
                    raise _fail(
                        "Wikimedia event endpoints violate typed node offsets"
                    )
                for index in range(rows):
                    timestamp = int(events["timestamp"][index])
                    pseudonym = int(events["revision_pseudonym"][index])
                    page_node = int(events["dst"][index])
                    split = int(events["split"][index])
                    order = (timestamp, pseudonym.to_bytes(8, "little"))
                    if (
                        timestamp < earliest
                        or timestamp > latest
                        or (previous_order is not None and order <= previous_order)
                    ):
                        raise _fail(
                            "Wikimedia events violate global deterministic order"
                        )
                    previous_order = order
                    try:
                        text_row = next(text_iterator)
                    except StopIteration as exc:
                        raise _fail(
                            "Wikimedia text rows end before numeric events"
                        ) from exc
                    expected_id = str(pseudonym)
                    if (
                        set(text_row) != {"id", "text", "timestamp"}
                        or text_row.get("id") != expected_id
                        or text_row.get("timestamp") != timestamp
                        or not isinstance(text_row.get("text"), str)
                        or not str(text_row["text"]).strip()
                    ):
                        raise _fail(
                            "Wikimedia text row is invalid or event-misaligned"
                        )
                    seen_batch.append((expected_id,))
                    page_batch.append((page_node, split, timestamp))
                    actual_event_split_counts[split] += 1
                    if len(seen_batch) == SQLITE_BATCH_ROWS:
                        flush_audit()
                    checked_rows += 1
            flush_audit()
            if seen.execute(
                "SELECT 1 FROM page_splits WHERE conflict=1 LIMIT 1"
            ).fetchone() is not None:
                raise _fail("a Wikimedia page appears in more than one split")
            actual_page_split_counts = [0, 0, 0]
            audited_pages = 0
            for split, last_timestamp in seen.execute(
                "SELECT split, last_timestamp FROM page_splits ORDER BY page_node"
            ):
                split_value = int(split)
                if split_value != _split_for_last_timestamp(int(last_timestamp)):
                    raise _fail(
                        "Wikimedia page split differs from its last event time"
                    )
                actual_page_split_counts[split_value] += 1
                audited_pages += 1
            if audited_pages != page_count:
                raise _fail("Wikimedia split audit did not cover every page")
            if any(
                int(event_counts_contract[name]) != actual_event_split_counts[index]
                or int(page_counts_contract[name])
                != actual_page_split_counts[index]
                for index, name in enumerate(SPLIT_NAMES)
            ):
                raise _fail("Wikimedia split counts differ from numeric events")
            if any(
                canonical_role_digests[index].digest()
                != access_role_digests[index].digest()
                for index in range(3)
            ):
                raise _fail(
                    "Wikimedia physical role shards are not exact canonical subsequences"
                )
            try:
                next(text_iterator)
            except StopIteration:
                pass
            else:
                raise _fail("Wikimedia text rows outlast numeric events")
        finally:
            seen.close()
    if checked_rows != count:
        raise _fail("Wikimedia checker did not consume every event")
    return manifest


def check_wikimedia(root: str | Path) -> dict[str, Any]:
    output = RuntimeLayout.from_root(root).processed_gfm / CORPUS_ID
    return _check_wikimedia_directory(output)
