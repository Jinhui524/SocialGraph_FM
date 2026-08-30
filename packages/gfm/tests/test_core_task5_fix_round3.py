import sqlite3

import pytest

from socialgraph_gfm.core.knowledge import (
    KnowledgeStore,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
)


_PROJECT_COLUMNS = """
    record_hash TEXT NOT NULL {record_unique},
    finding_hash TEXT NOT NULL,
    review_status TEXT NOT NULL {status_unique},
    reviewer_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    FOREIGN KEY (finding_hash) REFERENCES registered_findings(finding_hash)
"""


def _rebuild_project_memory(
    connection: sqlite3.Connection,
    *,
    autoincrement: bool = True,
    record_unique: bool = True,
    status_unique: bool = False,
) -> None:
    connection.executescript(
        """
        DROP TRIGGER project_memory_no_update;
        DROP TRIGGER project_memory_no_delete;
        ALTER TABLE project_memory RENAME TO old_project_memory;
        """
    )
    sequence = "INTEGER PRIMARY KEY AUTOINCREMENT" if autoincrement else "INTEGER PRIMARY KEY"
    columns = _PROJECT_COLUMNS.format(
        record_unique="UNIQUE" if record_unique else "",
        status_unique="UNIQUE" if status_unique else "",
    )
    connection.executescript(
        f"""
        CREATE TABLE project_memory (
            sequence {sequence},
            {columns}
        );
        DROP TABLE old_project_memory;
        CREATE TRIGGER project_memory_no_update
        BEFORE UPDATE ON project_memory
        BEGIN SELECT RAISE(ABORT, 'project memory is append-only'); END;
        CREATE TRIGGER project_memory_no_delete
        BEFORE DELETE ON project_memory
        BEGIN SELECT RAISE(ABORT, 'project memory is append-only'); END;
        """
    )


def _remove_metadata_check(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT schema_version, schema_fingerprint FROM schema_metadata WHERE singleton = 1"
    ).fetchone()
    assert row is not None
    connection.executescript(
        """
        ALTER TABLE schema_metadata RENAME TO old_schema_metadata;
        CREATE TABLE schema_metadata (
            singleton INTEGER PRIMARY KEY,
            schema_version TEXT NOT NULL,
            schema_fingerprint TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO schema_metadata VALUES (1, ?, ?)",
        row,
    )
    connection.execute("DROP TABLE old_schema_metadata")


def _replace_with_no_op_trigger(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TRIGGER project_memory_no_delete;
        CREATE TRIGGER project_memory_no_delete
        BEFORE DELETE ON project_memory
        BEGIN SELECT 1; END;
        """
    )


def _alter_fts_tokenizer(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        DROP TABLE knowledge_fts;
        CREATE VIRTUAL TABLE knowledge_fts USING fts5(
            document_hash UNINDEXED,
            title,
            body,
            category,
            tokenize = 'porter'
        );
        """
    )


def _add_extra_object(connection: sqlite3.Connection) -> None:
    connection.execute("CREATE TABLE unexpected_application_table(value TEXT)")


def _remove_fts_shadow(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA writable_schema = ON")
    connection.execute("DELETE FROM sqlite_master WHERE name = 'knowledge_fts_config'")
    connection.execute("PRAGMA writable_schema = OFF")


@pytest.mark.parametrize(
    "mutate",
    (
        lambda connection: _rebuild_project_memory(connection, record_unique=False),
        lambda connection: _rebuild_project_memory(connection, autoincrement=False),
        _remove_metadata_check,
        lambda connection: connection.execute(
            "CREATE UNIQUE INDEX extra_document_title_unique ON knowledge_documents(title)"
        ),
        lambda connection: _rebuild_project_memory(
            connection, record_unique=False, status_unique=True
        ),
        _replace_with_no_op_trigger,
        _alter_fts_tokenizer,
        _add_extra_object,
        _remove_fts_shadow,
    ),
    ids=(
        "missing-unique",
        "missing-autoincrement",
        "missing-check",
        "extra-unique-index",
        "wrong-unique-index",
        "same-name-no-op-trigger",
        "altered-fts-definition",
        "extra-application-object",
        "missing-fts-shadow",
    ),
)
def test_same_version_forged_layout_fails_before_use(tmp_path, mutate):
    path = tmp_path / "forged.sqlite3"
    KnowledgeStore(path)
    with sqlite3.connect(path) as connection:
        mutate(connection)
        metadata = connection.execute(
            "SELECT schema_version, schema_fingerprint FROM schema_metadata WHERE singleton = 1"
        ).fetchone()
        assert metadata == (SCHEMA_VERSION, SCHEMA_FINGERPRINT)
    with pytest.raises(ValueError, match="layout"):
        KnowledgeStore(path)


def test_exact_layout_reopens_and_older_versions_fail_closed(tmp_path):
    current = tmp_path / "current.sqlite3"
    KnowledgeStore(current)
    KnowledgeStore(current)
    assert SCHEMA_VERSION == "socialgraph-fm.core-knowledge-sqlite/2.2"

    for version in (
        "socialgraph-fm.core-knowledge-sqlite/2.0",
        "socialgraph-fm.core-knowledge-sqlite/2.1",
    ):
        path = tmp_path / f"old-{version[-3:]}.sqlite3"
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE schema_metadata(singleton INTEGER PRIMARY KEY, schema_version TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO schema_metadata(singleton, schema_version) VALUES (1, ?)",
                (version,),
            )
        with pytest.raises(ValueError, match="unsupported knowledge SQLite schema version"):
            KnowledgeStore(path)
