from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
import numpy as np

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.errors import ContractViolation
from socialgraph_gfm.gfm.corpus import openalex
from socialgraph_gfm.gfm.corpus.common import (
    append_jsonl_fsync,
    atomic_write_json,
    atomic_write_jsonl,
    exclusive_file_lock,
    portable_id_hash,
    read_jsonl,
)
from socialgraph_gfm.gfm.corpus.domains import load_domain


def _fetch_fixture_work(number: int) -> dict[str, object]:
    value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
    value.update(
        {
            "id": f"https://openalex.org/W{number}",
            "display_name": f"Work {number}",
            "publication_date": "2016-01-01",
            "publication_year": 2016,
            "type": "article",
            "authorships": [
                {
                    "author": {"id": f"https://openalex.org/A{number}"},
                    "institutions": [],
                }
            ],
            "topics": [],
            "referenced_works": [],
        }
    )
    return value


def _open_history_resume(
    tmp_path: Path, *, name: str = "history", batch_size: int = 50
) -> tuple[Path, sqlite3.Connection]:
    database = tmp_path / name / "newcomer-verification.sqlite3"
    database.parent.mkdir(parents=True)
    return database, openalex._open_newcomer_resume_database(
        database,
        binding="binding",
        corpus_source_hash="source",
        raw_sha256="raw",
        batch_size=batch_size,
    )


def _one_page_audited_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, quota: int
) -> tuple[Path, Path, list[dict[str, object]], dict[str, str]]:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    stratum_id = "graph-network-intelligence:2016:T1"
    strata: list[dict[str, object]] = [
        {
            "stratumId": stratum_id,
            "clusterId": "graph-network-intelligence",
            "year": 2016,
            "topicId": "T1",
            "quota": quota,
        }
    ]
    monkeypatch.setattr(openalex, "_strata", lambda _clusters: strata)
    topic_ids: dict[str, str] = {}
    first_seed = hashlib.sha256(f"{stratum_id}\0{0}".encode()).hexdigest()[:16]

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        if query["seed"] != first_seed:
            raise RuntimeError("stop after one committed page")
        return {
            "results": [_fetch_fixture_work(number) for number in range(1, 101)],
            "meta": {},
        }

    with pytest.raises(RuntimeError, match="one committed page"):
        openalex.fetch_openalex(None, tmp_path, transport=transport)
    raw = tmp_path / "datasets/raw/gfm/openalex"
    return raw / openalex.RAW_NAME, raw / openalex.RESUME_NAME, strata, topic_ids


def test_openalex_requires_environment_key_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    called = False

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ContractViolation, match="OPENALEX_API_KEY"):
        openalex.fetch_openalex(openalex.OpenAlexConfig.pinned(), tmp_path, transport=transport)
    assert called is False


def test_fetch_holds_exclusive_lock_across_topic_and_work_requests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    monkeypatch.setattr(
        openalex,
        "_strata",
        lambda _clusters: [
            {
                "stratumId": "graph-network-intelligence:2016:T1",
                "clusterId": "graph-network-intelligence",
                "year": 2016,
                "topicId": "T1",
                "quota": 1,
            }
        ],
    )
    lock_path = tmp_path / "datasets/raw/gfm/openalex" / openalex.FETCH_LOCK_NAME
    topic_ids: dict[str, str] = {}
    observed: list[str] = []

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        with pytest.raises(ContractViolation, match="already running"):
            with exclusive_file_lock(lock_path):
                raise AssertionError("fetch transport must remain inside its lock")
        if url.endswith("/topics"):
            observed.append("topic")
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        observed.append("work")
        return {"results": [_fetch_fixture_work(1)], "meta": {}}

    result = openalex.fetch_openalex(None, tmp_path, transport=transport)
    assert result["rows"] == 1
    assert "topic" in observed and "work" in observed
    with exclusive_file_lock(lock_path):
        pass


def test_topic_resolution_fails_closed_on_ambiguity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-not-persisted")

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        return {
            "results": [
                {"id": "https://openalex.org/T1", "display_name": "Network science"},
                {"id": "https://openalex.org/T2", "display_name": "Network science"},
            ]
        }

    with pytest.raises(ContractViolation, match="2 exact matches"):
        openalex.parse_topic_selector("network science", transport=transport)


def test_topic_resolution_persists_exact_response_summary_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret-not-persisted")

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        assert query["search"] == "Advanced Graph Neural Networks"
        return {
            "results": [
                {
                    "id": "https://openalex.org/T11273",
                    "display_name": "Advanced Graph Neural Networks",
                },
                {"id": "https://openalex.org/T1", "display_name": "Other"},
            ]
        }

    resolved = openalex.parse_topic_selector("Advanced Graph Neural Networks", transport=transport)
    assert resolved["id"] == "T11273"
    assert resolved["candidateCount"] == 2
    assert len(str(resolved["responseSummaryHash"])) == 64
    assert "secret" not in json.dumps(resolved)


def test_work_allowlist_blocks_snapshot_aggregates() -> None:
    work = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
    work.update(
        {
            "id": "https://openalex.org/W1",
            "publication_date": "2020-01-01",
            "publication_year": 2020,
            "type": "article",
            "authorships": [{"author": {"id": "https://openalex.org/A1"}}],
            "topics": [],
            "referenced_works": [],
            "cited_by_count": 999,
        }
    )
    with pytest.raises(ContractViolation, match="blocked leakage"):
        openalex._validate_work(work)


def test_historical_newcomer_query_supports_pre_2016_and_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    queries: list[dict[str, str]] = []

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        queries.append(dict(query))
        return {
            "results": [
                {
                    "publication_date": "2001-02-03",
                    "authorships": [{"author": {"id": "https://openalex.org/A1"}}],
                }
            ],
            "meta": {"next_cursor": None},
        }

    result = openalex.fetch_historical_newcomers(
        ["A1"], date(1900, 1, 1), date(2015, 12, 31), transport=transport
    )
    assert result == {"A1": date(2001, 2, 3)}
    assert "from_publication_date:1900-01-01" in queries[0]["filter"]
    assert queries[0]["per_page"] == "100"
    assert "secret" not in json.dumps(result, default=str)


def test_strata_use_exact_caps_and_bounded_requests() -> None:
    clusters = [
        {
            "clusterId": f"c{index}",
            "maximumWorks": cap,
            "topics": [{"id": f"T{index * 10 + item}"} for item in range(4 if index == 0 else 3)],
        }
        for index, cap in enumerate(openalex.CLUSTER_CAPS)
    ]
    strata = openalex._strata(clusters)
    assert sum(item["quota"] for item in strata) == 200_000
    assert max(item["quota"] for item in strata) <= 10_000
    assert sum((item["quota"] + 99) // 100 for item in strata) <= openalex.MAX_FETCH_REQUESTS


def test_newcomer_group_by_overlay_is_executable_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    raw = tmp_path / "datasets/raw/gfm/openalex"
    raw.mkdir(parents=True)
    topics = {
        "schemaVersion": "gfm.openalex-topics/1.0",
        "configHash": "fixture",
        "formalEligible": False,
        "clusters": [
            {"clusterId": name, "maximumWorks": cap, "topics": [{"id": f"T{i + 1}"}]}
            for i, (name, cap) in enumerate(zip(("c1", "c2", "c3"), openalex.CLUSTER_CAPS))
        ],
    }
    work = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
    work.update(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Fixture graph work",
            "publication_date": "2016-03-01",
            "publication_year": 2016,
            "type": "article",
            "authorships": [
                {"author": {"id": "https://openalex.org/A1"}, "institutions": []},
                {"author": {"id": "https://openalex.org/A2"}, "institutions": []},
            ],
            "topics": [],
            "referenced_works": [],
        }
    )
    atomic_write_json(raw / openalex.TOPICS_NAME, topics)
    atomic_write_jsonl(
        raw / openalex.RAW_NAME,
        [{"clusterId": "c1", "stratumId": "c1:2016:T1", "work": work}],
    )
    atomic_write_json(
        raw / openalex.RESUME_NAME,
        {
            "schemaVersion": openalex.RESUME_SCHEMA,
            "complete": True,
            "formalEligible": False,
            "rawRows": 1,
            "strata": [{"stratumId": "c1:2016:T1", "requested": 1, "received": 1}],
        },
    )
    openalex.prepare_openalex(raw, tmp_path)
    base_manifest_path = (
        tmp_path / "datasets/processed/gfm/openalex-graph-ai/manifest.json"
    )
    base_manifest_before = base_manifest_path.read_bytes()
    base_hash_before = openalex.check_openalex(tmp_path)["logicalHash"]
    absent_status = openalex.newcomer_overlay_status(tmp_path)
    assert absent_status["ready"] is False
    assert absent_status["state"] == "absent"
    assert "true_newcomer" not in load_domain(tmp_path, openalex.DOMAIN_ID)["arrays"]

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        assert query["group_by"] == "authorships.author.id"
        assert "to_publication_date:2016-02-29" in query["filter"]
        return {
            "group_by": [{"key": "https://openalex.org/A1", "count": 1}],
            "meta": {"groups_count": 1, "next_cursor": None},
        }

    overlay = openalex.verify_openalex_newcomers(tmp_path, transport=transport)
    final_overlay = (
        tmp_path
        / "datasets/processed/gfm"
        / openalex.NEWCOMER_OVERLAY_ID
    )
    assert final_overlay.is_dir()
    assert not list(
        final_overlay.parent.glob(f".{openalex.NEWCOMER_OVERLAY_ID}-publish-*.tmp")
    )

    def must_not_refetch(url: str, query: dict[str, str]) -> dict[str, object]:
        raise AssertionError("recovery must not repeat OpenAlex API requests")

    repeated = openalex.verify_openalex_newcomers(tmp_path, transport=must_not_refetch)
    assert repeated["logicalHash"] == overlay["logicalHash"]
    assert overlay["verifiedCount"] == 2
    assert overlay["trueNewcomerCount"] == 1
    assert overlay["historyQueryProtocol"] == openalex.NEWCOMER_HISTORY_QUERY_POLICY
    assert overlay["minimumRootRequests"] == 1
    assert overlay["strictWorstRequests"] == 3
    assert overlay["configuredRequestBudget"] == openalex.MAX_NEWCOMER_REQUESTS
    assert overlay["historyQueryAudit"] == {
        "protocol": openalex.NEWCOMER_HISTORY_QUERY_POLICY,
        "requestCount": 1,
        "minimumRootRequests": 1,
        "strictWorstRequests": 3,
        "configuredRequestBudget": openalex.MAX_NEWCOMER_REQUESTS,
        "groupByRequests": 1,
        "existenceRequests": 0,
        "truncatedResponses": 0,
        "extraGroupsIgnored": 0,
        "nullGroupsIgnored": 0,
        "returnedGroups": 1,
        "reportedGroups": 1,
    }
    assert overlay["selectionStore"] == "ephemeral-sqlite-not-published"
    processed_parent = tmp_path / "datasets/processed/gfm"
    assert not list(
        processed_parent.glob(
            f"{openalex.NEWCOMER_RESUME_PREFIX}*{openalex.NEWCOMER_RESUME_SUFFIX}"
        )
    )
    assert not list(final_overlay.rglob("*.sqlite3"))
    persisted = (final_overlay / "manifest.json").read_text()
    assert "A1" not in persisted and "secret" not in persisted
    parent = openalex.check_openalex(tmp_path)
    assert parent["logicalHash"] == base_hash_before
    assert base_manifest_path.read_bytes() == base_manifest_before
    assert "newcomerVerification" not in parent
    parent_paths = {item["path"] for item in parent["shards"]}
    assert not any(path.startswith("newcomer-verification/") for path in parent_paths)
    assert overlay["corpusId"] == openalex.NEWCOMER_OVERLAY_ID
    assert overlay["source"]["baseCorpusLogicalHash"] == base_hash_before
    assert overlay["source"]["baseCorpusSourceHash"] == overlay["source"]["corpusSourceHash"]
    arrays = openalex.load_openalex_newcomers(tmp_path)
    assert arrays["history_verified"].tolist() == [True, True]
    assert arrays["true_newcomer"].tolist() == [False, True]
    loaded_domain = load_domain(tmp_path, openalex.DOMAIN_ID)["arrays"]
    assert "true_newcomer" not in loaded_domain
    assert loaded_domain["newcomers.history_verified"].tolist() == [False, False]
    status = openalex.newcomer_overlay_status(tmp_path)
    assert status["ready"] is True
    assert status["manifestHash"] == overlay["logicalHash"]

    overlay_manifest_path = final_overlay / "manifest.json"
    overlay_manifest_bytes = overlay_manifest_path.read_bytes()
    rebound = json.loads(overlay_manifest_bytes)
    rebound["source"]["baseCorpusLogicalHash"] = "0" * 64
    rebound["logicalHash"] = canonical_sha256(
        {
            key: value
            for key, value in rebound.items()
            if key not in {"logicalHash", "createdAt"}
        }
    )
    atomic_write_json(overlay_manifest_path, rebound)
    with pytest.raises(ContractViolation, match="bound to another OpenAlex base corpus"):
        openalex.check_openalex_newcomers(tmp_path)
    with pytest.raises(ContractViolation, match="bound to another OpenAlex base corpus"):
        openalex.load_openalex_newcomers_view(tmp_path, maximum_role="validation")
    assert openalex.check_openalex(tmp_path)["logicalHash"] == base_hash_before
    overlay_manifest_path.write_bytes(overlay_manifest_bytes)

    validation_view = openalex.load_openalex_newcomers_view(tmp_path, maximum_role="validation")
    assert validation_view["arrays"]["author"].tolist() == [0, 1]
    assert validation_view["accessAudit"]["testArtifactsOpened"] is False
    test_role_path = Path(overlay["physicalAccess"]["roleShards"]["test"][0])
    test_role_artifact = final_overlay / test_role_path
    test_role_bytes = test_role_artifact.read_bytes()
    test_role_artifact.write_bytes(test_role_bytes + b"tamper")
    # A validation worker never hashes or opens the now-corrupt test cohort.
    openalex.load_openalex_newcomers_view(tmp_path, maximum_role="validation")
    with pytest.raises(ContractViolation, match="hash mismatch"):
        openalex.load_openalex_newcomers_view(tmp_path, maximum_role="test")
    # Overlay corruption never invalidates the independent base corpus.
    assert openalex.check_openalex(tmp_path)["logicalHash"] == base_hash_before
    test_role_artifact.write_bytes(test_role_bytes)

    artifact = final_overlay / "artifact.npz"
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    assert openalex.check_openalex(tmp_path)["logicalHash"] == base_hash_before
    with pytest.raises(ContractViolation, match="hash mismatch"):
        openalex.check_openalex_newcomers(tmp_path)
    with pytest.raises(ContractViolation, match="hash mismatch"):
        openalex.load_openalex_newcomers(tmp_path)

    legacy = base_manifest_path.parent / "newcomer-verification"
    legacy.mkdir()
    with pytest.raises(ContractViolation, match="legacy nested newcomer overlay"):
        openalex.check_openalex(tmp_path)
    legacy_status = openalex.newcomer_overlay_status(tmp_path)
    assert legacy_status["state"] == "legacy-nested-rejected"


def test_newcomer_resume_commits_ingest_and_stores_work_json_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "resume" / "newcomer-verification.sqlite3"
    database.parent.mkdir()
    connection = openalex._open_newcomer_resume_database(
        database,
        binding="binding",
        corpus_source_hash="source",
        raw_sha256="raw",
        batch_size=50,
    )
    selected_columns = {row[1] for row in connection.execute("PRAGMA table_info(selected_work)")}
    assert selected_columns == {"work_id"}
    candidate_columns = {row[1] for row in connection.execute("PRAGMA table_info(candidates)")}
    assert "work_json" not in candidate_columns
    assert {row[1] for row in connection.execute("PRAGMA table_info(candidate_work)")} == {
        "work_id",
        "work_json",
    }
    assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)
    assert connection.execute("PRAGMA synchronous").fetchone() == (2,)

    raw_path = tmp_path / "works.jsonl"
    atomic_write_jsonl(
        raw_path,
        [
            {
                "clusterId": "c1",
                "stratumId": "c1:2016:T1",
                "work": _fetch_fixture_work(1),
            },
            {"invalid": True},
        ],
    )
    monkeypatch.setattr(openalex, "NEWCOMER_INGEST_COMMIT_ROWS", 1)
    with pytest.raises(ContractViolation, match="raw work row is invalid"):
        openalex._ingest_newcomer_candidates(connection, raw_path, {"c1": 1})
    connection.close()

    resumed = openalex._open_newcomer_resume_database(
        database,
        binding="binding",
        corpus_source_hash="source",
        raw_sha256="raw",
        batch_size=50,
    )
    assert openalex._newcomer_meta(resumed, "raw_rows") == "1"
    assert resumed.execute("SELECT COUNT(*) FROM candidates").fetchone() == (1,)
    assert resumed.execute("SELECT COUNT(*) FROM candidate_work").fetchone() == (1,)
    resumed.close()


def test_newcomer_resume_does_not_repeat_committed_api_batch(tmp_path: Path) -> None:
    database = tmp_path / "resume" / "newcomer-verification.sqlite3"
    database.parent.mkdir()
    connection = openalex._open_newcomer_resume_database(
        database,
        binding="binding",
        corpus_source_hash="source",
        raw_sha256="raw",
        batch_size=1,
    )
    t0 = int(datetime(2016, 3, 1, tzinfo=UTC).timestamp())
    connection.executemany(
        "INSERT INTO authors(author_id, t0) VALUES (?, ?)", (("A1", t0), ("A2", t0))
    )
    openalex._set_newcomer_stage(connection, "authors_built")
    calls: list[str] = []

    def interrupted(url: str, query: dict[str, str]) -> dict[str, object]:
        assert query["select"] == "id"
        assert query["per_page"] == "1"
        assert "group_by" not in query
        calls.append(query["filter"])
        if len(calls) == 2:
            raise RuntimeError("interrupt after committed batch")
        return {"results": [], "meta": {"count": 0}}

    with pytest.raises(RuntimeError, match="committed batch"):
        openalex._verify_newcomer_t0_batches(
            connection,
            t0_value=t0,
            after_author="",
            batch_size=1,
            request=interrupted,
            api_key="secret",
        )
    connection.close()

    resumed = openalex._open_newcomer_resume_database(
        database,
        binding="binding",
        corpus_source_hash="source",
        raw_sha256="raw",
        batch_size=1,
    )
    assert openalex._newcomer_meta(resumed, "last_author") == "A1"
    assert openalex._newcomer_meta(resumed, "request_count") == "2"
    assert openalex._newcomer_meta(resumed, "history_existence_requests") == "2"
    resumed_calls: list[str] = []

    def finish(url: str, query: dict[str, str]) -> dict[str, object]:
        assert query["select"] == "id"
        assert query["per_page"] == "1"
        assert "group_by" not in query
        resumed_calls.append(query["filter"])
        return {"results": [], "meta": {"count": 0}}

    openalex._verify_newcomer_t0_batches(
        resumed,
        t0_value=t0,
        after_author=openalex._newcomer_meta(resumed, "last_author"),
        batch_size=1,
        request=finish,
        api_key="secret",
    )
    assert len(resumed_calls) == 1
    assert "A2" in resumed_calls[0]
    assert "A1" not in resumed_calls[0]
    # The interrupted second request consumed its persistent reservation.  A
    # retry therefore uses request three instead of silently reusing budget.
    assert openalex._newcomer_meta(resumed, "request_count") == "3"
    assert openalex._newcomer_meta(resumed, "history_existence_requests") == "3"
    resumed.close()


def test_newcomer_group_by_uses_cursor_and_ignores_audited_extra_groups(
    tmp_path: Path,
) -> None:
    _, connection = _open_history_resume(tmp_path)
    queries: list[dict[str, str]] = []

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        queries.append(dict(query))
        return {
            "group_by": [
                {"key": "https://openalex.org/A1", "count": 2},
                {"key": "https://openalex.org/A9", "count": 1},
                {"key": None, "count": 3},
                {"key": f"https://openalex.org/{openalex.NULL_AUTHOR_ID}", "count": 4},
            ],
            # OpenAlex can legitimately report 200 while returning fewer
            # group objects.  Only next_cursor is a completeness signal.
            "meta": {"groups_count": 200, "next_cursor": None},
        }

    resolved = openalex._resolve_newcomer_prior_history(
        connection,
        batch=("A2", "A1"),
        cutoff=date(2016, 2, 29),
        request=transport,
        api_key="secret",
    )
    assert resolved == {"A1"}
    assert len(queries) == 1
    assert queries[0]["group_by"] == "authorships.author.id"
    assert queries[0]["cursor"] == "*"
    assert queries[0]["per_page"] == "200"
    assert "authorships.author.id:A1|A2" in queries[0]["filter"]
    assert openalex._newcomer_history_query_audit(connection) == {
        "protocol": openalex.NEWCOMER_HISTORY_QUERY_POLICY,
        "requestCount": 1,
        "minimumRootRequests": 0,
        "strictWorstRequests": 0,
        "configuredRequestBudget": openalex.MAX_NEWCOMER_REQUESTS,
        "groupByRequests": 1,
        "existenceRequests": 0,
        "truncatedResponses": 0,
        "extraGroupsIgnored": 1,
        "nullGroupsIgnored": 2,
        "returnedGroups": 4,
        "reportedGroups": 200,
    }
    connection.close()


def test_newcomer_group_by_cursor_recurses_only_unresolved_authors(
    tmp_path: Path,
) -> None:
    _, connection = _open_history_resume(tmp_path)
    queries: list[dict[str, str]] = []

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        queries.append(dict(query))
        author_filter = query["filter"].split(",", 1)[0]
        if author_filter.endswith("A1|A2|A3|A4"):
            assert query["cursor"] == "*"
            return {
                "group_by": [
                    {"key": "https://openalex.org/A1", "count": 1},
                    {"key": "https://openalex.org/A9", "count": 1},
                ],
                "meta": {"groups_count": 200, "next_cursor": "continuation"},
            }
        if author_filter.endswith("A2"):
            assert query["select"] == "id"
            assert query["per_page"] == "1"
            assert "group_by" not in query
            return {
                "results": [{"id": "https://openalex.org/W2"}],
                "meta": {"count": 1},
            }
        assert author_filter.endswith("A3|A4")
        assert "A1" not in author_filter and "A2" not in author_filter
        return {
            "group_by": [{"key": "https://openalex.org/A3", "count": 2}],
            "meta": {"groups_count": 1, "next_cursor": None},
        }

    resolved = openalex._resolve_newcomer_prior_history(
        connection,
        batch=("A4", "A3", "A2", "A1"),
        cutoff=date(2016, 2, 29),
        request=transport,
        api_key="secret",
    )
    assert resolved == {"A1", "A2", "A3"}
    assert len(queries) == 3
    audit = openalex._newcomer_history_query_audit(connection)
    assert audit["requestCount"] == 3
    assert audit["groupByRequests"] == 2
    assert audit["existenceRequests"] == 1
    assert audit["truncatedResponses"] == 1
    assert audit["extraGroupsIgnored"] == 1
    assert audit["returnedGroups"] == 3
    assert audit["reportedGroups"] == 201
    connection.close()


@pytest.mark.parametrize(
    ("meta", "groups", "message"),
    [
        ({"groups_count": 0}, [], "no next_cursor"),
        ({"groups_count": 0, "next_cursor": ""}, [], "invalid next_cursor"),
        ({"groups_count": -1, "next_cursor": None}, [], "groups_count"),
        ({"groups_count": True, "next_cursor": None}, [], "groups_count"),
        ({"groups_count": 0, "next_cursor": 7}, [], "next_cursor"),
        (
            {"groups_count": 1, "next_cursor": None},
            [{"key": "https://openalex.org/A1", "count": 0}],
            "item count",
        ),
    ],
)
def test_newcomer_group_by_rejects_malformed_telemetry_or_groups(
    tmp_path: Path,
    meta: dict[str, object],
    groups: list[dict[str, object]],
    message: str,
) -> None:
    _, connection = _open_history_resume(tmp_path)

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        return {"group_by": groups, "meta": meta}

    with pytest.raises(ContractViolation, match=message):
        openalex._resolve_newcomer_prior_history(
            connection,
            batch=("A1", "A2"),
            cutoff=date(2016, 2, 29),
            request=transport,
            api_key="secret",
        )
    # Even a malformed response consumes the request reserved before transport.
    assert openalex._newcomer_meta(connection, "request_count") == "1"
    assert openalex._newcomer_meta(connection, "history_group_by_requests") == "1"
    connection.close()


def test_newcomer_history_budget_is_reserved_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, connection = _open_history_resume(tmp_path)
    monkeypatch.setattr(openalex, "MAX_NEWCOMER_REQUESTS", 1)
    calls = 0

    def interrupted(url: str, query: dict[str, str]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise RuntimeError("network interruption")

    with pytest.raises(RuntimeError, match="network interruption"):
        openalex._resolve_newcomer_prior_history(
            connection,
            batch=("A1",),
            cutoff=date(2016, 2, 29),
            request=interrupted,
            api_key="secret",
        )
    assert openalex._newcomer_meta(connection, "request_count") == "1"
    with pytest.raises(ContractViolation, match="request budget") as exc:
        openalex._resolve_newcomer_prior_history(
            connection,
            batch=("A1",),
            cutoff=date(2016, 2, 29),
            request=interrupted,
            api_key="secret",
        )
    assert "actual=1" in str(exc.value)
    assert "minimumRootRequests=0" in str(exc.value)
    assert "strictWorstRequests=0" in str(exc.value)
    assert "configuredRequestBudget=1" in str(exc.value)
    assert calls == 1
    assert openalex._newcomer_meta(connection, "request_count") == "1"
    connection.close()


def test_newcomer_history_protocol_migrates_only_unstarted_legacy_resume(
    tmp_path: Path,
) -> None:
    database, connection = _open_history_resume(tmp_path, name="unstarted")
    openalex._set_newcomer_stage(connection, "authors_built")
    names = ("history_query_protocol", *openalex.NEWCOMER_HISTORY_AUDIT_KEYS)
    connection.execute(
        f"DELETE FROM metadata WHERE key IN ({','.join('?' for _ in names)})", names
    )
    connection.commit()
    connection.close()

    migrated = openalex._open_newcomer_resume_database(
        database,
        binding="binding",
        corpus_source_hash="source",
        raw_sha256="raw",
        batch_size=50,
    )
    assert openalex._newcomer_meta(migrated, "stage") == "authors_built"
    assert openalex._newcomer_history_query_audit(migrated)["requestCount"] == 0
    migrated.close()


@pytest.mark.parametrize("partial_new_protocol", [False, True])
def test_newcomer_history_protocol_rejects_ambiguous_legacy_or_partial_state(
    tmp_path: Path, partial_new_protocol: bool
) -> None:
    name = "partial-new" if partial_new_protocol else "partial-legacy"
    database, connection = _open_history_resume(tmp_path, name=name)
    openalex._set_newcomer_stage(connection, "authors_built")
    if partial_new_protocol:
        connection.execute(
            "DELETE FROM metadata WHERE key = ?", (openalex.NEWCOMER_HISTORY_AUDIT_KEYS[-1],)
        )
    else:
        names = ("history_query_protocol", *openalex.NEWCOMER_HISTORY_AUDIT_KEYS)
        connection.execute(
            f"DELETE FROM metadata WHERE key IN ({','.join('?' for _ in names)})", names
        )
        openalex._set_newcomer_metadata(connection, {"request_count": "1"})
    connection.commit()
    connection.close()
    before = database.read_bytes()

    with pytest.raises(ContractViolation, match="corrupt or incompatible"):
        openalex._open_newcomer_resume_database(
            database,
            binding="binding",
            corpus_source_hash="source",
            raw_sha256="raw",
            batch_size=50,
        )
    assert database.read_bytes() == before


def test_newcomer_verifier_lock_is_full_lifecycle_fail_closed_and_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock_path = (
        tmp_path
        / "datasets/processed/gfm"
        / openalex.NEWCOMER_VERIFY_LOCK_NAME
    )
    invoked = False

    def fake_unlocked(
        root: str | Path,
        *,
        transport: object = None,
        api_key: str | None = None,
        batch_size: int = 50,
    ) -> dict[str, object]:
        nonlocal invoked
        invoked = True
        with pytest.raises(ContractViolation, match="already running"):
            with exclusive_file_lock(lock_path):
                raise AssertionError("the public verifier must retain its lock")
        return {"complete": True}

    monkeypatch.setattr(openalex, "_verify_openalex_newcomers_unlocked", fake_unlocked)
    with exclusive_file_lock(lock_path):
        with pytest.raises(ContractViolation, match="already running"):
            openalex.verify_openalex_newcomers(tmp_path)
    assert invoked is False

    assert openalex.verify_openalex_newcomers(tmp_path) == {"complete": True}
    assert invoked is True
    with exclusive_file_lock(lock_path):
        pass


def test_newcomer_resume_rejects_malformed_database_with_exact_path(tmp_path: Path) -> None:
    database = tmp_path / "resume" / "newcomer-verification.sqlite3"
    database.parent.mkdir()
    database.write_bytes(b"not a sqlite database")
    with pytest.raises(ContractViolation, match="remove only this exact staging directory") as exc:
        openalex._open_newcomer_resume_database(
            database,
            binding="binding",
            corpus_source_hash="source",
            raw_sha256="raw",
            batch_size=50,
        )
    assert str(database.parent) in str(exc.value)
    assert database.exists()


def test_fetch_resume_restores_mid_stratum_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    monkeypatch.setattr(
        openalex,
        "_strata",
        lambda _clusters: [
            {
                "stratumId": "graph-network-intelligence:2016:T1",
                "clusterId": "graph-network-intelligence",
                "year": 2016,
                "topicId": "T1",
                "quota": 150,
            }
        ],
    )
    topic_ids: dict[str, str] = {}
    interrupted = True

    def work(number: int) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        value.update(
            {
                "id": f"https://openalex.org/W{number}",
                "display_name": f"Work {number}",
                "publication_date": "2016-01-01",
                "publication_year": 2016,
                "type": "article",
                "authorships": [
                    {"author": {"id": f"https://openalex.org/A{number}"}, "institutions": []}
                ],
                "topics": [],
                "referenced_works": [],
            }
        )
        return value

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        nonlocal interrupted
        if url.endswith("/topics"):
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        assert "page" not in query
        draw_size = int(query["sample"])
        stratum = "graph-network-intelligence:2016:T1"
        first_seed = openalex.hashlib.sha256(f"{stratum}\0{0}".encode()).hexdigest()[:16]
        draw = 0 if query["seed"] == first_seed else 1
        if draw == 1 and interrupted:
            interrupted = False
            raise RuntimeError("mock interruption")
        start = draw * 100 + 1
        return {
            "results": [work(index) for index in range(start, start + draw_size)],
            "meta": {},
        }

    with pytest.raises(RuntimeError, match="interruption"):
        openalex.fetch_openalex(None, tmp_path, transport=transport)
    state = json.loads(
        (tmp_path / "datasets/raw/gfm/openalex/resume.json").read_text(encoding="utf-8")
    )
    assert state["currentStratumReceived"] == 100
    assert state["sampleDraw"] == 1
    result = openalex.fetch_openalex(None, tmp_path, transport=transport)
    assert result["rows"] == 150
    final_state = json.loads(
        (tmp_path / "datasets/raw/gfm/openalex/resume.json").read_text(encoding="utf-8")
    )
    assert final_state["strata"][0]["received"] == 150
    assert final_state["formalEligible"] is False
    resolved_topics = json.loads(
        (tmp_path / "datasets/raw/gfm/openalex/resolved-topics.json").read_text(encoding="utf-8")
    )
    assert resolved_topics["formalEligible"] is False


def test_fetch_rolls_back_one_audited_page_after_resume_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    stratum_id = "graph-network-intelligence:2016:T1"
    strata = [
        {
            "stratumId": stratum_id,
            "clusterId": "graph-network-intelligence",
            "year": 2016,
            "topicId": "T1",
            "quota": 250,
        }
    ]
    monkeypatch.setattr(
        openalex,
        "_strata",
        lambda _clusters: strata,
    )
    topic_ids: dict[str, str] = {}
    seeds = {
        hashlib.sha256(f"{stratum_id}\0{draw}".encode()).hexdigest()[:16]: draw for draw in range(3)
    }

    def work(number: int) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        value.update(
            {
                "id": f"https://openalex.org/W{number}",
                "display_name": f"Work {number}",
                "publication_date": "2016-01-01",
                "publication_year": 2016,
                "type": "article",
                "authorships": [
                    {
                        "author": {"id": f"https://openalex.org/A{number}"},
                        "institutions": [],
                    }
                ],
                "topics": [],
                "referenced_works": [],
            }
        )
        return value

    initial_draws: list[int] = []

    def initial_transport(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        draw = seeds[query["seed"]]
        initial_draws.append(draw)
        draw_size = int(query["sample"])
        start = draw * 100 + 1
        return {
            "results": [work(number) for number in range(start, start + draw_size)],
            "meta": {},
        }

    real_atomic_write_json = openalex.atomic_write_json
    resume_writes = 0

    def fail_second_resume_replace(path: Path, value: dict[str, object]) -> None:
        nonlocal resume_writes
        if path.name == openalex.RESUME_NAME:
            resume_writes += 1
            if resume_writes == 2:
                raise PermissionError("synthetic resume replace failure")
        real_atomic_write_json(path, value)

    monkeypatch.setattr(openalex, "atomic_write_json", fail_second_resume_replace)
    with pytest.raises(PermissionError, match="resume replace failure"):
        openalex.fetch_openalex(None, tmp_path, transport=initial_transport)
    assert initial_draws == [0, 1]

    raw = tmp_path / "datasets/raw/gfm/openalex"
    works_path = raw / openalex.RAW_NAME
    resume_path = raw / openalex.RESUME_NAME
    interrupted = json.loads(resume_path.read_text(encoding="utf-8"))
    assert interrupted["rawRows"] == 100
    assert interrupted["acceptedRows"] == 100
    assert interrupted["requests"] == 1
    assert interrupted["sampleDraw"] == 1
    assert len(list(read_jsonl(works_path))) == 200

    monkeypatch.setattr(openalex, "atomic_write_json", real_atomic_write_json)
    resumed_draws: list[int] = []

    def resumed_transport(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            return {"results": [{"id": topic_ids[selector], "display_name": selector}]}
        draw = seeds[query["seed"]]
        resumed_draws.append(draw)
        if draw == 1:
            # Recovery runs under the fetch lock before the first resumed works
            # request; the uncommitted 100-row page is already gone here.
            assert len(list(read_jsonl(works_path))) == 100
        draw_size = int(query["sample"])
        start = draw * 100 + 1
        return {
            "results": [work(number) for number in range(start, start + draw_size)],
            "meta": {},
        }

    result = openalex.fetch_openalex(None, tmp_path, transport=resumed_transport)
    assert resumed_draws == [1, 2]
    assert result["rows"] == 250
    rows = list(read_jsonl(works_path))
    assert len(rows) == 250
    ids = [row["work"]["id"] for row in rows]
    assert len(ids) == len(set(ids))
    assert ids == [f"https://openalex.org/W{number}" for number in range(1, 251)]
    final_state = json.loads(resume_path.read_text(encoding="utf-8"))
    final_bytes = works_path.read_bytes()
    assert openalex._rollback_uncommitted_raw_tail(final_state, works_path, strata=strata) is False
    assert works_path.read_bytes() == final_bytes


def test_fetch_fails_closed_when_raw_tail_is_not_the_current_stratum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    stratum_id = "graph-network-intelligence:2016:T1"
    monkeypatch.setattr(
        openalex,
        "_strata",
        lambda _clusters: [
            {
                "stratumId": stratum_id,
                "clusterId": "graph-network-intelligence",
                "year": 2016,
                "topicId": "T1",
                "quota": 150,
            }
        ],
    )
    topic_ids: dict[str, str] = {}
    first_seed = hashlib.sha256(f"{stratum_id}\0{0}".encode()).hexdigest()[:16]

    def work(number: int) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        value.update(
            {
                "id": f"https://openalex.org/W{number}",
                "display_name": f"Work {number}",
                "publication_date": "2016-01-01",
                "publication_year": 2016,
                "type": "article",
                "authorships": [
                    {
                        "author": {"id": f"https://openalex.org/A{number}"},
                        "institutions": [],
                    }
                ],
                "topics": [],
                "referenced_works": [],
            }
        )
        return value

    def interrupted_transport(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        if query["seed"] != first_seed:
            raise RuntimeError("stop after committed page")
        return {"results": [work(number) for number in range(1, 101)], "meta": {}}

    with pytest.raises(RuntimeError, match="stop after committed page"):
        openalex.fetch_openalex(None, tmp_path, transport=interrupted_transport)
    raw = tmp_path / "datasets/raw/gfm/openalex"
    works_path = raw / openalex.RAW_NAME
    append_jsonl_fsync(
        works_path,
        [
            {
                "clusterId": "different-cluster",
                "stratumId": "different-cluster:2016:T9",
                "work": work(999),
            }
        ],
    )

    def no_work_refetch(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            return {"results": [{"id": topic_ids[selector], "display_name": selector}]}
        raise AssertionError("tail validation must happen before another works request")

    with pytest.raises(ContractViolation, match="does not belong to the current stratum"):
        openalex.fetch_openalex(None, tmp_path, transport=no_work_refetch)
    assert len(list(read_jsonl(works_path))) == 101


def test_fetch_fails_closed_on_tampered_committed_prefix_without_rewriting_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    works_path, _resume_path, _strata, topic_ids = _one_page_audited_resume(
        tmp_path, monkeypatch, quota=150
    )
    rows = list(read_jsonl(works_path))
    rows[0]["work"]["authorships"] = []
    atomic_write_jsonl(works_path, rows)
    append_jsonl_fsync(
        works_path,
        [
            {
                "clusterId": "graph-network-intelligence",
                "stratumId": "graph-network-intelligence:2016:T1",
                "work": _fetch_fixture_work(101),
            }
        ],
    )
    before = works_path.read_bytes()

    def no_work_request(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            return {"results": [{"id": topic_ids[selector], "display_name": selector}]}
        raise AssertionError("prefix audit must fail before another works request")

    with pytest.raises(ContractViolation, match="committed raw prefix"):
        openalex.fetch_openalex(None, tmp_path, transport=no_work_request)
    assert works_path.read_bytes() == before


def test_fetch_fails_closed_on_more_than_one_uncommitted_page_without_rewriting_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    works_path, _resume_path, _strata, topic_ids = _one_page_audited_resume(
        tmp_path, monkeypatch, quota=250
    )
    append_jsonl_fsync(
        works_path,
        [
            {
                "clusterId": "graph-network-intelligence",
                "stratumId": "graph-network-intelligence:2016:T1",
                "work": _fetch_fixture_work(number),
            }
            for number in range(101, 202)
        ],
    )
    before = works_path.read_bytes()

    def no_work_request(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            return {"results": [{"id": topic_ids[selector], "display_name": selector}]}
        raise AssertionError("tail bound must fail before another works request")

    with pytest.raises(ContractViolation, match="exceeds one bounded API page"):
        openalex.fetch_openalex(None, tmp_path, transport=no_work_request)
    assert works_path.read_bytes() == before


def test_fetch_resume_v11_excludes_empty_authorship_and_refills_with_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real, authorless OpenAlex work is ineligible, not a fatal API defect.

    The first interrupted pass deliberately creates the same shape of legacy
    1.1 state seen in production: four complete 100-row draws and no eligibility
    audit fields.  Resume must migrate that state without replaying those draws,
    reject the authorless work deterministically, and draw again to fill quota.
    """

    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    stratum_id = "graph-network-intelligence:2016:T1"
    monkeypatch.setattr(
        openalex,
        "_strata",
        lambda _clusters: [
            {
                "stratumId": stratum_id,
                "clusterId": "graph-network-intelligence",
                "year": 2016,
                "topicId": "T1",
                "quota": 402,
            }
        ],
    )
    topic_ids: dict[str, str] = {}

    def work(number: int, *, authorships: object = ...) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        selected_authorships = (
            [
                {
                    "author": {"id": f"https://openalex.org/A{number}"},
                    "institutions": [],
                }
            ]
            if authorships is ...
            else authorships
        )
        value.update(
            {
                "id": f"https://openalex.org/W{number}",
                "display_name": f"Work {number}",
                "publication_date": "2016-01-01",
                "publication_year": 2016,
                "type": "article",
                "authorships": selected_authorships,
                "topics": [],
                "referenced_works": [],
            }
        )
        return value

    seeds = {
        hashlib.sha256(f"{stratum_id}\0{draw}".encode()).hexdigest()[:16]: draw for draw in range(6)
    }
    first_interrupted = False

    def initial_transport(url: str, query: dict[str, str]) -> dict[str, object]:
        nonlocal first_interrupted
        if url.endswith("/topics"):
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        draw = seeds[query["seed"]]
        if draw == 4:
            first_interrupted = True
            raise RuntimeError("legacy checkpoint interruption")
        assert draw < 4
        start = draw * 100 + 1
        return {
            "results": [work(number) for number in range(start, start + 100)],
            "meta": {},
        }

    with pytest.raises(RuntimeError, match="legacy checkpoint interruption"):
        openalex.fetch_openalex(None, tmp_path, transport=initial_transport)
    assert first_interrupted is True

    raw = tmp_path / "datasets/raw/gfm/openalex"
    resume_path = raw / openalex.RESUME_NAME
    legacy = json.loads(resume_path.read_text(encoding="utf-8"))
    assert legacy["schemaVersion"] == "gfm.openalex-resume/1.1"
    assert legacy["rawRows"] == 400
    assert legacy["currentStratumReceived"] == 400
    assert legacy["sampleDraw"] == 4
    # Simulate the exact pre-eligibility-audit 1.1 checkpoint shape even when
    # the current implementation already writes the new optional fields.
    for field in (
        "workEligibilityProtocol",
        "excludedByReason",
        "excludedWorkIdDigest",
        "fetchedRows",
        "discardedAuthorshipsByReason",
        "acceptedRows",
    ):
        legacy.pop(field, None)
    atomic_write_json(resume_path, legacy)

    resumed_draws: list[int] = []

    def resumed_transport(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            return {"results": [{"id": topic_ids[selector], "display_name": selector}]}
        draw = seeds[query["seed"]]
        resumed_draws.append(draw)
        if draw == 4:
            assert query["sample"] == "2"
            return {
                "results": [
                    work(4_253_646_894, authorships=[]),
                    work(
                        401,
                        authorships=[
                            {"author": {"id": None}, "institutions": []},
                            {
                                "author": {"id": "https://openalex.org/A401"},
                                "institutions": [],
                            },
                        ],
                    ),
                ],
                "meta": {},
            }
        assert draw == 5
        assert query["sample"] == "1"
        return {"results": [work(402)], "meta": {}}

    result = openalex.fetch_openalex(None, tmp_path, transport=resumed_transport)
    assert resumed_draws == [4, 5]
    assert result["rows"] == 402
    assert result["acceptedRows"] == 402
    assert result["inspectedRows"] == 403
    assert result["excludedRows"] == 1
    assert result["excludedByReason"] == {openalex.NO_VALID_AUTHORSHIPS_REASON: 1}
    assert result["workEligibilityProtocol"] == dict(openalex.WORK_ELIGIBILITY_POLICY)
    digest = result["excludedWorkIdDigest"]
    assert isinstance(digest, str)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(character in "0123456789abcdef" for character in digest)
    assert digest == openalex._extend_exclusion_digest(
        openalex._empty_exclusion_digest(),
        reason=openalex.NO_VALID_AUTHORSHIPS_REASON,
        work_id="W4253646894",
    )

    persisted = json.loads(resume_path.read_text(encoding="utf-8"))
    assert persisted["workEligibilityProtocol"] == result["workEligibilityProtocol"]
    assert persisted["excludedByReason"] == result["excludedByReason"]
    assert persisted["excludedWorkIdDigest"] == result["excludedWorkIdDigest"]
    assert persisted["fetchedRows"] == result["inspectedRows"]
    assert persisted["complete"] is True
    assert persisted["rawRows"] == 403
    assert persisted["requests"] == 6
    rows = list(read_jsonl(raw / openalex.RAW_NAME))
    assert len(rows) == 403
    excluded = [row for row in rows if row["work"]["id"] == "https://openalex.org/W4253646894"]
    assert len(excluded) == 1
    assert excluded[0]["work"]["authorships"] == []
    assert result["discardedAuthorshipsByReason"] == {
        openalex.UNRESOLVED_AUTHOR_ID_REASON: 1,
        openalex.NULL_AUTHOR_REASON: 0,
    }
    reused = openalex.fetch_openalex(None, tmp_path, transport=resumed_transport)
    assert reused["reused"] is True
    assert reused["discardedAuthorshipsByReason"] == result["discardedAuthorshipsByReason"]


def test_work_removes_null_author_but_preserves_real_authorship() -> None:
    work = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
    work.update(
        {
            "id": "https://openalex.org/W1",
            "publication_date": "2020-01-01",
            "publication_year": 2020,
            "type": "article",
            "authorships": [
                {
                    "author": {"id": "https://openalex.org/A9999999999"},
                    "institutions": [],
                },
                {
                    "author": {"id": "https://openalex.org/A1"},
                    "institutions": [],
                },
            ],
            "topics": [],
            "referenced_works": [],
        }
    )
    validated = openalex._validate_work(work)
    assert [item["author"]["id"] for item in validated["authorships"]] == [
        "https://openalex.org/A1"
    ]

    work["authorships"] = [
        {
            "author": {"id": "https://openalex.org/A9999999999"},
            "institutions": [],
        }
    ]
    with pytest.raises(ContractViolation, match="no valid authorships"):
        openalex._validate_work(work)


def test_legacy_resume_migration_audits_unresolved_author_ids(tmp_path: Path) -> None:
    def work(number: int, authorships: list[dict[str, object]]) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        value.update(
            {
                "id": f"https://openalex.org/W{number}",
                "publication_date": "2016-01-01",
                "publication_year": 2016,
                "type": "article",
                "authorships": authorships,
                "topics": [],
                "referenced_works": [],
            }
        )
        return value

    unresolved = {"author": {"id": None}, "institutions": []}
    real = {"author": {"id": "https://openalex.org/A1"}, "institutions": []}
    path = tmp_path / "works.jsonl"
    atomic_write_jsonl(
        path,
        [
            {"clusterId": "c1", "stratumId": "c1:2016:T1", "work": work(1, [real])},
            {
                "clusterId": "c1",
                "stratumId": "c1:2016:T1",
                "work": work(2, [unresolved]),
            },
            {
                "clusterId": "c1",
                "stratumId": "c1:2016:T1",
                "work": work(3, [unresolved, real]),
            },
        ],
    )
    state: dict[str, object] = {
        "rawRows": 3,
        "currentStratumReceived": 3,
        "strata": [],
        "complete": False,
    }
    assert openalex._migrate_or_validate_work_eligibility_state(state, path) is True
    assert state["rawRows"] == 3
    assert state["acceptedRows"] == 2
    assert state["currentStratumReceived"] == 2
    assert state["fetchedRows"] == 3
    assert state["excludedByReason"] == {openalex.NO_VALID_AUTHORSHIPS_REASON: 1}
    assert state["discardedAuthorshipsByReason"] == {
        openalex.UNRESOLVED_AUTHOR_ID_REASON: 2,
        openalex.NULL_AUTHOR_REASON: 0,
    }
    retained = list(read_jsonl(path))
    assert [row["work"]["id"] for row in retained] == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W3",
    ]
    assert retained[1]["work"]["authorships"][0]["author"]["id"] is None


@pytest.mark.parametrize("authorships", [None, "not-a-list"])
def test_fetch_fails_closed_for_non_list_authorships(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, authorships: object
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    monkeypatch.setattr(
        openalex,
        "_strata",
        lambda _clusters: [
            {
                "stratumId": "graph-network-intelligence:2016:T1",
                "clusterId": "graph-network-intelligence",
                "year": 2016,
                "topicId": "T1",
                "quota": 1,
            }
        ],
    )
    topic_ids: dict[str, str] = {}

    def transport(url: str, query: dict[str, str]) -> dict[str, object]:
        if url.endswith("/topics"):
            selector = query["search"]
            topic_id = topic_ids.setdefault(selector, f"T{len(topic_ids) + 1}")
            return {"results": [{"id": topic_id, "display_name": selector}]}
        work = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        work.update(
            {
                "id": "https://openalex.org/W1",
                "display_name": "Malformed authorship fixture",
                "publication_date": "2016-01-01",
                "publication_year": 2016,
                "type": "article",
                "authorships": authorships,
                "topics": [],
                "referenced_works": [],
            }
        )
        return {"results": [work], "meta": {}}

    with pytest.raises(ContractViolation, match="authorship"):
        openalex.fetch_openalex(None, tmp_path, transport=transport)

    raw = tmp_path / "datasets/raw/gfm/openalex"
    assert not (raw / openalex.RAW_NAME).exists()
    assert not (raw / openalex.RESUME_NAME).exists()


def test_work_text_mapping_is_explicit_sparse_and_order_independent(tmp_path: Path) -> None:
    raw = tmp_path / "datasets/raw/gfm/openalex"
    raw.mkdir(parents=True)
    topics = {
        "schemaVersion": "gfm.openalex-topics/1.0",
        "configHash": "fixture",
        "formalEligible": False,
        "clusters": [
            {"clusterId": name, "maximumWorks": cap, "topics": [{"id": f"T{i + 1}"}]}
            for i, (name, cap) in enumerate(zip(("c1", "c2", "c3"), openalex.CLUSTER_CAPS))
        ],
    }

    def work(
        number: int,
        publication: str,
        title: str | None,
        *,
        topic_name: str | None = None,
        source_name: str | None = None,
    ) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        topic = (
            {"id": f"https://openalex.org/T{number + 10}", "display_name": topic_name}
            if topic_name
            else None
        )
        source = {"display_name": source_name} if source_name else None
        value.update(
            {
                "id": f"https://openalex.org/W{number}",
                "display_name": title,
                "publication_date": publication,
                "publication_year": int(publication[:4]),
                "type": "article",
                "authorships": [
                    {
                        "author": {"id": f"https://openalex.org/A{number}"},
                        "institutions": [],
                    }
                ],
                "primary_topic": topic,
                "topics": [topic] if topic else [],
                "primary_location": {"source": source} if source else None,
                "referenced_works": [],
            }
        )
        return value

    works = [
        work(40, "2019-01-01", None),
        work(30, "2018-01-01", None, topic_name="Graph AI", source_name="Graph Journal"),
        work(20, "2017-01-01", "Network Governance"),
        work(10, "2016-01-01", "Temporal Graphs"),
    ]
    atomic_write_json(raw / openalex.TOPICS_NAME, topics)
    atomic_write_jsonl(
        raw / openalex.RAW_NAME,
        [{"clusterId": "c1", "stratumId": "c1:2019:T1", "work": item} for item in works],
    )
    atomic_write_json(
        raw / openalex.RESUME_NAME,
        {
            "schemaVersion": openalex.RESUME_SCHEMA,
            "complete": True,
            "formalEligible": False,
            "rawRows": len(works),
            "strata": [{"stratumId": "fixture", "requested": len(works), "received": len(works)}],
        },
    )
    manifest = openalex.prepare_openalex(raw, tmp_path, rows_per_shard=2)
    mapping = manifest["workTextMapping"]
    assert mapping["hashAlgorithm"] == openalex.PORTABLE_ID_HASH_ALGORITHM
    assert mapping["worksShards"] == ["works-00000.npz", "works-00001.npz"]
    assert mapping["summaryOrAggregateFieldsIncluded"] is False
    assert manifest["materialization"]["numericShardRowCap"] == 2
    numeric = [item for item in manifest["shards"] if item["arrays"]]
    assert all(item["rows"] <= 2 for item in numeric)
    event_paths = [
        item["path"] for item in manifest["shards"] if item["path"].startswith("events-")
    ]
    assert len(event_paths) >= 2
    assert event_paths == [f"events-{index:05d}.npz" for index in range(len(event_paths))]

    arrays = load_domain(tmp_path, openalex.DOMAIN_ID)["arrays"]
    expected_keys = {
        "work_id_hash",
        "publication_timestamp",
        "cluster",
        "text_available",
    }
    assert expected_keys.issubset(arrays)
    assert arrays["work_id_hash"].dtype == np.uint64
    assert arrays["text_available"].tolist() == [True, True, True, False]
    assert arrays["work_id_hash"].tolist() == [
        int(portable_id_hash(f"W{number}")) for number in (10, 20, 30, 40)
    ]

    output = tmp_path / "datasets/processed/gfm/openalex-graph-ai"
    rows = list(read_jsonl(output / "text.jsonl"))
    assert len(rows) == 3
    assert [row["id"] for row in rows] == ["W30", "W20", "W10"]
    by_hash = {int(portable_id_hash(str(row["id"]))): row for row in rows}
    work_index = {int(value): index for index, value in enumerate(arrays["work_id_hash"])}
    assert {work_index[value] for value in by_hash} == {0, 1, 2}
    rich_text = str(by_hash[int(portable_id_hash("W30"))]["text"])
    assert rich_text == "Graph AI [SEP] Graph Journal"
    assert "abstract" not in rich_text and "cited_by_count" not in rich_text

    work_arrays = {name: arrays[name] for name in expected_keys}
    shuffled = list(reversed(rows))
    joined = openalex._validate_work_text_mapping(work_arrays, shuffled, cluster_count=3)
    assert joined[int(portable_id_hash("W30"))] == 2
    with pytest.raises(ContractViolation, match="duplicate"):
        openalex._validate_work_text_mapping(work_arrays, shuffled + [shuffled[0]], cluster_count=3)
    with pytest.raises(ContractViolation, match="missing"):
        openalex._validate_work_text_mapping(work_arrays, shuffled[:-1], cluster_count=3)
    unknown = dict(shuffled[0], id="W999999")
    with pytest.raises(ContractViolation, match="unknown"):
        openalex._validate_work_text_mapping(work_arrays, [unknown, *shuffled[1:]], cluster_count=3)

    leak_probe = dict(works[1], cited_by_count="FUTURE-AGGREGATE", abstract_inverted_index="SECRET")
    composed = openalex._work_text(leak_probe)
    assert composed is not None
    assert "FUTURE-AGGREGATE" not in composed and "SECRET" not in composed


def test_openalex_mega_team_keeps_hyperedge_and_omits_entire_clique(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "datasets/raw/gfm/openalex"
    raw.mkdir(parents=True)
    atomic_write_json(
        raw / openalex.TOPICS_NAME,
        {
            "schemaVersion": "gfm.openalex-topics/1.0",
            "configHash": "fixture",
            "formalEligible": False,
            "clusters": [
                {
                    "clusterId": name,
                    "maximumWorks": cap,
                    "topics": [{"id": f"T{index + 1}"}],
                }
                for index, (name, cap) in enumerate(zip(("c1", "c2", "c3"), openalex.CLUSTER_CAPS))
            ],
        },
    )
    author_count = openalex.MAX_COAUTHOR_CLIQUE_AUTHORS + 1
    work = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
    work.update(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Large collaboration",
            "publication_date": "2020-01-01",
            "publication_year": 2020,
            "type": "article",
            "authorships": [
                {
                    "author": {"id": f"https://openalex.org/A{index + 1}"},
                    "institutions": [],
                }
                for index in range(author_count)
            ],
            "topics": [],
            "referenced_works": [],
        }
    )
    atomic_write_jsonl(
        raw / openalex.RAW_NAME,
        [{"clusterId": "c1", "stratumId": "c1:2020:T1", "work": work}],
    )
    atomic_write_json(
        raw / openalex.RESUME_NAME,
        {
            "schemaVersion": openalex.RESUME_SCHEMA,
            "complete": True,
            "formalEligible": False,
            "rawRows": 1,
            "strata": [{"stratumId": "c1:2020:T1", "requested": 1, "received": 1}],
        },
    )

    manifest = openalex.prepare_openalex(raw, tmp_path, rows_per_shard=10)
    policy = manifest["coauthorExpansionPolicy"]
    assert policy["aboveThresholdPolicy"] == "omit-entire-coauthor-clique"
    assert policy["arbitraryAuthorSubsetUsed"] is False
    assert policy["megaTeamWorkCount"] == 1
    assert policy["suppressedPotentialPairEventCount"] == (author_count * (author_count - 1) // 2)
    arrays = load_domain(tmp_path, openalex.DOMAIN_ID)["arrays"]
    assert arrays["src"].size == author_count
    assert np.all(arrays["relation"] == 0)
    assert arrays["targets.first_collaboration"].size == 0
    assert len([item for item in manifest["shards"] if item["path"].startswith("events-")]) == 4


def test_openalex_prepare_failure_never_publishes_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "datasets/raw/gfm/openalex"
    raw.mkdir(parents=True)
    atomic_write_json(
        raw / openalex.TOPICS_NAME,
        {
            "schemaVersion": "gfm.openalex-topics/1.0",
            "configHash": "fixture",
            "formalEligible": False,
            "clusters": [
                {
                    "clusterId": name,
                    "maximumWorks": cap,
                    "topics": [{"id": f"T{index + 1}"}],
                }
                for index, (name, cap) in enumerate(zip(("c1", "c2", "c3"), openalex.CLUSTER_CAPS))
            ],
        },
    )
    work = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
    work.update(
        {
            "id": "https://openalex.org/W1",
            "display_name": "Atomic fixture",
            "publication_date": "2020-01-01",
            "publication_year": 2020,
            "type": "article",
            "authorships": [{"author": {"id": "https://openalex.org/A1"}, "institutions": []}],
            "topics": [],
            "referenced_works": [],
        }
    )
    atomic_write_jsonl(
        raw / openalex.RAW_NAME,
        [{"clusterId": "c1", "stratumId": "c1:2020:T1", "work": work}],
    )
    atomic_write_json(
        raw / openalex.RESUME_NAME,
        {
            "schemaVersion": openalex.RESUME_SCHEMA,
            "complete": True,
            "formalEligible": False,
            "rawRows": 1,
            "strata": [],
        },
    )
    real_write = openalex.NumericShardWriter.write
    calls = 0

    def fail_after_first(
        writer: openalex.NumericShardWriter, arrays: dict[str, np.ndarray]
    ) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic shard interruption")
        return real_write(writer, arrays)

    monkeypatch.setattr(openalex.NumericShardWriter, "write", fail_after_first)
    with pytest.raises(OSError, match="shard interruption"):
        openalex.prepare_openalex(raw, tmp_path, rows_per_shard=1)
    parent = tmp_path / "datasets/processed/gfm"
    assert not (parent / openalex.CORPUS_ID).exists()
    assert not list(parent.glob(f".{openalex.CORPUS_ID}.*.tmp"))


def test_openalex_first_dates_and_collaboration_flags_are_chronological(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "datasets/raw/gfm/openalex"
    raw.mkdir(parents=True)
    atomic_write_json(
        raw / openalex.TOPICS_NAME,
        {
            "schemaVersion": "gfm.openalex-topics/1.0",
            "configHash": "fixture",
            "formalEligible": False,
            "clusters": [
                {
                    "clusterId": name,
                    "maximumWorks": cap,
                    "topics": [{"id": f"T{index + 1}"}],
                }
                for index, (name, cap) in enumerate(zip(("c1", "c2", "c3"), openalex.CLUSTER_CAPS))
            ],
        },
    )

    def work(work_id: str, publication: str) -> dict[str, object]:
        value = {field: None for field in openalex.ALLOWED_WORK_FIELDS}
        value.update(
            {
                "id": f"https://openalex.org/{work_id}",
                "display_name": work_id,
                "publication_date": publication,
                "publication_year": int(publication[:4]),
                "type": "article",
                "authorships": [
                    {
                        "author": {"id": f"https://openalex.org/A{index}"},
                        "institutions": [],
                    }
                    for index in (1, 2)
                ],
                "topics": [],
                "referenced_works": [],
            }
        )
        return value

    # Raw and identifier order deliberately place the later collaboration first.
    raw_rows = [
        {
            "clusterId": "c1",
            "stratumId": "c1:2020:T1",
            "work": work("W1", "2020-01-01"),
        },
        {
            "clusterId": "c1",
            "stratumId": "c1:2017:T1",
            "work": work("W9", "2017-01-01"),
        },
    ]
    atomic_write_jsonl(raw / openalex.RAW_NAME, raw_rows)
    atomic_write_json(
        raw / openalex.RESUME_NAME,
        {
            "schemaVersion": openalex.RESUME_SCHEMA,
            "complete": True,
            "formalEligible": False,
            "rawRows": len(raw_rows),
            "strata": [],
        },
    )

    manifest = openalex.prepare_openalex(raw, tmp_path, rows_per_shard=1)
    assert manifest["source"]["formalEligible"] is False
    assert manifest["privacy"]["publicCheckpointEligible"] is False
    arrays = load_domain(tmp_path, openalex.DOMAIN_ID)["arrays"]
    early = int(datetime(2017, 1, 1, tzinfo=UTC).timestamp())
    late = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp())
    assert arrays["newcomers.t0"].tolist() == [early, early]
    assert arrays["targets.timestamp"].tolist() == [early, late]
    assert arrays["targets.first_collaboration"].tolist() == [True, False]
