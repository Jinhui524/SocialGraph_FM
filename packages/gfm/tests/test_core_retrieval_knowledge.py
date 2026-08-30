import sqlite3

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_json
from socialgraph_gfm.core.knowledge import (
    KnowledgeDocument,
    KnowledgeStore,
    ProjectReviewRecord,
)
from socialgraph_gfm.core.retrieval import StructuralIndex, StructuralQuery, StructuralRecord


def _record(identifier: str, vector: tuple[float, ...], *, graph: str = "a" * 64):
    return StructuralRecord.create(
        record_id=identifier,
        kind="node",
        entity_ids=(identifier,),
        vector=vector,
        representation="embedding",
        graph_version_hash=graph,
        model_version="core-gfm/2",
        model_version_hash="b" * 64,
    )


def _query(vector=(1.0, 0.0), *, exclude=None):
    return StructuralQuery.create(
        vector=vector,
        graph_version_hash="a" * 64,
        model_version="core-gfm/2",
        model_version_hash="b" * 64,
        representation="embedding",
        kinds=("node", "ego", "community"),
        limit=3,
        exclude_record_hash=exclude,
    )


def test_structural_index_cosine_ranking_ties_and_version_filters():
    index = StructuralIndex()
    for record in (
        _record("z", (1.0, 0.0)),
        _record("a", (1.0, 0.0)),
        _record("middle", (1.0, 1.0)),
        _record("other-graph", (1.0, 0.0), graph="c" * 64),
    ):
        index.add(record)
    results = index.query(_query())
    assert [(item.record.record_id, item.score) for item in results] == [
        ("a", 1.0),
        ("z", 1.0),
        ("middle", pytest.approx(2**-0.5)),
    ]
    assert all(item.query_provenance_hash for item in results)


def test_structural_records_are_hash_bound_and_reject_zero_or_mixed_dimensions():
    valid = _record("a", (1.0, 0.0))
    with pytest.raises(ValidationError, match="recordHash"):
        StructuralRecord.model_validate(
            {**valid.model_dump(mode="python", by_alias=True), "vector": (0.0, 1.0)}
        )
    index = StructuralIndex((valid,))
    with pytest.raises(ValueError, match="dimension"):
        index.add(_record("wide", (1.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="non-zero"):
        _query((0.0, 0.0))
    with pytest.raises(ValueError, match="SHA-256"):
        StructuralQuery.create(
            vector=(1.0, 0.0),
            graph_version_hash="not-a-hash",
            model_version="core-gfm/2",
            model_version_hash="b" * 64,
            representation="embedding",
            kinds=("node",),
            limit=3,
            exclude_record_hash=None,
        )


def test_knowledge_fts5_bm25_and_sql_injection_safe_query(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    store = KnowledgeStore(path)
    store.add_document(
        KnowledgeDocument.create(
            category="governance-rule",
            title="Bridge review",
            body="Bridge nodes require manual review and contextual evidence.",
            source_uri="urn:rule:bridge",
        )
    )
    store.add_document(
        KnowledgeDocument.create(
            category="limitation",
            title="Prediction limitation",
            body="Scores are non-causal and do not predict future events.",
            source_uri="urn:limit:future",
        )
    )
    results = store.search("manual bridge", limit=5)
    assert results[0].document.title == "Bridge review"
    assert isinstance(results[0].bm25_score, float)
    assert store.search("' OR 1=1 --", limit=5) == ()

    with sqlite3.connect(path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'trigger')"
            )
        }
    assert {"schema_metadata", "knowledge_documents", "knowledge_fts", "project_memory"} <= tables


def test_knowledge_reload_fails_closed_if_stored_payload_is_tampered(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    store = KnowledgeStore(path)
    document = KnowledgeDocument.create(
        category="model-card", title="Static model", body="Validated card.", source_uri=None
    )
    store.add_document(document)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE knowledge_documents SET payload_json = ? WHERE document_hash = ?",
            ('{"forged":true}', document.document_hash),
        )
    with pytest.raises(ValueError, match="invalid stored knowledge"):
        store.get_document(document.document_hash)


def test_knowledge_search_fails_closed_if_fts_index_is_tampered(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    store = KnowledgeStore(path)
    document = KnowledgeDocument.create(
        category="governance-rule",
        title="Manual review rule",
        body="Candidates require contextual review.",
        source_uri="urn:rule:manual",
    )
    store.add_document(document)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE knowledge_fts SET body = ? WHERE document_hash = ?",
            ("forged manual review content", document.document_hash),
        )
    with pytest.raises(ValueError, match="invalid FTS index"):
        store.search("manual review")


def test_project_memory_is_append_only_and_separate_from_findings(tmp_path):
    path = tmp_path / "knowledge.sqlite3"
    store = KnowledgeStore(path)
    finding_hash = "d" * 64
    first = ProjectReviewRecord.create(
        finding_hash=finding_hash,
        review_status="confirmed",
        reviewer_id="human-1",
        annotation="Confirmed after checking source records.",
        created_at="2026-08-14T00:00:00Z",
    )
    second = ProjectReviewRecord.create(
        finding_hash=finding_hash,
        review_status="rejected",
        reviewer_id="human-2",
        annotation="Rejected after independent review.",
        created_at="2026-08-14T00:01:00Z",
    )
    with pytest.raises(ValueError, match="registered finding"):
        store.append_review(first)
    with sqlite3.connect(path) as connection:
        for review in (first, second):
            connection.execute(
                """
                INSERT INTO project_memory(
                    record_hash, finding_hash, review_status, reviewer_id, created_at, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    review.record_hash,
                    review.finding_hash,
                    review.review_status,
                    review.reviewer_id,
                    review.created_at,
                    canonical_json(review),
                ),
            )
    before = canonical_json(first)
    with pytest.raises(ValidationError, match="frozen"):
        first.adaptation_evidence_hashes = ("e" * 64,)  # type: ignore[misc]
    assert canonical_json(first) == before
    assert store.reviews_for_finding(finding_hash) == (first, second)
    with sqlite3.connect(path) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE project_memory SET reviewer_id = 'attacker'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM project_memory")


def test_review_and_document_models_reject_extra_fields():
    with pytest.raises(ValidationError):
        ProjectReviewRecord.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-project-review/2.0",
                "findingHash": "d" * 64,
                "reviewStatus": "confirmed",
                "reviewerId": "human",
                "annotation": "ok",
                "createdAt": "2026-08-14T00:00:00Z",
                "recordHash": "e" * 64,
                "mutateFinding": True,
            }
        )
