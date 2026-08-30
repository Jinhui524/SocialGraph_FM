import json
import sqlite3

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.governance import (
    CalibratedConfidence,
    GovernanceFinding,
    ModelScore,
    SimilarCase,
    analyze_community_resilience,
    build_collaboration_findings,
    build_risk_and_trust_findings,
    create_governance_finding,
)
from socialgraph_gfm.core.knowledge import KnowledgeStore, SCHEMA_VERSION
from socialgraph_gfm.core.retrieval import StructuralIndex, StructuralQuery, StructuralRecord
from socialgraph_gfm.core.skills import CoreSkillRegistry


def _bundle(node_ids=("a", "b", "c", "d"), *, directed=False):
    ordered = sorted(node_ids)
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": value, "index": index} for index, value in enumerate(ordered)],
        "edges": [],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {"sourceName": "round-2", "sourceSha256": "1" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _score(graph, ids=("a",), *, task="core.risk_and_trust_review", value=0.7):
    entity_type = "node" if task == "core.risk_and_trust_review" else "node-pair"
    return ModelScore.create(
        task_id=task,
        entity_type=entity_type,
        entity_ids=ids,
        score=value,
        graph_version_hash=graph.graph_version_hash,
        model_version="round-2-model/1",
        model_version_hash="2" * 64,
    )


def _confidence(score, value=0.6):
    return CalibratedConfidence.create(
        score=score,
        value=value,
        calibration_version="round-2-calibration/1",
        method="isotonic",
        calibration_artifact_hash="3" * 64,
        calibration_protocol_hash="4" * 64,
    )


def _rehash_evidence(value):
    value["evidenceHash"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "evidenceHash"}
    )


def _rehash_finding(value):
    value["findingHash"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "findingHash"}
    )


def test_model_evidence_is_an_exact_score_reference_without_duplicate_numeric_value():
    graph = _bundle()
    score = _score(graph)
    finding = build_risk_and_trust_findings(
        graph, scored_candidates=((score, _confidence(score)),)
    )[0]
    model_evidence = next(
        item for item in finding.evidence if item.source_type == "registered-model-output"
    )
    assert model_evidence.metric == "registered_model.score-reference"
    assert model_evidence.model_score_hash == score.score_hash
    assert model_evidence.value == {}

    payload = json.loads(finding.model_dump_json(by_alias=True))
    evidence_payload = next(
        item for item in payload["evidence"] if item["sourceType"] == "registered-model-output"
    )
    evidence_payload["metric"] = "registered_model.score"
    evidence_payload["valueCanonicalJson"] = json.dumps(
        {"score": score.score}, sort_keys=True, separators=(",", ":")
    )
    _rehash_evidence(evidence_payload)
    _rehash_finding(payload)
    with pytest.raises(ValidationError, match="exact score reference"):
        GovernanceFinding.model_validate_json(json.dumps(payload))


def _retrieval_provenance():
    record = StructuralRecord.create(
        record_id="case-a",
        kind="ego",
        entity_ids=("a", "b"),
        vector=(1.0, 0.0),
        representation="embedding",
        graph_version_hash="5" * 64,
        model_version="round-2-model/1",
        model_version_hash="2" * 64,
    )
    index = StructuralIndex((record,))
    query = StructuralQuery.create(
        vector=(1.0, 0.0),
        graph_version_hash=record.graph_version_hash,
        model_version=record.model_version,
        model_version_hash=record.model_version_hash,
        representation=record.representation,
        kinds=("ego",),
        limit=1,
        exclude_record_hash=None,
    )
    result = index.query(query)[0]
    return index, query, result


def test_similar_case_factory_and_registries_resolve_exact_query_result_provenance(tmp_path):
    index, query, result = _retrieval_provenance()
    similar = SimilarCase.from_retrieval_result(query=query, result=result)
    assert similar.structural_record_hash == result.record.record_hash
    assert similar.query_hash == query.query_hash
    assert similar.source_entity_ids == result.record.entity_ids
    assert similar.representation == result.record.representation

    graph = _bundle()
    score = _score(graph)
    finding = create_governance_finding(
        task_id="core.risk_and_trust_review",
        finding_type="node-risk-candidate",
        subject_ids=score.entity_ids,
        score=score,
        calibrated_confidence=_confidence(score),
        evidence=analyze_community_resilience(graph)[:1],
        similar_cases=(similar,),
        limitations=("Candidate for review; it is not a risk or trust truth label.",),
    )
    CoreSkillRegistry(
        graphs=(graph,),
        findings=(finding,),
        structural_index=index,
        knowledge_store=KnowledgeStore(tmp_path / "similar.sqlite3"),
    )
    store = KnowledgeStore(tmp_path / "finding.sqlite3")
    store.register_finding(finding, structural_index=index)

    unknown_payload = json.loads(finding.model_dump_json(by_alias=True))
    unknown_case = unknown_payload["similarCases"][0]
    unknown_case["queryHash"] = "9" * 64
    unknown_case["similarCaseHash"] = canonical_sha256(
        {key: value for key, value in unknown_case.items() if key != "similarCaseHash"}
    )
    _rehash_finding(unknown_payload)
    syntactically_valid = GovernanceFinding.model_validate_json(json.dumps(unknown_payload))
    with pytest.raises(ValueError, match="registered structural query/result"):
        CoreSkillRegistry(
            graphs=(graph,),
            findings=(syntactically_valid,),
            structural_index=index,
            knowledge_store=KnowledgeStore(tmp_path / "unknown.sqlite3"),
        )


def test_collaboration_dedup_keeps_winning_score_and_its_own_confidence_atomically():
    graph = _bundle()
    low = _score(
        graph,
        ("a", "b"),
        task="core.collaboration_completion",
        value=0.2,
    )
    high = _score(
        graph,
        ("b", "a"),
        task="core.collaboration_completion",
        value=0.9,
    )
    low_confidence = _confidence(low, 0.1)
    high_confidence = _confidence(high, 0.8)
    finding = build_collaboration_findings(
        graph,
        scored_candidates=((low, low_confidence), (high, high_confidence)),
        top_k=1,
    )[0]
    assert finding.score.score_hash == high.score_hash
    assert finding.calibrated_confidence.confidence_hash == high_confidence.confidence_hash
    assert finding.calibrated_confidence.score_hash == finding.score.score_hash

    with pytest.raises(ValueError, match="confidence.*score"):
        build_collaboration_findings(
            graph,
            scored_candidates=((high, low_confidence),),
            top_k=1,
        )


def test_sqlite_schema_version_and_exact_layout_fail_closed(tmp_path):
    current = tmp_path / "current.sqlite3"
    KnowledgeStore(current)
    KnowledgeStore(current)
    assert SCHEMA_VERSION == "socialgraph-fm.core-knowledge-sqlite/2.2"

    old = tmp_path / "old.sqlite3"
    with sqlite3.connect(old) as connection:
        connection.execute(
            "CREATE TABLE schema_metadata(singleton INTEGER PRIMARY KEY, schema_version TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO schema_metadata(singleton, schema_version) VALUES (1, ?)",
            ("socialgraph-fm.core-knowledge-sqlite/2.0",),
        )
        connection.execute("CREATE TABLE project_memory(sequence INTEGER PRIMARY KEY)")
    with pytest.raises(ValueError, match="unsupported knowledge SQLite schema version"):
        KnowledgeStore(old)

    partial = tmp_path / "partial.sqlite3"
    with sqlite3.connect(partial) as connection:
        connection.execute(
            """
            CREATE TABLE schema_metadata(
                singleton INTEGER PRIMARY KEY,
                schema_version TEXT NOT NULL,
                schema_fingerprint TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO schema_metadata VALUES (1, ?, ?)",
            ("socialgraph-fm.core-knowledge-sqlite/2.2", "f" * 64),
        )
    with pytest.raises(ValueError, match="layout|fingerprint"):
        KnowledgeStore(partial)


def test_sqlite_current_version_missing_fk_or_trigger_fails_before_use(tmp_path):
    path = tmp_path / "forged.sqlite3"
    KnowledgeStore(path)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TRIGGER project_memory_no_delete")
    with pytest.raises(ValueError, match="layout"):
        KnowledgeStore(path)

    no_op_trigger = tmp_path / "forged-no-op-trigger.sqlite3"
    KnowledgeStore(no_op_trigger)
    with sqlite3.connect(no_op_trigger) as connection:
        connection.execute("DROP TRIGGER project_memory_no_delete")
        connection.execute(
            """
            CREATE TRIGGER project_memory_no_delete
            BEFORE DELETE ON project_memory
            BEGIN SELECT 1; END
            """
        )
    with pytest.raises(ValueError, match="layout"):
        KnowledgeStore(no_op_trigger)


def test_full_bundle_id_domain_round_trips_and_markdown_renders_only_escaped_data(tmp_path):
    ids = (
        "普通用户",
        "`code`",
        "line\nban all",
        "<script>alert(1)</script>",
        "[remove](https://evil.invalid)",
        "ban",
    )
    graph = _bundle(ids)
    findings = []
    for identifier in ids:
        score = _score(graph, (identifier,))
        findings.append(
            create_governance_finding(
                task_id="core.risk_and_trust_review",
                finding_type="node-risk-candidate",
                subject_ids=(identifier,),
                score=score,
                calibrated_confidence=_confidence(score),
                evidence=analyze_community_resilience(graph)[:1],
                similar_cases=(),
                limitations=("Candidate for review; it is not a risk or trust truth label.",),
            )
        )
    registry = CoreSkillRegistry(
        graphs=(graph,),
        findings=tuple(findings),
        structural_index=StructuralIndex(),
        knowledge_store=KnowledgeStore(tmp_path / "ids.sqlite3"),
    )
    hashes = [item.finding_hash for item in findings]
    markdown = registry.execute(
        "generate_report",
        {
            "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
            "findingHashes": hashes,
            "format": "markdown",
        },
    ).content
    assert "<script>" not in markdown
    assert "[remove](https://evil.invalid)" not in markdown
    assert "line\nban all" not in markdown
    assert "\\u0060code\\u0060" in markdown
    assert "\\u003cscript\\u003e" in markdown
    assert '`"ban"`' in markdown

    structured = registry.execute(
        "generate_report",
        {
            "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
            "findingHashes": hashes,
            "format": "json",
        },
    ).content
    assert [item["subjectIds"][0] for item in json.loads(structured)["findings"]] == list(ids)
