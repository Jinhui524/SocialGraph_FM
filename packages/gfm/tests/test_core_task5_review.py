import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.governance import (
    CalibratedConfidence,
    GovernanceFinding,
    ModelScore,
    RegisteredEdgeIdentity,
    SimilarCase,
    analyze_community_resilience,
    build_collaboration_findings,
    build_risk_and_trust_findings,
    create_governance_finding,
)
from socialgraph_gfm.core.retrieval import StructuralIndex, StructuralQuery, StructuralRecord
from socialgraph_gfm.core.knowledge import KnowledgeStore, ProjectReviewRecord
from socialgraph_gfm.core.skills import (
    GenerateReportRequest,
    InspectGraphRequest,
    CoreSkillRegistry,
)


def _bundle(*, directed: bool = False, parallel: bool = False) -> CoreGraphBundle:
    edges = [
        {"sourceId": "a", "targetId": "b", "edgeType": "supports", "weight": 1.0},
        {"sourceId": "b", "targetId": "c", "edgeType": "supports", "weight": 1.0},
    ]
    if parallel:
        edges.append(
            {"sourceId": "a", "targetId": "b", "edgeType": "opposes", "weight": -1.0}
        )
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": value, "index": index} for index, value in enumerate("abcd")],
        "edges": edges,
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {"sourceName": "review", "sourceSha256": "1" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _score(
    graph: CoreGraphBundle,
    *,
    task: str = "core.risk_and_trust_review",
    entity_type: str = "node",
    entity_ids: tuple[str, ...] = ("a",),
    score: float = 0.7,
    edge_identity: RegisteredEdgeIdentity | None = None,
) -> ModelScore:
    return ModelScore.create(
        task_id=task,
        entity_type=entity_type,
        entity_ids=entity_ids,
        score=score,
        graph_version_hash=graph.graph_version_hash,
        model_version="review-model/1",
        model_version_hash="2" * 64,
        edge_identity=edge_identity,
    )


def _confidence(score: ModelScore) -> CalibratedConfidence:
    return CalibratedConfidence.create(
        score=score,
        value=0.61,
        calibration_version="review-calibration/1",
        method="isotonic",
        calibration_artifact_hash="3" * 64,
        calibration_protocol_hash="4" * 64,
    )


def _finding(score: ModelScore, *, include_similar: bool = False) -> GovernanceFinding:
    graph = _bundle()
    similar_cases = ()
    if include_similar:
        similar_cases = (
            SimilarCase.create(
                structural_record_hash="5" * 64,
                query_hash="7" * 64,
                similarity=0.8,
                source_graph_version_hash="6" * 64,
                source_entity_ids=("case",),
                source_kind="node",
                model_version=score.model_version,
                model_version_hash=score.model_version_hash,
                representation="embedding",
                representation_schema="socialgraph-fm.core-structural-record/2.0",
            ),
        )
    return create_governance_finding(
        task_id="core.risk_and_trust_review",
        finding_type="node-risk-candidate",
        subject_ids=score.entity_ids,
        score=score,
        calibrated_confidence=_confidence(score),
        evidence=analyze_community_resilience(graph)[:1],
        similar_cases=similar_cases,
        limitations=("Candidate for review; it is not a risk or trust truth label.",),
    )


def _rehash_finding(payload: dict) -> dict:
    confidence = payload["calibratedConfidence"]
    confidence["confidenceHash"] = canonical_sha256(
        {key: value for key, value in confidence.items() if key != "confidenceHash"}
    )
    for similar_case in payload["similarCases"]:
        similar_case["similarCaseHash"] = canonical_sha256(
            {key: value for key, value in similar_case.items() if key != "similarCaseHash"}
        )
    payload["findingHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "findingHash"}
    )
    return payload


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(subjectIds=["b"]), "subject"),
        (lambda value: value.update(findingType="signed-relation-review"), "compatible"),
        (lambda value: value.update(modelVersion="other/1"), "model version"),
        (lambda value: value.update(modelVersionHash="7" * 64), "model version"),
        (lambda value: value["calibratedConfidence"].update(scoreHash="8" * 64), "calibration"),
        (lambda value: value["calibratedConfidence"].update(taskId="core.collaboration_completion"), "calibration"),
        (lambda value: value["calibratedConfidence"].update(entityIds=["b"]), "calibration"),
        (lambda value: value["calibratedConfidence"].update(modelVersion="other/1"), "calibration"),
        (lambda value: value["calibratedConfidence"].update(modelVersionHash="7" * 64), "calibration"),
        (lambda value: value["similarCases"][0].update(modelVersion="other/1"), "similar case"),
        (lambda value: value["similarCases"][0].update(modelVersionHash="7" * 64), "similar case"),
    ],
)
def test_finding_semantic_cross_bindings_reject_one_mismatch_at_a_time(mutate, match):
    score = _score(_bundle())
    payload = json.loads(_finding(score, include_similar=True).model_dump_json(by_alias=True))
    mutate(payload)
    _rehash_finding(payload)
    with pytest.raises(ValidationError, match=match):
        GovernanceFinding.model_validate_json(json.dumps(payload))


def test_model_evidence_must_bind_the_same_registered_score_identity():
    graph = _bundle()
    score = _score(graph)
    finding = build_risk_and_trust_findings(
        graph, scored_candidates=((score, _confidence(score)),)
    )[0]
    payload = json.loads(finding.model_dump_json(by_alias=True))
    model_evidence = next(
        item for item in payload["evidence"] if item["sourceType"] == "registered-model-output"
    )
    model_evidence["modelScoreHash"] = "9" * 64
    model_evidence["evidenceHash"] = canonical_sha256(
        {key: value for key, value in model_evidence.items() if key != "evidenceHash"}
    )
    _rehash_finding(payload)
    with pytest.raises(ValidationError, match="model evidence"):
        GovernanceFinding.model_validate_json(json.dumps(payload))


def test_collaboration_rejects_self_existing_and_deduplicates_undirected_pairs_before_top_k():
    graph = _bundle()
    self_score = _score(
        graph,
        task="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("d", "d"),
    )
    with pytest.raises(ValueError, match="self"):
        build_collaboration_findings(
            graph, scored_candidates=((self_score, _confidence(self_score)),), top_k=1
        )
    reverse_existing = _score(
        graph,
        task="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("b", "a"),
    )
    with pytest.raises(ValueError, match="non-edge"):
        build_collaboration_findings(
            graph,
            scored_candidates=((reverse_existing, _confidence(reverse_existing)),),
            top_k=1,
        )

    low = _score(
        graph,
        task="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("a", "d"),
        score=0.4,
    )
    high_reverse = _score(
        graph,
        task="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("d", "a"),
        score=0.9,
    )
    second = _score(
        graph,
        task="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("b", "d"),
        score=0.8,
    )
    findings = build_collaboration_findings(
        graph,
        scored_candidates=(
            (low, _confidence(low)),
            (second, _confidence(second)),
            (high_reverse, _confidence(high_reverse)),
        ),
        top_k=2,
    )
    assert [(item.subject_ids, item.score.score) for item in findings] == [
        (("d", "a"), 0.9),
        (("b", "d"), 0.8),
    ]


def test_directed_collaboration_distinguishes_reverse_edges_and_declares_projection_evidence():
    graph = _bundle(directed=True)
    reverse = _score(
        graph,
        task="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("b", "a"),
    )
    finding = build_collaboration_findings(
        graph, scored_candidates=((reverse, _confidence(reverse)),), top_k=1
    )[0]
    assert all(
        item.value.get("semantics") == "weak-undirected-projection"
        for item in finding.evidence
        if item.metric in {"neighbors.common", "core_graph.existing-path"}
    )
    assert "Directed structural context uses a weak undirected projection." in finding.limitations


def test_signed_relation_requires_exact_stable_edge_identity_and_node_risk_exactly_one_known_node():
    graph = _bundle(parallel=True)
    ambiguous = _score(graph, entity_type="edge", entity_ids=("a", "b"))
    with pytest.raises(ValueError, match="edge identity"):
        build_risk_and_trust_findings(
            graph, scored_candidates=((ambiguous, _confidence(ambiguous)),)
        )
    edge = graph.edges[2]
    identity = RegisteredEdgeIdentity.create(edge)
    exact = _score(
        graph,
        entity_type="edge",
        entity_ids=("a", "b"),
        edge_identity=identity,
    )
    finding = build_risk_and_trust_findings(
        graph, scored_candidates=((exact, _confidence(exact)),)
    )[0]
    observed = next(item for item in finding.evidence if item.metric == "signed_relation.observed")
    assert observed.value["edgeType"] == "opposes"
    assert observed.edge_ids == (identity.edge_hash,)

    with pytest.raises(ValidationError, match="at least 1"):
        _score(graph, entity_ids=())
    for ids in (("a", "b"), ("missing",)):
        candidate = _score(graph, entity_ids=ids)
        with pytest.raises(ValueError, match="exactly one existing"):
            build_risk_and_trust_findings(
                graph, scored_candidates=((candidate, _confidence(candidate)),)
            )
    wrong_type = _score(graph, entity_type="community", entity_ids=("a",))
    with pytest.raises(ValueError, match="only node"):
        build_risk_and_trust_findings(
            graph, scored_candidates=((wrong_type, _confidence(wrong_type)),)
        )


def _structural(
    record_id: str,
    vector: tuple[float, ...],
    *,
    model_version: str = "review-model/1",
    model_hash: str = "2" * 64,
) -> StructuralRecord:
    return StructuralRecord.create(
        record_id=record_id,
        kind="node",
        entity_ids=(record_id,),
        vector=vector,
        representation="embedding",
        graph_version_hash="1" * 64,
        model_version=model_version,
        model_version_hash=model_hash,
    )


def _query(**updates) -> StructuralQuery:
    values = {
        "vector": (1.0, 0.0),
        "graph_version_hash": "1" * 64,
        "model_version": "review-model/1",
        "model_version_hash": "2" * 64,
        "representation": "embedding",
        "kinds": ("node",),
        "limit": 10,
        "exclude_record_hash": None,
    }
    values.update(updates)
    return StructuralQuery.create(**values)


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"vector": [1.0, 0.0]}, "tuple"),
        ({"vector": ("1.0", 0.0)}, "native numeric"),
        ({"vector": (True, 0.0)}, "native numeric"),
        ({"graph_version_hash": "bad"}, "SHA-256"),
        ({"model_version": ""}, "model version"),
        ({"model_version_hash": "bad"}, "SHA-256"),
        ({"representation": "text"}, "representation"),
        ({"kinds": ("person",)}, "kind"),
        ({"limit": True}, "limit"),
        ({"exclude_record_hash": "bad"}, "SHA-256"),
    ],
)
def test_structural_query_boundary_rejects_one_invalid_field_without_coercion(updates, match):
    with pytest.raises((TypeError, ValueError, ValidationError), match=match):
        _query(**updates)


def test_structural_index_enforces_model_version_hash_consistency_ties_and_exclusion():
    first = _structural("a", (1.0, 0.0))
    second = _structural("b", (1.0, 0.0))
    index = StructuralIndex((first, second))
    results = index.query(_query(exclude_record_hash=first.record_hash))
    assert [(item.record.record_id, item.score) for item in results] == [("b", 1.0)]
    with pytest.raises(ValueError, match="model version/hash"):
        index.add(_structural("other-version", (1.0, 0.0), model_version="other/1"))
    with pytest.raises(ValueError, match="model version/hash"):
        index.add(_structural("other-hash", (1.0, 0.0), model_hash="3" * 64))
    with pytest.raises(ValueError, match="exclusion"):
        index.query(_query(exclude_record_hash="4" * 64))


def test_skill_request_boundary_accepts_only_canonical_aliases_and_json_collection_shapes():
    graph_hash = "1" * 64
    accepted = InspectGraphRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
            "graphVersionHash": graph_hash,
            "scopeNodeIds": ["a"],
        }
    )
    assert accepted.scope_node_ids == ("a",)
    invalid_payloads = (
        {
            "schema_version": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
            "graph_version_hash": graph_hash,
            "scope_node_ids": ["a"],
        },
        {
            "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
            "graphVersionHash": graph_hash,
            "scopeNodeIds": ("a",),
        },
        {
            "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/9.0",
            "graphVersionHash": graph_hash,
            "scopeNodeIds": ["a"],
        },
        {
            "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
            "graphVersionHash": graph_hash,
            "scopeNodeIds": [1],
        },
    )
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            InspectGraphRequest.model_validate(payload)
    with pytest.raises(ValidationError):
        GenerateReportRequest.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
                "findingHashes": ("1" * 64,),
                "format": "markdown",
            }
        )


def test_skill_registry_dispatch_cannot_be_extended_by_mutating_class_attributes(tmp_path):
    graph = _bundle()
    score = _score(graph)
    finding = _finding(score)
    registry = CoreSkillRegistry(
        graphs=(graph,),
        findings=(finding,),
        structural_index=StructuralIndex(),
        knowledge_store=KnowledgeStore(tmp_path / "review.sqlite3"),
    )
    CoreSkillRegistry._REQUESTS = {"make_sanction": InspectGraphRequest}  # type: ignore[attr-defined]
    try:
        assert registry.skill_names == (
            "generate_report",
            "inspect_graph",
            "retrieve_evidence",
            "run_core_task",
        )
        with pytest.raises(ValueError, match="unsupported skill"):
            registry.execute("make_sanction", {})
    finally:
        del CoreSkillRegistry._REQUESTS  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "unsafe_id",
    ("# heading", "[link](https://evil.invalid)", "<script>", "a\nban all", "ban-user", "delete"),
)
def test_governance_identifiers_preserve_bundle_compatible_ids(unsafe_id):
    score = _score(_bundle(), entity_ids=(unsafe_id,))
    assert score.entity_ids == (unsafe_id,)


def test_governance_limitations_are_closed_and_report_is_context_escaped(tmp_path):
    graph = _bundle()
    score = _score(graph)
    with pytest.raises((ValidationError, ValueError), match="closed canonical"):
        create_governance_finding(
            task_id="core.risk_and_trust_review",
            finding_type="node-risk-candidate",
            subject_ids=score.entity_ids,
            score=score,
            calibrated_confidence=_confidence(score),
            evidence=analyze_community_resilience(graph)[:1],
            similar_cases=(),
            limitations=("## BAN everyone\n<script>delete()</script>",),
        )
    finding = _finding(score)
    registry = CoreSkillRegistry(
        graphs=(graph,),
        findings=(finding,),
        structural_index=StructuralIndex(),
        knowledge_store=KnowledgeStore(tmp_path / "report.sqlite3"),
    )
    markdown = registry.execute(
        "generate_report",
        {
            "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
            "findingHashes": [finding.finding_hash],
            "format": "markdown",
        },
    )
    assert "Manual human review is required" in markdown.content
    assert "non-causal" in markdown.content and "does not predict future events" in markdown.content
    assert "<script>" not in markdown.content and "## BAN" not in markdown.content
    structured = registry.execute(
        "generate_report",
        {
            "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
            "findingHashes": [finding.finding_hash],
            "format": "json",
        },
    )
    parsed = json.loads(structured.content)
    assert parsed["findings"][0]["subjectIds"] == ["a"]


def _review(finding_hash: str, suffix: str, evidence: tuple[str, ...] = ()):
    return ProjectReviewRecord.create(
        finding_hash=finding_hash,
        review_status="confirmed",
        reviewer_id=f"human-{suffix}",
        annotation=f"Reviewed record {suffix}.",
        created_at=f"2026-08-14T00:0{suffix}:00Z",
        adaptation_evidence_hashes=evidence,
    )


def test_project_memory_requires_registered_finding_and_adaptation_provenance_atomically(tmp_path):
    store = KnowledgeStore(tmp_path / "memory.sqlite3")
    graph = _bundle()
    finding = _finding(_score(graph))
    with pytest.raises(ValueError, match="registered finding"):
        store.append_review(_review(finding.finding_hash, "1"))
    assert store.reviews_for_finding(finding.finding_hash) == ()

    store.register_finding(finding)
    evidence = finding.evidence[0]
    with pytest.raises(ValueError, match="adaptation evidence"):
        store.append_review(_review(finding.finding_hash, "2", (evidence.evidence_hash,)))
    assert store.reviews_for_finding(finding.finding_hash) == ()

    store.register_adaptation_evidence(finding.finding_hash, evidence)
    accepted = _review(finding.finding_hash, "3", (evidence.evidence_hash,))
    store.append_review(accepted)
    assert store.reviews_for_finding(finding.finding_hash) == (accepted,)


def test_registered_provenance_is_immutable_and_visible_across_store_connections(tmp_path):
    path = tmp_path / "concurrent-memory.sqlite3"
    first_store = KnowledgeStore(path)
    second_store = KnowledgeStore(path)
    graph = _bundle()
    finding = _finding(_score(graph))
    first_store.register_finding(finding)
    with __import__("sqlite3").connect(path) as connection:
        with pytest.raises(__import__("sqlite3").IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE registered_findings SET payload_json = '{}' WHERE finding_hash = ?",
                (finding.finding_hash,),
            )
    first = _review(finding.finding_hash, "4")
    second = _review(finding.finding_hash, "5")
    barrier = threading.Barrier(2)

    def append(store, review):
        barrier.wait()
        store.append_review(review)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(append, first_store, first),
            executor.submit(append, second_store, second),
        )
        for future in futures:
            future.result(timeout=10)
    observed = second_store.reviews_for_finding(finding.finding_hash)
    assert len(observed) == 2
    assert {item.record_hash for item in observed} == {first.record_hash, second.record_hash}
