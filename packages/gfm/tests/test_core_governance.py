import json

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.governance import (
    CalibratedConfidence,
    GovernanceFinding,
    ModelScore,
    RegressionConfidenceInterval,
    RegisteredEdgeIdentity,
    SimilarCase,
    analyze_community_resilience,
    build_collaboration_findings,
    build_community_resilience_findings,
    build_risk_and_trust_findings,
    create_governance_finding,
    load_governance_finding_json,
)


def _bundle(*, directed: bool = False) -> CoreGraphBundle:
    edges = [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
        ("c", "d"),
    ]
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": directed,
        "nodes": [{"id": value, "index": index} for index, value in enumerate("abcd")],
        "edges": [
            {
                "sourceId": source,
                "targetId": target,
                "edgeType": "supports" if (source, target) != ("c", "d") else "opposes",
                "weight": 1.0,
            }
            for source, target in edges
        ],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {"sourceName": "hand-fixture", "sourceSha256": "1" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _score(entity_type: str, entity_ids: tuple[str, ...], score: float) -> ModelScore:
    edge_identity = None
    if entity_type == "edge":
        edge = next(
            item for item in _bundle().edges if (item.source_id, item.target_id) == entity_ids
        )
        edge_identity = RegisteredEdgeIdentity.create(edge)
    return ModelScore.create(
        task_id="core.risk_and_trust_review",
        entity_type=entity_type,
        entity_ids=entity_ids,
        score=score,
        graph_version_hash=_bundle().graph_version_hash,
        model_version="risk-model/2",
        model_version_hash="2" * 64,
        edge_identity=edge_identity,
    )


def _confidence(score: ModelScore, value: float = 0.8) -> CalibratedConfidence:
    return CalibratedConfidence.create(
        score=score,
        value=value,
        calibration_version="risk-calibration/4",
        method="isotonic",
        calibration_artifact_hash="3" * 64,
        calibration_protocol_hash="4" * 64,
    )


def _regression_interval(score: ModelScore) -> RegressionConfidenceInterval:
    return RegressionConfidenceInterval.create(
        score=score,
        lower_bound=0.42,
        upper_bound=0.71,
        coverage=0.9,
        validation_count=40,
        confidence_version="resilience-residuals/1",
        method="validation-residual-interval",
        confidence_artifact_hash="8" * 64,
        confidence_protocol_hash="9" * 64,
    )


def test_resilience_fixture_has_hand_derived_undirected_evidence():
    evidence = analyze_community_resilience(
        _bundle(),
        community_by_node={"a": "team-1", "b": "team-1", "c": "team-1", "d": "team-2"},
    )
    by_metric = {item.metric: item for item in evidence}

    assert by_metric["connectivity.components"].value == {"count": 1, "sizes": [4]}
    assert by_metric["connectivity.articulation_points"].node_ids == ("c",)
    assert by_metric["connectivity.bridges"].edge_ids == (
        RegisteredEdgeIdentity.create(_bundle().edges[-1]).edge_hash,
    )
    assert by_metric["k_core.node_core_numbers"].value == {
        "a": 2,
        "b": 2,
        "c": 2,
        "d": 1,
    }
    assert by_metric["community.concentration"].value == {
        "counts": {"team-1": 3, "team-2": 1},
        "herfindahl": 0.625,
    }
    stress = by_metric["stress.node_removal"].value["profiles"]
    assert stress["c"] == {"componentCount": 2, "largestComponentFraction": 2 / 3}
    assert all(item.graph_version_hash == _bundle().graph_version_hash for item in evidence)
    assert all(item.algorithm_config_hash for item in evidence)


def test_directed_resilience_declares_weak_projection_semantics():
    evidence = analyze_community_resilience(_bundle(directed=True))
    by_metric = {item.metric: item for item in evidence}
    assert by_metric["connectivity.components"].value["semantics"] == "weak-undirected-projection"
    assert by_metric["connectivity.bridges"].limitations == (
        "Directed edges are analyzed on a weak undirected projection.",
    )


def test_parallel_semantic_edges_are_not_misreported_as_bridges():
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "a", "index": 0}, {"id": "b", "index": 1}],
        "edges": [
            {"sourceId": "a", "targetId": "b", "edgeType": "supports", "weight": 1.0},
            {"sourceId": "a", "targetId": "b", "edgeType": "advises", "weight": 1.0},
        ],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {"sourceName": "parallel", "sourceSha256": "1" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    evidence = analyze_community_resilience(CoreGraphBundle.model_validate(payload))
    bridges = next(item for item in evidence if item.metric == "connectivity.bridges")
    assert bridges.value["count"] == 0
    assert bridges.edge_ids == ()


def test_finding_is_canonical_hash_bound_immutable_and_tamper_rejected():
    graph = _bundle()
    score = _score("node", ("c",), 0.91)
    evidence = analyze_community_resilience(graph)[:2]
    finding = create_governance_finding(
        task_id="core.risk_and_trust_review",
        finding_type="node-risk-candidate",
        subject_ids=("c",),
        score=score,
        calibrated_confidence=_confidence(score),
        evidence=evidence,
        similar_cases=(),
        limitations=("Candidate for review; it is not a risk or trust truth label.",),
    )
    reloaded = load_governance_finding_json(finding.model_dump_json(by_alias=True))
    assert reloaded == finding
    assert finding.review_status == "pending-human-review"
    assert (
        "Manual human review is required; no automatic sanction or action is authorized."
        in finding.limitations
    )
    assert "This finding is non-causal and does not predict future events." in finding.limitations

    tampered = json.loads(finding.model_dump_json(by_alias=True))
    tampered["score"]["score"] = 0.1
    with pytest.raises(ValidationError, match="scoreHash|findingHash"):
        GovernanceFinding.model_validate_json(json.dumps(tampered))
    with pytest.raises(ValidationError):
        GovernanceFinding.model_validate_json(json.dumps({**tampered, "sanction": "ban"}))


def test_finding_hash_bound_nested_payload_is_deeply_immutable():
    graph = _bundle()
    score = _score("node", ("c",), 0.91)
    evidence = analyze_community_resilience(graph)
    similar_case = SimilarCase.create(
        structural_record_hash="6" * 64,
        query_hash="7" * 64,
        similarity=0.75,
        source_graph_version_hash=graph.graph_version_hash,
        source_entity_ids=("c",),
        source_kind="node",
        model_version=score.model_version,
        model_version_hash=score.model_version_hash,
        representation="embedding",
        representation_schema="socialgraph-fm.core-structural-record/2.0",
    )
    finding = create_governance_finding(
        task_id="core.risk_and_trust_review",
        finding_type="node-risk-candidate",
        subject_ids=("c",),
        score=score,
        calibrated_confidence=_confidence(score),
        evidence=evidence,
        similar_cases=(similar_case,),
        limitations=("Candidate for review; it is not a risk or trust truth label.",),
    )
    before = canonical_json(finding)
    detached_value = finding.evidence[-1].value
    detached_value["profiles"]["c"]["componentCount"] = 999
    assert canonical_json(finding) == before
    assert finding.evidence[-1].value["profiles"]["c"]["componentCount"] == 2
    with pytest.raises(ValidationError, match="frozen"):
        finding.score.model_version = "attacker"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        finding.calibrated_confidence.method = "raw-score"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        finding.similar_cases[0].similarity = 0.0  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        finding.evidence[0].limitations = ("changed",)  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        finding.limitations = ("changed",)  # type: ignore[misc]
    reloaded = load_governance_finding_json(finding.model_dump_json(by_alias=True))
    assert canonical_json(reloaded) == before


def test_score_and_calibration_are_version_bound_not_renamed_raw_values():
    graph = _bundle()
    with pytest.raises(ValidationError):
        ModelScore.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-model-score/2.0",
                "taskId": "core.risk_and_trust_review",
                "entityType": "node",
                "entityIds": ["a"],
                "score": 0.4,
                "graphVersionHash": graph.graph_version_hash,
            }
        )
    with pytest.raises(ValidationError):
        CalibratedConfidence.model_validate({"value": 0.4})


def test_risk_signed_relation_and_collaboration_findings_only_use_supplied_scores():
    graph = _bundle()
    node_score = _score("node", ("c",), 0.91)
    edge_score = _score("edge", ("c", "d"), 0.74)
    risk = build_risk_and_trust_findings(
        graph,
        scored_candidates=(
            (node_score, _confidence(node_score)),
            (edge_score, _confidence(edge_score, 0.7)),
        ),
    )
    assert [item.finding_type for item in risk] == [
        "node-risk-candidate",
        "signed-relation-review",
    ]
    assert risk[1].score.score == 0.74
    assert any(item.metric == "signed_relation.observed" for item in risk[1].evidence)

    collab_score = ModelScore.create(
        task_id="core.collaboration_completion",
        entity_type="node-pair",
        entity_ids=("a", "d"),
        score=0.66,
        graph_version_hash=graph.graph_version_hash,
        model_version="collab/1",
        model_version_hash="4" * 64,
    )
    findings = build_collaboration_findings(
        graph,
        scored_candidates=((collab_score, _confidence(collab_score, 0.61)),),
        top_k=1,
    )
    assert findings[0].finding_type == "core-collaboration-completion"
    assert findings[0].score.score == 0.66
    path = next(
        item
        for item in findings[0].evidence
        if item.metric == "core_graph.existing-path"
    )
    assert path.value == {
        "distance": 2,
        "path": ["a", "c", "d"],
        "semantics": "undirected",
    }
    assert findings[0].finding_hash == canonical_sha256(
        findings[0].model_dump(mode="python", by_alias=True, exclude={"finding_hash"})
    )


def test_community_resilience_finding_combines_supplied_score_with_factual_evidence():
    graph = _bundle()
    score = ModelScore.create(
        task_id="core.community_resilience_review",
        entity_type="community",
        entity_ids=("a", "b", "c", "d"),
        score=0.58,
        graph_version_hash=graph.graph_version_hash,
        model_version="resilience/1",
        model_version_hash="5" * 64,
    )
    findings = build_community_resilience_findings(
        graph,
        scored_candidates=((score, _regression_interval(score)),),
        community_by_node={"a": "x", "b": "x", "c": "x", "d": "y"},
    )
    assert len(findings) == 1
    assert findings[0].finding_type == "community-resilience-candidate"
    assert findings[0].score.score == 0.58
    assert {item.metric for item in findings[0].evidence} >= {
        "registered_model.score-reference",
        "connectivity.components",
        "stress.node_removal",
    }
    assert "healthy" not in findings[0].model_dump_json().lower()
    assert findings[0].calibrated_confidence.coverage == 0.9
    assert "probability" in findings[0].limitations[-3].lower()


def test_regression_interval_rejects_coherently_rehashed_point_estimate_contradiction():
    graph = _bundle()
    score = ModelScore.create(
        task_id="core.community_resilience_review",
        entity_type="community",
        entity_ids=("a", "b", "c", "d"),
        score=0.58,
        graph_version_hash=graph.graph_version_hash,
        model_version="resilience/1",
        model_version_hash="5" * 64,
    )
    payload = _regression_interval(score).model_dump(mode="python", by_alias=True)
    payload["pointEstimate"] = 0.61
    payload["confidenceHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "confidenceHash"}
    )

    with pytest.raises(ValidationError, match="pointEstimate.*scoreHash"):
        RegressionConfidenceInterval.model_validate(payload)


def test_resilience_builder_rejects_bypassed_interval_point_estimate_contradiction():
    graph = _bundle()
    score = ModelScore.create(
        task_id="core.community_resilience_review",
        entity_type="community",
        entity_ids=("a", "b", "c", "d"),
        score=0.58,
        graph_version_hash=graph.graph_version_hash,
        model_version="resilience/1",
        model_version_hash="5" * 64,
    )
    interval = _regression_interval(score)
    payload = interval.model_dump(mode="python", by_alias=True)
    payload["pointEstimate"] = 0.61
    payload["confidenceHash"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "confidenceHash"}
    )
    bypassed = interval.model_copy(
        update={
            "point_estimate": payload["pointEstimate"],
            "confidence_hash": payload["confidenceHash"],
        }
    )

    with pytest.raises(ValueError, match="point estimate.*model score"):
        build_community_resilience_findings(
            graph,
            scored_candidates=((score, bypassed),),
            community_by_node={"a": "x", "b": "x", "c": "x", "d": "y"},
        )


def test_candidate_builders_fail_closed_on_wrong_graph_or_task_binding():
    graph = _bundle()
    wrong_graph = _score("node", ("c",), 0.5).model_copy(update={"graph_version_hash": "f" * 64})
    with pytest.raises(ValueError, match="graph version"):
        build_risk_and_trust_findings(
            graph, scored_candidates=((wrong_graph, _confidence(wrong_graph)),)
        )
    wrong_task = ModelScore.create(
        task_id="core.community_resilience_review",
        entity_type="node",
        entity_ids=("c",),
        score=0.5,
        graph_version_hash=graph.graph_version_hash,
        model_version="resilience/1",
        model_version_hash="5" * 64,
    )
    with pytest.raises(ValueError, match="task"):
        build_risk_and_trust_findings(
            graph, scored_candidates=((wrong_task, _confidence(wrong_task)),)
        )
