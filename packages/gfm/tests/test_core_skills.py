import pytest
from pydantic import ValidationError

from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.governance import (
    CalibratedConfidence,
    ModelScore,
    RegressionConfidenceInterval,
    analyze_community_resilience,
    build_community_resilience_findings,
    create_governance_finding,
)
from socialgraph_gfm.core.knowledge import KnowledgeDocument, KnowledgeStore
from socialgraph_gfm.core.retrieval import StructuralIndex, StructuralRecord
from socialgraph_gfm.core.skills import (
    GenerateReportRequest,
    InspectGraphRequest,
    RetrieveEvidenceRequest,
    RunCoreTaskRequest,
    CoreSkillRegistry,
)


def _bundle() -> CoreGraphBundle:
    payload = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "a", "index": 0}, {"id": "b", "index": 1}],
        "edges": [{"sourceId": "a", "targetId": "b", "edgeType": "supports", "weight": 1.0}],
        "nodeFeatures": [],
        "structuralFeatures": None,
        "source": {"sourceName": "fixture", "sourceSha256": "1" * 64},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _registry(tmp_path):
    graph = _bundle()
    score = ModelScore.create(
        task_id="core.risk_and_trust_review",
        entity_type="node",
        entity_ids=("a",),
        score=0.7,
        graph_version_hash=graph.graph_version_hash,
        model_version="risk/1",
        model_version_hash="2" * 64,
    )
    finding = create_governance_finding(
        task_id="core.risk_and_trust_review",
        finding_type="node-risk-candidate",
        subject_ids=("a",),
        score=score,
        calibrated_confidence=CalibratedConfidence.create(
            score=score,
            value=0.6,
            calibration_version="cal/1",
            method="isotonic",
            calibration_artifact_hash="3" * 64,
            calibration_protocol_hash="4" * 64,
        ),
        evidence=analyze_community_resilience(graph)[:1],
        similar_cases=(),
        limitations=("Candidate for review; it is not a risk or trust truth label.",),
    )
    knowledge = KnowledgeStore(tmp_path / "skills.sqlite3")
    knowledge.add_document(
        KnowledgeDocument.create(
            category="governance-rule",
            title="Review rule",
            body="Every candidate requires manual review.",
            source_uri="urn:rule:review",
        )
    )
    structural = StructuralIndex(
        (
            StructuralRecord.create(
                record_id="node-a",
                kind="node",
                entity_ids=("a",),
                vector=(1.0, 0.0),
                representation="embedding",
                graph_version_hash=graph.graph_version_hash,
                model_version="risk/1",
                model_version_hash="2" * 64,
            ),
        )
    )
    return (
        CoreSkillRegistry(
            graphs=(graph,),
            findings=(finding,),
            structural_index=structural,
            knowledge_store=knowledge,
        ),
        graph,
        finding,
    )


@pytest.mark.parametrize(
    "model,payload",
    [
        (
            InspectGraphRequest,
            {
                "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
                "graphVersionHash": "a" * 64,
                "graphFacts": {"nodes": 9},
            },
        ),
        (
            RunCoreTaskRequest,
            {
                "schemaVersion": "socialgraph-fm.core-skill.run-core-task.request/2.0",
                "taskId": "core.risk_and_trust_review",
                "graphVersionHash": "a" * 64,
                "modelScore": 0.99,
            },
        ),
        (
            RetrieveEvidenceRequest,
            {
                "schemaVersion": "socialgraph-fm.core-skill.retrieve-evidence.request/2.0",
                "query": "review",
                "evidence": [{"invented": True}],
            },
        ),
        (
            GenerateReportRequest,
            {
                "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
                "findingHashes": ["a" * 64],
                "sanction": "ban",
            },
        ),
    ],
)
def test_skill_requests_reject_injected_scores_facts_evidence_and_sanctions(model, payload):
    with pytest.raises(ValidationError, match="Extra inputs"):
        model.model_validate(payload)


def test_registry_has_exactly_four_skills_and_rejects_malformed_plans(tmp_path):
    registry, _, _ = _registry(tmp_path)
    assert registry.skill_names == (
        "generate_report",
        "inspect_graph",
        "retrieve_evidence",
        "run_core_task",
    )
    with pytest.raises(ValueError, match="unsupported skill"):
        registry.execute("make_sanction", {})
    with pytest.raises(ValidationError):
        registry.execute(
            "inspect_graph",
            {
                "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
                "graphVersionHash": 123,
            },
        )
    with pytest.raises(ValueError, match="natural-language plans"):
        registry.execute_plan("Inspect the graph and ban risky people")


def test_skill_requests_require_explicit_versioned_schema():
    with pytest.raises(ValidationError, match="schemaVersion"):
        InspectGraphRequest.model_validate({"graphVersionHash": "a" * 64})


def test_tool_outputs_derive_only_from_registered_records(tmp_path):
    registry, graph, finding = _registry(tmp_path)
    inspected = registry.execute(
        "inspect_graph",
        {
            "schemaVersion": "socialgraph-fm.core-skill.inspect-graph.request/2.0",
            "graphVersionHash": graph.graph_version_hash,
        },
    )
    assert inspected.node_count == 2
    assert inspected.edge_count == 1
    assert not hasattr(inspected, "sanction")

    run = registry.execute(
        "run_core_task",
        {
            "schemaVersion": "socialgraph-fm.core-skill.run-core-task.request/2.0",
            "taskId": "core.risk_and_trust_review",
            "graphVersionHash": graph.graph_version_hash,
            "scopeNodeIds": ["a"],
        },
    )
    assert run.finding_hashes == (finding.finding_hash,)
    assert run.manual_review_required is True

    retrieved = registry.execute(
        "retrieve_evidence",
        {
            "schemaVersion": "socialgraph-fm.core-skill.retrieve-evidence.request/2.0",
            "query": "manual review",
            "limit": 2,
        },
    )
    assert retrieved.knowledge_results[0].document_hash
    assert retrieved.limitations


def test_no_llm_report_fallback_cites_hashes_and_never_fabricates(tmp_path):
    registry, _, finding = _registry(tmp_path)
    report = registry.execute(
        "generate_report",
        {
            "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
            "findingHashes": [finding.finding_hash],
            "format": "markdown",
        },
    )
    assert finding.finding_hash in report.content
    assert finding.evidence[0].evidence_hash in report.content
    assert "Manual human review" in report.content
    assert "non-causal" in report.content
    assert finding.evidence[0].limitations[0] in report.content
    assert report.generated_without_llm is True
    assert "no automatic sanction" in report.content.lower()
    assert "recommended sanction" not in report.content.lower()

    with pytest.raises(ValueError, match="registered finding"):
        registry.execute(
            "generate_report",
            {
                "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
                "findingHashes": ["f" * 64],
                "format": "markdown",
            },
        )


def test_no_llm_markdown_report_renders_regression_interval_without_probability_claim(tmp_path):
    _, graph, _ = _registry(tmp_path)
    score = ModelScore.create(
        task_id="core.community_resilience_review",
        entity_type="community",
        entity_ids=("a",),
        score=0.6,
        graph_version_hash=graph.graph_version_hash,
        model_version="resilience/1",
        model_version_hash="5" * 64,
    )
    interval = RegressionConfidenceInterval.create(
        score=score,
        lower_bound=0.42,
        upper_bound=0.71,
        coverage=0.9,
        validation_count=40,
        confidence_version="resilience-residuals/1",
        method="validation-residual-interval",
        confidence_artifact_hash="6" * 64,
        confidence_protocol_hash="7" * 64,
    )
    finding = build_community_resilience_findings(
        graph,
        scored_candidates=((score, interval),),
    )[0]
    registry = CoreSkillRegistry(
        graphs=(graph,),
        findings=(finding,),
        structural_index=StructuralIndex(()),
        knowledge_store=KnowledgeStore(tmp_path / "interval-skills.sqlite3"),
    )

    report = registry.execute(
        "generate_report",
        {
            "schemaVersion": "socialgraph-fm.core-skill.generate-report.request/2.0",
            "findingHashes": [finding.finding_hash],
            "format": "markdown",
        },
    )

    assert "Regression interval (not a probability)" in report.content
    assert "point estimate 0.6" in report.content
    assert "[0.42, 0.71]" in report.content
    assert "coverage 0.9" in report.content
    assert "method validation-residual-interval" in report.content
