from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.gfm_schemas import (
    CheckpointManifest,
    CompatibilityMapping,
    CorpusManifest,
    FeatureManifest,
    GfmCompatibilityReport,
    CoreFinding,
    GraphSnapshotRef,
    TimeRange,
    TrainingRunManifest,
)
from app.main import create_app


@pytest.mark.anyio
async def test_gfm_capabilities_are_explicitly_fail_closed(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/v1/gfm/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schemaVersion"] == "socialgraph-fm.core-capabilities/2.0"
    assert payload["registryGeneration"] == 0
    assert payload["servingReady"] is False
    assert payload["models"] == []
    assert payload["tasks"] == []
    assert payload["readiness"] == {
        "modelValidated": False,
        "coreServingReady": False,
    }


@pytest.mark.anyio
async def test_stale_infrastructure_setting_never_overrides_registry(tmp_path) -> None:
    app = create_app(
        Settings(
            dataset_storage_root=str(tmp_path / "store"),
            gfm_infrastructure_ready=True,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/gfm/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["readiness"]["modelValidated"] is False
    assert payload["readiness"]["coreServingReady"] is False
    assert payload["servingReady"] is False
    assert payload["models"] == []


@pytest.mark.anyio
async def test_only_the_four_v2_gfm_routes_are_public(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.get("/api/v1/gfm/tasks")

    assert response.status_code == 404


@pytest.mark.anyio
async def test_gfm_run_never_fabricates_a_prediction(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.post(
        "/api/v1/gfm/runs",
        json={
            "schemaVersion": "socialgraph-fm.core-run-request/2.0",
            "graphVersionId": "graph-v1",
            "taskId": "core.risk_and_trust_review",
            "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
            "modelVersionId": "socialgraph-fm-core/review",
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 3},
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "GFM_CORE_MODEL_NOT_INSTALLED"


def test_gfm_contracts_preserve_privacy_and_smoke_registration_boundaries() -> None:
    feature = FeatureManifest(
        id="actor-tenure",
        attribute="tenureDays",
        target="node",
        modality="numeric",
        dtype="float32",
        missingPolicy="mask",
        privacyLevel="project",
        fitScope="train_only",
        inferenceAllowed=True,
    )
    snapshot = GraphSnapshotRef(
        graphVersionId="graph-v1",
        graphFactHash="a" * 64,
        contentHash="b" * 64,
        profile="collaboration.actor-interaction/1.0",
        nodeCount=2,
        edgeCount=1,
        featureManifest=[feature],
    )
    corpus = CorpusManifest(
        id="synthetic-actor-interaction",
        version="1",
        sourceHash="c" * 64,
        licenseId="generated",
        intendedUse="synthetic_test_only",
        split="synthetic",
        adapter="fixtures.actor_interaction",
    )
    run = TrainingRunManifest(
        id="run-1",
        taskId="core.newcomer_support",
        graphSnapshot=snapshot,
        corpus=[corpus],
        seed=7,
        status="smoke",
        codeHash="d" * 64,
        environmentHash="e" * 64,
        configHash="f" * 64,
        createdAt="2026-08-12T00:00:00Z",
    )
    checkpoint = CheckpointManifest(
        id="checkpoint-1",
        runId=run.id,
        step=1,
        weightsHash="1" * 64,
        optimizerHash="2" * 64,
        configHash="f" * 64,
        integrityHash="3" * 64,
    )

    assert run.status == "smoke"
    assert checkpoint.registrable is False
    with pytest.raises(ValidationError):
        CompatibilityMapping.model_validate(
            {
                "profile": "collaboration.actor-interaction/1.0",
                "nodeTypeMap": {"member": "actor"},
                "edgeTypeMap": {"interaction": "collaborates"},
                "privacyLevel": "project",
                "userDataTrainingOptIn": True,
            }
        )
    with pytest.raises(ValidationError, match="不兼容的关系"):
        CompatibilityMapping.model_validate(
            {
                "profile": "collaboration.actor-interaction/1.0",
                "nodeTypeMap": {"member": "actor"},
                "edgeTypeMap": {"authored": "actor_contributes_artifact"},
                "privacyLevel": "project",
            }
        )
    with pytest.raises(ValidationError, match="prohibited"):
        FeatureManifest.model_validate(
            {
                "id": "email",
                "attribute": "email",
                "target": "node",
                "modality": "text",
                "dtype": "string",
                "missingPolicy": "mask",
                "privacyLevel": "prohibited",
                "inferenceAllowed": True,
            }
        )


def test_activity_hetero_relation_contract_and_core_finding() -> None:
    mapping = CompatibilityMapping(
        profile="collaboration.activity-hetero/1.0",
        nodeTypeMap={
            "user": "actor",
            "pull_request": "artifact",
            "repository": "community",
            "language": "topic",
        },
        edgeTypeMap={
            "commented": "actor_interacts_actor",
            "authored": "actor_contributes_artifact",
            "member_of": "actor_joins_community",
            "in_repo": "artifact_belongs_community",
            "uses_language": "artifact_has_topic",
        },
        timestampField="created_at",
        inferenceAttributeAllowlist=["tenure_days"],
        privacyLevel="project",
    )
    report = GfmCompatibilityReport(compatible=True, mapping=mapping)
    finding = CoreFinding(
        id="finding-1",
        taskId="core.coordination_review",
        targetId="candidate-subgraph-1",
        score=0.7,
        uncertainty=0.2,
        reasonCodes=["TEMPORAL_COINCIDENCE"],
        evidence=[{"kind": "subgraph", "ref": "snapshot://graph-v1/subgraph/1"}],
        provenance={"modelId": "future-model", "graphFactHash": "a" * 64},
    )

    assert report.mapping is not None
    assert set(report.mapping.edge_type_map.values()) == {
        "actor_interacts_actor",
        "actor_contributes_artifact",
        "actor_joins_community",
        "artifact_belongs_community",
        "artifact_has_topic",
    }
    assert finding.human_review_required is True


def test_public_time_range_requires_timezone_aware_boundaries() -> None:
    with pytest.raises(ValidationError, match="start.*时区"):
        TimeRange(start="2026-08-12T00:00:00")
    with pytest.raises(ValidationError, match="end.*时区"):
        TimeRange(end="2026-08-12T01:00:00")

    time_range = TimeRange(
        start="2026-08-12T00:00:00+08:00",
        end="2026-08-12T01:00:00+08:00",
    )
    assert time_range.start is not None and time_range.start.utcoffset() is not None
    assert time_range.end is not None and time_range.end.utcoffset() is not None
