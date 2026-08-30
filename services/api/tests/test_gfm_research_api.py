from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
import numpy as np
import pytest

from app.dataset_imports import DatasetImportService, graph_fact_hash_v1
from app.dataset_schemas import (
    DatasetPreparationSpec,
    GraphDatasetHandoffRequest,
    GraphHandoffReserveRequest,
    GraphVersionTargetDomainEnvelope,
)
from app.gfm_hashing import canonical_sha256
from app.gfm_research import ResearchGraphRegistration, _simple_undirected_edge_count
from app.gfm_research_schemas import (
    RESEARCH_TASK_IDS,
    ResearchRunRequest,
    ResearchRunResult,
)
from app.main import create_app

from .test_atomic_handoff import _request as graph_handoff_request

MODEL_HASH = "a" * 64
ARTIFACT_HASH = "b" * 64
GRAPH_HASHES = {
    "twitch-content-policy": "1" * 64,
    "tolokers-account-risk": "2" * 64,
    "wiki-rfa-signed-relation": "3" * 64,
    "email-eu-collaboration": "4" * 64,
}


def test_simple_undirected_coo_requires_exactly_one_edge_per_direction() -> None:
    paired = np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    same_direction_duplicates = np.asarray([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.int64)
    unbalanced_duplicates = np.asarray([[0, 1, 1, 1], [1, 0, 2, 2]], dtype=np.int64)

    assert _simple_undirected_edge_count(paired) == 2
    assert _simple_undirected_edge_count(same_direction_duplicates) is None
    assert _simple_undirected_edge_count(unbalanced_duplicates) is None


def _with_hash(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _capabilities() -> dict[str, Any]:
    return _with_hash(
        {
            "schemaVersion": "socialgraph-fm.research/1.0",
            "channel": "research",
            "releaseLabel": "SocialGraph-FM Research",
            "seed": 1729,
            "preliminary": True,
            "researchServingReady": True,
            "unavailableReason": None,
            "model": {
                "modelVersionId": "socialgraph-fm-research/model-1",
                "modelVersionHash": MODEL_HASH,
                "artifactHash": ARTIFACT_HASH,
                "taskIds": list(RESEARCH_TASK_IDS),
                "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
                "maxNodes": 50_000,
                "maxEdges": 1_500_000,
                "claimStatus": "not_demonstrated",
            },
            "taskIds": list(RESEARCH_TASK_IDS),
            "upload": {
                "compatibleTaskIds": ["core.collaboration_completion"],
                "auxiliaryCapabilities": ["similar-nodes"],
                "minNodes": 5,
                "maxNodes": 50_000,
                "maxEdges": 1_500_000,
            },
        },
        "capabilityHash",
    )


def _scenario_rows() -> list[dict[str, Any]]:
    return [
        {
            "scenarioId": "twitch-content-policy",
            "datasetId": "twitch-language",
            "title": "Content policy review",
            "taskId": "research.content_policy_review",
            "graphVersionId": "research:twitch-language",
            "graphVersionHash": GRAPH_HASHES["twitch-content-policy"],
            "modelVersionId": "socialgraph-fm-research/model-1",
            "enabled": True,
            "unavailableReason": None,
            "defaultTargetScope": {"kind": "nodes", "nodeIds": ["0"]},
            "primaryMetric": {"name": "macro_f1", "value": 0.7},
            "scratchDelta": None,
        },
        {
            "scenarioId": "tolokers-account-risk",
            "datasetId": "tolokers",
            "title": "Historical account status review",
            "taskId": "research.account_risk_review",
            "graphVersionId": "research:tolokers",
            "graphVersionHash": GRAPH_HASHES["tolokers-account-risk"],
            "modelVersionId": "socialgraph-fm-research/model-1",
            "enabled": True,
            "unavailableReason": None,
            "defaultTargetScope": {"kind": "nodes", "nodeIds": ["0"]},
            "primaryMetric": {"name": "auprc", "value": 0.6},
            "scratchDelta": None,
        },
        {
            "scenarioId": "wiki-rfa-signed-relation",
            "datasetId": "wiki-rfa",
            "title": "Governance relation stance review",
            "taskId": "research.signed_relation_review",
            "graphVersionId": "research:wiki-rfa",
            "graphVersionHash": GRAPH_HASHES["wiki-rfa-signed-relation"],
            "modelVersionId": "socialgraph-fm-research/model-1",
            "enabled": True,
            "unavailableReason": None,
            "defaultTargetScope": {
                "kind": "directed-node-pairs",
                "pairs": [["0", "1"]],
            },
            "primaryMetric": {"name": "negative_auprc", "value": 0.55},
            "scratchDelta": None,
        },
        {
            "scenarioId": "email-eu-collaboration",
            "datasetId": "email-eu-core",
            "title": "Collaboration relation candidates",
            "taskId": "core.collaboration_completion",
            "graphVersionId": "research:email-eu-core",
            "graphVersionHash": GRAPH_HASHES["email-eu-collaboration"],
            "modelVersionId": "socialgraph-fm-research/model-1",
            "enabled": True,
            "unavailableReason": None,
            "defaultTargetScope": {
                "kind": "collaboration-candidates",
                "anchorNodeId": "0",
                "topK": 10,
            },
            "primaryMetric": {"name": "filtered_mrr", "value": 0.4},
            "scratchDelta": None,
        },
    ]


def _scenarios() -> dict[str, Any]:
    return _with_hash(
        {
            "schemaVersion": "socialgraph-fm.research/1.0",
            "releaseLabel": "SocialGraph-FM Research",
            "seed": 1729,
            "preliminary": True,
            "scenarios": _scenario_rows(),
        },
        "scenariosHash",
    )


def _run_request() -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "graphVersionId": "research:twitch-language",
        "taskId": "research.content_policy_review",
        "modelVersionId": "socialgraph-fm-research/model-1",
        "targetScope": {"kind": "nodes", "nodeIds": ["0"]},
        "scenarioId": "twitch-content-policy",
        "parameters": {"candidateLimit": 20},
    }


def _preview() -> dict[str, Any]:
    return _with_hash(
        {
            "schemaVersion": "socialgraph-fm.research/1.0",
            "scenarioId": "twitch-content-policy",
            "graphVersionId": "research:twitch-language",
            "graphVersionHash": GRAPH_HASHES["twitch-content-policy"],
            "modelVersionId": "socialgraph-fm-research/model-1",
            "modelVersionHash": MODEL_HASH,
            "nodes": [
                {"id": "0", "label": "0"},
                {"id": "1", "label": "1"},
            ],
            "edges": [
                {
                    "id": "edge:0:1",
                    "source": "0",
                    "target": "1",
                    "directed": False,
                }
            ],
            "partialPreview": True,
            "nodeCount": 100,
            "edgeCount": 200,
        },
        "previewHash",
    )


class FakeResearchClient:
    def __init__(self) -> None:
        self.capability_payload = _capabilities()
        self.scenario_payload = _scenarios()
        self.preview_payload = _preview()
        self.created_payload: dict[str, Any] | None = None
        self.status_payload: dict[str, Any] | None = None
        self.result_payload: dict[str, Any] | None = None
        self.similar_payload: dict[str, Any] | None = None

    async def research_capabilities(self) -> dict[str, Any]:
        return self.capability_payload

    async def research_scenarios(self) -> dict[str, Any]:
        return self.scenario_payload

    async def research_scenario_preview(self, _scenario_id: str) -> dict[str, Any]:
        return self.preview_payload

    async def create_research_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created_payload = payload
        assert self.status_payload is not None
        return self.status_payload

    async def get_research_run(self, _run_id: str) -> dict[str, Any]:
        assert self.status_payload is not None
        return self.status_payload

    async def get_research_result(self, _run_id: str) -> dict[str, Any]:
        assert self.result_payload is not None
        return self.result_payload

    async def research_similar_nodes(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created_payload = payload
        assert self.similar_payload is not None
        return self.similar_payload

    async def register_research_graph(self, _payload: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("registered scenarios do not use uploaded graph registration")


class RegisteringResearchClient(FakeResearchClient):
    def __init__(self) -> None:
        super().__init__()
        self.registration_payloads: list[dict[str, Any]] = []
        self.adapter_status = "ready"

    async def register_research_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.registration_payloads.append(payload)
        graph = payload["graphReference"]
        model = payload["expectedModel"]
        response = {
            "schemaVersion": "socialgraph-fm.research/1.0",
            "graphVersionId": graph["graphVersionId"],
            "graphVersionHash": graph["graphVersionHash"],
            "modelVersionId": model["modelVersionId"],
            "modelVersionHash": model["modelVersionHash"],
            "adapterStatus": self.adapter_status,
            "compatibleTaskIds": payload["compatibleTaskIds"],
            "auxiliaryCapabilities": payload["auxiliaryCapabilities"],
        }
        response["registrationHash"] = canonical_sha256(response)
        return response


class MismatchedRegistrationClient(RegisteringResearchClient):
    async def register_research_graph(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await super().register_research_graph(payload)
        response["graphVersionHash"] = "f" * 64
        response.pop("registrationHash")
        response["registrationHash"] = canonical_sha256(response)
        return response


@pytest.mark.anyio
async def test_research_channel_is_explicitly_unavailable_without_runtime(
    api_client: httpx.AsyncClient,
) -> None:
    capabilities = await api_client.get("/api/v1/gfm/research/capabilities")
    scenarios = await api_client.get("/api/v1/gfm/research/scenarios")
    create = await api_client.post("/api/v1/gfm/research/runs", json=_run_request())

    assert capabilities.status_code == 200
    assert capabilities.json()["researchServingReady"] is False
    assert capabilities.json()["unavailableReason"] == "RESEARCH_MODEL_NOT_INSTALLED"
    assert capabilities.json()["seed"] == 1729
    assert capabilities.json()["preliminary"] is True
    assert capabilities.json()["taskIds"] == list(RESEARCH_TASK_IDS)
    assert scenarios.status_code == 200
    assert len(scenarios.json()["scenarios"]) == 4
    assert all(not item["enabled"] for item in scenarios.json()["scenarios"])
    assert create.status_code == 503
    assert create.json()["detail"]["code"] == "GFM_RESEARCH_MODEL_NOT_INSTALLED"


@pytest.mark.anyio
async def test_research_lifecycle_is_hash_bound_and_does_not_change_formal_readiness(
    unconfigured_settings,
) -> None:
    fake = FakeResearchClient()
    request = _run_request()
    request_hash = canonical_sha256(request)
    status = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "runId": "research-run-1",
        "requestHash": request_hash,
        "status": "succeeded",
        "progress": 100,
        "createdAt": "2026-08-16T00:00:00.000000Z",
        "updatedAt": "2026-08-16T00:00:01.000000Z",
        "errorCode": None,
    }
    status["stateHash"] = canonical_sha256(status)
    fake.status_payload = status
    result = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "runId": "research-run-1",
        "requestHash": request_hash,
        "taskId": "research.content_policy_review",
        "graphVersionId": "research:twitch-language",
        "graphVersionHash": GRAPH_HASHES["twitch-content-policy"],
        "modelVersionId": "socialgraph-fm-research/model-1",
        "modelVersionHash": MODEL_HASH,
        "seed": 1729,
        "preliminary": True,
        "calibrationStatus": "ranking_only",
        "findings": [
            {
                "id": "finding-1",
                "rank": 1,
                "entityType": "node",
                "entityIds": ["0"],
                "score": 0.8,
                "scoreKind": "ranking-score",
                "calibrated": False,
                "reasonCodes": ["MODEL_RANKING"],
                "limitations": [
                    "Single-seed preliminary result; manual review is required."
                ],
                "reviewRequired": True,
            }
        ],
        "completedAt": "2026-08-16T00:00:01.000000Z",
    }
    result["resultHash"] = canonical_sha256(result)
    fake.result_payload = result
    app = create_app(unconfigured_settings, gfm_research_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        formal = await client.get("/api/v1/gfm/capabilities")
        capabilities = await client.get("/api/v1/gfm/research/capabilities")
        created = await client.post("/api/v1/gfm/research/runs", json=request)
        observed = await client.get("/api/v1/gfm/research/runs/research-run-1")
        completed = await client.get(
            "/api/v1/gfm/research/runs/research-run-1/result"
        )

    assert formal.json()["servingReady"] is False
    assert formal.json()["readiness"] == {
        "modelValidated": False,
        "coreServingReady": False,
    }
    assert capabilities.json()["researchServingReady"] is True
    assert created.status_code == 202
    assert observed.status_code == completed.status_code == 200
    assert fake.created_payload is not None
    assert fake.created_payload["graphReference"]["graphVersionHash"] == GRAPH_HASHES[
        "twitch-content-policy"
    ]
    assert completed.json()["preliminary"] is True
    assert completed.json()["findings"][0]["reviewRequired"] is True


@pytest.mark.anyio
async def test_scenario_preview_is_bounded_and_exactly_hash_bound(
    unconfigured_settings,
) -> None:
    fake = FakeResearchClient()
    app = create_app(unconfigured_settings, gfm_research_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        valid = await client.get(
            "/api/v1/gfm/research/scenarios/twitch-content-policy/graph-preview"
        )
        missing = await client.get(
            "/api/v1/gfm/research/scenarios/not-registered/graph-preview"
        )
        fake.preview_payload = _preview() | {"graphVersionHash": "f" * 64}
        fake.preview_payload["previewHash"] = canonical_sha256(
            {key: value for key, value in fake.preview_payload.items() if key != "previewHash"}
        )
        mismatched = await client.get(
            "/api/v1/gfm/research/scenarios/twitch-content-policy/graph-preview"
        )

    assert valid.status_code == 200
    assert valid.json()["partialPreview"] is True
    assert valid.json()["graphVersionHash"] == GRAPH_HASHES["twitch-content-policy"]
    assert missing.status_code == 404
    assert mismatched.status_code == 502
    assert mismatched.json()["detail"]["code"] == "GFM_RESEARCH_PREVIEW_BINDING_MISMATCH"


@pytest.mark.anyio
async def test_tampered_research_result_is_rejected_against_durable_binding(
    unconfigured_settings,
) -> None:
    fake = FakeResearchClient()
    request = _run_request()
    request_hash = canonical_sha256(request)
    status = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "runId": "research-run-tampered",
        "requestHash": request_hash,
        "status": "succeeded",
        "progress": 100,
        "createdAt": "2026-08-16T00:00:00.000000Z",
        "updatedAt": "2026-08-16T00:00:01.000000Z",
        "errorCode": None,
    }
    status["stateHash"] = canonical_sha256(status)
    fake.status_payload = status
    result = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "runId": "research-run-tampered",
        "requestHash": request_hash,
        "taskId": "research.content_policy_review",
        "graphVersionId": "research:twitch-language",
        "graphVersionHash": "f" * 64,
        "modelVersionId": "socialgraph-fm-research/model-1",
        "modelVersionHash": MODEL_HASH,
        "seed": 1729,
        "preliminary": True,
        "calibrationStatus": "ranking_only",
        "findings": [],
        "completedAt": "2026-08-16T00:00:01.000000Z",
    }
    result["resultHash"] = canonical_sha256(result)
    fake.result_payload = result
    app = create_app(unconfigured_settings, gfm_research_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        assert (await client.post("/api/v1/gfm/research/runs", json=request)).status_code == 202
        response = await client.get(
            "/api/v1/gfm/research/runs/research-run-tampered/result"
        )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "GFM_RESEARCH_RESULT_BINDING_MISMATCH"


@pytest.mark.anyio
async def test_similar_nodes_requires_exact_input_and_model_binding(
    unconfigured_settings,
) -> None:
    fake = FakeResearchClient()
    payload = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "graphVersionId": "research:twitch-language",
        "nodeId": "0",
        "topK": 3,
        "modelVersionId": "socialgraph-fm-research/model-1",
    }
    response = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "graphVersionId": "research:twitch-language",
        "nodeId": "0",
        "modelVersionId": "socialgraph-fm-research/model-1",
        "modelVersionHash": MODEL_HASH,
        "matches": [
            {
                "graphVersionId": "research:tolokers",
                "nodeId": "7",
                "datasetId": "tolokers",
                "similarity": 0.9,
                "structuralFacts": {
                    "degree": 5,
                    "inDegree": 5,
                    "outDegree": 5,
                    "pagerank": 0.01,
                    "clustering": 0.2,
                    "coreNumber": 3,
                },
            }
        ],
    }
    response["resultHash"] = canonical_sha256(response)
    fake.similar_payload = response
    app = create_app(unconfigured_settings, gfm_research_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        valid = await client.post("/api/v1/gfm/research/similar-nodes", json=payload)
        missing = await client.post(
            "/api/v1/gfm/research/similar-nodes",
            json={key: value for key, value in payload.items() if key != "nodeId"},
        )
        wrong_model = await client.post(
            "/api/v1/gfm/research/similar-nodes",
            json=payload | {"modelVersionId": "other-model"},
        )

    assert valid.status_code == 200
    assert valid.json()["matches"][0]["datasetId"] == "tolokers"
    assert missing.status_code == 422
    assert wrong_model.status_code == 409
    assert wrong_model.json()["detail"]["code"] == "GFM_RESEARCH_MODEL_MISMATCH"


@pytest.mark.anyio
async def test_research_handoff_reports_structural_compatibility_without_claiming_ready(
    unconfigured_settings,
) -> None:
    request, _, _ = graph_handoff_request(
        create_app(unconfigured_settings).state.dataset_imports,
        "research-small-graph",
    )
    request = request.model_copy(update={"intended_use": "gfm_research"})
    app = create_app(unconfigured_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/graph-dataset-handoffs/commit",
            json=request.model_dump(mode="json", by_alias=True),
        )

    assert response.status_code == 200
    compatibility = response.json()["researchCompatibility"]
    assert compatibility["status"] == "blocked"
    assert compatibility["compatibleTaskIds"] == []
    assert compatibility["auxiliaryCapabilities"] == []
    assert compatibility["adapterStatus"] == "pending_registration"
    assert {item["code"] for item in compatibility["blockers"]} >= {
        "RESEARCH_STRUCTURAL_RETRIEVAL_TOO_SMALL",
        "RESEARCH_COLLABORATION_NODE_COUNT_UNSUPPORTED",
    }


@pytest.mark.anyio
async def test_research_handoff_preserves_nonretryable_registration_failure_as_blocker(
    unconfigured_settings,
) -> None:
    fake = MismatchedRegistrationClient()
    app = create_app(unconfigured_settings, gfm_research_client=fake)
    nodes = [
        {"id": f"n{index}", "label": f"Node {index}", "attributes": {}}
        for index in range(20)
    ]
    edges = [
        {
            "id": f"e{index}",
            "source": f"n{index}",
            "target": f"n{(index + 1) % 20}",
            "directed": False,
            "attributes": {},
        }
        for index in range(20)
    ]
    envelope = GraphVersionTargetDomainEnvelope.model_validate(
        {
            "schemaVersion": "socialgraph-fm-graph/1.0",
            "graphVersionId": "registration-mismatch-graph",
            "contentHash": "c" * 64,
            "buildSpecHash": "d" * 64,
            "sourceFile": "uploaded.json",
            "directedness": "undirected",
            "nodes": nodes,
            "edges": edges,
        }
    )
    fact_hash = graph_fact_hash_v1(envelope)
    reservation = app.state.dataset_imports.reserve_graph_handoff(
        GraphHandoffReserveRequest(
            graphVersionId=envelope.graph_version_id,
            graphFactHash=fact_hash,
        )
    )
    request = GraphDatasetHandoffRequest(
        token=reservation.token,
        envelope=envelope,
        preparation=DatasetPreparationSpec(
            graphVersionId=envelope.graph_version_id,
            featureAttributes=[],
            taskKind="none",
            splitStrategy="none",
        ),
        intendedUse="gfm_research",
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/api/v1/graph-dataset-handoffs/commit",
            json=request.model_dump(mode="json", by_alias=True),
        )

    assert response.status_code == 200, response.text
    compatibility = response.json()["researchCompatibility"]
    assert compatibility["status"] == "blocked"
    assert compatibility["compatibleTaskIds"] == []
    assert compatibility["auxiliaryCapabilities"] == []
    assert {item["code"] for item in compatibility["blockers"]} >= {
        "GFM_RESEARCH_REGISTRATION_MISMATCH"
    }


@pytest.mark.anyio
async def test_existing_uploaded_binding_is_lazily_registered_before_inference(
    unconfigured_settings,
) -> None:
    service = DatasetImportService(unconfigured_settings)
    nodes = [
        {"id": f"n{index}", "label": f"Node {index}", "attributes": {}}
        for index in range(20)
    ]
    edges = [
        {
            "id": f"e{index}",
            "source": f"n{index}",
            "target": f"n{(index + 1) % 20}",
            "directed": False,
            "attributes": {},
        }
        for index in range(20)
    ]
    envelope = GraphVersionTargetDomainEnvelope.model_validate(
        {
            "schemaVersion": "socialgraph-fm-graph/1.0",
            "graphVersionId": "uploaded-collaboration",
            "contentHash": "c" * 64,
            "buildSpecHash": "d" * 64,
            "sourceFile": "uploaded.json",
            "directedness": "undirected",
            "nodes": nodes,
            "edges": edges,
        }
    )
    fact_hash = graph_fact_hash_v1(envelope)
    reservation = service.reserve_graph_handoff(
        GraphHandoffReserveRequest(
            graphVersionId=envelope.graph_version_id,
            graphFactHash=fact_hash,
        )
    )
    handoff = GraphDatasetHandoffRequest(
        token=reservation.token,
        envelope=envelope,
        preparation=DatasetPreparationSpec(
            graphVersionId=envelope.graph_version_id,
            featureAttributes=[],
            taskKind="none",
            splitStrategy="none",
        ),
    )
    committed = service.commit_graph_handoff(handoff)
    assert committed.research_compatibility is None

    fake = RegisteringResearchClient()
    request = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "graphVersionId": envelope.graph_version_id,
        "taskId": "core.collaboration_completion",
        "modelVersionId": "socialgraph-fm-research/model-1",
        "targetScope": {
            "kind": "collaboration-candidates",
            "anchorNodeId": "n0",
            "topK": 5,
        },
        "parameters": {"candidateLimit": 20},
    }
    status = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "runId": "uploaded-run-1",
        "requestHash": ResearchRunRequest.model_validate(request).request_hash,
        "status": "queued",
        "progress": 0,
        "createdAt": "2026-08-16T00:00:00.000000Z",
        "updatedAt": "2026-08-16T00:00:00.000000Z",
        "errorCode": None,
    }
    status["stateHash"] = canonical_sha256(status)
    fake.status_payload = status
    app = create_app(unconfigured_settings, gfm_research_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/api/v1/gfm/research/runs", json=request)

    assert response.status_code == 202, response.text
    assert len(fake.registration_payloads) == 1
    registered = fake.registration_payloads[0]["graphReference"]
    assert registered["kind"] == "uploaded-artifact"
    assert registered["artifactId"] == committed.artifact.id
    assert registered["artifactHash"] == committed.artifact.content_hash
    assert registered["graphVersionHash"] == committed.artifact.canonical_graph_hash
    assert fake.created_payload is not None
    assert fake.created_payload["graphReference"] == registered


def test_research_public_models_reject_task_scope_and_fabricated_fields() -> None:
    with pytest.raises(ValueError):
        ResearchRunRequest.model_validate(
            _run_request()
            | {
                "targetScope": {
                    "kind": "collaboration-candidates",
                    "anchorNodeId": "0",
                    "topK": 5,
                }
            }
        )
    injected = deepcopy(_run_request())
    injected["prediction"] = 0.9
    with pytest.raises(ValueError):
        ResearchRunRequest.model_validate(injected)


def test_research_result_calibration_status_is_uniform() -> None:
    base_finding = {
        "id": "finding-1",
        "rank": 1,
        "entityType": "node",
        "entityIds": ["0"],
        "score": 0.7,
        "scoreKind": "ranking-score",
        "calibrated": False,
        "reasonCodes": [],
        "limitations": ["Single-seed preliminary result."],
        "reviewRequired": True,
    }
    payload = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "runId": "run-calibration",
        "requestHash": "1" * 64,
        "taskId": "research.content_policy_review",
        "graphVersionId": "research:twitch-language",
        "graphVersionHash": "2" * 64,
        "modelVersionId": "socialgraph-fm-research/model-1",
        "modelVersionHash": MODEL_HASH,
        "seed": 1729,
        "preliminary": True,
        "calibrationStatus": "ranking_only",
        "findings": [
            base_finding,
            {
                **base_finding,
                "id": "finding-2",
                "rank": 2,
                "scoreKind": "probability",
                "calibrated": True,
            },
        ],
        "completedAt": "2026-08-16T00:00:00Z",
    }
    payload["resultHash"] = canonical_sha256(payload)
    with pytest.raises(ValueError, match="ranking_only"):
        ResearchRunResult.model_validate(payload)

    empty = {**payload, "calibrationStatus": "calibrated", "findings": []}
    empty.pop("resultHash")
    empty["resultHash"] = canonical_sha256(empty)
    with pytest.raises(ValueError, match="empty findings"):
        ResearchRunResult.model_validate(empty)


def test_research_registration_contract_accepts_hash_bound_pending_state() -> None:
    payload = {
        "schemaVersion": "socialgraph-fm.research/1.0",
        "graphVersionId": "uploaded-collaboration",
        "graphVersionHash": "1" * 64,
        "modelVersionId": "socialgraph-fm-research/model-1",
        "modelVersionHash": MODEL_HASH,
        "adapterStatus": "pending_registration",
        "compatibleTaskIds": ["core.collaboration_completion"],
        "auxiliaryCapabilities": ["similar-nodes"],
    }
    payload["registrationHash"] = canonical_sha256(payload)
    parsed = ResearchGraphRegistration.model_validate(payload)
    assert parsed.adapter_status == "pending_registration"
