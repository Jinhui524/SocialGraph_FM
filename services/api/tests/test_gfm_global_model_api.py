from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.gfm_hashing import canonical_sha256
from app.gfm_global_model_schemas import GLOBAL_MODEL_PROTOCOLS
from app.main import create_app

HASHES = {
    "model": "a" * 64,
    "artifact": "b" * 64,
    "corpus": "c" * 64,
    "code": "d" * 64,
    "graph": "e" * 64,
    "split": "f" * 64,
    "state": "9" * 64,
}
RUN_ID = "global-model-" + "1" * 32


def _hashed(payload: dict[str, Any], field: str) -> dict[str, Any]:
    payload[field] = canonical_sha256(payload)
    return payload


def _metric() -> dict[str, Any]:
    return {"macroF1": 0.8, "prAuc": 0.75, "threshold": 0.52, "labelledTrainNodes": 429}


def _capabilities() -> dict[str, Any]:
    protocol_models = {
        protocol: {
            "modelVersionId": f"socialgraph-fm-{protocol.replace('_', '-')}/test",
            "modelVersionHash": f"{index}" * 64,
            "modelStateHash": HASHES["state"],
            "state": "servingReady" if protocol == "global" else "frozenDemo",
        }
        for index, protocol in enumerate(GLOBAL_MODEL_PROTOCOLS, start=1)
    }
    protocol_models["global"]["modelVersionHash"] = HASHES["model"]
    return _hashed(
        {
            "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
            "channel": "socialgraph-global",
            "releaseLabel": "SocialGraph-FM Global",
            "seed": 12121995,
            "servingReady": True,
            "unavailableReason": None,
            "taskId": "coordination_risk",
            "datasetVersionId": "socialgraph-fm:russia",
            "model": {
                "modelVersionId": "socialgraph-fm-global/test",
                "modelVersionHash": HASHES["model"],
                "artifactHash": HASHES["artifact"],
                "corpusHash": HASHES["corpus"],
                "sourceCodeHash": HASHES["code"],
                "taskId": "coordination_risk",
                "protocols": list(GLOBAL_MODEL_PROTOCOLS),
                "protocolModels": protocol_models,
                "state": "servingReady",
            },
        },
        "capabilityHash",
    )


def _health() -> dict[str, Any]:
    service_identity = canonical_sha256(
        {
            "service": "socialgraph-fm-gfm/global-model",
            "datasetVersionId": "socialgraph-fm:russia",
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": HASHES["model"],
            "corpusHash": HASHES["corpus"],
        }
    )
    return _hashed(
        {
            "schemaVersion": "socialgraph-fm.global-model-health/1.0",
            "serviceIdentity": service_identity,
            "servingReady": True,
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": HASHES["model"],
            "corpusHash": HASHES["corpus"],
            "datasetVersionId": "socialgraph-fm:russia",
        },
        "healthHash",
    )


def _model_card() -> dict[str, Any]:
    capabilities = _capabilities()
    protocol_models = capabilities["model"]["protocolModels"]
    return _hashed(
        {
            "schemaVersion": "socialgraph-fm.global-model-card/1.0",
            "releaseId": "socialgraph-fm",
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": HASHES["model"],
            "taskId": "coordination_risk",
            "architecture": {
                "name": "SocialGraph-FM Global GraphSAGE",
                "textFeatures": "anonymous precomputed embeddings",
                "structuralFeatures": "factual fused-graph degree buckets",
                "gnnLayers": 2,
                "hiddenDim": 256,
                "router": "shared residual plus domain/null adapters",
            },
            "protocols": protocol_models,
            "trainingData": {
                "countries": ["china", "cuba", "iran", "russia", "UAE", "venezuela"],
                "nodeCount": 4296,
                "nodeCountByCountry": {
                    country: 716
                    for country in ("china", "cuba", "iran", "russia", "UAE", "venezuela")
                },
                "content": "anonymous graph data with no raw text",
            },
            "intendedUse": ["analyst-facing prioritization with human review"],
            "outOfScope": ["automatic enforcement"],
            "limitations": ["frozen static research snapshot"],
            "ethics": ["preserve anonymity and require human review"],
            "licenses": [
                {
                    "name": "SocialGraph-FM dataset",
                    "license": "CC-BY-4.0",
                    "url": "https://zenodo.org/records/13357621",
                },
                {
                    "name": "Upstream research code",
                    "license": "MIT",
                    "url": "https://example.invalid/upstream-research",
                },
            ],
            "sourceAttribution": {
                "kind": "inspired",
                "paperUrl": "https://proceedings.mlr.press/v267/yuan25h.html",
                "completeReproduction": False,
            },
            "metrics": {
                protocol: {"countryBalancedMacroF1": 0.8}
                for protocol in GLOBAL_MODEL_PROTOCOLS
            },
            "artifactHash": HASHES["artifact"],
        },
        "modelCardHash",
    )


def _scenario() -> dict[str, Any]:
    return _hashed(
        {
            "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
            "scenarioId": "russia-coordination-risk",
            "datasetVersionId": "socialgraph-fm:russia",
            "graphVersionHash": HASHES["graph"],
            "modelVersionId": "socialgraph-fm-global/test",
            "enabled": True,
            "unavailableReason": None,
            "nodeCount": 716,
            "edgeCount": 1,
            "protocols": list(GLOBAL_MODEL_PROTOCOLS),
            "metrics": {protocol: _metric() for protocol in GLOBAL_MODEL_PROTOCOLS},
            "limitations": ["Anonymous research identifiers only."],
        },
        "scenarioHash",
    )


def _preview() -> dict[str, Any]:
    return _hashed(
        {
            "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
            "datasetVersionId": "socialgraph-fm:russia",
            "graphVersionHash": HASHES["graph"],
            "nodes": [
                {"id": "0", "label": "Account 0", "degree": 1, "structureMissing": False},
                {"id": "1", "label": "Account 1", "degree": 1, "structureMissing": False},
            ],
            "edges": [{"id": "0:1", "source": "0", "target": "1", "modality": "fused"}],
            "nodeCount": 716,
            "edgeCount": 1,
            "partialPreview": True,
        },
        "previewHash",
    )


def _status(request_hash: str, status: str = "succeeded") -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
        "runId": RUN_ID,
        "requestHash": request_hash,
        "status": status,
        "progress": 100 if status == "succeeded" else 0,
        "createdAt": "2026-08-17T00:00:00Z",
        "updatedAt": "2026-08-17T00:00:01Z",
        "errorCode": None,
    }


def _result(request_hash: str) -> dict[str, Any]:
    return _hashed(
        {
            "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
            "runId": RUN_ID,
            "requestHash": request_hash,
            "taskId": "coordination_risk",
            "protocol": "global",
            "datasetVersionId": "socialgraph-fm:russia",
            "graphVersionHash": HASHES["graph"],
            "corpusHash": HASHES["corpus"],
            "splitHash": HASHES["split"],
            "modelVersionId": "socialgraph-fm-global/test",
            "modelVersionHash": HASHES["model"],
            "threshold": 0.52,
            "metrics": _metric(),
            "findings": [
                {
                    "nodeId": "0",
                    "score": 0.91,
                    "rank": 1,
                    "riskBand": "high",
                    "predictedPositive": True,
                    "structureMissing": False,
                    "routes": [
                        {"expert": "shared", "weight": 1.0},
                        {"expert": "russia", "weight": 0.72},
                    ],
                    "modalityEvidence": {
                        "coRT": 1,
                        "coURL": 0,
                        "hashSeq": 0,
                        "fastRT": 0,
                        "tweetSim": 1,
                    },
                }
            ],
            "limitations": ["Human review required."],
            "completedAt": "2026-08-17T00:00:01Z",
        },
        "resultHash",
    )


class FakeGlobalModelClient:
    def __init__(self) -> None:
        self.request_hash = "0" * 64

    async def global_model_capabilities(self) -> dict[str, Any]:
        return deepcopy(_capabilities())

    async def global_model_health(self) -> dict[str, Any]:
        return deepcopy(_health())

    async def global_model_card(self) -> dict[str, Any]:
        return deepcopy(_model_card())

    async def global_model_scenario(self) -> dict[str, Any]:
        return deepcopy(_scenario())

    async def global_model_scenario_preview(self) -> dict[str, Any]:
        return deepcopy(_preview())

    async def create_global_model_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.request_hash = canonical_sha256(payload["request"])
        return _status(self.request_hash, "queued")

    async def get_global_model_run(self, _run_id: str) -> dict[str, Any]:
        return _status(self.request_hash)

    async def get_global_model_result(self, _run_id: str) -> dict[str, Any]:
        return _result(self.request_hash)

    async def get_global_model_evidence(self, _run_id: str, node_id: str) -> dict[str, Any]:
        result = _result(self.request_hash)
        finding = result["findings"][0]
        assert finding["nodeId"] == node_id
        return _hashed(
            {
                "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
                "runId": RUN_ID,
                "resultHash": result["resultHash"],
                "graphVersionHash": HASHES["graph"],
                "modelVersionId": "socialgraph-fm-global/test",
                "modelVersionHash": HASHES["model"],
                "threshold": 0.52,
                "node": finding,
                "neighbors": [
                    {
                        "nodeId": "1",
                        "score": 0.2,
                        "hop": 1,
                        "riskBand": "low",
                        "predictedPositive": False,
                        "structureMissing": False,
                        "modalities": ["tweetSim"],
                        "relations": [{"modality": "tweetSim", "rawWeight": 0.75}],
                    }
                ],
                "structuralSignals": {
                    "fusedDegree": 1,
                    "structureMissing": False,
                    "relationNeighborCounts": {
                        "coRT": 0,
                        "coURL": 0,
                        "hashSeq": 0,
                        "fastRT": 0,
                        "tweetSim": 1,
                    },
                    "twoHopNodeCount": 1,
                    "relationEvidenceRole": "explanationOnly",
                },
                "evidenceSubgraph": {
                    "depth": 2,
                    "nodeCount": 2,
                    "edgeCount": 1,
                    "truncated": False,
                    "nodes": [
                        {
                            "nodeId": "0",
                            "score": 0.91,
                            "hop": 0,
                            "riskBand": "high",
                            "predictedPositive": True,
                            "structureMissing": False,
                        },
                        {
                            "nodeId": "1",
                            "score": 0.2,
                            "hop": 1,
                            "riskBand": "low",
                            "predictedPositive": False,
                            "structureMissing": False,
                        },
                    ],
                    "edges": [
                        {
                            "id": "0:1",
                            "source": "0",
                            "target": "1",
                            "relations": [
                                {"modality": "tweetSim", "rawWeight": 0.75}
                            ],
                            "evidenceRole": "explanationOnly",
                        }
                    ],
                },
                "limitation": (
                    "Relations and raw weights are explanation-only; model facts are limited "
                    "to the bound frozen prediction artifact."
                ),
            },
            "evidenceHash",
        )


@pytest.mark.anyio
async def test_global_model_unavailable_contract(unconfigured_settings: Settings) -> None:
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get("/api/v1/gfm/global-model/capabilities")
        assert response.status_code == 200
        assert response.json()["servingReady"] is False
        scenario = await client.get("/api/v1/gfm/global-model/scenario")
        assert scenario.status_code == 200
        assert scenario.json()["enabled"] is False
        health = await client.get("/api/v1/gfm/global-model/health")
        assert health.status_code == 200
        assert health.json()["servingReady"] is False
        model_card = await client.get("/api/v1/gfm/global-model/model-card")
        assert model_card.status_code == 503


@pytest.mark.anyio
async def test_global_model_run_evidence_and_review_are_hash_bound(
    unconfigured_settings: Settings,
) -> None:
    fake = FakeGlobalModelClient()
    app = create_app(unconfigured_settings, gfm_global_model_client=fake)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        health = await client.get("/api/v1/gfm/global-model/health")
        assert health.status_code == 200
        assert health.json()["serviceIdentity"] == _health()["serviceIdentity"]
        model_card = await client.get("/api/v1/gfm/global-model/model-card")
        assert model_card.status_code == 200
        assert model_card.json()["modelCardHash"] == _model_card()["modelCardHash"]

        request = {
            "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
            "taskId": "coordination_risk",
            "datasetVersionId": "socialgraph-fm:russia",
            "protocol": "global",
            "modelVersionId": "socialgraph-fm-global/test",
            "topK": 50,
        }
        created = await client.post("/api/v1/gfm/global-model/runs", json=request)
        assert created.status_code == 202
        assert created.json()["runId"] == RUN_ID

        result = await client.get(f"/api/v1/gfm/global-model/runs/{RUN_ID}/result")
        assert result.status_code == 200
        assert result.json()["findings"][0]["routes"][1]["expert"] == "russia"

        evidence = await client.get(
            f"/api/v1/gfm/global-model/runs/{RUN_ID}/nodes/0/evidence"
        )
        assert evidence.status_code == 200
        assert evidence.json()["node"]["nodeId"] == "0"
        assert evidence.json()["resultHash"] == result.json()["resultHash"]
        assert evidence.json()["evidenceSubgraph"]["depth"] == 2
        assert evidence.json()["neighbors"][0]["relations"][0]["rawWeight"] == 0.75

        review = await client.post(
            f"/api/v1/gfm/global-model/runs/{RUN_ID}/reviews",
            json={
                "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
                "nodeId": "0",
                "decision": "pending",
                "reason": "Needs an independent analyst review.",
            },
        )
        assert review.status_code == 201, review.text
        assert review.json()["decision"] == "pending"


@pytest.mark.anyio
async def test_global_model_rejects_non_registered_dataset(
    unconfigured_settings: Settings,
) -> None:
    app = create_app(unconfigured_settings, gfm_global_model_client=FakeGlobalModelClient())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.post(
            "/api/v1/gfm/global-model/runs",
            json={
                "schemaVersion": "socialgraph-fm.gfm-global-model/1.0",
                "taskId": "coordination_risk",
                "datasetVersionId": "uploaded:graph",
                "protocol": "global",
                "modelVersionId": "socialgraph-fm-global/test",
                "topK": 50,
            },
        )
        assert response.status_code == 422
