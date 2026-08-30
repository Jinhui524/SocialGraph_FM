from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.dataset_imports import DatasetImportService
from app.gfm_client import GfmServiceClient
from app.gfm_hashing import canonical_sha256
from app.gfm_core_schemas import (
    MAX_INTERNAL_REQUEST_BYTES,
    MAX_INTERNAL_RESPONSE_BYTES,
    CoreCapabilities,
    CoreCapabilitiesResponse,
    CoreRunRequest,
    CoreRunResult,
    CoreRunStatus,
    CoreFinding,
    CoreInternalErrorEnvelope,
)
from app.main import create_app

from .test_atomic_handoff import _request as graph_handoff_request

HASHES = {letter: letter * 64 for letter in "123456789abcdef"}


def test_language_neutral_run_request_conformance_vectors() -> None:
    vectors = json.loads(
        (Path(__file__).parents[3] / "contracts" / "core-inference-vectors.json").read_text(
            encoding="utf-8"
        )
    )
    for payload in vectors["validRunRequests"]:
        CoreRunRequest.model_validate(payload)
    for payload in vectors["invalidRunRequests"]:
        with pytest.raises(ValueError):
            CoreRunRequest.model_validate(payload)
    assert vectors["limits"] == {
        "maxRequestBytes": MAX_INTERNAL_REQUEST_BYTES,
        "maxResponseBytes": MAX_INTERNAL_RESPONSE_BYTES,
    }
    pairs = (
        (CoreCapabilitiesResponse, "validCapabilities", "invalidCapabilities"),
        (CoreRunStatus, "validStatuses", "invalidStatuses"),
        (CoreFinding, "validFindings", "invalidFindings"),
        (CoreRunResult, "validResults", "invalidResults"),
        (CoreInternalErrorEnvelope, "validErrors", "invalidErrors"),
    )
    for contract, valid_key, invalid_key in pairs:
        for payload in vectors[valid_key]:
            contract.model_validate_json(json.dumps(payload))
        for payload in vectors[invalid_key]:
            with pytest.raises(ValueError):
                contract.model_validate_json(json.dumps(payload))


def test_language_neutral_boundary_limits_accept_at_limit_and_reject_over_limit() -> None:
    vectors = json.loads(
        (Path(__file__).parents[3] / "contracts" / "core-inference-vectors.json").read_text(
            encoding="utf-8"
        )
    )
    limits = vectors["boundaryLimits"]

    def status(run_length: int, error_length: int) -> dict[str, object]:
        value: dict[str, object] = {
            "schemaVersion": "socialgraph-fm.core-run-status/2.0",
            "runId": "r" * run_length,
            "requestHash": "1" * 64,
            "status": "failed",
            "progress": 100,
            "createdAt": "2026-08-14T00:00:00.000000Z",
            "updatedAt": "2026-08-14T00:00:01.000000Z",
            "errorCode": "E" * error_length,
        }
        value["stateHash"] = canonical_sha256(value)
        return value

    CoreRunStatus.model_validate_json(
        json.dumps(status(limits["status.runId"], limits["status.errorCode"]))
    )
    with pytest.raises(ValueError):
        CoreRunStatus.model_validate_json(json.dumps(status(limits["status.runId"] + 1, 1)))
    with pytest.raises(ValueError):
        CoreRunStatus.model_validate_json(json.dumps(status(1, limits["status.errorCode"] + 1)))

    capability = deepcopy(vectors["validCapabilities"][0])
    capability["servingReady"] = True
    capability["readiness"] = {"modelValidated": True, "coreServingReady": True}
    capability["tasks"] = ["core.risk_and_trust_review"]
    task_bindings = [
        {
            "taskId": "core.risk_and_trust_review",
            "entityType": entity_type,
            "confidenceKind": "binary-calibration",
            "calibrationVersion": "calibration/1",
            "method": "sigmoid",
            "calibrationArtifactHash": "4" * 64,
            "calibrationProtocolHash": "5" * 64,
            "adapterDomain": f"risk-{entity_type}",
            "adapterSchemaHash": "6" * 64,
            "adapterStateHash": "7" * 64,
            "featureContractHash": "3" * 64,
        }
        for entity_type in ("node", "edge")
    ]
    capability["models"] = [{
        "modelVersionId": "m" * limits["capabilities.modelVersionId"],
        "modelVersionHash": "2" * 64,
        "state": "servingReady",
        "tasks": ["core.risk_and_trust_review"],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(
            [
                {
                    "taskId": task_binding["taskId"],
                    "entityType": task_binding["entityType"],
                    "featureContractHash": task_binding["featureContractHash"],
                }
                for task_binding in task_bindings
            ]
        ),
        "taskBindings": task_bindings,
        "maxNodes": 1,
        "maxEdges": 1,
    }]
    CoreCapabilitiesResponse.model_validate(capability)
    capability["models"][0]["modelVersionId"] += "x"
    with pytest.raises(ValueError):
        CoreCapabilitiesResponse.model_validate(capability)

    result = deepcopy(vectors["validResults"][0])
    for field, key in (
        ("runId", "result.runId"),
        ("graphVersionId", "result.graphVersionId"),
        ("modelVersionId", "result.modelVersionId"),
    ):
        bounded = deepcopy(result)
        bounded[field] = "x" * limits[key]
        bounded["resultHash"] = canonical_sha256(
            {name: value for name, value in bounded.items() if name != "resultHash"}
        )
        CoreRunResult.model_validate_json(json.dumps(bounded))
        bounded[field] += "x"
        bounded["resultHash"] = canonical_sha256(
            {name: value for name, value in bounded.items() if name != "resultHash"}
        )
        with pytest.raises(ValueError):
            CoreRunResult.model_validate_json(json.dumps(bounded))

    def finding_with_model_version(length: int) -> dict[str, object]:
        finding = deepcopy(vectors["validFindings"][0])
        version = "m" * length
        score = finding["score"]
        score["modelVersion"] = version
        score["scoreHash"] = canonical_sha256(
            {name: value for name, value in score.items() if name != "scoreHash"}
        )
        confidence = finding["calibratedConfidence"]
        confidence["modelVersion"] = version
        confidence["scoreHash"] = score["scoreHash"]
        confidence["confidenceHash"] = canonical_sha256(
            {name: value for name, value in confidence.items() if name != "confidenceHash"}
        )
        evidence = finding["evidence"][0]
        evidence["modelVersion"] = version
        evidence["modelScoreHash"] = score["scoreHash"]
        evidence["evidenceHash"] = canonical_sha256(
            {name: value for name, value in evidence.items() if name != "evidenceHash"}
        )
        finding["modelVersion"] = version
        finding["findingHash"] = canonical_sha256(
            {name: value for name, value in finding.items() if name != "findingHash"}
        )
        return finding

    CoreFinding.model_validate_json(
        json.dumps(finding_with_model_version(limits["finding.score.modelVersion"]))
    )
    with pytest.raises(ValueError):
        CoreFinding.model_validate_json(
            json.dumps(finding_with_model_version(limits["finding.score.modelVersion"] + 1))
        )

    CoreInternalErrorEnvelope.model_validate({"error": {"code": "E" * limits["error.code"]}})
    with pytest.raises(ValueError):
        CoreInternalErrorEnvelope.model_validate(
            {"error": {"code": "E" * (limits["error.code"] + 1)}}
        )


def test_default_registry_mirrors_are_v2_empty_and_byte_identical() -> None:
    repository = Path(__file__).resolve().parents[3]
    api_registry = (
        repository
        / "services"
        / "api"
        / "app"
        / "contracts"
        / "core-serving-registry.json"
    ).read_bytes()
    gfm_registry = (
        repository
        / "packages"
        / "gfm"
        / "contracts"
        / "core-serving-registry.json"
    ).read_bytes()
    assert api_registry == gfm_registry
    assert json.loads(api_registry) == {
        "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
        "generation": 0,
        "models": [],
    }
    for name in (
        "core-serving-control.json",
        "core-serving-graph-catalog.json",
    ):
        assert (
            repository / "services" / "api" / "app" / "contracts" / name
        ).read_bytes() == (
            repository / "packages" / "gfm" / "contracts" / name
        ).read_bytes()


def _public_request() -> dict[str, Any]:
    return {
        "schemaVersion": "socialgraph-fm.core-run-request/2.0",
        "graphVersionId": "graph-v1",
        "taskId": "core.risk_and_trust_review",
        "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
        "modelVersionId": "socialgraph-fm-core/review",
        "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 3},
    }


class FakeGfmClient:
    def __init__(self) -> None:
        self.capability_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "registryHash": HASHES["1"],
            "registryGeneration": 1,
            "servingReady": True,
            "models": [],
            "tasks": ["core.risk_and_trust_review"],
            "readiness": {"modelValidated": True, "coreServingReady": True},
        }
        self.created: dict[str, Any] | None = None
        self.create_payload: dict[str, Any] | None = None
        self.status_payload: dict[str, Any] | None = None
        self.result_payload: dict[str, Any] | None = None

    async def core_capabilities(self) -> dict[str, Any]:
        return self.capability_payload

    async def create_core_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.created = payload
        assert self.create_payload is not None
        return self.create_payload

    async def get_core_run(self, _run_id: str) -> dict[str, Any]:
        assert self.status_payload is not None
        return self.status_payload

    async def get_core_result(self, _run_id: str) -> dict[str, Any]:
        assert self.result_payload is not None
        return self.result_payload


def _seed_graph(settings: Settings) -> None:
    service = DatasetImportService(settings)
    request, _, _ = graph_handoff_request(service, "graph-v1")
    service.commit_graph_handoff(request)


def _with_serving_catalog(
    settings: Settings, *, model_feature_hash: str | None = None
) -> Settings:
    service = DatasetImportService(settings)
    binding = service.store.resolve_graph_version_binding("graph-v1")
    assert binding is not None
    artifact = service.store.get_artifact(binding.artifact_id)
    assert artifact is not None
    feature_contract = {
        "schemaVersion": "socialgraph-fm.core-graph-feature-contract/2.0",
        "nodeFeatures": [],
        "structuralFeatureNames": [],
    }
    catalog = Path(settings.dataset_storage_root) / "serving-graph-catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-graph-catalog/1.0",
                "generation": 1,
                "artifacts": [
                    {
                        "artifactId": binding.artifact_id,
                        "artifactHash": artifact.content_hash,
                        "bundleSha256": HASHES["d"],
                        "relativePath": "bundle.json",
                        "graphVersionId": binding.graph_version_id,
                        "sourceGraphFactHash": binding.graph_fact_hash,
                        "graphVersionHash": HASHES["e"],
                        "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
                        "featureContract": feature_contract,
                        "featureContractHash": canonical_sha256(feature_contract),
                        "nodeCount": artifact.profile.node_count or 0,
                        "edgeCount": artifact.profile.edge_count or 0,
                    }
                ],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    task_heads = [
        {
            "taskId": "core.risk_and_trust_review",
            "kind": "risk-and-trust",
            "nodeOutputIndex": 1,
            "calibrations": [
                {
                    "entityType": entity_type,
                    "confidenceKind": "binary-calibration",
                    "calibrationVersion": "calibration/1",
                    "calibrationMethod": "sigmoid",
                    "calibrationArtifactHash": HASHES["7"],
                    "calibrationRelativePath": "calibration/risk.json",
                    "calibrationSha256": HASHES["8"],
                    "calibrationProtocolHash": HASHES["9"],
                    "adapterDomain": "serving",
                    "adapterSchemaHash": HASHES["d"],
                    "adapterStateHash": HASHES["c"],
                    "graphFeatureContractHash": model_feature_hash
                    or canonical_sha256(feature_contract),
                }
                for entity_type in ("node", "edge")
            ],
        }
    ]
    manifest = {
        "schemaVersion": "socialgraph-fm.core-serving-checkpoint-manifest/1.1",
        "task4CheckpointSha256": HASHES["a"],
        "accepted": True,
        "promotable": True,
        "modelStateHash": HASHES["b"],
        "adapterStateHash": HASHES["c"],
        "adapterSchemaHash": HASHES["d"],
        "adapterDomain": "serving",
        "nodeClasses": 2,
        "multiHotBuckets": 32,
        "adapterBindings": [
            {
                "adapterDomain": "serving",
                "adapterSchemaHash": HASHES["d"],
                "adapterStateHash": HASHES["c"],
                "multiHotBuckets": 32,
            }
        ],
        "taskHeads": task_heads,
    }
    manifest_path = Path(settings.dataset_storage_root) / "model.serving.json"
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    model = {
        "modelVersionId": "socialgraph-fm-core/review",
        "state": "servingReady",
        "checkpoint": {
            "relativePath": "checkpoints/model.pt",
            "sha256": HASHES["a"],
            "servingManifestRelativePath": manifest_path.name,
            "servingManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "bindings": {
                "configHash": HASHES["1"],
                "dataHash": HASHES["2"],
                "codeHash": HASHES["3"],
                "environmentHash": HASHES["4"],
            },
            "adapterDomain": "serving",
            "nodeClasses": 2,
            "multiHotBuckets": 32,
        },
        "taskHeads": task_heads,
        "tasks": ["core.risk_and_trust_review"],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(
            [
                {
                    "taskId": head["taskId"],
                    "entityType": binding["entityType"],
                    "featureContractHash": binding["graphFeatureContractHash"],
                }
                for head in task_heads
                for binding in head["calibrations"]
            ]
        ),
        "maxNodes": 1000,
        "maxEdges": 5000,
    }
    model["modelVersionHash"] = canonical_sha256(
        {key: value for key, value in model.items() if key != "state"}
    )
    registry = Path(settings.dataset_storage_root) / "serving-registry.v1.json"
    registry.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
                "generation": 1,
                "models": [model],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    registry_payload = json.loads(registry.read_bytes())
    catalog_payload = json.loads(catalog.read_bytes())
    control_payload = {
        "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
        "generation": 1,
        "registry": {
            "relativePath": registry.name,
            "sha256": hashlib.sha256(registry.read_bytes()).hexdigest(),
            "semanticHash": canonical_sha256(registry_payload),
            "generation": 1,
        },
        "catalog": {
            "relativePath": catalog.name,
            "sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
            "semanticHash": canonical_sha256(catalog_payload),
            "generation": 1,
        },
    }
    control_payload["controlHash"] = canonical_sha256(control_payload)
    control = Path(settings.dataset_storage_root) / "serving-control.json"
    control.write_text(json.dumps(control_payload, separators=(",", ":")), encoding="utf-8")
    return settings.model_copy(update={"gfm_core_serving_control_file": str(control)})


def _bind_fake_control(fake: FakeGfmClient, settings: Settings) -> None:
    control = json.loads(Path(settings.gfm_core_serving_control_file or "").read_bytes())
    registry = json.loads(
        (Path(settings.gfm_core_serving_control_file or "").parent / control["registry"]["relativePath"])
        .read_bytes()
    )
    models = [
        {
            key: model[key]
            for key in (
                "modelVersionId",
                "modelVersionHash",
                "state",
                "tasks",
                "graphSchemaVersions",
                "graphFeatureContractHash",
                "taskHeads",
                "maxNodes",
                "maxEdges",
            )
        }
        for model in registry["models"]
    ]
    for model in models:
        model["taskBindings"] = [
            {
                "taskId": head["taskId"],
                "entityType": binding["entityType"],
                "confidenceKind": binding["confidenceKind"],
                "calibrationVersion": binding["calibrationVersion"],
                "method": binding["calibrationMethod"],
                "calibrationArtifactHash": binding["calibrationArtifactHash"],
                "calibrationProtocolHash": binding["calibrationProtocolHash"],
                "adapterDomain": binding["adapterDomain"],
                "adapterSchemaHash": binding["adapterSchemaHash"],
                "adapterStateHash": binding["adapterStateHash"],
                "featureContractHash": binding["graphFeatureContractHash"],
            }
            for head in model.pop("taskHeads")
            for binding in head["calibrations"]
        ]
    fake.capability_payload.update(
        {
            "controlHash": control["controlHash"],
            "controlGeneration": control["generation"],
            "registryHash": control["registry"]["semanticHash"],
            "registryGeneration": control["registry"]["generation"],
            "catalogHash": control["catalog"]["semanticHash"],
            "catalogGeneration": control["catalog"]["generation"],
            "models": models,
            "tasks": sorted({task for model in models for task in model["tasks"]}),
            "servingReady": any(model["state"] == "servingReady" for model in models),
            "readiness": {
                "modelValidated": bool(models),
                "coreServingReady": any(
                    model["state"] == "servingReady" for model in models
                ),
            },
        }
    )


def _create_receipt(
    *,
    status: dict[str, Any],
    graph,
    capabilities: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    control_path = Path(settings.gfm_core_serving_control_file or "")
    control = json.loads(control_path.read_bytes())
    registry_path = control_path.parent / control["registry"]["relativePath"]
    catalog_path = control_path.parent / control["catalog"]["relativePath"]
    registry = json.loads(registry_path.read_bytes())
    model = registry["models"][0]
    manifest_path = control_path.parent / model["checkpoint"][
        "servingManifestRelativePath"
    ]
    task_head = next(
        head
        for head in model["taskHeads"]
        if head["taskId"] == "core.risk_and_trust_review"
    )
    calibrations = [
        {
            "entityType": item["entityType"],
            "confidenceKind": item["confidenceKind"],
            "calibrationVersion": item["calibrationVersion"],
            "method": item["calibrationMethod"],
            "calibrationArtifactHash": item["calibrationArtifactHash"],
            "calibrationProtocolHash": item["calibrationProtocolHash"],
            "adapterDomain": item["adapterDomain"],
            "adapterSchemaHash": item["adapterSchemaHash"],
            "adapterStateHash": item["adapterStateHash"],
            "featureContractHash": item["graphFeatureContractHash"],
            "sha256": item["calibrationSha256"],
        }
        for item in sorted(task_head["calibrations"], key=lambda value: value["entityType"])
    ]
    snapshot: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-run-execution-snapshot/2.2",
        "runId": status["runId"],
        "requestHash": status["requestHash"],
        "controlSourceSha256": hashlib.sha256(control_path.read_bytes()).hexdigest(),
        "registryHash": capabilities["registryHash"],
        "registrySourceSha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        "registryGeneration": capabilities["registryGeneration"],
        "controlHash": capabilities["controlHash"],
        "controlGeneration": capabilities["controlGeneration"],
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": capabilities["models"][0]["modelVersionHash"],
        "checkpointSha256": model["checkpoint"]["sha256"],
        "servingManifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "adapterSchemaHash": next(
            item["adapterSchemaHash"]
            for item in calibrations
            if item["entityType"] == "node"
        ),
        "calibrationIdentities": calibrations,
        "calibrationSetHash": canonical_sha256(calibrations),
        "taskId": "core.risk_and_trust_review",
        "graphVersionId": graph.graph_version_id,
        "sourceGraphFactHash": graph.source_graph_fact_hash,
        "graphVersionHash": graph.graph_version_hash,
        "artifactId": graph.artifact_id,
        "artifactHash": graph.artifact_hash,
        "artifactCatalogSha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "artifactCatalogHash": capabilities["catalogHash"],
        "artifactCatalogGeneration": capabilities["catalogGeneration"],
        "bundleSha256": graph.bundle_sha256,
        "graphSchemaVersion": graph.graph_schema_version,
        "featureContractHash": graph.feature_contract_hash,
        "nodeCount": graph.node_count,
        "edgeCount": graph.edge_count,
        "createdAt": status["createdAt"],
    }
    snapshot["snapshotHash"] = canonical_sha256(snapshot)
    receipt = {
        "schemaVersion": "socialgraph-fm.core-internal-create-run-receipt/2.0",
        "status": status,
        "executionSnapshot": snapshot,
        "leaseIdentityHash": canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.core-run-lease-identity/2.2",
                **{
                    key: value
                    for key, value in snapshot.items()
                    if key not in {"schemaVersion", "snapshotHash"}
                },
            }
        ),
    }
    receipt["receiptHash"] = canonical_sha256(receipt)
    return receipt


@pytest.mark.anyio
async def test_default_registry_has_no_product_model_and_valid_run_is_disabled(
    unconfigured_settings: Settings,
) -> None:
    app = create_app(unconfigured_settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        capabilities = await client.get("/api/v1/gfm/capabilities")
        created = await client.post("/api/v1/gfm/runs", json=_public_request())

    assert capabilities.status_code == 200
    assert capabilities.json()["registryGeneration"] == 0
    assert capabilities.json()["models"] == []
    assert capabilities.json()["servingReady"] is False
    assert created.status_code == 503
    assert created.json()["detail"]["code"] == "GFM_CORE_MODEL_NOT_INSTALLED"


@pytest.mark.anyio
async def test_available_model_creates_and_proxies_hash_bound_lifecycle(
    unconfigured_settings: Settings,
) -> None:
    _seed_graph(unconfigured_settings)
    unconfigured_settings = _with_serving_catalog(unconfigured_settings)
    fake = FakeGfmClient()
    _bind_fake_control(fake, unconfigured_settings)
    app = create_app(unconfigured_settings, gfm_client=fake)
    capabilities = CoreCapabilities.model_validate(fake.capability_payload)
    graph = app.state.core_graph_resolver.resolve("graph-v1", capabilities)
    request = CoreRunRequest.model_validate(_public_request())
    envelope = {
        "schemaVersion": "socialgraph-fm.core-internal-create-run/2.1",
        "request": request.model_dump(mode="json", by_alias=True),
        "graphReference": graph.model_dump(mode="json", by_alias=True),
        "expectedServingControl": {
            "controlHash": capabilities.control_hash,
            "controlGeneration": capabilities.control_generation,
            "registryHash": capabilities.registry_hash,
            "registryGeneration": capabilities.registry_generation,
            "catalogHash": capabilities.catalog_hash,
            "catalogGeneration": capabilities.catalog_generation,
            "modelVersionHash": capabilities.models[0].model_version_hash,
        },
    }
    request_hash = app.state.core_gateway.request_hash(envelope)
    fake.status_payload = {
        "schemaVersion": "socialgraph-fm.core-run-status/2.0",
        "runId": "00000000-0000-0000-0000-000000000001",
        "requestHash": request_hash,
        "status": "succeeded",
        "progress": 100,
        "createdAt": "2026-08-14T00:00:00.000000Z",
        "updatedAt": "2026-08-14T00:00:01.000000Z",
        "errorCode": None,
        "stateHash": "",  # filled independently by the API test helper below
    }
    fake.status_payload["stateHash"] = app.state.core_gateway.canonical_hash(
        {key: value for key, value in fake.status_payload.items() if key != "stateHash"}
    )
    fake.create_payload = _create_receipt(
        status=fake.status_payload,
        graph=graph,
        capabilities=fake.capability_payload,
        settings=unconfigured_settings,
    )
    fake.result_payload = {
        "schemaVersion": "socialgraph-fm.core-run-result/2.0",
        "runId": fake.status_payload["runId"],
        "requestHash": request_hash,
        "taskId": "core.risk_and_trust_review",
        "graphVersionId": "graph-v1",
        "graphVersionHash": graph.graph_version_hash,
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": capabilities.models[0].model_version_hash,
        "findings": [],
        "completedAt": "2026-08-14T00:00:01.000000Z",
        "resultHash": "",
    }
    fake.result_payload["resultHash"] = app.state.core_gateway.canonical_hash(
        {key: value for key, value in fake.result_payload.items() if key != "resultHash"}
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        created = await client.post("/api/v1/gfm/runs", json=_public_request())
        status = await client.get(f"/api/v1/gfm/runs/{fake.status_payload['runId']}")
        result = await client.get(
            f"/api/v1/gfm/runs/{fake.status_payload['runId']}/result"
        )

    assert created.status_code == 202
    assert status.status_code == 200
    assert result.status_code == 200
    assert fake.created == envelope
    assert "path" not in json.dumps(fake.created).lower()
    assert result.json()["graphVersionHash"] == graph.graph_version_hash


@pytest.mark.anyio
async def test_api_rejects_unknown_graph_and_model_contract_mismatch_before_forwarding(
    unconfigured_settings: Settings,
) -> None:
    fake = FakeGfmClient()
    task_bindings = [
        {
            "taskId": "core.risk_and_trust_review",
            "entityType": entity_type,
            "confidenceKind": "binary-calibration",
            "calibrationVersion": "calibration/1",
            "method": "sigmoid",
            "calibrationArtifactHash": HASHES["3"],
            "calibrationProtocolHash": HASHES["4"],
            "adapterDomain": f"risk-{entity_type}",
            "adapterSchemaHash": HASHES["5"],
            "adapterStateHash": HASHES["6"],
            "featureContractHash": HASHES["7"],
        }
        for entity_type in ("node", "edge")
    ]
    fake.capability_payload["models"] = [
        {
            "modelVersionId": "socialgraph-fm-core/review",
            "modelVersionHash": HASHES["2"],
            "state": "servingReady",
            "tasks": ["core.risk_and_trust_review"],
            "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
            "graphFeatureContractHash": canonical_sha256(
                [
                    {
                        "taskId": item["taskId"],
                        "entityType": item["entityType"],
                        "featureContractHash": item["featureContractHash"],
                    }
                    for item in task_bindings
                ]
            ),
            "taskBindings": task_bindings,
            "maxNodes": 1000,
            "maxEdges": 5000,
        }
    ]
    app = create_app(unconfigured_settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        missing = await client.post("/api/v1/gfm/runs", json=_public_request())

    assert missing.status_code == 503
    assert missing.json()["detail"]["code"] == "GFM_CORE_SERVING_CONTROL_STALE"
    assert fake.created is None

    _seed_graph(unconfigured_settings)
    unconfigured_settings = _with_serving_catalog(
        unconfigured_settings, model_feature_hash=HASHES["7"]
    )
    _bind_fake_control(fake, unconfigured_settings)
    app = create_app(unconfigured_settings, gfm_client=fake)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        mismatch = await client.post("/api/v1/gfm/runs", json=_public_request())
    assert mismatch.status_code == 409
    assert mismatch.json()["detail"]["code"] == "GFM_MODEL_GRAPH_INCOMPATIBLE"
    assert fake.created is None


def test_service_client_configuration_is_literal_loopback_and_token_is_not_exposed(
    tmp_path: Path,
) -> None:
    token_file = tmp_path / "session.token"
    token_file.write_text("s" * 64, encoding="utf-8")
    with pytest.raises(ValueError, match="literal loopback"):
        GfmServiceClient("http://localhost:7788", token_file=token_file)
    with pytest.raises(ValueError, match="literal loopback"):
        GfmServiceClient("http://127.0.0.1.evil.test:7788", token_file=token_file)
    client = GfmServiceClient("http://127.0.0.1:7788", token_file=token_file)
    assert "s" * 64 not in repr(client)


@pytest.mark.anyio
async def test_api_maps_real_loopback_timeout_to_safe_unavailable(tmp_path: Path) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    token_file = tmp_path / "session.token"
    token_file.write_text("t" * 64, encoding="utf-8")

    def stall() -> None:
        connection, _ = listener.accept()
        try:
            time.sleep(0.25)
        finally:
            connection.close()
            listener.close()

    thread = threading.Thread(target=stall, daemon=True)
    thread.start()
    settings = Settings(
        dataset_storage_root=str(tmp_path / "store"),
        gfm_service_url=f"http://127.0.0.1:{port}",
        gfm_session_token_file=str(token_file),
        gfm_timeout_seconds=0.05,
    )
    app = create_app(settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/v1/gfm/capabilities")
    thread.join(timeout=1)

    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "GFM_CORE_SERVICE_UNAVAILABLE"}}
    assert "t" * 64 not in response.text


@pytest.mark.anyio
async def test_public_run_endpoint_requires_json_content_type(
    api_client: httpx.AsyncClient,
) -> None:
    response = await api_client.post(
        "/api/v1/gfm/runs",
        content=json.dumps(_public_request()),
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "GFM_JSON_REQUIRED"


def test_api_imports_with_torch_and_pyg_blocked() -> None:
    source = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.') or fullname == 'torch_geometric' or fullname.startswith('torch_geometric.'):
            raise ImportError('blocked heavyweight dependency')
        return None
sys.meta_path.insert(0, Block())
import app.main, app.gfm_client, app.gfm_core_schemas
assert not any(name == 'torch' or name.startswith('torch.') or name == 'torch_geometric' or name.startswith('torch_geometric.') for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_strict_public_models_reject_injected_facts_and_tampered_hashes() -> None:
    for key, value in (("score", 0.8), ("evidence", []), ("action", "ban")):
        with pytest.raises(ValueError):
            CoreRunRequest.model_validate(_public_request() | {key: value})
    with pytest.raises(ValueError):
        CoreRunStatus.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-run-status/2.0",
                "runId": "00000000-0000-0000-0000-000000000001",
                "requestHash": HASHES["1"],
                "status": "succeeded",
                "progress": 100,
                "createdAt": "2026-08-14T00:00:00Z",
                "updatedAt": "2026-08-14T00:00:01Z",
                "errorCode": None,
                "stateHash": HASHES["2"],
            }
        )
    with pytest.raises(ValueError):
        CoreRunResult.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-run-result/2.0",
                "runId": "00000000-0000-0000-0000-000000000001",
                "requestHash": HASHES["1"],
                "taskId": "core.risk_and_trust_review",
                "graphVersionId": "graph-v1",
                "graphVersionHash": HASHES["3"],
                "modelVersionId": "socialgraph-fm-core/review",
                "modelVersionHash": HASHES["4"],
                "findings": [],
                "completedAt": "2026-08-14T00:00:01Z",
                "resultHash": HASHES["5"],
            }
        )
