from __future__ import annotations

import hashlib

import pytest

from app.gfm_client import CoreGateway, GfmProxyError
from app.gfm_hashing import canonical_sha256
from app.gfm_core_schemas import (
    CoreAuthorizedGraphReference,
    CoreCapabilities,
    CoreRunRequest,
)

from .test_gfm_core_transport import _serving_snapshot, _CoreServingControlStoreStub


class _ReceiptClient:
    def __init__(self) -> None:
        self.response: dict[str, object] = {}

    async def core_capabilities(self):
        raise AssertionError

    async def create_core_run(self, _payload):
        return self.response

    async def get_core_run(self, _run_id):
        raise AssertionError

    async def get_core_result(self, _run_id):
        raise AssertionError


class _FailingBindingStore:
    def save(self, _binding) -> None:
        raise OSError("injected receipt publication failure")


@pytest.mark.anyio
async def test_receipt_persistence_failure_withholds_public_run_and_counts_safe_diagnostic() -> None:
    request = CoreRunRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-run-request/2.0",
            "graphVersionId": "graph-v1",
            "taskId": "core.risk_and_trust_review",
            "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
            "modelVersionId": "socialgraph-fm-core/review",
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 0},
        }
    )
    graph = CoreAuthorizedGraphReference.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-authorized-graph-reference/2.1",
            "graphVersionId": "graph-v1",
            "sourceGraphFactHash": "1" * 64,
            "graphVersionHash": "2" * 64,
            "artifactId": "artifact-v1",
            "artifactHash": "3" * 64,
            "bundleSha256": "4" * 64,
            "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "featureContractHash": "5" * 64,
            "nodeCount": 1,
            "edgeCount": 0,
        }
    )
    task_bindings = [
        {
            "taskId": "core.risk_and_trust_review",
            "entityType": entity_type,
            "confidenceKind": "binary-calibration",
            "calibrationVersion": "calibration/1",
            "method": "sigmoid",
            "calibrationArtifactHash": "7" * 64,
            "calibrationProtocolHash": "9" * 64,
            "adapterDomain": f"risk-{entity_type}",
            "adapterSchemaHash": "c" * 64,
            "adapterStateHash": "d" * 64,
            "featureContractHash": graph.feature_contract_hash,
        }
        for entity_type in ("node", "edge")
    ]
    capabilities = CoreCapabilities.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "controlHash": "6" * 64,
            "controlGeneration": 3,
            "registryHash": "7" * 64,
            "registryGeneration": 2,
            "catalogHash": "8" * 64,
            "catalogGeneration": 4,
            "servingReady": True,
            "models": [
                {
                    "modelVersionId": "socialgraph-fm-core/review",
                    "modelVersionHash": "9" * 64,
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
                    "maxNodes": 10,
                    "maxEdges": 10,
                }
            ],
            "tasks": ["core.risk_and_trust_review"],
            "readiness": {"modelValidated": True, "coreServingReady": True},
        }
    )
    serving_snapshot = _serving_snapshot(graph, capabilities)
    envelope = {
        "schemaVersion": "socialgraph-fm.core-internal-create-run/2.1",
        "request": request.model_dump(mode="json", by_alias=True),
        "graphReference": graph.model_dump(mode="json", by_alias=True),
        "expectedServingControl": {
            "controlHash": "6" * 64,
            "controlGeneration": 3,
            "registryHash": "7" * 64,
            "registryGeneration": 2,
            "catalogHash": "8" * 64,
            "catalogGeneration": 4,
            "modelVersionHash": "9" * 64,
        },
    }
    status: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-run-status/2.0",
        "runId": "00000000-0000-0000-0000-000000000001",
        "requestHash": canonical_sha256(envelope),
        "status": "queued",
        "progress": 0,
        "createdAt": "2026-08-14T00:00:00.000000Z",
        "updatedAt": "2026-08-14T00:00:00.000000Z",
        "errorCode": None,
    }
    status["stateHash"] = canonical_sha256(status)
    snapshot: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-run-execution-snapshot/2.2",
        "runId": status["runId"],
        "requestHash": status["requestHash"],
        "controlSourceSha256": hashlib.sha256(
            serving_snapshot.control_source_bytes
        ).hexdigest(),
        "registryHash": "7" * 64,
        "registrySourceSha256": "9" * 64,
        "registryGeneration": 2,
        "controlHash": "6" * 64,
        "controlGeneration": 3,
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": "9" * 64,
        "checkpointSha256": "a" * 64,
        "servingManifestSha256": "b" * 64,
        "adapterSchemaHash": "c" * 64,
        "calibrationIdentities": [
            {
                "entityType": binding.entity_type,
                "confidenceKind": binding.confidence_kind,
                "calibrationVersion": binding.calibration_version,
                "method": binding.calibration_method,
                "calibrationArtifactHash": binding.calibration_artifact_hash,
                "calibrationProtocolHash": binding.calibration_protocol_hash,
                "adapterDomain": binding.adapter_domain,
                "adapterSchemaHash": binding.adapter_schema_hash,
                "adapterStateHash": binding.adapter_state_hash,
                "featureContractHash": binding.graph_feature_contract_hash,
                "sha256": binding.calibration_sha256,
            }
            for binding in sorted(
                serving_snapshot.registry.models[0].task_heads[0].calibrations,
                key=lambda item: item.entity_type,
            )
        ],
        "taskId": request.task_id,
        "graphVersionId": graph.graph_version_id,
        "sourceGraphFactHash": graph.source_graph_fact_hash,
        "graphVersionHash": graph.graph_version_hash,
        "artifactId": graph.artifact_id,
        "artifactHash": graph.artifact_hash,
        "artifactCatalogSha256": "e" * 64,
        "artifactCatalogHash": "8" * 64,
        "artifactCatalogGeneration": 4,
        "bundleSha256": graph.bundle_sha256,
        "graphSchemaVersion": graph.graph_schema_version,
        "featureContractHash": graph.feature_contract_hash,
        "nodeCount": graph.node_count,
        "edgeCount": graph.edge_count,
        "createdAt": status["createdAt"],
    }
    snapshot["calibrationSetHash"] = canonical_sha256(
        snapshot["calibrationIdentities"]
    )
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
    client = _ReceiptClient()
    client.response = receipt
    gateway = CoreGateway(
        client,
        serving_control_store=_CoreServingControlStoreStub(serving_snapshot),  # type: ignore[arg-type]
        binding_store=_FailingBindingStore(),  # type: ignore[arg-type]
    )

    with pytest.raises(GfmProxyError) as caught:
        await gateway.create_run(request, graph, capabilities)
    assert caught.value.code == "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED"
    assert gateway.diagnostics() == {
        "code": "GFM_CORE_INTERNAL_ORPHANED_CREATE_COUNT",
        "count": 1,
    }
