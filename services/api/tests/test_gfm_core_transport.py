from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.gfm_client as gfm_client_module
from app.gfm_client import (
    CoreGateway,
    CoreGraphResolver,
    GfmProxyError,
    CoreRunBindingStore,
    GfmServiceClient,
)
from app.gfm_hashing import canonical_sha256
from app.gfm_core_serving_control import CoreServingControlStore
from app.gfm_core_schemas import CoreAuthorizedGraphReference, CoreCapabilities, CoreRunRequest

HASHES = {letter: letter * 64 for letter in "123456789abcdef"}
FEATURE_CONTRACT = {
    "schemaVersion": "socialgraph-fm.core-graph-feature-contract/2.0",
    "nodeFeatures": [{"kind": "numeric", "name": "score"}],
    "structuralFeatureNames": ["degree"],
}


class _Store:
    def __init__(self) -> None:
        self.binding = SimpleNamespace(
            graph_version_id="graph-v1",
            graph_fact_hash=HASHES["1"],
            artifact_id="artifact-v1",
        )
        self.artifact = SimpleNamespace(
            id="artifact-v1",
            source_format="graph-dataset",
            content_hash=HASHES["2"],
            profile=SimpleNamespace(node_count=2, edge_count=1),
        )

    def resolve_graph_version_binding(self, graph_version_id: str):
        return self.binding if graph_version_id == "graph-v1" else None

    def get_artifact(self, artifact_id: str):
        return self.artifact if artifact_id == "artifact-v1" else None


def _catalog_file(tmp_path: Path, **updates: object) -> Path:
    entry: dict[str, object] = {
        "artifactId": "artifact-v1",
        "artifactHash": HASHES["2"],
        "bundleSha256": HASHES["3"],
        "relativePath": "bundle.json",
        "graphVersionId": "graph-v1",
        "sourceGraphFactHash": HASHES["1"],
        "graphVersionHash": HASHES["4"],
        "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "featureContract": FEATURE_CONTRACT,
        "featureContractHash": canonical_sha256(FEATURE_CONTRACT),
        "nodeCount": 2,
        "edgeCount": 1,
    }
    entry.update(updates)
    path = tmp_path / "artifact-catalog.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-graph-catalog/1.0",
                "generation": 1,
                "artifacts": [entry],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return path


def _control_file(tmp_path: Path, catalog_path: Path) -> tuple[Path, CoreCapabilities]:
    registry_path = tmp_path / "registry.v1.json"
    registry_path.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
                "generation": 1,
                "models": [],
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    registry = json.loads(registry_path.read_bytes())
    catalog = json.loads(catalog_path.read_bytes())
    control: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
        "generation": 1,
        "registry": {
            "relativePath": registry_path.name,
            "sha256": hashlib.sha256(registry_path.read_bytes()).hexdigest(),
            "semanticHash": canonical_sha256(registry),
            "generation": 1,
        },
        "catalog": {
            "relativePath": catalog_path.name,
            "sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
            "semanticHash": canonical_sha256(catalog),
            "generation": 1,
        },
    }
    control["controlHash"] = canonical_sha256(control)
    control_path = tmp_path / "serving-control.json"
    control_path.write_text(json.dumps(control, separators=(",", ":")), encoding="utf-8")
    capabilities = CoreCapabilities.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "controlHash": control["controlHash"],
            "controlGeneration": 1,
            "registryHash": control["registry"]["semanticHash"],  # type: ignore[index]
            "registryGeneration": 1,
            "catalogHash": control["catalog"]["semanticHash"],  # type: ignore[index]
            "catalogGeneration": 1,
            "servingReady": False,
            "models": [],
            "tasks": [],
            "readiness": {"modelValidated": False, "coreServingReady": False},
        }
    )
    return control_path, capabilities


def test_api_authorizes_only_catalog_identity_substantiated_by_immutable_store(
    tmp_path: Path,
) -> None:
    control, capabilities = _control_file(tmp_path, _catalog_file(tmp_path))
    resolver = CoreGraphResolver(
        _Store(),
        serving_control_store=CoreServingControlStore(
            control, high_water_root=tmp_path / "high-water-authorized"
        ),
    )

    reference = resolver.resolve("graph-v1", capabilities)

    assert reference.artifact_id == "artifact-v1"
    assert reference.artifact_hash == HASHES["2"]
    assert reference.source_graph_fact_hash == HASHES["1"]
    assert reference.graph_version_hash == HASHES["4"]
    assert reference.bundle_sha256 == HASHES["3"]
    assert reference.feature_contract_hash == canonical_sha256(FEATURE_CONTRACT)


def test_api_rejects_catalog_store_identity_mismatch(tmp_path: Path) -> None:
    catalog = _catalog_file(tmp_path, artifactHash=HASHES["5"])
    control, capabilities = _control_file(tmp_path, catalog)
    mismatch = CoreGraphResolver(
        _Store(),
        serving_control_store=CoreServingControlStore(
            control, high_water_root=tmp_path / "high-water-artifact"
        ),
    )
    with pytest.raises(ValueError, match="catalog|artifact"):
        mismatch.resolve("graph-v1", capabilities)

    catalog = _catalog_file(tmp_path, sourceGraphFactHash=HASHES["6"])
    control, capabilities = _control_file(tmp_path, catalog)
    graph_mismatch = CoreGraphResolver(
        _Store(),
        serving_control_store=CoreServingControlStore(
            control, high_water_root=tmp_path / "high-water-graph"
        ),
    )
    with pytest.raises(ValueError, match="catalog|graph"):
        graph_mismatch.resolve("graph-v1", capabilities)


class _LifecycleClient:
    def __init__(self) -> None:
        self.status: dict[str, object] = {}
        self.result: dict[str, object] = {}
        self.create_response: dict[str, object] = {}

    async def core_capabilities(self):
        raise AssertionError("capabilities are supplied explicitly")

    async def create_core_run(self, _payload):
        return self.create_response

    async def get_core_run(self, _run_id):
        return self.status

    async def get_core_result(self, _run_id):
        return self.result


class _CoreServingControlStoreStub:
    def __init__(self, snapshot: SimpleNamespace) -> None:
        self.snapshot = snapshot

    def acquire(self, required_model_id: str | None = None) -> SimpleNamespace:
        if (
            required_model_id is not None
            and self.snapshot.registry.models[0].model_version_id != required_model_id
        ):
            raise LookupError(required_model_id)
        return self.snapshot


def _serving_snapshot(
    graph: CoreAuthorizedGraphReference, capabilities: CoreCapabilities
) -> SimpleNamespace:
    capability_model = capabilities.models[0]
    calibration_bindings = tuple(
        SimpleNamespace(
            entity_type=entity_type,
            confidence_kind="binary-calibration",
            calibration_version="calibration/1",
            calibration_method="sigmoid",
            calibration_artifact_hash=HASHES["7"],
            calibration_protocol_hash=HASHES["9"],
            calibration_sha256=HASHES["8"],
            adapter_domain=f"risk-{entity_type}",
            adapter_schema_hash=HASHES["c"],
            adapter_state_hash=HASHES["d"],
            graph_feature_contract_hash=graph.feature_contract_hash,
        )
        for entity_type in ("node", "edge")
    )
    model = SimpleNamespace(
        model_version_id=capability_model.model_version_id,
        model_version_hash=capability_model.model_version_hash,
        state=capability_model.state,
        tasks=capability_model.tasks,
        graph_schema_versions=capability_model.graph_schema_versions,
        graph_feature_contract_hash=capability_model.graph_feature_contract_hash,
        max_nodes=capability_model.max_nodes,
        max_edges=capability_model.max_edges,
        checkpoint=SimpleNamespace(sha256=HASHES["a"]),
        task_heads=(
            SimpleNamespace(
                task_id="core.risk_and_trust_review",
                calibrations=calibration_bindings,
            ),
        ),
    )
    manifest = SimpleNamespace(
        source_sha256=HASHES["b"],
        manifest=SimpleNamespace(adapter_schema_hash=HASHES["c"]),
    )
    snapshot = SimpleNamespace(
        control_source_bytes=b"round-1-serving-control",
        control=SimpleNamespace(
            control_hash=capabilities.control_hash,
            generation=capabilities.control_generation,
            registry=SimpleNamespace(semantic_hash=capabilities.registry_hash),
            catalog=SimpleNamespace(semantic_hash=capabilities.catalog_hash),
        ),
        registry_hash=capabilities.registry_hash,
        registry_source_sha256=HASHES["9"],
        registry=SimpleNamespace(
            generation=capabilities.registry_generation, models=(model,)
        ),
        catalog_hash=capabilities.catalog_hash,
        catalog_source_sha256=HASHES["e"],
        catalog=SimpleNamespace(
            generation=capabilities.catalog_generation,
            artifacts=(
                SimpleNamespace(
                    graph_version_id=graph.graph_version_id,
                    source_graph_fact_hash=graph.source_graph_fact_hash,
                    graph_version_hash=graph.graph_version_hash,
                    artifact_id=graph.artifact_id,
                    artifact_hash=graph.artifact_hash,
                    bundle_sha256=graph.bundle_sha256,
                    graph_schema_version=graph.graph_schema_version,
                    feature_contract_hash=graph.feature_contract_hash,
                    node_count=graph.node_count,
                    edge_count=graph.edge_count,
                ),
            ),
        ),
    )
    snapshot.model = lambda _model_version_id: model
    snapshot.manifest = lambda _model_version_id: manifest
    return snapshot


@pytest.mark.anyio
async def test_api_persists_original_binding_and_rejects_coherent_result_substitution(
    tmp_path: Path,
) -> None:
    client = _LifecycleClient()
    binding_store = CoreRunBindingStore(tmp_path / "api-run-bindings")
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
            "sourceGraphFactHash": HASHES["1"],
            "graphVersionHash": HASHES["4"],
            "artifactId": "artifact-v1",
            "artifactHash": HASHES["2"],
            "bundleSha256": HASHES["3"],
            "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "featureContractHash": canonical_sha256(FEATURE_CONTRACT),
            "nodeCount": 2,
            "edgeCount": 1,
        }
    )
    task_bindings = [
        {
            "taskId": "core.risk_and_trust_review",
            "entityType": entity_type,
            "confidenceKind": "binary-calibration",
            "calibrationVersion": "calibration/1",
            "method": "sigmoid",
            "calibrationArtifactHash": HASHES["7"],
            "calibrationProtocolHash": HASHES["9"],
            "adapterDomain": f"risk-{entity_type}",
            "adapterSchemaHash": HASHES["c"],
            "adapterStateHash": HASHES["d"],
            "featureContractHash": graph.feature_contract_hash,
        }
        for entity_type in ("node", "edge")
    ]
    capabilities = CoreCapabilities.model_validate(
        {
                "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
                "controlHash": HASHES["5"],
                "controlGeneration": 4,
                "registryHash": HASHES["7"],
                "registryGeneration": 3,
                "catalogHash": HASHES["6"],
                "catalogGeneration": 2,
            "servingReady": True,
            "models": [
                {
                    "modelVersionId": "socialgraph-fm-core/review",
                    "modelVersionHash": HASHES["8"],
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
                    "maxNodes": 100,
                    "maxEdges": 100,
                }
            ],
            "tasks": ["core.risk_and_trust_review"],
            "readiness": {"modelValidated": True, "coreServingReady": True},
        }
    )
    serving_snapshot = _serving_snapshot(graph, capabilities)
    gateway = CoreGateway(
        client,
        serving_control_store=_CoreServingControlStoreStub(serving_snapshot),  # type: ignore[arg-type]
        binding_store=binding_store,
    )
    envelope = {
        "schemaVersion": "socialgraph-fm.core-internal-create-run/2.1",
        "request": request.model_dump(mode="json", by_alias=True),
        "graphReference": graph.model_dump(mode="json", by_alias=True),
        "expectedServingControl": {
            "controlHash": HASHES["5"],
            "controlGeneration": 4,
            "registryHash": HASHES["7"],
            "registryGeneration": 3,
            "catalogHash": HASHES["6"],
            "catalogGeneration": 2,
            "modelVersionHash": HASHES["8"],
        },
    }
    request_hash = canonical_sha256(envelope)
    run_id = "00000000-0000-0000-0000-000000000001"
    client.status = {
        "schemaVersion": "socialgraph-fm.core-run-status/2.0",
        "runId": run_id,
        "requestHash": request_hash,
        "status": "succeeded",
        "progress": 100,
        "createdAt": "2026-08-14T00:00:00.000000Z",
        "updatedAt": "2026-08-14T00:00:01.000000Z",
        "errorCode": None,
    }
    client.status["stateHash"] = canonical_sha256(client.status)
    snapshot: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-run-execution-snapshot/2.2",
        "runId": run_id,
        "requestHash": request_hash,
        "controlSourceSha256": hashlib.sha256(
            serving_snapshot.control_source_bytes
        ).hexdigest(),
        "registryHash": HASHES["7"],
        "registrySourceSha256": HASHES["9"],
        "registryGeneration": 3,
        "controlHash": HASHES["5"],
        "controlGeneration": 4,
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": HASHES["8"],
        "checkpointSha256": HASHES["a"],
        "servingManifestSha256": HASHES["b"],
        "adapterSchemaHash": HASHES["c"],
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
        "artifactCatalogSha256": HASHES["e"],
        "artifactCatalogHash": HASHES["6"],
        "artifactCatalogGeneration": 2,
        "bundleSha256": graph.bundle_sha256,
        "graphSchemaVersion": graph.graph_schema_version,
        "featureContractHash": graph.feature_contract_hash,
        "nodeCount": graph.node_count,
        "edgeCount": graph.edge_count,
        "createdAt": "2026-08-14T00:00:00.000000Z",
    }
    snapshot["calibrationSetHash"] = canonical_sha256(
        snapshot["calibrationIdentities"]
    )
    snapshot["snapshotHash"] = canonical_sha256(snapshot)
    client.create_response = {
        "schemaVersion": "socialgraph-fm.core-internal-create-run-receipt/2.0",
        "status": client.status,
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
    client.create_response["receiptHash"] = canonical_sha256(client.create_response)
    await gateway.create_run(request, graph, capabilities)

    persisted = binding_store.get(run_id)
    concurrent_store = CoreRunBindingStore(tmp_path / "concurrent-bindings")
    barrier = threading.Barrier(8)

    def publish_same_binding() -> None:
        barrier.wait(timeout=5)
        concurrent_store.save(persisted)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(publish_same_binding) for _ in range(8)]
        for future in futures:
            future.result(timeout=5)
    assert concurrent_store.get(run_id) == persisted

    restarted = CoreGateway(client, binding_store=CoreRunBindingStore(tmp_path / "api-run-bindings"))
    assert (await restarted.get_run(run_id)).request_hash == request_hash
    client.result = {
        "schemaVersion": "socialgraph-fm.core-run-result/2.0",
        "runId": run_id,
        "requestHash": request_hash,
        "taskId": "core.risk_and_trust_review",
        "graphVersionId": "graph-v1",
        "graphVersionHash": HASHES["9"],
        "modelVersionId": "substituted/1",
        "modelVersionHash": HASHES["a"],
        "findings": [],
        "completedAt": "2026-08-14T00:00:01.000000Z",
    }
    client.result["resultHash"] = canonical_sha256(client.result)
    with pytest.raises(Exception, match="BINDING"):
        await restarted.get_result(run_id)


@pytest.mark.anyio
async def test_service_client_rejects_declared_size_before_body_read(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(gfm_client_module.MAX_GFM_RESPONSE_BYTES + 1))
            self.end_headers()
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("x" * 64, encoding="utf-8")
    client = GfmServiceClient(
        f"http://127.0.0.1:{server.server_address[1]}", token_file=token
    )
    try:
        with pytest.raises(GfmProxyError) as raised:
            await client.core_capabilities()
        assert raised.value.code == "GFM_CORE_RESPONSE_TOO_LARGE"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.anyio
async def test_service_client_aborts_stream_when_accumulated_bytes_exceed_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = 0
    total = 64 * 1024

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            return

        def do_GET(self):
            nonlocal sent
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            for _ in range(total // 1024):
                try:
                    self.wfile.write(b"x" * 1024)
                    self.wfile.flush()
                    sent += 1024
                    time.sleep(0.005)
                except (BrokenPipeError, ConnectionResetError):
                    break

    monkeypatch.setattr(gfm_client_module, "MAX_GFM_RESPONSE_BYTES", 1024)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    token = tmp_path / "token"
    token.write_text("x" * 64, encoding="utf-8")
    client = GfmServiceClient(
        f"http://127.0.0.1:{server.server_address[1]}", token_file=token
    )
    try:
        with pytest.raises(GfmProxyError) as raised:
            await client.core_capabilities()
        assert raised.value.code == "GFM_CORE_RESPONSE_TOO_LARGE"
        thread.join(timeout=2)
        assert sent < total
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
