from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import socialgraph_gfm.research.service as research_service
from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.core.adapters import (
    BundleInputAdapter,
    derive_training_selection,
)
from socialgraph_gfm.core.bundle import (
    CoreGraphBundle,
    calculate_graph_version_hash,
)
from socialgraph_gfm.core.inference_service import create_server
from socialgraph_gfm.core.model import EdgeHead, SymmetricEdgeHead
from socialgraph_gfm.research.contracts import RESEARCH_TASK_IDS
from socialgraph_gfm.research.service import (
    ResearchInvalid,
    ResearchNotFound,
    ResearchRegistrationFailed,
    ResearchServiceError,
    ResearchServingRuntime,
    ResearchUnavailable,
    _array_sha256,
    _canonical_graph_hash_from_arrays,
    _dataset_content_hash,
    _graph_fact_hash_from_arrays,
    _uploaded_bundle,
)
from socialgraph_gfm.research.wire import (
    WIRE_SCHEMA,
    WireRunEnvelope,
    capabilities_payload,
    model_capability,
    scenarios_payload,
)
from socialgraph_gfm.research.workflow import (
    evaluate_research_model,
    export_research_model,
    materialize_fixture_corpus,
    publish_research_model,
    smoke_research_export,
    train_research_comparison_matrix,
    train_research_model,
)
from socialgraph_gfm.core.structure_features import (
    STRUCTURE_FEATURE_NAMES,
    StructureAlgorithmConfig,
    compute_structure_rows,
)


def test_signed_relation_review_ranks_opposition_not_support() -> None:
    support_logits = torch.tensor([-2.0, 2.0])
    scores, calibrated = ResearchServingRuntime._review_scores(
        support_logits,
        {"adequate": False, "bias": 0.0, "temperature": 1.0},
        "research.signed_relation_review",
    )

    assert calibrated is False
    assert float(scores[0]) > float(scores[1])


def _registry(*, graph_hash: str = "1" * 64) -> dict[str, object]:
    return {
        "modelVersionId": "socialgraph-fm-research/test",
        "modelVersionHash": "a" * 64,
        "artifactHash": "b" * 64,
        "taskIds": list(RESEARCH_TASK_IDS),
        "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "maxNodes": 50_000,
        "maxEdges": 1_500_000,
        "claimStatus": "not_demonstrated",
        "scenarios": [
            {
                "scenarioId": "email-eu-collaboration",
                "datasetId": "email-eu-core",
                "domain": "email-eu-core",
                "taskId": "core.collaboration_completion",
                "route": "shared-null",
                "graphVersionId": "research:email-eu-core",
                "graphVersionHash": graph_hash,
            }
        ],
        "embeddings": [],
    }


def _array_role(name: str) -> str:
    if name == "edge_index":
        return "edge_index"
    if name == "node_id_map":
        return "node_id_map"
    return "auxiliary"


def _dataset_artifact(
    root: Path,
    *,
    node_count: int = 20,
    edges: list[tuple[int, int]] | None = None,
    node_prefix: str = "n",
    graph_version_id: str = "uploaded-test",
) -> tuple[dict[str, object], dict[str, np.ndarray], dict[str, object]]:
    if edges is None:
        edges = [(index, (index + 1) % node_count) for index in range(node_count)]
    artifact_id = str(uuid.uuid4())
    edge_index = np.asarray(edges, dtype="<i8").T
    arrays = {
        "directed": np.asarray(False, dtype=np.bool_),
        "edge_attributes_json": np.asarray(["{}"] * len(edges), dtype=np.str_),
        "edge_directed": np.asarray([0] * len(edges), dtype=np.int8),
        "edge_id_map": np.asarray([f"e{index}" for index in range(len(edges))]),
        "edge_index": edge_index,
        "edge_timestamp": np.asarray([""] * len(edges), dtype=np.str_),
        "edge_type": np.asarray([""] * len(edges), dtype=np.str_),
        "edge_weight": np.asarray([np.nan] * len(edges), dtype=np.float64),
        "node_attributes_json": np.asarray(["{}"] * node_count, dtype=np.str_),
        "node_id_map": np.asarray(
            [f"{node_prefix}{index}" for index in range(node_count)]
        ),
        "node_label": np.asarray([f"Node {index}" for index in range(node_count)]),
        "node_type": np.asarray([""] * node_count, dtype=np.str_),
        "num_nodes": np.asarray(node_count, dtype=np.int64),
    }
    descriptors = [
        {
            "name": name,
            "role": _array_role(name),
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    ]
    artifact: dict[str, object] = {
        "schemaVersion": "2.2",
        "id": artifact_id,
        "sourceFormat": "graph_version_target_domain",
        "profile": {"nodeCount": node_count, "edgeCount": len(edges)},
        "arrays": descriptors,
        "nodeIdentity": {
            "id": "node-identity-v1",
            "arrayName": "node_id_map",
            "kind": "source",
            "count": node_count,
            "unique": True,
        },
        "graphSemantics": {
            "directed": False,
            "directedness": "undirected",
            "edgeDirectedArray": "edge_directed",
            "edgeStorage": "coo",
            "selfLoopPolicy": "preserve",
            "duplicateEdgePolicy": "preserve",
            "weighted": False,
            "temporal": False,
            "heterogeneous": False,
        },
        "graphVariants": [
            {
                "id": "raw",
                "edgeIndexArray": "edge_index",
                "featureArray": None,
                "directed": False,
            }
        ],
        "featureSchemas": [],
        "labelSchemas": [],
        "featureRecipes": [
            {
                "id": "identity-v1",
                "graphVariant": "raw",
                "inputArray": None,
                "outputArray": None,
                "featureTransform": "identity",
                "fitScope": "none",
                "parameters": {},
            }
        ],
        "splitSets": [],
        "taskSpecs": [],
    }
    content_hash = _dataset_content_hash(artifact, arrays)
    graph_hash = _canonical_graph_hash_from_arrays(
        node_count=node_count, directed=False, edge_index=edge_index
    )
    artifact["contentHash"] = content_hash
    artifact["canonicalGraphHash"] = graph_hash
    artifact["rawManifest"] = {
        "graphVersionHandoff": {
            "graphVersionId": graph_version_id,
            "graphFactHash": "0" * 64,
        }
    }
    fact_hash = _graph_fact_hash_from_arrays(
        artifact=artifact, arrays=arrays, edge_index=edge_index
    )
    artifact["rawManifest"]["graphVersionHandoff"]["graphFactHash"] = fact_hash  # type: ignore[index]
    target = root / "artifacts" / artifact_id
    target.mkdir(parents=True)
    np.savez_compressed(target / "graph.npz", **arrays)
    (target / "artifact.json").write_text(
        json.dumps(artifact, ensure_ascii=False), encoding="utf-8"
    )
    reference = {
        "kind": "uploaded-artifact",
        "graphVersionId": graph_version_id,
        "graphVersionHash": graph_hash,
        "graphFactHash": fact_hash,
        "artifactId": artifact_id,
        "artifactHash": content_hash,
        "nodeCount": node_count,
        "edgeCount": len(edges),
    }
    return artifact, arrays, reference


def _run_envelope(registry: dict[str, object], graph_hash: str) -> dict[str, object]:
    return {
        "schemaVersion": WIRE_SCHEMA,
        "request": {
            "schemaVersion": WIRE_SCHEMA,
            "graphVersionId": "research:email-eu-core",
            "taskId": "core.collaboration_completion",
            "modelVersionId": registry["modelVersionId"],
            "targetScope": {
                "kind": "collaboration-candidates",
                "anchorNodeId": "n0",
                "topK": 5,
            },
            "scenarioId": "email-eu-collaboration",
            "parameters": {"candidateLimit": 3},
        },
        "graphReference": {
            "kind": "registered-scenario",
            "graphVersionId": "research:email-eu-core",
            "graphVersionHash": graph_hash,
            "nodeCount": 0,
            "edgeCount": 0,
        },
        "expectedModel": model_capability(registry),
    }


def test_wire_payloads_validate_against_api_contract(tmp_path: Path) -> None:
    api_root = Path(__file__).resolve().parents[3] / "services" / "api"
    sys.path.insert(0, str(api_root))
    try:
        from app.gfm_research_schemas import (  # type: ignore[import-not-found]
            ResearchCapabilities,
            ResearchScenariosResponse,
        )

        ResearchCapabilities.model_validate(capabilities_payload(tmp_path))
        ResearchScenariosResponse.model_validate(scenarios_payload(tmp_path))
    finally:
        sys.path.remove(str(api_root))


def test_utc_timestamp_keeps_microseconds_when_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FrozenDateTime:
        @classmethod
        def now(cls, timezone):
            assert timezone is UTC
            return datetime(2026, 8, 16, 12, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(research_service, "datetime", FrozenDateTime)
    assert research_service._utc_now() == "2026-08-16T12:00:00.000000Z"


def test_loopback_listener_authenticates_and_dispatches_research_routes() -> None:
    class ResearchStub:
        def close(self) -> None:
            return None

        def dispatch_get(self, path: str) -> dict[str, object]:
            if path != "/internal/research/capabilities":
                raise ResearchNotFound()
            return {"schemaVersion": WIRE_SCHEMA, "channel": "research"}

        def dispatch_post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
            assert path == "/internal/research/runs"
            assert payload == {"request": "bound"}
            return {"schemaVersion": WIRE_SCHEMA, "status": "queued"}

    token = "session-" + "x" * 64
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=SimpleNamespace(),
        research_runtime=ResearchStub(),  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request("GET", "/internal/research/capabilities")
        assert connection.getresponse().status == 401
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request(
            "GET",
            "/internal/research/capabilities",
            headers={"Authorization": f"Bearer {token}"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["channel"] == "research"
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request(
            "POST",
            "/internal/research/runs",
            body=json.dumps({"request": "bound"}),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 202
        assert json.loads(response.read())["status"] == "queued"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_uploaded_artifact_recomputes_all_identities_and_confines_paths(
    tmp_path: Path,
) -> None:
    store = tmp_path / "dataset-store"
    artifact, arrays, reference = _dataset_artifact(store)
    runtime = ResearchServingRuntime(tmp_path / "research", store)
    observed, arrays = runtime._artifact(reference)
    assert observed["contentHash"] == reference["artifactHash"]
    assert len(arrays) == 13
    for field in ("artifactHash", "graphVersionHash", "graphFactHash"):
        with pytest.raises(ResearchInvalid):
            runtime._artifact(reference | {field: "f" * 64})
    with pytest.raises(ResearchInvalid):
        runtime._artifact(reference | {"artifactId": "../escape"})

    arrays["num_nodes"] = np.asarray(19, dtype=np.int64)
    descriptor = next(item for item in artifact["arrays"] if item["name"] == "num_nodes")  # type: ignore[index]
    descriptor["sha256"] = _array_sha256(arrays["num_nodes"])
    artifact["contentHash"] = _dataset_content_hash(artifact, arrays)
    reference["artifactHash"] = artifact["contentHash"]
    target = store / "artifacts" / str(reference["artifactId"])
    np.savez_compressed(target / "graph.npz", **arrays)
    (target / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ResearchInvalid, match="semantics differ"):
        runtime._artifact(reference)


def test_registration_rejects_graphs_with_fewer_than_ten_nonedges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    all_pairs = [(left, right) for left in range(20) for right in range(left + 1, 20)]
    store = tmp_path / "dataset-store"
    _artifact, _arrays, reference = _dataset_artifact(store, edges=all_pairs[:-8])
    runtime = ResearchServingRuntime(tmp_path / "research", store)
    registry = _registry(graph_hash=str(reference["graphVersionHash"]))
    monkeypatch.setattr(runtime, "_published", lambda: (registry, {}))
    monkeypatch.setattr(
        runtime,
        "_model_runtime",
        lambda: (registry, {}, {}, {}, {}, SimpleNamespace(), {}),
    )
    payload = {
        "schemaVersion": WIRE_SCHEMA,
        "graphReference": reference,
        "compatibleTaskIds": ["core.collaboration_completion"],
        "auxiliaryCapabilities": ["similar-nodes"],
        "expectedModel": model_capability(registry),
    }
    assert runtime.register_graph(payload)["adapterStatus"] == "pending_registration"
    deadline = time.monotonic() + 5
    while True:
        try:
            runtime.register_graph(payload)
        except ResearchRegistrationFailed as error:
            assert error.code == "GFM_RESEARCH_REGISTRATION_FAILED"
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    runtime.close()


def test_registration_is_async_and_cache_binds_node_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "dataset-store"
    artifact_a, arrays_a, reference_a = _dataset_artifact(
        store, node_prefix="a", graph_version_id="uploaded-a"
    )
    artifact_b, arrays_b, reference_b = _dataset_artifact(
        store, node_prefix="b", graph_version_id="uploaded-b"
    )
    paired_edges = [
        pair
        for index in range(20)
        for pair in ((index, (index + 1) % 20), ((index + 1) % 20, index))
    ]
    artifact_p, arrays_p, reference_p = _dataset_artifact(
        store,
        edges=paired_edges,
        node_prefix="p",
        graph_version_id="uploaded-paired-coo",
    )
    duplicate_edges = [
        (index, (index + 1) % 20) for index in range(20)
    ] + [(0, 1)]
    artifact_d, arrays_d, reference_d = _dataset_artifact(
        store,
        edges=duplicate_edges,
        node_prefix="d",
        graph_version_id="uploaded-duplicate-coo",
    )
    assert reference_a["graphVersionHash"] == reference_b["graphVersionHash"]
    assert reference_a["graphFactHash"] != reference_b["graphFactHash"]
    bundle_a, _metadata = _uploaded_bundle(
        artifact=artifact_a, arrays=arrays_a, graph_reference=reference_a
    )
    bundle_b, _metadata = _uploaded_bundle(
        artifact=artifact_b, arrays=arrays_b, graph_reference=reference_b
    )
    bundle_p, paired_metadata = _uploaded_bundle(
        artifact=artifact_p, arrays=arrays_p, graph_reference=reference_p
    )
    _bundle_d, duplicate_metadata = _uploaded_bundle(
        artifact=artifact_d, arrays=arrays_d, graph_reference=reference_d
    )
    assert bundle_a.graph_version_hash != bundle_b.graph_version_hash
    assert paired_metadata["rawEdgeCount"] == 40
    assert paired_metadata["semanticEdgeCount"] == len(bundle_p.edges) == 20
    assert paired_metadata["selfLoopCount"] == 0
    assert paired_metadata["duplicateCount"] == 0
    assert not paired_metadata["directed"]
    assert duplicate_metadata["duplicateCount"] == 1
    adapter = BundleInputAdapter(bundle_a, mode="training", multi_hot_buckets=256)
    checkpoint = {"adapterStates": {"email-eu-core": adapter.state_dict()}}
    registry = _registry(graph_hash=str(reference_a["graphVersionHash"]))
    gate = threading.Event()

    class Model:
        def eval(self) -> None:
            return None

        def encode_domain(self, features, _edge_index, _prompt):
            gate.wait(timeout=5)
            return torch.zeros((features.shape[0], 128), dtype=features.dtype)

    runtime = ResearchServingRuntime(tmp_path / "research", store)
    monkeypatch.setattr(runtime, "_published", lambda: (registry, {}))
    monkeypatch.setattr(
        runtime,
        "_model_runtime",
        lambda: (registry, {}, checkpoint, {}, {}, Model(), {}),
    )

    def payload(reference: dict[str, object]) -> dict[str, object]:
        return {
            "schemaVersion": WIRE_SCHEMA,
            "graphReference": reference,
            "compatibleTaskIds": ["core.collaboration_completion"],
            "auxiliaryCapabilities": ["similar-nodes"],
            "expectedModel": model_capability(registry),
        }

    envelope_a = payload(reference_a)
    started_at = time.monotonic()
    first = runtime.register_graph(envelope_a)
    assert time.monotonic() - started_at < 0.5
    assert first["adapterStatus"] == "pending_registration"
    assert set(first) == {
        "schemaVersion",
        "graphVersionId",
        "graphVersionHash",
        "modelVersionId",
        "modelVersionHash",
        "adapterStatus",
        "compatibleTaskIds",
        "auxiliaryCapabilities",
        "registrationHash",
    }
    assert first["registrationHash"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "registrationHash"}
    )
    assert runtime.register_graph(envelope_a) == first
    gate.set()

    def wait_ready(envelope: dict[str, object]) -> dict[str, object]:
        deadline = time.monotonic() + 5
        while True:
            response = runtime.register_graph(envelope)
            if response["adapterStatus"] == "ready":
                return response
            assert time.monotonic() < deadline
            time.sleep(0.01)

    ready_a = wait_ready(envelope_a)
    ready_b = wait_ready(payload(reference_b))
    ready_p = wait_ready(payload(reference_p))
    assert ready_p["adapterStatus"] == "ready"
    assert runtime.register_graph(payload(reference_d))["adapterStatus"] == (
        "pending_registration"
    )
    deadline = time.monotonic() + 5
    while True:
        try:
            runtime.register_graph(payload(reference_d))
        except ResearchRegistrationFailed:
            break
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert ready_a["registrationHash"] != ready_b["registrationHash"]
    manifest_a = runtime._uploaded_manifest("uploaded-a")
    manifest_b = runtime._uploaded_manifest("uploaded-b")
    assert manifest_a["cacheKey"] != manifest_b["cacheKey"]
    assert manifest_a["graphReference"]["graphFactHash"] != (
        manifest_b["graphReference"]["graphFactHash"]
    )

    for reference, node_id in ((reference_a, "a0"), (reference_b, "b0")):
        response = runtime.similar_nodes(
            {
                "schemaVersion": WIRE_SCHEMA,
                "request": {
                    "schemaVersion": WIRE_SCHEMA,
                    "graphVersionId": reference["graphVersionId"],
                    "nodeId": node_id,
                    "topK": 3,
                    "modelVersionId": registry["modelVersionId"],
                },
                "graphReference": reference,
                "expectedModel": model_capability(registry),
            }
        )
        assert response["graphVersionId"] == reference["graphVersionId"]
        assert response["nodeId"] == node_id
    runtime.close()


def test_run_state_machine_is_deterministic_and_recovers_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph_hash = "1" * 64
    registry = _registry(graph_hash=graph_hash)
    runtime = ResearchServingRuntime(tmp_path / "research", None)
    monkeypatch.setattr(runtime, "_published", lambda: (registry, {}))
    gate = threading.Event()

    def result(run_id: str, envelope: WireRunEnvelope) -> dict[str, object]:
        gate.wait(timeout=5)
        payload: dict[str, object] = {
            "schemaVersion": WIRE_SCHEMA,
            "runId": run_id,
            "requestHash": envelope.request.request_hash,
            "taskId": envelope.request.task_id,
            "graphVersionId": envelope.graph_reference.graph_version_id,
            "graphVersionHash": envelope.graph_reference.graph_version_hash,
            "modelVersionId": registry["modelVersionId"],
            "modelVersionHash": registry["modelVersionHash"],
            "seed": 1729,
            "preliminary": True,
            "calibrationStatus": "ranking_only",
            "findings": [],
            "completedAt": "2026-08-16T00:00:00Z",
        }
        payload["resultHash"] = canonical_sha256(payload)
        return payload

    monkeypatch.setattr(runtime, "_result", result)
    envelope = _run_envelope(registry, graph_hash)
    first = runtime.create_run(envelope)
    second = runtime.create_run(envelope)
    assert first["status"] == "queued"
    assert second["runId"] == first["runId"]
    deadline = time.monotonic() + 5
    while runtime.get_run(str(first["runId"]))["status"] == "queued":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert runtime.get_run(str(first["runId"]))["status"] == "running"
    gate.set()
    while runtime.get_run(str(first["runId"]))["status"] != "succeeded":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert runtime.get_result(str(first["runId"]))["requestHash"] == first["requestHash"]

    failed = ResearchServingRuntime(tmp_path / "failed", None)
    monkeypatch.setattr(failed, "_published", lambda: (registry, {}))
    monkeypatch.setattr(
        failed,
        "_result",
        lambda *_args: (_ for _ in ()).throw(ResearchInvalid("bad graph")),
    )
    failed_status = failed.create_run(envelope)
    deadline = time.monotonic() + 5
    while failed.get_run(str(failed_status["runId"]))["status"] != "failed":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert failed.get_run(str(failed_status["runId"]))["errorCode"] == (
        "GFM_RESEARCH_REQUEST_INVALID"
    )
    runtime.close()
    failed.close()


def test_runtime_document_read_retries_only_transient_atomic_open_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = ResearchServingRuntime(tmp_path / "research", None)
    run_id = f"research-{'a' * 32}"
    status_path = tmp_path / "research/serving/runs" / run_id / "status.json"
    status_path.parent.mkdir(parents=True)
    status = runtime._status_payload(
        run_id=run_id,
        request_hash="1" * 64,
        status="succeeded",
        created_at="2026-08-16T00:00:00Z",
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    original_read = research_service._read_hashed_document
    transient_calls = 0

    def transient_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal transient_calls
        transient_calls += 1
        if transient_calls < 3:
            raise PermissionError("simulated Windows sharing violation")
        return original_read(*args, **kwargs)

    monkeypatch.setattr(research_service, "_read_hashed_document", transient_read)
    assert runtime.get_run(run_id) == status
    assert transient_calls == 3

    tampered = dict(status)
    tampered["progress"] = 99
    status_path.write_text(json.dumps(tampered), encoding="utf-8")
    integrity_calls = 0

    def integrity_read(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal integrity_calls
        integrity_calls += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(research_service, "_read_hashed_document", integrity_read)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        runtime.get_run(run_id)
    assert integrity_calls == 1
    runtime.close()


def test_uploaded_inference_is_ranking_only_and_scores_every_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = tmp_path / "dataset-store"
    artifact, arrays, reference = _dataset_artifact(store)
    bundle, _metadata = _uploaded_bundle(
        artifact=artifact, arrays=arrays, graph_reference=reference
    )
    registry = _registry(graph_hash=str(reference["graphVersionHash"]))

    class Model:
        def __init__(self) -> None:
            self.seen = 0

        def eval(self) -> None:
            return None

        def collaboration_head(self, _encoded, pairs):
            self.seen += len(pairs)
            return pairs[:, 1].to(torch.float32) / 100

    model = Model()
    runtime = ResearchServingRuntime(tmp_path / "research", store)
    checkpoint = {
        "calibrators": {
            "core.collaboration_completion": {
                "adequate": True,
                "bias": 0.0,
                "temperature": 1.0,
            }
        }
    }
    monkeypatch.setattr(
        runtime,
        "_graph_runtime",
        lambda *_args: (
            registry,
            checkpoint,
            model,
            bundle,
            np.zeros((len(bundle.nodes), 128), dtype="<f4"),
            None,
        ),
    )
    request = {
        "schemaVersion": WIRE_SCHEMA,
        "request": {
            "schemaVersion": WIRE_SCHEMA,
            "graphVersionId": reference["graphVersionId"],
            "taskId": "core.collaboration_completion",
            "modelVersionId": registry["modelVersionId"],
            "targetScope": {
                "kind": "collaboration-candidates",
                "anchorNodeId": "n0",
                "topK": 5,
            },
            "parameters": {"candidateLimit": 3},
        },
        "graphReference": reference,
        "expectedModel": model_capability(registry),
    }
    result = runtime._result("research-" + "1" * 32, WireRunEnvelope.model_validate(request))
    assert model.seen > 3
    assert result["calibrationStatus"] == "ranking_only"
    assert len(result["findings"]) == 3
    assert all(not item["calibrated"] for item in result["findings"])
    assert result["findings"][0]["entityIds"][1] == "n9"

    request["request"]["targetScope"]["topK"] = 100  # type: ignore[index]
    request["request"]["parameters"]["candidateLimit"] = 100  # type: ignore[index]
    complete = runtime._result(
        "research-" + "4" * 32, WireRunEnvelope.model_validate(request)
    )
    observed_targets = {
        item["entityIds"][1] for item in complete["findings"]
    }
    assert observed_targets == {f"n{index}" for index in range(2, 19)}


def test_registered_email_candidates_exclude_only_training_visible_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, arrays, reference = _dataset_artifact(tmp_path / "dataset-store")
    base_bundle, _metadata = _uploaded_bundle(
        artifact=artifact, arrays=arrays, graph_reference=reference
    )
    payload = base_bundle.model_dump(mode="json", by_alias=True)
    assignments = []
    held_out_id = "edge:n0:n1"
    for edge in base_bundle.edges:
        edge_id = f"edge:{edge.source_id}:{edge.target_id}"
        assignments.append(
            {
                "entityId": edge_id,
                "role": "validation" if edge_id == held_out_id else "train",
            }
        )
    assert any(item["role"] == "validation" for item in assignments)
    payload["splitManifest"] = {
        "strategy": "spanning-forest-80-10-10",
        "assignments": assignments,
    }
    payload["structuralFeatures"] = None
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    preliminary = CoreGraphBundle.model_validate(payload)
    selection = derive_training_selection(preliminary)
    held_out_index = next(
        index
        for index, edge in enumerate(preliminary.edges)
        if f"edge:{edge.source_id}:{edge.target_id}" == held_out_id
    )
    assert held_out_index not in selection.visible_edge_indices
    rows = compute_structure_rows(
        preliminary,
        visible_edge_indices=selection.visible_edge_indices,
        config=StructureAlgorithmConfig.fixed(),
    )
    payload["structuralFeatures"] = {
        "names": list(STRUCTURE_FEATURE_NAMES),
        "values": rows.tolist(),
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    bundle = CoreGraphBundle.model_validate(payload)

    class Model:
        def eval(self) -> None:
            return None

        def collaboration_head(self, _encoded, pairs):
            return pairs[:, 1].to(torch.float32) / 100

    registry = _registry(graph_hash=bundle.graph_version_hash)
    registry["taskCalibrationStatus"] = {
        "core.collaboration_completion": "ranking_only"
    }
    checkpoint = {
        "calibrators": {
            "core.collaboration_completion": {
                "adequate": False,
                "bias": 0.0,
                "temperature": 1.0,
            }
        }
    }
    runtime = ResearchServingRuntime(tmp_path / "research", None)
    monkeypatch.setattr(
        runtime,
        "_graph_runtime",
        lambda *_args: (
            registry,
            checkpoint,
            Model(),
            bundle,
            np.zeros((len(bundle.nodes), 128), dtype="<f4"),
            "email-eu-core",
        ),
    )
    envelope = WireRunEnvelope.model_validate(
        {
            "schemaVersion": WIRE_SCHEMA,
            "request": {
                "schemaVersion": WIRE_SCHEMA,
                "graphVersionId": "research:email-eu-core",
                "taskId": "core.collaboration_completion",
                "modelVersionId": registry["modelVersionId"],
                "targetScope": {
                    "kind": "collaboration-candidates",
                    "anchorNodeId": "n0",
                    "topK": 100,
                },
                "scenarioId": "email-eu-collaboration",
                "parameters": {"candidateLimit": 100},
            },
            "graphReference": {
                "kind": "registered-scenario",
                "graphVersionId": "research:email-eu-core",
                "graphVersionHash": bundle.graph_version_hash,
                "nodeCount": 0,
                "edgeCount": 0,
            },
            "expectedModel": model_capability(registry),
        }
    )
    result = runtime._result("research-" + "5" * 32, envelope)
    targets = {item["entityIds"][1] for item in result["findings"]}
    assert "n1" in targets
    train_visible_neighbors = {
        endpoint
        for index in selection.visible_edge_indices
        for endpoint in (
            bundle.edges[index].source_id,
            bundle.edges[index].target_id,
        )
        if "n0" in {
            bundle.edges[index].source_id,
            bundle.edges[index].target_id,
        }
        and endpoint != "n0"
    }
    assert train_visible_neighbors
    assert targets.isdisjoint(train_visible_neighbors)
    runtime.close()


def test_registered_calibration_requires_published_aggregate_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, arrays, _uploaded_reference = _dataset_artifact(
        tmp_path / "dataset-store"
    )
    graph_reference = {
        "kind": "registered-scenario",
        "graphVersionId": "research:tolokers",
        "graphVersionHash": "4" * 64,
        "nodeCount": 0,
        "edgeCount": 0,
    }
    bundle, _metadata = _uploaded_bundle(
        artifact=artifact,
        arrays=arrays,
        graph_reference={**_uploaded_reference, "graphVersionHash": "4" * 64},
    )
    registry = _registry(graph_hash="4" * 64)
    registry["taskCalibrationStatus"] = {
        "research.account_risk_review": "ranking_only"
    }

    class Model:
        def eval(self) -> None:
            return None

        def account_risk_head(self, encoded):
            return torch.ones(encoded.shape[0])

    checkpoint = {
        "calibrators": {
            "research.account_risk_review": {
                "adequate": True,
                "bias": 0.0,
                "temperature": 1.0,
            }
        }
    }
    runtime = ResearchServingRuntime(tmp_path / "research", None)
    monkeypatch.setattr(
        runtime,
        "_graph_runtime",
        lambda *_args: (
            registry,
            checkpoint,
            Model(),
            bundle,
            np.zeros((len(bundle.nodes), 128), dtype="<f4"),
            "tolokers",
        ),
    )
    envelope = WireRunEnvelope.model_validate(
        {
            "schemaVersion": WIRE_SCHEMA,
            "request": {
                "schemaVersion": WIRE_SCHEMA,
                "graphVersionId": "research:tolokers",
                "taskId": "research.account_risk_review",
                "modelVersionId": registry["modelVersionId"],
                "targetScope": {"kind": "nodes", "nodeIds": ["n0"]},
                "scenarioId": "tolokers-account-risk",
                "parameters": {"candidateLimit": 5},
            },
            "graphReference": graph_reference,
            "expectedModel": model_capability(registry),
        }
    )
    preliminary = runtime._result("research-" + "2" * 32, envelope)
    assert preliminary["calibrationStatus"] == "ranking_only"
    assert not preliminary["findings"][0]["calibrated"]

    registry["taskCalibrationStatus"]["research.account_risk_review"] = (  # type: ignore[index]
        "calibrated"
    )
    calibrated = runtime._result("research-" + "3" * 32, envelope)
    assert calibrated["calibrationStatus"] == "calibrated"
    assert calibrated["findings"][0]["calibrated"]
    runtime.close()


def test_email_head_is_symmetric_and_wiki_head_preserves_endpoint_order() -> None:
    encoded = torch.zeros((2, 128))
    encoded[0, 0] = 2
    encoded[1, 0] = 1
    pairs = torch.tensor([[0, 1], [1, 0]])
    symmetric = SymmetricEdgeHead()
    assert torch.equal(symmetric(encoded, pairs[:1]), symmetric(encoded, pairs[1:]))

    directed = EdgeHead()
    with torch.no_grad():
        first = directed.network[0]
        last = directed.network[2]
        first.weight.zero_()
        first.bias.zero_()
        first.weight[0, 0] = 1
        first.weight[0, 128] = -1
        last.weight.zero_()
        last.bias.zero_()
        last.weight[0, 0] = 1
    values = directed(encoded, pairs)
    assert values[0] != values[1]


def test_registered_email_scenario_cache_uses_hash_bound_shared_null_route(
    tmp_path: Path,
) -> None:
    artifact, arrays, reference = _dataset_artifact(tmp_path / "dataset-store")
    bundle, _metadata = _uploaded_bundle(
        artifact=artifact, arrays=arrays, graph_reference=reference
    )
    adapter = BundleInputAdapter(bundle, mode="training", multi_hot_buckets=256)
    seen_routes: list[str | None] = []

    class Model:
        def eval(self) -> None:
            return None

        def encode_domain(self, _features, _edge_index, domain):
            seen_routes.append(domain)
            return torch.zeros((len(bundle.nodes), 128), dtype=torch.float32)

    registry = _registry(graph_hash=bundle.graph_version_hash)
    export = {"checkpointSha256": "c" * 64}
    scenario = registry["scenarios"][0]
    runtime = ResearchServingRuntime(tmp_path / "research", None)
    first = runtime._scenario_head_embeddings(
        registry=registry,
        export=export,
        scenario=scenario,
        bundle=bundle,
        model=Model(),
        adapter=adapter,
        domain="email-eu-core",
    )
    second = runtime._scenario_head_embeddings(
        registry=registry,
        export=export,
        scenario=scenario,
        bundle=bundle,
        model=Model(),
        adapter=adapter,
        domain="email-eu-core",
    )
    assert seen_routes == [None]
    assert np.array_equal(first, second)
    manifest_path = next((tmp_path / "research/serving/e").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["route"] == "shared-null"
    runtime.close()


def test_preview_and_twitch_embeddings_are_hash_bound_and_domain_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preview: dict[str, object] = {
        "schemaVersion": WIRE_SCHEMA,
        "scenarioId": "scenario",
        "graphVersionId": "research:twitch-language",
        "graphVersionHash": "1" * 64,
        "modelVersionId": "model",
        "modelVersionHash": "2" * 64,
        "nodes": [],
        "edges": [],
        "partialPreview": True,
        "nodeCount": 20,
        "edgeCount": 20,
    }
    preview["previewHash"] = canonical_sha256(preview)
    preview_path = tmp_path / "exports/research/previews/scenario.json"
    preview_path.parent.mkdir(parents=True)
    preview_path.write_text(json.dumps(preview), encoding="utf-8")
    registry = {
        "modelVersionId": "model",
        "modelVersionHash": "2" * 64,
        "exportManifestPath": "exports/research/export-manifest.json",
        "scenarios": [
            {
                "scenarioId": "scenario",
                "graphVersionHash": "1" * 64,
                "previewPath": "previews/scenario.json",
                "previewSha256": file_sha256(preview_path),
            }
        ],
    }
    runtime = ResearchServingRuntime(tmp_path, None)
    monkeypatch.setattr(runtime, "_published", lambda: (registry, {}))
    assert runtime.graph_preview("scenario")["previewHash"] == preview["previewHash"]
    preview_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchServiceError, match="preview file identity"):
        runtime.graph_preview("scenario")


def test_registered_scenario_rejects_registry_bundle_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, arrays, reference = _dataset_artifact(tmp_path / "dataset-store")
    bundle, _metadata = _uploaded_bundle(
        artifact=artifact, arrays=arrays, graph_reference=reference
    )
    registry = _registry(graph_hash="5" * 64)
    registry["scenarios"][0]["domain"] = "email-eu-core"  # type: ignore[index]
    documents = {"email-eu-core": (bundle, {}, {"datasetFamily": "email-eu-core"})}
    runtime = ResearchServingRuntime(tmp_path / "research", None)
    monkeypatch.setattr(
        runtime,
        "_model_runtime",
        lambda: (registry, {}, {}, {}, documents, SimpleNamespace(), {}),
    )
    with pytest.raises(ResearchServiceError, match="bundle identity mismatch"):
        runtime._graph_runtime(
            {
                "kind": "registered-scenario",
                "graphVersionId": "research:email-eu-core",
                "graphVersionHash": "5" * 64,
            },
            "email-eu-collaboration",
            "core.collaboration_completion",
        )
    runtime.close()


def test_twitch_similarity_uses_unique_domain_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, arrays, reference = _dataset_artifact(tmp_path / "dataset-store")
    bundle, _metadata = _uploaded_bundle(
        artifact=artifact, arrays=arrays, graph_reference=reference
    )
    registry = _registry(graph_hash=bundle.graph_version_hash)
    registry["scenarios"] = [
        {
            "scenarioId": "twitch-content-policy",
            "datasetId": "twitch-language",
            "domain": "twitch-EN",
            "taskId": "research.content_policy_review",
            "graphVersionId": "research:twitch-language",
            "graphVersionHash": bundle.graph_version_hash,
        }
    ]
    values = np.zeros((len(bundle.nodes), 128), dtype="<f4")
    values[0, 0] = 1
    entries = []
    for domain in ("twitch-DE", "twitch-EN"):
        path = tmp_path / "exports/research/embeddings" / f"{domain}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=values,
            node_ids=np.asarray([node.id for node in bundle.nodes]),
        )
        entries.append(
            {
                "domain": domain,
                "route": "shared-null",
                "path": f"embeddings/{domain}.npz",
                "sha256": file_sha256(path),
            }
        )
    registry["embeddings"] = entries
    documents = {
        domain: (bundle, {}, {"datasetFamily": "twitch-language"})
        for domain in ("twitch-DE", "twitch-EN")
    }
    runtime = ResearchServingRuntime(tmp_path, None)
    monkeypatch.setattr(runtime, "_published", lambda: (registry, {}))
    monkeypatch.setattr(
        runtime,
        "_model_runtime",
        lambda: (registry, {}, {}, {}, documents, None, {}),
    )
    sources = list(runtime._embedding_sources())
    assert {item[0] for item in sources} == {
        "research:twitch-language:DE",
        "research:twitch-language:EN",
    }
    response = runtime.similar_nodes(
        {
            "schemaVersion": WIRE_SCHEMA,
            "request": {
                "schemaVersion": WIRE_SCHEMA,
                "graphVersionId": "research:twitch-language",
                "nodeId": bundle.nodes[0].id,
                "topK": 3,
                "modelVersionId": registry["modelVersionId"],
            },
            "graphReference": {
                "kind": "registered-scenario",
                "graphVersionId": "research:twitch-language",
                "graphVersionHash": bundle.graph_version_hash,
                "nodeCount": 0,
                "edgeCount": 0,
            },
            "expectedModel": model_capability(registry),
        }
    )
    assert response["matches"][0]["graphVersionId"] == (
        "research:twitch-language:DE"
    )
    assert response["matches"][0]["datasetId"] == "twitch-language:DE"
    assert all(
        item["graphVersionId"] != "research:twitch-language:EN"
        for item in response["matches"]
    )


def test_published_fixture_serves_four_real_scenarios_in_fresh_process(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    root = tmp_path_factory.mktemp("sr1")
    fixtures = Path(__file__).parent / "fixtures/core_datasets"
    materialize_fixture_corpus(root, fixtures)
    train_research_model(root, device="cpu", pretrain_epochs=1, head_epochs=1)
    train_research_comparison_matrix(
        root, device="cpu", pretrain_epochs=1, downstream_epochs=1
    )
    evaluate_research_model(root, device="cpu")
    export_research_model(root, allow_test_fixture=True)
    smoke_research_export(root, allow_test_fixture=True)
    publish_research_model(root, allow_test_fixture=True)

    blocked = ResearchServingRuntime(root, None)
    with pytest.raises(ResearchUnavailable, match="test fixture exports"):
        blocked.capabilities()
    blocked.close()

    # Serving must not read the mutable training checkpoint after export.
    with (root / "runs/shared/checkpoint.pt").open("ab") as stream:
        stream.write(b"training-source-tamper")

    script = """
import json
import sys
import time
from socialgraph_gfm.research.service import ResearchServingRuntime
from socialgraph_gfm.research.wire import WIRE_SCHEMA

runtime = ResearchServingRuntime(sys.argv[1], None)
capabilities = runtime.capabilities()
results = []
for scenario in runtime.scenarios()["scenarios"]:
    request = {
        "schemaVersion": WIRE_SCHEMA,
        "graphVersionId": scenario["graphVersionId"],
        "taskId": scenario["taskId"],
        "modelVersionId": capabilities["model"]["modelVersionId"],
        "targetScope": scenario["defaultTargetScope"],
        "scenarioId": scenario["scenarioId"],
        "parameters": {"candidateLimit": 20},
    }
    envelope = {
        "schemaVersion": WIRE_SCHEMA,
        "request": request,
        "graphReference": {
            "kind": "registered-scenario",
            "graphVersionId": scenario["graphVersionId"],
            "graphVersionHash": scenario["graphVersionHash"],
            "nodeCount": 0,
            "edgeCount": 0,
        },
        "expectedModel": capabilities["model"],
    }
    status = runtime.create_run(envelope)
    deadline = time.monotonic() + 30
    while status["status"] not in {"succeeded", "failed"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(status["runId"])
        time.sleep(0.02)
        status = runtime.get_run(status["runId"])
    if status["status"] != "succeeded":
        raise RuntimeError(status)
    result = runtime.get_result(status["runId"])
    results.append(
        [scenario["scenarioId"], len(result["findings"]), result["resultHash"]]
    )
runtime.close()
print(json.dumps(results))
"""
    environment = dict(os.environ)
    package_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = str(package_root / "src")
    environment["SOCIALGRAPH_FM_INTERNAL_TEST_FIXTURE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        cwd=package_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    observed = json.loads(completed.stdout)
    assert [item[0] for item in observed] == [
        "twitch-content-policy",
        "tolokers-account-risk",
        "wiki-rfa-signed-relation",
        "email-eu-collaboration",
    ]
    assert all(item[1] > 0 for item in observed)

    with (root / "exports/research/checkpoint.pt").open("ab") as stream:
        stream.write(b"export-tamper")
    rejected = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from socialgraph_gfm.research.service "
                "import ResearchServingRuntime; "
                "ResearchServingRuntime(sys.argv[1], None)._model_runtime()"
            ),
            str(root),
        ],
        cwd=package_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert rejected.returncode != 0
    assert "checkpoint hash mismatch" in rejected.stderr
