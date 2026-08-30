from __future__ import annotations

import http.client
import json
import os
import subprocess
import shutil
import threading
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from socialgraph_gfm.core.governance import GovernanceFinding
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.inference_contracts import (
    GfmCapabilities,
    GfmRunRequest,
    GfmRunResult,
    GfmRunStatus,
    InternalCreateRunRequest,
    InternalErrorEnvelope,
    MAX_INTERNAL_REQUEST_BYTES,
    MAX_INTERNAL_RESPONSE_BYTES,
)
from socialgraph_gfm.core.inference_service import (
    InferenceRuntime,
    RunStore,
    atomic_publish_session_token,
    create_server,
)
from socialgraph_gfm.core.serving_registry import ServingRegistry

from test_core_inference_fix_round1 import (
    _catalog as _round1_catalog,
    _serving_registry as _round1_serving_registry,
)
from _core_inference_test_support import (
    _make_test_internal_create_request,
    _make_test_only_run_store,
    _make_test_serving_control,
)


HASHES = {letter: letter * 64 for letter in "123456789abcdef"}


def test_language_neutral_run_request_conformance_vectors() -> None:
    vectors = json.loads(
        (Path(__file__).parents[3] / "contracts" / "core-inference-vectors.json").read_text(
            encoding="utf-8"
        )
    )
    for payload in vectors["validRunRequests"]:
        GfmRunRequest.model_validate(payload)
    for payload in vectors["invalidRunRequests"]:
        with pytest.raises(ValidationError):
            GfmRunRequest.model_validate(payload)
    assert vectors["limits"] == {
        "maxRequestBytes": MAX_INTERNAL_REQUEST_BYTES,
        "maxResponseBytes": MAX_INTERNAL_RESPONSE_BYTES,
    }
    pairs = (
        (GfmCapabilities, "validCapabilities", "invalidCapabilities"),
        (GfmRunStatus, "validStatuses", "invalidStatuses"),
        (GovernanceFinding, "validFindings", "invalidFindings"),
        (GfmRunResult, "validResults", "invalidResults"),
        (InternalErrorEnvelope, "validErrors", "invalidErrors"),
    )
    for contract, valid_key, invalid_key in pairs:
        for payload in vectors[valid_key]:
            contract.model_validate_json(json.dumps(payload))
        for payload in vectors[invalid_key]:
            with pytest.raises(ValidationError):
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

    GfmRunStatus.model_validate_json(
        json.dumps(status(limits["status.runId"], limits["status.errorCode"]))
    )
    with pytest.raises(ValidationError):
        GfmRunStatus.model_validate_json(json.dumps(status(limits["status.runId"] + 1, 1)))
    with pytest.raises(ValidationError):
        GfmRunStatus.model_validate_json(json.dumps(status(1, limits["status.errorCode"] + 1)))

    capability = deepcopy(vectors["validCapabilities"][0])
    capability["servingReady"] = True
    capability["readiness"] = {"modelValidated": True, "coreServingReady": True}
    capability["tasks"] = ["core.risk_and_trust_review"]
    task_bindings = [
        {
            "taskId": "core.risk_and_trust_review",
            "entityType": entity,
            "confidenceKind": "binary-calibration",
            "calibrationVersion": "calibration-v1",
            "method": "sigmoid",
            "calibrationArtifactHash": "4" * 64,
            "calibrationProtocolHash": "5" * 64,
            "adapterDomain": f"adapter-{entity}",
            "adapterSchemaHash": "6" * 64,
            "adapterStateHash": "7" * 64,
            "featureContractHash": "3" * 64,
        }
        for entity in ("node", "edge")
    ]
    feature_inventory = [
        {
            "taskId": item["taskId"],
            "entityType": item["entityType"],
            "featureContractHash": item["featureContractHash"],
        }
        for item in task_bindings
    ]
    capability["models"] = [
        {
            "modelVersionId": "m" * limits["capabilities.modelVersionId"],
            "modelVersionHash": "2" * 64,
            "state": "servingReady",
            "tasks": ["core.risk_and_trust_review"],
            "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
            "graphFeatureContractHash": canonical_sha256(feature_inventory),
            "taskBindings": task_bindings,
            "maxNodes": 1,
            "maxEdges": 1,
        }
    ]
    GfmCapabilities.model_validate(capability)
    capability["models"][0]["modelVersionId"] += "x"
    with pytest.raises(ValidationError):
        GfmCapabilities.model_validate(capability)

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
        GfmRunResult.model_validate_json(json.dumps(bounded))
        bounded[field] += "x"
        bounded["resultHash"] = canonical_sha256(
            {name: value for name, value in bounded.items() if name != "resultHash"}
        )
        with pytest.raises(ValidationError):
            GfmRunResult.model_validate_json(json.dumps(bounded))

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

    GovernanceFinding.model_validate_json(
        json.dumps(finding_with_model_version(limits["finding.score.modelVersion"]))
    )
    with pytest.raises(ValidationError):
        GovernanceFinding.model_validate_json(
            json.dumps(finding_with_model_version(limits["finding.score.modelVersion"] + 1))
        )

    InternalErrorEnvelope.model_validate({"error": {"code": "E" * limits["error.code"]}})
    with pytest.raises(ValidationError):
        InternalErrorEnvelope.model_validate({"error": {"code": "E" * (limits["error.code"] + 1)}})


def _registry(tmp_path: Path, *, serving: bool) -> ServingRegistry:
    if serving:
        catalog, reference, _ = _round1_catalog(tmp_path)
        registry = _round1_serving_registry(tmp_path, catalog=catalog)
        registry._test_artifact_catalog = catalog  # type: ignore[attr-defined]
        registry._test_reference = reference  # type: ignore[attr-defined]
        return registry
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": "socialgraph-fm.core-serving-registry/2.0",
        "generation": 1,
        "models": [],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    return ServingRegistry.load(registry_path, runtime_root=tmp_path)


def _request() -> dict:
    return {
        "schemaVersion": "socialgraph-fm.core-run-request/2.0",
        "graphVersionId": "graph-v1",
        "taskId": "core.risk_and_trust_review",
        "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
        "modelVersionId": "socialgraph-fm-core/review",
        "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 3},
    }


def _internal_request(reference, control) -> InternalCreateRunRequest:
    return _make_test_internal_create_request(reference, control)


def _finding(model_version_hash: str, graph) -> dict:
    # A complete pre-hashed GovernanceFinding v2 fixture is generated by the
    # established Task-5 constructors in the executor, not authored by the service.
    from socialgraph_gfm.core.governance import (
        CalibratedConfidence,
        ModelScore,
        analyze_community_resilience,
        create_governance_finding,
    )

    score = ModelScore.create(
        task_id="core.risk_and_trust_review",
        entity_type="node",
        entity_ids=("a",),
        score=0.75,
        graph_version_hash=graph.graph_version_hash,
        model_version="socialgraph-fm-core/review",
        model_version_hash=model_version_hash,
    )
    confidence = CalibratedConfidence.create(
        score=score,
        value=0.6,
        calibration_version="calibration/1",
        method="isotonic",
        calibration_artifact_hash=HASHES["a"],
        calibration_protocol_hash=HASHES["b"],
    )
    finding = create_governance_finding(
        task_id="core.risk_and_trust_review",
        finding_type="node-risk-candidate",
        subject_ids=("a",),
        score=score,
        calibrated_confidence=confidence,
        evidence=analyze_community_resilience(graph)[:1],
        similar_cases=(),
        limitations=("Candidate for review; it is not a risk or trust truth label.",),
    )
    return finding.model_dump(mode="json", by_alias=True)


def test_gfm_run_request_v2_rejects_extra_coercion_and_task_injected_fields() -> None:
    assert GfmRunRequest.model_validate(_request()).graph_version_id == "graph-v1"
    for mutation in (
        {"score": 0.99},
        {"evidence": [{"ref": "caller"}]},
        {"automaticAction": "ban"},
    ):
        payload = _request() | mutation
        with pytest.raises(ValidationError):
            GfmRunRequest.model_validate(payload)
    coerced = _request()
    coerced["parameters"] = {"kind": "risk-and-trust", "topKSimilarCases": "3"}
    with pytest.raises(ValidationError):
        GfmRunRequest.model_validate(coerced)


def test_registry_is_the_only_capability_source_and_rejects_checkpoint_tamper(
    tmp_path: Path,
) -> None:
    empty = _registry(tmp_path / "empty", serving=False)
    assert empty.capabilities()["servingReady"] is False
    assert empty.capabilities()["models"] == []

    serving = _registry(tmp_path / "serving", serving=True)
    assert serving.capabilities()["servingReady"] is True
    checkpoint = serving.runtime_root / "checkpoints" / "model.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        serving.resolve_model("socialgraph-fm-core/review")


def test_registry_generation_update_changes_readiness_without_stale_status_files(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path, serving=False)
    assert registry.capabilities()["servingReady"] is False
    serving = _registry(tmp_path / "source", serving=True)
    updated = json.loads(serving.path.read_text(encoding="utf-8"))
    registry.path.write_text(json.dumps(updated), encoding="utf-8")
    shutil.copytree(serving.runtime_root / "checkpoints", tmp_path / "checkpoints")
    shutil.copytree(serving.runtime_root / "calibration", tmp_path / "calibration")

    capabilities = registry.capabilities()
    assert capabilities["registryGeneration"] == 1
    assert capabilities["servingReady"] is True
    assert capabilities["readiness"] == {
        "modelValidated": True,
        "coreServingReady": True,
    }


def test_run_store_persists_immutable_bound_results_and_recovers_restart(tmp_path: Path) -> None:
    registry = _registry(tmp_path, serving=True)
    catalog = registry._test_artifact_catalog  # type: ignore[attr-defined]
    control = _make_test_serving_control(tmp_path, registry, catalog)
    store = _make_test_only_run_store(
        tmp_path / "runtime",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda _request, graph, model: [
            _finding(
                model.model_version_hash,
                registry._test_artifact_catalog.resolve(graph),  # type: ignore[attr-defined]
            )
        ],
    )
    created = store.create(_internal_request(registry._test_reference, control))  # type: ignore[attr-defined]
    run_id = created.status.run_id
    deadline = time.monotonic() + 5
    while store.get(run_id).status not in {"succeeded", "failed"}:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    completed = store.get(run_id)
    assert completed.status == "succeeded"
    result = store.get_result(run_id)
    assert result.graph_version_hash == registry._test_reference.graph_version_hash  # type: ignore[attr-defined]
    assert result.model_version_hash == registry.resolve_model("socialgraph-fm-core/review").model_version_hash
    assert result.findings[0].review_status == "pending-human-review"

    recovered = _make_test_only_run_store(
        tmp_path / "runtime",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: pytest.fail("terminal run must not be repeated"),
    )
    assert recovered.get(run_id) == completed
    assert recovered.get_result(run_id).result_hash == result.result_hash

    result_path = tmp_path / "runtime" / "runs" / run_id / "result.json"
    original_result = result_path.read_text(encoding="utf-8")
    tampered_result = json.loads(original_result)
    tampered_result["resultHash"] = HASHES["f"]
    result_path.write_text(json.dumps(tampered_result), encoding="utf-8")
    with pytest.raises(ValueError, match="result hash"):
        recovered.get_result(run_id)
    result_path.write_text(original_result, encoding="utf-8")

    state_path = tmp_path / "runtime" / "runs" / run_id / "state.json"
    tampered = json.loads(state_path.read_text(encoding="utf-8"))
    tampered["progress"] = 1
    state_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="progress|state hash"):
        recovered.get(run_id)


def test_result_is_unavailable_before_completion_and_interrupted_run_fails_on_restart(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path, serving=True)
    catalog = registry._test_artifact_catalog  # type: ignore[attr-defined]
    control = _make_test_serving_control(tmp_path, registry, catalog)
    gate = threading.Event()
    store = _make_test_only_run_store(
        tmp_path / "runtime",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda _request, graph, model: (
            gate.wait(5),
            [
                _finding(
                    model.model_version_hash,
                    registry._test_artifact_catalog.resolve(graph),  # type: ignore[attr-defined]
                )
            ],
        )[1],
    )
    run = store.create(_internal_request(registry._test_reference, control))  # type: ignore[attr-defined]
    with pytest.raises(LookupError, match="not ready"):
        store.get_result(run.status.run_id)

    restarted = _make_test_only_run_store(
        tmp_path / "runtime",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda *_args: pytest.fail("interrupted run must not be resumed"),
    )
    recovered = restarted.get(run.status.run_id)
    assert recovered.status == "failed"
    assert recovered.error_code == "GFM_CORE_RUN_INTERRUPTED"
    gate.set()


def test_concurrent_duplicate_requests_create_unique_immutable_runs(tmp_path: Path) -> None:
    registry = _registry(tmp_path, serving=True)
    catalog = registry._test_artifact_catalog  # type: ignore[attr-defined]
    control = _make_test_serving_control(tmp_path, registry, catalog)
    store = _make_test_only_run_store(
        tmp_path / "runtime",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
        executor=lambda _request, graph, model: [
            _finding(
                model.model_version_hash,
                registry._test_artifact_catalog.resolve(graph),  # type: ignore[attr-defined]
            )
        ],
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        states = list(
            executor.map(
                lambda _: store.create(_internal_request(registry._test_reference, control)),  # type: ignore[attr-defined]
                range(8),
            )
        )
    assert len({state.status.run_id for state in states}) == 8
    assert len({state.status.request_hash for state in states}) == 1
    deadline = time.monotonic() + 10
    for state in states:
        while store.get(state.status.run_id).status not in {"succeeded", "failed"}:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert store.get(state.status.run_id).status == "succeeded"
        assert store.get_result(state.status.run_id).request_hash == state.status.request_hash


def test_internal_http_requires_literal_loopback_host_and_constant_time_bearer(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path, serving=False)
    catalog, _reference, _ = _round1_catalog(tmp_path / "catalog")
    control = _make_test_serving_control(tmp_path, registry, catalog)
    runtime = InferenceRuntime(
        RunStore(
            tmp_path / "runtime",
            registry=registry,
            artifact_catalog=catalog,
            serving_control=control,
        ),
        registry,
        control,
    )
    token = "session-" + "x" * 64
    with pytest.raises(ValueError, match="literal loopback"):
        create_server("0.0.0.0", 0, token=token, runtime=runtime)
    server = create_server("127.0.0.1", 0, token=token, runtime=runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        for authorization in (None, "Bearer wrong"):
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            headers = {} if authorization is None else {"Authorization": authorization}
            connection.request("GET", "/internal/core/capabilities", headers=headers)
            response = connection.getresponse()
            assert response.status == 401
            assert token not in response.read().decode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.putrequest("GET", "/internal/core/capabilities", skip_host=True)
        connection.putheader("Host", "example.com")
        connection.putheader("Authorization", f"Bearer {token}")
        connection.endheaders()
        assert connection.getresponse().status == 403

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        connection.request(
            "GET",
            "/internal/core/capabilities",
            headers={"Authorization": f"Bearer {token}"},
        )
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["servingReady"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_session_token_publication_is_atomic_and_private(tmp_path: Path) -> None:
    token_path = tmp_path / "session.token"
    token = atomic_publish_session_token(token_path)
    assert len(token) >= 64
    assert token_path.read_text(encoding="utf-8") == token
    assert not list(tmp_path.glob("*.tmp"))
    if os.name == "nt":
        acl = subprocess.run(
            ["icacls", str(token_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert "(I)" not in acl
    else:
        assert token_path.stat().st_mode & 0o077 == 0
