from __future__ import annotations

import http.client
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.artifact_catalog import ArtifactCatalog
from socialgraph_gfm.core.bundle import CoreGraphBundle
from socialgraph_gfm.core.inference_service import InferenceRuntime, RunStore, create_server
from socialgraph_gfm.core.inference_service import atomic_publish_session_token
from socialgraph_gfm.core.serving_registry import ServingTaskHead
from _core_inference_test_support import (
    _catalog,
    _make_test_internal_create_request,
    _make_test_serving_control,
    _serving_registry,
    _wait_terminal,
)


def test_risk_serving_manifest_rejects_negative_class_output_index() -> None:
    calibrations = [
        {
            "entityType": entity_type,
            "confidenceKind": "binary-calibration",
            "calibrationVersion": f"cal-{entity_type}",
            "calibrationMethod": "sigmoid",
            "calibrationArtifactHash": "1" * 64,
            "calibrationRelativePath": f"calibration/{entity_type}.json",
            "calibrationSha256": "2" * 64,
            "calibrationProtocolHash": "3" * 64,
            "adapterDomain": "serving",
            "adapterSchemaHash": "4" * 64,
            "adapterStateHash": "5" * 64,
            "graphFeatureContractHash": "6" * 64,
        }
        for entity_type in ("node", "edge")
    ]
    with pytest.raises(ValueError, match="positive class 1"):
        ServingTaskHead.model_validate(
            {
                "taskId": "core.risk_and_trust_review",
                "kind": "risk-and-trust",
                "nodeOutputIndex": 0,
                "calibrations": calibrations,
            }
        )


def test_gfm_artifact_catalog_materializes_exact_hash_bound_bundle(tmp_path: Path) -> None:
    catalog, reference, _ = _catalog(tmp_path)

    bundle = catalog.resolve(reference)

    assert isinstance(bundle, CoreGraphBundle)
    assert tuple(node.id for node in bundle.nodes) == ("a", "b")
    assert bundle.graph_version_hash == reference.graph_version_hash


def test_gfm_artifact_catalog_rejects_tamper_and_unresolved_link_component(
    tmp_path: Path,
) -> None:
    catalog, reference, bundle_path = _catalog(tmp_path)
    bundle_path.write_bytes(bundle_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="bundle hash"):
        catalog.resolve(reference)

    link_root = tmp_path / "link-root"
    try:
        link_root.symlink_to(tmp_path / "artifacts", target_is_directory=True)
    except OSError:
        pytest.skip("host cannot create a directory symlink/reparse point")
    with pytest.raises(ValueError, match="link|reparse"):
        ArtifactCatalog.load(tmp_path / "artifact-catalog.json", artifact_root=link_root)


def test_serving_ready_requires_real_accepted_promotable_task4_checkpoint(
    tmp_path: Path,
) -> None:
    catalog, _, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    assert registry.capabilities()["servingReady"] is True
    calibration_path = tmp_path / "gfm-runtime" / "calibration" / "risk-node.json"
    calibration_path.write_bytes(calibration_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="calibration"):
        registry.capabilities()

    manifest_root = tmp_path / "manifest-tamper"
    manifest_catalog, _, _ = _catalog(manifest_root)
    manifest_registry = _serving_registry(manifest_root, catalog=manifest_catalog)
    assert manifest_registry.capabilities()["servingReady"] is True
    manifest_path = manifest_root / "gfm-runtime" / "checkpoints" / "model.serving.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="serving checkpoint manifest"):
        manifest_registry.capabilities()

    invalid_root = tmp_path / "invalid"
    invalid_catalog, _, _ = _catalog(invalid_root)
    invalid = _serving_registry(
        invalid_root,
        catalog=invalid_catalog,
        checkpoint_status="validated",
        promotable=True,
    )
    with pytest.raises(ValueError, match="accepted"):
        invalid.capabilities()


def test_real_production_checkpoint_bundle_head_runs_over_loopback_http(tmp_path: Path) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _make_test_serving_control(tmp_path, registry, catalog)
    store = RunStore(
        tmp_path / "gfm-runtime" / "inference",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
    )
    token = "session-" + "x" * 64
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=InferenceRuntime(store, registry, control),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        envelope = _make_test_internal_create_request(reference, control).model_dump(
            mode="json", by_alias=True
        )
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "POST",
            "/internal/core/runs",
            body=json.dumps(envelope),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        assert response.status == 202, response.read()
        run_id = json.loads(response.read())["status"]["runId"]
        deadline = time.monotonic() + 10
        status: dict[str, object] = {}
        while time.monotonic() < deadline:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request(
                "GET",
                f"/internal/core/runs/{run_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            polled = connection.getresponse()
            status = json.loads(polled.read())
            if status["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.02)
        assert status["status"] == "succeeded", status
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request(
            "GET",
            f"/internal/core/runs/{run_id}/result",
            headers={"Authorization": f"Bearer {token}"},
        )
        result_response = connection.getresponse()
        assert result_response.status == 200
        result = json.loads(result_response.read())
        assert result["taskId"] == "core.risk_and_trust_review"
        assert result["graphVersionHash"] == reference.graph_version_hash
        assert result["findings"][0]["reviewStatus"] == "pending-human-review"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_recovery_rolls_forward_valid_result_and_isolates_orphan_run(tmp_path: Path) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _make_test_serving_control(tmp_path, registry, catalog)
    runtime = tmp_path / "gfm-runtime" / "inference"
    store = RunStore(runtime, registry=registry, artifact_catalog=catalog, serving_control=control)
    created = store.create(_make_test_internal_create_request(reference, control))
    _wait_terminal(store, created.status.run_id)
    assert store.get(created.status.run_id).status == "succeeded"
    run_dir = runtime / "runs" / created.status.run_id
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "success.json").is_file()

    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(status="running", progress=10, errorCode=None)
    state["stateHash"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "stateHash"}
    )
    state_path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")
    orphan_id = "00000000-0000-0000-0000-000000000099"
    orphan = runtime / "runs" / orphan_id
    orphan.mkdir()
    (orphan / "request.json").write_text("{}", encoding="utf-8")

    recovered = RunStore(
        runtime, registry=registry, artifact_catalog=catalog, serving_control=control
    )
    assert recovered.get(created.status.run_id).status == "succeeded"
    assert recovered.get_result(created.status.run_id).run_id == created.status.run_id
    assert (run_dir / "success.json").is_file()
    assert recovered.recovery_diagnostics() == (
        {"runId": orphan_id, "code": "GFM_CORE_RUN_MANIFEST_MISSING"},
    )
    assert InferenceRuntime(recovered, registry, control).health()["recoveryIssueCount"] == 1

    # A succeeded-state/result ordering window without the marker is reconciled.
    (run_dir / "success.json").unlink()
    marker_recovered = RunStore(
        runtime, registry=registry, artifact_catalog=catalog, serving_control=control
    )
    assert marker_recovered.get_result(created.status.run_id).run_id == created.status.run_id
    assert (run_dir / "success.json").is_file()

    # A request+manifest publication window without state is isolated, never fatal.
    (run_dir / "state.json").unlink()
    damaged_recovered = RunStore(
        runtime, registry=registry, artifact_catalog=catalog, serving_control=control
    )
    assert {item["runId"] for item in damaged_recovered.recovery_diagnostics()} == {
        created.status.run_id,
        orphan_id,
    }


def test_ordinary_read_rejects_coherently_rehashed_request_result_substitution(
    tmp_path: Path,
) -> None:
    catalog, reference, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _make_test_serving_control(tmp_path, registry, catalog)
    runtime = tmp_path / "gfm-runtime" / "inference"
    store = RunStore(runtime, registry=registry, artifact_catalog=catalog, serving_control=control)
    created = store.create(_make_test_internal_create_request(reference, control))
    _wait_terminal(store, created.status.run_id)
    run_dir = runtime / "runs" / created.status.run_id

    request_path = run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["request"]["targetScope"]["nodeIds"] = ["b"]
    request_path.write_text(json.dumps(request, separators=(",", ":")), encoding="utf-8")
    substituted_hash = canonical_sha256(request)

    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["requestHash"] = substituted_hash
    state["stateHash"] = canonical_sha256(
        {key: value for key, value in state.items() if key != "stateHash"}
    )
    state_path.write_text(json.dumps(state, separators=(",", ":")), encoding="utf-8")

    result_path = run_dir / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["requestHash"] = substituted_hash
    result["findings"] = []
    result["resultHash"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "resultHash"}
    )
    result_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest|binding"):
        store.get_result(created.status.run_id)


def test_internal_http_rejects_every_x_forwarded_prefix_case_insensitively(
    tmp_path: Path,
) -> None:
    catalog, _, _ = _catalog(tmp_path)
    registry = _serving_registry(tmp_path, catalog=catalog)
    control = _make_test_serving_control(tmp_path, registry, catalog)
    store = RunStore(
        tmp_path / "gfm-runtime" / "inference",
        registry=registry,
        artifact_catalog=catalog,
        serving_control=control,
    )
    token = "session-" + "z" * 64
    server = create_server(
        "127.0.0.1", 0, token=token, runtime=InferenceRuntime(store, registry, control)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.request(
            "GET",
            "/internal/core/capabilities",
            headers={
                "Authorization": f"Bearer {token}",
                "x-FoRwArDeD-pRoTo": "https",
            },
        )
        response = connection.getresponse()
        assert response.status == 403
        assert json.loads(response.read())["error"]["code"] == "GFM_CORE_LOOPBACK_ONLY"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL behavior")
def test_windows_token_temp_has_verified_private_sid_dacl_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def sid() -> str:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def rules(path: Path) -> list[tuple[str, str, str]]:
        escaped = str(path).replace("'", "''")
        script = (
            f"$a=Get-Acl -LiteralPath '{escaped}';"
            "$a.Access|%{$s=$_.IdentityReference.Translate("
            "[System.Security.Principal.SecurityIdentifier]).Value;"
            "Write-Output ($s+'|'+$_.AccessControlType+'|'+$_.IsInherited)}"
        )
        environment = dict(os.environ)
        system_root = Path(environment.get("SystemRoot", r"C:\Windows"))
        environment["PSModulePath"] = str(
            system_root / "System32" / "WindowsPowerShell" / "v1.0" / "Modules"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return [tuple(line.split("|")) for line in completed.stdout.splitlines() if line]

    original_replace = os.replace
    inspected = False

    def verify_then_replace(source, destination):
        nonlocal inspected
        source_path = Path(source)
        if source_path.name.endswith(".tmp"):
            observed = rules(source_path)
            assert observed
            assert all(inherited == "False" for _, _, inherited in observed)
            allowed = {identity for identity, kind, _ in observed if kind == "Allow"}
            assert allowed == {sid()}
            inspected = True
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", verify_then_replace)
    token_path = tmp_path / "session.token"
    atomic_publish_session_token(token_path)
    assert inspected is True
    destination_rules = rules(token_path)
    assert {identity for identity, kind, _ in destination_rules if kind == "Allow"} == {sid()}
    token_path.unlink()
    assert not token_path.exists()
