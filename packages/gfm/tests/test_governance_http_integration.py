from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from socialgraph_gfm.core import inference_service as inference_service_module
from socialgraph_gfm.core.inference_cli import _parser
from socialgraph_gfm.core.inference_service import create_server


class _FakeGovernanceRuntime:
    def __init__(self) -> None:
        self.get_paths: list[str] = []
        self.posts: list[tuple[str, dict[str, object] | None]] = []
        self.closed = False

    def dispatch_get(self, path: str) -> dict[str, Any]:
        self.get_paths.append(path)
        return {"path": path, "onlineForwardReady": True}

    def dispatch_post(
        self, path: str, payload: dict[str, object] | None = None
    ) -> dict[str, Any]:
        self.posts.append((path, payload))
        return {"path": path, "payload": payload}

    def close(self) -> None:
        self.closed = True


def _request(
    *,
    port: int,
    token: str,
    method: str,
    path: str,
    body: dict[str, object] | None = None,
) -> tuple[int, dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    encoded = None if body is None else json.dumps(body)
    headers = {"Authorization": f"Bearer {token}"}
    if encoded is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=encoded, headers=headers)
    response = connection.getresponse()
    try:
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_governance_routes_are_authenticated_dispatched_and_closed() -> None:
    token = "session-" + "x" * 64
    online = _FakeGovernanceRuntime()
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=SimpleNamespace(),  # type: ignore[arg-type]
        governance_runtime=online,  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        status, payload = _request(
            port=port,
            token=token,
            method="GET",
            path="/internal/governance/health",
        )
        assert status == 200
        assert payload["onlineForwardReady"] is True

        run_request: dict[str, object] = {
            "schemaVersion": "socialgraph-fm.gfm-governance/2.0"
        }
        status, payload = _request(
            port=port,
            token=token,
            method="POST",
            path="/internal/governance/runs",
            body=run_request,
        )
        assert status == 202
        assert payload["payload"] == run_request

        status, payload = _request(
            port=port,
            token=token,
            method="POST",
            path="/internal/governance/runs/governance-" + "a" * 32 + "/cancel",
        )
        assert status == 200
        assert payload["payload"] is None
        retry_path = "/internal/governance/runs/governance-" + "b" * 32 + "/retry"
        status, payload = _request(
            port=port,
            token=token,
            method="POST",
            path=retry_path,
        )
        assert status == 202
        assert payload["payload"] is None
        assert online.get_paths == ["/internal/governance/health"]
        assert online.posts == [
            ("/internal/governance/runs", run_request),
            ("/internal/governance/runs/governance-" + "a" * 32 + "/cancel", None),
            (retry_path, None),
        ]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert online.closed is True


def test_governance_routes_fail_closed_when_runtime_is_not_configured() -> None:
    token = "session-" + "x" * 64
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=SimpleNamespace(),  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        status, payload = _request(
            port=int(server.server_address[1]),
            token=token,
            method="GET",
            path="/internal/governance/capabilities",
        )
        assert status == 503
        assert payload == {
            "error": {"code": "GFM_GOVERNANCE_MODEL_NOT_INSTALLED"}
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_governance_has_a_separate_bounded_response_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "session-" + "x" * 64
    online = _FakeGovernanceRuntime()

    def large_v2_response(_path: str) -> dict[str, Any]:
        return {"payload": "x" * 256}

    online.dispatch_get = large_v2_response  # type: ignore[assignment]
    legacy = SimpleNamespace(health=lambda: {"payload": "x" * 256})
    monkeypatch.setattr(inference_service_module, "MAX_RESPONSE_BYTES", 128)
    monkeypatch.setattr(
        inference_service_module, "MAX_GOVERNANCE_RESPONSE_BYTES", 1_024
    )
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=legacy,  # type: ignore[arg-type]
        governance_runtime=online,  # type: ignore[arg-type]
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        status, payload = _request(
            port=port,
            token=token,
            method="GET",
            path="/internal/governance/health",
        )
        assert status == 200
        assert len(payload["payload"]) == 256

        status, payload = _request(
            port=port,
            token=token,
            method="GET",
            path="/internal/core/health",
        )
        assert status == 500
        assert payload == {"error": {"code": "GFM_CORE_RESPONSE_TOO_LARGE"}}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_governance_unavailable_runtime_maps_service_error(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from socialgraph_gfm.governance.service import GovernanceServingRuntime

    token = "session-" + "x" * 64
    online = GovernanceServingRuntime(
        tmp_path / "online",
        global_model_root=tmp_path / "missing-global-model",
        device="cpu",
    )
    server = create_server(
        "127.0.0.1",
        0,
        token=token,
        runtime=SimpleNamespace(),  # type: ignore[arg-type]
        governance_runtime=online,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = int(server.server_address[1])
        status, health = _request(
            port=port,
            token=token,
            method="GET",
            path="/internal/governance/health",
        )
        assert status == 200
        assert health["onlineForwardReady"] is False

        status, payload = _request(
            port=port,
            token=token,
            method="POST",
            path="/internal/governance/runs",
            body={"schemaVersion": "socialgraph-fm.gfm-governance/2.0"},
        )
        assert status == 503
        assert payload == {
            "error": {"code": "GFM_GOVERNANCE_MODEL_NOT_INSTALLED"}
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cli_accepts_governance_root_without_a_device_override() -> None:
    arguments = _parser().parse_args(
        [
            "--runtime-root",
            "runtime",
            "--serving-control",
            "control.json",
            "--published-serving-root",
            "published-serving",
            "--published-artifact-root",
            "published-artifacts",
            "--artifact-root",
            "artifacts",
            "--token-file",
            "runtime/session.token",
            "--global-model-root",
            "global-model-runtime",
            "--governance-root",
            "governance",
        ]
    )
    assert str(arguments.governance_root) == "governance"
    assert str(arguments.published_serving_root) == "published-serving"
    assert str(arguments.published_artifact_root) == "published-artifacts"
    assert str(arguments.global_model_root) == "global-model-runtime"
    assert not hasattr(arguments, "global_model_device")


def test_cli_rejects_the_retired_governance_device_override() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--runtime-root",
                "runtime",
                "--serving-control",
                "control.json",
                "--artifact-root",
                "artifacts",
                "--token-file",
                "runtime/session.token",
                "--global-model-device",
                "cuda",
            ]
        )
