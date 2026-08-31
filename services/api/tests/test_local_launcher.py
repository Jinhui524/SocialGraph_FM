from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from app import __main__ as launcher


@pytest.fixture(autouse=True)
def _configured_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_BASE", "https://provider.example/v1")
    monkeypatch.setenv("LLM_MODEL", "model-a")
    monkeypatch.setenv("LLM_API_KEY", "secret-key")

    async def verified() -> None:
        return None

    monkeypatch.setattr(launcher, "verify_provider", verified)


def test_supported_launcher_binds_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: str, **options: Any) -> None:
        captured.update({"app": app, **options})

    monkeypatch.setattr(launcher.uvicorn, "run", fake_run)
    launcher.main()

    assert captured["app"] == "app.main:app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 5173
    assert captured["reload"] is False


def test_selector_loop_factory_returns_a_selector_loop() -> None:
    loop = launcher.selector_event_loop_factory()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()


def test_windows_launcher_avoids_the_proactor_accept_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: str, **options: Any) -> None:
        captured.update({"app": app, **options})

    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.uvicorn, "run", fake_run)
    launcher.main()

    assert captured["loop"] is launcher.selector_event_loop_factory


def test_supported_launcher_uses_the_managed_api_port(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: str, **options: Any) -> None:
        captured.update({"app": app, **options})

    monkeypatch.setenv("SOCIALGRAPH_CORE_API_PORT", "8123")
    monkeypatch.setattr(launcher.uvicorn, "run", fake_run)
    launcher.main()

    assert captured["port"] == 8123


def test_supported_launcher_rejects_missing_llm_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_API_KEY")
    with pytest.raises(RuntimeError, match="LLM_API_BASE"):
        launcher.main()


def test_runtime_identity_root_must_match_the_managed_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    expected = tmp_path / "expected-runtime"
    expected.mkdir()
    monkeypatch.setenv("GFM_GOVERNANCE_ROOT", str(expected))
    assert launcher.managed_runtime_identity(
        ["--runtime-identity-root", str(expected)]
    ) == os.path.abspath(expected)

    with pytest.raises(ValueError, match="runtime identity"):
        launcher.managed_runtime_identity(
            ["--runtime-identity-root", str(tmp_path / "different-runtime")]
        )


def test_runtime_identity_root_is_required_for_the_managed_process() -> None:
    with pytest.raises(ValueError, match="required"):
        launcher.managed_runtime_identity([])


def test_console_script_forwards_process_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str] | None] = []
    monkeypatch.setattr(launcher.sys, "argv", ["socialgraph-api", "--runtime-identity-root", "x"])
    monkeypatch.setattr(launcher, "main", lambda arguments=None: captured.append(arguments))

    launcher.console_main()

    assert captured == [["--runtime-identity-root", "x"]]
