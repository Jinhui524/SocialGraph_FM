from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from socialgraph_fm_runtime import operations
from socialgraph_fm_runtime.environment import CompatibilityResult
from socialgraph_fm_runtime.layout import RuntimeLayout, environment_python


def _result(python: Path) -> CompatibilityResult:
    fingerprint = {
        "fingerprintSha256": f"hash:{python}",
        "pythonVersion": "3.12.4",
        "versions": {"torch": "2.8.0+cpu"},
        "torch": {"cpuBuild": True},
        "neighborLoader": True,
    }
    return CompatibilityResult(True, (), fingerprint)


def _profile(system: str = "Windows") -> dict[str, object]:
    identifier = (
        "windows-x86_64-cpu-pt28"
        if system == "Windows"
        else "linux-x86_64-cpu-pt28"
    )
    return {
        "id": identifier,
        "system": system,
        "machine": "x86_64",
        "requirementsLockSha256": "a" * 64,
    }


def test_provider_check_injects_key_only_into_api_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, str] = {}

    def fake_run(_command, *, environment, **_kwargs):
        captured.update(environment)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "schemaVersion": "socialgraph-fm.llm-provider-check/1.0",
                    "ok": True,
                    "code": "OK",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(operations, "run_captured_process", fake_run)
    monkeypatch.setattr(operations, "resolve_python", lambda value: Path(value))
    operations.test_llm_configuration(
        RuntimeLayout(tmp_path),
        {
            "LLM_API_BASE": "https://provider.example/v1",
            "LLM_MODEL": "model-id",
            "LLM_API_KEY": "test-secret",
        },
        runtime_python=Path(sys.executable),
    )
    assert captured["LLM_API_KEY"] == "test-secret"
    assert {name for name in captured if name.startswith("LLM_")} == {
        "LLM_API_BASE",
        "LLM_API_KEY",
        "LLM_MODEL",
    }


def test_provider_check_surfaces_only_safe_failure_category(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(operations, "resolve_python", lambda value: Path(value))
    monkeypatch.setattr(
        operations,
        "run_captured_process",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout='{"code":"LLM_AUTH_ERROR"}',
            stderr="upstream leaked test-secret",
        ),
    )
    with pytest.raises(RuntimeError, match="AUTH") as captured:
        operations.test_llm_configuration(
            RuntimeLayout(tmp_path),
            {
                "LLM_API_BASE": "https://provider.example/v1",
                "LLM_MODEL": "model-id",
                "LLM_API_KEY": "test-secret",
            },
            runtime_python=Path(sys.executable),
        )
    assert "test-secret" not in str(captured.value)


def test_two_services_share_one_python_but_only_api_receives_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    python = Path(sys.executable)
    monkeypatch.setattr(
        operations,
        "resolve_llm",
        lambda _layout: {
            "LLM_API_BASE": "https://provider.example/v1",
            "LLM_MODEL": "model-id",
            "LLM_API_KEY": "test-secret",
        },
    )
    monkeypatch.setattr(
        Path,
        "stat",
        lambda self: SimpleNamespace(st_mtime_ns=1, st_size=2),
    )
    services = operations.build_services(
        layout,
        {"api": python, "gfm": python},
        ports=operations.Ports(5173, 5173, 8766),
    )

    assert [service.name for service in services] == ["gfm", "socialgraph-api"]
    assert all(service.executable == python for service in services)
    assert "LLM_API_KEY" not in services[0].environment
    assert services[1].environment["LLM_API_KEY"] == "test-secret"
    assert services[1].environment["SOCIALGRAPH_WEB_CLIENT_ROOT"] == str(
        layout.web_client_root
    )
    assert services[0].arguments[services[0].arguments.index("--global-model-device") + 1] == "cpu"


def _patch_setup(
    monkeypatch: pytest.MonkeyPatch,
    layout: RuntimeLayout,
    *,
    existing: bool = False,
) -> None:
    monkeypatch.setattr(RuntimeLayout, "initialize_serving_contracts", lambda _self: None)
    monkeypatch.setattr(operations, "_managed_services_stopped", lambda _layout: True)
    monkeypatch.setattr(
        operations,
        "migrate_private_environment_permissions",
        lambda _path, **_kwargs: None,
    )
    monkeypatch.setattr(operations, "_profile_for_host", lambda _layout: _profile())
    monkeypatch.setattr(operations, "ensure_bootstrap_python", lambda: Path(sys.executable))
    monkeypatch.setattr(
        operations,
        "probe_bootstrap_environment",
        lambda _python: _result(Path(sys.executable)),
    )
    monkeypatch.setattr(operations, "normalized_platform", lambda: ("windows", "x86_64"))
    monkeypatch.setattr(
        operations,
        "probe_runtime_environment",
        lambda python, _profile: _result(Path(python)),
    )

    def fake_install(_layout, _bootstrap, _profile, *, destination, **_kwargs):
        python = environment_python(destination)
        python.parent.mkdir(parents=True)
        python.write_text("python", encoding="utf-8")
        return python

    monkeypatch.setattr(operations, "install_runtime_environment", fake_install)
    monkeypatch.setattr(
        operations,
        "install_web_bundle",
        lambda _layout: {"fileCount": 1, "totalBytes": 1},
    )
    bundle = object()
    monkeypatch.setattr(operations, "install_public_runtime_bundle", lambda *_args: bundle)
    monkeypatch.setattr(
        operations,
        "materialize_target_examples",
        lambda *_args: {"zeroShot": {}, "fewShot": {}},
    )
    monkeypatch.setattr(operations, "_cleanup_legacy_installations", lambda *_args: None)
    if existing:
        layout.runtime_environment.mkdir(parents=True)
        (layout.runtime_environment / "marker").write_text("old", encoding="utf-8")


def test_setup_builds_temp_then_switches_one_runtime_and_runs_callback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_setup(monkeypatch, layout)
    callback: list[Path] = []

    runtime = operations.setup(
        layout,
        operations.SetupOptions(
            full_probe=False,
            after_runtime=lambda python: callback.append(python),
        ),
    )

    assert callback == [environment_python(layout.runtime_environment)]
    assert Path(runtime.interpreter["path"]) == environment_python(layout.runtime_environment)
    assert runtime.install_profile_id == "windows-x86_64-cpu-pt28"
    assert layout.profile_file.is_file()


def test_failed_required_llm_callback_rolls_back_previous_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_setup(monkeypatch, layout, existing=True)

    def fail(_python: Path) -> None:
        raise RuntimeError("LLM connection check failed: AUTH")

    with pytest.raises(RuntimeError, match="AUTH"):
        operations.setup(
            layout,
            operations.SetupOptions(full_probe=False, after_runtime=fail),
        )

    assert (layout.runtime_environment / "marker").read_text(encoding="utf-8") == "old"
    assert not layout.profile_file.exists()


def test_resolve_llm_never_offers_offline_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(operations, "parse_private_environment", lambda _path: {})
    with pytest.raises(RuntimeError, match="configure-llm"):
        operations.resolve_llm(RuntimeLayout(tmp_path))
