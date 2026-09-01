from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from socialgraph_fm_runtime import operations
from socialgraph_fm_runtime.environment import BytecodePruneResult, CompatibilityResult
from socialgraph_fm_runtime.layout import RuntimeLayout, environment_python
from socialgraph_fm_runtime.profile import RuntimeProfile


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
    command: list[str] = []

    def fake_run(selected, *, environment, **_kwargs):
        command.extend(selected)
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
    assert command[1:3] == ["-B", "-m"]


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
    assert services[0].arguments[:2] == ("-B", "-m")
    assert services[1].arguments[:2] == ("-B", "-m")
    assert "--global-model-device" not in services[0].arguments


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
        "prune_runtime_bytecode",
        lambda _python: BytecodePruneResult(removed_files=0, removed_bytes=0),
    )
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


def test_new_runtime_is_pruned_and_probed_before_russia_and_activation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_setup(monkeypatch, layout)
    events: list[str] = []
    installed = operations.install_runtime_environment
    probed = operations.probe_runtime_environment

    def record_install(*args, **kwargs):
        events.append("install")
        return installed(*args, **kwargs)

    def record_prune(_python: Path) -> BytecodePruneResult:
        events.append("prune")
        return BytecodePruneResult(removed_files=2, removed_bytes=128)

    def record_probe(python: Path, profile) -> CompatibilityResult:
        events.append(
            "probe-active"
            if Path(python) == environment_python(layout.runtime_environment)
            else "probe-staging"
        )
        return probed(python, profile)

    def record_russia(*_args, **_kwargs) -> dict[str, bool]:
        events.append("russia")
        return {"passed": True}

    def record_checkpoints(*_args, **_kwargs) -> dict[str, list[object]]:
        events.append("checkpoints")
        return {"protocols": []}

    monkeypatch.setattr(operations, "install_runtime_environment", record_install)
    monkeypatch.setattr(operations, "prune_runtime_bytecode", record_prune)
    monkeypatch.setattr(operations, "probe_runtime_environment", record_probe)
    monkeypatch.setattr(
        operations,
        "run_full_gfm_probe",
        record_russia,
    )
    monkeypatch.setattr(
        operations,
        "run_checkpoint_forward_probe",
        record_checkpoints,
    )

    operations.setup(layout)

    assert events == [
        "install",
        "prune",
        "probe-staging",
        "russia",
        "probe-active",
        "checkpoints",
    ]


def test_existing_runtime_is_pruned_reprobed_without_reinstalling_or_losing_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_setup(monkeypatch, layout)
    python = environment_python(layout.runtime_environment)
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("python", encoding="utf-8")
    state_files = (
        layout.llm_config_file,
        layout.model_root / "preserved-model.bin",
        layout.governance_root / "reviewed-cases" / "preserved-case.json",
    )
    for path in state_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserved", encoding="utf-8")
    previous = RuntimeProfile.create(
        install_profile_id="windows-x86_64-cpu-pt28",
        platform={"system": "windows", "machine": "x86_64"},
        interpreter={
            "path": str(python),
            "installLockSha256": "a" * 64,
            "fingerprint": _result(python).fingerprint,
        },
    )
    events: list[str] = []

    def record_prune(_python: Path) -> BytecodePruneResult:
        events.append("prune")
        return BytecodePruneResult(removed_files=3, removed_bytes=256)

    def record_reprobe(_python: Path, _profile) -> CompatibilityResult:
        events.append("reprobe")
        return _result(python)

    def record_russia(*_args, **_kwargs) -> dict[str, bool]:
        events.append("russia")
        return {"passed": True}

    def record_checkpoints(*_args, **_kwargs) -> dict[str, list[object]]:
        events.append("checkpoints")
        return {"protocols": []}

    monkeypatch.setattr(RuntimeProfile, "load", lambda _path: previous)
    monkeypatch.setattr(
        operations,
        "_validate_recorded_runtime",
        lambda *_args: (python, _result(python)),
    )
    monkeypatch.setattr(
        operations,
        "install_runtime_environment",
        lambda *_args, **_kwargs: pytest.fail("existing runtime must not be reinstalled"),
    )
    monkeypatch.setattr(
        operations,
        "prune_runtime_bytecode",
        record_prune,
    )
    monkeypatch.setattr(
        operations,
        "probe_runtime_environment",
        record_reprobe,
    )
    monkeypatch.setattr(
        operations,
        "run_full_gfm_probe",
        record_russia,
    )
    monkeypatch.setattr(
        operations,
        "run_checkpoint_forward_probe",
        record_checkpoints,
    )

    operations.setup(layout)

    assert events == ["prune", "reprobe", "russia", "checkpoints"]
    assert all(path.read_text(encoding="utf-8") == "preserved" for path in state_files)


@pytest.mark.parametrize("failure", ["prune", "probe"])
def test_existing_runtime_reuse_failure_builds_a_verified_staging_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_setup(monkeypatch, layout, existing=True)
    active_python = environment_python(layout.runtime_environment)
    active_python.parent.mkdir(parents=True, exist_ok=True)
    active_python.write_text("old-python", encoding="utf-8")
    previous = RuntimeProfile.create(
        install_profile_id="windows-x86_64-cpu-pt28",
        platform={"system": "windows", "machine": "x86_64"},
        interpreter={
            "path": str(active_python),
            "installLockSha256": "a" * 64,
            "fingerprint": _result(active_python).fingerprint,
        },
    )
    events: list[str] = []
    state = {"install_started": False}
    staged_destinations: list[Path] = []
    install = operations.install_runtime_environment

    def record_install(*args, **kwargs):
        state["install_started"] = True
        staged_destinations.append(Path(kwargs["destination"]))
        events.append("install")
        return install(*args, **kwargs)

    def fail_old_prune(python: Path) -> BytecodePruneResult:
        if Path(python) == active_python and not state["install_started"]:
            events.append("prune-old")
            if failure == "prune":
                raise RuntimeError("synthetic bytecode pruning failure")
            return BytecodePruneResult(removed_files=1, removed_bytes=64)
        events.append("prune-staging")
        return BytecodePruneResult(removed_files=0, removed_bytes=0)

    def fail_old_probe(python: Path, _profile) -> CompatibilityResult:
        if Path(python) == active_python and not state["install_started"]:
            events.append("probe-old")
            if failure == "probe":
                result = _result(active_python)
                return CompatibilityResult(
                    compatible=False,
                    errors=("synthetic post-prune probe failure",),
                    fingerprint=result.fingerprint,
                )
        events.append("probe-new")
        return _result(Path(python))

    monkeypatch.setattr(RuntimeProfile, "load", lambda _path: previous)
    monkeypatch.setattr(
        operations,
        "_validate_recorded_runtime",
        lambda *_args: (active_python, _result(active_python)),
    )
    monkeypatch.setattr(operations, "install_runtime_environment", record_install)
    monkeypatch.setattr(operations, "prune_runtime_bytecode", fail_old_prune)
    monkeypatch.setattr(operations, "probe_runtime_environment", fail_old_probe)

    runtime = operations.setup(layout, operations.SetupOptions(full_probe=False))

    assert events.count("install") == 1
    assert "prune-staging" in events
    assert staged_destinations and staged_destinations[0] != layout.runtime_environment
    assert Path(runtime.interpreter["path"]) == active_python
    assert not (layout.runtime_environment / "marker").exists()
    setup_log = layout.setup_log_file.read_text(encoding="utf-8")
    assert "Existing CPU runtime cannot be reused:" in setup_log
    assert "installing the verified Windows/Ubuntu lock" in setup_log


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
