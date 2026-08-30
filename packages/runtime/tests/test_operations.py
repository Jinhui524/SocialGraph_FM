from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import socialgraph_fm_runtime.operations as operations
from socialgraph_fm_runtime.environment import CompatibilityResult
from socialgraph_fm_runtime.layout import RuntimeLayout, environment_python
from socialgraph_fm_runtime.profile import RuntimeProfile


def test_provider_check_injects_key_only_into_the_api_child(
    monkeypatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    layout.api_root.mkdir(parents=True)
    runtime = SimpleNamespace(
        interpreters={"api": {"path": sys.executable}, "gfm": None}
    )
    monkeypatch.setattr(operations.RuntimeProfile, "load", lambda _path: runtime)
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["environment"]
        return SimpleNamespace(
            returncode=0,
            stdout=(
                '{"schemaVersion":"socialgraph-fm.llm-provider-check/1.0",'
                '"ok":true,"code":"OK"}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(operations, "run_captured_process", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "parent-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-anthropic-sentinel")
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_API_KEY": "configured-key",
        "LLM_MODEL": "configured-model",
        "LLM_API_MODE": "chat_completions",
        "LLM_TIMEOUT_SECONDS": "15",
        "LLM_ALLOW_INSECURE_LOOPBACK": "false",
        "LLM_VERIFICATION_STATUS": "configured_unverified",
    }

    operations.test_llm_configuration(layout, environment)

    child = captured["environment"]
    assert isinstance(child, dict)
    assert child["LLM_API_KEY"] == "configured-key"
    assert "OPENAI_API_KEY" not in child
    assert "ANTHROPIC_API_KEY" not in child
    assert captured["command"][-2:] == ["-m", "app.provider_check"]


def test_provider_check_surfaces_only_the_safe_failure_category(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    layout.api_root.mkdir(parents=True)
    secret = "provider-response-" + "sentinel"

    def fake_run(_command, **_kwargs):
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                '{"schemaVersion":"socialgraph-fm.llm-provider-check/1.0",'
                '"ok":false,"code":"LLM_AUTH_ERROR"}\n'
                + secret
            ),
        )

    monkeypatch.setattr(operations, "run_captured_process", fake_run)
    environment = {
        "LLM_API_BASE": "https://provider.example/v1",
        "LLM_API_KEY": "test-key",
        "LLM_MODEL": "test-model",
        "LLM_API_MODE": "chat_completions",
        "LLM_AUTH_SCHEME": "bearer",
        "LLM_ANTHROPIC_VERSION": "",
        "LLM_TIMEOUT_SECONDS": "15",
        "LLM_ALLOW_INSECURE_LOOPBACK": "false",
        "LLM_VERIFICATION_STATUS": "configured_unverified",
    }

    with pytest.raises(RuntimeError, match="AUTH") as captured:
        operations.test_llm_configuration(
            layout, environment, api_python=Path(sys.executable)
        )
    assert secret not in str(captured.value)


def test_web_and_gfm_service_environments_never_receive_llm_keys(
    monkeypatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    for path in (layout.web_root, layout.api_root, layout.gfm_package):
        path.mkdir(parents=True)
    vite = layout.web_root / "node_modules" / "vite" / "bin" / "vite.js"
    vite.parent.mkdir(parents=True)
    vite.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(operations.shutil, "which", lambda _name: sys.executable)
    monkeypatch.setenv("LLM_API_KEY", "parent-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "parent-anthropic-sentinel")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "parent-deepseek-sentinel")
    monkeypatch.setenv("ZHIPUAI_API_KEY", "parent-glm-sentinel")
    runtime = SimpleNamespace(
        profile="cuda", interpreters={}, install_profile_id="fixture-profile"
    )
    services = operations.build_services(
        layout,
        runtime,
        {"api": Path(sys.executable), "gfm": Path(sys.executable)},
        mode="development",
        enable_llm=False,
        ports=operations.Ports(web=15173, api=18000, gfm=18766),
    )

    for service in services:
        assert "LLM_API_KEY" not in service.environment
        assert "OPENAI_API_KEY" not in service.environment
        assert "ANTHROPIC_API_KEY" not in service.environment
        assert "DEEPSEEK_API_KEY" not in service.environment
        assert "ZHIPUAI_API_KEY" not in service.environment

    gfm = next(service for service in services if service.name == "gfm")
    assert gfm.health_json == {
        "schemaVersion": "socialgraph-fm.core-internal-health/2.0",
        "ok": True,
    }


@pytest.mark.parametrize(
    ("wheel_family", "device_policy", "expected"),
    [
        ("cpu", "auto", "cpu"),
        ("cuda", "auto", "auto"),
        ("cuda", "cpu", "cpu"),
        ("cuda", "cuda-required", "cuda"),
    ],
)
def test_service_device_argument_separates_wheels_from_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wheel_family: str,
    device_policy: str,
    expected: str,
) -> None:
    layout = RuntimeLayout(tmp_path)
    for path in (layout.web_root, layout.api_root, layout.gfm_package):
        path.mkdir(parents=True)
    vite = layout.web_root / "node_modules" / "vite" / "bin" / "vite.js"
    vite.parent.mkdir(parents=True)
    vite.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(operations.shutil, "which", lambda _name: sys.executable)
    runtime = SimpleNamespace(
        profile=wheel_family,
        device_policy=device_policy,
        interpreters={},
        install_profile_id="fixture-profile",
    )
    services = operations.build_services(
        layout,
        runtime,
        {"api": Path(sys.executable), "gfm": Path(sys.executable)},
        mode="development",
        enable_llm=False,
        ports=operations.Ports(web=15173, api=18000, gfm=18766),
    )
    gfm = next(service for service in services if service.name == "gfm")
    index = gfm.arguments.index("--global-model-device")
    assert gfm.arguments[index + 1] == expected


def test_full_doctor_never_executes_unverified_model_assets(
    monkeypatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    for path in (layout.web_root, layout.api_root, layout.gfm_package):
        path.mkdir(parents=True)
    vite = layout.web_root / "node_modules" / "vite" / "bin" / "vite.js"
    vite.parent.mkdir(parents=True)
    vite.write_text("fixture", encoding="utf-8")
    runtime = SimpleNamespace(
        profile="cuda", interpreters={}, install_profile_id="fixture-profile"
    )
    monkeypatch.setattr(operations.RuntimeProfile, "load", lambda _path: runtime)
    monkeypatch.setattr(
        operations,
        "validate_profile",
        lambda *_args: {"api": Path(sys.executable), "gfm": Path(sys.executable)},
    )
    monkeypatch.setattr(
        operations,
        "load_and_verify_bundle",
        lambda _layout: (_ for _ in ()).throw(RuntimeError("tampered bundle")),
    )
    executed = False
    checkpoint_executed = False

    def forbidden_forward(*_args, **_kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("unverified model was executed")

    def forbidden_checkpoint_forward(*_args, **_kwargs):
        nonlocal checkpoint_executed
        checkpoint_executed = True
        raise AssertionError("unverified checkpoint was executed")

    monkeypatch.setattr(operations, "run_full_gfm_probe", forbidden_forward)
    monkeypatch.setattr(
        operations, "run_checkpoint_forward_probe", forbidden_checkpoint_forward
    )
    monkeypatch.setattr(operations, "parse_private_environment", lambda _path: {})

    report = operations.doctor(layout, full=True)

    assert report["passed"] is False
    assert executed is False
    assert checkpoint_executed is False
    forward = next(check for check in report["checks"] if check["name"] == "global-forward")
    assert forward["passed"] is False
    assert "not verified" in forward["detail"]


def test_noninteractive_llm_modes_never_wait_for_input(
    monkeypatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    monkeypatch.setattr(operations.os, "isatty", lambda _fd: False)

    assert operations.resolve_llm(layout, "disabled", no_prompt=False) is False
    assert operations.resolve_llm(layout, "optional", no_prompt=False) is False
    with pytest.raises(RuntimeError, match="required but missing"):
        operations.resolve_llm(layout, "required", no_prompt=False)


def _compatible(capability: str) -> CompatibilityResult:
    fingerprint = {
        "fingerprintSha256": capability * 64,
        "pythonVersion": "3.12.4",
        "system": "Windows",
        "machine": "AMD64",
        "libc": None,
        "libcVersion": None,
        "versions": {},
        "imports": {},
        "torch": {},
        "neighborLoader": None,
    }
    return CompatibilityResult(True, (), fingerprint)


def _patch_minimal_offline_setup(
    monkeypatch: pytest.MonkeyPatch, layout: RuntimeLayout
) -> None:
    layout.api_root.mkdir(parents=True)
    (layout.api_root / "requirements.lock").write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(RuntimeLayout, "initialize_serving_contracts", lambda _self: None)
    monkeypatch.setattr(operations, "migrate_private_environment_permissions", lambda _path: None)
    monkeypatch.setattr(operations, "ensure_bootstrap_python", lambda _value: Path(sys.executable))
    monkeypatch.setattr(
        operations, "probe_bootstrap_environment", lambda _python: _compatible("b")
    )
    monkeypatch.setattr(
        operations,
        "_host_diagnostics",
        lambda: {
            "system": "windows",
            "machine": "x86_64",
            "libc": None,
            "libcVersion": None,
            "node": {"available": False, "version": None},
            "npm": {"available": False, "version": None},
            "gpuDriver": {"available": False, "driverVersion": None},
        },
    )
    monkeypatch.setattr(operations, "load_and_verify_bundle", lambda _layout: object())
    monkeypatch.setattr(
        operations,
        "materialize_target_examples",
        lambda _layout, _bundle: {"zeroShot": {}, "fewShot": {}},
    )
    monkeypatch.setattr(operations, "_managed_services_stopped", lambda _layout: True)


def test_broken_managed_environment_builds_a_new_generation_then_switches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_minimal_offline_setup(monkeypatch, layout)
    old_root = layout.managed_environment("api", "old")
    old_python = environment_python(old_root)
    old_python.parent.mkdir(parents=True)
    old_python.write_bytes(b"broken")
    old = RuntimeProfile.create(
        profile="offline",
        env_mode="managed",
        install_profile_id=None,
        platform={"system": "windows", "machine": "x86_64"},
        interpreters={
            "bootstrap": None,
            "api": {
                "path": str(old_python),
                "source": "managed",
                "fingerprint": _compatible("o").fingerprint,
            },
            "gfm": None,
        },
    )
    old.write(layout.profile_file)

    def probe(path: Path, _root: Path) -> CompatibilityResult:
        return CompatibilityResult(False, ("broken",), _compatible("x").fingerprint) if path == old_python else _compatible("a")

    installed: list[Path] = []

    def install(_layout, _bootstrap, *, destination, logger):
        installed.append(destination)
        selected = environment_python(destination)
        selected.parent.mkdir(parents=True)
        selected.write_bytes(b"new")
        logger("installed fixture")
        return selected

    monkeypatch.setattr(operations, "probe_api_environment", probe)
    monkeypatch.setattr(operations, "install_api_environment", install)

    runtime = operations.setup(
        layout,
        operations.SetupOptions(
            profile="offline", env_mode="managed", skip_web=True, full_probe=False
        ),
    )

    assert len(installed) == 1
    assert installed[0] != old_root
    assert runtime.interpreters["api"]["path"] == str(environment_python(installed[0]))
    assert not old_root.exists()
    assert installed[0].is_dir()


def test_reuse_mode_never_installs_into_an_external_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_minimal_offline_setup(monkeypatch, layout)
    monkeypatch.setattr(
        operations, "probe_api_environment", lambda _python, _root: _compatible("a")
    )
    monkeypatch.setattr(
        operations,
        "install_api_environment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reuse must not install")
        ),
    )

    runtime = operations.setup(
        layout,
        operations.SetupOptions(
            profile="offline",
            env_mode="reuse",
            api_python=sys.executable,
            skip_web=True,
            full_probe=False,
        ),
    )

    assert runtime.interpreters["api"]["source"] == "explicit"
    assert runtime.interpreters["api"]["path"] == str(Path(sys.executable))


def test_onboarding_callback_runs_after_api_and_selects_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_minimal_offline_setup(monkeypatch, layout)
    monkeypatch.setattr(
        operations, "probe_api_environment", lambda _python, _root: _compatible("a")
    )
    observed: list[Path] = []

    def after_api(api_python: Path) -> str:
        observed.append(api_python)
        assert api_python == Path(sys.executable)
        return "offline"

    runtime = operations.setup(
        layout,
        operations.SetupOptions(
            profile="auto",
            env_mode="reuse",
            api_python=sys.executable,
            skip_web=True,
            full_probe=False,
            after_api=after_api,
        ),
    )

    assert observed == [Path(sys.executable)]
    assert runtime.profile == "offline"
    assert runtime.device_policy == "auto"


def test_onboarding_auto_requires_an_explicit_wheel_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    _patch_minimal_offline_setup(monkeypatch, layout)
    monkeypatch.setattr(
        operations, "probe_api_environment", lambda _python, _root: _compatible("a")
    )
    with pytest.raises(RuntimeError, match="must select"):
        operations.setup(
            layout,
            operations.SetupOptions(
                profile="auto",
                env_mode="reuse",
                api_python=sys.executable,
                skip_web=True,
                full_probe=False,
                after_api=lambda _python: None,
            ),
        )


def test_cleanup_never_removes_a_generation_still_referenced_explicitly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = RuntimeLayout(tmp_path)
    root = layout.managed_environment("api", "same")
    python = environment_python(root)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"fixture")
    old = RuntimeProfile.create(
        profile="offline",
        env_mode="managed",
        install_profile_id=None,
        platform={"system": "windows", "machine": "x86_64"},
        interpreters={
            "bootstrap": None,
            "api": {"path": str(python), "source": "managed"},
            "gfm": None,
        },
    )
    current = RuntimeProfile.create(
        profile="offline",
        env_mode="auto",
        install_profile_id=None,
        platform={"system": "windows", "machine": "x86_64"},
        interpreters={
            "bootstrap": None,
            "api": {"path": str(python), "source": "explicit"},
            "gfm": None,
        },
    )
    monkeypatch.setattr(operations, "_managed_services_stopped", lambda _layout: True)

    operations._cleanup_replaced_generations(
        layout, old, current, operations._SetupReporter(layout)
    )

    assert root.is_dir()


def test_managed_generation_paths_are_short_and_legacy_paths_remain_recognized(
    tmp_path: Path,
) -> None:
    layout = RuntimeLayout(tmp_path)
    current = layout.managed_environment("gfm", "a" * 12)
    legacy = layout.legacy_managed_environment_root / "gfm" / ("b" * 20)

    assert current == tmp_path / "var" / "e" / "g" / ("a" * 12)
    assert operations._is_repo_managed_python(layout, environment_python(current))
    assert operations._is_repo_managed_python(layout, environment_python(legacy))


def test_host_compatibility_rejects_node_or_npm_drift() -> None:
    host = {
        "node": {"available": True, "version": "v23.9.0"},
        "npm": {"available": True, "version": "11.6.2"},
        "gpuDriver": {"available": True},
    }
    with pytest.raises(RuntimeError, match="Node.js 24.x"):
        operations._require_host_compatibility(host, web=True, cuda=False)

    host["node"] = {"available": True, "version": "v24.13.0"}
    host["npm"] = {"available": True, "version": "10.9.0"}
    with pytest.raises(RuntimeError, match="npm 11.x"):
        operations._require_host_compatibility(host, web=True, cuda=False)


def test_host_compatibility_requires_a_cuda_driver_for_cuda_profile() -> None:
    host = {
        "node": {"available": True, "version": "v24.13.0"},
        "npm": {"available": True, "version": "11.6.2"},
        "gpuDriver": {"available": False},
    }
    with pytest.raises(RuntimeError, match="NVIDIA driver"):
        operations._require_host_compatibility(host, web=True, cuda=True)
