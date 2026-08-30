from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def test_public_startup_entrypoints_and_legacy_wrappers_exist() -> None:
    for name in (
        "socialgraph.py",
        "onboard.ps1",
        "setup.ps1",
        "dev.ps1",
        "start.ps1",
        "stop.ps1",
        "doctor.ps1",
        "configure-llm.ps1",
        "install-model.ps1",
        "bootstrap-all.ps1",
        "dev-all.ps1",
        "start-all.ps1",
        "stop-all.ps1",
    ):
        assert (SCRIPTS / name).is_file(), name


def _cli_help(command: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(SCRIPTS / "socialgraph.py"), command, "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def test_python_cli_exposes_the_stable_cross_platform_contract() -> None:
    setup = _cli_help("setup")
    for option in (
        "--profile",
        "--wheel-profile",
        "--device-policy",
        "--env-mode",
        "--api-python",
        "--gfm-python",
        "--bootstrap-python",
        "--skip-api",
        "--skip-web",
        "--gfm-text",
    ):
        assert option in setup
    assert "--gfm-text-profile" not in setup

    for command in ("dev", "start"):
        lifecycle = _cli_help(command)
        for option in (
            "--llm-mode",
            "--no-llm-prompt",
            "--reconfigure-llm",
            "--test-llm",
        ):
            assert option in lifecycle

    configure = _cli_help("configure-llm")
    for option in (
        "--preset",
        "--api-base",
        "--model",
        "--api-mode",
        "--protocol",
        "--auth-scheme",
        "--anthropic-version",
        "--timeout-seconds",
        "--api-key-stdin",
        "--allow-insecure-loopback",
        "--test-llm",
        "--skip-llm-test",
    ):
        assert option in configure

    onboard = _cli_help("onboard")
    for option in (
        "--wheel-profile",
        "--device-policy",
        "--env-mode",
        "--api-python",
        "--gfm-python",
        "--bootstrap-python",
        "--preset",
        "--api-base",
        "--model",
        "--api-mode",
        "--protocol",
        "--auth-scheme",
        "--anthropic-version",
        "--timeout-seconds",
        "--api-key-stdin",
        "--allow-insecure-loopback",
    ):
        assert option in onboard

    exported = _cli_help("export-github")
    assert "--repository" in exported
    assert "--zip" in exported


def test_setup_without_profile_fails_closed_when_stdin_is_not_a_tty() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "socialgraph.py"),
            "setup",
            "--skip-api",
            "--skip-web",
        ],
        cwd=PROJECT_ROOT,
        input="",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "explicit --wheel-profile cpu|cuda|ID" in completed.stderr


def test_setup_tty_profile_prompt_defaults_to_cpu() -> None:
    source = """
from socialgraph_fm_runtime import cli
class InteractiveInput:
    def isatty(self):
        return True
    def readline(self):
        return "\\n"
cli.sys.stdin = InteractiveInput()
print(cli._select_setup_profile(None))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "packages" / "runtime" / "src")
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    assert completed.stdout.strip() == "cpu"
    assert "CPU wheel (default)" in completed.stderr

    doctor = _cli_help("doctor")
    assert "--test-llm" in doctor
    assert "--full" in doctor
    assert "--json" in doctor
    assert "usage:" in _cli_help("stop")


def test_powershell_wrappers_forward_values_and_keep_the_key_off_argv(tmp_path: Path) -> None:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is unavailable on this platform")

    wrapper_root = tmp_path / "wrapper fixture"
    wrapper_root.mkdir()
    (wrapper_root / "lib").mkdir()
    for name in ("setup.ps1", "configure-llm.ps1"):
        shutil.copy2(SCRIPTS / name, wrapper_root / name)
    shutil.copy2(SCRIPTS / "lib" / "PythonLauncher.ps1", wrapper_root / "lib")
    (wrapper_root / "socialgraph.py").write_text(
        """from __future__ import annotations
import json
import os
import sys
from pathlib import Path
payload = {\"argv\": sys.argv[1:], \"stdin\": \"\"}
if \"--api-key-stdin\" in sys.argv:
    payload[\"stdin\"] = sys.stdin.readline()
Path(os.environ[\"SOCIALGRAPH_WRAPPER_CAPTURE\"]).write_text(
    json.dumps(payload), encoding=\"utf-8\"
)
""",
        encoding="utf-8",
    )

    capture = tmp_path / "capture.json"
    environment = dict(os.environ)
    environment["SOCIALGRAPH_WRAPPER_CAPTURE"] = str(capture)
    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper_root / "setup.ps1"),
            "-Profile",
            "Cpu",
            "-EnvMode",
            "Reuse",
            "-ApiPython",
            sys.executable,
            "-GfmPython",
            sys.executable,
            "-SkipWeb",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    setup = json.loads(capture.read_text(encoding="utf-8"))
    assert setup["argv"] == [
        "setup",
        "--profile",
        "cpu",
        "--device-policy",
        "auto",
        "--env-mode",
        "reuse",
        "--api-python",
        sys.executable,
        "--gfm-python",
        sys.executable,
        "--skip-web",
    ]

    configure_path = str(wrapper_root / "configure-llm.ps1").replace("'", "''")
    command = (
        "$key=ConvertTo-SecureString 'wrapper-placeholder-key' -AsPlainText -Force; "
        f"& '{configure_path}' -Preset custom -ApiBase https://provider.example/v1 "
        "-Model wrapper-model -ApiMode responses -TimeoutSeconds 12 -ApiKey $key -SkipLlmTest"
    )
    subprocess.run(
        [powershell, "-NoProfile", "-Command", command],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )
    configured = json.loads(capture.read_text(encoding="utf-8"))
    assert configured["stdin"].rstrip("\r\n") == "wrapper-placeholder-key"
    assert "wrapper-placeholder-key" not in configured["argv"]
    assert configured["argv"] == [
        "configure-llm",
        "--preset",
        "custom",
        "--api-base",
        "https://provider.example/v1",
        "--model",
        "wrapper-model",
        "--api-mode",
        "responses",
        "--timeout-seconds",
        "12",
        "--skip-llm-test",
        "--api-key-stdin",
    ]


def test_layout_uses_the_public_repository_structure() -> None:
    layout = _read("scripts/lib/Layout.ps1")
    assert 'Join-Path $root "apps\\web"' in layout
    assert 'Join-Path $root "services\\api"' in layout
    assert 'Join-Path $root "packages\\gfm"' in layout
    assert "platform\\core" not in layout
    assert '"Offline", "Cpu", "Cuda"' in layout


def test_unified_operations_is_a_thin_compatibility_facade() -> None:
    facade = _read("scripts/lib/UnifiedOperations.ps1")
    legacy = _read("scripts/lib/LegacyOperations.ps1")
    assert len(facade.splitlines()) < 15
    assert '"LegacyOperations.ps1"' in facade
    for module in (
        "Layout.ps1",
        "PrivateConfiguration.ps1",
        "ModelRuntime.ps1",
        "RuntimeBundle.ps1",
        "ProcessManager.ps1",
        "Verification.ps1",
    ):
        assert f'"{module}"' in legacy


def test_setup_wrapper_preserves_profiles_and_environment_selection() -> None:
    setup = _read("scripts/setup.ps1")
    assert "[string]$Profile," in setup
    assert '[string]$Profile = "Offline"' not in setup
    assert '$PSBoundParameters.ContainsKey("Profile")' in setup
    assert '"Auto", "Reuse", "Managed"' in setup
    assert '[string]$EnvMode = "Auto"' in setup
    assert '"socialgraph.py"' in setup
    assert '"setup"' in setup
    assert '"--profile"' in setup
    assert '"--wheel-profile"' in setup
    assert '"--device-policy"' in setup
    assert '"--env-mode"' in setup
    assert '"--bootstrap-python"' in setup
    assert '"--api-python"' in setup
    assert '"--gfm-python"' in setup
    assert '"--skip-api"' in setup
    assert '"--skip-web"' in setup
    assert '"--gfm-text"' in setup
    assert "UnifiedOperations.ps1" not in setup


@pytest.mark.parametrize(
    "name",
    [
        "configure-llm.ps1",
        "dev-all.ps1",
        "dev.ps1",
        "doctor.ps1",
        "start-all.ps1",
        "start.ps1",
        "stop-all.ps1",
        "stop.ps1",
        "onboard.ps1",
    ],
)
def test_powershell_wrappers_accept_an_explicit_bootstrap_python(name: str) -> None:
    script = _read(f"scripts/{name}")
    assert "[string]$BootstrapPython" in script
    assert "Resolve-SocialGraphPythonLauncher -BootstrapPython $BootstrapPython" in script


@pytest.mark.parametrize("name", ["dev.ps1", "start.ps1"])
def test_startup_modes_never_prompt_when_no_prompt_is_requested(name: str) -> None:
    script = _read(f"scripts/{name}")
    assert '"Optional", "Required", "Disabled"' in script
    assert "[switch]$NoLlmPrompt" in script
    assert "[switch]$ReconfigureLlm" in script
    assert "[switch]$TestLlm" in script
    assert '"socialgraph.py"' in script
    assert f'"{name.removesuffix(".ps1")}"' in script
    assert '"--llm-mode"' in script
    assert '"--no-llm-prompt"' in script
    assert '"--reconfigure-llm"' in script
    assert '"--test-llm"' in script
    assert "Resolve-LlmStartup" not in script
    resolver = _read("scripts/lib/PrivateConfiguration.ps1")
    assert "[C]onfigure now / continue [O]ffline / [Q]uit" in resolver


def test_private_configuration_is_atomic_and_url_validation_is_fail_closed() -> None:
    private = _read("scripts/lib/PrivateConfiguration.ps1")
    assert "[IO.File]::Replace" in private
    assert "[IO.File]::Move" in private
    assert "Protect-UnifiedConfigDirectory" in private
    assert "Assert-PrivateConfigurationFile" in private
    assert "AllowInsecureLoopback" in private
    assert "embedded credentials" in private
    assert "query string or fragment" in private
    assert "Remote API Base URLs must use HTTPS" in private


def test_llm_wrapper_maps_existing_options_without_exposing_key_in_argv() -> None:
    configure = _read("scripts/configure-llm.ps1")
    assert '"socialgraph.py"' in configure
    assert '"configure-llm"' in configure
    for option in (
        "--preset",
        "--api-base",
        "--model",
        "--api-mode",
        "--auth-scheme",
        "--anthropic-version",
        "--timeout-seconds",
        "--allow-insecure-loopback",
        "--test-llm",
        "--skip-llm-test",
        "--api-key-stdin",
    ):
        assert f'"{option}"' in configure
    assert "SecureStringToBSTR" in configure
    assert "ZeroFreeBSTR" in configure
    assert '$arguments += @("--api-key"' not in configure
    assert (SCRIPTS / "tests" / "mock_llm_provider.py").is_file()


def test_onboard_wrapper_maps_noninteractive_options_without_key_in_argv() -> None:
    onboard = _read("scripts/onboard.ps1")
    assert '"onboard"' in onboard
    for option in (
        "--wheel-profile",
        "--device-policy",
        "--env-mode",
        "--preset",
        "--api-base",
        "--model",
        "--api-mode",
        "--auth-scheme",
        "--anthropic-version",
        "--timeout-seconds",
        "--api-key-stdin",
        "--allow-insecure-loopback",
    ):
        assert f'"{option}"' in onboard
    assert '"--api-key"' not in onboard


def test_all_legacy_all_entrypoints_delegate_to_python_cli() -> None:
    expected = {
        "bootstrap-all.ps1": "setup",
        "dev-all.ps1": "dev",
        "start-all.ps1": "start",
        "stop-all.ps1": "stop",
    }
    for name, command in expected.items():
        content = _read(f"scripts/{name}")
        assert '"socialgraph.py"' in content
        assert f'"{command}"' in content


def test_root_npm_lifecycle_uses_the_cross_platform_python_selector() -> None:
    package = json.loads(_read("package.json"))
    scripts = package["scripts"]
    for name in ("dev", "bootstrap:all", "dev:all", "start:all", "stop:all"):
        assert "node scripts/run-python-cli.mjs" in scripts[name]
    assert "node scripts/run-python-cli.mjs onboard" in scripts["onboard"]
    assert "verify-all.ps1" in scripts["verify:all"]
    selector = _read("scripts/run-python-cli.mjs")
    assert 'process.env.SOCIALGRAPH_PYTHON' in selector
    assert '["python3", []]' in selector
    assert '["py", ["-3.12"]]' in selector
    assert "socialgraph.py" in selector


def test_doctor_wrapper_maps_full_diagnostics() -> None:
    doctor = _read("scripts/doctor.ps1")
    assert "[switch]$TestLlm" in doctor
    assert "[switch]$Json" in doctor
    assert "[switch]$Full" in doctor
    assert '"--test-llm"' in doctor
    assert '"--json"' in doctor
    assert '"--full"' in doctor


def test_llm_secrets_are_scoped_to_the_api_child() -> None:
    manager = _read("scripts/lib/ProcessManager.ps1")
    private = _read("scripts/lib/PrivateConfiguration.ps1")
    assert "Get-ClearedLlmEnvironment" in manager
    assert "Get-ClearedLlmEnvironment" in private
    assert '$_.Name -like "LLM_*"' in private
    assert '"OPENAI_API_KEY"' in private
    assert '"ANTHROPIC_API_KEY"' in private
    assert '"DEEPSEEK_API_KEY"' in private
    assert '"ZHIPUAI_API_KEY"' in private
    assert "$apiEnvironment[$name] = $privateEnvironment[$name]" in manager
    assert "$gfmEnvironment = @{} + $common + $clearedLlm" in manager
    assert "$governanceWebEnvironment = @{} + $common + $clearedLlm" in manager


def test_provider_disables_proxy_inheritance_and_redirects() -> None:
    provider = _read("services/api/app/provider.py")
    settings = _read("services/api/app/config.py")
    assert "trust_env=False" in provider
    assert "follow_redirects=False" in provider
    assert "validate_llm_api_base" in settings
    assert "llm_allow_insecure_loopback" in settings
    assert "must be configured together" in settings


def test_authorized_model_install_is_local_verified_and_staged() -> None:
    runtime = _read("scripts/lib/ModelRuntime.ps1")
    assert "export-manifest.json" in runtime
    assert "smoke-report.json" in runtime
    assert "registry\\socialgraph-global.json" in runtime
    assert "_verify-export" in runtime
    assert "publish --root" in runtime
    assert ".stage" in runtime
    assert "Invoke-WebRequest" not in _read("scripts/install-model.ps1")


def test_public_runtime_bundle_is_hash_bound_and_machine_derived_files_are_excluded() -> None:
    runtime = _read("scripts/lib/RuntimeBundle.ps1")
    manifest = _read("bundles/runtime-manifest.json")
    assert "socialgraph-fm.runtime-bundle/1.0" in manifest
    assert "Assert-RuntimeBundle" in runtime
    assert '"_verify-export"' in runtime
    assert '"smoke"' in runtime
    assert '"publish"' in runtime
    assert "smoke-report.json" not in manifest
    assert "registry-candidate.json" not in manifest
    assert "research" not in manifest


def test_github_ci_uses_current_runtime_paths_and_explicit_gfm_roots() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "var/config/core-api.env" not in workflow
    assert "registry/socialgraph-fm.json" not in workflow
    assert "var/config/socialgraph-api.env" in workflow
    assert "registry/socialgraph-global.json" in workflow
    assert "--config-file .github/mypy-runtime.ini" in workflow
    assert "--config-file ../../.github/mypy-api.ini" in workflow
    assert "--config-file ../../.github/mypy-gfm.ini" in workflow
    assert "github.event_name == 'push' && github.ref == 'refs/heads/main'" in workflow
    assert (
        'python -m socialgraph_gfm.cli doctor --root "${{ runner.temp }}/socialgraph-fm"'
        in workflow
    )
    assert (
        '.venv/Scripts/python.exe -m socialgraph_gfm.cli doctor --device cpu --root "${{ runner.temp }}/socialgraph-fm"'
        in workflow
    )
    assert (
        '.venv/bin/python -m socialgraph_gfm.cli doctor --device cpu --root "${{ runner.temp }}/socialgraph-fm"'
        in workflow
    )


def test_startup_powershell_behavior() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is unavailable on this platform")
    subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPTS / "tests" / "Startup.Tests.ps1"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
