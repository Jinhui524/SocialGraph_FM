from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
PUBLIC_COMMANDS = ("onboard", "configure-llm", "start", "stop", "doctor")


def _read(relative: str) -> str:
    return (PROJECT_ROOT / relative).read_text(encoding="utf-8")


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "socialgraph.py"), *arguments],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_public_runtime_has_one_python_entrypoint_and_one_windows_wrapper() -> None:
    assert (SCRIPTS / "socialgraph.py").is_file()
    assert (SCRIPTS / "socialgraph.ps1").is_file()
    for removed in (
        "setup.ps1",
        "onboard.ps1",
        "start.ps1",
        "stop.ps1",
        "dev.ps1",
        "doctor.ps1",
        "configure-llm.ps1",
        "bootstrap-all.ps1",
        "dev-all.ps1",
        "start-all.ps1",
        "stop-all.ps1",
        "verify-all.ps1",
        "run-python-cli.mjs",
    ):
        assert not (SCRIPTS / removed).exists(), removed


def test_cli_exposes_only_the_five_public_commands() -> None:
    completed = _run_cli("--help")
    assert completed.returncode == 0, completed.stderr
    for command in PUBLIC_COMMANDS:
        assert command in completed.stdout
    for removed in ("setup", "dev", "export-github"):
        assert f"    {removed} " not in completed.stdout


@pytest.mark.parametrize("command", PUBLIC_COMMANDS)
def test_every_public_command_has_help(command: str) -> None:
    completed = _run_cli(command, "--help")
    assert completed.returncode == 0, completed.stderr


def test_onboard_and_reconfigure_expose_exactly_three_llm_inputs() -> None:
    for command in ("onboard", "configure-llm"):
        help_text = _run_cli(command, "--help").stdout
        for option in ("--api-base", "--model", "--api-key-stdin"):
            assert option in help_text
        for removed in (
            "--preset",
            "--api-mode",
            "--protocol",
            "--auth-scheme",
            "--anthropic-version",
            "--timeout-seconds",
            "--allow-insecure-loopback",
            "--wheel-profile",
            "--device-policy",
            "--env-mode",
        ):
            assert removed not in help_text
    assert "--llm-mode" not in _run_cli("start", "--help").stdout


def test_noninteractive_llm_configuration_fails_before_install_when_incomplete() -> None:
    completed = _run_cli("onboard")
    assert completed.returncode == 1
    assert "cancelled" in completed.stderr


def test_single_powershell_wrapper_delegates_to_the_python_entrypoint() -> None:
    content = _read("scripts/socialgraph.ps1")
    assert "PythonLauncher.ps1" in content
    assert "socialgraph.py" in content
    assert "ValueFromRemainingArguments" in content


def test_root_npm_scripts_are_developer_only() -> None:
    scripts = json.loads(_read("package.json"))["scripts"]
    assert set(scripts) == {
        "build:web",
        "bundle:web",
        "check:web-bundle",
        "test:web",
        "typecheck:web",
        "test:e2e",
    }
    assert all("socialgraph.py" not in command for command in scripts.values())


def test_provider_is_one_protocol_and_keeps_network_safety_defaults() -> None:
    provider = _read("services/api/app/provider.py")
    settings = _read("services/api/app/config.py")
    assert "chat/completions" in settings
    assert "derive_llm_endpoint" in provider
    assert '"temperature": 0' in provider
    assert '"max_tokens": 700' in provider
    assert "trust_env=False" in provider
    assert "follow_redirects=False" in provider
    assert "anthropic_messages" not in provider
    assert "LLM_API_MODE" not in settings
    assert "llm_api_base, llm_api_key, and llm_model" in settings


def test_runtime_keeps_llm_key_out_of_the_gfm_process() -> None:
    operations = _read("packages/runtime/src/socialgraph_fm_runtime/operations.py")
    assert "LLM_API_KEY" in operations
    assert 'for name in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL")' in operations
    assert "api_environment[name] = private[name]" in operations
    assert "gfm_environment" in operations
    assert "SOCIALGRAPH_WEB_CLIENT_ROOT" in operations


def test_removed_legacy_powershell_runtime_does_not_return() -> None:
    retained = {path.name for path in (SCRIPTS / "lib").glob("*.ps1")}
    assert retained == {"NativeCommand.ps1", "PythonLauncher.ps1"}


def test_ci_and_repository_have_no_public_cuda_or_macos_workflow() -> None:
    workflow = _read(".github/workflows/ci.yml")
    assert "windows-latest" in workflow and "ubuntu-latest" in workflow
    assert "macos-" not in workflow
    assert "self-hosted" not in workflow
    assert "wheel-profile" not in workflow
    assert "device-policy" not in workflow
    assert not (PROJECT_ROOT / ".github" / "workflows" / "compatibility-macos.yml").exists()
    assert not (PROJECT_ROOT / ".github" / "workflows" / "release-cuda.yml").exists()
    assert not (PROJECT_ROOT / "packages" / "gfm" / "Dockerfile").exists()


def test_public_runtime_bundle_and_web_bundle_are_hash_bound() -> None:
    runtime = json.loads(_read("bundles/runtime-manifest.json"))
    assert runtime["schemaVersion"] == "socialgraph-fm.runtime-bundle/1.0"
    assert "bundles/web" in runtime["contentRoots"]
    assert any(item["role"] == "web" for item in runtime["assets"])
    web = json.loads(_read("bundles/web/manifest.json"))
    assert web["schemaVersion"] == "socialgraph-fm.web-bundle/1.0"
    assert web["archive"]["path"] == "bundles/web/client.zip"


def test_windows_wrapper_is_syntax_valid_when_powershell_is_available() -> None:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    completed = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPTS / "socialgraph.ps1"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
