import re
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

import pytest

import socialgraph_gfm.runtime as runtime_module
from socialgraph_gfm.preflight import preflight_report
from socialgraph_gfm.runtime import (
    GIB,
    ArtifactRootNotConfigured,
    InsufficientDiskSpace,
    RuntimeLayout,
    artifact_root,
    core_runtime_root,
    gfm_optional_runtime_report,
    prepare_runtime_layout,
    require_storage_reserve,
    storage_report,
)

DiskUsage = namedtuple("DiskUsage", "total used free")


def test_artifact_root_requires_explicit_environment_or_override(monkeypatch, tmp_path):
    monkeypatch.delenv("SOCIALGRAPH_FM_HOME", raising=False)

    with pytest.raises(ArtifactRootNotConfigured, match="SOCIALGRAPH_FM_HOME"):
        artifact_root()

    assert artifact_root(tmp_path) == tmp_path.resolve()


def test_core_runtime_root_is_derived_from_socialgraph_home(monkeypatch, tmp_path):
    home = tmp_path / "var" / "gfm"
    monkeypatch.setenv("SOCIALGRAPH_FM_HOME", str(home))

    assert core_runtime_root() == (home / "core-runtime").resolve()


def test_runtime_layout_contains_every_heavyweight_directory(tmp_path):
    layout = RuntimeLayout(tmp_path)
    relative = {path.relative_to(tmp_path).as_posix() for path in layout.directories()}
    assert relative == {
        "datasets/raw/ogb",
        "datasets/raw/gfm/openalex",
        "datasets/raw/gfm/thgl-software",
        "datasets/raw/gfm/wikimedia-talk",
        "datasets/packages",
        "datasets/processed",
        "datasets/processed/gfm",
        "datasets/manifests",
        "datasets/manifests/gfm",
        "embeddings",
        "runs",
        "runs/gfm",
        "registry",
        "reports",
        "reports/gfm",
        "models/staging",
        "models/released",
        "cache/hf",
        "cache/pip",
        "cache/uv",
        "cache/torch",
        "cache/torchinductor",
        "cache/wandb",
        "tmp",
        "exports",
    }


def test_storage_reserves_are_30_gib_for_fetch_and_20_gib_for_run(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runtime_module.shutil,
        "disk_usage",
        lambda _path: DiskUsage(100 * GIB, 75 * GIB, 25 * GIB),
    )
    fetch = storage_report(tmp_path, operation="fetch")
    run = storage_report(tmp_path, operation="run")
    assert fetch["minimumFreeGiB"] == 30
    assert fetch["ready"] is False
    assert run["minimumFreeGiB"] == 20
    assert run["ready"] is True


def test_layout_is_not_created_when_storage_gate_fails(monkeypatch, tmp_path):
    root = tmp_path / "not-created"
    monkeypatch.setattr(
        runtime_module.shutil,
        "disk_usage",
        lambda _path: DiskUsage(100 * GIB, 81 * GIB, 19 * GIB),
    )
    with pytest.raises(InsufficientDiskSpace, match="20 GiB required"):
        prepare_runtime_layout(root, operation="run")
    assert not root.exists()


def test_prepare_runtime_layout_checks_then_creates(monkeypatch, tmp_path):
    root = tmp_path / "runtime"
    monkeypatch.setattr(
        runtime_module.shutil,
        "disk_usage",
        lambda _path: DiskUsage(100 * GIB, 60 * GIB, 40 * GIB),
    )
    layout = prepare_runtime_layout(root, operation="fetch")
    assert layout.root == root.resolve()
    assert all(path.is_dir() for path in layout.directories())
    assert require_storage_reserve(root, operation="run")["ready"] is True


def test_runtime_launcher_only_uses_process_environment_and_has_no_cleanup_commands():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "Enter-BaselineRuntime.ps1"
    ).read_text(encoding="utf-8")
    lowered = script.lower()
    assert "setx" not in lowered
    assert "[system.environment]::setenvironmentvariable" not in lowered
    assert "remove-item" not in lowered
    for variable in (
        "SOCIALGRAPH_FM_HOME",
        "SOCIALGRAPH_GFM_PYTHON",
        "PIP_CACHE_DIR",
        "UV_CACHE_DIR",
        "HF_HOME",
        "TORCH_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "WANDB_DIR",
        "TEMP",
        "TMP",
    ):
        assert f"$env:{variable}" in script


def test_gfm_runtime_launcher_delegates_to_audited_process_scoped_launcher():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "Enter-GfmRuntime.ps1").read_text(
        encoding="utf-8"
    )
    lowered = script.lower()
    assert "enter-baselineruntime.ps1" in lowered
    assert "$env:socialgraph_fm_home" in lowered
    assert "$env:socialgraph_gfm_python" in lowered
    assert "dependencyprofile" not in lowered
    assert "cu130" not in lowered
    assert "flagembedding" not in lowered
    assert "transformers" not in lowered
    assert 'notepropertyvalue "cpu"' in lowered
    assert "setx" not in lowered
    assert "remove-item" not in lowered


def test_windows_scripts_have_no_workstation_specific_defaults():
    package_root = Path(__file__).resolve().parents[1]
    repository_root = package_root.parents[1]
    script_paths = tuple(sorted((package_root / "scripts").glob("*.ps1")))
    assert {path.name for path in script_paths} == {
        "Enter-BaselineRuntime.ps1",
        "Enter-GfmRuntime.ps1",
        "Invoke-GfmCorpusContinue.ps1",
        "Invoke-GfmCorpusSetup.ps1",
        "Invoke-GfmDevAfterEmbedding.ps1",
        "Invoke-OgblCollabBaseline.ps1",
        "generate_locks.ps1",
    }
    legacy_root = repository_root / "scripts" / "legacy"
    assert not any(path.is_file() for path in legacy_root.rglob("*"))

    for script_path in script_paths:
        script = script_path.read_text(encoding="utf-8")
        assert re.search(r"(?i)\b[a-z]:[\\/]", script) is None, script_path
        lowered = script.lower()
        assert "\\users\\" not in lowered, script_path
        assert "/users/" not in lowered, script_path


def test_dev_after_embedding_automation_is_bounded_to_dev_and_collaboration():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "Invoke-GfmDevAfterEmbedding.ps1"
    ).read_text(encoding="utf-8")
    lowered = script.lower()
    assert "gfm-text-embed" in lowered and "wikimedia-talk" in lowered
    assert "gfm-task-assets" in lowered and '"collaboration"' in lowered
    assert "gfm-pretrain" in lowered and '"dev"' in lowered
    assert "socialgraph-core.json" in lowered
    assert '"core-base"' in lowered and '"core-moe"' in lowered
    assert "freephysicalmemory" in lowered
    assert "[validaterange(8, 64)]" in lowered
    assert "$minimumfreememorygib = 8" in lowered
    assert '"waiting-for-dev-memory"' in lowered
    assert '"waiting-for-core-moe-memory"' in lowered
    assert "$memorywaittimeoutminutes" in lowered
    assert "start-sleep" in lowered
    assert "[io.fileshare]::none" in lowered
    assert "dev-after-wikimedia-embedding.owner.json" in lowered
    assert "get-otherautomationprocesses" in lowered
    assert "test-automationprocess" in lowered
    assert "recoveredstaleownerattemptid" in lowered
    assert "automationpid" in lowered and "attemptid" in lowered
    assert "memorywaitdeadline" in lowered and "codehash" in lowered
    assert "get-codeidentity" in lowered and "assert-codeidentity" in lowered
    assert "code_identity_hash" in lowered
    assert "refusing to overwrite its state" in lowered
    assert "restart the automation explicitly" in lowered
    assert '"--phase" "formal"' not in " ".join(lowered.split())
    assert '"newcomer"' not in lowered
    assert "gfm-evaluate" not in lowered
    assert "gfm-adapt" not in lowered


def test_gfm_runtime_secret_prompt_is_scoped_and_zeroes_unmanaged_buffers():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "Enter-GfmRuntime.ps1").read_text(
        encoding="utf-8"
    )
    lowered = script.lower()
    assert "$promptforsecrets" in lowered
    assert "$secretaction" in lowered
    assert "read-host" in lowered and "-assecurestring" in lowered
    assert "securestringtobstr" in lowered
    assert "zerofreebstr" in lowered
    assert "[environmentvariabletarget]::process" in lowered
    assert "finally" in lowered
    assert "openalex_api_key" in lowered
    assert "socialgraph_gfm_pseudonym_salt" in lowered
    assert "[environmentvariabletarget]::user" not in lowered
    assert "[environmentvariabletarget]::machine" not in lowered
    assert "convertfrom-securestring" not in lowered
    assert "write-host" not in lowered
    assert "write-output" not in lowered


def test_gfm_runtime_wikimedia_only_prompt_restores_salt_and_does_not_touch_openalex(
    tmp_path,
):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required to exercise the Windows runtime launcher")
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "Enter-GfmRuntime.ps1"

    def ps_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    fake_salt = "fixture-wikimedia-salt-is-longer-than-32-bytes"
    command = f"""
      $ErrorActionPreference = 'Stop'
      function global:Read-Host {{
        param([string]$Prompt, [switch]$AsSecureString)
        if ($Prompt -notlike 'Stable Wikimedia*' -or -not $AsSecureString) {{
          throw 'unexpected secret prompt'
        }}
        ConvertTo-SecureString {ps_quote(fake_salt)} -AsPlainText -Force
      }}
      function global:Get-PSDrive {{
        param([string]$Name, [string]$PSProvider)
        [pscustomobject]@{{ Free = [int64]40 * 1GB }}
      }}
      $apiVariable = 'OPENALEX_API_KEY'
      $saltVariable = 'SOCIALGRAPH_GFM_PSEUDONYM_SALT'
      [Environment]::SetEnvironmentVariable($apiVariable, 'api-sentinel', [EnvironmentVariableTarget]::Process)
      [Environment]::SetEnvironmentVariable($saltVariable, 'salt-sentinel', [EnvironmentVariableTarget]::Process)
      $caughtExpectedFailure = $false
      try {{
        & {ps_quote(launcher)} `
          -RuntimeRoot {ps_quote(tmp_path / "runtime")} `
          -GfmPython {ps_quote(sys.executable)} `
          -Operation run `
          -PromptForWikimediaSalt `
          -SecretAction {{
            param($Runtime)
            if ($env:SOCIALGRAPH_GFM_PSEUDONYM_SALT -ne {ps_quote(fake_salt)}) {{ throw 'salt unavailable to action' }}
            if ($env:OPENALEX_API_KEY -ne 'api-sentinel') {{ throw 'OpenAlex variable changed in salt-only mode' }}
            if ($Runtime.RuntimeRoot -ne [IO.Path]::GetFullPath({ps_quote(tmp_path / "runtime")})) {{ throw 'runtime context mismatch' }}
            throw 'fixture-action-failure'
          }}
      }} catch {{
        if ($_.Exception.Message -ne 'fixture-action-failure') {{ throw }}
        $caughtExpectedFailure = $true
      }}
      if (-not $caughtExpectedFailure) {{ throw 'fixture action did not fail as expected' }}
      if ($env:SOCIALGRAPH_GFM_PSEUDONYM_SALT -ne 'salt-sentinel') {{ throw 'salt was not restored' }}
      if ($env:OPENALEX_API_KEY -ne 'api-sentinel') {{ throw 'OpenAlex variable was not preserved' }}
    """
    completed = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_gfm_runtime_existing_both_secret_prompt_restores_both(tmp_path):
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is required to exercise the Windows runtime launcher")
    launcher = Path(__file__).resolve().parents[1] / "scripts" / "Enter-GfmRuntime.ps1"

    def ps_quote(value):
        return "'" + str(value).replace("'", "''") + "'"

    fake_api = "fixture-openalex-key"
    fake_salt = "fixture-wikimedia-salt-is-longer-than-32-bytes"
    command = f"""
      $ErrorActionPreference = 'Stop'
      function global:Read-Host {{
        param([string]$Prompt, [switch]$AsSecureString)
        if (-not $AsSecureString) {{ throw 'plaintext prompt requested' }}
        if ($Prompt -eq 'OpenAlex API key') {{
          return ConvertTo-SecureString {ps_quote(fake_api)} -AsPlainText -Force
        }}
        if ($Prompt -like 'Stable Wikimedia*') {{
          return ConvertTo-SecureString {ps_quote(fake_salt)} -AsPlainText -Force
        }}
        throw 'unexpected secret prompt'
      }}
      function global:Get-PSDrive {{
        param([string]$Name, [string]$PSProvider)
        [pscustomobject]@{{ Free = [int64]40 * 1GB }}
      }}
      $apiVariable = 'OPENALEX_API_KEY'
      $saltVariable = 'SOCIALGRAPH_GFM_PSEUDONYM_SALT'
      [Environment]::SetEnvironmentVariable($apiVariable, 'api-sentinel', [EnvironmentVariableTarget]::Process)
      [Environment]::SetEnvironmentVariable($saltVariable, 'salt-sentinel', [EnvironmentVariableTarget]::Process)
      $caughtExpectedFailure = $false
      try {{
        & {ps_quote(launcher)} `
          -RuntimeRoot {ps_quote(tmp_path / "runtime")} `
          -GfmPython {ps_quote(sys.executable)} `
          -Operation run `
          -PromptForSecrets `
          -SecretAction {{
            param($Runtime)
            if ($env:OPENALEX_API_KEY -ne {ps_quote(fake_api)}) {{ throw 'OpenAlex key unavailable to action' }}
            if ($env:SOCIALGRAPH_GFM_PSEUDONYM_SALT -ne {ps_quote(fake_salt)}) {{ throw 'salt unavailable to action' }}
            throw 'fixture-action-failure'
          }}
      }} catch {{
        if ($_.Exception.Message -ne 'fixture-action-failure') {{ throw }}
        $caughtExpectedFailure = $true
      }}
      if (-not $caughtExpectedFailure) {{ throw 'fixture action did not fail as expected' }}
      if ($env:OPENALEX_API_KEY -ne 'api-sentinel') {{ throw 'OpenAlex key was not restored' }}
      if ($env:SOCIALGRAPH_GFM_PSEUDONYM_SALT -ne 'salt-sentinel') {{ throw 'salt was not restored' }}
    """
    completed = subprocess.run(
        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_corpus_continuation_skips_openalex_network_and_newcomer_overlay():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "Invoke-GfmCorpusContinue.ps1"
    ).read_text(encoding="utf-8")
    lowered = script.lower()
    assert "enter-gfmruntime.ps1" in lowered
    assert "-promptforwikimediasalt" in lowered
    assert "-promptforsecrets" not in lowered
    assert "gfm-corpus-fetch-openalex" not in lowered
    assert '"--domain" "openalex"' in lowered
    assert '"--newcomer-overlay" "skip"' in lowered
    assert '"--domain" "thgl-software"' in lowered
    assert '"--domain" "wikimedia-talk"' in lowered
    assert "verify-openalex-newcomers" not in lowered
    assert "read-host" not in lowered
    assert "openalex_api_key" not in lowered
    assert "setx" not in lowered
    assert "environmentvariabletarget" not in lowered


def test_corpus_setup_uses_only_the_scoped_secret_launcher():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "Invoke-GfmCorpusSetup.ps1"
    ).read_text(encoding="utf-8")
    lowered = script.lower()
    assert "enter-gfmruntime.ps1" in lowered
    assert "-promptforsecrets" in lowered
    assert "-secretaction" in lowered
    assert "gfm-corpus-fetch-openalex" in lowered
    assert '"--api-key-env" "openalex_api_key"' in lowered
    assert '"--domain" "openalex"' in lowered
    assert '"--newcomer-overlay" "skip"' in lowered
    assert '"--domain" "thgl-software"' in lowered
    assert '"--domain" "wikimedia-talk"' in lowered
    assert "invoke-gfmcli @(" not in lowered
    assert "read-host" not in lowered
    assert "setx" not in lowered
    assert "environmentvariabletarget" not in lowered


def test_empty_runtime_keeps_baseline_separate_from_model_and_serving_readiness(tmp_path):
    readiness = preflight_report(device="cpu", root=tmp_path)["readiness"]
    assert readiness["CorpusReady"] is False
    assert readiness["BaselineValidated"] is False
    assert readiness["GfmCorpusReady"] is False
    assert readiness["NewcomerOverlayReady"] is False
    assert readiness["GfmPretrainingValidated"] is False
    assert readiness["GfmProductValidated"] is False
    assert readiness["ModelValidated"] is False
    assert readiness["GfmServingReady"] is False


def test_optional_gfm_dependencies_do_not_weaken_base_runtime_contract():
    report = gfm_optional_runtime_report()
    assert report["schemaVersion"] == "gfm.optional-runtime/1.0"
    assert set(report["expected"]) == {"FlagEmbedding", "transformers"}
    assert report["dataReady"] is True
    assert isinstance(report["textReady"], bool)
