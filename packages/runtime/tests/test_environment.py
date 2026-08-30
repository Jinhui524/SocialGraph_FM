from __future__ import annotations

import json
import hashlib
import os
import sys
from pathlib import Path

import pytest

import socialgraph_fm_runtime.environment as environment


def test_auto_candidates_do_not_adopt_arbitrary_system_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    assert environment.candidate_pythons(None, managed=tmp_path / "missing") == []


def test_explicit_candidate_is_first_and_active_environment_is_authorized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    executable = Path(sys.executable).resolve()
    root = executable.parent.parent if executable.parent.name.lower() in {"bin", "scripts"} else tmp_path
    monkeypatch.setenv("VIRTUAL_ENV", str(root))
    candidates = environment.candidate_pythons(str(executable))
    assert candidates[0] == executable


def test_clean_python_does_not_inherit_pythonpath_or_llm_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "sentinel-python-path")
    monkeypatch.setenv("LLM_API_KEY", "sentinel-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sentinel-anthropic-secret")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sentinel-deepseek-secret")
    monkeypatch.setenv("PIP_INDEX_URL", "http://untrusted.invalid/simple")
    completed = environment.run_clean_python(
        Path(sys.executable),
        (
            "-c",
            "import json,os; print(json.dumps({'path':os.environ.get('PYTHONPATH'), "
            "'key':os.environ.get('LLM_API_KEY'), "
            "'anthropic':os.environ.get('ANTHROPIC_API_KEY'), "
            "'deepseek':os.environ.get('DEEPSEEK_API_KEY'), "
            "'pip':os.environ.get('PIP_INDEX_URL'), "
            "'pip_config':os.environ.get('PIP_CONFIG_FILE'), "
            "'no_user':os.environ.get('PYTHONNOUSERSITE')}))",
        ),
    )
    assert completed.returncode == 0
    document = json.loads(completed.stdout)
    assert document == {
        "path": "",
        "key": None,
        "anthropic": None,
        "deepseek": None,
        "pip": None,
        "pip_config": os.devnull,
        "no_user": "1",
    }


def test_managed_pip_commands_always_use_isolated_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)

    monkeypatch.setattr(environment, "run_streaming_process", fake_run)
    environment._pip_install(
        Path(sys.executable),
        ("--index-url", "https://pypi.org/simple", "fixture==1"),
        cwd=tmp_path,
    )

    assert captured[1:6] == ["-I", "-m", "pip", "--isolated", "install"]


@pytest.mark.parametrize(
    ("version", "requirement", "expected"),
    [
        ("3.12.4", ">=3.12,<3.13", True),
        ("3.11.9", ">=3.12,<3.13", False),
        ("2.13.5", ">=2.9,<3", True),
        ("3.0.0", ">=2.9,<3", False),
    ],
)
def test_requirement_evaluation(version: str, requirement: str, expected: bool) -> None:
    assert environment.python_satisfies(version, requirement) is expected


def test_install_profile_selection_uses_platform_and_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    catalog = tmp_path / "install-profiles.json"
    base = {
        "machine": "x86_64",
        "pythonRequires": ">=3.12,<3.13",
        "extra": "cpu",
        "torchBackend": "cpu",
        "indexUrl": "https://pypi.org/simple",
        "torchIndexUrl": "https://download.pytorch.org/whl/cpu",
        "findLinks": ["https://data.pyg.org/whl/test.html"],
        "distributionVersions": {"torch": "2.8.0+cpu"},
        "requirementsLock": "locks/test.txt",
    }
    catalog.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.gfm-install-profiles/1.0",
                "profiles": {
                    "windows-x86_64-cpu-pt28": {
                        **base,
                        "system": "Windows",
                        "device": "cpu",
                        "wheelFamily": "cpu",
                    },
                    "windows-x86_64-cu130-pt212": {
                        **base,
                        "system": "Windows",
                        "device": "cuda",
                        "wheelFamily": "cuda",
                        "torchBackend": "cu130",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    locks = tmp_path / "locks"
    locks.mkdir()
    lock = locks / "test.txt"
    lock.write_text("fixture==1 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8")
    constraints = tmp_path / "constraints"
    constraints.mkdir()
    build_requirements = constraints / "install-build.txt"
    build_requirements.write_text("hatchling==1.27.0\n", encoding="utf-8")
    catalog_hash = hashlib.sha256(catalog.read_bytes()).hexdigest()
    lock_hash = hashlib.sha256(lock.read_bytes()).hexdigest()
    (locks / "install-lock-manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.gfm-install-lock-manifest/1.0",
                "profilesFile": catalog.name,
                "profilesSha256": catalog_hash,
                "buildRequirementsFile": "constraints/install-build.txt",
                "buildRequirementsSha256": hashlib.sha256(
                    build_requirements.read_bytes()
                ).hexdigest(),
                "policy": {
                    "sourceBuildsAllowed": False,
                    "requireArtifactHashesForManagedInstall": True,
                },
                "profiles": {
                    identifier: {
                        "requirementsLock": "locks/test.txt",
                        "requirementsLockSha256": lock_hash,
                        "artifactHashesResolved": True,
                    }
                    for identifier in (
                        "windows-x86_64-cpu-pt28",
                        "windows-x86_64-cu130-pt212",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(environment, "normalized_platform", lambda: ("windows", "x86_64"))
    profiles = environment.load_install_profiles(catalog)
    assert environment.select_install_profile(profiles, "cpu")["id"] == (
        "windows-x86_64-cpu-pt28"
    )
    assert environment.select_install_profile(
        profiles, "windows-x86_64-cu130-pt212"
    )["id"] == "windows-x86_64-cu130-pt212"


def test_linux_musl_cannot_select_a_glibc_install_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(environment, "normalized_platform", lambda: ("linux", "x86_64"))
    monkeypatch.setattr(environment.platform, "libc_ver", lambda: ("musl", "1.2"))
    profiles = {
        "linux-x86_64-cpu-pt28": {
            "id": "linux-x86_64-cpu-pt28",
            "system": "Linux",
            "machine": "x86_64",
            "device": "cpu",
            "libc": "glibc",
        }
    }

    with pytest.raises(RuntimeError, match="No unique install profile"):
        environment.select_install_profile(profiles, "cpu")


@pytest.mark.skipif(os.name == "nt", reason="POSIX virtualenv launchers are symlinks")
def test_resolve_python_preserves_virtualenv_launcher_symlink(tmp_path: Path) -> None:
    base = tmp_path / "base-python"
    base.write_text("fixture", encoding="utf-8")
    launcher = tmp_path / "venv" / "bin" / "python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(base)

    assert environment.resolve_python(launcher) == launcher.absolute()


def test_api_probe_requires_every_declared_module_to_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {name: "1.0.0" for name in environment.API_DISTRIBUTIONS}
    versions.update({"torch": None, "torch-geometric": None})
    imports = {
        module: module != "multipart"
        for module in environment.API_DISTRIBUTIONS.values()
    }
    imports.update({"torch": False, "torch_geometric": False})
    report = {
        "executable": sys.executable,
        "executableSha256": "0" * 64,
        "implementation": "CPython",
        "pythonVersion": "3.12.4",
        "system": "Windows",
        "machine": "AMD64",
        "libc": None,
        "libcVersion": None,
        "versions": versions,
        "imports": imports,
        "torch": {
            "available": False,
            "version": None,
            "cudaRuntime": None,
            "cudaAvailable": False,
            "deviceName": None,
        },
        "neighborLoader": None,
    }
    monkeypatch.setattr(environment, "probe_python", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(environment, "pip_check", lambda _python: (True, "ok"))

    result = environment.probe_api_environment(Path(sys.executable))

    assert result.compatible is False
    assert "cannot import multipart (python-multipart)" in result.errors
    assert result.fingerprint["imports"]["multipart"] is False


def test_software_fingerprint_excludes_dynamic_cuda_hardware() -> None:
    base = {
        "executable": "python",
        "executableSha256": "a" * 64,
        "implementation": "CPython",
        "pythonVersion": "3.12.4",
        "system": "Windows",
        "machine": "AMD64",
        "libc": None,
        "libcVersion": None,
        "versions": {"torch": "2.12.0+cu130"},
        "imports": {"torch": True},
        "neighborLoader": True,
    }
    first = environment._fingerprint(
        {
            **base,
            "torch": {
                "available": True,
                "version": "2.12.0+cu130",
                "cudaRuntime": "13.0",
                "cudaAvailable": False,
                "deviceName": None,
            },
        },
        "gfm",
    )
    second = environment._fingerprint(
        {
            **base,
            "torch": {
                "available": True,
                "version": "2.12.0+cu130",
                "cudaRuntime": "13.0",
                "cudaAvailable": True,
                "deviceName": "different GPU",
            },
        },
        "gfm",
    )
    assert first == second


@pytest.mark.parametrize(
    ("wheel_family", "policy", "cuda_available", "resolved", "fallback"),
    [
        ("cpu", "auto", False, "cpu", "cpu-wheel"),
        ("cpu", "cpu", False, "cpu", "policy-forced-cpu"),
        ("cuda", "auto", False, "cpu", "cuda-unavailable"),
        ("cuda", "auto", True, "cuda", None),
        ("cuda", "cpu", True, "cpu", "policy-forced-cpu"),
        ("cuda", "cuda-required", True, "cuda", None),
    ],
)
def test_wheel_selection_and_execution_device_are_independent(
    wheel_family: str,
    policy: str,
    cuda_available: bool,
    resolved: str,
    fallback: str | None,
) -> None:
    result = environment.CompatibilityResult(
        True,
        (),
        {},
        {"cudaAvailable": cuda_available},
    )
    resolution = environment.resolve_execution_device(
        {"id": "fixture", "device": wheel_family}, result, policy
    )
    assert resolution.resolved_device == resolved
    assert resolution.fallback_reason == fallback


def test_cuda_required_fails_closed_without_cuda() -> None:
    result = environment.CompatibilityResult(
        True, (), {}, {"cudaAvailable": False}
    )
    with pytest.raises(RuntimeError, match="required"):
        environment.resolve_execution_device(
            {"id": "fixture", "device": "cuda"}, result, "cuda-required"
        )


def test_cuda_wheel_probe_is_compatible_without_a_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "torch": "2.12.0+cu130",
        "torch-geometric": "2.8.0.post1",
        "pyg-lib": "0.7.0+pt212cu130",
        "numpy": "2.3.3",
        "pydantic": "2.13.4",
        "ogb": "1.3.6",
    }
    report = {
        "executable": sys.executable,
        "executableSha256": "0" * 64,
        "implementation": "CPython",
        "pythonVersion": "3.12.4",
        "system": "Windows",
        "machine": "AMD64",
        "libc": None,
        "libcVersion": None,
        "versions": versions,
        "imports": {
            environment.GFM_MODULES[name]: True for name in versions
        },
        "torch": {
            "available": True,
            "version": versions["torch"],
            "cudaRuntime": "13.0",
            "cudaAvailable": False,
            "deviceName": None,
            "deviceCount": 0,
            "deviceCapability": None,
        },
        "neighborLoader": True,
    }
    profile = {
        "id": "windows-x86_64-cu130-pt212",
        "system": "Windows",
        "machine": "x86_64",
        "device": "cuda",
        "pythonRequires": ">=3.12,<3.13",
        "torchBackend": "cu130",
        "distributionVersions": versions,
    }
    monkeypatch.setattr(environment, "probe_python", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(environment, "pip_check", lambda _python: (True, "ok"))

    result = environment.probe_gfm_environment(Path(sys.executable), profile)

    assert result.compatible is True
    assert result.runtime_capabilities["cudaAvailable"] is False
    assert environment.resolve_execution_device(
        profile, result, "auto"
    ).resolved_device == "cpu"
