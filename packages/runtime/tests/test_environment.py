from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from socialgraph_fm_runtime import environment
from socialgraph_fm_runtime.layout import RuntimeLayout


PROJECT = Path(__file__).resolve().parents[2]


def test_candidate_discovery_never_adopts_ambient_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "ambient"))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "conda"))
    assert environment.candidate_pythons(None) == []


def test_clean_python_environment_removes_credentials_and_package_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.invalid")
    monkeypatch.setenv("PYTHONPATH", "ambient")
    clean = environment.clean_process_environment()
    assert "OPENAI_API_KEY" not in clean
    assert "LLM_API_KEY" not in clean
    assert "PIP_INDEX_URL" not in clean
    assert clean["PYTHONPATH"] == ""


def test_clean_python_always_disables_bytecode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(environment, "run_captured_process", fake_run)

    environment.run_clean_python(Path(sys.executable), ("-I", "-c", "pass"), cwd=tmp_path)

    assert captured[1:3] == ["-B", "-I"]


def test_clean_python_import_does_not_create_bytecode(tmp_path: Path) -> None:
    module = tmp_path / "clean_probe_fixture.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    source = (
        "import sys; "
        f"sys.path.insert(0, {str(tmp_path)!r}); "
        "import clean_probe_fixture; "
        "assert clean_probe_fixture.VALUE == 1"
    )

    completed = environment.run_clean_python(
        Path(sys.executable), ("-I", "-c", source), cwd=tmp_path
    )

    assert completed.returncode == 0
    assert not (tmp_path / "__pycache__").exists()


@pytest.mark.parametrize(
    ("version", "requirement", "expected"),
    [
        ("3.12.4", ">=3.12,<3.13", True),
        ("3.13.0", ">=3.12,<3.13", False),
        ("2.8.0+cpu", "==2.8.0", True),
    ],
)
def test_requirement_evaluation(
    version: str, requirement: str, expected: bool
) -> None:
    assert environment.distribution_satisfies(version, requirement) is expected


def test_checked_in_install_profiles_are_only_windows_and_linux_cpu() -> None:
    profiles = environment.load_install_profiles(
        PROJECT / "gfm" / "install-profiles.json"
    )
    assert set(profiles) == {
        "windows-x86_64-cpu-pt28",
        "linux-x86_64-cpu-pt28",
    }
    assert all("+cpu" in profile["distributionVersions"]["torch"] for profile in profiles.values())
    assert all("ogb" not in profile["distributionVersions"] for profile in profiles.values())


def test_install_profile_is_selected_only_from_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles = {
        "windows": {"id": "windows", "system": "Windows", "machine": "x86_64"},
        "linux": {
            "id": "linux",
            "system": "Linux",
            "machine": "x86_64",
            "libc": "glibc",
        },
    }
    monkeypatch.setattr(environment, "normalized_platform", lambda **_kwargs: ("linux", "x86_64"))
    monkeypatch.setattr(environment.platform, "libc_ver", lambda: ("glibc", "2.39"))
    assert environment.select_install_profile(profiles)["id"] == "linux"

    monkeypatch.setattr(environment, "normalized_platform", lambda **_kwargs: ("darwin", "arm64"))
    with pytest.raises(RuntimeError, match="supports only"):
        environment.select_install_profile(profiles)


def test_runtime_probe_requires_cpu_torch_and_neighbor_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = {
        "id": "fixture",
        "system": "Windows",
        "machine": "x86_64",
        "pythonRequires": ">=3.12,<3.13",
        "distributionVersions": {
            "torch": "2.8.0+cpu",
            "torch-geometric": "2.8.0.post1",
            "pyg-lib": "0.6.0+pt28cpu",
        },
    }
    report = {
        "executable": sys.executable,
        "executableSha256": "a" * 64,
        "implementation": "CPython",
        "pythonVersion": "3.12.4",
        "system": "Windows",
        "machine": "AMD64",
        "libc": None,
        "libcVersion": None,
        "versions": dict(profile["distributionVersions"]),
        "imports": {
            "torch": True,
            "torch_geometric": True,
            "pyg_lib": True,
        },
        "torch": {"available": True, "version": "2.8.0+cpu", "cpuBuild": True},
        "neighborLoader": True,
    }
    monkeypatch.setattr(environment, "probe_python", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(environment, "pip_check", lambda _python: (True, "ok"))
    result = environment.probe_runtime_environment(Path(sys.executable), profile)
    assert result.compatible
    assert result.fingerprint["torch"]["cpuBuild"] is True


def test_managed_install_uses_one_hash_locked_binary_only_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    lock = project / "packages" / "gfm" / "locks" / "cpu.txt"
    lock.parent.mkdir(parents=True)
    lock.write_text("fixture", encoding="utf-8")
    layout = RuntimeLayout(project)
    profile = {
        "id": "fixture",
        "pythonRequires": ">=3.12,<3.13",
        "requirementsLock": "locks/cpu.txt",
        "requirementsLockSha256": environment._sha256(lock),
        "indexUrl": "https://pypi.org/simple",
        "torchIndexUrl": "https://download.pytorch.org/whl/cpu",
        "findLinks": ["https://data.pyg.org/whl/torch-2.8.0+cpu.html"],
    }
    python = tmp_path / "stage" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        environment,
        "probe_python",
        lambda *_args, **_kwargs: {"pythonVersion": "3.12.4"},
    )
    monkeypatch.setattr(environment, "create_venv", lambda *_args, **_kwargs: python)
    monkeypatch.setattr(
        environment,
        "_pip_install",
        lambda _python, arguments, **_kwargs: calls.append(tuple(arguments)),
    )
    monkeypatch.setattr(environment, "prune_torch_build_assets", lambda _python: ())

    assert (
        environment.install_runtime_environment(
            layout,
            Path(sys.executable),
            profile,
            destination=tmp_path / "stage",
        )
        == python
    )
    assert len(calls) == 1
    assert "--require-hashes" in calls[0]
    assert "--only-binary=:all:" in calls[0]
    assert "https://download.pytorch.org/whl/cpu" in calls[0]


def test_pip_install_disables_bytecode_and_wheel_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)

    monkeypatch.setattr(environment, "run_streaming_process", fake_run)

    environment._pip_install(
        Path(sys.executable),
        ("--require-hashes", "-r", "runtime.txt"),
        cwd=tmp_path,
    )

    assert captured[1:3] == ["-B", "-I"]
    assert "--no-compile" in captured
    assert "--no-cache-dir" in captured
    assert "--no-deps" not in captured


def _bytecode_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    runtime = tmp_path / "runtime"
    python = runtime / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    purelib = runtime / "site-packages"
    package = purelib / "fixture"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    cache.write_bytes(b"source-backed-bytecode")
    return python, purelib, cache


def test_bytecode_pruning_is_source_backed_counted_and_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python, purelib, cache = _bytecode_fixture(tmp_path)
    orphan = cache.with_name("orphan.cpython-312.pyc")
    orphan.write_bytes(b"retain-sourceless")
    direct = purelib / "direct.pyc"
    direct.write_bytes(b"retain-outside-pycache")
    monkeypatch.setattr(environment, "_purelib", lambda _python: purelib)

    first = environment.prune_runtime_bytecode(python)
    second = environment.prune_runtime_bytecode(python)

    assert first == environment.BytecodePruneResult(
        removed_files=1, removed_bytes=len(b"source-backed-bytecode")
    )
    assert second == environment.BytecodePruneResult(removed_files=0, removed_bytes=0)
    assert not cache.exists()
    assert orphan.is_file()
    assert direct.is_file()


def test_bytecode_pruning_refuses_linked_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python, purelib, cache = _bytecode_fixture(tmp_path)
    original = environment._is_link_or_reparse_point
    monkeypatch.setattr(environment, "_purelib", lambda _python: purelib)
    monkeypatch.setattr(
        environment,
        "_is_link_or_reparse_point",
        lambda path: path == cache or original(path),
    )

    result = environment.prune_runtime_bytecode(python)

    assert result == environment.BytecodePruneResult(removed_files=0, removed_bytes=0)
    assert cache.is_file()


def test_bytecode_pruning_rejects_purelib_outside_managed_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    python, _purelib, _cache = _bytecode_fixture(tmp_path)
    outside = tmp_path / "outside-site-packages"
    outside.mkdir()
    monkeypatch.setattr(environment, "_purelib", lambda _python: outside)

    with pytest.raises(RuntimeError, match="escapes"):
        environment.prune_runtime_bytecode(python)


def test_torch_pruning_removes_only_version_pinned_compiler_assets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "runtime"
    python = root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python.parent.mkdir(parents=True)
    python.touch()
    site = root / "site-packages"
    for path in (site / "torch" / "include", site / "torch" / "share" / "cmake"):
        path.mkdir(parents=True)
        (path / "fixture.h").write_text("x", encoding="utf-8")
    library = site / "torch" / "lib"
    library.mkdir(parents=True)
    removable = library / ("fixture.lib" if os.name == "nt" else "fixture.a")
    retained = library / ("fixture.dll" if os.name == "nt" else "fixture.so")
    removable.write_text("x", encoding="utf-8")
    retained.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        environment,
        "probe_python",
        lambda *_args, **_kwargs: {
            "versions": {"torch": "2.8.0+cpu"},
            "torch": {"cpuBuild": True},
        },
    )
    monkeypatch.setattr(environment, "_purelib", lambda _python: site)

    removed = environment.prune_torch_build_assets(python)

    assert removed
    assert not removable.exists()
    assert retained.is_file()
    assert not (site / "torch" / "include").exists()
