from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from socialgraph_fm_runtime.profile import (
    FINGERPRINT_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION,
    RuntimeProfile,
)
from socialgraph_fm_runtime.layout import RuntimeLayout


def test_runtime_profile_v3_is_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config" / "runtime-profile.json"
    profile = RuntimeProfile.create(
        profile="cpu",
        env_mode="reuse",
        install_profile_id="windows-x86_64-cpu-pt28",
        platform={"system": "windows", "machine": "x86_64"},
        interpreters={
            "bootstrap": {"path": "bootstrap", "fingerprint": {"fingerprintSha256": "a"}},
            "api": {"path": "api", "fingerprint": {"fingerprintSha256": "b"}},
            "gfm": {"path": "gfm", "fingerprint": {"fingerprintSha256": "c"}},
        },
        device_policy="cpu",
    )
    profile.write(path)
    loaded = RuntimeProfile.load(path)
    assert loaded.to_document() == profile.to_document()
    assert json.loads(path.read_text(encoding="utf-8"))["schemaVersion"] == PROFILE_SCHEMA_VERSION
    assert not list(path.parent.glob("*.tmp"))


def test_runtime_profile_v2_migrates_dynamic_cuda_state_to_v3(tmp_path: Path) -> None:
    path = tmp_path / "runtime-profile.json"
    fingerprint = {
        "schemaVersion": "socialgraph-fm.python-environment-fingerprint/1.0",
        "capability": "gfm",
        "executable": "gfm-python",
        "executableSha256": "a" * 64,
        "implementation": "CPython",
        "pythonVersion": "3.12.4",
        "system": "Windows",
        "machine": "AMD64",
        "libc": None,
        "libcVersion": None,
        "versions": {"torch": "2.12.0+cu130"},
        "imports": {"torch": True},
        "torch": {
            "available": True,
            "version": "2.12.0+cu130",
            "cudaRuntime": "13.0",
            "cudaAvailable": True,
            "deviceName": "fixture GPU",
        },
        "neighborLoader": True,
        "fingerprintSha256": "b" * 64,
    }
    path.write_text(
        json.dumps(
            {
                "schemaVersion": "socialgraph-fm.runtime-profile/2.0",
                "profile": "cuda",
                "envMode": "managed",
                "installProfileId": "windows-x86_64-cu130-pt212",
                "platform": {"system": "windows", "machine": "x86_64"},
                "interpreters": {
                    "bootstrap": None,
                    "api": None,
                    "gfm": {"path": "gfm-python", "fingerprint": fingerprint},
                },
                "updatedAtUtc": "fixture",
            }
        ),
        encoding="utf-8",
    )

    migrated = RuntimeProfile.load(path)
    migrated_fingerprint = migrated.interpreters["gfm"]["fingerprint"]

    assert migrated.device_policy == "auto"
    assert migrated_fingerprint["schemaVersion"] == FINGERPRINT_SCHEMA_VERSION
    assert migrated_fingerprint["torch"] == {
        "available": True,
        "version": "2.12.0+cu130",
        "cudaRuntime": "13.0",
    }
    assert migrated.to_document()["schemaVersion"] == PROFILE_SCHEMA_VERSION


def test_runtime_profile_rejects_v1(tmp_path: Path) -> None:
    path = tmp_path / "runtime-profile.json"
    path.write_text('{"schemaVersion":"socialgraph-fm.runtime-profile/1.0"}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unsupported"):
        RuntimeProfile.load(path)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission assertion")
def test_private_atomic_profile_parent_is_not_world_writable(tmp_path: Path) -> None:
    path = tmp_path / "runtime-profile.json"
    profile = RuntimeProfile.create(
        profile="offline",
        env_mode="managed",
        install_profile_id=None,
        platform={"system": "linux", "machine": "x86_64"},
        interpreters={"bootstrap": None, "api": None, "gfm": None},
    )
    profile.write(path)
    assert path.is_file()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink fixture")
def test_runtime_state_rejects_a_var_symlink_escape(tmp_path: Path) -> None:
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    outside.mkdir()
    (project / "var").mkdir(parents=True)
    (project / "var" / "models").symlink_to(outside, target_is_directory=True)
    layout = RuntimeLayout(project)

    with pytest.raises(RuntimeError, match="cannot contain links"):
        layout.assert_safe_var_path(layout.model_root)
