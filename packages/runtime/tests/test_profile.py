from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from socialgraph_fm_runtime.layout import RuntimeLayout
from socialgraph_fm_runtime.profile import PROFILE_SCHEMA_VERSION, RuntimeProfile


def _interpreter() -> dict[str, object]:
    return {
        "path": "var/runtime/python",
        "source": "managed",
        "installLockSha256": "a" * 64,
        "fingerprint": {"fingerprintSha256": "b" * 64},
    }


def test_single_cpu_runtime_profile_is_atomic_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config" / "runtime-profile.json"
    profile = RuntimeProfile.create(
        install_profile_id="windows-x86_64-cpu-pt28",
        platform={"system": "windows", "machine": "x86_64"},
        interpreter=_interpreter(),
    )
    profile.write(path)

    loaded = RuntimeProfile.load(path)

    assert loaded.to_document() == profile.to_document()
    assert set(loaded.to_document()) == {
        "schemaVersion",
        "installProfileId",
        "platform",
        "interpreter",
        "updatedAtUtc",
    }
    assert json.loads(path.read_text(encoding="utf-8"))["schemaVersion"] == PROFILE_SCHEMA_VERSION
    assert not list(path.parent.glob("*.tmp"))


def test_retired_split_or_cuda_profile_requires_onboarding(tmp_path: Path) -> None:
    path = tmp_path / "runtime-profile.json"
    path.write_text(
        '{"schemaVersion":"socialgraph-fm.runtime-profile/3.0","profile":"cuda"}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="retired setup profile"):
        RuntimeProfile.load(path)


def test_profile_cannot_reference_a_non_cpu_install_lock() -> None:
    with pytest.raises(ValueError, match="CPU"):
        RuntimeProfile.create(
            install_profile_id="windows-x86_64-cu130-pt212",
            platform={"system": "windows", "machine": "x86_64"},
            interpreter=_interpreter(),
        )


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
