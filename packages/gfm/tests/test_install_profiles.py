from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_PROFILES = {
    "windows-x86_64-cpu-pt28",
    "linux-x86_64-cpu-pt28",
}
EXCLUDED_PUBLIC_DISTRIBUTIONS = {
    "flagembedding",
    "hatchling",
    "mypy",
    "ogb",
    "pandas",
    "pip-audit",
    "pytest",
    "ruff",
    "scikit-learn",
    "scipy",
    "transformers",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ;\\]+)", line)
        if match:
            requirements[match.group(1).casefold().replace("_", "-")] = match.group(2)
    return requirements


def _logical_records(path: Path) -> tuple[str, ...]:
    records: list[str] = []
    pending = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        pending += stripped
        records.append(pending.strip())
        pending = ""
    if pending:
        records.append(pending.strip())
    return tuple(records)


def test_public_runtime_matrix_has_exactly_two_cpu_hash_locks() -> None:
    profiles_path = PROJECT / "install-profiles.json"
    profiles_document = json.loads(profiles_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (PROJECT / "locks" / "install-lock-manifest.json").read_text(encoding="utf-8")
    )

    assert profiles_document["schemaVersion"] == "socialgraph-fm.gfm-install-profiles/2.0"
    assert manifest["schemaVersion"] == "socialgraph-fm.gfm-install-lock-manifest/2.0"
    assert set(profiles_document["profiles"]) == EXPECTED_PROFILES
    assert set(manifest["profiles"]) == EXPECTED_PROFILES
    assert manifest["profilesSha256"] == _sha256(profiles_path)
    requirements_source = PROJECT / manifest["runtimeRequirementsFile"]
    assert manifest["runtimeRequirementsSha256"] == _sha256(requirements_source)
    assert manifest["policy"]["supportedDevices"] == ["cpu"]
    assert manifest["policy"]["runtimeEnvironmentCount"] == 1
    assert manifest["policy"]["sourceBuildsAllowed"] is False

    for profile_id, profile in profiles_document["profiles"].items():
        assert profile["pythonRequires"] == ">=3.12,<3.13"
        assert profile["indexUrl"] == "https://pypi.org/simple"
        assert profile["torchIndexUrl"] == "https://download.pytorch.org/whl/cpu"
        assert profile["findLinks"] == [
            "https://data.pyg.org/whl/torch-2.8.0+cpu.html"
        ]
        assert "cuda" not in json.dumps(profile).casefold()
        if profile_id.startswith("linux-"):
            assert profile["libc"] == "glibc"
        else:
            assert "libc" not in profile

        lock = (PROJECT / profile["requirementsLock"]).resolve()
        assert lock.is_relative_to(PROJECT.resolve())
        assert lock.is_file()
        declaration = manifest["profiles"][profile_id]
        assert declaration["requirementsLock"] == profile["requirementsLock"]
        assert declaration["requirementsLockSha256"] == _sha256(lock)
        assert declaration["artifactHashesResolved"] is True
        assert all("--hash=sha256:" in record for record in _logical_records(lock))

        locked = _requirements(lock)
        assert EXCLUDED_PUBLIC_DISTRIBUTIONS.isdisjoint(locked)
        for distribution, version in profile["distributionVersions"].items():
            assert locked[distribution] == version


def test_retired_platform_and_device_locks_are_absent() -> None:
    retired = (
        "install-macos-arm64-cpu-pt28.requirements.txt",
        "install-windows-x86_64-cu130-pt212.requirements.txt",
        "install-linux-x86_64-cu130-pt212.requirements.txt",
        "windows-cu130.requirements.txt",
        "windows-cu130-gfm.requirements.txt",
        "linux-cu130.requirements.txt",
    )
    assert all(not (PROJECT / "locks" / name).exists() for name in retired)


def test_model_release_runtime_provenance_files_remain_frozen() -> None:
    assert _sha256(PROJECT / "runtime-profiles.json") == (
        "51b6e4efb86d1a548facc5bc814362ac356f4957a2cf92b36348c888f456cbf3"
    )
    assert _sha256(PROJECT / "locks" / "runtime-lock-manifest.json") == (
        "07d0782f4e76b748ae4d9f10c4ee5c51f0db0f555b87ece9f28fcefcf3f2a556"
    )
    environment_lock = (
        PROJECT.parents[1]
        / "bundles"
        / "models"
        / "socialgraph-global"
        / "exports"
        / "socialgraph-global"
        / "environment-lock.json"
    )
    assert _sha256(environment_lock) == (
        "a9f738773e7e85c7c462e3ee06e02d120b915c9bc35cd5add2555d92c666ff6c"
    )
