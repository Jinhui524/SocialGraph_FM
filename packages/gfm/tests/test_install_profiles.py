import hashlib
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXPECTED_PROFILES = {
    "windows-x86_64-cpu-pt28",
    "linux-x86_64-cpu-pt28",
    "macos-arm64-cpu-pt28",
    "windows-x86_64-cu130-pt212",
    "linux-x86_64-cu130-pt212",
}
DEV_DISTRIBUTIONS = {"mypy", "pip-audit", "pytest", "ruff"}


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


def test_install_profile_matrix_and_hash_locks_are_exact():
    profiles_path = PROJECT / "install-profiles.json"
    profiles_document = json.loads(profiles_path.read_text(encoding="utf-8"))
    lock_manifest = json.loads(
        (PROJECT / "locks" / "install-lock-manifest.json").read_text(encoding="utf-8")
    )

    assert profiles_document["schemaVersion"] == "socialgraph-fm.gfm-install-profiles/1.0"
    assert lock_manifest["schemaVersion"] == (
        "socialgraph-fm.gfm-install-lock-manifest/1.0"
    )
    assert set(profiles_document["profiles"]) == EXPECTED_PROFILES
    assert set(lock_manifest["profiles"]) == EXPECTED_PROFILES
    assert lock_manifest["profilesSha256"] == _sha256(profiles_path)
    assert lock_manifest["buildRequirementsFile"] == "constraints/install-build.txt"
    assert lock_manifest["buildRequirementsSha256"] == _sha256(
        PROJECT / lock_manifest["buildRequirementsFile"]
    )
    assert lock_manifest["policy"]["sourceBuildsAllowed"] is False

    for profile_id, profile in profiles_document["profiles"].items():
        assert profile["wheelFamily"] == profile["device"]
        assert profile["pythonRequires"] == ">=3.12,<3.13"
        assert profile["indexUrl"] == "https://pypi.org/simple"
        assert profile["torchIndexUrl"].startswith("https://")
        assert profile["findLinks"] and all(
            item.startswith("https://data.pyg.org/whl/") for item in profile["findLinks"]
        )
        if profile_id.startswith("linux-"):
            assert profile["libc"] == "glibc"
        else:
            assert "libc" not in profile

        relative_lock = Path(profile["requirementsLock"])
        assert not relative_lock.is_absolute()
        lock_path = (PROJECT / relative_lock).resolve()
        assert lock_path.is_relative_to(PROJECT.resolve())
        assert lock_path.is_file()

        manifest_profile = lock_manifest["profiles"][profile_id]
        assert manifest_profile["requirementsLock"] == profile["requirementsLock"]
        assert manifest_profile["requirementsLockSha256"] == _sha256(lock_path)
        assert manifest_profile["artifactHashesResolved"] is True

        lock_text = lock_path.read_text(encoding="utf-8")
        assert "--generate-hashes" in lock_text.splitlines()[1]
        assert "--no-build" in lock_text.splitlines()[1]
        assert all("--hash=sha256:" in record for record in _logical_records(lock_path))

        locked = _requirements(lock_path)
        assert DEV_DISTRIBUTIONS.isdisjoint(locked)
        assert locked["hatchling"] == "1.27.0"
        for distribution, version in profile["distributionVersions"].items():
            assert locked[distribution] == version

    linux_cuda = profiles_document["profiles"]["linux-x86_64-cu130-pt212"]
    assert linux_cuda["distributionVersions"]["torch-scatter"] == (
        "2.1.2+pt212cu130"
    )
    assert linux_cuda["distributionVersions"]["torch-sparse"] == (
        "0.6.18+pt212cu130"
    )


def test_model_release_runtime_provenance_files_remain_frozen():
    assert _sha256(PROJECT / "runtime-profiles.json") == (
        "f8c32e45133afb777bd7aff9a2f887106147e2d79d1db36d7b60154a57d26be0"
    )
    assert _sha256(PROJECT / "locks" / "runtime-lock-manifest.json") == (
        "46eef932c2055f5730b598a995b1f7155749288ac1a923c5eae1258c202b2064"
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


def test_macos_arm64_lock_accepts_the_pypi_torch_wheel() -> None:
    lock = (
        PROJECT / "locks" / "install-macos-arm64-cpu-pt28.requirements.txt"
    ).read_text(encoding="utf-8")
    assert (
        "--hash=sha256:619c2869db3ada2c0105487ba21b5008defcc472d23f8b80ed91ac4a380283b0"
        in lock
    )
