import json
import tomllib
from pathlib import Path

from socialgraph_gfm.canonical import file_sha256
from socialgraph_gfm.locks import verify_lock_manifest


def test_constraint_files_match_verifiable_lock_manifest():
    report = verify_lock_manifest()
    assert report["constraintIntegrityValid"] is True
    assert report["requirementsLockIntegrityValid"] is True
    assert report["artifactHashesVerified"] is True
    assert report["hashCoverageVerified"] is True
    assert report["verificationScope"] == "checked_in_lock_integrity_and_hash_coverage"
    assert set(report["profiles"]) == {
        "cpu-ci",
        "windows-cpu",
    }
    assert report["releaseLocksReady"] is True
    assert all(profile["requirementCount"] > 0 for profile in report["profiles"].values())
    assert all(profile["unhashedRequirements"] == [] for profile in report["profiles"].values())
    assert all(profile["invalidHashTokens"] == [] for profile in report["profiles"].values())
    assert report["profiles"]["windows-cpu"]["optionalExtensionsUnavailable"] == [
        "torch-scatter", "torch-sparse"
    ]


def test_runtime_lock_manifest_has_no_public_cuda_or_text_profile():
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project / "locks" / "runtime-lock-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["policy"]["sourceBuildsAllowed"] is False

    assert manifest["schemaVersion"] == "gfm.runtime-lock-manifest/2.0"
    assert manifest["policy"]["supportedDevices"] == ["cpu"]
    assert set(manifest["profiles"]) == {"windows-cpu", "cpu-ci"}
    assert "cuda" not in json.dumps(manifest).casefold()


def test_wheel_resources_include_gfm_lock_once_and_constraints_as_one_tree():
    project = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include["constraints"] == "socialgraph_gfm/resources/constraints"
    assert "locks/windows-cu130-gfm.requirements.txt" not in force_include
    assert force_include["locks/cpu-ci.requirements.txt"] == (
        "socialgraph_gfm/resources/locks/cpu-ci.requirements.txt"
    )
    assert force_include["locks/windows-cpu.requirements.txt"] == (
        "socialgraph_gfm/resources/locks/windows-cpu.requirements.txt"
    )
    assert force_include["install-profiles.json"] == (
        "socialgraph_gfm/resources/install-profiles.json"
    )
    assert force_include["locks/install-lock-manifest.json"] == (
        "socialgraph_gfm/resources/install-lock-manifest.json"
    )
    assert force_include["locks/install-windows-x86_64-cpu-pt28.requirements.txt"] == (
        "socialgraph_gfm/resources/locks/install-windows-x86_64-cpu-pt28.requirements.txt"
    )
    assert force_include["locks/install-linux-x86_64-cpu-pt28.requirements.txt"] == (
        "socialgraph_gfm/resources/locks/install-linux-x86_64-cpu-pt28.requirements.txt"
    )
    assert force_include["configs/ogbl-collab-baseline.json"] == (
        "socialgraph_gfm/resources/configs/ogbl-collab-baseline.json"
    )
    assert force_include["configs/socialgraph-core.json"] == (
        "socialgraph_gfm/resources/configs/socialgraph-core.json"
    )
    assert force_include["configs/openalex-graph-ai.json"] == (
        "socialgraph_gfm/resources/configs/openalex-graph-ai.json"
    )
    assert force_include["configs/socialgraph-research.json"] == (
        "socialgraph_gfm/resources/configs/socialgraph-research.json"
    )
    assert force_include["configs/socialgraph-global.json"] == (
        "socialgraph_gfm/resources/configs/socialgraph-global.json"
    )
    assert force_include["contracts/core-serving-registry.json"] == (
        "socialgraph_gfm/resources/core-serving-registry.json"
    )
    assert force_include["contracts/core-serving-graph-catalog.json"] == (
        "socialgraph_gfm/resources/core-serving-graph-catalog.json"
    )
    assert "constraints/windows-cu130-gfm.txt" not in force_include
    assert len(force_include.values()) == len(set(force_include.values()))

    assert pyproject["project"]["scripts"] == {
        "socialgraph-gfm": "socialgraph_gfm.cli:main",
        "socialgraph-gfm-core-experiment": (
            "socialgraph_gfm.core.experiment_cli:main"
        ),
        "socialgraph-gfm-core-serve": (
            "socialgraph_gfm.core.inference_cli:main"
        ),
        "socialgraph-gfm-research": "socialgraph_gfm.research.cli:main",
        "socialgraph-gfm-global": "socialgraph_gfm.global_model.cli:main",
        "socialgraph-gfm-governance": "socialgraph_gfm.governance.cli:main",
    }

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    assert pyproject["project"]["dependencies"] == [
        "numpy==2.3.3",
        "pydantic==2.13.4",
    ]
    assert optional_dependencies["cpu"] == [
        "pyg-lib==0.6.0",
        "torch==2.8.0",
        "torch-geometric==2.8.0.post1",
    ]
    assert optional_dependencies["research"] == [
        "ogb==1.3.6",
        "pandas>=2.2,<4",
        "scikit-learn>=1.5,<2",
        "scipy>=1.14,<2",
        "FlagEmbedding==1.4.0",
        "transformers==5.14.1",
    ]
    assert optional_dependencies["test"] == optional_dependencies["dev"]


def test_boolean_cannot_hide_an_unhashed_requirement(tmp_path):
    (tmp_path / "constraints").mkdir()
    (tmp_path / "locks").mkdir()
    constraints = tmp_path / "constraints" / "profile.txt"
    requirements = tmp_path / "locks" / "profile.requirements.txt"
    constraints.write_text("example==1.0\n", encoding="utf-8")
    requirements.write_text("example==1.0\n", encoding="utf-8")
    manifest = {
        "profiles": {
            "test": {
                "constraints": "constraints/profile.txt",
                "constraintsSha256": file_sha256(constraints),
                "requirementsLock": "locks/profile.requirements.txt",
                "requirementsLockSha256": file_sha256(requirements),
                "artifactHashesResolved": True,
                "optionalExtensionsUnavailable": [],
            }
        }
    }
    (tmp_path / "locks" / "runtime-lock-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    report = verify_lock_manifest(tmp_path)
    profile = report["profiles"]["test"]
    assert profile["requirementsLockIntegrityValid"] is True
    assert profile["hashCoverageValid"] is False
    assert profile["artifactHashesVerified"] is False
    assert report["releaseLocksReady"] is False


def test_missing_or_tampered_requirement_lock_fails_closed(tmp_path):
    project = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (project / "locks" / "runtime-lock-manifest.json").read_text(encoding="utf-8")
    )
    (tmp_path / "constraints").mkdir()
    (tmp_path / "locks").mkdir()
    profile = manifest["profiles"]["cpu-ci"]
    source_constraints = project / profile["constraints"]
    target_constraints = tmp_path / profile["constraints"]
    target_constraints.write_bytes(source_constraints.read_bytes())
    manifest["profiles"] = {"cpu-ci": profile}
    (tmp_path / "locks" / "runtime-lock-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    missing = verify_lock_manifest(tmp_path)
    assert missing["profiles"]["cpu-ci"]["requirementsLockPresent"] is False
    assert missing["releaseLocksReady"] is False

    target_lock = tmp_path / profile["requirementsLock"]
    target_lock.write_text(
        "example==1.0 --hash=sha256:" + "0" * 64 + "\n", encoding="utf-8"
    )
    tampered = verify_lock_manifest(tmp_path)
    assert tampered["profiles"]["cpu-ci"]["hashCoverageValid"] is True
    assert tampered["profiles"]["cpu-ci"]["requirementsLockIntegrityValid"] is False
    assert tampered["releaseLocksReady"] is False
