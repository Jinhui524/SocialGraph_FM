import json

import pytest

import socialgraph_gfm.smoke as smoke_module
from socialgraph_gfm.checkpoint import load_checkpoint, read_manifest
from socialgraph_gfm.errors import CheckpointIntegrityError
from socialgraph_gfm.registry import LocalRegistry
from socialgraph_gfm.preflight import preflight_report
from socialgraph_gfm.runtime import runtime_report
from socialgraph_gfm.smoke import run_smoke

CPU_AVAILABLE = runtime_report("cpu")["runtimeReady"]
CUDA_AVAILABLE = runtime_report("cuda")["runtimeReady"]
ML_AVAILABLE = CPU_AVAILABLE or CUDA_AVAILABLE
TEST_DEVICE = "cpu" if CPU_AVAILABLE else "cuda"
pytestmark = pytest.mark.skipif(not ML_AVAILABLE, reason="exact Torch/PyG/OGB runtime is not installed")


def test_smoke_runs_forward_backward_update_checkpoint_and_fresh_process(tmp_path):
    result = run_smoke(fixture="both", device=TEST_DEVICE, root=tmp_path, seed=7)
    assert result["ok"]
    assert result["elapsedSeconds"] >= 0
    assert result["maxMemoryMb"] >= 0
    assert len(result["manifestHash"]) == 64
    assert len(result["runs"]) == 2
    for run in result["runs"]:
        assert run["status"] == "succeeded"
        assert run["freshProcessVerified"] is True
        assert run["optimizerRestored"] is True
        assert run["loss"] >= 0
        assert len(run["runManifestHash"]) == 64
        assert len(run["checkpointManifestHash"]) == 64
        assert len(run["logicalRunManifestHash"]) == 64
        assert len(run["logicalCheckpointManifestHash"]) == 64
        assert len(run["reproducibilityHash"]) == 64
    registry = LocalRegistry(tmp_path / "registry" / "registry.sqlite3")
    assert registry.counts()["models"] == 0


def test_checkpoint_tampering_is_detected(tmp_path):
    result = run_smoke(fixture="actor", device=TEST_DEVICE, root=tmp_path, seed=11)
    checkpoint_payload = result["runs"][0]["checkpoint"]
    manifest_path = (
        tmp_path / "runs" / result["runs"][0]["runId"] / "checkpoints"
        / f"{checkpoint_payload['checkpointId']}.manifest.json"
    )
    manifest = read_manifest(manifest_path)
    artifact = __import__("pathlib").Path(manifest.artifact_path)
    with artifact.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        load_checkpoint(manifest)


def test_actor_and_hetero_double_run_have_stable_logical_hashes(tmp_path):
    first = run_smoke(fixture="both", device=TEST_DEVICE, root=tmp_path, seed=20260812)
    second = run_smoke(fixture="both", device=TEST_DEVICE, root=tmp_path, seed=20260812)
    assert [run["fixture"] for run in first["runs"]] == ["actor", "hetero"]
    assert [run["fixture"] for run in second["runs"]] == ["actor", "hetero"]

    deterministic_fields = (
        "materializationHash",
        "outputDigest",
        "logicalRunManifestHash",
        "logicalCheckpointManifestHash",
        "reproducibilityHash",
    )
    for earlier, later in zip(first["runs"], second["runs"], strict=True):
        assert earlier["runId"] != later["runId"]
        assert earlier["runManifestHash"] != later["runManifestHash"]
        assert earlier["checkpointManifestHash"] != later["checkpointManifestHash"]
        assert earlier["checkpoint"]["artifactPath"] != later["checkpoint"]["artifactPath"]
        assert earlier["checkpoint"]["stateHash"] == later["checkpoint"]["stateHash"]
        for field in deterministic_fields:
            assert earlier[field] == later[field], field

    registry = LocalRegistry(tmp_path / "registry" / "registry.sqlite3")
    assert registry.counts() == {"runs": 4, "checkpoints": 4, "models": 0}
    assert preflight_report(TEST_DEVICE, tmp_path)["readiness"]["GfmInfrastructureReady"] is True

    with registry.connect() as connection:
        rows = connection.execute("SELECT run_id, manifest_json FROM runs").fetchall()
        for run_id, raw in rows:
            manifest = json.loads(raw)
            manifest["smokeMetrics"]["elapsedSeconds"] = 120.001
            connection.execute(
                "UPDATE runs SET manifest_json=? WHERE run_id=?",
                (json.dumps(manifest), run_id),
            )
    exceeded = preflight_report(TEST_DEVICE, tmp_path)
    assert exceeded["readiness"]["GfmInfrastructureReady"] is False
    assert exceeded["smokeCoverage"] == []


def test_failed_smoke_never_creates_a_registrable_model(tmp_path, monkeypatch):
    def fail_model(*_args, **_kwargs):
        raise RuntimeError("injected smoke failure")

    monkeypatch.setattr(smoke_module, "_model", fail_model)
    with pytest.raises(RuntimeError, match="injected smoke failure"):
        run_smoke(fixture="actor", device=TEST_DEVICE, root=tmp_path, seed=17)
    registry = LocalRegistry(tmp_path / "registry" / "registry.sqlite3")
    assert registry.counts() == {"runs": 1, "checkpoints": 0, "models": 0}
