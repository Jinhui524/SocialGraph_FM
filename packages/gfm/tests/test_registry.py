from datetime import UTC, datetime

import pytest

from socialgraph_gfm.contracts import (
    RunStatus,
    SmokeCheckpointManifest,
    SmokeCorpusManifest,
    SmokeRunMetrics,
    SmokeTrainingRunManifest,
)
from socialgraph_gfm.errors import RegistrationRejected
from socialgraph_gfm.fixtures import actor_interaction_fixture
from socialgraph_gfm.registry import LocalRegistry


def make_run(status=RunStatus.SUCCEEDED):
    snapshot = actor_interaction_fixture()
    corpus = SmokeCorpusManifest(
        corpusId="synthetic-actor",
        version="1.0",
        adapter="fixture",
        sourceHash=snapshot.ref.content_hash,
        snapshotRefs=(snapshot.ref,),
    )
    now = datetime.now(UTC)
    values = dict(
        runId="run-1",
        status=status,
        seed=1,
        codeHash="1" * 64,
        environmentHash="2" * 64,
        corpus=corpus,
        configHash="3" * 64,
        startedAt=now,
        finishedAt=now if status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED) else None,
        failureCode="TEST_FAILURE" if status == RunStatus.FAILED else None,
    )
    if status == RunStatus.SUCCEEDED:
        values["smokeMetrics"] = SmokeRunMetrics(
            device="cpu",
            elapsedSeconds=1,
            maxMemoryMb=100,
            checkpointStateHash="4" * 64,
            checkpointArtifactSha256="5" * 64,
        )
    return SmokeTrainingRunManifest(**values)


def make_checkpoint(smoke_only=True):
    return SmokeCheckpointManifest(
        checkpointId="checkpoint-1",
        runId="run-1",
        step=1,
        smokeOnly=smoke_only,
        stateHash="4" * 64,
        configHash="3" * 64,
        artifactSha256="5" * 64,
        artifactPath="checkpoint.pt",
        createdAt=datetime.now(UTC),
    )


def test_registry_records_runs_and_checkpoints_and_reports_coverage(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    run = make_run()
    checkpoint = make_checkpoint()
    registry.record_run(run)
    registry.record_checkpoint(checkpoint)
    assert registry.counts() == {"runs": 1, "checkpoints": 1, "models": 0}
    assert registry.successful_smoke_coverage(
        code_hash="1" * 64,
        environment_hash="2" * 64,
        config_hashes={"actor": "3" * 64},
        device="cpu",
    ) == set()  # The synthetic checkpoint path does not exist, so coverage fails closed.


def test_stale_code_environment_config_or_missing_checkpoint_never_counts(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    run = make_run()
    registry.record_run(run)
    assert registry.successful_smoke_coverage(
        code_hash=run.code_hash,
        environment_hash=run.environment_hash,
        config_hashes={"actor": run.config_hash},
        device="cpu",
    ) == set()


@pytest.mark.parametrize("status", [RunStatus.SUCCEEDED, RunStatus.FAILED])
def test_smoke_or_failed_run_can_never_be_registered(tmp_path, status):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    run = make_run(status)
    checkpoint = make_checkpoint()
    registry.record_run(run)
    registry.record_checkpoint(checkpoint)
    with pytest.raises(RegistrationRejected):
        registry.register_model("model-1", run, checkpoint, validated=True)
    assert registry.counts()["models"] == 0


def test_unvalidated_registration_is_rejected(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    run = make_run()
    checkpoint = make_checkpoint(smoke_only=False)
    registry.record_run(run)
    registry.record_checkpoint(checkpoint)
    with pytest.raises(RegistrationRejected, match="validated=true"):
        registry.register_model("model-1", run, checkpoint, validated=False)
