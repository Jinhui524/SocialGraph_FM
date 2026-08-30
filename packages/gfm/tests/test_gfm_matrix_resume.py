from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from socialgraph_gfm.cli import build_parser
from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.errors import ContractViolation, GfmTrainingError, RegistrationRejected
from socialgraph_gfm.gfm.contracts import GfmCheckpointManifest, GfmRunManifest
from socialgraph_gfm.gfm.corpus.common import atomic_write_json
from socialgraph_gfm.gfm.lodo_execution import LodoCellIdentity, create_lodo_run_state
from socialgraph_gfm.gfm.registry import GfmRegistry
from socialgraph_gfm.gfm_workflow import (
    DOMAIN_IDS,
    _CompletedRunExpectation,
    _ensure_checkpoint_free_retry_directory,
    _experiment_id,
    _load_pretrain_config,
    _lodo_source_should_stop,
    _pretrain_heartbeat,
    _pretrain_worker,
    _probe_with_durable_preflight,
    _train_lodo_optimizer_block,
    _train_epoch_with_heartbeats,
    _validate_completed_matrix_run,
    _validate_lodo_formal_prerequisites,
    _warmup_cosine,
    pretrain_gfm,
    resume_gfm,
)
from socialgraph_gfm.runtime import RuntimeLayout

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


class _RegistryStub:
    def __init__(self, run: GfmRunManifest | None) -> None:
        self.run = run

    def get_run(self, _run_id: str) -> GfmRunManifest | None:
        return self.run


class _CompletionRegistryStub(_RegistryStub):
    def __init__(
        self,
        run: GfmRunManifest,
        checkpoint: GfmCheckpointManifest | None,
    ) -> None:
        super().__init__(run)
        self.run: GfmRunManifest = run
        self.checkpoint = checkpoint

    def list_checkpoints(self, *, experiment_id: str) -> tuple[GfmCheckpointManifest, ...]:
        assert experiment_id == self.run.experiment_id
        return () if self.checkpoint is None else (self.checkpoint,)

    def list_evaluations(self, *, experiment_id: str) -> tuple[()]:
        assert experiment_id == self.run.experiment_id
        return ()

    def record_evaluation(self, _report: object) -> None:
        raise AssertionError("dev reconciliation must not manufacture reports")


def _terminal_lag_case(
    tmp_path: Path,
    *,
    phase: str = "dev",
    experiment_id: str = "experiment",
    run_id: str = "run",
) -> tuple[
    RuntimeLayout,
    _CompletedRunExpectation,
    GfmRunManifest,
    GfmCheckpointManifest,
    dict[str, Any],
    Path,
]:
    domains = ("academic", "software", "community")
    config = {"configId": "reconcile-test"}
    config_hash = canonical_sha256(config)
    expectation = _CompletedRunExpectation(
        experiment_id=experiment_id,
        run_id=run_id,
        phase="pretrain",
        variant="core-base",
        seed=20260820,
        domain_ids=domains,
        corpus_hashes=(HASH_A, HASH_B, HASH_C),
        protocol_hashes=(HASH_D,),
        config_hash=config_hash,
        code_hash=HASH_D,
        environment_hash=HASH_E,
        pretrain_phase=cast(Any, phase),
        required_reports=(
            (
                (f"{run_id}-frozen-test", "in_domain"),
                (f"{run_id}-fresh-process", "fresh_process"),
            )
            if phase == "formal"
            else ()
        ),
    )
    now = datetime.now(UTC)
    run = GfmRunManifest.create(
        runId=expectation.run_id,
        experimentId=expectation.experiment_id,
        phase="pretrain",
        architectureVariant=expectation.variant,
        status="succeeded",
        domainIds=expectation.domain_ids,
        seed=expectation.seed,
        codeHash=expectation.code_hash,
        environmentHash=expectation.environment_hash,
        configHash=expectation.config_hash,
        corpusHashes=expectation.corpus_hashes,
        taskProtocolHashes=expectation.protocol_hashes,
        startedAt=now,
        finishedAt=now,
        peakCudaMemoryMiB=100.0,
    )
    layout = RuntimeLayout(tmp_path)
    run_dir = layout.gfm_runs / expectation.experiment_id / expectation.run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_id = f"{expectation.run_id}-best-100"
    checkpoint = GfmCheckpointManifest.create(
        checkpointId=checkpoint_id,
        runId=expectation.run_id,
        epoch=1,
        step=100,
        componentNames=("core",),
        stateHash=HASH_A,
        configHash=config_hash,
        corpusHashes=expectation.corpus_hashes,
        artifactSha256=HASH_B,
        artifactPath=str(checkpoint_dir / f"{checkpoint_id}.pt"),
        registrable=False,
    )
    atomic_write_json(
        run_dir / "run-manifest.json",
        run.model_dump(mode="json", by_alias=True),
    )
    atomic_write_json(
        checkpoint_dir / f"{checkpoint_id}.manifest.json",
        checkpoint.model_dump(mode="json", by_alias=True),
    )
    embedding_artifacts = {"academic": {"logicalHash": HASH_A}}
    access_audits = {domain: {"testArtifactsOpened": False} for domain in domains}
    state_path = run_dir / "run-state.json"
    atomic_write_json(
        state_path,
        {
            "schemaVersion": "gfm.workflow-run-state/1.0",
            "runKind": "pretrain",
            "runId": expectation.run_id,
            "experimentId": expectation.experiment_id,
            "phase": phase,
            "variant": expectation.variant,
            "seed": expectation.seed,
            "device": "cpu",
            "configHash": expectation.config_hash,
            "codeHash": expectation.code_hash,
            "environmentHash": expectation.environment_hash,
            "corpusHashes": list(expectation.corpus_hashes),
            "embeddingArtifacts": embedding_artifacts,
            "embeddingArtifactsHash": canonical_sha256(embedding_artifacts),
            "domainAccessAudits": access_audits,
            "domainAccessAuditsHash": canonical_sha256(access_audits),
            "startedAt": now.isoformat(),
            "batchSize": 64,
            "gradientAccumulation": 64,
            "probePeakCudaMemoryMiB": 0.0,
            "preflightAttemptCount": 1,
            "status": "running",
        },
    )
    payload = {
        "config": config,
        "sampler_state": {
            "optimizerStep": 100,
            "batchSize": 64,
            "gradientAccumulation": 64,
            "embeddingArtifacts": embedding_artifacts,
            "embeddingArtifactsHash": canonical_sha256(embedding_artifacts),
            "domainAccessAudits": access_audits,
            "domainAccessAuditsHash": canonical_sha256(access_audits),
            "streams": {
                domain: {
                    "negativeSamplingAudit": {
                        "futureAccessCount": 0,
                        "sampleCount": 100,
                    }
                }
                for domain in domains
            },
        },
        "best_state": {
            "step": 100,
            "validationLoss": 0.5,
            "embeddingArtifactsHash": canonical_sha256(embedding_artifacts),
        },
    }
    return layout, expectation, run, checkpoint, payload, state_path


def _expectation() -> _CompletedRunExpectation:
    return _CompletedRunExpectation(
        experiment_id="experiment",
        run_id="experiment-core-base-20260821",
        phase="pretrain",
        variant="core-base",
        seed=20260821,
        domain_ids=("academic", "software", "community"),
        corpus_hashes=(HASH_A, HASH_B, HASH_C),
        protocol_hashes=(HASH_D,),
        config_hash=HASH_E,
        code_hash=HASH_D,
        environment_hash=HASH_E,
    )


@pytest.mark.parametrize("status", ["preflight", "running"])
def test_matrix_rerun_points_interrupted_cell_to_explicit_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    expectation = _expectation()
    layout = RuntimeLayout(tmp_path)
    run_dir = layout.gfm_runs / expectation.experiment_id / expectation.run_id
    atomic_write_json(
        run_dir / "run-state.json",
        {"runId": expectation.run_id, "status": status},
    )
    monkeypatch.setattr(workflow, "_registry", lambda _layout: _RegistryStub(None))

    with pytest.raises(GfmTrainingError, match=r"gfm-resume --run-id"):
        _validate_completed_matrix_run(layout, expectation)


def test_matrix_rerun_points_durable_lodo_cell_to_explicit_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    config = {"configId": "lodo-matrix"}
    expectation = _CompletedRunExpectation(
        experiment_id="experiment",
        run_id="experiment-lodo-core-base-community-20260821",
        phase="lodo",
        variant="core-base",
        seed=20260821,
        domain_ids=("academic", "software"),
        held_out_domain="community",
        corpus_hashes=(HASH_A, HASH_B, HASH_C),
        protocol_hashes=(HASH_D,),
        config_hash=canonical_sha256(config),
        code_hash=HASH_D,
        environment_hash=HASH_E,
    )
    identity = LodoCellIdentity(
        experiment_id=expectation.experiment_id,
        run_id=expectation.run_id,
        held_out_domain="community",
        source_domain_ids=expectation.domain_ids,
        architecture_variant="core-base",
        seed=expectation.seed,
        config_hash=expectation.config_hash,
        code_hash=expectation.code_hash,
        environment_hash=expectation.environment_hash,
        corpus_hashes=expectation.corpus_hashes,
        protocol_hashes=expectation.protocol_hashes,
        role_view_contract={
            "maximumRole": "validation",
            "physicalBoundary": True,
            "testReadCount": 0,
        },
    )
    layout = RuntimeLayout(tmp_path)
    create_lodo_run_state(
        layout.gfm_runs / expectation.experiment_id / expectation.run_id / "run-state.json",
        identity=identity,
        device="cuda",
    )
    monkeypatch.setattr(workflow, "_registry", lambda _layout: _RegistryStub(None))

    with pytest.raises(GfmTrainingError, match=r"gfm-resume --run-id"):
        _validate_completed_matrix_run(layout, expectation)


def test_lodo_prerequisites_reject_dev_identity_and_partial_formal_matrix(
    tmp_path: Path,
) -> None:
    config = _load_pretrain_config(None, None)
    corpora = tuple(
        SimpleNamespace(domain_id=domain, logical_hash=hash_value)
        for domain, hash_value in zip(
            DOMAIN_IDS.values(), (HASH_A, HASH_B, HASH_C), strict=True
        )
    )
    protocols = (SimpleNamespace(protocol_hash=HASH_D),)
    all_runs = tuple(
        SimpleNamespace(architecture_variant=variant, seed=seed)
        for variant in config.architecture.candidates
        for seed in config.formal.seeds
    )
    formal_experiment = _experiment_id(
        phase="formal", config=config, corpora=cast(Any, corpora)
    )
    with pytest.raises(ContractViolation, match="six-cell"):
        _validate_lodo_formal_prerequisites(
            layout=RuntimeLayout(tmp_path),
            experiment_id=formal_experiment.replace("-formal-", "-dev-"),
            config=config,
            corpora=cast(Any, corpora),
            protocols=cast(Any, protocols),
            existing_pretrain_runs=cast(Any, all_runs),
            code_hash=HASH_D,
            environment_hash=HASH_E,
        )
    with pytest.raises(ContractViolation, match="six-cell"):
        _validate_lodo_formal_prerequisites(
            layout=RuntimeLayout(tmp_path),
            experiment_id=formal_experiment,
            config=config,
            corpora=cast(Any, corpora),
            protocols=cast(Any, protocols),
            existing_pretrain_runs=cast(Any, all_runs[:-1]),
            code_hash=HASH_D,
            environment_hash=HASH_E,
        )


def test_lodo_source_resume_honours_already_committed_patience_boundary() -> None:
    assert _lodo_source_should_stop(
        optimizer_step=12_000,
        minimum_steps=10_000,
        no_improvement=6,
        patience=6,
    )
    assert not _lodo_source_should_stop(
        optimizer_step=9_999,
        minimum_steps=10_000,
        no_improvement=6,
        patience=6,
    )


def test_matrix_reconciles_registry_committed_pretrain_terminal_lag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout, expectation, run, checkpoint, payload, state_path = _terminal_lag_case(tmp_path)
    registry = _CompletionRegistryStub(run, checkpoint)
    monkeypatch.setattr(workflow, "_registry", lambda _layout: registry)
    monkeypatch.setattr(workflow, "load_gfm_checkpoint", lambda _checkpoint, map_location: payload)

    completed = _validate_completed_matrix_run(layout, expectation)

    assert completed is not None
    assert completed.run_state is not None
    assert completed.run_state["status"] == "succeeded"
    assert completed.run_state["bestCheckpointManifest"].endswith(
        f"{checkpoint.checkpoint_id}.manifest.json"
    )
    evidence = completed.run_state["terminalReconciliationEvidence"]
    assert evidence["runManifestHash"] == run.manifest_hash
    assert evidence["checkpointLogicalHash"] == checkpoint.logical_hash
    assert evidence["requiredReportHashes"] == {}
    first_state = workflow.read_json_object(state_path)

    repeated = _validate_completed_matrix_run(layout, expectation)

    assert repeated is not None
    assert repeated.run_state == first_state
    assert workflow.read_json_object(state_path) == first_state


def test_completed_pretrain_rejects_a_new_current_embedding_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout, expectation, run, checkpoint, payload, state_path = _terminal_lag_case(tmp_path)
    registry = _CompletionRegistryStub(run, checkpoint)
    monkeypatch.setattr(workflow, "_registry", lambda _layout: registry)
    monkeypatch.setattr(workflow, "load_gfm_checkpoint", lambda _checkpoint, map_location: payload)
    current_embeddings = {"academic": {"logicalHash": HASH_B}}
    expectation = replace(
        expectation,
        current_embedding_artifacts=current_embeddings,
        current_embedding_artifacts_hash=canonical_sha256(current_embeddings),
    )

    with pytest.raises(GfmTrainingError, match="embeddingArtifacts"):
        _validate_completed_matrix_run(layout, expectation)
    assert workflow.read_json_object(state_path)["status"] == "running"


def test_completed_pretrain_rejects_resealed_state_embedding_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout, expectation, run, checkpoint, payload, state_path = _terminal_lag_case(tmp_path)
    registry = _CompletionRegistryStub(run, checkpoint)
    monkeypatch.setattr(workflow, "_registry", lambda _layout: registry)
    monkeypatch.setattr(workflow, "load_gfm_checkpoint", lambda _checkpoint, map_location: payload)
    assert _validate_completed_matrix_run(layout, expectation) is not None
    state = workflow.read_json_object(state_path)
    assert state["status"] == "succeeded"
    replacement = {"academic": {"logicalHash": HASH_C}}
    state["embeddingArtifacts"] = replacement
    state["embeddingArtifactsHash"] = canonical_sha256(replacement)
    atomic_write_json(state_path, state)

    with pytest.raises(GfmTrainingError, match="checkpoint embedding/access provenance"):
        _validate_completed_matrix_run(layout, expectation)
    assert workflow.read_json_object(state_path)["status"] == "succeeded"


def test_pretrain_terminal_reconciliation_fails_closed_without_checkpoint_or_reports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout, expectation, run, checkpoint, payload, state_path = _terminal_lag_case(
        tmp_path / "missing-checkpoint"
    )
    monkeypatch.setattr(
        workflow,
        "_registry",
        lambda _layout: _CompletionRegistryStub(run, None),
    )
    monkeypatch.setattr(workflow, "load_gfm_checkpoint", lambda _checkpoint, map_location: payload)
    with pytest.raises(GfmTrainingError, match="exactly one registered best checkpoint"):
        _validate_completed_matrix_run(layout, expectation)
    assert workflow.read_json_object(state_path)["status"] == "running"

    formal = _terminal_lag_case(tmp_path / "missing-reports", phase="formal")
    (
        formal_layout,
        formal_expectation,
        formal_run,
        formal_checkpoint,
        formal_payload,
        formal_state_path,
    ) = formal
    monkeypatch.setattr(
        workflow,
        "_registry",
        lambda _layout: _CompletionRegistryStub(formal_run, formal_checkpoint),
    )
    monkeypatch.setattr(
        workflow,
        "load_gfm_checkpoint",
        lambda _checkpoint, map_location: formal_payload,
    )
    with pytest.raises(GfmTrainingError, match="lacks required immutable evidence"):
        _validate_completed_matrix_run(formal_layout, formal_expectation)
    assert workflow.read_json_object(formal_state_path)["status"] == "running"


def test_resume_reconciles_a_registry_terminal_pretrain_without_training(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    experiment_id = "socialgraph-core-dev-0123456789abcdef"
    layout, _expectation_value, run, checkpoint, payload, state_path = _terminal_lag_case(
        tmp_path, experiment_id=experiment_id
    )
    registry = _CompletionRegistryStub(run, checkpoint)
    monkeypatch.setattr(workflow, "require_ml_runtime", lambda _device: None)
    monkeypatch.setattr(
        workflow,
        "prepare_runtime_layout",
        lambda _root, operation: layout,
    )
    monkeypatch.setattr(workflow, "_registry", lambda _layout: registry)
    monkeypatch.setattr(workflow, "load_gfm_checkpoint", lambda _checkpoint, map_location: payload)
    monkeypatch.setattr(
        workflow,
        "_train_run",
        lambda **_kwargs: pytest.fail("terminal reconciliation must not train"),
    )

    result = resume_gfm(root=tmp_path, run_id=run.run_id, device="cpu")

    assert result["ok"] is True
    assert result["alreadyCompleted"] is True
    assert result["reconciledTerminalState"] is True
    assert result["bestCheckpointId"] == checkpoint.checkpoint_id
    assert workflow.read_json_object(state_path)["status"] == "succeeded"


def test_failed_batch_probe_leaves_an_atomic_retryable_preflight_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    state_path = tmp_path / "experiment" / "run" / "run-state.json"

    def failed_probe(**_kwargs: object) -> tuple[int, float]:
        raise GfmTrainingError("synthetic probe failure")

    monkeypatch.setattr(workflow, "_probe_batch_size", failed_probe)
    with pytest.raises(GfmTrainingError, match="synthetic probe failure"):
        _probe_with_durable_preflight(
            state_path=state_path,
            state_base={
                "schemaVersion": "gfm.workflow-run-state/1.0",
                "runKind": "pretrain",
                "runId": "run",
                "experimentId": "experiment",
                "status": "must-be-replaced",
            },
            attempt=2,
            config=cast(Any, object()),
            variant="core-base",
            streams={},
            device="cpu",
            seed=20260820,
        )

    state = workflow.read_json_object(state_path)
    assert state["status"] == "preflight"
    assert state["preflightAttemptCount"] == 2
    assert state["runKind"] == "pretrain"
    assert state["runId"] == "run"
    assert isinstance(state["lastPreflightStartedAt"], str)


def test_train_run_persists_full_provenance_before_batch_probe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    corpus = SimpleNamespace(
        domain_id="domain",
        content_hash=HASH_A,
        logical_hash=HASH_B,
    )
    stream = SimpleNamespace(
        manifest={"logicalHash": HASH_A},
        access_audit={"testArtifactsOpened": False},
    )
    config = SimpleNamespace(
        config_hash=HASH_C,
        dev=object(),
        formal=object(),
    )
    monkeypatch.setattr(workflow, "set_seed", lambda *_args: None)
    monkeypatch.setattr(
        workflow, "_make_domain_streams", lambda *_args, **_kwargs: {"domain": stream}
    )
    monkeypatch.setattr(workflow, "code_identity_hash", lambda: HASH_D)
    monkeypatch.setattr(workflow, "_environment_hash", lambda _device: HASH_E)
    monkeypatch.setattr(workflow, "_embedding_artifact_evidence", lambda _value: {})

    def fail_probe(**_kwargs: object) -> tuple[int, float]:
        raise GfmTrainingError("probe failed after durable marker")

    monkeypatch.setattr(workflow, "_probe_batch_size", fail_probe)
    layout = SimpleNamespace(gfm_runs=tmp_path / "runs")
    with pytest.raises(GfmTrainingError, match="durable marker"):
        workflow._train_run(
            layout=cast(Any, layout),
            experiment_id="experiment",
            config=cast(Any, config),
            corpora=(cast(Any, corpus),),
            protocols=(),
            embeddings={},
            phase="dev",
            variant="core-base",
            seed=20260820,
            device="cpu",
        )

    state = workflow.read_json_object(
        tmp_path / "runs" / "experiment" / "experiment-core-base-20260820" / "run-state.json"
    )
    assert state["status"] == "preflight"
    assert state["runKind"] == "pretrain"
    assert state["configHash"] == HASH_C
    assert state["codeHash"] == HASH_D
    assert state["environmentHash"] == HASH_E
    assert state["corpusHashes"] == [HASH_B]


def test_resume_retries_a_checkpoint_free_pretrain_from_verified_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout = RuntimeLayout(tmp_path)
    run_id = "experiment-core-base-20260820"
    atomic_write_json(
        layout.gfm_runs / "experiment" / run_id / "run-state.json",
        {
            "runKind": "pretrain",
            "runId": run_id,
            "experimentId": "experiment",
            "phase": "dev",
            "variant": "core-base",
            "seed": 20260820,
            "status": "preflight",
            "heartbeat": {
                "schemaVersion": "gfm.pretrain-heartbeat/1.0",
                "optimizerStep": 50,
            },
        },
    )
    calls: list[dict[str, object]] = []

    def fake_train_run(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"runId": run_id}

    monkeypatch.setattr(workflow, "require_ml_runtime", lambda _device: None)
    monkeypatch.setattr(
        workflow,
        "prepare_runtime_layout",
        lambda _root, operation: layout,
    )
    monkeypatch.setattr(workflow, "_registry", lambda _layout: _RegistryStub(None))
    monkeypatch.setattr(workflow, "_load_pretrain_config", lambda *_args: object())
    monkeypatch.setattr(
        workflow,
        "_ensure_pretrain_evidence",
        lambda *_args, **_kwargs: ((), {}),
    )
    monkeypatch.setattr(workflow, "_register_prerequisites", lambda *_args: ())
    monkeypatch.setattr(workflow, "_experiment_id", lambda **_kwargs: "experiment")
    monkeypatch.setattr(workflow, "_train_run", fake_train_run)

    result = resume_gfm(root=tmp_path, run_id=run_id, device="cpu")

    assert result["resumedFromCheckpointId"] is None
    assert result["resumedFromPreflight"] is True
    assert calls[0]["retry_without_checkpoint"] is True
    assert "resume_manifest" not in calls[0]


def test_dev_entry_points_open_only_physical_validation_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout = RuntimeLayout(tmp_path)
    config = SimpleNamespace(
        architecture=SimpleNamespace(candidates=("core-base",)),
        dev=SimpleNamespace(seeds=(20260820,)),
        formal=SimpleNamespace(seeds=(20260821,)),
        config_hash=HASH_A,
        run_kind="formal",
    )
    calls: list[dict[str, object]] = []

    def evidence(_layout: RuntimeLayout, **kwargs: object) -> tuple[tuple[()], dict[str, object]]:
        calls.append(dict(kwargs))
        return (), {}

    monkeypatch.setattr(workflow, "require_ml_runtime", lambda _device: None)
    monkeypatch.setattr(workflow, "prepare_runtime_layout", lambda *_args, **_kwargs: layout)
    monkeypatch.setattr(workflow, "_load_pretrain_config", lambda *_args: config)
    monkeypatch.setattr(workflow, "_ensure_pretrain_evidence", evidence)
    monkeypatch.setattr(workflow, "_register_prerequisites", lambda *_args: ())
    monkeypatch.setattr(workflow, "_experiment_id", lambda **_kwargs: "experiment")
    monkeypatch.setattr(workflow, "_reuse_pretrain_matrix_cell", lambda **_kwargs: None)
    monkeypatch.setattr(
        workflow,
        "_train_run",
        lambda **_kwargs: {"runId": "dev", "bestCheckpointId": "best"},
    )

    pretrain_gfm(
        root=tmp_path,
        phase="dev",
        config="socialgraph-core.json",
        device="cpu",
        variant="core-base",
        seed=20260820,
    )
    _pretrain_worker(
        root=tmp_path,
        phase="dev",
        config="socialgraph-core.json",
        variant="core-base",
        seed=20260820,
        device="cpu",
    )

    assert calls == [
        {
            "formal_required": False,
            "maximum_role": "validation",
            "physical_boundary": True,
        },
        {
            "formal_required": False,
            "maximum_role": "validation",
            "physical_boundary": True,
        },
    ]


def test_checkpoint_free_dev_resume_uses_physical_validation_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    layout = RuntimeLayout(tmp_path)
    run_id = "experiment-core-base-20260820"
    atomic_write_json(
        layout.gfm_runs / "experiment" / run_id / "run-state.json",
        {
            "runKind": "pretrain",
            "runId": run_id,
            "experimentId": "experiment",
            "phase": "dev",
            "variant": "core-base",
            "seed": 20260820,
            "status": "preflight",
        },
    )
    calls: list[dict[str, object]] = []

    def evidence(_layout: RuntimeLayout, **kwargs: object) -> tuple[tuple[()], dict[str, object]]:
        calls.append(dict(kwargs))
        return (), {}

    monkeypatch.setattr(workflow, "require_ml_runtime", lambda _device: None)
    monkeypatch.setattr(workflow, "prepare_runtime_layout", lambda *_args, **_kwargs: layout)
    monkeypatch.setattr(workflow, "_registry", lambda _layout: _RegistryStub(None))
    monkeypatch.setattr(workflow, "_load_pretrain_config", lambda *_args: object())
    monkeypatch.setattr(workflow, "_ensure_pretrain_evidence", evidence)
    monkeypatch.setattr(workflow, "_register_prerequisites", lambda *_args: ())
    monkeypatch.setattr(workflow, "_experiment_id", lambda **_kwargs: "experiment")
    monkeypatch.setattr(workflow, "_train_run", lambda **_kwargs: {"runId": run_id})

    resume_gfm(root=tmp_path, run_id=run_id, device="cpu")

    assert calls == [{"maximum_role": "validation", "physical_boundary": True}]


def test_train_epoch_heartbeat_observes_every_fifty_optimizer_steps() -> None:
    class _Loss:
        def __init__(self, value: float) -> None:
            self.value = value

        def detached(self) -> dict[str, float]:
            return {"total": self.value}

    class _Trainer:
        def __init__(self) -> None:
            self.optimizer_step = 0
            self.global_step = 0

        def _forward_loss_and_moments(
            self, _batch: object, _reference: object | None = None
        ) -> tuple[_Loss, None]:
            return _Loss(float(self.global_step)), None

        def _apply_optimizer_step(self, *, partial_accumulation: int | None = None) -> None:
            assert partial_accumulation is None
            self.optimizer_step += 1

        def train_epoch(self, _loaders: object) -> str:
            for _ in range(120):
                self._forward_loss_and_moments(object())
                self.global_step += 1
                self._apply_optimizer_step()
            return "done"

    trainer = _Trainer()
    original_forward = trainer._forward_loss_and_moments
    original_apply = trainer._apply_optimizer_step
    observed: list[tuple[int, dict[str, float]]] = []
    optimizer_observed: list[int] = []

    result = _train_epoch_with_heartbeats(
        trainer,
        {},
        every_optimizer_steps=50,
        heartbeat=lambda losses: observed.append((trainer.optimizer_step, dict(losses))),
        after_optimizer_step=lambda: optimizer_observed.append(trainer.optimizer_step),
    )

    assert result == "done"
    assert observed == [(50, {"total": 49.0}), (100, {"total": 99.0})]
    assert optimizer_observed == list(range(1, 121))
    assert trainer._forward_loss_and_moments == original_forward
    assert trainer._apply_optimizer_step == original_apply


def test_pretrain_lr_advances_per_optimizer_step_and_resume_replays_trajectory() -> None:
    torch = pytest.importorskip("torch")

    class _Loss:
        def detached(self) -> dict[str, float]:
            return {"total": 1.0}

    class _Trainer:
        def __init__(self, optimizer: Any, *, optimizer_step: int = 0) -> None:
            self.optimizer = optimizer
            self.optimizer_step = optimizer_step
            self.global_step = optimizer_step

        def _forward_loss_and_moments(
            self, _batch: object, _reference: object | None = None
        ) -> tuple[_Loss, None]:
            return _Loss(), None

        def _apply_optimizer_step(self, *, partial_accumulation: int | None = None) -> None:
            assert partial_accumulation is None
            self.optimizer.step()
            self.optimizer_step += 1

        def train_epoch(self, loaders: object) -> None:
            for _ in range(int(loaders)):
                self._forward_loss_and_moments(object())
                self.global_step += 1
                self._apply_optimizer_step()

    def components(*, optimizer_step: int = 0) -> tuple[Any, Any, Any]:
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: _warmup_cosine(step, maximum=10, warmup_ratio=0.2),
        )
        return _Trainer(optimizer, optimizer_step=optimizer_step), optimizer, scheduler

    def advance(trainer: Any, scheduler: Any, count: int) -> list[float]:
        values: list[float] = []

        def after_step() -> None:
            scheduler.step()
            assert scheduler.last_epoch == trainer.optimizer_step
            values.append(float(trainer.optimizer.param_groups[0]["lr"]))

        _train_epoch_with_heartbeats(
            trainer,
            count,
            every_optimizer_steps=50,
            heartbeat=lambda _losses: None,
            after_optimizer_step=after_step,
        )
        return values

    full_trainer, _full_optimizer, full_scheduler = components()
    uninterrupted = advance(full_trainer, full_scheduler, 10)
    assert uninterrupted[0] == pytest.approx(1e-3)
    assert uninterrupted[4] == pytest.approx(
        1e-3 * _warmup_cosine(5, maximum=10, warmup_ratio=0.2)
    )
    assert uninterrupted[-1] == pytest.approx(0.0)

    first_trainer, first_optimizer, first_scheduler = components()
    prefix = advance(first_trainer, first_scheduler, 4)
    optimizer_state = first_optimizer.state_dict()
    scheduler_state = first_scheduler.state_dict()

    resumed_trainer, resumed_optimizer, resumed_scheduler = components(optimizer_step=4)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)
    assert resumed_scheduler.last_epoch == resumed_trainer.optimizer_step == 4
    resumed = prefix + advance(resumed_trainer, resumed_scheduler, 6)

    assert resumed == pytest.approx(uninterrupted, abs=0.0)


def test_lodo_non_multiple_step_limit_and_resumed_lr_trajectory() -> None:
    torch = pytest.importorskip("torch")
    from socialgraph_gfm.gfm.sampling import RoundRobinDomainScheduler

    class _Loss:
        def detached(self) -> dict[str, float]:
            return {"total": 1.0}

    class _Trainer:
        def __init__(self, optimizer: Any) -> None:
            self.optimizer = optimizer
            self.optimizer_step = 0
            self.global_step = 0
            self.scheduler = RoundRobinDomainScheduler(("a", "b", "c"))
            self.domain_history: list[str] = []

        def _forward_loss_and_moments(
            self, _batch: object, _reference: object | None = None
        ) -> tuple[_Loss, None]:
            return _Loss(), None

        def _apply_optimizer_step(self, *, partial_accumulation: int | None = None) -> None:
            assert partial_accumulation is None
            self.optimizer.step()
            self.optimizer_step += 1

        def train_epoch(self, loaders: object) -> SimpleNamespace:
            active = set(loaders)
            before = self.optimizer_step
            while active:
                domain = self.scheduler.next_domain(active)
                self.domain_history.append(domain)
                self._forward_loss_and_moments(object())
                self._apply_optimizer_step()
                self.global_step += 1
                active.remove(domain)
            return SimpleNamespace(optimizer_steps=self.optimizer_step - before)

    class _LrTrace:
        def __init__(self, optimizer: Any) -> None:
            self.optimizer = optimizer
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: _warmup_cosine(step, maximum=5, warmup_ratio=0.2),
            )
            self.history: list[float] = []

        @property
        def last_epoch(self) -> int:
            return int(self.scheduler.last_epoch)

        def step(self) -> None:
            self.scheduler.step()
            self.history.append(float(self.optimizer.param_groups[0]["lr"]))

        def state_dict(self) -> dict[str, Any]:
            return self.scheduler.state_dict()

        def load_state_dict(self, state: dict[str, Any]) -> None:
            self.scheduler.load_state_dict(state)

    def components() -> tuple[_Trainer, Any, _LrTrace]:
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        trainer = _Trainer(optimizer)
        return trainer, optimizer, _LrTrace(optimizer)

    def block(trainer: _Trainer, scheduler: _LrTrace) -> None:
        _train_lodo_optimizer_block(
            trainer,
            scheduler,
            {domain: [object()] for domain in ("a", "b", "c")},
            maximum_steps=5,
        )

    full_trainer, _full_optimizer, full_scheduler = components()
    block(full_trainer, full_scheduler)
    block(full_trainer, full_scheduler)
    assert full_trainer.optimizer_step == full_scheduler.last_epoch == 5
    assert full_trainer.domain_history == ["a", "b", "c", "a", "b"]
    assert len(full_scheduler.history) == 5

    first_trainer, first_optimizer, first_scheduler = components()
    block(first_trainer, first_scheduler)
    assert first_trainer.optimizer_step == first_scheduler.last_epoch == 3
    optimizer_state = first_optimizer.state_dict()
    lr_state = first_scheduler.state_dict()
    round_robin_state = first_trainer.scheduler.state_dict()
    prefix_lr = list(first_scheduler.history)
    prefix_domains = list(first_trainer.domain_history)

    resumed_trainer, resumed_optimizer, resumed_scheduler = components()
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(lr_state)
    resumed_trainer.optimizer_step = 3
    resumed_trainer.global_step = 3
    resumed_trainer.scheduler.load_state_dict(round_robin_state)
    assert resumed_scheduler.last_epoch == resumed_trainer.optimizer_step
    block(resumed_trainer, resumed_scheduler)

    assert resumed_trainer.optimizer_step == resumed_scheduler.last_epoch == 5
    assert prefix_domains + resumed_trainer.domain_history == full_trainer.domain_history
    assert prefix_lr + resumed_scheduler.history == pytest.approx(
        full_scheduler.history, abs=0.0
    )


def test_pretrain_heartbeat_is_atomic_complete_and_resume_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    state_path = tmp_path / "run-state.json"
    trainer = SimpleNamespace(optimizer_step=50, global_step=100)
    stream = SimpleNamespace(
        cursor=123,
        epoch=2,
        negative_sampling_audit={"futureAccessCount": 0, "sampleCount": 100},
    )
    monkeypatch.setattr(workflow, "_process_rss_mib", lambda: 321.5)

    _pretrain_heartbeat(
        state_path=state_path,
        state_base={"runId": "run", "startedAt": datetime.now(UTC).isoformat()},
        batch_size=2048,
        accumulation=2,
        probe_peak=1234.0,
        preflight_attempt=1,
        trainer=trainer,
        streams={"domain": cast(Any, stream)},
        started=datetime.now(UTC),
        device="cpu",
        last_losses={"total": 1.25, "temporal_next_event": 0.5},
    )

    state = workflow.read_json_object(state_path)
    assert state["status"] == "running"
    assert state["batchSize"] == 2048
    assert state["gradientAccumulation"] == 2
    heartbeat = state["heartbeat"]
    assert heartbeat["optimizerStep"] == 50
    assert heartbeat["globalStep"] == 100
    assert heartbeat["lastLosses"]["total"] == 1.25
    assert heartbeat["domainCursors"] == {"domain": {"cursor": 123, "epoch": 2}}
    assert heartbeat["rssMiB"] == 321.5
    assert heartbeat["peakCudaMemoryMiB"] == 1234.0
    assert heartbeat["negativeSamplingAudits"]["domain"]["futureAccessCount"] == 0
    assert heartbeat["negativeSamplingAuditsHash"] == canonical_sha256(
        heartbeat["negativeSamplingAudits"]
    )


def test_checkpoint_free_retry_refuses_unknown_run_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    atomic_write_json(run_dir / "run-state.json", {"status": "preflight"})
    (run_dir / "unknown.bin").write_bytes(b"do-not-overwrite")

    with pytest.raises(GfmTrainingError, match="uncommitted run artifacts"):
        _ensure_checkpoint_free_retry_directory(run_dir)
    assert (run_dir / "unknown.bin").read_bytes() == b"do-not-overwrite"


def test_registry_completed_run_and_checkpoint_publication_is_atomic(
    tmp_path: Path,
) -> None:
    _layout, _expectation_value, run, checkpoint, _payload, _state_path = _terminal_lag_case(
        tmp_path / "case"
    )

    def seed_references(registry: GfmRegistry) -> None:
        with registry.connect() as connection:
            for index, corpus_hash in enumerate(run.corpus_hashes):
                connection.execute(
                    "INSERT INTO gfm_domain_corpora "
                    "(corpus_id, logical_hash, domain_id, manifest_json) "
                    "VALUES (?, ?, ?, ?)",
                    (f"corpus-{index}", corpus_hash, f"domain-{index}", "{}"),
                )
            connection.execute(
                "INSERT INTO gfm_task_protocols "
                "(protocol_id, protocol_hash, task_id, manifest_json) "
                "VALUES (?, ?, ?, ?)",
                ("protocol", run.task_protocol_hashes[0], "task", "{}"),
            )

    committed = GfmRegistry(tmp_path / "committed.sqlite3")
    seed_references(committed)
    committed.record_completed_run(run, checkpoint)
    assert committed.get_run(run.run_id) == run
    assert committed.get_checkpoint(checkpoint.checkpoint_id) == checkpoint

    rolled_back = GfmRegistry(tmp_path / "rolled-back.sqlite3")
    seed_references(rolled_back)
    invalid_values = checkpoint.model_dump(
        mode="python",
        by_alias=True,
        exclude={"logical_hash", "created_at"},
    )
    invalid_values["configHash"] = HASH_A
    invalid = GfmCheckpointManifest.create(**invalid_values)
    with pytest.raises(RegistrationRejected, match="provenance differs"):
        rolled_back.record_completed_run(run, invalid)
    assert rolled_back.get_run(run.run_id) is None
    assert rolled_back.get_checkpoint(invalid.checkpoint_id) is None


def test_matrix_rerun_rejects_orphan_and_stale_registered_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socialgraph_gfm.gfm_workflow as workflow

    expectation = _expectation()
    layout = RuntimeLayout(tmp_path)
    run_dir = layout.gfm_runs / expectation.experiment_id / expectation.run_id
    run_dir.mkdir(parents=True)
    monkeypatch.setattr(workflow, "_registry", lambda _layout: _RegistryStub(None))
    with pytest.raises(GfmTrainingError, match="Orphan or incomplete"):
        _validate_completed_matrix_run(layout, expectation)

    now = datetime.now(UTC)
    stale = GfmRunManifest.create(
        runId=expectation.run_id,
        experimentId=expectation.experiment_id,
        phase="pretrain",
        architectureVariant=expectation.variant,
        status="succeeded",
        domainIds=expectation.domain_ids,
        seed=expectation.seed,
        codeHash=HASH_A,
        environmentHash=expectation.environment_hash,
        configHash=expectation.config_hash,
        corpusHashes=expectation.corpus_hashes,
        taskProtocolHashes=expectation.protocol_hashes,
        startedAt=now,
        finishedAt=now,
        peakCudaMemoryMiB=100.0,
    )
    monkeypatch.setattr(workflow, "_registry", lambda _layout: _RegistryStub(stale))
    with pytest.raises(GfmTrainingError, match="stale provenance: codeHash"):
        _validate_completed_matrix_run(layout, expectation)


def test_public_matrix_commands_accept_single_cell_selectors() -> None:
    parser = build_parser()
    pretrain = parser.parse_args(
        [
            "gfm-pretrain",
            "--phase",
            "formal",
            "--config",
            "socialgraph-core.json",
            "--variant",
            "core-base",
            "--seed",
            "20260821",
        ]
    )
    assert (pretrain.variant, pretrain.seed) == ("core-base", 20260821)
    adapt = parser.parse_args(
        [
            "gfm-adapt",
            "--task",
            "collaboration",
            "--experiment-id",
            "experiment",
            "--seed",
            "20260822",
        ]
    )
    assert adapt.seed == 20260822
    lodo = parser.parse_args(
        [
            "gfm-evaluate",
            "--protocol",
            "lodo",
            "--experiment-id",
            "experiment",
            "--held-out-domain",
            "academic",
            "--variant",
            "core-moe",
            "--seed",
            "20260823",
        ]
    )
    assert (lodo.held_out_domain, lodo.variant, lodo.seed) == (
        "academic",
        "core-moe",
        20260823,
    )
