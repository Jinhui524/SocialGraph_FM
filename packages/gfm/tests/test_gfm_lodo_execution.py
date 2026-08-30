from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.checkpoint import restore_rng_state
from socialgraph_gfm.errors import ContractViolation, GfmTrainingError
from socialgraph_gfm.gfm.corpus.common import atomic_write_json, read_json_object
from socialgraph_gfm.gfm.lodo_execution import (
    HEARTBEAT_EVERY_OPTIMIZER_STEPS,
    LodoCellIdentity,
    bind_lodo_role_views,
    bind_lodo_selected_indices,
    commit_lodo_progress,
    complete_lodo_stage,
    create_lodo_run_state,
    exclusive_lodo_execution_lock,
    load_lodo_resume_checkpoint,
    load_lodo_run_state,
    lodo_stage_plan,
    persist_lodo_run_state,
    record_lodo_heartbeat,
)


CONFIG = {"schemaVersion": "synthetic-lodo/1.0", "learningRate": 0.01}


def _hash(label: str) -> str:
    return canonical_sha256({"label": label})


def _identity(*, seed: int = 7) -> LodoCellIdentity:
    return LodoCellIdentity(
        experiment_id="formal-experiment",
        run_id="formal-experiment-lodo-core-base-community-7",
        held_out_domain="community",
        source_domain_ids=("academic", "software"),
        architecture_variant="core-base",
        seed=seed,
        config_hash=canonical_sha256(CONFIG),
        code_hash=_hash("code"),
        environment_hash=_hash("environment"),
        corpus_hashes=(_hash("academic"), _hash("software"), _hash("community")),
        protocol_hashes=(_hash("collaboration"), _hash("newcomer")),
        role_view_contract={
            "maximumRole": "validation",
            "physicalBoundary": True,
            "sourceDomains": ["academic", "software"],
            "targetDomain": "community",
            "targetOpensAfterSourceFreeze": True,
            "testReadCount": 0,
        },
    )


def _negative_audit() -> dict:
    return {
        "schemaVersion": "gfm.negative-sampling-audit/1.0",
        "futureUnseenCandidateCount": 0,
        "exactNoFalseNegative": True,
        "causal": True,
        "cutoffVisibleCandidatesOnly": True,
    }


def _new_state(tmp_path: Path, identity: LodoCellIdentity) -> Path:
    state_path = tmp_path / identity.run_id / "run-state.json"
    create_lodo_run_state(
        state_path,
        identity=identity,
        device="cuda",
        started_at=datetime(2026, 8, 13, tzinfo=UTC),
    )
    state = bind_lodo_role_views(
        load_lodo_run_state(state_path, identity=identity),
        role_views={
            "academic": {"maximumRole": "validation", "testArtifactsOpened": False},
            "software": {"maximumRole": "validation", "testArtifactsOpened": False},
        },
    )
    persist_lodo_run_state(state_path, state, identity=identity)
    return state_path


def _commit(
    state_path: Path,
    identity: LodoCellIdentity,
    *,
    model_state: dict,
    optimizer_state: dict,
    scheduler_state: dict,
    step: int,
):
    return commit_lodo_progress(
        state_path,
        identity=identity,
        stage="source:multi",
        optimizer_step=step,
        global_step=step * 2,
        last_losses={"total": 1.0 / max(1, step)},
        components={"current_core": model_state},
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        scaler_state={},
        trainer_state={
            "format": "synthetic-trainer/1.0",
            "optimizerStep": step,
            "globalStep": step * 2,
            "domainSchedulerState": {"cursor": step % 2},
        },
        stream_states={
            "academic": {
                "domainId": "academic",
                "cursor": 100 + step,
                "epoch": 1,
                "negativeSamplingAudit": _negative_audit(),
            },
            "software": {
                "domainId": "software",
                "cursor": 200 + step,
                "epoch": 2,
                "negativeSamplingAudit": _negative_audit(),
            },
        },
        best_state={"bestAvailable": False, "stage": "source:multi"},
        config=CONFIG,
        corpus_hashes=identity.corpus_hashes,
    )


def test_lodo_fixed_stage_plan_contains_three_source_and_twelve_controls() -> None:
    plan = lodo_stage_plan(("software", "academic"))
    assert len(plan) == 15
    assert plan[:3] == (
        "source:multi",
        "source:single:academic",
        "source:single:software",
    )
    assert sum(stage.startswith("target:") for stage in plan) == 12
    assert HEARTBEAT_EVERY_OPTIMIZER_STEPS == 50


def test_lodo_cell_execution_lock_rejects_second_worker(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    with exclusive_lodo_execution_lock(run_dir):
        with pytest.raises(GfmTrainingError, match="already owned"):
            with exclusive_lodo_execution_lock(run_dir):
                raise AssertionError("second worker unexpectedly acquired the cell")
    with exclusive_lodo_execution_lock(run_dir):
        assert True


def test_lodo_state_binds_identity_role_views_and_selected_rows(tmp_path: Path) -> None:
    identity = _identity()
    state_path = _new_state(tmp_path, identity)
    state = load_lodo_run_state(state_path, identity=identity)
    assert state["status"] == "preflight"
    assert state["runId"] == identity.run_id
    assert state["experimentId"] == identity.experiment_id
    assert state["execution"]["roleViewsHash"] == canonical_sha256(
        state["execution"]["roleViews"]
    )

    # Only the current target stage may receive few-shot evidence.
    with pytest.raises(ContractViolation, match="current target stage"):
        bind_lodo_selected_indices(
            state,
            stage="target:1pct:gfm",
            event_indices=(1, 2),
            fraction=0.01,
            full_train_event_count=1_000,
            eligible_pool_count=500,
            eligible_pool_hash=_hash("eligible"),
        )
    with pytest.raises(ContractViolation, match="provenance"):
        load_lodo_run_state(state_path, identity=_identity(seed=8))

    corrupted = deepcopy(read_json_object(state_path))
    corrupted["execution"]["roleViews"]["academic"]["maximumRole"] = "test"
    atomic_write_json(state_path, corrupted)
    with pytest.raises(ContractViolation, match="hash-inconsistent"):
        load_lodo_run_state(state_path, identity=identity)


def test_lodo_latest_tamper_falls_back_to_named_recovery_and_keeps_two_roles(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    identity = _identity()
    state_path = _new_state(tmp_path, identity)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    first, first_manifest = _commit(
        state_path,
        identity,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=50,
    )
    second, second_manifest = _commit(
        state_path,
        identity,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=100,
    )
    assert first_manifest.checkpoint_id != second_manifest.checkpoint_id
    assert second["latestCheckpointId"] == second_manifest.checkpoint_id
    assert second["recoveryCheckpointId"] is not None
    checkpoint_dir = state_path.parent / "checkpoints"
    assert len(tuple(checkpoint_dir.glob("*.manifest.json"))) == 2

    Path(second_manifest.artifact_path).write_bytes(b"tampered")
    resumed_state, resumed_manifest, payload = load_lodo_resume_checkpoint(
        state_path, identity=identity
    )
    assert resumed_manifest is not None and payload is not None
    assert resumed_manifest.step == 50
    assert resumed_state["heartbeat"]["optimizerStep"] == 50
    assert resumed_state["heartbeat"]["lastLosses"] == first["heartbeat"]["lastLosses"]
    assert resumed_state["heartbeat"]["rssMiB"] == first["heartbeat"]["rssMiB"]
    assert resumed_state["execution"]["progressSequence"] == first["execution"][
        "progressSequence"
    ]
    assert len(tuple(checkpoint_dir.glob("*.manifest.json"))) == 1


def test_lodo_stage_boundary_checkpoint_resumes_at_exact_next_stage(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    identity = _identity()
    state_path = _new_state(tmp_path, identity)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state = load_lodo_run_state(state_path, identity=identity)
    advanced = complete_lodo_stage(
        state,
        stage="source:multi",
        result={
            "completedSteps": 30_000,
            "stateHash": _hash("source-state"),
            "peakCudaMemoryMiB": 1024.0,
        },
    )
    committed, _ = commit_lodo_progress(
        state_path,
        identity=identity,
        stage="source:multi",
        optimizer_step=30_000,
        global_step=60_000,
        last_losses={"total": 0.25},
        components={
            "current_core": model.state_dict(),
            "source_multi": model.state_dict(),
        },
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        scaler_state={},
        trainer_state={
            "schemaVersion": "gfm.lodo-trainer-state/1.0",
            "optimizerStep": 30_000,
            "globalStep": 60_000,
            "roundRobinState": {"cursor": 0},
        },
        stream_states={
            "academic": {
                "domainId": "academic",
                "cursor": 1,
                "epoch": 1,
                "negativeSamplingAudit": _negative_audit(),
            },
            "software": {
                "domainId": "software",
                "cursor": 2,
                "epoch": 1,
                "negativeSamplingAudit": _negative_audit(),
            },
        },
        best_state={
            "bestAvailable": True,
            "validationLoss": 0.25,
            "stage": "source:multi",
        },
        config=CONFIG,
        corpus_hashes=identity.corpus_hashes,
        state_override=advanced,
    )
    assert committed["execution"]["currentStage"] == "source:single:academic"
    assert set(committed["execution"]["completedStages"]) == {"source:multi"}

    resumed, manifest, payload = load_lodo_resume_checkpoint(state_path, identity=identity)
    assert manifest is not None and payload is not None
    assert resumed["execution"]["currentStage"] == "source:single:academic"
    assert payload["sampler_state"]["stage"] == "source:multi"
    assert "source_multi" in payload["components"]


def test_lodo_lightweight_heartbeat_does_not_advance_resume_authority(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    identity = _identity()
    state_path = _new_state(tmp_path, identity)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    committed, manifest = _commit(
        state_path,
        identity,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=50,
    )
    assert committed["durableCheckpointStep"] == 50
    observed = record_lodo_heartbeat(
        state_path,
        identity=identity,
        stage="source:multi",
        optimizer_step=100,
        global_step=200,
        last_losses={"total": 0.4},
        stream_states={
            "academic": {
                "cursor": 150,
                "epoch": 2,
                "negativeSamplingAudit": _negative_audit(),
            },
            "software": {
                "cursor": 250,
                "epoch": 3,
                "negativeSamplingAudit": _negative_audit(),
            },
        },
        elapsed_seconds=12.5,
        rss_mib=512.0,
        peak_cuda_memory_mib=1024.0,
    )
    assert observed["observedHeartbeatStep"] == 100
    assert observed["durableCheckpointStep"] == 50
    assert observed["latestCheckpointId"] == manifest.checkpoint_id

    resumed, resumed_manifest, _ = load_lodo_resume_checkpoint(
        state_path, identity=identity
    )
    assert resumed_manifest is not None and resumed_manifest.step == 50
    assert resumed["durableCheckpointStep"] == 50
    assert resumed["observedHeartbeatStep"] == 50
    assert resumed["heartbeat"]["lastLosses"] == committed["heartbeat"]["lastLosses"]
    assert resumed["heartbeat"]["rssMiB"] == committed["heartbeat"]["rssMiB"]


def test_lodo_metadata_transitions_rebind_existing_heartbeat(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    identity = _identity()
    state_path = _new_state(tmp_path, identity)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    state, _ = _commit(
        state_path,
        identity,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=50,
    )
    rebound = bind_lodo_role_views(
        state,
        role_views={"community": {"maximumRole": "validation"}},
    )
    assert rebound["heartbeat"]["executionHash"] == canonical_sha256(
        rebound["execution"]
    )
    advanced = complete_lodo_stage(
        rebound,
        stage="source:multi",
        result={"completedSteps": 50},
    )
    assert advanced["heartbeat"]["executionHash"] == canonical_sha256(
        advanced["execution"]
    )


def test_lodo_target_metadata_transitions_rebind_existing_heartbeat() -> None:
    identity = _identity()
    plan = lodo_stage_plan(identity.source_domain_ids)
    completed = {
        stage: {
            "completedSteps": 1,
            "resultHash": canonical_sha256({"completedSteps": 1}),
        }
        for stage in plan[:3]
    }
    execution = {
        "schemaVersion": "gfm.lodo-execution-snapshot/1.0",
        "stagePlan": list(plan),
        "stagePlanHash": canonical_sha256(plan),
        "currentStageIndex": 3,
        "currentStage": plan[3],
        "completedStages": completed,
        "completedStagesHash": canonical_sha256(completed),
        "progressSequence": 1,
        "roleViews": {},
        "roleViewsHash": canonical_sha256({}),
        "selectedIndices": {},
        "selectedIndicesHash": canonical_sha256({}),
        "testReadCount": 0,
    }
    heartbeat = {
        "schemaVersion": "gfm.lodo-heartbeat/1.0",
        "recordedAt": "2026-08-13T00:00:00+00:00",
        "stage": plan[3],
        "optimizerStep": 50,
        "globalStep": 50,
        "lastLosses": {"total": 1.0},
        "domainCursors": {"community": {"cursor": 1, "epoch": 0}},
        "elapsedSeconds": 1.0,
        "rssMiB": 1.0,
        "peakCudaMemoryMiB": 0.0,
        "negativeSamplingAudits": {"community": _negative_audit()},
        "negativeSamplingAuditsHash": canonical_sha256(
            {"community": _negative_audit()}
        ),
        "executionHash": canonical_sha256(execution),
    }
    state = {
        "schemaVersion": "gfm.lodo-run-state/1.0",
        "runKind": "lodo",
        "runId": identity.run_id,
        "experimentId": identity.experiment_id,
        "status": "running",
        "identity": identity.payload(),
        "identityHash": identity.identity_hash,
        "device": "cuda",
        "startedAt": "2026-08-13T00:00:00+00:00",
        "execution": execution,
        "heartbeat": heartbeat,
        "latestCheckpointId": "synthetic-latest",
        "recoveryCheckpointId": None,
        "bestCheckpointId": None,
        "durableCheckpointStage": plan[3],
        "durableCheckpointStep": 50,
        "observedHeartbeatStep": 50,
        "finishedAt": None,
    }
    selected = bind_lodo_selected_indices(
        state,
        stage=plan[3],
        event_indices=(10, 20, 30),
        fraction=0.01,
        full_train_event_count=1_000,
        eligible_pool_count=300,
        eligible_pool_hash=_hash("eligible-pool"),
    )
    assert selected["heartbeat"]["executionHash"] == canonical_sha256(
        selected["execution"]
    )


def test_lodo_synthetic_interruption_resume_matches_continuous_trajectory(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    identity = _identity()

    def initialize():
        torch.manual_seed(12345)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: max(0.1, 1.0 - step / 120.0)
        )
        return model, optimizer, scheduler

    def train(model, optimizer, scheduler, count: int) -> float:
        loss_value = 0.0
        for _ in range(count):
            features = torch.randn(7, 4)
            target = torch.randn(7, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_value = float(loss.detach())
        return loss_value

    continuous_model, continuous_optimizer, continuous_scheduler = initialize()
    continuous_loss = train(
        continuous_model, continuous_optimizer, continuous_scheduler, 120
    )

    interrupted_model, interrupted_optimizer, interrupted_scheduler = initialize()
    first_loss = train(interrupted_model, interrupted_optimizer, interrupted_scheduler, 50)
    state_path = _new_state(tmp_path, identity)
    commit_lodo_progress(
        state_path,
        identity=identity,
        stage="source:multi",
        optimizer_step=50,
        global_step=50,
        last_losses={"total": first_loss},
        components={"current_core": interrupted_model.state_dict()},
        optimizer_state=interrupted_optimizer.state_dict(),
        scheduler_state=interrupted_scheduler.state_dict(),
        scaler_state={},
        trainer_state={
            "format": "synthetic-trainer/1.0",
            "optimizerStep": 50,
            "globalStep": 50,
            "domainSchedulerState": {"cursor": 0},
        },
        stream_states={
            "academic": {
                "domainId": "academic",
                "cursor": 50,
                "epoch": 0,
                "negativeSamplingAudit": _negative_audit(),
            },
            "software": {
                "domainId": "software",
                "cursor": 50,
                "epoch": 0,
                "negativeSamplingAudit": _negative_audit(),
            },
        },
        best_state={"bestAvailable": False, "stage": "source:multi"},
        config=CONFIG,
        corpus_hashes=identity.corpus_hashes,
    )
    # Simulate work after the last committed heartbeat; it must be discarded.
    train(interrupted_model, interrupted_optimizer, interrupted_scheduler, 13)

    _, manifest, payload = load_lodo_resume_checkpoint(state_path, identity=identity)
    assert manifest is not None and payload is not None
    resumed_model, resumed_optimizer, resumed_scheduler = initialize()
    resumed_model.load_state_dict(payload["components"]["current_core"])
    resumed_optimizer.load_state_dict(payload["optimizer_state"])
    resumed_scheduler.load_state_dict(payload["scheduler_state"])
    restore_rng_state(payload["rng_state"])
    resumed_loss = train(resumed_model, resumed_optimizer, resumed_scheduler, 70)

    assert resumed_loss == continuous_loss
    for expected, actual in zip(
        continuous_model.parameters(), resumed_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    resumed_optimizer_state = resumed_optimizer.state_dict()
    continuous_optimizer_state = continuous_optimizer.state_dict()
    assert resumed_optimizer_state["param_groups"] == continuous_optimizer_state["param_groups"]
    assert resumed_optimizer_state["state"].keys() == continuous_optimizer_state["state"].keys()
    for parameter_id in resumed_optimizer_state["state"]:
        resumed_parameter = resumed_optimizer_state["state"][parameter_id]
        continuous_parameter = continuous_optimizer_state["state"][parameter_id]
        assert resumed_parameter.keys() == continuous_parameter.keys()
        for name in resumed_parameter:
            torch.testing.assert_close(
                resumed_parameter[name], continuous_parameter[name], rtol=0.0, atol=0.0
            )
    assert resumed_scheduler.state_dict() == continuous_scheduler.state_dict()
