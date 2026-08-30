from datetime import UTC, datetime

import pytest

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.contracts import BaselineCheckpointManifest, BaselineRunManifest
from socialgraph_gfm.errors import RegistrationRejected
from socialgraph_gfm.registry import LocalRegistry


SHA = "1" * 64


def baseline_run(
    run_id="run-1", *, track="ogb_official", model="graphsage", seed=20260812
):
    now = datetime.now(UTC).isoformat()
    return {
        "schemaVersion": "gfm.baseline-run/1.0",
        "runId": run_id,
        "experimentId": "experiment-1",
        "runKind": "baseline",
        "phase": "formal",
        "track": track,
        "model": model,
        "status": "succeeded",
        "seed": seed,
        "codeHash": SHA,
        "environmentHash": SHA,
        "corpusHash": SHA,
        "configHash": SHA,
        "startedAt": now,
        "finishedAt": now,
        "bestEpoch": 10,
        "bestValidationHits50": 0.5,
        "peakCudaMemoryMiB": 1000.0,
        "failureCode": None,
        "artifacts": [],
        "registrable": False,
    }


def baseline_checkpoint():
    return {
        "schemaVersion": "gfm.baseline-checkpoint/1.0",
        "checkpointId": "checkpoint-1",
        "runId": "run-1",
        "epoch": 10,
        "track": "ogb_official",
        "model": "graphsage",
        "checkpointKind": "baseline",
        "registrable": False,
        "stateHash": SHA,
        "configHash": SHA,
        "corpusHash": SHA,
        "artifactSha256": SHA,
        "artifactPath": "checkpoint.pt",
        "verificationDigest": None,
        "createdAt": datetime.now(UTC).isoformat(),
    }


def baseline_evaluation():
    payload = {
        "schemaVersion": "gfm.baseline-evaluation/1.0",
        "experimentId": "experiment-1",
        "runId": "run-1",
        "phase": "formal",
        "track": "ogb_official",
        "model": "graphsage",
        "seed": 20260812,
        "validationMetrics": {"Hits@50": 0.5},
        "testMetrics": {"Hits@50": 0.4},
        "strata": {},
        "scoreCounts": {"validation": 1, "test": 1},
        "testReadAfterSelection": True,
    }
    payload["reportHash"] = canonical_sha256(payload)
    return payload


def test_baseline_tables_are_separate_and_never_promotable(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    registry.record_baseline_run(baseline_run())
    registry.record_baseline_checkpoint(baseline_checkpoint())
    registry.record_baseline_evaluation(baseline_evaluation())
    assert registry.counts() == {"runs": 0, "checkpoints": 0, "models": 0}
    assert registry.baseline_counts() == {
        "baseline_runs": 1,
        "baseline_checkpoints": 1,
        "baseline_evaluations": 1,
        "baseline_acceptances": 0,
    }
    with pytest.raises(RegistrationRejected, match="never promotable"):
        registry.register_baseline_model("model-1", "checkpoint-1")
    with pytest.raises(RegistrationRejected, match="never promotable"):
        registry.register_model(
            "model-1",
            BaselineRunManifest.model_validate(baseline_run()),  # type: ignore[arg-type]
            BaselineCheckpointManifest.model_validate(baseline_checkpoint()),  # type: ignore[arg-type]
            validated=True,
        )
    assert [run.run_id for run in registry.list_baseline_runs("experiment-1")] == ["run-1"]
    assert [
        checkpoint.checkpoint_id
        for checkpoint in registry.list_baseline_checkpoints("experiment-1")
    ] == ["checkpoint-1"]
    assert len(registry.list_baseline_evaluations("experiment-1")) == 1
    assert registry.get_baseline_acceptance("experiment-1") is None


def test_baseline_checkpoint_requires_a_known_baseline_run(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    with pytest.raises(RegistrationRejected, match="unknown baseline run"):
        registry.record_baseline_checkpoint(baseline_checkpoint())


def test_baseline_registry_rejects_a_promotable_checkpoint(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    registry.record_baseline_run(baseline_run())
    checkpoint = baseline_checkpoint()
    checkpoint["registrable"] = True
    with pytest.raises(RegistrationRejected, match="registrable=false"):
        registry.record_baseline_checkpoint(checkpoint)


def test_acceptance_cannot_make_readiness_true_without_complete_registry_evidence(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    payload = {
        "schemaVersion": "gfm.baseline-acceptance/1.0",
        "experimentId": "experiment-1",
        "accepted": True,
        "corpusHash": SHA,
        "configHash": SHA,
        "requiredLearningRuns": 12,
        "completedLearningRuns": 12,
        "completedHeuristicRuns": 6,
        "peakCudaMemoryMiB": 1000.0,
        "metricSummary": {"official_graphsage": {"validationHits50": 0.5}},
        "gates": {"allRuns": True},
        "warnings": [],
    }
    payload["reportHash"] = canonical_sha256(payload)
    payload["createdAt"] = datetime.now(UTC).isoformat()
    registry.record_baseline_acceptance(payload)
    evidence = registry.validate_baseline_acceptance(payload, corpus_manifest_hash=SHA)
    assert evidence["ready"] is False
    assert any("12 succeeded learning runs" in reason for reason in evidence["reasons"])


def test_acceptance_is_derived_from_complete_registry_matrix_and_metrics(tmp_path):
    registry = LocalRegistry(tmp_path / "registry.sqlite3")
    learning_run_ids = []
    all_run_ids = []
    for track in ("ogb_official", "strict_edge_time"):
        for model in ("mlp", "graphsage"):
            for seed in (20260812, 20260813, 20260814):
                run_id = f"{track}-{model}-{seed}"
                all_run_ids.append(run_id)
                learning_run_ids.append((run_id, track, model, seed))
                registry.record_baseline_run(
                    baseline_run(run_id, track=track, model=model, seed=seed)
                )
        for model in ("cn", "aa", "ra"):
            run_id = f"{track}-{model}"
            all_run_ids.append(run_id)
            registry.record_baseline_run(
                baseline_run(run_id, track=track, model=model, seed=20260812)
            )

    for run_id, track, model, seed in learning_run_ids:
        checkpoint = baseline_checkpoint()
        checkpoint.update(
            {
                "checkpointId": f"checkpoint-{run_id}",
                "runId": run_id,
                "track": track,
                "model": model,
                "verificationDigest": (
                    SHA if track == "ogb_official" and model == "graphsage" else None
                ),
            }
        )
        registry.record_baseline_checkpoint(checkpoint)

    for run_id in all_run_ids:
        parts = run_id.split("-")
        if run_id.startswith("strict_edge_time"):
            track = "strict_edge_time"
            model = parts[1]
        else:
            track = "ogb_official"
            model = parts[1]
        seed = int(parts[-1]) if parts[-1].isdigit() else 20260812
        test_hits = 0.4 if model == "graphsage" else 0.2
        evaluation = {
            "schemaVersion": "gfm.baseline-evaluation/1.0",
            "experimentId": "experiment-1",
            "runId": run_id,
            "phase": "formal",
            "track": track,
            "model": model,
            "seed": seed,
            "validationMetrics": {"hits@50": 0.5},
            "testMetrics": {"hits@50": test_hits},
            "strata": (
                {"first_time": {"count": 1.0}, "repeated": {"count": 1.0}}
                if track == "strict_edge_time"
                else {}
            ),
            "scoreCounts": {"validation": 1, "test": 1},
            "testReadAfterSelection": True,
        }
        evaluation["reportHash"] = canonical_sha256(evaluation)
        registry.record_baseline_evaluation(evaluation)

    gates = {
        "corpus_ready": True,
        "config_frozen": True,
        "formal_matrix_complete": True,
        "heuristic_matrix_complete": True,
        "metrics_complete": True,
        "cuda_memory_within_limit": True,
        "official_graphsage_validation_threshold": True,
        "official_graphsage_test_threshold": True,
        "official_graphsage_gain_over_mlp": True,
        "strict_edge_time_audit_passed": True,
        "test_read_after_selection": True,
        "checkpoint_recovery_verified": True,
    }
    acceptance = {
        "schemaVersion": "gfm.baseline-acceptance/1.0",
        "experimentId": "experiment-1",
        "accepted": True,
        "corpusHash": SHA,
        "configHash": SHA,
        "requiredLearningRuns": 12,
        "completedLearningRuns": 12,
        "completedHeuristicRuns": 6,
        "peakCudaMemoryMiB": 1000.0,
        "metricSummary": {"official_graphsage": {"validation_hits@50": 0.5}},
        "gates": gates,
        "warnings": [],
    }
    acceptance["reportHash"] = canonical_sha256(acceptance)
    acceptance["createdAt"] = datetime.now(UTC).isoformat()
    registry.record_baseline_acceptance(acceptance)
    evidence = registry.validate_baseline_acceptance(acceptance, corpus_manifest_hash=SHA)
    assert evidence["ready"] is True, evidence["reasons"]
    with registry.connect() as connection:
        connection.execute(
            "UPDATE baseline_evaluations SET manifest_json='{}' WHERE run_id=?",
            (all_run_ids[0],),
        )
    tampered = registry.validate_baseline_acceptance(acceptance, corpus_manifest_hash=SHA)
    assert tampered["ready"] is False
    assert "a baseline evaluation registry payload is invalid" in tampered["reasons"]
