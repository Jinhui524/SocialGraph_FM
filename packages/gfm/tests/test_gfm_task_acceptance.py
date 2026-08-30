from __future__ import annotations

from datetime import UTC, datetime
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256
from socialgraph_gfm.errors import (
    ContractViolation,
    GfmTrainingError,
    RegistrationRejected,
)
from socialgraph_gfm.gfm.contracts import (
    GfmCheckpointManifest,
    GfmEvaluationReport,
    GfmRunManifest,
    GfmTaskAcceptanceManifest,
    GfmTaskProtocolManifest,
)
from socialgraph_gfm.gfm.task_acceptance import (
    COLLABORATION_PROTOCOL_DOMAINS,
    COLLABORATION_TASK,
    FORMAL_SEEDS,
    build_collaboration_task_acceptance,
    collaboration_protocol,
)
from socialgraph_gfm.gfm.registry import GfmRegistry

HASHES = tuple(character * 64 for character in "abc")
CONFIG_HASH = "d" * 64
CODE_HASH = "e" * 64
ENVIRONMENT_HASH = "f" * 64
PRETRAINING_ACCEPTANCE_HASH = "9" * 64


def _protocol() -> GfmTaskProtocolManifest:
    return collaboration_protocol()


def _report(
    *, kind: str, seed: int, protocol: GfmTaskProtocolManifest
) -> GfmEvaluationReport:
    del protocol
    run_id = f"experiment-adapt-collaboration-core-base-{seed}"
    checkpoint_id = f"{run_id}-best"
    metrics = {
        "future_edge_access_count": 0.0,
        "cutoff_violation_count": 0.0,
        "split_overlap_count": 0.0,
    }
    task = None
    ece = None
    fresh = kind == "fresh_process"
    if kind == "product":
        task = COLLABORATION_TASK
        metrics.update(
            {
                "ndcg@20": 0.66,
                "baseline_ndcg@20": 0.60,
                "recall@20": 0.58,
                "baseline_recall@20": 0.52,
                "bootstrap_ci95_ndcg_gain_lower": 0.01,
                "query_count": 200.0,
            }
        )
    elif kind == "calibration":
        task = COLLABORATION_TASK
        ece = 0.02
        metrics.update(
            {
                "ece": 0.02,
                "brier": 0.10,
                "strata_complete": 1.0,
                "ece_institution_small": 0.02,
                "ece_institution_medium": 0.02,
                "ece_institution_large": 0.02,
                "ece_topic_cluster_0": 0.02,
                "ece_topic_cluster_1": 0.02,
                "ece_topic_cluster_2": 0.02,
                "ece_first_time": 0.02,
                "ece_repeated": 0.02,
            }
        )
    else:
        metrics["fresh_process_repeat_match"] = 1.0
    return GfmEvaluationReport.create(
        reportId=(
            f"{checkpoint_id}-test-{kind}-collaboration"
            if kind in {"product", "calibration"}
            else f"{run_id}-fresh-process"
        ),
        experimentId="experiment",
        runId=run_id,
        checkpointId=checkpoint_id,
        evaluationKind=kind,
        domainId="openalex-graph-ai",
        taskId=task,
        evaluatorCodeHash=(CODE_HASH if task is not None else None),
        evaluatorEnvironmentHash=(ENVIRONMENT_HASH if task is not None else None),
        seed=seed,
        metrics=metrics,
        evidenceArtifactHash="1" * 64,
        evidenceArtifactPath=f"E:/reports/{checkpoint_id}-{kind}.json",
        baselineDefinitionHash="2" * 64 if kind == "product" else None,
        strataDefinitionHash="3" * 64 if kind == "calibration" else None,
        ece=ece,
        brier=0.10 if kind == "calibration" else None,
        peakCudaMemoryMiB=1024.0,
        leakageAuditPassed=True,
        leakageAuditHash="4" * 64,
        leakageAuditPath=f"E:/reports/{checkpoint_id}-audit.json",
        freshProcessVerified=fresh,
        verificationDigest=(canonical_sha256({"seed": seed}) if fresh else None),
    )


def _matrix() -> tuple[
    tuple[GfmRunManifest, ...],
    tuple[GfmCheckpointManifest, ...],
    tuple[GfmEvaluationReport, ...],
    GfmTaskProtocolManifest,
    dict[str, dict[str, object]],
]:
    now = datetime.now(UTC)
    protocol = _protocol()
    runs = []
    checkpoints = []
    reports: list[GfmEvaluationReport] = []
    states: dict[str, dict[str, object]] = {}
    for seed in FORMAL_SEEDS:
        run_id = f"experiment-adapt-collaboration-core-base-{seed}"
        checkpoint_id = f"{run_id}-best"
        run = GfmRunManifest.create(
            runId=run_id,
            experimentId="experiment",
            phase="adapt",
            architectureVariant="core-base",
            status="succeeded",
            domainIds=("openalex-graph-ai",),
            seed=seed,
            codeHash=CODE_HASH,
            environmentHash=ENVIRONMENT_HASH,
            configHash=CONFIG_HASH,
            corpusHashes=HASHES,
            taskProtocolHashes=(protocol.protocol_hash,),
            startedAt=now,
            finishedAt=now,
            peakCudaMemoryMiB=1024.0,
        )
        checkpoint = GfmCheckpointManifest.create(
            checkpointId=checkpoint_id,
            runId=run_id,
            epoch=1,
            step=100,
            componentNames=("product", "product_config"),
            stateHash="5" * 64,
            configHash=CONFIG_HASH,
            corpusHashes=HASHES,
            artifactSha256="6" * 64,
            artifactPath=f"E:/runs/{checkpoint_id}.pt",
        )
        product = _report(kind="product", seed=seed, protocol=protocol)
        calibration = _report(kind="calibration", seed=seed, protocol=protocol)
        fresh = _report(kind="fresh_process", seed=seed, protocol=protocol)
        runs.append(run)
        checkpoints.append(checkpoint)
        reports.extend((product, calibration, fresh))
        states[checkpoint_id] = {
            "schemaVersion": "gfm.product-test-read-state/1.0",
            "experimentId": "experiment",
            "checkpointId": checkpoint_id,
            "task": "collaboration",
            "split": "test",
            "status": "completed",
            "readCount": 1,
            "physicalRoleView": True,
            "maximumRole": "test",
            "productReportHash": product.report_hash,
            "calibrationReportHash": calibration.report_hash,
        }
    return tuple(runs), tuple(checkpoints), tuple(reports), protocol, states


def _backbone_inputs(
    runs: tuple[GfmRunManifest, ...],
    checkpoints: tuple[GfmCheckpointManifest, ...],
) -> tuple[dict[str, dict[str, object]], tuple[str, ...]]:
    bindings: dict[str, dict[str, object]] = {}
    accepted_ids = []
    for run, checkpoint in zip(runs, checkpoints, strict=True):
        backbone_id = f"experiment-pretrain-core-base-{run.seed}-best"
        accepted_ids.append(backbone_id)
        bindings[checkpoint.checkpoint_id] = {
            "checkpointId": backbone_id,
            "stateHash": canonical_sha256({"backboneSeed": run.seed}),
            "seed": run.seed,
            "architectureVariant": run.architecture_variant,
            "configHash": run.config_hash,
            "codeHash": run.code_hash,
            "environmentHash": run.environment_hash,
            "corpusHashes": run.corpus_hashes,
        }
    return bindings, tuple(accepted_ids)


def _build_acceptance(
    *,
    runs: tuple[GfmRunManifest, ...],
    checkpoints: tuple[GfmCheckpointManifest, ...],
    reports: tuple[GfmEvaluationReport, ...],
    protocol: GfmTaskProtocolManifest,
    states: dict[str, dict[str, object]],
):
    bindings, accepted_ids = _backbone_inputs(runs, checkpoints)
    return build_collaboration_task_acceptance(
        experiment_id="experiment",
        runs=runs,
        checkpoints=checkpoints,
        evaluations=reports,
        protocol=protocol,
        test_read_states=states,
        backbone_bindings=bindings,
        accepted_pretraining_checkpoint_ids=accepted_ids,
        pretraining_acceptance_report_hash=PRETRAINING_ACCEPTANCE_HASH,
        accepted_pretraining_variant="core-base",
        selected_variant="core-base",
    )


def test_collaboration_task_acceptance_is_independent_and_non_promotable() -> None:
    runs, checkpoints, reports, protocol, states = _matrix()
    acceptance = _build_acceptance(
        runs=runs,
        checkpoints=checkpoints,
        reports=reports,
        protocol=protocol,
        states=states,
    )
    assert acceptance.accepted
    assert acceptance.formal_seeds == FORMAL_SEEDS
    assert acceptance.gates["physical_test_read_once"]
    assert len(acceptance.test_read_evidence_hashes) == 3
    assert not acceptance.registrable
    assert not acceptance.promotable
    assert not acceptance.exportable
    assert all("newcomer" not in value for value in acceptance.run_ids)


def test_collaboration_task_acceptance_fails_closed_without_completed_test_read() -> None:
    runs, checkpoints, reports, protocol, states = _matrix()
    states[checkpoints[0].checkpoint_id]["status"] = (
        "intent-persisted-before-array-access"
    )
    acceptance = _build_acceptance(
        runs=runs,
        checkpoints=checkpoints,
        reports=reports,
        protocol=protocol,
        states=states,
    )
    assert not acceptance.accepted
    assert not acceptance.gates["physical_test_read_once"]


def test_task_acceptance_contract_rejects_promotion_claims() -> None:
    runs, checkpoints, reports, protocol, states = _matrix()
    acceptance = _build_acceptance(
        runs=runs,
        checkpoints=checkpoints,
        reports=reports,
        protocol=protocol,
        states=states,
    )
    payload = acceptance.model_dump(mode="python", by_alias=True)
    payload["promotable"] = True
    with pytest.raises(ValidationError):
        GfmTaskAcceptanceManifest.model_validate(payload)


def test_task_acceptance_rejects_a_weakened_collaboration_protocol() -> None:
    runs, checkpoints, reports, _, states = _matrix()
    weak = GfmTaskProtocolManifest.create(
        protocolId="socialgraph-fm-collaboration",
        taskId=COLLABORATION_TASK,
        taskFamily="collaboration_ranking",
        domainIds=("openalex-graph-ai",),
        splitStrategy="temporal",
        objectives=("unconstrained-ranking",),
        primaryMetrics=("ndcg@20", "recall@20"),
    )
    bindings, accepted_ids = _backbone_inputs(runs, checkpoints)
    with pytest.raises(ValueError, match="exact collaboration protocol"):
        build_collaboration_task_acceptance(
            experiment_id="experiment",
            runs=runs,
            checkpoints=checkpoints,
            evaluations=reports,
            protocol=weak,
            test_read_states=states,
            backbone_bindings=bindings,
            accepted_pretraining_checkpoint_ids=accepted_ids,
            pretraining_acceptance_report_hash=PRETRAINING_ACCEPTANCE_HASH,
            accepted_pretraining_variant="core-base",
        )


def test_task_acceptance_rejects_cross_backbone_bindings() -> None:
    runs, checkpoints, reports, protocol, states = _matrix()
    bindings, accepted_ids = _backbone_inputs(runs, checkpoints)
    bindings[checkpoints[0].checkpoint_id]["checkpointId"] = "foreign-backbone"
    acceptance = build_collaboration_task_acceptance(
        experiment_id="experiment",
        runs=runs,
        checkpoints=checkpoints,
        evaluations=reports,
        protocol=protocol,
        test_read_states=states,
        backbone_bindings=bindings,
        accepted_pretraining_checkpoint_ids=accepted_ids,
        pretraining_acceptance_report_hash=PRETRAINING_ACCEPTANCE_HASH,
        accepted_pretraining_variant="core-base",
    )
    assert not acceptance.accepted
    assert not acceptance.gates["provenance_binding"]


def test_registry_rejects_product_report_reusing_other_checkpoint_evidence(
    tmp_path,
) -> None:
    registry = GfmRegistry(tmp_path / "registry" / "gfm-registry.sqlite3")
    _, _, reports, _, _ = _matrix()
    original = next(report for report in reports if report.evaluation_kind == "product")
    evidence_path = tmp_path / "reports" / "gfm" / "evidence" / "cross.json"
    audit_path = tmp_path / "reports" / "gfm" / "audits" / "cross.json"
    evidence_path.parent.mkdir(parents=True)
    audit_path.parent.mkdir(parents=True)
    baseline = {"name": "fixed"}
    evidence = {
        "schemaVersion": "gfm.evaluation-evidence/1.0",
        "experimentId": original.experiment_id,
        "evidenceId": original.report_id,
        "payload": {
            "checkpointId": "foreign-checkpoint",
            "evaluatorCodeHash": CODE_HASH,
            "evaluatorEnvironmentHash": ENVIRONMENT_HASH,
            "metrics": dict(original.metrics),
            "baselineDefinition": baseline,
        },
    }
    evidence["logicalHash"] = canonical_sha256(evidence)
    audit = {
        "schemaVersion": "gfm.leakage-audit/1.0",
        "experimentId": original.experiment_id,
        "auditId": original.report_id,
        "counters": {
            name: original.metrics[name]
            for name in (
                "future_edge_access_count",
                "cutoff_violation_count",
                "split_overlap_count",
            )
        },
        "evidence": {
            "checkpointId": "foreign-checkpoint",
            "evaluatorCodeHash": CODE_HASH,
            "evaluatorEnvironmentHash": ENVIRONMENT_HASH,
        },
    }
    audit["logicalHash"] = canonical_sha256(audit)
    evidence_path.write_text(canonical_json(evidence), encoding="utf-8")
    audit_path.write_text(canonical_json(audit), encoding="utf-8")
    values = original.model_dump(
        mode="python", by_alias=True, exclude={"report_hash", "created_at"}
    )
    values.update(
        {
            "evidenceArtifactPath": str(evidence_path),
            "evidenceArtifactHash": file_sha256(evidence_path),
            "leakageAuditPath": str(audit_path),
            "leakageAuditHash": file_sha256(audit_path),
            "baselineDefinitionHash": canonical_sha256(baseline),
        }
    )
    forged = GfmEvaluationReport.create(**values)
    with pytest.raises(RegistrationRejected, match="exact checkpoint/evaluator"):
        registry._verify_evaluation_artifacts(forged)


def test_registry_backbone_binding_rejects_wrong_state_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    registry = GfmRegistry(tmp_path / "registry" / "gfm-registry.sqlite3")
    runs, product_checkpoints, _, _, _ = _matrix()
    backbone_checkpoints: dict[str, GfmCheckpointManifest] = {}
    backbone_runs: dict[str, GfmRunManifest] = {}
    payloads: dict[str, dict[str, object]] = {}
    now = datetime.now(UTC)
    for run, product_checkpoint in zip(runs, product_checkpoints, strict=True):
        backbone_id = f"experiment-pretrain-core-base-{run.seed}-best"
        backbone_run_id = f"experiment-pretrain-core-base-{run.seed}"
        backbone_run = GfmRunManifest.create(
            runId=backbone_run_id,
            experimentId="experiment",
            phase="pretrain",
            architectureVariant="core-base",
            status="succeeded",
            domainIds=COLLABORATION_PROTOCOL_DOMAINS,
            seed=run.seed,
            codeHash=run.code_hash,
            environmentHash=run.environment_hash,
            configHash=run.config_hash,
            corpusHashes=run.corpus_hashes,
            taskProtocolHashes=run.task_protocol_hashes,
            startedAt=now,
            finishedAt=now,
        )
        backbone = GfmCheckpointManifest.create(
            checkpointId=backbone_id,
            runId=backbone_run_id,
            epoch=1,
            step=100,
            componentNames=("core",),
            stateHash=canonical_sha256({"backbone": run.seed}),
            configHash=run.config_hash,
            corpusHashes=run.corpus_hashes,
            artifactSha256=canonical_sha256({"artifact": backbone_id}),
            artifactPath=f"E:/runs/{backbone_id}.pt",
        )
        config: dict[str, object] = {
            "task": "collaboration",
            "seed": run.seed,
            "architectureVariant": run.architecture_variant,
            "backboneCheckpointId": backbone_id,
            "backboneStateHash": backbone.state_hash,
        }
        config_hash = canonical_sha256(config)
        config["taskConfigHash"] = config_hash
        payloads[product_checkpoint.checkpoint_id] = {
            "components": {"product": {}, "product_config": config},
            "best_state": {
                "task": "collaboration",
                "productConfigHash": config_hash,
            },
        }
        payloads[backbone_id] = {"components": {"core": {}}}
        backbone_checkpoints[backbone_id] = backbone
        backbone_runs[backbone_run_id] = backbone_run

    pretraining = SimpleNamespace(
        accepted=True,
        selected_variant="core-base",
        selected_checkpoint_ids=tuple(sorted(backbone_checkpoints)),
        report_hash=PRETRAINING_ACCEPTANCE_HASH,
    )
    checkpoint_map = {
        **{item.checkpoint_id: item for item in product_checkpoints},
        **backbone_checkpoints,
    }
    run_map = {**{item.run_id: item for item in runs}, **backbone_runs}
    monkeypatch.setattr(
        registry,
        "latest_pretraining_acceptance",
        lambda **kwargs: pretraining,
    )
    monkeypatch.setattr(
        registry, "verify_pretraining_acceptance", lambda value: value
    )
    monkeypatch.setattr(registry, "get_checkpoint", checkpoint_map.get)
    monkeypatch.setattr(registry, "get_run", run_map.get)
    import socialgraph_gfm.gfm.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "load_gfm_checkpoint",
        lambda checkpoint, *, map_location: payloads[checkpoint.checkpoint_id],
    )
    first_product = product_checkpoints[0].checkpoint_id
    product_config = payloads[first_product]["components"]["product_config"]
    product_config["backboneStateHash"] = "0" * 64
    checked = dict(product_config)
    checked.pop("taskConfigHash")
    product_config["taskConfigHash"] = canonical_sha256(checked)
    payloads[first_product]["best_state"]["productConfigHash"] = product_config[
        "taskConfigHash"
    ]
    with pytest.raises(RegistrationRejected, match="accepted backbone"):
        registry.collaboration_backbone_bindings(
            experiment_id="experiment",
            product_checkpoint_ids=tuple(
                item.checkpoint_id for item in product_checkpoints
            ),
        )


class _WorkflowRegistry:
    def __init__(
        self,
        runs: tuple[GfmRunManifest, ...],
        checkpoints: tuple[GfmCheckpointManifest, ...],
        reports: tuple[GfmEvaluationReport, ...],
    ) -> None:
        self.runs = runs
        self.checkpoints = checkpoints
        self.reports = reports

    def list_runs(self, *, experiment_id: str):
        assert experiment_id == "experiment"
        return self.runs

    def list_checkpoints(self, *, experiment_id: str):
        assert experiment_id == "experiment"
        return self.checkpoints

    def list_evaluations(self, *, experiment_id: str):
        assert experiment_id == "experiment"
        return self.reports

    def latest_task_acceptance(self, *, experiment_id: str):
        assert experiment_id == "experiment"
        return None

    def collaboration_backbone_bindings(
        self, *, experiment_id: str, product_checkpoint_ids: tuple[str, ...]
    ):
        assert experiment_id == "experiment"
        assert len(product_checkpoint_ids) == 3
        return SimpleNamespace(), {}


def _collaboration_checkpoint_payload() -> dict[str, object]:
    config: dict[str, object] = {"task": "collaboration"}
    task_config_hash = canonical_sha256(config)
    config["taskConfigHash"] = task_config_hash
    return {
        "components": {"product": {}, "product_config": config},
        "best_state": {
            "task": "collaboration",
            "productConfigHash": task_config_hash,
        },
    }


def _patch_task_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    *,
    seed_count: int,
):
    from socialgraph_gfm import gfm_workflow

    runs, checkpoints, reports, protocol, _ = _matrix()
    selected_runs = runs[:seed_count]
    selected_checkpoints = checkpoints[:seed_count]
    fresh = tuple(
        report
        for report in reports
        if report.evaluation_kind == "fresh_process"
        and report.seed in {run.seed for run in selected_runs}
    )
    registry = _WorkflowRegistry(selected_runs, selected_checkpoints, fresh)
    layout = SimpleNamespace(root=tmp_path, gfm_reports=tmp_path / "reports" / "gfm")
    monkeypatch.setattr(gfm_workflow, "prepare_runtime_layout", lambda *args, **kwargs: layout)
    monkeypatch.setattr(gfm_workflow, "_require_experiment_runs", lambda *args: None)
    monkeypatch.setattr(gfm_workflow, "_registry", lambda unused_layout: registry)
    monkeypatch.setattr(
        gfm_workflow,
        "load_gfm_checkpoint",
        lambda *args, **kwargs: _collaboration_checkpoint_payload(),
    )
    monkeypatch.setattr(
        gfm_workflow,
        "_load_pretrain_config",
        lambda *args: SimpleNamespace(
            formal=SimpleNamespace(seeds=FORMAL_SEEDS), config_hash=CONFIG_HASH
        ),
    )
    monkeypatch.setattr(gfm_workflow, "code_identity_hash", lambda: CODE_HASH)
    monkeypatch.setattr(
        gfm_workflow, "_environment_hash", lambda unused_device: ENVIRONMENT_HASH
    )
    monkeypatch.setattr(
        gfm_workflow, "_selected_core_variant", lambda *args: "core-base"
    )
    monkeypatch.setattr(gfm_workflow, "_task_protocols", lambda: (protocol,))
    return gfm_workflow, selected_checkpoints


def test_task_evaluation_refuses_incomplete_matrix_before_test_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    workflow, _ = _patch_task_workflow(monkeypatch, tmp_path, seed_count=2)
    opened = False

    def must_not_open(**kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("test role was opened")

    monkeypatch.setattr(workflow, "_evaluate_product_checkpoint", must_not_open)
    with pytest.raises(ContractViolation, match="exact compatible three-seed"):
        workflow.evaluate_gfm(
            root=tmp_path,
            protocol="product",
            experiment_id="experiment",
            task="collaboration",
        )
    assert not opened
    assert not (tmp_path / "reports" / "gfm" / "experiment" / "test-read").exists()


def test_task_evaluation_rejects_stale_evaluator_before_test_intent(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    workflow, _ = _patch_task_workflow(monkeypatch, tmp_path, seed_count=3)
    monkeypatch.setattr(workflow, "code_identity_hash", lambda: "0" * 64)
    opened = False

    def must_not_open(**kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("test role was opened")

    monkeypatch.setattr(workflow, "_evaluate_product_checkpoint", must_not_open)
    with pytest.raises(ContractViolation, match="evaluator code/environment/config"):
        workflow.evaluate_gfm(
            root=tmp_path,
            protocol="product",
            experiment_id="experiment",
            task="collaboration",
        )
    assert not opened
    assert not (tmp_path / "reports" / "gfm" / "experiment" / "test-read").exists()


def test_task_evaluation_persists_intent_before_open_and_reentry_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    workflow, checkpoints = _patch_task_workflow(monkeypatch, tmp_path, seed_count=3)
    calls = 0

    def observe_intent(**kwargs):
        nonlocal calls
        calls += 1
        checkpoint = kwargs["checkpoint"]
        path = (
            tmp_path
            / "reports"
            / "gfm"
            / "experiment"
            / "test-read"
            / f"{checkpoint.checkpoint_id}-test.json"
        )
        assert path.is_file()
        assert "intent-persisted-before-array-access" in path.read_text(encoding="utf-8")
        raise RuntimeError("physical-test-open-probe")

    monkeypatch.setattr(workflow, "_evaluate_product_checkpoint", observe_intent)
    with pytest.raises(RuntimeError, match="physical-test-open-probe"):
        workflow.evaluate_gfm(
            root=tmp_path,
            protocol="product",
            experiment_id="experiment",
            task="collaboration",
        )
    assert calls == 1
    first_state = (
        tmp_path
        / "reports"
        / "gfm"
        / "experiment"
        / "test-read"
        / f"{checkpoints[0].checkpoint_id}-test.json"
    )
    assert first_state.is_file()
    with pytest.raises(GfmTrainingError, match="already opened"):
        workflow.evaluate_gfm(
            root=tmp_path,
            protocol="product",
            experiment_id="experiment",
            task="collaboration",
        )
    assert calls == 1


def test_concurrent_task_evaluation_opens_each_physical_test_view_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    workflow, checkpoints = _patch_task_workflow(monkeypatch, tmp_path, seed_count=3)
    first_opened = Event()
    release_first = Event()
    opened: list[str] = []
    errors: list[BaseException] = []

    def hold_first_open(**kwargs):
        checkpoint = kwargs["checkpoint"]
        opened.append(checkpoint.checkpoint_id)
        first_opened.set()
        assert release_first.wait(timeout=5)
        raise RuntimeError("stop-after-physical-open")

    def invoke() -> None:
        try:
            workflow.evaluate_gfm(
                root=tmp_path,
                protocol="product",
                experiment_id="experiment",
                task="collaboration",
            )
        except BaseException as error:
            errors.append(error)

    monkeypatch.setattr(workflow, "_evaluate_product_checkpoint", hold_first_open)
    first = Thread(target=invoke)
    second = Thread(target=invoke)
    first.start()
    assert first_opened.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release_first.set()
    first.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert opened == [checkpoints[0].checkpoint_id]
    assert len(errors) == 2
    assert any("exclusive operation is already running" in str(error) for error in errors)
    assert any("stop-after-physical-open" in str(error) for error in errors)


def test_task_selector_is_rejected_for_non_product_protocol(tmp_path) -> None:
    from socialgraph_gfm.gfm_workflow import evaluate_gfm

    with pytest.raises(ContractViolation, match="only --protocol product"):
        evaluate_gfm(
            root=tmp_path,
            protocol="shadow",
            experiment_id="experiment",
            task="collaboration",
        )


def test_existing_reports_with_incomplete_test_state_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    workflow, checkpoints = _patch_task_workflow(monkeypatch, tmp_path, seed_count=3)
    _, _, all_reports, _, _ = _matrix()
    registry = workflow._registry(None)
    registry.reports = all_reports
    state_path = (
        tmp_path
        / "reports"
        / "gfm"
        / "experiment"
        / "test-read"
        / f"{checkpoints[0].checkpoint_id}-test.json"
    )
    state_path.parent.mkdir(parents=True)
    state_path.write_text(
        '{"status":"intent-persisted-before-array-access"}', encoding="utf-8"
    )
    opened = False

    def must_not_open(**kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("test role was opened twice")

    monkeypatch.setattr(workflow, "_evaluate_product_checkpoint", must_not_open)
    with pytest.raises(GfmTrainingError, match="incomplete one-shot"):
        workflow.evaluate_gfm(
            root=tmp_path,
            protocol="product",
            experiment_id="experiment",
            task="collaboration",
        )
    assert not opened
