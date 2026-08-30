from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256
from socialgraph_gfm.errors import RegistrationRejected
from socialgraph_gfm.gfm.contracts import (
    GfmCheckpointManifest,
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
    GfmRunManifest,
    GfmTaskProtocolManifest,
)
from socialgraph_gfm.gfm.pretraining_acceptance import (
    FORMAL_DOMAINS,
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    build_gfm_pretraining_acceptance,
)
from socialgraph_gfm.gfm.registry import GfmRegistry

CONFIG_HASH = "4" * 64
CODE_HASH = "5" * 64
ENVIRONMENT_HASH = "6" * 64
EXPERIMENT_ID = "formal-pretraining-test"


def _corpus(domain: str) -> GfmDomainCorpusManifest:
    return GfmDomainCorpusManifest.create(
        corpusId=domain,
        domainId=domain,
        datasetName=domain,
        datasetVersion="1",
        datasetRole="pretraining",
        licenseId="CC0-1.0",
        licenseEvidenceHash="a" * 64,
        sourceHash=canonical_sha256({"source": domain}),
        contentHash=canonical_sha256({"content": domain}),
        splitHash=canonical_sha256({"split": domain}),
        nodeCount=10,
        edgeCount=20,
        featureModalities=("temporal", "structural"),
        taskIds=("governance.collaboration_recommendation",),
        pointInTimeSafe=True,
        publicCheckpointEligible=True,
        temporalCutoff=datetime(2025, 1, 1, tzinfo=UTC),
        artifactPath=f"E:/runtime/{domain}",
    )


def _run(
    *, variant: str, seed: int, phase: str, held_out: str | None = None
) -> GfmRunManifest:
    run_id = (
        f"{EXPERIMENT_ID}-{variant}-{seed}"
        if phase == "pretrain"
        else f"{EXPERIMENT_ID}-lodo-{variant}-{held_out}-{seed}"
    )
    now = datetime.now(UTC)
    return GfmRunManifest.create(
        runId=run_id,
        experimentId=EXPERIMENT_ID,
        phase=phase,
        architectureVariant=variant,
        status="succeeded",
        domainIds=(
            tuple(sorted(FORMAL_DOMAINS))
            if held_out is None
            else tuple(sorted(FORMAL_DOMAINS - {held_out}))
        ),
        heldOutDomain=held_out,
        seed=seed,
        codeHash=CODE_HASH,
        environmentHash=ENVIRONMENT_HASH,
        configHash=CONFIG_HASH,
        corpusHashes=CORPUS_HASHES,
        taskProtocolHashes=("7" * 64,),
        startedAt=now,
        finishedAt=now,
        peakCudaMemoryMiB=512.0,
    )


def _checkpoint(run: GfmRunManifest) -> GfmCheckpointManifest:
    checkpoint_id = f"{run.run_id}-best-10000"
    return GfmCheckpointManifest.create(
        checkpointId=checkpoint_id,
        runId=run.run_id,
        epoch=1,
        step=10_000,
        componentNames=("core",),
        stateHash=canonical_sha256({"state": checkpoint_id}),
        configHash=CONFIG_HASH,
        corpusHashes=CORPUS_HASHES,
        artifactSha256=canonical_sha256({"artifact": checkpoint_id}),
        artifactPath=f"E:/runtime/{checkpoint_id}.pt",
        registrable=False,
    )


def _report(
    *,
    run: GfmRunManifest,
    checkpoint: GfmCheckpointManifest,
    kind: str,
    domain: str,
    metrics: dict[str, float],
    held_out: str | None = None,
) -> GfmEvaluationReport:
    identity = f"{run.run_id}-{kind}"
    fresh = kind == "fresh_process"
    return GfmEvaluationReport.create(
        reportId=identity,
        experimentId=EXPERIMENT_ID,
        runId=run.run_id,
        checkpointId=checkpoint.checkpoint_id,
        evaluationKind=kind,
        domainId=domain,
        heldOutDomain=held_out,
        seed=run.seed,
        metrics={
            **metrics,
            "future_edge_access_count": 0.0,
            "cutoff_violation_count": 0.0,
            "split_overlap_count": 0.0,
            **(
                {"target_domain_pretrain_access_count": 0.0}
                if kind == "lodo"
                else {}
            ),
            **({"fresh_process_repeat_match": 1.0} if fresh else {}),
        },
        evidenceArtifactHash="8" * 64,
        evidenceArtifactPath=f"E:/runtime/reports/{identity}.json",
        peakCudaMemoryMiB=512.0,
        leakageAuditPassed=True,
        leakageAuditHash="9" * 64,
        leakageAuditPath=f"E:/runtime/reports/{identity}-audit.json",
        freshProcessVerified=fresh,
        verificationDigest=(
            canonical_sha256({"fresh": identity}) if fresh else None
        ),
        warnings=(
            ("physical-test-view-read-once-after-best",)
            if kind == "in_domain"
            else ()
        ),
    )


CORPORA = tuple(_corpus(domain) for domain in sorted(FORMAL_DOMAINS))
CORPUS_HASHES = tuple(corpus.logical_hash for corpus in CORPORA)


def _evidence() -> tuple[
    tuple[GfmRunManifest, ...],
    tuple[GfmCheckpointManifest, ...],
    tuple[GfmEvaluationReport, ...],
]:
    runs: list[GfmRunManifest] = []
    checkpoints: list[GfmCheckpointManifest] = []
    reports: list[GfmEvaluationReport] = []
    lodo_metrics = {
        "few_shot_1_gfm": 0.61,
        "few_shot_1_random_init": 0.50,
        "few_shot_1_single_domain": 0.52,
        "few_shot_5_gfm": 0.66,
        "few_shot_5_random_init": 0.56,
        "few_shot_5_single_domain": 0.58,
        "few_shot_10_gfm": 0.70,
        "few_shot_10_random_init": 0.60,
        "few_shot_10_single_domain": 0.62,
    }
    for variant in FORMAL_VARIANTS:
        for seed in FORMAL_SEEDS:
            run = _run(variant=variant, seed=seed, phase="pretrain")
            checkpoint = _checkpoint(run)
            runs.append(run)
            checkpoints.append(checkpoint)
            reports.extend(
                (
                    _report(
                        run=run,
                        checkpoint=checkpoint,
                        kind="in_domain",
                        domain="multi-domain",
                        metrics={"validation_loss": 0.5, "test_loss": 0.6},
                    ),
                    _report(
                        run=run,
                        checkpoint=checkpoint,
                        kind="fresh_process",
                        domain="multi-domain",
                        metrics={"total": 0.5},
                    ),
                )
            )
        for domain in sorted(FORMAL_DOMAINS):
            for seed in FORMAL_SEEDS:
                run = _run(
                    variant=variant, seed=seed, phase="lodo", held_out=domain
                )
                checkpoint = _checkpoint(run)
                runs.append(run)
                checkpoints.append(checkpoint)
                reports.append(
                    _report(
                        run=run,
                        checkpoint=checkpoint,
                        kind="lodo",
                        domain=domain,
                        held_out=domain,
                        metrics=lodo_metrics,
                    )
                )
    return tuple(runs), tuple(checkpoints), tuple(reports)


def _build(
    runs: tuple[GfmRunManifest, ...],
    checkpoints: tuple[GfmCheckpointManifest, ...],
    reports: tuple[GfmEvaluationReport, ...],
):
    return build_gfm_pretraining_acceptance(
        experiment_id=EXPERIMENT_ID,
        corpora=CORPORA,
        runs=runs,
        checkpoints=checkpoints,
        evaluations=reports,
        config_hash=CONFIG_HASH,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )


def test_pretraining_acceptance_is_product_and_newcomer_independent() -> None:
    acceptance = _build(*_evidence())

    assert acceptance.accepted is True
    assert acceptance.selected_variant == "core-base"
    assert len(acceptance.pretrain_run_ids) == 6
    assert len(acceptance.lodo_run_ids) == 18
    assert len(acceptance.evaluation_report_hashes) == 30
    assert not hasattr(acceptance, "product_task_ids")
    assert all(acceptance.gates.values())


def test_pretraining_acceptance_rejects_one_missing_lodo_cell() -> None:
    runs, checkpoints, reports = _evidence()
    missing = runs[-1]
    acceptance = _build(
        tuple(run for run in runs if run.run_id != missing.run_id),
        tuple(
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.run_id != missing.run_id
        ),
        tuple(report for report in reports if report.run_id != missing.run_id),
    )

    assert acceptance.accepted is False
    assert acceptance.gates["lodo_complete"] is False
    assert acceptance.gates["variant_selection"] is False


def test_pretraining_acceptance_falls_back_to_base_when_moe_does_not_clear_gate() -> None:
    runs, checkpoints, reports = _evidence()
    assert _build(runs, checkpoints, reports).selected_variant == "core-base"


def test_registry_schema_migrates_existing_database_without_product_promotion(
    tmp_path,
) -> None:
    database = tmp_path / "gfm.sqlite3"
    registry = GfmRegistry(database)
    with registry.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert "gfm_pretraining_acceptances" in tables
    assert "gfm_acceptances" in tables
    assert "gfm_models" in tables
    assert registry.latest_pretraining_acceptance() is None


def test_pretraining_contract_rejects_invented_gate() -> None:
    acceptance = _build(*_evidence())
    values = acceptance.model_dump(
        mode="python", by_alias=True, exclude={"report_hash", "created_at"}
    )
    values["gates"] = {**values["gates"], "product_metrics": True}
    from socialgraph_gfm.gfm.contracts import GfmPretrainingAcceptanceManifest

    with pytest.raises(ValueError, match="exactly the fixed hard gates"):
        GfmPretrainingAcceptanceManifest.create(**values)


def _physical_report(
    root: Path,
    report: GfmEvaluationReport,
    run: GfmRunManifest,
) -> GfmEvaluationReport:
    evidence_path = root / "reports" / "gfm" / "evidence" / f"{report.report_id}.json"
    audit_path = root / "reports" / "gfm" / "audits" / f"{report.report_id}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_payload: dict[str, object] = {"metrics": dict(report.metrics)}
    audit_evidence: dict[str, object] = {"reportId": report.report_id}
    if report.evaluation_kind == "in_domain":
        evidence_payload["testReadCount"] = 1
    elif report.evaluation_kind == "fresh_process":
        evidence_payload.update(
            {
                "verificationDigest": report.verification_digest,
                "repeatVerificationDigest": report.verification_digest,
            }
        )
    elif report.evaluation_kind == "lodo":
        held_out = str(report.held_out_domain)
        isolation_hash = canonical_sha256({"isolation": report.report_id})
        evidence_payload["isolationHash"] = isolation_hash
        audit_evidence.update(
            {
                "sourceCorpusHashes": [
                    corpus.logical_hash
                    for corpus in CORPORA
                    if corpus.domain_id != held_out
                ],
                "targetCorpusHash": next(
                    corpus.logical_hash
                    for corpus in CORPORA
                    if corpus.domain_id == held_out
                ),
                "pretrainingLoadedDomainIds": list(run.domain_ids),
                "targetLoadedAfterSourcePretraining": held_out,
                "isolationHash": isolation_hash,
            }
        )
    evidence: dict[str, object] = {
        "schemaVersion": "gfm.evaluation-evidence/1.0",
        "experimentId": EXPERIMENT_ID,
        "evidenceId": report.report_id,
        "payload": evidence_payload,
    }
    evidence["logicalHash"] = canonical_sha256(evidence)
    counters = {
        name: report.metrics[name]
        for name in (
            "future_edge_access_count",
            "cutoff_violation_count",
            "split_overlap_count",
            *(("target_domain_pretrain_access_count",) if report.evaluation_kind == "lodo" else ()),
        )
    }
    audit: dict[str, object] = {
        "schemaVersion": "gfm.leakage-audit/1.0",
        "experimentId": EXPERIMENT_ID,
        "auditId": report.report_id,
        "counters": counters,
        "evidence": audit_evidence,
    }
    audit["logicalHash"] = canonical_sha256(audit)
    evidence_path.write_text(canonical_json(evidence), encoding="utf-8")
    audit_path.write_text(canonical_json(audit), encoding="utf-8")
    values = report.model_dump(
        mode="python", by_alias=True, exclude={"report_hash", "created_at"}
    )
    values.update(
        {
            "evidenceArtifactPath": str(evidence_path),
            "evidenceArtifactHash": file_sha256(evidence_path),
            "leakageAuditPath": str(audit_path),
            "leakageAuditHash": file_sha256(audit_path),
        }
    )
    return GfmEvaluationReport.create(**values)


def test_registry_recomputes_physical_pretraining_evidence_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm.registry as registry_module

    monkeypatch.setattr(
        registry_module,
        "load_gfm_checkpoint",
        lambda _checkpoint, *, map_location="cpu": {"mapLocation": map_location},
    )
    registry = GfmRegistry(tmp_path / "registry" / "gfm-registry.sqlite3")
    for corpus in CORPORA:
        registry.record_corpus(corpus)
    protocol = GfmTaskProtocolManifest.create(
        protocolId="pretraining-matrix-v1",
        taskId="governance.community_pulse_forecast",
        taskFamily="community_forecast",
        domainIds=tuple(sorted(FORMAL_DOMAINS)),
        splitStrategy="few_shot_temporal",
        objectives=("ranking",),
        primaryMetrics=("mrr",),
    )
    registry.record_protocol(protocol)
    raw_runs, _, raw_reports = _evidence()
    runs: list[GfmRunManifest] = []
    checkpoints: list[GfmCheckpointManifest] = []
    reports_by_run = {report.run_id: [] for report in raw_reports}
    for report in raw_reports:
        reports_by_run[report.run_id].append(report)
    for raw_run in raw_runs:
        values = raw_run.model_dump(
            mode="python", by_alias=True, exclude={"manifest_hash"}
        )
        values["taskProtocolHashes"] = (protocol.protocol_hash,)
        run = GfmRunManifest.create(**values)
        checkpoint = _checkpoint(run)
        registry.record_run(run)
        registry.record_checkpoint(checkpoint)
        runs.append(run)
        checkpoints.append(checkpoint)
        for raw_report in reports_by_run[run.run_id]:
            report_values = raw_report.model_dump(
                mode="python", by_alias=True, exclude={"report_hash", "created_at"}
            )
            report_values["checkpointId"] = checkpoint.checkpoint_id
            report = _physical_report(
                tmp_path, GfmEvaluationReport.create(**report_values), run
            )
            registry.record_evaluation(report)

    acceptance = registry.build_pretraining_acceptance(
        experiment_id=EXPERIMENT_ID
    )
    assert acceptance.accepted is True
    registry.record_pretraining_acceptance(acceptance)
    assert (
        registry.latest_pretraining_acceptance(experiment_id=EXPERIMENT_ID)
        == acceptance
    )

    with registry.connect() as connection:
        connection.execute(
            "UPDATE gfm_pretraining_acceptances SET accepted=0 WHERE report_hash=?",
            (acceptance.report_hash,),
        )
    with pytest.raises(RegistrationRejected, match="columns differ"):
        registry.verify_pretraining_acceptance(acceptance)
    with registry.connect() as connection:
        connection.execute(
            "UPDATE gfm_pretraining_acceptances SET accepted=1 WHERE report_hash=?",
            (acceptance.report_hash,),
        )

    first_path = Path(
        registry.list_evaluations(experiment_id=EXPERIMENT_ID)[0].evidence_artifact_path
    )
    first_path.write_text("{}", encoding="utf-8")
    with pytest.raises(RegistrationRejected, match="hash differs"):
        registry.verify_pretraining_acceptance(acceptance)
