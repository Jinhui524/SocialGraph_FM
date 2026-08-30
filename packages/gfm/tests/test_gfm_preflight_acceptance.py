from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256
from socialgraph_gfm.gfm import checkpoint as gfm_checkpoint_module
from socialgraph_gfm.gfm.acceptance import (
    COLLABORATION_TASK,
    NEWCOMER_TASK,
    build_gfm_acceptance,
)
from socialgraph_gfm.gfm.contracts import (
    GfmAcceptanceManifest,
    GfmCheckpointManifest,
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
    GfmRunManifest,
    GfmTaskProtocolManifest,
)
from socialgraph_gfm.gfm.registry import GfmRegistry
from socialgraph_gfm.gfm.task_acceptance import collaboration_protocol
from socialgraph_gfm.preflight import (
    _gfm_acceptance_evidence,
    _gfm_collaboration_task_acceptance_evidence,
    _gfm_corpus_evidence,
    _gfm_task_asset_evidence,
)

DOMAINS = (
    "openalex-graph-ai",
    "thgl-software-2.0.0",
    "wikimedia-talk-article-2011-2015",
)
CONFIG_HASH = "4" * 64
CODE_HASH = "5" * 64
ENVIRONMENT_HASH = "6" * 64


@pytest.fixture(autouse=True)
def _stub_physical_checkpoint_load(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract fixtures use synthetic paths; physical failure has its own test."""

    monkeypatch.setattr(
        gfm_checkpoint_module,
        "load_gfm_checkpoint",
        lambda _manifest, *, map_location="cpu": {"mapLocation": map_location},
    )


def _corpus(root: Path, domain: str, index: int) -> GfmDomainCorpusManifest:
    return GfmDomainCorpusManifest.create(
        corpusId=domain,
        domainId=domain,
        datasetName=domain,
        datasetVersion="1",
        datasetRole="pretraining",
        licenseId="CC0-1.0",
        licenseEvidenceHash="a" * 64,
        sourceHash=canonical_sha256({"source": domain, "index": index}),
        contentHash=canonical_sha256({"prepared": domain, "index": index}),
        splitHash=canonical_sha256({"split": domain}),
        nodeCount=10,
        edgeCount=20,
        featureModalities=("temporal", "structural"),
        taskIds=(COLLABORATION_TASK, NEWCOMER_TASK),
        pointInTimeSafe=True,
        publicCheckpointEligible=True,
        temporalCutoff=datetime(2025, 1, 1, tzinfo=UTC),
        artifactPath=str(root / "datasets" / "processed" / "gfm" / domain),
    )


def _checkpoint(
    *, checkpoint_id: str, run_id: str, corpus_hashes: tuple[str, ...]
) -> GfmCheckpointManifest:
    return GfmCheckpointManifest.create(
        checkpointId=checkpoint_id,
        runId=run_id,
        epoch=1,
        step=10_000,
        componentNames=("encoder",),
        stateHash=canonical_sha256({"state": checkpoint_id}),
        configHash=CONFIG_HASH,
        corpusHashes=corpus_hashes,
        artifactSha256=canonical_sha256({"artifact": checkpoint_id}),
        artifactPath=f"E:/runtime/{checkpoint_id}.pt",
        registrable=False,
        freshProcessDigest=canonical_sha256({"fresh": checkpoint_id}),
    )


def _run(
    *,
    run_id: str,
    phase: str,
    seed: int,
    domains: tuple[str, ...],
    corpus_hashes: tuple[str, ...],
    protocol_hashes: tuple[str, ...],
    held_out: str | None = None,
) -> GfmRunManifest:
    now = datetime.now(UTC)
    return GfmRunManifest.create(
        runId=run_id,
        experimentId="experiment-preflight",
        phase=phase,
        architectureVariant="core-base",
        status="succeeded",
        domainIds=domains,
        heldOutDomain=held_out,
        seed=seed,
        codeHash=CODE_HASH,
        environmentHash=ENVIRONMENT_HASH,
        configHash=CONFIG_HASH,
        corpusHashes=corpus_hashes,
        taskProtocolHashes=protocol_hashes,
        startedAt=now,
        finishedAt=now,
        peakCudaMemoryMiB=512.0,
    )


def _report(
    *,
    root: Path,
    report_id: str,
    run_id: str,
    checkpoint_id: str,
    kind: str,
    seed: int,
    domain: str,
    metrics: dict[str, float],
    held_out: str | None = None,
    task: str | None = None,
    ece: float | None = None,
    fresh: bool = False,
) -> GfmEvaluationReport:
    report_metrics = {
        **metrics,
        **({"brier": 0.01} if kind == "calibration" else {}),
        "future_edge_access_count": 0.0,
        "cutoff_violation_count": 0.0,
        "split_overlap_count": 0.0,
        **(
            {"target_domain_pretrain_access_count": 0.0}
            if kind == "lodo"
            else {}
        ),
        **({"fresh_process_repeat_match": 1.0} if fresh else {}),
    }
    audit_counters = {
        name: report_metrics[name]
        for name in (
            "future_edge_access_count",
            "cutoff_violation_count",
            "split_overlap_count",
            *(("target_domain_pretrain_access_count",) if kind == "lodo" else ()),
        )
    }
    baseline_definition = (
        {"schemaVersion": "test.product-baseline/1.0", "task": task}
        if kind == "product"
        else None
    )
    strata_definition = (
        {"schemaVersion": "test.calibration-strata/1.0", "task": task}
        if kind == "calibration"
        else None
    )
    evidence_path = root / "reports" / "gfm" / "evidence" / f"{report_id}.json"
    audit_path = root / "reports" / "gfm" / "audits" / f"{report_id}.json"
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_artifact = {
        "schemaVersion": "gfm.evaluation-evidence/1.0",
        "experimentId": "experiment-preflight",
        "evidenceId": report_id,
        "payload": {
            **(
                {
                    "checkpointId": checkpoint_id,
                    "evaluatorCodeHash": CODE_HASH,
                    "evaluatorEnvironmentHash": ENVIRONMENT_HASH,
                }
                if kind in {"product", "calibration"}
                else {}
            ),
            "metrics": report_metrics,
            **({"brier": 0.01} if ece is not None else {}),
            **(
                {"baselineDefinition": baseline_definition}
                if baseline_definition is not None
                else {}
            ),
            **(
                {"strataDefinition": strata_definition}
                if strata_definition is not None
                else {}
            ),
        },
    }
    evidence_artifact["logicalHash"] = canonical_sha256(evidence_artifact)
    evidence_path.write_text(
        canonical_json(evidence_artifact),
        encoding="utf-8",
    )
    audit_artifact = {
        "schemaVersion": "gfm.leakage-audit/1.0",
        "experimentId": "experiment-preflight",
        "auditId": report_id,
        "counters": audit_counters,
        "evidence": {
            "reportId": report_id,
            **(
                {
                    "checkpointId": checkpoint_id,
                    "evaluatorCodeHash": CODE_HASH,
                    "evaluatorEnvironmentHash": ENVIRONMENT_HASH,
                }
                if kind in {"product", "calibration"}
                else {}
            ),
        },
    }
    audit_artifact["logicalHash"] = canonical_sha256(audit_artifact)
    audit_path.write_text(
        canonical_json(audit_artifact),
        encoding="utf-8",
    )
    return GfmEvaluationReport.create(
        reportId=report_id,
        experimentId="experiment-preflight",
        runId=run_id,
        checkpointId=checkpoint_id,
        evaluationKind=kind,
        domainId=domain,
        heldOutDomain=held_out,
        taskId=task,
        evaluatorCodeHash=(
            CODE_HASH if kind in {"product", "calibration"} else None
        ),
        evaluatorEnvironmentHash=(
            ENVIRONMENT_HASH if kind in {"product", "calibration"} else None
        ),
        seed=seed,
        metrics=report_metrics,
        evidenceArtifactHash=file_sha256(evidence_path),
        evidenceArtifactPath=str(evidence_path),
        baselineDefinitionHash=(
            canonical_sha256(baseline_definition)
            if baseline_definition is not None
            else None
        ),
        strataDefinitionHash=(
            canonical_sha256(strata_definition)
            if strata_definition is not None
            else None
        ),
        ece=ece,
        brier=0.01 if ece is not None else None,
        peakCudaMemoryMiB=512.0,
        leakageAuditPassed=True,
        leakageAuditHash=file_sha256(audit_path),
        leakageAuditPath=str(audit_path),
        freshProcessVerified=fresh,
        verificationDigest=(
            canonical_sha256({"verification": report_id}) if fresh else None
        ),
    )


def _build_registered_acceptance(
    root: Path,
) -> tuple[GfmAcceptanceManifest, tuple[GfmDomainCorpusManifest, ...]]:
    registry = GfmRegistry(root / "registry" / "gfm-registry.sqlite3")
    corpora = tuple(_corpus(root, domain, 1) for domain in DOMAINS)
    manifest_directory = root / "datasets" / "manifests" / "gfm"
    manifest_directory.mkdir(parents=True)
    for corpus in corpora:
        (manifest_directory / f"{corpus.domain_id}.json").write_text(
            canonical_json(corpus), encoding="utf-8"
        )
        registry.record_corpus(corpus)

    protocols = (
        collaboration_protocol(),
        GfmTaskProtocolManifest.create(
            protocolId="newcomer-v1",
            taskId=NEWCOMER_TASK,
            taskFamily="newcomer_support",
            domainIds=(DOMAINS[0], DOMAINS[2]),
            splitStrategy="few_shot_temporal",
            objectives=("ranking", "participation"),
            primaryMetrics=("ndcg@20", "auprc"),
        ),
    )
    for protocol in protocols:
        registry.record_protocol(protocol)
    protocol_hashes = tuple(item.protocol_hash for item in protocols)
    corpus_by_domain = {item.domain_id: item for item in corpora}

    reports: list[GfmEvaluationReport] = []
    lodo_metrics = {
        "few_shot_1_gfm": 0.42,
        "few_shot_1_random_init": 0.38,
        "few_shot_1_single_domain": 0.39,
        "few_shot_5_gfm": 0.60,
        "few_shot_5_random_init": 0.54,
        "few_shot_5_single_domain": 0.56,
        "few_shot_10_gfm": 0.66,
        "few_shot_10_random_init": 0.58,
        "few_shot_10_single_domain": 0.61,
    }
    for domain in DOMAINS:
        training_domains = tuple(item for item in DOMAINS if item != domain)
        training_hashes = tuple(
            corpus_by_domain[item].logical_hash for item in training_domains
        )
        for seed in (1, 2, 3):
            run_id = f"lodo-{domain}-{seed}"
            checkpoint_id = f"checkpoint-{run_id}"
            registry.record_run(
                _run(
                    run_id=run_id,
                    phase="lodo",
                    seed=seed,
                    domains=training_domains,
                    corpus_hashes=training_hashes,
                    protocol_hashes=protocol_hashes,
                    held_out=domain,
                )
            )
            registry.record_checkpoint(
                _checkpoint(
                    checkpoint_id=checkpoint_id,
                    run_id=run_id,
                    corpus_hashes=training_hashes,
                )
            )
            reports.append(
                _report(
                    root=root,
                    report_id=f"report-{run_id}",
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    kind="lodo",
                    seed=seed,
                    domain=domain,
                    held_out=domain,
                    metrics=lodo_metrics,
                )
            )

    all_hashes = tuple(item.logical_hash for item in corpora)
    collaboration_metrics = {
        "ndcg@20": 0.63,
        "baseline_ndcg@20": 0.58,
        "recall@20": 0.55,
        "baseline_recall@20": 0.50,
        "bootstrap_ci95_ndcg_gain_lower": 0.01,
        "query_count": 200.0,
    }
    newcomer_metrics = {
        "ndcg@20": 0.62,
        "baseline_ndcg@20": 0.57,
        "auprc": 0.25,
        "label_prevalence": 0.15,
        "query_count": 200.0,
        "outcome_count": 200.0,
    }
    for seed in (1, 2, 3):
        run_id = f"formal-{seed}"
        checkpoint_id = f"checkpoint-{run_id}"
        registry.record_run(
            _run(
                run_id=run_id,
                phase="evaluate",
                seed=seed,
                domains=DOMAINS,
                corpus_hashes=all_hashes,
                protocol_hashes=protocol_hashes,
            )
        )
        registry.record_checkpoint(
            _checkpoint(
                checkpoint_id=checkpoint_id,
                run_id=run_id,
                corpus_hashes=all_hashes,
            )
        )
        for task, metrics in (
            (COLLABORATION_TASK, collaboration_metrics),
            (NEWCOMER_TASK, newcomer_metrics),
        ):
            reports.append(
                _report(
                    root=root,
                    report_id=f"product-{task}-{seed}",
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    kind="product",
                    seed=seed,
                    domain=DOMAINS[0],
                    task=task,
                    metrics=metrics,
                )
            )
            reports.append(
                _report(
                    root=root,
                    report_id=f"calibration-{task}-{seed}",
                    run_id=run_id,
                    checkpoint_id=checkpoint_id,
                    kind="calibration",
                    seed=seed,
                    domain=DOMAINS[0],
                    task=task,
                    ece=0.02,
                    metrics={
                        "ece": 0.02,
                        "brier": 0.01,
                        "strata_complete": 1.0,
                        "ece_institution_small": 0.02,
                        "ece_institution_medium": 0.02,
                        "ece_institution_large": 0.02,
                        "ece_topic_cluster_0": 0.02,
                        "ece_topic_cluster_1": 0.02,
                        "ece_topic_cluster_2": 0.02,
                        **(
                            {"ece_first_time": 0.02, "ece_repeated": 0.02}
                            if task == COLLABORATION_TASK
                            else {"ece_newcomer": 0.02}
                        ),
                    },
                )
            )
        reports.append(
            _report(
                root=root,
                report_id=f"fresh-{seed}",
                run_id=run_id,
                checkpoint_id=checkpoint_id,
                kind="fresh_process",
                seed=seed,
                domain="multi-domain",
                metrics={"score": 0.5},
                fresh=True,
            )
        )

    for report in reports:
        registry.record_evaluation(report)
    acceptance = build_gfm_acceptance(
        experiment_id="experiment-preflight",
        checkpoint_id="checkpoint-formal-1",
        corpora=corpora,
        evaluations=tuple(reports),
        config_hash=CONFIG_HASH,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert acceptance.accepted, acceptance.reasons
    # Insert the already independently derived immutable contract so this test
    # exercises the preflight read boundary, not the registry write workflow.
    with sqlite3.connect(root / "registry" / "gfm-registry.sqlite3") as connection:
        connection.execute(
            """
            INSERT INTO gfm_acceptances(
                report_hash, experiment_id, checkpoint_id, accepted, created_at,
                manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                acceptance.report_hash,
                acceptance.experiment_id,
                acceptance.checkpoint_id,
                int(acceptance.accepted),
                acceptance.created_at.isoformat(),
                canonical_json(acceptance),
            ),
        )
    return acceptance, corpora


def _checked_hashes(
    corpora: tuple[GfmDomainCorpusManifest, ...],
) -> dict[str, str]:
    return {item.domain_id: item.content_hash for item in corpora}


def test_gfm_corpus_gate_rejects_non_public_checkpoint_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm.corpus as corpus_module

    manifests = {
        domain: {
            "logicalHash": canonical_sha256({"domain": domain}),
            "privacy": {
                "publicCheckpointEligible": domain != DOMAINS[1],
            },
        }
        for domain in DOMAINS
    }

    def check_corpora(_: str | Path) -> dict[str, object]:
        return {
            "schemaVersion": "gfm.corpora-check/1.0",
            "ready": True,
            "domains": manifests,
        }

    monkeypatch.setattr(corpus_module, "check_all_gfm_corpora", check_corpora)

    evidence, hashes = _gfm_corpus_evidence(tmp_path)

    assert evidence["ready"] is False
    assert hashes == ()
    assert "not eligible for a public checkpoint" in str(evidence["reason"])


def test_base_corpora_unlock_collaboration_without_newcomer_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm.corpus as corpus_module

    def overlay_absent(_: str | Path) -> dict[str, object]:
        raise RuntimeError("newcomer verification overlay is absent")

    monkeypatch.setattr(corpus_module, "check_openalex_newcomers", overlay_absent)

    evidence = _gfm_task_asset_evidence(tmp_path, gfm_corpus_ready=True)

    assert evidence["schemaVersion"] == "gfm.task-assets/1.0"
    assert evidence["baseCorporaReady"] is True
    assert evidence["newcomerOverlay"]["ready"] is False
    assert evidence["tasks"][COLLABORATION_TASK] == {
        "ready": True,
        "requiredAssets": ["gfm-base-corpora"],
        "missingAssets": [],
    }
    assert evidence["tasks"][NEWCOMER_TASK]["ready"] is False
    assert evidence["tasks"][NEWCOMER_TASK]["missingAssets"] == [
        "openalex-newcomer-overlay"
    ]


def test_verified_newcomer_overlay_is_a_separate_hash_bound_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm.corpus as corpus_module

    overlay_hash = "a" * 64
    source_hash = "b" * 64
    monkeypatch.setattr(
        corpus_module,
        "newcomer_overlay_status",
        lambda _: {
            "ready": True,
            "state": "ready",
            "manifestHash": overlay_hash,
            "baseCorpusLogicalHash": "c" * 64,
            "baseCorpusSourceHash": source_hash,
            "verifiedCount": 321,
            "resumePresent": False,
            "reason": None,
        },
    )

    evidence = _gfm_task_asset_evidence(tmp_path, gfm_corpus_ready=True)

    assert evidence["newcomerOverlay"]["ready"] is True
    assert evidence["newcomerOverlay"]["state"] == "ready"
    assert evidence["newcomerOverlay"]["manifestHash"] == overlay_hash
    assert evidence["newcomerOverlay"]["sourceCorpusHash"] == source_hash
    assert evidence["newcomerOverlay"]["baseCorpusLogicalHash"] == "c" * 64
    assert evidence["newcomerOverlay"]["verifiedCount"] == 321
    assert evidence["newcomerOverlay"]["resumePresent"] is False
    assert evidence["newcomerOverlay"]["reason"] is None
    assert evidence["tasks"][COLLABORATION_TASK]["ready"] is True
    assert evidence["tasks"][NEWCOMER_TASK]["ready"] is True
    assert evidence["tasks"][NEWCOMER_TASK]["missingAssets"] == []


def test_overlay_alone_never_substitutes_for_base_corpora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socialgraph_gfm.gfm.corpus as corpus_module

    monkeypatch.setattr(
        corpus_module,
        "newcomer_overlay_status",
        lambda _: {
            "ready": True,
            "state": "ready",
            "manifestHash": "a" * 64,
            "baseCorpusLogicalHash": "c" * 64,
            "baseCorpusSourceHash": "b" * 64,
            "verifiedCount": 1,
            "resumePresent": False,
            "reason": None,
        },
    )

    evidence = _gfm_task_asset_evidence(tmp_path, gfm_corpus_ready=False)

    assert evidence["newcomerOverlay"]["ready"] is True
    assert evidence["tasks"][COLLABORATION_TASK]["ready"] is False
    assert evidence["tasks"][NEWCOMER_TASK]["ready"] is False
    assert evidence["tasks"][NEWCOMER_TASK]["missingAssets"] == [
        "gfm-base-corpora"
    ]


def test_gfm_preflight_recomputes_an_exact_accepted_registry_snapshot(
    tmp_path: Path,
) -> None:
    acceptance, corpora = _build_registered_acceptance(tmp_path)

    evidence = _gfm_acceptance_evidence(
        tmp_path, checked_domain_hashes=_checked_hashes(corpora)
    )

    assert evidence["ready"] is True
    assert evidence["accepted"] is True
    assert evidence["reportHash"] == acceptance.report_hash
    assert evidence["pretrainingValidated"] is True
    assert evidence["productValidated"] is True


def test_collaboration_preflight_treats_a_legacy_registry_as_absent_evidence(
    tmp_path: Path,
) -> None:
    database = tmp_path / "registry" / "gfm-registry.sqlite3"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE gfm_pretraining_acceptances (
                report_hash TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                manifest_json TEXT NOT NULL
            )
            """
        )
    before_hash = file_sha256(database)

    evidence = _gfm_collaboration_task_acceptance_evidence(
        tmp_path,
        checked_domain_hashes={
            domain: canonical_sha256({"domain": domain}) for domain in DOMAINS
        },
    )

    assert evidence["ready"] is False
    assert evidence["accepted"] is False
    assert evidence["reasons"] == [
        "formal collaboration task acceptance evidence is absent"
    ]
    assert file_sha256(database) == before_hash
    assert not database.with_name(f"{database.name}-wal").exists()
    assert not database.with_name(f"{database.name}-shm").exists()
    with sqlite3.connect(database) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert table_names == {"gfm_pretraining_acceptances"}


def test_collaboration_preflight_preserves_other_sqlite_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "registry" / "gfm-registry.sqlite3"
    GfmRegistry(database)

    def reject_read(
        _registry: GfmRegistry, *, experiment_id: str | None = None
    ) -> None:
        del experiment_id
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(GfmRegistry, "latest_task_acceptance", reject_read)

    evidence = _gfm_collaboration_task_acceptance_evidence(
        tmp_path,
        checked_domain_hashes={
            domain: canonical_sha256({"domain": domain}) for domain in DOMAINS
        },
    )

    assert evidence["ready"] is False
    assert evidence["reasons"] == ["database is locked"]


def test_gfm_preflight_rejects_a_missing_or_tampered_physical_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, corpora = _build_registered_acceptance(tmp_path)

    def reject_physical_load(_manifest: object, *, map_location: str) -> object:
        assert map_location == "cpu"
        raise OSError("physical checkpoint is absent or changed")

    monkeypatch.setattr(
        gfm_checkpoint_module, "load_gfm_checkpoint", reject_physical_load
    )
    evidence = _gfm_acceptance_evidence(
        tmp_path, checked_domain_hashes=_checked_hashes(corpora)
    )

    assert evidence["ready"] is False
    assert evidence["accepted"] is False
    assert "physical checkpoint is absent or changed" in str(evidence["reasons"])


def test_gfm_preflight_preserves_explicit_delivery_evidence_selection(
    tmp_path: Path,
) -> None:
    _, corpora = _build_registered_acceptance(tmp_path)
    database = tmp_path / "registry" / "gfm-registry.sqlite3"
    with sqlite3.connect(database) as connection:
        protocol_hashes = tuple(
            row[0]
            for row in connection.execute(
                "SELECT protocol_hash FROM gfm_task_protocols ORDER BY protocol_hash"
            ).fetchall()
        )

    registry = GfmRegistry(database)
    corpus_hashes = tuple(item.logical_hash for item in corpora)
    suite_run_id = "delivery-suite-1"
    suite_checkpoint_id = "checkpoint-delivery-suite-1"
    registry.record_run(
        _run(
            run_id=suite_run_id,
            phase="evaluate",
            seed=1,
            domains=DOMAINS,
            corpus_hashes=corpus_hashes,
            protocol_hashes=protocol_hashes,
        )
    )
    registry.record_checkpoint(
        _checkpoint(
            checkpoint_id=suite_checkpoint_id,
            run_id=suite_run_id,
            corpus_hashes=corpus_hashes,
        )
    )
    suite_fresh = _report(
        root=tmp_path,
        report_id="fresh-delivery-suite-1",
        run_id=suite_run_id,
        checkpoint_id=suite_checkpoint_id,
        kind="fresh_process",
        seed=1,
        domain="delivery-suite",
        metrics={"score": 0.5},
        fresh=True,
    )
    registry.record_evaluation(suite_fresh)

    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT manifest_json FROM gfm_evaluations ORDER BY report_id"
        ).fetchall()
    evaluations = tuple(
        GfmEvaluationReport.model_validate_json(row[0]) for row in rows
    )
    by_id = {report.report_id: report for report in evaluations}
    delivery_ids = (
        f"product-{COLLABORATION_TASK}-1",
        f"calibration-{COLLABORATION_TASK}-1",
        f"product-{NEWCOMER_TASK}-1",
        f"calibration-{NEWCOMER_TASK}-1",
        suite_fresh.report_id,
    )
    delivery_hashes = tuple(by_id[report_id].report_hash for report_id in delivery_ids)
    acceptance = build_gfm_acceptance(
        experiment_id="experiment-preflight",
        checkpoint_id=suite_checkpoint_id,
        corpora=corpora,
        evaluations=evaluations,
        config_hash=CONFIG_HASH,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
        delivery_evidence_report_hashes=delivery_hashes,
    )
    assert acceptance.accepted, acceptance.reasons

    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM gfm_acceptances")
        connection.execute(
            """
            INSERT INTO gfm_acceptances(
                report_hash, experiment_id, checkpoint_id, accepted, created_at,
                manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                acceptance.report_hash,
                acceptance.experiment_id,
                acceptance.checkpoint_id,
                int(acceptance.accepted),
                acceptance.created_at.isoformat(),
                canonical_json(acceptance),
            ),
        )

    evidence = _gfm_acceptance_evidence(
        tmp_path, checked_domain_hashes=_checked_hashes(corpora)
    )

    assert evidence["ready"] is True
    assert evidence["accepted"] is True
    assert evidence["reportHash"] == acceptance.report_hash


def test_gfm_preflight_rejects_an_externally_forged_acceptance_row(
    tmp_path: Path,
) -> None:
    acceptance, corpora = _build_registered_acceptance(tmp_path)
    forged_values = acceptance.model_dump(
        mode="python",
        by_alias=True,
        exclude={"report_hash", "created_at"},
    )
    forged_values["metricSummary"] = {
        **forged_values["metricSummary"],
        "forged": {"score": 1.0},
    }
    forged = GfmAcceptanceManifest.create(**forged_values)
    database = tmp_path / "registry" / "gfm-registry.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE gfm_acceptances
            SET report_hash=?, manifest_json=?
            WHERE report_hash=?
            """,
            (forged.report_hash, canonical_json(forged), acceptance.report_hash),
        )

    evidence = _gfm_acceptance_evidence(
        tmp_path, checked_domain_hashes=_checked_hashes(corpora)
    )

    assert evidence["ready"] is False
    assert "safely recomputed" in evidence["reasons"][0]


def test_gfm_preflight_rejects_forged_metrics_not_bound_to_evidence(
    tmp_path: Path,
) -> None:
    acceptance, corpora = _build_registered_acceptance(tmp_path)
    database = tmp_path / "registry" / "gfm-registry.sqlite3"
    report_id = f"product-{COLLABORATION_TASK}-1"
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT manifest_json FROM gfm_evaluations WHERE report_id=?",
            (report_id,),
        ).fetchone()
        assert row is not None
        original = GfmEvaluationReport.model_validate_json(row["manifest_json"])
        forged_values = original.model_dump(
            mode="python",
            by_alias=True,
            exclude={"report_hash", "created_at"},
        )
        forged_values["metrics"] = {
            **forged_values["metrics"],
            "ndcg@20": 0.99,
        }
        forged_report = GfmEvaluationReport.create(**forged_values)
        connection.execute(
            """
            UPDATE gfm_evaluations SET report_hash=?, manifest_json=?
            WHERE report_id=?
            """,
            (forged_report.report_hash, canonical_json(forged_report), report_id),
        )
        evaluation_rows = connection.execute(
            "SELECT manifest_json FROM gfm_evaluations ORDER BY report_id"
        ).fetchall()
        evaluations = tuple(
            GfmEvaluationReport.model_validate_json(item["manifest_json"])
            for item in evaluation_rows
        )
        forged_acceptance = build_gfm_acceptance(
            experiment_id=acceptance.experiment_id,
            checkpoint_id=acceptance.checkpoint_id,
            corpora=corpora,
            evaluations=evaluations,
            config_hash=CONFIG_HASH,
            code_hash=CODE_HASH,
            environment_hash=ENVIRONMENT_HASH,
        )
        assert forged_acceptance.accepted
        connection.execute(
            """
            UPDATE gfm_acceptances
            SET report_hash=?, manifest_json=?
            WHERE report_hash=?
            """,
            (
                forged_acceptance.report_hash,
                canonical_json(forged_acceptance),
                acceptance.report_hash,
            ),
        )

    evidence = _gfm_acceptance_evidence(
        tmp_path, checked_domain_hashes=_checked_hashes(corpora)
    )

    assert evidence["ready"] is False
    assert "metrics are not bound" in evidence["reasons"][0]


def test_gfm_preflight_rejects_stale_corpus_acceptance_after_artifact_change(
    tmp_path: Path,
) -> None:
    _, corpora = _build_registered_acceptance(tmp_path)
    changed = _corpus(tmp_path, DOMAINS[0], 2)
    manifest_path = (
        tmp_path / "datasets" / "manifests" / "gfm" / f"{DOMAINS[0]}.json"
    )
    manifest_path.write_text(canonical_json(changed), encoding="utf-8")
    checked = _checked_hashes(corpora)
    checked[DOMAINS[0]] = changed.content_hash

    evidence = _gfm_acceptance_evidence(tmp_path, checked_domain_hashes=checked)

    assert evidence["ready"] is False
    assert "checked contract artifacts" in evidence["reasons"][0]
