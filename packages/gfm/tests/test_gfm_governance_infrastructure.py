from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.errors import CheckpointIntegrityError, RegistrationRejected
from socialgraph_gfm.gfm.acceptance import build_gfm_acceptance
from socialgraph_gfm.gfm.checkpoint import load_gfm_checkpoint, save_gfm_checkpoint
from socialgraph_gfm.gfm.configuration import load_core_config
from socialgraph_gfm.gfm.contracts import (
    CollaborationRerankComponents,
    GfmAcceptanceManifest,
    GFM_ACCEPTANCE_GATES,
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
    GfmPretrainConfig,
    GfmRunManifest,
    GfmTaskProtocolManifest,
)
from socialgraph_gfm.gfm.product import (
    build_governance_case_artifact,
    collaboration_rerank_score,
    rerank_collaboration_candidates,
)
from socialgraph_gfm.gfm.registry import GfmRegistry

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
CODE_HASH = "d" * 64
ENVIRONMENT_HASH = "e" * 64


def _corpus(domain: str, suffix: str, path: str) -> GfmDomainCorpusManifest:
    return GfmDomainCorpusManifest.create(
        corpusId=f"corpus-{suffix}",
        domainId=domain,
        datasetName=f"dataset-{suffix}",
        datasetVersion="1",
        datasetRole="pretraining",
        licenseId="CC0-1.0",
        licenseEvidenceHash=HASH_A,
        sourceHash=HASH_B,
        contentHash=canonical_sha256({"domain": domain}),
        splitHash=HASH_C,
        nodeCount=10,
        edgeCount=20,
        featureModalities=("numeric", "temporal"),
        taskIds=("governance.collaboration_recommendation",),
        pointInTimeSafe=True,
        publicCheckpointEligible=True,
        temporalCutoff=datetime(2025, 1, 1, tzinfo=UTC),
        artifactPath=path,
    )


def _evaluation(
    *,
    report_id: str,
    kind: str,
    seed: int,
    domain: str = "academic",
    held_out: str | None = None,
    task: str | None = None,
    metrics: dict[str, float] | None = None,
    ece: float | None = None,
    fresh: bool = False,
    checkpoint: str | None = None,
    run: str | None = None,
) -> GfmEvaluationReport:
    values: dict[str, object] = {
        "reportId": report_id,
        "experimentId": "experiment-1",
        "runId": run or f"run-{seed}",
        "checkpointId": checkpoint or f"checkpoint-{seed}",
        "evaluationKind": kind,
        "domainId": domain,
        "heldOutDomain": held_out,
        "taskId": task,
        "evaluatorCodeHash": (
            CODE_HASH if kind in {"product", "calibration"} else None
        ),
        "evaluatorEnvironmentHash": (
            ENVIRONMENT_HASH if kind in {"product", "calibration"} else None
        ),
        "seed": seed,
        "metrics": {
            **(metrics or {"score": 0.2}),
            **({"brier": 0.1} if kind == "calibration" else {}),
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
        "evidenceArtifactHash": "8" * 64,
        "evidenceArtifactPath": "E:/runtime/reports/evidence.json",
        "baselineDefinitionHash": "7" * 64 if kind == "product" else None,
        "strataDefinitionHash": "6" * 64 if kind == "calibration" else None,
        "ece": ece,
        "brier": 0.1 if ece is not None else None,
        "peakCudaMemoryMiB": 512.0,
        "leakageAuditPassed": True,
        "leakageAuditHash": "9" * 64,
        "leakageAuditPath": "E:/runtime/reports/leakage-audit.json",
        "freshProcessVerified": fresh,
        "verificationDigest": canonical_sha256({"fresh": seed}) if fresh else None,
    }
    return GfmEvaluationReport.create(**values)


def _clone_evaluation(
    report: GfmEvaluationReport, *, report_id: str, **updates: object
) -> GfmEvaluationReport:
    values = report.model_dump(
        mode="python",
        by_alias=True,
        exclude={"report_hash", "created_at"},
    )
    values.update(updates)
    values["reportId"] = report_id
    return GfmEvaluationReport.create(**values)


def _acceptance_evidence() -> tuple[
    tuple[GfmDomainCorpusManifest, ...], tuple[GfmEvaluationReport, ...]
]:
    corpora = (
        _corpus("academic", "a", "E:/runtime/a.zip"),
        _corpus("software", "b", "E:/runtime/b.zip"),
        _corpus("community", "c", "E:/runtime/c.zip"),
    )
    reports: list[GfmEvaluationReport] = []
    for domain in ("academic", "software", "community"):
        for seed in (1, 2, 3):
            reports.append(
                _evaluation(
                    report_id=f"lodo-{domain}-{seed}",
                    kind="lodo",
                    seed=seed,
                    held_out=domain,
                    domain=domain,
                    metrics={
                        "few_shot_1_gfm": 0.42,
                        "few_shot_1_random_init": 0.38,
                        "few_shot_1_single_domain": 0.39,
                        "few_shot_5_gfm": 0.60,
                        "few_shot_5_random_init": 0.54,
                        "few_shot_5_single_domain": 0.56,
                        "few_shot_10_gfm": 0.66,
                        "few_shot_10_random_init": 0.58,
                        "few_shot_10_single_domain": 0.61,
                    },
                )
            )
    product_metrics = {
        "governance.collaboration_recommendation": {
            "ndcg@20": 0.63,
            "baseline_ndcg@20": 0.58,
            "recall@20": 0.55,
            "baseline_recall@20": 0.50,
            "bootstrap_ci95_ndcg_gain_lower": 0.01,
            "query_count": 100.0,
        },
        "core.newcomer_support": {
            "ndcg@20": 0.62,
            "baseline_ndcg@20": 0.57,
            "auprc": 0.25,
            "label_prevalence": 0.15,
            "query_count": 100.0,
            "outcome_count": 100.0,
        },
    }
    for task, metrics in product_metrics.items():
        for seed in (1, 2, 3):
            reports.append(
                _evaluation(
                    report_id=f"product-{task}-{seed}",
                    kind="product",
                    seed=seed,
                    task=task,
                    metrics=metrics,
                )
            )
            reports.append(
                _evaluation(
                    report_id=f"calibration-{task}-{seed}",
                    kind="calibration",
                    seed=seed,
                    task=task,
                    metrics={
                        "ece": 0.02,
                        "brier": 0.1,
                        "strata_complete": 1.0,
                        "ece_institution_small": 0.02,
                        "ece_institution_medium": 0.02,
                        "ece_institution_large": 0.02,
                        "ece_topic_cluster_0": 0.02,
                        "ece_topic_cluster_1": 0.02,
                        "ece_topic_cluster_2": 0.02,
                        **(
                            {"ece_first_time": 0.02, "ece_repeated": 0.02}
                            if task == "governance.collaboration_recommendation"
                            else {"ece_newcomer": 0.02}
                        ),
                    },
                    ece=0.02,
                )
            )
    for seed in (1, 2, 3):
        reports.append(
            _evaluation(
                report_id=f"fresh-{seed}",
                kind="fresh_process",
                seed=seed,
                domain="multi-domain",
                fresh=True,
            )
        )
    return corpora, tuple(reports)


def test_strict_contracts_build_hashes_and_ignore_absolute_storage_paths():
    first = _corpus("academic", "same", "C:/one/corpus.zip")
    second = _corpus("academic", "same", "E:/two/corpus.zip")
    assert first.logical_hash == second.logical_hash
    with pytest.raises(ValidationError, match="Extra inputs"):
        GfmDomainCorpusManifest.model_validate(
            {**first.model_dump(by_alias=True), "unexpected": True}
        )
    with pytest.raises(ValueError, match="derived"):
        GfmDomainCorpusManifest.create(
            **first.model_dump(by_alias=True, exclude={"created_at"})
        )


def test_checked_pretrain_config_matches_the_public_contract():
    checked = GfmPretrainConfig.model_validate(load_core_config())
    assert checked.architecture.hidden_channels == 128
    assert checked.architecture.time_channels == 32
    assert checked.architecture.relation_bases == 8
    assert checked.optimization.optimizer == "adamw"
    assert checked.dev.max_steps == 2000
    assert checked.formal.max_steps == 30000


def test_checkpoint_round_trip_is_weights_only_and_detects_tampering(tmp_path: Path):
    torch = pytest.importorskip("torch")
    manifest = save_gfm_checkpoint(
        tmp_path,
        checkpoint_id="checkpoint-1",
        run_id="run-1",
        epoch=2,
        step=100,
        components={"encoder": {"weight": torch.tensor([1.0, 2.0])}},
        optimizer_state={"state": {}, "param_groups": []},
        scheduler_state={"last_epoch": 2},
        scaler_state={"scale": 1024.0},
        sampler_state={"epoch": 2, "cursor": 17},
        best_state={"metric": 0.42, "step": 100},
        config={"model": "core-base"},
        corpus_hashes=(HASH_A, HASH_B, HASH_C),
    )
    loaded = load_gfm_checkpoint(manifest)
    assert loaded["best_state"]["metric"] == pytest.approx(0.42)
    assert set(loaded["components"]) == {"encoder"}

    moved = manifest.model_dump(by_alias=True)
    moved["artifactPath"] = "E:/a/different/physical/location.pt"
    assert type(manifest).model_validate(moved).logical_hash == manifest.logical_hash

    with Path(manifest.artifact_path).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(CheckpointIntegrityError, match="SHA-256 mismatch"):
        load_gfm_checkpoint(manifest)


def test_acceptance_requires_every_hard_gate_and_fails_closed():
    corpora, reports = _acceptance_evidence()
    accepted = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=reports,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert accepted.accepted
    assert all(accepted.gates.values())

    bad_report = _evaluation(
        report_id="calibration-bad",
        kind="calibration",
        seed=1,
        task="governance.collaboration_recommendation",
        metrics={"ece": 0.2, "brier": 0.1},
        ece=0.2,
    )
    without_good_calibration = tuple(
        report
        for report in reports
        if not (
            report.evaluation_kind == "calibration"
            and report.task_id == "governance.collaboration_recommendation"
        )
    ) + (bad_report,)
    rejected = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=without_good_calibration,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert not rejected.accepted
    assert not rejected.gates["calibration_ece"]

    unbound = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-without-direct-evidence",
        corpora=corpora,
        evaluations=reports,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert not unbound.accepted
    assert not unbound.gates["product_metrics"]
    assert not unbound.gates["calibration_ece"]
    assert not unbound.gates["fresh_process_verification"]


def test_acceptance_rejects_a_non_public_checkpoint_corpus():
    corpora, reports = _acceptance_evidence()
    blocked_values = corpora[0].model_dump(
        mode="python",
        by_alias=True,
        exclude={"logical_hash", "created_at"},
    )
    blocked_values["publicCheckpointEligible"] = False
    blocked = GfmDomainCorpusManifest.create(**blocked_values)

    acceptance = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=(blocked, *corpora[1:]),
        evaluations=reports,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )

    assert acceptance.accepted is False
    assert acceptance.gates["three_domains"] is False


def test_acceptance_rejects_duplicate_semantic_seed_weighting():
    corpora, reports = _acceptance_evidence()
    original = next(
        report
        for report in reports
        if report.evaluation_kind == "product"
        and report.task_id == "governance.collaboration_recommendation"
        and report.seed == 1
    )
    values = original.model_dump(
        mode="python",
        by_alias=True,
        exclude={"report_hash", "created_at"},
    )
    values["reportId"] = "duplicate-product-seed-1"
    duplicate = GfmEvaluationReport.create(**values)
    with pytest.raises(ValueError, match="duplicate semantic evaluation evidence"):
        build_gfm_acceptance(
            experiment_id="experiment-1",
            checkpoint_id="checkpoint-1",
            corpora=corpora,
            evaluations=(*reports, duplicate),
            config_hash=HASH_A,
            code_hash=CODE_HASH,
            environment_hash=ENVIRONMENT_HASH,
        )

    duplicate_fresh = next(
        report for report in reports if report.evaluation_kind == "fresh_process"
    )
    with pytest.raises(ValueError, match="duplicate immutable evaluation"):
        build_gfm_acceptance(
            experiment_id="experiment-1",
            checkpoint_id="checkpoint-1",
            corpora=corpora,
            evaluations=(*reports, duplicate_fresh),
            config_hash=HASH_A,
            code_hash=CODE_HASH,
            environment_hash=ENVIRONMENT_HASH,
        )


@pytest.mark.parametrize(
    ("kind", "updates"),
    [
        ("product", {"domainId": "different-product-domain"}),
        (
            "lodo",
            {"taskId": "governance.collaboration_recommendation"},
        ),
    ],
)
def test_semantic_seed_deduplication_cannot_be_bypassed_by_irrelevant_fields(
    kind: str, updates: dict[str, object]
):
    corpora, reports = _acceptance_evidence()
    original = next(report for report in reports if report.evaluation_kind == kind)
    duplicate = _clone_evaluation(
        original,
        report_id=f"semantic-alias-{kind}",
        **updates,
    )
    with pytest.raises(ValueError, match="duplicate semantic evaluation evidence"):
        build_gfm_acceptance(
            experiment_id="experiment-1",
            checkpoint_id="checkpoint-1",
            corpora=corpora,
            evaluations=(*reports, duplicate),
            config_hash=HASH_A,
            code_hash=CODE_HASH,
            environment_hash=ENVIRONMENT_HASH,
        )


def test_acceptance_requires_exactly_the_same_three_seed_matrix():
    corpora, reports = _acceptance_evidence()
    added: list[GfmEvaluationReport] = []
    for report in reports:
        if report.seed != 3:
            continue
        updates: dict[str, object] = {
            "seed": 4,
            "runId": "run-4",
            "checkpointId": "checkpoint-4",
        }
        if report.evaluation_kind == "fresh_process":
            updates["verificationDigest"] = canonical_sha256({"fresh": 4})
        added.append(
            _clone_evaluation(
                report,
                report_id=f"fourth-seed-{report.report_id}",
                **updates,
            )
        )
    rejected = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=(*reports, *added),
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert not rejected.gates["lodo_complete"]
    assert not rejected.gates["product_metrics"]
    assert not rejected.gates["calibration_ece"]


def test_product_and_calibration_seed_evidence_must_share_one_source():
    corpora, reports = _acceptance_evidence()
    original = next(
        report
        for report in reports
        if report.evaluation_kind == "calibration"
        and report.task_id == "governance.collaboration_recommendation"
        and report.seed == 2
    )
    mismatched = _clone_evaluation(
        original,
        report_id="calibration-mismatched-source",
        runId="different-run-2",
        checkpointId="different-checkpoint-2",
    )
    replaced = tuple(report for report in reports if report is not original) + (
        mismatched,
    )
    rejected = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=replaced,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert not rejected.gates["product_metrics"]
    assert not rejected.gates["calibration_ece"]


def test_delivery_inventory_is_exact_and_source_bound():
    corpora, reports = _acceptance_evidence()
    direct = tuple(
        report.report_hash
        for report in reports
        if report.checkpoint_id == "checkpoint-1"
        and report.evaluation_kind in {"product", "calibration", "fresh_process"}
    )
    assert len(direct) == 5
    extra = next(report.report_hash for report in reports if report.evaluation_kind == "lodo")
    oversized = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=reports,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
        delivery_evidence_report_hashes=(*direct, extra),
    )
    assert not oversized.gates["product_metrics"]
    assert not oversized.gates["calibration_ece"]
    assert not oversized.gates["fresh_process_verification"]

    collab_calibration_seed_2 = next(
        report.report_hash
        for report in reports
        if report.evaluation_kind == "calibration"
        and report.task_id == "governance.collaboration_recommendation"
        and report.seed == 2
    )
    collab_calibration_seed_1 = next(
        report.report_hash
        for report in reports
        if report.evaluation_kind == "calibration"
        and report.task_id == "governance.collaboration_recommendation"
        and report.seed == 1
    )
    wrong_source = tuple(
        collab_calibration_seed_2 if value == collab_calibration_seed_1 else value
        for value in direct
    )
    unbound = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=reports,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
        delivery_evidence_report_hashes=wrong_source,
    )
    assert not unbound.gates["product_metrics"]
    assert not unbound.gates["calibration_ece"]
    assert not unbound.gates["fresh_process_verification"]


def test_lodo_zero_controls_and_missing_product_fresh_evidence_fail_closed():
    corpora, reports = _acceptance_evidence()
    rewritten: list[GfmEvaluationReport] = []
    for report in reports:
        if report.evaluation_kind != "lodo":
            rewritten.append(report)
            continue
        metrics = dict(report.metrics)
        metrics["few_shot_5_random_init"] = 0.0
        metrics["few_shot_5_single_domain"] = 0.0
        rewritten.append(
            _clone_evaluation(
                report,
                report_id=f"zero-control-{report.report_id}",
                metrics=metrics,
            )
        )
    zero_control = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=tuple(rewritten),
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert not zero_control.gates["lodo_complete"]

    missing_fresh = tuple(
        report
        for report in reports
        if not (report.evaluation_kind == "fresh_process" and report.seed == 2)
    )
    fresh_rejected = build_gfm_acceptance(
        experiment_id="experiment-1",
        checkpoint_id="checkpoint-1",
        corpora=corpora,
        evaluations=missing_fresh,
        config_hash=HASH_A,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )
    assert not fresh_rejected.gates["fresh_process_verification"]


def test_evaluation_contract_binds_calibration_metrics_and_fresh_repeat():
    calibration = _evaluation(
        report_id="valid-calibration",
        kind="calibration",
        seed=1,
        task="governance.collaboration_recommendation",
        metrics={"ece": 0.02},
        ece=0.02,
    )
    values = calibration.model_dump(
        mode="python", by_alias=True, exclude={"report_hash", "created_at"}
    )
    values["metrics"] = {**calibration.metrics, "ece": 0.03}
    with pytest.raises(ValidationError, match="metric ECE"):
        GfmEvaluationReport.create(**values)

    values = calibration.model_dump(
        mode="python", by_alias=True, exclude={"report_hash", "created_at"}
    )
    values["metrics"] = {**calibration.metrics, "brier": 0.2}
    with pytest.raises(ValidationError, match="metric Brier"):
        GfmEvaluationReport.create(**values)

    fresh = _evaluation(
        report_id="valid-fresh", kind="fresh_process", seed=1, fresh=True
    )
    values = fresh.model_dump(
        mode="python", by_alias=True, exclude={"report_hash", "created_at"}
    )
    values["metrics"] = {**fresh.metrics, "fresh_process_repeat_match": 0.0}
    with pytest.raises(ValidationError, match="repeated-process match"):
        GfmEvaluationReport.create(**values)


def test_accepted_contract_cannot_be_forged_without_evidence():
    with pytest.raises(ValidationError, match="complete evaluation"):
        GfmAcceptanceManifest.create(
            experimentId="forged",
            checkpointId="unchecked",
            accepted=True,
            domainIds=("academic", "software", "community"),
            lodoDomains=("academic", "software", "community"),
            productTaskIds=(
                "governance.collaboration_recommendation",
                "core.newcomer_support",
            ),
            corpusHashes=(HASH_A, HASH_B, HASH_C),
            configHash=HASH_A,
            codeHash=CODE_HASH,
            environmentHash=ENVIRONMENT_HASH,
            evaluationReportHashes=(),
            deliveryEvidenceReportHashes=(),
            maximumEce=0.0,
            peakCudaMemoryMiB=1.0,
            freshProcessDigests=(),
            metricSummary={},
            gates={gate: True for gate in GFM_ACCEPTANCE_GATES},
            reasons=(),
        )


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ({"ndcg@20": 1.01}, r"ndcg@20.*\[0, 1\]"),
        ({"ece_topic_cluster_0": -0.01}, r"ece_topic_cluster_0.*\[0, 1\]"),
        ({"query_count": 1.5}, "non-negative integer"),
        ({"bootstrap_ci95_ndcg_gain_lower": 2.0}, r"\[-1, 1\]"),
    ],
)
def test_evaluation_contract_rejects_impossible_metric_evidence(
    metrics: dict[str, float], message: str
):
    with pytest.raises(ValidationError, match=message):
        _evaluation(
            report_id="invalid-metric",
            kind="product",
            seed=1,
            task="governance.collaboration_recommendation",
            metrics=metrics,
        )


def test_product_evidence_is_limited_to_fixed_v1_tasks():
    with pytest.raises(ValidationError, match="two fixed tasks"):
        _evaluation(
            report_id="out-of-scope-product",
            kind="product",
            seed=1,
            task="governance.conversation_escalation_watch",
        )


def test_evaluation_evidence_paths_are_portable_but_definition_hashes_are_required():
    report = _evaluation(
        report_id="portable-product",
        kind="product",
        seed=1,
        task="governance.collaboration_recommendation",
    )
    moved = report.model_dump(by_alias=True)
    moved["evidenceArtifactPath"] = "C:/relocated/evidence.json"
    moved["leakageAuditPath"] = "C:/relocated/audit.json"
    assert GfmEvaluationReport.model_validate(moved).report_hash == report.report_hash

    invalid = report.model_dump(by_alias=True)
    invalid["baselineDefinitionHash"] = None
    with pytest.raises(ValidationError, match="baseline definition"):
        GfmEvaluationReport.model_validate(invalid)


def test_product_rerank_and_immutable_case_language_policy():
    components = CollaborationRerankComponents(
        calibratedProbability=0.8,
        topicComplementarity=0.6,
        bridgeGain=0.4,
        institutionDiversity=0.2,
    )
    assert collaboration_rerank_score(components) == pytest.approx(0.70)
    ranked = rerank_collaboration_candidates(
        [
            {"candidateId": "b", "components": components},
            {
                "candidateId": "a",
                "components": {
                    "calibratedProbability": 0.9,
                    "topicComplementarity": 0.6,
                    "bridgeGain": 0.4,
                    "institutionDiversity": 0.2,
                },
            },
        ]
    )
    assert [item.candidate_id for item in ranked] == ["a", "b"]

    values = {
        "caseId": "case-1",
        "taskId": "governance.collaboration_recommendation",
        "graphVersionId": "graph-1",
        "graphFactHash": HASH_A,
        "inferenceCutoff": datetime(2025, 1, 1, tzinfo=UTC),
        "modelId": "model-1",
        "modelVersion": "1",
        "checkpointId": "checkpoint-1",
        "checkpointHash": HASH_B,
        "runId": "run-1",
        "target": {
            "kind": "candidate_relation",
            "primaryId": "person-a",
            "secondaryId": "person-b",
        },
        "score": components.weighted_score(),
        "uncertainty": 0.1,
        "rerankComponents": components,
        "reasonCodes": ("shared-topic", "bridge-opportunity"),
        "evidence": (
            {"kind": "path", "refs": ("person-a", "topic-x", "person-b"), "summary": "共同主题路径"},
        ),
        "counterfactuals": {"withoutSharedTopic": 0.5},
        "recommendedActions": ("邀请人工复核并发起合作讨论",),
        "dataSufficiency": "sufficient",
        "featureHash": HASH_C,
        "corpusHashes": (HASH_A,),
    }
    artifact = build_governance_case_artifact(**values)
    assert artifact.human_review_status == "pending"
    with pytest.raises(ValidationError, match="frozen"):
        artifact.score = 0.1  # type: ignore[misc]
    with pytest.raises(ValidationError, match="churn or low-value"):
        build_governance_case_artifact(
            **{**values, "recommendedActions": ("把这个成员标记为低价值",)}
        )


def test_corpus_reregistration_uses_portable_logical_identity(tmp_path: Path) -> None:
    registry = GfmRegistry(tmp_path / "registry.sqlite3")
    first = _corpus("academic", "portable", "E:/first/manifest.json")
    second_values = first.model_dump(
        mode="python",
        by_alias=True,
        exclude={"logical_hash", "created_at", "artifact_path"},
    )
    second = GfmDomainCorpusManifest.create(
        **second_values,
        artifactPath="E:/reprepared/manifest.json",
        createdAt=first.created_at + timedelta(seconds=1),
    )
    assert first.logical_hash == second.logical_hash
    assert first.created_at != second.created_at

    registry.record_corpus(first)
    registry.record_corpus(second)
    with registry.connect() as connection:
        stored = connection.execute(
            "SELECT manifest_json FROM gfm_domain_corpora WHERE corpus_id=?",
            (first.corpus_id,),
        ).fetchone()
    assert GfmDomainCorpusManifest.model_validate_json(stored[0]) == first

    changed_values = dict(second_values)
    changed_values["nodeCount"] = first.node_count + 1
    changed = GfmDomainCorpusManifest.create(
        **changed_values,
        artifactPath="E:/changed/manifest.json",
    )
    with pytest.raises(RegistrationRejected, match="different logical content"):
        registry.record_corpus(changed)


def test_gfm_registry_is_wal_and_blocks_promotion_before_acceptance(tmp_path: Path):
    torch = pytest.importorskip("torch")
    registry = GfmRegistry(tmp_path / "registry.sqlite3")
    corpora, reports = _acceptance_evidence()
    for corpus in corpora:
        registry.record_corpus(corpus)
    protocol = GfmTaskProtocolManifest.create(
        protocolId="protocol-1",
        taskId="governance.community_pulse_forecast",
        taskFamily="community_forecast",
        domainIds=tuple(corpus.domain_id for corpus in corpora),
        splitStrategy="temporal",
        objectives=("temporal_ranking",),
        primaryMetrics=("ndcg@10",),
    )
    registry.record_protocol(protocol)
    config = {"model": "core-base"}
    config_hash = canonical_sha256(config)
    now = datetime.now(UTC)
    for seed in (1, 2, 3):
        run = GfmRunManifest.create(
            runId=f"run-{seed}",
            experimentId="experiment-1",
            phase="pretrain",
            architectureVariant="core-base",
            status="succeeded",
            domainIds=tuple(corpus.domain_id for corpus in corpora),
            seed=seed,
            codeHash=CODE_HASH,
            environmentHash=ENVIRONMENT_HASH,
            configHash=config_hash,
            corpusHashes=tuple(corpus.logical_hash for corpus in corpora),
            taskProtocolHashes=(protocol.protocol_hash,),
            startedAt=now,
            finishedAt=now,
            peakCudaMemoryMiB=512.0,
        )
        registry.record_run(run)
        checkpoint = save_gfm_checkpoint(
            tmp_path / "checkpoints",
            checkpoint_id=f"checkpoint-{seed}",
            run_id=f"run-{seed}",
            epoch=1,
            step=1,
            components={"encoder": {"weight": torch.ones(2)}},
            optimizer_state={"state": {}, "param_groups": []},
            scheduler_state=None,
            scaler_state=None,
            sampler_state={"cursor": 1},
            best_state={"metric": 1.0},
            config=config,
            corpus_hashes=tuple(corpus.logical_hash for corpus in corpora),
        )
        registry.record_checkpoint(checkpoint)
    with pytest.raises(RegistrationRejected, match="before acceptance"):
        registry.promote_model(model_id="model-1", experiment_id="experiment-1")
    assert registry.counts()["gfm_models"] == 0
    # Contract-shaped rows with invented hashes/paths are no longer registry
    # evidence.  Full accepted promotion is exercised by the preflight E2E
    # fixture, which persists and re-hashes every physical evidence artifact.
    with pytest.raises(RegistrationRejected, match="outside the runtime report root"):
        registry.record_evaluation(reports[0])
    assert registry.counts()["gfm_evaluations"] == 0
    with registry.connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
