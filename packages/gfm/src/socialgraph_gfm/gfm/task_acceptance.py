"""Non-promotable acceptance for collaboration while newcomer work is deferred."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Any, Final

from ..canonical import canonical_sha256
from .contracts import (
    GFM_COLLABORATION_TASK_ACCEPTANCE_GATES,
    GfmCheckpointManifest,
    GfmEvaluationReport,
    GfmRunManifest,
    GfmTaskAcceptanceManifest,
    GfmTaskProtocolManifest,
)

COLLABORATION_TASK: Final = "governance.collaboration_recommendation"
COLLABORATION_PROTOCOL_ID: Final = "socialgraph-fm-collaboration"
COLLABORATION_PROTOCOL_DOMAINS: Final = (
    "openalex-graph-ai",
    "thgl-software-2.0.0",
    "wikimedia-talk-article-2011-2015",
)
COLLABORATION_PROTOCOL_OBJECTIVES: Final = (
    "label=future-12-month-first-collaboration;repeat-collaboration-is-separate-auxiliary-and-report",
    "train-cutoffs=2017,2018,2019,2020,2021;validation-cutoff=2022->2023;test-cutoff=2023->2024;shadow=2024->2025",
    "candidate-set=one-or-more-horizon-positives-plus-at-least-99-cutoff-safe-negatives-per-query;exclude-all-horizon-positives-from-negatives",
    "baseline=best-of-adamic-adar,resource-allocation,and-cutoff-feature-mlp;structural-cn-aa-ra-fit-on-train-cutoffs-only",
)
FORMAL_SEEDS: Final = (20260821, 20260822, 20260823)
ZERO_AUDIT_METRICS: Final = (
    "future_edge_access_count",
    "cutoff_violation_count",
    "split_overlap_count",
)
REQUIRED_CALIBRATION_STRATA: Final = (
    "ece_institution_small",
    "ece_institution_medium",
    "ece_institution_large",
    "ece_topic_cluster_0",
    "ece_topic_cluster_1",
    "ece_topic_cluster_2",
    "ece_first_time",
    "ece_repeated",
)


def collaboration_protocol() -> GfmTaskProtocolManifest:
    """Return the one exact protocol that may authorize task acceptance."""

    return GfmTaskProtocolManifest.create(
        protocolId=COLLABORATION_PROTOCOL_ID,
        taskId=COLLABORATION_TASK,
        taskFamily="collaboration_ranking",
        domainIds=COLLABORATION_PROTOCOL_DOMAINS,
        splitStrategy="temporal",
        objectives=COLLABORATION_PROTOCOL_OBJECTIVES,
        primaryMetrics=("ndcg@20", "recall@20"),
    )


COLLABORATION_PROTOCOL_HASH: Final = collaboration_protocol().protocol_hash


def _relative_gain(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), 1e-12)


def _one_by_seed(
    reports: Sequence[GfmEvaluationReport], kind: str
) -> dict[int, GfmEvaluationReport]:
    selected = [
        report
        for report in reports
        if report.evaluation_kind == kind
        and report.task_id == COLLABORATION_TASK
        and "shadow" not in report.warnings
    ]
    by_seed = {report.seed: report for report in selected}
    if len(by_seed) != len(selected):
        raise ValueError(f"duplicate collaboration {kind} evidence for one formal seed")
    return by_seed


def build_collaboration_task_acceptance(
    *,
    experiment_id: str,
    runs: Sequence[GfmRunManifest],
    checkpoints: Sequence[GfmCheckpointManifest],
    evaluations: Sequence[GfmEvaluationReport],
    protocol: GfmTaskProtocolManifest,
    test_read_states: Mapping[str, Mapping[str, Any]],
    backbone_bindings: Mapping[str, Mapping[str, Any]],
    accepted_pretraining_checkpoint_ids: Sequence[str],
    pretraining_acceptance_report_hash: str,
    accepted_pretraining_variant: str,
    selected_variant: str | None = None,
) -> GfmTaskAcceptanceManifest:
    """Derive the fixed three-seed collaboration gate from immutable evidence.

    The builder intentionally has no newcomer input and cannot construct the
    full model acceptance contract.
    """

    checked_runs = tuple(GfmRunManifest.model_validate(value) for value in runs)
    checked_checkpoints = tuple(
        GfmCheckpointManifest.model_validate(value) for value in checkpoints
    )
    checked_reports = tuple(GfmEvaluationReport.model_validate(value) for value in evaluations)
    checked_protocol = GfmTaskProtocolManifest.model_validate(protocol)
    expected_protocol = collaboration_protocol()
    if (
        checked_protocol.protocol_id != COLLABORATION_PROTOCOL_ID
        or checked_protocol.protocol_hash != COLLABORATION_PROTOCOL_HASH
        or checked_protocol.logical_payload() != expected_protocol.logical_payload()
    ):
        raise ValueError("collaboration task acceptance requires its exact collaboration protocol")

    product_by_seed = _one_by_seed(checked_reports, "product")
    calibration_by_seed = _one_by_seed(checked_reports, "calibration")
    product_seeds = tuple(sorted(product_by_seed))
    candidate_runs = {
        run.run_id: run
        for run in checked_runs
        if run.phase == "adapt" and run.experiment_id == experiment_id
    }
    candidate_checkpoints = {
        checkpoint.checkpoint_id: checkpoint
        for checkpoint in checked_checkpoints
        if checkpoint.run_id in candidate_runs
    }
    selected_run_ids = tuple(
        product_by_seed[seed].run_id for seed in product_seeds
    )
    selected_checkpoint_ids = tuple(
        product_by_seed[seed].checkpoint_id for seed in product_seeds
    )
    selected_runs = [candidate_runs.get(run_id) for run_id in selected_run_ids]
    selected_checkpoints = [
        candidate_checkpoints.get(checkpoint_id) for checkpoint_id in selected_checkpoint_ids
    ]
    variants = {
        run.architecture_variant for run in selected_runs if run is not None
    }
    architecture_variant = (
        selected_variant
        if selected_variant in {"core-base", "core-moe"}
        else next(iter(sorted(variants)), "core-base")
    )
    run_values = [run for run in selected_runs if run is not None]
    checkpoint_values = [value for value in selected_checkpoints if value is not None]
    config_hashes = {run.config_hash for run in run_values}
    code_hashes = {run.code_hash for run in run_values}
    environment_hashes = {run.environment_hash for run in run_values}
    corpus_sets = {tuple(sorted(run.corpus_hashes)) for run in run_values}
    corpus_hashes = next(iter(corpus_sets), ())
    config_hash = next(iter(config_hashes), "0" * 64)
    code_hash = next(iter(code_hashes), "0" * 64)
    environment_hash = next(iter(environment_hashes), "0" * 64)
    checked_backbone_ids = tuple(
        str(backbone_bindings.get(checkpoint_id, {}).get("checkpointId", ""))
        for checkpoint_id in selected_checkpoint_ids
    )
    checked_backbone_state_hashes = tuple(
        str(backbone_bindings.get(checkpoint_id, {}).get("stateHash", ""))
        for checkpoint_id in selected_checkpoint_ids
    )

    gates = {name: False for name in GFM_COLLABORATION_TASK_ACCEPTANCE_GATES}
    gates["formal_seed_matrix"] = (
        product_seeds == FORMAL_SEEDS
        and tuple(sorted(calibration_by_seed)) == FORMAL_SEEDS
        and len(selected_run_ids) == 3
        and len(set(selected_run_ids)) == 3
        and len(set(selected_checkpoint_ids)) == 3
    )
    provenance_ok = (
        gates["formal_seed_matrix"]
        and len(run_values) == 3
        and len(checkpoint_values) == 3
        and variants == {architecture_variant}
        and (selected_variant is None or architecture_variant == selected_variant)
        and len(config_hashes) == len(code_hashes) == len(environment_hashes) == 1
        and len(corpus_sets) == 1
        and len(corpus_hashes) == 3
        and all(
            run.status == "succeeded"
            and run.seed == seed
            and run.task_protocol_hashes == (checked_protocol.protocol_hash,)
            and checkpoint is not None
            and checkpoint.run_id == run.run_id
            and checkpoint.config_hash == run.config_hash
            and tuple(sorted(checkpoint.corpus_hashes)) == tuple(sorted(run.corpus_hashes))
            for seed, run, checkpoint in zip(
                product_seeds, selected_runs, selected_checkpoints, strict=True
            )
            if run is not None
        )
        and all(
            calibration_by_seed[seed].run_id == product_by_seed[seed].run_id
            and calibration_by_seed[seed].checkpoint_id
            == product_by_seed[seed].checkpoint_id
            and product_by_seed[seed].evaluator_code_hash == run.code_hash
            and product_by_seed[seed].evaluator_environment_hash
            == run.environment_hash
            and calibration_by_seed[seed].evaluator_code_hash == run.code_hash
            and calibration_by_seed[seed].evaluator_environment_hash
            == run.environment_hash
            for seed, run in zip(product_seeds, selected_runs, strict=True)
            if run is not None
        )
        and len(backbone_bindings) == 3
        and len(set(checked_backbone_ids)) == 3
        and set(checked_backbone_ids) == set(accepted_pretraining_checkpoint_ids)
        and accepted_pretraining_variant == architecture_variant
        and all(
            binding.get("seed") == seed
            and binding.get("architectureVariant") == architecture_variant
            and binding.get("configHash") == run.config_hash
            and binding.get("codeHash") == run.code_hash
            and binding.get("environmentHash") == run.environment_hash
            and tuple(sorted(binding.get("corpusHashes", ())))
            == tuple(sorted(run.corpus_hashes))
            for seed in product_seeds
            for run in (candidate_runs.get(product_by_seed[seed].run_id),)
            for binding in (
                backbone_bindings.get(product_by_seed[seed].checkpoint_id, {}),
            )
            if run is not None
        )
    )
    gates["provenance_binding"] = provenance_ok

    fresh_by_seed: dict[int, GfmEvaluationReport] = {}
    for seed in product_seeds:
        source = product_by_seed[seed]
        matching = [
            report
            for report in checked_reports
            if report.evaluation_kind == "fresh_process"
            and report.seed == seed
            and report.run_id == source.run_id
            and report.checkpoint_id == source.checkpoint_id
        ]
        if len(matching) == 1:
            fresh_by_seed[seed] = matching[0]

    test_read_hashes: list[str] = []
    physical_test_ok = gates["formal_seed_matrix"]
    for seed in product_seeds:
        product = product_by_seed[seed]
        calibration = calibration_by_seed.get(seed)
        state = test_read_states.get(product.checkpoint_id)
        if not isinstance(state, Mapping):
            physical_test_ok = False
            continue
        state_dict = dict(state)
        expected = {
            "schemaVersion": "gfm.product-test-read-state/1.0",
            "experimentId": experiment_id,
            "checkpointId": product.checkpoint_id,
            "task": "collaboration",
            "split": "test",
            "status": "completed",
            "readCount": 1,
            "physicalRoleView": True,
            "maximumRole": "test",
            "productReportHash": product.report_hash,
            "calibrationReportHash": (
                calibration.report_hash if calibration is not None else None
            ),
        }
        if any(state_dict.get(key) != value for key, value in expected.items()):
            physical_test_ok = False
        test_read_hashes.append(canonical_sha256(state_dict))
    gates["physical_test_read_once"] = physical_test_ok and len(test_read_hashes) == 3

    required = (
        "ndcg@20",
        "baseline_ndcg@20",
        "recall@20",
        "baseline_recall@20",
        "bootstrap_ci95_ndcg_gain_lower",
        "query_count",
    )
    product_reports = [product_by_seed[seed] for seed in product_seeds]
    values = {
        metric: [report.metrics[metric] for report in product_reports if metric in report.metrics]
        for metric in required
    }
    product_ok = (
        gates["formal_seed_matrix"]
        and all(len(metric_values) == 3 for metric_values in values.values())
        and all(report.leakage_audit_passed for report in product_reports)
    )
    metric_summary: dict[str, float] = {}
    if product_ok:
        ndcg = fmean(values["ndcg@20"])
        baseline_ndcg = fmean(values["baseline_ndcg@20"])
        recall = fmean(values["recall@20"])
        baseline_recall = fmean(values["baseline_recall@20"])
        ndcg_gain = _relative_gain(ndcg, baseline_ndcg)
        recall_gain = _relative_gain(recall, baseline_recall)
        ci_lower = min(values["bootstrap_ci95_ndcg_gain_lower"])
        metric_summary.update(
            {
                "ndcg@20": ndcg,
                "baseline_ndcg@20": baseline_ndcg,
                "ndcg_relative_gain": ndcg_gain,
                "recall@20": recall,
                "baseline_recall@20": baseline_recall,
                "recall_relative_gain": recall_gain,
                "bootstrap_ci95_ndcg_gain_lower": ci_lower,
                "minimum_query_count": min(values["query_count"]),
            }
        )
        product_ok = (
            baseline_ndcg > 0.0
            and baseline_recall > 0.0
            and ndcg_gain >= 0.05
            and recall_gain >= 0.05
            and ci_lower > 0.0
            and min(values["query_count"]) >= 100.0
        )
    gates["product_metrics"] = product_ok

    calibration_reports = [calibration_by_seed[seed] for seed in product_seeds]
    eces = [report.ece for report in calibration_reports]
    calibration_ok = (
        gates["formal_seed_matrix"]
        and len(eces) == 3
        and all(value is not None and value <= 0.05 for value in eces)
        and all(
            report.metrics.get("strata_complete") == 1.0
            and all(
                report.metrics.get(name) is not None
                and report.metrics[name] <= 0.05
                for name in REQUIRED_CALIBRATION_STRATA
            )
            for report in calibration_reports
        )
    )
    gates["calibration_ece"] = calibration_ok
    if all(value is not None for value in eces):
        metric_summary["maximum_ece"] = max(float(value) for value in eces if value is not None)

    fresh_reports = [fresh_by_seed[seed] for seed in sorted(fresh_by_seed)]
    gates["fresh_process_verification"] = (
        tuple(sorted(fresh_by_seed)) == FORMAL_SEEDS
        and len({report.verification_digest for report in fresh_reports}) == 3
        and all(
            report.fresh_process_verified
            and report.verification_digest is not None
            and report.metrics.get("fresh_process_repeat_match") == 1.0
            for report in fresh_reports
        )
    )
    acceptance_reports = (*product_reports, *calibration_reports, *fresh_reports)
    gates["temporal_leakage_audit"] = (
        len(acceptance_reports) == 9
        and all(report.experiment_id == experiment_id for report in acceptance_reports)
        and all("shadow" not in report.warnings for report in acceptance_reports)
        and all(report.leakage_audit_passed for report in acceptance_reports)
        and all(
            all(report.metrics.get(metric) == 0.0 for metric in ZERO_AUDIT_METRICS)
            for report in acceptance_reports
        )
    )
    memory_values = [
        report.peak_cuda_memory_mib
        for report in acceptance_reports
        if report.peak_cuda_memory_mib is not None
    ]
    gates["cuda_memory"] = (
        len(memory_values) == len(acceptance_reports)
        and bool(memory_values)
        and max(memory_values) < 7168.0
    )
    reasons = tuple(
        f"hard gate failed: {name}" for name in sorted(gates) if not gates[name]
    )
    return GfmTaskAcceptanceManifest.create(
        experimentId=experiment_id,
        taskId=COLLABORATION_TASK,
        accepted=not reasons,
        architectureVariant=architecture_variant,
        formalSeeds=product_seeds,
        runIds=selected_run_ids,
        checkpointIds=selected_checkpoint_ids,
        backboneCheckpointIds=checked_backbone_ids,
        backboneStateHashes=checked_backbone_state_hashes,
        pretrainingAcceptanceReportHash=pretraining_acceptance_report_hash,
        corpusHashes=tuple(corpus_hashes),
        protocolHash=checked_protocol.protocol_hash,
        configHash=config_hash,
        codeHash=code_hash,
        environmentHash=environment_hash,
        productReportHashes=tuple(product_by_seed[seed].report_hash for seed in product_seeds),
        calibrationReportHashes=tuple(
            calibration_by_seed[seed].report_hash for seed in product_seeds
        ),
        freshProcessReportHashes=tuple(
            fresh_by_seed[seed].report_hash for seed in sorted(fresh_by_seed)
        ),
        testReadEvidenceHashes=tuple(test_read_hashes),
        metricSummary=metric_summary,
        maximumEce=(
            max(float(value) for value in eces if value is not None)
            if eces and all(value is not None for value in eces)
            else None
        ),
        peakCudaMemoryMiB=max(memory_values) if memory_values else None,
        gates=gates,
        reasons=reasons,
    )


__all__ = [
    "COLLABORATION_PROTOCOL_HASH",
    "COLLABORATION_PROTOCOL_ID",
    "COLLABORATION_TASK",
    "FORMAL_SEEDS",
    "build_collaboration_task_acceptance",
    "collaboration_protocol",
]
