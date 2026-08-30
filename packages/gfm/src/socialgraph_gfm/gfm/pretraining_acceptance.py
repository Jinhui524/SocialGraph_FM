"""Fail-closed acceptance for the reusable three-domain pretraining backbone."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import isfinite
from statistics import fmean
from typing import Final, Literal

from .contracts import (
    GFM_PRETRAINING_ACCEPTANCE_GATES,
    GfmCheckpointManifest,
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
    GfmPretrainingAcceptanceManifest,
    GfmRunManifest,
)

FORMAL_VARIANTS: Final[tuple[Literal["core-base", "core-moe"], ...]] = (
    "core-base",
    "core-moe",
)
FORMAL_SEEDS: Final[tuple[int, ...]] = (20260821, 20260822, 20260823)
FORMAL_DOMAINS: Final[frozenset[str]] = frozenset(
    {
        "openalex-graph-ai",
        "thgl-software-2.0.0",
        "wikimedia-talk-article-2011-2015",
    }
)
MAXIMUM_CUDA_MEMORY_MIB: Final[float] = 7168.0
ZERO_AUDIT_METRICS: Final[tuple[str, ...]] = (
    "future_edge_access_count",
    "cutoff_violation_count",
    "split_overlap_count",
)


def _relative_gain(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), 1e-12)


def _finite_metric(report: GfmEvaluationReport, name: str) -> float | None:
    value = report.metrics.get(name)
    if value is None or not isfinite(float(value)):
        return None
    return float(value)


def build_gfm_pretraining_acceptance(
    *,
    experiment_id: str,
    corpora: Sequence[GfmDomainCorpusManifest],
    runs: Sequence[GfmRunManifest],
    checkpoints: Sequence[GfmCheckpointManifest],
    evaluations: Sequence[GfmEvaluationReport],
    config_hash: str,
    code_hash: str,
    environment_hash: str,
) -> GfmPretrainingAcceptanceManifest:
    """Derive the frozen formal matrix without consulting product evidence.

    Physical checkpoint and evidence-file integrity is the registry boundary's
    responsibility.  This pure aggregation layer validates the complete
    semantic matrix and its immutable contract provenance.
    """

    checked_corpora = tuple(
        GfmDomainCorpusManifest.model_validate(value) for value in corpora
    )
    checked_runs = tuple(GfmRunManifest.model_validate(value) for value in runs)
    checked_checkpoints = tuple(
        GfmCheckpointManifest.model_validate(value) for value in checkpoints
    )
    checked_reports = tuple(
        GfmEvaluationReport.model_validate(value) for value in evaluations
    )
    if len({value.run_id for value in checked_runs}) != len(checked_runs):
        raise ValueError("duplicate pretraining run IDs are not allowed")
    if len({value.checkpoint_id for value in checked_checkpoints}) != len(
        checked_checkpoints
    ):
        raise ValueError("duplicate pretraining checkpoint IDs are not allowed")
    if len({value.report_hash for value in checked_reports}) != len(checked_reports):
        raise ValueError("duplicate pretraining evaluation hashes are not allowed")
    if len({value.report_id for value in checked_reports}) != len(checked_reports):
        raise ValueError("duplicate pretraining evaluation IDs are not allowed")

    gates = {name: False for name in GFM_PRETRAINING_ACCEPTANCE_GATES}
    corpus_by_domain = {value.domain_id: value for value in checked_corpora}
    corpus_hashes = tuple(sorted(value.logical_hash for value in checked_corpora))
    gates["three_domains"] = (
        len(checked_corpora) == 3
        and set(corpus_by_domain) == FORMAL_DOMAINS
        and len(corpus_hashes) == 3
        and all(value.point_in_time_safe for value in checked_corpora)
        and all(value.public_checkpoint_eligible for value in checked_corpora)
    )
    expected_hashes = set(corpus_hashes)

    pretrain_runs = tuple(value for value in checked_runs if value.phase == "pretrain")
    lodo_runs = tuple(value for value in checked_runs if value.phase == "lodo")
    pretrain_by_key: dict[tuple[str, int], list[GfmRunManifest]] = defaultdict(list)
    lodo_by_key: dict[tuple[str, str, int], list[GfmRunManifest]] = defaultdict(list)
    for run in pretrain_runs:
        pretrain_by_key[(run.architecture_variant, run.seed)].append(run)
    for run in lodo_runs:
        lodo_by_key[
            (run.architecture_variant, str(run.held_out_domain), run.seed)
        ].append(run)

    expected_pretrain_keys = {
        (variant, seed) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS
    }
    expected_lodo_keys = {
        (variant, domain, seed)
        for variant in FORMAL_VARIANTS
        for domain in FORMAL_DOMAINS
        for seed in FORMAL_SEEDS
    }
    matrix_pretrain_runs = tuple(
        values[0]
        for key, values in sorted(pretrain_by_key.items())
        if key in expected_pretrain_keys and len(values) == 1
    )
    matrix_lodo_runs = tuple(
        values[0]
        for key, values in sorted(lodo_by_key.items())
        if key in expected_lodo_keys and len(values) == 1
    )
    run_by_id = {value.run_id: value for value in checked_runs}
    checkpoint_by_run: dict[str, list[GfmCheckpointManifest]] = defaultdict(list)
    for checkpoint in checked_checkpoints:
        checkpoint_by_run[checkpoint.run_id].append(checkpoint)
    report_by_run_kind: dict[tuple[str, str], list[GfmEvaluationReport]] = defaultdict(
        list
    )
    for report in checked_reports:
        report_by_run_kind[(report.run_id, report.evaluation_kind)].append(report)

    common_run_provenance = all(
        run.experiment_id == experiment_id
        and run.status == "succeeded"
        and run.config_hash == config_hash
        and run.code_hash == code_hash
        and run.environment_hash == environment_hash
        and set(run.corpus_hashes) == expected_hashes
        for run in (*matrix_pretrain_runs, *matrix_lodo_runs)
    )
    pretrain_shapes_valid = all(
        set(run.domain_ids) == FORMAL_DOMAINS and run.held_out_domain is None
        for run in matrix_pretrain_runs
    )
    lodo_shapes_valid = all(
        run.held_out_domain in FORMAL_DOMAINS
        and set(run.domain_ids) == FORMAL_DOMAINS - {str(run.held_out_domain)}
        for run in matrix_lodo_runs
    )

    relevant_run_ids = {
        value.run_id for value in (*matrix_pretrain_runs, *matrix_lodo_runs)
    }
    relevant_checkpoints = tuple(
        value
        for value in checked_checkpoints
        if value.run_id in relevant_run_ids
    )
    checkpoint_provenance = (
        len(relevant_checkpoints) == 24
        and all(len(checkpoint_by_run[run_id]) == 1 for run_id in relevant_run_ids)
        and all(
            checkpoint.run_id in run_by_id
            and checkpoint.config_hash == config_hash
            and set(checkpoint.corpus_hashes) == expected_hashes
            for checkpoint in relevant_checkpoints
        )
    )

    in_domain_reports = tuple(
        report
        for run in matrix_pretrain_runs
        for report in report_by_run_kind[(run.run_id, "in_domain")]
    )
    fresh_reports = tuple(
        report
        for run in matrix_pretrain_runs
        for report in report_by_run_kind[(run.run_id, "fresh_process")]
    )
    lodo_reports = tuple(
        report
        for run in matrix_lodo_runs
        for report in report_by_run_kind[(run.run_id, "lodo")]
    )
    in_domain_provenance = len(in_domain_reports) == 6 and all(
        len(report_by_run_kind[(run.run_id, "in_domain")]) == 1
        and report_by_run_kind[(run.run_id, "in_domain")][0].checkpoint_id
        == checkpoint_by_run[run.run_id][0].checkpoint_id
        and report_by_run_kind[(run.run_id, "in_domain")][0].seed == run.seed
        and report_by_run_kind[(run.run_id, "in_domain")][0].domain_id
        == "multi-domain"
        and "physical-test-view-read-once-after-best"
        in report_by_run_kind[(run.run_id, "in_domain")][0].warnings
        for run in matrix_pretrain_runs
        if len(checkpoint_by_run[run.run_id]) == 1
    )
    gates["formal_pretrain_matrix"] = (
        set(pretrain_by_key) == expected_pretrain_keys
        and len(matrix_pretrain_runs) == 6
        and common_run_provenance
        and pretrain_shapes_valid
        and checkpoint_provenance
        and in_domain_provenance
    )

    lodo_report_matrix: dict[
        tuple[str, str, int], GfmEvaluationReport
    ] = {}
    lodo_report_provenance = len(lodo_reports) == 18
    for key in expected_lodo_keys:
        run_values = lodo_by_key.get(key, [])
        if len(run_values) != 1:
            lodo_report_provenance = False
            continue
        run = run_values[0]
        reports = report_by_run_kind[(run.run_id, "lodo")]
        checkpoints_for_run = checkpoint_by_run[run.run_id]
        if (
            len(reports) != 1
            or len(checkpoints_for_run) != 1
            or reports[0].checkpoint_id != checkpoints_for_run[0].checkpoint_id
            or reports[0].seed != run.seed
            or reports[0].domain_id != key[1]
            or reports[0].held_out_domain != key[1]
        ):
            lodo_report_provenance = False
            continue
        lodo_report_matrix[key] = reports[0]

    metric_names = tuple(
        f"few_shot_{percentage}_{control}"
        for percentage in (1, 5, 10)
        for control in ("gfm", "random_init", "single_domain")
    )
    lodo_metric_complete = len(lodo_report_matrix) == 18 and all(
        all(_finite_metric(report, name) is not None for name in metric_names)
        for report in lodo_report_matrix.values()
    )
    per_variant_domain: dict[str, dict[str, float]] = {
        variant: {} for variant in FORMAL_VARIANTS
    }
    if lodo_metric_complete:
        for variant in FORMAL_VARIANTS:
            for domain in FORMAL_DOMAINS:
                per_variant_domain[variant][domain] = fmean(
                    float(lodo_report_matrix[(variant, domain, seed)].metrics[
                        "few_shot_5_gfm"
                    ])
                    for seed in FORMAL_SEEDS
                )

    selected_variant: Literal["core-base", "core-moe"] = "core-base"
    moe_mean_gain = 0.0
    moe_maximum_regression = 0.0
    if lodo_metric_complete:
        moe_gains = [
            _relative_gain(
                per_variant_domain["core-moe"][domain],
                per_variant_domain["core-base"][domain],
            )
            for domain in sorted(FORMAL_DOMAINS)
        ]
        moe_mean_gain = fmean(moe_gains)
        moe_maximum_regression = max(0.0, *(-value for value in moe_gains))
        if moe_mean_gain >= 0.02 and moe_maximum_regression <= 0.01:
            selected_variant = "core-moe"
    gates["variant_selection"] = (
        set(lodo_by_key) == expected_lodo_keys
        and len(matrix_lodo_runs) == 18
        and common_run_provenance
        and lodo_shapes_valid
        and checkpoint_provenance
        and lodo_report_provenance
        and lodo_metric_complete
    )

    selected_reports = tuple(
        lodo_report_matrix[(selected_variant, domain, seed)]
        for domain in sorted(FORMAL_DOMAINS)
        for seed in FORMAL_SEEDS
        if (selected_variant, domain, seed) in lodo_report_matrix
    )
    lodo_random_gain = 0.0
    lodo_single_gain = 0.0
    improved_domains = 0
    maximum_domain_regression = 1.0
    lodo_thresholds_passed = False
    if len(selected_reports) == 9 and lodo_metric_complete:
        mean_gfm = fmean(report.metrics["few_shot_5_gfm"] for report in selected_reports)
        mean_random = fmean(
            report.metrics["few_shot_5_random_init"] for report in selected_reports
        )
        mean_single = fmean(
            report.metrics["few_shot_5_single_domain"] for report in selected_reports
        )
        lodo_random_gain = _relative_gain(mean_gfm, mean_random)
        lodo_single_gain = _relative_gain(mean_gfm, mean_single)
        regressions: list[float] = []
        for domain in sorted(FORMAL_DOMAINS):
            domain_reports = tuple(
                lodo_report_matrix[(selected_variant, domain, seed)]
                for seed in FORMAL_SEEDS
            )
            domain_gfm = fmean(
                report.metrics["few_shot_5_gfm"] for report in domain_reports
            )
            domain_random = fmean(
                report.metrics["few_shot_5_random_init"] for report in domain_reports
            )
            domain_single = fmean(
                report.metrics["few_shot_5_single_domain"] for report in domain_reports
            )
            best_control = max(domain_random, domain_single)
            if domain_gfm > best_control:
                improved_domains += 1
            regressions.append(
                max(0.0, -_relative_gain(domain_gfm, best_control))
                if best_control > 0.0
                else 1.0
            )
        maximum_domain_regression = max(regressions)
        lodo_thresholds_passed = (
            mean_random > 0.0
            and mean_single > 0.0
            and lodo_random_gain >= 0.05
            and lodo_single_gain >= 0.03
            and improved_domains >= 2
            and maximum_domain_regression <= 0.02
        )
    gates["lodo_complete"] = (
        gates["variant_selection"] and lodo_thresholds_passed
    )

    fresh_digests = tuple(
        sorted(
            report.verification_digest
            for report in fresh_reports
            if report.verification_digest is not None
        )
    )
    gates["fresh_process_verification"] = (
        len(fresh_reports) == 6
        and len(fresh_digests) == 6
        and len(set(fresh_digests)) == 6
        and all(
            len(report_by_run_kind[(run.run_id, "fresh_process")]) == 1
            and report_by_run_kind[(run.run_id, "fresh_process")][0].checkpoint_id
            == checkpoint_by_run[run.run_id][0].checkpoint_id
            and report_by_run_kind[(run.run_id, "fresh_process")][0].seed
            == run.seed
            and report_by_run_kind[(run.run_id, "fresh_process")][0].domain_id
            == "multi-domain"
            and report_by_run_kind[(run.run_id, "fresh_process")][0].fresh_process_verified
            and report_by_run_kind[(run.run_id, "fresh_process")][0].metrics.get(
                "fresh_process_repeat_match"
            )
            == 1.0
            for run in matrix_pretrain_runs
            if len(checkpoint_by_run[run.run_id]) == 1
        )
    )

    relevant_reports = (*in_domain_reports, *fresh_reports, *lodo_reports)
    memory_values = [
        value
        for value in (
            *(run.peak_cuda_memory_mib for run in (*matrix_pretrain_runs, *matrix_lodo_runs)),
            *(report.peak_cuda_memory_mib for report in relevant_reports),
        )
        if value is not None
    ]
    gates["cuda_memory"] = (
        len(memory_values) == 24 + 30
        and bool(memory_values)
        and max(memory_values) < MAXIMUM_CUDA_MEMORY_MIB
    )
    gates["temporal_leakage_audit"] = (
        len(relevant_reports) == 30
        and all(corpus.point_in_time_safe for corpus in checked_corpora)
        and all(report.leakage_audit_passed for report in relevant_reports)
        and all("shadow" not in report.warnings for report in relevant_reports)
        and all(
            all(report.metrics.get(name) == 0.0 for name in ZERO_AUDIT_METRICS)
            for report in relevant_reports
        )
        and all(
            report.metrics.get("target_domain_pretrain_access_count") == 0.0
            for report in lodo_reports
        )
    )

    selected_checkpoint_ids = tuple(
        checkpoint_by_run[run.run_id][0].checkpoint_id
        for run in sorted(matrix_pretrain_runs, key=lambda value: value.seed)
        if run.architecture_variant == selected_variant
        and len(checkpoint_by_run[run.run_id]) == 1
    )
    metric_summary: dict[str, dict[str, float | str]] = {
        "variant_selection": {
            "selected_variant": selected_variant,
            "moe_mean_relative_gain": moe_mean_gain,
            "moe_maximum_domain_relative_regression": moe_maximum_regression,
        },
        "lodo:aggregate": {
            "few_shot_5_relative_gain_over_random_init": lodo_random_gain,
            "few_shot_5_relative_gain_over_single_domain": lodo_single_gain,
            "improved_domain_count": float(improved_domains),
            "maximum_domain_relative_regression": maximum_domain_regression,
        },
    }
    reasons = tuple(
        f"hard gate failed: {name}" for name in sorted(gates) if not gates[name]
    )
    return GfmPretrainingAcceptanceManifest.create(
        experimentId=experiment_id,
        accepted=not reasons,
        architectureVariants=FORMAL_VARIANTS,
        selectedVariant=selected_variant,
        formalSeeds=FORMAL_SEEDS,
        domainIds=tuple(sorted(corpus_by_domain)),
        lodoDomains=tuple(sorted({str(run.held_out_domain) for run in matrix_lodo_runs})),
        corpusHashes=corpus_hashes,
        pretrainRunIds=tuple(sorted(run.run_id for run in matrix_pretrain_runs)),
        lodoRunIds=tuple(sorted(run.run_id for run in matrix_lodo_runs)),
        evidenceCheckpointIds=tuple(
            sorted(checkpoint.checkpoint_id for checkpoint in relevant_checkpoints)
        ),
        selectedCheckpointIds=selected_checkpoint_ids,
        evaluationReportHashes=tuple(
            sorted(report.report_hash for report in relevant_reports)
        ),
        freshProcessDigests=fresh_digests,
        configHash=config_hash,
        codeHash=code_hash,
        environmentHash=environment_hash,
        peakCudaMemoryMiB=max(memory_values) if memory_values else None,
        metricSummary=metric_summary,
        gates=gates,
        reasons=reasons,
    )


__all__ = [
    "FORMAL_DOMAINS",
    "FORMAL_SEEDS",
    "FORMAL_VARIANTS",
    "build_gfm_pretraining_acceptance",
]
