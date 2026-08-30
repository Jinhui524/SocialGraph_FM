"""Fail-closed aggregation of the fixed SocialGraph-FM Core acceptance gates."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import fmean
from typing import Final

from .contracts import (
    GFM_ACCEPTANCE_GATES,
    GovernanceTaskId,
    GfmAcceptanceManifest,
    GfmDomainCorpusManifest,
    GfmEvaluationReport,
)

COLLABORATION_TASK: Final[GovernanceTaskId] = (
    "governance.collaboration_recommendation"
)
NEWCOMER_TASK: Final[GovernanceTaskId] = "core.newcomer_support"

DEFAULT_PRODUCT_THRESHOLDS: Final[dict[GovernanceTaskId, dict[str, float]]] = {
    COLLABORATION_TASK: {
        "ndcg_relative_gain": 0.05,
        "recall_relative_gain": 0.05,
        "bootstrap_ci95_ndcg_gain_lower": 0.0,
    },
    NEWCOMER_TASK: {
        "ndcg_relative_gain": 0.05,
        "auprc_above_prevalence": 0.05,
    },
}

LODO_FEW_SHOT_PERCENTAGES: Final[tuple[int, ...]] = (1, 5, 10)
LODO_RANDOM_GAIN_AT_5: Final[float] = 0.05
LODO_SINGLE_DOMAIN_GAIN_AT_5: Final[float] = 0.03
LODO_MAXIMUM_DOMAIN_REGRESSION: Final[float] = 0.02
LODO_MINIMUM_IMPROVED_DOMAINS: Final[int] = 2
FORMAL_SEED_COUNT: Final[int] = 3
CORE_FRESH_PROCESS_DOMAIN: Final[str] = "multi-domain"
ZERO_AUDIT_METRICS: Final[tuple[str, ...]] = (
    "future_edge_access_count",
    "cutoff_violation_count",
    "split_overlap_count",
)


def _relative_gain(value: float, baseline: float) -> float:
    return (value - baseline) / max(abs(baseline), 1e-12)


def _report_groups(
    reports: Sequence[GfmEvaluationReport], kind: str
) -> dict[str, list[GfmEvaluationReport]]:
    grouped: dict[str, list[GfmEvaluationReport]] = defaultdict(list)
    for report in reports:
        if report.evaluation_kind == kind and report.task_id is not None:
            grouped[report.task_id].append(report)
    return grouped


def _acceptance_semantic_key(
    report: GfmEvaluationReport,
) -> tuple[object, ...] | None:
    """Return the key whose values are averaged as one formal seed observation."""

    if report.evaluation_kind == "lodo":
        return ("lodo", report.held_out_domain, report.seed)
    if report.evaluation_kind in {"product", "calibration"}:
        return (report.evaluation_kind, report.task_id, report.seed)
    return None


def _same_report_source(
    first: GfmEvaluationReport, second: GfmEvaluationReport
) -> bool:
    return (
        first.seed == second.seed
        and first.run_id == second.run_id
        and first.checkpoint_id == second.checkpoint_id
        and first.domain_id == second.domain_id
    )


def build_gfm_acceptance(
    *,
    experiment_id: str,
    checkpoint_id: str,
    corpora: Sequence[GfmDomainCorpusManifest],
    evaluations: Sequence[GfmEvaluationReport],
    config_hash: str,
    code_hash: str,
    environment_hash: str,
    delivery_evidence_report_hashes: Sequence[str] | None = None,
    product_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    max_ece: float = 0.05,
    max_cuda_memory_mib: float = 7168.0,
    minimum_seeds: int = 3,
) -> GfmAcceptanceManifest:
    """Aggregate immutable evidence; missing, inconsistent, or weak evidence fails closed."""

    checked_corpora = tuple(
        GfmDomainCorpusManifest.model_validate(corpus) for corpus in corpora
    )
    checked_reports = tuple(
        GfmEvaluationReport.model_validate(report) for report in evaluations
    )
    if len({report.report_hash for report in checked_reports}) != len(checked_reports):
        raise ValueError("duplicate immutable evaluation reports are not allowed")
    if len({report.report_id for report in checked_reports}) != len(checked_reports):
        raise ValueError("evaluation report IDs must be unique")
    semantic_keys: set[tuple[object, ...]] = set()
    for report in checked_reports:
        semantic_key = _acceptance_semantic_key(report)
        if semantic_key is None:
            continue
        if semantic_key in semantic_keys:
            raise ValueError(
                "duplicate semantic evaluation evidence cannot be seed-weighted"
            )
        semantic_keys.add(semantic_key)
    if max_ece != 0.05 or max_cuda_memory_mib != 7168.0 or minimum_seeds != 3:
        raise ValueError("formal GFM acceptance gates cannot be weakened or overridden")
    thresholds = {
        task: dict(metrics)
        for task, metrics in (product_thresholds or DEFAULT_PRODUCT_THRESHOLDS).items()
    }
    if thresholds != DEFAULT_PRODUCT_THRESHOLDS:
        raise ValueError("formal GFM acceptance thresholds are frozen by SocialGraph-FM Core")
    domain_ids = tuple(sorted({corpus.domain_id for corpus in checked_corpora}))
    corpus_hashes = tuple(sorted({corpus.logical_hash for corpus in checked_corpora}))
    report_hashes = tuple(sorted({report.report_hash for report in checked_reports}))
    if delivery_evidence_report_hashes is None:
        delivery_hashes = tuple(
            sorted(
                report.report_hash
                for report in checked_reports
                if report.checkpoint_id == checkpoint_id
                and report.evaluation_kind
                in {"product", "calibration", "fresh_process"}
            )
        )
    else:
        delivery_hashes = tuple(sorted(delivery_evidence_report_hashes))
    delivery_hash_set = set(delivery_hashes)
    delivery_reports = tuple(
        report
        for report in checked_reports
        if report.report_hash in delivery_hash_set
    )
    delivery_evidence_valid = (
        len(delivery_hashes) == 5
        and len(delivery_hashes) == len(delivery_hash_set)
        and delivery_hash_set.issubset(set(report_hashes))
        and len(delivery_reports) == len(delivery_hashes)
        and all("shadow" not in report.warnings for report in delivery_reports)
    )
    delivery_fresh = tuple(
        report
        for report in delivery_reports
        if report.evaluation_kind == "fresh_process"
    )
    if (
        len(delivery_fresh) != 1
        or delivery_fresh[0].checkpoint_id != checkpoint_id
    ):
        delivery_evidence_valid = False
    delivery_task_sources: dict[str, tuple[GfmEvaluationReport, GfmEvaluationReport]] = {}
    for task in thresholds:
        selected_product = tuple(
            report
            for report in delivery_reports
            if report.evaluation_kind == "product" and report.task_id == task
        )
        selected_calibration = tuple(
            report
            for report in delivery_reports
            if report.evaluation_kind == "calibration" and report.task_id == task
        )
        if (
            len(selected_product) != 1
            or len(selected_calibration) != 1
            or not _same_report_source(
                selected_product[0], selected_calibration[0]
            )
        ):
            delivery_evidence_valid = False
            continue
        delivery_task_sources[task] = (
            selected_product[0],
            selected_calibration[0],
        )
    if len(delivery_task_sources) != len(thresholds):
        delivery_evidence_valid = False
    elif delivery_fresh:
        selected_seeds = {
            report.seed
            for pair in delivery_task_sources.values()
            for report in pair
        }
        if selected_seeds != {delivery_fresh[0].seed}:
            delivery_evidence_valid = False

    lodo_by_domain: dict[str, list[GfmEvaluationReport]] = defaultdict(list)
    for report in checked_reports:
        if report.evaluation_kind == "lodo" and report.held_out_domain is not None:
            lodo_by_domain[report.held_out_domain].append(report)
    product_groups = _report_groups(checked_reports, "product")
    calibration_groups = _report_groups(checked_reports, "calibration")
    formal_seed_sets = [
        {report.seed for report in lodo_by_domain.get(domain, ())}
        for domain in domain_ids
    ] + [
        {report.seed for report in groups.get(task, ())}
        for groups in (product_groups, calibration_groups)
        for task in thresholds
    ]
    seed_matrix_valid = (
        len(formal_seed_sets) == len(domain_ids) + 2 * len(thresholds)
        and all(len(seeds) == FORMAL_SEED_COUNT for seeds in formal_seed_sets)
        and all(seeds == formal_seed_sets[0] for seeds in formal_seed_sets[1:])
    )
    formal_seeds = (
        frozenset(formal_seed_sets[0]) if seed_matrix_valid else frozenset()
    )

    gates = {gate: False for gate in GFM_ACCEPTANCE_GATES}
    gates["three_domains"] = (
        len(domain_ids) == 3
        and len(checked_corpora) == 3
        and len(corpus_hashes) == 3
        and all(corpus.point_in_time_safe for corpus in checked_corpora)
        and all(corpus.public_checkpoint_eligible for corpus in checked_corpora)
    )

    lodo_domains = tuple(sorted(lodo_by_domain))
    lodo_complete = (
        seed_matrix_valid
        and set(lodo_by_domain) == set(domain_ids)
        and all(
            len(lodo_by_domain[domain]) == FORMAL_SEED_COUNT
            and {report.seed for report in lodo_by_domain[domain]} == formal_seeds
            and len({report.run_id for report in lodo_by_domain[domain]})
            == FORMAL_SEED_COUNT
            and len({report.checkpoint_id for report in lodo_by_domain[domain]})
            == FORMAL_SEED_COUNT
            and all(
                report.domain_id == domain and report.leakage_audit_passed
                for report in lodo_by_domain[domain]
            )
            for domain in domain_ids
        )
    )

    metric_summary: dict[str, dict[str, float]] = {}
    lodo_values: dict[int, dict[str, list[float]]] = {
        percentage: {"gfm": [], "random_init": [], "single_domain": []}
        for percentage in LODO_FEW_SHOT_PERCENTAGES
    }
    improved_domains = 0
    maximum_domain_regression = 0.0
    for domain in domain_ids:
        reports = lodo_by_domain.get(domain, [])
        if (
            not seed_matrix_valid
            or len(reports) != FORMAL_SEED_COUNT
            or {report.seed for report in reports} != formal_seeds
        ):
            reports = []
        domain_values: dict[str, list[float]] = defaultdict(list)
        for report in reports:
            for percentage in LODO_FEW_SHOT_PERCENTAGES:
                for model_name in ("gfm", "random_init", "single_domain"):
                    key = f"few_shot_{percentage}_{model_name}"
                    value = report.metrics.get(key)
                    if value is None:
                        lodo_complete = False
                        continue
                    lodo_values[percentage][model_name].append(value)
                    domain_values[f"{percentage}_{model_name}"].append(value)
        if not reports or any(
            len(domain_values[f"{percentage}_{model_name}"]) != len(reports)
            for percentage in LODO_FEW_SHOT_PERCENTAGES
            for model_name in ("gfm", "random_init", "single_domain")
        ):
            lodo_complete = False
            continue
        gfm_mean = fmean(domain_values["5_gfm"])
        random_mean = fmean(domain_values["5_random_init"])
        single_mean = fmean(domain_values["5_single_domain"])
        best_control = max(random_mean, single_mean)
        controls_positive = random_mean > 0.0 and single_mean > 0.0
        if not controls_positive:
            lodo_complete = False
        regression = (
            max(0.0, -_relative_gain(gfm_mean, best_control))
            if best_control > 0.0
            else 1.0
        )
        maximum_domain_regression = max(maximum_domain_regression, regression)
        if gfm_mean > best_control:
            improved_domains += 1
        metric_summary[f"lodo:{domain}"] = {
            "few_shot_5_gfm": gfm_mean,
            "few_shot_5_random_init": random_mean,
            "few_shot_5_single_domain": single_mean,
            "relative_regression": regression,
        }
    if lodo_complete:
        mean_gfm = fmean(lodo_values[5]["gfm"])
        mean_random = fmean(lodo_values[5]["random_init"])
        mean_single = fmean(lodo_values[5]["single_domain"])
        random_gain = _relative_gain(mean_gfm, mean_random)
        single_gain = _relative_gain(mean_gfm, mean_single)
        metric_summary["lodo:aggregate"] = {
            "few_shot_5_relative_gain_over_random_init": random_gain,
            "few_shot_5_relative_gain_over_single_domain": single_gain,
            "improved_domain_count": float(improved_domains),
            "maximum_domain_relative_regression": maximum_domain_regression,
        }
        lodo_complete = (
            random_gain >= LODO_RANDOM_GAIN_AT_5
            and single_gain >= LODO_SINGLE_DOMAIN_GAIN_AT_5
            and improved_domains >= LODO_MINIMUM_IMPROVED_DOMAINS
            and maximum_domain_regression <= LODO_MAXIMUM_DOMAIN_REGRESSION
        )
    gates["lodo_complete"] = lodo_complete

    product_ok = bool(thresholds) and delivery_evidence_valid and seed_matrix_valid
    for task in thresholds:
        task_reports = product_groups.get(task, [])
        task_seeds = {report.seed for report in task_reports}
        task_summary: dict[str, float] = {}
        task_provenance_valid = (
            len(task_reports) == FORMAL_SEED_COUNT
            and task_seeds == formal_seeds
            and len({report.checkpoint_id for report in task_reports})
            == FORMAL_SEED_COUNT
            and len({report.run_id for report in task_reports})
            == FORMAL_SEED_COUNT
        )
        calibration_by_seed = {
            report.seed: report for report in calibration_groups.get(task, [])
        }
        if task_provenance_valid:
            task_provenance_valid = all(
                report.seed in calibration_by_seed
                and _same_report_source(report, calibration_by_seed[report.seed])
                for report in task_reports
            )
        if not task_provenance_valid:
            product_ok = False
            task_reports = []
        required: tuple[str, ...]
        if task == COLLABORATION_TASK:
            required = (
                "ndcg@20",
                "baseline_ndcg@20",
                "recall@20",
                "baseline_recall@20",
                "bootstrap_ci95_ndcg_gain_lower",
                "query_count",
            )
        else:
            required = (
                "ndcg@20",
                "baseline_ndcg@20",
                "auprc",
                "label_prevalence",
                "query_count",
                "outcome_count",
            )
        values_by_metric = {
            metric: [report.metrics[metric] for report in task_reports if metric in report.metrics]
            for metric in required
        }
        if any(
            not values or len(values) != len(task_reports)
            for values in values_by_metric.values()
        ):
            product_ok = False
        else:
            ndcg = fmean(values_by_metric["ndcg@20"])
            baseline_ndcg = fmean(values_by_metric["baseline_ndcg@20"])
            ndcg_gain = _relative_gain(ndcg, baseline_ndcg)
            task_summary.update(
                {
                    "ndcg@20": ndcg,
                    "baseline_ndcg@20": baseline_ndcg,
                    "ndcg_relative_gain": ndcg_gain,
                }
            )
            if (
                baseline_ndcg <= 0.0
                or ndcg_gain < 0.05
                or min(values_by_metric["query_count"]) < 100.0
            ):
                product_ok = False
            if task == COLLABORATION_TASK:
                recall = fmean(values_by_metric["recall@20"])
                baseline_recall = fmean(values_by_metric["baseline_recall@20"])
                recall_gain = _relative_gain(recall, baseline_recall)
                ci_lower = min(values_by_metric["bootstrap_ci95_ndcg_gain_lower"])
                task_summary.update(
                    {
                        "recall@20": recall,
                        "baseline_recall@20": baseline_recall,
                        "recall_relative_gain": recall_gain,
                        "bootstrap_ci95_ndcg_gain_lower": ci_lower,
                    }
                )
                if baseline_recall <= 0.0 or recall_gain < 0.05 or ci_lower <= 0.0:
                    product_ok = False
            else:
                auprc = fmean(values_by_metric["auprc"])
                prevalence = fmean(values_by_metric["label_prevalence"])
                above_prevalence = auprc - prevalence
                task_summary.update(
                    {
                        "auprc": auprc,
                        "label_prevalence": prevalence,
                        "auprc_above_prevalence": above_prevalence,
                    }
                )
                if (
                    prevalence <= 0.0
                    or prevalence >= 1.0
                    or above_prevalence < 0.05
                    or min(values_by_metric["outcome_count"]) < 100.0
                ):
                    product_ok = False
        if not all(report.leakage_audit_passed for report in task_reports):
            product_ok = False
        selected_reports = [
            report
            for report in delivery_reports
            if report.evaluation_kind == "product" and report.task_id == task
        ]
        if len(selected_reports) != 1 or any(
            metric not in report.metrics
            for report in selected_reports
            for metric in required
        ):
            product_ok = False
        else:
            for report in selected_reports:
                selected_ndcg_gain = _relative_gain(
                    report.metrics["ndcg@20"], report.metrics["baseline_ndcg@20"]
                )
                if (
                    report.metrics["baseline_ndcg@20"] <= 0.0
                    or selected_ndcg_gain < 0.05
                    or report.metrics["query_count"] < 100.0
                ):
                    product_ok = False
                if task == COLLABORATION_TASK:
                    selected_recall_gain = _relative_gain(
                        report.metrics["recall@20"],
                        report.metrics["baseline_recall@20"],
                    )
                    if (
                        report.metrics["baseline_recall@20"] <= 0.0
                        or
                        selected_recall_gain < 0.05
                        or report.metrics["bootstrap_ci95_ndcg_gain_lower"] <= 0.0
                    ):
                        product_ok = False
                elif (
                    report.metrics["auprc"] - report.metrics["label_prevalence"]
                    < 0.05
                    or report.metrics["outcome_count"] < 100.0
                    or not 0.0 < report.metrics["label_prevalence"] < 1.0
                ):
                    product_ok = False
        metric_summary[f"product:{task}"] = task_summary
    gates["product_metrics"] = product_ok

    calibration_values: list[float] = []
    calibration_ok = bool(thresholds) and delivery_evidence_valid and seed_matrix_valid
    for task in thresholds:
        task_reports = calibration_groups.get(task, [])
        task_seeds = {report.seed for report in task_reports}
        task_provenance_valid = (
            len(task_reports) == FORMAL_SEED_COUNT
            and task_seeds == formal_seeds
            and len({report.checkpoint_id for report in task_reports})
            == FORMAL_SEED_COUNT
            and len({report.run_id for report in task_reports})
            == FORMAL_SEED_COUNT
        )
        product_by_seed = {
            report.seed: report for report in product_groups.get(task, [])
        }
        if task_provenance_valid:
            task_provenance_valid = all(
                report.seed in product_by_seed
                and _same_report_source(report, product_by_seed[report.seed])
                for report in task_reports
            )
        if not task_provenance_valid:
            calibration_ok = False
            task_reports = []
        values = [report.ece for report in task_reports if report.ece is not None]
        required_strata = {
            "ece_institution_small",
            "ece_institution_medium",
            "ece_institution_large",
            "ece_topic_cluster_0",
            "ece_topic_cluster_1",
            "ece_topic_cluster_2",
            *(
                {"ece_first_time", "ece_repeated"}
                if task == COLLABORATION_TASK
                else {"ece_newcomer"}
            ),
        }
        if len(values) != len(task_reports) or not values or any(
            value > max_ece for value in values
        ):
            calibration_ok = False
        if any(report.metrics.get("strata_complete") != 1.0 for report in task_reports):
            calibration_ok = False
        for report in task_reports:
            stratum_values = [report.metrics.get(name) for name in required_strata]
            if any(value is None or value > max_ece for value in stratum_values):
                calibration_ok = False
        selected_reports = [
            report
            for report in delivery_reports
            if report.evaluation_kind == "calibration" and report.task_id == task
        ]
        if len(selected_reports) != 1 or any(
            report.ece is None
            or report.ece > max_ece
            or report.metrics.get("strata_complete") != 1.0
            or any(
                report.metrics.get(name) is None
                or report.metrics[name] > max_ece
                for name in required_strata
            )
            for report in selected_reports
        ):
            calibration_ok = False
        calibration_values.extend(values)
        if values:
            metric_summary[f"calibration:{task}"] = {"ece": fmean(values)}
    gates["calibration_ece"] = calibration_ok

    memory_values = [
        report.peak_cuda_memory_mib
        for report in checked_reports
        if report.peak_cuda_memory_mib is not None
    ]
    formal_reports = [
        report
        for report in checked_reports
        if report.evaluation_kind in {"lodo", "product", "fresh_process"}
    ]
    gates["cuda_memory"] = (
        len(memory_values) == len(checked_reports)
        and len(formal_reports) >= minimum_seeds
        and max(memory_values, default=max_cuda_memory_mib) < max_cuda_memory_mib
    )

    fresh_reports = [
        report
        for report in checked_reports
        if report.evaluation_kind == "fresh_process"
    ]
    fresh_digests = tuple(
        sorted(
            {
                report.verification_digest
                for report in fresh_reports
                if report.verification_digest is not None
            }
        )
    )
    formal_core_fresh = [
        report
        for report in fresh_reports
        if report.domain_id == CORE_FRESH_PROCESS_DOMAIN
        and report.seed in formal_seeds
    ]
    # Legacy/synthetic acceptance fixtures may use the same formal checkpoint
    # as both the product and core-delivery candidate.  A real suite has three
    # explicit ``multi-domain`` pretrain verifications; when those reports are
    # present their matrix must be exact, never partial.
    core_fresh_valid = not formal_core_fresh or (
        seed_matrix_valid
        and len(formal_core_fresh) == FORMAL_SEED_COUNT
        and {report.seed for report in formal_core_fresh} == formal_seeds
        and len({report.run_id for report in formal_core_fresh})
        == FORMAL_SEED_COUNT
        and len({report.checkpoint_id for report in formal_core_fresh})
        == FORMAL_SEED_COUNT
    )
    product_source_reports = [
        report
        for task in thresholds
        for report in product_groups.get(task, ())
    ]
    product_fresh_valid = seed_matrix_valid and all(
        len(
            [
                fresh
                for fresh in fresh_reports
                if fresh.seed == report.seed
                and fresh.run_id == report.run_id
                and fresh.checkpoint_id == report.checkpoint_id
            ]
        )
        == 1
        for report in product_source_reports
    )
    gates["fresh_process_verification"] = (
        core_fresh_valid
        and product_fresh_valid
        and len(fresh_digests) >= FORMAL_SEED_COUNT
        and all(
            report.fresh_process_verified
            and report.verification_digest is not None
            and report.metrics.get("fresh_process_repeat_match") == 1.0
            for report in fresh_reports
        )
        and delivery_evidence_valid
    )

    identity_ok = all(report.experiment_id == experiment_id for report in checked_reports)
    gates["temporal_leakage_audit"] = (
        bool(checked_reports)
        and identity_ok
        and all("shadow" not in report.warnings for report in checked_reports)
        and all(report.leakage_audit_passed for report in checked_reports)
        and all(
            all(report.metrics.get(metric) == 0.0 for metric in ZERO_AUDIT_METRICS)
            for report in checked_reports
        )
        and all(
            report.evaluation_kind != "lodo"
            or report.metrics.get("target_domain_pretrain_access_count") == 0.0
            for report in checked_reports
        )
        and all(corpus.point_in_time_safe for corpus in checked_corpora)
    )

    reasons = tuple(
        f"hard gate failed: {gate}" for gate in sorted(gates) if not gates[gate]
    )
    return GfmAcceptanceManifest.create(
        experimentId=experiment_id,
        checkpointId=checkpoint_id,
        accepted=not reasons,
        domainIds=domain_ids,
        lodoDomains=lodo_domains,
        productTaskIds=tuple(sorted(thresholds)),
        corpusHashes=corpus_hashes,
        configHash=config_hash,
        codeHash=code_hash,
        environmentHash=environment_hash,
        evaluationReportHashes=report_hashes,
        deliveryEvidenceReportHashes=delivery_hashes,
        maximumEce=max(calibration_values) if calibration_values else None,
        peakCudaMemoryMiB=max(memory_values) if memory_values else None,
        freshProcessDigests=fresh_digests,
        metricSummary=metric_summary,
        gates=gates,
        reasons=reasons,
    )


__all__ = [
    "COLLABORATION_TASK",
    "DEFAULT_PRODUCT_THRESHOLDS",
    "NEWCOMER_TASK",
    "build_gfm_acceptance",
]
