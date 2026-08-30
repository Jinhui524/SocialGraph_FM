"""Small, dependency-free evaluators for GFM transfer and product-facing scores."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor


def _finite_vector(value: Tensor, *, name: str) -> Tensor:
    flattened = value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    if not flattened.numel() or not bool(torch.isfinite(flattened).all()):
        raise ValueError(f"{name} must contain finite scores")
    return flattened


@dataclass(frozen=True)
class RankingMetrics:
    """One-positive-per-query ranking metrics using pessimistic tie handling."""

    mrr: float
    mean_rank: float
    hits_at_k: Mapping[int, float]
    recall_at_k: Mapping[int, float]
    ndcg_at_k: Mapping[int, float]

    def as_dict(self) -> dict[str, float]:
        result = {"mrr": self.mrr, "mean_rank": self.mean_rank}
        for cutoff, value in self.hits_at_k.items():
            result[f"hits@{cutoff}"] = value
        for cutoff, value in self.recall_at_k.items():
            result[f"recall@{cutoff}"] = value
        for cutoff, value in self.ndcg_at_k.items():
            result[f"ndcg@{cutoff}"] = value
        return result


def ranking_metrics(
    positive_scores: Tensor,
    negative_scores: Tensor,
    *,
    ks: Sequence[int] = (10, 50),
) -> RankingMetrics:
    """Evaluate positive scores against per-query ``[Q, K]`` or shared ``[K]`` negatives."""

    positives = _finite_vector(positive_scores, name="positive_scores")
    negatives = negative_scores.detach().to(dtype=torch.float64, device="cpu")
    if negatives.ndim == 1:
        negatives = negatives.reshape(1, -1).expand(positives.shape[0], -1)
    if negatives.ndim != 2 or negatives.shape[0] != positives.shape[0]:
        raise ValueError("negative_scores must be [K] shared or [Q, K] per query")
    if not negatives.shape[1] or not bool(torch.isfinite(negatives).all()):
        raise ValueError("negative_scores must contain finite comparison scores")
    cutoffs = tuple(int(value) for value in ks)
    if not cutoffs or any(value < 1 for value in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("ks must contain unique positive cutoffs")

    # A tie is conservatively placed behind all negatives with the same score.
    ranks = 1 + (negatives >= positives.reshape(-1, 1)).sum(dim=1)
    reciprocal = 1.0 / ranks.to(torch.float64)
    hits: dict[int, float] = {}
    recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}
    discounts = 1.0 / torch.log2(ranks.to(torch.float64) + 1.0)
    for cutoff in cutoffs:
        present = ranks <= cutoff
        hits[cutoff] = float(present.to(torch.float64).mean())
        # With exactly one relevant item per query, recall@K equals hits@K.
        recall[cutoff] = hits[cutoff]
        ndcg[cutoff] = float(torch.where(present, discounts, 0.0).mean())
    return RankingMetrics(
        mrr=float(reciprocal.mean()),
        mean_rank=float(ranks.to(torch.float64).mean()),
        hits_at_k=hits,
        recall_at_k=recall,
        ndcg_at_k=ndcg,
    )


@dataclass(frozen=True)
class CalibrationMetrics:
    expected_calibration_error: float
    brier_score: float
    bin_count: int


def expected_calibration_error(
    probabilities: Tensor,
    labels: Tensor,
    *,
    bins: int = 15,
) -> CalibrationMetrics:
    """Return equal-width ECE and Brier score for binary event probabilities."""

    probability = _finite_vector(probabilities, name="probabilities")
    target = _finite_vector(labels, name="labels")
    if probability.shape != target.shape:
        raise ValueError("probabilities and labels must have identical shapes")
    if bins < 2:
        raise ValueError("bins must be at least two")
    if bool(((probability < 0.0) | (probability > 1.0)).any()):
        raise ValueError("probabilities must lie in [0, 1]")
    if bool(((target != 0.0) & (target != 1.0)).any()):
        raise ValueError("labels must be binary")

    boundaries = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    ece = torch.tensor(0.0, dtype=torch.float64)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        selected = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if bool(selected.any()):
            weight = selected.to(torch.float64).mean()
            ece += weight * torch.abs(probability[selected].mean() - target[selected].mean())
    brier = torch.mean((probability - target) ** 2)
    return CalibrationMetrics(
        expected_calibration_error=float(ece),
        brier_score=float(brier),
        bin_count=bins,
    )


@dataclass(frozen=True)
class DomainTransferResult:
    """Primary held-out-domain metric for one shared and two control models."""

    domain_id: str
    gfm_metric: float
    random_init_metric: float
    single_domain_metric: float | None = None

    def __post_init__(self) -> None:
        if not self.domain_id or not isfinite(self.gfm_metric) or not isfinite(
            self.random_init_metric
        ):
            raise ValueError("LODO domain results require an ID and finite metrics")
        if self.single_domain_metric is not None and not isfinite(self.single_domain_metric):
            raise ValueError("LODO single-domain metric must be finite when supplied")


@dataclass(frozen=True)
class LeaveOneDomainOutSummary:
    primary_metric: str
    mean_gfm_metric: float
    mean_gain_over_random: float
    mean_gain_over_single_domain: float | None
    positive_transfer_domains: tuple[str, ...]
    negative_transfer_domains: tuple[str, ...]
    domain_results: tuple[DomainTransferResult, ...]


def evaluate_lodo(
    results: Iterable[DomainTransferResult],
    *,
    primary_metric: str,
    tolerance: float = 0.0,
) -> LeaveOneDomainOutSummary:
    """Summarise leave-one-domain-out transfer without inventing acceptance thresholds."""

    records = tuple(results)
    if not records or not primary_metric:
        raise ValueError("LODO evaluation requires results and a primary metric name")
    if tolerance < 0.0 or not isfinite(tolerance):
        raise ValueError("LODO tolerance must be finite and non-negative")
    domains = tuple(record.domain_id for record in records)
    if len(set(domains)) != len(domains):
        raise ValueError("LODO domain IDs must be unique")
    random_gains = tuple(record.gfm_metric - record.random_init_metric for record in records)
    single_gains: list[float] = []
    for record in records:
        if record.single_domain_metric is not None:
            single_gains.append(record.gfm_metric - record.single_domain_metric)
    single_gain = sum(single_gains) / len(single_gains) if single_gains else None
    positive: list[str] = []
    negative: list[str] = []
    for record in records:
        controls = [record.random_init_metric]
        if record.single_domain_metric is not None:
            controls.append(record.single_domain_metric)
        best_control = max(controls)
        if record.gfm_metric > best_control + tolerance:
            positive.append(record.domain_id)
        elif record.gfm_metric < best_control - tolerance:
            negative.append(record.domain_id)
    return LeaveOneDomainOutSummary(
        primary_metric=primary_metric,
        mean_gfm_metric=sum(record.gfm_metric for record in records) / len(records),
        mean_gain_over_random=sum(random_gains) / len(random_gains),
        mean_gain_over_single_domain=single_gain,
        positive_transfer_domains=tuple(positive),
        negative_transfer_domains=tuple(negative),
        domain_results=records,
    )
