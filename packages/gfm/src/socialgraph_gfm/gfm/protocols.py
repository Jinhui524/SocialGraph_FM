"""Frozen temporal product and leave-one-domain-family-out protocols."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from ..errors import ContractViolation

COLLABORATION_TASK = "governance.collaboration_recommendation"
NEWCOMER_TASK = "core.newcomer_support"
DOMAIN_FAMILIES = (
    "academic-collaboration",
    "software-activity",
    "online-community",
)


@dataclass(frozen=True)
class SampleIdentity:
    domain_id: str
    graph_version: str
    cutoff: datetime
    horizon_days: int
    task_id: str
    source_corpus_hash: str

    def __post_init__(self) -> None:
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise ContractViolation("GFM sample cutoff must be timezone-aware")
        if self.horizon_days < 1:
            raise ContractViolation("GFM sample horizon must be positive")
        if len(self.source_corpus_hash) != 64:
            raise ContractViolation("GFM sample requires a SHA-256 corpus identity")


@dataclass(frozen=True)
class AnnualPredictionWindow:
    role: Literal["train", "validation", "test", "shadow"]
    cutoff_year: int
    target_year: int

    def __post_init__(self) -> None:
        if self.target_year != self.cutoff_year + 1:
            raise ContractViolation("Annual product windows must predict the next calendar year")

    @property
    def cutoff(self) -> datetime:
        return datetime(self.cutoff_year, 12, 31, 23, 59, 59, 999999, tzinfo=UTC)


COLLABORATION_WINDOWS = (
    *(AnnualPredictionWindow("train", year, year + 1) for year in range(2017, 2022)),
    AnnualPredictionWindow("validation", 2022, 2023),
    AnnualPredictionWindow("test", 2023, 2024),
    AnnualPredictionWindow("shadow", 2024, 2025),
)


@dataclass(frozen=True)
class NewcomerCohortProtocol:
    train_years: tuple[int, ...] = (2017, 2018, 2019, 2020)
    validation_year: int = 2021
    test_year: int = 2022
    observation_days: int = 90
    horizon_days: int = 365
    minimum_supporter_prior_works: int = 3
    supporter_recent_days: int = 730
    candidate_hops: tuple[int, ...] = (2, 3)


NEWCOMER_PROTOCOL = NewcomerCohortProtocol()


def lodo_training_families(held_out_family: str) -> tuple[str, ...]:
    """Return independent source families, never academic datasets as separate domains."""

    if held_out_family not in DOMAIN_FAMILIES:
        raise ContractViolation(f"Unknown held-out domain family: {held_out_family}")
    return tuple(family for family in DOMAIN_FAMILIES if family != held_out_family)


def assert_cutoff_safe(
    event_times: list[datetime] | tuple[datetime, ...], *, cutoff: datetime
) -> None:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ContractViolation("cutoff must be timezone-aware")
    for value in event_times:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ContractViolation("event timestamps must be timezone-aware")
        if value > cutoff:
            raise ContractViolation("point-in-time sample contains a future event")


def collaboration_stratum(
    source: int,
    target: int,
    historical_pairs: set[tuple[int, int]],
) -> Literal["first_time", "repeated"]:
    pair = (source, target) if source <= target else (target, source)
    return "repeated" if pair in historical_pairs else "first_time"


__all__ = [
    "COLLABORATION_TASK",
    "COLLABORATION_WINDOWS",
    "DOMAIN_FAMILIES",
    "NEWCOMER_PROTOCOL",
    "NEWCOMER_TASK",
    "AnnualPredictionWindow",
    "NewcomerCohortProtocol",
    "SampleIdentity",
    "assert_cutoff_safe",
    "collaboration_stratum",
    "lodo_training_families",
]
