"""Cutoff-local structural features and transparent product candidate rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import AbstractSet, Mapping


PAIR_FEATURE_NAMES = (
    "common_neighbors",
    "adamic_adar",
    "resource_allocation",
    "topic_similarity",
    "topic_complementarity",
    "institution_diversity",
    "recency",
    "common_neighbor_change",
)


def _common_neighbors(
    source: int, target: int, neighbors: Mapping[int, AbstractSet[int]]
) -> set[int]:
    return set(neighbors.get(source, set())).intersection(neighbors.get(target, set()))


def collaboration_pair_features(
    source: int,
    target: int,
    *,
    neighbors: Mapping[int, AbstractSet[int]],
    topics: Mapping[int, AbstractSet[str]],
    institutions: Mapping[int, str | None],
    inactive_days: Mapping[int, int],
    previous_common_neighbor_count: int = 0,
) -> tuple[float, ...]:
    """Build the eight auditable features used beside learned pair embeddings."""

    common = _common_neighbors(source, target, neighbors)
    cn = float(len(common))
    aa = sum(
        1.0 / math.log(max(2, len(neighbors.get(node, set()))))
        for node in common
        if len(neighbors.get(node, set())) > 1
    )
    ra = sum(1.0 / len(neighbors.get(node, set())) for node in common if neighbors.get(node))
    source_topics = set(topics.get(source, set()))
    target_topics = set(topics.get(target, set()))
    union = source_topics | target_topics
    overlap = source_topics & target_topics
    topic_similarity = len(overlap) / len(union) if union else 0.0
    topic_complementarity = len(source_topics ^ target_topics) / len(union) if union else 0.0
    source_institution = institutions.get(source)
    target_institution = institutions.get(target)
    institution_diversity = float(
        bool(source_institution and target_institution and source_institution != target_institution)
    )
    most_recent = min(inactive_days.get(source, 10_000), inactive_days.get(target, 10_000))
    recency = math.exp(-max(0, most_recent) / 365.0)
    return (
        cn,
        float(aa),
        float(ra),
        float(topic_similarity),
        float(topic_complementarity),
        institution_diversity,
        float(recency),
        cn - float(previous_common_neighbor_count),
    )


@dataclass(frozen=True)
class SupporterCandidate:
    candidate_id: int
    prior_work_count: int
    inactive_days: int
    previously_collaborated: bool
    graph_distance: int | None
    adjacent_topic: bool
    adjacent_community: bool


def eligible_supporter(candidate: SupporterCandidate) -> bool:
    """Apply the frozen newcomer support candidate gate without judging a person."""

    reachable = candidate.graph_distance in (2, 3)
    return (
        candidate.prior_work_count >= 3
        and candidate.inactive_days <= 730
        and not candidate.previously_collaborated
        and (reachable or candidate.adjacent_topic or candidate.adjacent_community)
    )


__all__ = [
    "PAIR_FEATURE_NAMES",
    "SupporterCandidate",
    "collaboration_pair_features",
    "eligible_supporter",
]
