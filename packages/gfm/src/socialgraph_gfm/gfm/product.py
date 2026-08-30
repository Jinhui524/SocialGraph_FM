"""Deterministic, review-first product artifacts derived from accepted GFM output."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pydantic import Field

from .contracts import (
    CollaborationRerankComponents,
    GfmContractModel,
    GovernanceCaseArtifact,
)


class CollaborationCandidate(GfmContractModel):
    candidate_id: str = Field(alias="candidateId", min_length=1, max_length=1000)
    components: CollaborationRerankComponents
    evidence_refs: tuple[str, ...] = Field((), alias="evidenceRefs")
    eligible: bool = True


class RerankedCollaborationCandidate(GfmContractModel):
    candidate_id: str = Field(alias="candidateId", min_length=1, max_length=1000)
    rank: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)
    components: CollaborationRerankComponents
    evidence_refs: tuple[str, ...] = Field((), alias="evidenceRefs")


def collaboration_rerank_score(
    components: CollaborationRerankComponents | dict[str, Any],
) -> float:
    """Apply the versioned 70/15/10/5 policy from the checked core config."""

    checked = CollaborationRerankComponents.model_validate(components)
    return checked.weighted_score()


def rerank_collaboration_candidates(
    candidates: Iterable[CollaborationCandidate | dict[str, Any]],
) -> tuple[RerankedCollaborationCandidate, ...]:
    """Filter ineligible candidates and rank deterministically by score then ID."""

    checked = [CollaborationCandidate.model_validate(candidate) for candidate in candidates]
    identifiers = [candidate.candidate_id for candidate in checked]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidateId values must be unique")
    ordered = sorted(
        (candidate for candidate in checked if candidate.eligible),
        key=lambda item: (-item.components.weighted_score(), item.candidate_id),
    )
    return tuple(
        RerankedCollaborationCandidate(
            candidateId=candidate.candidate_id,
            rank=rank,
            score=candidate.components.weighted_score(),
            components=candidate.components,
            evidenceRefs=candidate.evidence_refs,
        )
        for rank, candidate in enumerate(ordered, start=1)
    )


def build_governance_case_artifact(**values: Any) -> GovernanceCaseArtifact:
    """Build an immutable, hash-bound product artifact with policy validation."""

    return GovernanceCaseArtifact.create(**values)


__all__ = [
    "CollaborationCandidate",
    "RerankedCollaborationCandidate",
    "build_governance_case_artifact",
    "collaboration_rerank_score",
    "rerank_collaboration_candidates",
]
