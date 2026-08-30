"""SocialGraph-FM Core self-supervised masking and loss composition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


FIELD_MASK_RATE = 0.30
EDGE_MASK_RATE = 0.15
ALIGNMENT_WEIGHTS = frozenset({0.0, 0.02, 0.05})
_HASH_MODULUS = 2_147_483_647
_HASH_MULTIPLIER = 1_103_515_245
_NONCE_MULTIPLIER = 48_271


@dataclass(frozen=True)
class SourceValidationScores:
    weight_0: float
    weight_002: float
    weight_005: float

    def as_mapping(self) -> dict[float, float]:
        return {0.0: self.weight_0, 0.02: self.weight_002, 0.05: self.weight_005}


@dataclass(frozen=True)
class AlignmentSelection:
    selected_weight: float
    source_scores: dict[float, float]


def mask_feature_fields(
    shape: tuple[int, int], *, generator: torch.Generator, probability: float = FIELD_MASK_RATE
) -> Tensor:
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("field mask shape must contain positive node and field counts")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("field mask probability must be between zero and one")
    return torch.rand(shape, generator=generator, device=generator.device) < probability


def mask_paired_edges(
    edge_index: Tensor,
    *,
    generator: torch.Generator,
    pair_count: int,
    pair_inverse: Tensor,
    pair_representatives: Tensor,
    sampled: bool,
    probability: float = EDGE_MASK_RATE,
) -> tuple[Tensor, Tensor]:
    """Mask unordered endpoint pairs atomically, including both stored directions."""

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge mask probability must be between zero and one")
    if pair_inverse.shape != (edge_index.shape[1],):
        raise ValueError("paired-edge cache does not match edge topology")
    if pair_representatives.shape != pair_inverse.shape or pair_representatives.dtype != torch.bool:
        raise ValueError("pair representatives must be boolean and edge-aligned")
    if pair_count < 0:
        raise ValueError("pair count must be nonnegative")
    if pair_inverse.numel() and (
        bool(torch.any(pair_inverse < 0)) or bool(torch.any(pair_inverse >= pair_count))
    ):
        raise ValueError("sampled pair ID is outside the global pair cache")
    if edge_index.shape[1] == 0:
        return edge_index, edge_index.new_empty((0, 2))
    if sampled:
        nonce = torch.randint(
            _HASH_MODULUS,
            (1,),
            generator=generator,
            dtype=torch.long,
            device=edge_index.device,
        )
        edge_selected = stateless_pair_decisions(
            pair_inverse,
            nonce=nonce,
            probability=probability,
        )
    else:
        selected = torch.rand(
            (pair_count,),
            generator=generator,
            device=edge_index.device,
        ) < probability
        edge_selected = selected[pair_inverse]
    retained = edge_index[:, ~edge_selected]
    masked = edge_index[:, edge_selected & pair_representatives].t().contiguous()
    return retained, masked


def stateless_pair_decisions(
    pair_ids: Tensor,
    *,
    nonce: int | Tensor,
    probability: float,
) -> Tensor:
    """Hash sampled global pair IDs into deterministic Bernoulli decisions."""

    if pair_ids.dtype != torch.long:
        raise ValueError("pair IDs must use torch.long")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("edge mask probability must be between zero and one")
    nonce_tensor = torch.as_tensor(nonce, dtype=torch.long, device=pair_ids.device)
    pair_residue = torch.remainder(pair_ids, _HASH_MODULUS)
    nonce_residue = torch.remainder(nonce_tensor, _HASH_MODULUS)
    mixed = torch.remainder(pair_residue * _HASH_MULTIPLIER, _HASH_MODULUS)
    nonce_term = torch.remainder(nonce_residue * _NONCE_MULTIPLIER, _HASH_MODULUS)
    hashed = torch.remainder(mixed + nonce_term, _HASH_MODULUS)
    return hashed < int(probability * _HASH_MODULUS)


def domain_alignment_loss(source: Tensor, target: Tensor) -> Tensor:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("alignment inputs must be matrices with equal widths")
    return torch.mean((source.mean(dim=0) - target.mean(dim=0)).square())


def combine_objective_losses(
    *,
    field_loss: Tensor,
    edge_loss: Tensor,
    alignment_loss: Tensor,
    alignment_weight: float,
) -> Tensor:
    if alignment_weight not in ALIGNMENT_WEIGHTS:
        raise ValueError("alignment weight must be one of 0, 0.02, or 0.05")
    return field_loss + 0.5 * edge_loss + alignment_weight * alignment_loss


def select_alignment_weight(source_validation: SourceValidationScores) -> AlignmentSelection:
    """Choose only from source-validation metrics; target metrics are not accepted."""

    scores = source_validation.as_mapping()
    selected = max(sorted(scores), key=lambda weight: scores[weight])
    return AlignmentSelection(selected_weight=selected, source_scores=scores)


__all__ = [
    "ALIGNMENT_WEIGHTS",
    "AlignmentSelection",
    "EDGE_MASK_RATE",
    "FIELD_MASK_RATE",
    "SourceValidationScores",
    "combine_objective_losses",
    "domain_alignment_loss",
    "mask_feature_fields",
    "mask_paired_edges",
    "select_alignment_weight",
    "stateless_pair_decisions",
]
