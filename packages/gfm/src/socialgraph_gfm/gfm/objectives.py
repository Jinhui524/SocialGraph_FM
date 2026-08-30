"""The seven immutable SocialGraph-FM Core pretraining objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FixedObjectiveWeights:
    temporal_next_event: float = 1.0
    masked_attribute: float = 0.5
    masked_relation_type: float = 0.25
    log_time_delta: float = 0.25
    text_structure_alignment: float = 0.25
    cross_domain_distribution_alignment: float = 0.05
    moe_route_balance: float = 0.01

    def __post_init__(self) -> None:
        actual = (
            self.temporal_next_event,
            self.masked_attribute,
            self.masked_relation_type,
            self.log_time_delta,
            self.text_structure_alignment,
            self.cross_domain_distribution_alignment,
            self.moe_route_balance,
        )
        if actual != (1.0, 0.5, 0.25, 0.25, 0.25, 0.05, 0.01):
            raise ValueError("SocialGraph-FM Core objective weights are frozen")


OBJECTIVE_WEIGHTS = FixedObjectiveWeights()


@dataclass(frozen=True)
class LossBundle:
    total: Tensor
    components: Mapping[str, Tensor]

    def detached(self) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in self.components.items()}


def _zero(anchor: Tensor) -> Tensor:
    return anchor.sum() * 0.0


def masked_attribute_loss(
    predictions: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
    *,
    anchor: Tensor,
    text_modality: str | None,
) -> Tensor:
    losses: list[Tensor] = []
    if set(predictions) != set(targets) or set(targets) != set(masks):
        raise ValueError("attribute predictions, targets and masks must have identical keys")
    for name in sorted(targets):
        prediction, target = predictions[name], targets[name]
        selected = masks[name].reshape(-1)
        if selected.dtype != torch.bool or prediction.shape != target.shape:
            raise ValueError(f"invalid masked attribute tensors for {name!r}")
        if selected.shape[0] != target.shape[0]:
            raise ValueError(f"attribute mask for {name!r} does not align with nodes")
        if bool(selected.any()):
            if name == text_modality:
                cosine = nn.functional.cosine_similarity(
                    prediction[selected], target[selected], dim=-1
                )
                losses.append((1.0 - cosine).mean())
            else:
                losses.append(
                    nn.functional.smooth_l1_loss(prediction[selected], target[selected])
                )
    return torch.stack(losses).mean() if losses else _zero(anchor)


def temporal_next_event_loss(positive_logits: Tensor, negative_logits: Tensor) -> Tensor:
    """Sampled softmax: one future event versus K exact mixed non-events."""

    positive = positive_logits.reshape(-1)
    if not positive.numel():
        raise ValueError("temporal next-event loss needs at least one positive score")
    if negative_logits.ndim == 2:
        negative = negative_logits
        if negative.shape[0] != positive.shape[0]:
            raise ValueError("rank-2 negatives must contain one row per positive query")
    else:
        flattened = negative_logits.reshape(-1)
        if not flattened.numel() or flattened.numel() % positive.numel():
            raise ValueError("flat negatives must contain a fixed K per positive query")
        negative = flattened.reshape(positive.shape[0], -1)
    if not negative.shape[1]:
        raise ValueError("temporal next-event loss requires negative candidates")
    candidate_logits = torch.cat((positive.reshape(-1, 1), negative), dim=1)
    labels = torch.zeros(positive.shape[0], dtype=torch.long, device=positive.device)
    return nn.functional.cross_entropy(candidate_logits, labels)


def masked_relation_type_loss(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    selected = mask.reshape(-1)
    if (
        logits.ndim != 2
        or labels.dtype != torch.long
        or selected.dtype != torch.bool
        or labels.reshape(-1).shape[0] != logits.shape[0]
        or selected.shape[0] != logits.shape[0]
    ):
        raise ValueError("masked relation labels must align with rank-2 logits")
    if not bool(selected.any()):
        return _zero(logits)
    return nn.functional.cross_entropy(logits[selected], labels.reshape(-1)[selected])


def log_time_delta_loss(predictions: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    prediction = predictions.reshape(-1)
    target = targets.reshape(-1)
    selected = mask.reshape(-1)
    if (
        not targets.is_floating_point()
        or selected.dtype != torch.bool
        or prediction.shape != target.shape
        or selected.shape != target.shape
        or not bool(torch.isfinite(target).all())
        or bool((target < 0).any())
    ):
        raise ValueError("time-delta targets must be finite, non-negative and aligned")
    if not bool(selected.any()):
        return _zero(prediction)
    return nn.functional.smooth_l1_loss(prediction[selected], torch.log1p(target[selected]))


def text_structure_alignment_loss(
    text_embeddings: Tensor | None,
    structure_embeddings: Tensor,
    text_mask: Tensor | None,
) -> Tensor:
    if text_embeddings is None or text_mask is None:
        return _zero(structure_embeddings)
    selected = text_mask.reshape(-1)
    if (
        selected.dtype != torch.bool
        or text_embeddings.shape != structure_embeddings.shape
        or selected.shape[0] != structure_embeddings.shape[0]
    ):
        raise ValueError("text embeddings and visibility mask must align with structure")
    if not bool(selected.any()):
        return _zero(structure_embeddings)
    cosine = nn.functional.cosine_similarity(
        text_embeddings[selected], structure_embeddings[selected], dim=-1
    )
    return (1.0 - cosine).mean()


def embedding_distribution_moments(embeddings: Tensor) -> Tensor:
    """Return detached-compatible [mean, standard deviation] domain moments."""

    if embeddings.ndim != 2 or not embeddings.shape[0]:
        raise ValueError("domain embeddings must be nonempty [N, H]")
    return torch.stack((embeddings.mean(dim=0), embeddings.var(dim=0, unbiased=False).sqrt()))


def cross_domain_distribution_alignment_loss(
    embeddings: Tensor,
    reference_moments: Tensor | None,
) -> Tensor:
    if reference_moments is None:
        return _zero(embeddings)
    moments = embedding_distribution_moments(embeddings)
    reference = reference_moments.to(device=embeddings.device, dtype=embeddings.dtype)
    if reference.shape != moments.shape or not bool(torch.isfinite(reference).all()):
        raise ValueError("cross-domain reference moments must be finite [2, H]")
    return nn.functional.smooth_l1_loss(moments, reference)


def moe_route_balance_loss(expert_weights: Tensor, *, moe_enabled: bool) -> Tensor:
    if not moe_enabled:
        return _zero(expert_weights)
    if expert_weights.ndim != 2 or expert_weights.shape[1] != 3 or not expert_weights.shape[0]:
        raise ValueError("router weights must describe two experts plus one null expert")
    mean_route = expert_weights.mean(dim=0)
    target = torch.full_like(mean_route, 1.0 / 3.0)
    return torch.mean((mean_route - target) ** 2)


def compute_fixed_multiloss(
    *,
    attribute_predictions: Mapping[str, Tensor],
    attribute_targets: Mapping[str, Tensor],
    attribute_masks: Mapping[str, Tensor],
    positive_link_logits: Tensor,
    negative_link_logits: Tensor,
    relation_logits: Tensor,
    relation_labels: Tensor,
    relation_mask: Tensor,
    log_time_delta_predictions: Tensor,
    time_delta_targets: Tensor,
    time_delta_mask: Tensor,
    text_embeddings: Tensor | None,
    structure_embeddings: Tensor,
    text_mask: Tensor | None,
    cross_domain_reference: Tensor | None,
    expert_weights: Tensor,
    moe_enabled: bool,
    text_modality: str | None,
    weights: FixedObjectiveWeights = OBJECTIVE_WEIGHTS,
) -> LossBundle:
    """Compute exactly the seven weighted losses pinned in the formal v1 config."""

    components = {
        "temporal_next_event": temporal_next_event_loss(
            positive_link_logits, negative_link_logits
        ),
        "masked_attribute": masked_attribute_loss(
            attribute_predictions,
            attribute_targets,
            attribute_masks,
            anchor=structure_embeddings,
            text_modality=text_modality,
        ),
        "masked_relation_type": masked_relation_type_loss(
            relation_logits, relation_labels, relation_mask
        ),
        "log_time_delta": log_time_delta_loss(
            log_time_delta_predictions, time_delta_targets, time_delta_mask
        ),
        "text_structure_alignment": text_structure_alignment_loss(
            text_embeddings, structure_embeddings, text_mask
        ),
        "cross_domain_distribution_alignment": cross_domain_distribution_alignment_loss(
            structure_embeddings, cross_domain_reference
        ),
        "moe_route_balance": moe_route_balance_loss(
            expert_weights, moe_enabled=moe_enabled
        ),
    }
    total = (
        weights.temporal_next_event * components["temporal_next_event"]
        + weights.masked_attribute * components["masked_attribute"]
        + weights.masked_relation_type * components["masked_relation_type"]
        + weights.log_time_delta * components["log_time_delta"]
        + weights.text_structure_alignment * components["text_structure_alignment"]
        + weights.cross_domain_distribution_alignment
        * components["cross_domain_distribution_alignment"]
        + weights.moe_route_balance * components["moe_route_balance"]
    )
    if not bool(torch.isfinite(total)):
        raise RuntimeError("SocialGraph-FM Core objective is not finite")
    return LossBundle(total=total, components={**components, "total": total})
