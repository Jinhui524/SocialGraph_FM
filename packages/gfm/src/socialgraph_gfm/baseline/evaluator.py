"""OGB-compatible evaluation and strict-track strata reporting."""

from __future__ import annotations

from typing import Any, Callable, Mapping


def hits_at_k(positive_scores: Any, negative_scores: Any, k: int) -> float:
    """OGB Hits@K semantics, including per-positive negative matrices."""

    import numpy as np

    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("Hits@K requires non-empty positive and negative scores")
    if not bool(np.isfinite(positive).all() and np.isfinite(negative).all()):
        raise ValueError("evaluation scores must be finite")
    if negative.ndim == 2 and negative.shape[0] == positive.shape[0]:
        effective_k = min(k, negative.shape[1])
        thresholds = np.partition(negative, -effective_k, axis=1)[:, -effective_k]
        return float(np.mean(positive > thresholds))
    flat_negative = negative.reshape(-1)
    effective_k = min(k, flat_negative.size)
    threshold = np.partition(flat_negative, -effective_k)[-effective_k]
    return float(np.mean(positive > threshold))


def ogb_hits(positive_scores: Any, negative_scores: Any) -> dict[str, float]:
    """Evaluate with OGB's official implementation and normalized result keys."""

    import torch
    from ogb.linkproppred import Evaluator

    positive = torch.as_tensor(positive_scores, dtype=torch.float32).view(-1)
    negative = torch.as_tensor(negative_scores, dtype=torch.float32)
    evaluator = Evaluator(name="ogbl-collab")
    results: dict[str, float] = {}
    for k in (10, 50, 100):
        evaluator.K = k
        output = evaluator.eval({"y_pred_pos": positive, "y_pred_neg": negative})
        results[f"hits@{k}"] = float(output[f"hits@{k}"])
    return results


def evaluate_scores(
    positive_scores: Any,
    negative_scores: Any,
    *,
    evaluator: Callable[[Any, Any], Mapping[str, float]] = ogb_hits,
) -> dict[str, float]:
    values = {str(key).lower(): float(value) for key, value in evaluator(
        positive_scores, negative_scores
    ).items()}
    if "hits@50" not in values:
        raise ValueError("baseline evaluator must return hits@50")
    return values


def stratified_positive_metrics(
    positive_scores: Any,
    negative_scores: Any,
    repeated_mask: Any,
) -> dict[str, dict[str, float | int]]:
    """Report first-time and repeated collaboration Hits@50 independently."""

    import numpy as np

    positive = np.asarray(positive_scores).reshape(-1)
    mask = np.asarray(repeated_mask, dtype=np.bool_).reshape(-1)
    if positive.shape[0] != mask.shape[0]:
        raise ValueError("strata mask must align with positive scores")
    output: dict[str, dict[str, float | int]] = {}
    for label, selected in (("first_time", ~mask), ("repeated", mask)):
        count = int(np.sum(selected))
        output[label] = {"count": count}
        if count:
            output[label]["hits@50"] = hits_at_k(positive[selected], negative_scores, 50)
    return output
