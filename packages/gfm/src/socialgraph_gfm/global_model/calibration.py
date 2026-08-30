"""Validation-only logit calibration and country-balanced threshold selection."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch
from torch import Tensor, nn


class BinaryLogitCalibrator(nn.Module):
    """Temperature plus intercept calibration for sampled binary training logits."""

    def __init__(self, *, temperature: float = 1.0, bias: float = 0.0) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(bias):
            raise ValueError("calibration temperature/bias must be finite and temperature positive")
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature), dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        if not logits.is_floating_point():
            raise ValueError("calibrator logits must be floating point")
        return logits / self.temperature + self.bias


@dataclass(frozen=True)
class CalibrationFit:
    calibrator: BinaryLogitCalibrator
    before_loss: float
    after_loss: float
    sample_count: int


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    mean_macro_f1: float
    per_country_macro_f1: Mapping[str, float]
    candidate_count: int


def _validation_values(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    values = logits.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    targets = labels.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if values.shape != targets.shape or values.numel() == 0:
        raise ValueError("calibration requires aligned nonempty logits and labels")
    if not bool(torch.isfinite(values).all()) or not bool(torch.isfinite(targets).all()):
        raise ValueError("calibration values must be finite")
    if not bool(torch.all((targets == 0) | (targets == 1))):
        raise ValueError("calibration labels must be binary")
    return values, targets


def fit_binary_logit_calibrator(
    validation_logits: Tensor,
    validation_labels: Tensor,
    *,
    max_iter: int = 100,
) -> CalibrationFit:
    """Fit only on validation values; returns identity if optimization regresses NLL."""

    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    logits, labels = _validation_values(validation_logits, validation_labels)
    if torch.unique(labels).numel() != 2:
        raise ValueError("temperature/bias calibration requires both validation classes")
    calibrator = BinaryLogitCalibrator().to(dtype=torch.float64)
    criterion = nn.BCEWithLogitsLoss()
    before = float(criterion(logits, labels))
    optimizer = torch.optim.LBFGS(
        calibrator.parameters(),
        lr=0.1,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(calibrator(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        after = float(criterion(calibrator(logits), labels))
    if (
        not math.isfinite(after)
        or not bool(torch.isfinite(calibrator.temperature))
        or not bool(torch.isfinite(calibrator.bias))
        or after > before + 1e-10
    ):
        calibrator = BinaryLogitCalibrator().to(dtype=torch.float64)
        after = before
    calibrator = calibrator.to(dtype=torch.float32).eval()
    calibrator.requires_grad_(False)
    return CalibrationFit(
        calibrator=calibrator,
        before_loss=before,
        after_loss=after,
        sample_count=logits.numel(),
    )


def calibration_state(calibrator: BinaryLogitCalibrator) -> dict[str, float | str]:
    """Return the two deployment scalars in a versioned JSON-safe form."""

    return {
        "schemaVersion": "socialgraph-fm.global-model-calibration/1.0",
        "temperature": float(calibrator.temperature.detach().cpu()),
        "bias": float(calibrator.bias.detach().cpu()),
    }


def binary_ece(
    logits: Tensor,
    labels: Tensor,
    *,
    bins: int = 15,
    calibrator: BinaryLogitCalibrator | None = None,
) -> float:
    """Expected calibration error over equal-width probability bins."""

    if bins < 2 or bins > 1000:
        raise ValueError("bins must be between 2 and 1000")
    values, targets = _validation_values(logits, labels)
    if calibrator is not None:
        with torch.no_grad():
            values = calibrator(values.to(dtype=torch.float32)).to(dtype=torch.float64)
    probabilities = torch.sigmoid(values)
    total = probabilities.numel()
    error = torch.zeros((), dtype=torch.float64)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (
            (probabilities >= lower) & (probabilities <= upper)
            if index == bins - 1
            else (probabilities >= lower) & (probabilities < upper)
        )
        count = int(mask.sum())
        if count:
            confidence = probabilities[mask].mean()
            accuracy = targets[mask].mean()
            error += (count / total) * torch.abs(confidence - accuracy)
    return float(error)


def _macro_f1(probabilities: Tensor, labels: Tensor, threshold: float) -> float:
    predictions = probabilities >= threshold
    truth = labels.to(torch.bool)
    class_scores = []
    for positive in (False, True):
        predicted_positive = predictions == positive
        actual_positive = truth == positive
        true_positive = int(torch.logical_and(predicted_positive, actual_positive).sum())
        false_positive = int(torch.logical_and(predicted_positive, ~actual_positive).sum())
        false_negative = int(torch.logical_and(~predicted_positive, actual_positive).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        class_scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return sum(class_scores) / 2.0


def select_country_balanced_threshold(
    logits_by_country: Mapping[str, Tensor],
    labels_by_country: Mapping[str, Tensor],
    *,
    calibrator: BinaryLogitCalibrator | None = None,
    candidates: Sequence[float] | None = None,
) -> ThresholdSelection:
    """Maximize the equal-country mean Macro-F1 using validation data only."""

    if not logits_by_country or tuple(logits_by_country) != tuple(labels_by_country):
        raise ValueError("logit and label mappings must have the same nonempty country order")
    candidate_values = (
        tuple(float(value) for value in candidates)
        if candidates is not None
        else tuple(index / 200 for index in range(1, 200))
    )
    if not candidate_values or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in candidate_values
    ):
        raise ValueError("threshold candidates must be finite and strictly between zero and one")
    candidate_values = tuple(sorted(set(candidate_values)))

    probabilities: dict[str, Tensor] = {}
    targets: dict[str, Tensor] = {}
    for country in logits_by_country:
        country_logits, country_labels = _validation_values(
            logits_by_country[country], labels_by_country[country]
        )
        if torch.unique(country_labels).numel() != 2:
            raise ValueError(f"threshold selection requires both classes for country {country!r}")
        if calibrator is not None:
            with torch.no_grad():
                country_logits = calibrator(country_logits.to(torch.float32)).to(torch.float64)
        probabilities[country] = torch.sigmoid(country_logits)
        targets[country] = country_labels

    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_scores: dict[str, float] = {}
    for threshold in candidate_values:
        scores = {
            country: _macro_f1(probabilities[country], targets[country], threshold)
            for country in probabilities
        }
        mean_score = sum(scores.values()) / len(scores)
        key = (mean_score, -abs(threshold - 0.5), -threshold)
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = threshold
            best_scores = scores
    assert best_key is not None
    return ThresholdSelection(
        threshold=best_threshold,
        mean_macro_f1=best_key[0],
        per_country_macro_f1=MappingProxyType(best_scores),
        candidate_count=len(candidate_values),
    )


__all__ = [
    "BinaryLogitCalibrator",
    "CalibrationFit",
    "ThresholdSelection",
    "binary_ece",
    "calibration_state",
    "fit_binary_logit_calibrator",
    "select_country_balanced_threshold",
]
