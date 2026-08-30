"""Validation-only temperature scaling for product probability outputs."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature


def fit_temperature(
    validation_logits: Tensor,
    validation_labels: Tensor,
    *,
    maximum_iterations: int = 50,
) -> TemperatureScaler:
    logits = validation_logits.detach().float().reshape(-1)
    labels = validation_labels.detach().float().reshape(-1)
    if logits.shape != labels.shape or not logits.numel():
        raise ValueError("temperature scaling requires aligned nonempty validation values")
    if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(labels).all()):
        raise ValueError("temperature inputs must be finite")
    if not bool(torch.all((labels == 0) | (labels == 1))):
        raise ValueError("temperature labels must be binary")
    scaler = TemperatureScaler().to(logits.device)
    optimizer = torch.optim.LBFGS(
        scaler.parameters(), lr=0.1, max_iter=maximum_iterations, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(scaler(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    scaler.eval()
    if not bool(torch.isfinite(scaler.temperature)):
        raise RuntimeError("temperature optimization produced a non-finite result")
    return scaler


__all__ = ["TemperatureScaler", "fit_temperature"]
