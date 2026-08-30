"""Versioned core training limits and loader defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .objectives import SourceValidationScores, select_alignment_weight


@dataclass(frozen=True)
class TrainingConfig:
    preset: Literal["smoke", "dev", "formal"]
    max_steps: int
    min_steps: int
    validation_interval: int = 250
    patience: int = 8
    hidden_dim: int = 128
    encoder_layers: int = 3
    dropout: float = 0.2
    field_mask_rate: float = 0.30
    edge_mask_rate: float = 0.15
    alignment_weight: float = 0.0
    alignment_source_scores: tuple[float, float, float] | None = None
    full_batch_edge_threshold: int = 100_000
    node_batch_size: int = 1024
    edge_batch_size: int = 2048
    fanout: tuple[int, int, int] = (15, 10, 5)
    amp: bool = True
    gradient_accumulation: int = 1
    timeout_seconds: float = 6 * 60 * 60
    moe_enabled: bool = False
    future_moe_capability_version: str = "socialgraph-fm.core-moe/future-1-disabled"

    def __post_init__(self) -> None:
        limits = {"smoke": (0, 20), "dev": (0, 2_000), "formal": (2_000, 10_000)}
        required_minimum, maximum = limits[self.preset]
        if not required_minimum <= self.min_steps <= self.max_steps <= maximum:
            raise ValueError(f"invalid {self.preset} optimizer-step limits")
        if self.hidden_dim != 128 or self.encoder_layers != 3 or self.dropout != 0.2:
            raise ValueError("core encoder shape and dropout are fixed")
        if self.field_mask_rate != 0.30 or self.edge_mask_rate != 0.15:
            raise ValueError("core masking rates are fixed")
        if self.alignment_weight not in {0.0, 0.02, 0.05}:
            raise ValueError("alignment weight must be one of 0, 0.02, or 0.05")
        if self.alignment_weight != 0.0 and self.alignment_source_scores is None:
            raise ValueError("nonzero alignment requires source-validation selection")
        if self.alignment_source_scores is not None:
            source = SourceValidationScores(*self.alignment_source_scores)
            selected = select_alignment_weight(source).selected_weight
            if selected != self.alignment_weight:
                raise ValueError("alignment weight is not the selected source-validation candidate")
        if self.validation_interval != 250 or self.patience != 8:
            raise ValueError("core validation interval and patience are fixed")
        if self.gradient_accumulation < 1:
            raise ValueError("gradient accumulation must be positive")
        if self.moe_enabled:
            raise ValueError("MoE is unavailable in core core")

    @classmethod
    def smoke(cls, *, max_steps: int = 20) -> TrainingConfig:
        return cls(preset="smoke", min_steps=0, max_steps=max_steps)

    @classmethod
    def dev(cls, *, max_steps: int = 2_000) -> TrainingConfig:
        return cls(preset="dev", min_steps=0, max_steps=max_steps)

    @classmethod
    def smoke_from_source_validation(
        cls, source_validation: SourceValidationScores, *, max_steps: int = 20
    ) -> TrainingConfig:
        selection = select_alignment_weight(source_validation)
        scores = selection.source_scores
        return cls(
            preset="smoke",
            min_steps=0,
            max_steps=max_steps,
            alignment_weight=selection.selected_weight,
            alignment_source_scores=(scores[0.0], scores[0.02], scores[0.05]),
        )

    @classmethod
    def formal(
        cls, *, max_steps: int = 10_000, min_steps: int = 2_000
    ) -> TrainingConfig:
        return cls(preset="formal", min_steps=min_steps, max_steps=max_steps)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


__all__ = ["TrainingConfig"]
