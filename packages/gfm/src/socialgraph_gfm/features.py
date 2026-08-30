"""Leakage-conscious, deterministic feature transformations."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class NumericStandardizer:
    mean: float
    scale: float

    @classmethod
    def fit(cls, values: Iterable[float | None]) -> NumericStandardizer:
        present = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("numeric features must contain only int, float, or null")
            present.append(float(value))
        if not present:
            raise ValueError("numeric fit selection contains no observed values")
        if not all(math.isfinite(value) for value in present):
            raise ValueError("numeric features must be finite")
        mean = sum(present) / len(present)
        variance = sum((value - mean) ** 2 for value in present) / len(present)
        scale = math.sqrt(variance)
        return cls(mean=mean, scale=scale if scale > 0.0 else 1.0)

    def transform(self, values: Iterable[float | None]) -> tuple[list[float], list[bool]]:
        transformed: list[float] = []
        missing: list[bool] = []
        for value in values:
            if value is None:
                missing.append(True)
                transformed.append(0.0)
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise TypeError("numeric features must contain only int, float, or null")
                missing.append(False)
                transformed.append((float(value) - self.mean) / self.scale)
        return transformed, missing


@dataclass(frozen=True)
class CategoryVocabulary:
    categories: tuple[str, ...]

    MISSING_INDEX = 0
    UNKNOWN_INDEX = 1

    @classmethod
    def fit(cls, values: Iterable[str | None]) -> CategoryVocabulary:
        present = []
        for value in values:
            if value is None:
                continue
            if not isinstance(value, str):
                raise TypeError("categorical features must contain only string or null")
            present.append(value)
        if not present:
            raise ValueError("categorical fit selection contains no observed values")
        return cls(categories=tuple(sorted(set(present))))

    def transform(self, values: Iterable[str | None]) -> tuple[list[int], list[bool]]:
        lookup = {value: index + 2 for index, value in enumerate(self.categories)}
        encoded: list[int] = []
        missing: list[bool] = []
        for value in values:
            is_missing = value is None
            missing.append(is_missing)
            if is_missing:
                encoded.append(self.MISSING_INDEX)
            else:
                if not isinstance(value, str):
                    raise TypeError("categorical features must contain only string or null")
                encoded.append(lookup.get(value, self.UNKNOWN_INDEX))
        return encoded, missing


@dataclass(frozen=True)
class EmbeddingReference:
    model_ref: str
    dimensions: int
    value_ref: str

    def __post_init__(self) -> None:
        if not self.model_ref or not self.value_ref or self.dimensions < 1:
            raise ValueError("embedding reference requires model, value and positive dimensions")
