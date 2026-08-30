"""Dependency-neutral canonical JSON used by the GFM HTTP contract."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


def _number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite values are not canonical JSON")
    if value == 0:
        return "0"
    negative = value < 0
    raw = repr(abs(value)).lower()
    mantissa, marker, exponent_text = raw.partition("e")
    exponent = int(exponent_text) if marker else 0
    whole, dot, fraction = mantissa.partition(".")
    raw_digits = whole + (fraction if dot else "")
    leading_zeros = len(raw_digits) - len(raw_digits.lstrip("0"))
    digits = raw_digits.lstrip("0") or "0"
    decimal_position = len(whole) + exponent - leading_zeros
    magnitude = abs(value)
    if 1e-6 <= magnitude < 1e21:
        if decimal_position <= 0:
            rendered = "0." + "0" * (-decimal_position) + digits
        elif decimal_position >= len(digits):
            rendered = digits + "0" * (decimal_position - len(digits))
        else:
            rendered = digits[:decimal_position] + "." + digits[decimal_position:]
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
    else:
        digits = digits.rstrip("0") or "0"
        coefficient = digits[0] + (("." + digits[1:]) if len(digits) > 1 else "")
        scientific_exponent = decimal_position - 1
        sign = "+" if scientific_exponent >= 0 else "-"
        rendered = f"{coefficient}e{sign}{abs(scientific_exponent)}"
    return ("-" if negative else "") + rendered


def _normalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("canonical datetime must include a timezone")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite values are not canonical JSON")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("canonical JSON keys must be strings")
        return {key: _normalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    raise ValueError(f"value of type {type(value).__name__} is not canonical JSON")


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, list):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_encode(key)}:{_encode(item)}" for key, item in value.items()) + "}"
    raise ValueError("unsupported canonical JSON value")


def canonical_json(value: Any) -> str:
    return _encode(_normalize(value))


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


__all__ = ["canonical_json", "canonical_sha256"]
