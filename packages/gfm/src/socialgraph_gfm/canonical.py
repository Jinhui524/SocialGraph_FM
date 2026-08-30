"""Locale-independent canonical JSON and SHA-256 helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from .errors import ContractViolation


def _normalise(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _normalise(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _normalise(dataclasses.asdict(cast(Any, value)))
    if isinstance(value, Enum):
        return _normalise(value.value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ContractViolation("Canonical datetimes must include a timezone")
        utc = value.astimezone(UTC)
        return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation("NaN and Infinity are forbidden in canonical JSON")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ContractViolation("Canonical JSON object keys must be strings")
        # Python's default string ordering is Unicode code-point ordering and is locale free.
        return {key: _normalise(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    raise ContractViolation(f"Value of type {type(value).__name__} is not canonical JSON")


def canonical_json(value: Any) -> str:
    """Return UTF-8 canonical JSON aligned with ECMAScript ``JSON.stringify`` numbers.

    Python and ECMAScript use the same shortest round-trippable binary64 digits, but
    choose different fixed/scientific notation thresholds.  Rendering numbers here
    avoids identities such as ``1.0``/``1`` and ``1e-07``/``1e-7`` drifting across the
    Python API and JavaScript workbench.
    """

    return _encode(_normalise(value))


def _ecmascript_number(value: float) -> str:
    if not math.isfinite(value):
        raise ContractViolation("NaN and Infinity are forbidden in canonical JSON")
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

    # ECMAScript uses fixed notation for [1e-6, 1e21), scientific otherwise.
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
        return _ecmascript_number(value)
    if isinstance(value, list):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            f"{_encode(key)}:{_encode(item)}" for key, item in value.items()
        ) + "}"
    raise ContractViolation(f"Value of type {type(value).__name__} is not canonical JSON")


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
