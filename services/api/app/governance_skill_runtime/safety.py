"""Payload bounding, redaction, and safe error helpers for Governance skills."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from ..gfm_client import GfmProxyError
from ..gfm_hashing import canonical_sha256
from ..provider import ProviderFailure

_MAX_SKILL_RESULT_BYTES = 1_000_000
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/]|\\\\[^\\\s]+[\\/])")
_POSIX_SENSITIVE_PATH = re.compile(
    r"(?i)(?:^|[\s`'\"(])/(?:home|users|root|etc|var|tmp|opt|srv)/"
)
_SENSITIVE_TEXT = re.compile(
    r"(?i)(?:file://|\.env\b|session\.token\b|llm_api_key\b|api[_-]?key\s*[:=]|"
    r"password\s*[:=]|secret\s*[:=])"
)


def _safe_error_code(error: Exception) -> str:
    return error.code if isinstance(error, GfmProxyError) else "GOVERNANCE_SKILL_FAILED"


def _safe_provider_reason_code(error: Exception) -> str:
    if isinstance(error, ProviderFailure) and re.fullmatch(r"[A-Z][A-Z0-9_]{1,99}", error.code):
        return error.code
    if isinstance(error, ValidationError):
        return "LLM_INVALID_RESPONSE"
    return "LLM_NARRATION_FAILED"


def _contains_sensitive_text(value: str) -> bool:
    return bool(
        _WINDOWS_ABSOLUTE_PATH.search(value)
        or _POSIX_SENSITIVE_PATH.search(value)
        or _SENSITIVE_TEXT.search(value)
    )


def _ensure_no_vector_payload(value: Any, key: str = "") -> None:
    lowered = key.lower().replace("_", "")
    if isinstance(value, (dict, list, tuple)) and any(
        marker in lowered for marker in ("embedding", "featurevector", "vector768", "vector256")
    ):
        raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")
    if isinstance(value, dict):
        for child_key, child in value.items():
            _ensure_no_vector_payload(child, str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _ensure_no_vector_payload(child, key)


def _bounded_result(result: dict[str, Any]) -> None:
    _ensure_no_vector_payload(result)
    encoded = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > _MAX_SKILL_RESULT_BYTES:
        raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")


def _llm_summary(value: Any, *, key: str = "", depth: int = 0) -> Any:
    """Remove graph rows and vectors before any result can reach the provider."""

    lowered = key.lower().replace("_", "")
    if depth > 5:
        return None
    if any(
        marker in lowered
        for marker in (
            "embedding",
            "feature",
            "vector",
            "rawedge",
            "rawrelation",
            "allnode",
        )
    ):
        return None
    if lowered in {"nodes", "edges", "subgraph", "findings"}:
        if isinstance(value, (list, tuple, dict)):
            return {"omitted": True, "contentHash": canonical_sha256(value)}
        return None
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for child_key, child in value.items():
            summarized = _llm_summary(child, key=str(child_key), depth=depth + 1)
            if summarized is not None:
                output[str(child_key)] = summarized
        return output
    if isinstance(value, (list, tuple)):
        return [
            summarized
            for item in value[:10]
            if (summarized := _llm_summary(item, key=key, depth=depth + 1)) is not None
        ]
    if isinstance(value, str):
        if _contains_sensitive_text(value):
            return None
        return value[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def _cited_hashes(value: Any) -> tuple[str, ...]:
    hashes = sorted(set(re.findall(r"\b[0-9a-f]{64}\b", json.dumps(value))))
    return tuple(hashes[:50])

__all__ = [
    "_bounded_result",
    "_cited_hashes",
    "_contains_sensitive_text",
    "_ensure_no_vector_payload",
    "_llm_summary",
    "_safe_error_code",
    "_safe_provider_reason_code",
]
