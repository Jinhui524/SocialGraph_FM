"""Versioned, immutable configuration loading for SocialGraph-FM Core."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from ..canonical import canonical_sha256
from ..errors import ContractViolation


EXPECTED_SCHEMA = "gfm.pretrain-config/1.0"
EXPECTED_CONFIG_ID = "socialgraph-core"
OPENALEX_SPEC_ID = "graph-ai"


def _bundled_config() -> Path:
    candidate = resources.files("socialgraph_gfm").joinpath(
        "resources/configs/socialgraph-core.json"
    )
    if candidate.is_file():
        return Path(str(candidate))
    source_candidate = Path(__file__).resolve().parents[3] / "configs" / "socialgraph-core.json"
    if source_candidate.is_file():
        return source_candidate
    raise ContractViolation("The pinned SocialGraph-FM Core config is unavailable")


def load_core_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the checked-in formal configuration."""

    selected = Path(path).expanduser().resolve() if path is not None else _bundled_config()
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractViolation(f"Cannot load GFM config: {error}") from error
    if not isinstance(payload, dict):
        raise ContractViolation("GFM config must be a JSON object")
    if payload.get("schemaVersion") != EXPECTED_SCHEMA:
        raise ContractViolation(f"GFM config schema must be {EXPECTED_SCHEMA}")
    if payload.get("configId") != EXPECTED_CONFIG_ID:
        raise ContractViolation(f"GFM config ID must be {EXPECTED_CONFIG_ID}")
    domains = payload.get("domains")
    if not isinstance(domains, list) or len(domains) != 3:
        raise ContractViolation("GFM formal config must declare exactly three domains")
    families = {item.get("domainFamily") for item in domains if isinstance(item, dict)}
    if families != {"academic-collaboration", "software-activity", "online-community"}:
        raise ContractViolation("GFM config must contain three independent domain families")
    weights = payload.get("product", {}).get("collaborationRerankWeights", {})
    if not isinstance(weights, dict) or abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise ContractViolation("Collaboration rerank weights must sum to one")
    payload["configHash"] = canonical_sha256(payload)
    payload["runKind"] = "formal"
    return payload


def load_openalex_spec(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        source = Path(__file__).resolve().parents[3] / "configs" / "openalex-graph-ai.json"
        packaged = resources.files("socialgraph_gfm").joinpath(
            "resources/configs/openalex-graph-ai.json"
        )
        selected = source if source.is_file() else Path(str(packaged))
    else:
        selected = Path(path).expanduser().resolve()
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractViolation(f"Cannot load OpenAlex spec: {error}") from error
    if not isinstance(payload, dict):
        raise ContractViolation("OpenAlex spec must be a JSON object")
    if payload.get("schemaVersion") != "gfm.openalex-spec/1.0":
        raise ContractViolation("OpenAlex spec schema is unsupported")
    if payload.get("specId") != OPENALEX_SPEC_ID:
        raise ContractViolation(f"OpenAlex spec ID must be {OPENALEX_SPEC_ID}")
    clusters = payload.get("topicClusters")
    if not isinstance(clusters, list) or len(clusters) != 3:
        raise ContractViolation("OpenAlex spec must contain three topic clusters")
    total = sum(int(item.get("maximumWorks", 0)) for item in clusters if isinstance(item, dict))
    if total != int(payload.get("maximumUniqueWorks", -1)) or total != 200_000:
        raise ContractViolation("OpenAlex cluster caps must total exactly 200000")
    if payload.get("workTypes") != ["article", "preprint"]:
        raise ContractViolation("OpenAlex current API work types must be article/preprint")
    compatibility = payload.get("workTypeCompatibility")
    if (
        not isinstance(compatibility, dict)
        or compatibility.get("requestedCategories")
        != ["article", "preprint", "proceedings-article"]
        or not isinstance(compatibility.get("documentation"), str)
    ):
        raise ContractViolation("OpenAlex proceedings-article compatibility is unbound")
    forbidden = set(payload.get("forbiddenFields", ()))
    selected_fields = set(payload.get("workSelect", ()))
    if forbidden.intersection(selected_fields):
        raise ContractViolation("OpenAlex select list contains a future/leakage-blocked field")
    payload["specHash"] = canonical_sha256(payload)
    return payload


def apply_exploratory_overrides(
    config: Mapping[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Record explicit top-level overrides while making the run non-acceptable."""

    result = dict(config)
    result.pop("configHash", None)
    if overrides:
        unknown = set(overrides).difference(result)
        if unknown:
            raise ContractViolation(f"Unknown GFM override keys: {sorted(unknown)}")
        for key, value in overrides.items():
            current = result[key]
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                nested_unknown = set(value).difference(current)
                if nested_unknown:
                    raise ContractViolation(
                        f"Unknown GFM override keys below {key}: {sorted(nested_unknown)}"
                    )
                result[key] = {**current, **value}
            else:
                result[key] = value
        result["runKind"] = "exploratory"
        result["overrides"] = dict(overrides)
    else:
        result["runKind"] = str(config.get("runKind", "formal"))
    result["configHash"] = canonical_sha256(
        {
            key: value
            for key, value in result.items()
            if key not in {"configHash", "runKind", "overrides"}
        }
    )
    return result
