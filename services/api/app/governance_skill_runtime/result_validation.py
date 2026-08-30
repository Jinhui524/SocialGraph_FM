"""Fail-closed validation for isolated Governance GFM result envelopes."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..gfm_client import GfmProxyError
from ..gfm_hashing import canonical_sha256
from ..governance_skills_schemas import KnowledgeItem
from .safety import _bounded_result

_RESULT_CONTRACT_ALIAS = {
    "get_evidence_subgraph": "trace_evidence",
    "discover_coordination_groups": "summarize_groups",
    "rank_coordination_relations": "inspect_relations",
    "retrieve_similar_cases": "find_similar_cases",
    "get_model_dataset_cards": "get_model_card",
    "draft_review_report": "draft_report",
}


def _required_keys(result: dict[str, Any], required: set[str], code: str) -> None:
    if set(result) != required:
        raise GfmProxyError(502, code)


def _validate_result(command: str, result: dict[str, Any]) -> None:
    command = _RESULT_CONTRACT_ALIAS.get(command, command)
    if command == "inspect_graph":
        base = {
            "nodeCount",
            "fusedEdgeCount",
            "componentCount",
            "isolateCount",
            "modalities",
            "relationCounts",
            "scopeNodeIds",
            "inspectionHash",
        }
        extended = base | {"runId", "distribution", "topCandidates", "candidateLimit"}
        if set(result) not in (base, extended):
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
        if set(result) == extended:
            distribution = result["distribution"]
            candidates = result["topCandidates"]
            limit = result["candidateLimit"]
            if (
                not isinstance(distribution, dict)
                or set(distribution) != {"low", "review", "high", "predictedPositive", "total"}
                or any(not isinstance(value, int) or value < 0 for value in distribution.values())
                or not isinstance(limit, int)
                or not 1 <= limit <= 5
                or not isinstance(candidates, list)
                or len(candidates) > limit
            ):
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")
            allowed_candidate = {
                "nodeId",
                "label",
                "score",
                "rank",
                "riskBand",
                "structureMissing",
                "communityId",
            }
            if any(
                not isinstance(candidate, dict)
                or not {"nodeId", "score", "riskBand"} <= set(candidate)
                or not set(candidate) <= allowed_candidate
                for candidate in candidates
            ):
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
    required: dict[str, set[str]] = {
        "summarize_groups": {
            "schemaVersion",
            "runId",
            "items",
            "total",
            "offset",
            "limit",
            "pageHash",
        },
        "inspect_relations": {
            "runId",
            "items",
            "total",
            "offset",
            "limit",
            "relationKind",
            "modalities",
            "pageHash",
        },
        "find_similar_cases": {
            "query",
            "items",
            "weights",
            "indexHash",
            "retrievalHash",
        },
        "get_model_card": {"modelCard", "datasetCard", "inputContractCard", "cardHash"},
        "draft_report": {
            "format",
            "content",
            "caseId",
            "citedHashes",
            "generatedWithoutLlm",
            "draftHash",
        },
        "run_governance_analysis": {"confirmationPlan"},
        "search_knowledge": {"items", "indexHash", "searchHash"},
    }
    if command in required:
        _required_keys(result, required[command], "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
        if command == "inspect_relations":
            relation_kind = result["relationKind"]
            items = result["items"]
            modalities = result["modalities"]
            if relation_kind not in {"factual", "potential"} or not isinstance(items, list):
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
            if relation_kind == "factual" and any(
                not isinstance(item, dict)
                or item.get("kind") != "factual_relation"
                or item.get("factual") is not True
                for item in items
            ):
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
            if relation_kind == "potential" and (
                modalities
                or any(
                    not isinstance(item, dict)
                    or item.get("kind") != "potential_link"
                    or item.get("factual") is not False
                    or item.get("modalities") not in ([], ())
                    for item in items
                )
            ):
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
    elif command == "trace_evidence" and not result:
        raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")

    if command in {"summarize_groups", "inspect_relations"}:
        if not isinstance(result["items"], list) or len(result["items"]) > 100:
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")
    if command == "find_similar_cases":
        items = result["items"]
        if not isinstance(items, list) or len(items) > 25:
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")
        for item in items:
            if not isinstance(item, dict) or set(item) != {
                "caseId",
                "score",
                "components",
                "graphVersionHash",
                "modelStateHash",
                "kindKey",
                "kindEntries",
                "concludedAt",
                "recordHash",
            }:
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
            components = item.get("components")
            if not isinstance(components, dict) or set(components) != {
                "embedding",
                "structure",
                "modality",
            }:
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
            if not all(isinstance(score, (int, float)) for score in components.values()):
                raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
    if command == "draft_report":
        if result["generatedWithoutLlm"] is not True:
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_INVALID")
        content_size = len(
            json.dumps(result["content"], ensure_ascii=False).encode("utf-8")
        )
        if content_size > 200_000 or not isinstance(result["citedHashes"], list):
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")
    if command == "search_knowledge":
        if not isinstance(result["items"], list) or len(result["items"]) > 10:
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_UNBOUNDED")
        try:
            tuple(KnowledgeItem.model_validate(item) for item in result["items"])
        except ValidationError as error:
            raise GfmProxyError(502, "GFM_GOVERNANCE_KNOWLEDGE_RESULT_INVALID") from error
    hash_fields = {
        "inspect_graph": "inspectionHash",
        "summarize_groups": "pageHash",
        "inspect_relations": "pageHash",
        "find_similar_cases": "retrievalHash",
        "get_model_card": "cardHash",
        "draft_report": "draftHash",
        "search_knowledge": "searchHash",
    }
    hash_field = hash_fields.get(command)
    if hash_field is not None:
        logical = {key: value for key, value in result.items() if key != hash_field}
        if result.get(hash_field) != canonical_sha256(logical):
            raise GfmProxyError(502, "GFM_GOVERNANCE_SKILL_RESULT_HASH_INVALID")
    _bounded_result(result)

__all__ = ["_required_keys", "_validate_result"]
