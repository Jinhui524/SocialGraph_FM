"""Bounded evidence helpers for the read-only LLM Assistant Skills."""

from __future__ import annotations

import re
from typing import Any

from .safety import _llm_summary


def _inspection_answer_context(value: dict[str, Any]) -> dict[str, Any]:
    bounded = dict(value)
    relation_counts = value.get("relationCounts")
    if isinstance(relation_counts, dict):
        valid_counts = {
            str(key): count
            for key, count in relation_counts.items()
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0
        }
        bounded["relationCounts"] = valid_counts
        bounded["relationRecordCount"] = sum(valid_counts.values())
        bounded["modalities"] = [key for key, count in valid_counts.items() if count > 0]
    return bounded


def _numeric_facts(value: str) -> set[str]:
    without_ordered_list_markers = re.sub(
        r"(?m)^\\s{0,3}\\d{1,3}[.)、]\\s+", "", value
    )
    return set(re.findall(r"\\b\\d+(?:\\.\\d+)?\\b", without_ordered_list_markers))


def _case_answer_context(case: Any, selected_target: Any = None) -> dict[str, Any]:
    case_items = tuple(getattr(case, "items", ()))
    items = [
        {
            "targetType": str(getattr(item, "target_type", "unknown")),
            "targetId": str(getattr(item, "target_id", ""))[:300],
            "note": str(getattr(item, "note", ""))[:500],
            "itemHash": str(getattr(item, "item_hash", "")),
        }
        for item in case_items[:10]
    ]
    review_events: list[dict[str, Any]] = []
    for event in tuple(getattr(case, "review_events", ()))[-3:]:
        created_at = getattr(event, "created_at", None)
        event_context: dict[str, Any] = {
            "targetType": str(getattr(event, "target_type", "unknown")),
            "targetId": str(getattr(event, "target_id", ""))[:300],
            "decision": str(getattr(event, "decision", "pending")),
            "reason": str(getattr(event, "reason", ""))[:500],
            "actor": str(getattr(event, "actor", ""))[:100],
            "sequence": getattr(event, "sequence", None),
            "eventHash": str(getattr(event, "event_hash", "")),
        }
        if created_at is not None:
            event_context["createdAt"] = (
                created_at.isoformat()
                if hasattr(created_at, "isoformat")
                else str(created_at)
            )[:80]
        review_events.append(event_context)
    current_decisions = getattr(case, "current_decisions", {})
    bounded_decisions = (
        dict(sorted(current_decisions.items())[:50])
        if isinstance(current_decisions, dict)
        else {}
    )
    decision_counts = {
        decision: sum(1 for value in bounded_decisions.values() if value == decision)
        for decision in ("confirmed", "rejected", "pending")
    }
    selected_review: dict[str, Any] | None = None
    if selected_target is not None:
        target_type = str(getattr(selected_target, "target_type", ""))
        target_id = str(getattr(selected_target, "target_id", ""))[:300]
        selected_review = {
            "targetType": target_type,
            "targetId": target_id,
            "decision": bounded_decisions.get(f"{target_type}:{target_id}"),
        }
    payload = {
        "caseId": str(case.case_id),
        "runId": str(case.run_id),
        "title": str(getattr(case, "title", ""))[:300],
        "description": str(getattr(case, "description", ""))[:500],
        "state": str(case.state),
        "items": items,
        "reviewEvents": review_events,
        "currentDecisions": bounded_decisions,
        "reviewProgress": {
            "registeredCount": len(case_items),
            "reviewedCount": len(bounded_decisions),
            "confirmedCount": decision_counts["confirmed"],
            "rejectedCount": decision_counts["rejected"],
            "pendingCount": decision_counts["pending"],
            "latestReviews": list(reversed(review_events)),
            "selectedTarget": selected_review,
        },
        "caseHash": str(case.case_hash),
    }
    safe = _llm_summary(payload)
    return safe if isinstance(safe, dict) else {}


__all__ = [
    "_case_answer_context",
    "_inspection_answer_context",
    "_numeric_facts",
]
