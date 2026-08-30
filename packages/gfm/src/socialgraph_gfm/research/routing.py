"""Canonical SocialGraph-FM Research representation routes."""

from __future__ import annotations

from .contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    SIGNED_RELATION_TASK,
)

SHARED_NULL_ROUTE = "shared-null"
DOMAIN_TARGET_ROUTE = "domain-target"
ROUTE_CONTRACT_SCHEMA = "socialgraph-fm.research-route-contract/1.0"


def task_route_domain(task_id: str, domain: str) -> str | None:
    """Return the encoder domain argument required by a downstream task."""

    if not domain:
        raise ValueError("a registered SocialGraph-FM Research task route requires a domain")
    if task_id == COLLABORATION_TASK:
        return None
    if task_id not in {
        CONTENT_POLICY_TASK,
        ACCOUNT_RISK_TASK,
        SIGNED_RELATION_TASK,
    }:
        raise ValueError(f"unknown SocialGraph-FM Research task route: {task_id}")
    return domain


def route_name(domain: str | None) -> str:
    """Serialize an encoder route without overloading a missing JSON value."""

    return SHARED_NULL_ROUTE if domain is None else f"domain:{domain}"


def task_route_name(task_id: str, domain: str) -> str:
    return route_name(task_route_domain(task_id, domain))


def route_contract() -> dict[str, object]:
    """Return the immutable route policy included in every trained artifact."""

    return {
        "schemaVersion": ROUTE_CONTRACT_SCHEMA,
        "similarityRoute": SHARED_NULL_ROUTE,
        "uploadedGraphRoute": SHARED_NULL_ROUTE,
        "taskRoutes": {
            CONTENT_POLICY_TASK: DOMAIN_TARGET_ROUTE,
            ACCOUNT_RISK_TASK: DOMAIN_TARGET_ROUTE,
            SIGNED_RELATION_TASK: DOMAIN_TARGET_ROUTE,
            COLLABORATION_TASK: SHARED_NULL_ROUTE,
        },
    }


__all__ = [
    "DOMAIN_TARGET_ROUTE",
    "ROUTE_CONTRACT_SCHEMA",
    "SHARED_NULL_ROUTE",
    "route_contract",
    "route_name",
    "task_route_domain",
    "task_route_name",
]
