"""Explicit contract-version migration boundary.

Only current 1.0 contracts exist in this phase. Unknown/legacy shapes are rejected rather
than guessed; future migrations must be pure functions and add fixture-backed tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ContractViolation

CURRENT_VERSIONS = frozenset(
    {
        "gfm.feature/1.0",
        "gfm.graph-snapshot-ref/1.0",
        "gfm.graph-snapshot/1.0",
        "gfm.corpus/1.0",
        "gfm.governance-task/1.0",
        "gfm.training-run/1.0",
        "gfm.checkpoint/1.0",
        "gfm.model-capability/1.0",
        "gfm.governance-finding/1.0",
    }
)


def migrate_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    version = payload.get("schemaVersion")
    if version not in CURRENT_VERSIONS:
        raise ContractViolation(f"No safe migration is registered for schemaVersion={version!r}")
    return dict(payload)
