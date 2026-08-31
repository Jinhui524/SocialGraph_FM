from __future__ import annotations

import httpx
import pytest

from app.config import Settings
from app.main import create_app


_HTTP_METHODS = frozenset({"delete", "get", "patch", "post", "put"})

# Frozen public product surface from the complete system before repository
# normalization. This guards both removals and accidental additions.
_PRODUCT_ROUTES = frozenset(
    tuple(line.split(maxsplit=1))
    for line in """
GET /api/v1/capabilities
GET /api/v1/dataset-artifacts
GET /api/v1/dataset-artifacts/{artifact_id}
GET /api/v1/dataset-artifacts/{artifact_id}/deletion-impact
GET /api/v1/dataset-artifacts/{artifact_id}/materialized-contract
GET /api/v1/dataset-artifacts/{artifact_id}/readiness
GET /api/v1/dataset-imports/local-jobs/{job_id}
GET /api/v1/dataset-store/diagnostics
GET /api/v1/dataset-store/orphans
GET /api/v1/gfm/capabilities
GET /api/v1/gfm/global-model/capabilities
GET /api/v1/gfm/global-model/health
GET /api/v1/gfm/global-model/model-card
GET /api/v1/gfm/global-model/runs/{run_id}
GET /api/v1/gfm/global-model/runs/{run_id}/nodes/{node_id}/evidence
GET /api/v1/gfm/global-model/runs/{run_id}/result
GET /api/v1/gfm/global-model/scenario
GET /api/v1/gfm/global-model/scenario/graph-preview
GET /api/v1/gfm/research/capabilities
GET /api/v1/gfm/research/runs/{run_id}
GET /api/v1/gfm/research/runs/{run_id}/result
GET /api/v1/gfm/research/scenarios
GET /api/v1/gfm/research/scenarios/{scenario_id}/graph-preview
GET /api/v1/gfm/runs/{run_id}
GET /api/v1/gfm/runs/{run_id}/result
GET /api/v1/health
GET /api/v2/gfm/governance/adaptations/handoffs/{handoff_hash}
GET /api/v2/gfm/governance/adaptations/policies/{policy_hash}
GET /api/v2/gfm/governance/adaptations/runs/{run_id}/policies/{policy_hash}/comparison
GET /api/v2/gfm/governance/artifacts
GET /api/v2/gfm/governance/artifacts/{artifact_id}
GET /api/v2/gfm/governance/artifacts/{artifact_id}/preview
GET /api/v2/gfm/governance/capabilities
GET /api/v2/gfm/governance/cases
GET /api/v2/gfm/governance/cases/{case_id}
GET /api/v2/gfm/governance/cases/{case_id}/report
GET /api/v2/gfm/governance/health
GET /api/v2/gfm/governance/input-contract
GET /api/v2/gfm/governance/runs
GET /api/v2/gfm/governance/runs/compare
GET /api/v2/gfm/governance/runs/{run_id}
GET /api/v2/gfm/governance/runs/{run_id}/graph-preview
GET /api/v2/gfm/governance/runs/{run_id}/groups
GET /api/v2/gfm/governance/runs/{run_id}/nodes
GET /api/v2/gfm/governance/runs/{run_id}/nodes/{node_id}/evidence
GET /api/v2/gfm/governance/runs/{run_id}/potential-links
GET /api/v2/gfm/governance/runs/{run_id}/relations
GET /api/v2/gfm/governance/runs/{run_id}/result
GET /api/v2/gfm/governance/skill-audit/validation
GET /api/v2/gfm/governance/skills
GET /api/v2/gfm/governance/assistant/skills
GET /api/v2/gfm/governance/target-tasks/{registration_id}
POST /api/v1/dataset-artifacts/{artifact_id}/purge
POST /api/v1/dataset-artifacts/{artifact_id}/restore
POST /api/v1/dataset-artifacts/{artifact_id}/trash
POST /api/v1/dataset-imports/inspect
POST /api/v1/dataset-imports/inspect-local
POST /api/v1/dataset-imports/local-jobs/{job_id}/authorize
POST /api/v1/dataset-imports/local-jobs/{job_id}/cancel
POST /api/v1/dataset-imports/{inspection_id}/cancel
POST /api/v1/dataset-imports/{inspection_id}/commit
POST /api/v1/dataset-store/orphans/{artifact_id}/recover
POST /api/v1/gfm/global-model/runs
POST /api/v1/gfm/global-model/runs/{run_id}/reviews
POST /api/v1/gfm/research/runs
POST /api/v1/gfm/research/similar-nodes
POST /api/v1/gfm/runs
POST /api/v1/graph-build-intents/normalize
POST /api/v1/graph-dataset-handoffs/cancel
POST /api/v1/graph-dataset-handoffs/commit
POST /api/v1/graph-dataset-handoffs/reserve
POST /api/v1/intents/normalize
POST /api/v1/training-dataset-refs/resolve
POST /api/v2/gfm/governance/adaptations/handoffs
POST /api/v2/gfm/governance/adaptations/label-sets
POST /api/v2/gfm/governance/adaptations/label-sets/{label_set_hash}/policies
POST /api/v2/gfm/governance/adaptations/policies/{policy_hash}/activate
POST /api/v2/gfm/governance/adaptations/review-collections
POST /api/v2/gfm/governance/artifacts
POST /api/v2/gfm/governance/artifacts/compatibility
POST /api/v2/gfm/governance/artifacts/{artifact_id}/materialize
POST /api/v2/gfm/governance/assistant/execute
POST /api/v2/gfm/governance/case-index/backfill
POST /api/v2/gfm/governance/cases
POST /api/v2/gfm/governance/cases/{case_id}/items
POST /api/v2/gfm/governance/cases/{case_id}/review-events
POST /api/v2/gfm/governance/cases/{case_id}/transitions
POST /api/v2/gfm/governance/knowledge/search
POST /api/v2/gfm/governance/runs
POST /api/v2/gfm/governance/runs/{run_id}/cancel
POST /api/v2/gfm/governance/runs/{run_id}/retry
POST /api/v2/gfm/governance/similar-cases/search
POST /api/v2/gfm/governance/skills/confirm
POST /api/v2/gfm/governance/skills/execute
POST /api/v2/gfm/governance/skills/{skill}/execute
POST /api/v2/gfm/governance/target-tasks
""".strip().splitlines()
)


def test_public_product_routes_match_the_complete_system_contract(
    unconfigured_settings: Settings,
) -> None:
    openapi = create_app(unconfigured_settings).openapi()
    actual = frozenset(
        (method.upper(), path)
        for path, operations in openapi["paths"].items()
        for method in operations
        if method in _HTTP_METHODS
    )

    assert len(_PRODUCT_ROUTES) == 96
    assert actual == _PRODUCT_ROUTES


@pytest.mark.anyio
@pytest.mark.parametrize(
    "api_version",
    (
        "v1",
        "v2",
    ),
)
async def test_removed_brand_routes_return_not_found(
    api_version: str,
    unconfigured_settings: Settings,
) -> None:
    removed_brand = "io" + "hunter"
    legacy_path = f"/api/{api_version}/gfm/{removed_brand}/capabilities"
    app = create_app(unconfigured_settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get(legacy_path)

    assert response.status_code == 404
