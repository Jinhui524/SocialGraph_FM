from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app import gfm_core_schemas as schemas
from app.config import Settings
from app.gfm_client import CoreGateway
from app.gfm_hashing import canonical_sha256
from app.gfm_core_schemas import (
    CoreAuthorizedGraphReference,
    CoreCapabilities,
    CoreRunRequest,
    CoreLeaseCalibrationIdentity,
    CoreModelCapability,
    RiskTargetScope,
)
from app.main import create_app


def _task_entity_binding(
    *,
    task_id: str = "core.community_resilience_review",
    entity_type: str = "community",
) -> dict[str, object]:
    regression = entity_type == "community"
    return {
        "taskId": task_id,
        "entityType": entity_type,
        "confidenceKind": (
            "regression-interval" if regression else "binary-calibration"
        ),
        "calibrationVersion": "confidence/1",
        "method": "validation-residual-interval" if regression else "sigmoid",
        "calibrationArtifactHash": "1" * 64,
        "calibrationProtocolHash": "2" * 64,
        "adapterDomain": f"domain-{entity_type}",
        "adapterSchemaHash": "3" * 64,
        "adapterStateHash": "4" * 64,
        "featureContractHash": "5" * 64,
    }


def _lease_identity() -> dict[str, object]:
    binding = _task_entity_binding()
    return {
        key: value
        for key, value in binding.items()
        if key != "taskId"
    } | {"sha256": "6" * 64}


@pytest.mark.parametrize(
    ("node_ids", "edge_ids"),
    [([], []), (["node-a"], ["edge-a"])],
)
def test_risk_target_scope_requires_exactly_one_nonempty_entity_kind(
    node_ids: list[str], edge_ids: list[str]
) -> None:
    """Catch mixed or empty risk scopes crossing the API/GFM trust boundary."""

    with pytest.raises(ValidationError, match="exactly one"):
        RiskTargetScope.model_validate(
            {"kind": "risk-review", "nodeIds": node_ids, "edgeIds": edge_ids}
        )


def test_openapi_expresses_risk_scope_as_node_xor_edge(tmp_path) -> None:
    """Catch an OpenAPI client generator accepting mixed risk entity kinds."""

    app = create_app(Settings(dataset_storage_root=str(tmp_path / "store")))
    openapi = app.openapi()
    risk = openapi["components"]["schemas"]["RiskTargetScope"]

    assert risk["oneOf"] == [
        {
            "properties": {
                "edgeIds": {"maxItems": 0},
                "nodeIds": {"minItems": 1},
            }
        },
        {
            "properties": {
                "edgeIds": {"minItems": 1},
                "nodeIds": {"maxItems": 0},
            }
        },
    ]
    assert sorted(
        path
        for path in openapi["paths"]
        if path.startswith("/api/v1/gfm/")
        and not path.startswith(
            ("/api/v1/gfm/research/", "/api/v1/gfm/global-model/")
        )
    ) == [
        "/api/v1/gfm/capabilities",
        "/api/v1/gfm/runs",
        "/api/v1/gfm/runs/{run_id}",
        "/api/v1/gfm/runs/{run_id}/result",
    ]


def test_capability_binds_each_task_entity_adapter_feature_and_confidence() -> None:
    """Catch an aggregate capability hiding which adapter serves an output entity."""

    task_binding = _task_entity_binding()
    feature_inventory = [
        {
            "taskId": task_binding["taskId"],
            "entityType": task_binding["entityType"],
            "featureContractHash": task_binding["featureContractHash"],
        }
    ]
    payload = {
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": "7" * 64,
        "state": "servingReady",
        "tasks": ["core.community_resilience_review"],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(feature_inventory),
        "taskBindings": [task_binding],
        "maxNodes": 100,
        "maxEdges": 500,
    }

    parsed = CoreModelCapability.model_validate(payload)

    assert parsed.task_bindings[0].adapter_domain == "domain-community"
    assert parsed.task_bindings[0].confidence_kind == "regression-interval"
    wrong_confidence = {
        **payload,
        "taskBindings": [
            {
                **task_binding,
                "confidenceKind": "binary-calibration",
                "method": "sigmoid",
            }
        ],
    }
    with pytest.raises(ValidationError, match="confidence kind"):
        CoreModelCapability.model_validate(wrong_confidence)


def test_execution_snapshot_22_seals_task_entity_lease_identity() -> None:
    """Catch a run receipt omitting adapter, feature, or confidence-kind identity."""

    identity = _lease_identity()
    snapshot: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-run-execution-snapshot/2.2",
        "runId": "00000000-0000-0000-0000-000000000001",
        "requestHash": "7" * 64,
        "controlSourceSha256": "8" * 64,
        "controlHash": "9" * 64,
        "controlGeneration": 1,
        "registryHash": "a" * 64,
        "registrySourceSha256": "b" * 64,
        "registryGeneration": 1,
        "modelVersionId": "socialgraph-fm-core/review",
        "modelVersionHash": "c" * 64,
        "checkpointSha256": "d" * 64,
        "servingManifestSha256": "e" * 64,
        "adapterSchemaHash": identity["adapterSchemaHash"],
        "calibrationIdentities": [identity],
        "calibrationSetHash": canonical_sha256([identity]),
        "taskId": "core.community_resilience_review",
        "graphVersionId": "graph/1",
        "sourceGraphFactHash": "f" * 64,
        "graphVersionHash": "1" * 64,
        "artifactId": "artifact/1",
        "artifactHash": "2" * 64,
        "artifactCatalogSha256": "3" * 64,
        "artifactCatalogHash": "4" * 64,
        "artifactCatalogGeneration": 1,
        "bundleSha256": "5" * 64,
        "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "featureContractHash": identity["featureContractHash"],
        "nodeCount": 10,
        "edgeCount": 20,
        "createdAt": "2026-08-15T00:00:00.000000Z",
    }
    snapshot["snapshotHash"] = canonical_sha256(snapshot)

    parsed = schemas.CoreRunExecutionSnapshot.model_validate_json(json.dumps(snapshot))

    assert parsed.schema_version == "socialgraph-fm.core-run-execution-snapshot/2.2"
    assert parsed.calibration_identities == (
        CoreLeaseCalibrationIdentity.model_validate(identity),
    )


def test_regression_confidence_interval_is_non_probability_evidence() -> None:
    """Catch resilience uncertainty being forced through a sigmoid probability schema."""

    regression_type = getattr(schemas, "CoreRegressionConfidenceInterval", None)
    assert regression_type is not None
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-regression-confidence-interval/1.0",
        "pointEstimate": 0.25,
        "lowerBound": 0.1,
        "upperBound": 0.4,
        "coverage": 0.9,
        "validationCount": 20,
        "scoreHash": "1" * 64,
        "taskId": "core.community_resilience_review",
        "entityType": "community",
        "entityIds": ["community-a"],
        "graphVersionHash": "2" * 64,
        "modelVersion": "socialgraph-fm-core/review",
        "modelVersionHash": "3" * 64,
        "confidenceVersion": "interval/1",
        "method": "validation-residual-interval",
        "confidenceArtifactHash": "4" * 64,
        "confidenceProtocolHash": "5" * 64,
    }
    payload["confidenceHash"] = canonical_sha256(payload)

    parsed = regression_type.model_validate(payload)

    assert not hasattr(parsed, "value")
    assert parsed.lower_bound <= parsed.point_estimate <= parsed.upper_bound


def test_governance_finding_rejects_rehashed_interval_point_estimate_contradiction() -> None:
    """Catch interval evidence describing a different prediction than the score."""

    score: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-model-score/2.0",
        "taskId": "core.community_resilience_review",
        "entityType": "community",
        "entityIds": ["community-a"],
        "score": 0.25,
        "graphVersionHash": "2" * 64,
        "modelVersion": "socialgraph-fm-core/review",
        "modelVersionHash": "3" * 64,
        "edgeIdentity": None,
    }
    score["scoreHash"] = canonical_sha256(score)
    confidence: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-regression-confidence-interval/1.0",
        "pointEstimate": 0.3,
        "lowerBound": 0.1,
        "upperBound": 0.4,
        "coverage": 0.9,
        "validationCount": 20,
        "scoreHash": score["scoreHash"],
        "taskId": score["taskId"],
        "entityType": score["entityType"],
        "entityIds": score["entityIds"],
        "graphVersionHash": score["graphVersionHash"],
        "modelVersion": score["modelVersion"],
        "modelVersionHash": score["modelVersionHash"],
        "confidenceVersion": "interval/1",
        "method": "validation-residual-interval",
        "confidenceArtifactHash": "4" * 64,
        "confidenceProtocolHash": "5" * 64,
    }
    confidence["confidenceHash"] = canonical_sha256(confidence)
    evidence: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-evidence/2.0",
        "metric": "community.connectivity",
        "valueCanonicalJson": "{}",
        "graphVersionHash": score["graphVersionHash"],
        "sourceType": "deterministic-graph-algorithm",
        "nodeIds": ["community-a"],
        "edgeIds": [],
        "algorithmConfigHash": "6" * 64,
        "modelVersionHash": None,
        "modelVersion": None,
        "modelScoreHash": None,
        "modelTaskId": None,
        "modelEntityType": None,
        "modelEntityIds": None,
        "limitations": [],
    }
    evidence["evidenceHash"] = canonical_sha256(evidence)
    finding: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-finding/2.0",
        "taskId": score["taskId"],
        "findingType": "community-resilience-candidate",
        "subjectIds": score["entityIds"],
        "score": score,
        "calibratedConfidence": confidence,
        "evidence": [evidence],
        "similarCases": [],
        "graphVersionHash": score["graphVersionHash"],
        "modelVersion": score["modelVersion"],
        "modelVersionHash": score["modelVersionHash"],
        "limitations": [
            "Manual human review is required; no automatic sanction or action is authorized.",
            "This finding is non-causal and does not predict future events.",
            "The resilience interval reports validation residual coverage, not a probability.",
        ],
        "reviewStatus": "pending-human-review",
    }
    finding["findingHash"] = canonical_sha256(finding)

    with pytest.raises(ValidationError, match="point estimate"):
        schemas.CoreFinding.model_validate(finding)


def test_api_compatibility_uses_requested_task_entity_feature_binding() -> None:
    """Catch comparing a graph contract to the aggregate multi-schema inventory hash."""

    node = _task_entity_binding(
        task_id="core.risk_and_trust_review", entity_type="node"
    )
    edge = {
        **_task_entity_binding(
            task_id="core.risk_and_trust_review", entity_type="edge"
        ),
        "featureContractHash": "8" * 64,
    }
    bindings = [node, edge]
    aggregate = canonical_sha256(
        [
            {
                "taskId": item["taskId"],
                "entityType": item["entityType"],
                "featureContractHash": item["featureContractHash"],
            }
            for item in bindings
        ]
    )
    capabilities = CoreCapabilities.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-capabilities/2.0",
            "registryHash": "1" * 64,
            "registryGeneration": 1,
            "controlHash": "2" * 64,
            "controlGeneration": 1,
            "catalogHash": "3" * 64,
            "catalogGeneration": 1,
            "servingReady": True,
            "models": [
                {
                    "modelVersionId": "socialgraph-fm-core/review",
                    "modelVersionHash": "4" * 64,
                    "state": "servingReady",
                    "tasks": ["core.risk_and_trust_review"],
                    "graphSchemaVersions": [
                        "socialgraph-fm.core-graph-bundle/2.0"
                    ],
                    "graphFeatureContractHash": aggregate,
                    "taskBindings": bindings,
                    "maxNodes": 100,
                    "maxEdges": 500,
                }
            ],
            "tasks": ["core.risk_and_trust_review"],
            "readiness": {"modelValidated": True, "coreServingReady": True},
        }
    )
    request = CoreRunRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-run-request/2.0",
            "graphVersionId": "graph/1",
            "taskId": "core.risk_and_trust_review",
            "targetScope": {
                "kind": "risk-review",
                "nodeIds": ["node-a"],
                "edgeIds": [],
            },
            "modelVersionId": "socialgraph-fm-core/review",
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 0},
        }
    )
    graph = CoreAuthorizedGraphReference.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-authorized-graph-reference/2.1",
            "graphVersionId": "graph/1",
            "sourceGraphFactHash": "5" * 64,
            "graphVersionHash": "6" * 64,
            "artifactId": "artifact/1",
            "artifactHash": "7" * 64,
            "bundleSha256": "8" * 64,
            "graphSchemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "featureContractHash": node["featureContractHash"],
            "nodeCount": 10,
            "edgeCount": 20,
        }
    )

    CoreGateway.validate_compatibility(request, graph, capabilities)
