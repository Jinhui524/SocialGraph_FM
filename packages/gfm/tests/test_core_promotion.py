from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.adapters import BundleInputAdapter
from socialgraph_gfm.core.acceptance import (
    CandidateExecutionEvidence,
    CandidateGovernanceManifest,
    CandidateTaskEvidence,
    CandidateTrainingInventory,
    CoreAcceptance,
)
from socialgraph_gfm.core.bundle import CoreGraphBundle, calculate_graph_version_hash
from socialgraph_gfm.core.inference_contracts import (
    GfmRunRequest,
    LeaseCalibrationIdentity,
    ModelCapability,
    RiskTargetScope,
)
from socialgraph_gfm.core.governance import (
    ModelScore,
    RegressionConfidenceInterval,
    RegisteredEdgeIdentity,
)
from socialgraph_gfm.core.artifact_catalog import (
    ArtifactCatalogDocument,
    ArtifactEntry,
    feature_contract_for_bundle,
)
from socialgraph_gfm.core.checkpoint import CheckpointBindings
from socialgraph_gfm.core.experiments import ExperimentArtifactRef, ExperimentProtocol
from socialgraph_gfm.core.config import TrainingConfig
from socialgraph_gfm.core import promotion as promotion_module
from socialgraph_gfm.core import serving_control as serving_control_module
from socialgraph_gfm.core import formal_preflight as formal_preflight_module
from socialgraph_gfm.core.promotion import (
    AcceptanceDerivationInputs,
    AcceptedCandidate,
    CandidateStage,
    CandidateServingDefinition,
    ServingSmokeFixture,
    ServingSmokeReport,
    ServingSmokeTaskResult,
    accept_candidate,
    promote_serving_ready,
    run_fresh_process_serving_smoke,
    stage_candidate,
)
from socialgraph_gfm.core.model import CoreGFM
from socialgraph_gfm.core.serving_head import CoreServingHead
from socialgraph_gfm.core.serving_registry import (
    CalibrationBinding,
    RegistryDocument,
    RegressionConfidenceArtifact,
    ScoreCalibration,
    ServingAdapterBinding,
    ServingCheckpointManifest,
    ServingModel,
    ServingTaskHead,
    VerifiedCheckpoint,
)
from socialgraph_gfm.core.serving_control import ServingControlStore


HASH = {character: character * 64 for character in "123456789abcdef"}
_TASK_ENTITY_ORDER = promotion_module._TASK_ENTITY_ORDER


def _accepted_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-accepted-candidate/1.0",
        "status": "accepted",
        "accepted": True,
        "candidateStageHash": HASH["1"],
        "acceptanceHash": HASH["2"],
        "candidateManifestHash": HASH["3"],
        "experimentSummaryHash": HASH["4"],
        "sourceCheckpointSha256": HASH["5"],
        "servingCheckpointSha256": HASH["6"],
        "servingModelVersionId": "core/formal-20260825",
        "servingModelHash": HASH["7"],
        "taskBindingInventoryHash": HASH["8"],
        "artifactInventoryHash": HASH["9"],
        "acceptanceRevalidationHash": HASH["a"],
    }
    payload["acceptedHash"] = canonical_sha256(payload)
    return payload


def _acceptance_report_payload(*, accepted: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-acceptance/1.1",
        "protocolHash": __import__(
            "socialgraph_gfm.core.experiments", fromlist=["ExperimentProtocol"]
        )
        .ExperimentProtocol.fixed()
        .protocol_hash,
        "preflightEvidenceHash": HASH["1"] if accepted else None,
        "aggregateHashes": [HASH["2"]] if accepted else [],
        "transferDecisionHashes": [HASH["3"]] if accepted else [],
        "rawRecordHashes": [HASH["4"]] if accepted else [],
        "winnerSelectionHash": HASH["5"] if accepted else None,
        "candidateCellId": HASH["6"] if accepted else None,
        "candidateRecordHash": HASH["7"] if accepted else None,
        "candidateLatestCheckpointSha256": HASH["8"] if accepted else None,
        "candidateCheckpointSha256": HASH["9"] if accepted else None,
        "candidateManifestHash": HASH["a"] if accepted else None,
        "candidateTaskEvidenceHashes": (
            [HASH["b"], HASH["c"], HASH["d"], HASH["e"]] if accepted else []
        ),
        "freshProcessEvidenceHash": HASH["f"] if accepted else None,
        "failedGates": [] if accepted else ["formal-preflight"],
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "promotable": accepted,
    }
    payload["acceptanceHash"] = canonical_sha256(payload)
    return payload


def _task_result(
    task_id: str,
    entity_type: str,
    *,
    suffix: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke-task/1.0",
        "taskId": task_id,
        "entityType": entity_type,
        "fixtureArtifactHash": suffix * 64,
        "fixtureBundleSha256": HASH["1"],
        "graphVersionHash": HASH["2"],
        "featureContractHash": HASH["3"],
        "adapterDomain": f"adapter-{suffix}",
        "adapterSchemaHash": HASH["4"],
        "adapterStateHash": HASH["5"],
        "confidenceArtifactHash": HASH["6"],
        "confidenceProtocolHash": HASH["7"],
        "requestHash": HASH["8"],
        "resultHash": HASH["9"],
        "findingHashes": [HASH["a"]],
        "allPendingHumanReview": True,
    }
    payload["taskResultHash"] = canonical_sha256(payload)
    return payload


def _smoke_payload() -> dict[str, Any]:
    tasks = [
        _task_result("core.community_resilience_review", "community", suffix="b"),
        _task_result("core.risk_and_trust_review", "node", suffix="c"),
        _task_result("core.risk_and_trust_review", "edge", suffix="d"),
        _task_result("core.collaboration_completion", "node-pair", suffix="e"),
    ]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke/1.0",
        "acceptedCandidateHash": _accepted_payload()["acceptedHash"],
        "acceptanceHash": HASH["2"],
        "servingModelVersionId": "core/formal-20260825",
        "servingModelHash": HASH["7"],
        "sourceCheckpointSha256": HASH["5"],
        "servingCheckpointSha256": HASH["6"],
        "taskBindingInventoryHash": HASH["8"],
        "processInterpreterSha256": HASH["9"],
        "processEnvironmentHash": HASH["a"],
        "taskResults": tasks,
        "failedGates": [],
        "succeeded": True,
    }
    payload["smokeHash"] = canonical_sha256(payload)
    return payload


def test_accepted_candidate_is_hash_bound_and_has_no_caller_ready_flag() -> None:
    accepted = AcceptedCandidate.model_validate(_accepted_payload())
    assert accepted.status == "accepted"
    assert accepted.accepted is True

    forged = _accepted_payload()
    forged["servingCheckpointSha256"] = HASH["b"]
    with pytest.raises(ValidationError, match="acceptedHash"):
        AcceptedCandidate.model_validate(forged)

    caller_flag = _accepted_payload()
    caller_flag["servingReady"] = True
    with pytest.raises(ValidationError, match="servingReady"):
        AcceptedCandidate.model_validate(caller_flag)


def test_serving_smoke_requires_all_four_task_entity_results_and_exact_hashes() -> None:
    report = ServingSmokeReport.model_validate(_smoke_payload())
    assert report.succeeded is True
    assert tuple((item.task_id, item.entity_type) for item in report.task_results) == (
        ("core.community_resilience_review", "community"),
        ("core.risk_and_trust_review", "node"),
        ("core.risk_and_trust_review", "edge"),
        ("core.collaboration_completion", "node-pair"),
    )

    missing = _smoke_payload()
    missing["taskResults"].pop(2)
    missing["smokeHash"] = canonical_sha256(
        {key: value for key, value in missing.items() if key != "smokeHash"}
    )
    with pytest.raises(ValidationError, match="task/entity"):
        ServingSmokeReport.model_validate(missing)


def test_serving_smoke_cannot_claim_success_with_empty_findings_or_failed_gate() -> None:
    empty = _task_result("core.risk_and_trust_review", "node", suffix="c")
    empty["findingHashes"] = []
    empty["taskResultHash"] = canonical_sha256(
        {key: value for key, value in empty.items() if key != "taskResultHash"}
    )
    with pytest.raises(ValidationError):
        ServingSmokeTaskResult.model_validate(empty)

    failed = _smoke_payload()
    failed["failedGates"] = ["fresh-process"]
    failed["smokeHash"] = canonical_sha256(
        {key: value for key, value in failed.items() if key != "smokeHash"}
    )
    with pytest.raises(ValidationError, match="succeeded"):
        ServingSmokeReport.model_validate(failed)


def test_stage_rejects_nonaccepted_report_before_creating_candidate_bytes(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected = CoreAcceptance.model_validate(_acceptance_report_payload(accepted=False))
    derivation = object.__new__(AcceptanceDerivationInputs)
    definition = object.__new__(CandidateServingDefinition)
    monkeypatch.setattr(
        "socialgraph_gfm.core.promotion._ACCEPTANCE_DERIVER",
        lambda **_kwargs: rejected,
    )
    stage_root = tmp_path / "candidate-stage"

    with pytest.raises(ValueError, match="accepted formal report"):
        stage_candidate(
            report=rejected,
            derivation=derivation,
            definition=definition,
            stage_root=stage_root,
        )

    assert not stage_root.exists()


def _regression_confidence_payload() -> dict[str, Any]:
    residuals = tuple(index / 100 for index in range(1, 20)) + (0.25,)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-regression-confidence-artifact/1.0",
        "confidenceVersion": "penn-resilience-residual-v1",
        "method": "validation-residual-interval",
        "coverage": 0.95,
        "residualQuantile": 0.25,
        "validationCount": 20,
        "validationHeadReportHash": HASH["1"],
        "validationPartitionHash": HASH["2"],
        "validationPredictionHash": HASH["3"],
        "validationTargetHash": HASH["4"],
        "absoluteResiduals": residuals,
    }
    payload["protocolHash"] = canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.core-regression-confidence-protocol/1.0",
            "method": payload["method"],
            "coverage": payload["coverage"],
            "validationCount": payload["validationCount"],
            "validationHeadReportHash": payload["validationHeadReportHash"],
            "validationPartitionHash": payload["validationPartitionHash"],
            "validationPredictionHash": payload["validationPredictionHash"],
            "validationTargetHash": payload["validationTargetHash"],
        }
    )
    payload["artifactHash"] = canonical_sha256(payload)
    return payload


def test_risk_scope_is_one_task_entity_at_a_time() -> None:
    assert RiskTargetScope(kind="risk-review", nodeIds=("n1",), edgeIds=()).node_ids == ("n1",)
    assert RiskTargetScope(kind="risk-review", nodeIds=(), edgeIds=(HASH["1"],)).edge_ids == (
        HASH["1"],
    )
    with pytest.raises(ValidationError, match="exactly one"):
        RiskTargetScope(
            kind="risk-review",
            nodeIds=("n1",),
            edgeIds=(HASH["1"],),
        )


def test_regression_confidence_artifact_is_interval_not_probability() -> None:
    artifact = RegressionConfidenceArtifact.model_validate(_regression_confidence_payload())
    assert artifact.method == "validation-residual-interval"
    assert artifact.coverage == 0.95
    assert not hasattr(artifact, "temperature")

    forged = _regression_confidence_payload()
    forged["residualQuantile"] = 0.5
    with pytest.raises(ValidationError, match="residualQuantile"):
        RegressionConfidenceArtifact.model_validate(forged)


def test_task_entity_binding_binds_adapter_state_schema_feature_and_confidence_kind() -> None:
    binding = CalibrationBinding.model_validate(
        {
            "entityType": "community",
            "confidenceKind": "regression-interval",
            "calibrationVersion": "penn-resilience-residual-v1",
            "calibrationMethod": "validation-residual-interval",
            "calibrationArtifactHash": HASH["1"],
            "calibrationRelativePath": "confidence/penn.json",
            "calibrationSha256": HASH["2"],
            "calibrationProtocolHash": HASH["3"],
            "adapterDomain": "facebook100.penn94",
            "adapterSchemaHash": HASH["4"],
            "adapterStateHash": HASH["5"],
            "graphFeatureContractHash": HASH["6"],
        }
    )
    head = ServingTaskHead.model_validate(
        {
            "taskId": "core.community_resilience_review",
            "kind": "community-resilience",
            "nodeOutputIndex": None,
            "calibrations": [binding.model_dump(mode="python", by_alias=True)],
        }
    )
    assert head.calibration("community").adapter_domain == "facebook100.penn94"

    wrong = binding.model_dump(mode="python", by_alias=True)
    wrong["confidenceKind"] = "binary-calibration"
    wrong["calibrationMethod"] = "sigmoid"
    with pytest.raises(ValidationError, match="community"):
        ServingTaskHead.model_validate(
            {
                "taskId": "core.community_resilience_review",
                "kind": "community-resilience",
                "nodeOutputIndex": None,
                "calibrations": [wrong],
            }
        )


def _confidence_binding(
    *,
    entity_type: str,
    domain: str,
    marker: str,
) -> dict[str, Any]:
    regression = entity_type == "community"
    return {
        "entityType": entity_type,
        "confidenceKind": "regression-interval" if regression else "binary-calibration",
        "calibrationVersion": f"confidence-{marker}",
        "calibrationMethod": ("validation-residual-interval" if regression else "sigmoid"),
        "calibrationArtifactHash": marker * 64,
        "calibrationRelativePath": f"confidence/{marker}.json",
        "calibrationSha256": HASH["1"],
        "calibrationProtocolHash": HASH["2"],
        "adapterDomain": domain,
        "adapterSchemaHash": HASH["3"],
        "adapterStateHash": HASH["4"],
        "graphFeatureContractHash": HASH["5"],
    }


def _multi_task_heads() -> list[dict[str, Any]]:
    return [
        {
            "taskId": "core.community_resilience_review",
            "kind": "community-resilience",
            "nodeOutputIndex": None,
            "calibrations": [
                _confidence_binding(
                    entity_type="community", domain="facebook100.penn94", marker="b"
                )
            ],
        },
        {
            "taskId": "core.risk_and_trust_review",
            "kind": "risk-and-trust",
            "nodeOutputIndex": 1,
            "calibrations": [
                _confidence_binding(entity_type="node", domain="tolokers", marker="c"),
                _confidence_binding(entity_type="edge", domain="wiki-rfa", marker="d"),
            ],
        },
        {
            "taskId": "core.collaboration_completion",
            "kind": "collaboration-completion",
            "nodeOutputIndex": None,
            "calibrations": [
                _confidence_binding(entity_type="node-pair", domain="github-musae", marker="e")
            ],
        },
    ]


def _adapter_bindings() -> list[dict[str, Any]]:
    return [
        {
            "adapterDomain": domain,
            "adapterSchemaHash": HASH["3"],
            "adapterStateHash": HASH["4"],
            "multiHotBuckets": 32,
        }
        for domain in (
            "facebook100.penn94",
            "github-musae",
            "tolokers",
            "wiki-rfa",
        )
    ]


def test_manifest_and_registry_bind_every_task_entity_to_one_exact_adapter() -> None:
    manifest = ServingCheckpointManifest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-serving-checkpoint-manifest/1.1",
            "task4CheckpointSha256": HASH["1"],
            "accepted": True,
            "promotable": True,
            "modelStateHash": HASH["2"],
            "adapterStateHash": HASH["4"],
            "adapterSchemaHash": HASH["3"],
            "adapterDomain": "facebook100.penn94",
            "nodeClasses": 2,
            "multiHotBuckets": 32,
            "adapterBindings": _adapter_bindings(),
            "taskHeads": _multi_task_heads(),
        }
    )
    assert len(manifest.adapter_bindings) == 4

    feature_inventory = [
        {
            "taskId": head["taskId"],
            "entityType": binding["entityType"],
            "featureContractHash": binding["graphFeatureContractHash"],
        }
        for head in _multi_task_heads()
        for binding in head["calibrations"]
    ]
    model_payload: dict[str, Any] = {
        "modelVersionId": "core/formal-20260825",
        "state": "accepted",
        "checkpoint": {
            "relativePath": "checkpoints/model.pt",
            "sha256": HASH["1"],
            "servingManifestRelativePath": "checkpoints/model.json",
            "servingManifestSha256": HASH["2"],
            "bindings": {
                "configHash": HASH["3"],
                "dataHash": HASH["4"],
                "codeHash": HASH["5"],
                "environmentHash": HASH["6"],
            },
            "adapterDomain": "facebook100.penn94",
            "nodeClasses": 2,
            "multiHotBuckets": 32,
        },
        "taskHeads": _multi_task_heads(),
        "tasks": [
            "core.community_resilience_review",
            "core.risk_and_trust_review",
            "core.collaboration_completion",
        ],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(feature_inventory),
        "maxNodes": 100_000,
        "maxEdges": 2_000_000,
    }
    model_payload["modelVersionHash"] = canonical_sha256(
        {key: value for key, value in model_payload.items() if key != "state"}
    )
    model = ServingModel.model_validate(model_payload)
    assert (
        model.task_head("core.risk_and_trust_review").calibration("edge").adapter_domain
        == "wiki-rfa"
    )

    fork = manifest.model_dump(mode="python", by_alias=True)
    fork["adapterBindings"][3]["adapterStateHash"] = HASH["6"]
    with pytest.raises(ValidationError, match="task/entity adapter"):
        ServingCheckpointManifest.model_validate(fork)


def test_regression_interval_binds_score_and_exposes_coverage_without_probability() -> None:
    score = ModelScore.create(
        task_id="core.community_resilience_review",
        entity_type="community",
        entity_ids=("community-1",),
        score=0.75,
        graph_version_hash=HASH["1"],
        model_version="core/formal-20260825",
        model_version_hash=HASH["2"],
    )
    interval = RegressionConfidenceInterval.create(
        score=score,
        lower_bound=0.5,
        upper_bound=1.0,
        coverage=0.95,
        validation_count=20,
        confidence_version="penn-resilience-residual-v1",
        method="validation-residual-interval",
        confidence_artifact_hash=HASH["3"],
        confidence_protocol_hash=HASH["4"],
    )
    assert interval.point_estimate == score.score
    assert interval.coverage == 0.95
    assert not hasattr(interval, "value")

    forged = interval.model_dump(mode="python", by_alias=True)
    forged["upperBound"] = 2.0
    with pytest.raises(ValidationError, match="confidenceHash"):
        RegressionConfidenceInterval.model_validate(forged)


def _serving_bundle(feature_name: str) -> CoreGraphBundle:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [{"id": "a", "index": 0}, {"id": "b", "index": 1}],
        "edges": [{"sourceId": "a", "targetId": "b", "edgeType": "relation", "weight": 1.0}],
        "nodeFeatures": [{"kind": "numeric", "name": feature_name, "values": [0.2, 0.8]}],
        "structuralFeatures": {"names": ["degree"], "values": [[1.0], [1.0]]},
        "source": {"sourceName": "promotion-test", "sourceSha256": HASH["1"]},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def test_serving_head_selects_task_entity_adapter_and_regression_interval() -> None:
    bundle = _serving_bundle("community-score")
    incompatible = _serving_bundle("other-score")
    community_adapter = BundleInputAdapter(bundle, multi_hot_buckets=32, mode="training")
    other_adapter = BundleInputAdapter(incompatible, multi_hot_buckets=32, mode="training")
    model = CoreGFM(node_classes=2)
    trainer = {
        "model": model.state_dict(),
        "adapterSchemas": {
            "community": community_adapter.schema.model_dump(mode="json", by_alias=True),
            "other": other_adapter.schema.model_dump(mode="json", by_alias=True),
        },
        "adapters": {
            "community": community_adapter.state_dict(),
            "other": other_adapter.state_dict(),
        },
    }
    confidence = RegressionConfidenceArtifact.model_validate(_regression_confidence_payload())
    feature_hash = HASH["5"]
    task_head = {
        "taskId": "core.community_resilience_review",
        "kind": "community-resilience",
        "nodeOutputIndex": None,
        "calibrations": [
            {
                **_confidence_binding(entity_type="community", domain="community", marker="b"),
                "calibrationVersion": confidence.confidence_version,
                "calibrationArtifactHash": confidence.artifact_hash,
                "calibrationProtocolHash": confidence.protocol_hash,
                "graphFeatureContractHash": feature_hash,
            }
        ],
    }
    feature_inventory = [
        {
            "taskId": "core.community_resilience_review",
            "entityType": "community",
            "featureContractHash": feature_hash,
        }
    ]
    model_payload: dict[str, Any] = {
        "modelVersionId": "core/formal-20260825",
        "state": "accepted",
        "checkpoint": {
            "relativePath": "checkpoints/model.pt",
            "sha256": HASH["1"],
            "servingManifestRelativePath": "checkpoints/model.json",
            "servingManifestSha256": HASH["2"],
            "bindings": {
                "configHash": HASH["3"],
                "dataHash": HASH["4"],
                "codeHash": HASH["5"],
                "environmentHash": HASH["6"],
            },
            "adapterDomain": "other",
            "nodeClasses": 2,
            "multiHotBuckets": 32,
        },
        "taskHeads": [task_head],
        "tasks": ["core.community_resilience_review"],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(feature_inventory),
        "maxNodes": 10,
        "maxEdges": 10,
    }
    model_payload["modelVersionHash"] = canonical_sha256(
        {key: value for key, value in model_payload.items() if key != "state"}
    )
    record = ServingModel.model_validate(model_payload)
    checkpoint = VerifiedCheckpoint(
        sha256=HASH["1"],
        payload={"trainer": trainer},
        snapshot=b"checkpoint",
    )
    request = GfmRunRequest.model_validate(
        {
            "schemaVersion": "socialgraph-fm.core-run-request/2.0",
            "graphVersionId": "fixture-v1",
            "taskId": "core.community_resilience_review",
            "targetScope": {"kind": "community", "communityIds": ["a"]},
            "modelVersionId": record.model_version_id,
            "parameters": {"kind": "community-resilience", "topKSimilarCases": 0},
        }
    )

    findings = CoreServingHead().execute(
        request,
        bundle,
        record,
        checkpoint,
        {"community": confidence},
    )

    assert len(findings) == 1
    interval = findings[0].calibrated_confidence
    assert isinstance(interval, RegressionConfidenceInterval)
    assert interval.coverage == confidence.coverage
    assert interval.upper_bound - interval.lower_bound == pytest.approx(
        2 * confidence.residual_quantile
    )


def test_run_lease_identity_binds_task_entity_adapter_and_feature_contract() -> None:
    identity = LeaseCalibrationIdentity.model_validate(
        {
            "entityType": "node",
            "confidenceKind": "binary-calibration",
            "calibrationVersion": "tolokers-risk-v1",
            "method": "sigmoid",
            "calibrationArtifactHash": HASH["1"],
            "calibrationProtocolHash": HASH["2"],
            "adapterDomain": "tolokers",
            "adapterSchemaHash": HASH["3"],
            "adapterStateHash": HASH["4"],
            "featureContractHash": HASH["5"],
            "sha256": HASH["6"],
        }
    )
    assert identity.adapter_domain == "tolokers"

    missing = identity.model_dump(mode="python", by_alias=True)
    missing.pop("adapterStateHash")
    with pytest.raises(ValidationError, match="adapterStateHash"):
        LeaseCalibrationIdentity.model_validate(missing)

    regression = identity.model_dump(mode="python", by_alias=True)
    regression.update(
        entityType="community",
        confidenceKind="regression-interval",
        method="validation-residual-interval",
    )
    assert (
        LeaseCalibrationIdentity.model_validate(regression).method == "validation-residual-interval"
    )


def test_capability_exposes_exact_ordered_task_entity_binding_inventory() -> None:
    heads = _multi_task_heads()
    bindings = [
        {
            "taskId": head["taskId"],
            "entityType": binding["entityType"],
            "confidenceKind": binding["confidenceKind"],
            "calibrationVersion": binding["calibrationVersion"],
            "method": binding["calibrationMethod"],
            "calibrationArtifactHash": binding["calibrationArtifactHash"],
            "calibrationProtocolHash": binding["calibrationProtocolHash"],
            "adapterDomain": binding["adapterDomain"],
            "adapterSchemaHash": binding["adapterSchemaHash"],
            "adapterStateHash": binding["adapterStateHash"],
            "featureContractHash": binding["graphFeatureContractHash"],
        }
        for head in heads
        for binding in head["calibrations"]
    ]
    feature_inventory = [
        {
            "taskId": item["taskId"],
            "entityType": item["entityType"],
            "featureContractHash": item["featureContractHash"],
        }
        for item in bindings
    ]
    payload = {
        "modelVersionId": "core/formal-20260825",
        "modelVersionHash": HASH["1"],
        "state": "servingReady",
        "tasks": [head["taskId"] for head in heads],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(feature_inventory),
        "taskBindings": bindings,
        "maxNodes": 100,
        "maxEdges": 200,
    }
    capability = ModelCapability.model_validate(payload)
    assert tuple((item.task_id, item.entity_type) for item in capability.task_bindings) == (
        ("core.community_resilience_review", "community"),
        ("core.risk_and_trust_review", "node"),
        ("core.risk_and_trust_review", "edge"),
        ("core.collaboration_completion", "node-pair"),
    )
    payload["taskBindings"] = [item for item in bindings if item["entityType"] != "edge"]
    with pytest.raises(ValidationError, match="exact ordered"):
        ModelCapability.model_validate(payload)


def _score_calibration(marker: str) -> ScoreCalibration:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-score-calibration/2.0",
        "calibrationVersion": f"calibration-{marker}",
        "method": "sigmoid",
        "temperature": 1.0,
        "bias": 0.0,
        "protocolHash": marker * 64,
    }
    payload["artifactHash"] = canonical_sha256(payload)
    return ScoreCalibration.model_validate(payload)


def _smoke_bundle() -> CoreGraphBundle:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": False,
        "nodes": [
            {"id": "a", "index": 0},
            {"id": "b", "index": 1},
            {"id": "c", "index": 2},
        ],
        "edges": [
            {
                "sourceId": "a",
                "targetId": "b",
                "edgeType": "opposes",
                "weight": -1.0,
            }
        ],
        "nodeFeatures": [{"kind": "numeric", "name": "score", "values": [0.2, 0.8, 0.4]}],
        "structuralFeatures": {
            "names": ["degree"],
            "values": [[1.0], [1.0], [0.0]],
        },
        "source": {"sourceName": "promotion-smoke", "sourceSha256": HASH["1"]},
        "splitManifest": {"strategy": "official", "assignments": []},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    return CoreGraphBundle.model_validate(payload)


def _build_candidate_and_fixtures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    binding_permutation: tuple[int, int, int, int] | None = None,
    evidence_domain_permutation: tuple[int, int, int, int] | None = None,
    model_version_id: str = "core/promotion-test",
) -> tuple[
    CandidateStage,
    AcceptedCandidate,
    CandidateServingDefinition,
    tuple[ServingSmokeFixture, ...],
    Path,
    Path,
]:
    bundle = _smoke_bundle()
    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture_root = tmp_path / "fixtures"
    fixture_root.mkdir()
    bundle_bytes = (canonical_json(bundle) + "\n").encode()
    (fixture_root / "bundle.json").write_bytes(bundle_bytes)
    feature_contract = feature_contract_for_bundle(bundle)
    feature_hash = canonical_sha256(feature_contract.model_dump(mode="python", by_alias=True))
    adapter = BundleInputAdapter(bundle, multi_hot_buckets=32, mode="training")
    adapter_state = adapter.state_dict()
    adapter_state_hash = promotion_module._tensor_state_hash(adapter_state)
    domains = (
        "facebook100.penn94",
        "github-musae",
        "tolokers",
        "wiki-rfa",
    )
    confidence_objects: dict[tuple[str, str], ScoreCalibration | RegressionConfidenceArtifact] = {
        ("core.community_resilience_review", "community"): (
            RegressionConfidenceArtifact.model_validate(_regression_confidence_payload())
        ),
        ("core.risk_and_trust_review", "node"): _score_calibration("c"),
        ("core.risk_and_trust_review", "edge"): _score_calibration("d"),
        ("core.collaboration_completion", "node-pair"): _score_calibration("e"),
    }
    confidence_bytes = {
        key: (canonical_json(value) + "\n").encode() for key, value in confidence_objects.items()
    }
    canonical_public_domains = (domains[0], domains[2], domains[3], domains[1])
    permutation = binding_permutation or (0, 1, 2, 3)
    entity_domains = {
        key: canonical_public_domains[permutation[index]]
        for index, key in enumerate(_TASK_ENTITY_ORDER)
    }
    confidence_bindings: dict[tuple[str, str], dict[str, Any]] = {}
    for key in _TASK_ENTITY_ORDER:
        value = confidence_objects[key]
        regression = isinstance(value, RegressionConfidenceArtifact)
        confidence_bindings[key] = {
            "entityType": key[1],
            "confidenceKind": "regression-interval" if regression else "binary-calibration",
            "calibrationVersion": (
                value.confidence_version if regression else value.calibration_version
            ),
            "calibrationMethod": value.method,
            "calibrationArtifactHash": value.artifact_hash,
            "calibrationRelativePath": (
                f"artifacts/confidence-{hashlib.sha256(confidence_bytes[key]).hexdigest()}.json"
            ),
            "calibrationSha256": hashlib.sha256(confidence_bytes[key]).hexdigest(),
            "calibrationProtocolHash": value.protocol_hash,
            "adapterDomain": entity_domains[key],
            "adapterSchemaHash": adapter.schema.adapter_schema_hash,
            "adapterStateHash": adapter_state_hash,
            "graphFeatureContractHash": feature_hash,
        }
    task_heads = (
        ServingTaskHead.model_validate(
            {
                "taskId": "core.community_resilience_review",
                "kind": "community-resilience",
                "nodeOutputIndex": None,
                "calibrations": [
                    confidence_bindings[("core.community_resilience_review", "community")]
                ],
            }
        ),
        ServingTaskHead.model_validate(
            {
                "taskId": "core.risk_and_trust_review",
                "kind": "risk-and-trust",
                "nodeOutputIndex": 1,
                "calibrations": [
                    confidence_bindings[("core.risk_and_trust_review", "node")],
                    confidence_bindings[("core.risk_and_trust_review", "edge")],
                ],
            }
        ),
        ServingTaskHead.model_validate(
            {
                "taskId": "core.collaboration_completion",
                "kind": "collaboration-completion",
                "nodeOutputIndex": None,
                "calibrations": [
                    confidence_bindings[("core.collaboration_completion", "node-pair")]
                ],
            }
        ),
    )
    definition_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-definition/1.0",
        "modelVersionId": model_version_id,
        "taskHeads": [item.model_dump(mode="python", by_alias=True) for item in task_heads],
        "maxNodes": 100,
        "maxEdges": 200,
    }
    definition_payload["definitionHash"] = canonical_sha256(definition_payload)
    definition = CandidateServingDefinition.model_validate(definition_payload)
    model = CoreGFM(node_classes=2)
    bindings = CheckpointBindings(
        config_hash=HASH["1"],
        data_hash=HASH["2"],
        code_hash=HASH["3"],
        environment_hash=HASH["4"],
    )
    checkpoint_bytes = promotion_module._serialize_checkpoint(
        {
            "model": model.state_dict(),
            "adapterSchemas": {
                domain: adapter.schema.model_dump(mode="python", by_alias=True)
                for domain in domains
            },
            "adapters": {domain: adapter_state for domain in domains},
        },
        bindings=bindings,
    )
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    adapters = tuple(
        ServingAdapterBinding(
            adapterDomain=domain,
            adapterSchemaHash=adapter.schema.adapter_schema_hash,
            adapterStateHash=adapter_state_hash,
            multiHotBuckets=32,
        )
        for domain in domains
    )
    manifest = ServingCheckpointManifest(
        schemaVersion="socialgraph-fm.core-serving-checkpoint-manifest/1.1",
        task4CheckpointSha256=checkpoint_sha,
        accepted=True,
        promotable=True,
        modelStateHash=promotion_module._tensor_state_hash(model.state_dict()),
        adapterStateHash=adapter_state_hash,
        adapterSchemaHash=adapter.schema.adapter_schema_hash,
        adapterDomain=domains[0],
        nodeClasses=2,
        multiHotBuckets=32,
        adapterBindings=adapters,
        taskHeads=task_heads,
    )
    manifest_bytes = (canonical_json(manifest) + "\n").encode()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    feature_inventory = [
        {
            "taskId": head.task_id,
            "entityType": binding.entity_type,
            "featureContractHash": binding.graph_feature_contract_hash,
        }
        for head in task_heads
        for binding in head.calibrations
    ]
    model_payload: dict[str, Any] = {
        "modelVersionId": definition.model_version_id,
        "state": "accepted",
        "checkpoint": {
            "relativePath": f"artifacts/serving-checkpoint-{checkpoint_sha}.pt",
            "sha256": checkpoint_sha,
            "servingManifestRelativePath": f"artifacts/serving-manifest-{manifest_sha}.json",
            "servingManifestSha256": manifest_sha,
            "bindings": {
                "configHash": bindings.config_hash,
                "dataHash": bindings.data_hash,
                "codeHash": bindings.code_hash,
                "environmentHash": bindings.environment_hash,
            },
            "adapterDomain": domains[0],
            "nodeClasses": 2,
            "multiHotBuckets": 32,
        },
        "taskHeads": [item.model_dump(mode="python", by_alias=True) for item in task_heads],
        "tasks": [item.task_id for item in task_heads],
        "graphSchemaVersions": ["socialgraph-fm.core-graph-bundle/2.0"],
        "graphFeatureContractHash": canonical_sha256(feature_inventory),
        "maxNodes": 100,
        "maxEdges": 200,
    }
    model_payload["modelVersionHash"] = canonical_sha256(
        {key: value for key, value in model_payload.items() if key != "state"}
    )
    serving_model = ServingModel.model_validate(model_payload)
    internal_tasks = (
        "github.relation-completion",
        "penn94.community-resilience",
        "tolokers.risk",
        "wiki-rfa.vote-sign",
    )
    canonical_internal_domains = (domains[1], domains[0], domains[2], domains[3])
    evidence_permutation = evidence_domain_permutation or (0, 1, 2, 3)
    internal_domains = tuple(canonical_internal_domains[index] for index in evidence_permutation)
    task_evidence: list[CandidateTaskEvidence] = []
    for index, (task_id, domain) in enumerate(
        zip(internal_tasks, internal_domains, strict=True), start=1
    ):
        task_payload: dict[str, Any] = {
            "taskId": task_id,
            "cellId": f"{index}" * 64,
            "recordHash": f"{index + 4:x}" * 64,
            "recipeHash": f"{index + 8:x}" * 64,
            "graphVersionHash": bundle.graph_version_hash,
            "splitInventoryHash": HASH["1"],
            "adapterDomain": domain,
            "supervisedDataHash": HASH["2"],
            "headReportHash": HASH["3"],
            "calibrationHash": None if "resilience" in task_id else HASH["4"],
        }
        task_payload["evidenceHash"] = canonical_sha256(task_payload)
        task_evidence.append(CandidateTaskEvidence.model_validate(task_payload))
    inventory_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-candidate-training-inventory/1.0",
        "tasks": [item.model_dump(mode="python", by_alias=True) for item in task_evidence],
    }
    inventory_payload["inventoryHash"] = canonical_sha256(inventory_payload)
    training_inventory = CandidateTrainingInventory.model_validate(inventory_payload)
    execution_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-candidate-execution/1.0",
        "methodId": "multi-graph-shared-gfm",
        "seed": 1,
        "labelBudget": "full",
        "trainerConfig": TrainingConfig.formal().to_dict(),
        "taskCellIds": [item.cell_id for item in task_evidence],
        "recipeHashes": [item.recipe_hash for item in task_evidence],
        "sourceRecordHashes": [item.record_hash for item in task_evidence],
        "winnerSelectionHash": HASH["5"],
    }
    execution_payload["configHash"] = canonical_sha256(execution_payload)
    execution = CandidateExecutionEvidence.model_validate(execution_payload)
    latest_ref = ExperimentArtifactRef(
        role="latest-checkpoint",
        relativePath="formal/latest.pt",
        byteSha256=checkpoint_sha,
        semanticHash=checkpoint_sha,
        sizeBytes=len(checkpoint_bytes),
    )
    best_ref = ExperimentArtifactRef(
        role="best-checkpoint",
        relativePath="formal/best.pt",
        byteSha256=checkpoint_sha,
        semanticHash=checkpoint_sha,
        sizeBytes=len(checkpoint_bytes),
    )
    candidate_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-governance-candidate/1.0",
        "protocolHash": ExperimentProtocol.fixed().protocol_hash,
        "execution": execution.model_dump(mode="python", by_alias=True),
        "trainingInventory": training_inventory.model_dump(mode="python", by_alias=True),
        "latestCheckpoint": latest_ref.model_dump(mode="python", by_alias=True),
        "bestCheckpoint": best_ref.model_dump(mode="python", by_alias=True),
        "encoderSourceCellId": task_evidence[0].cell_id,
        "encoderSourceBestCheckpointSha256": checkpoint_sha,
        "codeHash": HASH["3"],
        "environmentHash": HASH["4"],
    }
    candidate_payload["manifestHash"] = canonical_sha256(candidate_payload)
    candidate_manifest = CandidateGovernanceManifest.model_validate(candidate_payload)
    candidate_manifest_bytes = (canonical_json(candidate_manifest) + "\n").encode()
    acceptance_payload = _acceptance_report_payload(accepted=True)
    acceptance_payload["candidateCellId"] = task_evidence[0].cell_id
    acceptance_payload["candidateLatestCheckpointSha256"] = checkpoint_sha
    acceptance_payload["candidateCheckpointSha256"] = checkpoint_sha
    acceptance_payload["candidateManifestHash"] = candidate_manifest.manifest_hash
    acceptance_payload["candidateTaskEvidenceHashes"] = sorted(
        item.evidence_hash for item in task_evidence
    )
    acceptance_payload["acceptanceHash"] = canonical_sha256(
        {key: value for key, value in acceptance_payload.items() if key != "acceptanceHash"}
    )
    report = CoreAcceptance.model_validate(acceptance_payload)
    derivation = object.__new__(AcceptanceDerivationInputs)
    for name, value in {
        "runtime_root": tmp_path,
        "preflight_path": tmp_path / "preflight.json",
        "protocol": ExperimentProtocol.fixed(),
        "aggregates": (),
        "transfer_decisions": (),
        "candidate_cell_id": report.candidate_cell_id,
        "candidate_manifest_path": tmp_path / "candidate.json",
        "fresh_process_evidence_path": tmp_path / "fresh.json",
        "telemetry_policy": object(),
    }.items():
        object.__setattr__(derivation, name, value)
    monkeypatch.setattr(promotion_module, "_ACCEPTANCE_DERIVER", lambda **_kwargs: report)
    evidence_by_task = {item.task_id: item for item in task_evidence}
    source_tasks = (
        "penn94.community-resilience",
        "tolokers.risk",
        "wiki-rfa.vote-sign",
        "github.relation-completion",
    )
    binding_by_key = {
        (head.task_id, binding.entity_type): binding
        for head in definition.task_heads
        for binding in head.calibrations
    }
    task_binding_inventory_hash = canonical_sha256(
        [
            {
                "publicTaskId": task,
                "entityType": entity,
                "servingBinding": binding_by_key[(task, entity)].model_dump(
                    mode="python", by_alias=True
                ),
                "acceptedTaskEvidence": evidence_by_task[source_task].model_dump(
                    mode="python", by_alias=True
                ),
            }
            for (task, entity), source_task in zip(_TASK_ENTITY_ORDER, source_tasks, strict=True)
        ]
    )
    materialized = promotion_module._CandidateBytes(
        acceptance=(canonical_json(report) + "\n").encode(),
        candidate_manifest=candidate_manifest_bytes,
        source_checkpoint=checkpoint_bytes,
        serving_checkpoint=checkpoint_bytes,
        serving_manifest=manifest_bytes,
        serving_model=serving_model,
        task_binding_inventory_hash=task_binding_inventory_hash,
        confidence=tuple(
            (task, entity, confidence_bytes[(task, entity)]) for task, entity in _TASK_ENTITY_ORDER
        ),
    )
    monkeypatch.setattr(
        promotion_module,
        "_CANDIDATE_MATERIALIZER",
        lambda *_args: materialized,
    )
    stage_root = tmp_path / "stage"
    stage = stage_candidate(
        report=report,
        derivation=derivation,
        definition=definition,
        stage_root=stage_root,
    )
    accepted = accept_candidate(
        stage=stage,
        report=report,
        derivation=derivation,
        definition=definition,
        stage_root=stage_root,
        accepted_root=tmp_path / "acceptance-area",
    )
    edge_hash = RegisteredEdgeIdentity.create(bundle.edges[0]).edge_hash
    requests = (
        {
            "taskId": _TASK_ENTITY_ORDER[0][0],
            "targetScope": {"kind": "community", "communityIds": ["a"]},
            "parameters": {"kind": "community-resilience", "topKSimilarCases": 0},
        },
        {
            "taskId": _TASK_ENTITY_ORDER[1][0],
            "targetScope": {"kind": "risk-review", "nodeIds": ["a"], "edgeIds": []},
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 0},
        },
        {
            "taskId": _TASK_ENTITY_ORDER[2][0],
            "targetScope": {"kind": "risk-review", "nodeIds": [], "edgeIds": [edge_hash]},
            "parameters": {"kind": "risk-and-trust", "topKSimilarCases": 0},
        },
        {
            "taskId": _TASK_ENTITY_ORDER[3][0],
            "targetScope": {"kind": "node-pairs", "pairs": [["a", "c"]]},
            "parameters": {
                "kind": "collaboration-completion",
                "topKSimilarCases": 0,
                "candidateLimit": 1,
            },
        },
    )
    fixtures: list[ServingSmokeFixture] = []
    for index, ((task, entity), request_values) in enumerate(
        zip(_TASK_ENTITY_ORDER, requests, strict=True)
    ):
        entry = ArtifactEntry(
            artifactId=f"fixture-{index}",
            artifactHash=canonical_sha256({"fixture": index}),
            bundleSha256=hashlib.sha256(bundle_bytes).hexdigest(),
            relativePath="bundle.json",
            graphVersionId="fixture-graph-v1",
            sourceGraphFactHash=HASH["f"],
            graphVersionHash=bundle.graph_version_hash,
            graphSchemaVersion=bundle.schema_version,
            featureContract=feature_contract,
            featureContractHash=feature_hash,
            nodeCount=len(bundle.nodes),
            edgeCount=len(bundle.edges),
        )
        request = GfmRunRequest.model_validate(
            {
                "schemaVersion": "socialgraph-fm.core-run-request/2.0",
                "graphVersionId": "fixture-graph-v1",
                "modelVersionId": serving_model.model_version_id,
                **request_values,
            }
        )
        fixture_payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-serving-smoke-fixture/1.0",
            "taskId": task,
            "entityType": entity,
            "artifact": entry.model_dump(mode="python", by_alias=True),
            "request": request.model_dump(mode="python", by_alias=True),
        }
        fixture_payload["fixtureHash"] = canonical_sha256(fixture_payload)
        fixtures.append(ServingSmokeFixture.model_validate(fixture_payload))
    return stage, accepted, definition, tuple(fixtures), stage_root, fixture_root


def _empty_control(root: Path) -> ServingControlStore:
    root.mkdir(parents=True, exist_ok=True)
    registry = RegistryDocument(
        schemaVersion="socialgraph-fm.core-serving-registry/2.0",
        generation=0,
        models=(),
    )
    catalog = ArtifactCatalogDocument(
        schemaVersion="socialgraph-fm.core-serving-graph-catalog/1.0",
        generation=0,
        artifacts=(),
    )
    registry_bytes = (canonical_json(registry) + "\n").encode()
    catalog_bytes = (canonical_json(catalog) + "\n").encode()
    (root / "registry.json").write_bytes(registry_bytes)
    (root / "catalog.json").write_bytes(catalog_bytes)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-control/1.0",
        "generation": 0,
        "registry": {
            "relativePath": "registry.json",
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "semanticHash": canonical_sha256(registry.model_dump(mode="python", by_alias=True)),
            "generation": 0,
        },
        "catalog": {
            "relativePath": "catalog.json",
            "sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "semanticHash": canonical_sha256(catalog.model_dump(mode="python", by_alias=True)),
            "generation": 0,
        },
    }
    payload["controlHash"] = canonical_sha256(payload)
    (root / "control.json").write_bytes((canonical_json(payload) + "\n").encode())
    return ServingControlStore.load(root / "control.json", high_water_root=root / "high-water")


def _sealed_transaction_smoke(
    stage: CandidateStage,
    accepted: AcceptedCandidate,
    fixtures: tuple[ServingSmokeFixture, ...],
):
    results: list[dict[str, Any]] = []
    for fixture in fixtures:
        binding = stage.serving_model.task_head(fixture.task_id).calibration(fixture.entity_type)
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-serving-smoke-task/1.0",
            "taskId": fixture.task_id,
            "entityType": fixture.entity_type,
            "fixtureArtifactHash": fixture.artifact.artifact_hash,
            "fixtureBundleSha256": fixture.artifact.bundle_sha256,
            "graphVersionHash": fixture.artifact.graph_version_hash,
            "featureContractHash": fixture.artifact.feature_contract_hash,
            "adapterDomain": binding.adapter_domain,
            "adapterSchemaHash": binding.adapter_schema_hash,
            "adapterStateHash": binding.adapter_state_hash,
            "confidenceArtifactHash": binding.calibration_artifact_hash,
            "confidenceProtocolHash": binding.calibration_protocol_hash,
            "requestHash": canonical_sha256(
                fixture.request.model_dump(mode="python", by_alias=True)
            ),
            "resultHash": HASH["1"],
            "findingHashes": [HASH["2"]],
            "allPendingHumanReview": True,
        }
        payload["taskResultHash"] = canonical_sha256(payload)
        results.append(payload)
    report_payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke/1.0",
        "acceptedCandidateHash": accepted.accepted_hash,
        "acceptanceHash": accepted.acceptance_hash,
        "servingModelVersionId": accepted.serving_model_version_id,
        "servingModelHash": accepted.serving_model_hash,
        "sourceCheckpointSha256": accepted.source_checkpoint_sha256,
        "servingCheckpointSha256": accepted.serving_checkpoint_sha256,
        "taskBindingInventoryHash": accepted.task_binding_inventory_hash,
        "processInterpreterSha256": HASH["3"],
        "processEnvironmentHash": HASH["4"],
        "taskResults": results,
        "failedGates": [],
        "succeeded": True,
    }
    report_payload["smokeHash"] = canonical_sha256(report_payload)
    report = ServingSmokeReport.model_validate(report_payload)
    inventory_hash = canonical_sha256(
        [item.model_dump(mode="python", by_alias=True) for item in fixtures]
    )
    return promotion_module._new_verified_smoke(report, inventory_hash)


@pytest.mark.parametrize(
    "binding_permutation",
    (
        (0, 2, 1, 3),
        (3, 1, 2, 0),
        (1, 2, 3, 0),
        (3, 2, 1, 0),
    ),
)
def test_candidate_definition_rejects_coherently_rehashed_semantic_task_domain_permutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_permutation: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="task.*evidence|semantic task"):
        _build_candidate_and_fixtures(
            tmp_path,
            monkeypatch,
            binding_permutation=binding_permutation,
        )


@pytest.mark.parametrize(
    "evidence_domain_permutation",
    (
        (0, 1, 3, 2),
        (1, 2, 3, 0),
    ),
)
def test_candidate_acceptance_rejects_semantically_permuted_formal_task_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_domain_permutation: tuple[int, int, int, int],
) -> None:
    with pytest.raises(ValueError, match="task.*evidence|semantic task"):
        _build_candidate_and_fixtures(
            tmp_path,
            monkeypatch,
            evidence_domain_permutation=evidence_domain_permutation,
        )


def test_fresh_process_smoke_executes_exact_four_bindings_and_seals_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = run_fresh_process_serving_smoke(
        accepted=accepted,
        stage=stage,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
    )
    verified.verify()
    assert verified.report.succeeded is True
    assert (
        tuple((item.task_id, item.entity_type) for item in verified.report.task_results)
        == _TASK_ENTITY_ORDER
    )
    assert all(item.all_pending_human_review for item in verified.report.task_results)

    forged = verified.report.model_copy(update={"process_environment_hash": HASH["f"]})
    control = object.__new__(ServingControlStore)
    with pytest.raises(TypeError, match="self-hashed smoke"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=forged,  # type: ignore[arg-type]
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
        )


def test_atomic_promotion_publishes_control_last_and_reloads_one_ready_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    observed_steps: list[str] = []
    receipt = promote_serving_ready(
        accepted=accepted,
        stage=stage,
        verified_smoke=verified,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
        serving_control=control,
        failure_injector=observed_steps.append,
    )
    winning = control.capture()
    assert receipt.control_generation == 1
    assert winning.document.generation == 1
    assert (
        len([model for model in winning.registry_document.models if model.state == "servingReady"])
        == 1
    )
    assert observed_steps[-2:] == ["after-control-replace", "before-high-water-accept"]


@pytest.mark.parametrize("crash_step", ("after-control-replace", "before-high-water-accept"))
def test_post_swap_process_abort_recovers_same_generation_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_step: str,
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    live_root = tmp_path / "live"
    control = _empty_control(live_root)

    def abort(stage_name: str) -> None:
        if stage_name == crash_step:
            raise SystemExit(f"simulated process abort at {stage_name}")

    with pytest.raises(SystemExit, match="simulated process abort"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
            failure_injector=abort,
        )

    crashed_control = control.capture()
    crashed_bytes = crashed_control.control_snapshot
    version_files = {
        path.name: path.read_bytes() for path in sorted((live_root / "versions").iterdir())
    }
    restarted = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    first_receipt = promote_serving_ready(
        accepted=accepted,
        stage=stage,
        verified_smoke=verified,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
        serving_control=restarted,
    )
    restarted_again = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    second_receipt = promote_serving_ready(
        accepted=accepted,
        stage=stage,
        verified_smoke=verified,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
        serving_control=restarted_again,
    )

    assert crashed_control.document.generation == 1
    assert first_receipt == second_receipt
    assert first_receipt.control_generation == 1
    assert restarted_again.capture().control_snapshot == crashed_bytes
    assert {
        path.name: path.read_bytes() for path in sorted((live_root / "versions").iterdir())
    } == version_files
    high_water = promotion_module.ServingHighWater.model_validate_json(
        restarted_again.high_water_path.read_bytes()
    )
    assert high_water.control_generation == crashed_control.document.generation
    assert high_water.control_hash == crashed_control.document.control_hash
    assert high_water.registry_hash == crashed_control.registry_hash
    assert high_water.catalog_hash == crashed_control.catalog_hash


def test_post_swap_process_abort_rejects_a_conflicting_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = _build_candidate_and_fixtures(
        first_root,
        monkeypatch,
        model_version_id="core/promotion-first",
    )
    stage, accepted, _definition, fixtures, stage_root, fixture_root = first
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    live_root = tmp_path / "live"
    control = _empty_control(live_root)

    def abort(stage_name: str) -> None:
        if stage_name == "before-high-water-accept":
            raise SystemExit("simulated process abort")

    with pytest.raises(SystemExit, match="simulated process abort"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
            failure_injector=abort,
        )
    crashed_bytes = control.capture().control_snapshot

    second = _build_candidate_and_fixtures(
        second_root,
        monkeypatch,
        model_version_id="core/promotion-second",
    )
    (
        second_stage,
        second_accepted,
        _definition,
        second_fixtures,
        second_stage_root,
        second_fixture_root,
    ) = second
    second_verified = _sealed_transaction_smoke(second_stage, second_accepted, second_fixtures)
    restarted = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    with pytest.raises(ValueError, match="incomplete|unaccepted|recovery"):
        promote_serving_ready(
            accepted=second_accepted,
            stage=second_stage,
            verified_smoke=second_verified,
            stage_root=second_stage_root,
            fixture_root=second_fixture_root,
            fixtures=second_fixtures,
            serving_control=restarted,
        )
    assert restarted.capture().control_snapshot == crashed_bytes


def test_post_swap_process_abort_rejects_a_different_sealed_smoke_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    live_root = tmp_path / "live"
    control = _empty_control(live_root)

    def abort(stage_name: str) -> None:
        if stage_name == "before-high-water-accept":
            raise SystemExit("simulated process abort")

    with pytest.raises(SystemExit, match="simulated process abort"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
            failure_injector=abort,
        )
    crashed_bytes = control.capture().control_snapshot
    smoke_payload = verified.report.model_dump(mode="python", by_alias=True, exclude={"smoke_hash"})
    smoke_payload["processEnvironmentHash"] = HASH["f"]
    smoke_payload["smokeHash"] = canonical_sha256(smoke_payload)
    conflicting_report = ServingSmokeReport.model_validate(smoke_payload)
    conflicting_smoke = promotion_module._new_verified_smoke(
        conflicting_report,
        verified.fixture_inventory_hash,
    )
    restarted = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    with pytest.raises(ValueError, match="recovery receipt differs"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=conflicting_smoke,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=restarted,
        )
    assert restarted.capture().control_snapshot == crashed_bytes


def test_post_swap_recovery_rechecks_visible_control_before_accepting_high_water(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    live_root = tmp_path / "live"
    control = _empty_control(live_root)

    def abort(stage_name: str) -> None:
        if stage_name == "before-high-water-accept":
            raise SystemExit("simulated process abort")

    with pytest.raises(SystemExit, match="simulated process abort"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
            failure_injector=abort,
        )
    high_water_before = control.high_water_path.read_bytes()
    competitor = b'{"thirdParty":"during-recovery"}\n'
    original_recovery = promotion_module._recover_published_promotion

    def replace_after_recovery(**kwargs):
        receipt = original_recovery(**kwargs)
        temporary = control.path.with_name(f".{control.path.name}.recovery-competitor")
        temporary.write_bytes(competitor)
        os.replace(temporary, control.path)
        return receipt

    monkeypatch.setattr(
        promotion_module,
        "_recover_published_promotion",
        replace_after_recovery,
    )
    restarted = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    with pytest.raises((OSError, RuntimeError, ValueError), match="changed|identity"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=restarted,
        )
    assert control.path.read_bytes() == competitor
    assert control.high_water_path.read_bytes() == high_water_before


def test_persisted_high_water_reread_failure_keeps_generation_for_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    (
        second_stage,
        second_accepted,
        _second_definition,
        second_fixtures,
        second_stage_root,
        second_fixture_root,
    ) = _build_candidate_and_fixtures(
        tmp_path / "conflicting-candidate",
        monkeypatch,
        model_version_id="core/commit-uncertain-conflict",
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    live_root = tmp_path / "live"
    control = _empty_control(live_root)
    original_control_bytes = control.path.read_bytes()
    original_read = serving_control_module.read_confined_snapshot
    failed = False
    before_handles: int | None = None
    handle_count = None
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessHandleCount.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        kernel32.GetProcessHandleCount.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()

        def windows_handle_count() -> int:
            count = ctypes.c_ulong()
            assert kernel32.GetProcessHandleCount(process, ctypes.byref(count))
            return int(count.value)

        warm_target = control.path.with_name(".accept-warm-target")
        warm_replacement = control.path.with_name(".accept-warm-replacement")
        warm_backup = control.path.with_name(".accept-warm-backup")
        warm_target.write_bytes(b"old")
        warm_replacement.write_bytes(b"new")
        promotion_module._windows_replace_with_backup(warm_target, warm_replacement, warm_backup)
        warm_target.unlink()
        warm_backup.unlink()
        handle_count = windows_handle_count
        before_handles = handle_count()

    def fail_first_persisted_generation_one_read(
        root: Path, relative_path: str, *, max_bytes: int
    ) -> bytes:
        nonlocal failed
        target = root / relative_path
        if (
            relative_path == control.high_water_path.name
            and target.is_file()
            and b'"controlGeneration":1' in target.read_bytes()
            and not failed
        ):
            failed = True
            raise OSError("injected persisted high-water reread failure")
        return original_read(root, relative_path, max_bytes=max_bytes)

    monkeypatch.setattr(
        serving_control_module,
        "read_confined_snapshot",
        fail_first_persisted_generation_one_read,
    )
    with pytest.raises(OSError, match="persisted high-water reread"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
        )

    assert failed is True
    uncertain = control.capture()
    uncertain_control_bytes = uncertain.control_snapshot
    uncertain_high_water_bytes = control.high_water_path.read_bytes()
    uncertain_high_water = promotion_module.ServingHighWater.model_validate_json(
        uncertain_high_water_bytes
    )
    uncertain_versions = {
        path.name: path.read_bytes() for path in sorted((live_root / "versions").iterdir())
    }
    uncertain_receipt_snapshots = tuple(
        snapshot
        for name, snapshot in uncertain_versions.items()
        if name.startswith("promotion-receipt-g1-")
    )
    assert len(uncertain_receipt_snapshots) == 1
    uncertain_receipt = promotion_module.PromotionReceipt.model_validate_json(
        uncertain_receipt_snapshots[0]
    )
    uncertain_transaction_files = {
        path.name: path.read_bytes() for path in sorted(live_root.glob(f".{control.path.name}.*"))
    }
    assert uncertain.document.generation == 1
    assert uncertain_high_water.control_generation == uncertain.document.generation
    assert uncertain_high_water.control_hash == uncertain.document.control_hash
    assert uncertain_high_water.registry_hash == uncertain.registry_hash
    assert uncertain_high_water.catalog_hash == uncertain.catalog_hash
    if handle_count is not None:
        assert handle_count() == before_handles
    smoke_payload = verified.report.model_dump(mode="python", by_alias=True, exclude={"smoke_hash"})
    smoke_payload["processEnvironmentHash"] = HASH["f"]
    smoke_payload["smokeHash"] = canonical_sha256(smoke_payload)
    conflicting_smoke = promotion_module._new_verified_smoke(
        ServingSmokeReport.model_validate(smoke_payload),
        verified.fixture_inventory_hash,
    )
    conflicting_store = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    with pytest.raises(ValueError, match="recovery receipt differs"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=conflicting_smoke,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=conflicting_store,
        )
    with pytest.raises(ValueError, match="recovery receipt differs"):
        promote_serving_ready(
            accepted=second_accepted,
            stage=second_stage,
            verified_smoke=_sealed_transaction_smoke(
                second_stage, second_accepted, second_fixtures
            ),
            stage_root=second_stage_root,
            fixture_root=second_fixture_root,
            fixtures=second_fixtures,
            serving_control=conflicting_store,
        )
    restarted = ServingControlStore.load(
        live_root / "control.json",
        high_water_root=live_root / "high-water",
    )
    receipt = promote_serving_ready(
        accepted=accepted,
        stage=stage,
        verified_smoke=verified,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
        serving_control=restarted,
    )
    high_water = promotion_module.ServingHighWater.model_validate_json(
        restarted.high_water_path.read_bytes()
    )

    assert receipt == uncertain_receipt
    assert receipt.control_generation == 1
    assert restarted.capture().control_snapshot == uncertain_control_bytes
    assert restarted.high_water_path.read_bytes() == uncertain_high_water_bytes
    assert {
        path.name: path.read_bytes() for path in sorted((live_root / "versions").iterdir())
    } == uncertain_versions
    assert {
        path.name: path.read_bytes() for path in sorted(live_root.glob(f".{control.path.name}.*"))
    } == uncertain_transaction_files
    assert high_water.control_generation == uncertain.document.generation
    assert high_water.control_hash == uncertain.document.control_hash
    assert high_water.registry_hash == uncertain.registry_hash
    assert high_water.catalog_hash == uncertain.catalog_hash
    retained_swaps = tuple(live_root.glob(f".{control.path.name}.*.swap"))
    promotion_lock = control.path.with_name(f".{control.path.name}.promotion.lock")
    if os.name == "nt":
        assert not retained_swaps
        assert not promotion_lock.exists()
    else:
        # renameat2 exchange leaves the exact predecessor under the random swap
        # name, and flock deliberately retains its stable lock inode.
        assert len(retained_swaps) == 1
        assert retained_swaps[0].read_bytes() == original_control_bytes
        assert promotion_lock.is_file() and not promotion_lock.is_symlink()
    if handle_count is not None:
        assert handle_count() == before_handles


@pytest.mark.parametrize(
    "failure_step",
    (
        "artifact-0",
        "artifact-5",
        "fixture-0",
        "fixture-3",
        "registry",
        "catalog",
        "receipt",
        "before-control-cas",
        "after-control-replace",
        "before-high-water-accept",
    ),
)
def test_promotion_failure_injection_preserves_original_live_bytes_and_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_step: str
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    before = control.capture()

    def fail(stage_name: str) -> None:
        if stage_name == failure_step:
            raise RuntimeError(f"injected {stage_name}")

    with pytest.raises(RuntimeError, match="injected"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
            failure_injector=fail,
        )
    after = control.capture()
    assert after.control_snapshot == before.control_snapshot
    assert after.registry_snapshot == before.registry_snapshot
    assert after.catalog_snapshot == before.catalog_snapshot
    assert after.document.generation == 0


@pytest.mark.parametrize("mutation_step", ("before-control-cas", "after-control-replace"))
def test_concurrent_or_final_reread_control_mismatch_fails_closed_and_restores_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation_step: str
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    before = control.capture()
    competitor = b'{"thirdParty":"control"}\n'

    def mutate(stage_name: str) -> None:
        if stage_name == mutation_step:
            temporary = control.path.with_name(f".{control.path.name}.third-party")
            temporary.write_bytes(competitor)
            os.replace(temporary, control.path)

    try:
        with pytest.raises((OSError, RuntimeError, ValueError), match="changed|reread|ownership"):
            promote_serving_ready(
                accepted=accepted,
                stage=stage,
                verified_smoke=verified,
                stage_root=stage_root,
                fixture_root=fixture_root,
                fixtures=fixtures,
                serving_control=control,
                failure_injector=mutate,
            )
    except AssertionError:
        # Windows may deny a third party replacing a control file while the
        # publisher holds its delete-exclusive handle.  That is also fail-closed.
        assert control.path.read_bytes() == before.control_snapshot
        return
    assert control.path.read_bytes() == competitor


@pytest.mark.skipif(os.name != "nt", reason="reviewer ReplaceFileW race reproduction")
def test_control_swap_restores_competitor_inserted_immediately_before_replace_and_closes_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    competitor = b'{"thirdParty":"immediately-before-replace"}\n'
    original_replace = promotion_module._windows_replace_with_backup
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessHandleCount.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetProcessHandleCount.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    inserted = False
    competitor_identity: tuple[int, int] | None = None

    def handle_count() -> int:
        count = ctypes.c_ulong()
        assert kernel32.GetProcessHandleCount(process, ctypes.byref(count))
        return int(count.value)

    def insert_competitor(target: Path, replacement: Path, backup: Path) -> None:
        nonlocal competitor_identity, inserted
        if not inserted:
            temporary = target.with_name(f".{target.name}.reviewer-competitor")
            temporary.write_bytes(competitor)
            os.replace(temporary, target)
            competitor_identity = promotion_module._path_identity(target)
            inserted = True
        original_replace(target, replacement, backup)

    monkeypatch.setattr(
        promotion_module,
        "_windows_replace_with_backup",
        insert_competitor,
    )
    warm_target = control.path.with_name(".replace-warm-target")
    warm_replacement = control.path.with_name(".replace-warm-replacement")
    warm_backup = control.path.with_name(".replace-warm-backup")
    warm_target.write_bytes(b"old")
    warm_replacement.write_bytes(b"new")
    original_replace(warm_target, warm_replacement, warm_backup)
    warm_target.unlink()
    warm_backup.unlink()
    before_handles = handle_count()
    with pytest.raises(ValueError, match="changed during exact CAS"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
        )

    assert control.path.read_bytes() == competitor
    assert competitor_identity is not None
    assert promotion_module._path_identity(control.path) == competitor_identity
    assert handle_count() == before_handles
    assert not tuple(control.path.parent.glob(f".{control.path.name}.*.swap"))
    assert not tuple(control.path.parent.glob(f".{control.path.name}.*.previous"))
    assert not control.path.with_name(f".{control.path.name}.promotion.lock").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows post-swap lease failure reproduction")
def test_control_swap_lease_construction_failure_returns_ownership_for_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    before = control.capture()
    before_identity = promotion_module._path_identity(control.path)
    original_lease = promotion_module._OwnedFileLease
    failed = False
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessHandleCount.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    kernel32.GetProcessHandleCount.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()

    def handle_count() -> int:
        count = ctypes.c_ulong()
        assert kernel32.GetProcessHandleCount(process, ctypes.byref(count))
        return int(count.value)

    def fail_backup_lease(target: Path, *args, **kwargs):
        nonlocal failed
        if target.name.endswith(".previous"):
            failed = True
            raise OSError("injected backup lease construction failure")
        return original_lease(target, *args, **kwargs)

    warm_target = control.path.with_name(".replace-warm-target")
    warm_replacement = control.path.with_name(".replace-warm-replacement")
    warm_backup = control.path.with_name(".replace-warm-backup")
    warm_target.write_bytes(b"old")
    warm_replacement.write_bytes(b"new")
    promotion_module._windows_replace_with_backup(warm_target, warm_replacement, warm_backup)
    warm_target.unlink()
    warm_backup.unlink()
    monkeypatch.setattr(promotion_module, "_OwnedFileLease", fail_backup_lease)
    before_handles = handle_count()
    with pytest.raises(
        (OSError, RuntimeError),
        match="injected backup lease|ownership changed",
    ):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
        )
    assert failed is True
    assert control.path.read_bytes() == before.control_snapshot
    assert promotion_module._path_identity(control.path) == before_identity
    assert handle_count() == before_handles
    assert not tuple(control.path.parent.glob(f".{control.path.name}.*.swap"))
    assert not tuple(control.path.parent.glob(f".{control.path.name}.*.previous"))
    assert not control.path.with_name(f".{control.path.name}.promotion.lock").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows reverse ReplaceFileW race reproduction")
def test_control_swap_reverse_replace_restores_a_competitor_inserted_after_rollback_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    competitor = b'{"thirdParty":"after-rollback-verify"}\n'
    competitor_identity: tuple[int, int] | None = None
    replace_calls = 0
    original_replace = promotion_module._windows_replace_with_backup

    def race_reverse_replace(target: Path, replacement: Path, backup: Path) -> None:
        nonlocal competitor_identity, replace_calls
        replace_calls += 1
        if replace_calls == 2:
            temporary = target.with_name(f".{target.name}.rollback-competitor")
            temporary.write_bytes(competitor)
            os.replace(temporary, target)
            competitor_identity = promotion_module._path_identity(target)
        original_replace(target, replacement, backup)

    def fail_after_swap(stage_name: str) -> None:
        if stage_name == "after-control-replace":
            raise RuntimeError("injected post-swap failure")

    monkeypatch.setattr(
        promotion_module,
        "_windows_replace_with_backup",
        race_reverse_replace,
    )
    with pytest.raises(RuntimeError, match="injected post-swap failure"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
            failure_injector=fail_after_swap,
        )
    assert replace_calls >= 2
    assert control.path.read_bytes() == competitor
    assert competitor_identity is not None
    assert promotion_module._path_identity(control.path) == competitor_identity


def test_control_swap_after_atomic_exchange_preserves_a_third_party_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    competitor = b'{"thirdParty":"after-atomic-exchange"}\n'
    observed = False
    replacement_blocked = False

    def replace_after_swap(stage_name: str, target: Path) -> None:
        nonlocal observed, replacement_blocked
        if stage_name != "after-atomic-swap":
            return
        observed = True
        temporary = target.with_name(f".{target.name}.after-swap-competitor")
        temporary.write_bytes(competitor)
        try:
            os.replace(temporary, target)
        except OSError:
            replacement_blocked = True
            temporary.unlink()

    monkeypatch.setattr(promotion_module, "_CONTROL_SWAP_SEAM", replace_after_swap)
    try:
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
        )
    except (OSError, RuntimeError, ValueError):
        assert control.path.read_bytes() == competitor
    else:
        assert replacement_blocked is True
        assert control.capture().document.generation == 1
    assert observed is True


def _create_directory_link(link: Path, target: Path) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            pytest.skip(f"directory junction unavailable: {completed.stderr}")
        return
    link.symlink_to(target, target_is_directory=True)


def _remove_directory_link(link: Path) -> None:
    if os.name == "nt":
        os.rmdir(link)
    else:
        link.unlink()


def test_promotion_immutable_publication_never_commits_through_replaced_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "artifacts" / "candidate.bin"
    outside = tmp_path / "outside-artifacts"
    linked = False

    def replace_parent(kind: str, committed: Path) -> None:
        nonlocal linked
        if kind != "evidence":
            return
        committed.parent.rename(outside)
        _create_directory_link(committed.parent, outside)
        linked = True

    monkeypatch.setattr(formal_preflight_module, "_PUBLICATION_SEAM", replace_parent)
    with pytest.raises((OSError, ValueError)):
        promotion_module._atomic_immutable(runtime, target, b"accepted-candidate")

    assert not (outside / target.name).exists()
    if linked:
        _remove_directory_link(target.parent)


def test_promotion_existing_exact_artifact_is_rechecked_after_directory_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    target = runtime / "artifacts" / "candidate.bin"
    expected = b"accepted-candidate"
    promotion_module._atomic_immutable(runtime, target, expected)
    original_flush = formal_preflight_module._PublicationParentLease.flush

    def replace_after_flush(lease) -> None:
        original_flush(lease)
        target.write_bytes(b"third-party-replacement")

    monkeypatch.setattr(
        formal_preflight_module._PublicationParentLease,
        "flush",
        replace_after_flush,
    )
    with pytest.raises((OSError, ValueError), match="changed|denied|access|used"):
        promotion_module._atomic_immutable(runtime, target, expected)


def test_control_swap_never_commits_or_deletes_a_replacement_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    saved: Path | None = None
    competitor = b'{"thirdParty":"temporary"}\n'

    def replace_temporary(stage_name: str, target: Path) -> None:
        nonlocal saved
        if stage_name != "before-final-read":
            return
        swaps = tuple(target.parent.glob(f".{target.name}.*.swap"))
        assert len(swaps) == 1
        temporary = swaps[0]
        saved = temporary.with_name(temporary.name + ".saved")
        temporary.rename(saved)
        temporary.write_bytes(competitor)

    monkeypatch.setattr(promotion_module, "_CONTROL_SWAP_SEAM", replace_temporary)
    with pytest.raises((RuntimeError, ValueError), match="ownership|identity|changed"):
        promote_serving_ready(
            accepted=accepted,
            stage=stage,
            verified_smoke=verified,
            stage_root=stage_root,
            fixture_root=fixture_root,
            fixtures=fixtures,
            serving_control=control,
        )
    assert saved is not None and saved.read_bytes() != competitor
    assert control.path.read_bytes() == competitor


def test_promotion_lock_cleanup_never_unlinks_a_replacement_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    original_close = promotion_module._PublisherLock.close
    replacement: Path | None = None
    attack_blocked = False

    def replace_lock(lock) -> None:
        nonlocal replacement, attack_blocked
        saved = lock.path.with_name(lock.path.name + ".saved")
        try:
            lock.path.rename(saved)
            lock.path.write_bytes(b"COMPETITOR-LOCK")
        except OSError:
            attack_blocked = True
        original_close(lock)
        if not attack_blocked:
            replacement = lock.path

    monkeypatch.setattr(promotion_module._PublisherLock, "close", replace_lock)
    promote_serving_ready(
        accepted=accepted,
        stage=stage,
        verified_smoke=verified,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
        serving_control=control,
    )

    if attack_blocked:
        assert replacement is None
    else:
        assert replacement is not None
        assert replacement.read_bytes() == b"COMPETITOR-LOCK"


def test_concurrent_promotions_have_one_winner_and_one_active_publisher_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    first_holds_lock = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()
    receipts: list[object] = []
    errors: list[BaseException] = []

    def first_inject(stage_name: str) -> None:
        if stage_name == "artifact-0":
            first_holds_lock.set()
            assert release_first.wait(timeout=10)

    def invoke(injector=None) -> None:
        try:
            receipts.append(
                promote_serving_ready(
                    accepted=accepted,
                    stage=stage,
                    verified_smoke=verified,
                    stage_root=stage_root,
                    fixture_root=fixture_root,
                    fixtures=fixtures,
                    serving_control=control,
                    failure_injector=injector,
                )
            )
        except BaseException as error:  # thread result is asserted below
            errors.append(error)

    first = threading.Thread(target=invoke, args=(first_inject,))
    second = threading.Thread(
        target=lambda: (invoke(), second_finished.set()),
    )
    first.start()
    assert first_holds_lock.wait(timeout=10)
    second.start()
    rejected_while_active = second_finished.wait(timeout=2)
    release_first.set()
    first.join(timeout=30)
    second.join(timeout=30)

    assert rejected_while_active is True
    assert len(receipts) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "active" in str(errors[0])
    assert control.capture().document.generation == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows promotion durability contract")
def test_windows_promotion_flushes_parent_directory_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage, accepted, _definition, fixtures, stage_root, fixture_root = (
        _build_candidate_and_fixtures(tmp_path, monkeypatch)
    )
    verified = _sealed_transaction_smoke(stage, accepted, fixtures)
    control = _empty_control(tmp_path / "live")
    calls: list[int] = []
    original = formal_preflight_module._FlushFileBuffers

    def observe(handle: int) -> int:
        calls.append(handle)
        return original(handle)

    monkeypatch.setattr(formal_preflight_module, "_FlushFileBuffers", observe)
    promote_serving_ready(
        accepted=accepted,
        stage=stage,
        verified_smoke=verified,
        stage_root=stage_root,
        fixture_root=fixture_root,
        fixtures=fixtures,
        serving_control=control,
    )
    assert calls
