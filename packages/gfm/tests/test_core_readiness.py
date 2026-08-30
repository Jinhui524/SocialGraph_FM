import hashlib
import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.core.acceptance import CoreAcceptance
from socialgraph_gfm.core.experiments import ExperimentProtocol
from socialgraph_gfm.core.formal_preflight import (
    FORMAL_CORPUS_REQUIREMENTS,
    FormalPreflightEvidence,
    run_formal_preflight,
)
from socialgraph_gfm.core.promotion import (
    AcceptedCandidate,
    ServingSmokeReport,
    ServingSmokeTaskResult,
    _new_verified_smoke,
)
from socialgraph_gfm.core.readiness import derive_core_readiness
from socialgraph_gfm.core.serving_control import ServingControlStore
from socialgraph_gfm.core_contracts import load_core_readiness


PROJECT = Path(__file__).resolve().parents[1]


class _CounterfeitCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    accepted_hash: str = "a" * 64


def _self_hash(payload: dict, field: str) -> dict:
    payload[field] = canonical_sha256(payload)
    return payload


def _write_control_fixture(root: Path, *, registry_generation: int) -> ServingControlStore:
    root.mkdir()
    source = PROJECT / "contracts"
    registry = json.loads((source / "core-serving-registry.json").read_text(encoding="utf-8"))
    registry["generation"] = registry_generation
    registry_bytes = (canonical_json(registry) + "\n").encode("utf-8")
    (root / "core-serving-registry.json").write_bytes(registry_bytes)
    catalog_bytes = (source / "core-serving-graph-catalog.json").read_bytes()
    (root / "core-serving-graph-catalog.json").write_bytes(catalog_bytes)

    control = json.loads((source / "core-serving-control.json").read_text(encoding="utf-8"))
    control["registry"] = {
        "relativePath": "core-serving-registry.json",
        "sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "semanticHash": canonical_sha256(registry),
        "generation": registry_generation,
    }
    control.pop("controlHash")
    _self_hash(control, "controlHash")
    (root / "core-serving-control.json").write_text(
        canonical_json(control) + "\n",
        encoding="utf-8",
    )
    return ServingControlStore.load(
        root / "core-serving-control.json",
        high_water_root=root / "high-water",
    )


def _ready_preflight(root: Path) -> FormalPreflightEvidence:
    root.mkdir()
    base = run_formal_preflight(root).model_dump(mode="json", by_alias=True)
    observations = []
    for ordinal, requirement in enumerate(FORMAL_CORPUS_REQUIREMENTS, start=1):
        prefix = f"experiment-corpus/{requirement.requirement_id}"
        files = [
            {
                "relativePath": requirement.manifest_relative_path,
                "sha256": f"{ordinal:064x}",
                "sizeBytes": 1,
                "purpose": "manifest",
            },
            {
                "relativePath": f"{prefix}/bundle.json",
                "sha256": f"{ordinal + 20:064x}",
                "sizeBytes": 1,
                "purpose": "bundle",
            },
            {
                "relativePath": f"{prefix}/labels.json",
                "sha256": f"{ordinal + 40:064x}",
                "sizeBytes": 1,
                "purpose": "labels",
            },
            {
                "relativePath": f"{prefix}/splits.json",
                "sha256": f"{ordinal + 60:064x}",
                "sizeBytes": 1,
                "purpose": "split-inventory",
            },
            *(
                {
                    "relativePath": path,
                    "sha256": f"{ordinal + 80 + index:064x}",
                    "sizeBytes": 1,
                    "purpose": "raw",
                }
                for index, path in enumerate(requirement.raw_relative_paths)
            ),
        ]
        observations.append(
            {
                "requirementId": requirement.requirement_id,
                "status": "ready",
                "reasonCode": "validated-formal-dataset",
                "manifestHash": f"{ordinal + 120:064x}",
                "graphVersionHash": f"{ordinal + 140:064x}",
                "splitManifestHash": f"{ordinal + 160:064x}",
                "files": sorted(files, key=lambda item: item["relativePath"]),
            }
        )
    base["observations"] = observations
    base["formalReady"] = True
    base["promotable"] = True
    base["evidenceHash"] = canonical_sha256(
        {key: value for key, value in base.items() if key != "evidenceHash"}
    )
    return FormalPreflightEvidence.model_validate(base)


def _accepted_acceptance(
    preflight: FormalPreflightEvidence,
    *,
    preflight_evidence_hash: str | None = None,
) -> CoreAcceptance:
    payload = {
        "schemaVersion": "socialgraph-fm.core-acceptance/1.1",
        "protocolHash": ExperimentProtocol.fixed().protocol_hash,
        "preflightEvidenceHash": (
            preflight.evidence_hash if preflight_evidence_hash is None else preflight_evidence_hash
        ),
        "aggregateHashes": [],
        "transferDecisionHashes": [],
        "rawRecordHashes": [],
        "winnerSelectionHash": "1" * 64,
        "candidateCellId": "2" * 64,
        "candidateRecordHash": "3" * 64,
        "candidateLatestCheckpointSha256": "4" * 64,
        "candidateCheckpointSha256": "5" * 64,
        "candidateManifestHash": "6" * 64,
        "candidateTaskEvidenceHashes": ["7" * 64, "8" * 64, "9" * 64, "a" * 64],
        "freshProcessEvidenceHash": "b" * 64,
        "failedGates": [],
        "status": "accepted",
        "accepted": True,
        "promotable": True,
    }
    _self_hash(payload, "acceptanceHash")
    return CoreAcceptance.model_validate(payload)


def _accepted_candidate(
    acceptance: CoreAcceptance,
    *,
    candidate_manifest_hash: str | None = None,
) -> AcceptedCandidate:
    payload = {
        "schemaVersion": "socialgraph-fm.core-accepted-candidate/1.0",
        "status": "accepted",
        "accepted": True,
        "candidateStageHash": "b" * 64,
        "acceptanceHash": acceptance.acceptance_hash,
        "candidateManifestHash": (
            acceptance.candidate_manifest_hash
            if candidate_manifest_hash is None
            else candidate_manifest_hash
        ),
        "experimentSummaryHash": "c" * 64,
        "sourceCheckpointSha256": acceptance.candidate_checkpoint_sha256,
        "servingCheckpointSha256": "d" * 64,
        "servingModelVersionId": "core/accepted-test",
        "servingModelHash": "e" * 64,
        "taskBindingInventoryHash": "f" * 64,
        "artifactInventoryHash": "1" * 64,
        "acceptanceRevalidationHash": "2" * 64,
    }
    _self_hash(payload, "acceptedHash")
    return AcceptedCandidate.model_validate(payload)


def _serving_smoke(
    candidate: AcceptedCandidate,
    *,
    serving_checkpoint_sha256: str | None = None,
) -> ServingSmokeReport:
    task_entities = (
        ("core.community_resilience_review", "community"),
        ("core.risk_and_trust_review", "node"),
        ("core.risk_and_trust_review", "edge"),
        ("core.collaboration_completion", "node-pair"),
    )
    task_results = []
    for ordinal, (task_id, entity_type) in enumerate(task_entities, start=1):
        payload = {
            "schemaVersion": "socialgraph-fm.core-serving-smoke-task/1.0",
            "taskId": task_id,
            "entityType": entity_type,
            "fixtureArtifactHash": f"{ordinal:064x}",
            "fixtureBundleSha256": f"{ordinal + 10:064x}",
            "graphVersionHash": f"{ordinal + 20:064x}",
            "featureContractHash": f"{ordinal + 30:064x}",
            "adapterDomain": f"domain-{ordinal}",
            "adapterSchemaHash": f"{ordinal + 40:064x}",
            "adapterStateHash": f"{ordinal + 50:064x}",
            "confidenceArtifactHash": f"{ordinal + 60:064x}",
            "confidenceProtocolHash": f"{ordinal + 70:064x}",
            "requestHash": f"{ordinal + 80:064x}",
            "resultHash": f"{ordinal + 90:064x}",
            "findingHashes": [f"{ordinal + 100:064x}"],
            "allPendingHumanReview": True,
        }
        _self_hash(payload, "taskResultHash")
        task_results.append(ServingSmokeTaskResult.model_validate(payload))
    payload = {
        "schemaVersion": "socialgraph-fm.core-serving-smoke/1.0",
        "acceptedCandidateHash": candidate.accepted_hash,
        "acceptanceHash": candidate.acceptance_hash,
        "servingModelVersionId": candidate.serving_model_version_id,
        "servingModelHash": candidate.serving_model_hash,
        "sourceCheckpointSha256": candidate.source_checkpoint_sha256,
        "servingCheckpointSha256": (
            candidate.serving_checkpoint_sha256
            if serving_checkpoint_sha256 is None
            else serving_checkpoint_sha256
        ),
        "taskBindingInventoryHash": candidate.task_binding_inventory_hash,
        "processInterpreterSha256": "3" * 64,
        "processEnvironmentHash": "4" * 64,
        "taskResults": [item.model_dump(mode="python", by_alias=True) for item in task_results],
        "failedGates": [],
        "succeeded": True,
    }
    _self_hash(payload, "smokeHash")
    return ServingSmokeReport.model_validate(payload)


def test_checked_generation_zero_readiness_is_derived_from_empty_control(
    tmp_path: Path,
) -> None:
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    observed = derive_core_readiness(serving_control=control)

    assert observed == load_core_readiness()
    assert observed.identity.control_generation == 0
    assert observed.identity.control_hash == (
        "3d99a1564e2fa5ae779abcaea854d32bb56e90b8473788bd574a628730a6c6d5"
    )
    assert observed.identity.registry_generation == 0
    assert observed.identity.registry_hash == (
        "5e5a906ba9e8e6c8d363c4ede4f74d66a695e2994eb39b717a41a9ca4a97051d"
    )
    assert observed.gates.corpus_ready.ready is False
    assert observed.gates.model_validated.ready is False
    assert observed.gates.accepted.ready is False
    assert observed.gates.core_serving_ready.ready is False


def test_readiness_rejects_boolean_claims_instead_of_treating_them_as_evidence(
    tmp_path: Path,
) -> None:
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    with pytest.raises(TypeError, match="FormalPreflightEvidence machine evidence"):
        derive_core_readiness(
            preflight=True,  # type: ignore[arg-type]
            acceptance=None,
            accepted_candidate=None,
            serving_smoke=None,
            serving_control=control,
        )


def test_readiness_rejects_self_consistent_control_registry_generation_conflict(
    tmp_path: Path,
) -> None:
    control = _write_control_fixture(tmp_path / "control", registry_generation=1)

    with pytest.raises(ValueError, match="generations contradict"):
        derive_core_readiness(
            preflight=None,
            acceptance=None,
            accepted_candidate=None,
            serving_smoke=None,
            serving_control=control,
        )


def test_readiness_rejects_generic_models_as_accepted_candidate_evidence(
    tmp_path: Path,
) -> None:
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    with pytest.raises(TypeError, match="AcceptedCandidate"):
        derive_core_readiness(
            preflight=None,
            acceptance=None,
            accepted_candidate=_CounterfeitCandidate(),  # type: ignore[arg-type]
            serving_smoke=None,
            serving_control=control,
        )


def test_readiness_requires_exact_sealed_evidence_types(tmp_path: Path) -> None:
    preflight = _ready_preflight(tmp_path / "preflight")
    acceptance = _accepted_acceptance(preflight)
    candidate = _accepted_candidate(acceptance)

    class PreflightSubclass(FormalPreflightEvidence):
        pass

    class AcceptanceSubclass(CoreAcceptance):
        pass

    class CandidateSubclass(AcceptedCandidate):
        pass

    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )
    cases = (
        {
            "preflight": PreflightSubclass.model_validate(
                preflight.model_dump(mode="python", by_alias=True)
            ),
            "acceptance": None,
            "accepted_candidate": None,
        },
        {
            "preflight": preflight,
            "acceptance": AcceptanceSubclass.model_validate(
                acceptance.model_dump(mode="python", by_alias=True)
            ),
            "accepted_candidate": None,
        },
        {
            "preflight": preflight,
            "acceptance": acceptance,
            "accepted_candidate": CandidateSubclass.model_validate(
                candidate.model_dump(mode="python", by_alias=True)
            ),
        },
    )
    for evidence in cases:
        with pytest.raises(TypeError, match="machine evidence"):
            derive_core_readiness(
                serving_control=control,
                serving_smoke=None,
                **evidence,
            )


def test_self_hashed_smoke_report_is_not_sealed_readiness_evidence(tmp_path: Path) -> None:
    preflight = _ready_preflight(tmp_path / "preflight")
    acceptance = _accepted_acceptance(preflight)
    candidate = _accepted_candidate(acceptance)
    report = _serving_smoke(candidate)
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    with pytest.raises(TypeError, match="sealed fresh-process smoke"):
        derive_core_readiness(
            preflight=preflight,
            acceptance=acceptance,
            accepted_candidate=candidate,
            serving_smoke=report,
            serving_control=control,
        )


def test_readiness_rejects_rehashed_candidate_with_wrong_accepted_manifest(
    tmp_path: Path,
) -> None:
    preflight = _ready_preflight(tmp_path / "preflight")
    acceptance = _accepted_acceptance(preflight)
    candidate = _accepted_candidate(acceptance, candidate_manifest_hash="f" * 64)
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    with pytest.raises(ValueError, match="candidate manifest"):
        derive_core_readiness(
            preflight=preflight,
            acceptance=acceptance,
            accepted_candidate=candidate,
            serving_smoke=None,
            serving_control=control,
        )


def test_readiness_rejects_accepted_report_bound_to_another_preflight(
    tmp_path: Path,
) -> None:
    preflight = _ready_preflight(tmp_path / "preflight")
    acceptance = _accepted_acceptance(preflight, preflight_evidence_hash="0" * 64)
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    with pytest.raises(ValueError, match="acceptance.*preflight"):
        derive_core_readiness(
            preflight=preflight,
            acceptance=acceptance,
            accepted_candidate=None,
            serving_smoke=None,
            serving_control=control,
        )


def test_accepted_candidate_is_not_serving_ready_without_matching_live_registry(
    tmp_path: Path,
) -> None:
    preflight = _ready_preflight(tmp_path / "preflight")
    acceptance = _accepted_acceptance(preflight)
    candidate = _accepted_candidate(acceptance)
    smoke = _new_verified_smoke(_serving_smoke(candidate), "9" * 64)
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    observed = derive_core_readiness(
        preflight=preflight,
        acceptance=acceptance,
        accepted_candidate=candidate,
        serving_smoke=smoke,
        serving_control=control,
    )

    assert observed.gates.corpus_ready.ready is True
    assert observed.gates.model_validated.ready is True
    assert observed.gates.accepted.ready is True
    assert observed.gates.core_serving_ready.ready is False
    assert observed.evidence.accepted_candidate_hash == candidate.accepted_hash
    assert observed.evidence.serving_smoke_hash == smoke.report.smoke_hash
    assert observed.evidence.serving_model_hash is None


def test_readiness_rejects_rehashed_smoke_with_wrong_candidate_checkpoint(
    tmp_path: Path,
) -> None:
    preflight = _ready_preflight(tmp_path / "preflight")
    acceptance = _accepted_acceptance(preflight)
    candidate = _accepted_candidate(acceptance)
    smoke = _new_verified_smoke(
        _serving_smoke(candidate, serving_checkpoint_sha256="0" * 64),
        "9" * 64,
    )
    control = ServingControlStore.load(
        PROJECT / "contracts" / "core-serving-control.json",
        high_water_root=tmp_path / "high-water",
    )

    with pytest.raises(ValueError, match="serving smoke.*candidate"):
        derive_core_readiness(
            preflight=preflight,
            acceptance=acceptance,
            accepted_candidate=candidate,
            serving_smoke=smoke,
            serving_control=control,
        )
