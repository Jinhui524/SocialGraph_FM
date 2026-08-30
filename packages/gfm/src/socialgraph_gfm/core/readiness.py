"""Evidence-derived readiness for the core experiment and serving path."""

from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core_contracts import (
    CoreReadiness,
    CoreReadinessEvidence,
    CoreReadinessGate,
    CoreReadinessGates,
    CoreReadinessIdentity,
    CoreReadinessReasonCode,
)

from .acceptance import CoreAcceptance
from .formal_preflight import FormalPreflightEvidence
from .promotion import AcceptedCandidate, ServingSmokeReport, VerifiedServingSmoke
from .serving_control import CapturedServingControl, ServingControlStore
from .serving_registry import ServingModel, ServingRegistry


def _gate(
    ready: bool, reason_code: CoreReadinessReasonCode, reason: str
) -> CoreReadinessGate:
    return CoreReadinessGate(
        ready=ready,
        reasonCode=reason_code,
        reason=reason,
    )


def _verify_control_registry(
    store: ServingControlStore,
) -> tuple[CapturedServingControl, dict[str, object]]:
    snapshot = store.capture()
    control = snapshot.document
    registry = snapshot.registry_document
    if hashlib.sha256(snapshot.registry_snapshot).hexdigest() != control.registry.sha256:
        raise ValueError("readiness registry byte hash contradicts serving control")
    registry_hash = canonical_sha256(registry.model_dump(mode="python", by_alias=True))
    if (
        registry_hash != control.registry.semantic_hash
        or registry.generation != control.registry.generation
    ):
        raise ValueError("readiness registry identity contradicts serving control")
    if (
        control.generation != registry.generation
        or control.generation != snapshot.catalog_document.generation
    ):
        raise ValueError("readiness control, registry, and catalog generations contradict")

    registry_path = store.control_root.joinpath(
        *PurePosixPath(control.registry.relative_path).parts
    )
    live_registry = ServingRegistry.load(registry_path, runtime_root=store.control_root)
    capabilities = live_registry.capabilities(registry_snapshot=snapshot.registry_snapshot)
    if (
        capabilities["registryGeneration"] != registry.generation
        or capabilities["registryHash"] != registry_hash
    ):
        raise ValueError("readiness registry capabilities contradict serving control")
    return snapshot, capabilities


def _optional_hash(value: AcceptedCandidate | ServingSmokeReport | None, field: str) -> str | None:
    if value is None:
        return None
    observed = getattr(value, field, None)
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError(f"{field} is missing from machine evidence")
    return observed


def _verify_smoke_model_bindings(
    smoke: ServingSmokeReport,
    model: ServingModel,
) -> None:
    required_tasks = {
        "core.community_resilience_review",
        "core.risk_and_trust_review",
        "core.collaboration_completion",
    }
    if len(model.tasks) != len(required_tasks) or set(model.tasks) != required_tasks:
        raise ValueError("live serving model does not declare the fixed governance tasks")
    for result in smoke.task_results:
        try:
            binding = model.task_head(result.task_id).calibration(result.entity_type)
        except LookupError as error:
            raise ValueError("serving smoke task/entity is absent from live registry") from error
        if (
            result.feature_contract_hash != binding.graph_feature_contract_hash
            or result.adapter_domain != binding.adapter_domain
            or result.adapter_schema_hash != binding.adapter_schema_hash
            or result.adapter_state_hash != binding.adapter_state_hash
            or result.confidence_artifact_hash != binding.calibration_artifact_hash
            or result.confidence_protocol_hash != binding.calibration_protocol_hash
        ):
            raise ValueError("serving smoke task/entity contradicts live registry binding")


def derive_core_readiness(
    *,
    serving_control: ServingControlStore,
    preflight: FormalPreflightEvidence | None = None,
    acceptance: CoreAcceptance | None = None,
    accepted_candidate: AcceptedCandidate | None = None,
    serving_smoke: VerifiedServingSmoke | None = None,
) -> CoreReadiness:
    """Derive readiness solely from validated evidence and a coherent live control."""

    if type(serving_control) is not ServingControlStore:
        raise TypeError("serving_control must be a ServingControlStore")
    if preflight is not None and type(preflight) is not FormalPreflightEvidence:
        raise TypeError("preflight must be FormalPreflightEvidence machine evidence")
    if acceptance is not None and type(acceptance) is not CoreAcceptance:
        raise TypeError("acceptance must be CoreAcceptance machine evidence")
    if accepted_candidate is not None and type(accepted_candidate) is not AcceptedCandidate:
        raise TypeError("accepted_candidate must be AcceptedCandidate machine evidence")
    if serving_smoke is not None:
        if type(serving_smoke) is not VerifiedServingSmoke:
            raise TypeError("serving_smoke must be sealed fresh-process smoke machine evidence")
        serving_smoke.verify()
    candidate = accepted_candidate
    smoke = None if serving_smoke is None else serving_smoke.report

    control, capabilities = _verify_control_registry(serving_control)
    preflight_hash = None if preflight is None else preflight.evidence_hash
    acceptance_hash = None if acceptance is None else acceptance.acceptance_hash
    if (
        preflight is not None
        and acceptance is not None
        and acceptance.accepted
        and acceptance.preflight_evidence_hash != preflight_hash
    ):
        raise ValueError("formal acceptance contradicts supplied preflight evidence")
    corpus_ready = bool(preflight is not None and preflight.formal_ready and preflight.promotable)
    model_validated = bool(
        corpus_ready
        and acceptance is not None
        and acceptance.accepted
        and acceptance.promotable
        and acceptance.preflight_evidence_hash == preflight_hash
    )

    candidate_hash = _optional_hash(candidate, "accepted_hash")
    candidate_acceptance_hash = (
        None if candidate is None else getattr(candidate, "acceptance_hash", None)
    )
    if candidate is not None and (
        getattr(candidate, "accepted", None) is not True
        or getattr(candidate, "status", None) != "accepted"
        or not model_validated
        or candidate_acceptance_hash != acceptance_hash
    ):
        raise ValueError("accepted candidate contradicts formal acceptance evidence")
    if (
        candidate is not None
        and acceptance is not None
        and (
            candidate.candidate_manifest_hash != acceptance.candidate_manifest_hash
            or candidate.source_checkpoint_sha256 != acceptance.candidate_checkpoint_sha256
        )
    ):
        raise ValueError("accepted candidate manifest/checkpoint contradicts acceptance")
    accepted = candidate is not None

    smoke_hash = _optional_hash(smoke, "smoke_hash")
    smoke_succeeded = bool(smoke is not None and getattr(smoke, "succeeded", None) is True)
    if smoke is not None:
        if candidate is None or (
            smoke.accepted_candidate_hash != candidate_hash
            or smoke.acceptance_hash != candidate.acceptance_hash
            or smoke.serving_model_version_id != candidate.serving_model_version_id
            or smoke.serving_model_hash != candidate.serving_model_hash
            or smoke.source_checkpoint_sha256 != candidate.source_checkpoint_sha256
            or smoke.serving_checkpoint_sha256 != candidate.serving_checkpoint_sha256
            or smoke.task_binding_inventory_hash != candidate.task_binding_inventory_hash
        ):
            raise ValueError("serving smoke contradicts accepted candidate evidence")

    capability_models = capabilities.get("models")
    if not isinstance(capability_models, list):
        raise ValueError("readiness registry capabilities contain invalid models")
    serving_models = [
        model for model in control.registry_document.models if model.state == "servingReady"
    ]
    smoke_model_hash = None if smoke is None else smoke.serving_model_hash
    matching_models = [
        model
        for model in serving_models
        if smoke is not None
        and model.model_version_id == smoke.serving_model_version_id
        and model.model_version_hash == smoke.serving_model_hash
        and model.checkpoint.sha256 == smoke.serving_checkpoint_sha256
    ]
    serving_ready = bool(smoke_succeeded and len(matching_models) == 1)
    if serving_ready and smoke is not None:
        _verify_smoke_model_bindings(smoke, matching_models[0])
    if smoke_succeeded and not serving_ready and serving_models:
        raise ValueError("serving smoke contradicts the live serving registry")

    corpus_reason: tuple[CoreReadinessReasonCode, str]
    if preflight is None:
        corpus_reason = ("FORMAL_PREFLIGHT_MISSING", "Formal preflight evidence is absent.")
    elif corpus_ready:
        corpus_reason = ("FORMAL_PREFLIGHT_READY", "All formal corpus requirements are ready.")
    else:
        corpus_reason = (
            "FORMAL_PREFLIGHT_INCOMPLETE",
            "Formal preflight evidence is present but not promotable.",
        )
    model_reason: tuple[CoreReadinessReasonCode, str]
    if acceptance is None:
        model_reason = ("ACCEPTANCE_MISSING", "Formal acceptance evidence is absent.")
    elif model_validated:
        model_reason = ("FORMAL_ACCEPTANCE_PASSED", "Formal acceptance evidence passed.")
    else:
        model_reason = (
            "FORMAL_ACCEPTANCE_REJECTED",
            "Formal acceptance evidence is rejected or not bound to this preflight.",
        )
    accepted_reason: tuple[CoreReadinessReasonCode, str] = (
        ("ACCEPTED_CANDIDATE_VERIFIED", "The accepted candidate is hash-bound.")
        if accepted
        else ("ACCEPTED_CANDIDATE_MISSING", "No accepted candidate evidence is installed.")
    )
    serving_reason: tuple[CoreReadinessReasonCode, str]
    if serving_ready:
        serving_reason = (
            "LIVE_SERVING_EVIDENCE_VERIFIED",
            "Serving smoke and the live registry identify the same serving-ready model.",
        )
    elif smoke is None:
        serving_reason = ("SERVING_SMOKE_MISSING", "Independent serving smoke is absent.")
    elif not smoke_succeeded:
        serving_reason = ("SERVING_SMOKE_FAILED", "Independent serving smoke did not pass.")
    else:
        serving_reason = (
            "LIVE_SERVING_MODEL_MISSING",
            "The smoke-verified model is not serving ready in the live registry.",
        )

    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-readiness/2.0",
        "milestone": "SocialGraph-FM Core",
        "identity": CoreReadinessIdentity(
            controlGeneration=control.document.generation,
            controlHash=control.document.control_hash,
            registryGeneration=control.registry_document.generation,
            registryHash=control.registry_hash,
        ).model_dump(mode="python", by_alias=True),
        "evidence": CoreReadinessEvidence(
            preflightEvidenceHash=preflight_hash,
            acceptanceHash=acceptance_hash,
            acceptedCandidateHash=candidate_hash,
            servingSmokeHash=smoke_hash,
            servingModelHash=(smoke_model_hash if serving_ready else None),
        ).model_dump(mode="python", by_alias=True),
        "gates": CoreReadinessGates(
            corpusReady=_gate(corpus_ready, *corpus_reason),
            modelValidated=_gate(model_validated, *model_reason),
            accepted=_gate(accepted, *accepted_reason),
            coreServingReady=_gate(serving_ready, *serving_reason),
        ).model_dump(mode="python", by_alias=True),
    }
    payload["contentHash"] = canonical_sha256(payload)
    return CoreReadiness.model_validate(payload)


__all__ = ["derive_core_readiness"]
