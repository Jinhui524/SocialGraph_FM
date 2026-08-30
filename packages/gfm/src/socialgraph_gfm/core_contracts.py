"""Fail-closed machine-readable contracts for SocialGraph-FM Core."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .canonical import canonical_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, protected_namespaces=("model_dump",)
    )


class _HashBoundModel(_StrictModel):
    content_hash: str = Field(alias="contentHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_content_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"content_hash"})
        )
        if self.content_hash != expected:
            raise ValueError("contentHash does not match the canonical contract content")
        return self


class CoreTask(_StrictModel):
    task_id: Literal[
        "core.community_resilience_review",
        "core.risk_and_trust_review",
        "core.collaboration_completion",
    ] = Field(alias="taskId")
    display_name: str = Field(alias="displayName", min_length=1, max_length=200)
    enabled: Literal[False]
    human_review_required: Literal[True] = Field(alias="humanReviewRequired")
    description: str = Field(min_length=1, max_length=2000)


class CoreTaskContract(_HashBoundModel):
    schema_version: Literal["socialgraph-fm.core-tasks/1.0"] = Field(alias="schemaVersion")
    family: Literal["collaboration_governance"]
    tasks: tuple[CoreTask, ...]
    legacy_compatibility_smoke_task_ids: tuple[
        Literal[
            "core.community_health_observation",
            "core.newcomer_support",
            "core.coordination_review",
        ],
        ...,
    ] = Field(alias="legacyCompatibilitySmokeTaskIds")
    model_or_serving_readiness_implied: Literal[False] = Field(
        alias="modelOrServingReadinessImplied"
    )

    @model_validator(mode="after")
    def validate_task_inventory(self):
        if {task.task_id for task in self.tasks} != {
            "core.community_resilience_review",
            "core.risk_and_trust_review",
            "core.collaboration_completion",
        } or len(self.tasks) != 3:
            raise ValueError("SocialGraph-FM Core task inventory must contain each public task exactly once")
        if self.legacy_compatibility_smoke_task_ids != (
            "core.community_health_observation",
            "core.newcomer_support",
            "core.coordination_review",
        ):
            raise ValueError("Legacy compatibility smoke task IDs must remain fixed")
        return self


_READINESS_MODEL_CONFIG = ConfigDict(
    extra="forbid",
    frozen=True,
    populate_by_name=True,
    protected_namespaces=("model_dump",),
    strict=True,
)

CoreReadinessReasonCode = Literal[
    "FORMAL_PREFLIGHT_MISSING",
    "FORMAL_PREFLIGHT_READY",
    "FORMAL_PREFLIGHT_INCOMPLETE",
    "ACCEPTANCE_MISSING",
    "FORMAL_ACCEPTANCE_PASSED",
    "FORMAL_ACCEPTANCE_REJECTED",
    "ACCEPTED_CANDIDATE_VERIFIED",
    "ACCEPTED_CANDIDATE_MISSING",
    "LIVE_SERVING_EVIDENCE_VERIFIED",
    "SERVING_SMOKE_MISSING",
    "SERVING_SMOKE_FAILED",
    "LIVE_SERVING_MODEL_MISSING",
]


class CoreReadinessGate(_StrictModel):
    model_config = _READINESS_MODEL_CONFIG

    ready: bool = Field(strict=True)
    reason_code: CoreReadinessReasonCode = Field(alias="reasonCode")
    reason: str = Field(min_length=1, max_length=2000)


class CoreReadinessIdentity(_StrictModel):
    model_config = _READINESS_MODEL_CONFIG

    control_generation: int = Field(alias="controlGeneration", ge=0)
    control_hash: str = Field(alias="controlHash", pattern=r"^[0-9a-f]{64}$")
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=r"^[0-9a-f]{64}$")


class CoreReadinessEvidence(_StrictModel):
    model_config = _READINESS_MODEL_CONFIG

    preflight_evidence_hash: str | None = Field(
        default=None, alias="preflightEvidenceHash", pattern=r"^[0-9a-f]{64}$"
    )
    acceptance_hash: str | None = Field(
        default=None, alias="acceptanceHash", pattern=r"^[0-9a-f]{64}$"
    )
    accepted_candidate_hash: str | None = Field(
        default=None, alias="acceptedCandidateHash", pattern=r"^[0-9a-f]{64}$"
    )
    serving_smoke_hash: str | None = Field(
        default=None, alias="servingSmokeHash", pattern=r"^[0-9a-f]{64}$"
    )
    serving_model_hash: str | None = Field(
        default=None, alias="servingModelHash", pattern=r"^[0-9a-f]{64}$"
    )


class CoreReadinessGates(_StrictModel):
    model_config = _READINESS_MODEL_CONFIG

    corpus_ready: CoreReadinessGate = Field(alias="corpusReady")
    model_validated: CoreReadinessGate = Field(alias="modelValidated")
    accepted: CoreReadinessGate
    core_serving_ready: CoreReadinessGate = Field(alias="coreServingReady")


class CoreReadiness(_HashBoundModel):
    model_config = _READINESS_MODEL_CONFIG

    schema_version: Literal["socialgraph-fm.core-readiness/2.0"] = Field(
        alias="schemaVersion"
    )
    milestone: Literal["SocialGraph-FM Core"]
    identity: CoreReadinessIdentity
    evidence: CoreReadinessEvidence
    gates: CoreReadinessGates

    @model_validator(mode="after")
    def validate_gate_evidence(self):
        reason_contracts = (
            (
                self.gates.corpus_ready,
                {
                    "FORMAL_PREFLIGHT_MISSING",
                    "FORMAL_PREFLIGHT_READY",
                    "FORMAL_PREFLIGHT_INCOMPLETE",
                },
                "FORMAL_PREFLIGHT_READY",
            ),
            (
                self.gates.model_validated,
                {
                    "ACCEPTANCE_MISSING",
                    "FORMAL_ACCEPTANCE_PASSED",
                    "FORMAL_ACCEPTANCE_REJECTED",
                },
                "FORMAL_ACCEPTANCE_PASSED",
            ),
            (
                self.gates.accepted,
                {"ACCEPTED_CANDIDATE_VERIFIED", "ACCEPTED_CANDIDATE_MISSING"},
                "ACCEPTED_CANDIDATE_VERIFIED",
            ),
            (
                self.gates.core_serving_ready,
                {
                    "LIVE_SERVING_EVIDENCE_VERIFIED",
                    "SERVING_SMOKE_MISSING",
                    "SERVING_SMOKE_FAILED",
                    "LIVE_SERVING_MODEL_MISSING",
                },
                "LIVE_SERVING_EVIDENCE_VERIFIED",
            ),
        )
        if any(
            gate.reason_code not in allowed or gate.ready != (gate.reason_code == ready_reason)
            for gate, allowed, ready_reason in reason_contracts
        ):
            raise ValueError("readiness reason code contradicts its gate state")
        if self.gates.model_validated.ready and not self.gates.corpus_ready.ready:
            raise ValueError("model validation requires formal corpus readiness")
        if self.gates.accepted.ready and not self.gates.model_validated.ready:
            raise ValueError("accepted readiness requires model validation")
        if self.gates.core_serving_ready.ready and not self.gates.accepted.ready:
            raise ValueError("serving readiness requires accepted model evidence")
        required_evidence = (
            (self.gates.corpus_ready.ready, self.evidence.preflight_evidence_hash),
            (self.gates.model_validated.ready, self.evidence.acceptance_hash),
            (self.gates.accepted.ready, self.evidence.accepted_candidate_hash),
            (self.gates.core_serving_ready.ready, self.evidence.serving_smoke_hash),
            (self.gates.core_serving_ready.ready, self.evidence.serving_model_hash),
        )
        if any(ready and evidence_hash is None for ready, evidence_hash in required_evidence):
            raise ValueError("ready gates require their machine evidence hashes")
        return self


def _read_contract(path: str) -> str:
    source_path = Path(__file__).resolve().parents[2] / "contracts" / path
    if source_path.is_file():
        return source_path.read_text(encoding="utf-8")
    return (
        resources.files("socialgraph_gfm").joinpath("resources", path).read_text(encoding="utf-8")
    )


def load_core_task_contract() -> CoreTaskContract:
    return CoreTaskContract.model_validate_json(
        _read_contract("core-task-contract.json")
    )


def load_core_readiness() -> CoreReadiness:
    return CoreReadiness.model_validate_json(_read_contract("core-readiness.json"))


__all__ = [
    "CoreReadiness",
    "CoreReadinessEvidence",
    "CoreReadinessGate",
    "CoreReadinessGates",
    "CoreReadinessIdentity",
    "CoreReadinessReasonCode",
    "CoreTaskContract",
    "load_core_readiness",
    "load_core_task_contract",
]
