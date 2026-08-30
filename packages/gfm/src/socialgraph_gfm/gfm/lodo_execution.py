"""Durable, provenance-bound execution state for one formal LODO matrix cell.

The LODO workflow is deliberately represented as one resumable cell rather
than fifteen unrelated training jobs.  A cell has three source stages followed
by twelve target-control stages.  Every committed progress checkpoint carries
the complete source outputs required by later controls and an execution
snapshot, so recovery never infers progress from filenames or modification
times.

This module owns no corpus or model construction.  It is a narrow persistence
boundary used by :mod:`socialgraph_gfm.gfm_workflow` and by small synthetic
trajectory tests.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence

from ..canonical import canonical_sha256
from ..errors import ContractViolation, GfmTrainingError
from .checkpoint import (
    load_gfm_checkpoint,
    read_gfm_checkpoint_manifest,
    save_gfm_checkpoint,
)
from .contracts import GfmCheckpointManifest
from .corpus.common import atomic_write_json, exclusive_file_lock, read_json_object

LODO_RUN_STATE_SCHEMA = "gfm.lodo-run-state/1.0"
LODO_EXECUTION_SCHEMA = "gfm.lodo-execution-snapshot/1.0"
LODO_HEARTBEAT_SCHEMA = "gfm.lodo-heartbeat/1.0"
LODO_CHECKPOINT_SCHEMA = "gfm.lodo-progress-checkpoint/1.0"
HEARTBEAT_EVERY_OPTIMIZER_STEPS = 50

LodoStatus = Literal["preflight", "running", "succeeded"]


@contextmanager
def exclusive_lodo_execution_lock(run_dir: str | Path) -> Iterator[None]:
    """Hold the existing OS-backed, crash-releasing lock for one full cell."""

    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / ".lodo-execution.lock"
    try:
        with exclusive_file_lock(lock):
            yield
    except ContractViolation as error:
        if "exclusive operation is already running" not in str(error):
            raise
        raise GfmTrainingError(
            "LODO cell is already owned by another worker; do not launch concurrent resume"
        ) from error


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_json_tree(value: Any, *, label: str) -> None:
    """Reject values that cannot participate in the canonical state hash."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation(f"{label} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for nested in value:
            _validate_json_tree(nested, label=label)
        return
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractViolation(f"{label} contains a non-string mapping key")
        for nested in value.values():
            _validate_json_tree(nested, label=label)
        return
    raise ContractViolation(f"{label} contains unsupported {type(value).__name__}")


@dataclass(frozen=True)
class LodoCellIdentity:
    """Exact immutable authority for one LODO matrix cell."""

    experiment_id: str
    run_id: str
    held_out_domain: str
    source_domain_ids: tuple[str, ...]
    architecture_variant: Literal["core-base", "core-moe"]
    seed: int
    config_hash: str
    code_hash: str
    environment_hash: str
    corpus_hashes: tuple[str, ...]
    protocol_hashes: tuple[str, ...]
    role_view_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.run_id or not self.held_out_domain:
            raise ContractViolation("LODO cell identity strings must be nonempty")
        if (
            len(self.source_domain_ids) != 2
            or len(set(self.source_domain_ids)) != 2
            or self.held_out_domain in self.source_domain_ids
        ):
            raise ContractViolation("LODO cell identity requires two isolated source domains")
        if self.architecture_variant not in ("core-base", "core-moe"):
            raise ContractViolation("LODO cell architecture variant is invalid")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ContractViolation("LODO cell seed must be a nonnegative integer")
        hashes = (
            self.config_hash,
            self.code_hash,
            self.environment_hash,
            *self.corpus_hashes,
            *self.protocol_hashes,
        )
        if not hashes or any(not _is_sha256(value) for value in hashes):
            raise ContractViolation("LODO cell provenance requires lowercase SHA-256 values")
        if len(self.corpus_hashes) != 3 or len(set(self.corpus_hashes)) != 3:
            raise ContractViolation("LODO cell identity requires the exact three corpus hashes")
        if not self.protocol_hashes or len(set(self.protocol_hashes)) != len(
            self.protocol_hashes
        ):
            raise ContractViolation("LODO cell protocol hashes must be nonempty and unique")
        _validate_json_tree(self.role_view_contract, label="LODO role-view contract")

    def payload(self) -> dict[str, Any]:
        return {
            "schemaVersion": "gfm.lodo-cell-identity/1.0",
            "experimentId": self.experiment_id,
            "runId": self.run_id,
            "heldOutDomain": self.held_out_domain,
            "sourceDomainIds": list(self.source_domain_ids),
            "architectureVariant": self.architecture_variant,
            "seed": self.seed,
            "configHash": self.config_hash,
            "codeHash": self.code_hash,
            "environmentHash": self.environment_hash,
            "corpusHashes": list(self.corpus_hashes),
            "protocolHashes": list(self.protocol_hashes),
            "roleViewContract": dict(self.role_view_contract),
            "roleViewContractHash": canonical_sha256(self.role_view_contract),
        }

    @property
    def identity_hash(self) -> str:
        return canonical_sha256(self.payload())


def lodo_stage_plan(source_domain_ids: Sequence[str]) -> tuple[str, ...]:
    """Return the fixed three-source plus twelve-control execution order."""

    sources = tuple(sorted(str(value) for value in source_domain_ids))
    if len(sources) != 2 or len(set(sources)) != 2 or any(not value for value in sources):
        raise ContractViolation("LODO stage plan requires exactly two source domains")
    stages = ["source:multi", *(f"source:single:{domain}" for domain in sources)]
    for fraction in ("1pct", "5pct", "10pct"):
        stages.extend(
            (
                f"target:{fraction}:gfm",
                f"target:{fraction}:random-init",
                *(f"target:{fraction}:single:{domain}" for domain in sources),
            )
        )
    if len(stages) != 15 or len(set(stages)) != 15:
        raise AssertionError("Internal LODO stage plan is not the fixed 15-stage protocol")
    return tuple(stages)


def _initial_execution(stage_plan: Sequence[str]) -> dict[str, Any]:
    plan = tuple(stage_plan)
    return {
        "schemaVersion": LODO_EXECUTION_SCHEMA,
        "stagePlan": list(plan),
        "stagePlanHash": canonical_sha256(plan),
        "currentStageIndex": 0,
        "currentStage": plan[0],
        "completedStages": {},
        "completedStagesHash": canonical_sha256({}),
        "progressSequence": 0,
        "roleViews": {},
        "roleViewsHash": canonical_sha256({}),
        "selectedIndices": {},
        "selectedIndicesHash": canonical_sha256({}),
        "testReadCount": 0,
    }


def _validate_execution(snapshot: Mapping[str, Any], stage_plan: Sequence[str]) -> None:
    plan = tuple(stage_plan)
    if (
        snapshot.get("schemaVersion") != LODO_EXECUTION_SCHEMA
        or tuple(snapshot.get("stagePlan", ())) != plan
        or snapshot.get("stagePlanHash") != canonical_sha256(plan)
    ):
        raise ContractViolation("LODO execution stage plan differs from its cell")
    index = snapshot.get("currentStageIndex")
    completed = snapshot.get("completedStages")
    sequence = snapshot.get("progressSequence")
    role_views = snapshot.get("roleViews")
    selected = snapshot.get("selectedIndices")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index <= len(plan)
        or isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or not isinstance(completed, dict)
        or not isinstance(role_views, dict)
        or not isinstance(selected, dict)
        or snapshot.get("completedStagesHash") != canonical_sha256(completed)
        or snapshot.get("roleViewsHash") != canonical_sha256(role_views)
        or snapshot.get("selectedIndicesHash") != canonical_sha256(selected)
        or snapshot.get("testReadCount") != 0
    ):
        raise ContractViolation("LODO execution snapshot is malformed or hash-inconsistent")
    expected_stage = None if index == len(plan) else plan[index]
    if snapshot.get("currentStage") != expected_stage:
        raise ContractViolation("LODO execution current stage differs from its cursor")
    if set(completed) != set(plan[:index]):
        raise ContractViolation("LODO completed stages are not an exact ordered prefix")
    for stage, result in completed.items():
        if not isinstance(result, dict):
            raise ContractViolation(f"LODO completed stage {stage} has invalid evidence")
        result_hash = result.get("resultHash")
        unhashed = {key: value for key, value in result.items() if key != "resultHash"}
        if result_hash != canonical_sha256(unhashed):
            raise ContractViolation(f"LODO completed stage {stage} evidence hash differs")
    for stage, evidence in selected.items():
        if stage not in plan or not stage.startswith("target:") or not isinstance(evidence, dict):
            raise ContractViolation("LODO selected-index evidence has an invalid stage")
        values = evidence.get("eventIndices")
        if (
            set(evidence)
            != {
                "eventIndices",
                "eventIndicesHash",
                "fraction",
                "fullTrainEventCount",
                "eligiblePoolCount",
                "eligiblePoolHash",
            }
            or
            not isinstance(values, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
            or len(values) != len(set(values))
            or evidence.get("eventIndicesHash") != canonical_sha256(values)
            or not isinstance(evidence.get("fraction"), (int, float))
            or float(evidence["fraction"]) not in (0.01, 0.05, 0.1)
            or isinstance(evidence.get("fullTrainEventCount"), bool)
            or not isinstance(evidence.get("fullTrainEventCount"), int)
            or isinstance(evidence.get("eligiblePoolCount"), bool)
            or not isinstance(evidence.get("eligiblePoolCount"), int)
            or not 0 < len(values) <= int(evidence["eligiblePoolCount"])
            <= int(evidence["fullTrainEventCount"])
            or not _is_sha256(evidence.get("eligiblePoolHash"))
        ):
            raise ContractViolation("LODO selected indices are malformed or hash-inconsistent")
    _validate_json_tree(snapshot, label="LODO execution snapshot")


def create_lodo_run_state(
    path: str | Path,
    *,
    identity: LodoCellIdentity,
    device: str,
    started_at: datetime | None = None,
) -> dict[str, Any]:
    """Claim a new cell before any optimizer work starts."""

    destination = Path(path)
    if destination.exists():
        raise GfmTrainingError("Immutable LODO run state already exists")
    if device != "cuda":
        raise ContractViolation("Formal LODO durable state requires CUDA")
    plan = lodo_stage_plan(identity.source_domain_ids)
    started = started_at or datetime.now(UTC)
    if started.tzinfo is None:
        raise ContractViolation("LODO start time must be timezone-aware")
    state = {
        "schemaVersion": LODO_RUN_STATE_SCHEMA,
        "runKind": "lodo",
        "runId": identity.run_id,
        "experimentId": identity.experiment_id,
        "status": "preflight",
        "identity": identity.payload(),
        "identityHash": identity.identity_hash,
        "device": device,
        "startedAt": started.isoformat(),
        "execution": _initial_execution(plan),
        "heartbeat": None,
        "latestCheckpointId": None,
        "recoveryCheckpointId": None,
        "bestCheckpointId": None,
        "durableCheckpointStage": None,
        "durableCheckpointStep": None,
        "observedHeartbeatStep": None,
        "finishedAt": None,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    unexpected = {
        entry.name
        for entry in destination.parent.iterdir()
        if entry.name != ".lodo-execution.lock"
    }
    if unexpected:
        raise GfmTrainingError("New LODO cell directory contains unexpected artifacts")
    atomic_write_json(destination, state)
    return state


def validate_lodo_run_state(
    value: Mapping[str, Any],
    *,
    identity: LodoCellIdentity | None = None,
    allowed_statuses: Sequence[LodoStatus] = ("preflight", "running", "succeeded"),
) -> dict[str, Any]:
    """Revalidate the complete durable authority; never trust a partial subset."""

    state = dict(value)
    identity_payload = state.get("identity")
    if (
        state.get("schemaVersion") != LODO_RUN_STATE_SCHEMA
        or state.get("runKind") != "lodo"
        or state.get("status") not in tuple(allowed_statuses)
        or not isinstance(identity_payload, dict)
        or state.get("identityHash") != canonical_sha256(identity_payload)
        or state.get("runId") != identity_payload.get("runId")
        or state.get("experimentId") != identity_payload.get("experimentId")
        or state.get("device") != "cuda"
    ):
        raise ContractViolation("LODO durable run state has invalid identity or status")
    if identity is not None and (
        identity_payload != identity.payload() or state.get("identityHash") != identity.identity_hash
    ):
        raise ContractViolation("Interrupted LODO cell provenance differs from this runtime")
    try:
        started = datetime.fromisoformat(str(state.get("startedAt")))
    except ValueError as error:
        raise ContractViolation("LODO durable state has an invalid start time") from error
    if started.tzinfo is None:
        raise ContractViolation("LODO durable state start time lacks a timezone")
    source_domains = identity_payload.get("sourceDomainIds")
    if not isinstance(source_domains, list):
        raise ContractViolation("LODO durable state lacks source domains")
    plan = lodo_stage_plan(tuple(str(value) for value in source_domains))
    execution = state.get("execution")
    if not isinstance(execution, dict):
        raise ContractViolation("LODO durable state lacks execution progress")
    _validate_execution(execution, plan)
    status = state["status"]
    if status == "preflight" and (
        execution["progressSequence"] != 0
        or state.get("heartbeat") is not None
        or state.get("latestCheckpointId") is not None
        or execution["currentStageIndex"] != 0
    ):
        raise ContractViolation("LODO preflight state already contains optimizer progress")
    if status == "running" and execution["currentStageIndex"] > len(plan):
        raise ContractViolation("LODO running state has passed its stage plan")
    if status == "succeeded" and (
        execution["currentStageIndex"] != len(plan)
        or execution["currentStage"] is not None
        or not isinstance(state.get("bestCheckpointId"), str)
        or state.get("finishedAt") is None
    ):
        raise ContractViolation("LODO succeeded state is not terminally complete")
    heartbeat = state.get("heartbeat")
    if heartbeat is not None:
        if not isinstance(heartbeat, dict) or heartbeat.get("schemaVersion") != LODO_HEARTBEAT_SCHEMA:
            raise ContractViolation("LODO heartbeat schema is invalid")
        losses = heartbeat.get("lastLosses")
        audits = heartbeat.get("negativeSamplingAudits")
        if (
            heartbeat.get("stage") not in plan
            or isinstance(heartbeat.get("optimizerStep"), bool)
            or not isinstance(heartbeat.get("optimizerStep"), int)
            or int(heartbeat["optimizerStep"]) < 0
            or isinstance(heartbeat.get("globalStep"), bool)
            or not isinstance(heartbeat.get("globalStep"), int)
            or int(heartbeat["globalStep"]) < 0
            or not isinstance(losses, dict)
            or not losses
            or any(not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in losses.values())
            or heartbeat.get("executionHash") != canonical_sha256(execution)
            or isinstance(heartbeat.get("elapsedSeconds"), bool)
            or not isinstance(heartbeat.get("elapsedSeconds"), (int, float))
            or not math.isfinite(float(heartbeat["elapsedSeconds"]))
            or float(heartbeat["elapsedSeconds"]) < 0.0
            or isinstance(heartbeat.get("rssMiB"), bool)
            or not isinstance(heartbeat.get("rssMiB"), (int, float))
            or not math.isfinite(float(heartbeat["rssMiB"]))
            or float(heartbeat["rssMiB"]) < 0.0
            or isinstance(heartbeat.get("peakCudaMemoryMiB"), bool)
            or not isinstance(heartbeat.get("peakCudaMemoryMiB"), (int, float))
            or not math.isfinite(float(heartbeat["peakCudaMemoryMiB"]))
            or float(heartbeat["peakCudaMemoryMiB"]) < 0.0
            or not isinstance(audits, dict)
            or set(audits) != set(heartbeat.get("domainCursors", {}))
            or heartbeat.get("negativeSamplingAuditsHash") != canonical_sha256(audits)
        ):
            raise ContractViolation("LODO heartbeat is malformed or stale")
        for audit in audits.values():
            if (
                not isinstance(audit, dict)
                or audit.get("futureUnseenCandidateCount") != 0
                or audit.get("exactNoFalseNegative") is not True
                or audit.get("causal") is not True
                or audit.get("cutoffVisibleCandidatesOnly") is not True
            ):
                raise ContractViolation("LODO heartbeat negative-sampling audit failed")
    if status == "running" and execution["progressSequence"] > 0:
        if not isinstance(state.get("latestCheckpointId"), str):
            raise ContractViolation("LODO running progress lacks a named latest checkpoint")
        if heartbeat is None:
            raise ContractViolation("LODO running progress lacks a heartbeat")
    durable_step = state.get("durableCheckpointStep")
    durable_stage = state.get("durableCheckpointStage")
    observed_step = state.get("observedHeartbeatStep")
    if durable_step is not None and (
        isinstance(durable_step, bool) or not isinstance(durable_step, int) or durable_step < 0
    ):
        raise ContractViolation("LODO durable checkpoint step is invalid")
    if (durable_step is None) != (durable_stage is None) or (
        durable_stage is not None and durable_stage not in plan
    ):
        raise ContractViolation("LODO durable checkpoint stage is invalid")
    if observed_step is not None and (
        isinstance(observed_step, bool)
        or not isinstance(observed_step, int)
        or observed_step < 0
        or durable_step is None
        or observed_step < durable_step
    ):
        raise ContractViolation("LODO observed heartbeat step is invalid")
    if heartbeat is not None and observed_step != heartbeat.get("optimizerStep"):
        raise ContractViolation("LODO observed heartbeat step differs from heartbeat")
    _validate_json_tree(state, label="LODO run state")
    return state


def load_lodo_run_state(
    path: str | Path,
    *,
    identity: LodoCellIdentity | None = None,
    allowed_statuses: Sequence[LodoStatus] = ("preflight", "running", "succeeded"),
) -> dict[str, Any]:
    return validate_lodo_run_state(
        read_json_object(Path(path)), identity=identity, allowed_statuses=allowed_statuses
    )


def persist_lodo_run_state(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    identity: LodoCellIdentity,
    allowed_statuses: Sequence[LodoStatus] = ("preflight", "running"),
) -> dict[str, Any]:
    """Atomically persist a validated metadata-only state transition.

    This is used before optimizer work to bind newly opened role views and the
    deterministic target few-shot rows.  Tensor progress must instead use
    :func:`commit_lodo_progress` so metadata can never run ahead of a checkpoint.
    """

    destination = Path(path)
    current = load_lodo_run_state(
        destination, identity=identity, allowed_statuses=allowed_statuses
    )
    checked = validate_lodo_run_state(
        value, identity=identity, allowed_statuses=allowed_statuses
    )
    immutable = (
        "schemaVersion",
        "runKind",
        "runId",
        "experimentId",
        "identity",
        "identityHash",
        "device",
        "startedAt",
        "latestCheckpointId",
        "recoveryCheckpointId",
        "bestCheckpointId",
        "durableCheckpointStage",
        "durableCheckpointStep",
        "observedHeartbeatStep",
        "finishedAt",
    )
    if any(current.get(key) != checked.get(key) for key in immutable):
        raise ContractViolation("LODO metadata transition changed immutable progress authority")
    if (
        current["execution"]["currentStageIndex"]
        != checked["execution"]["currentStageIndex"]
        or current["execution"]["completedStages"]
        != checked["execution"]["completedStages"]
        or current["execution"]["progressSequence"]
        != checked["execution"]["progressSequence"]
    ):
        raise ContractViolation("LODO metadata transition attempted to advance optimizer progress")
    atomic_write_json(destination, checked)
    return load_lodo_run_state(
        destination, identity=identity, allowed_statuses=allowed_statuses
    )


def _metadata_transition(
    state: Mapping[str, Any],
    *,
    execution: Mapping[str, Any],
    status: LodoStatus | None = None,
) -> dict[str, Any]:
    """Rebind operator observations when execution metadata changes atomically."""

    transitioned = dict(state)
    transitioned["execution"] = dict(execution)
    if status is not None:
        transitioned["status"] = status
    heartbeat = transitioned.get("heartbeat")
    if heartbeat is not None:
        transitioned["heartbeat"] = {
            **dict(heartbeat),
            "executionHash": canonical_sha256(execution),
        }
    return transitioned


def bind_lodo_role_views(
    state: Mapping[str, Any], *, role_views: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a state copy with exact opened role-view audits hash-bound."""

    checked = validate_lodo_run_state(state)
    execution = dict(checked["execution"])
    existing = dict(execution["roleViews"])
    for domain, evidence in role_views.items():
        if domain in existing and existing[domain] != evidence:
            raise ContractViolation("LODO role-view evidence changed after first access")
        existing[str(domain)] = dict(evidence)
    _validate_json_tree(existing, label="LODO role-view evidence")
    execution["roleViews"] = existing
    execution["roleViewsHash"] = canonical_sha256(existing)
    return _metadata_transition(checked, execution=execution)


def bind_lodo_selected_indices(
    state: Mapping[str, Any],
    *,
    stage: str,
    event_indices: Sequence[int],
    fraction: float | None = None,
    full_train_event_count: int | None = None,
    eligible_pool_count: int | None = None,
    eligible_pool_hash: str | None = None,
) -> dict[str, Any]:
    """Bind the deterministic train-only few-shot rows before target updates."""

    checked = validate_lodo_run_state(state)
    execution = dict(checked["execution"])
    if stage != execution.get("currentStage") or not stage.startswith("target:"):
        raise ContractViolation("LODO selected rows do not belong to the current target stage")
    indices = [int(value) for value in event_indices]
    if not indices or any(value < 0 for value in indices) or len(indices) != len(set(indices)):
        raise ContractViolation("LODO selected event indices must be nonempty and unique")
    selected = dict(execution["selectedIndices"])
    if (
        fraction not in (0.01, 0.05, 0.1)
        or isinstance(full_train_event_count, bool)
        or not isinstance(full_train_event_count, int)
        or isinstance(eligible_pool_count, bool)
        or not isinstance(eligible_pool_count, int)
        or not 0 < len(indices) <= eligible_pool_count <= full_train_event_count
        or not _is_sha256(eligible_pool_hash)
    ):
        raise ContractViolation("LODO few-shot pool provenance is invalid")
    evidence = {
        "eventIndices": indices,
        "eventIndicesHash": canonical_sha256(indices),
        "fraction": float(fraction),
        "fullTrainEventCount": full_train_event_count,
        "eligiblePoolCount": eligible_pool_count,
        "eligiblePoolHash": eligible_pool_hash,
    }
    if stage in selected and selected[stage] != evidence:
        raise ContractViolation("LODO deterministic few-shot rows changed during resume")
    selected[stage] = evidence
    execution["selectedIndices"] = selected
    execution["selectedIndicesHash"] = canonical_sha256(selected)
    status: LodoStatus = "running" if checked["status"] == "preflight" else checked["status"]
    return _metadata_transition(checked, execution=execution, status=status)


def complete_lodo_stage(
    state: Mapping[str, Any], *, stage: str, result: Mapping[str, Any]
) -> dict[str, Any]:
    """Advance exactly one ordered stage and hash-bind its finite result."""

    checked = validate_lodo_run_state(state, allowed_statuses=("preflight", "running"))
    execution = dict(checked["execution"])
    if stage != execution.get("currentStage"):
        raise ContractViolation("LODO can only complete its current ordered stage")
    result_payload = dict(result)
    if "resultHash" in result_payload:
        raise ContractViolation("LODO stage resultHash is execution-owned")
    _validate_json_tree(result_payload, label="LODO stage result")
    result_payload["resultHash"] = canonical_sha256(result_payload)
    completed = dict(execution["completedStages"])
    completed[stage] = result_payload
    index = int(execution["currentStageIndex"]) + 1
    plan = tuple(execution["stagePlan"])
    execution.update(
        {
            "currentStageIndex": index,
            "currentStage": None if index == len(plan) else plan[index],
            "completedStages": completed,
            "completedStagesHash": canonical_sha256(completed),
        }
    )
    return _metadata_transition(checked, execution=execution, status="running")


def record_lodo_heartbeat(
    state_path: str | Path,
    *,
    identity: LodoCellIdentity,
    stage: str,
    optimizer_step: int,
    global_step: int,
    last_losses: Mapping[str, float],
    stream_states: Mapping[str, Any],
    elapsed_seconds: float,
    rss_mib: float,
    peak_cuda_memory_mib: float,
) -> dict[str, Any]:
    """Persist lightweight observed progress without moving resume authority."""

    destination = Path(state_path)
    state = load_lodo_run_state(
        destination, identity=identity, allowed_statuses=("running",)
    )
    if stage != state["execution"]["currentStage"]:
        raise ContractViolation("LODO heartbeat does not belong to the current stage")
    durable = state.get("durableCheckpointStep")
    if (
        isinstance(optimizer_step, bool)
        or not isinstance(optimizer_step, int)
        or optimizer_step < 0
        or durable is None
        or state.get("durableCheckpointStage") != stage
        or optimizer_step < durable
        or isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < optimizer_step
    ):
        raise ContractViolation("LODO heartbeat counters are invalid")
    losses = {str(name): float(value) for name, value in last_losses.items()}
    if not losses or any(not math.isfinite(value) for value in losses.values()):
        raise GfmTrainingError("LODO heartbeat requires finite current losses")
    resources = (elapsed_seconds, rss_mib, peak_cuda_memory_mib)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in resources
    ):
        raise ContractViolation("LODO heartbeat resource observations are invalid")
    negative_audits: dict[str, Any] = {}
    for domain, value in sorted(stream_states.items()):
        audit = value.get("negativeSamplingAudit") if isinstance(value, Mapping) else None
        if (
            not isinstance(audit, dict)
            or audit.get("futureUnseenCandidateCount") != 0
            or audit.get("exactNoFalseNegative") is not True
            or audit.get("causal") is not True
            or audit.get("cutoffVisibleCandidatesOnly") is not True
        ):
            raise ContractViolation("LODO heartbeat requires passing per-domain sampler audits")
        negative_audits[str(domain)] = dict(audit)
    heartbeat = {
        "schemaVersion": LODO_HEARTBEAT_SCHEMA,
        "recordedAt": datetime.now(UTC).isoformat(),
        "stage": stage,
        "optimizerStep": optimizer_step,
        "globalStep": global_step,
        "lastLosses": losses,
        "domainCursors": {
            domain: {"cursor": value.get("cursor"), "epoch": value.get("epoch")}
            for domain, value in sorted(stream_states.items())
            if isinstance(value, Mapping)
        },
        "elapsedSeconds": float(elapsed_seconds),
        "rssMiB": float(rss_mib),
        "peakCudaMemoryMiB": float(peak_cuda_memory_mib),
        "negativeSamplingAudits": negative_audits,
        "negativeSamplingAuditsHash": canonical_sha256(negative_audits),
        "executionHash": canonical_sha256(state["execution"]),
    }
    observed = {
        **state,
        "heartbeat": heartbeat,
        "observedHeartbeatStep": optimizer_step,
    }
    atomic_write_json(destination, observed)
    return load_lodo_run_state(
        destination, identity=identity, allowed_statuses=("running",)
    )


def _remove_checkpoint(checkpoint_dir: Path, manifest_path: Path) -> None:
    identity = manifest_path.stem.removesuffix(".manifest")
    try:
        manifest = read_gfm_checkpoint_manifest(manifest_path)
        artifact = Path(manifest.artifact_path).resolve()
    except Exception:
        artifact = (checkpoint_dir / f"{identity}.pt").resolve()
    try:
        artifact.relative_to(checkpoint_dir.resolve())
        manifest_path.resolve().relative_to(checkpoint_dir.resolve())
    except ValueError as error:
        raise GfmTrainingError("LODO checkpoint cleanup escaped its cell") from error
    artifact.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)


def _copy_checkpoint_role(
    checkpoint_dir: Path,
    *,
    source: GfmCheckpointManifest,
    role: Literal["recovery"],
) -> GfmCheckpointManifest:
    payload = load_gfm_checkpoint(source, map_location="cpu")
    checkpoint_id = f"{source.run_id}-{role}-{source.step}-{source.state_hash[:10]}"
    return save_gfm_checkpoint(
        checkpoint_dir,
        checkpoint_id=checkpoint_id,
        run_id=source.run_id,
        epoch=source.epoch,
        step=source.step,
        components=payload["components"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=payload["scaler_state"],
        sampler_state=payload["sampler_state"],
        best_state=payload["best_state"],
        config=payload["config"],
        corpus_hashes=payload["corpus_hashes"],
        rng_state=payload["rng_state"],
    )


def _normalize_roles(
    checkpoint_dir: Path,
    *,
    run_id: str,
    retained_ids: Sequence[str | None],
) -> None:
    retained = {value for value in retained_ids if value is not None}
    for role in ("latest", "recovery", "best"):
        for path in checkpoint_dir.glob(f"{run_id}-{role}-*.manifest.json"):
            identity = path.stem.removesuffix(".manifest")
            if identity not in retained:
                _remove_checkpoint(checkpoint_dir, path)


def commit_lodo_progress(
    state_path: str | Path,
    *,
    identity: LodoCellIdentity,
    stage: str,
    optimizer_step: int,
    global_step: int,
    last_losses: Mapping[str, float],
    components: Mapping[str, Mapping[str, Any]],
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any],
    scaler_state: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
    stream_states: Mapping[str, Any],
    best_state: Mapping[str, Any],
    config: Mapping[str, Any],
    corpus_hashes: Sequence[str],
    state_override: Mapping[str, Any] | None = None,
    elapsed_seconds: float = 0.0,
    rss_mib: float = 0.0,
    peak_cuda_memory_mib: float = 0.0,
) -> tuple[dict[str, Any], GfmCheckpointManifest]:
    """Atomically commit an optimizer-aligned heartbeat and keep three roles.

    ``state_override`` is used for a stage-completion commit: the checkpoint
    carries the already-advanced execution snapshot, while the supplied
    ``stage`` still names the optimizer state that produced that output.
    """

    destination = Path(state_path)
    current = load_lodo_run_state(
        destination, identity=identity, allowed_statuses=("preflight", "running")
    )
    proposed = (
        current
        if state_override is None
        else validate_lodo_run_state(
            state_override, identity=identity, allowed_statuses=("preflight", "running")
        )
    )
    current_stage = current["execution"]["currentStage"]
    completed_now = (
        stage in proposed["execution"]["completedStages"]
        and stage not in current["execution"]["completedStages"]
    )
    if stage != current_stage or (state_override is not None and not completed_now):
        raise ContractViolation("LODO progress does not match the active or completed stage")
    if (
        isinstance(optimizer_step, bool)
        or optimizer_step < 0
        or isinstance(global_step, bool)
        or global_step < optimizer_step
    ):
        raise ContractViolation("LODO progress counters are invalid")
    losses = {str(name): float(value) for name, value in last_losses.items()}
    if not losses or any(not math.isfinite(value) for value in losses.values()):
        raise GfmTrainingError("LODO heartbeat requires finite current losses")
    _validate_json_tree(trainer_state, label="LODO trainer state")
    _validate_json_tree(stream_states, label="LODO stream state")
    if (
        trainer_state.get("optimizerStep") != optimizer_step
        or trainer_state.get("globalStep") != global_step
    ):
        raise ContractViolation("LODO checkpoint counters differ from trainer state")
    resources = (elapsed_seconds, rss_mib, peak_cuda_memory_mib)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in resources
    ):
        raise ContractViolation("LODO checkpoint resource observations are invalid")
    checkpoint_dir = destination.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    recovery: GfmCheckpointManifest | None = None
    latest_id = current.get("latestCheckpointId")
    if latest_id is not None:
        if not isinstance(latest_id, str):
            raise ContractViolation("LODO latest checkpoint identity is malformed")
        latest_path = checkpoint_dir / f"{latest_id}.manifest.json"
        latest = read_gfm_checkpoint_manifest(latest_path)
        load_gfm_checkpoint(latest, map_location="cpu")
        recovery = _copy_checkpoint_role(checkpoint_dir, source=latest, role="recovery")
    elif current["status"] == "running" and int(current["execution"]["progressSequence"]) > 0:
        raise ContractViolation("LODO running state lost its latest checkpoint authority")

    execution = dict(proposed["execution"])
    sequence = int(execution["progressSequence"]) + 1
    execution["progressSequence"] = sequence
    checkpoint_stage = stage.replace(":", "-")
    checkpoint_id = f"{identity.run_id}-latest-{sequence}-{checkpoint_stage}-{optimizer_step}"
    now = datetime.now(UTC)
    negative_audits = {
        domain: dict(value["negativeSamplingAudit"])
        for domain, value in sorted(stream_states.items())
        if isinstance(value, Mapping)
        and isinstance(value.get("negativeSamplingAudit"), Mapping)
    }
    heartbeat = {
        "schemaVersion": LODO_HEARTBEAT_SCHEMA,
        "recordedAt": now.isoformat(),
        "stage": stage,
        "optimizerStep": optimizer_step,
        "globalStep": global_step,
        "lastLosses": losses,
        "domainCursors": {
            domain: {
                "cursor": value.get("cursor"),
                "epoch": value.get("epoch"),
            }
            for domain, value in sorted(stream_states.items())
            if isinstance(value, Mapping)
        },
        "elapsedSeconds": float(elapsed_seconds),
        "rssMiB": float(rss_mib),
        "peakCudaMemoryMiB": float(peak_cuda_memory_mib),
        "negativeSamplingAudits": negative_audits,
        "negativeSamplingAuditsHash": canonical_sha256(negative_audits),
        "executionHash": canonical_sha256(execution),
    }
    embedded = {
        "schemaVersion": LODO_CHECKPOINT_SCHEMA,
        "identityHash": identity.identity_hash,
        "stage": stage,
        "optimizerStep": optimizer_step,
        "globalStep": global_step,
        "execution": execution,
        "executionHash": canonical_sha256(execution),
        "trainerState": dict(trainer_state),
        "streamStates": dict(stream_states),
        "roleViewsHash": execution["roleViewsHash"],
        "selectedIndicesHash": execution["selectedIndicesHash"],
        "heartbeat": heartbeat,
        "heartbeatHash": canonical_sha256(heartbeat),
        "split": "train-validation-only",
        "testReadCount": 0,
    }
    progress = save_gfm_checkpoint(
        checkpoint_dir,
        checkpoint_id=checkpoint_id,
        run_id=identity.run_id,
        epoch=0,
        step=optimizer_step,
        components=components,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        sampler_state=embedded,
        best_state=best_state,
        config=config,
        corpus_hashes=tuple(corpus_hashes),
    )
    committed = {
        **proposed,
        "status": "running",
        "execution": execution,
        "heartbeat": heartbeat,
        "latestCheckpointId": progress.checkpoint_id,
        "recoveryCheckpointId": None if recovery is None else recovery.checkpoint_id,
        "durableCheckpointStage": stage,
        "durableCheckpointStep": optimizer_step,
        "observedHeartbeatStep": optimizer_step,
    }
    atomic_write_json(destination, committed)
    checked = load_lodo_run_state(destination, identity=identity, allowed_statuses=("running",))
    _normalize_roles(
        checkpoint_dir,
        run_id=identity.run_id,
        retained_ids=(
            checked.get("bestCheckpointId"),
            checked.get("latestCheckpointId"),
            checked.get("recoveryCheckpointId"),
        ),
    )
    return checked, progress


def load_lodo_resume_checkpoint(
    state_path: str | Path,
    *,
    identity: LodoCellIdentity,
) -> tuple[dict[str, Any], GfmCheckpointManifest | None, dict[str, Any] | None]:
    """Return the named latest, or its named recovery, after deep validation."""

    path = Path(state_path)
    state = load_lodo_run_state(
        path, identity=identity, allowed_statuses=("preflight", "running")
    )
    if state["status"] == "preflight":
        return state, None, None
    checkpoint_dir = path.parent / "checkpoints"
    failures: list[str] = []
    for role in ("latestCheckpointId", "recoveryCheckpointId"):
        checkpoint_id = state.get(role)
        if checkpoint_id is None:
            continue
        if not isinstance(checkpoint_id, str):
            raise ContractViolation("LODO progress checkpoint identity is malformed")
        try:
            manifest = read_gfm_checkpoint_manifest(
                checkpoint_dir / f"{checkpoint_id}.manifest.json"
            )
            payload = load_gfm_checkpoint(manifest, map_location="cpu")
            sampler = payload.get("sampler_state")
            if (
                manifest.run_id != identity.run_id
                or manifest.config_hash != identity.config_hash
                or tuple(manifest.corpus_hashes) != identity.corpus_hashes
                or not isinstance(sampler, dict)
                or sampler.get("schemaVersion") != LODO_CHECKPOINT_SCHEMA
                or sampler.get("identityHash") != identity.identity_hash
                or sampler.get("executionHash") != canonical_sha256(sampler.get("execution"))
                or sampler.get("testReadCount") != 0
            ):
                raise ContractViolation("LODO progress checkpoint provenance differs")
            execution = sampler.get("execution")
            if not isinstance(execution, dict):
                raise ContractViolation("LODO progress checkpoint execution is absent")
            _validate_execution(execution, lodo_stage_plan(identity.source_domain_ids))
            selected = execution["selectedIndices"]
            if sampler.get("selectedIndicesHash") != canonical_sha256(selected):
                raise ContractViolation("LODO progress selected-index hash differs")
            checkpoint_heartbeat = sampler.get("heartbeat")
            trainer_state = sampler.get("trainerState")
            if (
                not isinstance(checkpoint_heartbeat, dict)
                or sampler.get("heartbeatHash") != canonical_sha256(checkpoint_heartbeat)
                or not isinstance(trainer_state, dict)
                or checkpoint_heartbeat.get("stage") != sampler.get("stage")
                or checkpoint_heartbeat.get("optimizerStep") != manifest.step
                or checkpoint_heartbeat.get("optimizerStep")
                != trainer_state.get("optimizerStep")
                or checkpoint_heartbeat.get("globalStep") != sampler.get("globalStep")
                or checkpoint_heartbeat.get("globalStep") != trainer_state.get("globalStep")
                or checkpoint_heartbeat.get("executionHash")
                != canonical_sha256(execution)
            ):
                raise ContractViolation("LODO checkpoint heartbeat differs from durable progress")
            # Rewind the mutable state only to an integrity-checked checkpoint.
            adopted = {
                **state,
                "status": "running",
                "execution": execution,
                "heartbeat": dict(checkpoint_heartbeat),
                "latestCheckpointId": manifest.checkpoint_id,
                "recoveryCheckpointId": (
                    state.get("recoveryCheckpointId")
                    if role == "latestCheckpointId"
                    else None
                ),
                "durableCheckpointStage": sampler["stage"],
                "durableCheckpointStep": manifest.step,
                "observedHeartbeatStep": manifest.step,
            }
            atomic_write_json(path, adopted)
            checked = load_lodo_run_state(
                path, identity=identity, allowed_statuses=("running",)
            )
            _normalize_roles(
                checkpoint_dir,
                run_id=identity.run_id,
                retained_ids=(
                    checked.get("bestCheckpointId"),
                    checked.get("latestCheckpointId"),
                    checked.get("recoveryCheckpointId"),
                ),
            )
            return checked, manifest, payload
        except Exception as error:
            failures.append(f"{role}:{type(error).__name__}")
    raise GfmTrainingError(
        "No LODO progress checkpoint passed integrity verification: " + ",".join(failures)
    )


def mark_lodo_succeeded(
    state_path: str | Path,
    *,
    identity: LodoCellIdentity,
    best_checkpoint_id: str,
    finished_at: datetime | None = None,
) -> dict[str, Any]:
    """Commit the terminal marker only after every selection stage completed."""

    destination = Path(state_path)
    state = load_lodo_run_state(
        destination, identity=identity, allowed_statuses=("running",)
    )
    execution = state["execution"]
    if execution["currentStage"] is not None or execution["testReadCount"] != 0:
        raise GfmTrainingError("LODO cannot succeed before all validation selections complete")
    manifest = read_gfm_checkpoint_manifest(
        destination.parent / "checkpoints" / f"{best_checkpoint_id}.manifest.json"
    )
    payload = load_gfm_checkpoint(manifest, map_location="cpu")
    sampler = payload.get("sampler_state")
    if (
        manifest.run_id != identity.run_id
        or manifest.checkpoint_id != best_checkpoint_id
        or manifest.config_hash != identity.config_hash
        or tuple(manifest.corpus_hashes) != identity.corpus_hashes
        or not isinstance(sampler, dict)
        or sampler.get("execution") != execution
        or sampler.get("executionHash") != canonical_sha256(execution)
        or sampler.get("roleViewsHash") != execution["roleViewsHash"]
        or sampler.get("testReadCount") != 0
    ):
        raise ContractViolation("LODO terminal best checkpoint provenance differs")
    finished = finished_at or datetime.now(UTC)
    terminal = {
        **state,
        "status": "succeeded",
        "bestCheckpointId": best_checkpoint_id,
        "finishedAt": finished.isoformat(),
    }
    atomic_write_json(destination, terminal)
    checked = load_lodo_run_state(
        destination, identity=identity, allowed_statuses=("succeeded",)
    )
    _normalize_roles(
        destination.parent / "checkpoints",
        run_id=identity.run_id,
        retained_ids=(
            checked.get("bestCheckpointId"),
            checked.get("latestCheckpointId"),
            checked.get("recoveryCheckpointId"),
        ),
    )
    return checked


__all__ = [
    "HEARTBEAT_EVERY_OPTIMIZER_STEPS",
    "LODO_CHECKPOINT_SCHEMA",
    "LODO_EXECUTION_SCHEMA",
    "LODO_HEARTBEAT_SCHEMA",
    "LODO_RUN_STATE_SCHEMA",
    "LodoCellIdentity",
    "bind_lodo_role_views",
    "bind_lodo_selected_indices",
    "commit_lodo_progress",
    "complete_lodo_stage",
    "create_lodo_run_state",
    "exclusive_lodo_execution_lock",
    "load_lodo_resume_checkpoint",
    "load_lodo_run_state",
    "lodo_stage_plan",
    "mark_lodo_succeeded",
    "persist_lodo_run_state",
    "record_lodo_heartbeat",
    "validate_lodo_run_state",
]
