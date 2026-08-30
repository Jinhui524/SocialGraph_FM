from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

import pytest
import torch
from pydantic import ValidationError

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.resource_telemetry import (
    ResourceTelemetryRecord,
    ResourceTelemetryRecorder,
    ResourceTelemetrySample,
    VerifiedResourceTelemetry,
    verify_resource_telemetry,
)
from socialgraph_gfm.tensor_digest import canonical_tensor_digest


CELL_ID = "1" * 64
CONFIG_HASH = "2" * 64
DATA_HASH = "3" * 64
CODE_HASH = "4" * 64
ENVIRONMENT_HASH = "5" * 64
LATEST_CHECKPOINT_HASH = "6" * 64
BEST_CHECKPOINT_HASH = "7" * 64


def _model_state(step: int) -> dict[str, torch.Tensor]:
    return {"encoder.weight": torch.tensor([[float(step), float(step + 1)]], dtype=torch.float32)}


def _model_hash(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


def _fit_state(step: int, state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.test-fit-state/1.0",
        "optimizerStep": step,
        "checkpointModelStateHash": _model_hash(state),
    }
    payload["stateHash"] = canonical_sha256(payload)
    return payload


def _recorder(*, phase: Literal["smoke", "dev", "formal"] = "formal") -> ResourceTelemetryRecorder:
    return ResourceTelemetryRecorder(
        cell_id=CELL_ID,
        run_id="formal-run-20260821",
        phase=phase,
        config_hash=CONFIG_HASH,
        data_hash=DATA_HASH,
        code_hash=CODE_HASH,
        environment_hash=ENVIRONMENT_HASH,
    )


def _record_formal_through(
    recorder: ResourceTelemetryRecorder, *, final_step: int
) -> VerifiedResourceTelemetry:
    state = _model_state(0)
    recorder.record_start(model_state=state, fit_state=_fit_state(0, state))
    for step in range(250, final_step, 250):
        state = _model_state(step)
        recorder.record_checkpoint(
            optimizer_step=step,
            model_state=state,
            fit_state=_fit_state(step, state),
        )
    state = _model_state(final_step)
    return recorder.finish(
        final_optimizer_step=final_step,
        model_state=state,
        fit_state=_fit_state(final_step, state),
        latest_checkpoint_semantic_hash=LATEST_CHECKPOINT_HASH,
        best_checkpoint_semantic_hash=BEST_CHECKPOINT_HASH,
    )


def test_recorder_captures_real_hash_chained_formal_resource_evidence() -> None:
    recorder = _recorder()
    state = _model_state(0)
    recorder.record_start(model_state=state, fit_state=_fit_state(0, state))
    for step in range(250, 2000, 250):
        state = _model_state(step)
        recorder.record_checkpoint(
            optimizer_step=step,
            model_state=state,
            fit_state=_fit_state(step, state),
        )

    with recorder.measure_data_wait():
        time.sleep(0.001)

    final_state = _model_state(2000)
    verified = recorder.finish(
        final_optimizer_step=2000,
        model_state=final_state,
        fit_state=_fit_state(2000, final_state),
        latest_checkpoint_semantic_hash=LATEST_CHECKPOINT_HASH,
        best_checkpoint_semantic_hash=BEST_CHECKPOINT_HASH,
    )

    record = verify_resource_telemetry(verified)
    assert record.cell_id == CELL_ID
    assert record.run_id == "formal-run-20260821"
    assert record.optimizer_steps == 2000
    assert record.latest_checkpoint_semantic_hash == LATEST_CHECKPOINT_HASH
    assert record.best_checkpoint_semantic_hash == BEST_CHECKPOINT_HASH
    assert tuple(sample.optimizer_step for sample in record.samples) == tuple(range(0, 2001, 250))
    assert all(
        right.monotonic_ns > left.monotonic_ns
        for left, right in zip(record.samples, record.samples[1:])
    )
    assert record.data_wait_ns > 0
    assert record.elapsed_ns > record.data_wait_ns
    assert record.elapsed_seconds == record.elapsed_ns / 1_000_000_000
    assert record.data_wait_seconds == record.data_wait_ns / 1_000_000_000

    previous_hash = record.chain_genesis_hash
    for sample in record.samples:
        assert sample.previous_sample_hash == previous_hash
        assert sample.sample_hash == canonical_sha256(
            sample.model_dump(mode="python", by_alias=True, exclude={"sample_hash"})
        )
        previous_hash = sample.sample_hash
    assert record.final_sample_hash == previous_hash

    persisted = ResourceTelemetryRecord.model_validate_json(record.model_dump_json(by_alias=True))
    assert persisted == record


def test_public_two_point_formal_record_cannot_forge_the_checkpoint_timeline() -> None:
    genesis = canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.core-resource-telemetry/2.0",
            "chainRole": "resource-telemetry-genesis",
            "cellId": CELL_ID,
            "runId": "forged-run",
            "phase": "formal",
            "configHash": CONFIG_HASH,
            "dataHash": DATA_HASH,
            "codeHash": CODE_HASH,
            "environmentHash": ENVIRONMENT_HASH,
        }
    )
    samples: list[dict[str, Any]] = []
    previous = genesis
    for ordinal, step in enumerate((0, 2000)):
        state = _model_state(step)
        fit_state = _fit_state(step, state)
        sample: dict[str, Any] = {
            "ordinal": ordinal,
            "previousSampleHash": previous,
            "optimizerStep": step,
            "monotonicNs": 10 + ordinal,
            "cumulativeDataWaitNs": 0,
            "cudaMaxAllocatedBytes": 0,
            "modelStateHash": _model_hash(state),
            "fitStateHash": fit_state["stateHash"],
        }
        sample["sampleHash"] = canonical_sha256(sample)
        samples.append(sample)
        previous = sample["sampleHash"]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-resource-telemetry/2.0",
        "cellId": CELL_ID,
        "runId": "forged-run",
        "phase": "formal",
        "configHash": CONFIG_HASH,
        "dataHash": DATA_HASH,
        "codeHash": CODE_HASH,
        "environmentHash": ENVIRONMENT_HASH,
        "chainGenesisHash": genesis,
        "samples": samples,
        "finalOptimizerStep": 2000,
        "elapsedNs": 1,
        "dataWaitNs": 0,
        "peakCudaBytes": 0,
        "finalModelStateHash": samples[-1]["modelStateHash"],
        "finalFitStateHash": samples[-1]["fitStateHash"],
        "latestCheckpointSemanticHash": LATEST_CHECKPOINT_HASH,
        "bestCheckpointSemanticHash": BEST_CHECKPOINT_HASH,
        "finalSampleHash": previous,
    }
    payload["telemetryHash"] = canonical_sha256(payload)

    with pytest.raises(ValidationError, match="formal telemetry timeline"):
        ResourceTelemetryRecord.model_validate(payload)


def test_recorder_rejects_fit_state_bound_to_a_different_model_state() -> None:
    recorder = _recorder(phase="dev")
    state = _model_state(0)
    mismatched = _model_state(1)

    with pytest.raises(ValueError, match="checkpoint model state"):
        recorder.record_start(model_state=state, fit_state=_fit_state(0, mismatched))


def test_formal_recorder_rejects_a_checkpoint_timeline_gap() -> None:
    recorder = _recorder()
    state = _model_state(0)
    recorder.record_start(model_state=state, fit_state=_fit_state(0, state))
    gap_state = _model_state(500)

    with pytest.raises(ValueError, match="next 250-step checkpoint"):
        recorder.record_checkpoint(
            optimizer_step=500,
            model_state=gap_state,
            fit_state=_fit_state(500, gap_state),
        )


def test_data_wait_context_rejects_caller_supplied_duration_injection() -> None:
    recorder = _recorder(phase="dev")

    with pytest.raises(TypeError):
        recorder.measure_data_wait(seconds=21_600)  # type: ignore[call-arg]


def test_finish_rejects_final_model_state_that_differs_from_last_sample() -> None:
    recorder = _recorder(phase="dev")
    state = _model_state(0)
    recorder.record_start(model_state=state, fit_state=_fit_state(0, state))
    sampled_state = _model_state(1)
    recorder.record_checkpoint(
        optimizer_step=1,
        model_state=sampled_state,
        fit_state=_fit_state(1, sampled_state),
    )
    substituted_state = _model_state(2)

    with pytest.raises(ValueError, match="final model/fit state"):
        recorder.finish(
            final_optimizer_step=1,
            model_state=substituted_state,
            fit_state=_fit_state(1, substituted_state),
            latest_checkpoint_semantic_hash=LATEST_CHECKPOINT_HASH,
            best_checkpoint_semantic_hash=BEST_CHECKPOINT_HASH,
        )


def test_finish_cannot_hide_measured_wait_after_the_last_step_sample() -> None:
    recorder = _recorder(phase="dev")
    state = _model_state(0)
    recorder.record_start(model_state=state, fit_state=_fit_state(0, state))
    final_state = _model_state(1)
    final_fit_state = _fit_state(1, final_state)
    recorder.record_checkpoint(
        optimizer_step=1,
        model_state=final_state,
        fit_state=final_fit_state,
    )
    with recorder.measure_data_wait():
        time.sleep(0.001)

    with pytest.raises(ValueError, match="activity after the last resource sample"):
        recorder.finish(
            final_optimizer_step=1,
            model_state=final_state,
            fit_state=final_fit_state,
            latest_checkpoint_semantic_hash=LATEST_CHECKPOINT_HASH,
            best_checkpoint_semantic_hash=BEST_CHECKPOINT_HASH,
        )


def test_raw_record_and_verified_subclass_cannot_cross_the_sealed_boundary() -> None:
    verified = _record_formal_through(_recorder(), final_step=2000)

    with pytest.raises(TypeError, match="exact VerifiedResourceTelemetry"):
        verify_resource_telemetry(verified.record)  # type: ignore[arg-type]

    class BypassVerifiedResourceTelemetry(VerifiedResourceTelemetry):
        pass

    bypass = object.__new__(BypassVerifiedResourceTelemetry)
    for name in ("record", "_sealed_telemetry_hash", "_factory_seal"):
        object.__setattr__(bypass, name, getattr(verified, name))
    with pytest.raises(TypeError, match="exact VerifiedResourceTelemetry"):
        verify_resource_telemetry(bypass)


def test_sample_and_record_hash_mutations_fail_closed() -> None:
    record = _record_formal_through(_recorder(), final_step=2000).record
    sample_payload = record.samples[1].model_dump(mode="python", by_alias=True)
    sample_payload["modelStateHash"] = "8" * 64
    with pytest.raises(ValidationError, match="sampleHash"):
        ResourceTelemetrySample.model_validate(sample_payload)

    record_payload = record.model_dump(mode="python", by_alias=True)
    record_payload["latestCheckpointSemanticHash"] = "9" * 64
    with pytest.raises(ValidationError, match="telemetryHash"):
        ResourceTelemetryRecord.model_validate(record_payload)
