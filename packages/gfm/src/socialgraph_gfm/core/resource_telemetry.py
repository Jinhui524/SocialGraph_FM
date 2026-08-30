"""Process-observed, hash-chained resource telemetry for core experiments.

The Pydantic records in this module are safe persistence envelopes, not proof that a
caller measured anything.  Formal experiment code must accept only the exact
``VerifiedResourceTelemetry`` value returned by ``ResourceTelemetryRecorder.finish``.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest


_SCHEMA = "socialgraph-fm.core-resource-telemetry/2.0"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$"
_FORMAL_VALIDATION_INTERVAL = 250
_FORMAL_MAX_OPTIMIZER_STEP = 10_000
_FACTORY_SEAL = object()


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _require_hash(value: object, *, name: str) -> str:
    if type(value) is not str or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _model_state_hash(state: Mapping[str, Tensor]) -> str:
    if not state or not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise ValueError("model state must be a non-empty string-to-tensor mapping")
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


def _state_hashes(
    *,
    optimizer_step: int,
    model_state: Mapping[str, Tensor],
    fit_state: Mapping[str, Any],
) -> tuple[str, str]:
    if type(optimizer_step) is not int or optimizer_step < 0:
        raise ValueError("optimizer step must be a non-negative integer")
    model_hash = _model_state_hash(model_state)
    if not isinstance(fit_state, Mapping) or not all(isinstance(name, str) for name in fit_state):
        raise ValueError("fit state must be a string-keyed mapping")
    fit_payload = dict(fit_state)
    fit_hash = fit_payload.get("stateHash")
    if type(fit_hash) is not str or _HASH_PATTERN.fullmatch(fit_hash) is None:
        raise ValueError("fit state must contain a lowercase stateHash")
    expected_fit_hash = canonical_sha256(
        {name: value for name, value in fit_payload.items() if name != "stateHash"}
    )
    if fit_hash != expected_fit_hash:
        raise ValueError("fit state stateHash does not match its content")
    if fit_payload.get("checkpointModelStateHash") != model_hash:
        raise ValueError("fit state checkpoint model state does not match the sampled model")
    declared_step = fit_payload.get("optimizerStep")
    if declared_step is not None and declared_step != optimizer_step:
        raise ValueError("fit state optimizer step does not match the sampled step")
    return model_hash, fit_hash


def _genesis_payload(
    *,
    cell_id: str,
    run_id: str,
    phase: Literal["smoke", "dev", "formal"],
    config_hash: str,
    data_hash: str,
    code_hash: str,
    environment_hash: str,
) -> dict[str, str]:
    return {
        "schemaVersion": _SCHEMA,
        "chainRole": "resource-telemetry-genesis",
        "cellId": cell_id,
        "runId": run_id,
        "phase": phase,
        "configHash": config_hash,
        "dataHash": data_hash,
        "codeHash": code_hash,
        "environmentHash": environment_hash,
    }


class ResourceTelemetrySample(_StrictModel):
    """One process-observed checkpoint or validation sample in a chained timeline."""

    ordinal: int = Field(ge=0)
    previous_sample_hash: str = Field(alias="previousSampleHash", pattern=r"^[0-9a-f]{64}$")
    optimizer_step: int = Field(alias="optimizerStep", ge=0, le=_FORMAL_MAX_OPTIMIZER_STEP)
    monotonic_ns: int = Field(alias="monotonicNs", ge=0)
    cumulative_data_wait_ns: int = Field(alias="cumulativeDataWaitNs", ge=0)
    cuda_max_allocated_bytes: int = Field(alias="cudaMaxAllocatedBytes", ge=0)
    model_state_hash: str = Field(alias="modelStateHash", pattern=r"^[0-9a-f]{64}$")
    fit_state_hash: str = Field(alias="fitStateHash", pattern=r"^[0-9a-f]{64}$")
    sample_hash: str = Field(alias="sampleHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_sample_hash(self) -> ResourceTelemetrySample:
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"sample_hash"})
        )
        if self.sample_hash != expected:
            raise ValueError("sampleHash does not match the resource observation")
        return self


class ResourceTelemetryRecord(_StrictModel):
    """Persistable telemetry bytes; trust requires a separate runtime factory seal."""

    schema_version: Literal["socialgraph-fm.core-resource-telemetry/2.0"] = Field(
        alias="schemaVersion"
    )
    cell_id: str = Field(alias="cellId", pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(alias="runId", pattern=_RUN_ID_PATTERN)
    phase: Literal["smoke", "dev", "formal"]
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    data_hash: str = Field(alias="dataHash", pattern=r"^[0-9a-f]{64}$")
    code_hash: str = Field(alias="codeHash", pattern=r"^[0-9a-f]{64}$")
    environment_hash: str = Field(alias="environmentHash", pattern=r"^[0-9a-f]{64}$")
    chain_genesis_hash: str = Field(alias="chainGenesisHash", pattern=r"^[0-9a-f]{64}$")
    samples: tuple[ResourceTelemetrySample, ...] = Field(strict=False, min_length=2)
    final_optimizer_step: int = Field(
        alias="finalOptimizerStep", ge=1, le=_FORMAL_MAX_OPTIMIZER_STEP
    )
    elapsed_ns: int = Field(alias="elapsedNs", gt=0)
    data_wait_ns: int = Field(alias="dataWaitNs", ge=0)
    peak_cuda_bytes: int = Field(alias="peakCudaBytes", ge=0)
    final_model_state_hash: str = Field(alias="finalModelStateHash", pattern=r"^[0-9a-f]{64}$")
    final_fit_state_hash: str = Field(alias="finalFitStateHash", pattern=r"^[0-9a-f]{64}$")
    latest_checkpoint_semantic_hash: str = Field(
        alias="latestCheckpointSemanticHash", pattern=r"^[0-9a-f]{64}$"
    )
    best_checkpoint_semantic_hash: str = Field(
        alias="bestCheckpointSemanticHash", pattern=r"^[0-9a-f]{64}$"
    )
    final_sample_hash: str = Field(alias="finalSampleHash", pattern=r"^[0-9a-f]{64}$")
    telemetry_hash: str = Field(alias="telemetryHash", pattern=r"^[0-9a-f]{64}$")

    @property
    def optimizer_steps(self) -> int:
        return self.final_optimizer_step

    @property
    def elapsed_seconds(self) -> float:
        return self.elapsed_ns / 1_000_000_000

    @property
    def data_wait_seconds(self) -> float:
        return self.data_wait_ns / 1_000_000_000

    @model_validator(mode="after")
    def validate_record(self) -> ResourceTelemetryRecord:
        expected_genesis = canonical_sha256(
            _genesis_payload(
                cell_id=self.cell_id,
                run_id=self.run_id,
                phase=self.phase,
                config_hash=self.config_hash,
                data_hash=self.data_hash,
                code_hash=self.code_hash,
                environment_hash=self.environment_hash,
            )
        )
        if self.chain_genesis_hash != expected_genesis:
            raise ValueError("chainGenesisHash does not match the telemetry run binding")

        expected_previous = self.chain_genesis_hash
        for ordinal, sample in enumerate(self.samples):
            if sample.ordinal != ordinal or sample.previous_sample_hash != expected_previous:
                raise ValueError("resource telemetry sample hash chain is not contiguous")
            expected_previous = sample.sample_hash
        if self.final_sample_hash != expected_previous:
            raise ValueError("finalSampleHash does not match the resource telemetry chain")

        steps = tuple(sample.optimizer_step for sample in self.samples)
        times = tuple(sample.monotonic_ns for sample in self.samples)
        waits = tuple(sample.cumulative_data_wait_ns for sample in self.samples)
        cuda_peaks = tuple(sample.cuda_max_allocated_bytes for sample in self.samples)
        if steps[0] != 0 or waits[0] != 0:
            raise ValueError("resource telemetry must start at optimizer step zero with no wait")
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError("resource telemetry optimizer steps must strictly increase")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("resource telemetry monotonicNs must strictly increase")
        if any(right < left for left, right in zip(waits, waits[1:])):
            raise ValueError("resource telemetry cumulative data wait must not decrease")
        if any(right < left for left, right in zip(cuda_peaks, cuda_peaks[1:])):
            raise ValueError("resource telemetry CUDA max allocation must not decrease")

        if self.phase == "formal":
            for index, (left, right) in enumerate(zip(steps, steps[1:])):
                difference = right - left
                is_final_interval = index == len(steps) - 2
                if (not is_final_interval and difference != _FORMAL_VALIDATION_INTERVAL) or (
                    is_final_interval and difference > _FORMAL_VALIDATION_INTERVAL
                ):
                    raise ValueError(
                        "formal telemetry timeline must include every 250-step checkpoint"
                    )

        derived_elapsed = times[-1] - times[0]
        derived_wait = waits[-1] - waits[0]
        if self.final_optimizer_step != steps[-1]:
            raise ValueError("finalOptimizerStep does not match the final resource sample")
        if self.elapsed_ns != derived_elapsed:
            raise ValueError("elapsedNs must be derived from monotonic resource samples")
        if self.data_wait_ns != derived_wait or self.data_wait_ns > self.elapsed_ns:
            raise ValueError("dataWaitNs must be measured within the telemetry interval")
        if self.peak_cuda_bytes != max(cuda_peaks):
            raise ValueError("peakCudaBytes must be derived from CUDA allocator samples")
        final_sample = self.samples[-1]
        if (
            self.final_model_state_hash != final_sample.model_state_hash
            or self.final_fit_state_hash != final_sample.fit_state_hash
        ):
            raise ValueError("final model/fit state does not match the final resource sample")

        expected_hash = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"telemetry_hash"})
        )
        if self.telemetry_hash != expected_hash:
            raise ValueError("telemetryHash does not match the resource evidence")
        return self


@dataclass(frozen=True, init=False)
class VerifiedResourceTelemetry:
    """Factory-sealed evidence that process-local measurement produced the raw record."""

    record: ResourceTelemetryRecord
    _sealed_telemetry_hash: str
    _factory_seal: object

    @property
    def cell_id(self) -> str:
        return self.record.cell_id

    @property
    def phase(self) -> Literal["smoke", "dev", "formal"]:
        return self.record.phase

    @property
    def optimizer_steps(self) -> int:
        return self.record.optimizer_steps

    @property
    def elapsed_seconds(self) -> float:
        return self.record.elapsed_seconds

    @property
    def data_wait_seconds(self) -> float:
        return self.record.data_wait_seconds

    @property
    def peak_cuda_bytes(self) -> int:
        return self.record.peak_cuda_bytes

    @property
    def telemetry_hash(self) -> str:
        return self.record.telemetry_hash


def _new_verified(record: ResourceTelemetryRecord) -> VerifiedResourceTelemetry:
    verified = object.__new__(VerifiedResourceTelemetry)
    object.__setattr__(verified, "record", record)
    object.__setattr__(verified, "_sealed_telemetry_hash", record.telemetry_hash)
    object.__setattr__(verified, "_factory_seal", _FACTORY_SEAL)
    return verified


def verify_resource_telemetry(value: VerifiedResourceTelemetry) -> ResourceTelemetryRecord:
    """Return the persisted record only after exact-type and runtime-seal verification."""

    if type(value) is not VerifiedResourceTelemetry:
        raise TypeError("formal runs require exact VerifiedResourceTelemetry evidence")
    if (
        value._factory_seal is not _FACTORY_SEAL
        or type(value.record) is not ResourceTelemetryRecord
        or value.record.telemetry_hash != value._sealed_telemetry_hash
    ):
        raise ValueError("verified resource telemetry runtime seal changed")
    reparsed = ResourceTelemetryRecord.model_validate(
        value.record.model_dump(mode="python", by_alias=True)
    )
    if reparsed != value.record:
        raise ValueError("verified resource telemetry record changed")
    return value.record


class ResourceTelemetryRecorder:
    """Measure one run using only the process monotonic clock and CUDA allocator."""

    def __init__(
        self,
        *,
        cell_id: str,
        run_id: str,
        phase: Literal["smoke", "dev", "formal"],
        config_hash: str,
        data_hash: str,
        code_hash: str,
        environment_hash: str,
        cuda_device: torch.device | str | int | None = None,
    ) -> None:
        self._cell_id = _require_hash(cell_id, name="cellId")
        if type(run_id) is not str or re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
            raise ValueError("runId has invalid syntax")
        self._run_id = run_id
        if phase not in {"smoke", "dev", "formal"}:
            raise ValueError("phase must be smoke, dev, or formal")
        self._phase: Literal["smoke", "dev", "formal"] = phase
        self._config_hash = _require_hash(config_hash, name="configHash")
        self._data_hash = _require_hash(data_hash, name="dataHash")
        self._code_hash = _require_hash(code_hash, name="codeHash")
        self._environment_hash = _require_hash(environment_hash, name="environmentHash")
        self._chain_genesis_hash = canonical_sha256(
            _genesis_payload(
                cell_id=self._cell_id,
                run_id=self._run_id,
                phase=self._phase,
                config_hash=self._config_hash,
                data_hash=self._data_hash,
                code_hash=self._code_hash,
                environment_hash=self._environment_hash,
            )
        )
        self._samples: list[ResourceTelemetrySample] = []
        self._cumulative_data_wait_ns = 0
        self._wait_active = False
        self._finished = False
        self._cuda_device: torch.device | None = None
        if torch.cuda.is_available():
            if cuda_device is None:
                selected = torch.device("cuda", torch.cuda.current_device())
            elif type(cuda_device) is int:
                selected = torch.device("cuda", cuda_device)
            else:
                selected = torch.device(cuda_device)
            if selected.type != "cuda":
                raise ValueError("cuda_device must identify a CUDA device")
            self._cuda_device = selected
            torch.cuda.reset_peak_memory_stats(selected)

    def _ensure_active(self) -> None:
        if self._finished:
            raise RuntimeError("resource telemetry recorder is already finished")
        if self._wait_active:
            raise RuntimeError("cannot sample resource telemetry during a data-wait interval")

    def _monotonic_after(self, previous: int | None = None) -> int:
        observed = time.monotonic_ns()
        for _ in range(1_000):
            if previous is None or observed > previous:
                return observed
            # Windows may expose monotonic_ns with millisecond resolution. Yielding here
            # preserves a real clock observation instead of fabricating ``previous + 1``.
            time.sleep(0.001)
            observed = time.monotonic_ns()
        raise RuntimeError("process monotonic clock did not advance")

    def _cuda_peak(self) -> int:
        if self._cuda_device is None:
            return 0
        return int(torch.cuda.max_memory_allocated(self._cuda_device))

    def _append_sample(
        self,
        *,
        optimizer_step: int,
        model_state: Mapping[str, Tensor],
        fit_state: Mapping[str, Any],
    ) -> ResourceTelemetrySample:
        self._ensure_active()
        if type(optimizer_step) is not int or not 0 <= optimizer_step <= 10_000:
            raise ValueError("optimizer step must be an integer between zero and 10000")
        model_hash, fit_hash = _state_hashes(
            optimizer_step=optimizer_step,
            model_state=model_state,
            fit_state=fit_state,
        )
        previous_time = self._samples[-1].monotonic_ns if self._samples else None
        payload: dict[str, Any] = {
            "ordinal": len(self._samples),
            "previousSampleHash": (
                self._samples[-1].sample_hash if self._samples else self._chain_genesis_hash
            ),
            "optimizerStep": optimizer_step,
            "monotonicNs": self._monotonic_after(previous_time),
            "cumulativeDataWaitNs": self._cumulative_data_wait_ns,
            "cudaMaxAllocatedBytes": self._cuda_peak(),
            "modelStateHash": model_hash,
            "fitStateHash": fit_hash,
        }
        payload["sampleHash"] = canonical_sha256(payload)
        sample = ResourceTelemetrySample.model_validate(payload)
        self._samples.append(sample)
        return sample

    def record_start(
        self,
        *,
        model_state: Mapping[str, Tensor],
        fit_state: Mapping[str, Any],
    ) -> None:
        """Capture the real step-zero state before any optimizer update."""

        if self._samples:
            raise RuntimeError("resource telemetry start was already recorded")
        self._append_sample(optimizer_step=0, model_state=model_state, fit_state=fit_state)

    def record_checkpoint(
        self,
        *,
        optimizer_step: int,
        model_state: Mapping[str, Tensor],
        fit_state: Mapping[str, Any],
    ) -> None:
        """Capture one real checkpoint/validation point from the running trainer."""

        if not self._samples:
            raise RuntimeError("record_start must precede checkpoint telemetry")
        previous_step = self._samples[-1].optimizer_step
        if type(optimizer_step) is not int or optimizer_step <= previous_step:
            raise ValueError("checkpoint optimizer steps must strictly increase")
        if (
            self._phase == "formal"
            and optimizer_step != previous_step + _FORMAL_VALIDATION_INTERVAL
        ):
            raise ValueError("formal telemetry requires the next 250-step checkpoint")
        self._append_sample(
            optimizer_step=optimizer_step,
            model_state=model_state,
            fit_state=fit_state,
        )

    @contextmanager
    def measure_data_wait(self) -> Iterator[None]:
        """Measure a data-fetch wait; callers cannot supply or overwrite its duration."""

        self._ensure_active()
        if not self._samples:
            raise RuntimeError("record_start must precede data-wait measurement")
        self._wait_active = True
        started = time.monotonic_ns()
        try:
            yield
        finally:
            ended = self._monotonic_after(started)
            self._cumulative_data_wait_ns += ended - started
            self._wait_active = False

    def finish(
        self,
        *,
        final_optimizer_step: int,
        model_state: Mapping[str, Tensor],
        fit_state: Mapping[str, Any],
        latest_checkpoint_semantic_hash: str,
        best_checkpoint_semantic_hash: str,
    ) -> VerifiedResourceTelemetry:
        """Seal the final live state and exact latest/best checkpoint identities."""

        self._ensure_active()
        if not self._samples:
            raise RuntimeError("record_start must precede resource telemetry finish")
        if type(final_optimizer_step) is not int or not 1 <= final_optimizer_step <= 10_000:
            raise ValueError("final optimizer step must be an integer between one and 10000")
        latest_hash = _require_hash(
            latest_checkpoint_semantic_hash, name="latestCheckpointSemanticHash"
        )
        best_hash = _require_hash(best_checkpoint_semantic_hash, name="bestCheckpointSemanticHash")
        model_hash, fit_hash = _state_hashes(
            optimizer_step=final_optimizer_step,
            model_state=model_state,
            fit_state=fit_state,
        )
        previous_step = self._samples[-1].optimizer_step
        if final_optimizer_step < previous_step:
            raise ValueError("final optimizer step precedes the last resource sample")
        if final_optimizer_step == previous_step:
            final_sample = self._samples[-1]
            if (
                final_sample.model_state_hash != model_hash
                or final_sample.fit_state_hash != fit_hash
            ):
                raise ValueError("final model/fit state differs from the last resource sample")
            if (
                final_sample.cumulative_data_wait_ns != self._cumulative_data_wait_ns
                or final_sample.cuda_max_allocated_bytes != self._cuda_peak()
            ):
                raise ValueError("resource activity after the last resource sample is unbound")
        else:
            if (
                self._phase == "formal"
                and final_optimizer_step > previous_step + _FORMAL_VALIDATION_INTERVAL
            ):
                raise ValueError("formal telemetry finish skips a required checkpoint")
            final_sample = self._append_sample(
                optimizer_step=final_optimizer_step,
                model_state=model_state,
                fit_state=fit_state,
            )
        if len(self._samples) < 2:
            raise ValueError("resource telemetry requires distinct start and final samples")

        first_sample = self._samples[0]
        payload: dict[str, Any] = {
            "schemaVersion": _SCHEMA,
            "cellId": self._cell_id,
            "runId": self._run_id,
            "phase": self._phase,
            "configHash": self._config_hash,
            "dataHash": self._data_hash,
            "codeHash": self._code_hash,
            "environmentHash": self._environment_hash,
            "chainGenesisHash": self._chain_genesis_hash,
            "samples": [
                sample.model_dump(mode="python", by_alias=True) for sample in self._samples
            ],
            "finalOptimizerStep": final_optimizer_step,
            "elapsedNs": final_sample.monotonic_ns - first_sample.monotonic_ns,
            "dataWaitNs": (
                final_sample.cumulative_data_wait_ns - first_sample.cumulative_data_wait_ns
            ),
            "peakCudaBytes": max(sample.cuda_max_allocated_bytes for sample in self._samples),
            "finalModelStateHash": model_hash,
            "finalFitStateHash": fit_hash,
            "latestCheckpointSemanticHash": latest_hash,
            "bestCheckpointSemanticHash": best_hash,
            "finalSampleHash": final_sample.sample_hash,
        }
        payload["telemetryHash"] = canonical_sha256(payload)
        record = ResourceTelemetryRecord.model_validate(payload)
        self._finished = True
        return _new_verified(record)


__all__ = [
    "ResourceTelemetryRecord",
    "ResourceTelemetryRecorder",
    "ResourceTelemetrySample",
    "VerifiedResourceTelemetry",
    "verify_resource_telemetry",
]
