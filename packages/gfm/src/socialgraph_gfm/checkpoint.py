"""Atomic, integrity-checked Torch checkpoints."""

from __future__ import annotations

import os
import random
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from .canonical import canonical_json, canonical_sha256, file_sha256
from .contracts import BaselineCheckpointManifest, SmokeCheckpointManifest
from .errors import CheckpointIntegrityError, MissingRuntimeDependency
from .tensor_digest import canonical_tensor_digest


def _tensor_state_hash(value: Any) -> str:
    torch = _torch()

    def describe(item: Any) -> Any:
        if torch.is_tensor(item):
            return canonical_tensor_digest(item)
        if isinstance(item, dict):
            return {str(key): describe(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [describe(nested) for nested in item]
        if isinstance(item, (str, int, float, bool)) or item is None:
            return item
        return {"type": type(item).__name__, "repr": repr(item)}

    return canonical_sha256(describe(value))


def _torch():
    try:
        import torch
    except ImportError as error:
        raise MissingRuntimeDependency("Torch is required for checkpoint operations") from error
    return torch


def _logical_checkpoint_state(payload: dict[str, Any]) -> dict[str, Any]:
    """Select deterministic state while excluding execution identity.

    ``run_id`` remains inside the serialized artifact and is checked against the physical
    manifest on load, but cannot influence the logical state hash used for reproducibility.
    """

    return {
        "format": payload["format"],
        "step": payload["step"],
        "model_state": payload["model_state"],
        "optimizer_state": payload["optimizer_state"],
        "config": payload["config"],
    }


def save_checkpoint(
    directory: str | Path,
    *,
    checkpoint_id: str,
    run_id: str,
    step: int,
    model_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    config: dict[str, Any],
) -> SmokeCheckpointManifest:
    torch = _torch()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / f"{checkpoint_id}.pt"
    payload = {
        "format": "gfm.smoke-checkpoint/1.0",
        "run_id": run_id,
        "step": step,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "config": config,
    }
    state_hash = _tensor_state_hash(_logical_checkpoint_state(payload))
    handle, temporary_name = tempfile.mkstemp(prefix=f".{checkpoint_id}.", suffix=".tmp", dir=destination)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, artifact)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest = SmokeCheckpointManifest(
        checkpointId=checkpoint_id,
        runId=run_id,
        step=step,
        smokeOnly=True,
        stateHash=state_hash,
        configHash=canonical_sha256(config),
        artifactSha256=file_sha256(artifact),
        artifactPath=str(artifact.resolve()),
        createdAt=datetime.now(UTC),
    )
    manifest_path = destination / f"{checkpoint_id}.manifest.json"
    manifest_text = canonical_json(manifest)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{checkpoint_id}.", suffix=".json.tmp", dir=destination)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(manifest_text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return manifest


def read_manifest(path: str | Path) -> SmokeCheckpointManifest:
    return SmokeCheckpointManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_checkpoint(manifest: SmokeCheckpointManifest, map_location: str = "cpu") -> dict[str, Any]:
    torch = _torch()
    artifact = Path(manifest.artifact_path)
    if not artifact.is_file():
        raise CheckpointIntegrityError(f"Checkpoint artifact does not exist: {artifact}")
    actual_hash = file_sha256(artifact)
    if actual_hash != manifest.artifact_sha256:
        raise CheckpointIntegrityError(
            f"Checkpoint SHA-256 mismatch: expected {manifest.artifact_sha256}, got {actual_hash}"
        )
    try:
        payload = torch.load(artifact, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointIntegrityError(f"Checkpoint cannot be loaded safely: {error}") from error
    if _tensor_state_hash(_logical_checkpoint_state(payload)) != manifest.state_hash:
        raise CheckpointIntegrityError("Checkpoint tensor state hash does not match its manifest")
    if payload.get("run_id") != manifest.run_id or payload.get("step") != manifest.step:
        raise CheckpointIntegrityError("Checkpoint identity does not match its manifest")
    return payload


def capture_rng_state() -> dict[str, Any]:
    """Capture restorable RNG state using only ``weights_only=True`` safe values."""

    torch = _torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    try:
        import numpy as np

        numpy_state: Any = np.random.get_state()
        algorithm, keys, position, has_gauss, cached_gaussian = numpy_state
        state["numpy"] = {
            "algorithm": algorithm,
            "keys": [int(value) for value in keys.tolist()],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        }
    except ImportError:
        state["numpy"] = None
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch = _torch()
    random.setstate(_as_tuple(state["python"]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        import numpy as np

        np.random.set_state(
            (
                numpy_state["algorithm"],
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )


def _as_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_as_tuple(item) for item in value)
    return value


def _baseline_logical_state(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload[key]
        for key in (
            "format",
            "epoch",
            "track",
            "model",
            "model_state",
            "predictor_state",
            "optimizer_state",
            "scheduler_state",
            "rng_state",
            "sampler_state",
            "selection_rng_state",
            "best_validation_hits50",
            "best_epoch",
            "best_model_state",
            "best_predictor_state",
            "selected_batch_size",
            "evaluations_without_improvement",
            "history",
            "terminal",
            "config",
            "corpus_hash",
        )
    }


def save_baseline_checkpoint(
    directory: str | Path,
    *,
    checkpoint_id: str,
    run_id: str,
    epoch: int,
    track: str,
    model: str,
    model_state: dict[str, Any],
    predictor_state: dict[str, Any],
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any] | None,
    sampler_state: dict[str, Any],
    selection_rng_state: dict[str, Any],
    best_validation_hits50: float,
    best_epoch: int,
    best_model_state: dict[str, Any],
    best_predictor_state: dict[str, Any],
    selected_batch_size: int,
    evaluations_without_improvement: int,
    history: list[dict[str, float]],
    terminal: bool,
    config: dict[str, Any],
    corpus_hash: str,
    verification_digest: str | None = None,
    rng_state: dict[str, Any] | None = None,
) -> BaselineCheckpointManifest:
    """Atomically save a non-promotable baseline checkpoint and integrity manifest."""

    if track not in ("ogb_official", "strict_edge_time"):
        raise ValueError(f"unsupported baseline track: {track}")
    if model not in ("mlp", "graphsage"):
        raise ValueError(f"unsupported checkpointed baseline model: {model}")
    torch = _torch()
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / f"{checkpoint_id}.pt"
    payload = {
        "format": "gfm.baseline-checkpoint/1.0",
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "epoch": epoch,
        "track": track,
        "model": model,
        "model_state": model_state,
        "predictor_state": predictor_state,
        "optimizer_state": optimizer_state,
        "scheduler_state": scheduler_state,
        "rng_state": rng_state or capture_rng_state(),
        "sampler_state": sampler_state,
        "selection_rng_state": selection_rng_state,
        "best_validation_hits50": best_validation_hits50,
        "best_epoch": best_epoch,
        "best_model_state": best_model_state,
        "best_predictor_state": best_predictor_state,
        "selected_batch_size": selected_batch_size,
        "evaluations_without_improvement": evaluations_without_improvement,
        "history": history,
        "terminal": terminal,
        "config": config,
        "corpus_hash": corpus_hash,
    }
    state_hash = _tensor_state_hash(_baseline_logical_state(payload))
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint_id}.", suffix=".tmp", dir=destination
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, artifact)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    manifest = BaselineCheckpointManifest(
        checkpointId=checkpoint_id,
        runId=run_id,
        epoch=epoch,
        track=cast(Literal["ogb_official", "strict_edge_time"], track),
        model=cast(Literal["mlp", "graphsage"], model),
        checkpointKind="baseline",
        registrable=False,
        stateHash=state_hash,
        configHash=canonical_sha256(config),
        corpusHash=corpus_hash,
        artifactSha256=file_sha256(artifact),
        artifactPath=str(artifact.resolve()),
        verificationDigest=verification_digest,
        createdAt=datetime.now(UTC),
    )
    _atomic_manifest(
        destination / f"{checkpoint_id}.manifest.json", canonical_json(manifest)
    )
    return manifest


def _atomic_manifest(path: Path, value: str) -> None:
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".json.tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_baseline_manifest(path: str | Path) -> BaselineCheckpointManifest:
    return BaselineCheckpointManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def load_baseline_checkpoint(
    manifest: BaselineCheckpointManifest, map_location: str = "cpu"
) -> dict[str, Any]:
    torch = _torch()
    artifact = Path(manifest.artifact_path)
    if not artifact.is_file():
        raise CheckpointIntegrityError(f"Checkpoint artifact does not exist: {artifact}")
    actual_hash = file_sha256(artifact)
    if actual_hash != manifest.artifact_sha256:
        raise CheckpointIntegrityError(
            f"Checkpoint SHA-256 mismatch: expected {manifest.artifact_sha256}, got {actual_hash}"
        )
    try:
        payload = torch.load(artifact, map_location=map_location, weights_only=True)
    except Exception as error:
        raise CheckpointIntegrityError(f"Checkpoint cannot be loaded safely: {error}") from error
    if payload.get("format") != "gfm.baseline-checkpoint/1.0":
        raise CheckpointIntegrityError("Checkpoint format is not a baseline checkpoint")
    if _tensor_state_hash(_baseline_logical_state(payload)) != manifest.state_hash:
        raise CheckpointIntegrityError("Checkpoint tensor state hash does not match its manifest")
    identities = {
        "checkpoint_id": manifest.checkpoint_id,
        "run_id": manifest.run_id,
        "epoch": manifest.epoch,
        "track": manifest.track,
        "model": manifest.model,
        "corpus_hash": manifest.corpus_hash,
    }
    if any(payload.get(key) != value for key, value in identities.items()):
        raise CheckpointIntegrityError("Checkpoint identity does not match its manifest")
    if canonical_sha256(payload["config"]) != manifest.config_hash:
        raise CheckpointIntegrityError("Checkpoint config does not match its manifest")
    return payload
