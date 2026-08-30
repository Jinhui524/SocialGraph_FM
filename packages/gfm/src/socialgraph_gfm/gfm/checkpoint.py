"""Atomic, weights-only formal GFM checkpoints with end-to-end tamper detection."""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..canonical import canonical_json, canonical_sha256, file_sha256
from ..checkpoint import capture_rng_state
from ..errors import CheckpointIntegrityError, ContractViolation, MissingRuntimeDependency
from ..tensor_digest import canonical_tensor_digest
from .contracts import GfmCheckpointManifest

_FORMAT = "gfm.formal-checkpoint/1.0"
_PAYLOAD_KEYS = frozenset(
    {
        "format",
        "checkpoint_id",
        "run_id",
        "epoch",
        "step",
        "components",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
        "sampler_state",
        "best_state",
        "config",
        "corpus_hashes",
    }
)


def _torch():
    try:
        import torch
    except ImportError as error:
        raise MissingRuntimeDependency("Torch is required for GFM checkpoints") from error
    return torch


def _describe(value: Any) -> Any:
    """Return a canonical, type-preserving description of weights-only-safe state."""

    torch = _torch()
    if torch.is_tensor(value):
        return {"kind": "tensor", "value": canonical_tensor_digest(value)}
    if isinstance(value, Mapping):
        entries = [(_describe(key), _describe(nested)) for key, nested in value.items()]
        entries.sort(key=lambda item: canonical_json(item[0]))
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, tuple):
        return {"kind": "tuple", "items": [_describe(item) for item in value]}
    if isinstance(value, list):
        return {"kind": "list", "items": [_describe(item) for item in value]}
    if isinstance(value, bytes):
        return {"kind": "bytes", "hex": value.hex()}
    if value is None or isinstance(value, (str, int, bool)):
        return {"kind": type(value).__name__, "value": value}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractViolation("Checkpoint state contains NaN or Infinity")
        return {"kind": "float", "value": value}
    raise ContractViolation(
        f"Checkpoint state contains unsupported weights-only value {type(value).__name__}"
    )


def _state_hash(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(_describe(payload))


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    torch = _torch()
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".pt.tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_text(path: Path, value: str) -> None:
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


def save_gfm_checkpoint(
    directory: str | Path,
    *,
    checkpoint_id: str,
    run_id: str,
    epoch: int,
    step: int,
    components: Mapping[str, Mapping[str, Any]],
    optimizer_state: Mapping[str, Any],
    scheduler_state: Mapping[str, Any] | None,
    scaler_state: Mapping[str, Any] | None,
    sampler_state: Mapping[str, Any],
    best_state: Mapping[str, Any],
    config: Mapping[str, Any],
    corpus_hashes: tuple[str, ...] | list[str],
    rng_state: Mapping[str, Any] | None = None,
    fresh_process_digest: str | None = None,
) -> GfmCheckpointManifest:
    """Save all formal resume state and its immutable manifest atomically."""

    if not components or any(not name for name in components):
        raise ValueError("components must contain named model state dictionaries")
    if not best_state:
        raise ValueError("best_state must identify the validation-selected state")
    component_names = tuple(sorted(components))
    corpus_ids = tuple(corpus_hashes)
    if len(corpus_ids) != len(set(corpus_ids)) or not corpus_ids:
        raise ValueError("corpus_hashes must be nonempty and unique")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    artifact = destination / f"{checkpoint_id}.pt"
    payload: dict[str, Any] = {
        "format": _FORMAT,
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "epoch": int(epoch),
        "step": int(step),
        "components": {name: dict(components[name]) for name in component_names},
        "optimizer_state": dict(optimizer_state),
        "scheduler_state": None if scheduler_state is None else dict(scheduler_state),
        "scaler_state": None if scaler_state is None else dict(scaler_state),
        "rng_state": dict(rng_state) if rng_state is not None else capture_rng_state(),
        "sampler_state": dict(sampler_state),
        "best_state": dict(best_state),
        "config": dict(config),
        "corpus_hashes": corpus_ids,
    }
    state_hash = _state_hash(payload)
    _atomic_torch_save(artifact, payload)
    manifest = GfmCheckpointManifest.create(
        checkpointId=checkpoint_id,
        runId=run_id,
        epoch=epoch,
        step=step,
        componentNames=component_names,
        stateHash=state_hash,
        configHash=canonical_sha256(payload["config"]),
        corpusHashes=corpus_ids,
        artifactSha256=file_sha256(artifact),
        artifactPath=str(artifact.resolve()),
        registrable=False,
        freshProcessDigest=fresh_process_digest,
    )
    _atomic_text(
        destination / f"{checkpoint_id}.manifest.json", canonical_json(manifest)
    )
    return manifest


def read_gfm_checkpoint_manifest(path: str | Path) -> GfmCheckpointManifest:
    try:
        return GfmCheckpointManifest.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise CheckpointIntegrityError(f"Invalid GFM checkpoint manifest: {error}") from error


def load_gfm_checkpoint(
    manifest: GfmCheckpointManifest, map_location: str = "cpu"
) -> dict[str, Any]:
    """Load through ``weights_only=True`` and re-check every bound identity."""

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
    if not isinstance(payload, dict) or set(payload) != _PAYLOAD_KEYS:
        raise CheckpointIntegrityError("Checkpoint payload has missing or unexpected fields")
    if payload.get("format") != _FORMAT:
        raise CheckpointIntegrityError("Checkpoint format is not a formal GFM checkpoint")
    identities = {
        "checkpoint_id": manifest.checkpoint_id,
        "run_id": manifest.run_id,
        "epoch": manifest.epoch,
        "step": manifest.step,
    }
    if any(payload.get(key) != expected for key, expected in identities.items()):
        raise CheckpointIntegrityError("Checkpoint identity does not match its manifest")
    components = payload.get("components")
    if not isinstance(components, dict) or tuple(sorted(components)) != manifest.component_names:
        raise CheckpointIntegrityError("Checkpoint components do not match its manifest")
    if tuple(payload.get("corpus_hashes", ())) != manifest.corpus_hashes:
        raise CheckpointIntegrityError("Checkpoint corpus hashes do not match its manifest")
    if canonical_sha256(payload.get("config")) != manifest.config_hash:
        raise CheckpointIntegrityError("Checkpoint config does not match its manifest")
    try:
        actual_state_hash = _state_hash(payload)
    except (ContractViolation, ValueError, TypeError) as error:
        raise CheckpointIntegrityError(f"Checkpoint state is not canonical: {error}") from error
    if actual_state_hash != manifest.state_hash:
        raise CheckpointIntegrityError("Checkpoint state hash does not match its manifest")
    return payload


__all__ = [
    "load_gfm_checkpoint",
    "read_gfm_checkpoint_manifest",
    "save_gfm_checkpoint",
]
