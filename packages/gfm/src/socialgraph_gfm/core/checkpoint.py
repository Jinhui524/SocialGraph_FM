"""Atomic, hash-bound core checkpoint publication and loading."""

from __future__ import annotations

import os
import re
import uuid
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any, Literal, Mapping

import torch


CHECKPOINT_SCHEMA = "socialgraph-fm.core-checkpoint/1.0"
_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CheckpointBindings:
    config_hash: str
    data_hash: str
    code_hash: str
    environment_hash: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not _HASH.fullmatch(value):
                raise ValueError(f"{name.replace('_', ' ')} must be a lowercase SHA-256")


def publish_checkpoint(
    path: Path,
    *,
    trainer_state: Mapping[str, Any],
    bindings: CheckpointBindings,
    status: Literal["training", "validated", "accepted", "timeout-non-promotable"],
    promotable: bool,
    before_commit: Callable[[], None] | None = None,
    replace_existing: bool = True,
) -> None:
    legal = {
        ("training", False),
        ("validated", False),
        ("validated", True),
        ("accepted", True),
        ("timeout-non-promotable", False),
    }
    if (status, bool(promotable)) not in legal:
        raise ValueError("checkpoint status/promotable combination is invalid")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.publisher.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError("checkpoint already has an active publisher") from error
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        os.close(lock_fd)
        payload = {
            "schemaVersion": CHECKPOINT_SCHEMA,
            "bindings": asdict(bindings),
            "status": status,
            "promotable": bool(promotable),
            "trainer": dict(trainer_state),
        }
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if before_commit is not None:
            before_commit()
        if replace_existing:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def load_checkpoint(path: Path | bytes, *, expected_bindings: CheckpointBindings) -> dict[str, Any]:
    source = BytesIO(path) if isinstance(path, bytes) else Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schemaVersion") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported core checkpoint schema")
    observed = payload.get("bindings")
    if not isinstance(observed, dict):
        raise ValueError("checkpoint bindings are missing")
    for field, expected in asdict(expected_bindings).items():
        if observed.get(field) != expected:
            raise ValueError(f"{field.replace('_', ' ')} mismatch")
    trainer = payload.get("trainer")
    if not isinstance(trainer, dict):
        raise ValueError("checkpoint trainer state is missing")
    legal = {
        ("training", False),
        ("validated", False),
        ("validated", True),
        ("accepted", True),
        ("timeout-non-promotable", False),
    }
    if (payload.get("status"), payload.get("promotable")) not in legal:
        raise ValueError("checkpoint status/promotable combination is invalid")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CheckpointBindings",
    "load_checkpoint",
    "publish_checkpoint",
]
