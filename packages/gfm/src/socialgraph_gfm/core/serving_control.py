"""Atomic operator serving control binding registry and graph catalog snapshots."""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .artifact_catalog import ArtifactCatalogDocument
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root
from .serving_registry import RegistryDocument, ServingRegistry


_HASH = r"^[0-9a-f]{64}$"
MAX_CONTROL_BYTES = 1024 * 1024
MAX_DOCUMENT_BYTES = 16 * 1024 * 1024
_HIGH_WATER_LOCK = threading.RLock()


def _CONTROL_ACQUIRE_SEAM(_stage: str) -> None:
    return


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True, frozen=True)


class ServingControlReference(_StrictModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    sha256: str = Field(pattern=_HASH)
    semantic_hash: str = Field(alias="semanticHash", pattern=_HASH)
    generation: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        parsed = PurePosixPath(value.replace("\\", "/"))
        if parsed.is_absolute() or ".." in parsed.parts or ":" in value:
            raise ValueError("serving control reference must be a safe relative path")
        return parsed.as_posix()


class ServingControlDocument(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-control/1.0"] = Field(
        alias="schemaVersion"
    )
    generation: int = Field(ge=0)
    registry: ServingControlReference
    catalog: ServingControlReference
    control_hash: str = Field(alias="controlHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        payload = self.model_dump(mode="python", by_alias=True, exclude={"control_hash"})
        if self.control_hash != canonical_sha256(payload):
            raise ValueError("serving control hash mismatch")
        return self


class ServingHighWater(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-control-high-water/1.0"] = Field(
        alias="schemaVersion"
    )
    control_generation: int = Field(alias="controlGeneration", ge=0)
    control_hash: str = Field(alias="controlHash", pattern=_HASH)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=_HASH)
    catalog_generation: int = Field(alias="catalogGeneration", ge=0)
    catalog_hash: str = Field(alias="catalogHash", pattern=_HASH)
    record_hash: str = Field(alias="recordHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.record_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        ):
            raise ValueError("serving high-water record hash mismatch")
        return self


@dataclass(frozen=True)
class CapturedServingControl:
    control_snapshot: bytes
    document: ServingControlDocument
    registry_snapshot: bytes
    registry_document: RegistryDocument
    catalog_snapshot: bytes
    catalog_document: ArtifactCatalogDocument

    @property
    def registry_hash(self) -> str:
        return self.document.registry.semantic_hash

    @property
    def catalog_hash(self) -> str:
        return self.document.catalog.semantic_hash


def _high_water_payload(snapshot: CapturedServingControl) -> dict[str, object]:
    payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.core-serving-control-high-water/1.0",
        "controlGeneration": snapshot.document.generation,
        "controlHash": snapshot.document.control_hash,
        "registryGeneration": snapshot.registry_document.generation,
        "registryHash": snapshot.registry_hash,
        "catalogGeneration": snapshot.catalog_document.generation,
        "catalogHash": snapshot.catalog_hash,
    }
    payload["recordHash"] = canonical_sha256(payload)
    return payload


def _atomic_private_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = (canonical_json(payload) + "\n").encode("utf-8")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


class ServingControlStore:
    """Acquire one coherent control and durably accept its monotonic generations."""

    def __init__(self, path: Path, *, high_water_root: Path) -> None:
        control_path = reject_link_components(path)
        if not control_path.is_file():
            raise ValueError("serving control must be an existing regular file")
        self.control_root = secure_existing_root(control_path.parent)
        self.path = self.control_root / control_path.name
        root = reject_link_components(high_water_root)
        root.mkdir(parents=True, exist_ok=True)
        self.high_water_root = secure_existing_root(root)
        self.high_water_path = self.high_water_root / "serving-control-high-water.json"

    @classmethod
    def load(cls, path: str | Path, *, high_water_root: str | Path) -> ServingControlStore:
        return cls(Path(path), high_water_root=Path(high_water_root))

    def _control_bytes(self) -> bytes:
        return read_confined_snapshot(
            self.control_root, self.path.name, max_bytes=MAX_CONTROL_BYTES
        )

    def _accept_high_water(self, snapshot: CapturedServingControl) -> None:
        with _HIGH_WATER_LOCK:
            current: ServingHighWater | None = None
            if self.high_water_path.exists():
                current = ServingHighWater.model_validate_json(
                    read_confined_snapshot(
                        self.high_water_root,
                        self.high_water_path.name,
                        max_bytes=MAX_CONTROL_BYTES,
                    )
                )
            candidate = ServingHighWater.model_validate(_high_water_payload(snapshot))
            if current is not None:
                pairs = (
                    (
                        candidate.control_generation,
                        candidate.control_hash,
                        current.control_generation,
                        current.control_hash,
                    ),
                    (
                        candidate.registry_generation,
                        candidate.registry_hash,
                        current.registry_generation,
                        current.registry_hash,
                    ),
                    (
                        candidate.catalog_generation,
                        candidate.catalog_hash,
                        current.catalog_generation,
                        current.catalog_hash,
                    ),
                )
                for generation, value_hash, previous_generation, previous_hash in pairs:
                    if generation < previous_generation:
                        raise ValueError("serving control generation rollback rejected")
                    if generation == previous_generation and value_hash != previous_hash:
                        raise ValueError("serving control same-generation fork rejected")
                if candidate == current:
                    return
            _atomic_private_json(
                self.high_water_path,
                candidate.model_dump(mode="python", by_alias=True),
            )
            persisted = ServingHighWater.model_validate_json(
                read_confined_snapshot(
                    self.high_water_root,
                    self.high_water_path.name,
                    max_bytes=MAX_CONTROL_BYTES,
                )
            )
            if persisted != candidate:
                raise ValueError("serving high-water publication verification failed")

    def capture(self) -> CapturedServingControl:
        for _attempt in range(3):
            control = self._control_bytes()
            document = ServingControlDocument.model_validate_json(control)
            registry_bytes = read_confined_snapshot(
                self.control_root,
                document.registry.relative_path,
                max_bytes=MAX_DOCUMENT_BYTES,
            )
            catalog_bytes = read_confined_snapshot(
                self.control_root,
                document.catalog.relative_path,
                max_bytes=MAX_DOCUMENT_BYTES,
            )
            if hashlib.sha256(registry_bytes).hexdigest() != document.registry.sha256:
                raise ValueError("serving control registry byte hash mismatch")
            if hashlib.sha256(catalog_bytes).hexdigest() != document.catalog.sha256:
                raise ValueError("serving control catalog byte hash mismatch")
            registry = RegistryDocument.model_validate_json(registry_bytes)
            catalog = ArtifactCatalogDocument.model_validate_json(catalog_bytes)
            ServingRegistry._validate_unique(registry)
            if (
                registry.generation != document.registry.generation
                or canonical_sha256(registry.model_dump(mode="python", by_alias=True))
                != document.registry.semantic_hash
            ):
                raise ValueError("serving control registry semantic binding mismatch")
            if (
                catalog.generation != document.catalog.generation
                or canonical_sha256(catalog.model_dump(mode="python", by_alias=True))
                != document.catalog.semantic_hash
            ):
                raise ValueError("serving control catalog semantic binding mismatch")
            _CONTROL_ACQUIRE_SEAM("after-references")
            if self._control_bytes() != control:
                continue
            snapshot = CapturedServingControl(
                control_snapshot=control,
                document=document,
                registry_snapshot=registry_bytes,
                registry_document=registry,
                catalog_snapshot=catalog_bytes,
                catalog_document=catalog,
            )
            return snapshot
        raise ValueError("serving control changed during bounded acquisition")

    def accept(self, snapshot: CapturedServingControl) -> None:
        """Durably accept an already fully validated captured snapshot."""

        self._accept_high_water(snapshot)

    def acquire(
        self,
        validator: Callable[[CapturedServingControl], None] | None = None,
    ) -> CapturedServingControl:
        snapshot = self.capture()
        if validator is not None:
            validator(snapshot)
        self.accept(snapshot)
        return snapshot


__all__ = [
    "CapturedServingControl",
    "ServingControlDocument",
    "ServingControlReference",
    "ServingControlStore",
]
