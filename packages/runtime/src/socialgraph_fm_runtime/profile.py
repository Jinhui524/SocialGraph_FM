"""Atomic runtime-profile persistence."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "socialgraph-fm.runtime-profile/3.0"
LEGACY_PROFILE_SCHEMA_VERSION = "socialgraph-fm.runtime-profile/2.0"
FINGERPRINT_SCHEMA_VERSION = "socialgraph-fm.python-environment-fingerprint/2.0"
VALID_PROFILES = {"offline", "cpu", "cuda"}
VALID_ENV_MODES = {"auto", "reuse", "managed"}
VALID_DEVICE_POLICIES = {"auto", "cpu", "cuda-required"}


def atomic_write_json(path: Path, document: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        if private and os.name != "nt":
            fchmod = getattr(os, "fchmod", None)
            if fchmod is None:  # pragma: no cover - guarded POSIX branch
                raise RuntimeError("POSIX fchmod is unavailable")
            fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        json.loads(temporary.read_text(encoding="utf-8"))
        os.replace(temporary, path)
        if private and os.name != "nt":
            os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RuntimeProfile:
    profile: str
    env_mode: str
    install_profile_id: str | None
    platform: dict[str, str]
    interpreters: dict[str, dict[str, Any] | None]
    updated_at_utc: str
    device_policy: str = "auto"
    setup_summary: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        profile: str,
        env_mode: str,
        install_profile_id: str | None,
        platform: dict[str, str],
        interpreters: dict[str, dict[str, Any] | None],
        device_policy: str = "auto",
        setup_summary: dict[str, Any] | None = None,
    ) -> "RuntimeProfile":
        normalized_profile = profile.lower()
        normalized_mode = env_mode.lower()
        normalized_device_policy = device_policy.lower()
        if normalized_profile not in VALID_PROFILES:
            raise ValueError(f"Unsupported runtime profile: {profile}")
        if normalized_mode not in VALID_ENV_MODES:
            raise ValueError(f"Unsupported environment mode: {env_mode}")
        if normalized_device_policy not in VALID_DEVICE_POLICIES:
            raise ValueError(f"Unsupported device policy: {device_policy}")
        if normalized_profile == "offline" and normalized_device_policy != "auto":
            raise ValueError("Offline runtime profiles require the auto device policy")
        if normalized_profile == "cpu" and normalized_device_policy == "cuda-required":
            raise ValueError("The CPU wheel profile cannot require CUDA execution")
        return cls(
            profile=normalized_profile,
            env_mode=normalized_mode,
            install_profile_id=install_profile_id,
            platform=platform,
            interpreters=interpreters,
            updated_at_utc=datetime.now(UTC).isoformat(),
            device_policy=normalized_device_policy,
            setup_summary=setup_summary,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROFILE_SCHEMA_VERSION,
            "profile": self.profile,
            "envMode": self.env_mode,
            "devicePolicy": self.device_policy,
            "installProfileId": self.install_profile_id,
            "platform": self.platform,
            "interpreters": self.interpreters,
            "updatedAtUtc": self.updated_at_utc,
        }

    def write(self, path: Path) -> None:
        # atomic_write_json validates the complete temporary document before the
        # replacement. Returning immediately after os.replace avoids treating a
        # transient post-commit read error as a failed switch and deleting the
        # generation now referenced by the committed profile.
        atomic_write_json(path, self.to_document())

    @classmethod
    def load(cls, path: Path) -> "RuntimeProfile":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError(f"Runtime profile is missing; run setup first: {path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Runtime profile is invalid: {path}") from error
        source_schema = document.get("schemaVersion")
        if source_schema not in {PROFILE_SCHEMA_VERSION, LEGACY_PROFILE_SCHEMA_VERSION}:
            raise RuntimeError(f"Unsupported runtime profile schema: {path}")
        profile = str(document.get("profile", "")).lower()
        env_mode = str(document.get("envMode", "")).lower()
        if profile not in VALID_PROFILES or env_mode not in VALID_ENV_MODES:
            raise RuntimeError(f"Runtime profile values are invalid: {path}")
        interpreters = document.get("interpreters")
        platform_document = document.get("platform")
        if not isinstance(interpreters, dict) or not isinstance(platform_document, dict):
            raise RuntimeError(f"Runtime profile interpreter/platform data is invalid: {path}")
        normalized_interpreters = deepcopy(interpreters)
        if source_schema == LEGACY_PROFILE_SCHEMA_VERSION:
            normalized_interpreters = _migrate_v2_interpreters(normalized_interpreters)
        device_policy = str(document.get("devicePolicy", "auto")).lower()
        if device_policy not in VALID_DEVICE_POLICIES:
            raise RuntimeError(f"Runtime profile device policy is invalid: {path}")
        if profile == "offline" and device_policy != "auto":
            raise RuntimeError(f"Offline runtime profile device policy is invalid: {path}")
        if profile == "cpu" and device_policy == "cuda-required":
            raise RuntimeError(f"CPU runtime profile cannot require CUDA: {path}")
        return cls(
            profile=profile,
            env_mode=env_mode,
            install_profile_id=document.get("installProfileId"),
            platform={str(key): str(value) for key, value in platform_document.items()},
            interpreters=normalized_interpreters,
            updated_at_utc=str(document.get("updatedAtUtc", "")),
            device_policy=device_policy,
            setup_summary=None,
        )


def _migrate_v2_interpreters(
    interpreters: dict[str, dict[str, Any] | None],
) -> dict[str, dict[str, Any] | None]:
    """Normalize v2 hardware-bound fingerprints into v3 software fingerprints."""

    for record in interpreters.values():
        if not isinstance(record, dict):
            continue
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        identity = {
            key: deepcopy(value)
            for key, value in fingerprint.items()
            if key != "fingerprintSha256"
        }
        torch_identity = identity.get("torch")
        if isinstance(torch_identity, dict):
            identity["torch"] = {
                key: value
                for key, value in torch_identity.items()
                if key not in {"cudaAvailable", "deviceName", "deviceCount", "deviceCapability"}
            }
        identity["schemaVersion"] = FINGERPRINT_SCHEMA_VERSION
        encoded = json.dumps(
            identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        identity["fingerprintSha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        record["fingerprint"] = identity
    return interpreters
