"""Atomic persistence for the single managed CPU runtime binding."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROFILE_SCHEMA_VERSION = "socialgraph-fm.runtime-profile/4.0"


def atomic_write_json(path: Path, document: dict[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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
    install_profile_id: str
    platform: dict[str, str]
    interpreter: dict[str, Any]
    updated_at_utc: str
    setup_summary: dict[str, Any] | None = None

    @classmethod
    def create(
        cls,
        *,
        install_profile_id: str,
        platform: dict[str, str],
        interpreter: dict[str, Any],
        setup_summary: dict[str, Any] | None = None,
    ) -> "RuntimeProfile":
        if not install_profile_id.endswith("-cpu-pt28"):
            raise ValueError("Runtime profile must reference a verified CPU install profile")
        if not isinstance(interpreter.get("fingerprint"), dict):
            raise ValueError("Runtime profile requires a verified interpreter fingerprint")
        return cls(
            install_profile_id=install_profile_id,
            platform={str(key): str(value) for key, value in platform.items()},
            interpreter=interpreter,
            updated_at_utc=datetime.now(UTC).isoformat(),
            setup_summary=setup_summary,
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROFILE_SCHEMA_VERSION,
            "installProfileId": self.install_profile_id,
            "platform": self.platform,
            "interpreter": self.interpreter,
            "updatedAtUtc": self.updated_at_utc,
        }

    def write(self, path: Path) -> None:
        atomic_write_json(path, self.to_document())

    @classmethod
    def load(cls, path: Path) -> "RuntimeProfile":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise RuntimeError(f"Runtime profile is missing; run onboard first: {path}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Runtime profile is invalid: {path}") from error
        if document.get("schemaVersion") != PROFILE_SCHEMA_VERSION:
            raise RuntimeError(
                "The recorded runtime uses a retired setup profile; run onboard again"
            )
        install_profile_id = document.get("installProfileId")
        platform_document = document.get("platform")
        interpreter = document.get("interpreter")
        if (
            not isinstance(install_profile_id, str)
            or not install_profile_id.endswith("-cpu-pt28")
            or not isinstance(platform_document, dict)
            or not isinstance(interpreter, dict)
            or not isinstance(interpreter.get("fingerprint"), dict)
        ):
            raise RuntimeError(f"Runtime profile values are invalid: {path}")
        return cls(
            install_profile_id=install_profile_id,
            platform={str(key): str(value) for key, value in platform_document.items()},
            interpreter=interpreter,
            updated_at_utc=str(document.get("updatedAtUtc", "")),
            setup_summary=None,
        )
