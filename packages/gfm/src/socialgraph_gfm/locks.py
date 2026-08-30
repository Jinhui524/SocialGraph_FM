"""Verify the checked-in runtime constraint lock manifest without network access."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .canonical import file_sha256

_SHA256_TOKEN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _logical_requirements(path: Path) -> tuple[str, ...]:
    """Parse pip's backslash-continued requirement records without resolving them."""

    records: list[str] = []
    pending = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1].strip() + " "
            continue
        pending += stripped
        records.append(pending.strip())
        pending = ""
    if pending:
        # A lock ending in a continuation is structurally incomplete and is retained as a
        # record so hash coverage cannot accidentally pass.
        records.append(pending.strip())
    return tuple(records)


def _verify_requirement_hashes(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "requirementCount": 0,
            "unhashedRequirements": [],
            "invalidHashTokens": [],
            "hashCoverageValid": False,
        }
    requirements = [
        record
        for record in _logical_requirements(path)
        if not record.startswith(("-", "--"))
    ]
    unhashed: list[str] = []
    invalid_tokens: list[str] = []
    for requirement in requirements:
        tokens = re.findall(r"--hash=([^\s\\]+)", requirement)
        valid = [token for token in tokens if _SHA256_TOKEN.fullmatch(token)]
        invalid_tokens.extend(token for token in tokens if token not in valid)
        if not valid:
            unhashed.append(requirement.split(" ", 1)[0])
    return {
        "requirementCount": len(requirements),
        "unhashedRequirements": unhashed,
        "invalidHashTokens": invalid_tokens,
        "hashCoverageValid": bool(requirements) and not unhashed and not invalid_tokens,
    }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def verify_lock_manifest(root: str | Path | None = None) -> dict[str, Any]:
    project = Path(root) if root is not None else repository_root()
    manifest_path = project / "locks" / "runtime-lock-manifest.json"
    constraints_root = project
    if not manifest_path.is_file():
        resources = Path(__file__).resolve().parent / "resources"
        manifest_path = resources / "runtime-lock-manifest.json"
        constraints_root = resources
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profiles: dict[str, Any] = {}
    for name, profile in manifest["profiles"].items():
        constraints = constraints_root / profile["constraints"]
        requirements_lock = constraints_root / profile["requirementsLock"]
        constraints_actual = file_sha256(constraints) if constraints.is_file() else None
        constraints_expected = profile["constraintsSha256"]
        lock_actual = file_sha256(requirements_lock) if requirements_lock.is_file() else None
        lock_expected = profile["requirementsLockSha256"]
        hash_coverage = _verify_requirement_hashes(requirements_lock)
        hash_coverage_verified = (
            bool(profile["artifactHashesResolved"])
            and lock_actual == lock_expected
            and hash_coverage["hashCoverageValid"]
        )
        release_ready = constraints_actual == constraints_expected and hash_coverage_verified
        profiles[name] = {
            "constraints": str(constraints),
            "constraintsPresent": constraints.is_file(),
            "constraintsSha256": constraints_actual,
            "constraintIntegrityValid": constraints_actual == constraints_expected,
            "requirementsLock": str(requirements_lock),
            "requirementsLockPresent": requirements_lock.is_file(),
            "requirementsLockSha256": lock_actual,
            "requirementsLockIntegrityValid": lock_actual == lock_expected,
            "artifactHashesResolved": profile["artifactHashesResolved"],
            "hashCoverageVerified": hash_coverage_verified,
            # Compatibility alias. This verifies checked-in hash declarations, not downloads.
            "artifactHashesVerified": hash_coverage_verified,
            **hash_coverage,
            "releaseLockReady": release_ready,
            "optionalExtensionsUnavailable": profile["optionalExtensionsUnavailable"],
        }
    return {
        "schemaVersion": "gfm.runtime-lock-verification/1.0",
        "manifest": str(manifest_path),
        "profiles": profiles,
        "constraintIntegrityValid": all(
            profile["constraintIntegrityValid"] for profile in profiles.values()
        ),
        "requirementsLockIntegrityValid": all(
            profile["requirementsLockIntegrityValid"] for profile in profiles.values()
        ),
        "artifactHashesVerified": all(
            profile["artifactHashesVerified"] for profile in profiles.values()
        ),
        "hashCoverageVerified": all(
            profile["hashCoverageVerified"] for profile in profiles.values()
        ),
        "verificationScope": "checked_in_lock_integrity_and_hash_coverage",
        "releaseLocksReady": all(profile["releaseLockReady"] for profile in profiles.values()),
    }
