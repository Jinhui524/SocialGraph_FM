from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "socialgraph-fm-environment-preflight/1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = PROJECT_ROOT / "constraints" / "environment-profiles.json"
EXACT_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)==(?P<version>[^\s;]+)$"
)


class PreflightConfigurationError(ValueError):
    """Raised when the checked-in environment contract is not exact or complete."""


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def parse_exact_constraints(path: Path) -> dict[str, str]:
    """Read exact ``name==version`` pins and reject ranges or duplicate names."""

    pins: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = EXACT_REQUIREMENT.fullmatch(line)
        if match is None:
            raise PreflightConfigurationError(
                f"{path}:{line_number} must use an exact name==version pin"
            )
        name = _canonical_package_name(match.group("name"))
        if name in pins:
            raise PreflightConfigurationError(
                f"{path}:{line_number} duplicates package {name}"
            )
        pins[name] = match.group("version")
    if not pins:
        raise PreflightConfigurationError(f"{path} contains no package pins")
    return pins


def _load_profile(profiles_path: Path, profile_name: str) -> tuple[dict[str, Any], Path]:
    document = json.loads(profiles_path.read_text(encoding="utf-8"))
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        available = ", ".join(sorted(profiles or {}))
        raise PreflightConfigurationError(
            f"unknown profile {profile_name!r}; available profiles: {available}"
        )
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise PreflightConfigurationError(f"profile {profile_name!r} must be an object")
    constraint_name = profile.get("constraints")
    if not isinstance(constraint_name, str) or not constraint_name:
        raise PreflightConfigurationError(
            f"profile {profile_name!r} has no constraints file"
        )
    constraints_path = (profiles_path.parent / constraint_name).resolve()
    return profile, constraints_path


_PROBE_CODE = r"""
import importlib
import importlib.metadata
import json
import os
import platform
import struct
import sys

request = json.loads(sys.argv[1])
packages = {}
for package_name in request["packages"]:
    try:
        packages[package_name] = importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        packages[package_name] = None

imports = {}
for module_name in request["modules"]:
    try:
        importlib.import_module(module_name)
        imports[module_name] = {"ok": True}
    except Exception as exc:
        imports[module_name] = {
            "ok": False,
            "errorType": type(exc).__name__,
            "message": str(exc)[:300],
        }

print(json.dumps({
    "python": {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "basePrefix": sys.base_prefix,
        "architectureBits": struct.calcsize("P") * 8,
        "isolation": {
            "prefixDiffersFromBase": sys.prefix != sys.base_prefix,
            "pyvenvConfigPresent": os.path.isfile(os.path.join(sys.prefix, "pyvenv.cfg")),
            "condaEnvironmentPath": (
                os.path.isdir(os.path.join(sys.prefix, "conda-meta"))
                and "envs" in {part.lower() for part in os.path.normpath(sys.prefix).split(os.sep)}
            ),
        },
    },
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    },
    "packages": packages,
    "imports": imports,
}, sort_keys=True))
"""


def _run(
    command: list[str],
    *,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )


def probe_python(
    executable: Path,
    packages: list[str],
    modules: list[str],
    *,
    timeout_seconds: float = 90,
) -> dict[str, Any]:
    request = json.dumps({"packages": packages, "modules": modules})
    completed = _run(
        [str(executable), "-c", _PROBE_CODE, request],
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:500]
        raise RuntimeError(
            f"Python probe exited with {completed.returncode}: {detail or 'no output'}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Python probe did not return valid JSON") from exc


def run_pip_check(executable: Path, *, timeout_seconds: float = 90) -> dict[str, Any]:
    try:
        completed = _run(
            [str(executable), "-m", "pip", "check"],
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "returnCode": None,
            "output": "pip check exceeded its hard timeout",
        }
    output = "\n".join(
        value.strip() for value in (completed.stdout, completed.stderr) if value.strip()
    )
    return {
        "ok": completed.returncode == 0,
        "returnCode": completed.returncode,
        "output": output[:4_000],
    }


def evaluate_environment(
    *,
    profile_name: str,
    profile: dict[str, Any],
    constraints_path: Path,
    expected_packages: dict[str, str],
    probe: dict[str, Any],
    pip_check: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    expected_python = str(profile.get("pythonExact", ""))
    expected_python_series = str(profile.get("pythonSeries", ""))
    actual_python = str(probe["python"]["version"])
    python_matches = (
        actual_python == expected_python
        if expected_python
        else bool(expected_python_series)
        and actual_python.startswith(expected_python_series + ".")
    )
    if not python_matches:
        expected_label = (
            expected_python
            or (f"{expected_python_series}.x" if expected_python_series else "unspecified")
        )
        issues.append(
            {
                "severity": "error",
                "code": "PYTHON_VERSION_MISMATCH",
                "message": f"expected Python {expected_label}, got {actual_python}",
            }
        )

    expected_system = profile.get("platformSystem")
    actual_system = probe["platform"].get("system")
    if expected_system and actual_system != expected_system:
        issues.append(
            {
                "severity": "error",
                "code": "PLATFORM_SYSTEM_MISMATCH",
                "message": f"expected {expected_system}, got {actual_system}",
            }
        )
    expected_bits = profile.get("architectureBits")
    actual_bits = probe["python"].get("architectureBits")
    if expected_bits and actual_bits != expected_bits:
        issues.append(
            {
                "severity": "error",
                "code": "PLATFORM_ARCHITECTURE_MISMATCH",
                "message": f"expected {expected_bits}-bit, got {actual_bits}-bit",
            }
        )

    if profile.get("requireIsolated", True):
        isolation = probe["python"].get("isolation", {})
        isolated = any(
            isolation.get(key) is True
            for key in (
                "prefixDiffersFromBase",
                "pyvenvConfigPresent",
                "condaEnvironmentPath",
            )
        )
        if not isolated:
            issues.append(
                {
                    "severity": "error",
                    "code": "ENVIRONMENT_NOT_ISOLATED",
                    "message": "sys.prefix equals sys.base_prefix; use a dedicated venv or Conda environment",
                }
            )

    actual_packages = {
        _canonical_package_name(name): version
        for name, version in probe.get("packages", {}).items()
    }
    package_rows: list[dict[str, Any]] = []
    for name, expected_version in sorted(expected_packages.items()):
        actual_version = actual_packages.get(name)
        matches = actual_version == expected_version
        package_rows.append(
            {
                "name": name,
                "expectedVersion": expected_version,
                "actualVersion": actual_version,
                "matches": matches,
            }
        )
        if not matches:
            issues.append(
                {
                    "severity": "error",
                    "code": "PACKAGE_VERSION_MISMATCH" if actual_version else "PACKAGE_MISSING",
                    "message": f"{name}: expected {expected_version}, got {actual_version or 'missing'}",
                }
            )

    for module_name, result in sorted(probe.get("imports", {}).items()):
        if not result.get("ok"):
            issues.append(
                {
                    "severity": "error",
                    "code": "IMPORT_SMOKE_FAILED",
                    "message": f"{module_name}: {result.get('errorType', 'Error')}: {result.get('message', '')}",
                }
            )

    if not pip_check.get("ok"):
        issues.append(
            {
                "severity": "error",
                "code": "PIP_CHECK_FAILED",
                "message": str(pip_check.get("output") or "pip check failed")[:500],
            }
        )

    constraint_hash = _sha256_bytes(constraints_path.read_bytes())
    fingerprint_payload = {
        "profile": profile_name,
        "python": probe["python"],
        "platform": probe["platform"],
        "packages": {row["name"]: row["actualVersion"] for row in package_rows},
        "constraintsSha256": constraint_hash,
    }
    return {
        "schemaVersion": REPORT_SCHEMA,
        "generatedAt": datetime.now(UTC).isoformat(),
        "profile": profile_name,
        "status": "ready" if not issues else "blocked",
        "python": probe["python"],
        "platform": probe["platform"],
        "constraints": {
            "path": str(constraints_path),
            "sha256": constraint_hash,
            "packageCount": len(package_rows),
        },
        "packages": package_rows,
        "importSmoke": probe.get("imports", {}),
        "pipCheck": pip_check,
        "runtimeFingerprint": _canonical_hash(fingerprint_payload),
        "issues": issues,
        "environmentValuesCaptured": False,
    }


def build_report(
    *,
    profile_name: str,
    executable: Path,
    profiles_path: Path = DEFAULT_PROFILES,
) -> dict[str, Any]:
    profile, constraints_path = _load_profile(profiles_path.resolve(), profile_name)
    expected_packages = parse_exact_constraints(constraints_path)
    modules = profile.get("importModules", [])
    if not isinstance(modules, list) or not all(isinstance(value, str) for value in modules):
        raise PreflightConfigurationError("importModules must be an array of module names")
    probe = probe_python(executable.resolve(), sorted(expected_packages), modules)
    pip_check = run_pip_check(executable.resolve())
    return evaluate_environment(
        profile_name=profile_name,
        profile=profile,
        constraints_path=constraints_path,
        expected_packages=expected_packages,
        probe=probe,
        pip_check=pip_check,
    )


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _fatal_report(profile_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "schemaVersion": REPORT_SCHEMA,
        "generatedAt": datetime.now(UTC).isoformat(),
        "profile": profile_name,
        "status": "blocked",
        "issues": [
            {
                "severity": "error",
                "code": "PREFLIGHT_EXECUTION_FAILED",
                "message": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
        ],
        "environmentValuesCaptured": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify an isolated Python runtime against checked-in exact pins."
    )
    parser.add_argument("--profile", required=True, help="Profile name from environment-profiles.json")
    parser.add_argument("--python", default=sys.executable, help="Python executable to inspect")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)

    try:
        report = build_report(
            profile_name=args.profile,
            executable=Path(args.python),
            profiles_path=args.profiles,
        )
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        report = _fatal_report(args.profile, exc)

    if args.output:
        _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())
