from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import Settings

_PROBE = r"""
import importlib.metadata
import json
import platform
import sys

packages = {}
for name in ("torch", "torch-geometric", "ogb", "numpy"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

cuda = {"compiledVersion": None, "available": False, "deviceCount": 0, "devices": []}
try:
    import torch
    cuda["compiledVersion"] = torch.version.cuda
    cuda["available"] = bool(torch.cuda.is_available())
    if cuda["available"]:
        cuda["deviceCount"] = int(torch.cuda.device_count())
        cuda["devices"] = [torch.cuda.get_device_name(i) for i in range(cuda["deviceCount"])]
except Exception as exc:
    cuda["probeError"] = type(exc).__name__

print(json.dumps({
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "python": platform.python_version(),
    "packages": packages,
    "cuda": cuda,
}, sort_keys=True, separators=(",", ":")))
"""


def _run(command: list[str], *, cwd: Path, timeout: float = 10) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _dependency_manifest(source_root: Path) -> dict[str, str]:
    candidates = [
        source_root / "pyproject.toml",
        source_root / "requirements-dev.txt",
        source_root / "requirements-dev.lock",
        *sorted((source_root / "constraints").glob("**/*")),
    ]
    result: dict[str, str] = {}
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _source_tree_manifest(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory in (source_root / "app", source_root / "tools"):
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            result[path.relative_to(source_root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return result


@lru_cache(maxsize=8)
def _interpreter_details(executable: str, source_root_text: str) -> dict[str, Any]:
    source_root = Path(source_root_text)
    output = _run([executable, "-c", _PROBE], cwd=source_root, timeout=20)
    if output is None:
        return {
            "executable": str(Path(executable).expanduser()),
            "probeError": "INTERPRETER_PROBE_FAILED",
            "packages": {},
            "cuda": {"available": False},
        }
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        return {
            "executable": str(Path(executable).expanduser()),
            "probeError": "INTERPRETER_PROBE_INVALID_JSON",
            "packages": {},
            "cuda": {"available": False},
        }
    return value if isinstance(value, dict) else {"probeError": "INTERPRETER_PROBE_INVALID"}


def converter_environment_details(settings: Settings) -> dict[str, Any]:
    source_root = Path(__file__).resolve().parents[1]
    executable = settings.trusted_converter_python or sys.executable
    source_commit = os.environ.get("SOURCE_COMMIT") or os.environ.get("GIT_COMMIT")
    if not source_commit:
        source_commit = _run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
        ) or "unknown"
    dependencies = _dependency_manifest(source_root)
    dependency_hash = hashlib.sha256(
        json.dumps(
            dependencies,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    source_tree = _source_tree_manifest(source_root)
    source_tree_hash = hashlib.sha256(
        json.dumps(
            source_tree,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schemaVersion": "converter-environment/1.0",
        "runtimeBuildId": settings.runtime_build_id,
        "trustedConversionEnabled": settings.enable_trusted_local_conversion,
        "interpreter": _interpreter_details(executable, str(source_root)),
        "sourceCommit": source_commit,
        "sourceTreeHash": source_tree_hash,
        "dependencyHash": dependency_hash,
        "dependencies": dependencies,
    }


def converter_environment_fingerprint(settings: Settings) -> str:
    details = converter_environment_details(settings)
    return hashlib.sha256(
        json.dumps(
            details,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def public_converter_environment_summary(details: dict[str, Any]) -> dict[str, Any]:
    """Return capability metadata without filesystem paths or device identities."""

    interpreter = details.get("interpreter")
    safe_interpreter: dict[str, Any] = {}
    if isinstance(interpreter, dict):
        for key in ("implementation", "python", "probeError"):
            value = interpreter.get(key)
            if isinstance(value, (str, bool, int, float)) or value is None:
                safe_interpreter[key] = value
        packages = interpreter.get("packages")
        if isinstance(packages, dict):
            safe_interpreter["packages"] = {
                str(key): value
                for key, value in packages.items()
                if isinstance(value, str) or value is None
            }
        cuda = interpreter.get("cuda")
        if isinstance(cuda, dict):
            safe_interpreter["cuda"] = {
                key: cuda.get(key)
                for key in ("compiledVersion", "available", "deviceCount", "probeError")
                if key in cuda
            }
    return {
        "schemaVersion": "converter-environment-public/1.0",
        "runtimeBuildId": details.get("runtimeBuildId"),
        "trustedConversionEnabled": bool(details.get("trustedConversionEnabled")),
        "interpreter": safe_interpreter,
        "sourceCommit": details.get("sourceCommit"),
        "sourceTreeHash": details.get("sourceTreeHash"),
        "dependencyHash": details.get("dependencyHash"),
    }
