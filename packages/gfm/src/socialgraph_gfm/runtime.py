"""Optional ML runtime discovery, seeds, roots, cancellation and run state."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .canonical import canonical_sha256
from .errors import (
    ArtifactRootNotConfigured,
    GfmError,
    MissingRuntimeDependency,
    RunCancelled,
    RuntimeVersionMismatch,
)

ARTIFACT_ROOT_ENV = "SOCIALGRAPH_FM_HOME"
FETCH_MIN_FREE_GIB = 30
RUN_MIN_FREE_GIB = 20
GIB = 1024**3
REQUIRED_VERSIONS = {
    "python": "3.12",
    "pydantic": "2.13.4",
    "numpy": "2.3.3",
    "torch": "2.12.0",
    "torch_geometric": "2.8.0.post1",
    "ogb": "1.3.6",
}
EXTENSION_VERSIONS = {
    "pyg_lib": "0.7.0",
    "torch_scatter": "2.1.2",
    "torch_sparse": "0.6.18",
}
PROFILE_LOCAL_VERSIONS = {
    "cpu-ci": {"torch": "2.8.0+cpu", "pyg_lib": "0.6.0+pt28cpu"},
    "windows-cpu": {"torch": "2.8.0+cpu", "pyg_lib": "0.6.0+pt28cpu"},
    "macos-arm64-cpu": {"torch": "2.8.0", "pyg_lib": "0.6.0+pt28"},
    "windows-cu130": {
        "torch": "2.12.0+cu130",
        "pyg_lib": "0.7.0+pt212cu130",
    },
    "linux-cu130": {
        "torch": "2.12.0+cu130",
        "pyg_lib": "0.7.0+pt212cu130",
        "torch_scatter": "2.1.2+pt212cu130",
        "torch_sparse": "0.6.18+pt212cu130",
    },
}
INSTALL_PROFILE_IDS = {
    "cpu-ci": "linux-x86_64-cpu-pt28",
    "windows-cpu": "windows-x86_64-cpu-pt28",
    "macos-arm64-cpu": "macos-arm64-cpu-pt28",
    "windows-cu130": "windows-x86_64-cu130-pt212",
    "linux-cu130": "linux-x86_64-cu130-pt212",
}
GFM_OPTIONAL_VERSIONS = {
    "FlagEmbedding": "1.4.0",
    "transformers": "5.14.1",
}


def artifact_root(override: str | Path | None = None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    configured = os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
    if not configured:
        raise ArtifactRootNotConfigured(
            f"{ARTIFACT_ROOT_ENV} must be set to the absolute GFM artifact root"
        )
    selected = Path(configured).expanduser()
    if not selected.is_absolute():
        raise ArtifactRootNotConfigured(
            f"{ARTIFACT_ROOT_ENV} must contain an absolute path"
        )
    return selected.resolve()


def core_runtime_root() -> Path:
    """Return the sole core runtime below the configured GFM artifact root."""

    return (artifact_root() / "core-runtime").resolve()


StorageOperation = Literal["fetch", "run"]


class InsufficientDiskSpace(GfmError):
    """Raised before a corpus download or run can consume the storage reserve."""

    code = "GFM_STORAGE_RESERVE_INSUFFICIENT"

    def __init__(self, *, root: Path, operation: StorageOperation, free_bytes: int, required_bytes: int):
        self.root = root
        self.operation = operation
        self.free_bytes = free_bytes
        self.required_bytes = required_bytes
        super().__init__(
            f"Insufficient free space for {operation}: "
            f"{free_bytes / GIB:.2f} GiB available at {root}, "
            f"{required_bytes / GIB:.0f} GiB required"
        )


@dataclass(frozen=True)
class RuntimeLayout:
    """All heavyweight baseline and GFM artifacts below one configurable non-Git root."""

    root: Path

    @classmethod
    def from_root(cls, override: str | Path | None = None) -> RuntimeLayout:
        return cls(artifact_root(override))

    @property
    def raw_ogb(self) -> Path:
        return self.root / "datasets" / "raw" / "ogb"

    @property
    def raw_gfm(self) -> Path:
        return self.root / "datasets" / "raw" / "gfm"

    @property
    def raw_openalex(self) -> Path:
        return self.raw_gfm / "openalex"

    @property
    def raw_thgl_software(self) -> Path:
        return self.raw_gfm / "thgl-software"

    @property
    def raw_wikimedia_talk(self) -> Path:
        return self.raw_gfm / "wikimedia-talk"

    @property
    def packages(self) -> Path:
        return self.root / "datasets" / "packages"

    @property
    def processed(self) -> Path:
        return self.root / "datasets" / "processed"

    @property
    def processed_gfm(self) -> Path:
        return self.processed / "gfm"

    @property
    def manifests(self) -> Path:
        return self.root / "datasets" / "manifests"

    @property
    def manifests_gfm(self) -> Path:
        return self.manifests / "gfm"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def gfm_runs(self) -> Path:
        return self.runs / "gfm"

    @property
    def registry(self) -> Path:
        return self.root / "registry"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def gfm_reports(self) -> Path:
        return self.reports / "gfm"

    @property
    def models_staging(self) -> Path:
        return self.root / "models" / "staging"

    @property
    def models_released(self) -> Path:
        return self.root / "models" / "released"

    @property
    def cache_hf(self) -> Path:
        return self.root / "cache" / "hf"

    @property
    def cache_pip(self) -> Path:
        return self.root / "cache" / "pip"

    @property
    def cache_uv(self) -> Path:
        return self.root / "cache" / "uv"

    @property
    def cache_torch(self) -> Path:
        return self.root / "cache" / "torch"

    @property
    def cache_torchinductor(self) -> Path:
        return self.root / "cache" / "torchinductor"

    @property
    def cache_wandb(self) -> Path:
        return self.root / "cache" / "wandb"

    @property
    def temporary(self) -> Path:
        return self.root / "tmp"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    def directories(self) -> tuple[Path, ...]:
        return (
            self.raw_ogb,
            self.raw_openalex,
            self.raw_thgl_software,
            self.raw_wikimedia_talk,
            self.packages,
            self.processed,
            self.processed_gfm,
            self.manifests,
            self.manifests_gfm,
            self.embeddings,
            self.runs,
            self.gfm_runs,
            self.registry,
            self.reports,
            self.gfm_reports,
            self.models_staging,
            self.models_released,
            self.cache_hf,
            self.cache_pip,
            self.cache_uv,
            self.cache_torch,
            self.cache_torchinductor,
            self.cache_wandb,
            self.temporary,
            self.exports,
        )

    def prepare(self) -> None:
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            "rawOgb": str(self.raw_ogb),
            "rawOpenAlex": str(self.raw_openalex),
            "rawThglSoftware": str(self.raw_thgl_software),
            "rawWikimediaTalk": str(self.raw_wikimedia_talk),
            "packages": str(self.packages),
            "processed": str(self.processed),
            "processedGfm": str(self.processed_gfm),
            "manifests": str(self.manifests),
            "manifestsGfm": str(self.manifests_gfm),
            "embeddings": str(self.embeddings),
            "runs": str(self.runs),
            "gfmRuns": str(self.gfm_runs),
            "registry": str(self.registry),
            "reports": str(self.reports),
            "gfmReports": str(self.gfm_reports),
            "modelsStaging": str(self.models_staging),
            "modelsReleased": str(self.models_released),
            "cacheHf": str(self.cache_hf),
            "cachePip": str(self.cache_pip),
            "cacheUv": str(self.cache_uv),
            "cacheTorch": str(self.cache_torch),
            "cacheTorchInductor": str(self.cache_torchinductor),
            "cacheWandb": str(self.cache_wandb),
            "temporary": str(self.temporary),
            "exports": str(self.exports),
        }


def _disk_usage_anchor(root: Path) -> Path:
    candidate = root
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if not candidate.exists():
        raise OSError(f"No existing filesystem anchor for runtime root: {root}")
    return candidate


def storage_report(
    root: str | Path | None = None,
    *,
    operation: StorageOperation = "run",
) -> dict[str, Any]:
    """Return the storage reserve decision without mutating the runtime root."""

    if operation not in ("fetch", "run"):
        raise ValueError("operation must be fetch or run")
    selected_root = artifact_root(root)
    anchor = _disk_usage_anchor(selected_root)
    usage = shutil.disk_usage(anchor)
    minimum_gib = FETCH_MIN_FREE_GIB if operation == "fetch" else RUN_MIN_FREE_GIB
    required_bytes = minimum_gib * GIB
    return {
        "schemaVersion": "gfm.storage/1.0",
        "operation": operation,
        "root": str(selected_root),
        "anchor": str(anchor),
        "freeBytes": usage.free,
        "freeGiB": round(usage.free / GIB, 3),
        "minimumFreeBytes": required_bytes,
        "minimumFreeGiB": minimum_gib,
        "ready": usage.free >= required_bytes,
    }


def require_storage_reserve(
    root: str | Path | None = None,
    *,
    operation: StorageOperation = "run",
) -> dict[str, Any]:
    """Fail closed if the operation would start below its minimum storage reserve."""

    report = storage_report(root, operation=operation)
    if not report["ready"]:
        raise InsufficientDiskSpace(
            root=Path(report["root"]),
            operation=operation,
            free_bytes=int(report["freeBytes"]),
            required_bytes=int(report["minimumFreeBytes"]),
        )
    return report


def prepare_runtime_layout(
    root: str | Path | None = None,
    *,
    operation: StorageOperation = "run",
) -> RuntimeLayout:
    """Check the reserve first, then create the fixed baseline directory layout."""

    require_storage_reserve(root, operation=operation)
    layout = RuntimeLayout.from_root(root)
    layout.prepare()
    return layout


def _version(module_name: str) -> str | None:
    distribution_name = {
        "torch_geometric": "torch-geometric",
        "pyg_lib": "pyg-lib",
        "torch_scatter": "torch-scatter",
        "torch_sparse": "torch-sparse",
    }.get(module_name, module_name)
    try:
        importlib.import_module(module_name)
        return importlib.metadata.version(distribution_name)
    except (ImportError, OSError, importlib.metadata.PackageNotFoundError):
        return None


def gfm_optional_runtime_report() -> dict[str, Any]:
    """Report optional corpus/text packages without weakening base runtime readiness."""

    installed: dict[str, str | None] = {}
    for distribution, expected in GFM_OPTIONAL_VERSIONS.items():
        try:
            installed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed[distribution] = None
    matches = {
        name: installed[name] == expected for name, expected in GFM_OPTIONAL_VERSIONS.items()
    }
    return {
        "schemaVersion": "gfm.optional-runtime/1.0",
        "expected": dict(GFM_OPTIONAL_VERSIONS),
        "installed": installed,
        "matches": matches,
        "dataReady": True,
        "textReady": matches["FlagEmbedding"] and matches["transformers"],
    }


def require_gfm_optional_runtime(*, data: bool = False, text: bool = False) -> dict[str, Any]:
    report = gfm_optional_runtime_report()
    missing: list[str] = []
    if data and not report["dataReady"]:  # pragma: no cover - built-in CSV adapter
        missing.append("the built-in pickle-free TGB CSV adapter")
    if text and not report["textReady"]:
        missing.extend(("FlagEmbedding==1.4.0", "transformers==5.14.1"))
    if missing:
        raise MissingRuntimeDependency(
            "Missing exact optional GFM dependencies: " + ", ".join(sorted(set(missing)))
        )
    return report


def runtime_report(device: str = "cpu") -> dict[str, Any]:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    torch_version = _version("torch")
    cuda_wheel_installed = bool(torch_version and "+cu130" in torch_version)
    system = platform.system()
    raw_machine = platform.machine()
    normalized_machine = {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(raw_machine.lower(), raw_machine.lower())
    libc_name, libc_version = platform.libc_ver()
    platform_mismatch: dict[str, str] | None = None
    selected_profile: str | None = None
    if device == "cuda" or cuda_wheel_installed:
        if system == "Windows" and normalized_machine == "x86_64":
            selected_profile = "windows-cu130"
        elif system == "Linux" and normalized_machine == "x86_64" and libc_name == "glibc":
            selected_profile = "linux-cu130"
        else:
            platform_mismatch = {
                "package": "platform",
                "required": "Windows/Linux glibc x86_64 for CUDA 13.0",
                "actual": f"{system}/{normalized_machine}/{libc_name or 'unknown-libc'}",
            }
    elif system == "Windows" and normalized_machine == "x86_64":
        selected_profile = "windows-cpu"
    elif system == "Linux" and normalized_machine == "x86_64" and libc_name == "glibc":
        selected_profile = "cpu-ci"
    elif system == "Darwin" and normalized_machine == "arm64":
        selected_profile = "macos-arm64-cpu"
    else:
        platform_mismatch = {
            "package": "platform",
            "required": "Windows/Linux x86_64 or macOS arm64",
            "actual": f"{system}/{normalized_machine}/{libc_name or 'none'}",
        }

    # Keep reporting useful dependency diagnostics even on an unsupported host.
    # The fallback is not considered ready because platform_mismatch is retained.
    version_profile = selected_profile or (
        "windows-cu130" if device == "cuda" or cuda_wheel_installed else "windows-cpu"
    )
    required_modules = ["pydantic", "numpy", "torch", "torch_geometric", "ogb", "pyg_lib"]
    if selected_profile == "linux-cu130":
        required_modules.extend(("torch_scatter", "torch_sparse"))
    versions: dict[str, str | None] = {
        "python": python_version,
        **{
            name: torch_version if name == "torch" else _version(name)
            for name in required_modules
        },
    }
    missing = [name for name in required_modules if versions[name] is None]
    mismatches = [platform_mismatch] if platform_mismatch is not None else []
    if not python_version.startswith(REQUIRED_VERSIONS["python"] + "."):
        mismatches.append({"package": "python", "required": "3.12.*", "actual": python_version})
    for name in required_modules:
        actual = versions[name]
        required = PROFILE_LOCAL_VERSIONS[version_profile].get(
            name, REQUIRED_VERSIONS.get(name, EXTENSION_VERSIONS.get(name))
        )
        if actual is not None and required is not None and actual != required:
            mismatches.append({"package": name, "required": required, "actual": actual})

    cuda: dict[str, Any] = {
        "requested": device == "cuda",
        "available": False,
        "runtime": None,
        "deviceName": None,
    }
    if versions["torch"] is not None:
        import torch

        cuda.update(
            {
                "available": bool(torch.cuda.is_available()),
                "runtime": torch.version.cuda,
                "deviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    if device == "cuda" and not cuda["available"]:
        mismatches.append({"package": "cuda", "required": "available", "actual": "unavailable"})
    if device == "cuda" and cuda["runtime"] is not None and not str(cuda["runtime"]).startswith("13.0"):
        mismatches.append({"package": "cuda", "required": "13.0", "actual": cuda["runtime"]})

    report: dict[str, Any] = {
        "schemaVersion": "gfm.doctor/1.0",
        "selectedProfile": selected_profile,
        "installProfile": INSTALL_PROFILE_IDS.get(selected_profile or ""),
        "platform": {
            "system": system,
            "release": platform.release(),
            "machine": raw_machine,
            "normalizedMachine": normalized_machine,
            "libc": libc_name or None,
            "libcVersion": libc_version or None,
        },
        "versions": versions,
        "cuda": cuda,
        "missing": missing,
        "mismatches": mismatches,
    }
    report["runtimeReady"] = not missing and not mismatches
    report["environmentHash"] = canonical_sha256(report)
    return report


def require_ml_runtime(device: str = "cpu"):
    report = runtime_report(device)
    if report["missing"]:
        raise MissingRuntimeDependency(
            "Missing required ML packages: " + ", ".join(report["missing"])
        )
    if report["mismatches"]:
        summary = ", ".join(
            f"{item['package']}={item['actual']} (required {item['required']})"
            for item in report["mismatches"]
        )
        raise RuntimeVersionMismatch("Runtime does not match the selected exact profile: " + summary)
    if device == "cuda" and not report["cuda"]["available"]:
        raise MissingRuntimeDependency("CUDA was requested but torch.cuda.is_available() is false")
    import torch
    import torch_geometric

    return torch, torch_geometric


def set_seed(seed: int, device: str = "cpu") -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


@dataclass
class RunContext:
    run_id: str
    root: Path

    @property
    def directory(self) -> Path:
        return self.root / "runs" / self.run_id

    @property
    def cancel_path(self) -> Path:
        return self.directory / "CANCEL"

    def prepare(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=False)

    def check_cancelled(self) -> None:
        if self.cancel_path.exists():
            raise RunCancelled(f"Run {self.run_id} was cancelled")

    def log(self, event: str, **details: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **details,
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n")
