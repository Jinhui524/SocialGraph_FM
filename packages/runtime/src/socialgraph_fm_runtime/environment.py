"""Interpreter discovery, clean probing, compatibility checks, and pip installation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .layout import RuntimeLayout, environment_python
from .subprocess_control import run_captured_process, run_streaming_process


PROBE_SCHEMA_VERSION = "socialgraph-fm.python-environment-probe/1.0"
FINGERPRINT_SCHEMA_VERSION = "socialgraph-fm.python-environment-fingerprint/2.0"
VALID_DEVICE_POLICIES = {"auto", "cpu", "cuda-required"}

# Parent-shell credentials are never an implicit configuration source. Only the
# private LLM_API_* file is whitelisted into the Torch-free API service.
AMBIENT_LLM_ENVIRONMENT_NAMES = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "COHERE_API_KEY",
        "DASHSCOPE_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "GEMINI_API_KEY",
        "GLM_API_KEY",
        "GLM_BASE_URL",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "MINIMAX_API_KEY",
        "MISTRAL_API_KEY",
        "MOONSHOT_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "ZHIPUAI_API_KEY",
        "ZHIPUAI_BASE_URL",
    }
)


def is_ambient_llm_environment_name(name: str) -> bool:
    upper = name.upper()
    return upper.startswith("LLM_") or upper in AMBIENT_LLM_ENVIRONMENT_NAMES

API_DISTRIBUTIONS = {
    "fastapi": "fastapi",
    "httpx": "httpx",
    "numpy": "numpy",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "python-multipart": "multipart",
    "uvicorn": "uvicorn",
}

GFM_MODULES = {
    "torch": "torch",
    "torch-geometric": "torch_geometric",
    "pyg-lib": "pyg_lib",
    "torch-scatter": "torch_scatter",
    "torch-sparse": "torch_sparse",
    "numpy": "numpy",
    "pydantic": "pydantic",
    "ogb": "ogb",
    "FlagEmbedding": "FlagEmbedding",
    "transformers": "transformers",
}


_PROBE_SOURCE = r"""
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys

request = json.load(sys.stdin)
versions = {}
imports = {}
for distribution, module in request["distributions"].items():
    try:
        versions[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        versions[distribution] = None
    try:
        importlib.import_module(module)
        imports[module] = True
    except Exception:
        imports[module] = False

torch_details = {"available": False, "version": None, "cudaRuntime": None,
                 "cudaAvailable": False, "deviceName": None, "deviceCount": 0,
                 "deviceCapability": None}
neighbor_loader = None
if imports.get("torch"):
    import torch
    cuda_available = bool(torch.cuda.is_available())
    torch_details.update({
        "available": True,
        "version": str(torch.__version__),
        "cudaRuntime": torch.version.cuda,
        "cudaAvailable": cuda_available,
        "deviceName": torch.cuda.get_device_name(0) if cuda_available else None,
        "deviceCount": int(torch.cuda.device_count()) if cuda_available else 0,
        "deviceCapability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
    })
if request.get("neighborLoader") and imports.get("torch") and imports.get("torch_geometric"):
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        graph = Data(
            edge_index=torch.tensor([[0, 1, 2, 2], [1, 2, 0, 1]], dtype=torch.long),
            num_nodes=3,
        )
        loader = NeighborLoader(
            graph,
            input_nodes=torch.arange(3),
            num_neighbors=[2],
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        batch = next(iter(loader))
        neighbor_loader = bool(batch.num_nodes >= batch.batch_size > 0)
    except Exception as error:
        neighbor_loader = False

print(json.dumps({
    "schemaVersion": "socialgraph-fm.python-environment-probe/1.0",
    "executable": sys.executable,
    "implementation": platform.python_implementation(),
    "pythonVersion": platform.python_version(),
    "system": platform.system(),
    "machine": platform.machine(),
    "libc": platform.libc_ver()[0] or None,
    "libcVersion": platform.libc_ver()[1] or None,
    "versions": versions,
    "imports": imports,
    "torch": torch_details,
    "neighborLoader": neighbor_loader,
}, sort_keys=True))
"""


def clean_process_environment(*, python_path: str | None = None) -> dict[str, str]:
    """Return the environment used for every compatibility probe and Python child."""

    environment = dict(os.environ)
    for name in tuple(environment):
        upper = name.upper()
        if (
            is_ambient_llm_environment_name(upper)
            or upper.startswith("PIP_")
            or upper in {"PYTHONHOME", "PYTHONUSERBASE"}
        ):
            environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = "" if python_path is None else python_path
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def resolve_python(value: str | os.PathLike[str]) -> Path:
    # Do not call Path.resolve() here. POSIX virtual environments normally expose
    # bin/python as a symlink to the base interpreter; dereferencing it would make
    # pip and child processes run outside the selected venv.
    candidate = Path(os.path.abspath(Path(value).expanduser()))
    if candidate.is_file():
        return candidate
    located = shutil.which(str(value))
    if located:
        return Path(os.path.abspath(located))
    raise RuntimeError(f"Python executable does not exist: {value}")


def candidate_pythons(
    explicit: str | None,
    *,
    recorded: str | None = None,
    managed: Path | None = None,
) -> list[Path]:
    """Return only authorized reuse candidates, in fail-closed priority order.

    An arbitrary system/PATH interpreter is deliberately not a reuse candidate.
    The bootstrap interpreter is allowed to create a managed environment, but it is
    not silently adopted as the application runtime.
    """

    candidates: list[str | os.PathLike[str]] = []
    if explicit:
        candidates.append(explicit)
    if recorded:
        candidates.append(recorded)
    for variable in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(variable, "").strip()
        if root:
            candidates.append(environment_python(Path(root)))
    if managed is not None:
        candidates.append(environment_python(managed))
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            selected = resolve_python(candidate)
        except RuntimeError:
            continue
        key = os.path.normcase(str(selected))
        if key not in seen:
            seen.add(key)
            result.append(selected)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_clean_python(
    python: Path,
    arguments: Iterable[str],
    *,
    input_text: str | None = None,
    python_path: str | None = None,
    cwd: Path | None = None,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return run_captured_process(
        [str(python), *arguments],
        cwd=cwd,
        environment=clean_process_environment(python_path=python_path),
        timeout=timeout,
        input_text=input_text,
        description="Python subprocess",
    )


def probe_python(
    python: Path,
    distributions: dict[str, str],
    *,
    neighbor_loader: bool = False,
) -> dict[str, Any]:
    selected = resolve_python(python)
    request = json.dumps(
        {"distributions": distributions, "neighborLoader": neighbor_loader},
        sort_keys=True,
    )
    completed = run_clean_python(selected, ("-I", "-c", _PROBE_SOURCE), input_text=request)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"Python environment probe failed for {selected}: {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Python environment probe returned invalid JSON: {selected}") from error
    if report.get("schemaVersion") != PROBE_SCHEMA_VERSION:
        raise RuntimeError(f"Python environment probe returned an unsupported schema: {selected}")
    report["requestedExecutable"] = str(selected)
    report["executableSha256"] = _sha256(selected)
    return report


def pip_check(python: Path) -> tuple[bool, str]:
    completed = run_clean_python(python, ("-I", "-m", "pip", "--isolated", "check"))
    detail = (completed.stdout.strip() or completed.stderr.strip()).replace("\n", "; ")
    return completed.returncode == 0, detail


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if not match:
        return ()
    return tuple(int(part or 0) for part in match.groups())


def python_satisfies(version: str, requirement: str) -> bool:
    actual = _version_tuple(version)
    if not actual:
        return False
    for expression in requirement.split(","):
        expression = expression.strip()
        match = re.fullmatch(r"(>=|<=|==|>|<)\s*(\d+(?:\.\d+){0,2})(?:\.\*)?", expression)
        if not match:
            raise RuntimeError(f"Unsupported pythonRequires expression: {requirement}")
        operator, expected_text = match.groups()
        expected = _version_tuple(expected_text)
        width = max(len(actual), len(expected))
        left = actual + (0,) * (width - len(actual))
        right = expected + (0,) * (width - len(expected))
        comparisons = {
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
            "==": left[: len(expected)] == right[: len(expected)],
        }
        if not comparisons[operator]:
            return False
    return True


def distribution_satisfies(version: str, requirement: str) -> bool:
    """Evaluate the simple PEP 440 ranges used by this repository."""

    public = version.split("+", 1)[0]
    return python_satisfies(public, requirement)


def _fingerprint(report: dict[str, Any], capability: str) -> dict[str, Any]:
    torch_report = report["torch"]
    static_torch = {
        key: torch_report.get(key)
        for key in ("available", "version", "cudaRuntime")
    }
    identity = {
        "schemaVersion": FINGERPRINT_SCHEMA_VERSION,
        "capability": capability,
        "executable": report["executable"],
        "executableSha256": report["executableSha256"],
        "implementation": report["implementation"],
        "pythonVersion": report["pythonVersion"],
        "system": report["system"],
        "machine": report["machine"],
        "libc": report["libc"],
        "libcVersion": report["libcVersion"],
        "versions": report["versions"],
        "imports": report["imports"],
        "torch": static_torch,
        "neighborLoader": report["neighborLoader"],
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    identity["fingerprintSha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return identity


@dataclass(frozen=True)
class CompatibilityResult:
    compatible: bool
    errors: tuple[str, ...]
    fingerprint: dict[str, Any]
    runtime_capabilities: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeviceResolution:
    wheel_family: str
    device_policy: str
    resolved_device: str
    cuda_available: bool
    fallback_reason: str | None

    def to_document(self) -> dict[str, Any]:
        return {
            "wheelFamily": self.wheel_family,
            "devicePolicy": self.device_policy,
            "resolvedDevice": self.resolved_device,
            "cudaAvailable": self.cuda_available,
            "fallbackReason": self.fallback_reason,
        }


def install_profile_wheel_family(profile: dict[str, Any]) -> str:
    family = str(profile.get("wheelFamily", profile.get("device", ""))).lower()
    if family not in {"cpu", "cuda"}:
        raise RuntimeError(f"Install profile {profile.get('id')} has an invalid wheel family")
    return family


def resolve_execution_device(
    profile: dict[str, Any], result: CompatibilityResult, device_policy: str
) -> DeviceResolution:
    policy = device_policy.lower()
    if policy not in VALID_DEVICE_POLICIES:
        raise RuntimeError(f"Unsupported GFM device policy: {device_policy}")
    family = install_profile_wheel_family(profile)
    cuda_available = result.runtime_capabilities.get("cudaAvailable") is True
    if family == "cpu":
        if policy == "cuda-required":
            raise RuntimeError("The selected CPU wheel profile cannot require CUDA execution")
        return DeviceResolution(
            family,
            policy,
            "cpu",
            False,
            "cpu-wheel" if policy == "auto" else "policy-forced-cpu",
        )
    if policy == "cpu":
        return DeviceResolution(family, policy, "cpu", cuda_available, "policy-forced-cpu")
    if policy == "cuda-required" and not cuda_available:
        raise RuntimeError("CUDA execution is required but torch.cuda.is_available() is false")
    if cuda_available:
        return DeviceResolution(family, policy, "cuda", True, None)
    return DeviceResolution(family, policy, "cpu", False, "cuda-unavailable")


def _api_declared_requirements(api_root: Path) -> dict[str, str]:
    pyproject = api_root / "pyproject.toml"
    try:
        document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RuntimeError(f"API pyproject is invalid: {pyproject}") from error
    requirements: dict[str, str] = {}
    for raw in document.get("project", {}).get("dependencies", []):
        match = re.match(r"^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?\s*(.*)$", str(raw))
        if not match:
            raise RuntimeError(f"Unsupported API dependency declaration: {raw}")
        name, specifier = match.groups()
        requirements[name.lower().replace("_", "-")] = specifier.strip()
    return requirements


def probe_api_environment(python: Path, api_root: Path | None = None) -> CompatibilityResult:
    report = probe_python(
        python,
        {**API_DISTRIBUTIONS, "torch": "torch", "torch-geometric": "torch_geometric"},
    )
    errors: list[str] = []
    if not python_satisfies(report["pythonVersion"], ">=3.12,<3.13"):
        errors.append(f"Python {report['pythonVersion']} does not satisfy >=3.12,<3.13")
    for distribution, module in API_DISTRIBUTIONS.items():
        if report["versions"].get(distribution) is None:
            errors.append(f"missing {distribution}")
        if report["imports"].get(module) is not True:
            errors.append(f"cannot import {module} ({distribution})")
    if api_root is not None:
        declared = _api_declared_requirements(api_root)
        for distribution, requirement in declared.items():
            actual = report["versions"].get(distribution)
            if actual is not None and requirement and not distribution_satisfies(actual, requirement):
                errors.append(f"{distribution}={actual} does not satisfy {requirement}")
    for forbidden in ("torch", "torch-geometric"):
        if report["versions"].get(forbidden) is not None:
            errors.append(f"API environment contains forbidden {forbidden}")
    ready, detail = pip_check(python)
    if not ready:
        errors.append(f"pip check failed: {detail}")
    return CompatibilityResult(not errors, tuple(errors), _fingerprint(report, "api"))


def probe_bootstrap_environment(python: Path) -> CompatibilityResult:
    report = probe_python(python, {})
    errors: list[str] = []
    if not python_satisfies(report["pythonVersion"], ">=3.12,<3.13"):
        errors.append(f"Python {report['pythonVersion']} does not satisfy >=3.12,<3.13")
    return CompatibilityResult(not errors, tuple(errors), _fingerprint(report, "bootstrap"))


def probe_gfm_environment(python: Path, profile: dict[str, Any]) -> CompatibilityResult:
    versions = profile.get("distributionVersions")
    if not isinstance(versions, dict) or not versions:
        raise RuntimeError(f"Install profile {profile.get('id')} has no distributionVersions")
    distributions = {
        str(name): GFM_MODULES.get(str(name), str(name).replace("-", "_"))
        for name in versions
    }
    # These modules are operational requirements even if a profile forgot to list one.
    for distribution in ("torch", "torch-geometric", "pyg-lib", "numpy", "pydantic", "ogb"):
        distributions.setdefault(distribution, GFM_MODULES[distribution])
    report = probe_python(python, distributions, neighbor_loader=True)
    errors: list[str] = []
    system_aliases = {
        "windows": "windows",
        "linux": "linux",
        "darwin": "macos",
        "macos": "macos",
    }
    actual_system = system_aliases.get(str(report["system"]).lower(), str(report["system"]).lower())
    expected_system = system_aliases.get(
        str(profile.get("system", "")).lower(), str(profile.get("system", "")).lower()
    )
    actual_machine = str(report["machine"]).lower()
    actual_machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(
        actual_machine, actual_machine
    )
    expected_machine = str(profile.get("machine", "")).lower()
    if actual_system != expected_system:
        errors.append(f"platform system {actual_system} (required {expected_system})")
    if actual_machine != expected_machine:
        errors.append(f"platform machine {actual_machine} (required {expected_machine})")
    expected_libc = profile.get("libc")
    if expected_libc is not None and report.get("libc") != expected_libc:
        errors.append(
            f"platform libc {report.get('libc') or 'unknown'} (required {expected_libc})"
        )
    requirement = str(profile.get("pythonRequires", ">=3.12,<3.13"))
    if not python_satisfies(report["pythonVersion"], requirement):
        errors.append(f"Python {report['pythonVersion']} does not satisfy {requirement}")
    for distribution, expected in versions.items():
        actual = report["versions"].get(distribution)
        if actual != expected:
            errors.append(f"{distribution}={actual!s} (required {expected})")
    for required in ("torch", "torch-geometric", "pyg-lib", "numpy", "pydantic", "ogb"):
        if report["versions"].get(required) is None:
            errors.append(f"missing {required}")
        module = GFM_MODULES[required]
        if report["imports"].get(module) is not True:
            errors.append(f"cannot import {module} ({required})")
    if report.get("neighborLoader") is not True:
        errors.append("PyG NeighborLoader smoke failed")
    wheel_family = install_profile_wheel_family(profile)
    torch_report = report["torch"]
    expected_backend = str(profile.get("torchBackend", ""))
    runtime = str(torch_report.get("cudaRuntime") or "")
    if wheel_family == "cpu":
        if expected_backend != "cpu":
            errors.append(f"CPU wheel profile has unexpected backend {expected_backend}")
        if torch_report.get("cudaRuntime") is not None:
            errors.append(f"CPU wheel profile contains CUDA runtime {runtime}")
    else:
        if not expected_backend.startswith("cu"):
            errors.append(f"CUDA wheel profile has unexpected backend {expected_backend}")
        else:
            expected_runtime = f"{int(expected_backend[2:]) // 10}.{int(expected_backend[2:]) % 10}"
            if not runtime.startswith(expected_runtime):
                errors.append(
                    f"Compiled CUDA runtime {runtime or 'missing'} does not match {expected_backend}"
                )
    ready, detail = pip_check(python)
    if not ready:
        errors.append(f"pip check failed: {detail}")
    runtime_capabilities = {
        "wheelFamily": wheel_family,
        "torchBackend": expected_backend,
        "cudaRuntime": torch_report.get("cudaRuntime"),
        "cudaAvailable": torch_report.get("cudaAvailable") is True,
        "deviceName": torch_report.get("deviceName"),
        "deviceCount": torch_report.get("deviceCount", 0),
        "deviceCapability": torch_report.get("deviceCapability"),
    }
    return CompatibilityResult(
        not errors,
        tuple(errors),
        _fingerprint(report, "gfm"),
        runtime_capabilities,
    )


def interpreter_record(
    python: Path,
    *,
    source: str,
    result: CompatibilityResult,
) -> dict[str, Any]:
    return {
        "path": str(resolve_python(python)),
        "source": source,
        "fingerprint": result.fingerprint,
    }


def normalized_platform() -> tuple[str, str]:
    system = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}.get(
        platform.system(), platform.system().lower()
    )
    machine = platform.machine().lower()
    machine = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    return system, machine


def load_install_profiles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise RuntimeError(
            f"GFM install profile catalog is missing: {path}. "
            "Managed ML installation cannot select verified wheels."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GFM install profile catalog is invalid: {path}") from error
    if document.get("schemaVersion") != "socialgraph-fm.gfm-install-profiles/1.0":
        raise RuntimeError(f"GFM install profile catalog schema is unsupported: {path}")
    manifest_path = path.parent / "locks" / "install-lock-manifest.json"
    try:
        lock_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GFM install lock manifest is invalid: {manifest_path}") from error
    if lock_manifest.get("schemaVersion") != "socialgraph-fm.gfm-install-lock-manifest/1.0":
        raise RuntimeError(f"GFM install lock manifest schema is unsupported: {manifest_path}")
    if (
        lock_manifest.get("profilesFile") != path.name
        or lock_manifest.get("profilesSha256") != _sha256(path)
    ):
        raise RuntimeError("GFM install profile catalog does not match its lock manifest")
    build_requirements = lock_manifest.get("buildRequirementsFile")
    if not isinstance(build_requirements, str):
        raise RuntimeError("GFM install lock manifest has no build requirements source")
    build_path = (path.parent / build_requirements).resolve()
    try:
        build_path.relative_to(path.parent.resolve())
    except ValueError as error:
        raise RuntimeError("GFM build requirements path escapes packages/gfm") from error
    if (
        not build_path.is_file()
        or lock_manifest.get("buildRequirementsSha256") != _sha256(build_path)
    ):
        raise RuntimeError("GFM build requirements source failed integrity validation")
    raw = document.get("profiles")
    profiles: dict[str, dict[str, Any]] = {}
    entries: Iterable[tuple[str, Any]]
    if isinstance(raw, dict):
        entries = ((str(identifier), value) for identifier, value in raw.items())
    elif isinstance(raw, list):
        entries = ((str(entry.get("id", "")), entry) for entry in raw if isinstance(entry, dict))
    else:
        raise RuntimeError(f"GFM install profile catalog has no profiles: {path}")
    for identifier, value in entries:
        if not identifier or not isinstance(value, dict):
            raise RuntimeError(f"GFM install profile entry is invalid: {identifier!r}")
        profile = dict(value)
        profile.setdefault("id", identifier)
        required = {
            "system",
            "machine",
            "device",
            "wheelFamily",
            "pythonRequires",
            "torchBackend",
            "indexUrl",
            "torchIndexUrl",
            "findLinks",
            "distributionVersions",
            "requirementsLock",
        }
        missing = sorted(required - profile.keys())
        if missing:
            raise RuntimeError(f"Install profile {identifier} is missing: {', '.join(missing)}")
        if profile["wheelFamily"] not in {"cpu", "cuda"} or (
            profile["device"] != profile["wheelFamily"]
        ):
            raise RuntimeError(
                f"Install profile {identifier} wheel family is inconsistent"
            )
        torch_index = str(profile["torchIndexUrl"])
        find_links = profile["findLinks"]
        if (
            profile["indexUrl"] != "https://pypi.org/simple"
            or torch_index
            not in {
                "https://pypi.org/simple",
                "https://download.pytorch.org/whl/cpu",
                "https://download.pytorch.org/whl/cu130",
            }
            or not isinstance(find_links, list)
            or not find_links
            or any(
                not isinstance(link, str)
                or not link.startswith("https://data.pyg.org/whl/")
                for link in find_links
            )
        ):
            raise RuntimeError(f"Install profile {identifier} uses an unapproved wheel source")
        manifest_profile = lock_manifest.get("profiles", {}).get(identifier)
        if not isinstance(manifest_profile, dict):
            raise RuntimeError(f"Install profile {identifier} is absent from the lock manifest")
        if manifest_profile.get("requirementsLock") != profile["requirementsLock"]:
            raise RuntimeError(f"Install profile {identifier} lock path does not match its manifest")
        lock_path = (path.parent / str(profile["requirementsLock"])).resolve()
        try:
            lock_path.relative_to(path.parent.resolve())
        except ValueError as error:
            raise RuntimeError(f"Install profile {identifier} lock escapes packages/gfm") from error
        if (
            not lock_path.is_file()
            or manifest_profile.get("requirementsLockSha256") != _sha256(lock_path)
            or manifest_profile.get("artifactHashesResolved") is not True
        ):
            raise RuntimeError(f"Install profile {identifier} lock integrity validation failed")
        profile["requirementsLockSha256"] = manifest_profile["requirementsLockSha256"]
        profiles[identifier] = profile
    manifest_profiles = lock_manifest.get("profiles")
    if not isinstance(manifest_profiles, dict) or set(manifest_profiles) != set(profiles):
        raise RuntimeError("GFM install profile and lock manifest inventories differ")
    policy = lock_manifest.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("sourceBuildsAllowed") is not False
        or policy.get("requireArtifactHashesForManagedInstall") is not True
    ):
        raise RuntimeError("GFM install lock manifest does not prohibit source builds")
    return profiles


def select_install_profile(
    profiles: dict[str, dict[str, Any]], wheel_selection: str
) -> dict[str, Any]:
    system, machine = normalized_platform()
    libc_name = platform.libc_ver()[0] or None
    system_aliases = {"windows": "windows", "linux": "linux", "darwin": "macos", "macos": "macos"}
    normalized_selection = wheel_selection.lower()
    exact = profiles.get(normalized_selection)
    if exact is not None:
        matches_platform = (
            system_aliases.get(
                str(exact["system"]).lower(), str(exact["system"]).lower()
            )
            == system
            and str(exact["machine"]).lower() == machine
            and (exact.get("libc") is None or exact.get("libc") == libc_name)
        )
        if not matches_platform:
            raise RuntimeError(
                f"Install profile {normalized_selection} does not match {system}/{machine}"
            )
        return exact
    if normalized_selection not in {"cpu", "cuda"}:
        available = ", ".join(sorted(profiles))
        raise RuntimeError(
            f"Unknown GFM wheel profile {wheel_selection!r}; available: {available}"
        )
    matches = [
        value
        for value in profiles.values()
        if system_aliases.get(str(value["system"]).lower(), str(value["system"]).lower()) == system
        and str(value["machine"]).lower() == machine
        and install_profile_wheel_family(value) == normalized_selection
        and (value.get("libc") is None or value.get("libc") == libc_name)
    ]
    if len(matches) != 1:
        available = ", ".join(sorted(profiles))
        raise RuntimeError(
            f"No unique install profile for {system}/{machine}/{normalized_selection}; "
            f"available: {available}"
        )
    return matches[0]


def ensure_bootstrap_python(
    value: str | None, requirement: str = ">=3.12,<3.13"
) -> Path:
    selected = resolve_python(value or sys.executable)
    report = probe_python(selected, {})
    if not python_satisfies(report["pythonVersion"], requirement):
        raise RuntimeError(
            f"Bootstrap Python {report['pythonVersion']} does not satisfy {requirement}: {selected}"
        )
    return selected


def create_venv(bootstrap_python: Path, destination: Path) -> Path:
    python = environment_python(destination)
    if not python.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        completed = run_clean_python(
            bootstrap_python,
            ("-I", "-m", "venv", str(destination)),
            timeout=300,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"Could not create Python environment {destination}: {detail}")
    return resolve_python(python)


def _pip_install(
    python: Path,
    arguments: Iterable[str],
    *,
    cwd: Path,
    timeout: int = 1800,
    logger: Callable[[str], None] | None = None,
) -> None:
    selected_arguments = tuple(arguments)
    if logger is not None:
        logger(f"pip install started: {python}")
    run_streaming_process(
        [
            str(python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            *selected_arguments,
        ],
        cwd=cwd,
        environment=clean_process_environment(),
        timeout=timeout,
        logger=logger,
        description=f"pip installation for {python}",
    )


def install_api_environment(
    layout: RuntimeLayout,
    bootstrap_python: Path,
    *,
    destination: Path,
    logger: Callable[[str], None] | None = None,
) -> Path:
    python = create_venv(bootstrap_python, destination)
    lock = layout.api_root / "requirements.lock"
    if not lock.is_file():
        raise RuntimeError(f"API runtime lock is missing: {lock}")
    _pip_install(
        python,
        (
            "--require-hashes",
            "--only-binary=:all:",
            "--index-url",
            "https://pypi.org/simple",
            "-r",
            str(lock),
        ),
        cwd=layout.api_root,
        logger=logger,
    )
    return python


def install_gfm_environment(
    layout: RuntimeLayout,
    bootstrap_python: Path,
    profile: dict[str, Any],
    *,
    destination: Path | None = None,
    logger: Callable[[str], None] | None = None,
) -> Path:
    requirement = str(profile["pythonRequires"])
    bootstrap_report = probe_python(bootstrap_python, {})
    if not python_satisfies(bootstrap_report["pythonVersion"], requirement):
        raise RuntimeError(
            f"Managed GFM profile requires Python {requirement}; bootstrap is "
            f"{bootstrap_report['pythonVersion']}"
        )
    python = create_venv(
        bootstrap_python,
        destination or layout.gfm_environment(str(profile["id"])),
    )
    lock = (layout.gfm_package / str(profile["requirementsLock"])).resolve()
    try:
        lock.relative_to(layout.gfm_package.resolve())
    except ValueError as error:
        raise RuntimeError(f"Install profile lock escapes packages/gfm: {lock}") from error
    if not lock.is_file():
        raise RuntimeError(f"GFM runtime lock is missing: {lock}")
    expected_lock_hash = profile.get("requirementsLockSha256")
    if not isinstance(expected_lock_hash, str) or _sha256(lock) != expected_lock_hash:
        raise RuntimeError(f"GFM runtime lock failed integrity validation: {lock}")
    arguments: list[str] = [
        "--require-hashes",
        "--only-binary=:all:",
        "--index-url",
        str(profile["indexUrl"]),
        "--extra-index-url",
        str(profile["torchIndexUrl"]),
    ]
    for link in profile.get("findLinks", []):
        arguments.extend(("--find-links", str(link)))
    arguments.extend(("-r", str(lock)))
    _pip_install(python, arguments, cwd=layout.gfm_package, logger=logger)
    _pip_install(
        python,
        ("--no-deps", "--no-build-isolation", str(layout.gfm_package)),
        cwd=layout.gfm_package,
        logger=logger,
    )
    return python
