"""CPU-only managed runtime discovery, verification, and installation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .layout import RuntimeLayout, environment_python
from .subprocess_control import run_captured_process, run_streaming_process


PROBE_SCHEMA_VERSION = "socialgraph-fm.python-environment-probe/2.0"
FINGERPRINT_SCHEMA_VERSION = "socialgraph-fm.python-environment-fingerprint/3.0"

# Parent-shell credentials are never an implicit configuration source. Only the
# three values loaded from the private configuration file enter the API child.
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

RUNTIME_MODULES = {
    "fastapi": "fastapi",
    "httpx": "httpx",
    "networkx": "networkx",
    "numpy": "numpy",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "pyg-lib": "pyg_lib",
    "python-multipart": "multipart",
    "torch": "torch",
    "torch-geometric": "torch_geometric",
    "uvicorn": "uvicorn",
}

_PROBE_SOURCE = r"""
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

torch_details = {"available": False, "version": None, "cpuBuild": None}
neighbor_loader = None
if imports.get("torch"):
    import torch
    torch_details = {
        "available": True,
        "version": str(torch.__version__),
        "cpuBuild": torch.version.cuda is None,
    }
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
    except Exception:
        neighbor_loader = False

print(json.dumps({
    "schemaVersion": "socialgraph-fm.python-environment-probe/2.0",
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


def is_ambient_llm_environment_name(name: str) -> bool:
    upper = name.upper()
    return upper.startswith("LLM_") or upper in AMBIENT_LLM_ENVIRONMENT_NAMES


def clean_process_environment(*, python_path: str | None = None) -> dict[str, str]:
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
    # POSIX venv launchers are symlinks. Do not resolve them outside the venv.
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
    """Compatibility helper: only explicitly authorized paths are candidates."""

    candidates: list[str | os.PathLike[str]] = []
    if explicit:
        candidates.append(explicit)
    if recorded:
        candidates.append(recorded)
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
    selected_arguments = tuple(arguments)
    if selected_arguments[:1] != ("-B",):
        selected_arguments = ("-B", *selected_arguments)
    return run_captured_process(
        [str(python), *selected_arguments],
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
    return python_satisfies(version.split("+", 1)[0], requirement)


def _fingerprint(report: dict[str, Any]) -> dict[str, Any]:
    identity = {
        "schemaVersion": FINGERPRINT_SCHEMA_VERSION,
        "capability": "runtime",
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
        "torch": report["torch"],
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


def probe_bootstrap_environment(python: Path) -> CompatibilityResult:
    report = probe_python(python, {})
    errors = (
        ()
        if python_satisfies(report["pythonVersion"], ">=3.12,<3.13")
        else (f"Python {report['pythonVersion']} does not satisfy >=3.12,<3.13",)
    )
    fingerprint = _fingerprint(report)
    fingerprint["capability"] = "bootstrap"
    return CompatibilityResult(not errors, errors, fingerprint)


def probe_runtime_environment(
    python: Path, profile: dict[str, Any]
) -> CompatibilityResult:
    versions = profile.get("distributionVersions")
    if not isinstance(versions, dict) or not versions:
        raise RuntimeError(f"Install profile {profile.get('id')} has no distributionVersions")
    distributions = {
        str(name): RUNTIME_MODULES.get(str(name), str(name).replace("-", "_"))
        for name in versions
    }
    report = probe_python(python, distributions, neighbor_loader=True)
    errors: list[str] = []
    actual_system, actual_machine = normalized_platform(
        system=str(report["system"]), machine=str(report["machine"])
    )
    expected_system = str(profile["system"]).lower()
    expected_machine = str(profile["machine"]).lower()
    if actual_system != expected_system:
        errors.append(f"platform system {actual_system} (required {expected_system})")
    if actual_machine != expected_machine:
        errors.append(f"platform machine {actual_machine} (required {expected_machine})")
    if profile.get("libc") is not None and report.get("libc") != profile["libc"]:
        errors.append(
            f"platform libc {report.get('libc') or 'unknown'} (required {profile['libc']})"
        )
    requirement = str(profile.get("pythonRequires", ">=3.12,<3.13"))
    if not python_satisfies(report["pythonVersion"], requirement):
        errors.append(f"Python {report['pythonVersion']} does not satisfy {requirement}")
    for distribution, expected in versions.items():
        actual = report["versions"].get(distribution)
        if actual != expected:
            errors.append(f"{distribution}={actual!s} (required {expected})")
        module = distributions[distribution]
        if report["imports"].get(module) is not True:
            errors.append(f"cannot import {module} ({distribution})")
    if report["torch"].get("cpuBuild") is not True:
        errors.append("Torch is not the verified CPU build")
    if report.get("neighborLoader") is not True:
        errors.append("PyG NeighborLoader smoke failed")
    ready, detail = pip_check(python)
    if not ready:
        errors.append(f"pip check failed: {detail}")
    return CompatibilityResult(not errors, tuple(errors), _fingerprint(report))


def interpreter_record(
    python: Path, *, source: str, result: CompatibilityResult
) -> dict[str, Any]:
    return {
        "path": str(resolve_python(python)),
        "source": source,
        "fingerprint": result.fingerprint,
    }


def normalized_platform(
    *, system: str | None = None, machine: str | None = None
) -> tuple[str, str]:
    selected_system = system or platform.system()
    normalized_system = {
        "Windows": "windows",
        "Linux": "linux",
    }.get(selected_system, selected_system.lower())
    selected_machine = (machine or platform.machine()).lower()
    normalized_machine = {
        "amd64": "x86_64",
        "x64": "x86_64",
    }.get(selected_machine, selected_machine)
    return normalized_system, normalized_machine


def load_install_profiles(path: Path) -> dict[str, dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / "locks" / "install-lock-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("CPU runtime profile catalog or lock manifest is invalid") from error
    if document.get("schemaVersion") != "socialgraph-fm.gfm-install-profiles/2.0":
        raise RuntimeError("CPU runtime profile catalog schema is unsupported")
    if manifest.get("schemaVersion") != "socialgraph-fm.gfm-install-lock-manifest/2.0":
        raise RuntimeError("CPU runtime lock manifest schema is unsupported")
    if manifest.get("profilesFile") != path.name or manifest.get("profilesSha256") != _sha256(path):
        raise RuntimeError("CPU runtime profile catalog does not match its lock manifest")
    source_candidate = path.parent / str(manifest.get("runtimeRequirementsFile", ""))
    source = source_candidate.resolve()
    try:
        source.relative_to(path.parent.resolve())
    except ValueError as error:
        raise RuntimeError("CPU runtime requirements source escapes packages/gfm") from error
    if (
        source_candidate.is_symlink()
        or not source.is_file()
        or manifest.get("runtimeRequirementsSha256") != _sha256(source)
    ):
        raise RuntimeError("CPU runtime requirements source failed integrity validation")
    policy = manifest.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("sourceBuildsAllowed") is not False
        or policy.get("hashesRequired") is not True
        or policy.get("runtimeEnvironmentCount") != 1
        or policy.get("supportedDevices") != ["cpu"]
        or policy.get("supportedSystems") != ["Windows", "Linux"]
    ):
        raise RuntimeError("CPU runtime lock policy is invalid")
    raw = document.get("profiles")
    manifest_profiles = manifest.get("profiles")
    if not isinstance(raw, dict) or not isinstance(manifest_profiles, dict):
        raise RuntimeError("CPU runtime profile inventories are invalid")
    profiles: dict[str, dict[str, Any]] = {}
    for identifier, value in raw.items():
        if not isinstance(identifier, str) or not isinstance(value, dict):
            raise RuntimeError("CPU runtime profile entry is invalid")
        profile = dict(value)
        profile["id"] = identifier
        required = {
            "system",
            "machine",
            "pythonRequires",
            "indexUrl",
            "torchIndexUrl",
            "findLinks",
            "distributionVersions",
            "requirementsLock",
        }
        missing = sorted(required - profile.keys())
        if missing:
            raise RuntimeError(f"Install profile {identifier} is missing: {', '.join(missing)}")
        if (
            profile["indexUrl"] != "https://pypi.org/simple"
            or profile["torchIndexUrl"] != "https://download.pytorch.org/whl/cpu"
            or not isinstance(profile["findLinks"], list)
            or profile["findLinks"] != ["https://data.pyg.org/whl/torch-2.8.0+cpu.html"]
        ):
            raise RuntimeError(f"Install profile {identifier} uses an unapproved wheel source")
        manifest_profile = manifest_profiles.get(identifier)
        if not isinstance(manifest_profile, dict):
            raise RuntimeError(f"Install profile {identifier} is absent from the lock manifest")
        lock_candidate = path.parent / str(profile["requirementsLock"])
        lock = lock_candidate.resolve()
        try:
            lock.relative_to(path.parent.resolve())
        except ValueError as error:
            raise RuntimeError(f"Install profile {identifier} lock escapes packages/gfm") from error
        if (
            manifest_profile.get("requirementsLock") != profile["requirementsLock"]
            or lock_candidate.is_symlink()
            or not lock.is_file()
            or manifest_profile.get("requirementsLockSha256") != _sha256(lock)
            or manifest_profile.get("artifactHashesResolved") is not True
        ):
            raise RuntimeError(f"Install profile {identifier} lock integrity validation failed")
        profile["requirementsLockSha256"] = manifest_profile["requirementsLockSha256"]
        profiles[identifier] = profile
    if set(profiles) != set(manifest_profiles):
        raise RuntimeError("CPU runtime profile and lock manifest inventories differ")
    return profiles


def select_install_profile(
    profiles: dict[str, dict[str, Any]], _selection: str | None = None
) -> dict[str, Any]:
    system, machine = normalized_platform()
    if system not in {"windows", "linux"} or machine != "x86_64":
        raise RuntimeError(
            f"SocialGraph-FM supports only Windows x64 and Ubuntu glibc x64: {system}/{machine}"
        )
    libc_name = platform.libc_ver()[0] or None
    matches = [
        profile
        for profile in profiles.values()
        if str(profile["system"]).lower() == system
        and str(profile["machine"]).lower() == machine
        and (profile.get("libc") is None or profile.get("libc") == libc_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"No verified CPU runtime lock for {system}/{machine}")
    return matches[0]


def ensure_bootstrap_python(requirement: str = ">=3.12,<3.13") -> Path:
    selected = resolve_python(sys.executable)
    report = probe_python(selected, {})
    if not python_satisfies(report["pythonVersion"], requirement):
        raise RuntimeError(
            f"Python {report['pythonVersion']} does not satisfy {requirement}: {selected}"
        )
    return selected


def create_venv(bootstrap_python: Path, destination: Path) -> Path:
    python = environment_python(destination)
    if python.is_file():
        raise RuntimeError(f"Refusing to install over an existing environment: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = run_clean_python(
        bootstrap_python, ("-I", "-m", "venv", str(destination)), timeout=300
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
    logger: Callable[[str], None] | None = None,
) -> None:
    if logger is not None:
        logger(f"CPU runtime dependency installation started: {python}")
    run_streaming_process(
        [
            str(python),
            "-B",
            "-I",
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-compile",
            "--no-cache-dir",
            *arguments,
        ],
        cwd=cwd,
        environment=clean_process_environment(),
        timeout=1800,
        logger=logger,
        description=f"CPU runtime installation for {python}",
    )


def _purelib(python: Path) -> Path:
    completed = run_clean_python(
        python,
        ("-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"),
    )
    if completed.returncode != 0:
        raise RuntimeError("Could not locate the managed runtime site-packages")
    return Path(completed.stdout.strip())


@dataclass(frozen=True)
class BytecodePruneResult:
    """Private setup statistics for safely removable Python bytecode."""

    removed_files: int
    removed_bytes: int


def _is_link_or_reparse_point(path: Path) -> bool:
    """Inspect a path itself without following a link or Windows reparse point."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise RuntimeError(f"Could not inspect managed runtime path: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)


def _safe_unlinked_member(path: Path, root: Path, resolved_root: Path) -> bool:
    """Return whether an existing path is contained without a linked component."""

    absolute = Path(os.path.abspath(path))
    try:
        absolute.relative_to(root)
    except ValueError:
        return False
    current = absolute
    while True:
        if _is_link_or_reparse_point(current):
            return False
        if current == root:
            break
        parent = current.parent
        if parent == current:
            return False
        current = parent
    try:
        absolute.resolve(strict=True).relative_to(resolved_root)
    except (OSError, ValueError):
        return False
    return True


def _bytecode_source(
    cache: Path, *, purelib: Path, resolved_purelib: Path
) -> Path | None:
    if not _safe_unlinked_member(cache, purelib, resolved_purelib):
        return None
    try:
        source = Path(importlib.util.source_from_cache(str(cache)))
    except (TypeError, ValueError):
        return None
    if source.suffix.lower() != ".py" or not source.is_file():
        return None
    if not _safe_unlinked_member(source, purelib, resolved_purelib):
        return None
    return source


def prune_runtime_bytecode(python: Path) -> BytecodePruneResult:
    """Remove only source-backed ``__pycache__/*.pyc`` inside managed purelib.

    The complete candidate inventory is validated before the first deletion. Linked,
    reparse-point, sourceless, malformed, and escaped candidates are retained.
    """

    environment_root = Path(os.path.abspath(Path(python).parent.parent))
    purelib = Path(os.path.abspath(_purelib(python)))
    if not environment_root.is_dir() or not purelib.is_dir():
        raise RuntimeError("Managed runtime or site-packages is missing")
    try:
        purelib.relative_to(environment_root)
        resolved_environment = environment_root.resolve(strict=True)
        resolved_purelib = purelib.resolve(strict=True)
        resolved_purelib.relative_to(resolved_environment)
    except (OSError, ValueError) as error:
        raise RuntimeError("Managed runtime site-packages escapes its environment") from error
    if not _safe_unlinked_member(purelib, environment_root, resolved_environment):
        raise RuntimeError("Managed runtime site-packages cannot contain linked path components")

    candidates: list[Path] = []
    pending = [purelib]
    while pending:
        directory = pending.pop()
        try:
            children = tuple(directory.iterdir())
        except OSError as error:
            raise RuntimeError(f"Could not inspect managed runtime directory: {directory}") from error
        for child in children:
            if _is_link_or_reparse_point(child):
                continue
            if child.is_dir():
                pending.append(child)
            elif (
                directory.name == "__pycache__"
                and child.suffix.lower() == ".pyc"
                and child.is_file()
                and _bytecode_source(
                    child, purelib=purelib, resolved_purelib=resolved_purelib
                )
                is not None
            ):
                candidates.append(child)

    removed_files = 0
    removed_bytes = 0
    for candidate in sorted(candidates):
        # Revalidate immediately before unlinking to fail closed if the tree changed.
        if (
            not candidate.is_file()
            or _bytecode_source(
                candidate, purelib=purelib, resolved_purelib=resolved_purelib
            )
            is None
        ):
            continue
        try:
            size = candidate.lstat().st_size
            candidate.unlink()
        except OSError as error:
            raise RuntimeError(f"Could not remove managed runtime bytecode: {candidate}") from error
        removed_files += 1
        removed_bytes += size
    return BytecodePruneResult(removed_files=removed_files, removed_bytes=removed_bytes)


def prune_torch_build_assets(python: Path) -> tuple[str, ...]:
    """Remove version-pinned compiler-only Torch payload after installation."""

    report = probe_python(python, {"torch": "torch"})
    if report["versions"].get("torch") != "2.8.0+cpu" or report["torch"].get("cpuBuild") is not True:
        raise RuntimeError("Torch build-asset pruning requires the verified 2.8.0 CPU wheel")
    environment_root = Path(python).parent.parent
    site_packages = _purelib(python)
    resolved_environment = environment_root.resolve()
    resolved_site_packages = site_packages.resolve()
    try:
        resolved_site_packages.relative_to(resolved_environment)
    except ValueError as error:
        raise RuntimeError("Torch build assets are outside the managed runtime") from error
    if site_packages.is_symlink():
        raise RuntimeError("Managed runtime site-packages cannot be a link")
    torch_root = site_packages / "torch"
    if torch_root.is_symlink() or torch_root.resolve() != resolved_site_packages / "torch":
        raise RuntimeError("Managed Torch package cannot be a link or reparse point")
    targets = [torch_root / "include", torch_root / "share" / "cmake"]
    suffix = "*.lib" if os.name == "nt" else "*.a"
    targets.extend((torch_root / "lib").glob(suffix))
    removed: list[str] = []
    for target in targets:
        if not target.exists():
            continue
        try:
            target.relative_to(torch_root)
        except ValueError as error:
            raise RuntimeError("Unsafe Torch build-asset path") from error
        relative = target.relative_to(site_packages).as_posix()
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append(relative)
    return tuple(sorted(removed))


def install_runtime_environment(
    layout: RuntimeLayout,
    bootstrap_python: Path,
    profile: dict[str, Any],
    *,
    destination: Path,
    logger: Callable[[str], None] | None = None,
) -> Path:
    requirement = str(profile["pythonRequires"])
    bootstrap_report = probe_python(bootstrap_python, {})
    if not python_satisfies(bootstrap_report["pythonVersion"], requirement):
        raise RuntimeError(
            f"Managed runtime requires Python {requirement}; bootstrap is "
            f"{bootstrap_report['pythonVersion']}"
        )
    python = create_venv(bootstrap_python, destination)
    lock = (layout.gfm_package / str(profile["requirementsLock"])).resolve()
    try:
        lock.relative_to(layout.gfm_package.resolve())
    except ValueError as error:
        raise RuntimeError(f"CPU runtime lock escapes packages/gfm: {lock}") from error
    if not lock.is_file() or _sha256(lock) != profile.get("requirementsLockSha256"):
        raise RuntimeError(f"CPU runtime lock failed integrity validation: {lock}")
    arguments: list[str] = [
        "--require-hashes",
        "--only-binary=:all:",
        "--index-url",
        str(profile["indexUrl"]),
        "--extra-index-url",
        str(profile["torchIndexUrl"]),
    ]
    for link in profile["findLinks"]:
        arguments.extend(("--find-links", str(link)))
    arguments.extend(("-r", str(lock)))
    _pip_install(python, arguments, cwd=layout.gfm_package, logger=logger)
    removed = prune_torch_build_assets(python)
    if logger is not None:
        logger(f"Removed {len(removed)} compiler-only Torch assets")
    return python
