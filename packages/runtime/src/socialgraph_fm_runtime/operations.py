"""Setup, lifecycle, and diagnostic operations behind the public CLI."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .bundle import (
    file_sha256,
    install_public_runtime_bundle,
    load_and_verify_bundle,
    materialize_target_examples,
    run_checkpoint_forward_probe,
    run_full_gfm_probe,
    verify_gfm_runtime_state,
    verify_installed_runtime_bundle,
    verify_installed_runtime_seeds,
    verify_target_examples,
)
from .environment import (
    CompatibilityResult,
    DeviceResolution,
    candidate_pythons,
    clean_process_environment,
    ensure_bootstrap_python,
    install_api_environment,
    install_gfm_environment,
    install_profile_wheel_family,
    interpreter_record,
    is_ambient_llm_environment_name,
    load_install_profiles,
    normalized_platform,
    probe_api_environment,
    probe_bootstrap_environment,
    probe_gfm_environment,
    resolve_python,
    resolve_execution_device,
    select_install_profile,
)
from .layout import RuntimeLayout
from .llm import (
    configuration_state,
    migrate_private_environment_permissions,
    parse_private_environment,
    write_private_environment,
)
from .processes import ProcessManager, ServiceSpec
from .profile import RuntimeProfile
from .subprocess_control import (
    redact_subprocess_text,
    run_captured_process,
    run_streaming_process,
)


def _port(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if not 1 <= value <= 65535:
        raise RuntimeError(f"{name} must be between 1 and 65535")
    return value


@dataclass(frozen=True)
class Ports:
    web: int
    api: int
    gfm: int

    @classmethod
    def environment(cls) -> "Ports":
        return cls(
            web=_port("SOCIALGRAPH_GOVERNANCE_WEB_PORT", 5173),
            api=_port("SOCIALGRAPH_CORE_API_PORT", 8000),
            gfm=_port("SOCIALGRAPH_GFM_PORT", 8766),
        )


@dataclass(frozen=True)
class SetupOptions:
    profile: str = "cpu"
    env_mode: str = "auto"
    api_python: str | None = None
    gfm_python: str | None = None
    bootstrap_python: str | None = None
    skip_api: bool = False
    skip_web: bool = False
    gfm_text_profile: bool = False
    full_probe: bool = True
    device_policy: str = "auto"
    after_api: Callable[[Path], str | None] | None = None


class _SetupReporter:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.path = layout.setup_log_file
        layout.assert_safe_var_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        layout.assert_safe_var_path(self.path.parent)
        self.path.write_text("SocialGraph-FM setup log\n", encoding="utf-8")

    @staticmethod
    def _redact(message: str) -> str:
        return redact_subprocess_text(message)

    def log(self, message: str) -> None:
        selected = self._redact(message).replace("\r\n", "\n")
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(selected.rstrip("\n") + "\n")

    def stage(self, message: str) -> None:
        selected = self._redact(message)
        self.log(f"[setup] {selected}")
        print(f"[setup] {selected}", file=sys.stderr, flush=True)

    def progress(self, message: str) -> None:
        selected = self._redact(message)
        self.log(selected)
        print(f"[setup] {selected}", file=sys.stderr, flush=True)


def _tool_version(*names: str) -> dict[str, Any]:
    executable = next((shutil.which(name) for name in names if shutil.which(name)), None)
    if executable is None:
        return {"available": False, "path": None, "version": None}
    try:
        completed = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            check=False,
            env=clean_process_environment(),
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "path": executable,
            "version": None,
            "error": str(error),
        }
    output = (completed.stdout.strip() or completed.stderr.strip()).splitlines()
    version = output[0].strip() if completed.returncode == 0 and output else None
    return {
        "available": version is not None,
        "path": executable,
        "version": version,
        "exitCode": completed.returncode,
    }


def _gpu_driver() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "path": None, "driverVersion": None, "devices": []}
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=driver_version,name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=clean_process_environment(),
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "path": executable,
            "driverVersion": None,
            "devices": [],
            "error": str(error),
        }
    devices = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    first = devices[0].split(",", 1)[0].strip() if devices else None
    return {
        "available": completed.returncode == 0 and bool(devices),
        "path": executable,
        "driverVersion": first,
        "devices": devices,
        "exitCode": completed.returncode,
    }


def _host_diagnostics() -> dict[str, Any]:
    system, machine = normalized_platform()
    libc_name, libc_version = platform.libc_ver()
    return {
        "system": system,
        "machine": machine,
        "libc": libc_name or None,
        "libcVersion": libc_version or None,
        "node": _tool_version("node"),
        "npm": _tool_version("npm.cmd" if os.name == "nt" else "npm", "npm"),
        "gpuDriver": _gpu_driver(),
    }


def _major_version(value: Any) -> int | None:
    match = re.search(r"(?:^|\s)v?(\d+)(?:\.\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def _require_host_compatibility(
    diagnostics: dict[str, Any], *, web: bool, cuda: bool
) -> None:
    if web:
        node = diagnostics["node"]
        npm = diagnostics["npm"]
        if not node.get("available") or _major_version(node.get("version")) != 24:
            raise RuntimeError(
                f"Node.js 24.x is required; detected {node.get('version') or 'missing'}"
            )
        if not npm.get("available") or _major_version(npm.get("version")) != 11:
            raise RuntimeError(f"npm 11.x is required; detected {npm.get('version') or 'missing'}")
    if cuda and not diagnostics["gpuDriver"].get("available"):
        raise RuntimeError("A working NVIDIA driver (nvidia-smi) is required for the CUDA profile")


def _generation_key(
    capability: str,
    *,
    bootstrap: CompatibilityResult,
    descriptor: dict[str, Any],
    assets: tuple[Path, ...],
) -> str:
    identity = {
        "schemaVersion": "socialgraph-fm.managed-environment-generation/1.0",
        "capability": capability,
        "bootstrap": {
            key: bootstrap.fingerprint.get(key)
            for key in ("pythonVersion", "system", "machine", "libc", "libcVersion")
        },
        "descriptor": descriptor,
        "assets": [
            {"name": path.name, "sha256": file_sha256(path)} for path in assets
        ],
    }
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]


def _new_generation_root(
    layout: RuntimeLayout, capability: str, generation_key: str
) -> Path:
    for sequence in range(1, 10_000):
        generation = generation_key if sequence == 1 else f"{generation_key}-{sequence}"
        candidate = layout.managed_environment(capability, generation)
        layout.assert_safe_var_path(candidate)
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a new managed {capability} environment generation")


def _environment_root(python: str | Path) -> Path:
    return Path(os.path.abspath(Path(python))).parent.parent


def _is_repo_managed_python(layout: RuntimeLayout, python: str | Path) -> bool:
    root = _environment_root(python)
    candidates = (
        (layout.managed_environment_root, {"a", "g"}),
        (layout.legacy_managed_environment_root, {"api", "gfm"}),
    )
    for managed_root, namespaces in candidates:
        try:
            relative = root.relative_to(Path(os.path.abspath(managed_root)))
        except ValueError:
            continue
        return len(relative.parts) == 2 and relative.parts[0] in namespaces
    return False


def _recorded_managed_candidate(
    layout: RuntimeLayout, recorded: RuntimeProfile | None, capability: str
) -> Path | None:
    if recorded is None:
        return None
    record = recorded.interpreters.get(capability)
    if not isinstance(record, dict) or record.get("source") != "managed":
        return None
    raw = record.get("path")
    if not isinstance(raw, str) or not _is_repo_managed_python(layout, raw):
        return None
    candidate = Path(raw)
    return resolve_python(candidate) if candidate.is_file() else None


def _managed_services_stopped(layout: RuntimeLayout) -> bool:
    manager = ProcessManager(layout.pid_root, layout.log_root)
    try:
        snapshots = tuple(
            manager.snapshot(name) for name in ("governance-web", "socialgraph-api", "gfm")
        )
    except Exception:
        return False
    return not any(
        snapshot is not None
        and (snapshot.get("alive") is True or snapshot.get("portOpen") is True)
        for snapshot in snapshots
    )


def _cleanup_replaced_generations(
    layout: RuntimeLayout,
    recorded: RuntimeProfile | None,
    runtime: RuntimeProfile,
    reporter: _SetupReporter,
) -> None:
    if recorded is None or not _managed_services_stopped(layout):
        return
    active = {
        os.path.normcase(str(_environment_root(str(record["path"]))))
        for record in runtime.interpreters.values()
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and _is_repo_managed_python(layout, str(record["path"]))
    }
    for capability in ("api", "gfm"):
        old = recorded.interpreters.get(capability)
        if (
            not isinstance(old, dict)
            or old.get("source") != "managed"
            or not isinstance(old.get("path"), str)
            or not _is_repo_managed_python(layout, str(old["path"]))
        ):
            continue
        root = _environment_root(str(old["path"]))
        if os.path.normcase(str(root)) in active or not root.is_dir():
            continue
        try:
            layout.assert_safe_var_path(root)
        except RuntimeError as error:
            reporter.log(
                f"Refusing to remove unsafe managed {capability} generation "
                f"{root.name}: {error}"
            )
            continue
        reporter.log(f"Removing replaced managed {capability} generation: {root.name}")
        try:
            shutil.rmtree(root)
        except OSError as error:
            reporter.log(
                f"Could not remove replaced managed {capability} generation "
                f"{root.name}: {error}"
            )


def _with_text_profile(
    profile: dict[str, Any], enabled: bool, gfm_package: Path
) -> dict[str, Any]:
    if not enabled:
        return profile
    if profile.get("id") != "windows-x86_64-cu130-pt212":
        raise RuntimeError(
            "--gfm-text is currently verified only for Windows x86_64 CUDA 13.0"
        )
    selected = dict(profile)
    versions = dict(profile["distributionVersions"])
    versions.update({"FlagEmbedding": "1.4.0", "transformers": "5.14.1"})
    selected["distributionVersions"] = versions
    runtime_manifest_path = gfm_package / "locks" / "runtime-lock-manifest.json"
    try:
        runtime_manifest = json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
        text_lock = runtime_manifest["profiles"]["windows-cu130-gfm"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError("The frozen Windows CUDA text lock manifest is invalid") from error
    relative_lock = "locks/windows-cu130-gfm.requirements.txt"
    lock_path = gfm_package / relative_lock
    expected_hash = text_lock.get("requirementsLockSha256")
    if (
        text_lock.get("requirementsLock") != relative_lock
        or text_lock.get("artifactHashesResolved") is not True
        or not isinstance(expected_hash, str)
        or not lock_path.is_file()
        or file_sha256(lock_path) != expected_hash
    ):
        raise RuntimeError("The frozen Windows CUDA text lock failed integrity validation")
    selected["requirementsLock"] = relative_lock
    selected["requirementsLockSha256"] = expected_hash
    selected["textProfile"] = True
    return selected


def _load_recorded(layout: RuntimeLayout) -> RuntimeProfile | None:
    if not layout.profile_file.is_file():
        return None
    try:
        return RuntimeProfile.load(layout.profile_file)
    except RuntimeError:
        # A v1 profile is not trusted as a v2 interpreter binding, but setup may
        # replace it after a complete new probe.
        return None


def _recorded_path(recorded: RuntimeProfile | None, capability: str) -> str | None:
    if recorded is None:
        return None
    value = recorded.interpreters.get(capability)
    return str(value.get("path")) if isinstance(value, dict) and value.get("path") else None


def _candidate_source(
    candidate: Path,
    *,
    layout: RuntimeLayout,
    explicit: str | None,
    recorded: str | None,
) -> str:
    if explicit and os.path.normcase(str(candidate)) == os.path.normcase(str(resolve_python(explicit))):
        return "explicit"
    if _is_repo_managed_python(layout, candidate):
        return "managed"
    if recorded:
        try:
            if os.path.normcase(str(candidate)) == os.path.normcase(str(resolve_python(recorded))):
                return "recorded"
        except RuntimeError:
            pass
    return "active"


def _select_compatible(
    candidates: list[Path],
    probe: Callable[[Path], CompatibilityResult],
    *,
    explicit: str | None,
    full_probe: Callable[[Path, CompatibilityResult], None] | None = None,
) -> tuple[Path, CompatibilityResult] | None:
    failures: list[str] = []
    for candidate in candidates:
        try:
            result = probe(candidate)
            if result.compatible and full_probe is not None:
                full_probe(candidate, result)
            if result.compatible:
                return candidate, result
            failures.append(f"{candidate}: {', '.join(result.errors)}")
        except Exception as error:  # candidate diagnostics are aggregated without secrets
            failures.append(f"{candidate}: {error}")
        if explicit:
            break
    if explicit and failures:
        raise RuntimeError("Explicit Python environment is incompatible: " + failures[0])
    return None


def _run_checked(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    logger: Callable[[str], None] | None = None,
    timeout: float = 1800,
) -> None:
    run_streaming_process(
        command,
        cwd=cwd,
        environment=environment,
        timeout=timeout,
        logger=logger,
        description="Command",
    )


def setup(layout: RuntimeLayout, options: SetupOptions) -> RuntimeProfile:
    requested_profile = options.profile.lower()
    env_mode = options.env_mode.lower()
    device_policy = options.device_policy.lower()
    if env_mode not in {"auto", "reuse", "managed"}:
        raise RuntimeError(f"Unsupported env mode: {options.env_mode}")
    if device_policy not in {"auto", "cpu", "cuda-required"}:
        raise RuntimeError(f"Unsupported GFM device policy: {options.device_policy}")
    if options.after_api is not None and options.skip_api:
        raise RuntimeError("The after-API onboarding callback requires an API environment")
    if env_mode == "managed" and options.api_python:
        raise RuntimeError("Explicit API Python is incompatible with --env-mode managed")
    layout.initialize_directories()
    reporter = _SetupReporter(layout)
    new_generation_roots: list[Path] = []
    try:
        reporter.stage("preflight: Python 3.12, platform, and Node/npm")
        recorded = _load_recorded(layout)
        bootstrap = ensure_bootstrap_python(options.bootstrap_python)
        bootstrap_result = probe_bootstrap_environment(bootstrap)
        if not bootstrap_result.compatible:
            raise RuntimeError(
                "Bootstrap Python is incompatible: " + ", ".join(bootstrap_result.errors)
            )
        host = _host_diagnostics()
        host["bootstrapPython"] = {
            "path": str(bootstrap),
            "version": bootstrap_result.fingerprint.get("pythonVersion"),
            "implementation": bootstrap_result.fingerprint.get("implementation"),
        }
        _require_host_compatibility(
            host,
            web=not options.skip_web,
            cuda=False,
        )
        reporter.log(json.dumps(host, ensure_ascii=False, sort_keys=True))
        migrate_private_environment_permissions(layout.llm_config_file)
        layout.initialize_serving_contracts()

        interpreters: dict[str, dict[str, Any] | None] = {
            "bootstrap": interpreter_record(
                bootstrap,
                source="explicit" if options.bootstrap_python else "launcher",
                result=bootstrap_result,
            ),
            "api": None,
            "gfm": None,
        }

        if not options.skip_api:
            reporter.stage("API environment: probe or install a Torch-free generation")
            api_recorded = _recorded_path(recorded, "api")
            if options.api_python:
                api_candidates = [resolve_python(options.api_python)]
            elif env_mode == "managed":
                previous = _recorded_managed_candidate(layout, recorded, "api")
                api_candidates = [previous] if previous is not None else []
            else:
                api_candidates = candidate_pythons(
                    None,
                    recorded=api_recorded,
                )
            selected = _select_compatible(
                api_candidates,
                lambda path: probe_api_environment(path, layout.api_root),
                explicit=options.api_python,
            )
            if selected is None:
                if env_mode == "reuse":
                    raise RuntimeError("No compatible reusable Torch-free API Python was found")
                api_lock = layout.api_root / "requirements.lock"
                if not api_lock.is_file():
                    raise RuntimeError(f"API runtime lock is missing: {api_lock}")
                api_key = _generation_key(
                    "api",
                    bootstrap=bootstrap_result,
                    descriptor={"requirementsLock": api_lock.name},
                    assets=(api_lock,),
                )
                destination = _new_generation_root(layout, "api", api_key)
                new_generation_roots.append(destination)
                try:
                    api_python = install_api_environment(
                        layout,
                        bootstrap,
                        destination=destination,
                        logger=reporter.progress,
                    )
                    api_result = probe_api_environment(api_python, layout.api_root)
                    if not api_result.compatible:
                        raise RuntimeError(
                            "Managed API environment failed verification: "
                            + ", ".join(api_result.errors)
                        )
                except Exception:
                    layout.assert_safe_var_path(destination)
                    shutil.rmtree(destination, ignore_errors=True)
                    raise
                source = "managed"
            else:
                api_python, api_result = selected
                source = _candidate_source(
                    api_python,
                    layout=layout,
                    explicit=options.api_python,
                    recorded=api_recorded,
                )
            interpreters["api"] = interpreter_record(
                api_python, source=source, result=api_result
            )

        callback_selection: str | None = None
        if options.after_api is not None:
            api_record = interpreters["api"]
            if not isinstance(api_record, dict):
                raise RuntimeError("The onboarding callback requires a verified API Python")
            reporter.stage("onboarding: complete LLM configuration and select GFM wheels")
            callback_selection = options.after_api(Path(str(api_record["path"])))
            if callback_selection is not None and not isinstance(callback_selection, str):
                raise RuntimeError("The onboarding wheel selector returned an invalid value")

        profile_selection = (callback_selection or requested_profile).strip().lower()
        if profile_selection == "auto":
            raise RuntimeError(
                "Onboarding must select offline, cpu, cuda, or an exact verified catalog ID"
            )
        install_profile: dict[str, Any] | None = None
        profiles: dict[str, dict[str, Any]] = {}
        if profile_selection == "offline":
            profile_name = "offline"
            if device_policy != "auto":
                raise RuntimeError("The offline profile requires the auto device policy")
            if options.gfm_python:
                raise RuntimeError("Explicit GFM Python is incompatible with the offline profile")
            if options.gfm_text_profile:
                raise RuntimeError("The GFM text profile is unavailable in offline mode")
        else:
            profiles = load_install_profiles(layout.install_profiles)
            install_profile = _with_text_profile(
                select_install_profile(profiles, profile_selection),
                options.gfm_text_profile,
                layout.gfm_package,
            )
            profile_name = install_profile_wheel_family(install_profile)
            if profile_name == "cpu" and device_policy == "cuda-required":
                raise RuntimeError("The selected CPU wheel profile cannot require CUDA")
            if env_mode == "managed" and options.gfm_python:
                raise RuntimeError(
                    "Explicit GFM Python is incompatible with --env-mode managed"
                )
            _require_host_compatibility(
                host,
                web=False,
                cuda=device_policy == "cuda-required",
            )

        bundle = None
        protocol_forward: dict[str, Any] | None = None
        cpu_fallback_forward: dict[str, Any] | None = None
        device_resolution: DeviceResolution | None = None
        if profile_name != "offline":
            assert install_profile is not None
            reporter.stage(
                f"GFM wheels: {install_profile['id']} ({install_profile['torchBackend']})"
            )
            gfm_recorded = _recorded_path(recorded, "gfm")
            legacy_managed = layout.gfm_environment(str(install_profile["id"]))
            if options.gfm_python:
                gfm_candidates = [resolve_python(options.gfm_python)]
            elif env_mode == "managed":
                previous = _recorded_managed_candidate(layout, recorded, "gfm")
                gfm_candidates = [previous] if previous is not None else []
            else:
                gfm_candidates = candidate_pythons(
                    None,
                    recorded=gfm_recorded,
                    managed=legacy_managed,
                )

            reporter.stage("Russia 1-4 forward: resolve and verify the execution device")

            def full(candidate: Path, result: CompatibilityResult) -> None:
                resolution = resolve_execution_device(
                    install_profile, result, device_policy
                )
                reporter.log(
                    "GFM device resolution: "
                    + json.dumps(resolution.to_document(), sort_keys=True)
                )
                if options.full_probe:
                    report = run_full_gfm_probe(
                        layout, candidate, device=resolution.resolved_device
                    )
                    reporter.log(json.dumps(report, ensure_ascii=False, sort_keys=True))
                    if resolution.resolved_device == "cuda":
                        cpu_report = run_full_gfm_probe(layout, candidate, device="cpu")
                        reporter.log(
                            "CUDA-wheel CPU fallback: "
                            + json.dumps(cpu_report, ensure_ascii=False, sort_keys=True)
                        )

            selected = _select_compatible(
                gfm_candidates,
                lambda path: probe_gfm_environment(path, install_profile),
                explicit=options.gfm_python,
                full_probe=full,
            )
            if selected is None:
                if env_mode == "reuse":
                    raise RuntimeError("No compatible reusable GFM Python was found")
                gfm_lock = layout.gfm_package / str(install_profile["requirementsLock"])
                gfm_key = _generation_key(
                    "gfm",
                    bootstrap=bootstrap_result,
                    descriptor=install_profile,
                    assets=(gfm_lock, layout.install_profiles),
                )
                destination = _new_generation_root(layout, "gfm", gfm_key)
                new_generation_roots.append(destination)
                try:
                    gfm_python = install_gfm_environment(
                        layout,
                        bootstrap,
                        install_profile,
                        destination=destination,
                        logger=reporter.progress,
                    )
                    gfm_result = probe_gfm_environment(gfm_python, install_profile)
                    if not gfm_result.compatible:
                        raise RuntimeError(
                            "Managed GFM environment failed verification: "
                            + ", ".join(gfm_result.errors)
                        )
                    device_resolution = resolve_execution_device(
                        install_profile, gfm_result, device_policy
                    )
                    reporter.log(
                        "GFM device resolution: "
                        + json.dumps(device_resolution.to_document(), sort_keys=True)
                    )
                    if options.full_probe:
                        report = run_full_gfm_probe(
                            layout,
                            gfm_python,
                            device=device_resolution.resolved_device,
                        )
                        reporter.log(json.dumps(report, ensure_ascii=False, sort_keys=True))
                        if device_resolution.resolved_device == "cuda":
                            cpu_report = run_full_gfm_probe(
                                layout, gfm_python, device="cpu"
                            )
                            reporter.log(
                                "CUDA-wheel CPU fallback: "
                                + json.dumps(
                                    cpu_report, ensure_ascii=False, sort_keys=True
                                )
                            )
                except Exception:
                    layout.assert_safe_var_path(destination)
                    shutil.rmtree(destination, ignore_errors=True)
                    raise
                source = "managed"
            else:
                gfm_python, gfm_result = selected
                device_resolution = resolve_execution_device(
                    install_profile, gfm_result, device_policy
                )
                source = _candidate_source(
                    gfm_python,
                    layout=layout,
                    explicit=options.gfm_python,
                    recorded=gfm_recorded,
                )
            gfm_interpreter = interpreter_record(
                gfm_python, source=source, result=gfm_result
            )
            assert device_resolution is not None
            gfm_interpreter["features"] = {
                "gfmText": options.gfm_text_profile,
                "wheelFamily": profile_name,
                "devicePolicy": device_policy,
            }
            interpreters["gfm"] = gfm_interpreter
            reporter.stage("bundle seeds: model, knowledge, reviewed cases, and examples")
            bundle = install_public_runtime_bundle(layout, gfm_python)
            reporter.stage("Global/In-domain/Low-label/Cross-domain checkpoint forward: verified installed model")
            protocol_forward = run_checkpoint_forward_probe(
                layout, gfm_python, device=device_resolution.resolved_device
            )
            reporter.log(json.dumps(protocol_forward, ensure_ascii=False, sort_keys=True))
            if device_resolution.resolved_device == "cuda":
                cpu_fallback_forward = run_checkpoint_forward_probe(
                    layout, gfm_python, device="cpu"
                )
                reporter.log(
                    "CUDA-wheel CPU checkpoint fallback: "
                    + json.dumps(
                        cpu_fallback_forward, ensure_ascii=False, sort_keys=True
                    )
                )

        reporter.stage("upload examples: materialize and verify zero/few-shot task copies")
        if bundle is None:
            bundle = load_and_verify_bundle(layout)
        examples = materialize_target_examples(layout, bundle)
        reporter.log(json.dumps(examples, ensure_ascii=False, sort_keys=True))

        if not options.skip_web:
            reporter.stage("npm dependencies: reproducible npm ci")
            npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
            if not npm:
                raise RuntimeError("npm is required to install Web dependencies")
            _run_checked(
                [npm, "--prefix", str(layout.web_root), "ci"],
                cwd=layout.project_root,
                environment=clean_process_environment(),
                logger=reporter.progress,
            )

        system, machine = normalized_platform()
        runtime = RuntimeProfile.create(
            profile=profile_name,
            env_mode=env_mode,
            device_policy=device_policy,
            install_profile_id=str(install_profile["id"]) if install_profile else None,
            platform={
                "system": system,
                "machine": machine,
                "libc": str(host.get("libc") or ""),
                "libcVersion": str(host.get("libcVersion") or ""),
                "pythonImplementation": platform.python_implementation(),
            },
            interpreters=interpreters,
            setup_summary={
                "exampleUploads": examples,
                "protocolForward": protocol_forward,
                "cpuFallbackForward": cpu_fallback_forward,
                "deviceResolution": (
                    device_resolution.to_document() if device_resolution else None
                ),
            },
        )
        runtime.write(layout.profile_file)
        new_generation_roots.clear()
        _cleanup_replaced_generations(layout, recorded, runtime, reporter)
        reporter.stage("complete: runtime profile switched atomically")
        return runtime
    except Exception as error:
        reporter.log(f"setup failed: {error}")
        for destination in new_generation_roots:
            layout.assert_safe_var_path(destination)
            shutil.rmtree(destination, ignore_errors=True)
        raise


def _assert_fingerprint(stored: dict[str, Any], result: CompatibilityResult, name: str) -> None:
    fingerprint = stored.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError(f"Runtime profile has no {name} environment fingerprint")
    if fingerprint.get("fingerprintSha256") != result.fingerprint.get("fingerprintSha256"):
        raise RuntimeError(f"{name} Python environment drifted after setup; rerun setup")


def validate_profile(layout: RuntimeLayout, runtime: RuntimeProfile) -> dict[str, Path | None]:
    system, machine = normalized_platform()
    if runtime.platform.get("system") != system or runtime.platform.get("machine") != machine:
        raise RuntimeError("Runtime profile was created for a different operating system or architecture")
    api_record = runtime.interpreters.get("api")
    if not isinstance(api_record, dict):
        raise RuntimeError("Runtime profile has no API Python; rerun setup without --skip-api")
    api_python = resolve_python(str(api_record.get("path", "")))
    api_result = probe_api_environment(api_python, layout.api_root)
    if not api_result.compatible:
        raise RuntimeError("API Python is incompatible: " + ", ".join(api_result.errors))
    _assert_fingerprint(api_record, api_result, "API")
    gfm_python: Path | None = None
    if runtime.profile != "offline":
        profiles = load_install_profiles(layout.install_profiles)
        install_profile = profiles.get(str(runtime.install_profile_id))
        if install_profile is None:
            raise RuntimeError("Runtime profile references an unknown GFM install profile")
        gfm_record = runtime.interpreters.get("gfm")
        if not isinstance(gfm_record, dict):
            raise RuntimeError("Runtime profile has no GFM Python")
        text_enabled = bool(
            isinstance(gfm_record.get("features"), dict)
            and gfm_record["features"].get("gfmText") is True
        )
        install_profile = _with_text_profile(
            install_profile, text_enabled, layout.gfm_package
        )
        features = gfm_record.get("features")
        if isinstance(features, dict):
            recorded_policy = features.get("devicePolicy")
            recorded_family = features.get("wheelFamily")
            if recorded_policy is not None and recorded_policy != runtime.device_policy:
                raise RuntimeError("Runtime profile GFM device policy binding differs")
            if (
                recorded_family is not None
                and recorded_family != install_profile_wheel_family(install_profile)
            ):
                raise RuntimeError("Runtime profile GFM wheel-family binding differs")
        gfm_python = resolve_python(str(gfm_record.get("path", "")))
        gfm_result = probe_gfm_environment(gfm_python, install_profile)
        if not gfm_result.compatible:
            raise RuntimeError("GFM Python is incompatible: " + ", ".join(gfm_result.errors))
        _assert_fingerprint(gfm_record, gfm_result, "GFM")
    return {"api": api_python, "gfm": gfm_python}


def _cleared_llm_environment() -> dict[str, str]:
    environment = clean_process_environment()
    for name in tuple(environment):
        if is_ambient_llm_environment_name(name):
            environment.pop(name, None)
    return environment


def resolve_llm(layout: RuntimeLayout, mode: str, *, no_prompt: bool) -> bool | str:
    selected = mode.lower()
    if selected not in {"optional", "required", "disabled"}:
        raise RuntimeError(f"Unsupported LLM mode: {mode}")
    if selected == "disabled":
        return False
    configuration = parse_private_environment(layout.llm_config_file)
    state = configuration_state(configuration)
    if state == "complete":
        return True
    if state == "partial":
        raise RuntimeError("LLM configuration is partial; reconfigure it before startup")
    if no_prompt or not os.isatty(0):
        if selected == "required":
            raise RuntimeError("LLM configuration is required but missing")
        return False
    choice = input("[C]onfigure now / continue [O]ffline / [Q]uit [C]: ").strip().lower() or "c"
    if choice == "c":
        return "configure"
    if choice == "o":
        if selected == "required":
            raise RuntimeError("Required LLM mode cannot continue offline")
        return False
    raise RuntimeError("Startup cancelled")


def _common_environment(layout: RuntimeLayout) -> dict[str, str]:
    environment = _cleared_llm_environment()
    environment.update(
        {
            "SOCIALGRAPH_FM_HOME": str(layout.gfm_home),
            "TEMP": str(layout.temp_root),
            "TMP": str(layout.temp_root),
            "PIP_CACHE_DIR": str(layout.cache_root / "pip"),
            "HF_HOME": str(layout.cache_root / "hf"),
            "TORCH_HOME": str(layout.cache_root / "torch"),
            "TORCHINDUCTOR_CACHE_DIR": str(layout.cache_root / "torchinductor"),
            "WANDB_DIR": str(layout.cache_root / "wandb"),
        }
    )
    return environment


def _runtime_service_device(runtime: RuntimeProfile) -> str:
    device_policy = getattr(runtime, "device_policy", "auto")
    if runtime.profile == "cpu" or device_policy == "cpu":
        return "cpu"
    if device_policy == "cuda-required":
        return "cuda"
    return "auto"


def build_services(
    layout: RuntimeLayout,
    runtime: RuntimeProfile,
    interpreters: dict[str, Path | None],
    *,
    mode: str,
    enable_llm: bool,
    ports: Ports,
) -> list[ServiceSpec]:
    common = _common_environment(layout)
    services: list[ServiceSpec] = []
    if runtime.profile != "offline":
        gfm_python = interpreters["gfm"]
        assert gfm_python is not None
        gfm_environment = dict(common)
        gfm_environment.update(
            {
                "PYTHONPATH": str(layout.gfm_package / "src"),
                "PYTHONNOUSERSITE": "1",
            }
        )
        gfm_arguments = (
            "-m",
            "socialgraph_gfm.core.inference_cli",
            "--runtime-root",
            str(layout.serving_root),
            "--serving-control",
            str(layout.serving_control),
            "--published-serving-root",
            str(layout.serving_root),
            "--published-artifact-root",
            str(layout.serving_artifacts),
            "--artifact-root",
            str(layout.serving_artifacts),
            "--research-root",
            str(layout.research_root),
            "--global-model-root",
            str(layout.model_root),
            "--governance-root",
            str(layout.governance_root),
            "--global-model-device",
            _runtime_service_device(runtime),
            "--dataset-store-root",
            str(layout.dataset_store),
            "--token-file",
            str(layout.serving_token),
            "--host",
            "127.0.0.1",
            "--port",
            str(ports.gfm),
        )
        services.append(
            ServiceSpec(
                "gfm",
                ports.gfm,
                gfm_python,
                gfm_arguments,
                layout.gfm_package,
                gfm_environment,
                ("socialgraph_gfm.core.inference_cli", "--port", str(ports.gfm)),
                health_path="/internal/core/health",
                health_json={
                    "schemaVersion": "socialgraph-fm.core-internal-health/2.0",
                    "ok": True,
                },
                health_token_file=layout.serving_token,
            )
        )

    api_python = interpreters["api"]
    assert api_python is not None
    api_environment = dict(common)
    api_generation = "llm-disabled"
    if enable_llm:
        private = parse_private_environment(layout.llm_config_file)
        if configuration_state(private) != "complete":
            raise RuntimeError("Launcher enabled LLM without a complete configuration")
        for name in (*LLM_ENVIRONMENT_NAMES, "LOG_LEVEL"):
            if name in private:
                api_environment[name] = private[name]
        config_stat = layout.llm_config_file.stat()
        api_generation = f"llm-configured:{config_stat.st_mtime_ns}:{config_stat.st_size}"
    api_environment.update(
        {
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
            "GFM_INFRASTRUCTURE_READY": "false",
            "SOCIALGRAPH_CORE_API_PORT": str(ports.api),
            "GFM_SERVICE_URL": "" if runtime.profile == "offline" else f"http://127.0.0.1:{ports.gfm}",
            "GFM_SESSION_TOKEN_FILE": "" if runtime.profile == "offline" else str(layout.serving_token),
            "GFM_CORE_SERVING_CONTROL_FILE": "" if runtime.profile == "offline" else str(layout.serving_control),
            "GFM_CORE_RUN_BINDING_ROOT": str(layout.bindings_root),
            "GFM_RESEARCH_RUN_BINDING_ROOT": str(layout.research_bindings_root),
            "GFM_GLOBAL_MODEL_RUN_BINDING_ROOT": str(layout.global_model_bindings_root),
            "GFM_GLOBAL_MODEL_REVIEW_ROOT": str(layout.global_model_reviews_root),
            "GFM_GOVERNANCE_ROOT": str(layout.governance_root),
            "GFM_GOVERNANCE_BUNDLE_MAX_BYTES": "268435456",
            "GFM_GOVERNANCE_EXPANDED_MAX_BYTES": "1073741824",
            "GFM_CORE_SERVING_HIGH_WATER_ROOT": str(layout.serving_high_water_root),
            "DATASET_STORAGE_ROOT": str(layout.dataset_store),
            "ALLOWED_ORIGINS": f"http://127.0.0.1:{ports.web},http://localhost:{ports.web}",
            "ENABLE_TRUSTED_LOCAL_CONVERSION": "false",
            "TRUSTED_DATA_ROOTS": "",
            "TRUSTED_CONVERTER_PYTHON": "",
            "LOCAL_DEMO_LOOPBACK_ONLY": "true",
            "RUNTIME_BUILD_ID": "unified-local",
        }
    )
    services.append(
        ServiceSpec(
            "socialgraph-api",
            ports.api,
            api_python,
            ("-m", "app", "--runtime-identity-root", str(layout.governance_root)),
            layout.api_root,
            api_environment,
            ("-m", "app", "--runtime-identity-root", str(layout.governance_root)),
            health_path="/api/v1/health",
            health_json={"status": "ok", "service": "socialgraph-fm-api"},
            generation=api_generation,
        )
    )

    node = shutil.which("node")
    if not node:
        raise RuntimeError("node is required to start the Web workbench")
    vite = layout.web_root / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite.is_file():
        raise RuntimeError("Web dependencies are missing; run setup first")
    web_environment = dict(common)
    web_environment["VITE_SOCIALGRAPH_API_BASE_URL"] = f"http://127.0.0.1:{ports.api}"
    services.append(
        ServiceSpec(
            "governance-web",
            ports.web,
            Path(node),
            (
                str(vite),
                "dev" if mode == "development" else "preview",
                "--host",
                "127.0.0.1",
                "--port",
                str(ports.web),
                "--strictPort",
            ),
            layout.web_root,
            web_environment,
            (str(vite), "--port", str(ports.web)),
            health_path="/",
            health_text='id="root"',
        )
    )
    return services


LLM_ENVIRONMENT_NAMES = (
    "LLM_API_BASE",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_API_MODE",
    "LLM_AUTH_SCHEME",
    "LLM_ANTHROPIC_VERSION",
    "LLM_TIMEOUT_SECONDS",
    "LLM_ALLOW_INSECURE_LOOPBACK",
    "LLM_VERIFICATION_STATUS",
)


def test_llm_configuration(
    layout: RuntimeLayout,
    environment: dict[str, str],
    *,
    api_python: Path | None = None,
) -> None:
    """Exercise the API package's real provider and JSON-marker parsing chain."""

    if configuration_state(environment) != "complete":
        raise RuntimeError("A complete LLM configuration is required for testing")
    selected_api_python = api_python
    if selected_api_python is None:
        runtime = RuntimeProfile.load(layout.profile_file)
        api_record = runtime.interpreters.get("api")
        if not isinstance(api_record, dict):
            raise RuntimeError("The configured API Python is missing; run setup first")
        selected_api_python = resolve_python(str(api_record.get("path", "")))
    else:
        selected_api_python = resolve_python(selected_api_python)
    child_environment = _cleared_llm_environment()
    for name in LLM_ENVIRONMENT_NAMES:
        if name in environment:
            child_environment[name] = environment[name]
    child_environment["PYTHONNOUSERSITE"] = "1"
    child_environment["PYTHONPATH"] = ""
    completed = run_captured_process(
        [str(selected_api_python), "-m", "app.provider_check"],
        cwd=layout.api_root,
        environment=child_environment,
        timeout=int(environment.get("LLM_TIMEOUT_SECONDS", "15")) + 10,
        description="LLM connection check",
    )
    raw_lines = [
        line.strip()
        for line in (completed.stdout + "\n" + completed.stderr).splitlines()
        if line.strip()
    ]
    document: dict[str, Any] | None = None
    for line in reversed(raw_lines):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            document = candidate
            break
    code = str(document.get("code", "LLM_CONFIGURATION_ERROR")) if document else (
        "LLM_CONFIGURATION_ERROR"
    )
    if completed.returncode != 0:
        categories = {
            "LLM_AUTH_ERROR": "AUTH",
            "LLM_ENDPOINT_ERROR": "ENDPOINT_OR_MODE",
            "LLM_REQUEST_REJECTED": "REQUEST_OR_MODEL",
            "LLM_RATE_LIMITED": "RATE_LIMIT",
            "LLM_UPSTREAM_ERROR": "UPSTREAM",
            "LLM_TIMEOUT": "TIMEOUT",
            "LLM_NETWORK_ERROR": "NETWORK_TLS",
            "LLM_INVALID_RESPONSE": "INVALID_RESPONSE",
            "LLM_RESPONSE_TOO_LARGE": "INVALID_RESPONSE",
            "LLM_CONFIGURATION_ERROR": "CONFIGURATION",
        }
        raise RuntimeError(
            "LLM connection check failed: " + categories.get(code, "INVALID_RESPONSE")
        )
    if (
        document is None
        or document.get("schemaVersion") != "socialgraph-fm.llm-provider-check/1.0"
        or document.get("ok") is not True
        or code != "OK"
    ):
        raise RuntimeError("LLM connection check failed: INVALID_RESPONSE")


def start_stack(layout: RuntimeLayout, *, development: bool, enable_llm: bool) -> Ports:
    runtime = RuntimeProfile.load(layout.profile_file)
    layout.initialize_directories()
    layout.initialize_serving_contracts()
    _require_host_compatibility(
        _host_diagnostics(),
        web=True,
        cuda=(
            runtime.profile == "cuda"
            and runtime.device_policy == "cuda-required"
        ),
    )
    interpreters = validate_profile(layout, runtime)
    mode = "development" if development else "production"
    if not development:
        npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
        if not npm:
            raise RuntimeError("npm is required to build the Web workbench")
        _run_checked(
            [npm, "--prefix", str(layout.web_root), "run", "build"],
            cwd=layout.project_root,
            environment=_cleared_llm_environment(),
        )
    ports = Ports.environment()
    services = build_services(
        layout,
        runtime,
        interpreters,
        mode=mode,
        enable_llm=enable_llm,
        ports=ports,
    )
    manager = ProcessManager(layout.pid_root, layout.log_root)
    started: list[str] = []
    try:
        for service in services:
            # Loading and identity-checking the four bundled checkpoints can
            # exceed one minute on a cold CPU host. API and Web remain bounded
            # to the shorter startup window.
            timeout = 180.0 if service.name == "gfm" else 60.0
            if manager.start(service, timeout=timeout):
                started.append(service.name)
    except Exception:
        for name in reversed(started):
            manager.stop(name)
        raise
    return ports


def stop_stack(layout: RuntimeLayout) -> None:
    manager = ProcessManager(layout.pid_root, layout.log_root)
    for name in ("governance-web", "socialgraph-api", "gfm"):
        manager.stop(name)
    if manager.read_record("gfm") is None:
        layout.serving_token.unlink(missing_ok=True)


def doctor(
    layout: RuntimeLayout, *, test_llm: bool = False, full: bool = False
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str, required: bool = True) -> None:
        checks.append({"name": name, "passed": passed, "required": required, "detail": detail})

    host = _host_diagnostics()
    try:
        bootstrap_python = resolve_python(sys.executable)
        bootstrap_probe = probe_bootstrap_environment(bootstrap_python)
        host["bootstrapPython"] = {
            "path": str(bootstrap_python),
            "version": bootstrap_probe.fingerprint.get("pythonVersion"),
            "implementation": bootstrap_probe.fingerprint.get("implementation"),
        }
        add(
            "bootstrap-python",
            bootstrap_probe.compatible,
            f"Python {bootstrap_probe.fingerprint.get('pythonVersion')} at {bootstrap_python}",
        )
    except Exception as error:
        host["bootstrapPython"] = {"path": sys.executable, "error": str(error)}
        add("bootstrap-python", False, str(error))
    add(
        "host-platform",
        True,
        f"{host['system']}/{host['machine']} "
        f"libc={host.get('libc') or 'n/a'} {host.get('libcVersion') or ''}".rstrip(),
    )
    node = host["node"]
    npm = host["npm"]
    node_ready = node.get("available") is True and _major_version(node.get("version")) == 24
    npm_ready = npm.get("available") is True and _major_version(npm.get("version")) == 11
    add(
        "node-runtime",
        node_ready and npm_ready,
        f"Node {node.get('version') or 'missing'}; npm {npm.get('version') or 'missing'}",
    )

    add(
        "repository-layout",
        all(path.is_dir() for path in (layout.web_root, layout.api_root, layout.gfm_package)),
        "apps/web, services/api, packages/gfm",
    )
    runtime: RuntimeProfile | None = None
    environment_summaries: dict[str, Any] = {}
    install_backend: dict[str, Any] = {}
    doctor_resolution: DeviceResolution | None = None
    try:
        runtime = RuntimeProfile.load(layout.profile_file)
        interpreters = validate_profile(layout, runtime)
        environment_summaries = _environment_summaries(runtime)
        api_summary = environment_summaries.get("api", {})
        gfm_summary = environment_summaries.get("gfm", {})
        detail = (
            f"API {api_summary.get('source')} Python {api_summary.get('pythonVersion')}"
            + (
                f"; GFM {gfm_summary.get('source')} {runtime.install_profile_id} "
                f"Torch {gfm_summary.get('versions', {}).get('torch')}"
                if runtime.profile != "offline"
                else "; GFM offline"
            )
        )
        add("python-environments", True, detail)
        if runtime.profile == "offline":
            install_backend = {"profile": "offline", "torchBackend": None}
            add("gfm-backend", True, "offline profile")
        else:
            profiles = load_install_profiles(layout.install_profiles)
            selected_profile = profiles.get(str(runtime.install_profile_id))
            if selected_profile is None:
                raise RuntimeError(
                    f"Recorded GFM install profile is unavailable: {runtime.install_profile_id}"
                )
            versions = gfm_summary.get("versions", {})
            gfm_python = interpreters.get("gfm")
            if not isinstance(gfm_python, Path):
                raise RuntimeError("GFM Python is unavailable for device resolution")
            runtime_gfm_record = runtime.interpreters.get("gfm")
            runtime_gfm_features = (
                runtime_gfm_record.get("features")
                if isinstance(runtime_gfm_record, dict)
                else None
            )
            text_enabled = bool(
                isinstance(runtime_gfm_features, dict)
                and runtime_gfm_features.get("gfmText") is True
            )
            selected_profile = _with_text_profile(
                selected_profile, text_enabled, layout.gfm_package
            )
            live_result = probe_gfm_environment(gfm_python, selected_profile)
            if not live_result.compatible:
                raise RuntimeError(
                    "GFM Python is incompatible: " + ", ".join(live_result.errors)
                )
            doctor_resolution = resolve_execution_device(
                selected_profile, live_result, runtime.device_policy
            )
            capabilities = live_result.runtime_capabilities
            install_backend = {
                "profile": runtime.install_profile_id,
                "wheelFamily": install_profile_wheel_family(selected_profile),
                "torchBackend": selected_profile.get("torchBackend"),
                "wheelIndex": selected_profile.get("torchIndexUrl"),
                "torch": versions.get("torch"),
                "torchGeometric": versions.get("torch-geometric"),
                "cudaRuntime": capabilities.get("cudaRuntime"),
                "cudaAvailable": capabilities.get("cudaAvailable"),
                "devicePolicy": runtime.device_policy,
                "resolvedDevice": doctor_resolution.resolved_device,
                "fallbackReason": doctor_resolution.fallback_reason,
                "gpuDriver": host.get("gpuDriver"),
            }
            add(
                "gfm-backend",
                True,
                f"{runtime.install_profile_id}; wheel={selected_profile.get('torchBackend')}; "
                f"Torch={versions.get('torch')}; PyG={versions.get('torch-geometric')}; "
                f"CUDA={capabilities.get('cudaRuntime') or 'none'}; "
                f"policy={runtime.device_policy}; "
                f"device={doctor_resolution.resolved_device}; "
                f"driver={host['gpuDriver'].get('driverVersion') or 'n/a'}",
            )
    except Exception as error:
        interpreters = {"api": None, "gfm": None}
        add("python-environments", False, str(error))
    add(
        "web-dependencies",
        (layout.web_root / "node_modules" / "vite" / "bin" / "vite.js").is_file(),
        str(layout.web_root),
    )
    published_bundle = None
    checkpoint_forward: dict[str, Any] = {}
    cpu_fallback_forward: dict[str, Any] = {}
    if runtime is not None and runtime.profile != "offline":
        runtime_assets_verified = False
        try:
            published_bundle = load_and_verify_bundle(layout)
            verified = verify_installed_runtime_bundle(layout, published_bundle)
            seeds = verify_installed_runtime_seeds(layout, published_bundle)
            gfm_python = interpreters.get("gfm")
            if not isinstance(gfm_python, Path):
                raise RuntimeError("GFM Python is unavailable for runtime-state validation")
            runtime_state = verify_gfm_runtime_state(layout, gfm_python)
            runtime_assets_verified = True
            add(
                "runtime-bundle",
                True,
                f"{verified['assetCount']} model assets, serving registry, and "
                f"{sum(seeds.values())} seeded runtime assets; reviewed cases "
                f"{str(runtime_state['reviewedCaseIndexHash'])[:12]}",
            )
        except Exception as error:
            add("runtime-bundle", False, str(error))
        gfm_python = interpreters.get("gfm")
        if (
            full
            and runtime_assets_verified
            and isinstance(gfm_python, Path)
            and doctor_resolution is not None
        ):
            try:
                report = run_full_gfm_probe(
                    layout,
                    gfm_python,
                    device=doctor_resolution.resolved_device,
                    use_installed_model=True,
                )
                add(
                    "global-forward",
                    report.get("passed") is True,
                    f"Russia bundles={report.get('bundleCount')} "
                    f"nodes={report.get('nodeCount')} device={report.get('device')}",
                )
            except Exception as error:
                add("global-forward", False, str(error))
            try:
                checkpoint_forward = run_checkpoint_forward_probe(
                    layout,
                    gfm_python,
                    device=doctor_resolution.resolved_device,
                )
                summaries = ", ".join(
                    f"{item['protocol']}={str(item['outputHash'])[:12]}"
                    for item in checkpoint_forward["protocols"]
                )
                add("protocol-forwards", True, summaries)
            except Exception as error:
                add("protocol-forwards", False, str(error))
            if doctor_resolution.resolved_device == "cuda":
                try:
                    cpu_global = run_full_gfm_probe(
                        layout,
                        gfm_python,
                        device="cpu",
                        use_installed_model=True,
                    )
                    add(
                        "global-forward-cpu-fallback",
                        cpu_global.get("passed") is True,
                        f"Russia bundles={cpu_global.get('bundleCount')} "
                        f"nodes={cpu_global.get('nodeCount')} device=cpu",
                    )
                except Exception as error:
                    add("global-forward-cpu-fallback", False, str(error))
                try:
                    cpu_fallback_forward = run_checkpoint_forward_probe(
                        layout, gfm_python, device="cpu"
                    )
                    summaries = ", ".join(
                        f"{item['protocol']}={str(item['outputHash'])[:12]}"
                        for item in cpu_fallback_forward["protocols"]
                    )
                    add("protocol-forwards-cpu-fallback", True, summaries)
                except Exception as error:
                    add("protocol-forwards-cpu-fallback", False, str(error))
        elif full:
            add(
                "global-forward",
                False,
                "skipped because installed runtime assets were not verified",
            )
            add(
                "protocol-forwards",
                False,
                "skipped because installed runtime assets were not verified",
            )
    example_uploads: dict[str, Any] = {}
    try:
        if published_bundle is None:
            published_bundle = load_and_verify_bundle(layout)
        example_uploads = verify_target_examples(layout, published_bundle)
        add(
            "target-examples",
            True,
            f"zero-shot={example_uploads['zeroShot']['path']}; "
            f"few-shot={example_uploads['fewShot']['path']}",
        )
    except Exception as error:
        add("target-examples", False, str(error))
    manager = ProcessManager(layout.pid_root, layout.log_root)
    process_ready = True
    details: list[str] = []
    for name in ("governance-web", "socialgraph-api", "gfm"):
        snapshot = manager.snapshot(name)
        if snapshot is None:
            continue
        passed = bool(
            snapshot["alive"]
            and snapshot["identityMatches"]
            and snapshot["portOpen"]
            and snapshot["healthReady"]
        )
        process_ready = process_ready and passed
        details.append(f"{name}={'ready' if passed else 'invalid'}")
    add("managed-processes", process_ready, ", ".join(details) or "stack stopped")
    try:
        llm = parse_private_environment(layout.llm_config_file)
        llm_state = configuration_state(llm)
        passed = llm_state != "partial"
        if test_llm:
            try:
                test_llm_configuration(layout, llm)
                llm["LLM_VERIFICATION_STATUS"] = "call_succeeded"
            except RuntimeError:
                if llm_state == "complete":
                    llm["LLM_VERIFICATION_STATUS"] = "fallback"
                    write_private_environment(
                        layout.llm_config_file, _llm_configuration_defaults(llm)
                    )
                raise
            write_private_environment(
                layout.llm_config_file, _llm_configuration_defaults(llm)
            )
        add("llm-configuration", passed, llm_state)
    except Exception as error:
        add("llm-configuration", False, str(error))
    fatal = [check for check in checks if check["required"] and not check["passed"]]
    return {
        "schemaVersion": "socialgraph-fm.doctor/2.0",
        "profile": runtime.profile if runtime else None,
        "passed": not fatal,
        "host": host,
        "installBackend": install_backend,
        "deviceResolution": (
            doctor_resolution.to_document() if doctor_resolution else None
        ),
        "protocolForward": checkpoint_forward,
        "cpuFallbackForward": cpu_fallback_forward,
        "environments": environment_summaries,
        "exampleUploads": example_uploads,
        "checks": checks,
    }


def _llm_configuration_defaults(environment: dict[str, str]) -> dict[str, str]:
    selected = dict(environment)
    mode = selected.get("LLM_API_MODE") or "chat_completions"
    selected["LLM_API_MODE"] = mode
    selected.setdefault(
        "LLM_AUTH_SCHEME", "x-api-key" if mode == "anthropic_messages" else "bearer"
    )
    selected.setdefault(
        "LLM_ANTHROPIC_VERSION",
        "2023-06-01" if mode == "anthropic_messages" else "",
    )
    selected.setdefault("LLM_TIMEOUT_SECONDS", "15")
    selected.setdefault("LLM_ALLOW_INSECURE_LOOPBACK", "false")
    selected.setdefault("LLM_VERIFICATION_STATUS", "configured_unverified")
    return selected


def _environment_summaries(runtime: RuntimeProfile) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for capability in ("bootstrap", "api", "gfm"):
        record = runtime.interpreters.get(capability)
        if not isinstance(record, dict):
            summaries[capability] = None
            continue
        fingerprint = record.get("fingerprint")
        if not isinstance(fingerprint, dict):
            summaries[capability] = {
                "path": record.get("path"),
                "source": record.get("source"),
            }
            continue
        summaries[capability] = {
            "path": record.get("path"),
            "source": record.get("source"),
            "pythonVersion": fingerprint.get("pythonVersion"),
            "system": fingerprint.get("system"),
            "machine": fingerprint.get("machine"),
            "libc": fingerprint.get("libc"),
            "libcVersion": fingerprint.get("libcVersion"),
            "versions": fingerprint.get("versions"),
            "torch": fingerprint.get("torch"),
            "neighborLoader": fingerprint.get("neighborLoader"),
            "fingerprintSha256": fingerprint.get("fingerprintSha256"),
        }
    return summaries
