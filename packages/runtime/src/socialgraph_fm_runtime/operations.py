"""Single-environment CPU setup, lifecycle, and diagnostics."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import (
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
    clean_process_environment,
    ensure_bootstrap_python,
    install_runtime_environment,
    interpreter_record,
    is_ambient_llm_environment_name,
    load_install_profiles,
    normalized_platform,
    probe_bootstrap_environment,
    probe_runtime_environment,
    prune_runtime_bytecode,
    resolve_python,
    select_install_profile,
)
from .layout import RuntimeLayout, environment_python
from .llm import (
    configuration_state,
    migrate_private_environment_permissions,
    parse_private_environment,
)
from .processes import ProcessManager, ServiceSpec
from .profile import RuntimeProfile
from .subprocess_control import (
    redact_subprocess_text,
    run_captured_process,
)
from .web_bundle import install_web_bundle


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
        public = _port("SOCIALGRAPH_PUBLIC_PORT", 5173)
        return cls(web=public, api=public, gfm=_port("SOCIALGRAPH_GFM_PORT", 8766))


@dataclass(frozen=True)
class SetupOptions:
    full_probe: bool = True


class _SetupReporter:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.path = layout.setup_log_file
        layout.assert_safe_var_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("SocialGraph-FM setup log\n", encoding="utf-8")

    def log(self, message: str) -> None:
        selected = redact_subprocess_text(message).replace("\r\n", "\n")
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(selected.rstrip("\n") + "\n")

    def stage(self, message: str) -> None:
        self.log(f"[setup] {message}")
        print(f"[setup] {message}", file=sys.stderr, flush=True)

    def progress(self, message: str) -> None:
        self.log(message)
        print(redact_subprocess_text(message), file=sys.stderr, flush=True)


def _cleared_llm_environment() -> dict[str, str]:
    environment = clean_process_environment()
    for name in tuple(environment):
        if is_ambient_llm_environment_name(name):
            environment.pop(name, None)
    return environment


def _managed_services_stopped(layout: RuntimeLayout) -> bool:
    manager = ProcessManager(layout.pid_root, layout.log_root)
    try:
        snapshots = tuple(
            manager.snapshot(name)
            for name in ("governance-web", "socialgraph-api", "gfm")
        )
    except Exception:
        return False
    return not any(
        snapshot is not None
        and (snapshot.get("alive") is True or snapshot.get("portOpen") is True)
        for snapshot in snapshots
    )


def _assert_fingerprint(
    stored: dict[str, Any], result: CompatibilityResult, name: str
) -> None:
    fingerprint = stored.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError(f"Runtime profile has no {name} environment fingerprint")
    if fingerprint.get("fingerprintSha256") != result.fingerprint.get("fingerprintSha256"):
        raise RuntimeError(f"{name} Python environment drifted after onboarding")


def _profile_for_host(layout: RuntimeLayout) -> dict[str, Any]:
    profiles = load_install_profiles(layout.install_profiles)
    return select_install_profile(profiles)


def _record_for_profile(
    python: Path, result: CompatibilityResult, profile: dict[str, Any]
) -> dict[str, Any]:
    record = interpreter_record(python, source="managed", result=result)
    record["installLockSha256"] = profile["requirementsLockSha256"]
    return record


def _validate_recorded_runtime(
    layout: RuntimeLayout, runtime: RuntimeProfile, profile: dict[str, Any]
) -> tuple[Path, CompatibilityResult]:
    system, machine = normalized_platform()
    if (
        runtime.platform.get("system") != system
        or runtime.platform.get("machine") != machine
        or runtime.install_profile_id != profile["id"]
        or runtime.interpreter.get("installLockSha256")
        != profile["requirementsLockSha256"]
    ):
        raise RuntimeError("The managed CPU runtime binding is stale")
    expected = environment_python(layout.runtime_environment)
    recorded = resolve_python(str(runtime.interpreter.get("path", "")))
    if os.path.normcase(str(recorded)) != os.path.normcase(str(expected)):
        raise RuntimeError("The runtime profile does not reference var/runtime")
    result = probe_runtime_environment(recorded, profile)
    if not result.compatible:
        raise RuntimeError("Managed CPU runtime is incompatible: " + ", ".join(result.errors))
    _assert_fingerprint(runtime.interpreter, result, "managed")
    return recorded, result


def _replace_runtime_environment(
    layout: RuntimeLayout,
    staging: Path,
    profile: dict[str, Any],
) -> tuple[Path, CompatibilityResult, Path | None]:
    active = layout.runtime_environment
    backup: Path | None = None
    if active.exists():
        backup = Path(
            tempfile.mkdtemp(
                prefix="runtime-backup-", suffix=".stage", dir=layout.temp_root
            )
        )
        backup.rmdir()
        layout.assert_safe_var_path(backup)
        os.replace(active, backup)
    try:
        os.replace(staging, active)
        python = resolve_python(environment_python(active))
        result = probe_runtime_environment(python, profile)
        if not result.compatible:
            raise RuntimeError(
                "Installed CPU runtime failed post-switch verification: "
                + ", ".join(result.errors)
            )
        return python, result, backup
    except Exception:
        if active.exists():
            shutil.rmtree(active, ignore_errors=True)
        if backup is not None and backup.exists():
            os.replace(backup, active)
        raise


def _rollback_runtime(layout: RuntimeLayout, backup: Path | None) -> None:
    active = layout.runtime_environment
    if backup is None:
        shutil.rmtree(active, ignore_errors=True)
        return
    if active.exists():
        shutil.rmtree(active, ignore_errors=True)
    if backup.exists():
        os.replace(backup, active)


def _safe_remove_known_tree(layout: RuntimeLayout, path: Path) -> None:
    if not path.exists():
        return
    absolute = Path(os.path.abspath(path))
    allowed = {
        Path(os.path.abspath(layout.managed_environment_root)),
        Path(os.path.abspath(layout.legacy_managed_environment_root)),
        Path(os.path.abspath(layout.var_root / "envs")),
        Path(os.path.abspath(layout.web_root / "node_modules")),
    }
    if absolute not in allowed or absolute in {
        Path(os.path.abspath(layout.project_root)),
        Path(os.path.abspath(layout.var_root)),
    }:
        raise RuntimeError(f"Refusing to remove an unknown legacy path: {absolute}")
    if path.is_symlink():
        raise RuntimeError(f"Refusing to follow a legacy path link: {path}")
    if path != layout.web_root / "node_modules":
        layout.assert_safe_var_path(path)
    elif path.resolve() != absolute:
        raise RuntimeError(f"Refusing to follow a legacy path reparse point: {path}")
    shutil.rmtree(path)


def _cleanup_legacy_installations(
    layout: RuntimeLayout, reporter: _SetupReporter
) -> None:
    for path in (
        layout.managed_environment_root,
        layout.legacy_managed_environment_root,
        layout.var_root / "envs",
        layout.web_root / "node_modules",
    ):
        if path.exists():
            reporter.log(f"Removing retired installation tree: {path}")
            try:
                _safe_remove_known_tree(layout, path)
            except (OSError, RuntimeError) as error:
                # The activated runtime is already verified. A locked legacy
                # cache is cleanup debt, not a reason to invalidate onboarding.
                reporter.log(f"Could not remove retired installation tree {path}: {error}")


def setup(layout: RuntimeLayout, options: SetupOptions | None = None) -> RuntimeProfile:
    """Build, verify, and atomically activate the one supported CPU runtime."""

    selected_options = options or SetupOptions()
    layout.initialize_directories()
    reporter = _SetupReporter(layout)
    if not _managed_services_stopped(layout):
        raise RuntimeError("Stop SocialGraph-FM before running onboard")
    # A retired Responses/Anthropic file may be replaced during this onboarding
    # run. Protect it before any read, but validate only the newly entered values.
    migrate_private_environment_permissions(
        layout.llm_config_file, validate_values=False
    )
    layout.initialize_serving_contracts()
    profile = _profile_for_host(layout)
    bootstrap = ensure_bootstrap_python()
    bootstrap_result = probe_bootstrap_environment(bootstrap)
    if not bootstrap_result.compatible:
        raise RuntimeError(
            "Python 3.12 bootstrap is incompatible: "
            + ", ".join(bootstrap_result.errors)
        )
    system, machine = normalized_platform()
    host = {
        "system": system,
        "machine": machine,
        "libc": platform.libc_ver()[0] or "",
        "libcVersion": platform.libc_ver()[1] or "",
        "pythonVersion": bootstrap_result.fingerprint.get("pythonVersion"),
    }
    reporter.log(json.dumps(host, ensure_ascii=False, sort_keys=True))

    runtime_python: Path | None = None
    runtime_result: CompatibilityResult | None = None
    previous: RuntimeProfile | None = None
    try:
        previous = RuntimeProfile.load(layout.profile_file)
        runtime_python, runtime_result = _validate_recorded_runtime(
            layout, previous, profile
        )
        bytecode = prune_runtime_bytecode(runtime_python)
        reporter.log(
            "Managed runtime bytecode pruning: "
            f"{bytecode.removed_files} files, {bytecode.removed_bytes} bytes"
        )
        runtime_result = probe_runtime_environment(runtime_python, profile)
        if not runtime_result.compatible:
            raise RuntimeError(
                "Managed CPU runtime failed verification after bytecode pruning: "
                + ", ".join(runtime_result.errors)
            )
        _assert_fingerprint(previous.interpreter, runtime_result, "managed")
        if bytecode.removed_files and selected_options.full_probe:
            reporter.stage("CPU runtime: verifying Russia forwards after bytecode pruning")
            report = run_full_gfm_probe(layout, runtime_python)
            reporter.log(json.dumps(report, ensure_ascii=False, sort_keys=True))
        reporter.stage("CPU runtime: verified existing var/runtime")
    except RuntimeError as error:
        reporter.log(f"Existing CPU runtime cannot be reused: {error}")
        runtime_python = None
        runtime_result = None
        reporter.stage("CPU runtime: installing the verified Windows/Ubuntu lock")

    backup: Path | None = None
    switched = False
    staging: Path | None = None
    try:
        if runtime_python is None or runtime_result is None:
            staging = Path(
                tempfile.mkdtemp(prefix="runtime-", suffix=".stage", dir=layout.temp_root)
            )
            # venv requires the destination itself not to exist.
            staging.rmdir()
            try:
                candidate = install_runtime_environment(
                    layout,
                    bootstrap,
                    profile,
                    destination=staging,
                    logger=reporter.progress,
                )
                bytecode = prune_runtime_bytecode(candidate)
                reporter.log(
                    "Managed runtime bytecode pruning: "
                    f"{bytecode.removed_files} files, {bytecode.removed_bytes} bytes"
                )
                candidate_result = probe_runtime_environment(candidate, profile)
                if not candidate_result.compatible:
                    raise RuntimeError(
                        "New CPU runtime failed verification: "
                        + ", ".join(candidate_result.errors)
                    )
                if selected_options.full_probe:
                    reporter.stage("CPU runtime: verifying four Russia forwards")
                    report = run_full_gfm_probe(layout, candidate)
                    reporter.log(json.dumps(report, ensure_ascii=False, sort_keys=True))
                runtime_python, runtime_result, backup = _replace_runtime_environment(
                    layout, staging, profile
                )
                switched = True
                staging = None
            except Exception:
                if staging is not None and staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)
                raise

        reporter.stage("Web client: verifying and installing the prebuilt bundle")
        web_summary = install_web_bundle(layout)
        reporter.log(json.dumps(web_summary, ensure_ascii=False, sort_keys=True))

        reporter.stage("Runtime assets: model, knowledge, cases, and examples")
        bundle = install_public_runtime_bundle(layout, runtime_python)
        examples = materialize_target_examples(layout, bundle)
        protocol_forward: dict[str, Any] = {}
        if selected_options.full_probe:
            reporter.stage("CPU runtime: verifying four protocol checkpoints")
            protocol_forward = run_checkpoint_forward_probe(layout, runtime_python)
            reporter.log(
                json.dumps(protocol_forward, ensure_ascii=False, sort_keys=True)
            )
        runtime = RuntimeProfile.create(
            install_profile_id=str(profile["id"]),
            platform={
                "system": system,
                "machine": machine,
                "libc": str(host["libc"]),
                "libcVersion": str(host["libcVersion"]),
                "pythonImplementation": platform.python_implementation(),
            },
            interpreter=_record_for_profile(runtime_python, runtime_result, profile),
            setup_summary={
                "exampleUploads": examples,
                "protocolForward": protocol_forward,
                "webBundle": web_summary,
            },
        )
        runtime.write(layout.profile_file)
        switched = False
        if backup is not None and backup.exists():
            try:
                shutil.rmtree(backup)
            except OSError as error:
                reporter.log(f"Could not remove previous runtime backup {backup}: {error}")
        _cleanup_legacy_installations(layout, reporter)
        reporter.stage("local runtime ready: CPU runtime and prebuilt Web client")
        return runtime
    except Exception as error:
        reporter.log(f"setup failed: {error}")
        if switched:
            _rollback_runtime(layout, backup)
        raise


def validate_profile(
    layout: RuntimeLayout, runtime: RuntimeProfile
) -> dict[str, Path]:
    profile = _profile_for_host(layout)
    python, _result = _validate_recorded_runtime(layout, runtime, profile)
    return {"api": python, "gfm": python}


def resolve_llm(layout: RuntimeLayout) -> dict[str, str]:
    configuration = parse_private_environment(layout.llm_config_file)
    if configuration_state(configuration) != "complete":
        raise RuntimeError(
            "A verified LLM configuration is required. Run: "
            "python scripts/socialgraph.py configure-llm"
        )
    return configuration


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


def build_services(
    layout: RuntimeLayout,
    interpreters: dict[str, Path],
    *,
    ports: Ports,
) -> list[ServiceSpec]:
    common = _common_environment(layout)
    gfm_python = interpreters["gfm"]
    gfm_environment = dict(common)
    gfm_environment.update(
        {
            "PYTHONPATH": str(layout.gfm_package / "src"),
            "PYTHONNOUSERSITE": "1",
        }
    )
    gfm_arguments = (
        "-B",
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
        "--dataset-store-root",
        str(layout.dataset_store),
        "--token-file",
        str(layout.serving_token),
        "--host",
        "127.0.0.1",
        "--port",
        str(ports.gfm),
    )
    gfm = ServiceSpec(
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

    api_python = interpreters["api"]
    private = resolve_llm(layout)
    api_environment = dict(common)
    for name in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL"):
        api_environment[name] = private[name]
    config_stat = layout.llm_config_file.stat()
    api_environment.update(
        {
            "PYTHONPATH": "",
            "PYTHONNOUSERSITE": "1",
            "GFM_INFRASTRUCTURE_READY": "false",
            "SOCIALGRAPH_CORE_API_PORT": str(ports.api),
            "SOCIALGRAPH_WEB_CLIENT_ROOT": str(layout.web_client_root),
            "GFM_SERVICE_URL": f"http://127.0.0.1:{ports.gfm}",
            "GFM_SESSION_TOKEN_FILE": str(layout.serving_token),
            "GFM_CORE_SERVING_CONTROL_FILE": str(layout.serving_control),
            "GFM_CORE_RUN_BINDING_ROOT": str(layout.bindings_root),
            "GFM_RESEARCH_RUN_BINDING_ROOT": str(layout.research_bindings_root),
            "GFM_GLOBAL_MODEL_RUN_BINDING_ROOT": str(layout.global_model_bindings_root),
            "GFM_GLOBAL_MODEL_REVIEW_ROOT": str(layout.global_model_reviews_root),
            "GFM_GOVERNANCE_ROOT": str(layout.governance_root),
            "GFM_GOVERNANCE_BUNDLE_MAX_BYTES": "268435456",
            "GFM_GOVERNANCE_EXPANDED_MAX_BYTES": "1073741824",
            "GFM_CORE_SERVING_HIGH_WATER_ROOT": str(layout.serving_high_water_root),
            "DATASET_STORAGE_ROOT": str(layout.dataset_store),
            "ENABLE_TRUSTED_LOCAL_CONVERSION": "false",
            "TRUSTED_DATA_ROOTS": "",
            "TRUSTED_CONVERTER_PYTHON": "",
            "LOCAL_DEMO_LOOPBACK_ONLY": "true",
            "RUNTIME_BUILD_ID": "single-cpu-local",
        }
    )
    api = ServiceSpec(
        "socialgraph-api",
        ports.api,
        api_python,
        ("-B", "-m", "app", "--runtime-identity-root", str(layout.governance_root)),
        layout.api_root,
        api_environment,
        ("-m", "app", "--runtime-identity-root", str(layout.governance_root)),
        health_path="/api/v1/health",
        health_json={"status": "ok", "service": "socialgraph-fm-api"},
        generation=f"llm:{config_stat.st_mtime_ns}:{config_stat.st_size}",
    )
    return [gfm, api]


_NETWORK_DIAGNOSTIC_MESSAGES = {
    "LOCAL_ENDPOINT": "本地大模型地址没有服务监听，请启动兼容服务并检查端口。",
    "DNS": "无法解析大模型服务域名，请检查 API 地址和网络 DNS。",
    "CONNECT": "无法连接大模型服务，请检查网络、防火墙、API 地址和服务状态。",
    "TLS_HOSTNAME": (
        "服务证书域名与 API 地址不匹配，请检查地址和中转服务证书；"
        "不能绕过 TLS 验证。"
    ),
    "TLS_CERTIFICATE": (
        "服务证书不可信、已过期或证书链不完整，请检查系统时间和服务证书；"
        "不能绕过 TLS 验证。"
    ),
    "TLS_HANDSHAKE": (
        "TLS 握手失败，请检查中转服务的 TLS 配置和中间网络设备。"
    ),
    "PROTOCOL": (
        "服务在 HTTP 通信完成前中断连接，请检查中转服务的 TLS/HTTP 兼容性。"
    ),
    "PROXY": (
        "连接需要代理或代理连接失败；SocialGraph-FM 不继承系统代理，"
        "请使用可直连服务。"
    ),
    "NETWORK": "无法连接大模型服务，请检查域名、网络、防火墙和服务状态。",
}

_LLM_FAILURE_MESSAGES = {
    "LLM_AUTH_ERROR": ("AUTH", "认证失败，请检查 API Key。"),
    "LLM_ENDPOINT_ERROR": ("ENDPOINT", "接口地址不存在，请检查 API 地址。"),
    "LLM_REQUEST_REJECTED": (
        "REQUEST_OR_MODEL",
        "请求或模型被服务拒绝，请检查模型 ID 与账户权限。",
    ),
    "LLM_RATE_LIMITED": (
        "RATE_LIMIT",
        "服务触发限流，请稍后重试或检查账户额度。",
    ),
    "LLM_UPSTREAM_ERROR": ("UPSTREAM", "上游服务暂时不可用，请稍后重试。"),
    "LLM_TIMEOUT": ("TIMEOUT", "服务在 15 秒内未响应，请检查服务状态。"),
    "LLM_INVALID_RESPONSE": (
        "INVALID_RESPONSE",
        "服务返回了无效内容，请检查模型兼容性。",
    ),
    "LLM_RESPONSE_TOO_LARGE": (
        "INVALID_RESPONSE",
        "服务响应超过安全上限，请检查模型兼容性。",
    ),
    "LLM_CONFIGURATION_ERROR": (
        "CONFIGURATION",
        "配置无法用于连接检查，请重新输入三项配置。",
    ),
}


def _llm_connection_failure_message(code: str, diagnostic_code: object) -> str:
    if code == "LLM_NETWORK_ERROR":
        selected = (
            diagnostic_code
            if isinstance(diagnostic_code, str)
            and diagnostic_code in _NETWORK_DIAGNOSTIC_MESSAGES
            else "NETWORK"
        )
        detail = _NETWORK_DIAGNOSTIC_MESSAGES[selected]
        return f"大模型连接检查失败（{selected}）：{detail}"
    category, detail = _LLM_FAILURE_MESSAGES.get(
        code,
        (
            "INVALID_RESPONSE",
            "验证进程未返回可识别的安全结果，请检查本地安装。",
        ),
    )
    return f"大模型连接检查失败（{category}）：{detail}"


def test_llm_configuration(
    layout: RuntimeLayout,
    environment: dict[str, str],
    *,
    runtime_python: Path | None = None,
    api_python: Path | None = None,
) -> None:
    """Exercise the API package's real OpenAI-compatible provider chain."""

    if configuration_state(environment) != "complete":
        raise RuntimeError("A complete three-field LLM configuration is required")
    selected = runtime_python or api_python
    if selected is None:
        runtime = RuntimeProfile.load(layout.profile_file)
        selected = validate_profile(layout, runtime)["api"]
    selected = resolve_python(selected)
    child_environment = _cleared_llm_environment()
    for name in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL"):
        child_environment[name] = environment[name]
    child_environment["PYTHONNOUSERSITE"] = "1"
    child_environment["PYTHONPATH"] = ""
    try:
        completed = run_captured_process(
            [str(selected), "-B", "-m", "app.provider_check"],
            cwd=layout.api_root,
            environment=child_environment,
            timeout=25,
            description="LLM connection check",
        )
    except RuntimeError as error:
        code = (
            "LLM_TIMEOUT"
            if str(error) == "LLM connection check timed out after 25s"
            else "LLM_CONFIGURATION_ERROR"
        )
        raise RuntimeError(_llm_connection_failure_message(code, None)) from None
    except OSError:
        raise RuntimeError(
            _llm_connection_failure_message("LLM_CONFIGURATION_ERROR", None)
        ) from None
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
    valid_schema = bool(
        document is not None
        and document.get("schemaVersion")
        == "socialgraph-fm.llm-provider-check/1.0"
    )
    code = str(document.get("code", "")) if valid_schema and document else ""
    if completed.returncode != 0:
        if not valid_schema or document is None or document.get("ok") is not False:
            raise RuntimeError(
                "大模型连接检查失败（INVALID_RESPONSE）："
                "验证进程未返回可识别的安全结果，请检查本地安装。"
            )
        diagnostic_code = (
            document.get("diagnosticCode") if isinstance(document, dict) else None
        )
        raise RuntimeError(
            _llm_connection_failure_message(code, diagnostic_code)
        )
    if (
        document is None
        or not valid_schema
        or document.get("ok") is not True
        or code != "OK"
    ):
        raise RuntimeError(
            "大模型连接检查失败（INVALID_RESPONSE）："
            "服务响应不符合验证协议，请检查 API 地址和模型 ID。"
        )


def start_stack(layout: RuntimeLayout) -> Ports:
    runtime = RuntimeProfile.load(layout.profile_file)
    layout.initialize_directories()
    layout.initialize_serving_contracts()
    interpreters = validate_profile(layout, runtime)
    if not layout.web_client_root.joinpath("index.html").is_file():
        raise RuntimeError("Prebuilt Web client is missing; run onboard again")
    configuration = resolve_llm(layout)
    # Startup always checks the configured service; there is no offline mode.
    test_llm_configuration(layout, configuration, runtime_python=interpreters["api"])
    ports = Ports.environment()
    services = build_services(layout, interpreters, ports=ports)
    manager = ProcessManager(layout.pid_root, layout.log_root)
    started: list[str] = []
    try:
        for service in services:
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
    # governance-web is a retired third process, stopped here for migration.
    for name in ("governance-web", "socialgraph-api", "gfm"):
        manager.stop(name)
    if manager.read_record("gfm") is None:
        layout.serving_token.unlink(missing_ok=True)


def doctor(
    layout: RuntimeLayout, *, test_llm: bool = False, full: bool = False
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "required": True, "detail": detail})

    system, machine = normalized_platform()
    host = {
        "system": system,
        "machine": machine,
        "libc": platform.libc_ver()[0] or None,
        "libcVersion": platform.libc_ver()[1] or None,
    }
    try:
        bootstrap = ensure_bootstrap_python()
        report = probe_bootstrap_environment(bootstrap)
        add(
            "python-3.12",
            report.compatible,
            f"Python {report.fingerprint.get('pythonVersion')} at {bootstrap}",
        )
    except Exception as error:
        add("python-3.12", False, str(error))

    runtime: RuntimeProfile | None = None
    interpreters: dict[str, Path] = {}
    environment_summary: dict[str, Any] = {}
    try:
        runtime = RuntimeProfile.load(layout.profile_file)
        interpreters = validate_profile(layout, runtime)
        fingerprint = runtime.interpreter["fingerprint"]
        environment_summary = {
            "path": runtime.interpreter.get("path"),
            "source": "managed",
            "pythonVersion": fingerprint.get("pythonVersion"),
            "versions": fingerprint.get("versions"),
            "torch": fingerprint.get("torch"),
            "neighborLoader": fingerprint.get("neighborLoader"),
            "fingerprintSha256": fingerprint.get("fingerprintSha256"),
        }
        add(
            "managed-runtime",
            True,
            f"{runtime.install_profile_id}; Torch "
            f"{fingerprint.get('versions', {}).get('torch')}; CPU",
        )
    except Exception as error:
        add("managed-runtime", False, str(error))

    add(
        "prebuilt-web",
        layout.web_client_root.joinpath("index.html").is_file(),
        str(layout.web_client_root),
    )
    bundle = None
    if runtime is not None and interpreters:
        try:
            bundle = load_and_verify_bundle(layout)
            verified = verify_installed_runtime_bundle(layout, bundle)
            seeds = verify_installed_runtime_seeds(layout, bundle)
            state = verify_gfm_runtime_state(layout, interpreters["gfm"])
            add(
                "runtime-bundle",
                True,
                f"{verified['assetCount']} model assets; {sum(seeds.values())} seeds; "
                f"cases {str(state['reviewedCaseIndexHash'])[:12]}",
            )
        except Exception as error:
            add("runtime-bundle", False, str(error))
        if full:
            try:
                forward = run_full_gfm_probe(
                    layout,
                    interpreters["gfm"],
                    use_installed_model=True,
                )
                checkpoints = run_checkpoint_forward_probe(
                    layout, interpreters["gfm"]
                )
                add(
                    "cpu-forwards",
                    forward.get("passed") is True,
                    f"Russia={forward.get('bundleCount')}; "
                    f"protocols={len(checkpoints.get('protocols', []))}",
                )
            except Exception as error:
                add("cpu-forwards", False, str(error))

    try:
        if bundle is None:
            bundle = load_and_verify_bundle(layout)
        examples = verify_target_examples(layout, bundle)
        add(
            "target-examples",
            True,
            f"zero-shot={examples['zeroShot']['path']}; few-shot={examples['fewShot']['path']}",
        )
    except Exception as error:
        examples = {}
        add("target-examples", False, str(error))

    try:
        llm = resolve_llm(layout)
        if test_llm:
            test_llm_configuration(
                layout,
                llm,
                runtime_python=interpreters.get("api"),
            )
        add("llm-configuration", True, "required OpenAI-compatible API configured")
    except Exception as error:
        add("llm-configuration", False, str(error))

    manager = ProcessManager(layout.pid_root, layout.log_root)
    process_ready = True
    details: list[str] = []
    for name in ("socialgraph-api", "gfm"):
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

    return {
        "schemaVersion": "socialgraph-fm.doctor/3.0",
        "profile": "cpu" if runtime else None,
        "passed": all(check["passed"] for check in checks),
        "host": host,
        "installBackend": (
            {
                "profile": runtime.install_profile_id,
                "device": "cpu",
                "environmentCount": 1,
            }
            if runtime
            else {}
        ),
        "environment": environment_summary,
        "exampleUploads": examples,
        "checks": checks,
    }


def _environment_summaries(runtime: RuntimeProfile) -> dict[str, Any]:
    fingerprint = runtime.interpreter.get("fingerprint", {})
    summary = {
        "path": runtime.interpreter.get("path"),
        "source": "managed",
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
    return {"runtime": summary, "api": summary, "gfm": summary}
