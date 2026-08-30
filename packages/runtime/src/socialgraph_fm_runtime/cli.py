"""Public `socialgraph-fm` command-line interface."""

from __future__ import annotations

import argparse
import getpass
import json
import re
import sys
from pathlib import Path

from .exporter import export_public_snapshot
from .layout import RuntimeLayout
from .llm import (
    API_MODES,
    AUTH_SCHEMES,
    CONFIGURATION_TEST_INTENT_KEY,
    DEFAULT_ANTHROPIC_VERSION,
    configuration_summary,
    configuration_state,
    configure_environment,
    normalize_relay_api_base,
    parse_private_environment,
    write_private_environment,
)
from .operations import (
    SetupOptions,
    doctor,
    resolve_llm,
    setup,
    start_stack,
    stop_stack,
    test_llm_configuration,
)
from .profile import RuntimeProfile


def _add_start_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--llm-mode", choices=("optional", "required", "disabled"), default="optional"
    )
    parser.add_argument("--no-llm-prompt", action="store_true")
    parser.add_argument("--reconfigure-llm", action="store_true")
    parser.add_argument("--test-llm", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="socialgraph-fm")
    parser.add_argument("--project-root", type=Path, default=None, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    setup_parser = commands.add_parser("setup", help="prepare Python, Web, and model runtime")
    setup_parser.add_argument("--profile", choices=("offline", "cpu", "cuda"), default=None)
    setup_parser.add_argument(
        "--wheel-profile",
        help="verified wheel family alias or exact install-profile catalog ID",
    )
    setup_parser.add_argument(
        "--device-policy",
        choices=("auto", "cpu", "cuda-required"),
        default="auto",
    )
    setup_parser.add_argument(
        "--env-mode", choices=("auto", "reuse", "managed"), default="auto"
    )
    setup_parser.add_argument("--api-python")
    setup_parser.add_argument("--gfm-python")
    setup_parser.add_argument("--bootstrap-python")
    setup_parser.add_argument("--skip-api", action="store_true")
    setup_parser.add_argument("--skip-web", action="store_true")
    setup_parser.add_argument("--gfm-text", dest="gfm_text_profile", action="store_true")
    setup_parser.add_argument(
        "--gfm-text-profile",
        dest="gfm_text_profile",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    configure = commands.add_parser(
        "configure-llm", help="configure an OpenAI- or Anthropic-compatible API"
    )
    configure.add_argument("--preset")
    configure.add_argument("--api-base")
    configure.add_argument("--model")
    configure.add_argument("--api-mode", "--protocol", dest="api_mode", choices=API_MODES)
    configure.add_argument("--auth-scheme", choices=AUTH_SCHEMES)
    configure.add_argument("--anthropic-version")
    configure.add_argument("--timeout-seconds", type=int, default=15)
    configure.add_argument("--api-key-stdin", action="store_true")
    configure.add_argument("--allow-insecure-loopback", action="store_true")
    testing = configure.add_mutually_exclusive_group()
    testing.add_argument("--test-llm", dest="test_llm", action="store_true")
    testing.add_argument("--skip-llm-test", dest="test_llm", action="store_false")
    configure.set_defaults(test_llm=None)

    onboard = commands.add_parser(
        "onboard", help="configure LLM first, then prepare verified model and Web runtimes"
    )
    onboard.add_argument("--wheel-profile")
    onboard.add_argument(
        "--device-policy",
        choices=("auto", "cpu", "cuda-required"),
        default="auto",
    )
    onboard.add_argument(
        "--env-mode", choices=("auto", "reuse", "managed"), default="auto"
    )
    onboard.add_argument("--api-python")
    onboard.add_argument("--gfm-python")
    onboard.add_argument("--bootstrap-python")
    onboard.add_argument("--skip-web", action="store_true")
    onboard.add_argument("--gfm-text", dest="gfm_text_profile", action="store_true")
    onboard.add_argument("--preset")
    onboard.add_argument("--api-base")
    onboard.add_argument("--model")
    onboard.add_argument("--api-mode", "--protocol", dest="api_mode", choices=API_MODES)
    onboard.add_argument("--auth-scheme", choices=AUTH_SCHEMES)
    onboard.add_argument("--anthropic-version")
    onboard.add_argument("--timeout-seconds", type=int, default=15)
    onboard.add_argument("--api-key-stdin", action="store_true")
    onboard.add_argument("--allow-insecure-loopback", action="store_true")

    development = commands.add_parser("dev", help="start the development stack")
    _add_start_arguments(development)
    production = commands.add_parser("start", help="build and start the production stack")
    _add_start_arguments(production)
    commands.add_parser("stop", help="stop managed local services")
    doctor_parser = commands.add_parser("doctor", help="inspect runtime compatibility")
    doctor_parser.add_argument("--test-llm", action="store_true")
    doctor_parser.add_argument("--full", action="store_true")
    doctor_parser.add_argument("--json", action="store_true")
    export = commands.add_parser(
        "export-github", help="create a clean single-commit GitHub repository and ZIP"
    )
    export.add_argument("--repository", type=Path, required=True)
    export.add_argument("--zip", dest="zip_destination", type=Path, required=True)
    export.add_argument("--message", default=None)
    return parser


def _default_configuration_arguments() -> argparse.Namespace:
    return argparse.Namespace(
        preset=None,
        api_base=None,
        model=None,
        api_mode=None,
        auth_scheme=None,
        anthropic_version=None,
        timeout_seconds=15,
        api_key_stdin=False,
        allow_insecure_loopback=False,
        test_llm=None,
    )


def _configuration_value(name: str, value: str) -> str:
    if not value or value != value.strip() or re.search(r"[\x00-\x1f\x7f]", value):
        raise RuntimeError(f"{name} must be non-empty single-line text")
    return value


def _print_configuration_summary(environment: dict[str, str]) -> None:
    summary = configuration_summary(environment)
    print("Updated LLM configuration (API key hidden):")
    for label, key in (
        ("API Base", "apiBase"),
        ("Endpoint", "endpoint"),
        ("Protocol", "apiMode"),
        ("Authentication", "authScheme"),
        ("Anthropic version", "anthropicVersion"),
        ("Model", "model"),
        ("Timeout seconds", "timeoutSeconds"),
    ):
        value = summary.get(key)
        if value is not None:
            print(f"  {label}: {value}")
    print("  API key: configured (hidden)")


def _edit_configuration(environment: dict[str, str], choice: str) -> None:
    selected = choice.lower()
    if selected == "b":
        current = environment["LLM_API_BASE"]
        value = input(f"API Base [{current}]: ").strip() or current
        environment["LLM_API_BASE"] = normalize_relay_api_base(
            value,
            allow_insecure_loopback=(
                environment.get("LLM_ALLOW_INSECURE_LOOPBACK") == "true"
            ),
        )
    elif selected == "m":
        current = environment["LLM_MODEL"]
        value = input(f"Model ID [{current}]: ").strip() or current
        environment["LLM_MODEL"] = _configuration_value("Model ID", value)
    elif selected == "p":
        print("Protocol: 1) chat_completions  2) responses  3) anthropic_messages")
        raw = input(f"Protocol [{environment['LLM_API_MODE']}]: ").strip()
        aliases = {"1": "chat_completions", "2": "responses", "3": "anthropic_messages"}
        mode = aliases.get(raw, raw or environment["LLM_API_MODE"])
        if mode not in API_MODES:
            raise RuntimeError("Invalid LLM protocol")
        environment["LLM_API_MODE"] = mode
        if mode == "anthropic_messages":
            environment["LLM_ANTHROPIC_VERSION"] = (
                environment.get("LLM_ANTHROPIC_VERSION") or DEFAULT_ANTHROPIC_VERSION
            )
            environment["LLM_AUTH_SCHEME"] = "x-api-key"
        else:
            environment["LLM_ANTHROPIC_VERSION"] = ""
            environment["LLM_AUTH_SCHEME"] = "bearer"
    elif selected == "a":
        raw = input(
            f"Authentication bearer|x-api-key [{environment['LLM_AUTH_SCHEME']}]: "
        ).strip()
        scheme = raw or environment["LLM_AUTH_SCHEME"]
        if scheme not in AUTH_SCHEMES:
            raise RuntimeError("Invalid LLM authentication scheme")
        environment["LLM_AUTH_SCHEME"] = scheme
    elif selected == "k":
        environment["LLM_API_KEY"] = _configuration_value(
            "API Key", getpass.getpass("API Key: ")
        )
    else:
        raise RuntimeError("Unknown LLM configuration edit")
    environment["LLM_VERIFICATION_STATUS"] = "configured_unverified"
    _print_configuration_summary(environment)


def _configure(
    layout: RuntimeLayout,
    arguments: argparse.Namespace | None = None,
    *,
    api_python: Path | None = None,
) -> None:
    layout.initialize_config_directory()
    if arguments is None:
        arguments = _default_configuration_arguments()
    while True:
        environment = configure_environment(
            preset_catalog=layout.project_root / "scripts" / "config" / "llm-presets.json",
            preset=getattr(arguments, "preset", None),
            api_base=getattr(arguments, "api_base", None),
            model=getattr(arguments, "model", None),
            api_mode=getattr(arguments, "api_mode", None),
            auth_scheme=getattr(arguments, "auth_scheme", None),
            anthropic_version=getattr(arguments, "anthropic_version", None),
            timeout_seconds=getattr(arguments, "timeout_seconds", 15),
            api_key_stdin=getattr(arguments, "api_key_stdin", False),
            allow_insecure_loopback=getattr(arguments, "allow_insecure_loopback", False),
            stdin=sys.stdin,
            stdout=sys.stdout,
            test_llm=getattr(arguments, "test_llm", None),
        )
        requested_marker = environment.pop(CONFIGURATION_TEST_INTENT_KEY, None)
        explicitly_requested = getattr(arguments, "test_llm", None)
        should_test = (
            explicitly_requested
            if explicitly_requested is not None
            else requested_marker != "false"
        )
        reenter = False
        while should_test:
            try:
                test_llm_configuration(
                    layout,
                    environment,
                    api_python=api_python,
                )
                environment["LLM_VERIFICATION_STATUS"] = "call_succeeded"
                should_test = False
            except RuntimeError as error:
                if not sys.stdin.isatty():
                    raise
                print(str(error), file=sys.stderr)
                choice = input(
                    "Edit [B]ase/[M]odel/[P]rotocol/[A]uth/[K]ey, "
                    "[R]etry, re-[E]nter, [S]ave unverified, [Q]uit [R]: "
                ).strip().lower() or "r"
                if choice in {"b", "m", "p", "a", "k"}:
                    _edit_configuration(environment, choice)
                    continue
                if choice == "r":
                    continue
                if choice == "e":
                    reenter = True
                    break
                if choice == "s":
                    should_test = False
                    break
                raise RuntimeError("LLM configuration cancelled") from None
        if reenter:
            allow_insecure_loopback = getattr(
                arguments, "allow_insecure_loopback", False
            )
            arguments = _default_configuration_arguments()
            arguments.allow_insecure_loopback = allow_insecure_loopback
            continue
        break
    write_private_environment(layout.llm_config_file, environment)
    print(
        "LLM configuration saved: "
        + ("verified" if environment["LLM_VERIFICATION_STATUS"] == "call_succeeded" else "unverified"),
        flush=True,
    )


def _start(layout: RuntimeLayout, arguments: argparse.Namespace, *, development: bool) -> None:
    if not layout.profile_file.is_file():
        raise RuntimeError(
            "onboarding is required before start. Run:\n"
            "  python scripts/socialgraph.py onboard\n"
            "or use the compatible explicit workflow:\n"
            "  python scripts/socialgraph.py setup --profile offline --skip-web\n"
            "  python scripts/socialgraph.py configure-llm\n"
            "  python scripts/socialgraph.py setup --wheel-profile cpu|cuda\n"
            "  python scripts/socialgraph.py start --llm-mode required"
        )
    if arguments.reconfigure_llm:
        _configure(layout)
    enabled = resolve_llm(layout, arguments.llm_mode, no_prompt=arguments.no_llm_prompt)
    if enabled == "configure":
        _configure(layout)
        enabled = True
    if arguments.test_llm:
        environment = parse_private_environment(layout.llm_config_file)
        if configuration_state(environment) != "complete":
            raise RuntimeError("A complete LLM configuration is required for --test-llm")
        try:
            test_llm_configuration(layout, environment)
            environment["LLM_VERIFICATION_STATUS"] = "call_succeeded"
        except RuntimeError:
            environment["LLM_VERIFICATION_STATUS"] = "fallback"
            write_private_environment(layout.llm_config_file, _configuration_defaults(environment))
            raise
        write_private_environment(layout.llm_config_file, _configuration_defaults(environment))
    ports = start_stack(layout, development=development, enable_llm=bool(enabled))
    print(f"Governance: http://127.0.0.1:{ports.web}")
    print(f"API: http://127.0.0.1:{ports.api}")
    runtime = RuntimeProfile.load(layout.profile_file)
    print(
        "GFM: offline profile"
        if runtime.profile == "offline"
        else f"GFM loopback: http://127.0.0.1:{ports.gfm}"
    )
    print(f"LLM: {'configured for the API process' if enabled else 'deterministic fallback'}")
    print(f"Logs: {layout.log_root}")


def _configuration_defaults(environment: dict[str, str]) -> dict[str, str]:
    selected = dict(environment)
    mode = selected.get("LLM_API_MODE") or "chat_completions"
    selected["LLM_API_MODE"] = mode
    selected.setdefault(
        "LLM_AUTH_SCHEME", "x-api-key" if mode == "anthropic_messages" else "bearer"
    )
    selected.setdefault(
        "LLM_ANTHROPIC_VERSION",
        DEFAULT_ANTHROPIC_VERSION if mode == "anthropic_messages" else "",
    )
    selected.setdefault("LLM_TIMEOUT_SECONDS", "15")
    selected.setdefault("LLM_ALLOW_INSECURE_LOOPBACK", "false")
    selected.setdefault("LLM_VERIFICATION_STATUS", "configured_unverified")
    return selected


def _select_setup_profile(value: str | None) -> str:
    if value is not None:
        return value
    if not sys.stdin.isatty():
        raise RuntimeError(
            "setup requires an explicit --wheel-profile cpu|cuda|ID (or legacy "
            "--profile cpu|cuda|offline) in a non-interactive session"
        )
    print("Select the verified SocialGraph-FM GFM wheel family:", file=sys.stderr)
    print(
        "  A fresh selection may download several hundred MiB of PyTorch/PyG wheels.",
        file=sys.stderr,
    )
    print("  1) CPU wheel (default)", file=sys.stderr)
    print("  2) CUDA wheel", file=sys.stderr)
    print("  3) Offline", file=sys.stderr)
    print("Profile [CPU]: ", end="", file=sys.stderr, flush=True)
    selected = sys.stdin.readline()
    if selected == "":
        raise RuntimeError("GFM wheel selection was cancelled")
    normalized = selected.strip().lower()
    aliases = {
        "": "cpu",
        "1": "cpu",
        "cpu": "cpu",
        "2": "cuda",
        "cuda": "cuda",
        "3": "offline",
        "offline": "offline",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise RuntimeError("Choose the CPU wheel, CUDA wheel, or Offline") from error


def _setup_document(layout: RuntimeLayout, runtime: RuntimeProfile) -> dict[str, object]:
    setup_summary = runtime.setup_summary or {}
    return {
        "schemaVersion": runtime.to_document()["schemaVersion"],
        "profile": runtime.profile,
        "envMode": runtime.env_mode,
        "devicePolicy": runtime.device_policy,
        "installProfileId": runtime.install_profile_id,
        "profileFile": str(layout.profile_file),
        "setupLog": str(layout.setup_log_file),
        "exampleUploads": setup_summary.get("exampleUploads"),
        "protocolForward": setup_summary.get("protocolForward"),
        "cpuFallbackForward": setup_summary.get("cpuFallbackForward"),
        "deviceResolution": setup_summary.get("deviceResolution"),
    }


def _setup_selection(arguments: argparse.Namespace) -> str:
    legacy = getattr(arguments, "profile", None)
    wheel = getattr(arguments, "wheel_profile", None)
    if legacy is not None and wheel is not None:
        raise RuntimeError("Use either --profile or --wheel-profile, not both")
    return _select_setup_profile(wheel or legacy)


def _onboard(layout: RuntimeLayout, arguments: argparse.Namespace) -> RuntimeProfile:
    interactive = sys.stdin.isatty()
    if not interactive:
        missing = [
            flag
            for flag, value in (
                ("--wheel-profile", getattr(arguments, "wheel_profile", None)),
                ("--preset", getattr(arguments, "preset", None)),
                ("--model", getattr(arguments, "model", None)),
                ("--api-key-stdin", getattr(arguments, "api_key_stdin", False)),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(
                "non-interactive onboard requires explicit " + ", ".join(missing)
            )

    def configure_then_select(api_python: Path) -> str:
        print(
            "\nConfigure the model API before any large Torch/PyG download.\n",
            flush=True,
        )
        configuration_arguments = argparse.Namespace(
            preset=getattr(arguments, "preset", None),
            api_base=getattr(arguments, "api_base", None),
            model=getattr(arguments, "model", None),
            api_mode=getattr(arguments, "api_mode", None),
            auth_scheme=getattr(arguments, "auth_scheme", None),
            anthropic_version=getattr(arguments, "anthropic_version", None),
            timeout_seconds=getattr(arguments, "timeout_seconds", 15),
            api_key_stdin=getattr(arguments, "api_key_stdin", False),
            allow_insecure_loopback=getattr(
                arguments, "allow_insecure_loopback", False
            ),
            # Automated onboarding is a release gate and always performs the
            # real provider check. The interactive wizard asks before calling.
            test_llm=True if not interactive else None,
        )
        _configure(layout, configuration_arguments, api_python=api_python)
        print(
            "\nSelect a verified GFM wheel family. Compatible environments are reused.\n",
            flush=True,
        )
        return _select_setup_profile(getattr(arguments, "wheel_profile", None))

    return setup(
        layout,
        SetupOptions(
            profile="auto",
            env_mode=arguments.env_mode,
            device_policy=arguments.device_policy,
            api_python=arguments.api_python,
            gfm_python=arguments.gfm_python,
            bootstrap_python=arguments.bootstrap_python,
            skip_api=False,
            skip_web=arguments.skip_web,
            gfm_text_profile=arguments.gfm_text_profile,
            after_api=configure_then_select,
        ),
    )


def run(arguments: argparse.Namespace) -> int:
    layout = RuntimeLayout.discover(arguments.project_root)
    if arguments.command == "setup":
        selected_profile = _setup_selection(arguments)
        runtime = setup(
            layout,
            SetupOptions(
                profile=selected_profile,
                env_mode=arguments.env_mode,
                device_policy=arguments.device_policy,
                api_python=arguments.api_python,
                gfm_python=arguments.gfm_python,
                bootstrap_python=arguments.bootstrap_python,
                skip_api=arguments.skip_api,
                skip_web=arguments.skip_web,
                gfm_text_profile=arguments.gfm_text_profile,
            ),
        )
        print(json.dumps(_setup_document(layout, runtime), sort_keys=True))
        return 0
    if arguments.command == "onboard":
        runtime = _onboard(layout, arguments)
        print(json.dumps(_setup_document(layout, runtime), sort_keys=True))
        print("Onboarding complete. Start with:", flush=True)
        print(
            "  python scripts/socialgraph.py start --llm-mode required", flush=True
        )
        return 0
    if arguments.command == "configure-llm":
        _configure(layout, arguments)
        return 0
    if arguments.command == "dev":
        _start(layout, arguments, development=True)
        return 0
    if arguments.command == "start":
        _start(layout, arguments, development=False)
        return 0
    if arguments.command == "stop":
        stop_stack(layout)
        return 0
    if arguments.command == "doctor":
        document = doctor(layout, test_llm=arguments.test_llm, full=arguments.full)
        if arguments.json:
            print(json.dumps(document, ensure_ascii=False, sort_keys=True))
        else:
            for check in document["checks"]:
                marker = "PASS" if check["passed"] else "FAIL"
                print(f"[{marker}] {check['name']}: {check['detail']}")
        return 0 if document["passed"] else 2
    if arguments.command == "export-github":
        keywords = {}
        if arguments.message is not None:
            keywords["commit_message"] = arguments.message
        result = export_public_snapshot(
            layout.project_root,
            arguments.repository,
            arguments.zip_destination,
            **keywords,
        )
        print(json.dumps(result.to_document(), ensure_ascii=False, sort_keys=True))
        return 0
    raise RuntimeError(f"Unsupported command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return run(arguments)
    except (RuntimeError, ValueError, OSError) as error:
        print(f"socialgraph-fm: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
