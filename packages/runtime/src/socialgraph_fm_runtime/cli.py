"""Public five-command SocialGraph-FM interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .layout import RuntimeLayout
from .llm import (
    configuration_summary,
    configure_environment,
    write_private_environment,
)
from .operations import (
    doctor,
    setup,
    start_stack,
    stop_stack,
    test_llm_configuration,
)
from .profile import RuntimeProfile


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api-base", help="OpenAI-compatible API root or endpoint")
    parser.add_argument("--model", help="model ID")
    parser.add_argument(
        "--api-key-stdin",
        action="store_true",
        help="read the API key from one stdin line (automation only)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="socialgraph-fm")
    parser.add_argument("--project-root", type=Path, default=None, help=argparse.SUPPRESS)
    commands = parser.add_subparsers(dest="command", required=True)

    onboard = commands.add_parser(
        "onboard",
        help="configure the required LLM and install the single CPU runtime",
    )
    _add_llm_arguments(onboard)

    configure = commands.add_parser(
        "configure-llm",
        help="replace and verify the three-field LLM configuration",
    )
    _add_llm_arguments(configure)

    commands.add_parser("start", help="verify LLM and start API/Web plus GFM")
    commands.add_parser("stop", help="stop both managed processes")
    doctor_parser = commands.add_parser("doctor", help="inspect the managed runtime")
    doctor_parser.add_argument("--test-llm", action="store_true")
    doctor_parser.add_argument("--full", action="store_true")
    doctor_parser.add_argument("--json", action="store_true")
    return parser


def _collect_configuration(arguments: argparse.Namespace) -> dict[str, str]:
    return configure_environment(
        api_base=arguments.api_base,
        model=arguments.model,
        api_key_stdin=arguments.api_key_stdin,
        stdin=sys.stdin,
        stdout=sys.stdout,
    )


def _save_verified_configuration(
    layout: RuntimeLayout,
    environment: dict[str, str],
    *,
    runtime_python: Path | None = None,
) -> None:
    test_llm_configuration(
        layout,
        environment,
        runtime_python=runtime_python,
    )
    write_private_environment(layout.llm_config_file, environment)
    summary = configuration_summary(environment)
    print("大模型 API 配置已验证并安全保存（API Key 已隐藏）：")
    print(f"  API 地址：{summary['apiBase']}")
    print(f"  模型 ID：{summary['model']}")


def _onboard(
    layout: RuntimeLayout, arguments: argparse.Namespace
) -> RuntimeProfile:
    environment = _collect_configuration(arguments)
    runtime = setup(layout)
    print(
        "[setup] LLM: validating the required three-field configuration",
        file=sys.stderr,
        flush=True,
    )
    runtime_python = Path(str(runtime.interpreter["path"]))
    try:
        _save_verified_configuration(
            layout,
            environment,
            runtime_python=runtime_python,
        )
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"{error}\n"
            "本地 CPU 环境和模型已完成并保留。\n"
            "候选 LLM 配置未保存。\n"
            "请修正后运行：python scripts/socialgraph.py configure-llm"
        ) from None
    print(
        "[setup] complete: CPU runtime, Web client, assets, and LLM are ready",
        file=sys.stderr,
        flush=True,
    )
    return runtime


def run(arguments: argparse.Namespace) -> int:
    layout = RuntimeLayout.discover(arguments.project_root)
    if arguments.command == "onboard":
        runtime = _onboard(layout, arguments)
        print(
            json.dumps(
                {
                    "schemaVersion": runtime.to_document()["schemaVersion"],
                    "runtime": str(layout.runtime_environment),
                    "installProfileId": runtime.install_profile_id,
                    "device": "cpu",
                    "processes": ["api-web", "gfm"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "configure-llm":
        environment = _collect_configuration(arguments)
        _save_verified_configuration(layout, environment)
        return 0
    if arguments.command == "start":
        if not layout.profile_file.is_file():
            raise RuntimeError(
                "请先完成初始化：python scripts/socialgraph.py onboard"
            )
        ports = start_stack(layout)
        print(f"SocialGraph-FM：http://127.0.0.1:{ports.web}")
        print(f"GFM 内部服务：http://127.0.0.1:{ports.gfm}")
        print("LLM：已验证，仅注入 API 进程")
        print(f"日志：{layout.log_root}")
        return 0
    if arguments.command == "stop":
        stop_stack(layout)
        print("SocialGraph-FM 已停止")
        return 0
    if arguments.command == "doctor":
        report = doctor(
            layout,
            test_llm=arguments.test_llm,
            full=arguments.full,
        )
        if arguments.json:
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        else:
            for check in report["checks"]:
                marker = "PASS" if check["passed"] else "FAIL"
                print(f"[{marker}] {check['name']}: {check['detail']}")
        return 0 if report["passed"] else 1
    raise RuntimeError(f"Unsupported command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    # GitHub's English Windows runners expose a legacy code page even though
    # this Chinese-first CLI writes through a Unicode-capable terminal. Force a
    # stable native encoding so diagnostics never fail while formatting text.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="backslashreplace")
    try:
        return run(build_parser().parse_args(argv))
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
