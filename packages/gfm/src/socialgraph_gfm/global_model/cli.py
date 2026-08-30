"""Command-line workflow for the isolated SocialGraph-FM Global release."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_json
from socialgraph_gfm.runtime import artifact_root

from .config import ProtocolId, global_model_root_from_home
from .forward_smoke import run_checkpoint_forward_smoke
from .workflow import (
    PROTOCOLS,
    convert_global_model_corpus,
    evaluate_global_model_protocol,
    export_global_model_release,
    publish_global_model_release,
    smoke_global_model_export,
    train_global_model_protocol,
    validate_global_model_corpus,
    verify_global_model_export,
)


def _root(value: str | None) -> Path:
    return (
        Path(value).expanduser().resolve()
        if value is not None
        else global_model_root_from_home(artifact_root())
    )


def _protocols(value: str) -> tuple[ProtocolId, ...]:
    return PROTOCOLS if value == "all" else (value,)  # type: ignore[return-value]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="socialgraph-gfm-global")
    commands = parser.add_subparsers(dest="command", required=True)

    convert = commands.add_parser("convert")
    convert.add_argument("--root")
    convert.add_argument("--source-root", required=True)
    convert.add_argument("--trusted-source", action="store_true", required=True)
    convert.add_argument("--all-regimes", action="store_true")

    validate = commands.add_parser("validate")
    validate.add_argument("--root")

    train = commands.add_parser("train")
    train.add_argument("--root")
    train.add_argument("--protocol", choices=(*PROTOCOLS, "all"), default="all")
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train.add_argument("--fast", action="store_true")
    train.add_argument("--no-resume", action="store_true")

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--root")
    evaluate.add_argument("--protocol", choices=(*PROTOCOLS, "all"), default="all")
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--fast", action="store_true")

    export = commands.add_parser("export")
    export.add_argument("--root")
    export.add_argument("--fast", action="store_true")

    smoke = commands.add_parser("smoke")
    smoke.add_argument("--root")
    smoke.add_argument("--fast", action="store_true")
    smoke.add_argument("--in-process", action="store_true")

    forward_smoke = commands.add_parser(
        "forward-smoke",
        help="run a read-only real forward through all four published checkpoints",
    )
    forward_smoke.add_argument("--root")
    forward_smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")

    publish = commands.add_parser("publish")
    publish.add_argument("--root")

    verify = commands.add_parser("_verify-export", help=argparse.SUPPRESS)
    verify.add_argument("--root", required=True)
    verify.add_argument("--fast", action="store_true")
    return parser


def _execute(arguments: argparse.Namespace) -> dict[str, Any]:
    root = _root(arguments.root)
    if arguments.command == "convert":
        path = convert_global_model_corpus(
            source_root=arguments.source_root,
            root=root,
            trusted_source=arguments.trusted_source,
            include_all_regimes=arguments.all_regimes,
        )
        return {"command": "convert", "root": str(root), "manifest": str(path)}
    if arguments.command == "validate":
        path = validate_global_model_corpus(root)
        return {"command": "validate", "root": str(root), "report": str(path)}
    if arguments.command == "train":
        artifacts = {
            protocol: str(
                train_global_model_protocol(
                    root,
                    protocol=protocol,
                    device=arguments.device,
                    resume=not arguments.no_resume,
                    fast=arguments.fast,
                )
            )
            for protocol in _protocols(arguments.protocol)
        }
        return {"command": "train", "root": str(root), "artifacts": artifacts}
    if arguments.command == "evaluate":
        artifacts = {
            protocol: str(
                evaluate_global_model_protocol(
                    root,
                    protocol=protocol,
                    device=arguments.device,
                    fast=arguments.fast,
                )
            )
            for protocol in _protocols(arguments.protocol)
        }
        return {"command": "evaluate", "root": str(root), "artifacts": artifacts}
    if arguments.command == "export":
        path = export_global_model_release(root, fast=arguments.fast)
        return {"command": "export", "root": str(root), "manifest": str(path)}
    if arguments.command == "smoke":
        path = smoke_global_model_export(
            root,
            fast=arguments.fast,
            fresh_process=not arguments.in_process,
        )
        return {"command": "smoke", "root": str(root), "report": str(path)}
    if arguments.command == "forward-smoke":
        report = run_checkpoint_forward_smoke(root, device=arguments.device)
        return {"command": "forward-smoke", "root": str(root), **report}
    if arguments.command == "publish":
        path = publish_global_model_release(root)
        return {"command": "publish", "root": str(root), "registry": str(path)}
    if arguments.command == "_verify-export":
        return verify_global_model_export(root, fast=arguments.fast)
    raise AssertionError(f"unsupported Global command: {arguments.command}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _execute(arguments)
    except Exception as exc:  # noqa: BLE001 - CLI emits one stable JSON failure envelope.
        print(
            canonical_json(
                {
                    "ok": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(canonical_json({"ok": True, **result}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
