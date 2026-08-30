"""Strict local CLI for core experiment, acceptance, and promotion evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal, Sequence

from .bundle import load_core_graph_bundle_json
from .formal_preflight import FormalPreflightEvidence, run_formal_preflight
from .local_experiments import (
    load_email_local_inputs,
    load_penn94_local_inputs,
    run_local_nonpromotable_experiment,
)
from .structure_features import build_structure_cache


def _existing_runtime(value: str) -> Path:
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise argparse.ArgumentTypeError("runtime root must be an existing directory")
    return path


def _output_in_runtime(runtime: Path, value: Path) -> Path:
    output = value.resolve()
    try:
        output.relative_to(runtime)
    except ValueError as error:
        raise SystemExit("output must remain inside the explicit runtime root") from error
    return output


def _input_in_runtime(runtime: Path, value: Path) -> Path:
    source = value.resolve(strict=True)
    try:
        source.relative_to(runtime)
    except ValueError as error:
        raise SystemExit("input must remain inside the explicit runtime root") from error
    if not source.is_file():
        raise SystemExit("input must be a regular file")
    return source


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-root", type=_existing_runtime, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="socialgraph-gfm-core-experiment",
        description="Derive hash-bound core experiment evidence without implicit downloads",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser("preflight")
    _add_runtime(preflight)
    preflight.add_argument("--output", type=Path, required=True)

    cache = commands.add_parser("structure-cache")
    _add_runtime(cache)
    cache.add_argument("--bundle", type=Path, required=True)
    cache.add_argument("--role", choices=("training", "inference"), required=True)

    for name in ("email-smoke", "penn-dev"):
        command = commands.add_parser(name)
        _add_runtime(command)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--optimizer-steps", type=int)
        command.add_argument("--head-steps", type=int, default=4)
        command.add_argument("--seed", type=int, default=20260821)
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    aggregate = commands.add_parser("aggregate")
    _add_runtime(aggregate)
    aggregate.add_argument("--ledger-root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)

    accept = commands.add_parser("accept")
    _add_runtime(accept)
    accept.add_argument("--request", type=Path, required=True)
    accept.add_argument("--output", type=Path, required=True)

    smoke = commands.add_parser("serving-smoke")
    _add_runtime(smoke)
    smoke.add_argument("--accepted-candidate", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)

    promote = commands.add_parser("promote")
    _add_runtime(promote)
    promote.add_argument("--accepted-candidate", type=Path, required=True)
    promote.add_argument("--serving-smoke", type=Path, required=True)
    promote.add_argument("--serving-control", type=Path, required=True)

    readiness = commands.add_parser("readiness")
    _add_runtime(readiness)
    readiness.add_argument("--serving-control", type=Path, required=True)
    readiness.add_argument("--output", type=Path, required=True)
    return parser


def _preflight(arguments: argparse.Namespace) -> int:
    runtime: Path = arguments.runtime_root
    output = _output_in_runtime(runtime, arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = run_formal_preflight(runtime, publish_to=output)
    print(
        json.dumps(
            {
                "evidenceHash": evidence.evidence_hash,
                "formalReady": evidence.formal_ready,
                "output": str(output),
                "promotable": evidence.promotable,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def _structure_cache(arguments: argparse.Namespace) -> int:
    runtime: Path = arguments.runtime_root
    bundle_path = _input_in_runtime(runtime, arguments.bundle)
    bundle = load_core_graph_bundle_json(bundle_path.read_bytes())
    cache = build_structure_cache(
        bundle,
        cache_root=runtime / "experiments-core" / "cache",
        role=arguments.role,
    )
    manifest = cache.manifest
    print(
        json.dumps(
            {
                "artifactId": manifest.artifact_id,
                "baseGraphVersionHash": bundle.graph_version_hash,
                "manifestHash": manifest.manifest_hash,
                "manifestPath": str(cache.manifest_path.resolve()),
                "role": arguments.role,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def _require_local_run_assets(command: str, runtime: Path) -> None:
    if command == "email-smoke":
        required = (
            runtime / "materialized" / "email-eu-core" / "1.0.0" / "bundle.json",
            runtime / "materialized" / "email-eu-core" / "1.0.0" / "materialization-manifest.json",
            runtime / "raw" / "email-eu-core" / "1.0.0" / "email-Eu-core.txt.gz",
            runtime / "raw" / "email-eu-core" / "1.0.0" / "email-Eu-core-department-labels.txt.gz",
        )
    else:
        required = (
            runtime / "raw" / "facebook100" / "1.0.0" / "Penn94.mat",
            runtime / "raw" / "facebook100" / "1.0.0" / "fb100-Penn94-splits.npy",
            runtime
            / "derived"
            / "facebook100"
            / "penn94-official-splits"
            / "1.0.0"
            / "penn94-official-splits-safe.npz",
            runtime
            / "derived"
            / "facebook100"
            / "penn94-official-splits"
            / "1.0.0"
            / "conversion-manifest.json",
        )
    if any(not path.is_file() for path in required):
        raise SystemExit(f"{command} requires validated local dataset assets; no download occurs")


def _local_run(arguments: argparse.Namespace) -> int:
    runtime: Path = arguments.runtime_root
    output = _output_in_runtime(runtime, arguments.output)
    _require_local_run_assets(arguments.command, runtime)
    preflight_path = runtime / "experiments-core" / "formal-preflight-v2-current.json"
    if not preflight_path.is_file():
        raise SystemExit("local runs require the current hash-bound formal preflight evidence")
    preflight = FormalPreflightEvidence.model_validate_json(preflight_path.read_bytes())
    dataset_id: Literal["email-eu-core", "penn94"]
    phase: Literal["smoke", "dev"]
    task_kind: Literal["node-binary", "edge-binary"]
    if arguments.command == "email-smoke":
        inputs = load_email_local_inputs(runtime)
        dataset_id = "email-eu-core"
        phase = "smoke"
        task_kind = "edge-binary"
        optimizer_steps = 20 if arguments.optimizer_steps is None else arguments.optimizer_steps
    else:
        inputs = load_penn94_local_inputs(runtime)
        dataset_id = "penn94"
        phase = "dev"
        task_kind = "node-binary"
        optimizer_steps = 1 if arguments.optimizer_steps is None else arguments.optimizer_steps
    report = run_local_nonpromotable_experiment(
        bundle=inputs.bundle,
        dataset_id=dataset_id,
        phase=phase,
        task_kind=task_kind,
        targets_by_entity=inputs.targets_by_entity,
        split_inventory=inputs.split_inventory,
        source_inventory=inputs.source_inventory,
        runtime_root=runtime,
        output_path=output,
        seed=arguments.seed,
        optimizer_steps=optimizer_steps,
        head_steps=arguments.head_steps,
        device_name=arguments.device,
        formal_preflight_evidence_hash=preflight.evidence_hash,
        formal_ready=preflight.formal_ready,
    )
    print(
        json.dumps(
            {
                "datasetId": report.dataset_id,
                "device": report.device,
                "formalReady": report.formal_ready,
                "output": str(output),
                "phase": report.phase,
                "promotable": report.promotable,
                "reportHash": report.report_hash,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "preflight":
        return _preflight(arguments)
    if arguments.command == "structure-cache":
        return _structure_cache(arguments)
    if arguments.command in {"email-smoke", "penn-dev"}:
        return _local_run(arguments)
    raise SystemExit(f"{arguments.command} requires its hash-bound evidence request")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
