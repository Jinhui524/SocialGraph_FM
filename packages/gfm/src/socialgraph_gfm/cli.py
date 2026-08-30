"""`socialgraph-gfm` command-line entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .canonical import canonical_json, canonical_sha256, file_sha256
from .errors import GfmError
from .fixtures import get_fixture, smoke_fit_node_ids
from .identity import DEFAULT_SMOKE_SEED
from .locks import verify_lock_manifest
from .materialize import materialize
from .preflight import preflight_report
from .runtime import artifact_root, require_ml_runtime, runtime_report
from .smoke import run_smoke


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            print(f"{key}: {value}")
    else:
        print(payload)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _materialize_command(fixture: str, output: str, device: str) -> dict[str, Any]:
    torch, _ = require_ml_runtime(device)
    snapshot = get_fixture(fixture)
    materialized = materialize(
        snapshot,
        purpose="training_smoke",
        fit_node_ids=smoke_fit_node_ids(fixture),
        device=device,
    )
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    graph_path = destination / "graph.pt"
    event_path = destination / "events.pt"
    snapshot_path = destination / "snapshot.json"
    manifest_path = destination / "materialization.manifest.json"
    transform_path = destination / "feature-transform.artifact.json"
    torch.save(materialized.graph, graph_path)
    torch.save(
        {
            "edge_ids": materialized.events.edge_ids,
            "source_ids": materialized.events.source_ids,
            "target_ids": materialized.events.target_ids,
            "relations": materialized.events.relations,
            "timestamps_micros": materialized.events.timestamps_micros,
        },
        event_path,
    )
    _atomic_text(snapshot_path, canonical_json(snapshot))
    _atomic_text(transform_path, canonical_json(materialized.transform_artifact))
    manifest = {
        **materialized.manifest,
        "artifacts": {
            "graph": {"path": str(graph_path), "sha256": file_sha256(graph_path)},
            "events": {"path": str(event_path), "sha256": file_sha256(event_path)},
            "snapshot": {"path": str(snapshot_path), "sha256": file_sha256(snapshot_path)},
            "featureTransform": {
                "path": str(transform_path),
                "sha256": file_sha256(transform_path),
            },
        },
    }
    manifest["artifactManifestHash"] = canonical_sha256(manifest)
    _atomic_text(manifest_path, canonical_json(manifest))
    return {
        "ok": True,
        "output": str(destination),
        "manifest": str(manifest_path),
        "materializationHash": materialized.manifest["materializationHash"],
        "artifactManifestHash": manifest["artifactManifestHash"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="socialgraph-gfm")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Inspect exact runtime compatibility")
    doctor.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    doctor.add_argument("--root", default=None)
    doctor.add_argument("--json", action="store_true")

    materialize_parser = subparsers.add_parser(
        "materialize", help="Materialize a deterministic synthetic fixture"
    )
    materialize_parser.add_argument(
        "--fixture",
        required=True,
        choices=(
            "actor",
            "hetero",
            "collaboration.actor-interaction/1.0",
            "collaboration.activity-hetero/1.0",
        ),
    )
    materialize_parser.add_argument("--output", required=True)
    materialize_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    materialize_parser.add_argument("--json", action="store_true")

    smoke_parser = subparsers.add_parser("smoke", help="Run one synthetic autograd step")
    smoke_parser.add_argument("--fixture", choices=("actor", "hetero", "both"), default="both")
    smoke_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    smoke_parser.add_argument("--root", default=None)
    smoke_parser.add_argument("--seed", type=int, default=DEFAULT_SMOKE_SEED)
    smoke_parser.add_argument("--json", action="store_true")

    preflight_parser = subparsers.add_parser("preflight", help="Aggregate independent readiness gates")
    preflight_parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    preflight_parser.add_argument("--root", default=None)
    preflight_parser.add_argument("--json", action="store_true")

    fetch_parser = subparsers.add_parser(
        "corpus-fetch-ogbl-collab",
        help="Download the pinned official corpus and build a local safe package",
    )
    fetch_parser.add_argument("--accept-license", required=True)
    fetch_parser.add_argument("--root", default=None)
    fetch_parser.add_argument("--json", action="store_true")

    prepare_parser = subparsers.add_parser(
        "corpus-prepare-ogbl-collab",
        help="Validate and materialize an immutable safe ogbl-collab package",
    )
    prepare_parser.add_argument("--package", required=True)
    prepare_parser.add_argument("--root", default=None)
    prepare_parser.add_argument("--json", action="store_true")

    check_parser = subparsers.add_parser(
        "corpus-check", help="Re-read and verify a prepared formal corpus"
    )
    check_parser.add_argument("--corpus-id", required=True, choices=("ogbl-collab",))
    check_parser.add_argument("--root", default=None)
    check_parser.add_argument("--json", action="store_true")

    baseline_parser = subparsers.add_parser(
        "baseline-run", help="Run the fixed offline ogbl-collab baseline matrix"
    )
    baseline_parser.add_argument("--phase", required=True, choices=("dev", "formal"))
    baseline_parser.add_argument(
        "--track",
        required=True,
        choices=("both", "ogb_official", "strict_edge_time"),
    )
    baseline_parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    baseline_parser.add_argument("--root", default=None)
    baseline_parser.add_argument("--json", action="store_true")

    resume_parser = subparsers.add_parser(
        "baseline-resume", help="Resume a failed/interrupted learning baseline run"
    )
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    resume_parser.add_argument("--root", default=None)
    resume_parser.add_argument("--json", action="store_true")

    validate_parser = subparsers.add_parser(
        "baseline-validate", help="Evaluate the fixed hard gates for a formal experiment"
    )
    validate_parser.add_argument("--experiment-id", required=True)
    validate_parser.add_argument("--root", default=None)
    validate_parser.add_argument("--json", action="store_true")

    gfm_openalex = subparsers.add_parser(
        "gfm-corpus-fetch-openalex",
        help="Fetch the pinned OpenAlex Graph-AI subset through the official API",
    )
    gfm_openalex.add_argument("--spec", required=True, choices=("graph-ai",))
    gfm_openalex.add_argument(
        "--api-key-env", required=True, choices=("OPENALEX_API_KEY",)
    )
    gfm_openalex.add_argument("--root", default=None)
    gfm_openalex.add_argument("--json", action="store_true")

    gfm_thgl = subparsers.add_parser(
        "gfm-corpus-fetch-thgl-software",
        help="Fetch the fixed TGB thgl-software 2.0.0 archive",
    )
    gfm_thgl.add_argument("--accept-license", required=True)
    gfm_thgl.add_argument("--root", default=None)
    gfm_thgl.add_argument("--json", action="store_true")

    gfm_wikimedia = subparsers.add_parser(
        "gfm-corpus-fetch-wikimedia-talk",
        help="Fetch the fixed 2011--2015 article-talk corpus",
    )
    gfm_wikimedia.add_argument("--years", required=True)
    gfm_wikimedia.add_argument("--namespace", required=True, choices=("article",))
    gfm_wikimedia.add_argument("--accept-license", required=True)
    gfm_wikimedia.add_argument("--root", default=None)
    gfm_wikimedia.add_argument("--json", action="store_true")

    gfm_prepare = subparsers.add_parser(
        "gfm-corpus-prepare",
        help="Prepare and register one immutable GFM domain corpus",
    )
    gfm_prepare.add_argument(
        "--domain",
        required=True,
        choices=("openalex", "thgl-software", "wikimedia-talk"),
    )
    gfm_prepare.add_argument("--root", default=None)
    gfm_prepare.add_argument(
        "--newcomer-overlay",
        choices=("skip", "require"),
        default="skip",
        help=(
            "OpenAlex only: defer the separately versioned newcomer-history "
            "overlay, or require it to finish"
        ),
    )
    gfm_prepare.add_argument("--json", action="store_true")

    gfm_task_assets = subparsers.add_parser(
        "gfm-task-assets", help="Check base and optional product task corpus assets"
    )
    gfm_task_assets.add_argument(
        "--task", choices=("collaboration", "newcomer"), default=None
    )
    gfm_task_assets.add_argument("--root", default=None)
    gfm_task_assets.add_argument("--json", action="store_true")

    gfm_embed = subparsers.add_parser(
        "gfm-text-embed", help="Materialize pinned BGE-M3 embeddings offline"
    )
    gfm_embed.add_argument("--encoder", required=True, choices=("BAAI/bge-m3",))
    gfm_embed.add_argument(
        "--domain",
        choices=("all", "openalex", "wikimedia-talk"),
        default="all",
        help="Embed both text domains or one immutable domain for resumable staging",
    )
    gfm_embed.add_argument("--root", default=None)
    gfm_embed.add_argument("--json", action="store_true")

    gfm_pretrain = subparsers.add_parser(
        "gfm-pretrain", help="Run the fixed multi-domain Core pretraining matrix"
    )
    gfm_pretrain.add_argument("--phase", required=True, choices=("dev", "formal"))
    gfm_pretrain.add_argument("--config", required=True)
    gfm_pretrain.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    gfm_pretrain.add_argument(
        "--variant", choices=("core-base", "core-moe"), default=None
    )
    gfm_pretrain.add_argument("--seed", type=int, default=None)
    gfm_pretrain.add_argument("--root", default=None)
    gfm_pretrain.add_argument("--json", action="store_true")

    gfm_adapt = subparsers.add_parser(
        "gfm-adapt", help="Adapt an accepted pretraining experiment to a product task"
    )
    gfm_adapt.add_argument(
        "--task", required=True, choices=("collaboration", "newcomer")
    )
    gfm_adapt.add_argument("--experiment-id", required=True)
    gfm_adapt.add_argument("--seed", type=int, default=None)
    gfm_adapt.add_argument("--root", default=None)
    gfm_adapt.add_argument("--json", action="store_true")

    gfm_evaluate = subparsers.add_parser(
        "gfm-evaluate", help="Read immutable LODO/product/shadow evaluation evidence"
    )
    gfm_evaluate.add_argument(
        "--protocol", required=True, choices=("lodo", "product", "shadow")
    )
    gfm_evaluate.add_argument("--experiment-id", required=True)
    gfm_evaluate.add_argument("--held-out-domain", default=None)
    gfm_evaluate.add_argument(
        "--variant", choices=("core-base", "core-moe"), default=None
    )
    gfm_evaluate.add_argument("--seed", type=int, default=None)
    gfm_evaluate.add_argument(
        "--task",
        choices=("collaboration",),
        default=None,
        help=(
            "Evaluate and independently accept only collaboration; valid only "
            "with --protocol product and never enables full model acceptance"
        ),
    )
    gfm_evaluate.add_argument("--root", default=None)
    gfm_evaluate.add_argument("--json", action="store_true")

    gfm_resume = subparsers.add_parser(
        "gfm-resume", help="Integrity-check and resume an interrupted Core run"
    )
    gfm_resume.add_argument("--run-id", required=True)
    gfm_resume.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    gfm_resume.add_argument("--root", default=None)
    gfm_resume.add_argument("--json", action="store_true")

    gfm_validate = subparsers.add_parser(
        "gfm-validate", help="Derive the fixed offline GFM acceptance gates"
    )
    gfm_validate.add_argument("--experiment-id", required=True)
    gfm_validate.add_argument(
        "--scope", choices=("pretraining", "full"), default="full"
    )
    gfm_validate.add_argument("--root", default=None)
    gfm_validate.add_argument("--json", action="store_true")

    gfm_export = subparsers.add_parser(
        "gfm-export", help="Export an integrity-checked accepted offline checkpoint"
    )
    gfm_export.add_argument("--experiment-id", required=True)
    gfm_export.add_argument("--root", default=None)
    gfm_export.add_argument("--json", action="store_true")

    gfm_worker = subparsers.add_parser(
        "_gfm-pretrain-run", help=argparse.SUPPRESS
    )
    gfm_worker.add_argument("--phase", required=True, choices=("dev", "formal"))
    gfm_worker.add_argument("--config", required=True)
    gfm_worker.add_argument("--variant", required=True, choices=("core-base", "core-moe"))
    gfm_worker.add_argument("--seed", required=True, type=int)
    gfm_worker.add_argument("--device", required=True, choices=("cpu", "cuda"))
    gfm_worker.add_argument("--root", required=True)
    gfm_worker.add_argument("--json", action="store_true")

    gfm_verify = subparsers.add_parser(
        "_gfm-verify-checkpoint", help=argparse.SUPPRESS
    )
    gfm_verify.add_argument("--checkpoint-manifest", required=True)
    gfm_verify.add_argument("--device", required=True, choices=("cpu", "cuda"))
    gfm_verify.add_argument("--root", required=True)
    gfm_verify.add_argument("--json", action="store_true")

    gfm_test_once = subparsers.add_parser(
        "_gfm-evaluate-test-once", help=argparse.SUPPRESS
    )
    gfm_test_once.add_argument("--checkpoint-manifest", required=True)
    gfm_test_once.add_argument("--device", required=True, choices=("cpu", "cuda"))
    gfm_test_once.add_argument("--root", required=True)
    gfm_test_once.add_argument("--json", action="store_true")

    gfm_adapt_worker = subparsers.add_parser(
        "_gfm-adapt-run", help=argparse.SUPPRESS
    )
    gfm_adapt_worker.add_argument(
        "--task", required=True, choices=("collaboration", "newcomer")
    )
    gfm_adapt_worker.add_argument("--experiment-id", required=True)
    gfm_adapt_worker.add_argument("--backbone-checkpoint-id", required=True)
    gfm_adapt_worker.add_argument("--device", required=True, choices=("cpu", "cuda"))
    gfm_adapt_worker.add_argument("--root", required=True)
    gfm_adapt_worker.add_argument("--json", action="store_true")

    gfm_product_verify = subparsers.add_parser(
        "_gfm-verify-product-checkpoint", help=argparse.SUPPRESS
    )
    gfm_product_verify.add_argument("--checkpoint-manifest", required=True)
    gfm_product_verify.add_argument(
        "--device", required=True, choices=("cpu", "cuda")
    )
    gfm_product_verify.add_argument("--root", required=True)
    gfm_product_verify.add_argument("--json", action="store_true")

    gfm_lodo_worker = subparsers.add_parser(
        "_gfm-lodo-run", help=argparse.SUPPRESS
    )
    gfm_lodo_worker.add_argument("--experiment-id", required=True)
    gfm_lodo_worker.add_argument("--held-out-domain", required=True)
    gfm_lodo_worker.add_argument(
        "--variant", required=True, choices=("core-base", "core-moe")
    )
    gfm_lodo_worker.add_argument("--seed", required=True, type=int)
    gfm_lodo_worker.add_argument("--device", required=True, choices=("cpu", "cuda"))
    gfm_lodo_worker.add_argument("--root", required=True)
    gfm_lodo_worker.add_argument("--json", action="store_true")

    gfm_suite_verify = subparsers.add_parser(
        "_gfm-verify-suite-checkpoint", help=argparse.SUPPRESS
    )
    gfm_suite_verify.add_argument("--checkpoint-manifest", required=True)
    gfm_suite_verify.add_argument("--root", required=True)
    gfm_suite_verify.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    as_json = bool(getattr(args, "json", False))
    try:
        if args.command == "doctor":
            payload = runtime_report(args.device)
            payload["artifactRoot"] = str(artifact_root(args.root))
            payload["runtimeLocks"] = verify_lock_manifest()
            payload["runtimeReady"] = payload["runtimeReady"] and payload["runtimeLocks"]["releaseLocksReady"]
            _print(payload, as_json)
            return 0 if payload["runtimeReady"] else 2
        if args.command == "materialize":
            payload = _materialize_command(args.fixture, args.output, args.device)
            _print(payload, as_json)
            return 0
        if args.command == "smoke":
            payload = run_smoke(
                fixture=args.fixture, device=args.device, root=args.root, seed=args.seed
            )
            _print(payload, as_json)
            return 0
        if args.command == "preflight":
            payload = preflight_report(device=args.device, root=args.root)
            _print(payload, as_json)
            return 0 if payload["readiness"]["GfmInfrastructureReady"] else 3
        if args.command == "corpus-fetch-ogbl-collab":
            from .corpus import fetch_ogbl_collab

            payload = fetch_ogbl_collab(
                args.root, accept_license=args.accept_license
            )
            _print(payload, as_json)
            return 0
        if args.command == "corpus-prepare-ogbl-collab":
            from .corpus import prepare_ogbl_collab_corpus

            manifest = prepare_ogbl_collab_corpus(args.package, args.root)
            payload = (
                manifest.model_dump(mode="json", by_alias=True, exclude_none=False)
                if hasattr(manifest, "model_dump")
                else manifest
            )
            _print(payload, as_json)
            return 0
        if args.command == "corpus-check":
            from .corpus import check_ogbl_collab_corpus

            manifest = check_ogbl_collab_corpus(args.root)
            payload = (
                manifest.model_dump(mode="json", by_alias=True, exclude_none=False)
                if hasattr(manifest, "model_dump")
                else manifest
            )
            _print(payload, as_json)
            return 0
        if args.command == "baseline-run":
            from .baseline_workflow import run_baseline_experiment

            payload = run_baseline_experiment(
                root=args.root,
                phase=args.phase,
                track=args.track,
                device=args.device,
            )
            _print(payload, as_json)
            return 0
        if args.command == "baseline-resume":
            from .baseline_workflow import resume_baseline_run

            payload = resume_baseline_run(
                root=args.root, run_id=args.run_id, device=args.device
            )
            _print(payload, as_json)
            return 0
        if args.command == "baseline-validate":
            from .baseline_workflow import validate_baseline_experiment

            payload = validate_baseline_experiment(
                root=args.root, experiment_id=args.experiment_id
            )
            _print(payload, as_json)
            return 0 if payload.get("accepted") is True else 6
        if args.command == "gfm-corpus-fetch-openalex":
            from .gfm_workflow import fetch_gfm_openalex

            payload = fetch_gfm_openalex(
                root=args.root,
                spec=args.spec,
                api_key_env=args.api_key_env,
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-corpus-fetch-thgl-software":
            from .gfm_workflow import fetch_gfm_thgl_software

            payload = fetch_gfm_thgl_software(
                root=args.root, accept_license=args.accept_license
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-corpus-fetch-wikimedia-talk":
            from .gfm_workflow import fetch_gfm_wikimedia_talk

            payload = fetch_gfm_wikimedia_talk(
                root=args.root,
                years=args.years,
                namespace=args.namespace,
                accept_license=args.accept_license,
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-corpus-prepare":
            from .gfm_workflow import prepare_gfm_corpus

            payload = prepare_gfm_corpus(
                root=args.root,
                domain=args.domain,
                newcomer_overlay=args.newcomer_overlay,
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-task-assets":
            from .gfm_workflow import check_gfm_task_assets

            payload = check_gfm_task_assets(root=args.root, task=args.task)
            _print(payload, as_json)
            return 0 if payload.get("ok") is True else 8
        if args.command == "gfm-text-embed":
            from .gfm_workflow import embed_gfm_text

            payload = embed_gfm_text(
                root=args.root, encoder=args.encoder, domain=args.domain
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-pretrain":
            from .gfm_workflow import pretrain_gfm

            payload = pretrain_gfm(
                root=args.root,
                phase=args.phase,
                config=args.config,
                device=args.device,
                variant=args.variant,
                seed=args.seed,
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-adapt":
            from .gfm_workflow import adapt_gfm

            payload = adapt_gfm(
                root=args.root,
                task=args.task,
                experiment_id=args.experiment_id,
                seed=args.seed,
            )
            _print(payload, as_json)
            return 0 if payload.get("ok") is True else 8
        if args.command == "gfm-evaluate":
            from .gfm_workflow import evaluate_gfm

            payload = evaluate_gfm(
                root=args.root,
                protocol=args.protocol,
                experiment_id=args.experiment_id,
                held_out_domain=args.held_out_domain,
                variant=args.variant,
                seed=args.seed,
                task=args.task,
            )
            _print(payload, as_json)
            return 0
        if args.command == "gfm-resume":
            from .gfm_workflow import resume_gfm

            payload = resume_gfm(root=args.root, run_id=args.run_id, device=args.device)
            _print(payload, as_json)
            return 0 if payload.get("ok") is True else 8
        if args.command == "gfm-validate":
            from .gfm_workflow import validate_gfm

            payload = validate_gfm(
                root=args.root,
                experiment_id=args.experiment_id,
                scope=args.scope,
            )
            _print(payload, as_json)
            return 0 if payload.get("accepted") is True else 7
        if args.command == "gfm-export":
            from .gfm_workflow import export_gfm

            payload = export_gfm(root=args.root, experiment_id=args.experiment_id)
            _print(payload, as_json)
            return 0
        if args.command == "_gfm-pretrain-run":
            from .gfm_workflow import _pretrain_worker

            run = _pretrain_worker(
                root=args.root,
                phase=args.phase,
                config=args.config,
                variant=args.variant,
                seed=args.seed,
                device=args.device,
            )
            _print(
                {
                    "schemaVersion": "gfm.workflow-pretrain-worker/1.0",
                    "ok": True,
                    "run": run,
                },
                as_json,
            )
            return 0
        if args.command == "_gfm-verify-checkpoint":
            from .gfm_workflow import verify_gfm_checkpoint_fresh

            payload = verify_gfm_checkpoint_fresh(
                root=args.root,
                checkpoint_manifest=args.checkpoint_manifest,
                device=args.device,
            )
            _print(payload, as_json)
            return 0
        if args.command == "_gfm-evaluate-test-once":
            from .gfm_workflow import evaluate_gfm_checkpoint_test_once

            payload = evaluate_gfm_checkpoint_test_once(
                root=args.root,
                checkpoint_manifest=args.checkpoint_manifest,
                device=args.device,
            )
            _print(payload, as_json)
            return 0
        if args.command == "_gfm-adapt-run":
            from .gfm_workflow import _adapt_worker

            payload = {
                "schemaVersion": "gfm.adapt-worker/1.0",
                "ok": True,
                "run": _adapt_worker(
                    root=args.root,
                    task=args.task,
                    experiment_id=args.experiment_id,
                    backbone_checkpoint_id=args.backbone_checkpoint_id,
                    device=args.device,
                ),
            }
            _print(payload, as_json)
            return 0
        if args.command == "_gfm-verify-product-checkpoint":
            from .gfm_workflow import verify_gfm_product_checkpoint_fresh

            payload = verify_gfm_product_checkpoint_fresh(
                root=args.root,
                checkpoint_manifest=args.checkpoint_manifest,
                device=args.device,
            )
            _print(payload, as_json)
            return 0
        if args.command == "_gfm-lodo-run":
            from .gfm_workflow import _lodo_worker

            payload = {
                "schemaVersion": "gfm.lodo-worker/1.0",
                "ok": True,
                "run": _lodo_worker(
                    root=args.root,
                    experiment_id=args.experiment_id,
                    held_out_domain=args.held_out_domain,
                    variant=args.variant,
                    seed=args.seed,
                    device=args.device,
                ),
            }
            _print(payload, as_json)
            return 0
        if args.command == "_gfm-verify-suite-checkpoint":
            from .gfm_workflow import verify_gfm_suite_checkpoint_fresh

            payload = verify_gfm_suite_checkpoint_fresh(
                root=args.root,
                checkpoint_manifest=args.checkpoint_manifest,
            )
            _print(payload, as_json)
            return 0
    except GfmError as error:
        payload = {"ok": False, "error": error.as_dict()}
        _print(payload, as_json)
        return 4
    except (OSError, ValueError, RuntimeError) as error:
        payload = {
            "ok": False,
            "error": {"code": "GFM_COMMAND_FAILED", "message": str(error)},
        }
        _print(payload, as_json)
        return 5
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
