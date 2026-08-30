"""Persistent CLI workflow for the fixed offline ogbl-collab baseline.

This module deliberately stops at offline evidence.  Every checkpoint is marked
non-registrable, and no code path writes the website/API model registry.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from .baseline.orchestrator import build_run_specs, run_core_spec
from .baseline.protocols import FEATURE_TIME_WARNING, build_protocol
from .baseline.trainer import evaluate_heuristic_bundle
from .baseline.types import CoreRunResult, CorpusArrays, RunSpec
from .canonical import canonical_sha256
from .checkpoint import (
    load_baseline_checkpoint,
    read_baseline_manifest,
    save_baseline_checkpoint,
)
from .contracts import (
    BaselineAcceptanceReport,
    BaselineConfig,
    BaselineEvaluationReport,
    BaselineRunManifest,
    RunStatus,
)
from .corpus import check_ogbl_collab_corpus, load_ogbl_collab_arrays
from .identity import code_identity_hash
from .registry import LocalRegistry
from .runtime import (
    RuntimeLayout,
    prepare_runtime_layout,
    require_ml_runtime,
    require_storage_reserve,
    runtime_report,
)

CONFIG_NAME = "ogbl-collab-baseline.json"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", by_alias=True, exclude_none=False)
    _atomic_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _config_path() -> Path:
    source_path = Path(__file__).resolve().parents[2] / "configs" / CONFIG_NAME
    if source_path.is_file():
        return source_path
    packaged_path = Path(__file__).resolve().parent / "resources" / "configs" / CONFIG_NAME
    if packaged_path.is_file():
        return packaged_path
    raise FileNotFoundError(f"fixed baseline config is absent: {CONFIG_NAME}")


def load_baseline_config() -> tuple[BaselineConfig, dict[str, Any], str]:
    payload = json.loads(_config_path().read_text(encoding="utf-8"))
    config = BaselineConfig.model_validate(payload)
    alias_payload = config.model_dump(mode="json", by_alias=True, exclude_none=False)
    return config, alias_payload, canonical_sha256(alias_payload)


def _load_corpus(root: Path) -> tuple[dict[str, Any], CorpusArrays]:
    manifest = check_ogbl_collab_corpus(root)
    corpus_hash = str(manifest["logicalHash"])
    arrays = CorpusArrays.from_mapping(
        load_ogbl_collab_arrays(root),
        corpus_hash=corpus_hash,
        expected_num_nodes=235_868,
        expected_feature_dim=128,
    )
    return manifest, arrays


def _experiment_id(phase: str) -> str:
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    entropy = canonical_sha256(
        {"phase": phase, "stamp": stamp, "pid": os.getpid(), "timeNs": time.time_ns()}
    )[:8]
    return f"collab-{phase}-{stamp}-{entropy}"


def _run_manifest_hash(manifest: BaselineRunManifest) -> str:
    return canonical_sha256(
        manifest.model_dump(mode="python", by_alias=True, exclude_none=False)
    )


def _manifest_for_run(
    spec: RunSpec,
    *,
    status: RunStatus,
    code_hash: str,
    environment_hash: str,
    corpus_hash: str,
    config_hash: str,
    started_at: datetime,
    finished_at: datetime | None = None,
    result: CoreRunResult | None = None,
    failure_code: str | None = None,
    artifacts: tuple[str, ...] = (),
    duration_seconds: float | None = None,
) -> BaselineRunManifest:
    return BaselineRunManifest(
        runId=spec.run_id,
        experimentId=spec.experiment_id,
        runKind="baseline",
        phase=spec.phase,
        track=spec.track,
        model=spec.model,
        status=status,
        seed=spec.seed,
        codeHash=code_hash,
        environmentHash=environment_hash,
        corpusHash=corpus_hash,
        configHash=config_hash,
        startedAt=started_at,
        finishedAt=finished_at,
        bestEpoch=result.best_epoch if result is not None else None,
        bestValidationHits50=(
            float(result.validation_metrics["hits@50"]) if result is not None else None
        ),
        peakCudaMemoryMiB=(result.peak_cuda_memory_mib if result is not None else 0.0),
        durationSeconds=duration_seconds,
        failureCode=failure_code,
        artifacts=artifacts,
        registrable=False,
    )


def _score_counts(result: CoreRunResult, corpus: CorpusArrays) -> dict[str, int]:
    protocol = build_protocol(corpus, result.spec.track)
    if protocol.validation.negative_edges is None or protocol.test.negative_edges is None:
        raise ValueError("baseline evaluation protocol requires fixed negative arrays")
    counts = {
        "validationPositive": int(len(protocol.validation.positive_edges)),
        "validationNegative": int(len(protocol.validation.negative_edges)),
        "testPositive": 0,
        "testNegative": 0,
    }
    if result.test_metrics is not None:
        counts["testPositive"] = int(len(protocol.test.positive_edges))
        counts["testNegative"] = int(len(protocol.test.negative_edges))
    return counts


def _evaluation_report(
    result: CoreRunResult, corpus: CorpusArrays
) -> BaselineEvaluationReport:
    payload: dict[str, Any] = {
        "schemaVersion": "gfm.baseline-evaluation/1.0",
        "experimentId": result.spec.experiment_id,
        "runId": result.spec.run_id,
        "phase": result.spec.phase,
        "track": result.spec.track,
        "model": result.spec.model,
        "seed": result.spec.seed,
        "validationMetrics": dict(result.validation_metrics),
        "testMetrics": dict(result.test_metrics) if result.test_metrics is not None else None,
        "strata": {
            str(name): {str(key): float(value) for key, value in metrics.items()}
            for name, metrics in result.strata.items()
        },
        "scoreCounts": _score_counts(result, corpus),
        "testReadAfterSelection": result.test_read_after_selection,
    }
    payload["reportHash"] = canonical_sha256(payload)
    return BaselineEvaluationReport.model_validate(payload)


class _CheckpointWriter:
    def __init__(
        self,
        *,
        run_dir: Path,
        spec: RunSpec,
        config_payload: dict[str, Any],
        corpus_hash: str,
        registry: LocalRegistry,
    ) -> None:
        self.directory = run_dir / "checkpoints"
        self.spec = spec
        self.config_payload = config_payload
        self.corpus_hash = corpus_hash
        self.registry = registry
        self.payloads: dict[str, dict[str, Any]] = {}
        self.manifests: dict[str, Any] = {}

    def __call__(self, kind: str, payload: Mapping[str, Any]) -> Any:
        if kind not in ("best", "latest"):
            raise ValueError(f"unsupported baseline checkpoint kind: {kind}")
        copied = dict(payload)
        self.payloads[kind] = copied
        return self.save(kind, copied)

    def save(
        self,
        kind: str,
        payload: Mapping[str, Any],
        *,
        verification_digest: str | None = None,
    ) -> Any:
        checkpoint_id = f"{self.spec.run_id}-{kind}"
        manifest = save_baseline_checkpoint(
            self.directory,
            checkpoint_id=checkpoint_id,
            run_id=self.spec.run_id,
            epoch=int(payload["epoch"]),
            track=self.spec.track,
            model=self.spec.model,
            model_state=dict(payload["model_state"]),
            predictor_state=dict(payload["predictor_state"]),
            optimizer_state=dict(payload["optimizer_state"]),
            scheduler_state=payload.get("scheduler_state"),
            sampler_state=dict(payload["sampler_state"]),
            selection_rng_state=dict(payload["selection_rng_state"]),
            best_validation_hits50=float(payload["best_validation_hits50"]),
            best_epoch=int(payload["best_epoch"]),
            best_model_state=dict(payload["best_model_state"]),
            best_predictor_state=dict(payload["best_predictor_state"]),
            selected_batch_size=int(payload["selected_batch_size"]),
            evaluations_without_improvement=int(
                payload["evaluations_without_improvement"]
            ),
            history=[dict(item) for item in payload["history"]],
            terminal=bool(payload["terminal"]),
            config=self.config_payload,
            corpus_hash=self.corpus_hash,
            verification_digest=verification_digest,
            rng_state=dict(payload["rng_state"]),
        )
        self.registry.record_baseline_checkpoint(manifest)
        self.manifests[kind] = manifest
        return manifest


def _verification_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = source_root + (os.pathsep + existing if existing else "")
    return environment


def _fresh_process_digest(
    manifest_path: Path, *, root: Path, device: str
) -> str:
    command = (
        sys.executable,
        "-m",
        "socialgraph_gfm.baseline_verify",
        "--manifest",
        str(manifest_path),
        "--root",
        str(root),
        "--device",
        device,
    )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env=_verification_environment(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1800,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "fresh-process checkpoint verification failed: "
            + (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")
        )
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1].lstrip("\ufeff"))
        digest = payload["verificationDigest"]
    except (IndexError, KeyError, json.JSONDecodeError, TypeError) as error:
        raise RuntimeError("fresh-process checkpoint verifier returned invalid JSON") from error
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("fresh-process checkpoint verifier returned an invalid digest")
    return digest


def _verify_official_graphsage_checkpoint(
    writer: _CheckpointWriter, *, root: Path, device: str
) -> str:
    manifest = writer.manifests.get("best")
    payload = writer.payloads.get("best")
    if manifest is None or payload is None:
        raise RuntimeError("official GraphSAGE run has no best checkpoint")
    manifest_path = writer.directory / f"{manifest.checkpoint_id}.manifest.json"
    first = _fresh_process_digest(manifest_path, root=root, device=device)
    second = _fresh_process_digest(manifest_path, root=root, device=device)
    if first != second:
        raise RuntimeError("fresh-process checkpoint verification is not deterministic")
    writer.save("best", payload, verification_digest=first)
    return first


def _persist_result(
    *,
    result: CoreRunResult,
    corpus: CorpusArrays,
    run_dir: Path,
    registry: LocalRegistry,
) -> BaselineEvaluationReport:
    evaluation = _evaluation_report(result, corpus)
    _write_json(run_dir / "evaluation.json", evaluation)
    _write_json(run_dir / "history.json", list(result.history))
    registry.record_baseline_evaluation(evaluation)
    return evaluation


def _run_one(
    spec: RunSpec,
    *,
    layout: RuntimeLayout,
    corpus: CorpusArrays,
    config: BaselineConfig,
    config_payload: dict[str, Any],
    config_hash: str,
    code_hash: str,
    environment_hash: str,
    device: str,
    registry: LocalRegistry,
    resume_state: Mapping[str, Any] | None = None,
    existing_started_at: datetime | None = None,
    precomputed_result: CoreRunResult | None = None,
    precomputed_duration_seconds: float = 0.0,
) -> dict[str, Any]:
    run_dir = layout.runs / spec.run_id
    if resume_state is None:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not run_dir.is_dir():
        raise FileNotFoundError(f"resume run directory is absent: {run_dir}")
    started_at = existing_started_at or _utc_now()
    running = _manifest_for_run(
        spec,
        status=RunStatus.RUNNING,
        code_hash=code_hash,
        environment_hash=environment_hash,
        corpus_hash=corpus.corpus_hash,
        config_hash=config_hash,
        started_at=started_at,
    )
    _write_json(run_dir / "run-manifest.json", running)
    registry.record_baseline_run(running)
    checkpoint_writer = (
        _CheckpointWriter(
            run_dir=run_dir,
            spec=spec,
            config_payload=config_payload,
            corpus_hash=corpus.corpus_hash,
            registry=registry,
        )
        if spec.model in ("mlp", "graphsage")
        else None
    )
    run_started = time.perf_counter()
    try:
        if precomputed_result is not None:
            if precomputed_result.spec != spec or resume_state is not None:
                raise ValueError("precomputed baseline result identity mismatch")
            result = precomputed_result
        else:
            result = run_core_spec(
                spec,
                corpus=corpus,
                config=config,
                device=device,
                checkpoint_sink=checkpoint_writer,
                resume_state=resume_state,
            )
        verification_digest = None
        if (
            spec.phase == "formal"
            and spec.track == "ogb_official"
            and spec.model == "graphsage"
        ):
            if checkpoint_writer is None:  # pragma: no cover - guarded by model
                raise RuntimeError("GraphSAGE checkpoint writer is absent")
            verification_digest = _verify_official_graphsage_checkpoint(
                checkpoint_writer, root=layout.root, device=device
            )
        evaluation = _persist_result(
            result=result,
            corpus=corpus,
            run_dir=run_dir,
            registry=registry,
        )
        artifacts = [
            str((run_dir / "evaluation.json").relative_to(layout.root).as_posix()),
            str((run_dir / "history.json").relative_to(layout.root).as_posix()),
        ]
        if checkpoint_writer is not None:
            artifacts.extend(
                str(
                    (checkpoint_writer.directory / f"{item.checkpoint_id}.manifest.json")
                    .relative_to(layout.root)
                    .as_posix()
                )
                for item in checkpoint_writer.manifests.values()
            )
        succeeded = _manifest_for_run(
            spec,
            status=RunStatus.SUCCEEDED,
            code_hash=code_hash,
            environment_hash=environment_hash,
            corpus_hash=corpus.corpus_hash,
            config_hash=config_hash,
            started_at=started_at,
            finished_at=_utc_now(),
            result=result,
            artifacts=tuple(sorted(set(artifacts))),
            duration_seconds=precomputed_duration_seconds
            + (time.perf_counter() - run_started),
        )
        _write_json(run_dir / "run-manifest.json", succeeded)
        registry.record_baseline_run(succeeded)
        return {
            "runId": spec.run_id,
            "status": "succeeded",
            "track": spec.track,
            "model": spec.model,
            "seed": spec.seed,
            "validationMetrics": dict(result.validation_metrics),
            "testMetrics": dict(result.test_metrics) if result.test_metrics else None,
            "strata": result.strata,
            "bestEpoch": result.best_epoch,
            "peakCudaMemoryMiB": result.peak_cuda_memory_mib,
            "selectedBatchSize": result.selected_batch_size,
            "durationSeconds": precomputed_duration_seconds
            + (time.perf_counter() - run_started),
            "evaluationHash": evaluation.report_hash,
            "verificationDigest": verification_digest,
            "runManifestHash": _run_manifest_hash(succeeded),
        }
    except BaseException as error:
        failed = _manifest_for_run(
            spec,
            status=RunStatus.FAILED,
            code_hash=code_hash,
            environment_hash=environment_hash,
            corpus_hash=corpus.corpus_hash,
            config_hash=config_hash,
            started_at=started_at,
            finished_at=_utc_now(),
            failure_code=type(error).__name__,
            duration_seconds=precomputed_duration_seconds
            + (time.perf_counter() - run_started),
        )
        _write_json(run_dir / "run-manifest.json", failed)
        registry.record_baseline_run(failed)
        _write_json(
            run_dir / "failure.json",
            {
                "schemaVersion": "gfm.baseline-failure/1.0",
                "runId": spec.run_id,
                "failureCode": type(error).__name__,
                "message": str(error),
                "failedAt": _utc_now().isoformat(),
            },
        )
        raise


def run_baseline_experiment(
    *,
    root: str | Path | None,
    phase: str,
    track: str,
    device: str,
) -> dict[str, Any]:
    """Run a new immutable dev/formal experiment using the checked-in config."""

    if phase not in ("dev", "formal"):
        raise ValueError("phase must be dev or formal")
    if track not in ("both", "ogb_official", "strict_edge_time"):
        raise ValueError("track is unsupported")
    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    manifest, corpus = _load_corpus(layout.root)
    config, config_payload, config_hash = load_baseline_config()
    environment = runtime_report(device)
    environment_hash = str(environment["environmentHash"])
    code_hash = code_identity_hash()
    experiment_id = _experiment_id(phase)
    experiment_dir = layout.runs / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=False)
    registry = LocalRegistry(layout.registry / "registry.sqlite3")
    specs = build_run_specs(
        experiment_id=experiment_id,
        phase=phase,
        config=config,
        tracks=track,
    )
    experiment_started = _utc_now()
    _write_json(
        experiment_dir / "experiment.json",
        {
            "schemaVersion": "gfm.baseline-experiment/1.0",
            "experimentId": experiment_id,
            "phase": phase,
            "track": track,
            "device": device,
            "runIds": [spec.run_id for spec in specs],
            "configHash": config_hash,
            "corpusHash": corpus.corpus_hash,
            "codeHash": code_hash,
            "environmentHash": environment_hash,
            "startedAt": experiment_started.isoformat(),
        },
    )
    results = []
    precomputed: dict[str, CoreRunResult] = {}
    precomputed_durations: dict[str, float] = {}
    for selected_track in ("ogb_official", "strict_edge_time"):
        group = tuple(
            spec
            for spec in specs
            if spec.track == selected_track and spec.model in ("cn", "aa", "ra")
        )
        if group:
            protocol = build_protocol(corpus, selected_track)
            group_started = time.perf_counter()
            group_results = evaluate_heuristic_bundle(
                group, corpus=corpus, protocol=protocol
            )
            per_run_seconds = (time.perf_counter() - group_started) / len(group_results)
            precomputed.update({item.spec.run_id: item for item in group_results})
            precomputed_durations.update(
                {item.spec.run_id: per_run_seconds for item in group_results}
            )
    for spec in specs:
        require_storage_reserve(layout.root, operation="run")
        results.append(
            _run_one(
                spec,
                layout=layout,
                corpus=corpus,
                config=config,
                config_payload=config_payload,
                config_hash=config_hash,
                code_hash=code_hash,
                environment_hash=environment_hash,
                device=device,
                registry=registry,
                precomputed_result=precomputed.get(spec.run_id),
                precomputed_duration_seconds=precomputed_durations.get(spec.run_id, 0.0),
            )
        )
    output = {
        "ok": True,
        "schemaVersion": "gfm.baseline-experiment-result/1.0",
        "experimentId": experiment_id,
        "phase": phase,
        "track": track,
        "device": device,
        "corpusHash": str(manifest["logicalHash"]),
        "configHash": config_hash,
        "codeHash": code_hash,
        "environmentHash": environment_hash,
        "startedAt": experiment_started.isoformat(),
        "finishedAt": _utc_now().isoformat(),
        "runs": results,
    }
    _write_json(experiment_dir / "result.json", output)
    return output


def _read_run_manifest(path: Path) -> BaselineRunManifest:
    return BaselineRunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def resume_baseline_run(
    *, root: str | Path | None, run_id: str, device: str
) -> dict[str, Any]:
    """Resume only from an integrity-checked latest learning checkpoint."""

    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    run_dir = layout.runs / run_id
    run_manifest = _read_run_manifest(run_dir / "run-manifest.json")
    if run_manifest.run_id != run_id:
        raise ValueError("resume run ID does not match its manifest")
    if run_manifest.model not in ("mlp", "graphsage"):
        raise ValueError("heuristic runs are deterministic and are not resumable")
    config, config_payload, config_hash = load_baseline_config()
    if run_manifest.config_hash != config_hash:
        raise ValueError("resume run config hash differs from the frozen config")
    _, corpus = _load_corpus(layout.root)
    if run_manifest.corpus_hash != corpus.corpus_hash:
        raise ValueError("resume run corpus hash differs from the checked corpus")
    if run_manifest.code_hash != code_identity_hash():
        raise ValueError("resume requires the exact original code identity")
    environment_hash = str(runtime_report(device)["environmentHash"])
    if run_manifest.environment_hash != environment_hash:
        raise ValueError("resume requires the exact original environment identity")
    manifest_path = run_dir / "checkpoints" / f"{run_id}-latest.manifest.json"
    checkpoint_manifest = read_baseline_manifest(manifest_path)
    resume_state = load_baseline_checkpoint(checkpoint_manifest, map_location="cpu")
    spec = RunSpec(
        experiment_id=run_manifest.experiment_id,
        run_id=run_manifest.run_id,
        phase=run_manifest.phase,
        track=run_manifest.track,
        model=run_manifest.model,
        seed=run_manifest.seed,
    )
    registry = LocalRegistry(layout.registry / "registry.sqlite3")
    result = _run_one(
        spec,
        layout=layout,
        corpus=corpus,
        config=config,
        config_payload=config_payload,
        config_hash=config_hash,
        code_hash=run_manifest.code_hash,
        environment_hash=environment_hash,
        device=device,
        registry=registry,
        resume_state=resume_state,
        existing_started_at=run_manifest.started_at,
    )
    return {
        "ok": True,
        "schemaVersion": "gfm.baseline-resume-result/1.0",
        "experimentId": run_manifest.experiment_id,
        "resumedFromEpoch": int(resume_state["epoch"]),
        "run": result,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "count": float(len(values)),
    }


def _metric_summary(evaluations: tuple[BaselineEvaluationReport, ...]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for track in ("ogb_official", "strict_edge_time"):
        for model in ("cn", "aa", "ra", "mlp", "graphsage"):
            selected = [
                item for item in evaluations if item.track == track and item.model == model
            ]
            if not selected:
                continue
            valid = [float(item.validation_metrics["hits@50"]) for item in selected]
            test = [
                float(item.test_metrics["hits@50"])
                for item in selected
                if item.test_metrics is not None
            ]
            summary = {
                "validationHits50Mean": _mean_std(valid)["mean"],
                "validationHits50Std": _mean_std(valid)["std"],
                "seedCount": float(len(selected)),
            }
            if test:
                summary["testHits50Mean"] = _mean_std(test)["mean"]
                summary["testHits50Std"] = _mean_std(test)["std"]
            output[f"{track}.{model}"] = summary
    return output


def _acceptance_markdown(
    report: BaselineAcceptanceReport,
    runs: tuple[BaselineRunManifest, ...],
    evaluations: tuple[BaselineEvaluationReport, ...],
    validation_reasons: list[str],
) -> str:
    lines = [
        "# ogbl-collab Baseline 验收报告",
        "",
        f"- Experiment: `{report.experiment_id}`",
        f"- Accepted / BaselineValidated: `{str(report.accepted).lower()}`",
        f"- Corpus hash: `{report.corpus_hash}`",
        f"- Config hash: `{report.config_hash}`",
        f"- Code hash: `{report.code_hash or 'unavailable'}`",
        f"- Environment hash: `{report.environment_hash or 'unavailable'}`",
        f"- Aggregated run time: `{(report.duration_seconds or 0.0):.2f} s`",
        f"- Formal learning runs: `{report.completed_learning_runs}/12`",
        f"- Heuristic runs: `{report.completed_heuristic_runs}/6`",
        f"- Peak CUDA memory: `{report.peak_cuda_memory_mib:.2f} MiB`",
        "",
        "## Hard gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {'yes' if passed else 'no'} |"
        for name, passed in sorted(report.gates.items())
    )
    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
        "| Track | Model | Seed | Validation Hits@50 | Test Hits@50 | Best epoch | Peak MiB | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    run_by_id = {run.run_id: run for run in runs}
    for item in evaluations:
        run = run_by_id[item.run_id]
        test = (
            f"{item.test_metrics['hits@50']:.6f}" if item.test_metrics is not None else "—"
        )
        lines.append(
            f"| {item.track} | {item.model} | {item.seed} | "
            f"{item.validation_metrics['hits@50']:.6f} | {test} | "
            f"{run.best_epoch if run.best_epoch is not None else '—'} | "
            f"{run.peak_cuda_memory_mib:.2f} | {run.duration_seconds or 0.0:.2f} |"
        )
    lines.extend(["", "## Protocol and limitations", ""])
    lines.extend(f"- {warning}" for warning in report.warnings)
    if validation_reasons:
        lines.extend(["", "## Failed evidence checks", ""])
        lines.extend(f"- {reason}" for reason in validation_reasons)
    lines.extend(
        [
            "",
            "This artifact validates an offline baseline only. It is permanently non-registrable "
            "as a website inference model; `ModelValidated` and `GfmServingReady` remain false.",
            "",
        ]
    )
    return "\n".join(lines)


def validate_baseline_experiment(
    *, root: str | Path | None, experiment_id: str
) -> dict[str, Any]:
    """Build and registry-check the hard-gated formal acceptance artifact."""

    layout = RuntimeLayout.from_root(root)
    require_storage_reserve(layout.root, operation="run")
    manifest = check_ogbl_collab_corpus(layout.root)
    config, _, config_hash = load_baseline_config()
    registry = LocalRegistry(layout.registry / "registry.sqlite3", initialize=False)
    runs = registry.list_baseline_runs(experiment_id)
    checkpoints = registry.list_baseline_checkpoints(experiment_id)
    evaluations = registry.list_baseline_evaluations(experiment_id)
    formal_runs = [
        run
        for run in runs
        if run.phase == "formal"
        and run.run_kind == "baseline"
        and run.status == RunStatus.SUCCEEDED
    ]
    learning = [run for run in formal_runs if run.model in ("mlp", "graphsage")]
    heuristics = [run for run in formal_runs if run.model in ("cn", "aa", "ra")]
    summary = _metric_summary(evaluations)
    graph_summary = summary.get("ogb_official.graphsage", {})
    mlp_summary = summary.get("ogb_official.mlp", {})
    strict_evaluations = [item for item in evaluations if item.track == "strict_edge_time"]
    verified_official_graphsage = {
        item.run_id
        for item in checkpoints
        if item.track == "ogb_official"
        and item.model == "graphsage"
        and item.verification_digest is not None
    }
    expected_official_graphsage = {
        run.run_id
        for run in learning
        if run.track == "ogb_official" and run.model == "graphsage"
    }
    peak = max((run.peak_cuda_memory_mib for run in formal_runs), default=0.0)
    graph_validation = graph_summary.get("validationHits50Mean", float("-inf"))
    graph_test = graph_summary.get("testHits50Mean", float("-inf"))
    mlp_test = mlp_summary.get("testHits50Mean", float("inf"))
    expected_learning = {
        (track, model, seed)
        for track in config.tracks
        for model in ("mlp", "graphsage")
        for seed in config.formal_seeds
    }
    actual_learning = {(run.track, run.model, run.seed) for run in learning}
    expected_heuristics = {
        (track, model)
        for track in config.tracks
        for model in ("cn", "aa", "ra")
    }
    actual_heuristics = {(run.track, run.model) for run in heuristics}
    gates = {
        "corpus_ready": str(manifest["logicalHash"]) == (runs[0].corpus_hash if runs else ""),
        "config_frozen": bool(runs)
        and all(run.config_hash == config_hash for run in runs),
        "formal_matrix_complete": actual_learning == expected_learning,
        "heuristic_matrix_complete": actual_heuristics == expected_heuristics,
        "metrics_complete": len(evaluations) == 18
        and all(item.test_metrics is not None for item in evaluations),
        "cuda_memory_within_limit": bool(formal_runs)
        and peak < float(config.cuda_memory_limit_mib),
        "official_graphsage_validation_threshold": graph_validation
        >= float(config.official_min_validation_hits50),
        "official_graphsage_test_threshold": graph_test
        >= float(config.official_min_test_hits50),
        "official_graphsage_gain_over_mlp": graph_test - mlp_test
        >= float(config.official_min_test_gain_over_mlp),
        "strict_edge_time_audit_passed": len(strict_evaluations) == 9
        and all(set(item.strata) == {"first_time", "repeated"} for item in strict_evaluations),
        "test_read_after_selection": len(evaluations) == 18
        and all(item.test_read_after_selection for item in evaluations),
        "checkpoint_recovery_verified": expected_official_graphsage
        == verified_official_graphsage
        and len(expected_official_graphsage) == 3,
    }
    accepted = all(gates.values())
    acceptance_payload: dict[str, Any] = {
        "schemaVersion": "gfm.baseline-acceptance/1.0",
        "experimentId": experiment_id,
        "accepted": accepted,
        "corpusHash": str(manifest["logicalHash"]),
        "configHash": config_hash,
        "requiredLearningRuns": 12,
        "completedLearningRuns": len(learning),
        "completedHeuristicRuns": len(heuristics),
        "peakCudaMemoryMiB": peak,
        "metricSummary": summary,
        "codeHash": formal_runs[0].code_hash if formal_runs else None,
        "environmentHash": formal_runs[0].environment_hash if formal_runs else None,
        "durationSeconds": sum(run.duration_seconds or 0.0 for run in formal_runs),
        "runResults": tuple(
            {
                "runId": item.run_id,
                "track": item.track,
                "model": item.model,
                "seed": item.seed,
                "validationMetrics": item.validation_metrics,
                "testMetrics": item.test_metrics,
                "strata": item.strata,
                "bestEpoch": next(
                    (run.best_epoch for run in formal_runs if run.run_id == item.run_id),
                    None,
                ),
                "peakCudaMemoryMiB": next(
                    (
                        run.peak_cuda_memory_mib
                        for run in formal_runs
                        if run.run_id == item.run_id
                    ),
                    0.0,
                ),
            }
            for item in evaluations
        ),
        "gates": gates,
        "warnings": (
            FEATURE_TIME_WARNING,
            "The official transductive protocol contains training targets in its message graph.",
            "This is an offline GraphSAGE baseline, not a trained or deployable GFM.",
        ),
    }
    acceptance_payload["reportHash"] = canonical_sha256(
        {key: value for key, value in acceptance_payload.items() if value is not None}
    )
    acceptance_payload["createdAt"] = _utc_now()
    report = BaselineAcceptanceReport.model_validate(acceptance_payload)
    experiment_report_dir = layout.reports / experiment_id
    _write_json(experiment_report_dir / "baseline-acceptance.json", report)
    _write_json(layout.reports / "baseline-acceptance.json", report)
    registry.record_baseline_acceptance(report)
    evidence = registry.validate_baseline_acceptance(
        report, corpus_manifest_hash=str(manifest["logicalHash"])
    )
    if report.accepted and not evidence["ready"]:
        raise RuntimeError(
            "acceptance report failed independent registry verification: "
            + "; ".join(evidence["reasons"])
        )
    markdown = _acceptance_markdown(
        report, runs, evaluations, list(evidence["reasons"])
    )
    _atomic_text(experiment_report_dir / "baseline-acceptance.md", markdown)
    return {
        **report.model_dump(mode="json", by_alias=True, exclude_none=False),
        "registryEvidence": evidence,
        "jsonReport": str(experiment_report_dir / "baseline-acceptance.json"),
        "markdownReport": str(experiment_report_dir / "baseline-acceptance.md"),
    }
