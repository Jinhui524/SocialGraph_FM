"""Pretraining execution, recovery, matrix reuse, and freshness checks.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def _process_rss_mib() -> float:
    """Return this process' resident set without adding a runtime dependency."""

    if sys.platform == "win32":
        import ctypes

        class _ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_fault_count", ctypes.c_ulong),
                ("peak_working_set_size", ctypes.c_size_t),
                ("working_set_size", ctypes.c_size_t),
                ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                ("quota_paged_pool_usage", ctypes.c_size_t),
                ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                ("quota_non_paged_pool_usage", ctypes.c_size_t),
                ("pagefile_usage", ctypes.c_size_t),
                ("peak_pagefile_usage", ctypes.c_size_t),
            ]

        counters = _ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            raise GfmTrainingError("Windows could not read the pretrain worker RSS")
        return float(counters.working_set_size / (1024**2))
    if sys.platform.startswith("linux"):
        try:
            resident_pages = int(Path("/proc/self/statm").read_text("ascii").split()[1])
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError) as error:
            raise GfmTrainingError("Linux could not read the pretrain worker RSS") from error
        return float(resident_pages * page_size / (1024**2))
    try:
        import resource

        resident = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError) as error:
        raise GfmTrainingError("This platform could not read the pretrain worker RSS") from error
    # macOS reports bytes; the other supported resource implementations use KiB.
    return resident / (1024**2 if sys.platform == "darwin" else 1024)


def _pretrain_heartbeat(
    *,
    state_path: Path,
    state_base: Mapping[str, Any],
    batch_size: int,
    accumulation: int,
    probe_peak: float,
    preflight_attempt: int,
    trainer: Any,
    streams: Mapping[str, _DomainStream],
    started: datetime,
    device: str,
    last_losses: Mapping[str, float],
) -> None:
    """Atomically expose bounded operator progress without creating a checkpoint."""

    import torch

    losses = {str(name): float(value) for name, value in last_losses.items()}
    if not losses or not all(math.isfinite(value) for value in losses.values()):
        raise GfmTrainingError("Pretrain heartbeat requires finite current losses")
    domain_cursors = {
        domain: {"cursor": int(stream.cursor), "epoch": int(stream.epoch)}
        for domain, stream in streams.items()
    }
    negative_audits = {
        domain: deepcopy(stream.negative_sampling_audit) for domain, stream in streams.items()
    }
    cuda_peak = float(torch.cuda.max_memory_allocated() / (1024**2)) if device == "cuda" else 0.0
    now = datetime.now(UTC)
    _save_run_state(
        state_path,
        {
            **dict(state_base),
            "batchSize": batch_size,
            "gradientAccumulation": accumulation,
            "probePeakCudaMemoryMiB": probe_peak,
            "preflightAttemptCount": preflight_attempt,
            "status": "running",
            "heartbeat": {
                "schemaVersion": "gfm.pretrain-heartbeat/1.0",
                "recordedAt": now.isoformat(),
                "optimizerStep": int(trainer.optimizer_step),
                "globalStep": int(trainer.global_step),
                "lastLosses": losses,
                "domainCursors": domain_cursors,
                "elapsedSeconds": max(0.0, float((now - started).total_seconds())),
                "rssMiB": _process_rss_mib(),
                "peakCudaMemoryMiB": max(probe_peak, cuda_peak),
                "negativeSamplingAudits": negative_audits,
                "negativeSamplingAuditsHash": canonical_sha256(negative_audits),
            },
        },
    )


def _train_epoch_with_heartbeats(
    trainer: Any,
    loaders: Mapping[str, Iterable[Any]],
    *,
    every_optimizer_steps: int,
    heartbeat: Callable[[Mapping[str, float]], None],
    after_optimizer_step: Callable[[], None] | None = None,
) -> Any:
    """Run an unchanged trainer epoch while observing each committed optimizer step.

    The trainer's evaluation/checkpoint interval remains intact.  The narrow
    wrappers only observe the last finite loss bundle and the optimizer-step
    boundary, then restore the original bound methods even on failure.
    """

    if every_optimizer_steps < 1:
        raise ValueError("Pretrain heartbeat cadence must be positive")
    original_forward = trainer._forward_loss_and_moments
    original_apply = trainer._apply_optimizer_step
    latest_loss: Any | None = None
    next_step = ((int(trainer.optimizer_step) // every_optimizer_steps) + 1) * every_optimizer_steps

    def observed_forward(batch: Any, cross_domain_reference: Any | None = None) -> Any:
        nonlocal latest_loss
        result = original_forward(batch, cross_domain_reference)
        latest_loss = result[0]
        return result

    def observed_apply(*, partial_accumulation: int | None = None) -> None:
        nonlocal next_step
        original_apply(partial_accumulation=partial_accumulation)
        # LR scheduling is an optimizer-step concern, not an evaluation-block
        # concern.  Invoke the observer immediately after the committed update
        # so warmup/cosine state and checkpointed optimizer progress remain in
        # exact lockstep, including after resume.
        if after_optimizer_step is not None:
            after_optimizer_step()
        while int(trainer.optimizer_step) >= next_step:
            if latest_loss is None:
                raise GfmTrainingError("Optimizer advanced without a heartbeat loss")
            heartbeat(latest_loss.detached())
            next_step += every_optimizer_steps

    trainer._forward_loss_and_moments = observed_forward
    trainer._apply_optimizer_step = observed_apply
    try:
        return trainer.train_epoch(loaders)
    finally:
        trainer._forward_loss_and_moments = original_forward
        trainer._apply_optimizer_step = original_apply


def _probe_with_durable_preflight(
    *,
    state_path: Path,
    state_base: Mapping[str, Any],
    attempt: int,
    config: GfmPretrainConfig,
    variant: str,
    streams: Mapping[str, _DomainStream],
    device: str,
    seed: int,
) -> tuple[int, float]:
    """Make an uncommitted training cell recoverable before probing hardware.

    The matrix owns the final run directory as soon as this atomic state write
    succeeds.  A probe or model-construction failure therefore leaves an
    explicitly retryable preflight state, rather than an anonymous directory.
    No optimizer progress is represented by this state.
    """

    if attempt < 1:
        raise ContractViolation("Preflight attempt must be a positive integer")
    _save_run_state(
        state_path,
        {
            **dict(state_base),
            "status": "preflight",
            "preflightAttemptCount": attempt,
            "lastPreflightStartedAt": datetime.now(UTC).isoformat(),
        },
    )
    return _probe_batch_size(
        config=config,
        variant=variant,
        streams=streams,
        device=device,
        seed=seed,
    )


def _ensure_checkpoint_free_retry_directory(run_dir: Path) -> None:
    """Accept only the durable marker and an optional empty owned checkpoint dir."""

    unexpected: list[str] = []
    for entry in run_dir.iterdir():
        if entry.name == "run-state.json":
            continue
        if entry.name == "checkpoints" and entry.is_dir() and not any(entry.iterdir()):
            continue
        unexpected.append(entry.name)
    unexpected.sort()
    if unexpected:
        raise GfmTrainingError(
            "Checkpoint-free retry found uncommitted run artifacts; refusing "
            "automatic overwrite: " + ", ".join(unexpected)
        )


def _retain_checkpoint_roles(directory: Path, *, run_id: str, current_id: str, role: str) -> None:
    """Retain at most one best/latest/recovery checkpoint inside this owned run."""

    for manifest_path in directory.glob(f"{run_id}-{role}-*.manifest.json"):
        if manifest_path.stem.removesuffix(".manifest") == current_id:
            continue
        stale = read_gfm_checkpoint_manifest(manifest_path)
        artifact = Path(stale.artifact_path)
        try:
            artifact.resolve().relative_to(directory.resolve())
        except ValueError as error:
            raise GfmTrainingError("Checkpoint retention resolved outside its run") from error
        artifact.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)


def _rotate_latest_to_recovery(directory: Path, *, run_id: str) -> None:
    """Copy the prior latest state into one recovery identity before replacement."""

    latest_paths = sorted(directory.glob(f"{run_id}-latest-*.manifest.json"))
    if not latest_paths:
        return
    if len(latest_paths) != 1:
        raise GfmTrainingError("Checkpoint directory has ambiguous latest states")
    latest = read_gfm_checkpoint_manifest(latest_paths[0])
    payload = load_gfm_checkpoint(latest, map_location="cpu")
    recovery_id = f"{run_id}-recovery-{latest.step}"
    save_gfm_checkpoint(
        directory,
        checkpoint_id=recovery_id,
        run_id=run_id,
        epoch=latest.epoch,
        step=latest.step,
        components=payload["components"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=payload["scaler_state"],
        sampler_state=payload["sampler_state"],
        best_state=payload["best_state"],
        config=payload["config"],
        corpus_hashes=payload["corpus_hashes"],
        rng_state=payload["rng_state"],
    )
    _retain_checkpoint_roles(
        directory,
        run_id=run_id,
        current_id=recovery_id,
        role="recovery",
    )


def _train_run(
    *,
    layout: RuntimeLayout,
    experiment_id: str,
    config: GfmPretrainConfig,
    corpora: Sequence[GfmDomainCorpusManifest],
    protocols: Sequence[GfmTaskProtocolManifest],
    embeddings: Mapping[str, _BoundedEmbeddingStore],
    phase: TrainingPhase,
    variant: str,
    seed: int,
    device: str,
    resume_manifest: str | Path | None = None,
    retry_without_checkpoint: bool = False,
) -> dict[str, Any]:
    import torch

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.trainer import CoreTrainer, CoreTrainerConfig

    set_seed(seed, device)
    streams = _make_domain_streams(layout, embeddings, maximum_role="validation")
    expected_corpus_by_domain = {corpus.domain_id: corpus.content_hash for corpus in corpora}
    if {
        domain: str(stream.manifest.get("logicalHash")) for domain, stream in streams.items()
    } != expected_corpus_by_domain:
        raise ContractViolation("Physical domain views differ from the registered corpus contracts")
    domain_access_audits = {domain: stream.access_audit for domain, stream in streams.items()}
    if any(
        not isinstance(audit, dict) or audit.get("testArtifactsOpened") is not False
        for audit in domain_access_audits.values()
    ):
        raise ContractViolation("Training opened a test-restricted numeric artifact")
    phase_config = config.dev if phase == "dev" else config.formal
    run_id = f"{experiment_id}-{variant}-{seed}"
    run_dir = layout.gfm_runs / experiment_id / run_id
    if resume_manifest is not None and retry_without_checkpoint:
        raise ContractViolation(
            "A pretrain run cannot resume a checkpoint and retry preflight together"
        )
    recovering = resume_manifest is not None or retry_without_checkpoint
    if run_dir.exists() and not recovering:
        raise GfmTrainingError(
            f"Run directory already exists for {run_id}; use gfm-resume or a new config"
        )
    if recovering and not run_dir.is_dir():
        raise GfmTrainingError("Interrupted pretrain run directory is absent")
    current_code_hash = code_identity_hash()
    current_environment_hash = _environment_hash(device)
    corpus_hashes = tuple(corpus.logical_hash for corpus in corpora)
    embedding_artifacts = _embedding_artifact_evidence(embeddings)
    embedding_artifacts_hash = canonical_sha256(embedding_artifacts)
    domain_access_audits_hash = canonical_sha256(domain_access_audits)
    started = datetime.now(UTC)
    state_base: dict[str, Any] = {
        "schemaVersion": "gfm.workflow-run-state/1.0",
        "runKind": "pretrain",
        "runId": run_id,
        "experimentId": experiment_id,
        "phase": phase,
        "variant": variant,
        "seed": seed,
        "device": device,
        "configHash": config.config_hash,
        "codeHash": current_code_hash,
        "environmentHash": current_environment_hash,
        "corpusHashes": corpus_hashes,
        "embeddingArtifacts": embedding_artifacts,
        "embeddingArtifactsHash": embedding_artifacts_hash,
        "domainAccessAudits": domain_access_audits,
        "domainAccessAuditsHash": domain_access_audits_hash,
        "startedAt": started.isoformat(),
    }
    run_state: dict[str, Any] | None = None
    preflight_attempt = 1
    if recovering:
        run_state = read_json_object(run_dir / "run-state.json")
        allowed_statuses = {"preflight", "running"} if retry_without_checkpoint else {"running"}
        if (
            run_state.get("runId") != run_id
            or run_state.get("experimentId") != experiment_id
            or run_state.get("phase") != phase
            or run_state.get("variant") != variant
            or run_state.get("seed") != seed
            or run_state.get("device") != device
            or run_state.get("configHash") != config.config_hash
            or run_state.get("codeHash") != current_code_hash
            or run_state.get("environmentHash") != current_environment_hash
            or tuple(run_state.get("corpusHashes", ())) != corpus_hashes
            or run_state.get("embeddingArtifactsHash") != embedding_artifacts_hash
            or run_state.get("domainAccessAuditsHash") != domain_access_audits_hash
            or run_state.get("status") not in allowed_statuses
            or run_state.get("runKind") not in (None, "pretrain")
        ):
            raise GfmTrainingError(
                "Interrupted run provenance differs from resume arguments/runtime"
            )
        try:
            started = datetime.fromisoformat(str(run_state["startedAt"]))
        except (KeyError, TypeError, ValueError) as error:
            raise GfmTrainingError("Interrupted run has an invalid durable start time") from error
        state_base["startedAt"] = started.isoformat()
        if retry_without_checkpoint:
            _ensure_checkpoint_free_retry_directory(run_dir)
            raw_attempt = run_state.get("preflightAttemptCount", 1)
            if isinstance(raw_attempt, bool) or not isinstance(raw_attempt, int) or raw_attempt < 1:
                raise GfmTrainingError("Interrupted preflight attempt counter is invalid")
            preflight_attempt = raw_attempt + 1
        else:
            batch_value = run_state.get("batchSize")
            accumulation_value = run_state.get("gradientAccumulation")
            probe_value = run_state.get("probePeakCudaMemoryMiB")
            if (
                isinstance(batch_value, bool)
                or not isinstance(batch_value, int)
                or batch_value < 1
                or isinstance(accumulation_value, bool)
                or not isinstance(accumulation_value, int)
                or accumulation_value
                != math.ceil(config.optimization.effective_batch_size / batch_value)
                or isinstance(probe_value, bool)
                or not isinstance(probe_value, (int, float))
                or not math.isfinite(float(probe_value))
                or float(probe_value) < 0.0
            ):
                raise GfmTrainingError("Interrupted run has invalid resolved batch state")
            batch_size = batch_value
            accumulation = accumulation_value
            probe_peak = float(probe_value)
    if resume_manifest is None:
        batch_size, probe_peak = _probe_with_durable_preflight(
            state_path=run_dir / "run-state.json",
            state_base=state_base,
            attempt=preflight_attempt,
            config=config,
            variant=variant,
            streams=streams,
            device=device,
            seed=seed,
        )
        accumulation = math.ceil(config.optimization.effective_batch_size / batch_size)
    model = SocialGraphFMCore(_model_config(config, variant))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    trainer = CoreTrainer(
        model,
        optimizer,
        CoreTrainerConfig(
            gradient_accumulation_steps=accumulation,
            gradient_clip=config.optimization.gradient_clip,
            amp=True,
        ),
        device,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_cosine(
            step,
            maximum=phase_config.max_steps,
            warmup_ratio=config.optimization.warmup_ratio,
        ),
    )
    fanout = (
        int(config.architecture.neighbor_fanout[0]),
        int(config.architecture.neighbor_fanout[1]),
    )
    best_metric = math.inf
    best_step = 0
    best_manifest = None
    no_improvement = 0
    last_losses: dict[str, float] = {}
    best_fixed_sample_digest: str | None = None
    best_fixed_losses: dict[str, float] = {}
    if resume_manifest is not None:
        assert run_state is not None
        checkpoint = read_gfm_checkpoint_manifest(resume_manifest)
        if (
            checkpoint.run_id != run_id
            or checkpoint.config_hash != config.config_hash
            or tuple(checkpoint.corpus_hashes) != corpus_hashes
        ):
            raise GfmTrainingError("Resume checkpoint provenance differs from run state")
        payload = load_gfm_checkpoint(checkpoint, map_location=device)
        model.load_state_dict(payload["components"]["core"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if payload["scheduler_state"] is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        trainer.scaler.load_state_dict(payload["scaler_state"] or {})
        sampler_state = payload["sampler_state"]
        trainer.scheduler.load_state_dict(dict(sampler_state["roundRobin"]))
        trainer.global_step = int(sampler_state["globalStep"])
        trainer.optimizer_step = int(sampler_state["optimizerStep"])
        if (
            int(sampler_state.get("batchSize", -1)) != batch_size
            or int(sampler_state.get("gradientAccumulation", -1)) != accumulation
            or sampler_state.get("embeddingArtifactsHash") != embedding_artifacts_hash
            or sampler_state.get("embeddingArtifacts") != embedding_artifacts
        ):
            raise GfmTrainingError("Resume checkpoint batch state differs from durable run state")
        for domain, stream_state in sampler_state["streams"].items():
            streams[domain].load_state_dict(stream_state)
        from ..checkpoint import restore_rng_state

        restore_rng_state(payload["rng_state"])
        best_metric = float(payload["best_state"]["validationLoss"])
        best_step = int(payload["best_state"]["step"])
        best_fixed_sample_digest = payload["best_state"].get("expectedFixedSampleDigest")
        best_fixed_losses = {
            str(name): float(value)
            for name, value in payload["best_state"].get("fixedValidationLosses", {}).items()
        }
        no_improvement = int(sampler_state.get("noImprovement", 0))
        if "best" in checkpoint.checkpoint_id:
            best_manifest = checkpoint
        else:
            best_paths = sorted((run_dir / "checkpoints").glob(f"{run_id}-best-*.manifest.json"))
            if len(best_paths) != 1:
                raise GfmTrainingError(
                    "Interrupted run must retain exactly one validation-selected best checkpoint"
                )
            best_manifest = read_gfm_checkpoint_manifest(best_paths[0])
            load_gfm_checkpoint(best_manifest, map_location="cpu")
            if best_manifest.step > checkpoint.step:
                raise GfmTrainingError(
                    "Resume state predates its validation-selected best checkpoint"
                )
    else:
        _save_run_state(
            run_dir / "run-state.json",
            {
                **state_base,
                "batchSize": batch_size,
                "gradientAccumulation": accumulation,
                "probePeakCudaMemoryMiB": probe_peak,
                "preflightAttemptCount": preflight_attempt,
                "status": "running",
            },
        )
    if int(scheduler.last_epoch) != int(trainer.optimizer_step):
        raise GfmTrainingError(
            "Learning-rate scheduler step differs from optimizer checkpoint progress"
        )
    while trainer.optimizer_step < phase_config.max_steps:
        remaining = phase_config.max_steps - trainer.optimizer_step
        optimizer_steps = min(phase_config.evaluation_every_steps, remaining)
        microbatches = optimizer_steps * accumulation
        quotient, remainder = divmod(microbatches, len(streams))

        def batches(domain: str, count: int) -> Iterator[Any]:
            stream = streams[domain]
            for offset in range(count):
                yield _core_batch(
                    stream,
                    batch_size=batch_size,
                    fanout=fanout,
                    seed=seed + trainer.global_step + offset,
                )

        loaders = {
            domain: batches(domain, quotient + (index < remainder))
            for index, domain in enumerate(streams)
        }
        result = _train_epoch_with_heartbeats(
            trainer,
            loaders,
            every_optimizer_steps=50,
            after_optimizer_step=scheduler.step,
            heartbeat=lambda losses: _pretrain_heartbeat(
                state_path=run_dir / "run-state.json",
                state_base=state_base,
                batch_size=batch_size,
                accumulation=accumulation,
                probe_peak=probe_peak,
                preflight_attempt=preflight_attempt,
                trainer=trainer,
                streams=streams,
                started=started,
                device=device,
                last_losses=losses,
            ),
        )
        if int(scheduler.last_epoch) != int(trainer.optimizer_step):
            raise GfmTrainingError(
                "Learning-rate scheduler did not advance with every optimizer step"
            )
        last_losses = dict(result.mean_losses)
        validation_losses: dict[str, float] = {}
        for domain, stream in streams.items():
            validation_start, validation_upper = _stream_role_bounds(stream, 1)
            validation = _core_batch(
                stream,
                batch_size=min(batch_size, 512),
                fanout=fanout,
                seed=seed,
                cursor=validation_start,
                upper_index=validation_upper,
                advance=False,
                split_role=1,
            )
            validation_losses[domain] = trainer.evaluate_batch(validation)["total"]
        validation_metric = float(sum(validation_losses.values()) / len(validation_losses))
        improved = validation_metric < best_metric
        if improved:
            best_metric = validation_metric
            best_step = trainer.optimizer_step
            best_fixed_losses = dict(validation_losses)
            best_fixed_sample_digest = canonical_sha256(
                {
                    "seed": seed,
                    "variant": variant,
                    "losses": best_fixed_losses,
                    "samplePolicy": "first-validation-target-window-batch-role-aware-v2",
                }
            )
            no_improvement = 0
        else:
            no_improvement += 1
        checkpoint_directory = run_dir / "checkpoints"
        checkpoint_epoch = max(stream.epoch for stream in streams.values())
        checkpoint_components = {"core": model.state_dict()}
        checkpoint_sampler = {
            "streams": {domain: stream.state_dict() for domain, stream in streams.items()},
            "roundRobin": trainer.scheduler.state_dict(),
            "globalStep": trainer.global_step,
            "optimizerStep": trainer.optimizer_step,
            "batchSize": batch_size,
            "gradientAccumulation": accumulation,
            "noImprovement": no_improvement,
            "embeddingArtifacts": embedding_artifacts,
            "embeddingArtifactsHash": embedding_artifacts_hash,
            "domainAccessAudits": domain_access_audits,
            "domainAccessAuditsHash": canonical_sha256(domain_access_audits),
        }
        checkpoint_best_state = {
            "validationLoss": best_metric,
            "step": best_step,
            "embeddingArtifactsHash": embedding_artifacts_hash,
            "fixedValidationLosses": best_fixed_losses,
            "expectedFixedSampleDigest": best_fixed_sample_digest,
        }
        checkpoint_corpora = corpus_hashes
        _rotate_latest_to_recovery(checkpoint_directory, run_id=run_id)
        latest_id = f"{run_id}-latest-{trainer.optimizer_step}"
        save_gfm_checkpoint(
            checkpoint_directory,
            checkpoint_id=latest_id,
            run_id=run_id,
            epoch=checkpoint_epoch,
            step=trainer.optimizer_step,
            components=checkpoint_components,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            scaler_state=trainer.scaler.state_dict(),
            sampler_state=checkpoint_sampler,
            best_state=checkpoint_best_state,
            config=config.logical_payload(),
            corpus_hashes=checkpoint_corpora,
        )
        _retain_checkpoint_roles(
            checkpoint_directory,
            run_id=run_id,
            current_id=latest_id,
            role="latest",
        )
        if improved:
            best_id = f"{run_id}-best-{trainer.optimizer_step}"
            best_manifest = save_gfm_checkpoint(
                checkpoint_directory,
                checkpoint_id=best_id,
                run_id=run_id,
                epoch=checkpoint_epoch,
                step=trainer.optimizer_step,
                components=checkpoint_components,
                optimizer_state=optimizer.state_dict(),
                scheduler_state=scheduler.state_dict(),
                scaler_state=trainer.scaler.state_dict(),
                sampler_state=checkpoint_sampler,
                best_state=checkpoint_best_state,
                config=config.logical_payload(),
                corpus_hashes=checkpoint_corpora,
            )
            _retain_checkpoint_roles(
                checkpoint_directory,
                run_id=run_id,
                current_id=best_id,
                role="best",
            )
        if (
            trainer.optimizer_step >= phase_config.minimum_steps
            and no_improvement >= phase_config.patience_evaluations
        ):
            break
    if best_manifest is None:
        raise GfmTrainingError("Training completed without a finite validation checkpoint")
    if best_fixed_sample_digest is None or not best_fixed_losses:
        raise GfmTrainingError("Best checkpoint lacks a frozen validation digest")
    # Formal test bytes remain physically unopened in this training process.
    # The orchestrator launches a one-shot evaluator only after this selected
    # checkpoint and terminal run manifest have been durably registered.
    test_metric: float | None = None
    peak = probe_peak
    if device == "cuda":
        peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
    if peak >= config.optimization.cuda_memory_limit_mib:
        raise GfmTrainingError("Training exceeded the fixed 7168 MiB CUDA memory limit")
    finished = datetime.now(UTC)
    # The durable terminal audits describe the validation-selected checkpoint,
    # not a later non-improving trainer state.  Re-read that immutable payload
    # so terminal state, registry checkpoint and future reuse share one exact
    # authority even when early stopping finishes after the best step.
    selected_payload = load_gfm_checkpoint(best_manifest, map_location="cpu")
    selected_sampler = selected_payload.get("sampler_state")
    selected_streams = (
        selected_sampler.get("streams") if isinstance(selected_sampler, dict) else None
    )
    if not isinstance(selected_streams, dict) or set(selected_streams) != set(streams):
        raise GfmTrainingError("Selected best checkpoint lacks all domain audit states")
    negative_sampling_audits = {}
    for domain, stream_state in selected_streams.items():
        audit = (
            stream_state.get("negativeSamplingAudit") if isinstance(stream_state, dict) else None
        )
        if not isinstance(audit, dict):
            raise GfmTrainingError("Selected best checkpoint has invalid sampling audits")
        negative_sampling_audits[str(domain)] = deepcopy(audit)
    run_manifest = GfmRunManifest.create(
        runId=run_id,
        experimentId=experiment_id,
        phase="pretrain",
        architectureVariant=variant,
        status="succeeded",
        domainIds=tuple(streams),
        seed=seed,
        codeHash=current_code_hash,
        environmentHash=current_environment_hash,
        configHash=config.config_hash,
        corpusHashes=corpus_hashes,
        taskProtocolHashes=tuple(protocol.protocol_hash for protocol in protocols),
        startedAt=started,
        finishedAt=finished,
        peakCudaMemoryMiB=peak,
        artifactPaths=(
            str(run_dir / "run-state.json"),
            str(run_dir / "checkpoints" / f"{best_manifest.checkpoint_id}.manifest.json"),
        ),
    )
    with exclusive_file_lock(run_dir / ".terminal-state.lock"):
        _write_contract(run_dir / "run-manifest.json", run_manifest)
        registry = _registry(layout)
        registry.record_completed_run(run_manifest, best_manifest)
        _save_run_state(
            run_dir / "run-state.json",
            {
                **state_base,
                "batchSize": batch_size,
                "gradientAccumulation": accumulation,
                "probePeakCudaMemoryMiB": probe_peak,
                "preflightAttemptCount": preflight_attempt,
                "negativeSamplingAudits": negative_sampling_audits,
                "negativeSamplingAuditsHash": canonical_sha256(negative_sampling_audits),
                "status": "succeeded",
                "bestCheckpointManifest": str(
                    run_dir / "checkpoints" / f"{best_manifest.checkpoint_id}.manifest.json"
                ),
            },
        )
    return {
        "runId": run_id,
        "variant": variant,
        "seed": seed,
        "steps": trainer.optimizer_step,
        "batchSize": batch_size,
        "gradientAccumulation": accumulation,
        "bestValidationLoss": best_metric,
        "bestCheckpointId": best_manifest.checkpoint_id,
        "peakCudaMemoryMiB": peak,
        "lastLosses": last_losses,
        "negativeSamplingAudits": negative_sampling_audits,
        "testLoss": test_metric,
        "testRead": False,
    }


def _pretrain_worker(
    *,
    root: str | Path,
    phase: TrainingPhase,
    config: str | Path | None,
    variant: str,
    seed: int,
    device: str,
    resume_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Execute exactly one run; the formal orchestrator invokes this in a new process."""

    if phase == "formal" and device != "cuda":
        raise ContractViolation("Formal SocialGraph-FM Core runs require CUDA")
    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    checked_config = _load_pretrain_config(config, None)
    corpora, embeddings = _ensure_pretrain_evidence(
        layout,
        formal_required=phase == "formal",
        maximum_role="validation",
        physical_boundary=True,
    )
    protocols = _register_prerequisites(layout, corpora)
    experiment_id = _experiment_id(phase=phase, config=checked_config, corpora=corpora)
    if variant not in checked_config.architecture.candidates:
        raise ContractViolation(f"Unknown SocialGraph-FM Core candidate {variant}")
    expected_seeds = checked_config.dev.seeds if phase == "dev" else checked_config.formal.seeds
    if seed not in expected_seeds:
        raise ContractViolation("Worker seed is outside the checked phase contract")
    return _train_run(
        layout=layout,
        experiment_id=experiment_id,
        config=checked_config,
        corpora=corpora,
        protocols=protocols,
        embeddings=embeddings,
        phase=phase,
        variant=variant,
        seed=seed,
        device=device,
        resume_manifest=resume_manifest,
    )


def _subprocess_json(arguments: Sequence[str]) -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, "-m", "socialgraph_gfm.cli", *arguments],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    lines = [line for line in process.stdout.splitlines() if line.strip()]
    if process.returncode != 0 or not lines:
        detail = process.stderr.strip() or process.stdout.strip() or "no subprocess output"
        raise GfmTrainingError(
            f"Independent GFM process failed with exit {process.returncode}: {detail[-2000:]}"
        )
    try:
        payload = json.loads(lines[-1])
    except (TypeError, json.JSONDecodeError) as error:
        raise GfmTrainingError("Independent GFM process returned invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("ok") is False:
        raise GfmTrainingError("Independent GFM process did not report success")
    return payload


@dataclass(frozen=True)
class _CompletedRunExpectation:
    """Exact immutable provenance required before a matrix cell may be reused."""

    experiment_id: str
    run_id: str
    phase: Literal["pretrain", "adapt", "lodo"]
    variant: Literal["core-base", "core-moe"]
    seed: int
    domain_ids: tuple[str, ...]
    corpus_hashes: tuple[str, ...]
    protocol_hashes: tuple[str, ...]
    config_hash: str
    code_hash: str
    environment_hash: str
    pretrain_phase: TrainingPhase | None = None
    held_out_domain: str | None = None
    required_reports: tuple[tuple[str, str], ...] = ()
    current_embedding_artifacts: Mapping[str, Any] | None = None
    current_embedding_artifacts_hash: str | None = None


@dataclass(frozen=True)
class _CompletedRunEvidence:
    run: GfmRunManifest
    checkpoint: Any
    checkpoint_payload: dict[str, Any]
    reports: dict[str, GfmEvaluationReport]
    run_state: dict[str, Any] | None


def _path_within(path: str | Path, parent: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as error:
        raise GfmTrainingError(f"{label} escaped its immutable run boundary") from error
    return resolved


def _existing_run_interruption_error(run_id: str) -> GfmTrainingError:
    return GfmTrainingError(
        f"Run {run_id} is interrupted and must be resumed explicitly with "
        f"gfm-resume --run-id {run_id}; the matrix orchestrator will not overwrite it"
    )


def _required_pretrain_reports(run_id: str, phase: TrainingPhase) -> tuple[tuple[str, str], ...]:
    if phase == "dev":
        return ()
    return (
        (f"{run_id}-frozen-test", "in_domain"),
        (f"{run_id}-fresh-process", "fresh_process"),
    )


def _pretrain_state_mismatches(
    state: Mapping[str, Any],
    expectation: _CompletedRunExpectation,
    registered: GfmRunManifest,
) -> list[str]:
    expected_phase = expectation.pretrain_phase
    if expected_phase not in ("dev", "formal"):
        raise GfmTrainingError("Completed pretrain expectation lacks its dev/formal phase")
    expected_values = {
        "schemaVersion": (
            state.get("schemaVersion"),
            "gfm.workflow-run-state/1.0",
        ),
        "runKind": (state.get("runKind"), "pretrain"),
        "runId": (state.get("runId"), expectation.run_id),
        "experimentId": (state.get("experimentId"), expectation.experiment_id),
        "phase": (state.get("phase"), expected_phase),
        "variant": (state.get("variant"), expectation.variant),
        "seed": (state.get("seed"), expectation.seed),
        "configHash": (state.get("configHash"), expectation.config_hash),
        "codeHash": (state.get("codeHash"), expectation.code_hash),
        "environmentHash": (
            state.get("environmentHash"),
            expectation.environment_hash,
        ),
        "corpusHashes": (
            tuple(state.get("corpusHashes", ())),
            expectation.corpus_hashes,
        ),
    }
    mismatches = [
        name for name, (actual, expected) in expected_values.items() if actual != expected
    ]
    state_embeddings = state.get("embeddingArtifacts")
    state_embeddings_hash = state.get("embeddingArtifactsHash")
    if (
        not isinstance(state_embeddings, dict)
        or not isinstance(state_embeddings_hash, str)
        or state_embeddings_hash != canonical_sha256(state_embeddings)
    ):
        mismatches.append("embeddingArtifactsHash")
    expected_embeddings = expectation.current_embedding_artifacts
    expected_embeddings_hash = expectation.current_embedding_artifacts_hash
    if (expected_embeddings is None) != (expected_embeddings_hash is None):
        raise GfmTrainingError("Pretrain expectation has incomplete embedding provenance")
    if expected_embeddings is not None:
        if (
            expected_embeddings_hash != canonical_sha256(expected_embeddings)
            or state_embeddings != expected_embeddings
            or state_embeddings_hash != expected_embeddings_hash
        ):
            mismatches.append("embeddingArtifacts")
    state_access = state.get("domainAccessAudits")
    state_access_hash = state.get("domainAccessAuditsHash")
    if (
        not isinstance(state_access, dict)
        or not isinstance(state_access_hash, str)
        or state_access_hash != canonical_sha256(state_access)
    ):
        mismatches.append("domainAccessAuditsHash")
    try:
        state_started = datetime.fromisoformat(str(state.get("startedAt")))
    except (TypeError, ValueError):
        mismatches.append("startedAt")
    else:
        if state_started != registered.started_at:
            mismatches.append("startedAt")
    if state.get("device") not in {"cpu", "cuda"}:
        mismatches.append("device")
    return sorted(set(mismatches))


def _pretrain_completion_evidence(
    *,
    registered: GfmRunManifest,
    checkpoint: Any,
    reports: Mapping[str, GfmEvaluationReport],
) -> dict[str, Any]:
    return {
        "schemaVersion": "gfm.pretrain-terminal-reconciliation/1.0",
        "runManifestHash": registered.manifest_hash,
        "checkpointId": checkpoint.checkpoint_id,
        "checkpointLogicalHash": checkpoint.logical_hash,
        "checkpointStateHash": checkpoint.state_hash,
        "requiredReportHashes": {
            report_id: report.report_hash for report_id, report in sorted(reports.items())
        },
    }


def _validate_pretrain_terminal_audits(
    state: Mapping[str, Any], *, checkpoint_payload: Mapping[str, Any] | None = None
) -> None:
    audits = state.get("negativeSamplingAudits")
    if not isinstance(audits, dict) or state.get("negativeSamplingAuditsHash") != canonical_sha256(
        audits
    ):
        raise GfmTrainingError(
            "Terminal pretrain state has invalid negative-sampling audit evidence"
        )
    if checkpoint_payload is None:
        return
    sampler = checkpoint_payload.get("sampler_state")
    best = checkpoint_payload.get("best_state")
    if not isinstance(sampler, dict) or not isinstance(best, dict):
        raise GfmTrainingError("Terminal pretrain checkpoint lacks trainer provenance")
    sampler_embeddings = sampler.get("embeddingArtifacts")
    sampler_embeddings_hash = sampler.get("embeddingArtifactsHash")
    sampler_access = sampler.get("domainAccessAudits")
    sampler_access_hash = sampler.get("domainAccessAuditsHash")
    if (
        not isinstance(sampler_embeddings, dict)
        or sampler_embeddings_hash != canonical_sha256(sampler_embeddings)
        or state.get("embeddingArtifacts") != sampler_embeddings
        or state.get("embeddingArtifactsHash") != sampler_embeddings_hash
        or best.get("embeddingArtifactsHash") != sampler_embeddings_hash
        or not isinstance(sampler_access, dict)
        or sampler_access_hash != canonical_sha256(sampler_access)
        or state.get("domainAccessAudits") != sampler_access
        or state.get("domainAccessAuditsHash") != sampler_access_hash
    ):
        raise GfmTrainingError(
            "Terminal pretrain state differs from checkpoint embedding/access provenance"
        )
    stream_states = sampler.get("streams")
    if not isinstance(stream_states, dict):
        raise GfmTrainingError("Terminal pretrain checkpoint lacks stream audit provenance")
    checkpoint_audits = {
        str(domain): stream.get("negativeSamplingAudit")
        for domain, stream in stream_states.items()
        if isinstance(stream, dict)
    }
    if set(checkpoint_audits) != set(stream_states) or audits != checkpoint_audits:
        raise GfmTrainingError(
            "Terminal pretrain state differs from checkpoint negative-sampling audits"
        )


def _reconcile_pretrain_terminal_state(
    *,
    run_dir: Path,
    expectation: _CompletedRunExpectation,
    registered: GfmRunManifest,
    state: dict[str, Any],
    checkpoint: Any,
    checkpoint_payload: Mapping[str, Any],
    checkpoint_manifest_path: Path,
    reports: Mapping[str, GfmEvaluationReport],
) -> dict[str, Any]:
    """Repair only the final state marker proven by immutable completion evidence."""

    if expectation.pretrain_phase not in ("dev", "formal"):
        raise GfmTrainingError("Pretrain reconciliation lacks its dev/formal phase")
    expected_reports = _required_pretrain_reports(expectation.run_id, expectation.pretrain_phase)
    if expectation.required_reports != expected_reports:
        raise GfmTrainingError(
            "Pretrain reconciliation was not supplied the complete phase evidence set"
        )
    if state.get("status") != "running":
        raise GfmTrainingError(
            "Only a running pretrain marker may be reconciled from terminal evidence"
        )
    if _pretrain_state_mismatches(state, expectation, registered):
        raise GfmTrainingError(
            "Interrupted pretrain state provenance is not eligible for reconciliation"
        )
    sampler = checkpoint_payload.get("sampler_state")
    best = checkpoint_payload.get("best_state")
    if not isinstance(sampler, dict) or not isinstance(best, dict):
        raise GfmTrainingError("Registered pretrain checkpoint lacks terminal trainer state")
    if (
        not checkpoint.checkpoint_id.startswith(f"{expectation.run_id}-best-")
        or sampler.get("optimizerStep") != checkpoint.step
        or best.get("step") != checkpoint.step
        or state.get("batchSize") != sampler.get("batchSize")
        or state.get("gradientAccumulation") != sampler.get("gradientAccumulation")
        or state.get("embeddingArtifacts") != sampler.get("embeddingArtifacts")
        or state.get("embeddingArtifactsHash") != sampler.get("embeddingArtifactsHash")
        or state.get("domainAccessAudits") != sampler.get("domainAccessAudits")
        or state.get("domainAccessAuditsHash") != sampler.get("domainAccessAuditsHash")
    ):
        raise GfmTrainingError(
            "Registered pretrain checkpoint cannot prove the running state provenance"
        )
    batch_size = state.get("batchSize")
    accumulation = state.get("gradientAccumulation")
    probe_peak = state.get("probePeakCudaMemoryMiB")
    attempt = state.get("preflightAttemptCount")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(accumulation, bool)
        or not isinstance(accumulation, int)
        or accumulation < 1
        or isinstance(probe_peak, bool)
        or not isinstance(probe_peak, (int, float))
        or not math.isfinite(float(probe_peak))
        or float(probe_peak) < 0.0
        or isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
    ):
        raise GfmTrainingError("Running pretrain state lacks valid resolved batch evidence")
    stream_states = sampler.get("streams")
    if not isinstance(stream_states, dict) or set(stream_states) != set(expectation.domain_ids):
        raise GfmTrainingError("Registered checkpoint lacks all pretrain stream states")
    negative_sampling_audits: dict[str, Any] = {}
    for domain_id in expectation.domain_ids:
        stream_state = stream_states.get(domain_id)
        audit = (
            stream_state.get("negativeSamplingAudit") if isinstance(stream_state, dict) else None
        )
        if not isinstance(audit, dict):
            raise GfmTrainingError(
                "Registered checkpoint has invalid negative-sampling audit state"
            )
        negative_sampling_audits[domain_id] = deepcopy(audit)
    if expectation.pretrain_phase == "formal":
        test_state_path = run_dir / "test-read-state.json"
        if not test_state_path.is_file():
            raise GfmTrainingError(
                "Formal pretrain reconciliation lacks durable one-shot test evidence"
            )
        test_state = read_json_object(test_state_path)
        if (
            test_state.get("runId") != expectation.run_id
            or test_state.get("checkpointId") != checkpoint.checkpoint_id
            or test_state.get("status") != "completed"
            or test_state.get("readCount") != 1
            or not isinstance(test_state.get("resultHash"), str)
        ):
            raise GfmTrainingError(
                "Formal pretrain reconciliation has invalid one-shot test evidence"
            )
    completion_evidence = _pretrain_completion_evidence(
        registered=registered,
        checkpoint=checkpoint,
        reports=reports,
    )
    terminal_state = {
        **state,
        "negativeSamplingAudits": negative_sampling_audits,
        "negativeSamplingAuditsHash": canonical_sha256(negative_sampling_audits),
        "status": "succeeded",
        "bestCheckpointManifest": str(checkpoint_manifest_path),
        "terminalReconciliationEvidence": completion_evidence,
        "terminalReconciliationEvidenceHash": canonical_sha256(completion_evidence),
    }
    state_path = run_dir / "run-state.json"
    with exclusive_file_lock(run_dir / ".terminal-state.lock"):
        current = read_json_object(state_path)
        if current.get("status") == "succeeded":
            if _pretrain_state_mismatches(current, expectation, registered):
                raise GfmTrainingError("Concurrent terminal pretrain state has stale provenance")
            _validate_pretrain_terminal_audits(current, checkpoint_payload=checkpoint_payload)
            recorded = current.get("bestCheckpointManifest")
            if (
                not isinstance(recorded, str)
                or Path(recorded).resolve() != checkpoint_manifest_path.resolve()
            ):
                raise GfmTrainingError(
                    "Concurrent terminal pretrain state selects another checkpoint"
                )
            return current
        if current != state:
            raise GfmTrainingError("Running pretrain state changed while completion was reconciled")
        _save_run_state(state_path, terminal_state)
        checked = read_json_object(state_path)
        if checked != terminal_state:
            raise GfmTrainingError("Atomic pretrain terminal-state reconciliation failed")
    return checked


def _validate_completed_matrix_run(
    layout: RuntimeLayout,
    expectation: _CompletedRunExpectation,
) -> _CompletedRunEvidence | None:
    """Return a reusable cell only after re-reading every immutable authority.

    Absence means that the matrix cell has never started.  Any filesystem or
    registry trace that is not a complete, current, succeeded cell fails
    closed; in particular, this function never removes or overwrites a run.
    """

    registry = _registry(layout)
    run_dir = layout.gfm_runs / expectation.experiment_id / expectation.run_id
    registered = registry.get_run(expectation.run_id)
    if registered is None:
        if not run_dir.exists():
            return None
        state_path = run_dir / "run-state.json"
        if state_path.is_file():
            orphan_state = read_json_object(state_path)
            if orphan_state.get("runId") != expectation.run_id:
                raise GfmTrainingError(
                    f"Orphan run directory for {expectation.run_id} has a different identity"
                )
            if orphan_state.get("status") in {"preflight", "running"}:
                raise _existing_run_interruption_error(expectation.run_id)
        raise GfmTrainingError(
            f"Orphan or incomplete immutable run directory exists for "
            f"{expectation.run_id}; refusing automatic recovery or overwrite"
        )
    if registered.status == "running":
        raise _existing_run_interruption_error(expectation.run_id)
    if registered.status != "succeeded":
        raise GfmTrainingError(
            f"Run {expectation.run_id} is terminal with status "
            f"{registered.status}; refusing automatic retry or overwrite"
        )
    expected_values = {
        "experimentId": (registered.experiment_id, expectation.experiment_id),
        "phase": (registered.phase, expectation.phase),
        "architectureVariant": (
            registered.architecture_variant,
            expectation.variant,
        ),
        "seed": (registered.seed, expectation.seed),
        "domainIds": (registered.domain_ids, expectation.domain_ids),
        "heldOutDomain": (
            registered.held_out_domain,
            expectation.held_out_domain,
        ),
        "configHash": (registered.config_hash, expectation.config_hash),
        "codeHash": (registered.code_hash, expectation.code_hash),
        "environmentHash": (
            registered.environment_hash,
            expectation.environment_hash,
        ),
        "corpusHashes": (registered.corpus_hashes, expectation.corpus_hashes),
        "taskProtocolHashes": (
            registered.task_protocol_hashes,
            expectation.protocol_hashes,
        ),
    }
    mismatches = [
        name for name, (actual, expected) in expected_values.items() if actual != expected
    ]
    if mismatches:
        raise GfmTrainingError(
            f"Completed run {expectation.run_id} has stale provenance: " + ", ".join(mismatches)
        )
    if not run_dir.is_dir():
        raise GfmTrainingError(
            f"Registered run {expectation.run_id} has no immutable run directory"
        )
    run_manifest_path = run_dir / "run-manifest.json"
    if not run_manifest_path.is_file():
        raise GfmTrainingError(f"Registered run {expectation.run_id} has no run manifest artifact")
    disk_run = GfmRunManifest.model_validate(read_json_object(run_manifest_path))
    if disk_run != registered:
        raise GfmTrainingError(
            f"Run manifest artifact differs from registry for {expectation.run_id}"
        )

    state: dict[str, Any] | None = None
    reconcile_pretrain_state = False
    state_path = run_dir / "run-state.json"
    if expectation.phase in {"pretrain", "adapt", "lodo"}:
        if not state_path.is_file():
            raise GfmTrainingError(
                f"Completed run {expectation.run_id} lacks durable terminal state"
            )
        state = read_json_object(state_path)
        if expectation.phase == "pretrain":
            mismatches = _pretrain_state_mismatches(state, expectation, registered)
            if mismatches:
                raise GfmTrainingError(
                    f"Terminal run state has stale provenance for {expectation.run_id}: "
                    + ", ".join(mismatches)
                )
            if state.get("status") == "running":
                reconcile_pretrain_state = True
            elif state.get("status") != "succeeded":
                raise GfmTrainingError(
                    f"Terminal pretrain state for {expectation.run_id} is not succeeded "
                    "or safely reconcilable"
                )
        elif expectation.phase == "adapt":
            if (
                state.get("runId") != expectation.run_id
                or state.get("experimentId") != expectation.experiment_id
                or state.get("status") != "succeeded"
                or state.get("seed") != expectation.seed
                or state.get("variant") != expectation.variant
                or state.get("configHash") != expectation.config_hash
                or state.get("codeHash") != expectation.code_hash
                or state.get("environmentHash") != expectation.environment_hash
                or tuple(state.get("corpusHashes", ())) != expectation.corpus_hashes
            ):
                raise GfmTrainingError(
                    f"Terminal run state has stale provenance for {expectation.run_id}"
                )
            if state.get("task") not in {"collaboration", "newcomer"}:
                raise GfmTrainingError("Terminal product state lacks its task identity")
        else:
            checked_lodo = validate_lodo_run_state(state, allowed_statuses=("running", "succeeded"))
            if checked_lodo["status"] == "running":
                raise _existing_run_interruption_error(expectation.run_id)
            lodo_identity = checked_lodo["identity"]
            lodo_expected = {
                "experimentId": expectation.experiment_id,
                "runId": expectation.run_id,
                "heldOutDomain": expectation.held_out_domain,
                "sourceDomainIds": list(expectation.domain_ids),
                "architectureVariant": expectation.variant,
                "seed": expectation.seed,
                "configHash": expectation.config_hash,
                "codeHash": expectation.code_hash,
                "environmentHash": expectation.environment_hash,
                "corpusHashes": list(expectation.corpus_hashes),
                "protocolHashes": list(expectation.protocol_hashes),
            }
            if any(lodo_identity.get(key) != value for key, value in lodo_expected.items()):
                raise GfmTrainingError(
                    f"Terminal LODO state has stale provenance for {expectation.run_id}"
                )

    checkpoints = tuple(
        checkpoint
        for checkpoint in registry.list_checkpoints(experiment_id=expectation.experiment_id)
        if checkpoint.run_id == expectation.run_id
    )
    if len(checkpoints) != 1:
        raise GfmTrainingError(
            f"Completed run {expectation.run_id} must have exactly one registered best "
            f"checkpoint, found {len(checkpoints)}"
        )
    checkpoint = checkpoints[0]
    checkpoint_dir = run_dir / "checkpoints"
    manifest_path = checkpoint_dir / f"{checkpoint.checkpoint_id}.manifest.json"
    if not manifest_path.is_file():
        raise GfmTrainingError(f"Registered checkpoint manifest is absent for {expectation.run_id}")
    disk_checkpoint = read_gfm_checkpoint_manifest(manifest_path)
    if disk_checkpoint != checkpoint:
        raise GfmTrainingError(
            f"Checkpoint manifest artifact differs from registry for {expectation.run_id}"
        )
    _path_within(checkpoint.artifact_path, checkpoint_dir, label="Checkpoint artifact")
    if (
        checkpoint.config_hash != expectation.config_hash
        or checkpoint.corpus_hashes != expectation.corpus_hashes
    ):
        raise GfmTrainingError(
            f"Checkpoint provenance differs from completed run {expectation.run_id}"
        )
    checkpoint_payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
    if canonical_sha256(checkpoint_payload["config"]) != expectation.config_hash:
        raise GfmTrainingError(f"Checkpoint embedded config differs for {expectation.run_id}")
    if expectation.phase == "pretrain":
        sampler = checkpoint_payload.get("sampler_state")
        stream_states = sampler.get("streams") if isinstance(sampler, dict) else None
        access_audits = sampler.get("domainAccessAudits") if isinstance(sampler, dict) else None
        if (
            not isinstance(stream_states, dict)
            or set(stream_states) != set(expectation.domain_ids)
            or not isinstance(access_audits, dict)
            or set(access_audits) != set(expectation.domain_ids)
        ):
            raise GfmTrainingError(
                f"Completed pretrain checkpoint lacks exact domain provenance for "
                f"{expectation.run_id}"
            )
    if expectation.phase == "lodo":
        if state is None:
            raise GfmTrainingError("Completed LODO run lacks durable execution state")
        sampler = checkpoint_payload.get("sampler_state")
        execution = state.get("execution")
        if (
            not isinstance(sampler, dict)
            or not isinstance(execution, dict)
            or sampler.get("execution") != execution
            or sampler.get("executionHash") != canonical_sha256(execution)
            or sampler.get("roleViewsHash") != execution.get("roleViewsHash")
            or sampler.get("testReadCount") != 0
            or execution.get("testReadCount") != 0
        ):
            raise GfmTrainingError(
                f"Completed LODO checkpoint differs from durable execution for {expectation.run_id}"
            )
    if state is not None:
        if expectation.phase == "pretrain":
            if not reconcile_pretrain_state:
                recorded = state.get("bestCheckpointManifest")
                if (
                    not isinstance(recorded, str)
                    or Path(recorded).resolve() != manifest_path.resolve()
                ):
                    raise GfmTrainingError(
                        f"Terminal pretrain state does not select the registered checkpoint "
                        f"for {expectation.run_id}"
                    )
        elif state.get("bestCheckpointId") != checkpoint.checkpoint_id:
            raise GfmTrainingError(
                f"Terminal product state does not select the registered checkpoint "
                f"for {expectation.run_id}"
            )

    all_reports = {
        report.report_id: report
        for report in registry.list_evaluations(experiment_id=expectation.experiment_id)
        if report.run_id == expectation.run_id
    }
    required_ids = {report_id for report_id, _ in expectation.required_reports}
    missing = sorted(required_ids.difference(all_reports))
    if missing:
        raise GfmTrainingError(
            f"Completed run {expectation.run_id} lacks required immutable evidence: "
            + ", ".join(missing)
        )
    selected_reports: dict[str, GfmEvaluationReport] = {}
    for report_id, kind in expectation.required_reports:
        report = all_reports[report_id]
        if (
            report.evaluation_kind != kind
            or report.experiment_id != expectation.experiment_id
            or report.run_id != expectation.run_id
            or report.checkpoint_id != checkpoint.checkpoint_id
            or report.seed != expectation.seed
            or not report.leakage_audit_passed
        ):
            raise GfmTrainingError(f"Immutable evidence {report_id} has stale matrix provenance")
        if kind == "fresh_process" and (
            not report.fresh_process_verified
            or report.verification_digest is None
            or report.metrics.get("fresh_process_repeat_match") != 1.0
        ):
            raise GfmTrainingError(f"Fresh-process evidence is incomplete for {expectation.run_id}")
        # Re-recording identical content is a read-time integrity audit: the
        # registry re-hashes both evidence and leakage-audit artifacts.
        registry.record_evaluation(report)
        selected_reports[report_id] = report
    if state is not None and expectation.phase == "pretrain":
        completion_evidence = _pretrain_completion_evidence(
            registered=registered,
            checkpoint=checkpoint,
            reports=selected_reports,
        )
        if reconcile_pretrain_state:
            state = _reconcile_pretrain_terminal_state(
                run_dir=run_dir,
                expectation=expectation,
                registered=registered,
                state=state,
                checkpoint=checkpoint,
                checkpoint_payload=checkpoint_payload,
                checkpoint_manifest_path=manifest_path,
                reports=selected_reports,
            )
        else:
            _validate_pretrain_terminal_audits(state, checkpoint_payload=checkpoint_payload)
            reconciliation = state.get("terminalReconciliationEvidence")
            reconciliation_hash = state.get("terminalReconciliationEvidenceHash")
            if reconciliation is not None or reconciliation_hash is not None:
                if reconciliation != completion_evidence or reconciliation_hash != canonical_sha256(
                    completion_evidence
                ):
                    raise GfmTrainingError(
                        f"Terminal reconciliation evidence is stale for {expectation.run_id}"
                    )
    return _CompletedRunEvidence(
        run=registered,
        checkpoint=checkpoint,
        checkpoint_payload=checkpoint_payload,
        reports=selected_reports,
        run_state=state,
    )


def _reuse_pretrain_matrix_cell(
    *,
    layout: RuntimeLayout,
    experiment_id: str,
    phase: TrainingPhase,
    config: GfmPretrainConfig,
    corpora: Sequence[GfmDomainCorpusManifest],
    protocols: Sequence[GfmTaskProtocolManifest],
    embeddings: Mapping[str, _BoundedEmbeddingStore],
    variant: Literal["core-base", "core-moe"],
    seed: int,
    device: str,
) -> dict[str, Any] | None:
    run_id = f"{experiment_id}-{variant}-{seed}"
    required_reports = _required_pretrain_reports(run_id, phase)
    current_embedding_artifacts = _embedding_artifact_evidence(embeddings)
    current_embedding_artifacts_hash = canonical_sha256(current_embedding_artifacts)
    evidence = _validate_completed_matrix_run(
        layout,
        _CompletedRunExpectation(
            experiment_id=experiment_id,
            run_id=run_id,
            phase="pretrain",
            variant=variant,
            seed=seed,
            domain_ids=tuple(DOMAIN_IDS.values()),
            corpus_hashes=tuple(corpus.logical_hash for corpus in corpora),
            protocol_hashes=tuple(protocol.protocol_hash for protocol in protocols),
            config_hash=config.config_hash,
            code_hash=code_identity_hash(),
            environment_hash=_environment_hash(device),
            pretrain_phase=phase,
            required_reports=required_reports,
            current_embedding_artifacts=current_embedding_artifacts,
            current_embedding_artifacts_hash=current_embedding_artifacts_hash,
        ),
    )
    if evidence is None:
        return None
    best = evidence.checkpoint_payload.get("best_state")
    sampler = evidence.checkpoint_payload.get("sampler_state")
    if not isinstance(best, dict) or not isinstance(sampler, dict):
        raise GfmTrainingError(f"Completed pretrain run {run_id} lacks trainer state")
    validation_loss = best.get("validationLoss")
    if (
        isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(float(validation_loss))
    ):
        raise GfmTrainingError(f"Completed pretrain run {run_id} has invalid validation evidence")
    summary: dict[str, Any] = {
        "runId": run_id,
        "variant": variant,
        "seed": seed,
        "steps": int(sampler.get("optimizerStep", evidence.checkpoint.step)),
        "batchSize": int(sampler.get("batchSize", 0)),
        "gradientAccumulation": int(sampler.get("gradientAccumulation", 0)),
        "bestValidationLoss": float(validation_loss),
        "bestCheckpointId": evidence.checkpoint.checkpoint_id,
        "peakCudaMemoryMiB": evidence.run.peak_cuda_memory_mib,
        "lastLosses": {},
        "testLoss": None,
        "testRead": False,
        "reused": True,
    }
    if phase == "formal":
        test_state_path = layout.gfm_runs / experiment_id / run_id / "test-read-state.json"
        if not test_state_path.is_file():
            raise GfmTrainingError(
                f"Completed formal run {run_id} lacks durable one-shot test state"
            )
        test_state = read_json_object(test_state_path)
        if (
            test_state.get("runId") != run_id
            or test_state.get("checkpointId") != evidence.checkpoint.checkpoint_id
            or test_state.get("status") != "completed"
            or test_state.get("readCount") != 1
            or not isinstance(test_state.get("resultHash"), str)
        ):
            raise GfmTrainingError(
                f"Completed formal run {run_id} has incomplete one-shot test evidence"
            )
        in_domain = evidence.reports[f"{run_id}-frozen-test"]
        fresh = evidence.reports[f"{run_id}-fresh-process"]
        test_loss = in_domain.metrics.get("test_loss")
        if (
            isinstance(test_loss, bool)
            or not isinstance(test_loss, (int, float))
            or not math.isfinite(float(test_loss))
        ):
            raise GfmTrainingError(f"Formal test loss is invalid for {run_id}")
        summary.update(
            {
                "freshProcessDigest": fresh.verification_digest,
                "testLoss": float(test_loss),
                "testRead": True,
            }
        )
    return summary


def verify_gfm_checkpoint_fresh(
    *,
    root: str | Path | None,
    checkpoint_manifest: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Restore one best checkpoint and compute a fixed validation digest in a new process."""

    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    config = _load_pretrain_config(None, None)
    _, embeddings = _ensure_pretrain_evidence(
        layout, maximum_role="validation", physical_boundary=True
    )
    checkpoint = read_gfm_checkpoint_manifest(checkpoint_manifest)
    payload = load_gfm_checkpoint(checkpoint, map_location=device)
    run_state_path = Path(checkpoint_manifest).resolve().parents[1] / "run-state.json"
    state = read_json_object(run_state_path)
    variant = str(state.get("variant"))
    if variant not in config.architecture.candidates:
        raise ContractViolation("Checkpoint run state has an unknown architecture candidate")
    import torch

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.trainer import CoreTrainer, CoreTrainerConfig

    model = SocialGraphFMCore(_model_config(config, variant))
    model.load_state_dict(payload["components"]["core"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    trainer = CoreTrainer(
        model,
        optimizer,
        CoreTrainerConfig(gradient_accumulation_steps=1, amp=False),
        device,
    )
    streams = _make_domain_streams(layout, embeddings)
    fanout = (
        int(config.architecture.neighbor_fanout[0]),
        int(config.architecture.neighbor_fanout[1]),
    )
    losses: dict[str, float] = {}
    for domain, stream in streams.items():
        validation_start, validation_upper = _stream_role_bounds(stream, 1)
        batch = _core_batch(
            stream,
            batch_size=64,
            fanout=fanout,
            seed=int(state["seed"]),
            cursor=validation_start,
            upper_index=validation_upper,
            advance=False,
            split_role=1,
        )
        losses[domain] = trainer.evaluate_batch(batch)["total"]
    fixed_sample_digest = canonical_sha256(
        {
            "seed": int(state["seed"]),
            "variant": variant,
            "losses": losses,
            "samplePolicy": "first-validation-target-window-batch-role-aware-v2",
        }
    )
    expected_digest = payload.get("best_state", {}).get("expectedFixedSampleDigest")
    if fixed_sample_digest != expected_digest:
        raise GfmTrainingError("Fresh core outputs differ from the digest frozen at training time")
    digest = canonical_sha256(
        {
            "checkpointId": checkpoint.checkpoint_id,
            "stateHash": checkpoint.state_hash,
            "fixedSampleDigest": fixed_sample_digest,
        }
    )
    return {
        "schemaVersion": "gfm.fresh-process-verification/1.0",
        "ok": True,
        "runId": checkpoint.run_id,
        "checkpointId": checkpoint.checkpoint_id,
        "seed": int(state["seed"]),
        "losses": losses,
        "fixedSampleDigest": fixed_sample_digest,
        "verificationDigest": digest,
    }


def evaluate_gfm_checkpoint_test_once(
    *,
    root: str | Path | None,
    checkpoint_manifest: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Open the physical test view exactly once, after best selection."""

    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    checkpoint = read_gfm_checkpoint_manifest(checkpoint_manifest)
    payload = load_gfm_checkpoint(checkpoint, map_location=device)
    run = _registry(layout).get_run(checkpoint.run_id)
    if run is None or run.phase != "pretrain" or run.status != "succeeded":
        raise ContractViolation("One-shot test evaluation requires a succeeded pretrain run")
    run_dir = Path(checkpoint_manifest).resolve().parents[1]
    run_state = read_json_object(run_dir / "run-state.json")
    recorded_best = run_state.get("bestCheckpointManifest")
    if (
        run_state.get("runId") != run.run_id
        or not isinstance(recorded_best, str)
        or Path(recorded_best).resolve() != Path(checkpoint_manifest).resolve()
    ):
        raise ContractViolation("One-shot test checkpoint differs from its terminal run state")
    test_read_path = run_dir / "test-read-state.json"
    if test_read_path.exists():
        raise GfmTrainingError("Formal test view was already attempted for this run")
    atomic_write_json(
        test_read_path,
        {
            "schemaVersion": "gfm.test-read-state/1.0",
            "runId": run.run_id,
            "checkpointId": checkpoint.checkpoint_id,
            "status": "intent-persisted-before-physical-test-view-open",
            "readCountCeiling": 1,
        },
    )
    # Nothing that follows may be retried: the durable intent above is the
    # authority boundary preceding the first possible test-shard open.
    config = _load_pretrain_config(None, None)
    _, embeddings = _ensure_pretrain_evidence(
        layout,
        maximum_role="test",
        physical_boundary=True,
    )
    streams = _make_domain_streams(layout, embeddings, maximum_role="test")
    variant = run.architecture_variant
    import torch

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.trainer import CoreTrainer, CoreTrainerConfig

    model = SocialGraphFMCore(_model_config(config, variant))
    model.load_state_dict(payload["components"]["core"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    trainer = CoreTrainer(
        model,
        optimizer,
        CoreTrainerConfig(gradient_accumulation_steps=1, amp=False),
        device,
    )
    fanout = tuple(int(value) for value in config.architecture.neighbor_fanout)
    losses: dict[str, float] = {}
    for domain, stream in streams.items():
        test_start, test_upper = _stream_role_bounds(stream, 2)
        batch = _core_batch(
            stream,
            batch_size=min(512, test_upper - test_start),
            fanout=fanout,  # type: ignore[arg-type]
            seed=run.seed,
            cursor=test_start,
            upper_index=test_upper,
            advance=False,
            split_role=2,
        )
        losses[domain] = trainer.evaluate_batch(batch)["total"]
    metric = float(np.mean(tuple(losses.values())))
    counters = _temporal_audit_counters(streams=tuple(streams.values()))
    domain_access_audits = {domain: stream.access_audit for domain, stream in streams.items()}
    result_hash = canonical_sha256(
        {
            "checkpointId": checkpoint.checkpoint_id,
            "checkpointStateHash": checkpoint.state_hash,
            "losses": losses,
            "testLoss": metric,
            "counters": counters,
            "domainAccessAudits": domain_access_audits,
            "embeddingAccess": _embedding_artifact_evidence(embeddings),
        }
    )
    atomic_write_json(
        test_read_path,
        {
            "schemaVersion": "gfm.test-read-state/1.0",
            "runId": run.run_id,
            "checkpointId": checkpoint.checkpoint_id,
            "status": "completed",
            "readCount": 1,
            "resultHash": result_hash,
        },
    )
    return {
        "schemaVersion": "gfm.one-shot-test-evaluation/1.0",
        "ok": True,
        "runId": run.run_id,
        "checkpointId": checkpoint.checkpoint_id,
        "seed": run.seed,
        "losses": losses,
        "testLoss": metric,
        "counters": counters,
        "domainAccessAudits": domain_access_audits,
        "embeddingAccess": _embedding_artifact_evidence(embeddings),
        "resultHash": result_hash,
        "testReadCount": 1,
    }


def pretrain_gfm(
    *,
    root: str | Path | None,
    phase: TrainingPhase,
    config: str | Path | None,
    device: str = "cuda",
    overrides: Mapping[str, Any] | None = None,
    variant: Literal["core-base", "core-moe"] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    if phase not in ("dev", "formal"):
        raise ContractViolation("GFM pretraining phase must be dev or formal")
    if phase == "formal" and device != "cuda":
        raise ContractViolation("Formal SocialGraph-FM Core pretraining requires CUDA")
    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    checked_config = _load_pretrain_config(config, overrides)
    corpora, embeddings = _ensure_pretrain_evidence(
        layout,
        formal_required=phase == "formal",
        maximum_role="validation",
        physical_boundary=True,
    )
    protocols = _register_prerequisites(layout, corpora)
    experiment_id = _experiment_id(phase=phase, config=checked_config, corpora=corpora)
    seeds = checked_config.dev.seeds if phase == "dev" else checked_config.formal.seeds
    if variant is not None and variant not in checked_config.architecture.candidates:
        raise ContractViolation("Requested pretrain variant is outside the checked config")
    if seed is not None and seed not in seeds:
        raise ContractViolation("Requested pretrain seed is outside the checked phase config")
    selected_variants = (
        (variant,) if variant is not None else checked_config.architecture.candidates
    )
    selected_seeds = (seed,) if seed is not None else seeds
    runs = []
    reused_count = 0
    for selected_variant in selected_variants:
        for selected_seed in selected_seeds:
            reused = _reuse_pretrain_matrix_cell(
                layout=layout,
                experiment_id=experiment_id,
                phase=phase,
                config=checked_config,
                corpora=corpora,
                protocols=protocols,
                embeddings=embeddings,
                variant=selected_variant,
                seed=selected_seed,
                device=device,
            )
            if reused is not None:
                runs.append(reused)
                reused_count += 1
                continue
            if phase == "formal" and overrides is None:
                worker = _subprocess_json(
                    (
                        "_gfm-pretrain-run",
                        "--phase",
                        phase,
                        "--config",
                        str(config or "socialgraph-core.json"),
                        "--variant",
                        selected_variant,
                        "--seed",
                        str(selected_seed),
                        "--device",
                        device,
                        "--root",
                        str(layout.root),
                        "--json",
                    )
                )
                run = worker["run"]
                if not isinstance(run, dict):
                    raise GfmTrainingError("Independent worker omitted its run summary")
                manifest_path = (
                    layout.gfm_runs
                    / experiment_id
                    / str(run["runId"])
                    / "checkpoints"
                    / f"{run['bestCheckpointId']}.manifest.json"
                )
                test_result = _subprocess_json(
                    (
                        "_gfm-evaluate-test-once",
                        "--checkpoint-manifest",
                        str(manifest_path),
                        "--device",
                        "cpu",
                        "--root",
                        str(layout.root),
                        "--json",
                    )
                )
                test_counters = test_result.get("counters")
                test_losses = test_result.get("losses")
                if (
                    not isinstance(test_counters, dict)
                    or not isinstance(test_losses, dict)
                    or test_result.get("testReadCount") != 1
                ):
                    raise GfmTrainingError("One-shot test evaluator returned invalid evidence")
                test_audit_hash, test_audit_path, checked_test_counters = _leakage_audit(
                    layout,
                    experiment_id=experiment_id,
                    audit_id=f"{run['runId']}-frozen-test",
                    evidence={
                        "checkpointId": run["bestCheckpointId"],
                        "checkpointManifestSha256": file_sha256(manifest_path),
                        "testReadPolicy": "physical-view-once-after-best-registration",
                        "testReadResultHash": test_result["resultHash"],
                        "domainAccessAudits": test_result["domainAccessAudits"],
                        "embeddingAccess": test_result["embeddingAccess"],
                        "sampler": "causal-visible-only-exact-mixed-v1",
                    },
                    counters={str(name): int(value) for name, value in test_counters.items()},
                )
                in_domain_metrics = {
                    "validation_loss": float(run["bestValidationLoss"]),
                    "test_loss": float(test_result["testLoss"]),
                    **checked_test_counters,
                }
                test_evidence_hash, test_evidence_path = _evaluation_evidence(
                    layout,
                    experiment_id=experiment_id,
                    evidence_id=f"{run['runId']}-frozen-test",
                    payload={
                        "checkpointId": run["bestCheckpointId"],
                        "checkpointStateHash": read_gfm_checkpoint_manifest(
                            manifest_path
                        ).state_hash,
                        "testReadCount": 1,
                        "testReadResultHash": test_result["resultHash"],
                        "domainLosses": test_losses,
                        "metrics": in_domain_metrics,
                    },
                )
                registered_run = _registry(layout).get_run(str(run["runId"]))
                if registered_run is None:
                    raise GfmTrainingError("Independent worker did not register its run")
                _registry(layout).record_evaluation(
                    GfmEvaluationReport.create(
                        reportId=f"{run['runId']}-frozen-test",
                        experimentId=experiment_id,
                        runId=run["runId"],
                        checkpointId=run["bestCheckpointId"],
                        evaluationKind="in_domain",
                        domainId="multi-domain",
                        seed=selected_seed,
                        metrics=in_domain_metrics,
                        evidenceArtifactHash=test_evidence_hash,
                        evidenceArtifactPath=test_evidence_path,
                        peakCudaMemoryMiB=registered_run.peak_cuda_memory_mib,
                        leakageAuditPassed=True,
                        leakageAuditHash=test_audit_hash,
                        leakageAuditPath=test_audit_path,
                        warnings=("physical-test-view-read-once-after-best",),
                    )
                )
                verification_args = (
                    "_gfm-verify-checkpoint",
                    "--checkpoint-manifest",
                    str(manifest_path),
                    "--device",
                    "cpu",
                    "--root",
                    str(layout.root),
                    "--json",
                )
                fresh = _subprocess_json(verification_args)
                fresh_repeat = _subprocess_json(verification_args)
                losses = fresh.get("losses")
                if not isinstance(losses, dict) or not losses:
                    raise GfmTrainingError("Fresh-process verification omitted fixed losses")
                if fresh.get("verificationDigest") != fresh_repeat.get(
                    "verificationDigest"
                ) or fresh.get("losses") != fresh_repeat.get("losses"):
                    raise GfmTrainingError(
                        "Repeated fresh-process checkpoint verification was not deterministic"
                    )
                audit_hash, audit_path, audit_counters = _leakage_audit(
                    layout,
                    experiment_id=experiment_id,
                    audit_id=f"{run['runId']}-fresh-process",
                    evidence={
                        "checkpointId": run["bestCheckpointId"],
                        "checkpointManifestSha256": file_sha256(manifest_path),
                        "firstVerificationDigest": fresh["verificationDigest"],
                        "secondVerificationDigest": fresh_repeat["verificationDigest"],
                        "repeatMatch": True,
                    },
                    counters=_temporal_audit_counters(
                        streams=tuple(_make_domain_streams(layout, embeddings).values())
                    ),
                )
                fresh_metrics = {
                    **{str(key): float(value) for key, value in losses.items()},
                    **audit_counters,
                    "fresh_process_repeat_match": 1.0,
                }
                evidence_hash, evidence_path = _evaluation_evidence(
                    layout,
                    experiment_id=experiment_id,
                    evidence_id=f"{run['runId']}-fresh-process",
                    payload={
                        "checkpointId": run["bestCheckpointId"],
                        "verificationDigest": fresh["verificationDigest"],
                        "repeatVerificationDigest": fresh_repeat["verificationDigest"],
                        "losses": losses,
                        "metrics": fresh_metrics,
                    },
                )
                _registry(layout).record_evaluation(
                    GfmEvaluationReport.create(
                        reportId=f"{run['runId']}-fresh-process",
                        experimentId=experiment_id,
                        runId=run["runId"],
                        checkpointId=run["bestCheckpointId"],
                        evaluationKind="fresh_process",
                        domainId="multi-domain",
                        seed=selected_seed,
                        metrics=fresh_metrics,
                        evidenceArtifactHash=evidence_hash,
                        evidenceArtifactPath=evidence_path,
                        peakCudaMemoryMiB=registered_run.peak_cuda_memory_mib,
                        leakageAuditPassed=True,
                        leakageAuditHash=audit_hash,
                        leakageAuditPath=audit_path,
                        freshProcessVerified=True,
                        verificationDigest=fresh["verificationDigest"],
                    )
                )
                run["freshProcessDigest"] = fresh["verificationDigest"]
                run["testLoss"] = float(test_result["testLoss"])
                run["testRead"] = True
                runs.append(run)
            else:
                runs.append(
                    _train_run(
                        layout=layout,
                        experiment_id=experiment_id,
                        config=checked_config,
                        corpora=corpora,
                        protocols=protocols,
                        embeddings=embeddings,
                        phase=phase,
                        variant=selected_variant,
                        seed=selected_seed,
                        device=device,
                    )
                )
    return {
        "schemaVersion": "gfm.workflow-pretrain/1.0",
        "ok": True,
        "experimentId": experiment_id,
        "phase": phase,
        "runKind": checked_config.run_kind,
        "configHash": checked_config.config_hash,
        "runs": runs,
        "testRead": phase == "formal",
        "independentProcesses": phase == "formal" and overrides is None,
        "selection": {
            "variants": list(selected_variants),
            "seeds": list(selected_seeds),
        },
        "reusedRunCount": reused_count,
    }


def _require_experiment_runs(
    layout: RuntimeLayout, experiment_id: str
) -> tuple[GfmRunManifest, ...]:
    registry = _registry(layout)
    runs = registry.list_runs(experiment_id=experiment_id)
    if not runs:
        raise ContractViolation(f"No registered GFM runs exist for {experiment_id}")
    return runs


__all__ = [
    "evaluate_gfm_checkpoint_test_once",
    "pretrain_gfm",
    "verify_gfm_checkpoint_fresh",
]
