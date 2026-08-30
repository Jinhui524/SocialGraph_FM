"""Leave-one-domain-out training, selection, scoring, and worker execution.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def _validate_lodo_formal_prerequisites(
    *,
    layout: RuntimeLayout,
    experiment_id: str,
    config: GfmPretrainConfig,
    corpora: Sequence[GfmDomainCorpusManifest],
    protocols: Sequence[GfmTaskProtocolManifest],
    existing_pretrain_runs: Sequence[GfmRunManifest],
    code_hash: str,
    environment_hash: str,
) -> None:
    """Require the exact deeply revalidated six-cell formal matrix before LODO."""

    expected_experiment = _experiment_id(phase="formal", config=config, corpora=corpora)
    expected_keys = {
        (candidate, formal_seed)
        for candidate in config.architecture.candidates
        for formal_seed in config.formal.seeds
    }
    actual_keys = {(run.architecture_variant, run.seed) for run in existing_pretrain_runs}
    if (
        experiment_id != expected_experiment
        or actual_keys != expected_keys
        or len(existing_pretrain_runs) != len(expected_keys)
    ):
        raise ContractViolation("LODO requires the exact fixed six-cell formal pretraining matrix")
    corpus_by_domain = {corpus.domain_id: corpus for corpus in corpora}
    full_domain_ids = tuple(DOMAIN_IDS.values())
    if set(corpus_by_domain) != set(full_domain_ids):
        raise ContractViolation("LODO formal prerequisites lack one of the three domains")
    full_corpus_hashes = tuple(corpus_by_domain[domain].logical_hash for domain in full_domain_ids)
    protocol_hashes = tuple(protocol.protocol_hash for protocol in protocols)
    for formal_variant, formal_seed in sorted(expected_keys):
        formal_run_id = f"{experiment_id}-{formal_variant}-{formal_seed}"
        completed = _validate_completed_matrix_run(
            layout,
            _CompletedRunExpectation(
                experiment_id=experiment_id,
                run_id=formal_run_id,
                phase="pretrain",
                variant=formal_variant,
                seed=formal_seed,
                domain_ids=full_domain_ids,
                corpus_hashes=full_corpus_hashes,
                protocol_hashes=protocol_hashes,
                config_hash=config.config_hash,
                code_hash=code_hash,
                environment_hash=environment_hash,
                pretrain_phase="formal",
                required_reports=_required_pretrain_reports(formal_run_id, "formal"),
            ),
        )
        if completed is None:
            raise ContractViolation(
                "LODO formal prerequisite cell is absent after matrix verification"
            )


def _lodo_source_should_stop(
    *, optimizer_step: int, minimum_steps: int, no_improvement: int, patience: int
) -> bool:
    """Pure resume-safe early-stop boundary used before every source block."""

    return optimizer_step >= minimum_steps and no_improvement >= patience


def _train_lodo_optimizer_block(
    trainer: Any,
    lr_scheduler: Any,
    domain_loaders: Mapping[str, Iterable[Any]],
    *,
    maximum_steps: int,
    heartbeat: Callable[[Mapping[str, float]], None] | None = None,
    heartbeat_every_steps: int = HEARTBEAT_EVERY_OPTIMIZER_STEPS,
) -> Any:
    """Commit a bounded LODO block with optimizer/LR steps in lockstep.

    LODO constructs one microbatch per source domain.  The final block may be
    shorter than that domain count; select only the next round-robin prefix so
    a non-multiple ``maximum_steps`` can never overshoot its contract.
    """

    current = int(trainer.optimizer_step)
    if maximum_steps < 1 or not 0 <= current < maximum_steps:
        raise GfmTrainingError("LODO source optimizer block has invalid step bounds")
    if int(lr_scheduler.last_epoch) != current:
        raise GfmTrainingError("LODO learning-rate scheduler differs from optimizer progress")
    scheduler_domains = tuple(trainer.scheduler.domains)
    unknown = set(domain_loaders).difference(scheduler_domains)
    if not domain_loaders or unknown:
        raise GfmTrainingError("LODO source block requires one batch per known domain")
    cursor = int(trainer.scheduler.cursor)
    ordered = tuple(
        domain
        for offset in range(len(scheduler_domains))
        if (domain := scheduler_domains[(cursor + offset) % len(scheduler_domains)])
        in domain_loaders
    )
    take = min(maximum_steps - current, len(ordered))
    selected = {domain: domain_loaders[domain] for domain in ordered[:take]}
    if not selected:
        raise GfmTrainingError("LODO source optimizer block selected no domain")
    result = _train_epoch_with_heartbeats(
        trainer,
        selected,
        every_optimizer_steps=(
            heartbeat_every_steps if heartbeat is not None else maximum_steps + 1
        ),
        heartbeat=(heartbeat or (lambda _losses: None)),
        after_optimizer_step=lr_scheduler.step,
    )
    completed = int(trainer.optimizer_step) - current
    if (
        completed != take
        or int(result.optimizer_steps) != take
        or int(trainer.optimizer_step) > maximum_steps
        or int(lr_scheduler.last_epoch) != int(trainer.optimizer_step)
    ):
        raise GfmTrainingError("LODO source optimizer step accounting failed")
    return result


@dataclass
class _LodoStageResult:
    model_state: dict[str, Any]
    trainer: Any
    scheduler: Any
    streams: Mapping[str, _DomainStream]
    best_model_state: dict[str, Any] | None
    best_validation_loss: float | None
    no_improvement_evaluations: int
    last_losses: dict[str, float]
    peak_cuda_memory_mib: float
    score: float | None = None
    selected_event_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class _LodoFewShotSelection:
    ordinals: np.ndarray
    event_indices: tuple[int, ...]
    fraction: float
    full_train_event_count: int
    eligible_pool_count: int
    eligible_pool_hash: str

    @property
    def evidence(self) -> dict[str, Any]:
        values = list(self.event_indices)
        return {
            "eventIndices": values,
            "eventIndicesHash": canonical_sha256(values),
            "fraction": self.fraction,
            "fullTrainEventCount": self.full_train_event_count,
            "eligiblePoolCount": self.eligible_pool_count,
            "eligiblePoolHash": self.eligible_pool_hash,
        }


def _lodo_trainer_resume_state(trainer: Any) -> dict[str, Any]:
    return {
        "schemaVersion": "gfm.lodo-trainer-state/1.0",
        "globalStep": int(trainer.global_step),
        "optimizerStep": int(trainer.optimizer_step),
        "roundRobinState": trainer.scheduler.state_dict(),
    }


def _restore_lodo_trainer(
    *,
    trainer: Any,
    scheduler: Any,
    streams: Mapping[str, _DomainStream],
    payload: Mapping[str, Any],
    stage: str,
) -> tuple[dict[str, Any] | None, float | None, int, float]:
    components = payload.get("components")
    sampler = payload.get("sampler_state")
    checkpoint_best = payload.get("best_state")
    if (
        not isinstance(components, dict)
        or "current_core" not in components
        or not isinstance(sampler, dict)
        or sampler.get("stage") != stage
        or not isinstance(checkpoint_best, dict)
    ):
        raise ContractViolation("LODO progress checkpoint has the wrong active stage")
    trainer_state = sampler.get("trainerState")
    stream_states = sampler.get("streamStates")
    if (
        not isinstance(trainer_state, dict)
        or trainer_state.get("schemaVersion") != "gfm.lodo-trainer-state/1.0"
        or not isinstance(stream_states, dict)
        or set(stream_states) != set(streams)
    ):
        raise ContractViolation("LODO progress lacks trainer or domain cursor state")
    trainer.model.load_state_dict(components["current_core"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    trainer.scaler.load_state_dict(payload["scaler_state"] or {})
    trainer.scheduler.load_state_dict(dict(trainer_state["roundRobinState"]))
    trainer.global_step = int(trainer_state["globalStep"])
    trainer.optimizer_step = int(trainer_state["optimizerStep"])
    scheduler_state = payload.get("scheduler_state")
    if not isinstance(scheduler_state, dict):
        raise ContractViolation("LODO progress lacks learning-rate scheduler state")
    scheduler.load_state_dict(scheduler_state)
    if int(scheduler.last_epoch) != int(trainer.optimizer_step):
        raise ContractViolation("LODO resumed LR scheduler differs from optimizer progress")
    for domain, stream in streams.items():
        stream.load_state_dict(stream_states[domain])
    restore_rng_state(dict(payload["rng_state"]))
    best_available = bool(checkpoint_best.get("bestAvailable", False))
    best_model = components.get("best_core") if best_available else None
    validation_loss = checkpoint_best.get("validationLoss")
    if best_available and (
        not isinstance(best_model, dict)
        or isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(float(validation_loss))
    ):
        raise ContractViolation("LODO progress has invalid selected validation state")
    no_improvement = checkpoint_best.get("noImprovementEvaluations", 0)
    prior_peak = checkpoint_best.get("peakCudaMemoryMiB", 0.0)
    if (
        isinstance(no_improvement, bool)
        or not isinstance(no_improvement, int)
        or no_improvement < 0
        or isinstance(prior_peak, bool)
        or not isinstance(prior_peak, (int, float))
        or not math.isfinite(float(prior_peak))
        or float(prior_peak) < 0.0
    ):
        raise ContractViolation("LODO progress has invalid selection counters")
    return (
        None if best_model is None else dict(best_model),
        None if validation_loss is None else float(validation_loss),
        int(no_improvement),
        float(prior_peak),
    )


def _train_lodo_source(
    *,
    config: GfmPretrainConfig,
    variant: str,
    streams: Mapping[str, _DomainStream],
    seed: int,
    device: str,
    maximum_steps: int,
    stage: str = "source:multi",
    resume_payload: Mapping[str, Any] | None = None,
    progress: Callable[[_LodoStageResult], None] | None = None,
    heartbeat: Callable[[_LodoStageResult], None] | None = None,
) -> _LodoStageResult:
    import torch

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.trainer import CoreTrainer, CoreTrainerConfig

    if not streams or maximum_steps < 1:
        raise ValueError("LODO source pretraining requires streams and positive steps")
    if resume_payload is None:
        set_seed(seed, device)
    model = SocialGraphFMCore(_model_config(config, variant, domains=tuple(streams)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    trainer = CoreTrainer(
        model,
        optimizer,
        CoreTrainerConfig(
            gradient_accumulation_steps=1,
            gradient_clip=config.optimization.gradient_clip,
            amp=True,
        ),
        device,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_cosine(
            step,
            maximum=maximum_steps,
            warmup_ratio=config.optimization.warmup_ratio,
        ),
    )
    fanout = tuple(int(value) for value in config.architecture.neighbor_fanout)
    formal = config.formal
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    prior_peak = 0.0
    last_losses: dict[str, float] = {"total": float("inf")}
    if resume_payload is not None:
        best_state, resumed_best_loss, no_improvement, prior_peak = _restore_lodo_trainer(
            trainer=trainer,
            scheduler=scheduler,
            streams=streams,
            payload=resume_payload,
            stage=stage,
        )
        best_loss = math.inf if resumed_best_loss is None else resumed_best_loss
        if resumed_best_loss is not None:
            last_losses = {"total": resumed_best_loss}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    if int(scheduler.last_epoch) != int(trainer.optimizer_step):
        raise GfmTrainingError("LODO learning-rate scheduler differs from optimizer progress")

    def early_stop_reached() -> bool:
        return _lodo_source_should_stop(
            optimizer_step=int(trainer.optimizer_step),
            minimum_steps=formal.minimum_steps,
            no_improvement=no_improvement,
            patience=formal.patience_evaluations,
        )

    while trainer.optimizer_step < maximum_steps and not early_stop_reached():
        scheduler_domains = tuple(trainer.scheduler.domains)
        cursor = int(trainer.scheduler.cursor)
        remaining = maximum_steps - int(trainer.optimizer_step)
        selected_domains = tuple(
            scheduler_domains[(cursor + offset) % len(scheduler_domains)]
            for offset in range(min(remaining, len(scheduler_domains)))
        )

        def one_batch(domain: str) -> Iterator[Any]:
            stream = streams[domain]
            train_start, train_count = _stream_role_bounds(stream, 0)
            if stream.cursor < train_start or stream.cursor >= train_count:
                stream.cursor = train_start
                stream.epoch += 1
            yield _core_batch(
                stream,
                batch_size=min(64, train_count - stream.cursor),
                fanout=fanout,  # type: ignore[arg-type]
                seed=seed + int(trainer.global_step),
                advance=True,
                split_role=0,
            )

        loaders: dict[str, Iterable[Any]] = {
            domain: one_batch(domain) for domain in selected_domains
        }

        def observe_losses(losses: Mapping[str, float]) -> None:
            nonlocal last_losses
            last_losses = {str(name): float(value) for name, value in losses.items()}
            if heartbeat is not None:
                current_peak = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
                )
                heartbeat(
                    _LodoStageResult(
                        model_state=dict(model.state_dict()),
                        trainer=trainer,
                        scheduler=scheduler,
                        streams=streams,
                        best_model_state=best_state,
                        best_validation_loss=(None if math.isinf(best_loss) else best_loss),
                        no_improvement_evaluations=no_improvement,
                        last_losses=last_losses,
                        peak_cuda_memory_mib=max(prior_peak, float(current_peak)),
                    )
                )

        block = _train_lodo_optimizer_block(
            trainer,
            scheduler,
            loaders,
            maximum_steps=maximum_steps,
            heartbeat=observe_losses,
        )
        last_losses = {str(name): float(value) for name, value in block.mean_losses.items()}
        if (
            trainer.optimizer_step % formal.evaluation_every_steps == 0
            or trainer.optimizer_step >= maximum_steps
        ):
            validation_values: list[float] = []
            for domain, stream in streams.items():
                validation_start, validation_count = _stream_role_bounds(stream, 1)
                validation = _core_batch(
                    stream,
                    batch_size=min(512, validation_count - validation_start),
                    fanout=fanout,  # type: ignore[arg-type]
                    seed=seed,
                    cursor=validation_start,
                    upper_index=validation_count,
                    advance=False,
                    split_role=1,
                )
                validation_values.append(trainer.evaluate_batch(validation)["total"])
            value = float(np.mean(validation_values))
            if value < best_loss:
                best_loss = value
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                no_improvement = 0
            else:
                no_improvement += 1
            if progress is not None:
                current_peak = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
                )
                progress(
                    _LodoStageResult(
                        model_state=dict(model.state_dict()),
                        trainer=trainer,
                        scheduler=scheduler,
                        streams=streams,
                        best_model_state=best_state,
                        best_validation_loss=best_loss,
                        no_improvement_evaluations=no_improvement,
                        last_losses=last_losses,
                        peak_cuda_memory_mib=max(prior_peak, float(current_peak)),
                    )
                )
            if (
                trainer.optimizer_step >= formal.minimum_steps
                and no_improvement >= formal.patience_evaluations
            ):
                break
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    peak = max(prior_peak, float(peak))
    if best_state is None:
        raise GfmTrainingError("LODO source pretraining produced no selected state")
    return _LodoStageResult(
        model_state=best_state,
        trainer=trainer,
        scheduler=scheduler,
        streams=streams,
        best_model_state=best_state,
        best_validation_loss=best_loss,
        no_improvement_evaluations=no_improvement,
        last_losses=last_losses,
        peak_cuda_memory_mib=float(peak),
    )


def _array_content_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_sha256(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _lodo_eligibility_identity(
    stream: _DomainStream, *, fanout: tuple[int, int], negatives_per_positive: int
) -> dict[str, Any]:
    train_events = _stream_role_indices(stream, 0)
    return {
        "schemaVersion": "gfm.lodo-eligibility-identity/1.0",
        "domainId": stream.domain_id,
        "preparedCorpusContentHash": str(stream.manifest["logicalHash"]),
        "maximumRole": stream.maximum_access_role,
        "accessAuditHash": canonical_sha256(stream.access_audit or {}),
        "eventArrayHashes": {
            "src": _array_content_hash(stream.src),
            "dst": _array_content_hash(stream.dst),
            "timestamp": _array_content_hash(stream.timestamp),
            "relation": _array_content_hash(stream.relation),
            "trainEvents": _array_content_hash(train_events),
            "eventSplit": (
                None if stream.event_split is None else _array_content_hash(stream.event_split)
            ),
        },
        "nodeTypeHash": _array_content_hash(stream.node_type),
        "trainEndInclusive": stream.train_end,
        "fanout": list(fanout),
        "negativesPerPositive": negatives_per_positive,
        "samplerPolicy": "causal-mixed-exact-typed-with-audited-uniform-fallback",
        "algorithm": "bounded-two-layer-local-visible-non-neighbor-v1",
        "codeHash": code_identity_hash(),
    }


def _build_lodo_eligible_ordinals(
    stream: _DomainStream,
    *,
    fanout: tuple[int, int],
    negatives_per_positive: int,
) -> np.ndarray:
    """One causal pass using the exact bounded local candidate universe.

    ``_core_batch`` exposes to the negative sampler only nodes occurring in
    the fixed two-layer message graph.  With batch fallback enabled, four
    distinct same-type, cutoff-visible non-neighbours in that graph are both
    necessary and sufficient for its four requested negatives.  The dynamic
    pair set supplies complete cutoff adjacency, including old edges omitted
    by fanout, without scanning a global type class for every event.
    """

    train_start, train_count = _stream_role_bounds(stream, 0)
    train_events = _stream_role_indices(stream, 0)
    node_seen = np.zeros(stream.node_count, dtype=np.bool_)
    visible_type_counts: defaultdict[int, int] = defaultdict(int)
    visible_pairs: set[int] = set()
    eligible_values: list[int] = []
    for ordinal in range(train_count):
        event_index = int(train_events[ordinal])
        source = int(stream.src[event_index])
        target = int(stream.dst[event_index])
        target_type = int(stream.node_type[target])
        if (
            ordinal >= train_start
            and (node_seen[source] or node_seen[target])
            and visible_type_counts[target_type] >= negatives_per_positive
        ):
            message_indices = _recent_causal_edges(
                stream,
                end=event_index,
                seeds={source, target},
                fanout=fanout,
                maximum_split_role=0 if stream.event_split is not None else None,
            )
            if message_indices.size:
                message_nodes = np.unique(
                    np.concatenate((stream.src[message_indices], stream.dst[message_indices]))
                )
                compatible = 0
                for raw_node in message_nodes:
                    node = int(raw_node)
                    if node not in (source, target) and int(stream.node_type[node]) == target_type:
                        left, right = (source, node) if source < node else (node, source)
                        if left * stream.node_count + right not in visible_pairs:
                            compatible += 1
                            if compatible == negatives_per_positive:
                                eligible_values.append(ordinal)
                                break
        # Ordinal zero is intentionally advanced before ordinal one is judged;
        # this matches `_core_batch(cursor=1)` and fixes the prefix boundary.
        left, right = (source, target) if source < target else (target, source)
        if left != right:
            visible_pairs.add(left * stream.node_count + right)
        for node in (source, target):
            if not node_seen[node]:
                node_seen[node] = True
                visible_type_counts[int(stream.node_type[node])] += 1
    result = np.ascontiguousarray(eligible_values, dtype=np.int64)
    result.setflags(write=False)
    return result


def _lodo_cached_eligible_ordinals(
    stream: _DomainStream,
    *,
    fanout: tuple[int, int],
    negatives_per_positive: int,
) -> np.ndarray:
    key = (fanout, negatives_per_positive)
    cached = stream.lodo_eligible_pool_cache.get(key)
    if cached is not None:
        return cached
    identity = _lodo_eligibility_identity(
        stream, fanout=fanout, negatives_per_positive=negatives_per_positive
    )
    identity_hash = canonical_sha256(identity)
    directory = stream.lodo_eligibility_cache_path
    if directory is None:
        result = _build_lodo_eligible_ordinals(
            stream, fanout=fanout, negatives_per_positive=negatives_per_positive
        )
        stream.lodo_eligible_pool_cache[key] = result
        return result
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{identity_hash}.npz"
    manifest_path = directory / f"{identity_hash}.json"

    def load_published() -> np.ndarray | None:
        if not artifact.exists() and not manifest_path.exists():
            return None
        if not artifact.is_file() or artifact.is_symlink() or not manifest_path.is_file():
            raise ContractViolation("LODO eligibility cache is incomplete or unsafe")
        manifest = read_json_object(manifest_path)
        if (
            manifest.get("schemaVersion") != "gfm.lodo-eligibility-cache/1.0"
            or manifest.get("identity") != identity
            or manifest.get("identityHash") != identity_hash
            or manifest.get("artifactName") != artifact.name
            or manifest.get("artifactSha256") != file_sha256(artifact)
        ):
            raise ContractViolation("LODO eligibility cache provenance or hash differs")
        arrays = load_npz_safe(
            artifact,
            expected={"eligible_ordinals": (np.dtype(np.int64).str, 1)},
        )
        values = arrays["eligible_ordinals"]
        train_start, train_count = _stream_role_bounds(stream, 0)
        train_events = _stream_role_indices(stream, 0)
        if (
            values.ndim != 1
            or values.size < 1
            or bool(np.any(values < train_start))
            or bool(np.any(values >= train_count))
            or bool(np.any(values[1:] <= values[:-1]))
            or manifest.get("eligiblePoolCount") != int(values.size)
            or manifest.get("eligiblePoolHash") != canonical_sha256(train_events[values].tolist())
        ):
            raise ContractViolation("LODO eligibility cache rows are invalid")
        values.setflags(write=False)
        return values

    with exclusive_file_lock(directory / f"{identity_hash}.lock"):
        try:
            published = load_published()
        except ContractViolation as error:
            # This directory contains derived, identity-named cache files only.
            # Under the exclusive OS lock, recover only an interrupted one-sided
            # publication; a complete but corrupted pair remains fail-closed.
            if artifact.exists() != manifest_path.exists():
                artifact.unlink(missing_ok=True)
                manifest_path.unlink(missing_ok=True)
                published = None
            else:
                raise error
        if published is None:
            published = _build_lodo_eligible_ordinals(
                stream, fanout=fanout, negatives_per_positive=negatives_per_positive
            )
            atomic_write_npz(artifact, {"eligible_ordinals": published})
            train_events = _stream_role_indices(stream, 0)
            atomic_write_json(
                manifest_path,
                {
                    "schemaVersion": "gfm.lodo-eligibility-cache/1.0",
                    "identity": identity,
                    "identityHash": identity_hash,
                    "artifactName": artifact.name,
                    "artifactSha256": file_sha256(artifact),
                    "eligiblePoolCount": int(published.size),
                    "eligiblePoolHash": canonical_sha256(train_events[published].tolist()),
                },
            )
            published = load_published()
            assert published is not None
    stream.lodo_eligible_pool_cache[key] = published
    return published


def _lodo_few_shot_selection(
    stream: _DomainStream,
    *,
    seed: int,
    fraction: float,
) -> _LodoFewShotSelection:
    """Select an exact fraction of the full causally-trainable target pool."""

    if fraction not in (0.01, 0.05, 0.1):
        raise ValueError("Formal LODO fraction must be 1%, 5% or 10%")
    train_start, train_count = _stream_role_bounds(stream, 0)
    train_events = _stream_role_indices(stream, 0)
    eligible_ordinals = _lodo_cached_eligible_ordinals(
        stream, fanout=(15, 10), negatives_per_positive=4
    )
    if eligible_ordinals.size < 100:
        raise GfmTrainingError("LODO target has fewer than 100 causally trainable temporal events")
    eligible_events = train_events[eligible_ordinals]
    desired = max(1, int(math.floor(eligible_events.size * fraction)))
    labels = stream.relation[eligible_events]
    rng = np.random.default_rng(seed + int(fraction * 10_000))
    initially_selected: list[int] = []
    unique_labels = np.unique(labels)
    if desired >= unique_labels.size:
        for label in unique_labels:
            candidates = np.flatnonzero(labels == label)
            initially_selected.append(int(candidates[int(rng.integers(0, candidates.size))]))
    remaining = desired - len(initially_selected)
    available = np.setdiff1d(
        np.arange(eligible_events.size, dtype=np.int64),
        np.asarray(initially_selected, dtype=np.int64),
        assume_unique=False,
    )
    if remaining > 0:
        initially_selected.extend(
            int(value) for value in rng.choice(available, size=remaining, replace=False).tolist()
        )
    selected_positions = np.asarray(sorted(initially_selected), dtype=np.int64)
    chosen_ordinals = np.ascontiguousarray(eligible_ordinals[selected_positions], dtype=np.int64)
    chosen_events = tuple(int(value) for value in eligible_events[selected_positions].tolist())
    return _LodoFewShotSelection(
        ordinals=chosen_ordinals,
        event_indices=chosen_events,
        fraction=fraction,
        full_train_event_count=train_count - train_start,
        eligible_pool_count=int(eligible_events.size),
        eligible_pool_hash=canonical_sha256(eligible_events.tolist()),
    )


def _target_few_shot_score(
    *,
    config: GfmPretrainConfig,
    variant: str,
    stream: _DomainStream,
    seed: int,
    fraction: float,
    device: str,
    source_state: Mapping[str, Any] | None,
    maximum_steps: int,
    stage: str,
    chosen_ordinals: np.ndarray,
    chosen_events: tuple[int, ...],
    resume_payload: Mapping[str, Any] | None = None,
    progress: Callable[[_LodoStageResult], None] | None = None,
    heartbeat: Callable[[_LodoStageResult], None] | None = None,
) -> _LodoStageResult:
    import torch

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.trainer import CoreTrainer, CoreTrainerConfig
    from ..gfm.transfer_workflow import load_lodo_shared_backbone

    _, train_count = _stream_role_bounds(stream, 0)
    expected = _lodo_few_shot_selection(stream, seed=seed, fraction=fraction)
    if (
        not np.array_equal(chosen_ordinals, expected.ordinals)
        or chosen_events != expected.event_indices
    ):
        raise ContractViolation("LODO target selection changed after its durable binding")
    # Target adapter and every target-specific parameter are instantiated only
    # now, after source pretraining has completed.
    if resume_payload is None:
        set_seed(seed + int(fraction * 100_000), device)
    model = SocialGraphFMCore(_model_config(config, variant, domains=(stream.domain_id,)))
    if source_state is not None:
        load_lodo_shared_backbone(model, source_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    trainer = CoreTrainer(
        model,
        optimizer,
        CoreTrainerConfig(
            gradient_accumulation_steps=1,
            gradient_clip=config.optimization.gradient_clip,
            amp=True,
        ),
        device,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_cosine(
            step,
            maximum=maximum_steps,
            warmup_ratio=config.optimization.warmup_ratio,
        ),
    )
    fanout = tuple(int(value) for value in config.architecture.neighbor_fanout)
    prior_peak = 0.0
    last_losses: dict[str, float] = {"total": 0.0}
    if resume_payload is not None:
        _, _, _, prior_peak = _restore_lodo_trainer(
            trainer=trainer,
            scheduler=scheduler,
            streams={stream.domain_id: stream},
            payload=resume_payload,
            stage=stage,
        )
    if int(scheduler.last_epoch) != int(trainer.optimizer_step):
        raise GfmTrainingError("LODO target LR scheduler differs from optimizer progress")
    schedule_cycle = -1
    schedule = np.empty(0, dtype=np.int64)
    while trainer.optimizer_step < maximum_steps:
        cursor = int(trainer.optimizer_step)
        cycle, position = divmod(cursor, chosen_ordinals.size)
        if cycle != schedule_cycle:
            cycle_rng = np.random.default_rng(
                seed + int(fraction * 100_000) + 71 + cycle * 1_000_003
            )
            schedule = cycle_rng.permutation(chosen_ordinals.size)
            schedule_cycle = cycle
        index = int(chosen_ordinals[int(schedule[position])])
        batch = _core_batch(
            stream,
            batch_size=1,
            fanout=fanout,  # type: ignore[arg-type]
            seed=seed + cursor,
            cursor=index,
            upper_index=train_count,
            advance=False,
            split_role=0,
        )

        def observe_losses(losses: Mapping[str, float]) -> None:
            nonlocal last_losses
            last_losses = {str(name): float(value) for name, value in losses.items()}
            if heartbeat is not None:
                current_peak = (
                    torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
                )
                heartbeat(
                    _LodoStageResult(
                        model_state=dict(model.state_dict()),
                        trainer=trainer,
                        scheduler=scheduler,
                        streams={stream.domain_id: stream},
                        best_model_state=None,
                        best_validation_loss=None,
                        no_improvement_evaluations=0,
                        last_losses=last_losses,
                        peak_cuda_memory_mib=max(prior_peak, float(current_peak)),
                        selected_event_indices=chosen_events,
                    )
                )

        result = _train_epoch_with_heartbeats(
            trainer,
            {stream.domain_id: [batch]},
            every_optimizer_steps=HEARTBEAT_EVERY_OPTIMIZER_STEPS,
            heartbeat=observe_losses,
            after_optimizer_step=scheduler.step,
        )
        last_losses = {str(name): float(value) for name, value in result.mean_losses.items()}
    validation_start, validation_upper = _stream_role_bounds(stream, 1)
    validation = _core_batch(
        stream,
        batch_size=min(512, validation_upper - validation_start),
        fanout=fanout,  # type: ignore[arg-type]
        seed=seed,
        cursor=validation_start,
        upper_index=validation_upper,
        advance=False,
        split_role=1,
        negatives_per_positive=99,
    )
    from ..gfm.evaluation import ranking_metrics

    model.to(device).eval()
    with torch.inference_mode():
        local_validation = validation.to(device)
        output = model(local_validation)
        if (
            local_validation.positive_edge_index is None
            or local_validation.negative_edge_index is None
        ):
            raise GfmTrainingError("LODO validation lacks fixed ranking candidates")
        positive_scores = model.score_links(
            output.node_embeddings, local_validation.positive_edge_index
        ).float()
        negative_scores = model.score_links(
            output.node_embeddings, local_validation.negative_edge_index
        ).float()
        if negative_scores.numel() % positive_scores.numel():
            raise GfmTrainingError("LODO validation negatives are not query aligned")
        negative_scores = negative_scores.reshape(positive_scores.numel(), -1)
        ranking = ranking_metrics(positive_scores, negative_scores, ks=(1, 10))
    if not math.isfinite(ranking.mrr):
        raise GfmTrainingError("LODO target validation MRR is non-finite")
    peak = torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
    return _LodoStageResult(
        model_state=dict(model.state_dict()),
        trainer=trainer,
        scheduler=scheduler,
        streams={stream.domain_id: stream},
        best_model_state=dict(model.state_dict()),
        best_validation_loss=None,
        no_improvement_evaluations=0,
        last_losses=last_losses,
        peak_cuda_memory_mib=max(prior_peak, float(peak)),
        score=ranking.mrr,
        selected_event_indices=chosen_events,
    )


def _lodo_worker(
    *,
    root: str | Path,
    experiment_id: str,
    held_out_domain: str,
    variant: Literal["core-base", "core-moe"],
    seed: int,
    device: str,
    source_steps: int | None = None,
    target_steps: int | None = None,
    resume_requested: bool = False,
    _lock_owned: bool = False,
) -> dict[str, Any]:
    """Execute or explicitly resume one leakage-isolated durable LODO cell."""

    # The lock wrapper calls the implementation again with its private marker;
    # this keeps the large implementation flat while holding cell ownership
    # across prerequisite checks, training, terminal registry publication and
    # run-state finalisation.
    if not _lock_owned:
        prospective_run_id = f"{experiment_id}-lodo-{variant}-{held_out_domain}-{seed}"
        prospective_run_dir = (
            prepare_runtime_layout(root, operation="run").gfm_runs
            / experiment_id
            / prospective_run_id
        )
        with exclusive_lodo_execution_lock(prospective_run_dir):
            return _lodo_worker(
                root=root,
                experiment_id=experiment_id,
                held_out_domain=held_out_domain,
                variant=variant,
                seed=seed,
                device=device,
                source_steps=source_steps,
                target_steps=target_steps,
                resume_requested=resume_requested,
                _lock_owned=True,
            )

    from ..gfm.transfer_workflow import LodoIsolationAudit, assert_lodo_isolation

    if held_out_domain not in DOMAIN_IDS.values():
        raise ContractViolation("LODO held-out domain is not one of the three families")
    if device != "cuda":
        raise ContractViolation("Formal LODO runs require CUDA")
    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    config = _load_pretrain_config(None, None)
    fixed_source_steps = config.formal.max_steps
    fixed_target_steps = config.transfer.lodo_target_adaptation_steps
    if source_steps is not None and source_steps != fixed_source_steps:
        raise ContractViolation("Formal LODO source pretraining must use config.formal.maxSteps")
    if target_steps is not None and target_steps != fixed_target_steps:
        raise ContractViolation(
            "Formal LODO target adaptation must use the fixed 5000-step protocol"
        )
    source_steps, target_steps = fixed_source_steps, fixed_target_steps
    if seed not in config.formal.seeds or variant not in config.architecture.candidates:
        raise ContractViolation("LODO seed or architecture is outside the fixed config")
    source_domain_ids = tuple(domain for domain in DOMAIN_IDS.values() if domain != held_out_domain)
    source_domain_ids = tuple(sorted(source_domain_ids))
    # Open only source embedding views during source pretraining.  Corpus
    # contracts are verified globally, but the held-out event and embedding
    # arrays remain physically unopened until all three source stages have a
    # durable frozen output.
    corpora, source_embeddings = _ensure_pretrain_evidence(
        layout,
        embedding_domain_ids=source_domain_ids,
        corpus_verification_domain_ids=source_domain_ids,
        maximum_role="validation",
        physical_boundary=True,
    )
    existing_formal = [
        run
        for run in _registry(layout).list_runs(experiment_id=experiment_id)
        if run.phase == "pretrain" and run.status == "succeeded"
    ]
    expected_corpus_hashes = {corpus.logical_hash for corpus in corpora}
    current_code_hash = code_identity_hash()
    current_environment_hash = _environment_hash(device)
    if not existing_formal or any(
        run.config_hash != config.config_hash
        or run.code_hash != current_code_hash
        or run.environment_hash != current_environment_hash
        or set(run.corpus_hashes) != expected_corpus_hashes
        for run in existing_formal
    ):
        raise ContractViolation(
            "LODO requires a completed three-domain formal experiment with identical corpora"
        )
    protocols = _register_prerequisites(layout, corpora)
    corpus_by_domain = {corpus.domain_id: corpus for corpus in corpora}
    _validate_lodo_formal_prerequisites(
        layout=layout,
        experiment_id=experiment_id,
        config=config,
        corpora=corpora,
        protocols=protocols,
        existing_pretrain_runs=existing_formal,
        code_hash=current_code_hash,
        environment_hash=current_environment_hash,
    )
    source_hashes = tuple(corpus_by_domain[domain].logical_hash for domain in source_domain_ids)
    target_hash = corpus_by_domain[held_out_domain].logical_hash
    corpus_hashes = (*source_hashes, target_hash)
    protocol_hashes = tuple(protocol.protocol_hash for protocol in protocols)
    run_id = f"{experiment_id}-lodo-{variant}-{held_out_domain}-{seed}"
    run_dir = layout.gfm_runs / experiment_id / run_id
    state_path = run_dir / "run-state.json"
    source_embedding_evidence = _embedding_artifact_evidence(source_embeddings)
    role_view_contract = {
        "schemaVersion": "gfm.lodo-role-view-contract/1.0",
        "maximumRole": "validation",
        "physicalBoundary": True,
        "testReadCount": 0,
        "sourceDomainIds": list(source_domain_ids),
        "heldOutDomain": held_out_domain,
        "targetOpensAfterAllSourceStages": True,
        "corpusHashesByDomain": {
            domain: corpus_by_domain[domain].logical_hash for domain in sorted(corpus_by_domain)
        },
        "sourceEmbeddingArtifacts": source_embedding_evidence,
    }
    identity = LodoCellIdentity(
        experiment_id=experiment_id,
        run_id=run_id,
        held_out_domain=held_out_domain,
        source_domain_ids=source_domain_ids,
        architecture_variant=variant,
        seed=seed,
        config_hash=config.config_hash,
        code_hash=current_code_hash,
        environment_hash=current_environment_hash,
        corpus_hashes=corpus_hashes,
        protocol_hashes=protocol_hashes,
        role_view_contract=role_view_contract,
    )
    registry = _registry(layout)
    if not state_path.exists():
        if resume_requested:
            raise GfmTrainingError("LODO resume requires an existing durable run state")
        if run_dir.exists():
            unexpected = {
                entry.name for entry in run_dir.iterdir() if entry.name != ".lodo-execution.lock"
            }
            if unexpected:
                raise GfmTrainingError("Immutable LODO run directory already exists")
        state = create_lodo_run_state(
            state_path,
            identity=identity,
            device=device,
        )
        resume_payload: Mapping[str, Any] | None = None
    else:
        if not resume_requested:
            raise GfmTrainingError(f"Run {run_id} is interrupted; use gfm-resume --run-id {run_id}")
        raw_state = validate_lodo_run_state(
            read_json_object(state_path),
            identity=identity,
            allowed_statuses=("preflight", "running"),
        )
        if raw_state["status"] == "preflight":
            state, resume_payload = raw_state, None
        else:
            state, _, resume_payload = load_lodo_resume_checkpoint(state_path, identity=identity)

    def output_components(payload: Mapping[str, Any] | None) -> dict[str, dict[str, Any]]:
        if payload is None:
            return {}
        components = payload.get("components")
        if not isinstance(components, dict):
            raise ContractViolation("LODO progress checkpoint lacks components")
        permitted = {
            "source_multi",
            *(f"source_single__{domain}" for domain in source_domain_ids),
            "selected_target_5pct",
        }
        return {
            name: dict(value)
            for name, value in components.items()
            if name in permitted and isinstance(value, Mapping)
        }

    outputs = output_components(resume_payload)

    def require_completed_outputs() -> None:
        completed = set(state["execution"]["completedStages"])
        required: set[str] = set()
        if "source:multi" in completed:
            required.add("source_multi")
        required.update(
            f"source_single__{domain}"
            for domain in source_domain_ids
            if f"source:single:{domain}" in completed
        )
        if "target:5pct:gfm" in completed:
            required.add("selected_target_5pct")
        if not required.issubset(outputs):
            raise ContractViolation("LODO progress checkpoint lost a completed stage output")

    require_completed_outputs()

    def bind_views(streams: Mapping[str, _DomainStream], embeddings: Mapping[str, Any]) -> None:
        nonlocal state
        evidence: dict[str, Any] = {}
        embedding_evidence = _embedding_artifact_evidence(embeddings)
        for domain, stream in streams.items():
            formal_corpus = corpus_by_domain.get(domain)
            if (
                formal_corpus is None
                or stream.manifest.get("logicalHash") != formal_corpus.content_hash
            ):
                raise ContractViolation(
                    "LODO opened role-view corpus differs from its formal domain contract"
                )
            evidence[domain] = {
                "maximumRole": "validation",
                "formalCorpusLogicalHash": formal_corpus.logical_hash,
                "preparedCorpusContentHash": stream.manifest["logicalHash"],
                "corpusAccessAudit": stream.access_audit,
                "corpusAccessAuditHash": canonical_sha256(stream.access_audit),
                "embeddingArtifact": embedding_evidence.get(domain),
                "embeddingArtifactHash": (
                    None
                    if domain not in embedding_evidence
                    else canonical_sha256(embedding_evidence[domain])
                ),
                "testArtifactsOpened": False,
            }
        updated = bind_lodo_role_views(state, role_views=evidence)
        if updated != state:
            state = persist_lodo_run_state(state_path, updated, identity=identity)

    def persist_stage(
        stage: str,
        value: _LodoStageResult,
        *,
        state_override: Mapping[str, Any] | None = None,
    ) -> None:
        nonlocal state, resume_payload
        components: dict[str, Mapping[str, Any]] = {
            "current_core": value.model_state,
            **outputs,
        }
        if value.best_model_state is not None:
            components["best_core"] = value.best_model_state
        best_state = {
            "schemaVersion": "gfm.lodo-stage-selection/1.0",
            "stage": stage,
            "bestAvailable": value.best_model_state is not None,
            "validationLoss": value.best_validation_loss,
            "noImprovementEvaluations": value.no_improvement_evaluations,
            "peakCudaMemoryMiB": value.peak_cuda_memory_mib,
            "selectedEventIndicesHash": (
                None
                if not value.selected_event_indices
                else canonical_sha256(value.selected_event_indices)
            ),
        }
        state, progress_manifest = commit_lodo_progress(
            state_path,
            identity=identity,
            stage=stage,
            optimizer_step=int(value.trainer.optimizer_step),
            global_step=int(value.trainer.global_step),
            last_losses=value.last_losses,
            components=components,
            optimizer_state=value.trainer.optimizer.state_dict(),
            scheduler_state=value.scheduler.state_dict(),
            scaler_state=value.trainer.scaler.state_dict(),
            trainer_state=_lodo_trainer_resume_state(value.trainer),
            stream_states={domain: stream.state_dict() for domain, stream in value.streams.items()},
            best_state=best_state,
            config=config.logical_payload(),
            corpus_hashes=corpus_hashes,
            state_override=state_override,
            elapsed_seconds=max(
                0.0,
                (
                    datetime.now(UTC) - datetime.fromisoformat(str(state["startedAt"]))
                ).total_seconds(),
            ),
            rss_mib=_process_rss_mib(),
            peak_cuda_memory_mib=value.peak_cuda_memory_mib,
        )
        resume_payload = load_gfm_checkpoint(progress_manifest, map_location="cpu")

    def observe_stage(stage: str, value: _LodoStageResult) -> None:
        nonlocal state
        # Lightweight, operator-visible progress is intentionally not resume
        # authority.  A crash resumes from durableCheckpointStep and replays at
        # most the tail since the last 1000-step/validation commit.
        step = int(value.trainer.optimizer_step)
        durable = state.get("durableCheckpointStep")
        if (
            durable is None
            or state.get("durableCheckpointStage") != stage
            or (stage.startswith("target:") and step % 1_000 == 0)
        ):
            persist_stage(stage, value)
            return
        state = record_lodo_heartbeat(
            state_path,
            identity=identity,
            stage=stage,
            optimizer_step=step,
            global_step=int(value.trainer.global_step),
            last_losses=value.last_losses,
            stream_states={domain: stream.state_dict() for domain, stream in value.streams.items()},
            elapsed_seconds=max(
                0.0,
                (
                    datetime.now(UTC) - datetime.fromisoformat(str(state["startedAt"]))
                ).total_seconds(),
            ),
            rss_mib=_process_rss_mib(),
            peak_cuda_memory_mib=value.peak_cuda_memory_mib,
        )

    target_embeddings: dict[str, Any] | None = None
    target_stream: _DomainStream | None = None
    target_initial_state: dict[str, Any] | None = None
    while state["execution"]["currentStage"] is not None:
        stage = str(state["execution"]["currentStage"])
        active_resume = (
            resume_payload
            if isinstance(resume_payload, Mapping)
            and isinstance(resume_payload.get("sampler_state"), Mapping)
            and resume_payload["sampler_state"].get("stage") == stage
            else None
        )
        if stage.startswith("source:"):
            if stage == "source:multi":
                active_domains = source_domain_ids
                stage_seed = seed
                output_name = "source_multi"
            else:
                single_domain = stage.removeprefix("source:single:")
                if single_domain not in source_domain_ids:
                    raise ContractViolation("LODO source stage names an unknown domain")
                active_domains = (single_domain,)
                stage_seed = seed + 17 + source_domain_ids.index(single_domain)
                output_name = f"source_single__{single_domain}"
            source_streams = _make_domain_streams(
                layout,
                source_embeddings,
                domain_ids=active_domains,
                maximum_role="validation",
            )
            bind_views(source_streams, source_embeddings)

            def on_source_progress(value: _LodoStageResult, current_stage: str = stage) -> None:
                persist_stage(current_stage, value)

            def on_source_heartbeat(value: _LodoStageResult, current_stage: str = stage) -> None:
                observe_stage(current_stage, value)

            result = _train_lodo_source(
                config=config,
                variant=variant,
                streams=source_streams,
                seed=stage_seed,
                device=device,
                maximum_steps=source_steps,
                stage=stage,
                resume_payload=active_resume,
                progress=on_source_progress,
                heartbeat=on_source_heartbeat,
            )
            outputs[output_name] = result.model_state
            stream_identities = {
                domain: {
                    "contentHash": stream.manifest["logicalHash"],
                    "trainEnd": stream.train_end,
                    "trainCount": _stream_role_bounds(stream, 0)[1],
                    "splitStrategy": (
                        "explicit-page-disjoint"
                        if stream.event_split is not None
                        else "contiguous-temporal"
                    ),
                }
                for domain, stream in source_streams.items()
            }
            advanced = complete_lodo_stage(
                state,
                stage=stage,
                result={
                    "stageKind": "source-pretraining",
                    "domainIds": list(active_domains),
                    "completedSteps": int(result.trainer.optimizer_step),
                    "bestValidationLoss": result.best_validation_loss,
                    "stateHash": _state_digest(result.model_state),
                    "streamIdentities": stream_identities,
                    "temporalAuditCounters": _temporal_audit_counters(
                        streams=tuple(source_streams.values())
                    ),
                    "peakCudaMemoryMiB": result.peak_cuda_memory_mib,
                    "testReadCount": 0,
                },
            )
            persist_stage(stage, result, state_override=advanced)
            continue

        # The held-out arrays are first opened only after the exact three source
        # stage prefix has been committed in the durable execution snapshot.
        completed_source = {
            "source:multi",
            *(f"source:single:{domain}" for domain in source_domain_ids),
        }
        if not completed_source.issubset(state["execution"]["completedStages"]):
            raise ContractViolation("LODO target opened before all source states were frozen")
        if target_embeddings is None:
            _, target_embeddings = _ensure_pretrain_evidence(
                layout,
                embedding_domain_ids=(held_out_domain,),
                corpus_verification_domain_ids=(held_out_domain,),
                maximum_role="validation",
                physical_boundary=True,
            )
            target_stream = _make_domain_streams(
                layout,
                target_embeddings,
                domain_ids=(held_out_domain,),
                maximum_role="validation",
            )[held_out_domain]
            target_initial_state = target_stream.state_dict()
            bind_views({held_out_domain: target_stream}, target_embeddings)
        assert target_stream is not None and target_initial_state is not None
        if active_resume is None:
            target_stream.load_state_dict(target_initial_state)
        parts = stage.split(":")
        if len(parts) < 3 or parts[0] != "target":
            raise ContractViolation("LODO target stage identity is malformed")
        fraction_by_name = {"1pct": 0.01, "5pct": 0.05, "10pct": 0.1}
        fraction = fraction_by_name.get(parts[1])
        if fraction is None:
            raise ContractViolation("LODO target fraction is outside the fixed protocol")
        selection = _lodo_few_shot_selection(target_stream, seed=seed, fraction=fraction)
        chosen_ordinals, chosen_events = selection.ordinals, selection.event_indices
        selected = state["execution"]["selectedIndices"].get(stage)
        if selected is None:
            updated = bind_lodo_selected_indices(
                state,
                stage=stage,
                event_indices=chosen_events,
                fraction=selection.fraction,
                full_train_event_count=selection.full_train_event_count,
                eligible_pool_count=selection.eligible_pool_count,
                eligible_pool_hash=selection.eligible_pool_hash,
            )
            state = persist_lodo_run_state(state_path, updated, identity=identity)
        elif selected != selection.evidence:
            raise ContractViolation("LODO resumed few-shot selection differs")
        control = ":".join(parts[2:])
        if control == "gfm":
            source_state = outputs["source_multi"]
        elif control == "random-init":
            source_state = None
        elif control.startswith("single:"):
            single_domain = control.removeprefix("single:")
            if single_domain not in source_domain_ids:
                raise ContractViolation("LODO single-domain target control is unknown")
            source_state = outputs[f"source_single__{single_domain}"]
        else:
            raise ContractViolation("LODO target control is outside the fixed protocol")

        def on_target_progress(value: _LodoStageResult, current_stage: str = stage) -> None:
            persist_stage(current_stage, value)

        def on_target_heartbeat(value: _LodoStageResult, current_stage: str = stage) -> None:
            observe_stage(current_stage, value)

        result = _target_few_shot_score(
            config=config,
            variant=variant,
            stream=target_stream,
            seed=seed,
            fraction=fraction,
            device=device,
            source_state=source_state,
            maximum_steps=target_steps,
            stage=stage,
            chosen_ordinals=chosen_ordinals,
            chosen_events=chosen_events,
            resume_payload=active_resume,
            progress=on_target_progress,
            heartbeat=on_target_heartbeat,
        )
        if result.score is None:
            raise GfmTrainingError("LODO target stage produced no validation MRR")
        if stage == "target:5pct:gfm":
            outputs["selected_target_5pct"] = result.model_state
        advanced = complete_lodo_stage(
            state,
            stage=stage,
            result={
                "stageKind": "target-few-shot-validation",
                "fraction": fraction,
                "control": control,
                "completedSteps": int(result.trainer.optimizer_step),
                "validationMrr": result.score,
                "stateHash": _state_digest(result.model_state),
                "selectedEventIndicesHash": canonical_sha256(chosen_events),
                "selectedEventCount": len(chosen_events),
                "fullTrainEventCount": selection.full_train_event_count,
                "eligiblePoolCount": selection.eligible_pool_count,
                "eligiblePoolHash": selection.eligible_pool_hash,
                "heldOutTrainEnd": target_stream.train_end,
                "validationStartOrdinal": _stream_role_bounds(target_stream, 1)[0],
                "validationEventCount": _stream_role_bounds(target_stream, 1)[1],
                "peakCudaMemoryMiB": result.peak_cuda_memory_mib,
                "testReadCount": 0,
            },
        )
        persist_stage(stage, result, state_override=advanced)

    require_completed_outputs()
    completed = state["execution"]["completedStages"]
    metrics: dict[str, float] = {}
    best_single_domains: dict[str, str] = {}
    selected_indices: dict[str, dict[str, tuple[int, ...]]] = {}
    for fraction_name, metric_name in (("1pct", "1"), ("5pct", "5"), ("10pct", "10")):
        stage_prefix = f"target:{fraction_name}:"
        gfm_result = completed[f"{stage_prefix}gfm"]
        random_result = completed[f"{stage_prefix}random-init"]
        single_results = [
            (
                float(completed[f"{stage_prefix}single:{domain}"]["validationMrr"]),
                domain,
            )
            for domain in source_domain_ids
        ]
        single_score, best_single_domain = max(single_results, key=lambda item: (item[0], item[1]))
        metrics[f"few_shot_{metric_name}_gfm"] = float(gfm_result["validationMrr"])
        metrics[f"few_shot_{metric_name}_random_init"] = float(random_result["validationMrr"])
        metrics[f"few_shot_{metric_name}_single_domain"] = single_score
        best_single_domains[metric_name] = best_single_domain
        stage_names = (
            f"{stage_prefix}gfm",
            f"{stage_prefix}random-init",
            *(f"{stage_prefix}single:{domain}" for domain in source_domain_ids),
        )
        index_values = [
            tuple(state["execution"]["selectedIndices"][name]["eventIndices"])
            for name in stage_names
        ]
        if any(value != index_values[0] for value in index_values[1:]):
            raise GfmTrainingError("LODO controls did not use identical few-shot rows")
        selected_indices[metric_name] = {"all_controls": index_values[0]}
    source_state = outputs["source_multi"]
    single_states = {domain: outputs[f"source_single__{domain}"] for domain in source_domain_ids}
    selected_state = outputs["selected_target_5pct"]
    source_multi_result = completed["source:multi"]
    stream_identities = source_multi_result["streamIdentities"]
    adapter_hashes = tuple(
        canonical_sha256({"domainId": domain, **stream_identities[domain]})
        for domain in source_domain_ids
    )
    family = next(
        domain.domain_family for domain in config.domains if domain.domain_id == held_out_domain
    )
    isolation = LodoIsolationAudit(
        held_out_family=family,
        source_domain_ids=source_domain_ids,
        target_domain_ids=(held_out_domain,),
        source_corpus_hashes=source_hashes,
        target_corpus_hashes=(target_hash,),
        verified_corpus_hashes=tuple(sorted(corpus.logical_hash for corpus in corpora)),
        adapter_statistic_hashes=adapter_hashes,
        excluded_academic_sibling_ids=("ogbl-collab", "ogbn-arxiv"),
        academic_sibling_access_count=0,
        academic_sibling_exclusion_evidence_hash=canonical_sha256(
            {
                "excluded": ["ogbl-collab", "ogbn-arxiv"],
                "pretrainingLoadedDomainIds": list(source_domain_ids),
                "targetLoadedAfterSourcePretraining": held_out_domain,
                "heldOutDomain": held_out_domain,
                "accessCount": 0,
            }
        ),
        target_adapter_initialized_after_pretraining=True,
    )
    isolation_hash = assert_lodo_isolation(isolation)
    peak = max(float(result["peakCudaMemoryMiB"]) for result in completed.values())
    if peak >= config.optimization.cuda_memory_limit_mib:
        raise GfmTrainingError("LODO exceeded the fixed 7168 MiB CUDA limit")
    lodo_config = {
        "schemaVersion": "gfm.lodo-config/1.0",
        "architectureVariant": variant,
        "seed": seed,
        "heldOutDomain": held_out_domain,
        "sourceDomainIds": list(source_domain_ids),
        "sourceSteps": source_steps,
        "targetStepsPerControl": target_steps,
        "fewShotFractions": [0.01, 0.05, 0.1],
        "isolationHash": isolation_hash,
        "sourceStateHash": _state_digest(source_state),
        "singleDomainStateHashes": {
            domain: _state_digest(state) for domain, state in sorted(single_states.items())
        },
        "bestSingleDomainByFraction": best_single_domains,
        "embeddingArtifacts": {
            domain: evidence.get("embeddingArtifact")
            for domain, evidence in sorted(state["execution"]["roleViews"].items())
            if evidence.get("embeddingArtifact") is not None
        },
        "roleViews": state["execution"]["roleViews"],
        "roleViewsHash": state["execution"]["roleViewsHash"],
        "executionHash": canonical_sha256(state["execution"]),
        "testReadCount": 0,
    }
    lodo_config["taskConfigHash"] = canonical_sha256(lodo_config)
    checkpoint_id = f"{run_id}-best-5pct"
    checkpoint_path = run_dir / "checkpoints" / f"{checkpoint_id}.manifest.json"
    if checkpoint_path.is_file():
        checkpoint = read_gfm_checkpoint_manifest(checkpoint_path)
        loaded_best = load_gfm_checkpoint(checkpoint, map_location="cpu")
        if (
            checkpoint.run_id != run_id
            or checkpoint.config_hash != config.config_hash
            or checkpoint.corpus_hashes != corpus_hashes
            or _state_digest(loaded_best["components"]["core"]) != _state_digest(selected_state)
        ):
            raise ContractViolation("LODO terminal checkpoint differs during reconciliation")
    else:
        checkpoint = save_gfm_checkpoint(
            run_dir / "checkpoints",
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            epoch=0,
            step=target_steps,
            components={"core": selected_state},
            optimizer_state={},
            scheduler_state={},
            scaler_state={},
            sampler_state={
                "selectedIndexHashes": {
                    fraction: canonical_sha256(indices["all_controls"])
                    for fraction, indices in selected_indices.items()
                },
                "heldOutTrainEnd": int(completed["target:5pct:gfm"]["heldOutTrainEnd"]),
                "roleViews": state["execution"]["roleViews"],
                "roleViewsHash": state["execution"]["roleViewsHash"],
                "execution": state["execution"],
                "executionHash": canonical_sha256(state["execution"]),
                "testReadCount": 0,
            },
            best_state={"metrics": metrics, "fraction": 0.05},
            config=config.logical_payload(),
            corpus_hashes=corpus_hashes,
        )
    terminal_prepared = state.get("terminalPreparedAt")
    if terminal_prepared is None:
        terminal_prepared = datetime.now(UTC).isoformat()
        state = persist_lodo_run_state(
            state_path,
            {**state, "terminalPreparedAt": terminal_prepared},
            identity=identity,
            allowed_statuses=("running",),
        )
    started = datetime.fromisoformat(str(state["startedAt"]))
    finished = datetime.fromisoformat(str(terminal_prepared))
    run = GfmRunManifest.create(
        runId=run_id,
        experimentId=experiment_id,
        phase="lodo",
        architectureVariant=variant,
        status="succeeded",
        domainIds=source_domain_ids,
        heldOutDomain=held_out_domain,
        seed=seed,
        codeHash=current_code_hash,
        environmentHash=current_environment_hash,
        configHash=config.config_hash,
        corpusHashes=corpus_hashes,
        taskProtocolHashes=protocol_hashes,
        startedAt=started,
        finishedAt=finished,
        peakCudaMemoryMiB=peak,
        artifactPaths=(str(run_dir / "checkpoints" / f"{checkpoint_id}.manifest.json"),),
    )
    _write_contract(run_dir / "run-manifest.json", run)
    atomic_write_json(run_dir / "lodo-config.json", lodo_config)
    registry.record_completed_run(run, checkpoint)
    access_ledger = [
        {
            "sequence": 0,
            "stage": "source-pretraining",
            "domainIds": list(source_domain_ids),
            "corpusHashes": list(source_hashes),
            "completedStageHashes": {
                stage: completed[stage]["resultHash"]
                for stage in ("source:multi", *(f"source:single:{d}" for d in source_domain_ids))
            },
        },
        {
            "sequence": 1,
            "stage": "target-few-shot-after-source-state-frozen",
            "domainIds": [held_out_domain],
            "sourceStateHash": _state_digest(source_state),
        },
    ]
    audit_hash, audit_path, counters = _leakage_audit(
        layout,
        experiment_id=experiment_id,
        audit_id=f"{run_id}-isolation",
        evidence={
            "heldOutFamily": family,
            "heldOutDomain": held_out_domain,
            "sourceDomainIds": list(source_domain_ids),
            "sourceCorpusHashes": source_hashes,
            "targetCorpusHash": target_hash,
            "adapterInitializedAfterPretraining": True,
            "pretrainingLoadedDomainIds": list(source_domain_ids),
            "targetLoadedAfterSourcePretraining": held_out_domain,
            "domainAccessLedger": access_ledger,
            "domainAccessLedgerHash": canonical_sha256(access_ledger),
            "isolationHash": isolation_hash,
            "selectedIndexHashes": {
                fraction: canonical_sha256(indices["all_controls"])
                for fraction, indices in selected_indices.items()
            },
            "sampler": "causal-visible-only-exact-mixed-v1",
            "validationStartOrdinal": int(completed["target:5pct:gfm"]["validationStartOrdinal"]),
            "validationEventCount": int(completed["target:5pct:gfm"]["validationEventCount"]),
            "validationCandidateCountPerQuery": 100,
        },
        counters={
            **completed["source:multi"]["temporalAuditCounters"],
            "target_domain_pretrain_access_count": 0,
        },
    )
    lodo_metrics = {**metrics, **counters}
    lodo_evidence_hash, lodo_evidence_path = _evaluation_evidence(
        layout,
        experiment_id=experiment_id,
        evidence_id=f"{run_id}-lodo",
        payload={
            "checkpointId": checkpoint_id,
            "checkpointStateHash": checkpoint.state_hash,
            "isolationHash": isolation_hash,
            "sourceStateHash": _state_digest(source_state),
            "singleDomainStateHashes": {
                domain: _state_digest(state) for domain, state in sorted(single_states.items())
            },
            "bestSingleDomainByFraction": best_single_domains,
            "selectedIndexHashes": {
                fraction: canonical_sha256(indices["all_controls"])
                for fraction, indices in selected_indices.items()
            },
            "metrics": lodo_metrics,
        },
    )
    report_id = f"{run_id}-lodo"
    existing_report = next(
        (
            value
            for value in registry.list_evaluations(experiment_id=experiment_id)
            if value.report_id == report_id
        ),
        None,
    )
    report = existing_report or GfmEvaluationReport.create(
        reportId=report_id,
        experimentId=experiment_id,
        runId=run_id,
        checkpointId=checkpoint_id,
        evaluationKind="lodo",
        domainId=held_out_domain,
        heldOutDomain=held_out_domain,
        seed=seed,
        metrics=lodo_metrics,
        evidenceArtifactHash=lodo_evidence_hash,
        evidenceArtifactPath=lodo_evidence_path,
        peakCudaMemoryMiB=peak,
        leakageAuditPassed=True,
        leakageAuditHash=audit_hash,
        leakageAuditPath=audit_path,
        createdAt=finished,
    )
    registry.record_evaluation(report)
    state = mark_lodo_succeeded(
        state_path,
        identity=identity,
        best_checkpoint_id=checkpoint.checkpoint_id,
        finished_at=finished,
    )
    return {
        "runId": run_id,
        "checkpointId": checkpoint_id,
        "heldOutDomain": held_out_domain,
        "architectureVariant": variant,
        "seed": seed,
        "metrics": metrics,
        "isolationAuditHash": audit_hash,
        "peakCudaMemoryMiB": peak,
        "durableState": state["status"],
        "completedStageCount": len(state["execution"]["completedStages"]),
    }


__all__ = []
