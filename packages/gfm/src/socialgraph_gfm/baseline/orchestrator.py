"""Deterministic run planning and callback-driven baseline orchestration."""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable, Mapping

from .protocols import build_protocol
from .trainer import evaluate_heuristic_run, train_learning_run
from .types import CoreRunResult, CorpusArrays, RunSpec


def _value(config: Any, snake: str, camel: str, default: Any) -> Any:
    if hasattr(config, snake):
        return getattr(config, snake)
    if isinstance(config, Mapping):
        return config.get(snake, config.get(camel, default))
    return default


def _run_id(experiment_id: str, phase: str, track: str, model: str, seed: int) -> str:
    readable = f"{experiment_id}-{phase}-{track}-{model}-{seed}"
    if len(readable) <= 128:
        return readable
    suffix = hashlib.sha256(readable.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:115]}-{suffix}"


def build_run_specs(
    *,
    experiment_id: str,
    phase: str,
    config: Any,
    tracks: Iterable[str] | str = "both",
) -> tuple[RunSpec, ...]:
    """Build 10 dev runs or the fixed 18-run formal matrix.

    Formal consists of six heuristic runs and twelve learning runs.  This is the
    single source of truth for acceptance-matrix cardinality.
    """

    if phase not in ("dev", "formal"):
        raise ValueError("phase must be dev or formal")
    selected_tracks = (
        ("ogb_official", "strict_edge_time")
        if tracks == "both"
        else ((tracks,) if isinstance(tracks, str) else tuple(tracks))
    )
    if not selected_tracks or any(
        track not in ("ogb_official", "strict_edge_time") for track in selected_tracks
    ):
        raise ValueError("tracks must select ogb_official, strict_edge_time, or both")
    models = tuple(_value(config, "models", "models", ("cn", "aa", "ra", "mlp", "graphsage")))
    if any(model not in ("cn", "aa", "ra", "mlp", "graphsage") for model in models):
        raise ValueError("baseline config contains an unsupported model")
    dev_seed = int(_value(config, "dev_seed", "devSeed", 20260811))
    formal_seeds = tuple(
        int(seed)
        for seed in _value(
            config, "formal_seeds", "formalSeeds", (20260812, 20260813, 20260814)
        )
    )
    specs: list[RunSpec] = []
    for track in selected_tracks:
        for model in models:
            seeds = (
                (dev_seed,)
                if phase == "dev" or model in ("cn", "aa", "ra")
                else formal_seeds
            )
            if phase == "formal" and model in ("cn", "aa", "ra"):
                seeds = (formal_seeds[0],)
            for seed in seeds:
                specs.append(
                    RunSpec(
                        experiment_id=experiment_id,
                        run_id=_run_id(experiment_id, phase, track, model, seed),
                        phase=phase,  # type: ignore[arg-type]
                        track=track,  # type: ignore[arg-type]
                        model=model,  # type: ignore[arg-type]
                        seed=seed,
                    )
                )
    return tuple(specs)


def run_core_spec(
    spec: RunSpec,
    *,
    corpus: CorpusArrays,
    config: Any,
    device: str,
    evaluator=None,
    checkpoint_sink=None,
    resume_state: Mapping[str, Any] | None = None,
) -> CoreRunResult:
    protocol = build_protocol(corpus, spec.track)
    if spec.model in ("cn", "aa", "ra"):
        if resume_state is not None:
            raise ValueError("deterministic heuristic runs do not have resumable state")
        return evaluate_heuristic_run(
            spec, corpus=corpus, protocol=protocol, evaluator=evaluator
        )
    return train_learning_run(
        spec,
        corpus=corpus,
        protocol=protocol,
        config=config,
        device=device,
        evaluator=evaluator,
        checkpoint_sink=checkpoint_sink,
        resume_state=resume_state,
    )


def run_experiment_core(
    *,
    experiment_id: str,
    phase: str,
    tracks: Iterable[str] | str,
    corpus: CorpusArrays,
    config: Any,
    device: str,
    on_result: Callable[[CoreRunResult], None] | None = None,
    checkpoint_sink_factory: Callable[[RunSpec], Any] | None = None,
) -> tuple[CoreRunResult, ...]:
    """Execute a matrix while leaving manifests/registry ownership to the caller."""

    results = []
    for spec in build_run_specs(
        experiment_id=experiment_id, phase=phase, config=config, tracks=tracks
    ):
        sink = checkpoint_sink_factory(spec) if checkpoint_sink_factory is not None else None
        result = run_core_spec(
            spec,
            corpus=corpus,
            config=config,
            device=device,
            checkpoint_sink=sink,
        )
        if on_result is not None:
            on_result(result)
        results.append(result)
    return tuple(results)

