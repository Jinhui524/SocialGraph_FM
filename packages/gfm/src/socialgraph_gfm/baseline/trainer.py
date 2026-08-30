"""Training, batch probing, early stopping and final-only test evaluation."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from .evaluator import evaluate_scores, stratified_positive_metrics
from .heuristics import score_all_heuristics, score_heuristic
from .sampling import ExactUndirectedNegativeSampler, forbidden_union
from .types import CoreRunResult, CorpusArrays, ProtocolBundle, RunSpec, TemporalStage


class CheckpointSink(Protocol):
    def __call__(self, kind: str, payload: Mapping[str, Any]) -> Any: ...


MetricEvaluator = Callable[[Any, Any], Mapping[str, float]]


def _config_value(config: Any, snake: str, camel: str, default: Any) -> Any:
    if hasattr(config, snake):
        return getattr(config, snake)
    if isinstance(config, Mapping):
        if snake in config:
            return config[snake]
        if camel in config:
            return config[camel]
    return default


def _edge_tensor(edges: Any, *, device: str = "cpu"):
    import torch

    return torch.as_tensor(edges, dtype=torch.long, device=device).t().contiguous()


def _capture_rng_state(torch) -> dict[str, Any]:
    import numpy as np

    numpy_state: Any = np.random.get_state()
    algorithm, keys, position, has_gauss, cached_gaussian = numpy_state
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": {
            "algorithm": str(algorithm),
            "keys": [int(value) for value in keys.tolist()],
            "position": int(position),
            "hasGauss": int(has_gauss),
            "cachedGaussian": float(cached_gaussian),
        },
        "torchCpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torchCuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(torch, state: Mapping[str, Any]) -> None:
    import numpy as np

    required = {"python", "numpy", "torchCpu"}
    if not required.issubset(state):
        raise ValueError("resume RNG state is incomplete")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("resume NumPy RNG state is not in the safe checkpoint format")
    np.random.set_state(
        (
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["hasGauss"]),
            float(numpy_state["cachedGaussian"]),
        )
    )
    torch.set_rng_state(state["torchCpu"].cpu())
    if "torchCuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torchCuda"])


def _set_seed(seed: int, device: str) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def probe_cuda_batch_size(
    probe_step: Callable[[int], None],
    *,
    device: str,
    candidates: tuple[int, ...] = (4096, 2048, 1024),
    memory_limit_mib: float = 7168.0,
) -> tuple[int, float]:
    """Select the largest candidate whose real probe stays below the hard limit."""

    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("batch probe candidates must be positive")
    if device != "cuda":
        probe_step(candidates[0])
        return candidates[0], 0.0
    import torch

    failures: list[str] = []
    for candidate in candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            probe_step(candidate)
            torch.cuda.synchronize()
            peak = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
            if peak < memory_limit_mib:
                return candidate, peak
            failures.append(f"{candidate}: peak {peak:.1f} MiB")
        except torch.cuda.OutOfMemoryError:
            failures.append(f"{candidate}: CUDA OOM")
        finally:
            torch.cuda.empty_cache()
    raise RuntimeError("no candidate batch size passed CUDA preflight: " + "; ".join(failures))


def _score_feature_model(model, x, pairs: Any, *, batch_size: int) -> Any:
    import numpy as np
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            edge_index = _edge_tensor(pairs[start : start + batch_size], device=str(x.device))
            scores.append(model(x, edge_index).detach().cpu())
    return torch.cat(scores).numpy() if scores else np.empty(0, dtype=np.float32)


def _score_graph_pairs(
    model, embeddings, pairs: Any, *, device: str, score_batch_size: int
) -> Any:
    import numpy as np
    import torch

    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(pairs), score_batch_size):
            edge_index = _edge_tensor(pairs[start : start + score_batch_size])
            source = embeddings[edge_index[0]].to(device)
            target = embeddings[edge_index[1]].to(device)
            scores.append(model.predictor(source, target).detach().cpu())
    return torch.cat(scores).numpy() if scores else np.empty(0, dtype=np.float32)


def _evaluate_learning(
    model,
    *,
    model_name: str,
    x,
    stage: TemporalStage,
    score_batch_size: int,
    inference_batch_size: int,
    evaluator: MetricEvaluator | None,
) -> tuple[dict[str, float], Any, Any]:
    if stage.negative_edges is None:
        raise ValueError(f"{stage.name} requires fixed evaluation negatives")
    if model_name == "mlp":
        positive = _score_feature_model(
            model, x, stage.positive_edges, batch_size=score_batch_size
        )
        negative = _score_feature_model(
            model, x, stage.negative_edges, batch_size=score_batch_size
        )
    else:
        model.eval()
        embeddings = model.inference(
            x.detach().cpu(),
            _edge_tensor(stage.message_edges),
            device=str(x.device),
            batch_size=inference_batch_size,
        )
        positive = _score_graph_pairs(
            model,
            embeddings,
            stage.positive_edges,
            device=str(x.device),
            score_batch_size=score_batch_size,
        )
        negative = _score_graph_pairs(
            model,
            embeddings,
            stage.negative_edges,
            device=str(x.device),
            score_batch_size=score_batch_size,
        )
    metrics = evaluate_scores(
        positive,
        negative,
        **({"evaluator": evaluator} if evaluator is not None else {}),
    )
    return metrics, positive, negative


def evaluate_heuristic_run(
    spec: RunSpec,
    *,
    corpus: CorpusArrays,
    protocol: ProtocolBundle,
    evaluator: MetricEvaluator | None = None,
) -> CoreRunResult:
    """Run one deterministic structural baseline; dev runs never access test."""

    def evaluate(stage: TemporalStage) -> tuple[dict[str, float], Any, Any]:
        if stage.negative_edges is None:
            raise ValueError(f"{stage.name} requires fixed negatives")
        positive = score_heuristic(
            spec.model,  # type: ignore[arg-type]
            num_nodes=corpus.num_nodes,
            message_edges=stage.message_edges,
            candidate_edges=stage.positive_edges,
        )
        negative = score_heuristic(
            spec.model,  # type: ignore[arg-type]
            num_nodes=corpus.num_nodes,
            message_edges=stage.message_edges,
            candidate_edges=stage.negative_edges,
        )
        metrics = evaluate_scores(
            positive,
            negative,
            **({"evaluator": evaluator} if evaluator is not None else {}),
        )
        return metrics, positive, negative

    validation, validation_pos, validation_neg = evaluate(protocol.validation)
    test = None
    strata_stage = protocol.validation
    strata_positive, strata_negative = validation_pos, validation_neg
    if spec.phase == "formal":
        # This is intentionally the sole test-stage read in this function and it
        # occurs after the immutable heuristic has been selected on validation.
        test, strata_positive, strata_negative = evaluate(protocol.test)
        strata_stage = protocol.test
    strata = (
        stratified_positive_metrics(
            strata_positive, strata_negative, strata_stage.repeated_mask
        )
        if protocol.track == "strict_edge_time"
        else {}
    )
    return CoreRunResult(
        spec=spec,
        validation_metrics=validation,
        test_metrics=test,
        strata=strata,
        best_epoch=None,
        peak_cuda_memory_mib=0.0,
        selected_batch_size=None,
        test_read_after_selection=spec.phase == "formal",
    )


def evaluate_heuristic_bundle(
    specs: tuple[RunSpec, ...],
    *,
    corpus: CorpusArrays,
    protocol: ProtocolBundle,
    evaluator: MetricEvaluator | None = None,
) -> tuple[CoreRunResult, ...]:
    """Evaluate the complete CN/AA/RA group without rebuilding adjacency per model."""

    import numpy as np

    if {spec.model for spec in specs} != {"cn", "aa", "ra"}:
        raise ValueError("heuristic bundle requires exactly CN, AA and RA")
    identity = {(spec.phase, spec.track, spec.experiment_id) for spec in specs}
    if len(identity) != 1:
        raise ValueError("heuristic bundle specs must belong to one experiment/track/phase")

    def evaluate(stage: TemporalStage) -> dict[str, tuple[dict[str, float], Any, Any]]:
        if stage.negative_edges is None:
            raise ValueError(f"{stage.name} requires fixed negatives")
        combined = np.concatenate((stage.positive_edges, stage.negative_edges), axis=0)
        combined_scores = score_all_heuristics(
            num_nodes=corpus.num_nodes,
            message_edges=stage.message_edges,
            candidate_edges=combined,
        )
        boundary = len(stage.positive_edges)
        values: dict[str, tuple[dict[str, float], Any, Any]] = {}
        for name, scores in combined_scores.items():
            positive = scores[:boundary]
            negative = scores[boundary:]
            metrics = evaluate_scores(
                positive,
                negative,
                **({"evaluator": evaluator} if evaluator is not None else {}),
            )
            values[name] = (metrics, positive, negative)
        return values

    validation = evaluate(protocol.validation)
    test = evaluate(protocol.test) if specs[0].phase == "formal" else None
    results = []
    for spec in specs:
        validation_metrics, validation_pos, validation_neg = validation[spec.model]
        test_metrics = None
        strata_stage = protocol.validation
        strata_positive, strata_negative = validation_pos, validation_neg
        if test is not None:
            test_metrics, strata_positive, strata_negative = test[spec.model]
            strata_stage = protocol.test
        strata = (
            stratified_positive_metrics(
                strata_positive, strata_negative, strata_stage.repeated_mask
            )
            if protocol.track == "strict_edge_time"
            else {}
        )
        results.append(
            CoreRunResult(
                spec=spec,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                strata=strata,
                best_epoch=None,
                peak_cuda_memory_mib=0.0,
                selected_batch_size=None,
                test_read_after_selection=spec.phase == "formal",
            )
        )
    return tuple(results)


@dataclass
class _BestState:
    epoch: int = -1
    hits50: float = float("-inf")
    model_state: dict[str, Any] | None = None


def _parameter_state(model, model_name: str) -> dict[str, Any]:
    return {
        "model_state": (
            copy.deepcopy(model.encoder.state_dict()) if model_name == "graphsage" else {}
        ),
        "predictor_state": copy.deepcopy(model.predictor.state_dict()),
    }


def _optimizer_to(optimizer, device: str) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _validate_resume_identity(
    state: Mapping[str, Any], spec: RunSpec, corpus: CorpusArrays
) -> None:
    expected = {
        "run_id": spec.run_id,
        "track": spec.track,
        "model": spec.model,
        "corpus_hash": corpus.corpus_hash,
    }
    for key, value in expected.items():
        if key in state and state[key] != value:
            raise ValueError(f"resume checkpoint {key} does not match the requested run")
    required = {
        "epoch",
        "model_state",
        "predictor_state",
        "optimizer_state",
        "rng_state",
        "sampler_state",
        "selection_rng_state",
        "best_validation_hits50",
        "best_epoch",
        "best_model_state",
        "best_predictor_state",
        "selected_batch_size",
        "evaluations_without_improvement",
        "history",
        "terminal",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError("resume checkpoint is incomplete: " + ", ".join(missing))


def _checkpoint_payload(
    *,
    model,
    model_name: str,
    optimizer,
    epoch: int,
    best: _BestState,
    sampler,
    selection_rng,
    selected_batch_size: int,
    evaluations_without_improvement: int,
    history: list[dict[str, float]],
    terminal: bool,
) -> dict[str, Any]:
    import torch

    if model_name == "graphsage":
        model_state = copy.deepcopy(model.encoder.state_dict())
        predictor_state = copy.deepcopy(model.predictor.state_dict())
    else:
        model_state = {}
        predictor_state = copy.deepcopy(model.predictor.state_dict())
    if best.model_state is None:
        raise RuntimeError("cannot checkpoint before a best model is selected")
    return {
        "epoch": epoch,
        "model_state": model_state,
        "predictor_state": predictor_state,
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler_state": None,
        "rng_state": _capture_rng_state(torch),
        "sampler_state": sampler.state_dict(),
        "selection_rng_state": copy.deepcopy(selection_rng.bit_generator.state),
        "best_validation_hits50": best.hits50,
        "best_epoch": best.epoch,
        "best_model_state": copy.deepcopy(best.model_state["model_state"]),
        "best_predictor_state": copy.deepcopy(best.model_state["predictor_state"]),
        "selected_batch_size": selected_batch_size,
        "evaluations_without_improvement": evaluations_without_improvement,
        "history": copy.deepcopy(history),
        "terminal": bool(terminal),
    }


def _restore_best(model, model_name: str, state: Mapping[str, Any]) -> None:
    if model_name == "graphsage":
        model.encoder.load_state_dict(state["model_state"])
    model.predictor.load_state_dict(state["predictor_state"])


def _restore_current(model, model_name: str, state: Mapping[str, Any]) -> None:
    if model_name == "graphsage":
        model.encoder.load_state_dict(state["model_state"])
    model.predictor.load_state_dict(state["predictor_state"])


def train_learning_run(
    spec: RunSpec,
    *,
    corpus: CorpusArrays,
    protocol: ProtocolBundle,
    config: Any,
    device: str,
    evaluator: MetricEvaluator | None = None,
    checkpoint_sink: CheckpointSink | None = None,
    resume_state: Mapping[str, Any] | None = None,
) -> CoreRunResult:
    """Train MLP/GraphSAGE with validation-only selection and final-only test.

    ``checkpoint_sink`` receives complete state dictionaries compatible with
    :func:`socialgraph_gfm.checkpoint.save_baseline_checkpoint`.
    """

    if spec.model not in ("mlp", "graphsage"):
        raise ValueError("train_learning_run supports only mlp and graphsage")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    import numpy as np
    import torch

    from .models import FeatureMLP, GraphSAGELinkModel

    _set_seed(spec.seed, device)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.reset_peak_memory_stats()
    hidden = int(_config_value(config, "hidden_channels", "hiddenChannels", 128))
    dropout = float(_config_value(config, "dropout", "dropout", 0.2))
    learning_rate = float(_config_value(config, "learning_rate", "learningRate", 0.001))
    weight_decay = float(_config_value(config, "weight_decay", "weightDecay", 0.0))
    gradient_clip = float(_config_value(config, "gradient_clip", "gradientClip", 1.0))
    negative_ratio = float(_config_value(config, "negative_ratio", "negativeRatio", 1.0))
    score_batch_size = int(_config_value(config, "score_batch_size", "scoreBatchSize", 65536))
    inference_batch_size = int(
        _config_value(config, "inference_batch_size", "inferenceBatchSize", 8192)
    )
    candidate_sizes = tuple(
        int(value)
        for value in _config_value(
            config, "candidate_batch_sizes", "candidateBatchSizes", (4096, 2048, 1024)
        )
    )
    memory_limit = float(
        _config_value(config, "cuda_memory_limit_mib", "cudaMemoryLimitMiB", 7168)
    )
    if spec.phase == "dev":
        max_epochs = int(_config_value(config, "dev_epochs", "devEpochs", 5))
        min_epochs = max_epochs
        positive_limit = int(
            _config_value(config, "dev_positive_limit", "devPositiveLimit", 50000)
        )
        eval_every = 1
        patience = max_epochs
    else:
        max_epochs = int(
            _config_value(config, "formal_max_epochs", "formalMaxEpochs", 50)
        )
        min_epochs = int(
            _config_value(config, "formal_min_epochs", "formalMinEpochs", 10)
        )
        positive_limit = int(
            _config_value(config, "train_positive_limit", "trainPositiveLimit", 262144)
        )
        eval_every = int(_config_value(config, "eval_every", "evalEvery", 2))
        patience = int(_config_value(config, "patience", "patience", 8))

    x = torch.as_tensor(corpus.node_features, dtype=torch.float32, device=device)
    model: Any
    if spec.model == "mlp":
        model = FeatureMLP(x.shape[1], hidden_channels=hidden, dropout=dropout).to(device)
    else:
        model = GraphSAGELinkModel(x.shape[1], hidden_channels=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    train_stage = protocol.train
    sampler = ExactUndirectedNegativeSampler(
        corpus.num_nodes,
        forbidden_union(train_stage.message_edges, train_stage.positive_edges),
        seed=spec.seed,
    )
    rng = np.random.default_rng(spec.seed)
    max_positive = min(positive_limit, len(train_stage.positive_edges))
    if max_positive < 1:
        raise ValueError("training protocol has no positive supervision")

    def training_pairs(epoch: int) -> tuple[Any, Any]:
        indices = rng.permutation(len(train_stage.positive_edges))[:max_positive]
        positives = train_stage.positive_edges[indices]
        negative_count = int(round(len(positives) * negative_ratio))
        negatives = sampler.sample(negative_count)
        pairs = np.concatenate((positives, negatives), axis=0)
        labels = np.concatenate(
            (np.ones(len(positives), dtype=np.float32), np.zeros(len(negatives), dtype=np.float32))
        )
        permutation = rng.permutation(len(pairs))
        return pairs[permutation], labels[permutation]

    best = _BestState()
    evaluations_without_improvement = 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    probe_peak = 0.0

    if resume_state is not None:
        _validate_resume_identity(resume_state, spec, corpus)
        resumed_epoch = int(resume_state["epoch"])
        if bool(resume_state["terminal"]) or resumed_epoch >= max_epochs:
            raise ValueError("a terminal or max-epoch checkpoint cannot be resumed")
        _restore_current(model, spec.model, resume_state)
        optimizer.load_state_dict(resume_state["optimizer_state"])
        _optimizer_to(optimizer, device)
        sampler.load_state_dict(dict(resume_state["sampler_state"]))
        rng.bit_generator.state = copy.deepcopy(resume_state["selection_rng_state"])
        _restore_rng_state(torch, resume_state["rng_state"])
        best = _BestState(
            epoch=int(resume_state["best_epoch"]),
            hits50=float(resume_state["best_validation_hits50"]),
            model_state={
                "model_state": copy.deepcopy(resume_state["best_model_state"]),
                "predictor_state": copy.deepcopy(resume_state["best_predictor_state"]),
            },
        )
        if best.epoch < 0 or not np.isfinite(best.hits50):
            raise ValueError("resume checkpoint has no valid historical best state")
        evaluations_without_improvement = int(
            resume_state["evaluations_without_improvement"]
        )
        history = [dict(item) for item in resume_state["history"]]
        selected_batch_size = int(resume_state["selected_batch_size"])
        if selected_batch_size not in candidate_sizes:
            raise ValueError("resume checkpoint batch size is outside the frozen config")
        start_epoch = resumed_epoch + 1
    else:
        # Probe must not alter the sampler/selection/global RNG sequence used by
        # epoch one.  This keeps runs identical when a larger candidate OOMs.
        sampler_state_before_probe = sampler.state_dict()
        selection_state_before_probe = copy.deepcopy(rng.bit_generator.state)
        global_state_before_probe = _capture_rng_state(torch)
        probe_pairs, probe_labels = training_pairs(0)

        def probe_step(batch_size: int) -> None:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            count = min(batch_size, len(probe_pairs))
            if spec.model == "mlp":
                logits = model(x, _edge_tensor(probe_pairs[:count], device=device))
                labels = torch.as_tensor(
                    probe_labels[:count], dtype=torch.float32, device=device
                )
            else:
                loader = _link_loader(
                    corpus.node_features,
                    train_stage.message_edges,
                    probe_pairs,
                    probe_labels,
                    batch_size=count,
                    fanout=tuple(
                        int(value)
                        for value in _config_value(
                            config, "neighbor_fanout", "neighborFanout", (15, 10)
                        )
                    ),
                    shuffle=False,
                )
                batch = next(iter(loader)).to(device)
                logits = model(batch.x, batch.edge_index, batch.edge_label_index)
                labels = batch.edge_label.float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("batch preflight produced non-finite loss")
            loss.backward()
            optimizer.zero_grad(set_to_none=True)

        selected_batch_size, probe_peak = probe_cuda_batch_size(
            probe_step,
            device=device,
            candidates=candidate_sizes,
            memory_limit_mib=memory_limit,
        )
        sampler.load_state_dict(sampler_state_before_probe)
        rng.bit_generator.state = selection_state_before_probe
        _restore_rng_state(torch, global_state_before_probe)

    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        pairs, labels_np = training_pairs(epoch)
        total_loss = 0.0
        examples = 0
        if spec.model == "mlp":
            order = torch.randperm(len(pairs), generator=torch.Generator().manual_seed(spec.seed + epoch))
            for start in range(0, len(pairs), selected_batch_size):
                indices = order[start : start + selected_batch_size].numpy()
                pair_batch = _edge_tensor(pairs[indices], device=device)
                labels = torch.as_tensor(labels_np[indices], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, pair_batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * len(indices)
                examples += len(indices)
        else:
            loader = _link_loader(
                corpus.node_features,
                train_stage.message_edges,
                pairs,
                labels_np,
                batch_size=selected_batch_size,
                fanout=tuple(
                    int(value)
                    for value in _config_value(
                        config, "neighbor_fanout", "neighborFanout", (15, 10)
                    )
                ),
                shuffle=True,
            )
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.x, batch.edge_index, batch.edge_label_index)
                labels = batch.edge_label.float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("training loss is not finite")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                optimizer.step()
                total_loss += float(loss.detach()) * int(labels.numel())
                examples += int(labels.numel())
        history.append({"epoch": float(epoch), "loss": total_loss / max(examples, 1)})

        should_evaluate = epoch % eval_every == 0 or epoch == max_epochs
        if not should_evaluate:
            continue
        validation, _, _ = _evaluate_learning(
            model,
            model_name=spec.model,
            x=x,
            stage=protocol.validation,
            score_batch_size=score_batch_size,
            inference_batch_size=inference_batch_size,
            evaluator=evaluator,
        )
        hits50 = float(validation["hits@50"])
        history[-1]["validation_hits@50"] = hits50
        improved = hits50 > best.hits50
        if improved:
            best.epoch = epoch
            best.hits50 = hits50
            best.model_state = _parameter_state(model, spec.model)
            evaluations_without_improvement = 0
        else:
            evaluations_without_improvement += 1
        early_stop = epoch >= min_epochs and evaluations_without_improvement >= patience
        terminal = epoch >= max_epochs or early_stop
        if checkpoint_sink is not None:
            payload = _checkpoint_payload(
                model=model,
                model_name=spec.model,
                optimizer=optimizer,
                epoch=epoch,
                best=best,
                sampler=sampler,
                selection_rng=rng,
                selected_batch_size=selected_batch_size,
                evaluations_without_improvement=evaluations_without_improvement,
                history=history,
                terminal=terminal,
            )
            if improved:
                checkpoint_sink("best", payload)
            checkpoint_sink("latest", payload)
        if early_stop:
            break

    if best.model_state is None:
        raise RuntimeError("training finished without a validation checkpoint")
    _restore_best(model, spec.model, best.model_state)
    validation, validation_pos, validation_neg = _evaluate_learning(
        model,
        model_name=spec.model,
        x=x,
        stage=protocol.validation,
        score_batch_size=score_batch_size,
        inference_batch_size=inference_batch_size,
        evaluator=evaluator,
    )
    test = None
    strata_stage = protocol.validation
    strata_positive, strata_negative = validation_pos, validation_neg
    if spec.phase == "formal":
        # The test stage is first dereferenced here, after best checkpoint restore.
        test, strata_positive, strata_negative = _evaluate_learning(
            model,
            model_name=spec.model,
            x=x,
            stage=protocol.test,
            score_batch_size=score_batch_size,
            inference_batch_size=inference_batch_size,
            evaluator=evaluator,
        )
        strata_stage = protocol.test
    strata = (
        stratified_positive_metrics(
            strata_positive, strata_negative, strata_stage.repeated_mask
        )
        if protocol.track == "strict_edge_time"
        else {}
    )
    peak = probe_peak
    if device == "cuda":
        peak = max(
            peak, float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
        )
        if peak >= memory_limit:
            raise RuntimeError(
                f"run exceeded CUDA memory limit: {peak:.1f} MiB >= {memory_limit:.1f} MiB"
            )
    return CoreRunResult(
        spec=spec,
        validation_metrics=validation,
        test_metrics=test,
        strata=strata,
        best_epoch=best.epoch,
        peak_cuda_memory_mib=peak,
        selected_batch_size=selected_batch_size,
        test_read_after_selection=spec.phase == "formal",
        history=tuple(history),
    )


def _link_loader(
    features: Any,
    message_edges: Any,
    label_edges: Any,
    labels: Any,
    *,
    batch_size: int,
    fanout: tuple[int, ...],
    shuffle: bool,
):
    """Build a PyG sampled loader with caller-supplied exact labels/negatives."""

    import torch
    from torch_geometric.data import Data
    from torch_geometric.loader import LinkNeighborLoader

    try:
        import pyg_lib  # noqa: F401
    except ImportError as error:
        raise RuntimeError("GraphSAGE LinkNeighborLoader requires pyg_lib") from error

    data = Data(
        x=torch.as_tensor(features, dtype=torch.float32),
        edge_index=_edge_tensor(message_edges),
    )
    return LinkNeighborLoader(
        data,
        num_neighbors=list(fanout),
        edge_label_index=_edge_tensor(label_edges),
        edge_label=torch.as_tensor(labels, dtype=torch.float32),
        neg_sampling=None,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
