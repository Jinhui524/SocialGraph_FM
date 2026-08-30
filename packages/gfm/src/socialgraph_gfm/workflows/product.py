"""Product-head preparation, scoring, baselines, and adaptation training.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def _selected_core_variant(
    layout: RuntimeLayout, experiment_id: str
) -> Literal["core-base", "core-moe"]:
    """Apply the fixed no-negative-transfer promotion rule."""

    from ..gfm.transfer_workflow import select_core_variant

    reports = [
        report
        for report in _registry(layout).list_evaluations(experiment_id=experiment_id)
        if report.evaluation_kind == "lodo"
    ]
    runs = {run.run_id: run for run in _require_experiment_runs(layout, experiment_id)}
    by_variant: dict[str, defaultdict[str, list[float]]] = {
        "core-base": defaultdict(list),
        "core-moe": defaultdict(list),
    }
    for report in reports:
        run = runs.get(report.run_id)
        if run is None or report.held_out_domain is None:
            continue
        metric = report.metrics.get("few_shot_5_gfm")
        if metric is not None:
            by_variant[run.architecture_variant][report.held_out_domain].append(float(metric))
    reduced: dict[str, dict[str, float]] = {}
    for variant, domains in by_variant.items():
        if set(domains) != set(DOMAIN_IDS.values()):
            raise ContractViolation(
                "Core variant selection requires complete base/MoE LODO evidence"
            )
        reduced[variant] = {domain: float(np.mean(values)) for domain, values in domains.items()}
    return select_core_variant(
        base_by_domain=reduced["core-base"],
        moe_by_domain=reduced["core-moe"],
    ).selected


def _formal_backbones(
    layout: RuntimeLayout,
    *,
    experiment_id: str,
    variant: Literal["core-base", "core-moe"],
) -> tuple[Any, ...]:
    from ..gfm.transfer_workflow import select_formal_checkpoints

    registry = _registry(layout)
    runs = registry.list_runs(experiment_id=experiment_id)
    config = _load_pretrain_config(None, None)
    formal_runs = [
        run
        for run in runs
        if run.phase == "pretrain"
        and run.status == "succeeded"
        and run.architecture_variant == variant
    ]
    if not formal_runs:
        raise ContractViolation("No formal pretraining runs exist for the selected variant")
    code_hashes = {run.code_hash for run in formal_runs}
    environment_hashes = {run.environment_hash for run in formal_runs}
    corpus_sets = {tuple(sorted(run.corpus_hashes)) for run in formal_runs}
    if len(code_hashes) != 1 or len(environment_hashes) != 1 or len(corpus_sets) != 1:
        raise ContractViolation("Formal pretraining provenance differs across seeds")
    checkpoints = {
        checkpoint.run_id: checkpoint
        for checkpoint in registry.list_checkpoints(experiment_id=experiment_id)
    }
    fresh = {
        report.checkpoint_id: report
        for report in registry.list_evaluations(experiment_id=experiment_id)
        if report.evaluation_kind == "fresh_process"
        and report.fresh_process_verified
        and report.metrics.get("fresh_process_repeat_match") == 1.0
    }
    records: list[dict[str, object]] = []
    for run in runs:
        checkpoint = checkpoints.get(run.run_id)
        if (
            run.phase != "pretrain"
            or run.status != "succeeded"
            or run.architecture_variant != variant
            or checkpoint is None
            or checkpoint.checkpoint_id not in fresh
        ):
            continue
        records.append(
            {
                "variant": run.architecture_variant,
                "seed": run.seed,
                "checkpointId": checkpoint.checkpoint_id,
                "freshProcessDigest": fresh[checkpoint.checkpoint_id].verification_digest,
                "freshProcessVerified": True,
                "checkpointRole": "best",
                "phase": "formal",
                "configHash": run.config_hash,
                "codeHash": run.code_hash,
                "environmentHash": run.environment_hash,
                "corpusHashes": run.corpus_hashes,
            }
        )
    selected = select_formal_checkpoints(
        records,
        selected_variant=variant,
        expected_config_hash=config.config_hash,
        expected_code_hash=next(iter(code_hashes)),
        expected_environment_hash=next(iter(environment_hashes)),
        expected_corpus_hashes=next(iter(corpus_sets)),
    )
    result = tuple(registry.get_checkpoint(checkpoint_id) for checkpoint_id in selected)
    if any(value is None for value in result):
        raise ContractViolation("Selected formal backbone is absent from the registry")
    return result


def _product_batches_for_split(
    *,
    task: ProductTask,
    stream: _DomainStream,
    arrays: Mapping[str, np.ndarray],
    newcomers: Mapping[str, np.ndarray] | None,
    split: Literal["train", "validation", "test", "shadow"],
    seed: int,
    transform: _FeatureTransform | None,
    collaboration_kind: Literal["first", "repeat", "both"] = "both",
) -> Iterator[_PreparedProductBatch]:
    if task == "collaboration":
        years = {
            "train": (2017, 2018, 2019, 2020, 2021),
            "validation": (2022,),
            "test": (2023,),
            "shadow": (2024,),
        }[split]
        kinds: tuple[Literal["first", "repeat"], ...]
        if collaboration_kind == "both":
            kinds = ("first", "repeat")
        else:
            kinds = (collaboration_kind,)
        query_offset = 0
        for kind in kinds:
            prepared = _collaboration_batches(
                stream,
                arrays,
                cutoff_years=years,
                seed=seed + (0 if kind == "first" else 1_000_000),
                transform=transform,
                target_kind=kind,
            )
            for item in prepared:
                shifted = item.with_query_offset(query_offset)
                if shifted.batch.query_ids.numel():
                    query_offset = int(shifted.batch.query_ids.max()) + 1
                yield shifted
        return
    if newcomers is None:
        raise ContractViolation("Newcomer product split requires verified cohorts")
    years = {
        "train": (2017, 2018, 2019, 2020),
        "validation": (2021,),
        "test": (2022,),
        "shadow": (2023, 2024),
    }[split]
    yield from _newcomer_batches(
        stream,
        arrays,
        newcomers,
        cohort_years=years,
        seed=seed,
        transform=transform,
    )


def _product_logits(
    model: Any,
    batches: Iterable[_PreparedProductBatch],
    *,
    device: str,
    baseline_config: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    import torch

    pair_logits: list[np.ndarray] = []
    pair_labels: list[np.ndarray] = []
    query_ids: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    raw_features: list[np.ndarray] = []
    participation_logits: list[np.ndarray] = []
    participation_labels: list[np.ndarray] = []
    participation_baselines: list[np.ndarray] = []
    query_offset = 0
    model.to(device).eval()
    with torch.inference_mode():
        for prepared in batches:
            batch = prepared.batch.to(device)
            pair, participation = model(batch)
            pair_value = pair.float().cpu().numpy()
            label_value = batch.pair_labels.float().cpu().numpy()
            query_value = batch.query_ids.cpu().numpy()
            if query_value.size:
                query_value = query_value + query_offset
                query_offset = int(query_value.max()) + 1
                pair_logits.append(pair_value)
                pair_labels.append(label_value)
                query_ids.append(query_value)
                raw_features.append(prepared.raw_features)
                baselines.append(
                    _product_baseline_scores(
                        prepared.raw_features,
                        task=prepared.batch.provenance.task_id,
                        baseline_config=baseline_config,
                        fallback=prepared.baseline_scores,
                    )
                )
            if participation is not None:
                if batch.participation_labels is None:
                    raise GfmTrainingError("Participation logits lack labels")
                participation_logits.append(participation.float().cpu().numpy())
                participation_labels.append(batch.participation_labels.float().cpu().numpy())
                if prepared.participation_baseline is None:
                    raise GfmTrainingError("Participation baseline is absent")
                participation_baselines.append(prepared.participation_baseline)

    def joined(values: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
        if not values:
            return np.empty(0, dtype=dtype)
        return np.concatenate(values).astype(dtype, copy=False)

    return {
        "pair_logits": joined(pair_logits, np.float32),
        "pair_labels": joined(pair_labels, np.float32),
        "query_ids": joined(query_ids, np.int64),
        "baseline_scores": joined(baselines, np.float32),
        "raw_features": joined(raw_features, np.float32).reshape(-1, 8),
        "participation_logits": joined(participation_logits, np.float32),
        "participation_labels": joined(participation_labels, np.float32),
        "participation_baselines": joined(participation_baselines, np.float32),
    }


def _mean_ndcg_at_20(scores: np.ndarray, labels: np.ndarray, query_ids: np.ndarray) -> float:
    values: list[float] = []
    for query in np.unique(query_ids):
        mask = query_ids == query
        target = labels[mask]
        order = np.argsort(-scores[mask], kind="stable")[:20]
        gains = target[order]
        discounts = 1.0 / np.log2(np.arange(gains.size, dtype=np.float64) + 2.0)
        dcg = float(np.sum(gains * discounts))
        ideal = np.sort(target)[::-1][:20]
        idcg = float(np.sum(ideal * discounts[: ideal.size]))
        values.append(0.0 if idcg <= 0.0 else dcg / idcg)
    if not values:
        raise GfmTrainingError("Baseline selection has no ranking queries")
    return float(np.mean(values))


def _ranking_stratum_metrics(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    groups: np.ndarray,
    axis: str,
    group_names: Mapping[int, str],
) -> dict[str, float]:
    """Report frozen-model ranking performance and query counts by one axis."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.float64).reshape(-1)
    queries = np.asarray(query_ids, dtype=np.int64).reshape(-1)
    strata = np.asarray(groups, dtype=np.int64).reshape(-1)
    if not (values.shape == targets.shape == queries.shape == strata.shape):
        raise GfmTrainingError("Ranking stratum arrays are misaligned")
    result: dict[str, float] = {}
    for group, name in sorted(group_names.items()):
        selected_queries: list[int] = []
        ndcg: list[float] = []
        recall: list[float] = []
        for query in np.unique(queries):
            mask = queries == query
            query_groups = np.unique(strata[mask])
            if query_groups.size != 1:
                raise GfmTrainingError("A ranking query crosses a declared stratum")
            if int(query_groups[0]) != group:
                continue
            selected_queries.append(int(query))
            query_scores = values[mask]
            query_labels = targets[mask]
            order = np.argsort(-query_scores, kind="stable")[:20]
            gains = query_labels[order]
            discounts = 1.0 / np.log2(np.arange(gains.size) + 2.0)
            ideal = np.sort(query_labels)[::-1][:20]
            denominator = float(np.sum(ideal * discounts[: ideal.size]))
            ndcg.append(
                0.0 if denominator <= 0.0 else float(np.sum(gains * discounts) / denominator)
            )
            positives = float(query_labels.sum())
            recall.append(0.0 if positives <= 0.0 else float(gains.sum() / positives))
        prefix = f"{axis}_{name}"
        result[f"{prefix}_query_count"] = float(len(selected_queries))
        if selected_queries:
            result[f"{prefix}_ndcg@20"] = float(np.mean(ndcg))
            result[f"{prefix}_recall@20"] = float(np.mean(recall))
    return result


def _product_candidate_groups(
    batches: Iterable[_PreparedProductBatch],
    *,
    name: Literal["institution", "topic"],
    collaboration_kind: Literal["first", "repeat", "both"] = "both",
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for item in batches:
        if collaboration_kind != "both" and item.collaboration_kind != collaboration_kind:
            continue
        count = int(item.batch.pair_labels.numel())
        if count == 0:
            continue
        value = item.institution_group if name == "institution" else item.topic_group
        if value is None:
            raise GfmTrainingError(f"Product batch lacks {name} group evidence")
        group = np.asarray(value, dtype=np.int64).reshape(-1)
        if group.size == 1:
            group = np.full(count, int(group[0]), dtype=np.int64)
        if group.shape != (count,):
            raise GfmTrainingError(f"Product {name} groups do not align to candidates")
        rows.append(group)
    if not rows:
        raise GfmTrainingError(f"Product {name} groups are empty")
    return np.concatenate(rows)


def _prepared_pair_arrays(
    batches: Iterable[_PreparedProductBatch],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    queries: list[np.ndarray] = []
    query_offset = 0
    for prepared in batches:
        if not prepared.batch.query_ids.numel():
            continue
        query = prepared.batch.query_ids.detach().cpu().numpy().astype(np.int64)
        query = query + query_offset
        query_offset = int(query.max()) + 1
        raw.append(prepared.raw_features.astype(np.float32, copy=False))
        labels.append(prepared.batch.pair_labels.detach().cpu().numpy().astype(np.float32))
        queries.append(query)
    if not raw:
        raise GfmTrainingError("Product baseline has no pair-ranking samples")
    return np.concatenate(raw), np.concatenate(labels), np.concatenate(queries)


def _fit_cutoff_feature_mlp(
    features: np.ndarray, labels: np.ndarray, *, seed: int
) -> dict[str, Any]:
    """Fit the frozen train-only feature MLP used as a strong baseline."""

    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[1] != 8 or matrix.shape[0] != target.size:
        raise GfmTrainingError("Feature baseline training arrays are misaligned")
    if set(np.unique(target).tolist()) != {0.0, 1.0}:
        raise GfmTrainingError("Feature baseline requires both outcome classes")
    rng = np.random.default_rng(seed)
    hidden = 16
    w1 = rng.normal(0.0, 0.05, size=(8, hidden))
    b1 = np.zeros(hidden, dtype=np.float64)
    w2 = rng.normal(0.0, 0.05, size=hidden)
    b2 = 0.0
    positive_weight = float((target == 0).sum() / max(1, (target == 1).sum()))
    batch_size = min(4096, target.size)
    for step in range(500):
        indices = rng.choice(target.size, size=batch_size, replace=False)
        x, y = matrix[indices], target[indices]
        hidden_pre = x @ w1 + b1
        hidden_value = np.maximum(hidden_pre, 0.0)
        logits = np.clip(hidden_value @ w2 + b2, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        weights = np.where(y > 0.5, positive_weight, 1.0)
        derivative = (probability - y) * weights / float(weights.sum())
        grad_w2 = hidden_value.T @ derivative + 1e-4 * w2
        grad_b2 = float(derivative.sum())
        hidden_grad = np.outer(derivative, w2) * (hidden_pre > 0.0)
        grad_w1 = x.T @ hidden_grad + 1e-4 * w1
        grad_b1 = hidden_grad.sum(axis=0)
        learning_rate = 0.03 * 0.5 * (1.0 + math.cos(math.pi * step / 500.0))
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
    parameters = {
        "inputToHidden": w1.tolist(),
        "hiddenBias": b1.tolist(),
        "hiddenToOutput": w2.tolist(),
        "outputBias": b2,
        "hiddenChannels": hidden,
        "optimizerSteps": 500,
        "fitSplit": "train-only",
    }
    parameters["parameterHash"] = canonical_sha256(parameters)
    return parameters


def _feature_mlp_scores(features: np.ndarray, parameters: Mapping[str, Any]) -> np.ndarray:
    expected = dict(parameters)
    parameter_hash = expected.pop("parameterHash", None)
    if parameter_hash != canonical_sha256(expected) or expected.get("fitSplit") != "train-only":
        raise ContractViolation("Feature MLP parameters lack train-only hash provenance")
    matrix = np.asarray(features, dtype=np.float64)
    hidden = np.maximum(
        matrix @ np.asarray(expected["inputToHidden"], dtype=np.float64)
        + np.asarray(expected["hiddenBias"], dtype=np.float64),
        0.0,
    )
    logits = np.clip(
        hidden @ np.asarray(expected["hiddenToOutput"], dtype=np.float64)
        + float(expected["outputBias"]),
        -30.0,
        30.0,
    )
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


def _select_collaboration_baseline(
    train: Iterable[_PreparedProductBatch],
    validation: Iterable[_PreparedProductBatch],
    *,
    transform: _FeatureTransform,
    seed: int,
) -> dict[str, Any]:
    train_raw, train_labels, _ = _prepared_pair_arrays(train)
    validation_raw, validation_labels, validation_queries = _prepared_pair_arrays(validation)
    parameters = _fit_cutoff_feature_mlp(transform.apply(train_raw), train_labels, seed=seed)
    candidates = {
        "adamic-adar": validation_raw[:, 1],
        "resource-allocation": validation_raw[:, 2],
        "cutoff-feature-mlp": _feature_mlp_scores(transform.apply(validation_raw), parameters),
    }
    metrics = {
        name: _mean_ndcg_at_20(scores, validation_labels, validation_queries)
        for name, scores in candidates.items()
    }
    selected = max(metrics, key=lambda name: (metrics[name], name))
    return {
        "schemaVersion": "gfm.collaboration-baseline/1.0",
        "selected": selected,
        "selectionMetric": "validation-ndcg@20",
        "validationNdcgAt20": metrics,
        "featureMlp": parameters,
        "featureTransformHash": canonical_sha256(transform.as_dict()),
        "fitSplit": "train-only",
        "selectionSplit": "validation-only",
    }


def _product_baseline_scores(
    raw_features: np.ndarray,
    *,
    task: str,
    baseline_config: Mapping[str, Any] | None,
    fallback: np.ndarray,
) -> np.ndarray:
    if task != "collaboration" or baseline_config is None:
        return np.asarray(fallback, dtype=np.float32)
    selected = baseline_config.get("selected")
    if selected == "adamic-adar":
        return np.asarray(raw_features[:, 1], dtype=np.float32)
    if selected == "resource-allocation":
        return np.asarray(raw_features[:, 2], dtype=np.float32)
    if selected == "cutoff-feature-mlp":
        feature_transform = baseline_config.get("featureTransform")
        if not isinstance(feature_transform, dict):
            raise ContractViolation("Feature baseline lacks its immutable transform")
        return _feature_mlp_scores(
            _FeatureTransform.from_dict(feature_transform).apply(raw_features),
            baseline_config["featureMlp"],
        )
    raise ContractViolation("Collaboration baseline selection is invalid")


def _collaboration_rerank_components(
    calibrated_probability: np.ndarray, raw_features: np.ndarray
) -> dict[str, np.ndarray]:
    probability = np.asarray(calibrated_probability, dtype=np.float64)
    raw = np.asarray(raw_features, dtype=np.float64)
    if raw.shape != (probability.size, 8):
        raise GfmTrainingError("Collaboration rerank features are misaligned")
    components = {
        "calibratedProbability": np.clip(probability, 0.0, 1.0),
        "topicComplementarity": np.clip(raw[:, 4], 0.0, 1.0),
        # A low-overlap edge is the frozen, transparent topological bridge
        # proxy.  It is presented as a component, never as a learned fact.
        "bridgeGain": 1.0 / (1.0 + np.maximum(raw[:, 0], 0.0)),
        "institutionDiversity": np.clip(raw[:, 5], 0.0, 1.0),
    }
    components["finalRerank"] = (
        0.70 * components["calibratedProbability"]
        + 0.15 * components["topicComplementarity"]
        + 0.10 * components["bridgeGain"]
        + 0.05 * components["institutionDiversity"]
    )
    return {name: value.astype(np.float32) for name, value in components.items()}


def _product_batch_contract(task: ProductTask) -> dict[str, Any]:
    """Return the fixed, hash-bound product materialization contract."""

    return {
        "schemaVersion": "gfm.product-batch-contract/1.1",
        "task": task,
        "pairFeatureChannels": 8,
        "collaborationQueriesPerMicrobatch": 8,
        "rankingNegativesPerQuery": 99,
        "candidateMinimumPerRankingQuery": 100,
        "newcomerQueriesPerMicrobatch": 1,
        "streamingRebuildableFactory": True,
        "resumeCursor": {
            "schemaVersion": "gfm.product-train-cursor/1.0",
            "semantics": "next-batch-within-deterministic-epoch",
            "fields": ["epoch", "batchOffset"],
            "replayGlobalRng": False,
        },
        "futurePositiveExclusion": "all-positives-in-declared-horizon",
    }


def _product_task_asset_evidence(
    layout: RuntimeLayout,
    *,
    task: ProductTask,
    corpora: Sequence[GfmDomainCorpusManifest],
) -> dict[str, Any]:
    """Bind task-only labels without mutating the three base corpus identities."""

    base_hashes = {
        corpus.domain_id: corpus.logical_hash
        for corpus in sorted(corpora, key=lambda item: item.domain_id)
    }
    if set(base_hashes) != set(DOMAIN_IDS.values()):
        raise ContractViolation("Product task assets require all three base corpora")
    payload: dict[str, Any] = {
        "schemaVersion": "gfm.product-task-assets/1.0",
        "task": task,
        "baseCorpusHashes": base_hashes,
        "newcomerOverlay": None,
    }
    if task == "newcomer":
        overlay = check_openalex_newcomers(layout.root)
        source = overlay.get("source")
        logical_hash = overlay.get("logicalHash")
        if (
            not isinstance(source, dict)
            or not isinstance(logical_hash, str)
            or len(logical_hash) != 64
            or source.get("baseCorpusId") != DOMAIN_IDS["openalex"]
            or not isinstance(source.get("baseCorpusLogicalHash"), str)
            or not isinstance(source.get("baseCorpusSourceHash"), str)
            or overlay.get("verifiedCount") != overlay.get("authorCount")
        ):
            raise ContractViolation(
                "Newcomer adaptation requires a complete overlay bound to the OpenAlex base corpus"
            )
        payload["newcomerOverlay"] = {
            "corpusId": source["baseCorpusId"],
            "baseCorpusLogicalHash": source["baseCorpusLogicalHash"],
            "baseCorpusSourceHash": source["baseCorpusSourceHash"],
            "overlayLogicalHash": logical_hash,
            "verifiedCount": int(overlay["verifiedCount"]),
            "historyQueryProtocol": overlay.get("historyQueryProtocol"),
        }
    return payload


def _product_checkpoint_rng_state(value: Any) -> dict[str, Any]:
    """Convert ProductResumeState RNG values to the safe checkpoint schema."""

    numpy_state = value.numpy_rng_state
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise GfmTrainingError("Product resume NumPy RNG state is malformed")
    algorithm, keys, position, has_gauss, cached_gaussian = numpy_state
    key_array = np.asarray(keys, dtype=np.uint32)
    return {
        "python": value.python_rng_state,
        "torch_cpu": value.torch_rng_state,
        "torch_cuda": list(value.cuda_rng_states),
        "numpy": {
            "algorithm": str(algorithm),
            "keys": [int(item) for item in key_array.tolist()],
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
    }


def _product_resume_from_payload(payload: Mapping[str, Any]) -> Any:
    """Reconstruct a ProductResumeState only from an integrity-checked payload."""

    import torch

    from ..gfm.product_training import ProductResumeState

    components = payload.get("components")
    sampler = payload.get("sampler_state")
    best = payload.get("best_state")
    rng = payload.get("rng_state")
    if (
        not isinstance(components, dict)
        or set(components) != {"product", "product_config"}
        or not isinstance(sampler, dict)
        or not isinstance(best, dict)
        or not isinstance(rng, dict)
    ):
        raise ContractViolation("Product progress checkpoint structure is invalid")
    numpy_state = rng.get("numpy")
    if not isinstance(numpy_state, dict):
        raise ContractViolation("Product progress checkpoint lacks NumPy RNG state")
    history = best.get("history")
    best_model = best.get("bestProductState")
    train_cursor = sampler.get("trainCursor")
    train_iterator_contract_hash = sampler.get("trainIteratorContractHash")
    torch_cpu = rng.get("torch_cpu")
    torch_cuda = rng.get("torch_cuda")
    if (
        not isinstance(history, (tuple, list))
        or not isinstance(best_model, dict)
        or not torch.is_tensor(torch_cpu)
        or not isinstance(torch_cuda, (tuple, list))
        or any(not torch.is_tensor(value) for value in torch_cuda)
        or not isinstance(train_cursor, dict)
        or set(train_cursor) != {"schemaVersion", "epoch", "batchOffset"}
        or train_cursor.get("schemaVersion") != "gfm.product-train-cursor/1.0"
        or isinstance(train_cursor.get("epoch"), bool)
        or not isinstance(train_cursor.get("epoch"), int)
        or int(train_cursor["epoch"]) < 0
        or isinstance(train_cursor.get("batchOffset"), bool)
        or not isinstance(train_cursor.get("batchOffset"), int)
        or int(train_cursor["batchOffset"]) < 0
        or not isinstance(train_iterator_contract_hash, str)
        or len(train_iterator_contract_hash) != 64
        or any(character not in "0123456789abcdef" for character in train_iterator_contract_hash)
    ):
        raise ContractViolation("Product progress checkpoint resume values are invalid")
    return ProductResumeState(
        completed_steps=int(sampler["completedSteps"]),
        train_epoch=int(train_cursor["epoch"]),
        train_batch_offset=int(train_cursor["batchOffset"]),
        train_iterator_contract_hash=train_iterator_contract_hash,
        latest_model_state=components["product"],
        optimizer_state=payload["optimizer_state"],
        scheduler_state=payload["scheduler_state"],
        scaler_state=payload["scaler_state"] or {},
        best_step=int(best["bestStep"]),
        best_validation_loss=float(best["validationLoss"]),
        best_state=best_model,
        no_improvement_evaluations=int(sampler["noImprovementEvaluations"]),
        history=tuple(dict(item) for item in history),
        python_rng_state=tuple(rng["python"]),
        numpy_rng_state=(
            str(numpy_state["algorithm"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        ),
        torch_rng_state=torch_cpu,
        cuda_rng_states=tuple(torch_cuda),
    )


def _remove_checkpoint_identity(directory: Path, manifest_path: Path) -> None:
    identity = manifest_path.stem.removesuffix(".manifest")
    try:
        manifest = read_gfm_checkpoint_manifest(manifest_path)
        artifact = Path(manifest.artifact_path).resolve()
    except Exception:
        # A corrupt unselected progress manifest cannot be trusted to name its
        # artifact.  The checkpoint writer's fixed identity path is the only
        # safe cleanup target inside this already-resolved run directory.
        artifact = (directory / f"{identity}.pt").resolve()
    try:
        artifact.relative_to(directory.resolve())
        manifest_path.resolve().relative_to(directory.resolve())
    except ValueError as error:
        raise GfmTrainingError("Product checkpoint cleanup escaped its run") from error
    artifact.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)


def _normalize_product_progress_roles(
    directory: Path, *, run_id: str, selected_checkpoint_id: str
) -> None:
    """Remove only orphaned latest files after the durable commit marker is read."""

    for path in directory.glob(f"{run_id}-latest-*.manifest.json"):
        identity = path.stem.removesuffix(".manifest")
        if identity != selected_checkpoint_id:
            _remove_checkpoint_identity(directory, path)


def _run_product_adaptation(
    *,
    layout: RuntimeLayout,
    task: ProductTask,
    experiment_id: str,
    backbone_checkpoint_id: str,
    device: str,
    maximum_steps: int = 5_000,
    minimum_steps: int = 1_000,
    evaluation_every_steps: int = 250,
    resume_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Train one real product head without materializing test labels."""

    import torch

    from ..gfm.calibration import fit_temperature
    from ..gfm.model import SocialGraphFMCore
    from ..gfm.product_training import (
        ProductTaskModule,
        ProductTrainingConfig,
        load_product_backbone,
        train_product_steps,
    )

    if maximum_steps < 1 or minimum_steps > maximum_steps or evaluation_every_steps < 1:
        raise ValueError("Invalid product step contract")
    registry = _registry(layout)
    backbone = registry.get_checkpoint(backbone_checkpoint_id)
    if backbone is None:
        raise ContractViolation("Product backbone checkpoint is not registered")
    backbone_run = registry.get_run(backbone.run_id)
    if (
        backbone_run is None
        or backbone_run.phase != "pretrain"
        or backbone_run.status != "succeeded"
    ):
        raise ContractViolation("Product backbone is not a succeeded pretraining checkpoint")
    seed = backbone_run.seed
    variant = backbone_run.architecture_variant
    set_seed(seed, device)
    config = _load_pretrain_config(None, None)
    if seed not in config.formal.seeds:
        raise ContractViolation("Product adaptation requires a fixed formal seed")
    corpora, embeddings = _ensure_pretrain_evidence(
        layout,
        maximum_role="validation",
        physical_boundary=True,
    )
    protocols = _register_prerequisites(layout, corpora)
    task_id = COLLABORATION_TASK if task == "collaboration" else NEWCOMER_TASK
    protocol = next(value for value in protocols if value.task_id == task_id)
    corpus_hashes = tuple(corpus.logical_hash for corpus in corpora)
    task_assets = _product_task_asset_evidence(layout, task=task, corpora=corpora)
    task_assets_hash = canonical_sha256(task_assets)
    current_code_hash = code_identity_hash()
    current_environment_hash = _environment_hash(device)
    if (
        backbone_run.code_hash != current_code_hash
        or backbone_run.environment_hash != current_environment_hash
        or backbone_run.config_hash != config.config_hash
        or set(backbone_run.corpus_hashes) != set(corpus_hashes)
    ):
        raise ContractViolation(
            "Product adaptation runtime differs from its formal backbone provenance"
        )
    run_id = f"{experiment_id}-adapt-{task}-{variant}-{seed}"
    run_dir = layout.gfm_runs / experiment_id / run_id
    checkpoint_dir = run_dir / "checkpoints"
    if registry.get_run(run_id) is not None:
        raise GfmTrainingError("Terminal product run cannot be resumed or overwritten")
    if resume_manifest is None:
        if run_dir.exists():
            raise GfmTrainingError(
                "Product run directory already exists; use gfm-resume to recover it"
            )
        # Do not claim the immutable run identity yet.  Stream construction,
        # feature fitting and backbone verification below are read-only and may
        # fail; the first filesystem claim is the atomic durable-state write.
        started = datetime.now(UTC)
    else:
        if not run_dir.is_dir():
            raise GfmTrainingError("Product resume run directory is absent")
        started = datetime.fromisoformat(
            str(read_json_object(run_dir / "run-state.json")["startedAt"])
        )
    streams = _make_domain_streams(layout, embeddings, maximum_role="validation")
    stream = streams[DOMAIN_IDS["openalex"]]
    arrays = load_domain_view(
        layout.root,
        DOMAIN_IDS["openalex"],
        maximum_role="validation",
        families=("targets",),
    )["arrays"]
    newcomers = (
        load_openalex_newcomers_view(layout.root, maximum_role="validation")["arrays"]
        if task == "newcomer"
        else None
    )

    def prepared(
        split: Literal["train", "validation"],
        transform_value: _FeatureTransform | None,
    ) -> Iterator[_PreparedProductBatch]:
        return _product_batches_for_split(
            task=task,
            stream=stream,
            arrays=arrays,
            newcomers=newcomers,
            split=split,
            seed=seed,
            transform=transform_value,
            collaboration_kind="first" if task == "collaboration" else "both",
        )

    transform = _FeatureTransform.fit(
        item.raw_features for item in prepared("train", None) if item.raw_features.size
    )
    collaboration_baseline = None
    if task == "collaboration":
        collaboration_baseline = _select_collaboration_baseline(
            prepared("train", None),
            prepared("validation", transform),
            transform=transform,
            seed=seed,
        )
        collaboration_baseline["featureTransform"] = transform.as_dict()
    backbone_payload = load_gfm_checkpoint(backbone, map_location="cpu")
    embedding_artifacts = _embedding_artifact_evidence(embeddings)
    embedding_artifacts_hash = canonical_sha256(embedding_artifacts)
    if (
        backbone_payload.get("sampler_state", {}).get("embeddingArtifacts") != embedding_artifacts
        or backbone_payload.get("sampler_state", {}).get("embeddingArtifactsHash")
        != embedding_artifacts_hash
    ):
        raise ContractViolation(
            "Product backbone is not bound to the current formal embedding artifacts"
        )
    batch_contract = _product_batch_contract(task)
    batch_contract_hash = canonical_sha256(batch_contract)
    progress_config = {
        "schemaVersion": "gfm.product-resume-config/1.1",
        "task": task,
        "taskId": task_id,
        "architectureVariant": variant,
        "seed": seed,
        "backboneCheckpointId": backbone.checkpoint_id,
        "backboneStateHash": backbone.state_hash,
        "embeddingArtifacts": embedding_artifacts,
        "embeddingArtifactsHash": embedding_artifacts_hash,
        "taskAssets": task_assets,
        "taskAssetsHash": task_assets_hash,
        "featureTransform": transform.as_dict(),
        "collaborationBaseline": collaboration_baseline,
        "collaborationRerank": {
            "weights": dict(config.product.collaboration_rerank_weights),
            "bridgeGainDefinition": "inverse-one-plus-cutoff-common-neighbors",
            "componentPresentationRequired": True,
        },
        "protocolHash": protocol.protocol_hash,
        "optimizer": {
            "name": "adamw",
            "learningRate": config.optimization.learning_rate,
            "weightDecay": config.optimization.weight_decay,
            "warmupRatio": config.optimization.warmup_ratio,
            "schedule": "cosine",
        },
        "steps": {
            "maximum": maximum_steps,
            "minimum": minimum_steps,
            "evaluationEvery": evaluation_every_steps,
            "patienceEvaluations": 6,
        },
        "batchContract": batch_contract,
        "batchContractHash": batch_contract_hash,
        "primaryTargetKind": "first-collaboration" if task == "collaboration" else None,
        "repeatCollaborationRole": (
            "frozen-model-auxiliary-evaluation-only" if task == "collaboration" else None
        ),
        "testRead": False,
    }
    progress_config_hash = canonical_sha256(progress_config)
    progress_config["taskConfigHash"] = progress_config_hash
    durable_state = {
        "schemaVersion": "gfm.product-run-state/1.1",
        "runKind": "product-adapt",
        "runId": run_id,
        "experimentId": experiment_id,
        "task": task,
        "variant": variant,
        "seed": seed,
        "device": device,
        "configHash": config.config_hash,
        "codeHash": current_code_hash,
        "environmentHash": current_environment_hash,
        "corpusHashes": list(corpus_hashes),
        "embeddingArtifactsHash": embedding_artifacts_hash,
        "taskAssetsHash": task_assets_hash,
        "backboneCheckpointId": backbone.checkpoint_id,
        "backboneStateHash": backbone.state_hash,
        "protocolHash": protocol.protocol_hash,
        "productProgressConfigHash": progress_config_hash,
        "batchContractHash": batch_contract_hash,
        "startedAt": started.isoformat(),
        "status": "running",
        "completedSteps": 0,
        "trainCursor": {
            "schemaVersion": "gfm.product-train-cursor/1.0",
            "epoch": 0,
            "batchOffset": 0,
        },
        "trainIteratorContractHash": progress_config_hash,
        "latestCheckpointId": None,
        "recoveryCheckpointId": None,
    }
    resume_value = None
    if resume_manifest is None:
        _save_run_state(run_dir / "run-state.json", durable_state)
    else:
        stored_state = read_json_object(run_dir / "run-state.json")
        immutable_keys = (
            "schemaVersion",
            "runKind",
            "runId",
            "experimentId",
            "task",
            "variant",
            "seed",
            "device",
            "configHash",
            "codeHash",
            "environmentHash",
            "corpusHashes",
            "embeddingArtifactsHash",
            "taskAssetsHash",
            "backboneCheckpointId",
            "backboneStateHash",
            "protocolHash",
            "productProgressConfigHash",
            "batchContractHash",
            "startedAt",
        )
        if stored_state.get("status") != "running" or any(
            stored_state.get(key) != durable_state.get(key) for key in immutable_keys
        ):
            raise GfmTrainingError(
                "Interrupted product run provenance differs from its current runtime"
            )
        progress_checkpoint = read_gfm_checkpoint_manifest(resume_manifest)
        if (
            progress_checkpoint.run_id != run_id
            or progress_checkpoint.config_hash != config.config_hash
            or tuple(progress_checkpoint.corpus_hashes) != corpus_hashes
            or progress_checkpoint.checkpoint_id
            not in {
                stored_state.get("latestCheckpointId"),
                stored_state.get("recoveryCheckpointId"),
            }
        ):
            raise GfmTrainingError("Product resume checkpoint differs from durable state")
        progress_payload = load_gfm_checkpoint(progress_checkpoint, map_location="cpu")
        embedded_progress = progress_payload.get("components", {}).get("product_config")
        if embedded_progress != progress_config:
            raise GfmTrainingError(
                "Product resume checkpoint configuration differs from recomputed evidence"
            )
        resume_value = _product_resume_from_payload(progress_payload)
        expected_cursor = {
            "schemaVersion": "gfm.product-train-cursor/1.0",
            "epoch": resume_value.train_epoch,
            "batchOffset": resume_value.train_batch_offset,
        }
        if (
            stored_state.get("trainCursor") != expected_cursor
            or stored_state.get("trainIteratorContractHash")
            != resume_value.train_iterator_contract_hash
            or resume_value.train_iterator_contract_hash != progress_config_hash
        ):
            raise GfmTrainingError("Product resume iterator cursor differs from durable run state")
        durable_state = stored_state
        _normalize_product_progress_roles(
            checkpoint_dir,
            run_id=run_id,
            selected_checkpoint_id=progress_checkpoint.checkpoint_id,
        )
    core = SocialGraphFMCore(_model_config(config, variant))
    load_product_backbone(core, backbone_payload["components"]["core"])
    model = ProductTaskModule(core, task=task, pair_feature_dim=8)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        weight_decay=config.optimization.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _warmup_cosine(
            step,
            maximum=maximum_steps,
            warmup_ratio=config.optimization.warmup_ratio,
        ),
    )

    def persist_progress(value: Any) -> None:
        nonlocal durable_state
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Keep the old latest until the new run-state commit marker has been
        # atomically replaced.  Either side of a crash therefore names a file
        # that still exists and passes the checkpoint digest.
        _rotate_latest_to_recovery(checkpoint_dir, run_id=run_id)
        checkpoint_id = f"{run_id}-latest-{value.completed_steps}"
        progress = save_gfm_checkpoint(
            checkpoint_dir,
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            epoch=0,
            step=value.completed_steps,
            components={
                "product": value.latest_model_state,
                "product_config": progress_config,
            },
            optimizer_state=value.optimizer_state,
            scheduler_state=value.scheduler_state,
            scaler_state=value.scaler_state,
            sampler_state={
                "completedSteps": value.completed_steps,
                "trainCursor": {
                    "schemaVersion": "gfm.product-train-cursor/1.0",
                    "epoch": value.train_epoch,
                    "batchOffset": value.train_batch_offset,
                },
                "trainIteratorContractHash": value.train_iterator_contract_hash,
                "noImprovementEvaluations": value.no_improvement_evaluations,
                "batchContract": batch_contract,
                "batchContractHash": batch_contract_hash,
                "embeddingArtifacts": embedding_artifacts,
                "embeddingArtifactsHash": embedding_artifacts_hash,
                "taskAssets": task_assets,
                "taskAssetsHash": task_assets_hash,
                "split": "train-validation-only",
            },
            best_state={
                "task": task,
                "productConfigHash": progress_config_hash,
                "bestStep": value.best_step,
                "validationLoss": value.best_validation_loss,
                "bestProductState": value.best_state,
                "history": value.history,
            },
            config=config.logical_payload(),
            corpus_hashes=corpus_hashes,
            rng_state=_product_checkpoint_rng_state(value),
        )
        recovery_paths = sorted(checkpoint_dir.glob(f"{run_id}-recovery-*.manifest.json"))
        if len(recovery_paths) > 1:
            raise GfmTrainingError("Product run has ambiguous recovery checkpoints")
        recovery_id = (
            None if not recovery_paths else recovery_paths[0].stem.removesuffix(".manifest")
        )
        durable_state = {
            **durable_state,
            "completedSteps": value.completed_steps,
            "trainCursor": {
                "schemaVersion": "gfm.product-train-cursor/1.0",
                "epoch": value.train_epoch,
                "batchOffset": value.train_batch_offset,
            },
            "trainIteratorContractHash": value.train_iterator_contract_hash,
            "latestCheckpointId": progress.checkpoint_id,
            "recoveryCheckpointId": recovery_id,
        }
        _save_run_state(run_dir / "run-state.json", durable_state)
        _retain_checkpoint_roles(
            checkpoint_dir,
            run_id=run_id,
            current_id=progress.checkpoint_id,
            role="latest",
        )

    training = train_product_steps(
        model,
        optimizer,
        train_batches=lambda: (item.transformed(transform) for item in prepared("train", None)),
        validation_batches=lambda: (item.batch for item in prepared("validation", transform)),
        device=device,
        config=ProductTrainingConfig(
            maximum_steps=maximum_steps,
            minimum_steps=minimum_steps,
            evaluation_every_steps=evaluation_every_steps,
            patience_evaluations=6,
            gradient_clip=config.optimization.gradient_clip,
            amp=True,
            train_iterator_contract_hash=progress_config_hash,
        ),
        scheduler=scheduler,
        resume_state=resume_value,
        progress_callback=persist_progress,
    )
    if training.peak_cuda_memory_mib >= config.optimization.cuda_memory_limit_mib:
        raise GfmTrainingError("Product adaptation exceeded the 7168 MiB CUDA limit")
    model.load_state_dict(training.best_state)
    validation_values = _product_logits(
        model,
        prepared("validation", transform),
        device=device,
        baseline_config=collaboration_baseline,
    )
    calibration_logits = (
        validation_values["participation_logits"]
        if task == "newcomer"
        else validation_values["pair_logits"]
    )
    calibration_labels = (
        validation_values["participation_labels"]
        if task == "newcomer"
        else validation_values["pair_labels"]
    )
    temperature = float(
        fit_temperature(torch.from_numpy(calibration_logits), torch.from_numpy(calibration_labels))
        .temperature.detach()
        .cpu()
    )
    product_config = {
        **{
            key: value
            for key, value in progress_config.items()
            if key not in {"schemaVersion", "taskConfigHash", "steps"}
        },
        "schemaVersion": "gfm.product-adapt-config/1.0",
        "temperature": temperature,
        "steps": {
            "maximum": maximum_steps,
            "minimum": minimum_steps,
            "evaluationEvery": evaluation_every_steps,
            "patienceEvaluations": 6,
            "best": training.best_step,
            "completed": training.completed_steps,
        },
        "testRead": False,
    }
    product_config["taskConfigHash"] = canonical_sha256(product_config)
    checkpoint_id = f"{run_id}-best-{training.best_step}"
    checkpoint = save_gfm_checkpoint(
        checkpoint_dir,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        epoch=0,
        step=training.best_step,
        components={
            "product": model.state_dict(),
            "product_config": product_config,
        },
        # Formal evaluation uses the validation-selected component.  Resume
        # always uses the separately retained optimizer-aligned latest/recovery
        # state, never this potentially older best step.
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        sampler_state={
            "split": "train-validation-only",
            "streamingRebuildableFactory": True,
            "batchContract": batch_contract,
            "batchContractHash": batch_contract_hash,
            "embeddingArtifacts": embedding_artifacts,
            "embeddingArtifactsHash": embedding_artifacts_hash,
            "taskAssets": task_assets,
            "taskAssetsHash": task_assets_hash,
            "latestResumeCheckpointId": durable_state.get("latestCheckpointId"),
            "recoveryCheckpointId": durable_state.get("recoveryCheckpointId"),
        },
        best_state={
            "validationLoss": training.best_validation_loss,
            "temperature": temperature,
            "task": task,
            "productConfigHash": product_config["taskConfigHash"],
            "expectedFixedSampleDigest": canonical_sha256(
                {
                    "modelState": _state_digest(model.state_dict()),
                    "validationPairLogits": canonical_sha256(
                        validation_values["pair_logits"].tolist()
                    ),
                    "validationParticipationLogits": canonical_sha256(
                        validation_values["participation_logits"].tolist()
                    ),
                    "productConfigHash": product_config["taskConfigHash"],
                }
            ),
        },
        config=config.logical_payload(),
        corpus_hashes=corpus_hashes,
    )
    _retain_checkpoint_roles(
        checkpoint_dir,
        run_id=run_id,
        current_id=checkpoint.checkpoint_id,
        role="best",
    )
    run = GfmRunManifest.create(
        runId=run_id,
        experimentId=experiment_id,
        phase="adapt",
        architectureVariant=variant,
        status="succeeded",
        domainIds=(DOMAIN_IDS["openalex"],),
        seed=seed,
        codeHash=current_code_hash,
        environmentHash=current_environment_hash,
        configHash=config.config_hash,
        corpusHashes=corpus_hashes,
        taskProtocolHashes=(protocol.protocol_hash,),
        startedAt=started,
        finishedAt=datetime.now(UTC),
        peakCudaMemoryMiB=training.peak_cuda_memory_mib,
        artifactPaths=(str(run_dir / "checkpoints" / f"{checkpoint_id}.manifest.json"),),
    )
    _write_contract(run_dir / "run-manifest.json", run)
    atomic_write_json(run_dir / "product-config.json", product_config)
    registry.record_run(run)
    registry.record_checkpoint(checkpoint)
    _save_run_state(
        run_dir / "run-state.json",
        {
            **durable_state,
            "status": "succeeded",
            "completedSteps": training.completed_steps,
            "bestCheckpointId": checkpoint.checkpoint_id,
            "finishedAt": run.finished_at.isoformat() if run.finished_at else None,
        },
    )
    return {
        "runId": run_id,
        "checkpointId": checkpoint_id,
        "seed": seed,
        "variant": variant,
        "task": task,
        "bestStep": training.best_step,
        "completedSteps": training.completed_steps,
        "bestValidationLoss": training.best_validation_loss,
        "temperature": temperature,
        "featureTransformHash": canonical_sha256(transform.as_dict()),
        "peakCudaMemoryMiB": training.peak_cuda_memory_mib,
        "testRead": False,
        "streamingBatches": True,
    }


def _state_digest(state: Mapping[str, Any]) -> str:
    from ..tensor_digest import canonical_tensor_digest

    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


__all__ = []
