"""Product evaluation, suite checkpoint construction, and acceptance.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def _product_task_from_checkpoint(payload: Mapping[str, Any]) -> ProductTask:
    components = payload.get("components")
    best_state = payload.get("best_state")
    if not isinstance(components, dict) or set(components) != {
        "product",
        "product_config",
    }:
        raise ContractViolation("Evaluation checkpoint is not a single product model")
    if not isinstance(best_state, dict):
        raise ContractViolation("Product checkpoint lacks task identity")
    task = best_state.get("task")
    if task not in ("collaboration", "newcomer"):
        raise ContractViolation("Product checkpoint task identity is invalid")
    config = components.get("product_config")
    if not isinstance(config, dict) or config.get("task") != task:
        raise ContractViolation("Product checkpoint lacks embedded task configuration")
    checked = dict(config)
    task_config_hash = checked.pop("taskConfigHash", None)
    if task_config_hash != canonical_sha256(checked):
        raise ContractViolation("Embedded product configuration hash is invalid")
    if best_state.get("productConfigHash") != task_config_hash:
        raise ContractViolation("Product best state differs from its embedded configuration")
    return task


def _product_config_from_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    _product_task_from_checkpoint(payload)
    return dict(payload["components"]["product_config"])


def _stratified_calibration_metrics(
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    institution_group: np.ndarray,
    topic_group: np.ndarray,
    collaboration_kind: np.ndarray | None,
    task: ProductTask,
) -> dict[str, float]:
    import torch

    from ..gfm.evaluation import expected_calibration_error

    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    institution = np.asarray(institution_group, dtype=np.int8).reshape(-1)
    topic = np.asarray(topic_group, dtype=np.int8).reshape(-1)
    if not (
        probability.shape == target.shape == institution.shape == topic.shape and probability.size
    ):
        raise GfmTrainingError("Calibration samples and strata are misaligned")
    if set(np.unique(institution).tolist()) != {0, 1, 2}:
        raise GfmTrainingError("Institution-size strata do not exactly cover three groups")
    if set(np.unique(topic).tolist()) != {0, 1, 2}:
        raise GfmTrainingError(
            "Topic calibration contains missing/unknown groups; cannot claim completeness"
        )
    masks: dict[str, np.ndarray] = {
        "institution_small": institution == 0,
        "institution_medium": institution == 1,
        "institution_large": institution == 2,
        "topic_cluster_0": topic == 0,
        "topic_cluster_1": topic == 1,
        "topic_cluster_2": topic == 2,
    }
    if task == "collaboration":
        if collaboration_kind is None:
            raise GfmTrainingError("Collaboration calibration lacks first/repeat strata")
        kinds = np.asarray(collaboration_kind, dtype=np.int8).reshape(-1)
        if kinds.shape != target.shape or set(np.unique(kinds).tolist()) != {0, 1}:
            raise GfmTrainingError("First/repeat strata are incomplete")
        masks.update({"first_time": kinds == 0, "repeated": kinds == 1})
    else:
        masks["newcomer"] = np.ones(target.shape[0], dtype=np.bool_)
    metrics: dict[str, float] = {"strata_complete": 1.0}
    for name, mask in masks.items():
        if not bool(mask.any()) or np.unique(target[mask]).size != 2:
            raise GfmTrainingError(f"Calibration stratum {name} lacks both actual outcomes")
        value = expected_calibration_error(
            torch.from_numpy(probability[mask]), torch.from_numpy(target[mask])
        )
        metrics[f"ece_{name}"] = value.expected_calibration_error
    return metrics


def _evaluate_product_checkpoint(
    *,
    layout: RuntimeLayout,
    experiment_id: str,
    checkpoint: Any,
    split: Literal["test", "shadow"],
    evaluator_code_hash: str,
    evaluator_environment_hash: str,
    device: str = "cpu",
) -> tuple[GfmEvaluationReport, GfmEvaluationReport]:
    from ..gfm.model import SocialGraphFMCore
    from ..gfm.product_training import ProductTaskModule, evaluate_product_predictions

    registry = _registry(layout)
    run = registry.get_run(checkpoint.run_id)
    if run is None or run.phase != "adapt":
        raise ContractViolation("Product evaluation requires an adapt checkpoint")
    if evaluator_code_hash != run.code_hash or evaluator_environment_hash != run.environment_hash:
        raise ContractViolation("Product evaluator provenance differs from its adaptation run")
    payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
    raw_config = _product_config_from_checkpoint(payload)
    task = _product_task_from_checkpoint(payload)
    variant = raw_config.get("architectureVariant")
    if variant != run.architecture_variant:
        raise ContractViolation("Product checkpoint architecture differs from its run")
    seed = int(raw_config["seed"])
    transform = _FeatureTransform.from_dict(raw_config["featureTransform"])
    temperature = float(raw_config["temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ContractViolation("Product calibration temperature is invalid")
    access_role: Literal["test", "shadow"] = split
    test_read_path = (
        layout.gfm_reports
        / experiment_id
        / "test-read"
        / f"{checkpoint.checkpoint_id}-{split}.json"
    )
    intent = read_json_object(test_read_path)
    if (
        intent.get("checkpointId") != checkpoint.checkpoint_id
        or intent.get("split") != split
        or intent.get("status") != "intent-persisted-before-array-access"
    ):
        raise ContractViolation("Product test/shadow role view requires a durable one-shot intent")
    config = _load_pretrain_config(None, None)
    corpora, embeddings = _ensure_pretrain_evidence(
        layout,
        maximum_role=access_role,
        physical_boundary=True,
    )
    current_task_assets = _product_task_asset_evidence(layout, task=task, corpora=corpora)
    if raw_config.get("taskAssets") != current_task_assets or raw_config.get(
        "taskAssetsHash"
    ) != canonical_sha256(current_task_assets):
        raise ContractViolation("Product evaluation task assets differ from its checkpoint binding")
    stream = _make_domain_streams(
        layout,
        embeddings,
        domain_ids=(DOMAIN_IDS["openalex"],),
        maximum_role=access_role,
    )[DOMAIN_IDS["openalex"]]
    target_view = load_domain_view(
        layout.root,
        DOMAIN_IDS["openalex"],
        maximum_role=access_role,
        families=("targets",),
    )
    arrays = target_view["arrays"]
    newcomer_view = (
        load_openalex_newcomers_view(layout.root, maximum_role=access_role)
        if task == "newcomer"
        else None
    )
    newcomers = newcomer_view["arrays"] if newcomer_view is not None else None

    def prepared() -> Iterator[_PreparedProductBatch]:
        return _product_batches_for_split(
            task=task,
            stream=stream,
            arrays=arrays,
            newcomers=newcomers,
            split=split,
            seed=seed,
            transform=transform,
            collaboration_kind="both",
        )

    model = ProductTaskModule(
        SocialGraphFMCore(_model_config(config, variant)),
        task=task,
        pair_feature_dim=8,
    )
    model.load_state_dict(payload["components"]["product"])
    if task == "collaboration":
        product_prepared = (item for item in prepared() if item.collaboration_kind == "first")
    else:
        product_prepared = (item for item in prepared() if item.batch.query_ids.numel())
    collaboration_baseline = raw_config.get("collaborationBaseline")
    product_values = _product_logits(
        model,
        product_prepared,
        device=device,
        baseline_config=collaboration_baseline,
    )
    all_values = _product_logits(
        model,
        prepared(),
        device=device,
        baseline_config=collaboration_baseline,
    )
    calibrated_pair_probability = 1.0 / (
        1.0 + np.exp(-product_values["pair_logits"].astype(np.float64) / temperature)
    )
    rerank_components = (
        _collaboration_rerank_components(
            calibrated_pair_probability, product_values["raw_features"]
        )
        if task == "collaboration"
        else None
    )
    ranking_probability = (
        rerank_components["finalRerank"]
        if rerank_components is not None
        else calibrated_pair_probability
    )
    participation_probability = (
        1.0 / (1.0 + np.exp(-all_values["participation_logits"].astype(np.float64) / temperature))
        if task == "newcomer"
        else None
    )
    repeat_values: dict[str, np.ndarray] | None = None
    repeat_prediction = None
    repeat_rerank_components: dict[str, np.ndarray] | None = None
    if task == "collaboration":
        repeat_values = _product_logits(
            model,
            (item for item in prepared() if item.collaboration_kind == "repeat"),
            device=device,
            baseline_config=collaboration_baseline,
        )
        repeat_probability = 1.0 / (
            1.0 + np.exp(-repeat_values["pair_logits"].astype(np.float64) / temperature)
        )
        repeat_rerank_components = _collaboration_rerank_components(
            repeat_probability, repeat_values["raw_features"]
        )
        repeat_prediction = evaluate_product_predictions(
            task="collaboration",
            ranking_probabilities=repeat_rerank_components["finalRerank"],
            ranking_labels=repeat_values["pair_labels"],
            query_ids=repeat_values["query_ids"],
            baseline_scores=repeat_values["baseline_scores"],
            seed=seed + 1_000_000,
        )
    ranking_prediction = evaluate_product_predictions(
        task=task,
        ranking_probabilities=ranking_probability,
        ranking_labels=product_values["pair_labels"],
        query_ids=product_values["query_ids"],
        baseline_scores=product_values["baseline_scores"],
        participation_probabilities=participation_probability,
        participation_labels=(all_values["participation_labels"] if task == "newcomer" else None),
        seed=seed,
    )
    probability_prediction = (
        evaluate_product_predictions(
            task="collaboration",
            ranking_probabilities=calibrated_pair_probability,
            ranking_labels=product_values["pair_labels"],
            query_ids=product_values["query_ids"],
            baseline_scores=product_values["baseline_scores"],
            seed=seed,
        )
        if task == "collaboration"
        else ranking_prediction
    )
    if task == "collaboration":
        calibration_probability = 1.0 / (
            1.0 + np.exp(-all_values["pair_logits"].astype(np.float64) / temperature)
        )
        calibration_labels = all_values["pair_labels"]
        institution = np.concatenate(
            [item.institution_group for item in prepared() if item.institution_group is not None]
        )
        topic = np.concatenate(
            [item.topic_group for item in prepared() if item.topic_group is not None]
        )
        kinds = np.concatenate(
            [
                np.full(
                    item.batch.pair_labels.numel(),
                    0 if item.collaboration_kind == "first" else 1,
                    dtype=np.int8,
                )
                for item in prepared()
            ]
        )
    else:
        if participation_probability is None:
            raise GfmTrainingError("Newcomer participation probability is absent")
        calibration_probability = participation_probability
        calibration_labels = all_values["participation_labels"]
        institution = np.concatenate(
            [item.institution_group for item in prepared() if item.institution_group is not None]
        )
        topic = np.concatenate(
            [item.topic_group for item in prepared() if item.topic_group is not None]
        )
        kinds = None
    strata = _stratified_calibration_metrics(
        probabilities=calibration_probability,
        labels=calibration_labels,
        institution_group=institution,
        topic_group=topic,
        collaboration_kind=kinds,
        task=task,
    )
    audit_hash, audit_path, counters = _leakage_audit(
        layout,
        experiment_id=experiment_id,
        audit_id=f"{checkpoint.checkpoint_id}-{split}-{task}",
        evidence={
            "checkpointId": checkpoint.checkpoint_id,
            "checkpointStateHash": checkpoint.state_hash,
            "evaluatorCodeHash": evaluator_code_hash,
            "evaluatorEnvironmentHash": evaluator_environment_hash,
            "split": split,
            "task": task,
            "featureTransformHash": canonical_sha256(raw_config["featureTransform"]),
            "featureTransformFitSplit": "train-only",
            "temperatureFitSplit": "validation-only",
            "sampleProvenanceHashes": sorted(
                {
                    canonical_sha256(
                        {
                            "domainId": item.batch.provenance.domain_id,
                            "graphVersion": item.batch.provenance.graph_version,
                            "cutoff": item.batch.provenance.cutoff,
                            "horizon": item.batch.provenance.horizon,
                            "taskId": item.batch.provenance.task_id,
                            "sourceCorpusHash": item.batch.provenance.source_corpus_hash,
                        }
                    )
                    for item in prepared()
                }
            ),
            "allHorizonPositivesExcludedFromNegatives": True,
            "candidateCountPerFirstCollaborationQuery": 100,
            "domainAccessAudit": stream.access_audit,
            "targetAccessAudit": target_view["accessAudit"],
            "newcomerAccessAudit": (
                newcomer_view["accessAudit"] if newcomer_view is not None else None
            ),
            "embeddingAccessEvidence": _embedding_artifact_evidence(embeddings),
        },
        counters=_product_audit_counters(prepared()),
    )
    baseline_definition = {
        "schemaVersion": "gfm.product-baseline-definition/1.0",
        "task": task,
        "collaboration": raw_config.get("collaborationBaseline"),
        "newcomer": "cutoff-topic-similarity-plus-cutoff-activity",
        "rawStructuralFeatures": ["cn", "aa", "ra"],
        "featureTransform": raw_config["featureTransform"],
        "futureFields": [],
    }
    baseline_definition_hash = canonical_sha256(baseline_definition)
    product_metrics = {
        **ranking_prediction.metrics(),
        "auprc": probability_prediction.auprc,
        "label_prevalence": probability_prediction.label_prevalence,
        "ece": probability_prediction.ece,
        "brier": probability_prediction.brier,
        **counters,
    }
    if repeat_prediction is not None:
        product_metrics.update(
            {
                "repeat_ndcg@20": repeat_prediction.ndcg_at_20,
                "repeat_baseline_ndcg@20": repeat_prediction.baseline_ndcg_at_20,
                "repeat_recall@20": repeat_prediction.recall_at_20,
                "repeat_baseline_recall@20": repeat_prediction.baseline_recall_at_20,
                "repeat_query_count": float(repeat_prediction.query_count),
            }
        )
    ranking_institution = _product_candidate_groups(
        prepared(),
        name="institution",
        collaboration_kind="first" if task == "collaboration" else "both",
    )
    ranking_topic = _product_candidate_groups(
        prepared(),
        name="topic",
        collaboration_kind="first" if task == "collaboration" else "both",
    )
    performance_strata = {
        **_ranking_stratum_metrics(
            scores=ranking_probability,
            labels=product_values["pair_labels"],
            query_ids=product_values["query_ids"],
            groups=ranking_institution,
            axis="institution",
            group_names={0: "small", 1: "medium", 2: "large"},
        ),
        **_ranking_stratum_metrics(
            scores=ranking_probability,
            labels=product_values["pair_labels"],
            query_ids=product_values["query_ids"],
            groups=ranking_topic,
            axis="topic",
            group_names={0: "cluster0", 1: "cluster1", 2: "cluster2"},
        ),
    }
    product_metrics.update(performance_strata)
    if rerank_components is not None:
        product_metrics.update(
            {
                f"mean_{name}": float(np.mean(values))
                for name, values in rerank_components.items()
                if name != "finalRerank"
            }
        )
    product_evidence_hash, product_evidence_path = _evaluation_evidence(
        layout,
        experiment_id=experiment_id,
        evidence_id=f"{checkpoint.checkpoint_id}-{split}-product-{task}",
        payload={
            "checkpointId": checkpoint.checkpoint_id,
            "checkpointStateHash": checkpoint.state_hash,
            "evaluatorCodeHash": evaluator_code_hash,
            "evaluatorEnvironmentHash": evaluator_environment_hash,
            "task": task,
            "split": split,
            "rankingProbabilityHash": canonical_sha256(ranking_probability.tolist()),
            "calibratedProbabilityHash": canonical_sha256(calibrated_pair_probability.tolist()),
            "rerankComponents": (
                None
                if rerank_components is None
                else {
                    name: {
                        "hash": canonical_sha256(values.tolist()),
                        "mean": float(np.mean(values)),
                    }
                    for name, values in rerank_components.items()
                }
            ),
            "rerankDefinition": raw_config.get("collaborationRerank"),
            "repeatCollaboration": (
                None
                if repeat_values is None
                or repeat_prediction is None
                or repeat_rerank_components is None
                else {
                    "role": "auxiliary-frozen-model-report-not-primary-hard-gate",
                    "rankingProbabilityHash": canonical_sha256(
                        repeat_rerank_components["finalRerank"].tolist()
                    ),
                    "rankingLabelHash": canonical_sha256(repeat_values["pair_labels"].tolist()),
                    "queryIdHash": canonical_sha256(repeat_values["query_ids"].tolist()),
                    "baselineScoreHash": canonical_sha256(
                        repeat_values["baseline_scores"].tolist()
                    ),
                    "metrics": {
                        "ndcg@20": repeat_prediction.ndcg_at_20,
                        "baseline_ndcg@20": repeat_prediction.baseline_ndcg_at_20,
                        "recall@20": repeat_prediction.recall_at_20,
                        "baseline_recall@20": repeat_prediction.baseline_recall_at_20,
                        "query_count": float(repeat_prediction.query_count),
                    },
                }
            ),
            "rankingLabelHash": canonical_sha256(product_values["pair_labels"].tolist()),
            "queryIdHash": canonical_sha256(product_values["query_ids"].tolist()),
            "baselineScoreHash": canonical_sha256(product_values["baseline_scores"].tolist()),
            "baselineDefinition": baseline_definition,
            "baselineDefinitionHash": baseline_definition_hash,
            "metrics": product_metrics,
            "brier": probability_prediction.brier,
            "participationProbabilityHash": (
                None
                if participation_probability is None
                else canonical_sha256(participation_probability.tolist())
            ),
            "participationLabelHash": canonical_sha256(all_values["participation_labels"].tolist()),
            "sampleProvenance": [
                {
                    "domainId": item.batch.provenance.domain_id,
                    "graphVersion": item.batch.provenance.graph_version,
                    "cutoff": item.batch.provenance.cutoff,
                    "horizon": item.batch.provenance.horizon,
                    "taskId": item.batch.provenance.task_id,
                    "sourceCorpusHash": item.batch.provenance.source_corpus_hash,
                }
                for item in prepared()
            ],
            "counts": {
                "rankingOutcomes": int(ranking_probability.size),
                "queries": int(np.unique(product_values["query_ids"]).size),
                "participationOutcomes": int(all_values["participation_labels"].size),
            },
            "performanceStrata": performance_strata,
        },
    )
    strata_definition = {
        "schemaVersion": "gfm.calibration-strata-definition/1.0",
        "institutionAxis": {
            "small": "count<10",
            "medium": "10<=count<100",
            "large": "count>=100",
        },
        "topicAxis": [0, 1, 2],
        "taskAxis": (["first_time", "repeated"] if task == "collaboration" else ["newcomer"]),
        "partitionPolicy": "each-axis-exactly-once",
        "cutoffVisibleOnly": True,
    }
    strata_definition_hash = canonical_sha256(strata_definition)
    calibration_metrics = {
        **strata,
        **counters,
        "ece": probability_prediction.ece,
        "brier": probability_prediction.brier,
    }
    calibration_evidence_hash, calibration_evidence_path = _evaluation_evidence(
        layout,
        experiment_id=experiment_id,
        evidence_id=f"{checkpoint.checkpoint_id}-{split}-calibration-{task}",
        payload={
            "checkpointId": checkpoint.checkpoint_id,
            "checkpointStateHash": checkpoint.state_hash,
            "evaluatorCodeHash": evaluator_code_hash,
            "evaluatorEnvironmentHash": evaluator_environment_hash,
            "task": task,
            "split": split,
            "probabilityHash": canonical_sha256(calibration_probability.tolist()),
            "labelHash": canonical_sha256(calibration_labels.tolist()),
            "institutionGroupHash": canonical_sha256(institution.tolist()),
            "topicGroupHash": canonical_sha256(topic.tolist()),
            "collaborationKindHash": (None if kinds is None else canonical_sha256(kinds.tolist())),
            "temperature": temperature,
            "temperatureFitSplit": "validation-only",
            "outcomeCount": int(calibration_probability.size),
            "strataDefinition": strata_definition,
            "strataDefinitionHash": strata_definition_hash,
            "metrics": calibration_metrics,
            "brier": probability_prediction.brier,
        },
    )
    task_id = COLLABORATION_TASK if task == "collaboration" else NEWCOMER_TASK
    warnings = ("shadow", "no-model-selection") if split == "shadow" else ()
    product_report = GfmEvaluationReport.create(
        reportId=f"{checkpoint.checkpoint_id}-{split}-product-{task}",
        experimentId=experiment_id,
        runId=run.run_id,
        checkpointId=checkpoint.checkpoint_id,
        evaluationKind="product",
        domainId=DOMAIN_IDS["openalex"],
        taskId=task_id,
        evaluatorCodeHash=evaluator_code_hash,
        evaluatorEnvironmentHash=evaluator_environment_hash,
        seed=seed,
        metrics=product_metrics,
        evidenceArtifactHash=product_evidence_hash,
        evidenceArtifactPath=product_evidence_path,
        baselineDefinitionHash=baseline_definition_hash,
        ece=probability_prediction.ece,
        brier=probability_prediction.brier,
        peakCudaMemoryMiB=run.peak_cuda_memory_mib,
        leakageAuditPassed=True,
        leakageAuditHash=audit_hash,
        leakageAuditPath=audit_path,
        warnings=warnings,
    )
    calibration_report = GfmEvaluationReport.create(
        reportId=f"{checkpoint.checkpoint_id}-{split}-calibration-{task}",
        experimentId=experiment_id,
        runId=run.run_id,
        checkpointId=checkpoint.checkpoint_id,
        evaluationKind="calibration",
        domainId=DOMAIN_IDS["openalex"],
        taskId=task_id,
        evaluatorCodeHash=evaluator_code_hash,
        evaluatorEnvironmentHash=evaluator_environment_hash,
        seed=seed,
        metrics=calibration_metrics,
        evidenceArtifactHash=calibration_evidence_hash,
        evidenceArtifactPath=calibration_evidence_path,
        strataDefinitionHash=strata_definition_hash,
        ece=probability_prediction.ece,
        brier=probability_prediction.brier,
        peakCudaMemoryMiB=run.peak_cuda_memory_mib,
        leakageAuditPassed=True,
        leakageAuditHash=audit_hash,
        leakageAuditPath=audit_path,
        warnings=warnings,
    )
    return product_report, calibration_report


def evaluate_gfm(
    *,
    root: str | Path | None,
    protocol: EvaluationProtocol,
    experiment_id: str,
    held_out_domain: str | None = None,
    variant: Literal["core-base", "core-moe"] | None = None,
    seed: int | None = None,
    task: Literal["collaboration"] | None = None,
) -> dict[str, Any]:
    if protocol not in ("lodo", "product", "shadow"):
        raise ContractViolation("GFM evaluation protocol must be lodo, product or shadow")
    if task is not None and (protocol != "product" or task != "collaboration"):
        raise ContractViolation(
            "Task-scoped evaluation supports only --protocol product --task collaboration"
        )
    layout = prepare_runtime_layout(root, operation="run")
    _require_experiment_runs(layout, experiment_id)
    if protocol == "lodo":
        require_ml_runtime("cuda")
        config = _load_pretrain_config(None, None)
        registry = _registry(layout)
        if held_out_domain is not None and held_out_domain not in DOMAIN_IDS.values():
            raise ContractViolation("Requested LODO held-out domain is unknown")
        if variant is not None and variant not in config.architecture.candidates:
            raise ContractViolation("Requested LODO variant is outside the checked config")
        if seed is not None and seed not in config.formal.seeds:
            raise ContractViolation("Requested LODO seed is outside the checked config")
        corpora = _load_corpus_contracts(layout, physical_boundary=True)
        protocols = _register_prerequisites(layout, corpora)
        corpus_by_domain = {corpus.domain_id: corpus for corpus in corpora}
        selected_variants = (variant,) if variant is not None else config.architecture.candidates
        selected_domains = (
            (held_out_domain,) if held_out_domain is not None else tuple(DOMAIN_IDS.values())
        )
        selected_seeds = (seed,) if seed is not None else config.formal.seeds
        generated: list[dict[str, Any]] = []
        reused: list[dict[str, Any]] = []
        for selected_variant in selected_variants:
            for held_out in selected_domains:
                source_domains = tuple(
                    sorted(domain for domain in DOMAIN_IDS.values() if domain != held_out)
                )
                corpus_hashes = tuple(
                    corpus_by_domain[domain].logical_hash for domain in source_domains
                ) + (corpus_by_domain[held_out].logical_hash,)
                for selected_seed in selected_seeds:
                    run_id = f"{experiment_id}-lodo-{selected_variant}-{held_out}-{selected_seed}"
                    completed = _validate_completed_matrix_run(
                        layout,
                        _CompletedRunExpectation(
                            experiment_id=experiment_id,
                            run_id=run_id,
                            phase="lodo",
                            variant=selected_variant,
                            seed=selected_seed,
                            domain_ids=source_domains,
                            held_out_domain=held_out,
                            corpus_hashes=corpus_hashes,
                            protocol_hashes=tuple(item.protocol_hash for item in protocols),
                            config_hash=config.config_hash,
                            code_hash=code_identity_hash(),
                            environment_hash=_environment_hash("cuda"),
                            required_reports=((f"{run_id}-lodo", "lodo"),),
                        ),
                    )
                    if completed is not None:
                        lodo_config_path = (
                            layout.gfm_runs / experiment_id / run_id / "lodo-config.json"
                        )
                        lodo_config = read_json_object(lodo_config_path)
                        checked_lodo_config = dict(lodo_config)
                        lodo_config_hash = checked_lodo_config.pop("taskConfigHash", None)
                        if (
                            lodo_config_hash != canonical_sha256(checked_lodo_config)
                            or lodo_config.get("heldOutDomain") != held_out
                            or lodo_config.get("architectureVariant") != selected_variant
                            or lodo_config.get("seed") != selected_seed
                            or lodo_config.get("sourceDomainIds") != list(source_domains)
                            or lodo_config.get("sourceSteps") != config.formal.max_steps
                            or lodo_config.get("targetStepsPerControl")
                            != config.transfer.lodo_target_adaptation_steps
                        ):
                            raise GfmTrainingError(
                                f"Completed LODO run {run_id} has stale task configuration"
                            )
                        report = completed.reports[f"{run_id}-lodo"]
                        reused.append(
                            {
                                "runId": run_id,
                                "checkpointId": completed.checkpoint.checkpoint_id,
                                "heldOutDomain": held_out,
                                "architectureVariant": selected_variant,
                                "seed": selected_seed,
                                "metrics": dict(report.metrics),
                                "isolationAuditHash": report.leakage_audit_hash,
                                "peakCudaMemoryMiB": completed.run.peak_cuda_memory_mib,
                                "reused": True,
                            }
                        )
                        continue
                    worker = _subprocess_json(
                        (
                            "_gfm-lodo-run",
                            "--experiment-id",
                            experiment_id,
                            "--held-out-domain",
                            held_out,
                            "--variant",
                            selected_variant,
                            "--seed",
                            str(selected_seed),
                            "--device",
                            "cuda",
                            "--root",
                            str(layout.root),
                            "--json",
                        )
                    )
                    run = worker.get("run")
                    if not isinstance(run, dict):
                        raise GfmTrainingError("LODO worker omitted its run summary")
                    generated.append(run)
        reports = [
            report
            for report in registry.list_evaluations(experiment_id=experiment_id)
            if report.evaluation_kind == "lodo"
        ]
        expected = len(config.architecture.candidates) * len(DOMAIN_IDS) * len(config.formal.seeds)
        filtered = any(value is not None for value in (held_out_domain, variant, seed))
        matrix_complete = len(reports) == expected
        if not filtered and not matrix_complete:
            raise ContractViolation(
                f"LODO matrix is incomplete: expected {expected}, found {len(reports)}"
            )
        return {
            "schemaVersion": "gfm.workflow-evaluate/1.0",
            "ok": True,
            "experimentId": experiment_id,
            "protocol": protocol,
            "independentProcesses": True,
            "generatedRuns": generated,
            "reusedRuns": reused,
            "selection": {
                "heldOutDomains": list(selected_domains),
                "variants": list(selected_variants),
                "seeds": list(selected_seeds),
            },
            "matrixComplete": matrix_complete,
            "expectedFullMatrixRuns": expected,
            "reports": [_contract_json(report) for report in reports],
        }
    if any(value is not None for value in (held_out_domain, variant, seed)):
        raise ContractViolation("LODO matrix selectors are only valid with --protocol lodo")
    registry = _registry(layout)
    split: Literal["test", "shadow"] = "shadow" if protocol == "shadow" else "test"
    adapt_runs = {
        run.run_id: run
        for run in registry.list_runs(experiment_id=experiment_id)
        if run.phase == "adapt" and run.status == "succeeded"
    }
    checkpoints = [
        checkpoint
        for checkpoint in registry.list_checkpoints(experiment_id=experiment_id)
        if checkpoint.run_id in adapt_runs
    ]
    if not checkpoints:
        raise ContractViolation(
            "Product/shadow evaluation requires completed collaboration and newcomer adaptation"
        )
    all_existing_reports = registry.list_evaluations(experiment_id=experiment_id)
    checkpoint_tasks: dict[str, ProductTask] = {}
    for checkpoint in checkpoints:
        checkpoint_tasks[checkpoint.checkpoint_id] = _product_task_from_checkpoint(
            load_gfm_checkpoint(checkpoint, map_location="cpu")
        )
    available_tasks = set(checkpoint_tasks.values())
    if task is None and available_tasks != {"collaboration", "newcomer"}:
        raise ContractViolation(
            "Both fixed product tasks must be adapted before full product/shadow evaluation"
        )
    evaluator_code_hash = code_identity_hash()
    evaluator_environment_hash = _environment_hash("cuda")
    evaluator_config = _load_pretrain_config(None, None)
    for checkpoint in checkpoints:
        evaluation_run = adapt_runs[checkpoint.run_id]
        if (
            evaluation_run.code_hash != evaluator_code_hash
            or evaluation_run.environment_hash != evaluator_environment_hash
            or evaluation_run.config_hash != evaluator_config.config_hash
            or checkpoint.config_hash != evaluation_run.config_hash
            or set(checkpoint.corpus_hashes) != set(evaluation_run.corpus_hashes)
        ):
            raise ContractViolation(
                "Product evaluator code/environment/config differs from its adaptation run"
            )
    if task == "collaboration":
        existing_acceptance = registry.latest_task_acceptance(experiment_id=experiment_id)
        if existing_acceptance is not None:
            verified = registry.verify_task_acceptance(existing_acceptance)
            acceptance_path = (
                layout.gfm_reports
                / experiment_id
                / "task-acceptance"
                / f"collaboration-product-{verified.report_hash}.json"
            )
            serialized_acceptance = _contract_json(verified)
            if acceptance_path.exists():
                if read_json_object(acceptance_path) != serialized_acceptance:
                    raise ContractViolation("Immutable collaboration acceptance artifact changed")
            else:
                atomic_write_json(acceptance_path, serialized_acceptance)
            return {
                "schemaVersion": "gfm.workflow-evaluate/1.0",
                "ok": True,
                "experimentId": experiment_id,
                "protocol": protocol,
                "task": task,
                "generatedReportIds": [],
                "reports": [
                    _contract_json(report)
                    for report in all_existing_reports
                    if report.task_id == COLLABORATION_TASK
                    and report.evaluation_kind in {"product", "calibration"}
                    and "shadow" not in report.warnings
                ],
                "taskAcceptance": _contract_json(verified),
                "taskAcceptanceArtifact": str(acceptance_path),
                "taskAcceptanceReused": True,
                "fullProductValidated": False,
                "modelValidated": False,
            }
        checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint_tasks[checkpoint.checkpoint_id] == "collaboration"
        ]
        config = _load_pretrain_config(None, None)
        selected_variant = _selected_core_variant(layout, experiment_id)
        selected_runs = [adapt_runs[checkpoint.run_id] for checkpoint in checkpoints]
        protocol_contract = next(
            value for value in _task_protocols() if value.task_id == COLLABORATION_TASK
        )
        fresh_by_source = {
            (report.run_id, report.checkpoint_id, report.seed): report
            for report in all_existing_reports
            if report.evaluation_kind == "fresh_process"
        }
        provenance = {
            (
                run.architecture_variant,
                run.config_hash,
                run.code_hash,
                run.environment_hash,
                tuple(sorted(run.corpus_hashes)),
                run.task_protocol_hashes,
            )
            for run in selected_runs
        }
        if (
            len(checkpoints) != 3
            or {run.seed for run in selected_runs} != set(config.formal.seeds)
            or len({run.run_id for run in selected_runs}) != 3
            or len(provenance) != 1
            or any(run.architecture_variant != selected_variant for run in selected_runs)
            or any(
                run.task_protocol_hashes != (protocol_contract.protocol_hash,)
                or checkpoint.config_hash != run.config_hash
                or tuple(sorted(checkpoint.corpus_hashes)) != tuple(sorted(run.corpus_hashes))
                or (
                    run.run_id,
                    checkpoint.checkpoint_id,
                    run.seed,
                )
                not in fresh_by_source
                for checkpoint, run in zip(checkpoints, selected_runs, strict=True)
            )
        ):
            raise ContractViolation(
                "Collaboration test evaluation requires the exact compatible three-seed "
                "formal adaptation and fresh-process matrix"
            )
        # Resolve and physically verify the accepted pretraining lineage before
        # persisting any test-read intent.  The task-only gate cannot be built
        # from a random or cross-experiment product head.
        registry.collaboration_backbone_bindings(
            experiment_id=experiment_id,
            product_checkpoint_ids=tuple(
                sorted(checkpoint.checkpoint_id for checkpoint in checkpoints)
            ),
        )
    generated_reports: list[GfmEvaluationReport] = []
    tasks_seen: set[ProductTask] = set()
    for checkpoint in checkpoints:
        checkpoint_task = checkpoint_tasks[checkpoint.checkpoint_id]
        tasks_seen.add(checkpoint_task)
        product_id = f"{checkpoint.checkpoint_id}-{split}-product-{checkpoint_task}"
        calibration_id = f"{checkpoint.checkpoint_id}-{split}-calibration-{checkpoint_task}"
        test_read_path = (
            layout.gfm_reports
            / experiment_id
            / "test-read"
            / f"{checkpoint.checkpoint_id}-{split}.json"
        )
        test_read_lock_path = test_read_path.with_suffix(f"{test_read_path.suffix}.lock")
        # The one-shot boundary is an inter-process critical section.  The
        # durable intent and the physical role-view open cannot be separated by
        # a check/write race, and reports cannot become authoritative before
        # their completed read state is persisted under the same lock.
        with exclusive_file_lock(test_read_lock_path):
            locked_reports = registry.list_evaluations(experiment_id=experiment_id)
            locked_by_id = {report.report_id: report for report in locked_reports}
            existing_pair = {
                identity for identity in (product_id, calibration_id) if identity in locked_by_id
            }
            if len(existing_pair) == 2:
                if not test_read_path.is_file():
                    raise GfmTrainingError(
                        "Product reports exist without completed one-shot read evidence"
                    )
                completed_read_state = read_json_object(test_read_path)
                if (
                    completed_read_state.get("status") != "completed"
                    or completed_read_state.get("readCount") != 1
                    or completed_read_state.get("physicalRoleView") is not True
                    or completed_read_state.get("maximumRole") != split
                    or completed_read_state.get("productReportHash")
                    != locked_by_id[product_id].report_hash
                    or completed_read_state.get("calibrationReportHash")
                    != locked_by_id[calibration_id].report_hash
                ):
                    raise GfmTrainingError(
                        "Product reports are bound to incomplete one-shot read evidence"
                    )
                continue
            if existing_pair:
                raise ContractViolation("Product evaluation has a partial immutable report pair")
            if test_read_path.exists():
                raise GfmTrainingError(
                    f"{split} arrays were already opened for this checkpoint; fail closed"
                )
            atomic_write_json(
                test_read_path,
                {
                    "schemaVersion": "gfm.product-test-read-state/1.0",
                    "experimentId": experiment_id,
                    "checkpointId": checkpoint.checkpoint_id,
                    "task": checkpoint_task,
                    "split": split,
                    "status": "intent-persisted-before-array-access",
                    "readCountCeiling": 1,
                },
            )
            product_report, calibration_report = _evaluate_product_checkpoint(
                layout=layout,
                experiment_id=experiment_id,
                checkpoint=checkpoint,
                split=split,
                evaluator_code_hash=evaluator_code_hash,
                evaluator_environment_hash=evaluator_environment_hash,
            )
            registry.record_evaluation(product_report)
            registry.record_evaluation(calibration_report)
            atomic_write_json(
                test_read_path,
                {
                    "schemaVersion": "gfm.product-test-read-state/1.0",
                    "experimentId": experiment_id,
                    "checkpointId": checkpoint.checkpoint_id,
                    "task": checkpoint_task,
                    "split": split,
                    "status": "completed",
                    "readCount": 1,
                    "physicalRoleView": True,
                    "maximumRole": split,
                    "productReportHash": product_report.report_hash,
                    "calibrationReportHash": calibration_report.report_hash,
                },
            )
            generated_reports.extend((product_report, calibration_report))
    if task is None and tasks_seen != {"collaboration", "newcomer"}:
        raise ContractViolation("Both fixed product tasks must be adapted before evaluation")
    all_reports = _registry(layout).list_evaluations(experiment_id=experiment_id)
    selected = [
        report
        for report in all_reports
        if (protocol == "shadow" and "shadow" in report.warnings)
        or (
            protocol == "product"
            and "shadow" not in report.warnings
            and report.evaluation_kind in {"product", "calibration"}
            and (task is None or report.task_id == COLLABORATION_TASK)
        )
    ]
    if not selected:
        raise ContractViolation(
            f"No immutable {protocol} evaluation evidence exists for {experiment_id}"
        )
    task_acceptance = None
    task_acceptance_path = None
    if task == "collaboration":
        task_acceptance = registry.build_collaboration_task_acceptance(experiment_id=experiment_id)
        registry.record_task_acceptance(task_acceptance)
        acceptance_directory = layout.gfm_reports / experiment_id / "task-acceptance"
        task_acceptance_path = (
            acceptance_directory / f"collaboration-product-{task_acceptance.report_hash}.json"
        )
        serialized_acceptance = _contract_json(task_acceptance)
        if task_acceptance_path.exists():
            if read_json_object(task_acceptance_path) != serialized_acceptance:
                raise ContractViolation("Immutable collaboration acceptance artifact changed")
        else:
            atomic_write_json(task_acceptance_path, serialized_acceptance)
    return {
        "schemaVersion": "gfm.workflow-evaluate/1.0",
        "ok": True,
        "experimentId": experiment_id,
        "protocol": protocol,
        "task": task,
        "generatedReportIds": [report.report_id for report in generated_reports],
        "reports": [_contract_json(report) for report in selected],
        "taskAcceptance": (None if task_acceptance is None else _contract_json(task_acceptance)),
        "taskAcceptanceArtifact": (
            None if task_acceptance_path is None else str(task_acceptance_path)
        ),
        "fullProductValidated": False if task is not None else None,
        "modelValidated": False if task is not None else None,
    }


def _build_product_suite_checkpoint(layout: RuntimeLayout, *, experiment_id: str) -> Any:
    """Bind both validated task heads into one immutable offline model candidate."""

    registry = _registry(layout)
    variant = _selected_core_variant(layout, experiment_id)
    config = _load_pretrain_config(None, None)
    seed = int(config.formal.seeds[0])
    run_id = f"{experiment_id}-product-suite-{variant}-{seed}"
    existing_run = registry.get_run(run_id)
    if existing_run is not None:
        matches = [
            value
            for value in registry.list_checkpoints(experiment_id=experiment_id)
            if value.run_id == run_id
        ]
        if len(matches) != 1:
            raise ContractViolation("Existing product suite checkpoint is ambiguous")
        return matches[0]
    runs = {
        run.run_id: run
        for run in registry.list_runs(experiment_id=experiment_id)
        if run.phase == "adapt"
        and run.status == "succeeded"
        and run.architecture_variant == variant
        and run.seed == seed
    }
    task_checkpoints: dict[ProductTask, tuple[Any, dict[str, Any]]] = {}
    for checkpoint in registry.list_checkpoints(experiment_id=experiment_id):
        if checkpoint.run_id not in runs:
            continue
        payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
        task = _product_task_from_checkpoint(payload)
        if task in task_checkpoints:
            raise ContractViolation("Product suite has ambiguous task checkpoint inputs")
        task_checkpoints[task] = (checkpoint, payload)
    if set(task_checkpoints) != {"collaboration", "newcomer"}:
        raise ContractViolation(
            "Product suite requires both task checkpoints for the same formal seed"
        )
    source_reports = registry.list_evaluations(experiment_id=experiment_id)
    direct: dict[tuple[ProductTask, str], GfmEvaluationReport] = {}
    for task, (checkpoint, _) in task_checkpoints.items():
        task_id = COLLABORATION_TASK if task == "collaboration" else NEWCOMER_TASK
        for kind in ("product", "calibration"):
            matched_reports = [
                report
                for report in source_reports
                if report.checkpoint_id == checkpoint.checkpoint_id
                and report.evaluation_kind == kind
                and report.task_id == task_id
                and "shadow" not in report.warnings
            ]
            if len(matched_reports) != 1:
                raise ContractViolation(
                    f"Product suite lacks one frozen test {kind} report for {task}"
                )
            direct[(task, kind)] = matched_reports[0]
    corpora = _load_corpus_contracts(layout)
    protocols = _register_prerequisites(layout, corpora)
    suite_config = {
        "schemaVersion": "gfm.product-suite-config/1.0",
        "architectureVariant": variant,
        "seed": seed,
        "taskCheckpoints": {
            task: {
                "checkpointId": checkpoint.checkpoint_id,
                "stateHash": checkpoint.state_hash,
                "componentStateHash": _state_digest(payload["components"]["product"]),
                "productConfigHash": payload["components"]["product_config"]["taskConfigHash"],
                "productReportHash": direct[(task, "product")].report_hash,
                "calibrationReportHash": direct[(task, "calibration")].report_hash,
            }
            for task, (checkpoint, payload) in sorted(task_checkpoints.items())
        },
        "selection": "fixed-first-formal-seed-after-base-moe-rule",
    }
    suite_config["taskConfigHash"] = canonical_sha256(suite_config)
    run_dir = layout.gfm_runs / experiment_id / run_id
    if run_dir.exists():
        raise GfmTrainingError("Unregistered product suite directory already exists")
    run_dir.mkdir(parents=True)
    checkpoint_id = f"{run_id}-best"
    checkpoint = save_gfm_checkpoint(
        run_dir / "checkpoints",
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        epoch=0,
        step=0,
        components={
            **{
                task: payload["components"]["product"]
                for task, (_, payload) in task_checkpoints.items()
            },
            **{
                f"{task}_config": payload["components"]["product_config"]
                for task, (_, payload) in task_checkpoints.items()
            },
            "suite_config": suite_config,
        },
        optimizer_state={},
        scheduler_state={},
        scaler_state={},
        sampler_state={
            "sourceCheckpointIds": tuple(
                task_checkpoints[task][0].checkpoint_id for task in sorted(task_checkpoints)
            )
        },
        best_state={
            "sourceReportHashes": tuple(
                direct[(task, kind)].report_hash
                for task in sorted(task_checkpoints)
                for kind in ("product", "calibration")
            ),
            "sourceBindings": suite_config["taskCheckpoints"],
            "expectedSuiteDigest": canonical_sha256(
                {
                    "componentStateHashes": {
                        task: _state_digest(payload["components"]["product"])
                        for task, (_, payload) in sorted(task_checkpoints.items())
                    },
                    "productConfigHashes": {
                        task: canonical_sha256(payload["components"]["product_config"])
                        for task, (_, payload) in sorted(task_checkpoints.items())
                    },
                    "suiteConfigHash": suite_config["taskConfigHash"],
                }
            ),
        },
        config=config.logical_payload(),
        corpus_hashes=tuple(corpus.logical_hash for corpus in corpora),
    )
    started = datetime.now(UTC)
    source_run = runs[task_checkpoints["collaboration"][0].run_id]
    peak = max(
        float(runs[checkpoint.run_id].peak_cuda_memory_mib or 0.0)
        for checkpoint, _ in task_checkpoints.values()
    )
    run = GfmRunManifest.create(
        runId=run_id,
        experimentId=experiment_id,
        phase="evaluate",
        architectureVariant=variant,
        status="succeeded",
        domainIds=(DOMAIN_IDS["openalex"],),
        seed=seed,
        codeHash=code_identity_hash(),
        environmentHash=source_run.environment_hash,
        configHash=config.config_hash,
        corpusHashes=tuple(corpus.logical_hash for corpus in corpora),
        taskProtocolHashes=tuple(protocol.protocol_hash for protocol in protocols),
        startedAt=started,
        finishedAt=datetime.now(UTC),
        peakCudaMemoryMiB=peak,
        artifactPaths=(str(run_dir / "checkpoints" / f"{checkpoint_id}.manifest.json"),),
    )
    _write_contract(run_dir / "run-manifest.json", run)
    atomic_write_json(run_dir / "suite-config.json", suite_config)
    registry.record_run(run)
    registry.record_checkpoint(checkpoint)
    suite_manifest_path = run_dir / "checkpoints" / f"{checkpoint_id}.manifest.json"
    verify_args = (
        "_gfm-verify-suite-checkpoint",
        "--checkpoint-manifest",
        str(suite_manifest_path),
        "--root",
        str(layout.root),
        "--json",
    )
    first, second = _subprocess_json(verify_args), _subprocess_json(verify_args)
    if first.get("verificationDigest") != second.get("verificationDigest") or first.get(
        "componentStateHashes"
    ) != second.get("componentStateHashes"):
        raise GfmTrainingError("Product suite fresh verification did not repeat")
    verification_digest = str(first["verificationDigest"])
    audit_hash, audit_path, counters = _leakage_audit(
        layout,
        experiment_id=experiment_id,
        audit_id=f"{checkpoint_id}-fresh-process",
        evidence={
            "checkpointId": checkpoint_id,
            "artifactSha256": checkpoint.artifact_sha256,
            "firstVerificationDigest": verification_digest,
            "secondVerificationDigest": second["verificationDigest"],
            "repeatMatch": True,
        },
        counters={
            name: int(
                max(source.metrics[name] for source in direct.values() if name in source.metrics)
            )
            for name in (
                "future_edge_access_count",
                "cutoff_violation_count",
                "split_overlap_count",
            )
        },
    )
    fresh_metrics = {**counters, "fresh_process_repeat_match": 1.0}
    evidence_hash, evidence_path = _evaluation_evidence(
        layout,
        experiment_id=experiment_id,
        evidence_id=f"{checkpoint_id}-fresh-process",
        payload={
            "checkpointId": checkpoint_id,
            "checkpointStateHash": checkpoint.state_hash,
            "firstVerification": first,
            "secondVerification": second,
            "metrics": fresh_metrics,
        },
    )
    registry.record_evaluation(
        GfmEvaluationReport.create(
            reportId=f"{checkpoint_id}-fresh-process",
            experimentId=experiment_id,
            runId=run_id,
            checkpointId=checkpoint_id,
            evaluationKind="fresh_process",
            domainId=DOMAIN_IDS["openalex"],
            seed=seed,
            metrics=fresh_metrics,
            evidenceArtifactHash=evidence_hash,
            evidenceArtifactPath=evidence_path,
            peakCudaMemoryMiB=peak,
            leakageAuditPassed=True,
            leakageAuditHash=audit_hash,
            leakageAuditPath=audit_path,
            freshProcessVerified=True,
            verificationDigest=verification_digest,
        )
    )
    return checkpoint


def validate_gfm(
    *,
    root: str | Path | None,
    experiment_id: str,
    scope: ValidationScope = "full",
) -> dict[str, Any]:
    """Derive immutable pretraining or full product acceptance evidence."""

    if scope not in ("pretraining", "full"):
        raise ContractViolation("GFM validation scope must be pretraining or full")

    layout = prepare_runtime_layout(root, operation="run")
    registry = _registry(layout)
    _require_experiment_runs(layout, experiment_id)
    if scope == "pretraining":
        check_all_gfm_corpora(layout.root)
        checked_corpora = _load_corpus_contracts(layout, physical_boundary=True)
        pretraining_acceptance = registry.build_pretraining_acceptance(experiment_id=experiment_id)
        if set(pretraining_acceptance.corpus_hashes) != {
            corpus.logical_hash for corpus in checked_corpora
        }:
            raise ContractViolation(
                "Pretraining acceptance differs from current checked corpus artifacts"
            )
        report_dir = layout.gfm_reports / experiment_id
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "gfm-pretraining-acceptance.json"
        _write_contract(path, pretraining_acceptance)
        registry.record_pretraining_acceptance(pretraining_acceptance)
        return {
            "schemaVersion": "gfm.workflow-validate/1.0",
            "ok": True,
            "scope": "pretraining",
            "experimentId": experiment_id,
            "accepted": pretraining_acceptance.accepted,
            "selectedVariant": pretraining_acceptance.selected_variant,
            "selectedCheckpointIds": list(pretraining_acceptance.selected_checkpoint_ids),
            "gates": dict(pretraining_acceptance.gates),
            "reasons": list(pretraining_acceptance.reasons),
            "reportHash": pretraining_acceptance.report_hash,
            "report": str(path),
        }
    checkpoint = _build_product_suite_checkpoint(layout, experiment_id=experiment_id)
    if registry.get_run(checkpoint.run_id) is None:
        raise ContractViolation("Selected checkpoint has no experiment run")
    acceptance = registry.build_acceptance(
        experiment_id=experiment_id,
        checkpoint_id=checkpoint.checkpoint_id,
    )
    report_dir = layout.gfm_reports / experiment_id
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "gfm-acceptance.json"
    _write_contract(path, acceptance)
    registry.record_acceptance(acceptance)
    return {
        "schemaVersion": "gfm.workflow-validate/1.0",
        "ok": True,
        "scope": "full",
        "experimentId": experiment_id,
        "accepted": acceptance.accepted,
        "gates": dict(acceptance.gates),
        "reasons": list(acceptance.reasons),
        "reportHash": acceptance.report_hash,
        "report": str(path),
    }


__all__ = [
    "evaluate_gfm",
    "validate_gfm",
]
