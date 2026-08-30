"""Adaptation resume, checkpoint verification, reuse, and orchestration.

The implementation is installed into the shared compatibility namespace by
:mod:`socialgraph_gfm.workflows` after all workflow stages are imported.
"""

# ruff: noqa: F403, F405
# mypy: disable-error-code=name-defined
from __future__ import annotations

from ._shared import *


def resume_gfm(*, root: str | Path | None, run_id: str, device: str = "cuda") -> dict[str, Any]:
    """Resume one interrupted run from its latest integrity-bound checkpoint."""

    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    registered = _registry(layout).get_run(run_id)
    if registered is not None:
        if registered.status == "succeeded" and registered.phase == "lodo":
            if device != "cuda":
                raise ContractViolation("Formal LODO resume requires CUDA")
            run_dir = layout.gfm_runs / registered.experiment_id / run_id
            state_path = run_dir / "run-state.json"
            if state_path.is_file():
                terminal_lag = validate_lodo_run_state(
                    read_json_object(state_path), allowed_statuses=("running", "succeeded")
                )
                if terminal_lag["status"] == "running":
                    resumed = _lodo_worker(
                        root=layout.root,
                        experiment_id=registered.experiment_id,
                        held_out_domain=str(registered.held_out_domain),
                        variant=registered.architecture_variant,
                        seed=registered.seed,
                        device=device,
                        resume_requested=True,
                    )
                    return {
                        "schemaVersion": "gfm.workflow-resume/1.0",
                        "ok": True,
                        "runId": run_id,
                        "runKind": "lodo",
                        "resumedFromCheckpointId": terminal_lag.get("latestCheckpointId"),
                        "run": resumed,
                    }
        if registered.status == "succeeded" and registered.phase == "pretrain":
            run_dir = layout.gfm_runs / registered.experiment_id / run_id
            state_path = run_dir / "run-state.json"
            if state_path.is_file():
                terminal_lag = read_json_object(state_path)
                if terminal_lag.get("status") == "running":
                    dev_prefix = "socialgraph-core-dev-"
                    formal_prefix = "socialgraph-core-formal-"
                    if registered.experiment_id.startswith(dev_prefix):
                        pretrain_phase: TrainingPhase = "dev"
                    elif registered.experiment_id.startswith(formal_prefix):
                        pretrain_phase = "formal"
                    else:
                        raise ContractViolation(
                            "Registered pretrain experiment has no fixed dev/formal identity"
                        )
                    expectation = _CompletedRunExpectation(
                        experiment_id=registered.experiment_id,
                        run_id=registered.run_id,
                        phase="pretrain",
                        variant=registered.architecture_variant,
                        seed=registered.seed,
                        domain_ids=registered.domain_ids,
                        corpus_hashes=registered.corpus_hashes,
                        protocol_hashes=registered.task_protocol_hashes,
                        config_hash=registered.config_hash,
                        code_hash=registered.code_hash,
                        environment_hash=registered.environment_hash,
                        pretrain_phase=pretrain_phase,
                        required_reports=_required_pretrain_reports(run_id, pretrain_phase),
                    )
                    completed = _validate_completed_matrix_run(layout, expectation)
                    if completed is None or completed.run_state is None:
                        raise GfmTrainingError(
                            "Registered terminal pretrain run could not be reconciled"
                        )
                    return {
                        "schemaVersion": "gfm.workflow-resume/1.0",
                        "ok": True,
                        "runId": run_id,
                        "runKind": "pretrain",
                        "resumedFromCheckpointId": None,
                        "alreadyCompleted": True,
                        "reconciledTerminalState": True,
                        "bestCheckpointId": completed.checkpoint.checkpoint_id,
                    }
        raise GfmTrainingError(
            f"Run {run_id} is already terminal ({registered.status}) and cannot be resumed"
        )
    matches = sorted(layout.gfm_runs.glob(f"*/{run_id}/run-state.json"))
    if len(matches) != 1:
        raise ContractViolation("Exactly one durable run state is required for resume")
    state = read_json_object(matches[0])
    if state.get("status") not in {"preflight", "running"}:
        raise GfmTrainingError("Only an interrupted preflight or running GFM run can be resumed")
    checkpoint_dir = matches[0].parent / "checkpoints"
    if state.get("runKind") == "lodo":
        if device != "cuda":
            raise ContractViolation("Formal LODO resume requires CUDA")
        checked_lodo = validate_lodo_run_state(state, allowed_statuses=("preflight", "running"))
        identity = checked_lodo["identity"]
        experiment_id = identity.get("experimentId")
        held_out_domain = identity.get("heldOutDomain")
        variant = identity.get("architectureVariant")
        seed = identity.get("seed")
        if (
            not isinstance(experiment_id, str)
            or not isinstance(held_out_domain, str)
            or variant not in ("core-base", "core-moe")
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or identity.get("runId") != run_id
        ):
            raise ContractViolation("Interrupted LODO cell identity is malformed")
        resumed = _lodo_worker(
            root=layout.root,
            experiment_id=experiment_id,
            held_out_domain=held_out_domain,
            variant=variant,
            seed=seed,
            device=device,
            resume_requested=True,
        )
        return {
            "schemaVersion": "gfm.workflow-resume/1.0",
            "ok": True,
            "runId": run_id,
            "runKind": "lodo",
            "resumedFromCheckpointId": checked_lodo.get("latestCheckpointId"),
            "run": resumed,
        }
    if state.get("runKind") == "product-adapt":
        if device != "cuda" or state.get("device") != "cuda":
            raise ContractViolation("Formal product adaptation resume requires CUDA")
        task = state.get("task")
        experiment_id = state.get("experimentId")
        backbone_checkpoint_id = state.get("backboneCheckpointId")
        if (
            task not in ("collaboration", "newcomer")
            or not isinstance(experiment_id, str)
            or not isinstance(backbone_checkpoint_id, str)
        ):
            raise ContractViolation("Interrupted product run identity is malformed")
        candidates: list[tuple[str, Any]] = []
        product_failures: list[str] = []
        for role, key in (
            ("latest", "latestCheckpointId"),
            ("recovery", "recoveryCheckpointId"),
        ):
            identity = state.get(key)
            if identity is None:
                continue
            if not isinstance(identity, str) or not identity.startswith(f"{run_id}-{role}-"):
                raise ContractViolation("Product progress checkpoint role is malformed")
            path = checkpoint_dir / f"{identity}.manifest.json"
            try:
                candidate = read_gfm_checkpoint_manifest(path)
                load_gfm_checkpoint(candidate, map_location="cpu")
            except Exception as error:
                product_failures.append(f"{role}:{type(error).__name__}")
                continue
            candidates.append((role, candidate))
        if not candidates:
            raise GfmTrainingError(
                "No product progress checkpoint passed integrity verification: "
                + ",".join(product_failures)
            )
        # The durable run-state names one committed latest.  Recovery is used
        # only when that exact file fails integrity; filesystem mtime/max-step
        # discovery is deliberately not an authority boundary.
        _, checkpoint = candidates[0]
        resumed = _run_product_adaptation(
            layout=layout,
            task=task,
            experiment_id=experiment_id,
            backbone_checkpoint_id=backbone_checkpoint_id,
            device=device,
            resume_manifest=(checkpoint_dir / f"{checkpoint.checkpoint_id}.manifest.json"),
        )
        return {
            "schemaVersion": "gfm.workflow-resume/1.0",
            "ok": True,
            "runId": run_id,
            "runKind": "product-adapt",
            "resumedFromCheckpointId": checkpoint.checkpoint_id,
            "run": resumed,
        }
    latest_paths = sorted(checkpoint_dir.glob(f"{run_id}-latest-*.manifest.json"))
    recovery_paths = sorted(checkpoint_dir.glob(f"{run_id}-recovery-*.manifest.json"))
    best_paths = sorted(checkpoint_dir.glob(f"{run_id}-best-*.manifest.json"))
    role_paths = {
        "latest": latest_paths,
        "recovery": recovery_paths,
        "best": best_paths,
    }
    if any(len(paths) > 1 for paths in role_paths.values()):
        raise GfmTrainingError("Interrupted run has ambiguous checkpoint roles")
    if not any(role_paths.values()):
        phase = state.get("phase")
        variant = state.get("variant")
        seed = state.get("seed")
        if (
            state.get("runKind") not in (None, "pretrain")
            or phase not in ("dev", "formal")
            or variant not in ("core-base", "core-moe")
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ContractViolation("Checkpoint-free interrupted pretrain identity is malformed")
        if phase == "formal" and device != "cuda":
            raise ContractViolation("Formal GFM resume requires CUDA")
        config = _load_pretrain_config(None, None)
        corpora, embeddings = _ensure_pretrain_evidence(
            layout,
            maximum_role="validation",
            physical_boundary=True,
        )
        protocols = _register_prerequisites(layout, corpora)
        experiment_id = _experiment_id(phase=phase, config=config, corpora=corpora)
        if state.get("experimentId") != experiment_id or state.get("runId") != run_id:
            raise ContractViolation("Interrupted GFM run identity is stale")
        resumed = _train_run(
            layout=layout,
            experiment_id=experiment_id,
            config=config,
            corpora=corpora,
            protocols=protocols,
            embeddings=embeddings,
            phase=phase,
            variant=variant,
            seed=seed,
            device=device,
            retry_without_checkpoint=True,
        )
        return {
            "schemaVersion": "gfm.workflow-resume/1.0",
            "ok": True,
            "runId": run_id,
            "resumedFromCheckpointId": None,
            "resumedFromPreflight": True,
            "run": resumed,
        }
    valid: list[tuple[int, int, Any]] = []
    priority = {"recovery": 0, "best": 1, "latest": 2}
    failures: list[str] = []
    for role, paths in role_paths.items():
        if not paths:
            continue
        try:
            candidate = read_gfm_checkpoint_manifest(paths[0])
            load_gfm_checkpoint(candidate, map_location="cpu")
        except Exception as error:
            failures.append(f"{role}:{type(error).__name__}")
            continue
        valid.append((candidate.step, priority[role], candidate))
    if not valid:
        raise GfmTrainingError(
            "No resumable checkpoint passed integrity verification: " + ",".join(sorted(failures))
        )
    _, _, checkpoint = max(valid, key=lambda item: (item[0], item[1]))
    phase = state.get("phase")
    variant = state.get("variant")
    seed = state.get("seed")
    if phase not in ("dev", "formal") or variant not in ("core-base", "core-moe"):
        raise ContractViolation("Interrupted GFM run state has invalid phase or variant")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ContractViolation("Interrupted GFM run state has an invalid seed")
    if phase == "formal" and device != "cuda":
        raise ContractViolation("Formal GFM resume requires CUDA")
    config = _load_pretrain_config(None, None)
    corpora, embeddings = _ensure_pretrain_evidence(
        layout,
        maximum_role="validation",
        physical_boundary=True,
    )
    protocols = _register_prerequisites(layout, corpora)
    experiment_id = _experiment_id(phase=phase, config=config, corpora=corpora)
    if state.get("experimentId") != experiment_id or state.get("runId") != run_id:
        raise ContractViolation("Interrupted GFM run identity is stale")
    resumed = _train_run(
        layout=layout,
        experiment_id=experiment_id,
        config=config,
        corpora=corpora,
        protocols=protocols,
        embeddings=embeddings,
        phase=phase,
        variant=variant,
        seed=seed,
        device=device,
        resume_manifest=(
            Path(checkpoint.artifact_path).parent / f"{checkpoint.checkpoint_id}.manifest.json"
        ),
    )
    return {
        "schemaVersion": "gfm.workflow-resume/1.0",
        "ok": True,
        "runId": run_id,
        "resumedFromCheckpointId": checkpoint.checkpoint_id,
        "run": resumed,
    }


def _adapt_worker(
    *,
    root: str | Path,
    task: ProductTask,
    experiment_id: str,
    backbone_checkpoint_id: str,
    device: str,
) -> dict[str, Any]:
    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    return _run_product_adaptation(
        layout=layout,
        task=task,
        experiment_id=experiment_id,
        backbone_checkpoint_id=backbone_checkpoint_id,
        device=device,
    )


def verify_gfm_product_checkpoint_fresh(
    *,
    root: str | Path | None,
    checkpoint_manifest: str | Path,
    device: str = "cpu",
) -> dict[str, Any]:
    """Rebuild one validation split and hash fixed product logits in a new process."""

    from ..gfm.model import SocialGraphFMCore
    from ..gfm.product_training import ProductTaskModule

    require_ml_runtime(device)
    layout = prepare_runtime_layout(root, operation="run")
    manifest = read_gfm_checkpoint_manifest(checkpoint_manifest)
    payload = load_gfm_checkpoint(manifest, map_location="cpu")
    raw_config = _product_config_from_checkpoint(payload)
    if not isinstance(raw_config, dict) or raw_config.get("schemaVersion") != (
        "gfm.product-adapt-config/1.0"
    ):
        raise ContractViolation("Checkpoint is not a product adaptation artifact")
    task = raw_config.get("task")
    variant = raw_config.get("architectureVariant")
    seed = raw_config.get("seed")
    if (
        task not in ("collaboration", "newcomer")
        or variant
        not in (
            "core-base",
            "core-moe",
        )
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ContractViolation("Product checkpoint configuration is malformed")
    transform = _FeatureTransform.from_dict(raw_config["featureTransform"])
    config = _load_pretrain_config(None, None)
    corpora, embeddings = _ensure_pretrain_evidence(
        layout,
        maximum_role="validation",
        physical_boundary=True,
    )
    current_task_assets = _product_task_asset_evidence(layout, task=task, corpora=corpora)
    if raw_config.get("taskAssets") != current_task_assets or raw_config.get(
        "taskAssetsHash"
    ) != canonical_sha256(current_task_assets):
        raise ContractViolation(
            "Product checkpoint task assets differ from the current immutable evidence"
        )
    stream = _make_domain_streams(layout, embeddings, maximum_role="validation")[
        DOMAIN_IDS["openalex"]
    ]
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
    prepared = _product_batches_for_split(
        task=task,
        stream=stream,
        arrays=arrays,
        newcomers=newcomers,
        split="validation",
        seed=seed,
        transform=transform,
        collaboration_kind="first" if task == "collaboration" else "both",
    )
    model = ProductTaskModule(
        SocialGraphFMCore(_model_config(config, variant)),
        task=task,
        pair_feature_dim=8,
    )
    model.load_state_dict(payload["components"]["product"])
    batch_count = 0

    def counted() -> Iterator[_PreparedProductBatch]:
        nonlocal batch_count
        for item in prepared:
            batch_count += 1
            yield item

    values = _product_logits(
        model,
        counted(),
        device=device,
        baseline_config=raw_config.get("collaborationBaseline"),
    )
    summary = {
        "pairLogitHash": canonical_sha256(values["pair_logits"].tolist()),
        "pairLabelHash": canonical_sha256(values["pair_labels"].tolist()),
        "participationLogitHash": canonical_sha256(values["participation_logits"].tolist()),
        "batchCount": batch_count,
        "temperature": float(raw_config["temperature"]),
    }
    fixed_sample_digest = canonical_sha256(
        {
            "modelState": _state_digest(model.state_dict()),
            "validationPairLogits": summary["pairLogitHash"],
            "validationParticipationLogits": summary["participationLogitHash"],
            "productConfigHash": raw_config["taskConfigHash"],
        }
    )
    if fixed_sample_digest != payload.get("best_state", {}).get("expectedFixedSampleDigest"):
        raise GfmTrainingError(
            "Fresh product outputs differ from the digest frozen at training time"
        )
    digest = canonical_sha256(
        {
            "checkpointId": manifest.checkpoint_id,
            "stateHash": manifest.state_hash,
            "fixedSampleDigest": fixed_sample_digest,
        }
    )
    return {
        "schemaVersion": "gfm.product-fresh-verification/1.0",
        "ok": True,
        "checkpointId": manifest.checkpoint_id,
        "runId": manifest.run_id,
        "seed": seed,
        "summary": summary,
        "fixedSampleDigest": fixed_sample_digest,
        "verificationDigest": digest,
    }


def verify_gfm_suite_checkpoint_fresh(
    *, root: str | Path | None, checkpoint_manifest: str | Path
) -> dict[str, Any]:
    """Weights-only suite verification intended for an isolated process."""

    prepare_runtime_layout(root, operation="run")
    checkpoint = read_gfm_checkpoint_manifest(checkpoint_manifest)
    payload = load_gfm_checkpoint(checkpoint, map_location="cpu")
    config = payload.get("components", {}).get("suite_config")
    expected_components = {
        "collaboration",
        "collaboration_config",
        "newcomer",
        "newcomer_config",
        "suite_config",
    }
    if (
        not isinstance(config, dict)
        or config.get("schemaVersion") != "gfm.product-suite-config/1.0"
        or set(payload["components"]) != expected_components
    ):
        raise ContractViolation("Checkpoint is not the fixed two-task product suite")
    checked_suite_config = dict(config)
    suite_config_hash = checked_suite_config.pop("taskConfigHash", None)
    if suite_config_hash != canonical_sha256(checked_suite_config):
        raise ContractViolation("Embedded product suite configuration hash is invalid")
    component_hashes = {
        name: _state_digest(payload["components"][name]) for name in ("collaboration", "newcomer")
    }
    config_hashes = {
        name: canonical_sha256(payload["components"][f"{name}_config"])
        for name in ("collaboration", "newcomer")
    }
    expected_digest = payload.get("best_state", {}).get("expectedSuiteDigest")
    fixed_digest = canonical_sha256(
        {
            "componentStateHashes": component_hashes,
            "productConfigHashes": config_hashes,
            "suiteConfigHash": suite_config_hash,
        }
    )
    if fixed_digest != expected_digest:
        raise ContractViolation("Suite fixed digest differs from its frozen best state")
    digest = canonical_sha256(
        {
            "checkpointId": checkpoint.checkpoint_id,
            "stateHash": checkpoint.state_hash,
            "componentStateHashes": component_hashes,
            "productConfigHashes": config_hashes,
            "fixedSuiteDigest": fixed_digest,
        }
    )
    return {
        "schemaVersion": "gfm.product-suite-fresh-verification/1.0",
        "ok": True,
        "checkpointId": checkpoint.checkpoint_id,
        "componentStateHashes": component_hashes,
        "productConfigHashes": config_hashes,
        "fixedSuiteDigest": fixed_digest,
        "verificationDigest": digest,
    }


def _reuse_adapt_matrix_cell(
    *,
    layout: RuntimeLayout,
    experiment_id: str,
    task: ProductTask,
    backbone: Any,
    corpora: Sequence[GfmDomainCorpusManifest],
    protocol: GfmTaskProtocolManifest,
) -> dict[str, Any] | None:
    backbone_run = _registry(layout).get_run(backbone.run_id)
    if backbone_run is None:
        raise ContractViolation("Product backbone run disappeared from the registry")
    run_id = f"{experiment_id}-adapt-{task}-{backbone_run.architecture_variant}-{backbone_run.seed}"
    evidence = _validate_completed_matrix_run(
        layout,
        _CompletedRunExpectation(
            experiment_id=experiment_id,
            run_id=run_id,
            phase="adapt",
            variant=backbone_run.architecture_variant,
            seed=backbone_run.seed,
            domain_ids=(DOMAIN_IDS["openalex"],),
            corpus_hashes=tuple(corpus.logical_hash for corpus in corpora),
            protocol_hashes=(protocol.protocol_hash,),
            config_hash=backbone_run.config_hash,
            code_hash=code_identity_hash(),
            environment_hash=_environment_hash("cuda"),
            required_reports=((f"{run_id}-fresh-process", "fresh_process"),),
        ),
    )
    if evidence is None:
        return None
    config = _product_config_from_checkpoint(evidence.checkpoint_payload)
    current_task_assets = _product_task_asset_evidence(layout, task=task, corpora=corpora)
    best = evidence.checkpoint_payload.get("best_state")
    state = evidence.run_state
    if (
        config.get("task") != task
        or config.get("backboneCheckpointId") != backbone.checkpoint_id
        or config.get("backboneStateHash") != backbone.state_hash
        or config.get("taskAssets") != current_task_assets
        or config.get("taskAssetsHash") != canonical_sha256(current_task_assets)
        or not isinstance(best, dict)
        or state is None
        or state.get("task") != task
        or state.get("backboneCheckpointId") != backbone.checkpoint_id
        or state.get("backboneStateHash") != backbone.state_hash
    ):
        raise GfmTrainingError(
            f"Completed product run {run_id} differs from its selected backbone/task"
        )
    validation_loss = best.get("validationLoss")
    temperature = config.get("temperature")
    transform = config.get("featureTransform")
    if (
        isinstance(validation_loss, bool)
        or not isinstance(validation_loss, (int, float))
        or not math.isfinite(float(validation_loss))
        or isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
        or not isinstance(transform, dict)
    ):
        raise GfmTrainingError(f"Completed product metrics are invalid for {run_id}")
    fresh = evidence.reports[f"{run_id}-fresh-process"]
    return {
        "runId": run_id,
        "checkpointId": evidence.checkpoint.checkpoint_id,
        "seed": backbone_run.seed,
        "variant": backbone_run.architecture_variant,
        "task": task,
        "bestStep": evidence.checkpoint.step,
        "completedSteps": int(state.get("completedSteps", evidence.checkpoint.step)),
        "bestValidationLoss": float(validation_loss),
        "temperature": float(temperature),
        "featureTransformHash": canonical_sha256(transform),
        "peakCudaMemoryMiB": evidence.run.peak_cuda_memory_mib,
        "testRead": False,
        "streamingBatches": True,
        "freshProcessDigest": fresh.verification_digest,
        "reused": True,
    }


def adapt_gfm(
    *,
    root: str | Path | None,
    task: ProductTask,
    experiment_id: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run three isolated, cutoff-safe product fine-tuning processes."""

    if task not in ("collaboration", "newcomer"):
        raise ContractViolation("GFM adaptation task must be collaboration or newcomer")
    require_ml_runtime("cuda")
    layout = prepare_runtime_layout(root, operation="run")
    _require_experiment_runs(layout, experiment_id)
    arrays = load_domain_view(
        layout.root,
        DOMAIN_IDS["openalex"],
        maximum_role="validation",
        families=("targets",),
    )["arrays"]
    corpora, validation_embeddings = _ensure_pretrain_evidence(
        layout,
        maximum_role="validation",
        physical_boundary=True,
    )
    validation_stream = _make_domain_streams(
        layout,
        validation_embeddings,
        domain_ids=(DOMAIN_IDS["openalex"],),
        maximum_role="validation",
    )[DOMAIN_IDS["openalex"]]
    newcomer_overlay: Mapping[str, np.ndarray] | None = None
    if task == "newcomer":
        newcomer_overlay = load_openalex_newcomers_view(layout.root, maximum_role="validation")[
            "arrays"
        ]
        if newcomer_overlay is None:
            raise ContractViolation("Newcomer adaptation requires a verified overlay")
        if not bool(np.asarray(newcomer_overlay["history_verified"], dtype=np.bool_).all()):
            raise ContractViolation(
                "Newcomer adaptation requires complete full-history verification"
            )
    elif not bool(_openalex_array(arrays, "first_collaboration").astype(np.bool_).any()):
        raise ContractViolation(
            "Collaboration adaptation requires first-collaboration target events"
        )
    variant = _selected_core_variant(layout, experiment_id)
    backbones = _formal_backbones(layout, experiment_id=experiment_id, variant=variant)
    config = _load_pretrain_config(None, None)
    if seed is not None and seed not in config.formal.seeds:
        raise ContractViolation("Requested adaptation seed is outside the formal config")
    selected_backbones = []
    for backbone in backbones:
        backbone_run = _registry(layout).get_run(backbone.run_id)
        if backbone_run is None:
            raise ContractViolation("Formal backbone run disappeared from the registry")
        if seed is None or backbone_run.seed == seed:
            selected_backbones.append(backbone)
    backbones = tuple(selected_backbones)
    if not backbones:
        raise ContractViolation("No formal backbone matches the requested adaptation seed")
    task_id = COLLABORATION_TASK if task == "collaboration" else NEWCOMER_TASK
    protocol = next(value for value in _task_protocols() if value.task_id == task_id)
    results: list[dict[str, Any]] = []
    registry = _registry(layout)
    reused_count = 0
    for backbone in backbones:
        reused = _reuse_adapt_matrix_cell(
            layout=layout,
            experiment_id=experiment_id,
            task=task,
            backbone=backbone,
            corpora=corpora,
            protocol=protocol,
        )
        if reused is not None:
            results.append(reused)
            reused_count += 1
            continue
        worker = _subprocess_json(
            (
                "_gfm-adapt-run",
                "--task",
                task,
                "--experiment-id",
                experiment_id,
                "--backbone-checkpoint-id",
                backbone.checkpoint_id,
                "--device",
                "cuda",
                "--root",
                str(layout.root),
                "--json",
            )
        )
        run = worker.get("run")
        if not isinstance(run, dict):
            raise GfmTrainingError("Independent product worker omitted its run summary")
        checkpoint_id = str(run["checkpointId"])
        manifest_path = (
            layout.gfm_runs
            / experiment_id
            / str(run["runId"])
            / "checkpoints"
            / f"{checkpoint_id}.manifest.json"
        )
        verify_args = (
            "_gfm-verify-product-checkpoint",
            "--checkpoint-manifest",
            str(manifest_path),
            "--device",
            "cpu",
            "--root",
            str(layout.root),
            "--json",
        )
        first, second = _subprocess_json(verify_args), _subprocess_json(verify_args)
        if first.get("verificationDigest") != second.get("verificationDigest") or first.get(
            "summary"
        ) != second.get("summary"):
            raise GfmTrainingError("Product checkpoint fresh verification did not repeat")
        audit_hash, audit_path, counters = _leakage_audit(
            layout,
            experiment_id=experiment_id,
            audit_id=f"{run['runId']}-fresh-process",
            evidence={
                "checkpointId": checkpoint_id,
                "checkpointManifestSha256": file_sha256(manifest_path),
                "firstVerificationDigest": first["verificationDigest"],
                "secondVerificationDigest": second["verificationDigest"],
                "split": "validation-only",
                "repeatMatch": True,
            },
            counters=_product_audit_counters(
                _product_batches_for_split(
                    task=task,
                    stream=validation_stream,
                    arrays=arrays,
                    newcomers=newcomer_overlay,
                    split="validation",
                    seed=int(run["seed"]),
                    transform=_FeatureTransform.from_dict(
                        _product_config_from_checkpoint(
                            load_gfm_checkpoint(
                                read_gfm_checkpoint_manifest(manifest_path),
                                map_location="cpu",
                            )
                        )["featureTransform"]
                    ),
                    collaboration_kind=("first" if task == "collaboration" else "both"),
                )
            ),
        )
        registered_run = registry.get_run(str(run["runId"]))
        if registered_run is None:
            raise GfmTrainingError("Product worker did not register its run")
        fresh_metrics = {
            **counters,
            "fresh_process_repeat_match": 1.0,
            "fixed_validation_batch_count": float(first["summary"]["batchCount"]),
        }
        evidence_hash, evidence_path = _evaluation_evidence(
            layout,
            experiment_id=experiment_id,
            evidence_id=f"{run['runId']}-fresh-process",
            payload={
                "checkpointId": checkpoint_id,
                "firstVerification": first,
                "secondVerification": second,
                "metrics": fresh_metrics,
            },
        )
        report = GfmEvaluationReport.create(
            reportId=f"{run['runId']}-fresh-process",
            experimentId=experiment_id,
            runId=run["runId"],
            checkpointId=checkpoint_id,
            evaluationKind="fresh_process",
            domainId=DOMAIN_IDS["openalex"],
            seed=int(run["seed"]),
            metrics=fresh_metrics,
            evidenceArtifactHash=evidence_hash,
            evidenceArtifactPath=evidence_path,
            peakCudaMemoryMiB=registered_run.peak_cuda_memory_mib,
            leakageAuditPassed=True,
            leakageAuditHash=audit_hash,
            leakageAuditPath=audit_path,
            freshProcessVerified=True,
            verificationDigest=first["verificationDigest"],
        )
        registry.record_evaluation(report)
        run["freshProcessDigest"] = first["verificationDigest"]
        results.append(run)
    return {
        "schemaVersion": "gfm.workflow-adapt/1.0",
        "ok": True,
        "experimentId": experiment_id,
        "task": task,
        "architectureVariant": variant,
        "independentProcesses": True,
        "runs": results,
        "testRead": False,
        "selection": {"seeds": [int(value["seed"]) for value in results]},
        "reusedRunCount": reused_count,
    }


__all__ = [
    "adapt_gfm",
    "resume_gfm",
    "verify_gfm_product_checkpoint_fresh",
    "verify_gfm_suite_checkpoint_fresh",
]
