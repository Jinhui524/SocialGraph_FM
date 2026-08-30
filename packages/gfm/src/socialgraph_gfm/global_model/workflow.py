"""Immutable SocialGraph-FM Global conversion, training, evaluation and release workflow."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np

from socialgraph_gfm.canonical import canonical_json, canonical_sha256, file_sha256
from socialgraph_gfm.gfm.corpus.common import atomic_write_json, resolve_within

from .calibration import (
    BinaryLogitCalibrator,
    binary_ece,
    calibration_state,
    fit_binary_logit_calibrator,
    select_country_balanced_threshold,
)
from .config import (
    COUNTRIES,
    RELEASE_ID,
    SEED,
    SOURCE_COUNTRIES,
    SPLIT_INDEX,
    TASK_ID,
    ProtocolId,
    ProtocolPlan,
    global_model_root_from_home,
    load_global_model_config,
    release_identity,
)
from .contracts import (
    COUNTRY_IDS,
    GRAPH_STAT_NAMES,
    TRACE_NAMES,
    CountryId,
    read_country_manifest,
)
from .corpus import GlobalCorpusIndex, GlobalCountryCorpus, load_corpus_index
from .model import GlobalModel, GlobalModelConfig
from .training import (
    DomainData,
    InferenceData,
    TrainingOptions,
    collect_masked_outputs,
    collect_split_logits,
    options_from_config,
    tensor_state_hash,
    train_balanced_neighbor_model,
)

PROTOCOLS: tuple[ProtocolId, ...] = ("in_domain", "low_label", "cross_domain", "global")
TRAINING_SCHEMA = "socialgraph-fm.global-model-training/1.0"
EVALUATION_SCHEMA = "socialgraph-fm.global-model-evaluation/1.0"
EXPORT_SCHEMA = "socialgraph-fm.global-model-export/1.0"
SMOKE_SCHEMA = "socialgraph-fm.global-model-smoke/1.0"
REGISTRY_SCHEMA = "socialgraph-fm.global-model-registry/1.0"
TEST_ACCESS_SCHEMA = "socialgraph-fm.global-model-test-access/1.0"
MODEL_CARD_SCHEMA = "socialgraph-fm.global-model-card/1.0"
_UPSTREAM_REFERENCE_NAME = "InfoOps" + "GFM"
_UPSTREAM_REFERENCE_URL = "https://github.com/mminici/" + _UPSTREAM_REFERENCE_NAME


def _allowed_experts(protocol: ProtocolId) -> tuple[str, ...]:
    if protocol in {"in_domain", "low_label"}:
        return ("domain:russia", "null")
    domains = SOURCE_COUNTRIES if protocol == "cross_domain" else COUNTRIES
    return (*(f"domain:{country}" for country in domains), "null")


def _safe_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("Global root must not be a filesystem root")
    return root


def _write_hashed(path: Path, payload: dict[str, Any], hash_field: str) -> Path:
    if hash_field in payload:
        raise ValueError(f"payload already contains {hash_field}")
    document = {**payload, hash_field: canonical_sha256(payload)}
    atomic_write_json(path, document)
    return path


def _write_hashed_exclusive(
    path: Path, payload: dict[str, Any], hash_field: str
) -> dict[str, Any]:
    """Create an audit claim once; a partial file still blocks unsafe reuse."""

    if hash_field in payload:
        raise ValueError(f"payload already contains {hash_field}")
    document = {**payload, hash_field: canonical_sha256(payload)}
    encoded = (canonical_json(document) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    # A partial claim deliberately remains present and fail-closed after a write failure.
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return document


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read_hashed(path: Path, *, schema: str, hash_field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schemaVersion") != schema:
        raise ValueError(f"unsupported Global artifact schema at {path}")
    observed = payload.get(hash_field)
    logical = {key: value for key, value in payload.items() if key != hash_field}
    if observed != canonical_sha256(logical):
        raise ValueError(f"Global artifact hash mismatch at {path}")
    return payload


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Global artifact escapes its root: {path}") from exc


def _split_id(variant: str) -> str:
    regime = "full" if variant == "base" else variant.removesuffix("U")
    return f"{regime}-fold-{SPLIT_INDEX}"


def _country_split(country: GlobalCountryCorpus, variant: str):
    split = country.split(_split_id(variant))
    expected_regime = "full" if variant == "base" else variant.removesuffix("U")
    if split.descriptor.regime != expected_regime or split.descriptor.fold != SPLIT_INDEX:
        raise ValueError("Global split descriptor does not match the fixed protocol")
    return split


def _model_config(config: Mapping[str, Any]) -> GlobalModelConfig:
    model = config["model"]
    if model.get("gnnLayers") != 2 or model.get("routerExperts") != 8:
        raise ValueError("SocialGraph-FM Global requires two GraphSAGE layers and eight router experts")
    return GlobalModelConfig(
        text_dim=int(model["textDim"]),
        structural_dim=int(model["structuralDim"]),
        branch_dim=int(model["branchDim"]),
        hidden_dim=int(model["hiddenDim"]),
        dropout=float(model["dropout"]),
        domains=COUNTRY_IDS,
        router_enabled=bool(model["routerEnabled"]),
        router_bottleneck_dim=int(model["routerBottleneckDim"]),
        router_top_k=int(model["routerTopK"]),
    )


def _source_inventory(index: GlobalCorpusIndex) -> dict[str, dict[str, str]]:
    return {
        country: dict(sorted(index.entries[cast(CountryId, country)].source_hashes.items()))
        for country in COUNTRIES
    }


def _split_inventory(index: GlobalCorpusIndex) -> dict[str, dict[str, str]]:
    return {
        country: dict(sorted(index.entries[cast(CountryId, country)].split_hashes.items()))
        for country in COUNTRIES
    }


def convert_global_model_corpus(
    *,
    source_root: str | Path,
    root: str | Path,
    trusted_source: bool,
    include_all_regimes: bool = False,
) -> Path:
    """Offline-convert the official trusted pickle/Torch sources to safe arrays."""

    from .contracts import read_country_manifest
    from .converter import (
        OFFICIAL_REGIMES,
        convert_country_in_worker,
        publish_corpus_manifest,
        require_conversion_disk_space,
    )

    if not trusted_source:
        raise ValueError("Global pickle conversion requires explicit trusted_source=True")
    source = Path(source_root).expanduser().resolve()
    destination = _safe_root(root) / "corpus"
    regimes = OFFICIAL_REGIMES if include_all_regimes else ("full", "0.95")
    required_split_ids = {f"{regime}-fold-{SPLIT_INDEX}" for regime in regimes}
    if (destination / "manifest.json").is_file():
        existing = load_corpus_index(destination, verify_manifests=True)
        for country_id in COUNTRY_IDS:
            available = set(existing.entries[country_id].split_hashes)
            if not required_split_ids.issubset(available):
                raise ValueError(
                    f"Global existing corpus lacks requested splits for {country_id}"
                )
        return destination / "manifest.json"
    require_conversion_disk_space(destination)
    destination.mkdir(parents=True, exist_ok=True)
    suffix = {
        "full": "",
        "0.5": "_0.5U",
        "0.75": "_0.75U",
        "0.9": "_0.9U",
        "0.95": "_0.95U",
        "0.99": "_0.99U",
        "0.999": "_0.999U",
    }
    country_paths: dict[CountryId, Path] = {}
    for country_id in COUNTRY_IDS:
        source_country = source / country_id
        country_root = destination / "countries" / country_id
        manifest_path = country_root / "manifest.json"
        if manifest_path.is_file():
            manifest = read_country_manifest(manifest_path)
            if (
                manifest.country_id != country_id
                or not required_split_ids.issubset(manifest.split_hashes)
            ):
                raise ValueError(f"Global existing country identity mismatch: {country_id}")
        else:
            receipt = convert_country_in_worker(
                country_id=country_id,
                source_root=source,
                destination_root=destination,
                pickle_sources={
                    regime: source_country / f"0.7_datasets.pkl{suffix[regime]}"
                    for regime in regimes
                },
                text_tensor_path=source_country / "sbert_nodeattributes_mostPop5.pt",
                destination=country_root,
                trusted_source=True,
            )
            if (
                receipt.country_id != country_id
                or receipt.manifest_path.resolve() != manifest_path.resolve()
                or receipt.manifest_hash != read_country_manifest(manifest_path).content_hash
            ):
                raise ValueError(f"Global worker receipt mismatch: {country_id}")
        country_paths[country_id] = manifest_path
    publish_corpus_manifest(destination, country_manifest_paths=country_paths)
    load_corpus_index(destination, verify_manifests=True)
    return destination / "manifest.json"


def validate_global_model_corpus(root: str | Path) -> Path:
    selected = _safe_root(root)
    index = load_corpus_index(selected / "corpus", verify_manifests=True)
    countries = []
    for country_id in COUNTRY_IDS:
        corpus = index.load_country(
            country_id, verify_hashes=True, verify_values=True, mmap_mode="r"
        )
        countries.append(
            {
                "country": country_id,
                "contentHash": corpus.manifest.content_hash,
                "nodeCount": corpus.manifest.node_count,
                "edgeCount": corpus.manifest.edge_count,
                "splitHashes": corpus.manifest.split_hashes,
            }
        )
    report = {
        "schemaVersion": "socialgraph-fm.global-model-corpus-validation/1.0",
        "releaseId": RELEASE_ID,
        "corpusHash": index.manifest.content_hash,
        "sourceHashes": _source_inventory(index),
        "splitHashes": _split_inventory(index),
        "countries": countries,
        "passed": True,
    }
    return _write_hashed(
        selected / "reports" / "corpus-validation.json", report, "validationHash"
    )


def _run_context(
    root: Path, protocol: ProtocolId, *, fast: bool
) -> tuple[GlobalCorpusIndex, dict[str, Any], ProtocolPlan, TrainingOptions, dict[str, str], str]:
    config = load_global_model_config()
    plan = config["protocolPlans"][protocol]
    options = options_from_config(config, fast=fast)
    index = load_corpus_index(root / "corpus", verify_manifests=True)
    identity = {
        **release_identity(),
        "corpusHash": index.manifest.content_hash,
        "protocolPlanHash": canonical_sha256(plan.to_dict()),
        "trainingOptionsHash": canonical_sha256(options.__dict__),
    }
    run_hash = canonical_sha256(
        {"releaseId": RELEASE_ID, "protocol": protocol, "fast": fast, **identity}
    )
    protocol_token = protocol.replace("_", "-")
    run_id = f"global-model-{protocol_token}{'-fast' if fast else ''}-{run_hash[:16]}"
    return index, config, plan, options, identity, run_id


def _training_domains(index: GlobalCorpusIndex, plan: ProtocolPlan) -> dict[str, DomainData]:
    if plan.train_domains != plan.selection_domains:
        raise ValueError("Global training and selection country inventories must match")
    domains: dict[str, DomainData] = {}
    for train_ref, select_ref in zip(plan.train, plan.select, strict=True):
        if train_ref.country != select_ref.country:
            raise ValueError("Global train/selection countries must align")
        country_id = cast(CountryId, train_ref.country)
        corpus = index.load_country(
            country_id, verify_hashes=True, verify_values=True, mmap_mode="r"
        )
        train_split = _country_split(corpus, train_ref.variant)
        validation_split = _country_split(corpus, select_ref.variant)
        domains[train_ref.country] = DomainData(
            country=train_ref.country,
            edge_index=corpus.edge_index,
            text_features=corpus.text_features,
            structural_features=corpus.degree_bucket,
            labels=corpus.labels,
            train_mask=train_split.train_mask,
            validation_mask=validation_split.validation_mask,
            structure_missing=corpus.structure_missing,
            graph_stats=corpus.graph_stats,
            source_hashes=corpus.manifest.source_hashes,
            train_split_hash=train_split.descriptor.split_hash,
            validation_split_hash=validation_split.descriptor.split_hash,
        )
    if plan.protocol == "cross_domain" and tuple(domains) != SOURCE_COUNTRIES:
        raise ValueError("Cross-domain trainer received a target-domain corpus")
    return domains


def _calibration_report(
    model: GlobalModel,
    domains: Mapping[str, DomainData],
    *,
    options: TrainingOptions,
    device: str,
    allowed_experts: tuple[str, ...],
) -> tuple[dict[str, Any], BinaryLogitCalibrator, dict[str, Any]]:
    import torch

    selected_device = torch.device(device)
    logits, labels = collect_split_logits(
        model,
        domains,
        split="validation",
        options=options,
        device=selected_device,
        allowed_experts=allowed_experts,
    )
    joined_logits = torch.cat([logits[country] for country in domains])
    joined_labels = torch.cat([labels[country] for country in domains])
    fit = fit_binary_logit_calibrator(joined_logits, joined_labels)
    threshold = select_country_balanced_threshold(
        logits, labels, calibrator=fit.calibrator
    )
    report = {
        **calibration_state(fit.calibrator),
        "fitRole": "validation-only",
        "countries": list(domains),
        "sampleCount": fit.sample_count,
        "beforeNll": fit.before_loss,
        "afterNll": fit.after_loss,
        "eceBefore": binary_ece(joined_logits, joined_labels),
        "eceAfter": binary_ece(
            joined_logits, joined_labels, calibrator=fit.calibrator
        ),
        "threshold": threshold.threshold,
        "thresholdMetric": "country-balanced-macro-f1",
        "countryBalancedMacroF1": threshold.mean_macro_f1,
        "perCountryMacroF1": dict(threshold.per_country_macro_f1),
        "candidateCount": threshold.candidate_count,
    }
    return report, fit.calibrator, {"logits": logits, "labels": labels}


def train_global_model_protocol(
    root: str | Path,
    *,
    protocol: ProtocolId,
    device: str = "cuda",
    resume: bool = True,
    fast: bool = False,
    on_step_complete: Any = None,
) -> Path:
    selected = _safe_root(root)
    index, config, plan, options, identity, run_id = _run_context(
        selected, protocol, fast=fast
    )
    run_dir = selected / "runs" / run_id
    manifest_path = run_dir / "training-manifest.json"
    if manifest_path.is_file():
        manifest = _read_hashed(
            manifest_path, schema=TRAINING_SCHEMA, hash_field="trainingHash"
        )
        if manifest.get("identity") != identity:
            raise ValueError("Global completed training identity is stale")
        return manifest_path
    domains = _training_domains(index, plan)
    model_config = _model_config(config)
    model = GlobalModel(model_config)
    allowed_experts = _allowed_experts(protocol)
    outcome = train_balanced_neighbor_model(
        model,
        domains,
        run_dir=run_dir,
        protocol=protocol,
        identity=identity,
        options=options,
        allowed_experts=allowed_experts,
        device=device,
        resume=resume,
        on_step_complete=on_step_complete,
    )
    calibration, _calibrator, _validation = _calibration_report(
        model,
        domains,
        options=options,
        device=device,
        allowed_experts=allowed_experts,
    )
    labelled_train_nodes = {
        country: int(np.asarray(domain.train_mask, dtype=np.bool_).sum())
        for country, domain in domains.items()
    }
    target_access = {
        "country": "russia",
        "labelsOrMasksPassedToTrainer": "russia" in domains,
        "labelsOrMasksUsedForSelection": "russia" in domains,
        "labelsOrMasksUsedForCalibration": "russia" in domains,
        "frozenTargetEvaluations": 0,
    }
    if protocol == "cross_domain" and any(
        target_access[key]
        for key in (
            "labelsOrMasksPassedToTrainer",
            "labelsOrMasksUsedForSelection",
            "labelsOrMasksUsedForCalibration",
        )
    ):
        raise RuntimeError("Cross-domain target labels entered a pre-freeze stage")
    manifest = {
        "schemaVersion": TRAINING_SCHEMA,
        "releaseId": RELEASE_ID,
        "taskId": TASK_ID,
        "state": "trained",
        "runId": run_id,
        "protocol": protocol,
        "seed": SEED,
        "splitIndex": SPLIT_INDEX,
        "domains": list(domains),
        "identity": identity,
        "corpusHash": index.manifest.content_hash,
        "sourceHashes": {
            country: dict(sorted(domain.source_hashes.items()))
            for country, domain in domains.items()
        },
        "splitHashes": {
            country: {
                "train": domain.train_split_hash,
                "validation": domain.validation_split_hash,
            }
            for country, domain in domains.items()
        },
        "codeHash": identity["codeHash"],
        "runtimeLockHash": identity["runtimeLockHash"],
        "protocolPlan": plan.to_dict(),
        "config": {
            "configHash": config["configSha256"],
            "model": config["model"],
            "training": options.__dict__,
        },
        "calibration": calibration,
        "metrics": {
            "validationCountryBalancedMacroF1": calibration[
                "countryBalancedMacroF1"
            ]
        },
        "labelledTrainNodes": labelled_train_nodes,
        "labelledTrainNodeCount": sum(labelled_train_nodes.values()),
        "targetAccess": target_access,
        "allowedExperts": list(allowed_experts),
        "expertNames": list(model.router.expert_names if model.router is not None else ()),
        "artifacts": {
            "checkpointPath": _relative(selected, outcome.best_checkpoint_path),
            "checkpointSha256": outcome.checkpoint_sha256,
            "latestCheckpointPath": _relative(selected, outcome.checkpoint_path),
            "latestCheckpointSha256": file_sha256(outcome.checkpoint_path),
            "modelStateHash": outcome.model_state_hash,
        },
        "training": {
            "bestStep": outcome.best_step,
            "stepsCompleted": outcome.steps_completed,
            "globalStep": outcome.global_step,
            "bestValidationMacroF1": outcome.best_validation_macro_f1,
            "stoppedEarly": outcome.stopped_early,
            "resumedFromStep": outcome.resumed_from_step,
            "ampEnabled": outcome.amp_enabled,
            "device": device,
            "memorySmoke": outcome.memory_smoke,
            "history": list(outcome.history),
        },
    }
    return _write_hashed(manifest_path, manifest, "trainingHash")


def _training_manifest_path(
    root: Path, protocol: ProtocolId, *, fast: bool
) -> tuple[Path, GlobalCorpusIndex, dict[str, Any], ProtocolPlan, TrainingOptions, dict[str, str]]:
    index, config, plan, options, identity, run_id = _run_context(
        root, protocol, fast=fast
    )
    return (
        root / "runs" / run_id / "training-manifest.json",
        index,
        config,
        plan,
        options,
        identity,
    )


def _load_trained_model(
    root: Path, protocol: ProtocolId, *, fast: bool, device: str
) -> tuple[
    GlobalModel,
    dict[str, Any],
    GlobalCorpusIndex,
    ProtocolPlan,
    TrainingOptions,
    BinaryLogitCalibrator,
]:
    import torch

    manifest_path, index, config, plan, options, identity = _training_manifest_path(
        root, protocol, fast=fast
    )
    manifest = _read_hashed(
        manifest_path, schema=TRAINING_SCHEMA, hash_field="trainingHash"
    )
    if manifest.get("identity") != identity:
        raise ValueError("Global training manifest identity is stale")
    artifact = manifest["artifacts"]
    checkpoint = resolve_within(root, artifact["checkpointPath"])
    if file_sha256(checkpoint) != artifact["checkpointSha256"]:
        raise ValueError("Global best checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("schemaVersion") != "socialgraph-fm.global-model-best-checkpoint/1.0":
        raise ValueError("unsupported Global best checkpoint")
    if payload.get("identity") != identity or payload.get("protocol") != protocol:
        raise ValueError("Global best checkpoint binding mismatch")
    state = payload["modelState"]
    if (
        tensor_state_hash(state) != artifact["modelStateHash"]
        or payload.get("modelStateHash") != artifact["modelStateHash"]
    ):
        raise ValueError("Global model state hash mismatch")
    model = GlobalModel(_model_config(config))
    model.load_state_dict(state)
    model.to(torch.device(device)).eval()
    model.requires_grad_(False)
    calibration = manifest["calibration"]
    calibrator = BinaryLogitCalibrator(
        temperature=float(calibration["temperature"]),
        bias=float(calibration["bias"]),
    ).eval()
    calibrator.requires_grad_(False)
    return model, manifest, index, plan, options, calibrator


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order].astype(np.int64, copy=False)
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(precision[ordered == 1].sum() / positives)


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    positive_count = int(positives.sum())
    negative_count = int(negatives.sum())
    if positive_count == 0 or negative_count == 0:
        return 0.0
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    cursor = 0
    while cursor < order.size:
        end = cursor + 1
        while end < order.size and scores[order[end]] == scores[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + 1 + end) / 2.0
        cursor = end
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _binary_metrics(
    scores: np.ndarray, labels: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    if scores.ndim != 1 or labels.shape != scores.shape or scores.size == 0:
        raise ValueError("Global metrics require aligned nonempty vectors")
    if not np.isfinite(scores).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError("Global metric inputs are invalid")
    predicted = scores >= threshold
    truth = labels.astype(np.bool_, copy=False)
    tp = int(np.logical_and(predicted, truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    positive_f1 = f1(tp, fp, fn)
    negative_f1 = f1(tn, fn, fp)
    return {
        "sampleCount": int(scores.size),
        "positiveCount": int(truth.sum()),
        "macroF1": (positive_f1 + negative_f1) / 2.0,
        "prAuc": _average_precision(scores, labels),
        "rocAuc": _roc_auc(scores, labels),
        "accuracy": (tp + tn) / scores.size,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _modality_counts(corpus: GlobalCountryCorpus) -> np.ndarray:
    counts = np.zeros((corpus.manifest.node_count, len(TRACE_NAMES)), dtype=np.int32)
    for column, trace_name in enumerate(TRACE_NAMES):
        relation = corpus.relation(trace_name)
        counts[:, column] = np.diff(
            np.asarray(relation.indptr, dtype=np.int64)
        ).astype(np.int32, copy=False)
    return counts


def _evaluation_inputs(
    index: GlobalCorpusIndex, plan: ProtocolPlan
) -> tuple[dict[str, InferenceData], dict[str, GlobalCountryCorpus], dict[str, np.ndarray]]:
    inputs: dict[str, InferenceData] = {}
    corpora: dict[str, GlobalCountryCorpus] = {}
    test_masks: dict[str, np.ndarray] = {}
    for reference in plan.evaluate:
        country_id = cast(CountryId, reference.country)
        corpus = index.load_country(
            country_id, verify_hashes=True, verify_values=True, mmap_mode="r"
        )
        split = _country_split(corpus, reference.variant)
        all_nodes = np.ones(corpus.manifest.node_count, dtype=np.bool_)
        inputs[reference.country] = InferenceData(
            country=reference.country,
            edge_index=corpus.edge_index,
            text_features=corpus.text_features,
            structural_features=corpus.degree_bucket,
            labels=corpus.labels,
            mask=all_nodes,
            structure_missing=corpus.structure_missing,
            graph_stats=corpus.graph_stats,
            source_hashes=corpus.manifest.source_hashes,
            split_hash=split.descriptor.split_hash,
        )
        corpora[reference.country] = corpus
        test_masks[reference.country] = np.asarray(split.test_mask, dtype=np.bool_)
    return inputs, corpora, test_masks


def evaluate_global_model_protocol(
    root: str | Path,
    *,
    protocol: ProtocolId,
    device: str = "cuda",
    fast: bool = False,
) -> Path:
    """Freeze first, then perform exactly one target-domain load/inference pass."""

    import torch

    selected = _safe_root(root)
    manifest_path, _index, _config, _plan, _options, _identity = _training_manifest_path(
        selected, protocol, fast=fast
    )
    run_dir = manifest_path.parent
    evaluation_path = run_dir / "evaluation.json"
    test_access_path = run_dir / "test-access.json"
    if evaluation_path.is_file():
        evaluation = _read_hashed(
            evaluation_path, schema=EVALUATION_SCHEMA, hash_field="evaluationHash"
        )
        access = _read_hashed(
            test_access_path, schema=TEST_ACCESS_SCHEMA, hash_field="testAccessHash"
        )
        if (
            access.get("state") != "completed"
            or access.get("protocol") != protocol
            or access.get("runId") != evaluation.get("runId")
            or access.get("trainingHash") != evaluation.get("trainingHash")
            or access.get("evaluationHash") != evaluation["evaluationHash"]
            or access.get("evaluationPath") != _relative(selected, evaluation_path)
            or evaluation.get("testAccessClaimHash") != access.get("claimHash")
        ):
            raise RuntimeError("Global completed evaluation lacks a matching test-access audit")
        result_hashes = access.get("resultHashes")
        if not isinstance(result_hashes, dict):
            raise RuntimeError("Global completed test-access result inventory is invalid")
        for country, references in evaluation["artifacts"]["resultPaths"].items():
            expected = result_hashes.get(country)
            if (
                not isinstance(expected, dict)
                or file_sha256(resolve_within(selected, references["jsonPath"]))
                != expected.get("jsonSha256")
                or file_sha256(resolve_within(selected, references["npzPath"]))
                != expected.get("npzSha256")
            ):
                raise RuntimeError("Global completed evaluation result integrity is invalid")
        return evaluation_path

    model, training, index, plan, options, calibrator = _load_trained_model(
        selected, protocol, fast=fast, device=device
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Global target evaluation requires a frozen model")
    if protocol == "cross_domain" and (
        tuple(reference.country for reference in plan.evaluate) != ("russia",)
        or training["targetAccess"]["labelsOrMasksPassedToTrainer"]
        or training["targetAccess"]["labelsOrMasksUsedForSelection"]
        or training["targetAccess"]["labelsOrMasksUsedForCalibration"]
    ):
        raise RuntimeError("Cross-domain source-only evidence is invalid before target evaluation")

    target_splits = {
        reference.country: index.entries[
            cast(CountryId, reference.country)
        ].split_hashes[_split_id(reference.variant)]
        for reference in plan.evaluate
    }
    claim_payload = {
        "schemaVersion": TEST_ACCESS_SCHEMA,
        "releaseId": RELEASE_ID,
        "taskId": TASK_ID,
        "protocol": protocol,
        "runId": training["runId"],
        "state": "claimed",
        "startedAt": _utc_now(),
        "trainingHash": training["trainingHash"],
        "identityHash": canonical_sha256(training["identity"]),
        "modelStateHash": training["artifacts"]["modelStateHash"],
        "targetCountries": [reference.country for reference in plan.evaluate],
        "targetSplitHashes": target_splits,
    }
    try:
        access_claim = _write_hashed_exclusive(
            test_access_path, claim_payload, "testAccessHash"
        )
    except FileExistsError as exc:
        raise RuntimeError(
            "Global test access was already claimed; incomplete evaluation cannot be retried"
        ) from exc

    inputs, corpora, test_masks = _evaluation_inputs(index, plan)
    outputs = collect_masked_outputs(
        model,
        inputs,
        options=options,
        device=torch.device(device),
        phase="frozen-test",
        allowed_experts=_allowed_experts(protocol),
    )
    result_paths: dict[str, dict[str, str]] = {}
    per_country: dict[str, dict[str, Any]] = {}
    threshold = float(training["calibration"]["threshold"])
    for country, output in outputs.items():
        corpus = corpora[country]
        node_ids = output.node_ids.numpy().astype(np.int64, copy=False)
        if not np.array_equal(node_ids, np.arange(corpus.manifest.node_count)):
            raise ValueError(f"{country} frozen inference did not cover every node exactly once")
        with torch.no_grad():
            calibrated_logits = calibrator(output.logits.float())
            scores = torch.sigmoid(calibrated_logits).cpu().numpy().astype(np.float32)
        logits = output.logits.numpy().astype(np.float32, copy=False)
        labels = output.labels.numpy().astype(np.uint8, copy=False)
        test_mask = test_masks[country]
        metrics = _binary_metrics(scores[test_mask], labels[test_mask], threshold=threshold)
        selected_test = torch.from_numpy(test_mask)
        metrics["ece"] = binary_ece(
            output.logits[selected_test],
            output.labels[selected_test],
            calibrator=calibrator,
        )
        result_npz = run_dir / "results" / f"{country}.npz"
        _atomic_npz(
            result_npz,
            node_ids=node_ids,
            scores=scores,
            logits=logits,
            structure_missing=output.structure_missing.numpy().astype(np.bool_, copy=False),
            router_indices=output.router_indices.numpy().astype(np.int64, copy=False),
            router_weights=output.router_weights.numpy().astype(np.float32, copy=False),
            modality_counts=_modality_counts(corpus),
        )
        result_json = run_dir / "results" / f"{country}.json"
        metadata = {
            "schemaVersion": "socialgraph-fm.global-model-result/1.0",
            "releaseId": RELEASE_ID,
            "taskId": TASK_ID,
            "protocol": protocol,
            "country": country,
            "nodeIdFormat": f"{country}:<zero-based-source-node-id>",
            "nodeCount": corpus.manifest.node_count,
            "testNodeCount": int(test_mask.sum()),
            "graphVersionHash": corpus.manifest.content_hash,
            "corpusHash": index.manifest.content_hash,
            "splitHash": inputs[country].split_hash,
            "threshold": threshold,
            "calibration": training["calibration"],
            "metrics": metrics,
            "testAccessClaimHash": access_claim["testAccessHash"],
            "expertNames": list(output.expert_names),
            "traceNames": list(TRACE_NAMES),
            "modalityCountSemantics": "factual directed neighbor count per relation CSR",
            "npzPath": _relative(selected, result_npz),
            "npzSha256": file_sha256(result_npz),
        }
        _write_hashed(result_json, metadata, "resultHash")
        result_paths[country] = {
            "jsonPath": _relative(selected, result_json),
            "jsonSha256": file_sha256(result_json),
            "npzPath": _relative(selected, result_npz),
            "npzSha256": file_sha256(result_npz),
        }
        per_country[country] = metrics
    evaluation = {
        "schemaVersion": EVALUATION_SCHEMA,
        "releaseId": RELEASE_ID,
        "taskId": TASK_ID,
        "state": "evaluated",
        "runId": training["runId"],
        "protocol": protocol,
        "seed": SEED,
        "identity": training["identity"],
        "trainingHash": training["trainingHash"],
        "corpusHash": index.manifest.content_hash,
        "sourceHashes": {
            country: dict(sorted(corpora[country].manifest.source_hashes.items()))
            for country in corpora
        },
        "splitHashes": {
            country: inputs[country].split_hash for country in inputs
        },
        "codeHash": training["codeHash"],
        "runtimeLockHash": training["runtimeLockHash"],
        "protocolPlan": plan.to_dict(),
        "domains": list(inputs),
        "config": training["config"],
        "calibration": training["calibration"],
        "metrics": {
            "perCountry": per_country,
            "countryBalancedMacroF1": sum(
                value["macroF1"] for value in per_country.values()
            )
            / len(per_country),
            "countryBalancedPrAuc": sum(
                value["prAuc"] for value in per_country.values()
            )
            / len(per_country),
        },
        "labelledTrainNodes": training["labelledTrainNodes"],
        "labelledTrainNodeCount": training["labelledTrainNodeCount"],
        "targetAccess": {
            **training["targetAccess"],
            "modelFrozenBeforeTargetAccess": True,
            "frozenTargetEvaluations": 1 if "russia" in inputs else 0,
        },
        "testAccessClaimHash": access_claim["testAccessHash"],
        "expertNames": training["expertNames"],
        "artifacts": {"resultPaths": result_paths},
    }
    if protocol == "cross_domain" and evaluation["targetAccess"]["frozenTargetEvaluations"] != 1:
        raise RuntimeError("Cross-domain must contain exactly one frozen Russia evaluation")
    _write_hashed(evaluation_path, evaluation, "evaluationHash")
    completed_evaluation = _read_hashed(
        evaluation_path, schema=EVALUATION_SCHEMA, hash_field="evaluationHash"
    )
    completed_access = {
        **claim_payload,
        "state": "completed",
        "completedAt": _utc_now(),
        "claimHash": access_claim["testAccessHash"],
        "evaluationPath": _relative(selected, evaluation_path),
        "evaluationHash": completed_evaluation["evaluationHash"],
        "resultHashes": {
            country: {
                "jsonSha256": references["jsonSha256"],
                "npzSha256": references["npzSha256"],
            }
            for country, references in result_paths.items()
        },
    }
    _write_hashed(test_access_path, completed_access, "testAccessHash")
    return evaluation_path


def _load_evaluations(
    root: Path, *, fast: bool
) -> tuple[dict[ProtocolId, dict[str, Any]], dict[ProtocolId, dict[str, Any]]]:
    evaluations: dict[ProtocolId, dict[str, Any]] = {}
    trainings: dict[ProtocolId, dict[str, Any]] = {}
    for protocol in PROTOCOLS:
        training_path, _index, _config, _plan, _options, identity = _training_manifest_path(
            root, protocol, fast=fast
        )
        training = _read_hashed(
            training_path, schema=TRAINING_SCHEMA, hash_field="trainingHash"
        )
        evaluation = _read_hashed(
            training_path.parent / "evaluation.json",
            schema=EVALUATION_SCHEMA,
            hash_field="evaluationHash",
        )
        if (
            training.get("identity") != identity
            or evaluation.get("identity") != identity
            or evaluation.get("trainingHash") != training["trainingHash"]
        ):
            raise ValueError(f"Global {protocol} evaluation identity is stale")
        evaluations[protocol] = evaluation
        trainings[protocol] = training
    return evaluations, trainings


def _preview_payload(
    corpus: GlobalCountryCorpus,
    *,
    scores: np.ndarray,
    structure_missing: np.ndarray,
    graph_version_hash: str,
) -> dict[str, Any]:
    node_count = corpus.manifest.node_count
    if scores.shape != (node_count,):
        raise ValueError("Russia preview scores do not cover the complete graph")
    edge_index = np.asarray(corpus.edge_index, dtype=np.int64)
    source, target = edge_index
    top = list(np.argsort(-scores, kind="stable")[:40].astype(int))
    selected = set(top)
    fused = corpus.fused_csr
    for anchor in top:
        start = int(fused.indptr[anchor])
        stop = int(fused.indptr[anchor + 1])
        neighbors = fused.indices[start:stop]
        for neighbor in neighbors[:3]:
            selected.add(int(neighbor))
            if len(selected) >= 120:
                break
        if len(selected) >= 120:
            break
    selected_ids = sorted(selected)
    selected_set = set(selected_ids)
    degrees = np.diff(np.asarray(fused.indptr, dtype=np.int64))
    relation_by_pair: dict[tuple[int, int], str] = {}
    for trace_name in TRACE_NAMES:
        relation = corpus.relation(trace_name)
        for left_id in selected_ids:
            start = int(relation.indptr[left_id])
            stop = int(relation.indptr[left_id + 1])
            for neighbor in relation.indices[start:stop]:
                right_id = int(neighbor)
                if right_id not in selected_set or right_id == left_id:
                    continue
                pair = (min(left_id, right_id), max(left_id, right_id))
                relation_by_pair.setdefault(pair, trace_name)
    nodes = [
        {
            "id": str(node_id),
            "label": f"Account {node_id}",
            "degree": int(degrees[node_id]),
            "score": float(scores[node_id]),
            "structureMissing": bool(structure_missing[node_id]),
        }
        for node_id in selected_ids
    ]
    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for left, right in zip(source, target, strict=True):
        left_id, right_id = int(left), int(right)
        if left_id not in selected_set or right_id not in selected_set or left_id == right_id:
            continue
        key = (min(left_id, right_id), max(left_id, right_id))
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "id": f"preview:{len(edges)}",
                "source": str(left_id),
                "target": str(right_id),
                "modality": relation_by_pair.get(key, "fused"),
            }
        )
        if len(edges) >= 400:
            break
    return {
        "schemaVersion": "socialgraph-fm.global-model-preview/1.0",
        "releaseId": RELEASE_ID,
        "datasetVersionId": "socialgraph-fm:russia",
        "graphVersionHash": graph_version_hash,
        "nodes": nodes,
        "edges": edges,
        "nodeCount": node_count,
        "edgeCount": corpus.manifest.edge_count // 2,
        "partialPreview": True,
        "traceNames": list(TRACE_NAMES),
    }


def _verify_result_npz(
    path: Path,
    *,
    expected_nodes: int | None = None,
    allowed_indices: set[int] | None = None,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        names = set(archive.files)
        required = {
            "node_ids",
            "scores",
            "logits",
            "structure_missing",
            "router_indices",
            "router_weights",
            "modality_counts",
        }
        if not required.issubset(names):
            raise ValueError(f"Global result is missing arrays: {sorted(required - names)}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    count = int(arrays["node_ids"].shape[0])
    if count < 1:
        raise ValueError("Global result must contain at least one node")
    if expected_nodes is not None and count != expected_nodes:
        raise ValueError("Global result node count is invalid")
    if (
        arrays["node_ids"].shape != (count,)
        or arrays["scores"].shape != (count,)
        or arrays["logits"].shape != (count,)
        or arrays["structure_missing"].shape != (count,)
        or arrays["router_indices"].shape != (count, 2)
        or arrays["router_weights"].shape != (count, 2)
        or arrays["modality_counts"].shape != (count, 5)
    ):
        raise ValueError("Global result array shapes disagree")
    if not np.array_equal(arrays["node_ids"], np.arange(count, dtype=np.int64)):
        raise ValueError("Global result node IDs are not the complete canonical inventory")
    scores = arrays["scores"]
    logits = arrays["logits"]
    if (
        not np.isfinite(scores).all()
        or not np.isfinite(logits).all()
        or not np.logical_and(scores >= 0, scores <= 1).all()
    ):
        raise ValueError("Global result scores are invalid")
    weights = arrays["router_weights"]
    indices = arrays["router_indices"]
    modality_counts = arrays["modality_counts"]
    if (
        not np.isfinite(weights).all()
        or not np.logical_and(weights >= 0, weights <= 1).all()
        or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-4)
        or indices.dtype.kind not in {"i", "u"}
        or bool(np.logical_or(indices < 1, indices > 7).any())
        or bool((indices[:, 0] == indices[:, 1]).any())
        or (
            allowed_indices is not None
            and not bool(np.isin(indices, tuple(sorted(allowed_indices))).all())
        )
    ):
        raise ValueError("Global top-2 router weights are invalid")
    if (
        arrays["structure_missing"].dtype != np.bool_
        or modality_counts.dtype.kind not in {"i", "u"}
        or bool((modality_counts < 0).any())
    ):
        raise ValueError("Global structural evidence arrays are invalid")
    return {"nodeCount": count, "arrayNames": sorted(names)}


def _allowed_expert_indices(
    expert_names: list[str], allowed_experts: list[str]
) -> set[int]:
    if len(expert_names) != 8 or expert_names[0] != "shared":
        raise ValueError("Global expert catalog is invalid")
    try:
        indices = {expert_names.index(name) for name in allowed_experts}
    except ValueError as exc:
        raise ValueError("Global expert allowlist is not in the fixed catalog") from exc
    if 0 in indices or len(indices) < 2:
        raise ValueError("Global routed expert allowlist is invalid")
    return indices


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_files() -> dict[str, str]:
    module_root = Path(__file__).resolve().parent
    return {
        f"socialgraph_gfm/global_model/{path.name}": file_sha256(path)
        for path in sorted(module_root.glob("*.py"), key=lambda item: item.name)
    }


def _protocol_model_identity(
    protocol: ProtocolId,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> tuple[str, str]:
    payload = {
        "schemaVersion": "socialgraph-fm.global-model-protocol-model-identity/1.0",
        "releaseId": RELEASE_ID,
        "taskId": TASK_ID,
        "protocol": protocol,
        "modelStateHash": training["artifacts"]["modelStateHash"],
        "corpusHash": evaluation["corpusHash"],
        "sourceCodeHash": evaluation["codeHash"],
        "runtimeLockHash": evaluation["runtimeLockHash"],
        "configHash": training["identity"]["configHash"],
        "allowedExperts": training["allowedExperts"],
        "expertNames": training["expertNames"],
    }
    version_hash = canonical_sha256(payload)
    protocol_token = protocol.replace("_", "-")
    return f"socialgraph-fm-{protocol_token}/{version_hash[:16]}", version_hash


def _write_export_document(
    staging: Path,
    inventory: dict[str, str],
    *,
    name: str,
    payload: dict[str, Any],
    hash_field: str,
) -> Path:
    path = staging / name
    _write_hashed(path, payload, hash_field)
    inventory[name] = file_sha256(path)
    return path


def export_global_model_release(root: str | Path, *, fast: bool = False) -> Path:
    """Create one immutable four-protocol export; it remains preliminary until smoke."""

    selected = _safe_root(root)
    evaluations, trainings = _load_evaluations(selected, fast=fast)
    corpus_hashes = {document["corpusHash"] for document in evaluations.values()}
    code_hashes = {document["codeHash"] for document in evaluations.values()}
    runtime_hashes = {document["runtimeLockHash"] for document in evaluations.values()}
    if len(corpus_hashes) != 1 or len(code_hashes) != 1 or len(runtime_hashes) != 1:
        raise ValueError("Global protocol artifacts do not share one release identity")
    target = selected / "exports" / (
        "socialgraph-global-fast" if fast else "socialgraph-global"
    )
    manifest_path = target / "export-manifest.json"
    if manifest_path.is_file():
        _read_hashed(manifest_path, schema=EXPORT_SCHEMA, hash_field="exportHash")
        return manifest_path
    if target.exists():
        raise FileExistsError(f"incomplete Global export already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.{os.getpid()}.staging"
    if staging.exists():
        raise FileExistsError(f"Global export staging path already exists: {staging}")
    staging.mkdir()
    try:
        import torch

        config = load_global_model_config()
        protocol_artifacts: dict[str, Any] = {}
        artifact_inventory: dict[str, str] = {}
        protocol_identities: dict[ProtocolId, tuple[str, str]] = {}
        for protocol in PROTOCOLS:
            evaluation = evaluations[protocol]
            training = trainings[protocol]
            model_version_id, model_version_hash = _protocol_model_identity(
                protocol, training, evaluation
            )
            protocol_identities[protocol] = (model_version_id, model_version_hash)
            source_checkpoint = resolve_within(
                selected, training["artifacts"]["checkpointPath"]
            )
            if file_sha256(source_checkpoint) != training["artifacts"]["checkpointSha256"]:
                raise ValueError(f"Global {protocol} source checkpoint hash mismatch")
            source_payload = torch.load(
                source_checkpoint, map_location="cpu", weights_only=True
            )
            state = source_payload["modelState"]
            model_state_hash = training["artifacts"]["modelStateHash"]
            if tensor_state_hash(state) != model_state_hash:
                raise ValueError(f"Global {protocol} source model state hash mismatch")
            checkpoint = staging / "checkpoints" / f"{protocol}.pt"
            _atomic_torch_save(
                checkpoint,
                {
                    "schemaVersion": "socialgraph-fm.global-model-weights/1.0",
                    "releaseId": RELEASE_ID,
                    "taskId": TASK_ID,
                    "protocol": protocol,
                    "protocolModelVersionId": model_version_id,
                    "protocolModelVersionHash": model_version_hash,
                    "modelStateHash": model_state_hash,
                    "modelConfig": config["model"],
                    "allowedExperts": training["allowedExperts"],
                    "expertNames": training["expertNames"],
                    "modelState": state,
                },
            )
            checkpoint_sha256 = file_sha256(checkpoint)
            artifact_inventory[f"checkpoints/{protocol}.pt"] = checkpoint_sha256
            copied_results: dict[str, dict[str, str]] = {}
            for country, references in evaluation["artifacts"]["resultPaths"].items():
                source_npz = resolve_within(selected, references["npzPath"])
                if file_sha256(source_npz) != references["npzSha256"]:
                    raise ValueError(f"Global {protocol}/{country} result hash mismatch")
                result_npz = staging / "results" / f"{protocol}-{country}.npz"
                result_npz.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_npz, result_npz)
                expected_nodes = 716 if country == "russia" else None
                _verify_result_npz(
                    result_npz,
                    expected_nodes=expected_nodes,
                    allowed_indices=_allowed_expert_indices(
                        training["expertNames"], training["allowedExperts"]
                    ),
                )
                source_json = resolve_within(selected, references["jsonPath"])
                if file_sha256(source_json) != references["jsonSha256"]:
                    raise ValueError(
                        f"Global {protocol}/{country} result metadata hash mismatch"
                    )
                source_metadata = _read_hashed(
                    source_json,
                    schema="socialgraph-fm.global-model-result/1.0",
                    hash_field="resultHash",
                )
                result_json = staging / "results" / f"{protocol}-{country}.json"
                metadata = {
                    key: value
                    for key, value in source_metadata.items()
                    if key != "resultHash"
                }
                metadata.update(
                    {
                        "modelVersionId": model_version_id,
                        "modelVersionHash": model_version_hash,
                        "modelStateHash": model_state_hash,
                        "npzPath": _relative(selected, target / "results" / result_npz.name),
                        "npzSha256": file_sha256(result_npz),
                    }
                )
                _write_hashed(result_json, metadata, "resultHash")
                final_npz = target / "results" / result_npz.name
                final_json = target / "results" / result_json.name
                copied_results[country] = {
                    "npzPath": _relative(selected, final_npz),
                    "npzSha256": file_sha256(result_npz),
                    "jsonPath": _relative(selected, final_json),
                    "jsonSha256": file_sha256(result_json),
                }
                artifact_inventory[
                    f"results/{result_npz.name}"
                ] = file_sha256(result_npz)
                artifact_inventory[
                    f"results/{result_json.name}"
                ] = file_sha256(result_json)
            russia_metrics = evaluation["metrics"]["perCountry"]["russia"]
            protocol_artifacts[protocol] = {
                "protocolModelVersionId": model_version_id,
                "protocolModelVersionHash": model_version_hash,
                "modelStateHash": model_state_hash,
                "state": "servingReady" if protocol == "global" else "frozenDemo",
                "checkpointPath": _relative(
                    selected, target / "checkpoints" / f"{protocol}.pt"
                ),
                "checkpointSha256": checkpoint_sha256,
                "allowedExperts": training["allowedExperts"],
                "resultPaths": copied_results,
                "splitHash": canonical_sha256(evaluation["splitHashes"]),
                "threshold": training["calibration"]["threshold"],
                "temperature": training["calibration"]["temperature"],
                "bias": training["calibration"]["bias"],
                "metrics": russia_metrics,
                "labelledTrainNodes": training["labelledTrainNodeCount"],
                "labelledTrainNodesByCountry": training["labelledTrainNodes"],
                "trainingHash": training["trainingHash"],
                "evaluationHash": evaluation["evaluationHash"],
                "targetAccess": evaluation["targetAccess"],
            }
        index = load_corpus_index(selected / "corpus", verify_manifests=True)
        russia = index.load_country(
            "russia", verify_hashes=True, verify_values=True, mmap_mode="r"
        )
        global_result_path = staging / "results" / "global-russia.npz"
        with np.load(global_result_path, allow_pickle=False) as result:
            preview = _preview_payload(
                russia,
                scores=np.asarray(result["scores"]),
                structure_missing=np.asarray(result["structure_missing"]),
                graph_version_hash=russia.manifest.content_hash,
            )
        preview_path = staging / "previews" / "russia.json"
        _write_hashed(preview_path, preview, "previewHash")
        artifact_inventory["previews/russia.json"] = file_sha256(preview_path)
        global_model_id, global_model_hash = protocol_identities["global"]
        artifact_hash = canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.global-model-core-artifacts/1.0",
                "modelVersionHash": global_model_hash,
                "corpusHash": index.manifest.content_hash,
                "files": artifact_inventory,
                "protocolArtifacts": protocol_artifacts,
            }
        )
        country_node_counts = {
            country: read_country_manifest(
                index.country_root(cast(CountryId, country))
            ).node_count
            for country in COUNTRIES
        }
        protocol_models = {
            protocol: {
                "modelVersionId": protocol_identities[protocol][0],
                "modelVersionHash": protocol_identities[protocol][1],
                "modelStateHash": trainings[protocol]["artifacts"]["modelStateHash"],
                "state": "servingReady" if protocol == "global" else "frozenDemo",
            }
            for protocol in PROTOCOLS
        }
        model_card_path = _write_export_document(
            staging,
            artifact_inventory,
            name="model-card.json",
            hash_field="modelCardHash",
            payload={
                "schemaVersion": MODEL_CARD_SCHEMA,
                "releaseId": RELEASE_ID,
                "modelVersionId": global_model_id,
                "modelVersionHash": global_model_hash,
                "taskId": TASK_ID,
                "architecture": {
                    "name": "Global cross-modal GraphSAGE with sparse routing",
                    "textFeatures": "anonymous precomputed 768-dimensional embeddings",
                    "structuralFeatures": "factual 128-bucket node degree",
                    "gnnLayers": 2,
                    "hiddenDim": 256,
                    "router": "shared residual plus top-2 domain/null adapters",
                },
                "protocols": protocol_models,
                "trainingData": {
                    "countries": list(COUNTRIES),
                    "nodeCount": sum(country_node_counts.values()),
                    "nodeCountByCountry": country_node_counts,
                    "content": "anonymous node IDs, graph relations, labels and precomputed embeddings; no raw text",
                },
                "intendedUse": [
                    "analyst-facing prioritization of potentially coordinated information-operation accounts",
                    "research and governance workflow demonstrations with mandatory human review",
                ],
                "outOfScope": [
                    "automated moderation, sanctions, account removal or legal attribution",
                    "identity inference or reconstruction of source text",
                    "deployment on an unregistered graph without a new governed evaluation",
                ],
                "limitations": [
                    "scores are model estimates and may contain false positives or domain shift",
                    "In-domain through Cross-domain are frozen demonstrations; only Global is the online primary model",
                    (
                        "the sparse router is an implementation-specific component, not a claim "
                        "of method reproduction"
                    ),
                ],
                "ethics": [
                    "human review is required before any governance action",
                    "preserve anonymity and do not use outputs as proof of individual intent",
                    "audit graph, model, threshold and protocol identities with every decision",
                ],
                "licenses": [
                    {
                        "name": "Information-operations dataset",
                        "license": "CC-BY-4.0",
                        "url": "https://zenodo.org/records/13357621",
                    },
                    {
                        "name": f"Official {_UPSTREAM_REFERENCE_NAME} reference code",
                        "license": "MIT",
                        "url": _UPSTREAM_REFERENCE_URL,
                    },
                ],
                "sourceAttribution": {
                    "kind": "inspired",
                    "paperUrl": "https://proceedings.mlr.press/v267/yuan25h.html",
                    "completeReproduction": False,
                },
                "metrics": {
                    protocol: evaluations[protocol]["metrics"] for protocol in PROTOCOLS
                },
                "artifactHash": artifact_hash,
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="model-config.json",
            hash_field="modelConfigHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-architecture/1.0",
                "releaseId": RELEASE_ID,
                "model": config["model"],
                "protocolModels": protocol_models,
                "allowedExperts": {
                    protocol: list(_allowed_experts(protocol)) for protocol in PROTOCOLS
                },
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="preprocess-config.json",
            hash_field="preprocessConfigHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-preprocess-config/1.0",
                "releaseId": RELEASE_ID,
                "splitIndex": SPLIT_INDEX,
                "traceNames": list(TRACE_NAMES),
                "graphStatNames": list(GRAPH_STAT_NAMES),
                "fusedGraph": "union of factual bidirectional relation CSR edges with self-loops removed",
                "structureMissing": "factual fused degree equals zero",
                "degreeBuckets": 128,
                "textFeatureDim": 768,
                "rawTextIncluded": False,
                "sourceNodeIdsAnonymized": True,
            },
        )
        expert_names = trainings["global"]["expertNames"]
        _write_export_document(
            staging,
            artifact_inventory,
            name="experts.json",
            hash_field="expertsHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-experts/1.0",
                "releaseId": RELEASE_ID,
                "expertNames": expert_names,
                "indexMapping": {str(index): name for index, name in enumerate(expert_names)},
                "sharedResidual": "always applied and excluded from top-2 routing",
                "routerTopK": 2,
                "allowedByProtocol": {
                    protocol: list(_allowed_experts(protocol)) for protocol in PROTOCOLS
                },
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="calibration.json",
            hash_field="calibrationHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-calibration/1.0",
                "releaseId": RELEASE_ID,
                "fitRole": "validation-only",
                "protocols": {
                    protocol: trainings[protocol]["calibration"] for protocol in PROTOCOLS
                },
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="metrics.json",
            hash_field="metricsHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-metrics/1.0",
                "releaseId": RELEASE_ID,
                "protocols": {
                    protocol: {
                        "validation": trainings[protocol]["metrics"],
                        "test": evaluations[protocol]["metrics"],
                    }
                    for protocol in PROTOCOLS
                },
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="environment-lock.json",
            hash_field="environmentLockHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-environment-lock/1.0",
                "releaseId": RELEASE_ID,
                "reference": "socialgraph_gfm/resources/runtime-lock-manifest.json",
                "runtimeLockHash": evaluations["global"]["runtimeLockHash"],
                "verificationScope": "checked-in lock integrity and requirement hash coverage",
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="source-files.json",
            hash_field="sourceFilesHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-source-files/1.0",
                "releaseId": RELEASE_ID,
                "sourceCodeHash": evaluations["global"]["codeHash"],
                "configHash": trainings["global"]["identity"]["configHash"],
                "files": _source_files(),
            },
        )
        global_result_path = staging / "results" / "global-russia.npz"
        with np.load(global_result_path, allow_pickle=False) as global_result:
            example_order = np.argsort(
                -np.asarray(global_result["scores"]), kind="stable"
            )[:3]
            example_output = {
                "schemaVersion": "socialgraph-fm.global-model-example-output/1.0",
                "releaseId": RELEASE_ID,
                "taskId": TASK_ID,
                "protocol": "global",
                "modelVersionId": global_model_id,
                "graphVersionHash": russia.manifest.content_hash,
                "findings": [
                    {
                        "nodeId": f"russia:{int(global_result['node_ids'][row])}",
                        "score": float(global_result["scores"][row]),
                        "structureMissing": bool(global_result["structure_missing"][row]),
                        "routes": [
                            {
                                "expert": expert_names[
                                    int(global_result["router_indices"][row, slot])
                                ],
                                "weight": float(global_result["router_weights"][row, slot]),
                            }
                            for slot in range(2)
                        ],
                        "modalityCounts": {
                            trace_name: int(global_result["modality_counts"][row, column])
                            for column, trace_name in enumerate(TRACE_NAMES)
                        },
                    }
                    for row in example_order
                ],
                "automaticEnforcement": False,
            }
        _write_export_document(
            staging,
            artifact_inventory,
            name="example-input.json",
            hash_field="exampleInputHash",
            payload={
                "schemaVersion": "socialgraph-fm.global-model-example-input/1.0",
                "releaseId": RELEASE_ID,
                "taskId": TASK_ID,
                "protocol": "global",
                "modelVersionId": global_model_id,
                "graphVersionHash": russia.manifest.content_hash,
                "datasetVersionId": "socialgraph-fm:russia",
            },
        )
        _write_export_document(
            staging,
            artifact_inventory,
            name="example-output.json",
            hash_field="exampleOutputHash",
            payload=example_output,
        )
        artifact_inventory_hash = canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.global-model-artifact-inventory/1.0",
                "artifactHash": artifact_hash,
                "files": artifact_inventory,
            }
        )
        model_card_sha256 = file_sha256(model_card_path)
        export = {
            "schemaVersion": EXPORT_SCHEMA,
            "releaseId": RELEASE_ID,
            "taskId": TASK_ID,
            "state": "preliminary",
            "fast": fast,
            "seed": SEED,
            "protocols": list(PROTOCOLS),
            "modelVersionId": global_model_id,
            "modelVersionHash": global_model_hash,
            "modelStateHash": trainings["global"]["artifacts"]["modelStateHash"],
            "artifactHash": artifact_hash,
            "artifactInventoryHash": artifact_inventory_hash,
            "corpusHash": index.manifest.content_hash,
            "sourceHashes": _source_inventory(index),
            "splitHashes": _split_inventory(index),
            "sourceCodeHash": evaluations["global"]["codeHash"],
            "runtimeLockHash": evaluations["global"]["runtimeLockHash"],
            "domains": {
                protocol: {
                    "train": list(config["protocolPlans"][protocol].train_domains),
                    "evaluate": [
                        reference.country
                        for reference in config["protocolPlans"][protocol].evaluate
                    ],
                }
                for protocol in PROTOCOLS
            },
            "config": {
                "configHash": trainings["global"]["identity"]["configHash"],
                "model": config["model"],
                "training": config["training"],
            },
            "calibration": {
                protocol: trainings[protocol]["calibration"] for protocol in PROTOCOLS
            },
            "metrics": {
                protocol: evaluations[protocol]["metrics"] for protocol in PROTOCOLS
            },
            "graphVersionHash": russia.manifest.content_hash,
            "checkpointPath": protocol_artifacts["global"]["checkpointPath"],
            "checkpointSha256": protocol_artifacts["global"]["checkpointSha256"],
            "protocolArtifacts": protocol_artifacts,
            "protocolModels": protocol_models,
            "expertNames": trainings["global"]["expertNames"],
            "traceNames": list(TRACE_NAMES),
            "russiaPreviewPath": _relative(selected, target / "previews" / "russia.json"),
            "russiaPreviewSha256": file_sha256(preview_path),
            "modelCardPath": _relative(selected, target / "model-card.json"),
            "modelCardSha256": model_card_sha256,
            "artifacts": artifact_inventory,
        }
        _write_hashed(staging / "export-manifest.json", export, "exportHash")
        candidate = {
            key: value
            for key, value in export.items()
            if key not in {"schemaVersion", "state", "fast", "artifacts"}
        }
        candidate.update(
            {
                "schemaVersion": REGISTRY_SCHEMA,
                "state": "preliminary",
                "exportPath": _relative(selected, target / "export-manifest.json"),
                "exportSha256": file_sha256(staging / "export-manifest.json"),
            }
        )
        _write_hashed(staging / "registry-candidate.json", candidate, "registryHash")
        os.replace(staging, target)
        return target / "export-manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_global_model_export(root: str | Path, *, fast: bool = False) -> dict[str, Any]:
    selected = _safe_root(root)
    export_root = selected / "exports" / (
        "socialgraph-global-fast" if fast else "socialgraph-global"
    )
    export = _read_hashed(
        export_root / "export-manifest.json",
        schema=EXPORT_SCHEMA,
        hash_field="exportHash",
    )
    if export.get("state") != "preliminary" or export.get("fast") is not fast:
        raise ValueError("Global export stage identity is invalid")
    import torch

    inventory = export.get("artifacts")
    if not isinstance(inventory, dict):
        raise TypeError("Global export artifact inventory is invalid")
    required_documents = {
        "model-card.json",
        "model-config.json",
        "preprocess-config.json",
        "experts.json",
        "calibration.json",
        "metrics.json",
        "environment-lock.json",
        "source-files.json",
        "example-input.json",
        "example-output.json",
        "previews/russia.json",
        *(f"checkpoints/{protocol}.pt" for protocol in PROTOCOLS),
    }
    if not required_documents.issubset(inventory):
        raise ValueError("Global export is missing required deployment artifacts")
    for relative, expected_hash in inventory.items():
        artifact_path = resolve_within(export_root, relative)
        if file_sha256(artifact_path) != expected_hash:
            raise ValueError(f"Global exported artifact hash mismatch: {relative}")
    inventory_hash = canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.global-model-artifact-inventory/1.0",
            "artifactHash": export["artifactHash"],
            "files": inventory,
        }
    )
    if inventory_hash != export.get("artifactInventoryHash"):
        raise ValueError("Global artifact inventory identity is invalid")
    core_inventory = {
        name: value
        for name, value in inventory.items()
        if name.startswith(("checkpoints/", "results/", "previews/"))
    }
    core_hash = canonical_sha256(
        {
            "schemaVersion": "socialgraph-fm.global-model-core-artifacts/1.0",
            "modelVersionHash": export["modelVersionHash"],
            "corpusHash": export["corpusHash"],
            "files": core_inventory,
            "protocolArtifacts": export["protocolArtifacts"],
        }
    )
    if core_hash != export["artifactHash"]:
        raise ValueError("Global core artifact identity is invalid")
    for protocol in PROTOCOLS:
        artifact = export["protocolArtifacts"][protocol]
        protocol_model = export["protocolModels"][protocol]
        if (
            artifact["protocolModelVersionId"] != protocol_model["modelVersionId"]
            or artifact["protocolModelVersionHash"] != protocol_model["modelVersionHash"]
            or artifact["modelStateHash"] != protocol_model["modelStateHash"]
            or artifact["allowedExperts"] != list(_allowed_experts(protocol))
        ):
            raise ValueError(f"Global {protocol} protocol model identity is invalid")
        checkpoint = resolve_within(selected, artifact["checkpointPath"])
        if file_sha256(checkpoint) != artifact["checkpointSha256"]:
            raise ValueError(f"Global {protocol} checkpoint hash mismatch")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (
            checkpoint_payload.get("schemaVersion")
            != "socialgraph-fm.global-model-weights/1.0"
            or checkpoint_payload.get("protocol") != protocol
            or checkpoint_payload.get("protocolModelVersionId")
            != artifact["protocolModelVersionId"]
            or checkpoint_payload.get("protocolModelVersionHash")
            != artifact["protocolModelVersionHash"]
            or checkpoint_payload.get("modelStateHash") != artifact["modelStateHash"]
            or tensor_state_hash(checkpoint_payload["modelState"])
            != artifact["modelStateHash"]
        ):
            raise ValueError(f"Global {protocol} weights checkpoint binding is invalid")
        if "russia" not in artifact["resultPaths"]:
            raise ValueError(f"Global exported {protocol} lacks the Russia result")
        for country, result_reference in artifact["resultPaths"].items():
            npz_path = resolve_within(selected, result_reference["npzPath"])
            json_path = resolve_within(selected, result_reference["jsonPath"])
            if (
                file_sha256(npz_path) != result_reference["npzSha256"]
                or file_sha256(json_path) != result_reference["jsonSha256"]
            ):
                raise ValueError(
                    f"Global exported {protocol}/{country} result hash mismatch"
                )
            _verify_result_npz(
                npz_path,
                expected_nodes=716 if country == "russia" else None,
                allowed_indices=_allowed_expert_indices(
                    export["expertNames"], artifact["allowedExperts"]
                ),
            )
            metadata = _read_hashed(
                json_path,
                schema="socialgraph-fm.global-model-result/1.0",
                hash_field="resultHash",
            )
            if (
                metadata.get("protocol") != protocol
                or metadata.get("country") != country
                or metadata.get("modelVersionId")
                != artifact["protocolModelVersionId"]
                or metadata.get("modelVersionHash")
                != artifact["protocolModelVersionHash"]
                or metadata.get("modelStateHash") != artifact["modelStateHash"]
                or metadata.get("expertNames") != export["expertNames"]
            ):
                raise ValueError(
                    f"Global exported {protocol}/{country} metadata binding mismatch"
                )
    preview = resolve_within(selected, export["russiaPreviewPath"])
    if file_sha256(preview) != export["russiaPreviewSha256"]:
        raise ValueError("Global Russia preview hash mismatch")
    _read_hashed(
        preview,
        schema="socialgraph-fm.global-model-preview/1.0",
        hash_field="previewHash",
    )
    documents = {
        "model-config.json": (
            "socialgraph-fm.global-model-architecture/1.0",
            "modelConfigHash",
        ),
        "preprocess-config.json": (
            "socialgraph-fm.global-model-preprocess-config/1.0",
            "preprocessConfigHash",
        ),
        "experts.json": ("socialgraph-fm.global-model-experts/1.0", "expertsHash"),
        "calibration.json": (
            "socialgraph-fm.global-model-calibration/1.0",
            "calibrationHash",
        ),
        "metrics.json": ("socialgraph-fm.global-model-metrics/1.0", "metricsHash"),
        "environment-lock.json": (
            "socialgraph-fm.global-model-environment-lock/1.0",
            "environmentLockHash",
        ),
        "source-files.json": (
            "socialgraph-fm.global-model-source-files/1.0",
            "sourceFilesHash",
        ),
        "example-input.json": (
            "socialgraph-fm.global-model-example-input/1.0",
            "exampleInputHash",
        ),
        "example-output.json": (
            "socialgraph-fm.global-model-example-output/1.0",
            "exampleOutputHash",
        ),
    }
    for name, (schema, hash_field) in documents.items():
        _read_hashed(export_root / name, schema=schema, hash_field=hash_field)
    model_card = _read_hashed(
        resolve_within(selected, export["modelCardPath"]),
        schema=MODEL_CARD_SCHEMA,
        hash_field="modelCardHash",
    )
    if (
        file_sha256(resolve_within(selected, export["modelCardPath"]))
        != export["modelCardSha256"]
        or model_card.get("modelVersionId") != export["modelVersionId"]
        or model_card.get("modelVersionHash") != export["modelVersionHash"]
        or model_card.get("artifactHash") != export["artifactHash"]
    ):
        raise ValueError("Global model card binding is invalid")
    if (
        export["checkpointPath"]
        != export["protocolArtifacts"]["global"]["checkpointPath"]
        or export["checkpointSha256"]
        != export["protocolArtifacts"]["global"]["checkpointSha256"]
        or export["modelVersionId"]
        != export["protocolArtifacts"]["global"]["protocolModelVersionId"]
        or export["modelVersionHash"]
        != export["protocolArtifacts"]["global"]["protocolModelVersionHash"]
    ):
        raise ValueError("Global top-level model is not the Global primary model")
    return {
        "passed": True,
        "modelVersionId": export["modelVersionId"],
        "modelVersionHash": export["modelVersionHash"],
        "artifactHash": export["artifactHash"],
        "protocols": list(PROTOCOLS),
    }


def smoke_global_model_export(
    root: str | Path,
    *,
    fast: bool = False,
    fresh_process: bool = True,
) -> Path:
    selected = _safe_root(root)
    export_root = selected / "exports" / (
        "socialgraph-global-fast" if fast else "socialgraph-global"
    )
    smoke_path = export_root / "smoke-report.json"
    if fresh_process:
        command = [
            sys.executable,
            "-m",
            "socialgraph_gfm.global_model.cli",
            "_verify-export",
            "--root",
            str(selected),
        ]
        if fast:
            command.append("--fast")
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "fresh Global export verification failed: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        result = json.loads(completed.stdout)
    else:
        command = []
        result = verify_global_model_export(selected, fast=fast)
    report = {
        "schemaVersion": SMOKE_SCHEMA,
        "releaseId": RELEASE_ID,
        "state": "passed",
        "passed": result.get("passed") is True,
        "fast": fast,
        "freshProcess": fresh_process,
        "command": command,
        "pythonExecutable": sys.executable,
        "pythonExecutableSha256": file_sha256(sys.executable),
        "modelVersionId": result["modelVersionId"],
        "modelVersionHash": result["modelVersionHash"],
        "artifactHash": result["artifactHash"],
        "protocols": result["protocols"],
    }
    if report["passed"] is not True:
        raise RuntimeError("Global smoke verification did not pass")
    return _write_hashed(smoke_path, report, "smokeHash")


def publish_global_model_release(root: str | Path) -> Path:
    selected = _safe_root(root)
    export_root = selected / "exports" / "socialgraph-global"
    export = _read_hashed(
        export_root / "export-manifest.json",
        schema=EXPORT_SCHEMA,
        hash_field="exportHash",
    )
    smoke = _read_hashed(
        export_root / "smoke-report.json",
        schema=SMOKE_SCHEMA,
        hash_field="smokeHash",
    )
    verification = verify_global_model_export(selected, fast=False)
    if (
        export.get("fast") is not False
        or export.get("state") != "preliminary"
        or tuple(export.get("protocols", ())) != PROTOCOLS
        or smoke.get("passed") is not True
        or smoke.get("freshProcess") is not True
        or smoke.get("modelVersionHash") != export["modelVersionHash"]
        or smoke.get("artifactHash") != export["artifactHash"]
        or verification.get("artifactHash") != export["artifactHash"]
    ):
        raise ValueError("Global release is not eligible for publication")
    registry = {
        key: export[key]
        for key in (
            "releaseId",
            "taskId",
            "seed",
            "protocols",
            "modelVersionId",
            "modelVersionHash",
            "modelStateHash",
            "artifactHash",
            "artifactInventoryHash",
            "corpusHash",
            "sourceCodeHash",
            "runtimeLockHash",
            "graphVersionHash",
            "checkpointPath",
            "checkpointSha256",
            "protocolArtifacts",
            "protocolModels",
            "expertNames",
            "traceNames",
            "russiaPreviewPath",
            "russiaPreviewSha256",
            "modelCardPath",
            "modelCardSha256",
        )
    }
    registry["protocolArtifacts"] = {
        protocol: {
            **export["protocolArtifacts"][protocol],
            "state": "servingReady" if protocol == "global" else "frozenDemo",
        }
        for protocol in PROTOCOLS
    }
    registry["protocolModels"] = {
        protocol: {
            **export["protocolModels"][protocol],
            "state": "servingReady" if protocol == "global" else "frozenDemo",
        }
        for protocol in PROTOCOLS
    }
    registry.update(
        {
            "schemaVersion": REGISTRY_SCHEMA,
            "state": "servingReady",
            "exportPath": _relative(selected, export_root / "export-manifest.json"),
            "exportSha256": file_sha256(export_root / "export-manifest.json"),
            "smokePath": _relative(selected, export_root / "smoke-report.json"),
            "smokeSha256": file_sha256(export_root / "smoke-report.json"),
        }
    )
    export_registry = export_root / "registry.json"
    registry_root = selected / "registry" / "socialgraph-global.json"
    existing = None
    if registry_root.is_file():
        existing = _read_hashed(
            registry_root, schema=REGISTRY_SCHEMA, hash_field="registryHash"
        )
    logical_hash = canonical_sha256(registry)
    if existing is not None:
        existing_logical = {
            key: value for key, value in existing.items() if key != "registryHash"
        }
        if canonical_sha256(existing_logical) != logical_hash:
            raise FileExistsError("a different Global serving registry is already published")
        return registry_root
    _write_hashed(export_registry, registry, "registryHash")
    _write_hashed(registry_root, registry, "registryHash")
    return registry_root


__all__ = [
    "PROTOCOLS",
    "convert_global_model_corpus",
    "evaluate_global_model_protocol",
    "export_global_model_release",
    "global_model_root_from_home",
    "publish_global_model_release",
    "smoke_global_model_export",
    "train_global_model_protocol",
    "validate_global_model_corpus",
    "verify_global_model_export",
]
