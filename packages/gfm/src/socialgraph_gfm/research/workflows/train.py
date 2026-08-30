"""Shared-model training and comparison-matrix training for SocialGraph-FM Research."""

from __future__ import annotations

import json
import math
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from ...core.bundle import CoreGraphBundle
from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RELEASE_ID,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from ..routing import route_contract, task_route_domain, task_route_name
from .common import (
    TRAINING_SCHEMA,
    _atomic_json,
    _domain_task_id,
    _read_hashed_document,
    _route_contract_hash,
    _safe_root,
    load_research_config,
)
from .materialize import _validate_tolokers_split_payload, load_corpus_manifest
from .runtime import _load_trained_runtime


@dataclass(frozen=True)
class ResearchTrainingConfig:
    preset: str = "research"
    max_steps: int = 540
    max_epochs: int = 60
    min_steps: int = 0
    validation_interval: int = 9
    patience: int = 8
    head_max_epochs: int = 100
    head_patience: int = 10
    hidden_dim: int = 128
    encoder_layers: int = 3
    dropout: float = 0.2
    field_mask_rate: float = 0.15
    edge_mask_rate: float = 0.10
    alignment_weight: float = 0.02
    alignment_source_scores: None = None
    full_batch_edge_threshold: int = 1_500_000
    node_batch_size: int = 2048
    edge_batch_size: int = 2048
    fanout: tuple[int, int, int] = (20, 10, 5)
    amp: bool = True
    gradient_accumulation: int = 1
    timeout_seconds: float = 6 * 60 * 60
    moe_enabled: bool = False
    future_moe_capability_version: str = "research-lightweight-route"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("research training requires at least one optimizer step")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tensor_state_hash(state: Mapping[str, Any]) -> str:
    from socialgraph_gfm.tensor_digest import canonical_tensor_digest

    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


def _torch_atomic_save(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bundle_edge_index(bundle: CoreGraphBundle, *, visible_only: bool):
    import torch

    from ...core.adapters import derive_training_selection

    selected = (
        derive_training_selection(bundle).visible_edge_indices
        if visible_only
        else tuple(range(len(bundle.edges)))
    )
    by_id = {node.id: node.index for node in bundle.nodes}
    pairs = [
        (by_id[bundle.edges[index].source_id], by_id[bundle.edges[index].target_id])
        for index in selected
    ]
    if not bundle.directed:
        pairs.extend((right, left) for left, right in tuple(pairs))
    return (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long)
    )


def _load_graph_documents(root: Path, manifest: Mapping[str, Any]):
    corpus_root = root / "materialized/corpus"
    documents: dict[str, tuple[CoreGraphBundle, dict[str, Any], Mapping[str, Any]]] = {}
    for entry in manifest["graphs"]:
        bundle = CoreGraphBundle.model_validate_json(
            (corpus_root / entry["bundlePath"]).read_bytes()
        )
        labels = json.loads((corpus_root / entry["labelsPath"]).read_text(encoding="utf-8"))
        if labels.get("labelsHash") != canonical_sha256(
            {key: value for key, value in labels.items() if key != "labelsHash"}
        ):
            raise ValueError("research labels content hash mismatch")
        documents[entry["graphId"]] = (bundle, labels, entry)
    return documents


def _prepare_training_runtime(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    device: str,
    force_neighbor_fallback: bool = False,
):
    from ...core.adapters import BundleInputAdapter
    from ...core.trainer import TrainingGraph
    from ...core.training_data import ExecutionPolicy, PreparedGraph

    documents = _load_graph_documents(root, manifest)
    model_protocol = load_research_config()["model"]
    graphs: dict[str, TrainingGraph] = {}
    adapters: dict[str, BundleInputAdapter] = {}
    for domain, (bundle, _labels, _entry) in sorted(documents.items()):
        adapter = BundleInputAdapter(bundle, mode="training", multi_hot_buckets=256).to(device)
        visible = _bundle_edge_index(bundle, visible_only=True).to(device)
        positives = _bundle_edge_index(bundle, visible_only=False).to(device)
        prepared = PreparedGraph.from_edge_index(
            num_nodes=len(bundle.nodes),
            edge_index=visible,
            directed=bundle.directed,
            positive_edge_index=positives,
        )
        policy = ExecutionPolicy(
            full_batch_edge_threshold=(
                0 if force_neighbor_fallback else int(model_protocol["fullBatchEdgeThreshold"])
            ),
            node_batch_size=int(model_protocol["neighborFallback"]["batchSize"]),
            edge_batch_size=int(model_protocol["neighborFallback"]["batchSize"]),
            fanout=cast(
                tuple[int, int, int],
                tuple(int(value) for value in model_protocol["neighborFallback"]["fanout"]),
            ),
        )
        graphs[domain] = TrainingGraph.from_bundle(
            adapter=adapter,
            graph=prepared,
            execution_policy=policy,
        )
        adapters[domain] = adapter
    return documents, graphs, adapters


def _role_ids(bundle: CoreGraphBundle, role: str) -> tuple[str, ...]:
    return tuple(
        assignment.entity_id
        for assignment in bundle.split_manifest.assignments
        if assignment.role == role
    )


def _clone_tensor_state(module) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _binary_logit(logits):
    return logits[:, 1] - logits[:, 0] if logits.ndim == 2 else logits


def _loss_for_logits(logits, targets):
    from torch.nn import functional

    return (
        functional.cross_entropy(logits, targets.long())
        if logits.ndim == 2
        else functional.binary_cross_entropy_with_logits(logits, targets.float())
    )


def _ece(labels: list[int], scores: list[float], *, bins: int = 10) -> float:
    """Return expected calibration error for aligned binary outcomes."""

    total = len(labels)
    value = 0.0
    for bucket in range(bins):
        lower = bucket / bins
        upper = (bucket + 1) / bins
        selected = [
            index
            for index, score in enumerate(scores)
            if lower <= score < upper or (bucket == bins - 1 and score == 1.0)
        ]
        if selected:
            confidence = sum(scores[index] for index in selected) / len(selected)
            accuracy = sum(labels[index] for index in selected) / len(selected)
            value += len(selected) / total * abs(confidence - accuracy)
    return value


def _calibration_metrics(logits, targets) -> tuple[float, float, float]:
    import torch
    from torch.nn import functional

    normalized_logits = logits.detach().to(dtype=torch.float64, device="cpu")
    normalized_targets = targets.detach().to(dtype=torch.float64, device="cpu")
    probabilities = torch.sigmoid(normalized_logits)
    nll = float(functional.binary_cross_entropy_with_logits(normalized_logits, normalized_targets))
    brier = float(torch.mean((probabilities - normalized_targets) ** 2))
    return (
        nll,
        _ece(
            [int(value) for value in normalized_targets.tolist()],
            [float(value) for value in probabilities.tolist()],
        ),
        brier,
    )


def _calibration_is_adequate(
    *, before_ece: float, after_ece: float, inadequacy_reason: str | None
) -> bool:
    return inadequacy_reason is None and after_ece <= 0.20 and after_ece <= before_ece + 1e-10


def _fit_research_calibrator(
    *, task_id: str, logits, targets, partition_hash: str
) -> dict[str, Any]:
    """Fit a validation-only bounded temperature+bias calibrator."""

    import torch
    from torch.nn import functional

    raw_logits = _binary_logit(logits).detach().to(dtype=torch.float64, device="cpu")
    raw_targets = targets.detach().to(dtype=torch.float64, device="cpu")
    before = _calibration_metrics(raw_logits, raw_targets)
    temperature = 1.0
    bias = 0.0
    reason: str | None = None
    if raw_logits.numel() < 10:
        reason = "VALIDATION_SAMPLE_TOO_SMALL"
    elif torch.unique(raw_targets).numel() != 2:
        reason = "VALIDATION_MISSING_CLASS"
    else:
        log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        bias_parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        optimizer = torch.optim.LBFGS(
            (log_temperature, bias_parameter),
            lr=0.5,
            max_iter=100,
            tolerance_grad=1e-12,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer.zero_grad(set_to_none=True)
            bounded_temperature = torch.exp(log_temperature.clamp(math.log(0.05), math.log(20.0)))
            bounded_bias = bias_parameter.clamp(-20.0, 20.0)
            loss = functional.binary_cross_entropy_with_logits(
                (raw_logits + bounded_bias) / bounded_temperature, raw_targets
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        temperature = float(
            torch.exp(log_temperature.detach().clamp(math.log(0.05), math.log(20.0)))
        )
        bias = float(bias_parameter.detach().clamp(-20.0, 20.0))
    calibrated = (raw_logits + bias) / temperature
    after = _calibration_metrics(calibrated, raw_targets)
    if after[0] > before[0] + 1e-10:
        temperature, bias, after = 1.0, 0.0, before
        reason = "CALIBRATION_WORSENED_NLL"
    adequate = _calibration_is_adequate(
        before_ece=before[1], after_ece=after[1], inadequacy_reason=reason
    )
    if not adequate and reason is None:
        reason = "VALIDATION_ECE_ABOVE_THRESHOLD"
    artifact: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.research-calibration/1.0",
        "taskId": task_id,
        "method": "validation-temperature-bias",
        "temperature": temperature,
        "bias": bias,
        "validationCount": int(raw_targets.numel()),
        "validationPartitionHash": partition_hash,
        "beforeNll": before[0],
        "afterNll": after[0],
        "beforeEce": before[1],
        "afterEce": after[1],
        "beforeBrier": before[2],
        "afterBrier": after[2],
        "adequate": adequate,
        "inadequacyReason": reason,
    }
    artifact["artifactHash"] = canonical_sha256(artifact)
    return artifact


def _task_batch(
    *,
    task_id: str,
    module,
    encoded: Mapping[str, Any],
    documents,
    role: str,
    device: str,
    tolokers_indices: Iterable[int] | None = None,
):
    import torch

    logits_parts = []
    target_parts = []
    if task_id == CONTENT_POLICY_TASK:
        for domain in (item for item in sorted(documents) if item.startswith("twitch-")):
            bundle, labels, _entry = documents[domain]
            binary_labels_by_id = {
                item["entityId"]: int(item["target"]) for item in labels["targets"]
            }
            nodes_by_id = {node.id: node.index for node in bundle.nodes}
            selected = [item for item in _role_ids(bundle, role) if item in binary_labels_by_id]
            if selected:
                indices = torch.tensor([nodes_by_id[item] for item in selected], device=device)
                logits_parts.append(module(encoded[domain][indices]))
                target_parts.append(
                    torch.tensor([binary_labels_by_id[item] for item in selected], device=device)
                )
    elif task_id == ACCOUNT_RISK_TASK:
        bundle, labels, _entry = documents["tolokers"]
        binary_labels_by_id = {
            item["entityId"]: int(item["target"]) for item in labels["targets"]
        }
        indices_values = tuple(tolokers_indices or ())
        if not indices_values:
            raise ValueError("Tolokers task fitting requires an explicit official split")
        indices = torch.tensor(indices_values, device=device)
        logits_parts.append(module(encoded["tolokers"][indices]))
        target_parts.append(
            torch.tensor(
                [binary_labels_by_id[bundle.nodes[index].id] for index in indices_values],
                device=device,
            )
        )
    elif task_id == SIGNED_RELATION_TASK:
        bundle, labels, _entry = documents["wiki-rfa"]
        signed_labels_by_id = {item["entityId"]: item for item in labels["targets"]}
        nodes_by_id = {node.id: node.index for node in bundle.nodes}
        selected = [item for item in _role_ids(bundle, role) if item in signed_labels_by_id]
        if selected:
            pairs = torch.tensor(
                [
                    (
                        nodes_by_id[signed_labels_by_id[item]["sourceId"]],
                        nodes_by_id[signed_labels_by_id[item]["targetId"]],
                    )
                    for item in selected
                ],
                device=device,
            )
            logits_parts.append(module(encoded["wiki-rfa"], pairs))
            target_parts.append(
                torch.tensor(
                    [signed_labels_by_id[item]["target"] for item in selected],
                    dtype=torch.float32,
                    device=device,
                )
            )
    else:
        bundle, labels, _entry = documents["email-eu-core"]
        nodes_by_id = {node.id: node.index for node in bundle.nodes}
        partition = labels["partitions"][role]
        rows = (*partition["positives"], *partition["negatives"])
        if rows:
            pairs = torch.tensor(
                [(nodes_by_id[item["sourceId"]], nodes_by_id[item["targetId"]]) for item in rows],
                device=device,
            )
            logits_parts.append(module(encoded["email-eu-core"], pairs))
            target_parts.append(
                torch.tensor([item["target"] for item in rows], dtype=torch.float32, device=device)
            )
    if not logits_parts:
        raise ValueError(f"{task_id} has no {role} examples")
    return torch.cat(logits_parts), torch.cat(target_parts)


def _fit_frozen_head(
    *,
    task_id: str,
    module,
    train_batch,
    validation_batch,
    epochs: int,
    patience: int,
    partition_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    best_state = _clone_tensor_state(module)
    stale = 0
    stopped_early = False
    for epoch in range(1, epochs + 1):
        module.train()
        logits, targets = train_batch()
        loss = _loss_for_logits(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        module.eval()
        with torch.inference_mode():
            validation_logits, validation_targets = validation_batch()
            validation_loss = float(
                _loss_for_logits(validation_logits, validation_targets).detach().cpu()
            )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _clone_tensor_state(module)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                stopped_early = True
                break
    module.load_state_dict(best_state, strict=True)
    module.eval()
    with torch.inference_mode():
        validation_logits, validation_targets = validation_batch()
    calibrator = _fit_research_calibrator(
        task_id=task_id,
        logits=validation_logits,
        targets=validation_targets,
        partition_hash=partition_hash,
    )
    report: dict[str, Any] = {
        "taskId": task_id,
        "maxEpochs": epochs,
        "patience": patience,
        "epochsCompleted": best_epoch + stale,
        "bestEpoch": best_epoch,
        "bestValidationLoss": best_loss,
        "stoppedEarly": stopped_early,
        "bestHeadStateHash": _tensor_state_hash(module.state_dict()),
        "validationPartitionHash": partition_hash,
        "calibrationArtifactHash": calibrator["artifactHash"],
    }
    report["reportHash"] = canonical_sha256(report)
    return report, calibrator


def _load_tolokers_folds(root: Path, documents) -> tuple[dict[str, Any], ...]:
    entry = documents["tolokers"][2]
    path = (root / "materialized/corpus" / entry["splitsPath"]).resolve()
    if not path.is_relative_to((root / "materialized/corpus").resolve()):
        raise ValueError("Tolokers split path escapes corpus root")
    if file_sha256(path) != entry["splitsSha256"]:
        raise ValueError("Tolokers split artifact hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _validate_tolokers_split_payload(
        payload,
        node_count=len(documents["tolokers"][0].nodes),
        bundle=documents["tolokers"][0],
    )


def _train_task_heads(
    *,
    root: Path,
    model,
    documents,
    graphs,
    adapters,
    device: str,
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[int, dict[str, Any]],
    tuple[dict[str, Any], ...],
]:
    import torch

    torch.manual_seed(seed + 100)
    model.eval()
    for adapter in adapters.values():
        adapter.eval()
    encoded: dict[str, Any] = {}
    with torch.no_grad():
        for domain in sorted(documents):
            task_id = _domain_task_id(domain)
            encoded[domain] = model.encode_domain(
                adapters[domain](),
                graphs[domain].graph.edge_index,
                task_route_domain(task_id, domain),
            ).detach()

    reports: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    modules = {
        CONTENT_POLICY_TASK: model.content_policy_head,
        SIGNED_RELATION_TASK: model.signed_edge_head,
        COLLABORATION_TASK: model.collaboration_head,
    }
    for ordinal, (task_id, module) in enumerate(modules.items()):
        torch.manual_seed(seed + 1_000 + ordinal)
        partition_hash = canonical_sha256(
            {
                domain: list(_role_ids(documents[domain][0], "validation"))
                if task_id != COLLABORATION_TASK
                else documents[domain][1].get("samplingHash")
                for domain in sorted(documents)
                if (
                    (task_id == CONTENT_POLICY_TASK and domain.startswith("twitch-"))
                    or (task_id == SIGNED_RELATION_TASK and domain == "wiki-rfa")
                    or (task_id == COLLABORATION_TASK and domain == "email-eu-core")
                )
            }
        )
        report, calibrator = _fit_frozen_head(
            task_id=task_id,
            module=module,
            train_batch=lambda task_id=task_id, module=module: _task_batch(
                task_id=task_id,
                module=module,
                encoded=encoded,
                documents=documents,
                role="train",
                device=device,
            ),
            validation_batch=lambda task_id=task_id, module=module: _task_batch(
                task_id=task_id,
                module=module,
                encoded=encoded,
                documents=documents,
                role="validation",
                device=device,
            ),
            epochs=epochs,
            patience=patience,
            partition_hash=partition_hash,
        )
        reports[task_id] = report
        calibrators[task_id] = calibrator

    folds = _load_tolokers_folds(root, documents)
    fold_states: dict[int, dict[str, Any]] = {}
    fold_reports: list[dict[str, Any]] = []
    fold_calibrators: list[dict[str, Any]] = []
    for fold in folds:
        torch.manual_seed(seed + 2_000 + int(fold["fold"]))
        module = torch.nn.Linear(128, 2).to(device)
        partition_hash = canonical_sha256({"fold": fold["fold"], "validation": fold["validation"]})
        report, calibrator = _fit_frozen_head(
            task_id=ACCOUNT_RISK_TASK,
            module=module,
            train_batch=lambda module=module, fold=fold: _task_batch(
                task_id=ACCOUNT_RISK_TASK,
                module=module,
                encoded=encoded,
                documents=documents,
                role="train",
                device=device,
                tolokers_indices=fold["train"],
            ),
            validation_batch=lambda module=module, fold=fold: _task_batch(
                task_id=ACCOUNT_RISK_TASK,
                module=module,
                encoded=encoded,
                documents=documents,
                role="validation",
                device=device,
                tolokers_indices=fold["validation"],
            ),
            epochs=epochs,
            patience=patience,
            partition_hash=partition_hash,
        )
        fold_states[int(fold["fold"])] = _clone_tensor_state(module)
        fold_reports.append({"fold": fold["fold"], **report})
        fold_calibration = {"fold": fold["fold"], "calibrator": calibrator}
        fold_calibration["wrapperHash"] = canonical_sha256(fold_calibration)
        fold_calibrators.append(fold_calibration)
    model.account_risk_head.load_state_dict(fold_states[0], strict=True)
    model.account_risk_head.eval()
    reports[ACCOUNT_RISK_TASK] = {
        "protocol": "official-10-splits/1.0",
        "officialSplits": fold_reports,
        "reportHash": canonical_sha256(fold_reports),
    }
    calibrators[ACCOUNT_RISK_TASK] = fold_calibrators[0]["calibrator"]
    return reports, calibrators, fold_states, tuple(fold_calibrators)


def _pretrain_validation_loss(*, model, documents, graphs, adapters, device: str) -> float:
    """Deterministic self-supervised validation, with no governance labels."""

    import torch
    from torch.nn import functional

    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for ordinal, domain in enumerate(sorted(documents)):
            bundle, labels, _entry = documents[domain]
            adapter = adapters[domain]
            adapter.eval()
            edge_index = graphs[domain].graph.edge_index
            assignments = bundle.split_manifest.assignments
            node_ids = {node.id for node in bundle.nodes}
            assignment_ids = {item.entity_id for item in assignments}
            if assignment_ids == node_ids:
                selected_ids = {item.entity_id for item in assignments if item.role == "validation"}
                selected_rows = [node.index for node in bundle.nodes if node.id in selected_ids]
                if not selected_rows:
                    continue
                field_mask = torch.zeros(
                    (len(bundle.nodes), len(adapter.field_names)),
                    dtype=torch.bool,
                    device=device,
                )
                field_mask[selected_rows] = True
                masked = adapter(field_mask)
                encoded = model.encode_domain(masked, edge_index, domain)
                decoded = model.decode_fields(encoded, edge_index, field_mask)
                generator = torch.Generator(device=device).manual_seed(
                    RESEARCH_SEED + 10_000 + ordinal
                )
                loss = adapter.reconstruction_loss(decoded, field_mask, generator=generator)
                values.append(float(loss.detach().cpu()))
                continue

            if domain == "email-eu-core":
                rows = (
                    *labels["partitions"]["validation"]["positives"],
                    *labels["partitions"]["validation"]["negatives"],
                )
                targets = [float(item["target"]) for item in rows]
                by_id = {node.id: node.index for node in bundle.nodes}
                pairs_values = [(by_id[item["sourceId"]], by_id[item["targetId"]]) for item in rows]
            else:
                validation_ids = {
                    item.entity_id for item in assignments if item.role == "validation"
                }
                by_label = {item["entityId"]: item for item in labels["targets"]}
                positives = [by_label[item] for item in sorted(validation_ids) if item in by_label]
                if not positives:
                    continue
                by_id = {node.id: node.index for node in bundle.nodes}
                positive_pairs = [
                    (by_id[item["sourceId"]], by_id[item["targetId"]]) for item in positives
                ]
                generator = torch.Generator(device=device).manual_seed(
                    RESEARCH_SEED + 20_000 + ordinal
                )
                negative_pairs = (
                    graphs[domain]
                    .graph.sample_negative_pairs(len(positive_pairs), generator=generator)
                    .detach()
                    .cpu()
                    .tolist()
                )
                pairs_values = [*positive_pairs, *negative_pairs]
                targets = [1.0] * len(positive_pairs) + [0.0] * len(negative_pairs)
            if not pairs_values:
                continue
            pairs = torch.tensor(pairs_values, dtype=torch.long, device=device)
            encoded = model.encode_domain(adapter(), edge_index, domain)
            logits = model.edge_reconstruction_logits(encoded, pairs, directed=bundle.directed)
            loss = functional.binary_cross_entropy_with_logits(
                logits, torch.tensor(targets, dtype=torch.float32, device=device)
            )
            values.append(float(loss.detach().cpu()))
    if not values:
        raise ValueError("self-supervised validation produced no observations")
    return math.fsum(values) / len(values)


def train_research_model(
    research_root: str | Path,
    *,
    device: str = "cpu",
    pretrain_epochs: int = 60,
    head_epochs: int = 100,
) -> Path:
    """Run real shared self-supervision and four-head fitting over the materialized corpus."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    import torch

    from socialgraph_gfm.runtime import set_seed

    from ...core.model import ResearchCoreGFM
    from ...core.trainer import CoreTrainer

    root = _safe_root(research_root)
    research_config = load_research_config()
    model_protocol = research_config["model"]
    manifest = load_corpus_manifest(root)
    output = root / "runs/shared"
    if output.exists():
        _load_trained_runtime(root, device="cpu")
        return output / "training-manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".shared.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        force_neighbor_fallback = False
        execution_device = device
        set_seed(RESEARCH_SEED, device=execution_device)
        documents, graphs, adapters = _prepare_training_runtime(
            root, manifest, device=execution_device
        )
        domains = tuple(sorted(graphs))
        model = ResearchCoreGFM(domains=domains).to(execution_device)
        if not 1 <= pretrain_epochs <= int(model_protocol["pretrainEpochs"]):
            raise ValueError("pretrain epochs exceed the pinned SocialGraph-FM Research maximum")
        if not 1 <= head_epochs <= int(model_protocol["headEpochs"]):
            raise ValueError("head epochs exceed the pinned SocialGraph-FM Research maximum")
        steps_per_epoch = len(graphs)
        config = ResearchTrainingConfig(
            max_steps=pretrain_epochs * steps_per_epoch,
            max_epochs=pretrain_epochs,
            head_max_epochs=head_epochs,
            patience=int(model_protocol["pretrainPatience"]),
            head_patience=int(model_protocol["headPatience"]),
            hidden_dim=int(model_protocol["hiddenDim"]),
            encoder_layers=int(model_protocol["encoderLayers"]),
            dropout=float(model_protocol["dropout"]),
            field_mask_rate=float(model_protocol["fieldMaskRate"]),
            edge_mask_rate=float(model_protocol["edgeMaskRate"]),
            full_batch_edge_threshold=int(model_protocol["fullBatchEdgeThreshold"]),
            node_batch_size=int(model_protocol["neighborFallback"]["batchSize"]),
            edge_batch_size=int(model_protocol["neighborFallback"]["batchSize"]),
            fanout=cast(
                tuple[int, int, int],
                tuple(int(value) for value in model_protocol["neighborFallback"]["fanout"]),
            ),
            learning_rate=float(model_protocol["learningRate"]),
            weight_decay=float(model_protocol["weightDecay"]),
        )
        trainer = CoreTrainer(
            model,
            graphs,
            config=config,  # type: ignore[arg-type]
            seed=RESEARCH_SEED,
        )
        history = []
        best_validation_loss = float("inf")
        best_epoch = 0
        stale_epochs = 0
        best_model_state = _clone_tensor_state(model)
        best_adapter_states = {
            domain: _clone_tensor_state(adapter) for domain, adapter in adapters.items()
        }
        validation_history: list[dict[str, Any]] = []
        epoch = 1
        while epoch <= pretrain_epochs:
            for adapter in adapters.values():
                adapter.train()
            try:
                history.extend(trainer.run_steps(steps_per_epoch))
                validation_loss = _pretrain_validation_loss(
                    model=model,
                    documents=documents,
                    graphs=graphs,
                    adapters=adapters,
                    device=execution_device,
                )
            except torch.OutOfMemoryError:
                if device != "cuda" or force_neighbor_fallback:
                    raise
                force_neighbor_fallback = True
                del trainer, model, graphs, adapters
                torch.cuda.empty_cache()
                execution_device = "cpu"
                set_seed(RESEARCH_SEED, device=execution_device)
                documents, graphs, adapters = _prepare_training_runtime(
                    root,
                    manifest,
                    device=execution_device,
                    force_neighbor_fallback=True,
                )
                model = ResearchCoreGFM(domains=domains).to(execution_device)
                trainer = CoreTrainer(
                    model,
                    graphs,
                    config=config,  # type: ignore[arg-type]
                    seed=RESEARCH_SEED,
                )
                history = []
                best_validation_loss = float("inf")
                best_epoch = 0
                stale_epochs = 0
                validation_history = []
                best_model_state = _clone_tensor_state(model)
                best_adapter_states = {
                    domain: _clone_tensor_state(adapter) for domain, adapter in adapters.items()
                }
                continue
            validation_history.append({"epoch": epoch, "loss": validation_loss})
            if validation_loss < best_validation_loss - 1e-10:
                best_validation_loss = validation_loss
                best_epoch = epoch
                stale_epochs = 0
                best_model_state = _clone_tensor_state(model)
                best_adapter_states = {
                    domain: _clone_tensor_state(adapter) for domain, adapter in adapters.items()
                }
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
            epoch += 1
        model.load_state_dict(best_model_state, strict=True)
        for domain, adapter in adapters.items():
            adapter.load_state_dict(best_adapter_states[domain], strict=True)
        head_device = "cpu"
        if execution_device != head_device:
            model.to(head_device)
            for adapter in adapters.values():
                adapter.to(head_device)
            _head_documents, head_graphs, _head_adapters = _prepare_training_runtime(
                root, manifest, device=head_device
            )
        else:
            head_graphs = graphs
        (
            head_reports,
            calibrators,
            tolokers_fold_states,
            tolokers_fold_calibrators,
        ) = _train_task_heads(
            root=root,
            model=model,
            documents=documents,
            graphs=head_graphs,
            adapters=adapters,
            device=head_device,
            epochs=head_epochs,
            patience=config.head_patience,
            seed=RESEARCH_SEED,
        )
        checkpoint_path = staging / "checkpoint.pt"
        checkpoint_payload = {
            "schemaVersion": "socialgraph-fm.research-checkpoint/1.0",
            "releaseId": RELEASE_ID,
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "transductive": True,
            "routeContract": route_contract(),
            "routeContractHash": _route_contract_hash(),
            "domains": domains,
            "modelState": model.state_dict(),
            "modelStateHash": _tensor_state_hash(model.state_dict()),
            "adapterSchemas": {
                domain: adapters[domain].schema.model_dump(mode="json", by_alias=True)
                for domain in domains
            },
            "adapterStates": {domain: adapters[domain].state_dict() for domain in domains},
            "adapterStateHashes": {
                domain: _tensor_state_hash(adapters[domain].state_dict()) for domain in domains
            },
            "calibrators": calibrators,
            "tolokersFoldHeadStates": tolokers_fold_states,
            "tolokersFoldCalibrators": tolokers_fold_calibrators,
            "corpusHash": manifest["corpusHash"],
            "researchConfigSha256": research_config["configSha256"],
            "trainingConfig": config.to_dict(),
            "headTrainingReports": head_reports,
            "executionMode": (
                "cpu-neighbor-fallback" if force_neighbor_fallback else "full-batch-first"
            ),
            "executionDevice": execution_device,
            "headDevice": head_device,
            "fallbackReason": (
                "CUDA_OUT_OF_MEMORY_FULL_RUN_RESTART" if force_neighbor_fallback else None
            ),
        }
        _torch_atomic_save(checkpoint_path, checkpoint_payload)
        training: dict[str, Any] = {
            "schemaVersion": TRAINING_SCHEMA,
            "releaseId": RELEASE_ID,
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "formalReadinessUnaffected": True,
            "variant": "multi-domain-shared-gfm",
            "routeContract": route_contract(),
            "routeContractHash": _route_contract_hash(),
            "corpusHash": manifest["corpusHash"],
            "researchConfigSha256": research_config["configSha256"],
            "config": config.to_dict(),
            "configHash": canonical_sha256(config.to_dict()),
            "optimizerSteps": len(history),
            "pretrainEpochsCompleted": len(validation_history),
            "pretrainBestEpoch": best_epoch,
            "pretrainBestValidationLoss": best_validation_loss,
            "pretrainStoppedEarly": len(validation_history) < pretrain_epochs,
            "pretrainValidationHistory": validation_history,
            "headMaxEpochs": head_epochs,
            "headPatience": config.head_patience,
            "executionMode": (
                "cpu-neighbor-fallback" if force_neighbor_fallback else "full-batch-first"
            ),
            "fallbackReason": (
                "CUDA_OUT_OF_MEMORY_FULL_RUN_RESTART" if force_neighbor_fallback else None
            ),
            "lastPretrainLoss": history[-1].loss,
            "headTrainingReports": head_reports,
            "calibrators": calibrators,
            "tolokersFoldCalibrationHash": canonical_sha256(tolokers_fold_calibrators),
            "modelStateHash": _tensor_state_hash(model.state_dict()),
            "adapterSchemaHashes": {
                domain: adapters[domain].schema.adapter_schema_hash for domain in domains
            },
            "checkpointPath": "checkpoint.pt",
            "checkpointSha256": file_sha256(checkpoint_path),
            "requestedDevice": device,
            "device": execution_device,
            "headDevice": head_device,
        }
        training["trainingHash"] = canonical_sha256(training)
        _atomic_json(staging / "training-manifest.json", training)
        os.replace(staging, output)
        return output / "training-manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


COMPARISON_SCHEMA = "socialgraph-fm.research-comparison-matrix/1.0"
COMPARISON_CHECKPOINT_SCHEMA = "socialgraph-fm.research-comparison-checkpoint/1.0"
COMPARISON_RECEIPT_SCHEMA = "socialgraph-fm.research-comparison-receipt/1.0"


def _comparison_cells(root: Path, documents) -> tuple[dict[str, Any], ...]:
    cells: list[dict[str, Any]] = [
        {
            "cellId": f"content-{domain}",
            "taskId": CONTENT_POLICY_TASK,
            "targetDomain": domain,
            "route": task_route_name(CONTENT_POLICY_TASK, domain),
            "fold": None,
        }
        for domain in sorted(item for item in documents if item.startswith("twitch-"))
    ]
    cells.extend(
        {
            "cellId": f"account-fold-{fold['fold']}",
            "taskId": ACCOUNT_RISK_TASK,
            "targetDomain": "tolokers",
            "route": task_route_name(ACCOUNT_RISK_TASK, "tolokers"),
            "fold": int(fold["fold"]),
        }
        for fold in _load_tolokers_folds(root, documents)
    )
    cells.extend(
        (
            {
                "cellId": "signed-wiki-rfa",
                "taskId": SIGNED_RELATION_TASK,
                "targetDomain": "wiki-rfa",
                "route": task_route_name(SIGNED_RELATION_TASK, "wiki-rfa"),
                "fold": None,
            },
            {
                "cellId": "collaboration-email-eu-core",
                "taskId": COLLABORATION_TASK,
                "targetDomain": "email-eu-core",
                "route": task_route_name(COLLABORATION_TASK, "email-eu-core"),
                "fold": None,
            },
        )
    )
    return tuple(cells)


def _cell_role_indices(
    *, root: Path, cell: Mapping[str, Any], documents, role: str
) -> tuple[int, ...] | None:
    if cell["taskId"] != ACCOUNT_RISK_TASK:
        return None
    fold = _load_tolokers_folds(root, documents)[int(cell["fold"])]
    return tuple(int(item) for item in fold[role])


def _cell_batch(
    *,
    root: Path,
    cell: Mapping[str, Any],
    model,
    documents,
    graphs,
    adapters,
    role: str,
    device: str,
):
    import torch

    task_id = cell["taskId"]
    domain = cell["targetDomain"]
    bundle, labels, _entry = documents[domain]
    encoded = model.encode_domain(
        adapters[domain](),
        graphs[domain].graph.edge_index,
        task_route_domain(task_id, domain),
    )
    if task_id in {CONTENT_POLICY_TASK, ACCOUNT_RISK_TASK}:
        binary_labels_by_id = {
            item["entityId"]: int(item["target"]) for item in labels["targets"]
        }
        if task_id == ACCOUNT_RISK_TASK:
            selected_indices = _cell_role_indices(
                root=root, cell=cell, documents=documents, role=role
            )
            assert selected_indices is not None
            selected_ids = [bundle.nodes[index].id for index in selected_indices]
        else:
            selected_ids = [
                item for item in _role_ids(bundle, role) if item in binary_labels_by_id
            ]
            selected_indices = tuple(
                next(node.index for node in bundle.nodes if node.id == item)
                for item in selected_ids
            )
        if not selected_indices:
            raise ValueError(f"comparison cell {cell['cellId']} has no {role} nodes")
        indices = torch.tensor(selected_indices, dtype=torch.long, device=device)
        head = (
            model.content_policy_head if task_id == CONTENT_POLICY_TASK else model.account_risk_head
        )
        return head(encoded[indices]), torch.tensor(
            [binary_labels_by_id[item] for item in selected_ids], device=device
        )
    by_id = {node.id: node.index for node in bundle.nodes}
    if task_id == SIGNED_RELATION_TASK:
        signed_labels_by_id = {item["entityId"]: item for item in labels["targets"]}
        selected = [item for item in _role_ids(bundle, role) if item in signed_labels_by_id]
        if not selected:
            raise ValueError(f"comparison Wiki cell has no {role} relations")
        pairs = torch.tensor(
            [
                (
                    by_id[signed_labels_by_id[item]["sourceId"]],
                    by_id[signed_labels_by_id[item]["targetId"]],
                )
                for item in selected
            ],
            dtype=torch.long,
            device=device,
        )
        return model.signed_edge_head(encoded, pairs), torch.tensor(
            [signed_labels_by_id[item]["target"] for item in selected],
            dtype=torch.float32,
            device=device,
        )
    partition = labels["partitions"][role]
    rows = (*partition["positives"], *partition["negatives"])
    if not rows:
        raise ValueError(f"comparison Email cell has no {role} pairs")
    pairs = torch.tensor(
        [(by_id[item["sourceId"]], by_id[item["targetId"]]) for item in rows],
        dtype=torch.long,
        device=device,
    )
    return model.collaboration_head(encoded, pairs), torch.tensor(
        [item["target"] for item in rows], dtype=torch.float32, device=device
    )


def _fit_comparison_cell(
    *,
    root: Path,
    cell: Mapping[str, Any],
    variant: str,
    model,
    documents,
    graphs,
    adapters,
    device: str,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from socialgraph_gfm.runtime import set_seed

    set_seed(seed, device=device)
    domain = cell["targetDomain"]
    parameters = [*model.parameters(), *adapters[domain].parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_model_state = _clone_tensor_state(model)
    best_adapter_state = _clone_tensor_state(adapters[domain])
    for epoch in range(1, max_epochs + 1):
        model.train()
        adapters[domain].train()
        logits, targets = _cell_batch(
            root=root,
            cell=cell,
            model=model,
            documents=documents,
            graphs=graphs,
            adapters=adapters,
            role="train",
            device=device,
        )
        loss = _loss_for_logits(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        adapters[domain].eval()
        with torch.inference_mode():
            validation_logits, validation_targets = _cell_batch(
                root=root,
                cell=cell,
                model=model,
                documents=documents,
                graphs=graphs,
                adapters=adapters,
                role="validation",
                device=device,
            )
            validation_loss = float(
                _loss_for_logits(validation_logits, validation_targets).detach().cpu()
            )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_model_state = _clone_tensor_state(model)
            best_adapter_state = _clone_tensor_state(adapters[domain])
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_model_state, strict=True)
    adapters[domain].load_state_dict(best_adapter_state, strict=True)
    model.eval()
    adapters[domain].eval()
    with torch.inference_mode():
        validation_logits, validation_targets = _cell_batch(
            root=root,
            cell=cell,
            model=model,
            documents=documents,
            graphs=graphs,
            adapters=adapters,
            role="validation",
            device=device,
        )
    partition_hash = canonical_sha256(
        {
            "cellId": cell["cellId"],
            "role": "validation",
            "fold": cell["fold"],
            "targetsHash": canonical_sha256(validation_targets.detach().cpu().tolist()),
        }
    )
    calibrator = _fit_research_calibrator(
        task_id=cell["taskId"],
        logits=validation_logits,
        targets=validation_targets,
        partition_hash=partition_hash,
    )
    report: dict[str, Any] = {
        "cellId": cell["cellId"],
        "variant": variant,
        "maxEpochs": max_epochs,
        "patience": patience,
        "bestEpoch": best_epoch,
        "bestValidationLoss": best_loss,
        "stoppedEarly": best_epoch + stale < max_epochs,
        "modelStateHash": _tensor_state_hash(model.state_dict()),
        "adapterStateHash": _tensor_state_hash(adapters[domain].state_dict()),
        "calibrationArtifactHash": calibrator["artifactHash"],
        "downstreamSeed": seed,
    }
    report["reportHash"] = canonical_sha256(report)
    return report, calibrator


def _pretrain_comparison_variant(
    *,
    model,
    documents,
    graphs,
    adapters,
    source_domains: tuple[str, ...],
    device: str,
    max_epochs: int,
    patience: int,
    seed: int,
) -> dict[str, Any]:
    from socialgraph_gfm.runtime import set_seed

    from ...core.trainer import CoreTrainer

    set_seed(seed, device=device)
    if not source_domains:
        return {
            "epochsCompleted": 0,
            "bestEpoch": 0,
            "bestValidationLoss": None,
            "optimizerSteps": 0,
            "pretrainSeed": seed,
        }
    selected_graphs = {domain: graphs[domain] for domain in source_domains}
    config = ResearchTrainingConfig(
        max_steps=max_epochs * len(selected_graphs),
        max_epochs=max_epochs,
    )
    trainer = CoreTrainer(model, selected_graphs, config=config, seed=seed)  # type: ignore[arg-type]
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    steps = 0
    best_model_state = _clone_tensor_state(model)
    best_adapter_states = {
        domain: _clone_tensor_state(adapters[domain]) for domain in source_domains
    }
    selected_documents = {domain: documents[domain] for domain in source_domains}
    selected_adapters = {domain: adapters[domain] for domain in source_domains}
    for epoch in range(1, max_epochs + 1):
        for adapter in selected_adapters.values():
            adapter.train()
        trainer.run_steps(len(selected_graphs))
        steps += len(selected_graphs)
        validation_loss = _pretrain_validation_loss(
            model=model,
            documents=selected_documents,
            graphs=selected_graphs,
            adapters=selected_adapters,
            device=device,
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_model_state = _clone_tensor_state(model)
            best_adapter_states = {
                domain: _clone_tensor_state(adapters[domain]) for domain in source_domains
            }
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_model_state, strict=True)
    for domain, state in best_adapter_states.items():
        adapters[domain].load_state_dict(state, strict=True)
    return {
        "epochsCompleted": best_epoch + stale,
        "bestEpoch": best_epoch,
        "bestValidationLoss": best_loss,
        "optimizerSteps": steps,
        "stoppedEarly": best_epoch + stale < max_epochs,
        "pretrainSeed": seed,
    }


def _train_research_comparison_once(
    research_root: str | Path,
    *,
    device: str = "cpu",
    pretrain_epochs: int = 60,
    downstream_epochs: int = 100,
    force_neighbor_fallback: bool,
) -> Path:
    """Train the genuine scratch/single-domain/target-excluded matrix without reading test."""

    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if not 1 <= pretrain_epochs <= 60 or not 1 <= downstream_epochs <= 100:
        raise ValueError("comparison epoch limits exceed SocialGraph-FM Research protocol")
    import torch

    from socialgraph_gfm.runtime import set_seed

    from ...core.model import ResearchCoreGFM

    root = _safe_root(research_root)
    execution_device = "cpu" if force_neighbor_fallback else device
    research_config = load_research_config()
    corpus = load_corpus_manifest(root)
    output = root / "runs/comparisons"
    if output.exists():
        raise FileExistsError(f"research comparison run already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / (".cmp-neighbor" if force_neighbor_fallback else ".cmp-full")
    staging.mkdir(exist_ok=True)
    try:
        set_seed(RESEARCH_SEED, device=execution_device)
        documents, graphs, adapters = _prepare_training_runtime(
            root,
            corpus,
            device=execution_device,
            force_neighbor_fallback=force_neighbor_fallback,
        )
        domains = tuple(sorted(documents))
        model = ResearchCoreGFM(domains=domains).to(execution_device)
        base_model_state = _clone_tensor_state(model)
        base_adapter_states = {
            domain: _clone_tensor_state(adapter) for domain, adapter in adapters.items()
        }
        cells = _comparison_cells(root, documents)
        rows: list[dict[str, Any]] = []
        variants = (
            "graphsage-scratch",
            "single-domain-masked-pretrain",
            "target-excluded-shared-gfm",
        )
        for target_ordinal, target_domain in enumerate(domains):
            target_cells = tuple(cell for cell in cells if cell["targetDomain"] == target_domain)
            if not target_cells:
                continue
            variant_initializations: dict[
                str, tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]
            ] = {}
            pretrain_seed = RESEARCH_SEED + 50_000 + target_ordinal
            for variant in variants:
                model.load_state_dict(base_model_state, strict=True)
                for domain, adapter in adapters.items():
                    adapter.load_state_dict(base_adapter_states[domain], strict=True)
                source_domains = (
                    ()
                    if variant == "graphsage-scratch"
                    else (target_domain,)
                    if variant == "single-domain-masked-pretrain"
                    else tuple(domain for domain in domains if domain != target_domain)
                )
                cache_name = (
                    canonical_sha256({"targetDomain": target_domain, "variant": variant})[:20]
                    + ".pt"
                )
                cache_path = staging / "pretrain" / cache_name
                if cache_path.is_file():
                    cached = torch.load(
                        cache_path, map_location=execution_device, weights_only=True
                    )
                    if (
                        cached.get("schemaVersion")
                        != "socialgraph-fm.research-comparison-pretrain-cache/1.0"
                        or cached.get("corpusHash") != corpus["corpusHash"]
                        or cached.get("researchConfigSha256") != research_config["configSha256"]
                        or cached.get("targetDomain") != target_domain
                        or cached.get("variant") != variant
                        or tuple(cached.get("sourceDomains", ())) != source_domains
                        or cached.get("pretrainSeed") != pretrain_seed
                        or cached.get("executionMode")
                        != (
                            "cpu-neighbor-fallback"
                            if force_neighbor_fallback
                            else "full-batch-first"
                        )
                        or cached.get("modelStateHash") != _tensor_state_hash(cached["modelState"])
                    ):
                        raise ValueError("research comparison pretrain cache mismatch")
                    adapter_states = cached["adapterStates"]
                    if cached.get("adapterStateHash") != canonical_sha256(
                        {
                            domain: _tensor_state_hash(state)
                            for domain, state in sorted(adapter_states.items())
                        }
                    ):
                        raise ValueError("research comparison adapter cache mismatch")
                    variant_initializations[variant] = (
                        cached["modelState"],
                        adapter_states,
                        cached["pretrainingReport"],
                    )
                    continue
                pretrain_report = _pretrain_comparison_variant(
                    model=model,
                    documents=documents,
                    graphs=graphs,
                    adapters=adapters,
                    source_domains=source_domains,
                    device=execution_device,
                    max_epochs=pretrain_epochs,
                    patience=8,
                    seed=pretrain_seed,
                )
                cached_model_state = _clone_tensor_state(model)
                cached_adapter_states = {
                    domain: _clone_tensor_state(adapter) for domain, adapter in adapters.items()
                }
                cached_report = {"sourceDomains": source_domains, **pretrain_report}
                cache_payload = {
                    "schemaVersion": ("socialgraph-fm.research-comparison-pretrain-cache/1.0"),
                    "corpusHash": corpus["corpusHash"],
                    "researchConfigSha256": research_config["configSha256"],
                    "targetDomain": target_domain,
                    "variant": variant,
                    "sourceDomains": source_domains,
                    "pretrainSeed": pretrain_seed,
                    "executionMode": (
                        "cpu-neighbor-fallback" if force_neighbor_fallback else "full-batch-first"
                    ),
                    "modelState": cached_model_state,
                    "modelStateHash": _tensor_state_hash(cached_model_state),
                    "adapterStates": cached_adapter_states,
                    "adapterStateHash": canonical_sha256(
                        {
                            domain: _tensor_state_hash(state)
                            for domain, state in sorted(cached_adapter_states.items())
                        }
                    ),
                    "pretrainingReport": cached_report,
                }
                _torch_atomic_save(cache_path, cache_payload)
                variant_initializations[variant] = (
                    cached_model_state,
                    cached_adapter_states,
                    cached_report,
                )
            for cell in target_cells:
                for variant in variants:
                    model_state, adapter_states, pretrain_report = variant_initializations[variant]
                    model.load_state_dict(model_state, strict=True)
                    for domain, adapter in adapters.items():
                        adapter.load_state_dict(adapter_states[domain], strict=True)
                    protocol: dict[str, Any] = {
                        "schemaVersion": "socialgraph-fm.research-comparison-protocol/1.0",
                        "cellId": cell["cellId"],
                        "taskId": cell["taskId"],
                        "targetDomain": target_domain,
                        "route": cell["route"],
                        "fold": cell["fold"],
                        "variant": variant,
                        "pretrainingSourceDomains": list(pretrain_report["sourceDomains"]),
                        "targetExcluded": (
                            variant == "target-excluded-shared-gfm"
                            and target_domain not in pretrain_report["sourceDomains"]
                        ),
                        "sameDownstreamPolicy": True,
                        "downstreamOptimization": "end-to-end-adapter-route-backbone-head",
                        "learningRate": 1e-3,
                        "weightDecay": 1e-4,
                        "maxEpochs": downstream_epochs,
                        "patience": 10,
                        "selectionRole": "validation",
                        "testReadDuringTraining": False,
                        "executionMode": (
                            "cpu-neighbor-fallback"
                            if force_neighbor_fallback
                            else "full-batch-first"
                        ),
                        "fallbackReason": (
                            "CUDA_OUT_OF_MEMORY_FULL_MATRIX_RESTART"
                            if force_neighbor_fallback
                            else None
                        ),
                        "seed": RESEARCH_SEED,
                        "pretrainSeed": pretrain_seed,
                        "downstreamSeed": (
                            RESEARCH_SEED
                            + 60_000
                            + target_ordinal * 100
                            + (0 if cell["fold"] is None else int(cell["fold"]))
                        ),
                        "corpusHash": corpus["corpusHash"],
                        "researchConfigSha256": research_config["configSha256"],
                        "targetGraphVersionHash": documents[target_domain][0].graph_version_hash,
                    }
                    protocol["protocolHash"] = canonical_sha256(protocol)
                    checkpoint_name = (
                        canonical_sha256({"cellId": cell["cellId"], "variant": variant})[:20]
                        + ".pt"
                    )
                    receipt_path = (
                        staging / "receipts" / (checkpoint_name.removesuffix(".pt") + ".json")
                    )
                    if receipt_path.is_file():
                        receipt = _read_hashed_document(
                            receipt_path,
                            schema=COMPARISON_RECEIPT_SCHEMA,
                            hash_field="receiptHash",
                        )
                        resumed = receipt.get("run")
                        if not isinstance(resumed, dict):
                            raise ValueError("research comparison receipt lacks a run")
                        resumed_checkpoint = staging / resumed["checkpointPath"]
                        if (
                            receipt.get("corpusHash") != corpus["corpusHash"]
                            or receipt.get("researchConfigSha256")
                            != research_config["configSha256"]
                            or receipt.get("executionMode") != protocol["executionMode"]
                            or resumed.get("cellId") != cell["cellId"]
                            or resumed.get("variant") != variant
                            or resumed.get("protocolHash") != protocol["protocolHash"]
                            or file_sha256(resumed_checkpoint) != resumed.get("checkpointSha256")
                        ):
                            raise ValueError("research comparison resume receipt mismatch")
                        resumed_payload = torch.load(
                            resumed_checkpoint, map_location="cpu", weights_only=True
                        )
                        if (
                            resumed_payload.get("schemaVersion") != COMPARISON_CHECKPOINT_SCHEMA
                            or resumed_payload.get("corpusHash") != corpus["corpusHash"]
                            or resumed_payload.get("researchConfigSha256")
                            != research_config["configSha256"]
                            or resumed_payload.get("protocol", {}).get("protocolHash")
                            != protocol["protocolHash"]
                        ):
                            raise ValueError("research comparison resumed checkpoint mismatch")
                        rows.append(resumed)
                        continue
                    report, calibrator = _fit_comparison_cell(
                        root=root,
                        cell=cell,
                        variant=variant,
                        model=model,
                        documents=documents,
                        graphs=graphs,
                        adapters=adapters,
                        device=execution_device,
                        max_epochs=downstream_epochs,
                        patience=10,
                        seed=protocol["downstreamSeed"],
                    )
                    checkpoint_path = staging / "checkpoints" / checkpoint_name
                    _torch_atomic_save(
                        checkpoint_path,
                        {
                            "schemaVersion": COMPARISON_CHECKPOINT_SCHEMA,
                            "seed": RESEARCH_SEED,
                            "corpusHash": corpus["corpusHash"],
                            "researchConfigSha256": research_config["configSha256"],
                            "cell": dict(cell),
                            "variant": variant,
                            "protocol": protocol,
                            "pretrainingReport": pretrain_report,
                            "downstreamReport": report,
                            "calibrator": calibrator,
                            "domains": domains,
                            "modelState": model.state_dict(),
                            "adapterSchema": adapters[target_domain].schema.model_dump(
                                mode="json", by_alias=True
                            ),
                            "adapterState": adapters[target_domain].state_dict(),
                        },
                    )
                    run_row = {
                        "cellId": cell["cellId"],
                        "taskId": cell["taskId"],
                        "targetDomain": target_domain,
                        "route": cell["route"],
                        "fold": cell["fold"],
                        "variant": variant,
                        "protocolHash": protocol["protocolHash"],
                        "pretrainingReport": pretrain_report,
                        "downstreamReportHash": report["reportHash"],
                        "calibrationArtifactHash": calibrator["artifactHash"],
                        "checkpointPath": f"checkpoints/{checkpoint_name}",
                        "checkpointSha256": file_sha256(checkpoint_path),
                    }
                    receipt_payload: dict[str, Any] = {
                        "schemaVersion": COMPARISON_RECEIPT_SCHEMA,
                        "corpusHash": corpus["corpusHash"],
                        "researchConfigSha256": research_config["configSha256"],
                        "executionMode": protocol["executionMode"],
                        "run": run_row,
                    }
                    receipt_payload["receiptHash"] = canonical_sha256(receipt_payload)
                    _atomic_json(receipt_path, receipt_payload)
                    rows.append(run_row)
        rows.sort(key=lambda item: (item["cellId"], item["variant"]))
        manifest: dict[str, Any] = {
            "schemaVersion": COMPARISON_SCHEMA,
            "releaseId": RELEASE_ID,
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "formalReadinessUnaffected": True,
            "corpusHash": corpus["corpusHash"],
            "researchConfigSha256": research_config["configSha256"],
            "variants": list(variants),
            "cellCount": len(cells),
            "runCount": len(rows),
            "pretrainMaxEpochs": pretrain_epochs,
            "pretrainPatience": 8,
            "downstreamMaxEpochs": downstream_epochs,
            "downstreamPatience": 10,
            "testReadDuringTraining": False,
            "executionMode": (
                "cpu-neighbor-fallback" if force_neighbor_fallback else "full-batch-first"
            ),
            "fallbackReason": (
                "CUDA_OUT_OF_MEMORY_FULL_MATRIX_RESTART" if force_neighbor_fallback else None
            ),
            "fallbackDevice": (execution_device if force_neighbor_fallback else None),
            "runs": rows,
        }
        if manifest["cellCount"] != 18 or manifest["runCount"] != 54:
            raise ValueError("SocialGraph-FM Research comparison matrix must contain 18 cells and 54 runs")
        manifest["matrixHash"] = canonical_sha256(manifest)
        _atomic_json(staging / "matrix-manifest.json", manifest)
        shutil.rmtree(staging / "pretrain", ignore_errors=True)
        shutil.rmtree(staging / "receipts", ignore_errors=True)
        os.replace(staging, output)
        return output / "matrix-manifest.json"
    except torch.OutOfMemoryError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except BaseException:
        # Validated per-run receipts remain for deterministic recovery.
        raise


def _comparison_full_attempt(
    root: Path, *, device: str, pretrain_epochs: int, downstream_epochs: int
) -> tuple[Path | None, bool]:
    import torch

    try:
        return (
            _train_research_comparison_once(
                root,
                device=device,
                pretrain_epochs=pretrain_epochs,
                downstream_epochs=downstream_epochs,
                force_neighbor_fallback=False,
            ),
            False,
        )
    except torch.OutOfMemoryError:
        return None, True


def train_research_comparison_matrix(
    research_root: str | Path,
    *,
    device: str = "cpu",
    pretrain_epochs: int = 60,
    downstream_epochs: int = 100,
) -> Path:
    root = _safe_root(research_root)
    completed = root / "runs/comparisons/matrix-manifest.json"
    if completed.is_file():
        load_comparison_manifest(root)
        return completed
    result, out_of_memory = _comparison_full_attempt(
        root,
        device=device,
        pretrain_epochs=pretrain_epochs,
        downstream_epochs=downstream_epochs,
    )
    if not out_of_memory:
        assert result is not None
        return result
    if device != "cuda":
        raise RuntimeError("unexpected CPU out-of-memory comparison state")
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return _train_research_comparison_once(
        root,
        device=device,
        pretrain_epochs=pretrain_epochs,
        downstream_epochs=downstream_epochs,
        force_neighbor_fallback=True,
    )


def load_comparison_manifest(research_root: str | Path) -> dict[str, Any]:
    root = _safe_root(research_root)
    manifest = _read_hashed_document(
        root / "runs/comparisons/matrix-manifest.json",
        schema=COMPARISON_SCHEMA,
        hash_field="matrixHash",
    )
    if manifest.get("cellCount") != 18 or manifest.get("runCount") != 54:
        raise ValueError("research comparison matrix inventory is incomplete")
    if manifest.get("researchConfigSha256") != load_research_config()["configSha256"]:
        raise ValueError("research comparison matrix configuration identity mismatch")
    corpus = load_corpus_manifest(root)
    documents = _load_graph_documents(root, corpus)
    cells = _comparison_cells(root, documents)
    expected_cells = {item["cellId"]: item for item in cells}
    expected_variants = {
        "graphsage-scratch",
        "single-domain-masked-pretrain",
        "target-excluded-shared-gfm",
    }
    if set(manifest.get("variants", ())) != expected_variants:
        raise ValueError("research comparison variant inventory mismatch")
    expected_inventory = {
        (cell_id, variant) for cell_id in expected_cells for variant in expected_variants
    }
    observed_inventory = [(item.get("cellId"), item.get("variant")) for item in manifest["runs"]]
    if (
        len(observed_inventory) != len(set(observed_inventory))
        or set(observed_inventory) != expected_inventory
    ):
        raise ValueError("research comparison cell/variant inventory mismatch")
    for item in manifest["runs"]:
        expected = expected_cells[item["cellId"]]
        if (
            item.get("taskId") != expected["taskId"]
            or item.get("targetDomain") != expected["targetDomain"]
            or item.get("route") != expected["route"]
            or item.get("fold") != expected["fold"]
        ):
            raise ValueError("research comparison row identity mismatch")
        checkpoint = (root / "runs/comparisons" / item["checkpointPath"]).resolve()
        if not checkpoint.is_relative_to((root / "runs/comparisons").resolve()):
            raise ValueError("research comparison checkpoint path escapes run root")
        if file_sha256(checkpoint) != item["checkpointSha256"]:
            raise ValueError("research comparison checkpoint hash mismatch")
    return manifest

COMPAT_EXPORTS = (
    'ResearchTrainingConfig',
    '_tensor_state_hash',
    '_torch_atomic_save',
    '_bundle_edge_index',
    '_load_graph_documents',
    '_prepare_training_runtime',
    '_role_ids',
    '_clone_tensor_state',
    '_binary_logit',
    '_loss_for_logits',
    '_ece',
    '_calibration_metrics',
    '_calibration_is_adequate',
    '_fit_research_calibrator',
    '_task_batch',
    '_fit_frozen_head',
    '_load_tolokers_folds',
    '_train_task_heads',
    '_pretrain_validation_loss',
    'train_research_model',
    'COMPARISON_SCHEMA',
    'COMPARISON_CHECKPOINT_SCHEMA',
    'COMPARISON_RECEIPT_SCHEMA',
    '_comparison_cells',
    '_cell_role_indices',
    '_cell_batch',
    '_fit_comparison_cell',
    '_pretrain_comparison_variant',
    '_train_research_comparison_once',
    '_comparison_full_attempt',
    'train_research_comparison_matrix',
    'load_comparison_manifest',
)

__all__ = [
    'load_comparison_manifest',
    'train_research_comparison_matrix',
    'train_research_model',
]
