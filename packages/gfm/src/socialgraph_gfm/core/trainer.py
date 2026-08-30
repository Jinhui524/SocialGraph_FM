"""Deterministic graph-balanced core training and recovery."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import marshal
import math
import os
import re
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from torch import Tensor

from .adapters import AdapterSchema, BundleInputAdapter
from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest
from .checkpoint import CheckpointBindings, load_checkpoint, publish_checkpoint
from .config import TrainingConfig
from .model import CoreGFM
from .objectives import (
    combine_objective_losses,
    domain_alignment_loss,
    mask_feature_fields,
    mask_paired_edges,
)
from .training_data import (
    BalancedDomainSampler,
    ExecutionPolicy,
    InsufficientNegativeCapacityError,
    NeighborBatchSource,
    PreparedGraph,
)


@dataclass(frozen=True)
class TrainingGraph:
    graph: PreparedGraph
    features: Tensor | None = None
    adapter: BundleInputAdapter | None = None
    execution_policy: ExecutionPolicy = field(default_factory=ExecutionPolicy)
    loader_kind: Literal["node", "link"] = "node"
    edge_label_index: Tensor | None = None

    def __post_init__(self) -> None:
        if (self.features is None) == (self.adapter is None):
            raise ValueError("training graph requires exactly one feature tensor or bundle adapter")
        if self.features is not None and self.features.shape != (self.graph.num_nodes, 128):
            raise ValueError("training graph features must have shape [num_nodes, 128]")
        if self.adapter is not None and self.adapter.num_nodes != self.graph.num_nodes:
            raise ValueError("bundle adapter node count does not match prepared graph")
        if self.loader_kind == "link" and self.edge_label_index is None:
            raise ValueError("link training graph requires edge_label_index")

    @classmethod
    def from_bundle(
        cls,
        *,
        adapter: BundleInputAdapter,
        graph: PreparedGraph,
        execution_policy: ExecutionPolicy | None = None,
        loader_kind: Literal["node", "link"] = "node",
        edge_label_index: Tensor | None = None,
    ) -> TrainingGraph:
        return cls(
            graph=graph,
            adapter=adapter,
            execution_policy=ExecutionPolicy() if execution_policy is None else execution_policy,
            loader_kind=loader_kind,
            edge_label_index=edge_label_index,
        )


@dataclass(frozen=True)
class StepResult:
    optimizer_step: int
    domain: str
    batch_index: int
    loss: float
    alignment_weight: float
    batch_ordinal: int
    batch_num_nodes: int
    execution_mode: str
    objective_signature: str


@dataclass(frozen=True)
class TrainingFitReport:
    status: Literal["complete", "early-stopped", "timeout-non-promotable"]
    latest_step: int
    best_step: int | None
    best_metric: float | None
    latest_checkpoint_path: Path
    best_checkpoint_path: Path | None


_VALIDATION_CONTRACT_SEAL = object()


@dataclass(frozen=True, init=False)
class ValidationContract:
    """Factory-sealed identities derived from exact validation evidence documents."""

    protocol_hash: str
    data_hash: str
    partition_hash: str
    callback_artifact_hash: str
    contract_hash: str
    _seal: object

    @classmethod
    def from_artifacts(
        cls,
        *,
        protocol: Mapping[str, Any],
        data: Mapping[str, Any],
        partition: Mapping[str, Any],
        callback: Mapping[str, Any],
    ) -> ValidationContract:
        hashes = {
            "protocolHash": canonical_sha256(protocol),
            "dataHash": canonical_sha256(data),
            "partitionHash": canonical_sha256(partition),
            "callbackArtifactHash": canonical_sha256(callback),
        }
        contract = object.__new__(cls)
        object.__setattr__(contract, "protocol_hash", hashes["protocolHash"])
        object.__setattr__(contract, "data_hash", hashes["dataHash"])
        object.__setattr__(contract, "partition_hash", hashes["partitionHash"])
        object.__setattr__(contract, "callback_artifact_hash", hashes["callbackArtifactHash"])
        object.__setattr__(contract, "contract_hash", canonical_sha256(hashes))
        object.__setattr__(contract, "_seal", _VALIDATION_CONTRACT_SEAL)
        return contract

    def verify(self) -> None:
        if type(self) is not ValidationContract or self._seal is not _VALIDATION_CONTRACT_SEAL:
            raise TypeError("validation contract must come from the artifact factory")
        expected = canonical_sha256(
            {
                "protocolHash": self.protocol_hash,
                "dataHash": self.data_hash,
                "partitionHash": self.partition_hash,
                "callbackArtifactHash": self.callback_artifact_hash,
            }
        )
        if self.contract_hash != expected:
            raise ValueError("validation contract hash does not match its artifact identities")


@dataclass(frozen=True)
class _ParsedFitState:
    best_step: int | None
    best_metric: float | None
    best_model_state_hash: str | None
    stale_validations: int
    last_validation_step: int | None
    last_validation_metric: float | None
    last_model_state_hash: str | None
    validation_protocol_hash: str | None
    validation_data_hash: str | None
    validation_partition_hash: str | None
    validation_callback_hash: str | None
    best_checkpoint_name: str | None
    best_checkpoint_sha256: str | None


_FIT_STATE_SCHEMA = "socialgraph-fm.core-fit-state/2.0"
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _model_state_hash(state: Mapping[str, Any]) -> str:
    if not all(
        isinstance(name, str) and isinstance(value, Tensor) for name, value in state.items()
    ):
        raise ValueError("checkpoint model state is not a tensor mapping")
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(state.items())}
    )


def _callback_state_value(value: Any) -> Any:
    if isinstance(value, Tensor):
        return {"tensor": canonical_tensor_digest(value)}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _callback_state_value(dataclasses.asdict(value))
    if value is None or isinstance(value, (str, int, bool, float, Path)):
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                normalized_key = f"str:{key}"
            elif type(key) is int:
                normalized_key = f"int:{key}"
            else:
                raise ValueError(
                    "validation callback state mappings require string or integer keys"
                )
            if normalized_key in normalized:
                raise ValueError("validation callback state mapping keys are ambiguous")
            normalized[normalized_key] = _callback_state_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [_callback_state_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        values = [_callback_state_value(item) for item in value]
        return sorted(values, key=canonical_json)
    raise ValueError(f"validation callback state of type {type(value).__name__} is not auditable")


def _callback_identity(
    callback: Callable[[CoreGFM], float], *, callback_artifact_hash: str
) -> str:
    candidate: Any = callback
    code = getattr(candidate, "__code__", None)
    state: Any
    if code is None:
        candidate = getattr(callback, "__call__", None)
        code = getattr(candidate, "__code__", None)
        state = _callback_state_value(vars(callback))
    else:
        closure = getattr(callback, "__closure__", None) or ()
        state = {
            "defaults": _callback_state_value(getattr(callback, "__defaults__", None)),
            "kwdefaults": _callback_state_value(getattr(callback, "__kwdefaults__", None)),
            "closure": [_callback_state_value(cell.cell_contents) for cell in closure],
        }
    if code is None:
        raise ValueError("validation callback must expose auditable Python code")
    return canonical_sha256(
        {
            "module": getattr(candidate, "__module__", type(callback).__module__),
            "qualname": getattr(candidate, "__qualname__", type(callback).__qualname__),
            "codeSha256": hashlib.sha256(marshal.dumps(code)).hexdigest(),
            "state": state,
            "callbackArtifactHash": callback_artifact_hash,
        }
    )


def _validation_context_hash(
    *,
    protocol_hash: str,
    data_hash: str,
    partition_hash: str,
    callback_hash: str,
) -> str:
    return canonical_sha256(
        {
            "protocolHash": protocol_hash,
            "dataHash": data_hash,
            "partitionHash": partition_hash,
            "callbackHash": callback_hash,
        }
    )


def _validation_identity(
    step: int,
    metric: float,
    *,
    model_state_hash: str,
    context_hash: str,
) -> str:
    return canonical_sha256(
        {
            "optimizerStep": step,
            "validationMetric": metric,
            "modelStateHash": model_state_hash,
            "validationContextHash": context_hash,
        }
    )


def _fit_state_payload(
    *,
    best_step: int | None,
    best_metric: float | None,
    best_model_state_hash: str | None,
    stale_validations: int,
    last_validation_step: int | None,
    last_validation_metric: float | None,
    last_model_state_hash: str | None,
    validation_protocol_hash: str | None,
    validation_data_hash: str | None,
    validation_partition_hash: str | None,
    validation_callback_hash: str | None,
    checkpoint_model_state_hash: str,
    best_checkpoint_name: str | None,
    best_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    context_values = (
        validation_protocol_hash,
        validation_data_hash,
        validation_partition_hash,
        validation_callback_hash,
    )
    context_hash = (
        None
        if all(value is None for value in context_values)
        else _validation_context_hash(
            protocol_hash=str(validation_protocol_hash),
            data_hash=str(validation_data_hash),
            partition_hash=str(validation_partition_hash),
            callback_hash=str(validation_callback_hash),
        )
    )
    best_identity = (
        None
        if best_step is None or best_metric is None or best_model_state_hash is None
        else _validation_identity(
            best_step,
            best_metric,
            model_state_hash=best_model_state_hash,
            context_hash=str(context_hash),
        )
    )
    last_identity = (
        None
        if (
            last_validation_step is None
            or last_validation_metric is None
            or last_model_state_hash is None
        )
        else _validation_identity(
            last_validation_step,
            last_validation_metric,
            model_state_hash=last_model_state_hash,
            context_hash=str(context_hash),
        )
    )
    payload: dict[str, Any] = {
        "schemaVersion": _FIT_STATE_SCHEMA,
        "bestStep": best_step,
        "bestMetric": best_metric,
        "bestModelStateHash": best_model_state_hash,
        "bestValidationHash": best_identity,
        "staleValidations": stale_validations,
        "lastValidationStep": last_validation_step,
        "lastValidationMetric": last_validation_metric,
        "lastModelStateHash": last_model_state_hash,
        "lastValidationHash": last_identity,
        "validationProtocolHash": validation_protocol_hash,
        "validationDataHash": validation_data_hash,
        "validationPartitionHash": validation_partition_hash,
        "validationCallbackHash": validation_callback_hash,
        "validationContextHash": context_hash,
        "checkpointModelStateHash": checkpoint_model_state_hash,
        "bestCheckpointName": best_checkpoint_name,
        "bestCheckpointSha256": best_checkpoint_sha256,
    }
    payload["stateHash"] = canonical_sha256(payload)
    return payload


def _parse_fit_state(
    payload: object,
    *,
    optimizer_step: int,
    config: TrainingConfig,
    checkpoint_model_state_hash: str,
) -> _ParsedFitState:
    if payload is None:
        return _ParsedFitState(
            None, None, None, 0, None, None, None, None, None, None, None, None, None
        )
    if not isinstance(payload, dict) or set(payload) != {
        "schemaVersion",
        "bestStep",
        "bestMetric",
        "bestModelStateHash",
        "bestValidationHash",
        "staleValidations",
        "lastValidationStep",
        "lastValidationMetric",
        "lastModelStateHash",
        "lastValidationHash",
        "validationProtocolHash",
        "validationDataHash",
        "validationPartitionHash",
        "validationCallbackHash",
        "validationContextHash",
        "checkpointModelStateHash",
        "bestCheckpointName",
        "bestCheckpointSha256",
        "stateHash",
    }:
        raise ValueError("checkpoint fit state has an invalid field inventory")
    if payload["schemaVersion"] != _FIT_STATE_SCHEMA:
        raise ValueError("checkpoint fit state schema is unsupported")
    expected_hash = canonical_sha256(
        {key: value for key, value in payload.items() if key != "stateHash"}
    )
    if payload["stateHash"] != expected_hash:
        raise ValueError("checkpoint fit state hash does not match its evidence")
    best_step = payload["bestStep"]
    best_metric = payload["bestMetric"]
    best_model_hash = payload["bestModelStateHash"]
    stale = payload["staleValidations"]
    last_step = payload["lastValidationStep"]
    last_metric = payload["lastValidationMetric"]
    last_model_hash = payload["lastModelStateHash"]
    context_values = (
        payload["validationProtocolHash"],
        payload["validationDataHash"],
        payload["validationPartitionHash"],
        payload["validationCallbackHash"],
    )
    if not (
        all(value is None for value in context_values)
        or all(
            isinstance(value, str) and _HASH_PATTERN.fullmatch(value) for value in context_values
        )
    ):
        raise ValueError("checkpoint validation context is incomplete")
    context_hash = (
        None
        if context_values[0] is None
        else _validation_context_hash(
            protocol_hash=context_values[0],
            data_hash=context_values[1],
            partition_hash=context_values[2],
            callback_hash=context_values[3],
        )
    )
    if payload["validationContextHash"] != context_hash:
        raise ValueError("checkpoint validation context hash does not match its evidence")
    if payload["checkpointModelStateHash"] != checkpoint_model_state_hash:
        raise ValueError("checkpoint fit state does not bind its exact model state")
    if type(stale) is not int or stale < 0:
        raise ValueError("checkpoint stale validation count is invalid")
    for label, step, metric, model_hash, identity in (
        ("best", best_step, best_metric, best_model_hash, payload["bestValidationHash"]),
        ("last", last_step, last_metric, last_model_hash, payload["lastValidationHash"]),
    ):
        if step is None:
            if metric is not None or model_hash is not None or identity is not None:
                raise ValueError(f"checkpoint {label} validation evidence is incomplete")
            continue
        if (
            type(step) is not int
            or step < 1
            or step > optimizer_step
            or (step % config.validation_interval and step != config.max_steps)
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not math.isfinite(float(metric))
            or not isinstance(model_hash, str)
            or _HASH_PATTERN.fullmatch(model_hash) is None
            or context_hash is None
            or identity
            != _validation_identity(
                step,
                float(metric),
                model_state_hash=model_hash,
                context_hash=context_hash,
            )
        ):
            raise ValueError(f"checkpoint {label} validation evidence is invalid")
    if best_step is not None and best_step < config.min_steps:
        raise ValueError("checkpoint best validation predates the minimum-step gate")
    if last_step is None:
        if best_step is not None or stale != 0:
            raise ValueError("checkpoint validation history is internally inconsistent")
    elif last_step == optimizer_step and last_model_hash != checkpoint_model_state_hash:
        raise ValueError("checkpoint last validation does not bind its exact model state")
    if best_step is None:
        if stale != 0:
            raise ValueError("checkpoint cannot be stale before a promotable best exists")
    else:
        if last_step is None or best_step > last_step or float(best_metric) < float(last_metric):
            raise ValueError("checkpoint best validation is not the historical maximum")
        expected_stale = sum(
            1
            for step in range(best_step + 1, last_step + 1)
            if step % config.validation_interval == 0 or step == config.max_steps
        )
        if stale != expected_stale:
            raise ValueError("checkpoint stale validation history is inconsistent")
        if best_step == last_step and (
            best_metric != last_metric or best_model_hash != last_model_hash
        ):
            raise ValueError("checkpoint coincident best and last validation differ")
    best_name = payload["bestCheckpointName"]
    best_sha = payload["bestCheckpointSha256"]
    if (best_name is None) != (best_sha is None):
        raise ValueError("checkpoint best artifact reference is incomplete")
    if best_name is not None and (
        best_step is None
        or not isinstance(best_name, str)
        or Path(best_name).name != best_name
        or ":" in best_name
        or "/" in best_name
        or "\\" in best_name
        or best_name in {"", ".", ".."}
        or not isinstance(best_sha, str)
        or _HASH_PATTERN.fullmatch(best_sha) is None
    ):
        raise ValueError("checkpoint best artifact reference is invalid")
    return _ParsedFitState(
        best_step=best_step,
        best_metric=None if best_metric is None else float(best_metric),
        best_model_state_hash=best_model_hash,
        stale_validations=stale,
        last_validation_step=last_step,
        last_validation_metric=None if last_metric is None else float(last_metric),
        last_model_state_hash=last_model_hash,
        validation_protocol_hash=context_values[0],
        validation_data_hash=context_values[1],
        validation_partition_hash=context_values[2],
        validation_callback_hash=context_values[3],
        best_checkpoint_name=best_name,
        best_checkpoint_sha256=best_sha,
    )


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torchCpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torchCuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torchCpu"])
    if "torchCuda" in state:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torchCuda"])


class _FitTimedOut(RuntimeError):
    pass


def _best_snapshot_path(base: Path, *, run_hash: str, step: int) -> Path:
    return base.with_name(f".{base.name}.run-{run_hash[:16]}.step-{step:010d}.pt")


class CoreTrainer:
    def __init__(
        self,
        model: CoreGFM,
        graphs: Mapping[str, TrainingGraph],
        *,
        config: TrainingConfig,
        seed: int,
    ) -> None:
        if not graphs:
            raise ValueError("training requires at least one domain")
        self.model = model
        self.graphs = dict(sorted(graphs.items()))
        self.config = config
        self.training_seed = int(seed)
        self.device = next(model.parameters()).device
        if any(
            value.features is not None and value.features.device != self.device
            for value in self.graphs.values()
        ):
            raise ValueError("model and graph features must use the same device")
        for value in self.graphs.values():
            if value.adapter is not None:
                value.adapter.to(self.device)
        self._batch_sources: dict[str, NeighborBatchSource] = {}
        batch_counts: dict[str, int] = {}
        for domain_index, (domain, value) in enumerate(self.graphs.items()):
            mode = value.execution_policy.mode(edge_count=value.graph.edge_index.shape[1])
            if mode == "neighbor":
                source_features = (
                    value.features
                    if value.features is not None
                    else torch.arange(value.graph.num_nodes, dtype=torch.float32).view(-1, 1)
                )
                source = NeighborBatchSource(
                    graph=value.graph,
                    features=source_features,
                    policy=value.execution_policy,
                    loader_kind=value.loader_kind,
                    edge_label_index=value.edge_label_index,
                    seed=seed + domain_index * 10_007,
                )
                self._batch_sources[domain] = source
                batch_counts[domain] = source.batch_count
            else:
                batch_counts[domain] = 1
        self.domain_sampler = BalancedDomainSampler(batch_counts, seed=seed)
        generator_device = self.device.type if self.device.type == "cuda" else "cpu"
        self.objective_generator = torch.Generator(device=generator_device).manual_seed(seed + 1)
        parameters = list(self.model.parameters())
        for value in self.graphs.values():
            if value.adapter is not None:
                parameters.extend(value.adapter.parameters())
        self.trainable_parameter_count = sum(
            parameter.numel() for parameter in {id(value): value for value in parameters}.values()
        )
        if self.trainable_parameter_count >= 5_000_000:
            raise ValueError(
                "model, adapters, decoders, and heads must stay below 5,000,000 parameters"
            )
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(getattr(config, "learning_rate", 1e-3)),
            weight_decay=float(getattr(config, "weight_decay", 0.01)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, config.max_steps)
        )
        amp_enabled = self.device.type == "cuda" and config.amp
        self.scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.optimizer_step = 0
        self.gradient_accumulation_cursor = 0
        self.history: list[StepResult] = []
        self.domain_prototypes: dict[str, Tensor] = {}
        self.fit_best_step: int | None = None
        self.fit_best_metric: float | None = None
        self.fit_best_model_state_hash: str | None = None
        self.fit_stale_validations = 0
        self.fit_last_validation_step: int | None = None
        self.fit_last_validation_metric: float | None = None
        self.fit_last_model_state_hash: str | None = None
        self.fit_validation_protocol_hash: str | None = None
        self.fit_validation_data_hash: str | None = None
        self.fit_validation_partition_hash: str | None = None
        self.fit_validation_callback_hash: str | None = None
        self.fit_best_checkpoint_name: str | None = None
        self.fit_best_checkpoint_sha256: str | None = None
        self._fit_state_present = True

    def _sample_negatives(self, graph: PreparedGraph, count: int) -> Tensor:
        return graph.sample_negative_pairs(count, generator=self.objective_generator)

    def _microbatch_loss(
        self, domain: str, batch_index: int, batch_ordinal: int
    ) -> tuple[Tensor, int, str, str]:
        original = self.graphs[domain]
        node_ids: Tensor | None = None
        if domain in self._batch_sources:
            sampled = self._batch_sources[domain].get(
                batch_index=batch_index, ordinal=batch_ordinal
            )
            graph = original.graph
            edge_index = sampled.edge_index.to(self.device)
            pair_inverse = sampled.global_pair_ids.to(self.device)
            pair_representatives = sampled.global_pair_representatives.to(self.device)
            node_ids = sampled.global_node_ids.to(self.device)
            adapter = original.adapter
            features = None if adapter is not None else sampled.features.to(self.device)
            execution_mode = "neighbor"
        else:
            graph = original.graph
            edge_index = graph.edge_index
            pair_inverse = graph.pair_mask_cache.inverse
            pair_representatives = graph.pair_mask_cache.representative_edges
            adapter = original.adapter
            features = original.features
            execution_mode = "full-batch"
        if adapter is not None:
            batch_node_count = node_ids.shape[0] if node_ids is not None else graph.num_nodes
            field_mask = mask_feature_fields(
                (batch_node_count, len(adapter.field_names)),
                generator=self.objective_generator,
                probability=self.config.field_mask_rate,
            ).to(self.device)
            masked_features = adapter(field_mask, node_ids=node_ids)
            reconstruction_mask: Tensor | None = None
        else:
            if features is None:  # pragma: no cover - enforced by TrainingGraph
                raise RuntimeError("training graph features are unavailable")
            field_mask = mask_feature_fields(
                (features.shape[0], features.shape[1]),
                generator=self.objective_generator,
                probability=self.config.field_mask_rate,
            ).to(self.device)
            target_features = features
            masked_features = features.masked_fill(field_mask, 0.0)
            reconstruction_mask = field_mask
        retained, masked_pairs = mask_paired_edges(
            edge_index,
            generator=self.objective_generator,
            probability=self.config.edge_mask_rate,
            pair_count=graph.pair_mask_cache.pair_keys.shape[0],
            pair_inverse=pair_inverse,
            pair_representatives=pair_representatives,
            sampled=node_ids is not None,
        )
        signature = hashlib.sha256()
        signature.update(edge_index.detach().cpu().numpy().tobytes())
        signature.update(field_mask.detach().cpu().numpy().tobytes())
        signature.update(masked_pairs.detach().cpu().numpy().tobytes())
        amp_enabled = self.scaler.is_enabled()
        with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
            encode_domain = getattr(self.model, "encode_domain", None)
            encoded = (
                encode_domain(masked_features, retained, domain)
                if callable(encode_domain)
                else self.model.encode(masked_features, retained)
            )
            decoded = self.model.field_decoder(encoded)
            if reconstruction_mask is None:
                if adapter is None:  # pragma: no cover - branch established above
                    raise RuntimeError("bundle adapter is unavailable")
                decoded_fields = self.model.decode_fields(encoded, retained, field_mask)
                field_loss = adapter.reconstruction_loss(
                    decoded_fields,
                    field_mask,
                    generator=self.objective_generator,
                    node_ids=node_ids,
                )
            else:
                field_loss = functional.mse_loss(
                    decoded[reconstruction_mask], target_features[reconstruction_mask]
                )
            try:
                if node_ids is None:
                    negatives = self._sample_negatives(graph, masked_pairs.shape[0])
                else:
                    negatives = graph.sample_negative_pairs_from_nodes(
                        node_ids,
                        masked_pairs.shape[0],
                        generator=self.objective_generator,
                    )
            except InsufficientNegativeCapacityError:
                negatives = masked_pairs.new_empty((0, 2))
            if masked_pairs.shape[0] and negatives.shape[0] == masked_pairs.shape[0]:
                edge_reconstruction = getattr(
                    self.model, "edge_reconstruction_logits", None
                )
                if callable(edge_reconstruction):
                    positive_logits = edge_reconstruction(
                        encoded, masked_pairs, directed=graph.directed
                    )
                    negative_logits = edge_reconstruction(
                        encoded, negatives, directed=graph.directed
                    )
                else:
                    positive_logits = self.model.binary_link_head(encoded, masked_pairs)
                    negative_logits = self.model.binary_link_head(encoded, negatives)
                edge_loss = functional.binary_cross_entropy_with_logits(
                    torch.cat((positive_logits, negative_logits)),
                    torch.cat(
                        (torch.ones_like(positive_logits), torch.zeros_like(negative_logits))
                    ),
                )
            else:
                edge_loss = encoded.sum() * 0.0
            previous = [
                prototype.to(self.device)
                for previous_domain, prototype in self.domain_prototypes.items()
                if previous_domain != domain
            ]
            if previous:
                alignment_loss = domain_alignment_loss(
                    encoded, torch.stack(previous).mean(dim=0, keepdim=True)
                )
            else:
                alignment_loss = encoded.sum() * 0.0
            loss = combine_objective_losses(
                field_loss=field_loss,
                edge_loss=edge_loss,
                alignment_loss=alignment_loss,
                alignment_weight=self.config.alignment_weight,
            )
        self.domain_prototypes[domain] = encoded.detach().mean(dim=0).cpu()
        return loss, masked_features.shape[0], execution_mode, signature.hexdigest()

    def run_steps(self, count: int) -> list[StepResult]:
        if count < 0 or self.optimizer_step + count > self.config.max_steps:
            raise ValueError("requested steps exceed the configured optimizer-step limit")
        produced: list[StepResult] = []
        self.model.train()
        while len(produced) < count:
            self.optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=self.device)
            last_domain = ""
            last_batch = -1
            last_ordinal = -1
            last_num_nodes = 0
            last_execution_mode = ""
            objective_signature = hashlib.sha256()
            for _microstep in range(self.config.gradient_accumulation):
                last_domain, last_batch = self.domain_sampler.next()
                last_ordinal = self.domain_sampler.last_ordinal(last_domain)
                loss, last_num_nodes, last_execution_mode, micro_signature = self._microbatch_loss(
                    last_domain, last_batch, last_ordinal
                )
                loss = loss / self.config.gradient_accumulation
                objective_signature.update(micro_signature.encode("ascii"))
                self.scaler.scale(loss).backward()
                total_loss = total_loss + loss.detach()
                self.gradient_accumulation_cursor = _microstep + 1
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.gradient_accumulation_cursor = 0
            if self.scaler.is_enabled() and self.scaler.get_scale() < scale_before:
                continue
            self.scheduler.step()
            self.optimizer_step += 1
            result = StepResult(
                optimizer_step=self.optimizer_step,
                domain=last_domain,
                batch_index=last_batch,
                loss=float(total_loss.cpu()),
                alignment_weight=self.config.alignment_weight,
                batch_ordinal=last_ordinal,
                batch_num_nodes=last_num_nodes,
                execution_mode=last_execution_mode,
                objective_signature=objective_signature.hexdigest(),
            )
            self.history.append(result)
            produced.append(result)
        return produced

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizerStep": self.optimizer_step,
            "trainingSeed": self.training_seed,
            "gradientAccumulationCursor": self.gradient_accumulation_cursor,
            "model": self.model.state_dict(),
            "adapters": {
                domain: value.adapter.state_dict()
                for domain, value in self.graphs.items()
                if value.adapter is not None
            },
            "adapterSchemas": {
                domain: value.adapter.schema.model_dump(mode="json", by_alias=True)
                for domain, value in self.graphs.items()
                if value.adapter is not None
            },
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng": _rng_state(),
            "objectiveGenerator": self.objective_generator.get_state(),
            "domainSampler": self.domain_sampler.state_dict(),
            "batchSources": {
                domain: source.state_dict() for domain, source in self._batch_sources.items()
            },
            "domainPrototypes": self.domain_prototypes,
            "config": self.config.to_dict(),
            "fitState": _fit_state_payload(
                best_step=self.fit_best_step,
                best_metric=self.fit_best_metric,
                best_model_state_hash=self.fit_best_model_state_hash,
                stale_validations=self.fit_stale_validations,
                last_validation_step=self.fit_last_validation_step,
                last_validation_metric=self.fit_last_validation_metric,
                last_model_state_hash=self.fit_last_model_state_hash,
                validation_protocol_hash=self.fit_validation_protocol_hash,
                validation_data_hash=self.fit_validation_data_hash,
                validation_partition_hash=self.fit_validation_partition_hash,
                validation_callback_hash=self.fit_validation_callback_hash,
                checkpoint_model_state_hash=_model_state_hash(self.model.state_dict()),
                best_checkpoint_name=self.fit_best_checkpoint_name,
                best_checkpoint_sha256=self.fit_best_checkpoint_sha256,
            ),
        }

    def _apply_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("config") != self.config.to_dict():
            raise ValueError("checkpoint training config does not match trainer config")
        training_seed = state.get("trainingSeed")
        if type(training_seed) is not int or training_seed < 0:
            raise ValueError("checkpoint training seed must be a non-negative integer")
        self.training_seed = training_seed
        self.model.load_state_dict(state["model"])
        expected_adapters = {
            domain for domain, value in self.graphs.items() if value.adapter is not None
        }
        if set(state.get("adapters", {})) != expected_adapters:
            raise ValueError("checkpoint adapter domains do not match trainer domains")
        serialized_schemas = state.get("adapterSchemas")
        if serialized_schemas is not None and set(serialized_schemas) != expected_adapters:
            raise ValueError("checkpoint adapter schema domains do not match trainer domains")
        for domain in expected_adapters:
            adapter = self.graphs[domain].adapter
            if adapter is None:  # pragma: no cover - guarded by expected_adapters
                raise RuntimeError("bundle adapter is unavailable")
            if serialized_schemas is not None:
                checkpoint_schema = AdapterSchema.model_validate_json(
                    canonical_json(serialized_schemas[domain])
                )
                if checkpoint_schema != adapter.schema:
                    raise ValueError(
                        "checkpoint adapter schema does not match trainer adapter schema"
                    )
            learned_state = {
                name: value
                for name, value in state["adapters"][domain].items()
                if not name.startswith("_field_")
            }
            adapter.load_state_dict(learned_state)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.scaler.load_state_dict(state["scaler"])
        self.optimizer_step = int(state["optimizerStep"])
        cursor = int(state["gradientAccumulationCursor"])
        if cursor != 0:
            raise ValueError("checkpoints are only valid at optimizer-step boundaries")
        self.gradient_accumulation_cursor = cursor
        self.objective_generator.set_state(state["objectiveGenerator"])
        if set(state.get("batchSources", {})) != set(self._batch_sources):
            raise ValueError("checkpoint neighbor-source domains do not match trainer domains")
        for domain, source in self._batch_sources.items():
            source.load_state_dict(state["batchSources"][domain])
        self.domain_sampler.load_state_dict(state["domainSampler"])
        self.domain_prototypes = dict(state["domainPrototypes"])
        serialized_fit_state = state.get("fitState")
        parsed_fit_state = _parse_fit_state(
            serialized_fit_state,
            optimizer_step=self.optimizer_step,
            config=self.config,
            checkpoint_model_state_hash=_model_state_hash(state["model"]),
        )
        self.fit_best_step = parsed_fit_state.best_step
        self.fit_best_metric = parsed_fit_state.best_metric
        self.fit_best_model_state_hash = parsed_fit_state.best_model_state_hash
        self.fit_stale_validations = parsed_fit_state.stale_validations
        self.fit_last_validation_step = parsed_fit_state.last_validation_step
        self.fit_last_validation_metric = parsed_fit_state.last_validation_metric
        self.fit_last_model_state_hash = parsed_fit_state.last_model_state_hash
        self.fit_validation_protocol_hash = parsed_fit_state.validation_protocol_hash
        self.fit_validation_data_hash = parsed_fit_state.validation_data_hash
        self.fit_validation_partition_hash = parsed_fit_state.validation_partition_hash
        self.fit_validation_callback_hash = parsed_fit_state.validation_callback_hash
        self.fit_best_checkpoint_name = parsed_fit_state.best_checkpoint_name
        self.fit_best_checkpoint_sha256 = parsed_fit_state.best_checkpoint_sha256
        self._fit_state_present = serialized_fit_state is not None
        _restore_rng_state(state["rng"])

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        live_rng = _rng_state()
        staged_graphs: dict[str, TrainingGraph] = {}
        for domain, value in self.graphs.items():
            staged_graphs[domain] = TrainingGraph(
                graph=value.graph,
                features=value.features,
                adapter=None if value.adapter is None else copy.deepcopy(value.adapter),
                execution_policy=value.execution_policy,
                loader_kind=value.loader_kind,
                edge_label_index=value.edge_label_index,
            )
        staged = CoreTrainer(
            copy.deepcopy(self.model),
            staged_graphs,
            config=self.config,
            seed=0,
        )
        try:
            staged._apply_state_dict(state)
        finally:
            _restore_rng_state(live_rng)
        self._apply_state_dict(state)

    def save_checkpoint(
        self,
        path: Path,
        *,
        bindings: CheckpointBindings,
        status: str = "training",
        promotable: bool = False,
    ) -> None:
        if status not in {"training", "validated", "accepted", "timeout-non-promotable"}:
            raise ValueError("unknown checkpoint status")
        if status in {"validated", "accepted"} or promotable:
            raise ValueError(
                "validated or accepted checkpoints require fit-internal validation evidence"
            )
        publish_checkpoint(
            path,
            trainer_state=self.state_dict(),
            bindings=bindings,
            status=status,  # type: ignore[arg-type]
            promotable=promotable,
        )

    def _publish_fit_checkpoint(
        self,
        path: Path,
        *,
        bindings: CheckpointBindings,
        status: Literal["validated", "timeout-non-promotable"],
        promotable: bool,
        best_step: int | None,
        best_metric: float | None,
        best_model_state_hash: str | None,
        stale_validations: int,
        last_validation_step: int | None,
        last_validation_metric: float | None,
        last_model_state_hash: str | None,
        validation_protocol_hash: str,
        validation_data_hash: str,
        validation_partition_hash: str,
        validation_callback_hash: str,
        best_checkpoint_name: str | None,
        best_checkpoint_sha256: str | None,
        before_commit: Callable[[], None] | None = None,
        replace_existing: bool = True,
    ) -> None:
        if status == "validated":
            if (
                last_validation_step != self.optimizer_step
                or last_validation_metric is None
                or not math.isfinite(last_validation_metric)
                or (promotable and self.optimizer_step < self.config.min_steps)
            ):
                raise ValueError("validated checkpoint lacks current fit validation evidence")
        elif promotable:
            raise ValueError("timeout checkpoint cannot be promotable")
        fit_state = _fit_state_payload(
            best_step=best_step,
            best_metric=best_metric,
            best_model_state_hash=best_model_state_hash,
            stale_validations=stale_validations,
            last_validation_step=last_validation_step,
            last_validation_metric=last_validation_metric,
            last_model_state_hash=last_model_state_hash,
            validation_protocol_hash=validation_protocol_hash,
            validation_data_hash=validation_data_hash,
            validation_partition_hash=validation_partition_hash,
            validation_callback_hash=validation_callback_hash,
            checkpoint_model_state_hash=_model_state_hash(self.model.state_dict()),
            best_checkpoint_name=best_checkpoint_name,
            best_checkpoint_sha256=best_checkpoint_sha256,
        )
        _parse_fit_state(
            fit_state,
            optimizer_step=self.optimizer_step,
            config=self.config,
            checkpoint_model_state_hash=_model_state_hash(self.model.state_dict()),
        )
        trainer_state = self.state_dict()
        trainer_state["fitState"] = fit_state
        publish_checkpoint(
            path,
            trainer_state=trainer_state,
            bindings=bindings,
            status=status,
            promotable=promotable,
            before_commit=before_commit,
            replace_existing=replace_existing,
        )

    def load_checkpoint(self, path: Path, *, bindings: CheckpointBindings) -> None:
        payload = load_checkpoint(path, expected_bindings=bindings)
        self.load_state_dict(payload["trainer"])

    def fit(
        self,
        checkpoint_path: Path | None = None,
        *,
        latest_checkpoint_path: Path | None = None,
        best_checkpoint_path: Path | None = None,
        bindings: CheckpointBindings,
        validate: Callable[[CoreGFM], float],
        validation_contract: ValidationContract,
    ) -> TrainingFitReport:
        ValidationContract.verify(validation_contract)
        callback_hash = _callback_identity(
            validate, callback_artifact_hash=validation_contract.callback_artifact_hash
        )
        requested_context = (
            validation_contract.protocol_hash,
            validation_contract.data_hash,
            validation_contract.partition_hash,
            callback_hash,
        )
        persisted_context = (
            self.fit_validation_protocol_hash,
            self.fit_validation_data_hash,
            self.fit_validation_partition_hash,
            self.fit_validation_callback_hash,
        )
        if all(value is None for value in persisted_context):
            if self.optimizer_step > 0 and not self._fit_state_present:
                raise ValueError("resumed fit requires persisted fit validation state")
            (
                self.fit_validation_protocol_hash,
                self.fit_validation_data_hash,
                self.fit_validation_partition_hash,
                self.fit_validation_callback_hash,
            ) = requested_context
        elif persisted_context != requested_context:
            raise ValueError("resumed fit validation contract or callback identity changed")

        if checkpoint_path is not None:
            raise ValueError(
                "legacy single-checkpoint fit is unsupported; use latest and immutable best paths"
            )
        if latest_checkpoint_path is None or best_checkpoint_path is None:
            raise ValueError("fit requires distinct latest and best checkpoint paths")
        latest_path = Path(latest_checkpoint_path)
        best_path = Path(best_checkpoint_path)
        resolved_latest = latest_path.resolve(strict=False)
        resolved_best = best_path.resolve(strict=False)
        same_existing_file = (
            (latest_path.exists() or latest_path.is_symlink())
            and (best_path.exists() or best_path.is_symlink())
            and os.path.samefile(latest_path, best_path)
        )
        if resolved_latest == resolved_best or same_existing_file:
            raise ValueError("latest and best checkpoint paths must be physically distinct")
        run_hash = canonical_sha256(
            {
                "bindings": {
                    "configHash": bindings.config_hash,
                    "dataHash": bindings.data_hash,
                    "codeHash": bindings.code_hash,
                    "environmentHash": bindings.environment_hash,
                },
                "validation": {
                    "protocolHash": validation_contract.protocol_hash,
                    "dataHash": validation_contract.data_hash,
                    "partitionHash": validation_contract.partition_hash,
                    "callbackHash": callback_hash,
                },
            }
        )
        if self.optimizer_step == 0:
            archive_prefix = f".{best_path.name}.run-{run_hash[:16]}.step-"
            archive_exists = best_path.parent.exists() and any(
                child.name.startswith(archive_prefix) for child in best_path.parent.iterdir()
            )
            if (
                latest_path.exists()
                or latest_path.is_symlink()
                or best_path.exists()
                or best_path.is_symlink()
                or archive_exists
            ):
                raise FileExistsError("fresh fit checkpoint destination already exists")

        def result(
            status: Literal["complete", "early-stopped", "timeout-non-promotable"],
            *,
            latest_step: int,
            best_step: int | None,
            best_metric: float | None,
        ) -> TrainingFitReport:
            return TrainingFitReport(
                status=status,
                latest_step=latest_step,
                best_step=best_step,
                best_metric=best_metric,
                latest_checkpoint_path=latest_path,
                best_checkpoint_path=(
                    None
                    if (best_step is None or self.fit_best_checkpoint_name is None)
                    else best_path.parent / self.fit_best_checkpoint_name
                ),
            )

        started = time.monotonic()
        best_step = self.fit_best_step
        best_metric = self.fit_best_metric
        best_model_state_hash = self.fit_best_model_state_hash
        stale_validations = self.fit_stale_validations
        last_validation_step = self.fit_last_validation_step
        last_validation_metric = self.fit_last_validation_metric
        last_model_state_hash = self.fit_last_model_state_hash
        best_checkpoint_name = self.fit_best_checkpoint_name
        best_checkpoint_sha256 = self.fit_best_checkpoint_sha256
        latest_step = self.optimizer_step

        def timed_out() -> bool:
            return time.monotonic() - started >= self.config.timeout_seconds

        def timeout_result() -> TrainingFitReport:
            self._publish_fit_checkpoint(
                latest_path,
                bindings=bindings,
                status="timeout-non-promotable",
                promotable=False,
                best_step=best_step,
                best_metric=best_metric,
                best_model_state_hash=best_model_state_hash,
                stale_validations=stale_validations,
                last_validation_step=last_validation_step,
                last_validation_metric=last_validation_metric,
                last_model_state_hash=last_model_state_hash,
                validation_protocol_hash=validation_contract.protocol_hash,
                validation_data_hash=validation_contract.data_hash,
                validation_partition_hash=validation_contract.partition_hash,
                validation_callback_hash=callback_hash,
                best_checkpoint_name=best_checkpoint_name,
                best_checkpoint_sha256=best_checkpoint_sha256,
            )
            return result(
                "timeout-non-promotable",
                latest_step=self.optimizer_step,
                best_step=best_step,
                best_metric=best_metric,
            )

        if self.optimizer_step > 0:
            if not self._fit_state_present:
                raise ValueError("resumed fit requires persisted fit validation state")
            if best_step is not None:
                if best_checkpoint_name is None or best_checkpoint_sha256 is None:
                    raise ValueError("resumed fit requires an exact best checkpoint reference")
                observed_best_path = best_path.parent / best_checkpoint_name
                if not observed_best_path.exists() or observed_best_path.is_symlink():
                    raise ValueError("resumed fit requires its existing best checkpoint")
                observed_best_bytes = observed_best_path.read_bytes()
                if hashlib.sha256(observed_best_bytes).hexdigest() != best_checkpoint_sha256:
                    raise ValueError("existing best checkpoint bytes do not match latest")
                previous = load_checkpoint(observed_best_bytes, expected_bindings=bindings)
                previous_step = int(previous["trainer"].get("optimizerStep", -1))
                previous_fit = _parse_fit_state(
                    previous["trainer"].get("fitState"),
                    optimizer_step=previous_step,
                    config=self.config,
                    checkpoint_model_state_hash=_model_state_hash(previous["trainer"]["model"]),
                )
                if (
                    previous.get("status") != "validated"
                    or previous.get("promotable") is not False
                    or previous_step != best_step
                    or previous_step > self.optimizer_step
                    or previous_fit.best_step != best_step
                    or previous_fit.best_metric != best_metric
                    or previous_fit.best_model_state_hash != best_model_state_hash
                    or previous_fit.last_validation_step != best_step
                    or previous_fit.last_validation_metric != best_metric
                    or previous_fit.last_model_state_hash != best_model_state_hash
                    or previous_fit.validation_protocol_hash != validation_contract.protocol_hash
                    or previous_fit.validation_data_hash != validation_contract.data_hash
                    or previous_fit.validation_partition_hash != validation_contract.partition_hash
                    or previous_fit.validation_callback_hash != callback_hash
                ):
                    raise ValueError("existing best checkpoint is not resumable")
                isolated_callback = copy.deepcopy(validate)
                if (
                    _callback_identity(
                        isolated_callback,
                        callback_artifact_hash=validation_contract.callback_artifact_hash,
                    )
                    != callback_hash
                ):
                    raise ValueError("validation callback cannot be isolated reproducibly")
                isolated_model = copy.deepcopy(self.model)
                isolated_model.load_state_dict(previous["trainer"]["model"])
                live_rng = _rng_state()
                try:
                    observed_best_metric = float(isolated_callback(isolated_model))
                finally:
                    _restore_rng_state(live_rng)
                if (
                    not math.isfinite(observed_best_metric)
                    or observed_best_metric != best_metric
                    or _callback_identity(
                        isolated_callback,
                        callback_artifact_hash=validation_contract.callback_artifact_hash,
                    )
                    != callback_hash
                ):
                    raise ValueError(
                        "existing best metric is not reproduced by the bound validation evaluator"
                    )
            elif (
                best_checkpoint_name is not None
                or best_checkpoint_sha256 is not None
                or best_path.exists()
                or best_path.is_symlink()
            ):
                raise ValueError("checkpoint fit state does not bind existing best file")
        while self.optimizer_step < self.config.max_steps:
            if timed_out():
                return timeout_result()
            self.run_steps(1)
            if timed_out():
                return timeout_result()
            if (
                self.optimizer_step % self.config.validation_interval
                and self.optimizer_step != self.config.max_steps
            ):
                continue
            callback_identity_before = _callback_identity(
                validate, callback_artifact_hash=validation_contract.callback_artifact_hash
            )
            metric = float(validate(self.model))
            callback_identity_after = _callback_identity(
                validate, callback_artifact_hash=validation_contract.callback_artifact_hash
            )
            if (
                callback_identity_before != callback_hash
                or callback_identity_after != callback_hash
            ):
                raise ValueError("validation callback state changed during evaluation")
            if not math.isfinite(metric):
                raise ValueError("validation metric must be finite")
            if timed_out():
                return timeout_result()
            promotable = self.optimizer_step >= self.config.min_steps
            candidate_best_step = best_step
            candidate_best_metric = best_metric
            candidate_best_model_hash = best_model_state_hash
            candidate_stale = stale_validations
            candidate_best_name = best_checkpoint_name
            candidate_best_sha = best_checkpoint_sha256
            current_model_state_hash = _model_state_hash(self.model.state_dict())
            if promotable and (best_metric is None or metric > best_metric):
                candidate_best_step = self.optimizer_step
                candidate_best_metric = metric
                candidate_best_model_hash = current_model_state_hash
                candidate_stale = 0
                if best_path is not None:
                    snapshot_path = _best_snapshot_path(
                        best_path, run_hash=run_hash, step=self.optimizer_step
                    )
                    if snapshot_path.is_symlink():
                        raise ValueError("immutable best checkpoint snapshot cannot be a symlink")
                    if snapshot_path.exists():
                        snapshot_bytes = snapshot_path.read_bytes()
                        orphan = load_checkpoint(snapshot_bytes, expected_bindings=bindings)
                        orphan_step = int(orphan["trainer"].get("optimizerStep", -1))
                        orphan_fit = _parse_fit_state(
                            orphan["trainer"].get("fitState"),
                            optimizer_step=orphan_step,
                            config=self.config,
                            checkpoint_model_state_hash=_model_state_hash(
                                orphan["trainer"]["model"]
                            ),
                        )
                        if (
                            orphan.get("status") != "validated"
                            or orphan.get("promotable") is not False
                            or orphan_step != candidate_best_step
                            or orphan_fit.best_step != candidate_best_step
                            or orphan_fit.best_metric != candidate_best_metric
                            or orphan_fit.best_model_state_hash != candidate_best_model_hash
                            or orphan_fit.last_validation_step != self.optimizer_step
                            or orphan_fit.last_validation_metric != metric
                            or orphan_fit.last_model_state_hash != current_model_state_hash
                            or orphan_fit.validation_protocol_hash
                            != validation_contract.protocol_hash
                            or orphan_fit.validation_data_hash != validation_contract.data_hash
                            or orphan_fit.validation_partition_hash
                            != validation_contract.partition_hash
                            or orphan_fit.validation_callback_hash != callback_hash
                            or orphan_fit.best_checkpoint_name is not None
                            or orphan_fit.best_checkpoint_sha256 is not None
                        ):
                            raise FileExistsError(
                                "immutable best checkpoint snapshot conflicts with retry"
                            )
                    else:
                        self._publish_fit_checkpoint(
                            snapshot_path,
                            bindings=bindings,
                            status="validated",
                            promotable=False,
                            best_step=candidate_best_step,
                            best_metric=candidate_best_metric,
                            best_model_state_hash=candidate_best_model_hash,
                            stale_validations=candidate_stale,
                            last_validation_step=self.optimizer_step,
                            last_validation_metric=metric,
                            last_model_state_hash=current_model_state_hash,
                            validation_protocol_hash=validation_contract.protocol_hash,
                            validation_data_hash=validation_contract.data_hash,
                            validation_partition_hash=validation_contract.partition_hash,
                            validation_callback_hash=callback_hash,
                            best_checkpoint_name=None,
                            best_checkpoint_sha256=None,
                            replace_existing=False,
                        )
                        snapshot_bytes = snapshot_path.read_bytes()
                    candidate_best_name = snapshot_path.name
                    candidate_best_sha = hashlib.sha256(snapshot_bytes).hexdigest()
                    if timed_out():
                        return timeout_result()
            elif promotable:
                candidate_stale += 1

            def reject_late_commit() -> None:
                if timed_out():
                    raise _FitTimedOut("training deadline crossed before checkpoint commit")

            try:
                self._publish_fit_checkpoint(
                    latest_path,
                    bindings=bindings,
                    status="validated",
                    promotable=promotable,
                    best_step=candidate_best_step,
                    best_metric=candidate_best_metric,
                    best_model_state_hash=candidate_best_model_hash,
                    stale_validations=candidate_stale,
                    last_validation_step=self.optimizer_step,
                    last_validation_metric=metric,
                    last_model_state_hash=current_model_state_hash,
                    validation_protocol_hash=validation_contract.protocol_hash,
                    validation_data_hash=validation_contract.data_hash,
                    validation_partition_hash=validation_contract.partition_hash,
                    validation_callback_hash=callback_hash,
                    best_checkpoint_name=candidate_best_name,
                    best_checkpoint_sha256=candidate_best_sha,
                    before_commit=reject_late_commit,
                )
            except _FitTimedOut:
                return timeout_result()
            if timed_out():
                return timeout_result()
            best_step = candidate_best_step
            best_metric = candidate_best_metric
            best_model_state_hash = candidate_best_model_hash
            stale_validations = candidate_stale
            last_validation_step = self.optimizer_step
            last_validation_metric = metric
            last_model_state_hash = current_model_state_hash
            best_checkpoint_name = candidate_best_name
            best_checkpoint_sha256 = candidate_best_sha
            self.fit_best_step = best_step
            self.fit_best_metric = best_metric
            self.fit_best_model_state_hash = best_model_state_hash
            self.fit_stale_validations = stale_validations
            self.fit_last_validation_step = last_validation_step
            self.fit_last_validation_metric = last_validation_metric
            self.fit_last_model_state_hash = last_model_state_hash
            self.fit_best_checkpoint_name = best_checkpoint_name
            self.fit_best_checkpoint_sha256 = best_checkpoint_sha256
            latest_step = self.optimizer_step
            if stale_validations >= self.config.patience and promotable:
                return result(
                    "early-stopped",
                    latest_step=latest_step,
                    best_step=best_step,
                    best_metric=best_metric,
                )
        return result(
            "complete",
            latest_step=latest_step,
            best_step=best_step,
            best_metric=best_metric,
        )


__all__ = [
    "CoreTrainer",
    "StepResult",
    "TrainingFitReport",
    "TrainingGraph",
    "ValidationContract",
]
