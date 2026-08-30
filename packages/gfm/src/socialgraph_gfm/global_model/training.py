"""Balanced, resumable NeighborLoader training for SocialGraph-FM Global."""

from __future__ import annotations

import hashlib
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_sha256, file_sha256
from socialgraph_gfm.gfm.corpus.common import atomic_write_json

from .config import SEED


@dataclass(frozen=True)
class DomainData:
    """Only data that the trainer is authorized to observe."""

    country: str
    edge_index: Any
    text_features: Any
    structural_features: Any
    labels: Any
    train_mask: Any
    validation_mask: Any
    structure_missing: Any
    graph_stats: Any
    source_hashes: Mapping[str, str]
    train_split_hash: str
    validation_split_hash: str


@dataclass(frozen=True)
class TrainingOptions:
    max_steps: int = 1000
    min_steps: int = 100
    eval_every_steps: int = 25
    patience_evals: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed_batch_size: int = 128
    num_neighbors: tuple[int, ...] = (20, 10)
    memory_smoke_batch_sizes: tuple[int, ...] = (256, 128, 64)
    max_peak_mib: int = 5632
    gradient_clip_norm: float = 1.0
    router_balance_weight: float = 0.01
    amp: bool = True
    checkpoint_every_steps: int = 25
    num_workers: int = 0

    def __post_init__(self) -> None:
        if (
            self.max_steps < 1
            or self.min_steps < 1
            or self.min_steps > self.max_steps
            or self.eval_every_steps < 1
            or self.patience_evals < 1
            or self.seed_batch_size < 2
            or self.seed_batch_size % 2
        ):
            raise ValueError("Global training limits must be positive")
        if not self.num_neighbors or any(value == 0 or value < -1 for value in self.num_neighbors):
            raise ValueError("Global num_neighbors must contain -1 or positive fanouts")
        if self.learning_rate <= 0 or self.weight_decay != 0:
            raise ValueError("SocialGraph-FM Global requires Adam with zero weight decay")
        if self.gradient_clip_norm <= 0 or self.router_balance_weight < 0:
            raise ValueError("Global regularization values are invalid")
        if self.num_workers < 0:
            raise ValueError("Global num_workers cannot be negative")
        if (
            not self.memory_smoke_batch_sizes
            or any(value < 2 or value % 2 for value in self.memory_smoke_batch_sizes)
            or tuple(sorted(self.memory_smoke_batch_sizes, reverse=True))
            != self.memory_smoke_batch_sizes
            or self.max_peak_mib < 1
            or self.checkpoint_every_steps < 1
        ):
            raise ValueError("Global memory/checkpoint protocol is invalid")


@dataclass(frozen=True)
class TrainingOutcome:
    checkpoint_path: Path
    best_checkpoint_path: Path
    checkpoint_sha256: str
    model_state_hash: str
    best_step: int
    steps_completed: int
    global_step: int
    best_validation_macro_f1: float
    validation_threshold: float
    stopped_early: bool
    resumed_from_step: int | None
    amp_enabled: bool
    history: tuple[dict[str, Any], ...]
    memory_smoke: dict[str, Any]


@dataclass(frozen=True)
class InferenceData:
    country: str
    edge_index: Any
    text_features: Any
    structural_features: Any
    labels: Any
    mask: Any
    structure_missing: Any
    graph_stats: Any
    source_hashes: Mapping[str, str]
    split_hash: str


@dataclass(frozen=True)
class CountryOutputs:
    node_ids: Any
    logits: Any
    labels: Any
    structure_missing: Any
    router_indices: Any
    router_weights: Any
    modality_evidence: Any
    expert_names: tuple[str, ...]


def options_from_config(config: Mapping[str, Any], *, fast: bool = False) -> TrainingOptions:
    training = config["training"]
    options = TrainingOptions(
        max_steps=int(training["maxSteps"]),
        min_steps=int(training["minSteps"]),
        eval_every_steps=int(training["evalEverySteps"]),
        patience_evals=int(training["patienceEvals"]),
        learning_rate=float(training["learningRate"]),
        weight_decay=float(training["weightDecay"]),
        seed_batch_size=int(training["seedBatchSize"]),
        num_neighbors=tuple(int(value) for value in training["numNeighbors"]),
        memory_smoke_batch_sizes=tuple(
            int(value) for value in training["memorySmokeBatchSizes"]
        ),
        max_peak_mib=int(training["maxPeakMiB"]),
        gradient_clip_norm=float(training["gradientClipNorm"]),
        router_balance_weight=float(training["routerBalanceWeight"]),
        amp=bool(training["amp"]),
        checkpoint_every_steps=int(training["checkpointEverySteps"]),
        num_workers=int(training["numWorkers"]),
    )
    if fast:
        return replace(
            options,
            max_steps=min(2, options.max_steps),
            min_steps=1,
            eval_every_steps=1,
            patience_evals=2,
            seed_batch_size=8,
            num_neighbors=tuple(min(4, value) if value > 0 else value for value in options.num_neighbors),
            memory_smoke_batch_sizes=(16, 8, 4),
            checkpoint_every_steps=1,
        )
    return options


def set_training_seed(seed: int = SEED, *, device: str = "cpu") -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader_length(loader: Iterable[Any]) -> int:
    try:
        length = len(loader)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("balanced loaders must expose a finite length") from exc
    if length < 1:
        raise ValueError("balanced loaders must not be empty")
    return length


def country_balanced_batches(
    loaders: Mapping[str, Iterable[Any]],
    *,
    max_steps: int | None = None,
) -> Iterator[tuple[str, Any]]:
    """Cycle every domain to the largest loader length and yield strict round-robin batches."""

    if not loaders:
        raise ValueError("at least one country loader is required")
    countries = tuple(loaders)
    rounds = max(_loader_length(loader) for loader in loaders.values())
    if max_steps is not None:
        rounds = min(rounds, max_steps)
    iterators = {country: iter(loader) for country, loader in loaders.items()}
    for _round in range(rounds):
        for country in countries:
            try:
                batch = next(iterators[country])
            except StopIteration:
                iterators[country] = iter(loaders[country])
                try:
                    batch = next(iterators[country])
                except StopIteration as exc:
                    raise ValueError(f"country loader {country!r} became empty") from exc
            yield country, batch


def _to_tensor(value: Any, *, dtype: Any = None):
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        result = value.detach().cpu()
    else:
        array = np.asarray(value)
        if not array.flags.writeable:
            array = array.copy()
        result = torch.from_numpy(array)
    return result.to(dtype=dtype) if dtype is not None else result


def _pyg_data(domain: DomainData):
    import torch
    from torch_geometric.data import Data

    edge_index = _to_tensor(domain.edge_index, dtype=torch.long)
    text_features = _to_tensor(domain.text_features, dtype=torch.float32)
    structural_features = _to_tensor(domain.structural_features, dtype=torch.long)
    labels = _to_tensor(domain.labels, dtype=torch.float32)
    train_mask = _to_tensor(domain.train_mask, dtype=torch.bool)
    validation_mask = _to_tensor(domain.validation_mask, dtype=torch.bool)
    structure_missing = _to_tensor(domain.structure_missing, dtype=torch.bool)
    graph_stats = _to_tensor(domain.graph_stats, dtype=torch.float32)
    node_count = int(labels.numel())
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise ValueError(f"{domain.country} edge_index must have shape [2, E]")
    if text_features.ndim != 2 or text_features.shape[0] != node_count:
        raise ValueError(f"{domain.country} text feature count does not match labels")
    if structural_features.shape[0] != node_count:
        raise ValueError(f"{domain.country} structural feature count does not match labels")
    if (
        train_mask.shape != labels.shape
        or validation_mask.shape != labels.shape
        or structure_missing.shape != labels.shape
    ):
        raise ValueError(f"{domain.country} masks do not match labels")
    if graph_stats.shape != (13,) or not bool(torch.isfinite(graph_stats).all()):
        raise ValueError(f"{domain.country} graph statistics are invalid")
    if not bool(train_mask.any()) or not bool(validation_mask.any()):
        raise ValueError(f"{domain.country} train and validation masks must be nonempty")
    if bool(torch.logical_and(train_mask, validation_mask).any()):
        raise ValueError(f"{domain.country} train and validation masks overlap")
    return Data(
        edge_index=edge_index.contiguous(),
        text_features=text_features.contiguous(),
        structural_features=structural_features.contiguous(),
        y=labels.contiguous(),
        train_mask=train_mask.contiguous(),
        validation_mask=validation_mask.contiguous(),
        structure_missing=structure_missing.contiguous(),
        num_nodes=node_count,
    )


def _pyg_inference_data(domain: InferenceData):
    import torch
    from torch_geometric.data import Data

    edge_index = _to_tensor(domain.edge_index, dtype=torch.long)
    text_features = _to_tensor(domain.text_features, dtype=torch.float32)
    structural_features = _to_tensor(domain.structural_features, dtype=torch.long)
    labels = _to_tensor(domain.labels, dtype=torch.float32)
    mask = _to_tensor(domain.mask, dtype=torch.bool)
    structure_missing = _to_tensor(domain.structure_missing, dtype=torch.bool)
    graph_stats = _to_tensor(domain.graph_stats, dtype=torch.float32)
    node_count = int(labels.numel())
    if edge_index.ndim != 2 or tuple(edge_index.shape[:1]) != (2,):
        raise ValueError(f"{domain.country} edge_index must have shape [2, E]")
    if text_features.ndim != 2 or text_features.shape[0] != node_count:
        raise ValueError(f"{domain.country} text feature count does not match labels")
    if (
        structural_features.shape[0] != node_count
        or mask.shape != labels.shape
        or structure_missing.shape != labels.shape
    ):
        raise ValueError(f"{domain.country} inference features or mask do not match labels")
    if graph_stats.shape != (13,) or not bool(torch.isfinite(graph_stats).all()):
        raise ValueError(f"{domain.country} graph statistics are invalid")
    if not bool(mask.any()):
        raise ValueError(f"{domain.country} inference mask must be nonempty")
    return Data(
        edge_index=edge_index.contiguous(),
        text_features=text_features.contiguous(),
        structural_features=structural_features.contiguous(),
        y=labels.contiguous(),
        selected_mask=mask.contiguous(),
        structure_missing=structure_missing.contiguous(),
        num_nodes=node_count,
    )


def _neighbor_loader(
    data: Any,
    *,
    input_nodes: Any,
    options: TrainingOptions,
    batch_size: int,
    shuffle: bool,
    seed: int | None = None,
    generator: Any = None,
):
    import torch
    from torch_geometric.loader import NeighborLoader

    if generator is None:
        if seed is None:
            raise ValueError("NeighborLoader requires a seed or generator")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=list(options.num_neighbors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=options.num_workers,
        persistent_workers=options.num_workers > 0,
        generator=generator,
    )


def _domain_seed(country: str, step: int, *, phase: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{phase}:{country}:{step}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _clone_state(model: Any) -> dict[str, Any]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"model state {name!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, payload: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _macro_f1(labels: Any, predictions: Any) -> float:
    import torch

    scores: list[float] = []
    for target in (0, 1):
        truth = labels == target
        predicted = predictions == target
        true_positive = int(torch.logical_and(truth, predicted).sum())
        false_positive = int(torch.logical_and(~truth, predicted).sum())
        false_negative = int(torch.logical_and(truth, ~predicted).sum())
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else (2.0 * true_positive / denominator))
    return sum(scores) / 2.0


def country_balanced_threshold(
    logits: Mapping[str, Any], labels: Mapping[str, Any]
) -> tuple[float, float, dict[str, float]]:
    import torch

    if set(logits) != set(labels) or not logits:
        raise ValueError("threshold selection requires matching nonempty country inventories")
    probabilities = {country: torch.sigmoid(values.detach().cpu().float()) for country, values in logits.items()}
    targets = {country: labels[country].detach().cpu().long() for country in logits}
    best_threshold = 0.5
    best_score = -1.0
    best_per_country: dict[str, float] = {}
    for step in range(5, 96):
        threshold = step / 100.0
        per_country = {
            country: _macro_f1(targets[country], probabilities[country] >= threshold)
            for country in probabilities
        }
        score = sum(per_country.values()) / len(per_country)
        if score > best_score + 1e-12 or (
            math.isclose(score, best_score, abs_tol=1e-12)
            and abs(threshold - 0.5) < abs(best_threshold - 0.5)
        ):
            best_threshold = threshold
            best_score = score
            best_per_country = per_country
    return best_threshold, best_score, best_per_country


def _autocast(device: Any, enabled: bool):
    import torch

    return torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=enabled and device.type == "cuda",
    )


def _forward(
    model: Any,
    batch: Any,
    domain_id: str | None,
    graph_stats: Any,
    allowed_experts: tuple[str, ...],
    device: Any,
    *,
    amp: bool,
):
    text = batch.text_features.to(device, non_blocking=True)
    structural = batch.structural_features.to(device, non_blocking=True)
    edge_index = batch.edge_index.to(device, non_blocking=True)
    statistics = _to_tensor(graph_stats, dtype=text.dtype).to(device, non_blocking=True)
    with _autocast(device, amp):
        return model(
            text,
            structural,
            edge_index,
            graph_stats=statistics,
            domain_id=domain_id,
            allowed_experts=allowed_experts,
        )


def collect_split_logits(
    model: Any,
    domains: Mapping[str, DomainData],
    *,
    split: str,
    options: TrainingOptions,
    device: Any,
    allowed_experts: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise ValueError("trainer may collect train or validation logits only")
    inference = {
        country: InferenceData(
            country=country,
            edge_index=domain.edge_index,
            text_features=domain.text_features,
            structural_features=domain.structural_features,
            labels=domain.labels,
            mask=domain.train_mask if split == "train" else domain.validation_mask,
            structure_missing=domain.structure_missing,
            graph_stats=domain.graph_stats,
            source_hashes=domain.source_hashes,
            split_hash=(
                domain.train_split_hash if split == "train" else domain.validation_split_hash
            ),
        )
        for country, domain in domains.items()
    }
    outputs = collect_masked_outputs(
        model,
        inference,
        options=options,
        device=device,
        phase=split,
        allowed_experts=allowed_experts,
    )
    return (
        {country: output.logits for country, output in outputs.items()},
        {country: output.labels for country, output in outputs.items()},
    )


def collect_masked_outputs(
    model: Any,
    domains: Mapping[str, InferenceData],
    *,
    options: TrainingOptions,
    device: Any,
    phase: str,
    allowed_experts: tuple[str, ...],
) -> dict[str, CountryOutputs]:
    import torch

    if not domains:
        raise ValueError("Global inference requires at least one authorized domain")
    selected_device = torch.device(device)
    model.eval()
    collected: dict[str, CountryOutputs] = {}
    with torch.inference_mode():
        for country, domain in domains.items():
            data = _pyg_inference_data(domain)
            loader = _neighbor_loader(
                data,
                input_nodes=data.selected_mask,
                options=options,
                batch_size=options.seed_batch_size,
                shuffle=False,
                seed=_domain_seed(country, 0, phase=phase),
            )
            rows: list[dict[str, Any]] = []
            expert_names: tuple[str, ...] | None = None
            for batch in loader:
                router_domain = (
                    country if f"domain:{country}" in allowed_experts else None
                )
                output = _forward(
                    model,
                    batch,
                    router_domain,
                    domain.graph_stats,
                    allowed_experts,
                    selected_device,
                    amp=options.amp,
                )
                count = int(batch.batch_size)
                if output.router_indices is None or output.router_weights is None:
                    raise ValueError("SocialGraph-FM Global inference requires the fixed top-2 router")
                indices = output.router_indices[:count].detach().long().cpu()
                weights = output.router_weights[:count].detach().float().cpu()
                if weights.ndim != 2 or indices.ndim != 2:
                    raise ValueError("Global router evidence must be rank-2")
                if weights.shape[1] != indices.shape[1]:
                    weights = torch.gather(weights, 1, indices)
                current_names = tuple(str(item) for item in output.expert_names)
                if expert_names is None:
                    expert_names = current_names
                elif expert_names != current_names:
                    raise ValueError("Global expert inventory changed between batches")
                rows.append(
                    {
                        "nodeIds": batch.n_id[:count].detach().long().cpu(),
                        "logits": output.logits[:count].detach().float().cpu(),
                        "labels": batch.y[:count].detach().long().cpu(),
                        "structureMissing": batch.structure_missing[:count]
                        .detach()
                        .bool()
                        .cpu(),
                        "routerIndices": indices,
                        "routerWeights": weights,
                        "modalityEvidence": output.modality_contributions[:count]
                        .detach()
                        .float()
                        .cpu(),
                    }
                )
            if not rows or expert_names is None:
                raise ValueError(f"{country} produced no Global inference rows")
            node_ids = torch.cat([row["nodeIds"] for row in rows])
            ordering = torch.argsort(node_ids)
            collected[country] = CountryOutputs(
                node_ids=node_ids[ordering],
                logits=torch.cat([row["logits"] for row in rows])[ordering],
                labels=torch.cat([row["labels"] for row in rows])[ordering],
                structure_missing=torch.cat(
                    [row["structureMissing"] for row in rows]
                )[ordering],
                router_indices=torch.cat([row["routerIndices"] for row in rows])[ordering],
                router_weights=torch.cat([row["routerWeights"] for row in rows])[ordering],
                modality_evidence=torch.cat(
                    [row["modalityEvidence"] for row in rows]
                )[ordering],
                expert_names=expert_names,
            )
    return collected


def _training_evidence(domains: Mapping[str, DomainData]) -> dict[str, Any]:
    return {
        country: {
            "sourceHashes": dict(sorted(domain.source_hashes.items())),
            "trainSplitHash": domain.train_split_hash,
            "validationSplitHash": domain.validation_split_hash,
        }
        for country, domain in domains.items()
    }


class BalancedSeedSampler:
    """Deterministic cyclic sampler with an exact 50/50 binary-class seed batch."""

    def __init__(self, labels: Any, mask: Any, *, seed: int) -> None:
        import torch

        values = _to_tensor(labels, dtype=torch.long)
        selected = _to_tensor(mask, dtype=torch.bool)
        self.negative = torch.nonzero(selected & (values == 0), as_tuple=False).reshape(-1)
        self.positive = torch.nonzero(selected & (values == 1), as_tuple=False).reshape(-1)
        if self.negative.numel() == 0 or self.positive.numel() == 0:
            raise ValueError("class-balanced Global sampling requires both train labels")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.negative_order = self.negative[
            torch.randperm(self.negative.numel(), generator=self.generator)
        ]
        self.positive_order = self.positive[
            torch.randperm(self.positive.numel(), generator=self.generator)
        ]
        self.negative_cursor = 0
        self.positive_cursor = 0

    def _draw(self, label: int, count: int):
        import torch

        source = self.positive if label else self.negative
        order_name = "positive_order" if label else "negative_order"
        cursor_name = "positive_cursor" if label else "negative_cursor"
        order = getattr(self, order_name)
        cursor = int(getattr(self, cursor_name))
        pieces = []
        remaining = count
        while remaining:
            available = int(order.numel()) - cursor
            take = min(remaining, available)
            pieces.append(order[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor == int(order.numel()):
                order = source[torch.randperm(source.numel(), generator=self.generator)]
                cursor = 0
        setattr(self, order_name, order)
        setattr(self, cursor_name, cursor)
        return torch.cat(pieces)

    def draw(self, batch_size: int):
        import torch

        if batch_size < 2 or batch_size % 2:
            raise ValueError("class-balanced seed batch size must be a positive even number")
        half = batch_size // 2
        values = torch.cat((self._draw(0, half), self._draw(1, half)))
        return values[torch.randperm(values.numel(), generator=self.generator)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "negative": self.negative.clone(),
            "positive": self.positive.clone(),
            "negativeOrder": self.negative_order.clone(),
            "positiveOrder": self.positive_order.clone(),
            "negativeCursor": self.negative_cursor,
            "positiveCursor": self.positive_cursor,
            "generatorState": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        import torch

        if not torch.equal(state["negative"], self.negative) or not torch.equal(
            state["positive"], self.positive
        ):
            raise ValueError("Global sampler class inventory changed")
        self.negative_order = state["negativeOrder"].clone()
        self.positive_order = state["positiveOrder"].clone()
        self.negative_cursor = int(state["negativeCursor"])
        self.positive_cursor = int(state["positiveCursor"])
        self.generator.set_state(state["generatorState"])


def _numpy_rng_state() -> dict[str, Any]:
    import numpy as np

    name, keys, position, has_gauss, cached = np.random.get_state()
    keys_array = np.asarray(keys, dtype=np.uint32)
    return {
        "name": name,
        "keys": keys_array.tolist(),
        "position": int(position),
        "hasGauss": int(has_gauss),
        "cachedGaussian": float(cached),
    }


def _set_numpy_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np

    np.random.set_state(
        (
            str(state["name"]),
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["position"]),
            int(state["hasGauss"]),
            float(state["cachedGaussian"]),
        )
    )


def _rng_state(
    samplers: Mapping[str, BalancedSeedSampler],
    neighbor_generators: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    return {
        "cpuRngState": torch.get_rng_state(),
        "cudaRngState": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "pythonRngState": random.getstate(),
        "numpyRngState": _numpy_rng_state(),
        "samplerStates": {
            country: sampler.state_dict() for country, sampler in samplers.items()
        },
        "neighborGeneratorStates": {
            country: generator.get_state()
            for country, generator in neighbor_generators.items()
        },
    }


def _restore_rng_state(
    state: Mapping[str, Any],
    samplers: Mapping[str, BalancedSeedSampler],
    neighbor_generators: Mapping[str, Any],
) -> None:
    import torch

    torch.set_rng_state(state["cpuRngState"])
    if torch.cuda.is_available() and state.get("cudaRngState") is not None:
        torch.cuda.set_rng_state_all(state["cudaRngState"])
    random.setstate(state["pythonRngState"])
    _set_numpy_rng_state(state["numpyRngState"])
    for country, sampler in samplers.items():
        sampler.load_state_dict(state["samplerStates"][country])
        neighbor_generators[country].set_state(
            state["neighborGeneratorStates"][country]
        )


def _domain_update_backward(
    model: Any,
    data_by_country: Mapping[str, Any],
    domains: Mapping[str, DomainData],
    samplers: Mapping[str, BalancedSeedSampler],
    neighbor_generators: Mapping[str, Any],
    *,
    batch_size: int,
    allowed_experts: tuple[str, ...],
    options: TrainingOptions,
    device: Any,
    scaler: Any,
) -> dict[str, float]:
    import torch

    from .model import router_load_balancing_loss

    domain_count = len(domains)
    losses: dict[str, float] = {}
    for country in domains:
        seeds = samplers[country].draw(batch_size)
        loader = _neighbor_loader(
            data_by_country[country],
            input_nodes=seeds,
            options=options,
            batch_size=batch_size,
            shuffle=False,
            generator=neighbor_generators[country],
        )
        batch = next(iter(loader))
        if int(batch.batch_size) != batch_size:
            raise RuntimeError("NeighborLoader did not preserve the balanced seed batch")
        output = _forward(
            model,
            batch,
            country,
            domains[country].graph_stats,
            allowed_experts,
            device,
            amp=options.amp,
        )
        labels = batch.y[:batch_size].to(device, non_blocking=True)
        if int((labels == 0).sum()) != batch_size // 2 or int(
            (labels == 1).sum()
        ) != batch_size // 2:
            raise RuntimeError("Global seed batch lost exact class balance")
        with _autocast(device, options.amp):
            classification = torch.nn.functional.binary_cross_entropy_with_logits(
                output.logits[:batch_size], labels
            )
            if output.router_weights is None or output.router_indices is None:
                raise ValueError("SocialGraph-FM Global training requires the fixed top-2 router")
            router = router_load_balancing_loss(
                output.router_weights[:batch_size],
                output.router_indices[:batch_size],
                expert_count=len(output.expert_names),
            )
            loss = classification + options.router_balance_weight * router
        scaler.scale(loss / domain_count).backward()
        losses[country] = float(loss.detach().cpu())
    return losses


def _memory_smoke(
    model: Any,
    optimizer: Any,
    scaler: Any,
    data_by_country: Mapping[str, Any],
    domains: Mapping[str, DomainData],
    samplers: Mapping[str, BalancedSeedSampler],
    neighbor_generators: Mapping[str, Any],
    *,
    allowed_experts: tuple[str, ...],
    options: TrainingOptions,
    device: Any,
) -> tuple[int, dict[str, Any]]:
    import torch

    baseline = _rng_state(samplers, neighbor_generators)
    candidates = (
        options.memory_smoke_batch_sizes
        if device.type == "cuda"
        else (options.seed_batch_size,)
    )
    attempts = []
    selected: int | None = None
    selected_peak: float | None = None
    for candidate in candidates:
        _restore_rng_state(baseline, samplers, neighbor_generators)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        passed = False
        failure: str | None = None
        try:
            model.train()
            _domain_update_backward(
                model,
                data_by_country,
                domains,
                samplers,
                neighbor_generators,
                batch_size=candidate,
                allowed_experts=allowed_experts,
                options=options,
                device=device,
                scaler=scaler,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = torch.cuda.max_memory_allocated(device) / (1024**2)
                passed = peak <= options.max_peak_mib
                if not passed:
                    failure = "peak-memory-limit"
            else:
                peak = None
                passed = True
        except torch.OutOfMemoryError:
            peak = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            )
            failure = "cuda-out-of-memory"
        elapsed = max(time.perf_counter() - started, 1e-9)
        throughput = candidate * len(domains) / elapsed
        attempts.append(
            {
                "batchSize": candidate,
                "passed": passed,
                "peakMiB": peak,
                "throughputSeedsPerSecond": throughput,
                "failure": failure,
            }
        )
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if passed:
            selected = min(options.seed_batch_size, candidate)
            selected_peak = peak
            break
    _restore_rng_state(baseline, samplers, neighbor_generators)
    optimizer.zero_grad(set_to_none=True)
    if selected is None:
        raise torch.OutOfMemoryError("no Global memory-smoke batch fits the release budget")
    smoke_steps = 2 if options.max_steps <= 2 else 50
    smoke_started = time.perf_counter()
    _restore_rng_state(baseline, samplers, neighbor_generators)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    try:
        for _ in range(smoke_steps):
            optimizer.zero_grad(set_to_none=True)
            _domain_update_backward(
                model,
                data_by_country,
                domains,
                samplers,
                neighbor_generators,
                batch_size=selected,
                allowed_experts=allowed_experts,
                options=options,
                device=device,
                scaler=scaler,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            selected_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            if selected_peak is None or selected_peak > options.max_peak_mib:
                raise torch.OutOfMemoryError(
                    "Global 50-step memory smoke exceeded the release memory budget"
                )
    finally:
        optimizer.zero_grad(set_to_none=True)
        _restore_rng_state(baseline, samplers, neighbor_generators)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    smoke_elapsed = max(time.perf_counter() - smoke_started, 1e-9)
    selected_throughput = smoke_steps * selected * len(domains) / smoke_elapsed
    eta = options.max_steps * len(domains) * selected / selected_throughput
    report = {
        "schemaVersion": "socialgraph-fm.global-model-memory-smoke/1.0",
        "device": str(device),
        "candidateBatchSizes": list(options.memory_smoke_batch_sizes),
        "configuredSeedBatchSize": options.seed_batch_size,
        "selectedSeedBatchSize": selected,
        "maxPeakMiB": options.max_peak_mib,
        "selectedPeakMiB": selected_peak,
        "smokeSteps": smoke_steps,
        "throughputSeedsPerSecond": selected_throughput,
        "estimatedTrainingSeconds": eta,
        "attempts": attempts,
        "oomFallbacks": [],
    }
    report["memorySmokeHash"] = canonical_sha256(report)
    return selected, report


def _restore_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scaler: Any,
    samplers: Mapping[str, BalancedSeedSampler],
    neighbor_generators: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schemaVersion") != "socialgraph-fm.global-model-training-checkpoint/1.0":
        raise ValueError("unsupported Global training checkpoint")
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"Global resume checkpoint {field} mismatch")
    model.load_state_dict(payload["modelState"])
    optimizer.load_state_dict(payload["optimizerState"])
    scaler.load_state_dict(payload["scalerState"])
    _restore_rng_state(payload["rngState"], samplers, neighbor_generators)
    return payload


def train_balanced_neighbor_model(
    model: Any,
    domains: Mapping[str, DomainData],
    *,
    run_dir: Path,
    protocol: str,
    identity: Mapping[str, str],
    options: TrainingOptions,
    allowed_experts: tuple[str, ...],
    device: str,
    resume: bool = True,
    on_step_complete: Callable[[int], None] | None = None,
) -> TrainingOutcome:
    """Run one equal-domain, equal-class update per fixed optimizer step."""

    import torch

    if not domains:
        raise ValueError("Global training requires at least one authorized domain")
    if len(domains) != len(set(domains)) or any(
        key != value.country for key, value in domains.items()
    ):
        raise ValueError("Global training domain keys must be unique and canonical")
    if not allowed_experts or "null" not in allowed_experts:
        raise ValueError("Global training requires an explicit expert allowlist")
    if any(f"domain:{country}" not in allowed_experts for country in domains):
        raise ValueError("Global training allowlist must include each observed domain expert")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    set_training_seed(SEED, device=device)
    model.to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=options.learning_rate)
    amp_enabled = options.amp and selected_device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    options_hash = canonical_sha256(asdict(options))
    evidence = _training_evidence(domains)
    evidence_hash = canonical_sha256(evidence)
    expected = {
        "protocol": protocol,
        "identity": dict(identity),
        "optionsHash": options_hash,
        "trainingEvidenceHash": evidence_hash,
        "allowedExperts": list(allowed_experts),
        "device": str(selected_device),
        "ampEnabled": amp_enabled,
    }
    latest = run_dir / "checkpoint-latest.pt"
    best = run_dir / "checkpoint-best.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    data_by_country = {country: _pyg_data(domain) for country, domain in domains.items()}
    samplers = {
        country: BalancedSeedSampler(
            data.y,
            data.train_mask,
            seed=_domain_seed(country, 0, phase="class-balanced-seeds"),
        )
        for country, data in data_by_country.items()
    }
    neighbor_generators = {}
    for country in domains:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_domain_seed(country, 0, phase="neighbor-sampling"))
        neighbor_generators[country] = generator

    start_step = 1
    resumed_from_step: int | None = None
    stale_evaluations = 0
    best_step = 0
    best_score = -1.0
    best_threshold = 0.5
    history: list[dict[str, Any]] = []
    best_state = _clone_state(model)
    stopped_early = False
    memory_report: dict[str, Any]
    selected_batch_size: int
    if latest.exists():
        if not resume:
            raise FileExistsError(f"Global checkpoint already exists: {latest}")
        restored = _restore_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            samplers=samplers,
            neighbor_generators=neighbor_generators,
            expected=expected,
        )
        resumed_from_step = int(restored["stepCompleted"])
        start_step = resumed_from_step + 1
        stale_evaluations = int(restored["staleEvaluations"])
        best_step = int(restored["bestStep"])
        best_score = float(restored["bestValidationMacroF1"])
        best_threshold = float(restored["validationThreshold"])
        history = list(restored["history"])
        best_state = restored["bestModelState"]
        stopped_early = bool(restored["stoppedEarly"])
        memory_report = dict(restored["memorySmoke"])
        selected_batch_size = int(memory_report["selectedSeedBatchSize"])
    else:
        existing = [path for path in run_dir.iterdir() if path.name != "memory-smoke.json"]
        if existing:
            raise FileExistsError(
                f"Global run directory is nonempty without a resumable checkpoint: {run_dir}"
            )
        selected_batch_size, memory_report = _memory_smoke(
            model,
            optimizer,
            scaler,
            data_by_country,
            domains,
            samplers,
            neighbor_generators,
            allowed_experts=allowed_experts,
            options=options,
            device=selected_device,
        )
        atomic_write_json(run_dir / "memory-smoke.json", memory_report)

    steps_completed = start_step - 1
    final_step = start_step - 1 if stopped_early else options.max_steps
    for step in range(start_step, final_step + 1):
        step_state = _rng_state(samplers, neighbor_generators)
        while True:
            optimizer.zero_grad(set_to_none=True)
            try:
                model.train()
                domain_losses = _domain_update_backward(
                    model,
                    data_by_country,
                    domains,
                    samplers,
                    neighbor_generators,
                    batch_size=selected_batch_size,
                    allowed_experts=allowed_experts,
                    options=options,
                    device=selected_device,
                    scaler=scaler,
                )
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), options.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                break
            except torch.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                _restore_rng_state(step_state, samplers, neighbor_generators)
                if selected_device.type == "cuda":
                    torch.cuda.empty_cache()
                smaller = next(
                    (
                        candidate
                        for candidate in options.memory_smoke_batch_sizes
                        if candidate < selected_batch_size
                    ),
                    None,
                )
                if smaller is None:
                    raise
                memory_report["oomFallbacks"].append(
                    {
                        "step": step,
                        "fromBatchSize": selected_batch_size,
                        "toBatchSize": smaller,
                    }
                )
                selected_batch_size = smaller
                memory_report["selectedSeedBatchSize"] = selected_batch_size
                logical = {
                    key: value
                    for key, value in memory_report.items()
                    if key != "memorySmokeHash"
                }
                memory_report["memorySmokeHash"] = canonical_sha256(logical)
                atomic_write_json(run_dir / "memory-smoke.json", memory_report)
        steps_completed = step
        evaluated = step % options.eval_every_steps == 0 or step == options.max_steps
        improved = False
        if evaluated:
            evaluation_options = replace(
                options, seed_batch_size=selected_batch_size
            )
            validation_logits, validation_labels = collect_split_logits(
                model,
                domains,
                split="validation",
                options=evaluation_options,
                device=selected_device,
                allowed_experts=allowed_experts,
            )
            threshold, score, per_country = country_balanced_threshold(
                validation_logits, validation_labels
            )
            improved = score > best_score + 1e-10
            if improved:
                best_score = score
                best_step = step
                best_threshold = threshold
                stale_evaluations = 0
                best_state = _clone_state(model)
            else:
                stale_evaluations += 1
            history.append(
                {
                    "step": step,
                    "meanDomainLoss": sum(domain_losses.values()) / len(domain_losses),
                    "domainLosses": domain_losses,
                    "optimizerSteps": step,
                    "domainBackwardPasses": len(domains),
                    "seedBatchSizePerDomain": selected_batch_size,
                    "countryBalancedMacroF1": score,
                    "threshold": threshold,
                    "perCountryMacroF1": per_country,
                    "improved": improved,
                }
            )
        checkpoint_due = (
            step % options.checkpoint_every_steps == 0
            or evaluated
            or step == options.max_steps
        )
        should_stop = (
            evaluated
            and step >= options.min_steps
            and stale_evaluations >= options.patience_evals
        )
        if checkpoint_due:
            checkpoint = {
                "schemaVersion": "socialgraph-fm.global-model-training-checkpoint/1.0",
                **expected,
                "seed": SEED,
                "stepCompleted": step,
                "nextStep": step + 1,
                "optimizerStepCount": step,
                "staleEvaluations": stale_evaluations,
                "stoppedEarly": should_stop,
                "bestStep": best_step,
                "bestValidationMacroF1": best_score,
                "validationThreshold": best_threshold,
                "history": history,
                "trainingEvidence": evidence,
                "modelState": _clone_state(model),
                "bestModelState": best_state,
                "optimizerState": optimizer.state_dict(),
                "scalerState": scaler.state_dict(),
                "rngState": _rng_state(samplers, neighbor_generators),
                "memorySmoke": memory_report,
                "ampEnabled": amp_enabled,
            }
            _atomic_torch_save(latest, checkpoint)
            if improved:
                _atomic_torch_save(
                    best,
                    {
                        "schemaVersion": "socialgraph-fm.global-model-best-checkpoint/1.0",
                        **expected,
                        "seed": SEED,
                        "bestStep": best_step,
                        "bestValidationMacroF1": best_score,
                        "validationThreshold": best_threshold,
                        "modelState": best_state,
                        "modelStateHash": tensor_state_hash(best_state),
                    },
                )
        if on_step_complete is not None:
            on_step_complete(step)
        if should_stop:
            stopped_early = True
            break
    if best_step < 1 or not best.exists():
        raise RuntimeError("Global training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    return TrainingOutcome(
        checkpoint_path=latest,
        best_checkpoint_path=best,
        checkpoint_sha256=file_sha256(best),
        model_state_hash=tensor_state_hash(best_state),
        best_step=best_step,
        steps_completed=steps_completed,
        global_step=steps_completed,
        best_validation_macro_f1=best_score,
        validation_threshold=best_threshold,
        stopped_early=stopped_early,
        resumed_from_step=resumed_from_step,
        amp_enabled=amp_enabled,
        history=tuple(history),
        memory_smoke=memory_report,
    )


__all__ = [
    "BalancedSeedSampler",
    "CountryOutputs",
    "DomainData",
    "InferenceData",
    "TrainingOptions",
    "TrainingOutcome",
    "collect_masked_outputs",
    "collect_split_logits",
    "country_balanced_batches",
    "country_balanced_threshold",
    "options_from_config",
    "tensor_state_hash",
    "train_balanced_neighbor_model",
]
