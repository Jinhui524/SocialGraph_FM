"""Cutoff-safe product heads, losses, training and ranking evaluation.

This module deliberately operates on already materialized ``CoreBatch``
objects.  Corpus adapters remain responsible for constructing the visible
graph and labels at a cutoff; the product trainer never receives a future
edge collection and therefore cannot use it for negative sampling.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from .evaluation import expected_calibration_error
from .model import SocialGraphFMCore
from .types import CoreBatch

ProductTask = Literal["collaboration", "newcomer"]


@dataclass(frozen=True)
class SampleProvenance:
    """Portable point-in-time identity carried by every product sample batch."""

    domain_id: str
    graph_version: str
    cutoff: float
    horizon: float
    task_id: ProductTask
    source_corpus_hash: str

    def validate(self) -> None:
        if not self.domain_id or not self.graph_version:
            raise ValueError("sample provenance requires domain and graph version")
        if not math.isfinite(self.cutoff) or not math.isfinite(self.horizon) or self.horizon <= 0:
            raise ValueError("sample provenance cutoff/horizon is invalid")
        if self.task_id not in ("collaboration", "newcomer"):
            raise ValueError("sample provenance has an unknown task")
        if len(self.source_corpus_hash) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_corpus_hash
        ):
            raise ValueError("source_corpus_hash must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ProductAdaptBatch:
    """One causal graph batch plus product supervision local to that graph."""

    core_batch: CoreBatch
    candidate_edge_index: Tensor
    pair_features: Tensor
    pair_labels: Tensor
    query_ids: Tensor
    provenance: SampleProvenance
    participation_node_index: Tensor | None = None
    participation_labels: Tensor | None = None

    def validate(self, *, pair_feature_dim: int) -> None:
        self.provenance.validate()
        self.core_batch.validate()
        if self.core_batch.domain_id != self.provenance.domain_id:
            raise ValueError("batch domain differs from its sample provenance")
        core_provenance = self.core_batch.provenance
        if (
            core_provenance.graph_version != self.provenance.graph_version
            or core_provenance.horizon != self.provenance.horizon
            or core_provenance.task_id != self.provenance.task_id
            or core_provenance.source_corpus_hash
            != self.provenance.source_corpus_hash
        ):
            raise ValueError("product and core sample provenance differ")
        core_cutoff = self.core_batch.cutoff_time
        if isinstance(core_cutoff, Tensor):
            if core_cutoff.numel() != 1:
                raise ValueError("core cutoff must be scalar")
            core_cutoff_value = float(core_cutoff.detach().cpu())
        else:
            core_cutoff_value = float(core_cutoff)
        if core_cutoff_value != self.provenance.cutoff:
            raise ValueError("core cutoff differs from its sample provenance")
        edges = self.candidate_edge_index
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("candidate_edge_index must be torch.long [2, E]")
        if edges.numel() and (
            int(edges.min()) < 0 or int(edges.max()) >= self.core_batch.num_nodes
        ):
            raise ValueError("candidate_edge_index contains an out-of-range node")
        count = int(edges.shape[1])
        if (
            self.pair_features.ndim != 2
            or self.pair_features.shape != (count, pair_feature_dim)
            or not self.pair_features.is_floating_point()
            or not bool(torch.isfinite(self.pair_features).all())
        ):
            raise ValueError("pair_features must be finite [E, pair_feature_dim]")
        labels = self.pair_labels.reshape(-1)
        queries = self.query_ids.reshape(-1)
        if labels.shape != (count,) or queries.shape != (count,):
            raise ValueError("pair labels and query IDs must align with candidate edges")
        if not labels.is_floating_point() or not bool(
            torch.all((labels == 0) | (labels == 1))
        ):
            raise ValueError("pair_labels must be binary floating point values")
        if queries.dtype != torch.long or (queries.numel() and int(queries.min()) < 0):
            raise ValueError("query_ids must be non-negative torch.long values")
        if count:
            candidates = edges.transpose(0, 1)
            unique_edges, inverse = torch.unique(
                candidates, dim=0, sorted=False, return_inverse=True
            )
            if unique_edges.shape[0] != count:
                for edge_index in range(int(unique_edges.shape[0])):
                    duplicate_labels = labels[inverse == edge_index]
                    if torch.unique(duplicate_labels).numel() > 1:
                        raise ValueError("a candidate edge has contradictory labels")
                raise ValueError("candidate edges must be globally unique within a batch")
        if count == 0 and self.participation_node_index is None:
            raise ValueError("an empty ranking batch requires participation supervision")
        for query in torch.unique(queries):
            selected = queries == query
            if torch.unique(edges[0, selected]).numel() != 1:
                raise ValueError("each ranking query must have exactly one focal source")
            if not bool((labels[selected] == 1).any()) or not bool(
                (labels[selected] == 0).any()
            ):
                raise ValueError("every ranking query requires positive and negative candidates")
        if (self.participation_node_index is None) != (
            self.participation_labels is None
        ):
            raise ValueError("participation nodes and labels must be supplied together")
        if self.participation_node_index is not None:
            nodes = self.participation_node_index.reshape(-1)
            outcomes = self.participation_labels.reshape(-1)  # type: ignore[union-attr]
            if (
                nodes.dtype != torch.long
                or nodes.shape != outcomes.shape
                or nodes.numel() == 0
                or (nodes.numel() and (int(nodes.min()) < 0 or int(nodes.max()) >= self.core_batch.num_nodes))
                or not outcomes.is_floating_point()
                or not bool(torch.all((outcomes == 0) | (outcomes == 1)))
            ):
                raise ValueError("participation supervision is malformed")

    def to(self, device: str | torch.device) -> ProductAdaptBatch:
        self.validate(pair_feature_dim=int(self.pair_features.shape[1]))

        def optional(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(device)

        moved = ProductAdaptBatch(
            core_batch=self.core_batch.to(device),
            candidate_edge_index=self.candidate_edge_index.to(device),
            pair_features=self.pair_features.to(device),
            pair_labels=self.pair_labels.to(device),
            query_ids=self.query_ids.to(device),
            provenance=self.provenance,
            participation_node_index=optional(self.participation_node_index),
            participation_labels=optional(self.participation_labels),
        )
        moved.validate(pair_feature_dim=int(self.pair_features.shape[1]))
        return moved


class ProductTaskModule(nn.Module):
    """SocialGraph-FM Core plus auditable collaboration/newcomer heads."""

    def __init__(
        self,
        core: SocialGraphFMCore,
        *,
        task: ProductTask,
        pair_feature_dim: int = 8,
    ) -> None:
        super().__init__()
        if task not in ("collaboration", "newcomer") or pair_feature_dim < 1:
            raise ValueError("unsupported product task or pair feature width")
        self.core = core
        self.task: ProductTask = task
        self.pair_feature_dim = int(pair_feature_dim)
        hidden = core.config.hidden_channels
        self.pair_head = nn.Sequential(
            nn.Linear(hidden * 2 + self.pair_feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(core.config.dropout),
            nn.Linear(hidden, 1),
        )
        self.participation_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(core.config.dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: ProductAdaptBatch) -> tuple[Tensor, Tensor | None]:
        batch.validate(pair_feature_dim=self.pair_feature_dim)
        if batch.provenance.task_id != self.task:
            raise ValueError("product module task differs from the sample provenance")
        output = self.core(batch.core_batch)
        source, target = batch.candidate_edge_index
        if source.numel():
            pair = torch.cat(
                (
                    output.node_embeddings[source] * output.node_embeddings[target],
                    torch.abs(
                        output.node_embeddings[source] - output.node_embeddings[target]
                    ),
                    batch.pair_features,
                ),
                dim=-1,
            )
            pair_logits = self.pair_head(pair).reshape(-1)
        else:
            pair_logits = output.node_embeddings.new_empty((0,))
        participation_logits: Tensor | None = None
        if batch.participation_node_index is not None:
            participation_logits = self.participation_head(
                output.node_embeddings[batch.participation_node_index]
            ).reshape(-1)
        return pair_logits, participation_logits


def load_product_backbone(
    target: SocialGraphFMCore,
    pretrained_state: Mapping[str, Tensor],
) -> tuple[str, ...]:
    """Load only shape-compatible shared encoder weights, never task heads."""

    blocked = ("link_head.", "node_head.", "graph_head.")
    destination = target.state_dict()
    selected: dict[str, Tensor] = {}
    for name, value in pretrained_state.items():
        if name.startswith(blocked):
            continue
        if name not in destination or destination[name].shape != value.shape:
            raise ValueError(f"pretrained shared component is incompatible: {name}")
        selected[name] = value
    required = {
        name for name in destination if not name.startswith(blocked)
    }
    if set(selected) != required:
        missing = sorted(required.difference(selected))
        raise ValueError(f"pretrained backbone is incomplete: {missing[:5]}")
    incompatible = target.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys or set(incompatible.missing_keys) != {
        name for name in destination if name.startswith(blocked)
    }:
        raise ValueError("product backbone load crossed an undeclared component boundary")
    return tuple(sorted(selected))


def pairwise_ranking_loss(logits: Tensor, labels: Tensor, query_ids: Tensor) -> Tensor:
    """Mean logistic pairwise loss, equally weighting each query."""

    scores = logits.reshape(-1)
    targets = labels.reshape(-1)
    queries = query_ids.reshape(-1)
    if scores.shape != targets.shape or scores.shape != queries.shape:
        raise ValueError("ranking values must be aligned")
    if not scores.numel():
        return scores.new_zeros(())
    losses: list[Tensor] = []
    for query in torch.unique(queries, sorted=True):
        selected = queries == query
        positive = scores[selected & (targets == 1)]
        negative = scores[selected & (targets == 0)]
        if not positive.numel() or not negative.numel():
            raise ValueError("each query needs both ranking classes")
        losses.append(torch.nn.functional.softplus(-(positive[:, None] - negative[None, :])).mean())
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class ProductLoss:
    total: Tensor
    ranking: Tensor
    pair_bce: Tensor
    participation_bce: Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().float().cpu()),
            "ranking": float(self.ranking.detach().float().cpu()),
            "pair_bce": float(self.pair_bce.detach().float().cpu()),
            "participation_bce": float(self.participation_bce.detach().float().cpu()),
        }


def product_multitask_loss(
    *,
    task: ProductTask,
    pair_logits: Tensor,
    batch: ProductAdaptBatch,
    participation_logits: Tensor | None,
) -> ProductLoss:
    ranking = pairwise_ranking_loss(pair_logits, batch.pair_labels, batch.query_ids)
    pair_bce = (
        torch.nn.functional.binary_cross_entropy_with_logits(
            pair_logits.reshape(-1), batch.pair_labels.reshape(-1)
        )
        if pair_logits.numel()
        else pair_logits.new_zeros(())
    )
    participation = pair_logits.new_zeros(())
    if task == "newcomer":
        if participation_logits is None or batch.participation_labels is None:
            raise ValueError("newcomer adaptation requires participation supervision")
        participation = torch.nn.functional.binary_cross_entropy_with_logits(
            participation_logits.reshape(-1), batch.participation_labels.reshape(-1)
        )
        total = ranking + 0.5 * participation
        if not pair_logits.numel():
            # Participation-only cohorts are essential: newcomers without a
            # future support relation remain valid negative outcome examples.
            total = 0.5 * participation
    elif task == "collaboration":
        if not pair_logits.numel():
            raise ValueError("collaboration adaptation requires ranking candidates")
        total = ranking + 0.5 * pair_bce
    else:  # pragma: no cover - guarded by the public type and module constructor
        raise ValueError("unknown product task")
    return ProductLoss(total, ranking, pair_bce, participation)


@dataclass(frozen=True)
class ProductTrainingConfig:
    maximum_steps: int
    minimum_steps: int
    evaluation_every_steps: int
    patience_evaluations: int
    gradient_clip: float = 1.0
    amp: bool = True
    train_iterator_contract_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            self.maximum_steps < 1
            or not 0 <= self.minimum_steps <= self.maximum_steps
            or self.evaluation_every_steps < 1
            or self.patience_evaluations < 1
            or self.gradient_clip <= 0
            or (
                self.train_iterator_contract_hash is not None
                and (
                    len(self.train_iterator_contract_hash) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in self.train_iterator_contract_hash
                    )
                )
            )
        ):
            raise ValueError("invalid product training configuration")


@dataclass(frozen=True)
class ProductResumeState:
    """Complete latest-step state required to resume an interrupted adaptation."""

    completed_steps: int
    train_epoch: int
    train_batch_offset: int
    train_iterator_contract_hash: str | None
    latest_model_state: Mapping[str, Tensor]
    optimizer_state: Mapping[str, Any]
    scheduler_state: Mapping[str, Any] | None
    scaler_state: Mapping[str, Any]
    best_step: int
    best_validation_loss: float
    best_state: Mapping[str, Tensor]
    no_improvement_evaluations: int
    history: tuple[Mapping[str, float], ...]
    python_rng_state: tuple[Any, ...]
    numpy_rng_state: tuple[Any, ...]
    torch_rng_state: Tensor
    cuda_rng_states: tuple[Tensor, ...]


@dataclass(frozen=True)
class ProductTrainingResult:
    best_step: int
    completed_steps: int
    best_validation_loss: float
    best_state: Mapping[str, Tensor]
    history: tuple[Mapping[str, float], ...]
    peak_cuda_memory_mib: float
    resume_state: ProductResumeState


BatchFactory = Callable[[], Iterable[ProductAdaptBatch]]
ProductProgressCallback = Callable[[ProductResumeState], None]


def _query_ids(batch: ProductAdaptBatch) -> set[int]:
    return {
        int(value)
        for value in batch.query_ids.detach().to(device="cpu", dtype=torch.long).tolist()
    }


def _assert_new_global_queries(
    batch: ProductAdaptBatch,
    seen: set[int],
    *,
    context: str,
) -> None:
    current = _query_ids(batch)
    overlap = current.intersection(seen)
    if overlap:
        example = min(overlap)
        raise ValueError(
            f"{context} query IDs must be globally unique across batches; collision={example}"
        )
    seen.update(current)


def _clone_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cpu(item) for item in value)
    return value


def _state_dict_cpu(module: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in module.state_dict().items()
    }


def _validation_loss(
    model: ProductTaskModule,
    batches: Iterable[ProductAdaptBatch],
    *,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    values: list[float] = []
    seen_queries: set[int] = set()
    with torch.inference_mode():
        for source in batches:
            _assert_new_global_queries(
                source, seen_queries, context="validation"
            )
            batch = source.to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp and device.type == "cuda",
            ):
                pair, participation = model(batch)
                loss = product_multitask_loss(
                    task=model.task,
                    pair_logits=pair,
                    batch=batch,
                    participation_logits=participation,
                ).total
            values.append(float(loss.float().cpu()))
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("product validation loss is empty or non-finite")
    return float(sum(values) / len(values))


def train_product_steps(
    model: ProductTaskModule,
    optimizer: torch.optim.Optimizer,
    *,
    train_batches: BatchFactory,
    validation_batches: Iterable[ProductAdaptBatch] | BatchFactory,
    device: str | torch.device,
    config: ProductTrainingConfig,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    resume_state: ProductResumeState | None = None,
    progress_callback: ProductProgressCallback | None = None,
) -> ProductTrainingResult:
    """Fine-tune without reading test data; only validation chooses the state."""

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA product adaptation was requested but is unavailable")
    model.to(selected_device)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(config.amp and selected_device.type == "cuda")
    )
    if resume_state is None:
        best_loss = math.inf
        best_step = 0
        best_state: dict[str, Tensor] | None = None
        no_improvement = 0
        history: list[Mapping[str, float]] = []
        step = 0
        train_epoch = 0
        train_batch_offset = 0
    else:
        if resume_state.completed_steps < 0 or resume_state.completed_steps > config.maximum_steps:
            raise ValueError("resume step is outside this training configuration")
        if (
            isinstance(resume_state.train_epoch, bool)
            or isinstance(resume_state.train_batch_offset, bool)
            or resume_state.train_epoch < 0
            or resume_state.train_batch_offset < 0
        ):
            raise ValueError("resume train iterator cursor is invalid")
        if resume_state.train_iterator_contract_hash != config.train_iterator_contract_hash:
            raise ValueError("resume train iterator contract differs from the current run")
        model.load_state_dict(resume_state.latest_model_state, strict=True)
        optimizer.load_state_dict(dict(resume_state.optimizer_state))
        if (scheduler is None) != (resume_state.scheduler_state is None):
            raise ValueError("resume scheduler presence differs from the current run")
        if scheduler is not None and resume_state.scheduler_state is not None:
            scheduler.load_state_dict(dict(resume_state.scheduler_state))
        scaler.load_state_dict(dict(resume_state.scaler_state))
        random.setstate(resume_state.python_rng_state)
        np.random.set_state(resume_state.numpy_rng_state)
        torch.set_rng_state(resume_state.torch_rng_state.detach().cpu())
        if selected_device.type == "cuda":
            if len(resume_state.cuda_rng_states) != torch.cuda.device_count():
                raise ValueError("resume CUDA RNG topology differs from the current process")
            torch.cuda.set_rng_state_all(
                [value.detach().cpu() for value in resume_state.cuda_rng_states]
            )
        best_loss = float(resume_state.best_validation_loss)
        best_step = int(resume_state.best_step)
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in resume_state.best_state.items()
        }
        no_improvement = int(resume_state.no_improvement_evaluations)
        history = list(resume_state.history)
        step = int(resume_state.completed_steps)
        train_epoch = int(resume_state.train_epoch)
        train_batch_offset = int(resume_state.train_batch_offset)
    if scheduler is not None and int(scheduler.last_epoch) != step:
        raise ValueError("scheduler progress differs from completed product optimizer steps")
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)

    def validation_source() -> Iterable[ProductAdaptBatch]:
        return validation_batches() if callable(validation_batches) else validation_batches

    def resumable_state() -> ProductResumeState:
        if best_state is None or not math.isfinite(best_loss):
            raise RuntimeError("product progress has no validation-selected state")
        if scheduler is not None and int(scheduler.last_epoch) != step:
            raise RuntimeError("scheduler progress differs from product checkpoint progress")
        return ProductResumeState(
            completed_steps=step,
            train_epoch=train_epoch,
            train_batch_offset=train_batch_offset,
            train_iterator_contract_hash=config.train_iterator_contract_hash,
            latest_model_state=_state_dict_cpu(model),
            optimizer_state=_clone_cpu(optimizer.state_dict()),
            scheduler_state=(
                None if scheduler is None else _clone_cpu(scheduler.state_dict())
            ),
            scaler_state=_clone_cpu(scaler.state_dict()),
            best_step=best_step,
            best_validation_loss=best_loss,
            best_state={
                name: value.detach().cpu().clone()
                for name, value in best_state.items()
            },
            no_improvement_evaluations=no_improvement,
            history=tuple(history),
            python_rng_state=random.getstate(),
            numpy_rng_state=cast(tuple[Any, ...], np.random.get_state(legacy=True)),
            torch_rng_state=torch.get_rng_state().detach().cpu().clone(),
            cuda_rng_states=tuple(
                value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all()
            ) if selected_device.type == "cuda" else (),
        )

    while step < config.maximum_steps:
        cursor_at_epoch_start = train_batch_offset
        iterator = iter(train_batches())
        seen_train_queries: set[int] = set()
        # A recovery checkpoint names the *next* batch in a deterministic,
        # rebuildable epoch.  Replaying the already-consumed prefix advances
        # generator-local state and re-establishes query uniqueness without
        # applying an optimizer update.  Restore global RNG afterwards: those
        # draws were already reflected in the checkpoint and must not be
        # consumed twice before the next model forward.
        replay_python_rng = random.getstate()
        replay_numpy_rng = cast(tuple[Any, ...], np.random.get_state(legacy=True))
        replay_torch_rng = torch.get_rng_state().detach().cpu().clone()
        replay_cuda_rng = (
            tuple(value.detach().cpu().clone() for value in torch.cuda.get_rng_state_all())
            if selected_device.type == "cuda"
            else ()
        )
        try:
            for _ in range(train_batch_offset):
                try:
                    skipped = next(iterator)
                except StopIteration as error:
                    raise ValueError(
                        "resume train iterator cursor exceeds the rebuilt epoch"
                    ) from error
                _assert_new_global_queries(
                    skipped, seen_train_queries, context="replayed training epoch"
                )
        finally:
            random.setstate(replay_python_rng)
            np.random.set_state(replay_numpy_rng)
            torch.set_rng_state(replay_torch_rng)
            if selected_device.type == "cuda":
                torch.cuda.set_rng_state_all(list(replay_cuda_rng))
        made_progress = False
        epoch_exhausted = True
        for source in iterator:
            made_progress = True
            _assert_new_global_queries(
                source, seen_train_queries, context="training epoch"
            )
            model.train()
            batch = source.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=torch.float16,
                enabled=scaler.is_enabled(),
            ):
                pair, participation = model(batch)
                losses = product_multitask_loss(
                    task=model.task,
                    pair_logits=pair,
                    batch=batch,
                    participation_logits=participation,
                )
            if not bool(torch.isfinite(losses.total)):
                raise RuntimeError("product adaptation loss is not finite")
            scaler.scale(losses.total).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            if not bool(torch.isfinite(torch.as_tensor(norm))):
                raise RuntimeError("product adaptation gradient norm is not finite")
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            step += 1
            train_batch_offset += 1
            if scheduler is not None and int(scheduler.last_epoch) != step:
                raise RuntimeError(
                    "scheduler did not advance with the product optimizer step"
                )
            if step % config.evaluation_every_steps == 0 or step == config.maximum_steps:
                validation = _validation_loss(
                    model,
                    validation_source(),
                    device=selected_device,
                    amp=config.amp,
                )
                record = {**losses.detached(), "step": float(step), "validation": validation}
                history.append(record)
                if validation < best_loss:
                    best_loss = validation
                    best_step = step
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1
                if progress_callback is not None:
                    # The callback receives a complete optimizer-aligned state
                    # only at validation boundaries.  Callers can therefore
                    # durably checkpoint without treating an unvalidated
                    # micro-step as a formal recovery point.
                    progress_callback(resumable_state())
                if (
                    step >= config.minimum_steps
                    and no_improvement >= config.patience_evaluations
                ):
                    epoch_exhausted = False
                    break
            if step >= config.maximum_steps:
                epoch_exhausted = False
                break
        if epoch_exhausted:
            train_epoch += 1
            train_batch_offset = 0
        if not made_progress and cursor_at_epoch_start == 0:
            raise RuntimeError("product train batch factory yielded no batches")
        if (
            step >= config.maximum_steps
            or (
                step >= config.minimum_steps
                and no_improvement >= config.patience_evaluations
            )
        ):
            break
    if best_state is None or not math.isfinite(best_loss):
        raise RuntimeError("product adaptation produced no validation-selected state")
    peak = (
        torch.cuda.max_memory_allocated(selected_device) / (1024**2)
        if selected_device.type == "cuda"
        else 0.0
    )
    resumable = resumable_state()
    # Callers export/evaluate the validation-selected state.  The independent
    # resume payload retains the latest optimizer-aligned model state.
    model.load_state_dict(best_state, strict=True)
    return ProductTrainingResult(
        best_step=best_step,
        completed_steps=step,
        best_validation_loss=best_loss,
        best_state=best_state,
        history=tuple(history),
        peak_cuda_memory_mib=float(peak),
        resume_state=resumable,
    )


def _as_numpy(
    value: Tensor | np.ndarray | Sequence[float] | Sequence[int],
    *,
    name: str,
) -> np.ndarray:
    if isinstance(value, Tensor):
        try:
            result = value.detach().cpu().numpy()
        except TypeError as error:
            raise ValueError(f"{name} uses a dtype NumPy cannot represent exactly") from error
    else:
        result = np.asarray(value)
    result = np.asarray(result).reshape(-1)
    if result.dtype.kind not in "biuf":
        raise ValueError(f"{name} must use a real numeric dtype")
    if not result.size or not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _probabilities(
    value: Tensor | np.ndarray | Sequence[float], *, name: str
) -> np.ndarray:
    result = _as_numpy(value, name=name)
    if result.dtype.kind != "f":
        raise ValueError(f"{name} must use a floating-point dtype")
    if not bool(np.all((result >= 0.0) & (result <= 1.0))):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result.astype(np.float64, copy=False)


def _binary_labels(
    value: Tensor | np.ndarray | Sequence[float] | Sequence[int], *, name: str
) -> np.ndarray:
    result = _as_numpy(value, name=name)
    if result.dtype.kind not in "biuf" or not bool(
        np.all((result == 0) | (result == 1))
    ):
        raise ValueError(f"{name} must contain exact binary labels before casting")
    return result.astype(np.int64, copy=False)


def _integer_ids(
    value: Tensor | np.ndarray | Sequence[int], *, name: str
) -> np.ndarray:
    result = _as_numpy(value, name=name)
    if result.dtype.kind not in "iu" or bool((result < 0).any()):
        raise ValueError(f"{name} must use a non-negative integer dtype")
    if np.unique(result).size != result.size and name == "sample_ids":
        raise ValueError("sample_ids must be globally unique")
    return result


def _per_query_ranking(
    scores: np.ndarray,
    labels: np.ndarray,
    queries: np.ndarray,
    *,
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray]:
    ndcg: list[float] = []
    recall: list[float] = []
    for query in np.unique(queries):
        selected = np.flatnonzero(queries == query)
        query_labels = labels[selected]
        relevant = int(query_labels.sum())
        if relevant < 1 or relevant == query_labels.size:
            raise ValueError(
                "ranking evaluation requires a positive and negative for every query"
            )
        # Stable input order is independent of the outcome label.  Using labels
        # as a secondary tie-breaker would let ground truth influence a metric.
        order = np.argsort(-scores[selected], kind="mergesort")
        ranked = query_labels[order][:cutoff]
        ranks = np.arange(1, ranked.size + 1, dtype=np.float64)
        dcg = float(np.sum(ranked / np.log2(ranks + 1.0)))
        ideal_count = min(relevant, cutoff)
        ideal = float(
            np.sum(1.0 / np.log2(np.arange(1, ideal_count + 1, dtype=np.float64) + 1.0))
        )
        ndcg.append(dcg / ideal)
        recall.append(float(ranked.sum() / relevant))
    return np.asarray(ndcg, dtype=np.float64), np.asarray(recall, dtype=np.float64)


def binary_average_precision(
    probabilities: Tensor | np.ndarray | Sequence[float],
    labels: Tensor | np.ndarray | Sequence[float],
) -> float:
    scores = _probabilities(probabilities, name="probabilities")
    targets = _binary_labels(labels, name="labels")
    if scores.shape != targets.shape:
        raise ValueError("average precision requires aligned binary labels")
    positives = int(targets.sum())
    if positives < 1:
        raise ValueError("average precision requires at least one positive")
    order = np.argsort(-scores, kind="mergesort")
    ranked_scores = scores[order]
    ranked_targets = targets[order]
    # AP is defined at distinct score thresholds.  Grouping a tie before
    # computing precision makes the result invariant to candidate ordering.
    group_end = np.flatnonzero(
        np.r_[ranked_scores[1:] != ranked_scores[:-1], True]
    )
    cumulative_positive = np.cumsum(ranked_targets)[group_end]
    retrieved = group_end + 1
    group_positive = np.diff(np.r_[0, cumulative_positive])
    precision = cumulative_positive / retrieved
    return float(np.sum((group_positive / positives) * precision))


@dataclass(frozen=True)
class ProductPredictionReport:
    ndcg_at_20: float
    recall_at_20: float
    baseline_ndcg_at_20: float
    baseline_recall_at_20: float
    bootstrap_ndcg_gain_lower: float
    bootstrap_ndcg_gain_upper: float
    auprc: float
    label_prevalence: float
    ece: float
    brier: float
    query_count: int
    outcome_count: int
    outcome_kind: Literal["pair_event", "participation_outcome"]

    def metrics(self) -> dict[str, float]:
        return {
            "ndcg@20": self.ndcg_at_20,
            "recall@20": self.recall_at_20,
            "baseline_ndcg@20": self.baseline_ndcg_at_20,
            "baseline_recall@20": self.baseline_recall_at_20,
            "bootstrap_ci95_ndcg_gain_lower": self.bootstrap_ndcg_gain_lower,
            "bootstrap_ci95_ndcg_gain_upper": self.bootstrap_ndcg_gain_upper,
            "auprc": self.auprc,
            "label_prevalence": self.label_prevalence,
            "ece": self.ece,
            "brier": self.brier,
            "query_count": float(self.query_count),
            "outcome_count": float(self.outcome_count),
        }


def calibration_by_stratum(
    *,
    probabilities: Tensor | np.ndarray | Sequence[float],
    labels: Tensor | np.ndarray | Sequence[float],
    sample_ids: Tensor | np.ndarray | Sequence[int],
    provenance: SampleProvenance,
    strata: Mapping[str, Tensor | np.ndarray | Sequence[bool]],
    required_partitions: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Compute ECE after proving each declared stratum axis exactly partitions samples."""

    provenance.validate()
    probability = _probabilities(probabilities, name="probabilities")
    target = _binary_labels(labels, name="labels")
    identities = _integer_ids(sample_ids, name="sample_ids")
    if probability.shape != target.shape or target.shape != identities.shape or not strata:
        raise ValueError("stratified calibration requires aligned values, IDs and strata")
    if not required_partitions:
        raise ValueError("calibration requires named partition axes")
    masks: dict[str, np.ndarray] = {}
    for name, raw_mask in sorted(strata.items()):
        if not name or not name.replace("_", "").isalnum():
            raise ValueError("calibration stratum name is unsafe")
        mask = np.asarray(raw_mask).reshape(-1)
        if mask.shape != probability.shape or mask.dtype != np.bool_ or not bool(mask.any()):
            raise ValueError(f"calibration stratum {name!r} is empty or misaligned")
        masks[name] = mask
    declared_names: list[str] = []
    for axis, names in sorted(required_partitions.items()):
        if not axis or not axis.replace("_", "").isalnum() or len(names) < 2:
            raise ValueError("calibration partition axis is unsafe or incomplete")
        if len(set(names)) != len(names) or any(name not in masks for name in names):
            raise ValueError(f"calibration partition {axis!r} references invalid strata")
        coverage = np.sum(np.stack([masks[name] for name in names]), axis=0)
        if not bool(np.all(coverage == 1)):
            raise ValueError(
                f"calibration partition {axis!r} must cover every sample exactly once"
            )
        declared_names.extend(names)
    if len(set(declared_names)) != len(declared_names) or set(declared_names) != set(masks):
        raise ValueError("every calibration stratum must belong to exactly one partition axis")
    result: dict[str, float] = {}
    for name, mask in sorted(masks.items()):
        if np.unique(target[mask]).size < 2:
            raise ValueError(f"calibration stratum {name!r} lacks both outcome classes")
        metric = expected_calibration_error(
            torch.from_numpy(probability[mask]), torch.from_numpy(target[mask])
        )
        result[f"ece_{name}"] = metric.expected_calibration_error
    return result


def evaluate_product_predictions(
    *,
    task: ProductTask,
    ranking_probabilities: Tensor | np.ndarray | Sequence[float],
    ranking_labels: Tensor | np.ndarray | Sequence[float],
    query_ids: Tensor | np.ndarray | Sequence[int],
    baseline_scores: Tensor | np.ndarray | Sequence[float],
    participation_probabilities: Tensor | np.ndarray | Sequence[float] | None = None,
    participation_labels: Tensor | np.ndarray | Sequence[float] | None = None,
    seed: int,
    bootstrap_samples: int = 2_000,
    minimum_query_count: int = 100,
) -> ProductPredictionReport:
    """Evaluate candidate ranking and the task's distinct calibrated outcome.

    For ``newcomer``, the ranking arrays are supporter candidates, while
    AUCPR/ECE/Brier are computed exclusively from the separately supplied
    participation probabilities and labels.  For ``collaboration``, the pair
    event itself is both the ranking target and calibrated outcome.
    """

    if task not in ("collaboration", "newcomer"):
        raise ValueError("unknown product evaluation task")
    probability = _probabilities(ranking_probabilities, name="ranking_probabilities")
    target = _binary_labels(ranking_labels, name="ranking_labels")
    query = _integer_ids(query_ids, name="query_ids")
    baseline_raw = _as_numpy(baseline_scores, name="baseline_scores")
    if baseline_raw.dtype.kind != "f":
        raise ValueError("baseline_scores must use a floating-point dtype")
    baseline = baseline_raw.astype(np.float64, copy=False)
    if not (probability.shape == target.shape == query.shape == baseline.shape):
        raise ValueError("product predictions must be aligned")
    if minimum_query_count < 1:
        raise ValueError("minimum_query_count must be positive")
    query_count = int(np.unique(query).size)
    if query_count < minimum_query_count:
        raise ValueError(
            f"formal product evaluation requires at least {minimum_query_count} queries"
        )
    model_ndcg, model_recall = _per_query_ranking(
        probability, target, query, cutoff=20
    )
    base_ndcg, base_recall = _per_query_ranking(baseline, target, query, cutoff=20)
    differences = model_ndcg - base_ndcg
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    rng = np.random.default_rng(int(seed))
    sampled = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        draw = rng.integers(0, differences.size, size=differences.size)
        sampled[index] = float(differences[draw].mean())
    lower, upper = np.quantile(sampled, (0.025, 0.975)).tolist()
    if task == "newcomer":
        if participation_probabilities is None or participation_labels is None:
            raise ValueError("newcomer evaluation requires participation outcomes")
        outcome_probability = _probabilities(
            participation_probabilities, name="participation_probabilities"
        )
        outcome_target = _binary_labels(
            participation_labels, name="participation_labels"
        )
        outcome_kind: Literal["pair_event", "participation_outcome"] = (
            "participation_outcome"
        )
    else:
        if participation_probabilities is not None or participation_labels is not None:
            raise ValueError("collaboration evaluation cannot mix participation outcomes")
        outcome_probability = probability
        outcome_target = target
        outcome_kind = "pair_event"
    if outcome_probability.shape != outcome_target.shape:
        raise ValueError("product outcome probabilities and labels must be aligned")
    if np.unique(outcome_target).size < 2:
        raise ValueError("product outcome supervision requires both classes")
    calibration = expected_calibration_error(
        torch.from_numpy(outcome_probability),
        torch.from_numpy(outcome_target.astype(np.float64)),
    )
    return ProductPredictionReport(
        ndcg_at_20=float(model_ndcg.mean()),
        recall_at_20=float(model_recall.mean()),
        baseline_ndcg_at_20=float(base_ndcg.mean()),
        baseline_recall_at_20=float(base_recall.mean()),
        bootstrap_ndcg_gain_lower=float(lower),
        bootstrap_ndcg_gain_upper=float(upper),
        auprc=binary_average_precision(outcome_probability, outcome_target),
        label_prevalence=float(outcome_target.mean()),
        ece=calibration.expected_calibration_error,
        brier=calibration.brier_score,
        query_count=query_count,
        outcome_count=int(outcome_target.size),
        outcome_kind=outcome_kind,
    )


__all__ = [
    "ProductAdaptBatch",
    "ProductLoss",
    "ProductPredictionReport",
    "ProductResumeState",
    "ProductProgressCallback",
    "ProductTaskModule",
    "ProductTrainingConfig",
    "ProductTrainingResult",
    "SampleProvenance",
    "binary_average_precision",
    "calibration_by_stratum",
    "evaluate_product_predictions",
    "load_product_backbone",
    "pairwise_ranking_loss",
    "product_multitask_loss",
    "train_product_steps",
]
