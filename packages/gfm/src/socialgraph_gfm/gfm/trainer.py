"""AMP and gradient-accumulated multi-domain trainer for SocialGraph-FM Core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import torch
from torch import Tensor

from .model import SocialGraphFMCore
from .objectives import (
    LossBundle,
    compute_fixed_multiloss,
    embedding_distribution_moments,
)
from .sampling import RoundRobinDomainScheduler
from .types import CoreBatch


@dataclass(frozen=True)
class CoreTrainerConfig:
    gradient_accumulation_steps: int = 4
    gradient_clip: float = 1.0
    amp: bool = True

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")


@dataclass(frozen=True)
class TrainingEpochResult:
    batch_steps: int
    optimizer_steps: int
    mean_losses: Mapping[str, float]
    domain_steps: Mapping[str, int]


class CoreTrainer:
    """Train one shared model while visiting source domains in round-robin order."""

    def __init__(
        self,
        model: SocialGraphFMCore,
        optimizer: torch.optim.Optimizer,
        config: CoreTrainerConfig,
        device: str | torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("CoreTrainer supports cpu or cuda")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.scheduler = RoundRobinDomainScheduler(model.config.domains)
        self.global_step = 0
        self.optimizer_step = 0
        self.model.to(self.device)

    def _forward_loss_and_moments(
        self, batch: CoreBatch, cross_domain_reference: Tensor | None = None
    ) -> tuple[LossBundle, Tensor]:
        batch.validate(self.model.config)
        if (
            batch.positive_edge_index is None
            or batch.negative_edge_index is None
            or batch.positive_relation is None
            or batch.positive_relation_mask is None
            or batch.time_delta_targets is None
            or batch.time_delta_mask is None
        ):
            raise ValueError(
                "v1 pretraining requires event, masked relation and time-delta supervision"
            )
        output = self.model(batch)
        predictions = {
            name: self.model.reconstruct_modality(output.node_embeddings, name)
            for name in batch.attribute_targets
        }
        positive_logits = self.model.score_links(
            output.node_embeddings,
            batch.positive_edge_index,
            batch.positive_pair_features,
        )
        negative_logits = self.model.score_links(
            output.node_embeddings,
            batch.negative_edge_index,
            batch.negative_pair_features,
        )
        relation_logits = self.model.classify_relations(
            output.node_embeddings, batch.positive_edge_index
        )
        time_delta_predictions = self.model.predict_log_time_delta(
            output.node_embeddings, batch.positive_edge_index
        )
        text_embeddings: Tensor | None = None
        text_mask: Tensor | None = None
        text_modality = self.model.config.text_modality
        if text_modality is not None and text_modality in batch.modalities:
            text_embeddings = self.model.project_modality(
                text_modality, batch.modalities[text_modality]
            )
            text_mask = batch.modality_masks[text_modality]
        losses = compute_fixed_multiloss(
            attribute_predictions=predictions,
            attribute_targets=batch.attribute_targets,
            attribute_masks=batch.attribute_masks,
            positive_link_logits=positive_logits,
            negative_link_logits=negative_logits,
            relation_logits=relation_logits,
            relation_labels=batch.positive_relation,
            relation_mask=batch.positive_relation_mask,
            log_time_delta_predictions=time_delta_predictions,
            time_delta_targets=batch.time_delta_targets,
            time_delta_mask=batch.time_delta_mask,
            text_embeddings=text_embeddings,
            structure_embeddings=output.semantic_embeddings,
            text_mask=text_mask,
            cross_domain_reference=cross_domain_reference,
            expert_weights=output.expert_weights,
            moe_enabled=self.model.config.variant == "moe",
            text_modality=self.model.config.text_modality,
        )
        return losses, embedding_distribution_moments(output.semantic_embeddings)

    def forward_loss(
        self, batch: CoreBatch, cross_domain_reference: Tensor | None = None
    ) -> LossBundle:
        losses, _ = self._forward_loss_and_moments(batch, cross_domain_reference)
        return losses

    def _apply_optimizer_step(self, *, partial_accumulation: int | None = None) -> None:
        self.scaler.unscale_(self.optimizer)
        if partial_accumulation is not None:
            correction = self.config.gradient_accumulation_steps / partial_accumulation
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise RuntimeError("SocialGraph-FM Core gradient norm is not finite")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step += 1

    def train_epoch(
        self, domain_loaders: Mapping[str, Iterable[CoreBatch]]
    ) -> TrainingEpochResult:
        if not domain_loaders:
            raise ValueError("train_epoch requires at least one domain loader")
        unknown = set(domain_loaders).difference(self.model.config.domains)
        if unknown:
            raise ValueError(f"train_epoch received unknown domains: {sorted(unknown)}")
        iterators = {domain: iter(loader) for domain, loader in domain_loaders.items()}
        active = set(iterators)
        component_sums: dict[str, float] = {}
        domain_steps = {domain: 0 for domain in domain_loaders}
        batch_steps = 0
        pending = 0
        domain_moments: dict[str, Tensor] = {}
        epoch_optimizer_start = self.optimizer_step
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        while active:
            domain = self.scheduler.next_domain(active)
            try:
                source_batch = next(iterators[domain])
            except StopIteration:
                active.remove(domain)
                continue
            if source_batch.domain_id != domain:
                raise ValueError("domain loader yielded a batch belonging to another domain")
            batch = source_batch.to(self.device)
            other_moments = [
                moments for other_domain, moments in domain_moments.items() if other_domain != domain
            ]
            cross_domain_reference = (
                torch.stack(other_moments).mean(dim=0) if other_moments else None
            )
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                losses, moments = self._forward_loss_and_moments(
                    batch, cross_domain_reference
                )
                scaled_loss = losses.total / self.config.gradient_accumulation_steps
            if not bool(torch.isfinite(scaled_loss)):
                raise RuntimeError("SocialGraph-FM Core training loss is not finite")
            self.scaler.scale(scaled_loss).backward()
            domain_moments[domain] = moments.detach()
            batch_steps += 1
            pending += 1
            self.global_step += 1
            domain_steps[domain] += 1
            for name, value in losses.detached().items():
                component_sums[name] = component_sums.get(name, 0.0) + value
            if pending == self.config.gradient_accumulation_steps:
                self._apply_optimizer_step()
                pending = 0

        if not batch_steps:
            raise ValueError("all domain loaders were empty")
        if pending:
            self._apply_optimizer_step(partial_accumulation=pending)
        return TrainingEpochResult(
            batch_steps=batch_steps,
            optimizer_steps=self.optimizer_step - epoch_optimizer_start,
            mean_losses={name: value / batch_steps for name, value in component_sums.items()},
            domain_steps=domain_steps,
        )

    @torch.no_grad()
    def evaluate_batch(self, batch: CoreBatch) -> dict[str, float]:
        self.model.eval()
        moved = batch.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        ):
            return self.forward_loss(moved).detached()

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "format": "gfm.core-trainer-state/1.0",
            "trainerConfig": asdict(self.config),
            "domains": self.model.config.domains,
            "modelState": self.model.state_dict(),
            "optimizerState": self.optimizer.state_dict(),
            "scalerState": self.scaler.state_dict(),
            "domainSchedulerState": self.scheduler.state_dict(),
            "globalStep": self.global_step,
            "optimizerStep": self.optimizer_step,
            "torchCpuRngState": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torchCudaRngState"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("format") != "gfm.core-trainer-state/1.0":
            raise ValueError("unsupported SocialGraph-FM Core trainer state")
        if state.get("trainerConfig") != asdict(self.config) or tuple(
            state.get("domains", ())
        ) != self.model.config.domains:
            raise ValueError("trainer state identity differs from the active trainer")
        self.model.load_state_dict(state["modelState"])
        self.optimizer.load_state_dict(state["optimizerState"])
        self.scaler.load_state_dict(state["scalerState"])
        self.scheduler.load_state_dict(dict(state["domainSchedulerState"]))
        self.global_step = int(state["globalStep"])
        self.optimizer_step = int(state["optimizerStep"])
        if self.global_step < 0 or self.optimizer_step < 0:
            raise ValueError("trainer step counters cannot be negative")
        torch.set_rng_state(state["torchCpuRngState"].cpu())
        if "torchCudaRngState" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torchCudaRngState"])
