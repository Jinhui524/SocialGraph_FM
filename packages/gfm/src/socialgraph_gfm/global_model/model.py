"""SocialGraph-FM Global cross-modal GraphSAGE with sparse routing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch_geometric.nn import SAGEConv

from .contracts import COUNTRY_IDS, GRAPH_STAT_NAMES

GRAPH_STATS_DIM = len(GRAPH_STAT_NAMES)


@dataclass(frozen=True)
class GlobalModelConfig:
    text_dim: int = 768
    structural_dim: int = 128
    branch_dim: int = 128
    hidden_dim: int = 256
    dropout: float = 0.2
    domains: tuple[str, ...] = COUNTRY_IDS
    router_enabled: bool = True
    router_bottleneck_dim: int = 64
    router_top_k: int = 2

    def __post_init__(self) -> None:
        if (
            self.text_dim != 768
            or self.structural_dim != 128
            or self.branch_dim != 128
            or self.hidden_dim != 256
        ):
            raise ValueError(
                "SocialGraph-FM Global requires text=768, structure=128, branch=128, and hidden=256"
            )
        if self.router_bottleneck_dim < 4:
            raise ValueError("router bottleneck dimension is too small")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if not self.domains or len(self.domains) != len(set(self.domains)):
            raise ValueError("domains must be nonempty and unique")
        if self.router_top_k != 2:
            raise ValueError("SocialGraph-FM Global uses an exact top-2 router")


@dataclass(frozen=True)
class GlobalBackboneOutput:
    node_embeddings: Tensor
    fused_features: Tensor
    modality_contributions: Tensor


@dataclass(frozen=True)
class RouterOutput:
    embeddings: Tensor
    weights: Tensor
    indices: Tensor
    expert_names: tuple[str, ...]


@dataclass(frozen=True)
class GlobalOutput:
    logits: Tensor
    node_embeddings: Tensor
    fused_features: Tensor
    modality_contributions: Tensor
    router_weights: Tensor | None
    router_indices: Tensor | None
    expert_names: tuple[str, ...]


def degree_bucket_one_hot(
    degree_bucket: Tensor,
    *,
    bucket_count: int = 128,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Expand compact percentile bucket IDs for the official structural modality."""

    if degree_bucket.ndim != 1 or degree_bucket.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("degree_bucket must be a rank-one integer tensor")
    values = degree_bucket.to(torch.long)
    if values.numel() and (
        bool((values < 0).any()) or bool((values >= bucket_count).any())
    ):
        raise ValueError("degree_bucket contains an out-of-range value")
    return functional.one_hot(values, num_classes=bucket_count).to(dtype=dtype)


def _validate_graph_inputs(
    text_features: Tensor,
    structural_features: Tensor,
    edge_index: Tensor,
    config: GlobalModelConfig,
) -> Tensor:
    if (
        text_features.ndim != 2
        or text_features.shape[1] != config.text_dim
        or not text_features.is_floating_point()
    ):
        raise ValueError(f"text_features must be floating [N,{config.text_dim}]")
    if structural_features.ndim == 1:
        structural = degree_bucket_one_hot(
            structural_features,
            bucket_count=config.structural_dim,
            dtype=text_features.dtype,
        )
    elif (
        structural_features.ndim == 2
        and structural_features.shape[1] == config.structural_dim
        and structural_features.is_floating_point()
    ):
        structural = structural_features
    else:
        raise ValueError(
            f"structural_features must be integer [N] or floating [N,{config.structural_dim}]"
        )
    if structural.shape[0] != text_features.shape[0]:
        raise ValueError("text and structural features must describe the same sampled nodes")
    if structural.device != text_features.device:
        raise ValueError("text and structural features must be on the same device")
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be torch.long with shape [2,E]")
    if edge_index.device != text_features.device:
        raise ValueError("edge_index and node features must be on the same device")
    if edge_index.numel() and (
        bool((edge_index < 0).any())
        or bool((edge_index >= text_features.shape[0]).any())
    ):
        raise ValueError("edge_index references a node outside the sampled batch")
    if not bool(torch.isfinite(text_features).all()) or not bool(torch.isfinite(structural).all()):
        raise ValueError("model inputs must be finite")
    return structural


class GlobalCrossModalBackbone(nn.Module):
    """Exact Global cross-gating/fusion followed by a two-layer GraphSAGE encoder."""

    def __init__(self, config: GlobalModelConfig) -> None:
        super().__init__()
        branch = config.branch_dim
        self.config = config
        self.cross_attention_to_text = nn.Linear(config.structural_dim, branch)
        self.cross_attention_to_struct = nn.Linear(config.text_dim, branch)
        self.struct_projector = nn.Sequential(nn.Linear(config.structural_dim, branch), nn.ReLU())
        self.text_projector = nn.Sequential(nn.Linear(config.text_dim, branch), nn.ReLU())
        self.joint_projector = nn.Sequential(
            nn.Linear(branch * 2, config.hidden_dim), nn.ReLU()
        )
        self.conv1 = SAGEConv(config.hidden_dim, config.hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(config.hidden_dim, config.hidden_dim, aggr="mean")
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        text_features: Tensor,
        structural_features: Tensor,
        edge_index: Tensor,
    ) -> GlobalBackboneOutput:
        structural = _validate_graph_inputs(
            text_features, structural_features, edge_index, self.config
        )
        structural_branch = self.struct_projector(structural) * functional.relu(
            self.cross_attention_to_struct(text_features)
        )
        text_branch = self.text_projector(text_features) * functional.relu(
            self.cross_attention_to_text(structural)
        )
        fused = self.joint_projector(torch.cat((structural_branch, text_branch), dim=-1))
        hidden = self.dropout(functional.relu(self.conv1(fused, edge_index)))
        embeddings = functional.relu(self.conv2(hidden, edge_index))

        text_norm = torch.linalg.vector_norm(text_branch, dim=-1)
        structural_norm = torch.linalg.vector_norm(structural_branch, dim=-1)
        total_norm = text_norm + structural_norm
        denominator = total_norm.clamp_min(torch.finfo(fused.dtype).eps)
        contributions = torch.stack((text_norm / denominator, structural_norm / denominator), dim=-1)
        contributions = torch.where(
            (total_norm == 0).unsqueeze(-1),
            torch.full_like(contributions, 0.5),
            contributions,
        )
        return GlobalBackboneOutput(
            node_embeddings=embeddings,
            fused_features=fused,
            modality_contributions=contributions,
        )


class _ResidualExpert(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float,
        *,
        zero_initialize_output: bool = False,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        if zero_initialize_output:
            output = self.network[-1]
            assert isinstance(output, nn.Linear)
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class _NullExpert(nn.Module):
    def forward(self, values: Tensor) -> Tensor:
        return torch.zeros_like(values)


class SparseTop2Router(nn.Module):
    """Sparse shared/domain/null residual experts with exactly two active routes."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        domains: Sequence[str],
        bottleneck_dim: int = 64,
        dropout: float = 0.2,
        top_k: int = 2,
    ) -> None:
        super().__init__()
        if top_k != 2:
            raise ValueError("SparseTop2Router requires top_k=2")
        if not domains or len(domains) != len(set(domains)):
            raise ValueError("router domains must be nonempty and unique")
        self.domains = tuple(domains)
        self.domain_to_index = {name: index for index, name in enumerate(self.domains)}
        self.expert_names = ("shared", *(f"domain:{name}" for name in self.domains), "null")
        self.top_k = top_k
        candidate_count = len(self.expert_names) - 1
        self.gate = nn.Linear(hidden_dim + GRAPH_STATS_DIM, candidate_count)
        self.domain_bias = nn.Embedding(len(self.domains), candidate_count)
        self.experts = nn.ModuleList(
            [
                _ResidualExpert(hidden_dim, bottleneck_dim, dropout),
                *(
                    _ResidualExpert(
                        hidden_dim,
                        bottleneck_dim,
                        dropout,
                        zero_initialize_output=True,
                    )
                    for _ in self.domains
                ),
                _NullExpert(),
            ]
        )
        nn.init.zeros_(self.domain_bias.weight)

    def _domain_indices(
        self,
        domain_id: str | int | Tensor | None,
        *,
        rows: int,
        device: torch.device,
    ) -> Tensor | None:
        if domain_id is None:
            return None
        if isinstance(domain_id, str):
            try:
                index = self.domain_to_index[domain_id]
            except KeyError as exc:
                raise ValueError(f"unknown router domain {domain_id!r}") from exc
            return torch.full((rows,), index, dtype=torch.long, device=device)
        if isinstance(domain_id, int):
            if isinstance(domain_id, bool) or not 0 <= domain_id < len(self.domains):
                raise ValueError("router domain index is out of range")
            return torch.full((rows,), domain_id, dtype=torch.long, device=device)
        if not isinstance(domain_id, Tensor) or domain_id.dtype != torch.long:
            raise ValueError("domain_id must be a domain name, integer, long Tensor, or None")
        values = domain_id.to(device=device)
        if values.ndim == 0:
            values = values.expand(rows)
        if values.shape != (rows,):
            raise ValueError("domain_id tensor must be scalar or have one value per sampled node")
        if values.numel() and (
            bool((values < 0).any()) or bool((values >= len(self.domains)).any())
        ):
            raise ValueError("domain_id tensor contains an out-of-range index")
        return values

    def _graph_statistics(
        self,
        graph_stats: Tensor | None,
        *,
        rows: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if graph_stats is None:
            return torch.zeros((rows, GRAPH_STATS_DIM), device=device, dtype=dtype)
        if not isinstance(graph_stats, Tensor) or not graph_stats.is_floating_point():
            raise ValueError("graph_stats must be a floating Tensor")
        values = graph_stats.to(device=device, dtype=dtype)
        if values.shape == (GRAPH_STATS_DIM,):
            values = values.unsqueeze(0).expand(rows, -1)
        if values.shape != (rows, GRAPH_STATS_DIM):
            raise ValueError(f"graph_stats must have shape [{GRAPH_STATS_DIM}] or [N,{GRAPH_STATS_DIM}]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("graph_stats must be finite")
        return values

    def _allowed_mask(
        self,
        allowed_experts: Sequence[str | int] | Tensor | None,
        *,
        device: torch.device,
    ) -> Tensor:
        candidate_count = len(self.expert_names) - 1
        if allowed_experts is None:
            return torch.ones(candidate_count, dtype=torch.bool, device=device)
        if isinstance(allowed_experts, Tensor):
            if allowed_experts.dtype == torch.bool and allowed_experts.shape == (candidate_count,):
                mask = allowed_experts.to(device=device)
            elif allowed_experts.dtype == torch.long and allowed_experts.ndim == 1:
                mask = torch.zeros(candidate_count, dtype=torch.bool, device=device)
                catalog_indices = allowed_experts.to(device=device)
                if catalog_indices.numel() and (
                    bool((catalog_indices < 1).any())
                    or bool((catalog_indices > candidate_count).any())
                ):
                    raise ValueError("allowed expert catalog index is out of range")
                mask[catalog_indices - 1] = True
            else:
                raise ValueError("allowed_experts Tensor must be bool [7] or long catalog indices")
        else:
            mask = torch.zeros(candidate_count, dtype=torch.bool, device=device)
            for item in allowed_experts:
                if isinstance(item, str):
                    normalized = f"domain:{item}" if item in self.domain_to_index else item
                    if normalized == "shared":
                        raise ValueError("shared is always applied and must not enter allowed_experts")
                    try:
                        catalog_index = self.expert_names.index(normalized)
                    except ValueError as exc:
                        raise ValueError(f"unknown allowed expert {item!r}") from exc
                elif isinstance(item, int) and not isinstance(item, bool):
                    catalog_index = item
                else:
                    raise TypeError("allowed_experts values must be names or catalog indices")
                if not 1 <= catalog_index <= candidate_count:
                    raise ValueError("allowed expert catalog index is out of range")
                mask[catalog_index - 1] = True
        if int(mask.sum()) < self.top_k:
            raise ValueError("allowed_experts must contain at least two routed candidates")
        return mask

    def forward(
        self,
        embeddings: Tensor,
        *,
        domain_id: str | int | Tensor | None = None,
        graph_stats: Tensor | None = None,
        allowed_experts: Sequence[str | int] | Tensor | None = None,
    ) -> RouterOutput:
        if embeddings.ndim != 2 or not embeddings.is_floating_point():
            raise ValueError("router embeddings must be a floating rank-two tensor")
        statistics = self._graph_statistics(
            graph_stats,
            rows=embeddings.shape[0],
            device=embeddings.device,
            dtype=embeddings.dtype,
        )
        logits = self.gate(torch.cat((embeddings, statistics), dim=-1))
        domain_indices = self._domain_indices(
            domain_id, rows=embeddings.shape[0], device=embeddings.device
        )
        if domain_indices is not None:
            logits = logits + self.domain_bias(domain_indices)
        allowed_mask = self._allowed_mask(allowed_experts, device=embeddings.device)
        logits = logits.masked_fill(~allowed_mask.unsqueeze(0), float("-inf"))
        top_logits, candidate_indices = torch.topk(logits, k=self.top_k, dim=-1, sorted=True)
        top_indices = candidate_indices + 1
        top_weights = functional.softmax(top_logits, dim=-1)

        residual = torch.zeros_like(embeddings)
        for expert_index, expert in enumerate(self.experts[1:], start=1):
            selections = torch.nonzero(top_indices == expert_index, as_tuple=False)
            if selections.numel() == 0:
                continue
            selected_rows = selections[:, 0]
            selected_slots = selections[:, 1]
            selected = embeddings.index_select(0, selected_rows)
            expert_residual = expert(selected)
            routing_weight = top_weights[selected_rows, selected_slots].to(
                dtype=expert_residual.dtype
            )
            scaled = (expert_residual * routing_weight.unsqueeze(-1)).to(
                dtype=residual.dtype
            )
            residual.index_add_(0, selected_rows, scaled)
        shared_residual = self.experts[0](embeddings).to(dtype=embeddings.dtype)
        return RouterOutput(
            embeddings=embeddings + shared_residual + residual,
            weights=top_weights,
            indices=top_indices,
            expert_names=self.expert_names,
        )


def router_load_balancing_loss(
    weights: Tensor,
    indices: Tensor,
    *,
    expert_count: int,
) -> Tensor:
    """Importance-balancing penalty for sparse top-k routes (minimum is one)."""

    if (
        weights.ndim != 2
        or weights.shape[1] != 2
        or not weights.is_floating_point()
        or indices.shape != weights.shape
        or indices.dtype != torch.long
    ):
        raise ValueError("router weights/indices must be aligned floating/long [N,2]")
    if expert_count < 4 or (
        indices.numel()
        and (bool((indices < 1).any()) or bool((indices >= expert_count).any()))
    ):
        raise ValueError("expert_count or router index is invalid")
    if weights.shape[0] == 0:
        return weights.sum()
    candidate_count = expert_count - 1
    dense = torch.zeros(
        (weights.shape[0], candidate_count), dtype=weights.dtype, device=weights.device
    ).scatter(1, indices - 1, weights)
    importance = dense.mean(dim=0)
    return candidate_count * torch.sum(importance.square())


class GlobalModel(nn.Module):
    def __init__(self, config: GlobalModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or GlobalModelConfig()
        self.backbone = GlobalCrossModalBackbone(self.config)
        self.router = (
            SparseTop2Router(
                hidden_dim=self.config.hidden_dim,
                domains=self.config.domains,
                bottleneck_dim=self.config.router_bottleneck_dim,
                dropout=self.config.dropout,
                top_k=self.config.router_top_k,
            )
            if self.config.router_enabled
            else None
        )
        self.node_head = nn.Linear(self.config.hidden_dim, 1)

    def forward(
        self,
        text_features: Tensor,
        structural_features: Tensor,
        edge_index: Tensor,
        *,
        domain_id: str | int | Tensor | None = None,
        graph_stats: Tensor | None = None,
        allowed_experts: Sequence[str | int] | Tensor | None = None,
    ) -> GlobalOutput:
        backbone = self.backbone(text_features, structural_features, edge_index)
        if self.router is None:
            embeddings = backbone.node_embeddings
            router_weights = None
            router_indices = None
            expert_names: tuple[str, ...] = ()
        else:
            routed = self.router(
                backbone.node_embeddings,
                domain_id=domain_id,
                graph_stats=graph_stats,
                allowed_experts=allowed_experts,
            )
            embeddings = routed.embeddings
            router_weights = routed.weights
            router_indices = routed.indices
            expert_names = routed.expert_names
        logits = self.node_head(embeddings).reshape(-1)
        return GlobalOutput(
            logits=logits,
            node_embeddings=embeddings,
            fused_features=backbone.fused_features,
            modality_contributions=backbone.modality_contributions,
            router_weights=router_weights,
            router_indices=router_indices,
            expert_names=expert_names,
        )


__all__ = [
    "GRAPH_STATS_DIM",
    "GlobalBackboneOutput",
    "GlobalCrossModalBackbone",
    "GlobalModel",
    "GlobalModelConfig",
    "GlobalOutput",
    "RouterOutput",
    "SparseTop2Router",
    "degree_bucket_one_hot",
    "router_load_balancing_loss",
]
