"""SocialGraph-FM Core: typed, temporal, multi-domain graph representation."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn

from .types import CoreBatch, CoreModelConfig, CoreOutput


def _mean_by_target(messages: Tensor, target: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    output = messages.new_zeros((num_nodes, messages.shape[-1]))
    counts = messages.new_zeros((num_nodes, 1))
    if messages.shape[0]:
        output.index_add_(0, target, messages)
        counts.index_add_(0, target, messages.new_ones((messages.shape[0], 1)))
    return output / counts.clamp_min(1.0), counts


def causal_edge_age(edge_time: Tensor, cutoff_time: float | Tensor) -> Tensor:
    """Return non-negative edge age and reject a future edge instead of filtering it."""

    cutoff = torch.as_tensor(cutoff_time, dtype=edge_time.dtype, device=edge_time.device)
    if cutoff.numel() != 1 or not bool(torch.isfinite(cutoff).all()):
        raise ValueError("cutoff_time must be one finite scalar")
    flattened = edge_time.reshape(-1)
    if flattened.numel() and bool(torch.any(flattened > cutoff)):
        raise ValueError("message graph contains an edge after the batch cutoff")
    return (cutoff - flattened).clamp_min(0.0)


class Time2Vec(nn.Module):
    """Learnable linear plus periodic encoding for causal edge age."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("Time2Vec requires at least two channels")
        self.linear_weight = nn.Parameter(torch.ones(1))
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.periodic_weight = nn.Parameter(torch.empty(channels - 1))
        self.periodic_bias = nn.Parameter(torch.zeros(channels - 1))
        nn.init.uniform_(self.periodic_weight, -1.0 / math.sqrt(channels), 1.0 / math.sqrt(channels))

    def forward(self, values: Tensor) -> Tensor:
        value = values.reshape(-1, 1)
        linear = value * self.linear_weight + self.linear_bias
        periodic = torch.sin(value * self.periodic_weight + self.periodic_bias)
        return torch.cat((linear, periodic), dim=-1)


class BasisRelationEmbedding(nn.Module):
    """Relation vocabulary represented by a fixed bank of shared bases."""

    def __init__(self, num_relations: int, hidden_channels: int, relation_bases: int) -> None:
        super().__init__()
        self.num_relations = num_relations
        self.relation_bases = relation_bases
        self.bases = nn.Parameter(torch.empty(relation_bases, hidden_channels))
        self.coefficients = nn.Embedding(num_relations, relation_bases)
        nn.init.normal_(self.bases, std=1.0 / math.sqrt(hidden_channels))
        nn.init.normal_(self.coefficients.weight, std=1.0 / math.sqrt(relation_bases))

    def forward(self, relation_type: Tensor) -> Tensor:
        coefficients = torch.softmax(self.coefficients(relation_type), dim=-1)
        return coefficients @ self.bases


class MultimodalProjector(nn.Module):
    """Project typed modalities and fuse visible values with learned missing tokens."""

    def __init__(self, modality_dims: Mapping[str, int], hidden_channels: int) -> None:
        super().__init__()
        self.modalities = tuple(sorted(modality_dims))
        self._keys = {name: f"m{index}" for index, name in enumerate(self.modalities)}
        self.projectors = nn.ModuleDict(
            {
                self._keys[name]: nn.Sequential(
                    nn.Linear(modality_dims[name], hidden_channels),
                    nn.LayerNorm(hidden_channels),
                    nn.GELU(),
                )
                for name in self.modalities
            }
        )
        self.gates = nn.ModuleDict(
            {self._keys[name]: nn.Linear(hidden_channels, 1) for name in self.modalities}
        )
        self.missing_tokens = nn.Parameter(torch.zeros(len(self.modalities), hidden_channels))
        nn.init.normal_(self.missing_tokens, std=0.02)

    def forward(
        self,
        modalities: Mapping[str, Tensor],
        masks: Mapping[str, Tensor],
        *,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        device = next(iter(modalities.values())).device
        values: list[Tensor] = []
        logits: list[Tensor] = []
        for index, name in enumerate(self.modalities):
            key = self._keys[name]
            if name in modalities:
                projected = self.projectors[key](modalities[name])
                visible = masks[name].reshape(-1, 1)
                missing = self.missing_tokens[index].reshape(1, -1).expand(num_nodes, -1)
                value = torch.where(visible, projected, missing)
                penalty = (~visible).to(projected.dtype) * -4.0
            else:
                value = self.missing_tokens[index].reshape(1, -1).expand(num_nodes, -1)
                penalty = torch.full((num_nodes, 1), -4.0, device=device, dtype=value.dtype)
            values.append(value)
            logits.append(self.gates[key](value) + penalty)
        stacked = torch.stack(values, dim=1)
        weights = torch.softmax(torch.cat(logits, dim=1), dim=1)
        return torch.sum(stacked * weights.unsqueeze(-1), dim=1), weights

    def project_modality(self, modality: str, values: Tensor) -> Tensor:
        try:
            return self.projectors[self._keys[modality]](values)
        except KeyError as error:
            raise ValueError(f"unknown projection modality: {modality}") from error


class DomainAdapter(nn.Module):
    """Small residual bottleneck retained per source domain."""

    def __init__(self, domains: tuple[str, ...], hidden_channels: int, bottleneck: int) -> None:
        super().__init__()
        self._keys = {domain: f"d{index}" for index, domain in enumerate(domains)}
        self.adapters = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.LayerNorm(hidden_channels),
                    nn.Linear(hidden_channels, bottleneck),
                    nn.GELU(),
                    nn.Linear(bottleneck, hidden_channels),
                )
                for key in self._keys.values()
            }
        )

    def forward(self, values: Tensor, domain_id: str) -> Tensor:
        try:
            adapter = self.adapters[self._keys[domain_id]]
        except KeyError as error:
            raise ValueError(f"unknown model domain: {domain_id}") from error
        return values + adapter(values)


class SemanticSAGELayer(nn.Module):
    """Mean GraphSAGE whose messages carry relation semantics and causal age."""

    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Linear(hidden_channels * 3, hidden_channels)
        self.self_projection = nn.Linear(hidden_channels, hidden_channels)
        self.neighbor_projection = nn.Linear(hidden_channels, hidden_channels)
        self.normalization = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: Tensor,
        edge_index: Tensor,
        relation_values: Tensor,
        time_values: Tensor,
    ) -> Tensor:
        source, target = edge_index
        if edge_index.shape[1]:
            messages = self.message(torch.cat((values[source], relation_values, time_values), dim=-1))
        else:
            messages = values.new_empty((0, values.shape[-1]))
        neighbors, _ = _mean_by_target(messages, target, values.shape[0])
        updated = self.self_projection(values) + self.neighbor_projection(neighbors)
        return self.normalization(values + self.dropout(torch.relu(updated)))


class TemporalNodeEncoder(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.query_projection = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.key_projection = nn.Linear(hidden_channels * 3, hidden_channels, bias=False)
        self.value_projection = nn.Linear(hidden_channels * 3, hidden_channels, bias=False)
        self.no_event = nn.Parameter(torch.zeros(hidden_channels))
        self.normalization = nn.LayerNorm(hidden_channels)

    def forward(
        self,
        node_values: Tensor,
        edge_index: Tensor,
        relation_values: Tensor,
        time_values: Tensor,
        *,
        num_nodes: int,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        event_values = torch.cat((node_values[source], relation_values, time_values), dim=-1)
        queries = self.query_projection(node_values)
        keys = self.key_projection(event_values)
        values = self.value_projection(event_values)
        scores = torch.sum(queries[target] * keys, dim=-1) / math.sqrt(self.hidden_channels)
        target_max = scores.new_full((num_nodes,), -torch.inf)
        if scores.numel():
            target_max.scatter_reduce_(
                0, target, scores, reduce="amax", include_self=True
            )
        exponentials = torch.exp(scores - target_max[target])
        denominators = scores.new_zeros(num_nodes)
        if scores.numel():
            denominators.index_add_(0, target, exponentials)
        attention = exponentials / denominators[target].clamp_min(
            torch.finfo(exponentials.dtype).tiny
        )
        weighted_values = values * attention.unsqueeze(-1)
        aggregated, counts = _mean_by_target(weighted_values, target, num_nodes)
        # _mean_by_target divides by event count; undo that because attention already sums to one.
        aggregated = aggregated * counts
        fallback = self.no_event.reshape(1, -1).expand(num_nodes, -1)
        encoded = self.normalization(torch.where(counts > 0, aggregated, fallback))
        return encoded, attention


class GatedSemanticTemporalFusion(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_channels * 2, hidden_channels)
        self.normalization = nn.LayerNorm(hidden_channels)

    def forward(self, semantic: Tensor, temporal: Tensor) -> Tensor:
        gate = torch.sigmoid(self.gate(torch.cat((semantic, temporal), dim=-1)))
        return self.normalization(gate * semantic + (1.0 - gate) * temporal)


class TwoExpertNullMoE(nn.Module):
    """Two residual experts plus a zero-valued null expert for negative transfer control."""

    def __init__(self, hidden_channels: int, dropout: float) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_channels, hidden_channels * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_channels * 2, hidden_channels),
                )
                for _ in range(2)
            ]
        )
        self.router = nn.Linear(hidden_channels, 3)
        self.normalization = nn.LayerNorm(hidden_channels)

    def forward(self, base: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax(self.router(base), dim=-1)
        expert_values = torch.stack((self.experts[0](base), self.experts[1](base)), dim=1)
        residual = torch.sum(expert_values * weights[:, :2].unsqueeze(-1), dim=1)
        # weights[:, 2] routes to an explicit zero residual: the shared base remains intact.
        return self.normalization(base + residual), weights


class LinkRankingHead(nn.Module):
    def __init__(self, hidden_channels: int, pair_feature_dim: int, dropout: float) -> None:
        super().__init__()
        input_channels = hidden_channels * 2 + pair_feature_dim
        self.pair_feature_dim = pair_feature_dim
        self.layers = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self, embeddings: Tensor, edge_index: Tensor, pair_features: Tensor | None = None
    ) -> Tensor:
        source, target = edge_index
        source_value, target_value = embeddings[source], embeddings[target]
        pieces = [source_value * target_value, torch.abs(source_value - target_value)]
        if self.pair_feature_dim:
            if pair_features is None or pair_features.shape != (
                edge_index.shape[1],
                self.pair_feature_dim,
            ):
                raise ValueError("the link head requires aligned structural pair features")
            pieces.append(pair_features)
        elif pair_features is not None and pair_features.shape[1]:
            raise ValueError("pair features were supplied to a head configured without them")
        return self.layers(torch.cat(pieces, dim=-1)).reshape(-1)


class RelationHead(nn.Module):
    def __init__(self, hidden_channels: int, num_relations: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_relations),
        )

    def forward(self, embeddings: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        return self.layers(
            torch.cat(
                (embeddings[source] * embeddings[target], torch.abs(embeddings[source] - embeddings[target])),
                dim=-1,
            )
        )


class SocialGraphFMCore(nn.Module):
    """Shared structure-first GFM kernel with isolated domain adapters and task heads."""

    def __init__(self, config: CoreModelConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_channels
        self.projector = MultimodalProjector(config.modality_dims, hidden)
        self.input_domain_adapter = DomainAdapter(
            config.domains, hidden, config.domain_bottleneck
        )
        self.output_domain_adapter = DomainAdapter(
            config.domains, hidden, config.domain_bottleneck
        )
        self.relation_embedding = BasisRelationEmbedding(
            config.num_relations, hidden, config.relation_bases
        )
        self.time2vec = Time2Vec(config.time_channels)
        self.time_projection = nn.Sequential(
            nn.Linear(config.time_channels, hidden), nn.LayerNorm(hidden), nn.GELU()
        )
        self.layers = nn.ModuleList(
            [SemanticSAGELayer(hidden, config.dropout) for _ in range(config.num_layers)]
        )
        self.temporal_encoder = TemporalNodeEncoder(hidden)
        self.fusion = GatedSemanticTemporalFusion(hidden)
        self.moe = (
            TwoExpertNullMoE(hidden, config.dropout) if config.variant == "moe" else None
        )
        self.link_head = LinkRankingHead(hidden, config.pair_feature_dim, config.dropout)
        self.relation_head = RelationHead(hidden, config.num_relations)
        self.time_delta_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.attribute_decoders = nn.ModuleDict(
            {
                f"m{index}": nn.Linear(hidden, config.modality_dims[name])
                for index, name in enumerate(sorted(config.modality_dims))
            }
        )
        self._modality_keys = {
            name: f"m{index}" for index, name in enumerate(sorted(config.modality_dims))
        }
        self.node_head = (
            nn.Linear(hidden, config.node_class_count) if config.node_class_count is not None else None
        )
        self.graph_head = nn.Linear(hidden, config.graph_output_channels)

    def forward(self, batch: CoreBatch) -> CoreOutput:
        batch.validate(self.config)
        num_nodes = batch.num_nodes
        projected, modality_weights = self.projector(
            batch.modalities, batch.modality_masks, num_nodes=num_nodes
        )
        base = self.input_domain_adapter(projected, batch.domain_id)
        ages = causal_edge_age(batch.edge_time, batch.cutoff_time)
        time_values = self.time_projection(self.time2vec(ages))
        relation_values = self.relation_embedding(batch.edge_type.reshape(-1))
        semantic = base
        for layer in self.layers:
            semantic = layer(semantic, batch.edge_index, relation_values, time_values)
        temporal, temporal_attention = self.temporal_encoder(
            base,
            batch.edge_index,
            relation_values,
            time_values,
            num_nodes=num_nodes,
        )
        fused = self.fusion(semantic, temporal)
        adapted = self.output_domain_adapter(fused, batch.domain_id)
        if self.moe is None:
            embeddings = adapted
            expert_weights = adapted.new_zeros((num_nodes, 3))
            expert_weights[:, 2] = 1.0
        else:
            embeddings, expert_weights = self.moe(adapted)
        return CoreOutput(
            node_embeddings=embeddings,
            base_embeddings=base,
            semantic_embeddings=semantic,
            temporal_embeddings=temporal,
            temporal_attention_weights=temporal_attention,
            modality_weights=modality_weights,
            expert_weights=expert_weights,
        )

    def reconstruct_modality(self, embeddings: Tensor, modality: str) -> Tensor:
        try:
            return self.attribute_decoders[self._modality_keys[modality]](embeddings)
        except KeyError as error:
            raise ValueError(f"unknown reconstruction modality: {modality}") from error

    def score_links(
        self,
        embeddings: Tensor,
        edge_index: Tensor,
        pair_features: Tensor | None = None,
    ) -> Tensor:
        return self.link_head(embeddings, edge_index, pair_features)

    def classify_relations(self, embeddings: Tensor, edge_index: Tensor) -> Tensor:
        return self.relation_head(embeddings, edge_index)

    def predict_log_time_delta(self, embeddings: Tensor, edge_index: Tensor) -> Tensor:
        source, target = edge_index
        pair = torch.cat(
            (embeddings[source] * embeddings[target], torch.abs(embeddings[source] - embeddings[target])),
            dim=-1,
        )
        return self.time_delta_head(pair).reshape(-1)

    def project_modality(self, modality: str, values: Tensor) -> Tensor:
        return self.projector.project_modality(modality, values)

    def classify_nodes(self, embeddings: Tensor) -> Tensor:
        if self.node_head is None:
            raise RuntimeError("the model was configured without a node classification head")
        return self.node_head(embeddings)

    def predict_graph(self, embeddings: Tensor, batch_index: Tensor | None = None) -> Tensor:
        if batch_index is None:
            pooled = embeddings.mean(dim=0, keepdim=True)
        else:
            if batch_index.dtype != torch.long or batch_index.shape[0] != embeddings.shape[0]:
                raise ValueError("batch_index must assign every node to a graph")
            graph_count = int(batch_index.max()) + 1 if batch_index.numel() else 0
            pooled, _ = _mean_by_target(embeddings, batch_index, graph_count)
        return self.graph_head(pooled)
