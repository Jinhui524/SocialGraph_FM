"""Tensor-native contracts for the SocialGraph-FM Core training kernel.

These types intentionally stay independent of public serving contracts.  They
describe already validated, point-in-time graph batches produced by corpus
adapters and are small enough to use in CPU/CUDA unit tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Mapping

import torch
from torch import Tensor


def _check_edge_index(value: Tensor, *, name: str, num_nodes: int) -> None:
    if value.dtype != torch.long or value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(f"{name} must be a torch.long tensor with shape [2, E]")
    if value.numel() and (int(value.min()) < 0 or int(value.max()) >= num_nodes):
        raise ValueError(f"{name} contains an out-of-range node index")


def _node_mask(value: Tensor, *, name: str, num_nodes: int) -> Tensor:
    if value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean")
    flattened = value.reshape(-1)
    if flattened.shape[0] != num_nodes:
        raise ValueError(f"{name} must contain one value per node")
    return flattened


@dataclass(frozen=True)
class CoreModelConfig:
    """Shape and architecture identity for one shared SocialGraph-FM Core model."""

    modality_dims: Mapping[str, int]
    domains: tuple[str, ...]
    num_relations: int
    variant: Literal["base", "moe"]
    hidden_channels: int = 128
    num_layers: int = 2
    time_channels: int = 32
    relation_bases: int = 8
    domain_bottleneck: int = 32
    expert_count: int = 2
    dropout: float = 0.2
    pair_feature_dim: int = 0
    text_modality: str | None = None
    node_class_count: int | None = None
    graph_output_channels: int = 1

    def __post_init__(self) -> None:
        if not self.modality_dims or any(not key or value < 1 for key, value in self.modality_dims.items()):
            raise ValueError("modality_dims must contain positive dimensions")
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("domains must be nonempty and unique")
        if self.num_relations < 1:
            raise ValueError("num_relations must be positive")
        if self.hidden_channels < 4 or self.num_layers < 1:
            raise ValueError("the shared encoder requires positive model capacity")
        if self.time_channels < 2 or self.domain_bottleneck < 1:
            raise ValueError("time and domain adapter dimensions are invalid")
        if self.relation_bases != 8:
            raise ValueError("SocialGraph-FM Core fixes relation_bases=8")
        if self.expert_count != 2:
            raise ValueError("SocialGraph-FM Core fixes two experts plus one null expert")
        if self.variant not in ("base", "moe"):
            raise ValueError("variant must be 'base' or 'moe'")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.pair_feature_dim < 0:
            raise ValueError("pair_feature_dim cannot be negative")
        if self.text_modality is not None and self.text_modality not in self.modality_dims:
            raise ValueError("text_modality must name a configured modality")
        if self.node_class_count is not None and self.node_class_count < 2:
            raise ValueError("node_class_count must be at least two")
        if self.graph_output_channels < 1:
            raise ValueError("graph_output_channels must be positive")


@dataclass(frozen=True)
class CoreSampleProvenance:
    """Portable causal identity carried by every pretraining/adaptation batch."""

    domain_id: str
    graph_version: str
    cutoff: float
    horizon: float
    task_id: str
    source_corpus_hash: str

    def validate(self) -> None:
        if not self.domain_id or not self.graph_version or not self.task_id:
            raise ValueError("core sample provenance requires domain, graph and task identity")
        if (
            not math.isfinite(self.cutoff)
            or not math.isfinite(self.horizon)
            or self.horizon <= 0.0
        ):
            raise ValueError("core sample provenance cutoff/horizon is invalid")
        if len(self.source_corpus_hash) != 64 or any(
            value not in "0123456789abcdef" for value in self.source_corpus_hash
        ):
            raise ValueError("core source_corpus_hash must be a lowercase SHA-256")


@dataclass(frozen=True)
class CoreBatch:
    """One homogeneous training subgraph from a named source domain.

    ``modality_masks`` uses ``True`` for a value visible at the batch cutoff.
    ``attribute_masks`` uses ``True`` for nodes selected as reconstruction
    targets.  Adapters must supply only edges whose timestamp is no later than
    ``cutoff_time``; the model and sampler both fail closed if that is violated.
    """

    domain_id: str
    modalities: Mapping[str, Tensor]
    modality_masks: Mapping[str, Tensor]
    edge_index: Tensor
    edge_type: Tensor
    edge_time: Tensor
    cutoff_time: float | Tensor
    provenance: CoreSampleProvenance
    attribute_targets: Mapping[str, Tensor] = field(default_factory=dict)
    attribute_masks: Mapping[str, Tensor] = field(default_factory=dict)
    positive_edge_index: Tensor | None = None
    negative_edge_index: Tensor | None = None
    positive_relation: Tensor | None = None
    positive_relation_mask: Tensor | None = None
    time_delta_targets: Tensor | None = None
    time_delta_mask: Tensor | None = None
    positive_pair_features: Tensor | None = None
    negative_pair_features: Tensor | None = None

    @property
    def num_nodes(self) -> int:
        first = next(iter(self.modalities.values()), None)
        return int(first.shape[0]) if first is not None else 0

    def validate(self, config: CoreModelConfig | None = None) -> None:
        self.provenance.validate()
        if not self.domain_id:
            raise ValueError("domain_id must be nonempty")
        if self.provenance.domain_id != self.domain_id:
            raise ValueError("core batch domain differs from its provenance")
        if not self.modalities:
            raise ValueError("a core batch requires at least one modality")
        num_nodes = self.num_nodes
        if num_nodes < 1:
            raise ValueError("a core batch requires at least one node")
        for name, value in self.modalities.items():
            if value.ndim != 2 or value.shape[0] != num_nodes or not value.is_floating_point():
                raise ValueError(f"modality {name!r} must be floating point [N, D]")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"modality {name!r} contains NaN or Infinity")
            mask = self.modality_masks.get(name)
            if mask is None:
                raise ValueError(f"modality {name!r} is missing its visibility mask")
            _node_mask(mask, name=f"modality_masks[{name!r}]", num_nodes=num_nodes)
        if set(self.modality_masks).difference(self.modalities):
            raise ValueError("modality_masks contains an undeclared modality")

        _check_edge_index(self.edge_index, name="edge_index", num_nodes=num_nodes)
        edge_count = int(self.edge_index.shape[1])
        if self.edge_type.dtype != torch.long or self.edge_type.reshape(-1).shape[0] != edge_count:
            raise ValueError("edge_type must be torch.long with one value per edge")
        if self.edge_time.reshape(-1).shape[0] != edge_count or not self.edge_time.is_floating_point():
            raise ValueError("edge_time must be floating point with one value per edge")
        if not bool(torch.isfinite(self.edge_time).all()):
            raise ValueError("edge_time contains NaN or Infinity")
        if isinstance(self.cutoff_time, Tensor) and self.cutoff_time.numel() != 1:
            raise ValueError("core batch cutoff must be scalar")
        cutoff_value = (
            float(self.cutoff_time.detach().cpu())
            if isinstance(self.cutoff_time, Tensor)
            else float(self.cutoff_time)
        )
        if not math.isfinite(cutoff_value) or cutoff_value != self.provenance.cutoff:
            raise ValueError("core batch cutoff differs from its provenance")
        if self.edge_time.numel() and bool((self.edge_time > cutoff_value).any()):
            raise ValueError("core batch contains an edge after the batch cutoff")

        for name, target in self.attribute_targets.items():
            source = self.modalities.get(name)
            mask = self.attribute_masks.get(name)
            if source is None or target.shape != source.shape or not target.is_floating_point():
                raise ValueError(f"attribute target {name!r} must match its modality shape")
            if not bool(torch.isfinite(target).all()):
                raise ValueError(f"attribute target {name!r} contains NaN or Infinity")
            if mask is None:
                raise ValueError(f"attribute target {name!r} requires a reconstruction mask")
            _node_mask(mask, name=f"attribute_masks[{name!r}]", num_nodes=num_nodes)
        if set(self.attribute_masks).difference(self.attribute_targets):
            raise ValueError("attribute_masks contains a target without values")

        for name in ("positive_edge_index", "negative_edge_index"):
            value = getattr(self, name)
            if value is not None:
                _check_edge_index(value, name=name, num_nodes=num_nodes)
        if self.positive_edge_index is not None and self.negative_edge_index is not None:
            positive_count = int(self.positive_edge_index.shape[1])
            negative_count = int(self.negative_edge_index.shape[1])
            if positive_count < 1 or negative_count < 1 or negative_count % positive_count:
                raise ValueError(
                    "negative edges must contain an equal positive-aligned number per query"
                )
        if self.positive_relation is not None:
            if self.positive_edge_index is None:
                raise ValueError("positive_relation requires positive_edge_index")
            if self.positive_relation.dtype != torch.long or (
                self.positive_relation.reshape(-1).shape[0] != self.positive_edge_index.shape[1]
            ):
                raise ValueError("positive_relation must align with positive edges")
            if self.positive_relation_mask is not None and (
                self.positive_relation_mask.dtype != torch.bool
                or self.positive_relation_mask.reshape(-1).shape
                != self.positive_relation.reshape(-1).shape
            ):
                raise ValueError("positive_relation_mask must align with relation labels")
        elif self.positive_relation_mask is not None:
            raise ValueError("positive_relation_mask requires positive_relation")
        if self.time_delta_targets is not None:
            if self.positive_edge_index is None:
                raise ValueError("time_delta_targets requires positive_edge_index")
            targets = self.time_delta_targets.reshape(-1)
            if (
                not self.time_delta_targets.is_floating_point()
                or targets.shape[0] != self.positive_edge_index.shape[1]
                or not bool(torch.isfinite(targets).all())
                or bool((targets < 0).any())
            ):
                raise ValueError("time_delta_targets must be finite, non-negative and aligned")
            if self.time_delta_mask is None or (
                self.time_delta_mask.dtype != torch.bool
                or self.time_delta_mask.reshape(-1).shape != targets.shape
            ):
                raise ValueError("time_delta_mask must align with time_delta_targets")
        elif self.time_delta_mask is not None:
            raise ValueError("time_delta_mask requires time_delta_targets")
        for edges_name, features_name in (
            ("positive_edge_index", "positive_pair_features"),
            ("negative_edge_index", "negative_pair_features"),
        ):
            edges = getattr(self, edges_name)
            features = getattr(self, features_name)
            if features is not None and (
                edges is None or features.ndim != 2 or features.shape[0] != edges.shape[1]
            ):
                raise ValueError(f"{features_name} must align with {edges_name}")

        if config is not None:
            if self.domain_id not in config.domains:
                raise ValueError(f"unknown training domain: {self.domain_id}")
            if set(self.modalities).difference(config.modality_dims):
                raise ValueError("batch contains a modality unknown to the model")
            for name, value in self.modalities.items():
                if value.shape[1] != config.modality_dims[name]:
                    raise ValueError(f"modality {name!r} has the wrong feature dimension")
            if self.edge_type.numel() and (
                int(self.edge_type.min()) < 0 or int(self.edge_type.max()) >= config.num_relations
            ):
                raise ValueError("edge_type is outside the configured relation vocabulary")
            for features in (self.positive_pair_features, self.negative_pair_features):
                if features is not None and features.shape[1] != config.pair_feature_dim:
                    raise ValueError("pair feature width differs from the model config")

    def to(self, device: str | torch.device) -> CoreBatch:
        def moved(values: Mapping[str, Tensor]) -> dict[str, Tensor]:
            return {name: value.to(device) for name, value in values.items()}

        def optional(value: Tensor | None) -> Tensor | None:
            return value.to(device) if value is not None else None

        cutoff = self.cutoff_time.to(device) if isinstance(self.cutoff_time, Tensor) else self.cutoff_time
        return CoreBatch(
            domain_id=self.domain_id,
            modalities=moved(self.modalities),
            modality_masks=moved(self.modality_masks),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_time=self.edge_time.to(device),
            cutoff_time=cutoff,
            provenance=self.provenance,
            attribute_targets=moved(self.attribute_targets),
            attribute_masks=moved(self.attribute_masks),
            positive_edge_index=optional(self.positive_edge_index),
            negative_edge_index=optional(self.negative_edge_index),
            positive_relation=optional(self.positive_relation),
            positive_relation_mask=optional(self.positive_relation_mask),
            time_delta_targets=optional(self.time_delta_targets),
            time_delta_mask=optional(self.time_delta_mask),
            positive_pair_features=optional(self.positive_pair_features),
            negative_pair_features=optional(self.negative_pair_features),
        )


@dataclass(frozen=True)
class CoreOutput:
    node_embeddings: Tensor
    base_embeddings: Tensor
    semantic_embeddings: Tensor
    temporal_embeddings: Tensor
    temporal_attention_weights: Tensor
    modality_weights: Tensor
    expert_weights: Tensor
