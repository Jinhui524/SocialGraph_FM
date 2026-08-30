"""Small real baselines for the core experiment ladder."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch
from torch import Tensor, nn


def _node_index(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer node index")
    return int(value)


def _positive_dimension(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _neighbors(*, num_nodes: int, edges: Iterable[tuple[int, int]]) -> tuple[frozenset[int], ...]:
    if type(num_nodes) is not int or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    rows: list[set[int]] = [set() for _ in range(num_nodes)]
    seen: set[tuple[int, int]] = set()
    for raw_left, raw_right in edges:
        left = _node_index(raw_left, label="baseline edge endpoint")
        right = _node_index(raw_right, label="baseline edge endpoint")
        if left < 0 or right < 0 or left >= num_nodes or right >= num_nodes:
            raise ValueError("baseline edge endpoint is outside the node inventory")
        if left == right:
            raise ValueError("baseline edges must not contain self loops")
        pair = (min(left, right), max(left, right))
        if pair in seen:
            raise ValueError("baseline edges must be unique undirected pairs")
        seen.add(pair)
        rows[left].add(right)
        rows[right].add(left)
    return tuple(frozenset(row) for row in rows)


def _candidate_pairs(
    candidates: Sequence[tuple[int, int]], *, num_nodes: int
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for raw_left, raw_right in candidates:
        left = _node_index(raw_left, label="candidate endpoint")
        right = _node_index(raw_right, label="candidate endpoint")
        if left < 0 or right < 0 or left >= num_nodes or right >= num_nodes:
            raise ValueError("candidate endpoint is outside the node inventory")
        if left == right:
            raise ValueError("candidate endpoints must be distinct")
        normalized.append((left, right))
    if not normalized:
        raise ValueError("candidate inventory must be nonempty")
    return tuple(normalized)


def common_neighbors_scores(
    *,
    num_nodes: int,
    edges: Iterable[tuple[int, int]],
    candidates: Sequence[tuple[int, int]],
) -> tuple[float, ...]:
    neighbors = _neighbors(num_nodes=num_nodes, edges=edges)
    return tuple(
        float(len(neighbors[left] & neighbors[right]))
        for left, right in _candidate_pairs(candidates, num_nodes=num_nodes)
    )


def adamic_adar_scores(
    *,
    num_nodes: int,
    edges: Iterable[tuple[int, int]],
    candidates: Sequence[tuple[int, int]],
) -> tuple[float, ...]:
    neighbors = _neighbors(num_nodes=num_nodes, edges=edges)
    result: list[float] = []
    for left, right in _candidate_pairs(candidates, num_nodes=num_nodes):
        result.append(
            math.fsum(
                1.0 / math.log(len(neighbors[common]))
                for common in neighbors[left] & neighbors[right]
                if len(neighbors[common]) > 1
            )
        )
    return tuple(result)


class _InputMlp(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int, label: str) -> None:
        super().__init__()
        self.input_dim = _positive_dimension(input_dim, label="MLP input dimension")
        normalized_output_dim = _positive_dimension(output_dim, label="MLP output dimension")
        self.label = label
        hidden = min(128, max(16, self.input_dim * 2))
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, normalized_output_dim),
        )

    def _project(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"{self.label} width does not match its baseline contract")
        if values.layout != torch.strided or not values.is_floating_point():
            raise ValueError(f"{self.label} values must be dense floating point tensors")
        if not bool(torch.isfinite(values).all()):
            raise ValueError(f"{self.label} values must be finite")
        return self.network(values)


class FeatureMlp(_InputMlp):
    def __init__(self, *, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim=input_dim, output_dim=output_dim, label="feature")

    def forward(self, *, attributes: Tensor) -> Tensor:
        return self._project(attributes)


class StructureMlp(_InputMlp):
    def __init__(self, *, structure_dim: int, output_dim: int) -> None:
        super().__init__(input_dim=structure_dim, output_dim=output_dim, label="structure")

    def forward(self, *, structure: Tensor) -> Tensor:
        return self._project(structure)


GfmMethodId = Literal[
    "graphsage-scratch",
    "graphmae2-single",
    "multi-graph-shared-gfm",
    "domain-aligned-gfm",
]


_GFM_FAMILY_DEFINITIONS: dict[
    GfmMethodId,
    tuple[
        Literal["random", "pretrained"],
        Literal["none", "single-source", "multi-source"],
        float,
        float,
        tuple[float, ...],
        bool,
    ],
] = {
    "graphsage-scratch": ("random", "none", 0.0, 0.0, (0.0,), False),
    "graphmae2-single": (
        "pretrained",
        "single-source",
        1.0,
        0.5,
        (0.0,),
        True,
    ),
    "multi-graph-shared-gfm": (
        "pretrained",
        "multi-source",
        1.0,
        0.5,
        (0.0,),
        True,
    ),
    "domain-aligned-gfm": (
        "pretrained",
        "multi-source",
        1.0,
        0.5,
        (0.0, 0.02, 0.05),
        True,
    ),
}


@dataclass(frozen=True)
class GfmFamilySpec:
    """Fixed backend intent; target labels are unavailable to every pretraining family."""

    method_id: GfmMethodId
    encoder_initialization: Literal["random", "pretrained"]
    pretraining_scope: Literal["none", "single-source", "multi-source"]
    field_reconstruction_weight: float
    edge_reconstruction_weight: float
    alignment_candidates: tuple[float, ...]
    target_unlabeled_adaptation: bool
    target_labels_in_pretraining: Literal[False] = False

    def __post_init__(self) -> None:
        expected = _GFM_FAMILY_DEFINITIONS.get(self.method_id)
        if expected is None:
            raise ValueError("GFM family configuration differs from the fixed protocol")
        observed = (
            self.encoder_initialization,
            self.pretraining_scope,
            self.field_reconstruction_weight,
            self.edge_reconstruction_weight,
            self.alignment_candidates,
            self.target_unlabeled_adaptation,
        )
        if observed != expected or self.target_labels_in_pretraining is not False:
            raise ValueError("GFM family configuration differs from the fixed protocol")


def fixed_gfm_family_specs() -> tuple[GfmFamilySpec, ...]:
    return tuple(
        GfmFamilySpec(
            method_id=method_id,
            encoder_initialization=definition[0],
            pretraining_scope=definition[1],
            field_reconstruction_weight=definition[2],
            edge_reconstruction_weight=definition[3],
            alignment_candidates=definition[4],
            target_unlabeled_adaptation=definition[5],
        )
        for method_id, definition in _GFM_FAMILY_DEFINITIONS.items()
    )


class SparseLinkx(nn.Module):
    """Sparse LINKX baseline with independent topology and feature branches."""

    def __init__(
        self,
        *,
        num_nodes: int,
        feature_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.num_nodes = _positive_dimension(num_nodes, label="SparseLinkx node count")
        self.feature_dim = _positive_dimension(feature_dim, label="SparseLinkx feature dimension")
        normalized_hidden_dim = _positive_dimension(
            hidden_dim, label="SparseLinkx hidden dimension"
        )
        normalized_output_dim = _positive_dimension(
            output_dim, label="SparseLinkx output dimension"
        )
        self.hidden_dim = normalized_hidden_dim
        self.node_embeddings = nn.Embedding(self.num_nodes, normalized_hidden_dim)
        self.adjacency_branch = nn.Sequential(
            nn.LayerNorm(normalized_hidden_dim),
            nn.ReLU(),
            nn.Linear(normalized_hidden_dim, normalized_hidden_dim),
            nn.ReLU(),
        )
        self.feature_branch = nn.Sequential(
            nn.Linear(self.feature_dim, normalized_hidden_dim),
            nn.LayerNorm(normalized_hidden_dim),
            nn.ReLU(),
            nn.Linear(normalized_hidden_dim, normalized_hidden_dim),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(normalized_hidden_dim * 2, normalized_hidden_dim),
            nn.LayerNorm(normalized_hidden_dim),
            nn.ReLU(),
            nn.Linear(normalized_hidden_dim, normalized_output_dim),
        )

    def forward(self, adjacency: Tensor, features: Tensor) -> Tensor:
        if adjacency.layout != torch.sparse_coo:
            raise ValueError("SparseLinkx adjacency must use sparse COO storage")
        if not adjacency.is_coalesced():
            raise ValueError("SparseLinkx adjacency must be coalesced")
        if adjacency.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("SparseLinkx adjacency shape is invalid")
        if features.shape != (self.num_nodes, self.feature_dim):
            raise ValueError("SparseLinkx feature shape is invalid")
        if features.layout != torch.strided:
            raise ValueError("SparseLinkx features must use dense strided storage")
        if not adjacency.is_floating_point() or not features.is_floating_point():
            raise ValueError("SparseLinkx inputs must use floating point tensors")
        parameter = self.node_embeddings.weight
        if (
            adjacency.device != parameter.device
            or features.device != parameter.device
            or adjacency.dtype != parameter.dtype
            or features.dtype != parameter.dtype
        ):
            raise ValueError("SparseLinkx inputs must match the model dtype and device")
        if not bool(torch.isfinite(adjacency.values()).all()) or not bool(
            torch.isfinite(features).all()
        ):
            raise ValueError("SparseLinkx inputs must be finite")
        # A @ W is the sparse first adjacency projection used instead of materializing
        # an N x N dense row matrix.  The two modalities remain separate until fusion.
        topology = self.adjacency_branch(torch.sparse.mm(adjacency, self.node_embeddings.weight))
        attributes = self.feature_branch(features)
        return self.fusion(torch.cat((topology, attributes), dim=1))


__all__ = [
    "FeatureMlp",
    "GfmFamilySpec",
    "SparseLinkx",
    "StructureMlp",
    "adamic_adar_scores",
    "common_neighbors_scores",
    "fixed_gfm_family_specs",
]
