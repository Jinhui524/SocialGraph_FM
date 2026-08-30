"""Cached static topology and deterministic graph-balanced scheduling."""

from __future__ import annotations

import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from typing import Literal
from typing import Any

import torch
from torch import Tensor


class InsufficientNegativeCapacityError(ValueError):
    """The requested unique negatives do not exist in the eligible node space."""


@dataclass(frozen=True)
class ExecutionPolicy:
    full_batch_edge_threshold: int = 100_000
    node_batch_size: int = 1024
    edge_batch_size: int = 2048
    fanout: tuple[int, int, int] = (15, 10, 5)

    def mode(self, *, edge_count: int) -> str:
        if edge_count < 0:
            raise ValueError("edge count must be nonnegative")
        return "full-batch" if edge_count < self.full_batch_edge_threshold else "neighbor"

    def node_loader(
        self,
        data: Any,
        *,
        input_nodes: Tensor | None = None,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> Any:
        from torch_geometric.loader import NeighborLoader

        return NeighborLoader(
            data,
            input_nodes=input_nodes,
            num_neighbors=list(self.fanout),
            batch_size=self.node_batch_size,
            shuffle=shuffle,
            generator=generator,
        )

    def link_loader(
        self,
        data: Any,
        *,
        edge_label_index: Tensor,
        shuffle: bool = True,
        generator: torch.Generator | None = None,
    ) -> Any:
        from torch_geometric.loader import LinkNeighborLoader

        return LinkNeighborLoader(
            data,
            edge_label_index=edge_label_index,
            num_neighbors=list(self.fanout),
            batch_size=self.edge_batch_size,
            shuffle=shuffle,
            generator=generator,
        )


@dataclass(frozen=True)
class FullGraphBatch:
    edge_index: Tensor


@dataclass(frozen=True)
class PairMaskCache:
    pair_keys: Tensor
    inverse: Tensor
    canonical_pairs: Tensor
    representative_edges: Tensor


@dataclass(frozen=True)
class SampledGraphBatch:
    features: Tensor
    edge_index: Tensor
    global_node_ids: Tensor
    global_pair_ids: Tensor
    global_pair_representatives: Tensor
    seed_count: int


class NeighborBatchSource:
    def __init__(
        self,
        *,
        graph: PreparedGraph,
        features: Tensor,
        policy: ExecutionPolicy,
        loader_kind: Literal["node", "link"],
        seed: int,
        edge_label_index: Tensor | None = None,
    ) -> None:
        from torch_geometric.data import Data

        if loader_kind == "link" and edge_label_index is None:
            raise ValueError("link neighbor batches require edge_label_index")
        self.graph = graph
        self.features = features.detach().cpu()
        self.policy = policy
        self.loader_kind = loader_kind
        self.seed = int(seed)
        self.edge_label_index = (
            None if edge_label_index is None else edge_label_index.detach().cpu()
        )
        self.data = Data(
            x=self.features,
            edge_index=graph.edge_index.detach().cpu(),
            global_pair_id=graph.pair_mask_cache.inverse.detach().cpu(),
            global_pair_representative=(graph.pair_mask_cache.representative_edges.detach().cpu()),
        )
        item_count = graph.num_nodes if loader_kind == "node" else edge_label_index.shape[1]  # type: ignore[union-attr]
        batch_size = policy.node_batch_size if loader_kind == "node" else policy.edge_batch_size
        self.batch_count = ceil(item_count / batch_size)
        self.loader_construction_count = 0

    @property
    def retained_batch_count(self) -> int:
        return 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": "socialgraph-fm.core-neighbor-source/1.0",
            "seed": self.seed,
            "loaderKind": self.loader_kind,
            "batchCount": self.batch_count,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != "socialgraph-fm.core-neighbor-source/1.0":
            raise ValueError("unsupported neighbor source state")
        if (
            state.get("loaderKind") != self.loader_kind
            or state.get("batchCount") != self.batch_count
        ):
            raise ValueError("neighbor source topology does not match checkpoint")
        self.seed = int(state["seed"])

    def get(self, *, batch_index: int, ordinal: int) -> SampledGraphBatch:
        generator = torch.Generator().manual_seed(self.seed + ordinal * 1_009 + batch_index)
        if self.loader_kind == "node":
            start = batch_index * self.policy.node_batch_size
            seeds = torch.arange(
                start, min(start + self.policy.node_batch_size, self.graph.num_nodes)
            )
            loader = self.policy.node_loader(
                self.data, input_nodes=seeds, shuffle=False, generator=generator
            )
        else:
            if self.edge_label_index is None:  # pragma: no cover - constructor enforces this
                raise RuntimeError("edge labels are unavailable")
            start = batch_index * self.policy.edge_batch_size
            labels = (
                self.edge_label_index[:, start : start + self.policy.edge_batch_size].detach().cpu()
            )
            seeds = torch.arange(labels.shape[1])
            loader = self.policy.link_loader(
                self.data,
                edge_label_index=labels,
                shuffle=False,
                generator=generator,
            )
        self.loader_construction_count += 1
        batch = next(iter(loader))
        global_nodes = batch.n_id.to(dtype=torch.long)
        return SampledGraphBatch(
            features=batch.x,
            edge_index=batch.edge_index.to(dtype=torch.long),
            global_node_ids=global_nodes,
            global_pair_ids=batch.global_pair_id.to(dtype=torch.long),
            global_pair_representatives=batch.global_pair_representative.to(dtype=torch.bool),
            seed_count=int(batch.batch_size if self.loader_kind == "node" else seeds.shape[0]),
        )


@dataclass
class PreparedGraph:
    num_nodes: int
    edge_index: Tensor
    directed: bool
    csr: Tensor
    csc: Tensor
    positive_edge_keys: Tensor
    pair_mask_cache: PairMaskCache
    structural_inputs: Tensor | None = None
    cache_build_count: int = 1
    pair_cache_build_count: int = 1

    @classmethod
    def from_edge_index(
        cls,
        *,
        num_nodes: int,
        edge_index: Tensor,
        directed: bool,
        positive_edge_index: Tensor | None = None,
        structural_inputs: Tensor | None = None,
    ) -> PreparedGraph:
        if num_nodes < 1:
            raise ValueError("prepared graph requires at least one node")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long")
        if edge_index.numel() and (
            bool(torch.any(edge_index < 0)) or bool(torch.any(edge_index >= num_nodes))
        ):
            raise ValueError("edge endpoint is outside the node range")
        values = torch.ones(edge_index.shape[1], device=edge_index.device)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse invariant checks are implicitly disabled.*",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            sparse = torch.sparse_coo_tensor(
                edge_index,
                values,
                (num_nodes, num_nodes),
                device=edge_index.device,
                check_invariants=False,
            ).coalesce()
            csr = sparse.to_sparse_csr()
            csc = sparse.to_sparse_csc()
        source, target = edge_index
        pair_keys_for_edges = torch.minimum(source, target) * num_nodes + torch.maximum(
            source, target
        )
        unique_pair_keys, pair_inverse = torch.unique(
            pair_keys_for_edges, sorted=True, return_inverse=True
        )
        edge_ids = torch.arange(edge_index.shape[1], device=edge_index.device)
        representative_ids = torch.full(
            (unique_pair_keys.shape[0],),
            edge_index.shape[1],
            dtype=torch.long,
            device=edge_index.device,
        )
        representative_ids.scatter_reduce_(
            0,
            pair_inverse,
            edge_ids,
            reduce="amin",
            include_self=True,
        )
        pair_cache = PairMaskCache(
            pair_keys=unique_pair_keys,
            inverse=pair_inverse,
            canonical_pairs=torch.stack(
                (
                    torch.div(unique_pair_keys, num_nodes, rounding_mode="floor"),
                    unique_pair_keys % num_nodes,
                ),
                dim=1,
            ),
            representative_edges=edge_ids == representative_ids[pair_inverse],
        )
        membership_edges = edge_index if positive_edge_index is None else positive_edge_index
        if membership_edges.ndim != 2 or membership_edges.shape[0] != 2:
            raise ValueError("positive_edge_index must have shape [2, edges]")
        source, target = membership_edges
        if directed:
            keys = source * num_nodes + target
        else:
            keys = torch.minimum(source, target) * num_nodes + torch.maximum(source, target)
        return cls(
            num_nodes=num_nodes,
            edge_index=edge_index,
            directed=directed,
            csr=csr,
            csc=csc,
            positive_edge_keys=torch.unique(keys, sorted=True),
            pair_mask_cache=pair_cache,
            structural_inputs=structural_inputs,
        )

    def full_batch(self) -> FullGraphBatch:
        return FullGraphBatch(edge_index=self.edge_index)

    def to(self, device: torch.device | str) -> PreparedGraph:
        return PreparedGraph(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index.to(device),
            directed=self.directed,
            csr=self.csr.to(device),
            csc=self.csc.to(device),
            positive_edge_keys=self.positive_edge_keys.to(device),
            pair_mask_cache=PairMaskCache(
                pair_keys=self.pair_mask_cache.pair_keys.to(device),
                inverse=self.pair_mask_cache.inverse.to(device),
                canonical_pairs=self.pair_mask_cache.canonical_pairs.to(device),
                representative_edges=self.pair_mask_cache.representative_edges.to(device),
            ),
            structural_inputs=(
                None if self.structural_inputs is None else self.structural_inputs.to(device)
            ),
            cache_build_count=self.cache_build_count,
            pair_cache_build_count=self.pair_cache_build_count,
        )

    def contains_positive(self, pairs: Tensor) -> Tensor:
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape [count, 2]")
        source, target = pairs[:, 0], pairs[:, 1]
        if self.directed:
            keys = source * self.num_nodes + target
        else:
            keys = torch.minimum(source, target) * self.num_nodes + torch.maximum(source, target)
        result_device = keys.device
        membership_keys = keys.to(self.positive_edge_keys.device)
        if self.positive_edge_keys.numel() == 0:
            return torch.zeros(keys.shape, dtype=torch.bool, device=result_device)
        positions = torch.searchsorted(self.positive_edge_keys, membership_keys)
        in_range = positions < self.positive_edge_keys.numel()
        clamped = positions.clamp(max=max(0, self.positive_edge_keys.numel() - 1))
        membership = in_range & (self.positive_edge_keys[clamped] == membership_keys)
        return membership.to(result_device)

    @property
    def negative_capacity(self) -> int:
        total = self.num_nodes * (self.num_nodes - 1)
        if not self.directed:
            total //= 2
        return total - self.positive_edge_keys.numel()

    def sample_negative_pairs(self, count: int, *, generator: torch.Generator) -> Tensor:
        count = int(count)
        if count < 0:
            raise ValueError("negative sample count must be nonnegative")
        if count == 0:
            return self.edge_index.new_empty((0, 2))
        if self.negative_capacity == 0:
            raise InsufficientNegativeCapacityError("graph has no negative edges")
        if count > self.negative_capacity:
            raise InsufficientNegativeCapacityError("insufficient unique negative edges")
        device = self.positive_edge_keys.device
        selected = self.edge_index.new_empty((0, 2), device=device)
        attempts = 0
        budget = max(1024, count * 32)
        while selected.shape[0] < count and attempts < budget:
            candidate_count = min(budget - attempts, max(64, (count - selected.shape[0]) * 8))
            attempts += candidate_count
            candidate = torch.randint(
                self.num_nodes,
                (candidate_count, 2),
                generator=generator,
                device=device,
            )
            legal = candidate[:, 0] != candidate[:, 1]
            if not self.directed:
                candidate = torch.stack(
                    (
                        torch.minimum(candidate[:, 0], candidate[:, 1]),
                        torch.maximum(candidate[:, 0], candidate[:, 1]),
                    ),
                    dim=1,
                )
            candidate = candidate[legal & ~self.contains_positive(candidate)]
            if candidate.numel():
                selected = torch.unique(torch.cat((selected, candidate)), dim=0)
        if selected.shape[0] < count:
            chunk = 65_536
            for start in range(0, self.num_nodes * self.num_nodes, chunk):
                keys = torch.arange(
                    start,
                    min(start + chunk, self.num_nodes * self.num_nodes),
                    device=device,
                )
                candidate = torch.stack((keys // self.num_nodes, keys % self.num_nodes), dim=1)
                legal = candidate[:, 0] != candidate[:, 1]
                if not self.directed:
                    legal &= candidate[:, 0] < candidate[:, 1]
                candidate = candidate[legal & ~self.contains_positive(candidate)]
                if candidate.numel():
                    selected = torch.unique(torch.cat((selected, candidate)), dim=0)
                if selected.shape[0] >= count:
                    break
        return selected[:count]

    def sample_negative_pairs_from_nodes(
        self,
        global_node_ids: Tensor,
        count: int,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Sample local endpoint indices while rejecting against global positive membership."""

        count = int(count)
        if global_node_ids.ndim != 1 or global_node_ids.numel() < 1:
            raise ValueError("global node IDs must be a nonempty vector")
        if count < 0:
            raise ValueError("negative sample count must be nonnegative")
        if count == 0:
            return global_node_ids.new_empty((0, 2))
        node_count = global_node_ids.shape[0]
        selected = global_node_ids.new_empty((0, 2))
        attempts = 0
        budget = max(1024, count * 32)
        while selected.shape[0] < count and attempts < budget:
            candidate_count = min(budget - attempts, max(64, (count - selected.shape[0]) * 8))
            attempts += candidate_count
            candidate = torch.randint(
                node_count,
                (candidate_count, 2),
                generator=generator,
                device=global_node_ids.device,
            )
            global_pairs = global_node_ids[candidate]
            legal = global_pairs[:, 0] != global_pairs[:, 1]
            if not self.directed:
                swap = global_pairs[:, 0] > global_pairs[:, 1]
                candidate = torch.where(swap.unsqueeze(1), candidate.flip(1), candidate)
                global_pairs = global_node_ids[candidate]
            candidate = candidate[legal & ~self.contains_positive(global_pairs)]
            if candidate.numel():
                selected = torch.unique(torch.cat((selected, candidate)), dim=0)
        if selected.shape[0] < count:
            chunk = 65_536
            for start in range(0, node_count * node_count, chunk):
                keys = torch.arange(
                    start,
                    min(start + chunk, node_count * node_count),
                    device=global_node_ids.device,
                )
                candidate = torch.stack((keys // node_count, keys % node_count), dim=1)
                global_pairs = global_node_ids[candidate]
                legal = global_pairs[:, 0] != global_pairs[:, 1]
                if not self.directed:
                    legal &= global_pairs[:, 0] < global_pairs[:, 1]
                candidate = candidate[legal & ~self.contains_positive(global_pairs)]
                if candidate.numel():
                    selected = torch.unique(torch.cat((selected, candidate)), dim=0)
                if selected.shape[0] >= count:
                    break
        if selected.shape[0] < count:
            raise InsufficientNegativeCapacityError(
                "insufficient unique negative edges among sampled nodes"
            )
        return selected[:count]


class BalancedDomainSampler:
    """Uniformly interleave domains and cycle each domain's batches."""

    def __init__(self, batch_counts: Mapping[str, int], *, seed: int) -> None:
        if not batch_counts or any(count < 1 for count in batch_counts.values()):
            raise ValueError("every domain must contain at least one batch")
        self._counts = dict(sorted(batch_counts.items()))
        self._rng = random.Random(int(seed))
        self._cursors = {domain: 0 for domain in self._counts}
        self._order: list[str] = []
        self._position = 0
        self._last_domain: str | None = None
        self._totals = {domain: 0 for domain in self._counts}
        self._last_ordinal: dict[str, int] = {}
        self._new_cycle()

    def _new_cycle(self) -> None:
        self._order = list(self._counts)
        self._rng.shuffle(self._order)
        if self._last_domain is not None and self._order[0] == self._last_domain:
            self._order = self._order[1:] + self._order[:1]
        self._position = 0

    def next(self) -> tuple[str, int]:
        if self._position == len(self._order):
            self._new_cycle()
        domain = self._order[self._position]
        self._position += 1
        batch = self._cursors[domain]
        self._cursors[domain] = (batch + 1) % self._counts[domain]
        ordinal = self._totals[domain]
        self._totals[domain] += 1
        self._last_ordinal[domain] = ordinal
        self._last_domain = domain
        return domain, batch

    def last_ordinal(self, domain: str) -> int:
        return self._last_ordinal[domain]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": "socialgraph-fm.core-domain-sampler/1.1",
            "counts": dict(self._counts),
            "rngState": self._rng.getstate(),
            "cursors": dict(self._cursors),
            "order": list(self._order),
            "position": self._position,
            "lastDomain": self._last_domain,
            "totals": dict(self._totals),
            "lastOrdinal": dict(self._last_ordinal),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != "socialgraph-fm.core-domain-sampler/1.1":
            raise ValueError("unsupported domain sampler state")
        if state.get("counts") != self._counts:
            raise ValueError("domain sampler batch counts do not match")
        self._rng.setstate(state["rngState"])
        self._cursors = dict(state["cursors"])
        self._order = list(state["order"])
        self._position = int(state["position"])
        self._last_domain = state["lastDomain"]
        self._totals = dict(state["totals"])
        self._last_ordinal = dict(state["lastOrdinal"])


__all__ = [
    "BalancedDomainSampler",
    "ExecutionPolicy",
    "FullGraphBatch",
    "InsufficientNegativeCapacityError",
    "NeighborBatchSource",
    "PairMaskCache",
    "PreparedGraph",
    "SampledGraphBatch",
]
