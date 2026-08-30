"""Deterministic, leakage-safe split primitives for static graph tasks."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .graph_ops import canonicalize_edges


@dataclass(frozen=True)
class IndexSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class GraphSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class EdgeSplit:
    train: tuple[tuple[int, int], ...]
    validation: tuple[tuple[int, int], ...]
    test: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SignedEdgeSplit:
    train: tuple[tuple[int, int, int], ...]
    validation: tuple[tuple[int, int, int], ...]
    test: tuple[tuple[int, int, int], ...]


def ingest_official_masks(
    *,
    train_mask: Iterable[bool],
    validation_mask: Iterable[bool],
    test_mask: Iterable[bool],
) -> IndexSplit:
    """Convert mutually exclusive official boolean masks into stable row indices."""

    raw_masks = (tuple(train_mask), tuple(validation_mask), tuple(test_mask))
    masks: list[tuple[bool, ...]] = []
    for mask in raw_masks:
        normalized: list[bool] = []
        for value in mask:
            if type(value) is bool:
                normalized.append(value)
            else:
                try:
                    import numpy as np
                except ImportError:  # pragma: no cover - NumPy is part of the GFM runtime.
                    np = None  # type: ignore[assignment]
                if np is not None and type(value) is np.bool_:
                    normalized.append(bool(value))
                    continue
                # Deliberately reject truthy integers such as 0/1. Official masks
                # are an identity-bearing input and must be explicitly boolean.
                raise ValueError("official masks must contain booleans only")
        masks.append(tuple(normalized))
    lengths = {len(mask) for mask in masks}
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("official masks must have the same nonzero length")
    roles: list[list[int]] = [[], [], []]
    for index, flags in enumerate(zip(*masks)):
        if sum(flags) != 1:
            raise ValueError("every official-mask row must belong to exactly one split")
        roles[flags.index(True)].append(index)
    return IndexSplit(*(tuple(role) for role in roles))


def graph_disjoint_split(
    *,
    graph_ids: Iterable[str],
    validation_graph_ids: Iterable[str],
    test_graph_ids: Iterable[str],
) -> GraphSplit:
    identifiers = tuple(graph_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise ValueError("graph IDs must be nonempty and unique")
    all_graphs = set(identifiers)
    validation = set(validation_graph_ids)
    test = set(test_graph_ids)
    if validation & test:
        raise ValueError("validation and test graph sets must be disjoint")
    if not validation | test <= all_graphs:
        raise ValueError("split references an unknown graph ID")
    train = all_graphs - validation - test
    return GraphSplit(tuple(sorted(train)), tuple(sorted(validation)), tuple(sorted(test)))


def leave_one_domain_out(
    *,
    graph_domains: Mapping[str, str],
    test_domain: str,
    validation_domain: str,
) -> GraphSplit:
    if not graph_domains:
        raise ValueError("leave-one-domain-out requires at least one graph")
    domains = set(graph_domains.values())
    if test_domain == validation_domain:
        raise ValueError("validation and test domains must differ")
    if test_domain not in domains or validation_domain not in domains:
        raise ValueError("validation and test domains must exist")
    validation = tuple(
        sorted(graph for graph, domain in graph_domains.items() if domain == validation_domain)
    )
    test = tuple(sorted(graph for graph, domain in graph_domains.items() if domain == test_domain))
    train = tuple(
        sorted(
            graph
            for graph, domain in graph_domains.items()
            if domain not in {validation_domain, test_domain}
        )
    )
    return GraphSplit(train, validation, test)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def spanning_forest_link_split(
    *, num_nodes: int, edges: Iterable[tuple[int, int]], seed: int
) -> EdgeSplit:
    """Split canonical undirected pairs, retaining a sorted spanning forest in train."""

    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    canonical = canonicalize_edges(edges, directed=False)
    if any(source < 0 or target >= num_nodes for source, target in canonical):
        raise ValueError("edge endpoint is outside the node range")

    union_find = _UnionFind(num_nodes)
    forest: list[tuple[int, int]] = []
    remainder: list[tuple[int, int]] = []
    for edge in canonical:
        if union_find.union(*edge):
            forest.append(edge)
        else:
            remainder.append(edge)

    edge_count = len(canonical)
    _, desired_validation, desired_test = _largest_remainder_counts(edge_count, (0.80, 0.10, 0.10))
    capacity = len(remainder)
    if capacity >= desired_validation + desired_test:
        validation_count, test_count = desired_validation, desired_test
    else:
        validation_count = capacity // 2
        test_count = capacity - validation_count

    shuffled = list(remainder)
    random.Random(int(seed)).shuffle(shuffled)
    train_extra_count = len(shuffled) - validation_count - test_count
    train = tuple(sorted((*forest, *shuffled[:train_extra_count])))
    validation_end = train_extra_count + validation_count
    validation = tuple(sorted(shuffled[train_extra_count:validation_end]))
    test = tuple(sorted(shuffled[validation_end:]))
    return EdgeSplit(train, validation, test)


def _largest_remainder_counts(
    size: int, ratios: tuple[float, float, float]
) -> tuple[int, int, int]:
    raw = tuple(size * ratio for ratio in ratios)
    counts = [math.floor(value) for value in raw]
    remaining = size - sum(counts)
    priorities = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in priorities[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def stratified_signed_edge_split(
    *, edges: Iterable[tuple[int, int, int]], seed: int
) -> SignedEdgeSplit:
    """Stratify unordered pair groups while preserving directed edges and their signs."""

    groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    directed_pairs: set[tuple[int, int]] = set()
    for source, target, sign in edges:
        source_i, target_i, sign_i = int(source), int(target), int(sign)
        if source_i < 0 or target_i < 0 or source_i == target_i:
            raise ValueError("signed edges require distinct nonnegative endpoints")
        if sign_i not in {-1, 1}:
            raise ValueError("signed edge sign must be -1 or 1")
        directed_pair = (source_i, target_i)
        if directed_pair in directed_pairs:
            raise ValueError("each directed signed pair must occur exactly once")
        directed_pairs.add(directed_pair)
        pair = (min(source_i, target_i), max(source_i, target_i))
        groups[pair].append((source_i, target_i, sign_i))

    strata: dict[tuple[int, ...], list[tuple[int, int]]] = defaultdict(list)
    for pair, group_edges in groups.items():
        strata[tuple(sorted({edge[2] for edge in group_edges}))].append(pair)

    assigned: list[list[tuple[int, int, int]]] = [[], [], []]
    rng = random.Random(int(seed))
    for label in sorted(strata):
        pairs = sorted(strata[label])
        rng.shuffle(pairs)
        counts = _largest_remainder_counts(len(pairs), (0.70, 0.15, 0.15))
        boundaries = (counts[0], counts[0] + counts[1], sum(counts))
        for role_index, selected_pairs in enumerate(
            (pairs[: boundaries[0]], pairs[boundaries[0] : boundaries[1]], pairs[boundaries[1] :])
        ):
            for pair in selected_pairs:
                assigned[role_index].extend(groups[pair])
    return SignedEdgeSplit(*(tuple(sorted(role_edges)) for role_edges in assigned))


__all__ = [
    "EdgeSplit",
    "GraphSplit",
    "IndexSplit",
    "SignedEdgeSplit",
    "graph_disjoint_split",
    "ingest_official_masks",
    "leave_one_domain_out",
    "spanning_forest_link_split",
    "stratified_signed_edge_split",
]
