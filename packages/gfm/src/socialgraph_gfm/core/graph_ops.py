"""Canonical edge masking and exact negative sampling without dense candidates."""

from __future__ import annotations

import random
from collections.abc import Iterable, Mapping


Edge = tuple[int, int]


def _canonical_edge(source: int, target: int, *, directed: bool) -> Edge:
    source_i, target_i = int(source), int(target)
    if source_i == target_i:
        raise ValueError("self-loop edges are not supported")
    if directed or source_i < target_i:
        return source_i, target_i
    return target_i, source_i


def canonicalize_edges(edges: Iterable[Edge], *, directed: bool) -> tuple[Edge, ...]:
    """Return deterministic unique edges, with unordered pairs atomic when undirected."""

    return tuple(
        sorted(
            {
                _canonical_edge(source, target, directed=directed)
                for source, target in edges
            }
        )
    )


def mask_message_passing_edges(
    edges: Iterable[Edge], *, masked_edges: Iterable[Edge], directed: bool
) -> tuple[Edge, ...]:
    """Remove exact directed edges or both orientations of an undirected pair."""

    forbidden = {
        _canonical_edge(source, target, directed=directed)
        for source, target in masked_edges
    }
    retained: list[Edge] = []
    for source, target in edges:
        key = _canonical_edge(source, target, directed=directed)
        if key not in forbidden:
            retained.append((int(source), int(target)))
    return tuple(retained)


def _known_positives(
    positive_splits: Mapping[str, Iterable[Edge]], *, num_nodes: int, directed: bool
) -> set[Edge]:
    legal_roles = {"train", "validation", "test"}
    if set(positive_splits) != legal_roles:
        raise ValueError("positive splits must include all train, validation, and test edge sets")
    positives: set[Edge] = set()
    for edges in positive_splits.values():
        for source, target in edges:
            edge = _canonical_edge(source, target, directed=directed)
            if min(edge) < 0 or max(edge) >= num_nodes:
                raise ValueError("positive edge endpoint is outside the node range")
            positives.add(edge)
    return positives


def sample_negative_edges(
    *,
    num_nodes: int,
    positive_splits: Mapping[str, Iterable[Edge]],
    count: int,
    seed: int,
    directed: bool,
) -> tuple[Edge, ...]:
    """Sample unique exact non-edges while excluding positives from every split.

    Random rejection handles normal sparse graphs. A streaming lexicographic fallback
    guarantees termination for dense graphs without ever materializing O(N^2) candidates.
    """

    num_nodes, count = int(num_nodes), int(count)
    if num_nodes < 2:
        raise ValueError("negative sampling requires at least two nodes")
    if count < 0:
        raise ValueError("negative sample count must be nonnegative")
    positives = _known_positives(
        positive_splits, num_nodes=num_nodes, directed=directed
    )
    total = num_nodes * (num_nodes - 1)
    if not directed:
        total //= 2
    available = total - len(positives)
    if count > available:
        raise ValueError(f"requested {count} negatives but only {available} are available")

    rng = random.Random(int(seed))
    selected: set[Edge] = set()
    ordered: list[Edge] = []
    attempts = 0
    budget = max(1024, count * 32)
    while len(ordered) < count and attempts < budget:
        attempts += 1
        source = rng.randrange(num_nodes)
        target = rng.randrange(num_nodes)
        if source == target:
            continue
        edge = _canonical_edge(source, target, directed=directed)
        if edge in positives or edge in selected:
            continue
        selected.add(edge)
        ordered.append(edge)

    if len(ordered) < count:
        for source in range(num_nodes):
            targets = range(num_nodes) if directed else range(source + 1, num_nodes)
            for target in targets:
                if source == target:
                    continue
                edge = (source, target)
                if edge in positives or edge in selected:
                    continue
                selected.add(edge)
                ordered.append(edge)
                if len(ordered) == count:
                    break
            if len(ordered) == count:
                break
    return tuple(ordered)


__all__ = [
    "canonicalize_edges",
    "mask_message_passing_edges",
    "sample_negative_edges",
]
