"""Exact, deterministic negative sampling for undirected simple graphs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from .types import edge_pairs


def canonical_pair(source: int, target: int) -> tuple[int, int]:
    return (source, target) if source < target else (target, source)


def canonical_edge_set(edges: Any) -> set[tuple[int, int]]:
    pairs = edge_pairs(edges, name="edges")
    return {
        canonical_pair(int(source), int(target))
        for source, target in pairs
        if int(source) != int(target)
    }


class ExactUndirectedNegativeSampler:
    """Sample unique non-edges without self-loops or approximate false negatives.

    The sampler owns its NumPy generator and exposes its complete state for
    checkpoint/resume.  It never consults any edge collection other than the
    caller-provided point-in-time forbidden set.
    """

    def __init__(self, num_nodes: int, forbidden_edges: Any, *, seed: int) -> None:
        if num_nodes < 2:
            raise ValueError("negative sampling requires at least two nodes")
        import numpy as np

        self.num_nodes = int(num_nodes)
        self.forbidden = canonical_edge_set(forbidden_edges)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.draw_count = 0

    @property
    def available_count(self) -> int:
        return self.num_nodes * (self.num_nodes - 1) // 2 - len(self.forbidden)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "gfm.exact-negative-sampler/1.0",
            "numNodes": self.num_nodes,
            "seed": self.seed,
            "drawCount": self.draw_count,
            "bitGeneratorState": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schemaVersion") != "gfm.exact-negative-sampler/1.0":
            raise ValueError("unsupported negative sampler state")
        if int(state.get("numNodes", -1)) != self.num_nodes or int(state.get("seed", -1)) != self.seed:
            raise ValueError("negative sampler state identity mismatch")
        self._rng.bit_generator.state = deepcopy(state["bitGeneratorState"])
        self.draw_count = int(state["drawCount"])

    def sample(self, count: int, *, additional_forbidden: Any | None = None) -> Any:
        import numpy as np

        count = int(count)
        if count < 0:
            raise ValueError("negative sample count must be non-negative")
        extra = canonical_edge_set(additional_forbidden) if additional_forbidden is not None else set()
        forbidden = self.forbidden | extra
        available = self.num_nodes * (self.num_nodes - 1) // 2 - len(forbidden)
        if count > available:
            raise ValueError(f"requested {count} negatives but only {available} non-edges exist")
        if count == 0:
            return np.empty((0, 2), dtype=np.int64)

        chosen: set[tuple[int, int]] = set()
        ordered: list[tuple[int, int]] = []
        # Rejection sampling is fast for sparse social graphs.  A deterministic
        # lexicographic completion guarantees termination on dense test graphs.
        budget = self.draw_count + max(1024, count * 32)
        while len(ordered) < count and self.draw_count < budget:
            batch_size = min(max(64, (count - len(ordered)) * 3), 65536)
            endpoints = self._rng.integers(0, self.num_nodes, size=(batch_size, 2))
            self.draw_count += batch_size
            for source, target in endpoints:
                source_i, target_i = int(source), int(target)
                if source_i == target_i:
                    continue
                pair = canonical_pair(source_i, target_i)
                if pair in forbidden or pair in chosen:
                    continue
                chosen.add(pair)
                ordered.append(pair)
                if len(ordered) == count:
                    break
        if len(ordered) < count:
            for source in range(self.num_nodes):
                for target in range(source + 1, self.num_nodes):
                    pair = (source, target)
                    if pair in forbidden or pair in chosen:
                        continue
                    chosen.add(pair)
                    ordered.append(pair)
                    if len(ordered) == count:
                        break
                if len(ordered) == count:
                    break
        return np.asarray(ordered, dtype=np.int64)


def forbidden_union(*collections: Iterable[Any] | Any) -> Any:
    """Return sorted canonical pairs, useful as sampler input and audit evidence."""

    import numpy as np

    union: set[tuple[int, int]] = set()
    for collection in collections:
        union.update(canonical_edge_set(collection))
    return np.asarray(sorted(union), dtype=np.int64).reshape(-1, 2)
