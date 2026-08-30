"""Causal exact negative sampling and deterministic multi-domain scheduling."""

from __future__ import annotations

import hashlib
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor


def _edge_pairs(edge_index: Tensor, *, name: str) -> list[tuple[int, int]]:
    value = torch.as_tensor(edge_index).detach().cpu()
    if value.dtype != torch.long or value.ndim != 2 or value.shape[0] != 2:
        raise ValueError(f"{name} must be torch.long [2, E]")
    return [(int(source), int(target)) for source, target in value.t().tolist()]


class CausalExactNegativeSampler:
    """Sample unique non-edges from an explicitly point-in-time relation graph.

    Each instance represents one homogeneous or bipartite relation.  It accepts
    no future-edge collection and rejects rather than filters history after the
    cutoff, making accidental future access visible to callers and tests.
    """

    def __init__(
        self,
        *,
        source_count: int,
        target_count: int,
        visible_edge_index: Tensor,
        visible_edge_time: Tensor,
        cutoff_time: float,
        seed: int,
        directed: bool,
        same_node_space: bool = True,
    ) -> None:
        if source_count < 1 or target_count < 1:
            raise ValueError("negative sampling requires nonempty endpoint spaces")
        if same_node_space and source_count != target_count:
            raise ValueError("same_node_space requires equal endpoint counts")
        times = torch.as_tensor(visible_edge_time, dtype=torch.float64).reshape(-1)
        pairs = _edge_pairs(visible_edge_index, name="visible_edge_index")
        if times.shape[0] != len(pairs) or not bool(torch.isfinite(times).all()):
            raise ValueError("visible_edge_time must contain one finite time per edge")
        if not np.isfinite(cutoff_time):
            raise ValueError("cutoff_time must be finite")
        if times.numel() and bool(torch.any(times > float(cutoff_time))):
            raise ValueError("visible history contains an edge after cutoff_time")

        self.source_count = int(source_count)
        self.target_count = int(target_count)
        self.cutoff_time = float(cutoff_time)
        self.seed = int(seed)
        self.directed = bool(directed)
        self.same_node_space = bool(same_node_space)
        self._rng = np.random.default_rng(self.seed)
        self.draw_count = 0
        self.forbidden: set[tuple[int, int]] = set()
        for source, target in pairs:
            pair = self._canonical(source, target)
            if pair is not None:
                self.forbidden.add(pair)
        self._validate_pairs(self.forbidden)
        self.forbidden_hash = self._pair_hash(self.forbidden)

    def _canonical(self, source: int, target: int) -> tuple[int, int] | None:
        if self.same_node_space and source == target:
            return None
        if not self.directed and self.same_node_space and source > target:
            return target, source
        return source, target

    def _validate_pairs(self, pairs: Iterable[tuple[int, int]]) -> None:
        for source, target in pairs:
            if not 0 <= source < self.source_count or not 0 <= target < self.target_count:
                raise ValueError("an edge endpoint is outside its relation node space")

    @staticmethod
    def _pair_hash(pairs: Iterable[tuple[int, int]]) -> str:
        digest = hashlib.sha256()
        for source, target in sorted(pairs):
            digest.update(source.to_bytes(8, "little", signed=False))
            digest.update(target.to_bytes(8, "little", signed=False))
        return digest.hexdigest()

    @property
    def total_candidates(self) -> int:
        if self.same_node_space and not self.directed:
            return self.source_count * (self.source_count - 1) // 2
        total = self.source_count * self.target_count
        return total - self.source_count if self.same_node_space else total

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "gfm.causal-exact-sampler/1.0",
            "sourceCount": self.source_count,
            "targetCount": self.target_count,
            "cutoffTime": self.cutoff_time,
            "seed": self.seed,
            "directed": self.directed,
            "sameNodeSpace": self.same_node_space,
            "forbiddenHash": self.forbidden_hash,
            "drawCount": self.draw_count,
            "bitGeneratorState": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = {
            "schemaVersion": "gfm.causal-exact-sampler/1.0",
            "sourceCount": self.source_count,
            "targetCount": self.target_count,
            "cutoffTime": self.cutoff_time,
            "seed": self.seed,
            "directed": self.directed,
            "sameNodeSpace": self.same_node_space,
            "forbiddenHash": self.forbidden_hash,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("causal negative sampler state identity mismatch")
        self._rng.bit_generator.state = deepcopy(state["bitGeneratorState"])
        self.draw_count = int(state["drawCount"])

    def sample(self, count: int, *, current_positive_edges: Tensor | None = None) -> Tensor:
        count = int(count)
        if count < 0:
            raise ValueError("negative sample count cannot be negative")
        additional: set[tuple[int, int]] = set()
        if current_positive_edges is not None:
            for source, target in _edge_pairs(
                current_positive_edges, name="current_positive_edges"
            ):
                pair = self._canonical(source, target)
                if pair is not None:
                    additional.add(pair)
        self._validate_pairs(additional)
        forbidden = self.forbidden | additional
        available = self.total_candidates - len(forbidden)
        if count > available:
            raise ValueError(f"requested {count} negatives but only {available} are available")
        if count == 0:
            return torch.empty((2, 0), dtype=torch.long)

        selected: list[tuple[int, int]] = []
        selected_set: set[tuple[int, int]] = set()
        budget = self.draw_count + max(2048, count * 48)
        while len(selected) < count and self.draw_count < budget:
            batch_size = min(max(64, (count - len(selected)) * 4), 65536)
            sources = self._rng.integers(0, self.source_count, size=batch_size)
            targets = self._rng.integers(0, self.target_count, size=batch_size)
            self.draw_count += batch_size
            for raw_source, raw_target in zip(sources, targets, strict=True):
                pair = self._canonical(int(raw_source), int(raw_target))
                if pair is None or pair in forbidden or pair in selected_set:
                    continue
                selected.append(pair)
                selected_set.add(pair)
                if len(selected) == count:
                    break

        if len(selected) < count:
            for source in range(self.source_count):
                for target in range(self.target_count):
                    pair = self._canonical(source, target)
                    if pair is None or pair in forbidden or pair in selected_set:
                        continue
                    selected.append(pair)
                    selected_set.add(pair)
                    if len(selected) == count:
                        break
                if len(selected) == count:
                    break
        return torch.tensor(selected, dtype=torch.long).t().contiguous()


@dataclass(frozen=True)
class MixedNegativeSample:
    """Query-major mixed negatives with requested and actual component audit."""

    edge_index: Tensor
    component_labels: tuple[str, ...]
    requested_component_counts: Mapping[str, int]
    actual_component_counts: Mapping[str, int]
    negatives_per_positive: int
    fallback_events: tuple[str, ...]
    future_unseen_candidate_count: int = 0

    def __post_init__(self) -> None:
        count = int(self.edge_index.shape[1])
        requested = dict(self.requested_component_counts)
        actual = dict(self.actual_component_counts)
        if (
            self.edge_index.dtype != torch.long
            or self.edge_index.ndim != 2
            or self.edge_index.shape[0] != 2
            or len(self.component_labels) != count
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*requested.values(), *actual.values())
            )
            or sum(requested.values()) != count
            or sum(actual.values()) != count
            or dict(Counter(self.component_labels)) != actual
            or self.negatives_per_positive < 1
            or count % self.negatives_per_positive
            or isinstance(self.future_unseen_candidate_count, bool)
            or self.future_unseen_candidate_count != 0
        ):
            raise ValueError("mixed negative sample audit does not align with its edges")


def _metadata_array(
    values: Tensor | Sequence[int] | None, *, name: str, count: int
) -> np.ndarray | None:
    if values is None:
        return None
    result = np.asarray(torch.as_tensor(values, dtype=torch.long).cpu(), dtype=np.int64).reshape(-1)
    if result.shape[0] != count:
        raise ValueError(f"{name} must contain one integer per target node")
    return result


def _array_hash(value: np.ndarray | None) -> str | None:
    if value is None:
        return None
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("ascii"))
    digest.update(value.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


class CausalMixedNegativeSampler:
    """Deterministic 50/25/25 hard, degree-matched and uniform exact sampler.

    Negatives are target corruptions arranged query-major, so flat model scores
    reshape directly to ``[positive_count, negatives_per_positive]``.  All
    pools are derived only from the explicitly visible graph.  A depleted hard
    or degree pool fails by default; a caller may allow a *single batch*
    fallback, which is relabelled and reported rather than silently counted as
    the requested component.
    """

    def __init__(
        self,
        *,
        source_count: int,
        target_count: int,
        visible_edge_index: Tensor,
        visible_edge_time: Tensor,
        cutoff_time: float,
        seed: int,
        directed: bool,
        same_node_space: bool = True,
        node_types: Tensor | Sequence[int] | None = None,
        topic_groups: Tensor | Sequence[int] | None = None,
    ) -> None:
        self.exact = CausalExactNegativeSampler(
            source_count=source_count,
            target_count=target_count,
            visible_edge_index=visible_edge_index,
            visible_edge_time=visible_edge_time,
            cutoff_time=cutoff_time,
            seed=seed,
            directed=directed,
            same_node_space=same_node_space,
        )
        self.source_count = source_count
        self.target_count = target_count
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.draw_count = 0
        self.node_types = _metadata_array(node_types, name="node_types", count=target_count)
        self.topic_groups = _metadata_array(
            topic_groups, name="topic_groups", count=target_count
        )
        pairs = _edge_pairs(visible_edge_index, name="visible_edge_index")
        # A negative tail is eligible only after it has appeared in the
        # cutoff-visible graph.  Positive endpoints are deliberately absent
        # from this inventory unless they also have genuine history: using all
        # local batch nodes would turn other future targets into easy in-batch
        # negatives and violate the point-in-time sampling contract.
        visible_targets = {
            endpoint
            for source, target in pairs
            for endpoint in ((source, target) if same_node_space else (target,))
        }
        self._visible_targets = np.asarray(sorted(visible_targets), dtype=np.int64)
        self._visible_target_mask = np.zeros(target_count, dtype=np.bool_)
        self._visible_target_mask[self._visible_targets] = True
        # Negative endpoint compatibility is a hard relation constraint, not
        # merely one possible source of "hard" candidates.  Preserve unknown
        # (negative) type ids here as real compatibility classes as well: if a
        # positive target has an unknown type, its corruptions must still have
        # that same value rather than escaping into another node type.
        self._type_groups = self._build_groups(
            self.node_types,
            include_negative=True,
            eligible=self._visible_targets,
        )
        self._topic_groups = self._build_groups(
            self.topic_groups,
            eligible=self._visible_targets,
        )
        target_degree = np.zeros(target_count, dtype=np.int64)
        for source, target in pairs:
            target_degree[target] += 1
            if same_node_space:
                target_degree[source] += 1
        self._degree_bucket = np.floor(np.log2(target_degree + 1)).astype(np.int64)
        self._degree_groups = self._build_groups(
            self._degree_bucket,
            eligible=self._visible_targets,
        )
        self._neighbor_offsets, self._neighbor_values = self._build_visible_csr(
            pairs, same_node_space=same_node_space
        )

    @staticmethod
    def _build_groups(
        values: np.ndarray | None,
        *,
        include_negative: bool = False,
        eligible: np.ndarray | None = None,
    ) -> dict[int, np.ndarray]:
        if values is None:
            return {}
        candidates = (
            np.arange(values.shape[0], dtype=np.int64)
            if eligible is None
            else np.asarray(eligible, dtype=np.int64).reshape(-1)
        )
        groups: dict[int, np.ndarray] = {}
        for group in np.unique(values[candidates]):
            if include_negative or int(group) >= 0:
                groups[int(group)] = candidates[values[candidates] == group]
        return groups

    def _compatible_targets(self, positive_target: int) -> np.ndarray:
        """Return the only legal target universe for a typed positive query."""

        if self.node_types is None:
            return self._visible_targets
        return self._type_groups.get(
            int(self.node_types[positive_target]), np.empty(0, dtype=np.int64)
        )

    def _restrict_to_compatible_type(
        self, pool: np.ndarray, *, positive_target: int
    ) -> np.ndarray:
        if not pool.size:
            return pool
        visible = pool[self._visible_target_mask[pool]]
        if self.node_types is None or not visible.size:
            return visible
        target_type = int(self.node_types[positive_target])
        return visible[self.node_types[visible] == target_type]

    def _build_visible_csr(
        self, pairs: Sequence[tuple[int, int]], *, same_node_space: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        if not same_node_space:
            return np.zeros(self.source_count + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
        sources: list[int] = []
        targets: list[int] = []
        for source, target in pairs:
            sources.extend((source, target))
            targets.extend((target, source))
        if not sources:
            return np.zeros(self.source_count + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
        source_array = np.asarray(sources, dtype=np.int64)
        target_array = np.asarray(targets, dtype=np.int64)
        order = np.argsort(source_array, kind="stable")
        counts = np.bincount(source_array, minlength=self.source_count)
        offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
        return offsets, target_array[order]

    def _neighbors(self, node: int) -> np.ndarray:
        start, stop = self._neighbor_offsets[node : node + 2]
        return self._neighbor_values[int(start) : int(stop)]

    def _two_three_hop(self, source: int) -> np.ndarray:
        if not self.exact.same_node_space:
            return np.empty(0, dtype=np.int64)
        visited = {source}
        frontier = set(int(value) for value in self._neighbors(source))
        visited.update(frontier)
        hard: set[int] = set()
        for _depth in (2, 3):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(int(value) for value in self._neighbors(node))
            next_frontier.difference_update(visited)
            hard.update(next_frontier)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return np.asarray(sorted(hard), dtype=np.int64)

    @staticmethod
    def _requested_counts(total: int) -> dict[str, int]:
        raw = {"hard": total * 0.5, "degree_matched": total * 0.25, "uniform": total * 0.25}
        counts = {name: int(value) for name, value in raw.items()}
        remaining = total - sum(counts.values())
        priority = sorted(raw, key=lambda name: (-(raw[name] - counts[name]), name))
        for name in priority[:remaining]:
            counts[name] += 1
        return counts

    def _canonical_candidate(
        self,
        source: int,
        target: int,
        positive_target: int,
        forbidden: set[tuple[int, int]],
        selected: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if not self._visible_target_mask[target] or (
            self.node_types is not None
            and
            int(self.node_types[target]) != int(self.node_types[positive_target])
        ):
            return None
        pair = self.exact._canonical(source, target)
        if pair is None or pair in forbidden or pair in selected:
            return None
        return pair

    def _choose_from_pools(
        self,
        pools: Sequence[np.ndarray],
        *,
        source: int,
        positive_target: int,
        forbidden: set[tuple[int, int]],
        selected: set[tuple[int, int]],
    ) -> tuple[int, tuple[int, int]] | None:
        available_pools = [
            compatible
            for pool in pools
            if (
                compatible := self._restrict_to_compatible_type(
                    pool, positive_target=positive_target
                )
            ).size
        ]
        for _ in range(max(32, len(available_pools) * 16)):
            if not available_pools:
                break
            pool = available_pools[int(self._rng.integers(0, len(available_pools)))]
            target = int(pool[int(self._rng.integers(0, pool.size))])
            self.draw_count += 1
            pair = self._canonical_candidate(
                source, target, positive_target, forbidden, selected
            )
            if pair is not None:
                return target, pair
        for pool in available_pools:
            for raw_target in pool:
                target = int(raw_target)
                pair = self._canonical_candidate(
                    source, target, positive_target, forbidden, selected
                )
                if pair is not None:
                    return target, pair
        return None

    def _uniform_candidate(
        self,
        *,
        source: int,
        positive_target: int,
        forbidden: set[tuple[int, int]],
        selected: set[tuple[int, int]],
    ) -> tuple[int, tuple[int, int]] | None:
        compatible_targets = self._compatible_targets(positive_target)
        for _ in range(256):
            if not compatible_targets.size:
                break
            target = int(
                compatible_targets[int(self._rng.integers(0, compatible_targets.size))]
            )
            self.draw_count += 1
            pair = self._canonical_candidate(
                source, target, positive_target, forbidden, selected
            )
            if pair is not None:
                return target, pair
        for raw_target in compatible_targets:
            target = int(raw_target)
            pair = self._canonical_candidate(
                source, target, positive_target, forbidden, selected
            )
            if pair is not None:
                return target, pair
        return None

    def _hard_pools(
        self,
        source: int,
        positive_target: int,
        hop_cache: dict[int, np.ndarray],
    ) -> list[np.ndarray]:
        """Return semantically hard pools inside the already enforced type universe.

        Endpoint type is a relation-validity constraint, not evidence that a
        candidate is difficult.  Treating the complete type class as a hard
        pool silently degraded the requested 50% hard mixture into uniform
        same-type sampling whenever topic or structural evidence was absent.
        """

        pools: list[np.ndarray] = []
        if self.topic_groups is not None:
            group = int(self.topic_groups[positive_target])
            if group >= 0:
                pools.append(self._topic_groups[group])
        if source not in hop_cache:
            hop_cache[source] = self._two_three_hop(source)
        pools.append(hop_cache[source])
        return pools

    def sample(
        self,
        positive_edge_index: Tensor,
        *,
        negatives_per_positive: int,
        allow_batch_fallback: bool = False,
    ) -> MixedNegativeSample:
        positives = _edge_pairs(positive_edge_index, name="positive_edge_index")
        if not positives or negatives_per_positive < 1:
            raise ValueError("mixed sampling requires positives and negatives_per_positive >= 1")
        self.exact._validate_pairs(positives)
        forbidden = set(self.exact.forbidden)
        for source, target in positives:
            pair = self.exact._canonical(source, target)
            if pair is not None:
                forbidden.add(pair)

        per_query_requested = self._requested_counts(int(negatives_per_positive))
        requested = {
            name: count * len(positives) for name, count in per_query_requested.items()
        }
        labels: list[str] = []
        for _query in positives:
            query_labels = [
                name for name, count in per_query_requested.items() for _ in range(count)
            ]
            labels.extend(query_labels[index] for index in self._rng.permutation(len(query_labels)))
        # Negatives must be unique inside one ranking query.  The same exact
        # non-edge may legitimately be sampled for another query (especially
        # when two positives share a source); globally forbidding it can
        # exhaust a typed pool at formal batch sizes without improving label
        # correctness.
        selected_by_query: dict[int, set[tuple[int, int]]] = {
            index: set() for index in range(len(positives))
        }
        edges: list[tuple[int, int]] = []
        actual_labels: list[str] = []
        fallback_events: list[str] = []
        hop_cache: dict[int, np.ndarray] = {}

        for position, requested_label in enumerate(labels):
            query = position // negatives_per_positive
            source, positive_target = positives[query]
            selected = selected_by_query[query]
            candidate: tuple[int, tuple[int, int]] | None
            if requested_label == "hard":
                candidate = self._choose_from_pools(
                    self._hard_pools(source, positive_target, hop_cache),
                    source=source,
                    positive_target=positive_target,
                    forbidden=forbidden,
                    selected=selected,
                )
            elif requested_label == "degree_matched":
                degree_group = int(self._degree_bucket[positive_target])
                candidate = self._choose_from_pools(
                    [self._degree_groups.get(degree_group, np.empty(0, dtype=np.int64))],
                    source=source,
                    positive_target=positive_target,
                    forbidden=forbidden,
                    selected=selected,
                )
            else:
                candidate = self._uniform_candidate(
                    source=source,
                    positive_target=positive_target,
                    forbidden=forbidden,
                    selected=selected,
                )

            actual_label = requested_label
            if candidate is None and requested_label != "uniform":
                if not allow_batch_fallback:
                    raise ValueError(
                        f"{requested_label} negative pool is empty for query {query}; "
                        "explicit batch fallback was not allowed"
                    )
                candidate = self._uniform_candidate(
                    source=source,
                    positive_target=positive_target,
                    forbidden=forbidden,
                    selected=selected,
                )
                actual_label = f"{requested_label}_fallback_uniform"
                fallback_events.append(f"query={query}:{requested_label}->uniform")
            if candidate is None:
                raise ValueError(f"no exact negative remains for query {query}")
            target, canonical = candidate
            selected.add(canonical)
            edges.append((source, target))
            actual_labels.append(actual_label)

        return MixedNegativeSample(
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            component_labels=tuple(actual_labels),
            requested_component_counts=requested,
            actual_component_counts=dict(Counter(actual_labels)),
            negatives_per_positive=int(negatives_per_positive),
            fallback_events=tuple(fallback_events),
            future_unseen_candidate_count=sum(
                int(not self._visible_target_mask[target]) for _, target in edges
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "gfm.causal-mixed-sampler/1.0",
            "exactSamplerState": self.exact.state_dict(),
            "nodeTypesHash": _array_hash(self.node_types),
            "topicGroupsHash": _array_hash(self.topic_groups),
            "degreeBucketHash": _array_hash(self._degree_bucket),
            "visibleTargetsHash": _array_hash(self._visible_targets),
            "drawCount": self.draw_count,
            "bitGeneratorState": deepcopy(self._rng.bit_generator.state),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = {
            "schemaVersion": "gfm.causal-mixed-sampler/1.0",
            "nodeTypesHash": _array_hash(self.node_types),
            "topicGroupsHash": _array_hash(self.topic_groups),
            "degreeBucketHash": _array_hash(self._degree_bucket),
            "visibleTargetsHash": _array_hash(self._visible_targets),
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise ValueError("causal mixed sampler state identity mismatch")
        self.exact.load_state_dict(dict(state["exactSamplerState"]))
        draw_count = int(state["drawCount"])
        if draw_count < 0:
            raise ValueError("causal mixed sampler draw count cannot be negative")
        self.draw_count = draw_count
        self._rng.bit_generator.state = deepcopy(state["bitGeneratorState"])


class RoundRobinDomainScheduler:
    """Deterministic domain choice with serializable cursor state."""

    def __init__(self, domains: Iterable[str]) -> None:
        self.domains = tuple(domains)
        if not self.domains or len(set(self.domains)) != len(self.domains):
            raise ValueError("round-robin domains must be nonempty and unique")
        self.cursor = 0
        self.steps = 0

    def next_domain(self, active_domains: Iterable[str] | None = None) -> str:
        active = set(active_domains if active_domains is not None else self.domains)
        unknown = active.difference(self.domains)
        if unknown:
            raise ValueError(f"unknown active domains: {sorted(unknown)}")
        if not active:
            raise StopIteration("no active training domains remain")
        for _ in range(len(self.domains)):
            domain = self.domains[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.domains)
            if domain in active:
                self.steps += 1
                return domain
        raise StopIteration("no active training domains remain")

    def state_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": "gfm.round-robin-domains/1.0",
            "domains": self.domains,
            "cursor": self.cursor,
            "steps": self.steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schemaVersion") != "gfm.round-robin-domains/1.0" or tuple(
            state.get("domains", ())
        ) != self.domains:
            raise ValueError("round-robin scheduler state identity mismatch")
        cursor = int(state["cursor"])
        steps = int(state["steps"])
        if not 0 <= cursor < len(self.domains) or steps < 0:
            raise ValueError("round-robin scheduler state is invalid")
        self.cursor = cursor
        self.steps = steps
