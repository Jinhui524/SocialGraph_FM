"""Bounded deterministic graph-preview projections for SocialGraph-FM Governance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .materialize import OnlineInferenceData

ProjectionPreset = Literal["overview", "relation", "evidence", "groups"]
_LIMITS = {
    "overview": (3_000, 12_000),
    "relation": (80, 160),
    "evidence": (60, 120),
}
_DEFAULTS = {
    "overview": (120, 240),
    "relation": (80, 160),
    "evidence": (60, 120),
}
_RECIPES = {
    "overview": "governance-preview-overview-risk-degree/1.0",
    "relation": "governance-preview-relation-weight/1.0",
    "evidence": "governance-preview-evidence-bfs/1.0",
    "groups": "governance-preview-groups-aggregate/1.0",
}


class ProjectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    preset: ProjectionPreset
    node_budget: int | None = Field(default=None, alias="nodeBudget")
    edge_budget: int | None = Field(default=None, alias="edgeBudget")
    relation: Literal["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"] | None = None
    anchor_node_ids: tuple[str, ...] = Field(default=(), alias="anchorNodeIds")
    group_budget: int | None = Field(default=None, alias="groupBudget")

    @field_validator("anchor_node_ids", mode="before")
    @classmethod
    def validate_anchor_inventory(cls, value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise ValueError("anchorNodeIds must be a JSON string array")
        return tuple(value)

    @model_validator(mode="after")
    def validate_budget(self) -> ProjectionRequest:
        if self.preset == "groups":
            if self.node_budget is not None or self.edge_budget is not None:
                raise ValueError("groups preset accepts groupBudget only")
            if self.relation is not None or self.anchor_node_ids:
                raise ValueError("groups preset does not accept relation or anchors")
            if self.group_budget is not None and not 1 <= self.group_budget <= 24:
                raise ValueError("groupBudget must be between 1 and 24")
            return self
        maximum_nodes, maximum_edges = _LIMITS[self.preset]
        if self.node_budget is not None and not 1 <= self.node_budget <= maximum_nodes:
            raise ValueError(f"nodeBudget exceeds the {self.preset} preset limit")
        if self.edge_budget is not None and not 0 <= self.edge_budget <= maximum_edges:
            raise ValueError(f"edgeBudget exceeds the {self.preset} preset limit")
        if self.group_budget is not None:
            raise ValueError("groupBudget is valid only for the groups preset")
        if self.preset == "relation" and self.relation is None:
            raise ValueError("relation preset requires one relation modality")
        if self.preset != "relation" and self.relation is not None:
            raise ValueError("relation is valid only for the relation preset")
        if self.preset != "evidence" and self.anchor_node_ids:
            raise ValueError("anchorNodeIds are valid only for the evidence preset")
        if len(self.anchor_node_ids) > 8 or len(set(self.anchor_node_ids)) != len(
            self.anchor_node_ids
        ):
            raise ValueError("anchorNodeIds must contain at most eight unique ids")
        return self

    @property
    def effective_node_budget(self) -> int:
        return self.node_budget or _DEFAULTS[self.preset][0]

    @property
    def effective_edge_budget(self) -> int:
        if self.edge_budget is not None:
            return self.edge_budget
        return _DEFAULTS[self.preset][1]

    @property
    def effective_group_budget(self) -> int:
        return self.group_budget or 12


@dataclass(frozen=True)
class ProjectionSelection:
    selected_order: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    groups: tuple[dict[str, Any], ...]
    supernodes: tuple[dict[str, Any], ...]
    aggregate_edges: tuple[dict[str, Any], ...]
    budgets: Mapping[str, int]
    source_counts: Mapping[str, int]
    selection_recipe_id: str


def _fused_edges(data: OnlineInferenceData) -> tuple[tuple[int, int], ...]:
    edge_index = np.asarray(data.edge_index)
    return tuple(
        (int(source), int(target))
        for source, target in zip(edge_index[0], edge_index[1], strict=True)
        if int(source) < int(target)
    )


def _node_order(
    data: OnlineInferenceData, arrays: Mapping[str, np.ndarray] | None
) -> tuple[int, ...]:
    if arrays is not None:
        return tuple(int(value) for value in arrays["rank_order"])
    degrees = np.diff(np.asarray(data.arrays["fused_indptr"]))
    return tuple(
        sorted(
            range(len(data.node_ids)),
            key=lambda index: (-int(degrees[index]), data.node_ids[index]),
        )
    )


def _edge_order(
    edges: Sequence[tuple[int, int]],
    data: OnlineInferenceData,
    arrays: Mapping[str, np.ndarray] | None,
) -> tuple[tuple[int, int], ...]:
    if arrays is None:
        return tuple(
            sorted(edges, key=lambda pair: (data.node_ids[pair[0]], data.node_ids[pair[1]]))
        )
    scores = np.asarray(arrays["scores"])
    return tuple(
        sorted(
            edges,
            key=lambda pair: (
                -max(float(scores[pair[0]]), float(scores[pair[1]])),
                data.node_ids[pair[0]],
                data.node_ids[pair[1]],
            ),
        )
    )


def _overview(
    data: OnlineInferenceData,
    request: ProjectionRequest,
    arrays: Mapping[str, np.ndarray] | None,
    *,
    threshold: float | None,
) -> ProjectionSelection:
    budget = request.effective_node_budget
    selected_values: list[int] = []
    selected_set: set[int] = set()
    source_counts: dict[str, int] = {
        "groupRepresentatives": 0,
        "componentRepresentatives": 0,
        "bridgeEndpoints": 0,
        "isolates": 0,
        "highRisk": 0,
        "reviewRisk": 0,
        "lowRisk": 0,
        "highDegree": 0,
        "midDegree": 0,
        "lowDegree": 0,
        "rankedFill": 0,
    }

    def take(values: Sequence[int], limit: int, source: str) -> None:
        count = 0
        for value in values:
            if len(selected_values) >= budget or count >= limit:
                break
            if value in selected_set:
                continue
            selected_set.add(value)
            selected_values.append(value)
            count += 1
        source_counts[source] = count

    degrees = np.diff(np.asarray(data.arrays["fused_indptr"]))
    edges = _fused_edges(data)
    isolate_quota = min(3, max(1, budget // 20))
    representative_quota = max(1, budget // 10)
    bridge_quota = max(1, budget // 10)
    ranked = _node_order(data, arrays)
    if arrays is not None:
        if threshold is None:
            raise ValueError("scored overview requires a model threshold")
        scores = np.asarray(arrays["scores"])
        communities = np.asarray(arrays["community_ids"])
        representatives: list[int] = []
        for community in sorted({int(value) for value in communities}):
            members = [index for index in ranked if int(communities[index]) == community]
            if len(members) >= 2:
                representatives.append(members[0])
        cross_counts: dict[int, int] = {}
        for source, target in edges:
            if int(communities[source]) != int(communities[target]):
                cross_counts[source] = cross_counts.get(source, 0) + 1
                cross_counts[target] = cross_counts.get(target, 0) + 1
        bridges = sorted(
            cross_counts,
            key=lambda index: (-cross_counts[index], -float(scores[index]), data.node_ids[index]),
        )
        isolates = [index for index in ranked if bool(data.structure_missing[index])]
        take(representatives, representative_quota, "groupRepresentatives")
        take(bridges, bridge_quota, "bridgeEndpoints")
        take(isolates, isolate_quota, "isolates")
        remaining = budget - len(selected_values)
        band_quotas = {
            "highRisk": (remaining * 5) // 10,
            "reviewRisk": (remaining * 3) // 10,
        }
        band_quotas["lowRisk"] = remaining - sum(band_quotas.values())
        high = [index for index in ranked if float(scores[index]) >= threshold]
        review = [
            index
            for index in ranked
            if max(0.0, threshold - 0.15) <= float(scores[index]) < threshold
        ]
        low = [index for index in ranked if float(scores[index]) < max(0.0, threshold - 0.15)]
        take(high, band_quotas["highRisk"], "highRisk")
        take(review, band_quotas["reviewRisk"], "reviewRisk")
        take(low, band_quotas["lowRisk"], "lowRisk")
    else:
        neighbors = [list[int]() for _ in data.node_ids]
        for source, target in edges:
            neighbors[source].append(target)
            neighbors[target].append(source)
        discovery = [-1] * len(data.node_ids)
        low_link = [-1] * len(data.node_ids)
        parent = [-1] * len(data.node_ids)
        component_representatives: list[int] = []
        bridge_nodes: set[int] = set()
        moment = 0
        for root in range(len(data.node_ids)):
            if discovery[root] >= 0:
                continue
            component: list[int] = []
            discovery[root] = low_link[root] = moment
            moment += 1
            stack: list[tuple[int, int]] = [(root, 0)]
            while stack:
                node, position = stack[-1]
                if position < len(neighbors[node]):
                    target = neighbors[node][position]
                    stack[-1] = (node, position + 1)
                    if discovery[target] < 0:
                        parent[target] = node
                        discovery[target] = low_link[target] = moment
                        moment += 1
                        stack.append((target, 0))
                    elif target != parent[node]:
                        low_link[node] = min(low_link[node], discovery[target])
                    continue
                stack.pop()
                component.append(node)
                parent_node = parent[node]
                if parent_node >= 0:
                    low_link[parent_node] = min(low_link[parent_node], low_link[node])
                    if low_link[node] > discovery[parent_node]:
                        bridge_nodes.update((parent_node, node))
            component_representatives.append(
                min(component, key=lambda index: (-int(degrees[index]), data.node_ids[index]))
            )
        bridge_order = sorted(
            bridge_nodes, key=lambda index: (-int(degrees[index]), data.node_ids[index])
        )
        isolates = [index for index in ranked if int(degrees[index]) == 0]
        take(component_representatives, representative_quota, "componentRepresentatives")
        take(bridge_order, bridge_quota, "bridgeEndpoints")
        take(isolates, isolate_quota, "isolates")
        nonisolates = [index for index in ranked if int(degrees[index]) > 0]
        first = (len(nonisolates) + 2) // 3
        second = (2 * len(nonisolates) + 2) // 3
        strata = (
            ("highDegree", nonisolates[:first]),
            ("midDegree", nonisolates[first:second]),
            ("lowDegree", nonisolates[second:]),
        )
        remaining = budget - len(selected_values)
        quotas = (remaining * 5 // 10, remaining * 3 // 10)
        final_quota = remaining - sum(quotas)
        for (name, values), quota in zip(strata, (*quotas, final_quota), strict=True):
            take(values, quota, name)
    before_fill = len(selected_values)
    take(ranked, budget, "rankedFill")
    source_counts["rankedFill"] = len(selected_values) - before_fill
    selected = tuple(selected_values)
    selected_set = set(selected)
    induced = [pair for pair in edges if pair[0] in selected_set and pair[1] in selected_set]
    preview_edges = _edge_order(induced, data, arrays)[: request.effective_edge_budget]
    return ProjectionSelection(
        selected_order=selected,
        edges=preview_edges,
        groups=(),
        supernodes=(),
        aggregate_edges=(),
        budgets={"nodes": request.effective_node_budget, "edges": request.effective_edge_budget},
        source_counts=source_counts,
        selection_recipe_id=_RECIPES[request.preset],
    )


def _relation(data: OnlineInferenceData, request: ProjectionRequest) -> ProjectionSelection:
    assert request.relation is not None
    token = request.relation.lower()
    indptr = np.asarray(data.arrays[f"relation_{token}_indptr"])
    indices = np.asarray(data.arrays[f"relation_{token}_indices"])
    weights = np.asarray(data.arrays[f"relation_{token}_weights"])
    candidates: list[tuple[float, int, int]] = []
    for source in range(len(data.node_ids)):
        for position in range(int(indptr[source]), int(indptr[source + 1])):
            target = int(indices[position])
            if source < target:
                candidates.append((float(weights[position]), source, target))
    candidates.sort(key=lambda item: (-item[0], data.node_ids[item[1]], data.node_ids[item[2]]))
    selected: list[int] = []
    selected_set: set[int] = set()
    edges: list[tuple[int, int]] = []
    for _weight, source, target in candidates:
        missing = int(source not in selected_set) + int(target not in selected_set)
        if len(selected) + missing > request.effective_node_budget:
            continue
        for endpoint in (source, target):
            if endpoint not in selected_set:
                selected_set.add(endpoint)
                selected.append(endpoint)
        edges.append((source, target))
        if len(edges) >= request.effective_edge_budget:
            break
    return ProjectionSelection(
        selected_order=tuple(selected),
        edges=tuple(edges),
        groups=(),
        supernodes=(),
        aggregate_edges=(),
        budgets={"nodes": request.effective_node_budget, "edges": request.effective_edge_budget},
        source_counts={"relationEndpoints": len(selected), "relationEdges": len(edges)},
        selection_recipe_id=_RECIPES[request.preset],
    )


def _evidence(
    data: OnlineInferenceData,
    request: ProjectionRequest,
    arrays: Mapping[str, np.ndarray] | None,
) -> ProjectionSelection:
    lookup = {node_id: index for index, node_id in enumerate(data.node_ids)}
    try:
        anchors = [lookup[value] for value in request.anchor_node_ids]
    except KeyError as error:
        raise ValueError("anchorNodeIds contains an unknown node") from error
    if not anchors:
        anchors = [int(_node_order(data, arrays)[0])]
    if len(anchors) > request.effective_node_budget:
        raise ValueError("nodeBudget is smaller than the anchor inventory")
    indptr = np.asarray(data.arrays["fused_indptr"])
    indices = np.asarray(data.arrays["fused_indices"])
    order = _node_order(data, arrays)
    rank = {node: position for position, node in enumerate(order)}
    selected = list(anchors)
    selected_set = set(anchors)
    frontier = list(anchors)
    while frontier and len(selected) < request.effective_node_budget:
        following: set[int] = set()
        for source in frontier:
            following.update(
                int(value) for value in indices[int(indptr[source]) : int(indptr[source + 1])]
            )
        candidates = sorted(
            following - selected_set,
            key=lambda index: (rank[index], data.node_ids[index]),
        )
        accepted = candidates[: request.effective_node_budget - len(selected)]
        selected.extend(accepted)
        selected_set.update(accepted)
        frontier = accepted
    induced = [
        pair for pair in _fused_edges(data) if pair[0] in selected_set and pair[1] in selected_set
    ]
    edges = _edge_order(induced, data, arrays)[: request.effective_edge_budget]
    return ProjectionSelection(
        selected_order=tuple(selected),
        edges=edges,
        groups=(),
        supernodes=(),
        aggregate_edges=(),
        budgets={"nodes": request.effective_node_budget, "edges": request.effective_edge_budget},
        source_counts={
            "anchors": len(anchors),
            "neighbors": len(selected) - len(anchors),
            "inducedEdges": len(edges),
        },
        selection_recipe_id=_RECIPES[request.preset],
    )


def select_projection(
    data: OnlineInferenceData,
    request: ProjectionRequest,
    *,
    arrays: Mapping[str, np.ndarray] | None,
    groups: Sequence[Mapping[str, Any]] = (),
    threshold: float | None = None,
) -> ProjectionSelection:
    """Select a stable bounded projection without changing graph or inference artifacts."""

    if request.preset == "overview":
        return _overview(data, request, arrays, threshold=threshold)
    if request.preset == "relation":
        return _relation(data, request)
    if request.preset == "evidence":
        return _evidence(data, request, arrays)
    if arrays is None:
        raise ValueError("groups projection requires a completed run")
    selected_groups = tuple(dict(value) for value in groups[: request.effective_group_budget])
    group_ids = {str(value["groupId"]) for value in selected_groups}
    supernodes = tuple(
        {
            "id": str(value["groupId"]),
            "label": str(value["groupId"]),
            "degree": 0,
            "structureMissing": False,
            "score": float(value["p90Risk"]),
            "riskBand": None,
            "groupId": str(value["groupId"]),
            "memberCount": int(value["memberCount"]),
            "riskP90": float(value["p90Risk"]),
            "meanRisk": float(value["averageRisk"]),
            "aggregate": True,
        }
        for value in selected_groups
    )
    community_ids = np.asarray(arrays["community_ids"])
    aggregate: dict[tuple[str, str], int] = {}
    for source, target in _fused_edges(data):
        left = f"group-{int(community_ids[source]) + 1}"
        right = f"group-{int(community_ids[target]) + 1}"
        if left == right or left not in group_ids or right not in group_ids:
            continue
        pair = (left, right) if left < right else (right, left)
        aggregate[pair] = aggregate.get(pair, 0) + 1
    aggregate_edges = tuple(
        {
            "id": f"group-edge-{left}-{right}",
            "source": left,
            "target": right,
            "modalities": [],
            "factual": True,
            "aggregate": True,
            "count": count,
            "weight": count,
        }
        for (left, right), count in sorted(
            aggregate.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:160]
    )
    return ProjectionSelection(
        selected_order=(),
        edges=(),
        groups=selected_groups,
        supernodes=supernodes,
        aggregate_edges=aggregate_edges,
        budgets={"groups": request.effective_group_budget, "maxGroups": 24},
        source_counts={
            "groupSupernodes": len(supernodes),
            "interGroupEdges": len(aggregate_edges),
        },
        selection_recipe_id=_RECIPES[request.preset],
    )


__all__ = [
    "ProjectionRequest",
    "ProjectionSelection",
    "select_projection",
]
