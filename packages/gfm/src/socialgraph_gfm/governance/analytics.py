"""Deterministic, explicitly derived governance candidates for online runs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import MODALITIES
from .inference import OnlineInferenceOutputs
from .materialize import OnlineInferenceData


@dataclass(frozen=True)
class DerivedAnalytics:
    groups: tuple[dict[str, Any], ...]
    relation_arrays: Mapping[str, np.ndarray]
    links: tuple[dict[str, Any], ...]
    community_ids: np.ndarray


def _undirected_edges(data: OnlineInferenceData) -> tuple[tuple[int, int], ...]:
    edge_index = np.asarray(data.edge_index)
    return tuple(
        (int(source), int(target))
        for source, target in zip(edge_index[0], edge_index[1], strict=True)
        if int(source) < int(target)
    )


def _communities(
    data: OnlineInferenceData, edges: Sequence[tuple[int, int]], *, seed: int
) -> tuple[tuple[int, ...], ...]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(len(data.node_ids)))
    graph.add_edges_from(edges)
    raw = nx.community.louvain_communities(graph, seed=seed)
    communities = [tuple(sorted(int(value) for value in group)) for group in raw]
    communities.sort(key=lambda group: (group[0], len(group)))
    return tuple(communities)


def _group_documents(
    data: OnlineInferenceData,
    scores: np.ndarray,
    communities: Sequence[Sequence[int]],
) -> tuple[tuple[dict[str, Any], ...], np.ndarray]:
    community_ids = np.full(len(data.node_ids), -1, dtype=np.int32)
    documents: list[dict[str, Any]] = []
    for index, members_sequence in enumerate(communities):
        members = np.asarray(members_sequence, dtype=np.int64)
        community_ids[members] = index
        member_scores = scores[members]
        average = float(member_scores.mean())
        p90 = float(np.percentile(member_scores, 90))
        priority = 0.6 * p90 + 0.4 * average
        if len(members) < 2:
            continue
        modality_counts: dict[str, int] = {}
        member_mask = np.zeros(len(data.node_ids), dtype=np.bool_)
        member_mask[members] = True
        for modality in MODALITIES:
            token = modality.lower()
            indptr = np.asarray(data.arrays[f"relation_{token}_indptr"])
            indices = np.asarray(data.arrays[f"relation_{token}_indices"])
            count = 0
            for source in members:
                start, stop = int(indptr[source]), int(indptr[source + 1])
                targets = indices[start:stop]
                count += int(member_mask[targets].sum())
            modality_counts[modality] = count // 2
        documents.append(
            {
                "groupId": f"group-{index + 1}",
                "memberCount": len(members),
                "memberNodeIds": [data.node_ids[int(value)] for value in members],
                "averageRisk": average,
                "p90Risk": p90,
                "priority": priority,
                "relationCounts": modality_counts,
                "derivation": "0.6 * member risk P90 + 0.4 * member mean risk",
            }
        )
    documents.sort(key=lambda item: (-float(item["priority"]), str(item["groupId"])))
    for rank, document in enumerate(documents, start=1):
        document["rank"] = rank
    return tuple(documents), community_ids


def _relation_arrays(
    data: OnlineInferenceData,
    scores: np.ndarray,
    edges: Sequence[tuple[int, int]],
) -> dict[str, np.ndarray]:
    edge_positions = {pair: index for index, pair in enumerate(edges)}
    modality_mask = np.zeros(len(edges), dtype=np.uint8)
    maximum_percentile = np.zeros(len(edges), dtype=np.float32)
    maximum_weight = np.zeros(len(edges), dtype=np.float64)
    for column, modality in enumerate(MODALITIES):
        token = modality.lower()
        indptr = np.asarray(data.arrays[f"relation_{token}_indptr"])
        indices = np.asarray(data.arrays[f"relation_{token}_indices"])
        weights = np.asarray(data.arrays[f"relation_{token}_weights"])
        undirected_values = [
            float(weights[position])
            for source in range(len(data.node_ids))
            for position in range(int(indptr[source]), int(indptr[source + 1]))
            if source < int(indices[position])
        ]
        ordered = np.sort(np.asarray(undirected_values, dtype=np.float64))
        denominator = max(1, ordered.size)
        for source in range(len(data.node_ids)):
            for position in range(int(indptr[source]), int(indptr[source + 1])):
                target = int(indices[position])
                if source >= target:
                    continue
                row = edge_positions[(source, target)]
                weight = float(weights[position])
                percentile = float(np.searchsorted(ordered, weight, side="right") / denominator)
                modality_mask[row] |= np.uint8(1 << column)
                maximum_percentile[row] = max(maximum_percentile[row], percentile)
                maximum_weight[row] = max(maximum_weight[row], weight)
    sources = np.asarray([pair[0] for pair in edges], dtype=np.int32)
    targets = np.asarray([pair[1] for pair in edges], dtype=np.int32)
    endpoint_risk = ((scores[sources] + scores[targets]) / 2).astype(np.float32)
    diversity = np.asarray(
        [int(value).bit_count() / len(MODALITIES) for value in modality_mask],
        dtype=np.float32,
    )
    priority = (
        0.6 * endpoint_risk + 0.2 * diversity + 0.2 * maximum_percentile
    ).astype(np.float32)
    order = np.lexsort((targets, sources, -priority)).astype(np.int32)
    rank = np.empty(len(edges), dtype=np.int32)
    rank[order] = np.arange(1, len(edges) + 1, dtype=np.int32)
    return {
        "source": sources,
        "target": targets,
        "modality_mask": modality_mask,
        "max_weight": maximum_weight,
        "weight_percentile": maximum_percentile,
        "endpoint_risk": endpoint_risk,
        "priority": priority,
        "rank": rank,
        "order": order,
    }


def _jaccard(neighbors: Sequence[set[int]], source: int, target: int) -> float:
    left, right = neighbors[source], neighbors[target]
    union = len(left | right)
    return 0.0 if not union else len(left & right) / union


def _potential_links(
    data: OnlineInferenceData,
    outputs: OnlineInferenceOutputs,
    communities: Sequence[Sequence[int]],
    edges: Sequence[tuple[int, int]],
) -> tuple[dict[str, Any], ...]:
    edge_set = set(edges)
    neighbors: list[set[int]] = [set() for _ in data.node_ids]
    for source, target in edges:
        neighbors[source].add(target)
        neighbors[target].add(source)
    embeddings = outputs.embeddings.astype(np.float32, copy=False)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, np.finfo(np.float32).eps)
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    for members_sequence in communities:
        if len(members_sequence) < 2:
            continue
        members = np.asarray(members_sequence, dtype=np.int64)
        if members.size > 512:
            member_order = np.lexsort((members, -outputs.scores[members]))[:512]
            members = members[member_order]
        similarities = normalized[members] @ normalized[members].T
        for row, source_value in enumerate(members):
            source = int(source_value)
            local_order = np.argsort(-similarities[row], kind="stable")[:32]
            for local_target in local_order:
                target = int(members[int(local_target)])
                if source == target:
                    continue
                pair = (min(source, target), max(source, target))
                if pair in edge_set or pair in candidates:
                    continue
                cosine = float(np.clip(similarities[row, int(local_target)], -1, 1))
                similarity = (cosine + 1) / 2
                endpoint_risk = float((outputs.scores[source] + outputs.scores[target]) / 2)
                common_neighbor_jaccard = _jaccard(neighbors, source, target)
                priority = (
                    0.6 * similarity
                    + 0.25 * endpoint_risk
                    + 0.15 * common_neighbor_jaccard
                )
                candidates[pair] = {
                    "linkId": f"link-{pair[0]}-{pair[1]}",
                    "source": data.node_ids[pair[0]],
                    "target": data.node_ids[pair[1]],
                    "embeddingCosine": cosine,
                    "embeddingSimilarity": similarity,
                    "endpointRisk": endpoint_risk,
                    "commonNeighborJaccard": common_neighbor_jaccard,
                    "priority": priority,
                    "evidenceRole": "potentialLeadNotFactualEdge",
                    "derivation": (
                        "0.6 * normalized embedding cosine similarity + 0.25 * endpoint risk "
                        "+ 0.15 * common-neighbor Jaccard"
                    ),
                }
    ordered = sorted(
        candidates.values(),
        key=lambda item: (-float(item["priority"]), str(item["linkId"])),
    )
    per_node: defaultdict[str, int] = defaultdict(int)
    accepted: list[dict[str, Any]] = []
    for item in ordered:
        source_id, target_id = str(item["source"]), str(item["target"])
        if per_node[source_id] >= 10 or per_node[target_id] >= 10:
            continue
        per_node[source_id] += 1
        per_node[target_id] += 1
        item["rank"] = len(accepted) + 1
        accepted.append(item)
        if len(accepted) == 500:
            break
    return tuple(accepted)


def derive_analytics(
    data: OnlineInferenceData,
    outputs: OnlineInferenceOutputs,
    *,
    seed: int,
) -> DerivedAnalytics:
    edges = _undirected_edges(data)
    communities = _communities(data, edges, seed=seed)
    groups, community_ids = _group_documents(data, outputs.scores, communities)
    relation_arrays = _relation_arrays(data, outputs.scores, edges)
    links = _potential_links(data, outputs, communities, edges)
    if bool((community_ids < 0).any()) or any(
        not math.isfinite(float(item["priority"])) for item in (*groups, *links)
    ):
        raise ValueError("derived governance analytics are incomplete or non-finite")
    return DerivedAnalytics(
        groups=groups,
        relation_arrays=relation_arrays,
        links=links,
        community_ids=community_ids,
    )


__all__ = ["DerivedAnalytics", "derive_analytics"]
