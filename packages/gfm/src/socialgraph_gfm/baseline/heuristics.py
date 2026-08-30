"""Classical structural link-prediction baselines."""

from __future__ import annotations

import math
from typing import Any, Literal

from .types import edge_pairs

HeuristicName = Literal["cn", "aa", "ra"]


def adjacency_sets(num_nodes: int, message_edges: Any) -> list[set[int]]:
    adjacency: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in edge_pairs(message_edges, name="message_edges"):
        source_i, target_i = int(source), int(target)
        if source_i == target_i:
            continue
        if not (0 <= source_i < num_nodes and 0 <= target_i < num_nodes):
            raise ValueError("message edge contains out-of-range node")
        adjacency[source_i].add(target_i)
        adjacency[target_i].add(source_i)
    return adjacency


def score_heuristic(
    name: HeuristicName,
    *,
    num_nodes: int,
    message_edges: Any,
    candidate_edges: Any,
) -> Any:
    """Score candidates with CN, Adamic-Adar or resource allocation."""

    import numpy as np

    if name not in ("cn", "aa", "ra"):
        raise ValueError(f"unsupported heuristic: {name}")
    adjacency = adjacency_sets(num_nodes, message_edges)
    pairs = edge_pairs(candidate_edges, name="candidate_edges")
    scores = np.empty(pairs.shape[0], dtype=np.float64)
    for index, (source, target) in enumerate(pairs):
        common = adjacency[int(source)].intersection(adjacency[int(target)])
        if name == "cn":
            scores[index] = float(len(common))
        elif name == "aa":
            scores[index] = sum(
                1.0 / math.log(len(adjacency[node]))
                for node in common
                if len(adjacency[node]) > 1
            )
        else:
            scores[index] = sum(1.0 / len(adjacency[node]) for node in common)
    return scores


def score_all_heuristics(
    *, num_nodes: int, message_edges: Any, candidate_edges: Any
) -> dict[HeuristicName, Any]:
    """Score CN/AA/RA in one adjacency build and one candidate traversal."""

    import numpy as np

    adjacency = adjacency_sets(num_nodes, message_edges)
    pairs = edge_pairs(candidate_edges, name="candidate_edges")
    output: dict[HeuristicName, Any] = {
        "cn": np.empty(pairs.shape[0], dtype=np.float64),
        "aa": np.empty(pairs.shape[0], dtype=np.float64),
        "ra": np.empty(pairs.shape[0], dtype=np.float64),
    }
    for index, (source, target) in enumerate(pairs):
        common = adjacency[int(source)].intersection(adjacency[int(target)])
        output["cn"][index] = float(len(common))
        aa = 0.0
        ra = 0.0
        for node in common:
            degree = len(adjacency[node])
            if degree > 1:
                aa += 1.0 / math.log(degree)
            if degree:
                ra += 1.0 / degree
        output["aa"][index] = aa
        output["ra"][index] = ra
    return output
