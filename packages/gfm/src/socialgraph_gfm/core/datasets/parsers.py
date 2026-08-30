"""Dataset-specific parsers producing small Torch-free static graph records."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .acquire import safe_load_mat_arrays, safe_load_numpy_arrays
from ..graph_ops import sample_negative_edges
from ..splits import (
    EdgeSplit,
    IndexSplit,
    SignedEdgeSplit,
    ingest_official_masks,
    spanning_forest_link_split,
    stratified_signed_edge_split,
)


@dataclass(frozen=True)
class ParsedGraph:
    graph_id: str
    directed: bool
    node_ids: tuple[str, ...]
    edges: tuple[tuple[int, int], ...] = ()
    signed_edges: tuple[tuple[int, int, int], ...] = ()
    numeric_features: dict[str, tuple[tuple[float, ...], ...]] = field(default_factory=dict)
    categorical_features: dict[str, tuple[str | None, ...]] = field(default_factory=dict)
    multi_hot_features: dict[str, tuple[tuple[str, ...], ...]] = field(default_factory=dict)
    targets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    official_splits: tuple[IndexSplit, ...] = ()
    offline_labels: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def model_payload(self) -> dict[str, Any]:
        return {
            "graphId": self.graph_id,
            "directed": self.directed,
            "nodeIds": self.node_ids,
            "edges": self.edges,
            "signedEdges": self.signed_edges,
            "numericFeatures": self.numeric_features,
            "categoricalFeatures": self.categorical_features,
            "multiHotFeatures": self.multi_hot_features,
            "targets": self.targets,
        }


@dataclass(frozen=True)
class PreparedLinkTask:
    graph: ParsedGraph
    split: EdgeSplit
    message_passing_edges: tuple[tuple[int, int], ...]

    def sample_negatives(self, *, count: int, seed: int) -> tuple[tuple[int, int], ...]:
        return sample_negative_edges(
            num_nodes=len(self.graph.node_ids),
            positive_splits={
                "train": self.split.train,
                "validation": self.split.validation,
                "test": self.split.test,
            },
            count=count,
            seed=seed,
            directed=False,
        )


@dataclass(frozen=True)
class PreparedSignedTask:
    graph: ParsedGraph
    split: SignedEdgeSplit
    message_passing_edges: tuple[tuple[int, int, int], ...]


def _read_fixture(path: Path, recipe_id: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("recipeId") != recipe_id:
        raise ValueError("fixture recipeId does not match parser")
    return raw


def _indexed_edges(
    node_ids: tuple[str, ...], raw_edges: list[list[str]], *, directed: bool
) -> tuple[tuple[int, int], ...]:
    indices = {identifier: index for index, identifier in enumerate(node_ids)}
    edges: set[tuple[int, int]] = set()
    for source, target in raw_edges:
        if source not in indices or target not in indices or source == target:
            raise ValueError("edge endpoints must reference distinct declared nodes")
        edge = (indices[source], indices[target])
        if not directed and edge[0] > edge[1]:
            edge = (edge[1], edge[0])
        edges.add(edge)
    return tuple(sorted(edges))


def parse_facebook_fixture(path: Path) -> ParsedGraph:
    raw = _read_fixture(path, "facebook100")
    fields = ("gender", "major", "secondMajor", "dorm", "year", "highSchool")
    ordered = sorted(raw["nodes"], key=lambda node: node["id"])
    node_ids = tuple(node["id"] for node in ordered)
    categorical = {field: tuple(node[field] for node in ordered) for field in fields}
    return ParsedGraph(
        graph_id=raw["graphId"],
        directed=False,
        node_ids=node_ids,
        edges=_indexed_edges(node_ids, raw["edges"], directed=False),
        categorical_features=categorical,
        targets={"gender": tuple(int(node["gender"]) for node in ordered)},
    )


def parse_twitch_fixture(path: Path) -> dict[str, ParsedGraph]:
    raw = _read_fixture(path, "twitch-language")
    parsed: dict[str, ParsedGraph] = {}
    for graph_id in ("DE", "EN", "ES", "FR", "PT", "RU"):
        graph = raw["graphs"][graph_id]
        ordered = sorted(graph["nodes"], key=lambda node: node["id"])
        node_ids = tuple(node["id"] for node in ordered)
        parsed[graph_id] = ParsedGraph(
            graph_id=graph_id,
            directed=False,
            node_ids=node_ids,
            edges=_indexed_edges(node_ids, graph["edges"], directed=False),
            multi_hot_features={
                "sharedAttributes": tuple(tuple(node["features"]) for node in ordered)
            },
            targets={"mature": tuple(int(node["mature"]) for node in ordered)},
        )
    return parsed


def _official_index_split(raw: dict[str, Any], size: int) -> IndexSplit:
    masks = []
    for role in ("train", "validation", "test"):
        indices = raw[role]
        if len(indices) != len(set(indices)) or any(
            index < 0 or index >= size for index in indices
        ):
            raise ValueError("official split index is invalid")
        selected = set(indices)
        masks.append(tuple(index in selected for index in range(size)))
    return ingest_official_masks(train_mask=masks[0], validation_mask=masks[1], test_mask=masks[2])


def parse_tolokers_fixture(path: Path) -> ParsedGraph:
    raw = _read_fixture(path, "tolokers")
    ordered = sorted(raw["nodes"], key=lambda node: node["id"])
    node_ids = tuple(node["id"] for node in ordered)
    splits = tuple(_official_index_split(split, len(node_ids)) for split in raw["officialSplits"])
    if len(splits) != 10:
        raise ValueError("Tolokers requires exactly ten official splits")
    return ParsedGraph(
        graph_id="tolokers",
        directed=False,
        node_ids=node_ids,
        edges=_indexed_edges(node_ids, raw["edges"], directed=False),
        numeric_features={
            "attributes": tuple(
                tuple(float(value) for value in node["features"]) for node in ordered
            )
        },
        targets={"banned": tuple(int(node["banned"]) for node in ordered)},
        official_splits=splits,
    )


def parse_wiki_rfa(path: Path) -> ParsedGraph:
    """Drop text/time, neutral votes and directed-pair ties; retain majority signs only."""

    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in (*path.read_text(encoding="utf-8").splitlines(), ""):
        if not line:
            if current:
                records.append(current)
                current = {}
            continue
        key, separator, value = line.partition(":")
        if not separator or key in current:
            raise ValueError("Wiki-RfA record is malformed")
        current[key] = value

    votes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for record in records:
        if set(record) != {"SRC", "TGT", "VOT", "RES", "YEA", "DAT", "TXT"}:
            raise ValueError("Wiki-RfA record inventory is invalid")
        vote = int(record["VOT"])
        if vote not in {-1, 0, 1}:
            raise ValueError("Wiki-RfA vote must be -1, 0, or 1")
        # The official dump contains self-votes and records with an empty user
        # name. Neither can form a valid CoreGraphBundle governance relation.
        if vote and record["SRC"] and record["TGT"] and record["SRC"] != record["TGT"]:
            votes[(record["SRC"], record["TGT"])].append(vote)

    majority: list[tuple[str, str, int]] = []
    for (source, target), pair_votes in votes.items():
        balance = sum(pair_votes)
        if balance:
            majority.append((source, target, 1 if balance > 0 else -1))
    node_ids = tuple(sorted({node for source, target, _ in majority for node in (source, target)}))
    indices = {identifier: index for index, identifier in enumerate(node_ids)}
    signed = tuple(
        sorted((indices[source], indices[target], sign) for source, target, sign in majority)
    )
    return ParsedGraph(graph_id="wiki-rfa", directed=True, node_ids=node_ids, signed_edges=signed)


def prepare_wiki_rfa(path: Path, *, seed: int) -> PreparedSignedTask:
    """Parse Wiki-RfA and bind reciprocal pairs to one split with train-only topology."""

    graph = parse_wiki_rfa(path)
    split = stratified_signed_edge_split(edges=graph.signed_edges, seed=seed)
    return PreparedSignedTask(graph=graph, split=split, message_passing_edges=split.train)


def parse_link_fixture(path: Path, *, seed: int) -> tuple[ParsedGraph, EdgeSplit]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recipe_id = raw.get("recipeId")
    if recipe_id not in {"github-musae", "email-eu-core"}:
        raise ValueError("link fixture has an unsupported recipeId")
    ordered = sorted(raw["nodes"], key=lambda node: node["id"])
    node_ids = tuple(node["id"] for node in ordered)
    graph = ParsedGraph(
        graph_id=recipe_id,
        directed=False,
        node_ids=node_ids,
        edges=_indexed_edges(node_ids, raw["edges"], directed=False),
        multi_hot_features=(
            {"attributes": tuple(tuple(node["features"]) for node in ordered)}
            if recipe_id == "github-musae"
            else {}
        ),
        offline_labels=(
            {"department": tuple(node["department"] for node in ordered)}
            if recipe_id == "email-eu-core"
            else {}
        ),
    )
    split = spanning_forest_link_split(num_nodes=len(node_ids), edges=graph.edges, seed=seed)
    return graph, split


def parse_email_files(edges_path: Path, departments_path: Path) -> ParsedGraph:
    raw_edges: list[tuple[str, str]] = []
    for line in edges_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError("Email-Eu-core edge row is malformed")
        if parts[0] == parts[1]:
            continue
        raw_edges.append((parts[0], parts[1]))
    departments: dict[str, str] = {}
    for line in departments_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or parts[0] in departments:
            raise ValueError("Email-Eu-core department row is malformed")
        departments[parts[0]] = parts[1]
    node_ids = tuple(sorted({node for edge in raw_edges for node in edge} | set(departments)))
    if set(node_ids) != set(departments):
        raise ValueError("Email-Eu-core department labels must cover every node")
    indices = {identifier: index for index, identifier in enumerate(node_ids)}
    edges = tuple(
        sorted(
            {
                (min(indices[source], indices[target]), max(indices[source], indices[target]))
                for source, target in raw_edges
            }
        )
    )
    return ParsedGraph(
        graph_id="email-eu-core",
        directed=False,
        node_ids=node_ids,
        edges=edges,
        offline_labels={"department": tuple(departments[node] for node in node_ids)},
    )


def parse_musae_files(
    *, graph_id: str, edges_path: Path, features_path: Path, target_path: Path | None = None
) -> ParsedGraph:
    features_raw = json.loads(features_path.read_text(encoding="utf-8"))
    raw_edges: list[list[str]] = []
    with edges_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames not in (["from", "to"], ["id_1", "id_2"]):
            raise ValueError("MUSAE edge CSV header is invalid")
        left, right = reader.fieldnames
        raw_edges.extend([[row[left], row[right]] for row in reader])

    stable_by_source: dict[str, str] = {key: key for key in features_raw}
    targets: dict[str, int] = {}
    if target_path is not None:
        with target_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                source_id = row.get("new_id", row.get("id"))
                stable_id = row.get("id", source_id)
                if source_id is None or stable_id is None:
                    raise ValueError("MUSAE target row lacks an ID")
                stable_by_source[source_id] = stable_id
                if "mature" in row:
                    targets[stable_id] = int(row["mature"].lower() == "true")

    source_ids = set(features_raw) | {node for edge in raw_edges for node in edge}
    if not source_ids <= set(stable_by_source):
        raise ValueError("MUSAE source ID has no stable node mapping")
    stable_ids = tuple(sorted(stable_by_source[source] for source in source_ids))
    stable_edges = [[stable_by_source[left], stable_by_source[right]] for left, right in raw_edges]
    features_by_stable = {
        stable_by_source[source]: tuple(str(value) for value in values)
        for source, values in features_raw.items()
    }
    return ParsedGraph(
        graph_id=graph_id,
        directed=False,
        node_ids=stable_ids,
        edges=_indexed_edges(stable_ids, stable_edges, directed=False),
        multi_hot_features={
            "sharedAttributes": tuple(features_by_stable.get(node, ()) for node in stable_ids)
        },
        targets=({"mature": tuple(targets[node] for node in stable_ids)} if targets else {}),
    )


def prepare_github_musae(
    *, edges_path: Path, features_path: Path, target_path: Path | None = None, seed: int
) -> PreparedLinkTask:
    """Parse GitHub MUSAE and expose a forest-preserving split and train-only topology."""

    graph = parse_musae_files(
        graph_id="github-musae",
        edges_path=edges_path,
        features_path=features_path,
        target_path=target_path,
    )
    split = spanning_forest_link_split(num_nodes=len(graph.node_ids), edges=graph.edges, seed=seed)
    return PreparedLinkTask(graph=graph, split=split, message_passing_edges=split.train)


def parse_tolokers_npz(path: Path) -> ParsedGraph:
    arrays = safe_load_numpy_arrays(
        path,
        expected_keys={
            "node_features",
            "node_labels",
            "edges",
            "train_masks",
            "val_masks",
            "test_masks",
        },
        max_array_elements=2_000_000,
        max_total_array_bytes=64 * 1024 * 1024,
    )
    features = arrays["node_features"]
    labels = arrays["node_labels"]
    raw_edges_array = arrays["edges"]
    masks = (arrays["train_masks"], arrays["val_masks"], arrays["test_masks"])
    if features.ndim != 2 or features.shape[1] != 10:
        raise ValueError("Tolokers node_features must have exact shape (nodes, 10)")
    node_count = int(arrays["node_features"].shape[0])
    if labels.shape != (node_count,):
        raise ValueError("Tolokers node_labels length must equal node count")
    if raw_edges_array.ndim != 2 or raw_edges_array.shape[1] != 2:
        raise ValueError("Tolokers edges must have exact shape (edges, 2)")
    if any(mask.shape != (10, node_count) for mask in masks):
        raise ValueError("Tolokers masks must have exact shape (10, nodes)")
    # SocialGraph-FM Core indexes the lexicographically sorted stable IDs. The
    # official NPZ is in numeric row order, so every row-aligned input and mask
    # must be remapped together rather than relabeling indices implicitly.
    stable_order = tuple(sorted(range(node_count), key=lambda index: str(index)))
    node_ids = tuple(str(index) for index in stable_order)
    raw_edges = raw_edges_array
    edges = _indexed_edges(
        node_ids, [[str(int(left)), str(int(right))] for left, right in raw_edges], directed=False
    )
    splits = tuple(
        ingest_official_masks(
            train_mask=masks[0][index][list(stable_order)],
            validation_mask=masks[1][index][list(stable_order)],
            test_mask=masks[2][index][list(stable_order)],
        )
        for index in range(10)
    )
    return ParsedGraph(
        graph_id="tolokers",
        directed=False,
        node_ids=node_ids,
        edges=edges,
        numeric_features={
            "attributes": tuple(
                tuple(float(value) for value in arrays["node_features"][index])
                for index in stable_order
            )
        },
        targets={"banned": tuple(int(arrays["node_labels"][index]) for index in stable_order)},
        official_splits=splits,
    )


def parse_facebook100_mat(
    path: Path, *, graph_id: str, official_splits_path: Path | None = None
) -> ParsedGraph:
    arrays = safe_load_mat_arrays(
        path,
        expected_keys={"A", "local_info"},
        max_array_elements=5_000_000,
        max_worker_memory_bytes=2 * 1024 * 1024 * 1024,
        timeout_seconds=60,
    )
    adjacency = arrays["A"].tocoo()
    profile = arrays["local_info"]
    if profile.ndim != 2 or profile.shape[1] != 7:
        raise ValueError("Facebook100 local_info must have exact shape (nodes, 7)")
    if adjacency.shape != (profile.shape[0], profile.shape[0]):
        raise ValueError("Facebook100 adjacency must have exact square node dimensions")
    node_ids = tuple(f"{graph_id}:{index}" for index in range(profile.shape[0]))
    edges = tuple(
        sorted(
            {
                (min(int(left), int(right)), max(int(left), int(right)))
                for left, right in zip(adjacency.row, adjacency.col)
                if left != right
            }
        )
    )
    field_names = ("studentStatus", "gender", "major", "secondMajor", "dorm", "year", "highSchool")
    categorical = {
        name: tuple(None if int(value) == 0 else str(int(value)) for value in profile[:, column])
        for column, name in enumerate(field_names)
    }
    official: tuple[IndexSplit, ...] = ()
    if official_splits_path is not None:
        from .penn94_conversion import load_penn94_safe_splits

        labeled_indices = [index for index, value in enumerate(profile[:, 1]) if int(value) != 0]
        official = load_penn94_safe_splits(
            official_splits_path, labeled_node_indices=labeled_indices
        )
    return ParsedGraph(
        graph_id=graph_id,
        directed=False,
        node_ids=node_ids,
        edges=edges,
        categorical_features=categorical,
        targets={
            "gender": tuple(-1 if value is None else int(value) for value in categorical["gender"])
        },
        official_splits=official,
    )


__all__ = [
    "ParsedGraph",
    "PreparedLinkTask",
    "PreparedSignedTask",
    "parse_email_files",
    "parse_facebook100_mat",
    "parse_facebook_fixture",
    "parse_link_fixture",
    "parse_musae_files",
    "prepare_github_musae",
    "parse_tolokers_fixture",
    "parse_tolokers_npz",
    "parse_twitch_fixture",
    "parse_wiki_rfa",
    "prepare_wiki_rfa",
]
