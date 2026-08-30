"""Convert validated parser records into leakage-explicit SocialGraph-FM Core inputs."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any, Literal

from .bundle import SourceProvenance, CoreGraphBundle, calculate_graph_version_hash
from .datasets.parsers import ParsedGraph
from .splits import EdgeSplit, IndexSplit, SignedEdgeSplit


def _validate_partition(
    roles: tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]],
) -> None:
    sets = tuple(set(role) for role in roles)
    if any(len(role) != len(role_set) for role, role_set in zip(roles, sets, strict=True)):
        raise ValueError("split roles must not contain duplicates")
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("split roles must be disjoint")


def _numeric_features(
    parsed: ParsedGraph,
    order: tuple[int, ...],
    excluded: set[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, rows in sorted(parsed.numeric_features.items()):
        if name in excluded:
            continue
        if len(rows) != len(parsed.node_ids) or not rows:
            raise ValueError("numeric feature rows must cover every parsed node")
        widths = {len(row) for row in rows}
        if len(widths) != 1 or next(iter(widths)) < 1:
            raise ValueError("numeric feature rows must have one fixed positive width")
        width = next(iter(widths))
        for column in range(width):
            field_name = name if width == 1 else f"{name}.{column}"
            result.append(
                {
                    "kind": "numeric",
                    "name": field_name,
                    "values": [float(rows[row][column]) for row in order],
                }
            )
    return result


def _categorical_features(
    parsed: ParsedGraph,
    order: tuple[int, ...],
    excluded: set[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, values in sorted(parsed.categorical_features.items()):
        if name in excluded:
            continue
        if len(values) != len(parsed.node_ids):
            raise ValueError("categorical feature rows must cover every parsed node")
        result.append(
            {
                "kind": "categorical",
                "name": name,
                "values": [values[row] for row in order],
            }
        )
    return result


def _multi_hot_features(
    parsed: ParsedGraph,
    order: tuple[int, ...],
    excluded: set[str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, rows in sorted(parsed.multi_hot_features.items()):
        if name in excluded:
            continue
        if len(rows) != len(parsed.node_ids):
            raise ValueError("multi-hot feature rows must cover every parsed node")
        values: list[str] = []
        offsets = [0]
        for row in order:
            values.extend(str(value) for value in rows[row])
            offsets.append(len(values))
        result.append(
            {
                "kind": "multiHot",
                "name": name,
                "rowOffsets": offsets,
                "values": values,
            }
        )
    return result


def bundle_from_parsed_graph(
    parsed: ParsedGraph,
    *,
    source: SourceProvenance,
    split: IndexSplit | EdgeSplit | SignedEdgeSplit,
    excluded_feature_names: Collection[str],
    index_split_strategy: Literal["official", "all-visible-training"] = "official",
) -> CoreGraphBundle:
    """Build one strict bundle while keeping targets in a separate experiment artifact."""

    if not parsed.node_ids or len(parsed.node_ids) != len(set(parsed.node_ids)):
        raise ValueError("parsed graph node IDs must be nonempty and unique")
    if parsed.edges and parsed.signed_edges:
        raise ValueError("a parsed graph cannot mix unsigned and signed edge inventories")
    sorted_ids = tuple(sorted(parsed.node_ids))
    old_by_id = {identifier: index for index, identifier in enumerate(parsed.node_ids)}
    order = tuple(old_by_id[identifier] for identifier in sorted_ids)
    new_by_old = {old: new for new, old in enumerate(order)}

    excluded = set(excluded_feature_names)
    if any(not isinstance(name, str) or not name for name in excluded):
        raise ValueError("excluded feature names must be nonempty strings")
    features = [
        *_numeric_features(parsed, order, excluded),
        *_categorical_features(parsed, order, excluded),
        *_multi_hot_features(parsed, order, excluded),
    ]

    edge_payloads: list[dict[str, object]] = []
    old_to_stable = dict(enumerate(parsed.node_ids))
    if parsed.signed_edges:
        for source_index, target_index, sign in parsed.signed_edges:
            if sign not in {-1, 1}:
                raise ValueError("signed graph edges require sign -1 or 1")
            source_id, target_id = old_to_stable[source_index], old_to_stable[target_index]
            edge_payloads.append(
                {
                    "sourceId": source_id,
                    "targetId": target_id,
                    "edgeType": "support" if sign == 1 else "oppose",
                    "weight": 1.0,
                }
            )
    else:
        for source_index, target_index in parsed.edges:
            source_id, target_id = old_to_stable[source_index], old_to_stable[target_index]
            if not parsed.directed and source_id > target_id:
                source_id, target_id = target_id, source_id
            edge_payloads.append(
                {
                    "sourceId": source_id,
                    "targetId": target_id,
                    "edgeType": "relation",
                    "weight": 1.0,
                }
            )
    edge_payloads.sort(
        key=lambda edge: (
            str(edge["sourceId"]),
            str(edge["targetId"]),
            str(edge["edgeType"]),
        )
    )

    assignments: list[dict[str, str]] = []
    strategy: Literal[
        "official",
        "all-visible-training",
        "spanning-forest-80-10-10",
        "signed-pair-stratified-70-15-15",
    ]
    if isinstance(split, IndexSplit):
        _validate_partition((split.train, split.validation, split.test))
        assigned_indices = set(split.train) | set(split.validation) | set(split.test)
        for role, indices in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for index in indices:
                if index < 0 or index >= len(parsed.node_ids):
                    raise ValueError("node split index is outside the parsed node inventory")
                assignments.append({"entityId": parsed.node_ids[index], "role": role})
        assignments.extend(
            {"entityId": parsed.node_ids[index], "role": "unlabeled"}
            for index in range(len(parsed.node_ids))
            if index not in assigned_indices
        )
        strategy = index_split_strategy
    elif isinstance(split, EdgeSplit):
        _validate_partition((split.train, split.validation, split.test))
        known = set(parsed.edges)
        if set(split.train) | set(split.validation) | set(split.test) != known:
            raise ValueError("edge split must cover the exact unsigned edge inventory")
        for role, split_edges in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for left, right in split_edges:
                source_id, target_id = old_to_stable[left], old_to_stable[right]
                if not parsed.directed and source_id > target_id:
                    source_id, target_id = target_id, source_id
                assignments.append(
                    {"entityId": f"edge:{source_id}:{target_id}", "role": role}
                )
        strategy = "spanning-forest-80-10-10"
    elif isinstance(split, SignedEdgeSplit):
        _validate_partition((split.train, split.validation, split.test))
        known_signed = set(parsed.signed_edges)
        if set(split.train) | set(split.validation) | set(split.test) != known_signed:
            raise ValueError("signed split must cover the exact signed edge inventory")
        unordered_role: dict[tuple[int, int], str] = {}
        for role, signed_split_edges in (
            ("train", split.train),
            ("validation", split.validation),
            ("test", split.test),
        ):
            for left, right, _sign in signed_split_edges:
                pair = (min(left, right), max(left, right))
                previous = unordered_role.setdefault(pair, role)
                if previous != role:
                    raise ValueError(
                        "signed split must keep each unordered user pair in one role"
                    )
                assignments.append(
                    {
                        "entityId": f"edge:{old_to_stable[left]}:{old_to_stable[right]}",
                        "role": role,
                    }
                )
        strategy = "signed-pair-stratified-70-15-15"
    else:  # pragma: no cover - the annotated union is closed
        raise TypeError("unsupported parsed graph split")

    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": parsed.directed,
        "nodes": [
            {"id": identifier, "index": index}
            for index, identifier in enumerate(sorted_ids)
        ],
        "edges": edge_payloads,
        "nodeFeatures": features,
        "structuralFeatures": None,
        "source": source.model_dump(mode="json", by_alias=True),
        "splitManifest": {
            "strategy": strategy,
            "assignments": sorted(assignments, key=lambda item: item["entityId"]),
        },
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    bundle = CoreGraphBundle.model_validate(payload)

    # The remapping is deliberately computed and checked even though stable IDs are used in
    # the serialized edge payload. It catches parser indices outside the declared inventory.
    if set(new_by_old) != set(range(len(parsed.node_ids))):
        raise ValueError("parsed node indices are incomplete")
    return bundle


__all__ = ["bundle_from_parsed_graph"]
