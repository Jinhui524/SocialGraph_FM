from collections import deque

import numpy as np
import pytest

from socialgraph_gfm.core.splits import (
    graph_disjoint_split,
    ingest_official_masks,
    leave_one_domain_out,
    spanning_forest_link_split,
    stratified_signed_edge_split,
)


def test_official_masks_are_ingested_fail_closed_with_hand_derived_indices():
    split = ingest_official_masks(
        train_mask=(True, False, False, True),
        validation_mask=(False, True, False, False),
        test_mask=(False, False, True, False),
    )
    assert split.train == (0, 3)
    assert split.validation == (1,)
    assert split.test == (2,)

    with pytest.raises(ValueError, match="exactly one"):
        ingest_official_masks(
            train_mask=(True, False),
            validation_mask=(True, False),
            test_mask=(False, True),
        )


def test_official_masks_accept_numpy_boolean_arrays_without_numeric_coercion():
    split = ingest_official_masks(
        train_mask=np.asarray([True, False]),
        validation_mask=np.asarray([False, True]),
        test_mask=np.asarray([False, False]),
    )
    assert split.train == (0,)
    assert split.validation == (1,)
    assert split.test == ()

    with pytest.raises(ValueError, match="booleans only"):
        ingest_official_masks(
            train_mask=np.asarray([1, 0]),
            validation_mask=np.asarray([0, 1]),
            test_mask=np.asarray([0, 0]),
        )


def test_graph_disjoint_and_leave_one_domain_out_never_split_a_graph():
    explicit = graph_disjoint_split(
        graph_ids=("g3", "g1", "g2", "g4"),
        validation_graph_ids=("g2",),
        test_graph_ids=("g4",),
    )
    assert explicit.train == ("g1", "g3")
    assert explicit.validation == ("g2",)
    assert explicit.test == ("g4",)

    lodo = leave_one_domain_out(
        graph_domains={"en-1": "en", "en-2": "en", "de-1": "de", "fr-1": "fr"},
        test_domain="fr",
        validation_domain="de",
    )
    assert lodo.train == ("en-1", "en-2")
    assert lodo.validation == ("de-1",)
    assert lodo.test == ("fr-1",)
    assert not (set(lodo.train) & set(lodo.validation) | set(lodo.train) & set(lodo.test))


def _connected(num_nodes, edges):
    adjacency = {node: set() for node in range(num_nodes)}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    seen = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node] - seen:
            seen.add(neighbor)
            queue.append(neighbor)
    return seen == set(range(num_nodes))


def test_static_link_split_is_reproducible_80_10_10_and_preserves_connectivity():
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
    )
    first = spanning_forest_link_split(num_nodes=6, edges=edges, seed=17)
    reordered = spanning_forest_link_split(num_nodes=6, edges=reversed(edges), seed=17)
    assert first == reordered
    assert (len(first.train), len(first.validation), len(first.test)) == (8, 1, 1)
    assert set(first.train) | set(first.validation) | set(first.test) == {
        tuple(sorted(edge)) for edge in edges
    }
    assert _connected(6, first.train)


def test_spanning_forest_stays_in_train_when_80_percent_is_insufficient():
    tree = ((0, 1), (1, 2), (2, 3), (3, 4))
    split = spanning_forest_link_split(num_nodes=5, edges=tree, seed=2)
    assert split.train == tree
    assert split.validation == ()
    assert split.test == ()


def test_static_link_split_uses_largest_remainder_for_80_10_10_counts():
    edges = tuple(
        (source, target)
        for source in range(7)
        for target in range(source + 1, 7)
    )[:19]
    split = spanning_forest_link_split(num_nodes=7, edges=edges, seed=5)
    assert (len(split.train), len(split.validation), len(split.test)) == (15, 2, 2)


def test_signed_split_is_70_15_15_stratified_and_reciprocal_groups_are_isolated():
    edges = []
    for index in range(20):
        edges.append((index, index + 100, 1))
    for index in range(20, 40):
        edges.append((index, index + 100, -1))
    # Reciprocal directions remain individual signed edges, but are assigned atomically.
    edges.extend((target, source, sign) for source, target, sign in edges[:6])

    first = stratified_signed_edge_split(edges=edges, seed=31)
    reordered = stratified_signed_edge_split(edges=reversed(edges), seed=31)
    assert first == reordered

    role_by_pair = {}
    for role, role_edges in (
        ("train", first.train),
        ("validation", first.validation),
        ("test", first.test),
    ):
        for source, target, _sign in role_edges:
            pair = tuple(sorted((source, target)))
            assert role_by_pair.setdefault(pair, role) == role

    group_counts = {"train": 0, "validation": 0, "test": 0}
    for role in set(role_by_pair.values()):
        group_counts[role] = sum(value == role for value in role_by_pair.values())
    assert group_counts == {"train": 28, "validation": 6, "test": 6}
    for role_edges, expected_per_sign in (
        (first.train, 14),
        (first.validation, 3),
        (first.test, 3),
    ):
        group_signs = [edge[2] for edge in role_edges if edge[0] < edge[1]]
        assert group_signs.count(1) == expected_per_sign
        assert group_signs.count(-1) == expected_per_sign
    assert {
        edge for role_edges in (first.train, first.validation, first.test) for edge in role_edges
    } == set(edges)
