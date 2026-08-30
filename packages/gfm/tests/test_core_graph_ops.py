import pytest

from socialgraph_gfm.core.graph_ops import (
    canonicalize_edges,
    mask_message_passing_edges,
    sample_negative_edges,
)


def test_undirected_canonicalization_is_atomic_and_duplicate_free():
    assert canonicalize_edges(((2, 1), (0, 3), (1, 2)), directed=False) == (
        (0, 3),
        (1, 2),
    )
    with pytest.raises(ValueError, match="self-loop"):
        canonicalize_edges(((1, 1),), directed=False)


def test_masking_an_undirected_pair_removes_both_message_passing_directions():
    adjacency = ((0, 1), (1, 0), (1, 2), (2, 1), (2, 3))
    masked = mask_message_passing_edges(
        adjacency,
        masked_edges=((1, 0),),
        directed=False,
    )
    assert masked == ((1, 2), (2, 1), (2, 3))
    assert (0, 1) not in masked and (1, 0) not in masked


def test_negative_sampling_excludes_all_positive_splits_and_has_no_duplicates():
    positives = {
        "train": ((0, 1), (1, 2)),
        "validation": ((3, 2),),
        "test": ((4, 0),),
    }
    first = sample_negative_edges(
        num_nodes=5,
        positive_splits=positives,
        count=5,
        seed=77,
        directed=False,
    )
    second = sample_negative_edges(
        num_nodes=5,
        positive_splits=positives,
        count=5,
        seed=77,
        directed=False,
    )
    known = {(0, 1), (1, 2), (2, 3), (0, 4)}
    assert first == second
    assert len(first) == len(set(first)) == 5
    assert not (set(first) & known)
    assert all(source < target for source, target in first)


def test_negative_sampling_handles_large_sparse_endpoint_space_without_candidate_matrix():
    sampled = sample_negative_edges(
        num_nodes=1_000_000,
        positive_splits={"train": ((0, 1),), "validation": (), "test": ()},
        count=8,
        seed=9,
        directed=False,
    )
    assert len(sampled) == 8
    assert len(set(sampled)) == 8


def test_negative_sampling_requires_all_split_positive_sets():
    with pytest.raises(ValueError, match="all train, validation, and test"):
        sample_negative_edges(
            num_nodes=3,
            positive_splits={"train": ((0, 1),)},
            count=1,
            seed=1,
            directed=False,
        )


def test_directed_negative_sampling_preserves_direction_semantics():
    sampled = sample_negative_edges(
        num_nodes=2,
        positive_splits={"train": ((0, 1),), "validation": (), "test": ()},
        count=1,
        seed=1,
        directed=True,
    )
    assert sampled == ((1, 0),)
