from __future__ import annotations

import gc
import warnings
import weakref

import pytest
import torch

from socialgraph_gfm.core import training_data
from socialgraph_gfm.core.training_data import (
    BalancedDomainSampler,
    ExecutionPolicy,
    NeighborBatchSource,
    PreparedGraph,
)


def test_balanced_sampler_interleaves_domains_and_restores_exact_next_batch() -> None:
    sampler = BalancedDomainSampler({"a": 2, "b": 3, "c": 1}, seed=41)
    first = [sampler.next() for _ in range(7)]
    assert {domain for domain, _ in first[:3]} == {"a", "b", "c"}
    assert {domain for domain, _ in first[3:6]} == {"a", "b", "c"}
    state = sampler.state_dict()
    expected = [sampler.next() for _ in range(8)]

    resumed = BalancedDomainSampler({"a": 2, "b": 3, "c": 1}, seed=999)
    resumed.load_state_dict(state)
    assert [resumed.next() for _ in range(8)] == expected


def test_prepared_graph_builds_sparse_topology_and_membership_once() -> None:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    graph = PreparedGraph.from_edge_index(num_nodes=3, edge_index=edge_index, directed=False)
    csr_id = id(graph.csr)
    membership_id = id(graph.positive_edge_keys)
    pair_cache_id = id(graph.pair_mask_cache)

    for _ in range(5):
        batch = graph.full_batch()
        assert batch.edge_index.data_ptr() == edge_index.data_ptr()
        assert graph.contains_positive(torch.tensor([[1, 0], [0, 2]])).tolist() == [True, False]

    assert graph.cache_build_count == 1
    assert id(graph.csr) == csr_id
    assert id(graph.positive_edge_keys) == membership_id
    assert id(graph.pair_mask_cache) == pair_cache_id
    assert graph.pair_cache_build_count == 1
    assert graph.pair_mask_cache.representative_edges.tolist() == [True, False, True, False]
    directed = PreparedGraph.from_edge_index(
        num_nodes=2,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        directed=True,
    )
    assert directed.pair_mask_cache.pair_keys.numel() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_positive_membership_crosses_devices_without_moving_the_held_cache() -> None:
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    cpu_graph = PreparedGraph.from_edge_index(num_nodes=3, edge_index=edge_index, directed=False)
    cpu_pointer = cpu_graph.positive_edge_keys.data_ptr()
    cuda_pairs = torch.tensor([[1, 0], [0, 2]], device="cuda")

    cuda_mask = cpu_graph.contains_positive(cuda_pairs)

    assert cuda_mask.device == cuda_pairs.device
    assert cuda_mask.tolist() == [True, False]
    assert cpu_graph.positive_edge_keys.device.type == "cpu"
    assert cpu_graph.positive_edge_keys.data_ptr() == cpu_pointer

    cuda_graph = cpu_graph.to("cuda")
    cuda_pointer = cuda_graph.positive_edge_keys.data_ptr()
    cpu_pairs = torch.tensor([[2, 1], [0, 2]])

    cpu_mask = cuda_graph.contains_positive(cpu_pairs)

    assert cpu_mask.device == cpu_pairs.device
    assert cpu_mask.tolist() == [True, False]
    assert cuda_graph.positive_edge_keys.device.type == "cuda"
    assert cuda_graph.positive_edge_keys.data_ptr() == cuda_pointer


def test_execution_policy_selects_full_batch_or_pyg_neighbor_defaults() -> None:
    policy = ExecutionPolicy()
    assert policy.mode(edge_count=99_999) == "full-batch"
    assert policy.mode(edge_count=100_000) == "neighbor"
    assert policy.node_batch_size == 1024
    assert policy.edge_batch_size == 2048
    assert policy.fanout == (15, 10, 5)


def test_real_node_and_link_loaders_iterate_train_only_topology_and_holdouts_stay_positive() -> (
    None
):
    train_edges = torch.tensor(
        [[0, 2, 1, 3, 2, 4, 3, 4], [2, 0, 3, 1, 4, 2, 4, 3]], dtype=torch.long
    )
    held_out = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graph = PreparedGraph.from_edge_index(
        num_nodes=5,
        edge_index=train_edges,
        positive_edge_index=torch.cat((train_edges, held_out), dim=1),
        directed=False,
    )
    policy = ExecutionPolicy(
        full_batch_edge_threshold=1,
        node_batch_size=2,
        edge_batch_size=2,
        fanout=(2, 2, 2),
    )
    features = torch.randn(5, 128)
    node_source = NeighborBatchSource(
        graph=graph, features=features, policy=policy, loader_kind="node", seed=7
    )
    link_source = NeighborBatchSource(
        graph=graph,
        features=features,
        policy=policy,
        loader_kind="link",
        edge_label_index=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        seed=7,
    )

    node_batch = node_source.get(batch_index=0, ordinal=0)
    link_batch = link_source.get(batch_index=0, ordinal=0)
    assert node_batch.seed_count == 2
    assert link_batch.seed_count == 2
    assert node_batch.edge_index.shape[1] <= train_edges.shape[1]
    assert link_batch.edge_index.shape[1] <= train_edges.shape[1]
    for batch in (node_batch, link_batch):
        assert batch.global_pair_representatives.shape == batch.global_pair_ids.shape
        for pair_id in batch.global_pair_ids.unique():
            selected = batch.global_pair_ids == pair_id
            assert batch.global_pair_representatives[selected].sum() <= 1
        global_edges = {
            tuple(batch.global_node_ids[edge].tolist()) for edge in batch.edge_index.t()
        }
        assert (0, 1) not in global_edges and (1, 0) not in global_edges
        assert torch.any(batch.global_node_ids == 0)
        assert torch.any(batch.global_node_ids == 1)
        assert graph.contains_positive(torch.tensor([[0, 1]])).item() is True
        local_negatives = graph.sample_negative_pairs_from_nodes(
            batch.global_node_ids,
            2,
            generator=torch.Generator().manual_seed(91),
        )
        global_negatives = batch.global_node_ids[local_negatives]
        assert not graph.contains_positive(global_negatives).any()
        assert not torch.any(torch.all(global_negatives == torch.tensor([0, 1]), dim=1))
        assert not torch.any(torch.all(global_negatives == torch.tensor([1, 0]), dim=1))
    assert node_source.loader_construction_count == 1
    assert link_source.loader_construction_count == 1

    default_source = NeighborBatchSource(
        graph=graph,
        features=features,
        policy=ExecutionPolicy(full_batch_edge_threshold=1),
        loader_kind="node",
        seed=8,
    )
    assert default_source.get(batch_index=0, ordinal=0).seed_count == 5
    assert default_source.policy.node_batch_size == 1024
    assert default_source.policy.edge_batch_size == 2048
    assert default_source.policy.fanout == (15, 10, 5)


def test_neighbor_batches_are_transient_views_over_one_global_cache() -> None:
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    graph = PreparedGraph.from_edge_index(num_nodes=5, edge_index=edge_index, directed=False)
    source = NeighborBatchSource(
        graph=graph,
        features=torch.randn(5, 128),
        policy=ExecutionPolicy(
            full_batch_edge_threshold=1,
            node_batch_size=2,
            edge_batch_size=2,
            fanout=(1, 0, 0),
        ),
        loader_kind="node",
        seed=13,
    )
    membership_ptr = graph.positive_edge_keys.data_ptr()
    pair_ptr = graph.pair_mask_cache.pair_keys.data_ptr()
    batch_references: list[weakref.ReferenceType[object]] = []

    for ordinal in range(64):
        batch = source.get(batch_index=ordinal % source.batch_count, ordinal=ordinal)
        batch_references.append(weakref.ref(batch))
        assert batch.edge_index.shape[0] == 2
        assert batch.global_pair_ids.shape == (batch.edge_index.shape[1],)
        assert batch.global_pair_representatives.shape == (batch.edge_index.shape[1],)
        for pair_id in batch.global_pair_ids.unique():
            selected = batch.global_pair_ids == pair_id
            assert batch.global_pair_representatives[selected].sum() <= 1
        assert not hasattr(batch, "graph")

    del batch
    gc.collect()
    assert source.loader_construction_count == 64
    assert source.retained_batch_count == 0
    assert all(reference() is None for reference in batch_references)
    assert graph.cache_build_count == 1
    assert graph.pair_cache_build_count == 1
    assert graph.positive_edge_keys.data_ptr() == membership_ptr
    assert graph.pair_mask_cache.pair_keys.data_ptr() == pair_ptr


def test_sparse_cache_makes_invariant_choice_without_emitting_framework_warnings() -> None:
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        PreparedGraph.from_edge_index(num_nodes=2, edge_index=edge_index, directed=False)
    messages = [str(item.message) for item in observed]
    assert not any("Sparse invariant checks" in message for message in messages)
    assert not any("Sparse CSR tensor support is in beta state" in message for message in messages)


def test_negative_sampling_fails_fast_when_complete_and_finds_unique_near_complete_gap() -> None:
    complete_edges = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    complete = PreparedGraph.from_edge_index(num_nodes=3, edge_index=complete_edges, directed=False)
    with pytest.raises(training_data.InsufficientNegativeCapacityError, match="no negative edges"):
        complete.sample_negative_pairs(1, generator=torch.Generator().manual_seed(1))

    near_complete_edges = torch.tensor(
        [[0, 1, 0, 2, 0, 3, 1, 2, 1, 3], [1, 0, 2, 0, 3, 0, 2, 1, 3, 1]],
        dtype=torch.long,
    )
    near_complete = PreparedGraph.from_edge_index(
        num_nodes=4, edge_index=near_complete_edges, directed=False
    )
    sampled = near_complete.sample_negative_pairs(1, generator=torch.Generator().manual_seed(2))
    assert sampled.tolist() == [[2, 3]]
    assert near_complete.contains_positive(sampled).tolist() == [False]
