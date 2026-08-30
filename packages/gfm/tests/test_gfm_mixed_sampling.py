from __future__ import annotations

import inspect
from copy import deepcopy

import pytest
import torch

from socialgraph_gfm.gfm import CausalMixedNegativeSampler, MixedNegativeSample


def _sampler(seed: int = 17) -> CausalMixedNegativeSampler:
    visible = torch.tensor([list(range(11)), list(range(1, 12))], dtype=torch.long)
    return CausalMixedNegativeSampler(
        source_count=12,
        target_count=12,
        visible_edge_index=visible,
        visible_edge_time=torch.arange(1, 12, dtype=torch.float32),
        cutoff_time=11.0,
        seed=seed,
        directed=False,
        node_types=torch.tensor([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]),
        topic_groups=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
    )


def test_mixed_sampler_exact_ratio_query_layout_and_resume() -> None:
    positives = torch.tensor([[0, 4], [10, 10]], dtype=torch.long)
    sampler = _sampler()
    state = deepcopy(sampler.state_dict())
    sample = sampler.sample(positives, negatives_per_positive=4)
    assert sample.edge_index.shape == (2, 8)
    assert sample.requested_component_counts == {
        "hard": 4,
        "degree_matched": 2,
        "uniform": 2,
    }
    assert sample.actual_component_counts == sample.requested_component_counts
    assert not sample.fallback_events
    assert torch.equal(sample.edge_index[0, :4], torch.zeros(4, dtype=torch.long))
    assert torch.equal(sample.edge_index[0, 4:], torch.full((4,), 4, dtype=torch.long))

    forbidden = {tuple(sorted((node, node + 1))) for node in range(11)}
    forbidden.update({(0, 10), (4, 10)})
    sampled = [tuple(sorted(pair)) for pair in sample.edge_index.t().tolist()]
    assert len(sampled) == len(set(sampled))
    assert not set(sampled).intersection(forbidden)
    assert all(source != target for source, target in sampled)

    restored = _sampler()
    restored.load_state_dict(state)
    replay = restored.sample(positives, negatives_per_positive=4)
    assert torch.equal(replay.edge_index, sample.edge_index)
    assert replay.component_labels == sample.component_labels


def test_mixed_sampler_fails_or_explicitly_audits_batch_fallback() -> None:
    # Source one makes three tails cutoff-visible without forbidding them for
    # the positive query from source zero.  The graph is bipartite so there is
    # deliberately no structural hard pool.
    visible = torch.tensor([[1, 1, 1], [0, 2, 3]], dtype=torch.long)
    sampler = CausalMixedNegativeSampler(
        source_count=2,
        target_count=4,
        visible_edge_index=visible,
        visible_edge_time=torch.ones(3),
        cutoff_time=3.0,
        seed=5,
        directed=True,
        same_node_space=False,
    )
    positive = torch.tensor([[0], [1]], dtype=torch.long)
    with pytest.raises(ValueError, match="explicit batch fallback was not allowed"):
        sampler.sample(positive, negatives_per_positive=1)
    fallback_sampler = CausalMixedNegativeSampler(
        source_count=2,
        target_count=4,
        visible_edge_index=visible,
        visible_edge_time=torch.ones(3),
        cutoff_time=3.0,
        seed=5,
        directed=True,
        same_node_space=False,
    )
    sampled = fallback_sampler.sample(
        positive, negatives_per_positive=1, allow_batch_fallback=True
    )
    assert sampled.component_labels == ("hard_fallback_uniform",)
    assert sampled.actual_component_counts == {"hard_fallback_uniform": 1}
    assert sampled.fallback_events == ("query=0:hard->uniform",)


def test_mixed_sampler_keeps_every_component_in_positive_target_type() -> None:
    node_types = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)
    # Topic zero intentionally crosses the endpoint-type boundary.  Before the
    # compatibility constraint, hard-topic, degree, and uniform draws could all
    # corrupt a relation with a target of the wrong type.
    topic_groups = torch.tensor([0, 1, 0, 1, 0, 1] * 2, dtype=torch.long)
    # A third source supplies cutoff-visible candidates of both types without
    # making them historical positives for either evaluated query.
    visible = torch.tensor([[2] * 12, list(range(12))], dtype=torch.long)
    positives = torch.tensor([[0, 1], [0, 6]], dtype=torch.long)

    for seed in range(8):
        sampler = CausalMixedNegativeSampler(
            source_count=3,
            target_count=12,
            visible_edge_index=visible,
            visible_edge_time=torch.ones(12),
            cutoff_time=2.0,
            seed=seed,
            directed=True,
            same_node_space=False,
            node_types=node_types,
            topic_groups=topic_groups,
        )
        sampled = sampler.sample(
            positives, negatives_per_positive=4, allow_batch_fallback=True
        )

        targets = sampled.edge_index[1].reshape(2, 4)
        expected_types = node_types[positives[1]].reshape(2, 1).expand_as(targets)
        assert torch.equal(node_types[targets], expected_types)
        assert sampled.requested_component_counts == {
            "hard": 4,
            "degree_matched": 2,
            "uniform": 2,
        }
        assert sum(sampled.actual_component_counts.values()) == 8
        assert all(
            label in {
                "hard",
                "hard_fallback_uniform",
                "degree_matched",
                "degree_matched_fallback_uniform",
                "uniform",
            }
            for label in sampled.component_labels
        )
        assert len(set(map(tuple, sampled.edge_index.t().tolist()))) == 8
        assert not {(0, 0), (1, 6)}.intersection(
            map(tuple, sampled.edge_index.t().tolist())
        )


def test_typed_fallback_is_compatible_or_fails_closed() -> None:
    # The positive target is the only node of type zero.  Even an explicitly
    # allowed batch fallback must not draw one of the available type-one nodes.
    sampler = CausalMixedNegativeSampler(
        source_count=1,
        target_count=3,
        visible_edge_index=torch.empty((2, 0), dtype=torch.long),
        visible_edge_time=torch.empty(0),
        cutoff_time=0.0,
        seed=3,
        directed=True,
        same_node_space=False,
        node_types=torch.tensor([0, 1, 1]),
    )
    with pytest.raises(ValueError, match="no exact negative remains"):
        sampler.sample(
            torch.tensor([[0], [0]], dtype=torch.long),
            negatives_per_positive=1,
            allow_batch_fallback=True,
        )


def test_degree_fallback_remains_typed_and_audited() -> None:
    node_types = torch.tensor([0] * 6 + [1] * 6, dtype=torch.long)
    # Every legal type-zero tail is visible.  Positive target zero has degree
    # three while every alternative has degree one, forcing the degree component
    # to use its explicitly audited uniform fallback.
    visible = torch.tensor(
        [[1, 2, 3, 1, 1, 1, 1, 1], [0, 0, 0, 1, 2, 3, 4, 5]], dtype=torch.long
    )
    positive = torch.tensor([[0], [0]], dtype=torch.long)

    for seed in range(4):
        sampler = CausalMixedNegativeSampler(
            source_count=4,
            target_count=12,
            visible_edge_index=visible,
            visible_edge_time=torch.ones(8),
            cutoff_time=1.0,
            seed=seed,
            directed=True,
            same_node_space=False,
            node_types=node_types,
        )
        sampled = sampler.sample(
            positive, negatives_per_positive=4, allow_batch_fallback=True
        )
        assert torch.all(node_types[sampled.edge_index[1]] == 0)
        assert sampled.requested_component_counts == {
            "hard": 2,
            "degree_matched": 1,
            "uniform": 1,
        }
        assert sampled.actual_component_counts == {
            "hard_fallback_uniform": 2,
            "degree_matched_fallback_uniform": 1,
            "uniform": 1,
        }
        assert sampled.fallback_events.count("query=0:hard->uniform") == 2
        assert sampled.fallback_events.count("query=0:degree_matched->uniform") == 1


def test_mixed_sampler_has_no_future_input_and_rejects_future_visible_edge() -> None:
    parameters = inspect.signature(CausalMixedNegativeSampler).parameters
    assert all("future" not in name for name in parameters)
    with pytest.raises(ValueError, match="after cutoff_time"):
        CausalMixedNegativeSampler(
            source_count=3,
            target_count=3,
            visible_edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            visible_edge_time=torch.tensor([4.0]),
            cutoff_time=3.0,
            seed=1,
            directed=False,
        )


def test_negative_uniqueness_is_query_local_when_sources_repeat() -> None:
    sampler = CausalMixedNegativeSampler(
        source_count=6,
        target_count=6,
        visible_edge_index=torch.tensor(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long
        ),
        visible_edge_time=torch.ones(5),
        cutoff_time=1.0,
        seed=91,
        directed=False,
    )
    sample = sampler.sample(
        torch.tensor([[0, 0], [1, 2]], dtype=torch.long),
        negatives_per_positive=2,
        allow_batch_fallback=True,
    )
    first = list(map(tuple, sample.edge_index[:, :2].t().tolist()))
    second = list(map(tuple, sample.edge_index[:, 2:].t().tolist()))
    assert len(first) == len(set(first)) == 2
    assert len(second) == len(set(second)) == 2


def test_future_positive_endpoints_never_enter_cutoff_visible_candidate_pool() -> None:
    # Nodes 6-9 are present only as current/future positive endpoints.  They
    # deliberately share the legal target type, so an all-local-node sampler
    # would select them.  The strict sampler may draw only tails 1-5, which
    # appeared in the cutoff-visible graph supplied by source one.
    node_types = torch.zeros(10, dtype=torch.long)
    visible = torch.tensor([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]], dtype=torch.long)
    positives = torch.tensor([[0, 0, 0, 0], [6, 7, 8, 9]], dtype=torch.long)
    sampler = CausalMixedNegativeSampler(
        source_count=2,
        target_count=10,
        visible_edge_index=visible,
        visible_edge_time=torch.ones(5),
        cutoff_time=1.0,
        seed=20260820,
        directed=True,
        same_node_space=False,
        node_types=node_types,
    )

    sampled = sampler.sample(
        positives, negatives_per_positive=4, allow_batch_fallback=True
    )

    assert set(sampled.edge_index[1].tolist()).issubset({1, 2, 3, 4, 5})
    assert not set(sampled.edge_index[1].tolist()).intersection({6, 7, 8, 9})
    assert sampled.future_unseen_candidate_count == 0


def test_mixed_sample_rejects_mislabelled_component_audit() -> None:
    with pytest.raises(ValueError, match="audit does not align"):
        MixedNegativeSample(
            edge_index=torch.tensor([[0, 0], [2, 3]], dtype=torch.long),
            component_labels=("hard_fallback_uniform", "uniform"),
            requested_component_counts={"hard": 1, "uniform": 1},
            # A fallback draw must never be reported as a genuine hard draw.
            actual_component_counts={"hard": 1, "uniform": 1},
            negatives_per_positive=2,
            fallback_events=("query=0:hard->uniform",),
        )
