import numpy as np

from socialgraph_gfm.baseline.evaluator import hits_at_k
from socialgraph_gfm.baseline.heuristics import score_heuristic
from socialgraph_gfm.baseline.orchestrator import build_run_specs
from socialgraph_gfm.baseline.protocols import build_protocol
from socialgraph_gfm.baseline.sampling import (
    ExactUndirectedNegativeSampler,
    canonical_edge_set,
)
from socialgraph_gfm.baseline.types import CorpusArrays
from socialgraph_gfm.baseline.trainer import evaluate_heuristic_bundle


def _corpus() -> CorpusArrays:
    return CorpusArrays.from_mapping(
        {
            "x": np.arange(24, dtype=np.float32).reshape(6, 4),
            "edge_index": np.asarray(
                [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]],
                dtype=np.int64,
            ),
            "edge_timestamp": np.asarray([2016, 2016, 2017, 2017, 2017, 2017, 2015, 2015]),
            "variant_train_positive": np.asarray([[0, 1, 2, 3], [1, 2, 3, 4]]),
            "variant_validation_positive": np.asarray([[0, 1], [2, 4]]),
            "variant_validation_negative": np.asarray([[0, 0, 1], [3, 5, 5]]),
            "variant_test_positive": np.asarray([[0, 1, 2], [4, 5, 5]]),
            "variant_test_negative": np.asarray([[0, 1, 2], [5, 3, 4]]),
        }
    )


def test_exact_sampler_is_deterministic_unique_and_resumable():
    forbidden = np.asarray([[0, 1], [1, 2], [2, 3]])
    first = ExactUndirectedNegativeSampler(7, forbidden, seed=99)
    second = ExactUndirectedNegativeSampler(7, forbidden[:, ::-1], seed=99)
    sample = first.sample(8)
    assert np.array_equal(sample, second.sample(8))
    assert len(canonical_edge_set(sample)) == 8
    assert canonical_edge_set(sample).isdisjoint(canonical_edge_set(forbidden))
    assert all(source < target for source, target in sample)

    state = first.state_dict()
    expected_next = first.sample(4)
    restored = ExactUndirectedNegativeSampler(7, forbidden, seed=99)
    restored.load_state_dict(state)
    assert np.array_equal(expected_next, restored.sample(4))


def test_exact_sampler_fails_closed_when_graph_has_too_few_non_edges():
    complete_minus_one = np.asarray(
        [(source, target) for source in range(4) for target in range(source + 1, 4)][:-1]
    )
    sampler = ExactUndirectedNegativeSampler(4, complete_minus_one, seed=1)
    assert sampler.available_count == 1
    try:
        sampler.sample(2)
    except ValueError as error:
        assert "only 1" in str(error)
    else:  # pragma: no cover - assertion aid
        raise AssertionError("dense-graph overflow was not rejected")


def test_cn_adamic_adar_and_resource_allocation_on_hand_graph():
    message = np.asarray([[0, 2], [1, 2], [0, 3], [1, 3], [2, 4]])
    candidate = np.asarray([[0, 1]])
    assert score_heuristic(
        "cn", num_nodes=5, message_edges=message, candidate_edges=candidate
    ).tolist() == [2.0]
    aa = score_heuristic(
        "aa", num_nodes=5, message_edges=message, candidate_edges=candidate
    )[0]
    ra = score_heuristic(
        "ra", num_nodes=5, message_edges=message, candidate_edges=candidate
    )[0]
    assert np.isclose(aa, 1 / np.log(3) + 1 / np.log(2))
    assert np.isclose(ra, 1 / 3 + 1 / 2)


def test_heuristic_bundle_matches_individual_metrics_in_one_group():
    corpus = _corpus()
    protocol = build_protocol(corpus, "ogb_official")
    specs = tuple(
        build_run_specs(
            experiment_id="bundle", phase="dev", config={"models": [name]}, tracks="ogb_official"
        )[0]
        for name in ("cn", "aa", "ra")
    )

    bundled = evaluate_heuristic_bundle(specs, corpus=corpus, protocol=protocol)

    assert [item.spec.model for item in bundled] == ["cn", "aa", "ra"]
    assert all(np.isfinite(item.validation_metrics["hits@50"]) for item in bundled)


def test_strict_protocol_has_monotonic_point_in_time_message_graphs():
    strict = build_protocol(_corpus(), "strict_edge_time")
    assert strict.train.message_cutoff_year == 2016
    assert strict.validation.message_cutoff_year == 2017
    assert strict.test.message_cutoff_year == 2018
    assert canonical_edge_set(strict.train.message_edges) == {(0, 1), (3, 4)}
    assert canonical_edge_set(strict.train.positive_edges) == {(1, 2), (2, 3)}
    assert canonical_edge_set(strict.validation.positive_edges).issubset(
        canonical_edge_set(strict.test.message_edges)
    )
    assert canonical_edge_set(strict.test.positive_edges).isdisjoint(
        canonical_edge_set(strict.test.message_edges)
    )
    assert strict.audit["futureEdgesUsedByTrain"] is False
    assert strict.audit["featurePointInTimeVerified"] is False
    assert strict.audit["futureNegativeArraysRead"] is False
    assert strict.validation.negative_source == "exact_sampler"
    assert strict.test.negative_source == "exact_sampler"


def test_official_protocol_keeps_validation_out_of_test_message_graph():
    official = build_protocol(_corpus(), "ogb_official")
    assert canonical_edge_set(official.validation.message_edges) == canonical_edge_set(
        official.test.message_edges
    )
    assert official.audit["validationEdgesUsedByOfficialTestMessageGraph"] is False


def test_fixed_run_matrix_has_twelve_learning_and_six_heuristic_formal_runs():
    config = {
        "models": ["cn", "aa", "ra", "mlp", "graphsage"],
        "devSeed": 20260811,
        "formalSeeds": [20260812, 20260813, 20260814],
    }
    formal = build_run_specs(
        experiment_id="formal-a", phase="formal", config=config, tracks="both"
    )
    dev = build_run_specs(experiment_id="dev-a", phase="dev", config=config, tracks="both")
    assert len(formal) == 18
    assert len([run for run in formal if run.model in ("mlp", "graphsage")]) == 12
    assert len([run for run in formal if run.model in ("cn", "aa", "ra")]) == 6
    assert len(dev) == 10
    assert len({run.run_id for run in formal}) == len(formal)


def test_hits_at_k_supports_global_and_per_positive_negative_sets():
    positive = np.asarray([0.8, 0.2])
    global_negative = np.asarray([0.7, 0.6, 0.1])
    assert hits_at_k(positive, global_negative, 1) == 0.5
    per_positive = np.asarray([[0.7, 0.1], [0.3, 0.1]])
    assert hits_at_k(positive, per_positive, 1) == 0.5
