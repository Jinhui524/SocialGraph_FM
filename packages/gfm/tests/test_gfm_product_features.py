import math

from socialgraph_gfm.gfm.product_features import (
    PAIR_FEATURE_NAMES,
    SupporterCandidate,
    collaboration_pair_features,
    eligible_supporter,
)


def test_collaboration_pair_features_include_strong_heuristics_and_product_signals():
    values = collaboration_pair_features(
        0,
        1,
        neighbors={0: {2, 3}, 1: {2}, 2: {0, 1, 3}, 3: {0, 2}},
        topics={0: {"graph", "nlp"}, 1: {"graph", "software"}},
        institutions={0: "A", 1: "B"},
        inactive_days={0: 30, 1: 90},
    )
    assert len(values) == len(PAIR_FEATURE_NAMES) == 8
    assert values[0] == 1.0
    assert math.isclose(values[1], 1.0 / math.log(3))
    assert math.isclose(values[2], 1.0 / 3.0)
    assert values[5] == 1.0


def test_supporter_gate_requires_experience_recency_no_prior_tie_and_relevance():
    eligible = SupporterCandidate(7, 3, 100, False, 2, False, False)
    assert eligible_supporter(eligible)
    assert not eligible_supporter(SupporterCandidate(8, 2, 100, False, 2, True, True))
    assert not eligible_supporter(SupporterCandidate(9, 5, 800, False, 2, True, True))
    assert not eligible_supporter(SupporterCandidate(10, 5, 100, True, 2, True, True))
