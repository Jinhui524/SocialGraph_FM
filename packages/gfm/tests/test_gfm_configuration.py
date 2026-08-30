import pytest

from socialgraph_gfm.gfm.configuration import (
    apply_exploratory_overrides,
    load_core_config,
    load_openalex_spec,
)
from socialgraph_gfm.gfm.contracts import GfmPretrainConfig


def test_pinned_gfm_config_has_three_independent_domain_families():
    config = load_core_config()
    assert len(config["configHash"]) == 64
    assert config["runKind"] == "formal"
    assert {item["domainFamily"] for item in config["domains"]} == {
        "academic-collaboration",
        "software-activity",
        "online-community",
    }
    assert config["transfer"] == {"lodoTargetAdaptationSteps": 5000}


def test_lodo_target_steps_are_required_and_hash_bound():
    config = load_core_config()
    original_hash = config["configHash"]
    changed = apply_exploratory_overrides(
        config, {"transfer": {"lodoTargetAdaptationSteps": 4999}}
    )
    assert changed["configHash"] != original_hash
    with pytest.raises(ValueError, match="5000"):
        GfmPretrainConfig.model_validate(changed)


def test_any_formal_override_is_explicitly_exploratory():
    config = load_core_config()
    changed = apply_exploratory_overrides(config, {"optimization": {"learningRate": 0.01}})
    assert changed["runKind"] == "exploratory"
    assert changed["configHash"] != config["configHash"]
    GfmPretrainConfig.model_validate(changed)


def test_openalex_spec_balances_three_clusters_and_blocks_future_aggregates():
    spec = load_openalex_spec()
    assert sum(item["maximumWorks"] for item in spec["topicClusters"]) == 200_000
    assert not set(spec["workSelect"]).intersection(spec["forbiddenFields"])
    assert spec["workTypes"] == ["article", "preprint"]
    assert "proceedings-article" in spec["workTypeCompatibility"]["requestedCategories"]
    # These are exact current OpenAlex Topic display names, not aspirational
    # search phrases.  Topic acquisition fails closed if the API can no longer
    # resolve any one of them exactly.
    selectors = {
        selector
        for cluster in spec["topicClusters"]
        for selector in cluster["selectors"]
    }
    assert {
        "Advanced Graph Neural Networks",
        "Complex Network Analysis Techniques",
        "Natural Language Processing Techniques",
        "Computational and Text Analysis Methods",
        "Software Engineering Research",
        "Open Source Software Innovations",
    }.issubset(selectors)
    assert len(spec["specHash"]) == 64
