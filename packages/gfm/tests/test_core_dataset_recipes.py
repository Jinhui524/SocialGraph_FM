from __future__ import annotations

import json
from importlib.resources import files

import pytest

from socialgraph_gfm.core.datasets.recipes import (
    load_dataset_recipes,
    serving_task_inventory,
)


def test_packaged_recipes_are_strict_hash_bound_and_cover_governance_inventory() -> None:
    recipes = load_dataset_recipes()

    assert set(recipes) == {
        "email-eu-core",
        "facebook100",
        "github-musae",
        "tolokers",
        "twitch-language",
        "wiki-rfa",
    }
    assert all(recipe.schema_version == "socialgraph-fm.core-dataset-recipe/1.0" for recipe in recipes.values())
    assert all(recipe.recipe_sha256 for recipe in recipes.values())
    assert all(source.url.startswith("https://") for recipe in recipes.values() for source in recipe.sources)
    assert all(source.max_bytes > 0 for recipe in recipes.values() for source in recipe.sources)
    assert all(source.inventory for recipe in recipes.values() for source in recipe.sources)

    resource = files("socialgraph_gfm.core.datasets").joinpath("recipes.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    assert payload["schemaVersion"] == "socialgraph-fm.core-dataset-recipe-catalog/1.0"


def test_recipe_governance_policies_match_core_boundaries() -> None:
    recipes = load_dataset_recipes()

    facebook = recipes["facebook100"]
    assert facebook.graph_ids == ("Reed98", "Amherst41", "Johns Hopkins55", "Cornell5", "Penn94")
    assert facebook.official_split_count == 5
    assert facebook.tasks["gender"].offline_benchmark_only is True
    assert "gender" not in facebook.serving_task_inventory
    assert facebook.feature_schema["profileFields"] == "categorical"

    twitch = recipes["twitch-language"]
    assert twitch.graph_ids == ("DE", "EN", "ES", "FR", "PT", "RU")
    assert tuple(fold.test_domain for fold in twitch.leave_one_domain_out_folds) == twitch.graph_ids
    assert twitch.feature_schema["sharedSparseAttributes"] == "multiHotNonText"

    tolokers = recipes["tolokers"]
    assert tolokers.official_split_count == 10
    assert tolokers.tasks["banned"].target_field == "banned"

    wiki = recipes["wiki-rfa"]
    assert wiki.excluded_model_fields == ("TXT", "DAT", "YEA")
    assert wiki.split_policy == "signed-pair-stratified-70-15-15"

    for recipe_id in ("github-musae", "email-eu-core"):
        recipe = recipes[recipe_id]
        assert recipe.split_policy == "spanning-forest-80-10-10"
        assert recipe.output_semantics == "static relation completion"
        assert "future" not in recipe.output_semantics


def test_recipe_loader_rejects_tampered_content(tmp_path) -> None:
    resource = files("socialgraph_gfm.core.datasets").joinpath("recipes.json")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    payload["recipes"][0]["licenseNote"] = "tampered"
    path = tmp_path / "recipes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="recipeSha256"):
        load_dataset_recipes(path)


def test_serving_inventory_enforces_local_research_graph_scopes() -> None:
    assert serving_task_inventory(
        "facebook100", graph_id="Penn94", deployment_scope="public-commercial"
    ) == ()
    assert serving_task_inventory(
        "wiki-rfa", graph_id="wiki-rfa", deployment_scope="public-commercial"
    ) == ()
    assert serving_task_inventory(
        "facebook100", graph_id="Penn94", deployment_scope="local-demo"
    ) == ("resilience",)
    assert "gender" not in serving_task_inventory(
        "facebook100", graph_id="Penn94", deployment_scope="local-demo"
    )
    assert serving_task_inventory(
        "facebook100", graph_id="Reed98", deployment_scope="public-commercial"
    ) == ("resilience",)


def test_serving_inventory_rejects_unknown_runtime_scope() -> None:
    with pytest.raises(ValueError, match="deployment scope"):
        serving_task_inventory(
            "facebook100", graph_id="Penn94", deployment_scope="production"  # type: ignore[arg-type]
        )
