"""Strict loader for the packaged core dataset recipe catalog."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, strict=True)


class SourceRecipe(_StrictModel):
    source_id: str = Field(alias="sourceId", min_length=1)
    url: str = Field(pattern=r"^https://")
    expected_sha256: str | None = Field(
        alias="expectedSha256", pattern=r"^[0-9a-f]{64}$"
    )
    archive_type: Literal["plain", "zip", "gzip", "mat", "npy", "npz"] = Field(
        alias="archiveType"
    )
    max_bytes: int = Field(alias="maxBytes", gt=0)
    inventory: tuple[str, ...] = Field(strict=False, min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self):
        if len(set(self.inventory)) != len(self.inventory):
            raise ValueError("source inventory must not contain duplicates")
        if any(not item or "\\" in item or item.startswith("/") for item in self.inventory):
            raise ValueError("source inventory must use nonempty relative POSIX paths")
        return self


class TaskRecipe(_StrictModel):
    target_field: str = Field(alias="targetField", min_length=1)
    offline_benchmark_only: bool = Field(alias="offlineBenchmarkOnly")


class LeaveOneDomainOutFold(_StrictModel):
    fold_id: str = Field(alias="foldId", min_length=1)
    validation_domain: str = Field(alias="validationDomain", min_length=1)
    test_domain: str = Field(alias="testDomain", min_length=1)


class DatasetRecipe(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-dataset-recipe/1.0"] = Field(alias="schemaVersion")
    recipe_id: str = Field(alias="recipeId", min_length=1)
    recipe_version: str = Field(alias="recipeVersion", min_length=1)
    recipe_sha256: str = Field(alias="recipeSha256", pattern=r"^[0-9a-f]{64}$")
    citation: str = Field(min_length=1)
    license_note: str = Field(alias="licenseNote", min_length=1)
    usage_scope: Literal[
        "public-serving-eligible", "local-research-demo-only", "mixed"
    ] = Field(alias="usageScope")
    graph_usage_scopes: dict[
        str, Literal["public-serving-eligible", "local-research-demo-only"]
    ] = Field(alias="graphUsageScopes")
    sources: tuple[SourceRecipe, ...] = Field(strict=False, min_length=1)
    graph_ids: tuple[str, ...] = Field(alias="graphIds", strict=False, min_length=1)
    feature_schema: dict[str, Any] = Field(alias="featureSchema")
    split_policy: str = Field(alias="splitPolicy", min_length=1)
    official_split_count: int | None = Field(alias="officialSplitCount", ge=1)
    leave_one_domain_out_folds: tuple[LeaveOneDomainOutFold, ...] = Field(
        alias="leaveOneDomainOutFolds", strict=False
    )
    tasks: dict[str, TaskRecipe]
    serving_task_inventory: tuple[str, ...] = Field(alias="servingTaskInventory", strict=False)
    excluded_model_fields: tuple[str, ...] = Field(alias="excludedModelFields", strict=False)
    output_semantics: str | None = Field(alias="outputSemantics")

    @model_validator(mode="after")
    def validate_recipe(self):
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source IDs must be unique")
        if not set(self.serving_task_inventory) <= set(self.tasks):
            raise ValueError("serving task inventory references an unknown task")
        if any(self.tasks[name].offline_benchmark_only for name in self.serving_task_inventory):
            raise ValueError("offline benchmark tasks cannot enter serving inventory")
        if set(self.graph_usage_scopes) != set(self.graph_ids):
            raise ValueError("graph usage scopes must cover the exact graph inventory")
        graph_scopes = set(self.graph_usage_scopes.values())
        if self.usage_scope == "mixed" and len(graph_scopes) < 2:
            raise ValueError("mixed dataset usage requires mixed graph scopes")
        if self.usage_scope != "mixed" and graph_scopes != {self.usage_scope}:
            raise ValueError("dataset and graph usage scopes disagree")
        return self


class RecipeCatalog(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-dataset-recipe-catalog/1.0"] = Field(
        alias="schemaVersion"
    )
    recipes: tuple[DatasetRecipe, ...] = Field(strict=False, min_length=1)


def _recipe_hash(payload: dict[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != "recipeSha256"})


def load_dataset_recipes(path: Path | None = None) -> dict[str, DatasetRecipe]:
    """Load and verify the immutable packaged recipe catalog."""

    if path is None:
        serialized = files(__package__).joinpath("recipes.json").read_text(encoding="utf-8")
    else:
        serialized = path.read_text(encoding="utf-8")
    raw = json.loads(serialized)
    for recipe in raw.get("recipes", ()):
        if recipe.get("recipeSha256") != _recipe_hash(recipe):
            raise ValueError(f"recipeSha256 mismatch for {recipe.get('recipeId', '<unknown>')}")
    catalog = RecipeCatalog.model_validate(raw)
    keyed = {recipe.recipe_id: recipe for recipe in catalog.recipes}
    if len(keyed) != len(catalog.recipes):
        raise ValueError("recipe IDs must be unique")
    return keyed


def serving_task_inventory(
    recipe_id: str,
    *,
    graph_id: str,
    deployment_scope: Literal["local-demo", "public-commercial"],
) -> tuple[str, ...]:
    """Return the serving inventory after enforcing graph-level usage scope."""

    if deployment_scope not in {"local-demo", "public-commercial"}:
        raise ValueError("deployment scope must be local-demo or public-commercial")
    recipe = load_dataset_recipes().get(recipe_id)
    if recipe is None or graph_id not in recipe.graph_usage_scopes:
        raise ValueError("recipe or graph identifier is not in the packaged catalog")
    if (
        deployment_scope == "public-commercial"
        and recipe.graph_usage_scopes[graph_id] == "local-research-demo-only"
    ):
        return ()
    return recipe.serving_task_inventory


__all__ = [
    "DatasetRecipe",
    "LeaveOneDomainOutFold",
    "SourceRecipe",
    "TaskRecipe",
    "load_dataset_recipes",
    "serving_task_inventory",
]
