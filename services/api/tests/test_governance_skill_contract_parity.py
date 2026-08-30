from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.governance_skill_runtime.catalog import load_product_skill_catalog
from app.governance_skills import GovernanceSkillsGateway
from app.governance_skills_schemas import (
    SKILL_SCHEMA_VERSION,
    _PARAM_MODELS,
)


def _catalog_root() -> Path:
    return Path(__file__).resolve().parents[3] / "skills" / "governance"


def _vectors() -> dict[str, Any]:
    return json.loads(
        (_catalog_root() / "vectors" / "contract-vectors.json").read_text(encoding="utf-8")
    )


def test_canonical_catalog_drives_api_order_permissions_and_parameter_schemas() -> None:
    canonical = load_product_skill_catalog()
    response = GovernanceSkillsGateway.catalog().model_dump(mode="json", by_alias=True)
    vectors = _vectors()

    assert canonical.schema_version == SKILL_SCHEMA_VERSION
    assert list(canonical.names) == vectors["orderedSkills"]
    assert list(item["name"] for item in response["items"]) == vectors["orderedSkills"]
    assert [item.name for item in canonical.items if item.read_only] == vectors["readOnlySkills"]
    assert [item.name for item in canonical.items if item.confirmation_required] == vectors[
        "confirmationGatedSkills"
    ]

    by_name = {item["name"]: item for item in response["items"]}
    for definition in canonical.items:
        assert by_name[definition.name]["readOnly"] is definition.read_only
        assert (
            by_name[definition.name]["confirmationRequired"]
            is definition.confirmation_required
        )
        assert by_name[definition.name]["parameterSchema"] == definition.parameter_schema


@pytest.mark.parametrize("vector", _vectors()["validParameters"])
def test_public_parameter_models_accept_canonical_valid_vectors(vector: dict[str, Any]) -> None:
    _PARAM_MODELS[vector["skill"]].model_validate(vector["params"])


@pytest.mark.parametrize("vector", _vectors()["invalidParameters"])
def test_public_parameter_models_reject_canonical_invalid_vectors(vector: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _PARAM_MODELS[vector["skill"]].model_validate(vector["params"])
