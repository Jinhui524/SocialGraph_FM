from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import ValidationError

from socialgraph_gfm.governance.skill_catalog import (
    CATALOG_NAMESPACE,
    CATALOG_SCHEMA_VERSION,
    load_runtime_skill_catalog,
)
from socialgraph_gfm.governance.skill_contracts import (
    DraftReportParams,
    EmptyParams,
    EvidenceParams,
    IndexCaseParams,
    InspectGraphParams,
    KnowledgeSearchParams,
    PageParams,
    RelationParams,
    RunGovernanceAnalysisParams,
    SimilarCaseParams,
)
from socialgraph_gfm.governance.skills import PUBLIC_SKILLS, CommandRequest


def _catalog_root() -> Path:
    return Path(__file__).resolve().parents[3] / "skills" / "governance"


def _vectors() -> dict[str, Any]:
    return json.loads(
        (_catalog_root() / "vectors" / "contract-vectors.json").read_text(encoding="utf-8")
    )


_INTERNAL_PARAM_MODELS = {
    "inspect_graph": InspectGraphParams,
    "run_governance_analysis": RunGovernanceAnalysisParams,
    "get_evidence_subgraph": EvidenceParams,
    "discover_coordination_groups": PageParams,
    "rank_coordination_relations": RelationParams,
    "retrieve_similar_cases": SimilarCaseParams,
    "get_model_dataset_cards": EmptyParams,
    "draft_review_report": DraftReportParams,
    "index_case": IndexCaseParams,
    "search_knowledge": KnowledgeSearchParams,
}


def test_isolated_runtime_uses_canonical_public_order_and_separate_internal_commands() -> None:
    vectors = _vectors()
    catalog = load_runtime_skill_catalog()

    assert CATALOG_NAMESPACE == "socialgraph-fm.product-skills.governance"
    assert CATALOG_SCHEMA_VERSION == vectors["catalogSchemaVersion"]
    assert list(PUBLIC_SKILLS) == vectors["orderedSkills"]
    assert [item.name for item in catalog.items if item.read_only] == vectors["readOnlySkills"]
    assert [item.name for item in catalog.items if item.confirmation_required] == vectors[
        "confirmationGatedSkills"
    ]

    commands = set(get_args(CommandRequest.model_fields["command"].annotation))
    assert commands == set(PUBLIC_SKILLS) | {"index_case", "search_knowledge"}
    assert all(item.internal_command in commands for item in catalog.items)


@pytest.mark.parametrize("vector", _vectors()["validInternalParameters"])
def test_internal_parameter_models_accept_canonical_vectors(vector: dict[str, Any]) -> None:
    _INTERNAL_PARAM_MODELS[vector["command"]].model_validate(vector["params"])


@pytest.mark.parametrize("vector", _vectors()["invalidInternalParameters"])
def test_internal_parameter_models_reject_canonical_vectors(vector: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _INTERNAL_PARAM_MODELS[vector["command"]].model_validate(vector["params"])
