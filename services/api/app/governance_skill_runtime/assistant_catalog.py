"""Load the canonical read-only LLM Assistant Skill catalog."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "socialgraph-fm.product-skills.assistant/1.0"
_NAMESPACE = "socialgraph-fm.product-skills.assistant"
_CATALOG_RELATIVE_PATH = Path("skills/assistant/catalog.json")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,99}$")
_EXPECTED_NAMES = (
    "answer_governance_question",
    "summarize_node_evidence",
    "generate_global_situation_report",
    "generate_account_evidence_report",
    "generate_coordination_report",
    "generate_case_review_draft",
)
_READ_ONLY_GOVERNANCE_SKILLS = frozenset(
    {
        "inspect_graph",
        "get_evidence_subgraph",
        "discover_coordination_groups",
        "rank_coordination_relations",
        "retrieve_similar_cases",
        "get_model_dataset_cards",
    }
)


@dataclass(frozen=True)
class AssistantSkillDefinition:
    name: str
    label: str
    description: str
    ui_location: str
    governance_skills: tuple[str, ...]
    parameter_schema: dict[str, Any]


@dataclass(frozen=True)
class AssistantSkillCatalogSource:
    root: Path
    items: tuple[AssistantSkillDefinition, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.items)


def _catalog_path() -> Path:
    explicit = os.environ.get("SOCIALGRAPH_ASSISTANT_SKILL_CATALOG")
    if explicit:
        candidate = Path(explicit).expanduser().resolve(strict=True)
        if candidate.name != "catalog.json":
            raise RuntimeError("SOCIALGRAPH_ASSISTANT_SKILL_CATALOG must name catalog.json")
        return candidate
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / _CATALOG_RELATIVE_PATH
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise RuntimeError("canonical Assistant Skill catalog was not found")


@lru_cache(maxsize=1)
def load_assistant_skill_catalog() -> AssistantSkillCatalogSource:
    path = _catalog_path()
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("invalid Assistant Skill catalog") from error
    if not isinstance(source, dict) or set(source) != {
        "$schema",
        "namespace",
        "schemaVersion",
        "items",
    }:
        raise RuntimeError("Assistant Skill catalog root fields are invalid")
    if (
        source["$schema"] != "https://json-schema.org/draft/2020-12/schema"
        or source["namespace"] != _NAMESPACE
        or source["schemaVersion"] != _SCHEMA_VERSION
        or not isinstance(source["items"], list)
        or len(source["items"]) != 6
    ):
        raise RuntimeError("Assistant Skill catalog identity is invalid")
    items: list[AssistantSkillDefinition] = []
    for raw in source["items"]:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "label",
            "description",
            "uiLocation",
            "readOnly",
            "confirmationRequired",
            "governanceSkills",
            "parameterSchema",
        }:
            raise RuntimeError("Assistant Skill catalog item fields are invalid")
        name = raw["name"]
        chains = raw["governanceSkills"]
        parameter_schema = raw["parameterSchema"]
        if (
            not isinstance(name, str)
            or _NAME.fullmatch(name) is None
            or raw["readOnly"] is not True
            or raw["confirmationRequired"] is not False
            or not all(
                isinstance(raw[key], str) and raw[key].strip()
                for key in ("label", "description", "uiLocation")
            )
            or not isinstance(chains, list)
            or not chains
            or len(chains) != len(set(chains))
            or not all(item in _READ_ONLY_GOVERNANCE_SKILLS for item in chains)
            or not isinstance(parameter_schema, dict)
            or parameter_schema.get("type") != "object"
            or parameter_schema.get("additionalProperties") is not False
        ):
            raise RuntimeError(f"invalid Assistant Skill catalog entry: {name!r}")
        skill_doc = path.parent / name / "SKILL.md"
        if not skill_doc.is_file():
            raise RuntimeError(f"Assistant Skill documentation is missing: {name}")
        items.append(
            AssistantSkillDefinition(
                name=name,
                label=raw["label"].strip(),
                description=raw["description"].strip(),
                ui_location=raw["uiLocation"].strip(),
                governance_skills=tuple(chains),
                parameter_schema=parameter_schema,
            )
        )
    catalog = AssistantSkillCatalogSource(root=path.parent, items=tuple(items))
    if catalog.names != _EXPECTED_NAMES:
        raise RuntimeError("Assistant Skill names or order do not match the public contract")
    return catalog


__all__ = [
    "AssistantSkillCatalogSource",
    "AssistantSkillDefinition",
    "load_assistant_skill_catalog",
]
