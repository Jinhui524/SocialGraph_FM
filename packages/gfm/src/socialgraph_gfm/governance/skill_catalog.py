"""Repository-backed SocialGraph-FM Governance product-skill catalog for the isolated runtime."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

CATALOG_SCHEMA_VERSION = "socialgraph-fm.governance-skills/1.0"
CATALOG_NAMESPACE = "socialgraph-fm.product-skills.governance"
_RELATIVE_PATH = Path("skills/governance/catalog.json")


@dataclass(frozen=True)
class RuntimeSkillDefinition:
    name: str
    read_only: bool
    confirmation_required: bool
    internal_command: str
    parameter_schema: dict[str, Any]


@dataclass(frozen=True)
class RuntimeSkillCatalog:
    schema_version: str
    implementation_version: str
    items: tuple[RuntimeSkillDefinition, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.items)


def _path() -> Path:
    explicit = os.environ.get("SOCIALGRAPH_GOVERNANCE_SKILL_CATALOG")
    if explicit:
        return Path(explicit).expanduser().resolve(strict=True)
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / _RELATIVE_PATH
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise RuntimeError(
        "canonical Governance skill catalog was not found; run from a SocialGraph-FM checkout "
        "or set SOCIALGRAPH_GOVERNANCE_SKILL_CATALOG"
    )


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid Governance skill contract: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Governance skill contract must be an object: {path}")
    return value


@lru_cache(maxsize=1)
def load_runtime_skill_catalog() -> RuntimeSkillCatalog:
    path = _path()
    root = path.parent.resolve(strict=True)
    source = _read_object(path)
    if (
        source.get("namespace") != CATALOG_NAMESPACE
        or source.get("schemaVersion") != CATALOG_SCHEMA_VERSION
        or not isinstance(source.get("implementationVersion"), str)
        or not isinstance(source.get("items"), list)
        or len(source["items"]) != 8
    ):
        raise RuntimeError("unsupported canonical SocialGraph-FM Governance catalog")
    definitions: list[RuntimeSkillDefinition] = []
    seen: set[str] = set()
    for raw in source["items"]:
        if not isinstance(raw, dict):
            raise TypeError("SocialGraph-FM Governance catalog entries must be objects")
        name = raw.get("name")
        read_only = raw.get("readOnly")
        confirmation = raw.get("confirmationRequired")
        internal = raw.get("internalCommand")
        schema_ref = raw.get("parameterSchema")
        if (
            not isinstance(name, str)
            or name in seen
            or not isinstance(read_only, bool)
            or not isinstance(confirmation, bool)
            or read_only == confirmation
            or not isinstance(internal, str)
            or not isinstance(schema_ref, str)
            or "\\" in schema_ref
            or Path(schema_ref).is_absolute()
        ):
            raise RuntimeError(f"invalid SocialGraph-FM Governance catalog entry: {name!r}")
        schema_path = (root / schema_ref).resolve(strict=True)
        try:
            schema_path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Governance parameter schema escaped its version root") from error
        schema = _read_object(schema_path)
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise RuntimeError(f"Governance parameter schema is not strict: {schema_ref}")
        seen.add(name)
        definitions.append(
            RuntimeSkillDefinition(
                name=name,
                read_only=read_only,
                confirmation_required=confirmation,
                internal_command=internal,
                parameter_schema=schema,
            )
        )
    return RuntimeSkillCatalog(
        schema_version=CATALOG_SCHEMA_VERSION,
        implementation_version=source["implementationVersion"],
        items=tuple(definitions),
    )


__all__ = [
    "CATALOG_NAMESPACE",
    "CATALOG_SCHEMA_VERSION",
    "RuntimeSkillCatalog",
    "RuntimeSkillDefinition",
    "load_runtime_skill_catalog",
]
