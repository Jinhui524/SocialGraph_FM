"""Load and validate the canonical SocialGraph-FM Governance product-skill catalog."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "socialgraph-fm.governance-skills/1.0"
_NAMESPACE = "socialgraph-fm.product-skills.governance"
_CATALOG_RELATIVE_PATH = Path("skills/governance/catalog.json")
_SKILL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,99}$")


@dataclass(frozen=True)
class ProductSkillDefinition:
    """One fully resolved catalog entry."""

    name: str
    read_only: bool
    confirmation_required: bool
    confirmation_action: str | None
    description: str
    parameter_schema_ref: str
    parameter_schema: dict[str, Any]
    internal_command: str


@dataclass(frozen=True)
class ProductSkillCatalog:
    """Validated immutable view of the repository contract."""

    root: Path
    namespace: str
    schema_version: str
    implementation_version: str
    items: tuple[ProductSkillDefinition, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.items)

    @property
    def read_only_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.items if item.read_only)

    @property
    def confirmation_gated_names(self) -> frozenset[str]:
        return frozenset(item.name for item in self.items if item.confirmation_required)

    @property
    def public_to_internal(self) -> dict[str, str]:
        return {item.name: item.internal_command for item in self.items}


def _catalog_path() -> Path:
    explicit = os.environ.get("SOCIALGRAPH_GOVERNANCE_SKILL_CATALOG")
    if explicit:
        candidate = Path(explicit).expanduser().resolve(strict=True)
        if candidate.name != "catalog.json":
            raise RuntimeError("SOCIALGRAPH_GOVERNANCE_SKILL_CATALOG must name catalog.json")
        return candidate
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / _CATALOG_RELATIVE_PATH
        if candidate.is_file():
            return candidate.resolve(strict=True)
    raise RuntimeError(
        "canonical Governance skill catalog was not found; run from a SocialGraph-FM checkout "
        "or set SOCIALGRAPH_GOVERNANCE_SKILL_CATALOG"
    )


def _object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid Governance skill contract: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Governance skill contract must be an object: {path}")
    return value


def _resolved_child(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative or Path(relative).is_absolute():
        raise RuntimeError("parameterSchema must be a repository-relative POSIX path")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError("parameterSchema escaped the skill directory") from error
    if not resolved.is_file():
        raise RuntimeError("parameterSchema must reference a regular file")
    return resolved


@lru_cache(maxsize=1)
def load_product_skill_catalog() -> ProductSkillCatalog:
    """Load the single repository catalog and fail closed on structural drift."""

    path = _catalog_path()
    root = path.parent.resolve(strict=True)
    source = _object(path)
    expected_root_keys = {
        "$schema",
        "contractSchema",
        "namespace",
        "schemaVersion",
        "implementationVersion",
        "items",
    }
    if set(source) != expected_root_keys:
        raise RuntimeError("Governance catalog root fields do not match the v1 contract")
    if source["namespace"] != _NAMESPACE or source["schemaVersion"] != _SCHEMA_VERSION:
        raise RuntimeError("Governance catalog namespace or schema version is unsupported")
    if (
        source["$schema"] != "https://json-schema.org/draft/2020-12/schema"
        or source["contractSchema"] != "schemas/public/catalog-source.schema.json"
    ):
        raise RuntimeError("Governance catalog schema reference is invalid")
    implementation_version = source["implementationVersion"]
    raw_items = source["items"]
    if not isinstance(implementation_version, str) or not isinstance(raw_items, list):
        raise RuntimeError("Governance catalog implementationVersion/items are invalid")
    if len(raw_items) != 8:
        raise RuntimeError("SocialGraph-FM Governance must expose exactly eight public product skills")

    items: list[ProductSkillDefinition] = []
    seen: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise RuntimeError("Governance catalog items must be objects")
        required = {
            "name",
            "readOnly",
            "confirmationRequired",
            "description",
            "parameterSchema",
            "internalCommand",
        }
        allowed = required | {"confirmationAction"}
        if not required <= set(raw) or set(raw) - allowed:
            raise RuntimeError("Governance catalog item fields do not match the v1 contract")
        name = raw["name"]
        internal_command = raw["internalCommand"]
        read_only = raw["readOnly"]
        confirmation_required = raw["confirmationRequired"]
        action = raw.get("confirmationAction")
        if (
            not isinstance(name, str)
            or _SKILL_NAME.fullmatch(name) is None
            or name in seen
            or not isinstance(internal_command, str)
            or _SKILL_NAME.fullmatch(internal_command) is None
            or not isinstance(read_only, bool)
            or not isinstance(confirmation_required, bool)
            or read_only == confirmation_required
            or (confirmation_required and action not in {"run_governance_analysis", "save_draft_report"})
            or (not confirmation_required and action is not None)
            or not isinstance(raw["description"], str)
            or not raw["description"].strip()
            or not isinstance(raw["parameterSchema"], str)
        ):
            raise RuntimeError(f"invalid Governance catalog entry: {name!r}")
        seen.add(name)
        schema_ref = raw["parameterSchema"]
        schema = _object(_resolved_child(root, schema_ref))
        if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
            raise RuntimeError(f"public parameter schema is not strict: {schema_ref}")
        items.append(
            ProductSkillDefinition(
                name=name,
                read_only=read_only,
                confirmation_required=confirmation_required,
                confirmation_action=action,
                description=raw["description"].strip(),
                parameter_schema_ref=schema_ref,
                parameter_schema=schema,
                internal_command=internal_command,
            )
        )
    return ProductSkillCatalog(
        root=root,
        namespace=_NAMESPACE,
        schema_version=_SCHEMA_VERSION,
        implementation_version=implementation_version,
        items=tuple(items),
    )


__all__ = [
    "ProductSkillCatalog",
    "ProductSkillDefinition",
    "load_product_skill_catalog",
]
