"""Workflow-facing checks and numeric loaders for formal GFM domains."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ...errors import ContractViolation
from ...runtime import RuntimeLayout
from ...canonical import canonical_sha256, file_sha256
from .common import (
    array_inventory,
    load_npz_safe,
    read_json_object,
    resolve_within,
    verify_manifest,
)

AccessRole = Literal["train", "validation", "test", "shadow"]
ACCESS_ROLES: tuple[AccessRole, ...] = ("train", "validation", "test", "shadow")
PHYSICAL_ACCESS_SCHEMA = "gfm.physical-role-views/1.0"

DOMAIN_CORPORA = {
    "openalex-graph-ai": ("gfm.openalex-corpus/1.0", "openalex-graph-ai"),
    "thgl-software-2.0.0": ("gfm.thgl-software-corpus/1.0", "thgl-software-2.0.0"),
    "wikimedia-talk-article-2011-2015": (
        "gfm.wikimedia-corpus/1.0",
        "wikimedia-talk-article-2011-2015",
    ),
}


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"GFM domains: {message}")


def _check(root: str | Path, domain_id: str) -> tuple[Path, dict[str, Any]]:
    if domain_id not in DOMAIN_CORPORA:
        raise _fail(f"unsupported domain {domain_id!r}")
    schema, directory_name = DOMAIN_CORPORA[domain_id]
    directory = RuntimeLayout.from_root(root).processed_gfm / directory_name
    # Route every load through the domain semantic checker, not merely the
    # portable manifest/hash verifier.  These lazy imports avoid an import
    # cycle while proving OpenAlex join semantics, official TGB splits, and
    # Wikimedia page-disjoint last-event split assignment before arrays load.
    if domain_id == "openalex-graph-ai":
        from .openalex import check_openalex

        manifest = check_openalex(root)
    elif domain_id == "thgl-software-2.0.0":
        from .thgl_software import check_thgl_software

        manifest = check_thgl_software(root)
    else:
        from .wikimedia import check_wikimedia

        manifest = check_wikimedia(root)
    if manifest.get("schemaVersion") != schema or manifest.get("domainId") != domain_id:
        raise _fail(f"domain manifest identity mismatch for {domain_id}")
    # Keep a uniform final portable boundary even though each checker also
    # revalidates its manifest and shards internally.
    verify_manifest(directory, manifest)
    return directory, manifest


def check_all_gfm_corpora(root: str | Path) -> dict[str, Any]:
    manifests = {}
    for domain_id in DOMAIN_CORPORA:
        _, manifest = _check(root, domain_id)
        manifests[domain_id] = manifest
    return {
        "schemaVersion": "gfm.corpora-check/1.0",
        "ready": len(manifests) == 3,
        "domains": manifests,
    }


def load_domain(root: str | Path, domain_id: str) -> dict[str, Any]:
    """Revalidate a domain and concatenate each declared numeric shard family.

    Qualified ``<family>.<array>`` keys are always exposed.  The canonical
    unqualified temporal graph is the ``events`` family; otherwise an
    unqualified convenience key is exposed only when one family owns that
    array name.  This prevents OpenAlex ``events.src`` and ``targets.src`` from
    ever being concatenated together.
    """

    directory, manifest = _check(root, domain_id)
    pieces: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    name_families: dict[str, set[str]] = defaultdict(set)
    for shard in manifest["shards"]:
        records = shard["arrays"]
        if not records:
            continue
        path = resolve_within(directory, str(shard["path"]))
        expected = {
            str(record["name"]): (str(record["dtype"]), len(record["shape"]))
            for record in records
        }
        loaded = load_npz_safe(path, expected=expected)
        stem = Path(str(shard["path"])).stem
        prefix, separator, suffix = stem.rpartition("-")
        family = prefix if separator and len(suffix) == 5 and suffix.isdigit() else stem
        for name, array in loaded.items():
            pieces[(family, name)].append(array)
            if not family.startswith("access-") and not family.startswith("rv-"):
                name_families[name].add(family)
    arrays: dict[str, np.ndarray] = {}
    for (family, name), values in pieces.items():
        combined = np.concatenate(values) if len(values) > 1 else values[0]
        arrays[f"{family}.{name}"] = combined
        # Compatibility for callers written before multi-shard corpora.  The
        # alias denotes the complete family, never merely physical shard zero.
        arrays[f"{family}-00000.{name}"] = combined
        if family == "events" or (
            not family.startswith("access-")
            and not family.startswith("rv-")
            and len(name_families[name]) == 1
        ):
            arrays[name] = combined
    if not arrays:
        raise _fail(f"domain {domain_id} contains no numeric arrays")
    return {"manifest": manifest, "arrays": arrays}


def _boundary_manifest(root: str | Path, domain_id: str) -> tuple[Path, dict[str, Any]]:
    """Validate manifest identity without touching any declared data artifact.

    A full corpus check intentionally reads every shard.  Formal training must
    have a stronger process boundary: test bytes cannot be opened merely to
    initialise a train/validation process.  This function therefore verifies
    the signed logical manifest payload and identity only.  Selected artifacts
    are hash- and inventory-checked later, immediately before they are opened.
    """

    if domain_id not in DOMAIN_CORPORA:
        raise _fail(f"unsupported domain {domain_id!r}")
    schema, directory_name = DOMAIN_CORPORA[domain_id]
    directory = RuntimeLayout.from_root(root).processed_gfm / directory_name
    manifest_path = directory / "manifest.json"
    manifest = read_json_object(manifest_path)
    logical_hash = manifest.get("logicalHash")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"logicalHash", "createdAt"}
    }
    if (
        manifest.get("schemaVersion") != schema
        or manifest.get("domainId") != domain_id
        or not isinstance(logical_hash, str)
        or logical_hash != canonical_sha256(payload)
    ):
        raise _fail(f"role-view manifest identity mismatch for {domain_id}")
    return directory, manifest


def _physical_access_contract(
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[str]], dict[str, str]]:
    raw = manifest.get("physicalAccess")
    if not isinstance(raw, dict) or raw.get("schemaVersion") != PHYSICAL_ACCESS_SCHEMA:
        raise _fail("corpus has no physical role-view contract")
    if raw.get("roles") != list(ACCESS_ROLES):
        raise _fail("physical role ordering is invalid")
    raw_families, raw_shared = raw.get("roleFamilies"), raw.get("sharedFamilies")
    merge_order = raw.get("mergeOrder", {})
    if (
        not isinstance(raw_families, dict)
        or not isinstance(raw_shared, dict)
        or not isinstance(merge_order, dict)
    ):
        raise _fail("physical role-view family inventory is invalid")
    declared_records = manifest.get("shards")
    if not isinstance(declared_records, list):
        raise _fail("manifest shard inventory is invalid")
    declared = {
        str(item.get("path"))
        for item in declared_records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    role_families: dict[str, dict[str, list[str]]] = {}
    used: set[str] = set()
    for family, raw_roles in raw_families.items():
        if not isinstance(family, str) or not family or not isinstance(raw_roles, dict):
            raise _fail("physical role family is malformed")
        if set(raw_roles) != set(ACCESS_ROLES):
            raise _fail(f"physical role family {family!r} lacks an exact role inventory")
        roles: dict[str, list[str]] = {}
        for role in ACCESS_ROLES:
            paths = raw_roles[role]
            if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
                raise _fail(f"physical role family {family!r} has invalid paths")
            if len(paths) != len(set(paths)) or not set(paths).issubset(declared):
                raise _fail(f"physical role family {family!r} references undeclared paths")
            if used.intersection(paths):
                raise _fail("a physical artifact belongs to more than one access role")
            used.update(paths)
            roles[role] = list(paths)
        role_families[family] = roles
    shared_families: dict[str, list[str]] = {}
    for family, paths in raw_shared.items():
        if (
            not isinstance(family, str)
            or not family
            or not isinstance(paths, list)
            or any(not isinstance(path, str) for path in paths)
            or len(paths) != len(set(paths))
            or not set(paths).issubset(declared)
        ):
            raise _fail("physical shared family inventory is invalid")
        if used.intersection(paths):
            raise _fail("a physical artifact is both shared and role-restricted")
        used.update(paths)
        shared_families[family] = list(paths)
    orders: dict[str, str] = {}
    for family, name in merge_order.items():
        if family not in role_families or name not in {"timestamp", "timestamp-pseudonym"}:
            raise _fail("physical role merge-order declaration is invalid")
        orders[str(family)] = str(name)
    return role_families, shared_families, orders


def _load_selected_record(
    directory: Path, record: dict[str, Any]
) -> dict[str, np.ndarray]:
    path = resolve_within(directory, str(record["path"]))
    if file_sha256(path) != record.get("sha256"):
        raise _fail(f"selected role artifact hash mismatch: {record['path']}")
    raw_inventory = record.get("arrays")
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise _fail(f"selected role artifact has no numeric inventory: {record['path']}")
    expected = {
        str(item["name"]): (str(item["dtype"]), len(item["shape"]))
        for item in raw_inventory
        if isinstance(item, dict)
        and isinstance(item.get("shape"), list)
        and isinstance(item.get("name"), str)
    }
    if len(expected) != len(raw_inventory):
        raise _fail(f"selected role artifact inventory is malformed: {record['path']}")
    loaded = load_npz_safe(path, expected=expected)
    if array_inventory(loaded) != raw_inventory:
        raise _fail(f"selected role artifact metadata mismatch: {record['path']}")
    return loaded


def load_domain_view(
    root: str | Path,
    domain_id: str,
    *,
    maximum_role: AccessRole,
    families: Sequence[str] = ("events",),
) -> dict[str, Any]:
    """Open only explicitly authorised cumulative role shards.

    Unlike :func:`load_domain`, this entry point never invokes a full semantic
    checker and never hashes or opens a later-role artifact.  The corpus must
    first have passed the independent CorpusReady check.  ``families`` is
    explicit and defaults to temporal events only so product labels cannot be
    pulled into pretraining by convenience.
    """

    if maximum_role not in ACCESS_ROLES:
        raise _fail(f"invalid maximum access role {maximum_role!r}")
    if not families or any(not isinstance(value, str) or not value for value in families):
        raise _fail("role-view family selection is invalid")
    selected_families = tuple(dict.fromkeys(families))
    directory, manifest = _boundary_manifest(root, domain_id)
    role_families, shared_families, merge_orders = _physical_access_contract(manifest)
    known = set(role_families) | set(shared_families)
    if not set(selected_families).issubset(known):
        raise _fail(
            f"role-view family selection is unsupported: {sorted(set(selected_families) - known)}"
        )
    role_index = ACCESS_ROLES.index(maximum_role)
    records = {
        str(item["path"]): item
        for item in manifest["shards"]
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    selected_paths: dict[str, list[str]] = {}
    all_restricted_paths: set[str] = set()
    for roles in role_families.values():
        for paths in roles.values():
            all_restricted_paths.update(paths)
    for family in selected_families:
        if family in role_families:
            selected_paths[family] = [
                path
                for role in ACCESS_ROLES[: role_index + 1]
                for path in role_families[family][role]
            ]
        else:
            selected_paths[family] = list(shared_families[family])

    opened: list[str] = []
    output: dict[str, np.ndarray] = {}
    owner_by_name: dict[str, set[str]] = defaultdict(set)
    family_arrays: dict[str, dict[str, np.ndarray]] = {}
    for family, paths in selected_paths.items():
        pieces: dict[str, list[np.ndarray]] = defaultdict(list)
        for path in paths:
            loaded = _load_selected_record(directory, records[path])
            opened.append(path)
            for name, value in loaded.items():
                pieces[name].append(value)
        if not pieces:
            continue
        combined = {
            name: np.concatenate(values) if len(values) > 1 else values[0]
            for name, values in pieces.items()
        }
        order_name = merge_orders.get(family)
        if order_name is not None and combined.get("timestamp") is not None:
            if order_name == "timestamp-pseudonym":
                if "revision_pseudonym" not in combined:
                    raise _fail("timestamp-pseudonym merge lacks revision pseudonyms")
                order = np.lexsort((combined["revision_pseudonym"], combined["timestamp"]))
            else:
                order = np.argsort(combined["timestamp"], kind="stable")
            combined = {name: np.ascontiguousarray(value[order]) for name, value in combined.items()}
        family_arrays[family] = combined
        for name in combined:
            owner_by_name[name].add(family)
    for family, combined in family_arrays.items():
        for name, value in combined.items():
            output[f"{family}.{name}"] = value
            output[f"{family}-00000.{name}"] = value
            if family == "events" or len(owner_by_name[name]) == 1:
                output[name] = value
    if not output:
        raise _fail("selected role view contains no numeric arrays")
    excluded = sorted(all_restricted_paths - set(opened))
    return {
        "manifest": manifest,
        "arrays": output,
        "accessAudit": {
            "schemaVersion": "gfm.domain-role-access-audit/1.0",
            "maximumRole": maximum_role,
            "families": list(selected_families),
            "openedPaths": opened,
            "excludedRestrictedPaths": excluded,
            "testArtifactsOpened": any(
                path in opened
                for family in role_families.values()
                for path in family["test"] + family["shadow"]
            ) if role_index < ACCESS_ROLES.index("test") else any(
                path in opened
                for family in role_families.values()
                for path in family["shadow"]
            ) if role_index < ACCESS_ROLES.index("shadow") else False,
            "manifestLogicalHash": manifest["logicalHash"],
        },
    }
