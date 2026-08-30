"""Validated atomic materialization for core datasets."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import urllib.parse
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from ..bundle import CoreGraphBundle, calculate_graph_version_hash, load_core_graph_bundle_json
from ..splits import spanning_forest_link_split
from .acquire import download_source, extract_source_atomic
from .parsers import parse_email_files
from .recipes import load_dataset_recipes


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8", newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_email_materialization(
    target: Path, *, raw_source_paths: Mapping[str, Path] | None = None
) -> Path:
    """Fully validate a published or staged Email-Eu-core materialization."""

    recipe = load_dataset_recipes()["email-eu-core"]
    bundle = load_core_graph_bundle_json((target / "bundle.json").read_bytes())
    manifest = json.loads((target / "materialization-manifest.json").read_text(encoding="utf-8"))
    expected_keys = {
        "schemaVersion", "recipeId", "recipeVersion", "recipeSha256",
        "observedRawSha256", "expectedRawSha256", "combinedSourceSha256",
        "graphVersionHash", "offlineLabelsSha256", "splitSeed", "outputSemantics",
        "manifestSha256",
    }
    if set(manifest) != expected_keys:
        raise ValueError("materialization manifest inventory is invalid")
    without_hash = {key: value for key, value in manifest.items() if key != "manifestSha256"}
    if manifest["manifestSha256"] != canonical_sha256(without_hash):
        raise ValueError("materialization manifest hash mismatch")
    if (
        manifest["schemaVersion"] != "socialgraph-fm.core-dataset-materialization/1.0"
        or manifest["recipeId"] != recipe.recipe_id
        or manifest["recipeVersion"] != recipe.recipe_version
        or manifest["recipeSha256"] != recipe.recipe_sha256
        or manifest["outputSemantics"] != recipe.output_semantics
    ):
        raise ValueError("materialization recipe identity mismatch")
    expected = {source.source_id: source.expected_sha256 for source in recipe.sources}
    observed = manifest["observedRawSha256"]
    if manifest["expectedRawSha256"] != expected or set(observed) != set(expected):
        raise ValueError("materialization raw source inventory mismatch")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in observed.values()
    ):
        raise ValueError("materialization raw source hash is invalid")
    if raw_source_paths is not None:
        if set(raw_source_paths) != set(observed):
            raise ValueError("materialization raw source path inventory mismatch")
        actual = {source_id: _sha256(path) for source_id, path in raw_source_paths.items()}
        if actual != observed:
            raise ValueError("materialization raw source hash mismatch")
    if type(manifest["splitSeed"]) is not int:
        raise ValueError("materialization split seed must be an integer")
    if manifest["combinedSourceSha256"] != canonical_sha256(dict(sorted(observed.items()))):
        raise ValueError("materialization combined source hash mismatch")
    if (
        manifest["graphVersionHash"] != bundle.graph_version_hash
        or bundle.source.source_sha256 != manifest["combinedSourceSha256"]
    ):
        raise ValueError("materialization graph hash mismatch")

    offline = json.loads((target / "offline-community-labels.json").read_text(encoding="utf-8"))
    if set(offline) != {"schemaVersion", "graphId", "labels", "labelsSha256"}:
        raise ValueError("offline label inventory is invalid")
    if (
        offline["schemaVersion"] != "socialgraph-fm.core-offline-community-labels/1.0"
        or offline["graphId"] != "email-eu-core"
        or set(offline["labels"]) != {"department"}
        or offline["labelsSha256"] != canonical_sha256(offline["labels"])
    ):
        raise ValueError("offline label hash or identity mismatch")
    if manifest["offlineLabelsSha256"] != offline["labelsSha256"]:
        raise ValueError("offline label hash is not bound by materialization manifest")
    node_ids = {node.id for node in bundle.nodes}
    if set(offline["labels"]["department"]) != node_ids:
        raise ValueError("offline labels must cover the exact bundle node inventory")
    return target


def materialize_email_from_files(
    *,
    edges_path: Path,
    departments_path: Path,
    raw_source_paths: dict[str, Path] | None = None,
    runtime_root: Path,
    seed: int,
) -> Path:
    """Build an Email-Eu-core bundle in staging and atomically publish after validation."""

    recipe = load_dataset_recipes()["email-eu-core"]
    raw_paths = raw_source_paths or {"edges": edges_path, "departments": departments_path}
    if set(raw_paths) != {"edges", "departments"}:
        raise ValueError("Email-Eu-core requires both declared source paths")
    source_hashes = {key: _sha256(path) for key, path in raw_paths.items()}
    root = runtime_root.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("runtime root must not be a filesystem root")
    target = root / "materialized" / recipe.recipe_id / recipe.recipe_version
    if target.exists():
        raise FileExistsError(f"materialization already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{recipe.recipe_version}.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        parsed = parse_email_files(edges_path, departments_path)
        split = spanning_forest_link_split(
            num_nodes=len(parsed.node_ids), edges=parsed.edges, seed=seed
        )
        role_by_edge = {
            edge: role
            for role, role_edges in (
                ("train", split.train),
                ("validation", split.validation),
                ("test", split.test),
            )
            for edge in role_edges
        }
        combined_source_sha = canonical_sha256(dict(sorted(source_hashes.items())))
        edges = [
            {
                "sourceId": parsed.node_ids[source],
                "targetId": parsed.node_ids[target_index],
                "edgeType": "email",
                "weight": 1.0,
            }
            for source, target_index in parsed.edges
        ]
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
            "directed": False,
            "nodes": [
                {"id": identifier, "index": index}
                for index, identifier in enumerate(parsed.node_ids)
            ],
            "edges": edges,
            "nodeFeatures": [],
            "structuralFeatures": None,
            "source": {
                "sourceName": "SNAP Email-Eu-core",
                "sourceUri": "https://snap.stanford.edu/data/email-Eu-core.html",
                "citation": recipe.citation,
                "sourceSha256": combined_source_sha,
            },
            "splitManifest": {
                "strategy": "spanning-forest-80-10-10",
                "assignments": [
                    {
                        "entityId": f"edge:{parsed.node_ids[source]}:{parsed.node_ids[target_index]}",
                        "role": role_by_edge[(source, target_index)],
                    }
                    for source, target_index in parsed.edges
                ],
            },
        }
        payload["graphVersionHash"] = calculate_graph_version_hash(payload)
        bundle = CoreGraphBundle.model_validate(payload)
        bundle_payload = bundle.model_dump(mode="json", by_alias=True)
        _write_json(staging / "bundle.json", bundle_payload)

        labels = {
            "department": {
                node_id: label
                for node_id, label in zip(
                    parsed.node_ids, parsed.offline_labels["department"], strict=True
                )
            }
        }
        offline = {
            "schemaVersion": "socialgraph-fm.core-offline-community-labels/1.0",
            "graphId": parsed.graph_id,
            "labels": labels,
            "labelsSha256": canonical_sha256(labels),
        }
        _write_json(staging / "offline-community-labels.json", offline)

        expected = {
            source.source_id: source.expected_sha256 for source in sorted(recipe.sources, key=lambda x: x.source_id)
        }
        manifest_without_hash = {
            "schemaVersion": "socialgraph-fm.core-dataset-materialization/1.0",
            "recipeId": recipe.recipe_id,
            "recipeVersion": recipe.recipe_version,
            "recipeSha256": recipe.recipe_sha256,
            "observedRawSha256": dict(sorted(source_hashes.items())),
            "expectedRawSha256": expected,
            "combinedSourceSha256": combined_source_sha,
            "graphVersionHash": bundle.graph_version_hash,
            "offlineLabelsSha256": offline["labelsSha256"],
            "splitSeed": int(seed),
            "outputSemantics": recipe.output_semantics,
        }
        manifest = {
            **manifest_without_hash,
            "manifestSha256": canonical_sha256(manifest_without_hash),
        }
        _write_json(staging / "materialization-manifest.json", manifest)

        validate_email_materialization(staging, raw_source_paths=raw_paths)
        os.replace(staging, target)
        return target
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def materialize_email_eu_core(*, runtime_root: Path, seed: int = 1729) -> Path:
    """Acquire the two official SNAP sources and materialize the real sanity dataset."""

    recipe = load_dataset_recipes()["email-eu-core"]
    root = runtime_root.expanduser().resolve()
    if root == Path(root.anchor):
        raise ValueError("runtime root must not be a filesystem root")
    final_target = root / "materialized" / recipe.recipe_id / recipe.recipe_version
    if final_target.exists():
        raw_directory = (root / "raw" / recipe.recipe_id / recipe.recipe_version).resolve()
        if not raw_directory.is_relative_to(root):
            raise ValueError("Email raw source directory escapes runtime root")
        raw_paths: dict[str, Path] = {}
        for source in recipe.sources:
            filename = Path(
                urllib.parse.unquote(urllib.parse.urlsplit(source.url).path)
            ).name
            raw_path = (raw_directory / filename).resolve()
            if not filename or not raw_path.is_relative_to(raw_directory):
                raise ValueError("Email raw source path escapes catalog runtime directory")
            raw_paths[source.source_id] = raw_path
        return validate_email_materialization(
            final_target, raw_source_paths=raw_paths
        )
    run_staging = root / ".staging" / f"email-{uuid.uuid4().hex}"
    run_staging.mkdir(parents=True)
    raw_downloads: dict[str, Path] = {}
    expanded: dict[str, Path] = {}
    try:
        for source in recipe.sources:
            result = download_source(
                recipe_id=recipe.recipe_id,
                source_id=source.source_id,
                runtime_root=runtime_root,
            )
            raw_downloads[source.source_id] = result.path
            expanded[source.source_id] = extract_source_atomic(
                source_path=result.path,
                source=source,
                target_directory=run_staging / source.source_id,
                max_expanded_bytes=2_000_000,
            )
        return materialize_email_from_files(
            edges_path=expanded["edges"] / "email-Eu-core.txt",
            departments_path=(
                expanded["departments"] / "email-Eu-core-department-labels.txt"
            ),
            raw_source_paths=raw_downloads,
            runtime_root=runtime_root,
            seed=seed,
        )
    finally:
        shutil.rmtree(run_staging, ignore_errors=True)


__all__ = [
    "materialize_email_eu_core",
    "materialize_email_from_files",
    "validate_email_materialization",
]
