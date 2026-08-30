"""Corpus acquisition, deterministic fixture materialization, and contract validation."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import shutil
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from socialgraph_gfm.canonical import canonical_sha256, file_sha256

from ...core.bundle import CoreGraphBundle, calculate_graph_version_hash
from ...core.datasets.parsers import (
    ParsedGraph,
    parse_email_files,
    parse_link_fixture,
    parse_tolokers_fixture,
    parse_tolokers_npz,
    parse_twitch_fixture,
    parse_wiki_rfa,
)
from ...core.splits import EdgeSplit, IndexSplit, SignedEdgeSplit, spanning_forest_link_split
from ...core.structure_features import (
    STRUCTURE_FEATURE_NAMES,
    StructureAlgorithmConfig,
    compute_structure_rows,
)
from ..contracts import (
    ACCOUNT_RISK_TASK,
    COLLABORATION_TASK,
    CONTENT_POLICY_TASK,
    RELEASE_ID,
    RESEARCH_SEED,
    SIGNED_RELATION_TASK,
)
from .common import (
    CORPUS_SCHEMA,
    EXPECTED_SOURCE_HASHES,
    MATERIALIZER_VERSION,
    PARSER_CONTRACTS,
    RESEARCH_SOURCE_RECIPES,
    TWITCH_ARCHIVE_MEMBERS,
    _atomic_json,
    _domain_task_id,
    _read_hashed_document,
    _safe_root,
    load_research_config,
    research_root_from_home,
)


def _stratified_node_split(labels: tuple[int, ...], *, seed: int) -> IndexSplit:
    roles: list[list[int]] = [[], [], []]
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(int(label), []).append(index)
    for label in sorted(by_label):
        indices = sorted(by_label[label])
        random.Random(seed + label * 10_007).shuffle(indices)
        size = len(indices)
        train_count = max(1, int(size * 0.70)) if size else 0
        validation_count = int(size * 0.15)
        if size >= 3 and validation_count == 0:
            validation_count = 1
        if train_count + validation_count > size:
            validation_count = max(0, size - train_count)
        boundaries = (train_count, train_count + validation_count)
        roles[0].extend(indices[: boundaries[0]])
        roles[1].extend(indices[boundaries[0] : boundaries[1]])
        roles[2].extend(indices[boundaries[1] :])
    return IndexSplit(*(tuple(sorted(role)) for role in roles))


def candidate_grouped_signed_split(
    edges: tuple[tuple[int, int, int], ...], *, seed: int
) -> SignedEdgeSplit:
    """Keep every vote for one candidate in one role while roughly stratifying sign mix."""

    by_candidate: dict[int, list[tuple[int, int, int]]] = {}
    for source, target, sign in edges:
        if source == target or sign not in {-1, 1}:
            raise ValueError("candidate grouped split requires valid signed directed edges")
        by_candidate.setdefault(target, []).append((source, target, sign))
    strata: dict[tuple[int, ...], list[int]] = {}
    for candidate, candidate_edges in by_candidate.items():
        signature = tuple(sorted({sign for _, _, sign in candidate_edges}))
        strata.setdefault(signature, []).append(candidate)
    assigned: list[list[tuple[int, int, int]]] = [[], [], []]
    for ordinal, signature in enumerate(sorted(strata)):
        candidates = sorted(strata[signature])
        random.Random(seed + ordinal * 10_007).shuffle(candidates)
        size = len(candidates)
        train_count = max(1, round(size * 0.70)) if size else 0
        validation_count = round(size * 0.15)
        if train_count + validation_count > size:
            validation_count = max(0, size - train_count)
        boundaries = (train_count, train_count + validation_count)
        for role, selected in enumerate(
            (
                candidates[: boundaries[0]],
                candidates[boundaries[0] : boundaries[1]],
                candidates[boundaries[1] :],
            )
        ):
            for candidate in selected:
                assigned[role].extend(by_candidate[candidate])
    return SignedEdgeSplit(*(tuple(sorted(role)) for role in assigned))


def _edge_id(node_ids: tuple[str, ...], edge: tuple[int, int]) -> str:
    return f"edge:{node_ids[edge[0]]}:{node_ids[edge[1]]}"


def _bundle_payload(
    *,
    graph: ParsedGraph,
    graph_id: str,
    source_name: str,
    source_uri: str,
    source_sha256: str,
    citation: str,
    split_strategy: str,
    edge_roles: Mapping[tuple[int, int], str] | None = None,
    node_roles: Mapping[int, str] | None = None,
) -> dict[str, Any]:
    if bool(edge_roles) == bool(node_roles):
        raise ValueError("research bundle requires exactly one node or edge split")
    raw_edges = graph.edges or tuple((left, right) for left, right, _ in graph.signed_edges)
    edges = [
        {
            "sourceId": graph.node_ids[left],
            "targetId": graph.node_ids[right],
            "edgeType": "governance-relation" if graph.signed_edges else "social-relation",
            # Sign labels are deliberately absent from the graph input.
            "weight": 1.0,
        }
        for left, right in raw_edges
    ]
    node_features: list[dict[str, Any]] = []
    for name, numeric_rows in sorted(graph.numeric_features.items()):
        width = len(numeric_rows[0]) if numeric_rows else 0
        for column in range(width):
            node_features.append(
                {
                    "kind": "numeric",
                    "name": f"{name}:{column}",
                    "values": [float(row[column]) for row in numeric_rows],
                }
            )
    for name, categorical_values in sorted(graph.categorical_features.items()):
        node_features.append(
            {"kind": "categorical", "name": name, "values": list(categorical_values)}
        )
    for name, multi_hot_rows in sorted(graph.multi_hot_features.items()):
        offsets = [0]
        flattened_values: list[str] = []
        for row in multi_hot_rows:
            flattened_values.extend(row)
            offsets.append(len(flattened_values))
        node_features.append(
            {
                "kind": "multiHot",
                "name": name,
                "rowOffsets": offsets,
                "values": flattened_values,
            }
        )
    if edge_roles is not None:
        assignments = [
            {"entityId": _edge_id(graph.node_ids, edge), "role": edge_roles[edge]}
            for edge in raw_edges
        ]
    else:
        assert node_roles is not None
        assignments = [
            {"entityId": node_id, "role": node_roles[index]}
            for index, node_id in enumerate(graph.node_ids)
        ]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-graph-bundle/2.0",
        "directed": graph.directed,
        "nodes": [{"id": node_id, "index": index} for index, node_id in enumerate(graph.node_ids)],
        "edges": edges,
        "nodeFeatures": node_features,
        "structuralFeatures": None,
        "source": {
            "sourceName": source_name,
            "sourceUri": source_uri,
            "citation": citation,
            "sourceSha256": source_sha256,
        },
        "splitManifest": {"strategy": split_strategy, "assignments": assignments},
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    preliminary = CoreGraphBundle.model_validate(payload)
    from ...core.adapters import derive_training_selection

    visible_indices = derive_training_selection(preliminary).visible_edge_indices
    structure_rows = compute_structure_rows(
        preliminary,
        visible_edge_indices=visible_indices,
        config=StructureAlgorithmConfig.fixed(),
    )
    payload["structuralFeatures"] = {
        "names": list(STRUCTURE_FEATURE_NAMES),
        "values": structure_rows.tolist(),
    }
    payload["graphVersionHash"] = calculate_graph_version_hash(payload)
    if graph.signed_edges and any(edge["weight"] != 1.0 for edge in edges):
        raise AssertionError("signed labels entered graph input")
    return CoreGraphBundle.model_validate(payload).model_dump(mode="json", by_alias=True)


def _labels_payload(
    *,
    graph_id: str,
    task_id: str,
    graph: ParsedGraph,
    target_name: str | None,
) -> dict[str, Any]:
    if target_name is None:
        targets: list[dict[str, Any]] = []
    elif graph.signed_edges:
        targets = [
            {
                "entityId": _edge_id(graph.node_ids, (left, right)),
                "sourceId": graph.node_ids[left],
                "targetId": graph.node_ids[right],
                "target": 1 if sign > 0 else 0,
            }
            for left, right, sign in graph.signed_edges
        ]
    else:
        targets = [
            {"entityId": node_id, "target": int(target)}
            for node_id, target in zip(graph.node_ids, graph.targets[target_name], strict=True)
        ]
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.research-labels/1.0",
        "graphId": graph_id,
        "taskId": task_id,
        "targetName": target_name,
        "targets": targets,
        "excludedFromModelInputs": True,
    }
    payload["labelsHash"] = canonical_sha256(payload)
    return payload


def _write_graph_artifacts(
    *,
    staging: Path,
    graph_id: str,
    graph: ParsedGraph,
    bundle: dict[str, Any],
    labels: dict[str, Any],
    dataset_family: str,
    source_hashes: Mapping[str, str],
    split_protocol: str,
    excluded_fields: tuple[str, ...],
    extra_splits: Any = None,
) -> dict[str, Any]:
    if bundle["splitManifest"]["strategy"] != split_protocol:
        raise ValueError("research bundle and corpus split protocols differ")
    parser_id, parser_version = PARSER_CONTRACTS[dataset_family]
    from ...core.datasets import parsers as parser_module

    directory = staging / "graphs" / graph_id
    directory.mkdir(parents=True)
    bundle_path = directory / "bundle.json"
    labels_path = directory / "labels.json"
    _atomic_json(bundle_path, bundle)
    _atomic_json(labels_path, labels)
    if extra_splits is not None:
        _atomic_json(directory / "splits.json", extra_splits)
    from ...core.adapters import derive_training_selection

    validated_bundle = CoreGraphBundle.model_validate(bundle)
    selection = derive_training_selection(validated_bundle)
    entry = {
        "graphId": graph_id,
        "datasetFamily": dataset_family,
        "parserId": parser_id,
        "parserVersion": parser_version,
        "parserCodeSha256": file_sha256(Path(parser_module.__file__).resolve()),
        "bundlePath": f"graphs/{graph_id}/bundle.json",
        "bundleSha256": file_sha256(bundle_path),
        "graphVersionHash": bundle["graphVersionHash"],
        "labelsPath": f"graphs/{graph_id}/labels.json",
        "labelsSha256": file_sha256(labels_path),
        "labelsHash": labels["labelsHash"],
        "nodeCount": len(graph.node_ids),
        "edgeCount": len(graph.edges or graph.signed_edges),
        "directed": graph.directed,
        "sourceSha256": dict(sorted(source_hashes.items())),
        "splitProtocol": split_protocol,
        "splitHash": canonical_sha256(bundle["splitManifest"]),
        "visibleTopologyHash": selection.visible_topology_hash,
        "visibleTopologyEdgeCount": len(selection.visible_edge_indices),
        "excludedInputFields": list(excluded_fields),
    }
    if extra_splits is not None:
        entry["splitsPath"] = f"graphs/{graph_id}/splits.json"
        entry["splitsSha256"] = file_sha256(directory / "splits.json")
    return entry


def _raw_source_paths(root: Path) -> dict[str, Path]:
    return {
        name: (root / relative).resolve()
        for name, (_recipe, _source, relative) in RESEARCH_SOURCE_RECIPES.items()
    }


def _acquire_missing_sources(root: Path, *, open_url: Any | None = None) -> None:
    """Acquire absent fixed sources, publishing nothing until every SHA-256 matches."""

    from ...core.datasets.acquire import download_source

    paths = _raw_source_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if not missing:
        return
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".research-source-acquire.{uuid.uuid4().hex}.staging"
    try:
        downloaded: dict[str, Path] = {}
        for name in missing:
            recipe_id, source_id, _relative = RESEARCH_SOURCE_RECIPES[name]
            result = download_source(
                recipe_id=recipe_id,
                source_id=source_id,
                runtime_root=staging,
                open_url=open_url,
            )
            if result.observed_sha256 != EXPECTED_SOURCE_HASHES[name]:
                raise ValueError(f"SocialGraph-FM Research downloaded source hash mismatch: {name}")
            downloaded[name] = result.path
        for name in missing:
            target = paths[name]
            if not target.is_relative_to(root):
                raise ValueError("SocialGraph-FM Research raw source path escapes its root")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if file_sha256(target) != EXPECTED_SOURCE_HASHES[name]:
                    raise FileExistsError(f"conflicting SocialGraph-FM Research raw source exists: {name}")
                continue
            os.replace(downloaded[name], target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _verify_sources(root: Path) -> dict[str, str]:
    paths = _raw_source_paths(root)
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required SocialGraph-FM Research raw source is missing: {path}")
        observed[name] = file_sha256(path)
        if observed[name] != EXPECTED_SOURCE_HASHES[name]:
            raise ValueError(f"SocialGraph-FM Research raw source hash mismatch: {name}")
    return observed


def _bounded_gzip_extract(source: Path, destination: Path, *, max_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    written = 0
    try:
        with gzip.open(source, "rb") as input_stream, temporary.open("xb") as output_stream:
            while chunk := input_stream.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise ValueError("compressed SocialGraph-FM Research source exceeds extraction limit")
                output_stream.write(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_extracted(root: Path) -> None:
    twitch_target = root / "extracted/twitch-language/1.0.0/twitch"
    expected_twitch = TWITCH_ARCHIVE_MEMBERS
    final = twitch_target.parent
    staging = final.parent / f".extract-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(root / "raw/twitch-language/1.0.0/twitch.zip") as archive:
            names = {
                item.filename.replace("\\", "/") for item in archive.infolist() if not item.is_dir()
            }
            if names != expected_twitch:
                raise ValueError("Twitch archive member inventory is unexpected")
            if sum(item.file_size for item in archive.infolist()) > 256 * 1024 * 1024:
                raise ValueError("Twitch archive exceeds the extraction limit")
            for item in archive.infolist():
                normalized = item.filename.replace("\\", "/")
                if item.is_dir():
                    continue
                parts = Path(normalized).parts
                if Path(normalized).is_absolute() or ".." in parts:
                    raise ValueError("Twitch archive contains an unsafe path")
                mode = (item.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError("Twitch archive symlinks are forbidden")
                destination = (staging / normalized).resolve()
                if not destination.is_relative_to(staging.resolve()):
                    raise ValueError("Twitch archive path escapes staging")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as input_stream, destination.open("xb") as output:
                    shutil.copyfileobj(input_stream, output, 1024 * 1024)
        existing_names = (
            {path.relative_to(final).as_posix() for path in final.rglob("*") if path.is_file()}
            if final.is_dir()
            else set()
        )
        existing_matches = existing_names == expected_twitch and all(
            file_sha256(final / member) == file_sha256(staging / member)
            for member in expected_twitch
        )
        if existing_matches:
            shutil.rmtree(staging)
        else:
            backup = final.parent / f".extract-old-{uuid.uuid4().hex[:8]}"
            if final.exists():
                os.replace(final, backup)
            try:
                os.replace(staging, final)
            except BaseException:
                if backup.exists() and not final.exists():
                    os.replace(backup, final)
                raise
            finally:
                shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    compressed = (
        (
            root / "raw/wiki-rfa/1.0.0/wiki-RfA.txt.gz",
            root / "extracted/wiki-rfa/1.0.0/wiki-RfA.txt",
            256 * 1024 * 1024,
        ),
        (
            root / "raw/email-eu-core/1.0.0/email-Eu-core.txt.gz",
            root / "extracted/email-eu-core/1.0.0/email-Eu-core.txt",
            64 * 1024 * 1024,
        ),
        (
            root / "raw/email-eu-core/1.0.0/email-Eu-core-department-labels.txt.gz",
            root / "extracted/email-eu-core/1.0.0/email-Eu-core-department-labels.txt",
            8 * 1024 * 1024,
        ),
    )
    for source_path, destination, limit in compressed:
        _bounded_gzip_extract(source_path, destination, max_bytes=limit)


def _email_labels_payload(
    graph: ParsedGraph,
    split: EdgeSplit,
    *,
    seed: int,
) -> dict[str, Any]:
    positives = {"train": split.train, "validation": split.validation, "test": split.test}
    true_edges = set(graph.edges)
    adjacency: list[set[int]] = [set() for _ in graph.node_ids]
    # Hardness is derived only from train-visible topology. Validation/test
    # positives remain usable solely for all-true exclusion.
    for left, right in split.train:
        adjacency[left].add(right)
        adjacency[right].add(left)
    absent = [
        (left, right)
        for left in range(len(graph.node_ids))
        for right in range(left + 1, len(graph.node_ids))
        if (left, right) not in true_edges
    ]
    hard = [pair for pair in absent if adjacency[pair[0]] & adjacency[pair[1]]]
    hard_set = set(hard)
    uniform_non_hard = [pair for pair in absent if pair not in hard_set]
    rng = random.Random(seed)
    rng.shuffle(absent)
    rng.shuffle(hard)
    rng.shuffle(uniform_non_hard)
    used: set[tuple[int, int]] = set()
    partitions: dict[str, Any] = {}
    component_counts: dict[str, dict[str, int]] = {}
    for role in ("train", "validation", "test"):
        requested = min(len(positives[role]), len(absent) - len(used))
        desired_hard = requested // 2
        selected_hard = [pair for pair in hard if pair not in used][:desired_hard]
        used.update(selected_hard)
        desired_uniform = requested - len(selected_hard)
        selected_uniform = [pair for pair in uniform_non_hard if pair not in used][:desired_uniform]
        # Tiny fixtures can exhaust the non-hard pool. Any fallback remains
        # explicitly accounted for as hard, never mislabeled as uniform.
        hard_fallback = [pair for pair in hard if pair not in used and pair not in selected_hard][
            : desired_uniform - len(selected_uniform)
        ]
        selected_hard.extend(hard_fallback)
        used.update(selected_uniform)
        used.update(hard_fallback)
        selected = (*selected_hard, *selected_uniform)
        component_counts[role] = {
            "twoHopHard": len(selected_hard),
            "uniform": len(selected_uniform),
            "total": len(selected),
        }
        partitions[role] = {
            "positives": [
                {
                    "sourceId": graph.node_ids[left],
                    "targetId": graph.node_ids[right],
                    "target": 1,
                }
                for left, right in positives[role]
            ],
            "negatives": [
                {
                    "sourceId": graph.node_ids[left],
                    "targetId": graph.node_ids[right],
                    "target": 0,
                    "samplingComponent": (
                        "two-hop-hard" if index < len(selected_hard) else "uniform-non-two-hop"
                    ),
                }
                for index, (left, right) in enumerate(selected)
            ],
        }
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.research-labels/1.0",
        "graphId": "email-eu-core",
        "taskId": COLLABORATION_TASK,
        "targetName": "static-relation-existence",
        "targets": [],
        "excludedFromModelInputs": True,
        "offlineGroups": {
            "department": [
                {"nodeId": node_id, "group": group}
                for node_id, group in zip(
                    graph.node_ids,
                    graph.offline_labels.get("department", ()),
                    strict=True,
                )
            ]
        },
        "partitions": partitions,
        "negativeSampling": {
            "seed": seed,
            "excludeAllTrueEdges": True,
            "roleDisjoint": True,
            "strategy": "half-two-hop-hard-half-uniform-non-two-hop",
            "componentCounts": component_counts,
            "requestedCount": sum(len(items) for items in positives.values()),
            "sampledCount": len(used),
        },
    }
    payload["samplingHash"] = canonical_sha256(
        {"partitions": partitions, "negativeSampling": payload["negativeSampling"]}
    )
    payload["labelsHash"] = canonical_sha256(payload)
    return payload


def materialize_research_corpus(research_root: str | Path) -> Path:
    """Parse all nine real graphs and atomically publish a hash-bound corpus."""

    root = _safe_root(research_root)
    config = load_research_config()
    _acquire_missing_sources(root)
    source_hashes = _verify_sources(root)
    _ensure_extracted(root)
    target = root / "materialized" / "corpus"
    if target.exists():
        load_corpus_manifest(root)
        return target / "corpus-manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".corpus.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        entries: list[dict[str, Any]] = []
        twitch_root = root / "extracted/twitch-language/1.0.0/twitch"
        twitch_layout = {
            "DE": ("DE", "musae_DE.json", "musae_DE_edges.csv", "musae_DE_target.csv"),
            "EN": (
                "ENGB",
                "musae_ENGB_features.json",
                "musae_ENGB_edges.csv",
                "musae_ENGB_target.csv",
            ),
            "ES": ("ES", "musae_ES_features.json", "musae_ES_edges.csv", "musae_ES_target.csv"),
            "FR": ("FR", "musae_FR_features.json", "musae_FR_edges.csv", "musae_FR_target.csv"),
            "PT": (
                "PTBR",
                "musae_PTBR_features.json",
                "musae_PTBR_edges.csv",
                "musae_PTBR_target.csv",
            ),
            "RU": ("RU", "musae_RU_features.json", "musae_RU_edges.csv", "musae_RU_target.csv"),
        }
        from ...core.datasets.parsers import parse_musae_files

        for language, (directory, features, edges, targets) in twitch_layout.items():
            graph = parse_musae_files(
                graph_id=f"twitch-{language}",
                edges_path=twitch_root / directory / edges,
                features_path=twitch_root / directory / features,
                target_path=twitch_root / directory / targets,
            )
            split = _stratified_node_split(graph.targets["mature"], seed=RESEARCH_SEED)
            roles = {
                index: role
                for role, indices in (
                    ("train", split.train),
                    ("validation", split.validation),
                    ("test", split.test),
                )
                for index in indices
            }
            bundle = _bundle_payload(
                graph=graph,
                graph_id=graph.graph_id,
                source_name=f"SNAP Twitch {language}",
                source_uri="https://snap.stanford.edu/data/twitch-social-networks.html",
                source_sha256=source_hashes["twitch.zip"],
                citation="Rozemberczki et al., Multi-scale Attributed Node Embedding, 2019",
                split_strategy="stratified-node-70-15-15/1.0",
                node_roles=roles,
            )
            entries.append(
                _write_graph_artifacts(
                    staging=staging,
                    graph_id=graph.graph_id,
                    graph=graph,
                    bundle=bundle,
                    labels=_labels_payload(
                        graph_id=graph.graph_id,
                        task_id=CONTENT_POLICY_TASK,
                        graph=graph,
                        target_name="mature",
                    ),
                    dataset_family="twitch-language",
                    source_hashes={"twitch.zip": source_hashes["twitch.zip"]},
                    split_protocol="stratified-node-70-15-15/1.0",
                    excluded_fields=("mature",),
                )
            )

        tolokers = parse_tolokers_npz(root / "raw/tolokers/1.0.0/tolokers.npz")
        fold = tolokers.official_splits[0]
        tolokers_roles = {
            index: role
            for role, indices in (
                ("train", fold.train),
                ("validation", fold.validation),
                ("test", fold.test),
            )
            for index in indices
        }
        tolokers_bundle = _bundle_payload(
            graph=tolokers,
            graph_id="tolokers",
            source_name="Tolokers",
            source_uri="https://github.com/Toloka/TolokerGraph",
            source_sha256=source_hashes["tolokers.npz"],
            citation="Platonov et al., A Critical Look at the Evaluation of GNNs, 2023",
            split_strategy="official-10-splits/1.0",
            node_roles=tolokers_roles,
        )
        tolokers_splits = {
            "schemaVersion": "socialgraph-fm.research-official-splits/1.0",
            "graphId": "tolokers",
            "folds": [
                {
                    "fold": index,
                    "train": list(split.train),
                    "validation": list(split.validation),
                    "test": list(split.test),
                }
                for index, split in enumerate(tolokers.official_splits)
            ],
        }
        tolokers_splits["splitsHash"] = canonical_sha256(tolokers_splits)
        entries.append(
            _write_graph_artifacts(
                staging=staging,
                graph_id="tolokers",
                graph=tolokers,
                bundle=tolokers_bundle,
                labels=_labels_payload(
                    graph_id="tolokers",
                    task_id=ACCOUNT_RISK_TASK,
                    graph=tolokers,
                    target_name="banned",
                ),
                dataset_family="tolokers",
                source_hashes={"tolokers.npz": source_hashes["tolokers.npz"]},
                split_protocol="official-10-splits/1.0",
                excluded_fields=("banned",),
                extra_splits=tolokers_splits,
            )
        )

        wiki = parse_wiki_rfa(root / "extracted/wiki-rfa/1.0.0/wiki-RfA.txt")
        wiki_split = candidate_grouped_signed_split(wiki.signed_edges, seed=RESEARCH_SEED)
        wiki_roles = {
            (left, right): role
            for role, edges in (
                ("train", wiki_split.train),
                ("validation", wiki_split.validation),
                ("test", wiki_split.test),
            )
            for left, right, _ in edges
        }
        wiki_bundle = _bundle_payload(
            graph=wiki,
            graph_id="wiki-rfa",
            source_name="SNAP Wiki-RfA",
            source_uri="https://snap.stanford.edu/data/wiki-RfA.html",
            source_sha256=source_hashes["wiki-RfA.txt.gz"],
            citation="West et al., Exploiting Social Network Structure for Person-to-Person Sentiment Analysis, 2014",
            split_strategy="candidate-grouped-signed-70-15-15/1.0",
            edge_roles=wiki_roles,
        )
        entries.append(
            _write_graph_artifacts(
                staging=staging,
                graph_id="wiki-rfa",
                graph=wiki,
                bundle=wiki_bundle,
                labels=_labels_payload(
                    graph_id="wiki-rfa",
                    task_id=SIGNED_RELATION_TASK,
                    graph=wiki,
                    target_name="vote-sign",
                ),
                dataset_family="wiki-rfa",
                source_hashes={"wiki-RfA.txt.gz": source_hashes["wiki-RfA.txt.gz"]},
                split_protocol="candidate-grouped-signed-70-15-15/1.0",
                excluded_fields=("TXT", "DAT", "YEA", "RES", "VOT"),
            )
        )

        email = parse_email_files(
            root / "extracted/email-eu-core/1.0.0/email-Eu-core.txt",
            root / "extracted/email-eu-core/1.0.0/email-Eu-core-department-labels.txt",
        )
        email_split = spanning_forest_link_split(
            num_nodes=len(email.node_ids), edges=email.edges, seed=RESEARCH_SEED
        )
        email_roles = {
            edge: role
            for role, edges in (
                ("train", email_split.train),
                ("validation", email_split.validation),
                ("test", email_split.test),
            )
            for edge in edges
        }
        email_bundle = _bundle_payload(
            graph=email,
            graph_id="email-eu-core",
            source_name="SNAP Email-Eu-core",
            source_uri="https://snap.stanford.edu/data/email-Eu-core.html",
            source_sha256=canonical_sha256(
                {
                    "email-Eu-core.txt.gz": source_hashes["email-Eu-core.txt.gz"],
                    "email-Eu-core-department-labels.txt.gz": source_hashes[
                        "email-Eu-core-department-labels.txt.gz"
                    ],
                }
            ),
            citation="Leskovec et al., SNAP Datasets: Stanford Large Network Dataset Collection",
            split_strategy="spanning-forest-80-10-10/1.0",
            edge_roles=email_roles,
        )
        email_labels = _email_labels_payload(email, email_split, seed=RESEARCH_SEED)
        email_labels["offlineDepartmentHash"] = canonical_sha256(
            list(email.offline_labels["department"])
        )
        email_labels["labelsHash"] = canonical_sha256(
            {key: value for key, value in email_labels.items() if key != "labelsHash"}
        )
        entries.append(
            _write_graph_artifacts(
                staging=staging,
                graph_id="email-eu-core",
                graph=email,
                bundle=email_bundle,
                labels=email_labels,
                dataset_family="email-eu-core",
                source_hashes={
                    "email-Eu-core.txt.gz": source_hashes["email-Eu-core.txt.gz"],
                    "email-Eu-core-department-labels.txt.gz": source_hashes[
                        "email-Eu-core-department-labels.txt.gz"
                    ],
                },
                split_protocol="spanning-forest-80-10-10/1.0",
                excluded_fields=("department",),
            )
        )

        entries = sorted(entries, key=lambda item: item["graphId"])
        manifest: dict[str, Any] = {
            "schemaVersion": CORPUS_SCHEMA,
            "releaseId": RELEASE_ID,
            "corpusKind": "real",
            "testOnly": False,
            "materializerVersion": MATERIALIZER_VERSION,
            "researchConfigSha256": config["configSha256"],
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "formalReadinessUnaffected": True,
            "graphCount": len(entries),
            "nodeCount": sum(int(item["nodeCount"]) for item in entries),
            "edgeCount": sum(int(item["edgeCount"]) for item in entries),
            "graphs": entries,
            "lodoDomains": [
                "twitch-DE",
                "twitch-EN",
                "twitch-ES",
                "twitch-FR",
                "twitch-PT",
                "twitch-RU",
                "tolokers",
                "wiki-rfa",
                "email-eu-core",
            ],
        }
        manifest["corpusHash"] = canonical_sha256(manifest)
        _atomic_json(staging / "corpus-manifest.json", manifest)
        os.replace(staging, target)
        return target / "corpus-manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _expand_fixture_node_graph(graph: ParsedGraph, *, size: int = 18) -> ParsedGraph:
    """Expand only tiny trusted fixtures so every deterministic role is nonempty."""

    if len(graph.node_ids) >= size:
        return graph
    source_count = len(graph.node_ids)
    if source_count < 1:
        raise ValueError("fixture node graph cannot be empty")
    source_indices = tuple(index % source_count for index in range(size))
    node_ids = tuple(f"fixture-{index:03d}" for index in range(size))
    edges: set[tuple[int, int]] = {
        cast(tuple[int, int], tuple(sorted((index, (index + offset) % size))))
        for index in range(size)
        for offset in (1, 3)
        if index != (index + offset) % size
    }
    official_splits = (
        tuple(
            IndexSplit(
                train=tuple((index + fold) % size for index in range(12)),
                validation=tuple((index + fold) % size for index in range(12, 15)),
                test=tuple((index + fold) % size for index in range(15, size)),
            )
            for fold in range(10)
        )
        if graph.official_splits
        else ()
    )
    return ParsedGraph(
        graph_id=graph.graph_id,
        directed=graph.directed,
        node_ids=node_ids,
        edges=tuple(sorted(edges)),
        numeric_features={
            name: tuple(rows[index] for index in source_indices)
            for name, rows in graph.numeric_features.items()
        },
        categorical_features={
            name: tuple(values[index] for index in source_indices)
            for name, values in graph.categorical_features.items()
        },
        multi_hot_features={
            name: tuple(rows[index] for index in source_indices)
            for name, rows in graph.multi_hot_features.items()
        },
        targets={
            name: tuple(values[index] for index in source_indices)
            for name, values in graph.targets.items()
        },
        official_splits=official_splits,
        offline_labels={
            name: tuple(values[index] for index in source_indices)
            for name, values in graph.offline_labels.items()
        },
    )


def _expand_fixture_signed_graph(graph: ParsedGraph, *, candidate_count: int = 18) -> ParsedGraph:
    if len({target for _source, target, _sign in graph.signed_edges}) >= 6:
        return graph
    node_ids = tuple(
        sorted(
            [f"candidate-{index:03d}" for index in range(candidate_count)]
            + [f"voter-{index:03d}" for index in range(candidate_count)]
        )
    )
    by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    signed_edges = tuple(
        (
            by_id[f"voter-{index:03d}"],
            by_id[f"candidate-{index:03d}"],
            1 if index % 2 == 0 else -1,
        )
        for index in range(candidate_count)
    )
    return ParsedGraph(
        graph_id=graph.graph_id,
        directed=True,
        node_ids=node_ids,
        signed_edges=signed_edges,
    )


def _expand_fixture_email_graph(graph: ParsedGraph, *, size: int = 24) -> ParsedGraph:
    if len(graph.node_ids) >= 12:
        return graph
    node_ids = tuple(f"fixture-{index:03d}" for index in range(size))
    edges: set[tuple[int, int]] = {
        cast(tuple[int, int], tuple(sorted((index, (index + offset) % size))))
        for index in range(size)
        for offset in (1, 4)
    }
    return ParsedGraph(
        graph_id=graph.graph_id,
        directed=False,
        node_ids=node_ids,
        edges=tuple(sorted(edges)),
        offline_labels={"department": tuple(str(index % 4) for index in range(size))},
    )


def materialize_fixture_corpus(research_root: str | Path, fixture_root: str | Path) -> Path:
    """Materialize tiny trusted fixtures for CPU integration tests only."""

    root = _safe_root(research_root)
    config = load_research_config()
    fixtures = Path(fixture_root).resolve()
    target = root / "materialized" / "corpus"
    if target.exists():
        raise FileExistsError(f"research fixture corpus already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".fixture-corpus.{uuid.uuid4().hex}.staging"
    staging.mkdir()
    try:
        graphs: list[tuple[str, str, ParsedGraph, str, tuple[str, ...], Any]] = []
        for language, parsed in parse_twitch_fixture(fixtures / "twitch-language.json").items():
            graph = _expand_fixture_node_graph(parsed)
            graphs.append(
                (
                    f"twitch-{language}",
                    CONTENT_POLICY_TASK,
                    graph,
                    "twitch-language",
                    ("mature",),
                    _stratified_node_split(graph.targets["mature"], seed=RESEARCH_SEED),
                )
            )
        tolokers = _expand_fixture_node_graph(parse_tolokers_fixture(fixtures / "tolokers.json"))
        attributes = tolokers.numeric_features.get("attributes")
        if attributes is None or any(len(row) > 10 for row in attributes):
            raise ValueError("Tolokers fixture attributes cannot satisfy the 10-field contract")
        tolokers = replace(
            tolokers,
            numeric_features={
                **tolokers.numeric_features,
                "attributes": tuple(
                    (*row, *(0.0 for _ in range(10 - len(row)))) for row in attributes
                ),
            },
        )
        graphs.append(
            (
                "tolokers",
                ACCOUNT_RISK_TASK,
                tolokers,
                "tolokers",
                ("banned",),
                tolokers.official_splits[0],
            )
        )
        wiki = _expand_fixture_signed_graph(parse_wiki_rfa(fixtures / "wiki-rfa.txt"))
        graphs.append(
            (
                "wiki-rfa",
                SIGNED_RELATION_TASK,
                wiki,
                "wiki-rfa",
                ("TXT", "DAT", "YEA", "RES", "VOT"),
                candidate_grouped_signed_split(wiki.signed_edges, seed=RESEARCH_SEED),
            )
        )
        email, _fixture_email_split = parse_link_fixture(
            fixtures / "email-eu-core.json", seed=RESEARCH_SEED
        )
        email = _expand_fixture_email_graph(email)
        email_split = spanning_forest_link_split(
            num_nodes=len(email.node_ids), edges=email.edges, seed=RESEARCH_SEED
        )
        graphs.append(
            (
                "email-eu-core",
                COLLABORATION_TASK,
                email,
                "email-eu-core",
                ("department",),
                email_split,
            )
        )
        entries: list[dict[str, Any]] = []
        for graph_id, task_id, graph, family, excluded, split in graphs:
            if isinstance(split, IndexSplit):
                node_roles = {
                    index: role
                    for role, indices in (
                        ("train", split.train),
                        ("validation", split.validation),
                        ("test", split.test),
                    )
                    for index in indices
                }
                edge_roles = None
            else:
                if isinstance(split, SignedEdgeSplit):
                    split_groups = (
                        ("train", tuple((a, b) for a, b, _ in split.train)),
                        ("validation", tuple((a, b) for a, b, _ in split.validation)),
                        ("test", tuple((a, b) for a, b, _ in split.test)),
                    )
                else:
                    split_groups = (
                        ("train", split.train),
                        ("validation", split.validation),
                        ("test", split.test),
                    )
                edge_roles = {
                    edge: role for role, split_edges in split_groups for edge in split_edges
                }
                node_roles = None
            source_hash = hashlib.sha256(graph_id.encode()).hexdigest()
            split_strategy = (
                "candidate-grouped-signed-70-15-15/1.0"
                if isinstance(split, SignedEdgeSplit)
                else "spanning-forest-80-10-10/1.0"
                if isinstance(split, EdgeSplit)
                else "official-10-splits/1.0"
                if graph_id == "tolokers"
                else "stratified-node-70-15-15/1.0"
            )
            bundle = _bundle_payload(
                graph=graph,
                graph_id=graph_id,
                source_name=f"Research fixture {graph_id}",
                source_uri="urn:socialgraph-fm:research-fixture",
                source_sha256=source_hash,
                citation="Test fixture only",
                split_strategy=split_strategy,
                edge_roles=edge_roles,
                node_roles=node_roles,
            )
            target_name = (
                "mature"
                if task_id == CONTENT_POLICY_TASK
                else "banned"
                if task_id == ACCOUNT_RISK_TASK
                else "vote-sign"
                if task_id == SIGNED_RELATION_TASK
                else None
            )
            labels = (
                _email_labels_payload(graph, split, seed=RESEARCH_SEED)
                if task_id == COLLABORATION_TASK
                else _labels_payload(
                    graph_id=graph_id,
                    task_id=task_id,
                    graph=graph,
                    target_name=target_name,
                )
            )
            extra_splits = None
            if graph_id == "tolokers":
                extra_splits = {
                    "schemaVersion": "socialgraph-fm.research-official-splits/1.0",
                    "graphId": "tolokers",
                    "folds": [
                        {
                            "fold": index,
                            "train": list(fold.train),
                            "validation": list(fold.validation),
                            "test": list(fold.test),
                        }
                        for index, fold in enumerate(graph.official_splits)
                    ],
                }
                extra_splits["splitsHash"] = canonical_sha256(extra_splits)
            entries.append(
                _write_graph_artifacts(
                    staging=staging,
                    graph_id=graph_id,
                    graph=graph,
                    bundle=bundle,
                    labels=labels,
                    dataset_family=family,
                    source_hashes={"fixture": source_hash},
                    split_protocol=split_strategy,
                    excluded_fields=excluded,
                    extra_splits=extra_splits,
                )
            )
        entries.sort(key=lambda item: item["graphId"])
        manifest = {
            "schemaVersion": CORPUS_SCHEMA,
            "releaseId": RELEASE_ID,
            "corpusKind": "test-fixture",
            "testOnly": True,
            "materializerVersion": MATERIALIZER_VERSION,
            "researchConfigSha256": config["configSha256"],
            "seed": RESEARCH_SEED,
            "preliminary": True,
            "formalReadinessUnaffected": True,
            "graphCount": len(entries),
            "nodeCount": sum(int(item["nodeCount"]) for item in entries),
            "edgeCount": sum(int(item["edgeCount"]) for item in entries),
            "graphs": entries,
            "lodoDomains": [item["graphId"] for item in entries],
        }
        manifest["corpusHash"] = canonical_sha256(manifest)
        _atomic_json(staging / "corpus-manifest.json", manifest)
        os.replace(staging, target)
        return target / "corpus-manifest.json"
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_tolokers_split_payload(
    payload: Mapping[str, Any], *, node_count: int, bundle: CoreGraphBundle | None = None
) -> tuple[dict[str, Any], ...]:
    if (
        payload.get("schemaVersion") != "socialgraph-fm.research-official-splits/1.0"
        or payload.get("graphId") != "tolokers"
        or payload.get("splitsHash")
        != canonical_sha256({key: value for key, value in payload.items() if key != "splitsHash"})
    ):
        raise ValueError("Tolokers official split artifact identity mismatch")
    rows = payload.get("folds")
    if not isinstance(rows, list) or len(rows) != 10 or any(
        not isinstance(item, dict)
        or type(item.get("fold")) is not int
        or item["fold"] != index
        for index, item in enumerate(rows)
    ):
        raise ValueError("Tolokers requires all ten ordered official splits")
    node_inventory = set(range(node_count))
    for item in rows:
        partitions: dict[str, set[int]] = {}
        for role in ("train", "validation", "test"):
            values = item.get(role)
            if not isinstance(values, list) or any(type(value) is not int for value in values):
                raise ValueError("Tolokers official split indices must be integer lists")
            if len(values) != len(set(values)) or any(value not in node_inventory for value in values):
                raise ValueError("Tolokers official split contains duplicate or out-of-range indices")
            partitions[role] = set(values)
        if (
            partitions["train"] & partitions["validation"]
            or partitions["train"] & partitions["test"]
            or partitions["validation"] & partitions["test"]
            or set().union(*partitions.values()) != node_inventory
        ):
            raise ValueError("Tolokers official split partitions must be disjoint and exhaustive")
    if bundle is not None:
        roles = {
            assignment.entity_id: assignment.role
            for assignment in bundle.split_manifest.assignments
        }
        expected = {
            role: {
                node.index for node in bundle.nodes if roles.get(node.id) == role
            }
            for role in ("train", "validation", "test")
        }
        if any(set(rows[0][role]) != expected[role] for role in expected):
            raise ValueError("Tolokers split 0 differs from the bundle split manifest")
    return tuple(rows)


def _validate_research_label_contract(
    domain: str,
    bundle: CoreGraphBundle,
    labels: Mapping[str, Any],
) -> None:
    if (
        labels.get("schemaVersion") != "socialgraph-fm.research-labels/1.0"
        or labels.get("graphId") != domain
        or labels.get("taskId") != _domain_task_id(domain)
        or labels.get("excludedFromModelInputs") is not True
    ):
        raise ValueError("research labels task/graph identity mismatch")
    assignments = {
        assignment.entity_id: assignment.role for assignment in bundle.split_manifest.assignments
    }
    node_ids = {node.id for node in bundle.nodes}
    targets = labels.get("targets")
    if not isinstance(targets, list):
        # This is malformed artifact content, not a caller API type mismatch.
        raise ValueError("research labels target inventory is invalid")  # noqa: TRY004
    if domain.startswith("twitch-") or domain == "tolokers":
        target_by_id: dict[str, int] = {}
        for item in targets:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("entityId"), str)
                or type(item.get("target")) is not int
                or item["target"] not in {0, 1}
                or item["entityId"] in target_by_id
            ):
                raise ValueError("research binary node labels are invalid")
            target_by_id[item["entityId"]] = item["target"]
        if set(target_by_id) != node_ids or set(assignments) != node_ids:
            raise ValueError("research node labels/split inventory does not cover the graph")
        return
    edge_ids = {
        f"edge:{edge.source_id}:{edge.target_id}": (edge.source_id, edge.target_id)
        for edge in bundle.edges
    }
    if domain == "wiki-rfa":
        signed_target_by_id: dict[str, tuple[str, str, int]] = {}
        for item in targets:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("entityId"), str)
                or not isinstance(item.get("sourceId"), str)
                or not isinstance(item.get("targetId"), str)
                or type(item.get("target")) is not int
                or item["target"] not in {0, 1}
                or item["entityId"] in signed_target_by_id
            ):
                raise ValueError("Wiki-RfA signed relation labels are invalid")
            signed_target_by_id[item["entityId"]] = (
                item["sourceId"],
                item["targetId"],
                item["target"],
            )
        if set(signed_target_by_id) != set(edge_ids) or set(assignments) != set(edge_ids):
            raise ValueError("Wiki-RfA labels/split inventory does not cover every edge")
        if any(
            signed_target_by_id[edge_id][:2] != endpoints
            for edge_id, endpoints in edge_ids.items()
        ):
            raise ValueError("Wiki-RfA label endpoints differ from the directed graph")
        return
    if domain != "email-eu-core" or targets:
        raise ValueError("unsupported SocialGraph-FM Research label contract")
    by_id = {node.id: node.index for node in bundle.nodes}
    true_pairs = {
        tuple(sorted((by_id[edge.source_id], by_id[edge.target_id]))) for edge in bundle.edges
    }
    true_by_role: dict[str, set[tuple[int, int]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for edge_id, endpoints in edge_ids.items():
        role = assignments.get(edge_id)
        if role not in true_by_role:
            raise ValueError("Email edge split role is invalid")
        true_by_role[role].add(
            cast(
                tuple[int, int],
                tuple(sorted((by_id[endpoints[0]], by_id[endpoints[1]]))),
            )
        )
    partitions = labels.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(true_by_role):
        raise ValueError("Email label partitions are incomplete")
    train_adjacency: list[set[int]] = [set() for _ in bundle.nodes]
    for left, right in true_by_role["train"]:
        train_adjacency[left].add(right)
        train_adjacency[right].add(left)
    all_negatives: set[tuple[int, int]] = set()
    observed_component_counts: dict[str, dict[str, int]] = {}
    for partition_role, expected_positives in true_by_role.items():
        partition = partitions.get(partition_role)
        if not isinstance(partition, dict):
            raise ValueError("Email label partition is invalid")  # noqa: TRY004
        observed_by_kind: dict[str, set[tuple[int, int]]] = {}
        for kind, target in (("positives", 1), ("negatives", 0)):
            rows = partition.get(kind)
            if not isinstance(rows, list):
                raise ValueError("Email label rows are invalid")  # noqa: TRY004
            observed: set[tuple[int, int]] = set()
            for item in rows:
                if (
                    not isinstance(item, dict)
                    or item.get("sourceId") not in by_id
                    or item.get("targetId") not in by_id
                    or item["sourceId"] == item["targetId"]
                    or type(item.get("target")) is not int
                    or item["target"] != target
                ):
                    raise ValueError("Email pair label is invalid")
                pair = cast(
                    tuple[int, int],
                    tuple(sorted((by_id[item["sourceId"]], by_id[item["targetId"]]))),
                )
                if pair in observed:
                    raise ValueError("Email pair labels contain duplicates")
                observed.add(pair)
            observed_by_kind[kind] = observed
        if observed_by_kind["positives"] != expected_positives:
            raise ValueError("Email positive labels differ from the bundle split")
        negatives = observed_by_kind["negatives"]
        if len(negatives) != len(expected_positives):
            raise ValueError("Email positive/negative role counts must match")
        if negatives & true_pairs or negatives & all_negatives:
            raise ValueError("Email negative labels leak true edges or overlap roles")
        component_counts = {"twoHopHard": 0, "uniform": 0, "total": len(negatives)}
        for item in partition["negatives"]:
            pair = cast(
                tuple[int, int],
                tuple(sorted((by_id[item["sourceId"]], by_id[item["targetId"]]))),
            )
            has_two_hop_witness = bool(train_adjacency[pair[0]] & train_adjacency[pair[1]])
            expected_component = (
                "two-hop-hard" if has_two_hop_witness else "uniform-non-two-hop"
            )
            if item.get("samplingComponent") != expected_component:
                raise ValueError("Email negative sampling component is inconsistent")
            component_counts["twoHopHard" if has_two_hop_witness else "uniform"] += 1
        observed_component_counts[partition_role] = component_counts
        all_negatives.update(negatives)
    sampling = labels.get("negativeSampling")
    requested_count = sum(len(items) for items in true_by_role.values())
    if (
        not isinstance(sampling, dict)
        or sampling.get("seed") != RESEARCH_SEED
        or sampling.get("excludeAllTrueEdges") is not True
        or sampling.get("roleDisjoint") is not True
        or sampling.get("strategy") != "half-two-hop-hard-half-uniform-non-two-hop"
        or sampling.get("componentCounts") != observed_component_counts
        or sampling.get("requestedCount") != requested_count
        or sampling.get("sampledCount") != len(all_negatives)
        or labels.get("samplingHash")
        != canonical_sha256({"partitions": partitions, "negativeSampling": sampling})
    ):
        raise ValueError("Email negative sampling contract/hash mismatch")
    groups = labels.get("offlineGroups", {}).get("department")
    if (
        not isinstance(groups, list)
        or len(groups) != len(bundle.nodes)
        or {item.get("nodeId") for item in groups if isinstance(item, dict)} != node_ids
    ):
        raise ValueError("Email offline department inventory is incomplete")


def _validate_research_input_contract(
    domain: str,
    bundle: CoreGraphBundle,
    entry: Mapping[str, Any],
) -> None:
    family = (
        "twitch-language"
        if domain.startswith("twitch-")
        else domain
    )
    expected_features: dict[str, tuple[tuple[str, str], ...]] = {
        "twitch-language": (("sharedAttributes", "multiHot"),),
        "tolokers": tuple((f"attributes:{index}", "numeric") for index in range(10)),
        "wiki-rfa": (),
        "email-eu-core": (),
    }
    expected_excluded = {
        "twitch-language": ("mature",),
        "tolokers": ("banned",),
        "wiki-rfa": ("TXT", "DAT", "YEA", "RES", "VOT"),
        "email-eu-core": ("department",),
    }
    observed_features = tuple((feature.name, feature.kind) for feature in bundle.node_features)
    if (
        entry.get("datasetFamily") != family
        or observed_features != expected_features[family]
        or tuple(entry.get("excludedInputFields", ())) != expected_excluded[family]
    ):
        raise ValueError("research model input/excluded-field contract mismatch")
    if domain == "wiki-rfa" and any(edge.weight != 1.0 for edge in bundle.edges):
        raise ValueError("Wiki-RfA vote labels must not enter edge weights")


def load_corpus_manifest(research_root: str | Path) -> dict[str, Any]:
    root = _safe_root(research_root)
    manifest = _read_hashed_document(
        root / "materialized/corpus/corpus-manifest.json",
        schema=CORPUS_SCHEMA,
        hash_field="corpusHash",
    )
    if manifest.get("graphCount") != 9 or len(manifest.get("graphs", ())) != 9:
        raise ValueError("SocialGraph-FM Research corpus requires exactly nine graph domains")
    expected_domains = {
        "twitch-DE",
        "twitch-EN",
        "twitch-ES",
        "twitch-FR",
        "twitch-PT",
        "twitch-RU",
        "tolokers",
        "wiki-rfa",
        "email-eu-core",
    }
    graph_ids = [entry.get("graphId") for entry in manifest["graphs"]]
    if len(graph_ids) != len(set(graph_ids)) or set(graph_ids) != expected_domains:
        raise ValueError("SocialGraph-FM Research corpus graph inventory mismatch")
    if len(manifest.get("lodoDomains", ())) != 9 or set(manifest["lodoDomains"]) != expected_domains:
        raise ValueError("SocialGraph-FM Research leave-one-domain-out inventory mismatch")
    if (manifest.get("corpusKind"), manifest.get("testOnly")) not in {
        ("real", False),
        ("test-fixture", True),
    }:
        raise ValueError("SocialGraph-FM Research corpus kind/test-only identity is invalid")
    if manifest.get("materializerVersion") != MATERIALIZER_VERSION:
        raise ValueError("SocialGraph-FM Research corpus materializer version mismatch")
    if manifest.get("researchConfigSha256") != load_research_config()["configSha256"]:
        raise ValueError("SocialGraph-FM Research corpus configuration identity mismatch")
    corpus_root = root / "materialized/corpus"
    from ...core.adapters import derive_training_selection
    from ...core.datasets import parsers as parser_module

    parser_code_sha = file_sha256(Path(parser_module.__file__).resolve())
    if manifest["corpusKind"] == "real":
        source_inventory: dict[str, str] = {}
        for entry in manifest["graphs"]:
            for name, digest in entry.get("sourceSha256", {}).items():
                prior = source_inventory.setdefault(name, digest)
                if prior != digest:
                    raise ValueError("real research corpus source hash inventory conflicts")
        if source_inventory != EXPECTED_SOURCE_HASHES:
            raise ValueError("real research corpus lacks the pinned source hash inventory")
    elif any(set(entry.get("sourceSha256", {})) != {"fixture"} for entry in manifest["graphs"]):
        raise ValueError("test fixture corpus source identity is invalid")
    observed_node_count = 0
    observed_edge_count = 0
    for entry in manifest["graphs"]:
        expected_parser = PARSER_CONTRACTS.get(entry.get("datasetFamily"))
        if (
            expected_parser is None
            or (entry.get("parserId"), entry.get("parserVersion")) != expected_parser
            or entry.get("parserCodeSha256") != parser_code_sha
        ):
            raise ValueError("research corpus parser identity mismatch")
        bundle_path = (corpus_root / entry["bundlePath"]).resolve()
        labels_path = (corpus_root / entry["labelsPath"]).resolve()
        if not bundle_path.is_relative_to(corpus_root.resolve()) or not labels_path.is_relative_to(
            corpus_root.resolve()
        ):
            raise ValueError("research corpus artifact path escapes its root")
        if file_sha256(bundle_path) != entry["bundleSha256"]:
            raise ValueError("research bundle hash mismatch")
        if file_sha256(labels_path) != entry["labelsSha256"]:
            raise ValueError("research labels hash mismatch")
        bundle = CoreGraphBundle.model_validate_json(bundle_path.read_bytes())
        if bundle.graph_version_hash != entry["graphVersionHash"]:
            raise ValueError("research graph identity mismatch")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        labels_hash = labels.get("labelsHash")
        if (
            labels_hash
            != canonical_sha256({key: value for key, value in labels.items() if key != "labelsHash"})
            or labels_hash != entry.get("labelsHash")
        ):
            raise ValueError("research labels content/entry hash mismatch")
        split_hash = canonical_sha256(
            bundle.split_manifest.model_dump(mode="json", by_alias=True)
        )
        if (
            split_hash != entry.get("splitHash")
            or bundle.split_manifest.strategy != entry.get("splitProtocol")
        ):
            raise ValueError("research split manifest/entry identity mismatch")
        if (
            type(entry.get("nodeCount")) is not int
            or entry["nodeCount"] != len(bundle.nodes)
            or type(entry.get("edgeCount")) is not int
            or entry["edgeCount"] != len(bundle.edges)
            or type(entry.get("directed")) is not bool
            or entry["directed"] is not bundle.directed
        ):
            raise ValueError("research graph count/direction entry mismatch")
        observed_node_count += len(bundle.nodes)
        observed_edge_count += len(bundle.edges)
        _validate_research_input_contract(str(entry["graphId"]), bundle, entry)
        _validate_research_label_contract(str(entry["graphId"]), bundle, labels)
        if entry["graphId"] == "tolokers":
            splits_path = (corpus_root / entry.get("splitsPath", "")).resolve()
            if (
                not splits_path.is_relative_to(corpus_root.resolve())
                or not splits_path.is_file()
                or file_sha256(splits_path) != entry.get("splitsSha256")
            ):
                raise ValueError("Tolokers official split artifact hash mismatch")
            _validate_tolokers_split_payload(
                json.loads(splits_path.read_text(encoding="utf-8")),
                node_count=len(bundle.nodes),
                bundle=bundle,
            )
        elif "splitsPath" in entry or "splitsSha256" in entry:
            raise ValueError("only Tolokers may carry an extra official split artifact")
        selection = derive_training_selection(bundle)
        if (
            entry.get("visibleTopologyEdgeCount") != len(selection.visible_edge_indices)
            or entry.get("visibleTopologyHash") != selection.visible_topology_hash
        ):
            raise ValueError("research visible topology contract mismatch")
        expected_structure = compute_structure_rows(
            bundle,
            visible_edge_indices=selection.visible_edge_indices,
            config=StructureAlgorithmConfig.fixed(),
        )
        if (
            bundle.structural_features is None
            or bundle.structural_features.names != STRUCTURE_FEATURE_NAMES
            or len(bundle.structural_features.values) != len(expected_structure)
            or any(
                not math.isclose(observed, float(expected), rel_tol=0.0, abs_tol=1e-12)
                for observed_row, expected_row in zip(
                    bundle.structural_features.values, expected_structure, strict=True
                )
                for observed, expected in zip(observed_row, expected_row, strict=True)
            )
        ):
            raise ValueError("research structural features differ from visible topology")
        source_uri = bundle.source.source_uri
        if manifest["testOnly"] is True and source_uri != "urn:socialgraph-fm:research-fixture":
            raise ValueError("test fixture corpus provenance is not marked as synthetic")
        if manifest["testOnly"] is False and (source_uri or "").startswith("urn:"):
            raise ValueError("real research corpus uses fixture provenance")
    if (
        manifest.get("nodeCount") != observed_node_count
        or manifest.get("edgeCount") != observed_edge_count
    ):
        raise ValueError("research corpus aggregate counts differ from graph artifacts")
    return manifest


def _require_publishable_corpus(
    corpus: Mapping[str, Any], *, allow_test_fixture: bool, stage: str
) -> None:
    if corpus.get("corpusKind") == "real" and corpus.get("testOnly") is False:
        return
    if allow_test_fixture and (
        corpus.get("corpusKind"), corpus.get("testOnly")
    ) == ("test-fixture", True):
        return
    raise ValueError(
        f"SocialGraph-FM Research {stage} refuses test-fixture corpus; only real materialization is publishable"
    )

COMPAT_EXPORTS = (
    '_stratified_node_split',
    'candidate_grouped_signed_split',
    '_edge_id',
    '_bundle_payload',
    '_labels_payload',
    '_write_graph_artifacts',
    '_raw_source_paths',
    '_acquire_missing_sources',
    '_verify_sources',
    '_bounded_gzip_extract',
    '_ensure_extracted',
    '_email_labels_payload',
    'materialize_research_corpus',
    '_expand_fixture_node_graph',
    '_expand_fixture_signed_graph',
    '_expand_fixture_email_graph',
    'materialize_fixture_corpus',
    '_validate_tolokers_split_payload',
    '_validate_research_label_contract',
    '_validate_research_input_contract',
    'load_corpus_manifest',
    '_require_publishable_corpus',
)

__all__ = [
    'candidate_grouped_signed_split',
    'load_corpus_manifest',
    'materialize_fixture_corpus',
    'materialize_research_corpus',
    'research_root_from_home',
]
