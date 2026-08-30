"""Deterministic graph views, hashes, and dataset contract construction."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

import numpy as np
from pydantic import ValidationError

from ..dataset_schemas import (
    ArrayDescriptor,
    ArtifactEdge,
    ArtifactGraphSummary,
    ArtifactGraphView,
    ArtifactNode,
    DataGovernancePolicy,
    FeatureRecipe,
    FeatureSchema,
    GraphSemantics,
    GraphVariant,
    LabelSchema,
    LicenseEvidence,
    LicensePolicy,
    NodeIdentitySchema,
    SourceFileDigest,
    SplitFoldCounts,
    SplitSet,
    TaskSpec,
    TrainingDatasetRef,
)

from .array_validation import _read_npz, _semantic_edge_index
from .models import (
    PREVIEW_EDGES,
    PREVIEW_NODES,
    _SPLIT_KEYS,
    ArrayRole,
    GraphPayload,
    UploadedEntry,
)

_LEGACY_PLANETOID_PATTERN = re.compile(
    r"(?:^|/)ind\.[^/]+\.(?:x|y|tx|ty|allx|ally|graph)$", re.IGNORECASE
)

def _checksum(entries: list[UploadedEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.name.casefold()):
        digest.update(entry.name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(entry.data)
        digest.update(b"\x00")
    return digest.hexdigest()


def _file_role(name: str) -> str:
    lowered = name.casefold()
    if lowered.endswith((".pt", ".pth")):
        return "unsafe_torch_archive"
    if _LEGACY_PLANETOID_PATTERN.search(lowered):
        return "unsafe_legacy_pickle"
    if lowered.endswith("out1_graph_edges.txt"):
        return "edges"
    if lowered.endswith("out1_node_feature_label.txt"):
        return "nodes_features_labels"
    if lowered.endswith(".sgfm-graph.json"):
        return "graph_version_handoff"
    if lowered.endswith(".npz"):
        return "safe_numeric_archive"
    if lowered.endswith(".json"):
        return "manifest"
    return "auxiliary"


def _connected_components(node_count: int, edge_index: np.ndarray) -> int:
    if node_count == 0:
        return 0
    parent = np.arange(node_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for source, target in edge_index.T:
        left = find(int(source))
        right = find(int(target))
        if left != right:
            parent[right] = left
    return len({find(index) for index in range(node_count)})


def _build_view(payload: GraphPayload, artifact_id: str) -> ArtifactGraphView:
    # A numeric first-N/first-E slice can make connected nodes look isolated.
    # Select nodes through real edges and reserve witness edges before filling
    # the remaining edge budget. The result is deterministic and every chosen
    # non-isolated node keeps at least one visible incident edge.
    semantic_edges = (
        np.asarray(payload.edge_index, dtype=np.int64)
        if payload.edge_ids is not None
        else _semantic_edge_index(
            payload.edge_index,
            payload.node_count,
            directed=payload.directed,
        )
    )
    selected: set[int] = set()
    if payload.node_count <= PREVIEW_NODES:
        selected.update(range(payload.node_count))
    else:
        for raw_source, raw_target in semantic_edges.T:
            source, target = int(raw_source), int(raw_target)
            missing = {source, target}.difference(selected)
            if len(selected) + len(missing) > PREVIEW_NODES:
                continue
            selected.add(source)
            selected.add(target)
            if len(selected) >= PREVIEW_NODES:
                break
        if len(selected) < PREVIEW_NODES:
            incident = np.zeros(payload.node_count, dtype=np.bool_)
            if semantic_edges.size:
                incident[semantic_edges[0]] = True
                incident[semantic_edges[1]] = True
            selected.update(
                index
                for index in range(payload.node_count)
                if index not in selected
                and not bool(incident[index])
                and len(selected) < PREVIEW_NODES
            )

    visible_ids = sorted(selected)
    node_ids = (
        [str(value) for value in payload.node_ids.tolist()]
        if payload.node_ids is not None
        else [str(index) for index in range(payload.node_count)]
    )
    node_labels = (
        [str(value) for value in payload.node_labels.tolist()]
        if payload.node_labels is not None
        else node_ids
    )
    node_types = (
        [str(value) or None for value in payload.node_types.tolist()]
        if payload.node_types is not None
        else [None] * payload.node_count
    )
    node_attributes = (
        [json.loads(str(value)) for value in payload.node_attributes.tolist()]
        if payload.node_attributes is not None
        else [{} for _index in range(payload.node_count)]
    )
    nodes = [
        ArtifactNode(
            id=node_ids[index],
            label=node_labels[index],
            nodeType=node_types[index],
            attributes=node_attributes[index],
        )
        for index in visible_ids
    ]
    chosen: list[tuple[int, int, int]] = []
    chosen_numbers: set[int] = set()
    covered: set[int] = set()
    for edge_number, (raw_source, raw_target) in enumerate(semantic_edges.T):
        source, target = int(raw_source), int(raw_target)
        if source not in selected or target not in selected:
            continue
        if source in covered and target in covered:
            continue
        chosen.append((edge_number, source, target))
        chosen_numbers.add(edge_number)
        covered.add(source)
        covered.add(target)
        if len(chosen) >= PREVIEW_EDGES:
            break
    if len(chosen) < PREVIEW_EDGES:
        for edge_number, (raw_source, raw_target) in enumerate(semantic_edges.T):
            if edge_number in chosen_numbers:
                continue
            source, target = int(raw_source), int(raw_target)
            if source not in selected or target not in selected:
                continue
            chosen.append((edge_number, source, target))
            if len(chosen) >= PREVIEW_EDGES:
                break

    edge_ids = (
        [str(value) for value in payload.edge_ids.tolist()]
        if payload.edge_ids is not None
        else [f"e{index}" for index in range(int(semantic_edges.shape[1]))]
    )
    edge_types = (
        [str(value) or None for value in payload.edge_types.tolist()]
        if payload.edge_types is not None
        else [None] * len(edge_ids)
    )
    edge_weights = (
        [None if np.isnan(float(value)) else float(value) for value in payload.edge_weights.tolist()]
        if payload.edge_weights is not None
        else [None] * len(edge_ids)
    )
    edge_timestamps = (
        [str(value) or None for value in payload.edge_timestamps.tolist()]
        if payload.edge_timestamps is not None
        else [None] * len(edge_ids)
    )
    edge_directed = (
        [None if int(value) < 0 else bool(value) for value in payload.edge_directed.tolist()]
        if payload.edge_directed is not None
        else [payload.directed] * len(edge_ids)
    )
    edge_attributes = (
        [json.loads(str(value)) for value in payload.edge_attributes.tolist()]
        if payload.edge_attributes is not None
        else [{} for _index in edge_ids]
    )
    visible_edges = [
        ArtifactEdge(
            id=edge_ids[edge_number],
            source=node_ids[source],
            target=node_ids[target],
            edgeType=edge_types[edge_number],
            weight=edge_weights[edge_number],
            timestamp=edge_timestamps[edge_number],
            directed=edge_directed[edge_number],
            attributes=edge_attributes[edge_number],
        )
        for edge_number, source, target in chosen
    ]
    edge_count = int(semantic_edges.shape[1])
    non_loop = semantic_edges[:, semantic_edges[0] != semantic_edges[1]]
    if non_loop.size:
        relation_keys = np.unique(
            non_loop[0] * np.int64(payload.node_count) + non_loop[1]
        )
        density_edge_count = int(relation_keys.size)
    else:
        density_edge_count = 0
    possible = payload.node_count * max(0, payload.node_count - 1)
    if not payload.directed:
        possible //= 2
    density = density_edge_count / possible if possible else 0.0
    return ArtifactGraphView(
        id=f"view-{artifact_id}",
        nodes=nodes,
        edges=visible_edges,
        summary=ArtifactGraphSummary(
            nodeCount=payload.node_count,
            edgeCount=edge_count,
            density=round(density, 8),
            connectedComponents=_connected_components(payload.node_count, semantic_edges),
            visibleNodeCount=len(nodes),
            visibleEdgeCount=len(visible_edges),
            partialPreview=len(nodes) < payload.node_count
            or len(visible_edges) < edge_count,
        ),
    )


def _canonical_graph_hash(payload: GraphPayload) -> str:
    digest = hashlib.sha256()
    digest.update(str(payload.node_count).encode("ascii"))
    digest.update(b"\x01" if payload.directed else b"\x00")
    edges = (
        np.asarray(payload.edge_index, dtype=np.int64)
        if payload.edge_ids is not None
        else _semantic_edge_index(
            payload.edge_index,
            payload.node_count,
            directed=payload.directed,
        )
    )
    if edges.size:
        order = np.lexsort((edges[1], edges[0]))
        digest.update(np.ascontiguousarray(edges[:, order]).tobytes())
    return digest.hexdigest()


def _payload_arrays(payload: GraphPayload) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "edge_index": np.asarray(payload.edge_index, dtype=np.int64),
        "num_nodes": np.asarray(payload.node_count, dtype=np.int64),
        "directed": np.asarray(payload.directed, dtype=np.bool_),
    }
    arrays["node_id_map"] = (
        np.asarray(payload.node_ids, dtype=np.str_)
        if payload.node_ids is not None
        else np.asarray([str(index) for index in range(payload.node_count)], dtype=np.str_)
    )
    optional_arrays = {
        "node_label": payload.node_labels,
        "node_type": payload.node_types,
        "node_attributes_json": payload.node_attributes,
        "edge_id_map": payload.edge_ids,
        "edge_type": payload.edge_types,
        "edge_weight": payload.edge_weights,
        "edge_timestamp": payload.edge_timestamps,
        "edge_directed": payload.edge_directed,
        "edge_attributes_json": payload.edge_attributes,
    }
    arrays.update(
        {
            name: np.asarray(value)
            for name, value in optional_arrays.items()
            if value is not None
        }
    )
    if payload.features is not None:
        arrays["x"] = np.asarray(payload.features)
    if payload.labels is not None:
        arrays["y"] = np.asarray(payload.labels)
    arrays.update({name: np.asarray(value) for name, value in payload.splits.items()})
    arrays.update(
        {name: np.asarray(value) for name, value in payload.variant_arrays.items()}
    )
    return arrays


def _default_transform_recipe(payload: GraphPayload) -> dict[str, object]:
    return {
        "id": "identity-v1",
        "graphVariant": "raw",
        "directed": payload.directed,
        "selfLoopPolicy": "preserve",
        "duplicatePolicy": "preserve",
        "featureTransform": "identity",
    }


def _content_hash(payload: GraphPayload, recipes: object) -> str:
    """Hash every training-relevant array and its transform recipe."""

    digest = hashlib.sha256(b"socialgraph-fm-dataset-artifact-v2\x00")
    structural_semantics = {
        "directed": payload.directed,
        "duplicateEdgePolicy": "preserve",
        "edgeStorage": "coo",
        "selfLoopPolicy": "preserve",
    }
    digest.update(
        json.dumps(
            structural_semantics,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(b"\x00")
    for name, value in sorted(_payload_arrays(payload).items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\x00")
        digest.update(memoryview(array).cast("B"))
        digest.update(b"\x00")
    digest.update(
        json.dumps(recipes, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256(b"socialgraph-fm-array-v1\x00")
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
    digest.update(b"\x00")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _source_file_digests(entries: list[UploadedEntry]) -> list[SourceFileDigest]:
    return [
        SourceFileDigest(
            path=entry.name,
            role=_file_role(entry.name),
            size=len(entry.data),
            sha256=hashlib.sha256(entry.data).hexdigest(),
        )
        for entry in sorted(entries, key=lambda item: item.name.casefold())
    ]


def _array_role(name: str) -> ArrayRole:
    if name == "edge_index":
        return "edge_index"
    if name == "node_id_map":
        return "node_id_map"
    if name == "x" or name.endswith("_x"):
        return "feature"
    if name == "y" or name.endswith("_labels"):
        return "label"
    if name in _SPLIT_KEYS or "train_" in name or "test_" in name or "val_" in name:
        return "split"
    if name.startswith("variant_"):
        return "variant"
    return "auxiliary"


def _attachment_arrays(
    attachments: dict[str, bytes],
) -> tuple[dict[str, np.ndarray], list[ArrayDescriptor]]:
    resolved: dict[str, np.ndarray] = {}
    descriptors: list[ArrayDescriptor] = []
    for path, data in sorted(attachments.items()):
        if not path.casefold().endswith(".npz"):
            continue
        arrays = _read_npz(UploadedEntry(path, data), trusted_generated=True)
        for key, value in sorted(arrays.items()):
            logical_name = f"{path}#{key}"
            resolved[logical_name] = value
            descriptors.append(
                ArrayDescriptor(
                    name=logical_name,
                    role=_array_role(key),
                    dtype=value.dtype.str,
                    shape=list(value.shape),
                    sha256=_array_sha256(value),
                )
            )
    return resolved, descriptors


def _array_descriptors(
    arrays: dict[str, np.ndarray],
    attachments: dict[str, bytes],
) -> tuple[list[ArrayDescriptor], dict[str, np.ndarray]]:
    descriptors = [
        ArrayDescriptor(
            name=name,
            role=_array_role(name),
            dtype=value.dtype.str,
            shape=list(value.shape),
            sha256=_array_sha256(value),
        )
        for name, value in sorted(arrays.items())
    ]
    attachment_values, attachment_descriptors = _attachment_arrays(attachments)
    return [*descriptors, *attachment_descriptors], attachment_values


def _normalise_feature_recipes(
    recipes: object,
    arrays: dict[str, np.ndarray],
) -> list[FeatureRecipe]:
    result: list[FeatureRecipe] = []
    for raw in recipes if isinstance(recipes, list) else []:
        if not isinstance(raw, dict):
            continue
        graph_variant = str(raw.get("graphVariant", "raw"))
        output = raw.get("featureArray")
        if not isinstance(output, str):
            output = "x" if graph_variant == "raw" and "x" in arrays else None
        fit_scope = raw.get("fitScope", "none")
        if fit_scope not in {"none", "train_only", "all_nodes_transductive"}:
            fit_scope = "none"
        result.append(
            FeatureRecipe(
                id=str(raw.get("id", "identity-v1")),
                graphVariant=graph_variant,
                inputArray="x" if "x" in arrays else None,
                outputArray=output,
                featureTransform=str(raw.get("featureTransform", "identity")),
                fitScope=fit_scope,
                parameters=raw.get("parameters", {})
                if isinstance(raw.get("parameters", {}), dict)
                else {},
            )
        )
    if not result:
        result.append(
            FeatureRecipe(
                id="identity-v1",
                graphVariant="raw",
                inputArray="x" if "x" in arrays else None,
                outputArray="x" if "x" in arrays else None,
                featureTransform="identity",
                fitScope="none",
            )
        )
    return result


def _graph_variants(
    payload: GraphPayload,
    recipes: list[FeatureRecipe],
    raw_recipes: object,
) -> list[GraphVariant]:
    raw_recipe_items = raw_recipes if isinstance(raw_recipes, list) else []
    raw_by_id = {
        str(recipe.get("id", "")): recipe
        for recipe in raw_recipe_items
        if isinstance(recipe, dict)
    }
    variants: dict[str, GraphVariant] = {}
    for recipe in recipes:
        raw = raw_by_id.get(recipe.id, {})
        edge_array = raw.get("edgeIndexArray") if isinstance(raw, dict) else None
        if not isinstance(edge_array, str):
            edge_array = "edge_index"
        directed = raw.get("directed", payload.directed) if isinstance(raw, dict) else payload.directed
        variants[recipe.graph_variant] = GraphVariant(
            id=recipe.graph_variant,
            edgeIndexArray=edge_array,
            featureArray=recipe.output_array,
            directed=bool(directed),
        )
    variants.setdefault(
        "raw",
        GraphVariant(
            id="raw",
            edgeIndexArray="edge_index",
            featureArray="x" if payload.features is not None else None,
            directed=payload.directed,
        ),
    )
    return list(variants.values())


def _split_fold_counts(arrays: dict[str, np.ndarray]) -> list[SplitFoldCounts]:
    names = [name for name in _SPLIT_KEYS if name in arrays]
    if not names:
        return []
    fold_count = max(1 if arrays[name].ndim == 1 else int(arrays[name].shape[1]) for name in names)
    result: list[SplitFoldCounts] = []
    for fold in range(fold_count):
        counts: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        for prefix, target in (("train", "train"), ("val", "validation"), ("test", "test")):
            name = next((key for key in names if key.startswith(prefix + "_")), None)
            if name is None:
                continue
            value = arrays[name]
            column = value if value.ndim == 1 else value[:, fold]
            counts[target] = int(np.count_nonzero(column)) if name.endswith("_mask") else int(column.size)
        result.append(SplitFoldCounts(**counts))
    return result


def _split_sets(
    payload: GraphPayload,
    raw_manifest: dict[str, object],
    attachment_values: dict[str, np.ndarray],
) -> list[SplitSet]:
    result: list[SplitSet] = []
    if payload.splits:
        names = set(payload.splits)
        representation: Literal["mask", "index"] = (
            "mask" if any(name.endswith("_mask") for name in names) else "index"
        )
        split_arrays: dict[Literal["train", "validation", "test"], str] = {}
        split_roles: tuple[
            tuple[str, Literal["train", "validation", "test"]], ...
        ] = (("train", "train"), ("val", "validation"), ("test", "test"))
        for prefix, split_role in split_roles:
            name = next((key for key in _SPLIT_KEYS if key.startswith(prefix + "_") and key in names), None)
            if name:
                split_arrays[split_role] = name
        raw_kind = raw_manifest.get("splitKind", "source")
        kind: Literal["official", "published", "source", "few_shot"]
        if raw_kind == "official":
            kind = "official"
        elif raw_kind == "published":
            kind = "published"
        else:
            kind = "source"
        counts = _split_fold_counts(payload.splits)
        split_files = raw_manifest.get("splitFiles", [])
        result.append(
            SplitSet(
                id="source-splits",
                kind=kind,
                target="node",
                representation=representation,
                arrays=split_arrays,
                foldCount=max(1, len(counts)),
                foldCounts=counts,
                source=",".join(str(value) for value in split_files)
                if isinstance(split_files, list)
                else None,
            )
        )
    episodes = raw_manifest.get("fewShotEpisodes", [])
    if isinstance(episodes, list):
        for episode in episodes:
            if not isinstance(episode, dict):
                continue
            path = episode.get("artifactPath")
            if not isinstance(path, str):
                continue
            prefix = f"{path}#"
            train_name = next(
                (
                    name
                    for name in attachment_values
                    if name.startswith(prefix) and name.endswith(("split_train_idx", "train_idx"))
                ),
                None,
            )
            test_name = next(
                (
                    name
                    for name in attachment_values
                    if name.startswith(prefix) and name.endswith(("split_test_idx", "test_idx"))
                ),
                None,
            )
            if train_name is None or test_name is None:
                continue
            shot = str(episode.get("shot", "few-shot"))
            episode_id = str(episode.get("episode", "0"))
            seed_value = next(
                (value for name, value in attachment_values.items() if name.startswith(prefix) and name.endswith("split_seed")),
                None,
            )
            seed = int(seed_value.reshape(-1)[0]) if seed_value is not None and seed_value.size else None
            result.append(
                SplitSet(
                    id=f"fewshot-{re.sub(r'[^a-zA-Z0-9_-]+', '-', shot)}-{re.sub(r'[^a-zA-Z0-9_-]+', '-', episode_id)}",
                    kind="few_shot",
                    target="node",
                    representation="index",
                    arrays={"train": train_name, "test": test_name},
                    foldCount=1,
                    foldCounts=[
                        SplitFoldCounts(
                            train=int(attachment_values[train_name].size),
                            validation=0,
                            test=int(attachment_values[test_name].size),
                        )
                    ],
                    seed=seed,
                    source=path,
                )
            )
    return result


def _license_evidence(raw_manifest: dict[str, object]) -> list[LicenseEvidence]:
    raw = raw_manifest.get("licenseEvidence")
    if not isinstance(raw, list):
        return []
    result: list[LicenseEvidence] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            result.append(LicenseEvidence.model_validate(item))
        except ValidationError:
            continue
    return result


def _license_policy(
    raw_manifest: dict[str, object],
    *,
    trusted_generated: bool = False,
) -> LicensePolicy:
    structured = raw_manifest.get("licensePolicy")
    if isinstance(structured, dict):
        policy = LicensePolicy.model_validate(structured)
        evidence = _license_evidence(raw_manifest)
        evidence_ids = {item.id for item in evidence}
        has_official_evidence = any(
            item.kind in {"official_metadata", "official_license"}
            and item.source_url is not None
            and item.source_url.startswith(("https://ogb.stanford.edu/", "https://snap.stanford.edu/"))
            for item in evidence
        )
        # An uploaded manifest is data, not a license authority. Only the
        # trusted local adapter may promote a claim backed by official evidence.
        if policy.status == "verified" and not (trusted_generated and has_official_evidence):
            return LicensePolicy(
                status="unknown",
                identifier=policy.identifier,
                sourceUrl=policy.source_url,
                allowedUses=[],
                attribution=policy.attribution,
                evidenceIds=sorted(evidence_ids),
            )
        if policy.status == "user_attested" and not any(
            item.kind == "user_attestation" for item in evidence
        ):
            return LicensePolicy(
                status="unknown",
                identifier=policy.identifier,
                allowedUses=[],
                evidenceIds=sorted(evidence_ids),
            )
        policy.evidence_ids = sorted(evidence_ids)
        return policy
    value = str(raw_manifest.get("license", "unknown")).strip() or "unknown"
    lowered = value.casefold()
    if lowered in {"unknown", "none", "unspecified"}:
        return LicensePolicy(status="unknown", identifier=value, allowedUses=[])
    if "research" in lowered:
        return LicensePolicy(
            status="restricted",
            identifier=value,
            allowedUses=["evaluation"],
        )
    return LicensePolicy(
        status="unknown",
        identifier=value,
        allowedUses=[],
    )


def _dataset_role(
    raw_manifest: dict[str, object],
) -> Literal["benchmark", "target_domain", "pretraining_candidate"]:
    value = raw_manifest.get("datasetRole", "target_domain")
    if value == "benchmark":
        return "benchmark"
    if value == "pretraining_candidate":
        return "pretraining_candidate"
    return "target_domain"


def _training_ref_hash(reference: TrainingDatasetRef) -> str:
    excluded = {"ref_hash"}
    if reference.schema_version == "1.0":
        excluded.add("split_fold")
    value = reference.model_dump(mode="json", by_alias=True, exclude=excluded)
    return hashlib.sha256(
        (
            b"socialgraph-fm-training-ref-v1.1\x00"
            if reference.schema_version == "1.1"
            else b"socialgraph-fm-training-ref-v1\x00"
        )
        + json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _contract_content_hash(
    *,
    descriptors: list[ArrayDescriptor],
    node_identity: NodeIdentitySchema,
    graph_semantics: GraphSemantics,
    graph_variants: list[GraphVariant],
    feature_schemas: list[FeatureSchema],
    label_schemas: list[LabelSchema],
    feature_recipes: list[FeatureRecipe],
    split_sets: list[SplitSet],
    task_specs: list[TaskSpec],
    schema_version: str = "2.2",
) -> str:
    value: dict[str, object] = {
        "arrays": [item.model_dump(mode="json", by_alias=True) for item in descriptors],
        "nodeIdentity": node_identity.model_dump(mode="json", by_alias=True),
        "graphSemantics": graph_semantics.model_dump(mode="json", by_alias=True),
        "graphVariants": [item.model_dump(mode="json", by_alias=True) for item in graph_variants],
        "featureSchemas": [item.model_dump(mode="json", by_alias=True) for item in feature_schemas],
        "labelSchemas": [item.model_dump(mode="json", by_alias=True) for item in label_schemas],
        "featureRecipes": [item.model_dump(mode="json", by_alias=True) for item in feature_recipes],
        "splitSets": [item.model_dump(mode="json", by_alias=True) for item in split_sets],
        "taskSpecs": [
            item.model_dump(
                mode="json",
                by_alias=True,
                exclude={"link_prediction_protocol"}
                if schema_version == "2.1"
                else None,
            )
            for item in task_specs
        ],
    }
    return hashlib.sha256(
        (f"socialgraph-fm-dataset-artifact-v{schema_version}\x00").encode("ascii")
        + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _contract_manifest_hash(
    *,
    content_hash: str,
    dataset_role: str,
    source_file_digests: list[SourceFileDigest],
    license_policy: LicensePolicy,
    raw_manifest: dict[str, object],
    derived_manifest: dict[str, object],
    schema_version: str = "2.2",
    license_evidence: list[LicenseEvidence] | None = None,
    data_governance: DataGovernancePolicy | None = None,
) -> str:
    stable_derived = {
        key: value
        for key, value in derived_manifest.items()
        if key not in {"manifestHash", "trainingReference", "trainingReferences"}
    }
    value: dict[str, object] = {
        "contentHash": content_hash,
        "datasetRole": dataset_role,
        "sourceFileDigests": [
            item.model_dump(mode="json", by_alias=True) for item in source_file_digests
        ],
        "licensePolicy": license_policy.model_dump(
            mode="json",
            by_alias=True,
            exclude={"evidence_ids"} if schema_version == "2.1" else None,
        ),
        "rawManifest": raw_manifest,
        "derivedManifest": stable_derived,
    }
    if schema_version == "2.2":
        value["licenseEvidence"] = [
            item.model_dump(mode="json", by_alias=True) for item in (license_evidence or [])
        ]
        value["dataGovernance"] = (
            data_governance.model_dump(mode="json", by_alias=True)
            if data_governance is not None
            else None
        )
    return hashlib.sha256(
        (f"socialgraph-fm-dataset-manifest-v{schema_version}\x00").encode("ascii")
        + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
