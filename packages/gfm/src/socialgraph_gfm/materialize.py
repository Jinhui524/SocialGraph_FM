"""Deterministic, leakage-safe GraphSnapshot to PyG materialization."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from .canonical import canonical_sha256
from .contracts import (
    TRANSFORM_RECIPE_VERSION,
    FeatureTransformArtifact,
    FeatureTransformState,
    FeatureModality,
    GraphSnapshot,
    InternalFeatureManifest,
    MissingPolicy,
    PrivacyLevel,
    feature_key,
)
from .errors import ContractViolation
from .features import CategoryVocabulary, NumericStandardizer
from .profiles import PROFILES, TemporalRule, check_compatibility
from .runtime import require_ml_runtime
from .tensor_digest import canonical_tensor_digest

MaterializationPurpose = Literal["training_smoke", "formal_training", "inference"]
MATERIALIZATION_RECIPE_VERSION = "gfm.materialization-recipe/3.1"


@dataclass(frozen=True)
class TemporalEventView:
    edge_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    relations: tuple[str, ...]
    timestamps_micros: Any

    def manifest(self) -> dict[str, Any]:
        return {
            "edgeIds": list(self.edge_ids),
            "sourceIds": list(self.source_ids),
            "targetIds": list(self.target_ids),
            "relations": list(self.relations),
            "timestampsMicros": self.timestamps_micros.tolist(),
        }


@dataclass(frozen=True)
class MaterializedGraph:
    graph: Any
    events: TemporalEventView
    transform_artifact: FeatureTransformArtifact
    manifest: dict[str, Any]


def _timestamp_micros(value: datetime) -> int:
    return round(value.timestamp() * 1_000_000)


def _feature_key(feature: InternalFeatureManifest) -> str:
    if not feature.owner_type:
        raise ContractViolation(f"Node feature {feature.name!r} requires ownerType")
    return feature_key(feature)


def _tensor_digest(tensor) -> dict[str, Any]:
    return canonical_tensor_digest(tensor)


def _validate_snapshot_hashes(snapshot: GraphSnapshot) -> tuple[str, str]:
    payload_hash = snapshot.payload_hash()
    fact_hash = snapshot.fact_payload_hash()
    if payload_hash != snapshot.ref.content_hash:
        raise ContractViolation(
            "Snapshot content hash mismatch; payload may have changed after validation: "
            f"expected {snapshot.ref.content_hash}, got {payload_hash}"
        )
    if fact_hash != snapshot.ref.fact_hash:
        raise ContractViolation(
            "Snapshot fact hash mismatch; facts may have changed after validation: "
            f"expected {snapshot.ref.fact_hash}, got {fact_hash}"
        )
    return payload_hash, fact_hash


def _validate_feature_contract(snapshot: GraphSnapshot) -> None:
    node_types = {node.node_type for node in snapshot.nodes}
    declared_by_type: dict[str, set[str]] = {node_type: set() for node_type in node_types}
    edge_feature_names: set[str] = set()
    for feature in snapshot.ref.feature_manifest:
        if feature.target.value == "node":
            if feature.owner_type not in node_types:
                raise ContractViolation(
                    f"Feature {feature.name!r} references unknown ownerType {feature.owner_type!r}"
                )
            declared_by_type[feature.owner_type].add(feature.name)
            _validate_dtype(feature)
        elif feature.target.value == "edge":
            edge_feature_names.add(feature.name)
            raise ContractViolation("Edge feature tensorization is not implemented in this phase")
        else:
            raise ContractViolation("Graph feature tensorization is not implemented in this phase")

    for node in snapshot.nodes:
        unknown = set(node.features).difference(declared_by_type[node.node_type])
        if unknown:
            raise ContractViolation(
                f"Node {node.node_id!r} contains undeclared features: {sorted(unknown)}"
            )
    for edge in snapshot.edges:
        unknown = set(edge.features).difference(edge_feature_names)
        if unknown:
            raise ContractViolation(
                f"Edge {edge.edge_id!r} contains undeclared features: {sorted(unknown)}"
            )


def _validate_dtype(feature: InternalFeatureManifest) -> None:
    supported = {
        FeatureModality.NUMERIC: {"float32", "float64", "int32", "int64"},
        FeatureModality.CATEGORICAL: {"string"},
        FeatureModality.TEXT: {"string"},
        FeatureModality.EMBEDDING: {"float32", "float64"},
    }[feature.modality]
    if feature.dtype not in supported:
        raise ContractViolation(
            f"Feature {feature.name!r} dtype {feature.dtype!r} is invalid for {feature.modality.value}"
        )


def _validate_raw_dtype(feature: InternalFeatureManifest, raw: Sequence[Any]) -> None:
    key = _feature_key(feature)
    present = [value for value in raw if value is not None]
    if feature.modality == FeatureModality.NUMERIC:
        if feature.dtype in ("int32", "int64"):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in present):
                raise ContractViolation(f"Feature {key!r} requires integer values for {feature.dtype}")
            lower, upper = (
                (-(2**31), 2**31 - 1)
                if feature.dtype == "int32"
                else (-(2**63), 2**63 - 1)
            )
            if any(value < lower or value > upper for value in present):
                raise ContractViolation(f"Feature {key!r} contains values outside {feature.dtype} range")
        elif any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in present
        ):
            raise ContractViolation(f"Feature {key!r} requires numeric values for {feature.dtype}")
    elif feature.modality in (
        FeatureModality.CATEGORICAL,
        FeatureModality.TEXT,
    ) and any(not isinstance(value, str) for value in present):
        raise ContractViolation(f"Feature {key!r} requires string values for {feature.dtype}")


def _selected_features(
    snapshot: GraphSnapshot,
    purpose: MaterializationPurpose,
    inference_at: datetime | None,
) -> tuple[InternalFeatureManifest, ...]:
    manifests = tuple(
        sorted(snapshot.ref.feature_manifest, key=lambda feature: (feature.owner_type or "", feature.name))
    )
    if purpose in ("training_smoke", "formal_training"):
        if inference_at is not None:
            raise ContractViolation(f"{purpose} does not accept inference_at")
        return manifests
    if inference_at is None or inference_at.tzinfo is None or inference_at.utcoffset() is None:
        raise ContractViolation("inference requires a timezone-aware inference_at")
    allowlist = set(snapshot.ref.resolved_inference_feature_keys())
    selected = tuple(feature for feature in manifests if _feature_key(feature) in allowlist)
    for feature in selected:
        if not feature.inference_allowed or feature.privacy_level == PrivacyLevel.PROHIBITED:
            raise ContractViolation(f"Feature {feature.name!r} is forbidden for inference")
        if feature.available_at is not None and feature.available_at > inference_at:
            raise ContractViolation(
                f"Feature {feature.name!r} is not available at the requested inference time"
            )
    return selected


def _fit_selection(
    *,
    purpose: MaterializationPurpose,
    features: Sequence[InternalFeatureManifest],
    nodes_by_type: Mapping[str, tuple],
    fit_node_ids: Mapping[str, Sequence[str]] | None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any] | None]:
    possible_types = {
        feature.owner_type
        for feature in features
        if feature.modality in (FeatureModality.NUMERIC, FeatureModality.CATEGORICAL)
    }
    fitted_types = {node_type for node_type in possible_types if node_type is not None}
    if purpose == "inference":
        if fit_node_ids is not None:
            raise ContractViolation("inference must use immutable fitted transforms, not fit_node_ids")
        return {}, None
    if fit_node_ids is None:
        raise ContractViolation(f"{purpose} requires explicit fit_node_ids by node type")
    unknown_types = set(fit_node_ids).difference(nodes_by_type)
    if unknown_types:
        raise ContractViolation(f"fit_node_ids contains unknown node types: {sorted(unknown_types)}")
    missing_types = fitted_types.difference(fit_node_ids)
    if missing_types:
        raise ContractViolation(f"fit_node_ids missing fitted node types: {sorted(missing_types)}")

    selection: dict[str, tuple[str, ...]] = {}
    fit_report: dict[str, Any] = {}
    for node_type in sorted(fitted_types):
        available = {node.node_id for node in nodes_by_type[node_type]}
        selected = tuple(sorted(set(fit_node_ids[node_type])))
        if not selected:
            raise ContractViolation(f"fit_node_ids[{node_type!r}] must not be empty")
        if len(selected) != len(tuple(fit_node_ids[node_type])):
            raise ContractViolation(f"fit_node_ids[{node_type!r}] must not contain duplicates")
        unknown_ids = set(selected).difference(available)
        if unknown_ids:
            raise ContractViolation(
                f"fit_node_ids[{node_type!r}] contains unknown IDs: {sorted(unknown_ids)}"
            )
        if purpose == "training_smoke" and len(selected) >= len(available):
            raise ContractViolation(
                f"synthetic training_smoke requires at least one held-out {node_type} node"
            )
        selection[node_type] = selected
        fit_report[node_type] = {
            "count": len(selected),
            "nodeIdsHash": canonical_sha256(selected),
        }
    return selection, fit_report


def _normalizer_from_artifact(feature: InternalFeatureManifest, artifact: FeatureTransformState):
    if artifact.kind != "numeric_standardizer":
        raise ContractViolation(f"Invalid numeric transform artifact for {_feature_key(feature)!r}")
    mean, scale = artifact.mean, artifact.scale
    if (
        isinstance(mean, bool)
        or isinstance(scale, bool)
        or not isinstance(mean, (int, float))
        or not isinstance(scale, (int, float))
        or not math.isfinite(float(mean))
        or not math.isfinite(float(scale))
        or float(scale) <= 0
    ):
        raise ContractViolation(f"Invalid numeric transform values for {_feature_key(feature)!r}")
    return NumericStandardizer(mean=float(mean), scale=float(scale))


def _vocabulary_from_artifact(feature: InternalFeatureManifest, artifact: FeatureTransformState):
    if artifact.kind != "category_vocabulary":
        raise ContractViolation(f"Invalid category transform artifact for {_feature_key(feature)!r}")
    assert artifact.categories is not None
    return CategoryVocabulary(categories=artifact.categories)


def _make_transform_state(
    feature: InternalFeatureManifest, raw_state: Mapping[str, Any]
) -> FeatureTransformState:
    payload: dict[str, Any] = {
        "schemaVersion": "gfm.feature-transform-state/1.0",
        "recipeVersion": TRANSFORM_RECIPE_VERSION,
        "featureKey": _feature_key(feature),
        "ownerType": feature.owner_type,
        "modality": feature.modality,
        "dtype": feature.dtype,
        "dimensions": feature.dimensions,
        "missingPolicy": feature.missing_policy,
        "kind": raw_state["kind"],
        "mean": raw_state.get("mean"),
        "scale": raw_state.get("scale"),
        "categories": tuple(raw_state["categories"]) if "categories" in raw_state else None,
        "categoriesHash": (
            canonical_sha256(tuple(raw_state["categories"])) if "categories" in raw_state else None
        ),
    }
    return FeatureTransformState(**payload, stateHash=canonical_sha256(payload))


def _validate_state_compatibility(
    feature: InternalFeatureManifest, state: FeatureTransformState
) -> None:
    expected = {
        "featureKey": _feature_key(feature),
        "ownerType": feature.owner_type,
        "modality": feature.modality,
        "dtype": feature.dtype,
        "dimensions": feature.dimensions,
        "missingPolicy": feature.missing_policy,
        "recipeVersion": TRANSFORM_RECIPE_VERSION,
    }
    actual = {
        "featureKey": state.feature_key,
        "ownerType": state.owner_type,
        "modality": state.modality,
        "dtype": state.dtype,
        "dimensions": state.dimensions,
        "missingPolicy": state.missing_policy,
        "recipeVersion": state.recipe_version,
    }
    if actual != expected:
        raise ContractViolation(
            f"Transform state is incompatible with feature {_feature_key(feature)!r}"
        )


def _node_features(
    *,
    node_type: str,
    ordered_nodes,
    features: Sequence[InternalFeatureManifest],
    purpose: MaterializationPurpose,
    fit_ids: tuple[str, ...] | None,
    supplied_transforms: Mapping[str, FeatureTransformState],
    torch,
) -> tuple[Any, Any, tuple[str, ...], dict[str, FeatureTransformState]]:
    columns = []
    missing_columns = []
    column_names: list[str] = []
    transforms: dict[str, FeatureTransformState] = {}
    by_id = {node.node_id: node for node in ordered_nodes}
    for feature in (item for item in features if item.owner_type == node_type):
        key = _feature_key(feature)
        raw = [node.features.get(feature.name) for node in ordered_nodes]
        _validate_raw_dtype(feature, raw)
        if feature.missing_policy == MissingPolicy.ERROR and any(value is None for value in raw):
            raise ContractViolation(f"Feature {key!r} rejects missing values")
        if feature.modality == FeatureModality.NUMERIC:
            if feature.missing_policy not in (MissingPolicy.ERROR, MissingPolicy.ZERO_WITH_MASK):
                raise ContractViolation(f"Feature {key!r} has incompatible numeric missing policy")
            if purpose in ("training_smoke", "formal_training"):
                assert fit_ids is not None
                try:
                    transform = NumericStandardizer.fit(
                        [by_id[node_id].features.get(feature.name) for node_id in fit_ids]
                    )
                except (TypeError, ValueError) as error:
                    raise ContractViolation(f"Cannot fit feature {key!r}: {error}") from error
                artifact = _make_transform_state(
                    feature,
                    {"kind": "numeric_standardizer", "mean": transform.mean, "scale": transform.scale},
                )
            else:
                artifact = supplied_transforms[key]
                _validate_state_compatibility(feature, artifact)
                transform = _normalizer_from_artifact(feature, artifact)
            try:
                values, missing = transform.transform(raw)
            except (TypeError, ValueError) as error:
                raise ContractViolation(f"Cannot transform feature {key!r}: {error}") from error
            columns.append(torch.tensor(values, dtype=torch.float32).reshape(-1, 1))
            missing_columns.append(torch.tensor(missing, dtype=torch.bool).reshape(-1, 1))
            column_names.append(feature.name)
            transforms[key] = artifact
        elif feature.modality == FeatureModality.CATEGORICAL:
            if feature.missing_policy not in (MissingPolicy.ERROR, MissingPolicy.CATEGORY_WITH_MASK):
                raise ContractViolation(f"Feature {key!r} has incompatible categorical missing policy")
            if purpose in ("training_smoke", "formal_training"):
                assert fit_ids is not None
                try:
                    category_transform = CategoryVocabulary.fit(
                        [by_id[node_id].features.get(feature.name) for node_id in fit_ids]
                    )
                except (TypeError, ValueError) as error:
                    raise ContractViolation(f"Cannot fit feature {key!r}: {error}") from error
                artifact = _make_transform_state(
                    feature,
                    {"kind": "category_vocabulary", "categories": list(category_transform.categories)},
                )
            else:
                artifact = supplied_transforms[key]
                _validate_state_compatibility(feature, artifact)
                category_transform = _vocabulary_from_artifact(feature, artifact)
            try:
                category_values, category_missing = category_transform.transform(raw)
            except (TypeError, ValueError) as error:
                raise ContractViolation(f"Cannot transform feature {key!r}: {error}") from error
            columns.append(torch.tensor(category_values, dtype=torch.float32).reshape(-1, 1))
            missing_columns.append(
                torch.tensor(category_missing, dtype=torch.bool).reshape(-1, 1)
            )
            column_names.append(feature.name)
            transforms[key] = artifact
        elif feature.modality == FeatureModality.TEXT:
            raise ContractViolation(f"Raw text feature {key!r} requires a versioned embedding adapter")
        else:
            embedding_values: list[list[float]] = []
            embedding_missing: list[list[bool]] = []
            for value in raw:
                if value is None:
                    if feature.missing_policy == MissingPolicy.ERROR:
                        raise ContractViolation(f"Embedding feature {key!r} rejects missing values")
                    embedding_values.append([0.0] * feature.dimensions)
                    embedding_missing.append([True] * feature.dimensions)
                elif not isinstance(value, (list, tuple)) or len(value) != feature.dimensions:
                    raise ContractViolation(f"Embedding feature {key!r} has invalid dimensions")
                elif not all(
                    not isinstance(item, bool)
                    and isinstance(item, (int, float))
                    and math.isfinite(float(item))
                    for item in value
                ):
                    raise ContractViolation(f"Embedding feature {key!r} must contain finite numbers")
                else:
                    embedding_values.append([float(item) for item in value])
                    embedding_missing.append([False] * feature.dimensions)
            columns.append(torch.tensor(embedding_values, dtype=torch.float32))
            missing_columns.append(torch.tensor(embedding_missing, dtype=torch.bool))
            column_names.extend(f"{feature.name}[{index}]" for index in range(feature.dimensions))

    if not columns:
        return (
            torch.ones((len(ordered_nodes), 1), dtype=torch.float32),
            torch.zeros((len(ordered_nodes), 1), dtype=torch.bool),
            ("__structural_constant__",),
            transforms,
        )
    return (
        torch.cat(columns, dim=1),
        torch.cat(missing_columns, dim=1),
        tuple(column_names),
        transforms,
    )


def _assert_edge_index_range(edge_index, source_count: int, target_count: int, label: str) -> None:
    if tuple(edge_index.shape[:1]) != (2,):
        raise ContractViolation(f"{label} edge_index must have shape [2, E]")
    if edge_index.numel() == 0:
        return
    source_min, source_max = int(edge_index[0].min()), int(edge_index[0].max())
    target_min, target_max = int(edge_index[1].min()), int(edge_index[1].max())
    if source_min < 0 or source_max >= source_count or target_min < 0 or target_max >= target_count:
        raise ContractViolation(f"{label} edge_index contains an out-of-range endpoint")


def _visible_edges(snapshot: GraphSnapshot, cutoff: datetime | None) -> tuple:
    """Return the point-in-time edge view used by every materialized representation.

    Event relations are observations and become visible at their timestamp. Stable
    relations use a half-open validity interval: ``validFrom <= cutoff < validTo``.
    A missing interval bound is unbounded. Synthetic smoke materialization deliberately
    passes no cutoff and therefore preserves its historical full-snapshot behaviour.
    """

    if cutoff is None:
        return tuple(snapshot.edges)
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ContractViolation("edge visibility cutoff must be timezone-aware")
    rules = {
        relation.relation: relation.temporal_rule
        for relation in PROFILES[snapshot.ref.profile].relations
    }
    visible = []
    for edge in snapshot.edges:
        rule = rules.get(edge.relation)
        if rule == TemporalRule.EVENT:
            if edge.timestamp is not None and edge.timestamp <= cutoff:
                visible.append(edge)
            continue
        if rule == TemporalRule.STABLE:
            starts = edge.valid_from is None or edge.valid_from <= cutoff
            has_not_ended = edge.valid_to is None or cutoff < edge.valid_to
            if starts and has_not_ended:
                visible.append(edge)
            continue
        raise ContractViolation(f"No temporal rule for relation {edge.relation!r}")
    return tuple(visible)


def materialize(
    snapshot: GraphSnapshot,
    *,
    purpose: MaterializationPurpose,
    fit_node_ids: Mapping[str, Sequence[str]] | None = None,
    transform_artifact: FeatureTransformArtifact | None = None,
    inference_at: datetime | None = None,
    observation_end: datetime | None = None,
    device: str = "cpu",
) -> MaterializedGraph:
    torch, _ = require_ml_runtime(device)
    from torch_geometric.data import Data, HeteroData

    if purpose not in ("training_smoke", "formal_training", "inference"):
        raise ContractViolation(f"Unsupported materialization purpose: {purpose!r}")
    if purpose == "training_smoke":
        if inference_at is not None or observation_end is not None:
            raise ContractViolation("training_smoke does not accept a point-in-time cutoff")
        cutoff = None
    elif purpose == "formal_training":
        if inference_at is not None:
            raise ContractViolation("formal_training uses observation_end, not inference_at")
        if (
            observation_end is None
            or observation_end.tzinfo is None
            or observation_end.utcoffset() is None
        ):
            raise ContractViolation(
                "formal_training requires a timezone-aware observation_end"
            )
        cutoff = observation_end
    else:
        if observation_end is not None:
            raise ContractViolation("inference uses inference_at, not observation_end")
        cutoff = inference_at
    snapshot_payload_hash, fact_payload_hash = _validate_snapshot_hashes(snapshot)
    _validate_feature_contract(snapshot)
    report = check_compatibility(snapshot)
    if not report.compatible:
        codes = ", ".join(issue.code for issue in report.blockers)
        raise ContractViolation(f"Graph snapshot is incompatible with its profile: {codes}")
    visible_edges = _visible_edges(snapshot, cutoff)

    node_types = sorted({node.node_type for node in snapshot.nodes})
    nodes_by_type = {
        node_type: tuple(
            sorted(
                (node for node in snapshot.nodes if node.node_type == node_type),
                key=lambda node: node.node_id,
            )
        )
        for node_type in node_types
    }
    node_index_by_type = {
        node_type: {node.node_id: index for index, node in enumerate(nodes_by_type[node_type])}
        for node_type in node_types
    }
    selected_features = _selected_features(snapshot, purpose, inference_at)
    selection, fit_report = _fit_selection(
        purpose=purpose,
        features=selected_features,
        nodes_by_type=nodes_by_type,
        fit_node_ids=fit_node_ids,
    )
    fittable_keys = {
        _feature_key(feature)
        for feature in selected_features
        if feature.modality in (FeatureModality.NUMERIC, FeatureModality.CATEGORICAL)
    }
    if purpose in ("training_smoke", "formal_training") and transform_artifact is not None:
        raise ContractViolation(f"{purpose} must fit a new transform artifact from fit_node_ids")
    supplied_transforms: dict[str, FeatureTransformState] = {}
    if purpose == "inference":
        if transform_artifact is None:
            raise ContractViolation("inference requires an immutable FeatureTransformArtifact")
        try:
            transform_artifact = FeatureTransformArtifact.model_validate(
                transform_artifact.model_dump(mode="python", by_alias=True)
            )
        except Exception as error:
            raise ContractViolation(f"Invalid FeatureTransformArtifact: {error}") from error
        if transform_artifact.recipe_version != TRANSFORM_RECIPE_VERSION:
            raise ContractViolation("Transform artifact recipeVersion is unsupported")
        supplied_transforms = {state.feature_key: state for state in transform_artifact.states}
        missing = fittable_keys.difference(supplied_transforms)
        if missing:
            raise ContractViolation(
                f"Inference transform keys mismatch; missing={sorted(missing)}"
            )

    column_manifest: dict[str, tuple[str, ...]] = {}
    transform_states: dict[str, FeatureTransformState] = {}
    tensor_digests: dict[str, Any] = {"nodes": {}, "edges": {}}
    if len(node_types) == 1 and snapshot.ref.profile.endswith("actor-interaction/1.0"):
        node_type = node_types[0]
        ordered_nodes = nodes_by_type[node_type]
        x, missing_mask, names, transforms = _node_features(
            node_type=node_type,
            ordered_nodes=ordered_nodes,
            features=selected_features,
            purpose=purpose,
            fit_ids=selection.get(node_type),
            supplied_transforms=supplied_transforms,
            torch=torch,
        )
        transform_states.update(transforms)
        column_manifest[node_type] = names
        ordered_edges = tuple(sorted(visible_edges, key=lambda edge: edge.edge_id))
        local_index = node_index_by_type[node_type]
        edge_index = torch.tensor(
            [
                [local_index[edge.source] for edge in ordered_edges],
                [local_index[edge.target] for edge in ordered_edges],
            ],
            dtype=torch.long,
        )
        _assert_edge_index_range(edge_index, len(ordered_nodes), len(ordered_nodes), "actor")
        edge_weight = torch.tensor([edge.weight for edge in ordered_edges], dtype=torch.float32)
        graph = Data(x=x, missing_mask=missing_mask, edge_index=edge_index, edge_weight=edge_weight)
        graph.node_ids = tuple(node.node_id for node in ordered_nodes)
        graph.edge_ids = tuple(edge.edge_id for edge in ordered_edges)
        graph.relation_names = tuple(edge.relation for edge in ordered_edges)
        tensor_digests["nodes"][node_type] = {
            "x": _tensor_digest(x),
            "missingMask": _tensor_digest(missing_mask),
        }
        tensor_digests["edges"]["actor"] = {
            "edgeIndex": _tensor_digest(edge_index),
            "edgeWeight": _tensor_digest(edge_weight),
        }
    else:
        graph = HeteroData()
        for node_type in node_types:
            ordered_nodes = nodes_by_type[node_type]
            x, missing_mask, names, transforms = _node_features(
                node_type=node_type,
                ordered_nodes=ordered_nodes,
                features=selected_features,
                purpose=purpose,
                fit_ids=selection.get(node_type),
                supplied_transforms=supplied_transforms,
                torch=torch,
            )
            transform_states.update(transforms)
            column_manifest[node_type] = names
            graph[node_type].x = x
            graph[node_type].missing_mask = missing_mask
            graph[node_type].node_ids = tuple(node.node_id for node in ordered_nodes)
            tensor_digests["nodes"][node_type] = {
                "x": _tensor_digest(x),
                "missingMask": _tensor_digest(missing_mask),
            }
        relation_groups: dict[tuple[str, str, str], list] = {}
        by_id = {node.node_id: node for node in snapshot.nodes}
        for edge in sorted(visible_edges, key=lambda item: item.edge_id):
            key = (by_id[edge.source].node_type, edge.relation, by_id[edge.target].node_type)
            relation_groups.setdefault(key, []).append(edge)
        for edge_type in sorted(relation_groups):
            source_type, relation, target_type = edge_type
            edges = relation_groups[edge_type]
            edge_index = torch.tensor(
                [
                    [node_index_by_type[source_type][edge.source] for edge in edges],
                    [node_index_by_type[target_type][edge.target] for edge in edges],
                ],
                dtype=torch.long,
            )
            _assert_edge_index_range(
                edge_index,
                len(nodes_by_type[source_type]),
                len(nodes_by_type[target_type]),
                f"{source_type}:{relation}:{target_type}",
            )
            edge_weight = torch.tensor([edge.weight for edge in edges], dtype=torch.float32)
            graph[edge_type].edge_index = edge_index
            graph[edge_type].edge_weight = edge_weight
            graph[edge_type].edge_ids = tuple(edge.edge_id for edge in edges)
            tensor_digests["edges"][f"{source_type}:{relation}:{target_type}"] = {
                "edgeIndex": _tensor_digest(edge_index),
                "edgeWeight": _tensor_digest(edge_weight),
            }

    profile = PROFILES[snapshot.ref.profile]
    event_relations = {
        relation.relation
        for relation in profile.relations
        if relation.temporal_rule == TemporalRule.EVENT
    }
    temporal_edges = tuple(
        edge
        for edge in visible_edges
        if edge.timestamp is not None and edge.relation in event_relations
    )

    def temporal_key(edge):
        assert edge.timestamp is not None
        return (_timestamp_micros(edge.timestamp), edge.edge_id)

    temporal_edges = tuple(sorted(temporal_edges, key=temporal_key))

    def event_micros(edge) -> int:
        assert edge.timestamp is not None
        return _timestamp_micros(edge.timestamp)

    timestamps = torch.tensor([event_micros(edge) for edge in temporal_edges], dtype=torch.int64)
    events = TemporalEventView(
        edge_ids=tuple(edge.edge_id for edge in temporal_edges),
        source_ids=tuple(edge.source for edge in temporal_edges),
        target_ids=tuple(edge.target for edge in temporal_edges),
        relations=tuple(edge.relation for edge in temporal_edges),
        timestamps_micros=timestamps,
    )
    tensor_digests["events"] = {"timestampsMicros": _tensor_digest(timestamps)}
    selection_payload = {key: list(selection[key]) for key in sorted(selection)}
    fit_selection_hash = (
        canonical_sha256(selection_payload)
        if purpose in ("training_smoke", "formal_training")
        else transform_artifact.fit_selection_hash  # type: ignore[union-attr]
    )
    if purpose in ("training_smoke", "formal_training"):
        artifact_payload: dict[str, Any] = {
            "schemaVersion": "gfm.feature-transform-artifact/1.0",
            "recipeVersion": TRANSFORM_RECIPE_VERSION,
            "sourceSnapshotPayloadHash": snapshot_payload_hash,
            "sourceSnapshotContractHash": canonical_sha256(snapshot),
            "fitSelectionHash": fit_selection_hash,
            "states": tuple(transform_states[key] for key in sorted(transform_states)),
        }
        transform_artifact = FeatureTransformArtifact(
            schemaVersion="gfm.feature-transform-artifact/1.0",
            recipeVersion=TRANSFORM_RECIPE_VERSION,
            sourceSnapshotPayloadHash=snapshot_payload_hash,
            sourceSnapshotContractHash=canonical_sha256(snapshot),
            fitSelectionHash=fit_selection_hash,
            states=tuple(transform_states[key] for key in sorted(transform_states)),
            artifactHash=canonical_sha256(artifact_payload),
        )
    assert transform_artifact is not None
    logical_manifest = {
        "schemaVersion": "gfm.materialization-logical/3.0",
        "recipeVersion": MATERIALIZATION_RECIPE_VERSION,
        "purpose": purpose,
        "graphVersion": snapshot.ref.graph_version,
        "profile": snapshot.ref.profile,
        "snapshotPayloadHash": snapshot_payload_hash,
        "factPayloadHash": fact_payload_hash,
        "snapshotContractHash": canonical_sha256(snapshot),
        "nodeCounts": {node_type: len(nodes_by_type[node_type]) for node_type in node_types},
        "edgeCount": len(visible_edges),
        "sourceEdgeCount": len(snapshot.edges),
        "visibleEdgeCount": len(visible_edges),
        "excludedFutureEdgeCount": len(snapshot.edges) - len(visible_edges),
        "edgeVisibilityCutoff": cutoff,
        "visibleEdgeDigest": canonical_sha256(
            tuple(sorted(visible_edges, key=lambda edge: edge.edge_id))
        ),
        "nodeIndexByType": node_index_by_type,
        "featureColumns": column_manifest,
        "selectedFeatureKeys": [_feature_key(feature) for feature in selected_features],
        "fitScope": (
            "synthetic_explicit_train_nodes"
            if purpose == "training_smoke"
            else "explicit_train_split_nodes"
            if purpose == "formal_training"
            else "immutable_prefit"
        ),
        "fitSelection": fit_report,
        "fitSelectionHash": fit_selection_hash,
        "transformArtifact": transform_artifact,
        "transformArtifactHash": transform_artifact.artifact_hash,
        "eventCount": len(temporal_edges),
        "tensorDigests": tensor_digests,
    }
    materialization_hash = canonical_sha256(logical_manifest)
    manifest = {
        **logical_manifest,
        "materializationHash": materialization_hash,
        "executionDevice": device,
    }
    graph = graph.to(device)
    execution_events = TemporalEventView(
        edge_ids=events.edge_ids,
        source_ids=events.source_ids,
        target_ids=events.target_ids,
        relations=events.relations,
        timestamps_micros=events.timestamps_micros.to(device),
    )
    return MaterializedGraph(
        graph=graph,
        events=execution_events,
        transform_artifact=transform_artifact,
        manifest=manifest,
    )


def homogeneous_tensors(materialized: MaterializedGraph):
    """Return one x/edge_index view for the smoke encoder without mutating PyG data."""

    graph = materialized.graph
    if not hasattr(graph, "node_types"):
        return graph.x, graph.edge_index

    import torch

    node_types = sorted(graph.node_types)
    max_width = max(int(graph[node_type].x.shape[1]) for node_type in node_types)
    offsets: dict[str, int] = {}
    pieces = []
    offset = 0
    for node_type in node_types:
        value = graph[node_type].x
        if value.shape[1] < max_width:
            value = torch.nn.functional.pad(value, (0, max_width - value.shape[1]))
        offsets[node_type] = offset
        offset += int(value.shape[0])
        pieces.append(value)
    edge_pieces = []
    for source_type, relation, target_type in sorted(graph.edge_types):
        edge_index = graph[(source_type, relation, target_type)].edge_index.clone()
        edge_index[0] += offsets[source_type]
        edge_index[1] += offsets[target_type]
        edge_pieces.append(edge_index)
    x = torch.cat(pieces, dim=0)
    edge_index = (
        torch.cat(edge_pieces, dim=1)
        if edge_pieces
        else torch.empty((2, 0), dtype=torch.long, device=x.device)
    )
    _assert_edge_index_range(edge_index, int(x.shape[0]), int(x.shape[0]), "homogeneous")
    return x, edge_index
