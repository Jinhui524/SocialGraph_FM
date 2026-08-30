"""Leakage-safe feature transforms and train-topology structure views."""

from __future__ import annotations

import math
from typing import Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256

from .bundle import CategoricalFeature, MultiHotFeature, NodeFeature, NumericFeature


class _MetadataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class NumericNormalizationMetadata(_MetadataModel):
    name: str
    mean: float
    scale: float = Field(gt=0.0)


class CategoricalVocabularyMetadata(_MetadataModel):
    name: str
    vocabulary: tuple[str, ...]
    unknown_index: Literal[0] = Field(default=0, alias="unknownIndex")


class MultiHotEncodingMetadata(_MetadataModel):
    name: str
    vocabulary: tuple[str, ...]
    unknown_index: Literal[0] = Field(default=0, alias="unknownIndex")


class FeatureTransformMetadata(_MetadataModel):
    schema_version: Literal["socialgraph-fm.core-static-feature-transforms/2.0"] = Field(
        default="socialgraph-fm.core-static-feature-transforms/2.0", alias="schemaVersion"
    )
    fitted_role: Literal["train"] = Field(default="train", alias="fittedRole")
    training_node_indices: tuple[int, ...] = Field(alias="trainingNodeIndices")
    numeric: tuple[NumericNormalizationMetadata, ...]
    categorical: tuple[CategoricalVocabularyMetadata, ...]
    multi_hot: tuple[MultiHotEncodingMetadata, ...] = Field(alias="multiHot")


class SparseMultiHotEncoding(_MetadataModel):
    row_offsets: tuple[int, ...] = Field(alias="rowOffsets")
    column_indices: tuple[int, ...] = Field(alias="columnIndices")
    vocabulary_size: int = Field(alias="vocabularySize", ge=1)


class StructureView(_MetadataModel):
    schema_version: Literal["socialgraph-fm.core-static-structure-view/2.0"] = Field(
        default="socialgraph-fm.core-static-structure-view/2.0", alias="schemaVersion"
    )
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    role: Literal["train"] = "train"
    directed: bool
    node_indices: tuple[int, ...] = Field(alias="nodeIndices")
    edges: tuple[tuple[int, int], ...]
    topology_hash: str = Field(alias="topologyHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash(self):
        expected = _structure_hash(
            self.graph_version_hash, self.directed, self.node_indices, self.edges
        )
        if self.topology_hash != expected:
            raise ValueError("topologyHash does not match the train structure view")
        return self


def _row_count(feature: NodeFeature) -> int:
    if isinstance(feature, MultiHotFeature):
        return len(feature.row_offsets) - 1
    return len(feature.values)


def _validate_training_rows(
    features: tuple[NodeFeature, ...], train_node_indices: Iterable[int]
) -> tuple[int, ...]:
    indices = tuple(int(index) for index in train_node_indices)
    if not indices or len(indices) != len(set(indices)) or any(index < 0 for index in indices):
        raise ValueError("training node indices must be nonempty, unique, and nonnegative")
    row_counts = {_row_count(feature) for feature in features}
    if len(row_counts) > 1:
        raise ValueError("feature row counts must agree")
    if row_counts and max(indices) >= next(iter(row_counts)):
        raise ValueError("training node index is outside the feature rows")
    return tuple(sorted(indices))


def fit_train_only_transforms(
    features: Iterable[NodeFeature], *, train_node_indices: Iterable[int]
) -> FeatureTransformMetadata:
    """Fit metadata from explicitly provided training rows, never topology or labels."""

    feature_tuple = tuple(features)
    indices = _validate_training_rows(feature_tuple, train_node_indices)
    numeric: list[NumericNormalizationMetadata] = []
    categorical: list[CategoricalVocabularyMetadata] = []
    multi_hot: list[MultiHotEncodingMetadata] = []

    for feature in sorted(feature_tuple, key=lambda item: item.name):
        if isinstance(feature, NumericFeature):
            train_values = tuple(feature.values[index] for index in indices)
            mean = math.fsum(train_values) / len(train_values)
            variance = math.fsum((value - mean) ** 2 for value in train_values) / len(
                train_values
            )
            scale = math.sqrt(variance)
            numeric.append(
                NumericNormalizationMetadata(
                    name=feature.name,
                    mean=mean,
                    scale=scale if scale > 0.0 else 1.0,
                )
            )
        elif isinstance(feature, CategoricalFeature):
            training_categories: set[str] = set()
            for index in indices:
                category = feature.values[index]
                if category is not None:
                    training_categories.add(category)
            vocabulary = tuple(sorted(training_categories))
            categorical.append(
                CategoricalVocabularyMetadata(name=feature.name, vocabulary=vocabulary)
            )
        elif isinstance(feature, MultiHotFeature):
            training_values: set[str] = set()
            for index in indices:
                start, end = feature.row_offsets[index : index + 2]
                training_values.update(feature.values[start:end])
            multi_hot.append(
                MultiHotEncodingMetadata(
                    name=feature.name, vocabulary=tuple(sorted(training_values))
                )
            )
    return FeatureTransformMetadata(
        trainingNodeIndices=indices,
        numeric=tuple(numeric),
        categorical=tuple(categorical),
        multiHot=tuple(multi_hot),
    )


def apply_numeric_normalization(
    values: Iterable[float], metadata: NumericNormalizationMetadata
) -> tuple[float, ...]:
    return tuple((float(value) - metadata.mean) / metadata.scale for value in values)


def encode_sparse_multi_hot(
    feature: MultiHotFeature, metadata: MultiHotEncodingMetadata
) -> SparseMultiHotEncoding:
    if feature.name != metadata.name:
        raise ValueError("multi-hot feature and encoding metadata names differ")
    lookup = {value: index + 1 for index, value in enumerate(metadata.vocabulary)}
    columns = tuple(lookup.get(value, metadata.unknown_index) for value in feature.values)
    return SparseMultiHotEncoding(
        rowOffsets=feature.row_offsets,
        columnIndices=columns,
        vocabularySize=len(metadata.vocabulary) + 1,
    )


def _structure_hash(
    graph_version_hash: str,
    directed: bool,
    node_indices: tuple[int, ...],
    edges: tuple[tuple[int, int], ...],
) -> str:
    return canonical_sha256(
        {
            "graphVersionHash": graph_version_hash,
            "role": "train",
            "directed": directed,
            "nodeIndices": node_indices,
            "edges": edges,
        }
    )


def build_training_structure_view(
    *,
    graph_version_hash: str,
    num_nodes: int,
    edges: Iterable[tuple[int, int]],
    directed: bool,
) -> StructureView:
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    canonical_edges: set[tuple[int, int]] = set()
    for source, target in edges:
        if source == target:
            raise ValueError("structure views do not support self-loops")
        if min(source, target) < 0 or max(source, target) >= num_nodes:
            raise ValueError("structure-view endpoint is outside node indices")
        pair = (source, target) if directed or source < target else (target, source)
        canonical_edges.add(pair)
    node_indices = tuple(range(num_nodes))
    ordered_edges = tuple(sorted(canonical_edges))
    return StructureView(
        graphVersionHash=graph_version_hash,
        directed=directed,
        nodeIndices=node_indices,
        edges=ordered_edges,
        topologyHash=_structure_hash(
            graph_version_hash, directed, node_indices, ordered_edges
        ),
    )


__all__ = [
    "CategoricalVocabularyMetadata",
    "FeatureTransformMetadata",
    "MultiHotEncodingMetadata",
    "NumericNormalizationMetadata",
    "SparseMultiHotEncoding",
    "StructureView",
    "apply_numeric_normalization",
    "build_training_structure_view",
    "encode_sparse_multi_hot",
    "fit_train_only_transforms",
]
