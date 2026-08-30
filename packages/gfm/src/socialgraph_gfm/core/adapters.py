"""Bounded, schema-specific adapters for SocialGraph-FM Core inputs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

import torch
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .bundle import CategoricalFeature, MultiHotFeature, NumericFeature, CoreGraphBundle


HIDDEN_DIM = 128
MAX_ADAPTER_FIELDS = 256
MAX_VOCABULARY_ENTRIES = 4_095
MAX_ADAPTER_SCHEMA_BYTES = 256 * 1024
MAX_MULTIHOT_TRAINING_TOKENS = MAX_VOCABULARY_ENTRIES


class _SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


class VocabularyEntry(_SchemaModel):
    index: int = Field(ge=1, le=MAX_VOCABULARY_ENTRIES)
    token: str = Field(min_length=1, max_length=4096)


class NumericFieldSchema(_SchemaModel):
    kind: Literal["numeric"]
    name: str = Field(min_length=1, max_length=200)
    mean: float
    scale: float = Field(gt=0)


class CategoricalFieldSchema(_SchemaModel):
    kind: Literal["categorical"]
    name: str = Field(min_length=1, max_length=200)
    vocabulary: tuple[VocabularyEntry, ...] = Field(max_length=MAX_VOCABULARY_ENTRIES)

    @model_validator(mode="after")
    def validate_vocabulary(self):
        if tuple(entry.index for entry in self.vocabulary) != tuple(
            range(1, len(self.vocabulary) + 1)
        ):
            raise ValueError("categorical vocabulary indices must be contiguous from one")
        tokens = tuple(entry.token for entry in self.vocabulary)
        if tokens != tuple(sorted(tokens)) or len(tokens) != len(set(tokens)):
            raise ValueError("categorical vocabulary tokens must be unique and sorted")
        return self


class MultiHotFieldSchema(_SchemaModel):
    kind: Literal["multiHot"]
    name: str = Field(min_length=1, max_length=200)
    hash_algorithm: Literal["sha256-64-be-oov-zero/1.0"] = Field(alias="hashAlgorithm")
    bucket_count: int = Field(alias="bucketCount", ge=2, le=65_536)
    known_token_digests: tuple[str, ...] = Field(
        alias="knownTokenDigests", max_length=MAX_MULTIHOT_TRAINING_TOKENS
    )

    @model_validator(mode="after")
    def validate_known_tokens(self):
        if self.known_token_digests != tuple(sorted(self.known_token_digests)):
            raise ValueError("multi-hot training token digests must be sorted")
        if len(self.known_token_digests) != len(set(self.known_token_digests)):
            raise ValueError("multi-hot training token digests must be unique")
        if any(
            len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            for value in self.known_token_digests
        ):
            raise ValueError("multi-hot training token digest is invalid")
        return self


class StructureFieldSchema(_SchemaModel):
    kind: Literal["structure"]
    names: tuple[str, ...] = Field(min_length=1, max_length=32)
    means: tuple[float, ...] = Field(min_length=1, max_length=32)
    scales: tuple[float, ...] = Field(min_length=1, max_length=32)
    algorithm_version: Literal["socialgraph-fm.core-visible-topology-structure/1.0"] = Field(
        alias="algorithmVersion"
    )

    @model_validator(mode="after")
    def validate_width(self):
        if len(self.names) != len(self.means) or len(self.names) != len(self.scales):
            raise ValueError("structure schema widths must match")
        if len(set(self.names)) != len(self.names):
            raise ValueError("structure field names must be unique")
        if not all(math.isfinite(value) for value in (*self.means, *self.scales)):
            raise ValueError("structure normalization must be finite")
        if not all(scale > 0 for scale in self.scales):
            raise ValueError("structure scales must be positive")
        return self


AdapterFieldSchema = Annotated[
    NumericFieldSchema | CategoricalFieldSchema | MultiHotFieldSchema | StructureFieldSchema,
    Field(discriminator="kind"),
]


class AdapterSchema(_SchemaModel):
    schema_version: Literal["socialgraph-fm.core-adapter-schema/1.1"] = Field(alias="schemaVersion")
    source_graph_version_hash: str = Field(
        alias="sourceGraphVersionHash", pattern=r"^[0-9a-f]{64}$"
    )
    fit_row_ids_hash: str = Field(alias="fitRowIdsHash", pattern=r"^[0-9a-f]{64}$")
    fit_row_count: int = Field(alias="fitRowCount", ge=1)
    visible_topology_hash: str = Field(alias="visibleTopologyHash", pattern=r"^[0-9a-f]{64}$")
    visible_topology_edge_count: int = Field(alias="visibleTopologyEdgeCount", ge=0)
    fields: tuple[AdapterFieldSchema, ...] = Field(min_length=1, max_length=MAX_ADAPTER_FIELDS)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_hash_and_bound(self):
        payload = self.model_dump(mode="python", by_alias=True, exclude={"adapter_schema_hash"})
        if self.adapter_schema_hash != canonical_sha256(payload):
            raise ValueError("adapterSchemaHash does not match canonical adapter schema")
        if len(canonical_json(self).encode("utf-8")) > MAX_ADAPTER_SCHEMA_BYTES:
            raise ValueError("adapter schema exceeds the serialized byte limit")
        return self


def _typed_token(value: object) -> str:
    if isinstance(value, bool):
        type_name = "boolean"
    elif isinstance(value, int):
        type_name = "integer"
    elif isinstance(value, float):
        type_name = "number"
    elif isinstance(value, str):
        type_name = "string"
    else:
        raise ValueError("categorical values must be typed JSON scalars")
    return canonical_json({"type": type_name, "value": value})


def _typed_token_digest(value: object) -> str:
    return hashlib.sha256(_typed_token(value).encode("utf-8")).hexdigest()


def _split_edge_id(source_id: str, target_id: str) -> str:
    return f"edge:{source_id}:{target_id}"


def _semantic_edge_payload(bundle: CoreGraphBundle, edge_index: int) -> dict[str, object]:
    edge = bundle.edges[edge_index]
    return {
        "sourceId": edge.source_id,
        "targetId": edge.target_id,
        "edgeType": edge.edge_type,
        "weight": edge.weight,
    }


def _topology_hash(bundle: CoreGraphBundle, edge_indices: Sequence[int]) -> str:
    payloads = [_semantic_edge_payload(bundle, index) for index in edge_indices]
    return canonical_sha256(sorted(payloads, key=canonical_json))


@dataclass(frozen=True)
class TrainingSelection:
    fit_row_ids: tuple[str, ...]
    visible_edge_indices: tuple[int, ...]
    fit_row_ids_hash: str
    visible_topology_hash: str


def derive_training_selection(bundle: CoreGraphBundle) -> TrainingSelection:
    """Derive the only authoritative fit rows and visible topology from the bundle split."""

    assignments = bundle.split_manifest.assignments
    strategy = bundle.split_manifest.strategy
    if not assignments:
        if strategy in {"graph-disjoint", "leave-one-domain-out"}:
            raise ValueError("graph-disjoint split strategies are unsupported for one bundle")
        fit_row_ids = tuple(node.id for node in bundle.nodes)
        visible = tuple(range(len(bundle.edges)))
    else:
        roles = {assignment.entity_id: assignment.role for assignment in assignments}
        assignment_ids = set(roles)
        node_ids = {node.id for node in bundle.nodes}
        split_edge_ids = tuple(
            _split_edge_id(edge.source_id, edge.target_id) for edge in bundle.edges
        )
        edge_ids = set(split_edge_ids)
        edge_ids_unambiguous = len(edge_ids) == len(split_edge_ids)
        node_match = assignment_ids == node_ids
        edge_match = edge_ids_unambiguous and assignment_ids == edge_ids
        if strategy in {"graph-disjoint", "leave-one-domain-out"}:
            raise ValueError("graph-disjoint split strategies are unsupported for one bundle")
        if strategy in {
            "spanning-forest-80-10-10",
            "signed-pair-stratified-70-15-15",
        }:
            node_match = False
        if node_match == edge_match:
            raise ValueError("split assignments must cover one complete node or edge inventory")
        if node_match:
            fit_row_ids = tuple(
                sorted(identifier for identifier in node_ids if roles[identifier] == "train")
            )
            if strategy in {
                "stratified-node-70-15-15/1.0",
                "official-10-splits/1.0",
            }:
                visible = tuple(range(len(bundle.edges)))
            else:
                train_set = set(fit_row_ids)
                visible = tuple(
                    index
                    for index, edge in enumerate(bundle.edges)
                    if edge.source_id in train_set and edge.target_id in train_set
                )
        else:
            visible = tuple(
                index
                for index, identifier in enumerate(split_edge_ids)
                if roles[identifier] == "train"
            )
            fit_row_ids = tuple(
                sorted(
                    {
                        endpoint
                        for index in visible
                        for endpoint in (
                            bundle.edges[index].source_id,
                            bundle.edges[index].target_id,
                        )
                    }
                )
            )
    if not fit_row_ids:
        raise ValueError("adapter fitting requires at least one training row")
    return TrainingSelection(
        fit_row_ids=fit_row_ids,
        visible_edge_indices=visible,
        fit_row_ids_hash=canonical_sha256(list(fit_row_ids)),
        visible_topology_hash=_topology_hash(bundle, visible),
    )


def _normalization(values: Sequence[float]) -> tuple[float, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    scale = math.sqrt(variance)
    return mean, scale if scale > 0 else 1.0


def _visible_structure_rows(
    bundle: CoreGraphBundle, visible_edge_indices: Sequence[int] | None
) -> tuple[tuple[float, ...], ...]:
    if bundle.structural_features is None:
        return ()
    names = bundle.structural_features.names
    from .structure_features import (
        STRUCTURE_FEATURE_NAMES,
        StructureAlgorithmConfig,
        compute_structure_rows,
    )

    supported = set(STRUCTURE_FEATURE_NAMES)
    unsupported = set(names) - supported
    if unsupported:
        raise ValueError(f"unsupported visible-topology structure fields: {sorted(unsupported)}")
    selected = (
        tuple(range(len(bundle.edges)))
        if visible_edge_indices is None
        else tuple(visible_edge_indices)
    )
    fixed = compute_structure_rows(
        bundle,
        visible_edge_indices=selected,
        config=StructureAlgorithmConfig.fixed(),
    )
    column_by_name = {name: index for index, name in enumerate(STRUCTURE_FEATURE_NAMES)}
    return tuple(
        tuple(float(fixed[row, column_by_name[name]]) for name in names)
        for row in range(len(bundle.nodes))
    )


def fit_adapter_schema(
    bundle: CoreGraphBundle,
    *,
    train_row_ids: Sequence[str] | None = None,
    multi_hot_buckets: int = 256,
    visible_edge_indices: Sequence[int] | None = None,
) -> AdapterSchema:
    if multi_hot_buckets < 2 or multi_hot_buckets > 65_536:
        raise ValueError("multi-hot bucket count must be between 2 and 65536")
    selection = derive_training_selection(bundle)
    if train_row_ids is not None and tuple(train_row_ids) != selection.fit_row_ids:
        raise ValueError("training row IDs do not match the authoritative split")
    if (
        visible_edge_indices is not None
        and tuple(visible_edge_indices) != selection.visible_edge_indices
    ):
        raise ValueError("visible edge indices do not match the authoritative split")
    train_row_ids = selection.fit_row_ids
    visible_edge_indices = selection.visible_edge_indices
    row_by_id = {node.id: node.index for node in bundle.nodes}
    train_rows = tuple(row_by_id[node_id] for node_id in train_row_ids)
    fields: list[AdapterFieldSchema] = []
    for feature in bundle.node_features:
        if isinstance(feature, NumericFeature):
            mean, scale = _normalization(tuple(feature.values[row] for row in train_rows))
            fields.append(
                NumericFieldSchema(kind="numeric", name=feature.name, mean=mean, scale=scale)
            )
        elif isinstance(feature, CategoricalFeature):
            tokens = sorted(
                {
                    _typed_token(feature.values[row])
                    for row in train_rows
                    if feature.values[row] is not None
                }
            )
            if len(tokens) > MAX_VOCABULARY_ENTRIES:
                raise ValueError("categorical vocabulary exceeds the bounded adapter capacity")
            fields.append(
                CategoricalFieldSchema(
                    kind="categorical",
                    name=feature.name,
                    vocabulary=tuple(
                        VocabularyEntry(index=index, token=token)
                        for index, token in enumerate(tokens, start=1)
                    ),
                )
            )
        elif isinstance(feature, MultiHotFeature):
            token_digests = sorted(
                {
                    _typed_token_digest(feature.values[position])
                    for row in train_rows
                    for position in range(feature.row_offsets[row], feature.row_offsets[row + 1])
                }
            )
            if len(token_digests) > MAX_MULTIHOT_TRAINING_TOKENS:
                raise ValueError(
                    "multi-hot training vocabulary exceeds the bounded adapter capacity"
                )
            fields.append(
                MultiHotFieldSchema(
                    kind="multiHot",
                    name=feature.name,
                    hashAlgorithm="sha256-64-be-oov-zero/1.0",
                    bucketCount=multi_hot_buckets,
                    knownTokenDigests=tuple(token_digests),
                )
            )
    structure_rows = _visible_structure_rows(bundle, visible_edge_indices)
    if structure_rows:
        means_and_scales = tuple(
            _normalization(tuple(structure_rows[row][column] for row in train_rows))
            for column in range(len(structure_rows[0]))
        )
        fields.append(
            StructureFieldSchema(
                kind="structure",
                names=bundle.structural_features.names,  # type: ignore[union-attr]
                means=tuple(value[0] for value in means_and_scales),
                scales=tuple(value[1] for value in means_and_scales),
                algorithmVersion="socialgraph-fm.core-visible-topology-structure/1.0",
            )
        )
    raw = {
        "schemaVersion": "socialgraph-fm.core-adapter-schema/1.1",
        "sourceGraphVersionHash": bundle.graph_version_hash,
        "fitRowIdsHash": selection.fit_row_ids_hash,
        "fitRowCount": len(train_rows),
        "visibleTopologyHash": selection.visible_topology_hash,
        "visibleTopologyEdgeCount": len(selection.visible_edge_indices),
        "fields": tuple(field.model_dump(mode="python", by_alias=True) for field in fields),
    }
    raw["adapterSchemaHash"] = canonical_sha256(raw)
    return AdapterSchema.model_validate(raw)


class NumericAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, HIDDEN_DIM)
        self.decoder = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, values: Tensor) -> Tensor:
        return self.projection(values.to(dtype=self.projection.weight.dtype))

    def reconstruction_loss(self, latent: Tensor, targets: Tensor, selected: Tensor) -> Tensor:
        return functional.mse_loss(self.decoder(latent[selected]), targets[selected])


class CategoricalAdapter(nn.Module):
    def __init__(self, *, cardinality: int) -> None:
        super().__init__()
        if cardinality < 1:
            raise ValueError("categorical cardinality must be positive")
        if cardinality > MAX_VOCABULARY_ENTRIES + 1:
            raise ValueError("categorical cardinality exceeds the bounded categorical capacity")
        self.embedding = nn.Embedding(cardinality, HIDDEN_DIM)
        self.decoder = nn.Linear(HIDDEN_DIM, self.embedding.num_embeddings)

    def forward(self, values: Tensor) -> Tensor:
        return self.embedding(values)

    def reconstruction_loss(self, latent: Tensor, targets: Tensor, selected: Tensor) -> Tensor:
        labels = targets[selected]
        return functional.cross_entropy(self.decoder(latent[selected]), labels)


class SparseMultiHotAdapter(nn.Module):
    """Hash sparse identifiers into a bounded EmbeddingBag table.

    The caller supplies CSR-style indices and offsets. No vocabulary-by-node or
    node-by-vocabulary tensor is ever created.
    """

    def __init__(self, *, bucket_count: int = 4096) -> None:
        super().__init__()
        if bucket_count < 2:
            raise ValueError("multi-hot bucket count must reserve bucket zero for OOV")
        self.embedding = nn.EmbeddingBag(
            bucket_count,
            HIDDEN_DIM,
            mode="mean",
            include_last_offset=True,
        )

    def forward(self, *, indices: Tensor, offsets: Tensor) -> Tensor:
        hashed = torch.remainder(indices, self.embedding.num_embeddings)
        return self.embedding(hashed, offsets)

    def sample_negative_buckets(
        self,
        *,
        positive_rows: Tensor,
        positive_ids: Tensor,
        selected_rows: Tensor,
        budget_per_row: int,
        generator: torch.Generator,
    ) -> tuple[Tensor, Tensor]:
        if budget_per_row < 0:
            raise ValueError("multi-hot negative budget must be nonnegative")
        if selected_rows.numel() == 0 or budget_per_row == 0:
            empty = selected_rows.new_empty((0,))
            return empty, empty
        bucket_count = self.embedding.num_embeddings
        selected_count = selected_rows.shape[0]
        mapped_positions = torch.searchsorted(selected_rows, positive_rows)
        in_range = mapped_positions < selected_count
        clamped = mapped_positions.clamp(max=selected_count - 1)
        selected_positive = in_range & (selected_rows[clamped] == positive_rows)
        positive_matrix = torch.zeros(
            (selected_count, bucket_count), dtype=torch.bool, device=selected_rows.device
        )
        # Bucket zero is the permanently reserved OOV sentinel and is never a
        # learned positive or sampled negative multi-hot identifier.
        positive_matrix[:, 0] = True
        mapped_rows = mapped_positions[selected_positive]
        positive_matrix[mapped_rows, positive_ids[selected_positive]] = True
        complement_count = (~positive_matrix).sum(dim=1)
        take_count = complement_count.clamp(max=budget_per_row)
        scores = torch.rand(
            (selected_count, bucket_count),
            generator=generator,
            device=selected_rows.device,
        ).masked_fill(positive_matrix, float("inf"))
        candidates = torch.argsort(scores, dim=1)[:, : min(budget_per_row, bucket_count)]
        ranks = torch.arange(candidates.shape[1], device=selected_rows.device).unsqueeze(0)
        retained = ranks < take_count.unsqueeze(1)
        negative_rows = selected_rows.unsqueeze(1).expand_as(candidates)[retained]
        return negative_rows, candidates[retained]

    def reconstruction_loss(
        self,
        *,
        latent: Tensor,
        positive_rows: Tensor,
        positive_ids: Tensor,
        selected_rows: Tensor,
        budget_per_row: int,
        generator: torch.Generator,
    ) -> Tensor:
        bucket_count = self.embedding.num_embeddings
        positive_keys = torch.unique(
            positive_rows * bucket_count + positive_ids,
            sorted=True,
        )
        positive_rows = torch.div(positive_keys, bucket_count, rounding_mode="floor")
        positive_ids = positive_keys % bucket_count
        negative_rows, negative_ids = self.sample_negative_buckets(
            positive_rows=positive_rows,
            positive_ids=positive_ids,
            selected_rows=selected_rows,
            budget_per_row=budget_per_row,
            generator=generator,
        )
        scale = HIDDEN_DIM**0.5
        positive_logits = (latent[positive_rows] * self.embedding.weight[positive_ids]).sum(
            dim=1
        ) / scale
        negative_logits = (latent[negative_rows] * self.embedding.weight[negative_ids]).sum(
            dim=1
        ) / scale
        logits = torch.cat((positive_logits, negative_logits))
        labels = torch.cat((torch.ones_like(positive_logits), torch.zeros_like(negative_logits)))
        if logits.numel() == 0:
            return latent.sum() * 0.0
        return functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum") / (
            logits.numel()
        )


class StructureViewAdapter(nn.Module):
    def __init__(self, *, input_width: int) -> None:
        super().__init__()
        if input_width < 1:
            raise ValueError("structure-view width must be positive")
        self.projection = nn.Linear(input_width, HIDDEN_DIM)
        self.decoder = nn.Linear(HIDDEN_DIM, input_width)

    def forward(self, values: Tensor) -> Tensor:
        return self.projection(values.to(dtype=self.projection.weight.dtype))

    def reconstruction_loss(self, latent: Tensor, targets: Tensor, selected: Tensor) -> Tensor:
        return functional.mse_loss(self.decoder(latent[selected]), targets[selected])


class AdapterParameterModule(nn.Module):
    """Learned adapter parameters reconstructed from a bounded fitted schema only."""

    def __init__(self, schema: AdapterSchema) -> None:
        super().__init__()
        modules: dict[str, nn.Module] = {}
        for index, field in enumerate(schema.fields):
            if isinstance(field, NumericFieldSchema):
                module: nn.Module = NumericAdapter()
            elif isinstance(field, CategoricalFieldSchema):
                module = CategoricalAdapter(cardinality=len(field.vocabulary) + 1)
            elif isinstance(field, MultiHotFieldSchema):
                module = SparseMultiHotAdapter(bucket_count=field.bucket_count)
            elif isinstance(field, StructureFieldSchema):
                module = StructureViewAdapter(input_width=len(field.names))
            else:  # pragma: no cover - discriminated union is closed
                raise TypeError("unsupported adapter schema field")
            modules[f"field_{index}"] = module
        self.adapters = nn.ModuleDict(modules)


class BundleInputAdapter(nn.Module):
    """Tensorize one validated bundle schema once and adapt all fields jointly."""

    def __init__(
        self,
        bundle: CoreGraphBundle,
        *,
        mode: Literal["training", "inference"],
        multi_hot_buckets: int = 256,
        schema: AdapterSchema | None = None,
        train_row_ids: Sequence[str] | None = None,
        visible_edge_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        selection: TrainingSelection | None = None
        if mode == "training":
            selection = derive_training_selection(bundle)
            if train_row_ids is not None and tuple(train_row_ids) != selection.fit_row_ids:
                raise ValueError("training row IDs do not match the authoritative split")
            if (
                visible_edge_indices is not None
                and tuple(visible_edge_indices) != selection.visible_edge_indices
            ):
                raise ValueError("visible edge indices do not match the authoritative split")
            visible_edge_indices = selection.visible_edge_indices
        elif train_row_ids is not None or visible_edge_indices is not None:
            raise ValueError("inference mode does not accept training topology overrides")
        if schema is None:
            if mode != "training":
                raise ValueError("adapter fitting is allowed only in training mode")
            schema = fit_adapter_schema(
                bundle,
                multi_hot_buckets=multi_hot_buckets,
            )
        elif mode == "training":
            if selection is None:  # pragma: no cover - established above
                raise RuntimeError("training selection is unavailable")
            if (
                schema.source_graph_version_hash != bundle.graph_version_hash
                or schema.fit_row_ids_hash != selection.fit_row_ids_hash
                or schema.fit_row_count != len(selection.fit_row_ids)
                or schema.visible_topology_hash != selection.visible_topology_hash
                or schema.visible_topology_edge_count != len(selection.visible_edge_indices)
            ):
                raise ValueError("adapter schema does not match source training provenance")
        self.schema = schema
        self.graph_version_hash = bundle.graph_version_hash
        self.num_nodes = len(bundle.nodes)
        self.adapters = nn.ModuleDict()
        self._kinds: list[str] = []
        field_names: list[str] = []
        expected_field_count = len(bundle.node_features) + int(
            bundle.structural_features is not None and bool(bundle.structural_features.names)
        )
        if len(schema.fields) != expected_field_count:
            raise ValueError("target raw feature field count does not match adapter schema")
        for index, feature in enumerate(bundle.node_features):
            key = f"field_{index}"
            field_names.append(feature.name)
            field_schema = schema.fields[index]
            if isinstance(feature, NumericFeature):
                if (
                    not isinstance(field_schema, NumericFieldSchema)
                    or field_schema.name != feature.name
                ):
                    raise ValueError(
                        "target numeric feature contract does not match adapter schema"
                    )
                self.adapters[key] = NumericAdapter()
                normalized = [
                    (value - field_schema.mean) / field_schema.scale for value in feature.values
                ]
                self.register_buffer(
                    f"_{key}_values",
                    torch.tensor(normalized, dtype=torch.float32).view(-1, 1),
                    persistent=False,
                )
                self._kinds.append("numeric")
            elif isinstance(feature, CategoricalFeature):
                if (
                    not isinstance(field_schema, CategoricalFieldSchema)
                    or field_schema.name != feature.name
                ):
                    raise ValueError(
                        "target categorical feature contract does not match adapter schema"
                    )
                vocabulary = {entry.token: entry.index for entry in field_schema.vocabulary}
                encoded = [
                    0 if value is None else vocabulary.get(_typed_token(value), 0)
                    for value in feature.values
                ]
                self.adapters[key] = CategoricalAdapter(cardinality=len(vocabulary) + 1)
                self.register_buffer(
                    f"_{key}_values",
                    torch.tensor(encoded, dtype=torch.long),
                    persistent=False,
                )
                self._kinds.append("categorical")
            elif isinstance(feature, MultiHotFeature):
                if (
                    not isinstance(field_schema, MultiHotFieldSchema)
                    or field_schema.name != feature.name
                ):
                    raise ValueError(
                        "target multi-hot feature contract does not match adapter schema"
                    )
                bucket_count = field_schema.bucket_count
                known = set(field_schema.known_token_digests)
                hashed = []
                for value in feature.values:
                    digest = _typed_token_digest(value)
                    hashed.append(
                        0 if digest not in known else 1 + int(digest[:16], 16) % (bucket_count - 1)
                    )
                self.adapters[key] = SparseMultiHotAdapter(bucket_count=bucket_count)
                self.register_buffer(
                    f"_{key}_indices", torch.tensor(hashed, dtype=torch.long), persistent=False
                )
                self.register_buffer(
                    f"_{key}_offsets",
                    torch.tensor(feature.row_offsets, dtype=torch.long),
                    persistent=False,
                )
                row_lengths = torch.diff(torch.tensor(feature.row_offsets, dtype=torch.long))
                self.register_buffer(
                    f"_{key}_row_ids",
                    torch.repeat_interleave(torch.arange(self.num_nodes), row_lengths),
                    persistent=False,
                )
                self._kinds.append("multiHot")
            else:  # pragma: no cover - the validated discriminated union is exhaustive
                raise TypeError("unsupported node feature schema")
        if bundle.structural_features is not None and bundle.structural_features.names:
            key = f"field_{len(self._kinds)}"
            field_names.append("structure-view")
            field_schema = schema.fields[len(self._kinds)]
            if (
                not isinstance(field_schema, StructureFieldSchema)
                or field_schema.names != bundle.structural_features.names
            ):
                raise ValueError("target structure feature contract does not match adapter schema")
            self.adapters[key] = StructureViewAdapter(
                input_width=len(bundle.structural_features.names)
            )
            raw_structure = _visible_structure_rows(bundle, visible_edge_indices)
            normalized_structure = tuple(
                tuple(
                    (value - field_schema.means[column]) / field_schema.scales[column]
                    for column, value in enumerate(row)
                )
                for row in raw_structure
            )
            self.register_buffer(
                f"_{key}_values",
                torch.tensor(normalized_structure, dtype=torch.float32),
                persistent=False,
            )
            self._kinds.append("structure")
        if not self._kinds:
            raise ValueError("bundle adapter requires at least one node or structure field")
        self.field_names = tuple(field_names)

    def _sparse_subset(self, key: str, node_ids: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        global_rows = getattr(self, f"_{key}_row_ids")
        global_indices = getattr(self, f"_{key}_indices")
        mapping = torch.full((self.num_nodes,), -1, dtype=torch.long, device=node_ids.device)
        mapping[node_ids] = torch.arange(node_ids.shape[0], device=node_ids.device)
        local_rows = mapping[global_rows]
        retained = local_rows >= 0
        local_rows = local_rows[retained]
        indices = global_indices[retained]
        order = torch.argsort(local_rows)
        local_rows, indices = local_rows[order], indices[order]
        counts = torch.bincount(local_rows, minlength=node_ids.shape[0])
        offsets = torch.cat((counts.new_zeros(1), torch.cumsum(counts, dim=0)))
        return indices, offsets, local_rows

    def field_outputs(self, node_ids: Tensor | None = None) -> Tensor:
        adapted: list[Tensor] = []
        for index, kind in enumerate(self._kinds):
            key = f"field_{index}"
            adapter = self.adapters[key]
            if kind == "multiHot":
                if node_ids is None:
                    indices = getattr(self, f"_{key}_indices")
                    offsets = getattr(self, f"_{key}_offsets")
                else:
                    indices, offsets, _ = self._sparse_subset(key, node_ids)
                output = adapter(  # type: ignore[call-arg]
                    indices=indices,
                    offsets=offsets,
                )
            else:
                values = getattr(self, f"_{key}_values")
                output = adapter(values if node_ids is None else values[node_ids])
            adapted.append(output)
        return torch.stack(adapted, dim=1)

    def forward(
        self, field_mask: Tensor | None = None, *, node_ids: Tensor | None = None
    ) -> Tensor:
        stacked = self.field_outputs(node_ids)
        if field_mask is None:
            return stacked.mean(dim=1)
        if field_mask.shape != stacked.shape[:2] or field_mask.dtype != torch.bool:
            raise ValueError("field mask must be boolean [num_nodes, num_fields]")
        active = (~field_mask).unsqueeze(-1)
        return (stacked * active).sum(dim=1) / active.sum(dim=1).clamp_min(1)

    def reconstruction_loss(
        self,
        decoded_fields: Tensor,
        field_mask: Tensor,
        *,
        generator: torch.Generator,
        node_ids: Tensor | None = None,
    ) -> Tensor:
        row_count = self.num_nodes if node_ids is None else node_ids.shape[0]
        if decoded_fields.shape != (row_count, len(self._kinds), HIDDEN_DIM):
            raise ValueError("decoded fields do not match bundle schema")
        if field_mask.shape != decoded_fields.shape[:2] or field_mask.dtype != torch.bool:
            raise ValueError("field mask must be boolean [num_nodes, num_fields]")
        losses: list[Tensor] = []
        for index, kind in enumerate(self._kinds):
            selected = field_mask[:, index]
            if not bool(torch.any(selected)):
                continue
            key = f"field_{index}"
            adapter = self.adapters[key]
            latent = decoded_fields[:, index]
            if kind in {"numeric", "categorical", "structure"}:
                targets = getattr(self, f"_{key}_values")
                if node_ids is not None:
                    targets = targets[node_ids]
                if isinstance(adapter, (NumericAdapter, CategoricalAdapter, StructureViewAdapter)):
                    losses.append(adapter.reconstruction_loss(latent, targets, selected))
                else:  # pragma: no cover - kind/module pairs are fixed by construction
                    raise RuntimeError("field adapter kind mismatch")
                continue
            if node_ids is None:
                indices = getattr(self, f"_{key}_indices")
                row_ids = getattr(self, f"_{key}_row_ids")
            else:
                indices, _, row_ids = self._sparse_subset(key, node_ids)
            positive_selected = selected[row_ids]
            positive_selected = positive_selected & (indices != 0)
            positive_rows = row_ids[positive_selected]
            positive_ids = indices[positive_selected]
            if not isinstance(adapter, SparseMultiHotAdapter):
                raise RuntimeError("multi-hot adapter kind mismatch")
            losses.append(
                adapter.reconstruction_loss(
                    latent=latent,
                    positive_rows=positive_rows,
                    positive_ids=positive_ids,
                    selected_rows=torch.nonzero(selected, as_tuple=False).flatten(),
                    budget_per_row=1,
                    generator=generator,
                )
            )
        if not losses:
            return decoded_fields.sum() * 0.0
        return torch.stack(losses).mean()


__all__ = [
    "AdapterSchema",
    "AdapterParameterModule",
    "BundleInputAdapter",
    "CategoricalAdapter",
    "HIDDEN_DIM",
    "NumericAdapter",
    "SparseMultiHotAdapter",
    "StructureViewAdapter",
    "TrainingSelection",
    "derive_training_selection",
    "fit_adapter_schema",
]
