"""Fail-closed, live fold evaluation for authoritative core experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .adapters import BundleInputAdapter, derive_training_selection
from .bundle import CoreGraphBundle, calculate_graph_version_hash
from .formal_preflight import ExperimentLabels, ExperimentSplitFold
from .model import CoreGFM
from .structure_features import (
    STRUCTURE_FEATURE_NAMES,
    StructureAlgorithmConfig,
    compute_structure_rows,
)
from .supervised import derive_encoder_identity, encode_supervised_graph


FoldTaskKind = Literal[
    "node-binary",
    "signed-edge",
    "resilience-regression",
    "edge-binary",
]
FoldAssignmentKind = Literal["node", "edge"]
FoldHeadName = Literal["node_head", "binary_link_head", "signed_edge_head", "resilience_head"]

_HASH = r"^[0-9a-f]{64}$"
_LINK_CANDIDATE_PROTOCOL = "socialgraph-fm.core-complete-filtered-endpoint-corruption/1.0"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


class FoldPreparationRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-fold-preparation/1.0"] = Field(
        alias="schemaVersion"
    )
    fold_id: str = Field(alias="foldId", min_length=1)
    assignment_kind: FoldAssignmentKind = Field(alias="assignmentKind")
    base_graph_version_hash: str = Field(alias="baseGraphVersionHash", pattern=_HASH)
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)
    prepared_graph_version_hash: str = Field(alias="preparedGraphVersionHash", pattern=_HASH)
    visible_topology_hash: str = Field(alias="visibleTopologyHash", pattern=_HASH)
    structure_rows_hash: str = Field(alias="structureRowsHash", pattern=_HASH)
    test_entity_ids: tuple[str, ...] = Field(alias="testEntityIds", strict=False, min_length=1)
    preparation_hash: str = Field(alias="preparationHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_record(self):
        if self.test_entity_ids != tuple(sorted(set(self.test_entity_ids))):
            raise ValueError("fold test entity IDs must be unique and sorted")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"preparation_hash"})
        )
        if self.preparation_hash != expected:
            raise ValueError("preparationHash does not match fold preparation")
        return self


class AuthoritativeFoldTestRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-authoritative-fold-test/1.0"] = Field(
        alias="schemaVersion"
    )
    fold_id: str = Field(alias="foldId", min_length=1)
    task_kind: FoldTaskKind = Field(alias="taskKind")
    target_name: str = Field(alias="targetName", min_length=1)
    requirement_id: str = Field(alias="requirementId", min_length=1)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)
    preparation_hash: str = Field(alias="preparationHash", pattern=_HASH)
    labels_hash: str = Field(alias="labelsHash", pattern=_HASH)
    entity_ids: tuple[str, ...] = Field(alias="entityIds", strict=False, min_length=1)
    node_indices: tuple[int, ...] = Field(default=(), alias="nodeIndices", strict=False)
    edge_pairs: tuple[tuple[int, int], ...] = Field(default=(), alias="edgePairs", strict=False)
    targets: tuple[int | float, ...] = Field(strict=False, min_length=1)
    binding_hash: str = Field(alias="bindingHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_record(self):
        count = len(self.entity_ids)
        if self.entity_ids != tuple(sorted(set(self.entity_ids))):
            raise ValueError("bound fold entity IDs must be unique and sorted")
        if bool(self.node_indices) == bool(self.edge_pairs):
            raise ValueError("bound fold requires exactly one locator kind")
        locators = self.node_indices if self.node_indices else self.edge_pairs
        if len(locators) != count or len(self.targets) != count:
            raise ValueError("bound fold entities, locators, and targets must align")
        if len(locators) != len(set(locators)):
            raise ValueError("bound fold locators must be unique")
        if self.node_indices and any(index < 0 for index in self.node_indices):
            raise ValueError("bound node locators must be nonnegative")
        if self.edge_pairs and any(
            left < 0 or right < 0 or left == right for left, right in self.edge_pairs
        ):
            raise ValueError("bound edge locators require distinct nonnegative endpoints")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"binding_hash"})
        )
        if self.binding_hash != expected:
            raise ValueError("bindingHash does not match authoritative fold test")
        return self


class LinkCandidateScores(_StrictModel):
    query_entity_id: str = Field(alias="queryEntityId", min_length=1)
    positive_endpoint_pair: tuple[str, str] = Field(alias="positiveEndpointPair", strict=False)
    positive_score: float = Field(alias="positiveScore")
    negative_endpoint_pairs: tuple[tuple[str, str], ...] = Field(
        alias="negativeEndpointPairs", strict=False, min_length=1
    )
    negative_scores: tuple[float, ...] = Field(alias="negativeScores", strict=False, min_length=1)
    candidate_inventory_hash: str = Field(alias="candidateInventoryHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_candidates(self):
        if (
            self.positive_endpoint_pair[0] == self.positive_endpoint_pair[1]
            or self.negative_endpoint_pairs != tuple(sorted(set(self.negative_endpoint_pairs)))
            or any(left == right for left, right in self.negative_endpoint_pairs)
        ):
            raise ValueError("link endpoint pairs must be distinct, unique, and sorted")
        if len(self.negative_endpoint_pairs) != len(self.negative_scores):
            raise ValueError("link negative endpoint pairs and scores must align")
        if not all(math.isfinite(value) for value in (self.positive_score, *self.negative_scores)):
            raise ValueError("link candidate scores must be finite")
        expected_hash = canonical_sha256(
            {
                "queryEntityId": self.query_entity_id,
                "positiveEndpointPair": self.positive_endpoint_pair,
                "negativeEndpointPairs": self.negative_endpoint_pairs,
            }
        )
        if self.candidate_inventory_hash != expected_hash:
            raise ValueError("candidateInventoryHash does not match filtered endpoint candidates")
        return self


class FoldPredictionRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-live-fold-predictions/1.0"] = Field(
        alias="schemaVersion"
    )
    fold_id: str = Field(alias="foldId", min_length=1)
    task_kind: FoldTaskKind = Field(alias="taskKind")
    head_name: FoldHeadName = Field(alias="headName")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    binding_hash: str = Field(alias="bindingHash", pattern=_HASH)
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=_HASH)
    head_state_hash: str = Field(alias="headStateHash", pattern=_HASH)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH)
    entity_ids: tuple[str, ...] = Field(alias="entityIds", strict=False, min_length=1)
    targets: tuple[int | float, ...] = Field(strict=False, min_length=1)
    scores: tuple[float, ...] = Field(strict=False, min_length=1)
    probabilities: tuple[float, ...] = Field(strict=False)
    candidate_protocol: str | None = Field(default=None, alias="candidateProtocol")
    endpoint_ids: tuple[str, ...] = Field(default=(), alias="endpointIds", strict=False)
    endpoint_inventory_hash: str | None = Field(
        default=None, alias="endpointInventoryHash", pattern=_HASH
    )
    known_positive_pairs_hash: str | None = Field(
        default=None, alias="knownPositivePairsHash", pattern=_HASH
    )
    link_candidates: tuple[LinkCandidateScores, ...] = Field(
        default=(), alias="linkCandidates", strict=False
    )
    prediction_hash: str = Field(alias="predictionHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_record(self):
        count = len(self.entity_ids)
        if (
            self.entity_ids != tuple(sorted(set(self.entity_ids)))
            or len(self.targets) != count
            or len(self.scores) != count
        ):
            raise ValueError("prediction entities, targets, and scores must align")
        if not all(math.isfinite(value) for value in self.scores):
            raise ValueError("fold prediction scores must be finite")
        if self.task_kind == "resilience-regression":
            if self.probabilities:
                raise ValueError("regression predictions must not contain probabilities")
        elif len(self.probabilities) != count or not all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in self.probabilities
        ):
            raise ValueError("classification probabilities must be finite unit values")
        is_link = self.task_kind == "edge-binary"
        link_metadata = (
            self.candidate_protocol is not None
            and bool(self.endpoint_ids)
            and self.endpoint_inventory_hash is not None
            and self.known_positive_pairs_hash is not None
            and bool(self.link_candidates)
        )
        if is_link != link_metadata:
            raise ValueError("complete link candidate metadata is required only for edge-binary")
        if is_link:
            if self.candidate_protocol != _LINK_CANDIDATE_PROTOCOL:
                raise ValueError("unknown link candidate enumeration protocol")
            if self.endpoint_ids != tuple(sorted(set(self.endpoint_ids))):
                raise ValueError("link endpoint inventory must be unique and sorted")
            if self.endpoint_inventory_hash != canonical_sha256(list(self.endpoint_ids)):
                raise ValueError("endpointInventoryHash does not match endpoint IDs")
            if tuple(item.query_entity_id for item in self.link_candidates) != self.entity_ids:
                raise ValueError("link candidates must align with positive test entities")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"prediction_hash"})
        )
        if self.prediction_hash != expected:
            raise ValueError("predictionHash does not match live fold predictions")
        return self


_PREPARED_SEAL = object()
_BOUND_SEAL = object()
_PREDICTION_SEAL = object()


@dataclass(frozen=True, init=False)
class PreparedAuthoritativeFold:
    base_bundle: CoreGraphBundle
    fold: ExperimentSplitFold
    bundle: CoreGraphBundle
    record: FoldPreparationRecord
    _factory_seal: object

    def __new__(cls, *args: object, **kwargs: object):
        del args, kwargs
        raise TypeError("PreparedAuthoritativeFold is emitted only by fold preparation")


@dataclass(frozen=True, init=False)
class BoundAuthoritativeFoldTest:
    prepared: PreparedAuthoritativeFold
    labels: ExperimentLabels
    record: AuthoritativeFoldTestRecord
    _factory_seal: object

    def __new__(cls, *args: object, **kwargs: object):
        del args, kwargs
        raise TypeError("BoundAuthoritativeFoldTest is emitted only by label binding")


@dataclass(frozen=True, init=False)
class VerifiedFoldPredictions:
    record: FoldPredictionRecord
    _sealed_prediction_hash: str
    _factory_seal: object

    def __new__(cls, *args: object, **kwargs: object):
        del args, kwargs
        raise TypeError("VerifiedFoldPredictions is emitted only by live inference")


def _new_prepared(
    *,
    base_bundle: CoreGraphBundle,
    fold: ExperimentSplitFold,
    bundle: CoreGraphBundle,
    record: FoldPreparationRecord,
) -> PreparedAuthoritativeFold:
    prepared = object.__new__(PreparedAuthoritativeFold)
    object.__setattr__(prepared, "base_bundle", base_bundle)
    object.__setattr__(prepared, "fold", fold)
    object.__setattr__(prepared, "bundle", bundle)
    object.__setattr__(prepared, "record", record)
    object.__setattr__(prepared, "_factory_seal", _PREPARED_SEAL)
    return prepared


def _new_bound(
    *,
    prepared: PreparedAuthoritativeFold,
    labels: ExperimentLabels,
    record: AuthoritativeFoldTestRecord,
) -> BoundAuthoritativeFoldTest:
    bound = object.__new__(BoundAuthoritativeFoldTest)
    object.__setattr__(bound, "prepared", prepared)
    object.__setattr__(bound, "labels", labels)
    object.__setattr__(bound, "record", record)
    object.__setattr__(bound, "_factory_seal", _BOUND_SEAL)
    return bound


def _new_predictions(record: FoldPredictionRecord) -> VerifiedFoldPredictions:
    predictions = object.__new__(VerifiedFoldPredictions)
    object.__setattr__(predictions, "record", record)
    object.__setattr__(predictions, "_sealed_prediction_hash", record.prediction_hash)
    object.__setattr__(predictions, "_factory_seal", _PREDICTION_SEAL)
    return predictions


def _assignment_inventory(
    bundle: CoreGraphBundle, fold: ExperimentSplitFold
) -> tuple[FoldAssignmentKind, tuple[str, ...]]:
    assignments = fold.split_manifest.assignments
    assignment_ids = {assignment.entity_id for assignment in assignments}
    node_ids = {node.id for node in bundle.nodes}
    edge_id_sequence = tuple(f"edge:{edge.source_id}:{edge.target_id}" for edge in bundle.edges)
    edge_ids = set(edge_id_sequence)
    node_match = assignment_ids == node_ids
    edge_match = len(edge_ids) == len(edge_id_sequence) and assignment_ids == edge_ids
    if node_match == edge_match:
        raise ValueError("authoritative fold must cover one exact node or edge inventory")
    role_counts = {
        role: sum(assignment.role == role for assignment in assignments)
        for role in ("train", "validation", "test")
    }
    if any(count == 0 for count in role_counts.values()):
        raise ValueError("authoritative fold requires nonempty train, validation, and test roles")
    kind: FoldAssignmentKind = "node" if node_match else "edge"
    test_ids = tuple(
        sorted(assignment.entity_id for assignment in assignments if assignment.role == "test")
    )
    return kind, test_ids


def _bundle_with_fold_and_structure(
    base_bundle: CoreGraphBundle, fold: ExperimentSplitFold
) -> tuple[CoreGraphBundle, FoldPreparationRecord]:
    assignment_kind, test_ids = _assignment_inventory(base_bundle, fold)
    raw = base_bundle.model_dump(mode="python", by_alias=True)
    raw["splitManifest"] = fold.split_manifest.model_dump(mode="python", by_alias=True)
    raw["structuralFeatures"] = None
    raw["graphVersionHash"] = calculate_graph_version_hash(raw)
    topology_bundle = CoreGraphBundle.model_validate(raw)
    selection = derive_training_selection(topology_bundle)
    rows = compute_structure_rows(
        topology_bundle,
        visible_edge_indices=selection.visible_edge_indices,
        config=StructureAlgorithmConfig.fixed(),
    )
    raw["structuralFeatures"] = {
        "names": STRUCTURE_FEATURE_NAMES,
        "values": tuple(tuple(float(value) for value in row) for row in rows),
    }
    raw["graphVersionHash"] = calculate_graph_version_hash(raw)
    prepared_bundle = CoreGraphBundle.model_validate(raw)
    prepared_selection = derive_training_selection(prepared_bundle)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-fold-preparation/1.0",
        "foldId": fold.fold_id,
        "assignmentKind": assignment_kind,
        "baseGraphVersionHash": base_bundle.graph_version_hash,
        "splitManifestHash": fold.split_manifest_hash,
        "preparedGraphVersionHash": prepared_bundle.graph_version_hash,
        "visibleTopologyHash": prepared_selection.visible_topology_hash,
        "structureRowsHash": canonical_sha256(prepared_bundle.structural_features),
        "testEntityIds": test_ids,
    }
    payload["preparationHash"] = canonical_sha256(payload)
    return prepared_bundle, FoldPreparationRecord.model_validate(payload)


def prepare_authoritative_fold(
    base_bundle: CoreGraphBundle, fold: ExperimentSplitFold
) -> PreparedAuthoritativeFold:
    """Replace one bundle split and recompute its train-visible structural view."""

    if type(base_bundle) is not CoreGraphBundle or type(fold) is not ExperimentSplitFold:
        raise TypeError("fold preparation requires exact validated bundle and fold types")
    prepared_bundle, record = _bundle_with_fold_and_structure(base_bundle, fold)
    return _new_prepared(
        base_bundle=base_bundle,
        fold=fold,
        bundle=prepared_bundle,
        record=record,
    )


def _verify_prepared(prepared: PreparedAuthoritativeFold) -> None:
    if type(prepared) is not PreparedAuthoritativeFold:
        raise TypeError("fold binding requires exact PreparedAuthoritativeFold evidence")
    if prepared._factory_seal is not _PREPARED_SEAL:
        raise ValueError("prepared fold runtime seal changed")
    expected_bundle, expected_record = _bundle_with_fold_and_structure(
        prepared.base_bundle, prepared.fold
    )
    if prepared.bundle != expected_bundle or prepared.record != expected_record:
        raise ValueError("prepared fold does not match its authoritative live derivation")


def _edge_locator_inventory(
    bundle: CoreGraphBundle,
) -> dict[str, tuple[int, int]]:
    node_index = {node.id: node.index for node in bundle.nodes}
    inventory: dict[str, tuple[int, int]] = {}
    for edge in bundle.edges:
        identifier = f"edge:{edge.source_id}:{edge.target_id}"
        if identifier in inventory:
            raise ValueError("edge stable IDs are ambiguous in the fold graph")
        inventory[identifier] = (
            node_index[edge.source_id],
            node_index[edge.target_id],
        )
    return inventory


def _normalize_targets(
    task_kind: FoldTaskKind, values: tuple[int | float | str, ...]
) -> tuple[int | float, ...]:
    if task_kind == "node-binary":
        if any(type(value) is not int or value not in {0, 1} for value in values):
            raise ValueError("node-binary labels must be integer zero or one")
        return tuple(int(value) for value in values)
    if task_kind == "edge-binary":
        if any(type(value) is not int or value != 1 for value in values):
            raise ValueError("edge completion split labels must identify known positive edges")
        return tuple(1 for _value in values)
    if task_kind == "signed-edge":
        if any(type(value) is not int or value not in {-1, 1} for value in values):
            raise ValueError("signed-edge labels must use the authoritative -1/+1 signs")
        return tuple(1 if value == 1 else 0 for value in values)
    normalized: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("resilience labels must be finite numeric scalars")
        normalized.append(float(value))
    return tuple(normalized)


def _derive_test_record(
    prepared: PreparedAuthoritativeFold,
    labels: ExperimentLabels,
    *,
    target_name: str,
    task_kind: FoldTaskKind,
) -> AuthoritativeFoldTestRecord:
    _verify_prepared(prepared)
    if type(labels) is not ExperimentLabels:
        raise TypeError("fold label binding requires exact ExperimentLabels evidence")
    if not target_name:
        raise ValueError("fold target name must be nonempty")
    expected_kind: FoldAssignmentKind = (
        "edge" if task_kind in {"signed-edge", "edge-binary"} else "node"
    )
    if prepared.record.assignment_kind != expected_kind:
        raise ValueError("task locator kind does not match authoritative fold assignments")
    matches = tuple(target for target in labels.targets if target.name == target_name)
    if len(matches) != 1:
        raise ValueError("authoritative labels must contain the exact requested target")
    target = matches[0]
    label_by_id = {item.entity_id: item.value for item in target.values}
    assignment_ids = {
        assignment.entity_id for assignment in prepared.bundle.split_manifest.assignments
    }
    if set(label_by_id) != assignment_ids:
        raise ValueError("label entities must equal the exact authoritative assignment inventory")
    entity_ids = prepared.record.test_entity_ids
    raw_values = tuple(label_by_id[identifier] for identifier in entity_ids)
    targets = _normalize_targets(task_kind, raw_values)
    node_indices: tuple[int, ...] = ()
    edge_pairs: tuple[tuple[int, int], ...] = ()
    if expected_kind == "node":
        by_id = {node.id: node.index for node in prepared.bundle.nodes}
        node_indices = tuple(by_id[identifier] for identifier in entity_ids)
    else:
        edge_inventory = _edge_locator_inventory(prepared.bundle)
        edge_pairs = tuple(edge_inventory[identifier] for identifier in entity_ids)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-authoritative-fold-test/1.0",
        "foldId": prepared.fold.fold_id,
        "taskKind": task_kind,
        "targetName": target_name,
        "requirementId": labels.requirement_id,
        "graphVersionHash": prepared.bundle.graph_version_hash,
        "splitManifestHash": prepared.fold.split_manifest_hash,
        "preparationHash": prepared.record.preparation_hash,
        "labelsHash": labels.labels_hash,
        "entityIds": entity_ids,
        "nodeIndices": node_indices,
        "edgePairs": edge_pairs,
        "targets": targets,
    }
    payload["bindingHash"] = canonical_sha256(payload)
    return AuthoritativeFoldTestRecord.model_validate(payload)


def bind_authoritative_fold_test(
    prepared: PreparedAuthoritativeFold,
    labels: ExperimentLabels,
    *,
    target_name: str,
    task_kind: FoldTaskKind,
) -> BoundAuthoritativeFoldTest:
    """Bind exact fold test roles to the complete, separately stored target inventory."""

    record = _derive_test_record(prepared, labels, target_name=target_name, task_kind=task_kind)
    return _new_bound(prepared=prepared, labels=labels, record=record)


def _verify_bound(bound: BoundAuthoritativeFoldTest) -> None:
    if type(bound) is not BoundAuthoritativeFoldTest:
        raise TypeError("fold inference requires exact BoundAuthoritativeFoldTest evidence")
    if bound._factory_seal is not _BOUND_SEAL:
        raise ValueError("authoritative fold test runtime seal changed")
    expected = _derive_test_record(
        bound.prepared,
        bound.labels,
        target_name=bound.record.target_name,
        task_kind=bound.record.task_kind,
    )
    if bound.record != expected:
        raise ValueError("authoritative fold test binding changed")


def _state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {
            name: canonical_tensor_digest(value)
            for name, value in sorted(module.state_dict().items())
        }
    )


def _head_for_task(task_kind: FoldTaskKind) -> FoldHeadName:
    return {
        "node-binary": "node_head",
        "edge-binary": "binary_link_head",
        "signed-edge": "signed_edge_head",
        "resilience-regression": "resilience_head",
    }[task_kind]  # type: ignore[return-value]


def _tensor_values(tensor: Tensor) -> tuple[float, ...]:
    values = tuple(float(value) for value in tensor.detach().cpu().reshape(-1).numpy())
    if not all(math.isfinite(value) for value in values):
        raise ValueError("live fold inference emitted non-finite values")
    return values


def _canonical_pair(left: str, right: str, *, directed: bool) -> tuple[str, str]:
    if directed or left < right:
        return left, right
    return right, left


def _legal_corruptions(
    *,
    positive: tuple[str, str],
    endpoint_ids: tuple[str, ...],
    known_positives: set[tuple[str, str]],
    directed: bool,
) -> tuple[tuple[str, str], ...]:
    left, right = positive
    candidates = {
        _canonical_pair(candidate, right, directed=directed)
        for candidate in endpoint_ids
        if candidate != right
    } | {
        _canonical_pair(left, candidate, directed=directed)
        for candidate in endpoint_ids
        if candidate != left
    }
    candidates.difference_update(known_positives)
    candidates.discard(positive)
    candidates = {pair for pair in candidates if pair[0] != pair[1]}
    return tuple(sorted(candidates))


def _link_candidate_records(
    model: CoreGFM,
    encoded: Tensor,
    bound: BoundAuthoritativeFoldTest,
    positive_scores: tuple[float, ...],
) -> tuple[
    tuple[str, ...],
    str,
    str,
    tuple[LinkCandidateScores, ...],
]:
    bundle = bound.prepared.bundle
    endpoint_ids = tuple(node.id for node in bundle.nodes)
    endpoint_hash = canonical_sha256(list(endpoint_ids))
    known_positives = {
        _canonical_pair(edge.source_id, edge.target_id, directed=bundle.directed)
        for edge in bundle.edges
    }
    known_hash = canonical_sha256(sorted(known_positives))
    node_index = {node.id: node.index for node in bundle.nodes}
    edge_inventory = {
        identifier: pair for identifier, pair in _edge_locator_inventory(bundle).items()
    }
    index_to_id = {node.index: node.id for node in bundle.nodes}
    records: list[LinkCandidateScores] = []
    for entity_id, positive_score in zip(bound.record.entity_ids, positive_scores, strict=True):
        positive_indices = edge_inventory[entity_id]
        positive_pair = _canonical_pair(
            index_to_id[positive_indices[0]],
            index_to_id[positive_indices[1]],
            directed=bundle.directed,
        )
        negatives = _legal_corruptions(
            positive=positive_pair,
            endpoint_ids=endpoint_ids,
            known_positives=known_positives,
            directed=bundle.directed,
        )
        if not negatives:
            raise ValueError("link test query has no legal filtered negative candidates")
        negative_indices = torch.tensor(
            [(node_index[left], node_index[right]) for left, right in negatives],
            dtype=torch.long,
            device=encoded.device,
        )
        negative_scores = _tensor_values(model.binary_link_head(encoded, negative_indices))
        candidate_payload = {
            "queryEntityId": entity_id,
            "positiveEndpointPair": positive_pair,
            "negativeEndpointPairs": negatives,
        }
        records.append(
            LinkCandidateScores(
                queryEntityId=entity_id,
                positiveEndpointPair=positive_pair,
                positiveScore=positive_score,
                negativeEndpointPairs=negatives,
                negativeScores=negative_scores,
                candidateInventoryHash=canonical_sha256(candidate_payload),
            )
        )
    return endpoint_ids, endpoint_hash, known_hash, tuple(records)


def _derive_prediction_record(
    model: CoreGFM,
    adapter: BundleInputAdapter,
    bound: BoundAuthoritativeFoldTest,
) -> FoldPredictionRecord:
    if type(model) is not CoreGFM or type(adapter) is not BundleInputAdapter:
        raise TypeError("fold inference requires exact CoreGFM and BundleInputAdapter types")
    _verify_bound(bound)
    bundle = bound.prepared.bundle
    if adapter.graph_version_hash != bundle.graph_version_hash:
        raise ValueError("fold adapter does not belong to the prepared fold graph")
    encoded_artifact = encode_supervised_graph(model, bundle, adapter)
    encoded = encoded_artifact.tensor
    record = bound.record
    head_name = _head_for_task(record.task_kind)
    head = getattr(model, head_name)
    head_was_training = head.training
    head.eval()
    try:
        with torch.no_grad():
            if record.node_indices:
                locator = torch.tensor(record.node_indices, dtype=torch.long, device=encoded.device)
            else:
                locator = torch.tensor(record.edge_pairs, dtype=torch.long, device=encoded.device)
            if record.task_kind == "node-binary":
                logits = model.node_head(encoded[locator])
                if logits.shape != (len(record.entity_ids), 2):
                    raise ValueError("node-binary live inference requires exactly two logits")
                scores = _tensor_values(logits[:, 1] - logits[:, 0])
                probabilities = _tensor_values(torch.softmax(logits, dim=-1)[:, 1])
            elif record.task_kind == "signed-edge":
                logits = model.signed_edge_head(encoded, locator)
                scores = _tensor_values(logits)
                probabilities = _tensor_values(torch.sigmoid(logits))
            elif record.task_kind == "edge-binary":
                logits = model.binary_link_head(encoded, locator)
                scores = _tensor_values(logits)
                probabilities = _tensor_values(torch.sigmoid(logits))
            else:
                scores = _tensor_values(model.resilience_head(encoded[locator]))
                probabilities = ()
            endpoint_ids: tuple[str, ...] = ()
            endpoint_hash: str | None = None
            known_hash: str | None = None
            link_candidates: tuple[LinkCandidateScores, ...] = ()
            candidate_protocol: str | None = None
            if record.task_kind == "edge-binary":
                (
                    endpoint_ids,
                    endpoint_hash,
                    known_hash,
                    link_candidates,
                ) = _link_candidate_records(model, encoded, bound, scores)
                candidate_protocol = _LINK_CANDIDATE_PROTOCOL
    finally:
        head.train(head_was_training)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-live-fold-predictions/1.0",
        "foldId": record.fold_id,
        "taskKind": record.task_kind,
        "headName": head_name,
        "graphVersionHash": bundle.graph_version_hash,
        "bindingHash": record.binding_hash,
        "modelIdentityHash": derive_encoder_identity(model),
        "headStateHash": _state_hash(head),
        "adapterSchemaHash": adapter.schema.adapter_schema_hash,
        "adapterStateHash": _state_hash(adapter),
        "entityIds": record.entity_ids,
        "targets": record.targets,
        "scores": scores,
        "probabilities": probabilities,
        "candidateProtocol": candidate_protocol,
        "endpointIds": endpoint_ids,
        "endpointInventoryHash": endpoint_hash,
        "knownPositivePairsHash": known_hash,
        "linkCandidates": tuple(
            item.model_dump(mode="python", by_alias=True) for item in link_candidates
        ),
    }
    payload["predictionHash"] = canonical_sha256(payload)
    return FoldPredictionRecord.model_validate(payload)


def infer_core_gfm_fold(
    model: CoreGFM,
    adapter: BundleInputAdapter,
    bound: BoundAuthoritativeFoldTest,
) -> VerifiedFoldPredictions:
    """Run one fold's test entities and seal the exact live prediction evidence."""

    record = _derive_prediction_record(model, adapter, bound)
    return _new_predictions(record)


def verify_core_gfm_fold_predictions(
    model: CoreGFM,
    adapter: BundleInputAdapter,
    bound: BoundAuthoritativeFoldTest,
    predictions: VerifiedFoldPredictions,
) -> None:
    """Re-run live inference and reject any stored score or candidate drift."""

    if type(predictions) is not VerifiedFoldPredictions:
        raise TypeError("fold verification requires exact VerifiedFoldPredictions evidence")
    if (
        predictions._factory_seal is not _PREDICTION_SEAL
        or predictions.record.prediction_hash != predictions._sealed_prediction_hash
    ):
        raise ValueError("verified fold prediction runtime seal changed")
    try:
        reparsed = FoldPredictionRecord.model_validate(
            predictions.record.model_dump(mode="python", by_alias=True)
        )
    except (TypeError, ValueError) as error:
        raise ValueError("verified fold prediction runtime seal changed") from error
    expected = _derive_prediction_record(model, adapter, bound)
    if reparsed != predictions.record or expected != predictions.record:
        raise ValueError("stored predictions do not match the live model and adapter")


__all__ = [
    "AuthoritativeFoldTestRecord",
    "BoundAuthoritativeFoldTest",
    "FoldPredictionRecord",
    "FoldPreparationRecord",
    "LinkCandidateScores",
    "PreparedAuthoritativeFold",
    "VerifiedFoldPredictions",
    "bind_authoritative_fold_test",
    "infer_core_gfm_fold",
    "prepare_authoritative_fold",
    "verify_core_gfm_fold_predictions",
]
