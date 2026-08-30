"""Train-only/validation-only contracts for real core governance heads."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping

import torch
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .adapters import BundleInputAdapter, derive_training_selection
from .bundle import CoreGraphBundle
from .model import CoreGFM


TaskKind = Literal[
    "node-binary",
    "node-multiclass",
    "edge-binary",
    "signed-edge",
    "resilience-regression",
]
HeadName = Literal["node_head", "binary_link_head", "signed_edge_head", "resilience_head"]
ScalarTarget = int | float
HeadRunMode = Literal["smoke", "formal"]
SplitEvidenceScope = Literal["authoritative", "smoke-only"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _module_state_hash(module: nn.Module) -> str:
    return canonical_sha256(
        {
            name: canonical_tensor_digest(value)
            for name, value in sorted(module.state_dict().items())
        }
    )


def derive_encoder_identity(model: CoreGFM) -> str:
    """Derive the immutable pretrained encoder identity, excluding mutable task heads."""

    return canonical_sha256(
        {
            "scope": "core-frozen-encoder",
            "modelClass": f"{type(model).__module__}.{type(model).__qualname__}",
            "encoderClass": (
                f"{type(model.encoder).__module__}.{type(model.encoder).__qualname__}"
            ),
            "hiddenDim": 128,
            "encoderStateHash": _module_state_hash(model.encoder),
        }
    )


def _visible_edge_index(bundle: CoreGraphBundle) -> tuple[Tensor, str]:
    selection = derive_training_selection(bundle)
    node_indices = {node.id: node.index for node in bundle.nodes}
    pairs: list[tuple[int, int]] = []
    for edge_index in selection.visible_edge_indices:
        edge = bundle.edges[edge_index]
        source = node_indices[edge.source_id]
        target = node_indices[edge.target_id]
        pairs.append((source, target))
        if not bundle.directed:
            pairs.append((target, source))
    tensor = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.empty((2, 0), dtype=torch.long)
    )
    return tensor, selection.visible_topology_hash


class EncodedGraphProvenance(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-encoded-graph/1.0"] = Field(
        alias="schemaVersion"
    )
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_scope: Literal["core-frozen-encoder"] = Field(
        alias="modelIdentityScope"
    )
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=r"^[0-9a-f]{64}$")
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=r"^[0-9a-f]{64}$")
    topology_hash: str = Field(alias="topologyHash", pattern=r"^[0-9a-f]{64}$")
    topology_tensor_hash: str = Field(alias="topologyTensorHash", pattern=r"^[0-9a-f]{64}$")
    input_tensor_hash: str = Field(alias="inputTensorHash", pattern=r"^[0-9a-f]{64}$")
    encoded_tensor_hash: str = Field(alias="encodedTensorHash", pattern=r"^[0-9a-f]{64}$")
    num_nodes: int = Field(alias="numNodes", ge=1)
    artifact_hash: str = Field(alias="artifactHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_artifact_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"artifact_hash"})
        )
        if self.artifact_hash != expected:
            raise ValueError("encoded graph artifactHash does not match its provenance")
        return self


@dataclass(frozen=True, init=False)
class VerifiedEncodedGraph:
    tensor: Tensor
    provenance: EncodedGraphProvenance
    bundle: CoreGraphBundle
    adapter: BundleInputAdapter

    def verify(self, model: CoreGFM) -> None:
        if (
            self.provenance.num_nodes != len(self.bundle.nodes)
            or self.tensor.ndim != 2
            or self.tensor.shape[0] != self.provenance.num_nodes
        ):
            raise ValueError("encoded graph numNodes does not match bundle and tensor inventory")
        if self.bundle.graph_version_hash != self.provenance.graph_version_hash:
            raise ValueError("encoded graph bundle identity changed")
        if self.adapter.graph_version_hash != self.bundle.graph_version_hash:
            raise ValueError("encoded graph adapter no longer matches its bundle")
        if self.adapter.schema.adapter_schema_hash != self.provenance.adapter_schema_hash:
            raise ValueError("encoded graph adapter schema identity changed")
        if _module_state_hash(self.adapter) != self.provenance.adapter_state_hash:
            raise ValueError("encoded graph adapter state identity changed")
        if derive_encoder_identity(model) != self.provenance.model_identity_hash:
            raise ValueError("encoded graph encoder identity changed")
        edge_index, topology_hash = _visible_edge_index(self.bundle)
        if (
            topology_hash != self.provenance.topology_hash
            or canonical_sha256(canonical_tensor_digest(edge_index))
            != self.provenance.topology_tensor_hash
        ):
            raise ValueError("encoded graph visible topology identity changed")
        device = next(model.parameters()).device
        self.adapter.to(device)
        encoder_training = model.encoder.training
        adapter_training = self.adapter.training
        model.encoder.eval()
        self.adapter.eval()
        try:
            with torch.no_grad():
                inputs = self.adapter()
                recomputed = model.encode(inputs, edge_index.to(device))
        finally:
            model.encoder.train(encoder_training)
            self.adapter.train(adapter_training)
        if canonical_sha256(canonical_tensor_digest(inputs)) != self.provenance.input_tensor_hash:
            raise ValueError("encoded graph input tensor identity changed")
        observed_encoded_hash = canonical_sha256(canonical_tensor_digest(self.tensor))
        if observed_encoded_hash != self.provenance.encoded_tensor_hash:
            raise ValueError("encoded tensor identity changed")
        if canonical_sha256(canonical_tensor_digest(recomputed)) != observed_encoded_hash:
            raise ValueError("encoded tensor is not derived from the bound encoder and graph")


def _new_verified_encoded_graph(
    *,
    tensor: Tensor,
    provenance: EncodedGraphProvenance,
    bundle: CoreGraphBundle,
    adapter: BundleInputAdapter,
) -> VerifiedEncodedGraph:
    artifact = object.__new__(VerifiedEncodedGraph)
    object.__setattr__(artifact, "tensor", tensor)
    object.__setattr__(artifact, "provenance", provenance)
    object.__setattr__(artifact, "bundle", bundle)
    object.__setattr__(artifact, "adapter", adapter)
    return artifact


def encode_supervised_graph(
    model: CoreGFM,
    bundle: CoreGraphBundle,
    adapter: BundleInputAdapter,
) -> VerifiedEncodedGraph:
    """Compute and bind one train-visible non-text graph encoding."""

    if adapter.graph_version_hash != bundle.graph_version_hash:
        raise ValueError("adapter does not belong to the supplied graph bundle")
    device = next(model.parameters()).device
    adapter.to(device)
    edge_index, topology_hash = _visible_edge_index(bundle)
    encoder_training = model.encoder.training
    adapter_training = adapter.training
    model.encoder.eval()
    adapter.eval()
    try:
        with torch.no_grad():
            inputs = adapter()
            encoded = model.encode(inputs, edge_index.to(device)).detach().clone()
    finally:
        model.encoder.train(encoder_training)
        adapter.train(adapter_training)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-encoded-graph/1.0",
        "graphVersionHash": bundle.graph_version_hash,
        "modelIdentityHash": derive_encoder_identity(model),
        "modelIdentityScope": "core-frozen-encoder",
        "adapterSchemaHash": adapter.schema.adapter_schema_hash,
        "adapterStateHash": _module_state_hash(adapter),
        "topologyHash": topology_hash,
        "topologyTensorHash": canonical_sha256(canonical_tensor_digest(edge_index)),
        "inputTensorHash": canonical_sha256(canonical_tensor_digest(inputs)),
        "encodedTensorHash": canonical_sha256(canonical_tensor_digest(encoded)),
        "numNodes": len(bundle.nodes),
    }
    payload["artifactHash"] = canonical_sha256(payload)
    artifact = _new_verified_encoded_graph(
        tensor=encoded,
        provenance=EncodedGraphProvenance.model_validate(payload),
        bundle=bundle,
        adapter=adapter,
    )
    VerifiedEncodedGraph.verify(artifact, model)
    return artifact


class SupervisedPartition(_StrictModel):
    entity_ids: tuple[str, ...] = Field(alias="entityIds", strict=False, min_length=1)
    node_indices: tuple[int, ...] = Field(default=(), alias="nodeIndices", strict=False)
    edge_pairs: tuple[tuple[int, int], ...] = Field(default=(), alias="edgePairs", strict=False)
    targets: tuple[ScalarTarget, ...] = Field(strict=False, min_length=1)

    @model_validator(mode="after")
    def validate_partition(self):
        count = len(self.entity_ids)
        if (
            count != len(set(self.entity_ids))
            or self.entity_ids != tuple(sorted(self.entity_ids))
            or any(not identifier for identifier in self.entity_ids)
        ):
            raise ValueError("supervised entity IDs must be nonempty, unique, and sorted")
        if bool(self.node_indices) == bool(self.edge_pairs):
            raise ValueError("partition requires exactly one node-index or edge-pair locator")
        locators = self.node_indices if self.node_indices else self.edge_pairs
        if len(locators) != count or len(self.targets) != count:
            raise ValueError("partition locators, targets, and entity IDs must align")
        if len(locators) != len(set(locators)):
            raise ValueError("supervised partition locators must be unique")
        if self.node_indices and any(index < 0 for index in self.node_indices):
            raise ValueError("node indices must be nonnegative")
        if self.edge_pairs and any(
            left < 0 or right < 0 or left == right for left, right in self.edge_pairs
        ):
            raise ValueError("edge pairs require distinct nonnegative endpoints")
        if any(
            isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value))
            for value in self.targets
        ):
            raise ValueError("supervised targets must be finite numeric scalars")
        return self

    @property
    def partition_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


def _head_for_task(task_kind: TaskKind) -> HeadName:
    return {
        "node-binary": "node_head",
        "node-multiclass": "node_head",
        "edge-binary": "binary_link_head",
        "signed-edge": "signed_edge_head",
        "resilience-regression": "resilience_head",
    }[task_kind]  # type: ignore[return-value]


def _validate_task_partition(task_kind: TaskKind, partition: SupervisedPartition) -> None:
    node_task = task_kind in {"node-binary", "node-multiclass", "resilience-regression"}
    if node_task != bool(partition.node_indices):
        raise ValueError("task kind and supervised entity locator disagree")
    if task_kind in {"node-binary", "edge-binary", "signed-edge"}:
        if any(type(target) is not int or target not in {0, 1} for target in partition.targets):
            raise ValueError("binary supervised targets must be integer zero or one")
    elif task_kind == "node-multiclass":
        if any(type(target) is not int or target < 0 for target in partition.targets):
            raise ValueError("multiclass targets must be nonnegative integers")


def _unordered_pairs(partition: SupervisedPartition) -> set[tuple[int, int]]:
    return {(min(left, right), max(left, right)) for left, right in partition.edge_pairs}


def _stable_edge_id(bundle: CoreGraphBundle, left: int, right: int) -> str:
    if left >= len(bundle.nodes) or right >= len(bundle.nodes):
        raise ValueError("supervised edge locator is outside bundle node inventory")
    source_id = bundle.nodes[left].id
    target_id = bundle.nodes[right].id
    if not bundle.directed and source_id > target_id:
        source_id, target_id = target_id, source_id
    return f"edge:{source_id}:{target_id}"


def _stable_partition_ids(
    bundle: CoreGraphBundle,
    partition: SupervisedPartition,
) -> tuple[str, ...]:
    if partition.node_indices:
        if any(index >= len(bundle.nodes) for index in partition.node_indices):
            raise ValueError("supervised node locator is outside bundle node inventory")
        expected = tuple(bundle.nodes[index].id for index in partition.node_indices)
        if partition.entity_ids != expected:
            raise ValueError("supervised entity IDs must equal stable node IDs at their locators")
        return expected
    expected = tuple(_stable_edge_id(bundle, left, right) for left, right in partition.edge_pairs)
    if partition.entity_ids != expected:
        raise ValueError(
            "supervised entity IDs must equal stable edge identities at their locators"
        )
    return expected


@dataclass(frozen=True)
class _SplitEvidence:
    scope: SplitEvidenceScope
    assignment_kind: Literal["node", "edge"] | None
    roles: Mapping[str, str]
    evidence_hash: str


def _split_evidence(bundle: CoreGraphBundle) -> _SplitEvidence:
    manifest = bundle.split_manifest
    payload = {
        "graphVersionHash": bundle.graph_version_hash,
        "splitManifest": manifest.model_dump(mode="python", by_alias=True),
    }
    if not manifest.assignments or manifest.strategy == "all-visible-training":
        return _SplitEvidence(
            scope="smoke-only",
            assignment_kind=None,
            roles={},
            evidence_hash=canonical_sha256(payload),
        )
    roles = {assignment.entity_id: assignment.role for assignment in manifest.assignments}
    assignment_ids = set(roles)
    node_ids = {node.id for node in bundle.nodes}
    edge_identifiers = tuple(f"edge:{edge.source_id}:{edge.target_id}" for edge in bundle.edges)
    edge_ids = set(edge_identifiers)
    node_match = assignment_ids == node_ids
    edge_match = len(edge_ids) == len(edge_identifiers) and assignment_ids == edge_ids
    if node_match == edge_match:
        raise ValueError("authoritative split must cover one exact node or edge inventory")
    return _SplitEvidence(
        scope="authoritative",
        assignment_kind="node" if node_match else "edge",
        roles=roles,
        evidence_hash=canonical_sha256(payload),
    )


def _verify_train_validation_split(
    bundle: CoreGraphBundle,
    data: SupervisedTrainValidation,
) -> _SplitEvidence:
    if data.graph_version_hash != bundle.graph_version_hash:
        raise ValueError("supervised data graph identity does not match authoritative bundle")
    train_ids = set(_stable_partition_ids(bundle, data.train))
    validation_ids = set(_stable_partition_ids(bundle, data.validation))
    evidence = _split_evidence(bundle)
    if evidence.scope == "smoke-only":
        return evidence
    expected_kind = "node" if data.train.node_indices else "edge"
    if evidence.assignment_kind != expected_kind:
        raise ValueError("supervised locator kind does not match authoritative split inventory")
    authoritative_train = {
        identifier for identifier, role in evidence.roles.items() if role == "train"
    }
    authoritative_validation = {
        identifier for identifier, role in evidence.roles.items() if role == "validation"
    }
    if not train_ids <= authoritative_train:
        raise ValueError("supervised train locators must be a subset of authoritative train role")
    if validation_ids != authoritative_validation:
        raise ValueError(
            "supervised validation locators must equal exact authoritative validation role"
        )
    return evidence


def _verify_test_split(
    bundle: CoreGraphBundle,
    test: SupervisedTestSet,
) -> _SplitEvidence:
    if test.graph_version_hash != bundle.graph_version_hash:
        raise ValueError("supervised test graph identity does not match authoritative bundle")
    test_ids = set(_stable_partition_ids(bundle, test.test))
    evidence = _split_evidence(bundle)
    if evidence.scope == "smoke-only":
        return evidence
    expected_kind = "node" if test.test.node_indices else "edge"
    if evidence.assignment_kind != expected_kind:
        raise ValueError(
            "supervised test locator kind does not match authoritative split inventory"
        )
    authoritative_test = {
        identifier for identifier, role in evidence.roles.items() if role == "test"
    }
    if test_ids != authoritative_test:
        raise ValueError("supervised test locators must equal exact authoritative test role")
    return evidence


class SupervisedTrainValidation(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-supervised-train-validation/1.0"] = Field(
        alias="schemaVersion"
    )
    task_kind: TaskKind = Field(alias="taskKind")
    head_name: HeadName = Field(alias="headName")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=r"^[0-9a-f]{64}$")
    encoding_artifact_hash: str = Field(alias="encodingArtifactHash", pattern=r"^[0-9a-f]{64}$")
    train: SupervisedPartition
    validation: SupervisedPartition
    data_hash: str = Field(alias="dataHash", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        task_kind: TaskKind,
        provenance: EncodedGraphProvenance,
        train: SupervisedPartition,
        validation: SupervisedPartition,
    ) -> SupervisedTrainValidation:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-supervised-train-validation/1.0",
            "taskKind": task_kind,
            "headName": _head_for_task(task_kind),
            "graphVersionHash": provenance.graph_version_hash,
            "modelIdentityHash": provenance.model_identity_hash,
            "encodingArtifactHash": provenance.artifact_hash,
            "train": train.model_dump(mode="python", by_alias=True),
            "validation": validation.model_dump(mode="python", by_alias=True),
        }
        payload["dataHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_data(self):
        if self.head_name != _head_for_task(self.task_kind):
            raise ValueError("supervised head does not match task kind")
        _validate_task_partition(self.task_kind, self.train)
        _validate_task_partition(self.task_kind, self.validation)
        if set(self.train.entity_ids) & set(self.validation.entity_ids):
            raise ValueError("train and validation entity IDs must be disjoint")
        if self.train.node_indices and set(self.train.node_indices) & set(
            self.validation.node_indices
        ):
            raise ValueError("train and validation node indices must be disjoint")
        if self.train.edge_pairs and set(self.train.edge_pairs) & set(self.validation.edge_pairs):
            raise ValueError("train and validation edge pairs must be disjoint")
        if self.train.edge_pairs and _unordered_pairs(self.train) & _unordered_pairs(
            self.validation
        ):
            raise ValueError("train and validation unordered edge pairs must be disjoint")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"data_hash"})
        )
        if self.data_hash != expected:
            raise ValueError("dataHash does not match supervised train/validation data")
        return self


class SupervisedTestSet(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-supervised-test/1.0"] = Field(
        alias="schemaVersion"
    )
    task_kind: TaskKind = Field(alias="taskKind")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=r"^[0-9a-f]{64}$")
    encoding_artifact_hash: str = Field(alias="encodingArtifactHash", pattern=r"^[0-9a-f]{64}$")
    test: SupervisedPartition
    test_hash: str = Field(alias="testHash", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        task_kind: TaskKind,
        provenance: EncodedGraphProvenance,
        test: SupervisedPartition,
    ) -> SupervisedTestSet:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-supervised-test/1.0",
            "taskKind": task_kind,
            "graphVersionHash": provenance.graph_version_hash,
            "modelIdentityHash": provenance.model_identity_hash,
            "encodingArtifactHash": provenance.artifact_hash,
            "test": test.model_dump(mode="python", by_alias=True),
        }
        payload["testHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_test(self):
        _validate_task_partition(self.task_kind, self.test)
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"test_hash"})
        )
        if self.test_hash != expected:
            raise ValueError("testHash does not match supervised test data")
        return self


def validate_supervised_test_isolation(
    train_validation: SupervisedTrainValidation,
    test: SupervisedTestSet,
    *,
    bundle: CoreGraphBundle,
) -> str:
    """Bind one held-out test set only after proving complete role isolation."""

    if (
        type(train_validation) is not SupervisedTrainValidation
        or type(test) is not SupervisedTestSet
    ):
        raise TypeError("test isolation requires exact supervised contract types")
    train_evidence = _verify_train_validation_split(bundle, train_validation)
    test_evidence = _verify_test_split(bundle, test)
    if (
        train_evidence.scope != test_evidence.scope
        or train_evidence.evidence_hash != test_evidence.evidence_hash
    ):
        raise ValueError("supervised test split evidence does not match train/validation")
    if (
        train_validation.task_kind != test.task_kind
        or train_validation.graph_version_hash != test.graph_version_hash
        or train_validation.model_identity_hash != test.model_identity_hash
        or train_validation.encoding_artifact_hash != test.encoding_artifact_hash
    ):
        raise ValueError("supervised test identity does not match train/validation data")
    used_entities = set(train_validation.train.entity_ids) | set(
        train_validation.validation.entity_ids
    )
    if used_entities & set(test.test.entity_ids):
        raise ValueError("supervised test entity IDs overlap train or validation")
    if test.test.node_indices:
        used_nodes = set(train_validation.train.node_indices) | set(
            train_validation.validation.node_indices
        )
        if used_nodes & set(test.test.node_indices):
            raise ValueError("supervised test node indices overlap train or validation")
    else:
        used_pairs = _unordered_pairs(train_validation.train) | _unordered_pairs(
            train_validation.validation
        )
        if used_pairs & _unordered_pairs(test.test):
            raise ValueError("supervised test unordered edge pairs overlap train or validation")
    return canonical_sha256(
        {
            "trainValidationDataHash": train_validation.data_hash,
            "testHash": test.test_hash,
            "splitEvidenceHash": train_evidence.evidence_hash,
        }
    )


class HeadTrainingConfig(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-head-training-config/1.0"] = Field(
        alias="schemaVersion"
    )
    max_steps: int = Field(alias="maxSteps", ge=1, le=10_000)
    validation_interval: int = Field(alias="validationInterval", ge=1, le=1000)
    patience: int = Field(ge=1, le=1000)
    learning_rate: float = Field(alias="learningRate", gt=0.0, le=1.0)
    weight_decay: float = Field(alias="weightDecay", ge=0.0, le=1.0)
    freeze_encoder: Literal[True] = Field(alias="freezeEncoder")
    run_mode: HeadRunMode = Field(alias="runMode")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def smoke(cls, *, max_steps: int = 20, learning_rate: float = 0.01) -> HeadTrainingConfig:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-head-training-config/1.0",
            "maxSteps": max_steps,
            "validationInterval": 1,
            "patience": max_steps,
            "learningRate": learning_rate,
            "weightDecay": 0.0,
            "freezeEncoder": True,
            "runMode": "smoke",
        }
        payload["configHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @classmethod
    def formal(
        cls,
        *,
        max_steps: int,
        learning_rate: float = 0.01,
        validation_interval: int = 1,
        patience: int | None = None,
        weight_decay: float = 0.0,
    ) -> HeadTrainingConfig:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-head-training-config/1.0",
            "maxSteps": max_steps,
            "validationInterval": validation_interval,
            "patience": max_steps if patience is None else patience,
            "learningRate": learning_rate,
            "weightDecay": weight_decay,
            "freezeEncoder": True,
            "runMode": "formal",
        }
        payload["configHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"config_hash"})
        )
        if self.config_hash != expected:
            raise ValueError("configHash does not match head training config")
        return self


class HeadValidationPoint(_StrictModel):
    step: int = Field(ge=1)
    train_loss: float = Field(alias="trainLoss")
    validation_loss: float = Field(alias="validationLoss")
    validation_metric: float = Field(alias="validationMetric")

    @model_validator(mode="after")
    def validate_finite(self):
        if not all(
            math.isfinite(value)
            for value in (self.train_loss, self.validation_loss, self.validation_metric)
        ):
            raise ValueError("head validation history must be finite")
        return self


def _head_state_hash(head: nn.Module) -> str:
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(head.state_dict().items())}
    )


class HeadTrainingReport(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-head-training-report/1.0"] = Field(
        alias="schemaVersion"
    )
    task_kind: TaskKind = Field(alias="taskKind")
    head_name: HeadName = Field(alias="headName")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_scope: Literal["core-frozen-encoder"] = Field(
        alias="modelIdentityScope"
    )
    encoding_artifact_hash: str = Field(alias="encodingArtifactHash", pattern=r"^[0-9a-f]{64}$")
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=r"^[0-9a-f]{64}$")
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=r"^[0-9a-f]{64}$")
    topology_hash: str = Field(alias="topologyHash", pattern=r"^[0-9a-f]{64}$")
    data_hash: str = Field(alias="dataHash", pattern=r"^[0-9a-f]{64}$")
    train_partition_hash: str = Field(alias="trainPartitionHash", pattern=r"^[0-9a-f]{64}$")
    validation_partition_hash: str = Field(
        alias="validationPartitionHash", pattern=r"^[0-9a-f]{64}$"
    )
    num_nodes: int = Field(alias="numNodes", ge=1)
    split_evidence_scope: SplitEvidenceScope = Field(alias="splitEvidenceScope")
    split_evidence_hash: str = Field(alias="splitEvidenceHash", pattern=r"^[0-9a-f]{64}$")
    training_config: HeadTrainingConfig = Field(alias="trainingConfig")
    config_hash: str = Field(alias="configHash", pattern=r"^[0-9a-f]{64}$")
    encoded_tensor_hash: str = Field(alias="encodedTensorHash", pattern=r"^[0-9a-f]{64}$")
    history: tuple[HeadValidationPoint, ...] = Field(strict=False, min_length=1)
    best_step: int = Field(alias="bestStep", ge=1)
    best_metric: float = Field(alias="bestMetric")
    head_state_hash: str = Field(alias="headStateHash", pattern=r"^[0-9a-f]{64}$")
    promotion_eligible: bool = Field(alias="promotionEligible")
    report_hash: str = Field(alias="reportHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self):
        if self.config_hash != self.training_config.config_hash:
            raise ValueError("configHash does not match embedded head training config")
        expected_eligible = (
            self.split_evidence_scope == "authoritative"
            and self.training_config.run_mode == "formal"
        )
        if self.promotion_eligible != expected_eligible:
            raise ValueError("promotionEligible does not match split evidence and run mode")
        if (
            self.training_config.run_mode == "formal"
            and self.split_evidence_scope != "authoritative"
        ):
            raise ValueError("formal head training report requires authoritative split evidence")
        if self.best_metric != max(item.validation_metric for item in self.history):
            raise ValueError("bestMetric is not the maximum validation history metric")
        matching = [item for item in self.history if item.validation_metric == self.best_metric]
        if self.best_step != matching[0].step:
            raise ValueError("bestStep must select the first strict validation maximum")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"report_hash"})
        )
        if self.report_hash != expected:
            raise ValueError("reportHash does not match head training report")
        return self

    def calculate_current_head_hash(self, model: CoreGFM) -> str:
        return _head_state_hash(getattr(model, self.head_name))


_HEAD_REPORT_FACTORY_SEAL = object()


@dataclass(frozen=True, init=False)
class VerifiedHeadTrainingReport:
    """Process-local proof that a report was emitted by real head training."""

    record: HeadTrainingReport
    _sealed_report_hash: str
    _factory_seal: object

    @property
    def report_hash(self) -> str:
        return self.record.report_hash

    @property
    def head_name(self) -> HeadName:
        return self.record.head_name

    @property
    def graph_version_hash(self) -> str:
        return self.record.graph_version_hash

    @property
    def model_identity_hash(self) -> str:
        return self.record.model_identity_hash

    @property
    def encoding_artifact_hash(self) -> str:
        return self.record.encoding_artifact_hash

    @property
    def encoded_tensor_hash(self) -> str:
        return self.record.encoded_tensor_hash

    @property
    def history(self) -> tuple[HeadValidationPoint, ...]:
        return self.record.history

    @property
    def best_step(self) -> int:
        return self.record.best_step

    @property
    def best_metric(self) -> float:
        return self.record.best_metric

    @property
    def head_state_hash(self) -> str:
        return self.record.head_state_hash

    @property
    def promotion_eligible(self) -> bool:
        return self.record.promotion_eligible

    def calculate_current_head_hash(self, model: CoreGFM) -> str:
        return _head_state_hash(getattr(model, self.record.head_name))


def _new_verified_head_training_report(
    record: HeadTrainingReport,
) -> VerifiedHeadTrainingReport:
    verified = object.__new__(VerifiedHeadTrainingReport)
    object.__setattr__(verified, "record", record)
    object.__setattr__(verified, "_sealed_report_hash", record.report_hash)
    object.__setattr__(verified, "_factory_seal", _HEAD_REPORT_FACTORY_SEAL)
    return verified


def _verify_head_report_runtime_seal(report: VerifiedHeadTrainingReport) -> HeadTrainingReport:
    if type(report) is not VerifiedHeadTrainingReport:
        raise TypeError("calibration requires exact VerifiedHeadTrainingReport evidence")
    if (
        report._factory_seal is not _HEAD_REPORT_FACTORY_SEAL
        or type(report.record) is not HeadTrainingReport
        or report.record.report_hash != report._sealed_report_hash
    ):
        raise ValueError("verified head training report runtime seal changed")
    reparsed = HeadTrainingReport.model_validate(
        report.record.model_dump(mode="python", by_alias=True)
    )
    if reparsed != report.record:
        raise ValueError("verified head training report record changed")
    return report.record


def verify_head_training_report(
    model: CoreGFM,
    encoded: VerifiedEncodedGraph,
    data: SupervisedTrainValidation,
    report: VerifiedHeadTrainingReport,
) -> None:
    """Re-derive every report binding needed by calibration from live inputs."""

    if type(encoded) is not VerifiedEncodedGraph:
        raise TypeError("head report verification requires exact VerifiedEncodedGraph")
    if type(data) is not SupervisedTrainValidation:
        raise TypeError("head report verification requires exact supervised data evidence")
    record = _verify_head_report_runtime_seal(report)
    VerifiedEncodedGraph.verify(encoded, model)
    provenance = encoded.provenance
    evidence = _verify_train_validation_split(encoded.bundle, data)
    if record.training_config.run_mode == "formal" and evidence.scope != "authoritative":
        raise ValueError("formal head training requires authoritative split evidence")
    current_head_hash = _head_state_hash(getattr(model, data.head_name))
    if record.head_state_hash != current_head_hash:
        raise ValueError("current head state does not match the restored validation best")
    expected = {
        "task_kind": data.task_kind,
        "head_name": data.head_name,
        "graph_version_hash": provenance.graph_version_hash,
        "model_identity_hash": derive_encoder_identity(model),
        "model_identity_scope": provenance.model_identity_scope,
        "encoding_artifact_hash": provenance.artifact_hash,
        "adapter_schema_hash": encoded.adapter.schema.adapter_schema_hash,
        "adapter_state_hash": _module_state_hash(encoded.adapter),
        "topology_hash": _visible_edge_index(encoded.bundle)[1],
        "data_hash": data.data_hash,
        "train_partition_hash": data.train.partition_hash,
        "validation_partition_hash": data.validation.partition_hash,
        "num_nodes": len(encoded.bundle.nodes),
        "split_evidence_scope": evidence.scope,
        "split_evidence_hash": evidence.evidence_hash,
        "encoded_tensor_hash": canonical_sha256(canonical_tensor_digest(encoded.tensor)),
        "promotion_eligible": (
            evidence.scope == "authoritative" and record.training_config.run_mode == "formal"
        ),
    }
    if any(getattr(record, name) != value for name, value in expected.items()):
        raise ValueError(
            "head training report does not match live model, graph, and data provenance"
        )


def _partition_tensors(
    partition: SupervisedPartition, *, device: torch.device
) -> tuple[Tensor, Tensor]:
    if partition.node_indices:
        locator = torch.tensor(partition.node_indices, dtype=torch.long, device=device)
    else:
        locator = torch.tensor(partition.edge_pairs, dtype=torch.long, device=device)
    if all(type(target) is int for target in partition.targets):
        targets = torch.tensor(partition.targets, dtype=torch.long, device=device)
    else:
        targets = torch.tensor(partition.targets, dtype=torch.float32, device=device)
    return locator, targets


def _loss(
    model: CoreGFM,
    encoded: Tensor,
    task_kind: TaskKind,
    partition: SupervisedPartition,
) -> Tensor:
    locator, targets = _partition_tensors(partition, device=encoded.device)
    if task_kind in {"node-binary", "node-multiclass"}:
        logits = model.node_head(encoded[locator])
        if targets.dtype != torch.long or int(targets.max()) >= logits.shape[1]:
            raise ValueError("node targets are outside the configured class inventory")
        return functional.cross_entropy(logits, targets)
    if task_kind == "edge-binary":
        logits = model.binary_link_head(encoded, locator)
        return functional.binary_cross_entropy_with_logits(logits, targets.float())
    if task_kind == "signed-edge":
        logits = model.signed_edge_head(encoded, locator)
        return functional.binary_cross_entropy_with_logits(logits, targets.float())
    predictions = model.resilience_head(encoded[locator])
    return functional.mse_loss(predictions, targets.float())


def fit_supervised_head(
    model: CoreGFM,
    encoded: VerifiedEncodedGraph,
    data: SupervisedTrainValidation,
    *,
    config: HeadTrainingConfig,
) -> VerifiedHeadTrainingReport:
    """Optimize one real head with train labels and select only on validation labels."""

    if type(encoded) is not VerifiedEncodedGraph:
        raise TypeError("fit_supervised_head requires exact VerifiedEncodedGraph evidence")
    if type(data) is not SupervisedTrainValidation or type(config) is not HeadTrainingConfig:
        raise TypeError("fit_supervised_head requires exact supervised data and config types")
    VerifiedEncodedGraph.verify(encoded, model)
    provenance = encoded.provenance
    if (
        data.graph_version_hash != provenance.graph_version_hash
        or data.model_identity_hash != provenance.model_identity_hash
        or data.encoding_artifact_hash != provenance.artifact_hash
    ):
        raise ValueError("supervised data identity does not match encoded graph provenance")
    split_evidence = _verify_train_validation_split(encoded.bundle, data)
    if config.run_mode == "formal" and split_evidence.scope != "authoritative":
        raise ValueError("formal head training requires authoritative split evidence")
    device = next(model.parameters()).device
    encoded_tensor = encoded.tensor
    if (
        encoded_tensor.ndim != 2
        or encoded_tensor.shape[1] != 128
        or encoded_tensor.shape[0] != provenance.num_nodes
    ):
        raise ValueError("encoded node representation must have shape [nodes, 128]")
    if not bool(torch.isfinite(encoded_tensor).all()):
        raise ValueError("encoded node representation must be finite")
    maximum_index = max((*data.train.node_indices, *data.validation.node_indices), default=-1)
    maximum_edge = max(
        (
            endpoint
            for pair in (*data.train.edge_pairs, *data.validation.edge_pairs)
            for endpoint in pair
        ),
        default=-1,
    )
    if max(maximum_index, maximum_edge) >= encoded_tensor.shape[0]:
        raise ValueError("supervised locator is outside encoded node inventory")
    features = encoded_tensor.detach().to(device)
    head: nn.Module = getattr(model, data.head_name)
    if data.task_kind == "node-binary":
        sample = model.node_head(features[:1])
        if sample.ndim != 2 or sample.shape[1] != 2:
            raise ValueError("node-binary head training requires exactly two class logits")
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[HeadValidationPoint] = []
    best_metric = float("-inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    try:
        for step in range(1, config.max_steps + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            train_loss = _loss(model, features, data.task_kind, data.train)
            if not bool(torch.isfinite(train_loss)):
                raise ValueError("head training loss became non-finite")
            train_loss.backward()
            optimizer.step()
            if step % config.validation_interval and step != config.max_steps:
                continue
            head.eval()
            with torch.no_grad():
                validation_loss = _loss(model, features, data.task_kind, data.validation)
            metric = -float(validation_loss.detach().cpu())
            point = HeadValidationPoint(
                step=step,
                trainLoss=float(train_loss.detach().cpu()),
                validationLoss=float(validation_loss.detach().cpu()),
                validationMetric=metric,
            )
            history.append(point)
            if metric > best_metric:
                best_state = copy.deepcopy(head.state_dict())
                best_metric = metric
                best_step = step
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                break
        if best_state is None:
            raise RuntimeError("head training did not produce validation evidence")
        head.load_state_dict(best_state)
        VerifiedEncodedGraph.verify(encoded, model)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-head-training-report/1.0",
        "taskKind": data.task_kind,
        "headName": data.head_name,
        "graphVersionHash": provenance.graph_version_hash,
        "modelIdentityHash": derive_encoder_identity(model),
        "modelIdentityScope": provenance.model_identity_scope,
        "encodingArtifactHash": provenance.artifact_hash,
        "adapterSchemaHash": provenance.adapter_schema_hash,
        "adapterStateHash": provenance.adapter_state_hash,
        "topologyHash": provenance.topology_hash,
        "dataHash": data.data_hash,
        "trainPartitionHash": data.train.partition_hash,
        "validationPartitionHash": data.validation.partition_hash,
        "numNodes": provenance.num_nodes,
        "splitEvidenceScope": split_evidence.scope,
        "splitEvidenceHash": split_evidence.evidence_hash,
        "trainingConfig": config.model_dump(mode="python", by_alias=True),
        "configHash": config.config_hash,
        "encodedTensorHash": provenance.encoded_tensor_hash,
        "history": [item.model_dump(mode="python", by_alias=True) for item in history],
        "bestStep": best_step,
        "bestMetric": best_metric,
        "headStateHash": _head_state_hash(head),
        "promotionEligible": (
            split_evidence.scope == "authoritative" and config.run_mode == "formal"
        ),
    }
    payload["reportHash"] = canonical_sha256(payload)
    record = HeadTrainingReport.model_validate(payload)
    report = _new_verified_head_training_report(record)
    verify_head_training_report(model, encoded, data, report)
    return report


__all__ = [
    "EncodedGraphProvenance",
    "HeadTrainingConfig",
    "HeadTrainingReport",
    "VerifiedHeadTrainingReport",
    "SupervisedPartition",
    "SupervisedTestSet",
    "SupervisedTrainValidation",
    "VerifiedEncodedGraph",
    "derive_encoder_identity",
    "encode_supervised_graph",
    "fit_supervised_head",
    "validate_supervised_test_isolation",
    "verify_head_training_report",
]
