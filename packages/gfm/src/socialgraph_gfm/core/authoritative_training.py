"""Sealed train/validation derivation from one authoritative core fold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256

from .fold_evaluation import (
    FoldTaskKind,
    PreparedAuthoritativeFold,
    _edge_locator_inventory,
    _normalize_targets,
    _verify_prepared,
)
from .formal_preflight import ExperimentLabels
from .supervised import (
    EncodedGraphProvenance,
    SupervisedPartition,
    SupervisedTrainValidation,
    _verify_train_validation_split,
)


LabelBudget = Literal["1", "5", "20", "full"]

_HASH = r"^[0-9a-f]{64}$"
_BUDGETS = {"1": 1, "5": 5, "20": 20}
_TASKS = {"node-binary", "signed-edge", "resilience-regression", "edge-binary"}
_SAMPLING_PROTOCOL = "socialgraph-fm.core-hash-ranked-label-budget/1.0"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _sampling_context_hash(
    *,
    fold_id: str,
    task_kind: FoldTaskKind,
    target_name: str,
    label_budget: LabelBudget,
    experiment_seed: int,
    graph_version_hash: str,
    split_manifest_hash: str,
    preparation_hash: str,
    labels_hash: str,
    train_role_hash: str,
) -> str:
    return canonical_sha256(
        {
            "samplingProtocol": _SAMPLING_PROTOCOL,
            "foldId": fold_id,
            "taskKind": task_kind,
            "targetName": target_name,
            "labelBudget": label_budget,
            "experimentSeed": experiment_seed,
            "graphVersionHash": graph_version_hash,
            "splitManifestHash": split_manifest_hash,
            "preparationHash": preparation_hash,
            "labelsHash": labels_hash,
            "trainRoleHash": train_role_hash,
        }
    )


def _validate_partition_task(task_kind: FoldTaskKind, partition: SupervisedPartition) -> None:
    node_task = task_kind in {"node-binary", "resilience-regression"}
    if node_task != bool(partition.node_indices):
        raise ValueError("authoritative task and partition locator kind disagree")
    if task_kind in {"node-binary", "signed-edge", "edge-binary"}:
        if any(type(value) is not int or value not in {0, 1} for value in partition.targets):
            raise ValueError("authoritative classification targets must be normalized to 0/1")
    elif any(type(value) is not float for value in partition.targets):
        raise ValueError("authoritative resilience targets must be normalized finite floats")


class AuthoritativeFoldTrainValidationRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-authoritative-fold-train-validation/1.0"] = (
        Field(alias="schemaVersion")
    )
    fold_id: str = Field(alias="foldId", min_length=1)
    task_kind: FoldTaskKind = Field(alias="taskKind")
    target_name: str = Field(alias="targetName", min_length=1)
    label_budget: LabelBudget = Field(alias="labelBudget")
    experiment_seed: int = Field(alias="experimentSeed", ge=0, le=9_223_372_036_854_775_807)
    requirement_id: str = Field(alias="requirementId", min_length=1)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)
    preparation_hash: str = Field(alias="preparationHash", pattern=_HASH)
    labels_hash: str = Field(alias="labelsHash", pattern=_HASH)
    sampling_protocol: Literal["socialgraph-fm.core-hash-ranked-label-budget/1.0"] = Field(
        alias="samplingProtocol"
    )
    sampling_context_hash: str = Field(alias="samplingContextHash", pattern=_HASH)
    train_role_entity_ids: tuple[str, ...] = Field(alias="trainRoleEntityIds", strict=False)
    validation_role_entity_ids: tuple[str, ...] = Field(
        alias="validationRoleEntityIds", strict=False, min_length=1
    )
    train_role_hash: str = Field(alias="trainRoleHash", pattern=_HASH)
    validation_role_hash: str = Field(alias="validationRoleHash", pattern=_HASH)
    train: SupervisedPartition
    validation: SupervisedPartition
    record_hash: str = Field(alias="recordHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_record(self):
        if (
            not self.train_role_entity_ids
            or self.train_role_entity_ids != tuple(sorted(set(self.train_role_entity_ids)))
            or self.validation_role_entity_ids
            != tuple(sorted(set(self.validation_role_entity_ids)))
        ):
            raise ValueError("authoritative train and validation role IDs must be sorted sets")
        if set(self.train_role_entity_ids) & set(self.validation_role_entity_ids):
            raise ValueError("authoritative train and validation roles must be disjoint")
        if self.train_role_hash != canonical_sha256(list(self.train_role_entity_ids)):
            raise ValueError("trainRoleHash does not match authoritative role IDs")
        if self.validation_role_hash != canonical_sha256(list(self.validation_role_entity_ids)):
            raise ValueError("validationRoleHash does not match authoritative role IDs")
        if not set(self.train.entity_ids) <= set(self.train_role_entity_ids):
            raise ValueError("selected training entities must be official train roles")
        if self.validation.entity_ids != self.validation_role_entity_ids:
            raise ValueError("validation partition must equal every official validation role")
        if self.label_budget == "full" and self.train.entity_ids != self.train_role_entity_ids:
            raise ValueError("full budget must equal every official train role")
        if set(self.train.entity_ids) & set(self.validation.entity_ids):
            raise ValueError("authoritative train and validation partitions must be disjoint")
        _validate_partition_task(self.task_kind, self.train)
        _validate_partition_task(self.task_kind, self.validation)
        expected_context = _sampling_context_hash(
            fold_id=self.fold_id,
            task_kind=self.task_kind,
            target_name=self.target_name,
            label_budget=self.label_budget,
            experiment_seed=self.experiment_seed,
            graph_version_hash=self.graph_version_hash,
            split_manifest_hash=self.split_manifest_hash,
            preparation_hash=self.preparation_hash,
            labels_hash=self.labels_hash,
            train_role_hash=self.train_role_hash,
        )
        if self.sampling_context_hash != expected_context:
            raise ValueError("samplingContextHash does not match authoritative sampling inputs")
        expected_record = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        )
        if self.record_hash != expected_record:
            raise ValueError("recordHash does not match authoritative train/validation data")
        return self


_EVIDENCE_SEAL = object()


@dataclass(frozen=True, init=False)
class AuthoritativeFoldTrainValidation:
    prepared: PreparedAuthoritativeFold
    labels: ExperimentLabels
    record: AuthoritativeFoldTrainValidationRecord
    _sealed_record_hash: str
    _factory_seal: object

    def __new__(cls, *args: object, **kwargs: object):
        del args, kwargs
        raise TypeError(
            "AuthoritativeFoldTrainValidation is emitted only by authoritative derivation"
        )


def _new_evidence(
    *,
    prepared: PreparedAuthoritativeFold,
    labels: ExperimentLabels,
    record: AuthoritativeFoldTrainValidationRecord,
) -> AuthoritativeFoldTrainValidation:
    evidence = object.__new__(AuthoritativeFoldTrainValidation)
    object.__setattr__(evidence, "prepared", prepared)
    object.__setattr__(evidence, "labels", labels)
    object.__setattr__(evidence, "record", record)
    object.__setattr__(evidence, "_sealed_record_hash", record.record_hash)
    object.__setattr__(evidence, "_factory_seal", _EVIDENCE_SEAL)
    return evidence


def _partition(
    prepared: PreparedAuthoritativeFold,
    entity_ids: tuple[str, ...],
    targets_by_id: dict[str, int | float],
    *,
    node_task: bool,
) -> SupervisedPartition:
    targets = tuple(targets_by_id[entity_id] for entity_id in entity_ids)
    if node_task:
        node_indices = {node.id: node.index for node in prepared.bundle.nodes}
        return SupervisedPartition(
            entityIds=entity_ids,
            nodeIndices=tuple(node_indices[entity_id] for entity_id in entity_ids),
            targets=targets,
        )
    edge_pairs = _edge_locator_inventory(prepared.bundle)
    return SupervisedPartition(
        entityIds=entity_ids,
        edgePairs=tuple(edge_pairs[entity_id] for entity_id in entity_ids),
        targets=targets,
    )


def _ranked_sample(
    entity_ids: tuple[str, ...],
    *,
    count: int,
    class_key: str,
    sampling_context_hash: str,
) -> tuple[str, ...]:
    if len(entity_ids) < count:
        raise ValueError(f"authoritative training class {class_key} has fewer than {count} labels")
    ranked = sorted(
        entity_ids,
        key=lambda entity_id: (
            canonical_sha256(
                {
                    "samplingContextHash": sampling_context_hash,
                    "classKey": class_key,
                    "entityId": entity_id,
                }
            ),
            entity_id,
        ),
    )
    return tuple(sorted(ranked[:count]))


def _derive_record(
    prepared: PreparedAuthoritativeFold,
    labels: ExperimentLabels,
    *,
    target_name: str,
    task_kind: FoldTaskKind,
    label_budget: LabelBudget,
    experiment_seed: int,
) -> AuthoritativeFoldTrainValidationRecord:
    _verify_prepared(prepared)
    if type(labels) is not ExperimentLabels:
        raise TypeError("authoritative training requires exact ExperimentLabels evidence")
    if type(target_name) is not str or not target_name:
        raise ValueError("authoritative training target name must be nonempty")
    if task_kind not in _TASKS:
        raise ValueError("unknown authoritative training task kind")
    if label_budget not in (*_BUDGETS, "full"):
        raise ValueError("label budget must be one of 1, 5, 20, or full")
    if type(experiment_seed) is not int or not 0 <= experiment_seed <= 9_223_372_036_854_775_807:
        raise ValueError("experiment seed must be a nonnegative signed 64-bit integer")

    expected_assignment_kind = "edge" if task_kind in {"signed-edge", "edge-binary"} else "node"
    if prepared.record.assignment_kind != expected_assignment_kind:
        raise ValueError("task locator kind does not match authoritative fold assignments")
    matching_targets = tuple(target for target in labels.targets if target.name == target_name)
    if len(matching_targets) != 1:
        raise ValueError("authoritative labels must contain the exact requested target")
    target = matching_targets[0]
    raw_targets_by_id = {item.entity_id: item.value for item in target.values}
    role_by_id = {
        assignment.entity_id: assignment.role
        for assignment in prepared.fold.split_manifest.assignments
    }
    if set(raw_targets_by_id) != set(role_by_id):
        raise ValueError("label entities must equal the exact authoritative fold inventory")
    all_ids = tuple(sorted(role_by_id))
    normalized = _normalize_targets(
        task_kind, tuple(raw_targets_by_id[entity_id] for entity_id in all_ids)
    )
    targets_by_id = dict(zip(all_ids, normalized, strict=True))
    train_role_ids = tuple(entity_id for entity_id in all_ids if role_by_id[entity_id] == "train")
    validation_role_ids = tuple(
        entity_id for entity_id in all_ids if role_by_id[entity_id] == "validation"
    )
    train_role_hash = canonical_sha256(list(train_role_ids))
    validation_role_hash = canonical_sha256(list(validation_role_ids))
    context_hash = _sampling_context_hash(
        fold_id=prepared.fold.fold_id,
        task_kind=task_kind,
        target_name=target_name,
        label_budget=label_budget,
        experiment_seed=experiment_seed,
        graph_version_hash=prepared.bundle.graph_version_hash,
        split_manifest_hash=prepared.fold.split_manifest_hash,
        preparation_hash=prepared.record.preparation_hash,
        labels_hash=labels.labels_hash,
        train_role_hash=train_role_hash,
    )
    if label_budget == "full":
        selected_train_ids = train_role_ids
    else:
        count = _BUDGETS[label_budget]
        if task_kind in {"node-binary", "signed-edge"}:
            selections: list[str] = []
            for class_value in (0, 1):
                candidates = tuple(
                    entity_id
                    for entity_id in train_role_ids
                    if targets_by_id[entity_id] == class_value
                )
                selections.extend(
                    _ranked_sample(
                        candidates,
                        count=count,
                        class_key=str(class_value),
                        sampling_context_hash=context_hash,
                    )
                )
            selected_train_ids = tuple(sorted(selections))
        else:
            class_key = "positive" if task_kind == "edge-binary" else "regression"
            selected_train_ids = _ranked_sample(
                train_role_ids,
                count=count,
                class_key=class_key,
                sampling_context_hash=context_hash,
            )

    node_task = task_kind in {"node-binary", "resilience-regression"}
    train = _partition(prepared, selected_train_ids, targets_by_id, node_task=node_task)
    validation = _partition(prepared, validation_role_ids, targets_by_id, node_task=node_task)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-authoritative-fold-train-validation/1.0",
        "foldId": prepared.fold.fold_id,
        "taskKind": task_kind,
        "targetName": target_name,
        "labelBudget": label_budget,
        "experimentSeed": experiment_seed,
        "requirementId": labels.requirement_id,
        "graphVersionHash": prepared.bundle.graph_version_hash,
        "splitManifestHash": prepared.fold.split_manifest_hash,
        "preparationHash": prepared.record.preparation_hash,
        "labelsHash": labels.labels_hash,
        "samplingProtocol": _SAMPLING_PROTOCOL,
        "samplingContextHash": context_hash,
        "trainRoleEntityIds": train_role_ids,
        "validationRoleEntityIds": validation_role_ids,
        "trainRoleHash": train_role_hash,
        "validationRoleHash": validation_role_hash,
        "train": train.model_dump(mode="python", by_alias=True),
        "validation": validation.model_dump(mode="python", by_alias=True),
    }
    payload["recordHash"] = canonical_sha256(payload)
    return AuthoritativeFoldTrainValidationRecord.model_validate(payload)


def derive_authoritative_fold_train_validation(
    prepared: PreparedAuthoritativeFold,
    labels: ExperimentLabels,
    *,
    target_name: str,
    task_kind: FoldTaskKind,
    label_budget: LabelBudget,
    experiment_seed: int,
) -> AuthoritativeFoldTrainValidation:
    """Derive immutable official train/validation data without caller-supplied targets."""

    record = _derive_record(
        prepared,
        labels,
        target_name=target_name,
        task_kind=task_kind,
        label_budget=label_budget,
        experiment_seed=experiment_seed,
    )
    return _new_evidence(prepared=prepared, labels=labels, record=record)


def verify_authoritative_fold_train_validation(
    evidence: AuthoritativeFoldTrainValidation,
) -> AuthoritativeFoldTrainValidationRecord:
    """Re-derive every role, locator, normalized target, and sampled entity exactly."""

    if type(evidence) is not AuthoritativeFoldTrainValidation:
        raise TypeError("verification requires exact authoritative training evidence")
    if evidence._factory_seal is not _EVIDENCE_SEAL:
        raise ValueError("authoritative training runtime seal changed")
    if evidence.record.record_hash != evidence._sealed_record_hash:
        raise ValueError("authoritative training sealed record changed")
    expected = _derive_record(
        evidence.prepared,
        evidence.labels,
        target_name=evidence.record.target_name,
        task_kind=evidence.record.task_kind,
        label_budget=evidence.record.label_budget,
        experiment_seed=evidence.record.experiment_seed,
    )
    if evidence.record != expected:
        raise ValueError("authoritative training data changed from its live derivation")
    return evidence.record


def bind_authoritative_supervised_train_validation(
    evidence: AuthoritativeFoldTrainValidation,
    provenance: EncodedGraphProvenance,
) -> SupervisedTrainValidation:
    """Bind sealed official partitions to one exact live encoding provenance."""

    record = verify_authoritative_fold_train_validation(evidence)
    if type(provenance) is not EncodedGraphProvenance:
        raise TypeError("supervised binding requires exact encoded graph provenance")
    if provenance.graph_version_hash != record.graph_version_hash or provenance.num_nodes != len(
        evidence.prepared.bundle.nodes
    ):
        raise ValueError("encoded graph identity does not match the authoritative fold")
    data = SupervisedTrainValidation.create(
        task_kind=record.task_kind,
        provenance=provenance,
        train=record.train,
        validation=record.validation,
    )
    _verify_train_validation_split(evidence.prepared.bundle, data)
    return data
