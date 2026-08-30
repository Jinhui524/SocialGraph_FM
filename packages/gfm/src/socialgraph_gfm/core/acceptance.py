"""Fail-closed, byte-revalidated acceptance for core formal experiments."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, TypeVar

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .adapters import AdapterParameterModule, AdapterSchema, BundleInputAdapter
from .authoritative_training import (
    bind_authoritative_supervised_train_validation,
    derive_authoritative_fold_train_validation,
    verify_authoritative_fold_train_validation,
)
from .bundle import SplitManifest, CoreGraphBundle
from .calibration import (
    BinaryScoreSemantics,
    CalibrationFitReport,
    CalibrationProtocol,
    derive_validation_scores,
    fit_score_calibration_report,
)
from .checkpoint import CheckpointBindings, load_checkpoint
from .config import TrainingConfig
from .experiments import (
    ExperimentAggregate,
    ExperimentArtifactRef,
    ExperimentExecutionConfigEvidence,
    ExperimentLedger,
    ExperimentProtocol,
    ExperimentRecipeEvidence,
    ExperimentRunRecord,
    ExperimentTrainingDataEvidence,
    PredictionEvidence,
    TargetEvidence,
    ThresholdSelectionEvidence,
    TransferDecision,
    aggregate_experiment,
    build_experiment_matrix,
    derive_transfer_advantage,
)
from .resource_telemetry import ResourceTelemetryRecord
from .telemetry_receipt import (
    TelemetryReceipt,
    TelemetryReceiptExpectations,
    TrustedTelemetryPolicy,
)
from .formal_preflight import (
    FORMAL_CORPUS_REQUIREMENTS,
    ExperimentDatasetManifest,
    ExperimentLabels,
    ExperimentSplitInventory,
    _publish_exact as _publish_immutable_exact,
    load_formal_preflight,
)
from .fold_evaluation import (
    FoldPredictionRecord,
    bind_authoritative_fold_test,
    infer_core_gfm_fold,
    prepare_authoritative_fold,
)
from .fold_inventory import (
    CellFoldEvaluationInventory,
    FoldEvaluationBinding,
    FoldRuntimeArtifactRef,
)
from .fold_recovery import verify_fold_recovery_state, verify_run_recovery_inventory
from .fold_metrics import FoldMetricInput, derive_equal_weight_fold_metrics
from .governance_winner import (
    GovernanceValidationObservation,
    GovernanceWinnerSelection,
    derive_governance_winner,
)
from .safe_paths import read_confined_snapshot, reject_link_components, secure_existing_root
from .model import CoreGFM
from .structure_features import StructureAlgorithmConfig, StructureCacheManifest
from .supervised import (
    HeadTrainingReport,
    SupervisedTrainValidation,
    _loss as _supervised_loss,
    _new_verified_head_training_report,
    encode_supervised_graph,
    verify_head_training_report,
)
from .trainer import _model_state_hash, _parse_fit_state
from .metrics import (
    TaskMetricSet,
    binary_auprc,
    binary_auroc,
    binary_brier,
    binary_ece,
    binary_metrics_at_threshold,
    filtered_ranking_metrics,
    mean_absolute_error,
    negative_class_auprc,
    recall_at_fixed_fpr,
    spearman_correlation,
)


_HASH = r"^[0-9a-f]{64}$"
_MAX_ACCEPTANCE_BYTES = 4 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 512 * 1024**2
_REQUIRED_GOVERNANCE_TASKS = (
    "github.relation-completion",
    "penn94.community-resilience",
    "tolokers.risk",
    "wiki-rfa.vote-sign",
)
_FOLD_TARGETS: dict[str, tuple[str, str]] = {
    "github.relation-completion": ("relationCompletion", "edge-binary"),
    "penn94.community-resilience": ("resilience", "resilience-regression"),
    "tolokers.risk": ("banned", "node-binary"),
    "wiki-rfa.vote-sign": ("voteSign", "signed-edge"),
}
_IMMUTABLE_BEST_NAME = re.compile(r"^\..+\.run-[0-9a-f]{16}\.step-(?P<step>[0-9]{10})\.pt$")


def _acceptance_publication_seam(_path: Path) -> None:
    return


_ACCEPTANCE_PUBLICATION_SEAM = _acceptance_publication_seam


def _training_config_from_mapping(payload: dict[str, Any]) -> TrainingConfig:
    normalized = dict(payload)
    normalized["fanout"] = tuple(normalized.get("fanout", ()))
    if normalized.get("alignment_source_scores") is not None:
        normalized["alignment_source_scores"] = tuple(normalized["alignment_source_scores"])
    return TrainingConfig(**normalized)


_GATE_ORDER = (
    "formal-preflight",
    "matrix-completeness",
    "aggregate-promotability",
    "raw-ledger",
    "artifact-revalidation",
    "transfer-advantage",
    "candidate",
    "best-checkpoint",
    "checkpoint-reload",
    "head-report",
    "calibration",
    "fresh-process",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


class CandidateTaskEvidence(_StrictModel):
    task_id: str = Field(alias="taskId", min_length=1)
    cell_id: str = Field(alias="cellId", pattern=_HASH)
    record_hash: str = Field(alias="recordHash", pattern=_HASH)
    recipe_hash: str = Field(alias="recipeHash", pattern=_HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH)
    split_inventory_hash: str = Field(alias="splitInventoryHash", pattern=_HASH)
    adapter_domain: str = Field(alias="adapterDomain", min_length=1)
    supervised_data_hash: str = Field(alias="supervisedDataHash", pattern=_HASH)
    head_report_hash: str = Field(alias="headReportHash", pattern=_HASH)
    calibration_hash: str | None = Field(default=None, alias="calibrationHash", pattern=_HASH)
    evidence_hash: str = Field(alias="evidenceHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_evidence(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        )
        if self.evidence_hash != expected:
            raise ValueError("candidate task evidenceHash does not match task evidence")
        return self


class CandidateTrainingInventory(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-candidate-training-inventory/1.0"] = Field(
        alias="schemaVersion"
    )
    tasks: tuple[CandidateTaskEvidence, ...] = Field(strict=False)
    inventory_hash: str = Field(alias="inventoryHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_inventory(self):
        task_ids = tuple(item.task_id for item in self.tasks)
        if task_ids != _REQUIRED_GOVERNANCE_TASKS:
            raise ValueError("candidate training inventory must bind all four governance tasks")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"inventory_hash"})
        )
        if self.inventory_hash != expected:
            raise ValueError("candidate training inventoryHash does not match task evidence")
        return self


class CandidateExecutionEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-candidate-execution/1.0"] = Field(
        alias="schemaVersion"
    )
    method_id: Literal["multi-graph-shared-gfm", "domain-aligned-gfm"] = Field(alias="methodId")
    seed: int
    label_budget: Literal["full"] = Field(alias="labelBudget")
    trainer_config: dict[str, Any] = Field(alias="trainerConfig")
    task_cell_ids: tuple[str, ...] = Field(alias="taskCellIds", strict=False)
    recipe_hashes: tuple[str, ...] = Field(alias="recipeHashes", strict=False)
    source_record_hashes: tuple[str, ...] = Field(alias="sourceRecordHashes", strict=False)
    winner_selection_hash: str | None = Field(
        default=None, alias="winnerSelectionHash", pattern=_HASH
    )
    config_hash: str = Field(alias="configHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_execution(self):
        config = _training_config_from_mapping(self.trainer_config)
        if config.preset != "formal" or canonical_json(config.to_dict()) != canonical_json(
            self.trainer_config
        ):
            raise ValueError("candidate execution requires one canonical formal trainer config")
        for values in (self.task_cell_ids, self.recipe_hashes, self.source_record_hashes):
            if len(values) != len(_REQUIRED_GOVERNANCE_TASKS) or len(set(values)) != len(values):
                raise ValueError("candidate execution requires four unique task bindings")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"config_hash"})
        )
        if self.config_hash != expected:
            raise ValueError("candidate configHash does not match execution evidence")
        return self


class CandidateGovernanceManifest(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-governance-candidate/1.0"] = Field(
        alias="schemaVersion"
    )
    protocol_hash: str = Field(alias="protocolHash", pattern=_HASH)
    execution: CandidateExecutionEvidence
    training_inventory: CandidateTrainingInventory = Field(alias="trainingInventory")
    latest_checkpoint: ExperimentArtifactRef = Field(alias="latestCheckpoint")
    best_checkpoint: ExperimentArtifactRef = Field(alias="bestCheckpoint")
    encoder_source_cell_id: str = Field(alias="encoderSourceCellId", pattern=_HASH)
    encoder_source_best_checkpoint_sha256: str = Field(
        alias="encoderSourceBestCheckpointSha256", pattern=_HASH
    )
    code_hash: str = Field(alias="codeHash", pattern=_HASH)
    environment_hash: str = Field(alias="environmentHash", pattern=_HASH)
    manifest_hash: str = Field(alias="manifestHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_manifest(self):
        if self.protocol_hash != ExperimentProtocol.fixed().protocol_hash:
            raise ValueError("candidate manifest protocol differs from the fixed experiment")
        if (
            self.latest_checkpoint.role != "latest-checkpoint"
            or self.best_checkpoint.role != "best-checkpoint"
            or self.latest_checkpoint.relative_path == self.best_checkpoint.relative_path
            or self.latest_checkpoint.semantic_hash != self.latest_checkpoint.byte_sha256
            or self.best_checkpoint.semantic_hash != self.best_checkpoint.byte_sha256
        ):
            raise ValueError("candidate manifest checkpoint inventory is invalid")
        if (
            self.execution.task_cell_ids
            != tuple(item.cell_id for item in self.training_inventory.tasks)
            or self.execution.recipe_hashes
            != tuple(item.recipe_hash for item in self.training_inventory.tasks)
            or self.execution.source_record_hashes
            != tuple(item.record_hash for item in self.training_inventory.tasks)
        ):
            raise ValueError("candidate execution and training inventory task order differ")
        if self.encoder_source_cell_id not in self.execution.task_cell_ids:
            raise ValueError("candidate encoder source is not one of the winning full-budget cells")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"manifest_hash"})
        )
        if self.manifest_hash != expected:
            raise ValueError("candidate manifestHash does not match governance candidate")
        return self


@dataclass(frozen=True)
class StrictCheckpointModel:
    model: CoreGFM
    adapter_schemas: dict[str, AdapterSchema]
    adapter_states: dict[str, dict[str, Any]]
    model_state_hash: str


@dataclass(frozen=True)
class ReproducedRawValidation:
    validation_metric: float
    head_report_hash: str
    validation_partition_hash: str
    validation_protocol_hash: str
    validation_data_hash: str
    validation_callback_hash: str
    validation_scores: tuple[float, ...]
    validation_targets: tuple[float, ...]
    calibration_hash: str | None


@dataclass(frozen=True)
class VerifiedCellFoldEvaluation:
    metrics: TaskMetricSet
    validation_metric: float
    head_report_inventory_hash: str
    best_checkpoint_inventory_hash: str
    validation_partition_inventory_hash: str
    validation_protocol_inventory_hash: str
    validation_data_inventory_hash: str
    validation_callback_inventory_hash: str


def _strict_core_gfm_from_checkpoint(payload: dict[str, Any]) -> StrictCheckpointModel:
    """Instantiate the exact production classes and reject partial/renamed state inventories."""

    try:
        trainer = payload["trainer"]
        state = trainer["model"]
        node_weight = state["node_head.weight"]
        if not hasattr(node_weight, "ndim") or node_weight.ndim != 2 or node_weight.shape[1] != 128:
            raise ValueError
        model = CoreGFM(node_classes=int(node_weight.shape[0]))
        model.load_state_dict(state, strict=True)
        serialized_schemas = trainer["adapterSchemas"]
        adapter_states = trainer["adapters"]
        if (
            not isinstance(serialized_schemas, dict)
            or not isinstance(adapter_states, dict)
            or not serialized_schemas
            or set(serialized_schemas) != set(adapter_states)
        ):
            raise ValueError
        schemas: dict[str, AdapterSchema] = {}
        normalized_states: dict[str, dict[str, Any]] = {}
        for domain in sorted(serialized_schemas):
            schema = AdapterSchema.model_validate_json(canonical_json(serialized_schemas[domain]))
            raw_state = adapter_states[domain]
            if not isinstance(raw_state, dict):
                raise ValueError
            parameter_module = AdapterParameterModule(schema)
            parameter_module.load_state_dict(raw_state, strict=True)
            schemas[domain] = schema
            normalized_states[domain] = raw_state
    except (KeyError, TypeError, RuntimeError, ValueError) as error:
        raise ValueError(
            "candidate checkpoint must strict-load a complete CoreGFM and adapter inventory"
        ) from error
    return StrictCheckpointModel(
        model=model,
        adapter_schemas=schemas,
        adapter_states=normalized_states,
        model_state_hash=_model_state_hash(state),
    )


def _strict_raw_gfm_checkpoint(
    payload: dict[str, Any],
    *,
    expected_domains: tuple[str, ...],
) -> StrictCheckpointModel:
    """Strict-load a formal raw GFM cell and its exact recipe adapter inventory."""

    if (
        not expected_domains
        or len(expected_domains) != len(set(expected_domains))
        or any(not domain for domain in expected_domains)
    ):
        raise ValueError("formal raw GFM checkpoint domain inventory is invalid")
    try:
        observed = _strict_core_gfm_from_checkpoint(payload)
    except ValueError as error:
        raise ValueError(
            "formal raw GFM checkpoint must contain a complete CoreGFM and adapters"
        ) from error
    if set(observed.adapter_schemas) != set(expected_domains):
        raise ValueError("formal raw GFM checkpoint adapter domains differ from its recipe")
    return observed


def _adapter_inventory_hash(checkpoint: StrictCheckpointModel) -> str:
    return canonical_sha256(
        {
            domain: {
                "adapterSchemaHash": checkpoint.adapter_schemas[domain].adapter_schema_hash,
                "state": {
                    name: canonical_tensor_digest(value)
                    for name, value in sorted(checkpoint.adapter_states[domain].items())
                },
            }
            for domain in sorted(checkpoint.adapter_schemas)
        }
    )


def _reproduce_raw_gfm_validation(
    *,
    checkpoint: StrictCheckpointModel,
    target_domain: str,
    bundle: CoreGraphBundle,
    adapter_schema: AdapterSchema,
    head_data: SupervisedTrainValidation,
    head_report: HeadTrainingReport,
    calibration: CalibrationFitReport | None,
) -> ReproducedRawValidation:
    """Reproduce a raw GFM cell's validation head and calibration from live state."""

    schema = checkpoint.adapter_schemas.get(target_domain)
    state = checkpoint.adapter_states.get(target_domain)
    if schema is None or state is None or schema != adapter_schema:
        raise ValueError("raw validation target adapter differs from the checkpoint")
    adapter = BundleInputAdapter(bundle, mode="training", schema=schema)
    adapter.load_state_dict(state, strict=True)
    encoded = encode_supervised_graph(checkpoint.model, bundle, adapter)
    verified_head = _new_verified_head_training_report(head_report)
    try:
        verify_head_training_report(checkpoint.model, encoded, head_data, verified_head)
    except (TypeError, ValueError) as error:
        raise ValueError("raw head training report is not reproduced from live state") from error
    with torch.no_grad():
        validation_loss = _supervised_loss(
            checkpoint.model,
            encoded.tensor,
            head_data.task_kind,
            head_data.validation,
        )
    validation_metric = -float(validation_loss.detach().cpu())
    if validation_metric != head_report.best_metric:
        raise ValueError("raw head best metric is not reproduced from validation data")

    requires_calibration = head_data.task_kind != "resilience-regression"
    if requires_calibration:
        if calibration is None:
            raise ValueError("raw binary head lacks validation calibration evidence")
        semantics = BinaryScoreSemantics.for_task(head_data.task_kind)
        scores = derive_validation_scores(
            checkpoint.model,
            encoded,
            head_data,
            verified_head,
            semantics=semantics,
        )
        protocol = CalibrationProtocol.fixed(scores)
        if fit_score_calibration_report(scores, protocol=protocol) != calibration:
            raise ValueError("raw calibration is not reproduced from validation scores")
        validation_scores = tuple(float(value) for value in scores.logits.tolist())
        validation_targets = tuple(float(value) for value in scores.targets.tolist())
        calibration_hash: str | None = calibration.report_hash
    else:
        if calibration is not None:
            raise ValueError("raw resilience head cannot carry binary calibration")
        validation_scores = ()
        validation_targets = ()
        calibration_hash = None
    return ReproducedRawValidation(
        validation_metric=validation_metric,
        head_report_hash=head_report.report_hash,
        validation_partition_hash=head_report.validation_partition_hash,
        validation_protocol_hash=head_report.config_hash,
        validation_data_hash=head_report.data_hash,
        validation_callback_hash=canonical_sha256(
            {
                "callback": "socialgraph-gfm.core.supervised-loss/1.0",
                "taskKind": head_data.task_kind,
                "splitEvidenceHash": head_report.split_evidence_hash,
            }
        ),
        validation_scores=validation_scores,
        validation_targets=validation_targets,
        calibration_hash=calibration_hash,
    )


def _verify_fold_checkpoint_pair(
    *,
    root: Path,
    record: ExperimentRunRecord,
    binding: FoldEvaluationBinding,
    recipe: ExperimentRecipeEvidence,
    execution_config: ExperimentExecutionConfigEvidence,
    telemetry: ResourceTelemetryRecord,
    bundle: CoreGraphBundle,
) -> tuple[StrictCheckpointModel, Any, str, str]:
    refs = _fold_artifact_map(binding)
    checkpoint_bindings = CheckpointBindings(
        config_hash=record.config_hash,
        data_hash=binding.fold_data_hash,
        code_hash=record.code_hash,
        environment_hash=record.environment_hash,
    )
    latest_bytes = _read_fold_ref(root, refs["latest-checkpoint"])
    best_bytes = _read_fold_ref(root, refs["best-checkpoint"])
    latest = load_checkpoint(latest_bytes, expected_bindings=checkpoint_bindings)
    best = load_checkpoint(best_bytes, expected_bindings=checkpoint_bindings)
    if binding.adapter_domain is None:
        raise ValueError("core-gfm fold lacks its target adapter domain")
    if execution_config.training_config is None:
        raise ValueError("fold checkpoint lacks a formal training configuration")
    config = _training_config_from_mapping(execution_config.training_config)
    _latest_recovery = verify_fold_recovery_state(
        latest["trainer"],
        bundle=bundle,
        adapter_domain=binding.adapter_domain,
        config=config,
        expected_seed=record.cell.seed,
        expected_cell_id=record.cell.cell_id,
        expected_fold_id=binding.fold_id,
    )
    best_recovery = verify_fold_recovery_state(
        best["trainer"],
        bundle=bundle,
        adapter_domain=binding.adapter_domain,
        config=config,
        expected_seed=record.cell.seed,
        expected_cell_id=record.cell.cell_id,
        expected_fold_id=binding.fold_id,
    )
    best_model = StrictCheckpointModel(
        model=best_recovery.model,
        adapter_schemas={binding.adapter_domain: best_recovery.adapter.schema},
        adapter_states={binding.adapter_domain: best_recovery.adapter.state_dict()},
        model_state_hash=_model_state_hash(best_recovery.model.state_dict()),
    )
    latest_step = int(latest["trainer"].get("optimizerStep", -1))
    best_step = int(best["trainer"].get("optimizerStep", -1))
    latest_fit = _parse_fit_state(
        latest["trainer"].get("fitState"),
        optimizer_step=latest_step,
        config=config,
        checkpoint_model_state_hash=_model_state_hash(latest["trainer"].get("model", {})),
    )
    best_fit = _parse_fit_state(
        best["trainer"].get("fitState"),
        optimizer_step=best_step,
        config=config,
        checkpoint_model_state_hash=_model_state_hash(best["trainer"].get("model", {})),
    )
    best_name = Path(refs["best-checkpoint"].relative_path).name
    match = _IMMUTABLE_BEST_NAME.fullmatch(best_name)
    if (
        latest.get("status") != "validated"
        or latest.get("promotable") is not True
        or best.get("status") != "validated"
        or best.get("promotable") is not False
        or latest["trainer"].get("experimentCellId") != record.cell.cell_id
        or best["trainer"].get("experimentCellId") != record.cell.cell_id
        or latest["trainer"].get("evaluationFoldId") != binding.fold_id
        or best["trainer"].get("evaluationFoldId") != binding.fold_id
        or _training_config_from_mapping(latest["trainer"].get("config", {})) != config
        or _training_config_from_mapping(best["trainer"].get("config", {})) != config
        or match is None
        or int(match.group("step")) != best_step
        or latest_fit.best_checkpoint_name != best_name
        or latest_fit.best_checkpoint_sha256 != refs["best-checkpoint"].byte_sha256
        or latest_fit.best_step != best_step
        or latest_fit.best_metric != best_fit.best_metric
        or latest_fit.best_model_state_hash != best_fit.best_model_state_hash
        or latest_fit.validation_protocol_hash != best_fit.validation_protocol_hash
        or latest_fit.validation_data_hash != best_fit.validation_data_hash
        or latest_fit.validation_partition_hash != best_fit.validation_partition_hash
        or latest_fit.validation_callback_hash != best_fit.validation_callback_hash
        or best_fit.best_step != best_step
        or best_fit.best_checkpoint_name is not None
        or best_fit.best_checkpoint_sha256 is not None
        or best_fit.best_model_state_hash != best_model.model_state_hash
        or best_fit.last_model_state_hash != best_model.model_state_hash
        or telemetry.cell_id != record.cell.cell_id
        or telemetry.phase != "formal"
        or telemetry.config_hash != record.config_hash
        or telemetry.data_hash != binding.fold_data_hash
        or telemetry.code_hash != record.code_hash
        or telemetry.environment_hash != record.environment_hash
        or telemetry.latest_checkpoint_semantic_hash != refs["latest-checkpoint"].semantic_hash
        or telemetry.best_checkpoint_semantic_hash != refs["best-checkpoint"].semantic_hash
        or telemetry.final_optimizer_step != latest_step
        or telemetry.final_model_state_hash != _model_state_hash(latest["trainer"].get("model", {}))
        or telemetry.final_fit_state_hash != latest["trainer"].get("fitState", {}).get("stateHash")
    ):
        raise ValueError("fold latest/best checkpoint or telemetry evidence is inconsistent")
    pair_composite_hash = canonical_sha256(
        {
            "latest": _latest_recovery.composite_state_hash,
            "best": best_recovery.composite_state_hash,
        }
    )
    pair_recovery_hash = canonical_sha256(
        {
            "latest": _latest_recovery.recovery_state_hash,
            "best": best_recovery.recovery_state_hash,
        }
    )
    return (
        best_model,
        best_fit,
        pair_composite_hash,
        pair_recovery_hash,
    )


def _verify_cell_gfm_fold_evaluations(
    *,
    root: Path,
    record: ExperimentRunRecord,
    inventory: CellFoldEvaluationInventory,
    manifest: ExperimentDatasetManifest,
    split_inventory: ExperimentSplitInventory,
    labels: ExperimentLabels,
    bundle: CoreGraphBundle,
    recipe: ExperimentRecipeEvidence,
    execution_config: ExperimentExecutionConfigEvidence,
    telemetry_policy: TrustedTelemetryPolicy,
) -> VerifiedCellFoldEvaluation:
    target = _FOLD_TARGETS.get(record.cell.task_id)
    if target is None:
        raise ValueError("formal core-gfm fold evaluator is unavailable for this task")
    target_name, task_kind = target
    if (
        inventory.cell_id != record.cell.cell_id
        or inventory.task_id != record.cell.task_id
        or inventory.dataset_manifest_hash != manifest.manifest_hash
        or inventory.split_inventory_hash != split_inventory.inventory_hash
        or inventory.labels_hash != labels.labels_hash
        or inventory.target_name != target_name
        or inventory.fold_ids != tuple(item.fold_id for item in split_inventory.folds)
        or tuple(item.fold_id for item in inventory.folds) != inventory.fold_ids
        or any(item.runtime_kind != "core-gfm" for item in inventory.folds)
        or any(item.task_kind != task_kind for item in inventory.folds)
    ):
        raise ValueError("cell fold evaluation inventory differs from formal corpus evidence")

    metric_inputs: list[FoldMetricInput] = []
    reproduced_validations: list[ReproducedRawValidation] = []
    for fold, binding in zip(split_inventory.folds, inventory.folds, strict=True):
        if (
            binding.split_manifest_hash != fold.split_manifest_hash
            or binding.adapter_domain != f"{record.cell.target_graph_id}::{fold.fold_id}"
        ):
            raise ValueError("fold runtime binding differs from the official split")
        prepared = prepare_authoritative_fold(bundle, fold)
        if binding.prepared_graph_version_hash != prepared.bundle.graph_version_hash:
            raise ValueError("fold prepared graph version differs from live derivation")
        bound = bind_authoritative_fold_test(
            prepared,
            labels,
            target_name=target_name,
            task_kind=task_kind,  # type: ignore[arg-type]
        )
        authoritative = derive_authoritative_fold_train_validation(
            prepared,
            labels,
            target_name=target_name,
            task_kind=task_kind,  # type: ignore[arg-type]
            label_budget=record.cell.label_budget,
            experiment_seed=record.cell.seed,
        )
        authoritative_record = verify_authoritative_fold_train_validation(authoritative)
        refs = _fold_artifact_map(binding)
        head_data = _load_canonical_model(
            _read_fold_ref(root, refs["head-data"]),
            SupervisedTrainValidation,
            label="fold supervised head data",
        )
        expected_fold_data_hash = canonical_sha256(
            {
                "cellId": record.cell.cell_id,
                "foldId": fold.fold_id,
                "preparedGraphVersionHash": prepared.bundle.graph_version_hash,
                "headDataHash": head_data.data_hash,
            }
        )
        if (
            refs["head-data"].semantic_hash != head_data.data_hash
            or binding.fold_data_hash != expected_fold_data_hash
            or head_data.train != authoritative_record.train
            or head_data.validation != authoritative_record.validation
        ):
            raise ValueError("fold training data identity differs from the prepared graph")
        telemetry = _load_canonical_model(
            _read_fold_ref(root, refs["resource-telemetry"]),
            ResourceTelemetryRecord,
            label="fold resource telemetry",
        )
        if refs["resource-telemetry"].semantic_hash != telemetry.telemetry_hash:
            raise ValueError("fold telemetry semantic hash changed")
        checkpoint, best_fit, composite_hash, recovery_hash = _verify_fold_checkpoint_pair(
            root=root,
            record=record,
            binding=binding,
            recipe=recipe,
            execution_config=execution_config,
            telemetry=telemetry,
            bundle=prepared.bundle,
        )
        adapter_domain = binding.adapter_domain
        if adapter_domain is None:
            raise ValueError("fold checkpoint lacks its prepared target adapter")
        expected_adapter = BundleInputAdapter(prepared.bundle, mode="training")
        schema = checkpoint.adapter_schemas.get(adapter_domain)
        state = checkpoint.adapter_states.get(adapter_domain)
        if schema != expected_adapter.schema or state is None:
            raise ValueError("fold checkpoint adapter schema differs from train-only transforms")
        expected_adapter.load_state_dict(state, strict=True)
        authoritative_data = bind_authoritative_supervised_train_validation(
            authoritative,
            encode_supervised_graph(checkpoint.model, prepared.bundle, expected_adapter).provenance,
        )
        if head_data != authoritative_data:
            raise ValueError("fold head data differs from authoritative roles, labels, or budget")

        receipt = _load_canonical_model(
            _read_fold_ref(root, refs["telemetry-receipt"]),
            TelemetryReceipt,
            label="fold telemetry receipt",
        )
        if refs["telemetry-receipt"].semantic_hash != receipt.receipt_hash:
            raise ValueError("fold telemetry receipt semantic hash changed")
        telemetry_policy.verify(
            record=telemetry,
            receipt=receipt,
            expected=TelemetryReceiptExpectations(
                cellId=record.cell.cell_id,
                foldId=binding.fold_id,
                runId=telemetry.run_id,
                configHash=record.config_hash,
                dataHash=binding.fold_data_hash,
                codeHash=record.code_hash,
                environmentHash=record.environment_hash,
                telemetryRecordHash=canonical_sha256(
                    telemetry.model_dump(mode="python", by_alias=True)
                ),
                latestCheckpointSemanticHash=refs["latest-checkpoint"].semantic_hash,
                bestCheckpointSemanticHash=refs["best-checkpoint"].semantic_hash,
                finalModelStateHash=telemetry.final_model_state_hash,
                finalFitStateHash=telemetry.final_fit_state_hash,
                compositeStateHash=composite_hash,
                recoveryStateHash=recovery_hash,
            ),
        )

        head_report = _load_canonical_model(
            _read_fold_ref(root, refs["head-report"]),
            HeadTrainingReport,
            label="fold head report",
        )
        if refs["head-report"].semantic_hash != head_report.report_hash:
            raise ValueError("fold head report semantic hash changed")
        calibration: CalibrationFitReport | None = None
        if "calibration-report" in refs:
            calibration = _load_canonical_model(
                _read_fold_ref(root, refs["calibration-report"]),
                CalibrationFitReport,
                label="fold calibration report",
            )
            if refs["calibration-report"].semantic_hash != calibration.report_hash:
                raise ValueError("fold calibration semantic hash changed")
        reproduced = _reproduce_raw_gfm_validation(
            checkpoint=checkpoint,
            target_domain=adapter_domain,
            bundle=prepared.bundle,
            adapter_schema=expected_adapter.schema,
            head_data=head_data,
            head_report=head_report,
            calibration=calibration,
        )
        reproduced_validations.append(reproduced)
        if (
            best_fit.best_metric != reproduced.validation_metric
            or best_fit.last_validation_metric != reproduced.validation_metric
            or best_fit.validation_protocol_hash != reproduced.validation_protocol_hash
            or best_fit.validation_data_hash != reproduced.validation_data_hash
            or best_fit.validation_partition_hash != reproduced.validation_partition_hash
            or best_fit.validation_callback_hash != reproduced.validation_callback_hash
        ):
            raise ValueError("fold best checkpoint is not bound to live validation evidence")

        live_predictions = infer_core_gfm_fold(checkpoint.model, expected_adapter, bound)
        persisted_predictions = _load_canonical_model(
            _read_fold_ref(root, refs["predictions"]),
            FoldPredictionRecord,
            label="fold prediction audit",
        )
        if (
            refs["predictions"].semantic_hash != live_predictions.record.prediction_hash
            or persisted_predictions != live_predictions.record
        ):
            raise ValueError("persisted fold predictions differ from live checkpoint inference")

        threshold = None
        evaluation_scores = live_predictions.record.scores
        evaluation_probabilities: tuple[float, ...] = ()
        if task_kind in {"node-binary", "signed-edge"}:
            if calibration is None or "threshold" not in refs:
                raise ValueError("classification fold lacks calibration or threshold evidence")
            evaluation_scores = _calibrated_probabilities(
                live_predictions.record.scores, calibration
            )
            evaluation_probabilities = evaluation_scores
            selection = _load_canonical_model(
                _read_fold_ref(root, refs["threshold"]),
                ThresholdSelectionEvidence,
                label="fold threshold selection",
            )
            validation_probabilities = _calibrated_probabilities(
                reproduced.validation_scores, calibration
            )
            if (
                refs["threshold"].semantic_hash != selection.evidence_hash
                or selection.validation_scores != validation_probabilities
                or selection.validation_targets != reproduced.validation_targets
                or selection.threshold.validation_partition_hash
                != head_data.validation.partition_hash
            ):
                raise ValueError("fold threshold is not selected from live validation evidence")
            threshold = selection.threshold
        metric_inputs.append(
            FoldMetricInput(
                prediction=live_predictions.record,
                evaluation_scores=evaluation_scores,
                evaluation_probabilities=evaluation_probabilities,
                threshold=threshold,
            )
        )
    metrics = derive_equal_weight_fold_metrics(
        task_id=record.cell.task_id,
        folds=tuple(metric_inputs),
    )
    return VerifiedCellFoldEvaluation(
        metrics=metrics,
        validation_metric=math.fsum(item.validation_metric for item in reproduced_validations)
        / len(reproduced_validations),
        head_report_inventory_hash=canonical_sha256(
            [item.head_report_hash for item in reproduced_validations]
        ),
        best_checkpoint_inventory_hash=canonical_sha256(
            [_fold_artifact_map(item)["best-checkpoint"].byte_sha256 for item in inventory.folds]
        ),
        validation_partition_inventory_hash=canonical_sha256(
            [item.validation_partition_hash for item in reproduced_validations]
        ),
        validation_protocol_inventory_hash=canonical_sha256(
            [item.validation_protocol_hash for item in reproduced_validations]
        ),
        validation_data_inventory_hash=canonical_sha256(
            [item.validation_data_hash for item in reproduced_validations]
        ),
        validation_callback_inventory_hash=canonical_sha256(
            [item.validation_callback_hash for item in reproduced_validations]
        ),
    )


class FreshProcessCheckpointEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-fresh-checkpoint-probe/1.0"] = Field(
        alias="schemaVersion"
    )
    latest_checkpoint_sha256: str = Field(alias="latestCheckpointSha256", pattern=_HASH)
    best_checkpoint_sha256: str = Field(alias="bestCheckpointSha256", pattern=_HASH)
    latest_status: Literal["validated"] = Field(alias="latestStatus")
    latest_promotable: Literal[True] = Field(alias="latestPromotable")
    best_status: Literal["validated"] = Field(alias="bestStatus")
    best_promotable: Literal[False] = Field(alias="bestPromotable")
    latest_optimizer_step: int = Field(alias="latestOptimizerStep", ge=1, le=10_000)
    best_optimizer_step: int = Field(alias="bestOptimizerStep", ge=1, le=10_000)
    best_checkpoint_name: str = Field(alias="bestCheckpointName", min_length=1)
    latest_model_state_hash: str = Field(alias="latestModelStateHash", pattern=_HASH)
    best_model_state_hash: str = Field(alias="bestModelStateHash", pattern=_HASH)
    best_adapter_inventory_hash: str = Field(alias="bestAdapterInventoryHash", pattern=_HASH)
    evidence_hash: str = Field(alias="evidenceHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_evidence(self):
        if Path(self.best_checkpoint_name).name != self.best_checkpoint_name:
            raise ValueError("fresh-process best checkpoint name must be a basename")
        match = _IMMUTABLE_BEST_NAME.fullmatch(self.best_checkpoint_name)
        if match is None or int(match.group("step")) != self.best_optimizer_step:
            raise ValueError("fresh-process best checkpoint name is not immutable and versioned")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        )
        if self.evidence_hash != expected:
            raise ValueError("fresh-process evidenceHash does not match observed bytes")
        return self


class CoreAcceptance(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-acceptance/1.1"] = Field(
        alias="schemaVersion"
    )
    protocol_hash: str = Field(alias="protocolHash", pattern=_HASH)
    preflight_evidence_hash: str | None = Field(
        default=None, alias="preflightEvidenceHash", pattern=_HASH
    )
    aggregate_hashes: tuple[str, ...] = Field(alias="aggregateHashes", strict=False)
    transfer_decision_hashes: tuple[str, ...] = Field(alias="transferDecisionHashes", strict=False)
    raw_record_hashes: tuple[str, ...] = Field(alias="rawRecordHashes", strict=False)
    winner_selection_hash: str | None = Field(
        default=None, alias="winnerSelectionHash", pattern=_HASH
    )
    candidate_cell_id: str | None = Field(default=None, alias="candidateCellId", pattern=_HASH)
    candidate_record_hash: str | None = Field(
        default=None, alias="candidateRecordHash", pattern=_HASH
    )
    candidate_latest_checkpoint_sha256: str | None = Field(
        default=None, alias="candidateLatestCheckpointSha256", pattern=_HASH
    )
    candidate_checkpoint_sha256: str | None = Field(
        default=None, alias="candidateCheckpointSha256", pattern=_HASH
    )
    candidate_manifest_hash: str | None = Field(
        default=None, alias="candidateManifestHash", pattern=_HASH
    )
    candidate_task_evidence_hashes: tuple[str, ...] = Field(
        alias="candidateTaskEvidenceHashes", strict=False
    )
    fresh_process_evidence_hash: str | None = Field(
        default=None, alias="freshProcessEvidenceHash", pattern=_HASH
    )
    failed_gates: tuple[str, ...] = Field(alias="failedGates", strict=False)
    status: Literal["accepted", "rejected"]
    accepted: bool
    promotable: bool
    acceptance_hash: str = Field(alias="acceptanceHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_acceptance(self):
        if self.protocol_hash != ExperimentProtocol.fixed().protocol_hash:
            raise ValueError("acceptance protocol is not the fixed formal protocol")
        if (
            len(self.failed_gates) != len(set(self.failed_gates))
            or any(gate not in _GATE_ORDER for gate in self.failed_gates)
            or tuple(gate for gate in _GATE_ORDER if gate in self.failed_gates) != self.failed_gates
        ):
            raise ValueError("acceptance failed gates must use fixed unique order")
        accepted = not self.failed_gates
        if (
            self.accepted != accepted
            or self.promotable != accepted
            or self.status != ("accepted" if accepted else "rejected")
        ):
            raise ValueError("acceptance status must be derived from failed gates")
        required = (
            self.preflight_evidence_hash,
            self.winner_selection_hash,
            self.candidate_cell_id,
            self.candidate_record_hash,
            self.candidate_latest_checkpoint_sha256,
            self.candidate_checkpoint_sha256,
            self.candidate_manifest_hash,
            self.fresh_process_evidence_hash,
        )
        if accepted and any(value is None for value in required):
            raise ValueError("accepted report requires every candidate evidence hash")
        if accepted and len(self.candidate_task_evidence_hashes) != len(_REQUIRED_GOVERNANCE_TASKS):
            raise ValueError("accepted report requires all governance task evidence hashes")
        for values in (
            self.aggregate_hashes,
            self.transfer_decision_hashes,
            self.raw_record_hashes,
            self.candidate_task_evidence_hashes,
        ):
            if values != tuple(sorted(set(values))):
                raise ValueError("acceptance evidence hash inventories must be sorted and unique")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"acceptance_hash"})
        )
        if self.acceptance_hash != expected:
            raise ValueError("acceptanceHash does not match acceptance evidence")
        return self


ModelT = TypeVar("ModelT", bound=BaseModel)


def _canonical_bytes(model: BaseModel) -> bytes:
    return (canonical_json(model) + "\n").encode("utf-8")


def _read_ref(root: Path, ref: ExperimentArtifactRef) -> bytes:
    serialized = read_confined_snapshot(root, ref.relative_path, max_bytes=_MAX_ARTIFACT_BYTES)
    if len(serialized) != ref.size_bytes:
        raise ValueError(f"{ref.role} artifact size changed")
    if hashlib.sha256(serialized).hexdigest() != ref.byte_sha256:
        raise ValueError(f"{ref.role} artifact byte hash changed")
    return serialized


def _read_fold_ref(root: Path, ref: FoldRuntimeArtifactRef) -> bytes:
    serialized = read_confined_snapshot(root, ref.relative_path, max_bytes=_MAX_ARTIFACT_BYTES)
    if len(serialized) != ref.size_bytes:
        raise ValueError(f"fold {ref.role} artifact size changed")
    if hashlib.sha256(serialized).hexdigest() != ref.byte_sha256:
        raise ValueError(f"fold {ref.role} artifact byte hash changed")
    return serialized


def _fold_artifact_map(binding: FoldEvaluationBinding) -> dict[str, FoldRuntimeArtifactRef]:
    result: dict[str, FoldRuntimeArtifactRef] = {item.role: item for item in binding.artifacts}
    if len(result) != len(binding.artifacts):
        raise ValueError("fold artifact roles are not unique")
    return result


def _calibrated_probabilities(
    scores: tuple[float, ...], calibration: CalibrationFitReport
) -> tuple[float, ...]:
    temperature = calibration.calibration.temperature
    bias = calibration.calibration.bias
    probabilities: list[float] = []
    for score in scores:
        value = (score + bias) / temperature
        probability = (
            1.0 / (1.0 + math.exp(-value))
            if value >= 0.0
            else math.exp(value) / (1.0 + math.exp(value))
        )
        if not math.isfinite(probability):
            raise ValueError("calibrated fold probability is non-finite")
        probabilities.append(probability)
    return tuple(probabilities)


def _load_canonical_model(serialized: bytes, model_type: type[ModelT], *, label: str) -> ModelT:
    model = model_type.model_validate_json(serialized)
    if serialized != _canonical_bytes(model):
        raise ValueError(f"{label} artifact is not canonical JSON")
    return model


def _artifact_map(record: ExperimentRunRecord) -> dict[str, ExperimentArtifactRef]:
    result: dict[str, ExperimentArtifactRef] = {item.role: item for item in record.artifacts}
    if len(result) != len(record.artifacts):
        raise ValueError("experiment artifact roles are not unique")
    return result


def _expected_head_contract(task_id: str) -> tuple[str, str, bool]:
    if task_id in {"tolokers.risk"} or task_id.startswith("twitch."):
        return "node-binary", "node_head", True
    if task_id == "wiki-rfa.vote-sign":
        return "signed-edge", "signed_edge_head", True
    if task_id in {"github.relation-completion", "email.relation-completion"}:
        return "edge-binary", "binary_link_head", True
    if task_id == "penn94.gender-offline":
        return "node-multiclass", "node_head", False
    if task_id == "penn94.community-resilience":
        return "resilience-regression", "resilience_head", False
    raise ValueError("experiment task has no fixed supervised head contract")


def _recompute_metric_evidence(
    record: ExperimentRunRecord,
    serialized: dict[str, bytes],
) -> None:
    predictions = _load_canonical_model(
        serialized["predictions"], PredictionEvidence, label="prediction"
    )
    targets = _load_canonical_model(serialized["targets"], TargetEvidence, label="target")
    if (
        predictions.task_id != record.cell.task_id
        or targets.task_id != record.cell.task_id
        or predictions.evaluation_kind != targets.evaluation_kind
        or predictions.entity_ids != targets.entity_ids
        or predictions.entity_fold_ids != targets.entity_fold_ids
        or predictions.fold_ids != record.evaluation_fold_ids
        or targets.fold_ids != record.evaluation_fold_ids
        or predictions.split_inventory_hash != record.split_inventory_hash
        or targets.split_inventory_hash != record.split_inventory_hash
        or predictions.prediction_hash != record.metrics.prediction_hash
        or targets.target_hash != record.metrics.target_hash
    ):
        raise ValueError("metric inputs are not bound to the experiment task")

    threshold = None
    if "threshold" in serialized:
        selection = _load_canonical_model(
            serialized["threshold"],
            ThresholdSelectionEvidence,
            label="threshold selection",
        )
        threshold = selection.threshold
        if threshold.threshold_hash != record.metrics.threshold_hash:
            raise ValueError("test threshold differs from validation-only selection")
    elif record.metrics.threshold_hash is not None:
        raise ValueError("classification metric set lacks threshold evidence")

    labels = targets.values
    task_id = record.cell.task_id
    metrics: dict[str, float]
    if task_id == "tolokers.risk":
        if threshold is None or predictions.evaluation_kind != "binary":
            raise ValueError("Tolokers metrics require binary threshold evidence")
        point = binary_metrics_at_threshold(predictions.scores, labels, threshold=threshold)
        metrics = {
            "auprc": binary_auprc(predictions.scores, labels),
            "auroc": binary_auroc(predictions.scores, labels),
            "brier": binary_brier(predictions.probabilities, labels),
            "ece": binary_ece(predictions.probabilities, labels),
            "macroF1": point["macroF1"],
            "recallAtFpr": recall_at_fixed_fpr(predictions.scores, labels, max_fpr=0.10),
        }
    elif task_id == "wiki-rfa.vote-sign":
        if threshold is None or predictions.evaluation_kind != "binary":
            raise ValueError("signed-edge metrics require binary threshold evidence")
        point = binary_metrics_at_threshold(predictions.scores, labels, threshold=threshold)
        metrics = {
            "auroc": binary_auroc(predictions.scores, labels),
            "macroF1": point["macroF1"],
            "mcc": point["mcc"],
            "negativeAuprc": negative_class_auprc(predictions.scores, labels),
        }
    elif task_id == "penn94.gender-offline" or task_id.startswith("twitch."):
        if threshold is None or predictions.evaluation_kind != "binary":
            raise ValueError("node classification metrics require threshold evidence")
        point = binary_metrics_at_threshold(predictions.scores, labels, threshold=threshold)
        metrics = {
            ("rocAuc" if task_id == "penn94.gender-offline" else "auroc"): binary_auroc(
                predictions.scores, labels
            ),
            "macroF1": point["macroF1"],
        }
        if task_id == "penn94.gender-offline":
            metrics["accuracy"] = point["accuracy"]
        else:
            metrics["auprc"] = binary_auprc(predictions.scores, labels)
    elif task_id in {"github.relation-completion", "email.relation-completion"}:
        if predictions.evaluation_kind != "link-ranking" or threshold is not None:
            raise ValueError("link metrics require filtered rankings without a threshold")
        metrics = filtered_ranking_metrics(
            positive_scores=predictions.scores,
            filtered_negative_scores=predictions.filtered_negative_scores,
            hits_at=(10,),
        )
    elif task_id == "penn94.community-resilience":
        if predictions.evaluation_kind != "regression" or threshold is not None:
            raise ValueError("resilience metrics require regression evidence")
        metrics = {
            "mae": mean_absolute_error(predictions.scores, labels),
            "spearman": spearman_correlation(predictions.scores, labels),
        }
    else:
        raise ValueError("experiment task has no metric recomputation contract")
    recomputed = TaskMetricSet.create(
        task_id=task_id,
        metrics=metrics,
        prediction_hash=predictions.prediction_hash,
        target_hash=targets.target_hash,
        threshold_hash=None if threshold is None else threshold.threshold_hash,
    )
    if recomputed != record.metrics:
        raise ValueError("reported task metrics are not reproduced from held-out evidence")


@dataclass(frozen=True)
class ObservedRunArtifacts:
    manifest: ExperimentDatasetManifest
    split_inventory: ExperimentSplitInventory
    bundle: CoreGraphBundle
    adapter_schema: AdapterSchema
    recipe: ExperimentRecipeEvidence
    execution_config: ExperimentExecutionConfigEvidence
    training_data: ExperimentTrainingDataEvidence
    head_data: SupervisedTrainValidation | None
    head_report: HeadTrainingReport | None
    calibration: CalibrationFitReport | None
    raw_validation: ReproducedRawValidation | None
    fold_inventory: CellFoldEvaluationInventory
    fold_evaluation: VerifiedCellFoldEvaluation | None


@dataclass(frozen=True)
class VerifiedExperimentRun:
    record: ExperimentRunRecord
    artifacts: ObservedRunArtifacts


def _verify_record_artifacts(
    root: Path,
    record: ExperimentRunRecord,
    *,
    preflight: Any,
    telemetry_policy: TrustedTelemetryPolicy,
) -> ObservedRunArtifacts:
    refs = _artifact_map(record)
    serialized = {role: _read_ref(root, ref) for role, ref in refs.items()}

    manifest = _load_canonical_model(
        serialized["dataset-manifest"],
        ExperimentDatasetManifest,
        label="dataset manifest",
    )
    split = _load_canonical_model(
        serialized["split-manifest"], SplitManifest, label="split manifest"
    )
    labels = _load_canonical_model(serialized["labels"], ExperimentLabels, label="label")
    split_inventory = _load_canonical_model(
        serialized["split-inventory"],
        ExperimentSplitInventory,
        label="split inventory",
    )
    fold_inventory = _load_canonical_model(
        serialized["fold-evaluation-inventory"],
        CellFoldEvaluationInventory,
        label="cell fold evaluation inventory",
    )
    adapter = _load_canonical_model(
        serialized["adapter-schema"], AdapterSchema, label="adapter schema"
    )
    recipe = _load_canonical_model(
        serialized["experiment-recipe"],
        ExperimentRecipeEvidence,
        label="experiment recipe",
    )
    execution_config = _load_canonical_model(
        serialized["configuration"],
        ExperimentExecutionConfigEvidence,
        label="execution configuration",
    )
    training_data = _load_canonical_model(
        serialized["training-data"],
        ExperimentTrainingDataEvidence,
        label="training data inventory",
    )
    telemetry = _load_canonical_model(
        serialized["resource-telemetry"],
        ResourceTelemetryRecord,
        label="resource telemetry",
    )
    telemetry_receipt = _load_canonical_model(
        serialized["telemetry-receipt"],
        TelemetryReceipt,
        label="telemetry receipt",
    )
    structure_cache = _load_canonical_model(
        serialized["structure-cache"],
        StructureCacheManifest,
        label="structure cache manifest",
    )
    structure_ref_path = PurePosixPath(refs["structure-cache"].relative_path)
    if structure_ref_path.name != "manifest.json":
        raise ValueError("structure cache artifact must reference its canonical manifest")
    structure_npz_relative = (structure_ref_path.parent / "structure.npz").as_posix()
    structure_npz = read_confined_snapshot(
        root,
        structure_npz_relative,
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    if hashlib.sha256(structure_npz).hexdigest() != structure_cache.npz_sha256:
        raise ValueError("structure cache NPZ bytes differ from the manifest")
    manifest_ref = refs["dataset-manifest"]
    labels_ref = refs["labels"]
    split_inventory_ref = refs["split-inventory"]
    bundle_serialized = read_confined_snapshot(
        root,
        manifest.bundle_relative_path,
        max_bytes=_MAX_ARTIFACT_BYTES,
    )
    if hashlib.sha256(bundle_serialized).hexdigest() != manifest.bundle_sha256:
        raise ValueError("experiment bundle bytes differ from the dataset manifest")
    bundle = _load_canonical_model(bundle_serialized, CoreGraphBundle, label="graph bundle")
    split_hash = canonical_sha256(split.model_dump(mode="python", by_alias=True))
    if (
        manifest.manifest_hash != record.dataset_manifest_hash
        or manifest_ref.relative_path != manifest.manifest_relative_path
        or record.split_manifest_hash not in manifest.split_manifest_hashes
        or split_hash != record.split_manifest_hash
        or split_inventory.inventory_hash != record.split_inventory_hash
        or fold_inventory.inventory_hash != record.fold_evaluation_inventory_hash
        or refs["fold-evaluation-inventory"].semantic_hash != fold_inventory.inventory_hash
        or split_inventory_ref.relative_path != manifest.split_inventory_relative_path
        or split_inventory_ref.byte_sha256 != manifest.split_inventory_sha256
        or tuple(fold.fold_id for fold in split_inventory.folds) != manifest.split_ids
        or tuple(fold.split_manifest_hash for fold in split_inventory.folds)
        != manifest.split_manifest_hashes
        or record.evaluation_fold_ids != manifest.split_ids
        or labels.labels_hash != record.label_artifact_hash
        or labels.requirement_id != manifest.requirement_id
        or labels_ref.relative_path != manifest.labels_relative_path
        or labels_ref.byte_sha256 != manifest.labels_sha256
        or tuple(target.name for target in labels.targets) != manifest.label_names
        or adapter.adapter_schema_hash != record.adapter_schema_hash
        or adapter.source_graph_version_hash != bundle.graph_version_hash
        or bundle.graph_version_hash != manifest.graph_version_hash
        or bundle.source.source_sha256 != manifest.source_sha256
        or bundle.split_manifest != split_inventory.folds[0].split_manifest
        or structure_cache.manifest_hash != record.structure_cache_hash
        or structure_cache.enriched_graph_version_hash != bundle.graph_version_hash
        or structure_cache.split_manifest_hash != record.split_manifest_hash
        or structure_cache.fit_row_ids_hash != adapter.fit_row_ids_hash
        or structure_cache.fit_row_count != adapter.fit_row_count
        or structure_cache.visible_topology_hash != adapter.visible_topology_hash
        or structure_cache.visible_topology_edge_count != adapter.visible_topology_edge_count
        or structure_cache.config_hash != StructureAlgorithmConfig.fixed().config_hash
        or refs["code"].semantic_hash != refs["code"].byte_sha256
        or refs["environment"].semantic_hash != refs["environment"].byte_sha256
    ):
        raise ValueError("experiment semantic artifacts do not match the raw record")

    requirement_by_graph = {
        identifier: requirement
        for requirement in FORMAL_CORPUS_REQUIREMENTS
        for identifier in (requirement.graph_id, requirement.requirement_id)
    }
    observation_by_id = {
        observation.requirement_id: observation for observation in preflight.observations
    }
    requirement = requirement_by_graph.get(record.cell.target_graph_id)
    requirement_id = None if requirement is None else requirement.requirement_id
    observation = observation_by_id.get(requirement_id)
    if (
        requirement is None
        or requirement_id is None
        or observation is None
        or observation.status != "ready"
        or observation.manifest_hash != record.dataset_manifest_hash
        or observation.split_manifest_hash != manifest.split_manifest_hash
        or manifest.requirement_id != requirement_id
        or manifest.graph_id != requirement.graph_id
        or manifest.experiment_split_policy != requirement.experiment_split_policy
        or manifest.split_count != (requirement.official_split_count or 1)
    ):
        raise ValueError("experiment dataset is not bound to ready formal preflight evidence")

    manifest_hashes: dict[str, str] = {}
    for graph_id in (
        *record.cell.pretraining_graph_ids,
        record.cell.target_graph_id,
        *((record.cell.validation_graph_id,) if record.cell.validation_graph_id else ()),
    ):
        graph_requirement = requirement_by_graph.get(graph_id)
        graph_observation = (
            None
            if graph_requirement is None
            else observation_by_id.get(graph_requirement.requirement_id)
        )
        if (
            graph_requirement is None
            or graph_observation is None
            or graph_observation.status != "ready"
            or graph_observation.manifest_hash is None
        ):
            raise ValueError("experiment recipe source graph lacks ready formal manifest evidence")
        manifest_hashes[graph_id] = graph_observation.manifest_hash
    expected_recipe = ExperimentRecipeEvidence.create(
        cell=record.cell,
        manifest_hashes=manifest_hashes,
    )
    expected_training_data = ExperimentTrainingDataEvidence.create(
        cell=record.cell,
        recipe=expected_recipe,
        target_split_inventory_hash=record.split_inventory_hash,
        head_data_hash=record.head_data_hash,
    )
    if (
        recipe != expected_recipe
        or recipe.recipe_hash != record.recipe_hash
        or execution_config.cell_id != record.cell.cell_id
        or execution_config.method_id != record.cell.method_id
        or execution_config.backend != record.cell.backend
        or execution_config.recipe_hash != recipe.recipe_hash
        or execution_config.config_hash != record.config_hash
        or training_data != expected_training_data
        or training_data.inventory_hash != record.training_data_hash
        or telemetry.cell_id != record.cell.cell_id
        or telemetry.phase != record.phase
        or telemetry.config_hash != record.config_hash
        or telemetry.data_hash != record.training_data_hash
        or telemetry.code_hash != record.code_hash
        or telemetry.environment_hash != record.environment_hash
        or telemetry.telemetry_hash != record.telemetry_hash
        or telemetry.optimizer_steps != record.optimizer_steps
        or telemetry.elapsed_seconds != record.elapsed_seconds
        or telemetry.data_wait_seconds != record.data_wait_seconds
        or telemetry.peak_cuda_bytes != record.peak_cuda_bytes
        or telemetry_receipt.receipt_hash != record.telemetry_receipt_hash
        or refs["telemetry-receipt"].semantic_hash != telemetry_receipt.receipt_hash
    ):
        raise ValueError(
            "experiment recipe, configuration, training, or telemetry evidence drifted"
        )

    if not record.cell.trainable:
        if execution_config.training_config is not None or record.head_data_hash is not None:
            raise ValueError("heuristic experiment cannot claim supervised training evidence")
        raise ValueError("formal heuristic fold evaluator is unavailable")
    if execution_config.training_config is None or (
        _training_config_from_mapping(execution_config.training_config).preset != "formal"
    ):
        raise ValueError("formal trainable experiment lacks a formal trainer configuration")
    run_bindings = CheckpointBindings(
        config_hash=record.config_hash,
        data_hash=record.training_data_hash,
        code_hash=record.code_hash,
        environment_hash=record.environment_hash,
    )
    latest_ref = refs["latest-checkpoint"]
    best_ref = refs["best-checkpoint"]
    latest_payload = load_checkpoint(_read_ref(root, latest_ref), expected_bindings=run_bindings)
    best_payload = load_checkpoint(_read_ref(root, best_ref), expected_bindings=run_bindings)
    best_gfm: StrictCheckpointModel | None = None
    if record.cell.backend in {
        "scratch",
        "pretrained-single",
        "pretrained-shared",
        "pretrained-aligned",
    }:
        expected_domains = (
            *(item.graph_id for item in recipe.source_manifests),
            recipe.target_manifest.graph_id,
            *(
                (recipe.validation_manifest.graph_id,)
                if recipe.validation_manifest is not None
                else ()
            ),
        )
        latest_gfm = _strict_raw_gfm_checkpoint(
            latest_payload,
            expected_domains=expected_domains,
        )
        best_gfm = _strict_raw_gfm_checkpoint(
            best_payload,
            expected_domains=expected_domains,
        )
        target_domain = recipe.target_manifest.graph_id
        if (
            latest_gfm.adapter_schemas != best_gfm.adapter_schemas
            or latest_gfm.adapter_schemas[target_domain] != adapter
            or {
                domain: {
                    name: canonical_tensor_digest(value) for name, value in sorted(state.items())
                }
                for domain, state in latest_gfm.adapter_states.items()
            }
            != {
                domain: {
                    name: canonical_tensor_digest(value) for name, value in sorted(state.items())
                }
                for domain, state in best_gfm.adapter_states.items()
            }
        ):
            raise ValueError("formal raw GFM latest/best adapter inventory is inconsistent")
    elif record.cell.backend in {"attribute", "structure", "linkx"}:
        raise ValueError("formal baseline backend strict evaluator is unavailable")
    run_config = _training_config_from_mapping(execution_config.training_config)
    latest_step = int(latest_payload["trainer"].get("optimizerStep", -1))
    best_step = int(best_payload["trainer"].get("optimizerStep", -1))
    latest_fit = _parse_fit_state(
        latest_payload["trainer"].get("fitState"),
        optimizer_step=latest_step,
        config=run_config,
        checkpoint_model_state_hash=_model_state_hash(latest_payload["trainer"].get("model", {})),
    )
    best_fit = _parse_fit_state(
        best_payload["trainer"].get("fitState"),
        optimizer_step=best_step,
        config=run_config,
        checkpoint_model_state_hash=_model_state_hash(best_payload["trainer"].get("model", {})),
    )
    latest_recovery = verify_run_recovery_inventory(
        latest_payload["trainer"],
        config=run_config,
        expected_seed=record.cell.seed,
        expected_cell_id=record.cell.cell_id,
    )
    best_recovery = verify_run_recovery_inventory(
        best_payload["trainer"],
        config=run_config,
        expected_seed=record.cell.seed,
        expected_cell_id=record.cell.cell_id,
    )
    pair_composite_hash = canonical_sha256(
        {
            "latest": latest_recovery.composite_state_hash,
            "best": best_recovery.composite_state_hash,
        }
    )
    pair_recovery_hash = canonical_sha256(
        {
            "latest": latest_recovery.recovery_state_hash,
            "best": best_recovery.recovery_state_hash,
        }
    )
    best_name = Path(best_ref.relative_path).name
    best_name_match = _IMMUTABLE_BEST_NAME.fullmatch(best_name)
    if (
        latest_payload.get("status") != "validated"
        or latest_payload.get("promotable") is not True
        or best_payload.get("status") != "validated"
        or best_payload.get("promotable") is not False
        or latest_payload["trainer"].get("experimentCellId") != record.cell.cell_id
        or best_payload["trainer"].get("experimentCellId") != record.cell.cell_id
        or _training_config_from_mapping(latest_payload["trainer"].get("config", {})) != run_config
        or _training_config_from_mapping(best_payload["trainer"].get("config", {})) != run_config
        or latest_step != record.optimizer_steps
        or best_name_match is None
        or int(best_name_match.group("step")) != best_step
        or latest_fit.best_checkpoint_name != best_name
        or latest_fit.best_checkpoint_sha256 != best_ref.byte_sha256
        or latest_fit.best_step != best_step
        or latest_fit.best_metric != best_fit.best_metric
        or latest_fit.best_model_state_hash != best_fit.best_model_state_hash
        or latest_fit.validation_protocol_hash != best_fit.validation_protocol_hash
        or latest_fit.validation_data_hash != best_fit.validation_data_hash
        or latest_fit.validation_partition_hash != best_fit.validation_partition_hash
        or latest_fit.validation_callback_hash != best_fit.validation_callback_hash
        or best_fit.best_step != best_step
        or best_fit.best_checkpoint_name is not None
        or best_fit.best_checkpoint_sha256 is not None
        or telemetry.latest_checkpoint_semantic_hash != latest_ref.semantic_hash
        or telemetry.best_checkpoint_semantic_hash != best_ref.semantic_hash
        or telemetry.final_optimizer_step != latest_step
        or telemetry.final_model_state_hash
        != _model_state_hash(latest_payload["trainer"].get("model", {}))
        or telemetry.final_fit_state_hash
        != latest_payload["trainer"].get("fitState", {}).get("stateHash")
    ):
        raise ValueError("experiment latest/best checkpoint evidence is inconsistent")
    telemetry_policy.verify(
        record=telemetry,
        receipt=telemetry_receipt,
        expected=TelemetryReceiptExpectations(
            cellId=record.cell.cell_id,
            foldId="cell-run",
            runId=telemetry.run_id,
            configHash=record.config_hash,
            dataHash=record.training_data_hash,
            codeHash=record.code_hash,
            environmentHash=record.environment_hash,
            telemetryRecordHash=canonical_sha256(
                telemetry.model_dump(mode="python", by_alias=True)
            ),
            latestCheckpointSemanticHash=latest_ref.semantic_hash,
            bestCheckpointSemanticHash=best_ref.semantic_hash,
            finalModelStateHash=_model_state_hash(latest_payload["trainer"]["model"]),
            finalFitStateHash=latest_payload["trainer"]["fitState"]["stateHash"],
            compositeStateHash=pair_composite_hash,
            recoveryStateHash=pair_recovery_hash,
        ),
    )
    task_kind, head_name, requires_calibration = _expected_head_contract(record.cell.task_id)
    head_data = _load_canonical_model(
        serialized["head-data"],
        SupervisedTrainValidation,
        label="supervised head data",
    )
    head = _load_canonical_model(serialized["head-report"], HeadTrainingReport, label="head report")
    if (
        head.report_hash != record.head_report_hash
        or head.task_kind != task_kind
        or head.head_name != head_name
        or head.data_hash != record.head_data_hash
        or head_data.data_hash != record.head_data_hash
        or head_data.graph_version_hash != bundle.graph_version_hash
        or head_data.task_kind != task_kind
        or head_data.head_name != head_name
        or head.adapter_schema_hash != record.adapter_schema_hash
        or not head.promotion_eligible
    ):
        raise ValueError("supervised head report is not bound to the experiment run")
    calibration: CalibrationFitReport | None = None
    if requires_calibration:
        calibration = _load_canonical_model(
            serialized["calibration-report"],
            CalibrationFitReport,
            label="calibration report",
        )
        if (
            calibration.report_hash != record.calibration_hash
            or calibration.head_training_report_hash != head.report_hash
            or calibration.head_state_hash != head.head_state_hash
            or not calibration.promotion_eligible
        ):
            raise ValueError("calibration report is not bound to the selected head")
    elif record.calibration_hash is not None or "calibration-report" in refs:
        raise ValueError("task does not permit binary calibration evidence")
    if best_gfm is None:
        raise ValueError("formal trainable backend lacks a strict runtime evaluator")
    raw_validation = _reproduce_raw_gfm_validation(
        checkpoint=best_gfm,
        target_domain=recipe.target_manifest.graph_id,
        bundle=bundle,
        adapter_schema=adapter,
        head_data=head_data,
        head_report=head,
        calibration=calibration,
    )
    fold_evaluation = _verify_cell_gfm_fold_evaluations(
        root=root,
        record=record,
        inventory=fold_inventory,
        manifest=manifest,
        split_inventory=split_inventory,
        labels=labels,
        bundle=bundle,
        recipe=recipe,
        execution_config=execution_config,
        telemetry_policy=telemetry_policy,
    )
    if fold_evaluation.metrics != record.metrics:
        raise ValueError("raw record metrics differ from equal-weight live fold inference")
    return ObservedRunArtifacts(
        manifest=manifest,
        split_inventory=split_inventory,
        bundle=bundle,
        adapter_schema=adapter,
        recipe=recipe,
        execution_config=execution_config,
        training_data=training_data,
        head_data=head_data,
        head_report=head,
        calibration=calibration,
        raw_validation=raw_validation,
        fold_inventory=fold_inventory,
        fold_evaluation=fold_evaluation,
    )


def _reload_matrix_records(
    *,
    root: Path,
    ledger: ExperimentLedger,
    protocol: ExperimentProtocol,
    matrix: tuple[Any, ...],
    aggregates: tuple[ExperimentAggregate, ...],
    preflight_hash: str,
    preflight: Any,
    telemetry_policy: TrustedTelemetryPolicy,
) -> tuple[
    tuple[ExperimentRunRecord, ...],
    tuple[VerifiedExperimentRun, ...],
    bool,
    bool,
]:
    records: list[ExperimentRunRecord] = []
    verified_runs: list[VerifiedExperimentRun] = []
    artifacts_valid = True
    for cell in matrix:
        try:
            record = ledger.load_run(cell.cell_id)
        except Exception:
            return tuple(records), tuple(verified_runs), False, artifacts_valid
        if record.cell != cell or record.preflight_evidence_hash != preflight_hash:
            return tuple(records), tuple(verified_runs), False, artifacts_valid
        try:
            observed_artifacts = _verify_record_artifacts(
                root,
                record,
                preflight=preflight,
                telemetry_policy=telemetry_policy,
            )
            verified_runs.append(VerifiedExperimentRun(record=record, artifacts=observed_artifacts))
        except Exception:
            artifacts_valid = False
        records.append(record)

    trainable = tuple(record for record in records if record.cell.trainable)
    for values in (
        tuple(record.checkpoint_sha256 for record in trainable),
        tuple(record.best_checkpoint_sha256 for record in trainable),
        tuple(record.head_report_hash for record in trainable),
        tuple(record.metrics.prediction_hash for record in records),
        tuple(record.config_hash for record in records),
        tuple(record.telemetry_hash for record in records),
    ):
        if len(values) != len(set(values)):
            return tuple(records), tuple(verified_runs), False, False

    aggregate_by_slice = {item.slice_id: item for item in aggregates}
    if len(aggregate_by_slice) != len(aggregates):
        return tuple(records), tuple(verified_runs), False, artifacts_valid
    by_slice: dict[str, list[ExperimentRunRecord]] = {}
    for record in records:
        by_slice.setdefault(record.cell.slice_id, []).append(record)
    if set(aggregate_by_slice) != set(by_slice):
        return tuple(records), tuple(verified_runs), False, artifacts_valid
    for slice_id, slice_records in by_slice.items():
        try:
            if aggregate_experiment(protocol, tuple(slice_records)) != aggregate_by_slice[slice_id]:
                return tuple(records), tuple(verified_runs), False, artifacts_valid
        except Exception:
            return tuple(records), tuple(verified_runs), False, artifacts_valid
    return tuple(records), tuple(verified_runs), True, artifacts_valid


def _derive_verified_governance_winner(
    protocol: ExperimentProtocol,
    verified_runs: tuple[VerifiedExperimentRun, ...],
) -> GovernanceWinnerSelection:
    observations: list[GovernanceValidationObservation] = []
    for item in verified_runs:
        record = item.record
        if (
            record.cell.task_id not in _REQUIRED_GOVERNANCE_TASKS
            or record.cell.method_id not in {"multi-graph-shared-gfm", "domain-aligned-gfm"}
            or record.cell.label_budget != "full"
        ):
            continue
        reproduced = item.artifacts.fold_evaluation
        if reproduced is None or record.fold_evaluation_inventory_hash is None:
            raise ValueError("governance winner lacks live validation evidence")
        observations.append(
            GovernanceValidationObservation(
                cell_id=record.cell.cell_id,
                task_id=record.cell.task_id,
                method_id=record.cell.method_id,
                seed=record.cell.seed,
                label_budget=record.cell.label_budget,
                validation_metric=reproduced.validation_metric,
                record_hash=record.record_hash,
                best_checkpoint_sha256=reproduced.best_checkpoint_inventory_hash,
                head_report_hash=reproduced.head_report_inventory_hash,
                validation_partition_hash=reproduced.validation_partition_inventory_hash,
                validation_protocol_hash=reproduced.validation_protocol_inventory_hash,
                validation_data_hash=reproduced.validation_data_inventory_hash,
                validation_callback_hash=reproduced.validation_callback_inventory_hash,
            )
        )
    return derive_governance_winner(protocol, tuple(observations))


def _checkpoint_head_state_hash(payload: dict[str, Any], head_name: str) -> str:
    model_state = payload.get("trainer", {}).get("model")
    if not isinstance(model_state, dict):
        raise ValueError("candidate checkpoint model state is missing")
    prefix = f"{head_name}."
    head_state = {
        name[len(prefix) :]: value
        for name, value in model_state.items()
        if isinstance(name, str) and name.startswith(prefix)
    }
    if not head_state:
        raise ValueError("candidate checkpoint does not contain the selected head")
    return canonical_sha256(
        {name: canonical_tensor_digest(value) for name, value in sorted(head_state.items())}
    )


def _probe_payload(
    root: Path,
    latest_relative_path: str,
    best_relative_path: str,
    bindings: CheckpointBindings,
) -> FreshProcessCheckpointEvidence:
    latest_bytes = read_confined_snapshot(root, latest_relative_path, max_bytes=_MAX_ARTIFACT_BYTES)
    best_bytes = read_confined_snapshot(root, best_relative_path, max_bytes=_MAX_ARTIFACT_BYTES)
    latest = load_checkpoint(latest_bytes, expected_bindings=bindings)
    best = load_checkpoint(best_bytes, expected_bindings=bindings)
    latest_checkpoint = _strict_core_gfm_from_checkpoint(latest)
    best_checkpoint = _strict_core_gfm_from_checkpoint(best)
    fit_state = latest["trainer"].get("fitState")
    if not isinstance(fit_state, dict):
        raise ValueError("latest checkpoint lacks fit-state best evidence")
    latest_config_raw = latest["trainer"].get("config")
    best_config_raw = best["trainer"].get("config")
    if not isinstance(latest_config_raw, dict) or latest_config_raw != best_config_raw:
        raise ValueError("fresh checkpoint probe lacks one exact training config")
    training_config = _training_config_from_mapping(latest_config_raw)
    if training_config.preset != "formal":
        raise ValueError("fresh checkpoint probe requires formal training evidence")
    latest_step = int(latest["trainer"].get("optimizerStep", -1))
    best_step = int(best["trainer"].get("optimizerStep", -1))
    parsed_latest = _parse_fit_state(
        fit_state,
        optimizer_step=latest_step,
        config=training_config,
        checkpoint_model_state_hash=_model_state_hash(latest["trainer"]["model"]),
    )
    parsed_best = _parse_fit_state(
        best["trainer"].get("fitState"),
        optimizer_step=best_step,
        config=training_config,
        checkpoint_model_state_hash=_model_state_hash(best["trainer"]["model"]),
    )
    if (
        latest.get("status") != "validated"
        or latest.get("promotable") is not True
        or best.get("status") != "validated"
        or best.get("promotable") is not False
        or parsed_latest.best_step != best_step
        or parsed_latest.best_metric != parsed_best.best_metric
        or parsed_latest.best_model_state_hash != parsed_best.best_model_state_hash
        or parsed_best.best_step != best_step
        or parsed_best.best_checkpoint_name is not None
        or parsed_best.best_checkpoint_sha256 is not None
        or parsed_latest.best_model_state_hash != best_checkpoint.model_state_hash
        or parsed_best.best_model_state_hash != best_checkpoint.model_state_hash
    ):
        raise ValueError("fresh checkpoint latest/best validation evidence is inconsistent")
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-fresh-checkpoint-probe/1.0",
        "latestCheckpointSha256": hashlib.sha256(latest_bytes).hexdigest(),
        "bestCheckpointSha256": hashlib.sha256(best_bytes).hexdigest(),
        "latestStatus": latest.get("status"),
        "latestPromotable": latest.get("promotable"),
        "bestStatus": best.get("status"),
        "bestPromotable": best.get("promotable"),
        "latestOptimizerStep": latest_step,
        "bestOptimizerStep": best_step,
        "bestCheckpointName": parsed_latest.best_checkpoint_name,
        "latestModelStateHash": latest_checkpoint.model_state_hash,
        "bestModelStateHash": best_checkpoint.model_state_hash,
        "bestAdapterInventoryHash": _adapter_inventory_hash(best_checkpoint),
    }
    payload["evidenceHash"] = canonical_sha256(payload)
    return FreshProcessCheckpointEvidence.model_validate(payload)


def _run_probe_subprocess(
    root: Path,
    latest_relative_path: str,
    best_relative_path: str,
    bindings: CheckpointBindings,
) -> FreshProcessCheckpointEvidence:
    command = [
        sys.executable,
        "-m",
        "socialgraph_gfm.core.acceptance",
        "--fresh-probe",
        str(root),
        latest_relative_path,
        best_relative_path,
        bindings.config_hash,
        bindings.data_hash,
        bindings.code_hash,
        bindings.environment_hash,
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        timeout=120,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError("fresh-process checkpoint probe failed")
    return FreshProcessCheckpointEvidence.model_validate_json(completed.stdout)


def publish_fresh_process_checkpoint_evidence(
    *,
    runtime_root: Path,
    latest_checkpoint_relative_path: str,
    best_checkpoint_relative_path: str,
    bindings: CheckpointBindings,
    publish_to: Path,
) -> FreshProcessCheckpointEvidence:
    root = secure_existing_root(runtime_root)
    evidence = _run_probe_subprocess(
        root,
        latest_checkpoint_relative_path,
        best_checkpoint_relative_path,
        bindings,
    )
    serialized = _canonical_bytes(evidence)
    _publish_immutable_exact(
        root,
        Path(publish_to),
        serialized,
        conflict_message="conflicting fresh-process checkpoint evidence already exists",
    )
    return evidence


def _load_fresh_process_evidence(root: Path, path: Path) -> FreshProcessCheckpointEvidence:
    lexical = reject_link_components(path)
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("fresh-process evidence escapes runtime root") from error
    serialized = read_confined_snapshot(root, relative, max_bytes=_MAX_ACCEPTANCE_BYTES)
    return _load_canonical_model(
        serialized,
        FreshProcessCheckpointEvidence,
        label="fresh-process checkpoint evidence",
    )


def _load_candidate_manifest(root: Path, path: Path) -> CandidateGovernanceManifest:
    lexical = reject_link_components(path)
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("candidate manifest escapes runtime root") from error
    serialized = read_confined_snapshot(root, relative, max_bytes=_MAX_ACCEPTANCE_BYTES)
    return _load_canonical_model(
        serialized,
        CandidateGovernanceManifest,
        label="governance candidate manifest",
    )


@dataclass(frozen=True)
class CandidateCheckpointObservation:
    latest_hash: str
    best_hash: str
    latest_payload: dict[str, Any]
    best_payload: dict[str, Any]
    latest_model: StrictCheckpointModel
    best_model: StrictCheckpointModel
    latest_step: int
    best_step: int
    best_name: str


def _verify_candidate_checkpoint(
    root: Path,
    manifest: CandidateGovernanceManifest,
) -> CandidateCheckpointObservation:
    bindings = CheckpointBindings(
        config_hash=manifest.execution.config_hash,
        data_hash=manifest.training_inventory.inventory_hash,
        code_hash=manifest.code_hash,
        environment_hash=manifest.environment_hash,
    )
    latest_bytes = _read_ref(root, manifest.latest_checkpoint)
    best_bytes = _read_ref(root, manifest.best_checkpoint)
    latest_hash = hashlib.sha256(latest_bytes).hexdigest()
    best_hash = hashlib.sha256(best_bytes).hexdigest()
    latest = load_checkpoint(latest_bytes, expected_bindings=bindings)
    best = load_checkpoint(best_bytes, expected_bindings=bindings)
    latest_model = _strict_core_gfm_from_checkpoint(latest)
    best_model = _strict_core_gfm_from_checkpoint(best)
    training_config = _training_config_from_mapping(manifest.execution.trainer_config)
    latest_config = _training_config_from_mapping(latest["trainer"].get("config", {}))
    best_config = _training_config_from_mapping(best["trainer"].get("config", {}))
    if latest_config != training_config or best_config != training_config:
        raise ValueError("candidate checkpoint trainer config differs from candidate execution")
    latest_step = int(latest["trainer"].get("optimizerStep", -1))
    best_step = int(best["trainer"].get("optimizerStep", -1))
    latest_fit = _parse_fit_state(
        latest["trainer"].get("fitState"),
        optimizer_step=latest_step,
        config=training_config,
        checkpoint_model_state_hash=latest_model.model_state_hash,
    )
    best_fit = _parse_fit_state(
        best["trainer"].get("fitState"),
        optimizer_step=best_step,
        config=training_config,
        checkpoint_model_state_hash=best_model.model_state_hash,
    )
    best_name = Path(manifest.best_checkpoint.relative_path).name
    match = _IMMUTABLE_BEST_NAME.fullmatch(best_name)
    if (
        latest_hash != manifest.latest_checkpoint.byte_sha256
        or best_hash != manifest.best_checkpoint.byte_sha256
        or latest.get("status") != "validated"
        or latest.get("promotable") is not True
        or best.get("status") != "validated"
        or best.get("promotable") is not False
        or latest_step < training_config.min_steps
        or latest_step > training_config.max_steps
        or best_step < training_config.min_steps
        or best_step > latest_step
        or match is None
        or int(match.group("step")) != best_step
        or latest_fit.best_checkpoint_name != best_name
        or latest_fit.best_checkpoint_sha256 != best_hash
        or latest_fit.best_step != best_step
        or latest_fit.best_metric != best_fit.best_metric
        or latest_fit.best_model_state_hash != best_model.model_state_hash
        or latest_fit.validation_protocol_hash != best_fit.validation_protocol_hash
        or latest_fit.validation_data_hash != best_fit.validation_data_hash
        or latest_fit.validation_partition_hash != best_fit.validation_partition_hash
        or latest_fit.validation_callback_hash != best_fit.validation_callback_hash
        or best_fit.best_step != best_step
        or best_fit.best_model_state_hash != best_model.model_state_hash
        or best_fit.best_checkpoint_name is not None
        or best_fit.best_checkpoint_sha256 is not None
        or latest_model.adapter_schemas != best_model.adapter_schemas
    ):
        raise ValueError("candidate latest/best checkpoint control is inconsistent")
    return CandidateCheckpointObservation(
        latest_hash=latest_hash,
        best_hash=best_hash,
        latest_payload=latest,
        best_payload=best,
        latest_model=latest_model,
        best_model=best_model,
        latest_step=latest_step,
        best_step=best_step,
        best_name=best_name,
    )


def _encoder_state_hash(model: CoreGFM) -> str:
    return canonical_sha256(
        {
            name: canonical_tensor_digest(value)
            for name, value in sorted(model.encoder.state_dict().items())
        }
    )


def _verify_candidate_task_inventory(
    *,
    root: Path,
    manifest: CandidateGovernanceManifest,
    checkpoint: CandidateCheckpointObservation,
    records: tuple[ExperimentRunRecord, ...],
    preflight: Any,
    telemetry_policy: TrustedTelemetryPolicy,
) -> tuple[str, ...]:
    record_by_hash = {record.record_hash: record for record in records}
    evidence_hashes: list[str] = []
    model = checkpoint.best_model.model
    model.eval()
    for task_evidence in manifest.training_inventory.tasks:
        record = record_by_hash.get(task_evidence.record_hash)
        if (
            record is None
            or record.cell.cell_id != task_evidence.cell_id
            or record.cell.task_id != task_evidence.task_id
            or record.cell.method_id != manifest.execution.method_id
            or record.cell.seed != manifest.execution.seed
            or record.cell.label_budget != "full"
            or record.recipe_hash != task_evidence.recipe_hash
            or record.head_data_hash != task_evidence.supervised_data_hash
            or record.head_report_hash != task_evidence.head_report_hash
            or record.calibration_hash != task_evidence.calibration_hash
        ):
            raise ValueError("candidate task does not bind the winning full-budget raw record")
        observed = _verify_record_artifacts(
            root,
            record,
            preflight=preflight,
            telemetry_policy=telemetry_policy,
        )
        if observed.head_data is None or observed.head_report is None:
            raise ValueError("candidate governance task lacks supervised head evidence")
        if (
            observed.bundle.graph_version_hash != task_evidence.graph_version_hash
            or observed.split_inventory.inventory_hash != task_evidence.split_inventory_hash
            or task_evidence.adapter_domain != record.cell.target_graph_id
        ):
            raise ValueError("candidate task graph or split evidence changed")
        domain = task_evidence.adapter_domain
        schema = checkpoint.best_model.adapter_schemas.get(domain)
        adapter_state = checkpoint.best_model.adapter_states.get(domain)
        if schema is None or adapter_state is None or schema != observed.adapter_schema:
            raise ValueError("candidate checkpoint lacks the exact task adapter schema")
        adapter = BundleInputAdapter(
            observed.bundle,
            mode="training",
            schema=schema,
        )
        adapter.load_state_dict(adapter_state, strict=True)
        encoded = encode_supervised_graph(model, observed.bundle, adapter)
        verified_head = _new_verified_head_training_report(observed.head_report)
        verify_head_training_report(model, encoded, observed.head_data, verified_head)
        with torch.no_grad():
            validation_loss = _supervised_loss(
                model,
                encoded.tensor,
                observed.head_data.task_kind,
                observed.head_data.validation,
            )
        if -float(validation_loss.detach().cpu()) != observed.head_report.best_metric:
            raise ValueError("candidate head best metric is not reproduced from validation data")
        requires_calibration = _expected_head_contract(record.cell.task_id)[2]
        if requires_calibration:
            if observed.calibration is None:
                raise ValueError("candidate binary task lacks calibration evidence")
            semantics = BinaryScoreSemantics.for_task(observed.head_data.task_kind)
            scores = derive_validation_scores(
                model,
                encoded,
                observed.head_data,
                verified_head,
                semantics=semantics,
            )
            protocol = CalibrationProtocol.fixed(scores)
            if fit_score_calibration_report(scores, protocol=protocol) != observed.calibration:
                raise ValueError("candidate calibration is not reproduced from validation scores")
        elif observed.calibration is not None:
            raise ValueError("candidate regression task cannot carry binary calibration")
        evidence_hashes.append(task_evidence.evidence_hash)

    source_record = next(
        (record for record in records if record.cell.cell_id == manifest.encoder_source_cell_id),
        None,
    )
    if source_record is None:
        raise ValueError("candidate encoder source record is missing")
    source_best_ref = _artifact_map(source_record).get("best-checkpoint")
    if source_best_ref is None:
        raise ValueError("candidate encoder source best checkpoint is missing")
    source_bindings = CheckpointBindings(
        config_hash=source_record.config_hash,
        data_hash=source_record.training_data_hash,
        code_hash=source_record.code_hash,
        environment_hash=source_record.environment_hash,
    )
    source_bytes = _read_ref(root, source_best_ref)
    source_payload = load_checkpoint(source_bytes, expected_bindings=source_bindings)
    source_model = _strict_core_gfm_from_checkpoint(source_payload)
    if hashlib.sha256(
        source_bytes
    ).hexdigest() != manifest.encoder_source_best_checkpoint_sha256 or _encoder_state_hash(
        source_model.model
    ) != _encoder_state_hash(model):
        raise ValueError("candidate encoder differs from the winning experiment checkpoint")
    return tuple(sorted(evidence_hashes))


def _derive_core_acceptance_once(
    *,
    runtime_root: Path,
    preflight_path: Path,
    protocol: ExperimentProtocol,
    aggregates: tuple[ExperimentAggregate, ...],
    transfer_decisions: tuple[TransferDecision, ...],
    candidate_cell_id: str | None,
    candidate_manifest_path: Path | None = None,
    fresh_process_evidence_path: Path | None,
    telemetry_policy: TrustedTelemetryPolicy,
) -> CoreAcceptance:
    """Derive acceptance from reopened evidence; callers cannot set acceptance state."""

    if type(telemetry_policy) is not TrustedTelemetryPolicy:
        raise TypeError("formal acceptance requires an exact trusted telemetry policy")
    root = secure_existing_root(runtime_root)
    failed: set[str] = set()
    preflight: Any | None = None
    preflight_hash: str | None = None
    try:
        preflight = load_formal_preflight(preflight_path, runtime_root=root)
        preflight_hash = preflight.evidence_hash
        if not preflight.formal_ready or not preflight.promotable:
            failed.add("formal-preflight")
    except Exception:
        failed.add("formal-preflight")

    matrix = build_experiment_matrix(protocol)
    expected_slices = {cell.slice_id for cell in matrix}
    aggregate_by_slice = {item.slice_id: item for item in aggregates}
    if len(aggregate_by_slice) != len(aggregates) or set(aggregate_by_slice) != expected_slices:
        failed.add("matrix-completeness")
    if len(aggregates) != len(expected_slices) or any(
        not aggregate.promotable for aggregate in aggregates
    ):
        failed.add("aggregate-promotability")

    records: tuple[ExperimentRunRecord, ...] = ()
    verified_runs: tuple[VerifiedExperimentRun, ...] = ()
    ledger = ExperimentLedger(root)
    if preflight_hash is None or preflight is None:
        failed.add("raw-ledger")
        failed.add("artifact-revalidation")
    else:
        records, verified_runs, ledger_valid, artifact_valid = _reload_matrix_records(
            root=root,
            ledger=ledger,
            protocol=protocol,
            matrix=matrix,
            aggregates=aggregates,
            preflight_hash=preflight_hash,
            preflight=preflight,
            telemetry_policy=telemetry_policy,
        )
        if not ledger_valid or len(records) != len(matrix):
            failed.add("raw-ledger")
        if not artifact_valid or len(records) != len(matrix):
            failed.add("artifact-revalidation")

    expected_transfer_tasks = {cell.task_id for cell in matrix}
    transfer_by_task = {item.task_id: item for item in transfer_decisions}
    transfer_valid = (
        len(transfer_by_task) == len(transfer_decisions)
        and set(transfer_by_task) == expected_transfer_tasks
        and all(item.transfer_advantage for item in transfer_decisions)
    )
    if transfer_valid and records:
        for decision in transfer_decisions:
            try:
                scratch = [
                    record
                    for record in records
                    if record.cell.task_id == decision.task_id
                    and record.cell.method_id == "graphsage-scratch"
                ]
                candidate_records = [
                    record
                    for record in records
                    if record.cell.task_id == decision.task_id
                    and record.cell.method_id == decision.candidate_method_id
                ]
                if derive_transfer_advantage(protocol, scratch, candidate_records) != decision:
                    transfer_valid = False
            except Exception:
                transfer_valid = False
    if not transfer_valid:
        failed.add("transfer-advantage")

    winner: GovernanceWinnerSelection | None = None
    try:
        if len(verified_runs) != len(matrix):
            raise ValueError("governance winner requires every matrix artifact to revalidate")
        winner = _derive_verified_governance_winner(protocol, verified_runs)
    except Exception:
        failed.add("candidate")

    record_by_cell = {record.cell.cell_id: record for record in records}
    selected_anchor_id = None if winner is None else winner.selected_runs[0].cell_id
    candidate = None if selected_anchor_id is None else record_by_cell.get(selected_anchor_id)
    if (
        candidate is None
        or candidate_cell_id != selected_anchor_id
        or not candidate.promotable
        or candidate.cell.method_id not in {"multi-graph-shared-gfm", "domain-aligned-gfm"}
        or candidate.cell.label_budget != "full"
    ):
        failed.add("candidate")

    latest_hash: str | None = None
    best_hash: str | None = None
    candidate_manifest_hash: str | None = None
    candidate_task_hashes: tuple[str, ...] = ()
    fresh_hash: str | None = None
    if candidate is None:
        failed.update(
            {"best-checkpoint", "checkpoint-reload", "head-report", "calibration", "fresh-process"}
        )
    else:
        candidate_manifest: CandidateGovernanceManifest | None = None
        checkpoint: CandidateCheckpointObservation | None = None
        if candidate_manifest_path is None:
            failed.add("candidate")
            failed.add("best-checkpoint")
            failed.add("checkpoint-reload")
        else:
            try:
                candidate_manifest = _load_candidate_manifest(root, candidate_manifest_path)
                governance_records = tuple(
                    record_by_cell.get(item.cell_id)
                    for item in candidate_manifest.training_inventory.tasks
                )
                selected_cell_ids = (
                    () if winner is None else tuple(item.cell_id for item in winner.selected_runs)
                )
                selected_record_hashes = (
                    ()
                    if winner is None
                    else tuple(item.record_hash for item in winner.selected_runs)
                )
                if (
                    winner is None
                    or candidate_manifest.execution.winner_selection_hash != winner.selection_hash
                    or candidate_manifest.execution.method_id != candidate.cell.method_id
                    or candidate_manifest.execution.seed != candidate.cell.seed
                    or candidate_manifest.execution.task_cell_ids != selected_cell_ids
                    or candidate_manifest.execution.source_record_hashes != selected_record_hashes
                    or candidate_manifest.encoder_source_cell_id != candidate.cell.cell_id
                    or any(record is None for record in governance_records)
                    or any(
                        record is not None
                        and (
                            record.cell.task_id != task_id
                            or record.cell.method_id != candidate.cell.method_id
                            or record.cell.label_budget != "full"
                            or record.cell.seed != candidate.cell.seed
                            or record.code_hash != candidate_manifest.code_hash
                            or record.environment_hash != candidate_manifest.environment_hash
                        )
                        for task_id, record in zip(
                            _REQUIRED_GOVERNANCE_TASKS,
                            governance_records,
                            strict=True,
                        )
                    )
                    or any(
                        decision.candidate_method_id != candidate.cell.method_id
                        for decision in transfer_decisions
                    )
                ):
                    raise ValueError("candidate is not the full-budget winner for every task")
                checkpoint = _verify_candidate_checkpoint(root, candidate_manifest)
                latest_hash = checkpoint.latest_hash
                best_hash = checkpoint.best_hash
                candidate_manifest_hash = candidate_manifest.manifest_hash
            except Exception:
                failed.add("candidate")
                failed.add("best-checkpoint")
                failed.add("checkpoint-reload")

        if candidate_manifest is None or checkpoint is None:
            failed.add("head-report")
            failed.add("calibration")
        else:
            try:
                candidate_task_hashes = _verify_candidate_task_inventory(
                    root=root,
                    manifest=candidate_manifest,
                    checkpoint=checkpoint,
                    records=records,
                    preflight=preflight,
                    telemetry_policy=telemetry_policy,
                )
            except Exception:
                failed.add("head-report")
                failed.add("calibration")

            if fresh_process_evidence_path is None:
                failed.add("fresh-process")
            else:
                try:
                    bindings = CheckpointBindings(
                        config_hash=candidate_manifest.execution.config_hash,
                        data_hash=candidate_manifest.training_inventory.inventory_hash,
                        code_hash=candidate_manifest.code_hash,
                        environment_hash=candidate_manifest.environment_hash,
                    )
                    observed = _load_fresh_process_evidence(root, fresh_process_evidence_path)
                    rerun = _run_probe_subprocess(
                        root,
                        candidate_manifest.latest_checkpoint.relative_path,
                        candidate_manifest.best_checkpoint.relative_path,
                        bindings,
                    )
                    if observed != rerun or (
                        observed.latest_checkpoint_sha256 != latest_hash
                        or observed.best_checkpoint_sha256 != best_hash
                        or observed.latest_optimizer_step != checkpoint.latest_step
                        or observed.best_checkpoint_name != checkpoint.best_name
                        or observed.best_model_state_hash != checkpoint.best_model.model_state_hash
                        or observed.best_adapter_inventory_hash
                        != _adapter_inventory_hash(checkpoint.best_model)
                    ):
                        raise ValueError("fresh-process checkpoint evidence is stale")
                    fresh_hash = observed.evidence_hash
                except Exception:
                    failed.add("fresh-process")

    failed_ordered = tuple(gate for gate in _GATE_ORDER if gate in failed)
    accepted = not failed_ordered
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-acceptance/1.1",
        "protocolHash": protocol.protocol_hash,
        "preflightEvidenceHash": preflight_hash,
        "aggregateHashes": sorted(item.aggregate_hash for item in aggregates),
        "transferDecisionHashes": sorted(item.decision_hash for item in transfer_decisions),
        "rawRecordHashes": sorted(record.record_hash for record in records),
        "winnerSelectionHash": None if winner is None else winner.selection_hash,
        "candidateCellId": candidate_cell_id,
        "candidateRecordHash": None if candidate is None else candidate.record_hash,
        "candidateLatestCheckpointSha256": latest_hash,
        "candidateCheckpointSha256": best_hash,
        "candidateManifestHash": candidate_manifest_hash,
        "candidateTaskEvidenceHashes": sorted(candidate_task_hashes),
        "freshProcessEvidenceHash": fresh_hash,
        "failedGates": list(failed_ordered),
        "status": "accepted" if accepted else "rejected",
        "accepted": accepted,
        "promotable": accepted,
    }
    payload["acceptanceHash"] = canonical_sha256(payload)
    return CoreAcceptance.model_validate(payload)


def derive_core_acceptance(
    *,
    runtime_root: Path,
    preflight_path: Path,
    protocol: ExperimentProtocol,
    aggregates: tuple[ExperimentAggregate, ...],
    transfer_decisions: tuple[TransferDecision, ...],
    candidate_cell_id: str | None,
    candidate_manifest_path: Path | None = None,
    fresh_process_evidence_path: Path | None,
    telemetry_policy: TrustedTelemetryPolicy,
    publish_to: Path | None = None,
) -> CoreAcceptance:
    """Derive, optionally publish, and finally rederive an acceptance decision."""

    report = _derive_core_acceptance_once(
        runtime_root=runtime_root,
        preflight_path=preflight_path,
        protocol=protocol,
        aggregates=aggregates,
        transfer_decisions=transfer_decisions,
        candidate_cell_id=candidate_cell_id,
        candidate_manifest_path=candidate_manifest_path,
        fresh_process_evidence_path=fresh_process_evidence_path,
        telemetry_policy=telemetry_policy,
    )
    if publish_to is None:
        return report
    root = secure_existing_root(runtime_root)
    target = reject_link_components(publish_to)
    try:
        relative = target.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("core acceptance report escapes runtime root") from error
    serialized = _canonical_bytes(report)
    if len(serialized) > _MAX_ACCEPTANCE_BYTES:
        raise ValueError("acceptance report exceeds the publication size limit")
    _publish_immutable_exact(
        root,
        target,
        serialized,
        conflict_message="conflicting core acceptance report already exists",
    )
    _ACCEPTANCE_PUBLICATION_SEAM(target)
    final_observation = _derive_core_acceptance_once(
        runtime_root=runtime_root,
        preflight_path=preflight_path,
        protocol=protocol,
        aggregates=aggregates,
        transfer_decisions=transfer_decisions,
        candidate_cell_id=candidate_cell_id,
        candidate_manifest_path=candidate_manifest_path,
        fresh_process_evidence_path=fresh_process_evidence_path,
        telemetry_policy=telemetry_policy,
    )
    if final_observation != report:
        raise RuntimeError("acceptance inputs changed during publication")
    if read_confined_snapshot(root, relative, max_bytes=_MAX_ACCEPTANCE_BYTES) != serialized:
        raise RuntimeError("published acceptance report changed after exact publication")
    return report


def load_core_acceptance(
    path: Path, *, runtime_root: Path | None = None
) -> CoreAcceptance:
    lexical = reject_link_components(path)
    root = secure_existing_root(lexical.parent if runtime_root is None else runtime_root)
    try:
        relative = lexical.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("core acceptance report escapes runtime root") from error
    serialized = read_confined_snapshot(root, relative, max_bytes=_MAX_ACCEPTANCE_BYTES)
    report = _load_canonical_model(
        serialized, CoreAcceptance, label="core acceptance report"
    )
    if report.accepted:
        raise ValueError(
            "accepted report requires derive_core_acceptance to revalidate runtime evidence"
        )
    return report


def _probe_main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fresh-probe", action="store_true")
    parser.add_argument("runtime_root")
    parser.add_argument("latest_relative_path")
    parser.add_argument("best_relative_path")
    parser.add_argument("config_hash")
    parser.add_argument("data_hash")
    parser.add_argument("code_hash")
    parser.add_argument("environment_hash")
    parsed = parser.parse_args(arguments)
    if not parsed.fresh_probe:
        return 2
    evidence = _probe_payload(
        secure_existing_root(Path(parsed.runtime_root)),
        parsed.latest_relative_path,
        parsed.best_relative_path,
        CheckpointBindings(
            config_hash=parsed.config_hash,
            data_hash=parsed.data_hash,
            code_hash=parsed.code_hash,
            environment_hash=parsed.environment_hash,
        ),
    )
    sys.stdout.buffer.write(_canonical_bytes(evidence))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through a fresh process
    raise SystemExit(_probe_main(sys.argv[1:]))


__all__ = [
    "CandidateExecutionEvidence",
    "CandidateGovernanceManifest",
    "CandidateTaskEvidence",
    "CandidateTrainingInventory",
    "FreshProcessCheckpointEvidence",
    "CoreAcceptance",
    "derive_core_acceptance",
    "load_core_acceptance",
    "publish_fresh_process_checkpoint_evidence",
]
