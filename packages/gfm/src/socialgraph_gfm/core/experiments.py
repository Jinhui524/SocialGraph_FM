"""Fixed experiment matrix, immutable raw ledger, aggregation, and transfer gates."""

from __future__ import annotations

import math
import random
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .formal_preflight import (
    _PublicationParentLease,
    _publish_exact as _publish_immutable_exact,
)
from .baselines import GfmFamilySpec, fixed_gfm_family_specs
from .config import TrainingConfig
from .metrics import BinaryThreshold, TaskMetricSet, select_binary_threshold
from .resource_telemetry import (
    ResourceTelemetryRecord,
    VerifiedResourceTelemetry,
    verify_resource_telemetry,
)
from .telemetry_receipt import TelemetryReceipt
from .safe_paths import read_confined_snapshot, secure_existing_root


_HASH = r"^[0-9a-f]{64}$"
_FORMAL_SEEDS = (20260821, 20260822, 20260823, 20260824, 20260825)
_BUDGETS = ("1", "5", "20", "full")
_MAX_ELAPSED_SECONDS = 21_600.0
_MAX_CUDA_BYTES = int(6.5 * 1024**3)
_MAX_LEDGER_BYTES = 4 * 1024 * 1024
_FIXED_PROTOCOL_HASH: str | None = None


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


def _safe_relative_artifact_path(value: str) -> str:
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("artifact path must use safe POSIX-relative syntax")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError("artifact path must be a safe relative path")
    return parsed.as_posix()


class ExperimentArtifactRef(_StrictModel):
    """One immutable runtime byte artifact and its domain semantic identity."""

    role: Literal[
        "adapter-schema",
        "best-checkpoint",
        "calibration-report",
        "code",
        "configuration",
        "dataset-manifest",
        "environment",
        "experiment-recipe",
        "fold-evaluation-inventory",
        "head-report",
        "head-data",
        "labels",
        "latest-checkpoint",
        "predictions",
        "resource-telemetry",
        "telemetry-receipt",
        "split-manifest",
        "split-inventory",
        "structure-cache",
        "targets",
        "threshold",
        "training-data",
    ]
    relative_path: str = Field(alias="relativePath")
    byte_sha256: str = Field(alias="byteSha256", pattern=_HASH)
    semantic_hash: str = Field(alias="semanticHash", pattern=_HASH)
    size_bytes: int = Field(alias="sizeBytes", gt=0)

    @model_validator(mode="after")
    def validate_artifact(self):
        normalized = _safe_relative_artifact_path(self.relative_path)
        if normalized != self.relative_path:
            raise ValueError("artifact path must already be canonical POSIX-relative syntax")
        return self


class PredictionEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-metric-predictions/1.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", min_length=1)
    split_inventory_hash: str = Field(alias="splitInventoryHash", pattern=_HASH)
    fold_ids: tuple[str, ...] = Field(alias="foldIds", strict=False, min_length=1)
    evaluation_kind: Literal["binary", "link-ranking", "regression"] = Field(alias="evaluationKind")
    entity_ids: tuple[str, ...] = Field(alias="entityIds", strict=False, min_length=2)
    entity_fold_ids: tuple[str, ...] = Field(alias="entityFoldIds", strict=False, min_length=2)
    scores: tuple[float, ...] = Field(strict=False, min_length=2)
    probabilities: tuple[float, ...] = Field(strict=False)
    filtered_negative_scores: tuple[tuple[float, ...], ...] = Field(
        alias="filteredNegativeScores", strict=False
    )
    prediction_hash: str = Field(alias="predictionHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_predictions(self):
        count = len(self.entity_ids)
        if (
            self.fold_ids != tuple(sorted(set(self.fold_ids)))
            or len(set(self.entity_ids)) != count
            or len(self.scores) != count
            or len(self.entity_fold_ids) != count
            or set(self.entity_fold_ids) != set(self.fold_ids)
        ):
            raise ValueError("prediction entities and scores must align uniquely")
        flattened = (*self.scores, *self.probabilities)
        if not all(math.isfinite(value) for value in flattened):
            raise ValueError("prediction evidence must be finite")
        if self.evaluation_kind == "binary":
            if (
                len(self.probabilities) != count
                or self.filtered_negative_scores
                or any(value < 0.0 or value > 1.0 for value in self.probabilities)
            ):
                raise ValueError("binary predictions require aligned probabilities only")
        elif self.evaluation_kind == "link-ranking":
            if self.probabilities or len(self.filtered_negative_scores) != count:
                raise ValueError("link predictions require filtered negatives only")
            if any(
                not row or not all(math.isfinite(value) for value in row)
                for row in self.filtered_negative_scores
            ):
                raise ValueError("filtered negative prediction evidence must be finite")
        elif self.probabilities or self.filtered_negative_scores:
            raise ValueError("regression predictions cannot contain classification fields")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"prediction_hash"})
        )
        if self.prediction_hash != expected:
            raise ValueError("predictionHash does not match prediction evidence")
        return self


class TargetEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-metric-targets/1.0"] = Field(alias="schemaVersion")
    task_id: str = Field(alias="taskId", min_length=1)
    split_inventory_hash: str = Field(alias="splitInventoryHash", pattern=_HASH)
    fold_ids: tuple[str, ...] = Field(alias="foldIds", strict=False, min_length=1)
    evaluation_kind: Literal["binary", "link-ranking", "regression"] = Field(alias="evaluationKind")
    entity_ids: tuple[str, ...] = Field(alias="entityIds", strict=False, min_length=2)
    entity_fold_ids: tuple[str, ...] = Field(alias="entityFoldIds", strict=False, min_length=2)
    values: tuple[float, ...] = Field(strict=False, min_length=2)
    target_hash: str = Field(alias="targetHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_targets(self):
        if (
            self.fold_ids != tuple(sorted(set(self.fold_ids)))
            or len(self.entity_ids) != len(self.values)
            or len(set(self.entity_ids)) != len(self.entity_ids)
            or len(self.entity_fold_ids) != len(self.entity_ids)
            or set(self.entity_fold_ids) != set(self.fold_ids)
            or not all(math.isfinite(value) for value in self.values)
        ):
            raise ValueError("target entities and finite values must align uniquely")
        if self.evaluation_kind in {"binary", "link-ranking"} and any(
            value not in {0.0, 1.0} for value in self.values
        ):
            raise ValueError("classification targets must contain zero or one")
        if self.evaluation_kind == "binary" and set(self.values) != {0.0, 1.0}:
            raise ValueError("binary test targets must contain both classes")
        if self.evaluation_kind == "link-ranking" and any(value != 1.0 for value in self.values):
            raise ValueError("link ranking targets must identify held-out positives")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"target_hash"})
        )
        if self.target_hash != expected:
            raise ValueError("targetHash does not match target evidence")
        return self


class ThresholdSelectionEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-metric-threshold-selection/1.0"] = Field(
        alias="schemaVersion"
    )
    threshold: BinaryThreshold
    validation_scores: tuple[float, ...] = Field(
        alias="validationScores", strict=False, min_length=2
    )
    validation_targets: tuple[float, ...] = Field(
        alias="validationTargets", strict=False, min_length=2
    )
    evidence_hash: str = Field(alias="evidenceHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_selection(self):
        expected_threshold = select_binary_threshold(
            self.validation_scores,
            self.validation_targets,
            validation_partition_hash=self.threshold.validation_partition_hash,
            objective="macro-f1",
        )
        if self.threshold != expected_threshold:
            raise ValueError("threshold is not reproduced from validation-only evidence")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        )
        if self.evidence_hash != expected:
            raise ValueError("threshold evidenceHash does not match validation evidence")
        return self


class MethodSpec(_StrictModel):
    method_id: str = Field(alias="methodId", min_length=1)
    backend: Literal[
        "attribute",
        "structure",
        "heuristic",
        "scratch",
        "linkx",
        "pretrained-single",
        "pretrained-shared",
        "pretrained-aligned",
    ]
    trainable: bool
    target_unlabeled_adaptation: bool = Field(alias="targetUnlabeledAdaptation")


class ExperimentTaskSpec(_StrictModel):
    task_id: str = Field(alias="taskId", min_length=1)
    target_graph_id: str = Field(alias="targetGraphId", min_length=1)
    validation_graph_id: str | None = Field(default=None, alias="validationGraphId", min_length=1)
    source_graph_ids: tuple[str, ...] = Field(alias="sourceGraphIds", strict=False)
    applicable_methods: tuple[str, ...] = Field(
        alias="applicableMethods", strict=False, min_length=1
    )
    required_metrics: tuple[str, ...] = Field(alias="requiredMetrics", strict=False, min_length=1)
    primary_metric: str = Field(alias="primaryMetric", min_length=1)

    @model_validator(mode="after")
    def validate_task(self):
        if (
            len(self.source_graph_ids) != len(set(self.source_graph_ids))
            or self.target_graph_id in self.source_graph_ids
            or (
                self.validation_graph_id is not None
                and (
                    self.validation_graph_id == self.target_graph_id
                    or self.validation_graph_id in self.source_graph_ids
                )
            )
        ):
            raise ValueError("experiment sources must be unique and graph-disjoint")
        if tuple(sorted(set(self.required_metrics))) != self.required_metrics:
            raise ValueError("required metrics must be unique and sorted")
        if self.primary_metric not in self.required_metrics:
            raise ValueError("primary metric must be in the required inventory")
        if len(self.applicable_methods) != len(set(self.applicable_methods)):
            raise ValueError("applicable methods must be unique")
        return self


class ExperimentProtocol(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-protocol/1.0"] = Field(
        alias="schemaVersion"
    )
    methods: tuple[MethodSpec, ...] = Field(strict=False, min_length=1)
    tasks: tuple[ExperimentTaskSpec, ...] = Field(strict=False, min_length=1)
    label_budgets: tuple[Literal["1", "5", "20", "full"], ...] = Field(
        alias="labelBudgets", strict=False
    )
    seeds: tuple[int, ...] = Field(strict=False)
    max_elapsed_seconds: float = Field(alias="maxElapsedSeconds")
    max_cuda_bytes: int = Field(alias="maxCudaBytes")
    max_data_wait_ratio: float = Field(alias="maxDataWaitRatio")
    protocol_hash: str = Field(alias="protocolHash", pattern=_HASH)

    @classmethod
    def fixed(cls) -> ExperimentProtocol:
        family_methods = tuple(
            MethodSpec(
                methodId=family.method_id,
                backend=(
                    "scratch"
                    if family.method_id == "graphsage-scratch"
                    else (
                        "pretrained-single"
                        if family.method_id == "graphmae2-single"
                        else (
                            "pretrained-shared"
                            if family.method_id == "multi-graph-shared-gfm"
                            else "pretrained-aligned"
                        )
                    )
                ),
                trainable=True,
                targetUnlabeledAdaptation=family.target_unlabeled_adaptation,
            )
            for family in fixed_gfm_family_specs()
        )
        methods = (
            MethodSpec(
                methodId="attribute-mlp",
                backend="attribute",
                trainable=True,
                targetUnlabeledAdaptation=False,
            ),
            MethodSpec(
                methodId="structure-only",
                backend="structure",
                trainable=True,
                targetUnlabeledAdaptation=False,
            ),
            MethodSpec(
                methodId="common-neighbors",
                backend="heuristic",
                trainable=False,
                targetUnlabeledAdaptation=False,
            ),
            MethodSpec(
                methodId="adamic-adar",
                backend="heuristic",
                trainable=False,
                targetUnlabeledAdaptation=False,
            ),
            MethodSpec(
                methodId="linkx", backend="linkx", trainable=True, targetUnlabeledAdaptation=False
            ),
            *family_methods,
        )
        near = (
            "facebook100.reed98",
            "facebook100.amherst41",
            "facebook100.johns-hopkins55",
            "facebook100.cornell5",
        )
        cross = (
            "twitch.de",
            "twitch.en",
            "twitch.es",
            "twitch.fr",
            "twitch.pt",
            "twitch.ru",
        )
        standard = (
            "structure-only",
            "graphsage-scratch",
            "linkx",
            "graphmae2-single",
            "multi-graph-shared-gfm",
            "domain-aligned-gfm",
        )
        link = (
            "structure-only",
            "common-neighbors",
            "adamic-adar",
            "graphsage-scratch",
            "linkx",
            "graphmae2-single",
            "multi-graph-shared-gfm",
            "domain-aligned-gfm",
        )
        twitch_validation = {
            "de": "en",
            "en": "es",
            "es": "fr",
            "fr": "pt",
            "pt": "ru",
            "ru": "de",
        }
        twitch_tasks = tuple(
            ExperimentTaskSpec(
                taskId=f"twitch.{target}.mature",
                targetGraphId=f"twitch.{target}",
                validationGraphId=f"twitch.{validation}",
                sourceGraphIds=tuple(
                    graph
                    for graph in cross
                    if graph not in {f"twitch.{target}", f"twitch.{validation}"}
                ),
                applicableMethods=("attribute-mlp", *standard),
                requiredMetrics=("auprc", "auroc", "macroF1"),
                primaryMetric="macroF1",
            )
            for target, validation in twitch_validation.items()
        )
        tasks = (
            ExperimentTaskSpec(
                taskId="tolokers.risk",
                targetGraphId="tolokers",
                sourceGraphIds=(*near, *cross),
                applicableMethods=("attribute-mlp", *standard),
                requiredMetrics=tuple(
                    sorted(("auprc", "auroc", "brier", "ece", "macroF1", "recallAtFpr"))
                ),
                primaryMetric="auprc",
            ),
            ExperimentTaskSpec(
                taskId="wiki-rfa.vote-sign",
                targetGraphId="wiki-rfa",
                sourceGraphIds=(*near, *cross),
                applicableMethods=standard,
                requiredMetrics=tuple(sorted(("auroc", "macroF1", "mcc", "negativeAuprc"))),
                primaryMetric="negativeAuprc",
            ),
            ExperimentTaskSpec(
                taskId="github.relation-completion",
                targetGraphId="github-musae",
                sourceGraphIds=(*near, *cross),
                applicableMethods=("attribute-mlp", *link),
                requiredMetrics=tuple(sorted(("auprc", "filteredMrr", "hitsAt10"))),
                primaryMetric="filteredMrr",
            ),
            ExperimentTaskSpec(
                taskId="email.relation-completion",
                targetGraphId="email-eu-core",
                sourceGraphIds=(*near, *cross),
                applicableMethods=link,
                requiredMetrics=tuple(sorted(("auprc", "filteredMrr", "hitsAt10"))),
                primaryMetric="filteredMrr",
            ),
            ExperimentTaskSpec(
                taskId="penn94.gender-offline",
                targetGraphId="facebook100.penn94",
                sourceGraphIds=near,
                applicableMethods=("attribute-mlp", *standard),
                requiredMetrics=tuple(sorted(("accuracy", "macroF1", "rocAuc"))),
                primaryMetric="macroF1",
            ),
            ExperimentTaskSpec(
                taskId="penn94.community-resilience",
                targetGraphId="facebook100.penn94",
                sourceGraphIds=near,
                applicableMethods=tuple(method for method in standard if method != "linkx"),
                requiredMetrics=("mae", "spearman"),
                primaryMetric="spearman",
            ),
            *twitch_tasks,
        )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-experiment-protocol/1.0",
            "methods": [item.model_dump(mode="python", by_alias=True) for item in methods],
            "tasks": [item.model_dump(mode="python", by_alias=True) for item in tasks],
            "labelBudgets": list(_BUDGETS),
            "seeds": list(_FORMAL_SEEDS),
            "maxElapsedSeconds": _MAX_ELAPSED_SECONDS,
            "maxCudaBytes": _MAX_CUDA_BYTES,
            "maxDataWaitRatio": 0.20,
        }
        payload["protocolHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_protocol(self):
        method_ids = tuple(item.method_id for item in self.methods)
        task_ids = tuple(item.task_id for item in self.tasks)
        if len(method_ids) != len(set(method_ids)) or len(task_ids) != len(set(task_ids)):
            raise ValueError("experiment method and task IDs must be unique")
        if any(not set(task.applicable_methods) <= set(method_ids) for task in self.tasks):
            raise ValueError("task applicability references an unknown method")
        if self.label_budgets != _BUDGETS or self.seeds != _FORMAL_SEEDS:
            raise ValueError("formal budgets and seeds are fixed")
        if (
            self.max_elapsed_seconds != _MAX_ELAPSED_SECONDS
            or self.max_cuda_bytes != _MAX_CUDA_BYTES
            or self.max_data_wait_ratio != 0.20
        ):
            raise ValueError("formal experiment resource limits are fixed")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"protocol_hash"})
        )
        if self.protocol_hash != expected:
            raise ValueError("protocolHash does not match fixed experiment protocol")
        if _FIXED_PROTOCOL_HASH is not None and self.protocol_hash != _FIXED_PROTOCOL_HASH:
            raise ValueError("experiment protocol inventory is not the fixed formal protocol")
        return self


_FIXED_PROTOCOL_HASH = ExperimentProtocol.fixed().protocol_hash


class ExperimentCell(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-cell/1.0"] = Field(
        alias="schemaVersion"
    )
    protocol_hash: str = Field(alias="protocolHash", pattern=_HASH)
    task_id: str = Field(alias="taskId", min_length=1)
    target_graph_id: str = Field(alias="targetGraphId", min_length=1)
    validation_graph_id: str | None = Field(default=None, alias="validationGraphId", min_length=1)
    method_id: str = Field(alias="methodId", min_length=1)
    backend: str = Field(min_length=1)
    trainable: bool
    label_budget: Literal["1", "5", "20", "full"] = Field(alias="labelBudget")
    seed: int
    pretraining_graph_ids: tuple[str, ...] = Field(alias="pretrainingGraphIds", strict=False)
    target_unlabeled_adaptation: bool = Field(alias="targetUnlabeledAdaptation")
    target_labels_in_pretraining: Literal[False] = Field(alias="targetLabelsInPretraining")
    required_metrics: tuple[str, ...] = Field(alias="requiredMetrics", strict=False)
    primary_metric: str = Field(alias="primaryMetric", min_length=1)
    slice_id: str = Field(alias="sliceId", pattern=_HASH)
    cell_id: str = Field(alias="cellId", pattern=_HASH)

    @model_validator(mode="after")
    def validate_cell(self):
        if self.target_graph_id in self.pretraining_graph_ids or (
            self.validation_graph_id is not None
            and self.validation_graph_id in self.pretraining_graph_ids
        ):
            raise ValueError("experiment cell pretraining must be graph-disjoint")
        base = self.model_dump(
            mode="python", by_alias=True, exclude={"slice_id", "cell_id", "seed"}
        )
        if self.slice_id != canonical_sha256(base):
            raise ValueError("sliceId does not match experiment cell slice")
        complete = self.model_dump(mode="python", by_alias=True, exclude={"cell_id"})
        if self.cell_id != canonical_sha256(complete):
            raise ValueError("cellId does not match complete experiment cell")
        return self


def build_experiment_matrix(
    protocol: ExperimentProtocol,
) -> tuple[ExperimentCell, ...]:
    method_by_id = {method.method_id: method for method in protocol.methods}
    cells: list[ExperimentCell] = []
    for task in protocol.tasks:
        for method_id in task.applicable_methods:
            method = method_by_id[method_id]
            if method.backend == "pretrained-single":
                pretraining = task.source_graph_ids[:1]
            elif method.backend in {"pretrained-shared", "pretrained-aligned"}:
                pretraining = task.source_graph_ids
            else:
                pretraining = ()
            for budget in protocol.label_budgets:
                for seed in protocol.seeds:
                    base: dict[str, Any] = {
                        "schemaVersion": "socialgraph-fm.core-experiment-cell/1.0",
                        "protocolHash": protocol.protocol_hash,
                        "taskId": task.task_id,
                        "targetGraphId": task.target_graph_id,
                        "validationGraphId": task.validation_graph_id,
                        "methodId": method.method_id,
                        "backend": method.backend,
                        "trainable": method.trainable,
                        "labelBudget": budget,
                        "pretrainingGraphIds": list(pretraining),
                        "targetUnlabeledAdaptation": method.target_unlabeled_adaptation,
                        "targetLabelsInPretraining": False,
                        "requiredMetrics": list(task.required_metrics),
                        "primaryMetric": task.primary_metric,
                    }
                    slice_id = canonical_sha256(base)
                    complete = {**base, "seed": seed, "sliceId": slice_id}
                    complete["cellId"] = canonical_sha256(complete)
                    cells.append(ExperimentCell.model_validate(complete))
    return tuple(cells)


class GraphManifestBinding(_StrictModel):
    graph_id: str = Field(alias="graphId", min_length=1)
    manifest_hash: str = Field(alias="manifestHash", pattern=_HASH)


class GfmFamilyEvidence(_StrictModel):
    method_id: str = Field(alias="methodId", min_length=1)
    encoder_initialization: Literal["random", "pretrained"] = Field(alias="encoderInitialization")
    pretraining_scope: Literal["none", "single-source", "multi-source"] = Field(
        alias="pretrainingScope"
    )
    field_reconstruction_weight: float = Field(alias="fieldReconstructionWeight")
    edge_reconstruction_weight: float = Field(alias="edgeReconstructionWeight")
    alignment_candidates: tuple[float, ...] = Field(alias="alignmentCandidates", strict=False)
    target_unlabeled_adaptation: bool = Field(alias="targetUnlabeledAdaptation")
    target_labels_in_pretraining: Literal[False] = Field(alias="targetLabelsInPretraining")

    @classmethod
    def from_spec(cls, spec: GfmFamilySpec) -> GfmFamilyEvidence:
        return cls(
            methodId=spec.method_id,
            encoderInitialization=spec.encoder_initialization,
            pretrainingScope=spec.pretraining_scope,
            fieldReconstructionWeight=spec.field_reconstruction_weight,
            edgeReconstructionWeight=spec.edge_reconstruction_weight,
            alignmentCandidates=spec.alignment_candidates,
            targetUnlabeledAdaptation=spec.target_unlabeled_adaptation,
            targetLabelsInPretraining=False,
        )


class ExperimentRecipeEvidence(_StrictModel):
    """Fixed method plus source/target manifest provenance for one exact cell."""

    schema_version: Literal["socialgraph-fm.core-experiment-recipe/1.0"] = Field(
        alias="schemaVersion"
    )
    cell: ExperimentCell
    method: MethodSpec
    family: GfmFamilyEvidence | None
    source_manifests: tuple[GraphManifestBinding, ...] = Field(
        alias="sourceManifests", strict=False
    )
    target_manifest: GraphManifestBinding = Field(alias="targetManifest")
    validation_manifest: GraphManifestBinding | None = Field(alias="validationManifest")
    recipe_hash: str = Field(alias="recipeHash", pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        cell: ExperimentCell,
        manifest_hashes: dict[str, str],
    ) -> ExperimentRecipeEvidence:
        required = (*cell.pretraining_graph_ids, cell.target_graph_id)
        if cell.validation_graph_id is not None:
            required = (*required, cell.validation_graph_id)
        if set(manifest_hashes) != set(required):
            raise ValueError(
                "recipe manifest inventory must bind every source, target, and validation graph"
            )
        protocol = ExperimentProtocol.fixed()
        method = next(item for item in protocol.methods if item.method_id == cell.method_id)
        family_spec = next(
            (item for item in fixed_gfm_family_specs() if item.method_id == cell.method_id),
            None,
        )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-experiment-recipe/1.0",
            "cell": cell.model_dump(mode="python", by_alias=True),
            "method": method.model_dump(mode="python", by_alias=True),
            "family": (
                None
                if family_spec is None
                else GfmFamilyEvidence.from_spec(family_spec).model_dump(
                    mode="python", by_alias=True
                )
            ),
            "sourceManifests": [
                {"graphId": graph_id, "manifestHash": manifest_hashes[graph_id]}
                for graph_id in cell.pretraining_graph_ids
            ],
            "targetManifest": {
                "graphId": cell.target_graph_id,
                "manifestHash": manifest_hashes[cell.target_graph_id],
            },
            "validationManifest": (
                None
                if cell.validation_graph_id is None
                else {
                    "graphId": cell.validation_graph_id,
                    "manifestHash": manifest_hashes[cell.validation_graph_id],
                }
            ),
        }
        payload["recipeHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_recipe(self):
        protocol = ExperimentProtocol.fixed()
        if self.cell.protocol_hash != protocol.protocol_hash:
            raise ValueError("experiment recipe cell is outside the fixed protocol")
        method = next(
            (item for item in protocol.methods if item.method_id == self.cell.method_id), None
        )
        if method is None or self.method != method:
            raise ValueError("experiment recipe method differs from the fixed protocol")
        family_spec = next(
            (item for item in fixed_gfm_family_specs() if item.method_id == self.cell.method_id),
            None,
        )
        expected_family = None if family_spec is None else GfmFamilyEvidence.from_spec(family_spec)
        if self.family != expected_family:
            raise ValueError("experiment recipe family differs from the fixed GFM ladder")
        if (
            tuple(item.graph_id for item in self.source_manifests)
            != self.cell.pretraining_graph_ids
        ):
            raise ValueError("experiment recipe source inventory differs from the cell")
        if self.target_manifest.graph_id != self.cell.target_graph_id or (
            (self.validation_manifest is None) != (self.cell.validation_graph_id is None)
        ):
            raise ValueError("experiment recipe target/validation inventory differs from the cell")
        if self.validation_manifest is not None and (
            self.validation_manifest.graph_id != self.cell.validation_graph_id
        ):
            raise ValueError("experiment recipe validation graph differs from the cell")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"recipe_hash"})
        )
        if self.recipe_hash != expected:
            raise ValueError("recipeHash does not match experiment recipe evidence")
        return self


class ExperimentExecutionConfigEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-execution-config/1.0"] = Field(
        alias="schemaVersion"
    )
    cell_id: str = Field(alias="cellId", pattern=_HASH)
    method_id: str = Field(alias="methodId", min_length=1)
    backend: str = Field(min_length=1)
    recipe_hash: str = Field(alias="recipeHash", pattern=_HASH)
    training_config: dict[str, Any] | None = Field(alias="trainingConfig")
    training_config_hash: str | None = Field(alias="trainingConfigHash", pattern=_HASH)
    config_hash: str = Field(alias="configHash", pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        cell: ExperimentCell,
        recipe: ExperimentRecipeEvidence,
        training_config: TrainingConfig | None,
    ) -> ExperimentExecutionConfigEvidence:
        if recipe.cell != cell:
            raise ValueError("execution config recipe differs from the cell")
        serialized_config = None if training_config is None else training_config.to_dict()
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-execution-config/1.0",
            "cellId": cell.cell_id,
            "methodId": cell.method_id,
            "backend": cell.backend,
            "recipeHash": recipe.recipe_hash,
            "trainingConfig": serialized_config,
            "trainingConfigHash": (
                None if serialized_config is None else canonical_sha256(serialized_config)
            ),
        }
        payload["configHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_execution_config(self):
        if self.training_config is None:
            if self.training_config_hash is not None:
                raise ValueError("heuristic execution config cannot claim a trainer config hash")
        else:
            parsed = TrainingConfig(**self.training_config)
            if parsed.to_dict() != self.training_config:
                raise ValueError("training config is not the canonical core configuration")
            if self.training_config_hash != canonical_sha256(self.training_config):
                raise ValueError("trainingConfigHash does not match the trainer configuration")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"config_hash"})
        )
        if self.config_hash != expected:
            raise ValueError("configHash does not match execution configuration evidence")
        return self


class ExperimentTrainingDataEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-training-data-inventory/1.0"] = Field(
        alias="schemaVersion"
    )
    cell_id: str = Field(alias="cellId", pattern=_HASH)
    recipe_hash: str = Field(alias="recipeHash", pattern=_HASH)
    dataset_manifests: tuple[GraphManifestBinding, ...] = Field(
        alias="datasetManifests", strict=False, min_length=1
    )
    target_split_inventory_hash: str = Field(alias="targetSplitInventoryHash", pattern=_HASH)
    head_data_hash: str | None = Field(default=None, alias="headDataHash", pattern=_HASH)
    inventory_hash: str = Field(alias="inventoryHash", pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        cell: ExperimentCell,
        recipe: ExperimentRecipeEvidence,
        target_split_inventory_hash: str,
        head_data_hash: str | None,
    ) -> ExperimentTrainingDataEvidence:
        if recipe.cell != cell:
            raise ValueError("training data recipe differs from the cell")
        bindings = (
            *recipe.source_manifests,
            recipe.target_manifest,
            *((recipe.validation_manifest,) if recipe.validation_manifest is not None else ()),
        )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-training-data-inventory/1.0",
            "cellId": cell.cell_id,
            "recipeHash": recipe.recipe_hash,
            "datasetManifests": [
                item.model_dump(mode="python", by_alias=True) for item in bindings
            ],
            "targetSplitInventoryHash": target_split_inventory_hash,
            "headDataHash": head_data_hash,
        }
        payload["inventoryHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_inventory(self):
        graph_ids = tuple(item.graph_id for item in self.dataset_manifests)
        if len(graph_ids) != len(set(graph_ids)):
            raise ValueError("training data graph inventory must be unique")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"inventory_hash"})
        )
        if self.inventory_hash != expected:
            raise ValueError("inventoryHash does not match experiment training data")
        return self


class ResourceTelemetrySample(_StrictModel):
    monotonic_seconds: float = Field(alias="monotonicSeconds", ge=0.0)
    cumulative_data_wait_seconds: float = Field(alias="cumulativeDataWaitSeconds", ge=0.0)
    optimizer_step: int = Field(alias="optimizerStep", ge=0, le=10_000)
    cuda_allocated_bytes: int = Field(alias="cudaAllocatedBytes", ge=0)


class ResourceTelemetryEvidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-resource-telemetry/1.0"] = Field(
        alias="schemaVersion"
    )
    cell_id: str = Field(alias="cellId", pattern=_HASH)
    phase: Literal["smoke", "dev", "formal"]
    samples: tuple[ResourceTelemetrySample, ...] = Field(strict=False, min_length=2)
    optimizer_steps: int = Field(alias="optimizerSteps", ge=0, le=10_000)
    elapsed_seconds: float = Field(alias="elapsedSeconds", gt=0.0)
    data_wait_seconds: float = Field(alias="dataWaitSeconds", ge=0.0)
    peak_cuda_bytes: int = Field(alias="peakCudaBytes", ge=0)
    telemetry_hash: str = Field(alias="telemetryHash", pattern=_HASH)

    @classmethod
    def create(
        cls,
        *,
        cell_id: str,
        phase: Literal["smoke", "dev", "formal"],
        samples: tuple[ResourceTelemetrySample, ...],
    ) -> ResourceTelemetryEvidence:
        if len(samples) < 2:
            raise ValueError("resource telemetry requires start and end samples")
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-resource-telemetry/1.0",
            "cellId": cell_id,
            "phase": phase,
            "samples": [item.model_dump(mode="python", by_alias=True) for item in samples],
            "optimizerSteps": samples[-1].optimizer_step - samples[0].optimizer_step,
            "elapsedSeconds": samples[-1].monotonic_seconds - samples[0].monotonic_seconds,
            "dataWaitSeconds": (
                samples[-1].cumulative_data_wait_seconds - samples[0].cumulative_data_wait_seconds
            ),
            "peakCudaBytes": max(item.cuda_allocated_bytes for item in samples),
        }
        payload["telemetryHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_telemetry(self):
        times = tuple(item.monotonic_seconds for item in self.samples)
        waits = tuple(item.cumulative_data_wait_seconds for item in self.samples)
        steps = tuple(item.optimizer_step for item in self.samples)
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("resource telemetry monotonic time must strictly increase")
        if any(right < left for left, right in zip(waits, waits[1:])) or any(
            right < left for left, right in zip(steps, steps[1:])
        ):
            raise ValueError("resource telemetry counters must not decrease")
        derived = (
            steps[-1] - steps[0],
            times[-1] - times[0],
            waits[-1] - waits[0],
            max(item.cuda_allocated_bytes for item in self.samples),
        )
        observed = (
            self.optimizer_steps,
            self.elapsed_seconds,
            self.data_wait_seconds,
            self.peak_cuda_bytes,
        )
        if observed != derived:
            raise ValueError("resource telemetry summaries must be derived from samples")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"telemetry_hash"})
        )
        if self.telemetry_hash != expected:
            raise ValueError("telemetryHash does not match resource samples")
        return self


class ExperimentRunRecord(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-run/1.4"] = Field(
        alias="schemaVersion"
    )
    cell: ExperimentCell
    phase: Literal["smoke", "dev", "formal"]
    preflight_evidence_hash: str = Field(alias="preflightEvidenceHash", pattern=_HASH)
    dataset_manifest_hash: str = Field(alias="datasetManifestHash", pattern=_HASH)
    split_manifest_hash: str = Field(alias="splitManifestHash", pattern=_HASH)
    split_inventory_hash: str = Field(alias="splitInventoryHash", pattern=_HASH)
    evaluation_fold_ids: tuple[str, ...] = Field(
        alias="evaluationFoldIds", strict=False, min_length=1
    )
    recipe_hash: str = Field(alias="recipeHash", pattern=_HASH)
    config_hash: str = Field(alias="configHash", pattern=_HASH)
    training_data_hash: str = Field(alias="trainingDataHash", pattern=_HASH)
    head_data_hash: str | None = Field(default=None, alias="headDataHash", pattern=_HASH)
    code_hash: str = Field(alias="codeHash", pattern=_HASH)
    environment_hash: str = Field(alias="environmentHash", pattern=_HASH)
    structure_cache_hash: str = Field(alias="structureCacheHash", pattern=_HASH)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH)
    label_artifact_hash: str = Field(alias="labelArtifactHash", pattern=_HASH)
    head_report_hash: str | None = Field(default=None, alias="headReportHash", pattern=_HASH)
    calibration_hash: str | None = Field(default=None, alias="calibrationHash", pattern=_HASH)
    checkpoint_sha256: str | None = Field(default=None, alias="checkpointSha256", pattern=_HASH)
    best_checkpoint_sha256: str | None = Field(
        default=None, alias="bestCheckpointSha256", pattern=_HASH
    )
    optimizer_steps: int = Field(alias="optimizerSteps", ge=0, le=10_000)
    elapsed_seconds: float = Field(alias="elapsedSeconds", gt=0.0)
    data_wait_seconds: float = Field(alias="dataWaitSeconds", ge=0.0)
    peak_cuda_bytes: int = Field(alias="peakCudaBytes", ge=0)
    telemetry_hash: str = Field(alias="telemetryHash", pattern=_HASH)
    telemetry_schema_version: str = Field(alias="telemetrySchemaVersion", min_length=1)
    telemetry_evidence_scope: Literal["sealed-runtime", "legacy-unverified"] = Field(
        alias="telemetryEvidenceScope"
    )
    telemetry_receipt_hash: str | None = Field(
        default=None, alias="telemetryReceiptHash", pattern=_HASH
    )
    fold_evaluation_inventory_hash: str | None = Field(
        default=None, alias="foldEvaluationInventoryHash", pattern=_HASH
    )
    metrics: TaskMetricSet
    artifacts: tuple[ExperimentArtifactRef, ...] = Field(strict=False)
    artifact_inventory_hash: str = Field(alias="artifactInventoryHash", pattern=_HASH)
    failed_gates: tuple[str, ...] = Field(alias="failedGates", strict=False)
    promotable: bool
    record_hash: str = Field(alias="recordHash", pattern=_HASH)

    @classmethod
    def create(cls, **values: Any) -> ExperimentRunRecord:
        cell: ExperimentCell = values["cell"]
        metrics: TaskMetricSet = values["metrics"]
        telemetry_value = values["telemetry"]
        telemetry: ResourceTelemetryRecord | ResourceTelemetryEvidence
        if type(telemetry_value) is VerifiedResourceTelemetry:
            telemetry = verify_resource_telemetry(telemetry_value)
            telemetry_scope: Literal["sealed-runtime", "legacy-unverified"] = "sealed-runtime"
        elif type(telemetry_value) is ResourceTelemetryEvidence:
            telemetry = telemetry_value
            telemetry_scope = "legacy-unverified"
        else:
            raise TypeError(
                "run telemetry must be exact VerifiedResourceTelemetry or legacy evidence"
            )
        receipt = values.get("telemetry_receipt")
        if receipt is not None and type(receipt) is not TelemetryReceipt:
            raise TypeError("telemetry receipt must be an exact authenticated receipt")
        if receipt is not None:
            reparsed_receipt = TelemetryReceipt.model_validate(
                receipt.model_dump(mode="python", by_alias=True)
            )
            if reparsed_receipt != receipt:
                raise ValueError("telemetry receipt changed during exact revalidation")
            if telemetry_scope != "sealed-runtime" or type(telemetry) is not ResourceTelemetryRecord:
                raise ValueError("telemetry receipt requires exact sealed runtime telemetry")
            if (
                receipt.cell_id != cell.cell_id
                or receipt.fold_id != "cell-run"
                or receipt.phase != values["phase"]
                or receipt.config_hash != telemetry.config_hash
                or receipt.data_hash != telemetry.data_hash
                or receipt.code_hash != telemetry.code_hash
                or receipt.environment_hash != telemetry.environment_hash
                or receipt.telemetry_hash != telemetry.telemetry_hash
                or receipt.telemetry_record_hash
                != canonical_sha256(telemetry.model_dump(mode="python", by_alias=True))
                or receipt.final_optimizer_step != telemetry.final_optimizer_step
                or receipt.final_model_state_hash != telemetry.final_model_state_hash
                or receipt.final_fit_state_hash != telemetry.final_fit_state_hash
                or receipt.latest_checkpoint_semantic_hash
                != telemetry.latest_checkpoint_semantic_hash
                or receipt.best_checkpoint_semantic_hash
                != telemetry.best_checkpoint_semantic_hash
                or (
                    cell.trainable
                    and (
                        receipt.latest_checkpoint_semantic_hash
                        != values["checkpoint_sha256"]
                        or receipt.best_checkpoint_semantic_hash
                        != values["best_checkpoint_sha256"]
                    )
                )
            ):
                raise ValueError(
                    "telemetry receipt does not bind the experiment telemetry and checkpoints"
                )
        if telemetry.cell_id != cell.cell_id or telemetry.phase != values["phase"]:
            raise ValueError("resource telemetry does not belong to the experiment cell")
        artifacts = tuple(sorted(tuple(values.get("artifacts", ())), key=lambda item: item.role))
        failed = _run_failed_gates(
            cell=cell,
            phase=values["phase"],
            dataset_manifest_hash=values["dataset_manifest_hash"],
            split_manifest_hash=values["split_manifest_hash"],
            split_inventory_hash=values["split_inventory_hash"],
            recipe_hash=values["recipe_hash"],
            config_hash=values["config_hash"],
            training_data_hash=values["training_data_hash"],
            head_data_hash=values["head_data_hash"],
            code_hash=values["code_hash"],
            environment_hash=values["environment_hash"],
            structure_cache_hash=values["structure_cache_hash"],
            adapter_schema_hash=values["adapter_schema_hash"],
            label_artifact_hash=values["label_artifact_hash"],
            optimizer_steps=telemetry.optimizer_steps,
            elapsed_seconds=telemetry.elapsed_seconds,
            data_wait_seconds=telemetry.data_wait_seconds,
            peak_cuda_bytes=telemetry.peak_cuda_bytes,
            telemetry_hash=telemetry.telemetry_hash,
            telemetry_evidence_scope=telemetry_scope,
            telemetry_receipt_hash=None if receipt is None else receipt.receipt_hash,
            fold_evaluation_inventory_hash=values.get("fold_evaluation_inventory_hash"),
            checkpoint_sha256=values["checkpoint_sha256"],
            best_checkpoint_sha256=values["best_checkpoint_sha256"],
            head_report_hash=values["head_report_hash"],
            calibration_hash=values["calibration_hash"],
            metrics=metrics,
            artifacts=artifacts,
        )
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-experiment-run/1.4",
            "cell": cell.model_dump(mode="python", by_alias=True),
            "phase": values["phase"],
            "preflightEvidenceHash": values["preflight_evidence_hash"],
            "datasetManifestHash": values["dataset_manifest_hash"],
            "splitManifestHash": values["split_manifest_hash"],
            "splitInventoryHash": values["split_inventory_hash"],
            "evaluationFoldIds": list(values["evaluation_fold_ids"]),
            "recipeHash": values["recipe_hash"],
            "configHash": values["config_hash"],
            "trainingDataHash": values["training_data_hash"],
            "headDataHash": values["head_data_hash"],
            "codeHash": values["code_hash"],
            "environmentHash": values["environment_hash"],
            "structureCacheHash": values["structure_cache_hash"],
            "adapterSchemaHash": values["adapter_schema_hash"],
            "labelArtifactHash": values["label_artifact_hash"],
            "headReportHash": values["head_report_hash"],
            "calibrationHash": values["calibration_hash"],
            "checkpointSha256": values["checkpoint_sha256"],
            "bestCheckpointSha256": values["best_checkpoint_sha256"],
            "optimizerSteps": telemetry.optimizer_steps,
            "elapsedSeconds": telemetry.elapsed_seconds,
            "dataWaitSeconds": telemetry.data_wait_seconds,
            "peakCudaBytes": telemetry.peak_cuda_bytes,
            "telemetryHash": telemetry.telemetry_hash,
            "telemetrySchemaVersion": telemetry.schema_version,
            "telemetryEvidenceScope": telemetry_scope,
            "telemetryReceiptHash": None if receipt is None else receipt.receipt_hash,
            "foldEvaluationInventoryHash": values.get("fold_evaluation_inventory_hash"),
            "metrics": metrics.model_dump(mode="python", by_alias=True),
            "artifacts": [item.model_dump(mode="python", by_alias=True) for item in artifacts],
            "artifactInventoryHash": canonical_sha256(artifacts),
            "failedGates": list(failed),
            "promotable": not failed,
        }
        payload["recordHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_record(self):
        failed = _run_failed_gates(
            cell=self.cell,
            phase=self.phase,
            dataset_manifest_hash=self.dataset_manifest_hash,
            split_manifest_hash=self.split_manifest_hash,
            split_inventory_hash=self.split_inventory_hash,
            recipe_hash=self.recipe_hash,
            config_hash=self.config_hash,
            training_data_hash=self.training_data_hash,
            head_data_hash=self.head_data_hash,
            code_hash=self.code_hash,
            environment_hash=self.environment_hash,
            structure_cache_hash=self.structure_cache_hash,
            adapter_schema_hash=self.adapter_schema_hash,
            label_artifact_hash=self.label_artifact_hash,
            optimizer_steps=self.optimizer_steps,
            elapsed_seconds=self.elapsed_seconds,
            data_wait_seconds=self.data_wait_seconds,
            peak_cuda_bytes=self.peak_cuda_bytes,
            telemetry_hash=self.telemetry_hash,
            telemetry_evidence_scope=self.telemetry_evidence_scope,
            telemetry_receipt_hash=self.telemetry_receipt_hash,
            fold_evaluation_inventory_hash=self.fold_evaluation_inventory_hash,
            checkpoint_sha256=self.checkpoint_sha256,
            best_checkpoint_sha256=self.best_checkpoint_sha256,
            head_report_hash=self.head_report_hash,
            calibration_hash=self.calibration_hash,
            metrics=self.metrics,
            artifacts=self.artifacts,
        )
        if self.failed_gates != failed or self.promotable != (not failed):
            raise ValueError("run promotability must be derived from fixed gates")
        if self.evaluation_fold_ids != tuple(sorted(set(self.evaluation_fold_ids))):
            raise ValueError("evaluation fold IDs must be unique and sorted")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"record_hash"})
        )
        if self.record_hash != expected:
            raise ValueError("recordHash does not match raw experiment record")
        if self.artifact_inventory_hash != canonical_sha256(self.artifacts):
            raise ValueError("artifactInventoryHash does not match artifact evidence")
        return self


def _run_failed_gates(
    *,
    cell: ExperimentCell,
    phase: str,
    dataset_manifest_hash: str,
    split_manifest_hash: str,
    split_inventory_hash: str,
    recipe_hash: str,
    config_hash: str,
    training_data_hash: str,
    head_data_hash: str | None,
    code_hash: str,
    environment_hash: str,
    structure_cache_hash: str,
    adapter_schema_hash: str,
    label_artifact_hash: str,
    optimizer_steps: int,
    elapsed_seconds: float,
    data_wait_seconds: float,
    peak_cuda_bytes: int,
    telemetry_hash: str,
    telemetry_evidence_scope: str,
    telemetry_receipt_hash: str | None,
    fold_evaluation_inventory_hash: str | None,
    checkpoint_sha256: str | None,
    best_checkpoint_sha256: str | None,
    head_report_hash: str | None,
    calibration_hash: str | None,
    metrics: TaskMetricSet,
    artifacts: tuple[ExperimentArtifactRef, ...],
) -> tuple[str, ...]:
    failed: list[str] = []
    if phase != "formal":
        failed.append("phase-eligibility")
    if telemetry_evidence_scope != "sealed-runtime":
        failed.append("resource-telemetry-unverified")
    if telemetry_receipt_hash is None:
        failed.append("resource-telemetry-receipt")
    if fold_evaluation_inventory_hash is None:
        failed.append("fold-evaluation-inventory")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds > _MAX_ELAPSED_SECONDS:
        failed.append("elapsed-time")
    if (
        not math.isfinite(data_wait_seconds)
        or data_wait_seconds > elapsed_seconds
        or data_wait_seconds / elapsed_seconds >= 0.20
    ):
        failed.append("data-wait-ratio")
    if peak_cuda_bytes >= _MAX_CUDA_BYTES:
        failed.append("peak-cuda-memory")
    if (
        metrics.task_id != cell.task_id
        or tuple(item.name for item in metrics.metrics) != cell.required_metrics
    ):
        failed.append("metric-inventory")
    expected_artifacts = {
        "adapter-schema": adapter_schema_hash,
        "code": code_hash,
        "configuration": config_hash,
        "dataset-manifest": dataset_manifest_hash,
        "environment": environment_hash,
        "experiment-recipe": recipe_hash,
        **(
            {"fold-evaluation-inventory": fold_evaluation_inventory_hash}
            if fold_evaluation_inventory_hash is not None
            else {}
        ),
        "labels": label_artifact_hash,
        "predictions": metrics.prediction_hash,
        "resource-telemetry": telemetry_hash,
        **(
            {"telemetry-receipt": telemetry_receipt_hash}
            if telemetry_receipt_hash is not None
            else {}
        ),
        "split-manifest": split_manifest_hash,
        "split-inventory": split_inventory_hash,
        "structure-cache": structure_cache_hash,
        "targets": metrics.target_hash,
        "training-data": training_data_hash,
    }
    if metrics.threshold_hash is not None:
        expected_artifacts["threshold"] = metrics.threshold_hash
    if cell.trainable:
        if not 2_000 <= optimizer_steps <= 10_000:
            failed.append("optimizer-steps")
        if checkpoint_sha256 is None or best_checkpoint_sha256 is None:
            failed.append("best-checkpoint")
        else:
            expected_artifacts["latest-checkpoint"] = checkpoint_sha256
            expected_artifacts["best-checkpoint"] = best_checkpoint_sha256
        if head_report_hash is None:
            failed.append("supervised-head")
        else:
            expected_artifacts["head-report"] = head_report_hash
        if head_data_hash is None:
            failed.append("supervised-head-data")
        else:
            expected_artifacts["head-data"] = head_data_hash
        requires_calibration = cell.task_id not in {
            "penn94.gender-offline",
            "penn94.community-resilience",
        }
        if requires_calibration and calibration_hash is None:
            failed.append("calibration")
        elif not requires_calibration and calibration_hash is not None:
            failed.append("calibration")
        elif calibration_hash is not None:
            expected_artifacts["calibration-report"] = calibration_hash
    elif (
        optimizer_steps != 0
        or checkpoint_sha256 is not None
        or best_checkpoint_sha256 is not None
        or head_data_hash is not None
    ):
        failed.append("heuristic-runtime")
    observed_roles = tuple(item.role for item in artifacts)
    if (
        observed_roles != tuple(sorted(expected_artifacts))
        or len({item.relative_path for item in artifacts}) != len(artifacts)
        or any(
            item.semantic_hash != expected_artifacts[item.role]
            or (
                item.role in {"latest-checkpoint", "best-checkpoint"}
                and item.byte_sha256 != item.semantic_hash
            )
            for item in artifacts
        )
    ):
        failed.append("artifact-inventory")
    return tuple(failed)


def _canonical_bytes(model: BaseModel) -> bytes:
    return (canonical_json(model) + "\n").encode("utf-8")


class ExperimentLedger:
    def __init__(self, runtime_root: Path) -> None:
        self.authorized_root = secure_existing_root(runtime_root)
        lease = _PublicationParentLease(
            self.authorized_root,
            self.authorized_root / "experiments-core" / "raw-runs",
            create=True,
        )
        try:
            self.root = lease.parent
        finally:
            lease.close()

    def _path(self, cell_id: str) -> Path:
        if len(cell_id) != 64 or any(character not in "0123456789abcdef" for character in cell_id):
            raise ValueError("ledger cell ID must be a lowercase SHA-256")
        return self.root / f"{cell_id}.json"

    def publish_run(self, record: ExperimentRunRecord) -> Path:
        target = self._path(record.cell.cell_id)
        serialized = _canonical_bytes(record)
        if len(serialized) > _MAX_LEDGER_BYTES:
            raise ValueError("raw experiment record exceeds the ledger size limit")
        _publish_immutable_exact(
            self.authorized_root,
            target,
            serialized,
            conflict_message="conflicting raw experiment record already exists",
        )
        if self.load_run(record.cell.cell_id) != record:
            raise ValueError("published raw experiment record failed exact reload")
        return target

    def load_run(self, cell_id: str) -> ExperimentRunRecord:
        target = self._path(cell_id)
        serialized = read_confined_snapshot(
            self.authorized_root,
            target.relative_to(self.authorized_root).as_posix(),
            max_bytes=_MAX_LEDGER_BYTES,
        )
        record = ExperimentRunRecord.model_validate_json(serialized)
        if record.cell.cell_id != cell_id or serialized != _canonical_bytes(record):
            raise ValueError("raw experiment record is not exact canonical evidence")
        return record


def _metric(record: ExperimentRunRecord) -> float:
    return next(
        item.value for item in record.metrics.metrics if item.name == record.cell.primary_metric
    )


def _bootstrap_interval(values: tuple[float, ...], *, seed_material: str) -> tuple[float, float]:
    generator = random.Random(int(seed_material[:16], 16))
    count = len(values)
    samples = sorted(
        math.fsum(values[generator.randrange(count)] for _ in range(count)) / count
        for _ in range(2_000)
    )
    return samples[49], samples[1949]


class ExperimentAggregate(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-experiment-aggregate/1.0"] = Field(
        alias="schemaVersion"
    )
    slice_id: str = Field(alias="sliceId", pattern=_HASH)
    task_id: str = Field(alias="taskId", min_length=1)
    method_id: str = Field(alias="methodId", min_length=1)
    label_budget: str = Field(alias="labelBudget", min_length=1)
    primary_metric: str = Field(alias="primaryMetric", min_length=1)
    seeds: tuple[int, ...] = Field(strict=False)
    record_hashes: tuple[str, ...] = Field(alias="recordHashes", strict=False)
    mean: float
    sample_std: float = Field(alias="sampleStd", ge=0.0)
    ci_lower: float = Field(alias="ciLower")
    ci_upper: float = Field(alias="ciUpper")
    promotable: bool
    aggregate_hash: str = Field(alias="aggregateHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_aggregate(self):
        if self.seeds != _FORMAL_SEEDS or len(self.record_hashes) != len(self.seeds):
            raise ValueError("aggregate requires the exact five formal seeds")
        if not all(
            math.isfinite(value)
            for value in (self.mean, self.sample_std, self.ci_lower, self.ci_upper)
        ):
            raise ValueError("aggregate statistics must be finite")
        if self.ci_lower > self.ci_upper:
            raise ValueError("aggregate confidence interval is invalid")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"aggregate_hash"})
        )
        if self.aggregate_hash != expected:
            raise ValueError("aggregateHash does not match aggregate evidence")
        return self


def aggregate_experiment(
    protocol: ExperimentProtocol, records: tuple[ExperimentRunRecord, ...]
) -> ExperimentAggregate:
    if len(records) != 5:
        raise ValueError("aggregate requires exactly five seeds")
    ordered = tuple(sorted(records, key=lambda record: record.cell.seed))
    first = ordered[0].cell
    if tuple(record.cell.seed for record in ordered) != protocol.seeds:
        raise ValueError("aggregate requires the exact five seeds")
    if any(
        record.cell.slice_id != first.slice_id
        or record.cell.protocol_hash != protocol.protocol_hash
        or record.metrics.task_id != first.task_id
        for record in ordered
    ):
        raise ValueError("aggregate records must belong to one fixed protocol slice")
    if not all(record.promotable for record in ordered):
        raise ValueError("aggregate requires promotable hash-bound raw records")
    fixed_cells = {cell.cell_id for cell in build_experiment_matrix(protocol)}
    if any(record.cell.cell_id not in fixed_cells for record in ordered):
        raise ValueError("aggregate record is outside the fixed experiment matrix")
    values = tuple(_metric(record) for record in ordered)
    mean = math.fsum(values) / len(values)
    sample_std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))
    lower, upper = _bootstrap_interval(values, seed_material=first.slice_id)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-experiment-aggregate/1.0",
        "sliceId": first.slice_id,
        "taskId": first.task_id,
        "methodId": first.method_id,
        "labelBudget": first.label_budget,
        "primaryMetric": first.primary_metric,
        "seeds": [record.cell.seed for record in ordered],
        "recordHashes": [record.record_hash for record in ordered],
        "mean": mean,
        "sampleStd": sample_std,
        "ciLower": lower,
        "ciUpper": upper,
        "promotable": True,
    }
    payload["aggregateHash"] = canonical_sha256(payload)
    return ExperimentAggregate.model_validate(payload)


class TransferDecision(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-transfer-decision/1.0"] = Field(
        alias="schemaVersion"
    )
    task_id: str = Field(alias="taskId", min_length=1)
    candidate_method_id: str = Field(alias="candidateMethodId", min_length=1)
    winning_budget_count: int = Field(alias="winningBudgetCount", ge=0, le=4)
    aggregate_improvement: float = Field(alias="aggregateImprovement")
    ci_lower: float = Field(alias="ciLower")
    ci_upper: float = Field(alias="ciUpper")
    transfer_advantage: bool = Field(alias="transferAdvantage")
    scratch_record_hashes: tuple[str, ...] = Field(alias="scratchRecordHashes", strict=False)
    candidate_record_hashes: tuple[str, ...] = Field(alias="candidateRecordHashes", strict=False)
    decision_hash: str = Field(alias="decisionHash", pattern=_HASH)

    @model_validator(mode="after")
    def validate_decision(self):
        expected_decision = (
            self.winning_budget_count >= 3
            and self.aggregate_improvement >= 0.02
            and self.ci_lower > 0.0
        )
        if self.transfer_advantage != expected_decision:
            raise ValueError("transferAdvantage must be derived from fixed gates")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"decision_hash"})
        )
        if self.decision_hash != expected:
            raise ValueError("decisionHash does not match transfer evidence")
        return self


def derive_transfer_advantage(
    protocol: ExperimentProtocol,
    scratch_records: list[ExperimentRunRecord],
    candidate_records: list[ExperimentRunRecord],
) -> TransferDecision:
    expected_count = len(protocol.label_budgets) * len(protocol.seeds)
    if len(scratch_records) != expected_count or len(candidate_records) != expected_count:
        raise ValueError("transfer comparison requires four complete five-seed budgets")
    scratch = {(item.cell.label_budget, item.cell.seed): item for item in scratch_records}
    candidate = {(item.cell.label_budget, item.cell.seed): item for item in candidate_records}
    expected_keys = {(budget, seed) for budget in protocol.label_budgets for seed in protocol.seeds}
    if set(scratch) != expected_keys or set(candidate) != expected_keys:
        raise ValueError("transfer comparison seed/budget inventory is incomplete")
    task_ids = {item.cell.task_id for item in (*scratch_records, *candidate_records)}
    candidate_methods = {item.cell.method_id for item in candidate_records}
    if (
        len(task_ids) != 1
        or {item.cell.method_id for item in scratch_records} != {"graphsage-scratch"}
        or len(candidate_methods) != 1
        or not candidate_methods <= {"multi-graph-shared-gfm", "domain-aligned-gfm"}
        or any(
            item.cell.protocol_hash != protocol.protocol_hash
            for item in (*scratch_records, *candidate_records)
        )
    ):
        raise ValueError("transfer comparison methods or protocol are invalid")
    if not all(item.promotable for item in (*scratch_records, *candidate_records)):
        raise ValueError("transfer comparison requires promotable raw records")
    ordered_keys = tuple(
        (budget, seed) for budget in protocol.label_budgets for seed in protocol.seeds
    )
    ordered_scratch = tuple(scratch[key] for key in ordered_keys)
    ordered_candidate = tuple(candidate[key] for key in ordered_keys)
    wins = 0
    for budget in protocol.label_budgets:
        budget_differences = tuple(
            _metric(candidate[(budget, seed)]) - _metric(scratch[(budget, seed)])
            for seed in protocol.seeds
        )
        if math.fsum(budget_differences) / len(budget_differences) > 0.0:
            wins += 1
    paired_differences = tuple(
        tuple(
            _metric(candidate[(budget, seed)]) - _metric(scratch[(budget, seed)])
            for budget in protocol.label_budgets
        )
        for seed in protocol.seeds
    )
    difference_tuple = tuple(value for seed_values in paired_differences for value in seed_values)
    improvement = math.fsum(difference_tuple) / len(difference_tuple)
    seed_material = canonical_sha256(
        {
            "scratch": [item.record_hash for item in ordered_scratch],
            "candidate": [item.record_hash for item in ordered_candidate],
        }
    )
    # Seed is the resampling unit: all four label budgets from one seed travel
    # together, preserving the paired cross-budget correlation by construction.
    seed_differences = tuple(
        math.fsum(seed_values) / len(seed_values) for seed_values in paired_differences
    )
    lower, upper = _bootstrap_interval(seed_differences, seed_material=seed_material)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-transfer-decision/1.0",
        "taskId": next(iter(task_ids)),
        "candidateMethodId": next(iter(candidate_methods)),
        "winningBudgetCount": wins,
        "aggregateImprovement": improvement,
        "ciLower": lower,
        "ciUpper": upper,
        "transferAdvantage": wins >= 3 and improvement >= 0.02 and lower > 0.0,
        "scratchRecordHashes": [item.record_hash for item in ordered_scratch],
        "candidateRecordHashes": [item.record_hash for item in ordered_candidate],
    }
    payload["decisionHash"] = canonical_sha256(payload)
    return TransferDecision.model_validate(payload)


__all__ = [
    "ExperimentAggregate",
    "ExperimentArtifactRef",
    "ExperimentCell",
    "ExperimentExecutionConfigEvidence",
    "ExperimentLedger",
    "ExperimentProtocol",
    "ExperimentRecipeEvidence",
    "ExperimentRunRecord",
    "ExperimentTaskSpec",
    "ExperimentTrainingDataEvidence",
    "GfmFamilyEvidence",
    "GraphManifestBinding",
    "MethodSpec",
    "PredictionEvidence",
    "ResourceTelemetryEvidence",
    "ResourceTelemetrySample",
    "TargetEvidence",
    "ThresholdSelectionEvidence",
    "TransferDecision",
    "aggregate_experiment",
    "build_experiment_matrix",
    "derive_transfer_advantage",
]
