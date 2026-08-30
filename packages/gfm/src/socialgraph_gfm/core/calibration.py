"""Validation-only temperature-plus-bias calibration for core scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.tensor_digest import canonical_tensor_digest

from .serving_registry import ScoreCalibration
from .model import CoreGFM
from .supervised import (
    EncodedGraphProvenance,
    HeadName,
    SupervisedTrainValidation,
    TaskKind,
    VerifiedEncodedGraph,
    VerifiedHeadTrainingReport,
    verify_head_training_report,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True, strict=True)


class BinaryScoreSemantics(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-binary-score-semantics/1.0"] = Field(
        alias="schemaVersion"
    )
    task_kind: Literal["node-binary", "edge-binary", "signed-edge"] = Field(alias="taskKind")
    head_name: HeadName = Field(alias="headName")
    entity_type: Literal["node", "directed-edge"] = Field(alias="entityType")
    transform: Literal["positive-minus-negative", "raw-logit"]
    positive_label: Literal[1] = Field(alias="positiveLabel")
    semantics_hash: str = Field(alias="semanticsHash", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def for_task(cls, task_kind: TaskKind) -> BinaryScoreSemantics:
        definitions: dict[str, tuple[str, str, str]] = {
            "node-binary": ("node_head", "node", "positive-minus-negative"),
            "edge-binary": ("binary_link_head", "directed-edge", "raw-logit"),
            "signed-edge": ("signed_edge_head", "directed-edge", "raw-logit"),
        }
        if task_kind not in definitions:
            raise ValueError(
                "binary sigmoid calibration is unavailable for multiclass or resilience regression"
            )
        head_name, entity_type, transform = definitions[task_kind]
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-binary-score-semantics/1.0",
            "taskKind": task_kind,
            "headName": head_name,
            "entityType": entity_type,
            "transform": transform,
            "positiveLabel": 1,
        }
        payload["semanticsHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_semantics(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"semantics_hash"})
        )
        if self.semantics_hash != expected:
            raise ValueError("score semantics hash does not match its definition")
        fixed = {
            "node-binary": ("node_head", "node", "positive-minus-negative"),
            "edge-binary": ("binary_link_head", "directed-edge", "raw-logit"),
            "signed-edge": ("signed_edge_head", "directed-edge", "raw-logit"),
        }[self.task_kind]
        if (self.head_name, self.entity_type, self.transform) != fixed:
            raise ValueError("score semantics do not match the fixed task formula")
        return self


class ValidationScoreBatch(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-validation-score-batch/1.0"] = Field(
        alias="schemaVersion"
    )
    role: Literal["validation"]
    task_kind: Literal["node-binary", "edge-binary", "signed-edge"] = Field(alias="taskKind")
    head_name: HeadName = Field(alias="headName")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=r"^[0-9a-f]{64}$")
    encoding_artifact_hash: str = Field(alias="encodingArtifactHash", pattern=r"^[0-9a-f]{64}$")
    head_training_report_hash: str = Field(
        alias="headTrainingReportHash", pattern=r"^[0-9a-f]{64}$"
    )
    validation_partition_hash: str = Field(
        alias="validationPartitionHash", pattern=r"^[0-9a-f]{64}$"
    )
    head_state_hash: str = Field(alias="headStateHash", pattern=r"^[0-9a-f]{64}$")
    score_semantics_hash: str = Field(alias="scoreSemanticsHash", pattern=r"^[0-9a-f]{64}$")
    logits_hash: str = Field(alias="logitsHash", pattern=r"^[0-9a-f]{64}$")
    targets_hash: str = Field(alias="targetsHash", pattern=r"^[0-9a-f]{64}$")
    count: int = Field(ge=2)
    promotion_eligible: bool = Field(alias="promotionEligible")
    batch_hash: str = Field(alias="batchHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_batch_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"batch_hash"})
        )
        if self.batch_hash != expected:
            raise ValueError("validation score batch hash does not match its evidence")
        return self


@dataclass(frozen=True, init=False)
class VerifiedValidationScores:
    logits: Tensor
    targets: Tensor
    record: ValidationScoreBatch
    model: CoreGFM
    encoded: VerifiedEncodedGraph
    data: SupervisedTrainValidation
    head_report: VerifiedHeadTrainingReport
    semantics: BinaryScoreSemantics

    @property
    def provenance(self) -> EncodedGraphProvenance:
        return self.encoded.provenance

    def verify(self) -> None:
        logits, targets = _validate_tensor_pair(self.logits, self.targets)
        expected_logits, expected_targets = _validation_score_tensors(
            self.model,
            self.encoded,
            self.data,
            self.head_report,
            semantics=self.semantics,
        )
        expected_record = _validation_score_batch(
            expected_logits,
            expected_targets,
            encoded=self.encoded,
            data=self.data,
            head_report=self.head_report,
            semantics=self.semantics,
        )
        if (
            type(self.record) is not ValidationScoreBatch
            or not torch.equal(logits, expected_logits)
            or not torch.equal(targets, expected_targets)
            or self.record != expected_record
        ):
            raise ValueError("validation score binding changed")


def _new_verified_validation_scores(
    *,
    logits: Tensor,
    targets: Tensor,
    record: ValidationScoreBatch,
    model: CoreGFM,
    encoded: VerifiedEncodedGraph,
    data: SupervisedTrainValidation,
    head_report: VerifiedHeadTrainingReport,
    semantics: BinaryScoreSemantics,
) -> VerifiedValidationScores:
    scores = object.__new__(VerifiedValidationScores)
    for name, value in (
        ("logits", logits),
        ("targets", targets),
        ("record", record),
        ("model", model),
        ("encoded", encoded),
        ("data", data),
        ("head_report", head_report),
        ("semantics", semantics),
    ):
        object.__setattr__(scores, name, value)
    return scores


def _verify_exact_scores(scores: VerifiedValidationScores) -> None:
    if type(scores) is not VerifiedValidationScores:
        raise TypeError("calibration requires exact VerifiedValidationScores evidence")
    VerifiedValidationScores.verify(scores)


class CalibrationProtocol(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-calibration-protocol/1.0"] = Field(
        alias="schemaVersion"
    )
    validation_score_batch_hash: str = Field(
        alias="validationScoreBatchHash", pattern=r"^[0-9a-f]{64}$"
    )
    head_training_report_hash: str = Field(
        alias="headTrainingReportHash", pattern=r"^[0-9a-f]{64}$"
    )
    validation_partition_hash: str = Field(
        alias="validationPartitionHash", pattern=r"^[0-9a-f]{64}$"
    )
    head_state_hash: str = Field(alias="headStateHash", pattern=r"^[0-9a-f]{64}$")
    score_semantics_hash: str = Field(alias="scoreSemanticsHash", pattern=r"^[0-9a-f]{64}$")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=r"^[0-9a-f]{64}$")
    model_identity_hash: str = Field(alias="modelIdentityHash", pattern=r"^[0-9a-f]{64}$")
    encoding_artifact_hash: str = Field(alias="encodingArtifactHash", pattern=r"^[0-9a-f]{64}$")
    calibration_role: Literal["validation"] = Field(alias="calibrationRole")
    promotion_eligible: bool = Field(alias="promotionEligible")
    bin_count: int = Field(alias="binCount", ge=2, le=100)
    max_iterations: int = Field(alias="maxIterations", ge=1, le=1000)
    min_temperature: float = Field(alias="minTemperature", gt=0.0)
    max_temperature: float = Field(alias="maxTemperature", gt=0.0)
    max_bias: float = Field(alias="maxBias", gt=0.0)
    protocol_hash: str = Field(alias="protocolHash", pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def fixed(cls, scores: VerifiedValidationScores) -> CalibrationProtocol:
        _verify_exact_scores(scores)
        record = scores.record
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-calibration-protocol/1.0",
            "validationScoreBatchHash": record.batch_hash,
            "headTrainingReportHash": record.head_training_report_hash,
            "validationPartitionHash": record.validation_partition_hash,
            "headStateHash": record.head_state_hash,
            "scoreSemanticsHash": record.score_semantics_hash,
            "graphVersionHash": record.graph_version_hash,
            "modelIdentityHash": record.model_identity_hash,
            "encodingArtifactHash": record.encoding_artifact_hash,
            "calibrationRole": "validation",
            "promotionEligible": record.promotion_eligible,
            "binCount": 10,
            "maxIterations": 100,
            "minTemperature": 0.05,
            "maxTemperature": 20.0,
            "maxBias": 20.0,
        }
        payload["protocolHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @model_validator(mode="after")
    def validate_protocol(self):
        if self.max_temperature <= self.min_temperature:
            raise ValueError("calibration temperature bounds are invalid")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"protocol_hash"})
        )
        if self.protocol_hash != expected:
            raise ValueError("protocolHash does not match calibration protocol")
        return self


class CalibrationFitReport(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-calibration-fit/1.0"] = Field(
        alias="schemaVersion"
    )
    calibration: ScoreCalibration
    validation_score_batch_hash: str = Field(
        alias="validationScoreBatchHash", pattern=r"^[0-9a-f]{64}$"
    )
    head_training_report_hash: str = Field(
        alias="headTrainingReportHash", pattern=r"^[0-9a-f]{64}$"
    )
    validation_partition_hash: str = Field(
        alias="validationPartitionHash", pattern=r"^[0-9a-f]{64}$"
    )
    head_state_hash: str = Field(alias="headStateHash", pattern=r"^[0-9a-f]{64}$")
    score_semantics_hash: str = Field(alias="scoreSemanticsHash", pattern=r"^[0-9a-f]{64}$")
    validation_logits_hash: str = Field(alias="validationLogitsHash", pattern=r"^[0-9a-f]{64}$")
    validation_targets_hash: str = Field(alias="validationTargetsHash", pattern=r"^[0-9a-f]{64}$")
    before_nll: float = Field(alias="beforeNll", ge=0.0)
    after_nll: float = Field(alias="afterNll", ge=0.0)
    before_ece: float = Field(alias="beforeEce", ge=0.0, le=1.0)
    after_ece: float = Field(alias="afterEce", ge=0.0, le=1.0)
    before_brier: float = Field(alias="beforeBrier", ge=0.0, le=1.0)
    after_brier: float = Field(alias="afterBrier", ge=0.0, le=1.0)
    promotion_eligible: bool = Field(alias="promotionEligible")
    report_hash: str = Field(alias="reportHash", pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_report(self):
        if not all(
            math.isfinite(value)
            for value in (
                self.before_nll,
                self.after_nll,
                self.before_ece,
                self.after_ece,
                self.before_brier,
                self.after_brier,
            )
        ):
            raise ValueError("calibration fit metrics must be finite")
        if self.after_nll > self.before_nll + 1e-10:
            raise ValueError("calibration must not worsen validation NLL")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"report_hash"})
        )
        if self.report_hash != expected:
            raise ValueError("reportHash does not match calibration fit evidence")
        return self


def _validate_tensor_pair(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    if logits.ndim != 1 or targets.ndim != 1 or logits.shape != targets.shape:
        raise ValueError("calibration logits and targets must be aligned rank-one tensors")
    if logits.numel() < 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError("calibration requires at least two finite logits")
    if not bool(torch.isfinite(targets).all()) or not bool(
        torch.all((targets == 0) | (targets == 1))
    ):
        raise ValueError("calibration targets must be finite binary values")
    if torch.unique(targets).numel() != 2:
        raise ValueError("calibration validation targets must contain both classes")
    return logits.detach().to(dtype=torch.float64, device="cpu"), targets.detach().to(
        dtype=torch.float64, device="cpu"
    )


def _tensor_hashes(logits: Tensor, targets: Tensor) -> tuple[str, str]:
    logits_hash = canonical_sha256(canonical_tensor_digest(logits))
    targets_hash = canonical_sha256(canonical_tensor_digest(targets))
    return logits_hash, targets_hash


def _validation_score_batch(
    logits: Tensor,
    targets: Tensor,
    *,
    encoded: VerifiedEncodedGraph,
    data: SupervisedTrainValidation,
    head_report: VerifiedHeadTrainingReport,
    semantics: BinaryScoreSemantics,
) -> ValidationScoreBatch:
    logits_hash, targets_hash = _tensor_hashes(logits, targets)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-validation-score-batch/1.0",
        "role": "validation",
        "taskKind": data.task_kind,
        "headName": data.head_name,
        "graphVersionHash": encoded.provenance.graph_version_hash,
        "modelIdentityHash": encoded.provenance.model_identity_hash,
        "encodingArtifactHash": encoded.provenance.artifact_hash,
        "headTrainingReportHash": head_report.report_hash,
        "validationPartitionHash": data.validation.partition_hash,
        "headStateHash": head_report.head_state_hash,
        "scoreSemanticsHash": semantics.semantics_hash,
        "logitsHash": logits_hash,
        "targetsHash": targets_hash,
        "count": logits.numel(),
        "promotionEligible": head_report.promotion_eligible,
    }
    payload["batchHash"] = canonical_sha256(payload)
    return ValidationScoreBatch.model_validate(payload)


def _validation_score_tensors(
    model: CoreGFM,
    encoded: VerifiedEncodedGraph,
    data: SupervisedTrainValidation,
    head_report: VerifiedHeadTrainingReport,
    *,
    semantics: BinaryScoreSemantics,
) -> tuple[Tensor, Tensor]:
    if type(encoded) is not VerifiedEncodedGraph:
        raise TypeError("validation scoring requires exact VerifiedEncodedGraph evidence")
    if type(semantics) is not BinaryScoreSemantics:
        raise TypeError("validation scoring requires exact binary score semantics")
    verify_head_training_report(model, encoded, data, head_report)
    fixed_semantics = BinaryScoreSemantics.for_task(data.task_kind)
    if semantics != fixed_semantics:
        raise ValueError("score semantics do not match the fixed task formula")
    device = next(model.parameters()).device
    with torch.no_grad():
        if data.task_kind == "node-binary":
            locator = torch.tensor(
                data.validation.node_indices,
                dtype=torch.long,
                device=device,
            )
            class_logits = model.node_head(encoded.tensor.to(device)[locator])
            if class_logits.ndim != 2 or class_logits.shape[1] != 2:
                raise ValueError("node-binary calibration requires exactly two class logits")
            logits = class_logits[:, 1] - class_logits[:, 0]
        elif data.task_kind in {"edge-binary", "signed-edge"}:
            pairs = torch.tensor(
                data.validation.edge_pairs,
                dtype=torch.long,
                device=device,
            )
            head = getattr(model, data.head_name)
            logits = head(encoded.tensor.to(device), pairs)
        else:
            raise ValueError(
                "binary sigmoid calibration is unavailable for multiclass or resilience regression"
            )
    targets = torch.tensor(data.validation.targets, dtype=torch.float64)
    return _validate_tensor_pair(logits, targets)


def derive_validation_scores(
    model: CoreGFM,
    encoded: VerifiedEncodedGraph,
    data: SupervisedTrainValidation,
    head_report: VerifiedHeadTrainingReport,
    *,
    semantics: BinaryScoreSemantics,
) -> VerifiedValidationScores:
    """Derive calibration scores from the restored best head and validation role only."""

    normalized_logits, normalized_targets = _validation_score_tensors(
        model,
        encoded,
        data,
        head_report,
        semantics=semantics,
    )
    scores = _new_verified_validation_scores(
        logits=normalized_logits,
        targets=normalized_targets,
        record=_validation_score_batch(
            normalized_logits,
            normalized_targets,
            encoded=encoded,
            data=data,
            head_report=head_report,
            semantics=semantics,
        ),
        model=model,
        encoded=encoded,
        data=data,
        head_report=head_report,
        semantics=semantics,
    )
    VerifiedValidationScores.verify(scores)
    return scores


def _metrics(logits: Tensor, targets: Tensor, *, bins: int) -> tuple[float, float, float]:
    nll = float(functional.binary_cross_entropy_with_logits(logits, targets))
    probabilities = torch.sigmoid(logits)
    brier = float(torch.mean((probabilities - targets) ** 2))
    ece = 0.0
    boundaries = torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)
    for index in range(bins):
        selected = (probabilities >= boundaries[index]) & (
            probabilities < boundaries[index + 1]
            if index + 1 < bins
            else probabilities <= boundaries[index + 1]
        )
        count = int(selected.sum())
        if count:
            confidence = float(probabilities[selected].mean())
            frequency = float(targets[selected].mean())
            ece += count / targets.numel() * abs(confidence - frequency)
    return nll, ece, brier


def _artifact(
    *, temperature: float, bias: float, protocol: CalibrationProtocol
) -> ScoreCalibration:
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-score-calibration/2.0",
        "calibrationVersion": f"core-temperature-bias-v1-{protocol.protocol_hash[:16]}",
        "method": "sigmoid",
        "temperature": temperature,
        "bias": bias,
        "protocolHash": protocol.protocol_hash,
    }
    payload["artifactHash"] = canonical_sha256(payload)
    return ScoreCalibration.model_validate(payload)


def fit_score_calibration_report(
    scores: VerifiedValidationScores,
    *,
    protocol: CalibrationProtocol,
) -> CalibrationFitReport:
    """Fit bounded temperature and bias using validation data only."""

    _verify_exact_scores(scores)
    record = scores.record
    bindings = {
        "validation_score_batch_hash": record.batch_hash,
        "head_training_report_hash": record.head_training_report_hash,
        "validation_partition_hash": record.validation_partition_hash,
        "head_state_hash": record.head_state_hash,
        "score_semantics_hash": record.score_semantics_hash,
        "graph_version_hash": record.graph_version_hash,
        "model_identity_hash": record.model_identity_hash,
        "encoding_artifact_hash": record.encoding_artifact_hash,
        "promotion_eligible": record.promotion_eligible,
    }
    if any(getattr(protocol, name) != value for name, value in bindings.items()):
        raise ValueError("calibration protocol validation score binding does not match inputs")
    logits, targets = _validate_tensor_pair(scores.logits, scores.targets)
    logits_hash, targets_hash = _tensor_hashes(logits, targets)
    before = _metrics(logits, targets, bins=protocol.bin_count)
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    bias_parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS(
        (log_temperature, bias_parameter),
        lr=0.5,
        max_iter=protocol.max_iterations,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )
    minimum_log = math.log(protocol.min_temperature)
    maximum_log = math.log(protocol.max_temperature)

    def closure() -> Tensor:
        optimizer.zero_grad(set_to_none=True)
        temperature = torch.exp(log_temperature.clamp(minimum_log, maximum_log))
        bias = bias_parameter.clamp(-protocol.max_bias, protocol.max_bias)
        loss = functional.binary_cross_entropy_with_logits((logits + bias) / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(torch.exp(log_temperature.detach().clamp(minimum_log, maximum_log)))
    bias = float(bias_parameter.detach().clamp(-protocol.max_bias, protocol.max_bias))
    calibrated_logits = (logits + bias) / temperature
    after = _metrics(calibrated_logits, targets, bins=protocol.bin_count)
    if after[0] > before[0] + 1e-12:
        temperature, bias, after = 1.0, 0.0, before
    calibration = _artifact(temperature=temperature, bias=bias, protocol=protocol)
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-calibration-fit/1.0",
        "calibration": calibration.model_dump(mode="python", by_alias=True),
        "validationScoreBatchHash": record.batch_hash,
        "headTrainingReportHash": record.head_training_report_hash,
        "validationPartitionHash": record.validation_partition_hash,
        "headStateHash": record.head_state_hash,
        "scoreSemanticsHash": record.score_semantics_hash,
        "validationLogitsHash": logits_hash,
        "validationTargetsHash": targets_hash,
        "beforeNll": before[0],
        "afterNll": after[0],
        "beforeEce": before[1],
        "afterEce": after[1],
        "beforeBrier": before[2],
        "afterBrier": after[2],
        "promotionEligible": record.promotion_eligible,
    }
    payload["reportHash"] = canonical_sha256(payload)
    return CalibrationFitReport.model_validate(payload)


def fit_score_calibration(
    scores: VerifiedValidationScores,
    *,
    protocol: CalibrationProtocol,
) -> ScoreCalibration:
    return fit_score_calibration_report(scores, protocol=protocol).calibration


__all__ = [
    "BinaryScoreSemantics",
    "CalibrationFitReport",
    "CalibrationProtocol",
    "ValidationScoreBatch",
    "VerifiedValidationScores",
    "derive_validation_scores",
    "fit_score_calibration",
    "fit_score_calibration_report",
]
