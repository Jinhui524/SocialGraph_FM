"""Deterministic target-review adaptation over frozen Governance embeddings.

This module deliberately has no training code.  It consumes a completed run's
frozen arrays, selects a scalar policy, and emits only bounded metadata and
rank comparisons.  Embeddings and centroids remain inside the GFM process.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..canonical import canonical_sha256

if TYPE_CHECKING:
    from .target_tasks import TargetLabelSetV2

LEGACY_LABEL_SET_SCHEMA_VERSION = "socialgraph-fm.governance-target-label-set/1.0"
LABEL_SET_SCHEMA_VERSION = "socialgraph-fm.governance-target-label-set/1.1"
LABEL_RECIPE_SCHEMA_VERSION = "socialgraph-fm.governance-target-label-recipe/1.1"
TARGET_PACKAGE_RECEIPT_SCHEMA_VERSION = "socialgraph-fm.governance-target-package-receipt/1.1"
POLICY_SCHEMA_VERSION = "socialgraph-fm.governance-target-review-policy/1.0"
COMPARISON_SCHEMA_VERSION = "socialgraph-fm.governance-adaptation-comparison/1.0"
HASH_PATTERN = r"^[0-9a-f]{64}$"
RUN_PATTERN = r"^governance-[0-9a-f]{32}$"
ARTIFACT_PATTERN = r"^governance-artifact-[0-9a-f]{32}$"
LAMBDA_CANDIDATES = (0.0, 0.25, 0.5, 1.0)
EPSILON = 1e-8
ADAPTATION_CODE_HASH = canonical_sha256(
    {
        "recipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "lambdaCandidates": list(LAMBDA_CANDIDATES),
        "normalizationEpsilon": EPSILON,
        "embeddingDimension": 256,
        "baseComparisonSource": "persisted-scores-and-ranks",
        "maxEligibleLabels": 256,
    }
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
    )


class LabelSelectionRecipe(_FrozenModel):
    version: Literal["graph-fused-degree-quartile-stable-hash-v2"]
    stratification: Literal["graph-fused-degree-rank-quartile"]
    structural_strata: Literal[4] = Field(alias="structuralStrata")
    labels_per_class: Literal[8] = Field(alias="labelsPerClass")
    labels_per_class_per_stratum: Literal[2] = Field(alias="labelsPerClassPerStratum")
    score_inputs: tuple[()] = Field(alias="scoreInputs")


class TargetPackageReceipt(_FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-package-receipt/1.1"] = Field(alias="schemaVersion")
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=200)
    source_schema_version: Literal["socialgraph-fm.anonymized-posts/1.0"] = Field(alias="sourceSchemaVersion")
    source_sha256: str = Field(alias="sourceSha256", pattern=HASH_PATTERN)
    authorization_reference: str = Field(alias="authorizationReference", min_length=1, max_length=300)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=HASH_PATTERN)
    labels_sha256: str = Field(alias="labelsSha256", pattern=HASH_PATTERN)
    encoder: Mapping[str, object]
    selection_recipe: Mapping[str, object] = Field(alias="selectionRecipe")
    label_selection_recipe: LabelSelectionRecipe = Field(alias="labelSelectionRecipe")
    coverage: Mapping[str, object]
    receipt_hash: str = Field(alias="receiptHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self) -> TargetPackageReceipt:
        if self.encoder != {
            "modelId": self.encoder.get("modelId"),
            "revision": self.encoder.get("revision"),
            "cacheSha256": self.encoder.get("cacheSha256"),
            "compatibility": "dimension-only-unverified",
            "dimension": 768,
        } or not isinstance(self.encoder.get("modelId"), str) or not isinstance(
            self.encoder.get("revision"), str
        ) or not isinstance(self.encoder.get("cacheSha256"), str):
            raise ValueError("target receipt encoder provenance is invalid")
        if not __import__("re").fullmatch(HASH_PATTERN, str(self.encoder["cacheSha256"])):
            raise ValueError("target receipt encoder digest is invalid")
        if self.selection_recipe != {
            "version": "connected-structural-hash-v2",
            "nodeCount": 128,
            "requiredIo": 16,
            "requiredControls": 64,
            "minimumNonemptyModalities": 4,
            "scoreInputs": [],
            "groupRelations": {"maxGroupAccounts": 256, "totalPotentialPairBudget": 50_000},
            "fastRT": {"windowSeconds": 10, "pairBudget": 50_000, "algorithm": "sorted-sliding-window-v1"},
            "tweetSim": {"mutualTopK": 5, "cosineThreshold": 0.8, "pairBudget": 10_000},
        }:
            raise ValueError("target receipt selection recipe is invalid")
        modalities = self.coverage.get("nonemptyModalities")
        if (
            self.coverage.get("nodeCount") != 128
            or not isinstance(self.coverage.get("ioCount"), int)
            or not isinstance(self.coverage.get("controlCount"), int)
            or cast(int, self.coverage["ioCount"]) + cast(int, self.coverage["controlCount"]) != 128
            or cast(int, self.coverage["ioCount"]) < 16
            or cast(int, self.coverage["controlCount"]) < 64
            or self.coverage.get("connected") is not True
            or not isinstance(modalities, list)
            or len(modalities) < 4
            or len(set(modalities)) != len(modalities)
        ):
            raise ValueError("target receipt coverage is invalid")
        logical = self.model_dump(mode="json", by_alias=True, exclude={"receipt_hash"})
        if self.receipt_hash != canonical_sha256(logical):
            raise ValueError("receiptHash mismatch")
        return self


def target_label_file_bytes(
    receipt: TargetPackageReceipt, rows: Sequence[Mapping[str, object]]
) -> bytes:
    document = {
        "schemaVersion": LABEL_RECIPE_SCHEMA_VERSION,
        "datasetId": receipt.dataset_id,
        "bundleSha256": receipt.bundle_sha256,
        "selectionRecipe": receipt.label_selection_recipe.model_dump(
            mode="json", by_alias=True
        ),
        "labels": sorted((dict(row) for row in rows), key=lambda row: str(row["nodeId"])),
    }
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"


class AdaptationBinding(_FrozenModel):
    artifact_id: str = Field(alias="artifactId", pattern=ARTIFACT_PATTERN)
    dataset_content_hash: str = Field(alias="datasetContentHash", pattern=HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    run_id: str = Field(alias="runId", pattern=RUN_PATTERN)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)
    run_artifact_hash: str = Field(alias="runArtifactHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=200)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH_PATTERN)
    recipe_hash: str = Field(alias="recipeHash", pattern=HASH_PATTERN)
    code_hash: str = Field(alias="codeHash", pattern=HASH_PATTERN)
    seed: int = Field(ge=0, le=2**63 - 1)


class LabelEvidence(_FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    label: Literal["positive", "negative", "pending"]
    source_type: Literal["concluded_review", "imported_sidecar"] = Field(
        alias="sourceType"
    )
    source_record_id: str = Field(alias="sourceRecordId", min_length=1, max_length=200)
    source_record_hash: str = Field(alias="sourceRecordHash", pattern=HASH_PATTERN)
    review_event_hash: str | None = Field(
        default=None, alias="reviewEventHash", pattern=HASH_PATTERN
    )
    binding: AdaptationBinding
    structural_stratum: int | None = Field(default=None, alias="structuralStratum", ge=0, le=3)
    fused_degree: int | None = Field(default=None, alias="fusedDegree", ge=0, le=9_999)
    labels_sha256: str | None = Field(default=None, alias="labelsSha256", pattern=HASH_PATTERN)
    receipt_hash: str | None = Field(default=None, alias="receiptHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_source(self) -> LabelEvidence:
        if (self.source_type == "concluded_review") != (
            self.review_event_hash is not None
        ):
            raise ValueError("concluded reviews require a reviewEventHash")
        sidecar_values = (
            self.structural_stratum,
            self.fused_degree,
            self.labels_sha256,
            self.receipt_hash,
        )
        if self.source_type == "imported_sidecar" and any(
            value is not None for value in sidecar_values
        ) != all(value is not None for value in sidecar_values):
            raise ValueError("imported sidecar provenance cannot be partial")
        return self


class LabelSourceRecord(_FrozenModel):
    source_type: Literal["concluded_review", "imported_sidecar"] = Field(
        alias="sourceType"
    )
    source_record_id: str = Field(alias="sourceRecordId")
    source_record_hash: str = Field(alias="sourceRecordHash", pattern=HASH_PATTERN)
    review_event_hash: str | None = Field(
        default=None, alias="reviewEventHash", pattern=HASH_PATTERN
    )


class TargetLabelSet(_FrozenModel):
    schema_version: Literal[
        "socialgraph-fm.governance-target-label-set/1.0",
        "socialgraph-fm.governance-target-label-set/1.1",
    ] = Field(
        alias="schemaVersion"
    )
    binding: AdaptationBinding
    sidecar_receipt: TargetPackageReceipt | None = Field(
        default=None, alias="sidecarReceipt"
    )
    source_records: tuple[LabelSourceRecord, ...] = Field(
        alias="sourceRecords", max_length=256
    )
    review_event_hashes: tuple[str, ...] = Field(
        alias="reviewEventHashes", max_length=256
    )
    labels: tuple[LabelEvidence, ...] = Field(max_length=256)
    conflicts: tuple[str, ...]
    positive_count: int = Field(alias="positiveCount", ge=4)
    negative_count: int = Field(alias="negativeCount", ge=4)
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_inventory_and_hash(self) -> TargetLabelSet:
        if self.conflicts:
            raise ValueError("label set contains conflicting source records")
        if any(label.label == "pending" for label in self.labels):
            raise ValueError("pending reviews are excluded from target labels")
        if len(self.labels) < 8:
            raise ValueError("label set requires at least eight eligible labels")
        if len({label.node_id for label in self.labels}) != len(self.labels):
            raise ValueError("label set contains a duplicate node")
        if any(label.binding != self.binding for label in self.labels):
            raise ValueError("label binding does not match the bound run")
        positive = sum(label.label == "positive" for label in self.labels)
        negative = sum(label.label == "negative" for label in self.labels)
        if positive == 0 or negative == 0:
            raise ValueError("label set requires both classes")
        if min(positive, negative) < 4:
            raise ValueError("label set requires at least four labels per class")
        if (positive, negative) != (self.positive_count, self.negative_count):
            raise ValueError("label class counts are inconsistent")
        expected_sources = tuple(
            LabelSourceRecord(
                sourceType=label.source_type,
                sourceRecordId=label.source_record_id,
                sourceRecordHash=label.source_record_hash,
                reviewEventHash=label.review_event_hash,
            )
            for label in self.labels
        )
        if self.source_records != expected_sources:
            raise ValueError("source record inventory is inconsistent")
        expected_review_hashes = tuple(
            label.review_event_hash
            for label in self.labels
            if label.review_event_hash is not None
        )
        if self.review_event_hashes != expected_review_hashes:
            raise ValueError("review-event hash inventory is inconsistent")
        imported = tuple(
            label for label in self.labels if label.source_type == "imported_sidecar"
        )
        if imported:
            if self.schema_version == LEGACY_LABEL_SET_SCHEMA_VERSION:
                if self.sidecar_receipt is not None or any(
                    label.receipt_hash is not None for label in imported
                ):
                    raise ValueError("legacy sidecars cannot claim 1.1 provenance")
            elif (
                self.sidecar_receipt is None
                or len(imported) != 16
                or len(imported) != len(self.labels)
            ):
                raise ValueError("imported sidecars require the 1.1 receipt-bound contract")
            if self.schema_version == LEGACY_LABEL_SET_SCHEMA_VERSION:
                imported = ()
            rows = [
                {
                    "nodeId": label.node_id,
                    "label": "io" if label.label == "positive" else "control",
                    "structuralStratum": label.structural_stratum,
                    "fusedDegree": label.fused_degree,
                }
                for label in imported
            ]
            if imported:
                receipt = self.sidecar_receipt
                if receipt is None:
                    raise ValueError("imported sidecar receipt is missing")
                if hashlib.sha256(
                    target_label_file_bytes(receipt, rows)
                ).hexdigest() != receipt.labels_sha256:
                    raise ValueError("labelsSha256 mismatch")
                for cohort in ("io", "control"):
                    for stratum in range(4):
                        if sum(
                            row["label"] == cohort
                            and row["structuralStratum"] == stratum
                            for row in rows
                        ) != 2:
                            raise ValueError("structural stratum label quota mismatch")
                for label, row in zip(imported, rows, strict=True):
                    if (
                        label.labels_sha256 != receipt.labels_sha256
                        or label.receipt_hash != receipt.receipt_hash
                        or label.source_record_hash
                        != canonical_sha256(
                            {
                                "schemaVersion": LABEL_RECIPE_SCHEMA_VERSION,
                                "datasetId": receipt.dataset_id,
                                "bundleSha256": receipt.bundle_sha256,
                                "labelsSha256": receipt.labels_sha256,
                                "receiptHash": receipt.receipt_hash,
                                **row,
                            }
                        )
                    ):
                        raise ValueError("imported sidecar source hash mismatch")
        elif self.schema_version != LEGACY_LABEL_SET_SCHEMA_VERSION or self.sidecar_receipt:
            raise ValueError("legacy concluded-review labels cannot claim sidecar provenance")
        logical = self.model_dump(
            mode="json", by_alias=True, exclude={"label_set_hash"}
        )
        if logical.get("sidecarReceipt") is None:
            logical.pop("sidecarReceipt", None)
        for label_document in cast(list[dict[str, Any]], logical["labels"]):
            for field in (
                "structuralStratum",
                "fusedDegree",
                "labelsSha256",
                "receiptHash",
            ):
                if label_document.get(field) is None:
                    label_document.pop(field, None)
        expected_hash = canonical_sha256(logical)
        if self.label_set_hash != expected_hash:
            raise ValueError("labelSetHash mismatch")
        return self


class TargetReviewPolicy(_FrozenModel):
    schema_version: Literal["socialgraph-fm.governance-target-review-policy/1.0"] = Field(
        alias="schemaVersion"
    )
    binding: AdaptationBinding
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)
    status: Literal["collecting_reviews", "ready", "insufficient_signal", "invalid"]
    selected_lambda: float = Field(alias="selectedLambda")
    lambda_candidates: tuple[float, ...] = Field(alias="lambdaCandidates")
    validation_losses: dict[str, float] = Field(alias="validationLosses")
    eligible_label_count: int = Field(alias="eligibleLabelCount", ge=8)
    positive_count: int = Field(alias="positiveCount", ge=4)
    negative_count: int = Field(alias="negativeCount", ge=4)
    embedding_dimension: Literal[256] = Field(alias="embeddingDimension")
    positive_centroid_hash: str = Field(alias="positiveCentroidHash", pattern=HASH_PATTERN)
    negative_centroid_hash: str = Field(alias="negativeCentroidHash", pattern=HASH_PATTERN)
    normalization_epsilon: float = Field(alias="normalizationEpsilon", gt=0)
    fitting_recipe: Literal[
        "l2-centroids+run-zscore+loo-balanced-log-loss-v1"
    ] = Field(alias="fittingRecipe")
    ready_policy_hash: str | None = Field(
        default=None, alias="readyPolicyHash", pattern=HASH_PATTERN
    )
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_policy(self) -> TargetReviewPolicy:
        if self.lambda_candidates != LAMBDA_CANDIDATES:
            raise ValueError("lambda candidate inventory is invalid")
        if self.normalization_epsilon != EPSILON:
            raise ValueError("normalization epsilon is invalid")
        if self.selected_lambda not in LAMBDA_CANDIDATES:
            raise ValueError("selected lambda is invalid")
        if tuple(self.validation_losses) != tuple(f"{value:g}" for value in LAMBDA_CANDIDATES):
            raise ValueError("validation loss inventory is invalid")
        if (self.status == "ready") != (self.selected_lambda != 0.0):
            raise ValueError("policy readiness disagrees with the selected lambda")
        if (self.status == "ready") != (self.ready_policy_hash is not None):
            raise ValueError("ready policy publication state is inconsistent")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"policy_hash"})
        )
        if self.policy_hash != expected:
            raise ValueError("policyHash mismatch")
        return self


class AdaptationComparisonRow(_FrozenModel):
    node_id: str = Field(alias="nodeId", min_length=1, max_length=128)
    base_score: float = Field(alias="baseScore", ge=0, le=1)
    base_rank: int = Field(alias="baseRank", ge=1)
    adapted_review_priority: float = Field(alias="adaptedReviewPriority", ge=0, le=1)
    adapted_rank: int = Field(alias="adaptedRank", ge=1)
    rank_delta: int = Field(alias="rankDelta")

    @model_validator(mode="after")
    def validate_delta(self) -> AdaptationComparisonRow:
        if self.rank_delta != self.adapted_rank - self.base_rank:
            raise ValueError("rankDelta is inconsistent")
        return self


class AdaptationComparison(_FrozenModel):
    schema_version: Literal[
        "socialgraph-fm.governance-adaptation-comparison/1.0"
    ] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    total: int = Field(ge=1)
    rows: tuple[AdaptationComparisonRow, ...]
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_comparison(self) -> AdaptationComparison:
        if self.total != len(self.rows):
            raise ValueError("comparison row inventory is incomplete")
        if len({row.node_id for row in self.rows}) != self.total:
            raise ValueError("comparison contains duplicate nodes")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"comparison_hash"})
        )
        if self.comparison_hash != expected:
            raise ValueError("comparisonHash mismatch")
        return self


class TargetReviewPolicyV2(_FrozenModel):
    """Generic low-resource policy contract without target-network constants."""

    schema_version: Literal["socialgraph-fm.governance-target-review-policy/2.0"] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    label_set_hash: str = Field(alias="labelSetHash", pattern=HASH_PATTERN)
    status: Literal["collecting_reviews", "ready", "insufficient_signal", "invalid"]
    selected_lambda: float = Field(alias="selectedLambda")
    eligible_label_count: int = Field(alias="eligibleLabelCount", ge=8, le=256)
    positive_count: int = Field(alias="positiveCount", ge=4)
    negative_count: int = Field(alias="negativeCount", ge=4)
    fitting_recipe: Literal[
        "l2-centroids+run-zscore+loo-balanced-log-loss-v1"
    ] = Field(alias="fittingRecipe")
    base_outputs_immutable: Literal[True] = Field(alias="baseOutputsImmutable")
    adapted_output_fields: tuple[
        Literal["adaptedReviewPriority"], Literal["adaptedRank"]
    ] = Field(alias="adaptedOutputFields")
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_policy(self) -> TargetReviewPolicyV2:
        if self.positive_count + self.negative_count != self.eligible_label_count:
            raise ValueError("policy label counts are inconsistent")
        if self.selected_lambda not in LAMBDA_CANDIDATES:
            raise ValueError("selected lambda is invalid")
        expected_status = (
            "insufficient_signal" if self.selected_lambda == 0.0 else "ready"
        )
        if self.status != expected_status:
            raise ValueError("policy readiness disagrees with the selected lambda")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"policy_hash"})
        )
        if self.policy_hash != expected:
            raise ValueError("policyHash mismatch")
        return self


class AdaptationComparisonV2(_FrozenModel):
    """Additive comparison contract that preserves every Global base output."""

    schema_version: Literal["socialgraph-fm.governance-adaptation-comparison/2.0"] = Field(alias="schemaVersion")
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    total: int = Field(ge=1)
    base_outputs_immutable: Literal[True] = Field(alias="baseOutputsImmutable")
    rows: tuple[AdaptationComparisonRow, ...]
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_comparison(self) -> AdaptationComparisonV2:
        if self.total != len(self.rows):
            raise ValueError("comparison row inventory is incomplete")
        if len({row.node_id for row in self.rows}) != self.total:
            raise ValueError("comparison contains duplicate nodes")
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"comparison_hash"})
        )
        if self.comparison_hash != expected:
            raise ValueError("comparisonHash mismatch")
        return self


class AdaptationGovernanceHandoff(_FrozenModel):
    """Hash-bound governance boundary for an adaptation candidate."""

    schema_version: Literal["socialgraph-fm.governance-adaptation-handoff/1.0"] = Field(
        alias="schemaVersion"
    )
    binding: AdaptationBinding
    policy_hash: str = Field(alias="policyHash", pattern=HASH_PATTERN)
    comparison_hash: str = Field(alias="comparisonHash", pattern=HASH_PATTERN)
    decision: Literal[
        "pending_governance_review", "approved", "rejected", "superseded"
    ]
    base_model_mutation: Literal[False] = Field(alias="baseModelMutation")
    handoff_hash: str = Field(alias="handoffHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_handoff(self) -> AdaptationGovernanceHandoff:
        expected = canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"handoff_hash"})
        )
        if self.handoff_hash != expected:
            raise ValueError("handoffHash mismatch")
        return self


@dataclass(frozen=True)
class FittedReviewPolicy:
    policy: TargetReviewPolicy
    comparison: AdaptationComparison
    positive_centroid: np.ndarray
    negative_centroid: np.ndarray


@dataclass(frozen=True)
class FittedReviewPolicyV2:
    policy: TargetReviewPolicyV2
    comparison: AdaptationComparisonV2
    positive_centroid: np.ndarray
    negative_centroid: np.ndarray


def build_target_label_set(
    binding: AdaptationBinding,
    labels: Sequence[LabelEvidence],
    *,
    schema_version: str = LEGACY_LABEL_SET_SCHEMA_VERSION,
    sidecar_receipt: TargetPackageReceipt | None = None,
) -> TargetLabelSet:
    ordered = tuple(sorted(labels, key=lambda item: (item.node_id, item.source_record_id)))
    by_node: dict[str, LabelEvidence] = {}
    for label in ordered:
        previous = by_node.get(label.node_id)
        if previous is not None:
            if previous.label != label.label:
                raise ValueError(f"conflicting sources for node {label.node_id}")
            raise ValueError(f"duplicate node label for {label.node_id}")
        by_node[label.node_id] = label
    if any(label.label == "pending" for label in ordered):
        raise ValueError("pending reviews are excluded from target labels")
    if any(label.binding != binding for label in ordered):
        raise ValueError("label binding does not match the bound run")
    if len(ordered) > 256:
        raise ValueError("label set supports at most 256 eligible labels")
    if len(ordered) < 8:
        raise ValueError("label set requires at least eight eligible labels")
    positive = sum(label.label == "positive" for label in ordered)
    negative = sum(label.label == "negative" for label in ordered)
    if positive == 0 or negative == 0:
        raise ValueError("label set requires both classes")
    if min(positive, negative) < 4:
        raise ValueError("label set requires at least four labels per class")
    payload: dict[str, object] = {
        "schemaVersion": schema_version,
        "binding": binding.model_dump(mode="json", by_alias=True),
        "sourceRecords": [
            {
                "sourceType": label.source_type,
                "sourceRecordId": label.source_record_id,
                "sourceRecordHash": label.source_record_hash,
                "reviewEventHash": label.review_event_hash,
            }
            for label in ordered
        ],
        "reviewEventHashes": [
            label.review_event_hash
            for label in ordered
            if label.review_event_hash is not None
        ],
        "labels": [
            {
                **label.model_dump(mode="json", by_alias=True, exclude_none=True),
                "reviewEventHash": label.review_event_hash,
            }
            for label in ordered
        ],
        "conflicts": [],
        "positiveCount": positive,
        "negativeCount": negative,
    }
    if sidecar_receipt is not None:
        payload["sidecarReceipt"] = sidecar_receipt.model_dump(
            mode="json", by_alias=True
        )
    payload["labelSetHash"] = canonical_sha256(payload)
    return TargetLabelSet.model_validate(payload)


def validate_sidecar_against_fused_graph(
    label_set: TargetLabelSet,
    *,
    artifact_document: Mapping[str, object],
    node_ids: Sequence[str],
    fused_indptr: np.ndarray,
) -> None:
    receipt = label_set.sidecar_receipt
    imported = tuple(
        label for label in label_set.labels if label.source_type == "imported_sidecar"
    )
    if not imported:
        return
    if receipt is None:
        raise ValueError("sidecar receipt is missing")
    if (
        artifact_document.get("artifactId") != label_set.binding.artifact_id
        or artifact_document.get("datasetContentHash")
        != label_set.binding.dataset_content_hash
        or artifact_document.get("graphVersionHash")
        != label_set.binding.graph_version_hash
        or artifact_document.get("bundleSha256") != receipt.bundle_sha256
        or artifact_document.get("nodeCount") != receipt.coverage.get("nodeCount")
        or len(node_ids) != receipt.coverage.get("nodeCount")
    ):
        raise ValueError("sidecar receipt does not match the bound artifact graph")
    indptr = np.asarray(fused_indptr)
    if indptr.shape != (len(node_ids) + 1,) or not np.issubdtype(
        indptr.dtype, np.integer
    ):
        raise ValueError("bound fused-degree inventory is invalid")
    degrees = np.diff(indptr)
    order = sorted(range(len(node_ids)), key=lambda index: (int(degrees[index]), node_ids[index]))
    strata = {
        node_ids[index]: min(3, position * 4 // len(node_ids))
        for position, index in enumerate(order)
    }
    degree_by_id = {
        node_id: int(degrees[index]) for index, node_id in enumerate(node_ids)
    }
    for label in imported:
        if label.node_id not in degree_by_id:
            raise ValueError("sidecar label node is absent from the bound graph")
        if label.fused_degree != degree_by_id[label.node_id]:
            raise ValueError("sidecar fused degree does not match the bound graph")
        if label.structural_stratum != strata[label.node_id]:
            raise ValueError("sidecar structural stratum does not match the bound graph")


def _unit_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if bool((norms <= EPSILON).any()):
        raise ValueError("frozen embeddings must have non-zero norm")
    return embeddings / norms


def _unit_centroid(rows: np.ndarray) -> np.ndarray:
    centroid = rows.mean(axis=0, dtype=np.float64)
    norm = float(np.linalg.norm(centroid))
    if norm <= EPSILON:
        raise ValueError("class centroid has insufficient direction")
    return centroid / norm


def _zscore(values: np.ndarray) -> np.ndarray:
    values64 = np.asarray(values, dtype=np.float64)
    return (values64 - float(values64.mean())) / max(float(values64.std()), EPSILON)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    bounded = np.clip(np.asarray(values, dtype=np.float64), -709.0, 709.0)
    return 1.0 / (1.0 + np.exp(-bounded))


def _centroid_hash(centroid: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(centroid, dtype="<f8").tobytes()).hexdigest()


def _ranks(values: np.ndarray, node_ids: Sequence[str]) -> np.ndarray:
    order = sorted(range(len(node_ids)), key=lambda index: (-float(values[index]), node_ids[index]))
    ranks = np.empty(len(node_ids), dtype=np.int64)
    ranks[np.asarray(order)] = np.arange(1, len(node_ids) + 1)
    return ranks


def fit_target_review_policy(
    label_set: TargetLabelSet,
    node_ids: Sequence[str],
    base_logits: np.ndarray,
    frozen_embeddings: np.ndarray,
    *,
    base_scores: np.ndarray,
    base_ranks: np.ndarray,
) -> FittedReviewPolicy:
    identifiers = tuple(node_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("bound run node inventory is invalid")
    logits = np.asarray(base_logits, dtype=np.float64)
    scores = np.asarray(base_scores)
    ranks = np.asarray(base_ranks)
    embeddings = np.asarray(frozen_embeddings)
    if logits.shape != (len(identifiers),):
        raise ValueError("base logits do not align to the bound run")
    if embeddings.shape != (len(identifiers), 256):
        raise ValueError("frozen embeddings must have shape [N, 256]")
    if scores.shape != (len(identifiers),) or ranks.shape != (len(identifiers),):
        raise ValueError("persisted base scores and ranks do not align to the bound run")
    if (
        not bool(np.isfinite(logits).all())
        or not bool(np.isfinite(scores).all())
        or not bool(np.isfinite(embeddings).all())
    ):
        raise ValueError("frozen run arrays must be finite")
    if bool(((scores < 0) | (scores > 1)).any()):
        raise ValueError("persisted base scores must be probabilities")
    if not np.issubdtype(ranks.dtype, np.integer) or set(map(int, ranks)) != set(
        range(1, len(identifiers) + 1)
    ):
        raise ValueError("persisted base ranks must be a complete one-based inventory")
    lookup = {node_id: index for index, node_id in enumerate(identifiers)}
    try:
        eligible = np.asarray([lookup[label.node_id] for label in label_set.labels])
    except KeyError as error:
        raise ValueError("eligible node is absent from the bound run") from error
    targets = np.asarray([label.label == "positive" for label in label_set.labels])
    normalized = _unit_rows(np.asarray(embeddings, dtype=np.float64))
    base_z = _zscore(logits)

    loo_margins = np.empty(len(eligible), dtype=np.float64)
    for position, (node_index, positive) in enumerate(zip(eligible, targets, strict=True)):
        train_mask = np.ones(len(eligible), dtype=bool)
        train_mask[position] = False
        train_indices = eligible[train_mask]
        train_targets = targets[train_mask]
        positive_centroid = _unit_centroid(normalized[train_indices[train_targets]])
        negative_centroid = _unit_centroid(normalized[train_indices[~train_targets]])
        run_margin = normalized @ positive_centroid - normalized @ negative_centroid
        loo_margins[position] = _zscore(run_margin)[node_index]

    losses: dict[str, float] = {}
    for candidate in LAMBDA_CANDIDATES:
        probabilities = _sigmoid(base_z[eligible] + candidate * loo_margins)
        clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
        positive_loss = -float(np.log(clipped[targets]).mean())
        negative_loss = -float(np.log(1.0 - clipped[~targets]).mean())
        losses[f"{candidate:g}"] = (positive_loss + negative_loss) / 2.0
    selected = min(LAMBDA_CANDIDATES, key=lambda value: (losses[f"{value:g}"], value))

    positive_centroid = _unit_centroid(normalized[eligible[targets]])
    negative_centroid = _unit_centroid(normalized[eligible[~targets]])
    margin_z = _zscore(normalized @ positive_centroid - normalized @ negative_centroid)
    priorities = _sigmoid(base_z + selected * margin_z)
    adapted_ranks = _ranks(priorities, identifiers)
    ready_hash = (
        canonical_sha256(
            {
                "labelSetHash": label_set.label_set_hash,
                "selectedLambda": selected,
                "positiveCentroidHash": _centroid_hash(positive_centroid),
                "negativeCentroidHash": _centroid_hash(negative_centroid),
                "recipeHash": label_set.binding.recipe_hash,
                "codeHash": label_set.binding.code_hash,
                "seed": label_set.binding.seed,
            }
        )
        if selected != 0.0
        else None
    )
    policy_payload: dict[str, object] = {
        "schemaVersion": POLICY_SCHEMA_VERSION,
        "binding": label_set.binding.model_dump(mode="json", by_alias=True),
        "labelSetHash": label_set.label_set_hash,
        "status": "ready" if selected != 0.0 else "insufficient_signal",
        "selectedLambda": selected,
        "lambdaCandidates": list(LAMBDA_CANDIDATES),
        "validationLosses": losses,
        "eligibleLabelCount": len(eligible),
        "positiveCount": label_set.positive_count,
        "negativeCount": label_set.negative_count,
        "embeddingDimension": 256,
        "positiveCentroidHash": _centroid_hash(positive_centroid),
        "negativeCentroidHash": _centroid_hash(negative_centroid),
        "normalizationEpsilon": EPSILON,
        "fittingRecipe": "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
        "readyPolicyHash": ready_hash,
    }
    policy_payload["policyHash"] = canonical_sha256(policy_payload)
    policy = TargetReviewPolicy.model_validate(policy_payload)
    rows = [
        {
            "nodeId": node_id,
            "baseScore": float(scores[index]),
            "baseRank": int(ranks[index]),
            "adaptedReviewPriority": float(priorities[index]),
            "adaptedRank": int(adapted_ranks[index]),
            "rankDelta": int(adapted_ranks[index] - ranks[index]),
        }
        for index, node_id in sorted(
            enumerate(identifiers), key=lambda item: int(ranks[item[0]])
        )
    ]
    comparison_payload: dict[str, object] = {
        "schemaVersion": COMPARISON_SCHEMA_VERSION,
        "binding": label_set.binding.model_dump(mode="json", by_alias=True),
        "policyHash": policy.policy_hash,
        "total": len(rows),
        "rows": rows,
    }
    comparison_payload["comparisonHash"] = canonical_sha256(comparison_payload)
    comparison = AdaptationComparison.model_validate(comparison_payload)
    for centroid in (positive_centroid, negative_centroid):
        centroid.setflags(write=False)
    return FittedReviewPolicy(
        policy=policy,
        comparison=comparison,
        positive_centroid=positive_centroid,
        negative_centroid=negative_centroid,
    )


def fit_target_review_policy_v2(
    label_set: TargetLabelSetV2,
    binding: AdaptationBinding,
    node_ids: Sequence[str],
    base_logits: np.ndarray,
    frozen_embeddings: np.ndarray,
    *,
    base_scores: np.ndarray,
    base_ranks: np.ndarray,
) -> FittedReviewPolicyV2:
    """Fit the generic v2 label contract against one fully bound frozen run."""

    from .target_tasks import TargetLabelSetV2

    label_set = TargetLabelSetV2.model_validate(label_set)
    evidence = []
    for row in label_set.labels:
        source_hash = canonical_sha256(
            {
                "schemaVersion": "socialgraph-fm.governance-target-label-evidence/2.0",
                "labelSetHash": label_set.label_set_hash,
                "nodeId": row.node_id,
                "label": row.label,
            }
        )
        evidence.append(
            LabelEvidence.model_validate(
                {
                    "nodeId": row.node_id,
                    "label": row.label,
                    "sourceType": "concluded_review",
                    "sourceRecordId": f"v2-label:{source_hash}",
                    "sourceRecordHash": source_hash,
                    "reviewEventHash": source_hash,
                    "binding": binding.model_dump(mode="json", by_alias=True),
                }
            )
        )
    legacy_labels = build_target_label_set(binding, evidence)
    fitted = fit_target_review_policy(
        legacy_labels,
        node_ids,
        base_logits,
        frozen_embeddings,
        base_scores=base_scores,
        base_ranks=base_ranks,
    )
    policy_payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-target-review-policy/2.0",
        "binding": binding.model_dump(mode="json", by_alias=True),
        "labelSetHash": label_set.label_set_hash,
        "status": fitted.policy.status,
        "selectedLambda": fitted.policy.selected_lambda,
        "eligibleLabelCount": len(label_set.labels),
        "positiveCount": label_set.positive_count,
        "negativeCount": label_set.negative_count,
        "fittingRecipe": fitted.policy.fitting_recipe,
        "baseOutputsImmutable": True,
        "adaptedOutputFields": ["adaptedReviewPriority", "adaptedRank"],
    }
    policy_payload["policyHash"] = canonical_sha256(policy_payload)
    policy = TargetReviewPolicyV2.model_validate(policy_payload)
    comparison_payload: dict[str, object] = {
        "schemaVersion": "socialgraph-fm.governance-adaptation-comparison/2.0",
        "binding": binding.model_dump(mode="json", by_alias=True),
        "policyHash": policy.policy_hash,
        "total": fitted.comparison.total,
        "baseOutputsImmutable": True,
        "rows": [
            row.model_dump(mode="json", by_alias=True)
            for row in fitted.comparison.rows
        ],
    }
    comparison_payload["comparisonHash"] = canonical_sha256(comparison_payload)
    return FittedReviewPolicyV2(
        policy=policy,
        comparison=AdaptationComparisonV2.model_validate(comparison_payload),
        positive_centroid=fitted.positive_centroid,
        negative_centroid=fitted.negative_centroid,
    )


__all__ = [
    "ADAPTATION_CODE_HASH",
    "AdaptationBinding",
    "AdaptationComparison",
    "AdaptationComparisonRow",
    "AdaptationComparisonV2",
    "AdaptationGovernanceHandoff",
    "FittedReviewPolicy",
    "FittedReviewPolicyV2",
    "LabelEvidence",
    "LabelSelectionRecipe",
    "LabelSourceRecord",
    "TargetLabelSet",
    "TargetPackageReceipt",
    "TargetReviewPolicy",
    "TargetReviewPolicyV2",
    "build_target_label_set",
    "fit_target_review_policy",
    "fit_target_review_policy_v2",
    "target_label_file_bytes",
    "validate_sidecar_against_fused_graph",
]
