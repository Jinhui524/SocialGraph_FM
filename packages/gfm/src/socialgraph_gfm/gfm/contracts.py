"""Strict, path-independent contracts for formal SocialGraph-FM GFM work."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from ..canonical import canonical_sha256

SHA256_PATTERN = r"^[0-9a-f]{64}$"
DomainRole: TypeAlias = Literal["pretraining", "adaptation", "evaluation"]
EvaluationKind: TypeAlias = Literal[
    "in_domain", "lodo", "product", "calibration", "fresh_process"
]
GovernanceTaskId: TypeAlias = Literal[
    "governance.collaboration_recommendation",
    "core.newcomer_support",
    "governance.community_pulse_forecast",
    "governance.conversation_escalation_watch",
    "core.community_health_observation",
    "core.coordination_review",
]

GFM_ACCEPTANCE_GATES = frozenset(
    {
        "three_domains",
        "lodo_complete",
        "product_metrics",
        "calibration_ece",
        "cuda_memory",
        "fresh_process_verification",
        "temporal_leakage_audit",
    }
)

GFM_PRETRAINING_ACCEPTANCE_GATES = frozenset(
    {
        "three_domains",
        "formal_pretrain_matrix",
        "lodo_complete",
        "variant_selection",
        "cuda_memory",
        "fresh_process_verification",
        "temporal_leakage_audit",
    }
)

GFM_COLLABORATION_TASK_ACCEPTANCE_GATES = frozenset(
    {
        "formal_seed_matrix",
        "provenance_binding",
        "physical_test_read_once",
        "product_metrics",
        "calibration_ece",
        "fresh_process_verification",
        "temporal_leakage_audit",
        "cuda_memory",
    }
)

PRODUCT_TASKS: frozenset[GovernanceTaskId] = frozenset(
    {
        "governance.collaboration_recommendation",
        "core.newcomer_support",
    }
)

_UNIT_INTERVAL_METRICS = frozenset(
    {
        "ndcg@20",
        "baseline_ndcg@20",
        "recall@20",
        "baseline_recall@20",
        "auprc",
        "label_prevalence",
        "strata_complete",
        "fresh_process_repeat_match",
        "brier",
        "ece",
    }
)
_SIGNED_UNIT_INTERVAL_METRICS = frozenset(
    {"bootstrap_ci95_ndcg_gain_lower", "bootstrap_ci95_ndcg_gain_upper"}
)

_FORBIDDEN_GOVERNANCE_PHRASES = (
    "流失",
    "低价值",
    "低 价值",
    "churn",
    "low-value",
    "low value",
)


class GfmContractModel(BaseModel):
    """Immutable JSON contract with strict fields and finite numeric values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        allow_inf_nan=False,
    )


def _create_hashed_contract(
    model: type[GfmContractModel],
    values: dict[str, Any],
    *,
    hash_alias: str,
    timestamped: bool = False,
) -> GfmContractModel:
    """Create a validated hash-bound contract without duplicating hash rules at call sites."""

    hash_name = model.model_fields[hash_alias].alias or hash_alias
    if hash_alias in values or hash_name in values:
        raise ValueError(f"{hash_name} is derived and must not be supplied")
    prepared = dict(values)
    if timestamped and "created_at" not in prepared and "createdAt" not in prepared:
        prepared["createdAt"] = datetime.now(UTC)
    provisional = model.model_validate(
        {**prepared, hash_name: "0" * 64}, context={"skip_hash_validation": True}
    )
    logical_payload = getattr(provisional, "logical_payload")()
    normalized = provisional.model_dump(mode="python", by_alias=True)
    normalized[hash_name] = canonical_sha256(logical_payload)
    return model.model_validate(normalized)


def _check_bound_hash(info: ValidationInfo) -> bool:
    return not bool(info.context and info.context.get("skip_hash_validation"))


def _require_aware(value: datetime | None, field_name: str) -> None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{field_name} must be timezone-aware")


def _assert_unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")


def _assert_safe_governance_text(value: Any) -> None:
    if isinstance(value, str):
        normalized = value.casefold()
        for phrase in _FORBIDDEN_GOVERNANCE_PHRASES:
            if phrase.casefold() in normalized:
                raise ValueError(
                    "governance artifacts must not use churn or low-value labels"
                )
    elif isinstance(value, dict):
        for nested in value.values():
            _assert_safe_governance_text(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_safe_governance_text(nested)


def _validate_evaluation_metrics(metrics: dict[str, float]) -> None:
    """Validate metric semantics before evidence reaches the registry boundary."""

    for name, value in metrics.items():
        if not name or len(name) > 160 or any(char.isspace() for char in name):
            raise ValueError("metric names must be non-empty, compact identifiers")
        if name in _UNIT_INTERVAL_METRICS or name.startswith(("few_shot_", "ece_")):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"metric {name} must be in [0, 1]")
        elif name in _SIGNED_UNIT_INTERVAL_METRICS:
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"metric {name} must be in [-1, 1]")
        elif name.endswith("_count") or name in {"query_count", "outcome_count"}:
            if value < 0.0 or not value.is_integer():
                raise ValueError(f"metric {name} must be a non-negative integer")
        elif "loss" in name.casefold() and value < 0.0:
            raise ValueError(f"metric {name} must be non-negative")
class GfmDomainCorpusManifest(GfmContractModel):
    schema_version: Literal["gfm.domain-corpus/1.0"] = Field(
        "gfm.domain-corpus/1.0", alias="schemaVersion"
    )
    corpus_id: str = Field(alias="corpusId", min_length=1, max_length=200)
    domain_id: str = Field(alias="domainId", min_length=1, max_length=200)
    dataset_name: str = Field(alias="datasetName", min_length=1, max_length=300)
    dataset_version: str = Field(alias="datasetVersion", min_length=1, max_length=100)
    dataset_role: DomainRole = Field(alias="datasetRole")
    license_id: str = Field(alias="licenseId", min_length=1, max_length=200)
    license_evidence_hash: str = Field(alias="licenseEvidenceHash", pattern=SHA256_PATTERN)
    source_hash: str = Field(alias="sourceHash", pattern=SHA256_PATTERN)
    content_hash: str = Field(alias="contentHash", pattern=SHA256_PATTERN)
    split_hash: str = Field(alias="splitHash", pattern=SHA256_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=1)
    edge_count: int = Field(alias="edgeCount", ge=1)
    feature_modalities: tuple[
        Literal["numeric", "categorical", "text", "temporal", "structural"], ...
    ] = Field(alias="featureModalities", min_length=1)
    task_ids: tuple[str, ...] = Field(alias="taskIds", min_length=1)
    point_in_time_safe: bool = Field(alias="pointInTimeSafe")
    public_checkpoint_eligible: bool = Field(alias="publicCheckpointEligible")
    temporal_cutoff: datetime | None = Field(None, alias="temporalCutoff")
    source_uri: str | None = Field(None, alias="sourceUri", max_length=4096)
    artifact_path: str = Field(alias="artifactPath", min_length=1)
    logical_hash: str = Field(alias="logicalHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        """Portable identity: storage path and observation time are deliberately excluded."""

        return self.model_dump(
            mode="python",
            by_alias=True,
            exclude={"artifact_path", "logical_hash", "created_at"},
        )

    @classmethod
    def create(cls, **values: Any) -> GfmDomainCorpusManifest:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="logical_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_manifest(self, info: ValidationInfo) -> GfmDomainCorpusManifest:
        _require_aware(self.created_at, "createdAt")
        _require_aware(self.temporal_cutoff, "temporalCutoff")
        _assert_unique(self.feature_modalities, "featureModalities")
        _assert_unique(self.task_ids, "taskIds")
        if self.dataset_role == "pretraining" and not self.point_in_time_safe:
            raise ValueError("pretraining corpora must be point-in-time safe")
        if _check_bound_hash(info) and self.logical_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("logicalHash does not match domain corpus payload")
        return self


class GfmTaskProtocolManifest(GfmContractModel):
    schema_version: Literal["gfm.task-protocol/1.0"] = Field(
        "gfm.task-protocol/1.0", alias="schemaVersion"
    )
    protocol_id: str = Field(alias="protocolId", min_length=1, max_length=200)
    task_id: GovernanceTaskId = Field(alias="taskId")
    task_family: Literal[
        "collaboration_ranking",
        "newcomer_support",
        "community_forecast",
        "conversation_escalation",
    ] = Field(alias="taskFamily")
    domain_ids: tuple[str, ...] = Field(alias="domainIds", min_length=1)
    split_strategy: Literal["temporal", "lodo", "few_shot_temporal"] = Field(
        alias="splitStrategy"
    )
    objectives: tuple[str, ...] = Field(min_length=1)
    primary_metrics: tuple[str, ...] = Field(alias="primaryMetrics", min_length=1)
    calibration_metrics: tuple[str, ...] = Field(
        ("ece", "brier"), alias="calibrationMetrics", min_length=1
    )
    minimum_seeds: int = Field(3, alias="minimumSeeds", ge=3)
    inference_cutoff_required: Literal[True] = Field(True, alias="inferenceCutoffRequired")
    future_edges_excluded: Literal[True] = Field(True, alias="futureEdgesExcluded")
    human_review_required: Literal[True] = Field(True, alias="humanReviewRequired")
    protected_attributes_excluded: Literal[True] = Field(
        True, alias="protectedAttributesExcluded"
    )
    protocol_hash: str = Field(alias="protocolHash", pattern=SHA256_PATTERN)

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python", by_alias=True, exclude={"protocol_hash"}
        )

    @classmethod
    def create(cls, **values: Any) -> GfmTaskProtocolManifest:
        return cls.model_validate(
            _create_hashed_contract(cls, values, hash_alias="protocol_hash")
        )

    @model_validator(mode="after")
    def validate_protocol(self, info: ValidationInfo) -> GfmTaskProtocolManifest:
        _assert_unique(self.domain_ids, "domainIds")
        _assert_unique(self.objectives, "objectives")
        _assert_unique(self.primary_metrics, "primaryMetrics")
        _assert_unique(self.calibration_metrics, "calibrationMetrics")
        if _check_bound_hash(info) and self.protocol_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("protocolHash does not match task protocol payload")
        return self


class GfmArchitectureConfig(GfmContractModel):
    candidates: tuple[Literal["core-base", "core-moe"], ...] = Field(min_length=2)
    hidden_channels: Literal[128] = Field(128, alias="hiddenChannels")
    num_layers: Literal[2] = Field(2, alias="numLayers")
    dropout: float = Field(0.2, ge=0.0, lt=1.0)
    relation_bases: Literal[8] = Field(8, alias="relationBases")
    time_channels: Literal[32] = Field(32, alias="timeChannels")
    neighbor_fanout: tuple[int, ...] = Field((15, 10), alias="neighborFanout", min_length=2)
    domain_adapter_bottleneck: int = Field(32, alias="domainAdapterBottleneck", ge=1)
    moe_experts: Literal[2] = Field(2, alias="moeExperts")
    include_null_expert: Literal[True] = Field(True, alias="includeNullExpert")


class GfmDomainConfig(GfmContractModel):
    domain_id: str = Field(alias="domainId", min_length=1)
    domain_family: Literal[
        "academic-collaboration", "software-activity", "online-community"
    ] = Field(alias="domainFamily")
    required: Literal[True] = True
    text_enabled: bool = Field(alias="textEnabled")


class GfmTextEncoderConfig(GfmContractModel):
    model_id: str = Field(alias="modelId", min_length=1)
    revision: str = Field(min_length=1)
    license: str = Field(min_length=1)
    frozen: Literal[True] = True
    output_channels: int = Field(alias="outputChannels", ge=1)
    max_tokens: int = Field(alias="maxTokens", ge=1)
    training_time_encoding_forbidden: Literal[True] = Field(
        True, alias="trainingTimeEncodingForbidden"
    )


class GfmLossWeights(GfmContractModel):
    temporal_next_event: float = Field(alias="temporalNextEvent", gt=0.0)
    masked_attribute: float = Field(alias="maskedAttribute", gt=0.0)
    masked_relation_type: float = Field(alias="maskedRelationType", gt=0.0)
    log_time_delta: float = Field(alias="logTimeDelta", gt=0.0)
    text_structure_alignment: float = Field(alias="textStructureAlignment", gt=0.0)
    cross_domain_distribution_alignment: float = Field(
        alias="crossDomainDistributionAlignment", gt=0.0
    )
    moe_route_balance: float = Field(alias="moeRouteBalance", gt=0.0)


class GfmNegativeSamplingConfig(GfmContractModel):
    hard_ratio: float = Field(alias="hardRatio", ge=0.0, le=1.0)
    degree_matched_ratio: float = Field(alias="degreeMatchedRatio", ge=0.0, le=1.0)
    uniform_ratio: float = Field(alias="uniformRatio", ge=0.0, le=1.0)
    exact: Literal[True] = True
    future_access_forbidden: Literal[True] = Field(True, alias="futureAccessForbidden")

    @model_validator(mode="after")
    def validate_ratios(self) -> GfmNegativeSamplingConfig:
        if abs(self.hard_ratio + self.degree_matched_ratio + self.uniform_ratio - 1.0) > 1e-12:
            raise ValueError("negative sampling ratios must sum to one")
        return self


class GfmOptimizationConfig(GfmContractModel):
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(alias="learningRate", gt=0.0)
    weight_decay: float = Field(alias="weightDecay", ge=0.0)
    warmup_ratio: float = Field(alias="warmupRatio", ge=0.0, lt=1.0)
    schedule: Literal["cosine"] = "cosine"
    gradient_clip: float = Field(alias="gradientClip", gt=0.0)
    precision: Literal["fp16"] = "fp16"
    candidate_batch_sizes: tuple[int, ...] = Field(alias="candidateBatchSizes", min_length=1)
    effective_batch_size: int = Field(alias="effectiveBatchSize", ge=1)
    cuda_memory_limit_mib: Literal[7168] = Field(7168, alias="cudaMemoryLimitMiB")
    num_workers: Literal[0] = Field(0, alias="numWorkers")


class GfmPhaseConfig(GfmContractModel):
    seeds: tuple[int, ...] = Field(min_length=1)
    max_steps: int = Field(alias="maxSteps", ge=1)
    minimum_steps: int = Field(alias="minimumSteps", ge=0)
    evaluation_every_steps: int = Field(alias="evaluationEverySteps", ge=1)
    patience_evaluations: int = Field(alias="patienceEvaluations", ge=1)
    read_test: Literal[False] | None = Field(None, alias="readTest")
    read_test_after_best_only: Literal[True] | None = Field(
        None, alias="readTestAfterBestOnly"
    )

    @model_validator(mode="after")
    def validate_phase(self) -> GfmPhaseConfig:
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("phase seeds must be unique")
        if self.minimum_steps > self.max_steps:
            raise ValueError("minimumSteps cannot exceed maxSteps")
        if (self.read_test is None) == (self.read_test_after_best_only is None):
            raise ValueError("phase must declare exactly one test-read policy")
        return self


class GfmCheckpointConfig(GfmContractModel):
    maximum_per_run: Literal[3] = Field(3, alias="maximumPerRun")
    fresh_process_verification_required: Literal[True] = Field(
        True, alias="freshProcessVerificationRequired"
    )
    registrable_before_acceptance: Literal[False] = Field(
        False, alias="registrableBeforeAcceptance"
    )


class GfmPromotionConfig(GfmContractModel):
    moe_minimum_mean_relative_gain: float = Field(alias="moeMinimumMeanRelativeGain", ge=0.0)
    moe_maximum_per_domain_relative_regression: float = Field(
        alias="moeMaximumPerDomainRelativeRegression", ge=0.0
    )


class GfmProductConfig(GfmContractModel):
    collaboration_rerank_weights: dict[str, float] = Field(
        alias="collaborationRerankWeights"
    )
    collaboration_horizon_days: int = Field(alias="collaborationHorizonDays", ge=1)
    newcomer_observation_days: int = Field(alias="newcomerObservationDays", ge=1)
    newcomer_horizon_days: int = Field(alias="newcomerHorizonDays", ge=1)

    @model_validator(mode="after")
    def validate_product(self) -> GfmProductConfig:
        expected = {
            "calibratedProbability": 0.70,
            "topicComplementarity": 0.15,
            "bridgeGain": 0.10,
            "institutionDiversity": 0.05,
        }
        if self.collaboration_rerank_weights != expected:
            raise ValueError("collaboration rerank weights must use the fixed 70/15/10/5 policy")
        return self


class GfmTransferConfig(GfmContractModel):
    """Hash-bound cross-domain adaptation protocol constants."""

    lodo_target_adaptation_steps: Literal[5000] = Field(
        5000, alias="lodoTargetAdaptationSteps"
    )


class GfmAcceptanceConfig(GfmContractModel):
    few_shot_fractions: tuple[float, ...] = Field(alias="fewShotFractions", min_length=1)
    minimum_random_init_relative_gain_at_5_percent: float = Field(
        alias="minimumRandomInitRelativeGainAt5Percent", ge=0.0
    )
    minimum_single_domain_relative_gain_at_5_percent: float = Field(
        alias="minimumSingleDomainRelativeGainAt5Percent", ge=0.0
    )
    maximum_domain_relative_regression: float = Field(
        alias="maximumDomainRelativeRegression", ge=0.0
    )
    minimum_improved_domain_families: int = Field(alias="minimumImprovedDomainFamilies", ge=1)
    minimum_product_relative_gain: float = Field(alias="minimumProductRelativeGain", ge=0.0)
    minimum_newcomer_auprc_above_prevalence: float = Field(
        alias="minimumNewcomerAuprcAbovePrevalence", ge=0.0
    )
    maximum_ece: float = Field(alias="maximumEce", ge=0.0, le=1.0)
    bootstrap_confidence: float = Field(alias="bootstrapConfidence", gt=0.0, lt=1.0)
    cuda_memory_limit_mib: Literal[7168] = Field(7168, alias="cudaMemoryLimitMiB")


class GfmPretrainConfig(GfmContractModel):
    """Exact, strict contract for the checked SocialGraph-FM Core config."""

    schema_version: Literal["gfm.pretrain-config/1.0"] = Field(
        "gfm.pretrain-config/1.0", alias="schemaVersion"
    )
    config_id: Literal["socialgraph-core"] = Field(
        "socialgraph-core", alias="configId"
    )
    architecture: GfmArchitectureConfig
    domains: tuple[GfmDomainConfig, ...] = Field(min_length=3, max_length=3)
    text_encoder: GfmTextEncoderConfig = Field(alias="textEncoder")
    loss_weights: GfmLossWeights = Field(alias="lossWeights")
    negative_sampling: GfmNegativeSamplingConfig = Field(alias="negativeSampling")
    optimization: GfmOptimizationConfig
    dev: GfmPhaseConfig
    formal: GfmPhaseConfig
    checkpoint: GfmCheckpointConfig
    promotion: GfmPromotionConfig
    product: GfmProductConfig
    transfer: GfmTransferConfig
    acceptance: GfmAcceptanceConfig
    run_kind: Literal["formal", "exploratory"] = Field("formal", alias="runKind")
    overrides: dict[str, Any] | None = None
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python",
            by_alias=True,
            exclude={"config_hash", "run_kind", "overrides"},
            exclude_none=True,
        )

    @classmethod
    def create(cls, **values: Any) -> GfmPretrainConfig:
        return cls.model_validate(
            _create_hashed_contract(cls, values, hash_alias="config_hash")
        )

    @model_validator(mode="after")
    def validate_config(self, info: ValidationInfo) -> GfmPretrainConfig:
        domain_ids = tuple(item.domain_id for item in self.domains)
        _assert_unique(domain_ids, "domains.domainId")
        expected_families = {
            "academic-collaboration",
            "software-activity",
            "online-community",
        }
        if {item.domain_family for item in self.domains} != expected_families:
            raise ValueError("formal config requires the three independent domain families")
        if len(self.formal.seeds) < 3:
            raise ValueError("formal config requires at least three seeds")
        if self.dev.max_steps != 2000 or self.formal.max_steps != 30000:
            raise ValueError("dev/formal maxSteps must match the checked v1 protocol")
        if _check_bound_hash(info) and self.config_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("configHash does not match pretrain config payload")
        return self


class GfmRunManifest(GfmContractModel):
    schema_version: Literal["gfm.run/1.0"] = Field("gfm.run/1.0", alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=200)
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=200)
    phase: Literal["pretrain", "adapt", "evaluate", "lodo"]
    architecture_variant: Literal["core-base", "core-moe"] = Field(
        alias="architectureVariant"
    )
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    domain_ids: tuple[str, ...] = Field(alias="domainIds", min_length=1)
    held_out_domain: str | None = Field(None, alias="heldOutDomain", max_length=200)
    seed: int = Field(ge=0, le=2**32 - 1)
    code_hash: str = Field(alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str = Field(alias="environmentHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    corpus_hashes: tuple[str, ...] = Field(alias="corpusHashes", min_length=1)
    task_protocol_hashes: tuple[str, ...] = Field(alias="taskProtocolHashes", min_length=1)
    started_at: datetime = Field(alias="startedAt")
    finished_at: datetime | None = Field(None, alias="finishedAt")
    peak_cuda_memory_mib: float | None = Field(None, alias="peakCudaMemoryMiB", ge=0.0)
    failure_code: str | None = Field(None, alias="failureCode", max_length=200)
    artifact_paths: tuple[str, ...] = Field((), alias="artifactPaths")
    promotion_eligible: Literal[False] = Field(False, alias="promotionEligible")
    manifest_hash: str = Field(alias="manifestHash", pattern=SHA256_PATTERN)

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python",
            by_alias=True,
            exclude={"artifact_paths", "manifest_hash", "started_at", "finished_at"},
        )

    @classmethod
    def create(cls, **values: Any) -> GfmRunManifest:
        return cls.model_validate(
            _create_hashed_contract(cls, values, hash_alias="manifest_hash")
        )

    @model_validator(mode="after")
    def validate_run(self, info: ValidationInfo) -> GfmRunManifest:
        _require_aware(self.started_at, "startedAt")
        _require_aware(self.finished_at, "finishedAt")
        _assert_unique(self.domain_ids, "domainIds")
        _assert_unique(self.corpus_hashes, "corpusHashes")
        _assert_unique(self.task_protocol_hashes, "taskProtocolHashes")
        if self.phase == "lodo" and not self.held_out_domain:
            raise ValueError("LODO runs require heldOutDomain")
        if self.phase != "lodo" and self.held_out_domain is not None:
            raise ValueError("heldOutDomain is only valid for LODO runs")
        if self.held_out_domain in self.domain_ids:
            raise ValueError("heldOutDomain must not be a training domain")
        terminal = self.status in {"succeeded", "failed", "cancelled"}
        if terminal != (self.finished_at is not None):
            raise ValueError("terminal run status and finishedAt must agree")
        if self.status == "failed" and not self.failure_code:
            raise ValueError("failed runs require failureCode")
        if self.status != "failed" and self.failure_code is not None:
            raise ValueError("failureCode is only valid for failed runs")
        if _check_bound_hash(info) and self.manifest_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("manifestHash does not match GFM run payload")
        return self


class GfmCheckpointManifest(GfmContractModel):
    schema_version: Literal["gfm.checkpoint/1.0"] = Field(
        "gfm.checkpoint/1.0", alias="schemaVersion"
    )
    checkpoint_id: str = Field(alias="checkpointId", min_length=1, max_length=200)
    run_id: str = Field(alias="runId", min_length=1, max_length=200)
    epoch: int = Field(ge=0)
    step: int = Field(ge=0)
    component_names: tuple[str, ...] = Field(alias="componentNames", min_length=1)
    state_hash: str = Field(alias="stateHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    corpus_hashes: tuple[str, ...] = Field(alias="corpusHashes", min_length=1)
    artifact_sha256: str = Field(alias="artifactSha256", pattern=SHA256_PATTERN)
    artifact_path: str = Field(alias="artifactPath", min_length=1)
    registrable: Literal[False] = False
    fresh_process_digest: str | None = Field(
        None, alias="freshProcessDigest", pattern=SHA256_PATTERN
    )
    logical_hash: str = Field(alias="logicalHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python",
            by_alias=True,
            exclude={"artifact_path", "logical_hash", "created_at"},
        )

    @classmethod
    def create(cls, **values: Any) -> GfmCheckpointManifest:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="logical_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_checkpoint(self, info: ValidationInfo) -> GfmCheckpointManifest:
        _require_aware(self.created_at, "createdAt")
        _assert_unique(self.component_names, "componentNames")
        _assert_unique(self.corpus_hashes, "corpusHashes")
        if tuple(sorted(self.component_names)) != self.component_names:
            raise ValueError("componentNames must use deterministic sorted order")
        if _check_bound_hash(info) and self.logical_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("logicalHash does not match checkpoint payload")
        return self


class GfmEvaluationReport(GfmContractModel):
    schema_version: Literal["gfm.evaluation/1.0"] = Field(
        "gfm.evaluation/1.0", alias="schemaVersion"
    )
    report_id: str = Field(alias="reportId", min_length=1, max_length=200)
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=200)
    run_id: str = Field(alias="runId", min_length=1, max_length=200)
    checkpoint_id: str = Field(alias="checkpointId", min_length=1, max_length=200)
    evaluation_kind: EvaluationKind = Field(alias="evaluationKind")
    domain_id: str = Field(alias="domainId", min_length=1, max_length=200)
    held_out_domain: str | None = Field(None, alias="heldOutDomain", max_length=200)
    task_id: GovernanceTaskId | None = Field(None, alias="taskId")
    evaluator_code_hash: str | None = Field(
        None, alias="evaluatorCodeHash", pattern=SHA256_PATTERN
    )
    evaluator_environment_hash: str | None = Field(
        None, alias="evaluatorEnvironmentHash", pattern=SHA256_PATTERN
    )
    seed: int = Field(ge=0, le=2**32 - 1)
    metrics: dict[str, float] = Field(min_length=1)
    evidence_artifact_hash: str = Field(
        alias="evidenceArtifactHash", pattern=SHA256_PATTERN
    )
    evidence_artifact_path: str = Field(alias="evidenceArtifactPath", min_length=1)
    baseline_definition_hash: str | None = Field(
        None, alias="baselineDefinitionHash", pattern=SHA256_PATTERN
    )
    strata_definition_hash: str | None = Field(
        None, alias="strataDefinitionHash", pattern=SHA256_PATTERN
    )
    ece: float | None = Field(None, ge=0.0, le=1.0)
    brier: float | None = Field(None, ge=0.0, le=1.0)
    peak_cuda_memory_mib: float | None = Field(None, alias="peakCudaMemoryMiB", ge=0.0)
    leakage_audit_passed: bool = Field(alias="leakageAuditPassed")
    leakage_audit_hash: str = Field(
        alias="leakageAuditHash", pattern=SHA256_PATTERN
    )
    leakage_audit_path: str = Field(alias="leakageAuditPath", min_length=1)
    fresh_process_verified: bool = Field(False, alias="freshProcessVerified")
    verification_digest: str | None = Field(
        None, alias="verificationDigest", pattern=SHA256_PATTERN
    )
    warnings: tuple[str, ...] = ()
    report_hash: str = Field(alias="reportHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python",
            by_alias=True,
            exclude={
                "report_hash",
                "created_at",
                "evidence_artifact_path",
                "leakage_audit_path",
            },
        )

    @classmethod
    def create(cls, **values: Any) -> GfmEvaluationReport:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="report_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_report(self, info: ValidationInfo) -> GfmEvaluationReport:
        _require_aware(self.created_at, "createdAt")
        _validate_evaluation_metrics(self.metrics)
        if self.evaluation_kind == "lodo":
            if not self.held_out_domain:
                raise ValueError("LODO reports require heldOutDomain")
        elif self.held_out_domain is not None:
            raise ValueError("heldOutDomain is only valid for LODO reports")
        if self.evaluation_kind in {"product", "calibration"} and self.task_id is None:
            raise ValueError("product and calibration reports require taskId")
        if (
            self.evaluation_kind in {"product", "calibration"}
            and self.task_id not in PRODUCT_TASKS
        ):
            raise ValueError("Core product evidence is limited to the two fixed tasks")
        if self.evaluation_kind in {"product", "calibration"}:
            if (
                self.evaluator_code_hash is None
                or self.evaluator_environment_hash is None
            ):
                raise ValueError(
                    "product and calibration reports require evaluator provenance"
                )
        elif (
            self.evaluator_code_hash is not None
            or self.evaluator_environment_hash is not None
        ):
            raise ValueError(
                "evaluator provenance is reserved for product and calibration reports"
            )
        if self.evaluation_kind == "product":
            if self.baseline_definition_hash is None:
                raise ValueError("product reports require an immutable baseline definition")
        elif self.baseline_definition_hash is not None:
            raise ValueError("baselineDefinitionHash is reserved for product reports")
        if self.evaluation_kind == "calibration" and self.ece is None:
            raise ValueError("calibration reports require ECE")
        if self.evaluation_kind == "calibration":
            if self.strata_definition_hash is None:
                raise ValueError("calibration reports require immutable strata definitions")
            if self.brier is None:
                raise ValueError("calibration reports require a Brier score")
            if self.metrics.get("ece") != self.ece:
                raise ValueError("calibration metric ECE must equal the report ECE")
            if self.metrics.get("brier") != self.brier:
                raise ValueError("calibration metric Brier must equal the report Brier")
        elif self.strata_definition_hash is not None:
            raise ValueError("strataDefinitionHash is reserved for calibration reports")
        if self.leakage_audit_passed and not self.leakage_audit_hash:
            raise ValueError("a passed leakage audit requires immutable evidence")
        if self.evaluation_kind == "fresh_process":
            if not self.fresh_process_verified or self.verification_digest is None:
                raise ValueError(
                    "fresh-process reports require verification and a digest"
                )
            if self.metrics.get("fresh_process_repeat_match") != 1.0:
                raise ValueError(
                    "fresh-process reports require an exact repeated-process match"
                )
        elif self.fresh_process_verified or self.verification_digest is not None:
            raise ValueError(
                "freshProcessVerified and verificationDigest are reserved for fresh-process reports"
            )
        if _check_bound_hash(info) and self.report_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("reportHash does not match evaluation payload")
        return self


class GfmAcceptanceManifest(GfmContractModel):
    schema_version: Literal["gfm.acceptance/1.0"] = Field(
        "gfm.acceptance/1.0", alias="schemaVersion"
    )
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=200)
    checkpoint_id: str = Field(alias="checkpointId", min_length=1, max_length=200)
    accepted: bool
    domain_ids: tuple[str, ...] = Field(alias="domainIds")
    lodo_domains: tuple[str, ...] = Field(alias="lodoDomains")
    product_task_ids: tuple[GovernanceTaskId, ...] = Field(alias="productTaskIds")
    corpus_hashes: tuple[str, ...] = Field(alias="corpusHashes")
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    code_hash: str = Field(alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str = Field(alias="environmentHash", pattern=SHA256_PATTERN)
    evaluation_report_hashes: tuple[str, ...] = Field(alias="evaluationReportHashes")
    delivery_evidence_report_hashes: tuple[str, ...] = Field(
        alias="deliveryEvidenceReportHashes"
    )
    maximum_ece: float | None = Field(None, alias="maximumEce", ge=0.0, le=1.0)
    peak_cuda_memory_mib: float | None = Field(None, alias="peakCudaMemoryMiB", ge=0.0)
    fresh_process_digests: tuple[str, ...] = Field(alias="freshProcessDigests")
    metric_summary: dict[str, dict[str, float]] = Field(alias="metricSummary")
    gates: dict[str, bool]
    reasons: tuple[str, ...] = ()
    report_hash: str = Field(alias="reportHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python", by_alias=True, exclude={"report_hash", "created_at"}
        )

    @classmethod
    def create(cls, **values: Any) -> GfmAcceptanceManifest:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="report_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_acceptance(self, info: ValidationInfo) -> GfmAcceptanceManifest:
        _require_aware(self.created_at, "createdAt")
        _assert_unique(self.domain_ids, "domainIds")
        _assert_unique(self.lodo_domains, "lodoDomains")
        _assert_unique(self.product_task_ids, "productTaskIds")
        _assert_unique(self.corpus_hashes, "corpusHashes")
        _assert_unique(self.evaluation_report_hashes, "evaluationReportHashes")
        _assert_unique(
            self.delivery_evidence_report_hashes,
            "deliveryEvidenceReportHashes",
        )
        _assert_unique(self.fresh_process_digests, "freshProcessDigests")
        if not set(self.delivery_evidence_report_hashes).issubset(
            self.evaluation_report_hashes
        ):
            raise ValueError(
                "deliveryEvidenceReportHashes must reference evaluationReportHashes"
            )
        if set(self.gates) != GFM_ACCEPTANCE_GATES:
            raise ValueError("acceptance must contain exactly the fixed hard gates")
        derived = all(self.gates.values()) and not self.reasons
        if self.accepted != derived:
            raise ValueError("accepted must be derived from all hard gates and reasons")
        if self.accepted and len(self.domain_ids) < 3:
            raise ValueError("accepted GFM requires at least three domains")
        if self.accepted and set(self.lodo_domains) != set(self.domain_ids):
            raise ValueError("accepted GFM requires LODO coverage for every domain")
        if self.accepted and set(self.product_task_ids) != PRODUCT_TASKS:
            raise ValueError("accepted Core GFM requires exactly the two product tasks")
        if self.accepted and (
            len(self.evaluation_report_hashes) < 24
            or len(self.delivery_evidence_report_hashes) != 5
            or len(self.fresh_process_digests) < 3
            or not self.metric_summary
        ):
            raise ValueError(
                "accepted GFM requires complete evaluation, fresh-process, and metric evidence"
            )
        if _check_bound_hash(info) and self.report_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("reportHash does not match acceptance payload")
        return self


class GfmPretrainingAcceptanceManifest(GfmContractModel):
    """Immutable acceptance boundary for formal pretraining and LODO only.

    Product adaptation evidence deliberately cannot appear in this contract.  A
    successful instance proves that the frozen two-architecture, three-seed,
    three-domain experiment is reusable as an offline backbone; it does not
    make a product model registrable or serving-ready.
    """

    schema_version: Literal["gfm.pretraining-acceptance/1.0"] = Field(
        "gfm.pretraining-acceptance/1.0", alias="schemaVersion"
    )
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=200)
    accepted: bool
    architecture_variants: tuple[Literal["core-base", "core-moe"], ...] = Field(
        alias="architectureVariants"
    )
    selected_variant: Literal["core-base", "core-moe"] = Field(
        alias="selectedVariant"
    )
    formal_seeds: tuple[int, ...] = Field(alias="formalSeeds")
    domain_ids: tuple[str, ...] = Field(alias="domainIds")
    lodo_domains: tuple[str, ...] = Field(alias="lodoDomains")
    corpus_hashes: tuple[str, ...] = Field(alias="corpusHashes")
    pretrain_run_ids: tuple[str, ...] = Field(alias="pretrainRunIds")
    lodo_run_ids: tuple[str, ...] = Field(alias="lodoRunIds")
    evidence_checkpoint_ids: tuple[str, ...] = Field(alias="evidenceCheckpointIds")
    selected_checkpoint_ids: tuple[str, ...] = Field(alias="selectedCheckpointIds")
    evaluation_report_hashes: tuple[str, ...] = Field(alias="evaluationReportHashes")
    fresh_process_digests: tuple[str, ...] = Field(alias="freshProcessDigests")
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    code_hash: str = Field(alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str = Field(alias="environmentHash", pattern=SHA256_PATTERN)
    peak_cuda_memory_mib: float | None = Field(
        None, alias="peakCudaMemoryMiB", ge=0.0
    )
    metric_summary: dict[str, dict[str, float | str]] = Field(alias="metricSummary")
    gates: dict[str, bool]
    reasons: tuple[str, ...] = ()
    report_hash: str = Field(alias="reportHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python", by_alias=True, exclude={"report_hash", "created_at"}
        )

    @classmethod
    def create(cls, **values: Any) -> GfmPretrainingAcceptanceManifest:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="report_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_acceptance(
        self, info: ValidationInfo
    ) -> GfmPretrainingAcceptanceManifest:
        _require_aware(self.created_at, "createdAt")
        for values, name in (
            (self.architecture_variants, "architectureVariants"),
            (self.domain_ids, "domainIds"),
            (self.lodo_domains, "lodoDomains"),
            (self.corpus_hashes, "corpusHashes"),
            (self.pretrain_run_ids, "pretrainRunIds"),
            (self.lodo_run_ids, "lodoRunIds"),
            (self.evidence_checkpoint_ids, "evidenceCheckpointIds"),
            (self.selected_checkpoint_ids, "selectedCheckpointIds"),
            (self.evaluation_report_hashes, "evaluationReportHashes"),
            (self.fresh_process_digests, "freshProcessDigests"),
        ):
            _assert_unique(values, name)
        if len(set(self.formal_seeds)) != len(self.formal_seeds):
            raise ValueError("formalSeeds must contain unique values")
        if tuple(sorted(self.architecture_variants)) != self.architecture_variants:
            raise ValueError("architectureVariants must use deterministic sorted order")
        if tuple(sorted(self.formal_seeds)) != self.formal_seeds:
            raise ValueError("formalSeeds must use deterministic sorted order")
        if self.selected_variant not in self.architecture_variants:
            raise ValueError("selectedVariant must be one of architectureVariants")
        if not set(self.selected_checkpoint_ids).issubset(
            self.evidence_checkpoint_ids
        ):
            raise ValueError(
                "selectedCheckpointIds must reference evidenceCheckpointIds"
            )
        if set(self.gates) != GFM_PRETRAINING_ACCEPTANCE_GATES:
            raise ValueError(
                "pretraining acceptance must contain exactly the fixed hard gates"
            )
        derived = all(self.gates.values()) and not self.reasons
        if self.accepted != derived:
            raise ValueError(
                "pretraining accepted must be derived from all hard gates and reasons"
            )
        if self.accepted and (
            self.architecture_variants != ("core-base", "core-moe")
            or self.formal_seeds != (20260821, 20260822, 20260823)
            or len(self.domain_ids) != 3
            or set(self.lodo_domains) != set(self.domain_ids)
            or len(self.corpus_hashes) != 3
            or len(self.pretrain_run_ids) != 6
            or len(self.lodo_run_ids) != 18
            or len(self.evidence_checkpoint_ids) != 24
            or len(self.selected_checkpoint_ids) != 3
            or len(self.evaluation_report_hashes) != 30
            or len(self.fresh_process_digests) != 6
        ):
            raise ValueError(
                "accepted pretraining requires the exact formal and LODO matrices"
            )
        if _check_bound_hash(info) and self.report_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("reportHash does not match pretraining acceptance payload")
        return self


class GfmTaskAcceptanceManifest(GfmContractModel):
    """Immutable, non-promotable acceptance for one deferred product task.

    This contract deliberately does not satisfy ``GfmAcceptanceManifest`` and
    cannot be referenced by the model promotion/export tables.  Core uses
    it to validate collaboration while the newcomer overlay is deferred.
    """

    schema_version: Literal["gfm.task-acceptance/1.0"] = Field(
        "gfm.task-acceptance/1.0", alias="schemaVersion"
    )
    experiment_id: str = Field(alias="experimentId", min_length=1, max_length=200)
    task_id: Literal["governance.collaboration_recommendation"] = Field(alias="taskId")
    accepted: bool
    architecture_variant: Literal["core-base", "core-moe"] = Field(
        alias="architectureVariant"
    )
    formal_seeds: tuple[int, ...] = Field(alias="formalSeeds")
    run_ids: tuple[str, ...] = Field(alias="runIds")
    checkpoint_ids: tuple[str, ...] = Field(alias="checkpointIds")
    backbone_checkpoint_ids: tuple[str, ...] = Field(alias="backboneCheckpointIds")
    backbone_state_hashes: tuple[str, ...] = Field(alias="backboneStateHashes")
    pretraining_acceptance_report_hash: str = Field(
        alias="pretrainingAcceptanceReportHash", pattern=SHA256_PATTERN
    )
    corpus_hashes: tuple[str, ...] = Field(alias="corpusHashes")
    protocol_hash: str = Field(alias="protocolHash", pattern=SHA256_PATTERN)
    config_hash: str = Field(alias="configHash", pattern=SHA256_PATTERN)
    code_hash: str = Field(alias="codeHash", pattern=SHA256_PATTERN)
    environment_hash: str = Field(alias="environmentHash", pattern=SHA256_PATTERN)
    product_report_hashes: tuple[str, ...] = Field(alias="productReportHashes")
    calibration_report_hashes: tuple[str, ...] = Field(alias="calibrationReportHashes")
    fresh_process_report_hashes: tuple[str, ...] = Field(alias="freshProcessReportHashes")
    test_read_evidence_hashes: tuple[str, ...] = Field(alias="testReadEvidenceHashes")
    metric_summary: dict[str, float] = Field(alias="metricSummary")
    maximum_ece: float | None = Field(None, alias="maximumEce", ge=0.0, le=1.0)
    peak_cuda_memory_mib: float | None = Field(
        None, alias="peakCudaMemoryMiB", ge=0.0
    )
    gates: dict[str, bool]
    reasons: tuple[str, ...] = ()
    registrable: Literal[False] = False
    promotable: Literal[False] = False
    exportable: Literal[False] = False
    report_hash: str = Field(alias="reportHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python", by_alias=True, exclude={"report_hash", "created_at"}
        )

    @classmethod
    def create(cls, **values: Any) -> GfmTaskAcceptanceManifest:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="report_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_acceptance(self, info: ValidationInfo) -> GfmTaskAcceptanceManifest:
        _require_aware(self.created_at, "createdAt")
        if len(set(self.formal_seeds)) != len(self.formal_seeds):
            raise ValueError("formalSeeds must contain unique values")
        for values, name in (
            (self.run_ids, "runIds"),
            (self.checkpoint_ids, "checkpointIds"),
            (self.backbone_checkpoint_ids, "backboneCheckpointIds"),
            (self.corpus_hashes, "corpusHashes"),
            (self.product_report_hashes, "productReportHashes"),
            (self.calibration_report_hashes, "calibrationReportHashes"),
            (self.fresh_process_report_hashes, "freshProcessReportHashes"),
            (self.test_read_evidence_hashes, "testReadEvidenceHashes"),
        ):
            _assert_unique(values, name)
        if set(self.gates) != GFM_COLLABORATION_TASK_ACCEPTANCE_GATES:
            raise ValueError(
                "collaboration task acceptance must contain exactly its fixed hard gates"
            )
        if self.accepted != (all(self.gates.values()) and not self.reasons):
            raise ValueError("task accepted must be derived from all hard gates and reasons")
        if self.accepted and (
            self.formal_seeds != (20260821, 20260822, 20260823)
            or len(self.run_ids) != 3
            or len(self.checkpoint_ids) != 3
            or len(self.backbone_checkpoint_ids) != 3
            or len(self.backbone_state_hashes) != 3
            or len(self.product_report_hashes) != 3
            or len(self.calibration_report_hashes) != 3
            or len(self.fresh_process_report_hashes) != 3
            or len(self.test_read_evidence_hashes) != 3
            or len(self.corpus_hashes) != 3
        ):
            raise ValueError(
                "accepted collaboration task requires the exact three-seed evidence matrix"
            )
        if _check_bound_hash(info) and self.report_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("reportHash does not match task acceptance payload")
        return self


class CollaborationRerankComponents(GfmContractModel):
    calibrated_probability: float = Field(alias="calibratedProbability", ge=0.0, le=1.0)
    topic_complementarity: float = Field(alias="topicComplementarity", ge=0.0, le=1.0)
    bridge_gain: float = Field(alias="bridgeGain", ge=0.0, le=1.0)
    institution_diversity: float = Field(alias="institutionDiversity", ge=0.0, le=1.0)

    def weighted_score(self) -> float:
        return (
            0.70 * self.calibrated_probability
            + 0.15 * self.topic_complementarity
            + 0.10 * self.bridge_gain
            + 0.05 * self.institution_diversity
        )


class GovernanceTargetRef(GfmContractModel):
    kind: Literal["node", "candidate_relation", "community_window", "conversation"]
    primary_id: str = Field(alias="primaryId", min_length=1, max_length=1000)
    secondary_id: str | None = Field(None, alias="secondaryId", max_length=1000)

    @model_validator(mode="after")
    def validate_target(self) -> GovernanceTargetRef:
        if self.kind == "candidate_relation" and not self.secondary_id:
            raise ValueError("candidate_relation targets require secondaryId")
        if self.kind != "candidate_relation" and self.secondary_id is not None:
            raise ValueError("secondaryId is reserved for candidate_relation targets")
        return self


class GovernanceEvidenceRef(GfmContractModel):
    kind: Literal["node", "edge", "path", "subgraph", "document", "metric"]
    refs: tuple[str, ...] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=4000)
    observed_at: datetime | None = Field(None, alias="observedAt")

    @model_validator(mode="after")
    def validate_evidence(self) -> GovernanceEvidenceRef:
        _assert_unique(self.refs, "evidence refs")
        _require_aware(self.observed_at, "observedAt")
        _assert_safe_governance_text(self.summary)
        return self


class GovernanceCaseArtifact(GfmContractModel):
    schema_version: Literal["gfm.governance-case/1.0"] = Field(
        "gfm.governance-case/1.0", alias="schemaVersion"
    )
    case_id: str = Field(alias="caseId", min_length=1, max_length=200)
    task_id: GovernanceTaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=1000)
    graph_fact_hash: str = Field(alias="graphFactHash", pattern=SHA256_PATTERN)
    inference_cutoff: datetime = Field(alias="inferenceCutoff")
    model_id: str = Field(alias="modelId", min_length=1, max_length=200)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=100)
    checkpoint_id: str = Field(alias="checkpointId", min_length=1, max_length=200)
    checkpoint_hash: str = Field(alias="checkpointHash", pattern=SHA256_PATTERN)
    run_id: str = Field(alias="runId", min_length=1, max_length=200)
    target: GovernanceTargetRef
    score: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    rerank_components: CollaborationRerankComponents | None = Field(
        None, alias="rerankComponents"
    )
    reason_codes: tuple[str, ...] = Field(alias="reasonCodes", min_length=1)
    evidence: tuple[GovernanceEvidenceRef, ...] = Field(min_length=1)
    counterfactuals: dict[str, float] = Field(default_factory=dict)
    recommended_actions: tuple[str, ...] = Field(alias="recommendedActions", min_length=1)
    data_sufficiency: Literal["sufficient", "insufficient", "refused"] = Field(
        alias="dataSufficiency"
    )
    refusal_reasons: tuple[str, ...] = Field((), alias="refusalReasons")
    human_review_required: Literal[True] = Field(True, alias="humanReviewRequired")
    human_review_status: Literal["pending"] = Field("pending", alias="humanReviewStatus")
    feature_hash: str = Field(alias="featureHash", pattern=SHA256_PATTERN)
    corpus_hashes: tuple[str, ...] = Field(alias="corpusHashes", min_length=1)
    artifact_hash: str = Field(alias="artifactHash", pattern=SHA256_PATTERN)
    created_at: datetime = Field(alias="createdAt")

    def logical_payload(self) -> dict[str, Any]:
        return self.model_dump(
            mode="python", by_alias=True, exclude={"artifact_hash", "created_at"}
        )

    @classmethod
    def create(cls, **values: Any) -> GovernanceCaseArtifact:
        return cls.model_validate(
            _create_hashed_contract(
                cls, values, hash_alias="artifact_hash", timestamped=True
            )
        )

    @model_validator(mode="after")
    def validate_case(self, info: ValidationInfo) -> GovernanceCaseArtifact:
        _require_aware(self.inference_cutoff, "inferenceCutoff")
        _require_aware(self.created_at, "createdAt")
        _assert_unique(self.reason_codes, "reasonCodes")
        _assert_unique(self.corpus_hashes, "corpusHashes")
        _assert_safe_governance_text(self.reason_codes)
        _assert_safe_governance_text(self.recommended_actions)
        _assert_safe_governance_text(self.refusal_reasons)
        collaboration = self.task_id == "governance.collaboration_recommendation"
        if collaboration != (self.rerank_components is not None):
            raise ValueError(
                "collaboration recommendation cases require rerankComponents exclusively"
            )
        if self.rerank_components is not None and abs(
            self.score - self.rerank_components.weighted_score()
        ) > 1e-12:
            raise ValueError("score must use the fixed 70/15/10/5 collaboration rerank")
        has_refusal = self.data_sufficiency == "refused"
        if has_refusal != bool(self.refusal_reasons):
            raise ValueError("refused cases and refusalReasons must agree")
        if _check_bound_hash(info) and self.artifact_hash != canonical_sha256(
            self.logical_payload()
        ):
            raise ValueError("artifactHash does not match governance case payload")
        return self


__all__ = [
    "CollaborationRerankComponents",
    "GFM_ACCEPTANCE_GATES",
    "GFM_PRETRAINING_ACCEPTANCE_GATES",
    "GFM_COLLABORATION_TASK_ACCEPTANCE_GATES",
    "PRODUCT_TASKS",
    "GfmAcceptanceManifest",
    "GfmPretrainingAcceptanceManifest",
    "GfmTaskAcceptanceManifest",
    "GfmCheckpointManifest",
    "GfmContractModel",
    "GfmDomainCorpusManifest",
    "GfmEvaluationReport",
    "GfmPretrainConfig",
    "GfmRunManifest",
    "GfmTaskProtocolManifest",
    "GovernanceCaseArtifact",
    "GovernanceEvidenceRef",
    "GovernanceTargetRef",
]
