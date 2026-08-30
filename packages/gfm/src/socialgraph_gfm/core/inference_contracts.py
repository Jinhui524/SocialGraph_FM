"""Strict, versioned contracts for the isolated core inference boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from socialgraph_gfm.canonical import canonical_sha256
from socialgraph_gfm.core.governance import GovernanceFinding, TaskId

_HASH_PATTERN = r"^[0-9a-f]{64}$"
MAX_INTERNAL_REQUEST_BYTES = 256 * 1024
MAX_INTERNAL_RESPONSE_BYTES = 8 * 1024 * 1024


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
        frozen=True,
        protected_namespaces=("model_dump",),
    )


class CommunityTargetScope(_StrictModel):
    kind: Literal["community"]
    community_ids: tuple[str, ...] = Field(alias="communityIds", min_length=1, strict=False)


class RiskTargetScope(_StrictModel):
    kind: Literal["risk-review"]
    node_ids: tuple[str, ...] = Field(alias="nodeIds", strict=False)
    edge_ids: tuple[str, ...] = Field(alias="edgeIds", strict=False)

    @model_validator(mode="after")
    def validate_nonempty(self):
        if bool(self.node_ids) == bool(self.edge_ids):
            raise ValueError("risk-review scope requires exactly one of nodeIds or edgeIds")
        return self


class CollaborationTargetScope(_StrictModel):
    kind: Literal["node-pairs"]
    pairs: tuple[Annotated[tuple[str, str], Field(strict=False)], ...] = Field(
        min_length=1, strict=False
    )

    @model_validator(mode="after")
    def validate_pairs(self):
        if any(source == target for source, target in self.pairs):
            raise ValueError("collaboration scope forbids self pairs")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("collaboration scope pairs must be unique")
        return self


TargetScope = Annotated[
    CommunityTargetScope | RiskTargetScope | CollaborationTargetScope,
    Field(discriminator="kind"),
]


class CommunityParameters(_StrictModel):
    kind: Literal["community-resilience"]
    top_k_similar_cases: int = Field(alias="topKSimilarCases", ge=0, le=20)


class RiskParameters(_StrictModel):
    kind: Literal["risk-and-trust"]
    top_k_similar_cases: int = Field(alias="topKSimilarCases", ge=0, le=20)


class CollaborationParameters(_StrictModel):
    kind: Literal["collaboration-completion"]
    top_k_similar_cases: int = Field(alias="topKSimilarCases", ge=0, le=20)
    candidate_limit: int = Field(alias="candidateLimit", ge=1, le=10_000)


RunParameters = Annotated[
    CommunityParameters | RiskParameters | CollaborationParameters,
    Field(discriminator="kind"),
]


class GfmRunRequest(_StrictModel):
    """Public v2 request: callers can select scope, never supply model facts."""

    schema_version: Literal["socialgraph-fm.core-run-request/2.0"] = Field(alias="schemaVersion")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    task_id: TaskId = Field(alias="taskId")
    target_scope: TargetScope = Field(alias="targetScope")
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    parameters: RunParameters

    @model_validator(mode="after")
    def validate_task_specific_records(self):
        expected = {
            "core.community_resilience_review": ("community", "community-resilience"),
            "core.risk_and_trust_review": ("risk-review", "risk-and-trust"),
            "core.collaboration_completion": (
                "node-pairs",
                "collaboration-completion",
            ),
        }
        if (self.target_scope.kind, self.parameters.kind) != expected[self.task_id]:
            raise ValueError("targetScope and parameters must match taskId")
        return self


class AuthorizedGraphReference(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-authorized-graph-reference/2.1"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=_HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=_HASH_PATTERN)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=_HASH_PATTERN)
    graph_schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="graphSchemaVersion"
    )
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)


class ServingControlExpectation(_StrictModel):
    control_hash: str = Field(alias="controlHash", pattern=_HASH_PATTERN)
    control_generation: int = Field(alias="controlGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=_HASH_PATTERN)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    catalog_hash: str = Field(alias="catalogHash", pattern=_HASH_PATTERN)
    catalog_generation: int = Field(alias="catalogGeneration", ge=0)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)


class InternalCreateRunRequest(_StrictModel):
    """The only request type admitted at the production create boundary."""

    schema_version: Literal["socialgraph-fm.core-internal-create-run/2.1"] = Field(alias="schemaVersion")
    request: GfmRunRequest
    graph_reference: AuthorizedGraphReference = Field(alias="graphReference")
    expected_serving_control: ServingControlExpectation = Field(alias="expectedServingControl")

    @model_validator(mode="after")
    def validate_graph_identity(self):
        if self.request.graph_version_id != self.graph_reference.graph_version_id:
            raise ValueError("request graphVersionId does not match authorized graph reference")
        return self

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class _EarlyHistoricalInternalCreateRunRequest(_StrictModel):
    """Exact FixRound1/2 persisted request shape, never HTTP input."""

    schema_version: Literal["socialgraph-fm.core-internal-create-run/2.0"] = Field(alias="schemaVersion")
    request: GfmRunRequest
    graph_reference: AuthorizedGraphReference = Field(alias="graphReference")

    @model_validator(mode="after")
    def validate_graph_identity(self):
        if self.request.graph_version_id != self.graph_reference.graph_version_id:
            raise ValueError("request graphVersionId does not match authorized graph reference")
        return self

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class _HistoricalInternalCreateRunRequest(_EarlyHistoricalInternalCreateRunRequest):
    """Exact FixRound3 uncontrolled persisted request shape, never HTTP input."""

    expected_serving_control: Literal[None] = Field(alias="expectedServingControl")


_PersistedCreateRunRequest = (
    InternalCreateRunRequest
    | _HistoricalInternalCreateRunRequest
    | _EarlyHistoricalInternalCreateRunRequest
)


def _decode_persisted_create_run_request(
    payload: bytes,
) -> _PersistedCreateRunRequest:
    """Decode a durable legacy record without expanding the create HTTP schema."""

    try:
        return InternalCreateRunRequest.model_validate_json(payload)
    except ValidationError:
        try:
            return _HistoricalInternalCreateRunRequest.model_validate_json(payload)
        except ValidationError:
            return _EarlyHistoricalInternalCreateRunRequest.model_validate_json(payload)


RunStatusValue = Literal["queued", "running", "succeeded", "failed"]


class GfmRunStatus(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-status/2.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=_HASH_PATTERN)
    status: RunStatusValue
    progress: int = Field(ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    error_code: str | None = Field(alias="errorCode", default=None, max_length=100)
    state_hash: str = Field(alias="stateHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_state(self):
        expected_progress = {"queued": 0, "running": 10, "succeeded": 100, "failed": 100}
        if self.progress != expected_progress[self.status]:
            raise ValueError("run progress does not match status")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("only failed runs carry an error code")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"state_hash"})
        )
        if self.state_hash != expected:
            raise ValueError("state hash does not match persisted run state")
        return self


class TaskEntityCapability(_StrictModel):
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(alias="entityType")
    confidence_kind: Literal["binary-calibration", "regression-interval"] = Field(
        alias="confidenceKind"
    )
    calibration_version: str = Field(alias="calibrationVersion", min_length=1, max_length=300)
    method: Literal["sigmoid", "validation-residual-interval"]
    calibration_artifact_hash: str = Field(alias="calibrationArtifactHash", pattern=_HASH_PATTERN)
    calibration_protocol_hash: str = Field(alias="calibrationProtocolHash", pattern=_HASH_PATTERN)
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH_PATTERN)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH_PATTERN)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_task_entity(self):
        expected_entities = {
            "core.community_resilience_review": {"community"},
            "core.risk_and_trust_review": {"node", "edge"},
            "core.collaboration_completion": {"node-pair"},
        }
        if self.entity_type not in expected_entities[self.task_id]:
            raise ValueError("capability task/entity pairing is not declared")
        expected_confidence = (
            ("regression-interval", "validation-residual-interval")
            if self.entity_type == "community"
            else ("binary-calibration", "sigmoid")
        )
        if (self.confidence_kind, self.method) != expected_confidence:
            raise ValueError("capability confidence kind does not match output entity")
        return self


class ModelCapability(_StrictModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    state: Literal["accepted", "servingReady"]
    tasks: tuple[TaskId, ...] = Field(strict=False)
    graph_schema_versions: tuple[str, ...] = Field(alias="graphSchemaVersions", strict=False)
    graph_feature_contract_hash: str = Field(
        alias="graphFeatureContractHash", pattern=_HASH_PATTERN
    )
    task_bindings: tuple[TaskEntityCapability, ...] = Field(
        alias="taskBindings", min_length=1, strict=False
    )
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)

    @model_validator(mode="after")
    def validate_task_bindings(self):
        observed = tuple((item.task_id, item.entity_type) for item in self.task_bindings)
        expected = tuple(
            pair
            for pair in (
                ("core.community_resilience_review", "community"),
                ("core.risk_and_trust_review", "node"),
                ("core.risk_and_trust_review", "edge"),
                ("core.collaboration_completion", "node-pair"),
            )
            if pair[0] in self.tasks
        )
        if observed != expected:
            raise ValueError(
                "capability task bindings must expose the exact ordered entity inventory"
            )
        feature_inventory = [
            {
                "taskId": item.task_id,
                "entityType": item.entity_type,
                "featureContractHash": item.feature_contract_hash,
            }
            for item in self.task_bindings
        ]
        if self.graph_feature_contract_hash != canonical_sha256(feature_inventory):
            raise ValueError("capability aggregate feature hash differs from task bindings")
        return self


class Readiness(_StrictModel):
    model_validated: bool = Field(alias="modelValidated")
    core_serving_ready: bool = Field(alias="coreServingReady")


class GfmCapabilities(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-capabilities/2.0"] = Field(
        alias="schemaVersion"
    )
    registry_hash: str = Field(alias="registryHash", pattern=_HASH_PATTERN)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    control_hash: str | None = Field(default=None, alias="controlHash", pattern=_HASH_PATTERN)
    control_generation: int | None = Field(default=None, alias="controlGeneration", ge=0)
    catalog_hash: str | None = Field(default=None, alias="catalogHash", pattern=_HASH_PATTERN)
    catalog_generation: int | None = Field(default=None, alias="catalogGeneration", ge=0)
    serving_ready: bool = Field(alias="servingReady")
    models: tuple[ModelCapability, ...] = Field(strict=False)
    tasks: tuple[TaskId, ...] = Field(strict=False)
    readiness: Readiness

    @model_validator(mode="after")
    def validate_registry_derivation(self):
        serving = tuple(model for model in self.models if model.state == "servingReady")
        if self.serving_ready != bool(serving):
            raise ValueError("servingReady must derive from registry models")
        if self.readiness.model_validated != bool(self.models):
            raise ValueError("modelValidated must derive from registry models")
        if self.readiness.core_serving_ready != bool(serving):
            raise ValueError("coreServingReady must derive from registry models")
        if set(self.tasks) != {task for model in self.models for task in model.tasks}:
            raise ValueError("tasks must derive from registry models")
        return self


class ErrorBody(_StrictModel):
    code: str = Field(pattern=r"^[A-Z0-9_]{1,100}$")


class InternalErrorEnvelope(_StrictModel):
    error: ErrorBody


class GfmRunResult(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-result/2.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=_HASH_PATTERN)
    task_id: TaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    findings: tuple[GovernanceFinding, ...] = Field(strict=False)
    completed_at: datetime = Field(alias="completedAt")
    result_hash: str = Field(alias="resultHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_bindings_and_hash(self):
        for finding in self.findings:
            if (
                finding.task_id != self.task_id
                or finding.graph_version_hash != self.graph_version_hash
                or finding.model_version != self.model_version_id
                or finding.model_version_hash != self.model_version_hash
            ):
                raise ValueError("finding identity does not match run result")
            if finding.review_status != "pending-human-review":
                raise ValueError("all findings must remain pending human review")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"result_hash"})
        )
        if self.result_hash != expected:
            raise ValueError("result hash does not match immutable run result")
        return self


class LeaseCalibrationIdentity(_StrictModel):
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(alias="entityType")
    calibration_version: str = Field(alias="calibrationVersion", min_length=1, max_length=300)
    method: Literal["sigmoid", "validation-residual-interval"]
    calibration_artifact_hash: str = Field(alias="calibrationArtifactHash", pattern=_HASH_PATTERN)
    calibration_protocol_hash: str = Field(alias="calibrationProtocolHash", pattern=_HASH_PATTERN)
    confidence_kind: Literal["binary-calibration", "regression-interval"] = Field(
        alias="confidenceKind"
    )
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH_PATTERN)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=_HASH_PATTERN)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH_PATTERN)
    sha256: str = Field(pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_confidence_kind(self):
        expected = (
            ("regression-interval", "validation-residual-interval")
            if self.entity_type == "community"
            else ("binary-calibration", "sigmoid")
        )
        if (self.confidence_kind, self.method) != expected:
            raise ValueError("lease confidence kind and method do not match output entity")
        return self


_EXPECTED_CALIBRATION_ENTITIES = {
    "core.community_resilience_review": {"community"},
    "core.risk_and_trust_review": {"node", "edge"},
    "core.collaboration_completion": {"node-pair"},
}


class RunExecutionSnapshot(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-execution-snapshot/2.2"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=_HASH_PATTERN)
    control_source_sha256: str = Field(alias="controlSourceSha256", pattern=_HASH_PATTERN)
    control_hash: str = Field(alias="controlHash", pattern=_HASH_PATTERN)
    control_generation: int = Field(alias="controlGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=_HASH_PATTERN)
    registry_source_sha256: str = Field(alias="registrySourceSha256", pattern=_HASH_PATTERN)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    checkpoint_sha256: str = Field(alias="checkpointSha256", pattern=_HASH_PATTERN)
    serving_manifest_sha256: str = Field(alias="servingManifestSha256", pattern=_HASH_PATTERN)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH_PATTERN)
    calibration_identities: tuple[LeaseCalibrationIdentity, ...] = Field(
        alias="calibrationIdentities", min_length=1, strict=False
    )
    calibration_set_hash: str = Field(alias="calibrationSetHash", pattern=_HASH_PATTERN)
    task_id: TaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=_HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=_HASH_PATTERN)
    artifact_catalog_sha256: str = Field(alias="artifactCatalogSha256", pattern=_HASH_PATTERN)
    artifact_catalog_hash: str = Field(alias="artifactCatalogHash", pattern=_HASH_PATTERN)
    artifact_catalog_generation: int = Field(alias="artifactCatalogGeneration", ge=0)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=_HASH_PATTERN)
    graph_schema_version: str = Field(alias="graphSchemaVersion", min_length=1, max_length=200)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    created_at: datetime = Field(alias="createdAt")
    snapshot_hash: str = Field(alias="snapshotHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        entities = tuple(identity.entity_type for identity in self.calibration_identities)
        if (
            entities != tuple(sorted(entities))
            or len(entities) != len(set(entities))
            or set(entities) != _EXPECTED_CALIBRATION_ENTITIES[self.task_id]
        ):
            raise ValueError("lease calibrations must be sorted and bind every task output exactly")
        calibration_payload = [
            identity.model_dump(mode="python", by_alias=True)
            for identity in self.calibration_identities
        ]
        if self.calibration_set_hash != canonical_sha256(calibration_payload):
            raise ValueError("calibration set hash mismatch")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"snapshot_hash"})
        )
        if self.snapshot_hash != expected:
            raise ValueError("execution snapshot hash mismatch")
        return self


class _HistoricalRunExecutionSnapshotRound1(_StrictModel):
    """Exact FixRound1 snapshot; private durable-state compatibility only."""

    schema_version: Literal[
        "socialgraph-fm.core-run-execution-snapshot/2.0",
        "socialgraph-fm.core-run-execution-snapshot/2.1",
    ] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=_HASH_PATTERN)
    registry_hash: str = Field(alias="registryHash", pattern=_HASH_PATTERN)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    checkpoint_sha256: str = Field(alias="checkpointSha256", pattern=_HASH_PATTERN)
    serving_manifest_sha256: str = Field(alias="servingManifestSha256", pattern=_HASH_PATTERN)
    task_id: TaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=_HASH_PATTERN)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=_HASH_PATTERN)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=_HASH_PATTERN)
    graph_schema_version: str = Field(alias="graphSchemaVersion", min_length=1, max_length=200)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=_HASH_PATTERN)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    created_at: datetime = Field(alias="createdAt")
    snapshot_hash: str = Field(alias="snapshotHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_historical_hash(self):
        if type(self) in {
            _HistoricalRunExecutionSnapshotRound1,
            _HistoricalRunExecutionSnapshotRound2,
        } and not self.schema_version.endswith("/2.0"):
            raise ValueError("early historical snapshot must use schema version 2.0")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"snapshot_hash"})
        )
        if self.snapshot_hash != expected:
            raise ValueError("historical execution snapshot hash mismatch")
        return self


class _HistoricalRunExecutionSnapshotRound2(_HistoricalRunExecutionSnapshotRound1):
    """Exact FixRound2 snapshot; private durable-state compatibility only."""

    registry_source_sha256: str = Field(alias="registrySourceSha256", pattern=_HASH_PATTERN)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=_HASH_PATTERN)
    calibration_set_hash: str = Field(alias="calibrationSetHash", pattern=_HASH_PATTERN)
    artifact_catalog_sha256: str = Field(alias="artifactCatalogSha256", pattern=_HASH_PATTERN)
    artifact_catalog_generation: int = Field(alias="artifactCatalogGeneration", ge=0)


class _HistoricalRunExecutionSnapshot(_HistoricalRunExecutionSnapshotRound2):
    """Exact FixRound3 snapshot; private durable-state compatibility only."""

    control_hash: str | None = Field(alias="controlHash", pattern=_HASH_PATTERN)
    control_generation: int | None = Field(alias="controlGeneration", ge=0)
    artifact_catalog_hash: str | None = Field(alias="artifactCatalogHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_historical_binding(self):
        is_uncontrolled = self.schema_version.endswith("/2.0")
        control_values = (
            self.control_hash,
            self.control_generation,
            self.artifact_catalog_hash,
        )
        if is_uncontrolled != all(value is None for value in control_values):
            raise ValueError("historical snapshot control binding is inconsistent")
        if not is_uncontrolled and any(value is None for value in control_values):
            raise ValueError("historical snapshot control binding is incomplete")
        return self


_PersistedRunExecutionSnapshot = (
    RunExecutionSnapshot
    | _HistoricalRunExecutionSnapshot
    | _HistoricalRunExecutionSnapshotRound2
    | _HistoricalRunExecutionSnapshotRound1
)


def _decode_persisted_execution_snapshot(
    payload: bytes,
) -> _PersistedRunExecutionSnapshot:
    """Decode current or exact historical durable state, never a network receipt."""

    try:
        return RunExecutionSnapshot.model_validate_json(payload)
    except ValidationError:
        try:
            return _HistoricalRunExecutionSnapshot.model_validate_json(payload)
        except ValidationError:
            try:
                return _HistoricalRunExecutionSnapshotRound2.model_validate_json(payload)
            except ValidationError:
                return _HistoricalRunExecutionSnapshotRound1.model_validate_json(payload)


def _lease_identity_projection(snapshot: RunExecutionSnapshot) -> dict[str, object]:
    payload = snapshot.model_dump(
        mode="python", by_alias=True, exclude={"snapshot_hash", "schema_version"}
    )
    return {
        "schemaVersion": "socialgraph-fm.core-run-lease-identity/2.2",
        **payload,
    }


class InternalCreateRunReceipt(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-internal-create-run-receipt/2.0"] = Field(
        alias="schemaVersion"
    )
    status: GfmRunStatus
    execution_snapshot: RunExecutionSnapshot = Field(alias="executionSnapshot")
    lease_identity_hash: str = Field(alias="leaseIdentityHash", pattern=_HASH_PATTERN)
    receipt_hash: str = Field(alias="receiptHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_receipt(self):
        snapshot = self.execution_snapshot
        if (
            self.status.run_id != snapshot.run_id
            or self.status.request_hash != snapshot.request_hash
            or self.status.created_at != snapshot.created_at
        ):
            raise ValueError("create receipt status does not match execution snapshot")
        if self.lease_identity_hash != canonical_sha256(_lease_identity_projection(snapshot)):
            raise ValueError("lease identity hash mismatch")
        if self.receipt_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"receipt_hash"})
        ):
            raise ValueError("create receipt hash mismatch")
        return self


class RunSuccessMarker(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-success-marker/2.0"] = Field(alias="schemaVersion")
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=_HASH_PATTERN)
    snapshot_hash: str = Field(alias="snapshotHash", pattern=_HASH_PATTERN)
    result_hash: str = Field(alias="resultHash", pattern=_HASH_PATTERN)
    completed_at: datetime = Field(alias="completedAt")
    marker_hash: str = Field(alias="markerHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"marker_hash"})
        )
        if self.marker_hash != expected:
            raise ValueError("success marker hash mismatch")
        return self


__all__ = [
    "AuthorizedGraphReference",
    "GfmCapabilities",
    "GfmRunRequest",
    "GfmRunResult",
    "GfmRunStatus",
    "InternalCreateRunRequest",
    "InternalCreateRunReceipt",
    "InternalErrorEnvelope",
    "LeaseCalibrationIdentity",
    "MAX_INTERNAL_REQUEST_BYTES",
    "MAX_INTERNAL_RESPONSE_BYTES",
    "ModelCapability",
    "RunExecutionSnapshot",
    "ServingControlExpectation",
    "TaskEntityCapability",
    "RunSuccessMarker",
]
