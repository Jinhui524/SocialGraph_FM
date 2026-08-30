"""Strict API-side mirror of the language-neutral core HTTP contracts."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .gfm_hashing import canonical_json, canonical_sha256

TaskId = Literal[
    "core.community_resilience_review",
    "core.risk_and_trust_review",
    "core.collaboration_completion",
]
HASH = r"^[0-9a-f]{64}$"


def _safe_metadata_relative_path(value: str, message: str) -> str:
    parsed = PurePosixPath(value.replace("\\", "/"))
    if (
        not parsed.parts
        or parsed.is_absolute()
        or ".." in parsed.parts
        or ":" in value
    ):
        raise ValueError(message)
    return parsed.as_posix()


MAX_INTERNAL_REQUEST_BYTES = 256 * 1024
MAX_INTERNAL_RESPONSE_BYTES = 8 * 1024 * 1024
MANUAL_REVIEW = (
    "Manual human review is required; no automatic sanction or action is authorized."
)
NON_CAUSAL = "This finding is non-causal and does not predict future events."
REGRESSION_INTERVAL = (
    "The resilience interval reports validation residual coverage, not a probability."
)
ALLOWED_LIMITATIONS = frozenset(
    {
        MANUAL_REVIEW,
        NON_CAUSAL,
        "Directed edges are analyzed on a weak undirected projection.",
        "Static topology only; edge direction over time is not represented.",
        "The score is a registered model output, not a graph fact or decision.",
        "Support/opposition semantics require contextual human review.",
        "Candidate for review; it is not a risk or trust truth label.",
        "Common-neighbor evidence describes only the registered static graph.",
        "Path evidence is static relation-completion context, not a future-event forecast.",
        "Static relation-completion recommendation only.",
        "Directed structural context uses a weak undirected projection.",
        "Connectivity evidence is factual topology context, not a community health label.",
        REGRESSION_INTERVAL,
    }
)


def _closed_limitations(value: tuple[str, ...]) -> tuple[str, ...]:
    if any(item not in ALLOWED_LIMITATIONS for item in value):
        raise ValueError("limitations must use the closed canonical vocabulary")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        populate_by_name=True,
        frozen=True,
        protected_namespaces=("model_dump",),
    )


class CommunityTargetScope(StrictModel):
    kind: Literal["community"]
    community_ids: tuple[str, ...] = Field(
        alias="communityIds", min_length=1, strict=False
    )


class RiskTargetScope(StrictModel):
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {
                    "properties": {
                        "edgeIds": {"maxItems": 0},
                        "nodeIds": {"minItems": 1},
                    }
                },
                {
                    "properties": {
                        "edgeIds": {"minItems": 1},
                        "nodeIds": {"maxItems": 0},
                    }
                },
            ]
        }
    )

    kind: Literal["risk-review"]
    node_ids: tuple[str, ...] = Field(alias="nodeIds", strict=False)
    edge_ids: tuple[str, ...] = Field(alias="edgeIds", strict=False)

    @model_validator(mode="after")
    def nonempty(self):
        if bool(self.node_ids) == bool(self.edge_ids):
            raise ValueError(
                "risk-review scope requires exactly one of nodeIds or edgeIds"
            )
        return self


class CollaborationTargetScope(StrictModel):
    kind: Literal["node-pairs"]
    pairs: tuple[Annotated[tuple[str, str], Field(strict=False)], ...] = Field(
        min_length=1, strict=False
    )

    @model_validator(mode="after")
    def valid_pairs(self):
        if any(source == target for source, target in self.pairs) or len(
            set(self.pairs)
        ) != len(self.pairs):
            raise ValueError("node pairs must be unique non-self pairs")
        return self


TargetScope = Annotated[
    CommunityTargetScope | RiskTargetScope | CollaborationTargetScope,
    Field(discriminator="kind"),
]


class CommunityParameters(StrictModel):
    kind: Literal["community-resilience"]
    top_k_similar_cases: int = Field(alias="topKSimilarCases", ge=0, le=20)


class RiskParameters(StrictModel):
    kind: Literal["risk-and-trust"]
    top_k_similar_cases: int = Field(alias="topKSimilarCases", ge=0, le=20)


class CollaborationParameters(StrictModel):
    kind: Literal["collaboration-completion"]
    top_k_similar_cases: int = Field(alias="topKSimilarCases", ge=0, le=20)
    candidate_limit: int = Field(alias="candidateLimit", ge=1, le=10_000)


RunParameters = Annotated[
    CommunityParameters | RiskParameters | CollaborationParameters,
    Field(discriminator="kind"),
]


class CoreRunRequest(StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-request/2.0"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    task_id: TaskId = Field(alias="taskId")
    target_scope: TargetScope = Field(alias="targetScope")
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    parameters: RunParameters

    @model_validator(mode="after")
    def task_specific(self):
        expected = {
            "core.community_resilience_review": (
                "community",
                "community-resilience",
            ),
            "core.risk_and_trust_review": ("risk-review", "risk-and-trust"),
            "core.collaboration_completion": (
                "node-pairs",
                "collaboration-completion",
            ),
        }
        if (self.target_scope.kind, self.parameters.kind) != expected[self.task_id]:
            raise ValueError("targetScope and parameters must match taskId")
        return self


class CoreAuthorizedGraphReference(StrictModel):
    schema_version: Literal["socialgraph-fm.core-authorized-graph-reference/2.1"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=HASH)
    graph_schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="graphSchemaVersion"
    )
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=HASH)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)


class CoreServingControlExpectation(StrictModel):
    control_hash: str = Field(alias="controlHash", pattern=HASH)
    control_generation: int = Field(alias="controlGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=HASH)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    catalog_hash: str = Field(alias="catalogHash", pattern=HASH)
    catalog_generation: int = Field(alias="catalogGeneration", ge=0)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)


class CoreInternalCreateRunRequest(StrictModel):
    schema_version: Literal["socialgraph-fm.core-internal-create-run/2.1"] = Field(
        alias="schemaVersion"
    )
    request: CoreRunRequest
    graph_reference: CoreAuthorizedGraphReference = Field(alias="graphReference")
    expected_serving_control: CoreServingControlExpectation = Field(
        alias="expectedServingControl"
    )

    @model_validator(mode="after")
    def validate_graph_identity(self):
        if self.request.graph_version_id != self.graph_reference.graph_version_id:
            raise ValueError(
                "request graphVersionId does not match authorized graph reference"
            )
        return self

    @property
    def request_hash(self) -> str:
        return canonical_sha256(self.model_dump(mode="python", by_alias=True))


class CoreServingFeatureField(StrictModel):
    kind: Literal["numeric", "categorical", "multiHot"]
    name: str = Field(min_length=1, max_length=200)


class CoreServingFeatureContract(StrictModel):
    schema_version: Literal["socialgraph-fm.core-graph-feature-contract/2.0"] = Field(
        alias="schemaVersion"
    )
    node_features: tuple[CoreServingFeatureField, ...] = Field(
        alias="nodeFeatures", strict=False
    )
    structural_feature_names: tuple[str, ...] = Field(
        alias="structuralFeatureNames", strict=False
    )

    @model_validator(mode="after")
    def validate_unique_names(self):
        names = [field.name for field in self.node_features]
        if len(names) != len(set(names)) or len(self.structural_feature_names) != len(
            set(self.structural_feature_names)
        ):
            raise ValueError("feature contract names must be unique")
        return self


class CoreServingGraphEntry(StrictModel):
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=HASH)
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    graph_schema_version: Literal["socialgraph-fm.core-graph-bundle/2.0"] = Field(
        alias="graphSchemaVersion"
    )
    feature_contract: CoreServingFeatureContract = Field(alias="featureContract")
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=HASH)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)

    @model_validator(mode="after")
    def validate_entry(self):
        _safe_metadata_relative_path(
            self.relative_path, "serving bundle path must be safe and relative"
        )
        if self.feature_contract_hash != canonical_sha256(
            self.feature_contract.model_dump(mode="python", by_alias=True)
        ):
            raise ValueError("feature contract hash mismatch")
        return self


class CoreServingGraphCatalog(StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-graph-catalog/1.0"] = (
        Field(alias="schemaVersion")
    )
    generation: int = Field(ge=0)
    artifacts: tuple[CoreServingGraphEntry, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_unique(self):
        ids = [entry.artifact_id for entry in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("artifactId values must be unique")
        return self


class CoreServingControlReference(StrictModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    sha256: str = Field(pattern=HASH)
    semantic_hash: str = Field(alias="semanticHash", pattern=HASH)
    generation: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_metadata_relative_path(
            value, "serving control reference must be safe and relative"
        )


class CoreServingControl(StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-control/1.0"] = Field(
        alias="schemaVersion"
    )
    generation: int = Field(ge=0)
    registry: CoreServingControlReference
    catalog: CoreServingControlReference
    control_hash: str = Field(alias="controlHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.control_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"control_hash"})
        ):
            raise ValueError("serving control hash mismatch")
        return self


class CoreCheckpointBindings(StrictModel):
    config_hash: str = Field(alias="configHash", pattern=HASH)
    data_hash: str = Field(alias="dataHash", pattern=HASH)
    code_hash: str = Field(alias="codeHash", pattern=HASH)
    environment_hash: str = Field(alias="environmentHash", pattern=HASH)


class CoreServingCheckpoint(StrictModel):
    relative_path: str = Field(alias="relativePath", min_length=1, max_length=500)
    sha256: str = Field(pattern=HASH)
    serving_manifest_relative_path: str = Field(
        alias="servingManifestRelativePath", min_length=1, max_length=500
    )
    serving_manifest_sha256: str = Field(alias="servingManifestSha256", pattern=HASH)
    bindings: CoreCheckpointBindings
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    node_classes: int = Field(alias="nodeClasses", ge=1, le=100_000)
    multi_hot_buckets: int = Field(alias="multiHotBuckets", ge=1, le=65_536)

    @field_validator("relative_path", "serving_manifest_relative_path")
    @classmethod
    def safe_paths(cls, value: str) -> str:
        return _safe_metadata_relative_path(
            value, "checkpoint metadata paths must be safe and relative"
        )


class CoreCalibrationBinding(StrictModel):
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(
        alias="entityType"
    )
    calibration_version: str = Field(
        alias="calibrationVersion", min_length=1, max_length=300
    )
    confidence_kind: Literal["binary-calibration", "regression-interval"] = Field(
        alias="confidenceKind"
    )
    calibration_method: Literal["sigmoid", "validation-residual-interval"] = Field(
        alias="calibrationMethod"
    )
    calibration_artifact_hash: str = Field(
        alias="calibrationArtifactHash", pattern=HASH
    )
    calibration_relative_path: str = Field(
        alias="calibrationRelativePath", min_length=1, max_length=500
    )
    calibration_sha256: str = Field(alias="calibrationSha256", pattern=HASH)
    calibration_protocol_hash: str = Field(
        alias="calibrationProtocolHash", pattern=HASH
    )
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=HASH)
    graph_feature_contract_hash: str = Field(
        alias="graphFeatureContractHash", pattern=HASH
    )

    @field_validator("calibration_relative_path")
    @classmethod
    def safe_path(cls, value: str) -> str:
        return _safe_metadata_relative_path(
            value, "calibration metadata path must be safe and relative"
        )

    @model_validator(mode="after")
    def validate_confidence_kind(self):
        expected = (
            ("regression-interval", "validation-residual-interval")
            if self.entity_type == "community"
            else ("binary-calibration", "sigmoid")
        )
        if (self.confidence_kind, self.calibration_method) != expected:
            raise ValueError("confidence kind and method do not match output entity")
        return self


class CoreServingTaskHead(StrictModel):
    task_id: TaskId = Field(alias="taskId")
    kind: Literal["community-resilience", "risk-and-trust", "collaboration-completion"]
    node_output_index: int | None = Field(default=None, alias="nodeOutputIndex", ge=0)
    calibrations: tuple[CoreCalibrationBinding, ...] = Field(min_length=1, strict=False)

    @model_validator(mode="after")
    def validate_task_binding(self):
        expected_kind = {
            "core.community_resilience_review": "community-resilience",
            "core.risk_and_trust_review": "risk-and-trust",
            "core.collaboration_completion": "collaboration-completion",
        }[self.task_id]
        if self.kind != expected_kind:
            raise ValueError("task head kind does not match taskId")
        if (self.task_id == "core.risk_and_trust_review") != (
            self.node_output_index is not None
        ):
            raise ValueError("only the risk task head requires nodeOutputIndex")
        expected_entities = {
            "core.community_resilience_review": ("community",),
            "core.risk_and_trust_review": ("node", "edge"),
            "core.collaboration_completion": ("node-pair",),
        }[self.task_id]
        entities = tuple(binding.entity_type for binding in self.calibrations)
        if entities != expected_entities:
            raise ValueError(
                "task head calibrations must bind each output exactly once"
            )
        if self.task_id == "core.risk_and_trust_review" and self.node_output_index != 1:
            raise ValueError("risk task nodeOutputIndex must bind positive class 1")
        return self


class CoreServingAdapterBinding(StrictModel):
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=HASH)
    multi_hot_buckets: int = Field(alias="multiHotBuckets", ge=1, le=65_536)


class CoreServingCheckpointManifest(StrictModel):
    schema_version: Literal[
        "socialgraph-fm.core-serving-checkpoint-manifest/1.1"
    ] = Field(alias="schemaVersion")
    task4_checkpoint_sha256: str = Field(alias="task4CheckpointSha256", pattern=HASH)
    accepted: Literal[True]
    promotable: Literal[True]
    model_state_hash: str = Field(alias="modelStateHash", pattern=HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=HASH)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    node_classes: int = Field(alias="nodeClasses", ge=1, le=100_000)
    multi_hot_buckets: int = Field(alias="multiHotBuckets", ge=1, le=65_536)
    adapter_bindings: tuple[CoreServingAdapterBinding, ...] = Field(
        alias="adapterBindings", min_length=1, strict=False
    )
    task_heads: tuple[CoreServingTaskHead, ...] = Field(
        alias="taskHeads", min_length=1, strict=False
    )

    @model_validator(mode="after")
    def validate_adapter_inventory(self):
        domains = tuple(item.adapter_domain for item in self.adapter_bindings)
        if domains != tuple(sorted(set(domains))):
            raise ValueError("serving adapter bindings must be unique and sorted")
        primary = self.adapter_bindings[0]
        if (
            self.adapter_domain != primary.adapter_domain
            or self.adapter_schema_hash != primary.adapter_schema_hash
            or self.adapter_state_hash != primary.adapter_state_hash
            or self.multi_hot_buckets != primary.multi_hot_buckets
        ):
            raise ValueError("serving manifest primary adapter differs from its inventory")
        by_domain = {item.adapter_domain: item for item in self.adapter_bindings}
        for head in self.task_heads:
            for entity in head.calibrations:
                adapter = by_domain.get(entity.adapter_domain)
                if (
                    adapter is None
                    or adapter.adapter_schema_hash != entity.adapter_schema_hash
                    or adapter.adapter_state_hash != entity.adapter_state_hash
                ):
                    raise ValueError(
                        "task/entity adapter binding differs from manifest inventory"
                    )
        if set(by_domain) != {
            entity.adapter_domain
            for head in self.task_heads
            for entity in head.calibrations
        }:
            raise ValueError(
                "manifest adapter inventory must be used by a task/entity binding"
            )
        return self


class CoreServingModel(StrictModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    state: Literal["accepted", "servingReady"]
    checkpoint: CoreServingCheckpoint
    task_heads: tuple[CoreServingTaskHead, ...] = Field(
        alias="taskHeads", min_length=1, strict=False
    )
    tasks: tuple[TaskId, ...] = Field(min_length=1, strict=False)
    graph_schema_versions: tuple[
        Literal["socialgraph-fm.core-graph-bundle/2.0"], ...
    ] = Field(alias="graphSchemaVersions", min_length=1, strict=False)
    graph_feature_contract_hash: str = Field(
        alias="graphFeatureContractHash", pattern=HASH
    )
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)

    @model_validator(mode="after")
    def validate_descriptor(self):
        head_tasks = tuple(head.task_id for head in self.task_heads)
        if (
            len(head_tasks) != len(set(head_tasks))
            or len(self.tasks) != len(set(self.tasks))
            or set(head_tasks) != set(self.tasks)
        ):
            raise ValueError("taskHeads must match tasks exactly")
        if len(self.graph_schema_versions) != len(set(self.graph_schema_versions)):
            raise ValueError("graphSchemaVersions must be unique")
        feature_inventory = [
            {
                "taskId": head.task_id,
                "entityType": binding.entity_type,
                "featureContractHash": binding.graph_feature_contract_hash,
            }
            for head in self.task_heads
            for binding in head.calibrations
        ]
        if self.graph_feature_contract_hash != canonical_sha256(feature_inventory):
            raise ValueError(
                "graphFeatureContractHash must bind the ordered task/entity feature inventory"
            )
        projection = self.model_dump(
            mode="python",
            by_alias=True,
            exclude={"model_version_hash", "state"},
        )
        if self.model_version_hash != canonical_sha256(projection):
            raise ValueError("modelVersionHash does not bind the model descriptor")
        return self


class CoreServingRegistry(StrictModel):
    schema_version: Literal["socialgraph-fm.core-serving-registry/2.0"] = Field(
        alias="schemaVersion"
    )
    generation: int = Field(ge=0)
    models: tuple[CoreServingModel, ...] = Field(strict=False)

    @model_validator(mode="after")
    def validate_unique_models(self):
        identifiers = [model.model_version_id for model in self.models]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("registry modelVersionId values must be unique")
        return self


class CoreTaskEntityCapability(StrictModel):
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(
        alias="entityType"
    )
    confidence_kind: Literal["binary-calibration", "regression-interval"] = Field(
        alias="confidenceKind"
    )
    calibration_version: str = Field(
        alias="calibrationVersion", min_length=1, max_length=300
    )
    method: Literal["sigmoid", "validation-residual-interval"]
    calibration_artifact_hash: str = Field(
        alias="calibrationArtifactHash", pattern=HASH
    )
    calibration_protocol_hash: str = Field(
        alias="calibrationProtocolHash", pattern=HASH
    )
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=HASH)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=HASH)

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


class CoreModelCapability(StrictModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    state: Literal["accepted", "servingReady"]
    tasks: tuple[TaskId, ...] = Field(strict=False)
    graph_schema_versions: tuple[str, ...] = Field(
        alias="graphSchemaVersions", strict=False
    )
    graph_feature_contract_hash: str = Field(
        alias="graphFeatureContractHash", pattern=HASH
    )
    task_bindings: tuple[CoreTaskEntityCapability, ...] = Field(
        alias="taskBindings", min_length=1, strict=False
    )
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)

    @model_validator(mode="after")
    def validate_task_bindings(self):
        observed = tuple(
            (item.task_id, item.entity_type) for item in self.task_bindings
        )
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


class CoreReadiness(StrictModel):
    model_validated: bool = Field(alias="modelValidated")
    core_serving_ready: bool = Field(alias="coreServingReady")


class CoreCapabilities(StrictModel):
    schema_version: Literal["socialgraph-fm.core-capabilities/2.0"] = Field(
        alias="schemaVersion"
    )
    registry_hash: str = Field(alias="registryHash", pattern=HASH)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    control_hash: str | None = Field(default=None, alias="controlHash", pattern=HASH)
    control_generation: int | None = Field(
        default=None, alias="controlGeneration", ge=0
    )
    catalog_hash: str | None = Field(default=None, alias="catalogHash", pattern=HASH)
    catalog_generation: int | None = Field(
        default=None, alias="catalogGeneration", ge=0
    )
    serving_ready: bool = Field(alias="servingReady")
    models: tuple[CoreModelCapability, ...] = Field(strict=False)
    tasks: tuple[TaskId, ...] = Field(strict=False)
    readiness: CoreReadiness

    @model_validator(mode="after")
    def registry_derived(self):
        accepted = [
            item for item in self.models if item.state in {"accepted", "servingReady"}
        ]
        serving = [item for item in accepted if item.state == "servingReady"]
        if self.serving_ready != bool(serving):
            raise ValueError("servingReady must derive from registry models")
        if self.readiness.model_validated != bool(accepted):
            raise ValueError("modelValidated must derive from registry models")
        if self.readiness.core_serving_ready != bool(serving):
            raise ValueError("coreServingReady must derive from registry models")
        if set(self.tasks) != {task for item in accepted for task in item.tasks}:
            raise ValueError("tasks must derive from registry models")
        return self


class CoreCapabilitiesResponse(StrictModel):
    schema_version: Literal["socialgraph-fm.core-capabilities/2.0"] = Field(
        alias="schemaVersion"
    )
    registry_hash: str = Field(alias="registryHash", pattern=HASH)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    control_hash: str | None = Field(default=None, alias="controlHash", pattern=HASH)
    control_generation: int | None = Field(
        default=None, alias="controlGeneration", ge=0
    )
    catalog_hash: str | None = Field(default=None, alias="catalogHash", pattern=HASH)
    catalog_generation: int | None = Field(
        default=None, alias="catalogGeneration", ge=0
    )
    serving_ready: bool = Field(alias="servingReady")
    models: tuple[CoreModelCapability, ...] = Field(strict=False)
    tasks: tuple[TaskId, ...] = Field(strict=False)
    readiness: CoreReadiness

    @model_validator(mode="after")
    def registry_derived(self):
        internal = self.model_dump(mode="python", by_alias=True)
        internal["schemaVersion"] = "socialgraph-fm.core-capabilities/2.0"
        CoreCapabilities.model_validate(internal)
        return self


class CoreErrorBody(StrictModel):
    code: str = Field(pattern=r"^[A-Z0-9_]{1,100}$")


class CoreInternalErrorEnvelope(StrictModel):
    error: CoreErrorBody


class CoreRunStatus(StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-status/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = Field(ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    error_code: str | None = Field(alias="errorCode", default=None, max_length=100)
    state_hash: str = Field(alias="stateHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        expected_progress = {
            "queued": 0,
            "running": 10,
            "succeeded": 100,
            "failed": 100,
        }
        if self.progress != expected_progress[self.status]:
            raise ValueError("progress does not match status")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("only failed runs carry errorCode")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"state_hash"})
        )
        if self.state_hash != expected:
            raise ValueError("stateHash mismatch")
        return self


class CoreLeaseCalibrationIdentity(StrictModel):
    entity_type: Literal["community", "node", "edge", "node-pair"] = Field(
        alias="entityType"
    )
    calibration_version: str = Field(
        alias="calibrationVersion", min_length=1, max_length=300
    )
    method: Literal["sigmoid", "validation-residual-interval"]
    calibration_artifact_hash: str = Field(
        alias="calibrationArtifactHash", pattern=HASH
    )
    calibration_protocol_hash: str = Field(
        alias="calibrationProtocolHash", pattern=HASH
    )
    confidence_kind: Literal["binary-calibration", "regression-interval"] = Field(
        alias="confidenceKind"
    )
    adapter_domain: str = Field(alias="adapterDomain", min_length=1, max_length=200)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    adapter_state_hash: str = Field(alias="adapterStateHash", pattern=HASH)
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=HASH)
    sha256: str = Field(pattern=HASH)

    @model_validator(mode="after")
    def validate_confidence_kind(self):
        expected = (
            ("regression-interval", "validation-residual-interval")
            if self.entity_type == "community"
            else ("binary-calibration", "sigmoid")
        )
        if (self.confidence_kind, self.method) != expected:
            raise ValueError(
                "lease confidence kind and method do not match output entity"
            )
        return self


_EXPECTED_CALIBRATION_ENTITIES = {
    "core.community_resilience_review": {"community"},
    "core.risk_and_trust_review": {"node", "edge"},
    "core.collaboration_completion": {"node-pair"},
}


def _validate_lease_calibrations(
    task_id: TaskId,
    identities: tuple[CoreLeaseCalibrationIdentity, ...],
    set_hash: str,
) -> None:
    entities = tuple(identity.entity_type for identity in identities)
    if (
        entities != tuple(sorted(entities))
        or len(entities) != len(set(entities))
        or set(entities) != _EXPECTED_CALIBRATION_ENTITIES[task_id]
    ):
        raise ValueError(
            "lease calibrations must be sorted and bind every task output exactly"
        )
    payload = [identity.model_dump(mode="python", by_alias=True) for identity in identities]
    if set_hash != canonical_sha256(payload):
        raise ValueError("calibration set hash mismatch")


class CoreRunExecutionSnapshot(StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-execution-snapshot/2.2"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    control_source_sha256: str = Field(alias="controlSourceSha256", pattern=HASH)
    control_hash: str = Field(alias="controlHash", pattern=HASH)
    control_generation: int = Field(alias="controlGeneration", ge=0)
    registry_hash: str = Field(alias="registryHash", pattern=HASH)
    registry_source_sha256: str = Field(alias="registrySourceSha256", pattern=HASH)
    registry_generation: int = Field(alias="registryGeneration", ge=0)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    checkpoint_sha256: str = Field(alias="checkpointSha256", pattern=HASH)
    serving_manifest_sha256: str = Field(alias="servingManifestSha256", pattern=HASH)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    calibration_identities: tuple[CoreLeaseCalibrationIdentity, ...] = Field(
        alias="calibrationIdentities", min_length=1, strict=False
    )
    calibration_set_hash: str = Field(alias="calibrationSetHash", pattern=HASH)
    task_id: TaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    source_graph_fact_hash: str = Field(alias="sourceGraphFactHash", pattern=HASH)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=300)
    artifact_hash: str = Field(alias="artifactHash", pattern=HASH)
    artifact_catalog_sha256: str = Field(alias="artifactCatalogSha256", pattern=HASH)
    artifact_catalog_hash: str = Field(alias="artifactCatalogHash", pattern=HASH)
    artifact_catalog_generation: int = Field(alias="artifactCatalogGeneration", ge=0)
    bundle_sha256: str = Field(alias="bundleSha256", pattern=HASH)
    graph_schema_version: str = Field(
        alias="graphSchemaVersion", min_length=1, max_length=200
    )
    feature_contract_hash: str = Field(alias="featureContractHash", pattern=HASH)
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    created_at: datetime = Field(alias="createdAt")
    snapshot_hash: str = Field(alias="snapshotHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        _validate_lease_calibrations(
            self.task_id, self.calibration_identities, self.calibration_set_hash
        )
        if self.snapshot_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"snapshot_hash"})
        ):
            raise ValueError("execution snapshot hash mismatch")
        return self


def _lease_identity_projection(snapshot: CoreRunExecutionSnapshot) -> dict[str, Any]:
    payload = snapshot.model_dump(
        mode="python", by_alias=True, exclude={"snapshot_hash", "schema_version"}
    )
    return {
        "schemaVersion": "socialgraph-fm.core-run-lease-identity/2.2",
        **payload,
    }


class CoreInternalCreateRunReceipt(StrictModel):
    schema_version: Literal["socialgraph-fm.core-internal-create-run-receipt/2.0"] = Field(
        alias="schemaVersion"
    )
    status: CoreRunStatus
    execution_snapshot: CoreRunExecutionSnapshot = Field(alias="executionSnapshot")
    lease_identity_hash: str = Field(alias="leaseIdentityHash", pattern=HASH)
    receipt_hash: str = Field(alias="receiptHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_receipt(self):
        if (
            self.status.run_id != self.execution_snapshot.run_id
            or self.status.request_hash != self.execution_snapshot.request_hash
            or self.status.created_at != self.execution_snapshot.created_at
            or self.lease_identity_hash
            != canonical_sha256(_lease_identity_projection(self.execution_snapshot))
            or self.receipt_hash
            != canonical_sha256(
                self.model_dump(mode="python", by_alias=True, exclude={"receipt_hash"})
            )
        ):
            raise ValueError("internal create receipt binding mismatch")
        return self


class CoreRunExpectation(StrictModel):
    schema_version: Literal["socialgraph-fm.core-api-run-expectation/2.2"] = Field(
        alias="schemaVersion"
    )
    create_request: CoreInternalCreateRunRequest = Field(alias="createRequest")
    control_source_sha256: str = Field(alias="controlSourceSha256", pattern=HASH)
    registry_source_sha256: str = Field(alias="registrySourceSha256", pattern=HASH)
    artifact_catalog_sha256: str = Field(alias="artifactCatalogSha256", pattern=HASH)
    checkpoint_sha256: str = Field(alias="checkpointSha256", pattern=HASH)
    serving_manifest_sha256: str = Field(alias="servingManifestSha256", pattern=HASH)
    adapter_schema_hash: str = Field(alias="adapterSchemaHash", pattern=HASH)
    calibration_identities: tuple[CoreLeaseCalibrationIdentity, ...] = Field(
        alias="calibrationIdentities", min_length=1, strict=False
    )
    calibration_set_hash: str = Field(alias="calibrationSetHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_calibrations(self):
        _validate_lease_calibrations(
            self.create_request.request.task_id,
            self.calibration_identities,
            self.calibration_set_hash,
        )
        return self


def _validate_receipt_expectation(
    receipt: CoreInternalCreateRunReceipt, expectation: CoreRunExpectation
) -> None:
    snapshot = receipt.execution_snapshot
    create = expectation.create_request
    graph = create.graph_reference
    control = create.expected_serving_control
    expected = (
        create.request_hash,
        expectation.control_source_sha256,
        control.control_hash,
        control.control_generation,
        expectation.registry_source_sha256,
        control.registry_hash,
        control.registry_generation,
        expectation.artifact_catalog_sha256,
        control.catalog_hash,
        control.catalog_generation,
        create.request.model_version_id,
        control.model_version_hash,
        expectation.checkpoint_sha256,
        expectation.serving_manifest_sha256,
        expectation.adapter_schema_hash,
        expectation.calibration_identities,
        expectation.calibration_set_hash,
        create.request.task_id,
        graph.graph_version_id,
        graph.source_graph_fact_hash,
        graph.graph_version_hash,
        graph.artifact_id,
        graph.artifact_hash,
        graph.bundle_sha256,
        graph.graph_schema_version,
        graph.feature_contract_hash,
        graph.node_count,
        graph.edge_count,
    )
    observed = (
        snapshot.request_hash,
        snapshot.control_source_sha256,
        snapshot.control_hash,
        snapshot.control_generation,
        snapshot.registry_source_sha256,
        snapshot.registry_hash,
        snapshot.registry_generation,
        snapshot.artifact_catalog_sha256,
        snapshot.artifact_catalog_hash,
        snapshot.artifact_catalog_generation,
        snapshot.model_version_id,
        snapshot.model_version_hash,
        snapshot.checkpoint_sha256,
        snapshot.serving_manifest_sha256,
        snapshot.adapter_schema_hash,
        snapshot.calibration_identities,
        snapshot.calibration_set_hash,
        snapshot.task_id,
        snapshot.graph_version_id,
        snapshot.source_graph_fact_hash,
        snapshot.graph_version_hash,
        snapshot.artifact_id,
        snapshot.artifact_hash,
        snapshot.bundle_sha256,
        snapshot.graph_schema_version,
        snapshot.feature_contract_hash,
        snapshot.node_count,
        snapshot.edge_count,
    )
    if observed != expected:
        raise ValueError("receipt does not match immutable API run expectation")


class CoreRunBinding(StrictModel):
    schema_version: Literal["socialgraph-fm.core-api-run-binding/2.2"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    receipt: CoreInternalCreateRunReceipt
    expectation: CoreRunExpectation
    binding_hash: str = Field(alias="bindingHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.run_id != self.receipt.status.run_id:
            raise ValueError("API run binding run identity mismatch")
        _validate_receipt_expectation(self.receipt, self.expectation)
        if self.binding_hash != canonical_sha256(
            self.model_dump(mode="json", by_alias=True, exclude={"binding_hash"})
        ):
            raise ValueError("API run binding hash mismatch")
        return self


class CoreRunBindingAnchor(StrictModel):
    schema_version: Literal["socialgraph-fm.core-api-run-binding-anchor/1.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    binding_hash: str = Field(alias="bindingHash", pattern=HASH)
    anchor_hash: str = Field(alias="anchorHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.anchor_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"anchor_hash"})
        ):
            raise ValueError("API run binding anchor hash mismatch")
        return self


class CoreRegisteredEdgeIdentity(StrictModel):
    schema_version: Literal["socialgraph-fm.core-edge-identity/2.0"] = Field(
        alias="schemaVersion"
    )
    source_id: str = Field(alias="sourceId", min_length=1, max_length=500)
    target_id: str = Field(alias="targetId", min_length=1, max_length=500)
    edge_type: str = Field(alias="edgeType", min_length=1, max_length=200)
    weight: float
    edge_hash: str = Field(alias="edgeHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_edge(self):
        if not math.isfinite(self.weight) or self.edge_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"edge_hash"})
        ):
            raise ValueError("edgeHash mismatch")
        return self


class CoreModelScore(StrictModel):
    schema_version: Literal["socialgraph-fm.core-model-score/2.0"] = Field(
        alias="schemaVersion"
    )
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["node", "edge", "node-pair", "community"] = Field(
        alias="entityType"
    )
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1, strict=False)
    score: float
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    edge_identity: CoreRegisteredEdgeIdentity | None = Field(
        alias="edgeIdentity", default=None
    )
    score_hash: str = Field(alias="scoreHash", pattern=HASH)

    @field_validator("entity_ids")
    @classmethod
    def valid_entity_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or len(item) > 500 for item in value):
            raise ValueError("entity IDs violate the static bundle contract")
        return value

    @model_validator(mode="after")
    def validate_score(self):
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.entity_type == "edge":
            if self.edge_identity is not None and self.entity_ids != (
                self.edge_identity.source_id,
                self.edge_identity.target_id,
            ):
                raise ValueError("edge identity endpoints mismatch")
        elif self.edge_identity is not None:
            raise ValueError("edgeIdentity is allowed only for edge scores")
        if self.score_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"score_hash"})
        ):
            raise ValueError("scoreHash mismatch")
        return self


class CoreCalibratedConfidence(StrictModel):
    schema_version: Literal["socialgraph-fm.core-calibrated-confidence/2.0"] = Field(
        alias="schemaVersion"
    )
    value: float = Field(ge=0, le=1)
    score_hash: str = Field(alias="scoreHash", pattern=HASH)
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["node", "edge", "node-pair", "community"] = Field(
        alias="entityType"
    )
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1, strict=False)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    calibration_version: str = Field(
        alias="calibrationVersion", min_length=1, max_length=300
    )
    method: str = Field(min_length=1, max_length=200)
    calibration_artifact_hash: str = Field(
        alias="calibrationArtifactHash", pattern=HASH
    )
    calibration_protocol_hash: str = Field(
        alias="calibrationProtocolHash", pattern=HASH
    )
    confidence_hash: str = Field(alias="confidenceHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_confidence(self):
        if not math.isfinite(self.value) or self.confidence_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"confidence_hash"})
        ):
            raise ValueError("confidenceHash mismatch")
        return self


class CoreRegressionConfidenceInterval(StrictModel):
    """Validation-derived uncertainty interval, never a probability."""

    schema_version: Literal["socialgraph-fm.core-regression-confidence-interval/1.0"] = (
        Field(alias="schemaVersion")
    )
    point_estimate: float = Field(alias="pointEstimate")
    lower_bound: float = Field(alias="lowerBound")
    upper_bound: float = Field(alias="upperBound")
    coverage: float = Field(gt=0, lt=1)
    validation_count: int = Field(alias="validationCount", ge=2)
    score_hash: str = Field(alias="scoreHash", pattern=HASH)
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["community"] = Field(alias="entityType")
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1, strict=False)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    confidence_version: str = Field(
        alias="confidenceVersion", min_length=1, max_length=300
    )
    method: Literal["validation-residual-interval"]
    confidence_artifact_hash: str = Field(
        alias="confidenceArtifactHash", pattern=HASH
    )
    confidence_protocol_hash: str = Field(
        alias="confidenceProtocolHash", pattern=HASH
    )
    confidence_hash: str = Field(alias="confidenceHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_interval(self):
        if not all(
            math.isfinite(value)
            for value in (self.point_estimate, self.lower_bound, self.upper_bound)
        ):
            raise ValueError("regression confidence interval must be finite")
        if not self.lower_bound <= self.point_estimate <= self.upper_bound:
            raise ValueError(
                "regression confidence interval must contain the point estimate"
            )
        if self.confidence_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"confidence_hash"})
        ):
            raise ValueError("confidenceHash mismatch")
        return self


class CoreEvidenceItem(StrictModel):
    schema_version: Literal["socialgraph-fm.core-evidence/2.0"] = Field(
        alias="schemaVersion"
    )
    metric: str = Field(min_length=1, max_length=300)
    value_canonical_json: str = Field(alias="valueCanonicalJson", min_length=2)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    source_type: Literal["deterministic-graph-algorithm", "registered-model-output"] = (
        Field(alias="sourceType")
    )
    node_ids: tuple[str, ...] = Field(default=(), alias="nodeIds", strict=False)
    edge_ids: tuple[str, ...] = Field(default=(), alias="edgeIds", strict=False)
    algorithm_config_hash: str | None = Field(
        alias="algorithmConfigHash", default=None, pattern=HASH
    )
    model_version_hash: str | None = Field(
        alias="modelVersionHash", default=None, pattern=HASH
    )
    model_version: str | None = Field(alias="modelVersion", default=None, min_length=1)
    model_score_hash: str | None = Field(
        alias="modelScoreHash", default=None, pattern=HASH
    )
    model_task_id: TaskId | None = Field(alias="modelTaskId", default=None)
    model_entity_type: Literal["node", "edge", "node-pair", "community"] | None = Field(
        alias="modelEntityType", default=None
    )
    model_entity_ids: tuple[str, ...] | None = Field(
        alias="modelEntityIds", default=None, strict=False
    )
    limitations: tuple[str, ...] = Field(strict=False)
    evidence_hash: str = Field(alias="evidenceHash", pattern=HASH)

    _limitations = field_validator("limitations")(_closed_limitations)

    @model_validator(mode="after")
    def validate_evidence(self):
        value = json.loads(self.value_canonical_json)
        if (
            not isinstance(value, dict)
            or canonical_json(value) != self.value_canonical_json
        ):
            raise ValueError("valueCanonicalJson is not canonical")
        deterministic = self.source_type == "deterministic-graph-algorithm"
        model_bindings = (
            self.model_version_hash,
            self.model_version,
            self.model_score_hash,
            self.model_task_id,
            self.model_entity_type,
            self.model_entity_ids,
        )
        if deterministic != (self.algorithm_config_hash is not None):
            raise ValueError("deterministic evidence binding mismatch")
        if deterministic and any(item is not None for item in model_bindings):
            raise ValueError("deterministic evidence cannot carry model identity")
        if not deterministic and any(item is None for item in model_bindings):
            raise ValueError("model evidence requires complete model identity")
        if len(set(self.node_ids)) != len(self.node_ids) or len(
            set(self.edge_ids)
        ) != len(self.edge_ids):
            raise ValueError("evidence IDs must be unique")
        if self.evidence_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        ):
            raise ValueError("evidenceHash mismatch")
        return self


class CoreSimilarCase(StrictModel):
    schema_version: Literal["socialgraph-fm.core-similar-case/2.0"] = Field(
        alias="schemaVersion"
    )
    structural_record_hash: str = Field(alias="structuralRecordHash", pattern=HASH)
    similarity: float = Field(ge=-1, le=1)
    source_graph_version_hash: str = Field(alias="sourceGraphVersionHash", pattern=HASH)
    source_entity_ids: tuple[str, ...] = Field(
        alias="sourceEntityIds", min_length=1, strict=False
    )
    source_kind: Literal["node", "ego", "community"] = Field(alias="sourceKind")
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    representation: Literal["embedding", "motif-signature"]
    query_hash: str = Field(alias="queryHash", pattern=HASH)
    representation_schema: Literal["socialgraph-fm.core-structural-record/2.0"] = Field(
        alias="representationSchema"
    )
    similar_case_hash: str = Field(alias="similarCaseHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_case(self):
        if not math.isfinite(
            self.similarity
        ) or self.similar_case_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"similar_case_hash"})
        ):
            raise ValueError("similarCaseHash mismatch")
        return self


class CoreFinding(StrictModel):
    schema_version: Literal["socialgraph-fm.core-finding/2.0"] = Field(
        alias="schemaVersion"
    )
    task_id: TaskId = Field(alias="taskId")
    finding_type: Literal[
        "community-resilience-candidate",
        "node-risk-candidate",
        "signed-relation-review",
        "core-collaboration-completion",
    ] = Field(alias="findingType")
    subject_ids: tuple[str, ...] = Field(alias="subjectIds", min_length=1, strict=False)
    score: CoreModelScore
    calibrated_confidence: CoreCalibratedConfidence | CoreRegressionConfidenceInterval = Field(
        alias="calibratedConfidence"
    )
    evidence: tuple[CoreEvidenceItem, ...] = Field(min_length=1, strict=False)
    similar_cases: tuple[CoreSimilarCase, ...] = Field(alias="similarCases", strict=False)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version: str = Field(alias="modelVersion", min_length=1)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    limitations: tuple[str, ...] = Field(min_length=2, strict=False)
    review_status: Literal["pending-human-review"] = Field(alias="reviewStatus")
    finding_hash: str = Field(alias="findingHash", pattern=HASH)

    _limitations = field_validator("limitations")(_closed_limitations)

    @model_validator(mode="after")
    def validate_finding(self):
        score = self.score
        confidence = self.calibrated_confidence
        compatibility = {
            (
                "core.community_resilience_review",
                "community-resilience-candidate",
            ): "community",
            ("core.risk_and_trust_review", "node-risk-candidate"): "node",
            ("core.risk_and_trust_review", "signed-relation-review"): "edge",
            (
                "core.collaboration_completion",
                "core-collaboration-completion",
            ): "node-pair",
        }
        if compatibility.get((self.task_id, self.finding_type)) != score.entity_type:
            raise ValueError("task, finding type, and score entity type mismatch")
        if self.subject_ids != score.entity_ids or self.task_id != score.task_id:
            raise ValueError("finding subject/task mismatch")
        if (self.graph_version_hash, self.model_version, self.model_version_hash) != (
            score.graph_version_hash,
            score.model_version,
            score.model_version_hash,
        ):
            raise ValueError("finding model/graph mismatch")
        if (
            confidence.score_hash != score.score_hash
            or confidence.task_id != score.task_id
            or confidence.entity_type != score.entity_type
            or confidence.entity_ids != score.entity_ids
            or confidence.graph_version_hash != score.graph_version_hash
            or confidence.model_version != score.model_version
            or confidence.model_version_hash != score.model_version_hash
        ):
            raise ValueError("calibration identity mismatch")
        if self.task_id == "core.community_resilience_review":
            if not isinstance(
                self.calibrated_confidence, CoreRegressionConfidenceInterval
            ):
                raise ValueError(
                    "community resilience requires a regression confidence interval"
                )
            if self.calibrated_confidence.point_estimate != score.score:
                raise ValueError(
                    "regression confidence point estimate must equal model score"
                )
            if REGRESSION_INTERVAL not in self.limitations:
                raise ValueError(
                    "community resilience must explain interval coverage semantics"
                )
        elif not isinstance(self.calibrated_confidence, CoreCalibratedConfidence):
            raise ValueError("binary governance findings require calibrated confidence")
        if any(
            item.graph_version_hash != self.graph_version_hash for item in self.evidence
        ):
            raise ValueError("evidence graph mismatch")
        for item in self.evidence:
            if item.source_type == "registered-model-output" and (
                item.metric != "registered_model.score-reference"
                or item.value_canonical_json != "{}"
                or item.model_score_hash != score.score_hash
                or item.model_task_id != score.task_id
                or item.model_entity_type != score.entity_type
                or item.model_entity_ids != score.entity_ids
                or item.model_version != score.model_version
                or item.model_version_hash != score.model_version_hash
            ):
                raise ValueError("model evidence is not an exact score reference")
        if any(
            item.model_version != self.model_version
            or item.model_version_hash != self.model_version_hash
            or item.representation_schema != "socialgraph-fm.core-structural-record/2.0"
            for item in self.similar_cases
        ):
            raise ValueError("similar case mismatch")
        if MANUAL_REVIEW not in self.limitations or NON_CAUSAL not in self.limitations:
            raise ValueError("mandatory limitations are missing")
        if self.finding_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"finding_hash"})
        ):
            raise ValueError("findingHash mismatch")
        return self


class CoreRunResult(StrictModel):
    schema_version: Literal["socialgraph-fm.core-run-result/2.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH)
    task_id: TaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH)
    findings: tuple[CoreFinding, ...] = Field(strict=False)
    completed_at: datetime = Field(alias="completedAt")
    result_hash: str = Field(alias="resultHash", pattern=HASH)

    @model_validator(mode="after")
    def validate_result(self):
        if any(
            finding.task_id != self.task_id
            or finding.graph_version_hash != self.graph_version_hash
            or finding.model_version != self.model_version_id
            or finding.model_version_hash != self.model_version_hash
            for finding in self.findings
        ):
            raise ValueError("finding does not match result")
        if self.result_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"result_hash"})
        ):
            raise ValueError("resultHash mismatch")
        return self


__all__ = [
    "MAX_INTERNAL_REQUEST_BYTES",
    "MAX_INTERNAL_RESPONSE_BYTES",
    "CoreCalibrationBinding",
    "CoreCheckpointBindings",
    "CoreRunBinding",
    "CoreRunBindingAnchor",
    "CoreRunExpectation",
    "CoreServingAdapterBinding",
    "CoreServingCheckpoint",
    "CoreServingCheckpointManifest",
    "CoreServingModel",
    "CoreServingRegistry",
    "CoreServingTaskHead",
    "CoreAuthorizedGraphReference",
    "CoreCapabilities",
    "CoreCapabilitiesResponse",
    "CoreRunRequest",
    "CoreRunResult",
    "CoreRunStatus",
    "CoreFinding",
    "CoreInternalCreateRunReceipt",
    "CoreInternalCreateRunRequest",
    "CoreInternalErrorEnvelope",
    "CoreLeaseCalibrationIdentity",
    "CoreModelCapability",
    "CoreRegressionConfidenceInterval",
    "CoreRunExecutionSnapshot",
    "CoreServingControl",
    "CoreServingControlExpectation",
    "CoreServingGraphCatalog",
    "CoreTaskEntityCapability",
]
