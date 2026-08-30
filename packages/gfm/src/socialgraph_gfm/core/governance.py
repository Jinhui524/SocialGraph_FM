"""Deterministic governance evidence and immutable review-candidate findings."""

from __future__ import annotations

import json
import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from socialgraph_gfm.canonical import canonical_json, canonical_sha256

from .bundle import StaticEdge, CoreGraphBundle

if TYPE_CHECKING:
    from .retrieval import StructuralQuery, StructuralSearchResult


TaskId = Literal[
    "core.community_resilience_review",
    "core.risk_and_trust_review",
    "core.collaboration_completion",
]
_HASH_PATTERN = r"^[0-9a-f]{64}$"
MANUAL_REVIEW_LIMITATION = (
    "Manual human review is required; no automatic sanction or action is authorized."
)
NON_CAUSAL_LIMITATION = "This finding is non-causal and does not predict future events."
REGRESSION_INTERVAL_LIMITATION = (
    "The resilience interval reports validation residual coverage, not a probability."
)
ALLOWED_GOVERNANCE_LIMITATIONS = frozenset(
    {
        MANUAL_REVIEW_LIMITATION,
        NON_CAUSAL_LIMITATION,
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
        REGRESSION_INTERVAL_LIMITATION,
    }
)


def _validate_closed_limitations(value: tuple[str, ...]) -> tuple[str, ...]:
    if any(item not in ALLOWED_GOVERNANCE_LIMITATIONS for item in value):
        raise ValueError("limitations must use the closed canonical governance vocabulary")
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class RegisteredEdgeIdentity(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-edge-identity/2.0"] = Field(alias="schemaVersion")
    source_id: str = Field(alias="sourceId", min_length=1, max_length=500)
    target_id: str = Field(alias="targetId", min_length=1, max_length=500)
    edge_type: str = Field(alias="edgeType", min_length=1, max_length=200)
    weight: float
    edge_hash: str = Field(alias="edgeHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_identity(self):
        if not math.isfinite(self.weight):
            raise ValueError("edge identity weight must be finite")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"edge_hash"})
        )
        if self.edge_hash != expected:
            raise ValueError("edgeHash does not match canonical edge identity")
        return self

    @classmethod
    def create(cls, edge: StaticEdge) -> RegisteredEdgeIdentity:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-edge-identity/2.0",
            "sourceId": edge.source_id,
            "targetId": edge.target_id,
            "edgeType": edge.edge_type,
            "weight": edge.weight,
        }
        payload["edgeHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class ModelScore(_StrictModel):
    """A supplied Task-4 output; governance never constructs the numeric score itself."""

    schema_version: Literal["socialgraph-fm.core-model-score/2.0"] = Field(alias="schemaVersion")
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["node", "edge", "node-pair", "community"] = Field(alias="entityType")
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1)
    score: float
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    edge_identity: RegisteredEdgeIdentity | None = Field(default=None, alias="edgeIdentity")
    score_hash: str = Field(alias="scoreHash", pattern=_HASH_PATTERN)

    @field_validator("entity_ids")
    @classmethod
    def validate_bundle_compatible_entity_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or len(item) > 500 for item in value):
            raise ValueError("entity IDs must match the CoreGraphBundle stable-ID contract")
        return value

    @model_validator(mode="after")
    def validate_score_hash(self):
        if not math.isfinite(self.score):
            raise ValueError("model score must be finite")
        if self.entity_type == "edge":
            if self.edge_identity is not None and self.entity_ids != (
                self.edge_identity.source_id,
                self.edge_identity.target_id,
            ):
                raise ValueError("edge identity endpoints must match score entity IDs")
        elif self.edge_identity is not None:
            raise ValueError("edge identity is allowed only for edge scores")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"score_hash"})
        )
        if self.score_hash != expected:
            raise ValueError("scoreHash does not match canonical model score content")
        return self

    @classmethod
    def create(
        cls,
        *,
        task_id: TaskId,
        entity_type: Literal["node", "edge", "node-pair", "community"],
        entity_ids: tuple[str, ...],
        score: float,
        graph_version_hash: str,
        model_version: str,
        model_version_hash: str,
        edge_identity: RegisteredEdgeIdentity | None = None,
    ) -> ModelScore:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-model-score/2.0",
            "taskId": task_id,
            "entityType": entity_type,
            "entityIds": entity_ids,
            "score": score,
            "graphVersionHash": graph_version_hash,
            "modelVersion": model_version,
            "modelVersionHash": model_version_hash,
            "edgeIdentity": edge_identity,
        }
        payload["scoreHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class CalibratedConfidence(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-calibrated-confidence/2.0"] = Field(
        alias="schemaVersion"
    )
    value: float = Field(ge=0.0, le=1.0)
    score_hash: str = Field(alias="scoreHash", pattern=_HASH_PATTERN)
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["node", "edge", "node-pair", "community"] = Field(alias="entityType")
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    calibration_version: str = Field(alias="calibrationVersion", min_length=1, max_length=300)
    method: str = Field(min_length=1, max_length=200)
    calibration_artifact_hash: str = Field(alias="calibrationArtifactHash", pattern=_HASH_PATTERN)
    calibration_protocol_hash: str = Field(alias="calibrationProtocolHash", pattern=_HASH_PATTERN)
    confidence_hash: str = Field(alias="confidenceHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_finite(self):
        if not math.isfinite(self.value):
            raise ValueError("calibrated confidence must be finite")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"confidence_hash"})
        )
        if self.confidence_hash != expected:
            raise ValueError("confidenceHash does not match canonical calibration content")
        return self

    @classmethod
    def create(
        cls,
        *,
        score: ModelScore,
        value: float,
        calibration_version: str,
        method: str,
        calibration_artifact_hash: str,
        calibration_protocol_hash: str,
    ) -> CalibratedConfidence:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-calibrated-confidence/2.0",
            "value": value,
            "scoreHash": score.score_hash,
            "taskId": score.task_id,
            "entityType": score.entity_type,
            "entityIds": score.entity_ids,
            "graphVersionHash": score.graph_version_hash,
            "modelVersion": score.model_version,
            "modelVersionHash": score.model_version_hash,
            "calibrationVersion": calibration_version,
            "method": method,
            "calibrationArtifactHash": calibration_artifact_hash,
            "calibrationProtocolHash": calibration_protocol_hash,
        }
        payload["confidenceHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class RegressionConfidenceInterval(_StrictModel):
    """Validation-derived uncertainty interval for a regression score."""

    schema_version: Literal["socialgraph-fm.core-regression-confidence-interval/1.0"] = Field(
        alias="schemaVersion"
    )
    point_estimate: float = Field(alias="pointEstimate")
    lower_bound: float = Field(alias="lowerBound")
    upper_bound: float = Field(alias="upperBound")
    coverage: float = Field(gt=0.0, lt=1.0)
    validation_count: int = Field(alias="validationCount", ge=2)
    score_hash: str = Field(alias="scoreHash", pattern=_HASH_PATTERN)
    task_id: TaskId = Field(alias="taskId")
    entity_type: Literal["community"] = Field(alias="entityType")
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    confidence_version: str = Field(alias="confidenceVersion", min_length=1, max_length=300)
    method: Literal["validation-residual-interval"]
    confidence_artifact_hash: str = Field(alias="confidenceArtifactHash", pattern=_HASH_PATTERN)
    confidence_protocol_hash: str = Field(alias="confidenceProtocolHash", pattern=_HASH_PATTERN)
    confidence_hash: str = Field(alias="confidenceHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_interval(self):
        if not all(
            math.isfinite(value)
            for value in (self.point_estimate, self.lower_bound, self.upper_bound)
        ):
            raise ValueError("regression confidence interval must be finite")
        if not self.lower_bound <= self.point_estimate <= self.upper_bound:
            raise ValueError("regression confidence interval must contain the point estimate")
        bound_score = ModelScore.create(
            task_id=self.task_id,
            entity_type=self.entity_type,
            entity_ids=self.entity_ids,
            score=self.point_estimate,
            graph_version_hash=self.graph_version_hash,
            model_version=self.model_version,
            model_version_hash=self.model_version_hash,
        )
        if bound_score.score_hash != self.score_hash:
            raise ValueError("pointEstimate does not match the bound scoreHash")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"confidence_hash"})
        )
        if self.confidence_hash != expected:
            raise ValueError("confidenceHash does not match canonical regression interval")
        return self

    @classmethod
    def create(
        cls,
        *,
        score: ModelScore,
        lower_bound: float,
        upper_bound: float,
        coverage: float,
        validation_count: int,
        confidence_version: str,
        method: Literal["validation-residual-interval"],
        confidence_artifact_hash: str,
        confidence_protocol_hash: str,
    ) -> RegressionConfidenceInterval:
        if score.entity_type != "community":
            raise ValueError("regression confidence intervals require a community score")
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-regression-confidence-interval/1.0",
            "pointEstimate": score.score,
            "lowerBound": lower_bound,
            "upperBound": upper_bound,
            "coverage": coverage,
            "validationCount": validation_count,
            "scoreHash": score.score_hash,
            "taskId": score.task_id,
            "entityType": score.entity_type,
            "entityIds": score.entity_ids,
            "graphVersionHash": score.graph_version_hash,
            "modelVersion": score.model_version,
            "modelVersionHash": score.model_version_hash,
            "confidenceVersion": confidence_version,
            "method": method,
            "confidenceArtifactHash": confidence_artifact_hash,
            "confidenceProtocolHash": confidence_protocol_hash,
        }
        payload["confidenceHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


ConfidenceEvidence = CalibratedConfidence | RegressionConfidenceInterval


class EvidenceItem(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-evidence/2.0"] = Field(alias="schemaVersion")
    metric: str = Field(min_length=1, max_length=300)
    value_canonical_json: str = Field(alias="valueCanonicalJson", min_length=2)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    source_type: Literal["deterministic-graph-algorithm", "registered-model-output"] = Field(
        alias="sourceType"
    )
    node_ids: tuple[str, ...] = Field(default=(), alias="nodeIds")
    edge_ids: tuple[str, ...] = Field(default=(), alias="edgeIds")
    algorithm_config_hash: str | None = Field(
        default=None, alias="algorithmConfigHash", pattern=_HASH_PATTERN
    )
    model_version_hash: str | None = Field(
        default=None, alias="modelVersionHash", pattern=_HASH_PATTERN
    )
    model_version: str | None = Field(default=None, alias="modelVersion", min_length=1)
    model_score_hash: str | None = Field(
        default=None, alias="modelScoreHash", pattern=_HASH_PATTERN
    )
    model_task_id: TaskId | None = Field(default=None, alias="modelTaskId")
    model_entity_type: Literal["node", "edge", "node-pair", "community"] | None = Field(
        default=None, alias="modelEntityType"
    )
    model_entity_ids: tuple[str, ...] | None = Field(default=None, alias="modelEntityIds")
    limitations: tuple[str, ...]
    evidence_hash: str = Field(alias="evidenceHash", pattern=_HASH_PATTERN)

    _closed_limitations = field_validator("limitations")(_validate_closed_limitations)

    @model_validator(mode="after")
    def validate_binding_and_hash(self):
        try:
            parsed_value = json.loads(self.value_canonical_json)
        except (TypeError, ValueError) as error:
            raise ValueError("valueCanonicalJson must contain canonical JSON") from error
        if (
            not isinstance(parsed_value, dict)
            or canonical_json(parsed_value) != self.value_canonical_json
        ):
            raise ValueError("valueCanonicalJson must be a canonical JSON object")
        deterministic = self.source_type == "deterministic-graph-algorithm"
        if deterministic != (self.algorithm_config_hash is not None):
            raise ValueError("deterministic evidence requires only algorithmConfigHash")
        model_bindings = (
            self.model_version_hash,
            self.model_version,
            self.model_score_hash,
            self.model_task_id,
            self.model_entity_type,
            self.model_entity_ids,
        )
        if deterministic and any(item is not None for item in model_bindings):
            raise ValueError("deterministic evidence cannot carry model identity")
        if not deterministic and any(item is None for item in model_bindings):
            raise ValueError("model evidence requires complete registered model identity")
        if len(set(self.node_ids)) != len(self.node_ids) or len(set(self.edge_ids)) != len(
            self.edge_ids
        ):
            raise ValueError("evidence node and edge IDs must be unique")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"evidence_hash"})
        )
        if self.evidence_hash != expected:
            raise ValueError("evidenceHash does not match canonical evidence content")
        return self

    @property
    def value(self) -> dict[str, Any]:
        """Return a defensive decode; callers cannot mutate the hash-bound record."""

        decoded = json.loads(self.value_canonical_json)
        if not isinstance(decoded, dict):  # guarded by validation; keeps the return type honest
            raise AssertionError("validated evidence value is not an object")
        return decoded

    @classmethod
    def create(
        cls,
        *,
        metric: str,
        value: dict[str, Any],
        graph_version_hash: str,
        source_type: Literal["deterministic-graph-algorithm", "registered-model-output"],
        node_ids: Sequence[str] = (),
        edge_ids: Sequence[str] = (),
        algorithm_config_hash: str | None = None,
        model_version_hash: str | None = None,
        model_version: str | None = None,
        model_score_hash: str | None = None,
        model_task_id: TaskId | None = None,
        model_entity_type: Literal["node", "edge", "node-pair", "community"] | None = None,
        model_entity_ids: tuple[str, ...] | None = None,
        limitations: Sequence[str],
    ) -> EvidenceItem:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-evidence/2.0",
            "metric": metric,
            "valueCanonicalJson": canonical_json(value),
            "graphVersionHash": graph_version_hash,
            "sourceType": source_type,
            "nodeIds": tuple(sorted(node_ids)),
            "edgeIds": tuple(sorted(edge_ids)),
            "algorithmConfigHash": algorithm_config_hash,
            "modelVersionHash": model_version_hash,
            "modelVersion": model_version,
            "modelScoreHash": model_score_hash,
            "modelTaskId": model_task_id,
            "modelEntityType": model_entity_type,
            "modelEntityIds": model_entity_ids,
            "limitations": tuple(limitations),
        }
        payload["evidenceHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class SimilarCase(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-similar-case/2.0"] = Field(alias="schemaVersion")
    structural_record_hash: str = Field(alias="structuralRecordHash", pattern=_HASH_PATTERN)
    similarity: float = Field(ge=-1.0, le=1.0)
    source_graph_version_hash: str = Field(alias="sourceGraphVersionHash", pattern=_HASH_PATTERN)
    source_entity_ids: tuple[str, ...] = Field(alias="sourceEntityIds", min_length=1)
    source_kind: Literal["node", "ego", "community"] = Field(alias="sourceKind")
    model_version: str = Field(alias="modelVersion", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    representation: Literal["embedding", "motif-signature"]
    query_hash: str = Field(alias="queryHash", pattern=_HASH_PATTERN)
    representation_schema: Literal["socialgraph-fm.core-structural-record/2.0"] = Field(
        alias="representationSchema"
    )
    similar_case_hash: str = Field(alias="similarCaseHash", pattern=_HASH_PATTERN)

    @model_validator(mode="after")
    def validate_case_hash(self):
        if not math.isfinite(self.similarity):
            raise ValueError("similarity must be finite")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"similar_case_hash"})
        )
        if self.similar_case_hash != expected:
            raise ValueError("similarCaseHash does not match canonical similar-case content")
        return self

    @classmethod
    def create(
        cls,
        *,
        structural_record_hash: str,
        similarity: float,
        source_graph_version_hash: str,
        source_entity_ids: tuple[str, ...],
        source_kind: Literal["node", "ego", "community"],
        model_version: str,
        model_version_hash: str,
        representation: Literal["embedding", "motif-signature"],
        query_hash: str,
        representation_schema: Literal["socialgraph-fm.core-structural-record/2.0"],
    ) -> SimilarCase:
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.core-similar-case/2.0",
            "structuralRecordHash": structural_record_hash,
            "similarity": similarity,
            "sourceGraphVersionHash": source_graph_version_hash,
            "sourceEntityIds": source_entity_ids,
            "sourceKind": source_kind,
            "modelVersion": model_version,
            "modelVersionHash": model_version_hash,
            "representation": representation,
            "queryHash": query_hash,
            "representationSchema": representation_schema,
        }
        payload["similarCaseHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)

    @classmethod
    def from_retrieval_result(
        cls, *, query: StructuralQuery, result: StructuralSearchResult
    ) -> SimilarCase:
        if result.query_provenance_hash != query.query_hash:
            raise ValueError("retrieval result does not bind the supplied structural query")
        record = result.record
        if (
            record.graph_version_hash != query.graph_version_hash
            or record.model_version != query.model_version
            or record.model_version_hash != query.model_version_hash
            or record.representation != query.representation
            or record.kind not in query.kinds
        ):
            raise ValueError("retrieval result is not compatible with the structural query")
        return cls.create(
            structural_record_hash=record.record_hash,
            similarity=result.score,
            source_graph_version_hash=record.graph_version_hash,
            source_entity_ids=record.entity_ids,
            source_kind=record.kind,
            model_version=record.model_version,
            model_version_hash=record.model_version_hash,
            representation=record.representation,
            query_hash=query.query_hash,
            representation_schema="socialgraph-fm.core-structural-record/2.0",
        )


class GovernanceFinding(_StrictModel):
    schema_version: Literal["socialgraph-fm.core-finding/2.0"] = Field(alias="schemaVersion")
    task_id: TaskId = Field(alias="taskId")
    finding_type: Literal[
        "community-resilience-candidate",
        "node-risk-candidate",
        "signed-relation-review",
        "core-collaboration-completion",
    ] = Field(alias="findingType")
    subject_ids: tuple[str, ...] = Field(alias="subjectIds", min_length=1)
    score: ModelScore
    calibrated_confidence: ConfidenceEvidence = Field(alias="calibratedConfidence")
    evidence: tuple[EvidenceItem, ...] = Field(min_length=1)
    similar_cases: tuple[SimilarCase, ...] = Field(alias="similarCases")
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=_HASH_PATTERN)
    model_version: str = Field(alias="modelVersion", min_length=1)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=_HASH_PATTERN)
    limitations: tuple[str, ...] = Field(min_length=2)
    review_status: Literal["pending-human-review"] = Field(alias="reviewStatus")
    finding_hash: str = Field(alias="findingHash", pattern=_HASH_PATTERN)

    _closed_limitations = field_validator("limitations")(_validate_closed_limitations)

    @model_validator(mode="after")
    def validate_bindings_and_hash(self):
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
        expected_entity_type = compatibility.get((self.task_id, self.finding_type))
        if expected_entity_type is None or self.score.entity_type != expected_entity_type:
            raise ValueError("task, finding type, and score entity type are not compatible")
        if self.subject_ids != self.score.entity_ids:
            raise ValueError("finding subject IDs must exactly match score entity IDs")
        if self.score.task_id != self.task_id:
            raise ValueError("finding task does not match model score task")
        if self.score.graph_version_hash != self.graph_version_hash:
            raise ValueError("finding graph version does not match model score")
        if self.score.model_version_hash != self.model_version_hash:
            raise ValueError("finding model version does not match model score")
        if self.score.model_version != self.model_version:
            raise ValueError("finding model version does not match model score")
        calibration_binding = (
            self.calibrated_confidence.score_hash == self.score.score_hash
            and self.calibrated_confidence.task_id == self.score.task_id
            and self.calibrated_confidence.entity_type == self.score.entity_type
            and self.calibrated_confidence.entity_ids == self.score.entity_ids
            and self.calibrated_confidence.graph_version_hash == self.score.graph_version_hash
            and self.calibrated_confidence.model_version == self.score.model_version
            and self.calibrated_confidence.model_version_hash == self.score.model_version_hash
        )
        if not calibration_binding:
            raise ValueError("calibration identity does not match the registered model score")
        if self.task_id == "core.community_resilience_review":
            if not isinstance(self.calibrated_confidence, RegressionConfidenceInterval):
                raise ValueError("community resilience requires a regression confidence interval")
            if self.calibrated_confidence.point_estimate != self.score.score:
                raise ValueError("regression point estimate does not match the model score")
            if REGRESSION_INTERVAL_LIMITATION not in self.limitations:
                raise ValueError("community resilience must explain interval coverage semantics")
        elif not isinstance(self.calibrated_confidence, CalibratedConfidence):
            raise ValueError("binary governance findings require calibrated confidence")
        if any(item.graph_version_hash != self.graph_version_hash for item in self.evidence):
            raise ValueError("all evidence must bind the finding graph version")
        for item in self.evidence:
            if item.source_type == "registered-model-output" and (
                item.metric != "registered_model.score-reference"
                or item.value != {}
                or item.model_score_hash != self.score.score_hash
                or item.model_task_id != self.score.task_id
                or item.model_entity_type != self.score.entity_type
                or item.model_entity_ids != self.score.entity_ids
                or item.model_version != self.score.model_version
                or item.model_version_hash != self.score.model_version_hash
            ):
                raise ValueError(
                    "model evidence must be an exact score reference to the registered score"
                )
        if any(
            item.model_version != self.model_version
            or item.model_version_hash != self.model_version_hash
            or item.representation_schema != "socialgraph-fm.core-structural-record/2.0"
            for item in self.similar_cases
        ):
            raise ValueError("similar case model/schema is not compatible with the finding")
        if MANUAL_REVIEW_LIMITATION not in self.limitations:
            raise ValueError("finding must require manual human review")
        if NON_CAUSAL_LIMITATION not in self.limitations:
            raise ValueError("finding must declare non-causal/non-future semantics")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"finding_hash"})
        )
        if self.finding_hash != expected:
            raise ValueError("findingHash does not match canonical finding content")
        return self


def _edge_id(edge: StaticEdge) -> str:
    return RegisteredEdgeIdentity.create(edge).edge_hash


def _adjacency(bundle: CoreGraphBundle) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node.id: set() for node in bundle.nodes}
    for edge in bundle.edges:
        adjacency[edge.source_id].add(edge.target_id)
        adjacency[edge.target_id].add(edge.source_id)
    return adjacency


def _components(adjacency: Mapping[str, set[str]]) -> list[list[str]]:
    unseen = set(adjacency)
    result: list[list[str]] = []
    while unseen:
        start = min(unseen)
        queue = deque((start,))
        unseen.remove(start)
        component: list[str] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in sorted(adjacency[node]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        result.append(sorted(component))
    return sorted(result, key=lambda item: (-len(item), item))


def _articulation_and_bridges(
    adjacency: Mapping[str, set[str]],
) -> tuple[set[str], set[tuple[str, str]]]:
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    articulation: set[str] = set()
    bridges: set[tuple[str, str]] = set()
    clock = 0

    def visit(node: str) -> None:
        nonlocal clock
        clock += 1
        discovery[node] = low[node] = clock
        children = 0
        for neighbor in sorted(adjacency[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent[node] is None and children > 1:
                    articulation.add(node)
                if parent[node] is not None and low[neighbor] >= discovery[node]:
                    articulation.add(node)
                if low[neighbor] > discovery[node]:
                    pair = (node, neighbor) if node < neighbor else (neighbor, node)
                    bridges.add(pair)
            elif neighbor != parent[node]:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(adjacency):
        if node not in discovery:
            parent[node] = None
            visit(node)
    return articulation, bridges


def _core_numbers(adjacency: Mapping[str, set[str]]) -> dict[str, int]:
    remaining = {node: set(neighbors) for node, neighbors in adjacency.items()}
    result: dict[str, int] = {}
    level = 0
    while remaining:
        minimum = min(len(neighbors) for neighbors in remaining.values())
        level = max(level, minimum)
        candidates = sorted(
            node for node, neighbors in remaining.items() if len(neighbors) <= level
        )
        while candidates:
            node = candidates.pop(0)
            if node not in remaining or len(remaining[node]) > level:
                continue
            result[node] = level
            neighbors = sorted(remaining.pop(node))
            for neighbor in neighbors:
                if neighbor in remaining:
                    remaining[neighbor].discard(node)
                    if len(remaining[neighbor]) <= level and neighbor not in candidates:
                        candidates.append(neighbor)
                        candidates.sort()
    return dict(sorted(result.items()))


def _shortest_path(adjacency: Mapping[str, set[str]], source: str, target: str) -> list[str] | None:
    queue = deque((source,))
    paths: dict[str, list[str]] = {source: [source]}
    while queue:
        node = queue.popleft()
        if node == target:
            return paths[node]
        for neighbor in sorted(adjacency[node]):
            if neighbor not in paths:
                paths[neighbor] = [*paths[node], neighbor]
                queue.append(neighbor)
    return None


def analyze_community_resilience(
    bundle: CoreGraphBundle,
    *,
    community_by_node: Mapping[str, str] | None = None,
) -> tuple[EvidenceItem, ...]:
    """Return facts, never a healthy/unhealthy label, over weak connectivity semantics."""

    node_ids = tuple(node.id for node in bundle.nodes)
    if community_by_node is not None and set(community_by_node) != set(node_ids):
        raise ValueError("community mapping must contain every graph node exactly once")
    config = {
        "algorithm": "static-resilience-evidence/1",
        "directedSemantics": "weak-undirected-projection" if bundle.directed else "undirected",
        "communityAssignments": dict(sorted((community_by_node or {}).items())),
    }
    config_hash = canonical_sha256(config)
    projection_limitations = (
        ("Directed edges are analyzed on a weak undirected projection.",)
        if bundle.directed
        else ("Static topology only; edge direction over time is not represented.",)
    )
    adjacency = _adjacency(bundle)
    components = _components(adjacency)
    articulation, bridge_pairs = _articulation_and_bridges(adjacency)
    pair_multiplicity: dict[tuple[str, str], int] = {}
    for edge in bundle.edges:
        pair = tuple(sorted((edge.source_id, edge.target_id)))
        pair_multiplicity[(pair[0], pair[1])] = pair_multiplicity.get((pair[0], pair[1]), 0) + 1
    bridge_ids = tuple(
        sorted(
            _edge_id(edge)
            for edge in bundle.edges
            if tuple(sorted((edge.source_id, edge.target_id))) in bridge_pairs
            and pair_multiplicity[
                (
                    min(edge.source_id, edge.target_id),
                    max(edge.source_id, edge.target_id),
                )
            ]
            == 1
        )
    )
    semantics = config["directedSemantics"]
    counts: dict[str, int] = {}
    for node in node_ids:
        community = community_by_node[node] if community_by_node else "__unassigned__"
        counts[community] = counts.get(community, 0) + 1
    total = len(node_ids)
    herfindahl = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
    profiles: dict[str, dict[str, float | int]] = {}
    for removed in node_ids:
        reduced = {
            node: {neighbor for neighbor in neighbors if neighbor != removed}
            for node, neighbors in adjacency.items()
            if node != removed
        }
        reduced_components = _components(reduced)
        remaining = len(reduced)
        profiles[removed] = {
            "componentCount": len(reduced_components),
            "largestComponentFraction": (
                len(reduced_components[0]) / remaining if reduced_components and remaining else 0.0
            ),
        }

    def fact(
        metric: str,
        value: dict[str, Any],
        *,
        fact_nodes: Sequence[str] = (),
        fact_edges: Sequence[str] = (),
    ) -> EvidenceItem:
        return EvidenceItem.create(
            metric=metric,
            value=value,
            graph_version_hash=bundle.graph_version_hash,
            source_type="deterministic-graph-algorithm",
            node_ids=fact_nodes,
            edge_ids=fact_edges,
            algorithm_config_hash=config_hash,
            limitations=projection_limitations,
        )

    component_value: dict[str, Any] = {
        "count": len(components),
        "sizes": [len(item) for item in components],
    }
    if bundle.directed:
        component_value["semantics"] = semantics
    return (
        fact(
            "connectivity.components",
            component_value,
            fact_nodes=node_ids,
        ),
        fact(
            "connectivity.articulation_points",
            {"count": len(articulation), "semantics": semantics},
            fact_nodes=sorted(articulation),
        ),
        fact(
            "connectivity.bridges",
            {"count": len(bridge_ids), "semantics": semantics},
            fact_edges=bridge_ids,
        ),
        fact("k_core.node_core_numbers", _core_numbers(adjacency), fact_nodes=node_ids),
        fact(
            "community.concentration",
            {"counts": dict(sorted(counts.items())), "herfindahl": herfindahl},
            fact_nodes=node_ids,
        ),
        fact("stress.node_removal", {"profiles": profiles}, fact_nodes=node_ids),
    )


def _model_evidence(score: ModelScore) -> EvidenceItem:
    return EvidenceItem.create(
        metric="registered_model.score-reference",
        value={},
        graph_version_hash=score.graph_version_hash,
        source_type="registered-model-output",
        node_ids=score.entity_ids if score.entity_type in {"node", "community"} else (),
        edge_ids=("|".join(score.entity_ids),)
        if score.entity_type in {"edge", "node-pair"}
        else (),
        model_version_hash=score.model_version_hash,
        model_version=score.model_version,
        model_score_hash=score.score_hash,
        model_task_id=score.task_id,
        model_entity_type=score.entity_type,
        model_entity_ids=score.entity_ids,
        limitations=("The score is a registered model output, not a graph fact or decision.",),
    )


def create_governance_finding(
    *,
    task_id: TaskId,
    finding_type: Literal[
        "community-resilience-candidate",
        "node-risk-candidate",
        "signed-relation-review",
        "core-collaboration-completion",
    ],
    subject_ids: tuple[str, ...],
    score: ModelScore,
    calibrated_confidence: ConfidenceEvidence,
    evidence: Sequence[EvidenceItem],
    similar_cases: Sequence[SimilarCase],
    limitations: Sequence[str],
) -> GovernanceFinding:
    if score.task_id != task_id:
        raise ValueError("model score task does not match requested governance task")
    complete_limitations = tuple(
        dict.fromkeys((*limitations, MANUAL_REVIEW_LIMITATION, NON_CAUSAL_LIMITATION))
    )
    payload: dict[str, Any] = {
        "schemaVersion": "socialgraph-fm.core-finding/2.0",
        "taskId": task_id,
        "findingType": finding_type,
        "subjectIds": subject_ids,
        "score": score,
        "calibratedConfidence": calibrated_confidence,
        "evidence": tuple(evidence),
        "similarCases": tuple(similar_cases),
        "graphVersionHash": score.graph_version_hash,
        "modelVersion": score.model_version,
        "modelVersionHash": score.model_version_hash,
        "limitations": complete_limitations,
        "reviewStatus": "pending-human-review",
    }
    payload["findingHash"] = canonical_sha256(payload)
    return GovernanceFinding.model_validate(payload)


def _validate_score_confidence_pair(score: ModelScore, confidence: ConfidenceEvidence) -> None:
    if (
        confidence.score_hash != score.score_hash
        or confidence.task_id != score.task_id
        or confidence.entity_type != score.entity_type
        or confidence.entity_ids != score.entity_ids
        or confidence.graph_version_hash != score.graph_version_hash
        or confidence.model_version != score.model_version
        or confidence.model_version_hash != score.model_version_hash
    ):
        raise ValueError("confidence does not bind the supplied model score")
    if (
        isinstance(confidence, RegressionConfidenceInterval)
        and confidence.point_estimate != score.score
    ):
        raise ValueError("regression point estimate does not match the model score")


def build_community_resilience_findings(
    bundle: CoreGraphBundle,
    *,
    scored_candidates: Sequence[tuple[ModelScore, RegressionConfidenceInterval]],
    community_by_node: Mapping[str, str] | None = None,
) -> tuple[GovernanceFinding, ...]:
    deterministic_evidence = analyze_community_resilience(
        bundle, community_by_node=community_by_node
    )
    known_nodes = {node.id for node in bundle.nodes}
    findings: list[GovernanceFinding] = []
    for score, confidence in scored_candidates:
        _validate_score_confidence_pair(score, confidence)
        if score.graph_version_hash != bundle.graph_version_hash:
            raise ValueError("model score graph version does not match registered graph")
        if score.task_id != "core.community_resilience_review":
            raise ValueError("model score task is not community resilience review")
        if score.entity_type != "community" or not set(score.entity_ids) <= known_nodes:
            raise ValueError(
                "community resilience scores must reference registered community nodes"
            )
        findings.append(
            create_governance_finding(
                task_id="core.community_resilience_review",
                finding_type="community-resilience-candidate",
                subject_ids=score.entity_ids,
                score=score,
                calibrated_confidence=confidence,
                evidence=(_model_evidence(score), *deterministic_evidence),
                similar_cases=(),
                limitations=(
                    "Connectivity evidence is factual topology context, not a community health label.",
                    REGRESSION_INTERVAL_LIMITATION,
                ),
            )
        )
    return tuple(findings)


def build_risk_and_trust_findings(
    bundle: CoreGraphBundle,
    *,
    scored_candidates: Sequence[tuple[ModelScore, CalibratedConfidence]],
) -> tuple[GovernanceFinding, ...]:
    known_nodes = {node.id for node in bundle.nodes}
    findings: list[GovernanceFinding] = []
    for score, confidence in scored_candidates:
        _validate_score_confidence_pair(score, confidence)
        if score.graph_version_hash != bundle.graph_version_hash:
            raise ValueError("model score graph version does not match registered graph")
        if score.task_id != "core.risk_and_trust_review":
            raise ValueError("model score task is not risk and trust review")
        evidence = [_model_evidence(score)]
        if score.entity_type == "node":
            if len(score.entity_ids) != 1 or score.entity_ids[0] not in known_nodes:
                raise ValueError("node-risk score must reference exactly one existing graph node")
            finding_type: Literal["node-risk-candidate", "signed-relation-review"] = (
                "node-risk-candidate"
            )
        elif score.entity_type == "edge" and len(score.entity_ids) == 2:
            finding_type = "signed-relation-review"
            if score.edge_identity is None:
                raise ValueError("signed relation score requires an exact stable edge identity")
            matches = [
                edge
                for edge in bundle.edges
                if RegisteredEdgeIdentity.create(edge) == score.edge_identity
            ]
            if len(matches) != 1:
                raise ValueError("signed relation edge identity is not uniquely registered")
            edge = matches[0]
            evidence.append(
                EvidenceItem.create(
                    metric="signed_relation.observed",
                    value={"edgeType": edge.edge_type, "weight": edge.weight},
                    graph_version_hash=bundle.graph_version_hash,
                    source_type="deterministic-graph-algorithm",
                    node_ids=score.entity_ids,
                    edge_ids=(score.edge_identity.edge_hash,),
                    algorithm_config_hash=canonical_sha256(
                        {"algorithm": "observed-signed-relation/1", "directed": bundle.directed}
                    ),
                    limitations=("Support/opposition semantics require contextual human review.",),
                )
            )
        else:
            raise ValueError("risk review supports only node and observed edge candidates")
        findings.append(
            create_governance_finding(
                task_id="core.risk_and_trust_review",
                finding_type=finding_type,
                subject_ids=score.entity_ids,
                score=score,
                calibrated_confidence=confidence,
                evidence=evidence,
                similar_cases=(),
                limitations=("Candidate for review; it is not a risk or trust truth label.",),
            )
        )
    return tuple(findings)


def build_collaboration_findings(
    bundle: CoreGraphBundle,
    *,
    scored_candidates: Sequence[tuple[ModelScore, CalibratedConfidence]],
    top_k: int,
) -> tuple[GovernanceFinding, ...]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    adjacency = _adjacency(bundle)
    known_nodes = set(adjacency)
    directed_edges = {(edge.source_id, edge.target_id) for edge in bundle.edges}
    selected: dict[tuple[str, str], tuple[ModelScore, CalibratedConfidence]] = {}
    for score, confidence in scored_candidates:
        _validate_score_confidence_pair(score, confidence)
        if score.graph_version_hash != bundle.graph_version_hash:
            raise ValueError("model score graph version does not match registered graph")
        if score.task_id != "core.collaboration_completion":
            raise ValueError("model score task is not collaboration completion")
        if score.entity_type != "node-pair" or len(score.entity_ids) != 2:
            raise ValueError("collaboration completion requires node-pair scores")
        source, target = score.entity_ids
        if source == target:
            raise ValueError("collaboration completion rejects self-pairs")
        if source not in known_nodes or target not in known_nodes:
            raise ValueError("collaboration candidate references an unknown node")
        exists = (
            (source, target) in directed_edges
            if bundle.directed
            else (source, target) in directed_edges or (target, source) in directed_edges
        )
        if exists:
            raise ValueError("collaboration completion candidates must be static non-edges")
        key = (source, target) if bundle.directed or source < target else (target, source)
        current = selected.get(key)
        if current is None or (
            -score.score,
            score.entity_ids,
            score.score_hash,
            confidence.confidence_hash,
        ) < (
            -current[0].score,
            current[0].entity_ids,
            current[0].score_hash,
            current[1].confidence_hash,
        ):
            selected[key] = (score, confidence)
    ranked = sorted(selected.values(), key=lambda item: (-item[0].score, item[0].entity_ids))
    findings: list[GovernanceFinding] = []
    semantics = "weak-undirected-projection" if bundle.directed else "undirected"
    config_hash = canonical_sha256(
        {
            "algorithm": "common-neighbor-shortest-path/1",
            "directed": bundle.directed,
            "semantics": semantics,
        }
    )
    for score, confidence in ranked[:top_k]:
        source, target = score.entity_ids
        common = sorted(adjacency[source] & adjacency[target])
        path = _shortest_path(adjacency, source, target)
        facts = (
            EvidenceItem.create(
                metric="neighbors.common",
                value={"count": len(common), "nodes": common, "semantics": semantics},
                graph_version_hash=bundle.graph_version_hash,
                source_type="deterministic-graph-algorithm",
                node_ids=(source, target, *common),
                algorithm_config_hash=config_hash,
                limitations=(
                    "Common-neighbor evidence describes only the registered static graph.",
                ),
            ),
            EvidenceItem.create(
                metric="core_graph.existing-path",
                value={
                    "distance": len(path) - 1 if path else None,
                    "path": path or [],
                    "semantics": semantics,
                },
                graph_version_hash=bundle.graph_version_hash,
                source_type="deterministic-graph-algorithm",
                node_ids=tuple(path or (source, target)),
                algorithm_config_hash=config_hash,
                limitations=(
                    "Path evidence is static relation-completion context, not a future-event forecast.",
                ),
            ),
        )
        findings.append(
            create_governance_finding(
                task_id="core.collaboration_completion",
                finding_type="core-collaboration-completion",
                subject_ids=score.entity_ids,
                score=score,
                calibrated_confidence=confidence,
                evidence=(_model_evidence(score), *facts),
                similar_cases=(),
                limitations=(
                    "Static relation-completion recommendation only.",
                    *(
                        ("Directed structural context uses a weak undirected projection.",)
                        if bundle.directed
                        else ()
                    ),
                ),
            )
        )
    return tuple(findings)


def load_governance_finding_json(serialized: str | bytes) -> GovernanceFinding:
    if not isinstance(serialized, (str, bytes)):
        raise TypeError("GovernanceFinding input must be UTF-8 JSON text or bytes")
    return GovernanceFinding.model_validate_json(serialized)


def validate_similar_case_provenance(case: SimilarCase, structural_index: Any) -> None:
    try:
        query, result = structural_index.resolve_query_result(
            query_hash=case.query_hash,
            record_hash=case.structural_record_hash,
        )
    except (AttributeError, ValueError) as error:
        raise ValueError("unknown registered structural query/result provenance") from error
    expected = SimilarCase.from_retrieval_result(query=query, result=result)
    if expected != case:
        raise ValueError(
            "registered structural query/result provenance does not match similar case"
        )


__all__ = [
    "CalibratedConfidence",
    "ConfidenceEvidence",
    "EvidenceItem",
    "GovernanceFinding",
    "MANUAL_REVIEW_LIMITATION",
    "ModelScore",
    "RegressionConfidenceInterval",
    "NON_CAUSAL_LIMITATION",
    "SimilarCase",
    "analyze_community_resilience",
    "build_collaboration_findings",
    "build_community_resilience_findings",
    "build_risk_and_trust_findings",
    "create_governance_finding",
    "load_governance_finding_json",
    "validate_similar_case_provenance",
]
