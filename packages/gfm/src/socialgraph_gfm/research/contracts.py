"""Versioned contracts for the independent SocialGraph-FM Research application path."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from socialgraph_gfm.canonical import canonical_sha256

HASH_PATTERN = r"^[0-9a-f]{64}$"
RELEASE_ID = "research"
RESEARCH_SEED = 1729

CONTENT_POLICY_TASK: Literal["research.content_policy_review"] = (
    "research.content_policy_review"
)
ACCOUNT_RISK_TASK: Literal["research.account_risk_review"] = (
    "research.account_risk_review"
)
SIGNED_RELATION_TASK: Literal["research.signed_relation_review"] = (
    "research.signed_relation_review"
)
COLLABORATION_TASK: Literal["core.collaboration_completion"] = (
    "core.collaboration_completion"
)

ResearchTaskId = Literal[
    "research.content_policy_review",
    "research.account_risk_review",
    "research.signed_relation_review",
    "core.collaboration_completion",
]
RESEARCH_TASK_IDS: tuple[ResearchTaskId, ...] = (
    CONTENT_POLICY_TASK,
    ACCOUNT_RISK_TASK,
    SIGNED_RELATION_TASK,
    COLLABORATION_TASK,
)
RESEARCH_DOMAIN_IDS = (
    "email-eu-core",
    "tolokers",
    "twitch-DE",
    "twitch-EN",
    "twitch-ES",
    "twitch-FR",
    "twitch-PT",
    "twitch-RU",
    "wiki-rfa",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        protected_namespaces=("model_dump",),
        strict=True,
    )


class ResearchTaskCapability(_StrictModel):
    task_id: ResearchTaskId = Field(alias="taskId")
    title: str = Field(min_length=1, max_length=100)
    dataset_family: Literal["twitch-language", "tolokers", "wiki-rfa", "email-eu-core"] = (
        Field(alias="datasetFamily")
    )
    graph_ids: tuple[str, ...] = Field(alias="graphIds", min_length=1, strict=False)
    target_scope: Literal["nodes", "directed-node-pairs", "collaboration-candidates"] = (
        Field(alias="targetScope")
    )
    head_name: Literal[
        "content_policy_head",
        "account_risk_head",
        "signed_edge_head",
        "collaboration_head",
    ] = Field(alias="headName")
    score_semantics: str = Field(alias="scoreSemantics", min_length=1, max_length=500)
    primary_metric: Literal["macro-f1", "auprc", "negative-auprc", "filtered-mrr"] = (
        Field(alias="primaryMetric")
    )
    user_graph_eligible: bool = Field(alias="userGraphEligible")
    excluded_fields: tuple[str, ...] = Field(alias="excludedFields", strict=False)


class ResearchModelIdentity(_StrictModel):
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    transductive: Literal[True] = True


class ResearchReadiness(_StrictModel):
    materialized: bool
    trained: bool
    evaluated: bool
    exported: bool
    smoke_passed: bool = Field(alias="smokePassed")
    published: bool

    @model_validator(mode="after")
    def validate_monotonic_stages(self):
        values = (
            self.materialized,
            self.trained,
            self.evaluated,
            self.exported,
            self.smoke_passed,
            self.published,
        )
        if any(right and not left for left, right in pairwise(values)):
            raise ValueError("research readiness stages must be monotonic")
        return self


class ResearchAdvantageClaim(_StrictModel):
    state: Literal["unavailable", "observed", "not-demonstrated"]
    qualifying_task_count: int = Field(alias="qualifyingTaskCount", ge=0, le=4)
    average_primary_metric_delta: float | None = Field(alias="averagePrimaryMetricDelta")
    label: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_claim(self):
        observed = self.qualifying_task_count >= 3 and (
            self.average_primary_metric_delta is not None
            and self.average_primary_metric_delta > 0.0
        )
        if (self.state == "observed") != observed:
            raise ValueError("advantage claim is not derived from the four primary metrics")
        if self.state == "unavailable" and self.average_primary_metric_delta is not None:
            raise ValueError("unavailable advantage evidence must not contain a metric delta")
        return self


class ResearchUploadPolicy(_StrictModel):
    task_ids: tuple[Literal["core.collaboration_completion"], ...] = Field(
        alias="taskIds", strict=False
    )
    structural_similarity: Literal[True] = Field(alias="structuralSimilarity")
    min_completion_nodes: int = Field(alias="minCompletionNodes", ge=1)
    max_nodes: int = Field(alias="maxNodes", ge=1)
    max_edges: int = Field(alias="maxEdges", ge=1)
    min_non_edge_candidates: int = Field(alias="minNonEdgeCandidates", ge=1)
    min_similarity_nodes: int = Field(alias="minSimilarityNodes", ge=1)
    min_similarity_edges: int = Field(alias="minSimilarityEdges", ge=0)


class ResearchCapabilities(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-capabilities/1.0"] = Field(
        alias="schemaVersion"
    )
    release_id: Literal["research"] = Field(alias="releaseId")
    release_label: Literal["SocialGraph-FM Research"] = Field(alias="releaseLabel")
    seed: Literal[1729]
    preliminary: Literal[True]
    result_qualifier: Literal["single-seed-preliminary"] = Field(alias="resultQualifier")
    formal_readiness_unchanged: Literal[True] = Field(alias="formalReadinessUnchanged")
    research_serving_ready: bool = Field(alias="researchServingReady")
    model: ResearchModelIdentity | None
    task_ids: tuple[ResearchTaskId, ...] = Field(alias="taskIds", strict=False)
    tasks: tuple[ResearchTaskCapability, ...] = Field(strict=False)
    upload: ResearchUploadPolicy
    advantage_claim: ResearchAdvantageClaim = Field(alias="advantageClaim")
    readiness: ResearchReadiness
    capability_hash: str = Field(alias="capabilityHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_derivation(self):
        if self.task_ids != tuple(task.task_id for task in self.tasks):
            raise ValueError("research taskIds must derive from the ordered task inventory")
        if self.research_serving_ready != (
            self.readiness.published and self.model is not None
        ):
            raise ValueError("researchServingReady must derive from publication and model identity")
        expected = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"capability_hash"})
        )
        if self.capability_hash != expected:
            raise ValueError("capabilityHash does not bind the research capability document")
        return self


class ResearchTarget(_StrictModel):
    scope: Literal["nodes", "directed-node-pairs", "collaboration-candidates"]
    node_ids: tuple[str, ...] = Field(default=(), alias="nodeIds", strict=False)
    pairs: tuple[tuple[str, str], ...] = Field(default=(), strict=False)
    anchor_node_id: str | None = Field(default=None, alias="anchorNodeId")
    candidate_limit: int = Field(default=20, alias="candidateLimit", ge=1, le=100)

    @model_validator(mode="after")
    def validate_scope(self):
        if self.scope == "nodes":
            if not self.node_ids or self.pairs or self.anchor_node_id is not None:
                raise ValueError("node scope requires only a nonempty nodeIds inventory")
        elif self.scope == "directed-node-pairs":
            if not self.pairs or self.node_ids or self.anchor_node_id is not None:
                raise ValueError("directed pair scope requires only a nonempty pairs inventory")
        elif self.node_ids or self.pairs:
            raise ValueError("collaboration candidate scope uses an optional anchor only")
        if any(left == right for left, right in self.pairs):
            raise ValueError("research target pairs require distinct endpoints")
        return self


class ResearchScenario(_StrictModel):
    scenario_id: str = Field(alias="scenarioId", min_length=1, max_length=200)
    task_id: ResearchTaskId = Field(alias="taskId")
    title: str = Field(min_length=1, max_length=100)
    dataset_family: str = Field(alias="datasetFamily", min_length=1, max_length=100)
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str | None = Field(
        default=None, alias="graphVersionHash", pattern=HASH_PATTERN
    )
    model_version_id: str | None = Field(default=None, alias="modelVersionId")
    model_version_hash: str | None = Field(
        default=None, alias="modelVersionHash", pattern=HASH_PATTERN
    )
    availability: Literal["ready", "blocked"]
    blocked_reasons: tuple[str, ...] = Field(alias="blockedReasons", strict=False)
    default_target: ResearchTarget = Field(alias="defaultTarget")

    @model_validator(mode="after")
    def validate_availability(self):
        identities = (
            self.graph_version_hash,
            self.model_version_id,
            self.model_version_hash,
        )
        if self.availability == "ready":
            if any(value is None for value in identities) or self.blocked_reasons:
                raise ValueError("ready research scenario requires complete identities")
        elif not self.blocked_reasons:
            raise ValueError("blocked research scenario requires an explicit reason")
        return self


class ResearchScenarios(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-scenarios/1.0"] = Field(
        alias="schemaVersion"
    )
    release_id: Literal["research"] = Field(alias="releaseId")
    scenarios: tuple[ResearchScenario, ...] = Field(strict=False)
    scenarios_hash: str = Field(alias="scenariosHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.scenarios_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"scenarios_hash"})
        ):
            raise ValueError("scenariosHash does not bind the scenario catalog")
        return self


class ResearchRunRequest(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-run-request/1.0"] = Field(
        alias="schemaVersion"
    )
    task_id: ResearchTaskId = Field(alias="taskId")
    scenario_id: str | None = Field(default=None, alias="scenarioId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    target: ResearchTarget
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_task_and_hash(self):
        expected_scope = {
            CONTENT_POLICY_TASK: "nodes",
            ACCOUNT_RISK_TASK: "nodes",
            SIGNED_RELATION_TASK: "directed-node-pairs",
            COLLABORATION_TASK: "collaboration-candidates",
        }[self.task_id]
        if self.target.scope != expected_scope:
            raise ValueError("research task and target scope disagree")
        expected_hash = canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"request_hash"})
        )
        if self.request_hash != expected_hash:
            raise ValueError("requestHash does not bind the research request")
        return self

    @classmethod
    def create(cls, **values: Any) -> ResearchRunRequest:
        payload = {"schemaVersion": "socialgraph-fm.research-run-request/1.0", **values}
        payload["requestHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class ResearchRunStatus(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-run-status/1.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    status: Literal["queued", "running", "succeeded", "failed"]
    progress: int = Field(ge=0, le=100)
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")
    error_code: str | None = Field(default=None, alias="errorCode")
    state_hash: str = Field(alias="stateHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.state_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"state_hash"})
        ):
            raise ValueError("stateHash does not bind the research run state")
        return self

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        request_hash: str,
        status: Literal["queued", "running", "succeeded", "failed"],
        progress: int,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        error_code: str | None = None,
    ) -> ResearchRunStatus:
        created = datetime.now(UTC) if created_at is None else created_at
        payload: dict[str, Any] = {
            "schemaVersion": "socialgraph-fm.research-run-status/1.0",
            "runId": run_id,
            "requestHash": request_hash,
            "status": status,
            "progress": progress,
            "createdAt": created,
            "updatedAt": created if updated_at is None else updated_at,
            "errorCode": error_code,
        }
        payload["stateHash"] = canonical_sha256(payload)
        return cls.model_validate(payload)


class ResearchFinding(_StrictModel):
    finding_id: str = Field(alias="findingId", min_length=1, max_length=200)
    task_id: ResearchTaskId = Field(alias="taskId")
    entity_type: Literal["node", "directed-node-pair", "node-pair"] = Field(
        alias="entityType"
    )
    entity_ids: tuple[str, ...] = Field(alias="entityIds", min_length=1, strict=False)
    score: float
    calibrated_score: float | None = Field(default=None, alias="calibratedScore")
    calibration_state: Literal["calibrated", "rank-only"] = Field(alias="calibrationState")
    preliminary: Literal[True]
    review_status: Literal["pending-human-review"] = Field(alias="reviewStatus")
    evidence_hashes: tuple[str, ...] = Field(alias="evidenceHashes", strict=False)
    finding_hash: str = Field(alias="findingHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_finding(self):
        expected_entities = {"node": 1, "directed-node-pair": 2, "node-pair": 2}
        if len(self.entity_ids) != expected_entities[self.entity_type]:
            raise ValueError("finding entity inventory does not match entityType")
        if (self.calibration_state == "calibrated") != (self.calibrated_score is not None):
            raise ValueError("calibration state and score disagree")
        if self.finding_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"finding_hash"})
        ):
            raise ValueError("findingHash does not bind the immutable finding")
        return self


class ResearchRunResult(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-run-result/1.0"] = Field(
        alias="schemaVersion"
    )
    run_id: str = Field(alias="runId", min_length=1, max_length=100)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    task_id: ResearchTaskId = Field(alias="taskId")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    findings: tuple[ResearchFinding, ...] = Field(strict=False)
    completed_at: datetime = Field(alias="completedAt")
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_result(self):
        if any(finding.task_id != self.task_id for finding in self.findings):
            raise ValueError("research findings must match their run task")
        if self.result_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"result_hash"})
        ):
            raise ValueError("resultHash does not bind the research result")
        return self


class ResearchSimilarityRequest(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-similarity-request/1.0"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=500)
    top_k: int = Field(alias="topK", ge=1, le=100)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.request_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"request_hash"})
        ):
            raise ValueError("similarity requestHash mismatch")
        return self


class ResearchSimilarityHit(_StrictModel):
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_version_hash: str = Field(alias="graphVersionHash", pattern=HASH_PATTERN)
    node_id: str = Field(alias="nodeId", min_length=1, max_length=500)
    score: float = Field(ge=-1.0, le=1.0)
    deterministic_facts: dict[str, float] = Field(alias="deterministicFacts")
    record_hash: str = Field(alias="recordHash", pattern=HASH_PATTERN)


class ResearchSimilarityResult(_StrictModel):
    schema_version: Literal["socialgraph-fm.research-similarity-result/1.0"] = Field(
        alias="schemaVersion"
    )
    request_hash: str = Field(alias="requestHash", pattern=HASH_PATTERN)
    model_version_id: str = Field(alias="modelVersionId", min_length=1, max_length=300)
    model_version_hash: str = Field(alias="modelVersionHash", pattern=HASH_PATTERN)
    hits: tuple[ResearchSimilarityHit, ...] = Field(strict=False)
    result_hash: str = Field(alias="resultHash", pattern=HASH_PATTERN)

    @model_validator(mode="after")
    def validate_hash(self):
        if self.result_hash != canonical_sha256(
            self.model_dump(mode="python", by_alias=True, exclude={"result_hash"})
        ):
            raise ValueError("similarity resultHash mismatch")
        return self


def task_capabilities() -> tuple[ResearchTaskCapability, ...]:
    return (
        ResearchTaskCapability(
            task_id=CONTENT_POLICY_TASK,
            title="内容策略复核",
            dataset_family="twitch-language",
            graph_ids=("twitch-DE", "twitch-EN", "twitch-ES", "twitch-FR", "twitch-PT", "twitch-RU"),
            target_scope="nodes",
            head_name="content_policy_head",
            score_semantics="explicit-language 历史标签的复核排序分数，不代表违法或有害内容判定",
            primary_metric="macro-f1",
            user_graph_eligible=False,
            excluded_fields=("mature",),
        ),
        ResearchTaskCapability(
            task_id=ACCOUNT_RISK_TASK,
            title="历史账号状态复核",
            dataset_family="tolokers",
            graph_ids=("tolokers",),
            target_scope="nodes",
            head_name="account_risk_head",
            score_semantics="历史 banned 标签相似性的复核排序分数，不用于自动封禁或提前预警",
            primary_metric="auprc",
            user_graph_eligible=False,
            excluded_fields=("banned",),
        ),
        ResearchTaskCapability(
            task_id=SIGNED_RELATION_TASK,
            title="治理关系立场复核",
            dataset_family="wiki-rfa",
            graph_ids=("wiki-rfa",),
            target_scope="directed-node-pairs",
            head_name="signed_edge_head",
            score_semantics="Wiki-RfA 支持/反对关系分数，不代表毒性或客观可信度",
            primary_metric="negative-auprc",
            user_graph_eligible=False,
            excluded_fields=("TXT", "DAT", "YEA", "RES", "VOT"),
        ),
        ResearchTaskCapability(
            task_id=COLLABORATION_TASK,
            title="协作关系候选",
            dataset_family="email-eu-core",
            graph_ids=("email-eu-core",),
            target_scope="collaboration-candidates",
            head_name="collaboration_head",
            score_semantics="静态图中未观察关系的补全排序分数，不代表未来关系预测",
            primary_metric="filtered-mrr",
            user_graph_eligible=True,
            excluded_fields=("department",),
        ),
    )


__all__ = [
    "ACCOUNT_RISK_TASK",
    "COLLABORATION_TASK",
    "CONTENT_POLICY_TASK",
    "RELEASE_ID",
    "RESEARCH_DOMAIN_IDS",
    "RESEARCH_SEED",
    "RESEARCH_TASK_IDS",
    "SIGNED_RELATION_TASK",
    "ResearchAdvantageClaim",
    "ResearchCapabilities",
    "ResearchFinding",
    "ResearchModelIdentity",
    "ResearchReadiness",
    "ResearchRunRequest",
    "ResearchRunResult",
    "ResearchRunStatus",
    "ResearchScenario",
    "ResearchScenarios",
    "ResearchSimilarityHit",
    "ResearchSimilarityRequest",
    "ResearchSimilarityResult",
    "ResearchTarget",
    "ResearchTaskCapability",
    "ResearchTaskId",
    "ResearchUploadPolicy",
    "task_capabilities",
]
