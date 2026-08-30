from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .gfm_research_schemas import ResearchGraphCompatibility


class DatasetModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class DatasetIssue(DatasetModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    file: str | None = None


class DatasetFileProfile(DatasetModel):
    name: str
    size: int = Field(ge=0)
    role: str


class DatasetProfile(DatasetModel):
    node_count: int | None = Field(alias="nodeCount", default=None, ge=0)
    edge_count: int | None = Field(alias="edgeCount", default=None, ge=0)
    feature_dimension: int | None = Field(alias="featureDimension", default=None, ge=0)
    label_count: int | None = Field(alias="labelCount", default=None, ge=0)
    split_names: list[str] = Field(alias="splitNames", default_factory=list)
    directed: bool = False


class DatasetInspection(DatasetModel):
    id: str
    detected_format: str = Field(alias="detectedFormat")
    status: Literal["accepted", "mapping_required", "conversion_required", "rejected"]
    profile: DatasetProfile | None = None
    files: list[DatasetFileProfile] = Field(default_factory=list)
    issues: list[DatasetIssue] = Field(default_factory=list)
    dataset_candidates: list[str] = Field(alias="datasetCandidates", default_factory=list)
    server_graph_fact_hash: str | None = Field(
        alias="serverGraphFactHash", default=None, pattern=r"^[0-9a-f]{64}$"
    )
    created_at: datetime = Field(alias="createdAt")


class DatasetInspectionCancellation(DatasetModel):
    status: Literal["released"] = "released"


class ArtifactNode(DatasetModel):
    id: str
    label: str
    node_type: str | None = Field(alias="nodeType", default=None)
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArtifactEdge(DatasetModel):
    id: str
    source: str
    target: str
    edge_type: str | None = Field(alias="edgeType", default=None)
    weight: float | None = None
    timestamp: str | None = None
    directed: bool | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ArtifactGraphSummary(DatasetModel):
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    density: float = Field(ge=0)
    connected_components: int = Field(alias="connectedComponents", ge=0)
    visible_node_count: int = Field(alias="visibleNodeCount", ge=0)
    visible_edge_count: int = Field(alias="visibleEdgeCount", ge=0)
    partial_preview: bool = Field(alias="partialPreview")


class ArtifactGraphView(DatasetModel):
    id: str
    nodes: list[ArtifactNode]
    edges: list[ArtifactEdge]
    summary: ArtifactGraphSummary


GraphFactScalar = str | int | float | bool | None


class GraphVersionExportNode(DatasetModel):
    """A browser GraphVersion node fact; arbitrary executable values are impossible."""

    id: str = Field(min_length=1, max_length=1000)
    label: str = Field(max_length=4000)
    node_type: str | None = Field(alias="type", default=None, max_length=1000)
    attributes: dict[str, GraphFactScalar | list[GraphFactScalar]] = Field(
        default_factory=dict,
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_attribute_values(self) -> GraphVersionExportNode:
        for key, value in self.attributes.items():
            if not key or len(key) > 1000:
                raise ValueError("节点属性名长度无效")
            values = value if isinstance(value, list) else [value]
            if len(values) > 1024:
                raise ValueError("节点属性数组过长")
            if any(isinstance(item, float) and not math.isfinite(item) for item in values):
                raise ValueError("节点属性不能包含 NaN 或无穷值")
            if any(isinstance(item, str) and len(item) > 16_000 for item in values):
                raise ValueError("节点属性文本过长")
        return self


class GraphVersionExportEdge(DatasetModel):
    """A browser GraphVersion edge fact with explicit, non-executable semantics."""

    id: str = Field(min_length=1, max_length=1000)
    source: str = Field(min_length=1, max_length=1000)
    target: str = Field(min_length=1, max_length=1000)
    edge_type: str | None = Field(alias="type", default=None, max_length=1000)
    weight: float | None = None
    timestamp: str | None = Field(default=None, max_length=1000)
    directed: bool | None = None
    attributes: dict[str, GraphFactScalar | list[GraphFactScalar]] = Field(
        default_factory=dict,
        max_length=256,
    )

    @model_validator(mode="after")
    def validate_fact_values(self) -> GraphVersionExportEdge:
        if self.weight is not None and not math.isfinite(self.weight):
            raise ValueError("边权重不能是 NaN 或无穷值")
        for key, value in self.attributes.items():
            if not key or len(key) > 1000:
                raise ValueError("边属性名长度无效")
            values = value if isinstance(value, list) else [value]
            if len(values) > 1024:
                raise ValueError("边属性数组过长")
            if any(isinstance(item, float) and not math.isfinite(item) for item in values):
                raise ValueError("边属性不能包含 NaN 或无穷值")
            if any(isinstance(item, str) and len(item) > 16_000 for item in values):
                raise ValueError("边属性文本过长")
        return self


class GraphVersionTargetDomainEnvelope(DatasetModel):
    """Safe text handoff from an immutable browser GraphVersion.

    This contract is intentionally data-only.  It cannot request training, set a
    dataset role, attest a license, or carry executable recipes.
    """

    schema_version: Literal["socialgraph-fm-graph/1.0", "socialgraph-fm-graph/1.1"] = Field(
        alias="schemaVersion"
    )
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    content_hash: str = Field(alias="contentHash", pattern=r"^[0-9a-f]{64}$")
    build_spec_hash: str = Field(alias="buildSpecHash", pattern=r"^[0-9a-f]{64}$")
    source_file: str = Field(alias="sourceFile", min_length=1, max_length=1000)
    graph_fact_hash: str | None = Field(
        alias="graphFactHash", default=None, pattern=r"^[0-9a-f]{64}$"
    )
    directedness: Literal["directed", "undirected", "mixed", "unspecified"]
    nodes: list[GraphVersionExportNode] = Field(max_length=2_000_000)
    edges: list[GraphVersionExportEdge] = Field(max_length=5_000_000)

    @model_validator(mode="after")
    def require_fact_hash_for_v11(self) -> GraphVersionTargetDomainEnvelope:
        if self.schema_version == "socialgraph-fm-graph/1.1" and self.graph_fact_hash is None:
            raise ValueError("GraphVersion 1.1 必须携带 graphFactHash")
        return self


# DatasetArtifact 2.1/2.2 contracts. They describe data and intended
# evaluation, but do not contain executable trainer/model configuration.
class SourceFileDigest(DatasetModel):
    path: str = Field(min_length=1, max_length=4096)
    role: str = Field(min_length=1, max_length=100)
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ArrayDescriptor(DatasetModel):
    name: str = Field(min_length=1, max_length=4096)
    role: Literal[
        "edge_index",
        "node_id_map",
        "feature",
        "label",
        "split",
        "variant",
        "auxiliary",
    ]
    dtype: str = Field(min_length=1, max_length=100)
    shape: list[int] = Field(max_length=4)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class NodeIdentitySchema(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    array_name: str = Field(alias="arrayName", min_length=1, max_length=4096)
    kind: Literal["source", "row_index"]
    count: int = Field(ge=0)
    unique: bool = True


class FeatureSchema(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    array_name: str = Field(alias="arrayName", min_length=1, max_length=4096)
    target: Literal["node", "edge", "graph"] = "node"
    dtype: str
    shape: list[int]
    layout: Literal["dense", "sparse"] = "dense"
    missing_value_policy: Literal["reject", "mask", "allow_nan"] = Field(
        alias="missingValuePolicy", default="reject"
    )


class LabelSchema(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    array_name: str = Field(alias="arrayName", min_length=1, max_length=4096)
    target: Literal["node", "edge", "graph"] = "node"
    mode: Literal["single_label", "multi_label", "regression"] = "single_label"
    dtype: str
    shape: list[int]
    class_count: int | None = Field(alias="classCount", default=None, ge=0)
    ignore_value: int | float | None = Field(alias="ignoreValue", default=-1)
    class_values: list[int | float | str] = Field(alias="classValues", default_factory=list)


class GraphSemantics(DatasetModel):
    directed: bool
    directedness: Literal["directed", "undirected", "mixed", "unspecified"] | None = None
    edge_directed_array: str | None = Field(alias="edgeDirectedArray", default=None)
    edge_storage: Literal["coo"] = Field(alias="edgeStorage", default="coo")
    self_loop_policy: Literal["preserve", "remove"] = Field(alias="selfLoopPolicy")
    duplicate_edge_policy: Literal["preserve", "deduplicate_sorted"] = Field(
        alias="duplicateEdgePolicy"
    )
    weighted: bool = False
    temporal: bool = False
    heterogeneous: bool = False


class GraphVariant(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    edge_index_array: str = Field(alias="edgeIndexArray", min_length=1, max_length=4096)
    feature_array: str | None = Field(alias="featureArray", default=None, max_length=4096)
    directed: bool


class FeatureRecipe(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    graph_variant: str = Field(alias="graphVariant", min_length=1, max_length=200)
    input_array: str | None = Field(alias="inputArray", default=None, max_length=4096)
    output_array: str | None = Field(alias="outputArray", default=None, max_length=4096)
    feature_transform: str = Field(alias="featureTransform", min_length=1, max_length=200)
    fit_scope: Literal["none", "train_only", "all_nodes_transductive"] = Field(
        alias="fitScope", default="none"
    )
    parameters: dict[str, Any] = Field(default_factory=dict)


class SplitFoldCounts(DatasetModel):
    train: int = Field(ge=0)
    validation: int = Field(ge=0)
    test: int = Field(ge=0)


class SplitSet(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    kind: Literal["official", "published", "source", "few_shot"]
    target: Literal["node", "edge", "graph"]
    representation: Literal["mask", "index"]
    arrays: dict[Literal["train", "validation", "test"], str]
    fold_count: int = Field(alias="foldCount", ge=1)
    fold_counts: list[SplitFoldCounts] = Field(alias="foldCounts", default_factory=list)
    seed: int | None = None
    source: str | None = Field(default=None, max_length=4096)


class TaskSpec(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    kind: Literal["node_classification", "link_prediction"]
    target: Literal["node", "edge"]
    label_schema_id: str | None = Field(alias="labelSchemaId", default=None, max_length=200)
    split_set_ids: list[str] = Field(alias="splitSetIds", default_factory=list)
    evaluation_protocol: Literal["transductive", "inductive", "temporal"] = Field(
        alias="evaluationProtocol"
    )
    metrics: list[str] = Field(default_factory=list)
    link_prediction_protocol: LinkPredictionProtocol | None = Field(
        alias="linkPredictionProtocol", default=None
    )


class LinkPredictionProtocol(DatasetModel):
    """Immutable temporal link-prediction inputs; no executable sampler code."""

    message_passing_edge_array: str = Field(alias="messagePassingEdgeArray")
    train_positive_array: str = Field(alias="trainPositiveArray")
    validation_positive_array: str = Field(alias="validationPositiveArray")
    test_positive_array: str = Field(alias="testPositiveArray")
    validation_negative_array: str | None = Field(alias="validationNegativeArray", default=None)
    test_negative_array: str | None = Field(alias="testNegativeArray", default=None)
    edge_year_array: str | None = Field(alias="edgeYearArray", default=None)
    edge_weight_array: str | None = Field(alias="edgeWeightArray", default=None)
    train_year_max: int | None = Field(alias="trainYearMax", default=None)
    validation_year: int | None = Field(alias="validationYear", default=None)
    test_year: int | None = Field(alias="testYear", default=None)
    negative_sampler: Literal["stored", "ogb_official"] = Field(
        alias="negativeSampler", default="stored"
    )
    undirected_canonicalization: Literal["min_max"] = Field(
        alias="undirectedCanonicalization", default="min_max"
    )
    reverse_edge_leakage_policy: Literal["reject"] = Field(
        alias="reverseEdgeLeakagePolicy", default="reject"
    )
    positive_overlap_policy: Literal["reject", "allow_temporal_recurrence"] = Field(
        alias="positiveOverlapPolicy", default="reject"
    )
    evaluator: str = Field(min_length=1, max_length=200)
    evaluator_version: str = Field(alias="evaluatorVersion", min_length=1, max_length=100)


class LicenseEvidence(DatasetModel):
    id: str = Field(min_length=1, max_length=200)
    kind: Literal["official_metadata", "official_license", "user_attestation"]
    source_url: str | None = Field(alias="sourceUrl", default=None, max_length=4096)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    recorded_at: datetime = Field(alias="recordedAt")
    recorded_by: str = Field(alias="recordedBy", min_length=1, max_length=200)


class DataGovernancePolicy(DatasetModel):
    contains_personal_data: bool = Field(alias="containsPersonalData", default=False)
    deidentified: bool = False
    attribute_allowlist: list[str] = Field(alias="attributeAllowlist", default_factory=list)
    excluded_attributes: list[str] = Field(alias="excludedAttributes", default_factory=list)
    retention: Literal["session", "project", "research_archive"] = "project"
    user_data_training_opt_in: Literal[False] = Field(alias="userDataTrainingOptIn", default=False)


class DatasetPreparationSpec(DatasetModel):
    schema_version: Literal["1.0"] = Field(alias="schemaVersion", default="1.0")
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    feature_attributes: list[str] = Field(alias="featureAttributes", default_factory=list)
    label_attribute: str | None = Field(alias="labelAttribute", default=None, max_length=1000)
    task_kind: Literal["none", "node_classification", "link_prediction"] = Field(
        alias="taskKind", default="none"
    )
    split_strategy: Literal["none", "provided", "temporal"] = Field(
        alias="splitStrategy", default="none"
    )
    excluded_attributes: list[str] = Field(alias="excludedAttributes", default_factory=list)
    deidentify: bool = True
    governance: DataGovernancePolicy = Field(default_factory=DataGovernancePolicy)

    @model_validator(mode="after")
    def validate_preparation_boundaries(self) -> DatasetPreparationSpec:
        features = set(self.feature_attributes)
        excluded = set(self.excluded_attributes) | set(self.governance.excluded_attributes)
        if features.intersection(excluded):
            raise ValueError("featureAttributes 与 excludedAttributes 不能重叠")
        if self.label_attribute and self.label_attribute in excluded:
            raise ValueError("labelAttribute 不能同时被排除")
        if self.label_attribute and self.label_attribute in features:
            raise ValueError("标签字段不能同时作为输入特征")
        if self.task_kind == "node_classification" and not self.label_attribute:
            raise ValueError("节点分类准备必须指定 labelAttribute")
        if self.task_kind == "link_prediction" and self.label_attribute is not None:
            raise ValueError("链路预测不使用节点 labelAttribute")
        return self


class GraphHandoffReserveRequest(DatasetModel):
    graph_version_id: str = Field(alias="graphVersionId", min_length=1, max_length=200)
    graph_fact_hash: str = Field(alias="graphFactHash", pattern=r"^[0-9a-f]{64}$")


class GraphHandoffReservation(DatasetModel):
    token: str
    graph_version_id: str = Field(alias="graphVersionId")
    graph_fact_hash: str = Field(alias="graphFactHash")
    expires_at: datetime = Field(alias="expiresAt")


class GraphHandoffCancelRequest(DatasetModel):
    token: str = Field(min_length=20, max_length=512)


class GraphHandoffCancellation(DatasetModel):
    cancelled: Literal[True] = True


class GraphDatasetHandoffRequest(DatasetModel):
    token: str = Field(min_length=20, max_length=512)
    envelope: GraphVersionTargetDomainEnvelope
    preparation: DatasetPreparationSpec
    intended_use: Literal["dataset", "gfm_research"] = Field(
        alias="intendedUse", default="dataset"
    )


class GraphDatasetBinding(DatasetModel):
    id: str
    graph_version_id: str = Field(alias="graphVersionId")
    graph_fact_hash: str = Field(alias="graphFactHash")
    artifact_id: str = Field(alias="artifactId")
    preparation_hash: str = Field(alias="preparationHash")
    created_at: datetime = Field(alias="createdAt")


class LicensePolicy(DatasetModel):
    status: Literal["verified", "user_attested", "restricted", "unknown"]
    identifier: str = Field(default="unknown", max_length=500)
    source_url: str | None = Field(alias="sourceUrl", default=None, max_length=4096)
    allowed_uses: list[Literal["evaluation", "adaptation", "inference", "pretraining"]] = Field(
        alias="allowedUses", default_factory=list
    )
    attribution: str | None = Field(default=None, max_length=4000)
    evidence_ids: list[str] = Field(alias="evidenceIds", default_factory=list)


class TrainingDatasetRef(DatasetModel):
    """Immutable reference consumed by future trainers instead of raw upload paths."""

    schema_version: Literal["1.0", "1.1"] = Field(alias="schemaVersion", default="1.0")
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=200)
    content_hash: str = Field(alias="contentHash", pattern=r"^[0-9a-f]{64}$")
    graph_variant: str = Field(alias="graphVariant", min_length=1, max_length=100)
    split_set_id: str | None = Field(alias="splitSetId", default=None, max_length=200)
    split_fold: int | None = Field(alias="splitFold", default=None, ge=0)
    feature_recipe_id: str = Field(alias="featureRecipeId", min_length=1, max_length=200)
    task_spec_id: str | None = Field(alias="taskSpecId", default=None, max_length=200)
    dataset_role: Literal["benchmark", "target_domain", "pretraining_candidate"] = Field(
        alias="datasetRole", default="target_domain"
    )
    intended_use: Literal["evaluation", "adaptation", "inference", "pretraining"] = Field(
        alias="intendedUse", default="evaluation"
    )
    ref_hash: str = Field(alias="refHash", default="", pattern=r"^$|^[0-9a-f]{64}$")


class DatasetArtifact(DatasetModel):
    schema_version: Literal["1.0", "2.0", "2.1", "2.2"] = Field(
        alias="schemaVersion", default="1.0"
    )
    id: str
    inspection_id: str = Field(alias="inspectionId")
    source_format: str = Field(alias="sourceFormat")
    source_files: list[str] = Field(alias="sourceFiles")
    checksum: str
    profile: DatasetProfile
    graph_view: ArtifactGraphView = Field(alias="graphView")
    dataset_name: str | None = Field(alias="datasetName", default=None)
    canonical_graph_hash: str = Field(alias="canonicalGraphHash", default="")
    content_hash: str = Field(alias="contentHash", default="")
    manifest_hash: str = Field(alias="manifestHash", default="")
    dataset_role: Literal["benchmark", "target_domain", "pretraining_candidate"] = Field(
        alias="datasetRole", default="target_domain"
    )
    source_file_digests: list[SourceFileDigest] = Field(alias="sourceFileDigests", default_factory=list)
    arrays: list[ArrayDescriptor] = Field(default_factory=list)
    node_identity: NodeIdentitySchema | None = Field(alias="nodeIdentity", default=None)
    graph_semantics: GraphSemantics | None = Field(alias="graphSemantics", default=None)
    graph_variants: list[GraphVariant] = Field(alias="graphVariants", default_factory=list)
    feature_schemas: list[FeatureSchema] = Field(alias="featureSchemas", default_factory=list)
    label_schemas: list[LabelSchema] = Field(alias="labelSchemas", default_factory=list)
    feature_recipes: list[FeatureRecipe] = Field(alias="featureRecipes", default_factory=list)
    split_sets: list[SplitSet] = Field(alias="splitSets", default_factory=list)
    task_specs: list[TaskSpec] = Field(alias="taskSpecs", default_factory=list)
    license_policy: LicensePolicy | None = Field(alias="licensePolicy", default=None)
    license_evidence: list[LicenseEvidence] = Field(alias="licenseEvidence", default_factory=list)
    data_governance: DataGovernancePolicy | None = Field(alias="dataGovernance", default=None)
    preparation_spec: DatasetPreparationSpec | None = Field(alias="preparationSpec", default=None)
    training_ref: TrainingDatasetRef | None = Field(alias="trainingRef", default=None)
    training_refs: list[TrainingDatasetRef] = Field(alias="trainingRefs", default_factory=list)
    scope: Literal["complete", "projection"] = "complete"
    raw_manifest: dict[str, Any] = Field(alias="rawManifest", default_factory=dict)
    derived_manifest: dict[str, Any] = Field(alias="derivedManifest", default_factory=dict)
    created_at: datetime = Field(alias="createdAt")

    @model_validator(mode="after")
    def validate_training_identity(self) -> DatasetArtifact:
        if self.schema_version == "1.0":
            return self
        if re.fullmatch(r"[0-9a-f]{64}", self.content_hash) is None:
            raise ValueError("DatasetArtifact v2 contentHash 必须是 64 位小写十六进制")
        references = [*self.training_refs]
        if self.training_ref is not None and all(
            reference.ref_hash != self.training_ref.ref_hash or not reference.ref_hash
            for reference in references
        ):
            references.append(self.training_ref)
        if self.schema_version == "2.0" and not references:
            raise ValueError("DatasetArtifact v2 至少需要一个 trainingRef")
        if any(reference.content_hash != self.content_hash for reference in references):
            raise ValueError("trainingRef contentHash 必须与 DatasetArtifact 一致")
        if self.schema_version in {"2.1", "2.2"}:
            if re.fullmatch(r"[0-9a-f]{64}", self.manifest_hash) is None:
                raise ValueError("DatasetArtifact 2.1/2.2 manifestHash 必须是 SHA-256")
            if self.node_identity is None or self.graph_semantics is None or self.license_policy is None:
                raise ValueError("DatasetArtifact 2.1/2.2 缺少节点身份、图语义或许可证合同")
            if any(reference.artifact_id != self.id for reference in references):
                raise ValueError("trainingRef artifactId 必须与 DatasetArtifact 一致")
            if any(reference.dataset_role != self.dataset_role for reference in references):
                raise ValueError("trainingRef datasetRole 必须与 DatasetArtifact 一致")
            if self.schema_version == "2.2" and self.data_governance is None:
                raise ValueError("DatasetArtifact 2.2 缺少 DataGovernancePolicy")
        return self


class DatasetArtifactRef(DatasetModel):
    schema_version: Literal["1.0", "2.0", "2.1", "2.2"] = Field(
        alias="schemaVersion", default="1.0"
    )
    id: str
    dataset_name: str | None = Field(alias="datasetName", default=None)
    checksum: str
    canonical_graph_hash: str = Field(alias="canonicalGraphHash")
    content_hash: str = Field(alias="contentHash", default="")
    manifest_hash: str = Field(alias="manifestHash", default="")
    dataset_role: Literal["benchmark", "target_domain", "pretraining_candidate"] = Field(
        alias="datasetRole", default="target_domain"
    )
    readiness_status: Literal["legacy", "unchecked"] = Field(alias="readinessStatus", default="legacy")
    scope: Literal["complete", "projection"]
    profile: DatasetProfile
    created_at: datetime = Field(alias="createdAt")
    lifecycle: Literal["active", "trashed"] = "active"


class ResourceLifecycle(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    status: Literal["active", "trashed"] = "active"
    updated_at: datetime = Field(alias="updatedAt")
    trashed_at: datetime | None = Field(alias="trashedAt", default=None)


class ArtifactReference(DatasetModel):
    kind: Literal["graph_dataset_binding", "embedded_training_ref"]
    id: str
    blocking: bool
    detail: dict[str, Any] = Field(default_factory=dict)


class DatasetArtifactDeletionImpact(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    lifecycle: Literal["active", "trashed"]
    blockers: list[ArtifactReference] = Field(default_factory=list)
    dependents: list[ArtifactReference] = Field(default_factory=list)
    preserved: list[str] = Field(default_factory=list)
    impact_hash: str = Field(alias="impactHash", pattern=r"^[0-9a-f]{64}$")


class DatasetArtifactLifecycleResponse(DatasetModel):
    lifecycle: ResourceLifecycle
    impact: DatasetArtifactDeletionImpact


class DatasetArtifactPurgeRequest(DatasetModel):
    impact_hash: str = Field(alias="impactHash", pattern=r"^[0-9a-f]{64}$")
    confirmation: str = Field(min_length=1, max_length=200)


class DatasetArtifactPurgeResponse(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    purged: Literal[True] = True
    cleanup_pending: bool = Field(alias="cleanupPending", default=False)


class OrphanArtifactDirectory(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    source: Literal["artifacts", "purge_recovery"]
    relative_path: str = Field(alias="relativePath")
    recoverable: bool
    reason: str | None = None


class OrphanArtifactRecoveryResponse(DatasetModel):
    artifact: DatasetArtifact
    lifecycle: ResourceLifecycle


class DatasetArtifactRowIssue(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    code: Literal["ARTIFACT_ROW_INVALID"]


class DatasetStoreDiagnostics(DatasetModel):
    isolated_artifact_rows: list[DatasetArtifactRowIssue] = Field(
        alias="isolatedArtifactRows", default_factory=list
    )
    trashed_artifact_ids: list[str] = Field(alias="trashedArtifactIds", default_factory=list)
    orphan_artifacts: list[OrphanArtifactDirectory] = Field(
        alias="orphanArtifacts", default_factory=list
    )


class DatasetReadinessIssue(DatasetModel):
    code: str
    message: str
    severity: Literal["blocker", "warning"] = "blocker"


class DatasetReadiness(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    status: Literal["ready", "blocked", "legacy", "corrupt"]
    content_hash: str = Field(alias="contentHash", default="")
    manifest_hash: str = Field(alias="manifestHash", default="")
    training_ref: TrainingDatasetRef | None = Field(alias="trainingRef", default=None)
    blockers: list[DatasetReadinessIssue] = Field(default_factory=list)
    warnings: list[DatasetReadinessIssue] = Field(default_factory=list)
    checked_at: datetime = Field(alias="checkedAt")


class MaterializedDatasetBundle(DatasetModel):
    artifact_id: str = Field(alias="artifactId")
    training_ref_hash: str = Field(alias="trainingRefHash")
    node_count: int = Field(alias="nodeCount", ge=0)
    edge_count: int = Field(alias="edgeCount", ge=0)
    feature_shape: list[int] = Field(alias="featureShape")
    label_shape: list[int] | None = Field(alias="labelShape", default=None)
    split_sizes: dict[str, int] = Field(alias="splitSizes")
    task_kind: Literal["node_classification", "link_prediction"] = Field(alias="taskKind")


class GraphDatasetHandoffResponse(DatasetModel):
    binding: GraphDatasetBinding
    artifact: DatasetArtifact
    reused: bool = False
    research_compatibility: ResearchGraphCompatibility | None = Field(
        alias="researchCompatibility", default=None
    )


class TrainingRefResolveRequest(DatasetModel):
    artifact_id: str = Field(alias="artifactId", min_length=1, max_length=200)
    content_hash: str = Field(alias="contentHash", pattern=r"^[0-9a-f]{64}$")
    graph_variant: str = Field(alias="graphVariant", min_length=1, max_length=100)
    split_set_id: str = Field(alias="splitSetId", min_length=1, max_length=200)
    split_fold: int = Field(alias="splitFold", default=0, ge=0)
    feature_recipe_id: str = Field(alias="featureRecipeId", min_length=1, max_length=200)
    task_spec_id: str = Field(alias="taskSpecId", min_length=1, max_length=200)
    intended_use: Literal["evaluation", "adaptation", "inference", "pretraining"] = Field(
        alias="intendedUse", default="evaluation"
    )


class TrainingRefResolveResponse(DatasetModel):
    reference: TrainingDatasetRef
    readiness: DatasetReadiness


class TrustedLocalInspectRequest(DatasetModel):
    source_path: str = Field(alias="sourcePath", min_length=1, max_length=4096)


class TrustedConversionAuthorizeRequest(DatasetModel):
    authorization_token: str = Field(alias="authorizationToken", min_length=20, max_length=512)
    confirm_trusted: Literal[True] = Field(alias="confirmTrusted")


class TrustedDiscoveredDataset(DatasetModel):
    name: str
    detected_format: str = Field(alias="detectedFormat")
    file_count: int = Field(alias="fileCount", ge=0)


class TrustedConversionJob(DatasetModel):
    id: str
    source_path: str = Field(alias="sourcePath")
    trusted_root: str = Field(alias="trustedRoot")
    status: Literal[
        "awaiting_authorization",
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    ]
    progress: int = Field(ge=0, le=100)
    file_count: int = Field(alias="fileCount", ge=0)
    total_bytes: int = Field(alias="totalBytes", ge=0)
    datasets: list[TrustedDiscoveredDataset] = Field(default_factory=list)
    artifact_ids: list[str] = Field(alias="artifactIds", default_factory=list)
    issues: list[DatasetIssue] = Field(default_factory=list)
    converter_python: str = Field(alias="converterPython")
    created_at: datetime = Field(alias="createdAt")
    updated_at: datetime = Field(alias="updatedAt")


class TrustedLocalInspection(TrustedConversionJob):
    authorization_token: str = Field(alias="authorizationToken")
