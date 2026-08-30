import type { JsonPrimitive } from "./graph";

export type CoreTaskId =
  | "core.community_resilience_review"
  | "core.risk_and_trust_review"
  | "core.collaboration_completion";

export type CoreEntityType = "node" | "edge" | "node-pair" | "community";
export type CoreModelState = "accepted" | "servingReady";
export type CoreConfidenceKind = "binary-calibration" | "regression-interval";

export interface CoreTaskEntityCapability {
  readonly taskId: CoreTaskId;
  readonly entityType: CoreEntityType;
  readonly confidenceKind: CoreConfidenceKind;
  readonly calibrationVersion: string;
  readonly method: "sigmoid" | "validation-residual-interval";
  readonly calibrationArtifactHash: string;
  readonly calibrationProtocolHash: string;
  readonly adapterDomain: string;
  readonly adapterSchemaHash: string;
  readonly adapterStateHash: string;
  readonly featureContractHash: string;
}

export interface CoreModelCapability {
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly state: CoreModelState;
  readonly tasks: readonly CoreTaskId[];
  readonly graphSchemaVersions: readonly string[];
  readonly graphFeatureContractHash: string;
  readonly taskBindings: readonly CoreTaskEntityCapability[];
  readonly maxNodes: number;
  readonly maxEdges: number;
}

export interface CoreCapabilities {
  readonly schemaVersion: "socialgraph-fm.core-capabilities/2.0";
  readonly registryHash: string;
  readonly registryGeneration: number;
  readonly controlHash?: string | null;
  readonly controlGeneration?: number | null;
  readonly catalogHash?: string | null;
  readonly catalogGeneration?: number | null;
  readonly servingReady: boolean;
  readonly models: readonly CoreModelCapability[];
  readonly tasks: readonly CoreTaskId[];
  readonly readiness: {
    readonly modelValidated: boolean;
    readonly coreServingReady: boolean;
  };
}

export interface CoreCommunityTargetScope {
  readonly kind: "community";
  readonly communityIds: readonly string[];
}

export interface CoreNodeRiskTargetScope {
  readonly kind: "risk-review";
  readonly nodeIds: readonly [string, ...string[]];
  readonly edgeIds: readonly [];
}

export interface CoreEdgeRiskTargetScope {
  readonly kind: "risk-review";
  readonly nodeIds: readonly [];
  readonly edgeIds: readonly [string, ...string[]];
}

export type CoreRiskTargetScope = CoreNodeRiskTargetScope | CoreEdgeRiskTargetScope;

export interface CoreCollaborationTargetScope {
  readonly kind: "node-pairs";
  readonly pairs: readonly (readonly [string, string])[];
}

export type CoreTargetScope =
  | CoreCommunityTargetScope
  | CoreRiskTargetScope
  | CoreCollaborationTargetScope;

export interface CoreCommunityParameters {
  readonly kind: "community-resilience";
  readonly topKSimilarCases: number;
}

export interface CoreRiskParameters {
  readonly kind: "risk-and-trust";
  readonly topKSimilarCases: number;
}

export interface CoreCollaborationParameters {
  readonly kind: "collaboration-completion";
  readonly topKSimilarCases: number;
  readonly candidateLimit: number;
}

export type CoreRunParameters =
  | CoreCommunityParameters
  | CoreRiskParameters
  | CoreCollaborationParameters;

export interface CoreRunRequest {
  readonly schemaVersion: "socialgraph-fm.core-run-request/2.0";
  readonly graphVersionId: string;
  readonly taskId: CoreTaskId;
  readonly targetScope: CoreTargetScope;
  readonly modelVersionId: string;
  readonly parameters: CoreRunParameters;
}

export type CoreRunStatusValue = "queued" | "running" | "succeeded" | "failed";

export interface CoreRunStatus {
  readonly schemaVersion: "socialgraph-fm.core-run-status/2.0";
  readonly runId: string;
  readonly requestHash: string;
  readonly status: CoreRunStatusValue;
  readonly progress: number;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly errorCode: string | null;
  readonly stateHash: string;
}

export interface CoreRegisteredEdgeIdentity {
  readonly schemaVersion: "socialgraph-fm.core-edge-identity/2.0";
  readonly sourceId: string;
  readonly targetId: string;
  readonly edgeType: string;
  readonly weight: number;
  readonly edgeHash: string;
}

export interface CoreModelScore {
  readonly schemaVersion: "socialgraph-fm.core-model-score/2.0";
  readonly taskId: CoreTaskId;
  readonly entityType: CoreEntityType;
  readonly entityIds: readonly string[];
  readonly score: number;
  readonly graphVersionHash: string;
  readonly modelVersion: string;
  readonly modelVersionHash: string;
  readonly edgeIdentity: CoreRegisteredEdgeIdentity | null;
  readonly scoreHash: string;
}

export interface CoreCalibratedConfidence {
  readonly schemaVersion: "socialgraph-fm.core-calibrated-confidence/2.0";
  readonly value: number;
  readonly scoreHash: string;
  readonly taskId: CoreTaskId;
  readonly entityType: CoreEntityType;
  readonly entityIds: readonly string[];
  readonly graphVersionHash: string;
  readonly modelVersion: string;
  readonly modelVersionHash: string;
  readonly calibrationVersion: string;
  readonly method: string;
  readonly calibrationArtifactHash: string;
  readonly calibrationProtocolHash: string;
  readonly confidenceHash: string;
}

export interface CoreRegressionConfidenceInterval {
  readonly schemaVersion: "socialgraph-fm.core-regression-confidence-interval/1.0";
  readonly pointEstimate: number;
  readonly lowerBound: number;
  readonly upperBound: number;
  readonly coverage: number;
  readonly validationCount: number;
  readonly scoreHash: string;
  readonly taskId: "core.community_resilience_review";
  readonly entityType: "community";
  readonly entityIds: readonly string[];
  readonly graphVersionHash: string;
  readonly modelVersion: string;
  readonly modelVersionHash: string;
  readonly confidenceVersion: string;
  readonly method: "validation-residual-interval";
  readonly confidenceArtifactHash: string;
  readonly confidenceProtocolHash: string;
  readonly confidenceHash: string;
}

export type CoreConfidenceEvidence =
  | CoreCalibratedConfidence
  | CoreRegressionConfidenceInterval;

export type CoreEvidenceJsonValue = JsonPrimitive | readonly CoreEvidenceJsonValue[] | CoreEvidenceJsonObject;
export interface CoreEvidenceJsonObject {
  readonly [key: string]: CoreEvidenceJsonValue;
}
export type CoreEvidenceValue = CoreEvidenceJsonObject;

export interface CoreEvidenceItem {
  readonly schemaVersion: "socialgraph-fm.core-evidence/2.0";
  readonly metric: string;
  readonly valueCanonicalJson: string;
  readonly graphVersionHash: string;
  readonly sourceType: "deterministic-graph-algorithm" | "registered-model-output";
  readonly nodeIds: readonly string[];
  readonly edgeIds: readonly string[];
  readonly algorithmConfigHash: string | null;
  readonly modelVersionHash: string | null;
  readonly modelVersion: string | null;
  readonly modelScoreHash: string | null;
  readonly modelTaskId: CoreTaskId | null;
  readonly modelEntityType: CoreEntityType | null;
  readonly modelEntityIds: readonly string[] | null;
  readonly limitations: readonly string[];
  readonly evidenceHash: string;
}

export interface CoreSimilarCase {
  readonly schemaVersion: "socialgraph-fm.core-similar-case/2.0";
  readonly structuralRecordHash: string;
  readonly similarity: number;
  readonly sourceGraphVersionHash: string;
  readonly sourceEntityIds: readonly string[];
  readonly sourceKind: "node" | "ego" | "community";
  readonly modelVersion: string;
  readonly modelVersionHash: string;
  readonly representation: "embedding" | "motif-signature";
  readonly queryHash: string;
  readonly representationSchema: "socialgraph-fm.core-structural-record/2.0";
  readonly similarCaseHash: string;
}

export type CoreFindingType =
  | "community-resilience-candidate"
  | "node-risk-candidate"
  | "signed-relation-review"
  | "core-collaboration-completion";

export interface CoreFinding {
  readonly schemaVersion: "socialgraph-fm.core-finding/2.0";
  readonly taskId: CoreTaskId;
  readonly findingType: CoreFindingType;
  readonly subjectIds: readonly string[];
  readonly score: CoreModelScore;
  readonly calibratedConfidence: CoreConfidenceEvidence;
  readonly evidence: readonly CoreEvidenceItem[];
  readonly similarCases: readonly CoreSimilarCase[];
  readonly graphVersionHash: string;
  readonly modelVersion: string;
  readonly modelVersionHash: string;
  readonly limitations: readonly string[];
  readonly reviewStatus: "pending-human-review";
  readonly findingHash: string;
}

export interface CoreRunResult {
  readonly schemaVersion: "socialgraph-fm.core-run-result/2.0";
  readonly runId: string;
  readonly requestHash: string;
  readonly taskId: CoreTaskId;
  readonly graphVersionId: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly findings: readonly CoreFinding[];
  readonly completedAt: string;
  readonly resultHash: string;
}

/** Browser-verifiable request identity plus the opaque server-envelope identity. */
export interface CoreRunBinding {
  readonly runId: string;
  readonly publicRequestHash: string;
  readonly serverRequestHash: string;
  readonly taskId: CoreTaskId;
  readonly graphVersionId: string;
  readonly modelVersionId: string;
}

export interface CoreCreatedRun {
  readonly status: CoreRunStatus;
  readonly binding: CoreRunBinding;
}

export interface CoreClientLike {
  capabilities(signal?: AbortSignal): Promise<CoreCapabilities>;
  createRun(request: CoreRunRequest, signal?: AbortSignal): Promise<CoreCreatedRun>;
  getRun(runId: string, binding: CoreRunBinding, signal?: AbortSignal): Promise<CoreRunStatus>;
  getResult(runId: string, binding: CoreRunBinding, signal?: AbortSignal): Promise<CoreRunResult>;
}

export type CoreWorkbenchServiceState =
  | { readonly state: "checking" }
  | { readonly state: "connected"; readonly capabilities: CoreCapabilities }
  | { readonly state: "unavailable"; readonly code: string };
