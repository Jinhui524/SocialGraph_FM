import { GOVERNANCE_PUBLIC_SKILLS } from "./governanceSkills";

export const GOVERNANCE_ONLINE_SCHEMA = "socialgraph-fm.gfm-governance/2.0" as const;
export const GOVERNANCE_INPUT_SCHEMA = "socialgraph-fm.governance-input/2.0" as const;
export const GOVERNANCE_TARGET_LABEL_RECIPE_SCHEMA = "socialgraph-fm.governance-target-label-recipe/1.1" as const;
export const GOVERNANCE_TARGET_PACKAGE_RECEIPT_SCHEMA = "socialgraph-fm.governance-target-package-receipt/1.1" as const;
export const GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA = "socialgraph-fm.governance-target-label-set/1.1" as const;
export const GOVERNANCE_ADAPTATION_POLICY_SCHEMA = "socialgraph-fm.governance-target-review-policy/1.0" as const;
export const GOVERNANCE_ADAPTATION_COMPARISON_SCHEMA = "socialgraph-fm.governance-adaptation-comparison/1.0" as const;
export const GOVERNANCE_TARGET_TASK_REGISTRATION_SCHEMA = "socialgraph-fm.governance-target-task-registration/1.0" as const;
export const GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA = "socialgraph-fm.governance-target-label-set/2.0" as const;
export const GOVERNANCE_TARGET_POLICY_SCHEMA = "socialgraph-fm.governance-target-review-policy/2.0" as const;
export const GOVERNANCE_TARGET_COMPARISON_SCHEMA = "socialgraph-fm.governance-adaptation-comparison/2.0" as const;
export const GOVERNANCE_ADAPTATION_HANDOFF_SCHEMA = "socialgraph-fm.governance-adaptation-handoff/1.0" as const;
export const GOVERNANCE_ADAPTATION_OVERLAY_SCHEMA = "socialgraph-fm.governance-adaptation-overlay/1.0" as const;
export const GOVERNANCE_RELATION_MODALITIES = ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"] as const;

export type GovernanceOnlineModality = typeof GOVERNANCE_RELATION_MODALITIES[number];
export type GovernanceOnlineRiskBand = "low" | "review" | "high";
export type GovernanceRunStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "interrupted";
export type GovernanceOnlineStage = "queued" | "validating" | "preprocessing" | "inferencing" | "deriving" | "freezing" | "completed";
export type GovernanceTargetKind = "node" | "relation" | "group";
export type GovernanceCaseStatus = "draft" | "active" | "concluded" | "archived";
export type GovernanceReviewDecision = "confirmed" | "rejected" | "pending";
export type GovernanceOnlineExpert = "shared" | "domain:china" | "domain:cuba" | "domain:iran" | "domain:russia" | "domain:UAE" | "domain:venezuela" | "null";

export interface GovernanceOnlineLimits {
  readonly maxNodes: number;
  readonly maxRelationRows: number;
  readonly maxEvidenceNodes: number;
  readonly maxEvidenceEdges: number;
  readonly maxPreviewNodes: number;
  readonly maxPreviewEdges: number;
}

export interface GovernanceOnlineHealth {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly serviceIdentity: string;
  readonly servingReady: boolean;
  readonly onlineForwardReady: boolean;
  readonly modelVersionId: string | null;
  readonly modelVersionHash: string | null;
  readonly modelStateHash: string | null;
  readonly device: "cpu";
  readonly dtype: "float32" | "float16" | "bfloat16";
  readonly loadedAt: string | null;
  readonly queueDepth: number;
  readonly activeRunId: string | null;
  readonly runtimeRecipeHash: string;
  readonly healthHash: string;
}

export interface GovernanceOnlineCapabilities {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly channel: "governance";
  readonly taskId: "coordination_risk";
  readonly servingReady: boolean;
  readonly onlineForwardReady: boolean;
  readonly unavailableReason: string | null;
  readonly modelVersionId: string | null;
  readonly modelVersionHash: string | null;
  readonly modelStateHash: string | null;
  readonly supportedProtocols: readonly ["global"];
  readonly skills: typeof GOVERNANCE_PUBLIC_SKILLS;
  readonly inputSchemaVersion: typeof GOVERNANCE_INPUT_SCHEMA;
  readonly modalities: typeof GOVERNANCE_RELATION_MODALITIES;
  readonly sampleArtifactId: string | null;
  readonly limits: GovernanceOnlineLimits;
  readonly capabilityHash: string;
}

export interface GovernanceArtifact {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly artifactId: string;
  readonly datasetId?: string;
  readonly displayName?: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly bundleSha256?: string;
  readonly manifestSha256?: string;
  readonly nodeCount: number;
  readonly relationRowCount: number;
  readonly selfLoopsRemoved: number;
  readonly cleanSelfLoops?: boolean;
  readonly modalities: readonly GovernanceOnlineModality[];
  readonly compatibility: "compatible";
  readonly createdAt: string;
  readonly artifactHash: string;
}

export interface GovernanceArtifactCompatibility {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly inputSchemaVersion: typeof GOVERNANCE_INPUT_SCHEMA;
  readonly compatible: boolean;
  readonly requiresSelfLoopCleaning: boolean;
  readonly prospectiveArtifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly nodeCount: number;
  readonly relationRowCount: number;
  readonly selfLoopsDetected: number;
  readonly modalities: readonly GovernanceOnlineModality[];
  readonly issues: readonly string[];
  readonly compatibilityHash: string;
}

export interface GovernanceOnlinePreviewNode {
  readonly id: string;
  readonly label: string;
  readonly degree: number;
  readonly structureMissing: boolean;
  readonly score: number | null;
  readonly riskBand: GovernanceOnlineRiskBand | null;
  readonly groupId: string | null;
}

export interface GovernanceOnlinePreviewEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly modalities: readonly GovernanceOnlineModality[];
  readonly factual: boolean;
}

export type GovernanceProjectionPreset = "overview" | "relation" | "evidence" | "groups";

export interface GovernanceProjectionRequest {
  readonly preset: GovernanceProjectionPreset;
  readonly nodeBudget?: number;
  readonly edgeBudget?: number;
  readonly relation?: GovernanceOnlineModality;
  readonly anchorNodeIds?: readonly string[];
  readonly groupBudget?: number;
}

export interface GovernanceOnlinePreview {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly runId: string | null;
  readonly resultHash: string | null;
  readonly nodes: readonly GovernanceOnlinePreviewNode[];
  readonly edges: readonly GovernanceOnlinePreviewEdge[];
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly partialPreview: boolean;
  readonly previewHash: string;
  readonly preset?: GovernanceProjectionPreset;
  readonly budgets?: Readonly<Record<string, number>>;
  readonly selectionRecipeId?: string;
  readonly isPartial?: boolean;
  readonly groups?: readonly Readonly<Record<string, unknown>>[];
  readonly sourceCounts?: Readonly<Record<string, number>>;
  readonly inventoryCounts?: Readonly<Record<string, number>>;
}

export interface GovernanceOnlineRunRequest {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly protocol: "global";
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelStateHash: string;
  readonly topK: number;
}

export interface GovernanceOnlineRun {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly runId: string;
  readonly requestHash: string;
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly modelStateHash: string;
  readonly status: GovernanceRunStatus;
  readonly stage: GovernanceOnlineStage;
  readonly progress: number;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly errorCode: string | null;
  readonly cancelRequested: boolean;
  readonly statusHash: string;
}

export interface GovernanceRunComparisonNode {
  readonly nodeId: string;
  readonly leftScore: number;
  readonly rightScore: number;
  readonly scoreDelta: number;
  readonly leftRank: number;
  readonly rightRank: number;
  readonly rankDelta: number;
  readonly riskBandChanged: boolean;
}
export interface GovernanceRunComparison {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly leftRunId: string;
  readonly rightRunId: string;
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly comparedNodes: number;
  readonly changes: readonly GovernanceRunComparisonNode[];
  readonly groupSummary: Readonly<Record<string, number>>;
  readonly reviewSummary: Readonly<Record<string, number>>;
  readonly comparisonHash: string;
}

export interface GovernanceOnlineRoute { readonly expert: GovernanceOnlineExpert; readonly weight: number; }
export type GovernanceModalityCounts = Readonly<Partial<Record<GovernanceOnlineModality, number>>>;
export interface GovernanceModalityContribution { readonly text: number; readonly structure: number; }

export interface GovernanceOnlineFinding {
  readonly nodeId: string;
  readonly label: string | null;
  readonly score: number;
  readonly logit: number;
  readonly rank: number;
  readonly riskBand: GovernanceOnlineRiskBand;
  readonly predictedPositive: boolean;
  readonly structureMissing: boolean;
  readonly routes: readonly GovernanceOnlineRoute[];
  readonly modalityContribution: GovernanceModalityContribution;
  readonly modalityEvidence: GovernanceModalityCounts;
  readonly communityId: string | null;
}

export interface GovernanceCalibration {
  readonly temperature: number;
  readonly bias: number;
  readonly referenceThreshold: number;
  readonly applicability: "reference_replay" | "out_of_domain_unverified";
}
export interface GovernanceDistribution { readonly low: number; readonly review: number; readonly high: number; readonly predictedPositive: number; readonly total: number; }

export interface GovernanceOnlineResult {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly runId: string;
  readonly requestHash: string;
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly modelStateHash: string;
  readonly threshold: number;
  readonly calibration: GovernanceCalibration;
  readonly referenceMetrics: Readonly<Record<string, unknown>>;
  readonly datasetMetrics: null;
  readonly distribution: GovernanceDistribution;
  readonly findings: readonly GovernanceOnlineFinding[];
  readonly totalFindings: number;
  readonly limitations: readonly string[];
  readonly completedAt: string;
  readonly resultHash: string;
}

export interface GovernanceFindingPage {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly runId: string;
  readonly items: readonly GovernanceOnlineFinding[];
  readonly total: number;
  readonly offset: number;
  readonly limit: number;
  readonly pageHash: string;
}

export interface GovernanceDerivation {
  readonly id: string;
  readonly kind: "group" | "factual_relation" | "potential_link";
  readonly priority: number;
  readonly nodeIds: readonly string[];
  readonly source: string | null;
  readonly target: string | null;
  readonly modalities: readonly GovernanceOnlineModality[];
  readonly memberCount: number | null;
  readonly meanScore: number | null;
  readonly p90Score: number | null;
  readonly scoreComponents: Readonly<Record<string, number>>;
  readonly factual: boolean;
  readonly limitation: string;
}
export interface GovernanceDerivationPage {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly runId: string;
  readonly items: readonly GovernanceDerivation[];
  readonly total: number;
  readonly offset: number;
  readonly limit: number;
  readonly pageHash: string;
}

export interface GovernanceEvidenceNeighbor {
  readonly nodeId: string;
  readonly score: number;
  readonly hop: 1;
  readonly riskBand: GovernanceOnlineRiskBand;
  readonly predictedPositive: boolean;
  readonly structureMissing: boolean;
  readonly modalities: readonly GovernanceOnlineModality[];
  readonly relations: readonly GovernanceEvidenceRelation[];
}
export interface GovernanceEvidenceRelation { readonly modality: GovernanceOnlineModality; readonly rawWeight: number; }
export interface GovernanceEvidenceStructuralSignals {
  readonly fusedDegree: number;
  readonly structureMissing: boolean;
  readonly relationNeighborCounts: Readonly<Record<GovernanceOnlineModality, number>>;
  readonly twoHopNodeCount: number;
  readonly relationEvidenceRole: "explanationOnly";
}
export interface GovernanceEvidenceSubgraphNode {
  readonly nodeId: string;
  readonly score: number;
  readonly hop: 0 | 1 | 2;
  readonly riskBand: GovernanceOnlineRiskBand;
  readonly predictedPositive: boolean;
  readonly structureMissing: boolean;
}
export interface GovernanceEvidenceSubgraphEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly relations: readonly GovernanceEvidenceRelation[];
  readonly evidenceRole: "explanationOnly";
}
export interface GovernanceEvidenceSubgraph {
  readonly depth: 2;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly truncated: boolean;
  readonly nodes: readonly GovernanceEvidenceSubgraphNode[];
  readonly edges: readonly GovernanceEvidenceSubgraphEdge[];
}
export interface GovernanceOnlineEvidence {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly runId: string;
  readonly resultHash: string;
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly modelStateHash: string;
  readonly threshold: number;
  readonly node: GovernanceOnlineFinding;
  readonly neighbors: readonly GovernanceEvidenceNeighbor[];
  readonly structuralSignals: GovernanceEvidenceStructuralSignals;
  readonly evidenceSubgraph: GovernanceEvidenceSubgraph;
  readonly truncated: boolean;
  readonly limitation: string;
  readonly evidenceHash: string;
}

export interface GovernanceCaseItem { readonly itemId: string; readonly targetType: GovernanceTargetKind; readonly targetId: string; readonly note: string; readonly createdAt: string; readonly itemHash: string; }
export interface GovernanceReviewEvent {
  readonly eventId: string; readonly targetType: GovernanceTargetKind; readonly targetId: string;
  readonly decision: GovernanceReviewDecision; readonly reason: string; readonly actor: string;
  readonly sequence: number; readonly createdAt: string; readonly previousEventHash: string | null; readonly eventHash: string;
}
export interface GovernanceCase {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly caseId: string; readonly runId: string; readonly title: string; readonly description: string;
  readonly state: GovernanceCaseStatus; readonly createdAt: string; readonly updatedAt: string;
  readonly items: readonly GovernanceCaseItem[]; readonly reviewEvents: readonly GovernanceReviewEvent[];
  readonly currentDecisions: Readonly<Record<string, GovernanceReviewDecision>>; readonly caseHash: string;
}

export interface GovernanceCasePage {
  readonly schemaVersion: typeof GOVERNANCE_ONLINE_SCHEMA;
  readonly items: readonly GovernanceCase[];
  readonly total: number;
  readonly offset: number;
  readonly limit: number;
}

export interface GovernanceTargetLabelRecipeRow {
  readonly nodeId: string;
  readonly label: "io" | "control";
  readonly structuralStratum: 0 | 1 | 2 | 3;
  readonly fusedDegree: number;
}

export interface GovernanceTargetLabelRecipe {
  readonly schemaVersion: typeof GOVERNANCE_TARGET_LABEL_RECIPE_SCHEMA;
  readonly datasetId: string;
  readonly bundleSha256: string;
  readonly selectionRecipe: {
    readonly version: "graph-fused-degree-quartile-stable-hash-v2";
    readonly stratification: "graph-fused-degree-rank-quartile";
    readonly structuralStrata: 4;
    readonly labelsPerClass: 8;
    readonly labelsPerClassPerStratum: 2;
    readonly scoreInputs: readonly [];
  };
  readonly labels: readonly GovernanceTargetLabelRecipeRow[];
}

export interface GovernanceTargetPackageReceipt {
  readonly schemaVersion: typeof GOVERNANCE_TARGET_PACKAGE_RECEIPT_SCHEMA;
  readonly datasetId: string;
  readonly sourceSchemaVersion: "socialgraph-fm.anonymized-posts/1.0";
  readonly sourceSha256: string;
  readonly authorizationReference: string;
  readonly bundleSha256: string;
  readonly labelsSha256: string;
  readonly encoder: {
    readonly modelId: string;
    readonly revision: string;
    readonly cacheSha256: string;
    readonly compatibility: "dimension-only-unverified" | "pinned-production";
    readonly dimension: 768;
  };
  readonly selectionRecipe: {
    readonly version: "connected-structural-hash-v2";
    readonly nodeCount: 128;
    readonly requiredIo: 16;
    readonly requiredControls: 64;
    readonly minimumNonemptyModalities: 4;
    readonly scoreInputs: readonly [];
    readonly groupRelations: { readonly maxGroupAccounts: 256; readonly totalPotentialPairBudget: 50_000 };
    readonly fastRT: { readonly windowSeconds: 10; readonly pairBudget: 50_000; readonly algorithm: "sorted-sliding-window-v1" };
    readonly tweetSim: { readonly mutualTopK: 5; readonly cosineThreshold: 0.8; readonly pairBudget: 10_000 };
  };
  readonly labelSelectionRecipe: GovernanceTargetLabelRecipe["selectionRecipe"];
  readonly coverage: {
    readonly nodeCount: 128;
    readonly ioCount: number;
    readonly controlCount: number;
    readonly nonemptyModalities: readonly GovernanceOnlineModality[];
    readonly connected: true;
  };
  readonly receiptHash: string;
}

export interface GovernanceAdaptationBinding {
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly runId: string;
  readonly requestHash: string;
  readonly resultHash: string;
  readonly runArtifactHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly modelStateHash: string;
  readonly recipeHash: string;
  readonly codeHash: string;
  readonly seed: number;
}

export interface ImportedSidecarLabelSource {
  readonly sourceType: "imported_sidecar";
  readonly sourceRecordId: string;
  readonly sourceRecordHash: string;
  readonly nodeId: string;
  readonly cohort: "io" | "control";
  readonly structuralStratum: 0 | 1 | 2 | 3;
  readonly fusedDegree: number;
  readonly labelsSha256: string;
  readonly receiptHash: string;
}

export interface ConcludedReviewLabelSource {
  readonly sourceType: "concluded_review";
  readonly caseId: string;
  readonly eventHash: string;
}

export type AdaptationLabelSource = ImportedSidecarLabelSource | ConcludedReviewLabelSource;

export interface AdaptationLabelSetCreateRequest {
  readonly schemaVersion: typeof GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA;
  readonly runId: string;
  readonly resultHash: string;
  readonly sidecarReceipt: GovernanceTargetPackageReceipt;
  readonly sources: readonly AdaptationLabelSource[];
}

export interface AdaptationSourceRecord {
  readonly sourceType: "concluded_review" | "imported_sidecar";
  readonly sourceRecordId: string;
  readonly sourceRecordHash: string;
  readonly reviewEventHash: string | null;
}

export interface AdaptationLabelEvidence extends AdaptationSourceRecord {
  readonly nodeId: string;
  readonly label: "positive" | "negative";
  readonly binding: GovernanceAdaptationBinding;
  readonly structuralStratum?: 0 | 1 | 2 | 3;
  readonly fusedDegree?: number;
  readonly labelsSha256?: string;
  readonly receiptHash?: string;
}

export interface AdaptationLabelSet {
  readonly schemaVersion: typeof GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA;
  readonly binding: GovernanceAdaptationBinding;
  readonly sidecarReceipt: GovernanceTargetPackageReceipt | null;
  readonly sourceRecords: readonly AdaptationSourceRecord[];
  readonly reviewEventHashes: readonly string[];
  readonly labels: readonly AdaptationLabelEvidence[];
  readonly conflicts: readonly string[];
  readonly positiveCount: number;
  readonly negativeCount: number;
  readonly labelSetHash: string;
}

export type ReviewPolicyStatus = "collecting_reviews" | "ready" | "insufficient_signal" | "invalid";

export interface AdaptationReviewPolicy {
  readonly schemaVersion: typeof GOVERNANCE_ADAPTATION_POLICY_SCHEMA;
  readonly binding: GovernanceAdaptationBinding;
  readonly labelSetHash: string;
  readonly status: ReviewPolicyStatus;
  readonly selectedLambda: number;
  readonly lambdaCandidates: readonly [0, 0.25, 0.5, 1];
  readonly validationLosses: Readonly<Record<"0" | "0.25" | "0.5" | "1", number>>;
  readonly eligibleLabelCount: number;
  readonly positiveCount: number;
  readonly negativeCount: number;
  readonly embeddingDimension: 256;
  readonly positiveCentroidHash: string;
  readonly negativeCentroidHash: string;
  readonly normalizationEpsilon: number;
  readonly fittingRecipe: "l2-centroids+run-zscore+loo-balanced-log-loss-v1";
  readonly readyPolicyHash: string | null;
  readonly policyHash: string;
}

export interface AdaptationComparisonRow {
  readonly nodeId: string;
  readonly baseScore: number;
  readonly baseRank: number;
  readonly adaptedReviewPriority: number;
  readonly adaptedRank: number;
  readonly rankDelta: number;
}

export interface AdaptationComparison {
  readonly schemaVersion: typeof GOVERNANCE_ADAPTATION_COMPARISON_SCHEMA;
  readonly binding: GovernanceAdaptationBinding;
  readonly policyHash: string;
  readonly total: number;
  readonly offset: number;
  readonly limit: number;
  readonly rows: readonly AdaptationComparisonRow[];
  readonly comparisonHash: string;
  readonly pageHash: string;
}

export interface TargetTaskRegistration {
  readonly schemaVersion: typeof GOVERNANCE_TARGET_TASK_REGISTRATION_SCHEMA;
  readonly registrationId: string;
  readonly outerBundleSha256: string;
  readonly task: {
    readonly schemaVersion: "socialgraph-fm.governance-target-task-bundle/1.0";
    readonly taskId: string;
    readonly displayName: string;
    readonly mode: "zero_shot" | "few_shot";
    readonly nodeCount: number;
    readonly fusedEdgeCount: number;
    readonly modalities: readonly GovernanceOnlineModality[];
  };
  readonly targetReceipt: {
    readonly schemaVersion: "socialgraph-fm.governance-target-domain-receipt/2.0";
    readonly taskId: string;
    readonly receiptHash: string;
    readonly inferenceSha256: string;
    readonly nodeSetSha256: string;
    readonly nodeCount: number;
    readonly fusedEdgeCount: number;
    readonly modalities: readonly GovernanceOnlineModality[];
    readonly connected: boolean;
  };
  readonly labels: RegisteredTargetLabelSet | null;
  readonly labelReceipt: {
    readonly schemaVersion: "socialgraph-fm.governance-target-label-receipt/2.0";
    readonly taskId: string;
    readonly targetReceiptHash: string;
    readonly receiptHash: string;
    readonly labelsSha256: string;
    readonly sourceLabelsSha256: string;
    readonly eligibilityMaskSha256: string;
    readonly eligibleNodeIds: readonly string[];
  } | null;
  readonly artifact: GovernanceArtifact;
  readonly createdAt: string;
  readonly registrationHash: string;
}

export interface RegisteredTargetLabelSetCreateRequest {
  readonly schemaVersion: typeof GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA;
  readonly sourceType: "imported_sidecar";
  readonly targetTaskRegistrationId: string;
  readonly runId: string;
  readonly resultHash: string;
}

export interface RegisteredTargetLabelSet {
  readonly schemaVersion: typeof GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA;
  readonly taskId: string;
  readonly inferenceSha256: string;
  readonly labels: readonly { readonly nodeId: string; readonly label: "positive" | "negative"; readonly structuralStratum: 0 | 1 | 2 | 3; readonly fusedDegree: number }[];
  readonly positiveCount: number;
  readonly negativeCount: number;
  readonly labelSetHash: string;
}

export interface TargetReviewPolicy {
  readonly schemaVersion: typeof GOVERNANCE_TARGET_POLICY_SCHEMA;
  readonly binding: GovernanceAdaptationBinding;
  readonly labelSetHash: string;
  readonly status: ReviewPolicyStatus;
  readonly selectedLambda: number;
  readonly eligibleLabelCount: number;
  readonly positiveCount: number;
  readonly negativeCount: number;
  readonly fittingRecipe: "l2-centroids+run-zscore+loo-balanced-log-loss-v1";
  readonly baseOutputsImmutable: true;
  readonly adaptedOutputFields: readonly ["adaptedReviewPriority", "adaptedRank"];
  readonly policyHash: string;
}

export interface TargetReviewPolicyFitRequest {
  readonly schemaVersion: "socialgraph-fm.governance-target-review-policy-fit-request/1.0";
  readonly targetTaskRegistrationId: string;
  readonly runId: string;
  readonly resultHash: string;
}

export interface TargetAdaptationComparison {
  readonly schemaVersion: typeof GOVERNANCE_TARGET_COMPARISON_SCHEMA;
  readonly binding: GovernanceAdaptationBinding;
  readonly policyHash: string;
  readonly total: number;
  readonly baseOutputsImmutable: true;
  readonly rows: readonly AdaptationComparisonRow[];
  readonly comparisonHash: string;
}

export interface AdaptationHandoffCreateRequest {
  readonly schemaVersion: typeof GOVERNANCE_ADAPTATION_HANDOFF_SCHEMA;
  readonly targetTaskRegistrationId: string;
  readonly policyHash: string;
  readonly decision: "pending_governance_review";
}

export interface AdaptationGovernanceHandoff extends AdaptationHandoffCreateRequest {
  readonly targetReceiptHash: string;
  readonly labelSetHash: string;
  readonly binding: GovernanceAdaptationBinding;
  readonly comparisonHash: string;
  readonly baseModelMutation: false;
  readonly handoffHash: string;
}

export interface AdaptationOverlayActivation {
  readonly schemaVersion: typeof GOVERNANCE_ADAPTATION_OVERLAY_SCHEMA;
  readonly targetTaskRegistrationId: string;
  readonly targetReceiptHash: string;
  readonly labelSetHash: string;
  readonly binding: GovernanceAdaptationBinding;
  readonly policyHash: string;
  readonly comparisonHash: string;
  readonly active: true;
  readonly baseModelMutation: false;
  readonly activationHash: string;
}

export interface TargetReviewCollectionCreateRequest {
  readonly schemaVersion: "socialgraph-fm.governance-review-collection/1.0";
  readonly idempotencyKey: string;
  readonly targetTaskRegistrationId: string;
  readonly runId: string;
  readonly resultHash: string;
  readonly title: string;
  readonly description: string;
  readonly items: readonly { readonly targetType: GovernanceTargetKind; readonly targetId: string; readonly note: string }[];
}

export interface TargetReviewCollection {
  readonly schemaVersion: "socialgraph-fm.governance-review-collection/1.0";
  readonly idempotencyKey: string;
  readonly targetTaskRegistrationId: string;
  readonly requestHash: string;
  readonly resultHash: string;
  readonly case: GovernanceCase;
  readonly collectionHash: string;
}

export interface GovernanceOnlineClientLike {
  health(signal?: AbortSignal): Promise<GovernanceOnlineHealth>;
  capabilities(signal?: AbortSignal): Promise<GovernanceOnlineCapabilities>;
  russiaSample(signal?: AbortSignal): Promise<GovernanceArtifact>;
  inspectArtifact(file: File, signal?: AbortSignal): Promise<GovernanceArtifactCompatibility>;
  uploadArtifact(file: File, cleanSelfLoops: boolean, signal?: AbortSignal): Promise<GovernanceArtifact>;
  artifact(artifactId: string, signal?: AbortSignal): Promise<GovernanceArtifact>;
  preview(artifactId: string, signal?: AbortSignal, projection?: GovernanceProjectionRequest): Promise<GovernanceOnlinePreview>;
  runPreview(runId: string, signal?: AbortSignal, projection?: GovernanceProjectionRequest): Promise<GovernanceOnlinePreview>;
  listRuns(signal?: AbortSignal): Promise<readonly GovernanceOnlineRun[]>;
  createRun(request: GovernanceOnlineRunRequest, signal?: AbortSignal): Promise<GovernanceOnlineRun>;
  run(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineRun>;
  cancelRun(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineRun>;
  retryRun(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineRun>;
  compareRuns(leftRunId: string, rightRunId: string, limit?: number, signal?: AbortSignal): Promise<GovernanceRunComparison>;
  result(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineResult>;
  findings(runId: string, offset: number, limit: number, signal?: AbortSignal): Promise<GovernanceFindingPage>;
  evidence(runId: string, nodeId: string, signal?: AbortSignal): Promise<GovernanceOnlineEvidence>;
  derivations(runId: string, kind: GovernanceDerivation["kind"], signal?: AbortSignal): Promise<readonly GovernanceDerivation[]>;
  listCases(runId: string, signal?: AbortSignal): Promise<readonly GovernanceCase[]>;
  createCase(runId: string, title: string, description: string, signal?: AbortSignal): Promise<GovernanceCase>;
  updateCase(caseId: string, state: GovernanceCaseStatus, reason: string, signal?: AbortSignal): Promise<GovernanceCase>;
  addCaseItem(caseId: string, targetType: GovernanceTargetKind, targetId: string, note: string, signal?: AbortSignal): Promise<GovernanceCase>;
  review(caseId: string, targetType: GovernanceTargetKind, targetId: string, decision: GovernanceReviewDecision, reason: string, signal?: AbortSignal): Promise<GovernanceCase>;
  case(caseId: string, signal?: AbortSignal): Promise<GovernanceCase>;
  report(caseId: string, format: "json" | "markdown" | "html", signal?: AbortSignal): Promise<Blob>;
  createAdaptationLabelSet(request: AdaptationLabelSetCreateRequest, signal?: AbortSignal): Promise<AdaptationLabelSet>;
  fitAdaptationPolicy(labelSetHash: string, signal?: AbortSignal): Promise<AdaptationReviewPolicy>;
  adaptationPolicy(policyHash: string, signal?: AbortSignal): Promise<AdaptationReviewPolicy>;
  adaptationComparison(runId: string, policyHash: string, offset?: number, limit?: number, signal?: AbortSignal): Promise<AdaptationComparison>;
  registerTargetTask(file: File, signal?: AbortSignal): Promise<TargetTaskRegistration>;
  targetTask(registrationId: string, signal?: AbortSignal): Promise<TargetTaskRegistration>;
  createTargetLabelSet(request: RegisteredTargetLabelSetCreateRequest, signal?: AbortSignal): Promise<RegisteredTargetLabelSet>;
  fitTargetPolicy(labelSetHash: string, request: TargetReviewPolicyFitRequest, signal?: AbortSignal): Promise<TargetReviewPolicy>;
  targetPolicy(policyHash: string, signal?: AbortSignal): Promise<TargetReviewPolicy>;
  targetComparison(runId: string, policyHash: string, offset?: number, limit?: number, signal?: AbortSignal): Promise<TargetAdaptationComparison>;
  createAdaptationHandoff(request: AdaptationHandoffCreateRequest, signal?: AbortSignal): Promise<AdaptationGovernanceHandoff>;
  adaptationHandoff(handoffHash: string, signal?: AbortSignal): Promise<AdaptationGovernanceHandoff>;
  activateTargetPolicy(policyHash: string, registrationId: string, signal?: AbortSignal): Promise<AdaptationOverlayActivation>;
  createTargetReviewCollection(request: TargetReviewCollectionCreateRequest, signal?: AbortSignal): Promise<TargetReviewCollection>;
}
