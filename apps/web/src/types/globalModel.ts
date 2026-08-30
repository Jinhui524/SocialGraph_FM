export const GLOBAL_MODEL_SCHEMA = "socialgraph-fm.global-model/1.0" as const;
export const GLOBAL_MODEL_HEALTH_SCHEMA = "socialgraph-fm.global-model-health/1.0" as const;
export const GLOBAL_MODEL_CARD_SCHEMA = "socialgraph-fm.global-model-card/1.0" as const;
export const GLOBAL_MODEL_PROTOCOLS = ["in_domain", "low_label", "cross_domain", "global"] as const;

export type GlobalModelProtocol = typeof GLOBAL_MODEL_PROTOCOLS[number];
export type GlobalModelReviewDecision = "confirmed" | "rejected" | "pending";

export interface GlobalModelMetric {
  readonly macroF1: number;
  readonly prAuc: number;
  readonly threshold: number;
  readonly labelledTrainNodes: number;
}

export interface GlobalModelProtocolModel {
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly modelStateHash: string;
  readonly state: "frozenDemo" | "servingReady";
}

export interface GlobalModelModelCapability {
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly artifactHash: string;
  readonly corpusHash: string;
  readonly sourceCodeHash: string;
  readonly taskId: "coordination_risk";
  readonly protocols: typeof GLOBAL_MODEL_PROTOCOLS;
  readonly protocolModels: Readonly<Record<GlobalModelProtocol, GlobalModelProtocolModel>>;
  readonly state: "preliminary" | "servingReady";
}

export interface GlobalModelCapabilities {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly channel: "global-model";
  readonly releaseLabel: "SocialGraph-FM Global";
  readonly seed: 12121995;
  readonly servingReady: boolean;
  readonly unavailableReason: string | null;
  readonly taskId: "coordination_risk";
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly model: GlobalModelModelCapability | null;
  readonly capabilityHash: string;
}

export interface GlobalModelHealth {
  readonly schemaVersion: typeof GLOBAL_MODEL_HEALTH_SCHEMA;
  readonly serviceIdentity: string;
  readonly servingReady: boolean;
  readonly modelVersionId: string | null;
  readonly modelVersionHash: string | null;
  readonly corpusHash: string | null;
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly healthHash: string;
}

export interface GlobalModelModelCard {
  readonly schemaVersion: typeof GLOBAL_MODEL_CARD_SCHEMA;
  readonly releaseId: "socialgraph-fm";
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly taskId: "coordination_risk";
  readonly architecture: Readonly<{
    name: string;
    textFeatures: string;
    structuralFeatures: string;
    gnnLayers: 2;
    hiddenDim: 256;
    router: string;
  }>;
  readonly protocols: Readonly<Record<GlobalModelProtocol, GlobalModelProtocolModel>>;
  readonly trainingData: Readonly<{
    countries: readonly ["china", "cuba", "iran", "russia", "UAE", "venezuela"];
    nodeCount: number;
    nodeCountByCountry: Readonly<Record<"china" | "cuba" | "iran" | "russia" | "UAE" | "venezuela", number>>;
    content: string;
  }>;
  readonly intendedUse: readonly string[];
  readonly outOfScope: readonly string[];
  readonly limitations: readonly string[];
  readonly ethics: readonly string[];
  readonly licenses: readonly Readonly<{
    name: string;
    license: "CC-BY-4.0" | "MIT";
    url: string;
  }>[];
  readonly sourceAttribution: Readonly<{
    kind: "inspired";
    paperUrl: string;
    completeReproduction: false;
  }>;
  readonly metrics: Readonly<Record<GlobalModelProtocol, Readonly<Record<string, unknown>>>>;
  readonly artifactHash: string;
  readonly modelCardHash: string;
}

export interface GlobalModelScenario {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly scenarioId: "russia-coordination-risk";
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly graphVersionHash: string | null;
  readonly modelVersionId: string | null;
  readonly enabled: boolean;
  readonly unavailableReason: string | null;
  readonly nodeCount: 716;
  readonly edgeCount: number;
  readonly protocols: typeof GLOBAL_MODEL_PROTOCOLS;
  readonly metrics: Readonly<Record<GlobalModelProtocol, GlobalModelMetric | null>>;
  readonly limitations: readonly string[];
  readonly scenarioHash: string;
}

export interface GlobalModelPreviewNode {
  readonly id: string;
  readonly label: string;
  readonly degree: number;
  readonly structureMissing: boolean;
}

export type GlobalModelModality = "coRT" | "coURL" | "hashSeq" | "fastRT" | "tweetSim" | "fused";
export type GlobalModelRelationModality = Exclude<GlobalModelModality, "fused">;

export interface GlobalModelPreviewEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly modality: GlobalModelModality;
}

export interface GlobalModelScenarioPreview {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly graphVersionHash: string;
  readonly nodes: readonly GlobalModelPreviewNode[];
  readonly edges: readonly GlobalModelPreviewEdge[];
  readonly nodeCount: 716;
  readonly edgeCount: number;
  readonly partialPreview: boolean;
  readonly previewHash: string;
}

export interface GlobalModelRunRequest {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly taskId: "coordination_risk";
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly protocol: GlobalModelProtocol;
  readonly modelVersionId: string;
  readonly topK: number;
}

export type GlobalModelRunStatusValue = "queued" | "running" | "succeeded" | "failed";

export interface GlobalModelRunStatus {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly runId: string;
  readonly requestHash: string;
  readonly status: GlobalModelRunStatusValue;
  readonly progress: number;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly errorCode: string | null;
}

export interface GlobalModelRoute {
  readonly expert: string;
  readonly weight: number;
}

export interface GlobalModelModalityEvidence {
  readonly coRT: number;
  readonly coURL: number;
  readonly hashSeq: number;
  readonly fastRT: number;
  readonly tweetSim: number;
}

export interface GlobalModelNodeFinding {
  readonly nodeId: string;
  readonly score: number;
  readonly rank: number;
  readonly riskBand: "high" | "review" | "low";
  readonly predictedPositive: boolean;
  readonly structureMissing: boolean;
  readonly routes: readonly GlobalModelRoute[];
  readonly modalityEvidence: GlobalModelModalityEvidence;
}

export interface GlobalModelRunResult {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly runId: string;
  readonly requestHash: string;
  readonly taskId: "coordination_risk";
  readonly protocol: GlobalModelProtocol;
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly graphVersionHash: string;
  readonly corpusHash: string;
  readonly splitHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly threshold: number;
  readonly metrics: GlobalModelMetric;
  readonly findings: readonly GlobalModelNodeFinding[];
  readonly limitations: readonly string[];
  readonly completedAt: string;
  readonly resultHash: string;
}

export interface GlobalModelEvidenceNeighbor {
  readonly nodeId: string;
  readonly score: number;
  readonly hop: 1;
  readonly riskBand: "high" | "review" | "low";
  readonly predictedPositive: boolean;
  readonly structureMissing: boolean;
  readonly modalities: readonly GlobalModelRelationModality[];
  readonly relations: readonly GlobalModelRelationEvidence[];
}

export interface GlobalModelRelationEvidence {
  readonly modality: GlobalModelRelationModality;
  readonly rawWeight: number;
}

export interface GlobalModelStructuralSignals {
  readonly fusedDegree: number;
  readonly structureMissing: boolean;
  readonly relationNeighborCounts: GlobalModelModalityEvidence;
  readonly twoHopNodeCount: number;
  readonly relationEvidenceRole: "explanationOnly";
}

export interface GlobalModelEvidenceSubgraphNode {
  readonly nodeId: string;
  readonly score: number;
  readonly hop: 0 | 1 | 2;
  readonly riskBand: "high" | "review" | "low";
  readonly predictedPositive: boolean;
  readonly structureMissing: boolean;
}

export interface GlobalModelEvidenceSubgraphEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly relations: readonly GlobalModelRelationEvidence[];
  readonly evidenceRole: "explanationOnly";
}

export interface GlobalModelEvidenceSubgraph {
  readonly depth: 2;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly truncated: boolean;
  readonly nodes: readonly GlobalModelEvidenceSubgraphNode[];
  readonly edges: readonly GlobalModelEvidenceSubgraphEdge[];
}

export interface GlobalModelNodeEvidence {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly runId: string;
  readonly resultHash: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly threshold: number;
  readonly node: GlobalModelNodeFinding;
  readonly neighbors: readonly GlobalModelEvidenceNeighbor[];
  readonly structuralSignals: GlobalModelStructuralSignals;
  readonly evidenceSubgraph: GlobalModelEvidenceSubgraph;
  readonly limitation: string;
  readonly evidenceHash: string;
}

export interface GlobalModelReviewRequest {
  readonly schemaVersion: typeof GLOBAL_MODEL_SCHEMA;
  readonly nodeId: string;
  readonly decision: GlobalModelReviewDecision;
  readonly reason: string;
}

export interface GlobalModelReviewRecord extends GlobalModelReviewRequest {
  readonly reviewId: string;
  readonly runId: string;
  readonly createdAt: string;
  readonly reviewHash: string;
}

export interface GlobalModelRunBinding {
  readonly runId: string;
  readonly publicRequestHash: string;
  readonly serverRequestHash: string;
  readonly taskId: "coordination_risk";
  readonly protocol: GlobalModelProtocol;
  readonly datasetVersionId: "socialgraph-fm:russia";
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
}

export interface GlobalModelCreatedRun {
  readonly status: GlobalModelRunStatus;
  readonly binding: GlobalModelRunBinding;
}

export interface GlobalModelRunIdentity {
  readonly graphVersionHash: string;
  readonly modelVersionHash: string;
}

export interface GlobalModelClientLike {
  health(signal?: AbortSignal): Promise<GlobalModelHealth>;
  capabilities(signal?: AbortSignal): Promise<GlobalModelCapabilities>;
  modelCard(signal?: AbortSignal): Promise<GlobalModelModelCard>;
  scenario(signal?: AbortSignal): Promise<GlobalModelScenario>;
  scenarioPreview(signal?: AbortSignal): Promise<GlobalModelScenarioPreview>;
  createRun(
    request: GlobalModelRunRequest,
    identity: GlobalModelRunIdentity,
    signal?: AbortSignal,
  ): Promise<GlobalModelCreatedRun>;
  getRun(runId: string, binding: GlobalModelRunBinding, signal?: AbortSignal): Promise<GlobalModelRunStatus>;
  getResult(runId: string, binding: GlobalModelRunBinding, signal?: AbortSignal): Promise<GlobalModelRunResult>;
  nodeEvidence(
    runId: string,
    nodeId: string,
    binding: GlobalModelRunBinding,
    signal?: AbortSignal,
  ): Promise<GlobalModelNodeEvidence>;
  submitReview(
    runId: string,
    request: GlobalModelReviewRequest,
    binding: GlobalModelRunBinding,
    signal?: AbortSignal,
  ): Promise<GlobalModelReviewRecord>;
}
