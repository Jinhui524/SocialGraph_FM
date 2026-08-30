export const RESEARCH_SCHEMA = "socialgraph-fm.research/1.0" as const;

export type ResearchTaskId =
  | "research.content_policy_review"
  | "research.account_risk_review"
  | "research.signed_relation_review"
  | "core.collaboration_completion";

export type ResearchTargetScope =
  | { readonly kind: "nodes"; readonly nodeIds: readonly [string, ...string[]] }
  | {
      readonly kind: "directed-node-pairs";
      readonly pairs: readonly (readonly [string, string])[];
    }
  | {
      readonly kind: "collaboration-candidates";
      readonly anchorNodeId: string;
      readonly topK: number;
    };

export interface ResearchModelCapability {
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly artifactHash: string;
  readonly taskIds: readonly ResearchTaskId[];
  readonly graphSchemaVersion: string;
  readonly maxNodes: number;
  readonly maxEdges: number;
  readonly claimStatus: "observed_transfer_gain" | "not_demonstrated";
}

export interface ResearchCapabilities {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly channel: "research";
  readonly releaseLabel: "SocialGraph-FM Research";
  readonly seed: 1729;
  readonly preliminary: true;
  readonly researchServingReady: boolean;
  readonly unavailableReason: string | null;
  readonly model: ResearchModelCapability | null;
  readonly taskIds: readonly ResearchTaskId[];
  readonly upload: {
    readonly compatibleTaskIds: readonly ["core.collaboration_completion"];
    readonly auxiliaryCapabilities: readonly ["similar-nodes"];
    readonly minNodes: number;
    readonly maxNodes: number;
    readonly maxEdges: number;
  };
  readonly capabilityHash: string;
}

export interface ResearchScenarioMetric {
  readonly name: string;
  readonly value: number;
}

export interface ResearchScenario {
  readonly scenarioId:
    | "twitch-content-policy"
    | "tolokers-account-risk"
    | "wiki-rfa-signed-relation"
    | "email-eu-collaboration";
  readonly datasetId: string;
  readonly title: string;
  readonly taskId: ResearchTaskId;
  readonly graphVersionId: string;
  readonly graphVersionHash: string | null;
  readonly modelVersionId: string | null;
  readonly enabled: boolean;
  readonly unavailableReason: string | null;
  readonly defaultTargetScope: ResearchTargetScope;
  readonly primaryMetric: ResearchScenarioMetric | null;
  readonly scratchDelta: number | null;
}

export interface ResearchScenarios {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly releaseLabel: "SocialGraph-FM Research";
  readonly seed: 1729;
  readonly preliminary: true;
  readonly scenarios: readonly ResearchScenario[];
  readonly scenariosHash: string;
}

export interface ResearchScenarioPreviewNode {
  readonly id: string;
  readonly label: string;
}

export interface ResearchScenarioPreviewEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly directed: boolean;
}

export interface ResearchScenarioPreview {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly scenarioId: ResearchScenario["scenarioId"];
  readonly graphVersionId: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly nodes: readonly ResearchScenarioPreviewNode[];
  readonly edges: readonly ResearchScenarioPreviewEdge[];
  readonly partialPreview: boolean;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly previewHash: string;
}

export interface ResearchRunRequest {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly graphVersionId: string;
  readonly taskId: ResearchTaskId;
  readonly modelVersionId: string;
  readonly targetScope: ResearchTargetScope;
  readonly scenarioId?: ResearchScenario["scenarioId"];
  readonly parameters: { readonly candidateLimit: number };
}

export type ResearchRunStatusValue = "queued" | "running" | "succeeded" | "failed";

export interface ResearchRunStatus {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly runId: string;
  readonly requestHash: string;
  readonly status: ResearchRunStatusValue;
  readonly progress: number;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly errorCode: string | null;
  readonly stateHash: string;
}

export interface ResearchFinding {
  readonly id: string;
  readonly rank: number;
  readonly entityType: "node" | "directed-edge" | "node-pair";
  readonly entityIds: readonly string[];
  readonly score: number;
  readonly scoreKind: "probability" | "ranking-score";
  readonly calibrated: boolean;
  readonly reasonCodes: readonly string[];
  readonly limitations: readonly string[];
  readonly reviewRequired: true;
}

export interface ResearchRunResult {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly runId: string;
  readonly requestHash: string;
  readonly taskId: ResearchTaskId;
  readonly graphVersionId: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly seed: 1729;
  readonly preliminary: true;
  readonly calibrationStatus: "calibrated" | "ranking_only";
  readonly findings: readonly ResearchFinding[];
  readonly completedAt: string;
  readonly resultHash: string;
}

export interface ResearchRunBinding {
  readonly runId: string;
  readonly publicRequestHash: string;
  readonly serverRequestHash: string;
  readonly graphVersionId: string;
  readonly modelVersionId: string;
  readonly taskId: ResearchTaskId;
}

export interface ResearchCreatedRun {
  readonly status: ResearchRunStatus;
  readonly binding: ResearchRunBinding;
}

export interface ResearchStructuralFacts {
  readonly degree: number;
  readonly inDegree: number;
  readonly outDegree: number;
  readonly pagerank: number;
  readonly clustering: number;
  readonly coreNumber: number;
}

export interface ResearchSimilarNodeMatch {
  readonly graphVersionId: string;
  readonly nodeId: string;
  readonly datasetId: string | null;
  readonly similarity: number;
  readonly structuralFacts: ResearchStructuralFacts;
}

export interface ResearchSimilarNodesRequest {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly graphVersionId: string;
  readonly nodeId: string;
  readonly topK: number;
  readonly modelVersionId: string;
}

export interface ResearchSimilarNodesResult {
  readonly schemaVersion: typeof RESEARCH_SCHEMA;
  readonly graphVersionId: string;
  readonly nodeId: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly matches: readonly ResearchSimilarNodeMatch[];
  readonly resultHash: string;
}

export interface ResearchClientLike {
  capabilities(signal?: AbortSignal): Promise<ResearchCapabilities>;
  scenarios(signal?: AbortSignal): Promise<ResearchScenarios>;
  scenarioPreview(
    scenarioId: ResearchScenario["scenarioId"],
    signal?: AbortSignal,
  ): Promise<ResearchScenarioPreview>;
  createRun(request: ResearchRunRequest, signal?: AbortSignal): Promise<ResearchCreatedRun>;
  getRun(
    runId: string,
    binding: ResearchRunBinding,
    signal?: AbortSignal,
  ): Promise<ResearchRunStatus>;
  getResult(
    runId: string,
    binding: ResearchRunBinding,
    signal?: AbortSignal,
  ): Promise<ResearchRunResult>;
  similarNodes(
    request: ResearchSimilarNodesRequest,
    signal?: AbortSignal,
  ): Promise<ResearchSimilarNodesResult>;
}

export type ResearchServiceState =
  | { readonly state: "checking" }
  | { readonly state: "connected"; readonly capabilities: ResearchCapabilities }
  | { readonly state: "unavailable"; readonly code: string };
