import { sha256Canonical } from "../../services/graphIdentity";
import type {
  GlobalModelCapabilities,
  GlobalModelHealth,
  GlobalModelModelCard,
  GlobalModelNodeEvidence,
  GlobalModelProtocol,
  GlobalModelProtocolModel,
  GlobalModelReviewDecision,
  GlobalModelReviewRecord,
  GlobalModelRunRequest,
  GlobalModelRunResult,
  GlobalModelRunStatus,
  GlobalModelScenario,
  GlobalModelScenarioPreview,
} from "../../types/globalModel";

export const GLOBAL_MODEL_TEST_HASHES = {
  artifact: "1".repeat(64),
  corpus: "2".repeat(64),
  source: "3".repeat(64),
  graph: "4".repeat(64),
  model: "5".repeat(64),
  split: "6".repeat(64),
  inDomainModel: "7".repeat(64),
  lowLabelModel: "8".repeat(64),
  crossDomainModel: "9".repeat(64),
  inDomainState: "a".repeat(64),
  lowLabelState: "b".repeat(64),
  crossDomainState: "c".repeat(64),
  globalState: "d".repeat(64),
};

export const GLOBAL_MODEL_TEST_PROTOCOL_MODELS: Readonly<Record<GlobalModelProtocol, GlobalModelProtocolModel>> = {
  in_domain: {
    modelVersionId: "socialgraph-fm-in-domain/test",
    modelVersionHash: GLOBAL_MODEL_TEST_HASHES.inDomainModel,
    modelStateHash: GLOBAL_MODEL_TEST_HASHES.inDomainState,
    state: "frozenDemo",
  },
  low_label: {
    modelVersionId: "socialgraph-fm-low-label/test",
    modelVersionHash: GLOBAL_MODEL_TEST_HASHES.lowLabelModel,
    modelStateHash: GLOBAL_MODEL_TEST_HASHES.lowLabelState,
    state: "frozenDemo",
  },
  cross_domain: {
    modelVersionId: "socialgraph-fm-cross-domain/test",
    modelVersionHash: GLOBAL_MODEL_TEST_HASHES.crossDomainModel,
    modelStateHash: GLOBAL_MODEL_TEST_HASHES.crossDomainState,
    state: "frozenDemo",
  },
  global: {
    modelVersionId: "socialgraph-fm-global/test",
    modelVersionHash: GLOBAL_MODEL_TEST_HASHES.model,
    modelStateHash: GLOBAL_MODEL_TEST_HASHES.globalState,
    state: "servingReady",
  },
};

export const GLOBAL_MODEL_TEST_RUN_ID = `global-model-${"a".repeat(32)}`;
export const GLOBAL_MODEL_TEST_REVIEW_ID = `review-${"b".repeat(32)}`;

function hashed<T extends Record<string, unknown>, K extends string>(payload: T, field: K): T & Record<K, string> {
  return { ...payload, [field]: sha256Canonical(payload) };
}

export function globalModelCapabilities(): GlobalModelCapabilities {
  return hashed({
    schemaVersion: "socialgraph-fm.global-model/1.0",
    channel: "global-model",
    releaseLabel: "SocialGraph-FM Global",
    seed: 12121995,
    servingReady: true,
    unavailableReason: null,
    taskId: "coordination_risk",
    datasetVersionId: "socialgraph-fm:russia",
    model: {
      modelVersionId: "socialgraph-fm-global/test",
      modelVersionHash: GLOBAL_MODEL_TEST_HASHES.model,
      artifactHash: GLOBAL_MODEL_TEST_HASHES.artifact,
      corpusHash: GLOBAL_MODEL_TEST_HASHES.corpus,
      sourceCodeHash: GLOBAL_MODEL_TEST_HASHES.source,
      taskId: "coordination_risk",
      protocols: ["in_domain", "low_label", "cross_domain", "global"],
      protocolModels: GLOBAL_MODEL_TEST_PROTOCOL_MODELS,
      state: "servingReady",
    },
  }, "capabilityHash") as unknown as GlobalModelCapabilities;
}

export function globalModelHealth(servingReady = true): GlobalModelHealth {
  const modelVersionId = servingReady ? "socialgraph-fm-global/test" : null;
  const modelVersionHash = servingReady ? GLOBAL_MODEL_TEST_HASHES.model : null;
  const corpusHash = servingReady ? GLOBAL_MODEL_TEST_HASHES.corpus : null;
  const serviceIdentity = sha256Canonical({
    service: "socialgraph-fm-gfm/global-model",
    datasetVersionId: "socialgraph-fm:russia",
    modelVersionId,
    modelVersionHash,
    corpusHash,
  });
  return hashed({
    schemaVersion: "socialgraph-fm.global-model-health/1.0",
    serviceIdentity,
    servingReady,
    modelVersionId,
    modelVersionHash,
    corpusHash,
    datasetVersionId: "socialgraph-fm:russia",
  }, "healthHash") as unknown as GlobalModelHealth;
}

export function globalModelModelCard(): GlobalModelModelCard {
  return hashed({
    schemaVersion: "socialgraph-fm.global-model-card/1.0",
    releaseId: "socialgraph-fm",
    modelVersionId: "socialgraph-fm-global/test",
    modelVersionHash: GLOBAL_MODEL_TEST_HASHES.model,
    taskId: "coordination_risk",
    architecture: {
      name: "SocialGraph-FM Governance cross-modal GraphSAGE",
      textFeatures: "anonymous precomputed embeddings",
      structuralFeatures: "factual fused-graph degree buckets",
      gnnLayers: 2,
      hiddenDim: 256,
      router: "shared residual plus domain/null adapters",
    },
    protocols: GLOBAL_MODEL_TEST_PROTOCOL_MODELS,
    trainingData: {
      countries: ["china", "cuba", "iran", "russia", "UAE", "venezuela"],
      nodeCount: 4_296,
      nodeCountByCountry: {
        china: 716,
        cuba: 716,
        iran: 716,
        russia: 716,
        UAE: 716,
        venezuela: 716,
      },
      content: "anonymous graph data with no raw text",
    },
    intendedUse: ["analyst-facing prioritization with human review"],
    outOfScope: ["automatic enforcement"],
    limitations: ["frozen research snapshot"],
    ethics: ["preserve anonymity and require human review"],
    licenses: [
      {
        name: "SocialGraph-FM Governance dataset",
        license: "CC-BY-4.0",
        url: "https://zenodo.org/records/13357621",
      },
      {
        name: "Reference implementation",
        license: "MIT",
        url: "https://example.invalid/reference-implementation",
      },
    ],
    sourceAttribution: {
      kind: "inspired",
      paperUrl: "https://proceedings.mlr.press/v267/yuan25h.html",
      completeReproduction: false,
    },
    metrics: {
      in_domain: { countryBalancedMacroF1: 0.842 },
      low_label: { countryBalancedMacroF1: 0.781 },
      cross_domain: { countryBalancedMacroF1: 0.724 },
      global: { countryBalancedMacroF1: 0.817 },
    },
    artifactHash: GLOBAL_MODEL_TEST_HASHES.artifact,
  }, "modelCardHash") as unknown as GlobalModelModelCard;
}

const metric = {
  macroF1: 0.842,
  prAuc: 0.791,
  threshold: 0.61,
  labelledTrainNodes: 310,
};

export function globalModelScenario(): GlobalModelScenario {
  return hashed({
    schemaVersion: "socialgraph-fm.global-model/1.0",
    scenarioId: "russia-coordination-risk",
    datasetVersionId: "socialgraph-fm:russia",
    graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
    modelVersionId: "socialgraph-fm-global/test",
    enabled: true,
    unavailableReason: null,
    nodeCount: 716,
    edgeCount: 2,
    protocols: ["in_domain", "low_label", "cross_domain", "global"],
    metrics: {
      in_domain: metric,
      low_label: { ...metric, macroF1: 0.781, labelledTrainNodes: 16 },
      cross_domain: { ...metric, macroF1: 0.724, labelledTrainNodes: 0 },
      global: { ...metric, macroF1: 0.817, labelledTrainNodes: 310 },
    },
    limitations: [
      "Anonymous research identifiers only.",
      "Predictions require human review.",
    ],
  }, "scenarioHash") as unknown as GlobalModelScenario;
}

export function globalModelPreview(): GlobalModelScenarioPreview {
  return hashed({
    schemaVersion: "socialgraph-fm.global-model/1.0",
    datasetVersionId: "socialgraph-fm:russia",
    graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
    nodes: [
      { id: "ru-001", label: "Account 001", degree: 14, structureMissing: false },
      { id: "ru-002", label: "Account 002", degree: 9, structureMissing: false },
      { id: "ru-003", label: "Account 003", degree: 2, structureMissing: true },
    ],
    edges: [
      { id: "ru-edge-1", source: "ru-001", target: "ru-002", modality: "coRT" },
      { id: "ru-edge-2", source: "ru-001", target: "ru-003", modality: "tweetSim" },
    ],
    nodeCount: 716,
    edgeCount: 2,
    partialPreview: true,
  }, "previewHash") as unknown as GlobalModelScenarioPreview;
}

export function globalModelRequest(protocol: GlobalModelRunRequest["protocol"] = "in_domain"): GlobalModelRunRequest {
  return {
    schemaVersion: "socialgraph-fm.global-model/1.0",
    taskId: "coordination_risk",
    datasetVersionId: "socialgraph-fm:russia",
    protocol,
    modelVersionId: GLOBAL_MODEL_TEST_PROTOCOL_MODELS[protocol].modelVersionId,
    topK: 50,
  };
}

export function globalModelStatus(
  requestHash: string,
  status: GlobalModelRunStatus["status"] = "succeeded",
): GlobalModelRunStatus {
  return {
    schemaVersion: "socialgraph-fm.global-model/1.0",
    runId: GLOBAL_MODEL_TEST_RUN_ID,
    requestHash,
    status,
    progress: status === "queued" ? 0 : status === "running" ? 45 : 100,
    createdAt: "2026-08-17T00:00:00.000000Z",
    updatedAt: "2026-08-17T00:00:01.000000Z",
    errorCode: status === "failed" ? "GFM_GLOBAL_MODEL_RUN_FAILED" : null,
  };
}

const findings = [
  {
    nodeId: "ru-001",
    score: 0.913,
    rank: 1,
    riskBand: "high" as const,
    predictedPositive: true,
    structureMissing: false,
    routes: [{ expert: "shared", weight: 0.68 }, { expert: "russia", weight: 0.32 }],
    modalityEvidence: { coRT: 8, coURL: 3, hashSeq: 2, fastRT: 4, tweetSim: 6 },
  },
  {
    nodeId: "ru-002",
    score: 0.704,
    rank: 2,
    riskBand: "review" as const,
    predictedPositive: true,
    structureMissing: false,
    routes: [{ expert: "shared", weight: 0.54 }, { expert: "null", weight: 0.46 }],
    modalityEvidence: { coRT: 4, coURL: 2, hashSeq: 1, fastRT: 0, tweetSim: 3 },
  },
];

export function globalModelResult(request = globalModelRequest()): GlobalModelRunResult {
  return hashed({
    schemaVersion: "socialgraph-fm.global-model/1.0",
    runId: GLOBAL_MODEL_TEST_RUN_ID,
    requestHash: sha256Canonical(request),
    taskId: "coordination_risk",
    protocol: request.protocol,
    datasetVersionId: "socialgraph-fm:russia",
    graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
    corpusHash: GLOBAL_MODEL_TEST_HASHES.corpus,
    splitHash: GLOBAL_MODEL_TEST_HASHES.split,
    modelVersionId: request.modelVersionId,
    modelVersionHash: GLOBAL_MODEL_TEST_PROTOCOL_MODELS[request.protocol].modelVersionHash,
    threshold: 0.61,
    metrics: metric,
    findings,
    limitations: ["Anonymous identifiers only.", "No automatic enforcement."],
    completedAt: "2026-08-17T00:00:02.000000Z",
  }, "resultHash") as unknown as GlobalModelRunResult;
}

export function globalModelEvidence(
  nodeId = "ru-001",
  request = globalModelRequest(),
): GlobalModelNodeEvidence {
  const result = globalModelResult(request);
  const finding = findings.find((item) => item.nodeId === nodeId) ?? findings[0]!;
  return hashed({
    schemaVersion: "socialgraph-fm.global-model/1.0",
    runId: GLOBAL_MODEL_TEST_RUN_ID,
    resultHash: result.resultHash,
    graphVersionHash: result.graphVersionHash,
    modelVersionId: result.modelVersionId,
    modelVersionHash: result.modelVersionHash,
    threshold: result.threshold,
    node: finding,
    neighbors: [
      {
        nodeId: "ru-002",
        score: 0.704,
        hop: 1,
        riskBand: "review",
        predictedPositive: true,
        structureMissing: false,
        modalities: ["coRT", "coURL"],
        relations: [
          { modality: "coRT", rawWeight: 2.5 },
          { modality: "coURL", rawWeight: 1.25 },
        ],
      },
      {
        nodeId: "ru-003",
        score: 0.67,
        hop: 1,
        riskBand: "review",
        predictedPositive: true,
        structureMissing: true,
        modalities: ["tweetSim"],
        relations: [{ modality: "tweetSim", rawWeight: 0.75 }],
      },
    ],
    structuralSignals: {
      fusedDegree: 2,
      structureMissing: false,
      relationNeighborCounts: { coRT: 1, coURL: 1, hashSeq: 0, fastRT: 0, tweetSim: 1 },
      twoHopNodeCount: 3,
      relationEvidenceRole: "explanationOnly",
    },
    evidenceSubgraph: {
      depth: 2,
      nodeCount: 4,
      edgeCount: 3,
      truncated: false,
      nodes: [
        {
          nodeId: "ru-001",
          score: 0.913,
          hop: 0,
          riskBand: "high",
          predictedPositive: true,
          structureMissing: false,
        },
        {
          nodeId: "ru-002",
          score: 0.704,
          hop: 1,
          riskBand: "review",
          predictedPositive: true,
          structureMissing: false,
        },
        {
          nodeId: "ru-003",
          score: 0.67,
          hop: 1,
          riskBand: "review",
          predictedPositive: true,
          structureMissing: true,
        },
        {
          nodeId: "ru-004",
          score: 0.31,
          hop: 2,
          riskBand: "low",
          predictedPositive: false,
          structureMissing: false,
        },
      ],
      edges: [
        {
          id: "ru-001:ru-002",
          source: "ru-001",
          target: "ru-002",
          relations: [{ modality: "coRT", rawWeight: 2.5 }],
          evidenceRole: "explanationOnly",
        },
        {
          id: "ru-001:ru-003",
          source: "ru-001",
          target: "ru-003",
          relations: [{ modality: "tweetSim", rawWeight: 0.75 }],
          evidenceRole: "explanationOnly",
        },
        {
          id: "ru-002:ru-004",
          source: "ru-002",
          target: "ru-004",
          relations: [{ modality: "coURL", rawWeight: 1.25 }],
          evidenceRole: "explanationOnly",
        },
      ],
    },
    limitation: (
      "Factual CSR relation types and stored raw weights are explanation-only; they are not "
      + "labels, proof of coordination, or additional model facts."
    ),
  }, "evidenceHash") as unknown as GlobalModelNodeEvidence;
}

export function globalModelReview(
  decision: GlobalModelReviewDecision,
  reason: string,
  nodeId = "ru-001",
): GlobalModelReviewRecord {
  return hashed({
    schemaVersion: "socialgraph-fm.global-model/1.0",
    reviewId: GLOBAL_MODEL_TEST_REVIEW_ID,
    runId: GLOBAL_MODEL_TEST_RUN_ID,
    nodeId,
    decision,
    reason,
    createdAt: "2026-08-17T00:00:03.000000Z",
  }, "reviewHash") as unknown as GlobalModelReviewRecord;
}
