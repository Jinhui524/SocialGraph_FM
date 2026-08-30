import { canonicalJson, sha256Canonical, sha256Text } from "../../services/graphIdentity";
import { onlineArtifact, onlineFinding, onlinePreview, onlineResult, onlineRun, GOVERNANCE_HASHES, GOVERNANCE_RUN_ID } from "./governanceOnline";
import type { GovernanceTargetPackageReceipt } from "../../types/governanceOnline";

export const ADAPTATION_HASH = Object.freeze({
  labelSet: "1".repeat(64), policy: "2".repeat(64), readyPolicy: "3".repeat(64), comparison: "4".repeat(64), page: "5".repeat(64),
  source: "6".repeat(64), review: "7".repeat(64), centroidPositive: "8".repeat(64), centroidNegative: "9".repeat(64), artifact: "a".repeat(64), code: "b".repeat(64),
});

const TARGET_LABEL_RECIPE_SCHEMA_V11 = "socialgraph-fm.governance-target-label-recipe/1.1" as const;
const TARGET_PACKAGE_RECEIPT_SCHEMA_V11 = "socialgraph-fm.governance-target-package-receipt/1.1" as const;
const TARGET_LABEL_SET_SCHEMA_V11 = "socialgraph-fm.governance-target-label-set/1.1" as const;

export const adaptationBinding = () => ({
  artifactId: onlineArtifact().artifactId, datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph, runId: GOVERNANCE_RUN_ID,
  requestHash: GOVERNANCE_HASHES.request, resultHash: GOVERNANCE_HASHES.result, runArtifactHash: ADAPTATION_HASH.artifact,
  modelVersionId: onlineResult().modelVersionId, modelVersionHash: GOVERNANCE_HASHES.model, modelStateHash: GOVERNANCE_HASHES.state,
  recipeHash: GOVERNANCE_HASHES.recipe, codeHash: ADAPTATION_HASH.code, seed: 20260820,
});

export const targetLabelRecipe = () => ({
  schemaVersion: TARGET_LABEL_RECIPE_SCHEMA_V11,
  datasetId: "thailand-authorized",
  bundleSha256: GOVERNANCE_HASHES.artifact,
  selectionRecipe: {
    version: "graph-fused-degree-quartile-stable-hash-v2",
    stratification: "graph-fused-degree-rank-quartile",
    structuralStrata: 4,
    labelsPerClass: 8,
    labelsPerClassPerStratum: 2,
    scoreInputs: [],
  },
  labels: [
    ...[0, 1, 32, 33, 64, 65, 96, 97].map((index) => ({ index, label: "io" as const })),
    ...[2, 3, 34, 35, 66, 67, 98, 99].map((index) => ({ index, label: "control" as const })),
  ].map(({ index, label }) => ({
    nodeId: `th:${String(index + 1).padStart(24, "0")}`,
    label,
    structuralStratum: Math.floor(index / 32) as 0 | 1 | 2 | 3,
    fusedDegree: index + 1,
  })).sort((left, right) => left.nodeId.localeCompare(right.nodeId)),
});

export const targetPackageReceipt = (): GovernanceTargetPackageReceipt => {
  const recipe = targetLabelRecipe();
  const logical = {
    schemaVersion: TARGET_PACKAGE_RECEIPT_SCHEMA_V11,
    datasetId: recipe.datasetId,
    sourceSchemaVersion: "socialgraph-fm.anonymized-posts/1.0",
    sourceSha256: ADAPTATION_HASH.source,
    authorizationReference: "fixture-approval-2026-08-20",
    bundleSha256: recipe.bundleSha256,
    labelsSha256: sha256Text(`${canonicalJson(recipe)}\n`),
    encoder: {
      modelId: "fixture/deterministic-encoder",
      revision: "fixture-v1",
      cacheSha256: "1".repeat(64),
      compatibility: "dimension-only-unverified",
      dimension: 768,
    },
    selectionRecipe: {
      version: "connected-structural-hash-v2",
      nodeCount: 128,
      requiredIo: 16,
      requiredControls: 64,
      minimumNonemptyModalities: 4,
      scoreInputs: [],
      groupRelations: { maxGroupAccounts: 256, totalPotentialPairBudget: 50_000 },
      fastRT: { windowSeconds: 10, pairBudget: 50_000, algorithm: "sorted-sliding-window-v1" },
      tweetSim: { mutualTopK: 5, cosineThreshold: 0.8, pairBudget: 10_000 },
    },
    labelSelectionRecipe: recipe.selectionRecipe,
    coverage: {
      nodeCount: 128,
      ioCount: 32,
      controlCount: 96,
      nonemptyModalities: ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"],
      connected: true,
    },
  };
  return { ...logical, receiptHash: sha256Canonical(logical) } as unknown as GovernanceTargetPackageReceipt;
};

export const targetLabelSourceRecordHash = (row: ReturnType<typeof targetLabelRecipe>["labels"][number]) => {
  const recipe = targetLabelRecipe();
  const receipt = targetPackageReceipt();
  return sha256Canonical({
    schemaVersion: recipe.schemaVersion,
    datasetId: recipe.datasetId,
    bundleSha256: recipe.bundleSha256,
    labelsSha256: receipt.labelsSha256,
    receiptHash: receipt.receiptHash,
    ...row,
  });
};

export const targetLabelSet = () => {
  const binding = adaptationBinding();
  const receipt = targetPackageReceipt();
  const labels = targetLabelRecipe().labels.map((row) => ({
    nodeId: row.nodeId, label: row.label === "io" ? "positive" : "negative", sourceType: "imported_sidecar",
    sourceRecordId: `thailand-authorized:${row.nodeId}`, sourceRecordHash: targetLabelSourceRecordHash(row), reviewEventHash: null,
    structuralStratum: row.structuralStratum, fusedDegree: row.fusedDegree,
    labelsSha256: receipt.labelsSha256, receiptHash: receipt.receiptHash, binding,
  }));
  return {
    schemaVersion: TARGET_LABEL_SET_SCHEMA_V11, binding, sidecarReceipt: receipt,
    sourceRecords: labels.map(({ sourceType, sourceRecordId, sourceRecordHash, reviewEventHash }) => ({ sourceType, sourceRecordId, sourceRecordHash, reviewEventHash })),
    reviewEventHashes: [], labels, conflicts: [], positiveCount: 8, negativeCount: 8, labelSetHash: ADAPTATION_HASH.labelSet,
  };
};

export const targetReviewPolicy = (status: "collecting_reviews" | "ready" | "insufficient_signal" | "invalid" = "ready") => ({
  schemaVersion: "socialgraph-fm.governance-target-review-policy/1.0", binding: adaptationBinding(), labelSetHash: ADAPTATION_HASH.labelSet, status,
  selectedLambda: status === "ready" ? 0.5 : 0, lambdaCandidates: [0, 0.25, 0.5, 1], validationLosses: { "0": 0.72, "0.25": 0.68, "0.5": 0.64, "1": 0.7 },
  eligibleLabelCount: 16, positiveCount: 8, negativeCount: 8, embeddingDimension: 256,
  positiveCentroidHash: ADAPTATION_HASH.centroidPositive, negativeCentroidHash: ADAPTATION_HASH.centroidNegative,
  normalizationEpsilon: 1e-8, fittingRecipe: "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
  readyPolicyHash: status === "ready" ? ADAPTATION_HASH.readyPolicy : null, policyHash: ADAPTATION_HASH.policy,
});

export const adaptationComparison = () => ({
  schemaVersion: "socialgraph-fm.governance-adaptation-comparison/1.0", binding: adaptationBinding(), policyHash: ADAPTATION_HASH.policy,
  total: 3, offset: 0, limit: 100,
  rows: [
    { nodeId: "n2", baseScore: 0.58, baseRank: 2, adaptedReviewPriority: 0.86, adaptedRank: 1, rankDelta: -1 },
    { nodeId: "n1", baseScore: 0.92, baseRank: 1, adaptedReviewPriority: 0.74, adaptedRank: 2, rankDelta: 1 },
    { nodeId: "n3", baseScore: 0.2, baseRank: 3, adaptedReviewPriority: 0.31, adaptedRank: 3, rankDelta: 0 },
  ],
  comparisonHash: ADAPTATION_HASH.comparison, pageHash: ADAPTATION_HASH.page,
});

export const adaptationNodeId = (index: number) => `th:${String(index + 1).padStart(24, "0")}`;

export const adaptationComparison128 = (returned = 128) => ({
  ...adaptationComparison(),
  total: 128,
  limit: 500,
  rows: Array.from({ length: returned }, (_, index) => {
    const baseRank = index + 1;
    const adaptedRank = index % 2 === 0 ? baseRank + 1 : baseRank - 1;
    return {
      nodeId: adaptationNodeId(index),
      baseScore: Math.max(0.01, 0.99 - index / 150),
      baseRank,
      adaptedReviewPriority: Math.max(0.01, 0.98 - adaptedRank / 150),
      adaptedRank,
      rankDelta: adaptedRank - baseRank,
    };
  }),
});

export const adaptationOverview128 = () => ({
  ...onlinePreview(),
  runId: GOVERNANCE_RUN_ID,
  resultHash: GOVERNANCE_HASHES.result,
  nodes: Array.from({ length: 128 }, (_, index) => ({
    id: adaptationNodeId(index),
    label: `Anonymous Thailand account ${index + 1}`,
    degree: index + 1,
    structureMissing: false,
    score: Math.max(0.01, 0.99 - index / 150),
    riskBand: index < 16 ? "high" : index < 64 ? "review" : "low",
    groupId: `th-group-${Math.floor(index / 16) + 1}`,
  })),
  edges: Array.from({ length: 127 }, (_, index) => ({
    id: `th-edge-${index + 1}`,
    source: adaptationNodeId(0),
    target: adaptationNodeId(index + 1),
    modalities: ["coRT"],
    factual: true,
  })),
  nodeCount: 128,
  edgeCount: 127,
  partialPreview: false,
});

export const adaptationRawPreview128 = () => ({
  ...adaptationOverview128(),
  runId: null,
  resultHash: null,
  nodes: adaptationOverview128().nodes.map((node) => ({ ...node, score: null, riskBand: null, groupId: null })),
});

export const adaptationFindings128 = () => Array.from({ length: 128 }, (_, index) => {
  const score = index < 16 ? 0.92 - index / 1_000 : index < 64 ? 0.68 - index / 1_000 : 0.31 - index / 1_000;
  return { ...onlineFinding(adaptationNodeId(index), index + 1, score), label: `Anonymous Thailand account ${index + 1}` };
});

export const adaptationResult128 = () => ({
  ...onlineResult(),
  distribution: { low: 64, review: 48, high: 16, predictedPositive: 16, total: 128 },
  findings: adaptationFindings128(),
  totalFindings: 128,
});

export const adaptationFindingPage128 = () => ({
  schemaVersion: "socialgraph-fm.gfm-governance/2.0",
  runId: GOVERNANCE_RUN_ID,
  items: adaptationFindings128(),
  total: 128,
  offset: 0,
  limit: 128,
  pageHash: GOVERNANCE_HASHES.page,
});

export const adaptationWorkspaceSnapshot = () => ({
  schemaVersion: "socialgraph-fm.governance-workspace/1.0" as const,
  sessionId: "governance-session", sourceFileName: "thailand-target.zip",
  artifact: { ...onlineArtifact(), datasetId: "thailand-authorized", displayName: "Thailand target graph", bundleSha256: GOVERNANCE_HASHES.artifact, nodeCount: 128, relationRowCount: 127 }, preview: onlinePreview(),
  run: onlineRun(), result: adaptationResult128(), updatedAt: "2026-08-20T08:00:00Z",
});
