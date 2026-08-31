import { GOVERNANCE_INPUT_SCHEMA, GOVERNANCE_ONLINE_SCHEMA, GOVERNANCE_RELATION_MODALITIES } from "../../types/governanceOnline";
import { GOVERNANCE_PUBLIC_SKILLS } from "../../types/governanceSkills";

export const GOVERNANCE_HASHES = Object.freeze({
  capability: "1".repeat(64), health: "2".repeat(64), model: "3".repeat(64), state: "4".repeat(64), recipe: "5".repeat(64),
  artifact: "6".repeat(64), dataset: "7".repeat(64), graph: "8".repeat(64), preview: "9".repeat(64), request: "a".repeat(64),
  status: "b".repeat(64), result: "c".repeat(64), page: "d".repeat(64), evidence: "e".repeat(64), comparison: "f".repeat(64),
});
export const GOVERNANCE_ARTIFACT_ID = `governance-artifact-${"1".repeat(32)}`;
export const GOVERNANCE_RUN_ID = `governance-${"2".repeat(32)}`;
export const GOVERNANCE_OTHER_RUN_ID = `governance-${"3".repeat(32)}`;

export const onlineHealth = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, serviceIdentity: "0".repeat(64), servingReady: true, onlineForwardReady: true,
  modelVersionId: "socialgraph-fm-global/test", modelVersionHash: GOVERNANCE_HASHES.model, modelStateHash: GOVERNANCE_HASHES.state,
  device: "cpu", dtype: "float32", loadedAt: "2026-08-18T04:00:00Z", queueDepth: 0, activeRunId: null,
  runtimeRecipeHash: GOVERNANCE_HASHES.recipe, healthHash: GOVERNANCE_HASHES.health,
});
export const onlineCapabilities = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, channel: "governance", taskId: "coordination_risk", servingReady: true, onlineForwardReady: true,
  unavailableReason: null, modelVersionId: "socialgraph-fm-global/test", modelVersionHash: GOVERNANCE_HASHES.model, modelStateHash: GOVERNANCE_HASHES.state,
  supportedProtocols: ["global"], skills: [...GOVERNANCE_PUBLIC_SKILLS], inputSchemaVersion: GOVERNANCE_INPUT_SCHEMA, modalities: [...GOVERNANCE_RELATION_MODALITIES],
  sampleArtifactId: GOVERNANCE_ARTIFACT_ID, limits: { maxNodes: 10000, maxRelationRows: 500000, maxEvidenceNodes: 300, maxEvidenceEdges: 1000, maxPreviewNodes: 3000, maxPreviewEdges: 12000 },
  capabilityHash: GOVERNANCE_HASHES.capability,
});
export const onlineArtifact = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, artifactId: GOVERNANCE_ARTIFACT_ID, datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph,
  nodeCount: 3, relationRowCount: 2, selfLoopsRemoved: 0, modalities: [...GOVERNANCE_RELATION_MODALITIES], compatibility: "compatible",
  createdAt: "2026-08-18T04:01:00Z", artifactHash: GOVERNANCE_HASHES.artifact,
});
export const artifactCompatibility = (selfLoopsDetected = 0) => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, inputSchemaVersion: GOVERNANCE_INPUT_SCHEMA, compatible: true,
  requiresSelfLoopCleaning: selfLoopsDetected > 0, prospectiveArtifactId: GOVERNANCE_ARTIFACT_ID, datasetContentHash: GOVERNANCE_HASHES.dataset,
  graphVersionHash: GOVERNANCE_HASHES.graph, nodeCount: 3, relationRowCount: 2, selfLoopsDetected,
  modalities: [...GOVERNANCE_RELATION_MODALITIES], issues: selfLoopsDetected ? ["self_loop_confirmation_required"] : [], compatibilityHash: GOVERNANCE_HASHES.capability,
});
export const onlinePreview = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, artifactId: GOVERNANCE_ARTIFACT_ID, datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph,
  runId: null, resultHash: null,
  nodes: [
    { id: "n1", label: "匿名账号 1", degree: 1, structureMissing: false, score: null, riskBand: null, groupId: null },
    { id: "n2", label: "匿名账号 2", degree: 1, structureMissing: false, score: null, riskBand: null, groupId: null },
    { id: "n3", label: "匿名账号 3", degree: 0, structureMissing: true, score: null, riskBand: null, groupId: null },
  ],
  edges: [{ id: "e1", source: "n1", target: "n2", modalities: ["coRT", "tweetSim"], factual: true }],
  nodeCount: 3, edgeCount: 1, partialPreview: false, previewHash: GOVERNANCE_HASHES.preview,
});
export const onlineRunPreview = () => ({
  ...onlinePreview(), runId: GOVERNANCE_RUN_ID, resultHash: GOVERNANCE_HASHES.result,
  nodes: onlinePreview().nodes.map((node, index) => ({
    ...node, score: [0.92, 0.58, 0.2][index], riskBand: (["high", "review", "low"] as const)[index], groupId: index < 2 ? "group-1" : null,
  })),
});
export const onlineFinding = (nodeId = "n1", rank = 1, score = 0.92) => ({
  nodeId, label: `匿名账号 ${nodeId.slice(1)}`, score, logit: 2.1, rank, riskBand: score > 0.8 ? "high" : score >= 0.4 ? "review" : "low",
  predictedPositive: score > 0.8, structureMissing: nodeId === "n3", routes: [{ expert: "shared", weight: 1 }, { expert: "domain:russia", weight: 0.7 }, { expert: "null", weight: 0.3 }],
  modalityContribution: { text: 0.62, structure: 0.38 }, modalityEvidence: { coRT: 2, coURL: 0, hashSeq: 0, fastRT: 0, tweetSim: 1 }, communityId: "group-1",
});
export const onlineRun = (status = "succeeded", runId = GOVERNANCE_RUN_ID) => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId, requestHash: GOVERNANCE_HASHES.request, artifactId: GOVERNANCE_ARTIFACT_ID,
  datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph, modelVersionId: "socialgraph-fm-global/test", modelVersionHash: GOVERNANCE_HASHES.model, modelStateHash: GOVERNANCE_HASHES.state,
  status, stage: status === "succeeded" ? "completed" : "inferencing", progress: status === "succeeded" ? 100 : 65,
  createdAt: "2026-08-18T04:02:00Z", updatedAt: "2026-08-18T04:03:00Z", errorCode: null, cancelRequested: false, statusHash: GOVERNANCE_HASHES.status,
});
export const onlineResult = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: GOVERNANCE_RUN_ID, requestHash: GOVERNANCE_HASHES.request, artifactId: GOVERNANCE_ARTIFACT_ID,
  datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph, modelVersionId: "socialgraph-fm-global/test", modelVersionHash: GOVERNANCE_HASHES.model,
  modelStateHash: GOVERNANCE_HASHES.state, threshold: 0.61, calibration: { temperature: 1.1, bias: 0.05, referenceThreshold: 0.61, applicability: "reference_replay" },
  referenceMetrics: { macroF1: 0.91 }, datasetMetrics: null, distribution: { low: 1, review: 1, high: 1, predictedPositive: 1, total: 3 },
  findings: [onlineFinding()], totalFindings: 3, limitations: ["风险候选必须由人工复核。"], completedAt: "2026-08-18T04:03:00Z", resultHash: GOVERNANCE_HASHES.result,
});
export const findingPage = () => ({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: GOVERNANCE_RUN_ID, items: [onlineFinding(), onlineFinding("n2", 2, 0.58), onlineFinding("n3", 3, 0.2)], total: 3, offset: 0, limit: 3, pageHash: GOVERNANCE_HASHES.page });
export const derivationPage = (kind: "group" | "factual_relation" | "potential_link") => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: GOVERNANCE_RUN_ID, items: [{
    id: `${kind}-1`, kind, priority: 0.8, nodeIds: ["n1", "n2"], source: kind === "group" ? null : "n1", target: kind === "group" ? null : "n2",
    modalities: kind === "group" ? [] : ["coRT"], memberCount: kind === "group" ? 2 : null, meanScore: kind === "group" ? 0.82 : null,
    p90Score: kind === "group" ? 0.9 : null, scoreComponents: { risk: 0.8 }, factual: kind === "factual_relation", limitation: kind === "potential_link" ? "潜在线索不是事实边。" : "派生治理优先级。",
  }], total: 1, offset: 0, limit: 100, pageHash: GOVERNANCE_HASHES.page,
});
export const evidencePayload = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: GOVERNANCE_RUN_ID, resultHash: GOVERNANCE_HASHES.result, artifactId: GOVERNANCE_ARTIFACT_ID,
  datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph, modelVersionId: "socialgraph-fm-global/test",
  modelVersionHash: GOVERNANCE_HASHES.model, modelStateHash: GOVERNANCE_HASHES.state, threshold: 0.61, node: onlineFinding(),
  neighbors: [{ nodeId: "n2", score: 0.58, hop: 1, riskBand: "review", predictedPositive: false, structureMissing: false, modalities: ["coRT"], relations: [{ modality: "coRT", rawWeight: 0.84 }] }],
  structuralSignals: { fusedDegree: 1, structureMissing: false, relationNeighborCounts: { coRT: 1, coURL: 0, hashSeq: 0, fastRT: 0, tweetSim: 0 }, twoHopNodeCount: 1, relationEvidenceRole: "explanationOnly" },
  evidenceSubgraph: {
    depth: 2, nodeCount: 2, edgeCount: 1, truncated: false,
    nodes: [
      { nodeId: "n1", score: 0.92, hop: 0, riskBand: "high", predictedPositive: true, structureMissing: false },
      { nodeId: "n2", score: 0.58, hop: 1, riskBand: "review", predictedPositive: false, structureMissing: false },
    ],
    edges: [{ id: "e1", source: "n1", target: "n2", relations: [{ modality: "coRT", rawWeight: 0.84 }], evidenceRole: "explanationOnly" }],
  },
  truncated: false, limitation: "关系权重仅用于解释，不构成事实标签。", evidenceHash: GOVERNANCE_HASHES.evidence,
});
export const comparisonPayload = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, leftRunId: GOVERNANCE_OTHER_RUN_ID, rightRunId: GOVERNANCE_RUN_ID, artifactId: GOVERNANCE_ARTIFACT_ID,
  datasetContentHash: GOVERNANCE_HASHES.dataset, graphVersionHash: GOVERNANCE_HASHES.graph, comparedNodes: 1,
  changes: [{ nodeId: "n1", leftScore: 0.8, rightScore: 0.92, scoreDelta: 0.12, leftRank: 2, rightRank: 1, rankDelta: -1, riskBandChanged: true }],
  groupSummary: { added: 1 }, reviewSummary: { changed: 1 }, comparisonHash: GOVERNANCE_HASHES.comparison,
});
export const casePayload = () => ({
  schemaVersion: GOVERNANCE_ONLINE_SCHEMA, caseId: `case-${"4".repeat(32)}`, runId: GOVERNANCE_RUN_ID, title: "Global 研判",
  description: "本机研判单", state: "draft", createdAt: "2026-08-18T04:04:00Z", updatedAt: "2026-08-18T04:04:00Z",
  items: [], reviewEvents: [], currentDecisions: {}, caseHash: "0".repeat(64),
});
