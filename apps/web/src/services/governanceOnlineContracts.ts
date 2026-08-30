import {
  GOVERNANCE_ADAPTATION_COMPARISON_SCHEMA,
  GOVERNANCE_INPUT_SCHEMA,
  GOVERNANCE_ONLINE_SCHEMA,
  GOVERNANCE_RELATION_MODALITIES,
  GOVERNANCE_TARGET_PACKAGE_RECEIPT_SCHEMA,
  GOVERNANCE_TARGET_LABEL_RECIPE_SCHEMA,
  GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA,
  GOVERNANCE_ADAPTATION_POLICY_SCHEMA,
  GOVERNANCE_TARGET_TASK_REGISTRATION_SCHEMA,
  GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA,
  GOVERNANCE_TARGET_POLICY_SCHEMA,
  GOVERNANCE_TARGET_COMPARISON_SCHEMA,
  GOVERNANCE_ADAPTATION_HANDOFF_SCHEMA,
  GOVERNANCE_ADAPTATION_OVERLAY_SCHEMA,
  type AdaptationComparison,
  type AdaptationLabelEvidence,
  type AdaptationSourceRecord,
  type GovernanceAdaptationBinding,
  type GovernanceTargetLabelRecipe,
  type GovernanceTargetPackageReceipt,
  type GovernanceArtifact,
  type GovernanceArtifactCompatibility,
  type GovernanceCase,
  type GovernanceCasePage,
  type GovernanceCaseItem,
  type GovernanceDerivation,
  type GovernanceDerivationPage,
  type GovernanceEvidenceNeighbor,
  type GovernanceEvidenceRelation,
  type GovernanceEvidenceStructuralSignals,
  type GovernanceEvidenceSubgraph,
  type GovernanceEvidenceSubgraphEdge,
  type GovernanceEvidenceSubgraphNode,
  type GovernanceFindingPage,
  type GovernanceModalityCounts,
  type GovernanceOnlineCapabilities,
  type GovernanceOnlineEvidence,
  type GovernanceOnlineFinding,
  type GovernanceOnlineHealth,
  type GovernanceOnlinePreview,
  type GovernanceOnlinePreviewEdge,
  type GovernanceOnlinePreviewNode,
  type GovernanceProjectionPreset,
  type GovernanceOnlineRoute,
  type GovernanceOnlineRun,
  type GovernanceOnlineRunRequest,
  type GovernanceOnlineResult,
  type GovernanceReviewEvent,
  type GovernanceRunComparison,
  type AdaptationLabelSet,
  type AdaptationLabelSetCreateRequest,
  type AdaptationReviewPolicy,
  type TargetTaskRegistration,
  type RegisteredTargetLabelSetCreateRequest,
  type RegisteredTargetLabelSet,
  type TargetReviewPolicy,
  type TargetAdaptationComparison,
  type AdaptationGovernanceHandoff,
  type AdaptationOverlayActivation,
  type TargetReviewCollection,
} from "../types/governanceOnline";
import { deepFreeze } from "./coreContracts";
import { GOVERNANCE_PUBLIC_SKILLS } from "../types/governanceSkills";
import { canonicalJson, sha256Canonical, sha256Text } from "./graphIdentity";

const HASH = /^[0-9a-f]{64}$/u;
const ARTIFACT_ID = /^governance-artifact-[0-9a-f]{32}$/u;
const RUN_ID = /^governance-[0-9a-f]{32}$/u;
const CASE_ID = /^case-[0-9a-f]{32}$/u;
const ITEM_ID = /^item-[0-9a-f]{32}$/u;
const EVENT_ID = /^event-[0-9a-f]{32}$/u;
const TARGET_TASK_ID = /^target-task-[0-9a-f]{32}$/u;
const SAFE_TEXT_ID = /^[^\u0000-\u001f\u007f]{1,300}$/u;
const MODALITIES = new Set<string>(GOVERNANCE_RELATION_MODALITIES);
const RUN_STATUSES = new Set(["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]);
const RUN_STAGES = new Set(["queued", "validating", "preprocessing", "inferencing", "deriving", "freezing", "completed"]);
const RISK_BANDS = new Set(["low", "review", "high"]);
const EXPERTS = new Set(["shared", "domain:china", "domain:cuba", "domain:iran", "domain:russia", "domain:UAE", "domain:venezuela", "null"]);
const PROJECTION_PRESETS = new Set(["overview", "relation", "evidence", "groups"]);

function fail(): never { throw new Error("GFM_GOVERNANCE_RESPONSE_INVALID"); }
function object(value: unknown): Record<string, unknown> { if (!value || typeof value !== "object" || Array.isArray(value)) return fail(); return value as Record<string, unknown>; }
function text(value: unknown, max = 2_000): string { if (typeof value !== "string" || value.length === 0 || value.length > max) return fail(); return value; }
function nullableText(value: unknown, max = 2_000): string | null { return value === null ? null : text(value, max); }
function finite(value: unknown, min = -Infinity, max = Infinity): number { if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) return fail(); return value; }
function integer(value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): number { const result = finite(value, min, max); if (!Number.isInteger(result)) return fail(); return result; }
function bool(value: unknown): boolean { if (typeof value !== "boolean") return fail(); return value; }
function list(value: unknown, max: number): readonly unknown[] { if (!Array.isArray(value) || value.length > max) return fail(); return value; }
function oneOf<T extends string>(value: unknown, values: ReadonlySet<string>): T { const result = text(value, 100); if (!values.has(result)) return fail(); return result as T; }
function pattern(value: unknown, expression: RegExp): string { const result = text(value, 300); if (!expression.test(result)) return fail(); return result; }
function hash(value: unknown): string { return pattern(value, HASH); }
function nullableHash(value: unknown): string | null { return value === null ? null : hash(value); }
function safeId(value: unknown, max = 300): string { const result = text(value, max); if (!SAFE_TEXT_ID.test(result) || result.trim() !== result) return fail(); return result; }
function date(value: unknown): string { const result = text(value, 64); if (!Number.isFinite(Date.parse(result))) return fail(); return result; }
function nullableDate(value: unknown): string | null { return value === null ? null : date(value); }
function schema(value: Record<string, unknown>): void { if (value.schemaVersion !== GOVERNANCE_ONLINE_SCHEMA) fail(); }
function stringArray(value: unknown, max: number): readonly string[] { return list(value, max).map((entry) => text(entry, 2_000)); }
function modalities(value: unknown, requireAll = false): readonly (typeof GOVERNANCE_RELATION_MODALITIES[number])[] {
  const result = list(value, 5).map((entry) => oneOf<typeof GOVERNANCE_RELATION_MODALITIES[number]>(entry, MODALITIES));
  if (!result.length || new Set(result).size !== result.length) fail();
  if (requireAll && result.join("\u0000") !== GOVERNANCE_RELATION_MODALITIES.join("\u0000")) fail();
  return result;
}
function numberMap(value: unknown): Readonly<Record<string, number>> { return Object.freeze(Object.fromEntries(Object.entries(object(value)).map(([key, item]) => [key, finite(item)]))); }

function route(value: unknown): GovernanceOnlineRoute { const item = object(value); return { expert: oneOf(item.expert, EXPERTS), weight: finite(item.weight, 0, 1) }; }
function modalityCounts(value: unknown, requireAll = false): GovernanceModalityCounts {
  const item = object(value); const result: Partial<Record<typeof GOVERNANCE_RELATION_MODALITIES[number], number>> = {};
  for (const [key, count] of Object.entries(item)) { if (!MODALITIES.has(key)) fail(); result[key as keyof typeof result] = integer(count); }
  if (requireAll && (Object.keys(result).length !== GOVERNANCE_RELATION_MODALITIES.length || GOVERNANCE_RELATION_MODALITIES.some((key) => result[key] === undefined))) fail();
  return Object.freeze(result);
}
function finding(value: unknown): GovernanceOnlineFinding {
  const item = object(value); const routes = list(item.routes, 3).map(route); const contribution = object(item.modalityContribution);
  const riskBand = oneOf<GovernanceOnlineFinding["riskBand"]>(item.riskBand, RISK_BANDS); const predictedPositive = bool(item.predictedPositive);
  if (routes.length !== 3 || routes[0]?.expert !== "shared" || routes[0].weight !== 1 || new Set(routes.map((entry) => entry.expert)).size !== 3 || Math.abs((routes[1]?.weight ?? 0) + (routes[2]?.weight ?? 0) - 1) > 1e-5 || (riskBand === "high") !== predictedPositive) fail();
  return {
    nodeId: safeId(item.nodeId, 128), label: item.label === null ? null : text(item.label, 256), score: finite(item.score, 0, 1),
    logit: finite(item.logit), rank: integer(item.rank, 1), riskBand,
    predictedPositive, structureMissing: bool(item.structureMissing), routes,
    modalityContribution: { text: finite(contribution.text, 0, 1), structure: finite(contribution.structure, 0, 1) },
    modalityEvidence: modalityCounts(item.modalityEvidence, true), communityId: item.communityId === null ? null : safeId(item.communityId),
  };
}

export function parseGovernanceOnlineHealth(value: unknown): GovernanceOnlineHealth {
  const item = object(value); schema(item); const servingReady = bool(item.servingReady); const onlineForwardReady = bool(item.onlineForwardReady);
  if (servingReady !== onlineForwardReady) fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_ONLINE_SCHEMA, serviceIdentity: hash(item.serviceIdentity), servingReady, onlineForwardReady,
    modelVersionId: nullableText(item.modelVersionId, 200), modelVersionHash: nullableHash(item.modelVersionHash), modelStateHash: nullableHash(item.modelStateHash),
    device: oneOf(item.device, new Set(["cpu", "cuda"])), dtype: oneOf(item.dtype, new Set(["float32", "float16", "bfloat16"])),
    loadedAt: nullableDate(item.loadedAt), queueDepth: integer(item.queueDepth), activeRunId: item.activeRunId === null ? null : pattern(item.activeRunId, RUN_ID),
    runtimeRecipeHash: hash(item.runtimeRecipeHash), healthHash: hash(item.healthHash),
  });
}

export function parseGovernanceOnlineCapabilities(value: unknown): GovernanceOnlineCapabilities {
  const item = object(value); schema(item);
  if (item.channel !== "governance" || item.taskId !== "coordination_risk" || item.inputSchemaVersion !== GOVERNANCE_INPUT_SCHEMA) fail();
  const supported = list(item.supportedProtocols, 1); if (supported.length !== 1 || supported[0] !== "global") fail();
  const skills = list(item.skills, GOVERNANCE_PUBLIC_SKILLS.length);
  if (skills.length !== GOVERNANCE_PUBLIC_SKILLS.length || GOVERNANCE_PUBLIC_SKILLS.some((skill, index) => skills[index] !== skill)) fail();
  const servingReady = bool(item.servingReady); const onlineForwardReady = bool(item.onlineForwardReady); if (servingReady !== onlineForwardReady) fail();
  if (servingReady && (item.modelVersionId === null || item.modelVersionHash === null || item.modelStateHash === null)) fail();
  const limits = object(item.limits);
  return deepFreeze({
    schemaVersion: GOVERNANCE_ONLINE_SCHEMA, channel: "governance", taskId: "coordination_risk", servingReady, onlineForwardReady,
    unavailableReason: nullableText(item.unavailableReason), modelVersionId: nullableText(item.modelVersionId, 200),
    modelVersionHash: nullableHash(item.modelVersionHash), modelStateHash: nullableHash(item.modelStateHash), supportedProtocols: ["global"],
    skills: GOVERNANCE_PUBLIC_SKILLS,
    inputSchemaVersion: GOVERNANCE_INPUT_SCHEMA, modalities: modalities(item.modalities, true) as typeof GOVERNANCE_RELATION_MODALITIES,
    sampleArtifactId: item.sampleArtifactId === null ? null : pattern(item.sampleArtifactId, ARTIFACT_ID),
    limits: { maxNodes: integer(limits.maxNodes, 1), maxRelationRows: integer(limits.maxRelationRows, 1), maxEvidenceNodes: integer(limits.maxEvidenceNodes, 1), maxEvidenceEdges: integer(limits.maxEvidenceEdges, 1), maxPreviewNodes: integer(limits.maxPreviewNodes, 1, 3000), maxPreviewEdges: integer(limits.maxPreviewEdges, 1, 12000) },
    capabilityHash: hash(item.capabilityHash),
  });
}

export function parseGovernanceArtifact(value: unknown): GovernanceArtifact {
  const item = object(value); schema(item); if (item.compatibility !== "compatible") fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_ONLINE_SCHEMA, artifactId: pattern(item.artifactId, ARTIFACT_ID),
    ...(item.datasetId === undefined ? {} : { datasetId: text(item.datasetId, 100) }), ...(item.displayName === undefined ? {} : { displayName: text(item.displayName, 200) }),
    datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash),
    ...(item.bundleSha256 === undefined ? {} : { bundleSha256: hash(item.bundleSha256) }), ...(item.manifestSha256 === undefined ? {} : { manifestSha256: hash(item.manifestSha256) }),
    nodeCount: integer(item.nodeCount, 1, 10_000), relationRowCount: integer(item.relationRowCount, 1, 500_000), selfLoopsRemoved: integer(item.selfLoopsRemoved),
    ...(item.cleanSelfLoops === undefined ? {} : { cleanSelfLoops: bool(item.cleanSelfLoops) }), modalities: modalities(item.modalities), compatibility: "compatible",
    createdAt: date(item.createdAt), artifactHash: hash(item.artifactHash),
  });
}

export function parseGovernanceArtifactCompatibility(value: unknown): GovernanceArtifactCompatibility {
  const item = object(value); schema(item); if (item.inputSchemaVersion !== GOVERNANCE_INPUT_SCHEMA) fail();
  const selfLoopsDetected = integer(item.selfLoopsDetected); const requiresSelfLoopCleaning = bool(item.requiresSelfLoopCleaning);
  if (requiresSelfLoopCleaning !== (selfLoopsDetected > 0)) fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_ONLINE_SCHEMA, inputSchemaVersion: GOVERNANCE_INPUT_SCHEMA,
    compatible: bool(item.compatible), requiresSelfLoopCleaning,
    prospectiveArtifactId: pattern(item.prospectiveArtifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash),
    nodeCount: integer(item.nodeCount, 1, 10_000), relationRowCount: integer(item.relationRowCount, 1, 500_000), selfLoopsDetected,
    modalities: modalities(item.modalities), issues: stringArray(item.issues, 100), compatibilityHash: hash(item.compatibilityHash),
  });
}

function previewNode(value: unknown): GovernanceOnlinePreviewNode { const item = object(value); return { id: safeId(item.id, 128), label: text(item.label, 256), degree: integer(item.degree), structureMissing: bool(item.structureMissing), score: item.score === null ? null : finite(item.score, 0, 1), riskBand: item.riskBand === null ? null : oneOf(item.riskBand, RISK_BANDS), groupId: item.groupId === null ? null : safeId(item.groupId) }; }
function previewEdge(value: unknown): GovernanceOnlinePreviewEdge { const item = object(value); return { id: safeId(item.id), source: safeId(item.source, 128), target: safeId(item.target, 128), modalities: modalities(item.modalities), factual: bool(item.factual) }; }
export function parseGovernanceOnlinePreview(value: unknown): GovernanceOnlinePreview {
  const item = object(value); schema(item); const nodes = list(item.nodes, 3000).map(previewNode); const edges = list(item.edges, 12000).map(previewEdge);
  const runId = item.runId === null ? null : pattern(item.runId, RUN_ID);
  const resultHash = item.resultHash === null ? null : hash(item.resultHash);
  if ((runId === null) !== (resultHash === null)) fail();
  const numericRecord = (value: unknown) => Object.fromEntries(Object.entries(object(value)).map(([key, count]) => [key, integer(count)]));
  const partialPreview = bool(item.partialPreview);
  const projected = item.preset === undefined ? {} : {
    preset: oneOf<GovernanceProjectionPreset>(item.preset, PROJECTION_PRESETS),
    budgets: numericRecord(item.budgets),
    selectionRecipeId: text(item.selectionRecipeId, 200),
    isPartial: bool(item.isPartial),
    groups: list(item.groups, 24).map((entry) => object(entry)),
    sourceCounts: numericRecord(item.sourceCounts),
    inventoryCounts: numericRecord(item.inventoryCounts),
  };
  if ("isPartial" in projected && projected.isPartial !== partialPreview) fail();
  return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash), runId, resultHash, nodes, edges, nodeCount: integer(item.nodeCount, nodes.length), edgeCount: integer(item.edgeCount, edges.length), partialPreview, previewHash: hash(item.previewHash), ...projected });
}

export function parseGovernanceOnlineRunRequest(value: unknown): GovernanceOnlineRunRequest {
  const item = object(value); schema(item); if (item.protocol !== "global") fail();
  return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, protocol: "global", artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash), modelVersionId: text(item.modelVersionId, 200), modelStateHash: hash(item.modelStateHash), topK: integer(item.topK, 1, 10_000) });
}
export function parseGovernanceOnlineRun(value: unknown): GovernanceOnlineRun {
  const item = object(value); schema(item);
  return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: pattern(item.runId, RUN_ID), requestHash: hash(item.requestHash), artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash), modelVersionId: text(item.modelVersionId, 200), modelVersionHash: hash(item.modelVersionHash), modelStateHash: hash(item.modelStateHash), status: oneOf(item.status, RUN_STATUSES), stage: oneOf(item.stage, RUN_STAGES), progress: integer(item.progress, 0, 100), createdAt: date(item.createdAt), updatedAt: date(item.updatedAt), errorCode: nullableText(item.errorCode, 100), cancelRequested: bool(item.cancelRequested), statusHash: hash(item.statusHash) });
}
export function parseGovernanceOnlineRuns(value: unknown): readonly GovernanceOnlineRun[] { const item = object(value); schema(item); const rows = list(item.items, 1000).map(parseGovernanceOnlineRun); if (integer(item.total) < rows.length) fail(); integer(item.offset); integer(item.limit, 1, 10_000); return deepFreeze(rows); }

export function parseGovernanceRunComparison(value: unknown): GovernanceRunComparison {
  const item = object(value); schema(item);
  const integerSummary = (source: unknown) => Object.freeze(Object.fromEntries(Object.entries(object(source)).map(([key, count]) => [key, integer(count)])));
  return deepFreeze({
    schemaVersion: GOVERNANCE_ONLINE_SCHEMA,
    leftRunId: pattern(item.leftRunId, RUN_ID), rightRunId: pattern(item.rightRunId, RUN_ID),
    artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash),
    comparedNodes: integer(item.comparedNodes, 1, 10_000),
    changes: list(item.changes, 10_000).map((entry) => { const change = object(entry); return {
      nodeId: safeId(change.nodeId, 128), leftScore: finite(change.leftScore, 0, 1), rightScore: finite(change.rightScore, 0, 1),
      scoreDelta: finite(change.scoreDelta, -1, 1), leftRank: integer(change.leftRank, 1), rightRank: integer(change.rightRank, 1),
      rankDelta: integer(change.rankDelta, -10_000, 10_000), riskBandChanged: bool(change.riskBandChanged),
    }; }),
    groupSummary: integerSummary(item.groupSummary), reviewSummary: integerSummary(item.reviewSummary), comparisonHash: hash(item.comparisonHash),
  });
}

export function parseGovernanceOnlineResult(value: unknown): GovernanceOnlineResult {
  const item = object(value); schema(item); if (item.datasetMetrics !== null) fail(); const calibration = object(item.calibration); const distribution = object(item.distribution);
  const result = {
    schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: pattern(item.runId, RUN_ID), requestHash: hash(item.requestHash), artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash), modelVersionId: text(item.modelVersionId, 200), modelVersionHash: hash(item.modelVersionHash), modelStateHash: hash(item.modelStateHash), threshold: finite(item.threshold, 0, 1),
    calibration: { temperature: finite(calibration.temperature, Number.MIN_VALUE), bias: finite(calibration.bias), referenceThreshold: finite(calibration.referenceThreshold, 0, 1), applicability: oneOf<"reference_replay" | "out_of_domain_unverified">(calibration.applicability, new Set(["reference_replay", "out_of_domain_unverified"])) },
    referenceMetrics: Object.freeze({ ...object(item.referenceMetrics) }), datasetMetrics: null,
    distribution: { low: integer(distribution.low), review: integer(distribution.review), high: integer(distribution.high), predictedPositive: integer(distribution.predictedPositive), total: integer(distribution.total, 1) },
    findings: list(item.findings, 10_000).map(finding), totalFindings: integer(item.totalFindings, 1), limitations: stringArray(item.limitations, 100), completedAt: date(item.completedAt), resultHash: hash(item.resultHash),
  } as const;
  if (result.distribution.low + result.distribution.review + result.distribution.high !== result.distribution.total || result.distribution.predictedPositive !== result.distribution.high || result.findings.length > result.totalFindings || !result.limitations.length || new Set(result.findings.map((entry) => entry.nodeId)).size !== result.findings.length || result.findings.some((entry, index) => entry.rank !== index + 1)) fail();
  return deepFreeze(result);
}

export function parseCoreFindingPage(value: unknown): GovernanceFindingPage { const item = object(value); schema(item); const items = list(item.items, 10_000).map(finding); const total = integer(item.total); const offset = integer(item.offset, 0, total); const limit = integer(item.limit, 1, 10_000); if (items.length > limit || offset + items.length > total || new Set(items.map((entry) => entry.nodeId)).size !== items.length || items.some((entry, index) => entry.rank !== offset + index + 1)) fail(); return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: pattern(item.runId, RUN_ID), items, total, offset, limit, pageHash: hash(item.pageHash) }); }

function derivation(value: unknown): GovernanceDerivation {
  const item = object(value); const kind = oneOf<GovernanceDerivation["kind"]>(item.kind, new Set(["group", "factual_relation", "potential_link"])); const nodeIds = list(item.nodeIds, 10_000).map((entry) => safeId(entry, 128));
  const source = item.source === null ? null : safeId(item.source, 128); const target = item.target === null ? null : safeId(item.target, 128);
  const relationModalities = item.modalities instanceof Array && item.modalities.length === 0 ? [] : modalities(item.modalities);
  const memberCount = item.memberCount === null ? null : integer(item.memberCount, 1); const meanScore = item.meanScore === null ? null : finite(item.meanScore, 0, 1); const p90Score = item.p90Score === null ? null : finite(item.p90Score, 0, 1); const factual = bool(item.factual);
  if (!nodeIds.length || factual !== (kind === "factual_relation")) fail();
  if (kind === "group") {
    if (source !== null || target !== null || memberCount !== nodeIds.length || memberCount < 2 || meanScore === null || p90Score === null) fail();
  } else if (source === null || target === null || source === target || nodeIds.length !== 2 || nodeIds[0] !== source || nodeIds[1] !== target || memberCount !== null || meanScore !== null || p90Score !== null || kind === "factual_relation" && !relationModalities.length) fail();
  return { id: safeId(item.id), kind, priority: finite(item.priority, 0, 1), nodeIds, source, target, modalities: relationModalities, memberCount, meanScore, p90Score, scoreComponents: numberMap(item.scoreComponents), factual, limitation: text(item.limitation, 2_000) };
}
export function parseGovernanceDerivationPage(value: unknown): GovernanceDerivationPage { const item = object(value); schema(item); const items = list(item.items, 10_000).map(derivation); const total = integer(item.total); const offset = integer(item.offset, 0, total); const limit = integer(item.limit, 1, 10_000); if (items.length > limit || offset + items.length > total) fail(); return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: pattern(item.runId, RUN_ID), items, total, offset, limit, pageHash: hash(item.pageHash) }); }

function evidenceRelation(value: unknown): GovernanceEvidenceRelation { const item = object(value); return { modality: oneOf(item.modality, MODALITIES), rawWeight: finite(item.rawWeight, 0) }; }
function evidenceRelations(value: unknown): readonly GovernanceEvidenceRelation[] { const result = list(value, 5).map(evidenceRelation); const keys = result.map((entry) => entry.modality); if (!result.length || new Set(keys).size !== keys.length || keys.some((key, index) => GOVERNANCE_RELATION_MODALITIES.indexOf(key) <= GOVERNANCE_RELATION_MODALITIES.indexOf(keys[index - 1] ?? key) && index > 0)) fail(); return result; }
function evidenceNeighbor(value: unknown): GovernanceEvidenceNeighbor { const item = object(value); const riskBand = oneOf<GovernanceOnlineFinding["riskBand"]>(item.riskBand, RISK_BANDS); const predictedPositive = bool(item.predictedPositive); const neighborModalities = modalities(item.modalities); const relations = evidenceRelations(item.relations); if (item.hop !== 1 || (riskBand === "high") !== predictedPositive || neighborModalities.length !== relations.length || neighborModalities.some((entry, index) => entry !== relations[index]?.modality)) fail(); return { nodeId: safeId(item.nodeId, 128), score: finite(item.score, 0, 1), hop: 1, riskBand, predictedPositive, structureMissing: bool(item.structureMissing), modalities: neighborModalities, relations }; }
function evidenceSubgraphNode(value: unknown): GovernanceEvidenceSubgraphNode { const item = object(value); const riskBand = oneOf<GovernanceOnlineFinding["riskBand"]>(item.riskBand, RISK_BANDS); const predictedPositive = bool(item.predictedPositive); const hop = integer(item.hop, 0, 2) as 0 | 1 | 2; if ((riskBand === "high") !== predictedPositive) fail(); return { nodeId: safeId(item.nodeId, 128), score: finite(item.score, 0, 1), hop, riskBand, predictedPositive, structureMissing: bool(item.structureMissing) }; }
function evidenceSubgraphEdge(value: unknown): GovernanceEvidenceSubgraphEdge { const item = object(value); if (item.evidenceRole !== "explanationOnly") fail(); return { id: safeId(item.id), source: safeId(item.source, 128), target: safeId(item.target, 128), relations: evidenceRelations(item.relations), evidenceRole: "explanationOnly" }; }
function evidenceSubgraph(value: unknown): GovernanceEvidenceSubgraph { const item = object(value); if (item.depth !== 2) fail(); const nodes = list(item.nodes, 300).map(evidenceSubgraphNode); const edges = list(item.edges, 1000).map(evidenceSubgraphEdge); const nodeIds = new Set(nodes.map((entry) => entry.nodeId)); if (!nodes.length || integer(item.nodeCount, 1, 300) !== nodes.length || integer(item.edgeCount, 0, 1_000) !== edges.length || nodeIds.size !== nodes.length || nodes.filter((entry) => entry.hop === 0).length !== 1 || edges.some((entry) => !nodeIds.has(entry.source) || !nodeIds.has(entry.target)) || new Set(edges.map((entry) => entry.id)).size !== edges.length) fail(); return { depth: 2, nodeCount: nodes.length, edgeCount: edges.length, truncated: bool(item.truncated), nodes, edges }; }
function evidenceStructuralSignals(value: unknown): GovernanceEvidenceStructuralSignals { const item = object(value); if (item.relationEvidenceRole !== "explanationOnly") fail(); return { fusedDegree: integer(item.fusedDegree, 0, 9_999), structureMissing: bool(item.structureMissing), relationNeighborCounts: modalityCounts(item.relationNeighborCounts, true) as Readonly<Record<typeof GOVERNANCE_RELATION_MODALITIES[number], number>>, twoHopNodeCount: integer(item.twoHopNodeCount, 0, 9_999), relationEvidenceRole: "explanationOnly" }; }
export function parseGovernanceOnlineEvidence(value: unknown): GovernanceOnlineEvidence {
  const item = object(value); schema(item); const node = finding(item.node); const neighbors = list(item.neighbors, 300).map(evidenceNeighbor); const structuralSignals = evidenceStructuralSignals(item.structuralSignals); const subgraph = evidenceSubgraph(item.evidenceSubgraph); const truncated = bool(item.truncated); const root = subgraph.nodes.find((entry) => entry.hop === 0);
  if (!root || root.nodeId !== node.nodeId || root.score !== node.score || truncated !== subgraph.truncated || structuralSignals.structureMissing !== node.structureMissing || structuralSignals.fusedDegree < neighbors.length || neighbors.some((neighbor) => !subgraph.nodes.some((entry) => entry.nodeId === neighbor.nodeId))) fail();
  return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: pattern(item.runId, RUN_ID), resultHash: hash(item.resultHash), artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash), modelVersionId: text(item.modelVersionId, 200), modelVersionHash: hash(item.modelVersionHash), modelStateHash: hash(item.modelStateHash), threshold: finite(item.threshold, 0, 1), node, neighbors, structuralSignals, evidenceSubgraph: subgraph, truncated, limitation: text(item.limitation, 1_000), evidenceHash: hash(item.evidenceHash) });
}

function caseItem(value: unknown): GovernanceCaseItem { const item = object(value); return { itemId: pattern(item.itemId, ITEM_ID), targetType: oneOf(item.targetType, new Set(["node", "relation", "group"])), targetId: safeId(item.targetId), note: typeof item.note === "string" ? item.note.slice(0, 2_000) : fail(), createdAt: date(item.createdAt), itemHash: hash(item.itemHash) }; }
function reviewEvent(value: unknown): GovernanceReviewEvent { const item = object(value); return { eventId: pattern(item.eventId, EVENT_ID), targetType: oneOf(item.targetType, new Set(["node", "relation", "group"])), targetId: safeId(item.targetId), decision: oneOf(item.decision, new Set(["confirmed", "rejected", "pending"])), reason: text(item.reason, 2_000), actor: text(item.actor, 100), sequence: integer(item.sequence, 1), createdAt: date(item.createdAt), previousEventHash: nullableHash(item.previousEventHash), eventHash: hash(item.eventHash) }; }
export function parseGovernanceCase(value: unknown): GovernanceCase { const item = object(value); schema(item); const decisions = object(item.currentDecisions); for (const decision of Object.values(decisions)) oneOf(decision, new Set(["confirmed", "rejected", "pending"])); return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, caseId: pattern(item.caseId, CASE_ID), runId: pattern(item.runId, RUN_ID), title: text(item.title, 200), description: typeof item.description === "string" ? item.description.slice(0, 2_000) : fail(), state: oneOf(item.state, new Set(["draft", "active", "concluded", "archived"])), createdAt: date(item.createdAt), updatedAt: date(item.updatedAt), items: list(item.items, 10_000).map(caseItem), reviewEvents: list(item.reviewEvents, 10_000).map(reviewEvent), currentDecisions: decisions as Readonly<Record<string, "confirmed" | "rejected" | "pending">>, caseHash: hash(item.caseHash) }); }
export function parseGovernanceCases(value: unknown): GovernanceCasePage { const item = object(value); schema(item); const items = list(item.items, 100).map(parseGovernanceCase); const total = integer(item.total); const offset = integer(item.offset, 0, total); const limit = integer(item.limit, 1, 100); if (items.length > limit || offset + items.length > total) fail(); return deepFreeze({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, items, total, offset, limit }); }

function adaptationBinding(value: unknown): GovernanceAdaptationBinding {
  const item = object(value);
  return {
    artifactId: pattern(item.artifactId, ARTIFACT_ID), datasetContentHash: hash(item.datasetContentHash), graphVersionHash: hash(item.graphVersionHash),
    runId: pattern(item.runId, RUN_ID), requestHash: hash(item.requestHash), resultHash: hash(item.resultHash), runArtifactHash: hash(item.runArtifactHash),
    modelVersionId: text(item.modelVersionId, 200), modelVersionHash: hash(item.modelVersionHash), modelStateHash: hash(item.modelStateHash),
    recipeHash: hash(item.recipeHash), codeHash: hash(item.codeHash), seed: integer(item.seed, 0, Number.MAX_SAFE_INTEGER),
  };
}

function sameBinding(left: GovernanceAdaptationBinding, right: GovernanceAdaptationBinding): boolean {
  return Object.keys(left).every((key) => left[key as keyof GovernanceAdaptationBinding] === right[key as keyof GovernanceAdaptationBinding]);
}

function parseLabelSelectionRecipe(value: unknown): GovernanceTargetLabelRecipe["selectionRecipe"] {
  const selection = object(value);
  if (selection.version !== "graph-fused-degree-quartile-stable-hash-v2"
    || selection.stratification !== "graph-fused-degree-rank-quartile"
    || selection.structuralStrata !== 4
    || selection.labelsPerClass !== 8
    || selection.labelsPerClassPerStratum !== 2
    || list(selection.scoreInputs, 0).length) fail();
  return {
    version: "graph-fused-degree-quartile-stable-hash-v2",
    stratification: "graph-fused-degree-rank-quartile",
    structuralStrata: 4,
    labelsPerClass: 8,
    labelsPerClassPerStratum: 2,
    scoreInputs: [],
  };
}

function validateLabelStrata(labels: readonly { label: "io" | "control"; structuralStratum: 0 | 1 | 2 | 3 }[]): void {
  for (const label of ["io", "control"] as const) {
    for (const stratum of [0, 1, 2, 3] as const) {
      if (labels.filter((row) => row.label === label && row.structuralStratum === stratum).length !== 2) fail();
    }
  }
}

export function parseTargetLabelRecipe(value: unknown): GovernanceTargetLabelRecipe {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_TARGET_LABEL_RECIPE_SCHEMA) fail();
  const selectionRecipe = parseLabelSelectionRecipe(item.selectionRecipe);
  const labels = list(item.labels, 16).map((entry) => {
    const row = object(entry);
    const structuralStratum = integer(row.structuralStratum, 0, 3) as 0 | 1 | 2 | 3;
    return { nodeId: safeId(row.nodeId, 128), label: oneOf<"io" | "control">(row.label, new Set(["io", "control"])), structuralStratum, fusedDegree: integer(row.fusedDegree) };
  });
  if (labels.length !== 16 || new Set(labels.map((row) => row.nodeId)).size !== 16 || labels.filter((row) => row.label === "io").length !== 8 || labels.filter((row) => row.label === "control").length !== 8) fail();
  validateLabelStrata(labels);
  return deepFreeze({
    schemaVersion: GOVERNANCE_TARGET_LABEL_RECIPE_SCHEMA, datasetId: safeId(item.datasetId, 200), bundleSha256: hash(item.bundleSha256),
    selectionRecipe, labels,
  });
}

export function parseTargetPackageReceipt(value: unknown): GovernanceTargetPackageReceipt {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_TARGET_PACKAGE_RECEIPT_SCHEMA
    || item.sourceSchemaVersion !== "socialgraph-fm.anonymized-posts/1.0") fail();
  const encoder = object(item.encoder);
  if (encoder.dimension !== 768) fail();
  const selection = object(item.selectionRecipe);
  const groupRelations = object(selection.groupRelations);
  const fastRT = object(selection.fastRT);
  const tweetSim = object(selection.tweetSim);
  if (selection.version !== "connected-structural-hash-v2" || selection.nodeCount !== 128
    || selection.requiredIo !== 16 || selection.requiredControls !== 64
    || selection.minimumNonemptyModalities !== 4 || list(selection.scoreInputs, 0).length
    || groupRelations.maxGroupAccounts !== 256 || groupRelations.totalPotentialPairBudget !== 50_000
    || fastRT.windowSeconds !== 10 || fastRT.pairBudget !== 50_000 || fastRT.algorithm !== "sorted-sliding-window-v1"
    || tweetSim.mutualTopK !== 5 || tweetSim.cosineThreshold !== 0.8 || tweetSim.pairBudget !== 10_000) fail();
  const labelSelectionRecipe = parseLabelSelectionRecipe(item.labelSelectionRecipe);
  const coverage = object(item.coverage);
  const nonemptyModalities = list(coverage.nonemptyModalities, 5).map((entry) => oneOf(entry, MODALITIES)) as GovernanceTargetPackageReceipt["coverage"]["nonemptyModalities"];
  if (coverage.nodeCount !== 128 || integer(coverage.ioCount, 16) + integer(coverage.controlCount, 64) !== 128
    || coverage.connected !== true || nonemptyModalities.length < 4 || new Set(nonemptyModalities).size !== nonemptyModalities.length) fail();
  const logical = {
    schemaVersion: GOVERNANCE_TARGET_PACKAGE_RECEIPT_SCHEMA,
    datasetId: safeId(item.datasetId, 200),
    sourceSchemaVersion: "socialgraph-fm.anonymized-posts/1.0" as const,
    sourceSha256: hash(item.sourceSha256),
    authorizationReference: text(item.authorizationReference, 300),
    bundleSha256: hash(item.bundleSha256),
    labelsSha256: hash(item.labelsSha256),
    encoder: {
      modelId: text(encoder.modelId, 300), revision: text(encoder.revision, 200), cacheSha256: hash(encoder.cacheSha256),
      compatibility: oneOf<GovernanceTargetPackageReceipt["encoder"]["compatibility"]>(encoder.compatibility, new Set(["dimension-only-unverified", "pinned-production"])), dimension: 768 as const,
    },
    selectionRecipe: {
      version: "connected-structural-hash-v2" as const, nodeCount: 128 as const, requiredIo: 16 as const, requiredControls: 64 as const,
      minimumNonemptyModalities: 4 as const, scoreInputs: [] as const,
      groupRelations: { maxGroupAccounts: 256 as const, totalPotentialPairBudget: 50_000 as const },
      fastRT: { windowSeconds: 10 as const, pairBudget: 50_000 as const, algorithm: "sorted-sliding-window-v1" as const },
      tweetSim: { mutualTopK: 5 as const, cosineThreshold: 0.8 as const, pairBudget: 10_000 as const },
    },
    labelSelectionRecipe,
    coverage: {
      nodeCount: 128 as const, ioCount: integer(coverage.ioCount, 16), controlCount: integer(coverage.controlCount, 64),
      nonemptyModalities, connected: true as const,
    },
  };
  const receiptHash = hash(item.receiptHash);
  if (receiptHash !== sha256Canonical(logical)) fail();
  return deepFreeze({ ...logical, receiptHash });
}

export function parseTargetLabelSidecar(labelsText: string, receiptText: string): { readonly recipe: GovernanceTargetLabelRecipe; readonly receipt: GovernanceTargetPackageReceipt } {
  try {
    const recipe = parseTargetLabelRecipe(JSON.parse(labelsText));
    const receipt = parseTargetPackageReceipt(JSON.parse(receiptText));
    if (labelsText !== `${canonicalJson(recipe)}\n`
      || sha256Text(labelsText) !== receipt.labelsSha256
      || recipe.datasetId !== receipt.datasetId
      || recipe.bundleSha256 !== receipt.bundleSha256
      || canonicalJson(recipe.selectionRecipe) !== canonicalJson(receipt.labelSelectionRecipe)) fail();
    return deepFreeze({ recipe, receipt });
  } catch {
    return fail();
  }
}

export function parseTargetLabelSetCreateRequest(value: unknown): AdaptationLabelSetCreateRequest {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA) fail();
  const sidecarReceipt = parseTargetPackageReceipt(item.sidecarReceipt);
  const sources = list(item.sources, 256).map((entry) => {
    const source = object(entry);
    if (source.sourceType === "imported_sidecar") return {
      sourceType: "imported_sidecar" as const, sourceRecordId: safeId(source.sourceRecordId, 200), sourceRecordHash: hash(source.sourceRecordHash),
      nodeId: safeId(source.nodeId, 128), cohort: oneOf<"io" | "control">(source.cohort, new Set(["io", "control"])),
      structuralStratum: integer(source.structuralStratum, 0, 3) as 0 | 1 | 2 | 3, fusedDegree: integer(source.fusedDegree),
      labelsSha256: hash(source.labelsSha256), receiptHash: hash(source.receiptHash),
    };
    if (source.sourceType === "concluded_review") return {
      sourceType: "concluded_review" as const, caseId: pattern(source.caseId, CASE_ID), eventHash: hash(source.eventHash),
    };
    return fail();
  });
  const imported = sources.filter((source) => source.sourceType === "imported_sidecar");
  if (sources.length !== 16 || imported.length !== 16 || new Set(imported.map((source) => source.nodeId)).size !== 16) fail();
  validateLabelStrata(imported.map((source) => ({ label: source.cohort, structuralStratum: source.structuralStratum })));
  for (const source of imported) {
    if (source.labelsSha256 !== sidecarReceipt.labelsSha256 || source.receiptHash !== sidecarReceipt.receiptHash) fail();
    const expectedHash = sha256Canonical({
      schemaVersion: GOVERNANCE_TARGET_LABEL_RECIPE_SCHEMA, datasetId: sidecarReceipt.datasetId, bundleSha256: sidecarReceipt.bundleSha256,
      labelsSha256: sidecarReceipt.labelsSha256, receiptHash: sidecarReceipt.receiptHash,
      nodeId: source.nodeId, label: source.cohort, structuralStratum: source.structuralStratum, fusedDegree: source.fusedDegree,
    });
    if (source.sourceRecordHash !== expectedHash) fail();
  }
  return deepFreeze({ schemaVersion: GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA, runId: pattern(item.runId, RUN_ID), resultHash: hash(item.resultHash), sidecarReceipt, sources });
}

export function parseRegisteredTargetLabelSetCreateRequest(value: unknown): RegisteredTargetLabelSetCreateRequest {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA || item.sourceType !== "imported_sidecar") fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA,
    sourceType: "imported_sidecar",
    targetTaskRegistrationId: pattern(item.targetTaskRegistrationId, TARGET_TASK_ID),
    runId: pattern(item.runId, RUN_ID),
    resultHash: hash(item.resultHash),
  });
}

export function parseRegisteredTargetLabelSet(value: unknown): RegisteredTargetLabelSet {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA) fail();
  const labels = list(item.labels, 256).map((entry) => {
    const row = object(entry);
    return {
      nodeId: safeId(row.nodeId, 128),
      label: oneOf<"positive" | "negative">(row.label, new Set(["positive", "negative"])),
      structuralStratum: integer(row.structuralStratum, 0, 3) as 0 | 1 | 2 | 3,
      fusedDegree: integer(row.fusedDegree, 1),
    };
  });
  const positiveCount = integer(item.positiveCount, 4, 256);
  const negativeCount = integer(item.negativeCount, 4, 256);
  if (labels.length < 8 || new Set(labels.map((row) => row.nodeId)).size !== labels.length
    || labels.filter((row) => row.label === "positive").length !== positiveCount
    || labels.filter((row) => row.label === "negative").length !== negativeCount) fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA,
    taskId: safeId(item.taskId, 100), inferenceSha256: hash(item.inferenceSha256), labels,
    positiveCount, negativeCount, labelSetHash: hash(item.labelSetHash),
  });
}

export function parseTargetTaskRegistration(value: unknown): TargetTaskRegistration {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_TARGET_TASK_REGISTRATION_SCHEMA) fail();
  const task = object(item.task); const receipt = object(item.targetReceipt);
  if (task.schemaVersion !== "socialgraph-fm.governance-target-task-bundle/1.0"
    || receipt.schemaVersion !== "socialgraph-fm.governance-target-domain-receipt/2.0") fail();
  const mode = oneOf<"zero_shot" | "few_shot">(task.mode, new Set(["zero_shot", "few_shot"]));
  const taskModalities = modalities(task.modalities);
  const receiptModalities = modalities(receipt.modalities);
  const artifact = parseGovernanceArtifact(item.artifact);
  const labels = item.labels === null ? null : parseRegisteredTargetLabelSet(item.labels);
  const labelReceiptItem = item.labelReceipt === null ? null : object(item.labelReceipt);
  const labelReceipt = labelReceiptItem ? {
    schemaVersion: labelReceiptItem.schemaVersion === "socialgraph-fm.governance-target-label-receipt/2.0" ? "socialgraph-fm.governance-target-label-receipt/2.0" as const : fail(),
    taskId: safeId(labelReceiptItem.taskId, 100), targetReceiptHash: hash(labelReceiptItem.targetReceiptHash),
    receiptHash: hash(labelReceiptItem.receiptHash), labelsSha256: hash(labelReceiptItem.labelsSha256),
    sourceLabelsSha256: hash(labelReceiptItem.sourceLabelsSha256), eligibilityMaskSha256: hash(labelReceiptItem.eligibilityMaskSha256),
    eligibleNodeIds: Object.freeze(stringArray(labelReceiptItem.eligibleNodeIds, 10_000)),
  } : null;
  const taskId = safeId(task.taskId, 100); const nodeCount = integer(task.nodeCount, 1, 10_000);
  const fusedEdgeCount = integer(task.fusedEdgeCount, 1, 500_000);
  const hasLabels = labels !== null;
  const hasLabelReceipt = labelReceipt !== null;
  if (taskId !== safeId(receipt.taskId, 100) || nodeCount !== integer(receipt.nodeCount, 1, 10_000)
    || fusedEdgeCount !== integer(receipt.fusedEdgeCount, 1, 500_000)
    || nodeCount !== artifact.nodeCount || artifact.relationRowCount < fusedEdgeCount
    || taskModalities.join("\0") !== receiptModalities.join("\0")
    || hash(receipt.inferenceSha256) !== artifact.bundleSha256
    || hasLabels !== hasLabelReceipt
    || (mode === "few_shot") !== hasLabels
    || labels && (labels.taskId !== taskId || labels.inferenceSha256 !== artifact.bundleSha256)
    || labelReceipt && (labelReceipt.taskId !== taskId || labelReceipt.targetReceiptHash !== hash(receipt.receiptHash)
      || labelReceipt.labelsSha256 !== hash(object(task.labels).sha256)
      || labelReceipt.eligibilityMaskSha256 !== hash(receipt.labelEligibilityMaskSha256)
      || new Set(labelReceipt.eligibleNodeIds).size !== labelReceipt.eligibleNodeIds.length
      || labels?.labels.some((row) => !labelReceipt.eligibleNodeIds.includes(row.nodeId)))) fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_TARGET_TASK_REGISTRATION_SCHEMA,
    registrationId: pattern(item.registrationId, TARGET_TASK_ID), outerBundleSha256: hash(item.outerBundleSha256),
    task: { schemaVersion: "socialgraph-fm.governance-target-task-bundle/1.0", taskId, displayName: text(task.displayName, 200), mode, nodeCount, fusedEdgeCount, modalities: taskModalities },
    targetReceipt: {
      schemaVersion: "socialgraph-fm.governance-target-domain-receipt/2.0", taskId,
      receiptHash: hash(receipt.receiptHash), inferenceSha256: hash(receipt.inferenceSha256), nodeSetSha256: hash(receipt.nodeSetSha256),
      nodeCount, fusedEdgeCount, modalities: receiptModalities, connected: bool(receipt.connected),
    },
    labels, labelReceipt, artifact, createdAt: date(item.createdAt), registrationHash: hash(item.registrationHash),
  });
}

function sourceRecord(value: unknown): AdaptationSourceRecord {
  const item = object(value);
  return {
    sourceType: oneOf(item.sourceType, new Set(["concluded_review", "imported_sidecar"])), sourceRecordId: safeId(item.sourceRecordId, 200),
    sourceRecordHash: hash(item.sourceRecordHash), reviewEventHash: item.reviewEventHash === null ? null : hash(item.reviewEventHash),
  };
}

export function parseTargetLabelSet(value: unknown): AdaptationLabelSet {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA) fail();
  const binding = adaptationBinding(item.binding);
  const sidecarReceipt = item.sidecarReceipt === null ? null : parseTargetPackageReceipt(item.sidecarReceipt);
  const sourceRecords = list(item.sourceRecords, 256).map(sourceRecord);
  const labels = list(item.labels, 256).map((entry): AdaptationLabelEvidence => {
    const label = object(entry); const source = sourceRecord(label); const rowBinding = adaptationBinding(label.binding);
    if (!sameBinding(rowBinding, binding)) fail();
    if ((source.sourceType === "concluded_review") !== (source.reviewEventHash !== null)) fail();
    const imported = source.sourceType === "imported_sidecar";
    return {
      ...source, nodeId: safeId(label.nodeId, 128), label: oneOf(label.label, new Set(["positive", "negative"])), binding: rowBinding,
      ...(imported ? {
        structuralStratum: integer(label.structuralStratum, 0, 3) as 0 | 1 | 2 | 3,
        fusedDegree: integer(label.fusedDegree), labelsSha256: hash(label.labelsSha256), receiptHash: hash(label.receiptHash),
      } : {}),
    };
  });
  const conflicts = stringArray(item.conflicts, 256); const positiveCount = integer(item.positiveCount, 4); const negativeCount = integer(item.negativeCount, 4);
  if (conflicts.length || labels.length < 8 || sourceRecords.length !== labels.length || new Set(labels.map((row) => row.nodeId)).size !== labels.length || labels.filter((row) => row.label === "positive").length !== positiveCount || labels.filter((row) => row.label === "negative").length !== negativeCount) fail();
  if (sourceRecords.some((record, index) => {
    const label = labels[index];
    return !label
      || record.sourceType !== label.sourceType
      || record.sourceRecordId !== label.sourceRecordId
      || record.sourceRecordHash !== label.sourceRecordHash
      || record.reviewEventHash !== label.reviewEventHash;
  })) fail();
  const reviewEventHashes = list(item.reviewEventHashes, 256).map(hash);
  if (reviewEventHashes.join("\0") !== labels.flatMap((row) => row.reviewEventHash ? [row.reviewEventHash] : []).join("\0")) fail();
  const importedLabels = labels.filter((label) => label.sourceType === "imported_sidecar");
  if (importedLabels.length && (!sidecarReceipt || importedLabels.some((label) => label.labelsSha256 !== sidecarReceipt.labelsSha256 || label.receiptHash !== sidecarReceipt.receiptHash))) fail();
  return deepFreeze({ schemaVersion: GOVERNANCE_ADAPTATION_LABEL_SET_SCHEMA, binding, sidecarReceipt, sourceRecords, reviewEventHashes, labels, conflicts, positiveCount, negativeCount, labelSetHash: hash(item.labelSetHash) });
}

export function parseAdaptationReviewPolicy(value: unknown): AdaptationReviewPolicy {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_ADAPTATION_POLICY_SCHEMA) fail();
  const status = oneOf<AdaptationReviewPolicy["status"]>(item.status, new Set(["collecting_reviews", "ready", "insufficient_signal", "invalid"]));
  const lambdaCandidates = list(item.lambdaCandidates, 4).map((entry) => finite(entry));
  if (lambdaCandidates.join(",") !== "0,0.25,0.5,1") fail();
  const selectedLambda = finite(item.selectedLambda); const losses = object(item.validationLosses);
  if (Object.keys(losses).length !== 4 || ["0", "0.25", "0.5", "1"].some((key) => !(key in losses))) fail();
  const validationLosses = { "0": finite(losses["0"]), "0.25": finite(losses["0.25"]), "0.5": finite(losses["0.5"]), "1": finite(losses["1"]) };
  const readyPolicyHash = item.readyPolicyHash === null ? null : hash(item.readyPolicyHash);
  if ((status === "ready") !== (selectedLambda !== 0) || (status === "ready") !== (readyPolicyHash !== null) || !lambdaCandidates.includes(selectedLambda) || item.embeddingDimension !== 256 || item.normalizationEpsilon !== 1e-8 || item.fittingRecipe !== "l2-centroids+run-zscore+loo-balanced-log-loss-v1") fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_ADAPTATION_POLICY_SCHEMA, binding: adaptationBinding(item.binding), labelSetHash: hash(item.labelSetHash), status, selectedLambda,
    lambdaCandidates: [0, 0.25, 0.5, 1], validationLosses, eligibleLabelCount: integer(item.eligibleLabelCount, 8), positiveCount: integer(item.positiveCount, 4), negativeCount: integer(item.negativeCount, 4),
    embeddingDimension: 256, positiveCentroidHash: hash(item.positiveCentroidHash), negativeCentroidHash: hash(item.negativeCentroidHash), normalizationEpsilon: 1e-8,
    fittingRecipe: "l2-centroids+run-zscore+loo-balanced-log-loss-v1", readyPolicyHash, policyHash: hash(item.policyHash),
  });
}

export function parseTargetReviewPolicy(value: unknown): TargetReviewPolicy {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_TARGET_POLICY_SCHEMA
    || item.fittingRecipe !== "l2-centroids+run-zscore+loo-balanced-log-loss-v1"
    || item.baseOutputsImmutable !== true) fail();
  const adapted = list(item.adaptedOutputFields, 2);
  if (adapted.length !== 2 || adapted[0] !== "adaptedReviewPriority" || adapted[1] !== "adaptedRank") fail();
  const eligibleLabelCount = integer(item.eligibleLabelCount, 8, 256);
  const positiveCount = integer(item.positiveCount, 4, 256); const negativeCount = integer(item.negativeCount, 4, 256);
  const status = oneOf<TargetReviewPolicy["status"]>(item.status, new Set(["ready", "insufficient_signal"]));
  const selectedLambda = finite(item.selectedLambda);
  if (positiveCount + negativeCount !== eligibleLabelCount
    || ![0, 0.25, 0.5, 1].includes(selectedLambda)
    || (status === "ready") !== (selectedLambda !== 0)) fail();
  return deepFreeze({
    schemaVersion: GOVERNANCE_TARGET_POLICY_SCHEMA, binding: adaptationBinding(item.binding), labelSetHash: hash(item.labelSetHash),
    status, selectedLambda, eligibleLabelCount, positiveCount, negativeCount,
    fittingRecipe: "l2-centroids+run-zscore+loo-balanced-log-loss-v1", baseOutputsImmutable: true,
    adaptedOutputFields: ["adaptedReviewPriority", "adaptedRank"], policyHash: hash(item.policyHash),
  });
}

export function parseAdaptationComparison(value: unknown): AdaptationComparison {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_ADAPTATION_COMPARISON_SCHEMA) fail();
  const total = integer(item.total, 1, 10_000); const offset = integer(item.offset, 0); const limit = integer(item.limit, 1, 500);
  const rows = list(item.rows, 500).map((entry) => {
    const row = object(entry); const baseRank = integer(row.baseRank, 1); const adaptedRank = integer(row.adaptedRank, 1); const rankDelta = integer(row.rankDelta, -10_000, 10_000);
    if (rankDelta !== adaptedRank - baseRank) fail();
    return { nodeId: safeId(row.nodeId, 128), baseScore: finite(row.baseScore, 0, 1), baseRank, adaptedReviewPriority: finite(row.adaptedReviewPriority, 0, 1), adaptedRank, rankDelta };
  });
  if (offset + rows.length > total || rows.length > limit || new Set(rows.map((row) => row.nodeId)).size !== rows.length) fail();
  return deepFreeze({ schemaVersion: GOVERNANCE_ADAPTATION_COMPARISON_SCHEMA, binding: adaptationBinding(item.binding), policyHash: hash(item.policyHash), total, offset, limit, rows, comparisonHash: hash(item.comparisonHash), pageHash: hash(item.pageHash) });
}

function adaptationComparisonRow(value: unknown): AdaptationComparison["rows"][number] {
  const row = object(value); const baseRank = integer(row.baseRank, 1); const adaptedRank = integer(row.adaptedRank, 1); const rankDelta = integer(row.rankDelta, -10_000, 10_000);
  if (rankDelta !== adaptedRank - baseRank) fail();
  return { nodeId: safeId(row.nodeId, 128), baseScore: finite(row.baseScore, 0, 1), baseRank, adaptedReviewPriority: finite(row.adaptedReviewPriority, 0, 1), adaptedRank, rankDelta };
}

export function parseTargetAdaptationComparison(value: unknown): TargetAdaptationComparison {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_TARGET_COMPARISON_SCHEMA || item.baseOutputsImmutable !== true) fail();
  const rows = list(item.rows, 10_000).map(adaptationComparisonRow);
  const total = integer(item.total, 1, 10_000);
  const isPermutation = (ranks: readonly number[]) => [...ranks].sort((left, right) => left - right).every((rank, index) => rank === index + 1);
  if (rows.length !== total || new Set(rows.map((row) => row.nodeId)).size !== total
    || !isPermutation(rows.map((row) => row.baseRank)) || !isPermutation(rows.map((row) => row.adaptedRank))) fail();
  return deepFreeze({ schemaVersion: GOVERNANCE_TARGET_COMPARISON_SCHEMA, binding: adaptationBinding(item.binding), policyHash: hash(item.policyHash), total, baseOutputsImmutable: true, rows, comparisonHash: hash(item.comparisonHash) });
}

export function parseAdaptationGovernanceHandoff(value: unknown): AdaptationGovernanceHandoff {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_ADAPTATION_HANDOFF_SCHEMA || item.decision !== "pending_governance_review" || item.baseModelMutation !== false) fail();
  const logical = {
    schemaVersion: GOVERNANCE_ADAPTATION_HANDOFF_SCHEMA, targetTaskRegistrationId: pattern(item.targetTaskRegistrationId, TARGET_TASK_ID),
    targetReceiptHash: hash(item.targetReceiptHash), labelSetHash: hash(item.labelSetHash), binding: adaptationBinding(item.binding),
    policyHash: hash(item.policyHash), comparisonHash: hash(item.comparisonHash), decision: "pending_governance_review" as const, baseModelMutation: false as const,
  };
  const handoffHash = hash(item.handoffHash); if (handoffHash !== sha256Canonical(logical)) fail();
  return deepFreeze({ ...logical, handoffHash });
}

export function parseAdaptationOverlayActivation(value: unknown): AdaptationOverlayActivation {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_ADAPTATION_OVERLAY_SCHEMA || item.active !== true || item.baseModelMutation !== false) fail();
  const logical = {
    schemaVersion: GOVERNANCE_ADAPTATION_OVERLAY_SCHEMA, targetTaskRegistrationId: pattern(item.targetTaskRegistrationId, TARGET_TASK_ID),
    targetReceiptHash: hash(item.targetReceiptHash), labelSetHash: hash(item.labelSetHash), binding: adaptationBinding(item.binding),
    policyHash: hash(item.policyHash), comparisonHash: hash(item.comparisonHash), active: true as const, baseModelMutation: false as const,
  };
  const activationHash = hash(item.activationHash); if (activationHash !== sha256Canonical(logical)) fail();
  return deepFreeze({ ...logical, activationHash });
}

export function parseTargetReviewCollection(value: unknown): TargetReviewCollection {
  const item = object(value);
  if (item.schemaVersion !== "socialgraph-fm.governance-review-collection/1.0") fail();
  const logical = {
    schemaVersion: "socialgraph-fm.governance-review-collection/1.0" as const, idempotencyKey: safeId(item.idempotencyKey, 200),
    targetTaskRegistrationId: pattern(item.targetTaskRegistrationId, TARGET_TASK_ID), requestHash: hash(item.requestHash),
    resultHash: hash(item.resultHash),
    case: parseGovernanceCase(item.case),
  };
  const collectionHash = hash(item.collectionHash); if (collectionHash !== sha256Canonical(logical)) fail();
  return deepFreeze({ ...logical, collectionHash });
}
