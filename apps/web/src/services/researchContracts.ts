import type {
  ResearchCapabilities,
  ResearchFinding,
  ResearchRunBinding,
  ResearchRunRequest,
  ResearchRunResult,
  ResearchRunStatus,
  ResearchScenario,
  ResearchScenarioPreview,
  ResearchScenarios,
  ResearchSimilarNodesRequest,
  ResearchSimilarNodesResult,
  ResearchTargetScope,
  ResearchTaskId,
} from "../types/research";
import { RESEARCH_SCHEMA } from "../types/research";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";

const HASH = /^[0-9a-f]{64}$/u;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u;
const TASK_IDS = [
  "research.content_policy_review",
  "research.account_risk_review",
  "research.signed_relation_review",
  "core.collaboration_completion",
] as const satisfies readonly ResearchTaskId[];
const TASK_SET = new Set<ResearchTaskId>(TASK_IDS);
const SCENARIO_TASKS = new Map<ResearchScenario["scenarioId"], ResearchTaskId>([
  ["twitch-content-policy", "research.content_policy_review"],
  ["tolokers-account-risk", "research.account_risk_review"],
  ["wiki-rfa-signed-relation", "research.signed_relation_review"],
  ["email-eu-collaboration", "core.collaboration_completion"],
]);

function invalid(): never {
  throw new TypeError("invalid research GFM contract");
}

function safeParse<T>(code: string, parser: () => T): T {
  try {
    return parser();
  } catch {
    throw new Error(code);
  }
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid();
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) invalid();
  return value as Record<string, unknown>;
}

function exactKeys(
  value: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = [],
): void {
  const allowed = new Set([...required, ...optional]);
  if (required.some((key) => !Object.prototype.hasOwnProperty.call(value, key))) invalid();
  if (Object.keys(value).some((key) => !allowed.has(key))) invalid();
}

function schema(value: unknown): typeof RESEARCH_SCHEMA {
  if (value !== RESEARCH_SCHEMA) invalid();
  return RESEARCH_SCHEMA;
}

function stringValue(value: unknown, maximum = 500): string {
  if (typeof value !== "string" || value.length < 1 || value.length > maximum) invalid();
  return value;
}

function nullableString(value: unknown, maximum = 1_000): string | null {
  return value === null ? null : stringValue(value, maximum);
}

function hashValue(value: unknown): string {
  const result = stringValue(value, 64);
  if (!HASH.test(result)) invalid();
  return result;
}

function integer(value: unknown, minimum: number, maximum: number): number {
  if (!Number.isInteger(value) || (value as number) < minimum || (value as number) > maximum) invalid();
  return value as number;
}

function finiteNumber(value: unknown, minimum = -Number.MAX_VALUE, maximum = Number.MAX_VALUE): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) invalid();
  return value;
}

function booleanValue(value: unknown): boolean {
  if (typeof value !== "boolean") invalid();
  return value;
}

function dateTime(value: unknown): string {
  const result = stringValue(value, 100);
  if (!DATE_TIME.test(result) || !Number.isFinite(Date.parse(result))) invalid();
  return result;
}

function arrayValue(value: unknown, minimum = 0, maximum = 10_000): unknown[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) invalid();
  return value;
}

function stringArray(value: unknown, minimum = 0, maximum = 10_000): string[] {
  const result = arrayValue(value, minimum, maximum).map((item) => stringValue(item));
  if (new Set(result).size !== result.length) invalid();
  return result;
}

function literal<T extends string>(value: unknown, values: ReadonlySet<T>): T {
  if (typeof value !== "string" || !values.has(value as T)) invalid();
  return value as T;
}

function taskId(value: unknown): ResearchTaskId {
  return literal(value, TASK_SET);
}

function exactTaskInventory(value: unknown): ResearchTaskId[] {
  const result = arrayValue(value, TASK_IDS.length, TASK_IDS.length).map(taskId);
  if (result.some((item, index) => item !== TASK_IDS[index])) invalid();
  return result;
}

function validCanonicalHash(value: Record<string, unknown>, field: string): void {
  const payload = Object.fromEntries(Object.entries(value).filter(([key]) => key !== field));
  if (hashValue(value[field]) !== sha256Canonical(payload)) invalid();
}

function parseTargetScope(value: unknown): ResearchTargetScope {
  const candidate = record(value);
  if (candidate.kind === "nodes") {
    exactKeys(candidate, ["kind", "nodeIds"]);
    return { kind: "nodes", nodeIds: stringArray(candidate.nodeIds, 1) as [string, ...string[]] };
  }
  if (candidate.kind === "directed-node-pairs") {
    exactKeys(candidate, ["kind", "pairs"]);
    const pairs = arrayValue(candidate.pairs, 1, 10_000).map((entry): readonly [string, string] => {
      const pair = arrayValue(entry, 2, 2);
      const source = stringValue(pair[0]);
      const target = stringValue(pair[1]);
      if (source === target) invalid();
      return [source, target];
    });
    if (new Set(pairs.map(([source, target]) => `${source}\u0000${target}`)).size !== pairs.length) invalid();
    return { kind: "directed-node-pairs", pairs };
  }
  if (candidate.kind === "collaboration-candidates") {
    exactKeys(candidate, ["kind", "anchorNodeId", "topK"]);
    return {
      kind: "collaboration-candidates",
      anchorNodeId: stringValue(candidate.anchorNodeId),
      topK: integer(candidate.topK, 1, 100),
    };
  }
  invalid();
}

function assertTaskScope(task: ResearchTaskId, scope: ResearchTargetScope): void {
  const expected = task === "research.signed_relation_review"
    ? "directed-node-pairs"
    : task === "core.collaboration_completion"
      ? "collaboration-candidates"
      : "nodes";
  if (scope.kind !== expected) invalid();
}

function parseScenarioId(value: unknown): ResearchScenario["scenarioId"] {
  return literal(value, new Set(SCENARIO_TASKS.keys()));
}

export function parseResearchCapabilities(value: unknown): ResearchCapabilities {
  return safeParse("GFM_RESEARCH_CAPABILITIES_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "channel", "releaseLabel", "seed", "preliminary",
      "researchServingReady", "unavailableReason", "model", "taskIds", "upload", "capabilityHash",
    ]);
    schema(candidate.schemaVersion);
    if (candidate.channel !== "research" || candidate.releaseLabel !== "SocialGraph-FM Research") invalid();
    if (candidate.seed !== 1729 || candidate.preliminary !== true) invalid();
    const researchServingReady = booleanValue(candidate.researchServingReady);
    const unavailableReason = nullableString(candidate.unavailableReason);
    const tasks = exactTaskInventory(candidate.taskIds);
    let model: ResearchCapabilities["model"] = null;
    if (candidate.model !== null) {
      const item = record(candidate.model);
      exactKeys(item, [
        "modelVersionId", "modelVersionHash", "artifactHash", "taskIds", "graphSchemaVersion",
        "maxNodes", "maxEdges", "claimStatus",
      ]);
      model = {
        modelVersionId: stringValue(item.modelVersionId),
        modelVersionHash: hashValue(item.modelVersionHash),
        artifactHash: hashValue(item.artifactHash),
        taskIds: exactTaskInventory(item.taskIds),
        graphSchemaVersion: item.graphSchemaVersion === "socialgraph-fm.core-graph-bundle/2.0"
          ? item.graphSchemaVersion
          : invalid(),
        maxNodes: integer(item.maxNodes, 1, 50_000_000),
        maxEdges: integer(item.maxEdges, 0, 500_000_000),
        claimStatus: literal(item.claimStatus, new Set(["observed_transfer_gain", "not_demonstrated"] as const)),
      };
    }
    if (researchServingReady !== Boolean(model) || researchServingReady === Boolean(unavailableReason)) invalid();
    const upload = record(candidate.upload);
    exactKeys(upload, ["compatibleTaskIds", "auxiliaryCapabilities", "minNodes", "maxNodes", "maxEdges"]);
    const compatibleTaskIds = arrayValue(upload.compatibleTaskIds, 1, 1);
    const auxiliaryCapabilities = arrayValue(upload.auxiliaryCapabilities, 1, 1);
    if (
      compatibleTaskIds[0] !== "core.collaboration_completion"
      || auxiliaryCapabilities[0] !== "similar-nodes"
    ) invalid();
    if (upload.minNodes !== 5 || upload.maxNodes !== 50_000 || upload.maxEdges !== 1_500_000) invalid();
    validCanonicalHash(candidate, "capabilityHash");
    return deepFreeze({
      schemaVersion: RESEARCH_SCHEMA,
      channel: "research",
      releaseLabel: "SocialGraph-FM Research",
      seed: 1729,
      preliminary: true,
      researchServingReady,
      unavailableReason,
      model,
      taskIds: tasks,
      upload: {
        compatibleTaskIds: ["core.collaboration_completion"],
        auxiliaryCapabilities: ["similar-nodes"],
        minNodes: integer(upload.minNodes, 1, 50_000),
        maxNodes: integer(upload.maxNodes, 1, 50_000),
        maxEdges: integer(upload.maxEdges, 0, 1_500_000),
      },
      capabilityHash: hashValue(candidate.capabilityHash),
    });
  });
}

function parseScenario(value: unknown): ResearchScenario {
  const candidate = record(value);
  exactKeys(candidate, [
    "scenarioId", "datasetId", "title", "taskId", "graphVersionId", "graphVersionHash", "modelVersionId",
    "enabled", "unavailableReason", "defaultTargetScope", "primaryMetric", "scratchDelta",
  ]);
  const scenarioId = parseScenarioId(candidate.scenarioId);
  const task = taskId(candidate.taskId);
  if (SCENARIO_TASKS.get(scenarioId) !== task) invalid();
  const target = parseTargetScope(candidate.defaultTargetScope);
  assertTaskScope(task, target);
  const enabled = booleanValue(candidate.enabled);
  const unavailableReason = nullableString(candidate.unavailableReason);
  const graphVersionHash = candidate.graphVersionHash === null ? null : hashValue(candidate.graphVersionHash);
  const modelVersionId = nullableString(candidate.modelVersionId);
  if (enabled !== Boolean(modelVersionId && graphVersionHash) || enabled === Boolean(unavailableReason)) invalid();
  let primaryMetric: ResearchScenario["primaryMetric"] = null;
  if (candidate.primaryMetric !== null) {
    const metric = record(candidate.primaryMetric);
    exactKeys(metric, ["name", "value"]);
    primaryMetric = { name: stringValue(metric.name, 100), value: finiteNumber(metric.value) };
  }
  return {
    scenarioId,
    datasetId: stringValue(candidate.datasetId),
    title: stringValue(candidate.title),
    taskId: task,
    graphVersionId: stringValue(candidate.graphVersionId),
    graphVersionHash,
    modelVersionId,
    enabled,
    unavailableReason,
    defaultTargetScope: target,
    primaryMetric,
    scratchDelta: candidate.scratchDelta === null ? null : finiteNumber(candidate.scratchDelta),
  };
}

export function parseResearchScenarios(value: unknown): ResearchScenarios {
  return safeParse("GFM_RESEARCH_SCENARIOS_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, ["schemaVersion", "releaseLabel", "seed", "preliminary", "scenarios", "scenariosHash"]);
    schema(candidate.schemaVersion);
    if (candidate.releaseLabel !== "SocialGraph-FM Research" || candidate.seed !== 1729 || candidate.preliminary !== true) invalid();
    const scenarios = arrayValue(candidate.scenarios, SCENARIO_TASKS.size, SCENARIO_TASKS.size).map(parseScenario);
    if ([...SCENARIO_TASKS.keys()].some((id, index) => scenarios[index]?.scenarioId !== id)) invalid();
    validCanonicalHash(candidate, "scenariosHash");
    return deepFreeze({
      schemaVersion: RESEARCH_SCHEMA,
      releaseLabel: "SocialGraph-FM Research",
      seed: 1729,
      preliminary: true,
      scenarios,
      scenariosHash: hashValue(candidate.scenariosHash),
    });
  });
}

export function parseResearchScenarioPreview(value: unknown): ResearchScenarioPreview {
  return safeParse("GFM_RESEARCH_PREVIEW_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "scenarioId", "graphVersionId", "graphVersionHash", "modelVersionId",
      "modelVersionHash", "nodes", "edges", "partialPreview", "nodeCount", "edgeCount", "previewHash",
    ]);
    schema(candidate.schemaVersion);
    const nodes = arrayValue(candidate.nodes, 1, 800).map((value) => {
      const node = record(value);
      exactKeys(node, ["id", "label"]);
      return { id: stringValue(node.id, 300), label: stringValue(node.label, 500) };
    });
    const nodeIds = new Set(nodes.map((node) => node.id));
    if (nodeIds.size !== nodes.length) invalid();
    const edges = arrayValue(candidate.edges, 0, 2_500).map((value) => {
      const edge = record(value);
      exactKeys(edge, ["id", "source", "target", "directed"]);
      const parsed = {
        id: stringValue(edge.id, 700),
        source: stringValue(edge.source, 300),
        target: stringValue(edge.target, 300),
        directed: booleanValue(edge.directed),
      };
      if (!nodeIds.has(parsed.source) || !nodeIds.has(parsed.target)) invalid();
      return parsed;
    });
    if (new Set(edges.map((edge) => edge.id)).size !== edges.length) invalid();
    const nodeCount = integer(candidate.nodeCount, 1, 50_000);
    const edgeCount = integer(candidate.edgeCount, 0, 1_500_000);
    const partialPreview = booleanValue(candidate.partialPreview);
    if (partialPreview !== (nodes.length < nodeCount || edges.length < edgeCount)) invalid();
    validCanonicalHash(candidate, "previewHash");
    return deepFreeze({
      schemaVersion: RESEARCH_SCHEMA,
      scenarioId: parseScenarioId(candidate.scenarioId),
      graphVersionId: stringValue(candidate.graphVersionId, 200),
      graphVersionHash: hashValue(candidate.graphVersionHash),
      modelVersionId: stringValue(candidate.modelVersionId, 300),
      modelVersionHash: hashValue(candidate.modelVersionHash),
      nodes,
      edges,
      partialPreview,
      nodeCount,
      edgeCount,
      previewHash: hashValue(candidate.previewHash),
    });
  });
}

export function parseResearchRunRequest(value: unknown): ResearchRunRequest {
  return safeParse("GFM_RESEARCH_RUN_REQUEST_INVALID", () => {
    const candidate = record(value);
    exactKeys(
      candidate,
      ["schemaVersion", "graphVersionId", "taskId", "modelVersionId", "targetScope", "parameters"],
      ["scenarioId"],
    );
    schema(candidate.schemaVersion);
    const task = taskId(candidate.taskId);
    const target = parseTargetScope(candidate.targetScope);
    assertTaskScope(task, target);
    const parameters = record(candidate.parameters);
    exactKeys(parameters, ["candidateLimit"]);
    const scenarioId = candidate.scenarioId === undefined ? undefined : parseScenarioId(candidate.scenarioId);
    if (scenarioId && SCENARIO_TASKS.get(scenarioId) !== task) invalid();
    return deepFreeze({
      schemaVersion: RESEARCH_SCHEMA,
      graphVersionId: stringValue(candidate.graphVersionId),
      taskId: task,
      modelVersionId: stringValue(candidate.modelVersionId),
      targetScope: target,
      ...(scenarioId ? { scenarioId } : {}),
      parameters: { candidateLimit: integer(parameters.candidateLimit, 1, 1_000) },
    });
  });
}

function parseStatus(value: unknown): ResearchRunStatus {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "runId", "requestHash", "status", "progress", "createdAt", "updatedAt",
    "errorCode", "stateHash",
  ]);
  schema(candidate.schemaVersion);
  const status = {
    schemaVersion: RESEARCH_SCHEMA,
    runId: stringValue(candidate.runId, 100),
    requestHash: hashValue(candidate.requestHash),
    status: literal(candidate.status, new Set(["queued", "running", "succeeded", "failed"] as const)),
    progress: integer(candidate.progress, 0, 100),
    createdAt: dateTime(candidate.createdAt),
    updatedAt: dateTime(candidate.updatedAt),
    errorCode: nullableString(candidate.errorCode, 100),
    stateHash: hashValue(candidate.stateHash),
  };
  const expectedProgress = status.status === "queued" ? 0 : status.status === "running" ? 10 : 100;
  if (status.progress !== expectedProgress) invalid();
  validCanonicalHash(candidate, "stateHash");
  return status;
}

export function parseResearchRunStatus(
  value: unknown,
  binding?: ResearchRunBinding,
): ResearchRunStatus {
  return safeParse("GFM_RESEARCH_RUN_STATUS_INVALID", () => {
    const status = parseStatus(value);
    if (binding && (status.runId !== binding.runId || status.requestHash !== binding.serverRequestHash)) invalid();
    if ((status.status === "failed") !== Boolean(status.errorCode)) invalid();
    return deepFreeze(status);
  });
}

function parseFinding(value: unknown, task: ResearchTaskId): ResearchFinding {
  const candidate = record(value);
  exactKeys(candidate, [
    "id", "rank", "entityType", "entityIds", "score", "scoreKind", "calibrated",
    "reasonCodes", "limitations", "reviewRequired",
  ]);
  const entityType = literal(candidate.entityType, new Set(["node", "directed-edge", "node-pair"] as const));
  const expectedType = task === "research.signed_relation_review"
    ? "directed-edge"
    : task === "core.collaboration_completion"
      ? "node-pair"
      : "node";
  if (entityType !== expectedType || candidate.reviewRequired !== true) invalid();
  const entityIds = stringArray(candidate.entityIds, entityType === "node" ? 1 : 2, entityType === "node" ? 1 : 2);
  const scoreKind = literal(candidate.scoreKind, new Set(["probability", "ranking-score"] as const));
  const score = finiteNumber(candidate.score, 0, 1);
  const calibrated = booleanValue(candidate.calibrated);
  if (calibrated !== (scoreKind === "probability")) invalid();
  return {
    id: stringValue(candidate.id),
    rank: integer(candidate.rank, 1, 100_000),
    entityType,
    entityIds,
    score,
    scoreKind,
    calibrated,
    reasonCodes: stringArray(candidate.reasonCodes, 0, 100),
    limitations: stringArray(candidate.limitations, 1, 20),
    reviewRequired: true,
  };
}

export function parseResearchRunResult(
  value: unknown,
  binding?: ResearchRunBinding,
): ResearchRunResult {
  return safeParse("GFM_RESEARCH_RUN_RESULT_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "runId", "requestHash", "taskId", "graphVersionId", "graphVersionHash",
      "modelVersionId", "modelVersionHash", "seed", "preliminary", "calibrationStatus", "findings",
      "completedAt", "resultHash",
    ]);
    schema(candidate.schemaVersion);
    const task = taskId(candidate.taskId);
    if (candidate.seed !== 1729 || candidate.preliminary !== true) invalid();
    const calibrationStatus = literal(candidate.calibrationStatus, new Set(["calibrated", "ranking_only"] as const));
    const findings = arrayValue(candidate.findings, 0, 100_000).map((item) => parseFinding(item, task));
    if (
      new Set(findings.map((item) => item.id)).size !== findings.length
      || findings.some((item, index) => item.rank !== index + 1)
      || (findings.length === 0 && calibrationStatus !== "ranking_only")
      || (calibrationStatus === "calibrated" && !findings.every((item) => item.calibrated))
      || (calibrationStatus === "ranking_only" && findings.some((item) => item.calibrated))
    ) invalid();
    const result: ResearchRunResult = {
      schemaVersion: RESEARCH_SCHEMA,
      runId: stringValue(candidate.runId, 100),
      requestHash: hashValue(candidate.requestHash),
      taskId: task,
      graphVersionId: stringValue(candidate.graphVersionId),
      graphVersionHash: hashValue(candidate.graphVersionHash),
      modelVersionId: stringValue(candidate.modelVersionId),
      modelVersionHash: hashValue(candidate.modelVersionHash),
      seed: 1729,
      preliminary: true,
      calibrationStatus,
      findings,
      completedAt: dateTime(candidate.completedAt),
      resultHash: hashValue(candidate.resultHash),
    };
    if (binding && (
      result.runId !== binding.runId
      || result.requestHash !== binding.serverRequestHash
      || result.taskId !== binding.taskId
      || result.graphVersionId !== binding.graphVersionId
      || result.modelVersionId !== binding.modelVersionId
    )) invalid();
    validCanonicalHash(candidate, "resultHash");
    return deepFreeze(result);
  });
}

export function parseResearchSimilarNodesRequest(value: unknown): ResearchSimilarNodesRequest {
  return safeParse("GFM_RESEARCH_SIMILAR_REQUEST_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, ["schemaVersion", "graphVersionId", "nodeId", "topK", "modelVersionId"]);
    schema(candidate.schemaVersion);
    return deepFreeze({
      schemaVersion: RESEARCH_SCHEMA,
      graphVersionId: stringValue(candidate.graphVersionId),
      nodeId: stringValue(candidate.nodeId),
      topK: integer(candidate.topK, 1, 50),
      modelVersionId: stringValue(candidate.modelVersionId),
    });
  });
}

export function parseResearchSimilarNodesResult(value: unknown): ResearchSimilarNodesResult {
  return safeParse("GFM_RESEARCH_SIMILAR_RESULT_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "graphVersionId", "nodeId", "modelVersionId", "modelVersionHash", "matches", "resultHash",
    ]);
    schema(candidate.schemaVersion);
    const matches = arrayValue(candidate.matches, 0, 50).map((value) => {
      const match = record(value);
      exactKeys(match, ["graphVersionId", "nodeId", "datasetId", "similarity", "structuralFacts"]);
      const facts = record(match.structuralFacts);
      exactKeys(facts, ["degree", "inDegree", "outDegree", "pagerank", "clustering", "coreNumber"]);
      return {
        graphVersionId: stringValue(match.graphVersionId),
        nodeId: stringValue(match.nodeId),
        datasetId: nullableString(match.datasetId),
        similarity: finiteNumber(match.similarity, -1, 1),
        structuralFacts: {
          degree: integer(facts.degree, 0, Number.MAX_SAFE_INTEGER),
          inDegree: integer(facts.inDegree, 0, Number.MAX_SAFE_INTEGER),
          outDegree: integer(facts.outDegree, 0, Number.MAX_SAFE_INTEGER),
          pagerank: finiteNumber(facts.pagerank, 0),
          clustering: finiteNumber(facts.clustering, 0, 1),
          coreNumber: integer(facts.coreNumber, 0, Number.MAX_SAFE_INTEGER),
        },
      };
    });
    validCanonicalHash(candidate, "resultHash");
    return deepFreeze({
      schemaVersion: RESEARCH_SCHEMA,
      graphVersionId: stringValue(candidate.graphVersionId),
      nodeId: stringValue(candidate.nodeId),
      modelVersionId: stringValue(candidate.modelVersionId),
      modelVersionHash: hashValue(candidate.modelVersionHash),
      matches,
      resultHash: hashValue(candidate.resultHash),
    });
  });
}

export const RESEARCH_TASK_IDS = TASK_IDS;
