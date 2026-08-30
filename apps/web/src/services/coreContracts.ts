import type {
  CoreCalibratedConfidence,
  CoreCapabilities,
  CoreConfidenceEvidence,
  CoreCollaborationParameters,
  CoreCollaborationTargetScope,
  CoreCommunityParameters,
  CoreCommunityTargetScope,
  CoreEntityType,
  CoreEvidenceItem,
  CoreFindingType,
  CoreModelCapability,
  CoreModelScore,
  CoreRegisteredEdgeIdentity,
  CoreRiskParameters,
  CoreRiskTargetScope,
  CoreRegressionConfidenceInterval,
  CoreRunBinding,
  CoreRunParameters,
  CoreRunRequest,
  CoreRunResult,
  CoreRunStatus,
  CoreSimilarCase,
  CoreTargetScope,
  CoreTaskId,
  CoreTaskEntityCapability,
  CoreFinding,
} from "../types/core";
import { canonicalJson, sha256Canonical } from "./graphIdentity";

const HASH = /^[0-9a-f]{64}$/u;
const ERROR_CODE = /^[A-Z0-9_]{1,100}$/u;
const PUBLIC_GFM_ERROR_CODES: ReadonlySet<string> = new Set([
  "GFM_CORE_CAPABILITIES_INVALID",
  "GFM_CORE_COMPATIBILITY_REJECTED",
  "GFM_CORE_CREATE_RECEIPT_PERSIST_FAILED",
  "GFM_CORE_GRAPH_VERSION_NOT_FOUND",
  "GFM_CORE_JSON_REQUIRED",
  "GFM_CORE_MODEL_GRAPH_INCOMPATIBLE",
  "GFM_CORE_MODEL_NOT_INSTALLED",
  "GFM_CORE_MODEL_UNAVAILABLE",
  "GFM_CORE_NOT_FOUND",
  "GFM_CORE_REDIRECT_REJECTED",
  "GFM_CORE_REGISTRY_INVALID",
  "GFM_CORE_REQUEST_INVALID",
  "GFM_CORE_REQUEST_SIZE_INVALID",
  "GFM_CORE_RESPONSE_INVALID",
  "GFM_CORE_RESPONSE_TOO_LARGE",
  "GFM_CORE_RESULT_BINDING_INVALID",
  "GFM_CORE_RESULT_NOT_READY",
  "GFM_CORE_RUN_BINDING_INVALID",
  "GFM_CORE_RUN_NOT_FOUND",
  "GFM_CORE_SERVICE_ERROR",
  "GFM_CORE_SERVICE_UNAVAILABLE",
  "GFM_CORE_SERVING_CONTROL_INVALID",
  "GFM_CORE_SERVING_CONTROL_STALE",
  "GFM_CORE_SESSION_TOKEN_INVALID",
]);
const RUN_FAILURE_CODES = new Set([
  "GFM_CORE_EXECUTION_FAILED",
  "GFM_CORE_RUN_INTERRUPTED",
] as const);

export function isPublicCoreErrorCode(value: unknown): value is string {
  return typeof value === "string" && PUBLIC_GFM_ERROR_CODES.has(value);
}
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u;
const TASK_IDS = new Set<CoreTaskId>([
  "core.community_resilience_review",
  "core.risk_and_trust_review",
  "core.collaboration_completion",
]);
const ENTITY_TYPES = new Set<CoreEntityType>(["node", "edge", "node-pair", "community"]);
const MANUAL_REVIEW = "Manual human review is required; no automatic sanction or action is authorized.";
const NON_CAUSAL = "This finding is non-causal and does not predict future events.";
const REGRESSION_INTERVAL = "The resilience interval reports validation residual coverage, not a probability.";
const ALLOWED_LIMITATIONS = new Set([
  MANUAL_REVIEW,
  NON_CAUSAL,
  "Directed edges are analyzed on a weak undirected projection.",
  "Registered topology only; edge direction over time is not represented.",
  "The score is a registered model output, not a graph fact or decision.",
  "Support/opposition semantics require contextual human review.",
  "Candidate for review; it is not a risk or trust truth label.",
  "Common-neighbor evidence describes only the registered graph.",
  "Path evidence is registered relation-completion context, not a future-event forecast.",
  "Core relation-completion recommendation only.",
  "Directed structural context uses a weak undirected projection.",
  "Connectivity evidence is factual topology context, not a community health label.",
  REGRESSION_INTERVAL,
]);

function invalid(): never {
  throw new TypeError("invalid contract value");
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
  if (Object.keys(value).some((key) => !allowed.has(key))) invalid();
  if (required.some((key) => !Object.prototype.hasOwnProperty.call(value, key))) invalid();
}

function literal<T extends string>(value: unknown, allowed: ReadonlySet<T>): T {
  if (typeof value !== "string" || !allowed.has(value as T)) invalid();
  return value as T;
}

function stringValue(value: unknown, minimum = 1, maximum = 1_000): string {
  if (typeof value !== "string" || value.length < minimum || value.length > maximum) invalid();
  return value;
}

function hashValue(value: unknown): string {
  const candidate = stringValue(value, 64, 64);
  if (!HASH.test(candidate)) invalid();
  return candidate;
}

function integer(value: unknown, minimum: number, maximum = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isInteger(value) || (value as number) < minimum || (value as number) > maximum) invalid();
  return value as number;
}

function finiteNumber(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) invalid();
  return value;
}

function booleanValue(value: unknown): boolean {
  if (typeof value !== "boolean") invalid();
  return value;
}

function dateTime(value: unknown): string {
  const candidate = stringValue(value, 1, 100);
  if (!DATE_TIME.test(candidate) || !Number.isFinite(Date.parse(candidate))) invalid();
  return candidate;
}

function arrayValue(value: unknown, minimum = 0, maximum = 10_000): unknown[] {
  if (!Array.isArray(value) || value.length < minimum || value.length > maximum) invalid();
  return value;
}

function stringArray(value: unknown, minimum = 0, maximum = 10_000, itemMaximum = 500): string[] {
  const result = arrayValue(value, minimum, maximum).map((item) => stringValue(item, 1, itemMaximum));
  if (new Set(result).size !== result.length) invalid();
  return result;
}

function nullableString(value: unknown, maximum = 300): string | null {
  return value === null ? null : stringValue(value, 1, maximum);
}

function nullableHash(value: unknown): string | null {
  return value === null ? null : hashValue(value);
}

function nullableTask(value: unknown): CoreTaskId | null {
  return value === null ? null : literal(value, TASK_IDS);
}

function nullableEntity(value: unknown): CoreEntityType | null {
  return value === null ? null : literal(value, ENTITY_TYPES);
}

function sameArray(left: readonly string[], right: readonly string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function omit(value: Record<string, unknown>, key: string): Record<string, unknown> {
  return Object.fromEntries(Object.entries(value).filter(([candidate]) => candidate !== key));
}

function validHash(value: Record<string, unknown>, field: string): void {
  if (hashValue(value[field]) !== sha256Canonical(omit(value, field))) invalid();
}

export function deepFreeze<T>(value: T): T {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  return Object.freeze(value);
}

function safeParse<T>(code: string, parser: () => T): T {
  try {
    return parser();
  } catch {
    throw new Error(code);
  }
}

function task(value: unknown): CoreTaskId {
  return literal(value, TASK_IDS);
}

function parseTarget(value: unknown): CoreTargetScope {
  const candidate = record(value);
  const kind = stringValue(candidate.kind, 1, 100);
  if (kind === "community") {
    exactKeys(candidate, ["kind", "communityIds"]);
    return { kind, communityIds: stringArray(candidate.communityIds, 1) } satisfies CoreCommunityTargetScope;
  }
  if (kind === "risk-review") {
    exactKeys(candidate, ["kind", "nodeIds", "edgeIds"]);
    const nodeIds = stringArray(candidate.nodeIds);
    const edgeIds = stringArray(candidate.edgeIds);
    if ((nodeIds.length > 0) === (edgeIds.length > 0)) invalid();
    if (nodeIds.length > 0) {
      return {
        kind,
        nodeIds: nodeIds as [string, ...string[]],
        edgeIds: [],
      } satisfies CoreRiskTargetScope;
    }
    return {
      kind,
      nodeIds: [],
      edgeIds: edgeIds as [string, ...string[]],
    } satisfies CoreRiskTargetScope;
  }
  if (kind === "node-pairs") {
    exactKeys(candidate, ["kind", "pairs"]);
    const pairs = arrayValue(candidate.pairs, 1).map((item): readonly [string, string] => {
      const pair = arrayValue(item, 2, 2);
      const source = stringValue(pair[0], 1, 500);
      const target = stringValue(pair[1], 1, 500);
      if (source === target) invalid();
      return [source, target];
    });
    if (new Set(pairs.map((pair) => canonicalJson(pair))).size !== pairs.length) invalid();
    return { kind, pairs } satisfies CoreCollaborationTargetScope;
  }
  invalid();
}

function parseParameters(value: unknown): CoreRunParameters {
  const candidate = record(value);
  const kind = stringValue(candidate.kind, 1, 100);
  if (kind === "community-resilience") {
    exactKeys(candidate, ["kind", "topKSimilarCases"]);
    return { kind, topKSimilarCases: integer(candidate.topKSimilarCases, 0, 20) } satisfies CoreCommunityParameters;
  }
  if (kind === "risk-and-trust") {
    exactKeys(candidate, ["kind", "topKSimilarCases"]);
    return { kind, topKSimilarCases: integer(candidate.topKSimilarCases, 0, 20) } satisfies CoreRiskParameters;
  }
  if (kind === "collaboration-completion") {
    exactKeys(candidate, ["kind", "topKSimilarCases", "candidateLimit"]);
    return {
      kind,
      topKSimilarCases: integer(candidate.topKSimilarCases, 0, 20),
      candidateLimit: integer(candidate.candidateLimit, 1, 10_000),
    } satisfies CoreCollaborationParameters;
  }
  invalid();
}

export function parseCoreRunRequest(value: unknown): CoreRunRequest {
  return safeParse("GFM_CORE_RUN_REQUEST_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, ["schemaVersion", "graphVersionId", "taskId", "targetScope", "modelVersionId", "parameters"]);
    if (candidate.schemaVersion !== "socialgraph-fm.core-run-request/2.0") invalid();
    const taskId = task(candidate.taskId);
    const targetScope = parseTarget(candidate.targetScope);
    const parameters = parseParameters(candidate.parameters);
    const expected: Record<CoreTaskId, readonly [CoreTargetScope["kind"], CoreRunParameters["kind"]]> = {
      "core.community_resilience_review": ["community", "community-resilience"],
      "core.risk_and_trust_review": ["risk-review", "risk-and-trust"],
      "core.collaboration_completion": ["node-pairs", "collaboration-completion"],
    };
    if (targetScope.kind !== expected[taskId][0] || parameters.kind !== expected[taskId][1]) invalid();
    return deepFreeze({
      schemaVersion: candidate.schemaVersion,
      graphVersionId: stringValue(candidate.graphVersionId, 1, 200),
      taskId,
      targetScope,
      modelVersionId: stringValue(candidate.modelVersionId, 1, 300),
      parameters,
    } as CoreRunRequest);
  });
}

const TASK_ENTITY_INVENTORY: readonly (readonly [CoreTaskId, CoreEntityType])[] = [
  ["core.community_resilience_review", "community"],
  ["core.risk_and_trust_review", "node"],
  ["core.risk_and_trust_review", "edge"],
  ["core.collaboration_completion", "node-pair"],
];

function parseTaskBinding(value: unknown): CoreTaskEntityCapability {
  const candidate = record(value);
  exactKeys(candidate, [
    "taskId", "entityType", "confidenceKind", "calibrationVersion", "method",
    "calibrationArtifactHash", "calibrationProtocolHash", "adapterDomain",
    "adapterSchemaHash", "adapterStateHash", "featureContractHash",
  ]);
  const taskId = task(candidate.taskId);
  const entityType = literal(candidate.entityType, ENTITY_TYPES);
  const confidenceKind = literal(
    candidate.confidenceKind,
    new Set(["binary-calibration", "regression-interval"] as const),
  );
  const method = literal(
    candidate.method,
    new Set(["sigmoid", "validation-residual-interval"] as const),
  );
  const expectedEntityTypes: Readonly<Record<CoreTaskId, ReadonlySet<CoreEntityType>>> = {
    "core.community_resilience_review": new Set(["community"]),
    "core.risk_and_trust_review": new Set(["node", "edge"]),
    "core.collaboration_completion": new Set(["node-pair"]),
  };
  if (!expectedEntityTypes[taskId].has(entityType)) invalid();
  const expectedConfidence = entityType === "community"
    ? ["regression-interval", "validation-residual-interval"]
    : ["binary-calibration", "sigmoid"];
  if (confidenceKind !== expectedConfidence[0] || method !== expectedConfidence[1]) invalid();
  return {
    taskId,
    entityType,
    confidenceKind,
    calibrationVersion: stringValue(candidate.calibrationVersion, 1, 300),
    method,
    calibrationArtifactHash: hashValue(candidate.calibrationArtifactHash),
    calibrationProtocolHash: hashValue(candidate.calibrationProtocolHash),
    adapterDomain: stringValue(candidate.adapterDomain, 1, 200),
    adapterSchemaHash: hashValue(candidate.adapterSchemaHash),
    adapterStateHash: hashValue(candidate.adapterStateHash),
    featureContractHash: hashValue(candidate.featureContractHash),
  };
}

function parseModel(value: unknown): CoreModelCapability {
  const candidate = record(value);
  exactKeys(candidate, [
    "modelVersionId", "modelVersionHash", "state", "tasks", "graphSchemaVersions",
    "graphFeatureContractHash", "taskBindings", "maxNodes", "maxEdges",
  ]);
  const tasks = arrayValue(candidate.tasks, 0, 3).map(task);
  if (new Set(tasks).size !== tasks.length) invalid();
  const taskBindings = arrayValue(candidate.taskBindings, 1, 4).map(parseTaskBinding);
  const observedInventory = taskBindings.map((binding) => [binding.taskId, binding.entityType]);
  const expectedInventory = TASK_ENTITY_INVENTORY.filter(([taskId]) => tasks.includes(taskId));
  if (canonicalJson(observedInventory) !== canonicalJson(expectedInventory)) invalid();
  const featureInventory = taskBindings.map((binding) => ({
    taskId: binding.taskId,
    entityType: binding.entityType,
    featureContractHash: binding.featureContractHash,
  }));
  const graphFeatureContractHash = hashValue(candidate.graphFeatureContractHash);
  if (graphFeatureContractHash !== sha256Canonical(featureInventory)) invalid();
  return {
    modelVersionId: stringValue(candidate.modelVersionId, 1, 300),
    modelVersionHash: hashValue(candidate.modelVersionHash),
    state: literal(candidate.state, new Set(["accepted", "servingReady"] as const)),
    tasks,
    graphSchemaVersions: stringArray(candidate.graphSchemaVersions, 1, 100, 200),
    graphFeatureContractHash,
    taskBindings,
    maxNodes: integer(candidate.maxNodes, 1),
    maxEdges: integer(candidate.maxEdges, 1),
  };
}

export function parseCoreCapabilities(value: unknown): CoreCapabilities {
  return safeParse("GFM_CORE_CAPABILITIES_INVALID", () => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "registryHash", "registryGeneration", "servingReady", "models", "tasks", "readiness",
    ], ["controlHash", "controlGeneration", "catalogHash", "catalogGeneration"]);
    if (candidate.schemaVersion !== "socialgraph-fm.core-capabilities/2.0") invalid();
    const models = arrayValue(candidate.models, 0, 1_000).map(parseModel);
    const ids = models.map((model) => model.modelVersionId);
    if (new Set(ids).size !== ids.length) invalid();
    const tasks = arrayValue(candidate.tasks, 0, 3).map(task);
    if (new Set(tasks).size !== tasks.length) invalid();
    const readiness = record(candidate.readiness);
    exactKeys(readiness, ["modelValidated", "coreServingReady"]);
    const modelValidated = booleanValue(readiness.modelValidated);
    const coreServingReady = booleanValue(readiness.coreServingReady);
    const servingReady = booleanValue(candidate.servingReady);
    const hasServing = models.some((model) => model.state === "servingReady");
    const derivedTasks = new Set(models.flatMap((model) => model.tasks));
    if (
      servingReady !== hasServing
      || coreServingReady !== hasServing
      || modelValidated !== (models.length > 0)
      || tasks.length !== derivedTasks.size
      || tasks.some((item) => !derivedTasks.has(item))
    ) invalid();

    const optionalPair = (hashField: string, generationField: string): readonly [string | null | undefined, number | null | undefined] => {
      const hashPresent = Object.prototype.hasOwnProperty.call(candidate, hashField);
      const generationPresent = Object.prototype.hasOwnProperty.call(candidate, generationField);
      if (hashPresent !== generationPresent) invalid();
      if (!hashPresent) return [undefined, undefined];
      const hash = candidate[hashField] === null ? null : hashValue(candidate[hashField]);
      const generation = candidate[generationField] === null ? null : integer(candidate[generationField], 0);
      if ((hash === null) !== (generation === null)) invalid();
      return [hash, generation];
    };
    const [controlHash, controlGeneration] = optionalPair("controlHash", "controlGeneration");
    const [catalogHash, catalogGeneration] = optionalPair("catalogHash", "catalogGeneration");
    return deepFreeze({
      schemaVersion: candidate.schemaVersion,
      registryHash: hashValue(candidate.registryHash),
      registryGeneration: integer(candidate.registryGeneration, 0),
      ...(controlHash !== undefined ? { controlHash, controlGeneration } : {}),
      ...(catalogHash !== undefined ? { catalogHash, catalogGeneration } : {}),
      servingReady,
      models,
      tasks,
      readiness: { modelValidated, coreServingReady },
    } as CoreCapabilities);
  });
}

function parseStatus(value: unknown): CoreRunStatus {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "runId", "requestHash", "status", "progress", "createdAt", "updatedAt", "errorCode", "stateHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-run-status/2.0") invalid();
  const status = literal(candidate.status, new Set(["queued", "running", "succeeded", "failed"] as const));
  const expectedProgress = { queued: 0, running: 10, succeeded: 100, failed: 100 } as const;
  const progress = integer(candidate.progress, 0, 100);
  const errorCode = candidate.errorCode === null ? null : stringValue(candidate.errorCode, 1, 100);
  if (
    progress !== expectedProgress[status]
    || ((status === "failed") !== (errorCode !== null))
    || (errorCode !== null && !RUN_FAILURE_CODES.has(errorCode as "GFM_CORE_EXECUTION_FAILED" | "GFM_CORE_RUN_INTERRUPTED"))
  ) invalid();
  validHash(candidate, "stateHash");
  return deepFreeze({
    schemaVersion: candidate.schemaVersion,
    runId: stringValue(candidate.runId, 1, 100),
    requestHash: hashValue(candidate.requestHash),
    status,
    progress,
    createdAt: dateTime(candidate.createdAt),
    updatedAt: dateTime(candidate.updatedAt),
    errorCode,
    stateHash: hashValue(candidate.stateHash),
  } as CoreRunStatus);
}

export function parseCoreRunStatus(value: unknown, binding?: CoreRunBinding): CoreRunStatus {
  const parsed = safeParse("GFM_CORE_RUN_STATUS_INVALID", () => parseStatus(value));
  if (binding && (parsed.runId !== binding.runId || parsed.requestHash !== binding.serverRequestHash)) {
    throw new Error("GFM_CORE_RESPONSE_BINDING_INVALID");
  }
  return parsed;
}

function limitations(value: unknown, minimum = 0): string[] {
  const result = stringArray(value, minimum, 100, 500);
  if (result.some((item) => !ALLOWED_LIMITATIONS.has(item))) invalid();
  return result;
}

function parseEdgeIdentity(value: unknown): CoreRegisteredEdgeIdentity {
  const candidate = record(value);
  exactKeys(candidate, ["schemaVersion", "sourceId", "targetId", "edgeType", "weight", "edgeHash"]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-edge-identity/2.0") invalid();
  validHash(candidate, "edgeHash");
  return {
    schemaVersion: candidate.schemaVersion,
    sourceId: stringValue(candidate.sourceId, 1, 500),
    targetId: stringValue(candidate.targetId, 1, 500),
    edgeType: stringValue(candidate.edgeType, 1, 200),
    weight: finiteNumber(candidate.weight),
    edgeHash: hashValue(candidate.edgeHash),
  };
}

function parseScore(value: unknown): CoreModelScore {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "taskId", "entityType", "entityIds", "score", "graphVersionHash", "modelVersion",
    "modelVersionHash", "edgeIdentity", "scoreHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-model-score/2.0") invalid();
  validHash(candidate, "scoreHash");
  const entityType = literal(candidate.entityType, ENTITY_TYPES);
  const entityIds = stringArray(candidate.entityIds, 1, 10_000, 500);
  const edgeIdentity = candidate.edgeIdentity === null ? null : parseEdgeIdentity(candidate.edgeIdentity);
  if (entityType !== "edge" && edgeIdentity !== null) invalid();
  if (entityType === "edge" && edgeIdentity && !sameArray(entityIds, [edgeIdentity.sourceId, edgeIdentity.targetId])) invalid();
  return {
    schemaVersion: candidate.schemaVersion,
    taskId: task(candidate.taskId),
    entityType,
    entityIds,
    score: finiteNumber(candidate.score),
    graphVersionHash: hashValue(candidate.graphVersionHash),
    modelVersion: stringValue(candidate.modelVersion, 1, 300),
    modelVersionHash: hashValue(candidate.modelVersionHash),
    edgeIdentity,
    scoreHash: hashValue(candidate.scoreHash),
  };
}

function parseCalibratedConfidence(candidate: Record<string, unknown>): CoreCalibratedConfidence {
  exactKeys(candidate, [
    "schemaVersion", "value", "scoreHash", "taskId", "entityType", "entityIds", "graphVersionHash", "modelVersion",
    "modelVersionHash", "calibrationVersion", "method", "calibrationArtifactHash", "calibrationProtocolHash", "confidenceHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-calibrated-confidence/2.0") invalid();
  validHash(candidate, "confidenceHash");
  const confidence = finiteNumber(candidate.value);
  if (confidence < 0 || confidence > 1) invalid();
  return {
    schemaVersion: candidate.schemaVersion,
    value: confidence,
    scoreHash: hashValue(candidate.scoreHash),
    taskId: task(candidate.taskId),
    entityType: literal(candidate.entityType, ENTITY_TYPES),
    entityIds: stringArray(candidate.entityIds, 1, 10_000, 500),
    graphVersionHash: hashValue(candidate.graphVersionHash),
    modelVersion: stringValue(candidate.modelVersion, 1, 300),
    modelVersionHash: hashValue(candidate.modelVersionHash),
    calibrationVersion: stringValue(candidate.calibrationVersion, 1, 300),
    method: stringValue(candidate.method, 1, 200),
    calibrationArtifactHash: hashValue(candidate.calibrationArtifactHash),
    calibrationProtocolHash: hashValue(candidate.calibrationProtocolHash),
    confidenceHash: hashValue(candidate.confidenceHash),
  };
}

function parseRegressionConfidence(candidate: Record<string, unknown>): CoreRegressionConfidenceInterval {
  exactKeys(candidate, [
    "schemaVersion", "pointEstimate", "lowerBound", "upperBound", "coverage", "validationCount",
    "scoreHash", "taskId", "entityType", "entityIds", "graphVersionHash", "modelVersion",
    "modelVersionHash", "confidenceVersion", "method", "confidenceArtifactHash",
    "confidenceProtocolHash", "confidenceHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-regression-confidence-interval/1.0") invalid();
  validHash(candidate, "confidenceHash");
  const pointEstimate = finiteNumber(candidate.pointEstimate);
  const lowerBound = finiteNumber(candidate.lowerBound);
  const upperBound = finiteNumber(candidate.upperBound);
  const coverage = finiteNumber(candidate.coverage);
  if (lowerBound > pointEstimate || pointEstimate > upperBound || coverage <= 0 || coverage >= 1) invalid();
  if (candidate.taskId !== "core.community_resilience_review" || candidate.entityType !== "community") invalid();
  if (candidate.method !== "validation-residual-interval") invalid();
  return {
    schemaVersion: candidate.schemaVersion,
    pointEstimate,
    lowerBound,
    upperBound,
    coverage,
    validationCount: integer(candidate.validationCount, 2),
    scoreHash: hashValue(candidate.scoreHash),
    taskId: candidate.taskId,
    entityType: candidate.entityType,
    entityIds: stringArray(candidate.entityIds, 1, 10_000, 500),
    graphVersionHash: hashValue(candidate.graphVersionHash),
    modelVersion: stringValue(candidate.modelVersion, 1, 300),
    modelVersionHash: hashValue(candidate.modelVersionHash),
    confidenceVersion: stringValue(candidate.confidenceVersion, 1, 300),
    method: candidate.method,
    confidenceArtifactHash: hashValue(candidate.confidenceArtifactHash),
    confidenceProtocolHash: hashValue(candidate.confidenceProtocolHash),
    confidenceHash: hashValue(candidate.confidenceHash),
  };
}

function parseConfidence(value: unknown): CoreConfidenceEvidence {
  const candidate = record(value);
  if (candidate.schemaVersion === "socialgraph-fm.core-calibrated-confidence/2.0") {
    return parseCalibratedConfidence(candidate);
  }
  if (candidate.schemaVersion === "socialgraph-fm.core-regression-confidence-interval/1.0") {
    return parseRegressionConfidence(candidate);
  }
  invalid();
}

function validateEvidenceJson(value: unknown): void {
  const source = stringValue(value, 2, 100_000);
  const parsed: unknown = JSON.parse(source);
  const candidate = record(parsed);
  if (canonicalJson(candidate) !== source) invalid();
}

function parseEvidence(value: unknown): CoreEvidenceItem {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "metric", "valueCanonicalJson", "graphVersionHash", "sourceType", "nodeIds", "edgeIds",
    "algorithmConfigHash", "modelVersionHash", "modelVersion", "modelScoreHash", "modelTaskId", "modelEntityType",
    "modelEntityIds", "limitations", "evidenceHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-evidence/2.0") invalid();
  validHash(candidate, "evidenceHash");
  const sourceType = literal(candidate.sourceType, new Set(["deterministic-graph-algorithm", "registered-model-output"] as const));
  const algorithmConfigHash = nullableHash(candidate.algorithmConfigHash);
  const modelVersionHash = nullableHash(candidate.modelVersionHash);
  const modelVersion = nullableString(candidate.modelVersion, 300);
  const modelScoreHash = nullableHash(candidate.modelScoreHash);
  const modelTaskId = nullableTask(candidate.modelTaskId);
  const modelEntityType = nullableEntity(candidate.modelEntityType);
  const modelEntityIds = candidate.modelEntityIds === null ? null : stringArray(candidate.modelEntityIds, 1, 10_000, 500);
  const modelBindings = [modelVersionHash, modelVersion, modelScoreHash, modelTaskId, modelEntityType, modelEntityIds];
  if (sourceType === "deterministic-graph-algorithm") {
    if (algorithmConfigHash === null || modelBindings.some((item) => item !== null)) invalid();
  } else if (algorithmConfigHash !== null || modelBindings.some((item) => item === null)) invalid();
  validateEvidenceJson(candidate.valueCanonicalJson);
  return deepFreeze({
    schemaVersion: candidate.schemaVersion,
    metric: stringValue(candidate.metric, 1, 300),
    valueCanonicalJson: stringValue(candidate.valueCanonicalJson, 2, 100_000),
    graphVersionHash: hashValue(candidate.graphVersionHash),
    sourceType,
    nodeIds: stringArray(candidate.nodeIds, 0, 10_000, 500),
    edgeIds: stringArray(candidate.edgeIds, 0, 10_000, 500),
    algorithmConfigHash,
    modelVersionHash,
    modelVersion,
    modelScoreHash,
    modelTaskId,
    modelEntityType,
    modelEntityIds,
    limitations: limitations(candidate.limitations),
    evidenceHash: hashValue(candidate.evidenceHash),
  } as CoreEvidenceItem);
}

function parseSimilarCase(value: unknown): CoreSimilarCase {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "structuralRecordHash", "similarity", "sourceGraphVersionHash", "sourceEntityIds", "sourceKind",
    "modelVersion", "modelVersionHash", "representation", "queryHash", "representationSchema", "similarCaseHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-similar-case/2.0") invalid();
  validHash(candidate, "similarCaseHash");
  const similarity = finiteNumber(candidate.similarity);
  if (similarity < -1 || similarity > 1) invalid();
  if (candidate.representationSchema !== "socialgraph-fm.core-structural-record/2.0") invalid();
  return {
    schemaVersion: candidate.schemaVersion,
    structuralRecordHash: hashValue(candidate.structuralRecordHash),
    similarity,
    sourceGraphVersionHash: hashValue(candidate.sourceGraphVersionHash),
    sourceEntityIds: stringArray(candidate.sourceEntityIds, 1, 10_000, 500),
    sourceKind: literal(candidate.sourceKind, new Set(["node", "ego", "community"] as const)),
    modelVersion: stringValue(candidate.modelVersion, 1, 300),
    modelVersionHash: hashValue(candidate.modelVersionHash),
    representation: literal(candidate.representation, new Set(["embedding", "motif-signature"] as const)),
    queryHash: hashValue(candidate.queryHash),
    representationSchema: candidate.representationSchema,
    similarCaseHash: hashValue(candidate.similarCaseHash),
  };
}

function parseFinding(value: unknown): CoreFinding {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "taskId", "findingType", "subjectIds", "score", "calibratedConfidence", "evidence",
    "similarCases", "graphVersionHash", "modelVersion", "modelVersionHash", "limitations", "reviewStatus", "findingHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-finding/2.0") invalid();
  validHash(candidate, "findingHash");
  const taskId = task(candidate.taskId);
  const findingType = literal(candidate.findingType, new Set<CoreFindingType>([
    "community-resilience-candidate", "node-risk-candidate", "signed-relation-review", "core-collaboration-completion",
  ]));
  const score = parseScore(candidate.score);
  const calibratedConfidence = parseConfidence(candidate.calibratedConfidence);
  const evidence = arrayValue(candidate.evidence, 1, 1_000).map(parseEvidence);
  const similarCases = arrayValue(candidate.similarCases, 0, 1_000).map(parseSimilarCase);
  const subjectIds = stringArray(candidate.subjectIds, 1, 10_000, 500);
  const graphVersionHash = hashValue(candidate.graphVersionHash);
  const modelVersion = stringValue(candidate.modelVersion, 1, 300);
  const modelVersionHash = hashValue(candidate.modelVersionHash);
  const findingLimitations = limitations(candidate.limitations, 2);
  const compatibility = new Map<string, CoreEntityType>([
    ["core.community_resilience_review|community-resilience-candidate", "community"],
    ["core.risk_and_trust_review|node-risk-candidate", "node"],
    ["core.risk_and_trust_review|signed-relation-review", "edge"],
    ["core.collaboration_completion|core-collaboration-completion", "node-pair"],
  ]);
  if (
    compatibility.get(`${taskId}|${findingType}`) !== score.entityType
    || taskId !== score.taskId
    || !sameArray(subjectIds, score.entityIds)
    || graphVersionHash !== score.graphVersionHash
    || modelVersion !== score.modelVersion
    || modelVersionHash !== score.modelVersionHash
  ) invalid();
  if (taskId === "core.community_resilience_review") {
    if (
      calibratedConfidence.schemaVersion !== "socialgraph-fm.core-regression-confidence-interval/1.0"
      || calibratedConfidence.pointEstimate !== score.score
      || !findingLimitations.includes(REGRESSION_INTERVAL)
    ) invalid();
  } else if (calibratedConfidence.schemaVersion !== "socialgraph-fm.core-calibrated-confidence/2.0") invalid();
  if (
    calibratedConfidence.scoreHash !== score.scoreHash
    || calibratedConfidence.taskId !== score.taskId
    || calibratedConfidence.entityType !== score.entityType
    || !sameArray(calibratedConfidence.entityIds, score.entityIds)
    || calibratedConfidence.graphVersionHash !== score.graphVersionHash
    || calibratedConfidence.modelVersion !== score.modelVersion
    || calibratedConfidence.modelVersionHash !== score.modelVersionHash
  ) invalid();
  if (evidence.some((item) => item.graphVersionHash !== graphVersionHash)) invalid();
  for (const item of evidence) {
    if (item.sourceType === "registered-model-output" && (
      item.metric !== "registered_model.score-reference"
      || item.valueCanonicalJson !== "{}"
      || item.modelScoreHash !== score.scoreHash
      || item.modelTaskId !== score.taskId
      || item.modelEntityType !== score.entityType
      || !item.modelEntityIds
      || !sameArray(item.modelEntityIds, score.entityIds)
      || item.modelVersion !== score.modelVersion
      || item.modelVersionHash !== score.modelVersionHash
    )) invalid();
  }
  if (similarCases.some((item) => item.modelVersion !== modelVersion || item.modelVersionHash !== modelVersionHash)) invalid();
  if (!findingLimitations.includes(MANUAL_REVIEW) || !findingLimitations.includes(NON_CAUSAL)) invalid();
  if (candidate.reviewStatus !== "pending-human-review") invalid();
  return deepFreeze({
    schemaVersion: candidate.schemaVersion,
    taskId,
    findingType,
    subjectIds,
    score,
    calibratedConfidence,
    evidence,
    similarCases,
    graphVersionHash,
    modelVersion,
    modelVersionHash,
    limitations: findingLimitations,
    reviewStatus: candidate.reviewStatus,
    findingHash: hashValue(candidate.findingHash),
  } as CoreFinding);
}

export function parseCoreFinding(value: unknown): CoreFinding {
  return safeParse("GFM_CORE_FINDING_INVALID", () => parseFinding(value));
}

function parseResult(value: unknown): CoreRunResult {
  const candidate = record(value);
  exactKeys(candidate, [
    "schemaVersion", "runId", "requestHash", "taskId", "graphVersionId", "graphVersionHash", "modelVersionId",
    "modelVersionHash", "findings", "completedAt", "resultHash",
  ]);
  if (candidate.schemaVersion !== "socialgraph-fm.core-run-result/2.0") invalid();
  validHash(candidate, "resultHash");
  const taskId = task(candidate.taskId);
  const graphVersionHash = hashValue(candidate.graphVersionHash);
  const modelVersionId = stringValue(candidate.modelVersionId, 1, 300);
  const modelVersionHash = hashValue(candidate.modelVersionHash);
  const findings = arrayValue(candidate.findings, 0, 10_000).map(parseFinding);
  if (findings.some((finding) => (
    finding.taskId !== taskId
    || finding.graphVersionHash !== graphVersionHash
    || finding.modelVersion !== modelVersionId
    || finding.modelVersionHash !== modelVersionHash
  ))) invalid();
  return deepFreeze({
    schemaVersion: candidate.schemaVersion,
    runId: stringValue(candidate.runId, 1, 100),
    requestHash: hashValue(candidate.requestHash),
    taskId,
    graphVersionId: stringValue(candidate.graphVersionId, 1, 200),
    graphVersionHash,
    modelVersionId,
    modelVersionHash,
    findings,
    completedAt: dateTime(candidate.completedAt),
    resultHash: hashValue(candidate.resultHash),
  } as CoreRunResult);
}

export function parseCoreRunResult(value: unknown, binding?: CoreRunBinding): CoreRunResult {
  const parsed = safeParse("GFM_CORE_RUN_RESULT_INVALID", () => parseResult(value));
  if (binding && (
    parsed.runId !== binding.runId
    || parsed.requestHash !== binding.serverRequestHash
    || parsed.taskId !== binding.taskId
    || parsed.graphVersionId !== binding.graphVersionId
    || parsed.modelVersionId !== binding.modelVersionId
  )) throw new Error("GFM_CORE_RESPONSE_BINDING_INVALID");
  return parsed;
}

export function parseCoreError(value: unknown): Readonly<{ code: string }> {
  return safeParse("GFM_CORE_ERROR_INVALID", () => {
    const envelope = record(value);
    const keys = Object.keys(envelope);
    if (keys.length !== 1 || (keys[0] !== "error" && keys[0] !== "detail")) invalid();
    const body = record(envelope[keys[0]]);
    exactKeys(body, ["code"]);
    const code = stringValue(body.code, 1, 100);
    if (!ERROR_CODE.test(code) || !isPublicCoreErrorCode(code)) invalid();
    return deepFreeze({ code });
  });
}
