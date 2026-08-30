import type {
  GlobalModelCapabilities,
  GlobalModelEvidenceSubgraph,
  GlobalModelHealth,
  GlobalModelMetric,
  GlobalModelModelCard,
  GlobalModelNodeEvidence,
  GlobalModelNodeFinding,
  GlobalModelPreviewEdge,
  GlobalModelProtocol,
  GlobalModelProtocolModel,
  GlobalModelRelationEvidence,
  GlobalModelRelationModality,
  GlobalModelReviewRecord,
  GlobalModelReviewRequest,
  GlobalModelRoute,
  GlobalModelRunBinding,
  GlobalModelRunRequest,
  GlobalModelRunResult,
  GlobalModelRunStatus,
  GlobalModelScenario,
  GlobalModelScenarioPreview,
} from "../types/globalModel";
import {
  GLOBAL_MODEL_SCHEMA,
  GLOBAL_MODEL_HEALTH_SCHEMA,
  GLOBAL_MODEL_CARD_SCHEMA,
  GLOBAL_MODEL_PROTOCOLS,
} from "../types/globalModel";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";

type JsonRecord = Record<string, unknown>;

const HASH = /^[0-9a-f]{64}$/u;
const RUN_ID = /^global-model-[0-9a-f]{32}$/u;
const REVIEW_ID = /^review-[0-9a-f]{32}$/u;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u;
const PROTOCOL_SET = new Set<GlobalModelProtocol>(GLOBAL_MODEL_PROTOCOLS);
const RELATION_MODALITIES = ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"] as const;
const RELATION_MODALITY_SET = new Set<string>(RELATION_MODALITIES);
const COUNTRIES = ["china", "cuba", "iran", "russia", "UAE", "venezuela"] as const;

function invalid(): never {
  throw new Error("GFM_GLOBAL_MODEL_RESPONSE_INVALID");
}

function safeParse<T>(operation: () => T): T {
  try {
    return operation();
  } catch (error) {
    if (error instanceof Error && error.message === "GFM_GLOBAL_MODEL_RESPONSE_INVALID") throw error;
    return invalid();
  }
}

function record(value: unknown): JsonRecord {
  if (!value || typeof value !== "object" || Array.isArray(value)) invalid();
  return value as JsonRecord;
}

function exactKeys(value: JsonRecord, keys: readonly string[]): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (actual.length !== expected.length || actual.some((key, index) => key !== expected[index])) invalid();
}

function stringValue(value: unknown, max = 500): string {
  if (typeof value !== "string" || value.length < 1 || value.length > max) invalid();
  return value;
}

function nullableString(value: unknown, max = 500): string | null {
  return value === null ? null : stringValue(value, max);
}

function bool(value: unknown): boolean {
  if (typeof value !== "boolean") invalid();
  return value;
}

function numberValue(value: unknown, min = -Number.MAX_VALUE, max = Number.MAX_VALUE): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < min || value > max) invalid();
  return value;
}

function integer(value: unknown, min = 0, max = Number.MAX_SAFE_INTEGER): number {
  const parsed = numberValue(value, min, max);
  if (!Number.isInteger(parsed)) invalid();
  return parsed;
}

function hash(value: unknown): string {
  const parsed = stringValue(value, 64);
  if (!HASH.test(parsed)) invalid();
  return parsed;
}

function nullableHash(value: unknown): string | null {
  return value === null ? null : hash(value);
}

function dateTime(value: unknown): string {
  const parsed = stringValue(value, 50);
  if (!DATE_TIME.test(parsed) || Number.isNaN(Date.parse(parsed))) invalid();
  return parsed;
}

function arrayValue(value: unknown, min = 0, max = 10_000): readonly unknown[] {
  if (!Array.isArray(value) || value.length < min || value.length > max) invalid();
  return value;
}

function stringArray(value: unknown, min = 0, max = 100): readonly string[] {
  const parsed = arrayValue(value, min, max).map((item) => stringValue(item));
  if (new Set(parsed).size !== parsed.length) invalid();
  return parsed;
}

function schema(value: unknown): void {
  if (value !== GLOBAL_MODEL_SCHEMA) invalid();
}

function protocol(value: unknown): GlobalModelProtocol {
  if (typeof value !== "string" || !PROTOCOL_SET.has(value as GlobalModelProtocol)) invalid();
  return value as GlobalModelProtocol;
}

function relationModality(value: unknown): GlobalModelRelationModality {
  if (typeof value !== "string" || !RELATION_MODALITY_SET.has(value)) invalid();
  return value as GlobalModelRelationModality;
}

function serviceIdentity(
  modelVersionId: string | null,
  modelVersionHash: string | null,
  corpusHash: string | null,
): string {
  return sha256Canonical({
    service: "socialgraph-fm-gfm/global-model",
    datasetVersionId: "socialgraph-fm:russia",
    modelVersionId,
    modelVersionHash,
    corpusHash,
  });
}

function protocolInventory(value: unknown): typeof GLOBAL_MODEL_PROTOCOLS {
  const parsed = arrayValue(value, 4, 4).map(protocol);
  if (parsed.some((item, index) => item !== GLOBAL_MODEL_PROTOCOLS[index])) invalid();
  return GLOBAL_MODEL_PROTOCOLS;
}

function verifyCanonicalHash(value: JsonRecord, field: string): void {
  const payload = Object.fromEntries(Object.entries(value).filter(([key]) => key !== field));
  if (hash(value[field]) !== sha256Canonical(payload)) invalid();
}

function parseMetric(value: unknown): GlobalModelMetric {
  const candidate = record(value);
  exactKeys(candidate, ["macroF1", "prAuc", "threshold", "labelledTrainNodes"]);
  return {
    macroF1: numberValue(candidate.macroF1, 0, 1),
    prAuc: numberValue(candidate.prAuc, 0, 1),
    threshold: numberValue(candidate.threshold, 0, 1),
    labelledTrainNodes: integer(candidate.labelledTrainNodes),
  };
}

function parseProtocolModels(value: unknown): Readonly<Record<GlobalModelProtocol, GlobalModelProtocolModel>> {
  const candidate = record(value);
  exactKeys(candidate, GLOBAL_MODEL_PROTOCOLS);
  const parsed = Object.fromEntries(GLOBAL_MODEL_PROTOCOLS.map((id) => {
    const item = record(candidate[id]);
    exactKeys(item, ["modelVersionId", "modelVersionHash", "modelStateHash", "state"]);
    const state = item.state === "frozenDemo"
      ? "frozenDemo"
      : item.state === "servingReady"
        ? "servingReady"
        : invalid();
    if ((id === "global") !== (state === "servingReady")) invalid();
    return [id, {
      modelVersionId: stringValue(item.modelVersionId, 300),
      modelVersionHash: hash(item.modelVersionHash),
      modelStateHash: hash(item.modelStateHash),
      state,
    }] as const;
  })) as Record<GlobalModelProtocol, GlobalModelProtocolModel>;
  if (
    new Set(Object.values(parsed).map((item) => item.modelVersionId)).size !== GLOBAL_MODEL_PROTOCOLS.length
    || new Set(Object.values(parsed).map((item) => item.modelVersionHash)).size !== GLOBAL_MODEL_PROTOCOLS.length
  ) invalid();
  return parsed;
}

function parseRoute(value: unknown): GlobalModelRoute {
  const candidate = record(value);
  exactKeys(candidate, ["expert", "weight"]);
  return {
    expert: stringValue(candidate.expert, 50),
    weight: numberValue(candidate.weight, 0, 1),
  };
}

function parseRelationEvidence(value: unknown): GlobalModelRelationEvidence {
  const candidate = record(value);
  exactKeys(candidate, ["modality", "rawWeight"]);
  return {
    modality: relationModality(candidate.modality),
    rawWeight: numberValue(candidate.rawWeight),
  };
}

function parseFinding(value: unknown): GlobalModelNodeFinding {
  const candidate = record(value);
  exactKeys(candidate, [
    "nodeId", "score", "rank", "riskBand", "predictedPositive", "structureMissing",
    "routes", "modalityEvidence",
  ]);
  const riskBand = candidate.riskBand;
  if (riskBand !== "high" && riskBand !== "review" && riskBand !== "low") invalid();
  const modalities = record(candidate.modalityEvidence);
  exactKeys(modalities, ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"]);
  return {
    nodeId: stringValue(candidate.nodeId, 100),
    score: numberValue(candidate.score, 0, 1),
    rank: integer(candidate.rank, 1),
    riskBand,
    predictedPositive: bool(candidate.predictedPositive),
    structureMissing: bool(candidate.structureMissing),
    routes: arrayValue(candidate.routes, 1, 3).map(parseRoute),
    modalityEvidence: {
      coRT: integer(modalities.coRT),
      coURL: integer(modalities.coURL),
      hashSeq: integer(modalities.hashSeq),
      fastRT: integer(modalities.fastRT),
      tweetSim: integer(modalities.tweetSim),
    },
  };
}

export function parseGlobalModelCapabilities(value: unknown): GlobalModelCapabilities {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "channel", "releaseLabel", "seed", "servingReady", "unavailableReason",
      "taskId", "datasetVersionId", "model", "capabilityHash",
    ]);
    schema(candidate.schemaVersion);
    if (
      candidate.channel !== "global-model"
      || candidate.releaseLabel !== "SocialGraph-FM Global"
      || candidate.seed !== 12121995
      || candidate.taskId !== "coordination_risk"
      || candidate.datasetVersionId !== "socialgraph-fm:russia"
    ) invalid();
    const servingReady = bool(candidate.servingReady);
    const model: GlobalModelCapabilities["model"] = candidate.model === null ? null : (() => {
      const item = record(candidate.model);
      exactKeys(item, [
        "modelVersionId", "modelVersionHash", "artifactHash", "corpusHash", "sourceCodeHash",
        "taskId", "protocols", "protocolModels", "state",
      ]);
      const state = item.state === "preliminary"
        ? "preliminary"
        : item.state === "servingReady"
          ? "servingReady"
          : invalid();
      if (item.taskId !== "coordination_risk") invalid();
      const modelVersionId = stringValue(item.modelVersionId, 300);
      const modelVersionHash = hash(item.modelVersionHash);
      const protocolModels = parseProtocolModels(item.protocolModels);
      if (
        protocolModels.global.modelVersionId !== modelVersionId
        || protocolModels.global.modelVersionHash !== modelVersionHash
      ) invalid();
      return {
        modelVersionId,
        modelVersionHash,
        artifactHash: hash(item.artifactHash),
        corpusHash: hash(item.corpusHash),
        sourceCodeHash: hash(item.sourceCodeHash),
        taskId: "coordination_risk" as const,
        protocols: protocolInventory(item.protocols),
        protocolModels,
        state: state as "preliminary" | "servingReady",
      };
    })();
    if (servingReady !== Boolean(model)) invalid();
    verifyCanonicalHash(candidate, "capabilityHash");
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      channel: "global-model",
      releaseLabel: "SocialGraph-FM Global",
      seed: 12121995,
      servingReady,
      unavailableReason: nullableString(candidate.unavailableReason),
      taskId: "coordination_risk",
      datasetVersionId: "socialgraph-fm:russia",
      model,
      capabilityHash: hash(candidate.capabilityHash),
    });
  });
}

export function parseGlobalModelHealth(value: unknown): GlobalModelHealth {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "serviceIdentity", "servingReady", "modelVersionId",
      "modelVersionHash", "corpusHash", "datasetVersionId", "healthHash",
    ]);
    if (
      candidate.schemaVersion !== GLOBAL_MODEL_HEALTH_SCHEMA
      || candidate.datasetVersionId !== "socialgraph-fm:russia"
    ) invalid();
    const servingReady = bool(candidate.servingReady);
    const modelVersionId = nullableString(candidate.modelVersionId, 300);
    const modelVersionHash = nullableHash(candidate.modelVersionHash);
    const corpusHash = nullableHash(candidate.corpusHash);
    if (servingReady !== Boolean(modelVersionId && modelVersionHash && corpusHash)) invalid();
    const parsedIdentity = hash(candidate.serviceIdentity);
    if (parsedIdentity !== serviceIdentity(modelVersionId, modelVersionHash, corpusHash)) invalid();
    verifyCanonicalHash(candidate, "healthHash");
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_HEALTH_SCHEMA,
      serviceIdentity: parsedIdentity,
      servingReady,
      modelVersionId,
      modelVersionHash,
      corpusHash,
      datasetVersionId: "socialgraph-fm:russia",
      healthHash: hash(candidate.healthHash),
    });
  });
}

export function parseGlobalModelModelCard(value: unknown): GlobalModelModelCard {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "releaseId", "modelVersionId", "modelVersionHash", "taskId",
      "architecture", "protocols", "trainingData", "intendedUse", "outOfScope",
      "limitations", "ethics", "licenses", "sourceAttribution", "metrics",
      "artifactHash", "modelCardHash",
    ]);
    if (
      candidate.schemaVersion !== GLOBAL_MODEL_CARD_SCHEMA
      || candidate.releaseId !== "socialgraph-fm"
      || candidate.taskId !== "coordination_risk"
    ) invalid();
    const modelVersionId = stringValue(candidate.modelVersionId, 300);
    const modelVersionHash = hash(candidate.modelVersionHash);
    const architecture = record(candidate.architecture);
    exactKeys(architecture, [
      "name", "textFeatures", "structuralFeatures", "gnnLayers", "hiddenDim", "router",
    ]);
    if (architecture.gnnLayers !== 2 || architecture.hiddenDim !== 256) invalid();
    const protocols = parseProtocolModels(candidate.protocols);
    if (
      protocols.global.modelVersionId !== modelVersionId
      || protocols.global.modelVersionHash !== modelVersionHash
    ) invalid();
    const trainingData = record(candidate.trainingData);
    exactKeys(trainingData, ["countries", "nodeCount", "nodeCountByCountry", "content"]);
    const countries = arrayValue(trainingData.countries, 6, 6).map((item) => stringValue(item, 20));
    if (countries.some((country, index) => country !== COUNTRIES[index])) invalid();
    const rawCounts = record(trainingData.nodeCountByCountry);
    exactKeys(rawCounts, COUNTRIES);
    const nodeCountByCountry = Object.fromEntries(
      COUNTRIES.map((country) => [country, integer(rawCounts[country], 1)]),
    ) as Record<typeof COUNTRIES[number], number>;
    const nodeCount = integer(trainingData.nodeCount, 1);
    if (Object.values(nodeCountByCountry).reduce((sum, count) => sum + count, 0) !== nodeCount) invalid();
    const licenses = arrayValue(candidate.licenses, 2, 2).map((value) => {
      const item = record(value);
      exactKeys(item, ["name", "license", "url"]);
      if (item.license !== "CC-BY-4.0" && item.license !== "MIT") invalid();
      return {
        name: stringValue(item.name, 300),
        license: item.license as "CC-BY-4.0" | "MIT",
        url: stringValue(item.url, 2_048),
      };
    });
    if (new Set(licenses.map((item) => item.license)).size !== 2) invalid();
    const attribution = record(candidate.sourceAttribution);
    exactKeys(attribution, ["kind", "paperUrl", "completeReproduction"]);
    if (attribution.kind !== "inspired" || attribution.completeReproduction !== false) invalid();
    const rawMetrics = record(candidate.metrics);
    exactKeys(rawMetrics, GLOBAL_MODEL_PROTOCOLS);
    const metrics = Object.fromEntries(GLOBAL_MODEL_PROTOCOLS.map((id) => [
      id,
      record(rawMetrics[id]),
    ])) as Record<GlobalModelProtocol, Readonly<Record<string, unknown>>>;
    verifyCanonicalHash(candidate, "modelCardHash");
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_CARD_SCHEMA,
      releaseId: "socialgraph-fm",
      modelVersionId,
      modelVersionHash,
      taskId: "coordination_risk",
      architecture: {
        name: stringValue(architecture.name),
        textFeatures: stringValue(architecture.textFeatures),
        structuralFeatures: stringValue(architecture.structuralFeatures),
        gnnLayers: 2,
        hiddenDim: 256,
        router: stringValue(architecture.router),
      },
      protocols,
      trainingData: {
        countries: COUNTRIES,
        nodeCount,
        nodeCountByCountry,
        content: stringValue(trainingData.content, 1_000),
      },
      intendedUse: stringArray(candidate.intendedUse, 1),
      outOfScope: stringArray(candidate.outOfScope, 1),
      limitations: stringArray(candidate.limitations, 1),
      ethics: stringArray(candidate.ethics, 1),
      licenses,
      sourceAttribution: {
        kind: "inspired",
        paperUrl: stringValue(attribution.paperUrl, 2_048),
        completeReproduction: false,
      },
      metrics,
      artifactHash: hash(candidate.artifactHash),
      modelCardHash: hash(candidate.modelCardHash),
    });
  });
}

export function parseGlobalModelScenario(value: unknown): GlobalModelScenario {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "scenarioId", "datasetVersionId", "graphVersionHash", "modelVersionId",
      "enabled", "unavailableReason", "nodeCount", "edgeCount", "protocols", "metrics",
      "limitations", "scenarioHash",
    ]);
    schema(candidate.schemaVersion);
    if (
      candidate.scenarioId !== "russia-coordination-risk"
      || candidate.datasetVersionId !== "socialgraph-fm:russia"
      || candidate.nodeCount !== 716
    ) invalid();
    const enabled = bool(candidate.enabled);
    const graphVersionHash = nullableHash(candidate.graphVersionHash);
    const modelVersionId = nullableString(candidate.modelVersionId, 300);
    if (enabled !== Boolean(graphVersionHash && modelVersionId)) invalid();
    protocolInventory(candidate.protocols);
    const rawMetrics = record(candidate.metrics);
    exactKeys(rawMetrics, GLOBAL_MODEL_PROTOCOLS);
    const metrics = Object.fromEntries(GLOBAL_MODEL_PROTOCOLS.map((id) => [
      id,
      rawMetrics[id] === null ? null : parseMetric(rawMetrics[id]),
    ])) as Record<GlobalModelProtocol, GlobalModelMetric | null>;
    verifyCanonicalHash(candidate, "scenarioHash");
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      scenarioId: "russia-coordination-risk",
      datasetVersionId: "socialgraph-fm:russia",
      graphVersionHash,
      modelVersionId,
      enabled,
      unavailableReason: nullableString(candidate.unavailableReason),
      nodeCount: 716,
      edgeCount: integer(candidate.edgeCount),
      protocols: GLOBAL_MODEL_PROTOCOLS,
      metrics,
      limitations: stringArray(candidate.limitations),
      scenarioHash: hash(candidate.scenarioHash),
    });
  });
}

export function parseGlobalModelScenarioPreview(value: unknown): GlobalModelScenarioPreview {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "datasetVersionId", "graphVersionHash", "nodes", "edges", "nodeCount",
      "edgeCount", "partialPreview", "previewHash",
    ]);
    schema(candidate.schemaVersion);
    if (candidate.datasetVersionId !== "socialgraph-fm:russia" || candidate.nodeCount !== 716) invalid();
    const nodes = arrayValue(candidate.nodes, 1, 716).map((value) => {
      const node = record(value);
      exactKeys(node, ["id", "label", "degree", "structureMissing"]);
      return {
        id: stringValue(node.id, 100),
        label: stringValue(node.label, 100),
        degree: integer(node.degree),
        structureMissing: bool(node.structureMissing),
      };
    });
    const nodeIds = new Set(nodes.map((node) => node.id));
    if (nodeIds.size !== nodes.length) invalid();
    const modalities = new Set<GlobalModelPreviewEdge["modality"]>(["coRT", "coURL", "hashSeq", "fastRT", "tweetSim", "fused"]);
    const edges = arrayValue(candidate.edges, 0, 20_000).map((value) => {
      const edge = record(value);
      exactKeys(edge, ["id", "source", "target", "modality"]);
      if (typeof edge.modality !== "string" || !modalities.has(edge.modality as GlobalModelPreviewEdge["modality"])) invalid();
      const parsed: GlobalModelPreviewEdge = {
        id: stringValue(edge.id, 300),
        source: stringValue(edge.source, 100),
        target: stringValue(edge.target, 100),
        modality: edge.modality as GlobalModelPreviewEdge["modality"],
      };
      if (!nodeIds.has(parsed.source) || !nodeIds.has(parsed.target)) invalid();
      return parsed;
    });
    if (new Set(edges.map((edge) => edge.id)).size !== edges.length) invalid();
    const edgeCount = integer(candidate.edgeCount);
    const partialPreview = bool(candidate.partialPreview);
    if (partialPreview !== (nodes.length < 716 || edges.length < edgeCount)) invalid();
    verifyCanonicalHash(candidate, "previewHash");
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      datasetVersionId: "socialgraph-fm:russia",
      graphVersionHash: hash(candidate.graphVersionHash),
      nodes,
      edges,
      nodeCount: 716,
      edgeCount,
      partialPreview,
      previewHash: hash(candidate.previewHash),
    });
  });
}

export function parseGlobalModelRunRequest(value: unknown): GlobalModelRunRequest {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, ["schemaVersion", "taskId", "datasetVersionId", "protocol", "modelVersionId", "topK"]);
    schema(candidate.schemaVersion);
    if (candidate.taskId !== "coordination_risk" || candidate.datasetVersionId !== "socialgraph-fm:russia") invalid();
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      taskId: "coordination_risk",
      datasetVersionId: "socialgraph-fm:russia",
      protocol: protocol(candidate.protocol),
      modelVersionId: stringValue(candidate.modelVersionId, 300),
      topK: integer(candidate.topK, 1, 500),
    });
  });
}

export function parseGlobalModelRunStatus(value: unknown, binding?: GlobalModelRunBinding): GlobalModelRunStatus {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "runId", "requestHash", "status", "progress", "createdAt", "updatedAt", "errorCode",
    ]);
    schema(candidate.schemaVersion);
    const runId = stringValue(candidate.runId, 64);
    if (!RUN_ID.test(runId)) invalid();
    const status = candidate.status;
    if (status !== "queued" && status !== "running" && status !== "succeeded" && status !== "failed") invalid();
    const requestHash = hash(candidate.requestHash);
    const errorCode = nullableString(candidate.errorCode, 100);
    if ((status === "failed") !== Boolean(errorCode)) invalid();
    if (binding && (runId !== binding.runId || requestHash !== binding.serverRequestHash)) invalid();
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      runId,
      requestHash,
      status,
      progress: integer(candidate.progress, 0, 100),
      createdAt: dateTime(candidate.createdAt),
      updatedAt: dateTime(candidate.updatedAt),
      errorCode,
    });
  });
}

export function parseGlobalModelRunResult(value: unknown, binding?: GlobalModelRunBinding): GlobalModelRunResult {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "runId", "requestHash", "taskId", "protocol", "datasetVersionId",
      "graphVersionHash", "corpusHash", "splitHash", "modelVersionId", "modelVersionHash", "threshold",
      "metrics", "findings", "limitations", "completedAt", "resultHash",
    ]);
    schema(candidate.schemaVersion);
    if (candidate.taskId !== "coordination_risk" || candidate.datasetVersionId !== "socialgraph-fm:russia") invalid();
    const findings = arrayValue(candidate.findings, 0, 500).map(parseFinding);
    if (
      new Set(findings.map((finding) => finding.nodeId)).size !== findings.length
      || findings.some((finding, index) => finding.rank !== index + 1)
    ) invalid();
    const result: GlobalModelRunResult = {
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      runId: stringValue(candidate.runId, 64),
      requestHash: hash(candidate.requestHash),
      taskId: "coordination_risk",
      protocol: protocol(candidate.protocol),
      datasetVersionId: "socialgraph-fm:russia",
      graphVersionHash: hash(candidate.graphVersionHash),
      corpusHash: hash(candidate.corpusHash),
      splitHash: hash(candidate.splitHash),
      modelVersionId: stringValue(candidate.modelVersionId, 300),
      modelVersionHash: hash(candidate.modelVersionHash),
      threshold: numberValue(candidate.threshold, 0, 1),
      metrics: parseMetric(candidate.metrics),
      findings,
      limitations: stringArray(candidate.limitations),
      completedAt: dateTime(candidate.completedAt),
      resultHash: hash(candidate.resultHash),
    };
    if (!RUN_ID.test(result.runId)) invalid();
    if (binding && (
      result.runId !== binding.runId
      || result.requestHash !== binding.serverRequestHash
      || result.taskId !== binding.taskId
      || result.protocol !== binding.protocol
      || result.datasetVersionId !== binding.datasetVersionId
      || result.graphVersionHash !== binding.graphVersionHash
      || result.modelVersionId !== binding.modelVersionId
      || result.modelVersionHash !== binding.modelVersionHash
    )) invalid();
    verifyCanonicalHash(candidate, "resultHash");
    return deepFreeze(result);
  });
}

export function parseGlobalModelNodeEvidence(
  value: unknown,
  binding?: Pick<GlobalModelRunBinding, "runId">,
  expectedNodeId?: string,
  expectedResult?: GlobalModelRunResult,
): GlobalModelNodeEvidence {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "runId", "resultHash", "graphVersionHash", "modelVersionId",
      "modelVersionHash", "threshold", "node", "neighbors", "structuralSignals",
      "evidenceSubgraph", "limitation", "evidenceHash",
    ]);
    schema(candidate.schemaVersion);
    const node = parseFinding(candidate.node);
    const neighbors = arrayValue(candidate.neighbors, 0, 500).map((value) => {
      const item = record(value);
      exactKeys(item, [
        "nodeId", "score", "hop", "riskBand", "predictedPositive", "structureMissing",
        "modalities", "relations",
      ]);
      if (item.hop !== 1) invalid();
      const riskBand = item.riskBand;
      if (riskBand !== "high" && riskBand !== "review" && riskBand !== "low") invalid();
      const relations = arrayValue(item.relations, 0, RELATION_MODALITIES.length)
        .map(parseRelationEvidence);
      const modalities = arrayValue(item.modalities, 0, RELATION_MODALITIES.length)
        .map(relationModality);
      if (
        new Set(modalities).size !== modalities.length
        || modalities.some((modality, index) => modality !== relations[index]?.modality)
      ) invalid();
      return {
        nodeId: stringValue(item.nodeId, 100),
        score: numberValue(item.score, 0, 1),
        hop: 1 as const,
        riskBand: riskBand as GlobalModelNodeFinding["riskBand"],
        predictedPositive: bool(item.predictedPositive),
        structureMissing: bool(item.structureMissing),
        modalities,
        relations,
      };
    });
    const runId = stringValue(candidate.runId, 64);
    if (!RUN_ID.test(runId) || (binding && runId !== binding.runId) || (expectedNodeId && node.nodeId !== expectedNodeId)) invalid();
    const signals = record(candidate.structuralSignals);
    exactKeys(signals, [
      "fusedDegree", "structureMissing", "relationNeighborCounts", "twoHopNodeCount",
      "relationEvidenceRole",
    ]);
    if (signals.relationEvidenceRole !== "explanationOnly") invalid();
    const rawCounts = record(signals.relationNeighborCounts);
    exactKeys(rawCounts, RELATION_MODALITIES);
    const relationNeighborCounts = {
      coRT: integer(rawCounts.coRT),
      coURL: integer(rawCounts.coURL),
      hashSeq: integer(rawCounts.hashSeq),
      fastRT: integer(rawCounts.fastRT),
      tweetSim: integer(rawCounts.tweetSim),
    };
    const rawSubgraph = record(candidate.evidenceSubgraph);
    exactKeys(rawSubgraph, ["depth", "nodeCount", "edgeCount", "truncated", "nodes", "edges"]);
    if (rawSubgraph.depth !== 2) invalid();
    const subgraphNodes = arrayValue(rawSubgraph.nodes, 1, 716).map((value) => {
      const item = record(value);
      exactKeys(item, [
        "nodeId", "score", "hop", "riskBand", "predictedPositive", "structureMissing",
      ]);
      const hop = integer(item.hop, 0, 2);
      const riskBand = item.riskBand;
      if (riskBand !== "high" && riskBand !== "review" && riskBand !== "low") invalid();
      return {
        nodeId: stringValue(item.nodeId, 100),
        score: numberValue(item.score, 0, 1),
        hop: hop as 0 | 1 | 2,
        riskBand: riskBand as GlobalModelNodeFinding["riskBand"],
        predictedPositive: bool(item.predictedPositive),
        structureMissing: bool(item.structureMissing),
      };
    });
    const subgraphNodeIds = new Set(subgraphNodes.map((item) => item.nodeId));
    const subgraphEdges = arrayValue(rawSubgraph.edges, 0, 20_000).map((value) => {
      const item = record(value);
      exactKeys(item, ["id", "source", "target", "relations", "evidenceRole"]);
      if (item.evidenceRole !== "explanationOnly") invalid();
      const source = stringValue(item.source, 100);
      const target = stringValue(item.target, 100);
      if (!subgraphNodeIds.has(source) || !subgraphNodeIds.has(target)) invalid();
      return {
        id: stringValue(item.id, 300),
        source,
        target,
        relations: arrayValue(item.relations, 0, RELATION_MODALITIES.length)
          .map(parseRelationEvidence),
        evidenceRole: "explanationOnly" as const,
      };
    });
    const evidenceSubgraph: GlobalModelEvidenceSubgraph = {
      depth: 2,
      nodeCount: integer(rawSubgraph.nodeCount, 1, 716),
      edgeCount: integer(rawSubgraph.edgeCount),
      truncated: bool(rawSubgraph.truncated),
      nodes: subgraphNodes,
      edges: subgraphEdges,
    };
    if (
      evidenceSubgraph.nodeCount !== subgraphNodes.length
      || evidenceSubgraph.edgeCount !== subgraphEdges.length
      || subgraphNodeIds.size !== subgraphNodes.length
      || new Set(subgraphEdges.map((item) => item.id)).size !== subgraphEdges.length
      || !subgraphNodes.some((item) => item.nodeId === node.nodeId && item.hop === 0 && item.score === node.score)
      || neighbors.some((item) => !subgraphNodeIds.has(item.nodeId))
      || integer(signals.fusedDegree) !== neighbors.length
      || bool(signals.structureMissing) !== node.structureMissing
      || integer(signals.twoHopNodeCount, 0, 715) !== subgraphNodes.length - 1
    ) invalid();
    const resultHash = hash(candidate.resultHash);
    const graphVersionHash = hash(candidate.graphVersionHash);
    const modelVersionId = stringValue(candidate.modelVersionId, 300);
    const modelVersionHash = hash(candidate.modelVersionHash);
    const threshold = numberValue(candidate.threshold, 0, 1);
    if (expectedResult && (
      resultHash !== expectedResult.resultHash
      || graphVersionHash !== expectedResult.graphVersionHash
      || modelVersionId !== expectedResult.modelVersionId
      || modelVersionHash !== expectedResult.modelVersionHash
      || threshold !== expectedResult.threshold
    )) invalid();
    verifyCanonicalHash(candidate, "evidenceHash");
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      runId,
      resultHash,
      graphVersionHash,
      modelVersionId,
      modelVersionHash,
      threshold,
      node,
      neighbors,
      structuralSignals: {
        fusedDegree: integer(signals.fusedDegree),
        structureMissing: bool(signals.structureMissing),
        relationNeighborCounts,
        twoHopNodeCount: integer(signals.twoHopNodeCount, 0, 715),
        relationEvidenceRole: "explanationOnly",
      },
      evidenceSubgraph,
      limitation: stringValue(candidate.limitation, 500),
      evidenceHash: hash(candidate.evidenceHash),
    });
  });
}

export function parseGlobalModelReviewRequest(value: unknown): GlobalModelReviewRequest {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, ["schemaVersion", "nodeId", "decision", "reason"]);
    schema(candidate.schemaVersion);
    const decision = candidate.decision;
    if (decision !== "confirmed" && decision !== "rejected" && decision !== "pending") invalid();
    return deepFreeze({
      schemaVersion: GLOBAL_MODEL_SCHEMA,
      nodeId: stringValue(candidate.nodeId, 100),
      decision,
      reason: stringValue(candidate.reason, 1000),
    });
  });
}

export function parseGlobalModelReviewRecord(
  value: unknown,
  binding?: Pick<GlobalModelRunBinding, "runId">,
  request?: GlobalModelReviewRequest,
): GlobalModelReviewRecord {
  return safeParse(() => {
    const candidate = record(value);
    exactKeys(candidate, [
      "schemaVersion", "reviewId", "runId", "nodeId", "decision", "reason", "createdAt", "reviewHash",
    ]);
    const parsedRequest = parseGlobalModelReviewRequest({
      schemaVersion: candidate.schemaVersion,
      nodeId: candidate.nodeId,
      decision: candidate.decision,
      reason: candidate.reason,
    });
    const reviewId = stringValue(candidate.reviewId, 39);
    const runId = stringValue(candidate.runId, 64);
    if (!REVIEW_ID.test(reviewId) || !RUN_ID.test(runId) || (binding && runId !== binding.runId)) invalid();
    if (request && (
      parsedRequest.nodeId !== request.nodeId
      || parsedRequest.decision !== request.decision
      || parsedRequest.reason !== request.reason
    )) invalid();
    verifyCanonicalHash(candidate, "reviewHash");
    return deepFreeze({
      ...parsedRequest,
      reviewId,
      runId,
      createdAt: dateTime(candidate.createdAt),
      reviewHash: hash(candidate.reviewHash),
    });
  });
}
