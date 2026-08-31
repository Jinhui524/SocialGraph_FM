import type {
  ColumnMapping,
  FileColumnProfile,
  GraphBuildIntentInput,
  GraphBuildIntentNormalizer,
  GraphBuildIntentResult,
  GraphBuildSpec,
  NodeColumnMapping,
} from "../types/graph";
import { SOCIALGRAPH_API_BASE_URL } from "./apiClient";

type Fetcher = typeof fetch;
interface JsonObject { readonly [key: string]: unknown }

function isObject(value: unknown): value is JsonObject {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function normalizeName(value: string): string {
  return value.trim().toLocaleLowerCase().replace(/[\s_-]+/gu, "");
}

function findColumn(columns: readonly FileColumnProfile[], aliases: readonly string[]): string | undefined {
  const normalized = new Set(aliases.map(normalizeName));
  return columns.find((column) => normalized.has(normalizeName(column.name)))?.name;
}

function inferEdgeMapping(columns: readonly FileColumnProfile[]): ColumnMapping | undefined {
  const source = findColumn(columns, ["source", "src", "from", "source_id", "源节点", "起点"]);
  const target = findColumn(columns, ["target", "dst", "to", "target_id", "目标节点", "终点"]);
  if (!source || !target) return undefined;
  const optional = (aliases: readonly string[]) => findColumn(columns, aliases);
  const sourceLabel = optional(["source_label", "src_label", "源节点名称", "起点名称"]);
  const targetLabel = optional(["target_label", "dst_label", "目标节点名称", "终点名称"]);
  const sourceType = optional(["source_type", "src_type", "源节点类型", "起点类型"]);
  const targetType = optional(["target_type", "dst_type", "目标节点类型", "终点类型"]);
  const edgeType = optional(["edge_type", "relation", "relationship", "type", "关系", "关系类型"]);
  const weight = optional(["weight", "value", "strength", "权重", "强度"]);
  const timestamp = optional(["timestamp", "time", "date", "datetime", "year", "时间", "日期", "年份", "年度"]);
  return {
    source,
    target,
    ...(sourceLabel ? { sourceLabel } : {}),
    ...(targetLabel ? { targetLabel } : {}),
    ...(sourceType ? { sourceType } : {}),
    ...(targetType ? { targetType } : {}),
    ...(edgeType ? { edgeType } : {}),
    ...(weight ? { weight } : {}),
    ...(timestamp ? { timestamp } : {}),
  };
}

function inferNodeMapping(columns: readonly FileColumnProfile[]): NodeColumnMapping | undefined {
  const id = findColumn(columns, ["id", "node_id", "nodeid", "节点id", "节点编号"]);
  if (!id) return undefined;
  const label = findColumn(columns, ["label", "name", "node_label", "节点名称", "名称"]);
  const type = findColumn(columns, ["node_type", "category", "节点类型", "类别", "type"]);
  return { id, ...(label ? { label } : {}), ...(type ? { type } : {}) };
}

function policiesFromDescription(description: string) {
  const duplicateEdgePolicy = /(?:拒绝|不允许|禁止).{0,4}(?:重复边|重复关系)/u.test(description)
    ? "reject" as const
    : /(?:合并|汇总|求和).{0,4}(?:重复边|重复关系)/u.test(description)
      ? "merge_sum" as const
      : "preserve" as const;
  const selfLoopPolicy = /(?:拒绝|不允许|禁止|删除|去除).{0,4}(?:自环|自连接)/u.test(description)
    ? "reject" as const
    : "preserve" as const;
  const timeFormat = /unix.{0,3}(?:毫秒|milliseconds?)/iu.test(description)
    ? "unix_milliseconds" as const
    : /unix.{0,3}(?:秒|seconds?)/iu.test(description)
      ? "unix_seconds" as const
      : /(?:年份|年度|year)/iu.test(description)
        ? "year" as const
        : /iso\s*8601/iu.test(description)
          ? "iso8601" as const
          : "auto" as const;
  return { duplicateEdgePolicy, selfLoopPolicy, timeFormat };
}

interface GraphBuildApiPayload {
  readonly description: string;
  readonly columnProfiles?: readonly GraphBuildApiColumnProfile[];
  readonly files?: readonly {
    readonly role: "nodes" | "edges";
    readonly columnProfiles: readonly GraphBuildApiColumnProfile[];
  }[];
}

interface GraphBuildApiColumnProfile {
    readonly name: string;
    readonly inferredType: "string" | "integer" | "float" | "boolean" | "datetime" | "unknown";
    readonly nonNullCount: number;
    readonly nullCount: number;
    readonly uniqueCount: number;
}

function toApiColumnProfiles(columns: readonly FileColumnProfile[]): GraphBuildApiColumnProfile[] {
  return columns.map((column) => ({
    name: column.name,
    inferredType: column.inferredType === "number" ? "float"
      : column.inferredType === "empty" ? "unknown"
        : column.inferredType,
    nonNullCount: column.nonNullCount,
    nullCount: column.nullCount,
    uniqueCount: column.cardinality,
  }));
}

/** Reconstructs the backend's strict allowlist; graph values and source rows cannot leak. */
export function buildGraphBuildIntentPayload(input: GraphBuildIntentInput): GraphBuildApiPayload {
  const edgeFile = input.files.find((file) => file.role === "edges") ?? input.files[0];
  const nodeFile = input.files.find((file) => file.role === "nodes");
  const description = input.description.trim().slice(0, 4_000) || "请按文件字段建立规范关系图。";
  if (edgeFile && nodeFile) {
    return {
      description,
      files: [
        { role: "nodes", columnProfiles: toApiColumnProfiles(nodeFile.columns) },
        { role: "edges", columnProfiles: toApiColumnProfiles(edgeFile.columns) },
      ],
    };
  }
  return {
    description,
    columnProfiles: toApiColumnProfiles(edgeFile?.columns ?? []),
  };
}

function parseMapping(value: unknown, allowedColumns: ReadonlySet<string>): ColumnMapping | undefined {
  if (!isObject(value)) return undefined;
  const read = (key: string): string | undefined => {
    const candidate = value[key];
    return typeof candidate === "string" && allowedColumns.has(candidate) ? candidate : undefined;
  };
  const source = read("source");
  const target = read("target");
  if (!source || !target || source === target) return undefined;
  const sourceLabel = read("sourceLabel");
  const targetLabel = read("targetLabel");
  const sourceType = read("sourceType");
  const targetType = read("targetType");
  const edgeType = read("edgeType");
  const weight = read("weight");
  const timestamp = read("timestamp");
  return {
    source,
    target,
    ...(sourceLabel ? { sourceLabel } : {}),
    ...(targetLabel ? { targetLabel } : {}),
    ...(sourceType ? { sourceType } : {}),
    ...(targetType ? { targetType } : {}),
    ...(edgeType ? { edgeType } : {}),
    ...(weight ? { weight } : {}),
    ...(timestamp ? { timestamp } : {}),
  };
}

function parseNodeMapping(value: unknown, allowedColumns: ReadonlySet<string>): NodeColumnMapping | undefined {
  if (!isObject(value) || typeof value.id !== "string" || !allowedColumns.has(value.id)) return undefined;
  const label = typeof value.label === "string" && allowedColumns.has(value.label) ? value.label : undefined;
  const type = typeof value.type === "string" && allowedColumns.has(value.type) ? value.type : undefined;
  return { id: value.id, ...(label ? { label } : {}), ...(type ? { type } : {}) };
}

function parseResponse(value: unknown, input: GraphBuildIntentInput): GraphBuildIntentResult | undefined {
  if (!isObject(value) || value.kind !== "graph_build_intent" || !isObject(value.meta)) return undefined;
  if (
    value.meta.requestId !== input.requestToken ||
    !["1.0", "1.1"].includes(String(value.meta.schemaVersion)) ||
    value.meta.source !== "llm"
  ) return undefined;
  const ids = input.files.map((file) => file.artifactId);
  const edgeFile = input.files.find((file) => file.role === "edges") ?? input.files[0];
  const nodeFile = input.files.find((file) => file.role === "nodes");
  const edgeColumns = new Set(edgeFile?.columns.map((column) => column.name));
  const nodeColumns = new Set(nodeFile?.columns.map((column) => column.name));
  const mapping = isObject(value.mapping) ? value.mapping : {};
  const responseMapping = {
    source: mapping.sourceColumn,
    target: mapping.targetColumn,
    edgeType: mapping.edgeTypeColumn,
    weight: mapping.weightColumn,
    timestamp: mapping.timestampColumn,
  };
  const parsedEdgeMapping = parseMapping(responseMapping, edgeColumns);
  const inferredMetadata = edgeFile ? inferEdgeMapping(edgeFile.columns) : undefined;
  const edgeMapping = parsedEdgeMapping ? {
    ...parsedEdgeMapping,
    ...(inferredMetadata?.sourceLabel ? { sourceLabel: inferredMetadata.sourceLabel } : {}),
    ...(inferredMetadata?.targetLabel ? { targetLabel: inferredMetadata.targetLabel } : {}),
    ...(inferredMetadata?.sourceType ? { sourceType: inferredMetadata.sourceType } : {}),
    ...(inferredMetadata?.targetType ? { targetType: inferredMetadata.targetType } : {}),
  } : undefined;
  const responseNodeMapping = isObject(value.nodeMapping) ? {
    id: value.nodeMapping.idColumn,
    label: value.nodeMapping.labelColumn,
    type: value.nodeMapping.typeColumn,
  } : undefined;
  const nodeMapping = nodeFile
    ? parseNodeMapping(responseNodeMapping, nodeColumns) ?? inferNodeMapping(nodeFile.columns)
    : undefined;
  const standardGraph = input.files.length === 1 && ["json", "graphml", "gexf"].includes(input.files[0]?.format ?? "");
  if (!standardGraph && !edgeMapping && value.requiresMapping !== true) return undefined;
  const directionPolicy = value.directedness === "directed" ? "directed"
    : value.directedness === "undirected" ? "undirected"
      : standardGraph ? "file"
        : /无向/u.test(input.description) ? "undirected"
          : /有向/u.test(input.description) ? "directed"
            : "undirected";
  const describedPolicies = policiesFromDescription(input.description);
  const parsedSpec: GraphBuildSpec = Object.freeze({
    schemaVersion: "1.0",
    inputShape: standardGraph ? "standard_graph" : input.files.length === 2 ? "node_edge_tables" : "edge_table",
    sourceArtifactIds: Object.freeze([...ids]),
    ...(nodeMapping ? { nodeMapping: Object.freeze(nodeMapping) } : {}),
    ...(edgeMapping ? { edgeMapping: Object.freeze(edgeMapping) } : {}),
    directionPolicy,
    duplicateEdgePolicy: describedPolicies.duplicateEdgePolicy,
    selfLoopPolicy: describedPolicies.selfLoopPolicy,
    danglingEndpointPolicy: input.files.length === 2 ? "reject" : "derive_nodes",
    timeFormat: describedPolicies.timeFormat,
    ...(input.description.trim() ? { description: input.description.trim().slice(0, 2_000) } : {}),
  });
  return {
    kind: "construction_revision",
    requestToken: input.requestToken,
    ...(input.baseGraphVersionId ? { baseGraphVersionId: input.baseGraphVersionId } : {}),
    spec: parsedSpec,
    source: "llm",
    warnings: Array.isArray(value.meta.warnings)
      ? value.meta.warnings.filter((warning): warning is string => typeof warning === "string").slice(0, 20)
      : [],
  };
}

export interface HttpGraphBuildIntentNormalizerOptions {
  readonly baseUrl?: string;
  readonly timeoutMs?: number;
  readonly fetcher?: Fetcher;
}

export class HttpGraphBuildIntentNormalizer implements GraphBuildIntentNormalizer {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly fetcher: Fetcher;

  constructor(options: HttpGraphBuildIntentNormalizerOptions = {}) {
    this.baseUrl = (options.baseUrl ?? SOCIALGRAPH_API_BASE_URL).replace(/\/$/u, "");
    this.timeoutMs = options.timeoutMs ?? 15_000;
    this.fetcher = options.fetcher ?? window.fetch.bind(window);
  }

  async normalizeGraphBuildIntent(input: GraphBuildIntentInput): Promise<GraphBuildIntentResult> {
    const edgeFile = input.files.find((file) => file.role === "edges") ?? input.files[0];
    if (!edgeFile?.columns.length) throw new Error("GRAPH_BUILD_INTENT_COLUMNS_REQUIRED");
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetcher(`${this.baseUrl}/api/v1/graph-build-intents/normalize`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Request-ID": input.requestToken },
        body: JSON.stringify(buildGraphBuildIntentPayload(input)),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP_${response.status}`);
      const parsed = parseResponse(await response.json(), input);
      if (!parsed) throw new Error("INVALID_GRAPH_BUILD_INTENT_RESPONSE");
      return parsed;
    } finally {
      window.clearTimeout(timeout);
    }
  }
}
