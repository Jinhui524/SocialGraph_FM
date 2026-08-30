import Papa from "papaparse";

import type {
  ColumnMapping,
  FileProfile,
  GraphAttributeValue,
  GraphAttributes,
  GraphEdge,
  GraphImportAdapter,
  GraphImportParseOptions,
  GraphBuildSpec,
  GraphNode,
  NodeColumnMapping,
  GraphPreview,
  GraphVersion,
  ImportFormat,
  ImportRun,
  FileColumnProfile,
  GraphTimeFormat,
  ValidationIssue,
} from "../types/graph";
import {
  MAX_IMPORT_BYTES,
  MAX_PREVIEW_EDGES,
  MAX_PREVIEW_NODES,
} from "../types/graph";
import { computeGraphSummary, inferGraphDirectedness } from "./graphAlgorithms";
import {
  canonicalJson,
  compareUnicodeCodePoints,
  createOpaqueId,
  sha256Canonical,
  sha256Text,
} from "./graphIdentity";

const SOURCE_ALIASES = ["source", "src", "from", "source_id", "源节点", "起点"];
const TARGET_ALIASES = ["target", "dst", "to", "target_id", "目标节点", "终点"];
const TYPE_ALIASES = ["edge_type", "edgetype", "relation", "relationship", "type", "关系", "关系类型"];
const WEIGHT_ALIASES = ["weight", "value", "strength", "权重", "强度"];
const TIMESTAMP_ALIASES = ["timestamp", "time", "date", "datetime", "year", "时间", "日期", "年份", "年度"];
const SOURCE_LABEL_ALIASES = ["source_label", "src_label", "from_label", "源节点名称", "起点名称"];
const TARGET_LABEL_ALIASES = ["target_label", "dst_label", "to_label", "目标节点名称", "终点名称"];
const SOURCE_TYPE_ALIASES = ["source_type", "src_type", "from_type", "源节点类型", "起点类型"];
const TARGET_TYPE_ALIASES = ["target_type", "dst_type", "to_type", "目标节点类型", "终点类型"];
const NODE_ID_ALIASES = ["id", "node_id", "nodeid", "节点id", "节点编号"];
const NODE_LABEL_ALIASES = ["label", "name", "node_label", "节点名称", "名称"];
const NODE_TYPE_ALIASES = ["node_type", "nodetype", "category", "节点类型", "类别", "type"];

type CsvRow = Record<string, string | undefined>;

interface ParsedCsv {
  headers: string[];
  rows: CsvRow[];
  issues: ValidationIssue[];
}

const MAX_XML_ELEMENTS = 250_000;
const MAX_XML_NODES = 100_000;
const MAX_XML_EDGES = 300_000;

interface JsonGraphRoot {
  nodes: unknown[];
  edges: unknown[];
}

function normalizeHeader(header: string): string {
  return header
    .replace(/^\uFEFF/, "")
    .trim()
    .toLocaleLowerCase()
    .replace(/[\s_-]+/g, "");
}

function findAlias(headers: readonly string[], aliases: readonly string[]): string | undefined {
  const normalizedAliases = new Set(aliases.map(normalizeHeader));
  return headers.find((header) => normalizedAliases.has(normalizeHeader(header)));
}

function suggestMapping(headers: readonly string[]): Partial<ColumnMapping> {
  const source = findAlias(headers, SOURCE_ALIASES);
  const target = findAlias(headers, TARGET_ALIASES);
  const edgeType = findAlias(headers, TYPE_ALIASES);
  const weight = findAlias(headers, WEIGHT_ALIASES);
  const timestamp = findAlias(headers, TIMESTAMP_ALIASES);
  const sourceLabel = findAlias(headers, SOURCE_LABEL_ALIASES);
  const targetLabel = findAlias(headers, TARGET_LABEL_ALIASES);
  const sourceType = findAlias(headers, SOURCE_TYPE_ALIASES);
  const targetType = findAlias(headers, TARGET_TYPE_ALIASES);

  return {
    ...(source ? { source } : {}),
    ...(target ? { target } : {}),
    ...(edgeType ? { edgeType } : {}),
    ...(weight ? { weight } : {}),
    ...(timestamp ? { timestamp } : {}),
    ...(sourceLabel ? { sourceLabel } : {}),
    ...(targetLabel ? { targetLabel } : {}),
    ...(sourceType ? { sourceType } : {}),
    ...(targetType ? { targetType } : {}),
  };
}

function suggestNodeMapping(headers: readonly string[]): Partial<NodeColumnMapping> {
  const id = findAlias(headers, NODE_ID_ALIASES);
  const label = findAlias(headers, NODE_LABEL_ALIASES);
  const type = findAlias(headers, NODE_TYPE_ALIASES);
  return {
    ...(id ? { id } : {}),
    ...(label ? { label } : {}),
    ...(type ? { type } : {}),
  };
}

function detectFormat(file: File): ImportFormat {
  const extension = file.name.split(".").pop()?.toLocaleLowerCase();
  if (extension === "csv" || file.type === "text/csv") return "csv";
  if (extension === "tsv" || file.type === "text/tab-separated-values") return "tsv";
  if (extension === "json" || file.type === "application/json") return "json";
  if (extension === "graphml" || file.type === "application/graphml+xml") return "graphml";
  if (extension === "gexf" || file.type === "application/gexf+xml") return "gexf";
  if (extension === "npz" || file.type === "application/x-npz") return "npz";
  return "unsupported";
}

async function readFileText(file: File): Promise<string> {
  if (typeof file.text === "function") return file.text();

  if (typeof FileReader !== "undefined") {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result ?? ""));
      reader.onerror = () => reject(reader.error ?? new Error("无法读取本地文件。"));
      reader.readAsText(file);
    });
  }

  throw new Error("当前运行环境不支持本地文件读取。请使用现代浏览器。 ");
}

async function parseDelimited(file: File, delimiter?: "\t"): Promise<ParsedCsv> {
  const text = (await readFileText(file)).replace(/^\uFEFF/, "");
  const useWorker = typeof window !== "undefined" && typeof Worker !== "undefined";

  return new Promise((resolve, reject) => {
    const handleResult = (result: Papa.ParseResult<CsvRow>) => {
      const rawHeaders = result.meta.fields ?? [];
      const headers = rawHeaders.map((header) => header.replace(/^\uFEFF/, "").trim());
      const rows = result.data.map((row) =>
        Object.fromEntries(rawHeaders.map((header, index) => [headers[index], row[header]])),
      );
      const issues: ValidationIssue[] = result.errors.map((error) => ({
        code: "csv_parse_warning",
        severity: "warning",
        message: `CSV 解析提示：${error.message}`,
        ...(typeof error.row === "number" ? { row: error.row + 2 } : {}),
      }));
      resolve({ headers, rows, issues });
    };

    try {
      if (useWorker) {
        Papa.parse<CsvRow>(text, {
          header: true,
          skipEmptyLines: "greedy",
          ...(delimiter ? { delimiter } : {}),
          worker: true,
          complete: handleResult,
        });
      } else {
        handleResult(
          Papa.parse<CsvRow>(text, {
            header: true,
            skipEmptyLines: "greedy",
            ...(delimiter ? { delimiter } : {}),
            worker: false,
          }),
        );
      }
    } catch (error) {
      reject(error);
    }
  });
}

function inferColumnType(values: readonly string[]): FileColumnProfile["inferredType"] {
  const present = values.map((value) => value.trim()).filter(Boolean);
  if (!present.length) return "empty";
  if (present.every((value) => /^(?:true|false|0|1)$/iu.test(value))) return "boolean";
  if (present.every((value) => /^[-+]?\d+$/u.test(value))) return "integer";
  if (present.every((value) => /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/iu.test(value))) return "number";
  if (present.every((value) => /^\d{4}(?:-\d{2}(?:-\d{2})?)?(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/u.test(value))) {
    return "datetime";
  }
  return "string";
}

function profileColumns(parsed: ParsedCsv): FileColumnProfile[] {
  return parsed.headers.map((name) => {
    const values = parsed.rows.map((row) => row[name] ?? "");
    const missing = values.filter((value) => !value.trim()).length;
    return Object.freeze({
      name,
      inferredType: inferColumnType(values),
      missingRate: values.length === 0 ? 0 : Number((missing / values.length).toFixed(6)),
      cardinality: new Set(values.filter((value) => value.trim()).map((value) => value.trim())).size,
      nonNullCount: values.length - missing,
      nullCount: missing,
    });
  });
}

function detectTimeFormat(values: readonly string[]): Exclude<GraphTimeFormat, "none" | "auto"> | "invalid" | "mixed" {
  const formats = new Set<Exclude<GraphTimeFormat, "none" | "auto"> | "invalid">();
  for (const raw of values) {
    const value = raw.trim();
    if (!value) continue;
    if (/^\d{4}$/u.test(value)) formats.add("year");
    else if (/^\d{10}$/u.test(value)) formats.add("unix_seconds");
    else if (/^\d{13}$/u.test(value)) formats.add("unix_milliseconds");
    else if (/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,3})?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/u.test(value) && Number.isFinite(Date.parse(value))) {
      formats.add("iso8601");
    } else formats.add("invalid");
  }
  if (formats.size === 0) return "invalid";
  if (formats.size > 1) return "mixed";
  return [...formats][0];
}

function normalizeTimestamp(raw: string, format: Exclude<GraphTimeFormat, "none" | "auto">): string | undefined {
  const value = raw.trim();
  if (!value) return undefined;
  if (format === "year") return /^\d{4}$/u.test(value) ? `${value}-01-01T00:00:00.000Z` : undefined;
  if (format === "unix_seconds" && /^\d{10}$/u.test(value)) return new Date(Number(value) * 1_000).toISOString();
  if (format === "unix_milliseconds" && /^\d{13}$/u.test(value)) return new Date(Number(value)).toISOString();
  if (format === "iso8601") {
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? new Date(timestamp).toISOString() : undefined;
  }
  return undefined;
}

function normalizeGraphEdgeTimestamps(
  edges: readonly GraphEdge[],
  requested: GraphTimeFormat | undefined,
): { readonly edges?: readonly GraphEdge[]; readonly issue?: ValidationIssue } {
  if (!requested || requested === "none") return { edges };
  const values = edges.map((edge) => edge.timestamp ?? "").filter(Boolean);
  if (!values.length) return { edges };
  const detected = requested === "auto" ? detectTimeFormat(values) : requested;
  if (detected === "invalid" || detected === "mixed") {
    return {
      issue: {
        code: detected === "mixed" ? "ambiguous_time_format" : "invalid_time_format",
        severity: "error",
        message: detected === "mixed"
          ? "图文件包含多种时间格式，必须确认后再生成 GraphVersion。"
          : "图文件时间格式无法确定，必须确认后再生成 GraphVersion。",
      },
    };
  }
  const normalized: GraphEdge[] = [];
  for (const edge of edges) {
    if (!edge.timestamp) {
      normalized.push(edge);
      continue;
    }
    const timestamp = normalizeTimestamp(edge.timestamp, detected);
    if (!timestamp) {
      return {
        issue: {
          code: "invalid_timestamp",
          severity: "error",
          message: `时间“${edge.timestamp}”不符合已确认格式。`,
          entityId: edge.id,
        },
      };
    }
    normalized.push(Object.freeze({ ...edge, timestamp }));
  }
  return { edges: Object.freeze(normalized) };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function asId(value: unknown): string | undefined {
  if (typeof value !== "string" && typeof value !== "number") return undefined;
  const id = String(value).trim();
  return id.length > 0 ? id : undefined;
}

function asAttributeValue(value: unknown): GraphAttributeValue | undefined {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (Array.isArray(value)) {
    const primitives = value.filter(
      (item): item is string | number | boolean | null =>
        item === null || ["string", "number", "boolean"].includes(typeof item),
    );
    if (primitives.length === value.length) return primitives;
  }
  if (value === undefined) return undefined;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function collectAttributes(
  record: Record<string, unknown>,
  excluded: ReadonlySet<string>,
): GraphAttributes {
  const attributes: Record<string, GraphAttributeValue> = {};
  const nested = asRecord(record.attributes);
  if (nested) {
    for (const [key, value] of Object.entries(nested)) {
      const attribute = asAttributeValue(value);
      if (attribute !== undefined) attributes[key] = attribute;
    }
  }

  for (const [key, value] of Object.entries(record)) {
    if (excluded.has(key) || key === "attributes") continue;
    const attribute = asAttributeValue(value);
    if (attribute !== undefined && attribute !== "") attributes[key] = attribute;
  }

  return Object.freeze(attributes);
}

function collectEndpointAttributes(
  record: Record<string, unknown>,
  endpoint: "source" | "target",
  excluded: ReadonlySet<string>,
): GraphAttributes {
  const prefixes = endpoint === "source"
    ? /^(?:source|src|from|源节点|起点)[_.-](.+)$/iu
    : /^(?:target|dst|to|目标节点|终点)[_.-](.+)$/iu;
  const attributes: Record<string, GraphAttributeValue> = {};
  for (const [key, value] of Object.entries(record)) {
    if (excluded.has(key)) continue;
    const match = key.match(prefixes);
    if (!match?.[1] || /^(?:id|label|name|type)$/iu.test(match[1])) continue;
    const attribute = asAttributeValue(value);
    if (attribute !== undefined && attribute !== "") attributes[match[1]] = attribute;
  }
  return Object.freeze(attributes);
}

function makePreview(nodes: readonly GraphNode[], edges: readonly GraphEdge[]): GraphPreview {
  const selectedIds = new Set<string>();

  if (nodes.length <= MAX_PREVIEW_NODES) {
    for (const node of nodes) selectedIds.add(node.id);
  } else {
    // Seed from relationships so a large JSON file with isolated nodes first still yields a useful view.
    for (const edge of edges) {
      if (selectedIds.size < MAX_PREVIEW_NODES) selectedIds.add(edge.source);
      if (selectedIds.size < MAX_PREVIEW_NODES) selectedIds.add(edge.target);
      if (selectedIds.size >= MAX_PREVIEW_NODES) break;
    }
    for (const node of nodes) {
      if (selectedIds.size >= MAX_PREVIEW_NODES) break;
      selectedIds.add(node.id);
    }
  }

  const previewNodes = nodes.filter((node) => selectedIds.has(node.id)).slice(0, MAX_PREVIEW_NODES);
  const finalIds = new Set(previewNodes.map((node) => node.id));
  const previewEdges = edges
    .filter((edge) => finalIds.has(edge.source) && finalIds.has(edge.target))
    .slice(0, MAX_PREVIEW_EDGES);
  const truncated = previewNodes.length < nodes.length || previewEdges.length < edges.length;

  return Object.freeze({
    nodes: Object.freeze([...previewNodes]),
    edges: Object.freeze([...previewEdges]),
    truncated,
    originalNodeCount: nodes.length,
    originalEdgeCount: edges.length,
  });
}

function freezeAttributes(attributes: GraphAttributes): GraphAttributes {
  return Object.freeze(
    Object.fromEntries(
      Object.entries(attributes).map(([key, value]) => [
        key,
        Array.isArray(value) ? Object.freeze([...value]) : value,
      ]),
    ),
  );
}

function freezeNode(node: GraphNode): GraphNode {
  return Object.freeze({ ...node, attributes: freezeAttributes(node.attributes) });
}

function freezeEdge(edge: GraphEdge): GraphEdge {
  return Object.freeze({ ...edge, attributes: freezeAttributes(edge.attributes) });
}

function freezeIssue(issue: ValidationIssue): ValidationIssue {
  return Object.freeze({
    ...issue,
    ...(issue.details ? { details: Object.freeze({ ...issue.details }) } : {}),
  });
}

function freezeBuildSpec(spec: GraphBuildSpec): GraphBuildSpec {
  return Object.freeze({
    ...spec,
    sourceArtifactIds: Object.freeze([...spec.sourceArtifactIds]),
    ...(spec.nodeMapping ? { nodeMapping: Object.freeze({ ...spec.nodeMapping }) } : {}),
    ...(spec.edgeMapping ? { edgeMapping: Object.freeze({ ...spec.edgeMapping }) } : {}),
  });
}

export function createGraphVersion(
  sourceFile: string,
  inputNodes: readonly GraphNode[],
  inputEdges: readonly GraphEdge[],
  inputIssues: readonly ValidationIssue[] = [],
  options: GraphImportParseOptions = {},
): GraphVersion {
  const nodes = Object.freeze(inputNodes.map(freezeNode));
  const edges = Object.freeze(inputEdges.map(freezeEdge));
  const preview = makePreview(nodes, edges);
  const issues = [...inputIssues];
  if (
    nodes.length > 0 &&
    nodes.every((node) => !node.type?.trim()) &&
    !issues.some((issue) => issue.code === "all_nodes_unclassified")
  ) {
    issues.push({
      code: "all_nodes_unclassified",
      severity: "warning",
      message: "所有节点都未识别实体类型；图谱将以“未分类”显示，后续异构图任务前请确认节点类型字段。",
      details: { unclassifiedNodeCount: nodes.length },
    });
  }
  if (preview.truncated) {
    issues.push({
      code: "preview_truncated",
      severity: "info",
      message: `图谱较大，画布仅展示最多 ${MAX_PREVIEW_NODES} 个节点和 ${MAX_PREVIEW_EDGES} 条关系；统计仍基于完整图。`,
    });
  }

  const canonicalGraph = {
    nodes: [...nodes].sort((left, right) => compareUnicodeCodePoints(left.id, right.id)).map((node) => ({
      id: node.id,
      label: node.label,
      type: node.type,
      attributes: node.attributes,
    })),
    edges: [...edges].sort((left, right) => compareUnicodeCodePoints(left.id, right.id)).map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: edge.type,
      weight: edge.weight,
      timestamp: edge.timestamp,
      directed: edge.directed,
      attributes: edge.attributes,
    })),
  };
  const contentHash = sha256Canonical(canonicalGraph);
  const sourceArtifactIds = options.sourceArtifacts?.map((artifact) => artifact.id)
    ?? options.buildSpec?.sourceArtifactIds;
  const sourceHash = options.sourceArtifacts?.length
    ? sha256Text(options.sourceArtifacts.map((artifact) => artifact.sha256).sort(compareUnicodeCodePoints).join("\n"))
    : sha256Text(sourceFile);
  const buildSpecHash = options.buildSpec ? sha256Text(canonicalJson(options.buildSpec)) : undefined;

  return Object.freeze({
    id: createOpaqueId("graph"),
    sourceFile,
    createdAt: new Date().toISOString(),
    nodes,
    edges,
    summary: Object.freeze(computeGraphSummary(nodes, edges)),
    issues: Object.freeze(issues.map(freezeIssue)),
    preview,
    truncated: preview.truncated,
    ...(sourceArtifactIds?.length ? { sourceArtifactIds: Object.freeze([...sourceArtifactIds]) } : {}),
    sourceHash,
    ...(buildSpecHash ? { buildSpecHash } : {}),
    contentHash,
    ...(options.parentVersionId ? { parentVersionId: options.parentVersionId } : {}),
    ...(options.buildSpec ? { buildSpec: freezeBuildSpec(options.buildSpec) } : {}),
    ...(options.provenance ? { provenance: Object.freeze({ ...options.provenance }) } : {}),
    metadata: Object.freeze({ directedness: inferGraphDirectedness({ edges }) }),
  });
}

function failed(error: string, issues: readonly ValidationIssue[] = []): ImportRun {
  return {
    status: "failed",
    issues,
    error,
  };
}

function validateMapping(
  headers: readonly string[],
  mapping: ColumnMapping,
): { mapping?: ColumnMapping; issues: ValidationIssue[] } {
  const issues: ValidationIssue[] = [];
  const resolve = (requested: string | undefined): string | undefined => {
    if (!requested) return undefined;
    const exact = headers.find((header) => header === requested);
    if (exact) return exact;
    const normalizedMatches = headers.filter(
      (header) => normalizeHeader(header) === normalizeHeader(requested),
    );
    return normalizedMatches.length === 1 ? normalizedMatches[0] : undefined;
  };
  const source = resolve(mapping.source);
  const target = resolve(mapping.target);

  if (!source || !target) {
    issues.push({
      code: "invalid_column_mapping",
      severity: "error",
      message: "字段映射无效：请选择文件中存在的起点列和终点列。",
    });
    return { issues };
  }
  if (source === target) {
    issues.push({
      code: "same_endpoint_column",
      severity: "error",
      message: "起点列和终点列不能是同一列。",
    });
    return { issues };
  }

  const optionalFields = [
    ["sourceLabel", mapping.sourceLabel],
    ["targetLabel", mapping.targetLabel],
    ["sourceType", mapping.sourceType],
    ["targetType", mapping.targetType],
    ["edgeType", mapping.edgeType],
    ["weight", mapping.weight],
    ["timestamp", mapping.timestamp],
  ] as const;
  const invalidOptionalFields = optionalFields
    .filter(([, requested]) => Boolean(requested) && !resolve(requested))
    .map(([field]) => field);
  if (invalidOptionalFields.length) {
    issues.push({
      code: "invalid_optional_column_mapping",
      severity: "error",
      message: `字段映射无效：${invalidOptionalFields.join("、")} 引用了不存在或有歧义的列。`,
    });
    return { issues };
  }

  return {
    mapping: {
      source,
      target,
      ...(resolve(mapping.sourceLabel) ? { sourceLabel: resolve(mapping.sourceLabel)! } : {}),
      ...(resolve(mapping.targetLabel) ? { targetLabel: resolve(mapping.targetLabel)! } : {}),
      ...(resolve(mapping.sourceType) ? { sourceType: resolve(mapping.sourceType)! } : {}),
      ...(resolve(mapping.targetType) ? { targetType: resolve(mapping.targetType)! } : {}),
      ...(resolve(mapping.edgeType) ? { edgeType: resolve(mapping.edgeType)! } : {}),
      ...(resolve(mapping.weight) ? { weight: resolve(mapping.weight)! } : {}),
      ...(resolve(mapping.timestamp) ? { timestamp: resolve(mapping.timestamp)! } : {}),
    },
    issues,
  };
}

function validateNodeMapping(
  headers: readonly string[],
  mapping: NodeColumnMapping,
): { mapping?: NodeColumnMapping; issues: ValidationIssue[] } {
  const resolve = (requested: string | undefined): string | undefined => {
    if (!requested) return undefined;
    const exact = headers.find((header) => header === requested);
    if (exact) return exact;
    const normalizedMatches = headers.filter(
      (header) => normalizeHeader(header) === normalizeHeader(requested),
    );
    return normalizedMatches.length === 1 ? normalizedMatches[0] : undefined;
  };
  const id = resolve(mapping.id);
  if (!id) {
    return {
      issues: [{
        code: "node_id_column_missing",
        severity: "error",
        message: "请为节点表指定唯一且无歧义的 ID 列。",
      }],
    };
  }
  const label = resolve(mapping.label);
  const type = resolve(mapping.type);
  const invalid = [
    ...(mapping.label && !label ? ["label"] : []),
    ...(mapping.type && !type ? ["type"] : []),
  ];
  if (invalid.length) {
    return {
      issues: [{
        code: "invalid_optional_node_column_mapping",
        severity: "error",
        message: `节点字段映射无效：${invalid.join("、")} 引用了不存在或有歧义的列。`,
      }],
    };
  }
  return {
    mapping: {
      id,
      ...(label ? { label } : {}),
      ...(type ? { type } : {}),
    },
    issues: [],
  };
}

function csvToGraph(
  sourceFile: string,
  parsed: ParsedCsv,
  requestedMapping?: ColumnMapping,
  options: GraphImportParseOptions = {},
): ImportRun {
  const suggestion = suggestMapping(parsed.headers);
  const mappingCandidate = requestedMapping ??
    (suggestion.source && suggestion.target
      ? ({ ...suggestion, source: suggestion.source, target: suggestion.target } as ColumnMapping)
      : undefined);

  if (!mappingCandidate) {
    return {
      status: "needs_mapping",
      headers: parsed.headers,
      suggestedMapping: suggestion,
      issues: [
        ...parsed.issues,
        {
          code: "endpoint_columns_missing",
          severity: "warning",
          message: "未识别到起点列或终点列，请手动指定字段映射。",
        },
      ],
    };
  }

  const validated = validateMapping(parsed.headers, mappingCandidate);
  if (!validated.mapping) {
    return {
      status: "needs_mapping",
      headers: parsed.headers,
      suggestedMapping: suggestion,
      issues: [...parsed.issues, ...validated.issues],
    };
  }

  const mapping = validated.mapping;
  const nodesById = new Map<string, GraphNode>();
  const edges: GraphEdge[] = [];
  const issues = [...parsed.issues];
  const excluded = new Set(
    [
      mapping.source,
      mapping.target,
      mapping.sourceLabel,
      mapping.targetLabel,
      mapping.sourceType,
      mapping.targetType,
      mapping.edgeType,
      mapping.weight,
      mapping.timestamp,
    ].filter(
      (header): header is string => Boolean(header),
    ),
  );
  const endpointReserved = new Set(excluded);
  for (const header of parsed.headers) {
    if (/^(?:source|src|from|target|dst|to|源节点|起点|目标节点|终点)[_.-]/iu.test(header)) excluded.add(header);
  }

  let timeFormat: Exclude<GraphTimeFormat, "none" | "auto"> | undefined;
  if (mapping.timestamp) {
    const requested = options.buildSpec?.timeFormat ?? "auto";
    const detected = requested === "auto"
      ? detectTimeFormat(parsed.rows.map((row) => row[mapping.timestamp!] ?? ""))
      : requested === "none"
        ? "invalid"
        : requested;
    if (detected === "invalid" || detected === "mixed") {
      const issue: ValidationIssue = {
        code: detected === "mixed" ? "ambiguous_time_format" : "invalid_time_format",
        severity: "error",
        message: detected === "mixed"
          ? "时间列包含多种格式。请明确选择时间格式后再构图。"
          : "无法确定时间列格式。请明确选择 ISO 8601、年份或 Unix 时间戳。",
      };
      return failed(issue.message, [...issues, issue]);
    }
    timeFormat = detected;
  }

  const upsertNode = (
    id: string,
    label: string | undefined,
    type: string | undefined,
    attributes: GraphAttributes,
    rowNumber: number,
  ) => {
    const existing = nodesById.get(id);
    if (!existing) {
      nodesById.set(id, Object.freeze({
        id,
        label: label ?? id,
        ...(type ? { type } : {}),
        attributes,
      }));
      return;
    }
    const canBackfillLabel = Boolean(label && existing.label === id);
    const canBackfillType = Boolean(type && !existing.type);
    if (
      (label && existing.label !== id && existing.label !== label) ||
      (type && existing.type && existing.type !== type)
    ) {
      issues.push({
        code: "conflicting_node_metadata",
        severity: "warning",
        message: `节点“${id}”在不同关系行中的名称或类型不一致，已保留首次出现的值。`,
        row: rowNumber,
        entityId: id,
      });
    }
    const mergedAttributes: Record<string, GraphAttributeValue> = { ...existing.attributes };
    let changed = canBackfillLabel || canBackfillType;
    for (const [key, value] of Object.entries(attributes)) {
      if (!(key in mergedAttributes)) {
        mergedAttributes[key] = value;
        changed = true;
      } else if (canonicalJson(mergedAttributes[key]) !== canonicalJson(value)) {
        issues.push({
          code: "conflicting_node_attribute",
          severity: "warning",
          message: `节点“${id}”的属性“${key}”在不同关系行不一致，已保留首次出现的值。`,
          row: rowNumber,
          entityId: id,
        });
      }
    }
    if (changed) {
      nodesById.set(id, Object.freeze({
        ...existing,
        ...(canBackfillLabel && label ? { label } : {}),
        ...(canBackfillType && type ? { type } : {}),
        attributes: Object.freeze(mergedAttributes),
      }));
    }
  };
  const duplicateKeys = new Set<string>();
  let hasRejectedRelationships = false;

  parsed.rows.forEach((row, index) => {
    const source = asId(row[mapping.source]);
    const target = asId(row[mapping.target]);
    const rowNumber = index + 2;
    if (!source || !target) {
      issues.push({
        code: "empty_endpoint",
        severity: "warning",
        message: "该行缺少起点或终点，已从图谱中排除。",
        row: rowNumber,
      });
      return;
    }

    const sourceLabel = mapping.sourceLabel ? asId(row[mapping.sourceLabel]) : undefined;
    const targetLabel = mapping.targetLabel ? asId(row[mapping.targetLabel]) : undefined;
    const sourceType = mapping.sourceType ? asId(row[mapping.sourceType]) : undefined;
    const targetType = mapping.targetType ? asId(row[mapping.targetType]) : undefined;
    upsertNode(source, sourceLabel, sourceType, collectEndpointAttributes(row, "source", endpointReserved), rowNumber);
    upsertNode(target, targetLabel, targetType, collectEndpointAttributes(row, "target", endpointReserved), rowNumber);

    let weight: number | undefined;
    const rawWeight = mapping.weight ? row[mapping.weight] : undefined;
    if (rawWeight?.trim()) {
      const parsedWeight = Number(rawWeight);
      if (Number.isFinite(parsedWeight)) {
        weight = parsedWeight;
      } else {
        issues.push({
          code: "invalid_weight",
          severity: "warning",
          message: `权重“${rawWeight}”不是有效数字，已忽略。`,
          row: rowNumber,
        });
      }
    }

    const edgeType = mapping.edgeType ? asId(row[mapping.edgeType]) : undefined;
    const rawTimestamp = mapping.timestamp ? asId(row[mapping.timestamp]) : undefined;
    const timestamp = rawTimestamp && timeFormat ? normalizeTimestamp(rawTimestamp, timeFormat) : undefined;
    if (rawTimestamp && timeFormat && !timestamp) {
      issues.push({
        code: "invalid_timestamp",
        severity: "error",
        message: `时间“${rawTimestamp}”不符合已确认格式。`,
        row: rowNumber,
      });
      hasRejectedRelationships = true;
      return;
    }
    const directionPolicy = options.buildSpec?.directionPolicy ?? "undirected";
    const directed = directionPolicy === "directed";
    if (source === target && options.buildSpec?.selfLoopPolicy === "reject") {
      issues.push({ code: "self_loop_rejected", severity: "error", message: "构图规则不允许自环。", row: rowNumber });
      hasRejectedRelationships = true;
      return;
    }
    const duplicateKey = directed || source <= target
      ? `${source}\u001f${target}\u001f${edgeType ?? ""}`
      : `${target}\u001f${source}\u001f${edgeType ?? ""}`;
    if (duplicateKeys.has(duplicateKey) && options.buildSpec?.duplicateEdgePolicy === "reject") {
      issues.push({ code: "duplicate_edge_rejected", severity: "error", message: "构图规则不允许重复关系。", row: rowNumber });
      hasRejectedRelationships = true;
      return;
    }
    duplicateKeys.add(duplicateKey);
    edges.push(
      Object.freeze({
        id: `edge-${index + 1}`,
        source,
        target,
        ...(edgeType ? { type: edgeType } : {}),
        ...(weight !== undefined ? { weight } : {}),
        ...(timestamp ? { timestamp } : {}),
        directed,
        attributes: collectAttributes(row, excluded),
      }),
    );
  });

  if (hasRejectedRelationships) {
    return failed("部分关系不符合已确认的构图规则；未生成 GraphVersion。", issues);
  }

  let finalEdges = edges;
  if (options.buildSpec?.duplicateEdgePolicy === "merge_sum") {
    const merged = new Map<string, GraphEdge>();
    for (const edge of edges) {
      const key = edge.directed || edge.source <= edge.target
        ? `${edge.source}\u001f${edge.target}\u001f${edge.type ?? ""}`
        : `${edge.target}\u001f${edge.source}\u001f${edge.type ?? ""}`;
      const existing = merged.get(key);
      if (!existing) merged.set(key, edge);
      else merged.set(key, Object.freeze({
        ...existing,
        weight: (existing.weight ?? 1) + (edge.weight ?? 1),
        attributes: Object.freeze({ ...existing.attributes, mergedRelationshipCount: Number(existing.attributes.mergedRelationshipCount ?? 1) + 1 }),
      }));
    }
    finalEdges = [...merged.values()];
    if (finalEdges.length < edges.length) {
      issues.push({
        code: "duplicate_edges_merged",
        severity: "info",
        message: `按规则合并了 ${edges.length - finalEdges.length} 条重复关系，原始源文件仍保存在 SourceArtifact 中。`,
      });
    }
  }

  if (nodesById.size === 0 || finalEdges.length === 0) {
    return failed("文件中没有可构图的有效关系。", [
      ...issues,
      {
        code: "no_valid_edges",
        severity: "error",
        message: "未找到同时具有起点和终点的有效记录。",
      },
    ]);
  }

  const graphVersion = createGraphVersion(sourceFile, [...nodesById.values()], finalEdges, issues, options);
  return {
    status: "ready",
    headers: parsed.headers,
    suggestedMapping: suggestion,
    graphVersion,
    issues: graphVersion.issues,
  };
}

function assertJsonRoot(value: unknown): JsonGraphRoot {
  const root = asRecord(value);
  if (!root || !Array.isArray(root.nodes) || !Array.isArray(root.edges)) {
    throw new Error('JSON 必须使用固定结构 { "nodes": [...], "edges": [...] }。');
  }
  return { nodes: root.nodes, edges: root.edges };
}

function jsonToGraph(
  sourceFile: string,
  json: unknown,
  options: GraphImportParseOptions = {},
): ImportRun {
  let root: JsonGraphRoot;
  try {
    root = assertJsonRoot(json);
  } catch (error) {
    return failed(error instanceof Error ? error.message : "JSON 结构无效。", [
      {
        code: "invalid_json_shape",
        severity: "error",
        message: 'JSON 必须包含数组字段 "nodes" 与 "edges"。',
      },
    ]);
  }

  const issues: ValidationIssue[] = [];
  const nodesById = new Map<string, GraphNode>();

  root.nodes.forEach((rawNode, index) => {
    const record = asRecord(rawNode);
    const id = record ? asId(record.id) : undefined;
    if (!record || !id) {
      issues.push({
        code: "invalid_node",
        severity: "warning",
        message: "节点缺少有效 id，已排除。",
        row: index + 1,
      });
      return;
    }
    if (nodesById.has(id)) {
      issues.push({
        code: "duplicate_node",
        severity: "warning",
        message: `节点 id“${id}”重复，已保留第一次出现的记录。`,
        row: index + 1,
        entityId: id,
      });
      return;
    }

    const label = asId(record.label) ?? asId(record.name) ?? id;
    const type = asId(record.type);
    nodesById.set(
      id,
      Object.freeze({
        id,
        label,
        ...(type ? { type } : {}),
        attributes: collectAttributes(record, new Set(["id", "label", "name", "type"])),
      }),
    );
  });

  if (nodesById.size === 0) {
    return failed("JSON 中没有具有有效 id 的节点。", [
      ...issues,
      { code: "no_valid_nodes", severity: "error", message: "没有可用于构图的有效节点。" },
    ]);
  }

  const edges: GraphEdge[] = [];
  const edgeIds = new Set<string>();
  root.edges.forEach((rawEdge, index) => {
    const record = asRecord(rawEdge);
    const source = record ? asId(record.source) : undefined;
    const target = record ? asId(record.target) : undefined;
    if (!record || !source || !target) {
      issues.push({
        code: "invalid_edge",
        severity: "warning",
        message: "关系缺少有效 source 或 target，已排除。",
        row: index + 1,
      });
      return;
    }
    if (!nodesById.has(source) || !nodesById.has(target)) {
      issues.push({
        code: "dangling_edge",
        severity: "warning",
        message: `关系 ${source} → ${target} 引用了不存在的节点，已排除。`,
        row: index + 1,
        details: { source, target },
      });
      return;
    }

    const requestedId = asId(record.id) ?? `edge-${index + 1}`;
    let id = requestedId;
    let suffix = 2;
    while (edgeIds.has(id)) {
      id = `${requestedId}#${suffix}`;
      suffix += 1;
    }
    if (id !== requestedId) {
      issues.push({
        code: "duplicate_edge_id",
        severity: "warning",
        message: `关系 id“${requestedId}”重复，已重命名为“${id}”。`,
        row: index + 1,
      });
    }
    edgeIds.add(id);

    const type = asId(record.type) ?? asId(record.edge_type);
    const timestamp = asId(record.timestamp);
    const parsedWeight = typeof record.weight === "number" ? record.weight : Number(record.weight);
    const weight = Number.isFinite(parsedWeight) ? parsedWeight : undefined;
    if (record.weight !== undefined && weight === undefined) {
      issues.push({
        code: "invalid_weight",
        severity: "warning",
        message: `关系“${id}”的 weight 不是有效数字，已忽略。`,
        row: index + 1,
      });
    }

    edges.push(
      Object.freeze({
        id,
        source,
        target,
        ...(type ? { type } : {}),
        ...(weight !== undefined ? { weight } : {}),
        ...(timestamp ? { timestamp } : {}),
        ...(typeof record.directed === "boolean" ? { directed: record.directed } : {}),
        attributes: collectAttributes(
          record,
          new Set(["id", "source", "target", "type", "edge_type", "weight", "timestamp", "directed"]),
        ),
      }),
    );
  });

  const temporal = normalizeGraphEdgeTimestamps(edges, options.buildSpec?.timeFormat);
  if (!temporal.edges || temporal.issue) {
    const issue = temporal.issue ?? { code: "invalid_time_format", severity: "error" as const, message: "时间规范化失败。" };
    return failed(issue.message, [...issues, issue]);
  }
  const graphVersion = createGraphVersion(sourceFile, [...nodesById.values()], temporal.edges, issues, options);
  return {
    status: "ready",
    graphVersion,
    issues: graphVersion.issues,
  };
}

interface XmlAttributeDefinition {
  readonly name: string;
  readonly type?: string;
  readonly scope?: string;
}

function elementsByLocalName(root: Document | Element, localName: string): Element[] {
  return Array.from(root.getElementsByTagNameNS("*", localName));
}

function directChildrenByLocalName(element: Element, localName: string): Element[] {
  return Array.from(element.children).filter((child) => child.localName === localName);
}

function parseSafeXml(text: string, format: "graphml" | "gexf"): XMLDocument {
  if (/<!\s*(?:DOCTYPE|ENTITY)\b/i.test(text)) {
    throw new Error("XML 包含被禁用的 DOCTYPE 或 ENTITY 声明。请移除外部实体后重试。");
  }
  if (typeof DOMParser === "undefined") {
    throw new Error("当前运行环境不支持安全的 XML 解析。");
  }

  const document = new DOMParser().parseFromString(text, "application/xml");
  const parserError = elementsByLocalName(document, "parsererror")[0];
  if (parserError || document.documentElement.localName === "parsererror") {
    throw new Error(`XML 语法无效：${parserError?.textContent?.trim() || "无法解析文档"}`);
  }

  const expectedRoot = format === "graphml" ? "graphml" : "gexf";
  if (document.documentElement.localName.toLocaleLowerCase() !== expectedRoot) {
    throw new Error(`文件扩展名与 XML 根元素不匹配：期望 <${expectedRoot}>。`);
  }

  const elementCount = document.getElementsByTagName("*").length;
  if (elementCount > MAX_XML_ELEMENTS) {
    throw new Error(`XML 元素数量超过前端安全上限 ${MAX_XML_ELEMENTS.toLocaleString()}。`);
  }
  return document;
}

function parseXmlAttributeValue(
  rawValue: string,
  type: string | undefined,
  issues: ValidationIssue[],
  entityId: string,
  name: string,
): GraphAttributeValue {
  const normalizedType = type?.trim().toLocaleLowerCase();
  if (["boolean", "bool"].includes(normalizedType ?? "")) {
    if (/^(?:true|1)$/i.test(rawValue)) return true;
    if (/^(?:false|0)$/i.test(rawValue)) return false;
  } else if (["byte", "short", "int", "integer", "long", "float", "double"].includes(normalizedType ?? "")) {
    const number = Number(rawValue);
    if (Number.isFinite(number)) return number;
  } else {
    return rawValue;
  }

  issues.push({
    code: "invalid_xml_attribute_value",
    severity: "warning",
    message: `实体“${entityId}”的属性“${name}”不符合声明类型 ${type}，已按文本保留。`,
    entityId,
  });
  return rawValue;
}

function graphMlDefinitions(document: XMLDocument): Map<string, XmlAttributeDefinition> {
  const definitions = new Map<string, XmlAttributeDefinition>();
  for (const key of elementsByLocalName(document, "key")) {
    const id = asId(key.getAttribute("id"));
    if (!id || definitions.has(id)) continue;
    definitions.set(id, {
      name: asId(key.getAttribute("attr.name")) ?? id,
      ...(asId(key.getAttribute("attr.type")) ? { type: asId(key.getAttribute("attr.type"))! } : {}),
      ...(asId(key.getAttribute("for")) ? { scope: asId(key.getAttribute("for"))! } : {}),
    });
  }
  return definitions;
}

function readGraphMlAttributes(
  element: Element,
  definitions: ReadonlyMap<string, XmlAttributeDefinition>,
  issues: ValidationIssue[],
  entityId: string,
  unknownKeys: Set<string>,
): Record<string, GraphAttributeValue> {
  const attributes: Record<string, GraphAttributeValue> = Object.create(null) as Record<string, GraphAttributeValue>;
  for (const data of directChildrenByLocalName(element, "data")) {
    const keyId = asId(data.getAttribute("key"));
    if (!keyId) continue;
    const definition = definitions.get(keyId);
    if (!definition && !unknownKeys.has(keyId)) {
      unknownKeys.add(keyId);
      issues.push({
        code: "unknown_xml_attribute_key",
        severity: "warning",
        message: `XML 属性键“${keyId}”没有对应定义，已使用键 ID 作为属性名。`,
      });
    }
    const name = definition?.name ?? keyId;
    attributes[name] = parseXmlAttributeValue(
      data.textContent?.trim() ?? "",
      definition?.type,
      issues,
      entityId,
      name,
    );
  }
  return attributes;
}

function gexfDefinitions(
  document: XMLDocument,
  scope: "node" | "edge",
): Map<string, XmlAttributeDefinition> {
  const definitions = new Map<string, XmlAttributeDefinition>();
  for (const container of elementsByLocalName(document, "attributes")) {
    if ((container.getAttribute("class") ?? "node").toLocaleLowerCase() !== scope) continue;
    for (const attribute of directChildrenByLocalName(container, "attribute")) {
      const id = asId(attribute.getAttribute("id"));
      if (!id || definitions.has(id)) continue;
      definitions.set(id, {
        name: asId(attribute.getAttribute("title")) ?? id,
        ...(asId(attribute.getAttribute("type")) ? { type: asId(attribute.getAttribute("type"))! } : {}),
        scope,
      });
    }
  }
  return definitions;
}

function readGexfAttributes(
  element: Element,
  definitions: ReadonlyMap<string, XmlAttributeDefinition>,
  issues: ValidationIssue[],
  entityId: string,
  unknownKeys: Set<string>,
): Record<string, GraphAttributeValue> {
  const attributes: Record<string, GraphAttributeValue> = Object.create(null) as Record<string, GraphAttributeValue>;
  const containers = directChildrenByLocalName(element, "attvalues");
  for (const container of containers) {
    for (const valueElement of directChildrenByLocalName(container, "attvalue")) {
      const keyId = asId(valueElement.getAttribute("for")) ?? asId(valueElement.getAttribute("id"));
      if (!keyId) continue;
      const definition = definitions.get(keyId);
      if (!definition && !unknownKeys.has(keyId)) {
        unknownKeys.add(keyId);
        issues.push({
          code: "unknown_xml_attribute_key",
          severity: "warning",
          message: `GEXF 属性键“${keyId}”没有对应定义，已使用键 ID 作为属性名。`,
        });
      }
      const name = definition?.name ?? keyId;
      attributes[name] = parseXmlAttributeValue(
        valueElement.getAttribute("value") ?? valueElement.textContent?.trim() ?? "",
        definition?.type,
        issues,
        entityId,
        name,
      );
    }
  }
  return attributes;
}

function readAttributeId(
  attributes: Readonly<Record<string, GraphAttributeValue>>,
  names: readonly string[],
): string | undefined {
  for (const name of names) {
    const exact = Object.keys(attributes).find((key) => normalizeHeader(key) === normalizeHeader(name));
    if (exact) {
      const value = asId(attributes[exact]);
      if (value) return value;
    }
  }
  return undefined;
}

function readAttributeNumber(
  attributes: Readonly<Record<string, GraphAttributeValue>>,
  names: readonly string[],
): number | undefined {
  const value = readAttributeId(attributes, names);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function omitXmlAttributes(
  attributes: Readonly<Record<string, GraphAttributeValue>>,
  omittedNames: readonly string[],
): GraphAttributes {
  const normalizedOmitted = new Set(omittedNames.map(normalizeHeader));
  return Object.freeze(Object.fromEntries(
    Object.entries(attributes).filter(([key]) => !normalizedOmitted.has(normalizeHeader(key))),
  ));
}

function uniqueXmlEdgeId(
  requestedId: string,
  edgeIds: Set<string>,
  issues: ValidationIssue[],
): string {
  let id = requestedId;
  let suffix = 2;
  while (edgeIds.has(id)) {
    id = `${requestedId}#${suffix}`;
    suffix += 1;
  }
  if (id !== requestedId) {
    issues.push({
      code: "duplicate_edge_id",
      severity: "warning",
      message: `关系 id“${requestedId}”重复，已重命名为“${id}”。`,
    });
  }
  edgeIds.add(id);
  return id;
}

function xmlToGraph(
  sourceFile: string,
  text: string,
  format: "graphml" | "gexf",
  options: GraphImportParseOptions = {},
): ImportRun {
  let document: XMLDocument;
  try {
    document = parseSafeXml(text, format);
  } catch (error) {
    const message = error instanceof Error ? error.message : "XML 无法解析。";
    const unsafe = /DOCTYPE|ENTITY/.test(message);
    return failed(message, [{
      code: unsafe ? "unsafe_xml_construct" : "invalid_xml",
      severity: "error",
      message,
    }]);
  }

  const issues: ValidationIssue[] = [];
  const nodeElements = elementsByLocalName(document, "node");
  const edgeElements = elementsByLocalName(document, "edge");
  if (nodeElements.length > MAX_XML_NODES || edgeElements.length > MAX_XML_EDGES) {
    const message = `XML 图规模超过前端安全上限（节点 ${MAX_XML_NODES.toLocaleString()}，关系 ${MAX_XML_EDGES.toLocaleString()}）。`;
    return failed(message, [{
      code: "xml_graph_limit_exceeded",
      severity: "error",
      message,
      details: { nodeCount: nodeElements.length, edgeCount: edgeElements.length },
    }]);
  }

  const graphElement = elementsByLocalName(document, "graph")[0];
  const graphMlKeys = format === "graphml" ? graphMlDefinitions(document) : new Map<string, XmlAttributeDefinition>();
  const gexfNodeKeys = format === "gexf" ? gexfDefinitions(document, "node") : new Map<string, XmlAttributeDefinition>();
  const gexfEdgeKeys = format === "gexf" ? gexfDefinitions(document, "edge") : new Map<string, XmlAttributeDefinition>();
  const unknownNodeKeys = new Set<string>();
  const unknownEdgeKeys = new Set<string>();
  const nodesById = new Map<string, GraphNode>();

  nodeElements.forEach((element, index) => {
    const id = asId(element.getAttribute("id"));
    if (!id) {
      issues.push({ code: "invalid_node", severity: "warning", message: "XML 节点缺少有效 id，已排除。", row: index + 1 });
      return;
    }
    if (nodesById.has(id)) {
      issues.push({
        code: "duplicate_node",
        severity: "warning",
        message: `节点 id“${id}”重复，已保留第一次出现的记录。`,
        row: index + 1,
        entityId: id,
      });
      return;
    }
    const attributes = format === "graphml"
      ? readGraphMlAttributes(element, graphMlKeys, issues, id, unknownNodeKeys)
      : readGexfAttributes(element, gexfNodeKeys, issues, id, unknownNodeKeys);
    const label = asId(element.getAttribute("label")) ?? readAttributeId(attributes, ["label", "name", "title"]) ?? id;
    const type = readAttributeId(attributes, ["type", "node_type", "category"]);
    nodesById.set(id, Object.freeze({
      id,
      label,
      ...(type ? { type } : {}),
      attributes: omitXmlAttributes(attributes, ["label", "name", "title", "type", "node_type", "category"]),
    }));
  });

  if (nodesById.size === 0) {
    return failed("XML 中没有具有有效 id 的节点。", [
      ...issues,
      { code: "no_valid_nodes", severity: "error", message: "没有可用于构图的有效节点。" },
    ]);
  }

  const graphDefault = format === "graphml"
    ? graphElement?.getAttribute("edgedefault")
    : graphElement?.getAttribute("defaultedgetype");
  const defaultDirected = graphDefault?.toLocaleLowerCase() === "directed";
  const edges: GraphEdge[] = [];
  const edgeIds = new Set<string>();
  edgeElements.forEach((element, index) => {
    const source = asId(element.getAttribute("source"));
    const target = asId(element.getAttribute("target"));
    if (!source || !target) {
      issues.push({ code: "invalid_edge", severity: "warning", message: "XML 关系缺少有效 source 或 target，已排除。", row: index + 1 });
      return;
    }
    if (!nodesById.has(source) || !nodesById.has(target)) {
      issues.push({
        code: "dangling_edge",
        severity: "warning",
        message: `关系 ${source} → ${target} 引用了不存在的节点，已排除。`,
        row: index + 1,
        details: { source, target },
      });
      return;
    }

    const requestedId = asId(element.getAttribute("id")) ?? `edge-${index + 1}`;
    const id = uniqueXmlEdgeId(requestedId, edgeIds, issues);
    const attributes = format === "graphml"
      ? readGraphMlAttributes(element, graphMlKeys, issues, id, unknownEdgeKeys)
      : readGexfAttributes(element, gexfEdgeKeys, issues, id, unknownEdgeKeys);
    const type = asId(element.getAttribute("label")) ?? readAttributeId(attributes, ["type", "edge_type", "relation", "label"]);
    const directWeight = element.getAttribute("weight");
    const parsedDirectWeight = directWeight === null ? undefined : Number(directWeight);
    const weight = parsedDirectWeight !== undefined && Number.isFinite(parsedDirectWeight)
      ? parsedDirectWeight
      : readAttributeNumber(attributes, ["weight", "value", "strength"]);
    if (directWeight !== null && weight === undefined) {
      issues.push({ code: "invalid_weight", severity: "warning", message: `关系“${id}”的 weight 不是有效数字，已忽略。`, entityId: id });
    }
    const timestamp = asId(element.getAttribute("timestamp"))
      ?? asId(element.getAttribute("start"))
      ?? readAttributeId(attributes, ["timestamp", "time", "date"]);
    const directedAttribute = element.getAttribute("directed") ?? element.getAttribute("type");
    const directed = directedAttribute
      ? ["true", "1", "directed", "mutual"].includes(directedAttribute.toLocaleLowerCase())
      : defaultDirected;
    edges.push(Object.freeze({
      id,
      source,
      target,
      ...(type ? { type } : {}),
      ...(weight !== undefined ? { weight } : {}),
      ...(timestamp ? { timestamp } : {}),
      directed,
      attributes: omitXmlAttributes(attributes, [
        "type", "edge_type", "relation", "label", "weight", "value", "strength", "timestamp", "time", "date",
      ]),
    }));
  });

  const temporal = normalizeGraphEdgeTimestamps(edges, options.buildSpec?.timeFormat);
  if (!temporal.edges || temporal.issue) {
    const issue = temporal.issue ?? { code: "invalid_time_format", severity: "error" as const, message: "时间规范化失败。" };
    return failed(issue.message, [...issues, issue]);
  }
  const graphVersion = createGraphVersion(sourceFile, [...nodesById.values()], temporal.edges, issues, options);
  return { status: "ready", graphVersion, issues: graphVersion.issues };
}

function backendConversionIssue(format: "npz"): ValidationIssue {
  return {
    code: "backend_conversion_required",
    severity: "error",
    message: `${format.toUpperCase()} 是训练数据容器，需通过后端安全检查并转换为规范图后再预览。`,
  };
}

function sizeIssue(file: File): ValidationIssue | undefined {
  if (file.size === 0) {
    return { code: "empty_file", severity: "error", message: "文件为空，无法构图。" };
  }
  if (file.size > MAX_IMPORT_BYTES) {
    return {
      code: "file_too_large",
      severity: "error",
      message: "文件超过 20MB。请压缩数据或等待后端分块导入能力。",
      details: { maxBytes: MAX_IMPORT_BYTES, actualBytes: file.size },
    };
  }
  return undefined;
}

function delimiterFor(file: File): "\t" | undefined {
  return detectFormat(file) === "tsv" ? "\t" : undefined;
}

async function parseNodeEdgeTables(
  files: readonly File[],
  buildSpec: GraphBuildSpec,
  options: GraphImportParseOptions,
): Promise<ImportRun> {
  if (files.length !== 2) {
    return failed("双表导入必须恰好包含 nodes 与 edges 两个文件。", [{
      code: "invalid_bundle_size",
      severity: "error",
      message: "请选择一个节点表和一个关系表。",
    }]);
  }
  if (files.some((file) => !["csv", "tsv"].includes(detectFormat(file)))) {
    return failed("双表导入当前只接受 CSV / TSV。", [{
      code: "unsupported_bundle_format",
      severity: "error",
      message: "nodes + edges 双表当前只接受 CSV / TSV；标准图请单文件上传。",
    }]);
  }

  const parsed = await Promise.all(files.map((file) => parseDelimited(file, delimiterFor(file))));
  const artifacts = options.sourceArtifacts;
  if (artifacts?.length && (
    buildSpec.sourceArtifactIds.length !== artifacts.length ||
    artifacts.some((artifact, index) => buildSpec.sourceArtifactIds[index] !== artifact.id)
  )) {
    return failed("双表与 SourceArtifact 绑定不一致。", [{
      code: "source_artifact_binding_mismatch",
      severity: "error",
      message: "构图规则引用的源文件顺序或身份已变化，未继续解析。",
    }]);
  }
  const roleEdgeIndexes = artifacts
    ?.map((artifact, index) => artifact.role === "edges" ? index : -1)
    .filter((index) => index >= 0) ?? [];
  const roleNodeIndexes = artifacts
    ?.map((artifact, index) => artifact.role === "nodes" ? index : -1)
    .filter((index) => index >= 0) ?? [];
  if (artifacts?.length && (
    artifacts.length !== files.length ||
    roleEdgeIndexes.length !== 1 ||
    roleNodeIndexes.length !== 1 ||
    roleEdgeIndexes[0] === roleNodeIndexes[0]
  )) {
    return failed("双表的文件角色无效。", [{
      code: "invalid_table_roles",
      severity: "error",
      message: "双表导入必须明确包含一个 nodes 文件和一个 edges 文件。",
    }]);
  }
  const edgeCandidateIndex = parsed.findIndex((table) => {
    const mapping = buildSpec.edgeMapping;
    return Boolean(mapping && table.headers.includes(mapping.source) && table.headers.includes(mapping.target));
  });
  const inferredEdgeIndex = roleEdgeIndexes[0] ?? (edgeCandidateIndex >= 0
    ? edgeCandidateIndex
    : parsed.findIndex((table) => Boolean(suggestMapping(table.headers).source && suggestMapping(table.headers).target)));
  if (inferredEdgeIndex < 0) {
    return failed("无法识别关系表。", [{
      code: "edge_table_not_found",
      severity: "error",
      message: "两个文件中都没有已确认的起点列和终点列。",
    }]);
  }
  const nodeIndex = roleNodeIndexes[0] ?? (inferredEdgeIndex === 0 ? 1 : 0);
  const edgeIndex = inferredEdgeIndex;
  const nodeTable = parsed[nodeIndex];
  const edgeTable = parsed[edgeIndex];
  const nodeFile = files[nodeIndex];
  const edgeFile = files[edgeIndex];
  const suggestedNodeMapping = suggestNodeMapping(nodeTable.headers);
  const nodeMappingCandidate = buildSpec.nodeMapping ?? (
    suggestedNodeMapping.id ? suggestedNodeMapping as NodeColumnMapping : undefined
  );
  const suggestedEdgeMapping = suggestMapping(edgeTable.headers);
  const mappingCandidate = buildSpec.edgeMapping ?? suggestMapping(edgeTable.headers);
  const missingFields = [
    ...(!nodeMappingCandidate?.id ? ["node.id" as const] : []),
    ...(!mappingCandidate.source ? ["edge.source" as const] : []),
    ...(!mappingCandidate.target ? ["edge.target" as const] : []),
  ];
  if (missingFields.length) {
    return {
      status: "needs_mapping",
      headers: edgeTable.headers,
      suggestedMapping: suggestedEdgeMapping,
      mappingRequest: {
        nodeTable: {
          ...(artifacts?.[nodeIndex] ? { artifactId: artifacts[nodeIndex].id } : {}),
          headers: nodeTable.headers,
          suggestedMapping: suggestedNodeMapping,
        },
        edgeTable: {
          ...(artifacts?.[edgeIndex] ? { artifactId: artifacts[edgeIndex].id } : {}),
          headers: edgeTable.headers,
          suggestedMapping: suggestedEdgeMapping,
        },
        missingFields,
      },
      issues: [{
        code: "table_mapping_incomplete",
        severity: "warning",
        message: "节点表或关系表包含无法自动确认的必填字段，请完成手工映射。",
      }],
    };
  }
  const validatedNode = validateNodeMapping(nodeTable.headers, nodeMappingCandidate!);
  if (!validatedNode.mapping) {
    return {
      status: "needs_mapping",
      headers: edgeTable.headers,
      suggestedMapping: suggestedEdgeMapping,
      mappingRequest: {
        nodeTable: {
          ...(artifacts?.[nodeIndex] ? { artifactId: artifacts[nodeIndex].id } : {}),
          headers: nodeTable.headers,
          suggestedMapping: suggestedNodeMapping,
        },
        edgeTable: {
          ...(artifacts?.[edgeIndex] ? { artifactId: artifacts[edgeIndex].id } : {}),
          headers: edgeTable.headers,
          suggestedMapping: suggestedEdgeMapping,
        },
        missingFields: validatedNode.issues.some((issue) => issue.code === "node_id_column_missing")
          ? ["node.id"]
          : [],
      },
      issues: [...nodeTable.issues, ...edgeTable.issues, ...validatedNode.issues],
    };
  }
  const nodeMapping = validatedNode.mapping;
  const validated = validateMapping(edgeTable.headers, mappingCandidate as ColumnMapping);
  if (!validated.mapping) {
    return {
      status: "needs_mapping",
      headers: edgeTable.headers,
      suggestedMapping: suggestedEdgeMapping,
      mappingRequest: {
        nodeTable: {
          ...(artifacts?.[nodeIndex] ? { artifactId: artifacts[nodeIndex].id } : {}),
          headers: nodeTable.headers,
          suggestedMapping: suggestedNodeMapping,
        },
        edgeTable: {
          ...(artifacts?.[edgeIndex] ? { artifactId: artifacts[edgeIndex].id } : {}),
          headers: edgeTable.headers,
          suggestedMapping: suggestedEdgeMapping,
        },
        missingFields: [],
      },
      issues: [...nodeTable.issues, ...edgeTable.issues, ...validated.issues],
    };
  }
  const edgeMapping = validated.mapping;
  const issues: ValidationIssue[] = [...nodeTable.issues, ...edgeTable.issues];
  const nodesById = new Map<string, GraphNode>();
  const nodeExcluded = new Set([nodeMapping.id, nodeMapping.label, nodeMapping.type].filter((value): value is string => Boolean(value)));
  let blocked = false;

  nodeTable.rows.forEach((row, index) => {
    const rowNumber = index + 2;
    const id = asId(row[nodeMapping.id]);
    if (!id) {
      issues.push({ code: "empty_node_id", severity: "error", message: "节点 ID 不能为空。", row: rowNumber });
      blocked = true;
      return;
    }
    if (nodesById.has(id)) {
      issues.push({ code: "duplicate_node", severity: "error", message: `节点 ID“${id}”重复。`, row: rowNumber, entityId: id });
      blocked = true;
      return;
    }
    const label = nodeMapping.label ? asId(row[nodeMapping.label]) : undefined;
    const type = nodeMapping.type ? asId(row[nodeMapping.type]) : undefined;
    nodesById.set(id, Object.freeze({
      id,
      label: label ?? id,
      ...(type ? { type } : {}),
      attributes: collectAttributes(row, nodeExcluded),
    }));
  });

  let timeFormat: Exclude<GraphTimeFormat, "none" | "auto"> | undefined;
  if (edgeMapping.timestamp) {
    const requested = buildSpec.timeFormat;
    const detected = requested === "auto"
      ? detectTimeFormat(edgeTable.rows.map((row) => row[edgeMapping.timestamp!] ?? ""))
      : requested === "none" ? "invalid" : requested;
    if (detected === "invalid" || detected === "mixed") {
      issues.push({
        code: detected === "mixed" ? "ambiguous_time_format" : "invalid_time_format",
        severity: "error",
        message: "关系表时间列格式不明确，必须确认后再构图。",
      });
      blocked = true;
    } else timeFormat = detected;
  }

  const edgeExcluded = new Set([
    edgeMapping.source,
    edgeMapping.target,
    edgeMapping.edgeType,
    edgeMapping.weight,
    edgeMapping.timestamp,
  ].filter((value): value is string => Boolean(value)));
  const edges: GraphEdge[] = [];
  const duplicateKeys = new Set<string>();
  edgeTable.rows.forEach((row, index) => {
    const rowNumber = index + 2;
    const source = asId(row[edgeMapping.source]);
    const target = asId(row[edgeMapping.target]);
    if (!source || !target) {
      issues.push({ code: "empty_endpoint", severity: "error", message: "关系端点不能为空。", row: rowNumber });
      blocked = true;
      return;
    }
    if (!nodesById.has(source) || !nodesById.has(target)) {
      issues.push({
        code: "dangling_edge",
        severity: "error",
        message: `关系引用了节点表中不存在的端点：${source} → ${target}。`,
        row: rowNumber,
      });
      blocked = true;
      return;
    }
    if (source === target && buildSpec.selfLoopPolicy === "reject") {
      issues.push({ code: "self_loop_rejected", severity: "error", message: "构图规则不允许自环。", row: rowNumber });
      blocked = true;
      return;
    }
    const type = edgeMapping.edgeType ? asId(row[edgeMapping.edgeType]) : undefined;
    const directed = buildSpec.directionPolicy === "directed";
    const duplicateKey = directed || source <= target
      ? `${source}\u001f${target}\u001f${type ?? ""}`
      : `${target}\u001f${source}\u001f${type ?? ""}`;
    if (duplicateKeys.has(duplicateKey) && buildSpec.duplicateEdgePolicy === "reject") {
      issues.push({ code: "duplicate_edge_rejected", severity: "error", message: "构图规则不允许重复关系。", row: rowNumber });
      blocked = true;
      return;
    }
    duplicateKeys.add(duplicateKey);
    const rawWeight = edgeMapping.weight ? asId(row[edgeMapping.weight]) : undefined;
    const weight = rawWeight ? Number(rawWeight) : undefined;
    if (rawWeight && !Number.isFinite(weight)) {
      issues.push({ code: "invalid_weight", severity: "error", message: `权重“${rawWeight}”不是有效数字。`, row: rowNumber });
      blocked = true;
      return;
    }
    const rawTimestamp = edgeMapping.timestamp ? asId(row[edgeMapping.timestamp]) : undefined;
    const timestamp = rawTimestamp && timeFormat ? normalizeTimestamp(rawTimestamp, timeFormat) : undefined;
    if (rawTimestamp && timeFormat && !timestamp) {
      issues.push({ code: "invalid_timestamp", severity: "error", message: `时间“${rawTimestamp}”不符合已确认格式。`, row: rowNumber });
      blocked = true;
      return;
    }
    edges.push(Object.freeze({
      id: `edge-${index + 1}`,
      source,
      target,
      ...(type ? { type } : {}),
      ...(weight !== undefined ? { weight } : {}),
      ...(timestamp ? { timestamp } : {}),
      directed,
      attributes: collectAttributes(row, edgeExcluded),
    }));
  });

  if (blocked) return failed("双表数据存在阻断性质量问题，未生成 GraphVersion。", issues);
  if (!nodesById.size || !edges.length) return failed("双表中没有可构图的有效节点与关系。", issues);

  let finalEdges = edges;
  if (buildSpec.duplicateEdgePolicy === "merge_sum") {
    const merged = new Map<string, GraphEdge>();
    for (const edge of edges) {
      const key = edge.directed || edge.source <= edge.target
        ? `${edge.source}\u001f${edge.target}\u001f${edge.type ?? ""}`
        : `${edge.target}\u001f${edge.source}\u001f${edge.type ?? ""}`;
      const existing = merged.get(key);
      merged.set(key, existing ? Object.freeze({ ...existing, weight: (existing.weight ?? 1) + (edge.weight ?? 1) }) : edge);
    }
    finalEdges = [...merged.values()];
  }
  const graphVersion = createGraphVersion(
    `${nodeFile.name} + ${edgeFile.name}`,
    [...nodesById.values()],
    finalEdges,
    issues,
    { ...options, buildSpec },
  );
  return { status: "ready", graphVersion, issues: graphVersion.issues };
}

export class LocalGraphImportAdapter implements GraphImportAdapter {
  async inspect(file: File): Promise<FileProfile> {
    const format = detectFormat(file);
    const blockingIssue = sizeIssue(file);
    if (blockingIssue) {
      return {
        name: file.name,
        size: file.size,
        format,
        supported: false,
        headers: [],
        needsMapping: false,
        issues: [blockingIssue],
      };
    }

    if (format === "unsupported") {
      return {
        name: file.name,
        size: file.size,
        format,
        supported: false,
        headers: [],
        needsMapping: false,
        issues: [
          {
            code: "unsupported_format",
            severity: "error",
            message: "当前前端支持 CSV、TSV、JSON、GraphML 与 GEXF。该文件格式无法识别。",
          },
        ],
      };
    }

    if (format === "npz") {
      return {
        name: file.name,
        size: file.size,
        format,
        supported: false,
        headers: [],
        needsMapping: false,
        issues: [backendConversionIssue(format)],
      };
    }

    if (format === "json") {
      try {
        const json = JSON.parse(await readFileText(file));
        assertJsonRoot(json);
        return {
          name: file.name,
          size: file.size,
          format,
          supported: true,
          headers: [],
          needsMapping: false,
          issues: [],
        };
      } catch (error) {
        return {
          name: file.name,
          size: file.size,
          format,
          supported: false,
          headers: [],
          needsMapping: false,
          issues: [
            {
              code: "invalid_json",
              severity: "error",
              message: error instanceof Error ? error.message : "JSON 无法解析。",
            },
          ],
        };
      }
    }

    if (format === "graphml" || format === "gexf") {
      try {
        const run = xmlToGraph(file.name, await readFileText(file), format);
        return {
          name: file.name,
          size: file.size,
          format,
          supported: run.status === "ready",
          headers: [],
          needsMapping: false,
          issues: run.issues,
        };
      } catch (error) {
        return {
          name: file.name,
          size: file.size,
          format,
          supported: false,
          headers: [],
          needsMapping: false,
          issues: [{
            code: "file_read_failed",
            severity: "error",
            message: error instanceof Error ? error.message : "XML 文件读取失败。",
          }],
        };
      }
    }

    try {
      const parsed = await parseDelimited(file, format === "tsv" ? "\t" : undefined);
      const suggestedMapping = suggestMapping(parsed.headers);
      const needsMapping = !suggestedMapping.source || !suggestedMapping.target;
      return {
        name: file.name,
        size: file.size,
        format,
        supported: true,
        headers: parsed.headers,
        columns: profileColumns(parsed),
        suggestedMapping,
        needsMapping,
        issues: [
          ...parsed.issues,
          ...(needsMapping
            ? [
                {
                  code: "endpoint_columns_missing",
                  severity: "warning" as const,
                  message: "未自动识别起点列和终点列，需要手动映射。",
                },
              ]
            : []),
        ],
      };
    } catch (error) {
      return {
        name: file.name,
        size: file.size,
        format,
        supported: false,
        headers: [],
        needsMapping: false,
        issues: [
          {
            code: "file_read_failed",
            severity: "error",
            message: error instanceof Error ? error.message : "文件读取失败。",
          },
        ],
      };
    }
  }

  async parse(
    file: File,
    mapping?: ColumnMapping,
    options: GraphImportParseOptions = {},
  ): Promise<ImportRun> {
    const blockingIssue = sizeIssue(file);
    if (blockingIssue) return failed(blockingIssue.message, [blockingIssue]);

    const format = detectFormat(file);
    if (format === "unsupported") {
      const issue: ValidationIssue = {
        code: "unsupported_format",
        severity: "error",
        message: "当前前端支持 CSV、TSV、JSON、GraphML 与 GEXF。该文件格式无法识别。",
      };
      return failed(issue.message, [issue]);
    }

    if (format === "npz") {
      const issue = backendConversionIssue(format);
      return failed(issue.message, [issue]);
    }

    try {
      if (format === "csv" || format === "tsv") {
        return csvToGraph(
          file.name,
          await parseDelimited(file, format === "tsv" ? "\t" : undefined),
          mapping,
          options,
        );
      }
      if (format === "graphml" || format === "gexf") {
        return xmlToGraph(file.name, await readFileText(file), format, options);
      }
      return jsonToGraph(file.name, JSON.parse(await readFileText(file)), options);
    } catch (error) {
      const message = error instanceof Error ? error.message : "文件解析失败。";
      return failed(message, [{ code: "parse_failed", severity: "error", message }]);
    }
  }

  async parseFiles(
    files: readonly File[],
    buildSpec: GraphBuildSpec,
    options: GraphImportParseOptions = {},
  ): Promise<ImportRun> {
    if (buildSpec.inputShape !== "node_edge_tables") {
      if (files.length !== 1) {
        return failed("当前构图规则要求单文件输入。", [{
          code: "invalid_input_shape",
          severity: "error",
          message: "标准图或单一关系表只能绑定一个 SourceArtifact。",
        }]);
      }
      return this.parse(files[0], buildSpec.edgeMapping, { ...options, buildSpec });
    }
    try {
      return await parseNodeEdgeTables(files, buildSpec, { ...options, buildSpec });
    } catch (error) {
      const message = error instanceof Error ? error.message : "双表解析失败。";
      return failed(message, [{ code: "parse_failed", severity: "error", message }]);
    }
  }
}

export function buildDemoGraphVersion(): GraphVersion {
  const demoEntities: Array<[string, string]> = [
    ["张三", "实验室成员"], ["李四", "实验室成员"], ["王五", "实验室成员"], ["赵六", "实验室成员"],
    ["陈七", "实验室成员"], ["孙八", "实验室成员"], ["周青", "实验室成员"], ["许诺", "实验室成员"],
    ["刘洋", "外部合作者"], ["陈晨", "外部合作者"], ["孙琳", "外部合作者"], ["吴昊", "外部合作者"],
    ["何露", "外部合作者"], ["高原", "外部合作者"], ["马骏", "外部合作者"], ["朱敏", "外部合作者"],
    ["智能计算中心", "合作机构"], ["数据科学实验室", "合作机构"], ["城市治理中心", "合作机构"],
    ["网络科学中心", "合作机构"], ["知识工程研究所", "合作机构"], ["区域创新中心", "合作机构"],
    ["协作网络项目", "研究项目"], ["社区韧性项目", "研究项目"], ["创新扩散项目", "研究项目"], ["知识图谱项目", "研究项目"],
  ];
  const nodes: GraphNode[] = demoEntities.map(([label, type], index) =>
    Object.freeze({
      id: `demo-node-${index + 1}`,
      label,
      type,
      attributes: Object.freeze({ demo: true }),
    }),
  );
  const pairs: Array<[number, number, string]> = [
    [0, 1, "合作"],
    [0, 2, "合作"],
    [0, 3, "指导"],
    [1, 2, "合作"],
    [1, 4, "交流"],
    [2, 5, "合作"],
    [3, 6, "交流"],
    [4, 7, "合作"],
    [2, 4, "协作"],
    [5, 7, "交流"],
    [0, 8, "合作"], [0, 9, "合作"], [0, 16, "参与"], [0, 22, "参与"],
    [1, 10, "合作"], [1, 11, "交流"], [1, 17, "参与"], [1, 22, "参与"],
    [2, 12, "合作"], [2, 13, "合作"], [2, 18, "参与"], [2, 23, "参与"],
    [3, 14, "合作"], [3, 16, "参与"], [3, 19, "参与"], [3, 24, "参与"],
    [4, 15, "合作"], [4, 17, "参与"], [4, 20, "参与"], [4, 25, "参与"],
    [5, 9, "交流"], [5, 18, "参与"], [5, 24, "参与"],
    [6, 11, "合作"], [6, 19, "参与"], [6, 25, "参与"],
    [7, 13, "交流"], [7, 20, "参与"], [7, 21, "参与"],
    [8, 16, "隶属"], [9, 16, "隶属"], [10, 17, "隶属"], [11, 19, "隶属"],
    [12, 18, "隶属"], [13, 20, "隶属"], [14, 21, "隶属"], [15, 21, "隶属"],
    [16, 22, "支持"], [17, 22, "支持"], [18, 23, "支持"], [19, 24, "支持"],
    [20, 25, "支持"], [21, 24, "支持"], [22, 23, "关联"], [24, 25, "关联"],
  ];
  const edges: GraphEdge[] = pairs.map(([source, target, type], index) =>
    Object.freeze({
      id: `demo-edge-${index + 1}`,
      source: nodes[source].id,
      target: nodes[target].id,
      type,
      weight: 1,
      attributes: Object.freeze({ demo: true }),
    }),
  );

  return createGraphVersion("内置结构数据（非分析结论）", nodes, edges, [
    {
      code: "demo_data",
      severity: "info",
      message: "当前图谱为内置结构数据，不代表真实分析结论。",
    },
  ], {
    provenance: {
      origin: "seed_demo",
      pipeline: "demo",
      pipelineVersion: "2.0.0",
    },
  });
}
