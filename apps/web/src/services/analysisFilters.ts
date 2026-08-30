import type {
  GraphFilters,
  GraphVersion,
  NormalizedIntent,
  ViewCommand,
} from "../types/graph";
import { inferGraphDirectedness } from "./graphAlgorithms";

const EXECUTABLE_FILTER_KEYS = new Set([
  "startYear",
  "endYear",
  "nodeType",
  "edgeType",
  "minWeight",
  "maxWeight",
  "directed",
  "component",
]);

function stringFilter(filters: NormalizedIntent["filters"], key: string): string | undefined {
  const value = filters[key];
  if (typeof value !== "string") return undefined;
  const cleaned = value.trim();
  return cleaned || undefined;
}

function numberFilter(filters: NormalizedIntent["filters"], key: string): number | undefined {
  const value = filters[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

export interface PreparedAnalysisFilters {
  readonly command?: ViewCommand;
  readonly minWeight?: number;
  readonly maxWeight?: number;
  readonly directed?: boolean;
  readonly emptyReason?: GraphFilters["emptyReason"];
  readonly warnings: readonly string[];
}

/** Converts the bounded backend filter contract into executable graph filters. */
export function prepareAnalysisFilters(
  graph: GraphVersion,
  intent: NormalizedIntent,
): PreparedAnalysisFilters {
  const warnings: string[] = [];
  for (const key of Object.keys(intent.filters)) {
    if (!EXECUTABLE_FILTER_KEYS.has(key)) warnings.push(`筛选字段“${key}”不在前端可执行合同中，已忽略。`);
  }
  if (intent.filters.component !== undefined) {
    warnings.push("component 筛选尚未实现安全的确定性语义，已从本次执行合同中移除。");
  }

  const nodeType = stringFilter(intent.filters, "nodeType");
  const edgeType = stringFilter(intent.filters, "edgeType");
  const start = intent.timeRange?.start ?? stringFilter(intent.filters, "startYear");
  const end = intent.timeRange?.end ?? stringFilter(intent.filters, "endYear");
  const inherited = intent.view;
  const nodeTypeTerms = [...new Set([...(inherited?.nodeTypeTerms ?? []), ...(nodeType ? [nodeType] : [])])];
  const edgeTypeTerms = [...new Set([...(inherited?.edgeTypeTerms ?? []), ...(edgeType ? [edgeType] : [])])];
  const timeRange = start || end
    ? { ...(start ? { start } : {}), ...(end ? { end } : {}) }
    : inherited?.timeRange;
  const hasCommand = Boolean(
    inherited
    || nodeTypeTerms.length
    || edgeTypeTerms.length
    || timeRange,
  );
  const command = hasCommand ? Object.freeze({
    ...(inherited?.mode ? { mode: inherited.mode } : {}),
    focusTerms: Object.freeze([...(inherited?.focusTerms ?? [])]),
    ...(inherited?.depth ? { depth: inherited.depth } : {}),
    nodeTypeTerms: Object.freeze(nodeTypeTerms),
    edgeTypeTerms: Object.freeze(edgeTypeTerms),
    ...(timeRange ? { timeRange: Object.freeze(timeRange) } : {}),
    ...(inherited?.layoutPreset ? { layoutPreset: inherited.layoutPreset } : {}),
    ...(inherited?.overlay ? { overlay: inherited.overlay } : {}),
  }) : undefined;

  const minWeight = numberFilter(intent.filters, "minWeight");
  const maxWeight = numberFilter(intent.filters, "maxWeight");
  let emptyReason: GraphFilters["emptyReason"];
  if (intent.filters.minWeight !== undefined && minWeight === undefined) warnings.push("minWeight 不是有效数字，已忽略。");
  if (intent.filters.maxWeight !== undefined && maxWeight === undefined) warnings.push("maxWeight 不是有效数字，已忽略。");
  if (minWeight !== undefined && maxWeight !== undefined && minWeight > maxWeight) {
    emptyReason = "invalid_weight_range";
    warnings.push("minWeight 大于 maxWeight，本次分析按空范围处理。");
  }

  const directed = typeof intent.filters.directed === "boolean" ? intent.filters.directed : undefined;
  if (intent.filters.directed !== undefined && directed === undefined) {
    warnings.push("directed 必须是布尔值，已忽略。");
  }
  if (directed !== undefined) {
    const directedness = graph.metadata?.directedness ?? inferGraphDirectedness(graph);
    if (directedness === "unspecified") {
      emptyReason = "direction_unknown";
      warnings.push("当前 GraphVersion 没有可验证的方向元数据，本次方向筛选按空范围处理。");
    } else if ((directed && directedness === "undirected") || (!directed && directedness === "directed")) {
      emptyReason = "direction_mismatch";
      warnings.push(`请求的${directed ? "有向" : "无向"}条件与 GraphVersion 方向元数据不一致，本次分析按空范围处理。`);
    }
  }

  return Object.freeze({
    ...(command ? { command } : {}),
    ...(minWeight !== undefined ? { minWeight } : {}),
    ...(maxWeight !== undefined ? { maxWeight } : {}),
    ...(directed !== undefined ? { directed } : {}),
    ...(emptyReason ? { emptyReason } : {}),
    warnings: Object.freeze(warnings),
  });
}

export function applyPreparedAnalysisFilters(
  base: GraphFilters,
  prepared: PreparedAnalysisFilters,
): GraphFilters {
  const minWeight = prepared.minWeight ?? base.minWeight;
  const maxWeight = prepared.maxWeight ?? base.maxWeight;
  const directed = prepared.directed ?? base.directed;
  let emptyReason = prepared.emptyReason;
  if (!emptyReason && minWeight !== undefined && maxWeight !== undefined && minWeight > maxWeight) {
    emptyReason = "invalid_weight_range";
  }
  if (
    !emptyReason
    && prepared.directed === undefined
    && (base.emptyReason === "direction_mismatch" || base.emptyReason === "direction_unknown")
  ) {
    emptyReason = base.emptyReason;
  }
  return Object.freeze({
    nodeTypes: Object.freeze([...base.nodeTypes]),
    edgeTypes: Object.freeze([...base.edgeTypes]),
    ...(base.timeRange ? { timeRange: Object.freeze({ ...base.timeRange }) } : {}),
    ...(minWeight !== undefined ? { minWeight } : {}),
    ...(maxWeight !== undefined ? { maxWeight } : {}),
    ...(directed !== undefined ? { directed } : {}),
    ...(emptyReason ? { emptyReason } : {}),
  });
}
