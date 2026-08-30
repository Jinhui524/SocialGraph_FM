import type {
  GraphEdge,
  GraphFilters,
  GraphNode,
  GraphSlice,
  GraphTimeRange,
} from "../types/graph";
import { compareUnicodeCodePoints } from "./graphIdentity";

function uniqueSorted(values: readonly string[]): readonly string[] {
  return Object.freeze(
    [...new Set(values.map((value) => value.trim()).filter(Boolean))].sort(compareUnicodeCodePoints),
  );
}

function finite(value: number | undefined): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function parseBoundary(value: string | undefined, endBoundary = false): number | undefined {
  if (!value) return undefined;
  if (/^\d{4}$/u.test(value)) {
    return Date.parse(endBoundary
      ? `${value}-12-31T23:59:59.999Z`
      : `${value}-01-01T00:00:00.000Z`);
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function normalizeTimeRange(timeRange: GraphTimeRange | undefined): {
  readonly value?: GraphTimeRange;
  readonly invalid: boolean;
} {
  if (!timeRange?.start && !timeRange?.end) return { invalid: false };
  const start = timeRange.start?.trim() || undefined;
  const end = timeRange.end?.trim() || undefined;
  const startValue = parseBoundary(start);
  const endValue = parseBoundary(end, true);
  if ((start && startValue === undefined) || (end && endValue === undefined)) {
    return { invalid: true };
  }
  if (startValue !== undefined && endValue !== undefined && startValue > endValue) {
    return { invalid: true };
  }
  return {
    invalid: false,
    value: Object.freeze({ ...(start ? { start } : {}), ...(end ? { end } : {}) }),
  };
}

/** Canonicalises set-like filters before persistence, comparison, and hashing. */
export function normalizeGraphFilters(filters: GraphFilters): GraphFilters {
  const minWeight = finite(filters.minWeight);
  const maxWeight = finite(filters.maxWeight);
  const timeRange = normalizeTimeRange(filters.timeRange);
  const invalidWeightRange = minWeight !== undefined
    && maxWeight !== undefined
    && minWeight > maxWeight;
  const emptyReason = filters.emptyReason
    ?? (invalidWeightRange ? "invalid_weight_range" : undefined)
    ?? (timeRange.invalid ? "invalid_time_range" : undefined);
  return Object.freeze({
    nodeTypes: uniqueSorted(filters.nodeTypes),
    edgeTypes: uniqueSorted(filters.edgeTypes),
    ...(timeRange.value ? { timeRange: timeRange.value } : {}),
    ...(minWeight !== undefined ? { minWeight } : {}),
    ...(maxWeight !== undefined ? { maxWeight } : {}),
    ...(typeof filters.directed === "boolean" ? { directed: filters.directed } : {}),
    ...(emptyReason ? { emptyReason } : {}),
  });
}

export function graphFiltersHaveRelationshipConstraints(filters: GraphFilters): boolean {
  return filters.edgeTypes.length > 0
    || Boolean(filters.timeRange?.start || filters.timeRange?.end)
    || filters.minWeight !== undefined
    || filters.maxWeight !== undefined
    || filters.directed !== undefined;
}

export function graphFilterConstraintCount(filters: GraphFilters): number {
  const normalized = normalizeGraphFilters(filters);
  return normalized.nodeTypes.length
    + normalized.edgeTypes.length
    + Number(Boolean(normalized.timeRange?.start))
    + Number(Boolean(normalized.timeRange?.end))
    + Number(normalized.minWeight !== undefined)
    + Number(normalized.maxWeight !== undefined)
    + Number(normalized.directed !== undefined)
    + Number(Boolean(normalized.emptyReason));
}

export function graphEdgeMatchesFilters(edge: GraphEdge, filters: GraphFilters): boolean {
  if (filters.emptyReason) return false;
  if (filters.edgeTypes.length > 0 && (!edge.type || !filters.edgeTypes.includes(edge.type))) return false;
  if (filters.minWeight !== undefined && (edge.weight === undefined || edge.weight < filters.minWeight)) return false;
  if (filters.maxWeight !== undefined && (edge.weight === undefined || edge.weight > filters.maxWeight)) return false;
  if (filters.directed !== undefined && edge.directed !== filters.directed) return false;
  if (filters.timeRange?.start || filters.timeRange?.end) {
    if (!edge.timestamp) return false;
    const candidate = parseBoundary(edge.timestamp);
    if (candidate === undefined) return false;
    const start = parseBoundary(filters.timeRange.start);
    const end = parseBoundary(filters.timeRange.end, true);
    if (start !== undefined && candidate < start) return false;
    if (end !== undefined && candidate > end) return false;
  }
  return true;
}

/** Applies semantic filters to immutable graph facts without any render projection. */
export function filterGraphFacts(
  graph: { readonly nodes: readonly GraphNode[]; readonly edges: readonly GraphEdge[] },
  inputFilters: GraphFilters,
): { readonly filters: GraphFilters; readonly slice: GraphSlice } {
  const filters = normalizeGraphFilters(inputFilters);
  if (filters.emptyReason) {
    return Object.freeze({
      filters,
      slice: Object.freeze({
        nodes: Object.freeze([]),
        edges: Object.freeze([]),
        nodeIds: Object.freeze([]),
        edgeIds: Object.freeze([]),
      }),
    });
  }
  const nodeTypes = new Set(filters.nodeTypes);
  const candidateNodes = graph.nodes.filter(
    (node) => nodeTypes.size === 0 || nodeTypes.has(node.type?.trim() || "未分类"),
  );
  const candidateNodeIds = new Set(candidateNodes.map((node) => node.id));
  const edges = graph.edges.filter(
    (edge) => candidateNodeIds.has(edge.source)
      && candidateNodeIds.has(edge.target)
      && graphEdgeMatchesFilters(edge, filters),
  );
  const incidentNodeIds = graphFiltersHaveRelationshipConstraints(filters)
    ? new Set(edges.flatMap((edge) => [edge.source, edge.target]))
    : undefined;
  const nodes = incidentNodeIds
    ? candidateNodes.filter((node) => incidentNodeIds.has(node.id))
    : candidateNodes;
  return Object.freeze({
    filters,
    slice: Object.freeze({
      nodes: Object.freeze([...nodes]),
      edges: Object.freeze([...edges]),
      nodeIds: Object.freeze(nodes.map((node) => node.id)),
      edgeIds: Object.freeze(edges.map((edge) => edge.id)),
    }),
  });
}
