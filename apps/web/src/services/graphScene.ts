import type {
  AnalysisOverlay,
  GraphEdge,
  GraphFilters,
  GraphNode,
  GraphPath,
  GraphScene,
  GraphSceneStatus,
  GraphSlice,
  GraphVersion,
  GraphViewState,
} from "../types/graph";
import { MAX_VISIBLE_EDGES, MAX_VISIBLE_NODES } from "../types/graph";
import { rankDegreeCentrality } from "./graphAlgorithms";
import { filterGraphFacts, normalizeGraphFilters } from "./graphFilters";
import { compareUnicodeCodePoints, sha256Canonical } from "./graphIdentity";
import { extractLocalSubgraph, findShortestPath } from "./graphTraversal";

export interface BuildGraphSceneOptions {
  readonly viewState?: GraphViewState;
  readonly overlay?: AnalysisOverlay;
  readonly path?: GraphPath | null;
  readonly maxNodes?: number;
  readonly maxEdges?: number;
}

export interface SemanticGraphSliceResult {
  readonly slice: GraphSlice;
  readonly filters: GraphFilters;
  readonly path: GraphPath | null;
  readonly status: GraphSceneStatus;
}

function compareIds(left: string, right: string): number {
  return compareUnicodeCodePoints(left, right);
}

export function createDefaultGraphViewState(graphVersionId: string): GraphViewState {
  return Object.freeze({
    graphVersionId,
    mode: "global" as const,
    focusNodeIds: Object.freeze([]),
    pathEndpointIds: Object.freeze([]),
    depth: 1 as const,
    filters: Object.freeze({
      nodeTypes: Object.freeze([]),
      edgeTypes: Object.freeze([]),
    }),
    theme: "brand-light" as const,
    layoutPreset: "balanced" as const,
    rendererPreference: "auto" as const,
    camera: Object.freeze({ x: 0, y: 0, zoom: 1 }),
    pinnedNodes: Object.freeze({}),
  });
}

/** Adds fields introduced after persisted view-state schema v1. */
export function normalizeGraphViewState(
  graphVersionId: string,
  stored?: Partial<GraphViewState> | null,
): GraphViewState {
  const fallback = createDefaultGraphViewState(graphVersionId);
  if (!stored || stored.graphVersionId !== graphVersionId) return fallback;
  const legacyFocusNodeIds = Array.isArray(stored.focusNodeIds) ? stored.focusNodeIds : [];
  const pathEndpointIds = Array.isArray(stored.pathEndpointIds)
    ? stored.pathEndpointIds.slice(0, 2)
    : stored.mode === "path"
      ? legacyFocusNodeIds.slice(0, 2)
      : [];
  return Object.freeze({
    ...fallback,
    ...stored,
    graphVersionId,
    rendererPreference:
      stored.rendererPreference === "canvas" ||
      stored.rendererPreference === "hybrid-webgl"
        ? stored.rendererPreference
        : "auto",
    focusNodeIds: Object.freeze([...(stored.mode === "path" && !stored.pathEndpointIds ? [] : legacyFocusNodeIds)]),
    pathEndpointIds: Object.freeze([...pathEndpointIds]),
    filters: normalizeGraphFilters({
      nodeTypes: Object.freeze([...(stored.filters?.nodeTypes ?? [])]),
      edgeTypes: Object.freeze([...(stored.filters?.edgeTypes ?? [])]),
      ...(stored.filters?.timeRange ? { timeRange: Object.freeze({ ...stored.filters.timeRange }) } : {}),
      ...(typeof stored.filters?.minWeight === "number" && Number.isFinite(stored.filters.minWeight)
        ? { minWeight: stored.filters.minWeight }
        : {}),
      ...(typeof stored.filters?.maxWeight === "number" && Number.isFinite(stored.filters.maxWeight)
        ? { maxWeight: stored.filters.maxWeight }
        : {}),
      ...(typeof stored.filters?.directed === "boolean" ? { directed: stored.filters.directed } : {}),
      ...(stored.filters?.emptyReason ? { emptyReason: stored.filters.emptyReason } : {}),
    }),
    camera: Object.freeze({ ...(stored.camera ?? fallback.camera) }),
    pinnedNodes: Object.freeze({ ...(stored.pinnedNodes ?? {}) }),
  });
}

function selectViewSlice(
  filtered: GraphSlice,
  state: GraphViewState,
  requestedPath?: GraphPath | null,
): { slice: GraphSlice; path: GraphPath | null; status: GraphSceneStatus } {
  if (state.mode === "local") {
    if (state.focusNodeIds.length === 0) {
      return { slice: filtered, path: null, status: Object.freeze({ kind: "awaiting_focus" }) };
    }
    const slice = extractLocalSubgraph(filtered, state.focusNodeIds, state.depth);
    if (slice.nodes.length === 0) {
      return { slice: filtered, path: null, status: Object.freeze({ kind: "awaiting_focus" }) };
    }
    return {
      slice,
      path: requestedPath ?? null,
      status: Object.freeze({ kind: "ready" }),
    };
  }
  if (state.mode === "path") {
    const endpoints = state.pathEndpointIds;
    if (endpoints.length === 0) {
      return { slice: filtered, path: null, status: Object.freeze({ kind: "awaiting_focus" }) };
    }
    if (endpoints.length === 1) {
      return {
        slice: filtered,
        path: null,
        status: Object.freeze({ kind: "awaiting_path_end", sourceId: endpoints[0] }),
      };
    }
    const path = requestedPath === undefined
      ? findShortestPath(filtered, endpoints[0], endpoints[1])
      : requestedPath;
    if (!path) {
      return {
        slice: filtered,
        path: null,
        status: Object.freeze({ kind: "no_path", sourceId: endpoints[0], targetId: endpoints[1] }),
      };
    }
    return { slice: path, path, status: Object.freeze({ kind: "ready" }) };
  }
  return { slice: filtered, path: requestedPath ?? null, status: Object.freeze({ kind: "ready" }) };
}

/** Builds the complete semantic slice used by algorithms before render caps apply. */
export function buildSemanticGraphSlice(
  graph: GraphVersion,
  options: Pick<BuildGraphSceneOptions, "viewState" | "path"> = {},
): SemanticGraphSliceResult {
  const state = normalizeGraphViewState(graph.id, options.viewState);
  const filtered = filterGraphFacts(graph, state.filters);
  const selected = selectViewSlice(filtered.slice, state, options.path);
  return Object.freeze({
    slice: selected.slice,
    filters: filtered.filters,
    path: selected.path,
    status: selected.status,
  });
}

function incidentDegree(nodes: readonly GraphNode[], edges: readonly GraphEdge[]): Map<string, number> {
  return new Map(rankDegreeCentrality({ nodes, edges }).map((entry) => [entry.nodeId, entry.degree]));
}

function communityRepresentatives(
  nodes: readonly GraphNode[],
  overlay: AnalysisOverlay | undefined,
  degree: ReadonlyMap<string, number>,
): string[] {
  if (!overlay || (overlay.kind !== "community" && overlay.kind !== "components")) return [];
  const membersByGroup = new Map<string, string[]>();
  for (const node of nodes) {
    const value = overlay.nodeValues[node.id];
    if (typeof value !== "string") continue;
    const members = membersByGroup.get(value) ?? [];
    members.push(node.id);
    membersByGroup.set(value, members);
  }
  return [...membersByGroup.entries()]
    .sort(([left], [right]) => compareIds(left, right))
    .map(([, members]) =>
      members.sort((left, right) => (degree.get(right) ?? 0) - (degree.get(left) ?? 0) || compareIds(left, right))[0],
    );
}

function sampleScene(
  slice: GraphSlice,
  focusNodeIds: readonly string[],
  path: GraphPath | null,
  overlay: AnalysisOverlay | undefined,
  maxNodes: number,
  maxEdges: number,
): { nodes: readonly GraphNode[]; edges: readonly GraphEdge[]; truncated: boolean } {
  if (slice.nodes.length <= maxNodes && slice.edges.length <= maxEdges) {
    return { nodes: slice.nodes, edges: slice.edges, truncated: false };
  }

  const availableIds = new Set(slice.nodes.map((node) => node.id));
  const selected = new Set<string>();
  const add = (nodeId: string): boolean => {
    if (!availableIds.has(nodeId) || selected.has(nodeId) || selected.size >= maxNodes) return false;
    selected.add(nodeId);
    return true;
  };
  const stableEdges = [...slice.edges].sort((left, right) => compareIds(left.id, right.id));
  if (overlay?.kind === "governance") {
    const governanceNodeIds = new Set<string>(Object.keys(overlay.nodeValues));
    for (const edge of overlay.candidateEdges ?? []) {
      governanceNodeIds.add(edge.sourceId);
      governanceNodeIds.add(edge.targetId);
    }
    for (const nodeId of governanceNodeIds) add(nodeId);
  }
  for (const nodeId of focusNodeIds) {
    add(nodeId);
    const incident = stableEdges.find(
      (edge) => edge.source === nodeId || edge.target === nodeId,
    );
    if (incident) add(incident.source === nodeId ? incident.target : incident.source);
  }
  for (const nodeId of path?.nodeIds ?? []) add(nodeId);

  const degree = incidentDegree(slice.nodes, slice.edges);
  const representatives = communityRepresentatives(slice.nodes, overlay, degree);

  const degreeOrder = slice.nodes
    .map((node) => node.id)
    .sort((left, right) => (degree.get(right) ?? 0) - (degree.get(left) ?? 0) || compareIds(left, right));
  const pathEdgeIds = new Set(path?.edgeIds ?? []);
  const focus = new Set(focusNodeIds);
  const relationshipOrder = [...slice.edges].sort((left, right) => {
    const leftPath = pathEdgeIds.has(left.id) ? 1 : 0;
    const rightPath = pathEdgeIds.has(right.id) ? 1 : 0;
    if (leftPath !== rightPath) return rightPath - leftPath;
    const leftFocus = focus.has(left.source) || focus.has(left.target) ? 1 : 0;
    const rightFocus = focus.has(right.source) || focus.has(right.target) ? 1 : 0;
    if (leftFocus !== rightFocus) return rightFocus - leftFocus;
    const weightDifference = (right.weight ?? 1) - (left.weight ?? 1);
    return weightDifference || compareIds(left.id, right.id);
  });

  const incidentEdges = new Map<string, GraphEdge[]>();
  for (const edge of relationshipOrder) {
    const sourceEdges = incidentEdges.get(edge.source) ?? [];
    sourceEdges.push(edge);
    incidentEdges.set(edge.source, sourceEdges);
    if (edge.target !== edge.source) {
      const targetEdges = incidentEdges.get(edge.target) ?? [];
      targetEdges.push(edge);
      incidentEdges.set(edge.target, targetEdges);
    }
  }

  // Grow the projection through real relationships. Adding high-degree nodes
  // independently and truncating edges afterwards produced visible nodes that
  // looked isolated even though the source graph connected them.
  const connectSelectedNode = (nodeId: string): boolean => {
    if (!availableIds.has(nodeId) || selected.has(nodeId)) return false;
    const candidates = incidentEdges.get(nodeId) ?? [];
    const toSelected = candidates.find((edge) =>
      selected.has(edge.source === nodeId ? edge.target : edge.source),
    );
    if (toSelected) return add(nodeId);
    const neighbourEdge = candidates.find((edge) => {
      const neighbour = edge.source === nodeId ? edge.target : edge.source;
      return availableIds.has(neighbour) && neighbour !== nodeId;
    });
    if (neighbourEdge && selected.size + 2 <= maxNodes) {
      const neighbour =
        neighbourEdge.source === nodeId ? neighbourEdge.target : neighbourEdge.source;
      add(nodeId);
      add(neighbour);
      return true;
    }
    if (candidates.length === 0 || selected.size === 0) return add(nodeId);
    return false;
  };

  // Give every protected seed a neighbour before spending the remaining
  // budget on ranked nodes.
  for (const nodeId of [...selected]) {
    if ((degree.get(nodeId) ?? 0) === 0) continue;
    const edge = (incidentEdges.get(nodeId) ?? []).find((candidate) => {
      const neighbour = candidate.source === nodeId ? candidate.target : candidate.source;
      return !selected.has(neighbour);
    });
    if (!edge || selected.size >= maxNodes) continue;
    add(edge.source === nodeId ? edge.target : edge.source);
  }
  for (const nodeId of representatives) connectSelectedNode(nodeId);
  for (const nodeId of degreeOrder) connectSelectedNode(nodeId);
  for (const edge of relationshipOrder) {
    if (selected.size >= maxNodes) break;
    if (selected.has(edge.source)) add(edge.target);
    else if (selected.has(edge.target)) add(edge.source);
    else if (selected.size + 2 <= maxNodes) {
      add(edge.source);
      add(edge.target);
    }
  }
  // True isolates are valid graph facts and may fill the remaining budget.
  for (const nodeId of degreeOrder) {
    if ((degree.get(nodeId) ?? 0) === 0) add(nodeId);
  }

  const nodes = slice.nodes.filter((node) => selected.has(node.id));
  const candidateEdges = relationshipOrder.filter(
    (edge) => selected.has(edge.source) && selected.has(edge.target),
  );
  const selectedEdgeIds = new Set<string>();
  const selectedEdges: GraphEdge[] = [];
  const appendEdge = (edge: GraphEdge) => {
    if (selectedEdges.length >= maxEdges || selectedEdgeIds.has(edge.id)) return;
    selectedEdgeIds.add(edge.id);
    selectedEdges.push(edge);
  };
  for (const edge of candidateEdges) {
    if (pathEdgeIds.has(edge.id)) appendEdge(edge);
  }

  // Cover as many non-isolated visible nodes as possible before filling by
  // weight. With the production 12k/3k budget this preserves at least one
  // incident relationship for every visible non-isolate.
  const uncovered = new Set(
    nodes.filter((node) => (degree.get(node.id) ?? 0) > 0).map((node) => node.id),
  );
  for (const edge of selectedEdges) {
    uncovered.delete(edge.source);
    uncovered.delete(edge.target);
  }
  while (selectedEdges.length < maxEdges && uncovered.size > 0) {
    let best: GraphEdge | undefined;
    let bestCoverage = 0;
    for (const edge of candidateEdges) {
      if (selectedEdgeIds.has(edge.id)) continue;
      const coverage = Number(uncovered.has(edge.source)) + Number(uncovered.has(edge.target));
      if (coverage > bestCoverage) {
        best = edge;
        bestCoverage = coverage;
        if (coverage === 2) break;
      }
    }
    if (!best || bestCoverage === 0) break;
    appendEdge(best);
    uncovered.delete(best.source);
    uncovered.delete(best.target);
  }
  for (const edge of candidateEdges) appendEdge(edge);
  return {
    nodes: Object.freeze(nodes),
    edges: Object.freeze(selectedEdges),
    truncated: nodes.length < slice.nodes.length || selectedEdges.length < slice.edges.length,
  };
}

/** Builds a render scene from immutable facts without mutating the graph or overlay. */
export function buildGraphScene(graph: GraphVersion, options: BuildGraphSceneOptions = {}): GraphScene {
  const state = normalizeGraphViewState(graph.id, options.viewState);
  const selected = buildSemanticGraphSlice(graph, options);
  const maxNodes = Math.max(1, Math.floor(options.maxNodes ?? MAX_VISIBLE_NODES));
  const maxEdges = Math.max(0, Math.floor(options.maxEdges ?? MAX_VISIBLE_EDGES));
  const sampled = sampleScene(
    selected.slice,
    state.mode === "path" ? state.pathEndpointIds : state.focusNodeIds,
    selected.path,
    options.overlay,
    maxNodes,
    maxEdges,
  );
  const visibleIds = new Set(sampled.nodes.map((node) => node.id));
  const visiblePathNodeIds = (selected.path?.nodeIds ?? []).filter((nodeId) => visibleIds.has(nodeId));
  const visibleEdgeIds = new Set(sampled.edges.map((edge) => edge.id));
  const visiblePathEdgeIds = (selected.path?.edgeIds ?? []).filter((edgeId) => visibleEdgeIds.has(edgeId));
  const projectionHash = sha256Canonical({
    graphVersionId: graph.id,
    nodeIds: sampled.nodes.map((node) => node.id).sort(compareUnicodeCodePoints),
    edgeIds: sampled.edges.map((edge) => edge.id).sort(compareUnicodeCodePoints),
    maxNodes,
    maxEdges,
  });

  return Object.freeze({
    graphVersionId: graph.id,
    nodes: Object.freeze([...sampled.nodes]),
    edges: Object.freeze([...sampled.edges]),
    focusNodeIds: Object.freeze(state.focusNodeIds.filter((nodeId) => visibleIds.has(nodeId))),
    pathEndpointIds: Object.freeze([...state.pathEndpointIds]),
    pathNodeIds: Object.freeze(visiblePathNodeIds),
    pathEdgeIds: Object.freeze(visiblePathEdgeIds),
    status: selected.status,
    ...(options.overlay ? { overlay: options.overlay } : {}),
    truncated: sampled.truncated,
    originalNodeCount: selected.slice.nodes.length,
    originalEdgeCount: selected.slice.edges.length,
    visibleNodeCount: sampled.nodes.length,
    visibleEdgeCount: sampled.edges.length,
    projectionHash,
  });
}
