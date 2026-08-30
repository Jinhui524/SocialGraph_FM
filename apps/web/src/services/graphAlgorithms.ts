import type {
  AnalysisResult,
  AnalysisTask,
  DegreeRankEntry,
  GraphEdge,
  GraphNode,
  GraphFilters,
  GraphScene,
  ScopedGraphSlice,
  GraphSummary,
  GraphVersion,
} from "../types/graph";
import { computeLouvainCommunities } from "./graphCommunities";
import { normalizeGraphFilters } from "./graphFilters";
import { compareUnicodeCodePoints, sha256Canonical } from "./graphIdentity";

type Adjacency = Map<string, Set<string>>;

export function inferGraphDirectedness(
  graph: Pick<GraphVersion, "edges">,
): "directed" | "undirected" | "mixed" | "unspecified" {
  if (!graph.edges.length) return "unspecified";
  let directed = 0;
  let undirected = 0;
  let unspecified = 0;
  for (const edge of graph.edges) {
    if (edge.directed === true) directed += 1;
    else if (edge.directed === false) undirected += 1;
    else unspecified += 1;
  }
  if (unspecified > 0) return directed === 0 && undirected === 0 ? "unspecified" : "mixed";
  if (directed > 0 && undirected === 0) return "directed";
  if (undirected > 0 && directed === 0) return "undirected";
  return "mixed";
}

function freezeFilters(filters: GraphFilters): GraphFilters {
  return normalizeGraphFilters(filters);
}

/**
 * Binds an exact semantic GraphSlice to an auditable scope. Render projections
 * are intentionally created later and cannot shrink algorithm input.
 */
export function createScopedGraphSlice(
  graphVersionId: string,
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
  filters: GraphFilters,
  truncated = false,
): ScopedGraphSlice {
  const nodeIds = Object.freeze(nodes.map((node) => node.id));
  const edgeIds = Object.freeze(edges.map((edge) => edge.id));
  const frozenFilters = freezeFilters(filters);
  const scopeHash = sha256Canonical({
    graphVersionId,
    nodeIds: [...nodeIds].sort(compareUnicodeCodePoints),
    edgeIds: [...edgeIds].sort(compareUnicodeCodePoints),
    filters: frozenFilters,
    truncated,
  });
  return Object.freeze({
    scope: Object.freeze({
      graphVersionId,
      nodeIds,
      edgeIds,
      nodeCount: nodeIds.length,
      edgeCount: edgeIds.length,
      scopeHash,
      truncated,
      filters: frozenFilters,
    }),
    slice: Object.freeze({
      nodes: Object.freeze([...nodes]),
      edges: Object.freeze([...edges]),
      nodeIds,
      edgeIds,
    }),
  });
}

export function createAnalysisScopeFromScene(
  scene: GraphScene,
  filters: GraphFilters,
): ScopedGraphSlice {
  if (scene.truncated) {
    throw new Error(
      "ANALYSIS_SCOPE_FROM_RENDER_PROJECTION：禁止将截断的渲染投影作为算法输入。",
    );
  }
  return createScopedGraphSlice(
    scene.graphVersionId,
    scene.nodes,
    scene.edges,
    filters,
    scene.truncated,
  );
}

function rounded(value: number): number {
  return Number(value.toFixed(6));
}

export function buildUndirectedAdjacency(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
): Adjacency {
  const adjacency: Adjacency = new Map(
    nodes.map((node) => [node.id, new Set<string>()]),
  );

  for (const edge of edges) {
    if (edge.source === edge.target) continue;
    const sourceNeighbours = adjacency.get(edge.source);
    const targetNeighbours = adjacency.get(edge.target);
    if (!sourceNeighbours || !targetNeighbours) continue;
    sourceNeighbours.add(edge.target);
    targetNeighbours.add(edge.source);
  }

  return adjacency;
}

function componentsFromAdjacency(adjacency: Adjacency): string[][] {
  const visited = new Set<string>();
  const components: string[][] = [];
  const orderedIds = [...adjacency.keys()].sort(compareUnicodeCodePoints);

  for (const start of orderedIds) {
    if (visited.has(start)) continue;
    const component: string[] = [];
    const queue = [start];
    visited.add(start);

    for (let cursor = 0; cursor < queue.length; cursor += 1) {
      const current = queue[cursor];
      component.push(current);
      const neighbours = [...(adjacency.get(current) ?? [])].sort(compareUnicodeCodePoints);
      for (const neighbour of neighbours) {
        if (!visited.has(neighbour)) {
          visited.add(neighbour);
          queue.push(neighbour);
        }
      }
    }

    components.push(component);
  }

  return components.sort((left, right) => {
    if (right.length !== left.length) return right.length - left.length;
    return compareUnicodeCodePoints(left[0] ?? "", right[0] ?? "");
  });
}

function computeIncidentDegrees(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
): Map<string, number> {
  const degree = new Map(nodes.map((node) => [node.id, 0]));
  for (const edge of edges) {
    if (!degree.has(edge.source) || !degree.has(edge.target)) continue;
    if (edge.source === edge.target) {
      degree.set(edge.source, degree.get(edge.source)! + 2);
    } else {
      degree.set(edge.source, degree.get(edge.source)! + 1);
      degree.set(edge.target, degree.get(edge.target)! + 1);
    }
  }
  return degree;
}

export function getConnectedComponents(graph: Pick<GraphVersion, "nodes" | "edges">): string[][] {
  return componentsFromAdjacency(buildUndirectedAdjacency(graph.nodes, graph.edges));
}

export function computeGraphSummary(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
): GraphSummary {
  const adjacency = buildUndirectedAdjacency(nodes, edges);
  const incidentDegrees = computeIncidentDegrees(nodes, edges);
  const degreeSum = [...incidentDegrees.values()].reduce((sum, degree) => sum + degree, 0);
  const validRelationshipCount = degreeSum / 2;
  const possibleRelationships = (nodes.length * (nodes.length - 1)) / 2;
  const components = componentsFromAdjacency(adjacency);

  return {
    nodeCount: nodes.length,
    edgeCount: edges.length,
    density: possibleRelationships === 0 ? 0 : rounded(validRelationshipCount / possibleRelationships),
    averageDegree: nodes.length === 0 ? 0 : rounded(degreeSum / nodes.length),
    connectedComponents: components.length,
    isolatedNodes: [...incidentDegrees.values()].filter((degree) => degree === 0).length,
  };
}

export function rankDegreeCentrality(
  graph: Pick<GraphVersion, "nodes" | "edges">,
  limit = Number.POSITIVE_INFINITY,
): DegreeRankEntry[] {
  const incidentDegrees = computeIncidentDegrees(graph.nodes, graph.edges);
  const normalizer = Math.max(1, graph.nodes.length - 1);

  return graph.nodes
    .map((node) => {
      const degree = incidentDegrees.get(node.id) ?? 0;
      return {
        nodeId: node.id,
        label: node.label,
        degree,
        normalizedScore: graph.nodes.length <= 1 ? 0 : rounded(degree / normalizer),
      };
    })
    .sort((left, right) => {
      if (right.degree !== left.degree) return right.degree - left.degree;
      return compareUnicodeCodePoints(left.nodeId, right.nodeId);
    })
    .slice(0, Math.max(0, limit));
}

/** Iterative Tarjan search; explicit frames avoid overflowing the browser stack on deep graphs. */
export function findArticulationPoints(
  graph: Pick<GraphVersion, "nodes" | "edges">,
): string[] {
  const adjacency = buildUndirectedAdjacency(graph.nodes, graph.edges);
  const discovery = new Map<string, number>();
  const low = new Map<string, number>();
  const parent = new Map<string, string | undefined>();
  const points = new Set<string>();
  let time = 0;

  interface Frame {
    readonly nodeId: string;
    readonly neighbours: readonly string[];
    nextIndex: number;
    childCount: number;
  }

  for (const nodeId of [...adjacency.keys()].sort(compareUnicodeCodePoints)) {
    if (discovery.has(nodeId)) continue;
    parent.set(nodeId, undefined);
    time += 1;
    discovery.set(nodeId, time);
    low.set(nodeId, time);
    const stack: Frame[] = [{
      nodeId,
      neighbours: [...(adjacency.get(nodeId) ?? [])].sort(compareUnicodeCodePoints),
      nextIndex: 0,
      childCount: 0,
    }];

    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      const neighbour = frame.neighbours[frame.nextIndex];
      if (neighbour !== undefined) {
        frame.nextIndex += 1;
        if (!discovery.has(neighbour)) {
          frame.childCount += 1;
          parent.set(neighbour, frame.nodeId);
          time += 1;
          discovery.set(neighbour, time);
          low.set(neighbour, time);
          stack.push({
            nodeId: neighbour,
            neighbours: [...(adjacency.get(neighbour) ?? [])].sort(compareUnicodeCodePoints),
            nextIndex: 0,
            childCount: 0,
          });
        } else if (neighbour !== parent.get(frame.nodeId)) {
          low.set(frame.nodeId, Math.min(low.get(frame.nodeId)!, discovery.get(neighbour)!));
        }
        continue;
      }

      stack.pop();
      const parentId = parent.get(frame.nodeId);
      if (parentId === undefined) {
        if (frame.childCount > 1) points.add(frame.nodeId);
        continue;
      }
      low.set(parentId, Math.min(low.get(parentId)!, low.get(frame.nodeId)!));
      if (parent.get(parentId) !== undefined && low.get(frame.nodeId)! >= discovery.get(parentId)!) {
        points.add(parentId);
      }
    }
  }

  return [...points].sort(compareUnicodeCodePoints);
}

export function runLocalAnalysis(graph: GraphVersion, task: AnalysisTask): AnalysisResult {
  switch (task) {
    case "overview":
      return {
        kind: "overview",
        summary: graph.summary,
        topDegree: rankDegreeCentrality(graph, 10),
        articulationPoints: findArticulationPoints(graph),
      };
    case "centrality":
      return { kind: "centrality", ranking: rankDegreeCentrality(graph) };
    case "bridge_detection":
      return { kind: "bridge_detection", articulationPoints: findArticulationPoints(graph) };
    case "community":
      {
        const result = computeLouvainCommunities(graph);
        return {
          kind: "community",
          communities: result.communities,
          assignments: result.assignments,
          modularity: result.modularity,
          message: `Louvain 识别到 ${result.communities.length} 个社区（模块度 ${result.modularity.toFixed(3)}）。`,
        };
      }
    case "link_prediction":
    case "node_role":
    case "similar_structure":
      return {
        kind: "unavailable",
        code: "GFM_CORE_NOT_CONNECTED",
        message: "该任务需要 GFM API；当前原型不会生成模拟研究结论。",
        requestedTask: task,
      };
  }
}
