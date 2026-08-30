import { computeGraphSummary } from "./graphAlgorithms";
import { compareUnicodeCodePoints } from "./graphIdentity";
import type { GraphEdge, GraphVersion } from "../types/graph";

export type GovernanceGraphLens = "risk" | "community" | "relations" | "router";

export type GovernanceProjectionPreset = "overview" | "relation" | "evidence" | "groups";

export interface GovernanceProjectionSpec {
  readonly preset: GovernanceProjectionPreset;
  readonly nodeBudget: number;
  readonly edgeBudget: number;
  readonly groupBudget?: number;
}

const OVERVIEW_SPEC: GovernanceProjectionSpec = Object.freeze({
  preset: "overview",
  nodeBudget: 120,
  edgeBudget: 240,
});
const RELATION_SPEC: GovernanceProjectionSpec = Object.freeze({
  preset: "relation",
  nodeBudget: 80,
  edgeBudget: 160,
});
const EVIDENCE_SPEC: GovernanceProjectionSpec = Object.freeze({
  preset: "evidence",
  nodeBudget: 60,
  edgeBudget: 120,
});
const GROUPS_SPEC: GovernanceProjectionSpec = Object.freeze({
  preset: "groups",
  nodeBudget: 120,
  edgeBudget: 240,
  groupBudget: 12,
});

export function governanceProjectionSpec(
  lens: GovernanceGraphLens,
  hasEvidenceFocus: boolean,
): GovernanceProjectionSpec {
  if (hasEvidenceFocus) return EVIDENCE_SPEC;
  if (lens === "relations") return RELATION_SPEC;
  if (lens === "community") return GROUPS_SPEC;
  return OVERVIEW_SPEC;
}

function edgeWeight(edge: GraphEdge): number {
  return typeof edge.weight === "number" && Number.isFinite(edge.weight) ? edge.weight : 0;
}

function connectedComponents(graph: GraphVersion): readonly (readonly string[])[] {
  const adjacency = new Map(graph.nodes.map((node) => [node.id, new Set<string>()]));
  for (const edge of graph.edges) {
    adjacency.get(edge.source)?.add(edge.target);
    adjacency.get(edge.target)?.add(edge.source);
  }
  const remaining = new Set(adjacency.keys());
  const components: string[][] = [];
  while (remaining.size) {
    const start = [...remaining].sort(compareUnicodeCodePoints)[0];
    const queue = [start];
    const component: string[] = [];
    remaining.delete(start);
    for (let index = 0; index < queue.length; index += 1) {
      const current = queue[index];
      component.push(current);
      for (const neighbour of [...(adjacency.get(current) ?? [])].sort(compareUnicodeCodePoints)) {
        if (!remaining.delete(neighbour)) continue;
        queue.push(neighbour);
      }
    }
    components.push(component.sort(compareUnicodeCodePoints));
  }
  return components.sort((left, right) => right.length - left.length || compareUnicodeCodePoints(left[0], right[0]));
}

export function projectGovernanceGraph(
  graph: GraphVersion,
  spec: GovernanceProjectionSpec,
  preferredNodeIds: readonly string[] = [],
): GraphVersion {
  const components = spec.preset === "overview" ? connectedComponents(graph) : [];
  const overviewIsolatedIds = components.filter((component) => component.length === 1).map(([id]) => id);
  const overviewNeedsStructureFocus = overviewIsolatedIds.length > 3;
  if (!overviewNeedsStructureFocus && graph.nodes.length <= spec.nodeBudget && graph.edges.length <= spec.edgeBudget) return graph;

  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  const degree = new Map<string, number>();
  for (const edge of graph.edges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  }
  const preferred = [...new Set(preferredNodeIds)].filter((id) => nodeById.has(id));
  const preferredSet = new Set(preferred);
  const rankedIds = (allowed?: ReadonlySet<string>) => [
    ...preferred.filter((id) => !allowed || allowed.has(id)),
    ...graph.nodes
      .filter((node) => (!allowed || allowed.has(node.id)) && !preferredSet.has(node.id))
      .sort((left, right) => (degree.get(right.id) ?? 0) - (degree.get(left.id) ?? 0)
        || compareUnicodeCodePoints(left.id, right.id))
      .map((node) => node.id),
  ];
  const isolatedSamples = spec.preset === "overview" ? overviewIsolatedIds.slice(0, 3) : [];
  // A governance selection is atomic: the bounded context may shrink, but its
  // required endpoints may never be sampled away.
  const effectiveNodeBudget = Math.max(spec.nodeBudget, preferred.length);
  const selectedIds = spec.preset === "overview"
    ? [
      ...rankedIds(new Set(graph.nodes.map((node) => node.id).filter((id) => !overviewIsolatedIds.includes(id))))
        .slice(0, Math.max(0, effectiveNodeBudget - isolatedSamples.length)),
      ...isolatedSamples,
    ]
    : rankedIds().slice(0, effectiveNodeBudget);
  const selectedSet = new Set(selectedIds);
  const selectedEdges = graph.edges
    .filter((edge) => selectedSet.has(edge.source) && selectedSet.has(edge.target))
    .sort((left, right) => {
      const rightPreferred = Number(preferredSet.has(right.source)) + Number(preferredSet.has(right.target));
      const leftPreferred = Number(preferredSet.has(left.source)) + Number(preferredSet.has(left.target));
      return rightPreferred - leftPreferred
        || edgeWeight(right) - edgeWeight(left)
        || compareUnicodeCodePoints(left.id, right.id);
    })
    .slice(0, spec.edgeBudget);
  const selectedNodes = selectedIds.map((id) => nodeById.get(id)!);
  const originalNodeCount = graph.preview.originalNodeCount || graph.summary.nodeCount;
  const originalEdgeCount = graph.preview.originalEdgeCount || graph.summary.edgeCount;
  const projectedSummary = computeGraphSummary(selectedNodes, selectedEdges);
  return Object.freeze({
    ...graph,
    // This is a view over one immutable factual graph, not a new GraphVersion.
    // A stable id lets GraphPreview incrementally replace the bounded scene
    // without destroying the engine or restarting layout.
    id: graph.id,
    nodes: Object.freeze(selectedNodes),
    edges: Object.freeze(selectedEdges),
    summary: Object.freeze({
      ...projectedSummary,
      ...(spec.preset === "overview" ? { isolatedNodes: graph.summary.isolatedNodes } : {}),
    }),
    preview: Object.freeze({
      nodes: Object.freeze(selectedNodes),
      edges: Object.freeze(selectedEdges),
      truncated: true,
      originalNodeCount,
      originalEdgeCount,
    }),
    truncated: true,
  });
}

export function projectGovernanceSkeletonGraph(
  graph: GraphVersion,
  nodeBudget = 1_000,
): GraphVersion {
  const degree = new Map<string, number>();
  for (const edge of graph.edges) {
    degree.set(edge.source, (degree.get(edge.source) ?? 0) + 1);
    degree.set(edge.target, (degree.get(edge.target) ?? 0) + 1);
  }
  const selectedNodes = [...graph.nodes]
    .sort((left, right) => (degree.get(right.id) ?? 0) - (degree.get(left.id) ?? 0)
      || compareUnicodeCodePoints(left.id, right.id))
    .slice(0, nodeBudget);
  const selectedSet = new Set(selectedNodes.map((node) => node.id));
  const parent = new Map(selectedNodes.map((node) => [node.id, node.id]));
  const find = (id: string): string => {
    const next = parent.get(id) ?? id;
    if (next === id) return id;
    const root = find(next);
    parent.set(id, root);
    return root;
  };
  const selectedEdges: GraphEdge[] = [];
  for (const edge of [...graph.edges]
    .filter((item) => selectedSet.has(item.source) && selectedSet.has(item.target) && item.source !== item.target)
    .sort((left, right) => edgeWeight(right) - edgeWeight(left) || compareUnicodeCodePoints(left.id, right.id))) {
    const sourceRoot = find(edge.source);
    const targetRoot = find(edge.target);
    if (sourceRoot === targetRoot) continue;
    parent.set(targetRoot, sourceRoot);
    selectedEdges.push(edge);
  }
  const originalNodeCount = graph.preview.originalNodeCount || graph.summary.nodeCount;
  const originalEdgeCount = graph.preview.originalEdgeCount || graph.summary.edgeCount;
  const projectedSummary = computeGraphSummary(selectedNodes, selectedEdges);
  return Object.freeze({
    ...graph,
    id: `${graph.id}:readable:skeleton:${nodeBudget}`,
    nodes: Object.freeze(selectedNodes),
    edges: Object.freeze(selectedEdges),
    summary: Object.freeze({ ...projectedSummary, isolatedNodes: graph.summary.isolatedNodes }),
    preview: Object.freeze({
      nodes: Object.freeze(selectedNodes),
      edges: Object.freeze(selectedEdges),
      truncated: selectedNodes.length < graph.nodes.length || selectedEdges.length < graph.edges.length,
      originalNodeCount,
      originalEdgeCount,
    }),
    truncated: selectedNodes.length < graph.nodes.length || selectedEdges.length < graph.edges.length,
  });
}
