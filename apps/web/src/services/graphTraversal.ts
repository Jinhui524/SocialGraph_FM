import type { GraphEdge, GraphPath, GraphSlice, GraphVersion } from "../types/graph";

interface NeighbourLink {
  readonly nodeId: string;
  readonly edgeId: string;
}

function compareIds(left: string, right: string): number {
  return left.localeCompare(right, "zh-CN");
}

function buildLinks(graph: Pick<GraphVersion, "nodes" | "edges">): Map<string, NeighbourLink[]> {
  const links = new Map(graph.nodes.map((node) => [node.id, [] as NeighbourLink[]]));
  for (const edge of graph.edges) {
    if (!links.has(edge.source) || !links.has(edge.target)) continue;
    if (edge.source === edge.target) continue;
    links.get(edge.source)!.push({ nodeId: edge.target, edgeId: edge.id });
    links.get(edge.target)!.push({ nodeId: edge.source, edgeId: edge.id });
  }
  for (const neighbours of links.values()) {
    neighbours.sort((left, right) => compareIds(left.nodeId, right.nodeId) || compareIds(left.edgeId, right.edgeId));
  }
  return links;
}

function inducedSlice(
  graph: Pick<GraphVersion, "nodes" | "edges">,
  selectedIds: ReadonlySet<string>,
): GraphSlice {
  const nodes = graph.nodes
    .filter((node) => selectedIds.has(node.id))
    .sort((left, right) => compareIds(left.id, right.id));
  const edges = graph.edges
    .filter((edge) => selectedIds.has(edge.source) && selectedIds.has(edge.target))
    .sort((left, right) => compareIds(left.id, right.id));
  return Object.freeze({
    nodes: Object.freeze(nodes),
    edges: Object.freeze(edges),
    nodeIds: Object.freeze(nodes.map((node) => node.id)),
  });
}

/** Deterministic multi-source BFS over the undirected projection. */
export function extractLocalSubgraph(
  graph: Pick<GraphVersion, "nodes" | "edges">,
  requestedFocusIds: readonly string[],
  requestedDepth: 1 | 2 | 3,
): GraphSlice {
  const existingIds = new Set(graph.nodes.map((node) => node.id));
  const focusIds = [...new Set(requestedFocusIds)]
    .filter((nodeId) => existingIds.has(nodeId))
    .sort(compareIds);
  if (focusIds.length === 0) return inducedSlice(graph, new Set());

  const links = buildLinks(graph);
  const distances = new Map<string, number>(focusIds.map((nodeId) => [nodeId, 0]));
  const queue = [...focusIds];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    const current = queue[cursor];
    const distance = distances.get(current)!;
    if (distance >= requestedDepth) continue;
    for (const link of links.get(current) ?? []) {
      if (distances.has(link.nodeId)) continue;
      distances.set(link.nodeId, distance + 1);
      queue.push(link.nodeId);
    }
  }

  return inducedSlice(graph, new Set(distances.keys()));
}

/** Returns one deterministic, unweighted shortest path, or null if disconnected. */
export function findShortestPath(
  graph: Pick<GraphVersion, "nodes" | "edges">,
  sourceId: string,
  targetId: string,
): GraphPath | null {
  const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
  if (!nodeById.has(sourceId) || !nodeById.has(targetId)) return null;
  if (sourceId === targetId) {
    const node = nodeById.get(sourceId)!;
    return Object.freeze({
      sourceId,
      targetId,
      nodes: Object.freeze([node]),
      edges: Object.freeze([]),
      nodeIds: Object.freeze([sourceId]),
      edgeIds: Object.freeze([]),
    });
  }

  const links = buildLinks(graph);
  const queue = [sourceId];
  const visited = new Set([sourceId]);
  const parent = new Map<string, { nodeId: string; edgeId: string }>();

  for (let cursor = 0; cursor < queue.length && !visited.has(targetId); cursor += 1) {
    const current = queue[cursor];
    for (const link of links.get(current) ?? []) {
      if (visited.has(link.nodeId)) continue;
      visited.add(link.nodeId);
      parent.set(link.nodeId, { nodeId: current, edgeId: link.edgeId });
      queue.push(link.nodeId);
      if (link.nodeId === targetId) break;
    }
  }
  if (!visited.has(targetId)) return null;

  const nodeIds = [targetId];
  const edgeIds: string[] = [];
  let current = targetId;
  while (current !== sourceId) {
    const previous = parent.get(current);
    if (!previous) return null;
    edgeIds.push(previous.edgeId);
    nodeIds.push(previous.nodeId);
    current = previous.nodeId;
  }
  nodeIds.reverse();
  edgeIds.reverse();

  const edgeById = new Map<string, GraphEdge>();
  for (const edge of graph.edges) if (!edgeById.has(edge.id)) edgeById.set(edge.id, edge);
  return Object.freeze({
    sourceId,
    targetId,
    nodes: Object.freeze(nodeIds.map((nodeId) => nodeById.get(nodeId)!)),
    edges: Object.freeze(edgeIds.map((edgeId) => edgeById.get(edgeId)!).filter(Boolean)),
    nodeIds: Object.freeze(nodeIds),
    edgeIds: Object.freeze(edgeIds),
  });
}
