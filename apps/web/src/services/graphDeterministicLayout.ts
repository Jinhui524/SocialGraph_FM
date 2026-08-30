import type { GraphEdge, GraphNode } from "../types/graph";

export interface GraphInitialPosition {
  readonly x: number;
  readonly y: number;
}

/** Stable identity for a topology, independent of API array order. */
export function graphTopologyKey(
  nodes: readonly Pick<GraphNode, "id">[],
  edges: readonly Pick<GraphEdge, "source" | "target" | "directed">[],
): string {
  const nodeIds = nodes.map((node) => node.id).sort(compareIds);
  const nodeSet = new Set(nodeIds);
  const edgeKeys = edges
    .filter((edge) => nodeSet.has(edge.source) && nodeSet.has(edge.target))
    .map((edge) => {
      const directed = edge.directed === true;
      const ordered = directed || compareIds(edge.source, edge.target) <= 0
        ? [edge.source, edge.target]
        : [edge.target, edge.source];
      return `${directed ? "d" : "u"}:${ordered[0]}:${ordered[1]}`;
    })
    .sort(compareIds);
  const canonical = `${nodeIds.join("\u0001")}|${edgeKeys.join("\u0001")}`;
  return `${nodeIds.length}:${edgeKeys.length}:${hashString(canonical).toString(16)}`;
}

function compareIds(left: string, right: string): number {
  return left.localeCompare(right, "zh-CN");
}

export function deterministicGraphInitialPositions(
  nodes: readonly Pick<GraphNode, "id">[],
  edges: readonly Pick<GraphEdge, "source" | "target" | "directed">[],
): ReadonlyMap<string, GraphInitialPosition> {
  // The old BFS/ring initializer made every breadth level a visible circle
  // and placed disconnected components in a vertical rail. A golden-angle
  // disk gives the seeded force pass an organic, order-independent start;
  // isolated nodes participate in the same field instead of forming a grid.
  void edges;
  const result = new Map<string, GraphInitialPosition>();
  const topology = graphTopologyKey(nodes, edges);
  const orderedIds = [...new Set(nodes.map((node) => node.id))].sort(compareIds);
  const count = orderedIds.length;
  if (count === 0) return result;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const radiusScale = Math.min(520, Math.max(150, 18 * Math.sqrt(count)));
  const centreX = 500;
  const centreY = 350;
  orderedIds.forEach((id, index) => {
    if (count === 1) {
      result.set(id, { x: centreX, y: centreY });
      return;
    }
    const fraction = Math.sqrt((index + 0.65) / count);
    const angle = index * goldenAngle;
    const wobble = ((hashString(`${topology}\u0000${id}`) % 1000) / 1000 - 0.5) * 0.12;
    result.set(id, {
      x: centreX + Math.cos(angle + wobble) * radiusScale * fraction,
      y: centreY + Math.sin(angle + wobble) * radiusScale * fraction * 0.78,
    });
  });
  return result;
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}
