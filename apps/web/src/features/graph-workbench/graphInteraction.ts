import { Graph, type IElementEvent } from "@antv/g6";
import type { GraphNode } from "../../types/graph";
import type { GraphPreviewRuntimeMetrics } from "./graphPreviewTypes";

export function includesShiftKey(event: IElementEvent) {
  const candidate = event as IElementEvent & {
    shiftKey?: boolean;
    nativeEvent?: { shiftKey?: boolean };
  };
  return Boolean(candidate.shiftKey ?? candidate.nativeEvent?.shiftKey);
}

export function readDragTarget(
  graph: Graph,
  nodeId: string | undefined,
  container?: HTMLElement | null,
): GraphPreviewRuntimeMetrics["dragTarget"] {
  if (!nodeId || graph.destroyed) return undefined;
  try {
    // Element render bounds may lag one paint behind a running force layout.
    // The model position is authoritative for pointer hit-testing and stays in
    // the same canvas coordinate system used by getClientByCanvas().
    const position = graph.getElementPosition(nodeId);
    const client = graph.getClientByCanvas(position);
    const containerBounds = container?.getBoundingClientRect();
    const x = Number(client[0]) - (containerBounds?.left ?? 0);
    const y = Number(client[1]) - (containerBounds?.top ?? 0);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return undefined;
    return { nodeId, x, y };
  } catch {
    return undefined;
  }
}

export function findCentralDragTarget(
  graph: Graph,
  nodes: readonly GraphNode[],
  container: HTMLElement,
): GraphPreviewRuntimeMetrics["dragTarget"] {
  if (graph.destroyed || nodes.length === 0) return undefined;
  const containerBounds = container.getBoundingClientRect();
  const centerX = containerBounds.width / 2;
  const centerY = containerBounds.height / 2;
  let best: GraphPreviewRuntimeMetrics["dragTarget"];
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const node of nodes) {
    const candidate = readDragTarget(graph, node.id, container);
    if (!candidate) continue;
    if (
      candidate.x < 18
      || candidate.y < 18
      || candidate.x > containerBounds.width - 18
      || candidate.y > containerBounds.height - 18
    ) {
      continue;
    }
    const dx = candidate.x - centerX;
    const dy = candidate.y - centerY;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = candidate;
    }
  }
  return best;
}
