import type { Graph, GraphData } from "@antv/g6";
import type { GraphEdge, GraphFilters, GraphNode } from "../types/graph";
import {
  graphEdgeMatchesFilters,
  graphFiltersHaveRelationshipConstraints,
  normalizeGraphFilters,
} from "./graphFilters";

export interface GraphVisibilityRequest {
  readonly filters: GraphFilters;
  /** Mode/depth/path slice. Undefined means every renderable node. */
  readonly sceneNodeIds?: ReadonlySet<string>;
  readonly sceneEdgeIds?: ReadonlySet<string>;
  /** Spatial viewport culling. Undefined disables viewport culling. */
  readonly viewportNodeIds?: ReadonlySet<string>;
  readonly protectedNodeIds?: ReadonlySet<string>;
}

export interface GraphVisibilityResult {
  readonly changes: Readonly<Record<string, "visible" | "hidden">>;
  readonly visibleNodeIds: ReadonlySet<string>;
  readonly visibleEdgeIds: ReadonlySet<string>;
  readonly visibleNodeCount: number;
  readonly visibleEdgeCount: number;
  readonly culledNodeCount: number;
  readonly culledEdgeCount: number;
  readonly computeMs?: number;
  readonly applyMs?: number;
  readonly applyBatchCount?: number;
}

const visibilityNow = () =>
  typeof performance === "undefined" ? Date.now() : performance.now();

/**
 * Combines semantic filters, view slicing, and viewport culling into one
 * visibility diff so independent effects cannot accidentally re-show nodes.
 */
export class GraphVisibilityController {
  private readonly nodeTypes = new Map<string, string>();
  private readonly edges: readonly GraphEdge[];
  private readonly previous = new Map<string, "visible" | "hidden">();

  constructor(nodes: readonly GraphNode[], edges: readonly GraphEdge[]) {
    for (const node of nodes) this.nodeTypes.set(node.id, node.type?.trim() || "未分类");
    this.edges = edges;
    // G6 creates and replaces scene elements as visible. Mirroring that real
    // renderer baseline prevents the first no-filter pass from redundantly
    // setting every node and edge to `visible` (and repainting the full Canvas).
    this.reset();
  }

  reset(): void {
    this.previous.clear();
    for (const id of this.nodeTypes.keys()) this.previous.set(id, "visible");
    for (const edge of this.edges) this.previous.set(edge.id, "visible");
  }

  compute(request: GraphVisibilityRequest): GraphVisibilityResult {
    const filters = normalizeGraphFilters(request.filters);
    const enabledNodeTypes = new Set(filters.nodeTypes);
    const enabledEdgeTypes = new Set(filters.edgeTypes);
    const protectedIds = request.protectedNodeIds ?? new Set<string>();
    const eligibleNodeIds = new Set<string>();
    const next = new Map<string, "visible" | "hidden">();

    for (const [id, type] of this.nodeTypes) {
      const typeVisible = enabledNodeTypes.size === 0 || enabledNodeTypes.has(type);
      const sceneVisible = !request.sceneNodeIds || request.sceneNodeIds.has(id);
      const viewportVisible =
        !request.viewportNodeIds || request.viewportNodeIds.has(id) || protectedIds.has(id);
      if (!filters.emptyReason && typeVisible && sceneVisible && viewportVisible) eligibleNodeIds.add(id);
    }

    const candidateEdgeIds = new Set<string>();
    const incidentNodeIds = new Set<string>();
    for (const edge of this.edges) {
      const edgeTypeVisible =
        enabledEdgeTypes.size === 0 || (edge.type ? enabledEdgeTypes.has(edge.type) : false);
      const visible = !filters.emptyReason &&
        (!request.sceneEdgeIds || request.sceneEdgeIds.has(edge.id)) &&
        edgeTypeVisible &&
        graphEdgeMatchesFilters(edge, filters) &&
        eligibleNodeIds.has(edge.source) &&
        eligibleNodeIds.has(edge.target);
      if (visible) {
        candidateEdgeIds.add(edge.id);
        incidentNodeIds.add(edge.source);
        incidentNodeIds.add(edge.target);
      }
    }

    const hasRelationshipFilter = graphFiltersHaveRelationshipConstraints(filters);
    const visibleNodeIds = new Set(
      [...eligibleNodeIds].filter((id) => !hasRelationshipFilter || incidentNodeIds.has(id)),
    );
    const visibleEdgeIds = new Set<string>();
    for (const edge of this.edges) {
      const visible = candidateEdgeIds.has(edge.id)
        && visibleNodeIds.has(edge.source)
        && visibleNodeIds.has(edge.target);
      next.set(edge.id, visible ? "visible" : "hidden");
      if (visible) visibleEdgeIds.add(edge.id);
    }
    for (const id of this.nodeTypes.keys()) next.set(id, visibleNodeIds.has(id) ? "visible" : "hidden");

    const changes: Record<string, "visible" | "hidden"> = {};
    for (const [id, visibility] of next) {
      if (this.previous.get(id) !== visibility) changes[id] = visibility;
    }
    this.previous.clear();
    for (const [id, visibility] of next) this.previous.set(id, visibility);

    return {
      changes,
      visibleNodeIds,
      visibleEdgeIds,
      visibleNodeCount: visibleNodeIds.size,
      visibleEdgeCount: visibleEdgeIds.size,
      culledNodeCount: this.nodeTypes.size - visibleNodeIds.size,
      culledEdgeCount: this.edges.length - visibleEdgeIds.size,
    };
  }

  async apply(
    graph: Graph,
    request: GraphVisibilityRequest,
    applyChanges?: (changes: Readonly<Record<string, "visible" | "hidden">>) => Promise<void>,
  ): Promise<GraphVisibilityResult> {
    const computeStartedAt = visibilityNow();
    const result = this.compute(request);
    const computeMs = visibilityNow() - computeStartedAt;
    const entries = Object.entries(result.changes);
    const applyStartedAt = visibilityNow();
    let applyBatchCount = 0;
    if (!graph.destroyed && entries.length > 0) {
      // G6 redraws the Canvas visibility stage for every call. Splitting one
      // logical delta into rAF batches therefore repeats the full redraw and is
      // substantially slower than one atomic visibility mutation.
      const changes = Object.fromEntries(entries);
      if (applyChanges) await applyChanges(changes);
      else await graph.setElementVisibility(changes, false);
      applyBatchCount = 1;
    }
    return {
      ...result,
      computeMs,
      applyMs: visibilityNow() - applyStartedAt,
      applyBatchCount,
    };
  }
}

/** Utility for tests and adapters that only expose G6 GraphData. */
export function graphDataIds(data: GraphData): {
  readonly nodeIds: ReadonlySet<string>;
  readonly edgeIds: ReadonlySet<string>;
} {
  return {
    nodeIds: new Set((data.nodes ?? []).map((node) => String(node.id))),
    edgeIds: new Set((data.edges ?? []).map((edge) => String(edge.id))),
  };
}
