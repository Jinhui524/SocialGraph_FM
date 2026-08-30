import { describe, expect, it } from "vitest";

import type { GraphEdge, GraphNode, NormalizedIntent } from "../types/graph";
import { applyPreparedAnalysisFilters, prepareAnalysisFilters } from "./analysisFilters";
import { createAnalysisScopeFromScene, createScopedGraphSlice } from "./graphAlgorithms";
import { createGraphVersion } from "./graphImport";
import { buildGraphScene, buildSemanticGraphSlice, createDefaultGraphViewState } from "./graphScene";
import { applyViewCommand } from "./viewCommand";

function node(id: string, type: string): GraphNode {
  return { id, label: id.toUpperCase(), type, attributes: {} };
}

function edge(
  id: string,
  source: string,
  target: string,
  type: string,
  weight: number,
  timestamp: string,
  directed?: boolean,
): GraphEdge {
  return { id, source, target, type, weight, timestamp, ...(directed === undefined ? {} : { directed }), attributes: {} };
}

function intent(
  filters: NormalizedIntent["filters"],
  timeRange?: NormalizedIntent["timeRange"],
): NormalizedIntent {
  return {
    kind: "analysis_request",
    normalizedText: "分析 2024 年人员合作关系",
    task: "centrality",
    targets: [],
    confidence: 1,
    ...(timeRange ? { timeRange } : {}),
    filters,
    meta: {
      schemaVersion: "1.1",
      source: "llm",
      requestId: "analysis-filter-test",
      warnings: [],
    },
  };
}

describe("analysis filter execution contract", () => {
  it("uses one exact slice for intent filters, rendered scene, and scope hash", () => {
    const graph = createGraphVersion(
      "filtered.json",
      [node("a", "人员"), node("b", "人员"), node("c", "组织"), node("d", "人员")],
      [
        edge("ab", "a", "b", "合作", 5, "2024-06-01", true),
        edge("ac", "a", "c", "合作", 7, "2024-05-01", true),
        edge("bd-friend", "b", "d", "朋友", 6, "2024-04-01", true),
        edge("ad-low", "a", "d", "合作", 3, "2024-03-01", true),
        edge("bd-old", "b", "d", "合作", 8, "2023-12-31", true),
      ],
    );
    const normalized = intent({
      nodeType: "人员",
      edgeType: "合作",
      minWeight: 4,
      maxWeight: 7,
      directed: true,
      component: 1,
    }, { start: "2024", end: "2024" });
    const prepared = prepareAnalysisFilters(graph, normalized);
    const applied = applyViewCommand(
      graph,
      createDefaultGraphViewState(graph.id),
      prepared.command!,
    );
    const filters = applyPreparedAnalysisFilters(applied.nextState.filters, prepared);
    const executableState = { ...applied.nextState, filters };
    const semantic = buildSemanticGraphSlice(graph, { viewState: executableState });
    const scene = buildGraphScene(graph, { viewState: executableState });
    const scoped = createScopedGraphSlice(
      graph.id,
      semantic.slice.nodes,
      semantic.slice.edges,
      semantic.filters,
    );

    expect(prepared.warnings).toContain("component 筛选尚未实现安全的确定性语义，已从本次执行合同中移除。");
    expect(filters).toEqual({
      nodeTypes: ["人员"],
      edgeTypes: ["合作"],
      timeRange: { start: "2024", end: "2024" },
      minWeight: 4,
      maxWeight: 7,
      directed: true,
    });
    expect(scene.nodes.map((entry) => entry.id)).toEqual(["a", "b"]);
    expect(scene.edges.map((entry) => entry.id)).toEqual(["ab"]);
    expect(scoped.slice.nodeIds).toEqual(scene.nodes.map((entry) => entry.id));
    expect(scoped.slice.edgeIds).toEqual(scene.edges.map((entry) => entry.id));
    expect(scoped.scope).toMatchObject({
      nodeCount: 2,
      edgeCount: 1,
      filters,
    });
  });

  it("keeps algorithm scope complete when the render projection is capped", () => {
    const graph = createGraphVersion(
      "projected.json",
      [node("a", "人员"), node("b", "人员"), node("c", "人员"), node("d", "人员")],
      [
        edge("ab", "a", "b", "合作", 1, "2024", false),
        edge("bc", "b", "c", "合作", 1, "2024", false),
        edge("cd", "c", "d", "合作", 1, "2024", false),
      ],
    );
    const viewState = createDefaultGraphViewState(graph.id);
    const semantic = buildSemanticGraphSlice(graph, { viewState });
    const scope = createScopedGraphSlice(
      graph.id,
      semantic.slice.nodes,
      semantic.slice.edges,
      semantic.filters,
    );
    const scene = buildGraphScene(graph, { viewState, maxNodes: 2, maxEdges: 1 });

    expect(scope.scope).toMatchObject({ nodeCount: 4, edgeCount: 3, truncated: false });
    expect(scene).toMatchObject({ visibleNodeCount: 2, visibleEdgeCount: 1, truncated: true });
    expect(scene.projectionHash).toHaveLength(64);
    expect(() => createAnalysisScopeFromScene(scene, viewState.filters))
      .toThrow("ANALYSIS_SCOPE_FROM_RENDER_PROJECTION");
  });

  it("fails closed when requested direction contradicts GraphVersion metadata", () => {
    const graph = createGraphVersion(
      "directed.json",
      [node("a", "人员"), node("b", "人员")],
      [edge("ab", "a", "b", "合作", 1, "2024", true)],
    );
    const prepared = prepareAnalysisFilters(graph, intent({ directed: false }));
    const filters = applyPreparedAnalysisFilters(createDefaultGraphViewState(graph.id).filters, prepared);
    const scene = buildGraphScene(graph, {
      viewState: { ...createDefaultGraphViewState(graph.id), filters },
    });

    expect(prepared.emptyReason).toBe("direction_mismatch");
    expect(prepared.warnings.join(" ")).toContain("方向元数据不一致");
    expect(scene.visibleNodeCount).toBe(0);
    expect(scene.visibleEdgeCount).toBe(0);
  });

  it("fails closed when direction metadata cannot be verified", () => {
    const graph = createGraphVersion(
      "unknown-direction.json",
      [node("a", "人员"), node("b", "人员")],
      [edge("ab", "a", "b", "合作", 1, "2024")],
    );
    const prepared = prepareAnalysisFilters(graph, intent({ directed: true }));

    expect(graph.metadata?.directedness).toBe("unspecified");
    expect(prepared.emptyReason).toBe("direction_unknown");
    expect(prepared.warnings.join(" ")).toContain("没有可验证的方向元数据");
  });

  it("fails closed for an inverted weight interval", () => {
    const graph = createGraphVersion(
      "weight.json",
      [node("a", "人员"), node("b", "人员")],
      [edge("ab", "a", "b", "合作", 5, "2024", false)],
    );
    const prepared = prepareAnalysisFilters(graph, intent({ minWeight: 10, maxWeight: 2 }));

    expect(prepared.emptyReason).toBe("invalid_weight_range");
    expect(prepared.warnings.join(" ")).toContain("minWeight 大于 maxWeight");
  });

  it("does not silently discard a persisted fail-closed condition", () => {
    const directionFailure = applyPreparedAnalysisFilters({
      nodeTypes: [],
      edgeTypes: [],
      directed: false,
      emptyReason: "direction_mismatch",
    }, { warnings: [] });
    expect(directionFailure.emptyReason).toBe("direction_mismatch");

    const stillInverted = applyPreparedAnalysisFilters({
      nodeTypes: [],
      edgeTypes: [],
      minWeight: 10,
      maxWeight: 2,
      emptyReason: "invalid_weight_range",
    }, { minWeight: 5, warnings: [] });
    expect(stillInverted).toMatchObject({ minWeight: 5, maxWeight: 2, emptyReason: "invalid_weight_range" });

    const recovered = applyPreparedAnalysisFilters(directionFailure, {
      directed: true,
      warnings: [],
    });
    expect(recovered).toMatchObject({ directed: true });
    expect(recovered).not.toHaveProperty("emptyReason");
  });
});
