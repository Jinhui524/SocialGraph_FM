import { describe, expect, it } from "vitest";

import type { AnalysisOverlay, GraphEdge, GraphNode, GraphViewState, ViewCommand } from "../types/graph";
import { computeLouvainCommunities } from "./graphCommunities";
import { createGraphVersion } from "./graphImport";
import { buildCommunityOverlay, buildDegreeOverlay, buildPathOverlay } from "./graphOverlays";
import {
  buildGraphScene,
  createDefaultGraphViewState,
  normalizeGraphViewState,
} from "./graphScene";
import { extractLocalSubgraph, findShortestPath } from "./graphTraversal";
import { createGraphWorkbenchViewState, reduceGraphView } from "./graphViewState";
import { runGraphTask } from "./graphWorkerRunner";
import { resolveViewTarget } from "./targetResolver";
import { applyViewCommand } from "./viewCommand";

function node(id: string, label = id, type?: string): GraphNode {
  return { id, label, ...(type ? { type } : {}), attributes: {} };
}

function edge(id: string, source: string, target: string, type = "合作"): GraphEdge {
  return { id, source, target, type, attributes: {} };
}

function workbenchGraph() {
  const nodes = [
    node("a", "张三", "人员"),
    node("b", "张三研究组", "组织"),
    node("c", "李四", "人员"),
    node("d", "王五", "人员"),
    node("e", "赵六", "人员"),
    node("f", "陈七", "人员"),
  ];
  const edges = [
    edge("ab", "a", "b"), edge("ac", "a", "c"), edge("bc", "b", "c"),
    edge("de", "d", "e"), edge("df", "d", "f"), edge("ef", "e", "f"),
    edge("cd", "c", "d", "交流"),
  ];
  return createGraphVersion("workbench.json", nodes, edges);
}

describe("graph traversal and target resolution", () => {
  it("extracts deterministic 1-3 hop local views", () => {
    const graph = workbenchGraph();
    expect(extractLocalSubgraph(graph, ["a"], 1).nodeIds).toEqual(["a", "b", "c"]);
    expect(extractLocalSubgraph(graph, ["a"], 2).nodeIds).toEqual(["a", "b", "c", "d"]);
    expect(extractLocalSubgraph(graph, ["a"], 3).nodeIds).toEqual(["a", "b", "c", "d", "e", "f"]);
  });

  it("finds a deterministic unweighted shortest path", () => {
    const path = findShortestPath(workbenchGraph(), "a", "f");
    expect(path?.nodeIds).toEqual(["a", "c", "d", "f"]);
    expect(path?.edgeIds).toEqual(["ac", "cd", "df"]);
  });

  it("distinguishes exact, unique substring, ambiguous and missing targets", () => {
    const graph = createGraphVersion("targets.json", [
      node("id-张三", "张三"),
      node("u2", "张三"),
      node("u3", "张三"),
      node("u4", "北区研究组"),
    ], []);
    expect(resolveViewTarget(graph, "id-张三")).toMatchObject({ status: "resolved", match: "id_exact" });
    expect(resolveViewTarget(graph, "研究组")).toMatchObject({ status: "resolved", nodeId: "u4", match: "unique_substring" });
    expect(resolveViewTarget(graph, "张三")).toEqual({
      status: "ambiguous",
      term: "张三",
      candidateNodeIds: ["id-张三", "u2", "u3"],
    });
    expect(resolveViewTarget(graph, "不存在")).toEqual({ status: "not_found", term: "不存在" });
  });
});

describe("community, overlays and render scenes", () => {
  it("keeps the filtered global graph visible while local/path tools await nodes", () => {
    const graph = workbenchGraph();
    const local = buildGraphScene(graph, {
      viewState: { ...createDefaultGraphViewState(graph.id), mode: "local" },
    });
    expect(local.status).toEqual({ kind: "awaiting_focus" });
    expect(local.visibleNodeCount).toBe(graph.nodes.length);

    const pathStart = buildGraphScene(graph, {
      viewState: {
        ...createDefaultGraphViewState(graph.id),
        mode: "path",
        pathEndpointIds: ["a"],
      },
    });
    expect(pathStart.status).toEqual({ kind: "awaiting_path_end", sourceId: "a" });
    expect(pathStart.visibleNodeCount).toBe(graph.nodes.length);
  });

  it("falls back to the filtered global graph when two nodes have no path", () => {
    const graph = createGraphVersion(
      "disconnected.json",
      [node("a"), node("b"), node("c"), node("d")],
      [edge("ab", "a", "b"), edge("cd", "c", "d")],
    );
    const scene = buildGraphScene(graph, {
      viewState: {
        ...createDefaultGraphViewState(graph.id),
        mode: "path",
        pathEndpointIds: ["a", "d"],
      },
    });
    expect(scene.status).toEqual({ kind: "no_path", sourceId: "a", targetId: "d" });
    expect(scene.visibleNodeCount).toBe(4);
    expect(scene.pathNodeIds).toEqual([]);
  });

  it("supports selecting nodes before or after activating local/path tools", () => {
    const graph = workbenchGraph();
    const initial = createGraphWorkbenchViewState(createDefaultGraphViewState(graph.id));

    const awaitingLocal = reduceGraphView(initial, { type: "activate_mode", mode: "local" });
    expect(awaitingLocal.interaction.tool).toBe("pick_local_focus");
    const committedLocal = reduceGraphView(awaitingLocal, { type: "select_node", nodeId: "a" });
    expect(committedLocal.viewState).toMatchObject({ mode: "local", focusNodeIds: ["a"] });

    const selectedFirst = reduceGraphView(initial, { type: "select_node", nodeId: "c" });
    const localAfterSelection = reduceGraphView(selectedFirst, { type: "activate_mode", mode: "local" });
    expect(localAfterSelection.viewState.focusNodeIds).toEqual(["c"]);

    const awaitingStart = reduceGraphView(initial, { type: "activate_mode", mode: "path" });
    const awaitingEnd = reduceGraphView(awaitingStart, { type: "select_node", nodeId: "a" });
    expect(awaitingEnd.interaction).toMatchObject({ tool: "pick_path_end", pendingPathStartId: "a" });
    const committedPath = reduceGraphView(awaitingEnd, { type: "select_node", nodeId: "f" });
    expect(committedPath.viewState.pathEndpointIds).toEqual(["a", "f"]);
    expect(committedPath.interaction.tool).toBe("browse");
  });

  it("always restores global mode and migrates legacy path focus safely", () => {
    const graph = workbenchGraph();
    const legacy = {
      ...createDefaultGraphViewState(graph.id),
      mode: "path" as const,
      focusNodeIds: ["a", "f"],
    } as GraphViewState & { pathEndpointIds?: readonly string[] };
    delete (legacy as { pathEndpointIds?: readonly string[] }).pathEndpointIds;
    const migrated = normalizeGraphViewState(graph.id, legacy);
    expect(migrated.focusNodeIds).toEqual([]);
    expect(migrated.pathEndpointIds).toEqual(["a", "f"]);

    const reset = reduceGraphView(createGraphWorkbenchViewState(migrated), {
      type: "activate_mode",
      mode: "global",
    });
    expect(reset.viewState).toMatchObject({ mode: "global", focusNodeIds: [], pathEndpointIds: [] });
    expect(reset.interaction).toEqual({ tool: "browse", selectedNodeId: null });
  });

  it("keeps canonical filters while changing tools, selection and view mode", () => {
    const graph = workbenchGraph();
    const initial = createGraphWorkbenchViewState(createDefaultGraphViewState(graph.id));
    const filtered = reduceGraphView(initial, {
      type: "update_view",
      patch: {
        filters: {
          nodeTypes: ["人员"],
          edgeTypes: ["合作"],
          timeRange: { start: "2022-01-01" },
        },
      },
    });
    const local = reduceGraphView(filtered, { type: "activate_mode", mode: "local" });
    const focused = reduceGraphView(local, { type: "select_node", nodeId: "a" });
    const restored = reduceGraphView(focused, { type: "activate_mode", mode: "global" });

    expect(restored.viewState.filters).toEqual({
      nodeTypes: ["人员"],
      edgeTypes: ["合作"],
      timeRange: { start: "2022-01-01" },
    });
  });

  it("runs seeded Louvain deterministically and returns real communities", () => {
    const graph = workbenchGraph();
    const first = computeLouvainCommunities(graph, "fixed-seed");
    const second = computeLouvainCommunities(graph, "fixed-seed");
    expect(first).toEqual(second);
    expect(first.communities).toHaveLength(2);
    expect(new Set(Object.values(first.assignments))).toEqual(new Set(["community-1", "community-2"]));
    expect(first.modularity).toBeGreaterThan(0);
  });

  it("builds overlays without changing immutable graph facts", () => {
    const graph = workbenchGraph();
    const before = JSON.stringify(graph);
    const community = computeLouvainCommunities(graph, "fixed-seed");
    const overlay = buildCommunityOverlay(graph, community);
    const path = findShortestPath(graph, "a", "f")!;
    const pathOverlay = buildPathOverlay(graph, path);
    expect(overlay.kind).toBe("community");
    expect(pathOverlay.edgeValues.cd).toBe(true);
    expect(JSON.stringify(graph)).toBe(before);
  });

  it("samples deterministically while preserving focus, path and overlay representatives", () => {
    const graph = workbenchGraph();
    const path = findShortestPath(graph, "a", "f")!;
    const state: GraphViewState = {
      ...createDefaultGraphViewState(graph.id),
      mode: "global",
      focusNodeIds: ["f"],
    };
    const overlay = buildCommunityOverlay(graph, computeLouvainCommunities(graph, "fixed-seed"));
    const scene = buildGraphScene(graph, { viewState: state, path, overlay, maxNodes: 4, maxEdges: 3 });
    expect(scene.truncated).toBe(true);
    expect(scene.nodes.map((entry) => entry.id)).toContain("f");
    expect(scene.pathNodeIds).toContain("f");
    expect(scene.visibleNodeCount).toBeLessThanOrEqual(4);
    expect(scene.visibleEdgeCount).toBeLessThanOrEqual(3);
    expect(buildGraphScene(graph, { viewState: state, path, overlay, maxNodes: 4, maxEdges: 3 })).toEqual(scene);
  });

  it("protects governance finding endpoints without adding candidate relations to graph facts", () => {
    const graph = workbenchGraph();
    const before = JSON.stringify(graph);
    const overlay: AnalysisOverlay = {
      id: "governance-overlay",
      graphVersionId: graph.id,
      kind: "governance",
      nodeValues: { a: "subject", f: "subject" },
      edgeValues: {},
      candidateEdges: [{ id: "candidate-af", sourceId: "a", targetId: "f", directed: false }],
      legend: { title: "治理", items: [] },
      provenance: { engine: "gfm_research", algorithm: "test" },
    };

    const scene = buildGraphScene(graph, {
      viewState: createDefaultGraphViewState(graph.id),
      overlay,
      maxNodes: 2,
      maxEdges: 1,
    });

    expect(scene.nodes.map((entry) => entry.id).sort()).toEqual(["a", "f"]);
    expect(scene.overlay?.candidateEdges).toEqual(overlay.candidateEdges);
    expect(graph.edges.some((entry) => entry.id === "candidate-af")).toBe(false);
    expect(JSON.stringify(graph)).toBe(before);
  });

  it("does not turn sampled non-isolates into unexplained isolated dots", () => {
    const graph = workbenchGraph();
    const scene = buildGraphScene(graph, {
      viewState: createDefaultGraphViewState(graph.id),
      maxNodes: 4,
      maxEdges: 3,
    });
    const sourceDegree = new Map(graph.nodes.map((node) => [node.id, 0]));
    for (const edge of graph.edges) {
      sourceDegree.set(edge.source, (sourceDegree.get(edge.source) ?? 0) + 1);
      sourceDegree.set(edge.target, (sourceDegree.get(edge.target) ?? 0) + 1);
    }
    const renderedEndpoints = new Set(
      scene.edges.flatMap((edge) => [edge.source, edge.target]),
    );
    for (const node of scene.nodes) {
      if ((sourceDegree.get(node.id) ?? 0) > 0) {
        expect(renderedEndpoints.has(node.id), node.id).toBe(true);
      }
    }
  });

  it("applies a view command through local validation without mutating current state", () => {
    const graph = workbenchGraph();
    const current = createDefaultGraphViewState(graph.id);
    const before = JSON.stringify(current);
    const command: ViewCommand = {
      mode: "local",
      focusTerms: ["张三"],
      depth: 2,
      nodeTypeTerms: ["人员"],
      edgeTypeTerms: ["合作"],
      overlay: "degree",
    };
    const result = applyViewCommand(graph, current, command);
    expect(result.nextState.mode).toBe("local");
    expect(result.nextState.focusNodeIds).toEqual(["a"]);
    expect(result.nextState.filters).toMatchObject({ nodeTypes: ["人员"], edgeTypes: ["合作"] });
    expect(result.requestedOverlay).toBe("degree");
    expect(JSON.stringify(current)).toBe(before);
  });

  it("stores natural-language path targets separately from local focus", () => {
    const graph = workbenchGraph();
    const result = applyViewCommand(graph, createDefaultGraphViewState(graph.id), {
      mode: "path",
      focusTerms: ["张三", "陈七"],
      nodeTypeTerms: [],
      edgeTypeTerms: [],
    });
    expect(result.nextState.mode).toBe("path");
    expect(result.nextState.focusNodeIds).toEqual([]);
    expect(result.nextState.pathEndpointIds).toEqual(["a", "f"]);
  });

  it("does not erase a committed view when a new command has too few targets", () => {
    const graph = workbenchGraph();
    const current: GraphViewState = {
      ...createDefaultGraphViewState(graph.id),
      mode: "path",
      pathEndpointIds: ["a", "f"],
    };
    const result = applyViewCommand(graph, current, {
      mode: "local",
      focusTerms: ["不存在"],
      nodeTypeTerms: [],
      edgeTypeTerms: [],
    });
    expect(result.nextState.mode).toBe("path");
    expect(result.nextState.pathEndpointIds).toEqual(["a", "f"]);
  });

  it("builds degree overlays from the full graph", () => {
    const overlay = buildDegreeOverlay(workbenchGraph());
    expect(overlay.nodeValues.c as number).toBeGreaterThan(overlay.nodeValues.a as number);
  });

  it("uses the typed direct fallback when workers are unavailable", async () => {
    const graph = workbenchGraph();
    const result = await runGraphTask(
      { id: "task-1", kind: "local_subgraph", graph, focusNodeIds: ["a"], depth: 1 },
      { preferWorker: false },
    );
    expect(result).toMatchObject({ nodeIds: ["a", "b", "c"] });
  });
});
