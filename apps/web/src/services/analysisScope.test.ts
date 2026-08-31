import { describe, expect, it } from "vitest";

import type { NormalizedIntent } from "../types/graph";
import { createScopedGraphSlice } from "./graphAlgorithms";
import { createGraphVersion } from "./graphImport";
import { LocalAnalysisExecutor } from "./localAnalysisExecutor";

const intent: NormalizedIntent = {
  kind: "analysis_request",
  normalizedText: "分析当前可见范围的中心性",
  task: "centrality",
  targets: [],
  confidence: 1,
  filters: {},
  meta: {
    schemaVersion: "1.1",
    source: "llm",
    requestId: "scope-request",
    warnings: [],
  },
};

describe("AnalysisScope", () => {
  it("runs the local algorithm on the exact hashed GraphSlice", async () => {
    const graph = createGraphVersion(
      "scope.json",
      [
        { id: "a", label: "A", attributes: {} },
        { id: "b", label: "B", attributes: {} },
        { id: "c", label: "C", attributes: {} },
      ],
      [
        { id: "ab", source: "a", target: "b", attributes: {} },
        { id: "bc", source: "b", target: "c", attributes: {} },
      ],
    );
    const scopedGraph = createScopedGraphSlice(
      graph.id,
      graph.nodes.slice(0, 2),
      graph.edges.slice(0, 1),
      { nodeTypes: [], edgeTypes: [] },
    );
    const executor = new LocalAnalysisExecutor([graph]);

    const run = await executor.createAnalysis({ graphVersionId: graph.id, intent, scopedGraph });

    expect(run.status).toBe("succeeded");
    expect(run.scope).toMatchObject({ nodeCount: 2, edgeCount: 1, scopeHash: scopedGraph.scope.scopeHash });
    expect(run.result?.kind).toBe("centrality");
    if (run.result?.kind === "centrality") {
      expect(run.result.ranking.map((entry) => entry.nodeId)).toEqual(["a", "b"]);
    }
  });

  it("rejects a tampered scope hash", async () => {
    const graph = createGraphVersion(
      "scope.json",
      [{ id: "a", label: "A", attributes: {} }],
      [],
    );
    const scopedGraph = createScopedGraphSlice(graph.id, graph.nodes, graph.edges, { nodeTypes: [], edgeTypes: [] });
    const executor = new LocalAnalysisExecutor([graph]);
    const tampered = {
      ...scopedGraph,
      scope: { ...scopedGraph.scope, scopeHash: "0".repeat(64) },
    };

    const run = await executor.createAnalysis({ graphVersionId: graph.id, intent, scopedGraph: tampered });

    expect(run.status).toBe("failed");
    expect(run.error).toContain("ANALYSIS_SCOPE_HASH_MISMATCH");
  });

  it("binds weight, time, type, and direction filters into the scope hash", () => {
    const graph = createGraphVersion(
      "scope-filtered.json",
      [
        { id: "a", label: "A", attributes: {} },
        { id: "b", label: "B", attributes: {} },
      ],
      [{ id: "ab", source: "a", target: "b", weight: 5, directed: true, attributes: {} }],
    );
    const base = {
      nodeTypes: ["人员"],
      edgeTypes: ["合作"],
      timeRange: { start: "2024", end: "2024" },
      minWeight: 4,
      maxWeight: 8,
      directed: true,
    } as const;
    const first = createScopedGraphSlice(graph.id, graph.nodes, graph.edges, base);
    const second = createScopedGraphSlice(graph.id, graph.nodes, graph.edges, { ...base, minWeight: 5 });

    expect(first.scope.filters).toEqual(base);
    expect(first.scope.scopeHash).not.toBe(second.scope.scopeHash);
  });

  it("canonicalises set-like filter order before hashing", () => {
    const graph = createGraphVersion(
      "scope-order.json",
      [
        { id: "a", label: "A", type: "人员", attributes: {} },
        { id: "b", label: "B", type: "组织", attributes: {} },
      ],
      [{ id: "ab", source: "a", target: "b", type: "合作", attributes: {} }],
    );
    const first = createScopedGraphSlice(graph.id, graph.nodes, graph.edges, {
      nodeTypes: ["组织", "人员"],
      edgeTypes: ["参与", "合作"],
    });
    const second = createScopedGraphSlice(graph.id, graph.nodes, graph.edges, {
      nodeTypes: ["人员", "组织", "人员"],
      edgeTypes: ["合作", "参与"],
    });

    expect(first.scope.scopeHash).toBe(second.scope.scopeHash);
    expect(first.scope.filters).toEqual({ nodeTypes: ["人员", "组织"], edgeTypes: ["参与", "合作"] });
  });
});
