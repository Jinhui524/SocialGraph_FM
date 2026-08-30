import { describe, expect, it } from "vitest";
import {
  createBenchmarkGraphVersion,
  GRAPH_BENCHMARK_CASES,
} from "./graphBenchmarkData";

describe("graph benchmark generator", () => {
  it.each(Object.values(GRAPH_BENCHMARK_CASES))(
    "creates an exact, connected and deterministic $id case",
    (benchmarkCase) => {
      const first = createBenchmarkGraphVersion(benchmarkCase);
      const second = createBenchmarkGraphVersion(benchmarkCase);

      expect(first.nodes).toHaveLength(benchmarkCase.nodeCount);
      expect(first.edges).toHaveLength(benchmarkCase.edgeCount);
      expect(first.summary.connectedComponents).toBe(1);
      expect(first.nodes).toEqual(second.nodes);
      expect(first.edges).toEqual(second.edges);
      expect(new Set(first.edges.map((edge) => {
        const pair = [edge.source, edge.target].sort();
        return pair.join(":");
      })).size).toBe(benchmarkCase.edgeCount);
      expect(first.edges.every((edge) => edge.source !== edge.target)).toBe(true);
      expect(first.edges.filter((edge) => edge.type === "社区协作")).toHaveLength(
        Math.floor(benchmarkCase.edgeCount * 0.8),
      );
      expect(first.edges.filter((edge) => edge.type === "跨区协调")).toHaveLength(
        benchmarkCase.edgeCount - Math.floor(benchmarkCase.edgeCount * 0.8),
      );
    },
  );
});
