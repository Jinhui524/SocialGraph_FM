import { describe, expect, it } from "vitest";

import type { GraphEdge, GraphNode } from "../types/graph";
import { filterGraphFacts, graphFilterConstraintCount } from "./graphFilters";

const nodes: readonly GraphNode[] = [
  { id: "a", label: "A", type: "人员", attributes: {} },
  { id: "b", label: "B", type: "人员", attributes: {} },
  { id: "c", label: "C", type: "组织", attributes: {} },
];
const edges: readonly GraphEdge[] = [
  { id: "ab", source: "a", target: "b", type: "合作", weight: 5, timestamp: "2024-06-01", directed: true, attributes: {} },
  { id: "ac", source: "a", target: "c", type: "参与", weight: 2, timestamp: "2023-01-01", directed: false, attributes: {} },
];

describe("graph semantic filters", () => {
  it("applies type, weight, time, and direction through one deterministic slice", () => {
    const result = filterGraphFacts({ nodes, edges }, {
      nodeTypes: ["人员"],
      edgeTypes: ["合作"],
      timeRange: { start: "2024", end: "2024" },
      minWeight: 4,
      directed: true,
    });

    expect(result.slice.nodeIds).toEqual(["a", "b"]);
    expect(result.slice.edgeIds).toEqual(["ab"]);
    expect(graphFilterConstraintCount(result.filters)).toBe(6);
  });

  it("fails closed for invalid dates instead of comparing strings", () => {
    const result = filterGraphFacts({ nodes, edges }, {
      nodeTypes: [],
      edgeTypes: [],
      timeRange: { start: "not-a-date", end: "2024" },
    });

    expect(result.filters.emptyReason).toBe("invalid_time_range");
    expect(result.slice.nodes).toEqual([]);
    expect(result.slice.edges).toEqual([]);
  });
});
