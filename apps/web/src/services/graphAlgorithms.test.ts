import { describe, expect, it } from "vitest";

import type { GraphEdge, GraphNode } from "../types/graph";
import { createGraphVersion } from "./graphImport";
import {
  computeGraphSummary,
  findArticulationPoints,
  getConnectedComponents,
  rankDegreeCentrality,
} from "./graphAlgorithms";

function graphFixture() {
  const nodes: GraphNode[] = ["a", "b", "c", "d", "isolated"].map((id) => ({
    id,
    label: id.toUpperCase(),
    attributes: {},
  }));
  const edges: GraphEdge[] = [
    { id: "ab", source: "a", target: "b", attributes: {} },
    { id: "bc", source: "b", target: "c", attributes: {} },
    { id: "cd", source: "c", target: "d", attributes: {} },
  ];
  return createGraphVersion("fixture.json", nodes, edges);
}

describe("graph algorithms", () => {
  it("computes summary and connected components including isolated nodes", () => {
    const graph = graphFixture();

    expect(computeGraphSummary(graph.nodes, graph.edges)).toEqual({
      nodeCount: 5,
      edgeCount: 3,
      density: 0.3,
      averageDegree: 1.2,
      connectedComponents: 2,
      isolatedNodes: 1,
    });
    expect(getConnectedComponents(graph)).toEqual([["a", "b", "c", "d"], ["isolated"]]);
  });

  it("ranks degree centrality deterministically", () => {
    const ranking = rankDegreeCentrality(graphFixture());

    expect(ranking.slice(0, 2).map(({ nodeId, degree }) => ({ nodeId, degree }))).toEqual([
      { nodeId: "b", degree: 2 },
      { nodeId: "c", degree: 2 },
    ]);
    expect(ranking[0].normalizedScore).toBe(0.5);
  });

  it("finds articulation points with Tarjan's algorithm", () => {
    expect(findArticulationPoints(graphFixture())).toEqual(["b", "c"]);
  });

  it("handles a 20,000-node chain without recursive stack overflow", () => {
    const count = 20_000;
    const nodes: GraphNode[] = Array.from({ length: count }, (_, index) => ({
      id: `n${String(index).padStart(5, "0")}`,
      label: `N${index}`,
      attributes: {},
    }));
    const edges: GraphEdge[] = Array.from({ length: count - 1 }, (_, index) => ({
      id: `e${String(index).padStart(5, "0")}`,
      source: nodes[index].id,
      target: nodes[index + 1].id,
      attributes: {},
    }));

    const points = findArticulationPoints({ nodes, edges });

    expect(points).toHaveLength(count - 2);
    expect(points[0]).toBe("n00001");
    expect(points.at(-1)).toBe("n19998");
  }, 15_000);
});
