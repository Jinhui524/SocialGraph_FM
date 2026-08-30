import { describe, expect, it } from "vitest";
import { GraphAdjacencyIndex, GraphSpatialIndex } from "./graphSpatialIndex";

describe("GraphSpatialIndex", () => {
  it("returns only nearby candidates for picking", () => {
    const index = new GraphSpatialIndex(64);
    index.rebuild([
      { id: "a", x: 10, y: 10 },
      { id: "b", x: 40, y: 10 },
      { id: "far", x: 1_000, y: 1_000 },
    ]);
    const pick = index.nearest(12, 12, 24);
    expect(pick.id).toBe("a");
    expect(pick.candidateCount).toBe(1);
  });

  it("updates cells without leaving stale positions", () => {
    const index = new GraphSpatialIndex(64);
    index.update([{ id: "a", x: 0, y: 0 }]);
    index.update([{ id: "a", x: 200, y: 200 }]);
    expect(index.queryRect({ minX: -10, minY: -10, maxX: 10, maxY: 10 })).toEqual(new Set());
    expect(index.nearest(200, 200, 4).id).toBe("a");
  });

  it("matches a brute-force picking oracle for known nodes and blank points", () => {
    const index = new GraphSpatialIndex(32);
    index.rebuild(Array.from({ length: 240 }, (_, value) => ({
      id: `n${value}`,
      x: (value % 24) * 12,
      y: Math.floor(value / 24) * 12,
    })));
    const report = index.diagnosePicking(200, 1);
    expect(report).toMatchObject({ checked: 200, mismatches: 0 });
    expect(report.candidateP95).toBeLessThanOrEqual(9);
  });
});

describe("GraphAdjacencyIndex", () => {
  const index = new GraphAdjacencyIndex([
    { id: "ab", source: "a", target: "b" },
    { id: "bc", source: "b", target: "c" },
    { id: "cd", source: "c", target: "d" },
  ]);

  it("returns neighbours and incident edges without scanning every edge", () => {
    expect([...index.neighbours("b")]).toEqual(["a", "c"]);
    expect([...index.edgeIds("b")]).toEqual(["ab", "bc"]);
  });

  it("bounds a breadth-first local set", () => {
    expect(index.localNodeIds("a", 2, 10)).toEqual(["a", "b", "c"]);
    expect(index.localNodeIds("a", 3, 2)).toEqual(["a", "b"]);
  });
});
