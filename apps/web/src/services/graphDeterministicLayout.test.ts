import { describe, expect, it } from "vitest";
import {
  deterministicGraphInitialPositions,
  graphTopologyKey,
} from "./graphDeterministicLayout";

const nodes = [
  { id: "a", label: "A" },
  { id: "b", label: "B" },
  { id: "c", label: "C" },
  { id: "d", label: "D" },
];

describe("deterministic graph layout", () => {
  it("uses an order-independent topology key", () => {
    const edges = [
      { id: "e1", source: "a", target: "b" },
      { id: "e2", source: "b", target: "c" },
    ];
    expect(graphTopologyKey(nodes, edges)).toBe(
      graphTopologyKey([...nodes].reverse(), [...edges].reverse()),
    );
    expect(graphTopologyKey(nodes, [...edges, { id: "e3", source: "c", target: "d" }]))
      .not.toBe(graphTopologyKey(nodes, edges));
  });

  it("returns stable, non-colliding seeded positions", () => {
    const edges = [{ id: "e1", source: "a", target: "b" }];
    const first = deterministicGraphInitialPositions(nodes, edges);
    const second = deterministicGraphInitialPositions(nodes, edges);
    expect([...first.entries()]).toEqual([...second.entries()]);
    expect(new Set([...first.values()].map((point) => `${point.x}:${point.y}`)).size)
      .toBe(nodes.length);
  });

  it("does not place sparse components on repeated rings or seven-row rails", () => {
    const sparseNodes = Array.from({ length: 120 }, (_, index) => ({ id: `node-${index}` }));
    const sparseEdges = Array.from({ length: 39 }, (_, index) => ({
      id: `edge-${index}`,
      source: `node-${index}`,
      target: `node-${index + 1}`,
    }));
    const positions = [...deterministicGraphInitialPositions(sparseNodes, sparseEdges).values()];
    const radiusFrequency = new Map<number, number>();
    for (const point of positions) {
      const radius = Math.round(Math.hypot(point.x - 500, (point.y - 350) / 0.78));
      radiusFrequency.set(radius, (radiusFrequency.get(radius) ?? 0) + 1);
    }
    expect(Math.max(...radiusFrequency.values())).toBeLessThanOrEqual(2);
    expect(new Set(positions.map((point) => Math.round(point.y))).size).toBeGreaterThan(80);
  });
});
