import { describe, expect, it, vi } from "vitest";
import { GraphVisibilityController } from "./graphVisibilityController";

const nodes = [
  { id: "p", label: "人", type: "person", attributes: {} },
  { id: "o", label: "组织", type: "organization", attributes: {} },
  { id: "q", label: "另一人", type: "person", attributes: {} },
] as const;
const edges = [
  { id: "po", source: "p", target: "o", type: "member", timestamp: "2024-01-01", weight: 5, directed: true, attributes: {} },
  { id: "pq", source: "p", target: "q", type: "knows", timestamp: "2022-01-01", weight: 2, directed: false, attributes: {} },
] as const;

describe("GraphVisibilityController", () => {
  it("combines type, scene, viewport, and protected-node rules", () => {
    const controller = new GraphVisibilityController(nodes, edges);
    const result = controller.compute({
      filters: { nodeTypes: ["person"], edgeTypes: [] },
      sceneNodeIds: new Set(["p", "q"]),
      viewportNodeIds: new Set(["p"]),
      protectedNodeIds: new Set(["q"]),
    });
    expect(result.visibleNodeIds).toEqual(new Set(["p", "q"]));
    expect(result.visibleEdgeIds).toEqual(new Set(["pq"]));
  });

  it("returns only the visibility delta after the first computation", () => {
    const controller = new GraphVisibilityController(nodes, edges);
    const request = { filters: { nodeTypes: [], edgeTypes: [] } } as const;
    expect(controller.compute(request).changes).toEqual({});
    expect(controller.compute(request).changes).toEqual({});
  });

  it("resets to G6's visible scene baseline", () => {
    const controller = new GraphVisibilityController(nodes, edges);
    controller.compute({
      filters: { nodeTypes: [], edgeTypes: [], emptyReason: "direction_mismatch" },
    });
    controller.reset();
    expect(controller.compute({ filters: { nodeTypes: [], edgeTypes: [] } }).changes).toEqual({});
  });

  it("filters relation types and time ranges and removes non-incident fake isolates", () => {
    const controller = new GraphVisibilityController(nodes, edges);
    const result = controller.compute({
      filters: {
        nodeTypes: [],
        edgeTypes: ["member"],
        timeRange: { start: "2023-01-01" },
      },
    });
    expect(result.visibleNodeCount).toBe(2);
    expect(result.visibleNodeIds).toEqual(new Set(["p", "o"]));
    expect(result.visibleEdgeIds).toEqual(new Set(["po"]));
  });

  it("applies weight and direction filters to both relationships and incident nodes", () => {
    const controller = new GraphVisibilityController(nodes, edges);
    const result = controller.compute({
      filters: {
        nodeTypes: [],
        edgeTypes: [],
        minWeight: 4,
        maxWeight: 6,
        directed: true,
      },
    });
    expect(result.visibleNodeIds).toEqual(new Set(["p", "o"]));
    expect(result.visibleEdgeIds).toEqual(new Set(["po"]));
  });

  it("fails closed when an analysis filter contract marks the range empty", () => {
    const controller = new GraphVisibilityController(nodes, edges);
    const result = controller.compute({
      filters: {
        nodeTypes: [],
        edgeTypes: [],
        emptyReason: "direction_mismatch",
      },
    });
    expect(result.visibleNodeCount).toBe(0);
    expect(result.visibleEdgeCount).toBe(0);
  });

  it("applies one large visibility delta with one Canvas redraw", async () => {
    const manyNodes = Array.from({ length: 1_200 }, (_, index) => ({
      id: `n${index}`,
      label: `N${index}`,
      attributes: {},
    }));
    const controller = new GraphVisibilityController(manyNodes, []);
    const setElementVisibility = vi.fn(async (
      _changes: Record<string, "visible" | "hidden">,
      _animation: boolean,
    ) => undefined);
    const graph = {
      destroyed: false,
      setElementVisibility,
    };
    const result = await controller.apply(graph as never, {
      filters: { nodeTypes: [], edgeTypes: [], emptyReason: "direction_mismatch" },
    });
    expect(result.applyBatchCount).toBe(1);
    expect(setElementVisibility).toHaveBeenCalledTimes(1);
    expect(Object.keys(setElementVisibility.mock.calls[0]![0])).toHaveLength(1_200);
  });
});
