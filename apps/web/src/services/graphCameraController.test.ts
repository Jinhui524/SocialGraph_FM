import { describe, expect, it, vi } from "vitest";
import {
  GraphCameraController,
  combineElementBounds,
  combineElementPositionBounds,
} from "./graphCameraController";

function cameraGraph() {
  let zoom = 1;
  let position: [number, number] = [0, 0];
  const bounds = new Map([
    ["a", { min: [0, 0, 0], max: [100, 80, 0] }],
    ["b", { min: [1_700, 520, 0], max: [1_800, 600, 0] }],
  ]);
  return {
    destroyed: false,
    getElementRenderBounds(id: string) {
      const value = bounds.get(id);
      if (!value) throw new Error("missing");
      return value;
    },
    getElementPosition(id: string) {
      const value = bounds.get(id);
      if (!value) throw new Error("missing");
      return [
        (value.min[0] + value.max[0]) / 2,
        (value.min[1] + value.max[1]) / 2,
      ] as [number, number];
    },
    getSize: () => [1_000, 700] as [number, number],
    getViewportByCanvas: ([x, y]: [number, number]) =>
      [x * zoom + position[0], y * zoom + position[1]] as [number, number],
    zoomTo: vi.fn(async (next: number) => {
      zoom = next;
    }),
    translateBy: vi.fn(async (delta: [number, number]) => {
      position = [position[0] + delta[0], position[1] + delta[1]];
    }),
    read: () => ({ zoom, position }),
  };
}

function tallCameraGraph() {
  let zoom = 1;
  let position: [number, number] = [0, 0];
  const bounds = { min: [0, 0, 0], max: [2_000, 8_000, 0] };
  return {
    destroyed: false,
    getElementRenderBounds: () => bounds,
    getElementPosition: (id: string) =>
      id === "top"
        ? [1_000, 0] as [number, number]
        : [1_000, 8_000] as [number, number],
    getSize: () => [1_000, 700] as [number, number],
    getViewportByCanvas: ([x, y]: [number, number]) =>
      [x * zoom + position[0], y * zoom + position[1]] as [number, number],
    zoomTo: vi.fn(async (next: number) => {
      zoom = next;
    }),
    translateBy: vi.fn(async (delta: [number, number]) => {
      position = [position[0] + delta[0], position[1] + delta[1]];
    }),
    read: () => ({ zoom, position }),
  };
}

describe("GraphCameraController", () => {
  it("combines public element render bounds", () => {
    const graph = cameraGraph();
    expect(combineElementBounds(graph as never, ["a", "missing", "b"])).toEqual({
      minX: 0,
      minY: 0,
      maxX: 1_800,
      maxY: 600,
    });
  });

  it("uses stable element positions for fit extents", () => {
    const graph = cameraGraph();
    expect(combineElementPositionBounds(graph as never, ["a", "missing", "b"]))
      .toEqual({
        minX: 18,
        minY: 8,
        maxX: 1_782,
        maxY: 592,
      });
  });

  it("computes an absolute fit with a 48px safe area", async () => {
    const graph = cameraGraph();
    const controller = new GraphCameraController(graph as never, {
      waitForStableFrames: async () => undefined,
    });
    const result = await controller.fit(["a", "b"]);
    expect(result).toMatchObject({ fitted: true, retries: 0, clipped: false });
    expect(result.zoom).toBeCloseTo(904 / 1_764, 6);
    expect(graph.zoomTo).toHaveBeenCalledTimes(1);
    expect(graph.translateBy).toHaveBeenCalledTimes(1);
    const { position } = graph.read();
    expect(position[0]).toBeCloseTo(500 - 900 * (904 / 1_764), 6);
    expect(position[1]).toBeCloseTo(350 - 300 * (904 / 1_764), 6);
  });

  it("does not mutate a destroyed or empty graph", async () => {
    const graph = cameraGraph();
    graph.destroyed = true;
    const controller = new GraphCameraController(graph as never, {
      waitForStableFrames: async () => undefined,
    });
    expect(await controller.fit(["a"])).toMatchObject({ fitted: false });
    expect(graph.zoomTo).not.toHaveBeenCalled();
  });

  it("re-centres through G6 after zooming instead of deriving camera translation", async () => {
    const graph = cameraGraph();
    const controller = new GraphCameraController(graph as never, {
      waitForStableFrames: async () => undefined,
    });

    const result = await controller.fit(["a", "b"]);

    expect(result.clipped).toBe(false);
    expect(graph.zoomTo).toHaveBeenCalledWith(
      expect.any(Number),
      false,
      [500, 350],
    );
    expect(graph.translateBy).toHaveBeenCalledWith(expect.any(Array), false);
    const { zoom, position } = graph.read();
    expect(18 * zoom + position[0]).toBeCloseTo(48, 6);
    expect(1_782 * zoom + position[0]).toBeCloseTo(952, 6);
    expect(8 * zoom + position[1]).toBeGreaterThanOrEqual(47.999_999);
    expect(592 * zoom + position[1]).toBeLessThanOrEqual(652.000_001);
  });

  it("fits a tall large-graph layout below the former 0.12 zoom floor", async () => {
    const graph = tallCameraGraph();
    const controller = new GraphCameraController(graph as never, {
      waitForStableFrames: async () => undefined,
    });

    const result = await controller.fit(["top", "bottom"]);

    expect(result).toMatchObject({ fitted: true, clipped: false });
    expect(result.zoom).toBeCloseTo(604 / 8_064, 6);
    expect(result.zoom).toBeLessThan(0.12);
    const { zoom, position } = graph.read();
    expect(-32 * zoom + position[1]).toBeCloseTo(48, 6);
    expect(8_032 * zoom + position[1]).toBeCloseTo(652, 6);
  });

  it("uses CSS viewport dimensions instead of a DPR-scaled backing store", async () => {
    const graph = cameraGraph();
    graph.getSize = () => [2_000, 1_400];
    const controller = new GraphCameraController(graph as never, {
      waitForStableFrames: async () => undefined,
      getViewportSize: () => [1_000, 700],
    });

    const result = await controller.fit(["a", "b"]);

    expect(result).toMatchObject({ fitted: true, clipped: false });
    const { zoom, position } = graph.read();
    expect(position[0]).toBeCloseTo(500 - 900 * zoom, 6);
    expect(position[1]).toBeCloseTo(350 - 300 * zoom, 6);
    expect(graph.zoomTo).toHaveBeenCalledWith(zoom, false, [500, 350]);
  });

  it("centres a searched anchor while deriving zoom from its neighbourhood", async () => {
    const graph = cameraGraph();
    const controller = new GraphCameraController(graph as never, {
      waitForStableFrames: async () => undefined,
    });

    const result = await controller.focus(["a", "b"], {
      anchorElementId: "a",
      minZoom: 0.72,
      maxZoom: 1.35,
    });

    expect(result).toMatchObject({ fitted: true, zoom: 0.72 });
    const { zoom, position } = graph.read();
    expect(50 * zoom + position[0]).toBeCloseTo(500, 6);
    expect(40 * zoom + position[1]).toBeCloseTo(350, 6);
    expect(controller.hasVisibleElement(["a"])).toBe(true);
  });
});
