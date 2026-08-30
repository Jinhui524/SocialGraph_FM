import { describe, expect, it, vi } from "vitest";
import { GraphEvent } from "@antv/g6";
import { GraphPerformanceProbe } from "./graphPerformanceProbe";

describe("GraphPerformanceProbe", () => {
  it("records measured phases with context and metadata", async () => {
    const times = [10, 16, 20];
    const probe = new GraphPerformanceProbe("graph-a", "canvas", () => times.shift() ?? 20);
    await probe.measure("spatial_pick", async () => "hit", { count: 4 });
    expect(probe.snapshot()).toEqual([
      expect.objectContaining({
        phase: "spatial_pick",
        durationMs: 6,
        graphEpoch: "graph-a",
        renderer: "canvas",
        count: 4,
      }),
    ]);
  });

  it("attaches and detaches first draw lifecycle listeners", () => {
    const listeners = new Map<string, () => void>();
    const graph = {
      on: vi.fn((event: string, listener: () => void) => listeners.set(event, listener)),
      off: vi.fn((event: string) => listeners.delete(event)),
    };
    const times = [1, 7, 9];
    const probe = new GraphPerformanceProbe("graph-a", "canvas", () => times.shift() ?? 9);
    const detach = probe.attach(graph as never);
    listeners.get(GraphEvent.BEFORE_DRAW)?.();
    listeners.get(GraphEvent.AFTER_DRAW)?.();
    expect(probe.snapshot()[0]).toEqual(expect.objectContaining({ phase: "first_draw", durationMs: 6 }));
    detach();
    expect(graph.off).toHaveBeenCalledTimes(4);
  });

  it("keeps diagnostic telemetry bounded", () => {
    const probe = new GraphPerformanceProbe("graph-a", "canvas", () => 1);
    for (let index = 0; index < 300; index += 1) {
      probe.record("pointer_dispatch", index);
    }
    expect(probe.snapshot()).toHaveLength(GraphPerformanceProbe.maxSamplesPerPhase);
    expect(probe.snapshot()[0]?.durationMs).toBe(268);
  });
});
