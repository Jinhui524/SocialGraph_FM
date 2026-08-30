import { describe, expect, it, vi } from "vitest";
import { LocalForceController, localForcePolicy } from "./localForceController";

class WorkerDouble {
  onmessage: ((event: MessageEvent) => void) | null = null;
  messages: unknown[] = [];
  postMessage(message: unknown): void { this.messages.push(message); }
  terminate = vi.fn();
}

describe("LocalForceController", () => {
  it("uses the bounded local-force policy and throttles pointer moves", () => {
    const worker = new WorkerDouble();
    let currentTime = 0;
    const controller = new LocalForceController({
      onFrame: vi.fn(),
      workerFactory: () => worker,
      now: () => currentTime,
    });
    const nodes = Array.from({ length: 1_200 }, (_, index) => ({
      id: `n${index}`, label: `N${index}`, attributes: {},
    }));
    controller.initialize("g1", nodes, [], new Map(), new Set(["n2"]));
    expect(worker.messages[0]).toMatchObject({
      type: "init",
      pinnedNodeIndices: new Uint32Array([2]),
    });
    expect(controller.dragStart("n0", 10, 20, nodes.length)).toBe(true);
    expect(worker.messages.at(-1)).toMatchObject({ depth: 1, limit: 24 });
    expect(controller.dragMove(11, 21)).toBe(true);
    currentTime = 10;
    expect(controller.dragMove(12, 22)).toBe(false);
    currentTime = 70;
    expect(controller.dragMove(13, 23)).toBe(true);
    controller.dragEnd({ x: 13, y: 23, pinned: false });
    expect(worker.messages.at(-1)).toMatchObject({
      type: "drag-end",
      x: 13,
      y: 23,
      pinned: false,
    });
  });

  it("uses two-hop relaxation for answer-scale graphs and bounded large-graph work", () => {
    expect(localForcePolicy(300)).toEqual({ depth: 2, limit: 64 });
    expect(localForcePolicy(301)).toEqual({ depth: 2, limit: 48 });
    expect(localForcePolicy(1_000)).toEqual({ depth: 2, limit: 48 });
    expect(localForcePolicy(1_001)).toEqual({ depth: 1, limit: 24 });
  });

  it("drops frames from a stale graph epoch", () => {
    const worker = new WorkerDouble();
    const onFrame = vi.fn();
    const controller = new LocalForceController({ onFrame, workerFactory: () => worker });
    controller.initialize("current", [], [], new Map());
    worker.onmessage?.({ data: { type: "frame", epoch: "old" } } as MessageEvent);
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("reports per-frame compute plus transfer latency instead of the cooling window", () => {
    const worker = new WorkerDouble();
    const onFrame = vi.fn();
    const controller = new LocalForceController({
      onFrame,
      workerFactory: () => worker,
      now: () => 20,
      wallClock: () => 1_020,
    });
    controller.initialize("current", [], [], new Map());
    worker.onmessage?.({
      data: {
        type: "frame",
        epoch: "current",
        sequence: 1,
        nodeIndices: new Uint32Array(),
        positions: new Float32Array(),
        alpha: 0.2,
        computeMs: 3,
        activeCount: 0,
        emittedAtEpochMs: 1_018,
      },
    } as MessageEvent);
    expect(onFrame).toHaveBeenCalledWith(expect.objectContaining({ roundTripMs: 5 }));
  });

  it("drops stale Worker frames instead of applying obsolete neighbour positions", () => {
    const worker = new WorkerDouble();
    const onFrame = vi.fn();
    const controller = new LocalForceController({
      onFrame,
      workerFactory: () => worker,
      wallClock: () => 2_000,
      maxFrameAgeMs: 120,
    });
    controller.initialize("current", [], [], new Map());
    worker.onmessage?.({
      data: {
        type: "frame",
        epoch: "current",
        sequence: 1,
        nodeIndices: new Uint32Array(),
        positions: new Float32Array(),
        alpha: 0.2,
        computeMs: 2,
        activeCount: 0,
        emittedAtEpochMs: 1_700,
      },
    } as MessageEvent);
    expect(onFrame).not.toHaveBeenCalled();
  });

  it("drops a frame whose command sequence predates the current drag", () => {
    const worker = new WorkerDouble();
    const onFrame = vi.fn();
    const controller = new LocalForceController({ onFrame, workerFactory: () => worker });
    const nodes = [{ id: "n0", label: "N0", attributes: {} }];
    controller.initialize("current", nodes, [], new Map([["n0", { x: 0, y: 0 }]]));
    controller.dragStart("n0", 10, 20, nodes.length);
    worker.onmessage?.({
      data: {
        type: "frame",
        epoch: "current",
        sequence: 1,
        nodeIndices: new Uint32Array([0]),
        positions: new Float32Array([0, 0]),
        alpha: 0.7,
        computeMs: 1,
        activeCount: 1,
      },
    } as MessageEvent);
    expect(onFrame).not.toHaveBeenCalled();
  });
});
