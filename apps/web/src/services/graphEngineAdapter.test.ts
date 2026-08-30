import { describe, expect, it, vi } from "vitest";
import type { Graph, GraphData, LayoutOptions } from "@antv/g6";
import { GraphEngineAdapter } from "./graphEngineAdapter";
import * as graphEngineModule from "./graphEngineAdapter";

function graphDouble(
  initialCamera: {
    readonly position: readonly [number, number];
    readonly zoom: number;
    readonly size?: readonly [number, number];
  } = { position: [18, 27], zoom: 1.4 },
) {
  const calls: string[] = [];
  let position: [number, number] = [...initialCamera.position];
  let zoom = initialCamera.zoom;
  let size: [number, number] = [...(initialCamera.size ?? [1_000, 700])];
  const graph = {
    destroyed: false,
    getSize: () => [...size] as [number, number],
    getPosition: () => [...position],
    getZoom: () => zoom,
    getCanvasByViewport: ([x, y]: readonly [number, number]) =>
      [(x - position[0]) / zoom, (y - position[1]) / zoom] as [number, number],
    getViewportByCanvas: ([x, y]: readonly [number, number]) =>
      [x * zoom + position[0], y * zoom + position[1]] as [number, number],
    getElementPosition: vi.fn(() => [41, 52]),
    getNodeData: vi.fn(() => [{ id: "a" }, { id: "b" }, { id: "node" }]),
    getEdgeData: vi.fn(() => [{ id: "edge" }]),
    setData: vi.fn(() => calls.push("setData")),
    updateNodeData: vi.fn(() => calls.push("updateNodeData")),
    translateElementTo: vi.fn(async () => calls.push("translateElementTo")),
    setElementState: vi.fn(async () => calls.push("setElementState")),
    setElementVisibility: vi.fn(async () => calls.push("setElementVisibility")),
    draw: vi.fn(async () => {
      calls.push("draw");
    }),
    layout: vi.fn(async () => {
      calls.push("layout");
    }),
    stopLayout: vi.fn(() => calls.push("stopLayout")),
    setLayout: vi.fn(() => calls.push("setLayout")),
    resize: vi.fn((width: number, height: number) => {
      calls.push("resize");
      size = [width, height];
    }),
    translateBy: vi.fn(async (delta: readonly [number, number]) => {
      calls.push("translateBy");
      position = [position[0] + delta[0], position[1] + delta[1]];
    }),
    translateTo: vi.fn(async (nextPosition: readonly [number, number]) => {
      calls.push("translateTo");
      position = [...nextPosition];
    }),
    zoomTo: vi.fn(
      async (
        nextZoom: number,
        _animation: boolean,
        origin?: readonly [number, number],
      ) => {
        calls.push("zoomTo");
        const ratio = nextZoom / zoom;
        if (origin) {
          position = [
            origin[0] + (position[0] - origin[0]) * ratio,
            origin[1] + (position[1] - origin[1]) * ratio,
          ];
        }
        zoom = nextZoom;
      },
    ),
  };
  return {
    graph: graph as unknown as Graph,
    calls,
    camera: () => ({ position: [...position], zoom }),
  };
}

describe("GraphEngineAdapter", () => {
  it("captures a complete versioned camera snapshot for the committed scene", () => {
    const { graph } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, () => [1_000, 700], "scene-a");

    expect(adapter.captureCamera()).toEqual({
      schemaVersion: "socialgraph-fm.graph-camera/2",
      sceneIdentity: "scene-a",
      position: [18, 27],
      zoom: 1.4,
      worldCenter: expect.any(Array),
      viewportSize: [1_000, 700],
    });
  });

  it("drops a restore captured for a stale scene before touching the camera", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, () => [1_000, 700], "scene-current");
    const restore = adapter.restoreCamera as unknown as (
      snapshot: {
        readonly schemaVersion: "socialgraph-fm.graph-camera/2";
        readonly sceneIdentity: string;
        readonly position: readonly [number, number];
        readonly zoom: number;
        readonly worldCenter: readonly [number, number];
        readonly viewportSize: readonly [number, number];
      },
      expectedSceneIdentity: string,
    ) => Promise<boolean>;

    await expect(restore.call(adapter, {
      schemaVersion: "socialgraph-fm.graph-camera/2",
      sceneIdentity: "scene-old",
      position: [90, 120],
      zoom: 2,
      worldCenter: [10, 20],
      viewportSize: [1_000, 700],
    }, "scene-current")).resolves.toBe(false);
    expect(calls).toEqual([]);
  });

  it("invalidates a queued camera command as soon as a newer scene is requested", async () => {
    const { graph, calls } = graphDouble();
    let releaseBlocker: (() => void) | undefined;
    graph.setElementState = vi.fn(() => new Promise<void>((resolve) => { releaseBlocker = resolve; }));
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-old");
    const snapshot = adapter.captureCamera();

    const blocker = adapter.applyElementStates({ edge: ["interaction-lod"] });
    const staleRestore = adapter.restoreCamera(snapshot, "scene-old");
    const replacement = adapter.replaceScene({ nodes: [{ id: "new" }], edges: [] } as GraphData, undefined, "scene-new");
    releaseBlocker?.();

    await blocker;
    await expect(staleRestore).resolves.toBe(false);
    await replacement;
    expect(calls.filter((call) => call === "setData")).toHaveLength(1);
  });

  it("rolls back in-flight camera work invalidated while its action is awaiting", async () => {
    const { graph, camera } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-old");
    let releaseFocus: (() => void) | undefined;
    let focusStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => { focusStarted = resolve; });
    const gate = new Promise<void>((resolve) => { releaseFocus = resolve; });

    const focus = adapter.runCameraForScene("scene-old", 1, async () => {
      await graph.translateTo([140, 160], false);
      focusStarted?.();
      await gate;
    });
    await started;
    const replacement = adapter.replaceScene(
      { nodes: [{ id: "new" }], edges: [] } as GraphData,
      undefined,
      "scene-new",
    );
    releaseFocus?.();

    await expect(focus).resolves.toBe(false);
    await replacement;
    expect(camera().position[0]).toBeCloseTo(18, 8);
    expect(camera().position[1]).toBeCloseTo(27, 8);
    expect(camera().zoom).toBeCloseTo(1.4, 8);
  });

  it("cancels in-flight camera work for a direct gesture without rolling over the user's pan", async () => {
    const { graph, camera } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-current");
    let releaseFocus: (() => void) | undefined;
    let focusStarted: (() => void) | undefined;
    const started = new Promise<void>((resolve) => { focusStarted = resolve; });
    const gate = new Promise<void>((resolve) => { releaseFocus = resolve; });

    const focus = adapter.runCameraForScene("scene-current", 1, async () => {
      await graph.translateTo([140, 160], false);
      focusStarted?.();
      await gate;
    });
    await started;
    expect(adapter.cancelCameraForGesture("scene-current")).toBe(true);
    await graph.translateBy([25, -10], false);
    releaseFocus?.();

    await expect(focus).resolves.toBe(false);
    expect(camera()).toEqual({ position: [165, 150], zoom: 1.4 });
  });

  it("rejects an older camera token delivered after a newer token for the same scene", async () => {
    const { graph, camera } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-current");

    await expect(adapter.runCameraForScene("scene-current", 2, async () => {
      await graph.translateTo([80, 90], false);
    })).resolves.toBe(true);
    await expect(adapter.runCameraForScene("scene-current", 1, async () => {
      await graph.translateTo([200, 220], false);
    })).resolves.toBe(false);

    expect(camera()).toEqual({ position: [80, 90], zoom: 1.4 });
  });

  it("rejects a late lower external restore token before it mutates the camera", async () => {
    const { graph, camera } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-current");
    const restore = adapter.restoreCamera as unknown as (
      snapshot: { readonly position: readonly [number, number]; readonly zoom: number },
      expectedSceneIdentity: string,
      requestToken: number,
    ) => Promise<boolean>;

    await expect(restore.call(adapter, { position: [80, 90], zoom: 1.4 }, "scene-current", 2)).resolves.toBe(true);
    await expect(restore.call(adapter, { position: [200, 220], zoom: 1.4 }, "scene-current", 1)).resolves.toBe(false);

    expect(camera()).toEqual({ position: [80, 90], zoom: 1.4 });
  });

  it("rejects a late lower external visibility token before focus preparation", async () => {
    const { graph } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-current");

    await expect(adapter.ensureVisible(["a"], "scene-current", 2)).resolves.toBe(true);
    await expect(adapter.ensureVisible(["b"], "scene-current", 1)).resolves.toBe(false);

    expect(graph.setElementVisibility).toHaveBeenCalledTimes(1);
    expect(graph.setElementVisibility).toHaveBeenCalledWith({ a: "visible" }, false);
  });

  it("keeps external governance and internal search visibility tokens independent", async () => {
    const { graph } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-current");
    const ensureVisible = adapter.ensureVisible as unknown as (
      nodeIds: readonly string[],
      expectedSceneIdentity: string,
      requestToken: number,
      tokenScope: string,
    ) => Promise<boolean>;

    await expect(ensureVisible.call(adapter, ["a"], "scene-current", 20, "external-focus")).resolves.toBe(true);
    await expect(ensureVisible.call(adapter, ["b"], "scene-current", 2, "internal-search")).resolves.toBe(true);
    await expect(ensureVisible.call(adapter, ["b"], "scene-current", 19, "external-focus")).resolves.toBe(false);

    expect(graph.setElementVisibility).toHaveBeenCalledTimes(2);
  });

  it("isolates restore token sources and replays one activation without accepting a lower token", async () => {
    const { graph } = graphDouble();
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-a");
    const restore = adapter.restoreCamera as unknown as (
      snapshot: { readonly position: readonly [number, number]; readonly zoom: number },
      expectedSceneIdentity: string,
      requestToken: number,
      tokenScope: string,
      activation: number,
    ) => Promise<boolean>;

    await expect(restore.call(adapter, { position: [80, 90], zoom: 1.4 }, "scene-a", 20, "workspace", 0)).resolves.toBe(true);
    await expect(restore.call(adapter, { position: [100, 110], zoom: 1.4 }, "scene-a", 2, "governance", 0)).resolves.toBe(true);
    await expect(restore.call(adapter, { position: [120, 130], zoom: 1.4 }, "scene-a", 2, "governance", 1)).resolves.toBe(true);
    await adapter.replaceScene({ nodes: [{ id: "b" }], edges: [] } as GraphData, undefined, "scene-b");
    await adapter.replaceScene({ nodes: [{ id: "a" }], edges: [] } as GraphData, undefined, "scene-a");
    await expect(restore.call(adapter, { position: [200, 210], zoom: 1.4 }, "scene-a", 19, "workspace", 2)).resolves.toBe(false);
    await expect(restore.call(adapter, { position: [80, 90], zoom: 1.4 }, "scene-a", 20, "workspace", 0)).resolves.toBe(true);
  });

  it("caches camera snapshots independently by workspace, graph identity, and lens", () => {
    const CameraCache = (graphEngineModule as typeof graphEngineModule & {
      GraphCameraSnapshotCache?: new () => {
        set(key: { workspace: string; graphIdentity: string; lens: string }, snapshot: unknown): void;
        get(key: { workspace: string; graphIdentity: string; lens: string }): unknown;
      };
    }).GraphCameraSnapshotCache;
    expect(CameraCache).toBeTypeOf("function");
    if (!CameraCache) return;
    const cache = new CameraCache();
    const governanceRisk = {
      schemaVersion: "socialgraph-fm.graph-camera/2",
      sceneIdentity: "governance-scene-risk",
      position: [18, 27], zoom: 1.4, worldCenter: [344, 230], viewportSize: [1_000, 700],
    };
    const governanceRelations = { ...governanceRisk, sceneIdentity: "governance-scene-relations", position: [99, 88] };
    cache.set({ workspace: "governance", graphIdentity: "graph-a", lens: "risk" }, governanceRisk);
    cache.set({ workspace: "governance", graphIdentity: "graph-a", lens: "relations" }, governanceRelations);

    expect(cache.get({ workspace: "governance", graphIdentity: "graph-a", lens: "risk" })).toEqual(governanceRisk);
    expect(cache.get({ workspace: "governance", graphIdentity: "graph-a", lens: "relations" })).toEqual(governanceRelations);
    expect(cache.get({ workspace: "chat", graphIdentity: "graph-a", lens: "risk" })).toBeUndefined();
  });

  it("updates scene data without fitting the viewport and restores the camera", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);

    await adapter.replaceScene({ nodes: [], edges: [] } as GraphData, {
      layout: "preserve",
      camera: "preserve",
    });

    expect(calls).toEqual([
      "setData",
      "draw",
      "zoomTo",
      "translateBy",
    ]);
  });

  it("runs a full layout only when explicitly requested", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);

    await adapter.replaceScene({ nodes: [], edges: [] } as GraphData, {
      layout: "full",
      camera: "preserve",
    });

    expect(calls).toEqual([
      "setData",
      "draw",
      "layout",
      "zoomTo",
      "translateBy",
    ]);
  });

  it("preserves positions shared by the previous and next scene", async () => {
    const { graph } = graphDouble();
    (graph.getNodeData as unknown as ReturnType<typeof vi.fn>).mockReturnValue([{ id: "n1" }]);
    const adapter = new GraphEngineAdapter(graph);

    await adapter.replaceScene({
      nodes: [{ id: "n1", style: { fill: "#fff" } }],
      edges: [],
    } as GraphData);

    expect(graph.setData).toHaveBeenCalledWith({
      nodes: [{ id: "n1", style: { fill: "#fff", x: 41, y: 52 } }],
      edges: [],
    });
  });

  it("updates appearance without moving the camera", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);

    await adapter.setAppearance(async () => {
      calls.push("appearance");
    });

    expect(calls).toEqual(["appearance", "zoomTo", "translateBy"]);
  });

  it("serializes scene replacement and appearance repaint in one renderer lane", async () => {
    const { graph, calls } = graphDouble();
    let finishDraw: (() => void) | undefined;
    graph.draw = vi.fn(() => new Promise<void>((resolve) => { finishDraw = resolve; }));
    const adapter = new GraphEngineAdapter(graph);

    const scene = adapter.replaceScene({ nodes: [], edges: [] } as GraphData);
    const appearance = adapter.setAppearance(async () => { calls.push("appearance"); });

    expect(calls).toEqual(["setData"]);
    expect(adapter.mutationDiagnostics()).toMatchObject({
      mutationInFlight: 1,
      queuedMutations: 1,
    });
    finishDraw?.();
    await Promise.all([scene, appearance]);
    expect(calls).toEqual(["setData", "zoomTo", "translateBy", "appearance", "zoomTo", "translateBy"]);
    expect(adapter.mutationDiagnostics().mutationInFlightMax).toBe(1);
  });

  it("serializes visibility changes behind an in-flight scene replacement", async () => {
    const { graph, calls } = graphDouble();
    let finishDraw: (() => void) | undefined;
    graph.draw = vi.fn(() => new Promise<void>((resolve) => { finishDraw = resolve; }));
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-old");

    const scene = adapter.replaceScene({ nodes: [], edges: [] } as GraphData, undefined, "scene-new");
    const visibility = adapter.applyVisibility({ edge: "hidden" }, "scene-new");

    expect(calls).toEqual(["setData"]);
    expect(graph.setElementVisibility).not.toHaveBeenCalled();
    expect(adapter.mutationDiagnostics()).toMatchObject({
      mutationInFlight: 1,
      queuedMutations: 1,
    });
    finishDraw?.();
    await Promise.all([scene, visibility]);
    expect(calls).toEqual(["setData", "zoomTo", "translateBy", "setElementVisibility"]);
  });

  it("drops queued visibility for a scene invalidated by a replacement request", async () => {
    const { graph } = graphDouble();
    let releaseBlocker: (() => void) | undefined;
    graph.setElementState = vi.fn(() => new Promise<void>((resolve) => { releaseBlocker = resolve; }));
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-old");

    const blocker = adapter.applyElementStates({ edge: ["interaction-lod"] });
    const staleVisibility = adapter.applyVisibility({ edge: "hidden" }, "scene-old");
    const replacement = adapter.replaceScene({ nodes: [{ id: "new" }], edges: [] } as GraphData, undefined, "scene-new");
    releaseBlocker?.();

    await blocker;
    await expect(staleVisibility).resolves.toBe(false);
    await replacement;
    expect(graph.setElementVisibility).not.toHaveBeenCalled();
  });

  it("rejects an obsolete queued appearance generation before it repaints", async () => {
    const { graph, calls } = graphDouble();
    let finishBlocker: (() => void) | undefined;
    graph.setElementState = vi.fn(() => new Promise<void>((resolve) => { finishBlocker = resolve; }));
    const adapter = new GraphEngineAdapter(graph);

    const blocker = adapter.applyElementStates({ edge: ["interaction-lod"] });
    const obsolete = adapter.setAppearance(async () => { calls.push("obsolete"); });
    const current = adapter.setAppearance(async () => { calls.push("current"); });
    finishBlocker?.();

    await expect(obsolete).resolves.toBe(false);
    await expect(current).resolves.toBe(true);
    await blocker;
    expect(calls).toEqual(["current", "zoomTo", "translateBy"]);
  });

  it("applies worker positions in one draw without moving the camera", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);

    await adapter.applyPositions({ a: { x: 10, y: 20 }, b: { x: 30, y: 40 } });

    expect(graph.translateElementTo).toHaveBeenCalledWith(
      { a: [10, 20], b: [30, 40] },
      false,
    );
    expect(calls).toEqual(["translateElementTo"]);
  });

  it("serializes position mutations and keeps only the latest pending frame", async () => {
    const { graph } = graphDouble();
    const resolvers: Array<() => void> = [];
    graph.translateElementTo = vi.fn(() => new Promise<void>((resolve) => resolvers.push(resolve)));
    const adapter = new GraphEngineAdapter(graph);

    const first = adapter.applyPositions({ a: { x: 1, y: 1 } });
    const superseded = adapter.applyPositions({ a: { x: 2, y: 2 } });
    const latest = adapter.applyPositions({ a: { x: 3, y: 3 } });
    expect(graph.translateElementTo).toHaveBeenCalledTimes(1);

    resolvers.shift()?.();
    await first;
    await Promise.resolve();
    expect(graph.translateElementTo).toHaveBeenCalledTimes(2);
    expect(graph.translateElementTo).toHaveBeenLastCalledWith({ a: [3, 3] }, false);
    resolvers.shift()?.();
    await Promise.all([superseded, latest]);
    expect(adapter.mutationDiagnostics()).toMatchObject({
      mutationInFlight: 0,
      mutationInFlightMax: 1,
      queuedMutations: 0,
      positionFramePending: false,
    });
  });

  it("merges target and neighbour coordinates while keeping the newest value per node", async () => {
    const { graph } = graphDouble();
    const resolvers: Array<() => void> = [];
    graph.translateElementTo = vi.fn(() => new Promise<void>((resolve) => resolvers.push(resolve)));
    const adapter = new GraphEngineAdapter(graph);

    const blocker = adapter.applyElementStates({ edge: ["interaction-lod"] });
    const target = adapter.applyPositions({ target: { x: 10, y: 20 } });
    const neighbours = adapter.applyPositions({
      neighbour: { x: 30, y: 40 },
      target: { x: 11, y: 21 },
    });

    resolvers.shift()?.();
    await blocker;
    await Promise.resolve();
    expect(graph.translateElementTo).toHaveBeenCalledWith(
      { target: [11, 21], neighbour: [30, 40] },
      false,
    );
    resolvers.shift()?.();
    await Promise.all([target, neighbours]);
  });

  it("serializes interaction LOD with pointer and Worker position mutations", async () => {
    const { graph } = graphDouble();
    const resolvers: Array<() => void> = [];
    graph.setElementState = vi.fn(() => new Promise<void>((resolve) => resolvers.push(resolve)));
    graph.translateElementTo = vi.fn(() => new Promise<void>((resolve) => resolvers.push(resolve)));
    const adapter = new GraphEngineAdapter(graph);

    const lod = adapter.applyElementStates({ edge: ["interaction-lod"] });
    const pointer = adapter.applyPositions({ a: { x: 1, y: 1 } });
    const worker = adapter.applyPositions({ a: { x: 3, y: 3 } });

    expect(graph.setElementState).toHaveBeenCalledTimes(1);
    expect(graph.translateElementTo).not.toHaveBeenCalled();
    expect(adapter.mutationDiagnostics()).toMatchObject({
      mutationInFlight: 1,
      mutationInFlightMax: 1,
      queuedMutations: 1,
      positionFramePending: true,
    });

    resolvers.shift()?.();
    await lod;
    await Promise.resolve();
    expect(graph.translateElementTo).toHaveBeenCalledTimes(1);
    expect(graph.translateElementTo).toHaveBeenLastCalledWith({ a: [3, 3] }, false);
    expect(adapter.mutationDiagnostics().mutationInFlightMax).toBe(1);

    resolvers.shift()?.();
    await Promise.all([pointer, worker]);
    expect(adapter.mutationDiagnostics()).toMatchObject({
      mutationInFlight: 0,
      mutationInFlightMax: 1,
      queuedMutations: 0,
      positionFramePending: false,
    });
  });

  it("applies only a visibility diff and never starts layout", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);
    await adapter.applyVisibility({ a: "hidden", b: "visible" });
    expect(calls).toEqual(["setElementVisibility"]);
  });

  it("drops stale visibility ids that are absent from the live renderer scene", async () => {
    const { graph } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);

    await adapter.applyVisibility({ edge: "hidden", "removed-edge": "hidden" });

    expect(graph.setElementVisibility).toHaveBeenCalledWith({ edge: "hidden" }, false);
  });

  it("reveals known nodes without fitting or translating the camera", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);
    await expect(adapter.ensureVisible(["a"])).resolves.toBe(true);
    expect(graph.setElementVisibility).toHaveBeenCalledWith({ a: "visible" }, false);
    expect(calls).toEqual(["setElementVisibility"]);
  });

  it("queues visibility behind the matching scene commit and lets only the newest focus token win", async () => {
    const { graph, calls } = graphDouble();
    let liveIds = new Set(["old"]);
    let drawing = false;
    let finishDraw: (() => void) | undefined;
    const mutableGraph = graph as unknown as Record<string, ReturnType<typeof vi.fn>>;
    mutableGraph.getNodeData = vi.fn(() => [...liveIds].map((id) => ({ id })));
    mutableGraph.getElementPosition = vi.fn((id: string) => {
      if (!liveIds.has(id)) throw new Error(`UNKNOWN_ELEMENT:${id}`);
      return [41, 52];
    });
    mutableGraph.setData = vi.fn((data: GraphData) => {
      calls.push("setData");
      liveIds = new Set((data.nodes ?? []).map((node) => String(node.id)));
    });
    mutableGraph.draw = vi.fn(() => {
      calls.push("draw");
      drawing = true;
      return new Promise<void>((resolve) => { finishDraw = () => { drawing = false; resolve(); }; });
    });
    mutableGraph.setElementVisibility = vi.fn(async (changes: Record<string, string>) => {
      if (drawing) throw new Error("UNKNOWN_ELEMENT_PAGE");
      if (Object.keys(changes).some((id) => !liveIds.has(id))) throw new Error("UNKNOWN_ELEMENT");
      calls.push(`visible:${Object.keys(changes).join(",")}`);
    });
    const adapter = new GraphEngineAdapter(graph, undefined, "scene-old");

    const replacement = adapter.replaceScene({ nodes: [{ id: "new-a" }, { id: "new-b" }], edges: [] } as GraphData, {
      layout: "preserve", camera: "preserve",
    }, "scene-new");
    const obsolete = adapter.ensureVisible(["new-a"], "scene-new", 1);
    const newest = adapter.ensureVisible(["new-b"], "scene-new", 2);

    expect(graph.setElementVisibility).not.toHaveBeenCalled();
    finishDraw?.();
    await replacement;
    await expect(obsolete).resolves.toBe(false);
    await expect(newest).resolves.toBe(true);
    expect(graph.setElementVisibility).toHaveBeenCalledTimes(1);
    expect(graph.setElementVisibility).toHaveBeenCalledWith({ "new-b": "visible" }, false);
    expect(calls).not.toContain("UNKNOWN_ELEMENT_PAGE");
  });

  it("commits force settings through the public layout API", async () => {
    const { graph, calls } = graphDouble();
    const adapter = new GraphEngineAdapter(graph);

    await adapter.setForces({ type: "d3-force" } as LayoutOptions);

    expect(calls).toEqual([
      "stopLayout",
      "setLayout",
      "layout",
      "zoomTo",
      "translateBy",
    ]);
  });

  it("resizes while preserving the world point at the CSS viewport centre", async () => {
    const { graph, calls, camera } = graphDouble();
    let viewport: readonly [number, number] = [1_000, 700];
    const adapter = new GraphEngineAdapter(graph, () => viewport);
    viewport = [1_400, 900];

    await adapter.resize();

    expect(graph.resize).toHaveBeenCalledWith(1_400, 900);
    expect(calls).toEqual(["resize", "translateBy"]);
    expect(camera().position[0]).toBeCloseTo(218, 8);
    expect(camera().position[1]).toBeCloseTo(127, 8);
    expect(camera().zoom).toBe(1.4);
  });

  it("uses the live G6 canvas size when the construction measurement is stale", async () => {
    const { graph } = graphDouble();
    let viewport: readonly [number, number] = [990, 700];
    const adapter = new GraphEngineAdapter(graph, () => viewport);
    viewport = [1_010, 700];

    await adapter.resize();

    expect(graph.resize).toHaveBeenCalledWith(1_010, 700);
    const resizeDelta = vi.mocked(graph.translateBy).mock.calls[0]?.[0];
    expect(resizeDelta?.[0]).toBeCloseTo(5, 8);
    expect(resizeDelta?.[1]).toBeCloseTo(0, 8);
  });

  it("re-centres an explicit stable world point after the backing canvas already resized", async () => {
    const { graph } = graphDouble({ position: [18, 27], zoom: 1.4, size: [1_010, 700] });
    const adapter = new GraphEngineAdapter(graph, () => [1_010, 700]);
    const stableWorldCenter = graph.getCanvasByViewport([500, 350]);

    await adapter.resizePreservingWorldCenter(1_010, 700, stableWorldCenter);

    expect(graph.resize).not.toHaveBeenCalled();
    const resizeDelta = vi.mocked(graph.translateBy).mock.calls[0]?.[0];
    expect(resizeDelta?.[0]).toBeCloseTo(5, 8);
    expect(resizeDelta?.[1]).toBeCloseTo(0, 8);
  });

  it("serializes resizes and applies only the latest pending viewport", async () => {
    const { graph } = graphDouble();
    const translations: Array<() => void> = [];
    graph.translateBy = vi.fn(() => new Promise<void>((resolve) => translations.push(resolve)));
    const adapter = new GraphEngineAdapter(graph, () => [1_000, 700]);

    const first = adapter.resizePreservingWorldCenter(1_100, 720);
    const superseded = adapter.resizePreservingWorldCenter(1_200, 760);
    const latest = adapter.resizePreservingWorldCenter(1_400, 900);
    expect(graph.resize).toHaveBeenCalledTimes(1);
    expect(graph.resize).toHaveBeenLastCalledWith(1_100, 720);

    translations.shift()?.();
    await vi.waitFor(() => expect(graph.resize).toHaveBeenCalledTimes(2));
    expect(graph.resize).toHaveBeenLastCalledWith(1_400, 900);
    translations.shift()?.();
    await Promise.all([first, superseded, latest]);
  });

  it("keeps a restore behind an in-flight resize in the shared renderer lane", async () => {
    const { graph, calls } = graphDouble();
    let releaseResize: (() => void) | undefined;
    graph.translateBy = vi.fn(() => new Promise<void>((resolve) => {
      calls.push("resize-translate");
      releaseResize = resolve;
    }));
    const adapter = new GraphEngineAdapter(graph, () => [1_000, 700], "scene-a");
    const snapshot = adapter.captureCamera();

    const resize = adapter.resizePreservingWorldCenter(1_200, 800);
    await vi.waitFor(() => expect(releaseResize).toBeTypeOf("function"));
    const restore = adapter.restoreCamera(snapshot, "scene-a");

    expect(graph.zoomTo).not.toHaveBeenCalled();
    expect(adapter.mutationDiagnostics().mutationInFlightMax).toBe(1);
    releaseResize?.();
    await resize;
    await vi.waitFor(() => expect(graph.resize).toHaveBeenCalledWith(1_000, 700));
    releaseResize?.();
    await vi.waitFor(() => expect(graph.zoomTo).toHaveBeenCalledTimes(1));
    // Let the restore's world-centre translation settle as well.
    releaseResize?.();
    await restore;
  });

  it("keeps a resize behind an in-flight restore in the shared renderer lane", async () => {
    const { graph } = graphDouble();
    let releaseRestore: (() => void) | undefined;
    graph.zoomTo = vi.fn(() => new Promise<void>((resolve) => { releaseRestore = resolve; }));
    const adapter = new GraphEngineAdapter(graph, () => [1_000, 700], "scene-a");
    const snapshot = adapter.captureCamera();

    const restore = adapter.restoreCamera(snapshot, "scene-a");
    await vi.waitFor(() => expect(releaseRestore).toBeTypeOf("function"));
    const resize = adapter.resizePreservingWorldCenter(1_200, 800);

    expect(graph.resize).not.toHaveBeenCalled();
    expect(adapter.mutationDiagnostics().mutationInFlightMax).toBe(1);
    releaseRestore?.();
    await restore;
    await resize;
  });

  it("restores the exact camera after zoom changes the viewport translation", async () => {
    const { graph, calls, camera } = graphDouble({
      position: [320, -120],
      zoom: 0.5,
    });
    const adapter = new GraphEngineAdapter(graph);

    await adapter.restoreCamera({ position: [18, 27], zoom: 1.4 });

    expect(calls).toEqual(["zoomTo", "translateTo"]);
    expect(graph.zoomTo).toHaveBeenCalledWith(1.4, false, [0, 0]);
    expect(graph.translateTo).toHaveBeenCalledWith([18, 27], false);
    expect(camera()).toEqual({ position: [18, 27], zoom: 1.4 });
  });

  it("keeps the exact saved position when the versioned snapshot viewport is unchanged", async () => {
    const { graph, calls, camera } = graphDouble({
      position: [320, -120],
      zoom: 0.5,
      size: [392, 665],
    });
    const adapter = new GraphEngineAdapter(graph, () => [392, 665], "scene-a");

    await adapter.restoreCamera({
      schemaVersion: "socialgraph-fm.graph-camera/2",
      sceneIdentity: "scene-a",
      position: [370.667, 447.042],
      zoom: 0.346,
      worldCenter: [
        (392 / 2 - 370.667) / 0.346,
        (665 / 2 - 447.042) / 0.346,
      ],
      viewportSize: [392, 665],
    }, "scene-a");

    expect(calls).toEqual(["zoomTo", "translateBy"]);
    expect(camera().position[0]).toBeCloseTo(370.667, 8);
    expect(camera().position[1]).toBeCloseTo(447.042, 8);
    expect(camera().zoom).toBeCloseTo(0.346, 8);
  });

  it("keeps the captured world centre centred when a remounted viewport is smaller", async () => {
    const { graph, calls, camera } = graphDouble();
    let viewport: readonly [number, number] = [1_000, 700];
    const adapter = new GraphEngineAdapter(graph, () => viewport);
    const snapshot = adapter.captureCamera();

    viewport = [400, 300];
    await adapter.restoreCamera(snapshot);

    expect(calls).toEqual(["resize", "translateBy", "zoomTo", "translateBy"]);
    expect(camera().position[0]).toBeCloseTo(-282, 8);
    expect(camera().position[1]).toBeCloseTo(-173, 8);
    expect(graph.getViewportByCanvas(snapshot.worldCenter!)).toEqual([200, 150]);
  });

  it("synchronizes a stale backing viewport before restoring a revealed pane", async () => {
    const { graph, calls } = graphDouble();
    let viewport: readonly [number, number] = [432, 333];
    const adapter = new GraphEngineAdapter(graph, () => viewport, "scene-a");
    const snapshot = adapter.captureCamera();

    viewport = [392, 665];
    await adapter.restoreCamera(snapshot, "scene-a");

    expect(graph.resize).toHaveBeenCalledWith(392, 665);
    expect(calls).toEqual(["resize", "translateBy", "zoomTo", "translateBy"]);
    const restoredCenter = graph.getViewportByCanvas(snapshot.worldCenter);
    expect(restoredCenter[0]).toBeCloseTo(196, 8);
    expect(restoredCenter[1]).toBeCloseTo(332.5, 8);
  });
});
