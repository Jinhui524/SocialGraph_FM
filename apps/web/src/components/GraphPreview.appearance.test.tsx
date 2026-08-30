import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ComponentType } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalysisOverlay, GraphVersion, GovernanceFocus } from "../types/graph";
import type { ViewportAnimationEffectTiming } from "@antv/g6";
import { governanceExactRelationKey } from "../services/graphPreviewPolicy";
import { GraphEngineAdapter } from "../services/graphEngineAdapter";

const graphHarness = vi.hoisted(() => ({
  instances: [] as Array<Record<string, unknown>>,
  focusCalls: [] as Array<{ ids: string[]; liveIds: string[] }>,
  stateCalls: [] as Array<Record<string, readonly string[]>>,
  fitCalls: 0,
  forceOffscreen: false,
  staleErrors: [] as string[],
  drawGate: null as null | { promise: Promise<void>; resolve: () => void },
  renderGate: null as null | { promise: Promise<void>; resolve: () => void; entered: () => void },
  postReplaceDrawGate: null as null | { promise: Promise<void>; resolve: () => void; entered: () => void },
  zoomGate: null as null | { promise: Promise<void>; entered: () => void },
}));

vi.mock("@antv/g6", () => {
  class Graph {
    destroyed = false;
    rendered = false;
    private nodes: Array<Record<string, any>>;
    private edges: Array<Record<string, any>>;
    private handlers = new Map<string, Set<(...args: any[]) => void>>();
    private position: [number, number] = [0, 0];
    private zoom = 1;
    private drawsSinceSetData = -1;

    constructor(options: { data?: { nodes?: Array<Record<string, any>>; edges?: Array<Record<string, any>> } }) {
      this.nodes = (options.data?.nodes ?? []).map((node) => ({ ...node, style: { ...node.style } }));
      this.edges = (options.data?.edges ?? []).map((edge) => ({ ...edge, style: { ...edge.style } }));
      graphHarness.instances.push(this as unknown as Record<string, unknown>);
    }

    async render() {
      if (graphHarness.renderGate) {
        graphHarness.renderGate.entered();
        await graphHarness.renderGate.promise;
      }
      this.rendered = true;
    }
    async draw() {
      this.rendered = true;
      if (graphHarness.drawGate) await graphHarness.drawGate.promise;
      if (this.drawsSinceSetData >= 0) {
        this.drawsSinceSetData += 1;
        if (this.drawsSinceSetData === 2 && graphHarness.postReplaceDrawGate) {
          graphHarness.postReplaceDrawGate.entered();
          await graphHarness.postReplaceDrawGate.promise;
        }
      }
    }
    async layout() {}
    destroy() { this.destroyed = true; this.rendered = false; }
    stopLayout() {}
    setLayout() {}
    resize() {}
    on(event: string, handler: (...args: any[]) => void) {
      const handlers = this.handlers.get(event) ?? new Set();
      handlers.add(handler); this.handlers.set(event, handlers);
    }
    off(event: string, handler: (...args: any[]) => void) { this.handlers.get(event)?.delete(handler); }
    emit(event: string) { for (const handler of this.handlers.get(event) ?? []) handler(); }
    getNodeData() { return this.nodes; }
    getEdgeData() { return this.edges; }
    setData(data: { nodes?: Array<Record<string, any>>; edges?: Array<Record<string, any>> }) {
      this.drawsSinceSetData = 0;
      this.nodes = (data.nodes ?? []).map((node) => ({ ...node, style: { ...node.style } }));
      this.edges = (data.edges ?? []).map((edge) => ({ ...edge, style: { ...edge.style } }));
    }
    updateNodeData(updates: Array<Record<string, any>>) { this.nodes = mergeById(this.nodes, updates); }
    updateEdgeData(updates: Array<Record<string, any>>) { this.edges = mergeById(this.edges, updates); }
    addNodeData(nodes: Array<Record<string, any>>) { this.nodes.push(...nodes); }
    addEdgeData(edges: Array<Record<string, any>>) { this.edges.push(...edges); }
    removeNodeData(ids: string[]) { this.nodes = this.nodes.filter((node) => !ids.includes(String(node.id))); }
    removeEdgeData(ids: string[]) { this.edges = this.edges.filter((edge) => !ids.includes(String(edge.id))); }
    getElementPosition(id: string) {
      const node = this.nodes.find((item) => item.id === id);
      if (!node) {
        graphHarness.staleErrors.push(`UNKNOWN_ELEMENT:${id}`);
        throw new Error(`UNKNOWN_ELEMENT:${id}`);
      }
      return [Number(node?.style?.x ?? 20), Number(node?.style?.y ?? 20)];
    }
    getElementRenderStyle(id: string) {
      return this.nodes.find((item) => item.id === id)?.style
        ?? this.edges.find((item) => item.id === id)?.style ?? {};
    }
    async translateElementTo(target: string | Record<string, [number, number]>, point?: [number, number]) {
      const positions = typeof target === "string" ? { [target]: point! } : target;
      for (const [id, next] of Object.entries(positions)) {
        this.nodes = mergeById(this.nodes, [{ id, style: { x: next[0], y: next[1] } }]);
      }
    }
    async setElementState(states: Record<string, readonly string[]>) {
      graphHarness.stateCalls.push(Object.fromEntries(
        Object.entries(states).map(([id, values]) => [id, [...values]]),
      ));
      const live = new Set([...this.nodes, ...this.edges].map((item) => String(item.id)));
      for (const id of Object.keys(states)) {
        if (!live.has(id)) graphHarness.staleErrors.push(`UNKNOWN_STATE:${id}`);
      }
    }
    async setElementVisibility(changes: Record<string, string>) {
      const live = new Set([...this.nodes, ...this.edges].map((item) => String(item.id)));
      if (graphHarness.drawGate || Object.keys(changes).some((id) => !live.has(id))) {
        graphHarness.staleErrors.push("UNKNOWN_ELEMENT_PAGE");
        throw new Error("UNKNOWN_ELEMENT_PAGE");
      }
    }
    getPosition() { return this.position; }
    getZoom() { return this.zoom; }
    getSize() { return [800, 600]; }
    getViewportCenter() { return [400, 300]; }
    getCanvasByViewport(point: [number, number]) {
      return [
        (point[0] - this.position[0]) / this.zoom,
        (point[1] - this.position[1]) / this.zoom,
      ];
    }
    getViewportByCanvas(point: [number, number]) {
      return [
        point[0] * this.zoom + this.position[0],
        point[1] * this.zoom + this.position[1],
      ];
    }
    getCanvasByClient(point: [number, number]) { return point; }
    getClientByCanvas(point: [number, number]) { return point; }
    async zoomTo(zoom: number) {
      const gate = graphHarness.zoomGate;
      if (gate) {
        gate.entered();
        await gate.promise;
      }
      this.zoom = zoom;
    }
    async translateTo(position: [number, number]) { this.position = position; }
    async translateBy(delta: [number, number]) { this.position = [this.position[0] + delta[0], this.position[1] + delta[1]]; }
    toDataURL() { return "data:image/png;base64,ZmFrZQ=="; }
  }

  function mergeById(current: Array<Record<string, any>>, updates: Array<Record<string, any>>) {
    const updateById = new Map(updates.map((update) => [String(update.id), update]));
    return current.map((item) => {
      const update = updateById.get(String(item.id));
      return update ? { ...item, ...update, style: { ...item.style, ...update.style }, data: { ...item.data, ...update.data } } : item;
    });
  }

  return {
    Graph,
    CanvasEvent: { CLICK: "canvas:click" },
    GraphEvent: { BEFORE_TRANSFORM: "before-transform", AFTER_TRANSFORM: "after-transform" },
    NodeEvent: { CLICK: "node:click", DBLCLICK: "node:dblclick" },
  };
});

vi.mock("../services/graphRenderer", () => ({
  detectWebGLSupport: () => false,
  resolveGraphRendererKind: () => "canvas",
  loadGraphRenderer: vi.fn(async () => ({
    renderer: () => ({}),
    status: { requested: "auto", resolved: "canvas", webglSupported: false, contextLossCount: 0 },
  })),
}));

vi.mock("../services/graphPerformanceProbe", () => ({
  GraphPerformanceProbe: class {
    begin() {}
    end() { return 0; }
    record() {}
    attach() { return () => undefined; }
    dispose() {}
    snapshot() { return []; }
  },
}));

vi.mock("../services/localForceController", () => ({
  LocalForceController: class {
    initialize() {}
    destroy() {}
    nodeId() { return undefined; }
  },
}));

vi.mock("../services/graphCameraController", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../services/graphCameraController")>();
  return {
    ...actual,
    GraphCameraController: class extends actual.GraphCameraController {
      override async fit(ids: readonly string[], animation: false | ViewportAnimationEffectTiming = false) {
        graphHarness.fitCalls += 1;
        return super.fit(ids, animation);
      }
      override async focus(ids: readonly string[], options: import("../services/graphCameraController").GraphCameraFocusOptions) {
        const liveIds = ((this as unknown as { graph: { getNodeData: () => Array<{ id: unknown }> } }).graph.getNodeData())
          .map((node) => String(node.id));
        graphHarness.focusCalls.push({ ids: [...ids], liveIds });
        return super.focus(ids, options);
      }
      override hasVisibleElement(ids: readonly string[]) {
        return graphHarness.forceOffscreen ? false : super.hasVisibleElement(ids);
      }
    },
  };
});

import GraphPreview, { graphSemanticBadges } from "./GraphPreview";

const graph: GraphVersion = {
  id: "appearance-graph",
  sourceFile: "appearance.csv",
  createdAt: "2026-08-20T00:00:00.000Z",
  nodes: [
    { id: "n1", label: "Candidate 1", type: "account", attributes: { rank: 1 } },
    { id: "n2", label: "Candidate 2", type: "account", attributes: { rank: 2 } },
    { id: "n3", label: "Context", type: "account", attributes: {} },
  ],
  edges: [
    { id: "exact", source: "n1", target: "n2", type: "factual_relation", directed: false, attributes: { modalities: ["coRT"] } },
    { id: "incident", source: "n1", target: "n3", type: "factual_relation", directed: false, attributes: { modalities: ["coURL"] } },
  ],
  summary: { nodeCount: 3, edgeCount: 2, density: 2 / 3, averageDegree: 4 / 3, connectedComponents: 1, isolatedNodes: 0 },
  issues: [],
  preview: {
    nodes: [
      { id: "n1", label: "Candidate 1", type: "account", attributes: { rank: 1 } },
      { id: "n2", label: "Candidate 2", type: "account", attributes: { rank: 2 } },
      { id: "n3", label: "Context", type: "account", attributes: {} },
    ],
    edges: [
      { id: "exact", source: "n1", target: "n2", type: "factual_relation", directed: false, attributes: { modalities: ["coRT"] } },
      { id: "incident", source: "n1", target: "n3", type: "factual_relation", directed: false, attributes: { modalities: ["coURL"] } },
    ],
    truncated: false,
    originalNodeCount: 3,
    originalEdgeCount: 2,
  },
  truncated: false,
};

const overlay: AnalysisOverlay = {
  id: "relations-overlay",
  graphVersionId: graph.id,
  kind: "governance",
  nodeValues: {},
  edgeValues: { exact: "factual", incident: "factual" },
  presentation: { governanceLens: "relations" },
  legend: { title: "Relations", items: [{ value: "factual", label: "Factual", color: "#5F7896" }] },
  provenance: { engine: "test", algorithm: "relations" },
};

type AppearanceSnapshot = {
  readonly focusCameraToken?: number;
  readonly nodeStyles: Readonly<Record<string, Record<string, unknown>>>;
  readonly edgeStyles: Readonly<Record<string, Record<string, unknown>>>;
};

describe("GraphPreview governance appearance integration", () => {
  it("keeps imported labels and human decisions in separate badge positions", () => {
    expect(graphSemanticBadges("positive", "pending")).toMatchObject([
      { text: "+", placement: "right-top", backgroundFill: "#D85C56" },
      { text: "?", placement: "right-bottom", backgroundFill: "#C58B2A" },
    ]);
    expect(graphSemanticBadges("negative", "rejected")).toMatchObject([
      { text: "−", placement: "right-top", backgroundFill: "#218B7C" },
      { text: "×", placement: "right-bottom", backgroundFill: "#218B7C" },
    ]);
  });
  beforeEach(() => {
    graphHarness.instances.length = 0;
    graphHarness.focusCalls.length = 0;
    graphHarness.stateCalls.length = 0;
    graphHarness.fitCalls = 0;
    graphHarness.forceOffscreen = false;
    graphHarness.staleErrors.length = 0;
    graphHarness.drawGate = null;
    graphHarness.renderGate = null;
    graphHarness.postReplaceDrawGate = null;
    graphHarness.zoomGate = null;
    vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(800);
    vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockReturnValue(600);
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => { callback(0); return 1; });
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
    vi.stubGlobal("ResizeObserver", class {
      observe() {}
      disconnect() {}
    });
  });

  afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it("applies dark styling to the viewport and G6 host without recreating the engine", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} theme="brand-light" />);
    await waitFor(() => expect(graphHarness.instances).toHaveLength(1));

    rendered.rerender(<GraphPreview graphVersion={graph} theme="focus-dark" />);

    const viewport = rendered.container.querySelector<HTMLElement>(".graph-preview__viewport");
    const canvasHost = rendered.container.querySelector<HTMLElement>(".graph-preview__canvas");
    expect(viewport).toHaveStyle({ backgroundColor: "#081327" });
    expect(canvasHost).toHaveStyle({ backgroundColor: "#081327" });
    expect(graphHarness.instances).toHaveLength(1);
  });

  it("exposes interaction proxy colours through the live graph theme", async () => {
    const nodes = Array.from({ length: 301 }, (_, index) => ({
      id: `proxy-${index}`,
      label: `Proxy ${index}`,
      type: "account",
      attributes: {},
    }));
    const largeGraph: GraphVersion = {
      ...graph,
      id: "interaction-proxy-graph",
      nodes,
      edges: [],
      summary: { nodeCount: 301, edgeCount: 0, density: 0, averageDegree: 0, connectedComponents: 301, isolatedNodes: 301 },
      preview: { nodes, edges: [], truncated: false, originalNodeCount: 301, originalEdgeCount: 0 },
    };
    const rendered = render(<GraphPreview graphVersion={largeGraph} theme="brand-light" />);
    await waitFor(() => expect(graphHarness.instances).toHaveLength(1));
    const root = rendered.container.querySelector<HTMLElement>(".graph-preview--workbench")!;
    const veil = rendered.container.querySelector<HTMLElement>(".graph-preview__interaction-veil")!;
    const marker = rendered.container.querySelector<HTMLElement>(".graph-preview__interaction-proxy-marker")!;
    const label = rendered.container.querySelector<HTMLElement>(".graph-preview__interaction-proxy-label")!;

    expect(veil.style.background).toBe("");
    expect(marker.style.borderColor).toBe("");
    expect(label.style.color).toBe("");

    rendered.rerender(<GraphPreview graphVersion={largeGraph} theme="focus-dark" />);
    expect(root).toHaveClass("graph-preview--focus-dark");
    expect(veil.style.background).toBe("");
    expect(label.style.color).toBe("");
    expect(graphHarness.instances).toHaveLength(1);
  });

  it("exposes a visible return-to-overview action for external chat focus", async () => {
    const onReturn = vi.fn();
    const Preview = GraphPreview as unknown as ComponentType<Record<string, unknown>>;
    render(<Preview
      graphVersion={graph}
      viewMode="local"
      focusNodeIds={["n1"]}
      returnToOverviewAction={{ label: "返回完整图", onReturn }}
    />);

    fireEvent.click(await screen.findByRole("button", { name: "返回完整图" }));
    expect(onReturn).toHaveBeenCalledOnce();
  });

  it("keeps graph clicks selection-only, honors later explicit focus, and distinguishes canvas pans", async () => {
    const onSelectNode = vi.fn();
    const onReturn = vi.fn();
    const rendered = render(<GraphPreview
      graphVersion={graph}
      onSelectNode={onSelectNode}
      returnToOverviewAction={{ label: "返回完整图", onReturn }}
    />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    const canvas = rendered.container.querySelector<HTMLElement>(".graph-preview__canvas")!;
    const graphInstance = graphHarness.instances[0] as unknown as {
      getElementPosition: (id: string) => readonly [number, number];
    };
    const [nodeX, nodeY] = graphInstance.getElementPosition("n1");

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 7, clientX: nodeX, clientY: nodeY });
    fireEvent.pointerUp(canvas, { button: 0, pointerId: 7, clientX: nodeX, clientY: nodeY });
    await waitFor(() => expect(onSelectNode).toHaveBeenLastCalledWith(expect.objectContaining({ id: "n1" })));

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      selectedNodeId="n1"
      onSelectNode={onSelectNode}
      cameraFocusCommand={{ nodeIds: ["n1"], anchorNodeId: "n1", token: 1 }}
      returnToOverviewAction={{ label: "返回完整图", onReturn }}
    />);
    await act(async () => { await Promise.resolve(); });
    await waitFor(() => expect(graphHarness.focusCalls).toHaveLength(1));

    onSelectNode.mockClear();
    fireEvent.pointerDown(canvas, { button: 0, pointerId: 8, clientX: 700, clientY: 500 });
    fireEvent.pointerMove(window, { pointerId: 8, clientX: 760, clientY: 500 });
    fireEvent.pointerUp(window, { pointerId: 8, clientX: 760, clientY: 500 });
    expect(onSelectNode).not.toHaveBeenCalled();
    expect(onReturn).not.toHaveBeenCalled();

    fireEvent.pointerDown(canvas, { button: 0, pointerId: 9, clientX: 700, clientY: 500 });
    fireEvent.pointerUp(window, { pointerId: 9, clientX: 700, clientY: 500 });
    expect(onSelectNode).toHaveBeenCalledTimes(1);
    expect(onSelectNode).toHaveBeenLastCalledWith(null);
    expect(onReturn).not.toHaveBeenCalled();

    expect(graphHarness.focusCalls).toHaveLength(1);
    expect(graphHarness.instances).toHaveLength(1);
  });

  it("restores an exact cached camera on engine entry without racing an initial fit", async () => {
    let captured: import("./GraphPreview").GraphPreviewViewSnapshot | undefined;
    const first = render(<GraphPreview graphVersion={graph} onViewStateChange={(snapshot) => { captured = snapshot; }} />);
    await waitFor(() => expect(captured?.camera.sceneIdentity).toContain(graph.id));
    first.unmount();
    graphHarness.fitCalls = 0;
    const camera = {
      ...captured!.camera,
      position: [91, 73] as [number, number],
      zoom: 1.15,
      worldCenter: [(400 - 91) / 1.15, (300 - 73) / 1.15] as [number, number],
      viewportSize: [800, 600] as [number, number],
    };

    render(<GraphPreview
      graphVersion={graph}
      cameraRestoreCommand={{ ...camera, x: 91, y: 73, token: 1 }}
    />);
    await waitFor(() => expect(graphHarness.instances).toHaveLength(2));
    await waitFor(() => {
      const position = (graphHarness.instances[1] as any).getPosition();
      expect(position[0]).toBeCloseTo(91, 8);
      expect(position[1]).toBeCloseTo(73, 8);
    });

    expect(graphHarness.fitCalls).toBe(0);
  });

  it("does not attach a previous graph world centre to a legacy restore command", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    const nextGraph: GraphVersion = { ...graph, id: "appearance-graph-next" };

    rendered.rerender(<GraphPreview
      graphVersion={nextGraph}
      cameraRestoreCommand={{ x: 91, y: 73, zoom: 1.15, token: 1 }}
    />);

    await waitFor(() => expect(graphHarness.instances).toHaveLength(2));
    const nextInstance = graphHarness.instances[1] as Record<string, any>;
    await waitFor(() => expect(nextInstance.getPosition()).toEqual([91, 73]));
    expect(nextInstance.getZoom()).toBeCloseTo(1.15, 8);
  });

  it("defers an uncached entry fit until measurement is visible and fits exactly once", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} isPaneVisible={false} />);
    await waitFor(() => expect(graphHarness.instances).toHaveLength(1));
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    expect(graphHarness.fitCalls).toBe(0);

    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible />);
    await waitFor(() => expect(graphHarness.fitCalls).toBe(1));
    expect(graphHarness.fitCalls).toBe(1);
  });

  it("honours a legacy camera command received while hidden before the first reveal", async () => {
    const rendered = render(<GraphPreview
      graphVersion={graph}
      isPaneVisible={false}
      cameraRestoreCommand={{ x: 91, y: 73, zoom: 1.15, token: 1 }}
    />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    expect(graphHarness.fitCalls).toBe(0);

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      isPaneVisible
      cameraRestoreCommand={{ x: 91, y: 73, zoom: 1.15, token: 1 }}
    />);

    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 80)); });
    const instance = graphHarness.instances[0] as Record<string, any>;
    expect(instance.getPosition()).toEqual([91, 73]);
    expect(instance.getZoom()).toBeCloseTo(1.15, 8);
    expect(graphHarness.fitCalls).toBe(0);
  });

  it("does not refit an initially visible topology after a workspace hide and reveal", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} isPaneVisible />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    await waitFor(() => expect(graphHarness.fitCalls).toBe(1));

    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible={false} />);
    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible />);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });

    expect(graphHarness.fitCalls).toBe(1);
  });

  it("preserves the saved world centre when a visited pane reveals at a new size", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} isPaneVisible />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    await waitFor(() => expect(graphHarness.fitCalls).toBe(1));
    const instance = graphHarness.instances[0] as Record<string, any>;
    const root = rendered.container.querySelector<HTMLElement>(".graph-preview")!;
    const viewport = rendered.container.querySelector<HTMLElement>(".graph-preview__viewport")!;

    await act(async () => {
      await instance.zoomTo(1.15);
      await instance.translateTo([91, 73]);
      instance.emit("after-transform");
    });
    await waitFor(() => expect(Number(root.dataset.cameraX)).toBeCloseTo(91, 8));
    const savedWorldCenter = instance.getCanvasByViewport([400, 300]) as [number, number];

    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible={false} />);
    Object.defineProperties(viewport, {
      clientWidth: { configurable: true, value: 400 },
      clientHeight: { configurable: true, value: 300 },
    });
    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible />);

    await waitFor(() => {
      const centre = instance.getViewportByCanvas(savedWorldCenter) as [number, number];
      expect(centre[0]).toBeCloseTo(200, 8);
      expect(centre[1]).toBeCloseTo(150, 8);
    }, { timeout: 2_000 });
    expect(graphHarness.fitCalls).toBe(1);
  });

  it("does not publish a camera snapshot from a hidden pane transform", async () => {
    const onViewStateChange = vi.fn();
    const rendered = render(<GraphPreview
      graphVersion={graph}
      isPaneVisible
      onViewStateChange={onViewStateChange}
    />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    await waitFor(() => expect(onViewStateChange).toHaveBeenCalled());
    const callsBeforeHide = onViewStateChange.mock.calls.length;

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      isPaneVisible={false}
      onViewStateChange={onViewStateChange}
    />);
    (graphHarness.instances[0] as { emit: (event: string) => void }).emit("after-transform");
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 120)); });

    expect(onViewStateChange).toHaveBeenCalledTimes(callsBeforeHide);
  });

  it("defers the first fit when the pane hides while its initial render is pending", async () => {
    let releaseRender!: () => void;
    const renderPromise = new Promise<void>((resolve) => { releaseRender = resolve; });
    const entered = vi.fn();
    graphHarness.renderGate = { promise: renderPromise, resolve: releaseRender, entered };
    const onViewStateChange = vi.fn();
    const rendered = render(<GraphPreview graphVersion={graph} isPaneVisible onViewStateChange={onViewStateChange} />);
    await waitFor(() => expect(entered).toHaveBeenCalledOnce());
    const viewport = rendered.container.querySelector<HTMLElement>(".graph-preview__viewport")!;
    Object.defineProperties(viewport, {
      clientWidth: { configurable: true, value: 0 },
      clientHeight: { configurable: true, value: 0 },
    });

    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible={false} onViewStateChange={onViewStateChange} />);
    graphHarness.renderGate = null;
    releaseRender();
    await act(async () => { await Promise.resolve(); });

    expect(graphHarness.fitCalls).toBe(0);
    expect(onViewStateChange).not.toHaveBeenCalled();

    Object.defineProperties(viewport, {
      clientWidth: { configurable: true, value: 800 },
      clientHeight: { configurable: true, value: 600 },
    });
    rendered.rerender(<GraphPreview graphVersion={graph} isPaneVisible onViewStateChange={onViewStateChange} />);
    await waitFor(() => expect(graphHarness.fitCalls).toBe(1));
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    expect(graphHarness.fitCalls).toBe(1);
  });

  it("ignores late lower external restore and focus tokens for the same scene", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    const instance = graphHarness.instances[0] as Record<string, any>;
    graphHarness.focusCalls.length = 0;

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraRestoreCommand={{ x: 80, y: 90, zoom: 1.2, token: 2 }}
    />);
    await waitFor(() => expect(instance.getPosition()).toEqual([80, 90]));

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraRestoreCommand={{ x: 180, y: 190, zoom: 1.4, token: 1 }}
    />);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });
    expect(instance.getPosition()).toEqual([80, 90]);

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraFocusCommand={{ nodeIds: ["n1"], token: 2 }}
    />);
    await waitFor(() => expect(graphHarness.focusCalls.at(-1)?.ids).toEqual(["n1"]));
    const focusCount = graphHarness.focusCalls.length;

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraFocusCommand={{ nodeIds: ["n2"], token: 1 }}
    />);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });

    expect(graphHarness.focusCalls).toHaveLength(focusCount);
    expect(graphHarness.focusCalls.at(-1)?.ids).toEqual(["n1"]);
  });

  it("isolates governance and adaptation focus token producers", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    graphHarness.focusCalls.length = 0;

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraFocusCommand={{ nodeIds: ["n1"], token: 20, commandScope: "governance" }}
    />);
    await waitFor(() => expect(graphHarness.focusCalls.at(-1)?.ids).toEqual(["n1"]));

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraFocusCommand={{ nodeIds: ["n2"], token: 2, commandScope: "adaptation" }}
    />);
    await waitFor(() => expect(graphHarness.focusCalls.at(-1)?.ids).toEqual(["n2"]));
    const focusCount = graphHarness.focusCalls.length;

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      cameraFocusCommand={{ nodeIds: ["n3"], token: 19, commandScope: "governance" }}
    />);
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 50)); });
    expect(graphHarness.focusCalls).toHaveLength(focusCount);
  });

  it("routes toolbar zoom through the current scene command lane", async () => {
    const lane = vi.spyOn(GraphEngineAdapter.prototype, "runCameraForScene");
    const rendered = render(<GraphPreview graphVersion={graph} />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    lane.mockClear();

    fireEvent.click(screen.getByRole("button", { name: "放大图谱" }));

    await waitFor(() => expect(lane).toHaveBeenCalled());
    expect(lane.mock.calls[0]?.[0]).toContain(graph.id);
  });

  it("composes rapid toolbar zoom intentions while the first camera command is pending", async () => {
    const rendered = render(<GraphPreview graphVersion={graph} />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    const instance = graphHarness.instances[0] as Record<string, any>;
    const initialZoom = instance.getZoom() as number;
    let releaseZoom!: () => void;
    const entered = vi.fn();
    graphHarness.zoomGate = {
      promise: new Promise<void>((resolve) => { releaseZoom = resolve; }),
      entered,
    };

    fireEvent.click(screen.getByRole("button", { name: "放大图谱" }));
    await waitFor(() => expect(entered).toHaveBeenCalledOnce());
    instance.emit("after-transform");
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 100)); });
    fireEvent.click(screen.getByRole("button", { name: "放大图谱" }));
    graphHarness.zoomGate = null;
    releaseZoom();

    await waitFor(() => expect(instance.getZoom()).toBeCloseTo(initialZoom * 1.22 * 1.22, 8));
  });

  it("routes search focus through the current scene command lane", async () => {
    const lane = vi.spyOn(GraphEngineAdapter.prototype, "runCameraForScene");
    const rendered = render(<GraphPreview
      graphVersion={graph}
      cameraFocusCommand={{ nodeIds: ["n2"], token: 20 }}
    />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    await waitFor(() => expect(graphHarness.focusCalls.at(-1)?.ids).toContain("n2"));
    rendered.rerender(<GraphPreview graphVersion={graph} />);
    graphHarness.focusCalls.length = 0;
    lane.mockClear();

    fireEvent.change(screen.getByRole("textbox", { name: "按名称或 ID 搜索节点" }), { target: { value: "Candidate 1" } });
    fireEvent.click(await screen.findByRole("option", { name: /Candidate 1/u }));

    await waitFor(() => expect(graphHarness.focusCalls.at(-1)?.ids).toContain("n1"));
    expect(lane).toHaveBeenCalled();
    expect(lane.mock.calls.at(-1)?.[0]).toContain(graph.id);
  });

  it("routes fullscreen recovery fit through the current scene command lane", async () => {
    const lane = vi.spyOn(GraphEngineAdapter.prototype, "runCameraForScene");
    const rendered = render(<GraphPreview graphVersion={graph} />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    lane.mockClear();
    graphHarness.forceOffscreen = true;

    fireEvent(document, new Event("fullscreenchange"));

    await waitFor(() => expect(lane).toHaveBeenCalled());
    expect(lane.mock.calls[0]?.[0]).toContain(graph.id);
  });

  it("renders imported labels as an independent governance appearance channel", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const labelOverlay: AnalysisOverlay = {
      id: "imported-labels",
      graphVersionId: graph.id,
      kind: "governance",
      nodeValues: {
        n1: "risk-low",
        n2: "risk-review",
      },
      edgeValues: {},
      presentation: {
        governanceLens: "risk",
        riskBands: { n1: "low", n2: "review" },
        referenceLabels: { n1: "positive", n2: "negative" },
      },
      legend: { title: "已知少样本标签", items: [] },
      provenance: { engine: "imported-label-set", algorithm: "few-shot-label-view-v1" },
    };
    render(<GraphPreview
      graphVersion={graph}
      activeOverlay={labelOverlay}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);

    await waitFor(() => expect(appearances.length).toBeGreaterThan(0));
    expect(appearances.at(-1)?.nodeStyles.n1).toMatchObject({
      fill: "#D85C56",
      stroke: "#8E3733",
      halo: true,
      haloStroke: "#D85C56",
      badges: [expect.objectContaining({ text: "+", placement: "right-top" })],
    });
    expect(appearances.at(-1)?.nodeStyles.n2).toMatchObject({
      fill: "#E5A53B",
      stroke: "#218B7C",
      halo: true,
      haloStroke: "#E5A53B",
      badges: [expect.objectContaining({ text: "−", placement: "right-top" })],
    });
    expect(appearances.at(-1)?.nodeStyles.n3.fill).not.toBe("#D85C56");
  });

  it("accepts bounded reference semantics from governance node values", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const semanticOverlay: AnalysisOverlay = {
      id: "semantic-imported-labels",
      graphVersionId: graph.id,
      kind: "governance",
      nodeValues: { n1: "reference-positive", n2: "reference-negative" },
      edgeValues: {},
      presentation: { governanceLens: "risk" },
      legend: { title: "已知少样本标签", items: [] },
      provenance: { engine: "imported-label-set", algorithm: "few-shot-label-view-v1" },
    };
    render(<GraphPreview
      graphVersion={graph}
      activeOverlay={semanticOverlay}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);

    await waitFor(() => expect(appearances.length).toBeGreaterThan(0));
    expect(appearances.at(-1)?.nodeStyles.n1).toMatchObject({
      fill: "#D85C56",
      stroke: "#8E3733",
    });
    expect(appearances.at(-1)?.nodeStyles.n2).toMatchObject({
      fill: "#5F7896",
      stroke: "#218B7C",
    });
  });

  it("keeps community and relationship fills while adding independent risk rings", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const communityOverlay: AnalysisOverlay = {
      id: "community-risk-bands",
      graphVersionId: graph.id,
      kind: "community",
      nodeValues: { n1: "community-a", n2: "community-a", n3: "community-b" },
      edgeValues: {},
      presentation: {
        riskBands: { n1: "high", n2: "review", n3: "low" },
        referenceLabels: { n1: "positive", n2: "negative" },
      },
      legend: {
        title: "风险群组",
        items: [
          { value: "community-a", label: "群组 A", color: "#345BE7" },
          { value: "community-b", label: "群组 B", color: "#218B7C" },
        ],
      },
      provenance: { engine: "test", algorithm: "community-risk" },
    };
    const rendered = render(<GraphPreview
      graphVersion={graph}
      activeOverlay={communityOverlay}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);

    await waitFor(() => expect(appearances.length).toBeGreaterThan(0));
    const community = appearances.at(-1)!;
    expect(community.nodeStyles.n1).toMatchObject({ halo: true, haloStroke: "#E75E58" });
    expect(community.nodeStyles.n2).toMatchObject({ halo: true, haloStroke: "#E5A53B" });
    expect(community.nodeStyles.n1.fill).toBe("#345BE7");
    expect(community.nodeStyles.n2.fill).toBe("#345BE7");
    expect(community.nodeStyles.n3.fill).toBe("#218B7C");
    expect(community.nodeStyles.n1.badges).toEqual([
      expect.objectContaining({ text: "+", placement: "right-top" }),
    ]);
    expect(community.nodeStyles.n2.badges).toEqual([
      expect.objectContaining({ text: "−", placement: "right-top" }),
    ]);

    const relationships: AnalysisOverlay = {
      ...overlay,
      id: "relationship-risk-bands",
      presentation: { governanceLens: "relations", riskBands: { n1: "high", n2: "review" } },
    };
    rendered.rerender(<GraphPreview
      graphVersion={graph}
      activeOverlay={relationships}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);
    await waitFor(() => expect(appearances.at(-1)?.nodeStyles.n1.haloStroke).toBe("#E75E58"));
    expect(appearances.at(-1)?.nodeStyles.n2).toMatchObject({ halo: true, haloStroke: "#E5A53B" });
  });

  it("highlights one-hop nodes and incident edges in the risk lens without moving the camera", async () => {
    const riskOverlay: AnalysisOverlay = {
      id: "risk-selection",
      graphVersionId: graph.id,
      kind: "governance",
      nodeValues: { n1: "risk-high", n2: "risk-review", n3: "risk-low" },
      edgeValues: { exact: "evidence-high", incident: "evidence-high" },
      presentation: { governanceLens: "risk", riskBands: { n1: "high", n2: "review", n3: "low" } },
      legend: { title: "风险节点", items: [] },
      provenance: { engine: "test", algorithm: "risk-selection" },
    };
    const rendered = render(<GraphPreview
      graphVersion={graph}
      activeOverlay={riskOverlay}
      selectedNodeId="n1"
    />);

    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    await waitFor(() => expect(graphHarness.stateCalls.some((states) => (
      states.n1?.includes("governance-selected")
      && states.n2?.includes("neighbour")
      && states.n3?.includes("neighbour")
      && states.exact?.includes("governance-focus")
      && states.incident?.includes("governance-focus")
    ))).toBe(true));
    expect(graphHarness.focusCalls).toHaveLength(0);
  });

  it("updates engine node and exact-edge styles on same-identity node-to-relation focus", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const nodeFocus: GovernanceFocus = { kind: "node", targetId: "n1", nodeIds: ["n1"], cameraToken: 1 };
    const rendered = render(<GraphPreview
      graphVersion={graph}
      activeOverlay={overlay}
      governanceFocus={nodeFocus}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);
    await waitFor(() => expect(graphHarness.instances).toHaveLength(1));
    const instance = graphHarness.instances[0] as Record<string, any>;
    await waitFor(() => expect(appearances.at(-1)?.focusCameraToken).toBe(1));
    expect(appearances.at(-1)?.nodeStyles.n1).toMatchObject({ halo: true, label: true, opacity: 1 });
    expect(Number(appearances.at(-1)?.nodeStyles.n1.size)).toBeGreaterThan(Number(appearances.at(-1)?.nodeStyles.n3.size));
    expect(appearances.at(-1)?.nodeStyles.n3.opacity).toBe(0.28);

    const relationFocus: GovernanceFocus = {
      kind: "relation",
      targetId: "relation-exact",
      nodeIds: ["n1", "n2"],
      exactRelationKey: governanceExactRelationKey("n1", "n2", ["coRT"]),
      cameraToken: 2,
    };
    rendered.rerender(<GraphPreview
      graphVersion={graph}
      activeOverlay={overlay}
      governanceFocus={relationFocus}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);
    await waitFor(() => expect(appearances.at(-1)?.focusCameraToken).toBe(2));
    const relationAppearance = appearances.at(-1)!;
    expect(relationAppearance.edgeStyles.exact).toMatchObject({ stroke: "#22B8C7", strokeOpacity: 0.94 });
    expect(Number(relationAppearance.edgeStyles.exact.lineWidth)).toBeGreaterThan(Number(relationAppearance.edgeStyles.incident.lineWidth));
    expect(relationAppearance.edgeStyles.incident.strokeOpacity).toBe(0.12);
    expect(relationAppearance.nodeStyles.n1).toMatchObject({ halo: true, label: true, opacity: 1 });
    expect(relationAppearance.nodeStyles.n2).toMatchObject({ halo: true, label: true, opacity: 1 });
    expect(relationAppearance.nodeStyles.n3.opacity).toBe(0.28);
    expect(graphHarness.instances).toHaveLength(1);
    const root = rendered.container.querySelector<HTMLElement>(".graph-preview")!;
    expect(root.dataset.engineCreateCount).toBe("1");
    expect(root.dataset.layoutCount).toBe("1");
  });

  it("emphasizes internal group edges without changing community fills or camera", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const communityOverlay: AnalysisOverlay = {
      id: "community-group-focus",
      graphVersionId: graph.id,
      kind: "community",
      nodeValues: { n1: "community-a", n2: "community-a", n3: "community-b" },
      edgeValues: {},
      presentation: { riskBands: { n1: "high", n2: "review", n3: "low" } },
      legend: { title: "风险群组", items: [] },
      provenance: { engine: "test", algorithm: "community-risk" },
    };
    const rendered = render(<GraphPreview
      graphVersion={graph}
      activeOverlay={communityOverlay}
      governanceFocus={{ kind: "group", targetId: "group-1", nodeIds: ["n1", "n2"], cameraToken: 3 }}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);

    await waitFor(() => expect(appearances.at(-1)?.focusCameraToken).toBe(3));
    const appearance = appearances.at(-1)!;
    expect(appearance.nodeStyles.n1.fill).toBe(appearance.nodeStyles.n2.fill);
    expect(appearance.nodeStyles.n3.opacity).toBe(0.28);
    expect(appearance.edgeStyles.exact.strokeOpacity).toBe(0.75);
    expect(appearance.edgeStyles.incident.strokeOpacity).toBe(0.06);
    expect(Number(appearance.edgeStyles.exact.lineWidth)).toBeGreaterThan(Number(appearance.edgeStyles.incident.lineWidth));
    expect(graphHarness.focusCalls).toHaveLength(0);
    expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.layoutCount).toBe("1");
  });

  it("renders the server adapted rank instead of combining a base rank with a delta marker", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const adaptedOverlay = {
      ...overlay,
      id: "adapted-overlay",
      presentation: {
        governanceLens: "risk",
        rankDeltas: { n1: 2 },
        adaptedRanks: { n1: 3 },
      },
    } as unknown as AnalysisOverlay;
    render(<GraphPreview
      graphVersion={graph}
      activeOverlay={adaptedOverlay}
      governanceFocus={{ kind: "node", targetId: "n1", nodeIds: ["n1"], cameraToken: 9 }}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);

    await waitFor(() => expect(String(appearances.at(-1)?.nodeStyles.n1?.labelText)).toMatch(/^#3 /u));
    expect(String(appearances.at(-1)?.nodeStyles.n1?.labelText)).not.toMatch(/↓2|#1/u);
  });

  it("can suppress rank prefixes for adaptation previews without hiding focused labels", async () => {
    const appearances: AppearanceSnapshot[] = [];
    const adaptedOverlay = {
      ...overlay,
      id: "adapted-overlay-with-hidden-ranks",
      presentation: {
        governanceLens: "risk",
        rankDeltas: { n1: 2 },
        adaptedRanks: { n1: 3 },
      },
    } as unknown as AnalysisOverlay;
    render(<GraphPreview
      graphVersion={graph}
      activeOverlay={adaptedOverlay}
      governanceFocus={{ kind: "node", targetId: "n1", nodeIds: ["n1"], cameraToken: 10 }}
      showNodeRanks={false}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);

    await waitFor(() => expect(String(appearances.at(-1)?.nodeStyles.n1?.labelText)).toMatch(/^Candidate/u));
    expect(String(appearances.at(-1)?.nodeStyles.n1?.labelText)).not.toMatch(/[#↑↓]/u);
  });

  it("does not let a cancelled scene publish post-replacement appearance or readiness", async () => {
    let releasePostReplaceDraw!: () => void;
    const postReplaceDraw = new Promise<void>((resolve) => { releasePostReplaceDraw = resolve; });
    const entered = vi.fn();
    const appearances: AppearanceSnapshot[] = [];
    const focusA: GovernanceFocus = { kind: "node", targetId: "n1", nodeIds: ["n1"], cameraToken: 10 };
    const rendered = render(<GraphPreview
      graphVersion={graph}
      governanceFocus={focusA}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));

    const graphB: GraphVersion = {
      ...graph,
      nodes: [
        { id: "b1", label: "B 1", type: "account", attributes: {} },
        { id: "b2", label: "B 2", type: "account", attributes: {} },
      ],
      edges: [{ id: "b-edge", source: "b1", target: "b2", type: "factual_relation", directed: false, attributes: {} }],
      summary: { nodeCount: 2, edgeCount: 1, density: 1, averageDegree: 1, connectedComponents: 1, isolatedNodes: 0 },
      preview: {
        nodes: [
          { id: "b1", label: "B 1", type: "account", attributes: {} },
          { id: "b2", label: "B 2", type: "account", attributes: {} },
        ],
        edges: [{ id: "b-edge", source: "b1", target: "b2", type: "factual_relation", directed: false, attributes: {} }],
        truncated: false,
        originalNodeCount: 2,
        originalEdgeCount: 1,
      },
    };
    const focusB: GovernanceFocus = { kind: "node", targetId: "b1", nodeIds: ["b1"], cameraToken: 20 };
    graphHarness.postReplaceDrawGate = { promise: postReplaceDraw, resolve: releasePostReplaceDraw, entered };
    rendered.rerender(<GraphPreview
      graphVersion={graphB}
      governanceFocus={focusB}
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);
    await waitFor(() => expect(entered).toHaveBeenCalledOnce());

    rendered.rerender(<GraphPreview
      graphVersion={graph}
      governanceFocus={focusA}
      selectedNodeId="n1"
      onAppearanceApplied={(snapshot: AppearanceSnapshot) => appearances.push(snapshot)}
    />);
    const appearancesAfterReturnRequest = appearances.length;
    graphHarness.postReplaceDrawGate = null;
    releasePostReplaceDraw();

    const instance = graphHarness.instances[0] as Record<string, any>;
    await waitFor(() => expect(instance.getNodeData().map((node: { id: string }) => node.id)).toEqual(["n1", "n2", "n3"]));
    await waitFor(() => expect((rendered.container.querySelector(".graph-preview") as HTMLElement).dataset.graphReady).toBe("true"));
    await waitFor(() => expect(appearances.at(-1)?.focusCameraToken).toBe(10));
    expect(appearances.slice(appearancesAfterReturnRequest).some((snapshot) => snapshot.focusCameraToken === 20)).toBe(false);
    expect(graphHarness.instances).toHaveLength(1);
    expect(graphHarness.staleErrors).toEqual([]);
  });

  it("waits for scene replacement and camera restoration, then applies only the newest focus", async () => {
    let resolveDraw!: () => void;
    const drawPromise = new Promise<void>((resolve) => { resolveDraw = resolve; });
    const metrics: Array<{ ready: boolean }> = [];
    const rendered = render(<GraphPreview graphVersion={graph} onRuntimeMetrics={(value) => metrics.push(value)} />);
    await waitFor(() => expect(metrics.some((value) => value.ready)).toBe(true));
    graphHarness.focusCalls.length = 0;
    graphHarness.staleErrors.length = 0;

    const replacement: GraphVersion = {
      ...graph,
      nodes: [
        { id: "new-a", label: "New A", type: "account", attributes: {} },
        { id: "new-b", label: "New B", type: "account", attributes: {} },
      ],
      edges: [{ id: "new-edge", source: "new-a", target: "new-b", type: "factual_relation", directed: false, attributes: {} }],
      summary: { nodeCount: 2, edgeCount: 1, density: 1, averageDegree: 1, connectedComponents: 1, isolatedNodes: 0 },
      preview: {
        nodes: [
          { id: "new-a", label: "New A", type: "account", attributes: {} },
          { id: "new-b", label: "New B", type: "account", attributes: {} },
        ],
        edges: [{ id: "new-edge", source: "new-a", target: "new-b", type: "factual_relation", directed: false, attributes: {} }],
        truncated: false, originalNodeCount: 2, originalEdgeCount: 1,
      },
    };
    graphHarness.drawGate = { promise: drawPromise, resolve: resolveDraw };
    rendered.rerender(<GraphPreview graphVersion={replacement} cameraFocusCommand={{ nodeIds: ["new-a"], anchorNodeId: "new-a", token: 10 }} />);
    rendered.rerender(<GraphPreview graphVersion={replacement} cameraFocusCommand={{ nodeIds: ["new-b"], anchorNodeId: "new-b", token: 11 }} />);
    await Promise.resolve();
    graphHarness.drawGate = null;
    resolveDraw();

    await waitFor(() => expect(graphHarness.focusCalls).toEqual([{ ids: ["new-b"], liveIds: ["new-a", "new-b"] }]));
    expect(graphHarness.staleErrors).not.toContain("UNKNOWN_ELEMENT_PAGE");
    expect(graphHarness.staleErrors).not.toContain("UNKNOWN_ELEMENT:new-a");
    expect(graphHarness.staleErrors).not.toContain("UNKNOWN_ELEMENT:new-b");
    expect(graphHarness.instances).toHaveLength(1);
  });
});
