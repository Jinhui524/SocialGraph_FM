import { describe, expect, it } from "vitest";
import {
  graphCanvasInteractionLodConfig,
  graphCanvasPerformanceProfile,
  graphLabelIdsForZoom,
  graphPresentationGhostNodeIds,
  shouldAutoFitVisibleGraphPane,
  shouldBeginGraphDrag,
} from "./GraphPreview";
import * as graphPreviewModule from "./GraphPreview";

describe("GraphPreview Canvas performance profiles", () => {
  it("derives non-colour node and exact-edge channels from one GovernanceFocus", () => {
    const focusChannels = (graphPreviewModule as unknown as {
      governanceFocusAppearanceChannels?: (
        focus: { readonly nodeIds: readonly string[]; readonly exactRelationKey?: string },
        options: { readonly nodeId?: string; readonly relationKey?: string },
      ) => {
        readonly focused: boolean;
        readonly opacity: number;
        readonly sizeMultiplier: number;
        readonly lineWidthMultiplier: number;
        readonly dualRing: boolean;
        readonly persistentLabel: boolean;
      };
    }).governanceFocusAppearanceChannels;
    expect(focusChannels).toBeTypeOf("function");
    if (!focusChannels) return;
    const focus = { nodeIds: ["n1", "n2"], exactRelationKey: "n1\u0000n2\u0000coRT" };
    expect(focusChannels(focus, { nodeId: "n1" })).toEqual({
      focused: true, opacity: 1, sizeMultiplier: 1.28, lineWidthMultiplier: 1.6, dualRing: true, persistentLabel: true,
    });
    expect(focusChannels(focus, { nodeId: "n3" })).toMatchObject({ focused: false, opacity: 0.28 });
    expect(focusChannels(focus, { relationKey: "n1\u0000n2\u0000coRT" })).toMatchObject({
      focused: true, opacity: 0.94, lineWidthMultiplier: 1.6,
    });
    expect(focusChannels(focus, { relationKey: "n1\u0000n3\u0000coRT" })).toMatchObject({
      focused: false, opacity: 0.12,
    });
  });

  it("does not relayout a replacement projection that is driven by an external governance focus", () => {
    const shouldRelayout = (graphPreviewModule as unknown as {
      shouldRelayoutProjection?: (options: {
        readonly topologyChanged: boolean;
        readonly hasCachedTopology: boolean;
        readonly externalFocusActive: boolean;
      }) => boolean;
    }).shouldRelayoutProjection;
    expect(shouldRelayout).toBeTypeOf("function");
    if (!shouldRelayout) return;
    expect(shouldRelayout({ topologyChanged: true, hasCachedTopology: false, externalFocusActive: true })).toBe(false);
    expect(shouldRelayout({ topologyChanged: true, hasCachedTopology: false, externalFocusActive: false })).toBe(true);
    expect(shouldRelayout({ topologyChanged: true, hasCachedTopology: true, externalFocusActive: false })).toBe(false);
  });

  it("updates non-visible camera and world-center dataset diagnostics together", () => {
    const publishDiagnostics = (graphPreviewModule as unknown as {
      publishGraphCameraDataset?: (
        root: HTMLElement,
        camera: { readonly x: number; readonly y: number; readonly zoom: number },
        worldCenter: { readonly x: number; readonly y: number },
      ) => void;
      publishGraphNodeCoordinateDataset?: (
        root: HTMLElement,
        node: { readonly id: string; readonly x: number; readonly y: number },
      ) => void;
    }).publishGraphCameraDataset;
    const publishNodeCoordinate = (graphPreviewModule as unknown as {
      publishGraphNodeCoordinateDataset?: (
        root: HTMLElement,
        node: { readonly id: string; readonly x: number; readonly y: number },
      ) => void;
    }).publishGraphNodeCoordinateDataset;
    const root = document.createElement("section");

    expect(publishDiagnostics).toBeTypeOf("function");
    publishDiagnostics?.(root, { x: 120, y: 80, zoom: 1.25 }, { x: -14.5, y: 22.25 });
    expect(root.dataset).toMatchObject({
      cameraX: "120",
      cameraY: "80",
      cameraZoom: "1.25",
      worldCenterX: "-14.5",
      worldCenterY: "22.25",
    });

    publishDiagnostics?.(root, { x: 160, y: 95, zoom: 1.25 }, { x: -14.5, y: 22.25 });
    expect(root.dataset.worldCenterX).toBe("-14.5");
    expect(root.dataset.worldCenterY).toBe("22.25");

    expect(publishNodeCoordinate).toBeTypeOf("function");
    publishNodeCoordinate?.(root, { id: "node-7", x: 31.5, y: -8.25 });
    expect(root.dataset).toMatchObject({
      coordinateNodeId: "node-7",
      coordinateNodeX: "31.5",
      coordinateNodeY: "-8.25",
    });
    expect(root).toBeEmptyDOMElement();
  });

  it("bounds synchronous initial force work by visible graph size", () => {
    expect(graphCanvasPerformanceProfile(50).initialIterations).toBe(96);
    expect(graphCanvasPerformanceProfile(180).initialIterations).toBe(96);
    expect(graphCanvasPerformanceProfile(300).initialIterations).toBe(80);
    expect(graphCanvasPerformanceProfile(301).initialIterations).toBe(64);
    expect(graphCanvasPerformanceProfile(1_000).initialIterations).toBe(64);
    expect(graphCanvasPerformanceProfile(1_001).initialIterations).toBe(48);
    expect(graphCanvasPerformanceProfile(3_000).initialIterations).toBe(48);
  });

  it("uses direct manipulation and disables expensive plugins for medium and large graphs", () => {
    const small = graphCanvasPerformanceProfile(120);
    const medium = graphCanvasPerformanceProfile(1_000);
    const large = graphCanvasPerformanceProfile(3_000);

    expect(small).toMatchObject({ directDrag: false, hover: true, minimap: true });
    expect(medium).toMatchObject({ directDrag: true, hover: false, minimap: false });
    expect(large).toMatchObject({
      directDrag: true,
      hover: false,
      minimap: false,
      collision: false,
    });
  });

  it("uses a DisplayObject-only interaction LOD for medium and large Canvas graphs", () => {
    expect(graphCanvasInteractionLodConfig(300)).toMatchObject({
      enabled: false,
      debounce: 160,
    });
    expect(graphCanvasInteractionLodConfig(301)).toEqual({
      enabled: true,
      debounce: 250,
      shapes: { node: ["key"], edge: [] },
    });
  });

  it("derives presentation-only endpoints without changing graph facts", () => {
    const nodes = [{ id: "anchor" }];
    const overlay = {
      id: "research-overlay",
      graphVersionId: "research:email-eu-core",
      kind: "governance" as const,
      nodeValues: { anchor: "subject", outside: "subject" },
      edgeValues: {},
      candidateEdges: [{ id: "candidate", sourceId: "anchor", targetId: "outside", directed: false }],
      legend: { title: "候选", items: [] },
      provenance: { engine: "gfm_research", algorithm: "test" },
    };

    expect(graphPresentationGhostNodeIds(nodes, overlay)).toEqual([]);
    expect(nodes).toEqual([{ id: "anchor" }]);
  });

  it("refits only when a hidden responsive graph pane becomes visible", () => {
    expect(shouldAutoFitVisibleGraphPane(false, true)).toBe(true);
    expect(shouldAutoFitVisibleGraphPane(true, true)).toBe(false);
    expect(shouldAutoFitVisibleGraphPane(false, false)).toBe(false);
  });

  it("keeps click wobble out of the drag/force lane", () => {
    expect(shouldBeginGraphDrag([10, 10], [13, 14])).toBe(false);
    expect(shouldBeginGraphDrag([10, 10], [16, 10])).toBe(false);
    expect(shouldBeginGraphDrag([10, 10], [17, 10])).toBe(true);
  });

  it("caps ordinary labels while preserving selected, focused, and path nodes", () => {
    const nodes = Array.from({ length: 18 }, (_, index) => ({ id: `node-${index}` }));
    const degrees = new Map(nodes.map((node, index) => [node.id, 100 - index]));

    const labels = graphLabelIdsForZoom(nodes, degrees, {
      zoom: 1,
      threshold: 0,
      labelLimit: 12,
      selectedNodeId: "node-17",
      focusNodeIds: ["node-16"],
      pathNodeIds: new Set(["node-15"]),
    });

    expect([...labels].slice(0, 12)).toEqual(
      Array.from({ length: 12 }, (_, index) => `node-${index}`),
    );
    expect(labels.has("node-12")).toBe(false);
    expect(labels.has("node-17")).toBe(true);
    expect(labels.has("node-16")).toBe(true);
    expect(labels.has("node-15")).toBe(true);
    expect(labels.size).toBe(15);
  });

  it("keeps the existing small-graph label behavior when no limit is supplied", () => {
    const nodes = Array.from({ length: 18 }, (_, index) => ({ id: `node-${index}` }));
    const labels = graphLabelIdsForZoom(nodes, new Map(), {
      zoom: 0.4,
      threshold: 100,
      focusNodeIds: [],
      pathNodeIds: new Set(),
    });

    expect(labels.size).toBe(nodes.length);
  });

  it("uses a selection ring and highlighted incident edges in governance views", () => {
    const resolveStates = (graphPreviewModule as unknown as {
      governanceSelectionStates?: (lens: "risk" | "relations" | undefined, related: boolean) => {
        readonly node: readonly string[];
        readonly edge: readonly string[];
      };
    }).governanceSelectionStates;
    expect(resolveStates).toBeTypeOf("function");
    if (!resolveStates) return;

    expect(resolveStates("risk", true)).toEqual({ node: ["governance-selected"], edge: ["governance-focus"] });
    expect(resolveStates("relations", true)).toEqual({ node: ["governance-selected"], edge: ["governance-focus"] });
    expect(resolveStates(undefined, true)).toEqual({ node: ["selected"], edge: ["related"] });
  });
});
