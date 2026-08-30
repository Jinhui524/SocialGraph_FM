import {
  ArrowsOut,
  BracketsCurly,
  CaretDown,
  CaretUp,
  Check,
  Crosshair,
  DownloadSimple,
  Funnel,
  Graph as GraphIcon,
  ImageSquare,
  MagnifyingGlass,
  Minus,
  MoonStars,
  Plus,
  PushPin,
  PushPinSlash,
  Selection,
  Sun,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import {
  Graph,
  GraphEvent,
  NodeEvent,
  type GraphData,
  type IElementEvent,
} from "@antv/g6";
import {
  FormEvent,
  type CSSProperties,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createPortal } from "react-dom";
import type {
  GraphFilters,
  GraphNode,
  GraphRendererPreference,
  GraphRendererStatus,
  GraphTheme,
  GraphViewMode,
} from "../../types/graph";
import { MAX_VISIBLE_EDGES, MAX_VISIBLE_NODES } from "../../types/graph";
import {
  GraphEngineAdapter,
  type GraphSceneLease,
} from "../../services/graphEngineAdapter";
import {
  deterministicGraphInitialPositions,
  graphTopologyKey,
} from "../../services/graphDeterministicLayout";
import { GraphCameraController } from "../../services/graphCameraController";
import { GraphAdjacencyIndex, GraphSpatialIndex } from "../../services/graphSpatialIndex";
import { GraphVisibilityController } from "../../services/graphVisibilityController";
import {
  filterGraphFacts,
  graphFilterConstraintCount,
  normalizeGraphFilters,
} from "../../services/graphFilters";
import {
  isSceneOutsideViewport,
  resolveViewportCullSnapshot,
  type ViewportCullSnapshot,
} from "../../services/graphViewportCulling";
import { LocalForceController, type LocalForceFrame } from "../../services/localForceController";
import {
  GraphPerformanceProbe,
} from "../../services/graphPerformanceProbe";
import {
  detectWebGLSupport,
  loadGraphRenderer,
  resolveGraphRendererKind,
  type LoadedGraphRenderer,
} from "../../services/graphRenderer";
import { graphTypeColour } from "../../services/graphTypePalette";
import {
  GRAPH_PREVIEW_POLICY,
  governanceEdgeStyle,
  governanceExactRelationKey,
  isExactGovernanceRelation,
} from "../../services/graphPreviewPolicy";
import {
  EMPTY_GRAPH_EDGES,
  EMPTY_GRAPH_NODES,
  REFERENCE_NEGATIVE_OUTLINE,
  REFERENCE_POSITIVE_FILL,
  REFERENCE_POSITIVE_OUTLINE,
  TYPE_COLOURS,
  governanceFocusAppearanceChannels,
  governanceSelectionStates,
  graphAppearanceRequestKey,
  graphPresentationGhostNodeIds,
  graphSemanticBadges,
  onlineRiskColour,
  shouldAutoFitVisibleGraphPane,
  shouldBeginGraphDrag,
  shouldRelayoutProjection,
  typeLabel,
} from "./graphPresentation";
import {
  completeCameraSnapshot,
  viewSnapshotKey,
  type GraphPreviewAppearanceSnapshot,
  type GraphPreviewExportHandlers,
  type GraphPreviewProps,
  type GraphPreviewRuntimeMetrics,
  type GraphPreviewViewSnapshot,
} from "./graphPreviewTypes";
import {
  DEFAULT_DISPLAY,
  FORCE_PRESETS,
  graphRendererFallbackWarning,
  rendererPreferenceForRuntime,
  type GraphElementStates,
} from "./graphRendererPolicy";
import {
  automaticLayoutPreset,
  forceLayoutOptions,
  hashText,
  setForceFixedPosition,
} from "./graphLayout";
import {
  graphCanvasInteractionLodConfig,
  graphCanvasPerformanceProfile,
} from "./graphPerformancePolicy";
import {
  downloadDataUrl,
  downloadJson,
  safeFileStem,
} from "./graphExport";
import {
  findCentralDragTarget,
  includesShiftKey,
  readDragTarget,
} from "./graphInteraction";
import {
  compactGraphLabel,
  graphAnimation,
  graphLabelIdsForZoom,
  joinClassNames,
  nodeType,
  prefersReducedMotion,
} from "./graphLabels";
import {
  publishGraphCameraDataset,
  publishGraphNodeCoordinateDataset,
} from "./graphDiagnostics";
import "../../graph-workbench.css";

export {
  governanceFocusAppearanceChannels,
  governanceSelectionStates,
  graphAppearanceRequestKey,
  graphPresentationGhostNodeIds,
  graphSemanticBadges,
  shouldAutoFitVisibleGraphPane,
  shouldBeginGraphDrag,
  shouldRelayoutProjection,
} from "./graphPresentation";
export type {
  GraphPreviewAppearanceSnapshot,
  GraphPreviewExportHandlers,
  GraphPreviewProps,
  GraphPreviewRuntimeMetrics,
  GraphPreviewViewSnapshot,
} from "./graphPreviewTypes";
export { graphRendererFallbackWarning } from "./graphRendererPolicy";
export {
  graphCanvasInteractionLodConfig,
  graphCanvasPerformanceProfile,
} from "./graphPerformancePolicy";
export type {
  GraphCanvasInteractionLodConfig,
  GraphCanvasPerformanceProfile,
} from "./graphPerformancePolicy";
export { graphLabelIdsForZoom } from "./graphLabels";
export {
  publishGraphCameraDataset,
  publishGraphNodeCoordinateDataset,
} from "./graphDiagnostics";

export function GraphPreview({
  graphVersion,
  scene,
  selectedNodeId,
  onSelectNode,
  onNodeSelect,
  className,
  title = "图谱预览",
  headerAccessory,
  ariaLabel = "交互式社交关系图预览",
  emptyState,
  viewMode,
  depth,
  theme,
  rendererPreference,
  labelLimit = GRAPH_PREVIEW_POLICY.labelLimit,
  showNodeRanks = true,
  focusNodeIds,
  pathEndpointIds,
  pinnedNodes,
  activeOverlay,
  governanceFocus,
  filters,
  onViewModeChange,
  onThemeChange,
  onRendererStatus,
  onFocusNodeIdsChange,
  onPathEndpointIdsChange,
  onPinnedNodesChange,
  onFiltersChange,
  onViewStateChange,
  cameraRestoreCommand,
  cameraFocusCommand,
  onExportReady,
  onExport,
  onRuntimeMetrics,
  onAppearanceApplied,
  enableMinimap = false,
  summaryCollapsed = false,
  summaryControlsId = "graph-summary-panel",
  onSummaryCollapsedChange,
  isPaneVisible = true,
  returnToOverviewAction,
}: GraphPreviewProps) {
  const rootRef = useRef<HTMLElement>(null);
  const viewportRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLDivElement>(null);
  const minimapRef = useRef<HTMLDivElement>(null);
  const graphRef = useRef<Graph | null>(null);
  const engineRef = useRef<GraphEngineAdapter | null>(null);
  const cameraControllerRef = useRef<GraphCameraController | null>(null);
  const spatialIndexRef = useRef(new GraphSpatialIndex(64));
  const viewportNodeIdsRef = useRef<ViewportCullSnapshot | undefined>(undefined);
  const adjacencyIndexRef = useRef<GraphAdjacencyIndex | null>(null);
  const visibilityControllerRef = useRef<GraphVisibilityController | null>(null);
  const localForceControllerRef = useRef<LocalForceController | null>(null);
  const visibilityFrameRef = useRef<number | null>(null);
  const viewportGestureActiveRef = useRef(false);
  const viewportCullingPausedRef = useRef(false);
  const visibilityApplyQueueRef = useRef<{
    inFlight: boolean;
    pending: { graph: Graph; sceneKey: string; viewportNodeIds?: ReadonlySet<string> } | null;
    waiters: Array<{ resolve: () => void; reject: (error: unknown) => void }>;
  }>({ inFlight: false, pending: null, waiters: [] });
  const pendingForceFrameRef = useRef<LocalForceFrame | null>(null);
  const forceFrameRef = useRef<number | null>(null);
  const forceApplyInFlightRef = useRef(false);
  const localForceMovedNodeIdsRef = useRef(new Set<string>());
  const localForceNeighborDeltaMaxRef = useRef(0);
  const localForceSettleTimerRef = useRef<number | null>(null);
  const performanceProbeRef = useRef<GraphPerformanceProbe | null>(null);
  const lastPerformancePublishAtRef = useRef(0);
  const lastCreatedGraphIdentityRef = useRef<string | null>(null);
  const positionCacheRef = useRef(
    new Map<string, Map<string, { readonly x: number; readonly y: number }>>(),
  );
  const labelZoomBandRef = useRef("");
  const transformTimerRef = useRef<number | null>(null);
  const resizeFrameRef = useRef<number | null>(null);
  const interactionReleaseTimerRef = useRef<number | null>(null);
  const searchFeedbackTimerRef = useRef<number | null>(null);
  // Camera focus is asynchronous. Incrementing this token prevents a stale
  // search task from restoring neighbour emphasis after Escape or a genuine
  // blank-canvas click has returned the graph to browse mode.
  const transientFocusRequestRef = useRef(0);
  const viewportFeedbackTimerRef = useRef<number | null>(null);
  const paneWasVisibleRef = useRef(isPaneVisible);
  const isPaneVisibleRef = useRef(isPaneVisible);
  isPaneVisibleRef.current = isPaneVisible;
  const paneActivationRef = useRef(0);
  const paneActivationVisibilityRef = useRef(isPaneVisible);
  if (isPaneVisible && !paneActivationVisibilityRef.current) {
    paneActivationRef.current += 1;
  }
  paneActivationVisibilityRef.current = isPaneVisible;
  const pendingVisiblePaneFitRef = useRef(false);
  const fittedVisiblePaneTopologiesRef = useRef(new Set<string>());
  const forceUpdateSequenceRef = useRef(0);
  const lastForceKeyRef = useRef("");
  const appliedPinnedIdsRef = useRef(new Set<string>());
  const highlightedElementsRef = useRef({
    nodes: new Set<string>(),
    edges: new Set<string>(),
  });
  const clearGraphHighlightRef = useRef<() => void>(() => undefined);
  const lastCameraRef = useRef({ x: 0, y: 0, zoom: 1 });
  const zoomRequestRef = useRef(0);
  const pendingZoomTargetRef = useRef<number | null>(null);
  const sceneCameraCommandTokenRef = useRef(0);
  const lastWorldCenterRef = useRef<{ x: number; y: number } | null>(null);
  const lastViewportSizeRef = useRef<readonly [number, number] | null>(null);
  const diagnosticNodeIdRef = useRef<string | null>(null);
  const hasStableCameraRef = useRef(false);
  const exportPngImplRef = useRef<() => Promise<void>>(async () => undefined);
  const exportJsonImplRef = useRef<() => void>(() => undefined);
  const exportHandlersRef = useRef<GraphPreviewExportHandlers>({
    exportPng: () => exportPngImplRef.current(),
    exportJson: () => exportJsonImplRef.current(),
  });
  const onNodeSelectRef = useRef(onSelectNode ?? onNodeSelect);
  const activeSelectedIdRef = useRef<string | null>(selectedNodeId ?? null);
  const onViewStateChangeRef = useRef(onViewStateChange);
  const onRuntimeMetricsRef = useRef(onRuntimeMetrics);
  const onAppearanceAppliedRef = useRef(onAppearanceApplied);
  onAppearanceAppliedRef.current = onAppearanceApplied;
  const returnToOverviewActionRef = useRef(returnToOverviewAction);
  returnToOverviewActionRef.current = returnToOverviewAction;
  const onRendererStatusRef = useRef(onRendererStatus);
  const lastEmittedViewSnapshotKeyRef = useRef("");
  const runtimeMetricsRef = useRef<GraphPreviewRuntimeMetrics>({
    ready: false,
    visibleNodes: 0,
    visibleEdges: 0,
    engineCreateCount: 0,
    engineDestroyCount: 0,
    layoutCount: 0,
    drawCount: 0,
    fitViewCount: 0,
    rendererRequested: rendererPreferenceForRuntime(rendererPreference),
    rendererResolved: "canvas",
    webglSupported: false,
    webglContextLossCount: 0,
    mutationInFlight: 0,
    mutationInFlightMax: 0,
    viewportCullingPaused: false,
    interactionLodActive: false,
    localForceFrameCount: 0,
    localForceMovedNodeCount: 0,
    localForceNeighborDeltaMax: 0,
    localForceSettledGeneration: 0,
  });
  const [internalSelectedId, setInternalSelectedId] = useState<string | null>(
    selectedNodeId ?? null,
  );
  const [internalViewMode, setInternalViewMode] =
    useState<GraphViewMode>("global");
  const [internalDepth] = useState<1 | 2 | 3>(2);
  const [internalTheme, setInternalTheme] =
    useState<GraphTheme>("brand-light");
  const [loadedRendererRequest, setLoadedRendererRequest] =
    useState<{
      readonly requestKey: string;
      readonly value: LoadedGraphRenderer;
    } | null>(null);
  const [rendererStatus, setRendererStatus] = useState<GraphRendererStatus>({
    requested: rendererPreferenceForRuntime(rendererPreference),
    resolved: "canvas",
    webglSupported: false,
    contextLossCount: 0,
  });
  const [rendererFailure, setRendererFailure] = useState<{
    readonly graphIdentity: string;
    readonly preference: GraphRendererPreference;
    readonly reason: string;
  } | null>(null);
  const [webglContextLossCount, setWebglContextLossCount] = useState(0);
  const [internalFocusNodeIds, setInternalFocusNodeIds] = useState<
    readonly string[]
  >([]);
  const [internalPathEndpointIds, setInternalPathEndpointIds] = useState<
    readonly string[]
  >([]);
  const [internalPinnedNodes, setInternalPinnedNodes] = useState<
    Readonly<Record<string, { readonly x: number; readonly y: number }>>
  >({});
  const displaySettings = DEFAULT_DISPLAY;
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [activeSearchIndex, setActiveSearchIndex] = useState(0);
  const [searchFeedback, setSearchFeedback] = useState("");
  const [pendingSearchNodeId, setPendingSearchNodeId] = useState<string | null>(null);
  const [status, setStatus] = useState("图谱可拖拽、缩放并选择节点");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [engineReady, setEngineReady] = useState(false);
  const [renderedLabelCount, setRenderedLabelCount] = useState(0);
  const [committedSceneDataKey, setCommittedSceneDataKey] = useState("");
  const committedSceneDataKeyRef = useRef("");
  const [renderError, setRenderError] = useState<string | null>(null);
  const [filterOpen, setFilterOpen] = useState(false);
  const [filterPosition, setFilterPosition] = useState<CSSProperties>({});
  const [filterAsSheet, setFilterAsSheet] = useState(false);
  const [internalFilters, setInternalFilters] = useState<GraphFilters>({
    nodeTypes: [],
    edgeTypes: [],
  });
  const cameraRestoreTokenScope = cameraRestoreCommand?.commandScope
    ?? (cameraRestoreCommand && "workspace" in cameraRestoreCommand
      ? `workspace:${String(cameraRestoreCommand.workspace)}`
      : "governance-presentation");
  const cameraFocusTokenScope = cameraFocusCommand?.commandScope ?? "external-focus";
  const [draftFilters, setDraftFilters] = useState<GraphFilters>({
    nodeTypes: [],
    edgeTypes: [],
  });
  const [isGraphOffscreen, setIsGraphOffscreen] = useState(false);
  const [viewportFeedback, setViewportFeedback] = useState("");
  const filterButtonRef = useRef<HTMLButtonElement>(null);
  const filterPopoverRef = useRef<HTMLDivElement>(null);
  const searchFormRef = useRef<HTMLFormElement>(null);

  useEffect(
    () => () => {
      if (searchFeedbackTimerRef.current !== null) {
        window.clearTimeout(searchFeedbackTimerRef.current);
      }
      if (viewportFeedbackTimerRef.current !== null) {
        window.clearTimeout(viewportFeedbackTimerRef.current);
      }
    },
    [],
  );

  const preview = graphVersion?.preview;
  const fullNodes = graphVersion?.nodes ?? scene?.nodes ?? preview?.nodes ?? EMPTY_GRAPH_NODES;
  const fullEdges = graphVersion?.edges ?? scene?.edges ?? preview?.edges ?? EMPTY_GRAPH_EDGES;
  const canRenderWholeVersion = Boolean(
    graphVersion &&
      graphVersion.nodes.length <= MAX_VISIBLE_NODES &&
      graphVersion.edges.length <= MAX_VISIBLE_EDGES,
  );
  // The renderer owns one stable preview topology. Mode, depth and filters are
  // visibility masks, so they cannot restart layout or reset the camera.
  const requestedSceneNodes = scene?.nodes.length
    ? scene.nodes
    : canRenderWholeVersion ? graphVersion?.nodes ?? EMPTY_GRAPH_NODES : preview?.nodes ?? EMPTY_GRAPH_NODES;
  const requestedSceneEdges = scene?.nodes.length
    ? scene.edges
    : canRenderWholeVersion ? graphVersion?.edges ?? EMPTY_GRAPH_EDGES : preview?.edges ?? EMPTY_GRAPH_EDGES;
  const nodes = canRenderWholeVersion
    ? graphVersion?.nodes ?? EMPTY_GRAPH_NODES
    : scene?.nodes ?? preview?.nodes ?? EMPTY_GRAPH_NODES;
  const edges = canRenderWholeVersion
    ? graphVersion?.edges ?? EMPTY_GRAPH_EDGES
    : scene?.edges ?? preview?.edges ?? EMPTY_GRAPH_EDGES;
  const sceneNodeIds = useMemo(
    () => new Set(requestedSceneNodes.map((node) => node.id)),
    [requestedSceneNodes],
  );
  const sceneEdgeIds = useMemo(
    () => new Set(requestedSceneEdges.map((edge) => edge.id)),
    [requestedSceneEdges],
  );
  const activeViewMode = viewMode ?? internalViewMode;
  const activeDepth = depth ?? internalDepth;
  const activeTheme = theme ?? internalTheme;
  const activeLayoutPreset = automaticLayoutPreset(nodes.length);
  const forceSettings = FORCE_PRESETS[activeLayoutPreset];
  const activeRendererPreference = rendererPreferenceForRuntime(rendererPreference);
  const sourceFocusNodeIds =
    focusNodeIds ?? scene?.focusNodeIds ?? internalFocusNodeIds;
  const sourcePathEndpointIds = pathEndpointIds ?? internalPathEndpointIds;
  const focusNodeIdsKey = sourceFocusNodeIds.join("\u0000");
  const pathEndpointIdsKey = sourcePathEndpointIds.join("\u0000");
  const activeFocusNodeIds = useMemo(
    () => [...sourceFocusNodeIds],
    // Keep controlled arrays stable when a parent emits an equivalent [] literal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [focusNodeIdsKey],
  );
  const activePathEndpointIds = useMemo(
    () => [...sourcePathEndpointIds],
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pathEndpointIdsKey],
  );
  const activePinnedNodes = pinnedNodes ?? internalPinnedNodes;
  const overlay = activeOverlay ?? scene?.overlay ?? null;
  const pathNodeIdsKey = (scene?.pathNodeIds ?? []).join("\u0000");
  const pathEdgeIdsKey = (scene?.pathEdgeIds ?? []).join("\u0000");
  const pathNodeIds = useMemo(
    () => new Set(scene?.pathNodeIds ?? []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pathNodeIdsKey],
  );
  const pathEdgeIds = useMemo(
    () => new Set(scene?.pathEdgeIds ?? []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [pathEdgeIdsKey],
  );

  useEffect(() => {
    if (graphVersion || scene) return;
    setStatus("等待图谱数据");
    setRenderError(null);
  }, [graphVersion, scene]);

  const types = useMemo(
    () =>
      Array.from(new Set(fullNodes.map(nodeType))).sort((a, b) =>
        a.localeCompare(b, "zh-CN"),
      ),
    [fullNodes],
  );
  const edgeTypes = useMemo(
    () =>
      Array.from(
        new Set(fullEdges.map((edge) => edge.type?.trim()).filter((type): type is string => Boolean(type))),
      ).sort((left, right) => left.localeCompare(right, "zh-CN")),
    [fullEdges],
  );
  const activeFilters = filters ?? internalFilters;
  const activeFilterConstraintCount = useMemo(
    () => graphFilterConstraintCount(activeFilters),
    [activeFilters],
  );
  const enabledTypes = useMemo(() => {
    if (activeFilters.nodeTypes.length === 0) return new Set(types);
    const available = new Set(types);
    return new Set(activeFilters.nodeTypes.filter((type) => available.has(type)));
  }, [activeFilters.nodeTypes, types]);
  const enabledTypesRef = useRef<ReadonlySet<string>>(enabledTypes);

  const colourByType = useMemo(
    () =>
      new Map(
        types.map((type) => [
          type,
          graphTypeColour(type),
        ]),
      ),
    [types],
  );

  const nodeById = useMemo(
    () => new Map(fullNodes.map((node) => [node.id, node])),
    [fullNodes],
  );

  const degreeById = useMemo(() => {
    const degrees = new Map(fullNodes.map((node) => [node.id, 0]));
    for (const edge of fullEdges) {
      degrees.set(edge.source, (degrees.get(edge.source) ?? 0) + 1);
      if (edge.target !== edge.source) {
        degrees.set(edge.target, (degrees.get(edge.target) ?? 0) + 1);
      }
    }
    return degrees;
  }, [fullEdges, fullNodes]);

  const renderedDegreeById = useMemo(() => {
    const degrees = new Map(nodes.map((node) => [node.id, 0]));
    for (const edge of edges) {
      degrees.set(edge.source, (degrees.get(edge.source) ?? 0) + 1);
      if (edge.target !== edge.source) {
        degrees.set(edge.target, (degrees.get(edge.target) ?? 0) + 1);
      }
    }
    return degrees;
  }, [edges, nodes]);

  const adjacencyIndex = useMemo(() => new GraphAdjacencyIndex(edges), [edges]);
  adjacencyIndexRef.current = adjacencyIndex;

  const selectionIsControlled =
    selectedNodeId !== undefined || Boolean(onSelectNode);
  const activeSelectedId = selectionIsControlled
    ? selectedNodeId ?? null
    : internalSelectedId;

  const protectedVisibilityIds = useMemo(
    () =>
      new Set([
        ...(activeSelectedId ? [activeSelectedId] : []),
        ...activeFocusNodeIds,
        ...activePathEndpointIds,
        ...pathNodeIds,
        ...Object.keys(activePinnedNodes),
        ...Object.keys(overlay?.nodeValues ?? {}),
      ]),
    [activeFocusNodeIds, activePathEndpointIds, activePinnedNodes, activeSelectedId, overlay, pathNodeIds],
  );

  const visibilityRequestRef = useRef({
    filters: activeFilters,
    sceneNodeIds,
    sceneEdgeIds,
    protectedNodeIds: protectedVisibilityIds,
  });
  visibilityRequestRef.current = {
    filters: activeFilters,
    sceneNodeIds,
    sceneEdgeIds,
    protectedNodeIds: protectedVisibilityIds,
  };

  const visualConfigRef = useRef({
    theme: activeTheme,
    display: displaySettings,
    overlay,
    governanceFocus,
    selectedNodeId: activeSelectedId ?? null,
    focusNodeIds: activeFocusNodeIds,
    pathNodeIds,
    pathEdgeIds,
    pinnedNodes: activePinnedNodes,
  });
  visualConfigRef.current = {
    theme: activeTheme,
    display: displaySettings,
    overlay,
    governanceFocus,
    selectedNodeId: activeSelectedId ?? null,
    focusNodeIds: activeFocusNodeIds,
    pathNodeIds,
    pathEdgeIds,
    pinnedNodes: activePinnedNodes,
  };
  const appearanceRequestKey = graphAppearanceRequestKey(governanceFocus);
  const graphIdentity =
    graphVersion?.id ?? scene?.graphVersionId ?? (scene ? "standalone-scene" : "empty");
  const sceneDataKey = useMemo(() => {
    const nodeKey = nodes.map((node) => node.id).join("\u0001");
    const edgeKey = edges
      .map((edge) => `${edge.id}\u0002${edge.source}\u0002${edge.target}`)
      .join("\u0001");
    return `${graphIdentity}\u0000${nodes.length}:${edges.length}:${hashText(nodeKey)}:${hashText(edgeKey)}`;
  }, [edges, graphIdentity, nodes]);
  const topologyKey = useMemo(() => graphTopologyKey(nodes, edges), [edges, nodes]);

  const publishRuntimeMetrics = useCallback(
    (patch: Partial<GraphPreviewRuntimeMetrics>) => {
      const next = { ...runtimeMetricsRef.current, ...patch };
      runtimeMetricsRef.current = next;
      const root = rootRef.current;
      if (root) {
        root.dataset.graphReady = String(next.ready);
        root.dataset.visibleNodes = String(next.visibleNodes);
        root.dataset.visibleEdges = String(next.visibleEdges);
        root.dataset.engineCreateCount = String(next.engineCreateCount);
        root.dataset.engineDestroyCount = String(next.engineDestroyCount);
        root.dataset.layoutCount = String(next.layoutCount);
        root.dataset.drawCount = String(next.drawCount);
        root.dataset.fitViewCount = String(next.fitViewCount);
        root.dataset.rendererRequested = next.rendererRequested;
        root.dataset.rendererResolved = next.rendererResolved;
        root.dataset.webglSupported = String(next.webglSupported);
        root.dataset.webglContextLossCount = String(next.webglContextLossCount);
        if (next.rendererFallbackReason) {
          root.dataset.rendererFallbackReason = next.rendererFallbackReason;
        } else {
          delete root.dataset.rendererFallbackReason;
        }
        if (next.webglLazyLoadMs !== undefined) {
          root.dataset.webglLazyLoadMs = String(next.webglLazyLoadMs);
        } else {
          delete root.dataset.webglLazyLoadMs;
        }
        if (next.dragTarget) {
          root.dataset.draggableNodeId = next.dragTarget.nodeId;
          root.dataset.draggableX = String(next.dragTarget.x);
          root.dataset.draggableY = String(next.dragTarget.y);
        } else {
          delete root.dataset.draggableNodeId;
          delete root.dataset.draggableX;
          delete root.dataset.draggableY;
        }
        if (next.lastDraggedNodeId) {
          root.dataset.lastDraggedId = next.lastDraggedNodeId;
          root.dataset.lastDragPinned = String(Boolean(next.lastDragPinned));
        }
        if (next.spatialPickMs !== undefined) root.dataset.spatialPickMs = String(next.spatialPickMs);
        if (next.spatialPickCandidates !== undefined) {
          root.dataset.spatialPickCandidates = String(next.spatialPickCandidates);
        }
        if (next.pickOracleChecked !== undefined) {
          root.dataset.pickOracleChecked = String(next.pickOracleChecked);
          root.dataset.pickOracleMismatches = String(next.pickOracleMismatches ?? 0);
          root.dataset.pickOracleP95Ms = String(next.pickOracleP95Ms ?? 0);
          root.dataset.pickOracleCandidatesP95 = String(next.pickOracleCandidatesP95 ?? 0);
        }
        if (next.workerRoundTripMs !== undefined) {
          root.dataset.workerRoundTripMs = String(next.workerRoundTripMs);
        }
        if (next.workerComputeMs !== undefined) root.dataset.workerComputeMs = String(next.workerComputeMs);
        if (next.positionApplyMs !== undefined) root.dataset.positionApplyMs = String(next.positionApplyMs);
        root.dataset.mutationInFlight = String(next.mutationInFlight);
        root.dataset.mutationInFlightMax = String(next.mutationInFlightMax);
        root.dataset.viewportCullingPaused = String(next.viewportCullingPaused);
        root.dataset.interactionLodActive = String(next.interactionLodActive);
        root.dataset.localForceFrameCount = String(next.localForceFrameCount);
        root.dataset.localForceMovedNodeCount = String(next.localForceMovedNodeCount);
        root.dataset.localForceNeighborDeltaMax = String(next.localForceNeighborDeltaMax);
        root.dataset.localForceSettledGeneration = String(next.localForceSettledGeneration);
        if (next.culledNodes !== undefined) root.dataset.culledNodes = String(next.culledNodes);
        if (next.culledEdges !== undefined) root.dataset.culledEdges = String(next.culledEdges);
      }
      onRuntimeMetricsRef.current?.(next);
    },
    [],
  );

  const publishNodeCoordinateDiagnostics = useCallback(() => {
    const root = rootRef.current;
    const graph = graphRef.current;
    const nodeId = diagnosticNodeIdRef.current;
    if (!root || !graph || graph.destroyed || !nodeId) return;
    try {
      const position = graph.getElementPosition(nodeId);
      const x = Number(position[0]);
      const y = Number(position[1]);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      publishGraphNodeCoordinateDataset(root, { id: nodeId, x, y });
    } catch {
      // A scene replacement can remove the diagnostic node before the next
      // stable target is selected. Keep the last finite snapshot until then.
    }
  }, []);

  const publishCameraDiagnostics = useCallback((
    camera: { readonly x: number; readonly y: number; readonly zoom: number },
    worldCenter: { readonly x: number; readonly y: number } | null = lastWorldCenterRef.current,
  ) => {
    lastCameraRef.current = camera;
    if (worldCenter) lastWorldCenterRef.current = worldCenter;
    const viewport = viewportRef.current;
    if (
      isPaneVisibleRef.current
      && viewport
      && viewport.clientWidth > 0
      && viewport.clientHeight > 0
    ) {
      lastViewportSizeRef.current = [viewport.clientWidth, viewport.clientHeight];
    }
    const root = rootRef.current;
    if (!root) return;
    if (worldCenter) {
      publishGraphCameraDataset(root, camera, worldCenter);
      publishNodeCoordinateDiagnostics();
      return;
    }
    root.dataset.cameraX = String(camera.x);
    root.dataset.cameraY = String(camera.y);
    root.dataset.cameraZoom = String(camera.zoom);
    publishNodeCoordinateDiagnostics();
  }, [publishNodeCoordinateDiagnostics]);

  const applyUnifiedVisibility = useCallback(
    (graph: Graph, viewportNodeIds?: ReadonlySet<string>): Promise<void> =>
      new Promise<void>((resolve, reject) => {
        // Viewport culling is a post-gesture optimization. Applying thousands
        // of visibility changes while the pointer is moving both competes
        // with the drag mutation lane and makes elements flicker at the edge.
        if (viewportNodeIds && viewportCullingPausedRef.current) {
          resolve();
          return;
        }
        const queue = visibilityApplyQueueRef.current;
        queue.pending = { graph, sceneKey: sceneDataKey, ...(viewportNodeIds ? { viewportNodeIds } : {}) };
        queue.waiters.push({ resolve, reject });
        if (queue.inFlight) return;
        queue.inFlight = true;
        const drain = async () => {
          while (queue.pending) {
            const request = queue.pending;
            const waiters = queue.waiters.splice(0);
            queue.pending = null;
            const controller = visibilityControllerRef.current;
            const engine = request.graph === graphRef.current ? engineRef.current : null;
            if (
              !controller
              || !engine
              || request.graph.destroyed
              || request.sceneKey !== renderedSceneKeyRef.current
            ) {
              for (const waiter of waiters) waiter.resolve();
              continue;
            }
            try {
              const startedAt = performance.now();
              const result = await controller.apply(request.graph, {
                ...visibilityRequestRef.current,
                ...(request.viewportNodeIds
                  ? { viewportNodeIds: request.viewportNodeIds }
                  : {}),
              }, async (changes) => {
                await engine.applyVisibility(changes, request.sceneKey);
              });
              const elapsed = performance.now() - startedAt;
              if (request.viewportNodeIds) {
                performanceProbeRef.current?.record("viewport_cull", elapsed, {
                  count: result.visibleNodeCount,
                  detail: {
                    culledNodes: result.culledNodeCount,
                    culledEdges: result.culledEdgeCount,
                    batches: result.applyBatchCount ?? 0,
                  },
                });
              }
              performanceProbeRef.current?.record("visibility_compute", result.computeMs ?? 0, {
                count: result.visibleNodeCount,
              });
              performanceProbeRef.current?.record("visibility_apply", result.applyMs ?? elapsed, {
                count: Object.keys(result.changes).length,
                detail: { batches: result.applyBatchCount ?? 0 },
              });
              publishRuntimeMetrics({
                visibleNodes: result.visibleNodeCount,
                visibleEdges: result.visibleEdgeCount,
                culledNodes: result.culledNodeCount,
                culledEdges: result.culledEdgeCount,
                performanceSamples: performanceProbeRef.current?.snapshot(),
              });
              for (const waiter of waiters) waiter.resolve();
            } catch (error) {
              for (const waiter of waiters) waiter.reject(error);
            }
          }
          queue.inFlight = false;
        };
        void drain();
      }),
    [publishRuntimeMetrics, sceneDataKey],
  );

  useEffect(() => {
    onNodeSelectRef.current = onSelectNode ?? onNodeSelect;
  }, [onNodeSelect, onSelectNode]);

  useEffect(() => {
    onViewStateChangeRef.current = onViewStateChange;
  }, [onViewStateChange]);

  useEffect(() => {
    if (!engineReady || !cameraRestoreCommand || committedSceneDataKey !== sceneDataKey) return;
    const viewport = viewportRef.current;
    if (!isPaneVisible || !viewport || viewport.clientWidth <= 0 || viewport.clientHeight <= 0) return;
    transientFocusRequestRef.current += 1;
    const legacyCamera = "position" in cameraRestoreCommand ? {
      x: Number(cameraRestoreCommand.position[0]),
      y: Number(cameraRestoreCommand.position[1]),
      zoom: cameraRestoreCommand.zoom,
    } : { x: cameraRestoreCommand.x, y: cameraRestoreCommand.y, zoom: cameraRestoreCommand.zoom };
    const snapshot = "position" in cameraRestoreCommand
      ? cameraRestoreCommand
      : {
          position: [legacyCamera.x, legacyCamera.y] as [number, number],
          zoom: legacyCamera.zoom,
        };
    let cancelled = false;
    void engineRef.current?.restoreCamera(
      snapshot,
      sceneDataKey,
      cameraRestoreCommand.token,
      cameraRestoreTokenScope,
      paneActivationRef.current,
    ).then((restored) => {
        if (cancelled || !restored) return;
        const captured = engineRef.current?.captureCamera();
        if (!captured) return;
        publishCameraDiagnostics({
          x: Number(captured.position[0]),
          y: Number(captured.position[1]),
          zoom: captured.zoom,
        }, captured.worldCenter
          ? { x: Number(captured.worldCenter[0]), y: Number(captured.worldCenter[1]) }
          : lastWorldCenterRef.current);
      });
    return () => {
      cancelled = true;
    };
  }, [
    cameraRestoreCommand?.token,
    cameraRestoreTokenScope,
    engineReady,
    committedSceneDataKey,
    isPaneVisible,
    sceneDataKey,
    publishCameraDiagnostics,
  ]);

  useEffect(() => {
    if (!engineReady || !cameraFocusCommand?.nodeIds.length || !isPaneVisible) return;
    const viewport = viewportRef.current;
    if (!viewport || viewport.clientWidth <= 0 || viewport.clientHeight <= 0) return;
    const request = transientFocusRequestRef.current + 1;
    transientFocusRequestRef.current = request;
    if (committedSceneDataKey !== sceneDataKey || renderedSceneKeyRef.current !== sceneDataKey) return;
    if (cameraFocusCommand.projectionIdentity && cameraFocusCommand.projectionIdentity !== topologyKey) return;
    const graph = graphRef.current;
    const camera = cameraControllerRef.current;
    if (!graph || graph.destroyed || !camera) return;
    const nodeIds = cameraFocusCommand.nodeIds.filter((id) => nodeById.has(id));
    if (nodeIds.length !== cameraFocusCommand.nodeIds.length) return;
    const anchorElementId = cameraFocusCommand.anchorNodeId && nodeIds.includes(cameraFocusCommand.anchorNodeId)
      ? cameraFocusCommand.anchorNodeId
      : nodeIds[0];
    if (!anchorElementId) return;
    void (async () => {
      try {
        const visible = await engineRef.current?.ensureVisible(
          nodeIds,
          sceneDataKey,
          cameraFocusCommand.token,
          cameraFocusTokenScope,
          paneActivationRef.current,
        );
        if (!visible || request !== transientFocusRequestRef.current || graph.destroyed) return;
        const focused = await engineRef.current?.runCameraForScene(
          sceneDataKey,
          cameraFocusCommand.token,
          async () => {
            await camera.focus(nodeIds, {
              anchorElementId,
              minZoom: 0.62,
              maxZoom: 1.28,
              animation: prefersReducedMotion() ? false : { duration: 280, easing: "ease-out" },
            });
          },
          cameraFocusTokenScope,
          paneActivationRef.current,
        );
        if (!focused || request !== transientFocusRequestRef.current || graph.destroyed) return;
        const captured = engineRef.current?.captureCamera();
        if (captured) {
          publishCameraDiagnostics({
            x: Number(captured.position[0]),
            y: Number(captured.position[1]),
            zoom: captured.zoom,
          }, captured.worldCenter
            ? { x: Number(captured.worldCenter[0]), y: Number(captured.worldCenter[1]) }
            : lastWorldCenterRef.current);
        }
      } catch (error) {
        if (request !== transientFocusRequestRef.current || graph.destroyed) return;
        setRenderError(`图谱聚焦失败：${error instanceof Error ? error.message : "未知错误"}`);
      }
    })();
    return () => {
      if (transientFocusRequestRef.current === request) transientFocusRequestRef.current += 1;
    };
  }, [cameraFocusCommand?.token, cameraFocusCommand?.projectionIdentity, cameraFocusTokenScope, committedSceneDataKey, engineReady, isPaneVisible, nodeById, nodes.length, publishCameraDiagnostics, sceneDataKey, topologyKey]);

  useEffect(() => {
    onRuntimeMetricsRef.current = onRuntimeMetrics;
    onRuntimeMetrics?.(runtimeMetricsRef.current);
  }, [onRuntimeMetrics]);

  useEffect(() => {
    onRendererStatusRef.current = onRendererStatus;
    onRendererStatus?.(rendererStatus);
  }, [onRendererStatus, rendererStatus]);

  useEffect(() => {
    if (engineReady) return;
    publishRuntimeMetrics({
      visibleNodes: requestedSceneNodes.length,
      visibleEdges: requestedSceneEdges.length,
    });
  }, [engineReady, publishRuntimeMetrics, requestedSceneEdges.length, requestedSceneNodes.length]);

  useEffect(() => {
    activeSelectedIdRef.current = activeSelectedId ?? null;
  }, [activeSelectedId]);

  useEffect(() => {
    enabledTypesRef.current = enabledTypes;
  }, [enabledTypes]);

  useEffect(() => {
    setInternalFilters({ nodeTypes: [], edgeTypes: [] });
    setDraftFilters({ nodeTypes: [], edgeTypes: [] });
    setInternalSelectedId(null);
    setInternalViewMode("global");
    setInternalFocusNodeIds([]);
    setInternalPathEndpointIds([]);
    setInternalPinnedNodes({});
    // Keep topology-scoped coordinates when a session is removed/reopened.
    // Clearing this map here was the reason Cornell could return in a radial
    // fallback after an empty-session transition.
    highlightedElementsRef.current = {
      nodes: new Set<string>(),
      edges: new Set<string>(),
    };
    setQuery("");
    setSearchOpen(false);
    setActiveSearchIndex(0);
    setSearchFeedback("");
    setPendingSearchNodeId(null);
    setRenderError(null);
  }, [graphVersion?.id, types]);

  const updateFilterPosition = useCallback(() => {
    const trigger = filterButtonRef.current;
    const root = rootRef.current;
    if (!trigger || !root) return;
    const triggerRect = trigger.getBoundingClientRect();
    const useSheet = window.innerWidth <= 720 || root.getBoundingClientRect().width <= 420;
    setFilterAsSheet(useSheet);
    if (useSheet) {
      setFilterPosition({});
      return;
    }
    const width = 320;
    const margin = 12;
    const desiredHeight = Math.min(360, window.innerHeight - margin * 2);
    const spaceBelow = window.innerHeight - triggerRect.bottom - margin;
    const placeBelow = spaceBelow >= Math.min(220, desiredHeight);
    setFilterPosition({
      width,
      left: Math.max(
        margin,
        Math.min(triggerRect.right - width, window.innerWidth - width - margin),
      ),
      top: placeBelow
        ? triggerRect.bottom + 8
        : Math.max(margin, triggerRect.top - desiredHeight - 8),
      maxHeight: desiredHeight,
    });
  }, []);

  const restoreFilterTriggerFocus = useCallback(() => {
    window.requestAnimationFrame(() => filterButtonRef.current?.focus());
  }, []);

  const closeFilter = useCallback(
    (restoreFocus = true) => {
      setFilterOpen(false);
      if (restoreFocus) restoreFilterTriggerFocus();
    },
    [restoreFilterTriggerFocus],
  );

  useEffect(() => {
    if (!filterOpen) return;
    updateFilterPosition();
    const focusFrame = window.requestAnimationFrame(() => {
      const firstCheckbox = filterPopoverRef.current?.querySelector<HTMLInputElement>(
        'input[type="checkbox"]:not(:disabled)',
      );
      const closeButton = filterPopoverRef.current?.querySelector<HTMLButtonElement>(
        ".graph-preview__filter-close:not(:disabled)",
      );
      (firstCheckbox ?? closeButton)?.focus();
    });
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node;
      if (
        filterButtonRef.current?.contains(target) ||
        filterPopoverRef.current?.contains(target)
      ) {
        return;
      }
      closeFilter();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      closeFilter();
    };
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    window.addEventListener("resize", updateFilterPosition);
    window.addEventListener("scroll", updateFilterPosition, true);
    return () => {
      window.cancelAnimationFrame(focusFrame);
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("resize", updateFilterPosition);
      window.removeEventListener("scroll", updateFilterPosition, true);
    };
  }, [closeFilter, filterOpen, updateFilterPosition]);

  useEffect(() => {
    if (!searchOpen) return;
    const closeSearch = (event: PointerEvent) => {
      if (searchFormRef.current?.contains(event.target as Node)) return;
      setSearchOpen(false);
    };
    document.addEventListener("pointerdown", closeSearch);
    return () => document.removeEventListener("pointerdown", closeSearch);
  }, [searchOpen]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (
      !engineReady
      || committedSceneDataKey !== sceneDataKey
      || !isPaneVisibleRef.current
      || !viewport
      || viewport.clientWidth <= 0
      || viewport.clientHeight <= 0
    ) return;
    const position = graphRef.current?.getPosition() ?? [0, 0];
    const legacyCamera = {
      x: Number(position[0] ?? lastCameraRef.current.x),
      y: Number(position[1] ?? lastCameraRef.current.y),
      zoom: graphRef.current?.getZoom() ?? lastCameraRef.current.zoom,
    };
    const captured = engineRef.current?.captureCamera();
    const camera = captured?.sceneIdentity === sceneDataKey
      ? captured
      : completeCameraSnapshot(
          sceneDataKey,
          legacyCamera,
          lastWorldCenterRef.current,
          [viewportRef.current?.clientWidth ?? 1, viewportRef.current?.clientHeight ?? 1],
        );
    const snapshot: GraphPreviewViewSnapshot = {
      graphVersionId: graphVersion?.id ?? scene?.graphVersionId,
      mode: activeViewMode,
      depth: activeDepth,
      theme: activeTheme,
      layoutPreset: activeLayoutPreset,
      rendererPreference: activeRendererPreference,
      focusNodeIds: [...activeFocusNodeIds],
      pathEndpointIds: [...activePathEndpointIds],
      pinnedNodes: activePinnedNodes,
      camera,
    };
    lastCameraRef.current = legacyCamera;
    const snapshotKey = viewSnapshotKey(snapshot);
    if (snapshotKey === lastEmittedViewSnapshotKeyRef.current) return;
    lastEmittedViewSnapshotKeyRef.current = snapshotKey;
    onViewStateChangeRef.current?.(snapshot);
  }, [
    activeDepth,
    activeFocusNodeIds,
    activeLayoutPreset,
    activeRendererPreference,
    activePathEndpointIds,
    activePinnedNodes,
    activeTheme,
    activeViewMode,
    committedSceneDataKey,
    engineReady,
    graphVersion?.id,
    isPaneVisible,
    sceneDataKey,
    scene?.graphVersionId,
  ]);

  const setSelectedNode = useCallback(
    (node: GraphNode | null) => {
      // Keep imperative renderer callbacks in lockstep with the user's latest
      // action; waiting for the controlled React prop to round-trip leaves a
      // short window where a camera/transform callback can reapply stale state.
      activeSelectedIdRef.current = node?.id ?? null;
      setInternalSelectedId(node?.id ?? null);
      onNodeSelectRef.current?.(node);
      setStatus(
        node
          ? `已选择 ${node.label}，并高亮其相邻节点`
          : "已清除节点选择",
      );
    },
    [],
  );

  const cancelTransientCamera = useCallback(() => {
    transientFocusRequestRef.current += 1;
    engineRef.current?.cancelCameraForGesture(renderedSceneKeyRef.current);
  }, []);

  const clearTransientUi = useCallback((notifySelection: boolean) => {
    if (searchFeedbackTimerRef.current !== null) {
      window.clearTimeout(searchFeedbackTimerRef.current);
      searchFeedbackTimerRef.current = null;
    }
    if (notifySelection) {
      setSelectedNode(null);
    } else {
      activeSelectedIdRef.current = null;
      setInternalSelectedId(null);
      setStatus("已清除节点选择");
    }
    setSearchOpen(false);
    setSearchFeedback("");
    setPendingSearchNodeId(null);
    clearGraphHighlightRef.current();
  }, [setSelectedNode]);

  const clearTransientFocus = useCallback(() => {
    cancelTransientCamera();
    clearTransientUi(true);
  }, [cancelTransientCamera, clearTransientUi]);

  const returnToOverview = useCallback(() => {
    cancelTransientCamera();
    clearTransientUi(false);
    returnToOverviewActionRef.current?.onReturn();
  }, [cancelTransientCamera, clearTransientUi]);

  const changeViewMode = useCallback(
    (next: GraphViewMode) => {
      setInternalViewMode(next);
      onViewModeChange?.(next);
      if (next === "local" && !activeFocusNodeIds.length && activeSelectedId) {
        const focus = [activeSelectedId];
        setInternalFocusNodeIds(focus);
        onFocusNodeIdsChange?.(focus);
      }
      if (next === "path" && activePathEndpointIds.length < 2) {
        setStatus(activePathEndpointIds.length === 1 ? "请选择路径终点" : "请选择路径起点");
      } else {
        setStatus(
          next === "global"
            ? "已切换为全局图"
            : next === "local"
              ? "已切换为局部图"
              : "已切换为路径图",
        );
      }
    },
    [
      activeFocusNodeIds.length,
      activePathEndpointIds.length,
      activeSelectedId,
      onFocusNodeIdsChange,
      onViewModeChange,
    ],
  );

  const changeTheme = useCallback(
    (next: GraphTheme) => {
      setInternalTheme(next);
      onThemeChange?.(next);
      setStatus(next === "focus-dark" ? "已启用专注深色图谱" : "已启用品牌浅色图谱");
    },
    [onThemeChange],
  );

  const changeFocusNodeIds = useCallback(
    (next: readonly string[]) => {
      setInternalFocusNodeIds(next);
      onFocusNodeIdsChange?.(next);
    },
    [onFocusNodeIdsChange],
  );

  const changePathEndpoints = useCallback(
    (next: readonly string[]) => {
      setInternalPathEndpointIds(next);
      onPathEndpointIdsChange?.(next);
    },
    [onPathEndpointIdsChange],
  );

  const changePinnedNodes = useCallback(
    (
      next: Readonly<
        Record<string, { readonly x: number; readonly y: number }>
      >,
    ) => {
      setInternalPinnedNodes(next);
      onPinnedNodesChange?.(next);
    },
    [onPinnedNodesChange],
  );

  const choosePathEndpoint = useCallback(
    (nodeId: string) => {
      const current = activePathEndpointIds;
      const next =
        current.length === 0
          ? [nodeId]
          : current.length === 1 && current[0] !== nodeId
            ? [current[0], nodeId]
            : [nodeId];
      changePathEndpoints(next);
      if (activeViewMode !== "path") changeViewMode("path");
      setStatus(
        next.length === 1
          ? `已选择路径起点 ${nodeById.get(nodeId)?.label ?? nodeId}，请选择终点`
          : "已选择两个端点并计算无权最短路径",
      );
    },
    [
      activePathEndpointIds,
      activeViewMode,
      changePathEndpoints,
      changeViewMode,
      nodeById,
    ],
  );

  const labelIdsForZoom = useCallback(
    (zoom: number) => {
      const config = visualConfigRef.current;
      return graphLabelIdsForZoom(nodes, degreeById, {
        zoom,
        threshold: config.display.labelThreshold,
        labelLimit,
        selectedNodeId: config.selectedNodeId,
        focusNodeIds: config.focusNodeIds,
        pathNodeIds: config.pathNodeIds,
      });
    },
    [degreeById, labelLimit, nodes],
  );

  const applyVisualStyles = useCallback(
    async (
      graph: Graph,
      zoom = graph.getZoom(),
      isCurrent: () => boolean = () => true,
    ) => {
      if (graph.destroyed || !graph.rendered || !isCurrent()) return;
      const config = visualConfigRef.current;
      const dark = config.theme === "focus-dark";
      const labelIds = labelIdsForZoom(zoom);
      const overlayValues = config.overlay?.nodeValues ?? {};
      const governanceFocusNodeIds = new Set(config.governanceFocus?.nodeIds ?? []);
      const numericValues = Object.values(overlayValues).filter(
        (value): value is number => typeof value === "number" && Number.isFinite(value),
      );
      const numericMax = Math.max(1, ...numericValues.map((value) => Math.abs(value)));

      const nextNodeData = nodes.map((node) => {
          const degree = degreeById.get(node.id) ?? 0;
          const renderedDegree = renderedDegreeById.get(node.id) ?? 0;
          const overlayValue = overlayValues[node.id];
          const riskBand = config.overlay?.presentation?.riskBands?.[node.id];
          const explicitReferenceLabel = config.overlay?.presentation?.referenceLabels?.[node.id];
          const semanticOverlayValue = typeof overlayValue === "string" ? overlayValue : "";
          const referenceLabel = explicitReferenceLabel
            ?? (semanticOverlayValue.includes("reference-positive") || semanticOverlayValue.endsWith("imported-positive")
              ? "positive"
              : semanticOverlayValue.includes("reference-negative") || semanticOverlayValue.endsWith("imported-negative")
                ? "negative"
                : undefined);
          const semanticBadges = graphSemanticBadges(
            referenceLabel,
            config.overlay?.presentation?.reviewDecisions?.[node.id],
          );
          const typeColour = colourByType.get(nodeType(node)) ?? TYPE_COLOURS[0];
          let fill = typeColour;
          let stroke = dark ? "#292a3b" : "#ffffff";
          let halo = false;
          let haloStroke = "#8b72ff";
          let haloStrokeOpacity = 0.22;
          let forceLabel = false;
          let nodeOpacity = 1;
          let nodeLineWidth = renderedDegree === 0 ? 2 : 2.5;
          let size = (
            GRAPH_PREVIEW_POLICY.node.minimumSize
            + Math.sqrt(Math.max(1, degree)) * GRAPH_PREVIEW_POLICY.node.degreeScale
          ) * config.display.nodeScale;

          if (
            config.overlay?.kind === "degree" &&
            typeof overlayValue === "number"
          ) {
            size *= 0.86 + Math.abs(overlayValue) / numericMax;
          } else if (
            config.overlay?.kind === "articulation" &&
            overlayValue === true
          ) {
            stroke = "#ff9d4c";
            halo = true;
            haloStroke = "#ff9d4c";
          } else if (
            (config.overlay?.kind === "components" ||
              config.overlay?.kind === "community") &&
            overlayValue !== undefined
          ) {
            const legendColour = config.overlay.legend.items.find(
              (item) => String(item.value) === String(overlayValue),
            )?.color;
            fill = legendColour ?? TYPE_COLOURS[
              hashText(String(overlayValue)) % TYPE_COLOURS.length
            ];
            stroke = typeColour;
          } else if (
            config.overlay?.kind === "governance" &&
            overlayValue !== undefined
          ) {
            if (typeof overlayValue === "number" && Number.isFinite(overlayValue)) {
              fill = onlineRiskColour(overlayValue);
              stroke = GRAPH_PREVIEW_POLICY.risk.outline;
              halo = overlayValue >= 0.65;
              haloStroke = GRAPH_PREVIEW_POLICY.risk.high;
              haloStrokeOpacity = 0.16;
            } else {
              const value = String(overlayValue);
              const selectedRisk = value.endsWith("-selected") || value === "subject";
              if (value.startsWith("risk-high") || riskBand === "high") fill = GRAPH_PREVIEW_POLICY.risk.high;
              else if (value.startsWith("risk-review") || riskBand === "review") fill = GRAPH_PREVIEW_POLICY.risk.review;
              else if (value.startsWith("risk-low") || riskBand === "low") fill = GRAPH_PREVIEW_POLICY.risk.low;
              else fill = GRAPH_PREVIEW_POLICY.risk.context;
              stroke = value.startsWith("risk-high") || riskBand === "high"
                ? GRAPH_PREVIEW_POLICY.risk.outline
                : dark ? "#293847" : "#ffffff";
              halo = value.startsWith("risk-high") || riskBand === "high" || selectedRisk;
              haloStroke = selectedRisk ? GRAPH_PREVIEW_POLICY.risk.selection : GRAPH_PREVIEW_POLICY.risk.high;
              haloStrokeOpacity = selectedRisk ? 0.28 : 0.16;
              forceLabel = selectedRisk;
            }
          }

          // Risk is an independent semantic channel. Community/relationship
          // fills remain intact while high and review bands receive an outer
          // ring that is still visible outside the selected-node stroke.
          if (riskBand === "high" || riskBand === "review") {
            halo = true;
            haloStroke = riskBand === "high"
              ? GRAPH_PREVIEW_POLICY.risk.high
              : GRAPH_PREVIEW_POLICY.risk.review;
            haloStrokeOpacity = riskBand === "high" ? 0.25 : 0.22;
          }

          // Imported target-domain labels are independent from model risk.
          // They may own the fill/outline in a governance risk view, but never
          // replace community colours. Selection is applied afterwards and
          // therefore remains the highest-priority interaction channel.
          if (config.overlay?.kind === "governance" && referenceLabel === "positive") {
            fill = REFERENCE_POSITIVE_FILL;
            stroke = REFERENCE_POSITIVE_OUTLINE;
            nodeLineWidth = Math.max(nodeLineWidth, 3);
            halo = true;
            haloStroke = REFERENCE_POSITIVE_FILL;
            haloStrokeOpacity = Math.max(haloStrokeOpacity, 0.26);
          } else if (config.overlay?.kind === "governance" && referenceLabel === "negative") {
            stroke = REFERENCE_NEGATIVE_OUTLINE;
            nodeLineWidth = Math.max(nodeLineWidth, 3);
          }

          const governanceChannels = governanceFocusAppearanceChannels(config.governanceFocus, { nodeId: node.id });
          const governanceFocused = governanceChannels.focused;
          if (config.governanceFocus) {
            if (governanceFocused) {
              size *= governanceChannels.sizeMultiplier;
              stroke = GRAPH_PREVIEW_POLICY.risk.selection;
              nodeLineWidth *= governanceChannels.lineWidthMultiplier;
              halo = governanceChannels.dualRing;
              if (riskBand !== "high" && riskBand !== "review") {
                haloStroke = GRAPH_PREVIEW_POLICY.edge.focusColour;
                haloStrokeOpacity = 0.34;
              }
              forceLabel = governanceChannels.persistentLabel;
            } else {
              nodeOpacity = governanceChannels.opacity;
            }
          }
          const rank = showNodeRanks && typeof node.attributes.rank === "number" ? node.attributes.rank : null;
          const rankDelta = showNodeRanks ? config.overlay?.presentation?.rankDeltas?.[node.id] : undefined;
          const adaptedRank = showNodeRanks ? config.overlay?.presentation?.adaptedRanks?.[node.id] : undefined;
          const rankDeltaMarker = typeof adaptedRank !== "number" && typeof rankDelta === "number" && rankDelta !== 0
            ? `${rankDelta < 0 ? "↑" : "↓"}${Math.abs(rankDelta)} `
            : "";
          const displayRank = typeof adaptedRank === "number" ? adaptedRank : rank;

          return {
            id: node.id,
            style: {
              size: Math.min(58, Math.max(GRAPH_PREVIEW_POLICY.node.minimumSize, size)),
              fill,
              opacity: nodeOpacity,
              fillOpacity: renderedDegree === 0 ? 0.12 : degree > 2 ? 0.94 : 0.84,
              stroke,
              lineWidth: nodeLineWidth,
              lineDash: renderedDegree === 0 ? [4, 3] : undefined,
              shadowColor: nodes.length > 300 ? "transparent" : `${fill}4d`,
              shadowBlur: nodes.length > 300 ? 0 : degree > 2 ? 15 : 7,
              halo,
              haloStroke,
              haloLineWidth: 7,
              haloStrokeOpacity,
              badge: semanticBadges.length > 0,
              badges: semanticBadges,
              label: forceLabel || labelIds.has(node.id),
              labelText: `${rankDeltaMarker}${displayRank && (governanceFocused || typeof adaptedRank === "number") ? `#${displayRank} ` : ""}${compactGraphLabel(node.label)}`,
              labelPlacement: "bottom" as const,
              labelOffsetY: 7,
              labelFontSize: degree > 4 ? 13 : 12,
              labelFontWeight: degree > 4 ? 600 : 500,
              labelFill: dark ? "#e6e7f2" : "#293250",
              labelStroke: dark ? "#151620" : "#ffffff",
              labelLineWidth: 3,
              labelBackground: false,
            },
          };
        });
      if (graph.destroyed || !graph.rendered || !isCurrent()) return;
      const currentNodeIds = new Set(graph.getNodeData().map((node) => String(node.id)));
      const currentNodeStyles = nextNodeData.filter((node) => currentNodeIds.has(node.id));
      if (currentNodeStyles.length) graph.updateNodeData(currentNodeStyles);

      const candidatePrefix = "__socialgraph_research_candidate__";
      const candidateColour = config.overlay?.legend.items.find((item) => item.value === "candidate")?.color
        ?? "#2297a5";
      const factNodeIds = new Set(nodes.map((node) => node.id));
      const desiredGhostIds = new Set(graphPresentationGhostNodeIds(nodes, config.overlay));
      const currentGhostIds = graph.getNodeData()
        .filter((node) => node.data?.analysisCandidateEndpoint === true)
        .map((node) => String(node.id));
      const promotedGhostIds = currentGhostIds.filter((id) => factNodeIds.has(id));
      if (promotedGhostIds.length) {
        graph.updateNodeData(promotedGhostIds.map((id) => ({
          id,
          data: { analysisCandidateEndpoint: false, analysisPresentationOnly: false },
        })));
      }
      const staleGhostIds = currentGhostIds.filter((id) => !desiredGhostIds.has(id) && !factNodeIds.has(id));
      if (staleGhostIds.length) graph.removeNodeData(staleGhostIds);
      const currentGhostSet = new Set(currentGhostIds);
      const newGhostIds = [...desiredGhostIds].filter((id) => !currentGhostSet.has(id));
      if (newGhostIds.length) {
        graph.addNodeData(newGhostIds.map((id, index) => ({
          id,
          data: { analysisCandidateEndpoint: true, analysisPresentationOnly: true },
          style: {
            x: 72 + (index % 4) * 88,
            y: 72 + Math.floor(index / 4) * 72,
            size: 24,
            fill: dark ? "#20242e" : "#ffffff",
            stroke: candidateColour,
            lineWidth: 2.5,
            lineDash: [5, 4],
            halo: true,
            haloStroke: candidateColour,
            haloStrokeOpacity: 0.2,
            label: true,
            labelText: compactGraphLabel(id),
            labelPlacement: "bottom" as const,
            labelFill: dark ? "#e6e7f2" : "#293250",
            labelStroke: dark ? "#151620" : "#ffffff",
            labelLineWidth: 3,
          },
        })));
      }

      const currentEdgeIds = new Set(graph.getEdgeData().map((edge) => String(edge.id)));
      const currentEdgeStyles = edges
        .filter((edge) => currentEdgeIds.has(edge.id))
        .map((edge) => {
          const inPath = config.pathEdgeIds.has(edge.id);
          const governanceValue = config.overlay?.kind === "governance"
            ? config.overlay.edgeValues[edge.id]
            : undefined;
          const governanceEvidence = governanceValue !== undefined;
          const factualColour = config.overlay?.presentation?.governanceLens === "relations"
            ? GRAPH_PREVIEW_POLICY.edge.relationshipFactualColour
            : dark ? "#77869a" : GRAPH_PREVIEW_POLICY.edge.factualColour;
          const governanceLens = config.overlay?.presentation?.governanceLens;
          const modalities = Array.isArray(edge.attributes.modalities)
            ? edge.attributes.modalities.filter((value): value is string => typeof value === "string")
            : [];
          const relationKey = governanceExactRelationKey(edge.source, edge.target, modalities);
          const governanceChannels = governanceFocusAppearanceChannels(config.governanceFocus, { relationKey });
          const governanceGroupFocused = config.governanceFocus?.kind === "group"
            && governanceFocusNodeIds.has(edge.source)
            && governanceFocusNodeIds.has(edge.target);
          const governanceGroupInactive = config.governanceFocus?.kind === "group"
            && !governanceGroupFocused;
          const governanceFocused = governanceLens === "relations"
            && governanceChannels.focused
            && isExactGovernanceRelation(
              config.governanceFocus?.exactRelationKey,
              edge.source,
              edge.target,
              modalities,
            );
          const governanceStyle = governanceLens
            ? governanceEdgeStyle(governanceLens, governanceValue, { dark, focused: governanceFocused })
            : null;
          const governanceStroke = governanceLens === "risk"
            ? governanceValue === "evidence-high"
              ? GRAPH_PREVIEW_POLICY.edge.riskHighColour
              : governanceValue === "evidence-review"
                ? GRAPH_PREVIEW_POLICY.edge.riskReviewColour
                : factualColour
            : governanceValue === "factual" ? factualColour : GRAPH_PREVIEW_POLICY.edge.potentialColour;
          return {
            id: edge.id,
            style: {
              stroke: inPath ? "#8b72ff" : governanceStyle?.stroke ?? (governanceEvidence ? governanceStroke : factualColour),
              strokeOpacity: inPath
                ? GRAPH_PREVIEW_POLICY.edge.selectedOpacity
                : governanceGroupFocused
                  ? GRAPH_PREVIEW_POLICY.edge.selectedOpacity
                  : governanceGroupInactive
                    ? 0.06
                : config.governanceFocus && governanceLens === "relations" && !governanceFocused
                  ? governanceChannels.opacity
                  : governanceStyle?.opacity ?? (governanceEvidence
                  ? GRAPH_PREVIEW_POLICY.edge.selectedOpacity
                  : dark ? 0.42 : GRAPH_PREVIEW_POLICY.edge.factualOpacity),
              lineWidth:
                Math.max(1.1, Math.min(3.2, edge.weight ?? 1)) *
                config.display.edgeScale *
                (inPath || governanceGroupFocused ? 1.6 : governanceStyle?.widthMultiplier ?? (governanceEvidence ? 1.6 : 1)),
              endArrow:
                nodes.length <= 1_000 &&
                config.display.arrows &&
                (edge.directed ?? false),
              endArrowFill: inPath
                ? "#8b72ff"
                : governanceStyle?.arrowFill ?? (dark ? "#8587a3" : "#8795dc"),
              endArrowSize: 5,
            },
          };
        });
      if (currentEdgeStyles.length) graph.updateEdgeData(currentEdgeStyles);
      const visibleNodeIds = new Set([...factNodeIds, ...desiredGhostIds]);
      const desiredCandidates = (config.overlay?.candidateEdges ?? []).filter((candidate) => (
        visibleNodeIds.has(candidate.sourceId) && visibleNodeIds.has(candidate.targetId)
      ));
      const desiredCandidateIds = new Set(desiredCandidates.map((candidate) => candidate.id));
      const currentCandidateIds = graph.getEdgeData()
        .map((edge) => String(edge.id ?? ""))
        .filter((id) => id.startsWith(candidatePrefix));
      const staleCandidateIds = currentCandidateIds.filter((id) => !desiredCandidateIds.has(id));
      if (staleCandidateIds.length) graph.removeEdgeData(staleCandidateIds);
      const currentCandidateSet = new Set(currentCandidateIds);
      const candidateStyle = (candidate: (typeof desiredCandidates)[number]) => {
        const channels = governanceFocusAppearanceChannels(config.governanceFocus, {
          relationKey: candidate.exactRelationKey ?? "",
        });
        const focused = config.overlay?.presentation?.governanceLens === "relations" && channels.focused;
        return {
          stroke: focused ? GRAPH_PREVIEW_POLICY.edge.focusColour : GRAPH_PREVIEW_POLICY.edge.potentialColour,
          strokeOpacity: focused ? channels.opacity : config.governanceFocus ? channels.opacity : 0.62,
          lineWidth: focused ? 2.6 : 2.4,
          lineDash: [...GRAPH_PREVIEW_POLICY.edge.potentialDash],
          endArrow: candidate.directed,
          endArrowFill: focused ? GRAPH_PREVIEW_POLICY.edge.focusColour : candidateColour,
          endArrowSize: 6,
        };
      };
      const newCandidates = desiredCandidates.filter((candidate) => !currentCandidateSet.has(candidate.id));
      if (newCandidates.length) {
        graph.addEdgeData(newCandidates.map((candidate) => ({
          id: candidate.id,
          source: candidate.sourceId,
          target: candidate.targetId,
          data: { analysisCandidate: true },
          style: candidateStyle(candidate),
        })));
      }
      if (desiredCandidates.length) {
        graph.updateEdgeData(desiredCandidates.map((candidate) => ({
          id: candidate.id,
          style: candidateStyle(candidate),
        })));
      }
      await graph.draw();
      if (graph.destroyed || !isCurrent()) return;
      setRenderedLabelCount(nextNodeData.filter((node) => node.style.label === true).length);
      const appearanceListener = onAppearanceAppliedRef.current;
      if (appearanceListener) {
        const snapshotStyles = (items: readonly { readonly id?: unknown; readonly style?: unknown }[]) => Object.freeze(Object.fromEntries(
          items.map((item) => [
            String(item.id),
            Object.freeze({ ...((item.style ?? {}) as Record<string, unknown>) }),
          ]),
        ));
        appearanceListener(Object.freeze({
          graphVersionId: graphIdentity,
          ...(config.governanceFocus ? { focusCameraToken: config.governanceFocus.cameraToken } : {}),
          nodeStyles: snapshotStyles(graph.getNodeData()),
          edgeStyles: snapshotStyles(graph.getEdgeData()),
        }));
      }
      publishRuntimeMetrics({
        drawCount: runtimeMetricsRef.current.drawCount + 1,
      });
    },
    [colourByType, degreeById, edges, labelIdsForZoom, nodes, publishRuntimeMetrics, renderedDegreeById, showNodeRanks],
  );

  const applyHighlight = useCallback(
    async (graph: Graph, nodeId: string | null) => {
      if (!nodes.length || graph.destroyed || !graph.rendered) return;

      const nodeStates: GraphElementStates = {};
      const edgeStates: GraphElementStates = {};
      const config = visualConfigRef.current;
      const governanceLens = config.overlay?.presentation?.governanceLens;

      const contextualNodeStates = (id: string) => {
        const states: string[] = [];
        if (config.pathNodeIds.has(id)) states.push("path");
        if (activePathEndpointIds.includes(id)) states.push("endpoint");
        if (config.pinnedNodes[id]) states.push("pinned");
        return states;
      };

      const contextualEdgeStates = (id: string) =>
        config.pathEdgeIds.has(id) ? ["path"] : [];

      // Product-sized governance graphs receive a complete inactive-state
      // repaint so selection always reveals one-hop context. Large benchmark
      // scenes retain the bounded incremental path.
      const useIncrementalHighlight = nodes.length > 300;
      if (useIncrementalHighlight) {
        const neighbours = new Set<string>();
        const relatedEdgeIds = new Set<string>();
        if (nodeId && nodeById.has(nodeId)) {
          for (const id of adjacencyIndex.neighbours(nodeId)) neighbours.add(id);
          for (const id of adjacencyIndex.edgeIds(nodeId)) relatedEdgeIds.add(id);
        }
        const contextualNodes = new Set([
          ...config.pathNodeIds,
          ...activePathEndpointIds,
          ...Object.keys(config.pinnedNodes),
        ]);
        const nextNodeIds = new Set([
          ...contextualNodes,
          ...(nodeId ? [nodeId] : []),
          ...neighbours,
        ]);
        const nextEdgeIds = new Set([...config.pathEdgeIds, ...relatedEdgeIds]);
        const changedNodeIds = new Set([
          ...highlightedElementsRef.current.nodes,
          ...nextNodeIds,
        ]);
        const changedEdgeIds = new Set([
          ...highlightedElementsRef.current.edges,
          ...nextEdgeIds,
        ]);
        for (const id of changedNodeIds) {
          if (!nodeById.has(id)) continue;
          const selectionStates =
            id === nodeId
              ? governanceSelectionStates(governanceLens, false).node
              : neighbours.has(id)
                ? ["neighbour"]
                : [];
          nodeStates[id] = [...selectionStates, ...contextualNodeStates(id)];
        }
        for (const id of changedEdgeIds) {
          edgeStates[id] = [
            ...(relatedEdgeIds.has(id) ? governanceSelectionStates(governanceLens, true).edge : []),
            ...contextualEdgeStates(id),
          ];
        }
        highlightedElementsRef.current = {
          nodes: nextNodeIds,
          edges: nextEdgeIds,
        };
      } else if (!nodeId || !nodeById.has(nodeId)) {
        for (const node of nodes) nodeStates[node.id] = contextualNodeStates(node.id);
        for (const edge of edges) edgeStates[edge.id] = contextualEdgeStates(edge.id);
      } else {
        const neighbours = new Set(adjacencyIndex.neighbours(nodeId));
        const relatedEdgeIds = new Set(adjacencyIndex.edgeIds(nodeId));

        for (const node of nodes) {
          const selectionStates =
            node.id === nodeId
              ? governanceSelectionStates(governanceLens, false).node
              : neighbours.has(node.id)
                ? ["neighbour"]
                : ["inactive"];
          nodeStates[node.id] = [
            ...selectionStates,
            ...contextualNodeStates(node.id),
          ];
        }
        for (const edge of edges) {
          edgeStates[edge.id] = [
            ...(relatedEdgeIds.has(edge.id) ? governanceSelectionStates(governanceLens, true).edge : ["inactive"]),
            ...contextualEdgeStates(edge.id),
          ];
        }
      }

      try {
        await graph.setElementState({ ...nodeStates, ...edgeStates }, false);
      } catch (error) {
        // React StrictMode may dispose the first graph while an async paint is
        // still resolving. A disposed graph is expected and needs no warning.
        if (!graph.destroyed) throw error;
      }
    },
    [activePathEndpointIds, adjacencyIndex, edges, nodeById, nodes],
  );

  const applyCommittedHighlight = useCallback(
    async (graph: Graph, nodeId: string | null, expectedSceneIdentity: string) => {
      const engine = engineRef.current;
      if (
        !engine
        || graph.destroyed
        || !graph.rendered
        || graphRef.current !== graph
        || !expectedSceneIdentity
        || committedSceneDataKeyRef.current !== expectedSceneIdentity
      ) return false;
      const lease = engine.captureSceneLease();
      if (!lease || lease.sceneIdentity !== expectedSceneIdentity) return false;
      return engine.runSceneTransaction(lease, async (isCurrent) => {
        if (!isCurrent()) return;
        await applyHighlight(graph, nodeId);
      });
    },
    [applyHighlight],
  );

  clearGraphHighlightRef.current = () => {
    const graph = graphRef.current;
    if (!graph || graph.destroyed || !graph.rendered) return;
    const expectedSceneIdentity = committedSceneDataKeyRef.current;
    void applyCommittedHighlight(graph, null, expectedSceneIdentity).then((applied) => {
      if (!applied || !rootRef.current || graph.destroyed) return;
      // The incremental state sets are the source of truth. A full residual
      // renderer scan is available through benchmark diagnostics, not the
      // ordinary Escape path.
      rootRef.current.dataset.transientHighlightCount = "0";
    });
  };

  // Layout coordinates belong to facts, not to the current overlay/filter.
  // Keep this key order-independent so a reordered API response cannot reset
  // the force-friendly positions.
  const webglSupported = useMemo(() => detectWebGLSupport(), []);
  const activeRendererFailure =
    rendererFailure?.graphIdentity === graphIdentity &&
    rendererFailure.preference === activeRendererPreference
      ? rendererFailure.reason
      : undefined;
  const desiredRendererKind = activeRendererFailure
    ? "canvas"
    : resolveGraphRendererKind(
        activeRendererPreference,
        nodes.length,
        edges.length,
        webglSupported,
      );
  const rendererRequestKey = `${activeRendererPreference}:${desiredRendererKind}:${activeRendererFailure ?? "ready"}`;
  const rendererLoadRequestKey = `${graphIdentity}\u0000${rendererRequestKey}`;
  const loadedRenderer =
    loadedRendererRequest?.requestKey === rendererLoadRequestKey
      ? loadedRendererRequest.value
      : null;

  useEffect(() => {
    let cancelled = false;
    const requestKey = rendererLoadRequestKey;
    const failureIdentity = graphIdentity;
    const failurePreference = activeRendererPreference;
    void loadGraphRenderer({
      preference: activeRendererPreference,
      nodeCount: nodes.length,
      edgeCount: edges.length,
      webglSupported,
      forcedCanvasReason: activeRendererFailure,
      contextLossCount: webglContextLossCount,
      onRuntimeFailure: (reason) => {
        if (cancelled) return;
        setWebglContextLossCount((count) => count + 1);
        setRendererFailure({
          graphIdentity: failureIdentity,
          preference: failurePreference,
          reason,
        });
        setStatus("WebGL 图层已中断，正在安全回退到 Canvas");
      },
    }).then((loaded) => {
      if (cancelled) return;
      setLoadedRendererRequest({ requestKey, value: loaded });
      setRendererStatus(loaded.status);
      publishRuntimeMetrics({
        rendererRequested: loaded.status.requested,
        rendererResolved: loaded.status.resolved,
        rendererFallbackReason: loaded.status.fallbackReason,
        webglSupported: loaded.status.webglSupported,
        webglLazyLoadMs: loaded.status.lazyLoadMs,
        webglContextLossCount: loaded.status.contextLossCount,
      });
      if (loaded.status.fallbackReason) {
        setStatus("WebGL 实验不可用，已回退到 Canvas 兼容模式");
      }
    });
    return () => {
      cancelled = true;
    };
    // Counts only matter when they cross the renderer threshold represented
    // in rendererRequestKey. Equivalent scene updates must not reload WebGL.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendererLoadRequestKey, webglSupported]);

  const initialPositions = useMemo(
    () => deterministicGraphInitialPositions(nodes, edges),
    [edges, nodes],
  );

  const graphData = useMemo<GraphData>(() => {
    const initialLabels = labelIdsForZoom(1);
    const initialConfig = visualConfigRef.current;
    return {
      nodes: nodes.map((node) => {
        const degree = degreeById.get(node.id) ?? 0;
        const renderedDegree = renderedDegreeById.get(node.id) ?? 0;
        const colour = colourByType.get(nodeType(node)) ?? TYPE_COLOURS[0];
        const cached =
          activePinnedNodes[node.id] ??
          positionCacheRef.current.get(topologyKey)?.get(node.id) ??
          initialPositions.get(node.id);
        return {
          id: node.id,
          type: "circle",
          data: {
            label: node.label,
            entityType: nodeType(node),
            degree,
          },
          style: {
            ...(cached ? { x: cached.x, y: cached.y } : {}),
            size: Math.min(
              58,
              (GRAPH_PREVIEW_POLICY.node.minimumSize
                + Math.sqrt(Math.max(1, degree)) * GRAPH_PREVIEW_POLICY.node.degreeScale) *
                initialConfig.display.nodeScale,
            ),
            fill: colour,
            fillOpacity: renderedDegree === 0 ? 0.12 : degree > 2 ? 0.94 : 0.84,
            stroke: initialConfig.theme === "focus-dark" ? "#292a3b" : "#ffffff",
            lineWidth: renderedDegree === 0 ? 2 : 2.5,
            lineDash: renderedDegree === 0 ? [4, 3] : undefined,
            shadowColor: nodes.length > 300 ? "transparent" : `${colour}4d`,
            shadowBlur: nodes.length > 300 ? 0 : degree > 2 ? 15 : 7,
            label: initialLabels.has(node.id),
            labelText: compactGraphLabel(node.label),
            labelPlacement: "bottom" as const,
            labelOffsetY: 7,
            labelFontSize: degree > 4 ? 13 : 12,
            labelFontWeight: degree > 4 ? 600 : 500,
            labelFill:
              initialConfig.theme === "focus-dark" ? "#e6e7f2" : "#293250",
            labelStroke:
              initialConfig.theme === "focus-dark" ? "#151620" : "#ffffff",
            labelLineWidth: 3,
            labelBackground: false,
          },
        };
      }),
      edges: edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "line",
        data: {
          relationType: edge.type,
          weight: edge.weight,
        },
        style: {
          stroke: edge.type
            ? TYPE_COLOURS[hashText(edge.type) % TYPE_COLOURS.length]
            : initialConfig.theme === "focus-dark" ? "#72758c" : "#9ca8ca",
          strokeOpacity:
            initialConfig.theme === "focus-dark" ? 0.42 : nodes.length <= 50 ? 0.52 : 0.44,
          lineWidth:
            Math.max(1.1, Math.min(3.2, edge.weight ?? 1)) *
            initialConfig.display.edgeScale,
          endArrow:
            nodes.length <= 1_000 &&
            initialConfig.display.arrows &&
            (edge.directed ?? false),
          endArrowFill:
            initialConfig.theme === "focus-dark" ? "#8587a3" : "#8795dc",
          endArrowSize: 5,
        },
      })),
    };
  }, [
    activePinnedNodes,
    colourByType,
    degreeById,
    edges,
    initialPositions,
    labelIdsForZoom,
    nodes,
    renderedDegreeById,
    topologyKey,
  ]);
  const graphDataRef = useRef(graphData);
  graphDataRef.current = graphData;
  const renderedSceneKeyRef = useRef("");
  const renderedTopologyKeyRef = useRef("");
  const renderedNodeIdsRef = useRef<readonly string[]>([]);

  const interactionRuntimeRef = useRef({
    activeDepth,
    activeLayoutPreset,
    activePathEndpointIds,
    activeViewMode,
    applyHighlight,
    applyVisualStyles,
    changeFocusNodeIds,
    changePinnedNodes,
    changeViewMode,
    clearTransientFocus,
    choosePathEndpoint,
    nodeById,
    nodes,
    edges,
    previewTruncated: preview?.truncated,
    sceneTruncated: scene?.truncated,
    setSelectedNode,
  });
  interactionRuntimeRef.current = {
    activeDepth,
    activeLayoutPreset,
    activePathEndpointIds,
    activeViewMode,
    applyHighlight,
    applyVisualStyles,
    changeFocusNodeIds,
    changePinnedNodes,
    changeViewMode,
    clearTransientFocus,
    choosePathEndpoint,
    nodeById,
    nodes,
    edges,
    previewTruncated: preview?.truncated,
    sceneTruncated: scene?.truncated,
    setSelectedNode,
  };

  useEffect(() => {
    const container = canvasRef.current;
    const viewport = viewportRef.current;
    const minimapContainer = minimapRef.current;
    if (
      !container ||
      !viewport ||
      (enableMinimap && !minimapContainer) ||
      !loadedRenderer ||
      (!graphVersion && !scene) ||
      nodes.length === 0
    ) {
      return;
    }

    let cancelled = false;
    let dragTargetRefreshTimer: number | null = null;
    let initialFitComplete = false;
    const cacheTopologyKey = topologyKey;
    const cachedPositions = positionCacheRef.current.get(cacheTopologyKey) ?? new Map<string, { readonly x: number; readonly y: number }>();
    positionCacheRef.current.set(cacheTopologyKey, cachedPositions);
    const hasCachedLayout = nodes.length > 0 && nodes.every((node) => cachedPositions.has(node.id));
    const externalCameraToRestore = cameraRestoreCommand
      && (
        !("sceneIdentity" in cameraRestoreCommand)
        || cameraRestoreCommand.sceneIdentity === sceneDataKey
      )
      ? cameraRestoreCommand
      : null;
    const shouldRestoreRetainedCamera =
      lastCreatedGraphIdentityRef.current === `${graphIdentity}\u0000${cacheTopologyKey}`
      && hasStableCameraRef.current;
    const shouldRestoreCamera = Boolean(externalCameraToRestore || shouldRestoreRetainedCamera);
    const cameraToRestore = externalCameraToRestore
      ? "position" in externalCameraToRestore
        ? {
            x: Number(externalCameraToRestore.position[0]),
            y: Number(externalCameraToRestore.position[1]),
            zoom: externalCameraToRestore.zoom,
          }
        : {
            x: externalCameraToRestore.x,
            y: externalCameraToRestore.y,
            zoom: externalCameraToRestore.zoom,
          }
      : { ...lastCameraRef.current };
    const worldCenterToRestore = externalCameraToRestore
      ? "worldCenter" in externalCameraToRestore
        ? { x: Number(externalCameraToRestore.worldCenter[0]), y: Number(externalCameraToRestore.worldCenter[1]) }
        : null
      : lastWorldCenterRef.current
        ? { ...lastWorldCenterRef.current }
        : null;
    lastCreatedGraphIdentityRef.current = `${graphIdentity}\u0000${cacheTopologyKey}`;
    const performanceProfile = graphCanvasPerformanceProfile(nodes.length);
    const interactionLodConfig = graphCanvasInteractionLodConfig(nodes.length);
    const initialLayout = forceLayoutOptions(
      forceSettings,
      nodes.length,
      displaySettings.nodeScale,
      `${graphIdentity}:${cacheTopologyKey}`,
    );
    lastForceKeyRef.current = JSON.stringify(forceSettings);
    const positionedData: GraphData = {
      ...graphDataRef.current,
      nodes: (graphDataRef.current.nodes ?? []).map((node) => {
        const cached = cachedPositions.get(String(node.id));
        return cached
          ? { ...node, style: { ...node.style, x: cached.x, y: cached.y } }
          : node;
      }),
    };
    const performanceProbe = new GraphPerformanceProbe(
      graphIdentity,
      loadedRenderer.status.resolved,
    );
    performanceProbeRef.current?.dispose();
    performanceProbeRef.current = performanceProbe;
    performanceProbe.begin("renderer_construct");
    const graph = new Graph({
      container,
      data: positionedData,
      renderer: loadedRenderer.renderer,
      padding: 42,
      zoomRange: [0.01, 4],
      // Animating thousands of element entrances makes Canvas spend seconds
      // creating per-shape tweens before it can report Ready.
      animation: nodes.length > 180 ? false : graphAnimation(),
      // A saved topology can be restored immediately. A new topology must
      // always receive one deterministic force pass, including medium/large
      // graphs; otherwise the seed is never relaxed and disconnected nodes
      // form conspicuous rails/rings.
      ...(hasCachedLayout
        ? {}
        : { layout: initialLayout }),
      node: {
        type: "circle",
        state: {
          selected: {
            label: true,
            halo: true,
            haloStroke: "#6d4cf5",
            haloLineWidth: 6,
            haloStrokeOpacity: 0.22,
            lineWidth: 3,
            stroke: "#5135d8",
            zIndex: 10,
          },
          "governance-selected": {
            label: true,
            halo: true,
            haloStroke: GRAPH_PREVIEW_POLICY.risk.selection,
            haloLineWidth: 7,
            haloStrokeOpacity: 0.28,
            zIndex: 10,
          },
          neighbour: {
            label: nodes.length <= 180,
            lineWidth: 2.5,
            stroke: "#5c74ec",
            opacity: 1,
            zIndex: 6,
          },
          hover: {
            label: true,
            lineWidth: 2.5,
            stroke: "#6d4cf5",
            zIndex: 8,
          },
          inactive: {
            opacity: 0.22,
          },
          path: {
            label: true,
            opacity: 1,
            lineWidth: 3,
            stroke: "#7b63ff",
            zIndex: 12,
          },
          endpoint: {
            label: true,
            halo: true,
            haloStroke: "#20b9c1",
            haloLineWidth: 8,
            haloStrokeOpacity: 0.26,
            zIndex: 14,
          },
          pinned: {
            lineWidth: 3,
            stroke: "#f2a154",
            zIndex: 13,
          },
        },
      },
      edge: {
        type: "line",
        state: {
          related: {
            stroke: "#6279ee",
            strokeOpacity: 0.92,
            lineWidth: 2.4,
            zIndex: 5,
          },
          "governance-focus": {
            stroke: GRAPH_PREVIEW_POLICY.edge.focusColour,
            endArrowFill: GRAPH_PREVIEW_POLICY.edge.focusColour,
            strokeOpacity: 0.94,
            lineWidth: 2.6,
            zIndex: 7,
          },
          inactive: {
            strokeOpacity: 0.12,
          },
          path: {
            stroke: "#7b63ff",
            strokeOpacity: 0.96,
            lineWidth: 3,
            zIndex: 9,
          },
        },
      },
      behaviors: [
        "drag-canvas",
        "zoom-canvas",
        {
          type: "optimize-viewport-transform",
          key: "workbench-viewport-optimizer",
          enable: interactionLodConfig.enabled,
          debounce: interactionLodConfig.debounce,
          shapes: interactionLodConfig.shapes,
        },
        ...(performanceProfile.hover
          ? [
              {
                type: "hover-activate",
                degree: 1,
                state: "hover",
                inactiveState: "",
              },
            ]
          : []),
      ],
      plugins:
        enableMinimap && performanceProfile.minimap && minimapContainer
          ? [
              {
                type: "minimap",
                key: "preview-minimap",
                container: minimapContainer,
                size: [112, 76],
                padding: 7,
                shape: "key",
                // Node-only minimaps avoid cloning thousands of edges after
                // every force tick; the long debounce coalesces drag updates.
                filter: (_id: string, elementType: string) => elementType === "node",
                delay: performanceProfile.minimapDelay,
              },
            ]
          : [],
    });
    performanceProbe.end("renderer_construct");
    const detachPerformanceProbe = performanceProbe.attach(graph);

    // Medium/large Canvas dragging uses an inexpensive DOM proxy. Repainting
    // thousands of Canvas edges for every pointer frame dominated the gesture
    // budget even though coordinate conversion and spatial picking were fast.
    // The real graph receives only the final coordinate on pointer release.
    const interactionVeil = document.createElement("div");
    const interactionProxy = document.createElement("div");
    const interactionProxyMarker = document.createElement("span");
    const interactionProxyLabel = document.createElement("span");
    interactionVeil.className = "graph-preview__interaction-veil";
    interactionProxy.className = "graph-preview__interaction-proxy";
    interactionProxyMarker.className = "graph-preview__interaction-proxy-marker";
    interactionProxyLabel.className = "graph-preview__interaction-proxy-label";
    if (
      interactionLodConfig.enabled &&
      loadedRenderer.status.resolved === "canvas"
    ) {
      interactionProxy.append(interactionProxyMarker, interactionProxyLabel);
      container.append(interactionVeil, interactionProxy);
    }

    graphRef.current?.destroy();
    diagnosticNodeIdRef.current = null;
    const diagnosticRoot = rootRef.current;
    if (diagnosticRoot) {
      delete diagnosticRoot.dataset.coordinateNodeId;
      delete diagnosticRoot.dataset.coordinateNodeX;
      delete diagnosticRoot.dataset.coordinateNodeY;
    }
    graphRef.current = graph;
    const engine = new GraphEngineAdapter(graph, () => [
      viewport.clientWidth,
      viewport.clientHeight,
    ], sceneDataKey);
    engineRef.current = engine;
    const isCurrentGraph = () => !cancelled
      && !graph.destroyed
      && graphRef.current === graph
      && engineRef.current === engine;
    const hasLiveViewport = () => isPaneVisibleRef.current
      && viewport.clientWidth > 0
      && viewport.clientHeight > 0;
    visibilityControllerRef.current = new GraphVisibilityController(nodes, edges);
    spatialIndexRef.current.clear();
    viewportNodeIdsRef.current = undefined;
    cameraControllerRef.current = new GraphCameraController(graph, {
      padding: 40,
      minZoom: 0.01,
      maxZoom: 4,
      getViewportSize: () => [viewport.clientWidth, viewport.clientHeight],
    });
    localForceControllerRef.current?.destroy();
    localForceControllerRef.current = new LocalForceController({
        onFrame: (frame) => {
          if (frame.epoch !== graphIdentity || graph.destroyed) return;
          pendingForceFrameRef.current = frame;
          if (forceFrameRef.current !== null || forceApplyInFlightRef.current) return;
          const scheduleApply = () => {
            if (
              forceFrameRef.current !== null ||
              forceApplyInFlightRef.current ||
              !pendingForceFrameRef.current
            ) return;
            forceFrameRef.current = window.requestAnimationFrame(() => {
              forceFrameRef.current = null;
              const latest = pendingForceFrameRef.current;
              pendingForceFrameRef.current = null;
              const controller = localForceControllerRef.current;
              const engine = engineRef.current;
              if (!latest || !controller || !engine || graph.destroyed) return;
              forceApplyInFlightRef.current = true;
              const positions: Record<string, { x: number; y: number }> = {};
              const indexed: Array<{ id: string; x: number; y: number }> = [];
              for (let index = 0; index < latest.nodeIndices.length; index += 1) {
                const nodeIndex = latest.nodeIndices[index];
                // The direct pointer lane already owns the dragged node. The
                // Worker only needs to relax its bounded neighbour set, so do
                // not enqueue a duplicate target mutation on Canvas.
                if (nodeIndex === latest.targetNodeIndex) continue;
                const id = controller.nodeId(nodeIndex);
                if (!id) continue;
                const point = {
                  x: latest.positions[index * 2],
                  y: latest.positions[index * 2 + 1],
                };
                const previous = cachedPositions.get(id);
                const delta = previous ? Math.hypot(point.x - previous.x, point.y - previous.y) : 0;
                if (previous && delta < 0.5) {
                  continue;
                }
                if (delta >= 0.5) {
                  localForceMovedNodeIdsRef.current.add(id);
                  localForceNeighborDeltaMaxRef.current = Math.max(
                    localForceNeighborDeltaMaxRef.current,
                    delta,
                  );
                }
                positions[id] = point;
                indexed.push({ id, ...point });
                cachedPositions.set(id, point);
              }
              spatialIndexRef.current.update(indexed);
              publishRuntimeMetrics({
                localForceFrameCount: runtimeMetricsRef.current.localForceFrameCount + 1,
                localForceMovedNodeCount: localForceMovedNodeIdsRef.current.size,
                localForceNeighborDeltaMax: localForceNeighborDeltaMaxRef.current,
              });
              const applyStartedAt = performance.now();
              void engine.applyPositions(positions).then(() => {
                const positionApplyMs = performance.now() - applyStartedAt;
                performanceProbe.record("worker_compute", latest.computeMs, {
                  count: latest.activeCount,
                });
                performanceProbe.record("worker_round_trip", latest.roundTripMs, {
                  count: latest.activeCount,
                });
                performanceProbe.record("position_apply", positionApplyMs, {
                  count: Object.keys(positions).length,
                });
                performanceProbe.record("drag_neighbour_apply", positionApplyMs, {
                  count: Object.keys(positions).length,
                  detail: { mergedFrame: false },
                });
                const now = performance.now();
                if (now - lastPerformancePublishAtRef.current >= 250) {
                  lastPerformancePublishAtRef.current = now;
                  publishRuntimeMetrics({
                    workerRoundTripMs: latest.roundTripMs,
                    workerComputeMs: latest.computeMs,
                    positionApplyMs,
                    performanceSamples: performanceProbe.snapshot(),
                  });
                }
              }).catch((error: unknown) => {
                if (!graph.destroyed) {
                  setRenderError(
                    `局部力布局更新失败：${error instanceof Error ? error.message : "未知错误"}`,
                  );
                }
              }).finally(() => {
                forceApplyInFlightRef.current = false;
                if (!graph.destroyed && pendingForceFrameRef.current) scheduleApply();
              });
            });
          };
          scheduleApply();
        },
    });
    if (onRuntimeMetricsRef.current) {
      (window as typeof window & { __SGFM_GRAPH_BENCHMARK_GRAPH__?: Graph })
        .__SGFM_GRAPH_BENCHMARK_GRAPH__ = graph;
    }
    publishRuntimeMetrics({
      ready: false,
      engineCreateCount: runtimeMetricsRef.current.engineCreateCount + 1,
      layoutCount:
        runtimeMetricsRef.current.layoutCount +
        (hasCachedLayout ? 0 : 1),
    });

    let activeInteractionLod = false;
    const mutationMetrics = () => {
      const diagnostics = engineRef.current?.mutationDiagnostics();
      return diagnostics
        ? {
            mutationInFlight: diagnostics.mutationInFlight,
            mutationInFlightMax: diagnostics.mutationInFlightMax,
          }
        : {};
    };
    const positionInteractionProxy = (clientX: number, clientY: number) => {
      const bounds = container.getBoundingClientRect();
      interactionProxy.style.transform =
        `translate3d(${clientX - bounds.left}px, ${clientY - bounds.top}px, 0) translate(-50%, -50%)`;
    };
    const beginInteractionLod = async (
      draggedNodeId: string,
      clientX: number,
      clientY: number,
    ) => {
      if (
        activeInteractionLod ||
        graph.destroyed ||
        !interactionLodConfig.enabled ||
        loadedRenderer.status.resolved !== "canvas"
      ) {
        return;
      }
      activeInteractionLod = true;
      const nodeStyle = graph.getElementRenderStyle(draggedNodeId);
      const rawSize = nodeStyle.size;
      const size = Array.isArray(rawSize)
        ? Math.max(...rawSize.map(Number).filter(Number.isFinite))
        : Number(rawSize);
      const proxySize = Math.max(18, Math.min(58, Number.isFinite(size) ? size : 28));
      interactionProxyMarker.style.width = `${proxySize}px`;
      interactionProxyMarker.style.height = `${proxySize}px`;
      interactionProxyMarker.style.background = String(nodeStyle.fill ?? "#7867d9");
      interactionProxyLabel.textContent =
        compactGraphLabel(nodeById.get(draggedNodeId)?.label ?? draggedNodeId);
      positionInteractionProxy(clientX, clientY);
      interactionVeil.style.display = "block";
      interactionProxy.style.display = "block";
      publishRuntimeMetrics({ interactionLodActive: true, ...mutationMetrics() });
    };
    const endInteractionLod = async () => {
      if (!activeInteractionLod) return;
      activeInteractionLod = false;
      interactionVeil.style.display = "none";
      interactionProxy.style.display = "none";
      if (!graph.destroyed) {
        publishRuntimeMetrics({ interactionLodActive: false, ...mutationMetrics() });
      }
    };

    const handleNodeClick = (event: IElementEvent) => {
      const id = String(event.target.id);
      const runtime = interactionRuntimeRef.current;
      const node = runtime.nodeById.get(id);
      if (!node) return;
      cancelTransientCamera();
      if (includesShiftKey(event)) {
        runtime.choosePathEndpoint(id);
        runtime.setSelectedNode(node);
        return;
      }
      runtime.setSelectedNode(node);
    };

    const handleNodeDoubleClick = (event: IElementEvent) => {
      const id = String(event.target.id);
      const runtime = interactionRuntimeRef.current;
      const node = runtime.nodeById.get(id);
      if (!node) return;
      runtime.setSelectedNode(node);
      runtime.changeFocusNodeIds([id]);
      runtime.changeViewMode("local");
      setStatus(`已进入 ${node.label} 的 ${runtime.activeDepth} 跳局部图`);
    };

    const finishNodeDrag = (id: string, shiftKey: boolean) => {
      const runtime = interactionRuntimeRef.current;
      if (!runtime.nodeById.has(id) || graph.destroyed) return;
      const position = graph.getElementPosition(id);
      const point = { x: Number(position[0]), y: Number(position[1]) };
      cachedPositions.set(id, point);
      spatialIndexRef.current.update([{ id, ...point }]);
      if (shiftKey) {
        const next = { ...visualConfigRef.current.pinnedNodes, [id]: point };
        runtime.changePinnedNodes(next);
        setForceFixedPosition(graph, id, [point.x, point.y]);
        setStatus(`已固定 ${runtime.nodeById.get(id)?.label ?? id}；可在节点卡片中取消固定`);
      } else {
        const currentPins = visualConfigRef.current.pinnedNodes;
        if (currentPins[id]) {
          const next = { ...currentPins };
          delete next[id];
          runtime.changePinnedNodes(next);
        }
        setForceFixedPosition(graph, id, null);
        setStatus(
          performanceProfile.directDrag
            ? `已移动 ${runtime.nodeById.get(id)?.label ?? id}`
            : `已移动 ${runtime.nodeById.get(id)?.label ?? id}，节点将继续参与力布局`,
        );
      }
      publishRuntimeMetrics({
        dragTarget: readDragTarget(graph, id, container),
        lastDraggedNodeId: id,
        lastDragPinned: shiftKey,
        ...mutationMetrics(),
      });
      interactionReleaseTimerRef.current = window.setTimeout(() => {
        rootRef.current?.classList.remove("is-direct-manipulation");
        interactionReleaseTimerRef.current = null;
      }, 180);
    };

    let directDrag:
      | {
           id: string;
           pointerId: number;
           shiftKey: boolean;
           moved: boolean;
           forceStarted: boolean;
           startClientPosition: [number, number];
           pointerOffset: [number, number];
           lodStarted: boolean;
           pendingPosition: [number, number] | null;
           pendingClientPosition: [number, number] | null;
        }
      | undefined;
    let blankCanvasGesture:
      | {
          pointerId: number;
          startClientPosition: [number, number];
          moved: boolean;
        }
      | undefined;
    let directDragFrame: number | null = null;

    const nearestDirectDragNode = (clientX: number, clientY: number) => {
      const dispatchStartedAt = performance.now();
      const transformStartedAt = performance.now();
      const canvasPoint = graph.getCanvasByClient([clientX, clientY]);
      performanceProbe.record("coordinate_transform", performance.now() - transformStartedAt);
      const tolerance = 28 / Math.max(0.12, graph.getZoom());
      const pick = spatialIndexRef.current.nearest(
        Number(canvasPoint[0]),
        Number(canvasPoint[1]),
        tolerance,
      );
      performanceProbe.record("spatial_pick", pick.durationMs, {
        count: pick.candidateCount,
      });
      performanceProbe.record("pointer_dispatch", performance.now() - dispatchStartedAt, {
        count: pick.candidateCount,
      });
      const now = performance.now();
      if (now - lastPerformancePublishAtRef.current >= 250) {
        lastPerformancePublishAtRef.current = now;
        publishRuntimeMetrics({
          spatialPickMs: pick.durationMs,
          spatialPickCandidates: pick.candidateCount,
          performanceSamples: performanceProbe.snapshot(),
        });
      }
      return pick.id;
    };

    const flushDirectDrag = async () => {
      directDragFrame = null;
      const active = directDrag;
      if (!active?.pendingPosition || graph.destroyed) return;
      const position = active.pendingPosition;
      if (active.forceStarted) {
        localForceControllerRef.current?.dragMove(position[0], position[1]);
      }
      if (interactionLodConfig.enabled) {
        const clientPosition = active.pendingClientPosition;
        if (clientPosition) positionInteractionProxy(clientPosition[0], clientPosition[1]);
        return;
      }
      active.pendingPosition = null;
      active.pendingClientPosition = null;
      const applyStartedAt = performance.now();
      await engineRef.current?.applyPositions({
        [active.id]: { x: position[0], y: position[1] },
      });
      performanceProbe.record("drag_target_apply", performance.now() - applyStartedAt, {
        count: 1,
      });
      cachedPositions.set(active.id, { x: position[0], y: position[1] });
    };

    const pauseViewportCulling = () => {
      if (viewportCullingPausedRef.current) return;
      viewportCullingPausedRef.current = true;
      publishRuntimeMetrics({ viewportCullingPaused: true });
    };

    const handleDirectPointerDown = (event: PointerEvent) => {
      if (event.button !== 0 || graph.destroyed) return;
      viewportGestureActiveRef.current = true;
      pauseViewportCulling();
      cancelTransientCamera();
      const id = nearestDirectDragNode(event.clientX, event.clientY);
      if (!id) {
        // Keep the press available to G6 drag-canvas. Selection is cleared only
        // after a click-sized pointerup, never while a pan is starting.
        blankCanvasGesture = {
          pointerId: event.pointerId,
          startClientPosition: [event.clientX, event.clientY],
          moved: false,
        };
        return;
      }
      blankCanvasGesture = undefined;
      event.preventDefault();
      event.stopPropagation();
      const startPosition = graph.getCanvasByClient([event.clientX, event.clientY]);
      const nodePosition = graph.getElementPosition(id);
      container.setPointerCapture?.(event.pointerId);
      rootRef.current?.classList.add("is-direct-manipulation");
      directDrag = {
        id,
        pointerId: event.pointerId,
        shiftKey: event.shiftKey,
        moved: false,
        forceStarted: false,
        startClientPosition: [event.clientX, event.clientY],
        pointerOffset: [
          Number(startPosition[0]) - Number(nodePosition[0]),
          Number(startPosition[1]) - Number(nodePosition[1]),
        ],
        lodStarted: false,
        pendingPosition: null,
        pendingClientPosition: null,
      };
    };

    const handleDirectPointerMove = (event: PointerEvent) => {
      if (!directDrag || directDrag.pointerId !== event.pointerId) return;
      if (!directDrag.moved && !shouldBeginGraphDrag(
        directDrag.startClientPosition,
        [event.clientX, event.clientY],
      )) return;
      event.preventDefault();
      event.stopPropagation();
      if (!directDrag.moved) {
        directDrag.moved = true;
      }
      if (!directDrag.lodStarted) {
        directDrag.lodStarted = true;
        void beginInteractionLod(directDrag.id, event.clientX, event.clientY).catch((error: unknown) => {
          if (!graph.destroyed) {
            setRenderError(
              `交互降级失败：${error instanceof Error ? error.message : "未知错误"}`,
            );
          }
        });
      }
      directDrag.shiftKey ||= event.shiftKey;
      const transformStartedAt = performance.now();
      const pointerPosition = graph.getCanvasByClient([
        event.clientX,
        event.clientY,
      ]) as [number, number];
      const nextPosition: [number, number] = [
        pointerPosition[0] - directDrag.pointerOffset[0],
        pointerPosition[1] - directDrag.pointerOffset[1],
      ];
      if (!directDrag.forceStarted) {
        if (localForceSettleTimerRef.current !== null) {
          window.clearTimeout(localForceSettleTimerRef.current);
          localForceSettleTimerRef.current = null;
        }
        localForceMovedNodeIdsRef.current.clear();
        localForceNeighborDeltaMaxRef.current = 0;
        publishRuntimeMetrics({
          localForceMovedNodeCount: 0,
          localForceNeighborDeltaMax: 0,
        });
        directDrag.forceStarted = Boolean(
          localForceControllerRef.current?.dragStart(
            directDrag.id,
            nextPosition[0],
            nextPosition[1],
            nodes.length,
          ),
        );
      }
      performanceProbe.record("coordinate_transform", performance.now() - transformStartedAt);
      const pendingPrevious = directDrag.pendingPosition;
      const cachedPrevious = cachedPositions.get(directDrag.id);
      const previous = pendingPrevious
        ? { x: pendingPrevious[0], y: pendingPrevious[1] }
        : cachedPrevious;
      if (
        previous &&
        Math.hypot(nextPosition[0] - previous.x, nextPosition[1] - previous.y) < 0.5
      ) {
        return;
      }
      directDrag.pendingPosition = nextPosition;
      directDrag.pendingClientPosition = [event.clientX, event.clientY];
      if (directDragFrame === null) {
        directDragFrame = window.requestAnimationFrame(() => {
          void flushDirectDrag();
        });
      }
    };

    const handleBlankCanvasPointerMove = (event: PointerEvent) => {
      if (!blankCanvasGesture || blankCanvasGesture.pointerId !== event.pointerId) return;
      if (blankCanvasGesture.moved) return;
      blankCanvasGesture.moved = shouldBeginGraphDrag(
        blankCanvasGesture.startClientPosition,
        [event.clientX, event.clientY],
      );
    };

    const handleDirectPointerEnd = (event: PointerEvent) => {
      const active = directDrag;
      if (!active || active.pointerId !== event.pointerId) return;
      event.preventDefault();
      event.stopPropagation();
      directDrag = undefined;
      if (directDragFrame !== null) {
        window.cancelAnimationFrame(directDragFrame);
        directDragFrame = null;
      }
      const finalPosition = active.pendingPosition;
      const complete = async () => {
        if (finalPosition && !graph.destroyed) {
          const applyStartedAt = performance.now();
          await engineRef.current?.applyPositions({
            [active.id]: { x: finalPosition[0], y: finalPosition[1] },
          });
          performanceProbe.record("drag_target_apply", performance.now() - applyStartedAt, {
            count: 1,
            detail: { final: true },
          });
          cachedPositions.set(active.id, {
            x: finalPosition[0],
            y: finalPosition[1],
          });
        }
        if (active.forceStarted && !graph.destroyed) {
          const settledPosition = finalPosition ?? graph.getElementPosition(active.id);
          localForceControllerRef.current?.dragEnd({
            x: Number(settledPosition[0]),
            y: Number(settledPosition[1]),
            pinned: active.shiftKey || event.shiftKey,
          });
          localForceSettleTimerRef.current = window.setTimeout(() => {
            localForceSettleTimerRef.current = null;
            publishRuntimeMetrics({
              localForceSettledGeneration:
                runtimeMetricsRef.current.localForceSettledGeneration + 1,
            });
          }, 700);
        }
        await endInteractionLod();
        if (active.moved) finishNodeDrag(active.id, active.shiftKey || event.shiftKey);
        else {
          const node = interactionRuntimeRef.current.nodeById.get(active.id);
          if (node) interactionRuntimeRef.current.setSelectedNode(node);
          rootRef.current?.classList.remove("is-direct-manipulation");
        }
      };
      void complete();
      container.releasePointerCapture?.(event.pointerId);
    };

    const finishBlankCanvasGesture = (event: PointerEvent) => {
      const active = blankCanvasGesture;
      if (!active || active.pointerId !== event.pointerId) return;
      const moved = active.moved || shouldBeginGraphDrag(
        active.startClientPosition,
        [event.clientX, event.clientY],
      );
      blankCanvasGesture = undefined;
      if (event.type !== "pointercancel" && !moved) {
        interactionRuntimeRef.current.clearTransientFocus();
      }
    };

    const handleViewportPointerEnd = (event: PointerEvent) => {
      finishBlankCanvasGesture(event);
      if (!viewportGestureActiveRef.current) return;
      viewportGestureActiveRef.current = false;
      handleTransform();
    };

    const handleTransform = () => {
      if (transformTimerRef.current !== null) {
        window.clearTimeout(transformTimerRef.current);
      }
      transformTimerRef.current = window.setTimeout(() => {
        if (graph.destroyed) return;
        if (
          !isPaneVisibleRef.current
          || viewport.clientWidth <= 0
          || viewport.clientHeight <= 0
        ) return;
        if (viewportGestureActiveRef.current) {
          handleTransform();
          return;
        }
        if (viewportCullingPausedRef.current) {
          viewportCullingPausedRef.current = false;
          publishRuntimeMetrics({ viewportCullingPaused: false });
        }
        const zoom = graph.getZoom();
        const band = `${zoom < 0.55 ? "xs" : zoom < 0.85 ? "sm" : zoom > 1.9 ? "xl" : zoom > 1.45 ? "lg" : "md"}:${visualConfigRef.current.display.labelThreshold}`;
        if (band !== labelZoomBandRef.current && nodes.length <= 180) {
          labelZoomBandRef.current = band;
          const runtime = interactionRuntimeRef.current;
          void engine.setAppearance(async (isCurrent) => {
            await runtime.applyVisualStyles(graph, zoom, isCurrent);
            if (!isCurrent()) return;
            await runtime.applyHighlight(graph, activeSelectedIdRef.current);
          }, renderedSceneKeyRef.current);
        }
        const capturedCamera = engine.captureCamera();
        const position = capturedCamera.position;
        publishCameraDiagnostics({
          x: Number(position[0]),
          y: Number(position[1]),
          zoom: capturedCamera.zoom,
        }, {
          x: Number(capturedCamera.worldCenter[0]),
          y: Number(capturedCamera.worldCenter[1]),
        });
        {
          const bounds = container.getBoundingClientRect();
          const topLeft = graph.getCanvasByClient([
            bounds.left - 120,
            bounds.top - 120,
          ]);
          const bottomRight = graph.getCanvasByClient([
            bounds.right + 120,
            bounds.bottom + 120,
          ]);
          const viewportNodeIds = spatialIndexRef.current.queryRect({
            minX: Math.min(Number(topLeft[0]), Number(bottomRight[0])),
            minY: Math.min(Number(topLeft[1]), Number(bottomRight[1])),
            maxX: Math.max(Number(topLeft[0]), Number(bottomRight[0])),
            maxY: Math.max(Number(topLeft[1]), Number(bottomRight[1])),
          });
          viewportNodeIdsRef.current = {
            sceneKey: renderedSceneKeyRef.current,
            nodeIds: viewportNodeIds,
          };
          const currentSceneIds = visibilityRequestRef.current.sceneNodeIds;
          setIsGraphOffscreen(isSceneOutsideViewport(currentSceneIds, viewportNodeIds));
          if (nodes.length > 300) {
            const coverage = viewportNodeIds.size / Math.max(1, currentSceneIds.size);
            void applyUnifiedVisibility(
              graph,
              coverage >= 0.6
                ? undefined
                : resolveViewportCullSnapshot(
                    viewportNodeIdsRef.current,
                    renderedSceneKeyRef.current,
                  ),
            );
          }
        }
        const snapshot: GraphPreviewViewSnapshot = {
          graphVersionId: graphVersion?.id ?? scene?.graphVersionId,
          mode: interactionRuntimeRef.current.activeViewMode,
          depth: interactionRuntimeRef.current.activeDepth,
          theme: visualConfigRef.current.theme,
          layoutPreset: interactionRuntimeRef.current.activeLayoutPreset,
          rendererPreference: activeRendererPreference,
          focusNodeIds: [...visualConfigRef.current.focusNodeIds],
          pathEndpointIds: [...interactionRuntimeRef.current.activePathEndpointIds],
          pinnedNodes: visualConfigRef.current.pinnedNodes,
          camera: capturedCamera,
        };
        const snapshotKey = viewSnapshotKey(snapshot);
        if (snapshotKey === lastEmittedViewSnapshotKeyRef.current) return;
        lastEmittedViewSnapshotKeyRef.current = snapshotKey;
        onViewStateChangeRef.current?.(snapshot);
      }, nodes.length > 300 ? 250 : 80);
    };

    const handleBeforeTransform = () => pauseViewportCulling();
    graph.on(NodeEvent.CLICK, handleNodeClick);
    graph.on(NodeEvent.DBLCLICK, handleNodeDoubleClick);
    graph.on(GraphEvent.BEFORE_TRANSFORM, handleBeforeTransform);
    graph.on(GraphEvent.AFTER_TRANSFORM, handleTransform);
    // A single spatial-indexed controller owns node dragging at every size.
    // Blank-space gestures still bubble to G6's drag-canvas. Neighbouring
    // nodes are relaxed by the worker, so a second G6 drag behavior cannot
    // compete for the same pointer and accidentally pan the whole canvas.
    container.addEventListener("pointerdown", handleDirectPointerDown, true);
    container.addEventListener("pointermove", handleDirectPointerMove, true);
    container.addEventListener("pointerup", handleDirectPointerEnd, true);
    container.addEventListener("pointercancel", handleDirectPointerEnd, true);
    window.addEventListener("pointermove", handleBlankCanvasPointerMove, true);
    window.addEventListener("pointerup", handleViewportPointerEnd, true);
    window.addEventListener("pointercancel", handleViewportPointerEnd, true);

    const resizeObserver = new ResizeObserver(() => {
      // The observer fires once during renderer construction. Restoring that
      // pre-fit camera would move every node offscreen.
      if (!initialFitComplete) return;
      if (resizeFrameRef.current !== null) {
        window.cancelAnimationFrame(resizeFrameRef.current);
      }
      resizeFrameRef.current = window.requestAnimationFrame(() => {
        resizeFrameRef.current = null;
        const width = viewport.clientWidth;
        const height = viewport.clientHeight;
        if (isPaneVisibleRef.current && isCurrentGraph() && width > 0 && height > 0) {
          const engine = engineRef.current;
          void engine?.resizePreservingWorldCenter(
            width,
            height,
          ).then(() => {
            if (cancelled || graph.destroyed) return;
            const captured = engine.captureCamera();
            publishCameraDiagnostics({
              x: Number(captured.position[0]),
              y: Number(captured.position[1]),
              zoom: captured.zoom,
            }, captured.worldCenter
              ? { x: Number(captured.worldCenter[0]), y: Number(captured.worldCenter[1]) }
              : lastWorldCenterRef.current);
          });
        }
      });
    });
    // G6 owns the host and its child canvas dimensions. Observe the stable
    // viewport wrapper so an inline renderer size cannot hide pane changes.
    resizeObserver.observe(viewport);

    void graph
      .render()
      .then(async () => {
        if (!isCurrentGraph()) return;
        publishRuntimeMetrics({
          drawCount: runtimeMetricsRef.current.drawCount + 1,
        });
        for (const [id, point] of Object.entries(activePinnedNodes)) {
          if (!isCurrentGraph()) return;
          if (!nodeById.has(id)) continue;
          setForceFixedPosition(graph, id, [point.x, point.y]);
          await graph.translateElementTo(id, [point.x, point.y], false);
          if (!isCurrentGraph()) return;
        }
        const indexedPositions = nodes.map((node) => {
          const point = graph.getElementPosition(node.id);
          return { id: node.id, x: Number(point[0]), y: Number(point[1]) };
        });
        spatialIndexRef.current.rebuild(indexedPositions);
        const benchmarkParams = new URLSearchParams(window.location.search);
        if (
          benchmarkParams.get("benchmark") === "graph" ||
          benchmarkParams.get("profile") === "canvas"
        ) {
          const oracle = spatialIndexRef.current.diagnosePicking(200, 1);
          publishRuntimeMetrics({
            pickOracleChecked: oracle.checked,
            pickOracleMismatches: oracle.mismatches,
            pickOracleP95Ms: oracle.durationP95Ms,
            pickOracleCandidatesP95: oracle.candidateP95,
          });
        }
        localForceControllerRef.current?.initialize(
          graphIdentity,
          nodes,
          edges,
          new Map(indexedPositions.map((point) => [point.id, point])),
          new Set(Object.keys(activePinnedNodes)),
        );
        await applyUnifiedVisibility(graph);
        if (!isCurrentGraph()) return;
        const initialVisualConfig = visualConfigRef.current;
        if (
          activeSelectedIdRef.current ||
          initialVisualConfig.pathNodeIds.size > 0 ||
          Object.keys(initialVisualConfig.pinnedNodes).length > 0
        ) {
          await applyHighlight(graph, activeSelectedIdRef.current);
          if (!isCurrentGraph()) return;
        }
        let fitClipped = false;
        let initialFitPerformed = false;
        let initialCameraApplied = false;
        if (shouldRestoreCamera && hasLiveViewport()) {
          initialCameraApplied = Boolean(await engine.restoreCamera(
            externalCameraToRestore && "position" in externalCameraToRestore
              ? externalCameraToRestore
              : {
              position: [cameraToRestore.x, cameraToRestore.y],
              zoom: cameraToRestore.zoom,
              ...(worldCenterToRestore
                ? { worldCenter: [worldCenterToRestore.x, worldCenterToRestore.y] }
                : {}),
                },
            sceneDataKey,
            externalCameraToRestore ? cameraRestoreCommand?.token : undefined,
            cameraRestoreTokenScope,
            paneActivationRef.current,
          ));
        } else if (!shouldRestoreCamera && hasLiveViewport()) {
          initialCameraApplied = Boolean(await engine.runCameraForScene(sceneDataKey, ++sceneCameraCommandTokenRef.current, async () => {
            if (!hasLiveViewport()) return;
            const result = await cameraControllerRef.current?.fit(
              nodes.map((node) => node.id),
              false,
            );
            fitClipped = Boolean(result?.clipped);
            initialFitPerformed = true;
          }));
        }
        if (!isCurrentGraph()) return;
        initialFitComplete = true;
        // The first ResizeObserver notification can arrive while the initial
        // fit is still running and is intentionally ignored above. Reconcile
        // the adapter with the final CSS viewport before publishing readiness;
        // otherwise the first user resize may preserve a stale construction
        // centre and visibly shift the graph.
        if (hasLiveViewport()) {
          await engine.resizePreservingWorldCenter(
            viewport.clientWidth,
            viewport.clientHeight,
          );
          if (!isCurrentGraph()) return;
        }
        const viewportIsLive = hasLiveViewport();
        const stableCamera = viewportIsLive ? engine.captureCamera() : undefined;
        if (stableCamera && initialCameraApplied) {
          const stableWorldCenter = stableCamera.worldCenter
            ? {
                x: Number(stableCamera.worldCenter[0]),
                y: Number(stableCamera.worldCenter[1]),
              }
            : lastWorldCenterRef.current;
          publishCameraDiagnostics({
            x: Number(stableCamera.position[0]),
            y: Number(stableCamera.position[1]),
            zoom: stableCamera.zoom,
          }, stableWorldCenter);
          hasStableCameraRef.current = true;
        }
        if (!isCurrentGraph()) return;
        renderedSceneKeyRef.current = sceneDataKey;
        renderedTopologyKeyRef.current = cacheTopologyKey;
        renderedNodeIdsRef.current = nodes.map((node) => node.id);
        if (viewportIsLive && initialCameraApplied) {
          fittedVisiblePaneTopologiesRef.current.add(cacheTopologyKey);
        }
        setEngineReady(true);
        committedSceneDataKeyRef.current = sceneDataKey;
        setCommittedSceneDataKey(sceneDataKey);
        if (!isCurrentGraph()) return;
        const initialDragTarget = findCentralDragTarget(graph, nodes, container);
        diagnosticNodeIdRef.current = initialDragTarget?.nodeId ?? nodes[0]?.id ?? null;
        publishNodeCoordinateDiagnostics();
        publishRuntimeMetrics({
          ready: true,
          fitViewCount:
            runtimeMetricsRef.current.fitViewCount + Number(initialFitPerformed),
          dragTarget: initialDragTarget,
        });
        if (fitClipped) {
          setStatus("图谱已适配视口，但部分标签仍可能靠近画布边缘");
        }
        // Fit/layout transforms can settle on the next few frames. Refresh the
        // benchmark target after that settling period so automation hits the
        // actual rendered node rather than a stale pre-transform coordinate.
        dragTargetRefreshTimer = window.setTimeout(() => {
          if (graph.destroyed) return;
          publishRuntimeMetrics({
            dragTarget: findCentralDragTarget(graph, nodes, container),
            performanceSamples: performanceProbe.snapshot(),
          });
          dragTargetRefreshTimer = null;
        }, 600);
        const runtimeViewMode = interactionRuntimeRef.current.activeViewMode;
        setStatus(
          (scene?.truncated ?? preview?.truncated)
            ? `局部预览已就绪：显示 ${nodes.length} 个节点与 ${edges.length} 条关系`
            : `${runtimeViewMode === "global" ? "全局图" : runtimeViewMode === "local" ? "局部图" : "路径图"}已就绪：${nodes.length} 个节点，${edges.length} 条关系`,
        );
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        const message = error instanceof Error ? error.message : "未知渲染错误";
        if (loadedRenderer.status.resolved === "hybrid-webgl") {
          setRendererFailure({
            graphIdentity,
            preference: activeRendererPreference,
            reason: `WEBGL_INITIALIZATION_FAILED: ${message}`,
          });
          setStatus("WebGL 初始化失败，正在回退到 Canvas");
          return;
        }
        setRenderError(`图谱渲染失败：${message}`);
        setStatus("图谱渲染失败");
      });

    return () => {
      cancelled = true;
      setEngineReady(false);
      committedSceneDataKeyRef.current = "";
      setCommittedSceneDataKey("");
      publishRuntimeMetrics({ ready: false });
      resizeObserver.disconnect();
      if (transformTimerRef.current !== null) {
        window.clearTimeout(transformTimerRef.current);
        transformTimerRef.current = null;
      }
      if (resizeFrameRef.current !== null) {
        window.cancelAnimationFrame(resizeFrameRef.current);
        resizeFrameRef.current = null;
      }
      if (interactionReleaseTimerRef.current !== null) {
        window.clearTimeout(interactionReleaseTimerRef.current);
        interactionReleaseTimerRef.current = null;
      }
      if (dragTargetRefreshTimer !== null) {
        window.clearTimeout(dragTargetRefreshTimer);
        dragTargetRefreshTimer = null;
      }
      if (directDragFrame !== null) {
        window.cancelAnimationFrame(directDragFrame);
        directDragFrame = null;
      }
      if (visibilityFrameRef.current !== null) {
        window.cancelAnimationFrame(visibilityFrameRef.current);
        visibilityFrameRef.current = null;
      }
      if (forceFrameRef.current !== null) {
        window.cancelAnimationFrame(forceFrameRef.current);
        forceFrameRef.current = null;
      }
      pendingForceFrameRef.current = null;
      forceApplyInFlightRef.current = false;
      if (localForceSettleTimerRef.current !== null) {
        window.clearTimeout(localForceSettleTimerRef.current);
        localForceSettleTimerRef.current = null;
      }
      localForceControllerRef.current?.destroy();
      localForceControllerRef.current = null;
      container.removeEventListener("pointerdown", handleDirectPointerDown, true);
      container.removeEventListener("pointermove", handleDirectPointerMove, true);
      container.removeEventListener("pointerup", handleDirectPointerEnd, true);
      container.removeEventListener("pointercancel", handleDirectPointerEnd, true);
      window.removeEventListener("pointermove", handleBlankCanvasPointerMove, true);
      window.removeEventListener("pointerup", handleViewportPointerEnd, true);
      window.removeEventListener("pointercancel", handleViewportPointerEnd, true);
      viewportGestureActiveRef.current = false;
      viewportCullingPausedRef.current = false;
      activeInteractionLod = false;
      interactionVeil.remove();
      interactionProxy.remove();
      rootRef.current?.classList.remove("is-direct-manipulation");
      if (!graph.destroyed && initialFitComplete) {
        const cameraPosition = graph.getPosition();
        lastCameraRef.current = {
          x: Number(cameraPosition[0]),
          y: Number(cameraPosition[1]),
          zoom: graph.getZoom(),
        };
        const viewportCenter = graph.getViewportCenter();
        lastWorldCenterRef.current = {
          x: Number(viewportCenter[0]),
          y: Number(viewportCenter[1]),
        };
        for (const node of nodes) {
          try {
            const position = graph.getElementPosition(node.id);
            cachedPositions.set(node.id, {
              x: Number(position[0]),
              y: Number(position[1]),
            });
          } catch {
            // A same-engine scene replacement may have removed the old node.
          }
        }
      }
      graph.off(NodeEvent.CLICK, handleNodeClick);
      graph.off(NodeEvent.DBLCLICK, handleNodeDoubleClick);
      graph.off(GraphEvent.BEFORE_TRANSFORM, handleBeforeTransform);
      graph.off(GraphEvent.AFTER_TRANSFORM, handleTransform);
      detachPerformanceProbe();
      graph.destroy();
      const benchmarkWindow = window as typeof window & {
        __SGFM_GRAPH_BENCHMARK_GRAPH__?: Graph;
      };
      if (benchmarkWindow.__SGFM_GRAPH_BENCHMARK_GRAPH__ === graph) {
        delete benchmarkWindow.__SGFM_GRAPH_BENCHMARK_GRAPH__;
      }
      publishRuntimeMetrics({
        ready: false,
        dragTarget: undefined,
        viewportCullingPaused: false,
        interactionLodActive: false,
        engineDestroyCount: runtimeMetricsRef.current.engineDestroyCount + 1,
      });
      if (graphRef.current === graph) {
        graphRef.current = null;
        engineRef.current = null;
        cameraControllerRef.current = null;
        visibilityControllerRef.current = null;
        performanceProbeRef.current?.dispose();
        performanceProbeRef.current = null;
      }
    };
    // Scene/style/selection changes are applied through GraphEngineAdapter and
    // refs below. Recreate only for a different immutable graph identity.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    graphIdentity,
    loadedRenderer?.status.resolved,
    nodes.length > 0,
    rendererLoadRequestKey,
  ]);

  useEffect(() => {
    const graph = graphRef.current;
    const engine = engineRef.current;
    if (
      !engineReady ||
      !graph ||
      !engine ||
      graph.destroyed ||
      !graph.rendered ||
      renderedSceneKeyRef.current === sceneDataKey
        && committedSceneDataKeyRef.current === sceneDataKey
    ) {
      return;
    }

    const previousTopologyKey = renderedTopologyKeyRef.current || topologyKey;
    const topologyChanged = previousTopologyKey !== topologyKey;
    const previousPositions = positionCacheRef.current.get(previousTopologyKey)
      ?? new Map<string, { readonly x: number; readonly y: number }>();
    positionCacheRef.current.set(previousTopologyKey, previousPositions);
    const currentLiveNodeIds = new Set(graph.getNodeData().map((node) => String(node.id)));
    for (const id of renderedNodeIdsRef.current) {
      if (!currentLiveNodeIds.has(id)) continue;
      try {
        const position = graph.getElementPosition(id);
        previousPositions.set(id, {
          x: Number(position[0]),
          y: Number(position[1]),
        });
      } catch {
        // The previous scene may have removed the element between frames.
      }
    }
    const nextPositions = positionCacheRef.current.get(topologyKey)
      ?? new Map<string, { readonly x: number; readonly y: number }>();
    for (const node of nodes) {
      const retained = previousPositions.get(node.id);
      if (retained && !nextPositions.has(node.id)) nextPositions.set(node.id, retained);
    }
    positionCacheRef.current.set(topologyKey, nextPositions);
    committedSceneDataKeyRef.current = "";
    setCommittedSceneDataKey("");
    viewportNodeIdsRef.current = undefined;
    publishRuntimeMetrics({
      ready: false,
      drawCount: runtimeMetricsRef.current.drawCount + 1,
      dragTarget: undefined,
    });

    let cancelled = false;
    let sceneLease: GraphSceneLease | null = null;
    const isCurrentScene = (lease = sceneLease) => Boolean(
      lease
      && !cancelled
      && !graph.destroyed
      && graphRef.current === graph
      && engineRef.current === engine
      && engine.isSceneLeaseCurrent(lease),
    );
    void engine
      .replaceScene(graphDataRef.current, {
        layout: "preserve",
        camera: "preserve",
      }, sceneDataKey)
      .then(async (lease) => {
        sceneLease = lease;
        if (!lease || !isCurrentScene(lease)) return;
        renderedSceneKeyRef.current = sceneDataKey;
        const hasCachedTopology = nodes.length > 0 && nodes.every((node) => nextPositions.has(node.id));
        if (shouldRelayoutProjection({
          topologyChanged,
          hasCachedTopology,
          externalFocusActive: Boolean(cameraFocusCommand?.nodeIds.length),
        })) {
          await engine.setForces(
            forceLayoutOptions(
              forceSettings,
              nodes.length,
              displaySettings.nodeScale,
              `${graphIdentity}:${topologyKey}`,
            ),
          );
          if (!isCurrentScene(lease)) return;
          publishRuntimeMetrics({
            layoutCount: runtimeMetricsRef.current.layoutCount + 1,
          });
        }
        if (!isCurrentScene(lease)) return;
        highlightedElementsRef.current = {
          nodes: new Set<string>(),
          edges: new Set<string>(),
        };
        const appearanceApplied = await engine.runSceneTransaction(
          lease,
          async (leaseIsCurrent) => {
            await interactionRuntimeRef.current.applyVisualStyles(
              graph,
              graph.getZoom(),
              leaseIsCurrent,
            );
          },
        );
        if (!appearanceApplied || !isCurrentScene(lease)) return;
        visibilityControllerRef.current = new GraphVisibilityController(
          interactionRuntimeRef.current.nodes,
          interactionRuntimeRef.current.edges,
        );
        const indexedPositions = interactionRuntimeRef.current.nodes.map((node) => {
          const point = graph.getElementPosition(node.id);
          nextPositions.set(node.id, { x: Number(point[0]), y: Number(point[1]) });
          return { id: node.id, x: Number(point[0]), y: Number(point[1]) };
        });
        renderedTopologyKeyRef.current = topologyKey;
        renderedNodeIdsRef.current = nodes.map((node) => node.id);
        spatialIndexRef.current.rebuild(indexedPositions);
        localForceControllerRef.current?.initialize(
          graphIdentity,
          interactionRuntimeRef.current.nodes,
          interactionRuntimeRef.current.edges,
          new Map(indexedPositions.map((point) => [point.id, point])),
          new Set(Object.keys(visualConfigRef.current.pinnedNodes)),
        );
        // Reveal the replacement topology before deriving a fresh cull set;
        // the previous scene's viewport ids may be empty or share unrelated
        // ids with this graph.
        await applyUnifiedVisibility(graph);
        if (!isCurrentScene(lease)) return;
        await new Promise<void>((resolve) =>
          window.requestAnimationFrame(() => resolve()),
        );
        if (!isCurrentScene(lease)) return;
        const container = canvasRef.current;
        if (container) {
          const bounds = container.getBoundingClientRect();
          const topLeft = graph.getCanvasByClient([
            bounds.left - 120,
            bounds.top - 120,
          ]);
          const bottomRight = graph.getCanvasByClient([
            bounds.right + 120,
            bounds.bottom + 120,
          ]);
          const viewportNodeIds = spatialIndexRef.current.queryRect({
            minX: Math.min(Number(topLeft[0]), Number(bottomRight[0])),
            minY: Math.min(Number(topLeft[1]), Number(bottomRight[1])),
            maxX: Math.max(Number(topLeft[0]), Number(bottomRight[0])),
            maxY: Math.max(Number(topLeft[1]), Number(bottomRight[1])),
          });
          const viewportSnapshot = { sceneKey: sceneDataKey, nodeIds: viewportNodeIds };
          await applyUnifiedVisibility(
            graph,
            interactionRuntimeRef.current.nodes.length > 300
              ? resolveViewportCullSnapshot(viewportSnapshot, sceneDataKey)
              : undefined,
          );
          if (!isCurrentScene(lease)) return;
          viewportNodeIdsRef.current = viewportSnapshot;
          setIsGraphOffscreen(
            isSceneOutsideViewport(
              visibilityRequestRef.current.sceneNodeIds,
              viewportNodeIds,
            ),
          );
        }
        if (!isCurrentScene(lease)) return;
        const highlightApplied = await engine.runSceneTransaction(
          lease,
          async (leaseIsCurrent) => {
            if (!leaseIsCurrent()) return;
            await interactionRuntimeRef.current.applyHighlight(
              graph,
              activeSelectedIdRef.current,
            );
          },
        );
        if (!highlightApplied || !isCurrentScene(lease)) return;
        publishRuntimeMetrics({
          ready: true,
          dragTarget: readDragTarget(
            graph,
            nodes.at(-1)?.id,
            canvasRef.current,
          ),
        });
        setEngineReady(true);
        committedSceneDataKeyRef.current = sceneDataKey;
        setCommittedSceneDataKey(sceneDataKey);
        const runtimeViewMode = interactionRuntimeRef.current.activeViewMode;
        setStatus(
          `${runtimeViewMode === "global" ? "全局图" : runtimeViewMode === "local" ? "局部图" : "路径图"}已更新：${nodes.length} 个节点，${edges.length} 条关系`,
        );
      })
      .catch((error: unknown) => {
        if (
          cancelled
          || graph.destroyed
          || graphRef.current !== graph
          || engineRef.current !== engine
          || (sceneLease && !engine.isSceneLeaseCurrent(sceneLease))
        ) return;
        setRenderError(
          `图谱更新失败：${error instanceof Error ? error.message : "未知错误"}`,
        );
        publishRuntimeMetrics({ ready: false });
        committedSceneDataKeyRef.current = "";
        setCommittedSceneDataKey("");
      });
    return () => {
      cancelled = true;
    };
  }, [
    activeViewMode,
    applyUnifiedVisibility,
    engineReady,
    publishRuntimeMetrics,
    sceneDataKey,
    topologyKey,
    graphIdentity,
    forceSettings,
  ]);

  useEffect(() => {
    const graph = graphRef.current;
    const engine = engineRef.current;
    if (!engineReady || !graph || !engine || graph.destroyed || !graph.rendered) return;
    const forceKey = JSON.stringify(forceSettings);
    if (lastForceKeyRef.current === forceKey) return;
    if (!shouldRelayoutProjection({
      topologyChanged: true,
      hasCachedTopology: false,
      externalFocusActive: Boolean(cameraFocusCommand?.nodeIds.length),
    })) return;
    lastForceKeyRef.current = forceKey;
    const sequence = ++forceUpdateSequenceRef.current;
    const layout = forceLayoutOptions(
      forceSettings,
      nodes.length,
      displaySettings.nodeScale,
      `${graphVersion?.id ?? "scene"}:${nodes.length}`,
    );
    void engine
      .setForces(layout)
      .then(() => {
        if (sequence !== forceUpdateSequenceRef.current || graph.destroyed) return;
        publishRuntimeMetrics({
          layoutCount: runtimeMetricsRef.current.layoutCount + 1,
        });
        setStatus("力学参数已应用，视图位置保持不变");
      })
      .catch((error: unknown) => {
        if (sequence !== forceUpdateSequenceRef.current || graph.destroyed) return;
        setRenderError(
          `布局更新失败：${error instanceof Error ? error.message : "未知错误"}`,
        );
      });
  }, [
    cameraFocusCommand?.token,
    forceSettings,
    graphVersion?.id,
    publishRuntimeMetrics,
  ]);

  useEffect(() => {
    const graph = graphRef.current;
    if (!engineReady || committedSceneDataKey !== sceneDataKey || !graph || graph.destroyed || !graph.rendered) return;
    const currentIds = new Set(Object.keys(activePinnedNodes));
    for (const id of appliedPinnedIdsRef.current) {
      if (!currentIds.has(id)) setForceFixedPosition(graph, id, null);
    }
    for (const [id, point] of Object.entries(activePinnedNodes)) {
      if (!nodeById.has(id)) continue;
      setForceFixedPosition(graph, id, [point.x, point.y]);
      void graph.translateElementTo(id, [point.x, point.y], false);
    }
    appliedPinnedIdsRef.current = currentIds;
    if (localForceControllerRef.current) {
      const currentPositions = new Map<string, { x: number; y: number }>();
      for (const node of nodes) {
        const point = graph.getElementPosition(node.id);
        currentPositions.set(node.id, { x: Number(point[0]), y: Number(point[1]) });
      }
      localForceControllerRef.current.initialize(
        graphIdentity,
        nodes,
        edges,
        currentPositions,
        currentIds,
      );
    }
    void applyCommittedHighlight(graph, activeSelectedIdRef.current, sceneDataKey);
  }, [activePinnedNodes, applyCommittedHighlight, committedSceneDataKey, edges, graphIdentity, nodeById, nodes, sceneDataKey]);

  useEffect(() => {
    const graph = graphRef.current;
    const engine = engineRef.current;
    if (
      !engineReady
      || committedSceneDataKey !== sceneDataKey
      || !graph
      || !engine
      || graph.destroyed
      || !graph.rendered
    ) return;
    labelZoomBandRef.current = "";
    void engine.setAppearance(async (isCurrent) => {
      await applyVisualStyles(graph, graph.getZoom(), isCurrent);
      if (!isCurrent()) return;
      await applyHighlight(graph, activeSelectedIdRef.current);
    }, sceneDataKey);
  }, [
    activeTheme,
    applyHighlight,
    applyVisualStyles,
    displaySettings.arrows,
    displaySettings.edgeScale,
    displaySettings.labelThreshold,
    displaySettings.nodeScale,
    engineReady,
    appearanceRequestKey,
    committedSceneDataKey,
    overlay,
    sceneDataKey,
  ]);

  useEffect(() => {
    const graph = graphRef.current;
    if (
      !engineReady
      || committedSceneDataKey !== sceneDataKey
      || !graph
      || graph.destroyed
    ) return;
    void applyCommittedHighlight(graph, activeSelectedId ?? null, sceneDataKey);
  }, [activeSelectedId, applyCommittedHighlight, committedSceneDataKey, engineReady, sceneDataKey]);

  useEffect(() => {
    const graph = graphRef.current;
    if (
      !engineReady ||
      !graph ||
      graph.destroyed ||
      !graph.rendered ||
      !nodes.length
    ) return;

    void applyUnifiedVisibility(
      graph,
      nodes.length > 300
        ? resolveViewportCullSnapshot(viewportNodeIdsRef.current, sceneDataKey)
        : undefined,
    );
  }, [
    activeFilters,
    activePinnedNodes,
    activeSelectedId,
    applyUnifiedVisibility,
    engineReady,
    nodes.length,
    pathNodeIds,
    sceneDataKey,
    sceneEdgeIds,
    sceneNodeIds,
  ]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === rootRef.current);
      window.requestAnimationFrame(() => {
        const engine = engineRef.current;
        const camera = cameraControllerRef.current;
        if (!engine || !camera) return;
        const stableWorldCenter = lastWorldCenterRef.current;
        void engine.resizePreservingWorldCenter(
          undefined,
          undefined,
          stableWorldCenter
            ? [stableWorldCenter.x, stableWorldCenter.y]
            : undefined,
        ).then(async () => {
          if (camera.hasVisibleElement(nodes.map((node) => node.id))) return;
          const request = ++sceneCameraCommandTokenRef.current;
          const completed = await engine.runCameraForScene(sceneDataKey, request, async () => {
            await camera.fit(nodes.map((node) => node.id), false);
          });
          if (completed) {
            setViewportFeedback("视图已自动找回");
            if (viewportFeedbackTimerRef.current !== null) {
              window.clearTimeout(viewportFeedbackTimerRef.current);
            }
            viewportFeedbackTimerRef.current = window.setTimeout(() => {
              setViewportFeedback("");
              viewportFeedbackTimerRef.current = null;
            }, 1_200);
          }
        }).then(() => {
          const captured = engine.captureCamera();
          publishCameraDiagnostics({
            x: Number(captured.position[0]),
            y: Number(captured.position[1]),
            zoom: captured.zoom,
          }, captured.worldCenter
            ? { x: Number(captured.worldCenter[0]), y: Number(captured.worldCenter[1]) }
            : lastWorldCenterRef.current);
        });
      });
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () =>
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [nodes, publishCameraDiagnostics, sceneDataKey]);

  const commitFilters = useCallback(
    (next: GraphFilters) => {
      const normalized = normalizeGraphFilters(next);
      setInternalFilters(normalized);
      onFiltersChange?.(normalized);
    },
    [onFiltersChange],
  );

  const commitEnabledTypes = useCallback(
    (next: ReadonlySet<string>) => {
      const ordered = types.filter((type) => next.has(type));
      commitFilters({
        ...activeFilters,
        // The canonical representation of "show all" is an empty list.
        nodeTypes: ordered.length === types.length ? [] : ordered,
      });
    },
    [activeFilters, commitFilters, types],
  );

  const selectAndFocusNode = useCallback(
    async (node: GraphNode) => {
      const focusRequest = transientFocusRequestRef.current + 1;
      transientFocusRequestRef.current = focusRequest;
      if (!enabledTypes.has(nodeType(node))) {
        commitEnabledTypes(new Set([...enabledTypes, nodeType(node)]));
        await Promise.resolve();
      }
      if (focusRequest !== transientFocusRequestRef.current) return;
      setSelectedNode(node);
      const graph = graphRef.current;
      const camera = cameraControllerRef.current;
      if (!graph || graph.destroyed || !camera) return;
      const neighbourhoodIds = [
        node.id,
        ...adjacencyIndex.neighbours(node.id),
      ].filter((id) => nodes.some((candidate) => candidate.id === id));
      const expectedSceneIdentity = renderedSceneKeyRef.current;
      const visible = await engineRef.current?.ensureVisible(
        neighbourhoodIds,
        expectedSceneIdentity,
        focusRequest,
        "internal-search",
      );
      if (!visible || focusRequest !== transientFocusRequestRef.current || graph.destroyed) return;
      const focused = await engineRef.current?.runCameraForScene(
        expectedSceneIdentity,
        ++sceneCameraCommandTokenRef.current,
        async () => camera.focus(neighbourhoodIds, {
          anchorElementId: node.id,
          minZoom: 0.72,
          maxZoom: 1.35,
          animation: prefersReducedMotion()
            ? false
            : { duration: 280, easing: "ease-out" },
        }).then(() => undefined),
      );
      if (!focused || focusRequest !== transientFocusRequestRef.current || graph.destroyed) return;
      await applyCommittedHighlight(graph, node.id, expectedSceneIdentity);
      if (focusRequest !== transientFocusRequestRef.current) {
        await applyCommittedHighlight(graph, null, expectedSceneIdentity);
        return;
      }
      setSearchFeedback(`已定位 ${node.label}`);
      if (searchFeedbackTimerRef.current !== null) {
        window.clearTimeout(searchFeedbackTimerRef.current);
      }
      searchFeedbackTimerRef.current = window.setTimeout(() => {
        setSearchFeedback("");
        searchFeedbackTimerRef.current = null;
      }, 1_200);
    },
    [adjacencyIndex, applyCommittedHighlight, commitEnabledTypes, enabledTypes, nodes, setSelectedNode],
  );

  const searchMatches = useMemo(() => {
    const term = query.trim().toLocaleLowerCase("zh-CN");
    if (!term) return [];
    return fullNodes
      .filter(
        (node) =>
          node.id.toLocaleLowerCase("zh-CN").includes(term) ||
          node.label.toLocaleLowerCase("zh-CN").includes(term),
      )
      .sort((left, right) => {
        const leftExact =
          left.id.toLocaleLowerCase("zh-CN") === term ||
          left.label.toLocaleLowerCase("zh-CN") === term;
        const rightExact =
          right.id.toLocaleLowerCase("zh-CN") === term ||
          right.label.toLocaleLowerCase("zh-CN") === term;
        return Number(rightExact) - Number(leftExact) || left.label.localeCompare(right.label, "zh-CN");
      })
      .slice(0, 8);
  }, [fullNodes, query]);

  const chooseSearchResult = useCallback(
    (node: GraphNode) => {
      if (!nodes.some((candidate) => candidate.id === node.id)) {
        setPendingSearchNodeId(node.id);
        setSearchFeedback(`正在载入 ${node.label} 的关联场景`);
        setSearchOpen(false);
        changeFocusNodeIds([node.id]);
        changeViewMode("local");
        return;
      }
      setPendingSearchNodeId(null);
      setSearchFeedback(`正在定位 ${node.label}`);
      setSearchOpen(false);
      void selectAndFocusNode(node);
    },
    [changeFocusNodeIds, changeViewMode, nodes, selectAndFocusNode],
  );

  useEffect(() => {
    if (!pendingSearchNodeId || !engineReady) return;
    const node = nodeById.get(pendingSearchNodeId);
    if (!node || !nodes.some((candidate) => candidate.id === node.id)) return;
    setPendingSearchNodeId(null);
    void selectAndFocusNode(node);
  }, [engineReady, nodeById, nodes, pendingSearchNodeId, selectAndFocusNode]);

  const submitSearch = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const match = searchMatches[Math.min(activeSearchIndex, searchMatches.length - 1)];
    if (!match) {
      setSearchFeedback(`未找到“${query.trim()}”`);
      setSearchOpen(false);
      return;
    }
    chooseSearchResult(match);
  };

  const toggleType = (type: string) => {
    const selected = nodeById.get(activeSelectedId ?? "");
    const next = new Set(enabledTypes);
    if (next.has(type)) {
      if (next.size === 1) {
        setStatus("至少需要保留一种节点类型");
        return;
      }
      next.delete(type);
      if (selected && nodeType(selected) === type) setSelectedNode(null);
    } else {
      next.add(type);
    }
    commitEnabledTypes(next);
    setStatus(`已显示 ${next.size}/${types.length} 种节点类型`);
  };

  const zoom = (ratio: number) => {
    const graph = graphRef.current;
    const container = canvasRef.current;
    const engine = engineRef.current;
    if (!graph || graph.destroyed || !container || !engine) return;
    const targetZoom = Math.max(0.01, Math.min(4, (pendingZoomTargetRef.current ?? lastCameraRef.current.zoom) * ratio));
    const request = ++zoomRequestRef.current;
    pendingZoomTargetRef.current = targetZoom;
    lastCameraRef.current = { ...lastCameraRef.current, zoom: targetZoom };
    const bounds = container.getBoundingClientRect();
    void engine.runCameraForScene(renderedSceneKeyRef.current, ++sceneCameraCommandTokenRef.current, async () => {
      await graph.zoomTo(
        targetZoom,
        false,
        [bounds.width / 2, bounds.height / 2],
      );
    }).then((completed) => {
      if (!completed || request !== zoomRequestRef.current || graphRef.current !== graph || graph.destroyed) return;
      const captured = engineRef.current?.captureCamera();
      const position = captured?.position ?? graph.getPosition();
      const legacyCamera = {
        x: Number(position[0]),
        y: Number(position[1]),
        zoom: captured?.zoom ?? graph.getZoom(),
      };
      const camera = captured ?? completeCameraSnapshot(
        sceneDataKey,
        legacyCamera,
        lastWorldCenterRef.current,
        [container.clientWidth || 1, container.clientHeight || 1],
      );
      publishCameraDiagnostics(legacyCamera, captured?.worldCenter
        ? { x: Number(captured.worldCenter[0]), y: Number(captured.worldCenter[1]) }
        : lastWorldCenterRef.current);
      const snapshot: GraphPreviewViewSnapshot = {
        graphVersionId: graphVersion?.id ?? scene?.graphVersionId,
        mode: activeViewMode,
        depth: activeDepth,
        theme: activeTheme,
        layoutPreset: activeLayoutPreset,
        rendererPreference: activeRendererPreference,
        focusNodeIds: [...visualConfigRef.current.focusNodeIds],
        pathEndpointIds: [...interactionRuntimeRef.current.activePathEndpointIds],
        pinnedNodes: visualConfigRef.current.pinnedNodes,
        camera,
      };
      const snapshotKey = viewSnapshotKey(snapshot);
      if (snapshotKey !== lastEmittedViewSnapshotKeyRef.current) {
        lastEmittedViewSnapshotKeyRef.current = snapshotKey;
        onViewStateChangeRef.current?.(snapshot);
      }
    }).finally(() => {
      if (request === zoomRequestRef.current) pendingZoomTargetRef.current = null;
    });
  };

  const fitView = () => {
    const graph = graphRef.current;
    const container = canvasRef.current;
    const cameraController = cameraControllerRef.current;
    if (!graph || graph.destroyed || !container || !cameraController) return;
    setViewportFeedback("正在适应视图…");
    const request = ++sceneCameraCommandTokenRef.current;
    let result: Awaited<ReturnType<GraphCameraController["fit"]>> | null = null;
    void engineRef.current?.runCameraForScene(sceneDataKey, request, async () => {
      result = await cameraController.fit(nodes.map((node) => node.id), false);
    }).then(async (completed) => {
      if (!completed || !result) return;
      await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));
      const bounds = container.getBoundingClientRect();
      const topLeft = graph.getCanvasByClient([bounds.left - 120, bounds.top - 120]);
      const bottomRight = graph.getCanvasByClient([bounds.right + 120, bounds.bottom + 120]);
      const viewportNodeIds = spatialIndexRef.current.queryRect({
        minX: Math.min(Number(topLeft[0]), Number(bottomRight[0])),
        minY: Math.min(Number(topLeft[1]), Number(bottomRight[1])),
        maxX: Math.max(Number(topLeft[0]), Number(bottomRight[0])),
        maxY: Math.max(Number(topLeft[1]), Number(bottomRight[1])),
      });
      viewportNodeIdsRef.current = { sceneKey: sceneDataKey, nodeIds: viewportNodeIds };
      await applyUnifiedVisibility(
        graph,
        nodes.length > 300
          ? resolveViewportCullSnapshot(viewportNodeIdsRef.current, sceneDataKey)
          : undefined,
      );
      const currentSceneIds = visibilityRequestRef.current.sceneNodeIds;
      setIsGraphOffscreen(isSceneOutsideViewport(currentSceneIds, viewportNodeIds));
      publishRuntimeMetrics({
        fitViewCount: runtimeMetricsRef.current.fitViewCount + 1,
        dragTarget: findCentralDragTarget(graph, nodes, container),
      });
      setStatus(
        result!.clipped
          ? "视图已缩放，部分标签仍靠近画布边缘"
          : "已适应当前可见图谱",
      );
      setViewportFeedback(result!.clipped ? "已缩放至可见范围" : "已适应视图");
      if (viewportFeedbackTimerRef.current !== null) {
        window.clearTimeout(viewportFeedbackTimerRef.current);
      }
      viewportFeedbackTimerRef.current = window.setTimeout(() => {
        setViewportFeedback("");
        viewportFeedbackTimerRef.current = null;
      }, 1_200);
    });
  };

  useEffect(() => {
    if (shouldAutoFitVisibleGraphPane(paneWasVisibleRef.current, isPaneVisible)) {
      pendingVisiblePaneFitRef.current = true;
    }
    paneWasVisibleRef.current = isPaneVisible;

    if (!isPaneVisible) {
      pendingVisiblePaneFitRef.current = false;
      return;
    }
    if (!engineReady || !pendingVisiblePaneFitRef.current) return;

    // The mobile tab removes the pane with display:none. Fit each topology only
    // on its first reveal; later tab switches retain the user's camera.
    pendingVisiblePaneFitRef.current = false;
    const shouldFit = !fittedVisiblePaneTopologiesRef.current.has(topologyKey);
    fittedVisiblePaneTopologiesRef.current.add(topologyKey);
    const hasSceneRestore = Boolean(cameraRestoreCommand && (
      !("sceneIdentity" in cameraRestoreCommand)
      || cameraRestoreCommand.sceneIdentity === sceneDataKey
    ));
    const retainedCamera = { ...lastCameraRef.current };
    const retainedWorldCenter = lastWorldCenterRef.current
      ? { ...lastWorldCenterRef.current }
      : null;
    const retainedViewportSize = lastViewportSizeRef.current
      ? [...lastViewportSizeRef.current] as [number, number]
      : null;
    let cancelled = false;
    const layoutFrame = window.requestAnimationFrame(() => {
      window.requestAnimationFrame(() => {
        if (cancelled) return;
        if (shouldFit && !hasSceneRestore) {
          fitView();
          return;
        }
        if (hasSceneRestore) return;
        const retainedSnapshot = retainedWorldCenter && retainedViewportSize
          ? {
              schemaVersion: "socialgraph-fm.graph-camera/2" as const,
              sceneIdentity: sceneDataKey,
              position: [retainedCamera.x, retainedCamera.y] as [number, number],
              zoom: retainedCamera.zoom,
              worldCenter: [retainedWorldCenter.x, retainedWorldCenter.y] as [number, number],
              viewportSize: retainedViewportSize,
            }
          : {
              position: [retainedCamera.x, retainedCamera.y] as [number, number],
              zoom: retainedCamera.zoom,
            };
        void engineRef.current?.restoreCamera(retainedSnapshot, sceneDataKey).then((restored) => {
          if (cancelled || !restored) return;
          const captured = engineRef.current?.captureCamera();
          if (!captured) return;
          publishCameraDiagnostics({
            x: Number(captured.position[0]),
            y: Number(captured.position[1]),
            zoom: captured.zoom,
          }, {
            x: Number(captured.worldCenter[0]),
            y: Number(captured.worldCenter[1]),
          });
        });
      });
    });
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(layoutFrame);
    };
  }, [cameraRestoreCommand?.token, engineReady, isPaneVisible, publishCameraDiagnostics, sceneDataKey, topologyKey]);

  useEffect(() => {
    const handleRecoverShortcut = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof Element &&
        target.matches("input, textarea, select, [contenteditable='true']")
      ) return;
      if (event.key !== "Home" && event.key !== "0") return;
      event.preventDefault();
      fitView();
    };
    window.addEventListener("keydown", handleRecoverShortcut);
    return () => window.removeEventListener("keydown", handleRecoverShortcut);
  });

  useEffect(() => {
    const handleClearFocus = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      clearTransientFocus();
    };
    window.addEventListener("keydown", handleClearFocus);
    return () => window.removeEventListener("keydown", handleClearFocus);
  }, [clearTransientFocus]);

  const toggleFullscreen = async () => {
    const root = rootRef.current;
    if (!root) return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else {
        await root.requestFullscreen();
      }
    } catch {
      setStatus("当前浏览器未允许进入全屏");
    }
  };

  const toggleSelectedPin = useCallback(() => {
    const id = activeSelectedId;
    const graph = graphRef.current;
    if (!id || !graph || graph.destroyed || !nodeById.has(id)) return;
    if (activePinnedNodes[id]) {
      const next = { ...activePinnedNodes };
      delete next[id];
      // Keep the last coordinate after unpinning; removing it would make the
      // next session appear to have an uncached topology and trigger a full
      // graph jump.
      setForceFixedPosition(graph, id, null);
      changePinnedNodes(next);
      setStatus(`已取消固定 ${nodeById.get(id)?.label ?? id}`);
      return;
    }
    const position = graph.getElementPosition(id);
    const point = { x: Number(position[0]), y: Number(position[1]) };
    setForceFixedPosition(graph, id, [point.x, point.y]);
    const topologyPositions = positionCacheRef.current.get(topologyKey)
      ?? new Map<string, { readonly x: number; readonly y: number }>();
    positionCacheRef.current.set(topologyKey, topologyPositions);
    topologyPositions.set(id, point);
    changePinnedNodes({ ...activePinnedNodes, [id]: point });
    setStatus(`已固定 ${nodeById.get(id)?.label ?? id}`);
  }, [
    activePinnedNodes,
    activeSelectedId,
    changePinnedNodes,
    nodeById,
    topologyKey,
  ]);

  const exportPng = useCallback(async () => {
    const graph = graphRef.current;
    if (!graph || graph.destroyed || !graph.rendered) {
      setStatus("图谱尚未就绪，无法导出 PNG");
      return;
    }
    try {
      const fileName = `${safeFileStem(graphVersion?.sourceFile)}-${activeViewMode}.png`;
      const dataUrl = await graph.toDataURL({
        mode: "overall",
        type: "image/png",
      });
      downloadDataUrl(dataUrl, fileName);
      onExport?.({ format: "png", fileName });
      setStatus(`已导出 ${fileName}`);
    } catch (error) {
      setStatus(
        `PNG 导出失败：${error instanceof Error ? error.message : "未知错误"}`,
      );
    }
  }, [activeViewMode, graphVersion?.sourceFile, onExport]);

  const exportJson = useCallback(() => {
    if (!graphVersion && !scene) {
      setStatus("图谱尚未就绪，无法导出视图配置");
      return;
    }
    const fileName = `${safeFileStem(graphVersion?.sourceFile)}-${activeViewMode}-view.json`;
    const position = graphRef.current?.getPosition() ?? [0, 0];
    downloadJson(
      {
        schemaVersion: "1.0",
        exportedAt: new Date().toISOString(),
        graphVersionId: graphVersion?.id ?? null,
        view: {
          mode: activeViewMode,
          depth: activeDepth,
          theme: activeTheme,
          layoutPreset: activeLayoutPreset,
          rendererPreference: activeRendererPreference,
          renderer: rendererStatus,
          focusNodeIds: activeFocusNodeIds,
          pathEndpointIds: activePathEndpointIds,
          enabledNodeTypes: [...enabledTypes],
          pinnedNodes: activePinnedNodes,
          force: forceSettings,
          display: displaySettings,
          camera: {
            x: Number(position[0]),
            y: Number(position[1]),
            zoom: graphRef.current?.getZoom() ?? 1,
          },
        },
      },
      fileName,
    );
    onExport?.({ format: "json", fileName });
    setStatus(`已导出 ${fileName}`);
  }, [
    activeDepth,
    activeFocusNodeIds,
    activeLayoutPreset,
    activePathEndpointIds,
    activePinnedNodes,
    activeRendererPreference,
    activeTheme,
    activeViewMode,
    displaySettings,
    enabledTypes,
    forceSettings,
    graphVersion,
    rendererStatus,
    onExport,
    scene,
  ]);

  exportPngImplRef.current = exportPng;
  exportJsonImplRef.current = exportJson;

  useEffect(() => {
    onExportReady?.(exportHandlersRef.current);
    return () => onExportReady?.(null);
  }, [onExportReady]);

  const selectedNode = nodeById.get(activeSelectedId ?? "");
  const hasGraph = Boolean((graphVersion || scene) && nodes.length);
  const isTruncated = scene?.truncated ?? preview?.truncated ?? false;
  const originalNodeCount =
    scene?.originalNodeCount ?? preview?.originalNodeCount ?? nodes.length;
  const originalEdgeCount =
    scene?.originalEdgeCount ?? preview?.originalEdgeCount ?? edges.length;
  const rendererFallbackWarning = graphRendererFallbackWarning(rendererStatus);
  const activeVisibleCounts = useMemo(() => {
    const countNodes = activeViewMode === "global" ? nodes : requestedSceneNodes;
    const countEdges = activeViewMode === "global" ? edges : requestedSceneEdges;
    const filtered = filterGraphFacts(
      { nodes: countNodes, edges: countEdges },
      activeFilters,
    );
    return {
      nodes: filtered.slice.nodes.length,
      edges: filtered.slice.edges.length,
    };
  }, [activeFilters, activeViewMode, edges, nodes, requestedSceneEdges, requestedSceneNodes]);
  const draftVisibleCounts = useMemo(() => {
    const countNodes = activeViewMode === "global" ? nodes : requestedSceneNodes;
    const countEdges = activeViewMode === "global" ? edges : requestedSceneEdges;
    const filtered = filterGraphFacts(
      { nodes: countNodes, edges: countEdges },
      draftFilters,
    );
    return {
      nodes: filtered.slice.nodes.length,
      edges: filtered.slice.edges.length,
      filters: filtered.filters,
    };
  }, [activeViewMode, draftFilters, edges, nodes, requestedSceneEdges, requestedSceneNodes]);

  const toggleDraftNodeType = (type: string) => {
    const current = draftFilters.nodeTypes.length ? new Set(draftFilters.nodeTypes) : new Set(types);
    if (current.has(type)) {
      if (current.size === 1) return;
      current.delete(type);
    } else current.add(type);
    setDraftFilters((value) => ({
      ...value,
      nodeTypes: current.size === types.length ? [] : types.filter((candidate) => current.has(candidate)),
    }));
  };

  const toggleDraftEdgeType = (type: string) => {
    const current = draftFilters.edgeTypes.length ? new Set(draftFilters.edgeTypes) : new Set(edgeTypes);
    if (current.has(type)) {
      if (current.size === 1) {
        setStatus("至少保留一种关系类型；如需恢复全部关系，请点击“显示全部”");
        return;
      }
      current.delete(type);
    }
    else current.add(type);
    setDraftFilters((value) => ({
      ...value,
      edgeTypes: current.size === edgeTypes.length ? [] : edgeTypes.filter((candidate) => current.has(candidate)),
    }));
  };

  const filterPopover = filterOpen ? (
    <div
      ref={filterPopoverRef}
      className={joinClassNames(
        "graph-preview__filter-portal",
        filterAsSheet && "graph-preview__filter-portal--sheet",
        activeTheme === "focus-dark" && "graph-preview__filter-portal--dark",
      )}
      style={filterPosition}
      role="dialog"
      aria-label="筛选节点类型"
    >
      <div className="graph-preview__filter-heading">
        <span>筛选图谱</span>
        <button
          type="button"
          onClick={() => setDraftFilters({ nodeTypes: [], edgeTypes: [] })}
          disabled={graphFilterConstraintCount(draftFilters) === 0}
        >
          显示全部
        </button>
        <button
          type="button"
          className="graph-preview__filter-close"
          onClick={() => closeFilter()}
          aria-label="关闭筛选"
        >
          <X size={15} aria-hidden="true" />
        </button>
      </div>
      <p className="graph-preview__filter-count" role="status">
        将显示 <strong>{draftVisibleCounts.nodes}</strong> 个节点 · <strong>{draftVisibleCounts.edges}</strong> 条关系
      </p>
      <strong className="graph-preview__filter-section-title">节点类型</strong>
      <div className="graph-preview__filter-options">
        {types.map((type) => {
          const enabled = draftFilters.nodeTypes.length === 0 || draftFilters.nodeTypes.includes(type);
          const count = fullNodes.filter((node) => nodeType(node) === type).length;
          return (
            <label key={type} className="graph-preview__filter-option">
              <input
                type="checkbox"
                checked={enabled}
                onChange={() => toggleDraftNodeType(type)}
              />
              <span
                className="graph-preview__legend-dot"
                style={{ backgroundColor: colourByType.get(type) }}
                aria-hidden="true"
              />
              <span>{typeLabel(type)}</span>
              <small>{count}</small>
              {enabled ? <Check size={14} aria-hidden="true" /> : null}
            </label>
          );
        })}
      </div>
      {edgeTypes.length ? (
        <>
          <strong className="graph-preview__filter-section-title">关系类型</strong>
          <div className="graph-preview__filter-options">
            {edgeTypes.map((type) => {
              const enabled = draftFilters.edgeTypes.length === 0 || draftFilters.edgeTypes.includes(type);
              const count = fullEdges.filter((edge) => edge.type === type).length;
              return (
                <label key={type} className="graph-preview__filter-option">
                  <input type="checkbox" checked={enabled} onChange={() => toggleDraftEdgeType(type)} />
                  <span className="graph-preview__legend-line" aria-hidden="true" />
                  <span>{typeLabel(type)}</span>
                  <small>{count}</small>
                  {enabled ? <Check size={14} aria-hidden="true" /> : null}
                </label>
              );
            })}
          </div>
        </>
      ) : null}
      <strong className="graph-preview__filter-section-title">时间范围</strong>
      <div className="graph-preview__filter-dates">
        <label>开始<input type="date" value={draftFilters.timeRange?.start ?? ""} onChange={(event) => setDraftFilters((value) => ({ ...value, timeRange: { ...value.timeRange, start: event.target.value || undefined } }))} /></label>
        <label>结束<input type="date" value={draftFilters.timeRange?.end ?? ""} onChange={(event) => setDraftFilters((value) => ({ ...value, timeRange: { ...value.timeRange, end: event.target.value || undefined } }))} /></label>
      </div>
      <div className="graph-preview__filter-actions">
        <button type="button" onClick={() => closeFilter()}>取消</button>
        <button
          type="button"
          className="is-primary"
          onClick={() => {
            const normalized = normalizeGraphFilters(draftFilters);
            commitFilters(normalized);
            closeFilter();
            setStatus(normalized.emptyReason
              ? "筛选条件无效，已安全显示空范围"
              : `筛选已应用：${draftVisibleCounts.nodes} 个节点，${draftVisibleCounts.edges} 条关系`);
          }}
        >
          应用筛选
        </button>
      </div>
    </div>
  ) : null;

  return (
    <>
    <section
      ref={rootRef}
        className={joinClassNames(
          "graph-preview",
          "graph-preview--workbench",
          nodes.length > 1_000 && "graph-preview--large",
          activeOverlay && "graph-preview--with-overlay",
        activeTheme === "focus-dark" && "graph-preview--focus-dark",
        isFullscreen && "graph-preview--fullscreen",
        className,
      )}
      aria-label={ariaLabel}
      data-graph-ready={engineReady}
      data-visible-nodes={runtimeMetricsRef.current.visibleNodes}
      data-visible-edges={runtimeMetricsRef.current.visibleEdges}
      data-engine-create-count={runtimeMetricsRef.current.engineCreateCount}
      data-engine-destroy-count={runtimeMetricsRef.current.engineDestroyCount}
      data-layout-count={runtimeMetricsRef.current.layoutCount}
      data-draw-count={runtimeMetricsRef.current.drawCount}
      data-fit-view-count={runtimeMetricsRef.current.fitViewCount}
      data-camera-x={lastCameraRef.current.x}
      data-camera-y={lastCameraRef.current.y}
      data-camera-zoom={lastCameraRef.current.zoom}
      data-renderer-requested={rendererStatus.requested}
      data-renderer-resolved={rendererStatus.resolved}
      data-renderer-fallback-reason={rendererStatus.fallbackReason}
      data-webgl-supported={rendererStatus.webglSupported}
      data-webgl-lazy-load-ms={rendererStatus.lazyLoadMs}
      data-webgl-context-loss-count={rendererStatus.contextLossCount}
      data-draggable-node-id={runtimeMetricsRef.current.dragTarget?.nodeId}
      data-draggable-x={runtimeMetricsRef.current.dragTarget?.x}
      data-draggable-y={runtimeMetricsRef.current.dragTarget?.y}
      data-local-force-frame-count={runtimeMetricsRef.current.localForceFrameCount}
      data-local-force-moved-node-count={runtimeMetricsRef.current.localForceMovedNodeCount}
      data-local-force-neighbor-delta-max={runtimeMetricsRef.current.localForceNeighborDeltaMax}
      data-local-force-settled-generation={runtimeMetricsRef.current.localForceSettledGeneration}
      data-appearance-request-key={appearanceRequestKey}
      data-rendered-label-count={renderedLabelCount}
      data-reference-label-count={Object.keys(overlay?.presentation?.referenceLabels ?? {}).length}
      data-review-decision-count={Object.keys(overlay?.presentation?.reviewDecisions ?? {}).length}
    >
      <header className={`graph-preview__header${headerAccessory ? " has-accessory" : ""}`}>
        <div className="graph-preview__title-group">
          <div className="graph-preview__title-line">
            <h2>{title}</h2>
            <span className="graph-preview__live-dot" aria-hidden="true" title="图谱已就绪" />
            {rendererFallbackWarning ? (
              <span
                className="graph-preview__renderer-warning"
                role="status"
                title={rendererFallbackWarning.title}
              >
                <WarningCircle size={13} weight="fill" aria-hidden="true" />
                {rendererFallbackWarning.label}
              </span>
            ) : null}
          </div>
          {isTruncated ? (
            <span
              className="graph-preview__scope-badge"
              title={`完整数据包含 ${originalNodeCount} 个节点和 ${originalEdgeCount} 条关系`}
            >
              <WarningCircle size={14} weight="regular" aria-hidden="true" />
              局部预览 · {nodes.length}/{originalNodeCount} 节点
            </span>
          ) : null}
        </div>
        <div className="graph-preview__header-actions">
          {overlay ? (
            <span className="graph-preview__overlay-badge" title={overlay.legend.title}>
              <GraphIcon size={14} aria-hidden="true" />
              {overlay.legend.title}
            </span>
          ) : null}
          <button
            type="button"
            className="graph-preview__icon-button"
            onClick={() =>
              changeTheme(
                activeTheme === "brand-light" ? "focus-dark" : "brand-light",
              )
            }
            aria-label={
              activeTheme === "brand-light"
                ? "切换到专注深色主题"
                : "切换到品牌浅色主题"
            }
            title={activeTheme === "brand-light" ? "深色图谱" : "浅色图谱"}
            disabled={!hasGraph}
          >
            {activeTheme === "brand-light" ? (
              <MoonStars size={18} />
            ) : (
              <Sun size={18} />
            )}
          </button>
          {onSummaryCollapsedChange ? (
            <button
              type="button"
              className="graph-preview__icon-button graph-preview__summary-toggle"
              onClick={() => onSummaryCollapsedChange(!summaryCollapsed)}
              aria-label={summaryCollapsed ? "展开图谱摘要" : "收起图谱摘要"}
              aria-expanded={!summaryCollapsed}
              aria-controls={summaryControlsId}
              title={summaryCollapsed ? "展开图谱摘要" : "收起图谱摘要"}
            >
              {summaryCollapsed ? (
                <CaretDown size={18} aria-hidden="true" />
              ) : (
                <CaretUp size={18} aria-hidden="true" />
              )}
            </button>
          ) : null}
          <details className="graph-preview__export">
            <summary
              className="graph-preview__icon-button"
              aria-label="导出图谱"
              aria-disabled={!hasGraph}
              title="导出"
            >
              <DownloadSimple size={18} aria-hidden="true" />
            </summary>
            <div className="graph-preview__export-menu">
              <button type="button" onClick={() => void exportPng()} disabled={!hasGraph}>
                <ImageSquare size={16} aria-hidden="true" />
                <span><strong>导出 PNG</strong><small>完整图谱画布</small></span>
              </button>
              <button type="button" onClick={exportJson} disabled={!hasGraph}>
                <BracketsCurly size={16} aria-hidden="true" />
                <span><strong>导出视图 JSON</strong><small>主题、筛选与视口</small></span>
              </button>
            </div>
          </details>
          <button
            type="button"
            className="graph-preview__icon-button"
            onClick={() => void toggleFullscreen()}
            aria-label={isFullscreen ? "退出图谱全屏" : "图谱全屏"}
            title={isFullscreen ? "退出全屏" : "全屏"}
            disabled={!hasGraph}
          >
            {isFullscreen ? <X size={18} /> : <ArrowsOut size={18} />}
          </button>
        </div>
        {headerAccessory ? (
          <div className="graph-preview__header-accessory">{headerAccessory}</div>
        ) : null}
      </header>

      <div className="graph-preview__controls">
        <form ref={searchFormRef} className="graph-preview__search" onSubmit={submitSearch} role="search">
          <MagnifyingGlass size={16} aria-hidden="true" />
          <input
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setSearchOpen(Boolean(event.target.value.trim()));
              setActiveSearchIndex(0);
              setSearchFeedback("");
              setPendingSearchNodeId(null);
            }}
            onFocus={() => setSearchOpen(Boolean(query.trim()))}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown" && searchMatches.length) {
                event.preventDefault();
                setActiveSearchIndex((index) => (index + 1) % searchMatches.length);
              } else if (event.key === "ArrowUp" && searchMatches.length) {
                event.preventDefault();
                setActiveSearchIndex(
                  (index) => (index - 1 + searchMatches.length) % searchMatches.length,
                );
              } else if (event.key === "Enter") {
                event.preventDefault();
                const match = searchMatches[Math.min(activeSearchIndex, searchMatches.length - 1)];
                if (match) chooseSearchResult(match);
                else {
                  setSearchFeedback(`未找到“${query.trim()}”`);
                  setSearchOpen(false);
                }
              }
              if (event.key === "Escape") {
                clearTransientFocus();
              }
            }}
            placeholder="搜索节点"
            aria-label="按名称或 ID 搜索节点"
            aria-autocomplete="list"
            aria-expanded={searchOpen}
            aria-controls="graph-node-search-results"
            aria-activedescendant={
              searchOpen && searchMatches[activeSearchIndex]
                ? `graph-search-option-${encodeURIComponent(searchMatches[activeSearchIndex].id)}`
                : undefined
            }
            disabled={!hasGraph}
          />
          {searchOpen ? (
            <div id="graph-node-search-results" className="graph-preview__search-results" role="listbox" aria-label="节点搜索结果">
              {searchMatches.length ? searchMatches.map((node, index) => (
                <button
                  key={node.id}
                  id={`graph-search-option-${encodeURIComponent(node.id)}`}
                  type="button"
                  role="option"
                  className={index === activeSearchIndex ? "is-active" : undefined}
                  aria-selected={index === activeSearchIndex}
                  onPointerMove={() => setActiveSearchIndex(index)}
                  onPointerDown={(event) => event.preventDefault()}
                  onClick={() => chooseSearchResult(node)}
                >
                  <span><strong>{node.label}</strong><small>{typeLabel(nodeType(node))} · {node.id}</small></span>
                  <small>{nodes.some((candidate) => candidate.id === node.id) ? "定位" : "载入关联场景"}</small>
                </button>
              )) : (
                <p>未找到匹配节点</p>
              )}
            </div>
          ) : null}
          {searchFeedback ? (
            <div className="graph-preview__search-feedback" role="status">
              <span>{searchFeedback}</span>
            </div>
          ) : null}
        </form>

        <div className="graph-preview__filter">
          <button
            ref={filterButtonRef}
            type="button"
            className="graph-preview__filter-trigger"
            aria-label={activeFilterConstraintCount > 0
              ? `筛选节点类型与关系，当前有 ${activeFilterConstraintCount} 项约束`
              : "筛选节点类型与关系，当前显示全部"}
            aria-expanded={filterOpen}
            aria-haspopup="dialog"
            disabled={!hasGraph}
            onClick={() => {
              if (filterOpen) {
                closeFilter(false);
                return;
              }
              setDraftFilters({
                ...activeFilters,
                nodeTypes: [...activeFilters.nodeTypes],
                edgeTypes: [...activeFilters.edgeTypes],
                ...(activeFilters.timeRange ? { timeRange: { ...activeFilters.timeRange } } : {}),
              });
              setFilterOpen(true);
              window.requestAnimationFrame(updateFilterPosition);
            }}
          >
            <Funnel size={16} aria-hidden="true" />
            <span>筛选</span>
            <strong>{activeFilterConstraintCount > 0 ? activeFilterConstraintCount : "全部"}</strong>
          </button>
        </div>

        <div className="graph-preview__zoom" aria-label="缩放与视图工具">
          <button
            type="button"
            onClick={() => zoom(1.22)}
            aria-label="放大图谱"
            title="放大"
            disabled={!hasGraph}
          >
            <Plus size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={() => zoom(0.82)}
            aria-label="缩小图谱"
            title="缩小"
            disabled={!hasGraph}
          >
            <Minus size={16} aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={fitView}
            aria-label="找回并适应图谱视图"
            title="找回图谱（Home / 0）"
            disabled={!hasGraph}
          >
            <Selection size={16} aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="graph-preview__stage">
        {viewportFeedback ? (
          <div className="graph-preview__viewport-feedback" role="status">
            <Check size={15} weight="bold" aria-hidden="true" />
            {viewportFeedback}
          </div>
        ) : null}
        {!graphVersion && !scene ? (
          <div className="graph-preview__empty" role="status">
            <span className="graph-preview__empty-icon">
              <Selection size={24} aria-hidden="true" />
            </span>
            <strong>{emptyState?.title ?? "等待图谱数据"}</strong>
            <span>{emptyState?.description ?? "上传 CSV 或 JSON 后将在这里生成真实预览"}</span>
          </div>
        ) : nodes.length === 0 ? (
          <div className="graph-preview__empty" role="status">
            <WarningCircle size={24} aria-hidden="true" />
            <strong>当前图谱没有可预览节点</strong>
            <span>请检查文件内容与字段映射</span>
          </div>
        ) : null}

        <div
          ref={viewportRef}
          className="graph-preview__viewport"
          style={{ backgroundColor: activeTheme === "focus-dark" ? "#081327" : "#f8fafc" }}
        >
          <div
            ref={canvasRef}
            className="graph-preview__canvas"
            style={{ backgroundColor: activeTheme === "focus-dark" ? "#081327" : "#f8fafc" }}
            role="img"
            aria-label={
              hasGraph
                ? `关系图，当前显示 ${activeVisibleCounts.nodes} 个节点和 ${activeVisibleCounts.edges} 条关系。可拖拽、缩放并点击节点。`
                : "空图谱画布"
            }
          />
        </div>

        {returnToOverviewAction ? (
          <button
            type="button"
            className="graph-preview__return-overview"
            onClick={returnToOverview}
          >
            <Crosshair size={16} aria-hidden="true" />
            {returnToOverviewAction.label}
          </button>
        ) : null}

        {isGraphOffscreen && hasGraph ? (
          <button type="button" className="graph-preview__recover" onClick={fitView}>
            <Crosshair size={16} aria-hidden="true" />
            图谱在视窗外 · 找回
          </button>
        ) : null}
        {enableMinimap ? (
          <div className="graph-preview__minimap" aria-hidden="true">
            <div ref={minimapRef} className="graph-preview__minimap-canvas" />
          </div>
        ) : null}

        {renderError ? (
          <div className="graph-preview__render-error" role="alert">
            <WarningCircle size={18} aria-hidden="true" />
            <span>{renderError}</span>
          </div>
        ) : null}

        {selectedNode ? (
          <aside className="graph-preview__selection" aria-label="已选节点详情">
            <span
              className="graph-preview__selection-colour"
              style={{ backgroundColor: colourByType.get(nodeType(selectedNode)) }}
              aria-hidden="true"
            />
            <span>
              <strong>{selectedNode.label}</strong>
              <small>
                {typeLabel(nodeType(selectedNode))} · 已显示 {renderedDegreeById.get(selectedNode.id) ?? 0} 条关系
                {(degreeById.get(selectedNode.id) ?? 0) > (renderedDegreeById.get(selectedNode.id) ?? 0)
                  ? ` · 另有 ${(degreeById.get(selectedNode.id) ?? 0) - (renderedDegreeById.get(selectedNode.id) ?? 0)} 条未展示`
                  : ""}
              </small>
            </span>
            <div className="graph-preview__selection-actions">
              <button
                type="button"
                className={activePinnedNodes[selectedNode.id] ? "is-active" : undefined}
                onClick={toggleSelectedPin}
                aria-label={
                  activePinnedNodes[selectedNode.id] ? "取消固定节点" : "固定节点"
                }
                title={activePinnedNodes[selectedNode.id] ? "取消固定" : "固定节点"}
              >
                {activePinnedNodes[selectedNode.id] ? (
                  <PushPinSlash size={14} aria-hidden="true" />
                ) : (
                  <PushPin size={14} aria-hidden="true" />
                )}
              </button>
              <button
                type="button"
                onClick={clearTransientFocus}
                aria-label="清除节点选择"
                title="关闭详情"
              >
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          </aside>
        ) : null}
      </div>

      <footer className="graph-preview__legend" aria-label="节点类型图例">
        {overlay
          ? overlay.legend.items.slice(0, 5).map((item, index) => {
              const relationKind = overlay.presentation?.governanceLens === "relations"
                ? item.value === "factual" ? "factual" : item.value === "candidate" ? "potential" : null
                : null;
              return (
                <span
                  className={joinClassNames(
                    "graph-preview__legend-item",
                    relationKind && "graph-preview__legend-relation",
                    relationKind === "potential" && "is-dashed",
                  )}
                  data-line-style={relationKind === "potential" ? "dashed" : relationKind === "factual" ? "solid" : undefined}
                  key={`${item.value}-${index}`}
                  {...(relationKind ? { "data-relation-kind": relationKind } : {})}
                >
                  <span
                    className={relationKind
                      ? joinClassNames("graph-preview__legend-edge", relationKind === "potential" && "is-dashed")
                      : "graph-preview__legend-dot"}
                    style={{
                      backgroundColor:
                        item.color ??
                        TYPE_COLOURS[hashText(item.value) % TYPE_COLOURS.length],
                    }}
                    aria-hidden="true"
                  />
                  {item.label}
                </span>
              );
            })
          : types.slice(0, 5).map((type) => (
          <button
            type="button"
            key={type}
            className={joinClassNames(
              "graph-preview__legend-item",
              !enabledTypes.has(type) && "is-muted",
            )}
            onClick={() => toggleType(type)}
            aria-pressed={enabledTypes.has(type)}
          >
            <span
              className="graph-preview__legend-dot"
              style={{ backgroundColor: colourByType.get(type) }}
              aria-hidden="true"
            />
            {typeLabel(type)}
          </button>
        ))}
        {!overlay && types.length > 5 ? (
          <span className="graph-preview__legend-more">+{types.length - 5}</span>
        ) : null}
        {edgeTypes.length ? edgeTypes
          .filter((type) => !overlay?.legend.items.some((item) => item.label === typeLabel(type)))
          .slice(0, 3).map((type) => (
          <span className="graph-preview__legend-item graph-preview__legend-relation" key={`edge-${type}`}>
            <span
              className="graph-preview__legend-edge"
              style={{ backgroundColor: TYPE_COLOURS[hashText(type) % TYPE_COLOURS.length] }}
              aria-hidden="true"
            />
            {typeLabel(type)}
          </span>
        )) : (
          <>
            <span className="graph-preview__legend-edge" aria-hidden="true" />
            <span className="graph-preview__legend-label">关系</span>
          </>
        )}
      </footer>

      <p className="sr-only" aria-live="polite" aria-atomic="true">
        {status}
      </p>
    </section>
    {filterPopover ? createPortal(filterPopover, document.body) : null}
    </>
  );
}

export default GraphPreview;
