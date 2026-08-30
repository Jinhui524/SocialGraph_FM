import type { ReactNode } from "react";
import type { GraphCameraSnapshot } from "../../services/graphEngineAdapter";
import type { GraphPerformanceSample } from "../../services/graphPerformanceProbe";
import type {
  AnalysisOverlay,
  GraphFilters,
  GovernanceFocus,
  GraphNode,
  GraphRendererPreference,
  GraphRendererStatus,
  GraphScene,
  GraphTheme,
  GraphVersion,
  GraphViewMode,
  LayoutPreset,
} from "../../types/graph";

export interface GraphPreviewProps {
  graphVersion: GraphVersion | null;
  /** Optional workbench scene. When supplied it replaces graphVersion.preview. */
  scene?: Pick<
    GraphScene,
    | "graphVersionId"
    | "nodes"
    | "edges"
    | "truncated"
    | "originalNodeCount"
    | "originalEdgeCount"
    | "focusNodeIds"
    | "pathNodeIds"
    | "pathEdgeIds"
    | "status"
    | "overlay"
  > | null;
  selectedNodeId?: string | null;
  /** Preferred callback name used by the research workspace. */
  onSelectNode?: (node: GraphNode | null) => void;
  /** Backwards-compatible alias for embedding the preview independently. */
  onNodeSelect?: (node: GraphNode | null) => void;
  className?: string;
  title?: string;
  /** Optional product context rendered as a full-width second header row. */
  headerAccessory?: ReactNode;
  ariaLabel?: string;
  /** Product-specific guidance for a graph pane whose ingestion path is not
   * the ordinary relationship-file composer. */
  emptyState?: { readonly title: string; readonly description: string };
  viewMode?: GraphViewMode;
  depth?: 1 | 2 | 3;
  theme?: GraphTheme;
  layoutPreset?: LayoutPreset;
  rendererPreference?: GraphRendererPreference;
  /** Caps ordinary priority labels; selected, focused, and path nodes remain labelled. */
  labelLimit?: number;
  /** Hides model/adapted rank prefixes while preserving node labels and focus. */
  showNodeRanks?: boolean;
  focusNodeIds?: readonly string[];
  pathEndpointIds?: readonly string[];
  pinnedNodes?: Readonly<Record<string, { readonly x: number; readonly y: number }>>;
  activeOverlay?: AnalysisOverlay | null;
  governanceFocus?: GovernanceFocus;
  /** Canonical view filters. Empty nodeTypes means all available node types. */
  filters?: GraphFilters;
  onViewModeChange?: (mode: GraphViewMode) => void;
  onDepthChange?: (depth: 1 | 2 | 3) => void;
  onThemeChange?: (theme: GraphTheme) => void;
  onLayoutPresetChange?: (preset: LayoutPreset) => void;
  onRendererPreferenceChange?: (preference: GraphRendererPreference) => void;
  onRendererStatus?: (status: GraphRendererStatus) => void;
  onFocusNodeIdsChange?: (nodeIds: readonly string[]) => void;
  onPathEndpointIdsChange?: (nodeIds: readonly string[]) => void;
  onPinnedNodesChange?: (
    nodes: Readonly<Record<string, { readonly x: number; readonly y: number }>>,
  ) => void;
  onFiltersChange?: (filters: GraphFilters) => void;
  onViewStateChange?: (state: GraphPreviewViewSnapshot) => void;
  /** Restores an explicitly captured camera without rebuilding the graph or layout. */
  cameraRestoreCommand?: (
    | (GraphCameraSnapshot & { readonly token: number; readonly commandScope?: string })
    | { readonly x: number; readonly y: number; readonly zoom: number; readonly token: number; readonly commandScope?: string }
  );
  /** Focuses a bounded external selection while preserving the stable graph layout. */
  cameraFocusCommand?: { readonly nodeIds: readonly string[]; readonly anchorNodeId?: string; readonly token: number; readonly projectionIdentity?: string; readonly commandScope?: string };
  onExportReady?: (handlers: GraphPreviewExportHandlers | null) => void;
  onExport?: (result: { format: "png" | "json"; fileName: string }) => void;
  /** Read-only diagnostics used by the local performance benchmark. */
  onRuntimeMetrics?: (metrics: GraphPreviewRuntimeMetrics) => void;
  /** Renderer-backed appearance diagnostics for focused integration and benchmark checks. */
  onAppearanceApplied?: (snapshot: GraphPreviewAppearanceSnapshot) => void;
  /** Diagnostic-only overview. Production workspaces leave this disabled. */
  enableMinimap?: boolean;
  /** Controlled graph-summary disclosure rendered in the preview header. */
  summaryCollapsed?: boolean;
  summaryControlsId?: string;
  onSummaryCollapsedChange?: (collapsed: boolean) => void;
  /** Signals that a responsive parent has made this pane visible again. */
  isPaneVisible?: boolean;
  /** Visible escape hatch owned by an external workspace projection. */
  returnToOverviewAction?: { readonly label: string; readonly onReturn: () => void };
}
export interface GraphPreviewAppearanceSnapshot {
  readonly graphVersionId: string;
  readonly focusCameraToken?: number;
  readonly nodeStyles: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
  readonly edgeStyles: Readonly<Record<string, Readonly<Record<string, unknown>>>>;
}

export interface GraphPreviewRuntimeMetrics {
  readonly ready: boolean;
  readonly visibleNodes: number;
  readonly visibleEdges: number;
  readonly engineCreateCount: number;
  readonly engineDestroyCount: number;
  readonly layoutCount: number;
  readonly drawCount: number;
  readonly fitViewCount: number;
  readonly rendererRequested: GraphRendererPreference;
  readonly rendererResolved: GraphRendererStatus["resolved"];
  readonly rendererFallbackReason?: string;
  readonly webglSupported: boolean;
  readonly webglLazyLoadMs?: number;
  readonly webglContextLossCount: number;
  readonly dragTarget?: {
    readonly nodeId: string;
    /** X relative to the graph canvas viewport. */
    readonly x: number;
    /** Y relative to the graph canvas viewport. */
    readonly y: number;
  };
  /** Last completed drag, exposed for deterministic browser QA. */
  readonly lastDraggedNodeId?: string;
  readonly lastDragPinned?: boolean;
  /** Read-only browser QA counters for bounded local-force interaction. */
  readonly localForceFrameCount: number;
  readonly localForceMovedNodeCount: number;
  readonly localForceNeighborDeltaMax: number;
  readonly localForceSettledGeneration: number;
  readonly spatialPickMs?: number;
  readonly spatialPickCandidates?: number;
  readonly pickOracleChecked?: number;
  readonly pickOracleMismatches?: number;
  readonly pickOracleP95Ms?: number;
  readonly pickOracleCandidatesP95?: number;
  readonly workerRoundTripMs?: number;
  readonly workerComputeMs?: number;
  readonly positionApplyMs?: number;
  readonly mutationInFlight: number;
  readonly mutationInFlightMax: number;
  readonly viewportCullingPaused: boolean;
  readonly interactionLodActive: boolean;
  readonly culledNodes?: number;
  readonly culledEdges?: number;
  readonly performanceSamples?: readonly GraphPerformanceSample[];
}

export interface GraphPreviewExportHandlers {
  exportPng(): Promise<void>;
  exportJson(): void;
}

export interface GraphPreviewViewSnapshot {
  readonly graphVersionId?: string;
  readonly mode: GraphViewMode;
  readonly depth: 1 | 2 | 3;
  readonly theme: GraphTheme;
  readonly layoutPreset: LayoutPreset;
  readonly rendererPreference: GraphRendererPreference;
  readonly focusNodeIds: readonly string[];
  readonly pathEndpointIds: readonly string[];
  readonly pinnedNodes: Readonly<
    Record<string, { readonly x: number; readonly y: number }>
  >;
  readonly camera: GraphCameraSnapshot;
}

export function completeCameraSnapshot(
  sceneIdentity: string,
  camera: { readonly x: number; readonly y: number; readonly zoom: number },
  worldCenter: { readonly x: number; readonly y: number } | null,
  viewportSize: readonly [number, number],
): GraphCameraSnapshot {
  return Object.freeze({
    schemaVersion: "socialgraph-fm.graph-camera/2" as const,
    sceneIdentity,
    position: Object.freeze([camera.x, camera.y]) as [number, number],
    zoom: camera.zoom,
    worldCenter: Object.freeze([worldCenter?.x ?? 0, worldCenter?.y ?? 0]) as [number, number],
    viewportSize: Object.freeze([...viewportSize]) as [number, number],
  });
}

export function viewSnapshotKey(snapshot: GraphPreviewViewSnapshot): string {
  return JSON.stringify({
    ...snapshot,
    camera: {
      ...snapshot.camera,
      position: [
        Math.round(Number(snapshot.camera.position[0]) * 10) / 10,
        Math.round(Number(snapshot.camera.position[1]) * 10) / 10,
      ],
      zoom: Math.round(snapshot.camera.zoom * 1_000) / 1_000,
    },
  });
}
