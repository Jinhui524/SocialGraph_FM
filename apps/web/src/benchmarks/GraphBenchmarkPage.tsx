import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { GraphPreview, type GraphPreviewRuntimeMetrics } from "../components/GraphPreview";
import type { GraphNode } from "../types/graph";
import {
  createBenchmarkGraphVersion,
  createBenchmarkScene,
  GRAPH_BENCHMARK_CASES,
  type GraphBenchmarkCaseId,
} from "./graphBenchmarkData";
import {
  requestedGraphBenchmarkRenderer,
  type GraphBenchmarkRenderer,
  type GraphBenchmarkResolvedRenderer,
} from "./graphBenchmarkRenderer";
import { requestedGpuProbeMode, type GpuProbeMode, type WebglGpuProbeSnapshot } from "./webglGpuProbe";
import type { GraphPerformanceSample } from "../services/graphPerformanceProbe";
import "./graphBenchmarkPage.css";

interface RendererRuntimeMetrics {
  readonly rendererRequested?: GraphBenchmarkRenderer;
  readonly rendererResolved?: GraphBenchmarkResolvedRenderer;
  /** Preferred cross-component metric names. */
  readonly fallbackReason?: string;
  readonly lazyLoadMs?: number;
  /** Current GraphPreview aliases retained during the renderer PoC merge. */
  readonly rendererFallbackReason?: string;
  readonly webglLazyLoadMs?: number;
  readonly webglContextLossCount?: number;
  readonly spatialPickMs?: number;
  readonly spatialPickCandidates?: number;
  readonly pickOracleChecked?: number;
  readonly pickOracleMismatches?: number;
  readonly pickOracleP95Ms?: number;
  readonly pickOracleCandidatesP95?: number;
  readonly workerRoundTripMs?: number;
  readonly workerComputeMs?: number;
  readonly positionApplyMs?: number;
  readonly mutationInFlight?: number;
  readonly mutationInFlightMax?: number;
  readonly viewportCullingPaused?: boolean;
  readonly interactionLodActive?: boolean;
  readonly culledNodes?: number;
  readonly culledEdges?: number;
  readonly performanceSamples?: readonly GraphPerformanceSample[];
}

export interface GraphBenchmarkRuntime {
  caseId: GraphBenchmarkCaseId;
  seed: string;
  rendererRequested: GraphBenchmarkRenderer;
  rendererResolved?: GraphBenchmarkResolvedRenderer;
  rendererFallbackReason?: string;
  webglContextLossCount: number;
  rendererLazyLoadMs?: number;
  probeMode: GpuProbeMode;
  gpuInitial?: WebglGpuProbeSnapshot;
  performanceSamples: readonly GraphPerformanceSample[];
  spatialPickMs?: number;
  spatialPickCandidates?: number;
  pickOracleChecked?: number;
  pickOracleMismatches?: number;
  pickOracleP95Ms?: number;
  pickOracleCandidatesP95?: number;
  workerRoundTripMs?: number;
  workerComputeMs?: number;
  positionApplyMs?: number;
  mutationInFlight: number;
  mutationInFlightMax: number;
  viewportCullingPaused: boolean;
  interactionLodActive: boolean;
  culledNodes?: number;
  culledEdges?: number;
  sceneBuildMs: number;
  pageStartedAt: number;
  readyAt?: number;
  initialReadyMs?: number;
  selectedNodeId: string | null;
  visibleNodes: number;
  visibleEdges: number;
  graphCreateCount: number;
  graphDestroyCount: number;
  layoutCount: number;
  fitViewCount: number;
  drawCount: number;
  dragTarget?: { x: number; y: number; nodeId: string };
  lastDraggedNodeId?: string;
  lastDragPinned?: boolean;
  errors: string[];
}

declare global {
  interface Window {
    __SGFM_GRAPH_BENCHMARK__?: GraphBenchmarkRuntime;
  }
}

function requestedCase(): GraphBenchmarkCaseId {
  const value = new URLSearchParams(window.location.search).get("case");
  return value === "small" || value === "medium" || value === "large" ? value : "large";
}

export function GraphBenchmarkPage() {
  const pageStartedAt = useRef(performance.now());
  const caseId = requestedCase();
  const rendererPreference = requestedGraphBenchmarkRenderer(window.location.search);
  const probeMode = requestedGpuProbeMode(window.location.search);
  const [{ graphVersion, scene, sceneBuildMs }] = useState(() => {
    const graphVersion = createBenchmarkGraphVersion(GRAPH_BENCHMARK_CASES[caseId]);
    const startedAt = performance.now();
    const scene = createBenchmarkScene(graphVersion);
    return { graphVersion, scene, sceneBuildMs: performance.now() - startedAt };
  });
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const benchmarkRuntime = useMemo<GraphBenchmarkRuntime>(() => ({
    caseId,
    seed: GRAPH_BENCHMARK_CASES[caseId].seed,
    rendererRequested: rendererPreference,
    probeMode,
    performanceSamples: [],
    // The pre-PoC renderer is known to be Canvas. Other requested modes must
    // publish their actual resolution through GraphPreview runtime metrics.
    rendererResolved: rendererPreference === "canvas" ? "canvas" : undefined,
    webglContextLossCount: 0,
    mutationInFlight: 0,
    mutationInFlightMax: 0,
    viewportCullingPaused: false,
    interactionLodActive: false,
    sceneBuildMs,
    pageStartedAt: pageStartedAt.current,
    selectedNodeId: null,
    visibleNodes: scene.visibleNodeCount,
    visibleEdges: scene.visibleEdgeCount,
    graphCreateCount: 0,
    graphDestroyCount: 0,
    layoutCount: 0,
    fitViewCount: 0,
    drawCount: 0,
    errors: [],
  }), [caseId, probeMode, rendererPreference, scene.visibleEdgeCount, scene.visibleNodeCount, sceneBuildMs]);

  useEffect(() => {
    window.__SGFM_GRAPH_BENCHMARK__ = benchmarkRuntime;
    return () => {
      delete window.__SGFM_GRAPH_BENCHMARK__;
    };
  }, [benchmarkRuntime]);

  const handleRuntimeMetrics = useCallback((metrics: GraphPreviewRuntimeMetrics) => {
    const rendererMetrics = metrics as GraphPreviewRuntimeMetrics & RendererRuntimeMetrics;
    benchmarkRuntime.visibleNodes = metrics.visibleNodes;
    benchmarkRuntime.visibleEdges = metrics.visibleEdges;
    benchmarkRuntime.graphCreateCount = metrics.engineCreateCount;
    benchmarkRuntime.graphDestroyCount = metrics.engineDestroyCount;
    benchmarkRuntime.layoutCount = metrics.layoutCount;
    benchmarkRuntime.fitViewCount = metrics.fitViewCount;
    benchmarkRuntime.drawCount = metrics.drawCount;
    benchmarkRuntime.dragTarget = metrics.dragTarget;
    benchmarkRuntime.lastDraggedNodeId = metrics.lastDraggedNodeId;
    benchmarkRuntime.lastDragPinned = metrics.lastDragPinned;
    benchmarkRuntime.rendererRequested = rendererMetrics.rendererRequested ?? rendererPreference;
    benchmarkRuntime.rendererResolved = rendererMetrics.rendererResolved ?? benchmarkRuntime.rendererResolved;
    benchmarkRuntime.rendererFallbackReason =
      rendererMetrics.fallbackReason ?? rendererMetrics.rendererFallbackReason;
    benchmarkRuntime.webglContextLossCount = rendererMetrics.webglContextLossCount ?? 0;
    benchmarkRuntime.rendererLazyLoadMs =
      rendererMetrics.lazyLoadMs ?? rendererMetrics.webglLazyLoadMs;
    benchmarkRuntime.spatialPickMs = rendererMetrics.spatialPickMs;
    benchmarkRuntime.spatialPickCandidates = rendererMetrics.spatialPickCandidates;
    benchmarkRuntime.pickOracleChecked = rendererMetrics.pickOracleChecked;
    benchmarkRuntime.pickOracleMismatches = rendererMetrics.pickOracleMismatches;
    benchmarkRuntime.pickOracleP95Ms = rendererMetrics.pickOracleP95Ms;
    benchmarkRuntime.pickOracleCandidatesP95 = rendererMetrics.pickOracleCandidatesP95;
    benchmarkRuntime.workerRoundTripMs = rendererMetrics.workerRoundTripMs;
    benchmarkRuntime.workerComputeMs = rendererMetrics.workerComputeMs;
    benchmarkRuntime.positionApplyMs = rendererMetrics.positionApplyMs;
    benchmarkRuntime.mutationInFlight = rendererMetrics.mutationInFlight ?? 0;
    benchmarkRuntime.mutationInFlightMax = Math.max(
      benchmarkRuntime.mutationInFlightMax,
      rendererMetrics.mutationInFlightMax ?? 0,
    );
    benchmarkRuntime.viewportCullingPaused = rendererMetrics.viewportCullingPaused ?? false;
    benchmarkRuntime.interactionLodActive = rendererMetrics.interactionLodActive ?? false;
    benchmarkRuntime.culledNodes = rendererMetrics.culledNodes;
    benchmarkRuntime.culledEdges = rendererMetrics.culledEdges;
    benchmarkRuntime.performanceSamples = rendererMetrics.performanceSamples ?? benchmarkRuntime.performanceSamples;
    if (metrics.ready && benchmarkRuntime.readyAt === undefined) {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        benchmarkRuntime.readyAt = performance.now();
        benchmarkRuntime.initialReadyMs = benchmarkRuntime.readyAt - benchmarkRuntime.pageStartedAt;
        const root = document.querySelector(".graph-preview__canvas");
        if (root && window.__SGFM_GPU_PROBE__) {
          benchmarkRuntime.gpuInitial = window.__SGFM_GPU_PROBE__.snapshot(root);
        }
        document.documentElement.dataset.graphBenchmarkReady = "true";
        console.info("__SGFM_BENCHMARK_READY__");
      }));
    }
  }, [benchmarkRuntime, rendererPreference]);

  useEffect(() => {
    benchmarkRuntime.selectedNodeId = selectedNode?.id ?? null;
  }, [benchmarkRuntime, selectedNode]);

  return (
    <main
      className="graph-benchmark-page"
      data-benchmark-case={caseId}
      data-benchmark-renderer={rendererPreference}
    >
      <header className="graph-benchmark-header">
        <div>
          <strong>SocialGraph-FM 图谱性能基准</strong>
          <span>
            {graphVersion.nodes.length.toLocaleString()} 节点 · {graphVersion.edges.length.toLocaleString()} 边
            {` · ${rendererPreference} · probe=${probeMode}`}
          </span>
        </div>
        <output aria-live="polite" data-testid="benchmark-selection">
          {selectedNode ? `已选择 ${selectedNode.label}` : "等待交互"}
        </output>
      </header>
      <GraphPreview
        {...{ rendererPreference }}
        graphVersion={graphVersion}
        scene={scene}
        selectedNodeId={selectedNode?.id ?? null}
        onSelectNode={setSelectedNode}
        title={`${caseId} 基准图`}
        ariaLabel={`${caseId} 图谱交互性能基准`}
        enableMinimap={new URLSearchParams(window.location.search).get("minimap") === "1"}
        onRuntimeMetrics={handleRuntimeMetrics}
      />
    </main>
  );
}
