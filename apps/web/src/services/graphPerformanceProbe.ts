import { GraphEvent, type Graph } from "@antv/g6";
import type { GraphRendererKind } from "../types/graph";

export type GraphPerformancePhase =
  | "renderer_import"
  | "renderer_construct"
  | "canvas_init"
  | "webgl_context_create"
  | "initial_buffer_upload"
  | "node_buffer_patch"
  | "edge_buffer_patch"
  | "first_draw"
  | "first_present"
  | "spatial_pick"
  | "layer_sync"
  | "worker_compute"
  | "worker_round_trip"
  | "position_apply"
  | "viewport_cull"
  | "pointer_dispatch"
  | "coordinate_transform"
  | "drag_target_apply"
  | "drag_neighbour_apply"
  | "visibility_compute"
  | "visibility_apply"
  | "canvas_draw"
  | "canvas_present";

export interface GraphPerformanceSample {
  readonly phase: GraphPerformancePhase;
  readonly durationMs: number;
  readonly graphEpoch: string;
  readonly renderer: GraphRendererKind;
  readonly calls?: number;
  readonly bytes?: number;
  readonly count?: number;
  readonly timestamp: number;
  readonly detail?: Readonly<Record<string, number | string | boolean>>;
}

export interface GraphPerformanceRecord {
  readonly calls?: number;
  readonly bytes?: number;
  readonly count?: number;
  readonly detail?: Readonly<Record<string, number | string | boolean>>;
}

type Clock = () => number;

/** Low-overhead lifecycle probe. Native WebGL calls are benchmark-only. */
export class GraphPerformanceProbe {
  static readonly maxSamples = 256;
  static readonly maxSamplesPerPhase = 32;
  private readonly samples: GraphPerformanceSample[] = [];
  private readonly starts = new Map<GraphPerformancePhase, number>();
  private readonly detachCallbacks = new Set<() => void>();

  constructor(
    private graphEpoch: string,
    private renderer: GraphRendererKind,
    private readonly clock: Clock = () => performance.now(),
  ) {}

  setContext(graphEpoch: string, renderer: GraphRendererKind): void {
    this.graphEpoch = graphEpoch;
    this.renderer = renderer;
  }

  begin(phase: GraphPerformancePhase): void {
    this.starts.set(phase, this.clock());
  }

  end(phase: GraphPerformancePhase, record: GraphPerformanceRecord = {}): GraphPerformanceSample | null {
    const startedAt = this.starts.get(phase);
    if (startedAt === undefined) return null;
    this.starts.delete(phase);
    return this.record(phase, this.clock() - startedAt, record);
  }

  record(
    phase: GraphPerformancePhase,
    durationMs: number,
    record: GraphPerformanceRecord = {},
  ): GraphPerformanceSample {
    const sample: GraphPerformanceSample = Object.freeze({
      phase,
      durationMs: Math.max(0, durationMs),
      graphEpoch: this.graphEpoch,
      renderer: this.renderer,
      timestamp: this.clock(),
      ...record,
    });
    const phaseSamples = this.samples.reduce(
      (count, current) => count + (current.phase === phase ? 1 : 0),
      0,
    );
    if (phaseSamples >= GraphPerformanceProbe.maxSamplesPerPhase) {
      const oldestPhaseSample = this.samples.findIndex((current) => current.phase === phase);
      if (oldestPhaseSample >= 0) this.samples.splice(oldestPhaseSample, 1);
    }
    this.samples.push(sample);
    if (this.samples.length > GraphPerformanceProbe.maxSamples) {
      this.samples.splice(0, this.samples.length - GraphPerformanceProbe.maxSamples);
    }
    return sample;
  }

  async measure<T>(
    phase: GraphPerformancePhase,
    action: () => T | Promise<T>,
    record: GraphPerformanceRecord = {},
  ): Promise<T> {
    const startedAt = this.clock();
    try {
      return await action();
    } finally {
      this.record(phase, this.clock() - startedAt, record);
    }
  }

  attach(graph: Graph): () => void {
    let firstDrawStartedAt: number | null = null;
    let firstDrawRecorded = false;
    let firstPresentRecorded = false;
    let attachLayerListeners = () => {};
    const schedulePresent = (startedAt: number, renderFinishedAt: number) => {
      requestAnimationFrame(() => requestAnimationFrame(() => {
        const durationMs = this.clock() - startedAt;
        this.record("canvas_present", durationMs, {
          detail: { renderSubmitMs: renderFinishedAt - startedAt },
        });
        if (!firstPresentRecorded) {
          firstPresentRecorded = true;
          this.record("first_present", durationMs, {
            detail: { renderSubmitMs: renderFinishedAt - startedAt },
          });
        }
      }));
    };
    const beforeDraw = () => {
      firstDrawStartedAt = this.clock();
    };
    const afterDraw = () => {
      if (firstDrawStartedAt === null) return;
      const durationMs = this.clock() - firstDrawStartedAt;
      if (!firstDrawRecorded) {
        firstDrawRecorded = true;
        this.record("first_draw", durationMs);
      }
      this.record("canvas_draw", durationMs);
      schedulePresent(firstDrawStartedAt, this.clock());
      firstDrawStartedAt = null;
      attachLayerListeners();
    };
    const beforeRender = () => {};
    const afterRender = () => attachLayerListeners();
    graph.on(GraphEvent.BEFORE_DRAW, beforeDraw);
    graph.on(GraphEvent.AFTER_DRAW, afterDraw);
    graph.on(GraphEvent.BEFORE_RENDER, beforeRender);
    graph.on(GraphEvent.AFTER_RENDER, afterRender);

    const layerListeners: Array<{
      layer: {
        addEventListener(type: string, listener: EventListener): unknown;
        removeEventListener(type: string, listener: EventListener): unknown;
      };
      listener: EventListener;
    }> = [];
    const layerTimes = new Map<string, number>();
    let layerFlushPending = false;
    let layersAttached = false;
    attachLayerListeners = () => {
      if (layersAttached) return;
      try {
      const layers = graph.getCanvas().getLayers();
      const flushLayerTimes = () => {
        layerFlushPending = false;
        const active = Object.entries(layers).filter(([, layer]) => layer.getStats().total > 0);
        const times = active
          .map(([name]) => layerTimes.get(name))
          .filter((value): value is number => value !== undefined);
        if (times.length > 0) {
          const matrices = active.map(([, layer]) => Array.from(layer.getCamera().getOrthoMatrix()));
          const reference = matrices[0] ?? [];
          const cameraMatrixDrift = matrices.reduce((maximum, matrix) => (
            Math.max(maximum, ...matrix.map((value, index) => Math.abs(value - (reference[index] ?? value))))
          ), 0);
          this.record("layer_sync", Math.max(...times) - Math.min(...times), {
            count: times.length,
            detail: { cameraMatrixDrift },
          });
        }
        layerTimes.clear();
      };
      for (const [name, layer] of Object.entries(layers)) {
        const listener: EventListener = () => {
          layerTimes.set(name, this.clock());
          if (!layerFlushPending) {
            layerFlushPending = true;
            requestAnimationFrame(flushLayerTimes);
          }
        };
        layer.addEventListener("afterrender", listener);
        layerListeners.push({ layer, listener });
      }
        layersAttached = layerListeners.length > 0;
      } catch {
        // Layers may not exist until the first completed draw.
      }
    };
    attachLayerListeners();
    const detach = () => {
      graph.off(GraphEvent.BEFORE_DRAW, beforeDraw);
      graph.off(GraphEvent.AFTER_DRAW, afterDraw);
      graph.off(GraphEvent.BEFORE_RENDER, beforeRender);
      graph.off(GraphEvent.AFTER_RENDER, afterRender);
      for (const { layer, listener } of layerListeners) {
        layer.removeEventListener("afterrender", listener);
      }
      this.detachCallbacks.delete(detach);
    };
    this.detachCallbacks.add(detach);
    return detach;
  }

  snapshot(): readonly GraphPerformanceSample[] {
    return this.samples.slice();
  }

  clear(): void {
    this.samples.length = 0;
    this.starts.clear();
  }

  dispose(): void {
    for (const detach of [...this.detachCallbacks]) detach();
    this.starts.clear();
  }
}
