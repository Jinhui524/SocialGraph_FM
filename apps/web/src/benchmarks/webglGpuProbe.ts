export type GpuProbeMode = "off" | "gpu";

interface NativeCallSample {
  readonly kind: "context" | "bufferData" | "bufferSubData";
  readonly durationMs: number;
  readonly bytes: number;
  readonly canvas: HTMLCanvasElement;
  readonly contextType?: string;
  readonly timestamp: number;
}

export interface WebglGpuProbeSnapshot {
  readonly contextCreateMs: number;
  readonly contextCalls: number;
  readonly bufferUploadMs: number;
  readonly bufferCalls: number;
  readonly bufferBytes: number;
}

declare global {
  interface Window {
    __SGFM_GPU_PROBE__?: {
      readonly samples: NativeCallSample[];
      snapshot(root: Element, since?: number): WebglGpuProbeSnapshot;
    };
  }
}

const bytesOf = (value: unknown): number => {
  if (typeof value === "number") return Math.max(0, value);
  if (value instanceof ArrayBuffer) return value.byteLength;
  if (ArrayBuffer.isView(value)) return value.byteLength;
  return 0;
};

export function requestedGpuProbeMode(search: string): GpuProbeMode {
  return new URLSearchParams(search).get("probe") === "gpu" ? "gpu" : "off";
}

export function installWebglGpuProbe(): void {
  if (window.__SGFM_GPU_PROBE__) return;
  const samples: NativeCallSample[] = [];
  const contexts = new WeakMap<object, HTMLCanvasElement>();
  const originalGetContext = HTMLCanvasElement.prototype.getContext;
  let bufferCalls = 0;
  let bufferBytes = 0;
  let bufferUploadMs = 0;

  const emitProgress = () => {
    // This path is enabled only for probe=gpu. Console delivery lets the Node
    // runner retain native attribution even if the renderer becomes
    // unresponsive before the React readiness flag can be published.
    console.info(`__SGFM_GPU_PROBE_SAMPLE__${JSON.stringify({
      contextCalls: samples.filter((sample) => sample.kind === "context").length,
      contextCreateMs: samples
        .filter((sample) => sample.kind === "context")
        .reduce((sum, sample) => sum + sample.durationMs, 0),
      bufferCalls,
      bufferBytes,
      bufferUploadMs,
    })}`);
  };

  HTMLCanvasElement.prototype.getContext = function patchedGetContext(
    this: HTMLCanvasElement,
    contextId: string,
    options?: unknown,
  ) {
    const startedAt = performance.now();
    const result = Reflect.apply(
      originalGetContext as unknown as (...args: unknown[]) => unknown,
      this,
      options === undefined ? [contextId] : [contextId, options],
    );
    const durationMs = performance.now() - startedAt;
    if ((contextId === "webgl" || contextId === "webgl2") && result) {
      contexts.set(result as object, this);
      samples.push({
        kind: "context",
        durationMs,
        bytes: 0,
        canvas: this,
        contextType: contextId,
        timestamp: performance.now(),
      });
      emitProgress();
    }
    return result as never;
  } as typeof HTMLCanvasElement.prototype.getContext;

  const patchPrototype = (prototype: object) => {
    const target = prototype as {
      bufferData: (...args: unknown[]) => unknown;
      bufferSubData: (...args: unknown[]) => unknown;
    };
    for (const methodName of ["bufferData", "bufferSubData"] as const) {
      const original = target[methodName];
      if (typeof original !== "function") continue;
      target[methodName] = function patchedBuffer(this: object, ...args: unknown[]) {
        const startedAt = performance.now();
        const result = original.apply(this, args);
        const canvas = contexts.get(this);
        if (canvas) {
          const dataIndex = methodName === "bufferData" ? 1 : 2;
          samples.push({
            kind: methodName,
            durationMs: performance.now() - startedAt,
            bytes: bytesOf(args[dataIndex]),
            canvas,
            timestamp: performance.now(),
          });
          bufferCalls += 1;
          bufferBytes += bytesOf(args[dataIndex]);
          bufferUploadMs += performance.now() - startedAt;
          if (bufferCalls % 100 === 0) emitProgress();
        }
        return result;
      };
    }
  };

  patchPrototype(WebGLRenderingContext.prototype);
  if (typeof WebGL2RenderingContext !== "undefined") patchPrototype(WebGL2RenderingContext.prototype);

  window.__SGFM_GPU_PROBE__ = {
    samples,
    snapshot(root, since = 0) {
      const relevant = samples.filter((sample) => sample.timestamp >= since && root.contains(sample.canvas));
      const context = relevant.filter((sample) => sample.kind === "context");
      const buffers = relevant.filter((sample) => sample.kind !== "context");
      return {
        contextCreateMs: context.reduce((sum, sample) => sum + sample.durationMs, 0),
        contextCalls: context.length,
        bufferUploadMs: buffers.reduce((sum, sample) => sum + sample.durationMs, 0),
        bufferCalls: buffers.length,
        bufferBytes: buffers.reduce((sum, sample) => sum + sample.bytes, 0),
      };
    },
  };
}
