import { Renderer as CanvasRenderer } from "@antv/g-canvas";
import type { GraphOptions } from "@antv/g6";
import type {
  GraphRendererKind,
  GraphRendererPreference,
  GraphRendererStatus,
} from "../types/graph";
import rendererPolicy from "../config/graph-renderer-policy.json";

type G6RendererFactory = NonNullable<GraphOptions["renderer"]>;
type CanvasLayer = Parameters<G6RendererFactory>[0];
type WebGLModule = typeof import("@antv/g-webgl");

export interface GraphRendererLoadOptions {
  readonly preference: GraphRendererPreference;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly forcedCanvasReason?: string;
  readonly contextLossCount?: number;
  readonly onRuntimeFailure?: (reason: string) => void;
  readonly webglSupported?: boolean;
  readonly loadWebGL?: () => Promise<WebGLModule>;
  readonly now?: () => number;
}

export interface LoadedGraphRenderer {
  readonly renderer: G6RendererFactory;
  readonly status: GraphRendererStatus;
}

/**
 * Hybrid WebGL remains an opt-in proof of concept until the medium and large
 * benchmark matrix passes. Keeping this false makes the persisted `auto`
 * preference a safe production default while still allowing an explicit
 * `hybrid-webgl` A/B run from the settings panel and benchmark route.
 */
export const HYBRID_WEBGL_AUTO_PROMOTED = rendererPolicy.autoWebglEnabled === true;

export function detectWebGLSupport(documentLike: Pick<Document, "createElement"> = document): boolean {
  try {
    const canvas = documentLike.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

export function resolveGraphRendererKind(
  preference: GraphRendererPreference,
  nodeCount: number,
  edgeCount: number,
  webglSupported: boolean,
): GraphRendererKind {
  if (preference === "canvas") return "canvas";
  if (!webglSupported) return "canvas";
  if (preference === "hybrid-webgl") return "hybrid-webgl";
  if (!HYBRID_WEBGL_AUTO_PROMOTED) return "canvas";
  return nodeCount <= 300 && edgeCount <= 1_000 ? "canvas" : "hybrid-webgl";
}

function createCanvasRendererFactory(): G6RendererFactory {
  const layers = new Map<CanvasLayer, CanvasRenderer>();
  return (layer) => {
    const existing = layers.get(layer);
    if (existing) return existing;
    const renderer = new CanvasRenderer();
    layers.set(layer, renderer);
    return renderer;
  };
}

function canvasResult(
  requested: GraphRendererPreference,
  webglSupported: boolean,
  fallbackReason: string | undefined,
  contextLossCount: number,
  lazyLoadMs?: number,
): LoadedGraphRenderer {
  return {
    renderer: createCanvasRendererFactory(),
    status: {
      requested,
      resolved: "canvas",
      webglSupported,
      fallbackReason,
      lazyLoadMs,
      contextLossCount,
    },
  };
}

/**
 * Loads WebGL only when the resolved renderer actually needs it. This keeps
 * the WebGL implementation out of the pristine-session and small-graph path.
 */
export async function loadGraphRenderer(
  options: GraphRendererLoadOptions,
): Promise<LoadedGraphRenderer> {
  const webglSupported = options.webglSupported ?? detectWebGLSupport();
  const contextLossCount = options.contextLossCount ?? 0;
  if (options.forcedCanvasReason) {
    return canvasResult(
      options.preference,
      webglSupported,
      options.forcedCanvasReason,
      contextLossCount,
    );
  }

  const resolved = resolveGraphRendererKind(
    options.preference,
    options.nodeCount,
    options.edgeCount,
    webglSupported,
  );
  if (resolved === "canvas") {
    const fallbackReason =
      options.preference === "hybrid-webgl" && !webglSupported
        ? "WEBGL_UNSUPPORTED"
        : undefined;
    return canvasResult(
      options.preference,
      webglSupported,
      fallbackReason,
      contextLossCount,
    );
  }

  const now = options.now ?? (() => performance.now());
  const startedAt = now();
  try {
    const module = await (options.loadWebGL ?? (() => import("@antv/g-webgl")))();
    const lazyLoadMs = Math.max(0, now() - startedAt);
    const canvasLayers = new Map<CanvasLayer, CanvasRenderer>();
    const runtimeFailure = (reason: string) => options.onRuntimeFailure?.(reason);
    const mainRenderer = new module.Renderer({
      targets: ["webgl2", "webgl1"],
      onContextLost: (event) => {
        event.preventDefault?.();
        runtimeFailure("WEBGL_CONTEXT_LOST");
      },
      onContextCreationError: () => runtimeFailure("WEBGL_CONTEXT_CREATION_ERROR"),
      onContextRestored: () => undefined,
    });
    const renderer: G6RendererFactory = (layer) => {
      if (layer === "main") return mainRenderer;
      const existing = canvasLayers.get(layer);
      if (existing) return existing;
      const canvas = new CanvasRenderer();
      canvasLayers.set(layer, canvas);
      return canvas;
    };
    return {
      renderer,
      status: {
        requested: options.preference,
        resolved: "hybrid-webgl",
        webglSupported,
        lazyLoadMs,
        contextLossCount,
      },
    };
  } catch (error) {
    const lazyLoadMs = Math.max(0, now() - startedAt);
    const message = error instanceof Error ? error.message : "unknown error";
    return canvasResult(
      options.preference,
      webglSupported,
      `WEBGL_IMPORT_FAILED: ${message}`,
      contextLossCount,
      lazyLoadMs,
    );
  }
}
