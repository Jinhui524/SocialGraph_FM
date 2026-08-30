import { describe, expect, it, vi } from "vitest";
import {
  loadGraphRenderer,
  resolveGraphRendererKind,
} from "./graphRenderer";

describe("graph renderer selection", () => {
  it("keeps the WebGL chunk out of the auto small-graph path", async () => {
    const loadWebGL = vi.fn();
    const result = await loadGraphRenderer({
      preference: "auto",
      nodeCount: 300,
      edgeCount: 1_000,
      webglSupported: true,
      loadWebGL,
    });
    expect(result.status.resolved).toBe("canvas");
    expect(loadWebGL).not.toHaveBeenCalled();
  });

  it("keeps auto on Canvas until the WebGL release gate is promoted", () => {
    expect(resolveGraphRendererKind("auto", 301, 1_000, true)).toBe("canvas");
    expect(resolveGraphRendererKind("auto", 3_000, 12_000, true)).toBe("canvas");
  });

  it("falls back when WebGL is unavailable", async () => {
    const result = await loadGraphRenderer({
      preference: "hybrid-webgl",
      nodeCount: 1_000,
      edgeCount: 5_000,
      webglSupported: false,
    });
    expect(result.status).toMatchObject({
      requested: "hybrid-webgl",
      resolved: "canvas",
      fallbackReason: "WEBGL_UNSUPPORTED",
    });
  });

  it("uses WebGL only on the main layer", async () => {
    let webglOptions: { onContextLost?: (event: Event) => void } | undefined;
    const onRuntimeFailure = vi.fn();
    class FakeWebGLRenderer {
      constructor(options?: unknown) {
        webglOptions = options as typeof webglOptions;
      }
    }
    const result = await loadGraphRenderer({
      preference: "hybrid-webgl",
      nodeCount: 1_000,
      edgeCount: 5_000,
      webglSupported: true,
      onRuntimeFailure,
      now: (() => {
        let value = 10;
        return () => (value += 5);
      })(),
      loadWebGL: async () =>
        ({ Renderer: FakeWebGLRenderer }) as unknown as typeof import("@antv/g-webgl"),
    });
    expect(result.status).toMatchObject({
      resolved: "hybrid-webgl",
      lazyLoadMs: 5,
    });
    expect(result.renderer("main")).toBeInstanceOf(FakeWebGLRenderer);
    expect(result.renderer("label")).not.toBeInstanceOf(FakeWebGLRenderer);
    expect(result.renderer("transient")).not.toBeInstanceOf(FakeWebGLRenderer);
    const preventDefault = vi.fn();
    webglOptions?.onContextLost?.({ preventDefault } as unknown as Event);
    expect(preventDefault).toHaveBeenCalledOnce();
    expect(onRuntimeFailure).toHaveBeenCalledWith("WEBGL_CONTEXT_LOST");
  });

  it("classifies a failed dynamic import and returns Canvas", async () => {
    const result = await loadGraphRenderer({
      preference: "hybrid-webgl",
      nodeCount: 1_000,
      edgeCount: 5_000,
      webglSupported: true,
      loadWebGL: async () => {
        throw new Error("chunk unavailable");
      },
    });
    expect(result.status.resolved).toBe("canvas");
    expect(result.status.fallbackReason).toContain("WEBGL_IMPORT_FAILED");
  });
});
