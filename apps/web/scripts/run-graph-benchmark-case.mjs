import { chromium } from "@playwright/test";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { withStageDeadline as deadline } from "./benchmark-stage-control.mjs";

const specPath = process.argv[2];
if (!specPath) throw new Error("A benchmark case specification path is required");
const spec = JSON.parse(await readFile(specPath, "utf8"));

const atomicJson = async (target, value) => {
  await mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, target);
};

const percentile = (values, percentage) => {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.max(0, Math.min(
    ordered.length - 1,
    Math.ceil(percentage * ordered.length) - 1,
  ))];
};

const summarize = (capture) => ({
  frameCount: capture.frames.length,
  inputSampleCount: capture.inputLatencies.length,
  frameP50: percentile(capture.frames, 0.5),
  frameP95: percentile(capture.frames, 0.95),
  frameP99: percentile(capture.frames, 0.99),
  frameMax: Math.max(0, ...capture.frames),
  over20Ratio: capture.frames.filter((value) => value > 20).length / Math.max(1, capture.frames.length),
  over33Ratio: capture.frames.filter((value) => value > 33.3).length / Math.max(1, capture.frames.length),
  inputLatencyP95: percentile(capture.inputLatencies, 0.95),
  longTaskMax: Math.max(0, ...capture.longTasks),
  totalBlockingTime: capture.longTasks.reduce((sum, value) => sum + Math.max(0, value - 50), 0),
});

async function startCapture(page) {
  await page.evaluate(() => {
    const state = {
      frames: [], inputLatencies: [], longTasks: [], lastFrame: performance.now(),
      pendingInputAt: null, active: true,
    };
    const tick = (timestamp) => {
      if (!state.active) return;
      state.frames.push(timestamp - state.lastFrame);
      state.lastFrame = timestamp;
      if (state.pendingInputAt !== null) {
        state.inputLatencies.push(Math.max(0, performance.now() - state.pendingInputAt));
        state.pendingInputAt = null;
      }
      requestAnimationFrame(tick);
    };
    const noteInput = () => {
      if (state.pendingInputAt === null) state.pendingInputAt = performance.now();
    };
    const observer = "PerformanceObserver" in window
      ? new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) state.longTasks.push(entry.duration);
        })
      : null;
    try { observer?.observe({ type: "longtask", buffered: false }); } catch { /* unsupported */ }
    window.addEventListener("pointermove", noteInput, { passive: true });
    window.addEventListener("wheel", noteInput, { passive: true });
    window.__SGFM_CAPTURE__ = { state, observer, noteInput };
    requestAnimationFrame(tick);
  });
}

async function stopCapture(page) {
  await page.evaluate(() => new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      resolve();
    };
    const timer = setTimeout(finish, 250);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      clearTimeout(timer);
      finish();
    }));
  }));
  return page.evaluate(() => {
    const capture = window.__SGFM_CAPTURE__;
    if (!capture) return { frames: [], inputLatencies: [], longTasks: [] };
    capture.state.active = false;
    capture.observer?.disconnect();
    window.removeEventListener("pointermove", capture.noteInput);
    window.removeEventListener("wheel", capture.noteInput);
    delete window.__SGFM_CAPTURE__;
    return {
      frames: capture.state.frames.slice(1),
      inputLatencies: capture.state.inputLatencies,
      longTasks: capture.state.longTasks,
    };
  });
}

async function measureAction(page, stage, action) {
  await deadline(`${stage}:capture-start`, () => startCapture(page), spec.evaluateTimeoutMs);
  let actionError;
  try {
    await deadline(stage, action, spec.actionTimeoutMs);
  } catch (error) {
    actionError = error;
  }
  let capture = { frames: [], inputLatencies: [], longTasks: [] };
  try {
    capture = await deadline(`${stage}:capture-stop`, () => stopCapture(page), spec.evaluateTimeoutMs);
  } finally {
    if (actionError) throw actionError;
  }
  return capture;
}

async function repeatedMove(page, points, duration) {
  const startedAt = Date.now();
  let index = 0;
  while (Date.now() - startedAt < duration) {
    const point = points[index % points.length];
    await page.mouse.move(point.x, point.y, { steps: 2 });
    index += 1;
  }
}

let browser;
let context;
let cdp;
let page;
let traceStarted = false;
let cpuProfileStarted = false;
const consoleErrors = [];
const pageErrors = [];
const stages = [];
const noteStage = (stage, status, detail = {}) => stages.push({
  stage, status, at: new Date().toISOString(), ...detail,
});

try {
  noteStage("launch", "started");
  browser = await deadline("launch", () => chromium.launch({
    ...(spec.executablePath ? { executablePath: spec.executablePath } : {}),
    headless: !spec.headed,
    args: ["--enable-gpu", "--ignore-gpu-blocklist", "--disable-extensions"],
  }), 30_000);
  noteStage("launch", "completed");
  context = await browser.newContext({
    viewport: { width: 1000, height: 756 },
    deviceScaleFactor: 1,
    reducedMotion: "no-preference",
  });
  page = await context.newPage();
  page.setDefaultTimeout(spec.evaluateTimeoutMs);
  page.setDefaultNavigationTimeout(spec.navigationTimeoutMs);
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  cdp = await context.newCDPSession(page);
  if (spec.cpuRate > 1) {
    await deadline("cpu-throttle", () => cdp.send("Emulation.setCPUThrottlingRate", {
      rate: spec.cpuRate,
    }), spec.evaluateTimeoutMs);
  }
  if (spec.profile === "canvas") {
    await mkdir(spec.profileDirectory, { recursive: true });
    await context.tracing.start({ screenshots: false, snapshots: false, sources: false });
    traceStarted = true;
    await cdp.send("Profiler.enable");
    await cdp.send("Performance.enable");
    await cdp.send("Profiler.start");
    cpuProfileStarted = true;
  }

  noteStage("navigate", "started");
  const navigationStarted = Date.now();
  await deadline("navigate", () => page.goto(spec.url, {
    waitUntil: "domcontentloaded",
    timeout: spec.navigationTimeoutMs,
  }), spec.navigationTimeoutMs);
  noteStage("navigate", "completed");
  noteStage("ready", "started");
  await deadline("ready", () => page.waitForFunction(() => (
    document.documentElement.dataset.graphBenchmarkReady === "true"
  ), undefined, { timeout: spec.readyTimeoutMs }), spec.readyTimeoutMs);
  const navigationMs = Date.now() - navigationStarted;
  noteStage("ready", "completed", { navigationMs });
  await deadline("settle", () => page.waitForTimeout(1_000), 2_000);

  const runtimeBefore = await deadline(
    "before",
    () => page.evaluate(() => window.__SGFM_GRAPH_BENCHMARK__),
    spec.evaluateTimeoutMs,
  );
  const gpuAttribution = spec.probeMode === "gpu"
    ? await deadline("gpu-attribution", () => page.evaluate(async () => {
        const root = document.querySelector(".graph-preview__canvas");
        const probe = window.__SGFM_GPU_PROBE__;
        const graph = window.__SGFM_GRAPH_BENCHMARK_GRAPH__;
        if (!root || !probe || !graph) return null;
        const waitForPresent = () => new Promise((resolve) => {
          let settled = false;
          const finish = () => {
            if (settled) return;
            settled = true;
            resolve();
          };
          const timer = setTimeout(finish, 250);
          requestAnimationFrame(() => requestAnimationFrame(() => {
            clearTimeout(timer);
            finish();
          }));
        });
        const measurePatch = async (kind) => {
          const since = performance.now();
          const startedAt = performance.now();
          if (kind === "node") {
            graph.updateNodeData(graph.getNodeData().map((node) => ({
              id: node.id,
              style: { ...node.style, opacity: 0.997 },
            })));
          } else {
            graph.updateEdgeData(graph.getEdgeData().map((edge) => ({
              id: edge.id,
              style: { ...edge.style, opacity: 0.997 },
            })));
          }
          await graph.draw();
          await waitForPresent();
          return { cpuPatchMs: performance.now() - startedAt, ...probe.snapshot(root, since) };
        };
        return {
          initial: probe.snapshot(root),
          nodePatch: await measurePatch("node"),
          edgePatch: await measurePatch("edge"),
        };
      }), spec.actionTimeoutMs)
    : null;
  const canvas = page.locator(".graph-preview__canvas canvas").first();
  const bounds = await deadline("canvas-bounds", () => canvas.boundingBox(), spec.evaluateTimeoutMs);
  if (!bounds || !runtimeBefore?.dragTarget) throw new Error("Benchmark drag target is unavailable");
  const dragTarget = {
    x: bounds.x + runtimeBefore.dragTarget.x,
    y: bounds.y + runtimeBefore.dragTarget.y,
  };

  noteStage("drag", "started");
  const drag = await measureAction(page, "drag", async () => {
    await page.mouse.move(dragTarget.x, dragTarget.y);
    await page.mouse.down();
    await repeatedMove(page, [
      { x: dragTarget.x + 120, y: dragTarget.y + 20 },
      { x: dragTarget.x + 60, y: dragTarget.y + 100 },
      { x: dragTarget.x - 40, y: dragTarget.y + 50 },
      dragTarget,
    ], spec.actionMs);
    await page.mouse.up();
  });
  noteStage("drag", "completed");

  const panOrigin = { x: bounds.x + 80, y: bounds.y + bounds.height - 80 };
  noteStage("pan", "started");
  const pan = await measureAction(page, "pan", async () => {
    await page.mouse.move(panOrigin.x, panOrigin.y);
    await page.mouse.down();
    await repeatedMove(page, [
      { x: panOrigin.x + 140, y: panOrigin.y },
      { x: panOrigin.x + 80, y: panOrigin.y - 80 },
      panOrigin,
    ], spec.actionMs);
    await page.mouse.up();
  });
  noteStage("pan", "completed");

  noteStage("zoom", "started");
  const zoom = await measureAction(page, "zoom", async () => {
    const startedAt = Date.now();
    let direction = -1;
    await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    while (Date.now() - startedAt < spec.actionMs) {
      await page.mouse.wheel(0, direction * 80);
      direction *= -1;
      await page.waitForTimeout(40);
    }
  });
  noteStage("zoom", "completed");

  noteStage("selection", "started");
  const selection = await measureAction(page, "selection", async () => {
    // Three rounds are enough to expose selection regressions while keeping
    // CPU-throttled large-graph trend runs inside the hard action deadline.
    for (let index = 0; index < 3; index += 1) {
      await page.mouse.click(dragTarget.x, dragTarget.y);
      await page.keyboard.press("Escape");
    }
  });
  noteStage("selection", "completed");

  const runtimeBeforeResize = await deadline(
    "resize:before",
    () => page.evaluate(() => window.__SGFM_GRAPH_BENCHMARK__),
    spec.evaluateTimeoutMs,
  );
  noteStage("resize", "started");
  await deadline("resize", async () => {
    await page.setViewportSize({ width: 900, height: 800 });
    await page.waitForTimeout(300);
    await page.setViewportSize({ width: 1000, height: 756 });
    await page.waitForTimeout(300);
  }, spec.actionTimeoutMs);
  noteStage("resize", "completed");
  const runtimeAfter = await deadline(
    "after",
    () => page.evaluate(() => window.__SGFM_GRAPH_BENCHMARK__),
    spec.evaluateTimeoutMs,
  );

  if (spec.screenshotPath) {
    noteStage("screenshot", "started");
    await deadline("screenshot", () => page.screenshot({
      path: spec.screenshotPath,
      fullPage: true,
    }), spec.screenshotTimeoutMs);
    noteStage("screenshot", "completed");
  }

  const memory = await deadline("memory", () => page.evaluate(() => {
    const candidate = performance;
    if (!("memory" in candidate)) return null;
    const value = candidate.memory;
    return {
      jsHeapSizeLimit: value.jsHeapSizeLimit,
      totalJSHeapSize: value.totalJSHeapSize,
      usedJSHeapSize: value.usedJSHeapSize,
    };
  }), spec.evaluateTimeoutMs);
  const gpu = await deadline("gpu", () => page.evaluate(() => {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl");
    if (!gl) return "unavailable";
    const extension = gl.getExtension("WEBGL_debug_renderer_info");
    return extension
      ? String(gl.getParameter(extension.UNMASKED_RENDERER_WEBGL))
      : String(gl.getParameter(gl.RENDERER));
  }), spec.evaluateTimeoutMs);

  let profileArtifacts = null;
  if (spec.profile === "canvas") {
    const performanceMetrics = await cdp.send("Performance.getMetrics");
    const cpuProfile = await cdp.send("Profiler.stop");
    cpuProfileStarted = false;
    const cpuProfilePath = path.join(spec.profileDirectory, "cpu-profile.json");
    const metricsPath = path.join(spec.profileDirectory, "performance-metrics.json");
    await writeFile(cpuProfilePath, JSON.stringify(cpuProfile.profile));
    await writeFile(metricsPath, JSON.stringify(performanceMetrics, null, 2));
    const tracePath = path.join(spec.profileDirectory, "playwright-trace.zip");
    await deadline("profile:trace-stop", () => context.tracing.stop({ path: tracePath }), 8_000);
    traceStarted = false;
    profileArtifacts = { cpuProfilePath, metricsPath, tracePath };
  }

  const result = {
    status: "completed",
    baseline: spec.baseline,
    rendererRequested: spec.renderer,
    rendererResolved: runtimeAfter?.rendererResolved ?? runtimeBefore?.rendererResolved ?? null,
    rendererFallbackReason: runtimeAfter?.rendererFallbackReason ?? runtimeBefore?.rendererFallbackReason ?? null,
    webglContextLossCount: Math.max(
      runtimeBefore?.webglContextLossCount ?? 0,
      runtimeAfter?.webglContextLossCount ?? 0,
    ),
    rendererLazyLoadMs: runtimeAfter?.rendererLazyLoadMs ?? runtimeBefore?.rendererLazyLoadMs ?? null,
    probeMode: spec.probeMode,
    gpuAttribution,
    seed: runtimeAfter?.seed ?? runtimeBefore?.seed ?? null,
    caseId: spec.caseId,
    suite: spec.suite,
    run: spec.run,
    navigationMs,
    sceneBuildMs: runtimeBefore?.sceneBuildMs ?? null,
    initialReadyMs: runtimeBefore?.initialReadyMs ?? null,
    drag: summarize(drag),
    pan: summarize(pan),
    zoom: summarize(zoom),
    selection: summarize(selection),
    runtimeBeforeResize,
    runtimeAfter,
    memory,
    gpu,
    browserVersion: browser.version(),
    consoleErrors,
    pageErrors,
    stages,
    profileArtifacts,
  };
  await atomicJson(spec.resultPath, result);
} catch (error) {
  let diagnostics = null;
  if (page) {
    const diagnosticErrors = [];
    const capture = async (label, action, timeoutMs = 2_000) => {
      try {
        return await deadline(`diagnostic:${label}`, action, timeoutMs);
      } catch (diagnosticError) {
        diagnosticErrors.push({
          label,
          error: diagnosticError instanceof Error ? diagnosticError.message : String(diagnosticError),
        });
        return null;
      }
    };
    const dom = await capture("dom", () => page.evaluate(() => {
      const root = document.querySelector(".graph-preview");
      const canvasHost = document.querySelector(".graph-preview__canvas");
      return {
        url: location.href,
        title: document.title,
        ready: document.documentElement.dataset.graphBenchmarkReady ?? null,
        documentDataset: { ...document.documentElement.dataset },
        rootDataset: root instanceof HTMLElement ? { ...root.dataset } : null,
        canvasHostDataset: canvasHost instanceof HTMLElement ? { ...canvasHost.dataset } : null,
        canvasCount: document.querySelectorAll("canvas").length,
        graphRootCount: document.querySelectorAll(".graph-preview").length,
        renderError: document.querySelector(".graph-preview__error")?.textContent?.trim() ?? null,
        bodyText: document.body.innerText.slice(0, 4_000),
        gpuProbeAvailable: Boolean(window.__SGFM_GPU_PROBE__),
        runtime: window.__SGFM_GRAPH_BENCHMARK__ ?? null,
      };
    }));
    const performanceMetrics = cdp
      ? await capture("performance", () => cdp.send("Performance.getMetrics"))
      : null;
    if (spec.failureScreenshotPath) {
      await capture("screenshot", () => page.screenshot({
        path: spec.failureScreenshotPath,
        fullPage: true,
        timeout: 5_000,
      }), 5_000);
    }
    diagnostics = {
      capturedAt: new Date().toISOString(),
      browserVersion: browser ? browser.version() : null,
      consoleErrors,
      pageErrors,
      dom,
      performanceMetrics,
      diagnosticErrors,
      screenshotPath: spec.failureScreenshotPath ?? null,
    };
    if (spec.failureDiagnosticPath) {
      await atomicJson(spec.failureDiagnosticPath, diagnostics).catch(() => undefined);
    }
  }
  const failure = {
    status: "failed",
    baseline: spec.baseline,
    renderer: spec.renderer,
    caseId: spec.caseId,
    suite: spec.suite,
    run: spec.run,
    error: error instanceof Error ? `${error.stack ?? error.message}` : String(error),
    code: error?.code ?? "CASE_FAILED",
    stage: error?.stage ?? stages.at(-1)?.stage ?? "unknown",
    stages,
    diagnostics,
  };
  await atomicJson(spec.resultPath, failure);
  process.exitCode = 1;
} finally {
  const cleanupFailures = [];
  const cleanupStage = async (stage, action, timeoutMs) => {
    noteStage(stage, "started");
    try {
      await deadline(stage, action, timeoutMs);
      noteStage(stage, "completed");
    } catch (error) {
      cleanupFailures.push(error);
      noteStage(stage, "failed", { code: error?.code ?? "CLEANUP_FAILED" });
    }
  };
  if (cpuProfileStarted && cdp) {
    await cleanupStage("cleanup:profiler", () => cdp.send("Profiler.stop"), 2_000);
  }
  if (traceStarted && context) {
    await cleanupStage("cleanup:trace", () => context.tracing.stop(), 2_000);
  }
  if (context) {
    await cleanupStage("cleanup:context", () => context.close(), spec.cleanupTimeoutMs);
  }
  if (browser) {
    await cleanupStage("cleanup:browser", () => browser.close(), spec.cleanupTimeoutMs);
  }
  if (cleanupFailures.length > 0) {
    const cleanupError = cleanupFailures[0];
    await atomicJson(spec.resultPath, {
      status: "failed",
      baseline: spec.baseline,
      renderer: spec.renderer,
      caseId: spec.caseId,
      suite: spec.suite,
      run: spec.run,
      error: cleanupError instanceof Error
        ? `${cleanupError.stack ?? cleanupError.message}`
        : String(cleanupError),
      code: cleanupError?.code ?? "CLEANUP_FAILED",
      stage: cleanupError?.stage ?? "cleanup",
      stages,
    });
    process.exitCode = 1;
  }
}
