import { chromium } from "@playwright/test";
import { preview } from "vite";
import { appendFile, mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { createHash } from "node:crypto";
import { waitForChildWithDeadline } from "./benchmark-watchdog.mjs";

const benchmarkSchemaVersion = "3.0";

const settleWithin = async (promise, timeoutMs, label = "operation") => {
  let timer;
  try {
    return await Promise.race([
      Promise.resolve(promise),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs}ms`)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
};

const allCases = ["small", "medium", "large"];
const allRenderers = ["canvas", "hybrid-webgl", "auto"];
const allBaselines = [
  { id: "native", cpuRate: 1 },
  { id: "cpu-4x", cpuRate: 4 },
];
const smoke = process.argv.includes("--smoke");
function commandLineValue(...names) {
  for (const name of names) {
    const equalsPrefix = `--${name}=`;
    const equalsMatch = process.argv.find((value) => value.startsWith(equalsPrefix));
    if (equalsMatch) return equalsMatch.slice(equalsPrefix.length);
    const index = process.argv.indexOf(`--${name}`);
    if (index >= 0 && process.argv[index + 1] && !process.argv[index + 1].startsWith("--")) {
      return process.argv[index + 1];
    }
  }
  return undefined;
}

function selectedIds(name, allowedIds) {
  const raw = process.env[name]?.trim();
  if (!raw) return allowedIds;
  const selected = [...new Set(raw.split(",").map((value) => value.trim()).filter(Boolean))];
  const invalid = selected.filter((value) => !allowedIds.includes(value));
  if (invalid.length > 0 || selected.length === 0) {
    throw new Error(`${name} must contain one or more of: ${allowedIds.join(", ")}`);
  }
  return selected;
}

function selectedValues(raw, name, allowedValues, fallback) {
  if (!raw?.trim()) return fallback;
  const selected = [...new Set(raw.split(",").map((value) => value.trim()).filter(Boolean))];
  const invalid = selected.filter((value) => !allowedValues.includes(value));
  if (invalid.length > 0 || selected.length === 0) {
    throw new Error(`${name} must contain one or more of: ${allowedValues.join(", ")}`);
  }
  return selected;
}

const cases = selectedIds("SGFM_BENCHMARK_CASES", allCases);
const selectedBaselineIds = selectedIds(
  "SGFM_BENCHMARK_BASELINES",
  allBaselines.map((baseline) => baseline.id),
);
const baselines = allBaselines.filter((baseline) => selectedBaselineIds.includes(baseline.id));
const rendererArgument = commandLineValue("renderers", "renderer");
const renderers = selectedValues(
  rendererArgument ?? process.env.SGFM_BENCHMARK_RENDERERS,
  "renderer",
  allRenderers,
  ["canvas"],
);
const gateRenderer = commandLineValue("gate-renderer") ??
  process.env.SGFM_BENCHMARK_GATE_RENDERER?.trim() ??
  renderers[0];
if (!allRenderers.includes(gateRenderer)) {
  throw new Error(`gate renderer must be one of: ${allRenderers.join(", ")}`);
}
if (!renderers.includes(gateRenderer)) {
  throw new Error(`gate renderer ${gateRenderer} must also be included in the renderer matrix`);
}
const probeMode = commandLineValue("probe") ?? process.env.SGFM_BENCHMARK_PROBE?.trim() ?? "off";
if (!['off', 'gpu'].includes(probeMode)) {
  throw new Error("probe must be one of: off, gpu");
}
const profileMode = commandLineValue("profile") ?? process.env.SGFM_BENCHMARK_PROFILE?.trim() ?? "off";
if (!["off", "canvas"].includes(profileMode)) {
  throw new Error("profile must be one of: off, canvas");
}
function positiveInteger(name, fallback) {
  const value = Number(process.env[name] ?? fallback);
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

const runs = positiveInteger("SGFM_BENCHMARK_RUNS", smoke ? 1 : 5);
const suites = positiveInteger("SGFM_BENCHMARK_SUITES", smoke ? 1 : 2);
const faultInjection = process.env.SGFM_BENCHMARK_FAULT_INJECTION?.trim();
const allowedFaultInjections = ["ready", "action", "raf", "cleanup"];
if (faultInjection && !allowedFaultInjections.includes(faultInjection)) {
  throw new Error(`SGFM_BENCHMARK_FAULT_INJECTION must be one of: ${allowedFaultInjections.join(", ")}`);
}
const benchmarkScope = profileMode === "canvas"
  ? "diagnostic-canvas-profile"
  : probeMode === "gpu"
  ? "diagnostic-gpu"
  : smoke
  ? "smoke"
  : cases.length === allCases.length &&
      allCases.every((caseId) => cases.includes(caseId)) &&
      baselines.length === allBaselines.length &&
      allBaselines.every((baseline) => selectedBaselineIds.includes(baseline.id)) &&
      runs >= 5 &&
      suites >= 2
    ? "full-release"
    : "selected";
const actionMs = positiveInteger("SGFM_BENCHMARK_ACTION_MS", smoke ? 300 : 3_000);
const readyTimeoutMs = positiveInteger("SGFM_BENCHMARK_READY_TIMEOUT_MS", 30_000);
const cpuReadyTimeoutMs = positiveInteger("SGFM_BENCHMARK_CPU_READY_TIMEOUT_MS", 45_000);
const headed = process.env.SGFM_BENCHMARK_HEADED === "1";
const requestedRunId = process.env.SGFM_BENCHMARK_RUN_ID?.trim();
if (requestedRunId && !/^[A-Za-z0-9._-]+$/.test(requestedRunId)) {
  throw new Error("SGFM_BENCHMARK_RUN_ID may contain only letters, digits, dot, underscore, and hyphen");
}
const runId = requestedRunId ?? new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-");
const outputRoot = process.env.SGFM_BENCHMARK_OUTPUT_ROOT?.trim()
  ? path.resolve(process.env.SGFM_BENCHMARK_OUTPUT_ROOT.trim())
  : path.resolve("artifacts", "benchmarks", "graph");
const outputDirectory = path.join(outputRoot, runId);
const screenshotDirectory = path.join(outputDirectory, "screenshots");
const traceDirectory = path.join(outputDirectory, "traces");
const caseResultDirectory = path.join(outputDirectory, "case-results");
const caseSpecDirectory = path.join(outputDirectory, "case-specs");
const profileDirectory = path.join(outputDirectory, "profiles");
const failureDiagnosticDirectory = path.join(outputDirectory, "failure-diagnostics");
const eventsPath = path.join(outputDirectory, "events.jsonl");

await mkdir(screenshotDirectory, { recursive: true });
await mkdir(traceDirectory, { recursive: true });
await mkdir(caseResultDirectory, { recursive: true });
await mkdir(caseSpecDirectory, { recursive: true });
await mkdir(failureDiagnosticDirectory, { recursive: true });
if (profileMode === "canvas") await mkdir(profileDirectory, { recursive: true });

const chromeCandidates = [
  process.env.SGFM_BENCHMARK_BROWSER,
  process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Google", "Chrome", "Application", "chrome.exe"),
  process.env.LOCALAPPDATA && path.join(process.env.LOCALAPPDATA, "Microsoft", "Edge", "Application", "msedge.exe"),
  process.env.ProgramFiles && path.join(process.env.ProgramFiles, "Google", "Chrome", "Application", "chrome.exe"),
  process.env.ProgramFiles && path.join(process.env.ProgramFiles, "Microsoft", "Edge", "Application", "msedge.exe"),
  process.env["ProgramFiles(x86)"] && path.join(process.env["ProgramFiles(x86)"], "Google", "Chrome", "Application", "chrome.exe"),
  process.env["ProgramFiles(x86)"] && path.join(process.env["ProgramFiles(x86)"], "Microsoft", "Edge", "Application", "msedge.exe"),
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);
const executablePath = chromeCandidates.find((candidate) => existsSync(candidate));
let server;
let browser;
let browserVersion = "unavailable";
let failure;
if (!faultInjection) {
  try {
    server = await preview({
      root: process.cwd(),
      preview: { host: "127.0.0.1", port: 0, strictPort: false },
    });
  } catch (error) {
    failure = error instanceof Error ? `${error.stack ?? error.message}` : String(error);
  }
}

function percentile(values, percentage) {
  if (!values.length) return 0;
  const ordered = [...values].sort((left, right) => left - right);
  const index = Math.min(ordered.length - 1, Math.ceil(percentage * ordered.length) - 1);
  return ordered[Math.max(0, index)];
}

function expectedResolvedRenderer(renderer, caseId) {
  if (renderer !== "auto") return renderer;
  // Auto stays on the validated Canvas path until the hybrid release gate is
  // explicitly promoted in graphRenderer.ts.
  return "canvas";
}

async function startCapture(page) {
  await page.evaluate(() => {
    const state = {
      frames: [],
      inputLatencies: [],
      longTasks: [],
      lastFrame: performance.now(),
      pendingInputAt: null,
      active: true,
    };
    const tick = (timestamp) => {
      if (!state.active) return;
      state.frames.push(timestamp - state.lastFrame);
      state.lastFrame = timestamp;
      if (state.pendingInputAt !== null) {
        // rAF's timestamp can represent the start of the frame and be slightly
        // earlier than a pointer event dispatched during that frame.
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
    requestAnimationFrame(() => requestAnimationFrame(resolve));
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

async function measureAction(page, action) {
  await startCapture(page);
  await action();
  return stopCapture(page);
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

async function runMeasurement(baseline, renderer, caseId, suite, run) {
  if (!browser) throw new Error("Browser did not start");
  const context = await browser.newContext({
    viewport: { width: 1000, height: 756 },
    deviceScaleFactor: 1,
    reducedMotion: "no-preference",
  });
  const page = await context.newPage();
  const traceEnabled = suite === 1 && caseId === "large" && (run === 0 || run === 1);
  let traceStarted = false;
  let gpuConsoleSnapshot = null;
  let signalBenchmarkReady;
  const benchmarkReady = new Promise((resolve) => {
    signalBenchmarkReady = resolve;
  });
  try {
    console.log(`[graph-benchmark] start ${baseline.id}/${renderer}/${caseId}/suite-${suite}/${run === 0 ? "warmup" : `run-${run}`}`);
    if (traceEnabled) {
      await context.tracing.start({ screenshots: true, snapshots: true, sources: false });
      traceStarted = true;
    }
    const consoleErrors = [];
    const pageErrors = [];
    page.on("console", (message) => {
      const messageText = message.text();
      if (messageText.startsWith("__SGFM_GPU_PROBE_SAMPLE__")) {
        try {
          gpuConsoleSnapshot = JSON.parse(messageText.slice("__SGFM_GPU_PROBE_SAMPLE__".length));
        } catch {
          // Keep the most recent valid sample; malformed probe telemetry is
          // surfaced by the missing diagnostics rather than failing the page.
        }
      }
      if (messageText === "__SGFM_BENCHMARK_READY__") signalBenchmarkReady();
      if (message.type() === "error") consoleErrors.push(messageText);
    });
    page.on("pageerror", (error) => pageErrors.push(String(error)));
    const session = await context.newCDPSession(page);
    if (baseline.cpuRate > 1) {
      await session.send("Emulation.setCPUThrottlingRate", { rate: baseline.cpuRate });
    }

  const url = `http://127.0.0.1:4173/?benchmark=graph&case=${caseId}&renderer=${renderer}&probe=${probeMode}`;
  const navigationStarted = Date.now();
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60_000 });
    await Promise.race([
      benchmarkReady,
      new Promise((_, reject) => setTimeout(() => reject(new Error(
        `Benchmark readiness signal timed out after ${readyTimeoutMs}ms`,
      )), readyTimeoutMs)),
    ]);
  const navigationMs = Date.now() - navigationStarted;
  console.log(`[graph-benchmark] ready ${baseline.id}/${renderer}/${caseId} in ${navigationMs}ms`);
  await page.waitForTimeout(1_000);
  const runtimeBefore = await page.evaluate(() => window.__SGFM_GRAPH_BENCHMARK__);
  const gpuAttribution = probeMode === "gpu"
    ? await page.evaluate(async () => {
        const root = document.querySelector(".graph-preview__canvas");
        const probe = window.__SGFM_GPU_PROBE__;
        const graph = window.__SGFM_GRAPH_BENCHMARK_GRAPH__;
        if (!root || !probe || !graph) return null;
        const waitForPresent = () => new Promise((resolve) => (
          requestAnimationFrame(() => requestAnimationFrame(resolve))
        ));
        const measurePatch = async (kind) => {
          const since = performance.now();
          const startedAt = performance.now();
          if (kind === "node") {
            const data = graph.getNodeData();
            graph.updateNodeData(data.map((node) => ({
              id: node.id,
              style: { ...node.style, opacity: 0.997 },
            })));
          } else {
            const data = graph.getEdgeData();
            graph.updateEdgeData(data.map((edge) => ({
              id: edge.id,
              style: { ...edge.style, opacity: 0.997 },
            })));
          }
          await graph.draw();
          await waitForPresent();
          return {
            cpuPatchMs: performance.now() - startedAt,
            ...probe.snapshot(root, since),
          };
        };
        return {
          initial: probe.snapshot(root),
          nodePatch: await measurePatch("node"),
          edgePatch: await measurePatch("edge"),
        };
      })
    : null;
  const canvas = page.locator(".graph-preview__canvas canvas").first();
  const bounds = await canvas.boundingBox();
  if (!bounds) throw new Error(`Canvas unavailable for ${caseId}`);
  if (!runtimeBefore?.dragTarget) {
    throw new Error(`No measurable node drag target was published for ${renderer}/${caseId}`);
  }
  const dragTarget = {
    x: bounds.x + runtimeBefore.dragTarget.x,
    y: bounds.y + runtimeBefore.dragTarget.y,
  };

  const drag = await measureAction(page, async () => {
    await page.mouse.move(dragTarget.x, dragTarget.y);
    await page.mouse.down();
    await repeatedMove(page, [
      { x: dragTarget.x + 120, y: dragTarget.y + 20 },
      { x: dragTarget.x + 60, y: dragTarget.y + 100 },
      { x: dragTarget.x - 40, y: dragTarget.y + 50 },
      dragTarget,
    ], actionMs);
    await page.mouse.up();
  });

  const panOrigin = { x: bounds.x + 80, y: bounds.y + bounds.height - 80 };
  const pan = await measureAction(page, async () => {
    await page.mouse.move(panOrigin.x, panOrigin.y);
    await page.mouse.down();
    await repeatedMove(page, [
      { x: panOrigin.x + 140, y: panOrigin.y },
      { x: panOrigin.x + 80, y: panOrigin.y - 80 },
      panOrigin,
    ], actionMs);
    await page.mouse.up();
  });

  const zoom = await measureAction(page, async () => {
    const startedAt = Date.now();
    let direction = -1;
    await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
    while (Date.now() - startedAt < actionMs) {
      await page.mouse.wheel(0, direction * 80);
      direction *= -1;
      await page.waitForTimeout(40);
    }
  });

  const selection = await measureAction(page, async () => {
    for (let index = 0; index < 20; index += 1) {
      await page.mouse.click(dragTarget.x, dragTarget.y);
      await page.keyboard.press("Escape");
    }
  });

  const runtimeBeforeResize = await page.evaluate(() => window.__SGFM_GRAPH_BENCHMARK__);
  await page.setViewportSize({ width: 900, height: 800 });
  await page.waitForTimeout(300);
  await page.setViewportSize({ width: 1000, height: 756 });
  await page.waitForTimeout(300);
  const runtimeAfter = await page.evaluate(() => window.__SGFM_GRAPH_BENCHMARK__);

  if (suite === 1 && run === 1) {
    await page.screenshot({
      path: path.join(screenshotDirectory, `${baseline.id}-${renderer}-${caseId}.png`),
      fullPage: true,
    });
  }

  const memory = await page.evaluate(() => {
    const candidate = performance;
    if (!("memory" in candidate)) return null;
    const memory = candidate.memory;
    return {
      jsHeapSizeLimit: memory.jsHeapSizeLimit,
      totalJSHeapSize: memory.totalJSHeapSize,
      usedJSHeapSize: memory.usedJSHeapSize,
    };
  });
  const gpu = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl");
    if (!gl) return "unavailable";
    const extension = gl.getExtension("WEBGL_debug_renderer_info");
    return extension ? String(gl.getParameter(extension.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
  });
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

    const result = {
    baseline: baseline.id,
    rendererRequested: renderer,
    rendererResolved: runtimeAfter?.rendererResolved ?? runtimeBefore?.rendererResolved ?? null,
    rendererFallbackReason: runtimeAfter?.rendererFallbackReason ?? runtimeBefore?.rendererFallbackReason ?? null,
    webglContextLossCount: Math.max(
      runtimeBefore?.webglContextLossCount ?? 0,
      runtimeAfter?.webglContextLossCount ?? 0,
    ),
    rendererLazyLoadMs: runtimeAfter?.rendererLazyLoadMs ?? runtimeBefore?.rendererLazyLoadMs ?? null,
    probeMode,
    gpuAttribution,
    seed: runtimeAfter?.seed ?? runtimeBefore?.seed ?? null,
    caseId,
    suite,
    run,
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
    consoleErrors,
    pageErrors,
    };
    console.log(`[graph-benchmark] complete ${baseline.id}/${renderer}/${caseId}/suite-${suite}/${run === 0 ? "warmup" : `run-${run}`}`);
    return result;
  } catch (error) {
    const diagnostics = {
      diagnosticError: "Benchmark page did not become ready; DOM inspection skipped to avoid waiting on an unresponsive renderer",
      gpuConsoleSnapshot,
    };
    if (error && typeof error === "object") {
      error.benchmarkDiagnostics = diagnostics;
    }
    console.error(`[graph-benchmark] failed ${baseline.id}/${renderer}/${caseId}/suite-${suite}/${run === 0 ? "warmup" : `run-${run}`}`);
    throw error;
  } finally {
    if (traceStarted) {
      await settleWithin(context.tracing.stop({
        path: path.join(
          traceDirectory,
          `${baseline.id}-${renderer}-${caseId}-${run === 0 ? "warmup" : `run-${run}`}.zip`,
        ),
      }).catch(() => undefined), 5_000);
    }
    await settleWithin(context.close().catch(() => undefined), 5_000);
  }
}

const appendEvent = (event) => appendFile(
  eventsPath,
  `${JSON.stringify({ at: new Date().toISOString(), ...event })}\n`,
);

const atomicJson = async (target, value) => {
  const temporary = `${target}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, target);
};

const runStartedAt = new Date().toISOString();
const serverAddress = server?.httpServer.address();
const serverPort = typeof serverAddress === "object" && serverAddress
  ? serverAddress.port
  : 4173;
const caseWorkerPath = path.resolve(
  faultInjection
    ? "scripts/run-graph-benchmark-fault-case.mjs"
    : "scripts/run-graph-benchmark-case.mjs",
);
const scenarioTimeoutMs = positiveInteger(
  "SGFM_BENCHMARK_SCENARIO_TIMEOUT_MS",
  smoke ? 90_000 : 180_000,
);
const actionTimeoutMs = positiveInteger(
  "SGFM_BENCHMARK_ACTION_TIMEOUT_MS",
  Math.max(actionMs * 4, 8_000),
);
const evaluateTimeoutMs = positiveInteger("SGFM_BENCHMARK_EVALUATE_TIMEOUT_MS", 8_000);
const cleanupTimeoutMs = positiveInteger("SGFM_BENCHMARK_CLEANUP_TIMEOUT_MS", 8_000);

await atomicJson(path.join(outputDirectory, "manifest.json"), {
  benchmarkSchemaVersion,
  runId,
  status: failure ? "startup_failed" : "running",
  startedAt: runStartedAt,
  matrix: {
    baselines: baselines.map((baseline) => baseline.id),
    renderers,
    cases,
    suites,
    runs,
  },
  actionMs,
  readyTimeoutMs,
  cpuReadyTimeoutMs,
  probeMode,
  profileMode,
  scenarioTimeoutMs,
  actionTimeoutMs,
  evaluateTimeoutMs,
  cleanupTimeoutMs,
  faultInjection: faultInjection ?? null,
});
await appendEvent({ event: "run_started", runId, serverPort });
let signalFinalizationStarted = false;
const persistSignalStatus = async (signal) => {
  if (signalFinalizationStarted) return;
  signalFinalizationStarted = true;
  await atomicJson(path.join(outputDirectory, "run-status.json"), {
    runId,
    status: "interrupted",
    signal,
    finishedAt: new Date().toISOString(),
  }).catch(() => undefined);
  await appendEvent({ event: "run_interrupted", signal }).catch(() => undefined);
  process.exit(signal === "SIGINT" ? 130 : 143);
};
process.once("SIGINT", () => { void persistSignalStatus("SIGINT"); });
process.once("SIGTERM", () => { void persistSignalStatus("SIGTERM"); });

async function runMeasurementIsolated(baseline, renderer, caseId, suite, run) {
  const caseKey = `${baseline.id}-${renderer}-${caseId}-suite-${suite}-${run === 0 ? "warmup" : `run-${run}`}`;
  const specPath = path.join(caseSpecDirectory, `${caseKey}.json`);
  const resultPath = path.join(caseResultDirectory, `${caseKey}.json`);
  const screenshotPath = suite === 1 && run === 1
    ? path.join(screenshotDirectory, `${baseline.id}-${renderer}-${caseId}.png`)
    : null;
  const caseProfileDirectory = profileMode === "canvas"
    ? path.join(profileDirectory, caseKey)
    : null;
  const effectiveActionTimeoutMs = baseline.cpuRate > 1
    ? Math.min(actionTimeoutMs * baseline.cpuRate, Math.floor(scenarioTimeoutMs / 6))
    : actionTimeoutMs;
  const spec = {
    baseline: baseline.id,
    cpuRate: baseline.cpuRate,
    renderer,
    caseId,
    suite,
    run,
    url: `http://127.0.0.1:${serverPort}/?benchmark=graph&case=${caseId}&renderer=${renderer}&probe=${probeMode}&profile=${profileMode}`,
    resultPath,
    failureDiagnosticPath: path.join(failureDiagnosticDirectory, `${caseKey}.json`),
    failureScreenshotPath: path.join(failureDiagnosticDirectory, `${caseKey}.png`),
    screenshotPath,
    profile: profileMode,
    profileDirectory: caseProfileDirectory,
    probeMode,
    actionMs,
    actionTimeoutMs: effectiveActionTimeoutMs,
    navigationTimeoutMs: 30_000,
    readyTimeoutMs: baseline.cpuRate > 1 ? cpuReadyTimeoutMs : readyTimeoutMs,
    evaluateTimeoutMs,
    screenshotTimeoutMs: 15_000,
    cleanupTimeoutMs,
    executablePath,
    headed,
    ...(faultInjection ? { faultInjection } : {}),
  };
  await atomicJson(specPath, spec);
  await appendEvent({ event: "case_started", caseKey });
  const child = spawn(process.execPath, [caseWorkerPath, specPath], {
    cwd: process.cwd(),
    env: process.env,
    windowsHide: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  let stdout = "";
  let stderr = "";
  child.stdout.on("data", (chunk) => {
    stdout = `${stdout}${chunk}`.slice(-32_000);
  });
  child.stderr.on("data", (chunk) => {
    stderr = `${stderr}${chunk}`.slice(-32_000);
  });
  const heartbeat = setInterval(() => {
    void appendEvent({ event: "case_heartbeat", caseKey, pid: child.pid });
  }, 5_000);
  const outcome = await waitForChildWithDeadline(child, scenarioTimeoutMs);
  clearInterval(heartbeat);
  const timedOut = outcome.timedOut;

  let result;
  try {
    result = JSON.parse(await readFile(resultPath, "utf8"));
  } catch {
    result = null;
  }
  if (timedOut) {
    const workerResultStatus = result?.status ?? "missing";
    result = {
      status: "failed",
      code: "SCENARIO_TIMEOUT",
      stage: result?.stage ?? (workerResultStatus === "completed" ? "cleanup" : "scenario"),
      error: `Scenario exceeded the ${scenarioTimeoutMs}ms hard deadline`,
      workerResultStatus,
      stages: result?.stages ?? [],
      ...(result?.faultInjection ? { faultInjection: result.faultInjection } : {}),
      stdout,
      stderr,
    };
    await atomicJson(resultPath, result);
  } else if (!result) {
    result = {
      status: "failed",
      code: "CASE_RESULT_MISSING",
      stage: "worker",
      error: `Case worker exited without a result (code=${outcome.code}, signal=${outcome.signal})`,
      stdout,
      stderr,
    };
    await atomicJson(resultPath, result);
  }
  await appendEvent({
    event: "case_finished",
    caseKey,
    status: result.status,
    code: result.code ?? outcome.code,
    stage: result.stage,
  });
  if (result.status !== "completed") {
    const error = new Error(result.error ?? `Benchmark case ${caseKey} failed`);
    error.code = result.code ?? "CASE_FAILED";
    error.stage = result.stage;
    error.benchmarkDiagnostics = { resultPath, stdout, stderr, stages: result.stages ?? [] };
    throw error;
  }
  browserVersion = result.browserVersion ?? browserVersion;
  return result;
}

const raw = [];
const measurementFailures = [];
const infrastructureTimeouts = new Map();
const timeoutKey = (baseline, renderer) => `${baseline}:${renderer}`;
const noteInfrastructureTimeout = (baseline, renderer, error) => {
  if (error?.code !== "SCENARIO_TIMEOUT" && error?.code !== "STAGE_TIMEOUT") return;
  const key = timeoutKey(baseline, renderer);
  infrastructureTimeouts.set(key, (infrastructureTimeouts.get(key) ?? 0) + 1);
};
try {
  if (failure) throw new Error(failure);
  for (const baseline of baselines) {
    for (const renderer of renderers) {
      for (let suite = 1; suite <= suites; suite += 1) {
        const orderedCases = suite % 2 === 1 ? cases : [...cases].reverse();
        for (const caseId of orderedCases) {
          if ((infrastructureTimeouts.get(timeoutKey(baseline.id, renderer)) ?? 0) >= 2) {
            measurementFailures.push({
              baseline: baseline.id,
              renderer,
              caseId,
              suite,
              run: null,
              phase: "skipped_after_infrastructure_failure",
              error: "Skipped after two isolated scenario timeouts",
              diagnostics: null,
            });
            continue;
          }
          // The first run is a warm-up and is not included in the report.
          try {
            await runMeasurementIsolated(baseline, renderer, caseId, suite, 0);
          } catch (error) {
            noteInfrastructureTimeout(baseline.id, renderer, error);
            measurementFailures.push({
              baseline: baseline.id,
              renderer,
              caseId,
              suite,
              run: 0,
              phase: "warmup",
              error: error instanceof Error ? `${error.stack ?? error.message}` : String(error),
              diagnostics: error && typeof error === "object" ? error.benchmarkDiagnostics ?? null : null,
            });
            continue;
          }
          for (let run = 1; run <= runs; run += 1) {
            try {
              raw.push(await runMeasurementIsolated(baseline, renderer, caseId, suite, run));
            } catch (error) {
              noteInfrastructureTimeout(baseline.id, renderer, error);
              measurementFailures.push({
                baseline: baseline.id,
                renderer,
                caseId,
                suite,
                run,
                phase: "measurement",
                error: error instanceof Error ? `${error.stack ?? error.message}` : String(error),
                diagnostics: error && typeof error === "object" ? error.benchmarkDiagnostics ?? null : null,
              });
            }
          }
        }
      }
    }
  }
  if (measurementFailures.length > 0) {
    failure = `${measurementFailures.length} benchmark case(s) failed; see raw.json`;
  }
} catch (error) {
  failure = error instanceof Error ? `${error.stack ?? error.message}` : String(error);
} finally {
  await settleWithin(browser?.close().catch(() => undefined), 5_000, "browser cleanup")
    .catch(() => undefined);
  if (server) {
    await settleWithin(new Promise((resolve) => {
      server.httpServer.close(() => resolve());
      server.httpServer.closeAllConnections?.();
    }), 5_000, "preview server cleanup").catch(() => undefined);
  }
}

const packageJson = JSON.parse(await readFile(path.resolve("package.json"), "utf8"));
const hashFiles = async (files) => {
  const hash = createHash("sha256");
  for (const file of files) {
    if (!existsSync(path.resolve(file))) continue;
    hash.update(file);
    hash.update(await readFile(path.resolve(file)));
  }
  return hash.digest("hex");
};
const lockfileHash = await hashFiles(["package-lock.json"]);
const graphSourceHash = await hashFiles([
  "src/components/GraphPreview.tsx",
  "src/services/graphEngineAdapter.ts",
  "src/services/graphRenderer.ts",
  "src/services/graphSpatialIndex.ts",
  "src/services/graphVisibilityController.ts",
  "src/services/graphLayout.worker.ts",
]);
const environment = {
  benchmarkSchemaVersion,
  lockfileHash,
  graphSourceHash,
  timestamp: new Date().toISOString(),
  os: `${os.type()} ${os.release()} ${os.arch()}`,
  cpu: os.cpus()[0]?.model ?? "unknown",
  logicalCores: os.cpus().length,
  memoryBytes: os.totalmem(),
  browser: executablePath ?? "playwright-bundled",
  browserVersion,
  gpuRenderers: [...new Set(raw.map((entry) => entry.gpu).filter(Boolean))],
  g6: packageJson.dependencies?.["@antv/g6"],
  runs,
  suites,
  actionMs,
  readyTimeoutMs,
  cpuReadyTimeoutMs,
  smoke,
  benchmarkScope,
  probeMode,
  cases,
  baselines: baselines.map((baseline) => baseline.id),
  renderers,
  gateRenderer,
};

const grouped = {};
for (const entry of raw) {
  const key = `${entry.baseline}:${entry.rendererRequested}:${entry.caseId}`;
  const group = grouped[key] ?? (grouped[key] = []);
  group.push(entry);
}

const summaries = Object.fromEntries(Object.entries(grouped).map(([key, entries]) => {
  const value = (selector) => percentile(entries.map(selector).filter(Number.isFinite), 0.95);
  const finiteCount = (selector) => entries.map(selector).filter(Number.isFinite).length;
  const phaseValues = (phase) => entries.flatMap((entry) => (
    (entry.runtimeAfter?.performanceSamples ?? [])
      .filter((sample) => sample.phase === phase)
      .map((sample) => sample.durationMs)
  ));
  const expectedMeasurementCount = runs * suites;
  const lifecycleFailureCount = entries.filter((entry) => (
    entry.runtimeBeforeResize?.graphCreateCount !== 1 ||
    entry.runtimeBeforeResize?.graphDestroyCount !== 0 ||
    entry.runtimeAfter?.graphCreateCount !== entry.runtimeBeforeResize?.graphCreateCount ||
    entry.runtimeAfter?.graphDestroyCount !== entry.runtimeBeforeResize?.graphDestroyCount
  )).length;
  const dragFailureCount = entries.filter((entry) => (
    !entry.runtimeAfter?.lastDraggedNodeId ||
    entry.runtimeAfter?.lastDragPinned !== false
  )).length;
  const missingMetricCount = entries.filter((entry) => [
    entry.initialReadyMs,
    entry.drag.frameP95,
    entry.pan.frameP95,
    entry.zoom.frameP95,
    entry.drag.inputLatencyP95,
    entry.pan.inputLatencyP95,
    entry.zoom.inputLatencyP95,
  ].some((metric) => !Number.isFinite(metric))).length;
  const [baselineId, rendererRequested, caseId] = key.split(":");
  const expectedRenderer = expectedResolvedRenderer(rendererRequested, caseId);
  const resolvedRenderers = [...new Set(
    entries.map((entry) => entry.rendererResolved).filter(Boolean),
  )];
  const rendererResolutionFailureCount = entries.filter((entry) => (
    entry.rendererResolved !== expectedRenderer ||
    (expectedRenderer === "hybrid-webgl" && !Number.isFinite(entry.rendererLazyLoadMs))
  )).length;
  const fallbackEntries = entries.filter((entry) => Boolean(entry.rendererFallbackReason));
  const fallbackReasons = Object.fromEntries(
    [...new Set(fallbackEntries.map((entry) => entry.rendererFallbackReason))]
      .map((reason) => [
        reason,
        fallbackEntries.filter((entry) => entry.rendererFallbackReason === reason).length,
      ]),
  );
  const webglContextLossCount = entries.reduce(
    (sum, entry) => sum + (entry.webglContextLossCount ?? 0),
    0,
  );
  const seeds = [...new Set(entries.map((entry) => entry.seed).filter(Boolean))];
  const seedMismatchCount = entries.filter((entry) => !entry.seed).length + (seeds.length > 1 ? entries.length : 0);
  const memorySamples = entries
    .map((entry) => entry.memory?.usedJSHeapSize)
    .filter(Number.isFinite);
  const memoryFirstBytes = memorySamples[0] ?? null;
  const memoryLastBytes = memorySamples.at(-1) ?? null;
  const memoryGrowthBytes = memorySamples.length >= 2
    ? memoryLastBytes - memoryFirstBytes
    : null;
  const memoryGrowthRatio = memorySamples.length >= 2 && memoryFirstBytes > 0
    ? memoryGrowthBytes / memoryFirstBytes
    : null;
  const memoryGate = smoke || memorySamples.length < 2
    ? "unknown"
    : memoryGrowthBytes <= 20 * 1024 * 1024 ||
        (Number.isFinite(memoryGrowthRatio) && memoryGrowthRatio <= 0.15)
      ? "passed"
      : "failed";
  const summary = {
    rendererRequested,
    expectedResolvedRenderer: expectedRenderer,
    resolvedRenderers,
    rendererLazyLoadP95: value((entry) => entry.rendererLazyLoadMs),
    rendererResolutionFailureCount,
    fallbackCount: fallbackEntries.length,
    fallbackReasons,
    webglContextLossCount,
    seeds,
    seedMismatchCount,
    measurementCount: entries.length,
    expectedMeasurementCount,
    initialReadyP95: value((entry) => entry.initialReadyMs),
    dragFrameP95: value((entry) => entry.drag.frameP95),
    panFrameP95: value((entry) => entry.pan.frameP95),
    zoomFrameP95: value((entry) => entry.zoom.frameP95),
    selectionFrameP95: value((entry) => entry.selection.frameP95),
    inputLatencyP95: value((entry) => Math.max(entry.drag.inputLatencyP95, entry.pan.inputLatencyP95, entry.zoom.inputLatencyP95)),
    over33RatioP95: value((entry) => Math.max(entry.drag.over33Ratio, entry.pan.over33Ratio, entry.zoom.over33Ratio)),
    longTaskMax: Math.max(0, ...entries.map((entry) => Math.max(entry.drag.longTaskMax, entry.pan.longTaskMax, entry.zoom.longTaskMax))),
    consoleErrorCount: entries.reduce((sum, entry) => sum + entry.consoleErrors.length + entry.pageErrors.length, 0),
    graphCreateCountMax: Math.max(0, ...entries.map((entry) => entry.runtimeAfter?.graphCreateCount ?? 0)),
    lifecycleFailureCount,
    dragFailureCount,
    missingMetricCount,
    memorySampleCount: memorySamples.length,
    memoryFirstBytes,
    memoryLastBytes,
    memoryGrowthBytes,
    memoryGrowthRatio,
    memoryGate,
    spatialPickP95: value((entry) => entry.runtimeAfter?.spatialPickMs),
    spatialPickCandidatesP95: value((entry) => entry.runtimeAfter?.spatialPickCandidates),
    pickOracleChecked: Math.min(
      ...entries.map((entry) => entry.runtimeAfter?.pickOracleChecked ?? 0),
    ),
    pickOracleMismatchCount: entries.reduce(
      (sum, entry) => sum + (entry.runtimeAfter?.pickOracleMismatches ?? 0),
      0,
    ),
    pickOracleP95: value((entry) => entry.runtimeAfter?.pickOracleP95Ms),
    pickOracleCandidatesP95: value((entry) => entry.runtimeAfter?.pickOracleCandidatesP95),
    workerRoundTripP95: value((entry) => entry.runtimeAfter?.workerRoundTripMs),
    workerComputeP95: value((entry) => entry.runtimeAfter?.workerComputeMs),
    positionApplyP95: value((entry) => entry.runtimeAfter?.positionApplyMs),
    pointerDispatchP95: percentile(phaseValues("pointer_dispatch"), 0.95),
    coordinateTransformP95: percentile(phaseValues("coordinate_transform"), 0.95),
    dragTargetApplyP95: percentile(phaseValues("drag_target_apply"), 0.95),
    dragNeighbourApplyP95: percentile(phaseValues("drag_neighbour_apply"), 0.95),
    visibilityComputeP95: percentile(phaseValues("visibility_compute"), 0.95),
    visibilityApplyP95: percentile(phaseValues("visibility_apply"), 0.95),
    canvasDrawP95: percentile(phaseValues("canvas_draw"), 0.95),
    canvasPresentP95: percentile(phaseValues("canvas_present"), 0.95),
    layerSyncP95: value((entry) => Math.max(
      0,
      ...(entry.runtimeAfter?.performanceSamples ?? [])
        .filter((sample) => sample.phase === "layer_sync")
        .map((sample) => sample.durationMs),
    )),
    cameraMatrixDriftMax: Math.max(0, ...entries.flatMap((entry) => (
      (entry.runtimeAfter?.performanceSamples ?? [])
        .filter((sample) => sample.phase === "layer_sync")
        .map((sample) => Number(sample.detail?.cameraMatrixDrift ?? 0))
    ))),
    webglContextCreateP95: value((entry) => entry.gpuAttribution?.initial?.contextCreateMs),
    initialBufferBytesP95: value((entry) => entry.gpuAttribution?.initial?.bufferBytes),
    nodeBufferPatchBytesP95: value((entry) => entry.gpuAttribution?.nodePatch?.bufferBytes),
    edgeBufferPatchBytesP95: value((entry) => entry.gpuAttribution?.edgePatch?.bufferBytes),
    performanceMetricCoverage: {
      spatialPick: finiteCount((entry) => entry.runtimeAfter?.spatialPickMs),
      workerRoundTrip: finiteCount((entry) => entry.runtimeAfter?.workerRoundTripMs),
      positionApply: finiteCount((entry) => entry.runtimeAfter?.positionApplyMs),
      layerSync: entries.filter((entry) => (
        (entry.runtimeAfter?.performanceSamples ?? []).some((sample) => (
          sample.phase === "layer_sync" && Number.isFinite(sample.durationMs)
        ))
      )).length,
      cameraMatrixDrift: entries.filter((entry) => (
        (entry.runtimeAfter?.performanceSamples ?? []).some((sample) => (
          sample.phase === "layer_sync" && Number.isFinite(Number(sample.detail?.cameraMatrixDrift))
        ))
      )).length,
      canvasDraw: entries.filter((entry) => (
        (entry.runtimeAfter?.performanceSamples ?? []).some((sample) => (
          sample.phase === "canvas_draw" && Number.isFinite(sample.durationMs)
        ))
      )).length,
      canvasPresent: entries.filter((entry) => (
        (entry.runtimeAfter?.performanceSamples ?? []).some((sample) => (
          sample.phase === "canvas_present" && Number.isFinite(sample.durationMs)
        ))
      )).length,
    },
  };
  const readyLimit = caseId === "small" ? 2_000 : caseId === "medium" ? 5_000 : 12_000;
  const native = baselineId === "native";
  return [key, {
    ...summary,
    releaseGate: !native
      ? "trend-only"
      : benchmarkScope !== "full-release"
        ? "not-evaluated"
        : (
      summary.initialReadyP95 <= readyLimit &&
      summary.dragFrameP95 < 20 &&
      summary.panFrameP95 < 20 &&
      summary.zoomFrameP95 < 20 &&
      summary.inputLatencyP95 < 50 &&
      summary.over33RatioP95 <= 0.015 &&
      summary.longTaskMax <= (caseId === "large" ? 150 : 100) &&
      summary.pickOracleChecked >= 200 &&
      summary.pickOracleMismatchCount === 0 &&
      summary.pickOracleP95 < 1 &&
      summary.performanceMetricCoverage.canvasDraw === summary.expectedMeasurementCount &&
      summary.performanceMetricCoverage.canvasPresent === summary.expectedMeasurementCount &&
      summary.consoleErrorCount === 0 &&
      summary.measurementCount === summary.expectedMeasurementCount &&
      summary.lifecycleFailureCount === 0 &&
      summary.dragFailureCount === 0 &&
      summary.memoryGate === "passed" &&
      summary.missingMetricCount === 0 &&
      summary.rendererResolutionFailureCount === 0 &&
      summary.fallbackCount === 0 &&
      summary.webglContextLossCount === 0 &&
      summary.seedMismatchCount === 0
    ) ? "passed" : "failed",
  }];
}));

const rendererComparisons = [];
for (const baseline of baselines) {
  for (const caseId of cases) {
    const canvas = summaries[`${baseline.id}:canvas:${caseId}`];
    if (!canvas) continue;
    for (const renderer of renderers.filter((candidate) => candidate !== "canvas")) {
      const candidate = summaries[`${baseline.id}:${renderer}:${caseId}`];
      if (!candidate) continue;
      const sameSeed = canvas.seeds.length === 1 &&
        candidate.seeds.length === 1 &&
        canvas.seeds[0] === candidate.seeds[0];
      const deltaPercent = (value, reference) => reference > 0
        ? ((value - reference) / reference) * 100
        : null;
      rendererComparisons.push({
        baseline: baseline.id,
        caseId,
        renderer,
        seed: sameSeed ? canvas.seeds[0] : null,
        sameSeed,
        initialReadyDeltaPercent: deltaPercent(candidate.initialReadyP95, canvas.initialReadyP95),
        dragDeltaPercent: deltaPercent(candidate.dragFrameP95, canvas.dragFrameP95),
        panDeltaPercent: deltaPercent(candidate.panFrameP95, canvas.panFrameP95),
        zoomDeltaPercent: deltaPercent(candidate.zoomFrameP95, canvas.zoomFrameP95),
        inputDeltaPercent: deltaPercent(candidate.inputLatencyP95, canvas.inputLatencyP95),
      });
    }
  }
}

const releaseGatesByRenderer = Object.fromEntries(renderers.map((renderer) => {
  const nativeSummaries = Object.entries(summaries).filter(([key]) => (
    key.startsWith(`native:${renderer}:`)
  ));
  const rendererMeasurementFailures = measurementFailures.filter((entry) => (
    entry.baseline === "native" && entry.renderer === renderer
  ));
  const gate = benchmarkScope !== "full-release"
    ? "not-evaluated"
    : rendererMeasurementFailures.length === 0 &&
        nativeSummaries.length === allCases.length &&
        nativeSummaries.every(([, summary]) => summary.releaseGate === "passed")
      ? "passed"
      : "failed";
  return [renderer, gate];
}));
const releaseGate = benchmarkScope !== "full-release"
  ? "not-evaluated"
  : releaseGatesByRenderer[gateRenderer] ?? "failed";
const promotionKeys = ["native:hybrid-webgl:medium", "native:hybrid-webgl:large"];
const promotionMatrixComplete = probeMode === "off" &&
  runs >= 5 && suites >= 2 &&
  promotionKeys.every((key) => summaries[key]);
const promotionGate = !promotionMatrixComplete
  ? "not-evaluated"
  : promotionKeys.every((key) => {
      const summary = summaries[key];
      const caseId = key.endsWith(":large") ? "large" : "medium";
      const readyLimit = caseId === "large" ? 12_000 : 5_000;
      const requiredCoverageComplete = Object.values(summary.performanceMetricCoverage ?? {})
        .every((count) => count === summary.expectedMeasurementCount);
      return requiredCoverageComplete &&
        summary.initialReadyP95 <= readyLimit &&
        summary.dragFrameP95 < 20 &&
        summary.panFrameP95 < 20 &&
        summary.zoomFrameP95 < 20 &&
        summary.inputLatencyP95 < 50 &&
        summary.over33RatioP95 <= 0.015 &&
        summary.longTaskMax <= (caseId === "large" ? 150 : 100) &&
        summary.spatialPickP95 < 1 &&
        summary.workerRoundTripP95 < 33 &&
        summary.positionApplyP95 < 8 &&
        summary.layerSyncP95 < 16.7 &&
        summary.cameraMatrixDriftMax < 1e-4 &&
        summary.consoleErrorCount === 0 &&
        summary.measurementCount === summary.expectedMeasurementCount &&
        summary.lifecycleFailureCount === 0 &&
        summary.dragFailureCount === 0 &&
        summary.memoryGate === "passed" &&
        summary.rendererResolutionFailureCount === 0 &&
        summary.fallbackCount === 0 &&
        summary.webglContextLossCount === 0 &&
        summary.seedMismatchCount === 0;
    })
    ? "passed"
    : "failed";
const decision = benchmarkScope !== "full-release"
  ? "benchmark_only"
  : releaseGate === "passed"
    ? gateRenderer === "canvas"
      ? "keep_canvas"
      : gateRenderer === "hybrid-webgl"
        ? "hybrid_webgl_candidate"
        : "auto_renderer_candidate"
    : gateRenderer === "canvas"
      ? "optimize_canvas"
      : gateRenderer === "hybrid-webgl"
        ? "hybrid_webgl_not_ready"
        : "auto_renderer_not_ready";
const report = {
  runId,
  environment,
  failure,
  measurementFailures,
  releaseGate,
  promotionGate,
  releaseGatesByRenderer,
  decision,
  summaries,
  rendererComparisons,
};

await writeFile(path.join(outputDirectory, "raw.json"), JSON.stringify({ environment, failure, measurementFailures, measurements: raw }, null, 2));
await writeFile(path.join(outputDirectory, "summary.json"), JSON.stringify(report, null, 2));
const csvCell = (value) => {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
};
await writeFile(path.join(outputDirectory, "metrics.csv"), [
  "baseline,renderer_requested,renderer_resolved,renderer_expected,case,seed,lazy_load_p95,fallbacks,fallback_reasons,webgl_context_losses,renderer_resolution_failures,measurements,ready_p95,drag_p95,pan_p95,zoom_p95,selection_p95,input_p95,over_33_ratio,long_task_max,spatial_pick_p95,spatial_candidates_p95,worker_roundtrip_p95,worker_compute_p95,position_apply_p95,layer_sync_p95,camera_matrix_drift,context_create_p95,initial_buffer_bytes,node_patch_bytes,edge_patch_bytes,lifecycle_failures,drag_failures,missing_metrics,memory_samples,memory_first_bytes,memory_last_bytes,memory_growth_bytes,memory_growth_ratio,memory_gate,gate",
  ...Object.entries(summaries).map(([key, summary]) => {
    const [baseline, renderer, caseId] = key.split(":");
    return [
      baseline,
      renderer,
      summary.resolvedRenderers.join("|"),
      summary.expectedResolvedRenderer,
      caseId,
      summary.seeds.join("|"),
      summary.rendererLazyLoadP95,
      summary.fallbackCount,
      JSON.stringify(summary.fallbackReasons),
      summary.webglContextLossCount,
      summary.rendererResolutionFailureCount,
      `${summary.measurementCount}/${summary.expectedMeasurementCount}`,
      summary.initialReadyP95,
      summary.dragFrameP95,
      summary.panFrameP95,
      summary.zoomFrameP95,
      summary.selectionFrameP95,
      summary.inputLatencyP95,
      summary.over33RatioP95,
      summary.longTaskMax,
      summary.spatialPickP95,
      summary.spatialPickCandidatesP95,
      summary.workerRoundTripP95,
      summary.workerComputeP95,
      summary.positionApplyP95,
      summary.layerSyncP95,
      summary.cameraMatrixDriftMax,
      summary.webglContextCreateP95,
      summary.initialBufferBytesP95,
      summary.nodeBufferPatchBytesP95,
      summary.edgeBufferPatchBytesP95,
      summary.lifecycleFailureCount,
      summary.dragFailureCount,
      summary.missingMetricCount,
      summary.memorySampleCount,
      summary.memoryFirstBytes ?? "unknown",
      summary.memoryLastBytes ?? "unknown",
      summary.memoryGrowthBytes ?? "unknown",
      summary.memoryGrowthRatio ?? "unknown",
      summary.memoryGate,
      summary.releaseGate,
    ].map(csvCell).join(",");
  }),
].join("\n"));
const formatDelta = (value) => Number.isFinite(value)
  ? `${value >= 0 ? "+" : ""}${value.toFixed(1)}%`
  : "unknown";
await writeFile(path.join(outputDirectory, "report.md"), [
  "# SocialGraph-FM 图谱性能基准",
  "",
  `- 运行：${runId}`,
  `- 运行范围：**${benchmarkScope}**`,
  `- 所选案例：${cases.join(", ")}`,
  `- 所选基线：${baselines.map((baseline) => baseline.id).join(", ")}`,
  `- 请求渲染器：${renderers.join(", ")}`,
  `- 发布门槛渲染器：**${gateRenderer}**`,
  `- 性能探针：**${probeMode}**${probeMode === "gpu" ? "（诊断运行，不参与帧率发布门槛）" : ""}`,
  `- Canvas profile：**${profileMode}**${profileMode === "canvas" ? "（独立诊断，不参与发布评分）" : ""}`,
  `- 测量配置：${runs} runs × ${suites} suites${smoke ? "（smoke）" : ""}`,
  `- 发布门槛：**${releaseGate}**`,
  `- 自动 WebGL 推广门槛（中/大图）：**${promotionGate}**`,
  `- 分渲染器门槛：${Object.entries(releaseGatesByRenderer).map(([renderer, gate]) => `${renderer}=${gate}`).join("；")}`,
  `- 决策：**${decision}**`,
  `- 环境：${environment.os}；${environment.cpu}；${Math.round(environment.memoryBytes / 1024 / 1024 / 1024)} GB`,
  benchmarkScope === "full-release"
    ? "- 本报告覆盖完整发布验收矩阵。"
    : "- 本报告仅覆盖所选范围，不代表完整发布验收。",
  "",
  "| 基线 | 请求/实际渲染器 | 案例 | 样本 | 懒加载 P95 | 回退 | Context loss | Ready P95 | Drag P95 | Pan P95 | Zoom P95 | Select P95 | Input P95 | >33ms | Long task | 生命周期异常 | 拖动失败 | 堆增长 | 内存门槛 | 结论 |",
  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
  ...Object.entries(summaries).map(([key, summary]) => {
    const [baseline, renderer, caseId] = key.split(":");
    const memoryGrowth = summary.memoryGrowthBytes === null
      ? "unknown"
      : `${(summary.memoryGrowthBytes / 1024 / 1024).toFixed(2)}MB / ${summary.memoryGrowthRatio === null ? "unknown" : `${(summary.memoryGrowthRatio * 100).toFixed(2)}%`}`;
    const resolved = summary.resolvedRenderers.length > 0
      ? summary.resolvedRenderers.join("/")
      : "unreported";
    return `| ${baseline} | ${renderer} / ${resolved} | ${caseId} | ${summary.measurementCount}/${summary.expectedMeasurementCount} | ${summary.rendererLazyLoadP95.toFixed(1)}ms | ${summary.fallbackCount} | ${summary.webglContextLossCount} | ${summary.initialReadyP95.toFixed(1)}ms | ${summary.dragFrameP95.toFixed(1)}ms | ${summary.panFrameP95.toFixed(1)}ms | ${summary.zoomFrameP95.toFixed(1)}ms | ${summary.selectionFrameP95.toFixed(1)}ms | ${summary.inputLatencyP95.toFixed(1)}ms | ${(summary.over33RatioP95 * 100).toFixed(2)}% | ${summary.longTaskMax.toFixed(1)}ms | ${summary.lifecycleFailureCount} | ${summary.dragFailureCount} | ${memoryGrowth} | ${summary.memoryGate} | ${summary.releaseGate} |`;
  }),
  "",
  "## 图引擎分阶段指标",
  "",
  "GPU probe 记录的是 CPU 调用与提交字节，不等同于 GPU 执行时间；节点/边上传量来自隔离 patch。",
  "",
  "| 基线 | 渲染器 | 案例 | Pick P95 | 候选数 P95 | Pointer P95 | 坐标转换 P95 | 目标提交 P95 | 邻居提交 P95 | 可见性计算 P95 | 可见性提交 P95 | Worker RTT P95 | 位置提交 P95 | 图层同步 P95 |",
  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ...Object.entries(summaries).map(([key, summary]) => {
    const [baseline, renderer, caseId] = key.split(":");
    return `| ${baseline} | ${renderer} | ${caseId} | ${summary.spatialPickP95.toFixed(2)}ms | ${summary.spatialPickCandidatesP95.toFixed(1)} | ${summary.pointerDispatchP95.toFixed(2)}ms | ${summary.coordinateTransformP95.toFixed(2)}ms | ${summary.dragTargetApplyP95.toFixed(2)}ms | ${summary.dragNeighbourApplyP95.toFixed(2)}ms | ${summary.visibilityComputeP95.toFixed(2)}ms | ${summary.visibilityApplyP95.toFixed(2)}ms | ${summary.workerRoundTripP95.toFixed(2)}ms | ${summary.positionApplyP95.toFixed(2)}ms | ${summary.layerSyncP95.toFixed(2)}ms |`;
  }),
  ...(rendererComparisons.length > 0 ? [
    "",
    "## 同 Seed Canvas 对照",
    "",
    "负数表示相对 Canvas 更快。只有 `same seed` 为 yes 的数据才是有效 A/B 对照。",
    "",
    "| 基线 | 案例 | 候选渲染器 | Same seed | Ready Δ | Drag Δ | Pan Δ | Zoom Δ | Input Δ |",
    "|---|---|---|---|---:|---:|---:|---:|---:|",
    ...rendererComparisons.map((comparison) => (
      `| ${comparison.baseline} | ${comparison.caseId} | ${comparison.renderer} | ${comparison.sameSeed ? "yes" : "no"} | ${formatDelta(comparison.initialReadyDeltaPercent)} | ${formatDelta(comparison.dragDeltaPercent)} | ${formatDelta(comparison.panDeltaPercent)} | ${formatDelta(comparison.zoomDeltaPercent)} | ${formatDelta(comparison.inputDeltaPercent)} |`
    )),
  ] : []),
  measurementFailures.length > 0 ? "\n## 未完成的测量\n" : "",
  ...measurementFailures.map((entry) => (
    `- ${entry.baseline}/${entry.renderer}/${entry.caseId} suite ${entry.suite} ${entry.phase}: ${entry.error.split("\n")[0]}`
  )),
  failure ? `\n## 执行失败\n\n\`\`\`text\n${failure}\n\`\`\`` : "",
].join("\n"));

const finishedAt = new Date().toISOString();
await atomicJson(path.join(outputDirectory, "run-status.json"), {
  runId,
  status: failure ? "failed" : "completed",
  failure,
  releaseGate,
  finishedAt,
});
await atomicJson(path.join(outputDirectory, "manifest.json"), {
  benchmarkSchemaVersion,
  runId,
  status: failure ? "failed" : "completed",
  startedAt: runStartedAt,
  finishedAt,
  matrix: {
    baselines: baselines.map((baseline) => baseline.id),
    renderers,
    cases,
    suites,
    runs,
  },
  actionMs,
  readyTimeoutMs,
  cpuReadyTimeoutMs,
  probeMode,
  profileMode,
  scenarioTimeoutMs,
  actionTimeoutMs,
  evaluateTimeoutMs,
  cleanupTimeoutMs,
  faultInjection: faultInjection ?? null,
  measurementCount: raw.length,
  failureCount: measurementFailures.length,
});
await appendEvent({
  event: "run_finished",
  status: failure ? "failed" : "completed",
  releaseGate,
});

console.log(JSON.stringify({
  outputDirectory,
  renderers,
  gateRenderer,
  releaseGate,
  releaseGatesByRenderer,
  decision,
  failure,
}, null, 2));
if (failure || (benchmarkScope === "full-release" && releaseGate !== "passed")) process.exitCode = 1;
