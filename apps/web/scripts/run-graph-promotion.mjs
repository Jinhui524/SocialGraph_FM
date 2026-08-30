import { spawn } from "node:child_process";
import path from "node:path";
import { waitForChildWithDeadline } from "./benchmark-watchdog.mjs";

const setId = `promotion-${new Date().toISOString().replaceAll(":", "-").replaceAll(".", "-")}`;
const benchmarkScript = path.resolve("scripts/run-graph-benchmark.mjs");
const promotionScript = path.resolve("scripts/promote-graph-renderer.mjs");
const baseEnvironment = {
  ...process.env,
  SGFM_BENCHMARK_CASES: "medium,large",
  SGFM_BENCHMARK_BASELINES: "native",
};

const runProcess = async (command, args, options, timeoutMs, label) => {
  const child = spawn(command, args, { ...options, windowsHide: true });
  const outcome = await waitForChildWithDeadline(child, timeoutMs);
  if (outcome.timedOut) {
    throw new Error(`${label} exceeded its ${timeoutMs}ms promotion watchdog`);
  }
  if (outcome.error) throw outcome.error;
  if (outcome.code !== 0) {
    throw new Error(`${label} failed with exit code ${outcome.code ?? "unknown"}`);
  }
};

const run = async (id, args, extraEnvironment = {}, timeoutMs = 4 * 60 * 60 * 1_000) => {
  await runProcess(process.execPath, [benchmarkScript, ...args], {
    cwd: process.cwd(),
    env: { ...baseEnvironment, SGFM_BENCHMARK_RUN_ID: id, ...extraEnvironment },
    stdio: "inherit",
  }, timeoutMs, id);
  return path.resolve("artifacts", "benchmarks", "graph", id);
};

const perf1 = await run(
  `${setId}-perf-1`,
  ["--renderers=canvas,hybrid-webgl,auto", "--gate-renderer=hybrid-webgl"],
);
const gpu1 = await run(
  `${setId}-gpu-1`,
  ["--renderer=hybrid-webgl", "--gate-renderer=hybrid-webgl", "--probe=gpu"],
  { SGFM_BENCHMARK_RUNS: "1", SGFM_BENCHMARK_SUITES: "1", SGFM_BENCHMARK_ACTION_MS: "300" },
  30 * 60 * 1_000,
);
const perf2 = await run(
  `${setId}-perf-2`,
  ["--renderers=canvas,hybrid-webgl,auto", "--gate-renderer=hybrid-webgl"],
);
const gpu2 = await run(
  `${setId}-gpu-2`,
  ["--renderer=hybrid-webgl", "--gate-renderer=hybrid-webgl", "--probe=gpu"],
  { SGFM_BENCHMARK_RUNS: "1", SGFM_BENCHMARK_SUITES: "1", SGFM_BENCHMARK_ACTION_MS: "300" },
  30 * 60 * 1_000,
);

await runProcess(process.execPath, [promotionScript, perf1, gpu1, perf2, gpu2], {
  cwd: process.cwd(),
  stdio: "inherit",
}, 5 * 60 * 1_000, "renderer promotion");
console.log(`Auto WebGL policy promoted from ${setId}`);
