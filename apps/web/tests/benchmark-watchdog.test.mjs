import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtemp, readFile, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { withStageDeadline } from "../scripts/benchmark-stage-control.mjs";
import { waitForChildWithDeadline } from "../scripts/benchmark-watchdog.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const benchmarkRunnerPath = path.join(projectRoot, "scripts", "run-graph-benchmark.mjs");

test("watchdog terminates a hung isolated benchmark worker", async () => {
  const startedAt = Date.now();
  const child = spawn(process.execPath, ["-e", "setInterval(() => {}, 1000)"], {
    windowsHide: true,
    stdio: "ignore",
  });
  const outcome = await waitForChildWithDeadline(child, 200);
  assert.equal(outcome.timedOut, true);
  assert.ok(Date.now() - startedAt < 5_000);
});

test("watchdog preserves a normally completed worker", async () => {
  const child = spawn(process.execPath, ["-e", "process.exit(0)"], {
    windowsHide: true,
    stdio: "ignore",
  });
  const outcome = await waitForChildWithDeadline(child, 5_000);
  assert.equal(outcome.timedOut, false);
  assert.equal(outcome.code, 0);
});

test("stage deadline rejects a never-settling action with auditable metadata", async () => {
  const startedAt = Date.now();
  await assert.rejects(
    withStageDeadline("ready", () => new Promise(() => undefined), 25),
    (error) => {
      assert.equal(error.code, "STAGE_TIMEOUT");
      assert.equal(error.stage, "ready");
      assert.equal(error.timeoutMs, 25);
      return true;
    },
  );
  assert.ok(Date.now() - startedAt < 1_000);
});

test("fault-injected ready/action/rAF/cleanup hangs terminate and retain all reports", async () => {
  const faultRoot = await mkdtemp(path.join(tmpdir(), "sgfm-benchmark-faults-"));
  const expectedStage = {
    ready: "ready",
    action: "drag",
    raf: "drag:capture-stop",
    cleanup: "cleanup:browser",
  };
  try {
    for (const injection of Object.keys(expectedStage)) {
      const runId = `fault-${injection}`;
      const child = spawn(process.execPath, [benchmarkRunnerPath, "--smoke"], {
        cwd: projectRoot,
        env: {
          ...process.env,
          SGFM_BENCHMARK_ACTION_MS: "1",
          SGFM_BENCHMARK_ACTION_TIMEOUT_MS: "35",
          SGFM_BENCHMARK_BASELINES: "native",
          SGFM_BENCHMARK_CASES: "small",
          SGFM_BENCHMARK_CLEANUP_TIMEOUT_MS: "35",
          SGFM_BENCHMARK_EVALUATE_TIMEOUT_MS: "35",
          SGFM_BENCHMARK_FAULT_INJECTION: injection,
          SGFM_BENCHMARK_OUTPUT_ROOT: faultRoot,
          SGFM_BENCHMARK_READY_TIMEOUT_MS: "35",
          SGFM_BENCHMARK_RENDERERS: "canvas",
          SGFM_BENCHMARK_RUN_ID: runId,
          SGFM_BENCHMARK_RUNS: "1",
          SGFM_BENCHMARK_SCENARIO_TIMEOUT_MS: "250",
          SGFM_BENCHMARK_SUITES: "1",
        },
        windowsHide: true,
        stdio: "ignore",
      });
      const startedAt = Date.now();
      const outcome = await waitForChildWithDeadline(child, 10_000);
      assert.equal(outcome.timedOut, false, `${injection} runner exceeded the outer test watchdog`);
      assert.equal(outcome.code, 1, `${injection} runner must fail closed`);
      assert.ok(Date.now() - startedAt < 8_000, `${injection} runner did not terminate promptly`);

      const runDirectory = path.join(faultRoot, runId);
      const caseFiles = await readdir(path.join(runDirectory, "case-results"));
      assert.equal(caseFiles.length, 1, `${injection} case result is missing`);
      const [caseResult, summary, runStatus, report] = await Promise.all([
        readFile(path.join(runDirectory, "case-results", caseFiles[0]), "utf8").then(JSON.parse),
        readFile(path.join(runDirectory, "summary.json"), "utf8").then(JSON.parse),
        readFile(path.join(runDirectory, "run-status.json"), "utf8").then(JSON.parse),
        readFile(path.join(runDirectory, "report.md"), "utf8"),
      ]);
      assert.equal(caseResult.status, "failed");
      assert.equal(caseResult.code, injection === "cleanup" ? "SCENARIO_TIMEOUT" : "STAGE_TIMEOUT");
      assert.equal(caseResult.stage, expectedStage[injection]);
      if (injection === "cleanup") assert.equal(caseResult.workerResultStatus, "completed");
      assert.equal(summary.runId, runId);
      assert.ok(summary.failure);
      assert.equal(runStatus.runId, runId);
      assert.equal(runStatus.status, "failed");
      assert.match(report, /执行失败/u);
    }
  } finally {
    await rm(faultRoot, { recursive: true, force: true });
  }
});
