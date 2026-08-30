import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  createNeverSettlingStage,
  withStageDeadline,
} from "./benchmark-stage-control.mjs";

const specPath = process.argv[2];
if (!specPath) throw new Error("A benchmark case specification path is required");
const spec = JSON.parse(await readFile(specPath, "utf8"));

const faultStages = {
  ready: { stage: "ready", timeoutKey: "readyTimeoutMs", keepProcessAlive: false },
  action: { stage: "drag", timeoutKey: "actionTimeoutMs", keepProcessAlive: false },
  raf: { stage: "drag:capture-stop", timeoutKey: "evaluateTimeoutMs", keepProcessAlive: false },
  cleanup: { stage: "cleanup:browser", timeoutKey: "cleanupTimeoutMs", keepProcessAlive: true },
};
const injection = faultStages[spec.faultInjection];
if (!injection) throw new Error(`Unsupported benchmark fault injection: ${spec.faultInjection}`);

const atomicJson = async (target, value) => {
  await mkdir(path.dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.tmp`;
  await writeFile(temporary, JSON.stringify(value, null, 2));
  await rename(temporary, target);
};

const stages = [];
const noteStage = (status, detail = {}) => stages.push({
  stage: injection.stage,
  status,
  at: new Date().toISOString(),
  ...detail,
});

let failure;
const fault = createNeverSettlingStage({ keepProcessAlive: injection.keepProcessAlive });
try {
  noteStage("started", { faultInjection: spec.faultInjection });
  await withStageDeadline(
    injection.stage,
    () => fault.promise,
    spec[injection.timeoutKey],
  );
} catch (error) {
  failure = error;
  noteStage("timed_out", { code: error?.code ?? "CASE_FAILED" });
}

const result = {
  // cleanup models the dangerous real-world ordering where a worker can write
  // a successful measurement before browser/context teardown hangs. The
  // parent watchdog must overwrite this provisional success as a timeout.
  status: spec.faultInjection === "cleanup" ? "completed" : "failed",
  baseline: spec.baseline,
  renderer: spec.renderer,
  caseId: spec.caseId,
  suite: spec.suite,
  run: spec.run,
  faultInjection: spec.faultInjection,
  error: failure instanceof Error ? `${failure.stack ?? failure.message}` : String(failure),
  code: failure?.code ?? "FAULT_INJECTION_FAILED",
  stage: failure?.stage ?? injection.stage,
  stages,
};
await atomicJson(spec.resultPath, result);
process.exitCode = 1;

// ready/action/rAF faults have no active handle and exit after their hard
// deadline. cleanup deliberately keeps a handle alive so the parent scenario
// watchdog must terminate the worker after the failed result is durable.
if (!injection.keepProcessAlive) fault.release();
