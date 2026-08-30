import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";

const inputs = process.argv.slice(2);
if (inputs.length !== 4) {
  throw new Error(
    "Usage: node scripts/promote-graph-renderer.mjs <performance-run-1> <gpu-run-1> <performance-run-2> <gpu-run-2>",
  );
}

const summaryPath = async (input) => {
  const resolved = path.resolve(input);
  return (await stat(resolved)).isDirectory() ? path.join(resolved, "summary.json") : resolved;
};
const load = async (input) => JSON.parse(await readFile(await summaryPath(input), "utf8"));
const [first, firstGpu, second, secondGpu] = await Promise.all(inputs.map(load));
const reports = [first, firstGpu, second, secondGpu];
const reportRunIds = reports.map((report) => report.runId).filter(Boolean);

if (reportRunIds.length !== reports.length || new Set(reportRunIds).size !== reports.length) {
  throw new Error("Promotion requires four distinct, non-empty performance and diagnostic run IDs");
}
if (first.environment?.probeMode !== "off" || second.environment?.probeMode !== "off") {
  throw new Error("Performance promotion reports must be recorded with probe=off");
}

const signatureParts = (report) => {
  const environment = report.environment ?? {};
  return {
    benchmarkSchemaVersion: environment.benchmarkSchemaVersion,
    lockfileHash: environment.lockfileHash,
    graphSourceHash: environment.graphSourceHash,
    browserMajor: String(environment.browserVersion ?? "").split(".")[0],
    gpuRenderers: [...(environment.gpuRenderers ?? [])].sort(),
  };
};
const signatures = [first, firstGpu, second, secondGpu].map(signatureParts);
const requiredSignature = signatures[0];
if (
  !requiredSignature.benchmarkSchemaVersion ||
  !requiredSignature.lockfileHash ||
  !requiredSignature.graphSourceHash ||
  !requiredSignature.browserMajor ||
  requiredSignature.gpuRenderers.length === 0 ||
  requiredSignature.gpuRenderers.includes("unavailable")
) {
  throw new Error("Promotion reports are missing a stable benchmark, source, browser, or GPU signature");
}
if (signatures.some((signature) => JSON.stringify(signature) !== JSON.stringify(signatures[0]))) {
  throw new Error("Promotion reports do not share the same source, lockfile, browser major version, and GPU renderer");
}
if (first.promotionGate !== "passed" || second.promotionGate !== "passed") {
  throw new Error("Both consecutive native medium/large performance reports must pass promotionGate");
}
if (firstGpu.environment?.probeMode !== "gpu" || secondGpu.environment?.probeMode !== "gpu") {
  throw new Error("Each performance report requires a matching probe=gpu diagnostic report");
}

const diagnosticKeys = ["native:hybrid-webgl:medium", "native:hybrid-webgl:large"];
const performanceReports = [first, second];
for (const report of performanceReports) {
  for (const key of diagnosticKeys) {
    const summary = report.summaries?.[key];
    if (!summary) throw new Error(`Missing performance summary ${key}`);
    const expected = Number(summary.expectedMeasurementCount);
    const coverage = summary.performanceMetricCoverage ?? {};
    for (const metric of ["spatialPick", "workerRoundTrip", "positionApply", "layerSync", "cameraMatrixDrift"]) {
      if (!Number.isFinite(expected) || expected <= 0 || coverage[metric] !== expected) {
        throw new Error(`${report.runId}/${key} has incomplete ${metric} coverage`);
      }
    }
  }
}
for (const key of diagnosticKeys) {
  const left = firstGpu.summaries?.[key];
  const right = secondGpu.summaries?.[key];
  if (!left || !right) throw new Error(`Missing GPU diagnostic summary ${key}`);
  if (
    !Number.isFinite(left.webglContextCreateP95) ||
    !Number.isFinite(right.webglContextCreateP95) ||
    left.webglContextCreateP95 <= 0 ||
    right.webglContextCreateP95 <= 0 ||
    left.webglContextCreateP95 >= 1_000 ||
    right.webglContextCreateP95 >= 1_000
  ) {
    throw new Error(`${key} WebGL context creation must remain below 1000ms`);
  }
  if (left.webglContextLossCount !== 0 || right.webglContextLossCount !== 0) {
    throw new Error(`${key} reported a WebGL context loss`);
  }
  for (const metric of ["initialBufferBytesP95", "nodeBufferPatchBytesP95", "edgeBufferPatchBytesP95"]) {
    const baseline = Number(left[metric]);
    const candidate = Number(right[metric]);
    if (!Number.isFinite(baseline) || !Number.isFinite(candidate) || baseline <= 0) {
      throw new Error(`${key} is missing ${metric}`);
    }
    if ((candidate - baseline) / baseline > 0.15) {
      throw new Error(`${key} ${metric} regressed by more than 15%`);
    }
  }
}

const benchmarkSignature = createHash("sha256")
  .update(JSON.stringify(signatures[0]))
  .digest("hex");
const policy = {
  schemaVersion: "1.0",
  benchmarkSchemaVersion: signatures[0].benchmarkSchemaVersion,
  autoWebglEnabled: true,
  approvedRunIds: [first.runId, second.runId],
  benchmarkSignature,
};
await writeFile(
  path.resolve("src/config/graph-renderer-policy.json"),
  `${JSON.stringify(policy, null, 2)}\n`,
  "utf8",
);
console.log(JSON.stringify(policy, null, 2));
