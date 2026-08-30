import { expect, test, type Locator, type Page, type Route } from "@playwright/test";
import { Buffer } from "node:buffer";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  targetCapabilities,
  targetComparison,
  targetEvidence,
  targetPolicy,
  targetPreview,
  targetResult,
  targetRun,
  targetTaskRegistration,
} from "../src/test/fixtures/governanceTargetTask";
import { onlineHealth } from "../src/test/fixtures/governanceOnline";
import { globalModelModelCard } from "../src/test/fixtures/globalModel";
import { sha256Canonical } from "../src/services/graphIdentity";

const BANNED_COPY = /In-domain|Low-label|Cross-domain|RQ4|常态化巡检|低标注快速启动|新型活动冷启动|多域联合治理|zero_shot|few_shot|AdaptationReviewPolicy|样例|演示|实验平台|概率|准确率|Macro-F1/iu;
const evidenceRoot = resolve(process.cwd(), "../../var/qa/task-5");

type LaneKey = "zero" | "few";

interface LaneRouteCounts {
  targetTasks: number;
  createRun: number;
  pollRun: number;
  result: number;
  runPreview: number;
  evidence: number;
  labelSet: number;
  fit: number;
  policy: number;
  comparison: number;
  activation: number;
  handoff: number;
  paths: string[];
}

interface RouteCounts {
  zero: LaneRouteCounts;
  few: LaneRouteCounts;
  crossLaneAlias: number;
}

interface LaneBundle {
  lane: LaneKey;
  registration: ReturnType<typeof targetTaskRegistration>;
  preview: ReturnType<typeof targetPreview>;
  scoredPreview: ReturnType<typeof targetPreview>;
  run: ReturnType<typeof targetRun>;
  result: ReturnType<typeof targetResult>;
  evidence: ReturnType<typeof targetEvidence>;
  policy: ReturnType<typeof targetPolicy>;
  comparison: ReturnType<typeof targetComparison>;
  activation: ReturnType<typeof targetActivationPayload>;
  handoff: ReturnType<typeof targetHandoffPayload>;
}

const copy = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const laneHash = (lane: LaneKey, field: string) => sha256Canonical({ lane, field });
const laneNodeId = (lane: LaneKey, index: number) => `${lane}-node-${String(index + 1).padStart(3, "0")}`;

function targetActivationPayload(bundle: Omit<LaneBundle, "activation" | "handoff">) {
  const logical = {
    schemaVersion: "socialgraph-fm.governance-adaptation-overlay/1.0" as const,
    targetTaskRegistrationId: bundle.registration.registrationId,
    targetReceiptHash: bundle.registration.targetReceipt.receiptHash,
    labelSetHash: bundle.policy.labelSetHash,
    binding: bundle.policy.binding,
    policyHash: bundle.policy.policyHash,
    comparisonHash: bundle.comparison.comparisonHash,
    active: true as const,
    baseModelMutation: false as const,
  };
  return { ...logical, activationHash: sha256Canonical(logical) };
}

function targetHandoffPayload(bundle: Omit<LaneBundle, "activation" | "handoff">) {
  const logical = {
    schemaVersion: "socialgraph-fm.governance-adaptation-handoff/1.0" as const,
    targetTaskRegistrationId: bundle.registration.registrationId,
    targetReceiptHash: bundle.registration.targetReceipt.receiptHash,
    labelSetHash: bundle.policy.labelSetHash,
    binding: bundle.policy.binding,
    policyHash: bundle.policy.policyHash,
    comparisonHash: bundle.comparison.comparisonHash,
    decision: "pending_governance_review" as const,
    baseModelMutation: false as const,
  };
  return { ...logical, handoffHash: sha256Canonical(logical) };
}

function createLaneBundle(lane: LaneKey): LaneBundle {
  const mode = lane === "zero" ? "zero_shot" : "few_shot";
  const idCharacter = lane === "zero" ? "a" : "b";
  const registration = copy(targetTaskRegistration(mode));
  const taskId = `regional-${lane}-immutable`;
  const registrationId = `target-task-${idCharacter.repeat(32)}`;
  const artifactId = `governance-artifact-${idCharacter.repeat(32)}`;
  const runId = `governance-${idCharacter.repeat(32)}`;
  const graphVersionHash = laneHash(lane, "graph-version");
  const datasetContentHash = laneHash(lane, "dataset-content");
  const inferenceHash = laneHash(lane, "inference");
  const resultHash = laneHash(lane, "result");
  const labelSetHash = laneHash(lane, "label-set");

  Object.assign(registration, {
    registrationId,
    outerBundleSha256: laneHash(lane, "outer-bundle"),
    registrationHash: laneHash(lane, "registration"),
  });
  Object.assign(registration.artifact, {
    artifactId,
    datasetId: `target-${lane}`,
    displayName: lane === "zero" ? "Zero immutable target" : "Few immutable target",
    bundleSha256: inferenceHash,
    manifestSha256: laneHash(lane, "artifact-manifest"),
    datasetContentHash,
    graphVersionHash,
    artifactHash: laneHash(lane, "artifact"),
  });
  Object.assign(registration.task, {
    taskId,
    displayName: registration.artifact.displayName,
    inference: { ...registration.task.inference, sha256: inferenceHash },
    targetReceipt: { ...registration.task.targetReceipt, sha256: laneHash(lane, "target-receipt-file") },
    labels: registration.task.labels ? { ...registration.task.labels, sha256: laneHash(lane, "labels-file") } : null,
    labelReceipt: registration.task.labelReceipt ? { ...registration.task.labelReceipt, sha256: laneHash(lane, "label-receipt-file") } : null,
  });
  Object.assign(registration.targetReceipt, {
    taskId,
    sourceContentHash: laneHash(lane, "source-content"),
    sourceManifestSha256: laneHash(lane, "source-manifest"),
    inferenceSha256: inferenceHash,
    nodeSetSha256: laneHash(lane, "node-set"),
    receiptHash: laneHash(lane, "target-receipt"),
  });
  if (registration.labels && registration.labelReceipt) {
    registration.labels = {
      ...registration.labels,
      taskId,
      inferenceSha256: inferenceHash,
      labelSetHash,
      labels: registration.labels.labels.map((row, index) => ({ ...row, nodeId: laneNodeId(lane, index) })),
    };
    registration.labelReceipt = {
      ...registration.labelReceipt,
      taskId,
      targetReceiptHash: registration.targetReceipt.receiptHash,
      labelsSha256: laneHash(lane, "labels-file"),
      sourceLabelsSha256: laneHash(lane, "source-labels"),
      eligibilityMaskSha256: registration.targetReceipt.labelEligibilityMaskSha256!,
      eligibleNodeIds: registration.labels.labels.map((row) => row.nodeId),
      receiptHash: laneHash(lane, "label-receipt"),
    };
  }

  const adaptPreview = (scored: boolean) => {
    const preview = copy(targetPreview(scored));
    const idMap = new Map(preview.nodes.map((node, index) => [node.id, laneNodeId(lane, index)]));
    return {
      ...preview,
      artifactId,
      datasetContentHash,
      graphVersionHash,
      runId: scored ? runId : null,
      resultHash: scored ? resultHash : null,
      previewHash: laneHash(lane, scored ? "scored-preview" : "raw-preview"),
      nodes: preview.nodes.map((node, index) => ({ ...node, id: laneNodeId(lane, index), label: `${lane} object ${index + 1}` })),
      edges: preview.edges.map((edge, index) => ({
        ...edge,
        id: `${lane}-edge-${String(index + 1).padStart(3, "0")}`,
        source: idMap.get(edge.source)!,
        target: idMap.get(edge.target)!,
      })),
    };
  };

  const run = {
    ...copy(targetRun()),
    runId,
    requestHash: laneHash(lane, "run-request"),
    artifactId,
    datasetContentHash,
    graphVersionHash,
    statusHash: laneHash(lane, "run-status"),
  };
  const result = {
    ...copy(targetResult()),
    runId,
    requestHash: run.requestHash,
    artifactId,
    datasetContentHash,
    graphVersionHash,
    resultHash,
    findings: targetResult().findings.map((finding, index) => ({ ...finding, nodeId: laneNodeId(lane, index), label: `${lane} object ${index + 1}` })),
  };
  const policy = {
    ...copy(targetPolicy()),
    binding: {
      ...copy(targetPolicy().binding),
      artifactId,
      datasetContentHash,
      graphVersionHash,
      runId,
      requestHash: run.requestHash,
      resultHash,
      runArtifactHash: laneHash(lane, "run-artifact"),
      recipeHash: laneHash(lane, "recipe"),
      codeHash: laneHash(lane, "code"),
    },
    labelSetHash,
    policyHash: laneHash(lane, "policy"),
  };
  const comparison = {
    ...copy(targetComparison()),
    binding: policy.binding,
    policyHash: policy.policyHash,
    rows: targetComparison().rows.map((row, index) => ({ ...row, nodeId: laneNodeId(lane, index) })),
    comparisonHash: laneHash(lane, "comparison"),
  };
  const evidence = {
    ...copy(targetEvidence(laneNodeId(lane, 0))),
    runId,
    resultHash,
    artifactId,
    datasetContentHash,
    graphVersionHash,
    node: { ...targetEvidence().node, nodeId: laneNodeId(lane, 0), label: `${lane} object 1` },
    evidenceHash: laneHash(lane, "evidence"),
  };
  const partial = { lane, registration, preview: adaptPreview(false), scoredPreview: adaptPreview(true), run, result, evidence, policy, comparison };
  return { ...partial, activation: targetActivationPayload(partial), handoff: targetHandoffPayload(partial) };
}

const ZERO_BUNDLE = createLaneBundle("zero");
const FEW_BUNDLE = createLaneBundle("few");

function emptyLaneCounts(): LaneRouteCounts {
  return { targetTasks: 0, createRun: 0, pollRun: 0, result: 0, runPreview: 0, evidence: 0, labelSet: 0, fit: 0, policy: 0, comparison: 0, activation: 0, handoff: 0, paths: [] };
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockTargetAdaptation(page: Page, counts: RouteCounts) {
  await page.route("**/api/v1/gfm/global-model/model-card", async (route) => {
    const base = copy(globalModelModelCard());
    const logical = {
      ...base,
      modelVersionId: ZERO_BUNDLE.result.modelVersionId,
      modelVersionHash: ZERO_BUNDLE.result.modelVersionHash,
      protocols: {
        ...base.protocols,
        global: {
          ...base.protocols.global,
          modelVersionId: ZERO_BUNDLE.result.modelVersionId,
          modelVersionHash: ZERO_BUNDLE.result.modelVersionHash,
          modelStateHash: ZERO_BUNDLE.result.modelStateHash,
        },
      },
    };
    const { modelCardHash: _ignored, ...withoutHash } = logical;
    return json(route, { ...withoutHash, modelCardHash: sha256Canonical(withoutHash) });
  });
  await page.route("**/api/v2/gfm/governance/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const method = request.method();
    if (method === "GET" && path.endsWith("/health")) return json(route, onlineHealth());
    if (method === "GET" && path.endsWith("/capabilities")) return json(route, targetCapabilities());
    if (method === "GET" && path.endsWith("/runs")) return json(route, { schemaVersion: "socialgraph-fm.gfm-governance/2.0", items: [], total: 0, offset: 0, limit: 50 });
    if (method === "POST" && path.endsWith("/target-tasks")) {
      const lane: LaneKey = request.postData()?.includes("regional-zero") ? "zero" : "few";
      counts[lane].targetTasks += 1; counts[lane].paths.push(`${method} ${path}`);
      return json(route, lane === "zero" ? ZERO_BUNDLE.registration : FEW_BUNDLE.registration, 201);
    }
    if (method === "GET" && /\/artifacts\/[^/]+\/preview$/u.test(path)) {
      const lane: LaneKey = path.includes(ZERO_BUNDLE.registration.artifact.artifactId) ? "zero" : "few";
      counts[lane].paths.push(`${method} ${path}`);
      return json(route, lane === "zero" ? ZERO_BUNDLE.preview : FEW_BUNDLE.preview);
    }
    if (method === "POST" && path.endsWith("/runs")) {
      const payload = request.postDataJSON() as { artifactId: string };
      const lane: LaneKey = payload.artifactId === ZERO_BUNDLE.run.artifactId ? "zero" : "few";
      const bundle = lane === "zero" ? ZERO_BUNDLE : FEW_BUNDLE;
      counts[lane].createRun += 1; counts[lane].paths.push(`${method} ${path}`);
      return json(route, { ...bundle.run, status: "running", stage: "inferencing", progress: 65 }, 201);
    }
    for (const [lane, bundle] of [["zero", ZERO_BUNDLE], ["few", FEW_BUNDLE]] as const) {
      if (method === "GET" && path.endsWith(`/runs/${bundle.run.runId}`)) { counts[lane].pollRun += 1; counts[lane].paths.push(`${method} ${path}`); return json(route, bundle.run); }
      if (method === "GET" && path.endsWith(`/runs/${bundle.run.runId}/result`)) { counts[lane].result += 1; counts[lane].paths.push(`${method} ${path}`); return json(route, bundle.result); }
      if (method === "GET" && path.endsWith(`/runs/${bundle.run.runId}/graph-preview`)) { counts[lane].runPreview += 1; counts[lane].paths.push(`${method} ${path}`); return json(route, bundle.scoredPreview); }
    }
    if (method === "GET" && /\/runs\/[^/]+\/nodes\/[^/]+\/evidence$/u.test(path)) {
      const lane: LaneKey = path.includes(ZERO_BUNDLE.run.runId) ? "zero" : "few";
      const bundle = lane === "zero" ? ZERO_BUNDLE : FEW_BUNDLE;
      const nodeId = decodeURIComponent(path.split("/").at(-2) ?? laneNodeId(lane, 0));
      counts[lane].evidence += 1; counts[lane].paths.push(`${method} ${path}`);
      return json(route, { ...bundle.evidence, node: { ...bundle.evidence.node, nodeId, label: `${lane} object ${Number(nodeId.slice(-3))}` } });
    }
    if (method === "POST" && path.endsWith("/adaptations/label-sets")) {
      const payload = request.postDataJSON() as { targetTaskRegistrationId: string; runId: string; resultHash: string };
      if (payload.targetTaskRegistrationId !== FEW_BUNDLE.registration.registrationId || payload.runId !== FEW_BUNDLE.run.runId || payload.resultHash !== FEW_BUNDLE.result.resultHash) return json(route, { detail: { code: "CROSS_LANE_LABEL_ALIAS" } }, 409);
      counts.few.labelSet += 1; counts.few.paths.push(`${method} ${path}`);
      return json(route, FEW_BUNDLE.registration.labels, 201);
    }
    if (method === "POST" && /\/adaptations\/label-sets\/[0-9a-f]{64}\/policies$/u.test(path)) {
      if (!path.includes(FEW_BUNDLE.policy.labelSetHash)) return json(route, { detail: { code: "CROSS_LANE_POLICY_ALIAS" } }, 409);
      const payload = request.postDataJSON() as { schemaVersion?: string; targetTaskRegistrationId?: string; runId?: string; resultHash?: string };
      if (payload.schemaVersion !== "socialgraph-fm.governance-target-review-policy-fit-request/1.0"
        || payload.targetTaskRegistrationId !== FEW_BUNDLE.registration.registrationId
        || payload.runId !== FEW_BUNDLE.run.runId
        || payload.resultHash !== FEW_BUNDLE.result.resultHash) return json(route, { detail: { code: "CROSS_LANE_POLICY_BINDING" } }, 409);
      counts.few.fit += 1; counts.few.paths.push(`${method} ${path}`);
      return json(route, FEW_BUNDLE.policy, 201);
    }
    if (method === "GET" && /\/adaptations\/policies\/[0-9a-f]{64}$/u.test(path)) {
      if (path.endsWith(ZERO_BUNDLE.policy.policyHash)) { counts.crossLaneAlias += 1; return json(route, { detail: { code: "ZERO_POLICY_NOT_AVAILABLE" } }, 409); }
      counts.few.policy += 1; counts.few.paths.push(`${method} ${path}`);
      return json(route, FEW_BUNDLE.policy);
    }
    if (method === "GET" && /\/adaptations\/runs\/[^/]+\/policies\/[0-9a-f]{64}\/comparison$/u.test(path)) {
      if (!path.includes(FEW_BUNDLE.run.runId) || !path.includes(FEW_BUNDLE.policy.policyHash)) return json(route, { detail: { code: "CROSS_LANE_COMPARISON_ALIAS" } }, 409);
      counts.few.comparison += 1; counts.few.paths.push(`${method} ${path}`);
      return json(route, FEW_BUNDLE.comparison);
    }
    if (method === "POST" && /\/adaptations\/policies\/[0-9a-f]{64}\/activate$/u.test(path)) {
      const payload = request.postDataJSON() as { targetTaskRegistrationId: string };
      if (payload.targetTaskRegistrationId !== FEW_BUNDLE.registration.registrationId || !path.includes(FEW_BUNDLE.policy.policyHash)) return json(route, { detail: { code: "CROSS_LANE_ACTIVATION_ALIAS" } }, 409);
      counts.few.activation += 1; counts.few.paths.push(`${method} ${path}`);
      return json(route, FEW_BUNDLE.activation, 201);
    }
    if (method === "POST" && path.endsWith("/adaptations/handoffs")) {
      const payload = request.postDataJSON() as { targetTaskRegistrationId: string; policyHash: string };
      if (payload.targetTaskRegistrationId !== FEW_BUNDLE.registration.registrationId || payload.policyHash !== FEW_BUNDLE.policy.policyHash) return json(route, { detail: { code: "CROSS_LANE_HANDOFF_ALIAS" } }, 409);
      counts.few.handoff += 1; counts.few.paths.push(`${method} ${path}`);
      return json(route, FEW_BUNDLE.handoff, 201);
    }
    return json(route, { detail: { code: "GFM_GOVERNANCE_ROUTE_NOT_MOCKED", path, method } }, 404);
  });
}

async function expectMinTarget(locator: Locator, label: string) {
  const box = await locator.boundingBox();
  expect(box, `${label} has a bounding box`).not.toBeNull();
  expect(box!.width, `${label} width`).toBeGreaterThanOrEqual(44);
  expect(box!.height, `${label} height`).toBeGreaterThanOrEqual(44);
}

async function expectNoHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(overflow.scroll).toBeLessThanOrEqual(overflow.client + 1);
}

async function expectRenderedLabelBudget(graph: Locator, maximum: number) {
  await expect.poll(async () => Number(await graph.getAttribute("data-rendered-label-count"))).toBeGreaterThan(0);
  expect(Number(await graph.getAttribute("data-rendered-label-count"))).toBeLessThanOrEqual(maximum);
}

async function canvasPaintSignal(graph: Locator) {
  return graph.locator(".graph-preview__canvas canvas").evaluateAll((canvases) => canvases.reduce((total, canvas) => {
    const element = canvas as HTMLCanvasElement;
    const context = element.getContext("2d", { willReadFrequently: true });
    if (!context || element.width === 0 || element.height === 0) return total;
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let painted = 0;
    for (let offset = 3; offset < pixels.length; offset += 64) {
      if (pixels[offset] > 8) painted += 1;
    }
    return total + painted;
  }, 0));
}

async function canvasSemanticColourCounts(graph: Locator) {
  return graph.locator(".graph-preview__canvas canvas").evaluateAll((canvases) => {
    const targets = {
      coral: [216, 92, 86],
      teal: [33, 139, 124],
      community: [120, 103, 217],
    } as const;
    const counts = { coral: 0, teal: 0, community: 0 };
    for (const canvas of canvases) {
      const element = canvas as HTMLCanvasElement;
      const context = element.getContext("2d", { willReadFrequently: true });
      if (!context || element.width === 0 || element.height === 0) continue;
      const pixels = context.getImageData(0, 0, element.width, element.height).data;
      for (let offset = 0; offset < pixels.length; offset += 4) {
        if (pixels[offset + 3] < 180) continue;
        for (const [name, target] of Object.entries(targets) as Array<[keyof typeof counts, readonly [number, number, number]]>) {
          if (Math.abs(pixels[offset] - target[0]) <= 12
            && Math.abs(pixels[offset + 1] - target[1]) <= 12
            && Math.abs(pixels[offset + 2] - target[2]) <= 12) counts[name] += 1;
        }
      }
    }
    return counts;
  });
}

async function canvasDiagnostics(graph: Locator) {
  return graph.locator(".graph-preview__canvas canvas").evaluateAll((canvases) => canvases.map((canvas) => {
    const element = canvas as HTMLCanvasElement;
    const context = element.getContext("2d", { willReadFrequently: true });
    const style = getComputedStyle(element);
    const bounds = element.getBoundingClientRect();
    if (!context || element.width === 0 || element.height === 0) {
      return { width: element.width, height: element.height, display: style.display, visibility: style.visibility, opacity: style.opacity, background: style.backgroundColor, zIndex: style.zIndex, bounds, painted: 0, contrast: 0 };
    }
    const pixels = context.getImageData(0, 0, element.width, element.height).data;
    let painted = 0;
    let contrast = 0;
    for (let offset = 0; offset < pixels.length; offset += 64) {
      const alpha = pixels[offset + 3] / 255;
      if (alpha <= 0.03) continue;
      painted += 1;
      const red = pixels[offset] * alpha + 8 * (1 - alpha);
      const green = pixels[offset + 1] * alpha + 19 * (1 - alpha);
      const blue = pixels[offset + 2] * alpha + 39 * (1 - alpha);
      if (Math.abs(red - 8) + Math.abs(green - 19) + Math.abs(blue - 39) > 40) contrast += 1;
    }
    return { width: element.width, height: element.height, display: style.display, visibility: style.visibility, opacity: style.opacity, background: style.backgroundColor, zIndex: style.zIndex, bounds, painted, contrast };
  }));
}

async function uploadLane(lane: Locator, label: string, name: string) {
  await lane.getByLabel(label).setInputFiles({
    name,
    mimeType: "application/zip",
    buffer: Buffer.from(`PK ${name}`),
  });
  await expect(lane).toContainText("108 个对象 · 220 条关系");
}

async function runLane(lane: Locator) {
  await lane.getByRole("button", { name: "开始分析" }).click();
  await lane.getByRole("button", { name: "确认分析" }).click();
  await expect(lane.getByText(/协同组群已就绪/u)).toBeVisible();
}

async function laneIdentitySnapshot(lane: Locator) {
  return lane.evaluate((element) => ({
    phase: element.getAttribute("data-phase"),
    registrationId: element.getAttribute("data-registration-id"),
    taskId: element.getAttribute("data-task-id"),
    artifactId: element.getAttribute("data-artifact-id"),
    inferenceHash: element.getAttribute("data-inference-hash"),
    graphVersionHash: element.getAttribute("data-graph-version-hash"),
    targetReceiptHash: element.getAttribute("data-target-receipt-hash"),
    registrationHash: element.getAttribute("data-registration-hash"),
    outerBundleHash: element.getAttribute("data-outer-bundle-hash"),
    nodeSetHash: element.getAttribute("data-node-set-hash"),
    runId: element.getAttribute("data-run-id"),
    resultHash: element.getAttribute("data-result-hash"),
    selectedNodeId: element.getAttribute("data-selected-node-id"),
    overlay: element.getAttribute("data-overlay"),
    artifactEpoch: element.getAttribute("data-artifact-epoch"),
    runEpoch: element.getAttribute("data-run-epoch"),
    policyEpoch: element.getAttribute("data-policy-epoch"),
    graphEpoch: element.getAttribute("data-graph-epoch"),
    cameraEpoch: element.getAttribute("data-camera-epoch"),
    focusEpoch: element.getAttribute("data-focus-epoch"),
    abortEpoch: element.getAttribute("data-abort-epoch"),
  }));
}

async function graphCameraSnapshot(graph: Locator) {
  return graph.evaluate((element) => ({
    x: Number(element.getAttribute("data-camera-x")),
    y: Number(element.getAttribute("data-camera-y")),
    zoom: Number(element.getAttribute("data-camera-zoom")),
  }));
}

async function stableGraphCameraSnapshot(graph: Locator) {
  let previous: Awaited<ReturnType<typeof graphCameraSnapshot>> | null = null;
  let stableSamples = 0;
  await expect.poll(async () => {
    const current = await graphCameraSnapshot(graph);
    if (previous
      && Math.abs(current.x - previous.x) < 0.01
      && Math.abs(current.y - previous.y) < 0.01
      && Math.abs(current.zoom - previous.zoom) < 0.0001) stableSamples += 1;
    else stableSamples = 0;
    previous = current;
    return stableSamples;
  }, { intervals: [100, 150, 200, 250] }).toBeGreaterThanOrEqual(2);
  return graphCameraSnapshot(graph);
}

type LaneIdentitySnapshot = Awaited<ReturnType<typeof laneIdentitySnapshot>>;

function splitLaneIdentitySnapshot(snapshot: LaneIdentitySnapshot) {
  const {
    artifactEpoch,
    runEpoch,
    policyEpoch,
    graphEpoch,
    cameraEpoch,
    focusEpoch,
    abortEpoch,
    ...nonEpoch
  } = snapshot;
  return {
    nonEpoch,
    stableEpochs: { artifactEpoch, runEpoch, policyEpoch, graphEpoch },
    reactivationEpochs: { cameraEpoch, focusEpoch, abortEpoch },
  };
}

function epochNumber(value: string | null, label: string) {
  expect(value, `${label} epoch is present`).not.toBeNull();
  expect(value, `${label} epoch is a non-negative integer`).toMatch(/^(?:0|[1-9]\d*)$/u);
  return Number(value);
}

function expectLaneReactivation(before: LaneIdentitySnapshot, after: LaneIdentitySnapshot) {
  const beforeParts = splitLaneIdentitySnapshot(before);
  const afterParts = splitLaneIdentitySnapshot(after);

  expect(afterParts.nonEpoch, "reactivation preserves every non-epoch lane field").toEqual(beforeParts.nonEpoch);
  expect(afterParts.stableEpochs, "reactivation preserves artifact/run/policy/graph epochs").toEqual(beforeParts.stableEpochs);
  for (const epochField of ["artifactEpoch", "runEpoch", "policyEpoch", "graphEpoch"] as const) {
    epochNumber(beforeParts.stableEpochs[epochField], `before ${epochField}`);
    epochNumber(afterParts.stableEpochs[epochField], `after ${epochField}`);
  }
  expect(epochNumber(afterParts.reactivationEpochs.cameraEpoch, "after camera")).toBe(epochNumber(beforeParts.reactivationEpochs.cameraEpoch, "before camera") + 1);
  expect(epochNumber(afterParts.reactivationEpochs.focusEpoch, "after focus")).toBe(epochNumber(beforeParts.reactivationEpochs.focusEpoch, "before focus") + 1);
  expect(epochNumber(afterParts.reactivationEpochs.abortEpoch, "after abort")).toBe(epochNumber(beforeParts.reactivationEpochs.abortEpoch, "before abort") + 1);
}

test("reactivation snapshot guard rejects a cross-lane non-epoch alias", () => {
  const before: LaneIdentitySnapshot = {
    phase: "ready",
    registrationId: "zero-registration",
    taskId: "zero-task",
    artifactId: "zero-artifact",
    inferenceHash: "zero-inference",
    graphVersionHash: "zero-graph-version",
    targetReceiptHash: "zero-target-receipt",
    registrationHash: "zero-registration-hash",
    outerBundleHash: "zero-outer-bundle",
    nodeSetHash: "zero-node-set",
    runId: "zero-run",
    resultHash: "zero-result",
    selectedNodeId: "zero-node-001",
    overlay: "community",
    artifactEpoch: "1",
    runEpoch: "2",
    policyEpoch: "0",
    graphEpoch: "3",
    cameraEpoch: "4",
    focusEpoch: "5",
    abortEpoch: "6",
  };
  const crossLaneAlias: LaneIdentitySnapshot = {
    ...before,
    taskId: "few-task",
    cameraEpoch: "5",
    focusEpoch: "6",
    abortEpoch: "7",
  };

  expect(() => expectLaneReactivation(before, crossLaneAlias)).toThrow(/few-task/u);
});

test("immutable two-lane journey preserves overlays, renderer, theme isolation, focus, and recovery", async ({ page }) => {
  test.setTimeout(60_000);
  const pageErrors: string[] = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  const counts: RouteCounts = { zero: emptyLaneCounts(), few: emptyLaneCounts(), crossLaneAlias: 0 };
  await mockTargetAdaptation(page, counts);
  await page.addInitScript(() => localStorage.setItem("socialgraph-fm.governance.theme.v1", "brand-light"));
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("/#/adaptation");

  const workspace = page.getByRole("region", { name: "适配能力工作台" });
  const guide = workspace.getByRole("navigation", { name: "选择适配路径" });
  await expect(workspace.getByRole("heading", { name: "面向新网络的风险迁移" })).toBeVisible();
  await expect(guide.getByRole("button")).toHaveCount(2);
  await expect(workspace.getByText(/最终结论由人工确认/u)).toBeVisible();

  const zero = workspace.getByRole("region", { name: "零样本路径" });
  const few = workspace.getByRole("region", { name: "少样本路径" });
  const graph = page.locator('.graph-preview[aria-label="适配任务关系图"]');
  await expectMinTarget(zero.locator(".adaptation-file-action"), "zero-shot upload");
  await expectMinTarget(few.locator(".adaptation-file-action"), "few-shot upload");
  await uploadLane(zero, "零样本目标任务包", "regional-zero.sgtask.zip");
  await runLane(zero);
  await expect(graph.getByRole("group", { name: "切换适配网络" })).toHaveCount(0);
  await expect(zero.getByRole("button", { name: "进入治理应用" })).toBeEnabled();
  await zero.getByRole("button", { name: "zero object 1，待治理核验", exact: true }).click();
  await expect(zero.getByText("直接关系证据")).toBeVisible();
  await expect(graph.getByLabel("已选节点详情")).toContainText("zero object 1");
  const zeroBeforeFew = await laneIdentitySnapshot(zero);
  expect(zeroBeforeFew).toMatchObject({
    phase: "ready",
    registrationId: ZERO_BUNDLE.registration.registrationId,
    taskId: ZERO_BUNDLE.registration.task.taskId,
    artifactId: ZERO_BUNDLE.registration.artifact.artifactId,
    inferenceHash: ZERO_BUNDLE.registration.artifact.bundleSha256,
    graphVersionHash: ZERO_BUNDLE.registration.artifact.graphVersionHash,
    targetReceiptHash: ZERO_BUNDLE.registration.targetReceipt.receiptHash,
    registrationHash: ZERO_BUNDLE.registration.registrationHash,
    outerBundleHash: ZERO_BUNDLE.registration.outerBundleSha256,
    nodeSetHash: ZERO_BUNDLE.registration.targetReceipt.nodeSetSha256,
    runId: ZERO_BUNDLE.run.runId,
    resultHash: ZERO_BUNDLE.result.resultHash,
    selectedNodeId: laneNodeId("zero", 0),
    overlay: "community",
  });

  await uploadLane(few, "少样本目标任务包", "regional-few.sgtask.zip");
  await expect(few).toHaveAttribute("data-phase", "raw");
  await expect(graph).toHaveAttribute("data-reference-label-count", "16");
  await expect.poll(async () => (await canvasSemanticColourCounts(graph)).coral).toBeGreaterThan(0);
  await expect.poll(async () => (await canvasSemanticColourCounts(graph)).teal).toBeGreaterThan(0);
  await runLane(few);
  await expect(few).toHaveAttribute("data-phase", "compared");
  await expect(few.getByRole("button", { name: "进入治理应用" })).toBeEnabled();
  await expect(few.getByRole("button", { name: "拟合冻结复核策略" })).toHaveCount(0);
  await expect(graph).toHaveAttribute("data-reference-label-count", "16");
  await expect.poll(async () => (await canvasSemanticColourCounts(graph)).community).toBeGreaterThan(0);
  const fewAfterComparison = await laneIdentitySnapshot(few);
  expect(fewAfterComparison).toMatchObject({
    phase: "compared",
    registrationId: FEW_BUNDLE.registration.registrationId,
    taskId: FEW_BUNDLE.registration.task.taskId,
    artifactId: FEW_BUNDLE.registration.artifact.artifactId,
    inferenceHash: FEW_BUNDLE.registration.artifact.bundleSha256,
    graphVersionHash: FEW_BUNDLE.registration.artifact.graphVersionHash,
    targetReceiptHash: FEW_BUNDLE.registration.targetReceipt.receiptHash,
    registrationHash: FEW_BUNDLE.registration.registrationHash,
    outerBundleHash: FEW_BUNDLE.registration.outerBundleSha256,
    nodeSetHash: FEW_BUNDLE.registration.targetReceipt.nodeSetSha256,
    runId: FEW_BUNDLE.run.runId,
    resultHash: FEW_BUNDLE.result.resultHash,
    selectedNodeId: null,
    overlay: "community",
  });
  for (const identityField of ["registrationId", "taskId", "artifactId", "inferenceHash", "graphVersionHash", "targetReceiptHash", "registrationHash", "outerBundleHash", "nodeSetHash", "runId", "resultHash"] as const) {
    expect(fewAfterComparison[identityField]).not.toBe(zeroBeforeFew[identityField]);
  }
  expect(counts.zero).toMatchObject({
    targetTasks: 1,
    createRun: 1,
    pollRun: 1,
    result: 1,
    runPreview: 1,
    evidence: 1,
    labelSet: 0,
    fit: 0,
    policy: 0,
    comparison: 0,
    activation: 0,
    handoff: 0,
  });
  expect(counts.few).toMatchObject({
    targetTasks: 1,
    createRun: 1,
    pollRun: 1,
    result: 1,
    runPreview: 1,
    evidence: 0,
    labelSet: 1,
    fit: 1,
    policy: 1,
    comparison: 1,
    activation: 0,
    handoff: 0,
  });
  expect(counts.few.paths).toContain(`GET /api/v2/gfm/governance/runs/${FEW_BUNDLE.run.runId}/result`);
  expect(counts.zero.paths).toContain(`GET /api/v2/gfm/governance/runs/${ZERO_BUNDLE.run.runId}/result`);
  expect(counts.few.paths).toContain(`GET /api/v2/gfm/governance/artifacts/${FEW_BUNDLE.registration.artifact.artifactId}/preview`);
  expect(counts.zero.paths).toContain(`GET /api/v2/gfm/governance/artifacts/${ZERO_BUNDLE.registration.artifact.artifactId}/preview`);
  expect(counts.few.paths).toContain(`GET /api/v2/gfm/governance/runs/${FEW_BUNDLE.run.runId}/graph-preview`);
  expect(counts.zero.paths).toContain(`GET /api/v2/gfm/governance/runs/${ZERO_BUNDLE.run.runId}/graph-preview`);
  expect(counts.zero.paths.join("\n")).not.toContain(FEW_BUNDLE.run.runId);
  expect(counts.few.paths.join("\n")).not.toContain(ZERO_BUNDLE.run.runId);

  const graphSwitcher = graph.getByRole("group", { name: "切换适配网络" });
  const zeroGraphButton = graphSwitcher.getByRole("button", { name: "零样本网络" });
  const fewGraphButton = graphSwitcher.getByRole("button", { name: "少样本网络" });
  await expect(zeroGraphButton).toHaveAttribute("aria-pressed", "false");
  await expect(fewGraphButton).toHaveAttribute("aria-pressed", "true");
  const routeCountsBeforeSwitch = JSON.stringify(counts);
  const layoutCountBeforeSwitch = Number(await graph.getAttribute("data-layout-count"));
  const laneZoomButton = graph.getByRole("button", { name: "放大图谱" });
  const fewZoomBefore = (await stableGraphCameraSnapshot(graph)).zoom;
  await laneZoomButton.click();
  await expect.poll(async () => (await graphCameraSnapshot(graph)).zoom).toBeGreaterThan(fewZoomBefore);
  const fewCamera = await stableGraphCameraSnapshot(graph);

  await zeroGraphButton.click();
  await expect(zeroGraphButton).toHaveAttribute("aria-pressed", "true");
  await expect(graph).toHaveAttribute("data-reference-label-count", "0");
  const zeroZoomBefore = (await stableGraphCameraSnapshot(graph)).zoom;
  await laneZoomButton.click();
  await expect.poll(async () => (await graphCameraSnapshot(graph)).zoom).toBeGreaterThan(zeroZoomBefore);
  const zeroCamera = await stableGraphCameraSnapshot(graph);
  expect(zeroCamera.zoom).not.toBeCloseTo(fewCamera.zoom, 4);

  await fewGraphButton.click();
  await expect(fewGraphButton).toHaveAttribute("aria-pressed", "true");
  await expect(graph).toHaveAttribute("data-reference-label-count", "16");
  await expect.poll(async () => (await graphCameraSnapshot(graph)).zoom).toBeCloseTo(fewCamera.zoom, 4);

  await zeroGraphButton.click();
  await expect(zeroGraphButton).toHaveAttribute("aria-pressed", "true");
  await expect(graph).toHaveAttribute("data-reference-label-count", "0");
  await fewGraphButton.click();
  await expect(fewGraphButton).toHaveAttribute("aria-pressed", "true");
  expect(JSON.stringify(counts)).toBe(routeCountsBeforeSwitch);
  expect(Number(await graph.getAttribute("data-layout-count"))).toBe(layoutCountBeforeSwitch);

  const negativeAlias = await page.evaluate(async (policyHash) => {
    const response = await fetch(`/api/v2/gfm/governance/adaptations/policies/${policyHash}`);
    const payload = await response.json() as { detail?: { code?: string } };
    return { status: response.status, code: payload.detail?.code };
  }, ZERO_BUNDLE.policy.policyHash);
  expect(negativeAlias).toEqual({ status: 409, code: "ZERO_POLICY_NOT_AVAILABLE" });
  expect(counts.crossLaneAlias).toBe(1);

  await expect(graph).toHaveClass(/graph-preview--focus-dark/u);
  await expect(few.getByRole("group", { name: "排序图层" })).toHaveCount(0);
  await expect(graph.getByTitle(/协同组群/u)).toBeVisible();
  expect(counts.few.activation).toBe(0);

  await few.getByRole("button", { name: "few object 1，待治理核验", exact: true }).click();
  await expect(few.getByText("直接关系证据")).toBeVisible();
  await expect(graph.getByLabel("已选节点详情")).toContainText("few object 1");
  await expect(graph.getByRole("button", { name: "返回适配全图" })).toBeVisible();
  expect(counts.few.evidence).toBe(1);

  await few.getByRole("button", { name: "查看迁移依据" }).click();
  const transferDialog = page.getByRole("dialog", { name: "少样本源域路由与校正" });
  await expect(transferDialog).toBeVisible();
  await expect(transferDialog.getByRole("table", { name: "匿名专家路由明细" })).toBeVisible();
  await expect(transferDialog.getByRole("region", { name: "当前选中账号的迁移依据" })).toContainText("few object 1");
  await expect(transferDialog).toContainText("λ 0.50");
  await expect(transferDialog).not.toContainText(/china|cuba|iran|russia|UAE|venezuela/iu);
  await expectMinTarget(transferDialog.getByRole("button", { name: "关闭迁移依据" }), "transfer evidence close");
  await transferDialog.getByRole("button", { name: "关闭迁移依据" }).click();
  await expect(transferDialog).toHaveCount(0);

  const themeButton = graph.getByRole("button", { name: "切换到品牌浅色主题" });
  const zoomButton = graph.getByRole("button", { name: "放大图谱" });
  const filterButton = graph.getByRole("button", { name: "筛选节点类型与关系，当前显示全部" });
  const searchInput = graph.getByRole("textbox", { name: "按名称或 ID 搜索节点" });
  await expectMinTarget(themeButton, "graph theme control");
  await expectMinTarget(zoomButton, "graph zoom control");
  await expectMinTarget(filterButton, "graph filter control");
  await expectMinTarget(searchInput, "graph search control");

  await graph.locator("canvas").first().evaluate((canvas) => {
    (window as Window & { __task5Canvas?: HTMLCanvasElement }).__task5Canvas = canvas;
  });
  await themeButton.click();
  await expect(graph).not.toHaveClass(/graph-preview--focus-dark/u);
  expect(await graph.locator("canvas").first().evaluate((canvas) => (
    (window as Window & { __task5Canvas?: HTMLCanvasElement }).__task5Canvas === canvas
  ))).toBe(true);
  expect(await page.evaluate(() => localStorage.getItem("socialgraph-fm.governance.theme.v1"))).toBe("brand-light");
  await expect(zero.getByText(/协同组群已就绪/u)).toBeVisible();
  await expect(few.getByText(/协同组群已就绪/u)).toBeVisible();
  await expect(graph).toHaveAttribute("data-visible-nodes", "108");
  await expectRenderedLabelBudget(graph, 13);
  const lightPaint = await canvasPaintSignal(graph);
  expect(lightPaint).toBeGreaterThan(100);
  await page.screenshot({ path: resolve(evidenceRoot, "12-adaptation-graph-light-runtime-1280x720.png") });
  await graph.getByRole("button", { name: "切换到专注深色主题" }).click();
  await expect(graph).toHaveClass(/graph-preview--focus-dark/u);
  expect(await graph.locator("canvas").first().evaluate((canvas) => (
    (window as Window & { __task5Canvas?: HTMLCanvasElement }).__task5Canvas === canvas
  ))).toBe(true);
  await expect(graph).toHaveAttribute("data-visible-nodes", "108");
  await expectRenderedLabelBudget(graph, 13);
  await expect.poll(() => canvasPaintSignal(graph)).toBeGreaterThan(100);
  const darkCanvasDiagnostics = await canvasDiagnostics(graph);
  expect(darkCanvasDiagnostics.some((canvas) => canvas.contrast > 100)).toBe(true);
  expect(darkCanvasDiagnostics.every((canvas) => canvas.background === "rgba(0, 0, 0, 0)")).toBe(true);
  await page.screenshot({ path: resolve(evidenceRoot, "13-adaptation-graph-dark-runtime-1280x720.png") });

  await graph.getByRole("button", { name: "返回适配全图" }).click();
  await expect(graph.getByLabel("已选节点详情")).toHaveCount(0);
  await expect(graph.getByRole("button", { name: "返回适配全图" })).toHaveCount(0);
  await expect(graph.getByRole("button", { name: "找回并适应图谱视图" })).toBeVisible();

  for (const viewport of [{ width: 1920, height: 1080 }, { width: 1440, height: 900 }, { width: 1280, height: 720 }]) {
    await page.setViewportSize(viewport);
    await expectMinTarget(zero.locator(".adaptation-file-action"), `zero upload at ${viewport.width}`);
    await expectMinTarget(few.locator(".adaptation-file-action"), `few upload at ${viewport.width}`);
    await expectMinTarget(few.getByRole("button", { name: "进入治理应用" }), `handoff at ${viewport.width}`);
    await expectMinTarget(zeroGraphButton, `zero graph switch at ${viewport.width}`);
    await expectMinTarget(fewGraphButton, `few graph switch at ${viewport.width}`);
    const [zeroBox, fewBox] = await Promise.all([zeroGraphButton.boundingBox(), fewGraphButton.boundingBox()]);
    expect(Math.abs(zeroBox!.width - fewBox!.width), `switch widths at ${viewport.width}`).toBeLessThanOrEqual(1);
    await expectNoHorizontalOverflow(page);
  }

  await page.setViewportSize({ width: 390, height: 844 });
  await expect(page.getByRole("tab", { name: "任务" })).toBeVisible();
  await expectMinTarget(zero.locator(".adaptation-file-action"), "mobile zero upload");
  await expectMinTarget(few.locator(".adaptation-file-action"), "mobile few upload");
  await expectMinTarget(few.getByRole("button", { name: "进入治理应用" }), "mobile handoff");
  await page.getByRole("tab", { name: "图谱" }).click();
  await expectMinTarget(zeroGraphButton, "mobile zero graph switch");
  await expectMinTarget(fewGraphButton, "mobile few graph switch");
  const [mobileZeroBox, mobileFewBox] = await Promise.all([zeroGraphButton.boundingBox(), fewGraphButton.boundingBox()]);
  expect(Math.abs(mobileZeroBox!.width - mobileFewBox!.width)).toBeLessThanOrEqual(1);
  await expectMinTarget(graph.getByRole("button", { name: "切换到品牌浅色主题" }), "mobile graph theme");
  await expectMinTarget(graph.getByRole("button", { name: "放大图谱" }), "mobile graph zoom");
  await expectMinTarget(graph.getByRole("button", { name: "筛选节点类型与关系，当前显示全部" }), "mobile graph filter");
  await expectMinTarget(graph.getByRole("textbox", { name: "按名称或 ID 搜索节点" }), "mobile graph search");
  await expectNoHorizontalOverflow(page);

  await page.setViewportSize({ width: 1920, height: 1080 });
  await few.getByRole("button", { name: "few object 2，待治理核验", exact: true }).click();
  await expect(graph.getByRole("button", { name: "返回适配全图" })).toBeVisible();
  await page.setViewportSize({ width: 960, height: 540 });
  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 960,
    height: 540,
    deviceScaleFactor: 2,
    mobile: false,
    screenWidth: 1920,
    screenHeight: 1080,
  });
  expect(await page.evaluate(() => ({ width: innerWidth, dpr: devicePixelRatio }))).toEqual({ width: 960, dpr: 2 });
  await expectNoHorizontalOverflow(page);
  await expect(graph.getByRole("button", { name: "返回适配全图" })).toBeVisible();
  const graphZoomCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  writeFileSync(resolve(evidenceRoot, "10-adaptation-focus-actual-browser-zoom-200-percent-1920x1080.png"), Buffer.from(graphZoomCapture.data, "base64"));
  await page.getByRole("tab", { name: "任务" }).click();
  await expect(few.getByRole("button", { name: "进入治理应用" })).toBeVisible();
  await expectNoHorizontalOverflow(page);
  const taskZoomCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  writeFileSync(resolve(evidenceRoot, "11-adaptation-actions-actual-browser-zoom-200-percent-1920x1080.png"), Buffer.from(taskZoomCapture.data, "base64"));
  await cdp.send("Emulation.clearDeviceMetricsOverride");
  await page.setViewportSize({ width: 1280, height: 720 });

  await graph.getByRole("button", { name: "返回适配全图" }).click();
  await few.getByRole("button", { name: "进入治理应用" }).click();
  await expect(page).toHaveURL(/#\/governance$/u);
  expect(counts.few.handoff).toBe(1);

  await page.getByRole("button", { name: "适配能力", exact: true }).click();
  await expect(page).toHaveURL(/#\/adaptation$/u);
  expect(await laneIdentitySnapshot(zero)).toEqual(zeroBeforeFew);
  await zero.getByRole("button", { name: "zero object 1，待治理核验", exact: true }).click();
  await expect(graph.getByLabel("已选节点详情")).toContainText("zero object 1");
  await expect(graph.getByTitle(/协同组群/u)).toBeVisible();
  const zeroAfterReactivation = await laneIdentitySnapshot(zero);
  expectLaneReactivation(zeroBeforeFew, zeroAfterReactivation);
  expect(counts.zero.evidence).toBe(2);
  expect(counts.few.evidence).toBe(2);
  expect(pageErrors).toEqual([]);
});
