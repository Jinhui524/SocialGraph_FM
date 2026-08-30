import { expect, test, type Page, type Route } from "@playwright/test";
import { Buffer } from "node:buffer";

import { onlineCapabilities, onlineHealth } from "../src/test/fixtures/governanceOnline";
import { globalModelModelCard } from "../src/test/fixtures/globalModel";
import {
  adaptationHandoff,
  targetActivation,
  targetComparison,
  targetDerivationPage,
  targetEvidence,
  targetFindingPage,
  targetPolicy,
  targetPreview,
  targetResult,
  targetReviewCollection,
  targetRun,
  targetTaskRegistration,
  type TargetFixtureMode,
} from "../src/test/fixtures/governanceTargetTask";
import {
  GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
  GOVERNANCE_ASSISTANT_SCHEMA,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
} from "../src/types/governanceSkills";
import { sha256Canonical } from "../src/services/graphIdentity";

const CASE_ID = `case-${"4".repeat(32)}`;
const FEW_CASE_ID = `case-${"3".repeat(32)}`;
const FOREIGN_CASE_ID = `case-${"2".repeat(32)}`;
const FOREIGN_RUN_ID = `governance-${"e".repeat(32)}`;

type CaseFixtureLane = TargetFixtureMode | "foreign";

function caseIdFor(lane: CaseFixtureLane): string {
  return lane === "zero_shot" ? CASE_ID : lane === "few_shot" ? FEW_CASE_ID : FOREIGN_CASE_ID;
}

function caseRunIdFor(lane: CaseFixtureLane): string {
  return lane === "foreign" ? FOREIGN_RUN_ID : targetRun(lane).runId;
}

function targetCase(lane: CaseFixtureLane, state: "draft" | "active" | "concluded", hasItem: boolean, reviewed: boolean) {
  const targetId = "target-node-001";
  return {
    schemaVersion: "socialgraph-fm.gfm-governance/2.0", caseId: caseIdFor(lane), runId: caseRunIdFor(lane),
    title: lane === "zero_shot" ? "A 风险候选复核" : lane === "few_shot" ? "B 独立研判单" : "Foreign run case",
    description: "独立目标任务的人工复核记录。", state,
    createdAt: "2026-08-21T00:00:00Z", updatedAt: "2026-08-21T00:05:00Z",
    items: hasItem ? [{ itemId: `item-${"5".repeat(32)}`, targetType: "node", targetId, note: "由治理工作台加入研判范围。", createdAt: "2026-08-21T00:01:00Z", itemHash: "5".repeat(64) }] : [],
    reviewEvents: reviewed ? [{ eventId: `event-${"6".repeat(32)}`, targetType: "node", targetId, decision: "pending", reason: "补充直接关系证据。", actor: "local-analyst", sequence: 1, createdAt: "2026-08-21T00:04:00Z", previousEventHash: null, eventHash: "6".repeat(64) }] : [],
    currentDecisions: reviewed ? { [`node:${targetId}`]: "pending" } : {}, caseHash: reviewed ? "7".repeat(64) : hasItem ? "6".repeat(64) : "5".repeat(64),
  };
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

async function mockIndependentTargets(page: Page) {
  let registrationCount = 0;
  const zeroRegistration = targetTaskRegistration("zero_shot");
  const fewRegistration = targetTaskRegistration("few_shot");
  const zeroRun = targetRun("zero_shot"); const fewRun = targetRun("few_shot");
  const fewPolicy = targetPolicy("few_shot"); const fewHandoff = adaptationHandoff("few_shot");
  const modes: readonly TargetFixtureMode[] = ["zero_shot", "few_shot"];
  const calls = {
    targetTaskReads: { [zeroRegistration.registrationId]: 0, [fewRegistration.registrationId]: 0 },
    artifactReads: { [zeroRegistration.artifact.artifactId]: 0, [fewRegistration.artifact.artifactId]: 0 },
    rawPreviewReads: { [zeroRegistration.artifact.artifactId]: 0, [fewRegistration.artifact.artifactId]: 0 },
    runReads: { [zeroRun.runId]: 0, [fewRun.runId]: 0 },
    resultReads: { [zeroRun.runId]: 0, [fewRun.runId]: 0 },
    scoredPreviewReads: { [zeroRun.runId]: 0, [fewRun.runId]: 0 },
    caseListReads: 0,
    caseListServedRunIds: [] as string[],
    evidenceReads: {} as Record<string, number>,
    assistantReads: { [zeroRun.runId]: 0, [fewRun.runId]: 0 },
    knowledgeReads: { [zeroRegistration.artifact.artifactId]: 0, [fewRegistration.artifact.artifactId]: 0 },
    similarReads: { [zeroRun.runId]: 0, [fewRun.runId]: 0 },
    handoffReads: { [fewHandoff.handoffHash]: 0 },
    policyReads: { [fewPolicy.policyHash]: 0 },
    comparisonReads: { [`${fewRun.runId}:${fewPolicy.policyHash}`]: 0 },
    unmatched: [] as string[],
  };
  const registrationFor = (mode: TargetFixtureMode) => mode === "zero_shot" ? zeroRegistration : fewRegistration;
  const runFor = (mode: TargetFixtureMode) => mode === "zero_shot" ? zeroRun : fewRun;
  const increment = (ledger: Record<string, number>, key: string) => { ledger[key] = (ledger[key] ?? 0) + 1; };
  const rejectIdentity = (route: Route, method: string, path: string) => {
    calls.unmatched.push(`${method} ${path}`);
    return json(route, { detail: { code: "GFM_GOVERNANCE_FIXTURE_IDENTITY_MISMATCH", path } }, 409);
  };
  const caseStates: Record<TargetFixtureMode, { exists: boolean; state: "draft" | "active" | "concluded"; hasItem: boolean; reviewed: boolean }> = {
    zero_shot: { exists: false, state: "draft", hasItem: false, reviewed: false },
    few_shot: { exists: true, state: "draft", hasItem: false, reviewed: false },
  };
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "GET" && path === "/api/v1/gfm/global-model/model-card") {
      const base = structuredClone(globalModelModelCard());
      const result = targetResult("zero_shot");
      const logical = {
        ...base,
        modelVersionId: result.modelVersionId,
        modelVersionHash: result.modelVersionHash,
        protocols: {
          ...base.protocols,
          global: {
            ...base.protocols.global,
            modelVersionId: result.modelVersionId,
            modelVersionHash: result.modelVersionHash,
            modelStateHash: result.modelStateHash,
          },
        },
      };
      const { modelCardHash: _ignored, ...withoutHash } = logical;
      return json(route, { ...withoutHash, modelCardHash: sha256Canonical(withoutHash) });
    }
    if (request.method() === "GET" && [
      "/api/v1/health",
      "/api/v1/gfm/capabilities",
      "/api/v1/gfm/research/capabilities",
    ].includes(path)) return json(route, {});
    calls.unmatched.push(`${request.method()} ${path}`);
    return json(route, { detail: { code: "OFFLINE_BOOTSTRAP_ROUTE_NOT_MOCKED", path } }, 404);
  });
  await page.route("**/api/v2/gfm/governance/**", async (route) => {
    const request = route.request(); const url = new URL(request.url()); const path = url.pathname; const method = request.method();
    const base = "/api/v2/gfm/governance";
    if (method === "GET" && path.endsWith("/health")) return json(route, onlineHealth());
    if (method === "GET" && path.endsWith("/capabilities")) return json(route, onlineCapabilities());
    if (method === "GET" && path.endsWith("/runs")) return json(route, { schemaVersion: "socialgraph-fm.gfm-governance/2.0", items: [], total: 0, offset: 0, limit: 50 });
    if (method === "POST" && path.endsWith("/target-tasks")) {
      const mode = registrationCount++ === 0 ? "zero_shot" : "few_shot";
      return json(route, targetTaskRegistration(mode), 201);
    }
    for (const mode of modes) {
      const registration = registrationFor(mode); const run = runFor(mode);
      if (method === "GET" && path === `${base}/target-tasks/${registration.registrationId}`) {
        increment(calls.targetTaskReads, registration.registrationId); return json(route, registration);
      }
      if (method === "GET" && path === `${base}/artifacts/${registration.artifact.artifactId}`) {
        increment(calls.artifactReads, registration.artifact.artifactId); return json(route, registration.artifact);
      }
      if (method === "GET" && path === `${base}/artifacts/${registration.artifact.artifactId}/preview`) {
        increment(calls.rawPreviewReads, registration.artifact.artifactId); return json(route, targetPreview(false, mode));
      }
      if (method === "GET" && path === `${base}/runs/${run.runId}/result`) {
        increment(calls.resultReads, run.runId); return json(route, targetResult(mode));
      }
      if (method === "GET" && path === `${base}/runs/${run.runId}/graph-preview`) {
        increment(calls.scoredPreviewReads, run.runId); return json(route, targetPreview(true, mode));
      }
      if (method === "GET" && path === `${base}/runs/${run.runId}/nodes`) return json(route, targetFindingPage(mode));
      if (method === "GET" && ["groups", "relations", "potential-links"].some((kind) => path === `${base}/runs/${run.runId}/${kind}`)) return json(route, targetDerivationPage(mode));
      const evidencePrefix = `${base}/runs/${run.runId}/nodes/`;
      if (method === "GET" && path.startsWith(evidencePrefix) && path.endsWith("/evidence")) {
        const nodeId = decodeURIComponent(path.slice(evidencePrefix.length, -"/evidence".length));
        if (!targetResult(mode).findings.some((finding) => finding.nodeId === nodeId)) return rejectIdentity(route, method, path);
        increment(calls.evidenceReads, `${run.runId}:${nodeId}`);
        return json(route, targetEvidence(nodeId, mode));
      }
      if (method === "GET" && path === `${base}/runs/${run.runId}`) {
        increment(calls.runReads, run.runId); return json(route, run);
      }
    }
    if (method === "POST" && path === `${base}/runs`) {
      const body = request.postDataJSON() as { artifactId?: string; datasetContentHash?: string; graphVersionHash?: string };
      const mode = modes.find((candidate) => {
        const registration = registrationFor(candidate);
        return body.artifactId === registration.artifact.artifactId
          && body.datasetContentHash === registration.artifact.datasetContentHash
          && body.graphVersionHash === registration.artifact.graphVersionHash;
      });
      return mode ? json(route, runFor(mode), 202) : rejectIdentity(route, method, path);
    }
    if (method === "GET" && path === `${base}/cases`
      && url.searchParams.get("offset") === "0" && url.searchParams.get("limit") === "100"
      && [...url.searchParams.keys()].length === 2) {
      const items = [
        ...(caseStates.zero_shot.exists ? [targetCase("zero_shot", caseStates.zero_shot.state, caseStates.zero_shot.hasItem, caseStates.zero_shot.reviewed)] : []),
        targetCase("few_shot", caseStates.few_shot.state, caseStates.few_shot.hasItem, caseStates.few_shot.reviewed),
        targetCase("foreign", "active", true, false),
      ];
      calls.caseListReads += 1;
      calls.caseListServedRunIds = items.map((item) => item.runId);
      return json(route, { schemaVersion: "socialgraph-fm.gfm-governance/2.0", items, total: items.length, offset: 0, limit: 100 });
    }
    if (method === "POST" && path === `${base}/adaptations/review-collections`) {
      const body = request.postDataJSON() as Parameters<typeof targetReviewCollection>[0];
      if (body?.targetTaskRegistrationId !== zeroRegistration.registrationId || body.runId !== zeroRun.runId || body.resultHash !== targetResult("zero_shot").resultHash) return rejectIdentity(route, method, path);
      caseStates.zero_shot = { exists: true, state: "active", hasItem: true, reviewed: false };
      return json(route, targetReviewCollection(body), 201);
    }
    if (method === "POST" && path === `${base}/adaptations/label-sets`) {
      const body = request.postDataJSON() as { targetTaskRegistrationId?: string; runId?: string; resultHash?: string };
      if (body.targetTaskRegistrationId !== fewRegistration.registrationId || body.runId !== fewRun.runId || body.resultHash !== targetResult("few_shot").resultHash) return rejectIdentity(route, method, path);
      return json(route, fewRegistration.labels, 201);
    }
    if (method === "POST" && path === `${base}/adaptations/label-sets/${fewRegistration.labels!.labelSetHash}/policies`) {
      const body = request.postDataJSON() as { schemaVersion?: string; targetTaskRegistrationId?: string; runId?: string; resultHash?: string };
      return body.schemaVersion === "socialgraph-fm.governance-target-review-policy-fit-request/1.0"
        && body.targetTaskRegistrationId === fewRegistration.registrationId
        && body.runId === fewRun.runId
        && body.resultHash === targetResult("few_shot").resultHash
        ? json(route, fewPolicy, 201)
        : rejectIdentity(route, method, path);
    }
    if (method === "GET" && path === `${base}/adaptations/policies/${fewPolicy.policyHash}`) {
      increment(calls.policyReads, fewPolicy.policyHash); return json(route, fewPolicy);
    }
    if (method === "GET" && path === `${base}/adaptations/runs/${fewRun.runId}/policies/${fewPolicy.policyHash}/comparison`) {
      if (url.searchParams.get("offset") !== "0" || url.searchParams.get("limit") !== "500" || [...url.searchParams.keys()].length !== 2) return rejectIdentity(route, method, `${path}${url.search}`);
      increment(calls.comparisonReads, `${fewRun.runId}:${fewPolicy.policyHash}`); return json(route, targetComparison(108, "few_shot"));
    }
    if (method === "POST" && path === `${base}/adaptations/policies/${fewPolicy.policyHash}/activate`) {
      const body = request.postDataJSON() as { targetTaskRegistrationId?: string };
      return body.targetTaskRegistrationId === fewRegistration.registrationId ? json(route, targetActivation("few_shot"), 201) : rejectIdentity(route, method, path);
    }
    if (method === "POST" && path === `${base}/adaptations/handoffs`) {
      const body = request.postDataJSON() as { targetTaskRegistrationId?: string; policyHash?: string };
      return body.targetTaskRegistrationId === fewRegistration.registrationId && body.policyHash === fewPolicy.policyHash ? json(route, fewHandoff, 201) : rejectIdentity(route, method, path);
    }
    if (method === "GET" && path === `${base}/adaptations/handoffs/${fewHandoff.handoffHash}`) {
      increment(calls.handoffReads, fewHandoff.handoffHash); return json(route, fewHandoff);
    }
    if (method === "GET" && path === `${base}/skills`) return json(route, { schemaVersion: GOVERNANCE_SKILLS_SCHEMA, items: GOVERNANCE_PUBLIC_SKILLS.map((name) => ({ name, readOnly: !["run_governance_analysis", "draft_review_report"].includes(name), confirmationRequired: ["run_governance_analysis", "draft_review_report"].includes(name), description: name, parameterSchema: { type: "object" } })), catalogHash: "d".repeat(64) });
    if (method === "POST" && path === `${base}/assistant/dispatch`) {
      const body = request.postDataJSON() as { answerMode?: string; graph?: { artifactId?: string; datasetContentHash?: string; graphVersionHash?: string }; model?: { modelVersionId?: string; modelStateHash?: string }; context?: { runId?: string; caseId?: string; selectedTarget?: { targetType?: string; targetId?: string } } };
      const mode = modes.find((candidate) => {
        const registration = registrationFor(candidate); const run = runFor(candidate);
        return body.graph?.artifactId === registration.artifact.artifactId && body.graph.datasetContentHash === registration.artifact.datasetContentHash
          && body.graph.graphVersionHash === registration.artifact.graphVersionHash && body.model?.modelVersionId === run.modelVersionId
          && body.model.modelStateHash === run.modelStateHash && body.context?.runId === run.runId
          && (!body.context.caseId || body.context.caseId === caseIdFor(candidate))
          && (!body.context.selectedTarget || body.context.selectedTarget.targetType === "node"
            && targetResult(candidate).findings.some((finding) => finding.nodeId === body.context?.selectedTarget?.targetId));
      });
      if (!mode) return rejectIdentity(route, method, path);
      increment(calls.assistantReads, runFor(mode).runId);
      return json(route, {
        schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
        dispatchId: `governance-dispatch-${"8".repeat(32)}`,
        intent: "answer",
        answerMode: body.answerMode ?? "overview",
        status: "completed",
        answer: "### 证据核对\n\n请核对直接关系与邻域对象，再记录人工判断。",
        result: {},
        deterministicFallback: true,
        generationMode: "deterministic_report",
        fallbackPhase: null,
        reasonCode: null,
        evidenceRefs: [],
        confirmation: null,
        navigation: null,
        skillCalls: [{ skill: "inspect_graph", requestHash: "8".repeat(64), resultHash: "9".repeat(64) }],
        citedHashes: ["a".repeat(64)],
        auditHash: "a".repeat(64),
      });
    }
    if (method === "POST" && path === `${base}/assistant/turn`) {
      const body = request.postDataJSON() as { graph?: { artifactId?: string; datasetContentHash?: string; graphVersionHash?: string }; model?: { modelVersionId?: string; modelStateHash?: string }; context?: { runId?: string; caseId?: string; selectedNodeIds?: string[] } };
      const mode = modes.find((candidate) => {
        const registration = registrationFor(candidate); const run = runFor(candidate);
        return body.graph?.artifactId === registration.artifact.artifactId && body.graph.datasetContentHash === registration.artifact.datasetContentHash
          && body.graph.graphVersionHash === registration.artifact.graphVersionHash && body.model?.modelVersionId === run.modelVersionId
          && body.model.modelStateHash === run.modelStateHash && body.context?.runId === run.runId
          && (!body.context.caseId || body.context.caseId === caseIdFor(candidate))
          && (body.context.selectedNodeIds ?? []).every((nodeId) => targetResult(candidate).findings.some((finding) => finding.nodeId === nodeId));
      });
      if (!mode) return rejectIdentity(route, method, path);
      increment(calls.assistantReads, runFor(mode).runId);
      return json(route, { schemaVersion: GOVERNANCE_ASSISTANT_SCHEMA, turnId: `governance-turn-${"8".repeat(32)}`, answer: "### 证据核对\n\n请核对直接关系与邻域对象，再记录人工判断。", deterministicFallback: true, skillCalls: [{ skill: "inspect_graph", requestHash: "8".repeat(64), resultHash: "9".repeat(64) }], citedHashes: ["a".repeat(64)], auditHash: "a".repeat(64) });
    }
    if (method === "POST" && path === `${base}/knowledge/search`) {
      const body = request.postDataJSON() as { graph?: { artifactId?: string; datasetContentHash?: string; graphVersionHash?: string }; model?: { modelVersionId?: string; modelStateHash?: string } };
      const mode = modes.find((candidate) => { const registration = registrationFor(candidate); const run = runFor(candidate); return body.graph?.artifactId === registration.artifact.artifactId && body.graph.datasetContentHash === registration.artifact.datasetContentHash && body.graph.graphVersionHash === registration.artifact.graphVersionHash && body.model?.modelVersionId === run.modelVersionId && body.model.modelStateHash === run.modelStateHash; });
      if (!mode) return rejectIdentity(route, method, path);
      increment(calls.knowledgeReads, registrationFor(mode).artifact.artifactId);
      return json(route, { schemaVersion: GOVERNANCE_SKILLS_SCHEMA, items: [{ sourceLabel: "目标任务治理知识", sourceUri: `project://target-governance/${mode}`, contentHash: "8".repeat(64), chunkHash: "9".repeat(64), text: "复核优先级与图事实分别保存。", rank: 1 }], indexHash: "a".repeat(64), auditHash: "b".repeat(64) });
    }
    if (method === "POST" && path === `${base}/similar-cases/search`) {
      const body = request.postDataJSON() as { graph?: { artifactId?: string; datasetContentHash?: string; graphVersionHash?: string }; model?: { modelVersionId?: string; modelStateHash?: string }; runId?: string; caseId?: string; kindEntries?: { kind?: string; targetIds?: string[] }[] };
      const mode = modes.find((candidate) => {
        const registration = registrationFor(candidate); const run = runFor(candidate);
        const exactQuery = body.caseId === caseIdFor(candidate) && !body.runId && !body.kindEntries
          || body.runId === run.runId && !body.caseId && Boolean(body.kindEntries?.length) && body.kindEntries!.every((entry) => entry.kind === "node" && entry.targetIds?.every((nodeId) => targetResult(candidate).findings.some((finding) => finding.nodeId === nodeId)));
        return body.graph?.artifactId === registration.artifact.artifactId && body.graph.datasetContentHash === registration.artifact.datasetContentHash
          && body.graph.graphVersionHash === registration.artifact.graphVersionHash && body.model?.modelVersionId === run.modelVersionId
          && body.model.modelStateHash === run.modelStateHash && exactQuery;
      });
      if (!mode) return rejectIdentity(route, method, path);
      const run = runFor(mode);
      increment(calls.similarReads, run.runId);
      return json(route, { schemaVersion: GOVERNANCE_SKILLS_SCHEMA, query: body.caseId ? { caseId: body.caseId } : { runId: body.runId, kindEntries: body.kindEntries }, items: [{ caseId: `case-${"9".repeat(32)}`, score: 0.84, components: { embedding: 0.8, structure: 0.9, modality: 0.7 }, graphVersionHash: run.graphVersionHash, modelStateHash: run.modelStateHash, kindKey: "node", kindEntries: [{ kind: "node", targetIds: ["target-node-001"] }], concludedAt: "2026-08-20T00:00:00Z", recordHash: "b".repeat(64) }], indexHash: "c".repeat(64), backfill: { indexed: 1 }, auditHash: "d".repeat(64) });
    }
    if (method === "POST" && path === `${base}/cases`) {
      const body = request.postDataJSON() as { runId?: string };
      const mode = modes.find((candidate) => body.runId === runFor(candidate).runId);
      if (!mode) return rejectIdentity(route, method, path);
      caseStates[mode] = { exists: true, state: "draft", hasItem: false, reviewed: false };
      return json(route, targetCase(mode, "draft", false, false), 201);
    }
    for (const mode of modes) {
      const caseId = caseIdFor(mode); const state = caseStates[mode];
      if (method === "POST" && path === `${base}/cases/${caseId}/items`) { state.hasItem = true; return json(route, targetCase(mode, state.state, true, state.reviewed), 201); }
      if (method === "POST" && path === `${base}/cases/${caseId}/transitions`) { const body = request.postDataJSON() as { state?: "draft" | "active" | "concluded" }; state.state = body.state ?? state.state; return json(route, targetCase(mode, state.state, state.hasItem, state.reviewed)); }
      if (method === "POST" && path === `${base}/cases/${caseId}/review-events`) { state.reviewed = true; return json(route, targetCase(mode, state.state, state.hasItem, true), 201); }
      if (method === "GET" && path === `${base}/cases/${caseId}`) return json(route, targetCase(mode, state.state, state.hasItem, state.reviewed));
      if (method === "GET" && path === `${base}/cases/${caseId}/report`) return route.fulfill({ status: 200, contentType: "text/html; charset=utf-8", headers: { "Content-Disposition": `attachment; filename="${caseId}.html"` }, body: "<!doctype html><title>Target review report</title>" });
    }
    calls.unmatched.push(`${method} ${path}${url.search}`);
    return json(route, { detail: { code: "GFM_GOVERNANCE_ROUTE_NOT_MOCKED", path } }, 404);
  });
  return calls;
}

async function uploadTask(page: Page, label: string, fileName: string) {
  await page.getByLabel(label).setInputFiles({ name: fileName, mimeType: "application/zip", buffer: Buffer.from(`PK ${fileName}`) });
}

async function runBaseline(lane: ReturnType<Page["getByRole"]>) {
  await lane.getByRole("button", { name: "开始分析" }).click();
  await lane.getByRole("button", { name: "确认分析" }).click();
  await expect(lane.getByText(/协同组群已就绪/u)).toBeVisible();
}

async function expectFocusRing(target: ReturnType<Page["getByRole"]>) {
  await target.focus(); await target.page().keyboard.press("Tab"); await target.page().keyboard.press("Shift+Tab");
  await expect(target).toBeFocused();
  expect(await target.evaluate((element) => {
    const style = getComputedStyle(element);
    return style.outlineStyle !== "none" && Number.parseFloat(style.outlineWidth) > 0 || style.boxShadow !== "none";
  })).toBe(true);
}

async function expectCenterHitTarget(target: ReturnType<Page["getByRole"]>) {
  const snapshot = await target.evaluate((element) => {
    const box = element.getBoundingClientRect();
    const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
    const describe = (candidate: Element | null) => {
      if (!(candidate instanceof HTMLElement)) return null;
      const rect = candidate.getBoundingClientRect();
      const style = getComputedStyle(candidate);
      return {
        className: candidate.className,
        rect: { left: rect.left, top: rect.top, width: rect.width, height: rect.height },
        scrollHeight: candidate.scrollHeight,
        scrollTop: candidate.scrollTop,
        overflowY: style.overflowY,
        position: style.position,
      };
    };
    return {
      box: { left: box.left, top: box.top, width: box.width, height: box.height },
      hitClass: hit?.getAttribute("class") ?? null,
      hitTag: hit?.tagName ?? null,
      hitTarget: hit === element || element.contains(hit),
      evidence: describe(element.closest(".governance-right")),
      body: describe(element.closest(".governance-body")),
      workspace: describe(element.closest(".governance-workspace")),
      page: describe(element.closest(".governance-page")),
      research: describe(element.closest(".research-scroll")),
    };
  });
  expect(snapshot.hitTarget, JSON.stringify(snapshot)).toBe(true);
}

async function contrastRatio(target: ReturnType<Page["getByRole"]>): Promise<number> {
  return target.evaluate((element) => {
    type Rgba = [number, number, number, number];
    const rgba = (value: string): Rgba => {
      const channels = value.match(/[\d.]+/gu)?.map(Number) ?? [];
      return [channels[0] ?? 0, channels[1] ?? 0, channels[2] ?? 0, channels[3] ?? 1];
    };
    const composite = (front: Rgba, back: Rgba): Rgba => {
      const alpha = front[3] + back[3] * (1 - front[3]);
      if (alpha === 0) return [0, 0, 0, 0];
      return [0, 1, 2].map((index) => (
        front[index] * front[3] + back[index] * back[3] * (1 - front[3])
      ) / alpha).concat(alpha) as Rgba;
    };
    const luminance = (color: Rgba) => color.slice(0, 3).map((value) => {
      const channel = value / 255;
      return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
    }).reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
    const ancestry: Element[] = [];
    for (let current: Element | null = element; current; current = current.parentElement) ancestry.push(current);
    let background: Rgba = [255, 255, 255, 1];
    for (const current of ancestry.reverse()) {
      background = composite(rgba(getComputedStyle(current).backgroundColor), background);
    }
    const foreground = luminance(composite(rgba(getComputedStyle(element).color), background));
    const back = luminance(background);
    return (Math.max(foreground, back) + 0.05) / (Math.min(foreground, back) + 0.05);
  });
}

test("independent target lanes hand off as selectable governance tasks without replacing the current session", async ({ page }) => {
  test.setTimeout(120_000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(`pageerror:${error.message}`));
  page.on("console", (message) => { if (message.type() === "error") errors.push(`console:${message.text()}`); });
  const apiCalls = await mockIndependentTargets(page);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/#/adaptation");
  const adaptation = page.getByRole("region", { name: "适配能力工作台" });
  await expect(adaptation).not.toContainText(/Thailand|sample|demo|experiment|schemaVersion|AdaptationReviewPolicy|probability|accuracy/iu);

  await uploadTask(page, "零样本目标任务包", "regional-a.sgtask.zip");
  const zero = page.getByRole("region", { name: "零样本路径" });
  await expect(zero).toContainText("108 个对象 · 220 条关系");
  await runBaseline(zero);
  await expect(zero.getByLabel("重点账号").getByRole("button", { name: /待治理核验/u })).toHaveCount(25);
  await zero.getByRole("button", { name: "进入治理应用" }).click();
  await expect(page).toHaveURL(/#\/governance$/u);
  const zeroRegistrationId = targetTaskRegistration("zero_shot").registrationId;
  const fewRegistrationId = targetTaskRegistration("few_shot").registrationId;
  await expect.poll(() => apiCalls.targetTaskReads[zeroRegistrationId]).toBeGreaterThanOrEqual(1);
  const selector = page.getByRole("navigation", { name: "治理任务" });
  await expect(selector.getByRole("button", { name: "当前会话治理" })).toBeVisible();
  await expect(selector.getByRole("button", { name: "Regional review task A" })).toHaveAttribute("aria-pressed", "true");
  await selector.getByRole("button", { name: "当前会话治理" }).click();
  await expect(selector.getByRole("button", { name: "当前会话治理" })).toHaveAttribute("aria-pressed", "true");
  const zeroReadsBeforeReselect = apiCalls.targetTaskReads[zeroRegistrationId];
  await selector.getByRole("button", { name: "Regional review task A" }).click();
  await expect(selector.getByRole("button", { name: "Regional review task A" })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => apiCalls.targetTaskReads[zeroRegistrationId]).toBeGreaterThan(zeroReadsBeforeReselect);

  const governance = page.getByTestId("governance-workspace");
  await expect(governance.getByText(/基础风险排序身份已重新校验/u)).toBeVisible();
  const governanceModes = governance.getByRole("navigation", { name: "治理工作模式" });
  await expect(governanceModes.getByRole("button", { name: "风险节点", exact: true })).toHaveAttribute("aria-pressed", "true");
  const firstCandidate = governance.locator(".governance-result-list[aria-label='风险节点'] .governance-result-list__select").first();
  await expect(firstCandidate).toBeVisible();
  await expect(firstCandidate).toHaveAccessibleName(/#1\s+对象 1.*风险排序 #1/u);
  await expect(firstCandidate).not.toContainText("适配后复核优先级");
  await firstCandidate.click();
  await expect(governance.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" })).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  const evidenceTrigger = governance.getByRole("button", { name: "查看 对象 1 的证据", exact: true });
  await evidenceTrigger.click();
  const evidence = page.getByRole("dialog", { name: "对象 1" });
  await evidence.getByRole("tab", { name: "人工复核", exact: true }).click();
  await expect(evidence.getByRole("combobox", { name: "当前研判单" })).toHaveValue(CASE_ID);
  await expect(evidence.getByRole("option")).toHaveCount(1);
  await expect(evidence).not.toContainText(/%/u);
  await page.keyboard.press("Escape");
  await expect(evidence).toHaveCount(0);
  await expect(evidenceTrigger).toBeFocused();
  await expect(governance.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" })).toBeVisible();

  await page.getByRole("button", { name: "研判助手", exact: true }).click();
  const rag = page.getByRole("complementary", { name: "案例研判助手" });
  await rag.getByLabel("输入研判问题").fill("请说明还需要核对哪些直接关系");
  await rag.getByRole("button", { name: "生成报告", exact: true }).click();
  await expect(rag.getByRole("heading", { name: "证据核对" })).toBeVisible();
  await expect(rag.getByRole("tab", { name: "分析链路" })).toHaveCount(0);
  await rag.getByRole("tab", { name: "历史案例" }).click();
  await rag.getByRole("button", { name: "检索相似历史案例" }).click();
  await expect(rag.getByText("历史案例 01")).toBeVisible();

  await governanceModes.getByRole("button", { name: "风险节点", exact: true }).click();
  await firstCandidate.click();
  await evidenceTrigger.click();
  await evidence.getByRole("tab", { name: "人工复核", exact: true }).click();
  await evidence.getByRole("textbox", { name: "复核理由" }).fill("补充直接关系证据。");
  await evidence.getByRole("button", { name: "待定", exact: true }).click();
  await expect(evidence.getByText(/当前人工结论：待定/u)).toBeVisible();
  expect(await contrastRatio(evidence.getByText(/当前人工结论：待定/u))).toBeGreaterThanOrEqual(4.5);
  await evidence.getByRole("button", { name: "关闭证据档案" }).click();
  await governanceModes.getByRole("button", { name: "研判单", exact: true }).click();
  const caseDetail = governance.locator(".governance-case-detail");
  await caseDetail.getByRole("button", { name: "形成结论" }).click();
  await expect(caseDetail).toContainText("已形成结论");
  const downloadPromise = page.waitForEvent("download");
  const htmlExport = governance.getByRole("button", { name: "HTML", exact: true });
  await htmlExport.click();
  expect((await downloadPromise).suggestedFilename()).toBe(`${CASE_ID}.html`);
  expect(await contrastRatio(htmlExport)).toBeGreaterThanOrEqual(3);

  await page.getByRole("button", { name: "适配能力", exact: true }).click();
  await uploadTask(page, "少样本目标任务包", "regional-b.sgtask.zip");
  const few = page.getByRole("region", { name: "少样本路径" });
  await expect(few).toHaveAttribute("data-phase", "raw");
  await expect(few).not.toContainText("正向 8 / 负向 8");
  await runBaseline(few);
  await expect(few).toHaveAttribute("data-phase", "compared");
  await expect(few).toContainText("协同组群已就绪 · 108 个账号");
  await expect(few.getByRole("group", { name: "排序图层" })).toHaveCount(0);
  await expect(few.getByRole("button", { name: "拟合冻结复核策略" })).toHaveCount(0);
  await few.getByRole("button", { name: "下一页" }).click();
  await expect(few.getByLabel("重点账号").getByRole("button", { name: "对象 26，待治理核验", exact: true })).toBeVisible();
  await few.getByRole("button", { name: "进入治理应用" }).click();
  await expect(page.getByRole("navigation", { name: "治理任务" }).getByRole("button", { name: "Regional review task B" })).toHaveAttribute("aria-pressed", "true");
  await expect.poll(() => apiCalls.targetTaskReads[fewRegistrationId]).toBeGreaterThanOrEqual(1);
  const adaptedGovernance = page.getByTestId("governance-workspace");
  await expect(adaptedGovernance.getByText(/少样本复核顺序已重新校验/u)).toBeVisible();
  const actionableNodeIds = new Set(targetResult("few_shot").findings
    .filter((finding) => finding.riskBand === "high" || finding.riskBand === "review")
    .map((finding) => finding.nodeId));
  const adaptedActionableTop = targetComparison(108, "few_shot").rows
    .filter((row) => actionableNodeIds.has(row.nodeId))
    .sort((left, right) => left.adaptedRank - right.adaptedRank)[0]!;
  const adaptedActionableLabel = `对象 ${Number(adaptedActionableTop.nodeId.slice(-3))}`;
  const adaptedActionableDelta = adaptedActionableTop.rankDelta < 0
    ? `上升 ${Math.abs(adaptedActionableTop.rankDelta)}`
    : `下降 ${adaptedActionableTop.rankDelta}`;
  const adaptedFirstCandidate = adaptedGovernance.locator(".governance-result-list[aria-label='风险节点'] .governance-result-list__select").first();
  await expect(adaptedFirstCandidate).toContainText(adaptedActionableLabel);
  await expect(adaptedFirstCandidate).toContainText(`基础排序 #${adaptedActionableTop.baseRank}`);
  await expect(adaptedFirstCandidate).toContainText(`适配后复核优先级 #${adaptedActionableTop.adaptedRank}`);
  await expect(adaptedFirstCandidate).toContainText(adaptedActionableDelta);
  await expect(page.getByRole("region", { name: "治理关系图" })).toContainText("适配后复核优先级");
  await adaptedFirstCandidate.click();
  await adaptedGovernance.getByRole("button", { name: `查看 ${adaptedActionableLabel} 的证据`, exact: true }).click();
  const fewEvidence = page.getByRole("dialog", { name: adaptedActionableLabel });
  await fewEvidence.getByRole("tab", { name: "人工复核", exact: true }).click();
  await expect(fewEvidence.getByRole("combobox", { name: "当前研判单" })).toHaveValue(FEW_CASE_ID);
  await expect(fewEvidence.getByRole("option")).toHaveCount(1);
  await expect(fewEvidence).toContainText("B 独立研判单");
  await expect(fewEvidence).toContainText("草稿");
  await expect(fewEvidence).not.toContainText("已形成结论");
  await expect(fewEvidence).not.toContainText("Foreign run case");
  await fewEvidence.getByRole("button", { name: "关闭证据档案" }).click();
  const zeroRegistration = targetTaskRegistration("zero_shot"); const fewRegistration = targetTaskRegistration("few_shot");
  const zeroRun = targetRun("zero_shot"); const fewRun = targetRun("few_shot");
  const fewPolicy = targetPolicy("few_shot"); const fewHandoff = adaptationHandoff("few_shot");
  expect(Object.keys(apiCalls.artifactReads).sort()).toEqual([zeroRegistration.artifact.artifactId, fewRegistration.artifact.artifactId].sort());
  expect(Object.keys(apiCalls.runReads).sort()).toEqual([zeroRun.runId, fewRun.runId].sort());
  expect(Object.keys(apiCalls.resultReads).sort()).toEqual([zeroRun.runId, fewRun.runId].sort());
  for (const [ledger, key] of [
    [apiCalls.artifactReads, fewRegistration.artifact.artifactId],
    [apiCalls.runReads, fewRun.runId],
    [apiCalls.resultReads, fewRun.runId],
    [apiCalls.scoredPreviewReads, fewRun.runId],
    [apiCalls.handoffReads, fewHandoff.handoffHash],
    [apiCalls.policyReads, fewPolicy.policyHash],
    [apiCalls.comparisonReads, `${fewRun.runId}:${fewPolicy.policyHash}`],
  ] as const) expect(ledger[key]).toBeGreaterThanOrEqual(1);
  expect(apiCalls.caseListReads).toBeGreaterThanOrEqual(1);
  expect(apiCalls.caseListServedRunIds.sort()).toEqual([zeroRun.runId, fewRun.runId, FOREIGN_RUN_ID].sort());
  expect(apiCalls.evidenceReads[`${zeroRun.runId}:target-node-001`]).toBeGreaterThanOrEqual(1);
  expect(apiCalls.evidenceReads[`${fewRun.runId}:${adaptedActionableTop.nodeId}`]).toBeGreaterThanOrEqual(1);
  expect(apiCalls.assistantReads[zeroRun.runId]).toBeGreaterThanOrEqual(1);
  expect(apiCalls.knowledgeReads[zeroRegistration.artifact.artifactId]).toBe(0);
  expect(apiCalls.similarReads[zeroRun.runId]).toBeGreaterThanOrEqual(1);

  await page.setViewportSize({ width: 640, height: 360 });
  const mobileWorkspace = page.getByTestId("governance-workspace");
  const mobileCandidate = mobileWorkspace.locator(".governance-result-list[aria-label='风险节点'] .governance-result-list__select").first();
  await mobileCandidate.click();
  await mobileWorkspace.getByRole("button", { name: "查看证据", exact: true }).click();
  const mobileEvidence = page.getByRole("dialog", { name: adaptedActionableLabel });
  const mobileCloseEvidence = mobileEvidence.getByRole("button", { name: "关闭证据档案" });
  await expectCenterHitTarget(mobileCloseEvidence);
  await mobileCloseEvidence.click();
  await expect(mobileCandidate).toBeVisible();
  await mobileCandidate.click();
  await page.getByRole("tab", { name: "图谱", exact: true }).click();
  await expect(page.getByRole("region", { name: "治理关系图" })).toBeVisible();
  await page.getByRole("tab", { name: "任务", exact: true }).click();
  await expect(mobileWorkspace.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" })).toBeVisible();
  const reflow = await page.evaluate(() => ({ client: document.documentElement.clientWidth, scroll: document.documentElement.scrollWidth }));
  expect(reflow.scroll).toBeLessThanOrEqual(reflow.client + 1);
  const motionRisks = await page.locator("button, .graph-preview").evaluateAll((elements) => elements.flatMap((element) => {
    const style = getComputedStyle(element); const durations = `${style.animationDuration},${style.transitionDuration}`.split(",").map((part) => Number.parseFloat(part) * (part.includes("ms") ? 0.001 : 1));
    return durations.some((value) => value > 0.01) ? [element.getAttribute("aria-label") ?? element.textContent?.trim().slice(0, 40) ?? element.tagName] : [];
  }));
  expect(motionRisks).toEqual([]);
  expect(Object.keys(apiCalls.targetTaskReads).sort()).toEqual([fewRegistrationId, zeroRegistrationId].sort());
  expect(apiCalls.unmatched).toEqual([]);
  expect(errors).toEqual([]);
});
