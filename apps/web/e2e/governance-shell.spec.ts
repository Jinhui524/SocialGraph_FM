import { expect, test, type Page, type Route } from "@playwright/test";
import { Buffer } from "node:buffer";

import {
  GOVERNANCE_ARTIFACT_ID,
  GOVERNANCE_RUN_ID,
  GOVERNANCE_HASHES,
  onlineArtifact,
  artifactCompatibility,
  onlineCapabilities,
  derivationPage,
  evidencePayload,
  findingPage,
  onlineHealth,
  onlineResult,
  onlineRun,
  onlineRunPreview,
} from "../src/test/fixtures/governanceOnline";
import {
  ASSISTANT_SKILL_RESULT_SCHEMA,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
} from "../src/types/governanceSkills";

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
}

function readablePreview() {
  const core = Array.from({ length: 100 }, (_, index) => ({
    id: index < 3 ? `n${index + 1}` : `core-${String(index).padStart(3, "0")}`,
    label: index < 3 ? `匿名账号 ${index + 1}` : `Core ${index}`,
    degree: index === 0 ? 99 : 1,
    structureMissing: false,
    score: null,
    riskBand: null,
    groupId: null,
  }));
  const isolated = Array.from({ length: 10 }, (_, index) => ({
    id: `isolated-${String(index).padStart(2, "0")}`,
    label: `Isolated ${index}`,
    degree: 0,
    structureMissing: true,
    score: null,
    riskBand: null,
    groupId: null,
  }));
  return {
    schemaVersion: "socialgraph-fm.gfm-governance/2.0",
    artifactId: `governance-artifact-${"1".repeat(32)}`,
    datasetContentHash: GOVERNANCE_HASHES.dataset,
    graphVersionHash: GOVERNANCE_HASHES.graph,
    runId: null,
    resultHash: null,
    nodes: [...core, ...isolated],
    edges: core.slice(1).map((node, index) => ({ id: `edge-${index}`, source: core[0].id, target: node.id, modalities: ["coRT"], factual: true })),
    nodeCount: 110,
    edgeCount: 99,
    partialPreview: false,
    previewHash: GOVERNANCE_HASHES.preview,
  };
}

async function mockGovernance(
  page: Page,
  previewRequests: string[],
  requestCounts: { materialize: number },
  options: { automaticReportGate?: Promise<void> } = {},
) {
  let analysisCompleted = false;
  await page.route("**/api/v2/gfm/governance/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    if (request.method() === "GET" && path.endsWith("/health")) return json(route, onlineHealth());
    if (request.method() === "GET" && path.endsWith("/capabilities")) return json(route, onlineCapabilities());
    if (request.method() === "GET" && path.endsWith("/runs")) return json(route, {
      schemaVersion: "socialgraph-fm.gfm-governance/2.0",
      items: analysisCompleted ? [onlineRun()] : [],
      total: analysisCompleted ? 1 : 0,
      offset: 0,
      limit: 50,
    });
    if (request.method() === "POST" && path.endsWith("/artifacts/compatibility")) return json(route, {
      ...artifactCompatibility(), nodeCount: 110, relationRowCount: 99,
    });
    if (request.method() === "POST" && path.endsWith("/artifacts")) {
      requestCounts.materialize += 1;
      return json(route, { ...onlineArtifact(), nodeCount: 110, relationRowCount: 99 }, 201);
    }
    if (request.method() === "GET" && /^\/api\/v2\/gfm\/governance\/artifacts\/[^/]+\/preview$/u.test(path)) {
      previewRequests.push(url.search);
      return json(route, readablePreview());
    }
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/artifacts/${GOVERNANCE_ARTIFACT_ID}`) {
      return json(route, { ...onlineArtifact(), nodeCount: 110, relationRowCount: 99 });
    }
    if (request.method() === "POST" && path.endsWith("/runs")) return json(route, onlineRun(), 201);
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}`) return json(route, onlineRun());
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/result`) return json(route, {
      ...onlineResult(),
      distribution: { low: 108, review: 1, high: 1, predictedPositive: 1, total: 110 },
    });
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/nodes`) return json(route, findingPage());
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/graph-preview`) {
      previewRequests.push(url.search);
      return json(route, {
        ...onlineRunPreview(),
        ...readablePreview(),
        runId: GOVERNANCE_RUN_ID,
        resultHash: GOVERNANCE_HASHES.result,
        nodes: readablePreview().nodes.map((node, index) => ({
          ...node,
          score: index === 0 ? 0.92 : index === 1 ? 0.58 : 0.2,
          riskBand: index === 0 ? "high" : index === 1 ? "review" : "low",
          groupId: index < 2 ? "group-1" : null,
        })),
      });
    }
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/groups`) {
      return json(route, { ...derivationPage("group"), limit: 10_000 });
    }
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/relations`) {
      return json(route, { ...derivationPage("factual_relation"), limit: 10_000 });
    }
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/potential-links`) {
      return json(route, { ...derivationPage("potential_link"), limit: 10_000 });
    }
    if (request.method() === "GET" && path === "/api/v2/gfm/governance/cases") return json(route, {
      schemaVersion: "socialgraph-fm.gfm-governance/2.0", items: [], total: 0, offset: 0, limit: 100,
    });
    if (request.method() === "GET" && path === `/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/nodes/n1/evidence`) return json(route, evidencePayload());
    if (request.method() === "GET" && path.endsWith("/skills")) return json(route, {
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      items: GOVERNANCE_PUBLIC_SKILLS.map((name) => ({
        name,
        readOnly: !["run_governance_analysis", "draft_review_report"].includes(name),
        confirmationRequired: ["run_governance_analysis", "draft_review_report"].includes(name),
        description: name,
        parameterSchema: { type: "object" },
      })),
      catalogHash: "d".repeat(64),
    });
    if (request.method() === "POST" && path.endsWith("/skills/execute")) {
      const payload = request.postDataJSON() as {
        readonly skill?: string;
      };
      if (payload.skill !== "run_governance_analysis") return route.abort("failed");
      return json(route, {
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        executionId: `governance-exec-${"4".repeat(32)}`,
        skill: "run_governance_analysis",
        status: "confirmation_required",
        result: { plan: { topK: 100 } },
        confirmation: {
          token: `governance-confirm-${"5".repeat(64)}`,
          action: "run_governance_analysis",
          requestDigest: "6".repeat(64),
          expiresAt: "2026-08-23T12:00:00Z",
        },
        provenance: { inputHash: "7".repeat(64) },
        auditHash: "8".repeat(64),
      });
    }
    if (request.method() === "POST" && path.endsWith("/assistant/execute")) {
      const payload = request.postDataJSON() as {
        readonly skill?: string;
        readonly context?: { readonly runId?: string };
      };
      if (payload.skill === "generate_global_situation_report") {
        await options.automaticReportGate;
      }
      return json(route, {
        schemaVersion: ASSISTANT_SKILL_RESULT_SCHEMA,
        executionId: `assistant-exec-${"7".repeat(32)}`,
        skill: payload.skill,
        answer: "## 研判结论\n当前结果已按事实关系与模型排序整理。\n\n## 人工复核建议\n优先核对高关注账号及其一跳关系。",
        result: {},
        evidenceRefs: [{ label: "当前分析结果", sourceKind: "skill", hash: GOVERNANCE_HASHES.result }],
        skillCalls: [{ skill: "inspect_graph", requestHash: GOVERNANCE_HASHES.request, resultHash: GOVERNANCE_HASHES.result }],
        citedHashes: [GOVERNANCE_HASHES.result],
        auditHash: "8".repeat(64),
      });
    }
    if (request.method() === "POST" && path.endsWith("/skills/confirm")) {
      analysisCompleted = true;
      return json(route, {
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        action: "run_governance_analysis",
        status: "completed",
        result: onlineRun(),
        auditHash: "8".repeat(64),
      });
    }
    return json(route, { detail: { code: "GFM_GOVERNANCE_ROUTE_NOT_MOCKED" } }, 404);
  });
}

async function expectNoControlOverlap(page: Page) {
  const collisions = await page.locator(".governance-topbar, .governance-runbar").evaluateAll((containers) => {
    const overlaps: string[] = [];
    for (const container of containers) {
      const controls = [...container.querySelectorAll<HTMLElement>("button, summary, input, progress")]
        .filter((element) => {
          const style = getComputedStyle(element);
          const rect = element.getBoundingClientRect();
          return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
        });
      for (let left = 0; left < controls.length; left += 1) {
        for (let right = left + 1; right < controls.length; right += 1) {
          const a = controls[left].getBoundingClientRect();
          const b = controls[right].getBoundingClientRect();
          const width = Math.min(a.right, b.right) - Math.max(a.left, b.left);
          const height = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
          if (width > 1 && height > 1) overlaps.push(`${controls[left].tagName}:${controls[left].textContent}|${controls[right].tagName}:${controls[right].textContent}`);
        }
      }
    }
    return overlaps;
  });
  expect(collisions).toEqual([]);
}

async function readCamera(page: Page) {
  return page.locator('.graph-preview[aria-label="治理关系图"]').evaluate((element) => ({
    x: Number((element as HTMLElement).dataset.cameraX),
    y: Number((element as HTMLElement).dataset.cameraY),
    zoom: Number((element as HTMLElement).dataset.cameraZoom),
    worldCenterX: Number((element as HTMLElement).dataset.worldCenterX),
    worldCenterY: Number((element as HTMLElement).dataset.worldCenterY),
  }));
}

async function expectOuterResizers(page: Page, expected: { left: boolean; right: boolean }) {
  await expect(page.getByRole("separator", { name: "调整项目导航宽度" })).toHaveCount(expected.left ? 1 : 0);
  await expect(page.getByRole("separator", { name: "调整图谱栏宽度" })).toHaveCount(expected.right ? 1 : 0);
}

async function readPaneGeometry(page: Page) {
  return page.evaluate(() => {
    const app = document.querySelector<HTMLElement>(".app-shell");
    const sidebar = document.querySelector<HTMLElement>(".sidebar-host");
    const graph = document.querySelector<HTMLElement>(".graph-column");
    if (!app || !sidebar || !graph) throw new Error("workspace pane geometry is unavailable");
    const style = getComputedStyle(app);
    return {
      leftVariable: Number.parseFloat(style.getPropertyValue("--workspace-left-width")),
      rightVariable: Number.parseFloat(style.getPropertyValue("--workspace-right-width")),
      sidebarWidth: sidebar.getBoundingClientRect().width,
      graphWidth: graph.getBoundingClientRect().width,
    };
  });
}

async function readGraphStability(page: Page) {
  return page.locator('.graph-preview[aria-label="治理关系图"]').evaluate((element) => {
    const root = element as HTMLElement;
    return {
      layoutCount: Number(root.dataset.layoutCount),
      fitViewCount: Number(root.dataset.fitViewCount),
      engineCreateCount: Number(root.dataset.engineCreateCount),
      engineDestroyCount: Number(root.dataset.engineDestroyCount),
      rendererRequested: root.dataset.rendererRequested ?? null,
      rendererResolved: root.dataset.rendererResolved ?? null,
      theme: root.classList.contains("graph-preview--focus-dark") ? "focus-dark" : "brand-light",
      overlay: root.querySelector(".graph-preview__overlay-badge")?.textContent?.trim() ?? null,
      mutationInFlight: Number(root.dataset.mutationInFlight),
      cameraX: Number(root.dataset.cameraX),
      cameraY: Number(root.dataset.cameraY),
      cameraZoom: Number(root.dataset.cameraZoom),
      worldCenterX: Number(root.dataset.worldCenterX),
      worldCenterY: Number(root.dataset.worldCenterY),
      dragNodeId: root.dataset.draggableNodeId ?? null,
      dragX: Number(root.dataset.draggableX),
      dragY: Number(root.dataset.draggableY),
      nodeId: root.dataset.coordinateNodeId ?? null,
      nodeX: Number(root.dataset.coordinateNodeX),
      nodeY: Number(root.dataset.coordinateNodeY),
    };
  });
}

function expectStableGraph(before: Awaited<ReturnType<typeof readGraphStability>>, after: Awaited<ReturnType<typeof readGraphStability>>) {
  expect(after).toMatchObject({
    layoutCount: before.layoutCount,
    fitViewCount: before.fitViewCount,
    engineCreateCount: before.engineCreateCount,
    engineDestroyCount: before.engineDestroyCount,
    rendererRequested: before.rendererRequested,
    rendererResolved: before.rendererResolved,
    theme: before.theme,
    overlay: before.overlay,
    mutationInFlight: 0,
    nodeId: before.nodeId,
  });
  expect([after.cameraX, after.cameraY, before.cameraX, before.cameraY].every(Number.isFinite)).toBe(true);
  expect(Math.abs(after.cameraZoom - before.cameraZoom)).toBeLessThan(0.001);
  expect([before.worldCenterX, before.worldCenterY, after.worldCenterX, after.worldCenterY].every(Number.isFinite)).toBe(true);
  expect(Math.hypot(after.worldCenterX - before.worldCenterX, after.worldCenterY - before.worldCenterY)).toBeLessThan(0.1);
  expect(before.nodeId).toBeTruthy();
  expect([before.nodeX, before.nodeY, after.nodeX, after.nodeY].every(Number.isFinite)).toBe(true);
  expect(Math.hypot(after.nodeX - before.nodeX, after.nodeY - before.nodeY)).toBeLessThan(0.1);
}

async function readReadyGraphStability(page: Page) {
  const graph = page.locator('.graph-preview[aria-label="治理关系图"]');
  await expect(graph).toHaveAttribute("data-world-center-x", /^-?\d/u);
  await expect(graph).toHaveAttribute("data-world-center-y", /^-?\d/u);
  await expect(graph).toHaveAttribute("data-draggable-node-id", /.+/u);
  await expect(graph).toHaveAttribute("data-coordinate-node-id", /.+/u);
  await expect.poll(async () => {
    const snapshot = await readGraphStability(page);
    return snapshot.mutationInFlight === 0
      && snapshot.dragNodeId === snapshot.nodeId
      && [
        snapshot.worldCenterX,
        snapshot.worldCenterY,
        snapshot.dragX,
        snapshot.dragY,
        snapshot.nodeX,
        snapshot.nodeY,
      ].every(Number.isFinite);
  }).toBe(true);
  return readGraphStability(page);
}

async function expectStableGraphAfterResize(
  page: Page,
  before: Awaited<ReturnType<typeof readGraphStability>>,
) {
  await page.evaluate(() => new Promise<void>((resolve) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolve()));
  }));
  await expect.poll(async () => (await readGraphStability(page)).mutationInFlight).toBe(0);
  await expect.poll(async () => {
    const after = await readGraphStability(page);
    return {
      layoutCount: after.layoutCount,
      fitViewCount: after.fitViewCount,
      engineCreateCount: after.engineCreateCount,
      engineDestroyCount: after.engineDestroyCount,
      nodeId: after.nodeId,
    };
  }).toEqual({
    layoutCount: before.layoutCount,
    fitViewCount: before.fitViewCount,
    engineCreateCount: before.engineCreateCount,
    engineDestroyCount: before.engineDestroyCount,
    nodeId: before.nodeId,
  });
  await expect.poll(async () => {
    const after = await readGraphStability(page);
    return Math.hypot(after.worldCenterX - before.worldCenterX, after.worldCenterY - before.worldCenterY);
  }).toBeLessThan(0.1);
  await expect.poll(async () => {
    const after = await readGraphStability(page);
    return Math.hypot(after.nodeX - before.nodeX, after.nodeY - before.nodeY);
  }).toBeLessThan(0.1);
  const after = await readGraphStability(page);
  expectStableGraph(before, after);
  return after;
}

async function openGovernancePackage(
  page: Page,
  viewport = { width: 1280, height: 720 },
) {
  const previewRequests: string[] = [];
  const requestCounts = { materialize: 0 };
  await mockGovernance(page, previewRequests, requestCounts);
  await page.setViewportSize(viewport);
  await page.goto("/");
  const conversationUpload = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(conversationUpload).toBeEnabled();
  await conversationUpload.setInputFiles({
    name: "current-session.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("mock governance zip"),
  });
  await expect(page.locator(".chat-message--user").filter({ hasText: "current-session.zip" })).toBeVisible({ timeout: 30_000 });
  await page.getByLabel("研究问题", { exact: true }).fill("开始分析");
  await page.getByRole("button", { name: "发送研究问题", exact: true }).click();
  const confirm = page.getByRole("button", { name: "确认开始分析", exact: true });
  await expect(confirm).toBeVisible();
  await confirm.click();
  await expect(page.getByRole("button", { name: "打开复核工作台", exact: true })).toBeVisible({ timeout: 20_000 });
  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  const workspace = page.getByTestId("governance-workspace");
  const graph = page.locator('.graph-preview[aria-label="治理关系图"]');
  await expect(workspace).toBeVisible();
  await expect(graph).toHaveAttribute("data-graph-ready", "true", { timeout: 20_000 });
  await expect(workspace.getByRole("button", { name: "风险节点", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(workspace.locator(".governance-result-list[aria-label='风险节点'] article")).toHaveCount(2);
  return { graph, previewRequests, requestCounts, workspace };
}

test("automatic completion holds at 95 until its LLM report is ready", async ({ page }) => {
  test.setTimeout(60_000);
  let releaseReport!: () => void;
  const reportGate = new Promise<void>((resolve) => { releaseReport = resolve; });
  const mockOptions: { automaticReportGate?: Promise<void> } = {};
  await mockGovernance(page, [], { materialize: 0 }, mockOptions);
  await page.goto("/");
  const conversationUpload = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(conversationUpload).toBeEnabled();
  await conversationUpload.setInputFiles({
    name: "current-session.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("mock governance zip"),
  });
  await expect(page.locator(".chat-message--user").filter({ hasText: "current-session.zip" })).toBeVisible({ timeout: 30_000 });
  mockOptions.automaticReportGate = reportGate;
  await page.getByLabel("研究问题", { exact: true }).fill("开始分析");
  await page.getByRole("button", { name: "发送研究问题", exact: true }).click();
  const automaticReportRequest = page.waitForRequest((request) => request.method() === "POST"
    && new URL(request.url()).pathname.endsWith("/assistant/execute")
    && (request.postData() ?? "").includes('"skill":"generate_global_situation_report"'));
  await page.getByRole("button", { name: "确认开始分析", exact: true }).click();

  const progress = page.getByRole("region", { name: "治理分析进度" });
  await expect(progress.getByRole("progressbar", { name: "治理分析完成 95%" })).toBeVisible();
  await expect(progress).toContainText("正在整理分析结论");
  expect((await automaticReportRequest).postDataJSON()).toMatchObject({ skill: "generate_global_situation_report" });
  await expect(page.getByRole("heading", { name: "研判结论", exact: true })).toHaveCount(0);
  await page.evaluate(() => {
    const states: Array<{ progress: string | null; report: boolean }> = [];
    const capture = () => states.push({
      progress: document.querySelector<HTMLProgressElement>('.governance-run-progress progress')?.getAttribute("value") ?? null,
      report: [...document.querySelectorAll(".chat-message--assistant h2")].some((heading) => heading.textContent?.trim() === "研判结论"),
    });
    capture();
    new MutationObserver(capture).observe(document.querySelector(".conversation")!, { childList: true, subtree: true, attributes: true });
    (window as typeof window & { __governanceSyncStates?: typeof states }).__governanceSyncStates = states;
  });

  releaseReport();
  await expect(progress.getByRole("progressbar", { name: "治理分析完成 100%" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "研判结论", exact: true })).toBeVisible();
  await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
  const states = await page.evaluate(() => (window as typeof window & {
    __governanceSyncStates?: Array<{ progress: string | null; report: boolean }>;
  }).__governanceSyncStates ?? []);
  expect(states.some((state) => state.progress === "100" && !state.report)).toBe(false);
  expect(states.at(-1)).toEqual({ progress: "100", report: true });
});

test("governance consumes the conversation result and assistant exposes reports and history only", async ({ page }) => {
  const { workspace } = await openGovernancePackage(page);
  const modeNavigation = workspace.getByRole("navigation", { name: "治理工作模式" });
  for (const mode of ["风险节点", "群组与关系", "研判单", "研判助手"]) {
    await expect(modeNavigation.getByRole("button", { name: mode, exact: true })).toBeVisible();
  }
  await expect(modeNavigation.getByLabel("运行记录", { exact: true })).toBeVisible();
  for (const removedCopy of ["推理包", "分析引擎在线", "分析完成", "风险分布", "风险概览", "群组概览"]) {
    await expect(workspace.getByText(removedCopy, { exact: false })).toHaveCount(0);
  }
  await expect(workspace.locator('input[type="file"]')).toHaveCount(0);
  await expect(workspace.getByRole("button", { name: /开始分析|重新分析/u })).toHaveCount(0);

  await modeNavigation.getByRole("button", { name: "研判助手", exact: true }).click();
  const assistantWorkbench = page.getByRole("complementary", { name: "案例研判助手" });
  await expect(assistantWorkbench).toBeVisible();
  await expect(assistantWorkbench.getByRole("tab", { name: "研判报告", exact: true })).toBeVisible();
  await expect(assistantWorkbench.getByRole("tab", { name: "历史案例", exact: true })).toBeVisible();
  await expect(assistantWorkbench.getByRole("tab", { name: "分析链路" })).toHaveCount(0);

  await expect(assistantWorkbench.getByRole("region", { name: "当前研判上下文" })).toHaveCount(0);
  await expect(assistantWorkbench.getByText(/未选择对象|研判单已建立/u)).toHaveCount(0);

  const reportTasks = assistantWorkbench.getByRole("group", { name: "研判报告任务" });
  for (const report of ["全局态势报告", "当前账号证据报告", "群组与关系研判报告", "人工研判草稿"]) {
    await expect(reportTasks.getByRole("button", { name: new RegExp(report, "u") })).toBeVisible();
  }
  await reportTasks.getByRole("button", { name: /全局态势报告/u }).click();
  const report = assistantWorkbench.locator(".governance-assistant-answer");
  await expect(report).toBeVisible();
  await expect(report).toContainText("人工复核建议");
  await expect(reportTasks).toHaveCount(0);
  await expect(assistantWorkbench.getByRole("button", { name: "更换报告任务", exact: true })).toBeVisible();
  await expect(report.locator("details").filter({ hasText: "依据来源" })).not.toHaveAttribute("open", "");

  await assistantWorkbench.getByRole("tab", { name: "历史案例", exact: true }).click();
  await expect(assistantWorkbench.getByRole("tabpanel", { name: "历史案例" })).toBeVisible();
});

test("assistant composer remains complete while task and report content scroll independently", async ({ page }) => {
  test.setTimeout(90_000);
  const { workspace } = await openGovernancePackage(page, { width: 1440, height: 900 });
  await workspace.getByRole("button", { name: "研判助手", exact: true }).click();
  const assistant = page.getByRole("complementary", { name: "案例研判助手" });
  await expect(assistant).toBeVisible();

  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 1280, height: 720 },
    { width: 1024, height: 768 },
    { width: 720, height: 792 },
    { width: 390, height: 844 },
    { width: 960, height: 540 },
  ]) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => new Promise<void>((resolve) => requestAnimationFrame(() => requestAnimationFrame(() => resolve()))));
    const bounds = await assistant.evaluate((panel) => {
      const view = panel.querySelector<HTMLElement>(".governance-rag-view--assistant")!;
      const content = panel.querySelector<HTMLElement>(".governance-assistant-content")!;
      const composer = panel.querySelector<HTMLElement>(".governance-assistant-composer")!;
      const panelRect = panel.getBoundingClientRect();
      const viewRect = view.getBoundingClientRect();
      const composerRect = composer.getBoundingClientRect();
      return {
        panelTop: panelRect.top,
        panelBottom: panelRect.bottom,
        viewTop: viewRect.top,
        viewBottom: viewRect.bottom,
        composerTop: composerRect.top,
        composerBottom: composerRect.bottom,
        composerHeight: composerRect.height,
        contentClientHeight: content.clientHeight,
        contentScrollHeight: content.scrollHeight,
        bodyOverflow: document.documentElement.scrollWidth - innerWidth,
      };
    });
    expect(bounds.composerHeight, `composer height at ${viewport.width}x${viewport.height}`).toBeGreaterThanOrEqual(44);
    expect(bounds.composerTop).toBeGreaterThanOrEqual(bounds.viewTop - 1);
    expect(bounds.composerBottom).toBeLessThanOrEqual(bounds.viewBottom + 1);
    expect(bounds.composerBottom).toBeLessThanOrEqual(bounds.panelBottom + 1);
    expect(bounds.composerBottom).toBeLessThanOrEqual(viewport.height + 1);
    expect(bounds.bodyOverflow).toBeLessThanOrEqual(0);
    expect(bounds.contentScrollHeight).toBeGreaterThanOrEqual(bounds.contentClientHeight);
    for (const task of ["全局态势报告", "当前账号证据报告", "群组与关系研判报告", "人工研判草稿"]) {
      await expect(assistant.getByRole("button", { name: new RegExp(task, "u") })).toBeAttached();
    }
  }
});

test("research task cards execute the explicit question Skill after a governed graph is ready", async ({ page }) => {
  await openGovernancePackage(page);
  await page.getByRole("button", { name: "对话研究", exact: true }).click();
  await page.getByRole("button", { name: "查看开始页", exact: true }).click();

  const taskCards = page.locator(".welcome-atlas .prompt-card");
  await expect(taskCards).toHaveCount(3);
  for (const card of await taskCards.all()) {
    await expect(card.locator(".prompt-card__copy")).toHaveCount(1);
    const icon = await card.locator(".prompt-icon").boundingBox();
    const glyph = await card.locator(".prompt-icon svg").boundingBox();
    expect(icon).not.toBeNull();
    expect(glyph).not.toBeNull();
    expect(Math.abs(icon!.x + icon!.width / 2 - (glyph!.x + glyph!.width / 2))).toBeLessThan(0.5);
    expect(Math.abs(icon!.y + icon!.height / 2 - (glyph!.y + glyph!.height / 2))).toBeLessThan(0.5);
  }

  const tasks = [
    { title: "图谱基本情况", skill: "answer_governance_question", carriesRun: false },
    { title: "人工复核流程", skill: "answer_governance_question", carriesRun: true },
    { title: "证据核对清单", skill: "answer_governance_question", carriesRun: true },
  ] as const;

  for (const [taskIndex, task] of tasks.entries()) {
    if (taskIndex > 0) await page.getByRole("button", { name: "查看开始页", exact: true }).click();
    const responsePromise = page.waitForResponse((response) => {
      const url = new URL(response.url());
      return response.request().method() === "POST"
        && url.pathname.endsWith("/api/v2/gfm/governance/assistant/execute")
        && (response.request().postData() ?? "").includes(`\"skill\":\"${task.skill}\"`);
    });
    await page.getByRole("button", { name: new RegExp(task.title, "u") }).click();
    const response = await responsePromise;
    const payload = response.request().postDataJSON() as {
      readonly skill?: string;
      readonly context?: { readonly runId?: string };
    };
    expect(payload.skill).toBe(task.skill);
    expect(Boolean(payload.context?.runId)).toBe(task.carriesRun);
    await expect(page.locator(".assistant-card.is-success").last()).toBeVisible();
  }
});

test("governance shell preserves one graph across viewport, resize, theme, and lens changes", async ({ page }) => {
  test.setTimeout(90_000);
  const { graph, previewRequests, workspace } = await openGovernancePackage(page);
  await expect(page.getByText("Russia 动态样例")).toHaveCount(0);
  await expect(page.getByText("协议验证")).toHaveCount(0);
  await expect(page.getByText(/In-domain|Low-label|Cross-domain|RQ4/)).toHaveCount(0);
  await expect(page.getByText("SocialGraph-FM Research")).toHaveCount(0);
  await expect(page.locator(".graph-workspace-surface:not([hidden]) .graph-preview")).toHaveCount(1);
  await expect(page.locator(".governance-right")).toHaveCount(0);
  expect(await workspace.evaluate((element) => getComputedStyle(element).position)).not.toBe("fixed");
  await expect(graph).toHaveClass(/graph-preview--focus-dark/);
  await expectNoControlOverlap(page);
  await expect(workspace).toContainText("当前会话治理");
  await expect(workspace.getByText("current-session.zip", { exact: true })).toHaveCount(0);
  await expect(page.getByText("输入合同", { exact: false })).toHaveCount(0);
  await expect(page.getByText("模型版本", { exact: true })).toHaveCount(0);
  await expect(page.getByText("modelStateHash", { exact: true })).toHaveCount(0);

  await expectOuterResizers(page, { left: false, right: true });
  const compactPaneBefore = await readPaneGeometry(page);
  const compactGraphBefore = await readReadyGraphStability(page);
  const compactRightResizer = page.getByRole("separator", { name: "调整图谱栏宽度" });
  const compactRightWidth = Number(await compactRightResizer.getAttribute("aria-valuenow"));
  await compactRightResizer.press("ArrowLeft");
  await expect.poll(async () => Number(await compactRightResizer.getAttribute("aria-valuenow"))).toBe(compactRightWidth + 10);
  await expect.poll(async () => (await readPaneGeometry(page)).graphWidth).toBeCloseTo(compactPaneBefore.graphWidth + 10, 1);
  const compactPaneAfter = await readPaneGeometry(page);
  expect(compactPaneAfter.rightVariable).toBe(compactRightWidth + 10);
  expect(compactPaneAfter.sidebarWidth).toBeCloseTo(compactPaneBefore.sidebarWidth, 1);
  let transitionSnapshot = await expectStableGraphAfterResize(page, compactGraphBefore);

  for (const viewport of [{ width: 1440, height: 900 }, { width: 1920, height: 1080 }]) {
    await page.setViewportSize(viewport);
    await expectOuterResizers(page, { left: true, right: true });
    await expectNoControlOverlap(page);
    transitionSnapshot = await expectStableGraphAfterResize(page, transitionSnapshot);
  }

  const desktopPaneBefore = await readPaneGeometry(page);
  const desktopLeftResizer = page.getByRole("separator", { name: "调整项目导航宽度" });
  const desktopLeftWidth = Number(await desktopLeftResizer.getAttribute("aria-valuenow"));
  await desktopLeftResizer.press("ArrowRight");
  await expect.poll(async () => Number(await desktopLeftResizer.getAttribute("aria-valuenow"))).toBe(desktopLeftWidth + 10);
  await expect.poll(async () => (await readPaneGeometry(page)).sidebarWidth).toBeCloseTo(desktopPaneBefore.sidebarWidth + 10, 1);
  const desktopPaneAfterLeft = await readPaneGeometry(page);
  expect(desktopPaneAfterLeft.leftVariable).toBe(desktopLeftWidth + 10);
  transitionSnapshot = await expectStableGraphAfterResize(page, transitionSnapshot);

  const desktopRightResizer = page.getByRole("separator", { name: "调整图谱栏宽度" });
  const desktopRightWidth = Number(await desktopRightResizer.getAttribute("aria-valuenow"));
  await desktopRightResizer.press("ArrowLeft");
  await expect.poll(async () => Number(await desktopRightResizer.getAttribute("aria-valuenow"))).toBe(desktopRightWidth + 10);
  await expect.poll(async () => (await readPaneGeometry(page)).graphWidth).toBeCloseTo(desktopPaneAfterLeft.graphWidth + 10, 1);
  const desktopPaneAfterRight = await readPaneGeometry(page);
  expect(desktopPaneAfterRight.rightVariable).toBe(desktopRightWidth + 10);
  transitionSnapshot = await expectStableGraphAfterResize(page, transitionSnapshot);

  await page.setViewportSize({ width: 1024, height: 768 });
  await expectOuterResizers(page, { left: false, right: true });
  transitionSnapshot = await expectStableGraphAfterResize(page, transitionSnapshot);
  await page.setViewportSize({ width: 1280, height: 720 });
  await expectOuterResizers(page, { left: false, right: true });
  await expectStableGraphAfterResize(page, transitionSnapshot);

  const canvas = graph.locator("canvas").first();
  await expect(canvas).toBeVisible();
  await expect.poll(() => canvas.evaluate((element) => (element as HTMLCanvasElement).toDataURL().length)).toBeGreaterThan(1_000);

  await page.getByRole("button", { name: "切换到品牌浅色主题" }).click();
  await expect(graph).not.toHaveClass(/graph-preview--focus-dark/);
  await page.getByRole("button", { name: "切换到专注深色主题" }).click();
  await expect(graph).toHaveClass(/graph-preview--focus-dark/);
  const modeNavigation = workspace.getByRole("navigation", { name: "治理工作模式" });
  await modeNavigation.getByRole("button", { name: "群组与关系", exact: true }).click();
  await workspace.getByRole("tab", { name: /事实关系/u }).click();
  await expect.poll(() => previewRequests.some((query) => query.includes("preset=relation") && query.includes("nodeBudget=80") && query.includes("edgeBudget=160"))).toBe(true);
  await expect(graph).toHaveClass(/graph-preview--focus-dark/);
  await modeNavigation.getByRole("button", { name: "风险节点", exact: true }).click();
  await expect(page.locator(".graph-workspace-surface:not([hidden]) .graph-preview")).toHaveCount(1);
  await expect(graph).toHaveAttribute("data-graph-ready", "true");
});

test("actionable risk nodes replace the old overview at every responsive breakpoint", async ({ page }) => {
  test.setTimeout(90_000);
  const { graph, workspace } = await openGovernancePackage(page);
  const riskList = workspace.locator(".governance-result-list[aria-label='风险节点']");

  await expect(riskList.locator("article")).toHaveCount(2);
  await expect(riskList).toContainText("高风险候选");
  await expect(riskList).toContainText("建议复核");
  await expect(riskList).not.toContainText("低风险参照");
  await expect(page.getByRole("region", { name: "治理分析摘要" })).toHaveCount(0);
  await expect(page.getByRole("complementary", { name: "治理分析摘要" })).toHaveCount(0);
  await expect(page.getByText("风险分布", { exact: true })).toHaveCount(0);

  for (const viewport of [
    { width: 1024, height: 768 },
    { width: 390, height: 844 },
    { width: 1440, height: 900 },
  ]) {
    await page.setViewportSize(viewport);
    await expect(workspace).toBeVisible();
    await expect(riskList.locator("article").first()).toBeVisible();
    await expect(page.locator("body")).toHaveJSProperty("scrollWidth", viewport.width);
  }
  await expect(graph).toHaveAttribute("data-graph-ready", "true");
});

test("candidate and graph selections highlight context without changing camera or projection", async ({ page }) => {
  test.setTimeout(90_000);
  const { graph, previewRequests, workspace } = await openGovernancePackage(page, { width: 1440, height: 900 });
  await expect(page.getByRole("separator", { name: "调整治理证据栏宽度" })).toHaveCount(0);
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("socialgraph-fm-workspace-layout-v1") ?? "{}").governanceEvidenceWidth)).toBeUndefined();

  const cameraBefore = await readCamera(page);
  const projectionRequestsBefore = previewRequests.length;
  const appearanceBefore = await graph.getAttribute("data-appearance-request-key");
  await workspace.locator(".governance-result-list__select").first().click();
  const selection = workspace.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" });
  await expect(selection).toContainText("视角位置保持不变");
  await expect.poll(() => graph.getAttribute("data-appearance-request-key")).not.toBe(appearanceBefore);
  expect(previewRequests).toHaveLength(projectionRequestsBefore);
  const cameraAfterCandidate = await readCamera(page);
  expect(cameraAfterCandidate).toEqual(cameraBefore);
  await expect(page.getByRole("dialog")).toHaveCount(0);

  await page.keyboard.press("Escape");
  await expect(selection).toHaveCount(0);
  const canvas = graph.locator(".graph-preview__canvas");
  const box = await canvas.boundingBox();
  const nodeX = Number(await graph.getAttribute("data-draggable-x"));
  const nodeY = Number(await graph.getAttribute("data-draggable-y"));
  expect(box).not.toBeNull();
  await page.mouse.click(box!.x + nodeX, box!.y + nodeY);
  await expect(selection).toBeVisible();
  expect(await readCamera(page)).toEqual(cameraBefore);
});

test("evidence opens as a centred, dismissible dossier with on-demand LLM assistance", async ({ page }) => {
  test.setTimeout(90_000);
  const { workspace } = await openGovernancePackage(page, { width: 1440, height: 900 });
  const evidenceTrigger = workspace.getByRole("button", { name: "查看 匿名账号 1 的证据", exact: true });
  await evidenceTrigger.click();
  const dossier = page.getByRole("dialog", { name: "匿名账号 1" });
  await expect(dossier).toBeVisible();
  await expect(dossier).toHaveAttribute("aria-modal", "true");
  await expect(dossier.getByRole("tablist", { name: "证据档案内容" })).toBeVisible();
  await expect(dossier.getByRole("button", { name: "关闭证据档案" })).toBeFocused();

  const dialogBox = await dossier.boundingBox();
  expect(dialogBox).not.toBeNull();
  expect(dialogBox!.width).toBeGreaterThanOrEqual(880);
  expect(dialogBox!.width).toBeLessThanOrEqual(960);
  expect(Math.abs(dialogBox!.x + dialogBox!.width / 2 - 720)).toBeLessThan(2);
  expect(dialogBox!.height).toBeLessThanOrEqual(900 * 0.82 + 1);

  await expect(dossier.getByRole("tabpanel", { name: "证据摘要" })).toContainText("关注原因");
  const generate = dossier.getByRole("button", { name: "生成证据研判摘要", exact: true });
  await expect(generate).toBeEnabled();
  await generate.click();
  await expect(dossier.locator(".governance-dossier-ai__answer")).toContainText("人工复核建议");

  await dossier.getByRole("tab", { name: "关系事实", exact: true }).click();
  const facts = dossier.getByRole("tabpanel", { name: "关系事实" });
  await expect(facts).toContainText("关联账号");
  await expect(facts).toContainText("关系模态");
  await expect(facts).toContainText("当前证据只包含关系两端账号、关系模态、可用权重与绑定哈希");
  await expect(facts).toContainText("发布时间、原帖内容及采集来源需要在人工复核中补充");

  await page.keyboard.press("Escape");
  await expect(dossier).toHaveCount(0);
  await expect(evidenceTrigger).toBeFocused();
  await expect(workspace.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" })).toBeVisible();
});

test("groups and relations give risk-linked structures distinct product views", async ({ page }) => {
  test.setTimeout(90_000);
  const { graph, workspace } = await openGovernancePackage(page, { width: 1440, height: 900 });
  await expect(graph.locator('.graph-preview__overlay-badge[title="风险优先级"]')).toBeVisible();
  await workspace.getByRole("button", { name: "群组与关系", exact: true }).click();
  const relationTabs = workspace.getByRole("tablist", { name: "群组与关系类型" });
  await expect(relationTabs.getByRole("tab", { name: /风险群组/u })).toHaveAttribute("aria-selected", "true");
  await expect(graph.locator('.graph-preview__overlay-badge[title="协同群组"]')).toBeVisible();
  const groupRow = workspace.locator(".governance-result-list__select").first();
  await expect(groupRow).toContainText("2 个账号的风险群组");
  await expect(groupRow).toContainText("高风险 1 · 建议复核 1");

  await relationTabs.getByRole("tab", { name: /事实关系/u }).click();
  await expect(graph.locator('.graph-preview__overlay-badge[title="事实关系 / 潜在线索"]')).toBeVisible();
  await expect(workspace.locator(".governance-result-list__select").first()).toContainText("事实关系");
  await expect(workspace.locator(".governance-result-list__select").first()).toContainText("高风险候选 / 建议复核");

  await relationTabs.getByRole("tab", { name: /潜在线索/u }).click();
  await expect(workspace.locator(".governance-result-list__select").first()).toContainText("潜在线索（非事实边）");
  await workspace.getByRole("button", { name: "风险节点", exact: true }).click();
  await expect(graph.locator('.graph-preview__overlay-badge[title="风险优先级"]')).toBeVisible();
  await expect(graph).toHaveAttribute("data-graph-ready", "true");
});

test("governance preserves camera and assistant drafts across workspace and viewport changes", async ({ page }) => {
  test.setTimeout(90_000);
  const { graph, requestCounts, workspace } = await openGovernancePackage(page, { width: 1440, height: 900 });

  await page.getByRole("button", { name: "研判助手", exact: true }).click();
  await expect(page.getByRole("complementary", { name: "案例研判助手" })).toBeVisible();
  const assistantDraft = page.getByPlaceholder("继续追问当前对象的事实关系、潜在线索或核验缺口");
  await assistantDraft.fill("保留这份未提交的研判草稿");

  await expect(page.locator(".graph-workspace-surface:not([hidden]) .graph-preview")).toHaveCount(1);
  await expect(graph).toHaveClass(/graph-preview--focus-dark/);
  await page.waitForLoadState("networkidle");
  await expect(graph).toHaveAttribute("data-graph-ready", "true");
  const canvas = graph.locator("canvas").first();
  await expect(canvas).toBeVisible();

  const initialCamera = await readCamera(page);
  await graph.getByRole("button", { name: "放大图谱" }).click();
  await graph.getByRole("button", { name: "放大图谱" }).click();
  // Both controls animate. Wait for the second operation, rather than taking
  // the first animation's intermediate camera as the persistence baseline.
  const expectedZoom = initialCamera.zoom * 1.22 * 1.22;
  await expect.poll(async () => Math.abs((await readCamera(page)).zoom - expectedZoom)).toBeLessThan(0.001);
  const transformedCamera = await readCamera(page);

  expect(requestCounts.materialize).toBe(1);
  await page.getByRole("button", { name: "对话研究", exact: true }).click();
  await expect(workspace).toBeHidden();
  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  await expect(workspace).toBeVisible();
  await expect(assistantDraft).toHaveValue("保留这份未提交的研判草稿");
  await expect(graph).toHaveClass(/graph-preview--focus-dark/);
  await expect(graph).toHaveAttribute("data-graph-ready", "true");
  await expect.poll(async () => {
    const restored = await readCamera(page);
    return Math.hypot(
      restored.worldCenterX - transformedCamera.worldCenterX,
      restored.worldCenterY - transformedCamera.worldCenterY,
    );
  }).toBeLessThan(0.1);
  await expect.poll(async () => Math.abs((await readCamera(page)).zoom - transformedCamera.zoom)).toBeLessThan(0.001);
  expect(requestCounts.materialize).toBe(1);

  for (const viewport of [
    { width: 1366, height: 768, left: false },
    { width: 1440, height: 900, left: true },
    { width: 1920, height: 1080, left: true },
    { width: 1024, height: 768, left: false },
  ]) {
    await page.setViewportSize(viewport);
    await expectOuterResizers(page, { left: viewport.left, right: true });
    await expectNoControlOverlap(page);
    await expect(page.locator(".graph-workspace-surface:not([hidden]) .graph-preview")).toHaveCount(1);
    await expect(canvas).toBeVisible();
  }
  await expect(page.getByRole("separator", { name: "调整治理证据栏宽度" })).toHaveCount(0);

  await page.setViewportSize({ width: 390, height: 844 });
  await expectOuterResizers(page, { left: false, right: false });
  await expect(page.getByRole("separator")).toHaveCount(0);
  await page.getByRole("tab", { name: "图谱" }).click();
  await expect(page.locator(".graph-workspace-surface:not([hidden]) .graph-preview")).toHaveCount(1);
  await expect(canvas).toBeVisible();
  await expect.poll(() => canvas.evaluate((element) => (element as HTMLCanvasElement).toDataURL().length)).toBeGreaterThan(1_000);
  await expect(page.getByRole("tab", { name: "证据" })).toHaveCount(0);
  await page.getByRole("tab", { name: "任务" }).click();
  await expect(workspace).toBeVisible();
  await expect(workspace.getByRole("navigation", { name: "治理工作模式" })).toBeVisible();

  await page.reload();
  await page.getByRole("button", { name: "打开导航" }).click();
  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  const restoredWorkspace = page.getByTestId("governance-workspace");
  await expect(restoredWorkspace.getByRole("button", { name: "风险节点", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(restoredWorkspace.locator(".governance-result-list[aria-label='风险节点'] article").first()).toBeVisible();
  await expect(restoredWorkspace.locator('input[type="file"]')).toHaveCount(0);
  await expect(page.locator('.graph-preview[aria-label="治理关系图"]')).toHaveClass(/graph-preview--focus-dark/);
});

test("chat package completion stays compact and exposes raw facts before analysis", async ({ page }) => {
  const previewRequests: string[] = [];
  const requestCounts = { materialize: 0 };
  await mockGovernance(page, previewRequests, requestCounts);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/#/research");
  await page.getByRole("button", { name: "打开导航" }).click();
  const previousSessionId = await page.evaluate(() => window.localStorage.getItem("socialgraph-fm-active-session"));
  await page.getByRole("button", { name: "新建研究会话", exact: true }).click();
  await expect.poll(() => page.evaluate(() => window.localStorage.getItem("socialgraph-fm-active-session")))
    .not.toBe(previousSessionId);

  const upload = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(upload).toBeEnabled();
  await upload.setInputFiles({
    name: "compact-governance.zip",
    mimeType: "application/zip",
    buffer: Buffer.from("mock compact governance package"),
  });

  const userMessage = page.locator(".chat-message--user").last();
  await expect(userMessage.getByText("compact-governance.zip", { exact: true })).toBeVisible();
  await expect(userMessage.locator(".user-message-attachments")).toHaveCount(1);
  await expect(userMessage.getByLabel("附件已就绪")).toBeVisible();
  expect(await userMessage.locator(".file-badge__copy small").evaluate((element) => (
    Number.parseFloat(getComputedStyle(element).fontSize)
  ))).toBeGreaterThanOrEqual(12);
  const guidance = "推理包已就绪。输入“开始分析”可创建一次需确认的治理分析，获得风险账号排序、协同群组和重点关系；结果不会改写图事实，完成后请进入治理应用复核并记录结论。";
  const guidanceMessage = page.locator(".chat-message--assistant").filter({ hasText: guidance });
  await expect(page.locator(".chat-message--assistant")).toHaveCount(1);
  await expect(guidanceMessage.getByText(guidance, { exact: true })).toHaveCount(1);
  await expect(page.getByText(/推理包已通过兼容性检查|上传完成/u)).toHaveCount(0);
  await expect(page.locator(".import-timeline")).toHaveCount(0);

  await page.reload();
  await expect(page.locator(".chat-message--assistant")).toHaveCount(1);
  await expect(page.getByText(guidance, { exact: true })).toHaveCount(1);
  await expect(page.getByText(/推理包已通过兼容性检查|上传完成/u)).toHaveCount(0);
  await expect(page.locator(".import-timeline")).toHaveCount(0);

  const graph = page.locator('.graph-preview[aria-label="交互式社交关系图预览"]');
  await expect(graph).toBeHidden();
  await expect(graph).toHaveClass(/graph-preview--with-overlay/);
  await expect(graph.locator(".graph-preview__overlay-badge")).toContainText("原始图事实");
  await page.getByRole("tab", { name: "图谱" }).click();
  await expect(graph).toBeVisible();
  await expect(graph).toHaveAttribute("data-graph-ready", "true", { timeout: 20_000 });
});
