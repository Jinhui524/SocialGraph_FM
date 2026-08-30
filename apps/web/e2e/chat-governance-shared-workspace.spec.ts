import { expect, test, type APIRequestContext, type Page } from "@playwright/test";
import { resolve } from "node:path";

interface ArtifactIdentity {
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly nodeCount: number;
}

interface RunIdentity extends ArtifactIdentity {
  readonly runId: string;
  readonly modelVersionId: string;
}

const apiOrigin = (process.env.SOCIALGRAPH_E2E_API_ORIGIN ?? "").replace(/\/$/u, "");

function apiUrl(path: string): string {
  return apiOrigin ? `${apiOrigin}${path}` : path;
}

async function artifactRunIds(request: APIRequestContext, artifactId: string): Promise<Set<string>> {
  const response = await request.get(apiUrl("/api/v2/gfm/governance/runs?offset=0&limit=100"));
  expect(response.ok()).toBe(true);
  const payload = await response.json() as { readonly items?: readonly RunIdentity[] };
  return new Set((payload.items ?? [])
    .filter((item) => item.artifactId === artifactId)
    .map((item) => item.runId));
}

async function requireOnlineForward(request: APIRequestContext): Promise<{
  readonly onlineForwardReady?: boolean;
  readonly modelVersionId?: string;
}> {
  const healthResponse = await request.get(apiUrl("/api/v2/gfm/governance/health"));
  expect(
    healthResponse.ok(),
    "The release E2E gate requires the managed SocialGraph-FM Governance service to be reachable.",
  ).toBe(true);
  const health = await healthResponse.json() as {
    readonly onlineForwardReady?: boolean;
    readonly modelVersionId?: string;
  };
  expect(
    health.onlineForwardReady,
    "The release E2E gate requires the deployed Global checkpoint to be online-forward ready.",
  ).toBe(true);
  return health;
}

async function graphVersionCount(page: Page): Promise<number> {
  return page.evaluate(() => new Promise<number>((resolveCount, rejectCount) => {
    const openRequest = indexedDB.open("socialgraph-fm");
    openRequest.onerror = () => rejectCount(openRequest.error);
    openRequest.onsuccess = () => {
      const database = openRequest.result;
      const transaction = database.transaction("graphVersions", "readonly");
      const countRequest = transaction.objectStore("graphVersions").count();
      countRequest.onerror = () => rejectCount(countRequest.error);
      countRequest.onsuccess = () => resolveCount(countRequest.result);
      transaction.oncomplete = () => database.close();
    };
  }));
}

test.describe("@backend managed SocialGraph-FM Governance shared workspace", () => {
test("chat upload is reused by governance and analysis stays confirmation-gated", async ({ page, request }) => {
  test.setTimeout(180_000);
  const health = await requireOnlineForward(request);

  let artifactUploadCount = 0;
  page.on("request", (browserRequest) => {
    const url = new URL(browserRequest.url());
    if (browserRequest.method() === "POST" && url.pathname === "/api/v2/gfm/governance/artifacts") {
      artifactUploadCount += 1;
    }
  });

  await page.goto("/");
  // Uploads are enabled only after the IndexedDB-backed session is hydrated.
  const uploadInput = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(uploadInput).toBeEnabled({ timeout: 30_000 });
  const uploadResponsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return response.request().method() === "POST"
      && url.pathname === "/api/v2/gfm/governance/artifacts";
  }, { timeout: 30_000 });
  await uploadInput.setInputFiles(resolve(
    process.cwd(),
    "../../examples/governance/russia/russia-04.zip",
  ));
  const uploadResponse = await uploadResponsePromise;
  expect(uploadResponse.ok()).toBe(true);
  const artifact = await uploadResponse.json() as ArtifactIdentity;

  await expect(page.getByText("russia-04.zip", { exact: true })).toBeVisible({ timeout: 30_000 });
  const compactUploadMessage = page.locator(".chat-message--user").filter({ hasText: "russia-04.zip" }).last();
  await expect(compactUploadMessage.locator(".user-message-attachments")).toHaveCount(1);
  await expect(compactUploadMessage.getByLabel("附件已就绪")).toBeVisible();
  await expect(compactUploadMessage.locator(".user-message-attachments + .user-message-bubble")).toHaveCount(1);
  await expect(page.getByText("推理包已就绪。输入“开始分析”可创建一次需确认的治理分析，获得风险账号排序、协同群组和重点关系；结果不会改写图事实，完成后请进入治理应用复核并记录结论。", { exact: true })).toHaveCount(1);
  await expect(page.locator('.conversation time')).toHaveCount(0);
  await expect(page.getByText(/推理包已通过兼容性检查|上传完成/u)).toHaveCount(0);
  const chatGraph = page.locator('.graph-preview[aria-label="交互式社交关系图预览"]');
  await expect(chatGraph).toHaveAttribute("data-graph-ready", "true", { timeout: 30_000 });
  const chatVisibleNodes = Number(await chatGraph.getAttribute("data-visible-nodes"));
  expect(chatVisibleNodes).toBeGreaterThan(0);
  expect(chatVisibleNodes).toBeLessThanOrEqual(artifact.nodeCount);
  expect(artifactUploadCount).toBe(1);

  await page.reload();
  await expect(page.getByText("russia-04.zip", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText("推理包已就绪。输入“开始分析”可创建一次需确认的治理分析，获得风险账号排序、协同群组和重点关系；结果不会改写图事实，完成后请进入治理应用复核并记录结论。", { exact: true })).toHaveCount(1);
  await expect(page.locator('.conversation time')).toHaveCount(0);
  await expect(chatGraph).toHaveAttribute("data-graph-ready", "true", { timeout: 30_000 });
  expect(Number(await chatGraph.getAttribute("data-visible-nodes"))).toBe(chatVisibleNodes);
  expect(artifactUploadCount).toBe(1);

  const canvas = chatGraph.locator(".graph-preview__canvas");
  const canvasBox = await canvas.boundingBox();
  expect(canvasBox).not.toBeNull();
  const targetId = await chatGraph.getAttribute("data-draggable-node-id");
  const targetX = Number(await chatGraph.getAttribute("data-draggable-x"));
  const targetY = Number(await chatGraph.getAttribute("data-draggable-y"));
  expect(targetId).toBeTruthy();
  expect(Number.isFinite(targetX) && Number.isFinite(targetY)).toBe(true);
  const start = { x: canvasBox!.x + targetX, y: canvasBox!.y + targetY };
  const initialLayoutCount = await chatGraph.getAttribute("data-layout-count");
  const initialCamera = await chatGraph.evaluate((element) => ({
    x: element.getAttribute("data-camera-x"),
    y: element.getAttribute("data-camera-y"),
    zoom: element.getAttribute("data-camera-zoom"),
  }));
  const frameCountBeforeJitter = Number(await chatGraph.getAttribute("data-local-force-frame-count"));
  const settledBeforeJitter = Number(await chatGraph.getAttribute("data-local-force-settled-generation"));

  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(start.x + 2, start.y + 1);
  await page.mouse.up();
  await page.waitForTimeout(150);
  expect(Number(await chatGraph.getAttribute("data-local-force-frame-count"))).toBe(frameCountBeforeJitter);
  expect(Number(await chatGraph.getAttribute("data-local-force-settled-generation"))).toBe(settledBeforeJitter);
  expect(await chatGraph.getAttribute("data-layout-count")).toBe(initialLayoutCount);

  const frameCountBeforeDrag = Number(await chatGraph.getAttribute("data-local-force-frame-count"));
  const settledBeforeDrag = Number(await chatGraph.getAttribute("data-local-force-settled-generation"));
  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(start.x + 40, start.y, { steps: 8 });
  await page.mouse.up();

  await expect.poll(
    async () => Number(await chatGraph.getAttribute("data-local-force-frame-count")),
    { timeout: 5_000 },
  ).toBeGreaterThan(frameCountBeforeDrag);
  await expect.poll(
    async () => Number(await chatGraph.getAttribute("data-local-force-moved-node-count")),
    { timeout: 5_000 },
  ).toBeGreaterThan(0);
  expect(Number(await chatGraph.getAttribute("data-local-force-neighbor-delta-max"))).toBeGreaterThan(0.5);
  await expect.poll(
    async () => Number(await chatGraph.getAttribute("data-local-force-settled-generation")),
    { timeout: 3_000 },
  ).toBeGreaterThan(settledBeforeDrag);
  expect(await chatGraph.getAttribute("data-last-dragged-id")).toBe(targetId);
  expect(Math.abs(Number(await chatGraph.getAttribute("data-draggable-x")) - targetX)).toBeGreaterThan(20);
  expect(await chatGraph.getAttribute("data-layout-count")).toBe(initialLayoutCount);
  const cameraAfterDrag = await chatGraph.evaluate((element) => ({
    x: Number(element.getAttribute("data-camera-x")),
    y: Number(element.getAttribute("data-camera-y")),
    zoom: Number(element.getAttribute("data-camera-zoom")),
  }));
  expect(Math.hypot(
    cameraAfterDrag.x - Number(initialCamera.x),
    cameraAfterDrag.y - Number(initialCamera.y),
  )).toBeLessThan(0.1);
  expect(Math.abs(cameraAfterDrag.zoom - Number(initialCamera.zoom))).toBeLessThan(0.001);

  const settledPosition = {
    x: await chatGraph.getAttribute("data-draggable-x"),
    y: await chatGraph.getAttribute("data-draggable-y"),
    frames: await chatGraph.getAttribute("data-local-force-frame-count"),
  };
  await page.waitForTimeout(250);
  expect({
    x: await chatGraph.getAttribute("data-draggable-x"),
    y: await chatGraph.getAttribute("data-draggable-y"),
    frames: await chatGraph.getAttribute("data-local-force-frame-count"),
  }).toEqual(settledPosition);

  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  const governance = page.getByTestId("governance-workspace");
  await expect(governance).toBeVisible();
  await expect(governance.getByRole("heading", { name: "当前会话暂无治理结果" })).toBeVisible();
  await expect(governance.getByRole("button", { name: "返回对话研究", exact: true })).toBeVisible();
  await expect(governance.getByText("russia-04.zip", { exact: true })).toHaveCount(0);
  await expect(governance.getByRole("button", { name: /开始分析|重新分析/u })).toHaveCount(0);
  await expect(governance.locator('input[type="file"]')).toHaveCount(0);
  const governanceGraph = page.locator('.graph-preview[aria-label="治理关系图"]');
  await expect(governanceGraph).toHaveAttribute("data-graph-ready", "true", { timeout: 30_000 });
  expect(Number(await governanceGraph.getAttribute("data-visible-nodes"))).toBe(chatVisibleNodes);
  expect(artifactUploadCount).toBe(1);

  await page.getByRole("button", { name: "对话研究", exact: true }).click();
  const beforeDispatch = await artifactRunIds(request, artifact.artifactId);
  await page.getByLabel("研究问题", { exact: true }).fill("开始分析");
  await page.getByRole("button", { name: "发送研究问题" }).click();

  const confirm = page.getByRole("button", { name: "确认开始分析", exact: true });
  await expect(confirm).toBeVisible({ timeout: 30_000 });
  const planningMessageFromConfirm = confirm.locator("xpath=ancestor::article");
  const planningMessageId = await planningMessageFromConfirm.getAttribute("data-message-id");
  expect(planningMessageId).toBeTruthy();
  const planningMessage = page.locator(`article[data-message-id="${planningMessageId}"]`);
  await expect(page.getByText("确认前不会产生运行或写入记录", { exact: true })).toBeVisible();
  const afterDispatch = await artifactRunIds(request, artifact.artifactId);
  expect(afterDispatch).toEqual(beforeDispatch);
  expect(artifactUploadCount).toBe(1);

  const confirmationResponsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return response.request().method() === "POST"
      && url.pathname === "/api/v2/gfm/governance/skills/confirm";
  });
  const reportResponsePromise = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return response.request().method() === "POST"
      && url.pathname === "/api/v2/gfm/governance/assistant/dispatch"
      && (response.request().postData() ?? "").includes('"runId"');
  }, { timeout: 120_000 });
  await confirm.click();
  const confirmationResponse = await confirmationResponsePromise;
  expect(confirmationResponse.ok()).toBe(true);
  const confirmation = await confirmationResponse.json() as {
    readonly action: string;
    readonly result: RunIdentity;
  };
  expect(confirmation.action).toBe("run_governance_analysis");
  await expect(planningMessage.getByRole("region", { name: "治理分析进度" })).toBeVisible({ timeout: 30_000 });
  await expect(planningMessage.getByRole("button", { name: "确认开始分析", exact: true })).toHaveCount(0);
  expect(await page.evaluate(async (messageId) => new Promise<boolean>((resolveMessage) => {
    const open = indexedDB.open("socialgraph-fm");
    open.onerror = () => resolveMessage(false);
    open.onsuccess = () => {
      const request = open.result.transaction("messages", "readonly").objectStore("messages").get(messageId!);
      request.onerror = () => resolveMessage(false);
      request.onsuccess = () => resolveMessage(Boolean(request.result));
    };
  }), planningMessageId)).toBe(true);
  expect(beforeDispatch.has(confirmation.result.runId)).toBe(false);
  expect(confirmation.result).toMatchObject({
    artifactId: artifact.artifactId,
    datasetContentHash: artifact.datasetContentHash,
    graphVersionHash: artifact.graphVersionHash,
    modelVersionId: health.modelVersionId,
  });

  const reportResponse = await reportResponsePromise;
  expect(reportResponse.ok()).toBe(true);
  const report = await reportResponse.json() as {
    readonly intent: string;
    readonly answerMode?: string;
    readonly status: string;
    readonly answer: string;
    readonly skillCalls?: readonly { readonly skill: string }[];
  };
  expect(report).toMatchObject({ intent: "answer", status: "completed" });
  expect(report.answer.length).toBeLessThanOrEqual(1_500);
  await expect(planningMessage).toContainText("五阶段分析完成", { timeout: 120_000 });
  const reportMessage = page.locator(".chat-message--assistant").last();
  if (report.answerMode) {
    expect(report.answerMode).toBe("analysis_summary");
    expect(report.skillCalls?.length).toBeGreaterThan(0);
    expect(report.skillCalls?.every(({ skill }) => !["run_governance_analysis", "draft_review_report"].includes(skill))).toBe(true);
    for (const heading of ["全局态势报告", "重点候选账号", "重点风险群组", "重点事实关系", "待核验潜在线索", "人工复核建议"]) {
      expect(report.answer).toContain(heading);
      await expect(reportMessage.getByRole("heading", { name: heading, exact: true })).toBeVisible({ timeout: 120_000 });
    }
  } else {
    for (const heading of ["治理摘要", "重点候选", "协同群组", "重点关系", "人工复核建议"]) {
      expect(report.answer).toContain(heading);
      await expect(reportMessage.getByRole("heading", { name: heading, exact: true })).toBeVisible({ timeout: 120_000 });
    }
  }
  await expect(page.getByRole("button", { name: "打开复核工作台", exact: true })).toHaveCount(1);
  expect(artifactUploadCount).toBe(1);

  await page.setViewportSize({ width: 390, height: 844 });
  const overviewCamera = {
    x: Number(await chatGraph.getAttribute("data-world-center-x")),
    y: Number(await chatGraph.getAttribute("data-world-center-y")),
  };
  await page.getByRole("button", { name: "定位重点候选", exact: true }).click();
  await expect(page.getByRole("tab", { name: "图谱" })).toHaveAttribute("aria-selected", "true");
  await expect(page.locator(".toast")).toContainText(/已定位 \d+ 个重点候选/u);
  await expect(chatGraph).toBeVisible();
  const focusedCandidate = chatGraph.getByRole("complementary", { name: "已选节点详情" });
  await expect(focusedCandidate).toBeVisible();
  await expect(focusedCandidate.locator("strong")).not.toHaveText("");
  const returnToOverview = chatGraph.getByRole("button", { name: "返回完整图", exact: true });
  await expect(returnToOverview).toBeVisible();
  await returnToOverview.click();
  await expect(returnToOverview).toHaveCount(0);
  await expect(focusedCandidate).toHaveCount(0);
  expect(Number(await chatGraph.getAttribute("data-visible-nodes"))).toBe(chatVisibleNodes);
  expect(Number(await chatGraph.getAttribute("data-world-center-x"))).toBeCloseTo(overviewCamera.x, 1);
  expect(Number(await chatGraph.getAttribute("data-world-center-y"))).toBeCloseTo(overviewCamera.y, 1);
  await page.getByRole("tab", { name: "对话" }).click();
  await page.setViewportSize({ width: 1440, height: 900 });

  await page.reload();
  await expect(page.getByRole("button", { name: "打开复核工作台", exact: true })).toHaveCount(1, { timeout: 30_000 });

  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  await expect(governance.getByText("russia-04.zip", { exact: true })).toHaveCount(0);
  await expect(governance.getByRole("button", { name: "风险节点", exact: true })).toHaveAttribute("aria-pressed", "true");
  await expect(governance.locator(".governance-result-list[aria-label='风险节点'] article").first()).toBeVisible({ timeout: 120_000 });
  expect(artifactUploadCount).toBe(1);
});

test("an in-flight chat package appears as a single generic sync state in governance", async ({ page, request }) => {
  test.setTimeout(90_000);
  await requireOnlineForward(request);
  let releaseCompatibility!: () => void;
  const compatibilityGate = new Promise<void>((resolveGate) => { releaseCompatibility = resolveGate; });
  await page.route("**/api/v2/gfm/governance/artifacts/compatibility", async (route) => {
    await compatibilityGate;
    await route.continue();
  });

  await page.goto("/");
  const uploadInput = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(uploadInput).toBeEnabled({ timeout: 30_000 });
  await uploadInput.setInputFiles(resolve(
    process.cwd(),
    "../../examples/governance/russia/russia-04.zip",
  ));

  await expect(page.getByLabel("研究问题", { exact: true })).toBeDisabled();
  await expect(page.getByRole("button", { name: "发送研究问题" })).toBeDisabled();
  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  await expect(page.getByRole("heading", { name: "正在同步当前会话" })).toBeVisible();
  await expect(page.getByText("治理结果准备完成后会自动出现在这里。", { exact: true })).toBeVisible();
  await expect(page.getByText("russia-04.zip", { exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "选择 ZIP", exact: true })).toHaveCount(0);
  await expect(page.getByTestId("governance-workspace").locator('input[type="file"]')).toHaveCount(0);

  releaseCompatibility();
  await expect(page.getByRole("heading", { name: "当前会话暂无治理结果" })).toBeVisible({ timeout: 60_000 });
  await expect(page.getByRole("button", { name: "返回对话研究", exact: true })).toBeVisible();
});

test("replacing a package invalidates an unconsumed chat confirmation", async ({ page, request }) => {
  test.setTimeout(90_000);
  await requireOnlineForward(request);
  await page.goto("/");
  const chatUpload = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(chatUpload).toBeEnabled({ timeout: 30_000 });
  await chatUpload.setInputFiles(resolve(process.cwd(), "../../examples/governance/russia/russia-04.zip"));
  await expect(page.locator(".chat-message--user").filter({ hasText: "russia-04.zip" })).toHaveCount(1, { timeout: 30_000 });
  await expect(page.getByText(/推理包已通过兼容性检查|上传完成/u)).toHaveCount(0);
  await page.getByLabel("研究问题", { exact: true }).fill("开始分析");
  await page.getByRole("button", { name: "发送研究问题", exact: true }).click();
  await expect(page.getByRole("button", { name: "确认开始分析", exact: true })).toBeVisible({ timeout: 30_000 });

  let releasePreview!: () => void;
  const previewGate = new Promise<void>((resolveGate) => { releasePreview = resolveGate; });
  let previewObserved!: () => void;
  const previewStarted = new Promise<void>((resolveStarted) => { previewObserved = resolveStarted; });
  await page.route(/\/api\/v2\/gfm\/governance\/artifacts\/[^/]+\/preview(?:\?.*)?$/u, async (route) => {
    previewObserved();
    await previewGate;
    await route.continue();
  });
  await chatUpload.setInputFiles(resolve(
    process.cwd(),
    "../../examples/governance/russia/russia-03.zip",
  ));
  await previewStarted;

  await expect(page.getByRole("button", { name: "确认开始分析", exact: true })).toHaveCount(0);
  await expect(page.getByText("输入已更换，此操作计划已失效。", { exact: true })).toBeVisible();
  releasePreview();
});

test("switching sessions while package preview is pending cannot persist the stale graph", async ({ page, request }) => {
  test.setTimeout(60_000);
  await requireOnlineForward(request);
  await page.goto("/");

  const uploadInput = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(uploadInput).toBeEnabled({ timeout: 30_000 });
  const initialSessionId = await page.evaluate(() => localStorage.getItem("socialgraph-fm-active-session"));
  const initialGraphVersionCount = await graphVersionCount(page);

  let previewObserved!: () => void;
  const previewStarted = new Promise<void>((resolveStarted) => { previewObserved = resolveStarted; });
  let releasePreview!: () => void;
  const previewGate = new Promise<void>((resolveGate) => { releasePreview = resolveGate; });
  await page.route(/\/api\/v2\/gfm\/governance\/artifacts\/[^/]+\/preview(?:\?.*)?$/u, async (route) => {
    previewObserved();
    await previewGate;
    await route.continue();
  });

  await uploadInput.setInputFiles(resolve(process.cwd(), "../../examples/governance/russia/russia-04.zip"));
  await previewStarted;
  await page.getByRole("button", { name: "新建研究会话", exact: true }).click();
  await expect.poll(() => page.evaluate(() => localStorage.getItem("socialgraph-fm-active-session")))
    .not.toBe(initialSessionId);
  const previewResponse = page.waitForResponse((response) => /\/api\/v2\/gfm\/governance\/artifacts\/[^/]+\/preview(?:\?.*)?$/u.test(response.url()));
  releasePreview();
  await previewResponse;
  await page.waitForLoadState("networkidle");

  expect(await graphVersionCount(page)).toBe(initialGraphVersionCount);
  await expect(page.locator(".chat-message--user").filter({ hasText: "russia-04.zip" })).toHaveCount(0);
});

test("re-uploading the same inference package creates another immutable graph revision", async ({ page, request }) => {
  test.setTimeout(60_000);
  await requireOnlineForward(request);
  await page.goto("/");
  const uploadInput = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(uploadInput).toBeEnabled({ timeout: 30_000 });
  const packagePath = resolve(process.cwd(), "../../examples/governance/russia/russia-04.zip");

  await uploadInput.setInputFiles(packagePath);
  await expect(page.locator(".chat-message--user").filter({ hasText: "russia-04.zip" })).toHaveCount(1, { timeout: 30_000 });
  await uploadInput.setInputFiles(packagePath);
  await expect(page.locator(".chat-message--user").filter({ hasText: "russia-04.zip" })).toHaveCount(2, { timeout: 30_000 });
  await expect(page.getByText(/推理包已通过兼容性检查|上传完成/u)).toHaveCount(0);

  await expect(page.getByText(/IMMUTABLE_GRAPH_VERSION_CONFLICT/)).toHaveCount(0);
  const graph = page.locator('.graph-preview[aria-label="交互式社交关系图预览"]');
  await expect(graph).toHaveAttribute("data-graph-ready", "true");
  await page.reload();
  await expect(graph).toHaveAttribute("data-graph-ready", "true", { timeout: 30_000 });
});

test("governance remains a read-only snapshot consumer while a chat run is in flight", async ({ page, request }) => {
  test.setTimeout(120_000);
  await requireOnlineForward(request);

  await page.goto("/");
  const chatUpload = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(chatUpload).toBeEnabled({ timeout: 30_000 });
  await chatUpload.setInputFiles(resolve(process.cwd(), "../../examples/governance/russia/russia-04.zip"));
  await expect(page.getByText("russia-04.zip", { exact: true })).toBeVisible({ timeout: 30_000 });

  await page.getByLabel("研究问题", { exact: true }).fill("开始分析");
  await page.getByRole("button", { name: "发送研究问题" }).click();
  const confirm = page.getByRole("button", { name: "确认开始分析", exact: true });
  await expect(confirm).toBeVisible({ timeout: 10_000 });

  let releaseRunPoll!: () => void;
  const runPollGate = new Promise<void>((resolveGate) => { releaseRunPoll = resolveGate; });
  await page.route("**/api/v2/gfm/governance/runs/*", async (route) => {
    const url = new URL(route.request().url());
    if (route.request().method() === "GET" && /^\/api\/v2\/gfm\/governance\/runs\/governance-[^/]+$/u.test(url.pathname)) {
      await runPollGate;
    }
    await route.continue();
  });
  const confirmedResponse = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/v2/gfm/governance/skills/confirm");
  await confirm.click();
  expect((await confirmedResponse).ok()).toBe(true);

  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  const governance = page.getByTestId("governance-workspace");
  await expect(governance).toBeVisible();
  await expect(governance.locator('input[type="file"]')).toHaveCount(0);
  await expect(governance.getByText(/russia-04\.zip|russia-03\.zip/iu)).toHaveCount(0);
  await expect(governance.getByText("分析引擎在线", { exact: true })).toHaveCount(0);
  await expect(governance.getByRole("button", { name: /开始分析|重新分析/u })).toHaveCount(0);

  releaseRunPoll();
  await expect(governance.getByRole("button", { name: "风险节点", exact: true })).toHaveAttribute("aria-pressed", "true", { timeout: 120_000 });
  await expect(governance.locator(".governance-result-list[aria-label='风险节点'] article").first()).toBeVisible({ timeout: 120_000 });
});
});
