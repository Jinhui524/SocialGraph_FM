import { expect, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

async function readGraphStability(page: Page) {
  return page.locator('.graph-preview[aria-label="治理关系图"]').evaluate((element) => {
    const root = element as HTMLElement;
    return {
      layoutCount: Number(root.dataset.layoutCount),
      fitViewCount: Number(root.dataset.fitViewCount),
      engineCreateCount: Number(root.dataset.engineCreateCount),
      engineDestroyCount: Number(root.dataset.engineDestroyCount),
      mutationInFlight: Number(root.dataset.mutationInFlight),
      cameraX: Number(root.dataset.cameraX),
      cameraY: Number(root.dataset.cameraY),
      zoom: Number(root.dataset.cameraZoom),
      worldCenterX: Number(root.dataset.worldCenterX),
      worldCenterY: Number(root.dataset.worldCenterY),
      nodeId: root.dataset.coordinateNodeId ?? null,
      nodeX: Number(root.dataset.coordinateNodeX),
      nodeY: Number(root.dataset.coordinateNodeY),
    };
  });
}

async function readReadyGraphStability(page: Page) {
  const graph = page.locator('.graph-preview[aria-label="治理关系图"]');
  await expect(graph).toHaveAttribute("data-world-center-x", /^-?\d/u);
  await expect(graph).toHaveAttribute("data-world-center-y", /^-?\d/u);
  await expect(graph).toHaveAttribute("data-coordinate-node-id", /.+/u);
  await expect.poll(async () => {
    const snapshot = await readGraphStability(page);
    return snapshot.mutationInFlight === 0
      && Boolean(snapshot.nodeId)
      && [
        snapshot.cameraX,
        snapshot.cameraY,
        snapshot.zoom,
        snapshot.worldCenterX,
        snapshot.worldCenterY,
        snapshot.nodeX,
        snapshot.nodeY,
      ].every(Number.isFinite);
  }).toBe(true);
  return readGraphStability(page);
}

async function expectGraphStability(
  page: Page,
  before: Awaited<ReturnType<typeof readGraphStability>>,
) {
  await page.evaluate(() => new Promise<void>((resolveFrame) => {
    requestAnimationFrame(() => requestAnimationFrame(() => resolveFrame()));
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
  expect([before.cameraX, before.cameraY, after.cameraX, after.cameraY].every(Number.isFinite)).toBe(true);
  expect(Math.abs(after.zoom - before.zoom)).toBeLessThan(0.001);
  return after;
}

async function expectGraphLayoutStability(
  page: Page,
  before: Awaited<ReturnType<typeof readGraphStability>>,
) {
  await expect.poll(async () => (await readGraphStability(page)).mutationInFlight).toBe(0);
  await expect.poll(async () => {
    const after = await readGraphStability(page);
    return {
      layoutCount: after.layoutCount,
      engineCreateCount: after.engineCreateCount,
      engineDestroyCount: after.engineDestroyCount,
    };
  }).toEqual({
    layoutCount: before.layoutCount,
    engineCreateCount: before.engineCreateCount,
    engineDestroyCount: before.engineDestroyCount,
  });
  await expect.poll(async () => {
    const after = await readGraphStability(page);
    return Math.hypot(after.nodeX - before.nodeX, after.nodeY - before.nodeY);
  }).toBeLessThan(0.1);
}

test.describe("@backend deployed SocialGraph-FM Governance governance loop", () => {
test("real SocialGraph-FM Governance path runs Global forward and closes the governance loop", async ({ page, request }) => {
  test.setTimeout(300_000);
  const healthResponse = await request.get("/api/v2/gfm/governance/health");
  expect(
    healthResponse.ok(),
    "The release E2E gate requires the managed SocialGraph-FM Governance service to be reachable.",
  ).toBe(true);
  const health = await healthResponse.json() as { onlineForwardReady?: boolean };
  expect(
    health.onlineForwardReady,
    "The release E2E gate requires the deployed Global checkpoint to be online-forward ready.",
  ).toBe(true);

  await page.goto("/");
  const conversationUpload = page.locator('.composer-wrap input[type="file"][accept*=".zip"]');
  await expect(conversationUpload).toBeEnabled({ timeout: 30_000 });
  await conversationUpload.setInputFiles(resolve(
    process.cwd(),
    "../../examples/governance/russia/russia-04.zip",
  ));
  await expect(page.locator(".chat-message--user").filter({ hasText: "russia-04.zip" })).toBeVisible({ timeout: 30_000 });
  await page.getByLabel("研究问题", { exact: true }).fill("开始分析");
  await page.getByRole("button", { name: "发送研究问题", exact: true }).click();
  const confirm = page.getByRole("button", { name: "确认开始分析", exact: true });
  await expect(confirm).toBeVisible({ timeout: 30_000 });
  await confirm.click();
  await expect(page.getByRole("button", { name: "打开复核工作台", exact: true })).toBeVisible({ timeout: 180_000 });
  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  const governance = page.getByTestId("governance-workspace");
  await expect(governance).toBeVisible();
  await expect(governance.locator('input[type="file"]')).toHaveCount(0);
  await expect(governance.getByText("分析引擎在线", { exact: true })).toHaveCount(0);
  const visibleListItem = governance.locator(".governance-result-list[aria-label='风险节点'] article").first();
  await expect(visibleListItem).toBeVisible({ timeout: 120_000 });
  const graphCanvas = page.locator('.graph-preview[aria-label="治理关系图"]').locator("canvas").first();
  await expect(graphCanvas).toBeVisible();
  await expect.poll(() => graphCanvas.evaluate((element) => (element as HTMLCanvasElement).toDataURL().length)).toBeGreaterThan(1_000);

  const governanceModes = governance.getByRole("navigation", { name: "治理工作模式" });
  await governanceModes.getByRole("button", { name: "群组与关系", exact: true }).click();
  const relationTabs = governance.getByRole("tablist", { name: "群组与关系类型" });
  await expect(relationTabs.getByRole("tab", { name: /风险群组/u })).toHaveAttribute("aria-selected", "true");
  await relationTabs.getByRole("tab", { name: /事实关系/u }).click();
  await expect(governance.locator(".governance-result-list__select").first()).toBeVisible();
  await relationTabs.getByRole("tab", { name: /潜在线索/u }).click();
  await expect(governance.locator(".governance-result-list__select").first()).toBeVisible();
  await governanceModes.getByRole("button", { name: "风险节点", exact: true }).click();
  const firstCandidate = governance.locator(".governance-result-list[aria-label='风险节点'] .governance-result-list__select").first();
  const graphRoot = page.locator('.graph-preview[aria-label="治理关系图"]');
  await page.waitForLoadState("networkidle");
  await expect(graphRoot).toHaveAttribute("data-graph-ready", "true");
  const stabilityBeforeSelection = await readReadyGraphStability(page);
  const layoutCountBeforeSelection = stabilityBeforeSelection.layoutCount;
  await firstCandidate.click();
  const selectionBanner = governance.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" });
  await expect(selectionBanner).toBeVisible();
  await expect(page.getByRole("dialog")).toHaveCount(0);
  expect(Number(await graphRoot.getAttribute("data-layout-count"))).toBe(layoutCountBeforeSelection);
  await expectGraphStability(page, stabilityBeforeSelection);

  const evidenceTrigger = visibleListItem.locator(".governance-result-list__evidence");
  await evidenceTrigger.click();
  const evidenceDialog = page.getByRole("dialog");
  await expect(evidenceDialog).toBeVisible();
  await evidenceDialog.getByRole("tab", { name: "关系事实", exact: true }).click();
  await expect(evidenceDialog.getByRole("tabpanel", { name: "关系事实" })).toContainText("关联账号", { timeout: 30_000 });
  await expect(evidenceDialog.getByRole("tabpanel", { name: "关系事实" }).getByText("依据来源")).toHaveCount(0);
  await expect(evidenceDialog).toContainText("发布时间、原帖内容及采集来源需要在人工复核中补充");
  expect(await evidenceDialog.locator("th").first().evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize))).toBeGreaterThanOrEqual(12);
  await page.setViewportSize({ width: 1440, height: 900 });
  const dialogBounds = await evidenceDialog.boundingBox();
  expect(dialogBounds).not.toBeNull();
  expect(Math.abs(dialogBounds!.x + dialogBounds!.width / 2 - 720)).toBeLessThan(2);
  await page.keyboard.press("Escape");
  await expect(evidenceDialog).toHaveCount(0);
  await expect(selectionBanner).toBeVisible();

  const graphBounds = await graphCanvas.boundingBox();
  expect(graphBounds).not.toBeNull();
  await page.mouse.click(graphBounds!.x + 8, graphBounds!.y + 8);
  await expect(selectionBanner).toHaveCount(0);
  expect(Number(await graphRoot.getAttribute("data-layout-count"))).toBe(layoutCountBeforeSelection);
  await expectGraphLayoutStability(page, stabilityBeforeSelection);
  await firstCandidate.click();
  await evidenceTrigger.click();
  await evidenceDialog.getByRole("tab", { name: "人工复核", exact: true }).click();
  for (const decision of ["确认", "驳回", "待定"]) await expect(evidenceDialog.getByRole("button", { name: decision, exact: true })).toBeVisible();
  const reviewConfirm = evidenceDialog.getByRole("button", { name: "确认", exact: true });
  await expect(reviewConfirm).toBeDisabled();
  expect(await reviewConfirm.evaluate((element) => getComputedStyle(element).cursor)).toBe("default");
  await evidenceDialog.getByRole("button", { name: "加入并开始复核" }).click();
  await expect(evidenceDialog.getByText("已加入研判单，可提交人工结论")).toBeVisible();
  await expect(evidenceDialog).toContainText("研判中");
  expect(await evidenceDialog.locator("label").first().evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize))).toBeGreaterThanOrEqual(13);
  await evidenceDialog.getByRole("textbox", { name: "复核理由" }).fill("答辩演示：已核对一跳关系和关系模态。");
  await reviewConfirm.click();
  await expect(evidenceDialog.getByText(/当前人工结论：确认/)).toBeVisible();
  expect(await evidenceDialog.locator(".governance-timeline strong").first().evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize))).toBeGreaterThanOrEqual(12);
  expect(await evidenceDialog.locator(".governance-timeline time").first().evaluate((element) => Number.parseFloat(getComputedStyle(element).fontSize))).toBeGreaterThanOrEqual(12);
  await page.setViewportSize({ width: 1440, height: 900 });
  await expect(evidenceDialog).toBeVisible();
  await page.setViewportSize({ width: 1200, height: 900 });
  await expect(evidenceDialog).toBeVisible();
  await page.setViewportSize({ width: 1440, height: 900 });
  await evidenceDialog.getByRole("button", { name: "关闭证据档案" }).click();
  await expect(graphRoot).toHaveAttribute("data-review-decision-count", "1");
  const reviewStateTabs = governance.getByRole("tablist", { name: "风险节点研判状态" });
  await expect(reviewStateTabs.getByRole("tab", { name: /待复核/u })).toHaveAttribute("aria-selected", "true");
  await expect(governance.locator(".governance-result-list[aria-label='风险节点']")).not.toContainText("匿名账号 458");
  await reviewStateTabs.getByRole("tab", { name: /已研判/u }).click();
  await expect(governance.locator(".governance-result-list[aria-label='风险节点']")).toContainText("匿名账号 458");
  await expect(governance.getByText("人工确认", { exact: true })).toBeVisible();
  await governanceModes.getByRole("button", { name: "研判单", exact: true }).click();
  const caseDetail = governance.locator(".governance-case-detail");
  await caseDetail.getByRole("button", { name: "形成结论" }).click();
  await expect(caseDetail).toContainText("已形成结论");

  const downloadPromise = page.waitForEvent("download");
  await caseDetail.getByRole("button", { name: "HTML", exact: true }).click();
  const report = await downloadPromise;
  expect(report.suggestedFilename()).toMatch(/^case-[0-9a-f]{32}\.html$/);

  await governanceModes.getByRole("button", { name: "风险节点", exact: true }).click();
  await firstCandidate.click();
  await governanceModes.getByRole("button", { name: "研判助手", exact: true }).click();
  const assistant = page.getByRole("complementary", { name: "案例研判助手" });
  await assistant.getByRole("tab", { name: "历史案例", exact: true }).click();
  await assistant.getByRole("button", { name: "检索相似历史案例", exact: true }).click();
  await expect(assistant.getByText("历史案例 01", { exact: true })).toBeVisible({ timeout: 30_000 });
  await expect(assistant).not.toContainText("请求未完成");
});
});
