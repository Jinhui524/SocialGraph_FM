import { expect, test } from "@playwright/test";
import { Buffer } from "node:buffer";
import { writeFileSync } from "node:fs";
import { resolve } from "node:path";

const evidenceRoot = resolve(process.cwd(), "../../var/qa/task-5");

async function expectComposerAndNoViewportOverflow(page: import("@playwright/test").Page) {
  await expect(page.locator(".composer-wrap")).toBeVisible();
  const overflow = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(overflow.scroll).toBeLessThanOrEqual(overflow.client + 1);
}

test("atlas welcome reflows across acceptance viewports and the 200% equivalent", async ({ page }) => {
  await page.goto("/#/research");
  const welcome = page.getByRole("region", { name: "学术网络图谱开始页" });
  await expect(welcome).toBeVisible();

  for (const viewport of [
    { width: 1920, height: 1080 },
    { width: 1440, height: 900 },
    { width: 1280, height: 720 },
    { width: 390, height: 844 },
  ]) {
    await page.setViewportSize(viewport);
    await expectComposerAndNoViewportOverflow(page);
  }

  await page.setViewportSize({ width: 960, height: 540 });
  await expectComposerAndNoViewportOverflow(page);
  await expect(page.getByRole("tab", { name: "对话" })).toBeVisible();
  await page.screenshot({ path: resolve(evidenceRoot, "06-welcome-200-percent-equivalent-960x540.png") });

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
  await expectComposerAndNoViewportOverflow(page);
  await expect(welcome.getByRole("button", { name: "上传关系数据" })).toBeVisible();
  const zoomCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  writeFileSync(resolve(evidenceRoot, "09-welcome-actual-browser-zoom-200-percent-1920x1080.png"), Buffer.from(zoomCapture.data, "base64"));
  await cdp.send("Emulation.clearDeviceMetricsOverride");
});

test("reduced motion preserves the static atlas hierarchy", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1280, height: 720 });
  await page.goto("/#/research");
  const welcome = page.getByRole("region", { name: "学术网络图谱开始页" });
  await expect(welcome).toBeVisible();
  await expectComposerAndNoViewportOverflow(page);
  const transition = await welcome.locator(".prompt-card").first().evaluate((element) => getComputedStyle(element).transitionDuration);
  expect(transition.split(",").every((value) => Number.parseFloat(value) <= 0.001)).toBe(true);
  await page.screenshot({ path: resolve(evidenceRoot, "07-welcome-reduced-motion-1280x720.png") });
});

test("a welcome task prepares the question and upload guidance before a graph exists", async ({ page }) => {
  await page.goto("/#/research");
  const welcome = page.getByRole("region", { name: "学术网络图谱开始页" });
  const task = welcome.getByRole("button", { name: /图谱基本情况/u });

  await task.click();

  await expect(page.getByLabel("研究问题", { exact: true })).toHaveValue("请概括当前图谱的账号规模、事实关系数量、关系类型和连通情况");
  await expect(page.locator(".chat-message")).toHaveCount(0);
  await expect(page.locator(".toast")).toHaveText("研究目标已填入；请先上传关系数据，再发送分析");
  await expect(page.getByText("请先上传关系数据，再发送这个研究目标", { exact: true })).toBeVisible();
  await expect(page.getByText("系统就绪", { exact: true })).toHaveCount(0);
});
