import { expect, test } from "@playwright/test";

const BANNED_RELEASE_COPY = /RQ\d?|SocialGraph-FM Governance|modelStateHash|输入合同|技术详情|TrainingDatasetRef|后续版本开放/u;

async function expectReleasedCopy(page: import("@playwright/test").Page) {
  await expect(page.locator("body")).not.toContainText(BANNED_RELEASE_COPY);
}

test("released workspaces keep stable URLs across reload and browser history", async ({ page }) => {
  await page.goto("/#/research");
  await expect(page.getByRole("heading", { level: 1, name: "对话研究" })).toBeVisible();
  await expectReleasedCopy(page);

  await expect(page.getByRole("button", { name: "图谱库" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "分析工具" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "实验记录" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "模板市场" })).toHaveCount(0);

  await page.getByRole("button", { name: "适配能力", exact: true }).click();
  await expect(page).toHaveURL(/#\/adaptation$/);
  await expect(page.getByRole("heading", { level: 1, name: "适配能力" })).toBeVisible();
  await expect(page.getByRole("region", { name: "面向新网络的风险迁移" })).toBeVisible();
  await expect(page.getByRole("region", { name: "零样本路径" })).toBeVisible();
  await expect(page.getByRole("region", { name: "少样本路径" })).toBeVisible();
  await expectReleasedCopy(page);

  await page.reload();
  await expect(page.getByRole("button", { name: "适配能力", exact: true })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { level: 1, name: "适配能力" })).toBeVisible();

  await page.getByRole("button", { name: "数据管理", exact: true }).click();
  await expect(page).toHaveURL(/#\/datasets$/);
  await expect(page.getByRole("dialog", { name: "数据管理" })).toBeVisible();
  await expect(page.getByRole("button", { name: "数据管理", exact: true })).toHaveAttribute("aria-current", "page");
  await expectReleasedCopy(page);

  await page.getByRole("button", { name: "关闭数据管理", exact: true }).click();
  await expect(page).toHaveURL(/#\/adaptation$/);
  await expect(page.getByRole("dialog", { name: "数据管理" })).toHaveCount(0);
  await page.goBack();
  await expect(page.getByRole("dialog", { name: "数据管理" })).toHaveCount(0);
  await expect(page).not.toHaveURL(/#\/datasets$/);

  await page.getByRole("button", { name: "数据管理", exact: true }).click();
  await expect(page.getByRole("dialog", { name: "数据管理" })).toBeVisible();

  await page.goBack();
  await expect(page).toHaveURL(/#\/adaptation$/);
  await expect(page.getByRole("dialog", { name: "数据管理" })).toHaveCount(0);
  await expect(page.getByRole("heading", { level: 1, name: "适配能力" })).toBeVisible();

  await page.getByRole("button", { name: "治理应用", exact: true }).click();
  await expect(page).toHaveURL(/#\/governance$/);
  await expect(page.getByRole("heading", { level: 1, name: "治理应用" })).toBeVisible();
  await expect(page.getByRole("button", { name: /^LLM$/ })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /^GFM$/ })).toHaveCount(0);
  await expectReleasedCopy(page);
});
