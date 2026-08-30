import { expect, test } from "@playwright/test";

test("production governance entry is a snapshot consumer with a formal empty state", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "治理应用", exact: true }).click();

  const workspace = page.getByTestId("governance-workspace");
  await expect(workspace).toBeVisible();
  await expect(workspace.getByRole("navigation", { name: "治理工作模式" })).toBeVisible();
  await expect(workspace.getByRole("heading", { name: "当前会话暂无治理结果" })).toBeVisible();
  await expect(workspace.getByRole("button", { name: "返回对话研究", exact: true })).toBeVisible();

  for (const removedLabel of [
    "风险研判",
    "治理数据",
    "等待数据",
    "推理包",
    "分析引擎在线",
    "Russia 动态样例",
    "协议验证",
    "通用图任务",
    "SocialGraph-FM Research",
    "SocialGraph-FM Governance 治理研判",
    "Russia 场景已绑定",
  ]) {
    await expect(page.getByText(removedLabel, { exact: true })).toHaveCount(0);
  }
  await expect(page.getByText(/In-domain|Low-label|Cross-domain|RQ4/)).toHaveCount(0);
  await expect(page.getByRole("button", { name: "运行 SocialGraph-FM Governance 研判" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: /开始分析|重新分析/u })).toHaveCount(0);
  await expect(workspace.locator('input[type="file"]')).toHaveCount(0);
});
