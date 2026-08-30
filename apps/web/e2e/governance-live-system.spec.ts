import { expect, test, type Locator, type Page } from "@playwright/test";
import { Buffer } from "node:buffer";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { basename, dirname, isAbsolute, relative, resolve } from "node:path";

const FORBIDDEN_RELEASE_COPY = /Cuba|UAE|Thailand|RQ\d?|sample|demo|experiment|schemaVersion|AdaptationReviewPolicy|probability|accuracy|labelSetHash|labels\.json/iu;

type CatalogEntry = {
  readonly role: "zero_shot" | "few_shot";
  readonly path: string;
  readonly sha256: string;
  readonly bytes: number;
};

type GlobalModelIdentity = {
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly modelStateHash: string;
};

type RunResultDocument = GlobalModelIdentity & {
  readonly runId: string;
  readonly requestHash: string;
  readonly resultHash: string;
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly findings: readonly { readonly nodeId: string; readonly score: number; readonly rank: number }[];
};

type FullGraphPreviewDocument = {
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly runId: string;
  readonly resultHash: string;
  readonly nodes: readonly Record<string, unknown>[];
  readonly edges: readonly Record<string, unknown>[];
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly partialPreview: false;
  readonly previewHash: string;
};

type HealthIdentity = GlobalModelIdentity & {
  readonly runtimeRecipeHash: string;
};

type KnowledgeSearchDocument = {
  readonly schemaVersion: "socialgraph-fm.governance-skills/1.0";
  readonly items: readonly {
    readonly sourceLabel: string;
    readonly sourceUri: string;
    readonly contentHash: string;
    readonly chunkHash: string;
    readonly text: string;
    readonly rank: number;
  }[];
  readonly indexHash: string;
  readonly auditHash: string;
};

type AdaptationBindingDocument = GlobalModelIdentity & {
  readonly artifactId: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly runId: string;
  readonly requestHash: string;
  readonly resultHash: string;
  readonly runArtifactHash: string;
  readonly recipeHash: string;
  readonly codeHash: string;
  readonly seed: number;
};

type PublishedCheckpointIdentity = GlobalModelIdentity & {
  readonly registryHash: string;
  readonly checkpointSha256: string;
  readonly checkpointBytes: number;
};

function expectHash(value: unknown): asserts value is string {
  expect(value).toMatch(/^[0-9a-f]{64}$/u);
}

function parseGlobalModelIdentity(value: unknown): GlobalModelIdentity {
  expect(value).toEqual(expect.objectContaining({
    modelVersionId: expect.any(String),
    modelVersionHash: expect.any(String),
    modelStateHash: expect.any(String),
  }));
  const identity = value as GlobalModelIdentity;
  expect(identity.modelVersionId.length).toBeGreaterThan(0);
  expectHash(identity.modelVersionHash);
  expectHash(identity.modelStateHash);
  return {
    modelVersionId: identity.modelVersionId,
    modelVersionHash: identity.modelVersionHash,
    modelStateHash: identity.modelStateHash,
  };
}

function publishedCheckpointIdentity(): PublishedCheckpointIdentity {
  const globalModelRoot = resolve(process.cwd(), "../../var/models/socialgraph-global");
  const registryPath = resolve(globalModelRoot, "registry/socialgraph-global.json");
  const registry = JSON.parse(readFileSync(registryPath, "utf8")) as {
    readonly registryHash: string;
    readonly modelVersionId: string;
    readonly modelVersionHash: string;
    readonly modelStateHash: string;
    readonly checkpointPath: string;
    readonly checkpointSha256: string;
    readonly protocolArtifacts: { readonly global: {
      readonly protocolModelVersionId: string;
      readonly protocolModelVersionHash: string;
      readonly modelStateHash: string;
      readonly checkpointPath: string;
      readonly checkpointSha256: string;
    } };
  };
  const global = registry.protocolArtifacts.global;
  expect(global).toMatchObject({
    protocolModelVersionId: registry.modelVersionId,
    protocolModelVersionHash: registry.modelVersionHash,
    modelStateHash: registry.modelStateHash,
    checkpointPath: registry.checkpointPath,
    checkpointSha256: registry.checkpointSha256,
  });
  expectHash(registry.registryHash);
  expectHash(registry.checkpointSha256);
  const checkpointPath = resolve(globalModelRoot, registry.checkpointPath);
  const relativePath = relative(globalModelRoot, checkpointPath);
  expect(relativePath.startsWith("..") || isAbsolute(relativePath)).toBe(false);
  const bytes = readFileSync(checkpointPath);
  expect(createHash("sha256").update(bytes).digest("hex")).toBe(registry.checkpointSha256);
  return {
    ...parseGlobalModelIdentity(registry),
    registryHash: registry.registryHash,
    checkpointSha256: registry.checkpointSha256,
    checkpointBytes: bytes.length,
  };
}

async function readRunResult(page: Page, runId: string): Promise<RunResultDocument> {
  const response = await page.request.get(`/api/v2/gfm/governance/runs/${runId}/result`);
  expect(response.ok()).toBe(true);
  const value = await response.json() as RunResultDocument;
  parseGlobalModelIdentity(value);
  expect(value).toEqual(expect.objectContaining({
    runId,
    requestHash: expect.stringMatching(/^[0-9a-f]{64}$/u),
    resultHash: expect.stringMatching(/^[0-9a-f]{64}$/u),
    artifactId: expect.stringMatching(/^governance-artifact-[0-9a-f]{32}$/u),
    datasetContentHash: expect.stringMatching(/^[0-9a-f]{64}$/u),
    graphVersionHash: expect.stringMatching(/^[0-9a-f]{64}$/u),
    findings: expect.any(Array),
  }));
  return value;
}

async function readFullGraphPreview(page: Page, result: RunResultDocument): Promise<FullGraphPreviewDocument> {
  const response = await page.request.get(`/api/v2/gfm/governance/runs/${result.runId}/graph-preview`);
  expect(response.ok()).toBe(true);
  const value = await response.json() as FullGraphPreviewDocument;
  expect(value).toMatchObject({
    artifactId: result.artifactId,
    datasetContentHash: result.datasetContentHash,
    graphVersionHash: result.graphVersionHash,
    runId: result.runId,
    resultHash: result.resultHash,
    nodeCount: 108,
    partialPreview: false,
  });
  expect(value.nodes).toHaveLength(value.nodeCount);
  expect(value.edges).toHaveLength(value.edgeCount);
  expectHash(value.previewHash);
  return value;
}

async function readHealthIdentity(page: Page): Promise<HealthIdentity> {
  const response = await page.request.get("/api/v2/gfm/governance/health");
  expect(response.ok()).toBe(true);
  const value = await response.json() as HealthIdentity;
  parseGlobalModelIdentity(value);
  expectHash(value.runtimeRecipeHash);
  return {
    ...parseGlobalModelIdentity(value),
    runtimeRecipeHash: value.runtimeRecipeHash,
  };
}

async function expectBuiltInKnowledge(page: Page, identity: RunResultDocument) {
  const response = await page.request.post("/api/v2/gfm/governance/knowledge/search", {
    data: {
      schemaVersion: "socialgraph-fm.governance-skills/1.0",
      graph: {
        artifactId: identity.artifactId,
        datasetContentHash: identity.datasetContentHash,
        graphVersionHash: identity.graphVersionHash,
      },
      model: {
        modelVersionId: identity.modelVersionId,
        modelStateHash: identity.modelStateHash,
      },
      query: "model-bound knowledge review governance",
      limit: 5,
    },
  });
  expect(response.ok()).toBe(true);
  const document = await response.json() as KnowledgeSearchDocument;
  expect(document.schemaVersion).toBe("socialgraph-fm.governance-skills/1.0");
  expectHash(document.indexHash);
  expectHash(document.auditHash);
  expect(document.items.length).toBeGreaterThan(0);
  expect(document.items.some(({ sourceUri }) => sourceUri.startsWith("repo://"))).toBe(true);
  for (const [index, item] of document.items.entries()) {
    expect(item.sourceLabel.length).toBeGreaterThan(0);
    expect(item.text.length).toBeGreaterThan(0);
    expect(item.rank).toBe(index + 1);
    expectHash(item.contentHash);
    expectHash(item.chunkHash);
    expect(item.sourceUri).toMatch(/^(?:model|repo):\/\//u);
    expect(item.sourceUri).not.toMatch(/^(?:file:|[\\/]|[a-z]:[\\/])/iu);
    expect([item.sourceLabel, item.sourceUri, item.text].join("\n"))
      .not.toMatch(/(?:(?<![a-z0-9])[a-z]:[\\/]|\/Users\/|\/home\/[^/]+\/)/iu);
  }
}

const GOVERNANCE_CATALOG_IDENTITY = {
  generationId: "132a0cda25c2116f654c80b3fc783567323eada273893452d6b21e032e2f29d9",
  catalogHash: "2623a7a8fce2f96a72b2c95235f10ec99fbc21662d3f2b00cf504c5c0b3924df",
  targets: {
    zero_shot: {
      fileName: "target-domain-a-zero.sgtask.zip",
      sha256: "95926d65cdfb1a26d959a5712c2b445801cec7bae58b91d75f8aa961df8569e8",
      bytes: 316_868,
    },
    few_shot: {
      fileName: "target-domain-b-few.sgtask.zip",
      sha256: "6a7af1d68524c4f1e96aabef4fb871ab21e1679d5fabc99709f1ccc34c6dece1",
      bytes: 342_633,
    },
  },
} as const;

function liveTargetPaths() {
  const catalogPath = process.env.SOCIALGRAPH_GOVERNANCE_TARGET_CATALOG;
  if (!catalogPath) throw new Error("SOCIALGRAPH_GOVERNANCE_TARGET_CATALOG is required");
  const root = dirname(catalogPath);
  const catalog = JSON.parse(readFileSync(catalogPath, "utf8")) as {
    readonly schemaVersion: string;
    readonly generationId: string;
    readonly catalogHash: string;
    readonly targets: readonly CatalogEntry[];
  };
  expect(catalog.schemaVersion).toBe("socialgraph-fm.governance-target-catalog/1.0");
  expect(catalog.generationId).toBe(GOVERNANCE_CATALOG_IDENTITY.generationId);
  expect(catalog.catalogHash).toBe(GOVERNANCE_CATALOG_IDENTITY.catalogHash);
  expect(catalog.targets.map(({ role }) => role)).toEqual(["zero_shot", "few_shot"]);
  const target = (role: CatalogEntry["role"]) => {
    const entry = catalog.targets.find((item) => item.role === role);
    if (!entry) throw new Error(`catalog role ${role} is unavailable`);
    const expected = GOVERNANCE_CATALOG_IDENTITY.targets[role];
    expect(entry).toMatchObject({ sha256: expected.sha256, bytes: expected.bytes });
    const path = resolve(root, entry.path);
    const relativePath = relative(root, path);
    expect(relativePath.startsWith("..") || isAbsolute(relativePath)).toBe(false);
    expect(basename(path)).toBe(expected.fileName);
    const bytes = readFileSync(path);
    expect(statSync(path).size).toBe(entry.bytes);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(entry.sha256);
    return path;
  };
  return { zero: target("zero_shot"), few: target("few_shot") };
}

async function runGlobal(lane: Locator) {
  await lane.getByRole("button", { name: "开始分析" }).click();
  await lane.getByRole("button", { name: "确认分析" }).click();
  await expect(lane.getByText(/协同组群已就绪/u)).toBeVisible({ timeout: 180_000 });
}

async function expectOnscreenGraph(graph: Locator, nodes = 108) {
  await expect(graph).toHaveAttribute("data-graph-ready", "true", { timeout: 30_000 });
  await expect(graph).toHaveAttribute("data-visible-nodes", String(nodes));
  const box = await graph.boundingBox();
  expect(box).not.toBeNull();
  const viewport = await graph.evaluate(() => ({ width: innerWidth, height: innerHeight }));
  expect(box!.x + box!.width).toBeGreaterThan(0);
  expect(box!.y + box!.height).toBeGreaterThan(0);
  expect(box!.x).toBeLessThan(viewport.width);
  expect(box!.y).toBeLessThan(viewport.height);
}

async function expectFocusRing(target: Locator) {
  await target.focus();
  await target.page().keyboard.press("Tab");
  await target.page().keyboard.press("Shift+Tab");
  await expect(target).toBeFocused();
  expect(await target.evaluate((element) => {
    const style = getComputedStyle(element);
    return style.outlineStyle !== "none" && Number.parseFloat(style.outlineWidth) > 0
      || style.boxShadow !== "none";
  })).toBe(true);
}

async function contrastRatio(target: Locator): Promise<number> {
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

async function laneIdentity(lane: Locator) {
  return lane.evaluate((element) => Object.fromEntries([
    "registration-id", "task-id", "artifact-id", "inference-hash", "graph-version-hash",
    "target-receipt-hash", "registration-hash", "outer-bundle-hash", "node-set-hash",
    "run-id", "result-hash",
  ].map((name) => [name, element.getAttribute(`data-${name}`)])));
}

async function assertNoApiErrors(page: Page, failures: string[]) {
  await page.waitForLoadState("networkidle");
  expect(failures).toEqual([]);
}

test.describe("@backend managed governance release gate", () => {
test("real governance catalog completes both target lanes and target governance without route mocks", async ({ page }) => {
  const liveGateEnabled = process.env.SOCIALGRAPH_GOVERNANCE_LIVE_E2E === "1";
  if (process.env.SOCIALGRAPH_GOVERNANCE_RELEASE_GATE === "1") {
    expect(liveGateEnabled, "the dedicated release gate must execute, never skip").toBe(true);
  }
  test.skip(!liveGateEnabled, "enabled by the managed release verifier");
  test.setTimeout(480_000);
  const targets = liveTargetPaths();
  const qaRoot = resolve(process.cwd(), "../../var/qa/task-6");
  mkdirSync(qaRoot, { recursive: true });
  const failures: string[] = [];
  let activationRequests = 0;
  page.on("pageerror", (error) => failures.push(`pageerror:${error.message}`));
  page.on("console", (message) => { if (message.type() === "error") failures.push(`console:${message.text()}`); });
  page.on("request", (request) => {
    if (/\/adaptations\/policies\/[0-9a-f]{64}\/activate$/u.test(new URL(request.url()).pathname)) activationRequests += 1;
  });
  page.on("response", (response) => {
    if (response.url().includes("/api/") && response.status() >= 400) {
      failures.push(`${response.status()}:${response.request().method()}:${new URL(response.url()).pathname}`);
    }
  });

  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.emulateMedia({ reducedMotion: "reduce", colorScheme: "light" });
  await page.goto("/#/adaptation");
  const adaptation = page.getByRole("region", { name: "适配能力工作台" });
  const zero = adaptation.getByRole("region", { name: "零样本路径" });
  const few = adaptation.getByRole("region", { name: "少样本路径" });
  const graph = page.locator('.graph-preview[aria-label="适配任务关系图"]');
  await expect(adaptation).not.toContainText(FORBIDDEN_RELEASE_COPY);

  await zero.getByLabel("零样本目标任务包").setInputFiles(targets.zero);
  await expect(zero).toHaveAttribute("data-phase", "raw", { timeout: 30_000 });
  await expect(zero).toContainText("目标域网络 A");
  await expect(zero).toContainText("108 个对象 · 220 条关系");
  await expectOnscreenGraph(graph);
  await expect(graph).not.toHaveClass(/graph-preview--with-overlay/u);
  await runGlobal(zero);
  await expect(zero.getByLabel("重点账号").getByRole("button", { name: /待治理核验/u })).toHaveCount(25);
  const zeroIdentity = await laneIdentity(zero);
  const zeroResult = await readRunResult(page, zeroIdentity["run-id"]!);
  await expectBuiltInKnowledge(page, zeroResult);

  const zeroCandidate = zero.getByLabel("重点账号").getByRole("button", { name: /待治理核验/u }).first();
  await zeroCandidate.click();
  await expect(zero.getByText("直接关系证据")).toBeVisible();
  const layoutBeforeClear = await graph.getAttribute("data-layout-count");
  await graph.getByRole("button", { name: "返回适配全图" }).click();
  await expect(graph.getByRole("button", { name: "返回适配全图" })).toHaveCount(0);
  await expect(graph).toHaveAttribute("data-layout-count", layoutBeforeClear!);

  const collectionResponsePromise = page.waitForResponse((response) => response.request().method() === "POST"
    && new URL(response.url()).pathname.endsWith("/adaptations/review-collections"));
  await zero.getByRole("button", { name: "进入治理应用" }).click();
  const collectionResponse = await collectionResponsePromise;
  expect(collectionResponse.ok()).toBe(true);
  const collectionDocument = await collectionResponse.json() as { readonly case: { readonly caseId: string } };
  const concludedCaseId = collectionDocument.case.caseId;
  await expect(page).toHaveURL(/#\/governance$/u);
  const governance = page.getByTestId("governance-workspace");
  const governanceGraph = page.locator('.graph-preview[aria-label="治理关系图"]');
  await expectOnscreenGraph(governanceGraph);
  const governanceModes = governance.getByRole("navigation", { name: "治理工作模式" });
  const firstCandidate = governance.locator(".governance-result-list[aria-label='风险节点'] .governance-result-list__select").first();
  await expect(firstCandidate).toBeVisible({ timeout: 30_000 });
  await firstCandidate.click();
  const selectionBanner = governance.getByRole("status").filter({ hasText: "已在图谱中突出关联节点与关系" });
  await expect(selectionBanner).toBeVisible();
  const evidenceTrigger = governance.locator(".governance-result-list[aria-label='风险节点'] article").first().locator(".governance-result-list__evidence");
  await evidenceTrigger.click();
  const evidence = page.getByRole("dialog");
  await evidence.getByRole("tab", { name: "关系事实", exact: true }).click();
  await expect(evidence.getByRole("tabpanel", { name: "关系事实" })).toContainText("关联账号");
  await page.screenshot({ path: resolve(qaRoot, "01-live-target-a-governance-1920x1080.png") });
  const governanceLayoutBeforeEscape = await governanceGraph.getAttribute("data-layout-count");
  await page.keyboard.press("Escape");
  await expect(evidence).toHaveCount(0);
  await expect(selectionBanner).toBeVisible();
  await expect(governanceGraph).toHaveAttribute("data-layout-count", governanceLayoutBeforeEscape!);

  await page.getByRole("button", { name: "研判助手", exact: true }).click();
  const rag = page.getByRole("complementary", { name: "案例研判助手" });
  await rag.getByLabel("输入研判问题").fill("请说明还需要核对哪些直接关系");
  const assistantResponse = page.waitForResponse((response) => response.request().method() === "POST"
    && new URL(response.url()).pathname.endsWith("/assistant/dispatch"));
  await rag.getByRole("button", { name: "生成报告", exact: true }).click();
  const assistantDocumentResponse = await assistantResponse;
  expect(assistantDocumentResponse.ok()).toBe(true);
  const assistantDocument = await assistantDocumentResponse.json() as {
    readonly answer: string;
    readonly deterministicFallback: boolean;
    readonly generationMode: string | null;
    readonly fallbackPhase: string | null;
    readonly reasonCode: string | null;
  };
  expect(assistantDocument).toMatchObject({
    deterministicFallback: false,
    generationMode: "llm_assisted",
    fallbackPhase: null,
    reasonCode: null,
  });
  expect(assistantDocument.answer).toMatch(/^## 证据核对要求\n\n/u);
  await expect(rag.locator(".governance-assistant-answer")).toBeVisible({ timeout: 30_000 });
  await expect(rag).not.toContainText(/skillCalls|auditHash|modelStateHash/u);

  await governanceModes.getByRole("button", { name: "风险节点", exact: true }).click();
  await firstCandidate.click();
  await evidenceTrigger.click();
  await evidence.getByRole("tab", { name: "人工复核", exact: true }).click();
  await evidence.getByRole("textbox", { name: "复核理由" }).fill("已核对直接关系与邻域事实。");
  await evidence.getByRole("button", { name: "待定", exact: true }).click();
  const currentDecision = evidence.getByText(/当前人工结论：待定/u);
  await expect(currentDecision).toBeVisible();
  expect(await contrastRatio(currentDecision)).toBeGreaterThanOrEqual(4.5);
  await evidence.getByRole("button", { name: "关闭证据档案" }).click();
  await governanceModes.getByRole("button", { name: "研判单", exact: true }).click();
  const caseDetail = governance.locator(".governance-case-detail");
  await caseDetail.getByRole("button", { name: "形成结论" }).click();
  await expect(caseDetail).toContainText("已形成结论");
  const downloadPromise = page.waitForEvent("download");
  await caseDetail.getByRole("button", { name: "HTML", exact: true }).click();
  const reportDownload = await downloadPromise;
  expect(reportDownload.suggestedFilename()).toMatch(/^case-[0-9a-f]{32}\.html$/u);
  const reportPath = await reportDownload.path();
  expect(reportPath).not.toBeNull();
  const reportHtml = readFileSync(reportPath!, "utf8");
  expect(reportHtml).toContain("已核对直接关系与邻域事实");
  expect(reportHtml).toContain(concludedCaseId);

  await page.getByRole("button", { name: "研判助手", exact: true }).click();
  await expect(rag.getByRole("tab", { name: "分析链路" })).toHaveCount(0);
  await rag.getByRole("tab", { name: "历史案例" }).click();
  const similarResponse = page.waitForResponse((response) => response.request().method() === "POST"
    && new URL(response.url()).pathname.endsWith("/similar-cases/search"));
  await rag.getByRole("button", { name: "检索相似历史案例" }).click();
  const similarDocumentResponse = await similarResponse;
  expect(similarDocumentResponse.ok()).toBe(true);
  const similarDocument = await similarDocumentResponse.json() as { readonly items: readonly { readonly caseId: string }[] };
  expect(similarDocument.items.length).toBeGreaterThan(0);
  expect(similarDocument.items.map(({ caseId }) => caseId)).not.toContain(concludedCaseId);
  await expect(rag.getByText(/历史案例 01/u).first()).toBeVisible();

  await page.getByRole("button", { name: "适配能力", exact: true }).click();
  await few.getByLabel("少样本目标任务包").setInputFiles(targets.few);
  await expect(few).toHaveAttribute("data-phase", "raw", { timeout: 30_000 });
  await expect(few).toContainText("目标域网络 B");
  await expect(few).not.toContainText("正向 8 / 负向 8");
  await expectOnscreenGraph(graph);
  await expect(graph).toHaveClass(/graph-preview--with-overlay/u);
  await expect(graph).toHaveAttribute("data-reference-label-count", "16");
  const policyResponsePromise = page.waitForResponse((response) => response.request().method() === "POST"
    && /\/adaptations\/label-sets\/[0-9a-f]{64}\/policies$/u.test(new URL(response.url()).pathname));
  const comparisonResponsePromise = page.waitForResponse((response) => response.request().method() === "GET"
    && /\/adaptations\/runs\/governance-[0-9a-f]{32}\/policies\/[0-9a-f]{64}\/comparison$/u.test(new URL(response.url()).pathname));
  await runGlobal(few);
  const fewIdentity = await laneIdentity(few);
  for (const key of Object.keys(zeroIdentity)) {
    expect(fewIdentity[key], `lane identity ${key} must not alias`).not.toBe(zeroIdentity[key]);
  }
  const baseResult = await readRunResult(page, fewIdentity["run-id"]!);
  expect(baseResult.findings).toHaveLength(108);
  const baseGraphPreview = await readFullGraphPreview(page, baseResult);
  const baseHealthIdentity = await readHealthIdentity(page);
  const baseCheckpointIdentity = publishedCheckpointIdentity();
  expect(parseGlobalModelIdentity(baseResult)).toEqual(parseGlobalModelIdentity(baseHealthIdentity));
  expect(parseGlobalModelIdentity(baseResult)).toEqual(parseGlobalModelIdentity(baseCheckpointIdentity));
  const policyResponse = await policyResponsePromise;
  expect(policyResponse.ok()).toBe(true);
  const policyDocument = await policyResponse.json() as {
    readonly policyHash: string;
    readonly labelSetHash: string;
    readonly status: "ready" | "insufficient_signal";
    readonly selectedLambda: number;
    readonly eligibleLabelCount: number;
    readonly positiveCount: number;
    readonly negativeCount: number;
    readonly baseOutputsImmutable: true;
    readonly binding: AdaptationBindingDocument;
  };
  expect(policyDocument).toMatchObject({
    status: "ready",
    eligibleLabelCount: 16,
    positiveCount: 8,
    negativeCount: 8,
    baseOutputsImmutable: true,
  });
  expectHash(policyDocument.policyHash);
  expectHash(policyDocument.labelSetHash);
  expect([0.25, 0.5, 1]).toContain(policyDocument.selectedLambda);
  expect(policyDocument.binding).toMatchObject({
    artifactId: baseResult.artifactId,
    datasetContentHash: baseResult.datasetContentHash,
    graphVersionHash: baseResult.graphVersionHash,
    runId: baseResult.runId,
    requestHash: baseResult.requestHash,
    resultHash: baseResult.resultHash,
    modelVersionId: baseResult.modelVersionId,
    modelVersionHash: baseResult.modelVersionHash,
    modelStateHash: baseResult.modelStateHash,
    recipeHash: baseHealthIdentity.runtimeRecipeHash,
  });
  for (const hash of [policyDocument.binding.runArtifactHash, policyDocument.binding.codeHash]) expectHash(hash);
  const comparisonResponse = await comparisonResponsePromise;
  expect(comparisonResponse.ok()).toBe(true);
  const comparisonDocument = await comparisonResponse.json() as {
    readonly policyHash: string;
    readonly comparisonHash: string;
    readonly baseOutputsImmutable: true;
    readonly binding: AdaptationBindingDocument;
    readonly total: number;
    readonly rows: readonly {
      readonly nodeId: string;
      readonly baseScore: number;
      readonly baseRank: number;
      readonly adaptedReviewPriority: number;
      readonly adaptedRank: number;
      readonly rankDelta: number;
    }[];
  };
  expect(comparisonDocument).toMatchObject({
    policyHash: policyDocument.policyHash,
    baseOutputsImmutable: true,
    binding: policyDocument.binding,
  });
  expectHash(comparisonDocument.comparisonHash);
  expect(comparisonDocument.total).toBe(108);
  expect(comparisonDocument.rows).toHaveLength(108);
  const completeRanks = Array.from({ length: 108 }, (_, index) => index + 1);
  expect(comparisonDocument.rows.map(({ baseRank }) => baseRank).sort((a, b) => a - b)).toEqual(completeRanks);
  expect(comparisonDocument.rows.map(({ adaptedRank }) => adaptedRank).sort((a, b) => a - b)).toEqual(completeRanks);
  const baseByNode = new Map(baseResult.findings.map((row) => [row.nodeId, row]));
  for (const row of comparisonDocument.rows) {
    expect(row).toMatchObject({ baseScore: baseByNode.get(row.nodeId)?.score, baseRank: baseByNode.get(row.nodeId)?.rank });
    expect(row.rankDelta).toBe(row.adaptedRank - row.baseRank);
    expect(row.adaptedReviewPriority).toBeGreaterThanOrEqual(0);
    expect(row.adaptedReviewPriority).toBeLessThanOrEqual(1);
  }
  const adaptedTop = comparisonDocument.rows.find(({ adaptedRank }) => adaptedRank === 1);
  expect(adaptedTop, "comparison must contain the adapted rank-one candidate").toBeDefined();
  await expect(few).toHaveAttribute("data-phase", "compared", { timeout: 120_000 });
  await expect(few).toContainText("协同组群已就绪 · 108 个账号");
  await expect(few).toHaveAttribute("data-overlay", "community");
  await expect(few.getByRole("group", { name: "排序图层" })).toHaveCount(0);
  await expect(few.locator(".adaptation-table-shell")).toHaveCount(0);
  expect(activationRequests).toBe(0);
  await few.getByLabel("重点账号").getByRole("button", { name: /待治理核验/u }).first().click();
  await expect(few.getByText("直接关系证据")).toBeVisible();
  await expectOnscreenGraph(graph);
  await expect(graph).toHaveClass(/graph-preview--with-overlay/u);
  await expect(graph).toHaveAttribute("data-rendered-label-count", /^(?:[0-9]|1[0-3])$/u);

  const canvas = graph.locator("canvas").first();
  await canvas.evaluate((element) => element.setAttribute("data-live-canvas", "retained"));
  await graph.getByRole("button", { name: "切换到品牌浅色主题" }).click();
  await expect(graph).not.toHaveClass(/graph-preview--focus-dark/u);
  await expect(graph.locator('canvas[data-live-canvas="retained"]')).toHaveCount(1);
  await graph.getByRole("button", { name: "切换到专注深色主题" }).click();
  await expect(graph).toHaveClass(/graph-preview--focus-dark/u);
  await expect(graph.locator('canvas[data-live-canvas="retained"]')).toHaveCount(1);

  await page.setViewportSize({ width: 1280, height: 720 });
  await expectOnscreenGraph(graph);
  await page.screenshot({ path: resolve(qaRoot, "02-live-target-b-adapted-1280x720.png") });
  const cameraBeforeHandoff = await graph.evaluate((element) => ({
    x: Number(element.getAttribute("data-camera-x")),
    y: Number(element.getAttribute("data-camera-y")),
    zoom: Number(element.getAttribute("data-camera-zoom")),
  }));
  const handoffResponsePromise = page.waitForResponse((response) => response.request().method() === "POST"
    && new URL(response.url()).pathname.endsWith("/adaptations/handoffs"));
  const governanceRegistrationPromise = page.waitForResponse((response) => response.request().method() === "GET"
    && /\/target-tasks\/target-task-[0-9a-f]{32}$/u.test(new URL(response.url()).pathname));
  const governanceHandoffPromise = page.waitForResponse((response) => response.request().method() === "GET"
    && /\/adaptations\/handoffs\/[0-9a-f]{64}$/u.test(new URL(response.url()).pathname));
  const governancePolicyPromise = page.waitForResponse((response) => response.request().method() === "GET"
    && /\/adaptations\/policies\/[0-9a-f]{64}$/u.test(new URL(response.url()).pathname));
  const governanceComparisonPromise = page.waitForResponse((response) => response.request().method() === "GET"
    && /\/adaptations\/runs\/governance-[0-9a-f]{32}\/policies\/[0-9a-f]{64}\/comparison$/u.test(new URL(response.url()).pathname));
  await few.getByRole("button", { name: "进入治理应用" }).click();
  const handoffResponse = await handoffResponsePromise;
  expect(handoffResponse.ok()).toBe(true);
  const handoffDocument = await handoffResponse.json() as {
    readonly handoffHash: string;
    readonly targetTaskRegistrationId: string;
    readonly policyHash: string;
    readonly comparisonHash: string;
    readonly decision: string;
    readonly baseModelMutation: boolean;
    readonly binding: AdaptationBindingDocument;
  };
  expect(handoffDocument).toMatchObject({
    targetTaskRegistrationId: fewIdentity["registration-id"],
    policyHash: policyDocument.policyHash,
    comparisonHash: comparisonDocument.comparisonHash,
    decision: "pending_governance_review",
    baseModelMutation: false,
    binding: policyDocument.binding,
  });
  expectHash(handoffDocument.handoffHash);
  await expect(page).toHaveURL(/#\/governance$/u);
  const [governanceRegistration, governanceHandoff, governancePolicy, governanceComparison] = await Promise.all([
    governanceRegistrationPromise,
    governanceHandoffPromise,
    governancePolicyPromise,
    governanceComparisonPromise,
  ]);
  for (const response of [governanceRegistration, governanceHandoff, governancePolicy, governanceComparison]) {
    expect(response.ok(), `${response.request().method()} ${new URL(response.url()).pathname}`).toBe(true);
  }
  expect(new URL(governanceRegistration.url()).pathname).toMatch(
    new RegExp(`/target-tasks/${fewIdentity["registration-id"]}$`, "u"),
  );
  expect(new URL(governanceHandoff.url()).pathname).toMatch(new RegExp(`/adaptations/handoffs/${handoffDocument.handoffHash}$`, "u"));
  expect(new URL(governancePolicy.url()).pathname).toMatch(new RegExp(`/adaptations/policies/${policyDocument.policyHash}$`, "u"));
  expect(await governanceHandoff.json()).toEqual(handoffDocument);
  expect(await governancePolicy.json()).toEqual(policyDocument);
  expect(await governanceComparison.json()).toEqual(comparisonDocument);
  await expect(governance.getByText(/少样本复核顺序已重新校验/u)).toBeVisible({ timeout: 30_000 });
  await governance.getByRole("navigation", { name: "治理工作模式" })
    .getByRole("button", { name: "风险节点", exact: true })
    .click();
  const adaptedFirstCandidate = governance.locator(".governance-result-list[aria-label='风险节点'] .governance-result-list__select").first();
  await expect(adaptedFirstCandidate).toBeVisible();
  await expect(adaptedFirstCandidate).toContainText("基础排序 #");
  await expect(adaptedFirstCandidate).toContainText("适配后复核优先级 #");
  await expect(governanceGraph).toHaveClass(/graph-preview--with-overlay/u);
  const unchangedResult = await readRunResult(page, baseResult.runId);
  const unchangedGraphPreview = await readFullGraphPreview(page, unchangedResult);
  const unchangedHealthIdentity = await readHealthIdentity(page);
  const unchangedCheckpointIdentity = publishedCheckpointIdentity();
  expect(unchangedResult).toEqual(baseResult);
  expect(unchangedGraphPreview).toEqual(baseGraphPreview);
  expect(unchangedGraphPreview.nodes).toEqual(baseGraphPreview.nodes);
  expect(unchangedGraphPreview.edges).toEqual(baseGraphPreview.edges);
  expect(unchangedHealthIdentity).toEqual(baseHealthIdentity);
  expect(unchangedCheckpointIdentity).toEqual(baseCheckpointIdentity);
  await page.getByRole("button", { name: "适配能力", exact: true }).click();
  await expect(few).toHaveAttribute("data-result-hash", fewIdentity["result-hash"]!);
  await expect(graph).toHaveAttribute("data-visible-nodes", "108");
  const cameraAfterReturn = await graph.evaluate((element) => ({
    x: Number(element.getAttribute("data-camera-x")),
    y: Number(element.getAttribute("data-camera-y")),
    zoom: Number(element.getAttribute("data-camera-zoom")),
  }));
  expect(cameraAfterReturn.x).toBeCloseTo(cameraBeforeHandoff.x, 1);
  expect(cameraAfterReturn.y).toBeCloseTo(cameraBeforeHandoff.y, 1);
  expect(cameraAfterReturn.zoom).toBeCloseTo(cameraBeforeHandoff.zoom, 2);

  const firstFewRunId = fewIdentity["run-id"]!;
  await few.getByLabel("少样本目标任务包").setInputFiles(targets.few);
  await expect(few).toHaveAttribute("data-phase", "raw", { timeout: 30_000 });
  const replayPolicyResponse = page.waitForResponse((response) => response.request().method() === "POST"
    && /\/adaptations\/label-sets\/[0-9a-f]{64}\/policies$/u.test(new URL(response.url()).pathname));
  const replayComparisonResponse = page.waitForResponse((response) => response.request().method() === "GET"
    && /\/adaptations\/runs\/governance-[0-9a-f]{32}\/policies\/[0-9a-f]{64}\/comparison$/u.test(new URL(response.url()).pathname));
  await runGlobal(few);
  expect((await replayPolicyResponse).ok()).toBe(true);
  expect((await replayComparisonResponse).ok()).toBe(true);
  await expect(few).toHaveAttribute("data-phase", "compared", { timeout: 120_000 });
  await expect(few).toContainText("协同组群已就绪 · 108 个账号");
  await expect(graph).toHaveAttribute("data-reference-label-count", "16");
  expect(activationRequests).toBe(0);
  const replayIdentity = await laneIdentity(few);
  expect(replayIdentity["run-id"]).not.toBe(firstFewRunId);
  const replayHandoffResponse = page.waitForResponse((response) => response.request().method() === "POST"
    && new URL(response.url()).pathname.endsWith("/adaptations/handoffs"));
  await few.getByRole("button", { name: "进入治理应用", exact: true }).click();
  expect((await replayHandoffResponse).ok()).toBe(true);
  await expect(page).toHaveURL(/#\/governance$/u);
  await page.getByRole("button", { name: "适配能力", exact: true }).click();

  const cdp = await page.context().newCDPSession(page);
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 960, height: 540, deviceScaleFactor: 2, mobile: false,
    screenWidth: 1920, screenHeight: 1080,
  });
  await expect.poll(() => page.evaluate(() => ({ width: innerWidth, dpr: devicePixelRatio })))
    .toEqual({ width: 960, dpr: 2 });
  const overflow = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(overflow.scroll).toBeLessThanOrEqual(overflow.client + 1);
  const zoomCapture = await cdp.send("Page.captureScreenshot", { format: "png", fromSurface: true });
  writeFileSync(resolve(qaRoot, "03-live-target-b-actual-200-percent-1920x1080.png"), Buffer.from(zoomCapture.data, "base64"));
  await cdp.send("Emulation.clearDeviceMetricsOverride");

  await expect(page.locator("body")).not.toContainText(FORBIDDEN_RELEASE_COPY);
  await assertNoApiErrors(page, failures);
});
});
