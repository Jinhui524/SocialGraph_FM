import { describe, expect, it, vi } from "vitest";

import type { GraphVersion } from "../types/graph";
import { buildDemoGraphVersion } from "./graphImport";
import {
  buildGraphContextSummary,
  HttpIntentNormalizer,
  normalizeIntentLocally,
} from "./intentNormalizer";

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function graphWithPrivateFacts(): GraphVersion {
  const demo = buildDemoGraphVersion();
  return {
    ...demo,
    sourceFile: "绝密成员关系.csv",
    nodes: demo.nodes.map((node, index) => ({
      ...node,
      label: `保密成员-${index}`,
      attributes: { privatePhone: "13800000000", privateNote: "不要发送这段属性值" },
    })),
    edges: demo.edges.map((edge) => ({
      ...edge,
      attributes: { contractValue: 9_999_999 },
    })),
  };
}

describe("HttpIntentNormalizer privacy boundary", () => {
  it("sends only user text and the whitelisted aggregate graph summary", async () => {
    const fetcher = vi.fn(async (..._args: Parameters<typeof fetch>) => jsonResponse({
      kind: "analysis_request",
      normalizedText: "计算节点度数中心性",
      task: "centrality",
      targets: [],
      confidence: 0.96,
      filters: {},
      meta: {
        schemaVersion: "1.0",
        source: "llm",
        requestId: "request-1",
        model: "test-model",
        warnings: [],
      },
    }));
    const normalizer = new HttpIntentNormalizer({
      baseUrl: "http://api.test",
      fetcher: fetcher as unknown as typeof fetch,
    });
    const graph = graphWithPrivateFacts();

    const result = await normalizer.normalizeIntent({
      text: "分析这张图中的中心性",
      graphContext: buildGraphContextSummary(graph),
    });

    expect(result.kind).toBe("analysis_request");
    const init = fetcher.mock.calls[0][1] as RequestInit;
    const payload = JSON.parse(String(init.body)) as Record<string, unknown>;
    expect(Object.keys(payload)).toEqual(["text", "graphContext"]);
    const context = payload.graphContext as Record<string, unknown>;
    expect(context).not.toHaveProperty("nodes");
    expect(context).not.toHaveProperty("edges");
    expect(context).not.toHaveProperty("sourceFile");
    const serialized = JSON.stringify(payload);
    expect(serialized).not.toContain("保密成员");
    expect(serialized).not.toContain("13800000000");
    expect(serialized).not.toContain("不要发送这段属性值");
    expect(serialized).not.toContain("9999999");
    expect(serialized).not.toContain("绝密成员关系.csv");
  });

  it("uses deterministic normalization when the API cannot be reached", async () => {
    const fetcher = vi.fn(async () => { throw new TypeError("network unavailable"); });
    const normalizer = new HttpIntentNormalizer({
      baseUrl: "http://api.test",
      fetcher: fetcher as unknown as typeof fetch,
    });

    const result = await normalizer.normalizeIntent({ text: "识别关键桥接节点" });

    expect(result.kind).toBe("analysis_request");
    expect(result.meta.source).toBe("deterministic_fallback");
    expect(result.meta.warnings[0]).toContain("本地规则");
    if (result.kind === "analysis_request") expect(result.task).toBe("bridge_detection");
  });

  it("removes targets that are not grounded in the original text", async () => {
    const fetcher = vi.fn(async () => jsonResponse({
      kind: "analysis_request",
      normalizedText: "比较指定成员",
      task: "centrality",
      targets: ["张三", "模型臆造的人名"],
      confidence: 0.9,
      filters: {},
      meta: { schemaVersion: "1.0", source: "llm", requestId: "request-2", warnings: [] },
    }));
    const normalizer = new HttpIntentNormalizer({
      baseUrl: "http://api.test",
      fetcher: fetcher as unknown as typeof fetch,
    });

    const result = await normalizer.normalizeIntent({ text: "分析张三的中心性" });

    expect(result.kind).toBe("analysis_request");
    if (result.kind === "analysis_request") expect(result.targets).toEqual(["张三"]);
  });

  it("accepts grounded schema 1.1 terms but ignores legacy mode and layout controls", async () => {
    const fetcher = vi.fn(async () => jsonResponse({
      kind: "analysis_request",
      normalizedText: "查看张三的两跳邻域",
      task: "overview",
      targets: ["张三"],
      confidence: 0.95,
      filters: {},
      view: {
        mode: "local",
        focusTerms: ["张三", "模型臆造节点"],
        depth: 2,
        nodeTypeTerms: [],
        edgeTypeTerms: ["合作"],
        layoutPreset: "compact",
      },
      meta: { schemaVersion: "1.1", source: "llm", requestId: "request-v11", warnings: [] },
    }));
    const normalizer = new HttpIntentNormalizer({
      baseUrl: "http://api.test",
      fetcher: fetcher as unknown as typeof fetch,
    });

    const result = await normalizer.normalizeIntent({ text: "查看张三的两跳邻居，只看合作关系" });

    expect(result.meta.schemaVersion).toBe("1.1");
    expect(result.kind).toBe("analysis_request");
    if (result.kind === "analysis_request") {
      expect(result.view).toMatchObject({
        focusTerms: ["张三"],
        depth: 2,
        edgeTypeTerms: ["合作"],
      });
      expect(result.view).not.toHaveProperty("mode");
      expect(result.view).not.toHaveProperty("layoutPreset");
    }
  });

  it("does not re-create ignored legacy layout fields through local fallback", async () => {
    const fetcher = vi.fn(async () => jsonResponse({
      kind: "analysis_request",
      normalizedText: "查看张三的两跳邻域",
      task: "overview",
      targets: ["张三"],
      confidence: 0.95,
      filters: {},
      view: { mode: "local", layoutPreset: "compact" },
      meta: { schemaVersion: "1.1", source: "llm", requestId: "request-legacy-only", warnings: [] },
    }));
    const normalizer = new HttpIntentNormalizer({
      baseUrl: "http://api.test",
      fetcher: fetcher as unknown as typeof fetch,
    });

    const result = await normalizer.normalizeIntent({ text: "查看张三的两跳邻居" });

    expect(result.kind).toBe("analysis_request");
    if (result.kind === "analysis_request") {
      expect(result.view?.focusTerms).toEqual(["张三"]);
      expect(result.view).not.toHaveProperty("mode");
      expect(result.view).not.toHaveProperty("layoutPreset");
    }
  });
});

describe("intent behavior", () => {
  it("keeps ordinary conversation out of the graph analysis executor", () => {
    const result = normalizeIntentLocally({ text: "你好" });

    expect(result.kind).toBe("chat");
    if (result.kind === "chat") expect(result.reply).toContain("CSV / JSON");
  });

  it("reports configured LLM capability", async () => {
    const fetcher = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/health")) return jsonResponse({ status: "ok", service: "socialgraph-fm-api", version: "0.1.0" });
      return jsonResponse({
        intentNormalization: { configured: true, mode: "llm_with_fallback", model: "example-model" },
      });
    });
    const normalizer = new HttpIntentNormalizer({
      baseUrl: "http://api.test",
      fetcher: fetcher as unknown as typeof fetch,
    });

    await expect(normalizer.checkStatus()).resolves.toEqual({
      state: "llm",
      label: "LLM 已配置 · 等待调用验证",
      model: "example-model",
    });
  });

  it("creates deterministic local and path view commands during fallback", () => {
    const local = normalizeIntentLocally({ text: "查看张三的两跳邻居，只看合作关系" });
    expect(local.kind).toBe("analysis_request");
    if (local.kind === "analysis_request") {
      expect(local.targets).toContain("张三");
      expect(local.view).toMatchObject({
        mode: "local",
        depth: 2,
        focusTerms: ["张三"],
        edgeTypeTerms: ["合作"],
      });
      expect(local.meta.schemaVersion).toBe("1.1");
    }

    const path = normalizeIntentLocally({ text: "显示张三到李四的最短路径" });
    expect(path.kind).toBe("analysis_request");
    if (path.kind === "analysis_request") {
      expect(path.targets).toEqual(["张三", "李四"]);
      expect(path.view).toMatchObject({ mode: "path", focusTerms: ["张三", "李四"] });
    }
  });
});
