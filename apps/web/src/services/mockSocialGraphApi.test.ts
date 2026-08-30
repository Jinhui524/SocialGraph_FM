import { describe, expect, it } from "vitest";

import { buildDemoGraphVersion } from "./graphImport";
import { normalizeIntentLocally } from "./intentNormalizer";
import { LocalAnalysisExecutor } from "./localAnalysisExecutor";

describe("deterministic intent normalization", () => {
  it("routes Chinese keywords and extracts targets and time ranges", () => {
    const intent = normalizeIntentLocally({
      text: "请比较张三、李四在 2020-2024 年的影响力与中心性",
    });

    expect(intent.kind).toBe("analysis_request");
    if (intent.kind !== "analysis_request") throw new Error("expected analysis intent");
    expect(intent.task).toBe("centrality");
    expect(intent.targets).toEqual(["张三", "李四"]);
    expect(intent.timeRange).toEqual({ start: "2020", end: "2024" });
    expect(intent.confidence).toBeGreaterThan(0.9);
  });

  it("uses a low-confidence overview instead of inventing an unsupported intent", () => {
    const intent = normalizeIntentLocally({ text: "帮我看看这个数据" });

    expect(intent.kind).toBe("analysis_request");
    if (intent.kind !== "analysis_request") throw new Error("expected analysis intent");
    expect(intent.task).toBe("overview");
    expect(intent.confidence).toBe(0.55);
  });
});

describe("LocalAnalysisExecutor", () => {
  it("runs local graph baselines against a registered graph version", async () => {
    const graph = buildDemoGraphVersion();
    const api = new LocalAnalysisExecutor([graph]);
    const intent = normalizeIntentLocally({ text: "找出关键成员和中心性排名" });
    if (intent.kind !== "analysis_request") throw new Error("expected analysis intent");

    const run = await api.createAnalysis({ graphVersionId: graph.id, intent });

    expect(run.engine).toBe("local_algorithm");
    expect(run.status).toBe("succeeded");
    expect(run.result?.kind).toBe("centrality");
    await expect(api.getAnalysis(run.id)).resolves.toEqual(run);
  });

  it("returns an explicit unavailable result for GFM-only work", async () => {
    const graph = buildDemoGraphVersion();
    const api = new LocalAnalysisExecutor([graph]);
    const intent = normalizeIntentLocally({ text: "预测潜在合作关系" });
    if (intent.kind !== "analysis_request") throw new Error("expected analysis intent");

    const run = await api.createAnalysis({ graphVersionId: graph.id, intent });

    expect(run.engine).toBe("unavailable");
    expect(run.status).toBe("failed");
    expect(run.result).toMatchObject({
      kind: "unavailable",
      code: "GFM_CORE_NOT_CONNECTED",
      requestedTask: "link_prediction",
    });
  });

  it("does not mistake a research-dataset projection for the complete training graph", async () => {
    const base = buildDemoGraphVersion();
    const projection = Object.freeze({
      ...base,
      id: "artifact-projection",
      datasetArtifact: Object.freeze({
        id: "artifact-1",
        datasetName: "Cora",
        checksum: "checksum",
        canonicalGraphHash: "canonical-hash",
        scope: "projection" as const,
      }),
    });
    const api = new LocalAnalysisExecutor([projection]);
    const intent = normalizeIntentLocally({ text: "找出关键成员和中心性排名" });
    if (intent.kind !== "analysis_request") throw new Error("expected analysis intent");

    const run = await api.createAnalysis({ graphVersionId: projection.id, intent });

    expect(run.engine).toBe("unavailable");
    expect(run.status).toBe("failed");
    expect(run.result).toMatchObject({
      kind: "unavailable",
      code: "DATASET_PROJECTION_REQUIRES_SERVER_ANALYSIS",
    });
  });

  it("does not analyse a missing graph", async () => {
    const api = new LocalAnalysisExecutor();
    const intent = normalizeIntentLocally({ text: "生成概览" });
    if (intent.kind !== "analysis_request") throw new Error("expected analysis intent");
    const run = await api.createAnalysis({ graphVersionId: "missing", intent });

    expect(run.status).toBe("failed");
    expect(run.error).toContain("GRAPH_VERSION_NOT_FOUND");
  });
});
