import { describe, expect, it } from "vitest";
import type { AnalysisResult } from "../types/graph";
import type { CoreWorkbenchServiceState } from "../types/core";
import { analysisEngineLabel, describeUnavailableAnalysis } from "./analysisUnavailable";

const coreResult: Extract<AnalysisResult, { kind: "unavailable" }> = {
  kind: "unavailable",
  code: "GFM_CORE_NOT_CONNECTED",
  message: "legacy generic message",
  requestedTask: "community",
};

function connected(servingReady: boolean): CoreWorkbenchServiceState {
  return {
    state: "connected",
    capabilities: {
      schemaVersion: "socialgraph-fm.core-capabilities/2.0",
      registryHash: "1".repeat(64),
      registryGeneration: servingReady ? 1 : 0,
      servingReady,
      models: [],
      tasks: [],
      readiness: { modelValidated: servingReady, coreServingReady: servingReady },
    },
  };
}

describe("analysis unavailable messaging", () => {
  it("distinguishes connection state from an empty formal registry", () => {
    expect(describeUnavailableAnalysis(coreResult, { state: "checking" })).toMatch(/仍在检查/);
    expect(
      describeUnavailableAnalysis(coreResult, { state: "unavailable", code: "offline" }),
    ).toMatch(/服务当前不可用/);
    expect(describeUnavailableAnalysis(coreResult, connected(false))).toMatch(
      /服务已连接.*尚无正式晋级/,
    );
    expect(describeUnavailableAnalysis(coreResult, connected(true))).toMatch(/没有.*兼容/);
  });

  it("preserves non-GFM unavailable evidence", () => {
    const projection: Extract<AnalysisResult, { kind: "unavailable" }> = {
      ...coreResult,
      code: "DATASET_PROJECTION_REQUIRES_SERVER_ANALYSIS",
      message: "科研投影必须由服务端完整分析。",
    };

    expect(describeUnavailableAnalysis(projection, connected(false))).toBe(projection.message);
  });

  it("labels unavailable engines without implying that a connected API is missing", () => {
    const baseRun = {
      id: "run-1",
      graphVersionId: "graph-1",
      intent: {
        kind: "analysis_request" as const,
        normalizedText: "分析社区",
        task: "community" as const,
        targets: [],
        confidence: 1,
        filters: {},
        meta: {
          schemaVersion: "1.1" as const,
          source: "deterministic_fallback" as const,
          requestId: "request-1",
          warnings: [],
        },
      },
      status: "failed" as const,
      createdAt: "2026-08-16T00:00:00Z",
    };

    expect(analysisEngineLabel(undefined)).toBe("等待图数据");
    expect(analysisEngineLabel({ ...baseRun, engine: "local_algorithm" })).toBe("本地图算法");
    expect(analysisEngineLabel({ ...baseRun, engine: "gfm" })).toBe("GFM 模型");
    expect(analysisEngineLabel({ ...baseRun, engine: "unavailable", result: coreResult })).toBe(
      "GFM 模型未就绪",
    );
    expect(
      analysisEngineLabel({
        ...baseRun,
        engine: "unavailable",
        result: {
          ...coreResult,
          code: "DATASET_PROJECTION_REQUIRES_SERVER_ANALYSIS",
        },
      }),
    ).toBe("分析不可用");
  });
});
