import { describe, expect, it } from "vitest";

import { buildNormalizeIntentPayload } from "../services/intentNormalizer";
import { normalizeIntentLocally } from "../services/mockSocialGraphApi";
import {
  CHINESE_INTENT_EVALUATION_CASES,
  SAFE_GRAPH_CONTEXT_FIXTURE,
} from "./fixtures/intentEvaluation";
import {
  findDisallowedGraphContextPaths,
  findForbiddenIntentRequestPaths,
  scoreIntentEvaluation,
} from "./helpers/intentEvaluation";

describe("Chinese intent evaluation fixture", () => {
  it("contains at least 50 unique cases and covers chat plus every analysis task", () => {
    const ids = CHINESE_INTENT_EVALUATION_CASES.map(({ id }) => id);
    const tasks = new Set(CHINESE_INTENT_EVALUATION_CASES.map(({ expectedTask }) => expectedTask));

    expect(CHINESE_INTENT_EVALUATION_CASES.length).toBeGreaterThanOrEqual(50);
    expect(new Set(ids).size).toBe(ids.length);
    expect(CHINESE_INTENT_EVALUATION_CASES.some(({ expectedKind }) => expectedKind === "chat")).toBe(true);
    expect(tasks).toEqual(new Set([
      undefined,
      "overview",
      "centrality",
      "bridge_detection",
      "community",
      "link_prediction",
      "node_role",
      "similar_structure",
    ]));
    expect(CHINESE_INTENT_EVALUATION_CASES.some(({ expectedTargets }) => expectedTargets?.length)).toBe(true);
    expect(CHINESE_INTENT_EVALUATION_CASES.some(({ expectedTimeRange }) => expectedTimeRange)).toBe(true);
  });

  it("keeps the deterministic fallback at the 90% routing baseline", async () => {
    const report = await scoreIntentEvaluation(CHINESE_INTENT_EVALUATION_CASES, (text) => {
      const normalized = normalizeIntentLocally({ text });
      if (normalized.kind === "chat") return { kind: "chat" };
      return {
        kind: normalized.kind,
        task: normalized.task,
        targets: normalized.targets,
        timeRange: normalized.timeRange,
      };
    });

    // The fallback handles a deliberately narrow set of help/greeting phrases;
    // the HTTP LLM layer can improve conversational recall without weakening
    // deterministic task, target, and date parsing.
    expect(report.kindAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(report.taskAccuracy).toBeGreaterThanOrEqual(0.9);
    expect(report.targetAccuracy).toBe(1);
    expect(report.timeRangeAccuracy).toBe(1);
  });
});

describe("intent request privacy contract", () => {
  it("accepts only the aggregate graph-context whitelist", () => {
    expect(findDisallowedGraphContextPaths(SAFE_GRAPH_CONTEXT_FIXTURE)).toEqual([]);
    expect(findDisallowedGraphContextPaths({
      ...SAFE_GRAPH_CONTEXT_FIXTURE,
      nodes: [{ id: "person-1", label: "张三" }],
      sourceFile: "private.csv",
      timeRange: { start: "2020", end: "2024", rawTimestamps: ["2020-01-01"] },
    })).toEqual([
      "graphContext.nodes",
      "graphContext.sourceFile",
      "graphContext.timeRange.rawTimestamps",
    ]);
  });

  it("detects raw graph data even when it is nested in an intent payload", () => {
    const safeRequest = {
      text: "分析 2020-2024 年的社区结构",
      graphContext: SAFE_GRAPH_CONTEXT_FIXTURE,
    };
    expect(findForbiddenIntentRequestPaths(safeRequest)).toEqual([]);

    expect(findForbiddenIntentRequestPaths({
      ...safeRequest,
      debug: {
        sourceFile: "relationships.csv",
        canonicalGraph: {
          nodes: [{ id: "n1", attributes: { phone: "13800000000" } }],
          edges: [{ source: "n1", target: "n2" }],
        },
      },
    })).toEqual([
      "request.debug.sourceFile",
      "request.debug.canonicalGraph",
      "request.debug.canonicalGraph.nodes",
      "request.debug.canonicalGraph.nodes[0].attributes",
      "request.debug.canonicalGraph.edges",
    ]);
  });

  it("production payload reconstruction strips excess graph fields", () => {
    const graphContextWithExcessData = {
      ...SAFE_GRAPH_CONTEXT_FIXTURE,
      nodes: [{ id: "n1", label: "不得发送的节点名称" }],
      edges: [{ source: "n1", target: "n2" }],
      sourceFile: "private-relationships.csv",
      attributes: { phone: "13800000000" },
    };
    const payload = buildNormalizeIntentPayload({
      text: "找出关键成员",
      graphContext: graphContextWithExcessData,
    });

    expect(payload.graphContext).toBeDefined();
    expect(findDisallowedGraphContextPaths(payload.graphContext)).toEqual([]);
    expect(findForbiddenIntentRequestPaths(payload)).toEqual([]);
    expect(JSON.stringify(payload)).not.toContain("private-relationships.csv");
    expect(JSON.stringify(payload)).not.toContain("不得发送的节点名称");
    expect(JSON.stringify(payload)).not.toContain("13800000000");
  });
});
