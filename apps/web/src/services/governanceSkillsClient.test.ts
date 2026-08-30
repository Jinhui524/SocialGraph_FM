import { describe, expect, it, vi } from "vitest";

import {
  GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
  type GovernanceSkillsContext,
} from "../types/governanceSkills";
import { onlineRun } from "../test/fixtures/governanceOnline";
import { GovernanceSkillsClient } from "./governanceSkillsClient";

const HASH = "a".repeat(64);
const context: GovernanceSkillsContext = {
  graph: {
    artifactId: `governance-artifact-${"1".repeat(32)}`,
    datasetContentHash: HASH,
    graphVersionHash: "b".repeat(64),
  },
  model: { modelVersionId: "socialgraph-fm-global/test", modelStateHash: "c".repeat(64) },
  runId: `governance-${"2".repeat(32)}`,
  caseId: `case-${"3".repeat(32)}`,
  caseHash: "d".repeat(64),
  selectedNodeIds: ["node-b", "node-a"],
  selectedTarget: { kind: "node", targetId: "node-a" },
};

function response(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "Content-Type": "application/json" } });
}

describe("SocialGraph-FM Governance Skills browser client", () => {
  it("dispatches only bounded identities and selected target context", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "start_analysis",
      answerMode: null,
      status: "confirmation_required",
      answer: "请确认开始分析。",
      result: { plan: { topK: 50 } },
      deterministicFallback: true,
      confirmation: {
        token: `governance-confirm-${"5".repeat(64)}`,
        action: "run_governance_analysis",
        requestDigest: "6".repeat(64),
        expiresAt: "2026-08-19T12:00:00Z",
      },
      navigation: null,
      skillCalls: [],
      citedHashes: ["7".repeat(64)],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);

    await expect(client.dispatchAssistant(context, "  开始分析  ", { topK: 50 })).resolves.toMatchObject({
      intent: "start_analysis",
      status: "confirmation_required",
      confirmation: { action: "run_governance_analysis" },
    });

    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(call[0]).toBe("http://api.test/api/v2/gfm/governance/assistant/dispatch");
    const body = JSON.parse(String(call[1].body));
    expect(body).toEqual({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      graph: context.graph,
      model: context.model,
      message: "开始分析",
      context: {
        runId: context.runId,
        caseId: context.caseId,
        caseHash: context.caseHash,
        selectedTarget: { targetType: "node", targetId: "node-a" },
        topK: 50,
      },
    });
    expect(JSON.stringify(body)).not.toContain("findings");
    expect(JSON.stringify(body)).not.toContain("textFeatures");
  });

  it("sends a closed intent override for system-generated summaries", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "answer",
      answerMode: "analysis_summary",
      status: "completed",
      answer: "## 治理摘要\n已完成分析。",
      result: {},
      deterministicFallback: true,
      generationMode: "deterministic_report",
      fallbackPhase: "narration",
      reasonCode: "LLM_DOWN",
      evidenceRefs: [{ label: "分析结果 · 图谱检查", sourceKind: "skill", hash: "9".repeat(64) }],
      confirmation: null,
      navigation: null,
      skillCalls: [{ skill: "inspect_graph", requestHash: "6".repeat(64), resultHash: "7".repeat(64) }],
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);

    await expect(client.dispatchAssistant(context, "请概括本次治理分析结论。", { intent: "answer", answerMode: "analysis_summary", narrationMode: "deterministic_only" }))
      .resolves.toMatchObject({
        answerMode: "analysis_summary",
        generationMode: "deterministic_report",
        fallbackPhase: "narration",
        reasonCode: "LLM_DOWN",
        evidenceRefs: [{ label: "分析结果 · 图谱检查", sourceKind: "skill" }],
        skillCalls: [{ skill: "inspect_graph" }],
      });

    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(call[1].body))).toMatchObject({
      intent: "answer",
      answerMode: "analysis_summary",
      narrationMode: "deterministic_only",
      message: "请概括本次治理分析结论。",
    });
  });

  it("keeps same-version legacy dispatch responses compatible", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "answer",
      status: "completed",
      answer: "## 治理摘要\n旧服务回答。",
      result: {},
      deterministicFallback: true,
      confirmation: null,
      navigation: null,
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient(
      "http://api.test/api/v2/gfm/governance",
      fetcher as unknown as typeof fetch,
    );

    await expect(client.dispatchAssistant(context, "请概括治理风险"))
      .resolves.toMatchObject({ answerMode: "overview", skillCalls: [] });
  });

  it.each([
    ["start_analysis", "submit_review"],
    ["submit_review", "run_governance_analysis"],
    ["draft_report", "submit_review"],
  ])("rejects a %s dispatch carrying the wrong %s confirmation", async (intent, action) => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent,
      answerMode: null,
      status: "confirmation_required",
      answer: "需要确认。",
      result: {},
      deterministicFallback: true,
      confirmation: {
        token: `governance-confirm-${"5".repeat(64)}`,
        action,
        requestDigest: "6".repeat(64),
        expiresAt: "2026-08-19T12:00:00Z",
      },
      navigation: null,
      skillCalls: [],
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.dispatchAssistant(context, "继续")).rejects.toMatchObject({
      code: "GFM_GOVERNANCE_SKILLS_RESPONSE_INVALID",
    });
  });

  it("rejects navigation unless it belongs to a completed open-review intent", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "answer",
      answerMode: "review_guidance",
      status: "completed",
      answer: "回答。",
      result: {},
      deterministicFallback: true,
      confirmation: null,
      navigation: {
        view: "governance_review",
        runId: context.runId,
        caseId: context.caseId,
        target: { targetType: "node", targetId: "node-a" },
      },
      skillCalls: [],
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.dispatchAssistant(context, "回答问题")).rejects.toMatchObject({
      code: "GFM_GOVERNANCE_SKILLS_RESPONSE_INVALID",
    });
  });

  it("accepts a completed open-review navigation bound to the exact target", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "open_review",
      answerMode: null,
      status: "completed",
      answer: "已打开复核。",
      result: {},
      deterministicFallback: true,
      confirmation: null,
      navigation: {
        view: "governance_review",
        runId: context.runId,
        caseId: context.caseId,
        target: { targetType: "node", targetId: "node-a" },
      },
      skillCalls: [],
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.dispatchAssistant(context, "打开复核")).resolves.toMatchObject({
      navigation: { view: "governance_review", target: { targetType: "node", targetId: "node-a" } },
    });
  });

  it("rejects a completed open-review dispatch without navigation", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "open_review",
      answerMode: null,
      status: "completed",
      answer: "已打开复核。",
      result: {},
      deterministicFallback: true,
      confirmation: null,
      navigation: null,
      skillCalls: [],
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.dispatchAssistant(context, "打开复核")).rejects.toMatchObject({
      code: "GFM_GOVERNANCE_SKILLS_RESPONSE_INVALID",
    });
  });

  it("accepts only the complete public skill catalog", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      items: GOVERNANCE_PUBLIC_SKILLS.map((name) => ({
        name,
        readOnly: !["run_governance_analysis", "draft_review_report"].includes(name),
        confirmationRequired: ["run_governance_analysis", "draft_review_report"].includes(name),
        description: name,
        parameterSchema: { type: "object" },
      })),
      catalogHash: HASH,
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.catalog()).resolves.toMatchObject({ items: expect.arrayContaining([
      expect.objectContaining({ name: "get_evidence_subgraph" }),
      expect.objectContaining({ name: "draft_review_report" }),
    ]) });
  });

  it("rejects catalog order or permission drift from the generated contract", async () => {
    const reversed = [...GOVERNANCE_PUBLIC_SKILLS].reverse();
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      items: reversed.map((name) => ({
        name,
        readOnly: name !== "run_governance_analysis" && name !== "draft_review_report",
        confirmationRequired: name === "run_governance_analysis" || name === "draft_review_report",
        description: name,
        parameterSchema: { type: "object" },
      })),
      catalogHash: HASH,
    }));
    const client = new GovernanceSkillsClient(
      "http://api.test/api/v2/gfm/governance",
      fetcher as unknown as typeof fetch,
    );

    await expect(client.catalog()).rejects.toMatchObject({
      code: "GFM_GOVERNANCE_SKILLS_RESPONSE_INVALID",
    });
  });

  it("binds knowledge search to the exact graph and model identity", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      items: [{ sourceLabel: "Model card", sourceUri: "local:model", contentHash: HASH, chunkHash: "d".repeat(64), text: "Evidence", rank: 1 }],
      indexHash: "e".repeat(64),
      auditHash: "f".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.searchKnowledge(context, "  Russia model evidence  ")).resolves.toMatchObject({ items: [{ rank: 1 }] });
    expect(fetcher).toHaveBeenCalledWith(
      "http://api.test/api/v2/gfm/governance/knowledge/search",
      expect.objectContaining({ method: "POST" }),
    );
    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    const body = JSON.parse(String(call[1].body));
    expect(body).toEqual({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      graph: context.graph,
      model: context.model,
      query: "Russia model evidence",
      limit: 5,
    });
  });

  it("sends bounded assistant context and validates citations", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: "socialgraph-fm.governance-assistant/1.0",
      turnId: `governance-turn-${"4".repeat(32)}`,
      answer: "Verified answer",
      deterministicFallback: false,
      skillCalls: [{ skill: "inspect_graph", requestHash: "d".repeat(64), resultHash: "e".repeat(64) }],
      citedHashes: ["f".repeat(64)],
      auditHash: "0".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.assistantTurn(context, "  What supports this?  ")).resolves.toMatchObject({ answer: "Verified answer", skillCalls: [{ skill: "inspect_graph" }] });
    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(call[0]).toBe("http://api.test/api/v2/gfm/governance/assistant/turn");
    expect(JSON.parse(String(call[1].body))).toMatchObject({
      graph: context.graph,
      model: context.model,
      message: "What supports this?",
      context: { runId: context.runId, caseId: context.caseId, selectedNodeIds: ["node-a", "node-b"] },
    });
  });

  it("keeps execution separate from the one-time confirmation", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response({
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        executionId: `governance-exec-${"5".repeat(32)}`,
        skill: "run_governance_analysis",
        status: "confirmation_required",
        result: { confirmationPlan: { topK: 100 } },
        confirmation: { token: `governance-confirm-${"6".repeat(64)}`, action: "run_governance_analysis", requestDigest: "7".repeat(64), expiresAt: "2026-08-18T12:00:00Z" },
        provenance: { inputHash: "8".repeat(64) },
        auditHash: "9".repeat(64),
      }))
      .mockResolvedValueOnce(response({
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        action: "run_governance_analysis",
        status: "completed",
        result: { ...onlineRun(), runId: context.runId },
        auditHash: "a".repeat(64),
      }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    const prepared = await client.executeSkill(context, "run_governance_analysis", { protocol: "global", topK: 100 });
    expect(prepared.status).toBe("confirmation_required");
    expect(fetcher.mock.calls[0]?.[0]).toBe("http://api.test/api/v2/gfm/governance/skills/execute");
    await expect(client.confirmSkill(prepared.confirmation!.token)).resolves.toMatchObject({ action: "run_governance_analysis", status: "completed" });
    expect(fetcher.mock.calls[1]?.[0]).toBe("http://api.test/api/v2/gfm/governance/skills/confirm");
  });

  it("validates weighted same-model similar cases", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      query: { caseId: context.caseId },
      items: [{
        caseId: `case-${"4".repeat(32)}`,
        score: 0.83,
        components: { embedding: 0.8, structure: 0.7, modality: 0.6 },
        graphVersionHash: context.graph.graphVersionHash,
        modelStateHash: context.model.modelStateHash,
        kindKey: "node",
        kindEntries: [{ kind: "node", targetIds: ["node-a"] }],
        concludedAt: "2026-08-18T08:00:00Z",
        recordHash: "b".repeat(64),
      }],
      indexHash: "c".repeat(64),
      backfill: { succeeded: 1, failed: 0 },
      auditHash: "d".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.searchSimilarCases(context, {})).resolves.toMatchObject({ items: [{ score: 0.83, components: { embedding: 0.8 } }] });
    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(call[0]).toBe("http://api.test/api/v2/gfm/governance/similar-cases/search");
    expect(JSON.parse(String(call[1].body))).toMatchObject({
      runId: context.runId,
      kindEntries: [{ kind: "node", targetIds: ["node-a"] }],
      limit: 10,
    });
    expect(JSON.parse(String(call[1].body))).not.toHaveProperty("caseId");
  });

  it("keeps explicit case retrieval exclusive from run-object retrieval", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      query: { caseId: context.caseId },
      items: [],
      indexHash: "c".repeat(64),
      backfill: { succeeded: 0, failed: 0 },
      auditHash: "d".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);

    await client.searchSimilarCases(context, { caseId: context.caseId });
    const explicitCase = JSON.parse(String((fetcher.mock.calls[0] as unknown as [string, RequestInit])[1].body));
    expect(explicitCase).toMatchObject({ caseId: context.caseId, limit: 10 });
    expect(explicitCase).not.toHaveProperty("runId");
    expect(explicitCase).not.toHaveProperty("kindEntries");

    await expect(client.searchSimilarCases(context, {
      caseId: context.caseId,
      runId: context.runId,
      kindEntries: [{ kind: "node", targetIds: ["node-a"] }],
    })).rejects.toMatchObject({ code: "GFM_GOVERNANCE_SIMILAR_CASE_QUERY_INVALID" });
    expect(fetcher).toHaveBeenCalledTimes(1);
  });
});
