import { describe, expect, it, vi } from "vitest";

import {
  ASSISTANT_PUBLIC_SKILLS,
  ASSISTANT_SKILLS_SCHEMA,
  ASSISTANT_SKILL_REQUEST_SCHEMA,
  ASSISTANT_SKILL_RESULT_SCHEMA,
  ASSISTANT_SKILL_POLICIES,
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
  it("executes one named read-only Assistant Skill with bounded context", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: ASSISTANT_SKILL_RESULT_SCHEMA,
      executionId: `assistant-exec-${"4".repeat(32)}`,
      skill: "generate_global_situation_report",
      answer: "## 全局态势报告\n\n请人工复核。",
      result: { runId: context.runId },
      skillCalls: [{ skill: "inspect_graph", requestHash: "6".repeat(64), resultHash: "7".repeat(64) }],
      evidenceRefs: [{ label: "图谱概况", sourceKind: "skill", hash: "9".repeat(64) }],
      citedHashes: ["7".repeat(64)],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);

    await expect(client.executeAssistant(context, "generate_global_situation_report", "  生成态势报告  ")).resolves.toMatchObject({
      skill: "generate_global_situation_report",
      answer: expect.stringContaining("全局态势报告"),
      skillCalls: [{ skill: "inspect_graph" }],
    });

    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(call[0]).toBe("http://api.test/api/v2/gfm/governance/assistant/execute");
    const body = JSON.parse(String(call[1].body));
    expect(body).toEqual({
      schemaVersion: ASSISTANT_SKILL_REQUEST_SCHEMA,
      skill: "generate_global_situation_report",
      graph: context.graph,
      model: context.model,
      message: "生成态势报告",
      context: {
        runId: context.runId,
        caseId: context.caseId,
        caseHash: context.caseHash,
        selectedTarget: { targetType: "node", targetId: "node-a" },
        topK: 100,
      },
    });
    expect(JSON.stringify(body)).not.toContain("findings");
    expect(JSON.stringify(body)).not.toContain("textFeatures");
  });

  it("accepts only the complete ordered Assistant Skill catalog", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: ASSISTANT_SKILLS_SCHEMA,
      items: ASSISTANT_SKILL_POLICIES.map((policy) => ({
        ...policy,
      })),
      catalogHash: HASH,
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);

    await expect(client.assistantCatalog()).resolves.toMatchObject({ items: expect.arrayContaining([
      expect.objectContaining({ name: "summarize_node_evidence", readOnly: true }),
      expect.objectContaining({ name: "generate_case_review_draft", confirmationRequired: false }),
    ]) });
    const call = fetcher.mock.calls[0] as unknown as [string, RequestInit];
    expect(call[0]).toBe("http://api.test/api/v2/gfm/governance/assistant/skills");
  });

  it("rejects Assistant catalog order and permission drift", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: ASSISTANT_SKILLS_SCHEMA,
      items: [...ASSISTANT_SKILL_POLICIES].reverse().map((policy) => ({
        ...policy,
      })),
      catalogHash: HASH,
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.assistantCatalog()).rejects.toMatchObject({
      code: "GFM_GOVERNANCE_SKILLS_RESPONSE_INVALID",
    });
  });

  it("rejects fallback-shaped Assistant responses", async () => {
    const fetcher = vi.fn(async () => response({
      schemaVersion: "socialgraph-fm.governance-assistant-dispatch/1.0",
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      answer: "回答。",
      result: {},
      skillCalls: [],
      evidenceRefs: [],
      citedHashes: [],
      auditHash: "8".repeat(64),
    }));
    const client = new GovernanceSkillsClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.executeAssistant(context, "answer_governance_question", "回答问题")).rejects.toMatchObject({
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
