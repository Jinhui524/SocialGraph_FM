import { describe, expect, it } from "vitest";

import { sha256Canonical } from "./graphIdentity";
import {
  parseResearchCapabilities,
  parseResearchRunResult,
  parseResearchScenarioPreview,
  parseResearchScenarios,
} from "./researchContracts";

const TASKS = [
  "research.content_policy_review",
  "research.account_risk_review",
  "research.signed_relation_review",
  "core.collaboration_completion",
] as const;

function hashed<T extends Record<string, unknown>>(payload: T, field: string) {
  return { ...payload, [field]: sha256Canonical(payload) };
}

function capabilityPayload() {
  return hashed({
    schemaVersion: "socialgraph-fm.research/1.0",
    channel: "research",
    releaseLabel: "SocialGraph-FM Research",
    seed: 1729,
    preliminary: true,
    researchServingReady: false,
    unavailableReason: "RESEARCH_MODEL_NOT_INSTALLED",
    model: null,
    taskIds: TASKS,
    upload: {
      compatibleTaskIds: ["core.collaboration_completion"],
      auxiliaryCapabilities: ["similar-nodes"],
      minNodes: 5,
      maxNodes: 50_000,
      maxEdges: 1_500_000,
    },
  }, "capabilityHash");
}

function scenarioPayload() {
  return hashed({
    schemaVersion: "socialgraph-fm.research/1.0",
    releaseLabel: "SocialGraph-FM Research",
    seed: 1729,
    preliminary: true,
    scenarios: [
      ["twitch-content-policy", "twitch-language", "research.content_policy_review", { kind: "nodes", nodeIds: ["0"] }],
      ["tolokers-account-risk", "tolokers", "research.account_risk_review", { kind: "nodes", nodeIds: ["0"] }],
      ["wiki-rfa-signed-relation", "wiki-rfa", "research.signed_relation_review", { kind: "directed-node-pairs", pairs: [["0", "1"]] }],
      ["email-eu-collaboration", "email-eu-core", "core.collaboration_completion", { kind: "collaboration-candidates", anchorNodeId: "0", topK: 20 }],
    ].map(([scenarioId, datasetId, taskId, defaultTargetScope]) => ({
      scenarioId,
      datasetId,
      title: String(scenarioId),
      taskId,
      graphVersionId: `research:${datasetId}`,
      graphVersionHash: null,
      modelVersionId: null,
      enabled: false,
      unavailableReason: "RESEARCH_MODEL_NOT_INSTALLED",
      defaultTargetScope,
      primaryMetric: null,
      scratchDelta: null,
    })),
  }, "scenariosHash");
}

describe("SocialGraph-FM Research GFM contracts", () => {
  it("accepts the isolated unavailable capability without promoting formal readiness", () => {
    const parsed = parseResearchCapabilities(capabilityPayload());
    expect(parsed.researchServingReady).toBe(false);
    expect(parsed.model).toBeNull();
    expect(parsed.seed).toBe(1729);
    expect(parsed.upload.compatibleTaskIds).toEqual(["core.collaboration_completion"]);
  });

  it("rejects capability mutation even when the declared capability hash is retained", () => {
    const payload = capabilityPayload();
    expect(() => parseResearchCapabilities({ ...payload, researchServingReady: true }))
      .toThrow("GFM_RESEARCH_CAPABILITIES_INVALID");
    expect(() => parseResearchCapabilities({ ...payload, privateCheckpointPath: "E:\\secret" }))
      .toThrow("GFM_RESEARCH_CAPABILITIES_INVALID");
  });

  it("requires the four scenario identities in canonical order and verifies scenariosHash", () => {
    expect(parseResearchScenarios(scenarioPayload()).scenarios).toHaveLength(4);
    const payload = scenarioPayload();
    expect(() => parseResearchScenarios({ ...payload, scenarios: [...payload.scenarios].reverse() }))
      .toThrow("GFM_RESEARCH_SCENARIOS_INVALID");
  });

  it("accepts only bounded hash-bound scenario graph projections", () => {
    const base = {
      schemaVersion: "socialgraph-fm.research/1.0",
      scenarioId: "twitch-content-policy",
      graphVersionId: "research:twitch-language",
      graphVersionHash: "1".repeat(64),
      modelVersionId: "socialgraph-fm-research/model",
      modelVersionHash: "2".repeat(64),
      nodes: [{ id: "0", label: "Node 0" }, { id: "1", label: "Node 1" }],
      edges: [{ id: "edge:0:1", source: "0", target: "1", directed: false }],
      partialPreview: true,
      nodeCount: 20,
      edgeCount: 40,
    };
    const valid = hashed(base, "previewHash");
    expect(parseResearchScenarioPreview(valid).nodes).toHaveLength(2);
    expect(() => parseResearchScenarioPreview({
      ...valid,
      edges: [{ id: "edge:0:missing", source: "0", target: "missing", directed: false }],
    })).toThrow("GFM_RESEARCH_PREVIEW_INVALID");
    expect(() => parseResearchScenarioPreview({ ...valid, nodeCount: 2, edgeCount: 1 }))
      .toThrow("GFM_RESEARCH_PREVIEW_INVALID");
  });

  it("rejects a result whose task/entity semantics or hash are changed", () => {
    const base = {
      schemaVersion: "socialgraph-fm.research/1.0",
      runId: "run-1",
      requestHash: "1".repeat(64),
      taskId: "research.content_policy_review",
      graphVersionId: "research:twitch-language",
      graphVersionHash: "2".repeat(64),
      modelVersionId: "socialgraph-fm-research/model",
      modelVersionHash: "3".repeat(64),
      seed: 1729,
      preliminary: true,
      calibrationStatus: "ranking_only",
      findings: [{
        id: "finding-1",
        rank: 1,
        entityType: "node",
        entityIds: ["0"],
        score: 0.72,
        scoreKind: "ranking-score",
        calibrated: false,
        reasonCodes: ["structure.shared-encoder"],
        limitations: ["Single-seed preliminary result."],
        reviewRequired: true,
      }],
      completedAt: "2026-08-16T00:00:00.000000Z",
    };
    const valid = hashed(base, "resultHash");
    expect(parseResearchRunResult(valid).findings[0]?.entityType).toBe("node");
    expect(() => parseResearchRunResult({ ...valid, taskId: "research.signed_relation_review" }))
      .toThrow("GFM_RESEARCH_RUN_RESULT_INVALID");
    expect(() => parseResearchRunResult({ ...valid, findings: [{ ...base.findings[0], score: 0.73 }] }))
      .toThrow("GFM_RESEARCH_RUN_RESULT_INVALID");

    const probabilityFinding = {
      ...base.findings[0],
      id: "finding-2",
      rank: 2,
      scoreKind: "probability",
      calibrated: true,
    };
    expect(() => parseResearchRunResult(hashed({
      ...base,
      findings: [base.findings[0], probabilityFinding],
    }, "resultHash"))).toThrow("GFM_RESEARCH_RUN_RESULT_INVALID");
    expect(() => parseResearchRunResult(hashed({
      ...base,
      calibrationStatus: "calibrated",
      findings: [],
    }, "resultHash"))).toThrow("GFM_RESEARCH_RUN_RESULT_INVALID");
  });
});
