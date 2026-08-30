import { describe, expect, it } from "vitest";

import type { CoreModelCapability } from "../types/core";
import { createGraphVersion } from "./graphImport";
import {
  assessCoreCompatibility,
  buildCollaborationRunRequest,
  buildCommunityRunRequest,
  buildRiskRunRequest,
  buildCoreRunRequest,
  inspectCollaborationPair,
} from "./coreCompatibility";
import { registeredEdgeIdentityForLocalId } from "./coreEdgeIdentity";
import { sha256Canonical } from "./graphIdentity";

const graph = createGraphVersion("core-social.json", [
  { id: "community-a", label: "Community A", type: "community", attributes: {} },
  { id: "actor-a", label: "Actor A", type: "person", attributes: {} },
  { id: "actor-b", label: "Actor B", type: "person", attributes: {} },
  { id: "actor-c", label: "Actor C", type: "person", attributes: {} },
], [
  { id: "membership-a", source: "actor-a", target: "community-a", type: "member_of", weight: 1, directed: true, attributes: {} },
  { id: "collaboration-ab", source: "actor-a", target: "actor-b", type: "collaborates", weight: 1, directed: true, attributes: {} },
]);

const taskBindings: CoreModelCapability["taskBindings"] = [
  {
    taskId: "core.community_resilience_review", entityType: "community",
    confidenceKind: "regression-interval", calibrationVersion: "community/1",
    method: "validation-residual-interval", calibrationArtifactHash: "2".repeat(64),
    calibrationProtocolHash: "3".repeat(64), adapterDomain: "community",
    adapterSchemaHash: "4".repeat(64), adapterStateHash: "5".repeat(64),
    featureContractHash: "6".repeat(64),
  },
  ...(["node", "edge"] as const).map((entityType, index) => ({
    taskId: "core.risk_and_trust_review" as const, entityType,
    confidenceKind: "binary-calibration" as const, calibrationVersion: `risk-${entityType}/1`,
    method: "sigmoid" as const, calibrationArtifactHash: (index + 7).toString().repeat(64),
    calibrationProtocolHash: (index + 3).toString().repeat(64), adapterDomain: `risk-${entityType}`,
    adapterSchemaHash: "a".repeat(64), adapterStateHash: "b".repeat(64),
    featureContractHash: (index + 8).toString().repeat(64),
  })),
  {
    taskId: "core.collaboration_completion", entityType: "node-pair",
    confidenceKind: "binary-calibration", calibrationVersion: "collaboration/1",
    method: "sigmoid", calibrationArtifactHash: "c".repeat(64),
    calibrationProtocolHash: "d".repeat(64), adapterDomain: "collaboration",
    adapterSchemaHash: "e".repeat(64), adapterStateHash: "f".repeat(64),
    featureContractHash: "1".repeat(64),
  },
];

const servingModel: CoreModelCapability = {
  modelVersionId: "socialgraph-fm-core/review",
  modelVersionHash: "1".repeat(64),
  state: "servingReady",
  tasks: [
    "core.community_resilience_review",
    "core.risk_and_trust_review",
    "core.collaboration_completion",
  ],
  graphSchemaVersions: ["socialgraph-fm.core-graph-bundle/2.0"],
  graphFeatureContractHash: sha256Canonical(taskBindings.map((binding) => ({
    taskId: binding.taskId,
    entityType: binding.entityType,
    featureContractHash: binding.featureContractHash,
  }))),
  taskBindings,
  maxNodes: 100,
  maxEdges: 500,
};

describe("core GFM compatibility", () => {
  it("marks a serving Core model as requiring authoritative POST validation", () => {
    expect(assessCoreCompatibility(
      graph,
      servingModel,
      "core.risk_and_trust_review",
    )).toMatchObject({
      state: "server-validation-required",
      runnable: true,
      authoritativeAtPost: true,
      blockers: [],
    });
  });

  it("shows an accepted model as a validated candidate but never runnable", () => {
    expect(assessCoreCompatibility(
      graph,
      { ...servingModel, state: "accepted" },
      "core.risk_and_trust_review",
    )).toMatchObject({ state: "candidate", runnable: false, modelValidated: true });
  });

  it.each([
    ["unsupported task", { ...servingModel, tasks: [] }],
    ["wrong graph schema", { ...servingModel, graphSchemaVersions: ["legacy/1.0"] }],
    ["node cap", { ...servingModel, maxNodes: 2 }],
    ["edge cap", { ...servingModel, maxEdges: 1 }],
  ])("blocks %s before submission", (_label, model) => {
    expect(assessCoreCompatibility(graph, model, "core.risk_and_trust_review"))
      .toMatchObject({ state: "blocked", runnable: false });
  });

  it("blocks fact projections but does not confuse a bounded UI preview with missing graph facts", () => {
    const projected = {
      ...graph,
      datasetArtifact: {
        id: "artifact-1",
        datasetName: "projection",
        checksum: "checksum",
        canonicalGraphHash: "hash",
        scope: "projection" as const,
      },
    };
    expect(assessCoreCompatibility(projected, servingModel, "core.risk_and_trust_review"))
      .toMatchObject({ state: "blocked", runnable: false });
    expect(assessCoreCompatibility(
      { ...graph, truncated: true, preview: { ...graph.preview, truncated: true } },
      servingModel,
      "core.risk_and_trust_review",
    )).toMatchObject({ state: "server-validation-required", runnable: true });
    expect(assessCoreCompatibility(
      { ...graph, summary: { ...graph.summary, nodeCount: graph.summary.nodeCount + 1 } },
      servingModel,
      "core.risk_and_trust_review",
    )).toMatchObject({ state: "blocked", runnable: false });
  });
});

describe("core exact target builders", () => {
  it("builds community scope only from an existing explicit community fact", () => {
    expect(buildCommunityRunRequest(graph, servingModel.modelVersionId, "community-a", 3)).toEqual({
      schemaVersion: "socialgraph-fm.core-run-request/2.0",
      graphVersionId: graph.id,
      taskId: "core.community_resilience_review",
      targetScope: { kind: "community", communityIds: ["community-a"] },
      modelVersionId: servingModel.modelVersionId,
      parameters: { kind: "community-resilience", topKSimilarCases: 3 },
    });
    expect(() => buildCommunityRunRequest(graph, servingModel.modelVersionId, "community-1", 3))
      .toThrow("GFM_CORE_TARGET_NOT_FOUND");
    expect(() => buildCommunityRunRequest(graph, servingModel.modelVersionId, "actor-a", 3))
      .toThrow("GFM_CORE_TARGET_NOT_REGISTERED_COMMUNITY");
  });

  it("builds risk scope only from an existing node or existing edge ID", () => {
    expect(buildRiskRunRequest(graph, servingModel.modelVersionId, { kind: "node", nodeId: "actor-a" }, 4))
      .toMatchObject({ targetScope: { kind: "risk-review", nodeIds: ["actor-a"], edgeIds: [] } });
    expect(buildRiskRunRequest(graph, servingModel.modelVersionId, { kind: "edge", edgeId: "collaboration-ab" }, 4))
      .toMatchObject({
        targetScope: {
          kind: "risk-review",
          nodeIds: [],
          edgeIds: [registeredEdgeIdentityForLocalId(graph, "collaboration-ab").edgeHash],
        },
      });
    expect(() => buildRiskRunRequest(graph, servingModel.modelVersionId, { kind: "edge", edgeId: "missing" }, 4))
      .toThrow("GFM_CORE_TARGET_NOT_FOUND");
  });

  it("builds only a missing directed relation and blocks an already-recorded pair without changing facts", () => {
    const before = JSON.stringify(graph);
    const missing = buildCollaborationRunRequest(graph, servingModel.modelVersionId, "actor-a", "actor-c", 2, 50);
    expect(missing).toMatchObject({
      relationState: "missing",
      existingEdgeIds: [],
      request: {
        schemaVersion: "socialgraph-fm.core-run-request/2.0",
        graphVersionId: graph.id,
        taskId: "core.collaboration_completion",
        targetScope: { kind: "node-pairs", pairs: [["actor-a", "actor-c"]] },
        modelVersionId: servingModel.modelVersionId,
        parameters: { kind: "collaboration-completion", topKSimilarCases: 2, candidateLimit: 50 },
      },
    });
    expect(inspectCollaborationPair(graph, "actor-a", "actor-b"))
      .toEqual({ relationState: "recorded", existingEdgeIds: ["collaboration-ab"] });
    expect(() => buildCollaborationRunRequest(graph, servingModel.modelVersionId, "actor-a", "actor-b", 2, 50))
      .toThrow("GFM_CORE_COLLABORATION_RELATION_RECORDED");
    expect(inspectCollaborationPair(graph, "actor-b", "actor-a"))
      .toEqual({ relationState: "missing", existingEdgeIds: [] });
    expect(JSON.stringify(graph)).toBe(before);
    expect(() => buildCollaborationRunRequest(graph, servingModel.modelVersionId, "actor-a", "actor-a", 2, 50))
      .toThrow("GFM_CORE_TARGET_SELF_PAIR");
    expect(() => buildCollaborationRunRequest(graph, servingModel.modelVersionId, "actor-a", "missing", 2, 50))
      .toThrow("GFM_CORE_TARGET_NOT_FOUND");
  });

  it("treats either orientation as recorded only for an explicitly undirected graph", () => {
    const undirected = createGraphVersion("undirected.json", graph.nodes, [
      { id: "ab", source: "actor-a", target: "actor-b", type: "collaborates", weight: 1, directed: false, attributes: {} },
    ]);

    expect(inspectCollaborationPair(undirected, "actor-b", "actor-a"))
      .toEqual({ relationState: "recorded", existingEdgeIds: ["ab"] });
    expect(() => buildCollaborationRunRequest(
      undirected,
      servingModel.modelVersionId,
      "actor-b",
      "actor-a",
      2,
      50,
    )).toThrow("GFM_CORE_COLLABORATION_RELATION_RECORDED");
  });

  it("rejects a task/selection mismatch before a request can be fetched", () => {
    expect(() => buildCoreRunRequest({
      graph,
      modelVersionId: servingModel.modelVersionId,
      taskId: "core.community_resilience_review",
      selection: { kind: "node", nodeId: "actor-a" },
      topKSimilarCases: 3,
    })).toThrow("GFM_CORE_TASK_TARGET_MISMATCH");
  });
});
