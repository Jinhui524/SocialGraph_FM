import { describe, expect, it, vi } from "vitest";

import type {
  ResearchRunBinding,
  ResearchRunRequest,
  ResearchRunResult,
  ResearchRunStatus,
} from "../types/research";
import { SocialGraphApiError } from "./apiClient";
import { ResearchRunController } from "./researchRunController";

const HASH = "1".repeat(64);

function fixtures() {
  const request: ResearchRunRequest = {
    schemaVersion: "socialgraph-fm.research/1.0",
    graphVersionId: "uploaded-graph",
    taskId: "core.collaboration_completion",
    modelVersionId: "socialgraph-fm-research/model",
    targetScope: { kind: "collaboration-candidates", anchorNodeId: "n0", topK: 5 },
    parameters: { candidateLimit: 10 },
  };
  const binding: ResearchRunBinding = {
    runId: "run-1",
    publicRequestHash: HASH,
    serverRequestHash: "2".repeat(64),
    graphVersionId: request.graphVersionId,
    modelVersionId: request.modelVersionId,
    taskId: request.taskId,
  };
  const status: ResearchRunStatus = {
    schemaVersion: request.schemaVersion,
    runId: binding.runId,
    requestHash: binding.serverRequestHash,
    status: "succeeded",
    progress: 100,
    createdAt: "2026-08-16T00:00:00Z",
    updatedAt: "2026-08-16T00:00:01Z",
    errorCode: null,
    stateHash: "3".repeat(64),
  };
  const result: ResearchRunResult = {
    schemaVersion: request.schemaVersion,
    runId: binding.runId,
    requestHash: binding.serverRequestHash,
    taskId: request.taskId,
    graphVersionId: request.graphVersionId,
    graphVersionHash: "4".repeat(64),
    modelVersionId: request.modelVersionId,
    modelVersionHash: "5".repeat(64),
    seed: 1729,
    preliminary: true,
    calibrationStatus: "ranking_only",
    findings: [],
    completedAt: "2026-08-16T00:00:01Z",
    resultHash: "6".repeat(64),
  };
  return { request, binding, status, result };
}

describe("ResearchRunController registration", () => {
  it("waits through bounded pending registration responses before creating the run", async () => {
    const { request, binding, status, result } = fixtures();
    const client = {
      createRun: vi.fn()
        .mockRejectedValueOnce(new SocialGraphApiError(
          "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING",
          "pending",
          503,
        ))
        .mockRejectedValueOnce(new SocialGraphApiError(
          "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING",
          "pending",
          503,
        ))
        .mockResolvedValue({ status, binding }),
      getRun: vi.fn(),
      getResult: vi.fn().mockResolvedValue(result),
    };
    const controller = new ResearchRunController(client, {
      initialPollMs: 1,
      maxPollMs: 1,
      maxRegistrationAttempts: 3,
    });

    await controller.start(request);

    expect(client.createRun).toHaveBeenCalledTimes(3);
    expect(client.getResult).toHaveBeenCalledWith(binding.runId, binding, expect.any(AbortSignal));
    expect(controller.getState()).toMatchObject({ phase: "succeeded", result });
    controller.dispose();
  });
});
