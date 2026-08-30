import { describe, expect, it, vi } from "vitest";

import {
  GLOBAL_MODEL_TEST_HASHES,
  GLOBAL_MODEL_TEST_PROTOCOL_MODELS,
  GLOBAL_MODEL_TEST_RUN_ID,
  globalModelCapabilities,
  globalModelEvidence,
  globalModelHealth,
  globalModelModelCard,
  globalModelPreview,
  globalModelRequest,
  globalModelResult,
  globalModelReview,
  globalModelScenario,
  globalModelStatus,
} from "../test/fixtures/globalModel";
import { GlobalModelClient } from "./globalModelClient";
import { sha256Canonical } from "./graphIdentity";

function response(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("SocialGraph-FM Global browser client", () => {
  it("uses only the versioned public route inventory and preserves bindings", async () => {
    const request = globalModelRequest("low_label");
    const requestHash = sha256Canonical(request);
    const reviewReason = "转交治理人员核验协同上下文。";
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(globalModelHealth()))
      .mockResolvedValueOnce(response(globalModelCapabilities()))
      .mockResolvedValueOnce(response(globalModelModelCard()))
      .mockResolvedValueOnce(response(globalModelScenario()))
      .mockResolvedValueOnce(response(globalModelPreview()))
      .mockResolvedValueOnce(response(globalModelStatus(requestHash), 202))
      .mockResolvedValueOnce(response(globalModelResult(request)))
      .mockResolvedValueOnce(response(globalModelEvidence("ru-001", request)))
      .mockResolvedValueOnce(response(globalModelReview("confirmed", reviewReason)));
    const client = new GlobalModelClient("http://api.test/api/v1/gfm/global-model", fetcher as unknown as typeof fetch);

    await client.health();
    await client.capabilities();
    await client.modelCard();
    await client.scenario();
    await client.scenarioPreview();
    const created = await client.createRun(request, {
      graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
      modelVersionHash: GLOBAL_MODEL_TEST_PROTOCOL_MODELS.low_label.modelVersionHash,
    });
    const result = await client.getResult(created.binding.runId, created.binding);
    await client.nodeEvidence(created.binding.runId, "ru-001", created.binding);
    await client.submitReview(created.binding.runId, {
      schemaVersion: "socialgraph-fm.global-model/1.0",
      nodeId: "ru-001",
      decision: "confirmed",
      reason: reviewReason,
    }, created.binding);

    expect(result.protocol).toBe("low_label");
    expect(created.binding).toMatchObject({
      runId: GLOBAL_MODEL_TEST_RUN_ID,
      graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
      modelVersionHash: GLOBAL_MODEL_TEST_PROTOCOL_MODELS.low_label.modelVersionHash,
      publicRequestHash: requestHash,
      serverRequestHash: requestHash,
    });
    expect(fetcher.mock.calls.map((call) => call[0])).toEqual([
      "http://api.test/api/v1/gfm/global-model/health",
      "http://api.test/api/v1/gfm/global-model/capabilities",
      "http://api.test/api/v1/gfm/global-model/model-card",
      "http://api.test/api/v1/gfm/global-model/scenario",
      "http://api.test/api/v1/gfm/global-model/scenario/graph-preview",
      "http://api.test/api/v1/gfm/global-model/runs",
      `http://api.test/api/v1/gfm/global-model/runs/${GLOBAL_MODEL_TEST_RUN_ID}/result`,
      `http://api.test/api/v1/gfm/global-model/runs/${GLOBAL_MODEL_TEST_RUN_ID}/nodes/ru-001/evidence`,
      `http://api.test/api/v1/gfm/global-model/runs/${GLOBAL_MODEL_TEST_RUN_ID}/reviews`,
    ]);
  });

  it("rejects unsafe path identifiers before issuing a request", async () => {
    const client = new GlobalModelClient("http://api.test/api/v1/gfm/global-model", vi.fn() as unknown as typeof fetch);
    const binding = {
      runId: GLOBAL_MODEL_TEST_RUN_ID,
      publicRequestHash: "1".repeat(64),
      serverRequestHash: "1".repeat(64),
      taskId: "coordination_risk",
      protocol: "in_domain",
      datasetVersionId: "socialgraph-fm:russia",
      graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
      modelVersionId: GLOBAL_MODEL_TEST_PROTOCOL_MODELS.in_domain.modelVersionId,
      modelVersionHash: GLOBAL_MODEL_TEST_PROTOCOL_MODELS.in_domain.modelVersionHash,
    } as const;

    await expect(client.nodeEvidence(binding.runId, "../private", binding))
      .rejects.toThrow("GFM_GLOBAL_MODEL_NODE_ID_INVALID");
  });
});
