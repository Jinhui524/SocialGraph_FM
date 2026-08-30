import { describe, expect, it } from "vitest";

import {
  GLOBAL_MODEL_TEST_HASHES,
  GLOBAL_MODEL_TEST_PROTOCOL_MODELS,
  globalModelCapabilities,
  globalModelEvidence,
  globalModelHealth,
  globalModelModelCard,
  globalModelRequest,
  globalModelResult,
  globalModelReview,
  globalModelScenario,
} from "../test/fixtures/globalModel";
import {
  parseGlobalModelCapabilities,
  parseGlobalModelHealth,
  parseGlobalModelModelCard,
  parseGlobalModelNodeEvidence,
  parseGlobalModelReviewRecord,
  parseGlobalModelRunResult,
  parseGlobalModelScenario,
} from "./globalModelContracts";
import { sha256Canonical } from "./graphIdentity";

describe("SocialGraph-FM Global public contracts", () => {
  it("accepts the canonical protocol inventory and Russia scenario metrics", () => {
    const capabilities = parseGlobalModelCapabilities(globalModelCapabilities());
    const health = parseGlobalModelHealth(globalModelHealth());
    const modelCard = parseGlobalModelModelCard(globalModelModelCard());
    const scenario = parseGlobalModelScenario(globalModelScenario());

    expect(capabilities.model?.protocols).toEqual(["in_domain", "low_label", "cross_domain", "global"]);
    expect(capabilities.model?.protocolModels.cross_domain).toEqual(GLOBAL_MODEL_TEST_PROTOCOL_MODELS.cross_domain);
    expect(capabilities.model?.protocolModels.global.modelVersionId).toBe(capabilities.model?.modelVersionId);
    expect(scenario.metrics.low_label).toMatchObject({ macroF1: 0.781, labelledTrainNodes: 16 });
    expect(health.modelVersionHash).toBe(capabilities.model?.modelVersionHash);
    expect(modelCard.modelVersionId).toBe(capabilities.model?.modelVersionId);
    expect(modelCard.licenses.map((item) => item.license)).toEqual(["CC-BY-4.0", "MIT"]);
    expect(Object.isFrozen(capabilities)).toBe(true);
    expect(Object.isFrozen(scenario.metrics)).toBe(true);
  });

  it("rejects a mutated capability even when its declared hash is retained", () => {
    const payload = globalModelCapabilities();
    expect(() => parseGlobalModelCapabilities({
      ...payload,
      model: { ...payload.model!, corpusHash: "9".repeat(64) },
    })).toThrow("GFM_GLOBAL_MODEL_RESPONSE_INVALID");
    expect(() => parseGlobalModelCapabilities({ ...payload, checkpointPath: "E:\\private\\model.pt" }))
      .toThrow("GFM_GLOBAL_MODEL_RESPONSE_INVALID");
  });

  it("rejects protocol inventories that reuse the Global model identity", () => {
    const payload = globalModelCapabilities();
    const body = {
      ...payload,
      model: {
        ...payload.model!,
        protocolModels: {
          ...payload.model!.protocolModels,
          in_domain: {
            ...payload.model!.protocolModels.global,
            state: "frozenDemo" as const,
          },
        },
      },
    };
    const unsigned = Object.fromEntries(
      Object.entries(body).filter(([key]) => key !== "capabilityHash"),
    );

    expect(() => parseGlobalModelCapabilities({
      ...unsigned,
      capabilityHash: sha256Canonical(unsigned),
    })).toThrow("GFM_GLOBAL_MODEL_RESPONSE_INVALID");
  });

  it("binds results to run, protocol, graph and model identities", () => {
    const request = globalModelRequest("cross_domain");
    const result = globalModelResult(request);
    const binding = {
      runId: result.runId,
      publicRequestHash: sha256Canonical(request),
      serverRequestHash: result.requestHash,
      taskId: request.taskId,
      protocol: request.protocol,
      datasetVersionId: request.datasetVersionId,
      graphVersionHash: GLOBAL_MODEL_TEST_HASHES.graph,
      modelVersionId: request.modelVersionId,
      modelVersionHash: GLOBAL_MODEL_TEST_PROTOCOL_MODELS.cross_domain.modelVersionHash,
    } as const;

    expect(parseGlobalModelRunResult(result, binding).findings[0]).toMatchObject({
      nodeId: "ru-001",
      riskBand: "high",
      routes: [{ expert: "shared", weight: 0.68 }, { expert: "russia", weight: 0.32 }],
    });
    expect(() => parseGlobalModelRunResult({ ...result, protocol: "in_domain" }, binding))
      .toThrow("GFM_GLOBAL_MODEL_RESPONSE_INVALID");
  });

  it("validates evidence and review records against the selected run and node", () => {
    const result = globalModelResult();
    const binding = { runId: result.runId };
    const evidence = parseGlobalModelNodeEvidence(
      globalModelEvidence(), binding, "ru-001", result,
    );
    expect(evidence.neighbors).toHaveLength(2);
    expect(evidence.evidenceSubgraph.nodes.some((item) => item.hop === 2)).toBe(true);
    expect(evidence.resultHash).toBe(result.resultHash);
    expect(() => parseGlobalModelNodeEvidence(globalModelEvidence(), binding, "ru-002"))
      .toThrow("GFM_GLOBAL_MODEL_RESPONSE_INVALID");

    const stale = { ...globalModelEvidence(), modelVersionHash: "0".repeat(64) };
    expect(() => parseGlobalModelNodeEvidence(stale, binding, "ru-001", result))
      .toThrow("GFM_GLOBAL_MODEL_RESPONSE_INVALID");

    const review = globalModelReview("confirmed", "协同行为证据需进入案件复核。", "ru-001");
    expect(parseGlobalModelReviewRecord(review, binding, {
      schemaVersion: "socialgraph-fm.global-model/1.0",
      nodeId: "ru-001",
      decision: "confirmed",
      reason: "协同行为证据需进入案件复核。",
    }).decision).toBe("confirmed");
  });
});
