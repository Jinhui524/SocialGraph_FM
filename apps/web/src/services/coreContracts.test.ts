import { describe, expect, it } from "vitest";

import vectors from "../../../../contracts/core-inference-vectors.json";
import type { CoreRunBinding } from "../types/core";
import { createValidatedCoreFixture } from "../test/fixtures/core";
import { sha256Canonical } from "./graphIdentity";
import {
  parseCoreCapabilities,
  parseCoreError,
  parseCoreRunRequest,
  parseCoreRunResult,
  parseCoreRunStatus,
  parseCoreFinding,
} from "./coreContracts";

describe("core GFM contract parsers", () => {
  it("consumes every neutral valid public-request vector and rejects every invalid one", () => {
    for (const item of vectors.validRunRequests) {
      expect(() => parseCoreRunRequest(item)).not.toThrow();
    }
    for (const item of vectors.invalidRunRequests) {
      expect(() => parseCoreRunRequest(item)).toThrow("GFM_CORE_RUN_REQUEST_INVALID");
    }
  });

  it("accepts exactly one non-empty risk target kind and rejects mixed or empty scopes", () => {
    const request = {
      schemaVersion: "socialgraph-fm.core-run-request/2.0",
      graphVersionId: "graph-v1",
      taskId: "core.risk_and_trust_review",
      targetScope: { kind: "risk-review", nodeIds: ["node-a"], edgeIds: [] },
      modelVersionId: "socialgraph-fm-core/review",
      parameters: { kind: "risk-and-trust", topKSimilarCases: 3 },
    };

    expect(parseCoreRunRequest(request).targetScope).toEqual(request.targetScope);
    expect(() => parseCoreRunRequest({
      ...request,
      targetScope: { kind: "risk-review", nodeIds: ["node-a"], edgeIds: ["edge-a"] },
    })).toThrow("GFM_CORE_RUN_REQUEST_INVALID");
    expect(() => parseCoreRunRequest({
      ...request,
      targetScope: { kind: "risk-review", nodeIds: [], edgeIds: [] },
    })).toThrow("GFM_CORE_RUN_REQUEST_INVALID");
  });

  it("consumes the neutral capability vector through the public schema and rejects contradictions", () => {
    const parsed = parseCoreCapabilities(vectors.validCapabilities[0]);

    expect(parsed.schemaVersion).toBe("socialgraph-fm.core-capabilities/2.0");
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(Object.isFrozen(parsed.models)).toBe(true);
    expect(() => parseCoreCapabilities(vectors.invalidCapabilities[0]))
      .toThrow("GFM_CORE_CAPABILITIES_INVALID");
  });

  it("accepts accepted-only and serving-ready registries only when readiness and tasks derive exactly", () => {
    const taskBindings = [
      {
        taskId: "core.risk_and_trust_review",
        entityType: "node",
        confidenceKind: "binary-calibration",
        calibrationVersion: "risk-node-calibration/1",
        method: "sigmoid",
        calibrationArtifactHash: "6".repeat(64),
        calibrationProtocolHash: "7".repeat(64),
        adapterDomain: "risk-node",
        adapterSchemaHash: "8".repeat(64),
        adapterStateHash: "9".repeat(64),
        featureContractHash: "a".repeat(64),
      },
      {
        taskId: "core.risk_and_trust_review",
        entityType: "edge",
        confidenceKind: "binary-calibration",
        calibrationVersion: "risk-edge-calibration/1",
        method: "sigmoid",
        calibrationArtifactHash: "b".repeat(64),
        calibrationProtocolHash: "c".repeat(64),
        adapterDomain: "risk-edge",
        adapterSchemaHash: "d".repeat(64),
        adapterStateHash: "e".repeat(64),
        featureContractHash: "f".repeat(64),
      },
    ];
    const graphFeatureContractHash = sha256Canonical(taskBindings.map((binding) => ({
      taskId: binding.taskId,
      entityType: binding.entityType,
      featureContractHash: binding.featureContractHash,
    })));
    const accepted = {
      schemaVersion: "socialgraph-fm.core-capabilities/2.0",
      registryHash: "1".repeat(64),
      registryGeneration: 7,
      controlHash: "2".repeat(64),
      controlGeneration: 7,
      catalogHash: "3".repeat(64),
      catalogGeneration: 7,
      servingReady: false,
      models: [{
        modelVersionId: "socialgraph-fm-core/review",
        modelVersionHash: "4".repeat(64),
        state: "accepted",
        tasks: ["core.risk_and_trust_review"],
        graphSchemaVersions: ["socialgraph-fm.core-graph-bundle/2.0"],
        graphFeatureContractHash,
        taskBindings,
        maxNodes: 100,
        maxEdges: 500,
      }],
      tasks: ["core.risk_and_trust_review"],
      readiness: { modelValidated: true, coreServingReady: false },
    };
    expect(parseCoreCapabilities(accepted).models[0]?.state).toBe("accepted");

    const serving = {
      ...accepted,
      servingReady: true,
      models: [{ ...accepted.models[0], state: "servingReady" }],
      readiness: { modelValidated: true, coreServingReady: true },
    };
    expect(parseCoreCapabilities(serving).servingReady).toBe(true);
    expect(() => parseCoreCapabilities({
      ...serving,
      readiness: { modelValidated: true, coreServingReady: false },
    })).toThrow("GFM_CORE_CAPABILITIES_INVALID");
    expect(() => parseCoreCapabilities({ ...serving, tasks: [] }))
      .toThrow("GFM_CORE_CAPABILITIES_INVALID");
    expect(() => parseCoreCapabilities({
      ...serving,
      models: [{ ...serving.models[0], taskBindings: [...taskBindings].reverse() }],
    })).toThrow("GFM_CORE_CAPABILITIES_INVALID");
    expect(() => parseCoreCapabilities({
      ...serving,
      models: [{ ...serving.models[0], taskBindings: taskBindings.slice(0, 1) }],
    })).toThrow("GFM_CORE_CAPABILITIES_INVALID");
  });

  it("parses community uncertainty as a residual-coverage interval and rejects probability substitution", () => {
    const fixture = createValidatedCoreFixture({
      graphVersionId: "graph-v1",
      taskId: "core.community_resilience_review",
      findingType: "community-resilience-candidate",
      entityType: "community",
      subjectIds: ["community-a"],
    });

    expect(fixture.finding.calibratedConfidence).toMatchObject({
      schemaVersion: "socialgraph-fm.core-regression-confidence-interval/1.0",
      method: "validation-residual-interval",
      coverage: 0.9,
      validationCount: 32,
    });
    const probabilityPayload = {
      schemaVersion: "socialgraph-fm.core-calibrated-confidence/2.0",
      value: 0.6,
      scoreHash: fixture.finding.score.scoreHash,
      taskId: fixture.finding.taskId,
      entityType: fixture.finding.score.entityType,
      entityIds: fixture.finding.score.entityIds,
      graphVersionHash: fixture.finding.graphVersionHash,
      modelVersion: fixture.finding.modelVersion,
      modelVersionHash: fixture.finding.modelVersionHash,
      calibrationVersion: "calibration/1",
      method: "sigmoid",
      calibrationArtifactHash: "b".repeat(64),
      calibrationProtocolHash: "c".repeat(64),
    };
    const probability = {
      ...probabilityPayload,
      confidenceHash: sha256Canonical(probabilityPayload),
    };
    const findingPayload = {
      ...Object.fromEntries(Object.entries(fixture.finding).filter(([key]) => key !== "findingHash")),
      calibratedConfidence: probability,
    };

    expect(() => parseCoreFinding({
      ...findingPayload,
      findingHash: sha256Canonical(findingPayload),
    })).toThrow("GFM_CORE_FINDING_INVALID");
  });

  it("rejects a coherently rehashed resilience interval for a different point estimate", () => {
    const fixture = createValidatedCoreFixture({
      graphVersionId: "graph-v1",
      taskId: "core.community_resilience_review",
      findingType: "community-resilience-candidate",
      entityType: "community",
      subjectIds: ["community-a"],
    });
    const confidencePayload = {
      ...Object.fromEntries(Object.entries(fixture.finding.calibratedConfidence)
        .filter(([key]) => key !== "confidenceHash")),
      pointEstimate: fixture.finding.score.score + 0.05,
    };
    const findingPayload = {
      ...Object.fromEntries(Object.entries(fixture.finding)
        .filter(([key]) => key !== "findingHash")),
      calibratedConfidence: {
        ...confidencePayload,
        confidenceHash: sha256Canonical(confidencePayload),
      },
    };

    expect(() => parseCoreFinding({
      ...findingPayload,
      findingHash: sha256Canonical(findingPayload),
    })).toThrow("GFM_CORE_FINDING_INVALID");
  });

  it("validates neutral status hashes and status/progress/error combinations", () => {
    const parsed = parseCoreRunStatus(vectors.validStatuses[0]);
    expect(parsed.stateHash).toBe(vectors.validStatuses[0].stateHash);
    expect(Object.isFrozen(parsed)).toBe(true);
    for (const item of vectors.invalidStatuses) {
      expect(() => parseCoreRunStatus(item)).toThrow("GFM_CORE_RUN_STATUS_INVALID");
    }
    expect(() => parseCoreRunStatus({ ...vectors.validStatuses[0], unexpected: true }))
      .toThrow("GFM_CORE_RUN_STATUS_INVALID");
  });

  it("rejects a canonically rehashed failed status carrying an unregistered upstream error string", () => {
    const withoutStateHash = {
      ...Object.fromEntries(
        Object.entries(vectors.validStatuses[0]).filter(([key]) => key !== "stateHash"),
      ),
      status: "failed",
      progress: 100,
      errorCode: "C:\\private\\checkpoint.pt",
    };
    const unsafe = {
      ...withoutStateHash,
      stateHash: sha256Canonical(withoutStateHash),
    };

    expect(() => parseCoreRunStatus(unsafe)).toThrow("GFM_CORE_RUN_STATUS_INVALID");
  });

  it("validates the neutral finding and rejects nested hash or canonical-evidence mutations", () => {
    const original = vectors.validFindings[0];
    const parsed = parseCoreFinding(original);
    expect(parsed.findingHash).toBe(original.findingHash);
    const parsedEvidence = parsed.evidence[0]!;
    expect(parsedEvidence).not.toHaveProperty("value");
    expect(sha256Canonical(Object.fromEntries(
      Object.entries(parsedEvidence).filter(([key]) => key !== "evidenceHash"),
    ))).toBe(parsedEvidence.evidenceHash);
    expect(() => parseCoreFinding({
      ...original,
      score: { ...original.score, score: original.score.score + 0.1 },
    })).toThrow("GFM_CORE_FINDING_INVALID");
    expect(() => parseCoreFinding({
      ...original,
      evidence: [{ ...original.evidence[0], valueCanonicalJson: "{ }" }],
    })).toThrow("GFM_CORE_FINDING_INVALID");
    for (const item of vectors.invalidFindings) {
      expect(() => parseCoreFinding(item)).toThrow("GFM_CORE_FINDING_INVALID");
    }
  });

  it("validates the neutral result and rejects coherently rehashed identity substitution against a binding", () => {
    const original = vectors.validResults[0];
    const binding = {
      runId: original.runId,
      publicRequestHash: "9".repeat(64),
      serverRequestHash: original.requestHash,
      taskId: original.taskId,
      graphVersionId: original.graphVersionId,
      modelVersionId: original.modelVersionId,
    } as CoreRunBinding;
    const parsed = parseCoreRunResult(original, binding);
    expect(parsed.resultHash).toBe(original.resultHash);
    expect(Object.isFrozen(parsed)).toBe(true);
    expect(sha256Canonical(Object.fromEntries(
      Object.entries(parsed).filter(([key]) => key !== "resultHash"),
    ))).toBe(parsed.resultHash);

    const substitutedWithoutHash = { ...original, graphVersionId: "graph-other" };
    const substituted = {
      ...substitutedWithoutHash,
      resultHash: sha256Canonical(Object.fromEntries(
        Object.entries(substitutedWithoutHash).filter(([key]) => key !== "resultHash"),
      )),
    };
    expect(() => parseCoreRunResult(substituted, binding))
      .toThrow("GFM_CORE_RESPONSE_BINDING_INVALID");
    for (const item of vectors.invalidResults) {
      expect(() => parseCoreRunResult(item)).toThrow("GFM_CORE_RUN_RESULT_INVALID");
    }
  });

  it("accepts only closed stable error envelopes from the neutral and public forms", () => {
    for (const item of vectors.validErrors) {
      expect(parseCoreError(item)).toEqual({ code: "GFM_CORE_NOT_FOUND" });
    }
    expect(parseCoreError({ detail: { code: "GFM_CORE_MODEL_NOT_INSTALLED" } }))
      .toEqual({ code: "GFM_CORE_MODEL_NOT_INSTALLED" });
    for (const item of vectors.invalidErrors) {
      expect(() => parseCoreError(item)).toThrow("GFM_CORE_ERROR_INVALID");
    }
    expect(() => parseCoreError({ detail: { code: "GFM_CORE_NOT_FOUND", message: "private" } }))
      .toThrow("GFM_CORE_ERROR_INVALID");
    expect(() => parseCoreError({ detail: { code: "CHECKPOINT_C_USERS_PRIVATE_SECRET" } }))
      .toThrow("GFM_CORE_ERROR_INVALID");
  });
});
