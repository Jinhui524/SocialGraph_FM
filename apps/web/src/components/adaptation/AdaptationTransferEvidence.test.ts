import { describe, expect, it } from "vitest";

import { targetComparison, targetEvidence, targetPolicy, targetResult, targetTaskRegistration } from "../../test/fixtures/governanceTargetTask";
import { globalModelModelCard } from "../../test/fixtures/globalModel";
import type { GlobalModelModelCard } from "../../types/globalModel";
import type { TargetAdaptationComparison, GovernanceOnlineEvidence, GovernanceOnlineResult, TargetReviewPolicy, TargetTaskRegistration } from "../../types/governanceOnline";
import { buildAdaptationTransferEvidence, type AdaptationModelCardState } from "./AdaptationTransferEvidence";

function readyModelCard(result: GovernanceOnlineResult): AdaptationModelCardState {
  const card = globalModelModelCard();
  return {
    status: "ready",
    card: {
      ...card,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
      protocols: {
        ...card.protocols,
        global: {
          ...card.protocols.global,
          modelVersionId: result.modelVersionId,
          modelVersionHash: result.modelVersionHash,
          modelStateHash: result.modelStateHash,
        },
      },
    } as GlobalModelModelCard,
  };
}

describe("buildAdaptationTransferEvidence", () => {
  it("aggregates anonymous source routing, representation response, selection, and calibration", () => {
    const result = targetResult() as unknown as GovernanceOnlineResult;
    const state = buildAdaptationTransferEvidence({
      result,
      registration: targetTaskRegistration() as unknown as TargetTaskRegistration,
      modelCardState: readyModelCard(result),
      selectedNodeId: "target-node-001",
      selectedEvidence: targetEvidence("target-node-001") as unknown as GovernanceOnlineEvidence,
      policy: targetPolicy() as unknown as TargetReviewPolicy,
      comparison: targetComparison() as unknown as TargetAdaptationComparison,
    });

    expect(state.status).toBe("ready");
    if (state.status !== "ready") throw new Error(state.reason);
    const evidence = state.value;
    expect(evidence.experts).toHaveLength(7);
    expect(evidence.experts.reduce((sum, expert) => sum + expert.routingMass, 0)).toBeCloseTo(1, 8);
    expect(evidence.activeSourceCount).toBe(1);
    expect(evidence.primaryRoute).toMatchObject({ id: "source-04", label: "源域专家 04", coverage: 1, trainingNodeCount: 716 });
    expect(evidence.primaryRoute?.routingMass).toBeCloseTo(0.7, 8);
    expect(evidence.primaryRoute?.averageWeight).toBeCloseTo(0.7, 8);
    expect(evidence.experts.reduce((sum, expert) => sum + expert.averageWeight, 0)).toBeCloseTo(1, 8);
    expect(evidence.nullRoutingMass).toBeCloseTo(0.3, 8);
    expect(evidence.textResponseMean).toBeCloseTo(0.62, 8);
    expect(evidence.structureResponseMean).toBeCloseTo(0.38, 8);
    expect(evidence.selectedObject).toMatchObject({
      nodeId: "target-node-001",
      routes: [{ label: "源域专家 04", weight: 0.7 }, { label: "保守未知域", weight: 0.3 }],
      relationEvidenceCount: 1,
    });
    expect(evidence.calibration).toEqual({
      positiveCount: 8,
      negativeCount: 8,
      selectedLambda: 0.5,
      raisedCount: 54,
      loweredCount: 54,
      unchangedCount: 0,
      maxRankChange: 107,
    });
    expect(JSON.stringify(evidence)).not.toMatch(/china|cuba|iran|russia|UAE|venezuela/u);
  });

  it("fails closed while the model card is loading or when model identity differs", () => {
    const result = targetResult() as unknown as GovernanceOnlineResult;
    expect(buildAdaptationTransferEvidence({
      result,
      registration: targetTaskRegistration() as unknown as TargetTaskRegistration,
      modelCardState: { status: "loading", card: null },
    })).toEqual({ status: "unavailable", reason: "model_card_loading" });

    const modelCardState = readyModelCard(result);
    expect(buildAdaptationTransferEvidence({
      result: { ...result, modelStateHash: "f".repeat(64) },
      registration: targetTaskRegistration() as unknown as TargetTaskRegistration,
      modelCardState,
    })).toEqual({ status: "unavailable", reason: "identity_mismatch" });
  });

  it("rejects incomplete or non-normalized top-two routes", () => {
    const result = targetResult() as unknown as GovernanceOnlineResult;
    const malformed = {
      ...result,
      findings: result.findings.map((finding, index) => index ? finding : {
        ...finding,
        routes: [finding.routes[0], { ...finding.routes[1], weight: 0.4 }, finding.routes[2]],
      }),
    } as GovernanceOnlineResult;
    expect(buildAdaptationTransferEvidence({
      result: malformed,
      registration: targetTaskRegistration() as unknown as TargetTaskRegistration,
      modelCardState: readyModelCard(result),
    })).toEqual({ status: "unavailable", reason: "invalid_routes" });
  });
});
