import { describe, expect, it } from "vitest";

import { adaptationComparison, targetLabelRecipe, targetLabelSet, targetPackageReceipt, targetReviewPolicy } from "../test/fixtures/governanceAdaptation";
import { canonicalJson, sha256Canonical } from "./graphIdentity";
import * as onlineContractsModule from "./governanceOnlineContracts";
import {
  comparisonPayload,
  casePayload,
  artifactCompatibility,
  derivationPage,
  evidencePayload,
  findingPage,
  onlineArtifact,
  onlineCapabilities,
  onlineHealth,
  onlinePreview,
  onlineResult,
  onlineRun,
  onlineRunPreview,
} from "../test/fixtures/governanceOnline";
import {
  parseGovernanceArtifact,
  parseGovernanceArtifactCompatibility,
  parseGovernanceDerivationPage,
  parseCoreFindingPage,
  parseGovernanceOnlineCapabilities,
  parseGovernanceOnlineHealth,
  parseGovernanceOnlineEvidence,
  parseGovernanceOnlinePreview,
  parseGovernanceOnlineResult,
  parseGovernanceOnlineRun,
  parseGovernanceOnlineRunRequest,
  parseGovernanceRunComparison,
  parseAdaptationComparison,
  parseTargetLabelRecipe,
  parseTargetLabelSet,
  parseAdaptationReviewPolicy,
  parseTargetTaskRegistration,
  parseTargetReviewPolicy,
  parseTargetAdaptationComparison,
  parseAdaptationGovernanceHandoff,
  parseAdaptationOverlayActivation,
  parseTargetReviewCollection,
} from "./governanceOnlineContracts";
import { GOVERNANCE_PUBLIC_SKILLS } from "../types/governanceSkills";
import { adaptationHandoff, targetActivation, targetComparison, targetFixtureIdentity, targetPolicy, targetPreview, targetResult, targetRun, targetTaskRegistration } from "../test/fixtures/governanceTargetTask";

describe("SocialGraph-FM Governance Global online contracts", () => {
  it("parses target-task registration and frozen adaptation handoff contracts", () => {
    expect(parseTargetTaskRegistration(targetTaskRegistration()).task).toMatchObject({ mode: "few_shot", nodeCount: 108 });
    expect(parseTargetReviewPolicy(targetPolicy())).toMatchObject({ positiveCount: 8, negativeCount: 8, baseOutputsImmutable: true });
    expect(parseTargetAdaptationComparison(targetComparison()).rows).toHaveLength(108);
    expect(parseAdaptationGovernanceHandoff(adaptationHandoff())).toMatchObject({ baseModelMutation: false, decision: "pending_governance_review" });
  });

  it("keeps lane-bound identities disjoint while sharing the frozen implementation identity", () => {
    const zeroIdentity = targetFixtureIdentity("zero_shot");
    const fewIdentity = targetFixtureIdentity("few_shot");
    const sharedImplementationFields = new Set(["recipeHash", "codeHash"]);
    for (const [field, value] of Object.entries(zeroIdentity)) {
      if (sharedImplementationFields.has(field)) expect(value, field).toBe(fewIdentity[field as keyof typeof fewIdentity]);
      else expect(value, field).not.toBe(fewIdentity[field as keyof typeof fewIdentity]);
    }

    const zeroRegistration = parseTargetTaskRegistration(targetTaskRegistration("zero_shot"));
    const fewRegistration = parseTargetTaskRegistration(targetTaskRegistration("few_shot"));
    const zeroRun = parseGovernanceOnlineRun(targetRun("zero_shot"));
    const fewRun = parseGovernanceOnlineRun(targetRun("few_shot"));
    const zeroPolicy = parseTargetReviewPolicy(targetPolicy("zero_shot"));
    const fewPolicy = parseTargetReviewPolicy(targetPolicy("few_shot"));
    expect(zeroRun).toMatchObject({ modelVersionId: fewRun.modelVersionId, modelVersionHash: fewRun.modelVersionHash, modelStateHash: fewRun.modelStateHash });
    expect(zeroPolicy.binding).toMatchObject({ recipeHash: fewPolicy.binding.recipeHash, codeHash: fewPolicy.binding.codeHash, seed: fewPolicy.binding.seed });
    expect(zeroRegistration.labels).toBeNull();
    expect(fewRegistration.labels).not.toBeNull();
    expect([zeroRun.artifactId, zeroRun.datasetContentHash, zeroRun.graphVersionHash]).toEqual([
      zeroRegistration.artifact.artifactId,
      zeroRegistration.artifact.datasetContentHash,
      zeroRegistration.artifact.graphVersionHash,
    ]);
    expect([fewRun.artifactId, fewRun.datasetContentHash, fewRun.graphVersionHash]).toEqual([
      fewRegistration.artifact.artifactId,
      fewRegistration.artifact.datasetContentHash,
      fewRegistration.artifact.graphVersionHash,
    ]);
    expect(parseGovernanceOnlineResult(targetResult("zero_shot")).runId).toBe(zeroRun.runId);
    expect(parseGovernanceOnlineResult(targetResult("few_shot")).runId).toBe(fewRun.runId);
    expect(parseGovernanceOnlinePreview(targetPreview(true, "zero_shot"))).toMatchObject({ runId: zeroRun.runId, resultHash: targetResult("zero_shot").resultHash });
    expect(parseGovernanceOnlinePreview(targetPreview(true, "few_shot"))).toMatchObject({ runId: fewRun.runId, resultHash: targetResult("few_shot").resultHash });
  });

  it.each([
    ["ready", 0],
    ["insufficient_signal", 0.25],
    ["ready", 0.75],
  ])("rejects v2 policy status/lambda disagreement (%s, %s)", (status, selectedLambda) => {
    expect(() => parseTargetReviewPolicy({ ...targetPolicy(), status, selectedLambda }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("keeps modality relation rows distinct from the fused target edge inventory", () => {
    const registration = targetTaskRegistration("zero_shot");
    const realShape = {
      ...registration,
      artifact: { ...registration.artifact, relationRowCount: 264 },
    };

    expect(parseTargetTaskRegistration(realShape)).toMatchObject({
      task: { fusedEdgeCount: 220 },
      artifact: { relationRowCount: 264 },
    });
  });

  it("rejects comparison rank inventories that are not complete permutations", () => {
    const duplicate = targetComparison();
    duplicate.rows[1] = { ...duplicate.rows[1], baseRank: 1, rankDelta: duplicate.rows[1].adaptedRank - 1 };
    expect(() => parseTargetAdaptationComparison(duplicate)).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    const outOfRange = targetComparison();
    outOfRange.rows[107] = { ...outOfRange.rows[107], adaptedRank: 109, rankDelta: 109 - outOfRange.rows[107].baseRank };
    expect(() => parseTargetAdaptationComparison(outOfRange)).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("rejects rebound handoff and activation documents even when every field is valid-shaped", () => {
    expect(parseAdaptationGovernanceHandoff(adaptationHandoff()).handoffHash).toHaveLength(64);
    expect(parseAdaptationOverlayActivation(targetActivation()).activationHash).toHaveLength(64);
    expect(() => parseAdaptationGovernanceHandoff({ ...adaptationHandoff(), targetTaskRegistrationId: `target-task-${"6".repeat(32)}` }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseAdaptationOverlayActivation({ ...targetActivation(), policyHash: "6".repeat(64) }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("rejects a rebound zero-shot review collection", () => {
    const logical = {
      schemaVersion: "socialgraph-fm.governance-review-collection/1.0",
      idempotencyKey: "zero-review",
      targetTaskRegistrationId: targetTaskRegistration("zero_shot").registrationId,
      requestHash: "8".repeat(64),
      resultHash: "3".repeat(64),
      case: { ...casePayload(), state: "active" },
    };
    const collection = { ...logical, collectionHash: sha256Canonical(logical) };
    expect(parseTargetReviewCollection(collection).case.state).toBe("active");
    expect(() => parseTargetReviewCollection({ ...collection, idempotencyKey: "foreign-review" }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });
  it("rejects a few-shot label receipt rebound to another target receipt", () => {
    const registration = targetTaskRegistration();
    expect(() => parseTargetTaskRegistration({ ...registration, labelReceipt: { ...registration.labelReceipt, targetReceiptHash: "6".repeat(64) } }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });
  it("rejects either detached-label member when a zero-shot registration contains only one", () => {
    const zero = targetTaskRegistration("zero_shot");
    const few = targetTaskRegistration("few_shot");
    const reboundLabels = {
      ...few.labels,
      taskId: zero.task.taskId,
      inferenceSha256: zero.artifact.bundleSha256,
    };

    expect(() => parseTargetTaskRegistration({ ...zero, labels: reboundLabels }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseTargetTaskRegistration({ ...zero, labelReceipt: few.labelReceipt }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });
  it("accepts the exact versioned model, artifact, run and comparison contracts", () => {
    expect(parseGovernanceOnlineHealth(onlineHealth()).onlineForwardReady).toBe(true);
    expect(parseGovernanceOnlineCapabilities(onlineCapabilities()).skills).toEqual(GOVERNANCE_PUBLIC_SKILLS);
    expect(parseGovernanceArtifact(onlineArtifact()).compatibility).toBe("compatible");
    expect(parseGovernanceArtifactCompatibility(artifactCompatibility(2)).requiresSelfLoopCleaning).toBe(true);
    expect(parseGovernanceOnlinePreview(onlinePreview()).nodes).toHaveLength(3);
    expect(parseGovernanceOnlinePreview(onlineRunPreview()).resultHash).toBe(onlineResult().resultHash);
    expect(parseGovernanceOnlineRun(onlineRun()).stage).toBe("completed");
    expect(parseCoreFindingPage(findingPage()).total).toBe(3);
    expect(parseGovernanceDerivationPage(derivationPage("factual_relation")).items[0]?.factual).toBe(true);
    expect(parseGovernanceOnlineEvidence(evidencePayload()).structuralSignals.relationEvidenceRole).toBe("explanationOnly");
    expect(parseGovernanceRunComparison(comparisonPayload()).changes[0]?.rankDelta).toBe(-1);
    expect(parseGovernanceOnlineRunRequest({
      schemaVersion: "socialgraph-fm.gfm-governance/2.0", protocol: "global", artifactId: onlineArtifact().artifactId,
      datasetContentHash: onlineArtifact().datasetContentHash, graphVersionHash: onlineArtifact().graphVersionHash,
      modelVersionId: "socialgraph-fm-global/test", modelStateHash: onlineCapabilities().modelStateHash, topK: 100,
    }).modelStateHash).toBe(onlineCapabilities().modelStateHash);
  });

  it("rejects protocol drift, health identity drift and invalid distributions", () => {
    expect(() => parseGovernanceOnlineRunRequest({
      schemaVersion: "socialgraph-fm.gfm-governance/2.0", protocol: "cross_domain", artifactId: onlineArtifact().artifactId,
      datasetContentHash: onlineArtifact().datasetContentHash, graphVersionHash: onlineArtifact().graphVersionHash,
      modelVersionId: "socialgraph-fm-global/test", topK: 100,
    })).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseGovernanceOnlineHealth({ ...onlineHealth(), onlineForwardReady: false })).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseGovernanceOnlineResult({ ...onlineResult(), distribution: { low: 1, review: 1, high: 9, predictedPositive: 2, total: 3 } }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseGovernanceArtifactCompatibility({ ...artifactCompatibility(1), requiresSelfLoopCleaning: false }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseGovernanceOnlinePreview({ ...onlinePreview(), runId: onlineRun().runId }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseGovernanceOnlineCapabilities({ ...onlineCapabilities(), skills: [...GOVERNANCE_PUBLIC_SKILLS].reverse() }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("rejects an oversized preview before graph construction", () => {
    expect(() => parseGovernanceOnlinePreview({ ...onlinePreview(), nodes: Array.from({ length: 3001 }, (_, index) => ({ ...onlinePreview().nodes[0], id: `n-${index}` })) }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("accepts the bounded projection metadata while preserving legacy previews", () => {
    const projected = parseGovernanceOnlinePreview({
      ...onlinePreview(),
      preset: "overview",
      budgets: { nodes: 120, edges: 240 },
      selectionRecipeId: "risk-degree-mmr-v1",
      isPartial: onlinePreview().partialPreview,
      groups: [],
      sourceCounts: { high: 40, review: 40, low: 40 },
      inventoryCounts: { nodes: onlinePreview().nodeCount, edges: onlinePreview().edgeCount, groups: 0 },
    });
    expect(projected).toMatchObject({ preset: "overview", budgets: { nodes: 120, edges: 240 } });
    expect(parseGovernanceOnlinePreview(onlinePreview()).preset).toBeUndefined();
  });

  it("accepts trimmed node identifiers containing slash and backslash", () => {
    const page = findingPage();
    page.items[0] = { ...page.items[0], nodeId: "tenant/account\\2026" };
    expect(parseCoreFindingPage(page).items[0]?.nodeId).toBe("tenant/account\\2026");
    expect(() => parseCoreFindingPage({ ...page, items: [{ ...page.items[0], nodeId: " bad/id" }] }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("accepts the exact target recipe, label-set, policy, and comparison contracts", () => {
    expect(parseTargetLabelRecipe(targetLabelRecipe()).labels).toHaveLength(16);
    expect(parseTargetLabelSet(targetLabelSet()).positiveCount).toBe(8);
    expect(parseAdaptationReviewPolicy(targetReviewPolicy()).status).toBe("ready");
    expect(parseAdaptationComparison(adaptationComparison()).rows[0]).toMatchObject({
      baseScore: 0.58, baseRank: 2, adaptedReviewPriority: 0.86, adaptedRank: 1, rankDelta: -1,
    });
  });

  it("rejects label imbalance, policy publication drift, and recomputed rank deltas", () => {
    expect(() => parseTargetLabelRecipe({ ...targetLabelRecipe(), labels: targetLabelRecipe().labels.slice(0, 7) }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseTargetLabelSet({ ...targetLabelSet(), conflicts: ["n1"] }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseAdaptationReviewPolicy({ ...targetReviewPolicy(), readyPolicyHash: null }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    const comparison = adaptationComparison();
    expect(() => parseAdaptationComparison({ ...comparison, rows: [{ ...comparison.rows[0], rankDelta: 0 }] }))
      .toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("rejects a source-record inventory that no longer matches label provenance", () => {
    const labelSet = targetLabelSet();
    expect(() => parseTargetLabelSet({
      ...labelSet,
      sourceRecords: labelSet.sourceRecords.map((record, index) => index === 0
        ? { ...record, sourceRecordHash: "f".repeat(64) }
        : record),
    })).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("binds the exact canonical labels file to a hashed generator receipt", () => {
    const parseSidecar = (onlineContractsModule as typeof onlineContractsModule & {
      parseTargetLabelSidecar?: (labelsText: string, receiptText: string) => { recipe: ReturnType<typeof targetLabelRecipe>; receipt: ReturnType<typeof targetPackageReceipt> };
    }).parseTargetLabelSidecar;
    expect(parseSidecar).toBeTypeOf("function");
    if (!parseSidecar) return;
    const labelsText = `${canonicalJson(targetLabelRecipe())}\n`;
    const receiptText = canonicalJson(targetPackageReceipt());
    const parsed = parseSidecar(labelsText, receiptText);
    expect(parsed.receipt.labelsSha256).toBe(targetPackageReceipt().labelsSha256);
    expect(parsed.receipt.receiptHash).toBe(targetPackageReceipt().receiptHash);
    expect(parsed.recipe.labels).toHaveLength(16);
  });

  it.each([
    ["duplicate node", () => {
      const recipe = targetLabelRecipe();
      return { ...recipe, labels: recipe.labels.map((row, index) => index === 1 ? { ...row, nodeId: recipe.labels[0].nodeId } : row) };
    }],
    ["wrong structural quota", () => {
      const recipe = targetLabelRecipe();
      return { ...recipe, labels: recipe.labels.map((row, index) => index === 1 ? { ...row, structuralStratum: 1 } : row) };
    }],
    ["missing declared fused degree", () => {
      const recipe = targetLabelRecipe();
      return { ...recipe, labels: recipe.labels.map((row, index) => index === 0 ? { nodeId: row.nodeId, label: row.label, structuralStratum: row.structuralStratum } : row) };
    }],
  ])("rejects a %s in the structural sidecar", (_name, mutate) => {
    expect(() => parseTargetLabelRecipe(mutate())).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });

  it("rejects tampered labels bytes and receipt hashes before import", () => {
    const parseSidecar = (onlineContractsModule as typeof onlineContractsModule & {
      parseTargetLabelSidecar?: (labelsText: string, receiptText: string) => unknown;
    }).parseTargetLabelSidecar;
    expect(parseSidecar).toBeTypeOf("function");
    if (!parseSidecar) return;
    const labelsText = `${canonicalJson(targetLabelRecipe())}\n`;
    const receipt = targetPackageReceipt();
    expect(() => parseSidecar(`${labelsText} `, canonicalJson(receipt))).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
    expect(() => parseSidecar(labelsText, canonicalJson({ ...receipt, receiptHash: "f".repeat(64) }))).toThrow("GFM_GOVERNANCE_RESPONSE_INVALID");
  });
});
