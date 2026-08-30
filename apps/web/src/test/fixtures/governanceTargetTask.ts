import { casePayload, evidencePayload, onlineArtifact, onlineCapabilities, onlinePreview, onlineResult, onlineRun } from "./governanceOnline";
import type { TargetReviewCollectionCreateRequest } from "../../types/governanceOnline";
import { sha256Canonical } from "../../services/graphIdentity";

export type TargetFixtureMode = "zero_shot" | "few_shot";

const fixtureHash = (mode: TargetFixtureMode, field: string): string => sha256Canonical({ fixture: "governance-target-task", mode, field });
const sharedFixtureHash = (field: string): string => sha256Canonical({ fixture: "governance-target-task", field });

export const TARGET_REGISTRATION_ID = `target-task-${"7".repeat(32)}`;
export const TARGET_ZERO_REGISTRATION_ID = `target-task-${"6".repeat(32)}`;
export const TARGET_ARTIFACT_ID = `governance-artifact-${"b".repeat(32)}`;
export const TARGET_ZERO_ARTIFACT_ID = `governance-artifact-${"a".repeat(32)}`;
export const TARGET_RUN_ID = `governance-${"d".repeat(32)}`;
export const TARGET_ZERO_RUN_ID = `governance-${"c".repeat(32)}`;

export function targetFixtureIdentity(mode: TargetFixtureMode = "few_shot") {
  return Object.freeze({
    registrationId: mode === "few_shot" ? TARGET_REGISTRATION_ID : TARGET_ZERO_REGISTRATION_ID,
    artifactId: mode === "few_shot" ? TARGET_ARTIFACT_ID : TARGET_ZERO_ARTIFACT_ID,
    runId: mode === "few_shot" ? TARGET_RUN_ID : TARGET_ZERO_RUN_ID,
    artifactHash: fixtureHash(mode, "artifact"),
    bundleSha256: fixtureHash(mode, "inference-bundle"),
    manifestSha256: fixtureHash(mode, "inference-manifest"),
    datasetContentHash: fixtureHash(mode, "dataset-content"),
    graphVersionHash: fixtureHash(mode, "graph-version"),
    requestHash: fixtureHash(mode, "run-request"),
    statusHash: fixtureHash(mode, "run-status"),
    resultHash: fixtureHash(mode, "run-result"),
    rawPreviewHash: fixtureHash(mode, "raw-preview"),
    scoredPreviewHash: fixtureHash(mode, "scored-preview"),
    findingsPageHash: fixtureHash(mode, "findings-page"),
    derivationPageHash: fixtureHash(mode, "derivation-page"),
    evidenceHash: fixtureHash(mode, "evidence"),
    sourceContentHash: fixtureHash(mode, "source-content"),
    sourceManifestSha256: fixtureHash(mode, "source-manifest"),
    labelEligibilityMaskSha256: fixtureHash(mode, "label-eligibility-mask"),
    nodeSetSha256: fixtureHash(mode, "node-set"),
    targetReceiptHash: fixtureHash(mode, "target-receipt"),
    labelsSha256: fixtureHash(mode, "labels-sidecar"),
    sourceLabelsSha256: fixtureHash(mode, "source-labels"),
    labelSetHash: fixtureHash(mode, "label-set"),
    labelReceiptHash: fixtureHash(mode, "label-receipt"),
    outerBundleSha256: fixtureHash(mode, "outer-bundle"),
    targetReceiptSha256: fixtureHash(mode, "target-receipt-entry"),
    labelReceiptSha256: fixtureHash(mode, "label-receipt-entry"),
    registrationHash: fixtureHash(mode, "registration"),
    runArtifactHash: fixtureHash(mode, "run-artifact"),
    recipeHash: sharedFixtureHash("adaptation-recipe"),
    codeHash: sharedFixtureHash("adaptation-code"),
    policyHash: fixtureHash(mode, "policy"),
    comparisonHash: fixtureHash(mode, "comparison"),
  });
}

export function targetTaskRegistration(mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  const artifact = {
    ...onlineArtifact(),
    artifactId: identity.artifactId,
    datasetId: `target-${mode}`,
    displayName: mode === "few_shot" ? "Regional review task B" : "Regional review task A",
    bundleSha256: identity.bundleSha256,
    manifestSha256: identity.manifestSha256,
    datasetContentHash: identity.datasetContentHash,
    graphVersionHash: identity.graphVersionHash,
    artifactHash: identity.artifactHash,
    nodeCount: 108,
    relationRowCount: 220,
  };
  const targetReceipt = {
    schemaVersion: "socialgraph-fm.governance-target-domain-receipt/2.0",
    taskId: `regional-${mode}`,
    countryId: "regional-target",
    sourceContentHash: identity.sourceContentHash,
    sourceManifestSha256: identity.sourceManifestSha256,
    graphPopulation: "full",
    graphPopulationMaskSha256: null,
    labelEligibility: mode === "few_shot" ? "fold0-test" : "none",
    labelEligibilityMaskSha256: mode === "few_shot" ? identity.labelEligibilityMaskSha256 : null,
    inferenceSha256: artifact.bundleSha256,
    nodeSetSha256: identity.nodeSetSha256,
    nodeCount: 108,
    fusedEdgeCount: 220,
    modalities: ["coRT", "coURL", "hashSeq", "fastRT", "tweetSim"],
    connected: true,
    selectionRecipe: { version: "connected-target-v1", scoreInputs: [] },
    receiptHash: identity.targetReceiptHash,
  };
  const labels = mode === "few_shot" ? {
    schemaVersion: "socialgraph-fm.governance-target-label-set/2.0",
    taskId: targetReceipt.taskId,
    inferenceSha256: artifact.bundleSha256,
    labels: Array.from({ length: 16 }, (_, index) => ({
      nodeId: `target-node-${String(index + 1).padStart(3, "0")}`,
      label: index < 8 ? "positive" : "negative",
      structuralStratum: index % 4,
      fusedDegree: index + 1,
    })),
    positiveCount: 8,
    negativeCount: 8,
    labelSetHash: identity.labelSetHash,
  } : null;
  const labelReceipt = mode === "few_shot" ? {
    schemaVersion: "socialgraph-fm.governance-target-label-receipt/2.0",
    taskId: targetReceipt.taskId,
    targetReceiptHash: targetReceipt.receiptHash,
    labelsSha256: identity.labelsSha256,
    sourceLabelsSha256: identity.sourceLabelsSha256,
    eligibilityMaskSha256: identity.labelEligibilityMaskSha256,
    eligibleNodeIds: labels!.labels.map((row) => row.nodeId),
    selectionRecipe: { version: "target-quartile-v2", scoreInputs: [] },
    receiptHash: identity.labelReceiptHash,
  } : null;
  return {
    schemaVersion: "socialgraph-fm.governance-target-task-registration/1.0",
    registrationId: identity.registrationId,
    outerBundleSha256: identity.outerBundleSha256,
    task: {
      schemaVersion: "socialgraph-fm.governance-target-task-bundle/1.0",
      taskId: targetReceipt.taskId,
      displayName: artifact.displayName,
      mode,
      nodeCount: 108,
      fusedEdgeCount: 220,
      modalities: targetReceipt.modalities,
      inference: { name: "inference.zip", sha256: artifact.bundleSha256, bytes: 1000 },
      targetReceipt: { name: "target-receipt.json", sha256: identity.targetReceiptSha256, bytes: 800 },
      labels: mode === "few_shot" ? { name: "labels.json", sha256: identity.labelsSha256, bytes: 500 } : null,
      labelReceipt: mode === "few_shot" ? { name: "label-receipt.json", sha256: identity.labelReceiptSha256, bytes: 600 } : null,
    },
    targetReceipt,
    labels,
    labelReceipt,
    artifact,
    createdAt: "2026-08-21T00:00:00Z",
    registrationHash: identity.registrationHash,
  };
}

export function targetRun(mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  return {
    ...onlineRun("succeeded", identity.runId),
    requestHash: identity.requestHash,
    artifactId: identity.artifactId,
    datasetContentHash: identity.datasetContentHash,
    graphVersionHash: identity.graphVersionHash,
    status: "succeeded" as const,
    stage: "completed" as const,
    progress: 100,
    statusHash: identity.statusHash,
  };
}

export function targetResult(mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  const template = onlineResult().findings[0];
  return {
    ...onlineResult(),
    runId: identity.runId,
    requestHash: identity.requestHash,
    artifactId: identity.artifactId,
    datasetContentHash: identity.datasetContentHash,
    graphVersionHash: identity.graphVersionHash,
    findings: Array.from({ length: 108 }, (_, index) => ({
      ...template,
      nodeId: `target-node-${String(index + 1).padStart(3, "0")}`,
      label: `对象 ${index + 1}`,
      score: Number((0.9 - index / 1000).toFixed(4)),
      rank: index + 1,
      riskBand: index < 36 ? "high" : index < 72 ? "review" : "low",
      predictedPositive: index < 36,
      communityId: `group-${index % 6}`,
    })),
    distribution: { low: 36, review: 36, high: 36, predictedPositive: 36, total: 108 },
    totalFindings: 108,
    resultHash: identity.resultHash,
  };
}

export function targetPolicy(mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  const run = targetRun(mode);
  const result = targetResult(mode);
  return {
    schemaVersion: "socialgraph-fm.governance-target-review-policy/2.0",
    binding: {
      artifactId: run.artifactId,
      datasetContentHash: run.datasetContentHash,
      graphVersionHash: run.graphVersionHash,
      runId: run.runId,
      requestHash: run.requestHash,
      resultHash: result.resultHash,
      runArtifactHash: identity.runArtifactHash,
      modelVersionId: run.modelVersionId,
      modelVersionHash: run.modelVersionHash,
      modelStateHash: run.modelStateHash,
      recipeHash: identity.recipeHash,
      codeHash: identity.codeHash,
      seed: 17,
    },
    labelSetHash: identity.labelSetHash,
    status: "ready",
    selectedLambda: 0.5,
    eligibleLabelCount: 16,
    positiveCount: 8,
    negativeCount: 8,
    fittingRecipe: "l2-centroids+run-zscore+loo-balanced-log-loss-v1",
    baseOutputsImmutable: true,
    adaptedOutputFields: ["adaptedReviewPriority", "adaptedRank"],
    policyHash: identity.policyHash,
  };
}

export function targetComparison(total = 108, mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  const policy = targetPolicy(mode);
  const findings = targetResult(mode).findings.slice(0, total);
  return {
    schemaVersion: "socialgraph-fm.governance-adaptation-comparison/2.0",
    binding: policy.binding,
    policyHash: policy.policyHash,
    total,
    baseOutputsImmutable: true,
    rows: findings.map((finding, index) => ({
      nodeId: finding.nodeId,
      baseScore: finding.score,
      baseRank: finding.rank,
      adaptedReviewPriority: Number((0.9 - (total - index - 1) / 1000).toFixed(4)),
      adaptedRank: total - index,
      rankDelta: total - index - finding.rank,
    })),
    comparisonHash: identity.comparisonHash,
  };
}

export function adaptationHandoff(mode: TargetFixtureMode = "few_shot") {
  const registration = targetTaskRegistration(mode);
  const policy = targetPolicy(mode);
  const logical = {
    schemaVersion: "socialgraph-fm.governance-adaptation-handoff/1.0",
    targetTaskRegistrationId: registration.registrationId,
    targetReceiptHash: registration.targetReceipt.receiptHash,
    labelSetHash: policy.labelSetHash,
    binding: policy.binding,
    policyHash: policy.policyHash,
    comparisonHash: targetComparison(108, mode).comparisonHash,
    decision: "pending_governance_review",
    baseModelMutation: false,
  };
  return { ...logical, handoffHash: sha256Canonical(logical) };
}

export function targetActivation(mode: TargetFixtureMode = "few_shot") {
  const registration = targetTaskRegistration(mode); const policy = targetPolicy(mode); const comparison = targetComparison(108, mode);
  const logical = {
    schemaVersion: "socialgraph-fm.governance-adaptation-overlay/1.0",
    targetTaskRegistrationId: registration.registrationId,
    targetReceiptHash: registration.targetReceipt.receiptHash,
    labelSetHash: policy.labelSetHash,
    binding: policy.binding,
    policyHash: policy.policyHash,
    comparisonHash: comparison.comparisonHash,
    active: true,
    baseModelMutation: false,
  };
  return { ...logical, activationHash: sha256Canonical(logical) };
}

export function targetReviewCollection(request?: TargetReviewCollectionCreateRequest) {
  const registration = targetTaskRegistration("zero_shot"); const result = targetResult("zero_shot");
  const expectedRequest: TargetReviewCollectionCreateRequest = request ?? {
    schemaVersion: "socialgraph-fm.governance-review-collection/1.0",
    idempotencyKey: `${registration.registrationId}:${result.resultHash.slice(0, 16)}`,
    targetTaskRegistrationId: registration.registrationId,
    runId: result.runId,
    resultHash: result.resultHash,
    title: `${registration.task.displayName} 风险候选复核`,
    description: "由零样本风险排序显式移交。",
    items: result.findings.slice(0, 25).map((finding) => ({ targetType: "node", targetId: finding.nodeId, note: `风险排序 #${finding.rank}` })),
  };
  const baseCase = casePayload();
  const reviewCase = {
    ...baseCase,
    runId: expectedRequest.runId,
    state: "active",
    title: expectedRequest.title,
    description: expectedRequest.description,
    items: expectedRequest.items.map((item, index) => ({
      itemId: `item-${(index + 1).toString(16).padStart(32, "0")}`,
      targetType: item.targetType,
      targetId: item.targetId,
      note: item.note,
      createdAt: "2026-08-21T00:00:00Z",
      itemHash: (index + 1).toString(16).padStart(64, "0"),
    })),
    reviewEvents: [],
    currentDecisions: {},
  };
  const logical = {
    schemaVersion: "socialgraph-fm.governance-review-collection/1.0",
    idempotencyKey: expectedRequest.idempotencyKey,
    targetTaskRegistrationId: expectedRequest.targetTaskRegistrationId,
    requestHash: sha256Canonical(expectedRequest),
    resultHash: expectedRequest.resultHash,
    case: reviewCase,
  };
  return { ...logical, collectionHash: sha256Canonical(logical) };
}

export function targetPreview(scored = false, mode: TargetFixtureMode = "few_shot") {
  const registration = targetTaskRegistration(mode);
  const identity = targetFixtureIdentity(mode);
  const nodes = Array.from({ length: 108 }, (_, index) => ({
    id: `target-node-${String(index + 1).padStart(3, "0")}`,
    label: `对象 ${index + 1}`,
    degree: 4,
    structureMissing: false,
    score: scored ? Number((0.9 - index / 1000).toFixed(4)) : null,
    riskBand: scored ? index < 36 ? "high" : index < 72 ? "review" : "low" : null,
    groupId: scored ? `group-${index % 6}` : null,
  }));
  const edges = Array.from({ length: 220 }, (_, index) => {
    const source = index % 108;
    const target = (source + 1 + Math.floor(index / 108)) % 108;
    return { id: `target-edge-${index + 1}`, source: nodes[source].id, target: nodes[target].id, modalities: ["coRT"], factual: true };
  });
  return {
    ...onlinePreview(),
    artifactId: registration.artifact.artifactId,
    datasetContentHash: registration.artifact.datasetContentHash,
    graphVersionHash: registration.artifact.graphVersionHash,
    runId: scored ? identity.runId : null,
    resultHash: scored ? identity.resultHash : null,
    nodes,
    edges,
    nodeCount: 108,
    edgeCount: 220,
    partialPreview: false,
    previewHash: scored ? identity.scoredPreviewHash : identity.rawPreviewHash,
  };
}

export function targetFindingPage(mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  return {
    schemaVersion: "socialgraph-fm.gfm-governance/2.0",
    runId: identity.runId,
    items: targetResult(mode).findings,
    total: 108,
    offset: 0,
    limit: 10_000,
    pageHash: identity.findingsPageHash,
  };
}

export function targetDerivationPage(mode: TargetFixtureMode = "few_shot") {
  const identity = targetFixtureIdentity(mode);
  return {
    schemaVersion: "socialgraph-fm.gfm-governance/2.0",
    runId: identity.runId,
    items: [],
    total: 0,
    offset: 0,
    limit: 10_000,
    pageHash: identity.derivationPageHash,
  };
}

export function targetCapabilities() {
  return onlineCapabilities();
}

export function targetEvidence(nodeId = "target-node-001", mode: TargetFixtureMode = "few_shot") {
  const payload = evidencePayload();
  const identity = targetFixtureIdentity(mode);
  const result = targetResult(mode);
  const finding = result.findings.find((candidate) => candidate.nodeId === nodeId) ?? result.findings[0];
  const neighbor = nodeId === "target-node-002" ? result.findings[0] : result.findings[1];
  return {
    ...payload,
    runId: identity.runId,
    resultHash: identity.resultHash,
    artifactId: identity.artifactId,
    datasetContentHash: identity.datasetContentHash,
    graphVersionHash: identity.graphVersionHash,
    node: finding,
    neighbors: [{ ...payload.neighbors[0], nodeId: neighbor.nodeId, score: neighbor.score, riskBand: neighbor.riskBand, predictedPositive: neighbor.predictedPositive }],
    evidenceSubgraph: {
      ...payload.evidenceSubgraph,
      nodes: [
        { ...payload.evidenceSubgraph.nodes[0], nodeId: finding.nodeId, score: finding.score, riskBand: finding.riskBand, predictedPositive: finding.predictedPositive },
        { ...payload.evidenceSubgraph.nodes[1], nodeId: neighbor.nodeId, score: neighbor.score, riskBand: neighbor.riskBand, predictedPositive: neighbor.predictedPositive },
      ],
      edges: [{ ...payload.evidenceSubgraph.edges[0], id: `${mode}-evidence-edge`, source: finding.nodeId, target: neighbor.nodeId }],
    },
    evidenceHash: identity.evidenceHash,
  };
}
