import { sha256Canonical } from "./graphIdentity";
import type { AnalysisOverlay, GraphVersion } from "../types/graph";
import type {
  TargetAdaptationComparison,
  AdaptationGovernanceHandoff,
  GovernanceArtifact,
  GovernanceDerivation,
  GovernanceOnlineClientLike,
  GovernanceOnlineFinding,
  GovernanceOnlineResult,
  GovernanceOnlineRun,
  TargetReviewPolicy,
  TargetTaskRegistration,
} from "../types/governanceOnline";
import type { GovernanceWorkspaceSnapshot } from "./governanceWorkspaceStore";

export interface GovernanceAdaptationState {
  readonly lane: "zero_shot" | "few_shot";
  readonly registration: TargetTaskRegistration;
  readonly handoff?: AdaptationGovernanceHandoff;
  readonly policy?: TargetReviewPolicy;
  readonly comparison?: TargetAdaptationComparison;
  readonly adaptedOverlay?: AnalysisOverlay;
}

export interface ValidatedGovernanceAdaptationState extends GovernanceAdaptationState {
  readonly registration: TargetTaskRegistration;
}

function fail(): never {
  throw new Error("GFM_GOVERNANCE_ADAPTATION_GOVERNANCE_IDENTITY_INVALID");
}

function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) throw new DOMException("The operation was aborted.", "AbortError");
}

function sameBinding(
  left: TargetReviewPolicy["binding"],
  right: TargetReviewPolicy["binding"],
): boolean {
  return sha256Canonical(left) === sha256Canonical(right);
}

function isPermutation(values: readonly number[]): boolean {
  return [...values]
    .sort((left, right) => left - right)
    .every((value, index) => value === index + 1);
}

function exactLiveExecution(
  live: TargetTaskRegistration,
  expected: TargetTaskRegistration,
  snapshot: GovernanceWorkspaceSnapshot,
  lane: GovernanceAdaptationState["lane"],
  artifact: GovernanceArtifact,
  run: GovernanceOnlineRun,
  result: GovernanceOnlineResult,
): boolean {
  const storedRun = snapshot.run;
  const storedResult = snapshot.result;
  return live.registrationId === expected.registrationId
    && live.registrationHash === expected.registrationHash
    && sha256Canonical(live) === sha256Canonical(expected)
    && live.task.mode === lane
    && live.targetReceipt.receiptHash === expected.targetReceipt.receiptHash
    && live.artifact.artifactId === snapshot.artifact.artifactId
    && live.artifact.datasetContentHash === snapshot.artifact.datasetContentHash
    && live.artifact.graphVersionHash === snapshot.artifact.graphVersionHash
    && live.artifact.bundleSha256 === live.targetReceipt.inferenceSha256
    && sha256Canonical(artifact) === sha256Canonical(live.artifact)
    && Boolean(storedRun && storedResult && storedRun.status === "succeeded")
    && run.status === "succeeded"
    && run.runId === storedRun?.runId
    && run.statusHash === storedRun?.statusHash
    && result.resultHash === storedResult?.resultHash
    && sha256Canonical(result) === sha256Canonical(storedResult)
    && run.runId === result.runId
    && run.artifactId === live.artifact.artifactId
    && run.datasetContentHash === live.artifact.datasetContentHash
    && run.graphVersionHash === live.artifact.graphVersionHash
    && run.requestHash === result.requestHash
    && run.modelVersionId === result.modelVersionId
    && run.modelVersionHash === result.modelVersionHash
    && run.modelStateHash === result.modelStateHash;
}

function exactComparison(
  comparison: TargetAdaptationComparison,
  policy: TargetReviewPolicy,
  result: GovernanceOnlineResult,
): boolean {
  const findings = new Map(result.findings.map((finding) => [finding.nodeId, finding]));
  const rowIds = new Set(comparison.rows.map((row) => row.nodeId));
  return comparison.policyHash === policy.policyHash
    && sameBinding(comparison.binding, policy.binding)
    && comparison.baseOutputsImmutable
    && comparison.total === result.totalFindings
    && comparison.rows.length === result.totalFindings
    && rowIds.size === findings.size
    && rowIds.size === comparison.total
    && isPermutation(comparison.rows.map((row) => row.baseRank))
    && isPermutation(comparison.rows.map((row) => row.adaptedRank))
    && comparison.rows.every((row) => {
      const finding = findings.get(row.nodeId);
      return Boolean(finding)
        && row.baseScore === finding?.score
        && row.baseRank === finding?.rank
        && row.rankDelta === row.adaptedRank - row.baseRank
        && Number.isFinite(row.adaptedReviewPriority)
        && row.adaptedReviewPriority >= 0
        && row.adaptedReviewPriority <= 1;
    });
}

/**
 * Reload every server-owned identity before an adaptation may affect governance.
 * Values retained by React are request coordinates only, never display authority.
 */
export async function revalidateGovernanceAdaptation(
  client: GovernanceOnlineClientLike,
  expected: GovernanceAdaptationState,
  snapshot: GovernanceWorkspaceSnapshot,
  signal?: AbortSignal,
): Promise<ValidatedGovernanceAdaptationState> {
  throwIfAborted(signal);
  const registration = await client.targetTask(expected.registration.registrationId, signal);
  throwIfAborted(signal);
  if (!snapshot.run || !snapshot.result) fail();
  const artifact = await client.artifact(registration.artifact.artifactId, signal);
  throwIfAborted(signal);
  const run = await client.run(snapshot.run.runId, signal);
  throwIfAborted(signal);
  const result = await client.result(snapshot.run.runId, signal);
  throwIfAborted(signal);
  if (!exactLiveExecution(registration, expected.registration, snapshot, expected.lane, artifact, run, result)) fail();
  if (expected.lane === "zero_shot") {
    if (expected.handoff || expected.policy || expected.comparison || expected.adaptedOverlay) fail();
    return Object.freeze({ lane: "zero_shot", registration });
  }

  if (!expected.handoff || !expected.policy || !expected.comparison || !expected.adaptedOverlay) fail();
  const handoff = await client.adaptationHandoff(expected.handoff.handoffHash, signal);
  throwIfAborted(signal);
  const policy = await client.targetPolicy(expected.policy.policyHash, signal);
  throwIfAborted(signal);
  const comparison = await client.targetComparison(snapshot.run.runId, expected.policy.policyHash, 0, 500, signal);
  throwIfAborted(signal);
  if (sha256Canonical(handoff) !== sha256Canonical(expected.handoff)
    || sha256Canonical(policy) !== sha256Canonical(expected.policy)
    || sha256Canonical(comparison) !== sha256Canonical(expected.comparison)
    || policy.status !== "ready"
    || ![0.25, 0.5, 1].includes(policy.selectedLambda)
    || !policy.baseOutputsImmutable
    || policy.adaptedOutputFields[0] !== "adaptedReviewPriority"
    || policy.adaptedOutputFields[1] !== "adaptedRank"
    || handoff.baseModelMutation
    || handoff.targetTaskRegistrationId !== registration.registrationId
    || handoff.targetReceiptHash !== registration.targetReceipt.receiptHash
    || handoff.labelSetHash !== policy.labelSetHash
    || registration.labels?.labelSetHash !== policy.labelSetHash
    || registration.labels?.inferenceSha256 !== registration.artifact.bundleSha256
    || registration.labelReceipt?.targetReceiptHash !== registration.targetReceipt.receiptHash
    || registration.labelReceipt?.taskId !== registration.task.taskId
    || handoff.policyHash !== policy.policyHash
    || handoff.comparisonHash !== comparison.comparisonHash
    || !sameBinding(handoff.binding, policy.binding)
    || !sameBinding(policy.binding, comparison.binding)
    || policy.binding.runId !== run.runId
    || policy.binding.requestHash !== run.requestHash
    || policy.binding.resultHash !== result.resultHash
    || policy.binding.artifactId !== snapshot.artifact.artifactId
    || policy.binding.datasetContentHash !== snapshot.artifact.datasetContentHash
    || policy.binding.graphVersionHash !== snapshot.artifact.graphVersionHash
    || policy.binding.modelVersionId !== result.modelVersionId
    || policy.binding.modelVersionHash !== result.modelVersionHash
    || policy.binding.modelStateHash !== result.modelStateHash
    || !exactComparison(comparison, policy, result)) fail();

  return Object.freeze({
    lane: "few_shot",
    registration,
    handoff,
    policy,
    comparison,
    adaptedOverlay: expected.adaptedOverlay,
  });
}

export function buildAdaptedReviewPriorityOverlay(
  graph: GraphVersion,
  result: GovernanceOnlineResult,
  comparison: TargetAdaptationComparison,
  governanceLens: "risk" | "relations" = "risk",
  baseOverlay?: AnalysisOverlay,
): AnalysisOverlay {
  return Object.freeze({
    id: `${graph.id}:adapted-review-priority:${comparison.comparisonHash}`,
    graphVersionId: graph.id,
    kind: baseOverlay?.kind ?? "governance",
    nodeValues: Object.freeze(Object.fromEntries(comparison.rows.map((row) => [row.nodeId, row.adaptedReviewPriority]))),
    edgeValues: baseOverlay?.edgeValues ?? Object.freeze({}),
    candidateEdges: baseOverlay?.candidateEdges,
    presentation: Object.freeze({
      governanceLens,
      rankDeltas: Object.freeze(Object.fromEntries(comparison.rows.map((row) => [row.nodeId, row.rankDelta]))),
      adaptedRanks: Object.freeze(Object.fromEntries(comparison.rows.map((row) => [row.nodeId, row.adaptedRank]))),
    }),
    legend: Object.freeze({
      title: governanceLens === "relations" ? "适配复核优先级 / 关系" : "适配后复核优先级",
      items: Object.freeze([
        { value: "adapted", label: "适配后复核优先级", color: "var(--purple)" },
        ...(governanceLens === "relations" ? baseOverlay?.legend?.items ?? [] : []),
      ]),
    }),
    provenance: Object.freeze({
      engine: "socialgraph-governance",
      algorithm: "few_shot",
      runId: result.runId,
      resultHash: result.resultHash,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
    }),
  });
}

export function withAdaptedRankPresentation(
  overlay: AnalysisOverlay,
  comparison: TargetAdaptationComparison,
): AnalysisOverlay {
  return Object.freeze({
    ...overlay,
    presentation: Object.freeze({
      ...overlay.presentation,
      rankDeltas: Object.freeze(Object.fromEntries(comparison.rows.map((row) => [row.nodeId, row.rankDelta]))),
      adaptedRanks: Object.freeze(Object.fromEntries(comparison.rows.map((row) => [row.nodeId, row.adaptedRank]))),
    }),
  });
}

export function sortFindingsByAdaptedRank(
  findings: readonly GovernanceOnlineFinding[],
  comparison: TargetAdaptationComparison | null,
): readonly GovernanceOnlineFinding[] {
  if (!comparison) return findings;
  const ranks = new Map(comparison.rows.map((row) => [row.nodeId, row.adaptedRank]));
  return [...findings].sort((left, right) => (ranks.get(left.nodeId) ?? Number.MAX_SAFE_INTEGER)
    - (ranks.get(right.nodeId) ?? Number.MAX_SAFE_INTEGER)
    || left.rank - right.rank
    || left.nodeId.localeCompare(right.nodeId));
}

export function sortDerivationsByAdaptedRank(
  items: readonly GovernanceDerivation[],
  comparison: TargetAdaptationComparison | null,
): readonly GovernanceDerivation[] {
  if (!comparison) return items;
  const ranks = new Map(comparison.rows.map((row) => [row.nodeId, row.adaptedRank]));
  const priority = (item: GovernanceDerivation) => Math.min(...item.nodeIds.map((nodeId) => ranks.get(nodeId) ?? Number.MAX_SAFE_INTEGER));
  return [...items].sort((left, right) => priority(left) - priority(right) || left.id.localeCompare(right.id));
}
