import type { GlobalModelModelCard } from "../../types/globalModel";
import type {
  TargetAdaptationComparison,
  GovernanceOnlineEvidence,
  GovernanceOnlineFinding,
  GovernanceOnlineModality,
  GovernanceOnlineResult,
  TargetReviewPolicy,
  TargetTaskRegistration,
} from "../../types/governanceOnline";

export interface AdaptationModelCardState {
  readonly status: "loading" | "ready" | "error";
  readonly card: GlobalModelModelCard | null;
}

export interface AdaptationExpertEvidence {
  readonly id: `source-${string}` | "null";
  readonly label: string;
  readonly routingMass: number;
  readonly coverage: number;
  readonly selectedNodeCount: number;
  readonly averageWeight: number;
  readonly trainingNodeCount: number | null;
}

export interface AdaptationSelectedObjectEvidence {
  readonly nodeId: string;
  readonly label: string;
  readonly rank: number;
  readonly routes: readonly Readonly<{ label: string; weight: number }>[];
  readonly textResponse: number;
  readonly structureResponse: number;
  readonly relationEvidenceCount: number;
}

export interface AdaptationCalibrationEvidence {
  readonly positiveCount: number;
  readonly negativeCount: number;
  readonly selectedLambda: number;
  readonly raisedCount: number;
  readonly loweredCount: number;
  readonly unchangedCount: number;
  readonly maxRankChange: number;
}

export interface AdaptationTransferEvidence {
  readonly sharedRepresentation: "fixed";
  readonly trainingNodeCount: number;
  readonly experts: readonly AdaptationExpertEvidence[];
  readonly activeSourceCount: number;
  readonly primaryRoute: AdaptationExpertEvidence | null;
  readonly nullRoutingMass: number;
  readonly textResponseMean: number;
  readonly structureResponseMean: number;
  readonly structureAvailableCount: number;
  readonly structureMissingCount: number;
  readonly targetModalities: readonly GovernanceOnlineModality[];
  readonly selectedObject: AdaptationSelectedObjectEvidence | null;
  readonly calibration: AdaptationCalibrationEvidence | null;
}

export type AdaptationTransferEvidenceState =
  | { readonly status: "ready"; readonly value: AdaptationTransferEvidence }
  | { readonly status: "unavailable"; readonly reason: "model_card_loading" | "model_card_unavailable" | "identity_mismatch" | "invalid_routes" };

type AdaptationTransferUnavailableReason = Extract<AdaptationTransferEvidenceState, { status: "unavailable" }>["reason"];

interface BuildAdaptationTransferEvidenceInput {
  readonly result: GovernanceOnlineResult;
  readonly registration: TargetTaskRegistration;
  readonly modelCardState: AdaptationModelCardState | undefined;
  readonly selectedNodeId?: string | null;
  readonly selectedEvidence?: GovernanceOnlineEvidence | null;
  readonly policy?: TargetReviewPolicy | null;
  readonly comparison?: TargetAdaptationComparison | null;
}

const ROUTE_EPSILON = 1e-5;

function unavailable(reason: AdaptationTransferUnavailableReason): AdaptationTransferEvidenceState {
  return Object.freeze({ status: "unavailable", reason });
}

function finiteUnit(value: number): boolean {
  return Number.isFinite(value) && value >= 0 && value <= 1;
}

function selectedObject(
  finding: GovernanceOnlineFinding | undefined,
  expertLabels: ReadonlyMap<string, string>,
  evidence: GovernanceOnlineEvidence | null | undefined,
): AdaptationSelectedObjectEvidence | null {
  if (!finding) return null;
  const relationEvidenceCount = Object.values(finding.modalityEvidence).reduce((sum, count) => sum + (count ?? 0), 0);
  const evidenceCount = evidence?.node.nodeId === finding.nodeId
    ? Object.values(evidence.structuralSignals.relationNeighborCounts).reduce((sum, count) => sum + count, 0)
    : relationEvidenceCount;
  return Object.freeze({
    nodeId: finding.nodeId,
    label: finding.label?.trim() || "匿名账号",
    rank: finding.rank,
    routes: Object.freeze(finding.routes
      .filter((route) => route.expert !== "shared")
      .map((route) => Object.freeze({ label: expertLabels.get(route.expert) ?? "保守未知域", weight: route.weight }))),
    textResponse: finding.modalityContribution.text,
    structureResponse: finding.modalityContribution.structure,
    relationEvidenceCount: evidenceCount,
  });
}

function calibrationEvidence(
  policy: TargetReviewPolicy | null | undefined,
  comparison: TargetAdaptationComparison | null | undefined,
): AdaptationCalibrationEvidence | null {
  if (!policy || !comparison || policy.policyHash !== comparison.policyHash || policy.status !== "ready") return null;
  let raisedCount = 0;
  let loweredCount = 0;
  let unchangedCount = 0;
  let maxRankChange = 0;
  for (const row of comparison.rows) {
    if (row.rankDelta < 0) raisedCount += 1;
    else if (row.rankDelta > 0) loweredCount += 1;
    else unchangedCount += 1;
    maxRankChange = Math.max(maxRankChange, Math.abs(row.rankDelta));
  }
  return Object.freeze({
    positiveCount: policy.positiveCount,
    negativeCount: policy.negativeCount,
    selectedLambda: policy.selectedLambda,
    raisedCount,
    loweredCount,
    unchangedCount,
    maxRankChange,
  });
}

export function buildAdaptationTransferEvidence({
  result,
  registration,
  modelCardState,
  selectedNodeId,
  selectedEvidence,
  policy,
  comparison,
}: BuildAdaptationTransferEvidenceInput): AdaptationTransferEvidenceState {
  if (!modelCardState || modelCardState.status === "loading") return unavailable("model_card_loading");
  if (modelCardState.status !== "ready" || !modelCardState.card) return unavailable("model_card_unavailable");
  const card = modelCardState.card;
  if (card.modelVersionId !== result.modelVersionId
    || card.modelVersionHash !== result.modelVersionHash
    || card.protocols.global.modelVersionId !== result.modelVersionId
    || card.protocols.global.modelVersionHash !== result.modelVersionHash
    || card.protocols.global.modelStateHash !== result.modelStateHash) return unavailable("identity_mismatch");
  if (!result.findings.length || result.findings.length !== result.totalFindings || result.totalFindings !== registration.task.nodeCount) return unavailable("invalid_routes");

  const expertLabels = new Map<string, string>();
  const trainingCounts = new Map<string, number>();
  card.trainingData.countries.forEach((country, index) => {
    const expert = `domain:${country}`;
    expertLabels.set(expert, `源域专家 ${String(index + 1).padStart(2, "0")}`);
    trainingCounts.set(expert, card.trainingData.nodeCountByCountry[country]);
  });
  expertLabels.set("null", "保守未知域");

  const routeWeight = new Map<string, number>();
  const selectedCount = new Map<string, number>();
  let textResponseSum = 0;
  let structureResponseSum = 0;
  let structureMissingCount = 0;
  for (const finding of result.findings) {
    const shared = finding.routes.filter((route) => route.expert === "shared");
    const routed = finding.routes.filter((route) => route.expert !== "shared");
    if (shared.length !== 1 || shared[0]?.weight !== 1 || routed.length !== 2
      || new Set(routed.map((route) => route.expert)).size !== 2
      || routed.some((route) => !expertLabels.has(route.expert) || !finiteUnit(route.weight))
      || Math.abs(routed.reduce((sum, route) => sum + route.weight, 0) - 1) > ROUTE_EPSILON
      || !finiteUnit(finding.modalityContribution.text) || !finiteUnit(finding.modalityContribution.structure)) return unavailable("invalid_routes");
    for (const route of routed) {
      routeWeight.set(route.expert, (routeWeight.get(route.expert) ?? 0) + route.weight);
      selectedCount.set(route.expert, (selectedCount.get(route.expert) ?? 0) + 1);
    }
    textResponseSum += finding.modalityContribution.text;
    structureResponseSum += finding.modalityContribution.structure;
    if (finding.structureMissing) structureMissingCount += 1;
  }

  const expertKeys = [...card.trainingData.countries.map((country) => `domain:${country}`), "null"];
  const experts = Object.freeze(expertKeys.map((key, index) => {
    const totalWeight = routeWeight.get(key) ?? 0;
    const count = selectedCount.get(key) ?? 0;
    return Object.freeze({
      id: key === "null" ? "null" as const : `source-${String(index + 1).padStart(2, "0")}` as const,
      label: expertLabels.get(key)!,
      routingMass: totalWeight / result.findings.length,
      coverage: count / result.findings.length,
      selectedNodeCount: count,
      averageWeight: totalWeight / result.findings.length,
      trainingNodeCount: trainingCounts.get(key) ?? null,
    });
  }));
  const routingTotal = experts.reduce((sum, expert) => sum + expert.routingMass, 0);
  if (Math.abs(routingTotal - 1) > ROUTE_EPSILON) return unavailable("invalid_routes");
  const domainExperts = experts.filter((expert) => expert.id !== "null");
  const primaryRoute = domainExperts.reduce<AdaptationExpertEvidence | null>((best, expert) => !best || expert.routingMass > best.routingMass ? expert : best, null);
  const selectedFinding = selectedNodeId ? result.findings.find((finding) => finding.nodeId === selectedNodeId) : undefined;

  return Object.freeze({
    status: "ready",
    value: Object.freeze({
      sharedRepresentation: "fixed",
      trainingNodeCount: card.trainingData.nodeCount,
      experts,
      activeSourceCount: domainExperts.filter((expert) => expert.routingMass > ROUTE_EPSILON).length,
      primaryRoute: primaryRoute?.routingMass ? primaryRoute : null,
      nullRoutingMass: experts.find((expert) => expert.id === "null")?.routingMass ?? 0,
      textResponseMean: textResponseSum / result.findings.length,
      structureResponseMean: structureResponseSum / result.findings.length,
      structureAvailableCount: result.findings.length - structureMissingCount,
      structureMissingCount,
      targetModalities: Object.freeze([...registration.task.modalities]),
      selectedObject: selectedObject(selectedFinding, expertLabels, selectedEvidence),
      calibration: calibrationEvidence(policy, comparison),
    }),
  });
}
