import type {
  CoreEntityType,
  CoreFindingType,
  CoreRunBinding,
  CoreRunResult,
  CoreTaskId,
  CoreFinding,
} from "../../types/core";
import { parseCoreRunResult } from "../../services/coreContracts";
import { sha256Canonical } from "../../services/graphIdentity";

const graphVersionHash = "d".repeat(64);
const modelVersionHash = "5".repeat(64);
const serverRequestHash = "1".repeat(64);

function withHash<T extends Record<string, unknown>, K extends string>(
  value: T,
  field: K,
): T & Record<K, string> {
  return { ...value, [field]: sha256Canonical(value) } as T & Record<K, string>;
}

export interface ValidatedCoreFixture {
  readonly binding: CoreRunBinding;
  readonly result: CoreRunResult;
  readonly finding: CoreFinding;
}

export function createValidatedCoreFixture(options: {
  readonly graphVersionId: string;
  readonly taskId?: CoreTaskId;
  readonly findingType?: CoreFindingType;
  readonly entityType?: CoreEntityType;
  readonly subjectIds?: readonly string[];
  readonly pathNodeIds?: readonly string[];
  readonly pathEdgeIds?: readonly string[];
  readonly includeSimilarCase?: boolean;
}): ValidatedCoreFixture {
  const taskId = options.taskId ?? "core.risk_and_trust_review";
  const findingType = options.findingType ?? "node-risk-candidate";
  const entityType = options.entityType ?? "node";
  const subjectIds = [...(options.subjectIds ?? ["a"])] as string[];
  const score = withHash({
    schemaVersion: "socialgraph-fm.core-model-score/2.0",
    taskId,
    entityType,
    entityIds: subjectIds,
    score: 0.25,
    graphVersionHash,
    modelVersion: "socialgraph-fm-core/review",
    modelVersionHash,
    edgeIdentity: null,
  }, "scoreHash");
  const confidence = taskId === "core.community_resilience_review"
    ? withHash({
      schemaVersion: "socialgraph-fm.core-regression-confidence-interval/1.0",
      pointEstimate: score.score,
      lowerBound: 0.1,
      upperBound: 0.4,
      coverage: 0.9,
      validationCount: 32,
      scoreHash: score.scoreHash,
      taskId,
      entityType: "community",
      entityIds: subjectIds,
      graphVersionHash,
      modelVersion: "socialgraph-fm-core/review",
      modelVersionHash,
      confidenceVersion: "residual-interval/1",
      method: "validation-residual-interval",
      confidenceArtifactHash: "b".repeat(64),
      confidenceProtocolHash: "c".repeat(64),
    }, "confidenceHash")
    : withHash({
      schemaVersion: "socialgraph-fm.core-calibrated-confidence/2.0",
      value: 0.6,
      scoreHash: score.scoreHash,
      taskId,
      entityType,
      entityIds: subjectIds,
      graphVersionHash,
      modelVersion: "socialgraph-fm-core/review",
      modelVersionHash,
      calibrationVersion: "calibration/1",
      method: "sigmoid",
      calibrationArtifactHash: "b".repeat(64),
      calibrationProtocolHash: "c".repeat(64),
    }, "confidenceHash");
  const modelEvidence = withHash({
    schemaVersion: "socialgraph-fm.core-evidence/2.0",
    metric: "registered_model.score-reference",
    valueCanonicalJson: "{}",
    graphVersionHash,
    sourceType: "registered-model-output",
    nodeIds: subjectIds,
    edgeIds: [],
    algorithmConfigHash: null,
    modelVersionHash,
    modelVersion: "socialgraph-fm-core/review",
    modelScoreHash: score.scoreHash,
    modelTaskId: taskId,
    modelEntityType: entityType,
    modelEntityIds: subjectIds,
    limitations: ["The score is a registered model output, not a graph fact or decision."],
  }, "evidenceHash");
  const evidence: Record<string, unknown>[] = [modelEvidence];
  if (options.pathNodeIds?.length || options.pathEdgeIds?.length) {
    evidence.push(withHash({
      schemaVersion: "socialgraph-fm.core-evidence/2.0",
      metric: "core_graph.existing-path",
      valueCanonicalJson: "{\"length\":2}",
      graphVersionHash,
      sourceType: "deterministic-graph-algorithm",
      nodeIds: [...(options.pathNodeIds ?? [])],
      edgeIds: [...(options.pathEdgeIds ?? [])],
      algorithmConfigHash: "6".repeat(64),
      modelVersionHash: null,
      modelVersion: null,
      modelScoreHash: null,
      modelTaskId: null,
      modelEntityType: null,
      modelEntityIds: null,
      limitations: ["Path evidence is registered relation-completion context, not a future-event forecast."],
    }, "evidenceHash"));
  }
  const similarCases = options.includeSimilarCase ? [withHash({
    schemaVersion: "socialgraph-fm.core-similar-case/2.0",
    structuralRecordHash: "7".repeat(64),
    similarity: 0.75,
    sourceGraphVersionHash: "8".repeat(64),
    sourceEntityIds: subjectIds,
    sourceKind: entityType === "community" ? "community" : "node",
    modelVersion: "socialgraph-fm-core/review",
    modelVersionHash,
    representation: "embedding",
    queryHash: "9".repeat(64),
    representationSchema: "socialgraph-fm.core-structural-record/2.0",
  }, "similarCaseHash")] : [];
  const finding = withHash({
    schemaVersion: "socialgraph-fm.core-finding/2.0",
    taskId,
    findingType,
    subjectIds,
    score,
    calibratedConfidence: confidence,
    evidence,
    similarCases,
    graphVersionHash,
    modelVersion: "socialgraph-fm-core/review",
    modelVersionHash,
    limitations: [
      "Manual human review is required; no automatic sanction or action is authorized.",
      "This finding is non-causal and does not predict future events.",
      ...(taskId === "core.community_resilience_review"
        ? ["The resilience interval reports validation residual coverage, not a probability."]
        : []),
    ],
    reviewStatus: "pending-human-review",
  }, "findingHash");
  const runId = "00000000-0000-0000-0000-000000000001";
  const resultPayload = withHash({
    schemaVersion: "socialgraph-fm.core-run-result/2.0",
    runId,
    requestHash: serverRequestHash,
    taskId,
    graphVersionId: options.graphVersionId,
    graphVersionHash,
    modelVersionId: "socialgraph-fm-core/review",
    modelVersionHash,
    findings: [finding],
    completedAt: "2026-08-15T00:00:01.000000Z",
  }, "resultHash");
  const binding: CoreRunBinding = {
    runId,
    publicRequestHash: "9".repeat(64),
    serverRequestHash,
    taskId,
    graphVersionId: options.graphVersionId,
    modelVersionId: "socialgraph-fm-core/review",
  };
  const result = parseCoreRunResult(resultPayload, binding);
  return { binding, result, finding: result.findings[0]! };
}
