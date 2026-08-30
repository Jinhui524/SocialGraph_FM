import type {
  CoreConfidenceEvidence,
  CoreEvidenceItem,
  CoreFindingType,
  CoreModelScore,
  CoreRunBinding,
  CoreRunResult,
  CoreSimilarCase,
  CoreTaskId,
  CoreFinding,
} from "../types/core";
import { canonicalJson, compareUnicodeCodePoints, sha256Canonical } from "./graphIdentity";
import { deepFreeze } from "./coreContracts";
import type { LocalReviewRecord } from "./localReview";

export interface CoreReport {
  readonly schemaVersion: "socialgraph-fm.core-report/2.0";
  readonly generatedAt: string;
  readonly taskId: CoreTaskId;
  readonly taskLabel: string;
  readonly runId: string;
  readonly graphVersionId: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly publicRequestHash: string;
  readonly serverRequestHash: string;
  readonly resultHash: string;
  readonly findingHash: string;
  readonly reviewStatus: "pending-human-review";
  readonly score: CoreModelScore;
  readonly calibratedConfidence: CoreConfidenceEvidence;
  readonly evidence: readonly CoreEvidenceItem[];
  readonly similarCases: readonly CoreSimilarCase[];
  readonly limitations: readonly string[];
  readonly localReviews: readonly LocalReviewRecord[];
  readonly safetyStatements: readonly string[];
  readonly reportHash: string;
}

export function coreFindingLabel(findingType: CoreFindingType): string {
  if (findingType === "core-collaboration-completion") return "静态关系补全建议";
  if (findingType === "node-risk-candidate") return "风险候选（待人工复核）";
  if (findingType === "signed-relation-review") return "有符号关系复核（待人工复核）";
  return "社区韧性结构复核";
}

export function buildCoreReport(input: {
  readonly binding: CoreRunBinding;
  readonly result: CoreRunResult;
  readonly finding: CoreFinding;
  readonly reviews: readonly LocalReviewRecord[];
}): CoreReport {
  const { binding, result, finding } = input;
  if (
    result.runId !== binding.runId
    || result.requestHash !== binding.serverRequestHash
    || result.taskId !== binding.taskId
    || result.graphVersionId !== binding.graphVersionId
    || result.modelVersionId !== binding.modelVersionId
    || finding.taskId !== result.taskId
    || finding.graphVersionHash !== result.graphVersionHash
    || finding.modelVersion !== result.modelVersionId
    || finding.modelVersionHash !== result.modelVersionHash
    || !result.findings.some((candidate) => candidate.findingHash === finding.findingHash)
    || input.reviews.some((review) => (
      review.findingHash !== finding.findingHash
      || review.runId !== binding.runId
      || review.resultHash !== result.resultHash
      || review.graphVersionId !== binding.graphVersionId
    ))
  ) throw new Error("GFM_CORE_REPORT_BINDING_INVALID");

  const localReviews = [...input.reviews].sort((left, right) => (
    Date.parse(left.reviewedAt) - Date.parse(right.reviewedAt)
    || compareUnicodeCodePoints(left.recordHash, right.recordHash)
  ));
  const generatedAt = [result.completedAt, ...localReviews.map((review) => review.reviewedAt)]
    .reduce((latest, candidate) => (
      Date.parse(candidate) > Date.parse(latest) ? candidate : latest
    ));
  const confidenceSafety = finding.calibratedConfidence.schemaVersion
    === "socialgraph-fm.core-regression-confidence-interval/1.0"
    ? "验证残差覆盖描述回归区间在验证集上的覆盖，不是概率，必须待人工复核。"
    : "校准置信度不是违规、风险或事实为真的概率，必须由人工结合上下文复核。";
  const payload = {
    schemaVersion: "socialgraph-fm.core-report/2.0" as const,
    generatedAt,
    taskId: result.taskId,
    taskLabel: coreFindingLabel(finding.findingType),
    runId: binding.runId,
    graphVersionId: binding.graphVersionId,
    graphVersionHash: result.graphVersionHash,
    modelVersionId: binding.modelVersionId,
    modelVersionHash: result.modelVersionHash,
    publicRequestHash: binding.publicRequestHash,
    serverRequestHash: binding.serverRequestHash,
    resultHash: result.resultHash,
    findingHash: finding.findingHash,
    reviewStatus: finding.reviewStatus,
    score: finding.score,
    calibratedConfidence: finding.calibratedConfidence,
    evidence: finding.evidence,
    similarCases: finding.similarCases,
    limitations: finding.limitations,
    localReviews,
    safetyStatements: [
      "不授权自动处罚或执法",
      "本结果仅分析静态图结构，属于非因果复核，不预测未来事件。",
      confidenceSafety,
      "浏览器可重算 publicRequestHash；serverRequestHash 绑定包含隐藏授权信息的服务端 envelope，浏览器仅核对其跨状态与结果一致性。",
    ],
  };
  return deepFreeze({ ...payload, reportHash: sha256Canonical(payload) });
}

export function serializeCoreReportJson(report: CoreReport): string {
  return canonicalJson(report);
}

function indentedCanonicalJson(value: unknown): string {
  return canonicalJson(value)
    .split("\n")
    .map((line) => `    ${line}`)
    .join("\n");
}

export function serializeCoreReportMarkdown(report: CoreReport): string {
  const provenance = {
    schemaVersion: report.schemaVersion,
    generatedAt: report.generatedAt,
    taskId: report.taskId,
    taskLabel: report.taskLabel,
    runId: report.runId,
    graphVersionId: report.graphVersionId,
    graphVersionHash: report.graphVersionHash,
    modelVersionId: report.modelVersionId,
    modelVersionHash: report.modelVersionHash,
    publicRequestHash: report.publicRequestHash,
    serverRequestHash: report.serverRequestHash,
    resultHash: report.resultHash,
    findingHash: report.findingHash,
    reviewStatus: report.reviewStatus,
  };
  const lines = [
    `# ${report.taskLabel}`,
    "",
    "## 不可变来源与哈希",
    "",
    indentedCanonicalJson(provenance),
    "",
    "publicRequestHash 由浏览器对公共请求重算；serverRequestHash 来自服务端隐藏 envelope，浏览器仅校验其跨状态与结果绑定。",
    "",
    report.calibratedConfidence.schemaVersion === "socialgraph-fm.core-regression-confidence-interval/1.0"
      ? "## 回归区间与验证残差覆盖（非概率）"
      : "## 模型分数与完整校准来源",
    "",
    indentedCanonicalJson({
      score: report.score,
      calibratedConfidence: report.calibratedConfidence,
    }),
    "",
    "## 完整证据记录",
    "",
    ...report.evidence.flatMap((item, index) => [
      `### 证据 ${index + 1}`,
      "",
      indentedCanonicalJson(item),
      "",
    ]),
    "## 完整相似结构案例",
    "",
    ...(report.similarCases.length
      ? report.similarCases.flatMap((item, index) => [
        `### 相似案例 ${index + 1}`,
        "",
        indentedCanonicalJson(item),
        "",
      ])
      : ["- 无"]),
    "## 限制与安全说明",
    "",
    indentedCanonicalJson(report.limitations),
    "",
    ...report.safetyStatements.map((item) => `- ${item}`),
    "",
    "## 完整本地人工复核记录",
    "",
    ...(report.localReviews.length
      ? report.localReviews.flatMap((review, index) => [
        `### 本地复核 ${index + 1}`,
        "",
        indentedCanonicalJson(review),
        "",
      ])
      : ["- 无；服务器发现仍为 pending-human-review。"]),
    "",
    "## 报告哈希",
    "",
    indentedCanonicalJson(report.reportHash),
    "",
  ];
  return lines.join("\n");
}
