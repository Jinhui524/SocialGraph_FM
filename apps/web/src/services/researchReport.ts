import type {
  ResearchFinding,
  ResearchRunBinding,
  ResearchRunResult,
  ResearchTaskId,
} from "../types/research";
import { deepFreeze } from "./coreContracts";
import { canonicalJson, sha256Canonical } from "./graphIdentity";
import type { LocalReviewRecord } from "./localReview";

const TASK_LABELS: Readonly<Record<ResearchTaskId, string>> = {
  "research.content_policy_review": "内容策略复核",
  "research.account_risk_review": "历史账号状态复核",
  "research.signed_relation_review": "治理关系立场复核",
  "core.collaboration_completion": "协作关系候选",
};

export interface ResearchCoreReport {
  readonly schemaVersion: "socialgraph-fm.research-report/1.0";
  readonly releaseLabel: "SocialGraph-FM Research";
  readonly preliminary: true;
  readonly seed: 1729;
  readonly taskId: ResearchTaskId;
  readonly taskLabel: string;
  readonly runId: string;
  readonly graphVersionId: string;
  readonly graphVersionHash: string;
  readonly modelVersionId: string;
  readonly modelVersionHash: string;
  readonly publicRequestHash: string;
  readonly serverRequestHash: string;
  readonly resultHash: string;
  readonly findingId: string;
  readonly localFindingHash: string;
  readonly calibrationStatus: ResearchRunResult["calibrationStatus"];
  readonly finding: ResearchFinding;
  readonly localReviews: readonly LocalReviewRecord[];
  readonly safetyStatements: readonly string[];
  readonly reportHash: string;
}

export function researchFindingHash(resultHash: string, findingId: string): string {
  return sha256Canonical({ schemaVersion: "research-finding-local-key/1", resultHash, findingId });
}

export function buildResearchCoreReport(input: {
  readonly binding: ResearchRunBinding;
  readonly result: ResearchRunResult;
  readonly finding: ResearchFinding;
  readonly reviews: readonly LocalReviewRecord[];
}): ResearchCoreReport {
  const { binding, result, finding } = input;
  const localFindingHash = researchFindingHash(result.resultHash, finding.id);
  if (
    result.runId !== binding.runId
    || result.requestHash !== binding.serverRequestHash
    || result.graphVersionId !== binding.graphVersionId
    || result.modelVersionId !== binding.modelVersionId
    || result.taskId !== binding.taskId
    || !result.findings.some((candidate) => candidate.id === finding.id)
    || input.reviews.some((review) => (
      review.findingHash !== localFindingHash
      || review.runId !== result.runId
      || review.resultHash !== result.resultHash
      || review.graphVersionId !== result.graphVersionId
    ))
  ) throw new Error("GFM_RESEARCH_REPORT_BINDING_INVALID");
  const payload = {
    schemaVersion: "socialgraph-fm.research-report/1.0" as const,
    releaseLabel: "SocialGraph-FM Research" as const,
    preliminary: true as const,
    seed: 1729 as const,
    taskId: result.taskId,
    taskLabel: TASK_LABELS[result.taskId],
    runId: result.runId,
    graphVersionId: result.graphVersionId,
    graphVersionHash: result.graphVersionHash,
    modelVersionId: result.modelVersionId,
    modelVersionHash: result.modelVersionHash,
    publicRequestHash: binding.publicRequestHash,
    serverRequestHash: binding.serverRequestHash,
    resultHash: result.resultHash,
    findingId: finding.id,
    localFindingHash,
    calibrationStatus: result.calibrationStatus,
    finding,
    localReviews: [...input.reviews],
    safetyStatements: [
      "单随机种子初步结果，不代表稳定泛化结论。",
      "模型排序不是图事实，不授权自动处罚、封禁或执法。",
      "结果只绑定当前静态图、登记模型与不可变哈希。",
    ],
  };
  return deepFreeze({ ...payload, reportHash: sha256Canonical(payload) });
}

export function serializeResearchCoreReportJson(report: ResearchCoreReport): string {
  return canonicalJson(report);
}

export function serializeResearchCoreReportMarkdown(report: ResearchCoreReport): string {
  return [
    `# ${report.taskLabel}`,
    "",
    `SocialGraph-FM Research · seed ${report.seed} · 单次实验初步结果`,
    "",
    "## 排名发现",
    "",
    `    ${canonicalJson(report.finding)}`,
    "",
    "## 不可变来源",
    "",
    `    ${canonicalJson({
      runId: report.runId,
      graphVersionId: report.graphVersionId,
      graphVersionHash: report.graphVersionHash,
      modelVersionId: report.modelVersionId,
      modelVersionHash: report.modelVersionHash,
      publicRequestHash: report.publicRequestHash,
      serverRequestHash: report.serverRequestHash,
      resultHash: report.resultHash,
    })}`,
    "",
    "## 限制",
    "",
    ...report.safetyStatements.map((item) => `- ${item}`),
    ...report.finding.limitations.map((item) => `- ${item}`),
    "",
    "## 本地人工复核",
    "",
    report.localReviews.length ? `    ${canonicalJson(report.localReviews)}` : "- 尚无本地人工复核记录。",
    "",
    "## 报告哈希",
    "",
    `    ${report.reportHash}`,
    "",
  ].join("\n");
}
