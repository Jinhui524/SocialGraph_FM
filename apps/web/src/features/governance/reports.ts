import { governanceAccountLabel, governanceModalityLabel } from "../../services/governancePresentation";
import type { AnalysisRun } from "../../types/graph";
import type { GovernanceOnlinePreview, GovernanceOnlineResult } from "../../types/governanceOnline";

export function resultDescription(run?: AnalysisRun) {
  if (!run?.result) return null;
  const result = run.result;
  if (result.kind === "overview") {
    const names = result.topDegree.slice(0, 3).map((item) => `${item.label}（度 ${item.degree}）`);
    return `${run.scope ? `本次范围 ${run.scope.nodeCount} 个节点、${run.scope.edgeCount} 条关系。` : ""}图中共有 ${result.summary.nodeCount} 个节点、${result.summary.edgeCount} 条关系。连接最活跃的节点为 ${names.join("、") || "暂无"}。`;
  }
  if (result.kind === "centrality") {
    return `度数排名前列：${result.ranking
      .slice(0, 5)
      .map((item) => `${item.label} ${item.degree}`)
      .join("、")}。`;
  }
  if (result.kind === "bridge_detection") {
    return result.articulationPoints.length
      ? `识别到 ${result.articulationPoints.length} 个割点：${result.articulationPoints.join("、")}。移除这些节点可能使网络分裂。`
      : "当前网络中未识别到割点，结构没有单一节点失效导致的直接断裂风险。";
  }
  if (result.kind === "connected_components") {
    return `网络包含 ${result.components.length} 个连通分量，最大分量覆盖 ${Math.max(
      0,
      ...result.components.map((component) => component.length),
    )} 个节点。`;
  }
  if (result.kind === "community") {
    return `Louvain 基线识别到 ${result.communities.length} 个社区，模块度为 ${result.modularity.toFixed(3)}。`;
  }
  return result.message;
}
const HUMAN_REVIEW_GUIDANCE = [
  "### 人工复核建议",
  "",
  "- 选择候选",
  "- 核对关系与邻域",
  "- 加入研判单并记录确认、驳回或待定理由",
].join("\n");

export function ensureHumanReviewGuidance(markdown: string): string {
  if (/^#{1,6}\s+人工复核建议\s*$/mu.test(markdown)) return markdown;
  return `${markdown.trimEnd()}\n\n${HUMAN_REVIEW_GUIDANCE}`;
}

export function deterministicGovernanceCompletionReport(result: GovernanceOnlineResult, preview?: GovernanceOnlinePreview): string {
  const candidates = [...result.findings]
    .filter((finding) => finding.riskBand === "high" || finding.riskBand === "review")
    .sort((left, right) => left.rank - right.rank)
    .slice(0, 5);
  const labels = new Map(preview?.nodes.map((node) => [node.id, governanceAccountLabel(node.label, node.id)]) ?? []);
  const candidateIds = new Set(candidates.map((finding) => finding.nodeId));
  const factualRelations = preview?.edges
    .filter((edge) => candidateIds.has(edge.source) || candidateIds.has(edge.target))
    .slice(0, 3) ?? [];
  return ensureHumanReviewGuidance([
    "## 模型分析结果",
    "",
    "模型已完成风险排序与关系结构梳理，以下内容用于安排人工复核，不代表自动定性。",
    "",
    "### 高关注账号",
    ...(candidates.length ? candidates.map((finding) => `- **${governanceAccountLabel(finding.label, finding.nodeId)}** · 原模型排名 #${finding.rank} · ${finding.riskBand === "high" ? "高风险候选" : "建议复核"}${finding.communityId ? ` · 群组 ${finding.communityId}` : ""}`) : ["- 当前结果未返回高风险或建议复核对象。"]),
    "",
    "### 重点事实关系",
    ...(factualRelations.length ? factualRelations.map((edge) => `- ${labels.get(edge.source) ?? governanceAccountLabel(undefined, edge.source)} — ${labels.get(edge.target) ?? governanceAccountLabel(undefined, edge.target)} · ${edge.modalities.map(governanceModalityLabel).join("、") || "关系类型待核"}`) : ["- 当前预览未返回与高关注账号相连的事实关系。"]),
    "",
    "群组和潜在线索可在治理应用中继续核对；潜在线索不得作为已登记事实关系使用。",
    "模型排序用于安排核验顺序，不构成人工结论或处置依据。",
  ].join("\n"));
}

export function buildAnalysisResultMarkdown(run?: AnalysisRun): string | null {
  const description = resultDescription(run);
  if (!description) return null;
  const markdown = `### 分析结果\n\n${description}`;
  return run?.status === "succeeded" && run.engine !== "unavailable" && run.result?.kind !== "unavailable"
    ? ensureHumanReviewGuidance(markdown)
    : markdown;
}
