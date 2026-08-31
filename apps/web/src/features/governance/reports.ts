import type { AnalysisRun } from "../../types/graph";

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

export function buildAnalysisResultMarkdown(run?: AnalysisRun): string | null {
  const description = resultDescription(run);
  if (!description) return null;
  const markdown = `### 分析结果\n\n${description}`;
  return run?.status === "succeeded" && run.engine !== "unavailable" && run.result?.kind !== "unavailable"
    ? ensureHumanReviewGuidance(markdown)
    : markdown;
}
