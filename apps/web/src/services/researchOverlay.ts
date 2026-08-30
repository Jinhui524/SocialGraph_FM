import type { AnalysisOverlay, GraphVersion } from "../types/graph";
import type {
  ResearchFinding,
  ResearchRunBinding,
  ResearchRunResult,
  ResearchTaskId,
} from "../types/research";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";

const TASK_COLOURS: Readonly<Record<ResearchTaskId, string>> = {
  "research.content_policy_review": "#7659ef",
  "research.account_risk_review": "#dd6b5d",
  "research.signed_relation_review": "#d18a33",
  "core.collaboration_completion": "#2297a5",
};

export function buildResearchOverlay(
  graph: GraphVersion,
  binding: ResearchRunBinding,
  result: ResearchRunResult,
  finding: ResearchFinding,
  expectedGraphHash?: string,
): AnalysisOverlay {
  const knownHash = expectedGraphHash ?? graph.datasetArtifact?.canonicalGraphHash ?? graph.contentHash;
  if (
    graph.id !== binding.graphVersionId
    || result.runId !== binding.runId
    || result.requestHash !== binding.serverRequestHash
    || result.graphVersionId !== graph.id
    || result.modelVersionId !== binding.modelVersionId
    || result.taskId !== binding.taskId
    || (knownHash && knownHash !== result.graphVersionHash)
    || !result.findings.some((candidate) => candidate.id === finding.id)
  ) throw new Error("GFM_RESEARCH_OVERLAY_BINDING_INVALID");

  const nodeValues: Record<string, "subject"> = {};
  for (const nodeId of finding.entityIds) {
    nodeValues[nodeId] = "subject";
  }
  const candidateEdges = finding.entityType === "node-pair" && finding.entityIds.length === 2
    ? [{
        id: `__socialgraph_research_candidate__${sha256Canonical({ resultHash: result.resultHash, findingId: finding.id }).slice(0, 20)}`,
        sourceId: finding.entityIds[0]!,
        targetId: finding.entityIds[1]!,
        directed: false,
      }]
    : finding.entityType === "directed-edge" && finding.entityIds.length === 2
      ? [{
          id: `__socialgraph_research_candidate__${sha256Canonical({ resultHash: result.resultHash, findingId: finding.id }).slice(0, 20)}`,
          sourceId: finding.entityIds[0]!,
          targetId: finding.entityIds[1]!,
          directed: true,
        }]
      : [];
  const colour = TASK_COLOURS[result.taskId];
  return deepFreeze({
    id: `${graph.id}:research-governance:${result.resultHash}:${finding.id}`,
    graphVersionId: graph.id,
    kind: "governance",
    nodeValues,
    edgeValues: {},
    candidateEdges,
    legend: {
      title: "SocialGraph-FM Research 推断覆盖层",
      items: [
        { value: "subject", label: "复核对象", color: colour },
        ...(candidateEdges.length
          ? [{ value: "candidate", label: "模型候选关系（非图事实）", color: colour }]
          : []),
      ],
    },
    provenance: {
      engine: "gfm_research",
      algorithm: "research-governance-finding-overlay",
      runId: binding.runId,
      resultHash: result.resultHash,
      findingHash: sha256Canonical({ resultHash: result.resultHash, findingId: finding.id }),
      publicRequestHash: binding.publicRequestHash,
      serverRequestHash: binding.serverRequestHash,
      taskId: result.taskId,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
    },
  });
}
