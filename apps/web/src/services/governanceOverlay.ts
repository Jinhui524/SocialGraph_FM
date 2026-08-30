import type { AnalysisOverlay, GraphVersion } from "../types/graph";
import type {
  GlobalModelNodeFinding,
  GlobalModelRunBinding,
  GlobalModelRunResult,
} from "../types/globalModel";
import type {
  GovernanceDerivation,
  GovernanceOnlineFinding,
  GovernanceOnlineResult,
  GovernanceOnlineRun,
} from "../types/governanceOnline";
import type { GovernanceGraphLens } from "./governanceReadableProjection";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";
import { governanceExactRelationKey } from "./graphPreviewPolicy";

export function buildGlobalModelOverlay(
  graph: GraphVersion,
  binding: GlobalModelRunBinding,
  result: GlobalModelRunResult,
  selected: GlobalModelNodeFinding,
): AnalysisOverlay {
  const graphHash = graph.datasetArtifact?.canonicalGraphHash ?? graph.contentHash;
  if (
    result.runId !== binding.runId
    || result.requestHash !== binding.serverRequestHash
    || result.protocol !== binding.protocol
    || result.graphVersionHash !== binding.graphVersionHash
    || result.modelVersionId !== binding.modelVersionId
    || result.modelVersionHash !== binding.modelVersionHash
    || graphHash !== result.graphVersionHash
    || !result.findings.some((finding) => finding.nodeId === selected.nodeId)
  ) throw new Error("GFM_GLOBAL_MODEL_OVERLAY_BINDING_INVALID");

  const nodeValues: Record<string, "candidate" | "subject"> = {};
  const riskBands: Record<string, "high" | "review" | "low"> = {};
  for (const finding of result.findings.slice(0, 100)) {
    nodeValues[finding.nodeId] = "candidate";
    riskBands[finding.nodeId] = finding.riskBand;
  }
  nodeValues[selected.nodeId] = "subject";
  riskBands[selected.nodeId] = selected.riskBand;
  return deepFreeze({
    id: `${graph.id}:governance:${result.resultHash}:${selected.nodeId}`,
    graphVersionId: graph.id,
    kind: "governance",
    nodeValues,
    edgeValues: {},
    presentation: { governanceLens: "risk", riskBands },
    legend: {
      title: "风险优先级",
      items: [
        { value: "risk-high", label: "高风险", color: "#E75E58" },
        { value: "risk-review", label: "建议复核", color: "#E5A53B" },
        { value: "risk-low", label: "低风险", color: "#3F8F8A" },
      ],
    },
    provenance: {
      engine: "socialgraph_fm_governance",
      algorithm: "coordination-risk-ranking",
      runId: binding.runId,
      resultHash: result.resultHash,
      findingHash: sha256Canonical({ resultHash: result.resultHash, nodeId: selected.nodeId }),
      publicRequestHash: binding.publicRequestHash,
      serverRequestHash: binding.serverRequestHash,
      taskId: result.taskId,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
    },
  });
}

function undirectedPair(source: string, target: string): string {
  return source.localeCompare(target) <= 0 ? `${source}\u0000${target}` : `${target}\u0000${source}`;
}

/** Presentation-only SocialGraph-FM Governance online overlay. It never mutates graph facts or
 * promotes a potential relation into a factual edge. */
export function buildGovernanceOnlineGovernanceOverlay(
  graph: GraphVersion,
  lens: GovernanceGraphLens,
  findings: readonly GovernanceOnlineFinding[],
  relations: readonly GovernanceDerivation[],
  links: readonly GovernanceDerivation[],
  run: GovernanceOnlineRun,
  result: GovernanceOnlineResult,
): AnalysisOverlay {
  const nodeValues: Record<string, string | number | boolean> = {};
  if (lens === "risk") {
    for (const node of graph.nodes) nodeValues[node.id] = "context";
    for (const finding of findings) nodeValues[finding.nodeId] = `risk-${finding.riskBand}`;
  } else if (lens === "community") {
    for (const finding of findings) nodeValues[finding.nodeId] = finding.communityId ?? "未分组";
  } else if (lens === "router") {
    for (const finding of findings) {
      nodeValues[finding.nodeId] = finding.routes.find((route) => route.expert !== "shared")?.expert ?? "null";
    }
  }

  const edgeValues: Record<string, string | number | boolean> = {};
  if (lens === "relations") {
    for (const edge of graph.edges) edgeValues[edge.id] = "factual";
  } else if (lens === "risk") {
    const bandByNode = new Map(findings.map((finding) => [finding.nodeId, finding.riskBand]));
    const evidenceBandByPair = new Map<string, "evidence-high" | "evidence-review" | "context">();
    for (const relation of relations) {
      const source = relation.source ?? relation.nodeIds[0];
      const target = relation.target ?? relation.nodeIds[1];
      if (!source || !target || !relation.factual) continue;
      const bands = [bandByNode.get(source), bandByNode.get(target)];
      const value = bands.includes("high")
        ? "evidence-high"
        : bands.includes("review") ? "evidence-review" : "context";
      evidenceBandByPair.set(undirectedPair(source, target), value);
    }
    for (const edge of graph.edges) {
      const value = evidenceBandByPair.get(undirectedPair(edge.source, edge.target));
      if (value) edgeValues[edge.id] = value;
    }
  }

  const relationshipLens = lens === "relations";
  const riskBands = Object.freeze(Object.fromEntries(
    findings.map((finding) => [finding.nodeId, finding.riskBand]),
  ));
  return Object.freeze({
    id: `governance:${run.runId}:${lens}`,
    graphVersionId: graph.id,
    kind: lens === "community" || lens === "router" ? "community" : "governance",
    nodeValues: Object.freeze(nodeValues),
    edgeValues: Object.freeze(edgeValues),
    candidateEdges: relationshipLens ? Object.freeze(links.slice(0, 500).flatMap((link) => {
      const sourceId = link.source ?? link.nodeIds[0];
      const targetId = link.target ?? link.nodeIds[1];
      return sourceId && targetId ? [Object.freeze({
        id: `__socialgraph_research_candidate__${link.id}`,
        sourceId,
        targetId,
        directed: false,
        exactRelationKey: governanceExactRelationKey(sourceId, targetId, link.modalities),
      })] : [];
    })) : undefined,
    presentation: Object.freeze({
      ...(lens === "risk" || relationshipLens ? { governanceLens: lens } : {}),
      riskBands,
    }),
    legend: {
      title: lens === "risk" ? "风险优先级" : lens === "community" ? "协同群组" : lens === "router" ? "专家路径" : "事实关系 / 潜在线索",
      items: lens === "risk"
        ? [
            { value: "risk-low", label: "低风险", color: "#3F8F8A" },
            { value: "risk-review", label: "建议复核", color: "#E5A53B" },
            { value: "risk-high", label: "高风险", color: "#E75E58" },
          ]
        : relationshipLens ? [
            { value: "factual", label: "事实关系", color: "#5F7896" },
            { value: "candidate", label: "潜在线索", color: "#7659EF" },
          ] : [],
    },
    provenance: {
      engine: "socialgraph-governance",
      algorithm: lens,
      runId: run.runId,
      resultHash: result.resultHash,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
    },
  });
}
