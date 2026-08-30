import type { AnalysisOverlay, GraphVersion } from "../types/graph";
import type { CoreRunBinding, CoreRunResult, CoreFinding } from "../types/core";
import { deepFreeze } from "./coreContracts";
import { matchRegisteredEdgeHashes } from "./coreEdgeIdentity";

export function buildCoreOverlay(
  graph: GraphVersion,
  binding: CoreRunBinding,
  result: CoreRunResult,
  finding: CoreFinding,
): AnalysisOverlay {
  if (
    graph.id !== binding.graphVersionId
    || result.runId !== binding.runId
    || result.requestHash !== binding.serverRequestHash
    || result.taskId !== binding.taskId
    || result.graphVersionId !== binding.graphVersionId
    || result.modelVersionId !== binding.modelVersionId
    || finding.taskId !== result.taskId
    || finding.graphVersionHash !== result.graphVersionHash
    || finding.modelVersion !== result.modelVersionId
    || finding.modelVersionHash !== result.modelVersionHash
    || !result.findings.some((candidate) => candidate.findingHash === finding.findingHash)
  ) {
    throw new Error("GFM_CORE_OVERLAY_BINDING_INVALID");
  }

  const nodeIds = new Set(graph.nodes.map((node) => node.id));
  const requiredEdgeHashes = new Set(finding.evidence.flatMap((evidence) => evidence.edgeIds));
  const edgeHashToLocalId = new Map(
    matchRegisteredEdgeHashes(graph, requiredEdgeHashes).map((entry) => (
      [entry.identity.edgeHash, entry.localEdgeId] as const
    )),
  );
  const nodeValues: Record<string, "subject" | "evidence"> = {};
  const edgeValues: Record<string, "evidence"> = {};
  for (const subjectId of finding.subjectIds) {
    if (nodeIds.has(subjectId)) nodeValues[subjectId] = "subject";
  }
  if (finding.score.edgeIdentity) {
    if (nodeIds.has(finding.score.edgeIdentity.sourceId)) {
      nodeValues[finding.score.edgeIdentity.sourceId] = "subject";
    }
    if (nodeIds.has(finding.score.edgeIdentity.targetId)) {
      nodeValues[finding.score.edgeIdentity.targetId] = "subject";
    }
  }
  for (const evidence of finding.evidence) {
    for (const nodeId of evidence.nodeIds) {
      if (nodeIds.has(nodeId) && nodeValues[nodeId] !== "subject") nodeValues[nodeId] = "evidence";
    }
    for (const edgeHash of evidence.edgeIds) {
      const localEdgeId = edgeHashToLocalId.get(edgeHash);
      if (localEdgeId) edgeValues[localEdgeId] = "evidence";
    }
  }

  return deepFreeze({
    id: `${graph.id}:core:${finding.findingHash}`,
    graphVersionId: graph.id,
    kind: "governance",
    nodeValues,
    edgeValues,
    legend: {
      title: "SocialGraph-FM Core 治理证据",
      items: [
        { value: "subject", label: "复核对象", color: "#7b63ff" },
        { value: "evidence", label: "现有结构证据", color: "#ef9a52" },
      ],
    },
    provenance: {
      engine: "gfm_core",
      algorithm: "core-finding-overlay",
      runId: binding.runId,
      resultHash: result.resultHash,
      findingHash: finding.findingHash,
      publicRequestHash: binding.publicRequestHash,
      serverRequestHash: binding.serverRequestHash,
      taskId: result.taskId,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: binding.modelVersionId,
      modelVersionHash: result.modelVersionHash,
    },
  });
}
