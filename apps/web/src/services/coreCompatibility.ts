import type { GraphVersion } from "../types/graph";
import type { CoreModelCapability, CoreRunRequest, CoreTaskId } from "../types/core";
import { deepFreeze, parseCoreRunRequest } from "./coreContracts";
import { registeredEdgeIdentityForLocalId } from "./coreEdgeIdentity";
import { compareUnicodeCodePoints } from "./graphIdentity";

const CORE_BUNDLE_SCHEMA = "socialgraph-fm.core-graph-bundle/2.0";

export interface CoreCompatibilityIssue {
  readonly code: string;
  readonly message: string;
}

export interface CoreCompatibility {
  readonly state: "blocked" | "candidate" | "server-validation-required";
  readonly runnable: boolean;
  readonly modelValidated: boolean;
  readonly authoritativeAtPost: true;
  readonly blockers: readonly CoreCompatibilityIssue[];
  readonly notice: string;
}

export type CoreTargetSelection =
  | { readonly kind: "community"; readonly communityId: string }
  | { readonly kind: "node"; readonly nodeId: string }
  | { readonly kind: "edge"; readonly edgeId: string }
  | { readonly kind: "node-pair"; readonly sourceId: string; readonly targetId: string };

export interface CoreRequestBuildInput {
  readonly graph: GraphVersion;
  readonly modelVersionId: string;
  readonly taskId: CoreTaskId;
  readonly selection: CoreTargetSelection;
  readonly topKSimilarCases: number;
  readonly candidateLimit?: number;
}

export interface BuiltCollaborationRunRequest {
  readonly request: CoreRunRequest;
  readonly relationState: "missing";
  readonly existingEdgeIds: readonly string[];
}

export interface CollaborationPairInspection {
  readonly relationState: "missing" | "recorded";
  readonly existingEdgeIds: readonly string[];
}

function issue(code: string, message: string): CoreCompatibilityIssue {
  return Object.freeze({ code, message });
}

export function assessCoreCompatibility(
  graph: GraphVersion,
  model: CoreModelCapability,
  taskId: CoreTaskId,
): CoreCompatibility {
  const blockers: CoreCompatibilityIssue[] = [];
  if (!model.tasks.includes(taskId)) {
    blockers.push(issue("TASK_NOT_SUPPORTED", "该模型未登记当前治理任务。"));
  }
  if (!model.graphSchemaVersions.includes(CORE_BUNDLE_SCHEMA)) {
    blockers.push(issue("CORE_BUNDLE_SCHEMA_NOT_SUPPORTED", "该模型未登记 Core graph contract 2.0。"));
  }
  if (graph.summary.nodeCount > model.maxNodes) {
    blockers.push(issue("MODEL_NODE_CAP_EXCEEDED", "当前图节点数超过模型上限。"));
  }
  if (graph.summary.edgeCount > model.maxEdges) {
    blockers.push(issue("MODEL_EDGE_CAP_EXCEEDED", "当前图边数超过模型上限。"));
  }
  if (
    graph.datasetArtifact?.scope === "projection"
    || graph.summary.nodeCount !== graph.nodes.length
    || graph.summary.edgeCount !== graph.edges.length
  ) {
    blockers.push(issue("GRAPH_VERSION_INCOMPLETE", "当前 GraphVersion 不是完整、非截断图事实。"));
  }
  if (blockers.length > 0) {
    return deepFreeze({
      state: "blocked",
      runnable: false,
      modelValidated: model.state === "accepted" || model.state === "servingReady",
      authoritativeAtPost: true,
      blockers,
      notice: "浏览器检查仅用于阻断明显不兼容；服务端 POST 合同检查才是权威结论。",
    });
  }
  if (model.state === "accepted") {
    return deepFreeze({
      state: "candidate",
      runnable: false,
      modelValidated: true,
      authoritativeAtPost: true,
      blockers,
      notice: "模型已通过登记验收，但尚未 servingReady，不能发起运行。",
    });
  }
  return deepFreeze({
    state: "server-validation-required",
    runnable: true,
    modelValidated: true,
    authoritativeAtPost: true,
    blockers,
    notice: "本地只确认静态 schema、任务和规模；特征与制品合同由服务端 POST 权威校验。",
  });
}

function existingNode(graph: GraphVersion, nodeId: string) {
  const node = graph.nodes.find((candidate) => candidate.id === nodeId);
  if (!node) throw new Error("GFM_CORE_TARGET_NOT_FOUND");
  return node;
}

function existingEdge(graph: GraphVersion, edgeId: string) {
  const edge = graph.edges.find((candidate) => candidate.id === edgeId);
  if (!edge) throw new Error("GFM_CORE_TARGET_NOT_FOUND");
  return edge;
}

export function buildCoreRunRequest(input: CoreRequestBuildInput): CoreRunRequest {
  const base = {
    schemaVersion: "socialgraph-fm.core-run-request/2.0" as const,
    graphVersionId: input.graph.id,
    taskId: input.taskId,
    modelVersionId: input.modelVersionId,
  };
  if (input.taskId === "core.community_resilience_review") {
    if (input.selection.kind !== "community") throw new Error("GFM_CORE_TASK_TARGET_MISMATCH");
    const node = existingNode(input.graph, input.selection.communityId);
    if (node.type?.trim().toLocaleLowerCase("en-US") !== "community") {
      throw new Error("GFM_CORE_TARGET_NOT_REGISTERED_COMMUNITY");
    }
    return parseCoreRunRequest({
      ...base,
      targetScope: { kind: "community", communityIds: [node.id] },
      parameters: { kind: "community-resilience", topKSimilarCases: input.topKSimilarCases },
    });
  }
  if (input.taskId === "core.risk_and_trust_review") {
    if (input.selection.kind !== "node" && input.selection.kind !== "edge") {
      throw new Error("GFM_CORE_TASK_TARGET_MISMATCH");
    }
    const nodeIds = input.selection.kind === "node"
      ? [existingNode(input.graph, input.selection.nodeId).id]
      : [];
    const edgeIds = input.selection.kind === "edge"
      ? [registeredEdgeIdentityForLocalId(input.graph, existingEdge(input.graph, input.selection.edgeId).id).edgeHash]
      : [];
    return parseCoreRunRequest({
      ...base,
      targetScope: { kind: "risk-review", nodeIds, edgeIds },
      parameters: { kind: "risk-and-trust", topKSimilarCases: input.topKSimilarCases },
    });
  }
  if (input.selection.kind !== "node-pair") throw new Error("GFM_CORE_TASK_TARGET_MISMATCH");
  const source = existingNode(input.graph, input.selection.sourceId);
  const target = existingNode(input.graph, input.selection.targetId);
  if (source.id === target.id) throw new Error("GFM_CORE_TARGET_SELF_PAIR");
  return parseCoreRunRequest({
    ...base,
    targetScope: { kind: "node-pairs", pairs: [[source.id, target.id]] },
    parameters: {
      kind: "collaboration-completion",
      topKSimilarCases: input.topKSimilarCases,
      candidateLimit: input.candidateLimit ?? 50,
    },
  });
}

export function buildCommunityRunRequest(
  graph: GraphVersion,
  modelVersionId: string,
  communityId: string,
  topKSimilarCases: number,
): CoreRunRequest {
  return buildCoreRunRequest({
    graph,
    modelVersionId,
    taskId: "core.community_resilience_review",
    selection: { kind: "community", communityId },
    topKSimilarCases,
  });
}

export function buildRiskRunRequest(
  graph: GraphVersion,
  modelVersionId: string,
  selection: { readonly kind: "node"; readonly nodeId: string } | { readonly kind: "edge"; readonly edgeId: string },
  topKSimilarCases: number,
): CoreRunRequest {
  return buildCoreRunRequest({
    graph,
    modelVersionId,
    taskId: "core.risk_and_trust_review",
    selection,
    topKSimilarCases,
  });
}

export function buildCollaborationRunRequest(
  graph: GraphVersion,
  modelVersionId: string,
  sourceId: string,
  targetId: string,
  topKSimilarCases: number,
  candidateLimit: number,
): BuiltCollaborationRunRequest {
  existingNode(graph, sourceId);
  existingNode(graph, targetId);
  if (sourceId === targetId) throw new Error("GFM_CORE_TARGET_SELF_PAIR");
  const inspection = inspectCollaborationPair(graph, sourceId, targetId);
  if (inspection.relationState === "recorded") {
    throw new Error("GFM_CORE_COLLABORATION_RELATION_RECORDED");
  }
  return deepFreeze({
    request: buildCoreRunRequest({
      graph,
      modelVersionId,
      taskId: "core.collaboration_completion",
      selection: { kind: "node-pair", sourceId, targetId },
      topKSimilarCases,
      candidateLimit,
    }),
    relationState: "missing",
    existingEdgeIds: [],
  });
}

export function inspectCollaborationPair(
  graph: GraphVersion,
  sourceId: string,
  targetId: string,
): CollaborationPairInspection {
  existingNode(graph, sourceId);
  existingNode(graph, targetId);
  if (sourceId === targetId) throw new Error("GFM_CORE_TARGET_SELF_PAIR");
  const existingEdgeIds = graph.edges
    .filter((edge) => (
      (edge.source === sourceId && edge.target === targetId)
      || (
        edge.directed !== true
        && edge.source === targetId
        && edge.target === sourceId
      )
    ))
    .map((edge) => edge.id)
    .sort(compareUnicodeCodePoints);
  return deepFreeze({
    relationState: existingEdgeIds.length ? "recorded" : "missing",
    existingEdgeIds,
  });
}
