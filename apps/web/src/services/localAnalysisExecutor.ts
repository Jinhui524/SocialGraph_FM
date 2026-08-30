import type {
  AnalysisExecutor,
  AnalysisRun,
  CreateAnalysisInput,
  GraphVersion,
} from "../types/graph";
import { computeGraphSummary, createScopedGraphSlice, runLocalAnalysis } from "./graphAlgorithms";

/** Browser-local deterministic baselines. This class never sends graph data over the network. */
export class LocalAnalysisExecutor implements AnalysisExecutor {
  private readonly graphVersions = new Map<string, GraphVersion>();
  private readonly runs = new Map<string, AnalysisRun>();
  private sequence = 0;

  constructor(initialGraphVersions: readonly GraphVersion[] = []) {
    for (const version of initialGraphVersions) this.registerGraphVersion(version);
  }

  registerGraphVersion(version: GraphVersion): void {
    this.graphVersions.set(version.id, version);
  }

  async createAnalysis(input: CreateAnalysisInput): Promise<AnalysisRun> {
    if (input.graphVersion) this.registerGraphVersion(input.graphVersion);
    this.sequence += 1;
    const id = `analysis-${this.sequence}`;
    const createdAt = new Date().toISOString();
    const graph = this.graphVersions.get(input.graphVersionId);

    if (!graph) {
      const run = Object.freeze({
        id,
        graphVersionId: input.graphVersionId,
        intent: input.intent,
        engine: "local_algorithm" as const,
        status: "failed" as const,
        createdAt,
        completedAt: createdAt,
        error: "GRAPH_VERSION_NOT_FOUND：请先导入关系数据。",
      });
      this.runs.set(id, run);
      return run;
    }

    if (graph.datasetArtifact?.scope === "projection") {
      const completedAt = new Date().toISOString();
      const result = Object.freeze({
        kind: "unavailable" as const,
        code: "DATASET_PROJECTION_REQUIRES_SERVER_ANALYSIS" as const,
        message: "当前图谱是科研数据集的抽样投影；为避免把局部结论误当作全图结论，请使用服务端完整 Artifact 分析。",
        requestedTask: input.intent.task,
      });
      const run: AnalysisRun = Object.freeze({
        id,
        graphVersionId: input.graphVersionId,
        intent: input.intent,
        engine: "unavailable",
        status: "failed",
        createdAt,
        completedAt,
        result,
        error: result.message,
      });
      this.runs.set(id, run);
      return run;
    }

    let analysisGraph = graph;
    let analysisScope = input.scopedGraph?.scope;
    if (input.scopedGraph) {
      if (input.scopedGraph.scope.graphVersionId !== graph.id) {
        const completedAt = new Date().toISOString();
        const run: AnalysisRun = Object.freeze({
          id,
          graphVersionId: input.graphVersionId,
          intent: input.intent,
          engine: "local_algorithm",
          status: "failed",
          createdAt,
          completedAt,
          error: "ANALYSIS_SCOPE_GRAPH_MISMATCH：分析范围不属于当前 GraphVersion。",
        });
        this.runs.set(id, run);
        return run;
      }
      const nodeById = new Map(graph.nodes.map((node) => [node.id, node]));
      const edgeById = new Map(graph.edges.map((edge) => [edge.id, edge]));
      const nodes = input.scopedGraph.slice.nodeIds.map((nodeId) => nodeById.get(nodeId)).filter((node): node is GraphVersion["nodes"][number] => Boolean(node));
      const allowedNodeIds = new Set(nodes.map((node) => node.id));
      const edges = (input.scopedGraph.slice.edgeIds ?? input.scopedGraph.slice.edges.map((edge) => edge.id))
        .map((edgeId) => edgeById.get(edgeId))
        .filter((edge): edge is GraphVersion["edges"][number] => Boolean(
          edge && allowedNodeIds.has(edge.source) && allowedNodeIds.has(edge.target),
        ));
      const verified = createScopedGraphSlice(
        graph.id,
        nodes,
        edges,
        input.scopedGraph.scope.filters,
        input.scopedGraph.scope.truncated,
      );
      if (verified.scope.scopeHash !== input.scopedGraph.scope.scopeHash) {
        const completedAt = new Date().toISOString();
        const run: AnalysisRun = Object.freeze({
          id,
          graphVersionId: input.graphVersionId,
          intent: input.intent,
          engine: "local_algorithm",
          status: "failed",
          createdAt,
          completedAt,
          error: "ANALYSIS_SCOPE_HASH_MISMATCH：分析范围校验失败。",
        });
        this.runs.set(id, run);
        return run;
      }
      analysisScope = verified.scope;
      analysisGraph = Object.freeze({
        ...graph,
        nodes: Object.freeze(nodes),
        edges: Object.freeze(edges),
        summary: Object.freeze(computeGraphSummary(nodes, edges)),
      });
    }

    const result = runLocalAnalysis(analysisGraph, input.intent.task);
    const unavailable = result.kind === "unavailable";
    const completedAt = new Date().toISOString();
    const run: AnalysisRun = Object.freeze({
      id,
      graphVersionId: input.graphVersionId,
      intent: input.intent,
      engine: unavailable ? "unavailable" : "local_algorithm",
      status: unavailable ? "failed" : "succeeded",
      createdAt,
      completedAt,
      result,
      ...(analysisScope ? { scope: analysisScope } : {}),
      ...(unavailable ? { error: result.message } : {}),
    });
    this.runs.set(id, run);
    return run;
  }

  async getAnalysis(runId: string): Promise<AnalysisRun> {
    const run = this.runs.get(runId);
    if (!run) throw new Error(`未找到分析任务：${runId}`);
    return run;
  }
}
