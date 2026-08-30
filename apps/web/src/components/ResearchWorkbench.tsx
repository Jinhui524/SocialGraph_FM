import {
  CheckCircle,
  CircleNotch,
  Database,
  DownloadSimple,
  Graph,
  MagnifyingGlass,
  Play,
  ShieldCheck,
  TreeStructure,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { useEffect, useId, useMemo, useRef, useState } from "react";

import { SocialGraphApiError } from "../services/apiClient";
import { deepFreeze } from "../services/coreContracts";
import { sha256Canonical } from "../services/graphIdentity";
import {
  appendLocalReview,
  createLocalReviewRecord,
  parseLocalReviewRecord,
  type LocalReviewDecision,
  type LocalReviewRecord,
} from "../services/localReview";
import { buildResearchOverlay } from "../services/researchOverlay";
import type { ResearchGraphCompatibility } from "../services/researchDatasetClient";
import {
  buildResearchCoreReport,
  researchFindingHash,
  serializeResearchCoreReportJson,
  serializeResearchCoreReportMarkdown,
} from "../services/researchReport";
import { parseResearchRunRequest } from "../services/researchContracts";
import {
  ResearchRunController,
  type ResearchRunControllerState,
} from "../services/researchRunController";
import type { AnalysisOverlay, GraphRepository, GraphVersion } from "../types/graph";
import {
  RESEARCH_SCHEMA,
  type ResearchClientLike,
  type ResearchFinding,
  type ResearchScenario,
  type ResearchScenarios,
  type ResearchServiceState,
  type ResearchSimilarNodesResult,
  type ResearchTargetScope,
  type ResearchTaskId,
} from "../types/research";

const TASKS: readonly {
  readonly id: ResearchTaskId;
  readonly label: string;
  readonly shortLabel: string;
  readonly dataset: string;
  readonly semantics: string;
  readonly tone: "violet" | "coral" | "amber" | "cyan";
}[] = [
  {
    id: "research.content_policy_review",
    label: "内容策略复核",
    shortLabel: "内容策略",
    dataset: "Twitch Language",
    semantics: "依据 explicit-language 历史标签排序；不等同违法或有害内容判定。",
    tone: "violet",
  },
  {
    id: "research.account_risk_review",
    label: "历史账号状态复核",
    shortLabel: "账号状态",
    dataset: "Tolokers",
    semantics: "依据历史 ban 案例排序；不执行自动封禁或提前预警。",
    tone: "coral",
  },
  {
    id: "research.signed_relation_review",
    label: "治理关系立场复核",
    shortLabel: "关系立场",
    dataset: "Wiki-RfA",
    semantics: "复核支持/反对关系；不解释为毒性或客观可信度。",
    tone: "amber",
  },
  {
    id: "core.collaboration_completion",
    label: "协作关系候选",
    shortLabel: "协作候选",
    dataset: "Email-EU-core",
    semantics: "补全静态图中未观察的关系；不声称预测未来协作。",
    tone: "cyan",
  },
] as const;

export type ResearchSourceMode = "examples" | "my-graph";
type SourceMode = ResearchSourceMode;
type ScenarioState =
  | { readonly state: "loading" }
  | { readonly state: "ready"; readonly value: ResearchScenarios }
  | { readonly state: "unavailable" };

export interface ResearchWorkbenchProps {
  readonly graph: GraphVersion | null;
  readonly selectedNodeId: string | null;
  readonly pathEndpointIds: readonly string[];
  readonly service: ResearchServiceState;
  readonly client: ResearchClientLike;
  readonly sessionId: string;
  readonly repository: Pick<GraphRepository, "appendEvent" | "listEvents">;
  readonly onOverlayChange: (overlay: AnalysisOverlay | null) => void;
  readonly onClose: () => void;
  readonly onPrepareGraph?: (graph: GraphVersion) => Promise<{
    readonly graphVersionHash: string;
    readonly compatibility: ResearchGraphCompatibility | null;
  }>;
  readonly onScenarioSelect?: (scenario: ResearchScenario) => Promise<boolean>;
  readonly onSourceModeChange?: (mode: ResearchSourceMode) => void;
}

function activeRun(state: ResearchRunControllerState): boolean {
  return state.phase === "submitting" || state.phase === "polling" || state.phase === "loading-result";
}

function taskMeta(taskId: ResearchTaskId) {
  return TASKS.find((task) => task.id === taskId)!;
}

function defaultScopeValues(scope: ResearchTargetScope | undefined): {
  readonly nodeId: string;
  readonly sourceId: string;
  readonly targetId: string;
  readonly anchorNodeId: string;
  readonly topK: number;
} {
  if (scope?.kind === "nodes") {
    return { nodeId: scope.nodeIds[0], sourceId: "", targetId: "", anchorNodeId: scope.nodeIds[0], topK: 20 };
  }
  if (scope?.kind === "directed-node-pairs") {
    return {
      nodeId: scope.pairs[0]?.[0] ?? "",
      sourceId: scope.pairs[0]?.[0] ?? "",
      targetId: scope.pairs[0]?.[1] ?? "",
      anchorNodeId: scope.pairs[0]?.[0] ?? "",
      topK: 20,
    };
  }
  if (scope?.kind === "collaboration-candidates") {
    return {
      nodeId: scope.anchorNodeId,
      sourceId: "",
      targetId: "",
      anchorNodeId: scope.anchorNodeId,
      topK: scope.topK,
    };
  }
  return { nodeId: "", sourceId: "", targetId: "", anchorNodeId: "", topK: 20 };
}

function uploadAssessment(graph: GraphVersion | null, maxNodes: number, maxEdges: number) {
  const structuralBlockers: string[] = [];
  const collaborationBlockers: string[] = [];
  if (!graph) {
    structuralBlockers.push("当前会话没有 GraphVersion。");
    collaborationBlockers.push("当前会话没有 GraphVersion。");
    return { structuralBlockers, collaborationBlockers };
  }
  const incomplete = graph.datasetArtifact?.scope === "projection"
    || graph.truncated
    || graph.nodes.length !== graph.summary.nodeCount
    || graph.edges.length !== graph.summary.edgeCount;
  if (incomplete) {
    structuralBlockers.push("当前图是投影或截断视图，不能作为完整 SocialGraph-FM Research 输入。");
    collaborationBlockers.push("当前图是投影或截断视图，不能作为完整 SocialGraph-FM Research 输入。");
  }
  if (graph.summary.nodeCount < 5) structuralBlockers.push("结构检索至少需要 5 个节点。");
  if (graph.summary.edgeCount < 4) structuralBlockers.push("结构检索至少需要 4 条关系。");
  if (graph.summary.nodeCount > maxNodes) structuralBlockers.push(`节点数超过 ${maxNodes.toLocaleString()} 上限。`);
  if (graph.summary.edgeCount > maxEdges) structuralBlockers.push(`关系数超过 ${maxEdges.toLocaleString()} 上限。`);
  if (graph.summary.nodeCount < 20) collaborationBlockers.push("协作关系候选至少需要 20 个节点。");
  if (graph.summary.nodeCount > maxNodes) collaborationBlockers.push(`节点数超过 ${maxNodes.toLocaleString()} 上限。`);
  if (graph.summary.edgeCount > maxEdges) collaborationBlockers.push(`关系数超过 ${maxEdges.toLocaleString()} 上限。`);
  if (graph.metadata?.directedness !== "undirected" || graph.edges.some((edge) => edge.directed === true)) {
    collaborationBlockers.push("上传图的协作候选只接受方向已明确为无向的简单图。");
  }
  const seenPairs = new Set<string>();
  let hasInvalidPair = false;
  for (const edge of graph.edges) {
    if (edge.source === edge.target) {
      hasInvalidPair = true;
      break;
    }
    const pair = edge.source < edge.target
      ? `${edge.source}\u0000${edge.target}`
      : `${edge.target}\u0000${edge.source}`;
    if (seenPairs.has(pair)) {
      hasInvalidPair = true;
      break;
    }
    seenPairs.add(pair);
  }
  if (hasInvalidPair) collaborationBlockers.push("上传图包含自环或重复无向关系，不满足简单图合同。");
  const possiblePairs = graph.summary.nodeCount * (graph.summary.nodeCount - 1) / 2;
  if (possiblePairs - seenPairs.size < 10) collaborationBlockers.push("当前图不足 10 个未记录节点对候选。");
  if (!graph.contentHash && !graph.datasetArtifact?.canonicalGraphHash) {
    structuralBlockers.push("当前图缺少可绑定的内容哈希。");
    collaborationBlockers.push("当前图缺少可绑定的内容哈希。");
  }
  return { structuralBlockers, collaborationBlockers };
}

function failureMessage(code: string): string {
  const messages: Readonly<Record<string, string>> = {
    GFM_RESEARCH_NOT_READY: "SocialGraph-FM Research 模型尚未就绪。",
    GFM_RESEARCH_MODEL_NOT_INSTALLED: "SocialGraph-FM Research 模型尚未安装。",
    GFM_RESEARCH_GRAPH_NOT_FOUND: "当前图版本尚未登记到 SocialGraph-FM Research。",
    GFM_RESEARCH_GRAPH_VERSION_NOT_FOUND: "当前图版本尚未完成 SocialGraph-FM Research 准备。",
    GFM_RESEARCH_GRAPH_REGISTRATION_PENDING: "图适配器登记仍在进行，请稍后重试。",
    GFM_RESEARCH_INCOMPATIBLE: "服务端判定当前任务、图或模型制品不兼容。",
    GFM_RESEARCH_GRAPH_INCOMPATIBLE: "服务端判定当前上传图不满足 SocialGraph-FM Research 合同。",
    GFM_RESEARCH_MODEL_MISMATCH: "模型版本与登记场景或运行请求不一致。",
    GFM_RESEARCH_SCENARIO_MISMATCH: "任务、图版本或模型与登记场景不一致。",
    GFM_RESEARCH_GRAPH_IDENTITY_CONFLICT: "图版本标识与其不可变图哈希冲突。",
    GFM_RESEARCH_GRAPH_ARTIFACT_MISSING: "图制品缺失，无法执行 SocialGraph-FM Research 推理。",
    GFM_RESEARCH_RESPONSE_INVALID: "SocialGraph-FM Research 返回未通过客户端合同校验。",
    GFM_RESEARCH_SCENARIO_UNAVAILABLE: "登记示例场景尚未发布。",
    GFM_RESEARCH_SCENARIO_NOT_FOUND: "未找到登记示例场景。",
    GFM_RESEARCH_RESULT_NOT_READY: "结果尚未就绪，可稍后按运行 ID 查询。",
    GFM_RESEARCH_RUN_NOT_FOUND: "未找到对应的 SocialGraph-FM Research 运行。",
    GFM_RESEARCH_SERVICE_UNAVAILABLE: "SocialGraph-FM Research 服务暂不可用。",
    GFM_RESEARCH_POLL_LIMIT_REACHED: "浏览器已停止自动跟踪，服务端运行可能仍在继续。",
    GRAPH_FACT_HASH_MISMATCH: "浏览器图事实与服务端哈希不一致，交接已拒绝。",
    PREPARATION_GRAPH_VERSION_MISMATCH: "图准备合同不属于当前 GraphVersion。",
    ARTIFACT_HASH_MISMATCH: "目标域制品哈希校验失败。",
    ARTIFACT_INTEGRITY_FAILURE: "目标域制品完整性校验失败。",
    GRAPH_HANDOFF_REJECTED: "GraphVersion 目标域交接被服务端拒绝。",
  };
  return messages[code] ?? "SocialGraph-FM Research 运行未完成；图事实与本地分析未受影响。";
}

function eventReviews(events: Awaited<ReturnType<GraphRepository["listEvents"]>>, findingHash: string) {
  return events.flatMap((event) => {
    if (event.type !== "local_review_recorded") return [];
    try {
      const review = parseLocalReviewRecord(event.payload);
      return review.findingHash === findingHash ? [review] : [];
    } catch {
      return [];
    }
  });
}

function downloadFile(fileName: string, content: string, type: string): void {
  const href = URL.createObjectURL(new Blob([content], { type }));
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = fileName;
  anchor.click();
  URL.revokeObjectURL(href);
}

export function ResearchWorkbench({
  graph,
  selectedNodeId,
  pathEndpointIds,
  service,
  client,
  sessionId,
  repository,
  onOverlayChange,
  onClose,
  onPrepareGraph,
  onScenarioSelect,
  onSourceModeChange,
}: ResearchWorkbenchProps) {
  const [sourceMode, setSourceMode] = useState<SourceMode>("examples");
  const [taskId, setTaskId] = useState<ResearchTaskId>("research.content_policy_review");
  const [scenarioState, setScenarioState] = useState<ScenarioState>({ state: "loading" });
  const [nodeId, setNodeId] = useState("");
  const [sourceId, setSourceId] = useState("");
  const [targetId, setTargetId] = useState("");
  const [anchorNodeId, setAnchorNodeId] = useState("");
  const [topK, setTopK] = useState(20);
  const [candidateLimit, setCandidateLimit] = useState(50);
  const [selectedFindingId, setSelectedFindingId] = useState<string | null>(null);
  const [reviews, setReviews] = useState<readonly LocalReviewRecord[]>([]);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [localNotice, setLocalNotice] = useState<string | null>(null);
  const [preparationState, setPreparationState] = useState<
    | { readonly state: "idle" }
    | { readonly state: "preparing" }
    | {
        readonly state: "ready";
        readonly graphVersionId: string;
        readonly graphVersionHash: string;
        readonly adapterStatus: ResearchGraphCompatibility["adapterStatus"];
      }
    | { readonly state: "blocked"; readonly message: string }
  >({ state: "idle" });
  const [scenarioGraphState, setScenarioGraphState] = useState<"idle" | "loading" | "loaded" | "unavailable">("idle");
  const [similarState, setSimilarState] = useState<
    | { readonly state: "idle" }
    | { readonly state: "loading" }
    | { readonly state: "ready"; readonly result: ResearchSimilarNodesResult }
    | { readonly state: "unavailable" }
  >({ state: "idle" });
  const nodeListId = useId().replace(/:/gu, "-");
  const taskRef = useRef<HTMLDivElement>(null);
  const resultRef = useRef<HTMLElement>(null);
  const controller = useMemo(() => new ResearchRunController(client), [client]);
  const [runState, setRunState] = useState<ResearchRunControllerState>(() => controller.getState());

  useEffect(() => controller.subscribe(setRunState), [controller]);
  useEffect(() => () => controller.dispose(), [controller]);
  useEffect(() => {
    const abortController = new AbortController();
    setScenarioState({ state: "loading" });
    void client.scenarios(abortController.signal)
      .then((value) => setScenarioState({ state: "ready", value }))
      .catch((error) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setScenarioState({ state: "unavailable" });
        }
      });
    return () => abortController.abort();
  }, [client]);

  const scenarios = scenarioState.state === "ready" ? scenarioState.value.scenarios : [];
  const scenario = scenarios.find((item) => item.taskId === taskId);
  const model = service.state === "connected" ? service.capabilities.model : null;
  const uploadLimits = service.state === "connected"
    ? service.capabilities.upload
    : { minNodes: 5, maxNodes: 50_000, maxEdges: 1_500_000 };
  const assessment = useMemo(
    () => uploadAssessment(graph, uploadLimits.maxNodes, uploadLimits.maxEdges),
    [graph, uploadLimits.maxEdges, uploadLimits.maxNodes],
  );
  const currentTask = taskMeta(taskId);

  useEffect(() => {
    if (sourceMode !== "examples" || !scenario || !onScenarioSelect) {
      setScenarioGraphState("idle");
      return;
    }
    let active = true;
    setScenarioGraphState("loading");
    void onScenarioSelect(scenario).then((loaded) => {
      if (active) setScenarioGraphState(loaded ? "loaded" : "unavailable");
    }).catch(() => {
      if (active) setScenarioGraphState("unavailable");
    });
    return () => { active = false; };
  }, [onScenarioSelect, scenario, sourceMode]);

  useEffect(() => {
    controller.reset("research-context-changed");
    onOverlayChange(null);
    setSelectedFindingId(null);
    setReviews([]);
    setSimilarState({ state: "idle" });
    setPreparationState({ state: "idle" });
    setLocalNotice(null);
    const fallbackNodeId = selectedNodeId && graph?.nodes.some((node) => node.id === selectedNodeId)
      ? selectedNodeId
      : graph?.nodes[0]?.id ?? "";
    const scenarioDefaults = defaultScopeValues(scenario?.defaultTargetScope);
    const selectedExampleNode = graph?.id === scenario?.graphVersionId
      && selectedNodeId
      && graph?.nodes.some((node) => node.id === selectedNodeId)
      ? selectedNodeId
      : "";
    const defaults = sourceMode === "examples"
      ? taskId === "research.signed_relation_review"
        ? {
            ...scenarioDefaults,
            sourceId: pathEndpointIds[0] || selectedExampleNode || scenarioDefaults.sourceId,
            targetId: pathEndpointIds[1] ?? scenarioDefaults.targetId,
          }
        : taskId === "core.collaboration_completion"
          ? { ...scenarioDefaults, anchorNodeId: selectedExampleNode || scenarioDefaults.anchorNodeId }
          : { ...scenarioDefaults, nodeId: selectedExampleNode || scenarioDefaults.nodeId }
      : {
          nodeId: fallbackNodeId,
          sourceId: pathEndpointIds[0] ?? fallbackNodeId,
          targetId: pathEndpointIds[1] ?? graph?.nodes.find((node) => node.id !== fallbackNodeId)?.id ?? "",
          anchorNodeId: fallbackNodeId,
          topK: 20,
        };
    setNodeId(defaults.nodeId);
    setSourceId(defaults.sourceId);
    setTargetId(defaults.targetId);
    setAnchorNodeId(defaults.anchorNodeId);
    setTopK(defaults.topK);
  }, [controller, graph, onOverlayChange, pathEndpointIds, scenario, selectedNodeId, sourceMode, taskId]);

  const modelVersionId = sourceMode === "examples" ? scenario?.modelVersionId ?? null : model?.modelVersionId ?? null;
  const graphVersionId = sourceMode === "examples" ? scenario?.graphVersionId ?? null : graph?.id ?? null;
  const targetScope = useMemo<ResearchTargetScope | null>(() => {
    if (taskId === "research.content_policy_review" || taskId === "research.account_risk_review") {
      return nodeId.trim() ? { kind: "nodes", nodeIds: [nodeId.trim()] } : null;
    }
    if (taskId === "research.signed_relation_review") {
      return sourceId.trim() && targetId.trim() && sourceId.trim() !== targetId.trim()
        ? { kind: "directed-node-pairs", pairs: [[sourceId.trim(), targetId.trim()]] }
        : null;
    }
    return anchorNodeId.trim()
      ? { kind: "collaboration-candidates", anchorNodeId: anchorNodeId.trim(), topK }
      : null;
  }, [anchorNodeId, nodeId, sourceId, targetId, taskId, topK]);

  const uploadTargetExists = sourceMode !== "my-graph" || !graph || !targetScope
    ? Boolean(targetScope)
    : targetScope.kind === "collaboration-candidates"
      && graph.nodes.some((node) => node.id === targetScope.anchorNodeId);
  const uploadTaskBlockers = taskId === "core.collaboration_completion"
    ? assessment.collaborationBlockers
    : ["上传图首版只开放协作关系候选；三个标签专用任务仅运行登记示例。"];
  const ready = service.state === "connected"
    && service.capabilities.researchServingReady
    && Boolean(model);
  const exampleReady = sourceMode === "examples"
    && Boolean(scenario?.enabled)
    && scenario?.modelVersionId === model?.modelVersionId;
  const uploadReady = sourceMode === "my-graph"
    && uploadTaskBlockers.length === 0
    && uploadTargetExists;
  const canRun = ready
    && Boolean(graphVersionId && modelVersionId && targetScope)
    && (exampleReady || uploadReady)
    && preparationState.state !== "preparing"
    && !activeRun(runState);

  const succeededState = runState.phase === "succeeded" ? runState : null;
  const expectedGraphHash = sourceMode === "examples"
    ? scenario?.graphVersionHash ?? null
    : preparationState.state === "ready" && preparationState.graphVersionId === graph?.id
      ? preparationState.graphVersionHash
      : graph?.datasetArtifact?.canonicalGraphHash ?? null;
  const resultBindingValid = Boolean(
    succeededState
    && expectedGraphHash
    && succeededState.result.graphVersionHash === expectedGraphHash,
  );
  const findings = resultBindingValid ? succeededState!.result.findings : [];
  const selectedFinding = findings.find((finding) => finding.id === selectedFindingId) ?? findings[0] ?? null;

  useEffect(() => {
    if (!succeededState || !resultBindingValid) return;
    setSelectedFindingId(succeededState.result.findings[0]?.id ?? null);
    resultRef.current?.focus();
  }, [resultBindingValid, succeededState]);

  useEffect(() => {
    if (!succeededState || !selectedFinding || !resultBindingValid || !graph || graph.id !== succeededState.result.graphVersionId) {
      onOverlayChange(null);
      return;
    }
    try {
      onOverlayChange(buildResearchOverlay(
        graph,
        succeededState.binding,
        succeededState.result,
        selectedFinding,
        expectedGraphHash ?? undefined,
      ));
    } catch {
      onOverlayChange(null);
      setLocalNotice("结果与当前图哈希不一致，已拒绝生成图谱覆盖层。");
    }
  }, [expectedGraphHash, graph, onOverlayChange, resultBindingValid, selectedFinding, succeededState]);

  useEffect(() => {
    if (!succeededState || !selectedFinding || !resultBindingValid) {
      setReviews([]);
      return;
    }
    let active = true;
    const findingHash = researchFindingHash(succeededState.result.resultHash, selectedFinding.id);
    void repository.listEvents(succeededState.result.graphVersionId).then((events) => {
      if (active) setReviews(eventReviews(events, findingHash));
    }).catch(() => {
      if (active) setReviews([]);
    });
    return () => { active = false; };
  }, [repository, resultBindingValid, selectedFinding, succeededState]);

  const prepareMyGraph = async (): Promise<boolean> => {
    if (sourceMode !== "my-graph") return true;
    if (!graph) return false;
    if (preparationState.state === "ready" && preparationState.graphVersionId === graph.id) return true;
    if (!onPrepareGraph) return Boolean(graph.datasetArtifact?.canonicalGraphHash);
    setPreparationState({ state: "preparing" });
    const prepared = await onPrepareGraph(graph);
    if (prepared.compatibility?.status === "blocked") {
      const message = prepared.compatibility.blockers.map((blocker) => blocker.message).join(" ")
        || "服务端判定当前图不兼容。";
      setPreparationState({ state: "blocked", message });
      return false;
    }
    setPreparationState({
      state: "ready",
      graphVersionId: graph.id,
      graphVersionHash: prepared.graphVersionHash,
      adapterStatus: prepared.compatibility?.adapterStatus ?? "pending_registration",
    });
    return true;
  };

  const submit = async () => {
    if (!canRun || !graphVersionId || !modelVersionId || !targetScope) return;
    try {
      if (!await prepareMyGraph()) return;
      const request = parseResearchRunRequest({
        schemaVersion: RESEARCH_SCHEMA,
        graphVersionId,
        taskId,
        modelVersionId,
        targetScope,
        ...(sourceMode === "examples" && scenario ? { scenarioId: scenario.scenarioId } : {}),
        parameters: { candidateLimit },
      });
      setLocalNotice(null);
      setSimilarState({ state: "idle" });
      void controller.start(request);
    } catch (error) {
      setPreparationState({ state: "blocked", message: "图准备或请求合同校验未完成。" });
      setLocalNotice(error instanceof SocialGraphApiError
        ? failureMessage(error.code)
        : "图准备或目标范围未通过 SocialGraph-FM Research 合同校验。");
    }
  };

  const similarityNodeId = selectedFinding?.entityIds[0]
    ?? (targetScope?.kind === "nodes" ? targetScope.nodeIds[0]
      : targetScope?.kind === "directed-node-pairs" ? targetScope.pairs[0]?.[0]
        : targetScope?.kind === "collaboration-candidates" ? targetScope.anchorNodeId : undefined);
  const structuralReady = ready
    && Boolean(graphVersionId && modelVersionId && similarityNodeId)
    && (sourceMode === "examples" ? Boolean(scenario?.enabled) : assessment.structuralBlockers.length === 0);
  const findSimilar = async () => {
    if (!structuralReady || !graphVersionId || !modelVersionId || !similarityNodeId) return;
    try {
      if (!await prepareMyGraph()) return;
      setSimilarState({ state: "loading" });
      const request = deepFreeze({
        schemaVersion: RESEARCH_SCHEMA,
        graphVersionId,
        nodeId: similarityNodeId,
        topK: Math.min(20, Math.max(1, topK)),
        modelVersionId,
      });
      let result: ResearchSimilarNodesResult | null = null;
      for (let attempt = 0; attempt < 8; attempt += 1) {
        try {
          result = await client.similarNodes(request);
          break;
        } catch (error) {
          if (!(error instanceof SocialGraphApiError)
            || error.code !== "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING"
            || attempt === 7) throw error;
          await new Promise((resolve) => window.setTimeout(resolve, Math.min(4_000, 500 * (2 ** attempt))));
        }
      }
      if (!result) throw new Error("GFM_RESEARCH_SIMILAR_RESULT_UNAVAILABLE");
      setSimilarState({ state: "ready", result });
    } catch (error) {
      setLocalNotice(error instanceof SocialGraphApiError
        ? failureMessage(error.code)
        : "结构相似检索未完成；不会生成替代案例。");
      setSimilarState({ state: "unavailable" });
    }
  };

  const review = async (decision: LocalReviewDecision) => {
    if (!succeededState || !selectedFinding || !resultBindingValid || reviewBusy) return;
    setReviewBusy(true);
    try {
      const record = createLocalReviewRecord({
        findingHash: researchFindingHash(succeededState.result.resultHash, selectedFinding.id),
        runId: succeededState.result.runId,
        resultHash: succeededState.result.resultHash,
        graphVersionId: succeededState.result.graphVersionId,
        sessionId,
        decision,
      });
      await appendLocalReview(repository, record);
      setReviews((current) => [...current, record]);
      setLocalNotice("人工决定已追加到本地事件；服务器 finding 与图事实未改变。");
    } catch {
      setLocalNotice("本地人工复核写入失败；服务器 finding 未改变。");
    } finally {
      setReviewBusy(false);
    }
  };

  const exportReport = (format: "json" | "markdown") => {
    if (!succeededState || !selectedFinding || !resultBindingValid || reviewBusy) return;
    try {
      const report = buildResearchCoreReport({
        binding: succeededState.binding,
        result: succeededState.result,
        finding: selectedFinding,
        reviews,
      });
      const content = format === "json"
        ? serializeResearchCoreReportJson(report)
        : serializeResearchCoreReportMarkdown(report);
      downloadFile(
        `research-${report.localFindingHash.slice(0, 12)}.${format === "json" ? "json" : "md"}`,
        content,
        format === "json" ? "application/json;charset=utf-8" : "text/markdown;charset=utf-8",
      );
    } catch {
      setLocalNotice("报告绑定校验失败，未导出不完整结果。");
    }
  };

  const nodeSuggestions = graph?.nodes.slice(0, 100) ?? [];
  const latestReview = reviews.at(-1);
  const scoreLabel = (finding: ResearchFinding) => {
    if (taskId === "research.signed_relation_review") {
      return finding.scoreKind === "probability" ? "反对概率" : "反对排序";
    }
    return finding.scoreKind === "probability" ? "校准分数" : "排序分数";
  };
  const claimStatus = model?.claimStatus === "observed_transfer_gain"
    ? "观察到跨域预训练收益"
    : "尚未证明优于单域基线";

  return (
    <section className={`research-governance is-${currentTask.tone}`} aria-labelledby="research-governance-title">
      <header className="research-governance__header">
        <div className="research-governance__title">
          <span className="research-governance__mark"><ShieldCheck size={20} weight="fill" /></span>
          <div>
            <h2 id="research-governance-title">社交治理应用</h2>
            <p>四任务静态图基础模型工作台</p>
          </div>
        </div>
        <div className="research-governance__badges" aria-label="研究版本信息">
          <span>SocialGraph-FM Research</span><span>seed 1729</span><span className="is-warning">单次实验初步结果</span>
          <button type="button" aria-label="关闭治理应用" title="关闭" onClick={onClose}><X size={17} /></button>
        </div>
      </header>

      <div className="research-governance__status-row">
        <div className="research-segmented" role="group" aria-label="数据源">
          <button type="button" aria-pressed={sourceMode === "examples"} onClick={() => {
            onSourceModeChange?.("examples");
            setSourceMode("examples");
          }}>
            <Database size={15} />示例数据
          </button>
          <button type="button" aria-pressed={sourceMode === "my-graph"} onClick={() => {
            onSourceModeChange?.("my-graph");
            setSourceMode("my-graph");
          }}>
            <Graph size={15} />我的图谱
          </button>
        </div>
        <span className={`research-readiness is-${ready ? "ready" : "blocked"}`}>
          {service.state === "checking" ? <CircleNotch size={14} className="spin" /> : ready ? <CheckCircle size={14} weight="fill" /> : <WarningCircle size={14} weight="fill" />}
          {service.state === "checking" ? "检查模型" : ready ? "模型可运行" : "模型未就绪"}
        </span>
      </div>

      <div className="research-mobile-jump" role="group" aria-label="治理任务与证据快捷切换">
        <button type="button" onClick={() => taskRef.current?.scrollIntoView({ block: "start", behavior: "smooth" })}>任务</button>
        <button
          type="button"
          disabled={!succeededState || !resultBindingValid}
          onClick={() => resultRef.current?.scrollIntoView({ block: "start", behavior: "smooth" })}
        >证据</button>
      </div>

      <div ref={taskRef} className="research-task-tabs" role="tablist" aria-label="SocialGraph-FM Research 治理任务">
        {TASKS.map((task) => (
          <button
            key={task.id}
            type="button"
            role="tab"
            aria-selected={taskId === task.id}
            data-tone={task.tone}
            onClick={() => setTaskId(task.id)}
          >
            <span>{task.shortLabel}</span><small>{task.dataset}</small>
          </button>
        ))}
      </div>

      <div className="research-governance__context">
        <div>
          <strong>{currentTask.label}</strong>
          <span>{currentTask.semantics}</span>
        </div>
        <span className="research-claim">{claimStatus}</span>
      </div>

      {sourceMode === "examples" ? (
        scenarioState.state === "loading" ? (
          <p className="research-notice" role="status"><CircleNotch size={15} className="spin" />正在读取登记场景…</p>
        ) : scenarioState.state === "unavailable" || !scenario ? (
          <p className="research-notice is-warning" role="status"><WarningCircle size={15} />登记场景不可用；不会生成示例模型输出。</p>
        ) : (
          <section className="research-scenario" aria-label="登记示例场景">
            <div><small>登记场景</small><strong>{scenario.title}</strong><span>{scenario.datasetId}</span></div>
            <div className="research-scenario__metrics">
              {scenario.primaryMetric ? <span><small>{scenario.primaryMetric.name}</small><strong>{scenario.primaryMetric.value.toFixed(4)}</strong></span> : null}
              {scenario.scratchDelta !== null ? <span><small>相对 scratch</small><strong>{scenario.scratchDelta >= 0 ? "+" : ""}{scenario.scratchDelta.toFixed(4)}</strong></span> : null}
            </div>
            {!scenario.enabled ? <p>{scenario.unavailableReason ?? "该场景尚未发布。"}</p> : null}
            {scenarioGraphState === "loading" ? <p>正在加载后端登记的同哈希只读投影…</p> : null}
            {scenarioGraphState === "loaded" ? <p>已载入同哈希图谱投影。</p> : null}
            {scenarioGraphState === "unavailable" ? <p>本地没有同哈希可视投影；推理仍严格使用后端登记图。</p> : null}
          </section>
        )
      ) : (
        <div className={`research-notice ${uploadTaskBlockers.length ? "is-warning" : ""}`} role="status">
          {uploadTaskBlockers.length ? <WarningCircle size={15} /> : <CheckCircle size={15} weight="fill" />}
          <span>{uploadTaskBlockers.join(" ") || "浏览器基础兼容检查通过；服务端 POST 将执行权威合同校验。"}</span>
        </div>
      )}

      {service.state === "unavailable" ? (
        <p className="research-notice is-warning" role="status"><WarningCircle size={15} />SocialGraph-FM Research 服务暂不可用；本地图分析仍可使用。</p>
      ) : service.state === "connected" && !service.capabilities.researchServingReady ? (
        <p className="research-notice is-warning" role="status"><WarningCircle size={15} />{service.capabilities.unavailableReason ?? "研究模型尚未发布。"}</p>
      ) : null}

      <datalist id={`${nodeListId}-nodes`}>
        {nodeSuggestions.map((node) => <option key={node.id} value={node.id}>{node.label}</option>)}
      </datalist>
      <div className="research-targets">
        {taskId === "research.content_policy_review" || taskId === "research.account_risk_review" ? (
          <label><span>复核节点</span><input aria-label="复核节点" list={`${nodeListId}-nodes`} value={nodeId} onChange={(event) => setNodeId(event.target.value)} /></label>
        ) : null}
        {taskId === "research.signed_relation_review" ? (
          <>
            <label><span>投票者节点</span><input aria-label="投票者节点" list={`${nodeListId}-nodes`} value={sourceId} onChange={(event) => setSourceId(event.target.value)} /></label>
            <label><span>候选者节点</span><input aria-label="候选者节点" list={`${nodeListId}-nodes`} value={targetId} onChange={(event) => setTargetId(event.target.value)} /></label>
          </>
        ) : null}
        {taskId === "core.collaboration_completion" ? (
          <>
            <label><span>锚点节点</span><input aria-label="锚点节点" list={`${nodeListId}-nodes`} value={anchorNodeId} onChange={(event) => setAnchorNodeId(event.target.value)} /></label>
            <label><span>返回候选</span><input aria-label="返回候选" type="number" min={1} max={100} value={topK} onChange={(event) => setTopK(Math.min(100, Math.max(1, Number.parseInt(event.target.value, 10) || 1)))} /></label>
          </>
        ) : null}
        <label><span>运行候选上限</span><input aria-label="运行候选上限" type="number" min={1} max={1000} value={candidateLimit} onChange={(event) => setCandidateLimit(Math.min(1000, Math.max(1, Number.parseInt(event.target.value, 10) || 1)))} /></label>
      </div>

      <div className="research-governance__actions">
        <button className="is-primary" type="button" disabled={!canRun} onClick={submit}>
          {activeRun(runState) || preparationState.state === "preparing" ? <CircleNotch size={16} className="spin" /> : <Play size={16} weight="fill" />}
          {sourceMode === "my-graph" ? "准备并运行 SocialGraph-FM Research 分析" : "运行治理任务"}
        </button>
        <button type="button" disabled={!structuralReady || similarState.state === "loading"} onClick={findSimilar}>
          {similarState.state === "loading" ? <CircleNotch size={16} className="spin" /> : <TreeStructure size={16} />}
          结构相似检索
        </button>
        {runState.phase === "polling" || runState.phase === "loading-result" ? (
          <button type="button" onClick={() => controller.stopFollowing()}>停止跟踪</button>
        ) : null}
      </div>

      {runState.phase === "submitting" ? <p className="research-notice" role="status">正在提交 SocialGraph-FM Research 运行…</p> : null}
      {runState.phase === "polling" ? <p className="research-notice" role="status">运行 {runState.binding.runId} · {runState.status.progress}%</p> : null}
      {runState.phase === "loading-result" ? <p className="research-notice" role="status">运行成功，正在校验不可变结果…</p> : null}
      {runState.phase === "detached" ? <p className="research-notice is-warning" role="status">已停止浏览器跟踪；服务端运行可能继续。</p> : null}
      {runState.phase === "failed" ? <p className="research-notice is-warning" role="alert"><WarningCircle size={15} />{failureMessage(runState.code)}</p> : null}
      {succeededState && !resultBindingValid ? <p className="research-notice is-warning" role="alert"><WarningCircle size={15} />结果图哈希与当前登记场景或 GraphVersion 不一致，已拒绝展示。</p> : null}
      {localNotice ? <p className="research-notice" role="status">{localNotice}</p> : null}
      {preparationState.state === "preparing" ? <p className="research-notice" role="status"><CircleNotch size={15} className="spin" />正在原子交接 GraphVersion 并准备 GFM 适配器…</p> : null}
      {preparationState.state === "ready" && preparationState.adapterStatus === "pending_registration" && activeRun(runState) ? (
        <p className="research-notice" role="status"><CircleNotch size={15} className="spin" />结构适配器正在登记；运行会在有界重试内自动继续。</p>
      ) : null}
      {preparationState.state === "blocked" ? <p className="research-notice is-warning" role="alert"><WarningCircle size={15} />{preparationState.message}</p> : null}

      {succeededState && resultBindingValid ? (
        <section ref={resultRef} className="research-results" aria-label="SocialGraph-FM Research 治理结果" aria-live="polite" tabIndex={-1}>
          <header>
            <div><strong>候选排序</strong><span>{findings.length} 条 · {succeededState.result.calibrationStatus === "calibrated" ? "已校准" : "仅排序分数"}</span></div>
            <span>待人工复核</span>
          </header>
          {findings.length ? (
            <div className="research-result-layout">
              <div className="research-ranking" role="listbox" aria-label="治理候选排名">
                {findings.map((finding) => (
                  <button
                    key={finding.id}
                    type="button"
                    role="option"
                    aria-selected={selectedFinding?.id === finding.id}
                    onClick={() => setSelectedFindingId(finding.id)}
                  >
                    <span className="research-rank">{finding.rank}</span>
                    <span><strong>{finding.entityIds.join(" → ")}</strong><small>{finding.entityType}</small></span>
                    <span className="research-score"><strong>{finding.score.toFixed(4)}</strong><small>{scoreLabel(finding)}</small></span>
                  </button>
                ))}
              </div>
              {selectedFinding ? (
                <div className="research-evidence">
                  <section>
                    <h3>图事实</h3>
                    {graph && graph.id === succeededState.result.graphVersionId ? (
                      <dl>
                        {graph.datasetArtifact?.scope === "projection" ? (
                          <>
                            <div><dt>完整图节点</dt><dd>{graph.preview.originalNodeCount.toLocaleString()}</dd></div>
                            <div><dt>完整图关系</dt><dd>{graph.preview.originalEdgeCount.toLocaleString()}</dd></div>
                            <div><dt>投影可见</dt><dd>{graph.nodes.length.toLocaleString()} 节点 / {graph.edges.length.toLocaleString()} 关系</dd></div>
                            <div><dt>投影密度</dt><dd>{graph.summary.density.toFixed(4)}</dd></div>
                          </>
                        ) : (
                          <>
                            <div><dt>节点</dt><dd>{graph.summary.nodeCount.toLocaleString()}</dd></div>
                            <div><dt>关系</dt><dd>{graph.summary.edgeCount.toLocaleString()}</dd></div>
                            <div><dt>密度</dt><dd>{graph.summary.density.toFixed(4)}</dd></div>
                          </>
                        )}
                      </dl>
                    ) : <p>示例图由后端登记；本页只使用返回的不可变图哈希绑定结果。</p>}
                  </section>
                  <section>
                    <h3>GFM 推断</h3>
                    <p><strong>{selectedFinding.score.toFixed(4)}</strong> · {taskId === "research.signed_relation_review" ? scoreLabel(selectedFinding) : selectedFinding.calibrated ? "验证集校准" : "仅用于排序"}</p>
                    {selectedFinding.reasonCodes.length ? (
                      <div className="research-reason-list">{selectedFinding.reasonCodes.map((reason) => <code key={reason}>{reason}</code>)}</div>
                    ) : <p>服务端未返回额外依据代码。</p>}
                  </section>
                  <section>
                    <h3>限制与复核</h3>
                    <ul>
                      <li>单随机种子初步结果，不代表稳定泛化。</li>
                      {selectedFinding.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}
                    </ul>
                    <p>{latestReview ? `本地复核：${latestReview.decision === "confirmed" ? "已确认" : "已驳回"}` : "尚无本地人工复核记录。"}</p>
                    <div className="research-inline-actions">
                      <button type="button" disabled={reviewBusy} onClick={() => void review("confirmed")}><CheckCircle size={15} />确认</button>
                      <button type="button" disabled={reviewBusy} onClick={() => void review("rejected")}><X size={15} />驳回</button>
                    </div>
                  </section>
                  <details>
                    <summary>技术详情与不可变哈希</summary>
                    <dl className="research-hashes">
                      <div><dt>runId</dt><dd><code>{succeededState.result.runId}</code></dd></div>
                      <div><dt>graphVersionHash</dt><dd><code>{succeededState.result.graphVersionHash}</code></dd></div>
                      <div><dt>modelVersionHash</dt><dd><code>{succeededState.result.modelVersionHash}</code></dd></div>
                      <div><dt>resultHash</dt><dd><code>{succeededState.result.resultHash}</code></dd></div>
                      <div><dt>publicRequestHash</dt><dd><code>{succeededState.binding.publicRequestHash}</code></dd></div>
                      <div><dt>serverRequestHash</dt><dd><code>{succeededState.binding.serverRequestHash}</code></dd></div>
                    </dl>
                  </details>
                  <div className="research-inline-actions">
                    <button type="button" disabled={reviewBusy} onClick={() => exportReport("json")}><DownloadSimple size={15} />JSON</button>
                    <button type="button" disabled={reviewBusy} onClick={() => exportReport("markdown")}><DownloadSimple size={15} />Markdown</button>
                  </div>
                </div>
              ) : null}
            </div>
          ) : <p className="research-empty-result">服务端返回空候选集；未生成任何实体级发现。</p>}
        </section>
      ) : null}

      {similarState.state === "ready" ? (
        <section className="research-similar" aria-labelledby="research-similar-title">
          <header><div><MagnifyingGlass size={16} /><strong id="research-similar-title">结构相似节点</strong></div><span>{similarState.result.matches.length} 条</span></header>
          {similarState.result.matches.length ? (
            <div className="research-similar__list">
              {similarState.result.matches.map((match, index) => (
                <article key={`${match.graphVersionId}:${match.nodeId}`}>
                  <span>{index + 1}</span>
                  <div><strong>{match.nodeId}</strong><small>{match.datasetId ?? match.graphVersionId}</small></div>
                  <strong>{match.similarity.toFixed(4)}</strong>
                  <small>度 {match.structuralFacts.degree} · core {match.structuralFacts.coreNumber} · clustering {match.structuralFacts.clustering.toFixed(3)}</small>
                </article>
              ))}
            </div>
          ) : <p>索引中没有满足条件的相似节点。</p>}
          <details><summary>检索结果哈希</summary><code>{similarState.result.resultHash}</code></details>
        </section>
      ) : similarState.state === "unavailable" ? (
        <p className="research-notice is-warning" role="status"><WarningCircle size={15} />结构相似检索未完成；不会生成替代案例。</p>
      ) : null}
    </section>
  );
}
