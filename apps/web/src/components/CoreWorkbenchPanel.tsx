import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react";

import {
  assessCoreCompatibility,
  buildCollaborationRunRequest,
  buildCommunityRunRequest,
  buildRiskRunRequest,
  inspectCollaborationPair,
} from "../services/coreCompatibility";
import { registeredEdgeIdentityForLocalId } from "../services/coreEdgeIdentity";
import { buildCoreOverlay } from "../services/coreOverlay";
import {
  appendLocalReview,
  createLocalReviewRecord,
  parseLocalReviewRecord,
  type LocalReviewDecision,
  type LocalReviewRecord,
} from "../services/localReview";
import {
  buildCoreReport,
  coreFindingLabel,
  serializeCoreReportJson,
  serializeCoreReportMarkdown,
  type CoreReport,
} from "../services/coreReport";
import {
  CoreRunController,
  type CoreRunControllerState,
} from "../services/coreRunController";
import { isPublicCoreErrorCode } from "../services/coreContracts";
import { coreRunContextKey } from "../services/coreRunContext";
import type {
  AnalysisOverlay,
  GraphRepository,
  GraphVersion,
  SemanticEvent,
} from "../types/graph";
import type {
  CoreClientLike,
  CoreModelCapability,
  CoreTaskId,
  CoreWorkbenchServiceState,
  CoreFinding,
} from "../types/core";

const TASKS: readonly {
  readonly id: CoreTaskId;
  readonly label: string;
  readonly description: string;
}[] = [
  {
    id: "core.community_resilience_review",
    label: "社区韧性复核",
    description: "复核已登记社区的结构脆弱性、关键节点与现有证据。",
  },
  {
    id: "core.risk_and_trust_review",
    label: "风险与信任复核",
    description: "对现有节点或关系生成候选排序，交由人工结合上下文判断。",
  },
  {
    id: "core.collaboration_completion",
    label: "协作关系补全",
    description: "对两个真实节点生成静态关系补全建议，不向事实图添加关系。",
  },
] as const;

const INITIAL_TASK: CoreTaskId = "core.risk_and_trust_review";
const TARGET_SUGGESTION_LIMIT = 100;

export interface CoreWorkbenchPanelProps {
  readonly graph: GraphVersion;
  readonly service: CoreWorkbenchServiceState;
  readonly client: CoreClientLike;
  readonly selectedNodeId: string | null;
  readonly pathEndpointIds: readonly string[];
  readonly sessionId?: string;
  readonly repository: Pick<GraphRepository, "appendEvent" | "listEvents">;
  readonly onOverlayChange: (overlay: AnalysisOverlay | null) => void;
  readonly onReportExport?: (
    report: CoreReport,
    format: "json" | "markdown",
    content: string,
  ) => void;
}

function selectedModelForTask(
  models: readonly CoreModelCapability[],
  taskId: CoreTaskId,
  preferredId?: string,
): CoreModelCapability | undefined {
  const supported = models.filter((model) => model.tasks.includes(taskId));
  return supported.find((model) => model.modelVersionId === preferredId)
    ?? supported.find((model) => model.state === "servingReady")
    ?? supported[0];
}

function activeRun(state: CoreRunControllerState): boolean {
  return state.phase === "submitting" || state.phase === "polling" || state.phase === "loading-result";
}

function detachableRun(state: CoreRunControllerState): boolean {
  return state.phase === "polling" || state.phase === "loading-result";
}

function safeFailureGuidance(code: string): string {
  if (code === "GFM_CORE_GRAPH_VERSION_NOT_FOUND") {
    return "服务端未找到该版本；请精确交接当前不可变 GraphVersion 后重试。";
  }
  if (code === "GFM_CORE_MODEL_GRAPH_INCOMPATIBLE") {
    return "服务端权威合同判定不兼容；请检查目标域特征与登记制品，不会自动切换模型或图。";
  }
  if (code === "GFM_CORE_MODEL_NOT_INSTALLED" || code === "GFM_CORE_SERVICE_UNAVAILABLE") {
    return "模型尚未安装或不可服务；本地图导入与确定性分析仍可使用。";
  }
  if (code === "GFM_CORE_POLL_LIMIT_REACHED") {
    return "已达到本地跟踪上限；服务端运行可能仍在继续，可稍后按运行 ID 查询。";
  }
  if (code === "GFM_CORE_EDGE_IDENTITY_UNPROVABLE" || code === "GFM_CORE_EDGE_IDENTITY_DUPLICATE") {
    return "关系目标缺少可证明的 Core edge 身份，或存在重复语义；请使用方向明确且显式填写 edgeType/weight 的唯一关系。";
  }
  if (code === "GFM_CORE_COLLABORATION_RELATION_RECORDED") {
    return "该节点对已有记录关系；静态关系补全只接受尚未记录的节点对。";
  }
  return `治理运行未完成（${code}）。本地图事实与本地分析未受影响。`;
}

function eventReview(event: SemanticEvent): LocalReviewRecord | null {
  if (event.type !== "local_review_recorded") return null;
  try {
    return parseLocalReviewRecord(event.payload);
  } catch {
    return null;
  }
}

function downloadReport(format: "json" | "markdown", content: string, findingHash: string): void {
  const blob = new Blob([content], {
    type: format === "json" ? "application/json;charset=utf-8" : "text/markdown;charset=utf-8",
  });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = `core-${findingHash.slice(0, 12)}.${format === "json" ? "json" : "md"}`;
  anchor.click();
  URL.revokeObjectURL(href);
}

function modelStateLabel(model: CoreModelCapability): string {
  return model.state === "servingReady" ? "可运行" : "已验收，未服务";
}

function boundedInteger(event: ChangeEvent<HTMLInputElement>, minimum: number, maximum: number): number {
  const parsed = Number.parseInt(event.target.value, 10);
  if (!Number.isFinite(parsed)) return minimum;
  return Math.min(maximum, Math.max(minimum, parsed));
}

function boundedIdSuggestions<T extends { readonly id: string }>(
  values: readonly T[],
  preferredIds: readonly string[],
  byId: ReadonlyMap<string, T>,
): readonly T[] {
  const selected: T[] = [];
  const seen = new Set<string>();
  for (const id of preferredIds) {
    const value = byId.get(id);
    if (value && !seen.has(id)) {
      selected.push(value);
      seen.add(id);
    }
  }
  for (const value of values) {
    if (selected.length >= TARGET_SUGGESTION_LIMIT) break;
    if (!seen.has(value.id)) {
      selected.push(value);
      seen.add(value.id);
    }
  }
  return selected;
}

export function CoreWorkbenchPanel({
  graph,
  service,
  client,
  selectedNodeId,
  pathEndpointIds,
  sessionId,
  repository,
  onOverlayChange,
  onReportExport,
}: CoreWorkbenchPanelProps) {
  const models = service.state === "connected" ? service.capabilities.models : [];
  const unavailableCode = service.state === "unavailable" && isPublicCoreErrorCode(service.code)
    ? service.code
    : "GFM_CORE_SERVICE_UNAVAILABLE";
  const [taskId, setTaskId] = useState<CoreTaskId>(INITIAL_TASK);
  const [modelVersionId, setModelVersionId] = useState<string>(() => (
    selectedModelForTask(models, INITIAL_TASK)?.modelVersionId ?? ""
  ));
  const communities = useMemo(
    () => graph.nodes.filter((node) => node.type?.trim().toLocaleLowerCase("en-US") === "community"),
    [graph.nodes],
  );
  const nodeById = useMemo(
    () => new Map(graph.nodes.map((node) => [node.id, node] as const)),
    [graph.nodes],
  );
  const edgeById = useMemo(
    () => new Map(graph.edges.map((edge) => [edge.id, edge] as const)),
    [graph.edges],
  );
  const communityById = useMemo(
    () => new Map(communities.map((node) => [node.id, node] as const)),
    [communities],
  );
  const defaultNodeId = nodeById.has(selectedNodeId ?? "")
    ? selectedNodeId as string
    : graph.nodes[0]?.id ?? "";
  const defaultPairSource = pathEndpointIds.find((id) => nodeById.has(id))
    ?? defaultNodeId;
  const defaultPairTarget = pathEndpointIds.find((id) => id !== defaultPairSource && nodeById.has(id))
    ?? graph.nodes.find((node) => node.id !== defaultPairSource)?.id
    ?? "";
  const [communityId, setCommunityId] = useState(communities[0]?.id ?? "");
  const [riskKind, setRiskKind] = useState<"node" | "edge">("node");
  const [riskNodeId, setRiskNodeId] = useState(defaultNodeId);
  const [riskEdgeId, setRiskEdgeId] = useState(graph.edges[0]?.id ?? "");
  const [pairSourceId, setPairSourceId] = useState(defaultPairSource);
  const [pairTargetId, setPairTargetId] = useState(defaultPairTarget);
  const [topKSimilarCases, setTopKSimilarCases] = useState(5);
  const [candidateLimit, setCandidateLimit] = useState(50);
  const [selectedFindingHash, setSelectedFindingHash] = useState<string | null>(null);
  const [reviews, setReviews] = useState<readonly LocalReviewRecord[]>([]);
  const [reviewBusy, setReviewBusy] = useState(false);
  const [localNotice, setLocalNotice] = useState<string | null>(null);
  const targetListBaseId = useId().replace(/:/gu, "-");
  const resultRegionRef = useRef<HTMLElement>(null);
  const controller = useMemo(() => new CoreRunController(client), [client]);
  const [runState, setRunState] = useState<CoreRunControllerState>(() => controller.getState());

  const nodeSuggestions = useMemo(() => boundedIdSuggestions(
    graph.nodes,
    [riskNodeId, pairSourceId, pairTargetId, selectedNodeId ?? "", ...pathEndpointIds],
    nodeById,
  ), [graph.nodes, nodeById, pairSourceId, pairTargetId, pathEndpointIds, riskNodeId, selectedNodeId]);
  const edgeSuggestions = useMemo(
    () => boundedIdSuggestions(graph.edges, [riskEdgeId], edgeById),
    [edgeById, graph.edges, riskEdgeId],
  );
  const communitySuggestions = useMemo(
    () => boundedIdSuggestions(communities, [communityId], communityById),
    [communities, communityById, communityId],
  );

  useEffect(() => controller.subscribe(setRunState), [controller]);
  useEffect(() => () => controller.dispose(), [controller]);

  const selectedModel = useMemo(
    () => selectedModelForTask(models, taskId, modelVersionId),
    [modelVersionId, models, taskId],
  );

  useEffect(() => {
    const next = selectedModelForTask(models, taskId, modelVersionId);
    if ((next?.modelVersionId ?? "") !== modelVersionId) setModelVersionId(next?.modelVersionId ?? "");
  }, [modelVersionId, models, taskId]);

  useEffect(() => {
    if (!nodeById.has(riskNodeId)) setRiskNodeId(defaultNodeId);
    if (!edgeById.has(riskEdgeId)) setRiskEdgeId(graph.edges[0]?.id ?? "");
    if (!communityById.has(communityId)) setCommunityId(communities[0]?.id ?? "");
    if (!nodeById.has(pairSourceId)) setPairSourceId(defaultPairSource);
    if (!nodeById.has(pairTargetId) || pairTargetId === pairSourceId) {
      setPairTargetId(defaultPairTarget);
    }
  }, [
    communities,
    communityId,
    communityById,
    defaultNodeId,
    defaultPairSource,
    defaultPairTarget,
    edgeById,
    graph.edges,
    nodeById,
    pairSourceId,
    pairTargetId,
    riskEdgeId,
    riskNodeId,
  ]);

  useEffect(() => {
    if (selectedNodeId && nodeById.has(selectedNodeId)) {
      setRiskNodeId(selectedNodeId);
    }
  }, [nodeById, selectedNodeId]);

  useEffect(() => {
    const validEndpoints = pathEndpointIds.filter((id) => nodeById.has(id));
    if (validEndpoints[0]) setPairSourceId(validEndpoints[0]);
    if (validEndpoints[1] && validEndpoints[1] !== validEndpoints[0]) setPairTargetId(validEndpoints[1]);
  }, [nodeById, pathEndpointIds]);

  const requestContextKey = coreRunContextKey({
    graphVersionId: graph.id,
    taskId,
    modelVersionId,
    target: taskId === "core.community_resilience_review"
      ? { kind: "community", communityId }
      : taskId === "core.risk_and_trust_review"
        ? riskKind === "node"
          ? { kind: "node", nodeId: riskNodeId }
          : { kind: "edge", edgeId: riskEdgeId }
        : { kind: "node-pair", sourceId: pairSourceId, targetId: pairTargetId },
    parameters: { topKSimilarCases, candidateLimit },
  });

  useEffect(() => {
    controller.reset("core-context-changed");
    setSelectedFindingHash(null);
    setReviews([]);
    setLocalNotice(null);
    onOverlayChange(null);
  }, [controller, graph.id, modelVersionId, onOverlayChange, requestContextKey, taskId]);

  const compatibility = selectedModel
    ? assessCoreCompatibility(graph, selectedModel, taskId)
    : null;
  const selectedRiskEdgeIdentity = useMemo(() => {
    if (riskKind !== "edge" || !edgeById.has(riskEdgeId)) return null;
    try {
      return registeredEdgeIdentityForLocalId(graph, riskEdgeId);
    } catch {
      return null;
    }
  }, [edgeById, graph, riskEdgeId, riskKind]);
  const collaborationPair = useMemo(() => {
    if (
      !nodeById.has(pairSourceId)
      || !nodeById.has(pairTargetId)
      || pairSourceId === pairTargetId
    ) return null;
    return inspectCollaborationPair(graph, pairSourceId, pairTargetId);
  }, [graph, nodeById, pairSourceId, pairTargetId]);
  const targetReady = taskId === "core.community_resilience_review"
    ? communityById.has(communityId)
    : taskId === "core.risk_and_trust_review"
      ? riskKind === "node"
        ? nodeById.has(riskNodeId)
        : selectedRiskEdgeIdentity !== null
      : collaborationPair?.relationState === "missing";
  const canRun = service.state === "connected"
    && Boolean(selectedModel)
    && Boolean(compatibility?.runnable)
    && targetReady
    && !activeRun(runState);

  const succeededState = runState.phase === "succeeded" ? runState : null;
  const findings = succeededState?.result.findings ?? [];
  const selectedFinding: CoreFinding | undefined = findings.find(
    (finding) => finding.findingHash === selectedFindingHash,
  ) ?? findings[0];
  const selectedReport = useMemo(() => {
    if (!succeededState || !selectedFinding) return null;
    try {
      return buildCoreReport({
        binding: succeededState.binding,
        result: succeededState.result,
        finding: selectedFinding,
        reviews,
      });
    } catch {
      return null;
    }
  }, [reviews, selectedFinding, succeededState]);

  useEffect(() => {
    if (!succeededState) return;
    const finding = succeededState.result.findings[0];
    if (finding) setSelectedFindingHash((current) => current ?? finding.findingHash);
  }, [succeededState]);

  useEffect(() => {
    if (succeededState && selectedFinding) resultRegionRef.current?.focus();
  }, [selectedFinding, succeededState]);

  useEffect(() => {
    if (!succeededState || !selectedFinding) {
      setReviews([]);
      return;
    }
    let active = true;
    void repository.listEvents(graph.id).then((events) => {
      if (!active) return;
      setReviews(events
        .map(eventReview)
        .filter((review): review is LocalReviewRecord => Boolean(review))
        .filter((review) => (
          review.findingHash === selectedFinding.findingHash
          && review.runId === succeededState.binding.runId
          && review.resultHash === succeededState.result.resultHash
          && review.graphVersionId === graph.id
        )));
    }).catch(() => {
      if (active) setLocalNotice("本地复核记录暂时无法读取。");
    });
    return () => { active = false; };
  }, [graph.id, repository, selectedFinding, succeededState]);

  useEffect(() => {
    if (!succeededState || !selectedFinding) return;
    try {
      onOverlayChange(buildCoreOverlay(
        graph,
        succeededState.binding,
        succeededState.result,
        selectedFinding,
      ));
    } catch {
      onOverlayChange(null);
    }
  }, [graph, onOverlayChange, selectedFinding, succeededState]);

  const submit = () => {
    if (!selectedModel || !canRun) return;
    try {
      const request = taskId === "core.community_resilience_review"
        ? buildCommunityRunRequest(graph, selectedModel.modelVersionId, communityId, topKSimilarCases)
        : taskId === "core.risk_and_trust_review"
          ? buildRiskRunRequest(
            graph,
            selectedModel.modelVersionId,
            riskKind === "node" ? { kind: "node", nodeId: riskNodeId } : { kind: "edge", edgeId: riskEdgeId },
            topKSimilarCases,
          )
          : buildCollaborationRunRequest(
            graph,
            selectedModel.modelVersionId,
            pairSourceId,
            pairTargetId,
            topKSimilarCases,
            candidateLimit,
          ).request;
      setLocalNotice(null);
      void controller.start(request);
    } catch (error) {
      const code = error instanceof Error && /^[A-Z0-9_]{1,100}$/u.test(error.message)
        ? error.message
        : "GFM_CORE_REQUEST_INVALID";
      setLocalNotice(safeFailureGuidance(code));
    }
  };

  const review = async (decision: LocalReviewDecision) => {
    if (!succeededState || !selectedFinding || reviewBusy) return;
    setReviewBusy(true);
    setLocalNotice(null);
    try {
      const record = createLocalReviewRecord({
        findingHash: selectedFinding.findingHash,
        runId: succeededState.binding.runId,
        resultHash: succeededState.result.resultHash,
        graphVersionId: graph.id,
        ...(sessionId ? { sessionId } : {}),
        decision,
      });
      await appendLocalReview(repository, record);
      setReviews((current) => [...current, record]);
      setLocalNotice("已追加到本地 append-only 事件；服务器状态、模型与图事实均未改变。");
    } catch {
      setLocalNotice("本地人工复核记录写入失败；服务器 finding 未发生变化。");
    } finally {
      setReviewBusy(false);
    }
  };

  const exportReport = (format: "json" | "markdown") => {
    if (reviewBusy) return;
    if (!selectedFinding || !selectedReport) {
      setLocalNotice("报告绑定校验失败，未导出不完整结果。");
      return;
    }
    try {
      const content = format === "json"
        ? serializeCoreReportJson(selectedReport)
        : serializeCoreReportMarkdown(selectedReport);
      if (onReportExport) onReportExport(selectedReport, format, content);
      else downloadReport(format, content, selectedFinding.findingHash);
    } catch {
      setLocalNotice("报告绑定校验失败，未导出不完整结果。");
    }
  };

  const latestReview = reviews.at(-1);

  return (
    <section className="core-workbench" aria-labelledby="core-workbench-title">
      <div className="core-workbench__heading">
        <div>
          <strong id="core-workbench-title" role="heading" aria-level={2}>静态图治理任务</strong>
          <span>SocialGraph-FM Core · 证据与人工复核</span>
        </div>
        <code title={graph.id}>{graph.id.slice(0, 12)}</code>
      </div>

      <fieldset className="core-task-grid">
        <legend className="sr-only">治理任务</legend>
        {TASKS.map((task) => (
          <button
            key={task.id}
            type="button"
            aria-label={task.label}
            aria-pressed={task.id === taskId}
            className={task.id === taskId ? "is-active" : ""}
            onClick={() => setTaskId(task.id)}
          >
            <strong>{task.label}</strong>
            <span>{task.description}</span>
          </button>
        ))}
      </fieldset>

      {service.state === "checking" ? (
        <p className="core-callout" role="status">正在读取 registry 能力…</p>
      ) : service.state === "unavailable" ? (
        <div className="core-callout is-warning" role="status">
          <strong>SocialGraph-FM Core 服务暂不可用（{unavailableCode}）</strong>
          <span>本地图导入与确定性分析仍可使用；这里不会生成实体级模型发现。</span>
        </div>
      ) : (
        <>
          <label className="core-field">
            <span>登记模型</span>
            <select
              value={selectedModel?.modelVersionId ?? ""}
              disabled={models.length === 0 || activeRun(runState)}
              onChange={(event) => setModelVersionId(event.target.value)}
            >
              {models.length === 0 ? <option value="">registry 暂无模型</option> : null}
              {models.filter((model) => model.tasks.includes(taskId)).map((model) => (
                <option key={model.modelVersionId} value={model.modelVersionId}>
                  {model.modelVersionId} · {modelStateLabel(model)}
                </option>
              ))}
            </select>
          </label>

          {models.length === 0 ? (
            <div className="core-callout is-warning" role="status">
              GFM 服务已连接；正式模型未就绪（registry 模型数为 0）。本地图工作区仍可正常使用。
            </div>
          ) : !selectedModel ? (
            <div className="core-callout is-warning" role="status">
              registry 中没有支持当前治理任务的 servingReady 模型。
            </div>
          ) : compatibility?.state === "candidate" ? (
            <div className="core-callout is-warning" role="status">
              模型已登记验收，但尚未 servingReady，不能发起运行。
            </div>
          ) : compatibility?.state === "blocked" ? (
            <div className="core-callout is-warning" role="status">
              {compatibility.blockers.map((blocker) => blocker.message).join(" ")}
            </div>
          ) : (
            <div className="core-callout" role="status">
              浏览器只检查静态 schema、任务、规模与完整图；特征和制品合同以服务端 POST 的权威判定为准。
            </div>
          )}

          <datalist id={`${targetListBaseId}-nodes`}>
            {nodeSuggestions.map((node) => (
              <option key={node.id} value={node.id}>{node.label}</option>
            ))}
          </datalist>
          <datalist id={`${targetListBaseId}-edges`}>
            {edgeSuggestions.map((edge) => (
              <option key={edge.id} value={edge.id}>{edge.source} → {edge.target}</option>
            ))}
          </datalist>
          <datalist id={`${targetListBaseId}-communities`}>
            {communitySuggestions.map((node) => (
              <option key={node.id} value={node.id}>{node.label}</option>
            ))}
          </datalist>

          <div className="core-targets">
            {taskId === "core.community_resilience_review" ? (
              <label className="core-field">
                <span>社区节点 ID</span>
                <input
                  type="text"
                  list={`${targetListBaseId}-communities`}
                  value={communityId}
                  placeholder={communities.length ? "输入完整社区节点 ID" : "没有 type=community 的节点"}
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(event) => setCommunityId(event.target.value)}
                />
              </label>
            ) : null}

            {taskId === "core.risk_and_trust_review" ? (
              <>
                <label className="core-field">
                  <span>复核对象类型</span>
                  <select value={riskKind} onChange={(event) => setRiskKind(event.target.value as "node" | "edge")}>
                    <option value="node">现有节点</option>
                    <option value="edge">现有关系</option>
                  </select>
                </label>
                <label className="core-field">
                  <span>{riskKind === "node" ? "节点 ID" : "关系本地 ID"}</span>
                  {riskKind === "node" ? (
                    <input
                      type="text"
                      list={`${targetListBaseId}-nodes`}
                      value={riskNodeId}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(event) => setRiskNodeId(event.target.value)}
                    />
                  ) : (
                    <input
                      type="text"
                      list={`${targetListBaseId}-edges`}
                      value={riskEdgeId}
                      placeholder={graph.edges.length ? "输入完整关系本地 ID" : "当前图没有关系"}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(event) => setRiskEdgeId(event.target.value)}
                    />
                  )}
                </label>
                {riskKind === "edge" && riskEdgeId && !selectedRiskEdgeIdentity ? (
                  <p className="core-inline-note">
                    该关系不能证明为唯一 RegisteredEdgeIdentity；仅支持全图方向明确、edgeType 与 weight 均显式且语义不重复的关系。
                  </p>
                ) : null}
              </>
            ) : null}

            {taskId === "core.collaboration_completion" ? (
              <>
                <label className="core-field">
                  <span>起点节点 ID</span>
                  <input
                    type="text"
                    list={`${targetListBaseId}-nodes`}
                    value={pairSourceId}
                    autoComplete="off"
                    spellCheck={false}
                    onChange={(event) => setPairSourceId(event.target.value)}
                  />
                </label>
                <label className="core-field">
                  <span>终点节点 ID</span>
                  <input
                    type="text"
                    list={`${targetListBaseId}-nodes`}
                    value={pairTargetId}
                    autoComplete="off"
                    spellCheck={false}
                    onChange={(event) => setPairTargetId(event.target.value)}
                  />
                </label>
                {collaborationPair?.relationState === "recorded" ? (
                  <p className="core-inline-note">
                    这两个节点已有{graph.metadata?.directedness === "directed" ? "同向" : "无向"}记录关系；静态关系补全不会运行，请选择尚未记录的节点对。
                  </p>
                ) : null}
              </>
            ) : null}

            <p className="core-inline-note">
              目标建议固定最多 {TARGET_SUGGESTION_LIMIT} 项；可直接输入完整 ID，提交前仍按不可变 GraphVersion 事实校验。
            </p>

            <label className="core-field">
              <span>相似结构案例数（0–20）</span>
              <input
                type="number"
                min={0}
                max={20}
                value={topKSimilarCases}
                onChange={(event) => setTopKSimilarCases(boundedInteger(event, 0, 20))}
              />
            </label>
            {taskId === "core.collaboration_completion" ? (
              <label className="core-field">
                <span>候选上限（1–10000）</span>
                <input
                  type="number"
                  min={1}
                  max={10_000}
                  value={candidateLimit}
                  onChange={(event) => setCandidateLimit(boundedInteger(event, 1, 10_000))}
                />
              </label>
            ) : null}
          </div>

          <div className="core-actions">
            <button type="button" className="core-primary" disabled={!canRun} onClick={submit}>
              运行治理复核
            </button>
            {detachableRun(runState) ? (
              <button type="button" onClick={() => controller.stopFollowing()}>停止跟踪</button>
            ) : null}
          </div>
        </>
      )}

      {runState.phase === "submitting" ? <p className="core-callout" role="status">正在提交静态图复核…</p> : null}
      {runState.phase === "polling" ? (
        <p className="core-callout" role="status">
          运行 {runState.binding.runId} · {runState.status.status} · {runState.status.progress}%
        </p>
      ) : null}
      {runState.phase === "loading-result" ? <p className="core-callout" role="status">运行成功，正在校验并加载不可变结果…</p> : null}
      {runState.phase === "detached" ? (
        <div className="core-callout is-warning" role="status">
          已停止跟踪运行 {runState.runId}；这里只中止浏览器轮询，服务端运行可能继续。
        </div>
      ) : null}
      {runState.phase === "failed" ? (
        <div className="core-callout is-warning" role="alert">
          <strong>{runState.code}</strong>
          <span>{safeFailureGuidance(runState.code)}</span>
        </div>
      ) : null}
      {localNotice ? <p className="core-callout" role="status">{localNotice}</p> : null}

      {succeededState && selectedFinding ? (
        <section
          ref={resultRegionRef}
          className="core-result"
          aria-label="治理复核结果"
          aria-live="polite"
          tabIndex={-1}
        >
          <div className="core-result__heading">
            <div>
              <strong role="heading" aria-level={3}>
                模型发现 #{Math.max(0, findings.indexOf(selectedFinding)) + 1}
              </strong>
              <span>{coreFindingLabel(selectedFinding.findingType)}</span>
            </div>
            {findings.length > 1 ? (
              <label className="core-field is-compact">
                <span>选择发现</span>
                <select value={selectedFinding.findingHash} onChange={(event) => setSelectedFindingHash(event.target.value)}>
                  {findings.map((finding, index) => (
                    <option key={finding.findingHash} value={finding.findingHash}>#{index + 1} · {finding.subjectIds.join(" / ")}</option>
                  ))}
                </select>
              </label>
            ) : null}
          </div>

          <div className="core-score-grid">
            <span><small>模型分数</small><strong>{selectedFinding.score.score.toFixed(4)}</strong></span>
            {selectedFinding.calibratedConfidence.schemaVersion === "socialgraph-fm.core-regression-confidence-interval/1.0" ? (
              <>
                <span>
                  <small>回归区间（非概率）</small>
                  <strong>
                    {selectedFinding.calibratedConfidence.lowerBound.toFixed(4)} – {selectedFinding.calibratedConfidence.upperBound.toFixed(4)}
                  </strong>
                </span>
                <span>
                  <small>验证残差覆盖</small>
                  <strong>
                    验证残差覆盖率 {(selectedFinding.calibratedConfidence.coverage * 100).toFixed(2)}%
                    {` · n=${selectedFinding.calibratedConfidence.validationCount}`}
                  </strong>
                </span>
              </>
            ) : (
              <span><small>校准置信度</small><strong>{selectedFinding.calibratedConfidence.value.toFixed(4)}</strong></span>
            )}
            <span><small>方法</small><strong>{selectedFinding.calibratedConfidence.method}</strong></span>
            <span><small>服务器状态</small><strong>待人工复核</strong></span>
          </div>
          <p className="core-safety">
            {selectedFinding.calibratedConfidence.schemaVersion === "socialgraph-fm.core-regression-confidence-interval/1.0"
              ? "验证残差覆盖描述验证集上的回归区间覆盖，不是概率，且必须待人工复核；本结果不授权自动处罚或执法，也不预测未来事件。"
              : "校准置信度不是违规、风险或事实为真的概率；本结果不授权自动处罚或执法，且不预测未来事件。"}
          </p>
          <p className="core-inline-note">服务器状态：待人工复核（pending-human-review）。</p>

          <section className="core-evidence" aria-labelledby="core-evidence-title">
            <strong id="core-evidence-title">结构与模型证据</strong>
            {selectedFinding.evidence.map((evidence) => (
              <article key={evidence.evidenceHash}>
                <div><strong>{evidence.metric}</strong><span>{evidence.sourceType}</span></div>
                <code>{evidence.valueCanonicalJson}</code>
                <small>节点：{evidence.nodeIds.join("、") || "无"} · 关系：{evidence.edgeIds.join("、") || "无"}</small>
                {evidence.limitations.map((limitation) => <p key={limitation}>{limitation}</p>)}
              </article>
            ))}
          </section>

          <section className="core-similar" aria-labelledby="core-similar-title">
            <strong id="core-similar-title">相似结构案例</strong>
            {selectedFinding.similarCases.length === 0 ? <p>本次结果没有返回相似结构案例。</p> : (
              <ul>
                {selectedFinding.similarCases.map((similar) => (
                  <li key={similar.similarCaseHash}>
                    {similar.sourceKind} · 相似度 {similar.similarity.toFixed(4)} · <code>{similar.similarCaseHash}</code>
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="core-limitations" aria-labelledby="core-limitations-title">
            <strong id="core-limitations-title">模型限制</strong>
            <ul>{selectedFinding.limitations.map((limitation) => <li key={limitation}>{limitation}</li>)}</ul>
          </section>

          <details className="core-provenance">
            <summary>不可变来源与哈希</summary>
            <dl>
              <div><dt>runId</dt><dd><code>{succeededState.binding.runId}</code></dd></div>
              <div><dt>graphVersionId</dt><dd><code>{succeededState.binding.graphVersionId}</code></dd></div>
              <div><dt>graphVersionHash</dt><dd><code>{succeededState.result.graphVersionHash}</code></dd></div>
              <div><dt>modelVersionId</dt><dd><code>{succeededState.binding.modelVersionId}</code></dd></div>
              <div><dt>modelVersionHash</dt><dd><code>{succeededState.result.modelVersionHash}</code></dd></div>
              <div><dt>resultHash</dt><dd><code>{succeededState.result.resultHash}</code></dd></div>
              <div><dt>findingHash</dt><dd><code>{selectedFinding.findingHash}</code></dd></div>
              <div><dt>publicRequestHash（浏览器可重算）</dt><dd><code>{succeededState.binding.publicRequestHash}</code></dd></div>
              <div><dt>serverRequestHash（服务返回，浏览器不可重算隐藏 envelope）</dt><dd><code>{succeededState.binding.serverRequestHash}</code></dd></div>
            </dl>
          </details>

          <section className="core-review" aria-labelledby="core-review-title">
            <strong id="core-review-title">本地人工复核记录</strong>
            <p>
              {latestReview
                ? `本地人工复核：${latestReview.decision === "confirmed" ? "已确认" : "已驳回"}`
                : "尚无本地人工复核记录。"}
               本地决定不会改写服务器 finding，其状态仍为 pending-human-review。
            </p>
            <div className="core-actions">
              <button type="button" disabled={reviewBusy} onClick={() => void review("confirmed")}>本地确认</button>
              <button type="button" disabled={reviewBusy} onClick={() => void review("rejected")}>本地驳回</button>
            </div>
            {reviews.length ? (
              <ol>
                {reviews.map((item) => (
                  <li key={item.recordHash}>{item.reviewedAt} · {item.decision === "confirmed" ? "确认" : "驳回"} · <code>{item.recordHash}</code></li>
                ))}
              </ol>
            ) : null}
          </section>

          <div className="core-actions">
            <button type="button" disabled={reviewBusy} onClick={() => exportReport("json")}>导出 JSON 报告</button>
            <button type="button" disabled={reviewBusy} onClick={() => exportReport("markdown")}>导出 Markdown 报告</button>
          </div>
        </section>
      ) : null}
    </section>
  );
}
