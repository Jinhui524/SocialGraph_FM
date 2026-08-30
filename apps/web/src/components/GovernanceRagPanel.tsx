import {
  Brain,
  CheckCircle,
  CircleNotch,
  Graph,
  MagnifyingGlass,
  Play,
  Scales,
  ShieldCheck,
  TreeStructure,
  UsersThree,
  X,
} from "@phosphor-icons/react";
import { useEffect, useLayoutEffect, useRef, useState, type FormEvent } from "react";

import { SocialGraphApiError } from "../services/apiClient";
import type {
  GovernanceAssistantDispatchResponse,
  GovernanceAnswerMode,
  GovernanceSimilarCasesResponse,
  GovernanceSkillsClientLike,
  GovernanceSkillsContext,
} from "../types/governanceSkills";
import { SafeMarkdown } from "./SafeMarkdown";

type PanelView = "assistant" | "cases";
type ReportTask = "overview" | "node" | "coordination" | "draft";
type RequestState<T> =
  | { readonly state: "idle" }
  | { readonly state: "loading"; readonly previous?: T }
  | { readonly state: "ready"; readonly value: T }
  | { readonly state: "error"; readonly message: string; readonly previous?: T };

interface ContextOwner {
  readonly key: string;
  readonly epoch: number;
}

interface ActiveRequest extends ContextOwner {
  readonly controller: AbortController;
}

const VIEW_ITEMS = [
  { id: "assistant", label: "研判报告", Icon: Brain },
  { id: "cases", label: "历史案例", Icon: Scales },
] as const;

const REPORT_TASKS: ReadonlyArray<{
  readonly id: ReportTask;
  readonly label: string;
  readonly description: string;
  readonly prompt: string;
  readonly answerMode: GovernanceAnswerMode;
  readonly requires: "run" | "node" | "case";
  readonly Icon: typeof Graph;
}> = [
  {
    id: "overview",
    label: "全局态势报告",
    description: "整理重点风险节点与人工复核顺序",
    prompt: "请生成当前网络的全局态势报告，重点列出高关注候选、关系证据和下一步人工复核顺序，不输出风险分布概览。",
    answerMode: "analysis_summary",
    requires: "run",
    Icon: Graph,
  },
  {
    id: "node",
    label: "当前账号证据报告",
    description: "区分事实、模型信号与待补证据",
    prompt: "请生成当前选中账号的证据报告，区分已登记事实、模型风险排序、结构线索和仍需补充核验的信息。",
    answerMode: "evidence_requirements",
    requires: "node",
    Icon: ShieldCheck,
  },
  {
    id: "coordination",
    label: "群组与关系研判报告",
    description: "核对风险群组、事实关系与潜在线索",
    prompt: "请生成群组与关系研判报告，分别说明风险群组、事实关系与未登记为事实边的潜在线索，并给出核验顺序。",
    answerMode: "coordination_summary",
    requires: "run",
    Icon: UsersThree,
  },
  {
    id: "draft",
    label: "人工研判草稿",
    description: "形成可继续修订的研判单草稿",
    prompt: "请基于当前研判单生成一份人工研判草稿，严格区分图事实、模型发现、派生线索和人工结论。",
    answerMode: "case_draft",
    requires: "case",
    Icon: TreeStructure,
  },
];

const TARGET_KIND_LABELS: Readonly<Record<string, string>> = {
  node: "账号",
  relation: "关系",
  group: "群组",
};

const TRACE_LABELS: Readonly<Record<string, string>> = {
  inspect_graph: "图谱概况",
  get_evidence_subgraph: "关联证据",
  discover_coordination_groups: "协同行为群组",
  rank_coordination_relations: "重点关系",
  retrieve_similar_cases: "相似案例",
  get_model_dataset_cards: "分析说明",
};

function describeError(error: unknown): string {
  if (error instanceof SocialGraphApiError && /CASE_HASH_STALE|CASE_REVISION/u.test(error.code)) {
    return "研判单已更新，请按最新人工记录重新生成报告。";
  }
  if (error instanceof SocialGraphApiError) return "请求未完成，请检查当前数据后重试。";
  return "请求未完成，请稍后重试。";
}

function describeSimilarCaseError(error: unknown): string {
  if (!(error instanceof SocialGraphApiError)) return "历史案例服务暂时不可用，请稍后重试。";
  const signature = `${error.code} ${error.message}`.toUpperCase();
  if (/SIMILAR_CASE_TARGETS_REQUIRED|TARGETS_REQUIRED|NO_CASE_ITEMS|EMPTY/u.test(signature)) {
    return "当前研判单还没有治理对象，加入对象后即可检索。";
  }
  if (/SIMILAR_CASE_INDEX_NOT_READY|INDEX_NOT_READY/u.test(signature)) {
    return "当前研判单尚未形成可检索索引，请先完成审结，或选择当前对象直接检索。";
  }
  if (/SIMILAR_CASE_STATE_UNSUPPORTED|STATE_UNSUPPORTED/u.test(signature)) {
    return "当前研判单状态暂不支持历史案例检索，请选择活动对象或已审结案例。";
  }
  if (error.status === 404 || /NOT_FOUND|STALE|EXPIRED|IDENTITY|MISMATCH/u.test(signature)) {
    return "当前研判对象已失效，请重新选择后检索。";
  }
  if (error.status >= 500 || /UNAVAILABLE|TIMEOUT/u.test(signature)) {
    return "历史案例服务暂时不可用，请稍后重试。";
  }
  if (error.status === 409) return "当前研判上下文与案例索引不一致，请刷新对象后重试。";
  return "当前研判上下文尚未满足检索条件，请选择对象后重试。";
}

function contextCaseItemCount(context: GovernanceSkillsContext | null): number {
  const value = (context as (GovernanceSkillsContext & { readonly caseItemCount?: number }) | null)?.caseItemCount;
  return typeof value === "number" && Number.isInteger(value) && value >= 0 ? value : 0;
}

function similarQueryIdentity(context: GovernanceSkillsContext | null): string | null {
  if (!context) return null;
  if (context.runId && context.selectedTarget) {
    return JSON.stringify(["target", context.runId, context.selectedTarget.kind, context.selectedTarget.targetId, context.caseHash ?? null]);
  }
  if (context.caseId && contextCaseItemCount(context) > 0) return JSON.stringify(["case", context.caseId, context.caseHash ?? null]);
  return null;
}

function contextIdentityKey(context: GovernanceSkillsContext | null): string {
  if (!context) return "no-context";
  return JSON.stringify([
    context.graph.artifactId,
    context.graph.datasetContentHash,
    context.graph.graphVersionHash,
    context.model.modelVersionId,
    context.model.modelStateHash,
    context.runId ?? null,
    context.caseId ?? null,
    context.caseHash ?? null,
    contextCaseItemCount(context),
    [...new Set(context.selectedNodeIds ?? [])].sort(),
    context.selectedTarget ? [context.selectedTarget.kind, context.selectedTarget.targetId] : null,
  ]);
}

function sameOwner(left: ContextOwner, right: ContextOwner): boolean {
  return left.key === right.key && left.epoch === right.epoch;
}

function shortHash(value: string): string {
  return `${value.slice(0, 8)}…${value.slice(-6)}`;
}

function reportTaskBlocker(context: GovernanceSkillsContext | null, task: typeof REPORT_TASKS[number]): string | null {
  if (!context) return "当前会话尚未形成治理上下文";
  if (task.requires === "run" && !context.runId) return "请先在对话研究中完成分析";
  if (task.requires === "node" && (!context.runId || context.selectedTarget?.kind !== "node")) return "请先选择已分析账号";
  if (task.requires === "case" && !context.caseId) return "请先建立或选择研判单";
  return null;
}

export interface GovernanceRagPanelProps {
  readonly client: GovernanceSkillsClientLike;
  readonly context: GovernanceSkillsContext | null;
  readonly onClose: () => void;
  readonly embedded?: boolean;
}

export function GovernanceRagPanel({ client, context, onClose, embedded = false }: GovernanceRagPanelProps) {
  const contextKey = contextIdentityKey(context);
  const similarQueryKey = similarQueryIdentity(context);
  const [view, setView] = useState<PanelView>("assistant");
  const [assistantMessage, setAssistantMessage] = useState("");
  const [activeReportTask, setActiveReportTask] = useState<ReportTask | null>(null);
  const [showReportTasks, setShowReportTasks] = useState(true);
  const [assistant, setAssistant] = useState<RequestState<GovernanceAssistantDispatchResponse>>({ state: "idle" });
  const [similar, setSimilar] = useState<RequestState<GovernanceSimilarCasesResponse>>({ state: "idle" });
  const [resetNotice, setResetNotice] = useState<string | null>(null);
  const contextOwnerRef = useRef<ContextOwner>({ key: contextKey, epoch: 0 });
  const previousContextKeyRef = useRef(contextKey);
  const previousCaseHashRef = useRef(context?.caseHash);
  const requestRef = useRef<ActiveRequest | null>(null);
  const similarCacheRef = useRef(new Map<string, GovernanceSimilarCasesResponse>());
  const similarLatestIndexRef = useRef(new Map<string, string>());

  const cachedSimilarFor = (queryKey: string | null): GovernanceSimilarCasesResponse | undefined => {
    if (!queryKey) return undefined;
    const indexHash = similarLatestIndexRef.current.get(queryKey);
    return indexHash ? similarCacheRef.current.get(`${queryKey}:${indexHash}`) : undefined;
  };

  useLayoutEffect(() => {
    if (previousContextKeyRef.current !== contextKey) setResetNotice(
      previousCaseHashRef.current !== context?.caseHash && Boolean(previousCaseHashRef.current || context?.caseHash)
        ? "研判单已更新，请按最新人工记录重新生成报告"
        : "对象已变化，上一结果已安全清除",
    );
    previousContextKeyRef.current = contextKey;
    previousCaseHashRef.current = context?.caseHash;
    const owner = { key: contextKey, epoch: contextOwnerRef.current.epoch + 1 };
    contextOwnerRef.current = owner;
    requestRef.current?.controller.abort();
    requestRef.current = null;
    setAssistant({ state: "idle" });
    setActiveReportTask(null);
    setShowReportTasks(true);
    const cachedSimilar = cachedSimilarFor(similarQueryKey);
    setSimilar(cachedSimilar ? { state: "ready", value: cachedSimilar } : { state: "idle" });
    return () => {
      if (!sameOwner(contextOwnerRef.current, owner)) return;
      requestRef.current?.controller.abort();
      requestRef.current = null;
    };
  }, [contextKey]);

  useEffect(() => {
    if (!resetNotice) return;
    const timer = window.setTimeout(() => setResetNotice(null), 3600);
    return () => window.clearTimeout(timer);
  }, [resetNotice]);

  const ownerForCurrentRender = (): ContextOwner | null => {
    const owner = contextOwnerRef.current;
    return owner.key === contextKey ? owner : null;
  };

  const nextRequest = (owner: ContextOwner): ActiveRequest => {
    requestRef.current?.controller.abort();
    const request = { ...owner, controller: new AbortController() };
    requestRef.current = request;
    return request;
  };

  const isActiveRequest = (request: ActiveRequest): boolean => !request.controller.signal.aborted
    && requestRef.current === request
    && sameOwner(contextOwnerRef.current, request);

  const requestAssistantReport = (message: string, task: typeof REPORT_TASKS[number] | null = null) => {
    const owner = ownerForCurrentRender();
    if (!context || !owner || !message.trim() || assistant.state === "loading") return;
    const request = nextRequest(owner);
    setActiveReportTask(task?.id ?? null);
    setAssistant({ state: "loading" });
    setShowReportTasks(false);
    client.dispatchAssistant(context, message, {
      intent: "answer",
      ...(task ? { answerMode: task.answerMode } : {}),
    }, request.controller.signal)
      .then((value) => { if (isActiveRequest(request)) setAssistant({ state: "ready", value }); })
      .catch((error) => { if (isActiveRequest(request)) setAssistant({ state: "error", message: describeError(error) }); });
  };

  const askAssistant = (event: FormEvent) => {
    event.preventDefault();
    requestAssistantReport(assistantMessage, null);
  };

  const searchSimilar = () => {
    const owner = ownerForCurrentRender();
    if (!context || !owner || !similarQueryKey || similar.state === "loading") return;
    const request = nextRequest(owner);
    const previous = similar.state === "ready"
      ? similar.value
      : similar.state === "error"
        ? similar.previous
        : cachedSimilarFor(similarQueryKey);
    setSimilar({ state: "loading", ...(previous ? { previous } : {}) });
    const query = context.runId && context.selectedTarget
      ? { runId: context.runId, kindEntries: [{ kind: context.selectedTarget.kind, targetIds: [context.selectedTarget.targetId] }] }
      : {};
    client.searchSimilarCases(context, query, request.controller.signal)
      .then((value) => {
        if (!isActiveRequest(request)) return;
        similarCacheRef.current.set(`${similarQueryKey}:${value.indexHash}`, value);
        similarLatestIndexRef.current.set(similarQueryKey, value.indexHash);
        setSimilar({ state: "ready", value });
      })
      .catch((error) => {
        if (isActiveRequest(request)) setSimilar({ state: "error", message: describeSimilarCaseError(error), ...(previous ? { previous } : {}) });
      });
  };

  const canSearchSimilar = Boolean(similarQueryKey);
  const similarValue = similar.state === "ready"
    ? similar.value
    : similar.state === "loading" || similar.state === "error"
      ? similar.previous
      : undefined;
  const similarIdleHint = context?.caseId && !context?.selectedTarget && contextCaseItemCount(context) === 0
    ? "当前研判单还没有治理对象，加入对象后即可检索。"
    : canSearchSimilar
      ? "可从当前对象查找相似处置经验。"
      : "选择治理对象后开放检索。";

  return <aside id="governance-rag-panel" className={`governance-rag-panel ${embedded ? "is-embedded" : ""}`} aria-label="案例研判助手">
    <header><div><strong>研判助手</strong><span>将当前事实与模型发现整理为可复核报告</span></div><button type="button" aria-label="关闭案例研判助手" title="关闭" onClick={onClose}><X size={17} /></button></header>

    {resetNotice ? <p className="governance-context-reset" role="status"><CheckCircle />{resetNotice}</p> : null}

    <nav className="governance-rag-tabs" role="tablist" aria-label="案例研判工具区">{VIEW_ITEMS.map(({ id, label, Icon }) => <button key={id} type="button" role="tab" aria-selected={view === id} onClick={() => setView(id)}><Icon size={15} />{label}</button>)}</nav>

    {view === "assistant" ? <section className={`governance-rag-view governance-rag-view--assistant ${assistant.state === "ready" ? "has-report" : ""}`} role="tabpanel" aria-label="研判报告">
      <div className="governance-assistant-content">
      {showReportTasks ? <div className="governance-report-task-area">
        <div className="governance-report-intro"><strong>选择报告任务</strong><span>报告依据已登记事实与绑定分析结果生成，仍需人工复核后形成结论。</span></div>
        <div className="governance-report-tasks" role="group" aria-label="研判报告任务">{REPORT_TASKS.map((task) => {
          const TaskIcon = task.Icon;
          const blocker = reportTaskBlocker(context, task);
          const pending = assistant.state === "loading" && activeReportTask === task.id;
          return <button key={task.id} type="button" disabled={Boolean(blocker) || assistant.state === "loading"} title={blocker ?? task.description} onClick={() => requestAssistantReport(task.prompt, task)}><TaskIcon size={20} /><span><strong>{task.label}</strong><small>{blocker ?? task.description}</small></span>{pending ? <CircleNotch className="spin" /> : <Play size={14} />}</button>;
        })}</div>
      </div> : <><button className="governance-report-change" type="button" onClick={() => setShowReportTasks(true)}>更换报告任务</button>
      <div className="governance-report-document">
        {assistant.state === "loading" ? <p className="governance-report-progress"><CircleNotch className="spin" />正在核对事实、模型信号与登记资料…</p> : null}
        {assistant.state === "error" ? <p className="is-error">{assistant.message} 当前图谱与复核记录未被修改，可以稍后重试。</p> : null}
        {assistant.state === "ready" ? <article className="governance-assistant-answer"><header><div><strong>{REPORT_TASKS.find((task) => task.id === activeReportTask)?.label ?? "研判报告"}</strong><span>{assistant.value.generationMode === "deterministic_report" || assistant.value.deterministicFallback ? "本地可审计报告" : "智能整理"}</span></div></header><div className="governance-report-page"><SafeMarkdown text={assistant.value.answer} /><details className="governance-report-trace"><summary><ShieldCheck />依据来源</summary>{assistant.value.skillCalls.length ? <ul>{assistant.value.skillCalls.map((trace, index) => <li key={`${trace.skill}-${index}`}><CheckCircle /><span>{TRACE_LABELS[trace.skill] ?? "只读分析"}</span><code title={trace.resultHash}>{shortHash(trace.resultHash)}</code></li>)}</ul> : <p>本次回答依据当前绑定上下文生成，未执行额外只读调用。</p>}{assistant.value.evidenceRefs?.length ? <div className="governance-report-sources">{assistant.value.evidenceRefs.map((reference, index) => <span key={`${reference.hash}-${index}`} title={reference.hash}>{reference.label} · {shortHash(reference.hash)}</span>)}</div> : null}{assistant.value.citedHashes.length ? <div className="governance-report-sources">{assistant.value.citedHashes.map((hash) => <span key={hash} title={hash}>来源指纹 · {shortHash(hash)}</span>)}</div> : null}</details></div></article> : null}
      </div></>}
      </div>

      <form className="governance-assistant-composer" onSubmit={askAssistant}><label className="sr-only" htmlFor="governance-assistant-message">输入研判问题</label><Brain size={17} /><textarea id="governance-assistant-message" rows={2} value={assistantMessage} maxLength={2_000} onChange={(event) => setAssistantMessage(event.target.value)} placeholder="继续追问当前对象的事实关系、潜在线索或核验缺口" /><button type="submit" disabled={!context || !assistantMessage.trim() || assistant.state === "loading"}>{assistant.state === "loading" ? <CircleNotch className="spin" /> : <Play />}生成报告</button></form>
    </section> : null}

    {view === "cases" ? <section className="governance-rag-view governance-rag-view--cases" role="tabpanel" aria-label="历史案例"><div className="governance-rag-panel__section-title"><strong>历史案例</strong><span>仅检索已审结记录</span></div><button className="governance-similar-search" type="button" disabled={!canSearchSimilar || similar.state === "loading"} title={!canSearchSimilar ? similarIdleHint : "检索与当前对象接近的已审结案例"} onClick={searchSimilar}>{similar.state === "loading" ? <CircleNotch className="spin" /> : <MagnifyingGlass />}检索相似历史案例</button>{similar.state === "idle" ? <p>{similarIdleHint}</p> : null}{similar.state === "error" ? <p className="is-error">{similar.message}{similar.previous ? " 上次成功结果已保留。" : ""}</p> : null}{similarValue && !similarValue.items.length ? <p>没有同类已审结案例。</p> : null}{similarValue ? <ol className="governance-similar-results">{similarValue.items.map((item, index) => <li key={item.caseId}><header><strong>历史案例 {String(index + 1).padStart(2, "0")}</strong><span>{(item.score * 100).toFixed(1)}%</span></header><div className="governance-similarity-components"><div><span>语义接近 {(item.components.embedding * 100).toFixed(0)}%</span><small>对象内容与行为语义</small></div><div><span>结构接近 {(item.components.structure * 100).toFixed(0)}%</span><small>邻域与连接模式</small></div><div><span>关系接近 {(item.components.modality * 100).toFixed(0)}%</span><small>关系类型构成</small></div></div><small>{TARGET_KIND_LABELS[item.kindKey] ?? "治理对象"} · {item.concludedAt.slice(0, 10)} 审结</small><details className="governance-case-hashes"><summary>依据来源</summary><small title={item.modelStateHash}>模型状态 {shortHash(item.modelStateHash)}</small><small title={item.graphVersionHash}>图谱版本 {shortHash(item.graphVersionHash)}</small><small title={item.recordHash}>案例记录 {shortHash(item.recordHash)}</small></details></li>)}</ol> : null}</section> : null}
  </aside>;
}
