import {
  ArrowClockwise,
  ArrowRight,
  BracketsCurly,
  CheckCircle,
  CircleNotch,
  FileCsv,
  FileZip,
  Graph,
  ShieldCheck,
  Sparkle,
  WarningCircle,
  XCircle,
} from "@phosphor-icons/react";
import { useMemo, useState } from "react";
import {
  AssistantActivityView,
} from "../../components/AssistantActivity";
import { AssistantGuidance } from "../../components/AssistantGuidance";
import { SafeMarkdown } from "../../components/SafeMarkdown";
import type { GraphVersion } from "../../types/graph";
import { ORDINARY_PRESENTATION_COPY } from "../app-shell/presentationCopy";
import {
  GOVERNANCE_ANALYSIS_STAGES,
  assistantActivityForEntry,
  assistantGuidanceStateForEntry,
  canOpenGovernanceReview,
  governanceProgressStageIndex,
  type ChatEntry,
} from "../assistant/chatModel";
import { buildAnalysisResultMarkdown } from "../governance/reports";
import type { ImportViewState, PendingTargetResolution } from "./importModel";

const formatBytes = (bytes: number) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
};

export function FileBadge({ file }: { file: { name: string; size: number } }) {
  const isJson = file.name.toLowerCase().endsWith(".json");
  const isZip = file.name.toLowerCase().endsWith(".zip");
  const Icon = isZip ? FileZip : isJson ? BracketsCurly : FileCsv;
  return (
    <div className="file-badge" title={file.name}>
      <span className={`file-badge__icon ${isJson ? "is-json" : ""}`}>
        <Icon size={14} weight="light" />
      </span>
      <span className="file-badge__copy">
        <strong>{file.name}</strong>
        <small>{formatBytes(file.size)}</small>
      </span>
      <CheckCircle size={14} weight="fill" className="file-badge__check" aria-label="附件已就绪" />
    </div>
  );
}

export function ImportTimeline({ state }: { state: ImportViewState }) {
  if (
    state.kind === "idle"
    || state.kind === "roles"
    || state.kind === "mapping"
    || state.kind === "review"
    || state.kind === "success"
  ) return null;
  const parsingLabel = state.kind === "parsing"
    ? state.stage === "inspect"
      ? `正在检查 ${state.fileName}`
      : state.stage === "parse"
        ? `正在解析 ${state.fileName}`
        : `正在生成 ${state.fileName}`
    : null;
  return (
    <div
      className={`import-timeline import-timeline--${state.kind}`}
      aria-live="polite"
      role={state.kind === "error" ? "alert" : "status"}
      title={state.kind === "error" ? state.message : parsingLabel ?? undefined}
    >
      {state.kind === "parsing" ? <CircleNotch size={14} className="spin" /> : <XCircle size={14} weight="fill" />}
      <span>{state.kind === "error" ? state.message : parsingLabel}</span>
    </div>
  );
}

export function GraphBuildReviewCard({
  state,
  onConfirm,
  onEdit,
  onCancel,
}: {
  state: Extract<ImportViewState, { kind: "review" }>;
  onConfirm: () => void;
  onEdit: () => void;
  onCancel: () => void;
}) {
  const version = state.run.graphVersion;
  const errors = state.run.issues.filter((issue) => issue.severity === "error");
  const warnings = state.run.issues.filter((issue) => issue.severity !== "error");
  return (
    <section className="mapping-card" aria-labelledby="graph-build-review-title">
      <div className="mapping-card__heading">
        <span className="mapping-icon"><ShieldCheck size={20} weight="light" /></span>
        <div>
          <strong id="graph-build-review-title">确认图谱</strong>
          <p>请确认关系方向与关键字段；保存前不会替换当前图谱。</p>
        </div>
      </div>
      <details className="assistant-technical-details" open={errors.length > 0}>
        <summary>{ORDINARY_PRESENTATION_COPY.graphReviewDetails}</summary>
        <div className="diagnostic-grid">
        <div className="diagnostic-row"><strong>输入形态</strong><code>{state.spec.inputShape}</code></div>
        <div className="diagnostic-row"><strong>方向</strong><code>{state.spec.directionPolicy}</code></div>
        {state.spec.nodeMapping ? <div className="diagnostic-row"><strong>节点字段</strong><code>{state.spec.nodeMapping.id} / {state.spec.nodeMapping.label ?? "ID 作为显示名"} / {state.spec.nodeMapping.type ?? "未分类"}</code></div> : null}
        <div className="diagnostic-row"><strong>端点</strong><code>{state.spec.edgeMapping ? `${state.spec.edgeMapping.source} → ${state.spec.edgeMapping.target}` : "标准图元数据"}</code></div>
        {state.spec.edgeMapping ? <div className="diagnostic-row"><strong>关系字段</strong><code>{state.spec.edgeMapping.edgeType ?? "无类型"} / {state.spec.edgeMapping.weight ?? "无权重"} / {state.spec.edgeMapping.timestamp ?? "无时间"}</code></div> : null}
        <div className="diagnostic-row"><strong>关系策略</strong><code>{state.spec.duplicateEdgePolicy} / {state.spec.selfLoopPolicy}</code></div>
        <div className="diagnostic-row"><strong>时间格式</strong><code>{state.spec.timeFormat}</code></div>
        <div className="diagnostic-row"><strong>草稿摘要</strong><code>{version.summary.nodeCount} 节点 / {version.summary.edgeCount} 关系</code></div>
        <div className="diagnostic-row"><strong>内容哈希</strong><code>{version.contentHash?.slice(0, 16)}…</code></div>
        <div className="diagnostic-row"><strong>质量报告</strong><code>{errors.length} 错误 / {warnings.length} 提示</code></div>
        </div>
        {state.warnings.length ? <p className="warning-text">{state.warnings.join("；")}</p> : null}
        {state.run.issues.length ? (
        <details>
          <summary>展开质量报告（{state.run.issues.length}）</summary>
          <ul>
            {state.run.issues.map((issue, index) => (
              <li key={`${issue.code}:${issue.row ?? "graph"}:${index}`}>
                [{issue.severity}] {issue.message}{issue.row ? `（第 ${issue.row} 行）` : ""}
              </li>
            ))}
          </ul>
        </details>
        ) : null}
      </details>
      <div className="mapping-card__actions">
        <button className="secondary-button" type="button" onClick={onCancel}>取消，不修改当前图</button>
        {state.spec.inputShape !== "standard_graph" ? <button className="secondary-button" type="button" onClick={onEdit}>返回修改字段</button> : null}
        <button className="primary-button primary-button--small" type="button" disabled={errors.length > 0} onClick={onConfirm}>
          确认图谱 <ArrowRight size={16} />
        </button>
      </div>
    </section>
  );
}

export function TargetResolutionCard({
  graph,
  pending,
  onApply,
  onCancel,
}: {
  graph: GraphVersion;
  pending: PendingTargetResolution;
  onApply: (nodeIds: readonly string[]) => void;
  onCancel: () => void;
}) {
  const [selected, setSelected] = useState<Record<string, string>>({});
  const nodeById = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph]);
  const complete = pending.resolutions.every((resolution) => Boolean(selected[resolution.term]));

  return (
    <section className="target-resolution-card" aria-labelledby="target-resolution-title">
      <div className="target-resolution-card__heading">
        <span className="mapping-icon"><Graph size={20} weight="light" /></span>
        <div>
          <strong id="target-resolution-title">请选择自然语言中指向的节点</strong>
          <p>出现同名或相似节点时只在本地消歧；候选信息不会发送给 LLM。</p>
        </div>
      </div>
      <div className="target-resolution-groups">
        {pending.resolutions.map((resolution) => (
          <fieldset key={resolution.term}>
            <legend>“{resolution.term}”</legend>
            {resolution.candidateNodeIds.slice(0, 8).map((nodeId) => {
              const node = nodeById.get(nodeId);
              return (
                <label key={nodeId}>
                  <input
                    type="radio"
                    name={`target-${resolution.term}`}
                    value={nodeId}
                    checked={selected[resolution.term] === nodeId}
                    onChange={() => setSelected((current) => ({ ...current, [resolution.term]: nodeId }))}
                  />
                  <span><strong>{node?.label ?? nodeId}</strong><small>{node?.type ?? "未分类"} · {nodeId}</small></span>
                </label>
              );
            })}
          </fieldset>
        ))}
      </div>
      <div className="mapping-card__actions">
        <button className="secondary-button" type="button" onClick={onCancel}>保持当前视图</button>
        <button
          className="primary-button primary-button--small"
          type="button"
          disabled={!complete}
          onClick={() => onApply(Object.values(selected))}
        >
          应用到交互图 <ArrowRight size={16} />
        </button>
      </div>
    </section>
  );
}

export function UserEntry({ entry }: { entry: Extract<ChatEntry, { role: "user" }> }) {
  const files = entry.files ?? (entry.file ? [entry.file] : []);
  return (
    <article className="chat-message chat-message--user">
      <div className="message-meta">
        <span className="user-avatar" aria-label="你">你</span>
      </div>
      {files.length ? (
        <div className="user-message-attachments">
          {files.map((file) => (
            <FileBadge file={file} key={`${file.name}:${file.size}`} />
          ))}
        </div>
      ) : null}
      <div className="user-message-bubble">
        <p>{entry.text}</p>
      </div>
    </article>
  );
}

export function publicAssistantCopy(text: string): string {
  return text
    .replace(/[ \t]*Global 治理运行/gu, "治理分析运行")
    .replace(/[ \t]*Global 分析运行/gu, "治理分析运行")
    .replace(/[ \t]*Global 风险排序/gu, "基础风险排序")
    .replace(/[ \t]*Global 基线/gu, "基础风险排序");
}

export function AssistantEntry({
  entry,
  onRetry,
  onConfirm,
  activeGovernanceRunId,
  onOpenReview,
  onLocateGraph,
}: {
  entry: Extract<ChatEntry, { role: "assistant" }>;
  onRetry?: (text: string) => void;
  onConfirm?: (entry: Extract<ChatEntry, { role: "assistant" }>) => void;
  activeGovernanceRunId?: string;
  onOpenReview?: (runId: string) => void;
  onLocateGraph?: (entry: Extract<ChatEntry, { role: "assistant" }>) => void;
}) {
  const resultText = buildAnalysisResultMarkdown(entry.run);
  const activity = assistantActivityForEntry(entry);
  const guidanceState = assistantGuidanceStateForEntry(entry);
  const governanceStageIndex = entry.governanceProgress
    ? governanceProgressStageIndex(entry.governanceProgress.stage)
    : -1;
  return (
    <article className="chat-message chat-message--assistant" data-message-id={entry.id}>
      <div className="message-meta">
        <span className="assistant-avatar"><Sparkle size={15} weight="fill" /></span>
        <strong>SocialGraph-FM 助手</strong>
      </div>
      <div className={`assistant-card is-${entry.state}`}>
        {activity && !entry.governanceProgress ? <AssistantActivityView activity={activity} /> : null}
        {entry.state !== "working" || entry.governanceProgress ? <div className="assistant-card__state">
          {entry.state === "warning" ? <WarningCircle size={18} weight="fill" /> : null}
          {entry.state === "error" ? <XCircle size={18} weight="fill" /> : null}
          <SafeMarkdown text={publicAssistantCopy(entry.text)} />
        </div> : null}
        {guidanceState ? <AssistantGuidance state={guidanceState} /> : null}
        {entry.governanceProgress ? <section className={`governance-run-progress is-${entry.state}`} aria-label="治理分析进度">
          <div className="governance-run-progress__summary">
            <strong>{entry.governanceProgress.stage === "completed"
              ? "五阶段分析完成"
              : entry.state === "error"
                ? `第 ${governanceStageIndex + 1} / ${GOVERNANCE_ANALYSIS_STAGES.length} 阶段未完成`
                : entry.governanceProgress.stage === "reporting"
                  ? "正在整理分析结论"
                  : `正在执行第 ${governanceStageIndex + 1} / ${GOVERNANCE_ANALYSIS_STAGES.length} 阶段`}</strong>
            <span>{Math.max(0, Math.min(100, entry.governanceProgress.progress))}%</span>
          </div>
          <progress max={100} value={Math.max(0, Math.min(100, entry.governanceProgress.progress))} aria-label={`治理分析完成 ${Math.max(0, Math.min(100, entry.governanceProgress.progress))}%`} />
          <ol>
            {GOVERNANCE_ANALYSIS_STAGES.map((stage, index) => <li
              className={governanceStageIndex > index || entry.governanceProgress!.stage === "completed"
                ? "is-complete"
                : governanceStageIndex === index
                  ? "is-active"
                  : ""}
              key={stage.id}
            ><span>{index + 1}</span><small>{stage.label}</small></li>)}
          </ol>
        </section> : null}
        {entry.state !== "working" && resultText ? <div className="analysis-result"><SafeMarkdown text={resultText} /></div> : null}
        {entry.retryText && onRetry ? (
          <button className="secondary-button assistant-retry" type="button" onClick={() => onRetry(entry.retryText!)}>
            <ArrowClockwise size={15} />重试这次请求
          </button>
        ) : null}
        {entry.confirmation && onConfirm ? <div className="assistant-confirmation">
          <div><strong>{entry.confirmation.action === "run_governance_analysis" ? "准备分析当前治理图谱" : entry.confirmation.action === "submit_review" ? "准备写入人工结论" : "准备保存研判草稿"}</strong>
            <span>{entry.confirmation.action === "run_governance_analysis" ? "系统将生成风险账号排序、协同群组和重点关系；完成后请进入治理应用核对证据并记录结论。" : "系统将按当前上下文执行这项受控操作，并保留审计记录。"}</span>
            <small><span>确认前不会产生运行或写入记录</span>{entry.confirmation.action === "run_governance_analysis" ? <span>模型发现不会改写原始图事实</span> : null}</small>
          </div>
          <button type="button" className="primary-button primary-button--small" onClick={() => onConfirm(entry)}>
            {entry.confirmation.action === "run_governance_analysis" ? "确认开始分析" : entry.confirmation.action === "submit_review" ? "确认写入复核" : "确认保存草稿"}
          </button>
        </div> : null}
        {canOpenGovernanceReview(entry, activeGovernanceRunId) && onOpenReview ? <div className="assistant-next-actions">
          <button type="button" className="primary-button primary-button--small" onClick={() => onOpenReview(entry.governanceRunId!)}><ShieldCheck size={15} />打开复核工作台</button>
          {onLocateGraph ? <button type="button" className="secondary-button" onClick={() => onLocateGraph(entry)}><Graph size={15} />定位重点候选</button> : null}
        </div> : null}
      </div>
    </article>
  );
}
