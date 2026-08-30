import {
  CircleNotch,
  Link,
  ShieldCheck,
  Sparkle,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

import { governanceModalityLabel } from "../services/governancePresentation";
import type { GovernanceSkillsClientLike, GovernanceSkillsContext, GovernanceTargetKind } from "../types/governanceSkills";
import type { GovernanceDerivation, GovernanceOnlineEvidence, GovernanceOnlineFinding } from "../types/governanceOnline";
import { SafeMarkdown } from "./SafeMarkdown";

export type EvidenceState =
  | { readonly state: "idle" | "loading" }
  | { readonly state: "ready"; readonly value: GovernanceOnlineEvidence }
  | { readonly state: "error"; readonly message: string };

export interface EvidenceDossierTarget {
  readonly kind: GovernanceTargetKind;
  readonly id: string;
  readonly nodeIds: readonly string[];
}

interface EvidenceDossierProps {
  readonly open: boolean;
  readonly target: EvidenceDossierTarget | null;
  readonly title: string;
  readonly finding: GovernanceOnlineFinding | null;
  readonly derivation: GovernanceDerivation | null;
  readonly derivationRank: number | null;
  readonly evidence: EvidenceState;
  readonly skillsContext: GovernanceSkillsContext | null;
  readonly summaryClient?: GovernanceSkillsClientLike;
  readonly reviewContent: ReactNode;
  readonly candidateLabel: (nodeId: string) => string;
  readonly reviewRank: (finding: GovernanceOnlineFinding) => number;
  readonly onSelectNeighbor: (nodeId: string) => void;
  readonly onClose: () => void;
}

type DossierTab = "summary" | "facts" | "review";
type SummaryState =
  | { readonly state: "idle" }
  | { readonly state: "loading"; readonly key: string }
  | { readonly state: "ready"; readonly key: string; readonly text: string }
  | { readonly state: "error"; readonly key: string; readonly message: string };

const SUMMARY_CACHE = new Map<string, string>();
export const EVIDENCE_SUMMARY_TIMEOUT_MS = 30_000;
const FOCUSABLE = "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";

function riskLabel(band: GovernanceOnlineFinding["riskBand"]): string {
  return band === "high" ? "高风险候选" : band === "review" ? "建议复核" : "低风险参照";
}

function compactSummary(text: string): string {
  const normalized = text.trim();
  return normalized.length > 600 ? `${normalized.slice(0, 599).trimEnd()}…` : normalized;
}

function deterministicNodeSummary(
  title: string,
  finding: GovernanceOnlineFinding,
  evidence: GovernanceOnlineEvidence | null,
): ReactNode {
  const modalities = evidence
    ? Object.entries(evidence.structuralSignals.relationNeighborCounts)
      .filter(([, count]) => count > 0)
      .map(([modality, count]) => `${governanceModalityLabel(modality as keyof typeof evidence.structuralSignals.relationNeighborCounts)} ${count}`)
    : Object.entries(finding.modalityEvidence)
      .filter(([, count]) => Boolean(count))
      .map(([modality, count]) => `${governanceModalityLabel(modality as keyof typeof finding.modalityEvidence)} ${count}`);
  return <div className="governance-dossier-narrative">
    <p><strong>关注原因</strong>{title}处于{riskLabel(finding.riskBand)}，当前风险排序为第 {finding.rank} 位。这一结果用于安排人工复核顺序，不代表已经确认账号性质。</p>
    <p><strong>事实支撑</strong>{modalities.length ? `已登记的一跳关系包括 ${modalities.join("、")}。` : "当前结果未提供可展示的一跳关系记录。"}{evidence ? `融合邻居共 ${evidence.structuralSignals.fusedDegree} 个。` : "打开档案后将读取绑定的一跳关系。"}</p>
    <p><strong>关联线索</strong>{evidence ? `两跳范围包含 ${evidence.structuralSignals.twoHopNodeCount} 个账号；该统计仅用于发现需要继续核对的邻域。` : "邻域结构将在证据读取完成后展示。"}</p>
    <p><strong>核验建议</strong>优先核对关系两端账号、关系模态与权重，再补充原帖内容、发布时间和采集来源。模型排序和结构统计不能单独作为处置依据。</p>
  </div>;
}

function deterministicDerivationSummary(title: string, derivation: GovernanceDerivation): ReactNode {
  if (derivation.kind === "group") {
    return <div className="governance-dossier-narrative">
      <p><strong>关注原因</strong>{title}包含 {derivation.memberCount ?? derivation.nodeIds.length} 个账号，并在当前派生结果中具有较高复核优先级。</p>
      <p><strong>事实支撑</strong>{derivation.modalities.length ? `成员之间已登记 ${derivation.modalities.map(governanceModalityLabel).join("、")} 关系。` : "当前结果未提供可展示的内部关系模态。"}</p>
      <p><strong>研判边界</strong>群组来自结构统计，不能据此推断成员具有共同意图。应逐条回到成员和内部事实关系核验。</p>
    </div>;
  }
  return <div className="governance-dossier-narrative">
    <p><strong>关系属性</strong>{derivation.factual ? "该对象是图谱中已登记的事实关系。" : "该对象是结构派生的潜在线索，并非已登记事实边。"}</p>
    <p><strong>事实支撑</strong>当前可核对两端账号及{derivation.modalities.length ? `${derivation.modalities.map(governanceModalityLabel).join("、")}关系模态` : "现有关系属性"}。</p>
    <p><strong>核验建议</strong>回到原始内容、发布时间与采集来源确认关系语境；当前优先级只用于安排复核顺序。</p>
  </div>;
}

export function EvidenceDossier({
  open,
  target,
  title,
  finding,
  derivation,
  derivationRank,
  evidence,
  skillsContext,
  summaryClient,
  reviewContent,
  candidateLabel,
  reviewRank,
  onSelectNeighbor,
  onClose,
}: EvidenceDossierProps) {
  const [tab, setTab] = useState<DossierTab>("summary");
  const [summary, setSummary] = useState<SummaryState>({ state: "idle" });
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const requestTimeoutRef = useRef<number | null>(null);
  const titleId = useId();
  const readyEvidence = evidence.state === "ready" ? evidence.value : null;
  const targetIdentity = target ? `${target.kind}:${target.id}` : null;
  const summaryKey = target?.kind === "node" && readyEvidence
    ? `${readyEvidence.runId}:${target.id}:${readyEvidence.evidenceHash}`
    : null;

  useEffect(() => {
    if (requestTimeoutRef.current !== null) window.clearTimeout(requestTimeoutRef.current);
    requestTimeoutRef.current = null;
    requestRef.current?.abort();
    requestRef.current = null;
    setTab("summary");
    setSummary({ state: "idle" });
  }, [targetIdentity]);

  useEffect(() => {
    if (requestTimeoutRef.current !== null) window.clearTimeout(requestTimeoutRef.current);
    requestTimeoutRef.current = null;
    requestRef.current?.abort();
    requestRef.current = null;
    if (summaryKey && SUMMARY_CACHE.has(summaryKey)) {
      setSummary({ state: "ready", key: summaryKey, text: SUMMARY_CACHE.get(summaryKey)! });
    } else {
      setSummary({ state: "idle" });
    }
  }, [summaryKey]);

  useEffect(() => {
    if (!open) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = window.requestAnimationFrame(() => closeRef.current?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      if (requestTimeoutRef.current !== null) window.clearTimeout(requestTimeoutRef.current);
      requestTimeoutRef.current = null;
      requestRef.current?.abort();
      requestRef.current = null;
      setSummary((current) => current.state === "loading" ? { state: "idle" } : current);
      returnFocusRef.current?.focus();
    };
  }, [open]);

  const requestSummary = () => {
    if (!summaryClient || !skillsContext || !finding || !readyEvidence || !summaryKey || summary.state === "loading") return;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const prompt = [
      "请为当前账号生成一份300至600个中文字符的证据研判摘要。",
      "固定使用“关注原因、事实支撑、关联线索、核验建议”四个小标题。",
      "只可使用本次受控只读分析返回的图谱检查、证据子图、事实关系与潜在线索。",
      "明确模型风险排序不是恶意判定，也不要声称图基础模型具有因果可解释性。",
      "发布时间、原帖内容和采集来源未包含在当前证据中，必须列为待补充证据。",
    ].join("\n");
    setSummary({ state: "loading", key: summaryKey });
    summaryClient.dispatchAssistant(skillsContext, prompt, {
      intent: "answer",
      answerMode: "evidence_requirements",
    }, controller.signal).then((response) => {
      if (controller.signal.aborted || requestRef.current !== controller) return;
      if (requestTimeoutRef.current !== null) window.clearTimeout(requestTimeoutRef.current);
      requestTimeoutRef.current = null;
      requestRef.current = null;
      const text = compactSummary(response.answer);
      if (!text) throw new Error("EMPTY_EVIDENCE_SUMMARY");
      SUMMARY_CACHE.set(summaryKey, text);
      setSummary({ state: "ready", key: summaryKey, text });
    }).catch(() => {
      if (requestRef.current === controller) {
        if (requestTimeoutRef.current !== null) window.clearTimeout(requestTimeoutRef.current);
        requestTimeoutRef.current = null;
        requestRef.current = null;
      }
      if (!controller.signal.aborted) {
        setSummary({ state: "error", key: summaryKey, message: "智能摘要暂未生成，结构化证据仍可继续核对。" });
      }
    });
    requestTimeoutRef.current = window.setTimeout(() => {
      if (requestRef.current !== controller) return;
      requestRef.current = null;
      requestTimeoutRef.current = null;
      controller.abort();
      setSummary({ state: "error", key: summaryKey, message: "智能摘要生成超时，结构化证据仍可继续核对。" });
    }, EVIDENCE_SUMMARY_TIMEOUT_MS);
  };

  const trapFocus = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = [...(dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [])];
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const factualRows = useMemo(() => readyEvidence?.neighbors.flatMap((neighbor) => neighbor.relations.map((relation, index) => ({
    key: `${neighbor.nodeId}:${relation.modality}:${index}`,
    neighborId: neighbor.nodeId,
    modality: relation.modality,
    weight: relation.rawWeight,
    riskBand: neighbor.riskBand,
  }))) ?? [], [readyEvidence]);

  if (!open || !target) return null;
  const deterministic = finding
    ? deterministicNodeSummary(title, finding, readyEvidence)
    : derivation
      ? deterministicDerivationSummary(title, derivation)
      : <p>当前对象的结构化证据尚未就绪。</p>;

  return createPortal(<div className="governance-dossier-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div ref={dialogRef} className="governance-dossier-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={trapFocus}>
      <header className="governance-dossier-dialog__header">
        <div><span>证据档案</span><h2 id={titleId}>{title}</h2><small>{target.kind === "node" ? "账号风险排序与关系证据" : target.kind === "group" ? "风险群组与内部关系" : "关系属性与核验线索"}</small></div>
        <button ref={closeRef} type="button" aria-label="关闭证据档案" title="关闭" onClick={onClose}><X /></button>
      </header>
      <nav className="governance-dossier-tabs" role="tablist" aria-label="证据档案内容">
        {([ ["summary", "证据摘要"], ["facts", "关系事实"], ["review", "人工复核"] ] as const).map(([id, label]) => <button key={id} type="button" role="tab" aria-selected={tab === id} onClick={() => setTab(id)}>{label}</button>)}
      </nav>
      <div className="governance-dossier-dialog__body">
        {tab === "summary" ? <section role="tabpanel" aria-label="证据摘要" className="governance-dossier-summary">
          <div className="governance-dossier-deterministic"><header><ShieldCheck /><strong>结构化研判摘要</strong><span>基于绑定结果</span></header>{deterministic}</div>
          {target.kind === "node" ? <section className="governance-dossier-ai">
            <header><Sparkle /><div><strong>智能证据研判</strong><span>受事实约束</span></div></header>
            {summary.state === "ready" && summary.key === summaryKey ? <div className="governance-dossier-ai__answer"><SafeMarkdown text={summary.text} /></div> : null}
            {summary.state === "loading" ? <p role="status"><CircleNotch className="spin" />正在组织证据摘要…</p> : null}
            {summary.state === "error" ? <p className="is-error" role="status"><WarningCircle />{summary.message}</p> : null}
            {summary.state !== "ready" ? <button type="button" disabled={!summaryClient || !skillsContext || !readyEvidence || summary.state === "loading"} onClick={requestSummary}><Sparkle />{summary.state === "error" ? "重新生成证据研判摘要" : "生成证据研判摘要"}</button> : null}
            {!summaryClient ? <small>当前未连接智能整理服务，结构化证据可正常使用。</small> : !readyEvidence ? <small>关系事实读取完成后即可生成摘要。</small> : null}
          </section> : <p className="governance-dossier-boundary">群组和关系使用绑定派生结果，不生成目标级大模型解释，避免将全局分析误写为对象证据。</p>}
        </section> : null}

        {tab === "facts" ? <section role="tabpanel" aria-label="关系事实" className="governance-dossier-facts">
          {target.kind === "node" ? <>
            {evidence.state === "loading" ? <p role="status"><CircleNotch className="spin" />正在读取一跳关系…</p> : null}
            {evidence.state === "error" ? <p className="is-error"><WarningCircle />{evidence.message}</p> : null}
            {finding ? <dl className="governance-dossier-metrics"><div><dt>风险档位</dt><dd>{riskLabel(finding.riskBand)}</dd></div><div><dt>复核顺序</dt><dd>#{reviewRank(finding)}</dd></div><div><dt>融合度数</dt><dd>{readyEvidence?.structuralSignals.fusedDegree ?? "--"}</dd></div><div><dt>两跳节点</dt><dd>{readyEvidence?.structuralSignals.twoHopNodeCount ?? "--"}</dd></div></dl> : null}
            {factualRows.length ? <div className="governance-dossier-table-wrap"><table><thead><tr><th>关联账号</th><th>关系模态</th><th>权重</th><th>邻居状态</th><th><span className="sr-only">操作</span></th></tr></thead><tbody>{factualRows.map((row) => <tr key={row.key}><th scope="row">{candidateLabel(row.neighborId)}</th><td>{governanceModalityLabel(row.modality)}</td><td>{row.weight.toLocaleString()}</td><td>{riskLabel(row.riskBand)}</td><td><button type="button" onClick={() => onSelectNeighbor(row.neighborId)}>选中</button></td></tr>)}</tbody></table></div> : evidence.state === "ready" ? <p>当前对象没有可展示的一跳事实关系。</p> : null}
          </> : derivation ? <>
            <dl className="governance-dossier-metrics"><div><dt>对象属性</dt><dd>{derivation.kind === "group" ? "结构派生群组" : derivation.factual ? "已登记事实关系" : "潜在线索"}</dd></div><div><dt>涉及账号</dt><dd>{derivation.nodeIds.length}</dd></div><div><dt>关系类型</dt><dd>{derivation.modalities.length || "--"}</dd></div><div><dt>复核顺序</dt><dd>{derivationRank ? `#${derivationRank}` : "待编排"}</dd></div></dl>
            {derivation.kind === "group" ? <div className="governance-dossier-members">{derivation.nodeIds.map((nodeId) => <button type="button" key={nodeId} onClick={() => onSelectNeighbor(nodeId)}>{candidateLabel(nodeId)}</button>)}</div> : <div className="governance-dossier-relation"><Link /><strong>{candidateLabel(derivation.source ?? derivation.nodeIds[0])}</strong><span>—</span><strong>{candidateLabel(derivation.target ?? derivation.nodeIds[1])}</strong></div>}
            <p>{derivation.modalities.length ? `关系模态：${derivation.modalities.map(governanceModalityLabel).join("、")}` : "当前派生结果未提供直接关系模态。"}</p>
          </> : null}
          <p className="governance-dossier-gap"><WarningCircle />当前证据只包含关系两端账号、关系模态、可用权重与绑定哈希。发布时间、原帖内容及采集来源需要在人工复核中补充。</p>
        </section> : null}

        {tab === "review" ? <section role="tabpanel" aria-label="人工复核" className="governance-dossier-review">{reviewContent}</section> : null}
      </div>
    </div>
  </div>, document.body);
}
