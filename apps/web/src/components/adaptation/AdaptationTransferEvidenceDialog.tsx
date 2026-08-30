import { ChartDonut, ShieldCheck, X } from "@phosphor-icons/react";
import { useEffect, useId, useRef, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { createPortal } from "react-dom";

import type { AdaptationLane } from "./AdaptationWorkspace";
import type { AdaptationTransferEvidence } from "./AdaptationTransferEvidence";

interface AdaptationTransferEvidenceDialogProps {
  readonly open: boolean;
  readonly lane: AdaptationLane;
  readonly evidence: AdaptationTransferEvidence | null;
  readonly nodeCount: number;
  readonly relationCount: number;
  readonly onClose: () => void;
}

const FOCUSABLE = "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])";

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function responseLabel(value: number): string {
  return value.toFixed(2);
}

export function AdaptationTransferEvidenceDialog({
  open,
  lane,
  evidence,
  nodeCount,
  relationCount,
  onClose,
}: AdaptationTransferEvidenceDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const titleId = useId();
  const descriptionId = useId();

  useEffect(() => {
    if (!open) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = window.requestAnimationFrame(() => closeRef.current?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      returnFocusRef.current?.focus();
    };
  }, [open]);

  const trapFocus = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      onClose();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = [...(dialogRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE) ?? [])];
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  if (!open || !evidence) return null;
  const selected = evidence.selectedObject;
  const calibration = lane === "few_shot" ? evidence.calibration : null;

  return createPortal(<div className="adaptation-transfer-dialog-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <div
      ref={dialogRef}
      className="adaptation-transfer-dialog"
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-describedby={descriptionId}
      onKeyDown={trapFocus}
    >
      <header className="adaptation-transfer-dialog__header">
        <div className="adaptation-transfer-dialog__mark"><ChartDonut size={21} /></div>
        <div><span>迁移依据</span><h2 id={titleId}>{lane === "zero_shot" ? "零样本源域路由凭证" : "少样本源域路由与校正"}</h2><p id={descriptionId}>展示冻结模型在当前目标网络上的匿名路由和表征响应。</p></div>
        <button ref={closeRef} type="button" aria-label="关闭迁移依据" title="关闭" onClick={onClose}><X size={18} /></button>
      </header>

      <div className="adaptation-transfer-dialog__body">
        <section className="adaptation-transfer-routing" aria-labelledby={`${titleId}-routing`}>
          <header><div><span>01 · 专家路由</span><h3 id={`${titleId}-routing`}>目标域使用了哪些既有知识</h3></div><strong>{evidence.activeSourceCount} 个活跃源域</strong></header>
          <div className="adaptation-transfer-shared"><ShieldCheck size={18} /><div><strong>共享表征</strong><span>固定参与全部对象，不计入路由权重合计</span></div><b>100%</b></div>
          <div className="adaptation-route-stack" role="img" aria-label={`六个匿名源域专家与保守未知域的路由权重合计 100%，保守未知域 ${percent(evidence.nullRoutingMass)}`}>
            {evidence.experts.map((expert) => <i key={expert.id} data-expert={expert.id} style={{ width: percent(expert.routingMass) }} title={`${expert.label} ${percent(expert.routingMass)}`} />)}
          </div>
          <div className="adaptation-route-table" role="table" aria-label="匿名专家路由明细">
            <div role="row" className="adaptation-route-table__head"><span role="columnheader">知识来源</span><span role="columnheader">全网平均权重</span><span role="columnheader">覆盖率</span><span role="columnheader">训练对象</span></div>
            {evidence.experts.map((expert) => <div role="row" key={expert.id} className="adaptation-route-table__row" data-expert={expert.id}>
              <span role="cell"><i />{expert.label}</span>
              <strong role="cell">{percent(expert.averageWeight)}</strong>
              <span role="cell">{percent(expert.coverage)}</span>
              <span role="cell">{expert.trainingNodeCount?.toLocaleString() ?? "--"}</span>
            </div>)}
          </div>
        </section>

        <section className="adaptation-transfer-context" aria-labelledby={`${titleId}-context`}>
          <header><span>02 · 表征响应</span><h3 id={`${titleId}-context`}>当前网络如何被读取</h3></header>
          <div className="adaptation-response-pair">
            <div><span>匿名内容表征</span><strong>{responseLabel(evidence.textResponseMean)}</strong><i><b style={{ width: percent(evidence.textResponseMean) }} /></i></div>
            <div><span>网络结构表征</span><strong>{responseLabel(evidence.structureResponseMean)}</strong><i><b style={{ width: percent(evidence.structureResponseMean) }} /></i></div>
          </div>
          <dl className="adaptation-transfer-metrics">
            <div><dt>目标对象</dt><dd>{nodeCount.toLocaleString()}</dd></div>
            <div><dt>事实关系</dt><dd>{relationCount.toLocaleString()}</dd></div>
            <div><dt>结构可用</dt><dd>{evidence.structureAvailableCount.toLocaleString()}</dd></div>
            <div><dt>训练对象</dt><dd>{evidence.trainingNodeCount.toLocaleString()}</dd></div>
          </dl>
          <div className="adaptation-transfer-modalities"><span>当前关系类型</span><div>{evidence.targetModalities.map((modality) => <b key={modality}>{modality}</b>)}</div></div>

          {selected ? <section className="adaptation-transfer-object" aria-label="当前选中账号的迁移依据">
            <header><div><span>当前账号</span><h4>{selected.label}</h4></div><b>风险排序 #{selected.rank}</b></header>
            <div className="adaptation-transfer-object__routes">{selected.routes.map((route) => <div key={route.label}><span>{route.label}</span><strong>{percent(route.weight)}</strong></div>)}</div>
            <dl><div><dt>内容响应</dt><dd>{responseLabel(selected.textResponse)}</dd></div><div><dt>结构响应</dt><dd>{responseLabel(selected.structureResponse)}</dd></div><div><dt>关系证据记录</dt><dd>{selected.relationEvidenceCount}</dd></div></dl>
          </section> : <p className="adaptation-transfer-object-empty">从风险排序中选择一个账号，可查看该对象实际使用的两个专家及表征响应。</p>}

          {calibration ? <section className="adaptation-transfer-calibration" aria-labelledby={`${titleId}-calibration`}>
            <header><div><span>03 · 目标域校正</span><h3 id={`${titleId}-calibration`}>少量已核对对象调整复核顺序</h3></div><strong>λ {calibration.selectedLambda.toFixed(2)}</strong></header>
            <dl><div><dt>正向 / 负向</dt><dd>{calibration.positiveCount} / {calibration.negativeCount}</dd></div><div><dt>顺序上升</dt><dd>{calibration.raisedCount}</dd></div><div><dt>顺序下降</dt><dd>{calibration.loweredCount}</dd></div><div><dt>顺序不变</dt><dd>{calibration.unchangedCount}</dd></div><div><dt>最大变化</dt><dd>{calibration.maxRankChange} 位</dd></div></dl>
            <p>校正只改变人工复核顺序，专家路由、基础分数和模型权重保持不变。</p>
          </section> : null}
        </section>
      </div>
      <footer><ShieldCheck size={16} /><p>路由权重表示域残差适配器的门控组合比例，不是风险概率、因果贡献或人工结论。</p></footer>
    </div>
  </div>, document.body);
}
