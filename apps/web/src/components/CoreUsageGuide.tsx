import { ChatTeardropText, FileArrowUp, ShieldCheck } from "@phosphor-icons/react";

const STEPS = [
  {
    title: "上传关系数据",
    detail: "普通图可在对话研究中探索结构；SocialGraph-FM Governance 推理包可继续进行风险分析。",
    Icon: FileArrowUp,
  },
  {
    title: "用自然语言研究",
    detail: "描述希望了解的节点、群组或关系，系统会结合图谱与分析能力给出精简结论。",
    Icon: ChatTeardropText,
  },
  {
    title: "进入治理应用复核",
    detail: "查看候选与证据，记录人工判断，并导出可核对的研判报告。",
    Icon: ShieldCheck,
  },
] as const;

export function CoreUsageGuide() {
  return <div className="core-usage-guide">
    <ol>
      {STEPS.map(({ title, detail, Icon }, index) => <li key={title}>
        <span className="core-usage-guide__index">{String(index + 1).padStart(2, "0")}</span>
        <Icon size={20} weight="duotone" />
        <div><strong>{title}</strong><p>{detail}</p></div>
      </li>)}
    </ol>
    <p className="core-usage-guide__boundary">模型结果用于辅助研判，不替代人工结论。</p>
  </div>;
}
