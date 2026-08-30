import {
  CheckCircle,
  CircleNotch,
  MagnifyingGlass,
  ShieldCheck,
  UploadSimple,
  WarningCircle,
} from "@phosphor-icons/react";

export type AssistantGuidanceState =
  | "upload_ready"
  | "awaiting_confirmation"
  | "running"
  | "completed"
  | "evidence_followup"
  | "failed";

interface GuidanceCopy {
  readonly title: string;
  readonly doing: string;
  readonly outcome: string;
  readonly next: string;
  readonly Icon: typeof CheckCircle;
}

const GUIDANCE_COPY: Readonly<Record<AssistantGuidanceState, GuidanceCopy>> = {
  upload_ready: {
    title: "图谱已就绪",
    doing: "已完成输入登记与兼容性检查",
    outcome: "可生成结构结论或治理候选",
    next: "描述分析目标，治理任务可输入“开始分析”",
    Icon: UploadSimple,
  },
  awaiting_confirmation: {
    title: "等待你的确认",
    doing: "已组织当前治理分析计划",
    outcome: "将生成风险候选、协同群组与重点关系",
    next: "核对分析对象后确认开始",
    Icon: ShieldCheck,
  },
  running: {
    title: "正在生成治理研判",
    doing: "依次校验输入、准备图谱并执行推理",
    outcome: "将形成可复核的候选与关系结果",
    next: "完成后进入治理应用核对证据",
    Icon: CircleNotch,
  },
  completed: {
    title: "分析结果已固化",
    doing: "模型结果与图事实已分层保存",
    outcome: "已获得风险候选、群组与重点关系",
    next: "进入治理应用选择对象并开始复核",
    Icon: CheckCircle,
  },
  evidence_followup: {
    title: "进入证据复核",
    doing: "当前结果已绑定治理工作台",
    outcome: "可核对直接事实、结构线索与缺口",
    next: "选择对象，形成或追问研判报告",
    Icon: MagnifyingGlass,
  },
  failed: {
    title: "本次处理未完成",
    doing: "当前步骤已停止，原始图谱未被修改",
    outcome: "已保留可用上下文与审计信息",
    next: "检查输入或服务状态后重试",
    Icon: WarningCircle,
  },
};

export function AssistantGuidance({ state }: { readonly state: AssistantGuidanceState }) {
  const copy = GUIDANCE_COPY[state];
  const Icon = copy.Icon;
  return (
    <section className={`assistant-guidance is-${state}`} aria-label="下一步指引">
      <header><Icon className={state === "running" ? "spin" : ""} size={17} /><strong>{copy.title}</strong></header>
      <dl>
        <div><dt>正在做什么</dt><dd>{copy.doing}</dd></div>
        <div><dt>将得到什么</dt><dd>{copy.outcome}</dd></div>
        <div><dt>下一步</dt><dd>{copy.next}</dd></div>
      </dl>
    </section>
  );
}
