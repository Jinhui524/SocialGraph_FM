import { CheckCircle, CircleNotch } from "@phosphor-icons/react";

export type AssistantActivityKind = "graph_import" | "graph_analysis" | "governance";

export interface AssistantActivity {
  readonly kind: AssistantActivityKind;
  readonly state: "working" | "completed";
}

const ACTIVITY_COPY: Readonly<Record<AssistantActivityKind, {
  readonly working: string;
  readonly steps: readonly string[];
}>> = {
  graph_import: {
    working: "正在核对文件与字段",
    steps: ["读取文件结构", "核对字段与质量", "整理构图建议"],
  },
  graph_analysis: {
    working: "正在核对当前图谱",
    steps: ["理解问题", "核对图谱范围", "整理分析结论"],
  },
  governance: {
    working: "正在核对治理上下文",
    steps: ["理解问题", "核对图谱与证据", "整理复核建议"],
  },
};

export function AssistantActivityView({ activity }: { readonly activity: AssistantActivity }) {
  const copy = ACTIVITY_COPY[activity.kind];
  if (activity.state === "working") {
    return (
      <div className="assistant-activity is-working" role="status" aria-live="polite">
        <CircleNotch className="spin" size={16} />
        <span><strong>思考中…</strong>{copy.working}</span>
      </div>
    );
  }

  return (
    <details className="assistant-activity is-completed">
      <summary><CheckCircle size={16} weight="fill" />思考已完成</summary>
      <ol>
        {copy.steps.map((step) => <li key={step}>{step}</li>)}
      </ol>
    </details>
  );
}
