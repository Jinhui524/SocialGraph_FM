import type {
  AssistantActivity,
  AssistantActivityKind,
} from "../../components/AssistantActivity";
import type { AssistantGuidanceState } from "../../components/AssistantGuidance";
import type {
  AnalysisRun,
  ConversationMessage,
  IntentMeta,
  NormalizedIntent,
} from "../../types/graph";
import type {
  GovernanceAssistantDispatchIntent,
  GovernanceConfirmationTicket,
} from "../../types/governanceSkills";
import type {
  GovernanceOnlineRun,
  GovernanceOnlineStage,
} from "../../types/governanceOnline";

export type ChatEntry =
  | {
      id: string;
      role: "user";
      text: string;
      timestamp: string;
      file?: { name: string; size: number };
      files?: readonly { name: string; size: number }[];
    }
  | {
      id: string;
      role: "assistant";
      text: string;
      timestamp: string;
      state: "working" | "success" | "warning" | "error";
      intent?: NormalizedIntent;
      intentMeta?: IntentMeta;
      run?: AnalysisRun;
      demo?: boolean;
      retryText?: string;
      confirmation?: GovernanceConfirmationTicket;
      dispatchIntent?: GovernanceAssistantDispatchIntent;
      activity?: AssistantActivity;
      governanceProgress?: {
        readonly stage: GovernanceProgressStage;
        readonly progress: number;
      };
      /** Links this completed governance report to the matching review workspace. */
      governanceRunId?: string;
    };

const STALE_CONFIRMATION_NOTICE = "输入已更换，此操作计划已失效。";

export function invalidateChatConfirmations(entries: readonly ChatEntry[]): ChatEntry[] {
  return entries.map((entry) => {
    if (entry.role !== "assistant" || !entry.confirmation) return entry;
    const { confirmation: _confirmation, ...rest } = entry;
    return {
      ...rest,
      text: entry.text.includes(STALE_CONFIRMATION_NOTICE)
        ? entry.text
        : `${entry.text}\n\n> ${STALE_CONFIRMATION_NOTICE}`,
    };
  });
}

export function completeConfirmedPlanningMessage(
  entries: readonly ChatEntry[],
  planningMessageId: string,
  report?: ChatEntry,
): ChatEntry[] {
  const remaining = entries.map((entry) => entry.id === planningMessageId && entry.role === "assistant"
    ? {
        ...entry,
        text: report
          ? "当前治理图谱分析已完成。系统已生成风险账号排序、协同群组和重点关系；请查看下方报告，再进入治理应用核对证据并记录结论。"
          : "正在分析当前治理图谱。系统将依次完成输入检查、图谱准备、风险推理、研判生成与结果固化；完成后可进入治理应用核对证据。",
        state: report ? "success" as const : "working" as const,
        confirmation: undefined,
        activity: {
          kind: "governance" as const,
          state: report ? "completed" as const : "working" as const,
        },
        governanceProgress: {
          stage: report ? "completed" as const : "queued" as const,
          progress: report ? 100 : 0,
        },
      }
    : entry);
  if (!report) return remaining;
  const reportRunId = report.role === "assistant" ? report.governanceRunId : undefined;
  if (remaining.some((entry) => entry.id === report.id
    || reportRunId && entry.role === "assistant" && entry.governanceRunId === reportRunId)) {
    return remaining;
  }
  return [...remaining, report];
}

export type GovernanceProgressStage = GovernanceOnlineStage | "reporting";

export function presentGovernanceRunProgress(
  run: Pick<GovernanceOnlineRun, "stage" | "progress" | "status">,
): { readonly stage: GovernanceProgressStage; readonly progress: number } {
  if (run.status === "succeeded" || run.stage === "completed") {
    return Object.freeze({ stage: "reporting", progress: 95 });
  }
  return Object.freeze({
    stage: run.stage,
    progress: run.stage === "freezing" ? Math.min(run.progress, 95) : run.progress,
  });
}

export function updateGovernancePlanningProgress(
  entries: readonly ChatEntry[],
  planningMessageId: string,
  run: { readonly stage: GovernanceProgressStage; readonly progress: number },
): ChatEntry[] {
  const stageLabels: Readonly<Record<GovernanceProgressStage, string>> = {
    queued: "等待输入检查",
    validating: "正在检查输入与模型身份",
    preprocessing: "正在准备当前关系图谱",
    inferencing: "正在执行风险推理",
    deriving: "正在整理候选、群组与关系",
    freezing: "正在固化可复核结果",
    reporting: "模型分析已完成，正在整理研判结论",
    completed: "治理分析已完成",
  };
  return entries.map((entry) => entry.id === planningMessageId && entry.role === "assistant"
    ? {
        ...entry,
        text: `${stageLabels[run.stage]}。分析对象是当前已登记的治理图谱，完成后将生成风险账号排序、协同群组和重点关系，并交由你继续核对证据。`,
        state: run.stage === "completed" ? "success" as const : "working" as const,
        confirmation: undefined,
        activity: {
          kind: "governance" as const,
          state: run.stage === "completed" ? "completed" as const : "working" as const,
        },
        governanceProgress: { stage: run.stage, progress: run.progress },
      }
    : entry);
}

export const GOVERNANCE_ANALYSIS_STAGES = Object.freeze([
  Object.freeze({ id: "validating" as const, label: "输入检查" }),
  Object.freeze({ id: "preprocessing" as const, label: "准备图谱" }),
  Object.freeze({ id: "inferencing" as const, label: "风险推理" }),
  Object.freeze({ id: "deriving" as const, label: "生成研判" }),
  Object.freeze({ id: "reporting" as const, label: "整理结论" }),
]);

export function governanceProgressStageIndex(stage: GovernanceProgressStage): number {
  if (stage === "completed") return GOVERNANCE_ANALYSIS_STAGES.length;
  if (stage === "reporting" || stage === "freezing") return GOVERNANCE_ANALYSIS_STAGES.length - 1;
  if (stage === "queued") return 0;
  return Math.max(0, GOVERNANCE_ANALYSIS_STAGES.findIndex((item) => item.id === stage));
}

function assistantActivityKind(
  entry: Extract<ChatEntry, { role: "assistant" }>,
): AssistantActivityKind {
  if (entry.activity) return entry.activity.kind;
  if (entry.confirmation || entry.dispatchIntent) return "governance";
  return "graph_analysis";
}

export function assistantActivityForEntry(
  entry: Extract<ChatEntry, { role: "assistant" }>,
): AssistantActivity | null {
  if (entry.state === "working") {
    return entry.activity ?? { kind: assistantActivityKind(entry), state: "working" };
  }
  if (entry.state !== "success") return null;
  if (entry.governanceRunId) return { kind: "governance", state: "completed" };
  if (entry.run?.status === "succeeded" && entry.run.result && entry.run.engine !== "unavailable") {
    return { kind: assistantActivityKind(entry), state: "completed" };
  }
  return null;
}

export function assistantGuidanceStateForEntry(
  entry: Extract<ChatEntry, { role: "assistant" }>,
): AssistantGuidanceState | null {
  if (entry.state === "error") return "failed";
  if (entry.confirmation) return "awaiting_confirmation";
  if (entry.governanceProgress) {
    return entry.governanceProgress.stage === "completed" ? "completed" : "running";
  }
  if (entry.governanceRunId) return "evidence_followup";
  if (entry.activity?.kind === "graph_import" && entry.state === "success") return "upload_ready";
  if (entry.run?.status === "succeeded" && entry.run.result && entry.run.engine !== "unavailable") {
    return "completed";
  }
  return null;
}

export function canOpenGovernanceReview(
  entry: Extract<ChatEntry, { role: "assistant" }>,
  activeRunId?: string,
): boolean {
  return entry.state === "success"
    && Boolean(entry.governanceRunId)
    && entry.governanceRunId === activeRunId;
}

export function governanceRunIdForPersistence(entry: ChatEntry): string | undefined {
  return entry.role === "assistant" && entry.state === "success"
    ? entry.governanceRunId
    : undefined;
}

export function governanceRunIdFromStoredMessage(
  message: ConversationMessage,
): string | undefined {
  return message.role === "assistant" && message.status === "completed"
    ? message.governanceRunId
    : undefined;
}
