import { ShieldCheck, Target } from "@phosphor-icons/react";

import type { GovernanceWorkspaceSnapshot } from "../services/governanceWorkspaceStore";
import type { GovernanceAdaptationState } from "../services/governanceAdaptation";
import type { GraphVersion } from "../types/graph";

export interface GovernanceTaskEntry {
  readonly id: string;
  readonly label: string;
  readonly kind: "session" | "target";
  readonly snapshot: GovernanceWorkspaceSnapshot | null;
  readonly graph: GraphVersion | null;
  readonly adaptation?: GovernanceAdaptationState;
  readonly validationToken?: number;
}

export function resolveGovernanceTask(entries: readonly GovernanceTaskEntry[], activeId: string): GovernanceTaskEntry | null {
  return entries.find((entry) => entry.id === activeId) ?? entries.find((entry) => entry.kind === "session") ?? null;
}

export function governanceWorkspaceMountKey(entry: Pick<GovernanceTaskEntry, "id" | "kind"> | null | undefined): string {
  return entry?.kind === "target" ? entry.id : "session-governance";
}

export function GovernanceTaskSelector({ entries, activeId, onSelect }: {
  readonly entries: readonly GovernanceTaskEntry[];
  readonly activeId: string;
  readonly onSelect: (id: string) => void;
}) {
  if (entries.length < 2) return null;
  return <nav className="governance-task-selector" aria-label="治理任务">
    <span className="governance-task-selector__label">任务空间</span>
    <div className="governance-task-selector__track" role="group" aria-label="切换治理任务">
      {entries.map((entry) => {
        const Icon = entry.kind === "session" ? ShieldCheck : Target;
        return <button
          className="governance-task-selector__task"
          key={entry.id}
          type="button"
          aria-label={entry.label}
          aria-pressed={entry.id === activeId}
          title={entry.label}
          onClick={() => onSelect(entry.id)}
        >
          <span className="governance-task-selector__icon" aria-hidden="true"><Icon size={17} weight={entry.id === activeId ? "fill" : "regular"} /></span>
          <span className="governance-task-selector__copy"><strong>{entry.label}</strong><small aria-hidden="true">{entry.kind === "session" ? "当前会话" : "目标域任务"}</small></span>
          <span className="governance-task-selector__state" aria-hidden="true" />
        </button>;
      })}
    </div>
  </nav>;
}
