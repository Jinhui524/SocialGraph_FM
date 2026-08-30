import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { adaptationWorkspaceSnapshot } from "../test/fixtures/governanceAdaptation";
import { targetTaskRegistration } from "../test/fixtures/governanceTargetTask";
import type { GovernanceWorkspaceSnapshot } from "../services/governanceWorkspaceStore";
import { GovernanceTaskSelector, governanceWorkspaceMountKey, resolveGovernanceTask, type GovernanceTaskEntry } from "./GovernanceTaskSelector";

afterEach(cleanup);

describe("governance task selector", () => {
  it("keeps the current session mounted across chat hydration while isolating target task state", () => {
    const session = { id: "session", kind: "session" } as GovernanceTaskEntry;
    const targetA = { id: "target-a", kind: "target" } as GovernanceTaskEntry;
    const targetB = { id: "target-b", kind: "target" } as GovernanceTaskEntry;

    expect(governanceWorkspaceMountKey(session)).toBe("session-governance");
    expect(governanceWorkspaceMountKey({ ...session, id: "hydrated-chat-session" })).toBe("session-governance");
    expect(governanceWorkspaceMountKey(targetA)).toBe("target-a");
    expect(governanceWorkspaceMountKey(targetB)).toBe("target-b");
  });

  it("switches to an independent target and restores the exact session snapshot", () => {
    const session = adaptationWorkspaceSnapshot() as unknown as GovernanceWorkspaceSnapshot;
    const target = { ...session, sessionId: targetTaskRegistration().registrationId, sourceFileName: "regional-b.sgtask.zip" };
    const entries: readonly GovernanceTaskEntry[] = [
      { id: "session", label: "当前会话治理", kind: "session", snapshot: session, graph: null },
      { id: target.sessionId, label: "Regional review task B", kind: "target", snapshot: target, graph: null },
    ];
    const onSelect = vi.fn();
    const { rerender } = render(<GovernanceTaskSelector entries={entries} activeId="session" onSelect={onSelect} />);
    expect(screen.getByRole("navigation", { name: "治理任务" })).toHaveClass("governance-task-selector");
    expect(screen.getByText("任务空间", { exact: true })).toBeVisible();
    expect(screen.getByRole("button", { name: "当前会话治理" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Regional review task B" })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: "Regional review task B" }).querySelector("svg")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Regional review task B" }));
    expect(onSelect).toHaveBeenCalledWith(target.sessionId);
    expect(resolveGovernanceTask(entries, target.sessionId)?.snapshot).toBe(target);
    rerender(<GovernanceTaskSelector entries={entries} activeId={target.sessionId} onSelect={onSelect} />);
    expect(screen.getByRole("button", { name: "Regional review task B" })).toHaveAttribute("aria-pressed", "true");
    fireEvent.click(screen.getByRole("button", { name: "当前会话治理" }));
    expect(onSelect).toHaveBeenLastCalledWith("session");
    expect(resolveGovernanceTask(entries, "session")?.snapshot).toBe(session);
    expect(entries[0].snapshot).toBe(session);
  });
});
