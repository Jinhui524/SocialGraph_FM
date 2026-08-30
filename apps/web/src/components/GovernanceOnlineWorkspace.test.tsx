import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";

import {
  artifactCompatibility,
  casePayload,
  derivationPage,
  evidencePayload,
  findingPage,
  onlineArtifact,
  onlineCapabilities,
  onlineHealth,
  onlinePreview,
  onlineResult,
  onlineRun,
  onlineRunPreview,
  GOVERNANCE_OTHER_RUN_ID,
} from "../test/fixtures/governanceOnline";
import { GOVERNANCE_WORKSPACE_SCHEMA, type GovernanceWorkspaceSnapshot } from "../services/governanceWorkspaceStore";
import type { GraphNode } from "../types/graph";
import type { GovernanceCase, GovernanceDerivation, GovernanceOnlineClientLike, GovernanceOnlineRun } from "../types/governanceOnline";
import { GovernanceOnlineWorkspace, type GovernanceGraphPresentation } from "./GovernanceOnlineWorkspace";

afterEach(cleanup);

function client(overrides: Partial<GovernanceOnlineClientLike> = {}): GovernanceOnlineClientLike {
  const base = {
    health: vi.fn(async () => onlineHealth()),
    capabilities: vi.fn(async () => onlineCapabilities()),
    russiaSample: vi.fn(async () => onlineArtifact()),
    inspectArtifact: vi.fn(async () => artifactCompatibility()),
    uploadArtifact: vi.fn(async () => onlineArtifact()),
    artifact: vi.fn(async () => onlineArtifact()),
    preview: vi.fn(async () => onlinePreview()),
    runPreview: vi.fn(async () => onlineRunPreview()),
    listRuns: vi.fn(async () => [onlineRun()]),
    createRun: vi.fn(async () => onlineRun()),
    run: vi.fn(async () => onlineRun()),
    cancelRun: vi.fn(async () => ({ ...onlineRun(), status: "cancelled" as const })),
    retryRun: vi.fn(async () => onlineRun()),
    compareRuns: vi.fn(async () => { throw new Error("unused"); }),
    result: vi.fn(async () => onlineResult()),
    findings: vi.fn(async () => findingPage()),
    evidence: vi.fn(async () => evidencePayload()),
    derivations: vi.fn(async (_runId: string, kind: GovernanceDerivation["kind"]) => derivationPage(kind).items),
    listCases: vi.fn(async () => []),
    createCase: vi.fn(async () => casePayload()),
    updateCase: vi.fn(async () => casePayload()),
    addCaseItem: vi.fn(async () => casePayload()),
    review: vi.fn(async () => casePayload()),
    case: vi.fn(async () => casePayload()),
    report: vi.fn(async () => new Blob(["report"])),
  };
  return { ...base, ...overrides } as unknown as GovernanceOnlineClientLike;
}

function snapshot(): GovernanceWorkspaceSnapshot {
  return Object.freeze({
    schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
    sessionId: "session-governance",
    sourceFileName: "source.zip",
    artifact: onlineArtifact(),
    preview: onlineRunPreview(),
    run: onlineRun(),
    result: onlineResult(),
    updatedAt: "2026-08-22T08:00:00Z",
  } as GovernanceWorkspaceSnapshot);
}

function workspace(
  api: GovernanceOnlineClientLike,
  onGraphPresentationChange = vi.fn(),
  sharedSnapshot: GovernanceWorkspaceSnapshot | null = snapshot(),
  ragOpen = false,
) {
  return <GovernanceOnlineWorkspace
    client={api}
    onGraphPresentationChange={onGraphPresentationChange}
    ragOpen={ragOpen}
    onRagOpenChange={vi.fn()}
    assistantPanel={<div>助手报告工作面</div>}
    sessionId="session-governance"
    sharedSnapshot={sharedSnapshot}
  />;
}

function latestPresentation(spy: ReturnType<typeof vi.fn>): GovernanceGraphPresentation | null {
  return [...spy.mock.calls].reverse().map(([value]) => value).find((value) => value !== null) ?? null;
}

async function waitForHydration() {
  await screen.findByText("匿名账号 1");
  await screen.findByText("匿名账号 2");
}

describe("product governance workspace", () => {
  it("collapses navigation by pane width before the compact center can overflow", () => {
    const governanceWorkspaceCss = readFileSync("src/governance-online.css", "utf8");
    expect(governanceWorkspaceCss).toMatch(/@container \(max-width: 780px\)[\s\S]*grid-template-columns:\s*repeat\(4, minmax\(0, 1fr\)\) 40px/u);
    expect(governanceWorkspaceCss).toMatch(/@container \(max-width: 600px\)[\s\S]*\.governance-mode-nav > button > svg/u);
  });

  it("consumes only the shared conversation snapshot and has no intake or overview UI", async () => {
    render(workspace(client(), vi.fn(), null));

    expect(screen.getByRole("navigation", { name: "治理工作模式" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "风险节点" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "群组与关系" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "研判单" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "研判助手" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "概览" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /上传|选择 ZIP|重新分析/u })).not.toBeInTheDocument();
    expect(screen.queryByText(/推理包|分析引擎在线|风险概览|群组概览/u)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "当前会话暂无治理结果" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "返回对话研究" })).toBeInTheDocument();
    expect(document.querySelector('input[type="file"]')).toBeNull();
  });

  it("opens on high and review risk nodes while keeping low-risk nodes as graph context", async () => {
    const onGraphPresentationChange = vi.fn();
    render(workspace(client(), onGraphPresentationChange));
    await waitForHydration();

    expect(screen.getByText("匿名账号 1")).toBeInTheDocument();
    expect(screen.getByText("匿名账号 2")).toBeInTheDocument();
    expect(screen.queryByText("匿名账号 3")).not.toBeInTheDocument();
    expect(screen.getByText("高风险候选")).toBeInTheDocument();
    expect(screen.getByText("建议复核")).toBeInTheDocument();
    expect(latestPresentation(onGraphPresentationChange)?.graph?.nodes.map((node) => node.id)).toContain("n3");
  });

  it("separates selection, graph highlighting, camera, and the evidence dialog", async () => {
    const api = client();
    const onGraphPresentationChange = vi.fn();
    render(workspace(api, onGraphPresentationChange));
    await waitForHydration();
    const row = screen.getByText("匿名账号 1").closest("article");
    expect(row).not.toBeNull();

    fireEvent.click(within(row!).getAllByRole("button")[0]);
    await waitFor(() => expect(latestPresentation(onGraphPresentationChange)?.focus).toMatchObject({ kind: "node", targetId: "n1", nodeIds: ["n1"] }));
    expect("cameraFocusCommand" in latestPresentation(onGraphPresentationChange)!).toBe(false);
    expect(latestPresentation(onGraphPresentationChange)?.projectionSpec.preset).toBe("overview");
    expect(api.evidence).not.toHaveBeenCalled();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.getByText(/视角位置保持不变/u)).toBeInTheDocument();

    fireEvent.click(within(row!).getByRole("button", { name: "查看 匿名账号 1 的证据" }));
    const dialog = await screen.findByRole("dialog", { name: "匿名账号 1" });
    await waitFor(() => expect(api.evidence).toHaveBeenCalledWith(onlineRun().runId, "n1", expect.any(AbortSignal)));
    expect(within(dialog).getByRole("tab", { name: "证据摘要" })).toBeInTheDocument();
    expect(within(dialog).getByRole("tab", { name: "关系事实" })).toBeInTheDocument();
    expect(within(dialog).getByRole("tab", { name: "人工复核" })).toBeInTheDocument();

    fireEvent.click(within(dialog).getByRole("button", { name: "关闭证据档案" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(row).toHaveClass("is-selected");
    expect(latestPresentation(onGraphPresentationChange)?.skillsContext?.selectedTarget?.targetId).toBe("n1");
  });

  it("gives canvas node selection the same highlight-only behavior and Escape clears it", async () => {
    const api = client();
    const onGraphPresentationChange = vi.fn();
    render(workspace(api, onGraphPresentationChange));
    await waitForHydration();

    act(() => latestPresentation(onGraphPresentationChange)?.onSelectNode({ id: "n2", label: "匿名账号 2", type: "账号", attributes: {} } as GraphNode));
    await waitFor(() => expect(latestPresentation(onGraphPresentationChange)?.focus?.targetId).toBe("n2"));
    expect("cameraFocusCommand" in latestPresentation(onGraphPresentationChange)!).toBe(false);
    expect(api.evidence).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "Escape" });
    await waitFor(() => expect(latestPresentation(onGraphPresentationChange)?.focus).toBeUndefined());
    expect(latestPresentation(onGraphPresentationChange)?.selectedNodeId).toBeNull();
  });

  it("prepares review in one action and keeps all three human decisions visible", async () => {
    const draft = casePayload() as GovernanceCase;
    const item = {
      itemId: `item-${"5".repeat(32)}`,
      targetType: "node" as const,
      targetId: "n1",
      note: "由治理工作台加入研判范围。",
      createdAt: "2026-08-22T08:00:00Z",
      itemHash: "5".repeat(64),
    };
    const draftWithItem = { ...draft, items: [item] } as GovernanceCase;
    const active = { ...draftWithItem, state: "active" as const } as GovernanceCase;
    const confirmed = {
      ...active,
      reviewEvents: [{
        eventId: `event-${"6".repeat(32)}`,
        targetType: "node" as const,
        targetId: "n1",
        decision: "confirmed" as const,
        reason: "已核对一跳关系与关系模态。",
        actor: "local-analyst",
        sequence: 1,
        createdAt: "2026-08-22T08:05:00Z",
        previousEventHash: null,
        eventHash: "6".repeat(64),
      }],
      currentDecisions: { "node:n1": "confirmed" as const },
      caseHash: "9".repeat(64),
    } as GovernanceCase;
    const api = client({
      createCase: vi.fn(async () => draft),
      addCaseItem: vi.fn(async () => draftWithItem),
      updateCase: vi.fn(async () => active),
      review: vi.fn(async () => confirmed),
    });
    const onGraphPresentationChange = vi.fn();
    render(workspace(api, onGraphPresentationChange));
    await waitForHydration();
    const row = screen.getByText("匿名账号 1").closest("article");
    expect(row).not.toBeNull();
    fireEvent.click(within(row!).getByRole("button", { name: "查看 匿名账号 1 的证据" }));
    const dialog = await screen.findByRole("dialog", { name: "匿名账号 1" });
    await waitFor(() => expect(api.evidence).toHaveBeenCalledWith(onlineRun().runId, "n1", expect.any(AbortSignal)));
    await within(dialog).findByText(/融合邻居共 1 个/u);
    fireEvent.click(within(dialog).getByRole("tab", { name: "人工复核" }));
    await within(dialog).findByRole("tabpanel", { name: "人工复核" });

    const confirm = within(dialog).getByRole("button", { name: "确认" });
    expect(confirm).toBeVisible();
    expect(within(dialog).getByRole("button", { name: "驳回" })).toBeVisible();
    expect(within(dialog).getByRole("button", { name: "待定" })).toBeVisible();
    expect(confirm).toBeDisabled();
    expect(within(dialog).getByText("先将当前对象加入研判单并开始复核。")).toBeInTheDocument();

    fireEvent.click(within(dialog).getByRole("button", { name: "加入并开始复核" }));
    expect(await within(dialog).findByText("已加入研判单，可提交人工结论")).toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: /已加入研判单/u })).not.toBeInTheDocument();
    expect(api.createCase).toHaveBeenCalledTimes(1);
    expect(api.addCaseItem).toHaveBeenCalledWith(draft.caseId, "node", "n1", expect.any(String));
    expect(api.updateCase).toHaveBeenCalledWith(draft.caseId, "active", expect.any(String));

    expect(confirm).toBeDisabled();
    expect(within(dialog).getByText("填写理由后可提交。")).toBeInTheDocument();
    fireEvent.change(within(dialog).getByRole("textbox", { name: "复核理由" }), { target: { value: "已核对一跳关系与关系模态。" } });
    expect(confirm).toBeEnabled();
    fireEvent.click(confirm);
    await waitFor(() => expect(api.review).toHaveBeenCalledWith(draft.caseId, "node", "n1", "confirmed", "已核对一跳关系与关系模态。"));
    await waitFor(() => expect(latestPresentation(onGraphPresentationChange)?.skillsContext?.caseHash).toBe(confirmed.caseHash));
    expect(latestPresentation(onGraphPresentationChange)?.activeOverlay?.presentation?.reviewDecisions).toMatchObject({ n1: "confirmed" });
    fireEvent.click(within(dialog).getByRole("button", { name: "关闭证据档案" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: /#1 匿名账号 1/u })).not.toBeInTheDocument());
    fireEvent.click(screen.getByRole("tab", { name: /已研判/u }));
    expect(await screen.findByRole("button", { name: /匿名账号 1.*人工确认/u })).toBeInTheDocument();
    expect(screen.getByText("人工确认")).toBeInTheDocument();
  });

  it("presents risk groups, factual relations, and potential clues with governance-specific evidence", async () => {
    const group: GovernanceDerivation = { ...derivationPage("group").items[0], modalities: ["coRT", "coURL"] } as GovernanceDerivation;
    const api = client({
      derivations: vi.fn(async (_runId: string, kind: GovernanceDerivation["kind"]): Promise<readonly GovernanceDerivation[]> => kind === "group" ? [group] : derivationPage(kind).items as readonly GovernanceDerivation[]),
    });
    render(workspace(api));
    await waitForHydration();
    fireEvent.click(screen.getByRole("button", { name: "群组与关系" }));

    const tabs = screen.getByRole("tablist", { name: "群组与关系类型" });
    expect(within(tabs).getByRole("tab", { name: /风险群组/u })).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText(/高风险 1 · 建议复核 1 · 协同转发、共链传播/u)).toBeInTheDocument();

    fireEvent.click(within(tabs).getByRole("tab", { name: /事实关系/u }));
    expect(await screen.findByText(/事实关系 · 高风险候选 \/ 建议复核/u)).toBeInTheDocument();
    fireEvent.click(within(tabs).getByRole("tab", { name: /潜在线索/u }));
    expect(await screen.findByText(/潜在线索（非事实边）/u)).toBeInTheDocument();
  });

  it("keeps run history in an icon menu and assistant content in its own mode", async () => {
    const api = client();
    const rendered = render(workspace(api));
    await waitForHydration();

    fireEvent.click(screen.getByRole("button", { name: "研判助手" }));
    rendered.rerender(workspace(api, vi.fn(), snapshot(), true));
    expect(screen.getByText("助手报告工作面")).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText("运行记录"));
    expect(screen.getByText("运行记录")).toBeInTheDocument();
    expect(screen.getByText("已完成")).toBeInTheDocument();
  });

  it("keeps incomplete history read-only and never retries from governance", async () => {
    const failed = { ...onlineRun("failed", GOVERNANCE_OTHER_RUN_ID), statusHash: "f".repeat(64) } as GovernanceOnlineRun;
    const succeeded = onlineRun() as GovernanceOnlineRun;
    const api = client({ listRuns: vi.fn(async (): Promise<readonly GovernanceOnlineRun[]> => [failed, succeeded]) });
    render(workspace(api));
    await waitForHydration();
    fireEvent.click(screen.getByLabelText("运行记录"));

    const failedEntry = screen.getByTitle("未完成记录仅供查阅");
    expect(failedEntry).toBeDisabled();
    fireEvent.click(failedEntry);
    expect(api.retryRun).not.toHaveBeenCalled();
    expect(api.createRun).not.toHaveBeenCalled();
  });
});
