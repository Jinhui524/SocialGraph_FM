import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { evidencePayload, onlineFinding } from "../test/fixtures/governanceOnline";
import { ASSISTANT_SKILL_RESULT_SCHEMA, type GovernanceSkillsClientLike, type GovernanceSkillsContext } from "../types/governanceSkills";
import type { GovernanceOnlineEvidence, GovernanceOnlineFinding } from "../types/governanceOnline";
import { EVIDENCE_SUMMARY_TIMEOUT_MS, EvidenceDossier } from "./EvidenceDossier";

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

const context: GovernanceSkillsContext = {
  graph: { artifactId: `governance-artifact-${"1".repeat(32)}`, datasetContentHash: "a".repeat(64), graphVersionHash: "b".repeat(64) },
  model: { modelVersionId: "socialgraph-fm-global/test", modelStateHash: "c".repeat(64) },
  runId: `governance-${"2".repeat(32)}`,
  selectedNodeIds: ["n1"],
  selectedTarget: { kind: "node", targetId: "n1" },
};

function response(answer = "### 关注原因\n\n当前账号需要继续人工核验。") {
  return {
    schemaVersion: ASSISTANT_SKILL_RESULT_SCHEMA,
    executionId: `assistant-exec-${"3".repeat(32)}`,
    skill: "summarize_node_evidence" as const,
    answer,
    result: {},
    evidenceRefs: [],
    skillCalls: [],
    citedHashes: [],
    auditHash: "4".repeat(64),
  };
}

function summaryClient(implementation?: GovernanceSkillsClientLike["executeAssistant"]): GovernanceSkillsClientLike {
  return {
    executeAssistant: implementation ?? vi.fn(async () => response()),
  } as unknown as GovernanceSkillsClientLike;
}

function props(overrides: Record<string, unknown> = {}) {
  return {
    open: true,
    target: { kind: "node" as const, id: "n1", nodeIds: ["n1"] },
    title: "匿名账号 1",
    finding: onlineFinding() as GovernanceOnlineFinding,
    derivation: null,
    derivationRank: null,
    evidence: { state: "ready" as const, value: evidencePayload() as GovernanceOnlineEvidence },
    skillsContext: context,
    summaryClient: summaryClient(),
    reviewContent: <div>人工复核表单</div>,
    candidateLabel: (nodeId: string) => nodeId === "n1" ? "匿名账号 1" : "匿名账号 2",
    reviewRank: () => 1,
    onSelectNeighbor: vi.fn(),
    onClose: vi.fn(),
    ...overrides,
  };
}

describe("EvidenceDossier", () => {
  it("is a centered accessible dialog with three tabs and explicit evidence gaps", () => {
    render(<EvidenceDossier {...props()} />);
    const dialog = screen.getByRole("dialog", { name: "匿名账号 1" });

    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(within(dialog).getAllByRole("tab").map((tab) => tab.textContent)).toEqual(["证据摘要", "关系事实", "人工复核"]);
    expect(within(dialog).getByText("结构化研判摘要")).toBeInTheDocument();
    expect(within(dialog).getByText("受事实约束")).toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole("tab", { name: "关系事实" }));
    expect(within(dialog).getByRole("columnheader", { name: "关系模态" })).toBeInTheDocument();
    expect(within(dialog).getByText("协同转发")).toBeInTheDocument();
    expect(within(dialog).getByText("0.84")).toBeInTheDocument();
    expect(within(dialog).getByText(/发布时间、原帖内容及采集来源需要在人工复核中补充/u)).toBeInTheDocument();
    expect(within(dialog).queryByText("依据来源")).not.toBeInTheDocument();
    fireEvent.click(within(dialog).getByRole("tab", { name: "人工复核" }));
    expect(within(dialog).getByText("人工复核表单")).toBeInTheDocument();
  });

  it("keeps the selected tab when node evidence finishes loading", () => {
    const rendered = render(<EvidenceDossier {...props({ evidence: { state: "loading" } })} />);
    const dialog = screen.getByRole("dialog", { name: "匿名账号 1" });
    fireEvent.click(within(dialog).getByRole("tab", { name: "关系事实" }));
    expect(within(dialog).getByRole("tab", { name: "关系事实" })).toHaveAttribute("aria-selected", "true");

    rendered.rerender(<EvidenceDossier {...props()} />);
    expect(within(dialog).getByRole("tab", { name: "关系事实" })).toHaveAttribute("aria-selected", "true");
    expect(within(dialog).getByRole("columnheader", { name: "关联账号" })).toBeInTheDocument();
  });

  it("generates a node summary only after an explicit click and reuses its session cache", async () => {
    const executeAssistant = vi.fn(async () => response("### 关注原因\n\n这是受事实约束的证据研判。"));
    const api = summaryClient(executeAssistant);
    const evidence = { ...evidencePayload(), evidenceHash: "5".repeat(64) } as GovernanceOnlineEvidence;
    const rendered = render(<EvidenceDossier {...props({ summaryClient: api, evidence: { state: "ready", value: evidence } })} />);
    expect(executeAssistant).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "生成证据研判摘要" }));
    expect(await screen.findByText("这是受事实约束的证据研判。")).toBeInTheDocument();
    expect(executeAssistant).toHaveBeenCalledWith(
      context,
      "summarize_node_evidence",
      expect.stringContaining("300至600个中文字符"),
      expect.any(AbortSignal),
    );
    const submittedPrompt = (executeAssistant.mock.calls as unknown as readonly (readonly unknown[])[])[0]?.[2];
    expect(submittedPrompt).toBeTypeOf("string");
    if (typeof submittedPrompt !== "string") throw new Error("summary prompt was not captured");
    expect(submittedPrompt.length).toBeLessThanOrEqual(2_000);
    expect(submittedPrompt).not.toMatch(/匿名账号|rawWeight|nodeId|relations/u);

    rendered.rerender(<EvidenceDossier {...props({ open: false, summaryClient: api, evidence: { state: "ready", value: evidence } })} />);
    rendered.rerender(<EvidenceDossier {...props({ summaryClient: api, evidence: { state: "ready", value: evidence } })} />);
    expect(await screen.findByText("这是受事实约束的证据研判。")).toBeInTheDocument();
    expect(executeAssistant).toHaveBeenCalledTimes(1);
  });

  it("cancels an old summary when the selected target changes", async () => {
    let resolve!: (value: ReturnType<typeof response>) => void;
    let signal: AbortSignal | undefined;
    const pending = new Promise<ReturnType<typeof response>>((resolvePromise) => { resolve = resolvePromise; });
    const executeAssistant = vi.fn((_context, _skill, _message, requestSignal?: AbortSignal) => {
      signal = requestSignal;
      return pending;
    });
    const api = summaryClient(executeAssistant);
    const firstEvidence = { ...evidencePayload(), evidenceHash: "6".repeat(64) } as GovernanceOnlineEvidence;
    const secondEvidence = { ...evidencePayload(), evidenceHash: "7".repeat(64), node: onlineFinding("n2", 2, 0.58) } as GovernanceOnlineEvidence;
    const rendered = render(<EvidenceDossier {...props({ summaryClient: api, evidence: { state: "ready", value: firstEvidence } })} />);
    fireEvent.click(screen.getByRole("button", { name: "生成证据研判摘要" }));
    await waitFor(() => expect(signal).toBeDefined());

    rendered.rerender(<EvidenceDossier {...props({
      target: { kind: "node", id: "n2", nodeIds: ["n2"] },
      title: "匿名账号 2",
      finding: onlineFinding("n2", 2, 0.58) as GovernanceOnlineFinding,
      evidence: { state: "ready", value: secondEvidence },
      summaryClient: api,
    })} />);
    expect(signal?.aborted).toBe(true);
    await act(async () => { resolve(response("旧对象摘要不得显示")); await pending; });
    expect(screen.queryByText("旧对象摘要不得显示")).not.toBeInTheDocument();
  });

  it("resets an aborted summary when the same dossier is closed and reopened", async () => {
    let signal: AbortSignal | undefined;
    const pending = new Promise<ReturnType<typeof response>>(() => undefined);
    const api = summaryClient(vi.fn((_context, _skill, _message, requestSignal?: AbortSignal) => {
      signal = requestSignal;
      return pending;
    }));
    const rendered = render(<EvidenceDossier {...props({ summaryClient: api })} />);
    fireEvent.click(screen.getByRole("button", { name: "生成证据研判摘要" }));
    await waitFor(() => expect(signal).toBeDefined());

    rendered.rerender(<EvidenceDossier {...props({ open: false, summaryClient: api })} />);
    expect(signal?.aborted).toBe(true);
    rendered.rerender(<EvidenceDossier {...props({ summaryClient: api })} />);
    expect(screen.getByRole("button", { name: "生成证据研判摘要" })).toBeEnabled();
  });

  it("times out an explicit summary without hiding deterministic evidence", async () => {
    vi.useFakeTimers();
    const api = summaryClient(vi.fn(() => new Promise<ReturnType<typeof response>>(() => undefined)));
    render(<EvidenceDossier {...props({ summaryClient: api })} />);
    fireEvent.click(screen.getByRole("button", { name: "生成证据研判摘要" }));

    await act(async () => { await vi.advanceTimersByTimeAsync(EVIDENCE_SUMMARY_TIMEOUT_MS); });
    expect(screen.getByText("智能摘要生成超时，结构化证据仍可继续核对。")).toBeInTheDocument();
    expect(screen.getByText("结构化研判摘要")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重新生成证据研判摘要" })).toBeEnabled();
  });

  it("keeps deterministic evidence usable when generation fails", async () => {
    const api = summaryClient(vi.fn(async () => { throw new Error("offline"); }));
    const evidence = { ...evidencePayload(), evidenceHash: "8".repeat(64) } as GovernanceOnlineEvidence;
    render(<EvidenceDossier {...props({ summaryClient: api, evidence: { state: "ready", value: evidence } })} />);
    fireEvent.click(screen.getByRole("button", { name: "生成证据研判摘要" }));

    expect(await screen.findByText("智能摘要暂未生成，结构化证据仍可继续核对。")).toBeInTheDocument();
    expect(screen.getByText("结构化研判摘要")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重新生成证据研判摘要" })).toBeInTheDocument();
  });

  it("closes with Escape without conflating close and selection", () => {
    const onClose = vi.fn();
    render(<EvidenceDossier {...props({ onClose })} />);
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });
});
