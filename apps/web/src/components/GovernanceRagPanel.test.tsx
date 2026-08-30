import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SocialGraphApiError } from "../services/apiClient";
import {
  GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
  GOVERNANCE_SKILLS_SCHEMA,
  type GovernanceSkillsClientLike,
  type GovernanceSkillsContext,
} from "../types/governanceSkills";
import { GovernanceRagPanel } from "./GovernanceRagPanel";

afterEach(cleanup);

const context: GovernanceSkillsContext = {
  graph: { artifactId: `governance-artifact-${"1".repeat(32)}`, datasetContentHash: "a".repeat(64), graphVersionHash: "b".repeat(64) },
  model: { modelVersionId: "socialgraph-fm-global/test", modelStateHash: "c".repeat(64) },
  runId: `governance-${"2".repeat(32)}`,
  caseId: `case-${"3".repeat(32)}`,
  caseHash: "d".repeat(64),
  selectedNodeIds: ["node-1"],
  selectedTarget: { kind: "node", targetId: "node-1" },
};

const otherContext: GovernanceSkillsContext = {
  ...context,
  selectedNodeIds: ["node-2"],
  selectedTarget: { kind: "node", targetId: "node-2" },
};

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((resolvePromise) => { resolve = resolvePromise; });
  return { promise, resolve };
}

function client(): GovernanceSkillsClientLike {
  return {
    catalog: vi.fn(async () => ({ schemaVersion: GOVERNANCE_SKILLS_SCHEMA, items: [], catalogHash: "d".repeat(64) })),
    executeSkill: vi.fn(async () => { throw new Error("direct skills are not exposed"); }),
    confirmSkill: vi.fn(async () => { throw new Error("direct skills are not exposed"); }),
    assistantTurn: vi.fn(async () => { throw new Error("unused"); }),
    dispatchAssistant: vi.fn(async () => ({
      schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
      dispatchId: `governance-dispatch-${"4".repeat(32)}`,
      intent: "answer" as const,
      answerMode: "overview" as const,
      status: "completed" as const,
      answer: "### 研判结论\n\n当前证据支持继续人工核验。",
      result: {},
      deterministicFallback: false,
      generationMode: "llm_assisted" as const,
      fallbackPhase: null,
      reasonCode: null,
      evidenceRefs: [{ label: "图谱检查", sourceKind: "skill" as const, hash: "e".repeat(64) }],
      confirmation: null,
      navigation: null,
      skillCalls: [{ skill: "inspect_graph" as const, requestHash: "f".repeat(64), resultHash: "1".repeat(64) }],
      citedHashes: ["2".repeat(64)],
      auditHash: "3".repeat(64),
    })),
    searchKnowledge: vi.fn(async () => ({ schemaVersion: GOVERNANCE_SKILLS_SCHEMA, items: [], indexHash: "4".repeat(64), auditHash: "5".repeat(64) })),
    searchSimilarCases: vi.fn(async () => ({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      query: { runId: context.runId },
      items: [{
        caseId: `case-${"6".repeat(32)}`,
        score: 0.91,
        components: { embedding: 0.9, structure: 0.8, modality: 0.7 },
        graphVersionHash: "7".repeat(64),
        modelStateHash: "8".repeat(64),
        kindKey: "node",
        kindEntries: [{ kind: "node" as const, targetIds: ["node-2"] }],
        concludedAt: "2026-08-18T08:00:00Z",
        recordHash: "9".repeat(64),
      }],
      indexHash: "a".repeat(64),
      backfill: { succeeded: 1 },
      auditHash: "b".repeat(64),
    })),
  };
}

describe("governance report assistant", () => {
  it("removes the repeated object/case context and keeps reports plus historical cases", () => {
    const api = client();
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} embedded />);

    expect(screen.queryByRole("region", { name: "当前研判上下文" })).not.toBeInTheDocument();
    expect(screen.queryByText("研判单已建立")).not.toBeInTheDocument();
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual(["研判报告", "历史案例"]);
    expect(screen.queryByRole("tab", { name: "分析链路" })).not.toBeInTheDocument();
    expect(api.catalog).not.toHaveBeenCalled();
    expect(api.executeSkill).not.toHaveBeenCalled();
  });

  it("renders four wide report tasks and dispatches every task through an explicit answer mode", async () => {
    const api = client();
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} embedded />);
    const taskArea = screen.getByRole("group", { name: "研判报告任务" });
    expect(within(taskArea).getAllByRole("button")).toHaveLength(4);

    const modes = [
      ["全局态势报告", "analysis_summary"],
      ["当前账号证据报告", "evidence_requirements"],
      ["群组与关系研判报告", "coordination_summary"],
      ["人工研判草稿", "case_draft"],
    ] as const;
    for (const [name, answerMode] of modes) {
      fireEvent.click(screen.getByRole("button", { name: new RegExp(name, "u") }));
      await waitFor(() => expect(api.dispatchAssistant).toHaveBeenLastCalledWith(context, expect.any(String), { intent: "answer", answerMode }, expect.any(AbortSignal)));
      fireEvent.click(await screen.findByRole("button", { name: "更换报告任务" }));
    }
  });

  it("collapses tasks after generation and renders a document with folded source hashes", async () => {
    const api = client();
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} embedded />);
    fireEvent.click(screen.getByRole("button", { name: /全局态势报告/u }));

    expect(await screen.findByRole("heading", { level: 3, name: "研判结论" })).toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "研判报告任务" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "更换报告任务" })).toBeInTheDocument();
    expect(screen.getByText("智能整理")).toBeInTheDocument();
    const sources = screen.getByText("依据来源").closest("details");
    expect(sources).not.toBeNull();
    expect(within(sources!).getByTitle("1".repeat(64))).toHaveTextContent("11111111…111111");
    expect(within(sources!).getByTitle("e".repeat(64))).toHaveTextContent("eeeeeeee…eeeeee");
    expect(within(sources!).getByTitle("2".repeat(64))).toHaveTextContent("22222222…222222");
  });

  it("supports a free-form report while preserving the selected context", async () => {
    const api = client();
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.change(screen.getByLabelText("输入研判问题"), { target: { value: "当前还缺哪些证据？" } });
    fireEvent.click(screen.getByRole("button", { name: "生成报告" }));
    await waitFor(() => expect(api.dispatchAssistant).toHaveBeenCalledWith(context, "当前还缺哪些证据？", { intent: "answer" }, expect.any(AbortSignal)));
  });

  it("aborts and discards an old report when the selected object changes", async () => {
    const api = client();
    const pending = deferred<Awaited<ReturnType<GovernanceSkillsClientLike["dispatchAssistant"]>>>();
    let signal: AbortSignal | undefined;
    vi.mocked(api.dispatchAssistant).mockImplementationOnce((_context, _message, _options, requestSignal) => {
      signal = requestSignal;
      return pending.promise;
    });
    const { rerender } = render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /全局态势报告/u }));
    await waitFor(() => expect(signal).toBeDefined());

    rerender(<GovernanceRagPanel client={api} context={otherContext} onClose={vi.fn()} />);
    expect(signal?.aborted).toBe(true);
    expect(screen.getByRole("status")).toHaveTextContent("对象已变化，上一结果已安全清除");
    await act(async () => {
      pending.resolve({
        schemaVersion: GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA,
        dispatchId: `governance-dispatch-${"0".repeat(32)}`,
        intent: "answer",
        answerMode: "overview",
        status: "completed",
        answer: "旧对象报告不得显示",
        result: {},
        deterministicFallback: false,
        confirmation: null,
        navigation: null,
        skillCalls: [],
        citedHashes: [],
        auditHash: "0".repeat(64),
      });
      await pending.promise;
    });
    expect(screen.queryByText("旧对象报告不得显示")).not.toBeInTheDocument();
  });

  it("invalidates an in-flight report when the current case revision changes", async () => {
    const api = client();
    const pending = deferred<Awaited<ReturnType<GovernanceSkillsClientLike["dispatchAssistant"]>>>();
    let signal: AbortSignal | undefined;
    vi.mocked(api.dispatchAssistant).mockImplementationOnce((_context, _message, _options, requestSignal) => {
      signal = requestSignal;
      return pending.promise;
    });
    const rendered = render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: /全局态势报告/u }));
    await waitFor(() => expect(signal).toBeDefined());

    rendered.rerender(<GovernanceRagPanel client={api} context={{ ...context, caseHash: "e".repeat(64) }} onClose={vi.fn()} />);
    expect(signal?.aborted).toBe(true);
    expect(screen.getByRole("status")).toHaveTextContent("研判单已更新，请按最新人工记录重新生成报告");
    expect(screen.getByRole("group", { name: "研判报告任务" })).toBeInTheDocument();
  });

  it("retrieves similar reviewed cases with provenance kept behind a disclosure", async () => {
    const api = client();
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "历史案例" }));
    fireEvent.click(screen.getByRole("button", { name: "检索相似历史案例" }));

    expect(await screen.findByText("历史案例 01")).toBeInTheDocument();
    expect(screen.getByText("语义接近 90%")).toBeInTheDocument();
    expect(screen.getByText("依据来源")).toBeInTheDocument();
    expect(api.searchSimilarCases).toHaveBeenCalledWith(context, {
      runId: context.runId,
      kindEntries: [{ kind: "node", targetIds: ["node-1"] }],
    }, expect.any(AbortSignal));
  });

  it("disables case-only retrieval for an empty review case", () => {
    const emptyCaseContext = {
      ...context,
      selectedNodeIds: undefined,
      selectedTarget: undefined,
      caseItemCount: 0,
    } as GovernanceSkillsContext & { readonly caseItemCount: number };
    render(<GovernanceRagPanel client={client()} context={emptyCaseContext} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "历史案例" }));

    expect(screen.getByRole("button", { name: "检索相似历史案例" })).toBeDisabled();
    expect(screen.getByText("当前研判单还没有治理对象，加入对象后即可检索。")).toBeInTheDocument();
  });

  it("allows a populated active case to use its bound objects", async () => {
    const api = client();
    const populatedCaseContext = {
      ...context,
      selectedNodeIds: undefined,
      selectedTarget: undefined,
      caseItemCount: 1,
    } as GovernanceSkillsContext & { readonly caseItemCount: number };
    render(<GovernanceRagPanel client={api} context={populatedCaseContext} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "历史案例" }));
    fireEvent.click(screen.getByRole("button", { name: "检索相似历史案例" }));

    await waitFor(() => expect(api.searchSimilarCases).toHaveBeenCalledWith(populatedCaseContext, {}, expect.any(AbortSignal)));
  });

  it("keeps the last successful cases when a typed service retry fails", async () => {
    const api = client();
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "历史案例" }));
    const search = screen.getByRole("button", { name: "检索相似历史案例" });
    fireEvent.click(search);
    expect(await screen.findByText("历史案例 01")).toBeInTheDocument();

    vi.mocked(api.searchSimilarCases).mockRejectedValueOnce(new SocialGraphApiError("CASE_INDEX_UNAVAILABLE", "index unavailable", 503));
    fireEvent.click(search);
    expect(await screen.findByText(/历史案例服务暂时不可用/u)).toHaveTextContent("上次成功结果已保留");
    expect(screen.getByText("历史案例 01")).toBeInTheDocument();
  });

  it.each([
    ["GOVERNANCE_SIMILAR_CASE_TARGETS_REQUIRED", "当前研判单还没有治理对象"],
    ["GOVERNANCE_SIMILAR_CASE_INDEX_NOT_READY", "当前研判单尚未形成可检索索引"],
    ["GOVERNANCE_SIMILAR_CASE_STATE_UNSUPPORTED", "当前研判单状态暂不支持历史案例检索"],
    ["GOVERNANCE_SIMILAR_CASE_IDENTITY_MISMATCH", "当前研判对象已失效"],
  ])("maps similar-case conflict %s to a specific recovery message", async (code, message) => {
    const api = client();
    vi.mocked(api.searchSimilarCases).mockRejectedValueOnce(new SocialGraphApiError(code, code, 409));
    render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "历史案例" }));
    fireEvent.click(screen.getByRole("button", { name: "检索相似历史案例" }));
    expect(await screen.findByText(new RegExp(message, "u"))).toBeInTheDocument();
  });

  it("restores a successful object query from the run/object/index cache", async () => {
    const api = client();
    const rendered = render(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    fireEvent.click(screen.getByRole("tab", { name: "历史案例" }));
    fireEvent.click(screen.getByRole("button", { name: "检索相似历史案例" }));
    expect(await screen.findByText("历史案例 01")).toBeInTheDocument();

    rendered.rerender(<GovernanceRagPanel client={api} context={otherContext} onClose={vi.fn()} />);
    await waitFor(() => expect(screen.queryByText("历史案例 01")).not.toBeInTheDocument());
    rendered.rerender(<GovernanceRagPanel client={api} context={context} onClose={vi.fn()} />);
    expect(await screen.findByText("历史案例 01")).toBeInTheDocument();
    expect(api.searchSimilarCases).toHaveBeenCalledTimes(1);
  });

  it("explains task prerequisites without referring to package intake", () => {
    render(<GovernanceRagPanel client={client()} context={null} onClose={vi.fn()} />);
    expect(screen.getByRole("button", { name: /全局态势报告/u })).toBeDisabled();
    expect(screen.getAllByText("当前会话尚未形成治理上下文").length).toBeGreaterThan(0);
    expect(screen.queryByText(/推理包/u)).not.toBeInTheDocument();
  });
});
