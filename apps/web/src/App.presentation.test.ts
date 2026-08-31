import { createElement, type ComponentType } from "react";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import * as appPresentation from "./App";
import { createGraphVersion } from "./services/graphImport";
import { buildCommunityOverlay, buildPathOverlay } from "./services/graphOverlays";
import { buildGraphScene, createDefaultGraphViewState } from "./services/graphScene";
import { createGraphWorkbenchViewState, reduceGraphView, type GraphViewAction } from "./services/graphViewState";
import type { AnalysisOverlay, AnalysisRun, GraphVersion } from "./types/graph";
import { GraphCameraSnapshotCache, type GraphCameraSnapshotCacheKey } from "./services/graphEngineAdapter";
import { adaptationHandoff, targetComparison, targetPolicy, targetPreview, targetResult, targetRun, targetTaskRegistration } from "./test/fixtures/governanceTargetTask";
import { buildAdaptedReviewPriorityOverlay } from "./services/governanceAdaptation";
import { governanceImportedGraphVersion } from "./components/GovernanceOnlineWorkspace";

afterEach(cleanup);

function overlayTestGraph(sourceFile: string, nodeId: string): GraphVersion {
  return createGraphVersion(sourceFile, [{ id: nodeId, label: nodeId, type: "person", attributes: {} }], []);
}

function localOverviewRun(): AnalysisRun {
  return {
    id: "run-local-overview",
    graphVersionId: "graph-russia",
    intent: {
      kind: "analysis_request",
      normalizedText: "生成当前范围概览",
      task: "overview",
      targets: [],
      confidence: 1,
      filters: {},
      meta: {
        schemaVersion: "1.1",
        source: "llm",
        requestId: "request-local-overview",
        warnings: [],
      },
    },
    engine: "local_algorithm",
    status: "succeeded",
    createdAt: "2026-08-20T00:00:00.000Z",
    completedAt: "2026-08-20T00:00:01.000Z",
    scope: {
      graphVersionId: "graph-russia",
      scopeHash: "sensitive-scope-hash",
      nodeCount: 12,
      edgeCount: 19,
      filters: { nodeTypes: [], edgeTypes: [] },
      nodeIds: ["account-a", "account-b"],
      edgeIds: ["relation-a-b"],
      truncated: false,
    },
    result: {
      kind: "overview",
      summary: {
        nodeCount: 12,
        edgeCount: 19,
        density: 0.288,
        averageDegree: 3.167,
        connectedComponents: 1,
        isolatedNodes: 0,
      },
      topDegree: [
        { nodeId: "account-a", label: "账号 A", degree: 7, normalizedScore: 1 },
      ],
      articulationPoints: [],
    },
  };
}

describe("App conversation presentation", () => {
  it("keeps the complete immutable few-shot handoff when creating an independent governance task", () => {
    const registration = targetTaskRegistration();
    const graph = governanceImportedGraphVersion(targetPreview(true) as any, registration.task.displayName);
    const policy = targetPolicy();
    const comparison = targetComparison();
    const result = targetResult();
    const snapshot = {
      schemaVersion: "socialgraph-fm.governance-workspace/1.0" as const,
      sessionId: registration.registrationId,
      sourceFileName: "target-domain-b-few.sgtask.zip",
      artifact: registration.artifact,
      preview: targetPreview(true),
      run: targetRun(),
      result,
      updatedAt: "2026-08-21T10:00:00Z",
    };
    const entry = appPresentation.governanceTaskEntryFromAdaptationTarget({
      lane: "few_shot",
      registration,
      snapshot,
      graph,
      handoff: adaptationHandoff(),
      policy,
      comparison,
      adaptedOverlay: buildAdaptedReviewPriorityOverlay(graph, result as any, comparison as any),
    } as any);

    expect(entry).toMatchObject({
      id: registration.registrationId,
      kind: "target",
      snapshot,
      graph,
      validationToken: 0,
      adaptation: {
        lane: "few_shot",
        registration,
        handoff: { handoffHash: adaptationHandoff().handoffHash },
        policy: { policyHash: policy.policyHash },
        comparison: { comparisonHash: comparison.comparisonHash },
      },
    });
    expect(entry.adaptation?.adaptedOverlay?.presentation?.adaptedRanks?.["target-node-108"]).toBe(1);
    expect(() => appPresentation.governanceTaskEntryFromAdaptationTarget({ lane: "few_shot", registration, snapshot, graph } as any)).toThrow("ADAPTATION_GOVERNANCE_HANDOFF_INCOMPLETE");
  });

  it("renders the atlas welcome as semantic research entry paths without a duplicate hero image", () => {
    const WelcomeAtlas = (appPresentation as typeof appPresentation & {
      WelcomeAtlas?: ComponentType<{ onPrompt: (prompt: appPresentation.ResearchPrompt) => void; onUpload: () => void }>;
    }).WelcomeAtlas;
    const onPrompt = vi.fn();
    expect(WelcomeAtlas).toBeTypeOf("function");
    const { container } = render(createElement(WelcomeAtlas!, { onPrompt, onUpload: vi.fn() }));
    expect(screen.getByRole("region", { name: "学术网络图谱开始页" })).toBeVisible();
    expect(screen.getByRole("heading", { name: "从关系图谱到治理研判" })).toBeVisible();
    expect(screen.getByText("描述目标，系统编排分析", { exact: true })).toBeVisible();
    expect(screen.getByRole("list", { name: "研究流程" })).toHaveTextContent("提出问题组织分析进入复核");
    expect(screen.getByRole("navigation", { name: "研究入口" })).toBeVisible();
    expect(screen.getAllByRole("button", { name: /图谱基本情况|人工复核流程|证据核对清单/u })).toHaveLength(3);
    fireEvent.click(screen.getByRole("button", { name: /图谱基本情况/u }));
    expect(onPrompt).toHaveBeenCalledWith(expect.objectContaining({
      text: "请概括当前图谱的账号规模、事实关系数量、关系类型和连通情况",
      skill: "answer_governance_question",
      contextScope: "graph",
    }));
    expect(container.querySelector("img")).not.toBeInTheDocument();
    expect(appPresentation.welcomePromptAction(false)).toBe("prepare_upload");
    expect(appPresentation.welcomePromptAction(true)).toBe("send");
  });

  it("executes each research card through the question Skill with explicit graph scope", () => {
    expect(appPresentation.researchPrompts.map(({ title, skill, contextScope }) => ({ title, skill, contextScope })))
      .toEqual([
        { title: "图谱基本情况", skill: "answer_governance_question", contextScope: "graph" },
        { title: "人工复核流程", skill: "answer_governance_question", contextScope: "workspace" },
        { title: "证据核对清单", skill: "answer_governance_question", contextScope: "workspace" },
      ]);
    const context = {
      graph: { artifactId: "artifact", datasetContentHash: "dataset", graphVersionHash: "graph" },
      model: { modelVersionId: "model", modelStateHash: "state" },
      runId: "run",
      caseId: "case",
      selectedNodeIds: ["account-a"],
      selectedTarget: { kind: "node" as const, targetId: "account-a" },
    };
    const overview = appPresentation.researchPromptSkillRequest(context, appPresentation.researchPrompts[0]);
    expect(overview.skill).toBe("answer_governance_question");
    expect(overview.context).toEqual({ graph: context.graph, model: context.model });
    const evidence = appPresentation.researchPromptSkillRequest(context, appPresentation.researchPrompts[2]);
    expect(evidence.skill).toBe("answer_governance_question");
    expect(evidence.context).toBe(context);
    expect(appPresentation.researchPromptForText(`  ${appPresentation.researchPrompts[1].text}  `)?.skill)
      .toBe("answer_governance_question");
  });

  it("keeps the graph summary concise without scope, layer, or clean-quality rows", () => {
    const graph = createGraphVersion("summary.csv", [
      { id: "a", label: "账号 A", type: "account", attributes: {} },
      { id: "b", label: "账号 B", type: "account", attributes: {} },
    ], [{ id: "a-b", source: "a", target: "b", type: "coURL", attributes: {} }]);
    const viewState = { ...createDefaultGraphViewState(graph.id), theme: "focus-dark" as const };
    render(createElement(appPresentation.RightSummary, {
      graph,
      selectedNode: null,
      viewState,
      scene: buildGraphScene(graph, { viewState }),
      onExport: vi.fn(),
    }));
    expect(screen.getByText("当前视图", { exact: true })).toBeVisible();
    expect(screen.queryByText("范围", { exact: true })).not.toBeInTheDocument();
    expect(screen.queryByText("完整图", { exact: true })).not.toBeInTheDocument();
    expect(screen.queryByText("清除图层", { exact: true })).not.toBeInTheDocument();
    expect(screen.queryByText("未发现阻断性数据问题", { exact: true })).not.toBeInTheDocument();
  });

  it("stores and reactivates graph presentation independently for both adaptation lanes", () => {
    const createState = (appPresentation as typeof appPresentation & {
      createAdaptationLanePresentationState?: () => any;
    }).createAdaptationLanePresentationState;
    const updateLane = (appPresentation as typeof appPresentation & {
      updateAdaptationLanePresentation?: (state: any, lane: "zero_shot" | "few_shot", patch: any) => any;
    }).updateAdaptationLanePresentation;
    const activateLane = (appPresentation as typeof appPresentation & {
      activateAdaptationLanePresentation?: (state: any, lane: "zero_shot" | "few_shot") => any;
    }).activateAdaptationLanePresentation;
    expect(createState).toBeTypeOf("function"); expect(updateLane).toBeTypeOf("function"); expect(activateLane).toBeTypeOf("function");
    const zeroGraph = overlayTestGraph("zero-target", "zero-node"); const fewGraph = overlayTestGraph("few-target", "few-node");
    const zero = { graph: zeroGraph, overlay: buildCommunityOverlay(zeroGraph), focus: { kind: "node", targetId: "zero-node", nodeIds: ["zero-node"], cameraToken: 1 }, camera: { nodeIds: ["zero-node"], token: 1 }, abortEpoch: 4 };
    const few = { graph: fewGraph, overlay: buildCommunityOverlay(fewGraph), focus: { kind: "node", targetId: "few-node", nodeIds: ["few-node"], cameraToken: 2 }, camera: { nodeIds: ["few-node"], token: 2 }, abortEpoch: 7 };
    let state = updateLane!(updateLane!(createState!(), "zero_shot", zero), "few_shot", few);
    const preservedFew = state.lanes.few_shot;
    state = updateLane!(state, "zero_shot", { graph: null, overlay: null, focus: undefined, camera: undefined, abortEpoch: 5 });
    expect(state.lanes.few_shot).toBe(preservedFew);
    expect(state.lanes.few_shot).toEqual(few);
    expect(state.activeLane).toBe("few_shot");
    state = activateLane!(state, "few_shot");
    expect(state.activeLane).toBe("few_shot");
    expect(state.lanes[state.activeLane]).toBe(preservedFew);
  });

  it("shows the adaptation graph switcher only for two complete lanes", () => {
    const zeroGraph = overlayTestGraph("zero-target", "zero-node");
    const fewGraph = overlayTestGraph("few-target", "few-node");
    const onSelect = vi.fn();
    let state = appPresentation.updateAdaptationLanePresentation(
      appPresentation.createAdaptationLanePresentationState(),
      "zero_shot",
      { graph: zeroGraph },
    );
    const view = render(createElement(appPresentation.AdaptationGraphSwitcher, { state, onSelect }));
    expect(screen.queryByRole("group", { name: "切换适配网络" })).not.toBeInTheDocument();

    state = appPresentation.updateAdaptationLanePresentation(state, "few_shot", { graph: fewGraph });
    view.rerender(createElement(appPresentation.AdaptationGraphSwitcher, { state, onSelect }));
    const zero = screen.getByRole("button", { name: "零样本网络" });
    const few = screen.getByRole("button", { name: "少样本网络" });
    expect(zero).toHaveAttribute("aria-pressed", "true");
    expect(few).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(few);
    expect(onSelect).toHaveBeenCalledWith("few_shot");

    state = appPresentation.activateAdaptationLanePresentation(state, "few_shot");
    view.rerender(createElement(appPresentation.AdaptationGraphSwitcher, { state, onSelect }));
    expect(zero).toHaveAttribute("aria-pressed", "false");
    expect(few).toHaveAttribute("aria-pressed", "true");
  });

  it("keeps the existing lane visible while another lane clears for upload", () => {
    const zeroGraph = overlayTestGraph("zero-target", "zero-node");
    const fewGraph = overlayTestGraph("few-target", "few-node");
    let state = appPresentation.createAdaptationLanePresentationState();
    state = appPresentation.updateAdaptationLanePresentation(state, "zero_shot", { graph: zeroGraph });

    expect(appPresentation.activateAdaptationLanePresentation(state, "few_shot")).toBe(state);
    state = appPresentation.updateAdaptationLanePresentation(state, "few_shot", { graph: null });
    expect(state.activeLane).toBe("zero_shot");
    expect(state.lanes[state.activeLane].graph).toBe(zeroGraph);

    state = appPresentation.updateAdaptationLanePresentation(state, "few_shot", { graph: fewGraph });
    state = appPresentation.activateAdaptationLanePresentation(state, "few_shot");
    state = appPresentation.updateAdaptationLanePresentation(state, "few_shot", { graph: null });
    expect(state.activeLane).toBe("zero_shot");
    expect(state.lanes[state.activeLane].graph).toBe(zeroGraph);
  });
  it("does not let inactive lane completions steal the explicitly active adaptation lane", () => {
    const zeroGraph = overlayTestGraph("late-zero", "zero-node"); const fewGraph = overlayTestGraph("active-few", "few-node");
    let state = appPresentation.createAdaptationLanePresentationState();
    state = appPresentation.updateAdaptationLanePresentation(state, "few_shot", { graph: fewGraph, overlay: buildCommunityOverlay(fewGraph), abortEpoch: 3 });
    state = appPresentation.activateAdaptationLanePresentation(state, "few_shot");
    const activeFew = state.lanes.few_shot;
    state = appPresentation.updateAdaptationLanePresentation(state, "zero_shot", {
      graph: zeroGraph,
      overlay: buildCommunityOverlay(zeroGraph),
      focus: { kind: "node", targetId: "zero-node", nodeIds: ["zero-node"], cameraToken: 9 },
      camera: { nodeIds: ["zero-node"], token: 9 },
      abortEpoch: 8,
    });
    expect(state.activeLane).toBe("few_shot");
    expect(state.lanes[state.activeLane]).toBe(activeFew);
    expect(state.lanes.zero_shot).toMatchObject({ graph: zeroGraph, focus: { targetId: "zero-node" }, abortEpoch: 8 });
    state = appPresentation.activateAdaptationLanePresentation(state, "zero_shot");
    expect(state.lanes[state.activeLane].graph).toBe(zeroGraph);
  });
  it("renders compact attachments above the user text and keeps the full filename accessible", () => {
    const UserEntry = (appPresentation as typeof appPresentation & {
      UserEntry?: ComponentType<{ entry: unknown }>;
    }).UserEntry;

    expect(UserEntry).toBeTypeOf("function");
    if (!UserEntry) return;
    const { container } = render(createElement(UserEntry, {
      entry: {
        id: "user-upload",
        role: "user",
        text: "请分析这份关系数据",
        timestamp: "10:00",
        files: [{ name: "一份名称很长的关系网络研究数据.csv", size: 2_048 }],
      },
    }));

    const rail = container.querySelector(".user-message-attachments");
    const bubble = container.querySelector(".user-message-bubble");
    expect(rail).toBeInTheDocument();
    expect(bubble).toHaveTextContent("请分析这份关系数据");
    expect(rail?.compareDocumentPosition(bubble!)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(screen.getByTitle("一份名称很长的关系网络研究数据.csv")).toBeInTheDocument();
    expect(screen.getByLabelText("附件已就绪")).toBeInTheDocument();
  });

  it("renders neither user nor assistant timestamps while keeping timestamp data in entries", () => {
    const UserEntry = (appPresentation as typeof appPresentation & {
      UserEntry?: ComponentType<{ entry: unknown }>;
    }).UserEntry;
    const AssistantEntry = (appPresentation as typeof appPresentation & {
      AssistantEntry?: ComponentType<{ entry: unknown }>;
    }).AssistantEntry;

    const user = render(createElement(UserEntry!, {
      entry: { id: "user-time", role: "user", text: "用户问题", timestamp: "10:00" },
    }));
    expect(user.container.querySelector("time")).not.toBeInTheDocument();
    user.unmount();

    const assistant = render(createElement(AssistantEntry!, {
      entry: {
        id: "assistant-time",
        role: "assistant",
        text: "助手回答",
        timestamp: "10:01",
        state: "success",
      },
    }));
    expect(assistant.container.querySelector("time")).not.toBeInTheDocument();
  });

  it("renders the current Global stage with all five analysis phases", () => {
    render(createElement(appPresentation.AssistantEntry, {
      entry: {
        id: "governance-running",
        role: "assistant",
        text: "正在执行风险推理。",
        timestamp: "10:02",
        state: "working",
        governanceProgress: { stage: "inferencing", progress: 58 },
      },
    }));

    const progress = screen.getByRole("region", { name: "治理分析进度" });
    expect(progress).toHaveTextContent("正在执行第 3 / 5 阶段");
    expect(Array.from(progress.querySelectorAll("li"), (item) => item.textContent)).toEqual([
      "1输入检查", "2准备图谱", "3风险推理", "4生成研判", "5整理结论",
    ]);
    expect(screen.getByRole("progressbar", { name: "治理分析完成 58%" })).toHaveValue(58);
  });

  it("presents legacy persisted Global copy as product-facing governance language", () => {
    render(createElement(appPresentation.AssistantEntry, {
      entry: {
        id: "legacy-global-copy",
        role: "assistant",
        text: "分析计划已准备好。确认后才会创建 Global 分析运行。",
        timestamp: "10:02",
        state: "success",
      },
    }));

    expect(screen.getByText("分析计划已准备好。确认后才会创建治理分析运行。")).toBeInTheDocument();
    expect(screen.queryByText(/Global/u)).not.toBeInTheDocument();
  });

  it("explains the governance object, outputs, boundary, and next review before confirmation", () => {
    render(createElement(appPresentation.AssistantEntry, {
      entry: {
        id: "governance-confirm",
        role: "assistant",
        text: "已准备治理分析。",
        timestamp: "10:02",
        state: "warning",
        confirmation: { token: "token", action: "run_governance_analysis", requestDigest: "digest", expiresAt: "later" },
      },
      onConfirm: vi.fn(),
    }));

    expect(screen.getByText("准备分析当前治理图谱", { exact: true })).toBeVisible();
    expect(screen.getByText(/系统将生成风险账号排序、协同群组和重点关系/u)).toBeVisible();
    expect(screen.getByText("确认前不会产生运行或写入记录", { exact: true })).toBeVisible();
    expect(screen.getByText("模型发现不会改写原始图事实", { exact: true })).toBeVisible();
    expect(screen.getByRole("button", { name: "确认开始分析" })).toBeVisible();
  });

  it("keeps only compact processing and error import states and renders no success timeline", () => {
    const ImportTimeline = (appPresentation as typeof appPresentation & {
      ImportTimeline?: ComponentType<{ state: unknown }>;
    }).ImportTimeline;

    expect(ImportTimeline).toBeTypeOf("function");
    if (!ImportTimeline) return;
    const success = render(createElement(ImportTimeline, {
      state: { kind: "success", fileName: "network.zip", version: {} },
    }));
    expect(success.container).toBeEmptyDOMElement();
    success.unmount();

    const parsing = render(createElement(ImportTimeline, {
      state: { kind: "parsing", fileName: "network.zip", stage: "inspect" },
    }));
    expect(screen.getByRole("status")).toHaveTextContent("正在检查 network.zip");
    expect(screen.queryByText("SocialGraph-FM 助手")).not.toBeInTheDocument();
    parsing.unmount();

    render(createElement(ImportTimeline, {
      state: { kind: "error", fileName: "network.zip", message: "包结构不兼容", issues: [] },
    }));
    expect(screen.getByRole("alert")).toHaveTextContent("包结构不兼容");
  });

  it("inserts upload guidance exactly once per artifact and content identity", () => {
    const completionEntries = (appPresentation as typeof appPresentation & {
      buildGovernanceUploadConversationEntries?: (
        file: { name: string; size: number },
        timestamp: string,
        identity: { artifactId: string; datasetContentHash: string },
      ) => readonly unknown[];
    }).buildGovernanceUploadConversationEntries;
    const mergeEntries = (appPresentation as typeof appPresentation & {
      mergeGovernanceUploadConversationEntries?: (
        current: readonly unknown[],
        additions: readonly unknown[],
      ) => readonly unknown[];
    }).mergeGovernanceUploadConversationEntries;
    const identity = {
      artifactId: `governance-artifact-${"1".repeat(32)}`,
      datasetContentHash: "a".repeat(64),
    };

    expect(completionEntries).toBeTypeOf("function");
    expect(mergeEntries).toBeTypeOf("function");
    const first = completionEntries!({ name: "network.zip", size: 4_096 }, "10:00", identity);
    const repeated = completionEntries!({ name: "network-copy.zip", size: 4_096 }, "10:01", identity);
    const changed = completionEntries!(
      { name: "network-v2.zip", size: 4_096 },
      "10:02",
      { ...identity, datasetContentHash: "b".repeat(64) },
    );
    const merged = mergeEntries!(mergeEntries!(first, repeated), changed) as readonly {
      role: string;
      text: string;
    }[];

    expect(first).toEqual([
      expect.objectContaining({
        role: "user",
        text: "上传推理包并准备分析。",
        timestamp: "10:00",
        file: { name: "network.zip", size: 4_096 },
      }),
      expect.objectContaining({
        role: "assistant",
        text: expect.stringContaining("获得风险账号排序、协同群组和重点关系"),
        timestamp: "10:00",
      }),
    ]);
    expect(merged.filter((entry) => entry.role === "user")).toHaveLength(3);
    expect(merged.filter((entry) => entry.role === "assistant")).toHaveLength(2);
  });

  it("keeps an explicit analysis overlay ahead of the default Louvain overlay for the same graph", () => {
    const resolveOverlay = (appPresentation as typeof appPresentation & {
      resolveGraphVersionOverlay?: (graphVersionId: string, explicit: unknown, fallback: unknown) => unknown;
    }).resolveGraphVersionOverlay;
    const explicit = { id: "analysis", graphVersionId: "graph-1" };
    const louvain = { id: "louvain", graphVersionId: "graph-1" };

    expect(resolveOverlay).toBeTypeOf("function");
    expect(resolveOverlay?.("graph-1", explicit, louvain)).toBe(explicit);
    expect(resolveOverlay?.("graph-1", { ...explicit, graphVersionId: "old" }, louvain)).toBe(louvain);
    expect(resolveOverlay?.("graph-2", explicit, louvain)).toBeNull();
  });

  it("focuses at most eight ranked candidates that are present in the current graph", () => {
    const resolveFocus = (appPresentation as typeof appPresentation & {
      resolveGovernanceCandidateFocus?: (input: unknown) => { status: string; nodeIds: readonly string[] };
    }).resolveGovernanceCandidateFocus;
    const previewNodes = [
      { id: "missing", score: 0.99 },
      ...Array.from({ length: 10 }, (_, index) => ({ id: `node-${index}`, score: 0.9 - index / 100 })),
    ];

    expect(resolveFocus).toBeTypeOf("function");
    expect(resolveFocus?.({
      messageRunId: "run-1",
      currentRunId: "run-1",
      runGraphVersionHash: "hash-1",
      currentGraphVersionHash: "hash-1",
      previewNodes,
      graphNodeIds: previewNodes.slice(1).map((node) => node.id),
    })).toEqual({
      status: "ready",
      nodeIds: ["node-0", "node-1", "node-2", "node-3", "node-4", "node-5", "node-6", "node-7"],
    });
  });

  it("refuses stale candidate reports and visibly distinguishable empty matches", () => {
    const resolveFocus = (appPresentation as typeof appPresentation & {
      resolveGovernanceCandidateFocus?: (input: unknown) => { status: string; nodeIds: readonly string[] };
    }).resolveGovernanceCandidateFocus;
    const base = {
      messageRunId: "run-old",
      currentRunId: "run-current",
      runGraphVersionHash: "hash-1",
      currentGraphVersionHash: "hash-1",
      previewNodes: [{ id: "candidate", score: 0.8 }],
      graphNodeIds: ["candidate"],
    };

    expect(resolveFocus).toBeTypeOf("function");
    expect(resolveFocus?.(base)).toEqual({ status: "stale", nodeIds: [] });
    expect(resolveFocus?.({ ...base, messageRunId: "run-current", graphNodeIds: [] })).toEqual({
      status: "empty",
      nodeIds: [],
    });
  });

  it("publishes raw facts until a hash-matched Global result schedules Louvain exactly once", async () => {
    type Deferred = { resolve: (overlay: AnalysisOverlay | null) => void };
    const OverlayController = appPresentation.GraphVersionOverlayController;
    const scheduled: string[] = [];
    const deferred: Deferred[] = [];
    const state: { activeOverlay: AnalysisOverlay | null } = { activeOverlay: null };
    const errors: unknown[] = [];

    expect(OverlayController).toBeTypeOf("function");
    const controller = new OverlayController({
      computeDefaultOverlay: (version) => {
        scheduled.push(version.id);
        return new Promise((resolve) => deferred.push({ resolve }));
      },
      onOverlayChange: (overlay) => { state.activeOverlay = overlay; },
      onError: (error) => { errors.push(error); },
    });
    const graphOne = overlayTestGraph("graph-1.json", "one");
    const graphTwo = overlayTestGraph("graph-2.json", "two");

    controller.activate(graphOne, null);
    expect(scheduled).toEqual([]);
    expect(state.activeOverlay).toMatchObject({ graphVersionId: graphOne.id, kind: "raw" });
    const acceptGlobalResult = (controller as typeof controller & {
      acceptGlobalResult?: (
        version: GraphVersion,
        result: { readonly protocol: "global"; readonly status: "succeeded"; readonly graphVersionHash: string; readonly runId: string; readonly resultHash: string },
      ) => boolean;
    }).acceptGlobalResult;
    expect(acceptGlobalResult).toBeTypeOf("function");
    if (!acceptGlobalResult) return;
    expect(acceptGlobalResult.call(controller, graphOne, {
      protocol: "global", status: "succeeded", graphVersionHash: graphOne.contentHash!, runId: "run-one", resultHash: "result-one",
    })).toBe(true);
    expect(acceptGlobalResult.call(controller, graphOne, {
      protocol: "global", status: "succeeded", graphVersionHash: graphOne.contentHash!, runId: "run-one", resultHash: "result-one",
    })).toBe(false);
    expect(scheduled).toEqual([graphOne.id]);

    controller.activate(graphTwo, null);
    expect(state.activeOverlay).toMatchObject({ graphVersionId: graphTwo.id, kind: "raw" });
    expect(acceptGlobalResult.call(controller, graphOne, {
      protocol: "global", status: "succeeded", graphVersionHash: graphOne.contentHash!, runId: "run-one", resultHash: "result-one",
    })).toBe(false);
    expect(acceptGlobalResult.call(controller, graphTwo, {
      protocol: "global", status: "succeeded", graphVersionHash: "stale-hash", runId: "run-two", resultHash: "result-two",
    })).toBe(false);
    expect(acceptGlobalResult.call(controller, graphTwo, {
      protocol: "global", status: "succeeded", graphVersionHash: graphTwo.contentHash!, runId: "run-two", resultHash: "result-two",
    })).toBe(true);
    expect(scheduled).toEqual([graphOne.id, graphTwo.id]);

    deferred[0].resolve(buildCommunityOverlay(graphOne));
    deferred[1].resolve(buildCommunityOverlay(graphTwo));
    await Promise.resolve();
    await Promise.resolve();
    expect(state.activeOverlay).toMatchObject({ graphVersionId: graphTwo.id, kind: "community" });
    expect(errors).toEqual([]);
  });

  it("keys Global/Louvain work by graph hash, run id, and result hash", async () => {
    const graph = overlayTestGraph("identity.csv", "identity");
    const computeDefaultOverlay = vi.fn(async () => buildCommunityOverlay(graph));
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay,
      onOverlayChange: vi.fn(),
      onError: vi.fn(),
    });
    controller.activate(graph);
    const first = {
      protocol: "global" as const,
      status: "succeeded" as const,
      graphVersionHash: graph.contentHash!,
      runId: "run-a",
      resultHash: "result-a",
    };

    expect(controller.acceptGlobalResult(graph, first)).toBe(true);
    await Promise.resolve(); await Promise.resolve();
    expect(controller.acceptGlobalResult(graph, { ...first, runId: "run-b", resultHash: "result-b" })).toBe(true);
    await Promise.resolve(); await Promise.resolve();

    expect(computeDefaultOverlay).toHaveBeenCalledTimes(2);
  });

  it("reattaches pending Louvain work after graph reactivation and publishes only its exact result", async () => {
    const graph = overlayTestGraph("pending.csv", "pending");
    const other = overlayTestGraph("other.csv", "other");
    const community = buildCommunityOverlay(graph);
    let resolvePending: ((overlay: AnalysisOverlay | null) => void) | undefined;
    let activeOverlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay: vi.fn(() => new Promise<AnalysisOverlay | null>((resolve) => { resolvePending = resolve; })),
      onOverlayChange: (overlay) => { activeOverlay = overlay; },
      onError: vi.fn(),
    });
    const identity = {
      protocol: "global" as const,
      status: "succeeded" as const,
      graphVersionHash: graph.contentHash!,
      runId: "run-pending",
      resultHash: "result-pending",
    };

    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, identity)).toBe(true);
    controller.activate(other);
    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, identity)).toBe(false);
    resolvePending?.(community);
    await Promise.resolve(); await Promise.resolve();

    expect(activeOverlay).toBe(community);
  });

  it("never publishes an older same-graph Louvain result after a newer result identity is accepted", async () => {
    const graph = overlayTestGraph("stale-result.csv", "stale-result");
    const older = { ...buildCommunityOverlay(graph), id: "older-overlay" };
    const newer = { ...buildCommunityOverlay(graph), id: "newer-overlay" };
    const pending: Array<(overlay: AnalysisOverlay | null) => void> = [];
    let activeOverlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay: vi.fn(() => new Promise<AnalysisOverlay | null>((resolve) => pending.push(resolve))),
      onOverlayChange: (overlay) => { activeOverlay = overlay; },
      onError: vi.fn(),
    });
    const base = {
      protocol: "global" as const,
      status: "succeeded" as const,
      graphVersionHash: graph.contentHash!,
    };

    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, { ...base, runId: "run-old", resultHash: "result-old" })).toBe(true);
    expect(controller.acceptGlobalResult(graph, { ...base, runId: "run-new", resultHash: "result-new" })).toBe(true);
    pending[0]?.(older);
    await Promise.resolve(); await Promise.resolve();
    expect(activeOverlay).toMatchObject({ kind: "raw" });
    pending[1]?.(newer);
    await Promise.resolve(); await Promise.resolve();
    expect(activeOverlay).toBe(newer);
  });

  it("does not republish a cached older Louvain identity after a newer success", async () => {
    const graph = overlayTestGraph("cached-redelivery.csv", "cached-redelivery");
    const oldOverlay = { ...buildCommunityOverlay(graph), id: "cached-old" };
    const newOverlay = { ...buildCommunityOverlay(graph), id: "cached-new" };
    const computeDefaultOverlay = vi.fn()
      .mockResolvedValueOnce(oldOverlay)
      .mockResolvedValueOnce(newOverlay);
    let activeOverlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay,
      onOverlayChange: (overlay) => { activeOverlay = overlay; },
      onError: vi.fn(),
    });
    const base = { protocol: "global" as const, status: "succeeded" as const, graphVersionHash: graph.contentHash! };
    const oldIdentity = { ...base, runId: "run-old-cached", resultHash: "result-old-cached" };
    const newIdentity = { ...base, runId: "run-new-cached", resultHash: "result-new-cached" };

    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, oldIdentity)).toBe(true);
    await Promise.resolve(); await Promise.resolve();
    expect(controller.acceptGlobalResult(graph, newIdentity)).toBe(true);
    await Promise.resolve(); await Promise.resolve();
    expect(activeOverlay).toBe(newOverlay);

    expect(controller.acceptGlobalResult(graph, oldIdentity)).toBe(false);
    expect(activeOverlay).toBe(newOverlay);
  });

  it("selects a camera only by exact workspace, current graph, and current lens", () => {
    const select = (appPresentation as typeof appPresentation & {
      resolveWorkspaceCameraSnapshot?: (
        cache: GraphCameraSnapshotCache,
        key: GraphCameraSnapshotCacheKey | null,
      ) => ReturnType<GraphCameraSnapshotCache["get"]>;
    }).resolveWorkspaceCameraSnapshot;
    const camera = (sceneIdentity: string, x: number, y = 20, zoom = 1) => ({
      schemaVersion: "socialgraph-fm.graph-camera/2" as const,
      sceneIdentity,
      position: [x, y] as [number, number],
      zoom,
      worldCenter: [30, 40] as [number, number],
      viewportSize: [800, 600] as [number, number],
    });
    const cache = new GraphCameraSnapshotCache();
    cache.set({ workspace: "governance", graphIdentity: "graph-a", lens: "risk" }, camera("a-risk", 10));
    cache.set({ workspace: "governance", graphIdentity: "graph-b", lens: "risk" }, camera("b-risk", 20));
    cache.set({ workspace: "governance", graphIdentity: "graph-b", lens: "relations" }, camera("b-relations", 30));
    cache.set({ workspace: "adaptation", graphIdentity: "shared-graph", lens: appPresentation.adaptationCameraLens("zero_shot") }, camera("shared-graph", 40, 41, 1.4));
    cache.set({ workspace: "adaptation", graphIdentity: "shared-graph", lens: appPresentation.adaptationCameraLens("few_shot") }, camera("shared-graph", 50, 51, 0.8));

    expect(select).toBeTypeOf("function");
    if (!select) return;
    expect(select(cache, { workspace: "governance", graphIdentity: "graph-b", lens: "relations" })?.sceneIdentity).toBe("b-relations");
    expect(select(cache, { workspace: "governance", graphIdentity: "graph-b", lens: "community" })).toBeUndefined();
    expect(select(cache, { workspace: "governance", graphIdentity: "graph-c", lens: "risk" })).toBeUndefined();
    expect(select(cache, { workspace: "adaptation", graphIdentity: "shared-graph", lens: "adaptation:zero_shot" })).toMatchObject({ position: [40, 41], zoom: 1.4 });
    expect(select(cache, { workspace: "adaptation", graphIdentity: "shared-graph", lens: "adaptation:few_shot" })).toMatchObject({ position: [50, 51], zoom: 0.8 });
    expect(select(cache, null)).toBeUndefined();
  });

  it("keeps a completed planning entry and appends its final report only once", () => {
    const complete = (appPresentation as typeof appPresentation & {
      completeConfirmedPlanningMessage?: (
        entries: readonly appPresentation.ChatEntry[],
        planningMessageId: string,
        report?: appPresentation.ChatEntry,
      ) => appPresentation.ChatEntry[];
    }).completeConfirmedPlanningMessage;
    const planning: appPresentation.ChatEntry = {
      id: "planning", role: "assistant", text: "分析计划", timestamp: "10:00", state: "warning",
      confirmation: { token: "token", action: "run_governance_analysis", requestDigest: "digest", expiresAt: "later" },
    };
    const report: appPresentation.ChatEntry = {
      id: "report", role: "assistant", text: "最终报告", timestamp: "10:01", state: "success", governanceRunId: "run-1",
    };

    expect(complete).toBeTypeOf("function");
    if (!complete) return;
    const running = appPresentation.updateGovernancePlanningProgress([planning], planning.id, {
      stage: "deriving",
      progress: 78,
    });
    expect(running[0]).toMatchObject({
      state: "working",
      confirmation: undefined,
      governanceProgress: { stage: "deriving", progress: 78 },
    });
    expect(running[0]?.text).toContain("整理候选、群组与关系");
    const reporting = appPresentation.updateGovernancePlanningProgress(running, planning.id, {
      stage: "reporting",
      progress: 95,
    });
    expect(reporting[0]).toMatchObject({
      state: "working",
      governanceProgress: { stage: "reporting", progress: 95 },
    });
    expect(reporting[0]?.text).toContain("正在整理研判结论");
    const once = complete([planning], planning.id, report);
    const twice = complete(once, planning.id, { ...report, id: "report-duplicate" });
    expect(twice.map((entry) => entry.id)).toEqual(["planning", "report"]);
    expect(twice[0]).toMatchObject({
      state: "success",
      confirmation: undefined,
      governanceProgress: { stage: "completed", progress: 100 },
    });
  });

  it("never presents the backend terminal state as 100% before the report is ready", () => {
    expect(appPresentation.presentGovernanceRunProgress({
      stage: "freezing",
      progress: 99,
      status: "running",
    })).toEqual({ stage: "freezing", progress: 95 });
    expect(appPresentation.presentGovernanceRunProgress({
      stage: "completed",
      progress: 100,
      status: "succeeded",
    })).toEqual({ stage: "reporting", progress: 95 });
  });

  it("restores an unanalyzed graph in raw-facts state without a community worker", () => {
    const graph = overlayTestGraph("restored.csv", "restored");
    const computeDefaultOverlay = vi.fn(async () => buildCommunityOverlay(graph));
    let overlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay,
      onOverlayChange: (next) => { overlay = next; },
      onError: vi.fn(),
    });
    controller.activate(graph);
    controller.deactivate();
    controller.activate(graph);
    expect(computeDefaultOverlay).not.toHaveBeenCalled();
    expect(overlay).toMatchObject({ graphVersionId: graph.id, kind: "raw" });
  });

  it("reactivates a graph with the exact cached Louvain result without scheduling another worker", async () => {
    const graph = overlayTestGraph("cached.csv", "cached");
    const community = buildCommunityOverlay(graph);
    const computeDefaultOverlay = vi.fn(async () => community);
    let overlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay,
      onOverlayChange: (next) => { overlay = next; },
      onError: vi.fn(),
    });
    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, {
      protocol: "global", status: "succeeded", graphVersionHash: graph.contentHash!, runId: "run-cached", resultHash: "result-cached",
    })).toBe(true);
    await Promise.resolve();
    await Promise.resolve();
    expect(overlay).toBe(community);
    controller.deactivate();
    controller.activate(graph);
    expect(overlay).toBe(community);
    expect(computeDefaultOverlay).toHaveBeenCalledTimes(1);
  });

  it("allows the same identity-matched Global result to retry after a Louvain error", async () => {
    const graph = overlayTestGraph("retry-error.csv", "retry-error");
    const community = buildCommunityOverlay(graph);
    const computeDefaultOverlay = vi.fn()
      .mockRejectedValueOnce(new Error("worker unavailable"))
      .mockResolvedValueOnce(community);
    const onError = vi.fn();
    let overlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay,
      onOverlayChange: (next) => { overlay = next; },
      onError,
    });
    const result = { protocol: "global" as const, status: "succeeded" as const, graphVersionHash: graph.contentHash!, runId: "run-retry", resultHash: "result-retry" };

    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, result)).toBe(true);
    await Promise.resolve(); await Promise.resolve();
    expect(onError).toHaveBeenCalledOnce();
    expect(overlay).toMatchObject({ kind: "raw" });
    expect(controller.acceptGlobalResult(graph, result)).toBe(true);
    await Promise.resolve(); await Promise.resolve();
    expect(computeDefaultOverlay).toHaveBeenCalledTimes(2);
    expect(overlay).toBe(community);
  });

  it("reattaches an in-flight result across reactivation and publishes only while current", async () => {
    const graph = overlayTestGraph("retry-stale.csv", "retry-stale");
    const other = overlayTestGraph("retry-other.csv", "retry-other");
    let resolveStale: ((overlay: AnalysisOverlay | null) => void) | undefined;
    const community = buildCommunityOverlay(graph);
    const computeDefaultOverlay = vi.fn()
      .mockImplementationOnce(() => new Promise<AnalysisOverlay | null>((resolve) => { resolveStale = resolve; }))
      .mockResolvedValueOnce(community);
    let overlay: AnalysisOverlay | null = null;
    const controller = new appPresentation.GraphVersionOverlayController({
      computeDefaultOverlay,
      onOverlayChange: (next) => { overlay = next; },
      onError: vi.fn(),
    });
    const result = { protocol: "global" as const, status: "succeeded" as const, graphVersionHash: graph.contentHash!, runId: "run-pending", resultHash: "result-pending" };

    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, result)).toBe(true);
    controller.activate(other);
    resolveStale?.(community);
    await Promise.resolve(); await Promise.resolve();
    expect(overlay).toMatchObject({ graphVersionId: other.id, kind: "raw" });

    controller.activate(graph);
    expect(controller.acceptGlobalResult(graph, result)).toBe(false);
    expect(computeDefaultOverlay).toHaveBeenCalledTimes(1);
    expect(overlay).toBe(community);
  });

  it("clears an explicit path overlay back to the cached Louvain overlay for only the active graph", async () => {
    const OverlayController = appPresentation.GraphVersionOverlayController;
    const state: { activeOverlay: AnalysisOverlay | null } = { activeOverlay: null };
    let resolveDefault: ((overlay: AnalysisOverlay | null) => void) | undefined;

    expect(OverlayController).toBeTypeOf("function");
    const controller = new OverlayController({
      computeDefaultOverlay: () => new Promise((resolve) => { resolveDefault = resolve; }),
      onOverlayChange: (overlay) => { state.activeOverlay = overlay; },
      onError: () => undefined,
    });
    const graph = overlayTestGraph("graph-clear.json", "one");
    const defaultOverlay = buildCommunityOverlay(graph);
    const pathOverlay = buildPathOverlay(graph, {
      sourceId: "one",
      targetId: "one",
      nodes: graph.nodes,
      edges: [],
      nodeIds: ["one"],
      edgeIds: [],
    });
    controller.activate(graph, null);
    (controller as typeof controller & { acceptGlobalResult?: (version: GraphVersion, result: { protocol: "global"; status: "succeeded"; graphVersionHash: string; runId: string; resultHash: string }) => boolean })
      .acceptGlobalResult?.(graph, { protocol: "global", status: "succeeded", graphVersionHash: graph.contentHash!, runId: "run-clear", resultHash: "result-clear" });
    resolveDefault?.(defaultOverlay);
    await Promise.resolve();
    await Promise.resolve();
    controller.setExplicit(graph.id, pathOverlay);
    expect(state.activeOverlay).toBe(pathOverlay);

    controller.clearExplicit(graph.id);
    expect(state.activeOverlay).toBe(defaultOverlay);
    controller.clearExplicit("stale-graph");
    expect(state.activeOverlay).toBe(defaultOverlay);
  });

  it("locates candidates through the real graph reducer and applies pane, mobile, and feedback effects", () => {
    const locateCandidates = (appPresentation as typeof appPresentation & {
      locateGovernanceCandidates?: (
        input: unknown,
        effects: {
          applyGraphAction: (action: GraphViewAction) => void;
          expandGraph: () => void;
          switchMobilePanel: (panel: "graph") => void;
          notify: (message: string) => void;
          saveOverview?: () => void;
        },
      ) => { status: string; nodeIds: readonly string[] };
    }).locateGovernanceCandidates;
    let graphState = createGraphWorkbenchViewState(createDefaultGraphViewState("graph-1"));
    let expanded = false;
    let mobilePanel = "chat";
    let notice = "";
    let overviewSaved = false;
    const effects = {
      applyGraphAction: (action: GraphViewAction) => { graphState = reduceGraphView(graphState, action); },
      expandGraph: () => { expanded = true; },
      switchMobilePanel: (panel: "graph") => { mobilePanel = panel; },
      notify: (message: string) => { notice = message; },
      saveOverview: () => { overviewSaved = true; },
    };
    const readyInput = {
      messageRunId: "run-1",
      currentRunId: "run-1",
      runGraphVersionHash: "hash-1",
      currentGraphVersionHash: "hash-1",
      previewNodes: [{ id: "second", score: 0.7 }, { id: "first", score: 0.9 }],
      graphNodeIds: ["first", "second"],
    };

    expect(locateCandidates).toBeTypeOf("function");
    if (!locateCandidates) return;
    expect(locateCandidates(readyInput, effects)).toEqual({ status: "ready", nodeIds: ["first", "second"] });
    expect(graphState.viewState).toMatchObject({ mode: "local", focusNodeIds: ["first", "second"] });
    expect(graphState.interaction.selectedNodeId).toBe("first");
    expect(expanded).toBe(true);
    expect(mobilePanel).toBe("graph");
    expect(notice).toBe("已定位 2 个重点候选");
    expect(overviewSaved).toBe(true);
    const focusedCamera = graphState.viewState.camera;
    graphState = reduceGraphView(graphState, { type: "activate_mode", mode: "global" });
    expect(graphState.viewState).toMatchObject({ mode: "global", focusNodeIds: [], camera: focusedCamera });

    expanded = false;
    notice = "";
    expect(locateCandidates({ ...readyInput, messageRunId: "stale" }, effects).status).toBe("stale");
    expect(expanded).toBe(false);
    expect(notice).toBe("该候选报告不属于当前图谱，请重新运行分析");

    notice = "";
    expect(locateCandidates({ ...readyInput, graphNodeIds: [] }, effects).status).toBe("empty");
    expect(expanded).toBe(false);
    expect(notice).toBe("当前图谱预览中没有可定位的重点候选");
  });

  it("never exposes internal package or geography labels as the public data source", () => {
    const publicGraphSourceLabel = (appPresentation as typeof appPresentation & {
      publicGraphSourceLabel?: (graph: { sourceFile: string; datasetArtifact?: { datasetName: string } }) => string;
    }).publicGraphSourceLabel;

    expect(publicGraphSourceLabel).toBeTypeOf("function");
    expect(publicGraphSourceLabel!({
      sourceFile: `governance-artifact-${"7".repeat(32)}`,
      datasetArtifact: { datasetName: `governance-artifact-${"7".repeat(32)}` },
    })).toBe("当前会话治理图");
    expect(publicGraphSourceLabel!({
      sourceFile: `governance-artifact-${"7".repeat(32)}`,
      datasetArtifact: { datasetName: `SocialGraph-FM Governance · governance-artifact-${"7".repeat(32)}` },
    })).toBe("当前会话治理图");
    expect(publicGraphSourceLabel!({
      sourceFile: "russia-04.zip",
      datasetArtifact: { datasetName: "Russia answer pack 04" },
    })).toBe("当前会话治理图");
    expect(publicGraphSourceLabel!({ sourceFile: "community-network.csv" })).toBe("community-network.csv");
  });

  it("describes a local analysis scope by counts without exposing its scope hash", () => {
    const describeResult = (appPresentation as typeof appPresentation & {
      resultDescription?: (run?: AnalysisRun) => string | null;
    }).resultDescription;

    expect(describeResult).toBeTypeOf("function");
    const description = describeResult!(localOverviewRun());

    expect(description).toContain("本次范围 12 个节点、19 条关系。");
    expect(description).not.toContain("sensitive-scope-hash");
    expect(description).not.toContain("scope");
  });

  it("appends exactly three human-review bullets to a successful local analysis result", () => {
    const buildResultMarkdown = (appPresentation as typeof appPresentation & {
      buildAnalysisResultMarkdown?: (run?: AnalysisRun) => string | null;
    }).buildAnalysisResultMarkdown;

    expect(buildResultMarkdown).toBeTypeOf("function");
    const markdown = buildResultMarkdown!(localOverviewRun())!;

    expect(markdown).toContain("### 分析结果");
    expect(markdown).toContain("### 人工复核建议");
    expect(markdown.match(/^### 人工复核建议$/gmu)).toHaveLength(1);
    expect(markdown.match(/^- /gmu)).toHaveLength(3);
    expect(markdown).toContain("- 选择候选");
    expect(markdown).toContain("- 核对关系与邻域");
    expect(markdown).toContain("- 加入研判单并记录确认、驳回或待定理由");
  });

  it("also appends human-review guidance to a successful GFM research result", () => {
    const buildResultMarkdown = (appPresentation as typeof appPresentation & {
      buildAnalysisResultMarkdown?: (run?: AnalysisRun) => string | null;
    }).buildAnalysisResultMarkdown;
    const coreRun: AnalysisRun = { ...localOverviewRun(), id: "run-gfm-overview", engine: "gfm" };

    expect(buildResultMarkdown).toBeTypeOf("function");
    const markdown = buildResultMarkdown!(coreRun)!;

    expect(markdown.match(/^### 人工复核建议$/gmu)).toHaveLength(1);
    expect(markdown.match(/^- /gmu)).toHaveLength(3);
  });

  it("does not duplicate human-review guidance already returned by the governance API", () => {
    const ensureGuidance = (appPresentation as typeof appPresentation & {
      ensureHumanReviewGuidance?: (markdown: string) => string;
    }).ensureHumanReviewGuidance;
    const apiAnswer = [
      "### 治理结论",
      "",
      "建议优先复核高风险候选。",
      "",
      "### 人工复核建议",
      "",
      "- 选择候选",
      "- 核对关系与邻域",
      "- 加入研判单并记录确认、驳回或待定理由",
    ].join("\n");

    expect(ensureGuidance).toBeTypeOf("function");
    const markdown = ensureGuidance!(apiAnswer);

    expect(markdown).toBe(apiAnswer);
    expect(markdown.match(/^### 人工复核建议$/gmu)).toHaveLength(1);
    expect(markdown.match(/^- /gmu)).toHaveLength(3);
  });

  it("recognizes the governance review section at any Markdown heading level", () => {
    const ensureGuidance = (appPresentation as typeof appPresentation & {
      ensureHumanReviewGuidance?: (markdown: string) => string;
    }).ensureHumanReviewGuidance;
    const apiAnswer = [
      "## 治理摘要",
      "",
      "已形成复核顺序。",
      "",
      "## 人工复核建议",
      "",
      "- 选择候选",
      "- 核对关系与邻域",
      "- 加入研判单并记录理由",
    ].join("\n");

    expect(ensureGuidance!(apiAnswer)).toBe(apiAnswer);
    expect(ensureGuidance!(apiAnswer).match(/人工复核建议/gmu)).toHaveLength(1);
  });

  it("shows completed activity only for successful analysis or governance report messages", () => {
    const activityForEntry = (appPresentation as typeof appPresentation & {
      assistantActivityForEntry?: (entry: unknown) => { kind: string; state: string } | null;
    }).assistantActivityForEntry;

    expect(activityForEntry).toBeTypeOf("function");
    expect(activityForEntry!({
      id: "working",
      role: "assistant",
      text: "正在分析",
      timestamp: "10:00",
      state: "working",
      activity: { kind: "graph_analysis", state: "working" },
    })).toEqual({ kind: "graph_analysis", state: "working" });
    expect(activityForEntry!({
      id: "local-result",
      role: "assistant",
      text: "分析完成",
      timestamp: "10:01",
      state: "success",
      run: localOverviewRun(),
    })).toEqual({ kind: "graph_analysis", state: "completed" });
    expect(activityForEntry!({
      id: "governance-report",
      role: "assistant",
      text: "治理报告",
      timestamp: "10:02",
      state: "success",
      governanceRunId: "governance-run-1",
    })).toEqual({ kind: "governance", state: "completed" });
    expect(activityForEntry!({
      id: "ordinary-notice",
      role: "assistant",
      text: "文件已保存",
      timestamp: "10:03",
      state: "success",
      activity: { kind: "graph_import", state: "completed" },
    })).toBeNull();
    expect(activityForEntry!({
      id: "warning-result",
      role: "assistant",
      text: "请确认字段",
      timestamp: "10:04",
      state: "warning",
      run: localOverviewRun(),
    })).toBeNull();
    expect(activityForEntry!({
      id: "failed-governance",
      role: "assistant",
      text: "治理运行失败",
      timestamp: "10:05",
      state: "error",
      governanceRunId: "governance-run-1",
    })).toBeNull();
  });

  it("offers review only on the completed report for the currently active governance run", () => {
    const canOpenReview = (appPresentation as typeof appPresentation & {
      canOpenGovernanceReview?: (entry: unknown, activeRunId?: string) => boolean;
    }).canOpenGovernanceReview;
    const report = {
      id: "governance-report",
      role: "assistant",
      text: "治理报告",
      timestamp: "10:02",
      state: "success",
      governanceRunId: "governance-run-1",
    };

    expect(canOpenReview).toBeTypeOf("function");
    expect(canOpenReview!(report, "governance-run-1")).toBe(true);
    expect(canOpenReview!(report, "governance-run-2")).toBe(false);
    expect(canOpenReview!({ ...report, governanceRunId: undefined }, "governance-run-1")).toBe(false);
    expect(canOpenReview!({ ...report, state: "warning" }, "governance-run-1")).toBe(false);
  });

  it("uses business-facing copy for graph review and package compatibility notices", () => {
    const copy = (appPresentation as typeof appPresentation & {
      ORDINARY_PRESENTATION_COPY?: Readonly<Record<string, string>>;
    }).ORDINARY_PRESENTATION_COPY;

    expect(copy).toBeDefined();
    expect(copy).toMatchObject({
      graphReviewDetails: "字段与质量信息",
      governanceCompatibilityError: "推理包未通过兼容性检查。",
      governanceReady: "推理包已通过兼容性检查。输入“开始分析”后，系统将梳理全图风险顺序，整理风险账号、协同群组和重点关系；完成后可进入治理应用核对证据。",
    });
    expect(Object.values(copy!).join(" ")).not.toMatch(/输入合同|技术详情|GraphVersion|SourceArtifact/u);
  });

  it("persists and restores the governance run marker only for completed report messages", () => {
    const persistenceRunId = (appPresentation as typeof appPresentation & {
      governanceRunIdForPersistence?: (entry: unknown) => string | undefined;
    }).governanceRunIdForPersistence;
    const restoredRunId = (appPresentation as typeof appPresentation & {
      governanceRunIdFromStoredMessage?: (message: unknown) => string | undefined;
    }).governanceRunIdFromStoredMessage;
    const report = {
      id: "governance-report",
      role: "assistant",
      text: "治理报告",
      timestamp: "10:02",
      state: "success",
      governanceRunId: "governance-run-1",
    };

    expect(persistenceRunId).toBeTypeOf("function");
    expect(restoredRunId).toBeTypeOf("function");
    expect(persistenceRunId!(report)).toBe("governance-run-1");
    expect(persistenceRunId!({ ...report, state: "warning" })).toBeUndefined();
    expect(persistenceRunId!({ ...report, governanceRunId: undefined })).toBeUndefined();
    expect(restoredRunId!({
      id: "stored-governance-report",
      sessionId: "session-1",
      role: "assistant",
      text: "治理报告",
      status: "completed",
      governanceRunId: "governance-run-1",
      createdAt: "2026-08-20T00:00:00.000Z",
    })).toBe("governance-run-1");
    expect(restoredRunId!({
      id: "stored-warning",
      sessionId: "session-1",
      role: "assistant",
      text: "治理报告",
      status: "warning",
      governanceRunId: "governance-run-1",
      createdAt: "2026-08-20T00:00:00.000Z",
    })).toBeUndefined();
  });
});
