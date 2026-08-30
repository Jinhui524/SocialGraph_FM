import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { casePayload } from "../../test/fixtures/governanceOnline";
import {
  adaptationHandoff,
  targetCapabilities,
  targetActivation,
  targetComparison,
  targetEvidence,
  targetPolicy,
  targetPreview,
  targetResult,
  targetRun,
  targetReviewCollection,
  targetTaskRegistration,
} from "../../test/fixtures/governanceTargetTask";
import type { GovernanceOnlineClientLike } from "../../types/governanceOnline";
import type { AnalysisOverlay, GraphVersion } from "../../types/graph";
import { sha256Canonical } from "../../services/graphIdentity";
import { AdaptationWorkspace } from "./AdaptationWorkspace";

afterEach(cleanup);

function client(overrides: Partial<GovernanceOnlineClientLike> = {}): GovernanceOnlineClientLike {
  const registration = targetTaskRegistration();
  const zeroRegistration = targetTaskRegistration("zero_shot");
  const policy = targetPolicy();
  const modeForArtifact = (artifactId: string) => artifactId === zeroRegistration.artifact.artifactId ? "zero_shot" as const : "few_shot" as const;
  const modeForRun = (runId: string) => runId === targetRun("zero_shot").runId ? "zero_shot" as const : "few_shot" as const;
  return {
    capabilities: vi.fn().mockResolvedValue(targetCapabilities()),
    registerTargetTask: vi.fn((file: File) => Promise.resolve(file.name.includes("zero") || file.name.includes("regional-a") ? zeroRegistration : registration)),
    preview: vi.fn((artifactId: string) => Promise.resolve(targetPreview(false, modeForArtifact(artifactId)))),
    createRun: vi.fn((request: { artifactId: string }) => Promise.resolve(targetRun(modeForArtifact(request.artifactId)))),
    run: vi.fn((runId: string) => Promise.resolve(targetRun(modeForRun(runId)))),
    result: vi.fn((runId: string) => Promise.resolve(targetResult(modeForRun(runId)))),
    runPreview: vi.fn((runId: string) => Promise.resolve(targetPreview(true, modeForRun(runId)))),
    evidence: vi.fn((runId: string, nodeId: string) => Promise.resolve(targetEvidence(nodeId, modeForRun(runId)))),
    createTargetLabelSet: vi.fn().mockResolvedValue(registration.labels),
    fitTargetPolicy: vi.fn().mockResolvedValue(policy),
    targetPolicy: vi.fn().mockResolvedValue(policy),
    targetComparison: vi.fn().mockResolvedValue(targetComparison()),
    activateTargetPolicy: vi.fn().mockResolvedValue(targetActivation()),
    createAdaptationHandoff: vi.fn().mockResolvedValue(adaptationHandoff()),
    createTargetReviewCollection: vi.fn((request) => Promise.resolve(targetReviewCollection(request))),
    ...overrides,
  } as unknown as GovernanceOnlineClientLike;
}

function upload(label: string, name: string): void {
  fireEvent.change(screen.getByLabelText(label), { target: { files: [new File([`PK ${name}`], name, { type: "application/zip" })] } });
}

async function confirmRun(lane: HTMLElement): Promise<void> {
  fireEvent.click(within(lane).getByRole("button", { name: "开始分析" }));
  fireEvent.click(within(lane).getByRole("button", { name: "确认分析" }));
  await within(lane).findByText(/协同组群已就绪/u);
}

async function waitForFewShotOrder(lane: HTMLElement): Promise<void> {
  await waitFor(() => expect(lane).toHaveAttribute("data-phase", "compared"));
}

describe("independent target-task adaptation lanes", () => {
  it("frames both lanes with a concise guide and focuses a selected path", () => {
    render(<AdaptationWorkspace client={client()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);

    const guide = screen.getByRole("navigation", { name: "选择适配路径" });
    expect(screen.getByRole("heading", { name: "面向新网络的风险迁移" })).toBeVisible();
    expect(within(guide).getAllByRole("button")).toHaveLength(2);
    expect(screen.queryByText("登记目标域")).not.toBeInTheDocument();
    const zero = screen.getByRole("region", { name: "零样本路径" });
    const few = screen.getByRole("region", { name: "少样本路径" });
    expect(zero).toBeVisible();
    expect(few).toBeVisible();
    fireEvent.click(within(guide).getByRole("button", { name: /跨域新活动 · 零样本/u }));
    expect(document.activeElement).toBe(zero);
    expect(within(zero).getByRole("button", { name: "进入治理应用" })).toBeDisabled();
    expect(within(zero).getByText("完成网络分析后可进入治理应用")).toBeVisible();
  });

  it("publishes imported few-shot labels as a presentation-only graph overlay", async () => {
    const onGraphChange = vi.fn();
    const onOverlayChange = vi.fn();
    const onLanePresentationChange = vi.fn();
    const onActiveLaneChange = vi.fn();
    render(<AdaptationWorkspace
      client={client()}
      onGraphChange={onGraphChange}
      onOverlayChange={onOverlayChange}
      onLanePresentationChange={onLanePresentationChange}
      onActiveLaneChange={onActiveLaneChange}
      onClose={vi.fn()}
    />);

    upload("少样本目标任务包", "few-labels.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "少样本路径" });
    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw"));
    expect(within(lane).queryByText(/正向|负向/u)).not.toBeInTheDocument();

    const overlay = onOverlayChange.mock.lastCall?.[0] as AnalysisOverlay;
    const graphValue = onGraphChange.mock.lastCall?.[1] as GraphVersion;
    expect(lane).toHaveAttribute("data-overlay", "labels");
    expect(overlay).toMatchObject({
      graphVersionId: graphValue.id,
      kind: "governance",
      legend: { title: "已知标签" },
      provenance: {
        engine: "imported-label-set",
        scopeHash: targetTaskRegistration().labels?.labelSetHash,
      },
    });
    expect(Object.values(overlay.nodeValues).filter((value) => value === "reference-positive")).toHaveLength(8);
    expect(Object.values(overlay.nodeValues).filter((value) => value === "reference-negative")).toHaveLength(8);
    expect(Object.values(overlay.presentation?.referenceLabels ?? {}).filter((value) => value === "positive")).toHaveLength(8);
    expect(Object.values(overlay.presentation?.referenceLabels ?? {}).filter((value) => value === "negative")).toHaveLength(8);
    expect(overlay.presentation?.referenceLabels?.["target-node-001"]).toBe("positive");
    expect(overlay.presentation?.referenceLabels?.["target-node-009"]).toBe("negative");
    expect(overlay.presentation?.riskBands).toBeUndefined();
    expect(graphValue.nodes.every((node) => !("fewShotLabel" in node.attributes))).toBe(true);
    expect(onLanePresentationChange).toHaveBeenCalledWith("few_shot", expect.objectContaining({
      graph: graphValue,
      overlay,
    }));
    expect(onActiveLaneChange).toHaveBeenLastCalledWith("few_shot");
    const graphPublicationIndex = onLanePresentationChange.mock.calls.findIndex(([, patch]) => patch.graph === graphValue);
    expect(onLanePresentationChange.mock.invocationCallOrder[graphPublicationIndex]).toBeLessThan(onActiveLaneChange.mock.invocationCallOrder.at(-1) ?? 0);
    expect(within(lane).getByRole("button", { name: "开始分析" })).toBeVisible();
    expect(within(lane).queryByText(/Global 基线/u)).not.toBeInTheDocument();
  });

  it("runs a confirmed zero-shot journey over the complete graph and hands candidates to governance explicitly", async () => {
    const api = client({ registerTargetTask: vi.fn().mockResolvedValue(targetTaskRegistration("zero_shot")) });
    const onGraphChange = vi.fn(); const onOverlayChange = vi.fn(); const onFocusChange = vi.fn(); const onCameraFocusChange = vi.fn(); const onGovernanceHandoff = vi.fn();
    render(<AdaptationWorkspace client={api} onGraphChange={onGraphChange} onOverlayChange={onOverlayChange} onFocusChange={onFocusChange} onCameraFocusChange={onCameraFocusChange} onGovernanceHandoff={onGovernanceHandoff} onClose={vi.fn()} />);
    upload("零样本目标任务包", "regional-a.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "零样本路径" });
    await within(lane).findByText("108 个对象 · 220 条关系");
    expect(onGraphChange).toHaveBeenLastCalledWith("zero_shot", expect.objectContaining({ nodes: expect.arrayContaining([expect.objectContaining({ id: "target-node-108" })]) }));
    expect(within(lane).queryByLabelText("重点账号")).not.toBeInTheDocument();
    await confirmRun(lane);
    expect(within(lane).getByText("协同组群已就绪 · 108 个账号")).toBeVisible();
    expect(onOverlayChange.mock.lastCall?.[0]).toMatchObject({ kind: "community", legend: { title: "协同组群" }, nodeValues: { "target-node-001": "group-0" } });
    const candidateList = within(lane).getByLabelText("重点账号");
    expect(within(candidateList).getAllByRole("button", { name: /待治理核验/u })).toHaveLength(25);
    expect(candidateList).not.toHaveTextContent("#1");
    fireEvent.click(within(candidateList).getByRole("button", { name: "对象 1，待治理核验" }));
    await within(lane).findByText("直接关系证据");
    expect(onFocusChange).toHaveBeenLastCalledWith(expect.objectContaining({ targetId: "target-node-001" }));
    expect(onFocusChange.mock.lastCall?.[0].cameraToken).toBe(onCameraFocusChange.mock.lastCall?.[0].token);
    expect(api.createTargetReviewCollection).not.toHaveBeenCalled();
    fireEvent.click(within(lane).getByRole("button", { name: "进入治理应用" }));
    await waitFor(() => expect(onGovernanceHandoff).toHaveBeenCalledWith(expect.objectContaining({ lane: "zero_shot", snapshot: expect.objectContaining({ sessionId: targetTaskRegistration("zero_shot").registrationId, sourceFileName: "Regional review task A.sgtask.zip", activeCaseId: casePayload().caseId }) })));
  });

  it("keeps few-shot policy work in the background while presenting communities and the shared account list", async () => {
    const api = client(); const onFocusChange = vi.fn(); const onOverlayChange = vi.fn(); const onGovernanceHandoff = vi.fn();
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={onOverlayChange} onFocusChange={onFocusChange} onCameraFocusChange={vi.fn()} onGovernanceHandoff={onGovernanceHandoff} onClose={vi.fn()} />);
    upload("少样本目标任务包", "regional-b.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "少样本路径" });
    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw"));
    await confirmRun(lane);
    await waitForFewShotOrder(lane);
    expect(api.fitTargetPolicy).toHaveBeenCalledWith(
      targetTaskRegistration().labels?.labelSetHash,
      {
        schemaVersion: "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
        targetTaskRegistrationId: targetTaskRegistration().registrationId,
        runId: targetRun().runId,
        resultHash: targetResult().resultHash,
      },
      expect.any(AbortSignal),
    );
    const publishedOverlays = onOverlayChange.mock.calls.map(([overlay]) => overlay as AnalysisOverlay | null).filter(Boolean) as AnalysisOverlay[];
    const rawOverlay = publishedOverlays.find((overlay) => overlay.legend.title === "已知标签");
    const communityOverlay = publishedOverlays.find((overlay) => overlay.legend.title === "协同组群");
    for (const overlay of [rawOverlay, communityOverlay]) {
      expect(Object.values(overlay?.presentation?.referenceLabels ?? {}).filter((value) => value === "positive")).toHaveLength(8);
      expect(Object.values(overlay?.presentation?.referenceLabels ?? {}).filter((value) => value === "negative")).toHaveLength(8);
    }
    expect(communityOverlay).toMatchObject({ kind: "community", nodeValues: { "target-node-001": "group-0", "target-node-002": "group-1" } });
    expect(publishedOverlays.some((overlay) => overlay.legend.title.includes("风险排序") || overlay.legend.title.includes("复核顺序"))).toBe(false);
    expect(api.activateTargetPolicy).not.toHaveBeenCalled();
    expect(within(lane).queryByRole("table")).not.toBeInTheDocument();
    expect(within(lane).queryByRole("group", { name: "排序图层" })).not.toBeInTheDocument();
    expect(within(lane).queryByText(/正向 8|负向 8|少样本复核顺序已就绪/u)).not.toBeInTheDocument();
    expect(within(lane).queryByLabelText("迁移依据摘要")).not.toBeInTheDocument();
    expect(within(lane).getByRole("button", { name: /查看迁移依据/u })).toBeDisabled();
    const candidateList = within(lane).getByLabelText("重点账号");
    expect(within(candidateList).getAllByRole("button", { name: /待治理核验/u })).toHaveLength(25);
    expect(candidateList.textContent).not.toMatch(/#\d+/u);
    fireEvent.click(within(lane).getByRole("button", { name: "下一页" }));
    const pageTwo = within(lane).getByLabelText("重点账号");
    fireEvent.click(within(pageTwo).getByRole("button", { name: /对象 26.*待治理核验/u }));
    expect(onFocusChange).toHaveBeenLastCalledWith(expect.objectContaining({ targetId: "target-node-026" }));
    expect(api.createAdaptationHandoff).not.toHaveBeenCalled();
    fireEvent.click(within(lane).getByRole("button", { name: "进入治理应用" }));
    await waitFor(() => expect(onGovernanceHandoff).toHaveBeenCalledWith(expect.objectContaining({ lane: "few_shot", handoff: expect.objectContaining({ baseModelMutation: false }) })));
    const handoffTarget = onGovernanceHandoff.mock.lastCall?.[0];
    expect(Object.values(handoffTarget.adaptedOverlay.presentation.referenceLabels).filter((value) => value === "positive")).toHaveLength(8);
    expect(Object.values(handoffTarget.adaptedOverlay.presentation.referenceLabels).filter((value) => value === "negative")).toHaveLength(8);
  });

  it("surfaces an insufficient-signal policy without requesting a comparison or handoff", async () => {
    const insufficient = {
      ...targetPolicy(),
      status: "insufficient_signal" as const,
      selectedLambda: 0,
    };
    const api = client({ fitTargetPolicy: vi.fn().mockResolvedValue(insufficient) });
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("少样本目标任务包", "insufficient.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "少样本路径" });
    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw"));
    await confirmRun(lane);
    await within(lane).findByText("标签信号不足；当前少样本网络暂不能适配。请更换标签或目标任务包后重新登记，基础结果保持不变。");
    expect(api.targetPolicy).not.toHaveBeenCalled();
    expect(api.targetComparison).not.toHaveBeenCalled();
    expect(within(lane).getByRole("button", { name: "进入治理应用" })).toBeDisabled();
    expect(within(lane).getByText("当前标签信号不足，暂不能移交")).toBeVisible();
    expect(within(lane).queryByRole("button", { name: "重试适配" })).not.toBeInTheDocument();
    const warning = within(lane).getByRole("alert");
    expect(warning).toHaveTextContent("请更换标签或目标任务包后重新登记");
    expect(warning).toHaveClass("adaptation-workspace__notice");
    expect(warning).not.toHaveClass("is-error");
  });

  it("keeps handoff disabled after a transient organization failure and retries only the frozen ordering step", async () => {
    const fitTargetPolicy = vi.fn()
      .mockRejectedValueOnce(new Error("temporary unavailable"))
      .mockResolvedValueOnce(targetPolicy());
    const api = client({ fitTargetPolicy });
    const onOverlayChange = vi.fn();
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={onOverlayChange} onClose={vi.fn()} />);
    upload("少样本目标任务包", "retry-order.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "少样本路径" });
    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw"));
    await confirmRun(lane);

    await within(lane).findByText("少样本网络适配未完成；基础结果保持不变，可重试适配。");
    expect(within(lane).getByRole("button", { name: "进入治理应用" })).toBeDisabled();
    expect(within(lane).getByText("完成网络适配后可进入治理应用")).toBeVisible();
    expect(api.createRun).toHaveBeenCalledOnce();
    fireEvent.click(within(lane).getByRole("button", { name: "重试适配" }));
    await waitForFewShotOrder(lane);

    expect(fitTargetPolicy).toHaveBeenCalledTimes(2);
    expect(api.createRun).toHaveBeenCalledOnce();
    expect(within(lane).getByRole("button", { name: "进入治理应用" })).toBeEnabled();
    expect(Object.values((onOverlayChange.mock.lastCall?.[0] as AnalysisOverlay).presentation?.referenceLabels ?? {})).toHaveLength(16);
  });

  it("does not enable governance handoff until the exact comparison has resolved", async () => {
    let resolveComparison!: (value: ReturnType<typeof targetComparison>) => void;
    const comparison = new Promise<ReturnType<typeof targetComparison>>((resolve) => { resolveComparison = resolve; });
    const api = client({ targetComparison: vi.fn().mockReturnValue(comparison) });
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("少样本目标任务包", "deferred-order.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "少样本路径" });
    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw"));
    await confirmRun(lane);

    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "fitting"));
    expect(within(lane).getAllByText("正在完成少样本网络适配")).toHaveLength(2);
    expect(within(lane).getByRole("button", { name: "进入治理应用" })).toBeDisabled();
    expect(within(lane).queryByLabelText("风险排序候选")).not.toBeInTheDocument();
    resolveComparison(targetComparison());
    await waitForFewShotOrder(lane);
    expect(within(lane).getByRole("button", { name: "进入治理应用" })).toBeEnabled();
  });

  it("replaces one loaded lane without clearing the other lane presentation or policy state", async () => {
    const registerTargetTask = vi.fn().mockImplementation((file: File) => Promise.resolve(targetTaskRegistration(file.name.includes("zero") ? "zero_shot" : "few_shot")));
    const api = client({ registerTargetTask });
    const presentations: Record<string, Record<string, unknown>> = { zero_shot: {}, few_shot: {} };
    const onLanePresentationChange = vi.fn((lane: "zero_shot" | "few_shot", patch: object) => { presentations[lane] = { ...presentations[lane], ...patch }; });
    const onActiveLaneChange = vi.fn();
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onLanePresentationChange={onLanePresentationChange} onActiveLaneChange={onActiveLaneChange} onClose={vi.fn()} />);

    upload("零样本目标任务包", "zero-first.sgtask.zip");
    const zero = await screen.findByRole("region", { name: "零样本路径" }); await within(zero).findByText("108 个对象 · 220 条关系"); await confirmRun(zero);
    fireEvent.click(within(within(zero).getByLabelText("重点账号")).getByRole("button", { name: "对象 1，待治理核验" })); await within(zero).findByText("直接关系证据");

    upload("少样本目标任务包", "few.sgtask.zip");
    const few = screen.getByRole("region", { name: "少样本路径" }); await waitFor(() => expect(few).toHaveAttribute("data-phase", "raw")); await confirmRun(few);
    await waitForFewShotOrder(few);
    fireEvent.click(within(within(few).getByLabelText("重点账号")).getByRole("button", { name: "对象 1，待治理核验" }));
    const fewPresentation = presentations.few_shot; const fewPolicyEpoch = few.dataset.policyEpoch; const fewAbortEpoch = few.dataset.abortEpoch;
    expect(fewPresentation.graph).toBeTruthy(); expect(fewPresentation.focus).toMatchObject({ targetId: "target-node-001" });

    upload("零样本目标任务包", "zero-replacement.sgtask.zip");
    await within(zero).findByText("108 个对象 · 220 条关系");
    expect(presentations.few_shot).toBe(fewPresentation);
    expect(few.dataset.policyEpoch).toBe(fewPolicyEpoch); expect(few.dataset.abortEpoch).toBe(fewAbortEpoch);
    expect(within(few).queryByRole("table")).not.toBeInTheDocument();
    expect(within(within(few).getByLabelText("重点账号")).getByRole("button", { name: "对象 1，待治理核验" })).toHaveClass("is-selected");
    expect(presentations.few_shot.graph).toBe(fewPresentation.graph);
  });

  it("rejects foreign zero-shot review collections and few-shot handoffs", async () => {
    const collection = targetReviewCollection();
    const collectionLogical = { ...collection, targetTaskRegistrationId: `target-task-${"5".repeat(32)}` };
    const { collectionHash: _collectionHash, ...collectionWithoutHash } = collectionLogical;
    const foreignCollection = { ...collectionWithoutHash, collectionHash: sha256Canonical(collectionWithoutHash) };
    const handoff = adaptationHandoff();
    const handoffLogical = { ...handoff, targetReceiptHash: "6".repeat(64) };
    const { handoffHash: _handoffHash, ...handoffWithoutHash } = handoffLogical;
    const foreignHandoff = { ...handoffWithoutHash, handoffHash: sha256Canonical(handoffWithoutHash) };
    const onGovernanceHandoff = vi.fn();
    const api = client({
      registerTargetTask: vi.fn().mockImplementation((file: File) => Promise.resolve(targetTaskRegistration(file.name.includes("zero") ? "zero_shot" : "few_shot"))),
      createTargetReviewCollection: vi.fn().mockResolvedValue(foreignCollection),
      createAdaptationHandoff: vi.fn().mockResolvedValue(foreignHandoff),
    });
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onGovernanceHandoff={onGovernanceHandoff} onClose={vi.fn()} />);
    upload("零样本目标任务包", "zero-foreign.sgtask.zip");
    const zero = await screen.findByRole("region", { name: "零样本路径" }); await within(zero).findByText("108 个对象 · 220 条关系"); await confirmRun(zero);
    fireEvent.click(within(zero).getByRole("button", { name: "进入治理应用" })); await within(zero).findByText("治理移交未完成；当前结果仍保留。");
    upload("少样本目标任务包", "few-foreign.sgtask.zip");
    const few = screen.getByRole("region", { name: "少样本路径" }); await waitFor(() => expect(few).toHaveAttribute("data-phase", "raw")); await confirmRun(few);
    await waitForFewShotOrder(few);
    fireEvent.click(within(few).getByRole("button", { name: "进入治理应用" })); await within(few).findByText("治理移交未完成；当前结果仍保留。");
    expect(onGovernanceHandoff).not.toHaveBeenCalled();
  });

  it("rejects a self-consistent zero-shot collection with an extra duplicate item and altered case copy", async () => {
    const collection = targetReviewCollection();
    const alteredCase = {
      ...collection.case,
      title: "Altered review title",
      description: "Altered review description",
      items: [...collection.case.items, { ...collection.case.items[0], itemId: `item-${"f".repeat(32)}` }],
    };
    const logical = { ...collection, case: alteredCase };
    const { collectionHash: _ignored, ...withoutHash } = logical;
    const rebound = { ...withoutHash, collectionHash: sha256Canonical(withoutHash) };
    const onGovernanceHandoff = vi.fn();
    const api = client({
      registerTargetTask: vi.fn().mockResolvedValue(targetTaskRegistration("zero_shot")),
      createTargetReviewCollection: vi.fn().mockResolvedValue(rebound),
    });
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onGovernanceHandoff={onGovernanceHandoff} onClose={vi.fn()} />);
    upload("零样本目标任务包", "zero-duplicate.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "零样本路径" }); await within(lane).findByText("108 个对象 · 220 条关系"); await confirmRun(lane);
    fireEvent.click(within(lane).getByRole("button", { name: "进入治理应用" }));
    await within(lane).findByText("治理移交未完成；当前结果仍保留。");
    expect(onGovernanceHandoff).not.toHaveBeenCalled();
  });
  it("rejects a self-consistent zero-shot collection rebound to another result hash", async () => {
    const collection = targetReviewCollection();
    const logical = { ...collection, resultHash: "6".repeat(64) };
    const { collectionHash: _ignored, ...withoutHash } = logical;
    const rebound = { ...withoutHash, collectionHash: sha256Canonical(withoutHash) };
    const onGovernanceHandoff = vi.fn();
    const api = client({
      registerTargetTask: vi.fn().mockResolvedValue(targetTaskRegistration("zero_shot")),
      createTargetReviewCollection: vi.fn().mockResolvedValue(rebound),
    });
    render(<AdaptationWorkspace client={api} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onGovernanceHandoff={onGovernanceHandoff} onClose={vi.fn()} />);
    upload("零样本目标任务包", "zero-result-rebound.sgtask.zip");
    const lane = await screen.findByRole("region", { name: "零样本路径" }); await within(lane).findByText("108 个对象 · 220 条关系"); await confirmRun(lane);
    fireEvent.click(within(lane).getByRole("button", { name: "进入治理应用" }));
    await within(lane).findByText("治理移交未完成；当前结果仍保留。");
    expect(onGovernanceHandoff).not.toHaveBeenCalled();
  });

  it("does not expose location, showcase, protocol, or predictive-performance copy in the ordinary workspace", () => {
    const { container } = render(<AdaptationWorkspace client={client()} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    expect(container.textContent).not.toMatch(/Thailand|sample|demo|experiment|schemaVersion|AdaptationReviewPolicy|probability|accuracy/iu);
  });
});
