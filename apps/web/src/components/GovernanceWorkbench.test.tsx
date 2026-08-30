import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { targetCapabilities, targetComparison, targetPolicy, targetPreview, targetResult, targetRun, targetTaskRegistration } from "../test/fixtures/governanceTargetTask";
import type { GovernanceOnlineClientLike } from "../types/governanceOnline";
import { AdaptationWorkspace } from "./GovernanceWorkbench";

afterEach(cleanup);

function api(overrides: Partial<GovernanceOnlineClientLike> = {}): GovernanceOnlineClientLike {
  const registration = targetTaskRegistration(); const zeroRegistration = targetTaskRegistration("zero_shot"); const policy = targetPolicy();
  const modeForArtifact = (artifactId: string) => artifactId === zeroRegistration.artifact.artifactId ? "zero_shot" as const : "few_shot" as const;
  const modeForRun = (runId: string) => runId === targetRun("zero_shot").runId ? "zero_shot" as const : "few_shot" as const;
  return {
    capabilities: vi.fn().mockResolvedValue(targetCapabilities()), registerTargetTask: vi.fn().mockResolvedValue(registration), preview: vi.fn((artifactId: string) => Promise.resolve(targetPreview(false, modeForArtifact(artifactId)))),
    createRun: vi.fn((request: { artifactId: string }) => Promise.resolve(targetRun(modeForArtifact(request.artifactId)))), run: vi.fn((runId: string) => Promise.resolve(targetRun(modeForRun(runId)))), result: vi.fn((runId: string) => Promise.resolve(targetResult(modeForRun(runId)))), runPreview: vi.fn((runId: string) => Promise.resolve(targetPreview(true, modeForRun(runId)))),
    createTargetLabelSet: vi.fn().mockResolvedValue(registration.labels), fitTargetPolicy: vi.fn().mockResolvedValue(policy), targetPolicy: vi.fn().mockResolvedValue(policy), targetComparison: vi.fn().mockResolvedValue(targetComparison()),
    ...overrides,
  } as unknown as GovernanceOnlineClientLike;
}

function upload(label: string, name = "target.sgtask.zip") { fireEvent.change(screen.getByLabelText(label), { target: { files: [new File(["PK"], name, { type: "application/zip" })] } }); }

async function runAndFit(lane: HTMLElement) {
  fireEvent.click(within(lane).getByRole("button", { name: "开始分析" })); fireEvent.click(within(lane).getByRole("button", { name: "确认分析" }));
  await within(lane).findByText(/协同组群已就绪/u);
}

describe("target task fail-closed behavior", () => {
  it("rejects an incomplete graph before publishing it", async () => {
    const incomplete = { ...targetPreview(false, "zero_shot"), nodes: targetPreview(false, "zero_shot").nodes.slice(0, 107) };
    const onGraphChange = vi.fn();
    render(<AdaptationWorkspace client={api({ registerTargetTask: vi.fn().mockResolvedValue(targetTaskRegistration("zero_shot")), preview: vi.fn().mockResolvedValue(incomplete) })} onGraphChange={onGraphChange} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("零样本目标任务包");
    await screen.findByText("目标任务包未通过完整性或路径校验。");
    expect(onGraphChange).toHaveBeenCalledOnce(); expect(onGraphChange).toHaveBeenLastCalledWith("zero_shot", null);
  });

  it("does not publish a truncated comparison", async () => {
    const onOverlayChange = vi.fn();
    render(<AdaptationWorkspace client={api({ targetComparison: vi.fn().mockResolvedValue(targetComparison(107)) })} onGraphChange={vi.fn()} onOverlayChange={onOverlayChange} onClose={vi.fn()} />);
    upload("少样本目标任务包"); const lane = await screen.findByRole("region", { name: "少样本路径" }); await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw")); await runAndFit(lane);
    await within(lane).findByText("少样本网络适配未完成；基础结果保持不变，可重试适配。");
    expect(within(lane).queryByRole("table")).not.toBeInTheDocument();
    expect(onOverlayChange.mock.calls.some(([overlay]) => overlay?.legend?.title.startsWith("少样本复核顺序"))).toBe(false);
  });

  it("drops a deferred old fit after the package is replaced", async () => {
    let resolveLabels!: (value: NonNullable<ReturnType<typeof targetTaskRegistration>["labels"]>) => void;
    const labels = new Promise<NonNullable<ReturnType<typeof targetTaskRegistration>["labels"]>>((resolve) => { resolveLabels = resolve; });
    const createTargetLabelSet = vi.fn().mockReturnValueOnce(labels);
    const client = api({ createTargetLabelSet });
    render(<AdaptationWorkspace client={client} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("少样本目标任务包", "old.sgtask.zip"); const lane = await screen.findByRole("region", { name: "少样本路径" }); await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw")); await runAndFit(lane);
    await waitFor(() => expect(createTargetLabelSet).toHaveBeenCalledOnce());
    upload("少样本目标任务包", "new.sgtask.zip"); await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw")); resolveLabels(targetTaskRegistration().labels!); await Promise.resolve();
    expect(client.fitTargetPolicy).not.toHaveBeenCalled(); expect(within(lane).queryByRole("table")).not.toBeInTheDocument();
  });

  it("keeps a failed fit retryable and leaves the base result visible", async () => {
    const fitTargetPolicy = vi.fn().mockRejectedValueOnce(new Error("stale")).mockResolvedValueOnce(targetPolicy());
    const client = api({ fitTargetPolicy });
    render(<AdaptationWorkspace client={client} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("少样本目标任务包"); const lane = await screen.findByRole("region", { name: "少样本路径" }); await waitFor(() => expect(lane).toHaveAttribute("data-phase", "raw")); await runAndFit(lane);
    await within(lane).findByText("少样本网络适配未完成；基础结果保持不变，可重试适配。");
    expect(within(lane).getByText(/协同组群已就绪/u)).toBeVisible();
    fireEvent.click(within(lane).getByRole("button", { name: "重试适配" }));
    await waitFor(() => expect(lane).toHaveAttribute("data-phase", "compared"));
    expect(within(lane).getByText("协同组群已就绪 · 108 个账号")).toBeVisible();
    expect(fitTargetPolicy).toHaveBeenCalledTimes(2);
  });

  it("blocks few-shot fitting unless the registered target has exactly sixteen balanced labels", async () => {
    const registration = targetTaskRegistration();
    const labels = registration.labels!;
    const unready = {
      ...registration,
      labels: { ...labels, labels: labels.labels.slice(0, 12), positiveCount: 6, negativeCount: 6 },
    };
    const client = api({ registerTargetTask: vi.fn().mockResolvedValue(unready) });
    render(<AdaptationWorkspace client={client} onGraphChange={vi.fn()} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("少样本目标任务包");
    const lane = await screen.findByRole("region", { name: "少样本路径" });
    await within(lane).findByText("少样本标签未通过完整性校验，请核对目标任务包。");
    fireEvent.click(within(lane).getByRole("button", { name: "开始分析" }));
    fireEvent.click(within(lane).getByRole("button", { name: "确认分析" }));
    await within(lane).findByText(/协同组群已就绪/u);
    expect(within(lane).queryByRole("button", { name: "拟合冻结复核策略" })).not.toBeInTheDocument();
    expect(client.createTargetLabelSet).not.toHaveBeenCalled();
  });

  it("rejects a full-sized result whose finding IDs differ from the exact preview set", async () => {
    const result = targetResult("zero_shot");
    result.findings[107] = { ...result.findings[107], nodeId: "foreign-node" };
    const onGraphChange = vi.fn();
    render(<AdaptationWorkspace client={api({ registerTargetTask: vi.fn().mockResolvedValue(targetTaskRegistration("zero_shot")), result: vi.fn().mockResolvedValue(result) })} onGraphChange={onGraphChange} onOverlayChange={vi.fn()} onClose={vi.fn()} />);
    upload("零样本目标任务包");
    const lane = await screen.findByRole("region", { name: "零样本路径" });
    await within(lane).findByText("108 个对象 · 220 条关系");
    fireEvent.click(within(lane).getByRole("button", { name: "开始分析" }));
    fireEvent.click(within(lane).getByRole("button", { name: "确认分析" }));
    await within(lane).findByText("目标网络分析未完成；没有发布协同组群。");
    expect(within(lane).queryByText(/协同组群已就绪/u)).not.toBeInTheDocument();
    expect(onGraphChange.mock.calls.filter(([, graph]) => graph?.nodes.some((node: { id: string }) => node.id === "foreign-node"))).toHaveLength(0);
  });
});
