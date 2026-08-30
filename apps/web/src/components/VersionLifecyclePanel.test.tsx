import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { createGraphVersion } from "../services/graphImport";
import { createLocalGraphRepository } from "../services/graphRepository";
import { createSourceArtifact } from "../services/sourceArtifact";
import type { GraphVersion } from "../types/graph";
import { VersionLifecyclePanel } from "./VersionLifecyclePanel";

function fixtureVersion(
  id: string,
  sourceFile: string,
  parentVersionId?: string,
  typed = true,
): GraphVersion {
  const created = createGraphVersion(sourceFile, [
    {
      id: "n1",
      label: sourceFile,
      ...(typed ? { type: "person" } : {}),
      attributes: {},
    },
  ], [], [], {
    ...(parentVersionId ? { parentVersionId } : {}),
  });
  return Object.freeze({ ...created, id });
}

beforeEach(() => {
  Object.defineProperty(URL, "createObjectURL", {
    configurable: true,
    writable: true,
    value: vi.fn(() => "blob:version-lifecycle-test"),
  });
  Object.defineProperty(URL, "revokeObjectURL", {
    configurable: true,
    writable: true,
    value: vi.fn(),
  });
  vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("VersionLifecyclePanel", () => {
  it("renders lineage, activates a version, and defaults child diff to its parent", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const base = fixtureVersion("graph-parent-00000001", "base.csv");
    const child = fixtureVersion("graph-child-00000002", "child.csv", base.id);
    await repository.saveGraphVersion(base);
    await repository.saveGraphVersion(child);
    const onActivateVersion = vi.fn();

    render(
      <VersionLifecyclePanel
        repository={repository}
        currentGraphVersionId={base.id}
        onActivateVersion={onActivateVersion}
      />,
    );

    expect(await screen.findByText("child.csv")).toBeInTheDocument();
    expect(screen.getByText(/父版本：base.csv/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "设为当前 child.csv" }));
    await waitFor(() => expect(onActivateVersion).toHaveBeenCalledWith(expect.objectContaining({ id: child.id })));

    fireEvent.click(screen.getByRole("button", { name: "比较版本 child.csv" }));
    const dialog = await screen.findByRole("dialog", { name: "版本差异" });
    expect(dialog).toHaveTextContent(`${base.id} → ${child.id}`);
    expect(dialog).toHaveTextContent("关系事实");
    expect(dialog).toHaveTextContent("edge ID churn");

    fireEvent.click(screen.getByRole("button", { name: "导出差异 JSON" }));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);
    expect(HTMLAnchorElement.prototype.click).toHaveBeenCalledTimes(1);
    repository.dispose();
  });

  it("explains why a legacy untyped version has a single node color", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const legacy = fixtureVersion("graph-legacy-00000003", "legacy.csv", undefined, false);
    await repository.saveGraphVersion(legacy);
    render(
      <VersionLifecyclePanel
        repository={repository}
        onActivateVersion={vi.fn()}
      />,
    );

    expect(await screen.findByText("兼容版本")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "检查兼容性 legacy.csv" }));

    const dialog = await screen.findByRole("dialog", { name: "版本兼容性" });
    expect(dialog).toHaveTextContent("兼容只读版本");
    expect(dialog).toHaveTextContent("所有节点均未保存类型");
    expect(dialog).toHaveTextContent("不是渲染器丢失配色");
    repository.dispose();
  });

  it("previews references and moves a leaf version through trash and restore", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureVersion("graph-lifecycle-12345678", "lifecycle.csv");
    await repository.saveGraphVersion(graph);
    render(
      <VersionLifecyclePanel
        repository={repository}
        onActivateVersion={vi.fn()}
      />,
    );

    expect(await screen.findByText("lifecycle.csv")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看引用与删除 lifecycle.csv" }));
    const activeDialog = await screen.findByRole("dialog", { name: "引用预演" });
    expect(activeDialog).toHaveTextContent("没有外部引用");
    fireEvent.click(screen.getByRole("button", { name: "移入回收站" }));

    await waitFor(async () => {
      expect((await repository.getResourceLifecycle("graph_version", graph.id)).state).toBe("trashed");
    });
    fireEvent.click(screen.getByRole("tab", { name: "回收站" }));
    expect(await screen.findByText("lifecycle.csv")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看引用与删除 lifecycle.csv" }));
    await screen.findByText(/输入 ID 后 8 位/);
    fireEvent.click(screen.getByRole("button", { name: "从回收站恢复" }));

    await waitFor(async () => {
      expect((await repository.getResourceLifecycle("graph_version", graph.id)).state).toBe("active");
    });
    repository.dispose();
  });

  it("requires the exact ID suffix before an unreferenced trashed resource can be purged", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureVersion("graph-purge-abcdefgh", "purge.csv");
    await repository.saveGraphVersion(graph);
    const initialImpact = await repository.inspectGraphVersionDeletion(graph.id);
    await repository.trashGraphVersion(graph.id, initialImpact.impactHash);
    render(
      <VersionLifecyclePanel
        repository={repository}
        onActivateVersion={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("tab", { name: "回收站" }));
    expect(await screen.findByText("purge.csv")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "查看引用与删除 purge.csv" }));
    const input = await screen.findByRole("textbox", { name: "输入 ID 后 8 位确认永久删除" });
    const purgeButton = screen.getByRole("button", { name: "永久删除" });
    fireEvent.change(input, { target: { value: "错误确认" } });
    expect(purgeButton).toBeDisabled();
    fireEvent.change(input, { target: { value: "abcdefgh" } });
    expect(purgeButton).toBeEnabled();
    fireEvent.click(purgeButton);

    await waitFor(async () => expect(await repository.getGraphVersion(graph.id)).toBeUndefined());
    repository.dispose();
  });

  it("exports SourceArtifact, blocks referenced deletion, and refreshes from repository events", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    render(
      <VersionLifecyclePanel
        repository={repository}
        onActivateVersion={vi.fn()}
      />,
    );
    expect(await screen.findByText("此状态下没有 SourceArtifact")).toBeInTheDocument();

    const artifact = await createSourceArtifact(new File(["source,target\na,b\n"], "source.csv", { type: "text/csv" }));
    const graph = Object.freeze({
      ...fixtureVersion("graph-source-00000004", "source-graph.csv"),
      sourceArtifactIds: Object.freeze([artifact.id]),
    });
    await act(async () => {
      await repository.saveSourceArtifact(artifact);
      await repository.saveGraphVersion(graph);
    });

    expect(await screen.findByText("source.csv")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "导出原文件" }));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole("button", { name: "查看引用与删除 source.csv" }));
    const dialog = await screen.findByRole("dialog", { name: "引用预演" });
    expect(dialog).toHaveTextContent("source-graph.csv");
    expect(dialog).toHaveTextContent("阻止永久删除");
    expect(screen.getByRole("button", { name: "移入回收站" })).toBeDisabled();
    repository.dispose();
  });
});
