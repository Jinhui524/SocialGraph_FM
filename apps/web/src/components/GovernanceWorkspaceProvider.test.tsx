import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  onlineArtifact,
  onlineCapabilities,
  onlinePreview,
  onlineResult,
  onlineRun,
} from "../test/fixtures/governanceOnline";
import type {
  GovernanceArtifact,
  GovernanceOnlinePreview,
  GovernanceOnlineResult,
  GovernanceOnlineRun,
} from "../types/governanceOnline";
import {
  deleteGovernanceWorkspaceBinding,
  GOVERNANCE_WORKSPACE_SCHEMA,
  loadGovernanceWorkspaceBinding,
  saveGovernanceWorkspaceSnapshot,
} from "../services/governanceWorkspaceStore";

const client = vi.hoisted(() => ({
  capabilities: vi.fn(),
  artifact: vi.fn(),
  preview: vi.fn(),
  run: vi.fn(),
  result: vi.fn(),
}));

vi.mock("../services/governanceOnlineClient", () => ({
  GovernanceOnlineClient: class {
    capabilities = client.capabilities;
    artifact = client.artifact;
    preview = client.preview;
    run = client.run;
    result = client.result;
  },
}));

import { GovernanceWorkspaceProvider, useGovernanceWorkspace } from "./GovernanceWorkspaceProvider";

const SESSION_IDS = ["session-match", "session-receipt-hash", "session-transient", "session-run-only-state", "session-modelVersionId", "session-modelVersionHash", "session-modelStateHash"] as const;

function WorkspaceProbe({ sessionId }: { readonly sessionId: string }) {
  const workspace = useGovernanceWorkspace();
  return (
    <>
      <button type="button" onClick={() => void workspace.activateSession(sessionId)}>恢复</button>
      <output aria-label="恢复状态">{workspace.restoreState}</output>
      <output aria-label="绑定文件">{workspace.snapshot?.sourceFileName ?? "none"}</output>
      <output aria-label="运行身份">{workspace.snapshot?.run?.runId ?? "none"}</output>
      <output aria-label="恢复提示">{workspace.restoreMessage ?? "none"}</output>
    </>
  );
}

async function seedBinding(sessionId: string): Promise<void> {
  await saveGovernanceWorkspaceSnapshot({
    schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
    sessionId,
    sourceFileName: "russia-04.zip",
    artifact: onlineArtifact() as GovernanceArtifact,
    preview: onlinePreview() as GovernanceOnlinePreview,
    run: onlineRun() as GovernanceOnlineRun,
    result: onlineResult() as GovernanceOnlineResult,
    updatedAt: "2026-08-19T00:00:00Z",
  });
}

function renderProvider(sessionId: string) {
  render(<GovernanceWorkspaceProvider><WorkspaceProbe sessionId={sessionId} /></GovernanceWorkspaceProvider>);
  fireEvent.click(screen.getByRole("button", { name: "恢复" }));
}

beforeEach(() => {
  vi.stubGlobal("indexedDB", undefined);
  vi.clearAllMocks();
  client.capabilities.mockResolvedValue(onlineCapabilities());
  client.artifact.mockResolvedValue(onlineArtifact());
  client.preview.mockResolvedValue(onlinePreview());
  client.run.mockResolvedValue(onlineRun());
  client.result.mockResolvedValue(onlineResult());
});

afterEach(async () => {
  cleanup();
  await Promise.all(SESSION_IDS.map((sessionId) => deleteGovernanceWorkspaceBinding(sessionId)));
  vi.unstubAllGlobals();
});

describe("GovernanceWorkspaceProvider identity restoration", () => {
  it("restores artifact, preview, run and result only when every model identity matches", async () => {
    await seedBinding("session-match");
    renderProvider("session-match");

    await waitFor(() => expect(screen.getByLabelText("恢复状态")).toHaveTextContent("ready"));
    expect(screen.getByLabelText("绑定文件")).toHaveTextContent("russia-04.zip");
    expect(screen.getByLabelText("运行身份")).toHaveTextContent(onlineRun().runId);
    expect(client.capabilities).toHaveBeenCalledTimes(1);
    expect(client.artifact).toHaveBeenCalledTimes(1);
    expect(client.preview).toHaveBeenCalledWith(
      onlineArtifact().artifactId,
      undefined,
      { preset: "overview", nodeBudget: 120, edgeBudget: 240 },
    );
    expect(client.run).toHaveBeenCalledWith(onlineRun().runId);
    expect(client.result).toHaveBeenCalledWith(onlineRun().runId);
    await expect(loadGovernanceWorkspaceBinding("session-match")).resolves.not.toBeNull();
  });

  it("restores when the API receipt hash differs from the materialized artifact hash", async () => {
    await seedBinding("session-receipt-hash");
    client.artifact.mockResolvedValue({ ...onlineArtifact(), artifactHash: "f".repeat(64) });
    renderProvider("session-receipt-hash");

    await waitFor(() => expect(screen.getByLabelText("恢复状态")).toHaveTextContent("ready"));
    expect(screen.getByLabelText("绑定文件")).toHaveTextContent("russia-04.zip");
    expect(client.preview).toHaveBeenCalledTimes(1);
    await expect(loadGovernanceWorkspaceBinding("session-receipt-hash")).resolves.not.toBeNull();
  });

  it("keeps the local binding when restoration fails temporarily", async () => {
    await seedBinding("session-transient");
    client.capabilities.mockRejectedValue(new Error("service temporarily unavailable"));
    renderProvider("session-transient");

    await waitFor(() => expect(screen.getByLabelText("恢复状态")).toHaveTextContent("unavailable"));
    expect(screen.getByLabelText("绑定文件")).toHaveTextContent("none");
    expect(screen.getByLabelText("恢复提示")).toHaveTextContent("绑定已保留");
    await expect(loadGovernanceWorkspaceBinding("session-transient")).resolves.not.toBeNull();
  });

  it("rejects a run-only snapshot when the restored run changes model state", async () => {
    await saveGovernanceWorkspaceSnapshot({
      schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
      sessionId: "session-run-only-state",
      sourceFileName: "russia-04.zip",
      artifact: onlineArtifact() as GovernanceArtifact,
      preview: onlinePreview() as GovernanceOnlinePreview,
      run: { ...onlineRun(), status: "running", stage: "inferencing", progress: 50 } as GovernanceOnlineRun,
      updatedAt: "2026-08-19T00:00:00Z",
    });
    client.run.mockResolvedValue({ ...onlineRun(), modelStateHash: "e".repeat(64) });
    renderProvider("session-run-only-state");

    await waitFor(() => expect(screen.getByLabelText("恢复状态")).toHaveTextContent("unavailable"));
    expect(screen.getByLabelText("绑定文件")).toHaveTextContent("none");
    expect(screen.getByLabelText("恢复提示")).toHaveTextContent("身份已变化");
    expect(client.result).not.toHaveBeenCalled();
    await expect(loadGovernanceWorkspaceBinding("session-run-only-state")).resolves.toBeNull();
  });

  it.each([
    ["modelVersionId", "socialgraph-fm-global/changed"],
    ["modelVersionHash", "d".repeat(64)],
    ["modelStateHash", "e".repeat(64)],
  ] as const)("fails closed and clears the binding when capabilities %s changes", async (field, value) => {
    const sessionId = `session-${field}`;
    await seedBinding(sessionId);
    client.capabilities.mockResolvedValue({ ...onlineCapabilities(), [field]: value });
    renderProvider(sessionId);

    await waitFor(() => expect(screen.getByLabelText("恢复状态")).toHaveTextContent("unavailable"));
    expect(screen.getByLabelText("绑定文件")).toHaveTextContent("none");
    expect(screen.getByLabelText("运行身份")).toHaveTextContent("none");
    expect(screen.getByLabelText("恢复提示")).toHaveTextContent("身份已变化");
    expect(client.capabilities).toHaveBeenCalledTimes(1);
    expect(client.artifact).not.toHaveBeenCalled();
    expect(client.preview).not.toHaveBeenCalled();
    expect(client.run).not.toHaveBeenCalled();
    expect(client.result).not.toHaveBeenCalled();
    await expect(loadGovernanceWorkspaceBinding(sessionId)).resolves.toBeNull();
  });
});
