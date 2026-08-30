import { afterEach, describe, expect, it, vi } from "vitest";

import { onlineArtifact, onlinePreview, onlineResult, onlineRun } from "../test/fixtures/governanceOnline";
import type { GovernanceArtifact, GovernanceOnlinePreview, GovernanceOnlineResult, GovernanceOnlineRun } from "../types/governanceOnline";
import {
  deleteGovernanceWorkspaceBinding,
  GOVERNANCE_WORKSPACE_SCHEMA,
  loadGovernanceWorkspaceBinding,
  saveGovernanceWorkspaceSnapshot,
  storedBindingFromSnapshot,
} from "./governanceWorkspaceStore";

afterEach(() => vi.unstubAllGlobals());

describe("governanceWorkspaceStore", () => {
  it("persists identities only and never stores vectors or full results", async () => {
    vi.stubGlobal("indexedDB", undefined);
    const snapshot = {
      schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
      sessionId: "session-a",
      sourceFileName: "russia-04.zip",
      artifact: onlineArtifact() as GovernanceArtifact,
      preview: onlinePreview() as GovernanceOnlinePreview,
      run: onlineRun() as GovernanceOnlineRun,
      result: onlineResult() as GovernanceOnlineResult,
      updatedAt: "2026-08-19T00:00:00Z",
    } as const;
    const identity = storedBindingFromSnapshot(snapshot);
    expect(identity).toMatchObject({ sessionId: "session-a", sourceFileName: "russia-04.zip", artifactId: snapshot.artifact.artifactId, runId: snapshot.run.runId });
    expect(JSON.stringify(identity)).not.toContain("findings");
    expect(JSON.stringify(identity)).not.toContain("nodes");
    expect(JSON.stringify(identity)).not.toContain("features");
    expect(Object.keys(identity).sort()).toEqual([
      "artifactHash",
      "artifactId",
      "datasetContentHash",
      "graphVersionHash",
      "modelStateHash",
      "modelVersionHash",
      "modelVersionId",
      "resultHash",
      "runId",
      "runRequestHash",
      "schemaVersion",
      "sessionId",
      "sourceFileName",
      "updatedAt",
    ]);

    await saveGovernanceWorkspaceSnapshot(snapshot);
    await expect(loadGovernanceWorkspaceBinding("session-a")).resolves.toEqual(identity);
    await deleteGovernanceWorkspaceBinding("session-a");
    await expect(loadGovernanceWorkspaceBinding("session-a")).resolves.toBeNull();
  });

  it("isolates bindings by session and retains only the latest immutable identities", async () => {
    vi.stubGlobal("indexedDB", undefined);
    const base = {
      schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
      sourceFileName: "russia-04.zip",
      artifact: onlineArtifact() as GovernanceArtifact,
      preview: onlinePreview() as GovernanceOnlinePreview,
      updatedAt: "2026-08-19T00:00:00Z",
    } as const;
    await saveGovernanceWorkspaceSnapshot({ ...base, sessionId: "session-one" });
    await saveGovernanceWorkspaceSnapshot({
      ...base,
      sessionId: "session-two",
      sourceFileName: "russia-03.zip",
      activeCaseId: `case-${"9".repeat(32)}`,
    });

    await expect(loadGovernanceWorkspaceBinding("session-one")).resolves.toMatchObject({
      sessionId: "session-one",
      sourceFileName: "russia-04.zip",
    });
    await expect(loadGovernanceWorkspaceBinding("session-two")).resolves.toMatchObject({
      sessionId: "session-two",
      sourceFileName: "russia-03.zip",
      activeCaseId: `case-${"9".repeat(32)}`,
    });
    await deleteGovernanceWorkspaceBinding("session-one");
    await deleteGovernanceWorkspaceBinding("session-two");
  });

  it("persists the model state for a run before any result exists", () => {
    const snapshot = {
      schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
      sessionId: "session-running",
      sourceFileName: "russia-04.zip",
      artifact: onlineArtifact() as GovernanceArtifact,
      preview: onlinePreview() as GovernanceOnlinePreview,
      run: { ...onlineRun(), status: "running", stage: "inferencing", progress: 50 } as GovernanceOnlineRun,
      updatedAt: "2026-08-19T00:00:00Z",
    } as const;

    expect(storedBindingFromSnapshot(snapshot)).toMatchObject({
      runId: snapshot.run.runId,
      modelVersionId: snapshot.run.modelVersionId,
      modelVersionHash: snapshot.run.modelVersionHash,
      modelStateHash: snapshot.run.modelStateHash,
    });
  });

  it("waits for the IndexedDB transaction commit after a successful put request", async () => {
    const transaction = {
      error: new Error("commit failed"),
      objectStore: vi.fn(),
      onabort: null as null | (() => void),
      oncomplete: null as null | (() => void),
      onerror: null as null | (() => void),
    };
    const database = {
      objectStoreNames: { contains: () => true },
      createObjectStore: vi.fn(),
      transaction: vi.fn(() => transaction),
      close: vi.fn(),
    };
    const openRequest = {
      result: database,
      error: null,
      onupgradeneeded: null as null | (() => void),
      onsuccess: null as null | (() => void),
      onerror: null as null | (() => void),
    };
    const putRequest = {
      result: undefined,
      error: null,
      onsuccess: null as null | (() => void),
      onerror: null as null | (() => void),
    };
    transaction.objectStore.mockReturnValue({
      put: vi.fn(() => {
        queueMicrotask(() => {
          putRequest.onsuccess?.();
          queueMicrotask(() => transaction.onabort?.());
        });
        return putRequest;
      }),
    });
    vi.stubGlobal("indexedDB", {
      open: vi.fn(() => {
        queueMicrotask(() => openRequest.onsuccess?.());
        return openRequest;
      }),
    });
    const snapshot = {
      schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
      sessionId: "session-commit-failure",
      sourceFileName: "russia-04.zip",
      artifact: onlineArtifact() as GovernanceArtifact,
      preview: onlinePreview() as GovernanceOnlinePreview,
      updatedAt: "2026-08-19T00:00:00Z",
    } as const;

    await expect(saveGovernanceWorkspaceSnapshot(snapshot)).rejects.toThrow("commit failed");

    vi.stubGlobal("indexedDB", undefined);
    await expect(loadGovernanceWorkspaceBinding(snapshot.sessionId)).resolves.toBeNull();
    await deleteGovernanceWorkspaceBinding(snapshot.sessionId);
  });
});
