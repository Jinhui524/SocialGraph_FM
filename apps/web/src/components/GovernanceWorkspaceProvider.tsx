import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { GovernanceOnlineClient } from "../services/governanceOnlineClient";
import {
  deleteGovernanceWorkspaceBinding,
  GOVERNANCE_WORKSPACE_SCHEMA,
  loadGovernanceWorkspaceBinding,
  saveGovernanceWorkspaceSnapshot,
  type GovernanceWorkspaceSnapshot,
} from "../services/governanceWorkspaceStore";

type WorkspaceRestoreState = "idle" | "restoring" | "ready" | "unavailable";

class WorkspaceIdentityMismatchError extends Error {
  constructor() { super("GOVERNANCE_WORKSPACE_IDENTITY_MISMATCH"); }
}

interface GovernanceWorkspaceContextValue {
  readonly snapshot: GovernanceWorkspaceSnapshot | null;
  readonly restoreState: WorkspaceRestoreState;
  readonly restoreMessage: string | null;
  readonly activateSession: (sessionId: string) => Promise<void>;
  readonly bindSnapshot: (snapshot: GovernanceWorkspaceSnapshot) => Promise<void>;
  readonly clearSession: (sessionId: string) => Promise<void>;
}

const GovernanceWorkspaceContext = createContext<GovernanceWorkspaceContextValue | null>(null);

function sameGraphIdentity(
  value: { artifactId: string; datasetContentHash: string; graphVersionHash: string },
  expected: { artifactId: string; datasetContentHash: string; graphVersionHash: string },
): boolean {
  return value.artifactId === expected.artifactId
    && value.datasetContentHash === expected.datasetContentHash
    && value.graphVersionHash === expected.graphVersionHash;
}

export function GovernanceWorkspaceProvider({ children }: { readonly children: ReactNode }) {
  const client = useMemo(() => new GovernanceOnlineClient(), []);
  const [snapshot, setSnapshot] = useState<GovernanceWorkspaceSnapshot | null>(null);
  const [restoreState, setRestoreState] = useState<WorkspaceRestoreState>("idle");
  const [restoreMessage, setRestoreMessage] = useState<string | null>(null);
  const epochRef = useRef(0);
  const activeSessionRef = useRef("");

  const activateSession = useCallback(async (sessionId: string) => {
    activeSessionRef.current = sessionId;
    const epoch = ++epochRef.current;
    setSnapshot(null);
    setRestoreMessage(null);
    if (!sessionId) { setRestoreState("idle"); return; }
    setRestoreState("restoring");
    try {
      const stored = await loadGovernanceWorkspaceBinding(sessionId);
      if (epochRef.current !== epoch || activeSessionRef.current !== sessionId) return;
      if (!stored) { setRestoreState("ready"); return; }
      const capabilities = await client.capabilities();
      if (!capabilities.onlineForwardReady) {
        throw new Error("GOVERNANCE_WORKSPACE_SERVICE_NOT_READY");
      }
      if (stored.runId && !stored.modelStateHash) {
        throw new WorkspaceIdentityMismatchError();
      }
      if (stored.modelVersionId && capabilities.modelVersionId !== stored.modelVersionId
        || stored.modelVersionHash && capabilities.modelVersionHash !== stored.modelVersionHash
        || stored.modelStateHash && capabilities.modelStateHash !== stored.modelStateHash) {
        throw new WorkspaceIdentityMismatchError();
      }
      const artifact = await client.artifact(stored.artifactId);
      if (epochRef.current !== epoch || activeSessionRef.current !== sessionId) return;
      // POST /artifacts returns the materialized GFM artifact while
      // GET /artifacts/{id} returns the API inbox receipt. Both objects are
      // independently hash-verified, but their artifactHash values bind
      // different schemas and therefore must not be compared to each other.
      if (!sameGraphIdentity(artifact, stored)) throw new WorkspaceIdentityMismatchError();
      const preview = await client.preview(artifact.artifactId, undefined, { preset: "overview", nodeBudget: 120, edgeBudget: 240 });
      if (!sameGraphIdentity(preview, stored)) throw new WorkspaceIdentityMismatchError();
      let run;
      let result;
      if (stored.runId) {
        run = await client.run(stored.runId);
        if (!sameGraphIdentity(run, stored)
          || stored.runRequestHash && run.requestHash !== stored.runRequestHash
          || stored.modelVersionId && run.modelVersionId !== stored.modelVersionId
          || stored.modelVersionHash && run.modelVersionHash !== stored.modelVersionHash
          || stored.modelStateHash && run.modelStateHash !== stored.modelStateHash) {
          throw new WorkspaceIdentityMismatchError();
        }
        if (run.status === "succeeded") {
          result = await client.result(run.runId);
          if (!sameGraphIdentity(result, stored)
            || result.requestHash !== run.requestHash
            || result.modelVersionId !== run.modelVersionId
            || result.modelVersionHash !== run.modelVersionHash
            || result.modelStateHash !== run.modelStateHash
            || stored.resultHash && result.resultHash !== stored.resultHash
            || stored.modelVersionId && result.modelVersionId !== stored.modelVersionId
            || stored.modelStateHash && result.modelStateHash !== stored.modelStateHash) {
            throw new WorkspaceIdentityMismatchError();
          }
        }
      }
      if (epochRef.current !== epoch || activeSessionRef.current !== sessionId) return;
      setSnapshot(Object.freeze({
        schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
        sessionId,
        sourceFileName: stored.sourceFileName,
        artifact,
        preview,
        ...(run ? { run } : {}),
        ...(result ? { result } : {}),
        ...(stored.activeCaseId ? { activeCaseId: stored.activeCaseId } : {}),
        updatedAt: stored.updatedAt,
      }));
      setRestoreState("ready");
    } catch (error) {
      if (epochRef.current !== epoch || activeSessionRef.current !== sessionId) return;
      setSnapshot(null);
      setRestoreState("unavailable");
      if (error instanceof WorkspaceIdentityMismatchError) {
        await deleteGovernanceWorkspaceBinding(sessionId).catch(() => undefined);
        setRestoreMessage("此前绑定的推理包身份已变化，请重新选择本地文件。现有服务端制品未被删除。");
      } else {
        setRestoreMessage("推理包暂时无法恢复，绑定已保留。请确认本机分析服务在线后重试。");
      }
    }
  }, [client]);

  const bindSnapshot = useCallback(async (next: GovernanceWorkspaceSnapshot) => {
    if (!next.sessionId || next.sessionId !== activeSessionRef.current) return;
    // A new upload is authoritative for the active session and cancels any
    // slower restoration of its previous binding.
    epochRef.current += 1;
    setSnapshot(next);
    setRestoreState("ready");
    setRestoreMessage(null);
    await saveGovernanceWorkspaceSnapshot(next);
  }, []);

  const clearSession = useCallback(async (sessionId: string) => {
    await deleteGovernanceWorkspaceBinding(sessionId);
    if (activeSessionRef.current === sessionId) {
      epochRef.current += 1;
      setSnapshot(null);
      setRestoreState("ready");
      setRestoreMessage(null);
    }
  }, []);

  const value = useMemo<GovernanceWorkspaceContextValue>(() => ({
    snapshot, restoreState, restoreMessage, activateSession, bindSnapshot, clearSession,
  }), [activateSession, bindSnapshot, clearSession, restoreMessage, restoreState, snapshot]);
  return <GovernanceWorkspaceContext.Provider value={value}>{children}</GovernanceWorkspaceContext.Provider>;
}

export function useGovernanceWorkspace(): GovernanceWorkspaceContextValue {
  const value = useContext(GovernanceWorkspaceContext);
  if (!value) throw new Error("GovernanceWorkspaceProvider is missing");
  return value;
}
