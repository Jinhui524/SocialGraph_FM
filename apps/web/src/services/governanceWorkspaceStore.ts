import type {
  GovernanceArtifact,
  GovernanceOnlinePreview,
  GovernanceOnlineResult,
  GovernanceOnlineRun,
} from "../types/governanceOnline";

export const GOVERNANCE_WORKSPACE_SCHEMA = "socialgraph-fm.governance-workspace/1.0" as const;

export interface GovernanceWorkspaceSnapshot {
  readonly schemaVersion: typeof GOVERNANCE_WORKSPACE_SCHEMA;
  readonly sessionId: string;
  readonly sourceFileName: string;
  readonly artifact: GovernanceArtifact;
  readonly preview: GovernanceOnlinePreview;
  readonly run?: GovernanceOnlineRun;
  readonly result?: GovernanceOnlineResult;
  readonly activeCaseId?: string;
  readonly updatedAt: string;
}

interface StoredGovernanceBinding {
  readonly schemaVersion: typeof GOVERNANCE_WORKSPACE_SCHEMA;
  readonly sessionId: string;
  readonly sourceFileName: string;
  readonly artifactId: string;
  readonly artifactHash: string;
  readonly datasetContentHash: string;
  readonly graphVersionHash: string;
  readonly runId?: string;
  readonly runRequestHash?: string;
  readonly resultHash?: string;
  readonly modelVersionId?: string;
  readonly modelVersionHash?: string;
  readonly modelStateHash?: string;
  readonly activeCaseId?: string;
  readonly updatedAt: string;
}

const DATABASE_NAME = "socialgraph-fm-governance-workspace";
const STORE_NAME = "session-bindings";
const memoryBindings = new Map<string, StoredGovernanceBinding>();

function indexedDbAvailable(): boolean {
  return typeof indexedDB !== "undefined";
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) database.createObjectStore(STORE_NAME, { keyPath: "sessionId" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("GOVERNANCE_WORKSPACE_DATABASE_UNAVAILABLE"));
  });
}

async function withStore<T>(mode: IDBTransactionMode, action: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  const database = await openDatabase();
  try {
    return await new Promise<T>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, mode);
      const request = action(transaction.objectStore(STORE_NAME));
      let requestSucceeded = false;
      let result: T;
      request.onsuccess = () => {
        requestSucceeded = true;
        result = request.result;
      };
      request.onerror = () => reject(request.error ?? new Error("GOVERNANCE_WORKSPACE_DATABASE_UNAVAILABLE"));
      transaction.oncomplete = () => {
        if (requestSucceeded) resolve(result);
        else reject(new Error("GOVERNANCE_WORKSPACE_DATABASE_UNAVAILABLE"));
      };
      transaction.onerror = () => reject(transaction.error ?? new Error("GOVERNANCE_WORKSPACE_DATABASE_UNAVAILABLE"));
      transaction.onabort = () => reject(transaction.error ?? new Error("GOVERNANCE_WORKSPACE_DATABASE_UNAVAILABLE"));
    });
  } finally {
    database.close();
  }
}

export function storedBindingFromSnapshot(snapshot: GovernanceWorkspaceSnapshot): StoredGovernanceBinding {
  return Object.freeze({
    schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
    sessionId: snapshot.sessionId,
    sourceFileName: snapshot.sourceFileName,
    artifactId: snapshot.artifact.artifactId,
    artifactHash: snapshot.artifact.artifactHash,
    datasetContentHash: snapshot.artifact.datasetContentHash,
    graphVersionHash: snapshot.artifact.graphVersionHash,
    ...(snapshot.run?.runId ? { runId: snapshot.run.runId } : {}),
    ...(snapshot.run?.requestHash ? { runRequestHash: snapshot.run.requestHash } : {}),
    ...(snapshot.result?.resultHash ? { resultHash: snapshot.result.resultHash } : {}),
    ...((snapshot.result?.modelVersionId ?? snapshot.run?.modelVersionId)
      ? { modelVersionId: snapshot.result?.modelVersionId ?? snapshot.run?.modelVersionId }
      : {}),
    ...((snapshot.result?.modelVersionHash ?? snapshot.run?.modelVersionHash)
      ? { modelVersionHash: snapshot.result?.modelVersionHash ?? snapshot.run?.modelVersionHash }
      : {}),
    ...((snapshot.result?.modelStateHash ?? snapshot.run?.modelStateHash)
      ? { modelStateHash: snapshot.result?.modelStateHash ?? snapshot.run?.modelStateHash }
      : {}),
    ...(snapshot.activeCaseId ? { activeCaseId: snapshot.activeCaseId } : {}),
    updatedAt: snapshot.updatedAt,
  });
}

export async function saveGovernanceWorkspaceSnapshot(snapshot: GovernanceWorkspaceSnapshot): Promise<void> {
  const binding = storedBindingFromSnapshot(snapshot);
  if (indexedDbAvailable()) {
    await withStore("readwrite", (store) => store.put(binding));
  }
  memoryBindings.set(binding.sessionId, binding);
}

export async function loadGovernanceWorkspaceBinding(sessionId: string): Promise<StoredGovernanceBinding | null> {
  if (!sessionId) return null;
  if (!indexedDbAvailable()) return memoryBindings.get(sessionId) ?? null;
  const stored = await withStore<StoredGovernanceBinding | undefined>("readonly", (store) => store.get(sessionId));
  return stored ?? memoryBindings.get(sessionId) ?? null;
}

export async function deleteGovernanceWorkspaceBinding(sessionId: string): Promise<void> {
  if (indexedDbAvailable()) {
    await withStore("readwrite", (store) => store.delete(sessionId));
  }
  memoryBindings.delete(sessionId);
}
