import Dexie, { type Table } from "dexie";

import type {
  AnalysisRun,
  ConversationMessage,
  DeletionDependentGroup,
  DeletionImpact,
  DeletionReference,
  GraphRepository,
  GraphVersion,
  GraphVersionManifest,
  GraphViewState,
  ManagedResourceKind,
  RepositoryChange,
  RepositoryChangeKind,
  RepositoryInitializationMetadata,
  ResourceLifecycle,
  ResourceListState,
  ResearchSession,
  SessionListState,
  SemanticEvent,
  SemanticEventType,
  JsonPrimitive,
  SourceArtifact,
  GraphImportPersistenceBundle,
  RetainedDependency,
} from "../types/graph";
import { compareUnicodeCodePoints, sha256Canonical } from "./graphIdentity";

export { createDefaultGraphViewState } from "./graphScene";

interface StoredInitializationMetadata extends RepositoryInitializationMetadata {
  readonly key: "initialization";
}

interface StoredManifestBackfillMetadata {
  readonly key: "graph-version-manifest-v4";
  readonly completedAt: string;
}

type StoredMetadata = StoredInitializationMetadata | StoredManifestBackfillMetadata;

class SocialGraphDatabase extends Dexie {
  graphVersions!: Table<GraphVersion, string>;
  viewStates!: Table<GraphViewState, string>;
  analysisRuns!: Table<AnalysisRun, string>;
  semanticEvents!: Table<SemanticEvent, string>;
  sessions!: Table<ResearchSession, string>;
  messages!: Table<ConversationMessage, string>;
  metadata!: Table<StoredMetadata, string>;
  sourceArtifacts!: Table<SourceArtifact, string>;
  graphVersionManifests!: Table<GraphVersionManifest, string>;
  resourceLifecycles!: Table<ResourceLifecycle, string>;

  constructor(name: string) {
    super(name);
    this.version(1).stores({
      graphVersions: "&id, createdAt",
      viewStates: "&graphVersionId",
      analysisRuns: "&id, graphVersionId, createdAt",
      semanticEvents: "&id, graphVersionId, sessionId, createdAt",
      sessions: "&id, graphVersionId, updatedAt",
    });
    this.version(2)
      .stores({
        graphVersions: "&id, createdAt",
        viewStates: "&graphVersionId",
        analysisRuns: "&id, graphVersionId, createdAt",
        semanticEvents: "&id, graphVersionId, sessionId, createdAt",
        sessions: "&id, lifecycle, deletedAt, graphVersionId, updatedAt",
        messages: "&id, sessionId, createdAt",
        metadata: "&key",
      })
      .upgrade(async (transaction) => {
        await transaction
          .table<ResearchSession, string>("sessions")
          .toCollection()
          .modify((session) => {
            if (!session.lifecycle) {
              (session as ResearchSession & { lifecycle: "active" }).lifecycle = "active";
            }
          });
      });
    this.version(3).stores({
      graphVersions: "&id, createdAt, parentVersionId, contentHash",
      viewStates: "&graphVersionId",
      analysisRuns: "&id, graphVersionId, createdAt",
      semanticEvents: "&id, graphVersionId, sessionId, createdAt",
      sessions: "&id, lifecycle, deletedAt, graphVersionId, updatedAt",
      messages: "&id, sessionId, createdAt",
      metadata: "&key",
      sourceArtifacts: "&id, sha256, createdAt",
    });
    this.version(4).stores({
      graphVersions: "&id, createdAt, parentVersionId, contentHash",
      graphVersionManifests: "&id, createdAt, parentVersionId, contentHash, *sourceArtifactIds, datasetArtifactId",
      resourceLifecycles: "&key, [kind+targetId], kind, targetId, state, trashedAt",
      viewStates: "&graphVersionId",
      analysisRuns: "&id, graphVersionId, createdAt",
      semanticEvents: "&id, graphVersionId, sessionId, createdAt",
      sessions: "&id, lifecycle, deletedAt, graphVersionId, updatedAt",
      messages: "&id, sessionId, analysisRunId, *sourceArtifactIds, createdAt",
      metadata: "&key",
      sourceArtifacts: "&id, sha256, createdAt",
    });
  }
}

export interface LocalGraphRepositoryOptions {
  readonly databaseName?: string;
  readonly forceMemory?: boolean;
}

function cloneValue<T>(value: T): T {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value)) as T;
}

function randomId(prefix: string): string {
  const suffix = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return `${prefix}-${suffix}`;
}

function lifecycleKey(kind: ManagedResourceKind, targetId: string) {
  return `${kind}:${targetId}`;
}

function activeLifecycle(kind: ManagedResourceKind, targetId: string): ResourceLifecycle {
  return Object.freeze({
    key: lifecycleKey(kind, targetId),
    kind,
    targetId,
    state: "active" as const,
    updatedAt: "",
  });
}

function createLifecycle(
  kind: ManagedResourceKind,
  targetId: string,
  state: "active" | "trashed",
  timestamp = new Date().toISOString(),
): ResourceLifecycle {
  return Object.freeze({
    key: lifecycleKey(kind, targetId),
    kind,
    targetId,
    state,
    updatedAt: timestamp,
    ...(state === "trashed" ? { trashedAt: timestamp } : {}),
  });
}

function deriveGraphVersionManifest(version: GraphVersion): GraphVersionManifest {
  const sourceArtifactIds = [...new Set([
    ...(version.sourceArtifactIds ?? []),
    ...(version.buildSpec?.sourceArtifactIds ?? []),
  ])].sort(compareUnicodeCodePoints);
  return Object.freeze({
    id: version.id,
    sourceFile: version.sourceFile,
    createdAt: version.createdAt,
    ...(version.parentVersionId ? { parentVersionId: version.parentVersionId } : {}),
    sourceArtifactIds: Object.freeze(sourceArtifactIds),
    ...(version.sourceHash ? { sourceHash: version.sourceHash } : {}),
    ...(version.buildSpecHash ? { buildSpecHash: version.buildSpecHash } : {}),
    ...(version.contentHash ? { contentHash: version.contentHash } : {}),
    nodeCount: version.summary.nodeCount,
    edgeCount: version.summary.edgeCount,
    directedness: version.metadata?.directedness ?? "unspecified",
    ...(version.datasetArtifact?.id ? { datasetArtifactId: version.datasetArtifact.id } : {}),
    ...(version.provenance ? { provenance: Object.freeze({ ...version.provenance }) } : {}),
  });
}

function messageSourceArtifactIds(message: ConversationMessage): readonly string[] {
  return [...new Set([
    ...(message.sourceArtifactIds ?? []),
    ...(message.attachments ?? []).flatMap((attachment) => attachment.sourceArtifactId ? [attachment.sourceArtifactId] : []),
  ])];
}

function withMessageSourceArtifacts(
  message: ConversationMessage,
  artifacts: readonly Pick<SourceArtifact, "id">[],
): ConversationMessage {
  const sourceArtifactIds = [...new Set([
    ...messageSourceArtifactIds(message),
    ...artifacts.map((artifact) => artifact.id),
  ])];
  const attachments = message.attachments?.map((attachment, index) => ({
    ...attachment,
    ...(attachment.sourceArtifactId
      ? { sourceArtifactId: attachment.sourceArtifactId }
      : artifacts[index]?.id
        ? { sourceArtifactId: artifacts[index].id }
        : {}),
  }));
  return Object.freeze({
    ...message,
    sourceArtifactIds: Object.freeze(sourceArtifactIds),
    ...(attachments ? { attachments: Object.freeze(attachments) } : {}),
  });
}

function matchesResourceState(
  state: ResourceListState,
  lifecycle: ResourceLifecycle | undefined,
) {
  if (state === "all") return true;
  return (lifecycle?.state ?? "active") === state;
}

function sortReferences(references: readonly DeletionReference[]) {
  return [...references].sort((left, right) => (
    compareUnicodeCodePoints(left.kind, right.kind) || compareUnicodeCodePoints(left.id, right.id)
  ));
}

function sortDependents(dependents: readonly DeletionDependentGroup[]) {
  return [...dependents]
    .map((dependent) => ({ ...dependent, ids: [...dependent.ids].sort(compareUnicodeCodePoints) }))
    .sort((left, right) => compareUnicodeCodePoints(left.kind, right.kind));
}

function sortRetained(dependencies: readonly RetainedDependency[]) {
  return [...dependencies].sort((left, right) => (
    compareUnicodeCodePoints(left.kind, right.kind) || compareUnicodeCodePoints(left.id, right.id)
  ));
}

function finalizeDeletionImpact(
  impact: Omit<DeletionImpact, "impactHash">,
): DeletionImpact {
  const references = sortReferences(impact.references);
  const dependents = sortDependents(impact.dependents);
  const retainedDependencies = sortRetained(impact.retainedDependencies);
  const impactHash = sha256Canonical({
    targetKind: impact.targetKind,
    targetId: impact.targetId,
    state: impact.state,
    references,
    dependents,
    retainedDependencies,
  });
  return Object.freeze({
    ...impact,
    references: Object.freeze(references.map((reference) => Object.freeze(reference))),
    dependents: Object.freeze(dependents.map((dependent) => Object.freeze({
      ...dependent,
      ids: Object.freeze([...dependent.ids]),
    }))),
    retainedDependencies: Object.freeze(retainedDependencies.map((dependency) => Object.freeze(dependency))),
    impactHash,
  });
}

/** Creates a compact semantic event; pointer-move data should never be passed here. */
export function createSemanticEvent(
  type: SemanticEventType,
  options: {
    readonly graphVersionId?: string;
    readonly sessionId?: string;
    readonly payload?: Readonly<Record<string, JsonPrimitive>>;
    readonly id?: string;
    readonly createdAt?: string;
  } = {},
): SemanticEvent {
  return Object.freeze({
    id: options.id ?? randomId("event"),
    type,
    createdAt: options.createdAt ?? new Date().toISOString(),
    ...(options.graphVersionId ? { graphVersionId: options.graphVersionId } : {}),
    ...(options.sessionId ? { sessionId: options.sessionId } : {}),
    payload: Object.freeze({ ...(options.payload ?? {}) }),
  });
}

export function createResearchSession(
  title: string,
  options: {
    readonly id?: string;
    readonly graphVersionId?: string;
    readonly lifecycle?: "active" | "trashed";
    readonly deletedAt?: string;
    readonly updatedAt?: string;
  } = {},
): ResearchSession {
  const lifecycle = options.lifecycle ?? "active";
  const updatedAt = options.updatedAt ?? new Date().toISOString();
  return Object.freeze({
    id: options.id ?? randomId("session"),
    title: title.trim() || "未命名研究",
    ...(options.graphVersionId ? { graphVersionId: options.graphVersionId } : {}),
    lifecycle,
    ...(lifecycle === "trashed" ? { deletedAt: options.deletedAt ?? updatedAt } : {}),
    updatedAt,
  });
}

function normalizeSession(session: ResearchSession): ResearchSession {
  // Versions before the compact header appended this presentation-only suffix
  // to persisted titles. Strip it at the repository boundary so existing
  // browser databases immediately adopt the new naming without a destructive
  // migration.
  const title = session.title.replace(/\s*·\s*图谱研究\s*$/u, "").trim();
  return createResearchSession(title, {
    id: session.id,
    ...(session.graphVersionId ? { graphVersionId: session.graphVersionId } : {}),
    lifecycle: session.lifecycle ?? "active",
    ...(session.deletedAt ? { deletedAt: session.deletedAt } : {}),
    updatedAt: session.updatedAt,
  });
}

/**
 * IndexedDB repository with a mirrored in-memory copy. If IndexedDB is absent,
 * blocked, private, or becomes unavailable, all methods continue to work for
 * the lifetime of the page.
 */
export class LocalGraphRepository implements GraphRepository {
  private database?: SocialGraphDatabase;
  private readonly graphVersions = new Map<string, GraphVersion>();
  private readonly graphVersionManifests = new Map<string, GraphVersionManifest>();
  private readonly resourceLifecycles = new Map<string, ResourceLifecycle>();
  private readonly viewStates = new Map<string, GraphViewState>();
  private readonly analysisRuns = new Map<string, AnalysisRun>();
  private readonly semanticEvents = new Map<string, SemanticEvent>();
  private readonly sessions = new Map<string, ResearchSession>();
  private readonly messages = new Map<string, ConversationMessage>();
  private readonly sourceArtifacts = new Map<string, SourceArtifact>();
  private readonly changeListeners = new Set<(change: RepositoryChange) => void>();
  private readonly originId = randomId("repository");
  private changeChannel?: BroadcastChannel;
  private manifestBackfillPromise?: Promise<void>;
  private persistentStorageFailed = false;
  private disposed = false;
  private initializationMetadata?: RepositoryInitializationMetadata;

  constructor(options: LocalGraphRepositoryOptions = {}) {
    const indexedDbAvailable = !options.forceMemory && typeof indexedDB !== "undefined";
    const databaseName = options.databaseName ?? "socialgraph-fm";
    if (indexedDbAvailable) {
      this.database = new SocialGraphDatabase(databaseName);
    }
    if (
      indexedDbAvailable
      && typeof window !== "undefined"
      && typeof BroadcastChannel !== "undefined"
    ) {
      this.changeChannel = new BroadcastChannel(`${databaseName}:repository-changes`);
      this.changeChannel.onmessage = (event: MessageEvent<RepositoryChange>) => {
        const change = event.data;
        if (!change || change.originId === this.originId) return;
        for (const listener of this.changeListeners) listener(change);
      };
    }
  }

  get storageMode(): "indexeddb" | "memory" {
    return this.database ? "indexeddb" : "memory";
  }

  subscribe(listener: (change: RepositoryChange) => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }

  dispose(): void {
    this.disposed = true;
    this.changeChannel?.close();
    this.changeChannel = undefined;
    this.database?.close();
    this.changeListeners.clear();
  }

  private publishChange(kind: RepositoryChangeKind, ids: readonly string[]) {
    const change: RepositoryChange = Object.freeze({
      kind,
      ids: Object.freeze([...ids]),
      createdAt: new Date().toISOString(),
      originId: this.originId,
    });
    for (const listener of this.changeListeners) listener(change);
    this.changeChannel?.postMessage(change);
  }

  private async fromDatabase<T>(operation: (database: SocialGraphDatabase) => Promise<T>): Promise<T | undefined> {
    if (!this.database) return undefined;
    try {
      return await operation(this.database);
    } catch {
      this.database.close();
      this.database = undefined;
      this.persistentStorageFailed = true;
      return undefined;
    }
  }

  private async mirrorWrite(operation: (database: SocialGraphDatabase) => Promise<unknown>): Promise<void> {
    await this.fromDatabase(operation);
  }

  /** Destructive writes must never silently fall back after a failed persistent transaction. */
  private async strictDatabaseWrite<T>(
    operation: (database: SocialGraphDatabase) => Promise<T>,
  ): Promise<T | undefined> {
    if (this.disposed) throw new Error("REPOSITORY_DISPOSED：仓库已经关闭。");
    if (!this.database) {
      if (this.persistentStorageFailed) {
        throw new Error("PERSISTENT_STORAGE_UNAVAILABLE：持久存储已失败，未执行破坏性操作。");
      }
      return undefined;
    }
    try {
      return await operation(this.database);
    } catch (error) {
      if (error instanceof Error && error.message.includes("：")) throw error;
      throw new Error(
        `PERSISTENT_TRANSACTION_FAILED：${error instanceof Error ? error.message : "IndexedDB 写入失败"}`,
        { cause: error },
      );
    }
  }

  private assertDestructiveStorageAvailable() {
    if (this.disposed) throw new Error("REPOSITORY_DISPOSED：仓库已经关闭。");
    if (this.persistentStorageFailed) {
      throw new Error("PERSISTENT_STORAGE_UNAVAILABLE：持久存储已失败，未执行破坏性操作。");
    }
  }

  private async ensureGraphVersionManifests(): Promise<void> {
    if (this.manifestBackfillPromise) return this.manifestBackfillPromise;
    this.manifestBackfillPromise = (async () => {
      const backfilled = await this.fromDatabase(async (database) => {
        const marker = await database.metadata.get("graph-version-manifest-v4");
        if (marker?.key === "graph-version-manifest-v4") return [] as GraphVersionManifest[];
        return database.transaction(
          "rw",
          [database.graphVersions, database.graphVersionManifests, database.metadata],
          async () => {
            const versions = await database.graphVersions.toArray();
            const manifests = versions.map(deriveGraphVersionManifest);
            if (manifests.length) await database.graphVersionManifests.bulkPut(manifests);
            await database.metadata.put({
              key: "graph-version-manifest-v4",
              completedAt: new Date().toISOString(),
            });
            return manifests;
          },
        );
      });
      for (const manifest of backfilled ?? []) this.graphVersionManifests.set(manifest.id, manifest);
      for (const version of this.graphVersions.values()) {
        if (!this.graphVersionManifests.has(version.id)) {
          this.graphVersionManifests.set(version.id, deriveGraphVersionManifest(version));
        }
      }
    })().catch((error) => {
      this.manifestBackfillPromise = undefined;
      throw error;
    });
    return this.manifestBackfillPromise;
  }

  async saveGraphVersion(version: GraphVersion): Promise<void> {
    const copy = cloneValue(version);
    const manifest = deriveGraphVersionManifest(copy);
    const existing = await this.getGraphVersion(copy.id);
    if (existing && JSON.stringify(existing) !== JSON.stringify(copy)) {
      throw new Error(`IMMUTABLE_GRAPH_VERSION_CONFLICT：${copy.id} 已存在且内容不同。`);
    }
    this.graphVersions.set(copy.id, copy);
    this.graphVersionManifests.set(copy.id, manifest);
    await this.mirrorWrite((database) => database.transaction(
      "rw",
      [database.graphVersions, database.graphVersionManifests],
      async () => {
        if (!existing) await database.graphVersions.add(copy);
        await database.graphVersionManifests.put(manifest);
      },
    ));
    this.publishChange("graph_saved", [copy.id]);
  }

  async saveSourceArtifact(artifact: SourceArtifact): Promise<void> {
    const copy = cloneValue(artifact);
    const existing = await this.getSourceArtifact(copy.id);
    if (existing && existing.sha256 !== copy.sha256) {
      throw new Error(`IMMUTABLE_SOURCE_ARTIFACT_CONFLICT：${copy.id} 已存在且哈希不同。`);
    }
    this.sourceArtifacts.set(copy.id, copy);
    if (!existing) await this.mirrorWrite((database) => database.sourceArtifacts.add(copy));
    this.publishChange("source_saved", [copy.id]);
  }

  async getSourceArtifact(id: string): Promise<SourceArtifact | undefined> {
    const stored = await this.fromDatabase((database) => database.sourceArtifacts.get(id));
    const value = stored ?? this.sourceArtifacts.get(id);
    return value ? cloneValue(value) : undefined;
  }

  async listSourceArtifacts(state: ResourceListState = "active"): Promise<readonly SourceArtifact[]> {
    const stored = await this.fromDatabase((database) => database.sourceArtifacts.toArray());
    const values = stored ?? [...this.sourceArtifacts.values()];
    const lifecycleRecords = await this.fromDatabase((database) => database.resourceLifecycles
      .where("kind")
      .equals("source_artifact")
      .toArray());
    const lifecycleById = new Map(
      (lifecycleRecords ?? [...this.resourceLifecycles.values()].filter((item) => item.kind === "source_artifact"))
        .map((item) => [item.targetId, item]),
    );
    return values
      .filter((artifact) => matchesResourceState(state, lifecycleById.get(artifact.id)))
      .map(cloneValue)
      .sort((left, right) => compareUnicodeCodePoints(right.createdAt, left.createdAt) || compareUnicodeCodePoints(left.id, right.id));
  }

  async saveImportBundle(bundle: GraphImportPersistenceBundle, guard?: () => boolean): Promise<void> {
    const ensureCurrent = () => {
      if (guard && !guard()) throw new Error("STALE_IMPORT_SESSION");
    };
    ensureCurrent();
    const artifacts = bundle.sourceArtifacts.map(cloneValue);
    const graphVersion = cloneValue(bundle.graphVersion);
    const manifest = deriveGraphVersionManifest(graphVersion);
    const viewState = cloneValue(bundle.viewState);
    const session = cloneValue(normalizeSession(bundle.session));
    const event = cloneValue(bundle.event);
    const existingGraph = await this.getGraphVersion(graphVersion.id);
    ensureCurrent();
    if (existingGraph && JSON.stringify(existingGraph) !== JSON.stringify(graphVersion)) {
      throw new Error(`IMMUTABLE_GRAPH_VERSION_CONFLICT：${graphVersion.id} 已存在且内容不同。`);
    }
    const newArtifacts: SourceArtifact[] = [];
    for (const artifact of artifacts) {
      const existing = await this.getSourceArtifact(artifact.id);
      ensureCurrent();
      if (existing && existing.sha256 !== artifact.sha256) {
        throw new Error(`IMMUTABLE_SOURCE_ARTIFACT_CONFLICT：${artifact.id} 已存在且哈希不同。`);
      }
      if (!existing) newArtifacts.push(artifact);
    }

    let linkedMessage: ConversationMessage | undefined;
    const persistentLinkedMessage = await this.strictDatabaseWrite((database) => database.transaction(
      "rw",
      [
        database.sourceArtifacts,
        database.graphVersions,
        database.graphVersionManifests,
        database.resourceLifecycles,
        database.viewStates,
        database.sessions,
        database.semanticEvents,
        database.messages,
      ],
      async () => {
        ensureCurrent();
        const sourceLifecycles = artifacts.length
          ? await database.resourceLifecycles.bulkGet(
            artifacts.map((artifact) => lifecycleKey("source_artifact", artifact.id)),
          )
          : [];
        ensureCurrent();
        if (sourceLifecycles.some((lifecycle) => lifecycle?.state === "trashed")) {
          throw new Error("SOURCE_ARTIFACT_TRASHED：请先恢复源文件，再创建新的图版本引用。");
        }
        if (newArtifacts.length) await database.sourceArtifacts.bulkAdd(newArtifacts);
        if (!existingGraph) await database.graphVersions.add(graphVersion);
        await database.graphVersionManifests.put(manifest);
        await database.viewStates.put(viewState);
        await database.sessions.put(session);
        await database.semanticEvents.put(event);
        ensureCurrent();
        if (!bundle.sourceMessageId) return undefined;
        const sourceMessage = await database.messages.get(bundle.sourceMessageId);
        ensureCurrent();
        if (!sourceMessage) throw new Error(`SOURCE_MESSAGE_NOT_FOUND：${bundle.sourceMessageId}`);
        const nextMessage = withMessageSourceArtifacts(sourceMessage, artifacts);
        await database.messages.put(nextMessage);
        ensureCurrent();
        return nextMessage;
      },
    ));
    ensureCurrent();

    if (!this.database && bundle.sourceMessageId) {
      const sourceMessage = this.messages.get(bundle.sourceMessageId);
      if (!sourceMessage) throw new Error(`SOURCE_MESSAGE_NOT_FOUND：${bundle.sourceMessageId}`);
      linkedMessage = withMessageSourceArtifacts(sourceMessage, artifacts);
    } else {
      linkedMessage = persistentLinkedMessage;
    }
    if (!this.database && artifacts.some((artifact) => (
      this.resourceLifecycles.get(lifecycleKey("source_artifact", artifact.id))?.state === "trashed"
    ))) {
      throw new Error("SOURCE_ARTIFACT_TRASHED：请先恢复源文件，再创建新的图版本引用。");
    }
    ensureCurrent();

    for (const artifact of artifacts) this.sourceArtifacts.set(artifact.id, artifact);
    this.graphVersions.set(graphVersion.id, graphVersion);
    this.graphVersionManifests.set(graphVersion.id, manifest);
    this.viewStates.set(viewState.graphVersionId, viewState);
    this.sessions.set(session.id, session);
    this.semanticEvents.set(event.id, event);
    if (linkedMessage) this.messages.set(linkedMessage.id, linkedMessage);
    this.publishChange("graph_saved", [graphVersion.id]);
    if (artifacts.length) this.publishChange("source_saved", artifacts.map((artifact) => artifact.id));
    if (linkedMessage) this.publishChange("message_changed", [linkedMessage.id]);
  }

  async getGraphVersion(id: string): Promise<GraphVersion | undefined> {
    const stored = await this.fromDatabase((database) => database.graphVersions.get(id));
    const value = stored ?? this.graphVersions.get(id);
    return value ? cloneValue(value) : undefined;
  }

  async getGraphVersionManifest(id: string): Promise<GraphVersionManifest | undefined> {
    await this.ensureGraphVersionManifests();
    const stored = await this.fromDatabase((database) => database.graphVersionManifests.get(id));
    const value = stored ?? this.graphVersionManifests.get(id);
    if (value) return cloneValue(value);
    const version = await this.getGraphVersion(id);
    if (!version) return undefined;
    const manifest = deriveGraphVersionManifest(version);
    this.graphVersionManifests.set(id, manifest);
    await this.mirrorWrite((database) => database.graphVersionManifests.put(manifest));
    return cloneValue(manifest);
  }

  async listGraphVersionManifests(state: ResourceListState = "active"): Promise<readonly GraphVersionManifest[]> {
    await this.ensureGraphVersionManifests();
    const [storedManifests, storedLifecycles] = await Promise.all([
      this.fromDatabase((database) => database.graphVersionManifests.toArray()),
      this.fromDatabase((database) => database.resourceLifecycles
        .where("kind")
        .equals("graph_version")
        .toArray()),
    ]);
    const manifests = storedManifests ?? [...this.graphVersionManifests.values()];
    const lifecycles = storedLifecycles
      ?? [...this.resourceLifecycles.values()].filter((item) => item.kind === "graph_version");
    const lifecycleById = new Map(lifecycles.map((item) => [item.targetId, item]));
    return manifests
      .filter((manifest) => matchesResourceState(state, lifecycleById.get(manifest.id)))
      .map(cloneValue)
      .sort((left, right) => compareUnicodeCodePoints(right.createdAt, left.createdAt) || compareUnicodeCodePoints(left.id, right.id));
  }

  async listGraphVersions(state: ResourceListState = "active"): Promise<readonly GraphVersion[]> {
    const manifests = await this.listGraphVersionManifests(state);
    const stored = await this.fromDatabase((database) => database.graphVersions.bulkGet(
      manifests.map((manifest) => manifest.id),
    ));
    const values = stored
      ? stored.filter((version): version is GraphVersion => Boolean(version))
      : manifests
        .map((manifest) => this.graphVersions.get(manifest.id))
        .filter((version): version is GraphVersion => Boolean(version));
    return values.map(cloneValue);
  }

  async getResourceLifecycle(kind: ManagedResourceKind, id: string): Promise<ResourceLifecycle> {
    const key = lifecycleKey(kind, id);
    const stored = await this.fromDatabase((database) => database.resourceLifecycles.get(key));
    return cloneValue(stored ?? this.resourceLifecycles.get(key) ?? activeLifecycle(kind, id));
  }

  private async graphDeletionImpactFromDatabase(
    database: SocialGraphDatabase,
    id: string,
  ): Promise<DeletionImpact | null> {
    const graph = await database.graphVersions.get(id);
    if (!graph) return null;
    const manifest = await database.graphVersionManifests.get(id) ?? deriveGraphVersionManifest(graph);
    const lifecycle = await database.resourceLifecycles.get(lifecycleKey("graph_version", id));
    const sessions = await database.sessions.where("graphVersionId").equals(id).toArray();
    const children = await database.graphVersions.where("parentVersionId").equals(id).toArray();
    const childLifecycles = children.length
      ? await database.resourceLifecycles.bulkGet(children.map((child) => lifecycleKey("graph_version", child.id)))
      : [];
    const runs = await database.analysisRuns.where("graphVersionId").equals(id).toArray();
    const messages = runs.length
      ? await database.messages.where("analysisRunId").anyOf(runs.map((run) => run.id)).toArray()
      : [];
    const events = await database.semanticEvents.where("graphVersionId").equals(id).toArray();
    const viewState = await database.viewStates.get(id);
    const sourceArtifacts = manifest.sourceArtifactIds.length
      ? await database.sourceArtifacts.bulkGet([...manifest.sourceArtifactIds])
      : [];
    const references: DeletionReference[] = [
      ...sessions.map((session) => ({
        kind: session.lifecycle === "trashed" ? "trashed_session" as const : "active_session" as const,
        id: session.id,
        label: session.title,
        blocksTrash: true,
        blocksPurge: true,
      })),
      ...children.map((child, index) => {
        const childState = childLifecycles[index]?.state ?? "active";
        return {
          kind: childState === "trashed" ? "trashed_child_version" as const : "active_child_version" as const,
          id: child.id,
          label: child.sourceFile,
          blocksTrash: childState === "active",
          blocksPurge: true,
        };
      }),
      ...messages.map((message) => ({
        kind: "analysis_message" as const,
        id: message.id,
        label: message.text.slice(0, 80) || message.id,
        blocksTrash: false,
        blocksPurge: true,
      })),
    ];
    const dependents: DeletionDependentGroup[] = [
      { kind: "manifest", count: 1, ids: [manifest.id] },
      ...(lifecycle ? [{ kind: "lifecycle" as const, count: 1, ids: [lifecycle.key] }] : []),
      ...(viewState ? [{ kind: "view_state" as const, count: 1, ids: [viewState.graphVersionId] }] : []),
      ...(runs.length ? [{ kind: "analysis_run" as const, count: runs.length, ids: runs.map((run) => run.id) }] : []),
      ...(events.length ? [{ kind: "semantic_event" as const, count: events.length, ids: events.map((event) => event.id) }] : []),
    ];
    const retainedDependencies: RetainedDependency[] = [
      ...manifest.sourceArtifactIds.map((sourceArtifactId, index) => ({
        kind: "source_artifact" as const,
        id: sourceArtifactId,
        label: sourceArtifacts[index]?.name ?? sourceArtifactId,
      })),
      ...(graph.datasetArtifact ? [{
        kind: "dataset_artifact" as const,
        id: graph.datasetArtifact.id,
        label: graph.datasetArtifact.datasetName,
      }] : []),
    ];
    const state = lifecycle?.state ?? "active";
    return finalizeDeletionImpact({
      targetKind: "graph_version",
      targetId: id,
      targetLabel: graph.sourceFile,
      state,
      references,
      dependents,
      retainedDependencies,
      canTrash: state === "active" && !references.some((reference) => reference.blocksTrash),
      canPurge: state === "trashed" && !references.some((reference) => reference.blocksPurge),
    });
  }

  private graphDeletionImpactFromMemory(id: string): DeletionImpact | null {
    const graph = this.graphVersions.get(id);
    if (!graph) return null;
    const manifest = this.graphVersionManifests.get(id) ?? deriveGraphVersionManifest(graph);
    const lifecycle = this.resourceLifecycles.get(lifecycleKey("graph_version", id));
    const sessions = [...this.sessions.values()].filter((session) => session.graphVersionId === id);
    const children = [...this.graphVersions.values()].filter((version) => version.parentVersionId === id);
    const runs = [...this.analysisRuns.values()].filter((run) => run.graphVersionId === id);
    const runIds = new Set(runs.map((run) => run.id));
    const messages = [...this.messages.values()].filter((message) => Boolean(message.analysisRunId && runIds.has(message.analysisRunId)));
    const events = [...this.semanticEvents.values()].filter((event) => event.graphVersionId === id);
    const viewState = this.viewStates.get(id);
    const references: DeletionReference[] = [
      ...sessions.map((session) => ({
        kind: session.lifecycle === "trashed" ? "trashed_session" as const : "active_session" as const,
        id: session.id,
        label: session.title,
        blocksTrash: true,
        blocksPurge: true,
      })),
      ...children.map((child) => {
        const childState = this.resourceLifecycles.get(lifecycleKey("graph_version", child.id))?.state ?? "active";
        return {
          kind: childState === "trashed" ? "trashed_child_version" as const : "active_child_version" as const,
          id: child.id,
          label: child.sourceFile,
          blocksTrash: childState === "active",
          blocksPurge: true,
        };
      }),
      ...messages.map((message) => ({
        kind: "analysis_message" as const,
        id: message.id,
        label: message.text.slice(0, 80) || message.id,
        blocksTrash: false,
        blocksPurge: true,
      })),
    ];
    const dependents: DeletionDependentGroup[] = [
      { kind: "manifest", count: 1, ids: [manifest.id] },
      ...(lifecycle ? [{ kind: "lifecycle" as const, count: 1, ids: [lifecycle.key] }] : []),
      ...(viewState ? [{ kind: "view_state" as const, count: 1, ids: [viewState.graphVersionId] }] : []),
      ...(runs.length ? [{ kind: "analysis_run" as const, count: runs.length, ids: runs.map((run) => run.id) }] : []),
      ...(events.length ? [{ kind: "semantic_event" as const, count: events.length, ids: events.map((event) => event.id) }] : []),
    ];
    const retainedDependencies: RetainedDependency[] = [
      ...manifest.sourceArtifactIds.map((sourceArtifactId) => ({
        kind: "source_artifact" as const,
        id: sourceArtifactId,
        label: this.sourceArtifacts.get(sourceArtifactId)?.name ?? sourceArtifactId,
      })),
      ...(graph.datasetArtifact ? [{
        kind: "dataset_artifact" as const,
        id: graph.datasetArtifact.id,
        label: graph.datasetArtifact.datasetName,
      }] : []),
    ];
    const state = lifecycle?.state ?? "active";
    return finalizeDeletionImpact({
      targetKind: "graph_version",
      targetId: id,
      targetLabel: graph.sourceFile,
      state,
      references,
      dependents,
      retainedDependencies,
      canTrash: state === "active" && !references.some((reference) => reference.blocksTrash),
      canPurge: state === "trashed" && !references.some((reference) => reference.blocksPurge),
    });
  }

  async inspectGraphVersionDeletion(id: string): Promise<DeletionImpact> {
    await this.ensureGraphVersionManifests();
    const impact = this.database
      ? await this.strictDatabaseWrite((database) => this.graphDeletionImpactFromDatabase(database, id))
      : this.graphDeletionImpactFromMemory(id);
    if (!impact) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
    return cloneValue(impact);
  }

  private assertExpectedImpact(impact: DeletionImpact, expectedImpactHash?: string) {
    if (expectedImpactHash && impact.impactHash !== expectedImpactHash) {
      throw new Error("REFERENCE_SET_CHANGED：引用关系已经变化，请重新检查后再操作。");
    }
  }

  async trashGraphVersion(id: string, expectedImpactHash?: string): Promise<void> {
    await this.ensureGraphVersionManifests();
    this.assertDestructiveStorageAvailable();
    const nextLifecycle = createLifecycle("graph_version", id, "trashed");
    if (this.database) {
      await this.strictDatabaseWrite((database) => database.transaction(
        "rw",
        [
          database.graphVersions,
          database.graphVersionManifests,
          database.resourceLifecycles,
          database.sessions,
          database.analysisRuns,
          database.messages,
          database.semanticEvents,
          database.viewStates,
          database.sourceArtifacts,
        ],
        async () => {
          const impact = await this.graphDeletionImpactFromDatabase(database, id);
          if (!impact) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
          this.assertExpectedImpact(impact, expectedImpactHash);
          if (!impact.canTrash) throw new Error("GRAPH_VERSION_IN_USE：该版本仍有会话或活动子版本引用。");
          await database.resourceLifecycles.put(nextLifecycle);
        },
      ));
    } else {
      const impact = this.graphDeletionImpactFromMemory(id);
      if (!impact) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
      this.assertExpectedImpact(impact, expectedImpactHash);
      if (!impact.canTrash) throw new Error("GRAPH_VERSION_IN_USE：该版本仍有会话或活动子版本引用。");
    }
    this.resourceLifecycles.set(nextLifecycle.key, nextLifecycle);
    this.publishChange("graph_trashed", [id]);
  }

  async restoreGraphVersion(id: string): Promise<void> {
    const graph = await this.getGraphVersion(id);
    if (!graph) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
    const nextLifecycle = createLifecycle("graph_version", id, "active");
    await this.strictDatabaseWrite((database) => database.transaction(
      "rw",
      [database.graphVersions, database.resourceLifecycles],
      async () => {
        if (!await database.graphVersions.get(id)) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
        await database.resourceLifecycles.put(nextLifecycle);
      },
    ));
    this.resourceLifecycles.set(nextLifecycle.key, nextLifecycle);
    this.publishChange("graph_restored", [id]);
  }

  async purgeGraphVersion(id: string, expectedImpactHash?: string): Promise<void> {
    await this.ensureGraphVersionManifests();
    this.assertDestructiveStorageAvailable();
    if (this.database) {
      await this.strictDatabaseWrite((database) => database.transaction(
        "rw",
        [
          database.graphVersions,
          database.graphVersionManifests,
          database.resourceLifecycles,
          database.sessions,
          database.analysisRuns,
          database.messages,
          database.semanticEvents,
          database.viewStates,
          database.sourceArtifacts,
        ],
        async () => {
          const impact = await this.graphDeletionImpactFromDatabase(database, id);
          if (!impact) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
          this.assertExpectedImpact(impact, expectedImpactHash);
          if (!impact.canPurge) throw new Error("GRAPH_VERSION_PURGE_BLOCKED：版本必须先进入回收站，且不能仍被引用。");
          await database.analysisRuns.where("graphVersionId").equals(id).delete();
          await database.semanticEvents.where("graphVersionId").equals(id).delete();
          await database.viewStates.delete(id);
          await database.resourceLifecycles.delete(lifecycleKey("graph_version", id));
          await database.graphVersionManifests.delete(id);
          await database.graphVersions.delete(id);
        },
      ));
    } else {
      const impact = this.graphDeletionImpactFromMemory(id);
      if (!impact) throw new Error(`GRAPH_VERSION_NOT_FOUND：${id}`);
      this.assertExpectedImpact(impact, expectedImpactHash);
      if (!impact.canPurge) throw new Error("GRAPH_VERSION_PURGE_BLOCKED：版本必须先进入回收站，且不能仍被引用。");
    }
    this.graphVersions.delete(id);
    this.graphVersionManifests.delete(id);
    this.viewStates.delete(id);
    this.resourceLifecycles.delete(lifecycleKey("graph_version", id));
    for (const [runId, run] of this.analysisRuns) if (run.graphVersionId === id) this.analysisRuns.delete(runId);
    for (const [eventId, event] of this.semanticEvents) if (event.graphVersionId === id) this.semanticEvents.delete(eventId);
    this.publishChange("graph_purged", [id]);
  }

  private async sourceDeletionImpactFromDatabase(
    database: SocialGraphDatabase,
    id: string,
  ): Promise<DeletionImpact | null> {
    const artifact = await database.sourceArtifacts.get(id);
    if (!artifact) return null;
    const lifecycle = await database.resourceLifecycles.get(lifecycleKey("source_artifact", id));
    const manifests = await database.graphVersionManifests.where("sourceArtifactIds").equals(id).toArray();
    const messages = await database.messages.filter((message) => messageSourceArtifactIds(message).includes(id)).toArray();
    const references: DeletionReference[] = [
      ...manifests.map((manifest) => ({
        kind: "graph_version" as const,
        id: manifest.id,
        label: manifest.sourceFile,
        blocksTrash: true,
        blocksPurge: true,
      })),
      ...messages.map((message) => ({
        kind: "source_message" as const,
        id: message.id,
        label: message.text.slice(0, 80) || message.id,
        blocksTrash: true,
        blocksPurge: true,
      })),
    ];
    const state = lifecycle?.state ?? "active";
    return finalizeDeletionImpact({
      targetKind: "source_artifact",
      targetId: id,
      targetLabel: artifact.name,
      state,
      references,
      dependents: lifecycle ? [{ kind: "lifecycle", count: 1, ids: [lifecycle.key] }] : [],
      retainedDependencies: [],
      canTrash: state === "active" && references.length === 0,
      canPurge: state === "trashed" && references.length === 0,
    });
  }

  private sourceDeletionImpactFromMemory(id: string): DeletionImpact | null {
    const artifact = this.sourceArtifacts.get(id);
    if (!artifact) return null;
    const lifecycle = this.resourceLifecycles.get(lifecycleKey("source_artifact", id));
    const manifests = [...this.graphVersionManifests.values()].filter((manifest) => manifest.sourceArtifactIds.includes(id));
    const messages = [...this.messages.values()].filter((message) => messageSourceArtifactIds(message).includes(id));
    const references: DeletionReference[] = [
      ...manifests.map((manifest) => ({
        kind: "graph_version" as const,
        id: manifest.id,
        label: manifest.sourceFile,
        blocksTrash: true,
        blocksPurge: true,
      })),
      ...messages.map((message) => ({
        kind: "source_message" as const,
        id: message.id,
        label: message.text.slice(0, 80) || message.id,
        blocksTrash: true,
        blocksPurge: true,
      })),
    ];
    const state = lifecycle?.state ?? "active";
    return finalizeDeletionImpact({
      targetKind: "source_artifact",
      targetId: id,
      targetLabel: artifact.name,
      state,
      references,
      dependents: lifecycle ? [{ kind: "lifecycle", count: 1, ids: [lifecycle.key] }] : [],
      retainedDependencies: [],
      canTrash: state === "active" && references.length === 0,
      canPurge: state === "trashed" && references.length === 0,
    });
  }

  async inspectSourceArtifactDeletion(id: string): Promise<DeletionImpact> {
    await this.ensureGraphVersionManifests();
    const impact = this.database
      ? await this.strictDatabaseWrite((database) => this.sourceDeletionImpactFromDatabase(database, id))
      : this.sourceDeletionImpactFromMemory(id);
    if (!impact) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
    return cloneValue(impact);
  }

  async trashSourceArtifact(id: string, expectedImpactHash?: string): Promise<void> {
    await this.ensureGraphVersionManifests();
    this.assertDestructiveStorageAvailable();
    const nextLifecycle = createLifecycle("source_artifact", id, "trashed");
    if (this.database) {
      await this.strictDatabaseWrite((database) => database.transaction(
        "rw",
        [database.sourceArtifacts, database.graphVersionManifests, database.resourceLifecycles, database.messages],
        async () => {
          const impact = await this.sourceDeletionImpactFromDatabase(database, id);
          if (!impact) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
          this.assertExpectedImpact(impact, expectedImpactHash);
          if (!impact.canTrash) throw new Error("SOURCE_ARTIFACT_IN_USE：仍有版本或消息引用该源文件。");
          await database.resourceLifecycles.put(nextLifecycle);
        },
      ));
    } else {
      const impact = this.sourceDeletionImpactFromMemory(id);
      if (!impact) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
      this.assertExpectedImpact(impact, expectedImpactHash);
      if (!impact.canTrash) throw new Error("SOURCE_ARTIFACT_IN_USE：仍有版本或消息引用该源文件。");
    }
    this.resourceLifecycles.set(nextLifecycle.key, nextLifecycle);
    this.publishChange("source_trashed", [id]);
  }

  async restoreSourceArtifact(id: string): Promise<void> {
    const artifact = await this.getSourceArtifact(id);
    if (!artifact) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
    const nextLifecycle = createLifecycle("source_artifact", id, "active");
    await this.strictDatabaseWrite((database) => database.transaction(
      "rw",
      [database.sourceArtifacts, database.resourceLifecycles],
      async () => {
        if (!await database.sourceArtifacts.get(id)) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
        await database.resourceLifecycles.put(nextLifecycle);
      },
    ));
    this.resourceLifecycles.set(nextLifecycle.key, nextLifecycle);
    this.publishChange("source_restored", [id]);
  }

  private async purgeSourceArtifactInternal(
    id: string,
    expectedImpactHash: string | undefined,
    requireTrash: boolean,
  ): Promise<void> {
    await this.ensureGraphVersionManifests();
    this.assertDestructiveStorageAvailable();
    if (this.database) {
      await this.strictDatabaseWrite((database) => database.transaction(
        "rw",
        [database.sourceArtifacts, database.graphVersionManifests, database.resourceLifecycles, database.messages],
        async () => {
          const impact = await this.sourceDeletionImpactFromDatabase(database, id);
          if (!impact) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
          this.assertExpectedImpact(impact, expectedImpactHash);
          const allowed = requireTrash ? impact.canPurge : impact.references.length === 0;
          if (!allowed) throw new Error("SOURCE_ARTIFACT_IN_USE：仍有版本或消息引用该源文件。");
          await database.resourceLifecycles.delete(lifecycleKey("source_artifact", id));
          await database.sourceArtifacts.delete(id);
        },
      ));
    } else {
      const impact = this.sourceDeletionImpactFromMemory(id);
      if (!impact) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${id}`);
      this.assertExpectedImpact(impact, expectedImpactHash);
      const allowed = requireTrash ? impact.canPurge : impact.references.length === 0;
      if (!allowed) throw new Error("SOURCE_ARTIFACT_IN_USE：仍有版本或消息引用该源文件。");
    }
    this.sourceArtifacts.delete(id);
    this.resourceLifecycles.delete(lifecycleKey("source_artifact", id));
    this.publishChange("source_purged", [id]);
  }

  async purgeSourceArtifact(id: string, expectedImpactHash?: string): Promise<void> {
    return this.purgeSourceArtifactInternal(id, expectedImpactHash, true);
  }

  /** Backwards-compatible permanent-delete path used by the existing diagnostics UI. */
  async deleteSourceArtifact(id: string): Promise<void> {
    return this.purgeSourceArtifactInternal(id, undefined, false);
  }

  async saveViewState(state: GraphViewState): Promise<void> {
    const copy = cloneValue(state);
    this.viewStates.set(copy.graphVersionId, copy);
    await this.mirrorWrite((database) => database.viewStates.put(copy));
  }

  async getViewState(graphVersionId: string): Promise<GraphViewState | undefined> {
    const stored = await this.fromDatabase((database) => database.viewStates.get(graphVersionId));
    const value = stored ?? this.viewStates.get(graphVersionId);
    return value ? cloneValue(value) : undefined;
  }

  async saveAnalysisRun(run: AnalysisRun): Promise<void> {
    const copy = cloneValue(run);
    this.analysisRuns.set(copy.id, copy);
    await this.mirrorWrite((database) => database.analysisRuns.put(copy));
  }

  async getAnalysisRun(id: string): Promise<AnalysisRun | undefined> {
    const stored = await this.fromDatabase((database) => database.analysisRuns.get(id));
    const value = stored ?? this.analysisRuns.get(id);
    return value ? cloneValue(value) : undefined;
  }

  async listAnalysisRuns(graphVersionId?: string): Promise<readonly AnalysisRun[]> {
    const stored = await this.fromDatabase((database) =>
      graphVersionId
        ? database.analysisRuns.where("graphVersionId").equals(graphVersionId).toArray()
        : database.analysisRuns.toArray(),
    );
    const values = stored ?? [...this.analysisRuns.values()].filter(
      (run) => !graphVersionId || run.graphVersionId === graphVersionId,
    );
    return values
      .map(cloneValue)
      .sort((left, right) => compareUnicodeCodePoints(right.createdAt, left.createdAt) || compareUnicodeCodePoints(left.id, right.id));
  }

  async saveEvent(event: SemanticEvent): Promise<void> {
    const copy = cloneValue(event);
    this.semanticEvents.set(copy.id, copy);
    await this.mirrorWrite((database) => database.semanticEvents.put(copy));
  }

  async appendEvent(event: SemanticEvent): Promise<void> {
    const copy = cloneValue(event);
    if (this.semanticEvents.has(copy.id)) {
      throw new Error(`SEMANTIC_EVENT_ALREADY_EXISTS：${copy.id}`);
    }
    if (!this.database) {
      this.assertDestructiveStorageAvailable();
      this.semanticEvents.set(copy.id, copy);
      this.publishChange("semantic_event_appended", [copy.id]);
      return;
    }
    await this.strictDatabaseWrite((database) => database.transaction(
      "rw",
      database.semanticEvents,
      async () => {
        if (await database.semanticEvents.get(copy.id)) {
          throw new Error(`SEMANTIC_EVENT_ALREADY_EXISTS：${copy.id}`);
        }
        await database.semanticEvents.add(copy);
      },
    ));
    this.semanticEvents.set(copy.id, copy);
    this.publishChange("semantic_event_appended", [copy.id]);
  }

  async listEvents(graphVersionId?: string): Promise<readonly SemanticEvent[]> {
    const stored = await this.fromDatabase((database) =>
      graphVersionId
        ? database.semanticEvents.where("graphVersionId").equals(graphVersionId).toArray()
        : database.semanticEvents.toArray(),
    );
    const values = stored ?? [...this.semanticEvents.values()].filter(
      (event) => !graphVersionId || event.graphVersionId === graphVersionId,
    );
    return values
      .map(cloneValue)
      .sort((left, right) => compareUnicodeCodePoints(left.createdAt, right.createdAt) || compareUnicodeCodePoints(left.id, right.id));
  }

  async saveSession(session: ResearchSession): Promise<void> {
    const copy = cloneValue(normalizeSession(session));
    this.sessions.set(copy.id, copy);
    await this.mirrorWrite((database) => database.sessions.put(copy));
    this.publishChange("session_changed", [copy.id]);
  }

  async getSession(id: string): Promise<ResearchSession | undefined> {
    const stored = await this.fromDatabase((database) => database.sessions.get(id));
    const value = stored ?? this.sessions.get(id);
    return value ? cloneValue(normalizeSession(value)) : undefined;
  }

  async listSessions(state: SessionListState = "active"): Promise<readonly ResearchSession[]> {
    const stored = await this.fromDatabase((database) =>
      state === "all"
        ? database.sessions.toArray()
        : database.sessions.where("lifecycle").equals(state).toArray(),
    );
    const values = (stored ?? [...this.sessions.values()])
      .map(normalizeSession)
      .filter((session) => state === "all" || session.lifecycle === state);
    return values
      .map(cloneValue)
      .sort((left, right) => compareUnicodeCodePoints(right.updatedAt, left.updatedAt) || compareUnicodeCodePoints(left.id, right.id));
  }

  async saveMessage(message: ConversationMessage): Promise<void> {
    const copy = cloneValue(message);
    this.messages.set(copy.id, copy);
    await this.mirrorWrite((database) => database.messages.put(copy));
    this.publishChange("message_changed", [copy.id]);
  }

  async listMessages(sessionId: string): Promise<readonly ConversationMessage[]> {
    const stored = await this.fromDatabase((database) =>
      database.messages.where("sessionId").equals(sessionId).toArray(),
    );
    const values = stored ?? [...this.messages.values()].filter((message) => message.sessionId === sessionId);
    return values
      .map(cloneValue)
      .sort((left, right) => compareUnicodeCodePoints(left.createdAt, right.createdAt) || compareUnicodeCodePoints(left.id, right.id));
  }

  async deleteMessage(messageId: string): Promise<void> {
    await this.strictDatabaseWrite((database) => database.messages.delete(messageId));
    this.messages.delete(messageId);
    this.publishChange("message_changed", [messageId]);
  }

  async associateMessageSourceArtifacts(
    messageId: string,
    sourceArtifactIds: readonly string[],
  ): Promise<void> {
    const ids = [...new Set(sourceArtifactIds)].sort(compareUnicodeCodePoints);
    let nextMessage: ConversationMessage | undefined;
    const stored = await this.strictDatabaseWrite((database) => database.transaction(
      "rw",
      [database.messages, database.sourceArtifacts, database.resourceLifecycles],
      async () => {
        const [message, artifacts, lifecycles] = await Promise.all([
          database.messages.get(messageId),
          database.sourceArtifacts.bulkGet(ids),
          database.resourceLifecycles.bulkGet(ids.map((id) => lifecycleKey("source_artifact", id))),
        ]);
        if (!message) throw new Error(`SOURCE_MESSAGE_NOT_FOUND：${messageId}`);
        const missing = ids.filter((_id, index) => !artifacts[index]);
        if (missing.length) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${missing.join(", ")}`);
        if (lifecycles.some((lifecycle) => lifecycle?.state === "trashed")) {
          throw new Error("SOURCE_ARTIFACT_TRASHED：请先恢复源文件，再关联消息。");
        }
        const linked = withMessageSourceArtifacts(message, ids.map((id) => ({ id })));
        await database.messages.put(linked);
        return linked;
      },
    ));
    if (!this.database) {
      const message = this.messages.get(messageId);
      if (!message) throw new Error(`SOURCE_MESSAGE_NOT_FOUND：${messageId}`);
      const missing = ids.filter((id) => !this.sourceArtifacts.has(id));
      if (missing.length) throw new Error(`SOURCE_ARTIFACT_NOT_FOUND：${missing.join(", ")}`);
      if (ids.some((id) => this.resourceLifecycles.get(lifecycleKey("source_artifact", id))?.state === "trashed")) {
        throw new Error("SOURCE_ARTIFACT_TRASHED：请先恢复源文件，再关联消息。");
      }
      nextMessage = withMessageSourceArtifacts(message, ids.map((id) => ({ id })));
    } else {
      nextMessage = stored;
    }
    if (!nextMessage) throw new Error(`SOURCE_MESSAGE_NOT_FOUND：${messageId}`);
    this.messages.set(nextMessage.id, nextMessage);
    this.publishChange("message_changed", [messageId]);
  }

  async trashSession(id: string): Promise<void> {
    const session = await this.getSession(id);
    if (!session) return;
    const timestamp = new Date().toISOString();
    const next = createResearchSession(session.title, {
      id: session.id,
      ...(session.graphVersionId ? { graphVersionId: session.graphVersionId } : {}),
      lifecycle: "trashed",
      deletedAt: timestamp,
      updatedAt: timestamp,
    });
    await this.strictDatabaseWrite((database) => database.sessions.put(next));
    this.sessions.set(id, next);
    this.publishChange("session_changed", [id]);
  }

  async restoreSession(id: string): Promise<void> {
    const session = await this.getSession(id);
    if (!session) return;
    const next = createResearchSession(session.title, {
      id: session.id,
      ...(session.graphVersionId ? { graphVersionId: session.graphVersionId } : {}),
      lifecycle: "active",
      updatedAt: new Date().toISOString(),
    });
    await this.strictDatabaseWrite((database) => database.sessions.put(next));
    this.sessions.set(id, next);
    this.publishChange("session_changed", [id]);
  }

  async purgeSession(id: string): Promise<void> {
    await this.strictDatabaseWrite((database) => database.transaction(
      "rw",
      [database.sessions, database.messages, database.semanticEvents],
      async () => {
        await database.messages.where("sessionId").equals(id).delete();
        await database.semanticEvents.where("sessionId").equals(id).delete();
        await database.sessions.delete(id);
      },
    ));

    this.sessions.delete(id);
    for (const [messageId, message] of this.messages) {
      if (message.sessionId === id) this.messages.delete(messageId);
    }
    for (const [eventId, event] of this.semanticEvents) {
      if (event.sessionId === id) this.semanticEvents.delete(eventId);
    }
    this.publishChange("session_changed", [id]);
  }

  async getInitializationMetadata(): Promise<RepositoryInitializationMetadata | undefined> {
    const stored = await this.fromDatabase((database) => database.metadata.get("initialization"));
    const value = stored ?? this.initializationMetadata;
    if (!value || !("initializedAt" in value)) return undefined;
    const { initializedAt, seededDemoVersion, updatedAt } = value;
    return cloneValue({ initializedAt, seededDemoVersion, updatedAt });
  }

  async saveInitializationMetadata(metadata: RepositoryInitializationMetadata): Promise<void> {
    const copy = cloneValue(metadata);
    this.initializationMetadata = copy;
    await this.mirrorWrite((database) => database.metadata.put({
      key: "initialization",
      ...copy,
    }));
  }
}

export function createLocalGraphRepository(
  options: LocalGraphRepositoryOptions = {},
): LocalGraphRepository {
  return new LocalGraphRepository(options);
}
