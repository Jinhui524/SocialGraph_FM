import { describe, expect, it } from "vitest";

import type { AnalysisRun, ConversationMessage, GraphEdge, GraphNode } from "../types/graph";
import { createGraphVersion } from "./graphImport";
import { createSourceArtifact } from "./sourceArtifact";
import {
  createDefaultGraphViewState,
  createLocalGraphRepository,
  createResearchSession,
  createSemanticEvent,
} from "./graphRepository";

function fixtureGraph() {
  const nodes: GraphNode[] = [
    { id: "a", label: "A", attributes: {} },
    { id: "b", label: "B", attributes: {} },
  ];
  const edges: GraphEdge[] = [{ id: "ab", source: "a", target: "b", attributes: {} }];
  return createGraphVersion("repository.json", nodes, edges);
}

function fixtureRun(graphVersionId: string, id = "run-lifecycle"): AnalysisRun {
  return {
    id,
    graphVersionId,
    intent: {
      kind: "analysis_request",
      normalizedText: "生成图谱概览",
      task: "overview",
      targets: [],
      confidence: 1,
      filters: {},
      meta: { schemaVersion: "1.1", source: "deterministic_fallback", requestId: `${id}-request`, warnings: [] },
    },
    engine: "local_algorithm",
    status: "succeeded",
    createdAt: "2026-08-10T00:00:00.001Z",
  };
}

describe("LocalGraphRepository memory fallback", () => {
  it("deletes a confirmed planning message from the repository mirror", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const session = createResearchSession("confirmation", { id: "confirmation-session" });
    const message: ConversationMessage = {
      id: "planning-message",
      sessionId: session.id,
      role: "assistant",
      text: "分析计划已准备好",
      status: "warning",
      createdAt: "2026-08-21T00:00:00.000Z",
    };
    const deleteMessage = (repository as typeof repository & {
      deleteMessage?: (messageId: string) => Promise<void>;
    }).deleteMessage;

    await repository.saveSession(session);
    await repository.saveMessage(message);
    expect(deleteMessage).toBeTypeOf("function");
    if (!deleteMessage) return;
    await deleteMessage.call(repository, message.id);

    expect(await repository.listMessages(session.id)).toEqual([]);
  });

  it("persists immutable SourceArtifacts and rejects GraphVersion id collisions", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const file = new File(["source,target\na,b\n"], "edges.csv", { type: "text/csv" });
    const artifact = await createSourceArtifact(file);
    const graph = fixtureGraph();

    await repository.saveSourceArtifact(artifact);
    await repository.saveGraphVersion(graph);

    expect((await repository.getSourceArtifact(artifact.id))?.sha256).toBe(artifact.sha256);
    expect(await repository.listSourceArtifacts()).toHaveLength(1);
    await expect(repository.saveGraphVersion({ ...graph, sourceFile: "changed.json" })).rejects.toThrow(
      "IMMUTABLE_GRAPH_VERSION_CONFLICT",
    );
    await repository.deleteSourceArtifact(artifact.id);
    expect(await repository.listSourceArtifacts()).toEqual([]);
  });

  it("does not delete a SourceArtifact referenced by a GraphVersion", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const file = new File(["{}"], "graph.json", { type: "application/json" });
    const artifact = await createSourceArtifact(file);
    const base = fixtureGraph();
    const graph = { ...base, sourceArtifactIds: [artifact.id] };
    await repository.saveSourceArtifact(artifact);
    await repository.saveGraphVersion(graph);

    await expect(repository.deleteSourceArtifact(artifact.id)).rejects.toThrow("SOURCE_ARTIFACT_IN_USE");
  });

  it("commits an import bundle through one repository operation", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const file = new File(["source,target\na,b\n"], "edges.csv", { type: "text/csv" });
    const artifact = await createSourceArtifact(file);
    const base = fixtureGraph();
    const graph = { ...base, sourceArtifactIds: [artifact.id] };
    const viewState = createDefaultGraphViewState(graph.id);
    const session = createResearchSession("原子导入", { id: "bundle-session", graphVersionId: graph.id });
    const event = createSemanticEvent("graph_imported", {
      id: "bundle-event",
      graphVersionId: graph.id,
      sessionId: session.id,
    });

    await repository.saveImportBundle({
      sourceArtifacts: [artifact],
      graphVersion: graph,
      viewState,
      session,
      event,
    });

    expect(await repository.getSourceArtifact(artifact.id)).toBeDefined();
    expect(await repository.getGraphVersion(graph.id)).toEqual(graph);
    expect(await repository.getViewState(graph.id)).toEqual(viewState);
    expect(await repository.getSession(session.id)).toEqual(session);
    expect(await repository.listEvents(graph.id)).toEqual([event]);
  });

  it("does not commit an import bundle after its session guard becomes stale", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = { ...fixtureGraph(), id: "guarded-graph" };
    const viewState = createDefaultGraphViewState(graph.id);
    const session = createResearchSession("已切换会话", { id: "guarded-session", graphVersionId: graph.id });
    const event = createSemanticEvent("graph_imported", {
      id: "guarded-event",
      graphVersionId: graph.id,
      sessionId: session.id,
    });

    await expect(repository.saveImportBundle({
      sourceArtifacts: [],
      graphVersion: graph,
      viewState,
      session,
      event,
    }, () => false)).rejects.toThrow("STALE_IMPORT_SESSION");

    await expect(repository.getGraphVersion(graph.id)).resolves.toBeUndefined();
    await expect(repository.getViewState(graph.id)).resolves.toBeUndefined();
    await expect(repository.getSession(session.id)).resolves.toBeUndefined();
    await expect(repository.listEvents(graph.id)).resolves.toEqual([]);
  });

  it("persists graph, view, analysis, event, message and research session in memory", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    const view = createDefaultGraphViewState(graph.id);
    const run: AnalysisRun = {
      id: "run-1",
      graphVersionId: graph.id,
      intent: {
        kind: "analysis_request",
        normalizedText: "生成图谱概览",
        task: "overview",
        targets: [],
        confidence: 1,
        filters: {},
        meta: { schemaVersion: "1.1", source: "deterministic_fallback", requestId: "request-1", warnings: [] },
      },
      engine: "local_algorithm",
      status: "succeeded",
      createdAt: "2026-08-10T00:00:00.000Z",
      completedAt: "2026-08-10T00:00:00.001Z",
    };
    const session = createResearchSession("测试研究", {
      id: "session-1",
      graphVersionId: graph.id,
      updatedAt: "2026-08-10T00:00:00.000Z",
    });
    const event = createSemanticEvent("graph_imported", {
      id: "event-1",
      graphVersionId: graph.id,
      sessionId: session.id,
      createdAt: "2026-08-10T00:00:00.000Z",
      payload: { nodeCount: 2 },
    });
    const message: ConversationMessage = {
      id: "message-1",
      sessionId: session.id,
      role: "user",
      text: "请生成图谱概览",
      status: "completed",
      attachment: { name: "repository.json", size: 256, kind: "file" },
      createdAt: "2026-08-10T00:00:00.002Z",
    };

    await repository.saveGraphVersion(graph);
    await repository.saveViewState(view);
    await repository.saveAnalysisRun(run);
    await repository.saveSession(session);
    await repository.saveEvent(event);
    await repository.saveMessage(message);

    expect(repository.storageMode).toBe("memory");
    expect((await repository.getGraphVersion(graph.id))?.sourceFile).toBe("repository.json");
    expect(await repository.getViewState(graph.id)).toEqual(view);
    expect(await repository.getAnalysisRun(run.id)).toEqual(run);
    expect(await repository.listAnalysisRuns(graph.id)).toEqual([run]);
    expect(await repository.getSession(session.id)).toEqual(session);
    expect(await repository.listSessions()).toEqual([session]);
    expect(await repository.listEvents(graph.id)).toEqual([event]);
    expect(await repository.listMessages(session.id)).toEqual([message]);
  });

  it("returns clones so callers cannot change stored values", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    await repository.saveGraphVersion(graph);
    const loaded = await repository.getGraphVersion(graph.id);
    expect(loaded).not.toBe(graph);
    if (loaded) (loaded.nodes as GraphNode[]).push({ id: "c", label: "C", attributes: {} });
    expect((await repository.getGraphVersion(graph.id))?.nodes).toHaveLength(2);
  });

  it("moves sessions through the local trash without reviving deletedAt on restore", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const session = createResearchSession("可删除会话", {
      id: "session-trash",
      updatedAt: "2026-08-10T00:00:00.000Z",
    });
    await repository.saveSession(session);

    await repository.trashSession(session.id);
    expect(await repository.listSessions()).toEqual([]);
    const trashed = await repository.listSessions("trashed");
    expect(trashed).toHaveLength(1);
    expect(trashed[0]).toMatchObject({ id: session.id, lifecycle: "trashed" });
    expect(trashed[0]?.deletedAt).toBeTruthy();

    await repository.restoreSession(session.id);
    const restored = await repository.getSession(session.id);
    expect(restored).toMatchObject({ id: session.id, lifecycle: "active" });
    expect(restored?.deletedAt).toBeUndefined();
    expect(await repository.listSessions("trashed")).toEqual([]);
    expect(await repository.listSessions("all")).toEqual([restored]);
  });

  it("purges only session-owned records and preserves its formerly exclusive graph", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    const view = createDefaultGraphViewState(graph.id);
    const session = createResearchSession("独占图会话", {
      id: "session-exclusive",
      graphVersionId: graph.id,
      updatedAt: "2026-08-10T00:00:00.000Z",
    });
    const message: ConversationMessage = {
      id: "message-exclusive",
      sessionId: session.id,
      role: "assistant",
      text: "分析完成",
      status: "completed",
      createdAt: "2026-08-10T00:00:00.001Z",
    };
    const run: AnalysisRun = {
      id: "run-exclusive",
      graphVersionId: graph.id,
      intent: {
        kind: "analysis_request",
        normalizedText: "生成图谱概览",
        task: "overview",
        targets: [],
        confidence: 1,
        filters: {},
        meta: { schemaVersion: "1.1", source: "deterministic_fallback", requestId: "request-exclusive", warnings: [] },
      },
      engine: "local_algorithm",
      status: "succeeded",
      createdAt: "2026-08-10T00:00:00.001Z",
    };
    const event = createSemanticEvent("analysis_completed", {
      id: "event-exclusive",
      graphVersionId: graph.id,
      sessionId: session.id,
      createdAt: "2026-08-10T00:00:00.001Z",
    });

    await repository.saveGraphVersion(graph);
    await repository.saveViewState(view);
    await repository.saveAnalysisRun(run);
    await repository.saveSession(session);
    await repository.saveMessage(message);
    await repository.saveEvent(event);
    await repository.purgeSession(session.id);

    expect(await repository.getSession(session.id)).toBeUndefined();
    expect(await repository.listMessages(session.id)).toEqual([]);
    expect(await repository.getGraphVersion(graph.id)).toEqual(graph);
    expect(await repository.getViewState(graph.id)).toEqual(view);
    expect(await repository.listAnalysisRuns(graph.id)).toEqual([run]);
    expect(await repository.listEvents(graph.id)).toEqual([]);
  });

  it("keeps a shared graph and its graph-derived records when one session is purged", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    const view = createDefaultGraphViewState(graph.id);
    const first = createResearchSession("会话一", {
      id: "session-shared-1",
      graphVersionId: graph.id,
      updatedAt: "2026-08-10T00:00:00.000Z",
    });
    const second = createResearchSession("会话二", {
      id: "session-shared-2",
      graphVersionId: graph.id,
      updatedAt: "2026-08-10T00:00:00.001Z",
    });
    const firstEvent = createSemanticEvent("intent_applied", {
      id: "event-shared-session",
      graphVersionId: graph.id,
      sessionId: first.id,
    });
    const graphEvent = createSemanticEvent("graph_imported", {
      id: "event-shared-graph",
      graphVersionId: graph.id,
    });

    await repository.saveGraphVersion(graph);
    await repository.saveViewState(view);
    await repository.saveSession(first);
    await repository.saveSession(second);
    await repository.saveEvent(firstEvent);
    await repository.saveEvent(graphEvent);
    await repository.purgeSession(first.id);

    expect(await repository.getSession(first.id)).toBeUndefined();
    expect(await repository.getSession(second.id)).toEqual(second);
    expect(await repository.getGraphVersion(graph.id)).toEqual(graph);
    expect(await repository.getViewState(graph.id)).toEqual(view);
    expect(await repository.listEvents(graph.id)).toEqual([graphEvent]);
  });

  it("stores lightweight manifests and enforces conservative leaf-first version deletion", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const base = fixtureGraph();
    const child = createGraphVersion("repository-child.json", base.nodes, base.edges, [], {
      parentVersionId: base.id,
    });
    await repository.saveGraphVersion(base);
    await repository.saveGraphVersion(child);

    expect(await repository.getGraphVersionManifest(base.id)).toMatchObject({
      id: base.id,
      nodeCount: 2,
      edgeCount: 1,
    });
    const blockedBase = await repository.inspectGraphVersionDeletion(base.id);
    expect(blockedBase.canTrash).toBe(false);
    expect(blockedBase.references).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: "active_child_version", id: child.id }),
    ]));

    const childImpact = await repository.inspectGraphVersionDeletion(child.id);
    await repository.trashGraphVersion(child.id, childImpact.impactHash);
    expect(await repository.listGraphVersions()).toEqual([base]);
    expect(await repository.listGraphVersions("trashed")).toEqual([child]);

    const nowTrashableBase = await repository.inspectGraphVersionDeletion(base.id);
    expect(nowTrashableBase.canTrash).toBe(true);
    await repository.trashGraphVersion(base.id, nowTrashableBase.impactHash);
    expect((await repository.inspectGraphVersionDeletion(base.id)).canPurge).toBe(false);

    const trashedChildImpact = await repository.inspectGraphVersionDeletion(child.id);
    await repository.purgeGraphVersion(child.id, trashedChildImpact.impactHash);
    const purgeableBase = await repository.inspectGraphVersionDeletion(base.id);
    expect(purgeableBase.canPurge).toBe(true);
    await repository.purgeGraphVersion(base.id, purgeableBase.impactHash);

    expect(await repository.getGraphVersion(base.id)).toBeUndefined();
    expect(await repository.getGraphVersionManifest(base.id)).toBeUndefined();
    expect(await repository.listGraphVersions("all")).toEqual([]);
  });

  it("rejects stale deletion impacts after a new reference appears", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    await repository.saveGraphVersion(graph);
    const impact = await repository.inspectGraphVersionDeletion(graph.id);
    await repository.saveSession(createResearchSession("新增引用", {
      id: "late-session",
      graphVersionId: graph.id,
    }));

    await expect(repository.trashGraphVersion(graph.id, impact.impactHash)).rejects.toThrow("REFERENCE_SET_CHANGED");
    expect((await repository.getResourceLifecycle("graph_version", graph.id)).state).toBe("active");
  });

  it("restores a trashed version by changing only its lifecycle sidecar", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    const immutableBytes = JSON.stringify(graph);
    await repository.saveGraphVersion(graph);
    const impact = await repository.inspectGraphVersionDeletion(graph.id);
    await repository.trashGraphVersion(graph.id, impact.impactHash);
    expect((await repository.getResourceLifecycle("graph_version", graph.id)).state).toBe("trashed");

    await repository.restoreGraphVersion(graph.id);

    expect((await repository.getResourceLifecycle("graph_version", graph.id)).state).toBe("active");
    expect(JSON.stringify(await repository.getGraphVersion(graph.id))).toBe(immutableBytes);
    expect(await repository.listGraphVersions()).toEqual([graph]);
  });

  it("publishes repository changes for same-tab refresh subscribers", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    const changes: string[] = [];
    const unsubscribe = repository.subscribe((change) => changes.push(`${change.kind}:${change.ids.join(",")}`));
    await repository.saveGraphVersion(graph);
    const impact = await repository.inspectGraphVersionDeletion(graph.id);
    await repository.trashGraphVersion(graph.id, impact.impactHash);
    unsubscribe();
    await repository.restoreGraphVersion(graph.id);

    expect(changes).toEqual([
      `graph_saved:${graph.id}`,
      `graph_trashed:${graph.id}`,
    ]);
  });

  it("allows trash but blocks purge while an analysis message references a run", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = fixtureGraph();
    const run = fixtureRun(graph.id);
    const session = createResearchSession("历史分析消息", { id: "analysis-message-session" });
    const message: ConversationMessage = {
      id: "analysis-message",
      sessionId: session.id,
      role: "assistant",
      text: "分析证据",
      status: "completed",
      analysisRunId: run.id,
      createdAt: "2026-08-10T00:00:00.002Z",
    };
    await repository.saveGraphVersion(graph);
    await repository.saveAnalysisRun(run);
    await repository.saveSession(session);
    await repository.saveMessage(message);

    const activeImpact = await repository.inspectGraphVersionDeletion(graph.id);
    expect(activeImpact.canTrash).toBe(true);
    await repository.trashGraphVersion(graph.id, activeImpact.impactHash);
    expect((await repository.inspectGraphVersionDeletion(graph.id)).canPurge).toBe(false);

    await repository.purgeSession(session.id);
    const purgeable = await repository.inspectGraphVersionDeletion(graph.id);
    expect(purgeable.canPurge).toBe(true);
    await repository.purgeGraphVersion(graph.id, purgeable.impactHash);
    expect(await repository.listAnalysisRuns(graph.id)).toEqual([]);
  });

  it("retains SourceArtifacts when a version is purged and deletes them only through their own lifecycle", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const artifact = await createSourceArtifact(new File(["source,target\na,b\n"], "source.csv"));
    const graph = { ...fixtureGraph(), sourceArtifactIds: [artifact.id] };
    await repository.saveSourceArtifact(artifact);
    await repository.saveGraphVersion(graph);

    const blocked = await repository.inspectSourceArtifactDeletion(artifact.id);
    expect(blocked.canTrash).toBe(false);
    expect(blocked.references).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: "graph_version", id: graph.id }),
    ]));

    const graphImpact = await repository.inspectGraphVersionDeletion(graph.id);
    await repository.trashGraphVersion(graph.id, graphImpact.impactHash);
    const purgeImpact = await repository.inspectGraphVersionDeletion(graph.id);
    await repository.purgeGraphVersion(graph.id, purgeImpact.impactHash);
    expect(await repository.getSourceArtifact(artifact.id)).toBeDefined();

    const artifactImpact = await repository.inspectSourceArtifactDeletion(artifact.id);
    await repository.trashSourceArtifact(artifact.id, artifactImpact.impactHash);
    expect(await repository.listSourceArtifacts()).toEqual([]);
    expect(await repository.listSourceArtifacts("trashed")).toHaveLength(1);
    const trashedImpact = await repository.inspectSourceArtifactDeletion(artifact.id);
    await repository.purgeSourceArtifact(artifact.id, trashedImpact.impactHash);
    expect(await repository.getSourceArtifact(artifact.id)).toBeUndefined();
  });

  it("associates message attachments with SourceArtifacts for reference-aware deletion", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const artifact = await createSourceArtifact(new File(["{}"], "message.json", { type: "application/json" }));
    const session = createResearchSession("附件会话", { id: "attachment-session" });
    const message: ConversationMessage = {
      id: "attachment-message",
      sessionId: session.id,
      role: "user",
      text: "读取附件",
      status: "completed",
      attachments: [{ name: artifact.name, size: artifact.size, kind: "file" }],
      createdAt: "2026-08-10T00:00:00.000Z",
    };
    await repository.saveSourceArtifact(artifact);
    await repository.saveSession(session);
    await repository.saveMessage(message);
    await repository.associateMessageSourceArtifacts(message.id, [artifact.id]);

    const stored = (await repository.listMessages(session.id))[0];
    expect(stored?.sourceArtifactIds).toEqual([artifact.id]);
    expect(stored?.attachments?.[0]?.sourceArtifactId).toBe(artifact.id);
    expect((await repository.inspectSourceArtifactDeletion(artifact.id)).references).toEqual(expect.arrayContaining([
      expect.objectContaining({ kind: "source_message", id: message.id }),
    ]));
  });

  it("links the source message in the same import bundle operation", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const artifact = await createSourceArtifact(new File(["source,target\na,b\n"], "bundle.csv"));
    const base = fixtureGraph();
    const graph = { ...base, sourceArtifactIds: [artifact.id] };
    const session = createResearchSession("原子消息关联", { id: "bundle-link-session", graphVersionId: graph.id });
    const message: ConversationMessage = {
      id: "bundle-link-message",
      sessionId: session.id,
      role: "user",
      text: "导入数据",
      status: "completed",
      attachments: [{ name: artifact.name, size: artifact.size, kind: "file" }],
      createdAt: "2026-08-10T00:00:00.000Z",
    };
    await repository.saveSession(session);
    await repository.saveMessage(message);
    await repository.saveImportBundle({
      sourceArtifacts: [artifact],
      graphVersion: graph,
      viewState: createDefaultGraphViewState(graph.id),
      session,
      event: createSemanticEvent("graph_imported", { graphVersionId: graph.id, sessionId: session.id }),
      sourceMessageId: message.id,
    });

    expect((await repository.listMessages(session.id))[0]?.sourceArtifactIds).toEqual([artifact.id]);
  });

  it("propagates persistent transaction failures without partially mutating the memory mirror", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const artifact = await createSourceArtifact(new File(["{}"], "transaction.json"));
    await repository.saveSourceArtifact(artifact);
    await repository.listGraphVersionManifests();

    const internals = repository as unknown as {
      database?: { transaction: () => Promise<never> };
    };
    internals.database = {
      transaction: () => Promise.reject(new Error("simulated-commit-failure")),
    };
    await expect(repository.deleteSourceArtifact(artifact.id)).rejects.toThrow("PERSISTENT_TRANSACTION_FAILED");
    internals.database = undefined;

    expect(await repository.getSourceArtifact(artifact.id)).toBeDefined();
  });

  it("persists one-time initialization metadata independently of sessions", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const metadata = {
      initializedAt: "2026-08-10T00:00:00.000Z",
      seededDemoVersion: 1,
      updatedAt: "2026-08-10T00:00:00.000Z",
    } as const;

    expect(await repository.getInitializationMetadata()).toBeUndefined();
    await repository.saveInitializationMetadata(metadata);
    expect(await repository.getInitializationMetadata()).toEqual(metadata);
  });

  it("normalizes legacy presentation suffixes from persisted session titles", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const legacy = Object.freeze({
      id: "session-legacy-title",
      title: "governance-collaboration · 图谱研究",
      lifecycle: "active" as const,
      updatedAt: "2026-08-10T00:00:00.000Z",
    });

    await repository.saveSession(legacy);

    expect((await repository.getSession(legacy.id))?.title).toBe("governance-collaboration");
    expect((await repository.listSessions())[0]?.title).toBe("governance-collaboration");
  });
});
