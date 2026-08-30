import { describe, expect, it, vi } from "vitest";

import type { GraphEdge, GraphNode, GraphVersion } from "../types/graph";
import { createGraphVersion } from "./graphImport";
import { computeGraphVersionDiff } from "./graphVersionDiff";

function version(
  nodes: readonly GraphNode[],
  edges: readonly GraphEdge[],
  parentVersionId?: string,
) {
  return createGraphVersion("diff.json", nodes, edges, [], parentVersionId ? { parentVersionId } : {});
}

describe("GraphVersion diff", () => {
  it("does not replay a timed-out large diff on the UI thread", async () => {
    const worker = {
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: Event) => void) | null,
      postMessage: vi.fn(),
      terminate: vi.fn(),
    };
    const before = version([{ id: "a", label: "A", attributes: {} }], []);
    const after = version([{ id: "b", label: "B", attributes: {} }], []);

    await expect(computeGraphVersionDiff(before, after, {
      workerThreshold: 0,
      timeoutMs: 1,
      workerFactory: () => worker as unknown as Worker,
    })).rejects.toThrow("GRAPH_VERSION_DIFF_TIMEOUT");
    expect(worker.postMessage).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("does not replay a failed large diff on the UI thread", async () => {
    const worker = {
      onmessage: null as ((event: MessageEvent) => void) | null,
      onerror: null as ((event: Event) => void) | null,
      postMessage: vi.fn(),
      terminate: vi.fn(),
    };
    const before = version([{ id: "a", label: "A", attributes: {} }], []);
    const after = version([{ id: "b", label: "B", attributes: {} }], []);
    const promise = computeGraphVersionDiff(before, after, {
      workerThreshold: 0,
      workerFactory: () => worker as unknown as Worker,
    });
    worker.onerror?.(new Event("error"));

    await expect(promise).rejects.toThrow("GRAPH_VERSION_DIFF_WORKER_FAILED");
    expect(worker.postMessage).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("reports node identity changes and edge fact changes independently", async () => {
    const base = version([
      { id: "a", label: "成员甲", type: "person", attributes: { score: 1 } },
      { id: "b", label: "组织乙", type: "organization", attributes: {} },
      { id: "d", label: "旧项目", type: "project", attributes: {} },
    ], [
      { id: "e1", source: "a", target: "b", type: "合作", weight: 1, directed: false, attributes: {} },
    ]);
    const child = version([
      { id: "a", label: "成员甲（修订）", type: "organization", attributes: { score: 2 } },
      { id: "b", label: "组织乙", type: "organization", attributes: {} },
      { id: "c", label: "项目丙", type: "project", attributes: {} },
    ], [
      {
        id: "e1",
        source: "a",
        target: "b",
        type: "合作",
        weight: 2,
        timestamp: "2025-01-01T00:00:00.000Z",
        directed: false,
        attributes: { reviewed: true },
      },
      { id: "e2", source: "c", target: "a", attributes: {} },
    ], base.id);

    const report = await computeGraphVersionDiff(base, child, { forceSynchronous: true });

    expect(report.sameContent).toBe(false);
    expect(report.sameLineage).toBe(true);
    expect(report.summary.nodes).toEqual({ added: 1, removed: 1, modified: 1 });
    expect(report.summary.edges).toEqual({ added: 1, removed: 0, modified: 1 });
    expect(report.samples).toEqual(expect.arrayContaining([
      expect.objectContaining({ entity: "node", id: "a", kind: "modified" }),
      expect.objectContaining({ entity: "node", id: "d", kind: "removed" }),
      expect.objectContaining({ entity: "edge", id: "e1", kind: "modified" }),
    ]));
    expect(report.samples.find((item) => item.id === "e1")?.fields.map((field) => field.field)).toEqual(
      expect.arrayContaining(["weight", "timestamp", "attributes"]),
    );
    expect(report.edgeIdChurn.count).toBe(0);
  });

  it("treats CSV row reorder as id churn, not relationship changes", async () => {
    const nodes = [
      { id: "a", label: "A", attributes: {} },
      { id: "b", label: "B", attributes: {} },
    ] satisfies readonly GraphNode[];
    const base = version(nodes, [
      { id: "edge-row-1", source: "a", target: "b", directed: false, weight: 1, attributes: {} },
      { id: "edge-row-2", source: "a", target: "b", directed: false, weight: 2, attributes: {} },
    ]);
    const reordered = version(nodes, [
      { id: "edge-row-1", source: "a", target: "b", directed: false, weight: 2, attributes: {} },
      { id: "edge-row-2", source: "b", target: "a", directed: false, weight: 1, attributes: {} },
    ], base.id);

    const report = await computeGraphVersionDiff(base, reordered, { forceSynchronous: true });

    expect(report.summary.edges).toEqual({ added: 0, removed: 0, modified: 0 });
    expect(report.samples.filter((sample) => sample.entity === "edge")).toEqual([]);
    expect(report.edgeIdChurn.count).toBe(2);
    expect(report.edgeIdChurn.samples).toEqual(expect.arrayContaining([
      expect.objectContaining({ beforeId: "edge-row-1", afterId: "edge-row-2" }),
      expect.objectContaining({ beforeId: "edge-row-2", afterId: "edge-row-1" }),
    ]));
  });

  it("normalizes undirected endpoints without reporting fact or id changes", async () => {
    const nodes = [
      { id: "a", label: "A", attributes: {} },
      { id: "b", label: "B", attributes: {} },
    ] satisfies readonly GraphNode[];
    const base = version(nodes, [
      { id: "e1", source: "a", target: "b", directed: false, attributes: { channel: "offline" } },
    ]);
    const reversed = version(nodes, [
      { id: "e1", source: "b", target: "a", directed: false, attributes: { channel: "offline" } },
    ], base.id);

    const report = await computeGraphVersionDiff(base, reversed, { forceSynchronous: true });

    expect(report.summary.edges).toEqual({ added: 0, removed: 0, modified: 0 });
    expect(report.edgeIdChurn.count).toBe(0);
  });

  it("matches parallel-edge fact multisets before pairing modifications", async () => {
    const nodes = [
      { id: "a", label: "A", attributes: {} },
      { id: "b", label: "B", attributes: {} },
    ] satisfies readonly GraphNode[];
    const base = version(nodes, [
      { id: "e1", source: "a", target: "b", directed: false, weight: 1, attributes: {} },
      { id: "e2", source: "a", target: "b", directed: false, weight: 1, attributes: {} },
      { id: "e3", source: "a", target: "b", directed: false, weight: 2, attributes: {} },
    ]);
    const changed = version(nodes, [
      { id: "z1", source: "b", target: "a", directed: false, weight: 1, attributes: {} },
      { id: "z2", source: "a", target: "b", directed: false, weight: 3, attributes: {} },
    ], base.id);

    const report = await computeGraphVersionDiff(base, changed, { forceSynchronous: true });

    expect(report.summary.edges).toEqual({ added: 0, removed: 1, modified: 1 });
    expect(report.edgeIdChurn.count).toBe(2);
  });

  it("reports structural edge changes as removed and added facts", async () => {
    const nodes = [
      { id: "a", label: "A", attributes: {} },
      { id: "b", label: "B", attributes: {} },
      { id: "c", label: "C", attributes: {} },
    ] satisfies readonly GraphNode[];
    const base = version(nodes, [
      { id: "e1", source: "a", target: "b", type: "合作", directed: false, attributes: {} },
    ]);
    const changed = version(nodes, [
      { id: "e1", source: "a", target: "c", type: "参与", directed: true, attributes: {} },
    ], base.id);

    const report = await computeGraphVersionDiff(base, changed, { forceSynchronous: true });

    expect(report.summary.edges).toEqual({ added: 1, removed: 1, modified: 0 });
    expect(report.edgeIdChurn.count).toBe(0);
  });

  it("recognizes fact-identical UUID versions while retaining metadata comparison", async () => {
    const nodes = [{ id: "a", label: "A", attributes: {} }] satisfies readonly GraphNode[];
    const first = version(nodes, []);
    const second = version(nodes, []);
    const report = await computeGraphVersionDiff(first, second, { forceSynchronous: true });

    expect(first.id).not.toBe(second.id);
    expect(report.sameContent).toBe(true);
    expect(report.summary).toEqual({
      nodes: { added: 0, removed: 0, modified: 0 },
      edges: { added: 0, removed: 0, modified: 0 },
    });
    expect(report.samples).toEqual([]);
  });

  it("derives missing legacy hashes without mutating either version", async () => {
    const current = version([{ id: "a", label: "A", attributes: {} }], []);
    const { contentHash: _contentHash, ...legacyFields } = current;
    const legacy = legacyFields as GraphVersion;
    const before = JSON.stringify(legacy);

    const report = await computeGraphVersionDiff(legacy, current, { forceSynchronous: true });

    expect(report.fromHashSource).toBe("derived");
    expect(report.toHashSource).toBe("stored");
    expect(report.sameContent).toBe(true);
    expect(JSON.stringify(legacy)).toBe(before);
    expect("contentHash" in legacy).toBe(false);
  });

  it("keeps exact totals when deterministic samples are bounded", async () => {
    const before = version([], []);
    const after = version(
      Array.from({ length: 8 }, (_, index) => ({ id: `n${index}`, label: `N${index}`, attributes: {} })),
      [],
    );
    const report = await computeGraphVersionDiff(before, after, { sampleLimit: 3, forceSynchronous: true });

    expect(report.summary.nodes.added).toBe(8);
    expect(report.samples).toHaveLength(3);
    expect(report.truncated).toBe(true);
  });
});
