import { describe, expect, it } from "vitest";

import { createValidatedCoreFixture } from "../test/fixtures/core";
import { registeredEdgeIdentityForLocalId } from "./coreEdgeIdentity";
import { createGraphVersion } from "./graphImport";
import { buildCoreOverlay } from "./coreOverlay";

describe("Core analysis overlay", () => {
  it("highlights only existing facts and never inserts a proposed collaboration edge", () => {
    const graph = createGraphVersion("collaboration.json", [
      { id: "a", label: "A", type: "person", attributes: {} },
      { id: "b", label: "B", type: "person", attributes: {} },
      { id: "c", label: "C", type: "person", attributes: {} },
    ], [
      { id: "ab", source: "a", target: "b", type: "collaborates", weight: 1, directed: false, attributes: {} },
      { id: "bc", source: "b", target: "c", type: "collaborates", weight: 2, directed: false, attributes: {} },
    ]);
    const abHash = registeredEdgeIdentityForLocalId(graph, "ab").edgeHash;
    const bcHash = registeredEdgeIdentityForLocalId(graph, "bc").edgeHash;
    const fixture = createValidatedCoreFixture({
      graphVersionId: graph.id,
      taskId: "core.collaboration_completion",
      findingType: "core-collaboration-completion",
      entityType: "node-pair",
      subjectIds: ["a", "c"],
      pathNodeIds: ["a", "b", "c", "missing-node"],
      pathEdgeIds: [abHash, bcHash, "f".repeat(64)],
    });
    const before = JSON.stringify(graph);

    const overlay = buildCoreOverlay(graph, fixture.binding, fixture.result, fixture.finding);

    expect(overlay.kind).toBe("governance");
    expect(overlay.nodeValues).toEqual({ a: "subject", b: "evidence", c: "subject" });
    expect(overlay.edgeValues).toEqual({ ab: "evidence", bc: "evidence" });
    expect(overlay.provenance).toMatchObject({
      engine: "gfm_core",
      runId: fixture.binding.runId,
      taskId: fixture.result.taskId,
      graphVersionHash: fixture.result.graphVersionHash,
      modelVersionHash: fixture.result.modelVersionHash,
      resultHash: fixture.result.resultHash,
      findingHash: fixture.finding.findingHash,
      publicRequestHash: fixture.binding.publicRequestHash,
      serverRequestHash: fixture.binding.serverRequestHash,
    });
    expect(Object.isFrozen(overlay)).toBe(true);
    expect(JSON.stringify(graph)).toBe(before);
    expect(graph.edges.some((edge) => (
      (edge.source === "a" && edge.target === "c") || (edge.source === "c" && edge.target === "a")
    ))).toBe(false);
  });

  it("rejects a result or finding bound to another graph/run", () => {
    const graph = createGraphVersion("risk.json", [
      { id: "a", label: "A", attributes: {} },
    ], []);
    const fixture = createValidatedCoreFixture({ graphVersionId: graph.id });

    expect(() => buildCoreOverlay(
      { ...graph, id: "other-graph" },
      fixture.binding,
      fixture.result,
      fixture.finding,
    )).toThrow("GFM_CORE_OVERLAY_BINDING_INVALID");
  });

  it("maps registered evidence beyond the bounded UI preview back to its local edge ID", () => {
    const nodes = Array.from({ length: 1_005 }, (_value, index) => ({
      id: `n-${index.toString().padStart(4, "0")}`,
      label: `N ${index}`,
      attributes: {},
    }));
    const edges = Array.from({ length: 1_004 }, (_value, index) => ({
      id: `edge-${index.toString().padStart(4, "0")}`,
      source: nodes[index]!.id,
      target: nodes[index + 1]!.id,
      type: "collaborates",
      weight: index + 1,
      directed: false,
      attributes: {},
    }));
    const graph = createGraphVersion("large.json", nodes, edges);
    const target = edges.at(-1)!;
    expect(graph.preview.edges.some((edge) => edge.id === target.id)).toBe(false);
    const targetHash = registeredEdgeIdentityForLocalId(graph, target.id).edgeHash;
    const fixture = createValidatedCoreFixture({
      graphVersionId: graph.id,
      taskId: "core.collaboration_completion",
      findingType: "core-collaboration-completion",
      entityType: "node-pair",
      subjectIds: [target.source, target.target],
      pathNodeIds: [target.source, target.target],
      pathEdgeIds: [targetHash],
    });

    const overlay = buildCoreOverlay(graph, fixture.binding, fixture.result, fixture.finding);

    expect(overlay.edgeValues).toEqual({ [target.id]: "evidence" });
  });
});
