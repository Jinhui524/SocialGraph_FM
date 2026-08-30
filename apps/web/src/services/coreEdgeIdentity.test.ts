import { describe, expect, it } from "vitest";

import vectors from "../../../../contracts/core-edge-identity-vectors.json";
import type { GraphEdge } from "../types/graph";
import { createGraphVersion } from "./graphImport";
import {
  buildRegisteredEdgeIdentityIndex,
  matchRegisteredEdgeHashes,
  registeredEdgeIdentityForLocalId,
} from "./coreEdgeIdentity";

function graphFor(edge: GraphEdge) {
  return createGraphVersion("edge-identity.json", [
    { id: "alpha", label: "Alpha", attributes: {} },
    { id: "zeta", label: "Zeta", attributes: {} },
  ], [edge]);
}

describe("registered edge identity parity", () => {
  it.each(vectors.cases)("matches the GFM-generated $name vector", (item) => {
    const graph = graphFor({ ...item.local, attributes: {} });

    expect(registeredEdgeIdentityForLocalId(graph, item.local.id))
      .toEqual(item.registeredIdentity);
  });

  it.each([
    ["missing type", { id: "edge", source: "alpha", target: "zeta", weight: 1, directed: true, attributes: {} }],
    ["missing weight", { id: "edge", source: "alpha", target: "zeta", type: "support", directed: true, attributes: {} }],
    ["unknown direction", { id: "edge", source: "alpha", target: "zeta", type: "support", weight: 1, attributes: {} }],
  ] as const)("fails closed for %s", (_label, edge) => {
    expect(() => registeredEdgeIdentityForLocalId(graphFor(edge), edge.id))
      .toThrow("GFM_CORE_EDGE_IDENTITY_UNPROVABLE");
  });

  it("rejects duplicate semantic identities instead of ambiguously mapping a hash", () => {
    const graph = createGraphVersion("duplicate.json", [
      { id: "alpha", label: "Alpha", attributes: {} },
      { id: "zeta", label: "Zeta", attributes: {} },
    ], [
      { id: "edge-1", source: "alpha", target: "zeta", type: "support", weight: 1, directed: true, attributes: {} },
      { id: "edge-2", source: "alpha", target: "zeta", type: "support", weight: 1, directed: true, attributes: {} },
    ]);

    expect(() => buildRegisteredEdgeIdentityIndex(graph))
      .toThrow("GFM_CORE_EDGE_IDENTITY_DUPLICATE");
  });

  it("ignores non-hash model evidence without touching the full edge collection", () => {
    const graph = graphFor({
      id: "edge",
      source: "alpha",
      target: "zeta",
      type: "support",
      weight: 1,
      directed: true,
      attributes: {},
    });
    const guardedGraph = Object.create(graph) as typeof graph;
    Object.defineProperty(guardedGraph, "edges", {
      get() {
        throw new Error("FULL_EDGE_SCAN_FOR_NON_HASH_EVIDENCE");
      },
    });

    expect(matchRegisteredEdgeHashes(guardedGraph, new Set(["alpha|zeta", "not-a-hash"])))
      .toEqual([]);
  });

  it("caches a proven local edge identity instead of rescanning the immutable graph", () => {
    const graph = graphFor({
      id: "edge",
      source: "alpha",
      target: "zeta",
      type: "support",
      weight: 1,
      directed: true,
      attributes: {},
    });
    let edgeReads = 0;
    const guardedGraph = Object.create(graph) as typeof graph;
    Object.defineProperty(guardedGraph, "edges", {
      get() {
        edgeReads += 1;
        return graph.edges;
      },
    });

    const first = registeredEdgeIdentityForLocalId(guardedGraph, "edge");
    const readsAfterFirstProof = edgeReads;
    const second = registeredEdgeIdentityForLocalId(guardedGraph, "edge");

    expect(second).toBe(first);
    expect(edgeReads).toBe(readsAfterFirstProof);
  });

  it("maps a warmed selected edge hash without rereading the full graph for overlay", () => {
    const graph = graphFor({
      id: "edge",
      source: "alpha",
      target: "zeta",
      type: "support",
      weight: 1,
      directed: true,
      attributes: {},
    });
    let denyEdgeReads = false;
    const guardedGraph = Object.create(graph) as typeof graph;
    Object.defineProperty(guardedGraph, "edges", {
      get() {
        if (denyEdgeReads) throw new Error("WARM_HASH_TRIGGERED_FULL_EDGE_SCAN");
        return graph.edges;
      },
    });
    const identity = registeredEdgeIdentityForLocalId(guardedGraph, "edge");
    denyEdgeReads = true;

    expect(matchRegisteredEdgeHashes(guardedGraph, new Set([identity.edgeHash])))
      .toEqual([{ localEdgeId: "edge", identity }]);
  });
});
