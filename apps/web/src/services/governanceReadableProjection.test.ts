import { describe, expect, it } from "vitest";

import { createGraphVersion } from "./graphImport";
import { deterministicGraphInitialPositions } from "./graphDeterministicLayout";
import {
  governanceProjectionSpec,
  projectGovernanceGraph,
  projectGovernanceSkeletonGraph,
} from "./governanceReadableProjection";

function graph(nodeCount: number, edgeCount: number) {
  const nodes = Array.from({ length: nodeCount }, (_, index) => ({
    id: `n${String(index).padStart(3, "0")}`,
    label: `Node ${index}`,
    type: "account",
    attributes: {},
  }));
  const edges = Array.from({ length: edgeCount }, (_, index) => ({
    id: `e${String(index).padStart(4, "0")}`,
    source: `n${String(index % nodeCount).padStart(3, "0")}`,
    target: `n${String((index * 7 + 1) % nodeCount).padStart(3, "0")}`,
    weight: edgeCount - index,
    directed: false,
    attributes: {},
  }));
  return createGraphVersion("projection-test", nodes, edges);
}

function fragmentedOverviewGraph() {
  const coreNodes = Array.from({ length: 100 }, (_, index) => ({ id: `core-${String(index).padStart(3, "0")}`, label: `Core ${index}`, type: "account", attributes: {} }));
  const isolatedNodes = Array.from({ length: 10 }, (_, index) => ({ id: `isolated-${String(index).padStart(2, "0")}`, label: `Isolated ${index}`, type: "account", attributes: { structureMissing: true } }));
  const edges = coreNodes.slice(1).map((node, index) => ({ id: `core-edge-${index}`, source: "core-000", target: node.id, directed: false, attributes: {} }));
  return createGraphVersion("fragmented-overview", [...coreNodes, ...isolatedNodes], edges);
}

function xBounds(ids: readonly string[], positions: ReadonlyMap<string, { readonly x: number }>) {
  const values = ids.map((id) => positions.get(id)?.x ?? 0);
  return { minimum: Math.min(...values), maximum: Math.max(...values) };
}

describe("SocialGraph-FM Governance readable graph projection", () => {
  it("keeps small graphs unchanged", () => {
    const source = graph(20, 30);
    expect(projectGovernanceGraph(source, governanceProjectionSpec("risk", false))).toBe(source);
  });

  it("applies the published overview budget deterministically", () => {
    const source = graph(300, 1_200);
    const first = projectGovernanceGraph(source, governanceProjectionSpec("risk", false));
    const second = projectGovernanceGraph(source, governanceProjectionSpec("risk", false));
    expect(first.nodes).toHaveLength(120);
    expect(first.edges.length).toBeLessThanOrEqual(240);
    expect(first.nodes.map((node) => node.id)).toEqual(second.nodes.map((node) => node.id));
    expect(first.edges.map((edge) => edge.id)).toEqual(second.edges.map((edge) => edge.id));
    expect(first.preview.originalNodeCount).toBe(300);
    expect(first.preview.originalEdgeCount).toBe(1_200);
  });

  it("keeps focused evidence nodes and never emits dangling edges", () => {
    const source = graph(300, 1_200);
    const projected = projectGovernanceGraph(
      source,
      governanceProjectionSpec("relations", true),
      ["n299", "n298"],
    );
    const ids = new Set(projected.nodes.map((node) => node.id));
    expect(projected.id).toBe(source.id);
    expect(projected.nodes).toHaveLength(60);
    expect(projected.edges.length).toBeLessThanOrEqual(120);
    expect(ids.has("n299")).toBe(true);
    expect(ids.has("n298")).toBe(true);
    expect(projected.edges.every((edge) => ids.has(edge.source) && ids.has(edge.target))).toBe(true);
  });

  it("maps four UI lenses to the approved presets", () => {
    expect(governanceProjectionSpec("risk", false)).toMatchObject({ preset: "overview", nodeBudget: 120, edgeBudget: 240 });
    expect(governanceProjectionSpec("relations", false)).toMatchObject({ preset: "relation", nodeBudget: 80, edgeBudget: 160 });
    expect(governanceProjectionSpec("community", false)).toMatchObject({ preset: "groups", groupBudget: 12 });
    expect(governanceProjectionSpec("router", true)).toMatchObject({ preset: "evidence", nodeBudget: 60, edgeBudget: 120 });
  });

  it("keeps the overview core readable and preserves the full isolated count", () => {
    const source = fragmentedOverviewGraph();
    const projected = projectGovernanceGraph(source, governanceProjectionSpec("risk", false));
    const selectedIsolates = projected.nodes.filter((node) => node.id.startsWith("isolated-"));
    expect(projected.nodes).toHaveLength(103);
    expect(selectedIsolates).toHaveLength(3);
    expect(projected.summary.isolatedNodes).toBe(10);
    expect(projected.preview.originalNodeCount).toBe(110);

    const positions = deterministicGraphInitialPositions(projected.nodes, projected.edges);
    const coreBounds = xBounds(projected.nodes.filter((node) => node.id.startsWith("core-")).map((node) => node.id), positions);
    const allBounds = xBounds(projected.nodes.map((node) => node.id), positions);
    const coreWidthRatio = (coreBounds.maximum - coreBounds.minimum) / (allBounds.maximum - allBounds.minimum);
    // The production initializer is an organic, topology-keyed seed. It must
    // not reserve a far-right rail for isolates or force the core into a
    // narrow ring-shaped band.
    expect(coreWidthRatio).toBeGreaterThan(0.8);
    const isolateXs = selectedIsolates.map((node) => positions.get(node.id)?.x ?? 0);
    expect(Math.max(...isolateXs) - Math.min(...isolateXs)).toBeGreaterThan(20);
    expect(selectedIsolates.every((node) => Number.isFinite(positions.get(node.id)?.x))).toBe(true);
  });

  it("keeps full bounded nodes but only a deterministic spanning skeleton in Advanced", () => {
    const source = graph(716, 9_715);
    const first = projectGovernanceSkeletonGraph(source);
    const second = projectGovernanceSkeletonGraph(source);
    expect(first.nodes).toHaveLength(716);
    expect(first.edges.length).toBeLessThanOrEqual(715);
    expect(first.edges.map((edge) => edge.id)).toEqual(second.edges.map((edge) => edge.id));
    expect(first.preview.originalEdgeCount).toBe(9_715);
  });
});
