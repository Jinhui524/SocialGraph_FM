import { describe, expect, it, vi } from "vitest";

import type { GraphBuildIntentInput } from "../types/graph";
import { buildGraphBuildIntentPayload, HttpGraphBuildIntentNormalizer } from "./graphBuildIntent";

function input(): GraphBuildIntentInput {
  return {
    description: "from_user 指向 to_user，表示有向消息关系",
    requestToken: "build-request-1",
    baseGraphVersionId: "graph-base",
    files: [{
      artifactId: "source-1",
      role: "single",
      format: "csv",
      columns: [
        { name: "from_user", inferredType: "string", missingRate: 0, cardinality: 10, nonNullCount: 20, nullCount: 0 },
        { name: "to_user", inferredType: "string", missingRate: 0, cardinality: 12, nonNullCount: 20, nullCount: 0 },
        { name: "weight", inferredType: "number", missingRate: 0, cardinality: 8, nonNullCount: 20, nullCount: 0 },
      ],
    }],
    allowedPolicies: {
      direction: ["file", "directed", "undirected"],
      duplicateEdges: ["preserve", "merge_sum", "reject"],
      selfLoops: ["preserve", "reject"],
      danglingEndpoints: ["derive_nodes", "reject"],
      timeFormats: ["none", "auto", "iso8601", "year", "unix_seconds", "unix_milliseconds"],
    },
  };
}

describe("graph build intent client", () => {
  it("sends only aggregate column profiles accepted by the backend contract", () => {
    const payload = buildGraphBuildIntentPayload(input());
    expect(Object.keys(payload)).toEqual(["description", "columnProfiles"]);
    expect(payload.columnProfiles?.[2]).toEqual({
      name: "weight",
      inferredType: "float",
      nonNullCount: 20,
      nullCount: 0,
      uniqueCount: 8,
    });
    const serialized = JSON.stringify(payload);
    expect(serialized).not.toContain("source-1");
    expect(serialized).not.toContain("graph-base");
    expect(serialized).not.toContain("rows");
    expect(serialized).not.toContain("sampleValues");
  });

  it("grounds returned mappings, converts directedness and binds the request id", async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({
      kind: "graph_build_intent",
      mapping: {
        sourceColumn: "from_user",
        targetColumn: "to_user",
        edgeTypeColumn: null,
        weightColumn: "weight",
        timestampColumn: null,
      },
      directedness: "directed",
      confidence: 0.94,
      requiresMapping: false,
      meta: {
        schemaVersion: "1.0",
        source: "llm",
        requestId: "build-request-1",
        warnings: [],
      },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    const normalizer = new HttpGraphBuildIntentNormalizer({ baseUrl: "http://api.test", fetcher: fetcher as unknown as typeof fetch });

    const result = await normalizer.normalizeGraphBuildIntent(input());

    expect(result).toMatchObject({
      kind: "construction_revision",
      requestToken: "build-request-1",
      baseGraphVersionId: "graph-base",
      source: "llm",
      spec: {
        inputShape: "edge_table",
        directionPolicy: "directed",
        edgeMapping: { source: "from_user", target: "to_user", weight: "weight" },
      },
    });
    const calls = fetcher.mock.calls as unknown as [RequestInfo | URL, RequestInit][];
    const init = calls[0][1];
    expect(new Headers(init.headers).get("X-Request-ID")).toBe("build-request-1");
  });

  it("rejects a stale request id and falls back to grounded deterministic rules", async () => {
    const fetcher = vi.fn(async () => new Response(JSON.stringify({
      kind: "graph_build_intent",
      mapping: { sourceColumn: "invented", targetColumn: "to_user" },
      directedness: "directed",
      confidence: 1,
      requiresMapping: false,
      meta: { schemaVersion: "1.0", source: "llm", requestId: "stale", warnings: [] },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    const normalizer = new HttpGraphBuildIntentNormalizer({ baseUrl: "http://api.test", fetcher: fetcher as unknown as typeof fetch });

    const result = await normalizer.normalizeGraphBuildIntent(input());

    expect(result.source).toBe("deterministic_fallback");
    expect(result.requestToken).toBe("build-request-1");
    expect(result.warnings[0]).toContain("本地确定性建议");
  });

  it("sends role-scoped aggregate profiles for dual tables and accepts grounded node mapping v1.1", async () => {
    const dualInput: GraphBuildIntentInput = {
      ...input(),
      files: [
        {
          artifactId: "node-artifact-secret",
          role: "nodes",
          format: "csv",
          columns: [
            { name: "entity_key", inferredType: "string", missingRate: 0, cardinality: 3, nonNullCount: 3, nullCount: 0 },
            { name: "entity_kind", inferredType: "string", missingRate: 0, cardinality: 2, nonNullCount: 3, nullCount: 0 },
          ],
        },
        { ...input().files[0], artifactId: "edge-artifact-secret", role: "edges" },
      ],
    };
    const fetcher = vi.fn(async () => new Response(JSON.stringify({
      kind: "graph_build_intent",
      mapping: {
        sourceColumn: "from_user",
        targetColumn: "to_user",
        edgeTypeColumn: null,
        weightColumn: "weight",
        timestampColumn: null,
      },
      nodeMapping: { idColumn: "entity_key", labelColumn: null, typeColumn: "entity_kind" },
      directedness: "directed",
      confidence: 0.94,
      requiresMapping: false,
      meta: { schemaVersion: "1.1", source: "llm", requestId: "build-request-1", warnings: [] },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    const normalizer = new HttpGraphBuildIntentNormalizer({ baseUrl: "http://api.test", fetcher: fetcher as unknown as typeof fetch });

    const result = await normalizer.normalizeGraphBuildIntent(dualInput);
    const payload = JSON.parse(String((fetcher.mock.calls as unknown as [RequestInfo | URL, RequestInit][])[0][1].body));

    expect(payload.files).toEqual([
      expect.objectContaining({ role: "nodes", columnProfiles: expect.arrayContaining([expect.objectContaining({ name: "entity_key" })]) }),
      expect.objectContaining({ role: "edges", columnProfiles: expect.arrayContaining([expect.objectContaining({ name: "from_user" })]) }),
    ]);
    expect(JSON.stringify(payload)).not.toContain("artifact-secret");
    expect(result.spec).toMatchObject({
      inputShape: "node_edge_tables",
      nodeMapping: { id: "entity_key", type: "entity_kind" },
      edgeMapping: { source: "from_user", target: "to_user", weight: "weight" },
    });
  });
});
