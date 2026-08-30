import { describe, expect, it } from "vitest";

import type { GraphBuildSpec } from "../types/graph";
import { createGraphVersion, LocalGraphImportAdapter } from "./graphImport";
import { createSourceArtifact } from "./sourceArtifact";

function edgeSpec(artifactIds: readonly string[], overrides: Partial<GraphBuildSpec> = {}): GraphBuildSpec {
  return {
    schemaVersion: "1.0",
    inputShape: "edge_table",
    sourceArtifactIds: artifactIds,
    edgeMapping: {
      source: "source",
      target: "target",
      sourceLabel: "source_label",
      targetLabel: "target_label",
      sourceType: "source_type",
      targetType: "target_type",
      timestamp: "year",
    },
    directionPolicy: "undirected",
    duplicateEdgePolicy: "preserve",
    selfLoopPolicy: "preserve",
    danglingEndpointPolicy: "derive_nodes",
    timeFormat: "auto",
    ...overrides,
  };
}

describe("GraphBuildSpec import", () => {
  it("preserves endpoint metadata, normalizes time and creates independent UUID versions with stable hashes", async () => {
    const file = new File([
      "source,target,source_label,target_label,source_type,target_type,source_department,year\n" +
      "u1,o1,成员甲,机构甲,person,organization,治理学院,2024\n",
    ], "relationships.csv", { type: "text/csv" });
    const artifact = await createSourceArtifact(file);
    const spec = edgeSpec([artifact.id]);
    const adapter = new LocalGraphImportAdapter();

    const first = await adapter.parseFiles([file], spec, { buildSpec: spec, sourceArtifacts: [artifact] });
    const second = await adapter.parseFiles([file], spec, { buildSpec: spec, sourceArtifacts: [artifact] });

    expect(first.status).toBe("ready");
    expect(first.graphVersion?.nodes).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "u1", label: "成员甲", type: "person", attributes: { department: "治理学院" } }),
      expect.objectContaining({ id: "o1", label: "机构甲", type: "organization" }),
    ]));
    expect(first.graphVersion?.edges[0]).toMatchObject({
      directed: false,
      timestamp: "2024-01-01T00:00:00.000Z",
    });
    expect(first.graphVersion?.id).not.toBe(second.graphVersion?.id);
    expect(first.graphVersion?.contentHash).toBe(second.graphVersion?.contentHash);
    expect(first.graphVersion?.contentHash).toMatch(/^[a-f0-9]{64}$/u);
    expect(first.graphVersion?.sourceHash).toMatch(/^[a-f0-9]{64}$/u);
    expect(first.graphVersion?.buildSpecHash).toMatch(/^[a-f0-9]{64}$/u);
  });

  it("blocks mixed time formats instead of comparing raw strings", async () => {
    const file = new File([
      "source,target,time\na,b,2024\nb,c,2025-01-02\n",
    ], "mixed.csv", { type: "text/csv" });
    const artifact = await createSourceArtifact(file);
    const spec = {
      ...edgeSpec([artifact.id]),
      edgeMapping: { source: "source", target: "target", timestamp: "time" },
    } satisfies GraphBuildSpec;
    const run = await new LocalGraphImportAdapter().parseFiles([file], spec, {
      buildSpec: spec,
      sourceArtifacts: [artifact],
    });

    expect(run.status).toBe("failed");
    expect(run.issues.map((issue) => issue.code)).toContain("ambiguous_time_format");
  });

  it("imports nodes + edges together and blocks dangling endpoints atomically", async () => {
    const nodes = new File(["id,label,type\nn1,成员甲,person\nn2,机构甲,organization\n"], "nodes.csv", { type: "text/csv" });
    const validEdges = new File(["source,target,relation\nn1,n2,隶属\n"], "edges.csv", { type: "text/csv" });
    const invalidEdges = new File(["source,target,relation\nn1,missing,隶属\n"], "edges.csv", { type: "text/csv" });
    const nodeArtifact = await createSourceArtifact(nodes, "nodes");
    const edgeArtifact = await createSourceArtifact(validEdges, "edges");
    const spec: GraphBuildSpec = {
      schemaVersion: "1.0",
      inputShape: "node_edge_tables",
      sourceArtifactIds: [nodeArtifact.id, edgeArtifact.id],
      nodeMapping: { id: "id", label: "label", type: "type" },
      edgeMapping: { source: "source", target: "target", edgeType: "relation" },
      directionPolicy: "undirected",
      duplicateEdgePolicy: "preserve",
      selfLoopPolicy: "preserve",
      danglingEndpointPolicy: "reject",
      timeFormat: "none",
    };
    const adapter = new LocalGraphImportAdapter();

    const valid = await adapter.parseFiles([nodes, validEdges], spec, {
      buildSpec: spec,
      sourceArtifacts: [nodeArtifact, edgeArtifact],
    });
    const invalid = await adapter.parseFiles([nodes, invalidEdges], spec, { buildSpec: spec });

    expect(valid.status).toBe("ready");
    expect(valid.graphVersion?.summary).toMatchObject({ nodeCount: 2, edgeCount: 1 });
    expect(valid.graphVersion?.parentVersionId).toBeUndefined();
    expect(invalid.status).toBe("failed");
    expect(invalid.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: "dangling_edge", severity: "error" }),
    ]));
  });

  it("records parentVersionId without reusing version identity", () => {
    const base = createGraphVersion("base.json", [{ id: "a", label: "A", attributes: {} }], []);
    const child = createGraphVersion("base.json", [{ id: "a", label: "A2", attributes: {} }], [], [], {
      parentVersionId: base.id,
      provenance: {
        origin: "browser_import",
        pipeline: "browser-import",
        pipelineVersion: "2.0.0",
        buildSpecSchemaVersion: "1.0",
        sourceHashScheme: "artifact-sha256-list-v1",
        reconstructionReason: "construction_revision",
      },
    });
    expect(child.parentVersionId).toBe(base.id);
    expect(child.id).not.toBe(base.id);
    expect(child.contentHash).not.toBe(base.contentHash);
    expect(child.provenance).toEqual(expect.objectContaining({
      origin: "browser_import",
      pipelineVersion: "2.0.0",
      reconstructionReason: "construction_revision",
    }));
    expect(Object.isFrozen(child.provenance)).toBe(true);
  });

  it("requests and applies file-scoped manual mappings for non-standard node and edge tables", async () => {
    const nodes = new File([
      "entity_key,display_name,entity_kind,department\n" +
      "n1,成员甲,person,治理学院\n" +
      "n2,机构甲,organization,公共管理\n",
    ], "entities.csv", { type: "text/csv" });
    const edges = new File([
      "actor_key,object_key,relation_kind,strength,event_year,note\n" +
      "n1,n2,隶属,2,2024,正式关系\n",
    ], "connections.csv", { type: "text/csv" });
    const nodeArtifact = await createSourceArtifact(nodes, "nodes");
    const edgeArtifact = await createSourceArtifact(edges, "edges");
    const incomplete: GraphBuildSpec = {
      schemaVersion: "1.0",
      inputShape: "node_edge_tables",
      sourceArtifactIds: [nodeArtifact.id, edgeArtifact.id],
      directionPolicy: "undirected",
      duplicateEdgePolicy: "preserve",
      selfLoopPolicy: "preserve",
      danglingEndpointPolicy: "reject",
      timeFormat: "auto",
    };
    const adapter = new LocalGraphImportAdapter();

    const first = await adapter.parseFiles([nodes, edges], incomplete, {
      buildSpec: incomplete,
      sourceArtifacts: [nodeArtifact, edgeArtifact],
    });
    expect(first.status).toBe("needs_mapping");
    expect(first.mappingRequest?.missingFields).toEqual(["node.id", "edge.source", "edge.target"]);
    expect(first.mappingRequest?.nodeTable?.headers).toContain("entity_kind");
    expect(first.mappingRequest?.edgeTable.headers).toContain("actor_key");

    const mapped: GraphBuildSpec = {
      ...incomplete,
      nodeMapping: { id: "entity_key", label: "display_name", type: "entity_kind" },
      edgeMapping: {
        source: "actor_key",
        target: "object_key",
        edgeType: "relation_kind",
        weight: "strength",
        timestamp: "event_year",
      },
    };
    const ready = await adapter.parseFiles([nodes, edges], mapped, {
      buildSpec: mapped,
      sourceArtifacts: [nodeArtifact, edgeArtifact],
    });

    expect(ready.status).toBe("ready");
    expect(ready.graphVersion?.nodes).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "n1", label: "成员甲", type: "person", attributes: { department: "治理学院" } }),
      expect.objectContaining({ id: "n2", label: "机构甲", type: "organization" }),
    ]));
    expect(ready.graphVersion?.edges[0]).toMatchObject({
      type: "隶属",
      weight: 2,
      timestamp: "2024-01-01T00:00:00.000Z",
      attributes: { note: "正式关系" },
    });
    expect(ready.graphVersion?.issues.map((issue) => issue.code)).not.toContain("all_nodes_unclassified");
  });

  it("uses SourceArtifact roles rather than file order to identify dual tables", async () => {
    const edges = new File(["left,right\nn1,n2\n"], "a.csv", { type: "text/csv" });
    const nodes = new File(["key,title,kind\nn1,甲,person\nn2,乙,organization\n"], "b.csv", { type: "text/csv" });
    const edgeArtifact = await createSourceArtifact(edges, "edges");
    const nodeArtifact = await createSourceArtifact(nodes, "nodes");
    const spec: GraphBuildSpec = {
      schemaVersion: "1.0",
      inputShape: "node_edge_tables",
      sourceArtifactIds: [edgeArtifact.id, nodeArtifact.id],
      nodeMapping: { id: "key", label: "title", type: "kind" },
      edgeMapping: { source: "left", target: "right" },
      directionPolicy: "undirected",
      duplicateEdgePolicy: "preserve",
      selfLoopPolicy: "preserve",
      danglingEndpointPolicy: "reject",
      timeFormat: "none",
    };

    const run = await new LocalGraphImportAdapter().parseFiles([edges, nodes], spec, {
      buildSpec: spec,
      sourceArtifacts: [edgeArtifact, nodeArtifact],
    });
    expect(run.status).toBe("ready");
    expect(run.graphVersion?.summary).toMatchObject({ nodeCount: 2, edgeCount: 1 });
    expect(run.graphVersion?.nodes.find((node) => node.id === "n2")?.type).toBe("organization");
  });
});
