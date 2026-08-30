import { describe, expect, it } from "vitest";

import { MAX_IMPORT_BYTES, MAX_PREVIEW_EDGES, MAX_PREVIEW_NODES } from "../types/graph";
import type { GraphEdge, GraphNode } from "../types/graph";
import {
  LocalGraphImportAdapter,
  createGraphVersion,
} from "./graphImport";

describe("LocalGraphImportAdapter CSV", () => {
  it("parses common endpoint aliases and computes real summary metrics", async () => {
    const file = new File(
      ["src,dst,relation,weight,year\n张三,李四,合作,2,2024\n李四,王五,协作,1,2025\n"],
      "relationships.csv",
      { type: "text/csv" },
    );
    const adapter = new LocalGraphImportAdapter();

    const profile = await adapter.inspect(file);
    const run = await adapter.parse(file);

    expect(profile.needsMapping).toBe(false);
    expect(profile.suggestedMapping).toMatchObject({ source: "src", target: "dst", edgeType: "relation" });
    expect(run.status).toBe("ready");
    expect(run.graphVersion?.summary).toMatchObject({
      nodeCount: 3,
      edgeCount: 2,
      density: 0.666667,
      averageDegree: 1.333333,
      connectedComponents: 1,
    });
    expect(run.graphVersion?.edges[0]).toMatchObject({ source: "张三", target: "李四", type: "合作", weight: 2 });
  });

  it("returns needs_mapping and can be reparsed with an explicit mapping", async () => {
    const file = new File(
      ["成员A,成员B,关系说明\n张三,李四,合作\n李四,王五,推荐\n"],
      "custom.csv",
      { type: "text/csv" },
    );
    const adapter = new LocalGraphImportAdapter();

    const firstRun = await adapter.parse(file);
    expect(firstRun.status).toBe("needs_mapping");
    expect(firstRun.headers).toEqual(["成员A", "成员B", "关系说明"]);

    const secondRun = await adapter.parse(file, {
      source: "成员A",
      target: "成员B",
      edgeType: "关系说明",
    });
    expect(secondRun.status).toBe("ready");
    expect(secondRun.graphVersion?.summary).toMatchObject({ nodeCount: 3, edgeCount: 2 });
    expect(secondRun.graphVersion?.edges[1].type).toBe("推荐");
  });

  it("recognises Chinese endpoint aliases", async () => {
    const file = new File(["源节点,目标节点,权重\n甲,乙,0.8\n"], "zh.csv", { type: "text/csv" });
    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("ready");
    expect(run.graphVersion?.edges[0]).toMatchObject({ source: "甲", target: "乙", weight: 0.8 });
  });

  it("parses TSV edge tables with the same deterministic mapping rules", async () => {
    const file = new File(
      ["成员A\t成员B\t关系\t权重\n张三\t李四\t合作\t2\n李四\t王五\t协作\t1\n"],
      "relationships.tsv",
      { type: "text/tab-separated-values" },
    );
    const adapter = new LocalGraphImportAdapter();
    const profile = await adapter.inspect(file);
    const firstRun = await adapter.parse(file);
    const run = await adapter.parse(file, {
      source: "成员A",
      target: "成员B",
      edgeType: "关系",
      weight: "权重",
    });

    expect(profile.format).toBe("tsv");
    expect(profile.headers).toEqual(["成员A", "成员B", "关系", "权重"]);
    expect(firstRun.status).toBe("needs_mapping");
    expect(run.status).toBe("ready");
    expect(run.graphVersion?.summary).toMatchObject({ nodeCount: 3, edgeCount: 2 });
    expect(run.graphVersion?.edges[0]).toMatchObject({ type: "合作", weight: 2 });
  });

  it("backfills node labels and types when later rows provide missing endpoint metadata", async () => {
    const file = new File([
      "source,target,source_label,target_label,source_type,target_type\n" +
      "a,b,,,,\n" +
      "c,a,成员丙,机构甲,person,organization\n",
    ], "metadata.csv", { type: "text/csv" });

    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("ready");
    expect(run.graphVersion?.nodes.find((node) => node.id === "a")).toMatchObject({
      label: "机构甲",
      type: "organization",
    });
    expect(run.issues.map((issue) => issue.code)).not.toContain("conflicting_node_metadata");
  });

  it("rejects invalid optional mappings instead of silently dropping them", async () => {
    const file = new File(["source,target\na,b\n"], "strict.csv", { type: "text/csv" });
    const run = await new LocalGraphImportAdapter().parse(file, {
      source: "source",
      target: "target",
      weight: "invented_weight",
    });

    expect(run.status).toBe("needs_mapping");
    expect(run.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: "invalid_optional_column_mapping", severity: "error" }),
    ]));
  });

  it("reports when every imported node is unclassified", async () => {
    const file = new File(["source,target\na,b\n"], "unclassified.csv", { type: "text/csv" });
    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("ready");
    expect(run.issues).toEqual(expect.arrayContaining([
      expect.objectContaining({ code: "all_nodes_unclassified", severity: "warning" }),
    ]));
  });
});

describe("LocalGraphImportAdapter JSON", () => {
  it("parses the fixed nodes/edges shape and preserves attributes", async () => {
    const file = new File(
      [
        JSON.stringify({
          nodes: [
            { id: "a", label: "张三", type: "person", department: "计算机" },
            { id: "b", label: "李四", type: "person" },
          ],
          edges: [{ id: "ab", source: "a", target: "b", type: "合作", weight: 3 }],
        }),
      ],
      "graph.json",
      { type: "application/json" },
    );

    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("ready");
    expect(run.graphVersion?.nodes[0]).toMatchObject({
      id: "a",
      label: "张三",
      type: "person",
      attributes: { department: "计算机" },
    });
    expect(run.graphVersion?.summary).toMatchObject({ nodeCount: 2, edgeCount: 1, density: 1 });
  });

  it("reports duplicate nodes and removes dangling edges without inventing nodes", async () => {
    const file = new File(
      [
        JSON.stringify({
          nodes: [{ id: "a" }, { id: "a", label: "重复" }, { id: "b" }],
          edges: [
            { source: "a", target: "b" },
            { source: "a", target: "missing" },
          ],
        }),
      ],
      "quality-issues.json",
      { type: "application/json" },
    );

    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("ready");
    expect(run.graphVersion?.summary).toMatchObject({ nodeCount: 2, edgeCount: 1 });
    expect(run.issues.map((issue) => issue.code)).toEqual(
      expect.arrayContaining(["duplicate_node", "dangling_edge"]),
    );
    expect(run.graphVersion?.nodes.some((node) => node.id === "missing")).toBe(false);
  });
});

describe("LocalGraphImportAdapter XML graph formats", () => {
  it("parses GraphML typed attributes and excludes dangling endpoints", async () => {
    const file = new File([
      `<?xml version="1.0" encoding="UTF-8"?>
      <graphml xmlns="http://graphml.graphdrawing.org/xmlns">
        <key id="name" for="node" attr.name="label" attr.type="string" />
        <key id="kind" for="node" attr.name="type" attr.type="string" />
        <key id="strength" for="edge" attr.name="weight" attr.type="double" />
        <graph id="G" edgedefault="undirected">
          <node id="n1"><data key="name">张三</data><data key="kind">person</data></node>
          <node id="n2"><data key="name">李四</data><data key="kind">person</data></node>
          <edge id="e1" source="n1" target="n2"><data key="strength">2.5</data></edge>
          <edge id="e2" source="n1" target="missing" />
        </graph>
      </graphml>`,
    ], "network.graphml", { type: "application/graphml+xml" });
    const adapter = new LocalGraphImportAdapter();
    const profile = await adapter.inspect(file);
    const run = await adapter.parse(file);

    expect(profile).toMatchObject({ format: "graphml", supported: true, needsMapping: false });
    expect(run.status).toBe("ready");
    expect(run.graphVersion?.nodes[0]).toMatchObject({ id: "n1", label: "张三", type: "person" });
    expect(run.graphVersion?.edges[0]).toMatchObject({ id: "e1", source: "n1", target: "n2", weight: 2.5, directed: false });
    expect(run.graphVersion?.summary).toMatchObject({ nodeCount: 2, edgeCount: 1 });
    expect(run.issues.map((issue) => issue.code)).toContain("dangling_edge");
  });

  it("parses GEXF node and edge attributes without inventing entities", async () => {
    const file = new File([
      `<?xml version="1.0" encoding="UTF-8"?>
      <gexf xmlns="http://gexf.net/1.3" version="1.3">
        <graph defaultedgetype="directed">
          <attributes class="node"><attribute id="0" title="type" type="string" /></attributes>
          <attributes class="edge"><attribute id="1" title="timestamp" type="string" /></attributes>
          <nodes>
            <node id="a" label="机构A"><attvalues><attvalue for="0" value="organization" /></attvalues></node>
            <node id="b" label="成员B"><attvalues><attvalue for="0" value="person" /></attvalues></node>
          </nodes>
          <edges><edge id="ab" source="a" target="b" label="隶属" weight="3"><attvalues><attvalue for="1" value="2026-08-10" /></attvalues></edge></edges>
        </graph>
      </gexf>`,
    ], "network.gexf", { type: "application/gexf+xml" });

    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("ready");
    expect(run.graphVersion?.nodes).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "a", label: "机构A", type: "organization" }),
      expect.objectContaining({ id: "b", label: "成员B", type: "person" }),
    ]));
    expect(run.graphVersion?.edges[0]).toMatchObject({
      id: "ab",
      type: "隶属",
      weight: 3,
      timestamp: "2026-08-10",
      directed: true,
    });
  });

  it("rejects XML entity declarations before DOM parsing", async () => {
    const file = new File([
      `<?xml version="1.0"?><!DOCTYPE graphml [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><graphml><graph><node id="&xxe;" /></graph></graphml>`,
    ], "unsafe.graphml", { type: "application/graphml+xml" });

    const run = await new LocalGraphImportAdapter().parse(file);

    expect(run.status).toBe("failed");
    expect(run.issues[0]).toMatchObject({ code: "unsafe_xml_construct", severity: "error" });
  });
});

describe("graph preview", () => {
  it("uses deterministic caps while keeping full-graph statistics", () => {
    const nodes: GraphNode[] = Array.from({ length: 305 }, (_, index) => ({
      id: `n-${index}`,
      label: `节点 ${index}`,
      attributes: {},
    }));
    const edges: GraphEdge[] = Array.from({ length: 1_005 }, (_, index) => ({
      id: `e-${index}`,
      source: `n-${index % 305}`,
      target: `n-${(index + 1) % 305}`,
      attributes: {},
    }));

    const first = createGraphVersion("large.json", nodes, edges);
    const second = createGraphVersion("large.json", nodes, edges);

    expect(first.summary).toMatchObject({ nodeCount: 305, edgeCount: 1_005 });
    expect(first.truncated).toBe(true);
    expect(first.preview.nodes.length).toBeLessThanOrEqual(MAX_PREVIEW_NODES);
    expect(first.preview.edges.length).toBeLessThanOrEqual(MAX_PREVIEW_EDGES);
    expect(first.preview.nodes.map((node) => node.id)).toEqual(second.preview.nodes.map((node) => node.id));
    expect(first.preview.edges.map((edge) => edge.id)).toEqual(second.preview.edges.map((edge) => edge.id));
  });
});

describe("import guardrails", () => {
  it("blocks files larger than 20MB before reading them", async () => {
    const file = new File([new Uint8Array(MAX_IMPORT_BYTES + 1)], "too-large.csv", {
      type: "text/csv",
    });
    const adapter = new LocalGraphImportAdapter();

    const profile = await adapter.inspect(file);
    const run = await adapter.parse(file);

    expect(profile.supported).toBe(false);
    expect(profile.issues[0].code).toBe("file_too_large");
    expect(run).toMatchObject({ status: "failed", issues: [{ code: "file_too_large" }] });
  });

  it("requires backend conversion for NPZ instead of loading arrays in the browser", async () => {
    const file = new File([new Uint8Array([0x50, 0x4b, 0x03, 0x04])], "research-graph.npz", {
      type: "application/x-npz",
    });
    const adapter = new LocalGraphImportAdapter();
    const profile = await adapter.inspect(file);
    const run = await adapter.parse(file);

    expect(profile).toMatchObject({ format: "npz", supported: false });
    expect(run.status).toBe("failed");
    expect(run.issues[0]).toMatchObject({ code: "backend_conversion_required", severity: "error" });
  });
});
