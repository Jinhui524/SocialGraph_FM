import { describe, expect, it } from "vitest";

import { createGraphVersion } from "./graphImport";
import { createLocalGraphRepository } from "./graphRepository";
import { createSourceArtifact } from "./sourceArtifact";
import {
  CURRENT_BROWSER_IMPORT_PIPELINE_VERSION,
  inspectGraphVersionCompatibility,
} from "./graphVersionCompatibility";

describe("GraphVersion compatibility", () => {
  it("identifies the uniform-colour legacy condition without mutating the graph", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = createGraphVersion("legacy.csv", [
      { id: "a", label: "A", attributes: {} },
      { id: "b", label: "B", attributes: {} },
    ], [{ id: "ab", source: "a", target: "b", attributes: {} }]);
    const legacy = { ...graph, id: "graph-z06396", sourceArtifactIds: undefined, buildSpec: undefined, provenance: undefined };
    const before = JSON.stringify(legacy);

    const compatibility = await inspectGraphVersionCompatibility(repository, legacy);

    expect(compatibility).toMatchObject({
      status: "legacy_read_only",
      allNodesUntyped: true,
      canDeterministicallyRebuild: false,
    });
    expect(compatibility.message).toContain("未分类");
    expect(JSON.stringify(legacy)).toBe(before);
  });

  it("offers deterministic rebuilding only when every SourceArtifact is present", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const artifact = await createSourceArtifact(new File(["source,target\na,b\n"], "edges.csv"));
    await repository.saveSourceArtifact(artifact);
    const graph = {
      ...createGraphVersion("edges.csv", [{ id: "a", label: "A", type: "person", attributes: {} }], []),
      sourceArtifactIds: [artifact.id],
    };

    expect(await inspectGraphVersionCompatibility(repository, graph)).toMatchObject({
      status: "upgrade_available",
      canDeterministicallyRebuild: true,
      missingSourceArtifactIds: [],
    });
  });

  it("recognizes provenance from the current browser import pipeline", async () => {
    const repository = createLocalGraphRepository({ forceMemory: true });
    const graph = {
      ...createGraphVersion("current.json", [{ id: "a", label: "A", type: "person", attributes: {} }], []),
      provenance: {
        origin: "browser_import" as const,
        pipeline: "browser-import" as const,
        pipelineVersion: CURRENT_BROWSER_IMPORT_PIPELINE_VERSION,
      },
    };

    expect(await inspectGraphVersionCompatibility(repository, graph)).toMatchObject({
      status: "current",
      allNodesUntyped: false,
    });
  });
});

