import { afterEach, describe, expect, it, vi } from "vitest";
import { SocialGraphApiError } from "./apiClient";
import { LocalAnalysisExecutor } from "./localAnalysisExecutor";
import {
  graphVersionFromDatasetArtifact,
  graphVersionTargetDomainFile,
  ResearchDatasetClient,
  type DatasetArtifact,
} from "./researchDatasetClient";
import type { GraphVersion } from "../types/graph";

function artifactFixture(overrides: Partial<DatasetArtifact> = {}): DatasetArtifact {
  return {
    id: "artifact-1",
    inspectionId: "inspection-1",
    sourceFormat: "sgfm-package",
    sourceFiles: ["graph.npz"],
    checksum: "checksum",
    canonicalGraphHash: "abcdef012345",
    scope: "projection",
    profile: { nodeCount: 100, edgeCount: 300, splitNames: [], directed: false },
    datasetName: "Cora",
    createdAt: "2026-08-11T00:00:00Z",
    rawManifest: {},
    derivedManifest: {},
    graphView: {
      id: "view-1",
      nodes: [{ id: "1", label: "Paper 1" }],
      edges: [],
      summary: {
        nodeCount: 100,
        edgeCount: 300,
        density: 0.06,
        connectedComponents: 1,
        visibleNodeCount: 1,
        visibleEdgeCount: 0,
        partialPreview: true,
      },
    },
    ...overrides,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

function graphVersionFixture(): GraphVersion {
  const nodes = [
    { id: "node-a", label: "社区甲", type: "社区", attributes: { district: "东区" } },
    { id: "node-b", label: "机构乙", type: "机构", attributes: {} },
  ] as const;
  const edges = [{
    id: "edge-1",
    source: "node-a",
    target: "node-b",
    type: "协作",
    weight: 0.75,
    timestamp: "2024-08-01",
    directed: true,
    attributes: { evidence: "公开记录" },
  }] as const;
  return {
    id: "graph-v1",
    sourceFile: "治理关系.csv",
    createdAt: "2026-08-11T00:00:00Z",
    nodes,
    edges,
    summary: {
      nodeCount: 2,
      edgeCount: 1,
      density: 1,
      averageDegree: 1,
      connectedComponents: 1,
      isolatedNodes: 0,
    },
    issues: [],
    preview: {
      nodes,
      edges,
      truncated: false,
      originalNodeCount: 2,
      originalEdgeCount: 1,
    },
    truncated: false,
    contentHash: "a".repeat(64),
    buildSpecHash: "b".repeat(64),
    metadata: { directedness: "directed" },
  };
}

describe("graphVersionFromDatasetArtifact", () => {
  it("marks partial research projections and preserves full summary counts", () => {
    const artifact = artifactFixture();
    const version = graphVersionFromDatasetArtifact(artifact);
    expect(version.nodes).toHaveLength(1);
    expect(version.summary.nodeCount).toBe(100);
    expect(version.datasetArtifact?.scope).toBe("projection");
    expect(version.issues[0]?.code).toBe("DATASET_ARTIFACT_PROJECTION");
  });

  it("prevents a browser-local algorithm from treating an Artifact projection as the full graph", async () => {
    const version = graphVersionFromDatasetArtifact(artifactFixture());
    const executor = new LocalAnalysisExecutor([version]);
    const run = await executor.createAnalysis({
      graphVersionId: version.id,
      intent: {
        kind: "analysis_request",
        normalizedText: "分析中心性",
        task: "centrality",
        targets: [],
        confidence: 1,
        filters: {},
        meta: {
          schemaVersion: "1.1",
          source: "llm",
          requestId: "projection-test",
          warnings: [],
        },
      },
    });

    expect(run.engine).toBe("unavailable");
    expect(run.status).toBe("failed");
    expect(run.result).toMatchObject({
      kind: "unavailable",
      code: "DATASET_PROJECTION_REQUIRES_SERVER_ANALYSIS",
      requestedTask: "centrality",
    });
  });
});

describe("ResearchDatasetClient API contract", () => {
  it("builds the strict data-only GraphVersion target-domain envelope", async () => {
    const file = graphVersionTargetDomainFile(graphVersionFixture());
    expect(file.name).toBe("graph-v1.sgfm-graph.json");
    const payload = JSON.parse(await file.text()) as Record<string, unknown>;
    expect(payload).toMatchObject({
      schemaVersion: "socialgraph-fm-graph/1.1",
      graphVersionId: "graph-v1",
      contentHash: "a".repeat(64),
      buildSpecHash: "b".repeat(64),
      directedness: "directed",
    });
    expect(payload).not.toHaveProperty("datasetRole");
    expect(payload).not.toHaveProperty("licensePolicy");
    expect(payload).not.toHaveProperty("trainingRef");
    expect(payload.nodes).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "node-a", label: "社区甲", type: "社区" }),
    ]));
    expect(payload.edges).toEqual([
      expect.objectContaining({ id: "edge-1", source: "node-a", target: "node-b", directed: true }),
    ]);
    // Cross-runtime fixture: the Python API's graph_fact_hash_v1 returns this exact digest.
    expect(payload.graphFactHash).toBe("3938b6dffcdbbea96d483347bc2e4578fca0c60bff14da2e1331fbe1a583f50c");
  });

  it("only prepares a target-domain Artifact when explicitly called", async () => {
    const inspection = {
      id: "inspection-graph-v1",
      detectedFormat: "graph_version_target_domain",
      status: "accepted",
      issues: [],
      datasetCandidates: [],
      serverGraphFactHash: "__GRAPH_FACT_HASH__",
    };
    const artifact = artifactFixture({
      id: "artifact-target",
      scope: "complete",
      datasetRole: "target_domain",
    });
    const envelope = JSON.parse(await graphVersionTargetDomainFile(graphVersionFixture()).text()) as {
      graphFactHash: string;
    };
    const binding = {
      id: "binding-1",
      graphVersionId: "graph-v1",
      graphFactHash: envelope.graphFactHash,
      artifactId: "artifact-target",
      preparationHash: "c".repeat(64),
      createdAt: "2026-08-11T00:00:00Z",
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        researchDatasets: {
          persistentArtifacts: true,
          trustedLocalEnabled: true,
          loopbackOnly: true,
          safeUploadFormats: ["graph_npz"],
        },
        runtime: {
          buildId: "test-build",
          apiContract: "socialgraph-fm-api/1.1",
          storageSchema: "dataset-store/2",
          datasetArtifactSchemas: ["1.0", "2.0", "2.1", "2.2"],
          trainingRefSchemas: ["1.0", "1.1"],
          graphHandoffSchemas: ["socialgraph-fm-graph/1.0", "socialgraph-fm-graph/1.1"],
          graphFactHash: "graph-fact-hash/1",
          converterEnvironmentFingerprint: "d".repeat(64),
        },
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...inspection,
        serverGraphFactHash: envelope.graphFactHash,
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        token: "handoff-token-with-at-least-twenty-characters",
        graphVersionId: "graph-v1",
        graphFactHash: envelope.graphFactHash,
        expiresAt: "2026-08-11T00:05:00Z",
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ binding, artifact, reused: false }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const prepared = await new ResearchDatasetClient("http://127.0.0.1:8000/api/v1")
      .prepareGraphVersionTargetDomain(graphVersionFixture());

    expect(prepared.artifact.datasetRole).toBe("target_domain");
    expect(prepared.binding).toEqual(binding);
    expect(prepared.reused).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("http://127.0.0.1:8000/api/v1/capabilities");
    expect(fetchMock.mock.calls[1]?.[0]).toBe("http://127.0.0.1:8000/api/v1/dataset-imports/inspect");
    expect(fetchMock.mock.calls[2]?.[0]).toBe(
      "http://127.0.0.1:8000/api/v1/graph-dataset-handoffs/reserve",
    );
    expect(fetchMock.mock.calls[3]?.[0]).toBe(
      "http://127.0.0.1:8000/api/v1/graph-dataset-handoffs/commit",
    );
  });

  it("reads FastAPI camelCase capability fields without translating them", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      researchDatasets: {
        persistentArtifacts: true,
        trustedLocalEnabled: true,
        loopbackOnly: true,
        safeUploadFormats: ["graph_npz"],
      },
      runtime: {
        buildId: "test-build",
        apiContract: "socialgraph-fm-api/1.1",
        storageSchema: "dataset-store/2",
        datasetArtifactSchemas: ["1.0", "2.0", "2.1", "2.2"],
        trainingRefSchemas: ["1.0", "1.1"],
        graphHandoffSchemas: ["socialgraph-fm-graph/1.0", "socialgraph-fm-graph/1.1"],
        graphFactHash: "graph-fact-hash/1",
        converterEnvironmentFingerprint: "d".repeat(64),
      },
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(new ResearchDatasetClient("http://127.0.0.1:8000/api/v1").capabilities())
      .resolves.toEqual({
        persistentArtifacts: true,
        trustedLocalEnabled: true,
        loopbackOnly: true,
        safeUploadFormats: ["graph_npz"],
        runtime: {
          buildId: "test-build",
          apiContract: "socialgraph-fm-api/1.1",
          storageSchema: "dataset-store/2",
          datasetArtifactSchemas: ["1.0", "2.0", "2.1", "2.2"],
          trainingRefSchemas: ["1.0", "1.1"],
          graphHandoffSchemas: ["socialgraph-fm-graph/1.0", "socialgraph-fm-graph/1.1"],
          graphFactHash: "graph-fact-hash/1",
          converterEnvironmentFingerprint: "d".repeat(64),
        },
      });
  });

  it("sends the trusted-local authorization aliases required by FastAPI", async () => {
    const responseJob = {
      id: "job-1",
      sourcePath: "E:\\trusted\\data",
      trustedRoot: "E:\\trusted",
      status: "queued",
      progress: 0,
      fileCount: 1,
      totalBytes: 10,
      datasets: [],
      artifactIds: [],
      issues: [],
      converterPython: "python",
      createdAt: "2026-08-11T00:00:00Z",
      updatedAt: "2026-08-11T00:00:00Z",
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...responseJob,
        status: "awaiting_authorization",
        authorizationToken: "token-with-at-least-twenty-characters",
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(responseJob), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    const client = new ResearchDatasetClient("http://127.0.0.1:8000/api/v1");
    await client.inspectLocal("E:\\trusted\\data");
    await client.authorize("job-1", "token-with-at-least-twenty-characters");

    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toEqual({
      sourcePath: "E:\\trusted\\data",
    });
    expect(JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body))).toEqual({
      authorizationToken: "token-with-at-least-twenty-characters",
      confirmTrusted: true,
    });
  });

  it("surfaces FastAPI validation messages instead of a generic HTTP 422", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: [{ loc: ["body", "sourcePath"], msg: "String should have at least 1 character", type: "string_too_short" }],
    }), { status: 422, headers: { "Content-Type": "application/json" } })));

    await expect(new ResearchDatasetClient().inspectLocal(""))
      .rejects.toThrow("String should have at least 1 character");
  });

  it("preserves a stable FastAPI detail.code for graph handoff failures", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(JSON.stringify({
      detail: { code: "GRAPH_FACT_HASH_MISMATCH" },
    }), { status: 409, headers: { "Content-Type": "application/json" } })));

    const error = await new ResearchDatasetClient().reserveGraphHandoff("graph-v1", "a".repeat(64))
      .catch((candidate: unknown) => candidate);
    expect(error).toBeInstanceOf(SocialGraphApiError);
    expect(error).toMatchObject({ code: "GRAPH_FACT_HASH_MISMATCH", status: 409 });
  });

  it("re-inspects a multi-dataset package with the selected dataset field", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      id: "inspection-2",
      detectedFormat: "socialgraph_dataset_package",
      status: "accepted",
      issues: [],
      datasetCandidates: ["alpha", "beta"],
    }), { status: 200, headers: { "Content-Type": "application/json" } }));
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["package"], "graphs.sgfm.zip", { type: "application/zip" });

    const result = await new ResearchDatasetClient().inspectPackage(file, "beta");

    expect(result.datasetCandidates).toEqual(["alpha", "beta"]);
    const request = fetchMock.mock.calls[0]?.[1] as RequestInit;
    expect(request.body).toBeInstanceOf(FormData);
    const body = request.body as FormData;
    expect(body.get("dataset")).toBe("beta");
    expect((body.get("files") as File).name).toBe("graphs.sgfm.zip");
  });

  it("checks an immutable training reference and resolves only the selected contract ids", async () => {
    const readiness = {
      artifactId: "artifact-21",
      status: "ready",
      contentHash: "a".repeat(64),
      manifestHash: "b".repeat(64),
      blockers: [],
      warnings: [],
      checkedAt: "2026-08-11T00:00:00Z",
    };
    const reference = {
      schemaVersion: "1.0",
      artifactId: "artifact-21",
      contentHash: "a".repeat(64),
      graphVariant: "raw",
      splitSetId: "official",
      splitFold: 0,
      featureRecipeId: "identity-v1",
      taskSpecId: "node-classification-v1",
      datasetRole: "benchmark",
      intendedUse: "evaluation",
      refHash: "c".repeat(64),
    } as const;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(readiness), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ reference, readiness }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new ResearchDatasetClient("http://127.0.0.1:8000/api/v1");

    await client.getReadiness("artifact-21", reference.refHash);
    await client.resolveTrainingRef(reference);

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      `http://127.0.0.1:8000/api/v1/dataset-artifacts/artifact-21/readiness?trainingRefHash=${reference.refHash}`,
    );
    expect(JSON.parse(String(fetchMock.mock.calls[1]?.[1]?.body))).toEqual({
      artifactId: "artifact-21",
      contentHash: "a".repeat(64),
      graphVariant: "raw",
      splitSetId: "official",
      splitFold: 0,
      featureRecipeId: "identity-v1",
      taskSpecId: "node-classification-v1",
      intendedUse: "evaluation",
    });
  });

  it("uses the reference-safe Artifact lifecycle contract and forwards the fresh impact hash", async () => {
    const impact = {
      artifactId: "artifact-life",
      lifecycle: "trashed",
      blockers: [],
      dependents: [{
        kind: "embedded_training_ref",
        id: "ref-1",
        blocking: false,
        detail: {},
      }],
      preserved: ["dataset_store_audit"],
      impactHash: "f".repeat(64),
    } as const;
    const lifecycleResponse = {
      lifecycle: {
        artifactId: "artifact-life",
        status: "trashed",
        updatedAt: "2026-08-11T00:00:00Z",
        trashedAt: "2026-08-11T00:00:00Z",
      },
      impact,
    } as const;
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify([{ id: "artifact-life", lifecycle: "trashed" }]), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(impact), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(lifecycleResponse), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...lifecycleResponse,
        lifecycle: { ...lifecycleResponse.lifecycle, status: "active", trashedAt: null },
      }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        artifactId: "artifact-life",
        purged: true,
        cleanupPending: false,
      }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    const client = new ResearchDatasetClient("http://127.0.0.1:8000/api/v1");

    await client.listArtifacts(true);
    await client.getDeletionImpact("artifact-life");
    await client.trashArtifact("artifact-life");
    await client.restoreArtifact("artifact-life");
    await client.purgeArtifact("artifact-life", impact.impactHash, "act-life");

    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      "http://127.0.0.1:8000/api/v1/dataset-artifacts?includeTrashed=true",
      "http://127.0.0.1:8000/api/v1/dataset-artifacts/artifact-life/deletion-impact",
      "http://127.0.0.1:8000/api/v1/dataset-artifacts/artifact-life/trash",
      "http://127.0.0.1:8000/api/v1/dataset-artifacts/artifact-life/restore",
      "http://127.0.0.1:8000/api/v1/dataset-artifacts/artifact-life/purge",
    ]);
    expect(fetchMock.mock.calls[2]?.[1]).toMatchObject({ method: "POST" });
    expect(fetchMock.mock.calls[3]?.[1]).toMatchObject({ method: "POST" });
    expect(JSON.parse(String(fetchMock.mock.calls[4]?.[1]?.body))).toEqual({
      impactHash: impact.impactHash,
      confirmation: "act-life",
    });
  });
});
