import { expect, test, type APIRequestContext, type APIResponse } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const SCHEMA = "socialgraph-fm.gfm-governance/2.0";

async function json<T>(response: APIResponse): Promise<T> {
  expect(response.ok(), `${response.status()} ${response.url()}: ${await response.text()}`).toBe(true);
  return response.json() as Promise<T>;
}

async function runRussiaPack(request: APIRequestContext, fileName: string) {
  const health = await json<{
    readonly onlineForwardReady: boolean;
    readonly modelVersionId: string;
    readonly modelStateHash: string;
  }>(await request.get("/api/v2/gfm/governance/health"));
  expect(health.onlineForwardReady).toBe(true);

  const buffer = readFileSync(resolve(process.cwd(), `../../examples/governance/russia/${fileName}`));
  const artifact = await json<{
    readonly artifactId: string;
    readonly datasetContentHash: string;
    readonly graphVersionHash: string;
    readonly nodeCount: number;
  }>(await request.post("/api/v2/gfm/governance/artifacts", {
    multipart: {
      file: { name: fileName, mimeType: "application/zip", buffer },
      cleanSelfLoops: "false",
    },
  }));
  const materialized = await json<{ readonly artifactId: string }>(
    await request.post(`/api/v2/gfm/governance/artifacts/${artifact.artifactId}/materialize`),
  );
  expect(materialized.artifactId).toBe(artifact.artifactId);

  const run = await json<{ readonly runId: string }>(await request.post("/api/v2/gfm/governance/runs", {
    data: {
      schemaVersion: SCHEMA,
      protocol: "global",
      artifactId: artifact.artifactId,
      datasetContentHash: artifact.datasetContentHash,
      graphVersionHash: artifact.graphVersionHash,
      modelVersionId: health.modelVersionId,
      modelStateHash: health.modelStateHash,
      topK: Math.min(100, artifact.nodeCount),
    },
  }));
  let completed: { readonly status: string; readonly stage: string } | undefined;
  await expect.poll(async () => {
    completed = await json<{ readonly status: string; readonly stage: string }>(
      await request.get(`/api/v2/gfm/governance/runs/${run.runId}`),
    );
    return completed.status;
  }, { timeout: 180_000, intervals: [250, 500, 1_000, 2_000] }).toBe("succeeded");
  expect(completed?.stage).toBe("completed");

  const result = await json<{ readonly resultHash: string }>(
    await request.get(`/api/v2/gfm/governance/runs/${run.runId}/result`),
  );
  expect(result.resultHash).toMatch(/^[0-9a-f]{64}$/u);
  const nodes = await json<{ readonly items: readonly { readonly nodeId: string }[] }>(
    await request.get(`/api/v2/gfm/governance/runs/${run.runId}/nodes?limit=5`),
  );
  expect(nodes.items.length).toBeGreaterThan(0);
  const nodeId = nodes.items[0]!.nodeId;
  await json(await request.get(
    `/api/v2/gfm/governance/runs/${run.runId}/nodes/${encodeURIComponent(nodeId)}/evidence`,
  ));
  await json(await request.get(`/api/v2/gfm/governance/runs/${run.runId}/groups?limit=10`));
  await json(await request.get(`/api/v2/gfm/governance/runs/${run.runId}/relations?limit=10`));

  let governanceCase = await json<{ readonly caseId: string; readonly state: string }>(
    await request.post("/api/v2/gfm/governance/cases", {
      data: {
        schemaVersion: SCHEMA,
        runId: run.runId,
        title: `${fileName} runtime acceptance`,
        description: "Public runtime matrix acceptance.",
      },
    }),
  );
  if (governanceCase.state === "draft") {
    governanceCase = await json<{ readonly caseId: string; readonly state: string }>(await request.post(
      `/api/v2/gfm/governance/cases/${governanceCase.caseId}/transitions`,
      { data: { schemaVersion: SCHEMA, state: "active", reason: "Begin acceptance review." } },
    ));
  }
  await json(await request.post(`/api/v2/gfm/governance/cases/${governanceCase.caseId}/items`, {
    data: { schemaVersion: SCHEMA, targetType: "node", targetId: nodeId, note: "Evidence checked." },
  }));
  await json(await request.post(`/api/v2/gfm/governance/cases/${governanceCase.caseId}/review-events`, {
    data: {
      schemaVersion: SCHEMA,
      targetType: "node",
      targetId: nodeId,
      decision: "confirmed",
      reason: "Runtime acceptance evidence review.",
      actor: "clean-clone-acceptance",
    },
  }));
  if (governanceCase.state === "active") {
    governanceCase = await json<{ readonly caseId: string; readonly state: string }>(await request.post(
      `/api/v2/gfm/governance/cases/${governanceCase.caseId}/transitions`,
      { data: { schemaVersion: SCHEMA, state: "concluded", reason: "Acceptance review completed." } },
    ));
  }
  expect(governanceCase.state).toBe("concluded");
  const report = await json<{ readonly case: { readonly caseId: string } }>(
    await request.get(`/api/v2/gfm/governance/cases/${governanceCase.caseId}/report?format=json`),
  );
  expect(report.case.caseId).toBe(governanceCase.caseId);
}

test.describe("@backend Russia 1-4 public runtime matrix", () => {
  for (const fileName of ["russia-01.zip", "russia-02.zip", "russia-03.zip", "russia-04.zip"]) {
    test(`${fileName} completes inference, evidence, governance, and report`, async ({ request }) => {
      test.setTimeout(240_000);
      await runRussiaPack(request, fileName);
    });
  }
});
