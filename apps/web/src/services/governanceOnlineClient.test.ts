import { describe, expect, it, vi } from "vitest";

import { adaptationComparison, targetLabelRecipe, targetLabelSet, targetLabelSourceRecordHash, targetPackageReceipt, targetReviewPolicy } from "../test/fixtures/governanceAdaptation";
import {
  casePayload,
  artifactCompatibility,
  comparisonPayload,
  derivationPage,
  evidencePayload,
  onlineArtifact,
  onlineCapabilities,
  onlineRunPreview,
  onlineRun,
  GOVERNANCE_ARTIFACT_ID,
  GOVERNANCE_OTHER_RUN_ID,
  GOVERNANCE_RUN_ID,
} from "../test/fixtures/governanceOnline";
import { GovernanceOnlineClient } from "./governanceOnlineClient";
import { adaptationHandoff, TARGET_REGISTRATION_ID, targetActivation, targetComparison, targetPolicy, targetReviewCollection, targetTaskRegistration } from "../test/fixtures/governanceTargetTask";

function response(payload: unknown, status = 200, contentType = "application/json"): Response {
  return new Response(contentType === "application/json" ? JSON.stringify(payload) : String(payload), { status, headers: { "Content-Type": contentType } });
}

describe("SocialGraph-FM Governance browser client", () => {
  it("registers one sgtask package and uses the v2 fit, handoff, and activation routes", async () => {
    const registration = targetTaskRegistration();
    const policy = targetPolicy();
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(registration, 201))
      .mockResolvedValueOnce(response(registration))
      .mockResolvedValueOnce(response(registration.labels, 201))
      .mockResolvedValueOnce(response(policy, 201))
      .mockResolvedValueOnce(response(targetComparison()))
      .mockResolvedValueOnce(response(adaptationHandoff(), 201))
      .mockResolvedValueOnce(response(targetActivation(), 201));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    const file = new File(["PK target"], "regional.sgtask.zip", { type: "application/zip" });
    await client.registerTargetTask(file);
    await client.targetTask(TARGET_REGISTRATION_ID);
    await client.createTargetLabelSet({ schemaVersion: "socialgraph-fm.governance-target-label-set/2.0", sourceType: "imported_sidecar", targetTaskRegistrationId: TARGET_REGISTRATION_ID, runId: GOVERNANCE_RUN_ID, resultHash: onlineRunPreview().resultHash! });
    await client.fitTargetPolicy(policy.labelSetHash, {
      schemaVersion: "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
      targetTaskRegistrationId: TARGET_REGISTRATION_ID,
      runId: GOVERNANCE_RUN_ID,
      resultHash: onlineRunPreview().resultHash!,
    });
    await client.targetComparison(GOVERNANCE_RUN_ID, policy.policyHash);
    await client.createAdaptationHandoff({ schemaVersion: "socialgraph-fm.governance-adaptation-handoff/1.0", targetTaskRegistrationId: TARGET_REGISTRATION_ID, policyHash: policy.policyHash, decision: "pending_governance_review" });
    await client.activateTargetPolicy(policy.policyHash, TARGET_REGISTRATION_ID);
    expect(fetcher.mock.calls.map((call) => [String(call[0]), (call[1] as RequestInit).method ?? "GET"])).toEqual([
      ["http://api.test/api/v2/gfm/governance/target-tasks", "POST"],
      [`http://api.test/api/v2/gfm/governance/target-tasks/${TARGET_REGISTRATION_ID}`, "GET"],
      ["http://api.test/api/v2/gfm/governance/adaptations/label-sets", "POST"],
      [`http://api.test/api/v2/gfm/governance/adaptations/label-sets/${policy.labelSetHash}/policies`, "POST"],
      [`http://api.test/api/v2/gfm/governance/adaptations/runs/${GOVERNANCE_RUN_ID}/policies/${policy.policyHash}/comparison?offset=0&limit=500`, "GET"],
      ["http://api.test/api/v2/gfm/governance/adaptations/handoffs", "POST"],
      [`http://api.test/api/v2/gfm/governance/adaptations/policies/${policy.policyHash}/activate`, "POST"],
    ]);
    expect(JSON.parse(String((fetcher.mock.calls[3]?.[1] as RequestInit).body))).toEqual({
      schemaVersion: "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
      targetTaskRegistrationId: TARGET_REGISTRATION_ID,
      runId: GOVERNANCE_RUN_ID,
      resultHash: onlineRunPreview().resultHash,
    });
  });

  it("creates an atomic target review collection for a zero-shot handoff", async () => {
    const request = {
      schemaVersion: "socialgraph-fm.governance-review-collection/1.0" as const,
      idempotencyKey: "zero-review", targetTaskRegistrationId: TARGET_REGISTRATION_ID, runId: GOVERNANCE_RUN_ID,
      resultHash: onlineRunPreview().resultHash!, title: "风险候选复核", description: "",
      items: [{ targetType: "node" as const, targetId: "n1", note: "风险排序 #1" }],
    };
    const collection = targetReviewCollection(request);
    const fetcher = vi.fn().mockResolvedValueOnce(response(collection, 201));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.createTargetReviewCollection(request)).resolves.toMatchObject({ case: { state: "active" } });
    expect(fetcher).toHaveBeenCalledWith(
      "http://api.test/api/v2/gfm/governance/adaptations/review-collections",
      expect.objectContaining({ method: "POST" }),
    );
  });
  it("materializes the registered Russia sample instead of parsing its receipt as an inference artifact", async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response(onlineCapabilities())).mockResolvedValueOnce(response(onlineArtifact()));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.russiaSample()).resolves.toMatchObject({ artifactId: GOVERNANCE_ARTIFACT_ID });
    expect(fetcher.mock.calls.map((call) => [call[0], call[1]?.method ?? "GET"])).toEqual([
      ["http://api.test/api/v2/gfm/governance/capabilities", "GET"],
      [`http://api.test/api/v2/gfm/governance/artifacts/${GOVERNANCE_ARTIFACT_ID}/materialize`, "POST"],
    ]);
  });

  it("uploads only ZIP bundles with an explicit self-loop decision", async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response(artifactCompatibility(1))).mockResolvedValueOnce(response(onlineArtifact(), 201));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    const file = new File(["PK contract"], "graph.zip", { type: "application/zip" });
    await expect(client.inspectArtifact(file)).resolves.toMatchObject({ requiresSelfLoopCleaning: true });
    await client.uploadArtifact(file, true);
    const init = fetcher.mock.calls[1]?.[1] as RequestInit;
    expect(fetcher.mock.calls[0]?.[0]).toBe("http://api.test/api/v2/gfm/governance/artifacts/compatibility");
    expect(fetcher.mock.calls[1]?.[0]).toBe("http://api.test/api/v2/gfm/governance/artifacts");
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    expect((init.body as FormData).get("cleanSelfLoops")).toBe("true");
    await expect(client.uploadArtifact(new File(["{}"], "graph.json"), false)).rejects.toMatchObject({ code: "GFM_GOVERNANCE_UPLOAD_TYPE_INVALID" });
    const oversized = new File(["x"], "oversized.zip", { type: "application/zip" });
    Object.defineProperty(oversized, "size", { value: 256 * 1024 * 1024 + 1 });
    await expect(client.inspectArtifact(oversized)).rejects.toMatchObject({ code: "GFM_GOVERNANCE_UPLOAD_SIZE_INVALID" });
    expect(fetcher).toHaveBeenCalledTimes(2);
  });

  it("uses the read-only compare route and append-only governance routes", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(comparisonPayload()))
      .mockResolvedValueOnce(response(casePayload()))
      .mockResolvedValueOnce(response({ ...casePayload(), state: "active" }))
      .mockResolvedValueOnce(response(casePayload()));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await client.compareRuns(GOVERNANCE_OTHER_RUN_ID, GOVERNANCE_RUN_ID);
    await client.createCase(GOVERNANCE_RUN_ID, "Global 研判", "本机研判单");
    await client.updateCase(casePayload().caseId, "active", "开始人工研判");
    await client.review(casePayload().caseId, "node", "n1", "pending", "等待补充证据");
    const urls = fetcher.mock.calls.map((call) => String(call[0]));
    expect(urls[0]).toContain("/runs/compare?leftRunId=");
    expect(urls[1]?.endsWith("/cases")).toBe(true);
    expect(urls[2]?.endsWith(`/cases/${casePayload().caseId}/transitions`)).toBe(true);
    expect(urls[3]?.endsWith(`/cases/${casePayload().caseId}/review-events`)).toBe(true);
    expect(JSON.parse(String((fetcher.mock.calls[3]?.[1] as RequestInit).body))).toMatchObject({ targetType: "node", actor: "local-analyst" });
  });

  it("loads the scored graph projection from the run-bound route", async () => {
    const fetcher = vi.fn().mockResolvedValueOnce(response(onlineRunPreview()));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.runPreview(GOVERNANCE_RUN_ID)).resolves.toMatchObject({ runId: GOVERNANCE_RUN_ID, resultHash: onlineRunPreview().resultHash });
    expect(fetcher).toHaveBeenCalledWith(
      `http://api.test/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/graph-preview`,
      expect.objectContaining({ headers: expect.objectContaining({ Accept: "application/json" }) }),
    );
  });

  it("encodes bounded projection presets without changing the preview endpoints", async () => {
    const fetcher = vi.fn().mockResolvedValue(response(onlineRunPreview()));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await client.runPreview(GOVERNANCE_RUN_ID, undefined, {
      preset: "evidence",
      nodeBudget: 60,
      edgeBudget: 120,
      anchorNodeIds: ["n1", "node/2"],
    });
    expect(String(fetcher.mock.calls[0]?.[0])).toBe(
      `http://api.test/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/graph-preview?preset=evidence&nodeBudget=60&edgeBudget=120&anchorNodeId=n1&anchorNodeId=node%2F2`,
    );
  });

  it("rejects unsafe identifiers before issuing network requests", async () => {
    const fetcher = vi.fn();
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    expect(() => client.run("../run")).toThrow("GFM_GOVERNANCE_PATH_ID_INVALID");
    expect(fetcher).not.toHaveBeenCalled();
    expect(onlineRun().runId).toBe(GOVERNANCE_RUN_ID);
  });

  it("encodes slash and backslash in a legal node-id path segment", async () => {
    const nodeId = "../tenant\\account";
    const payload = evidencePayload();
    payload.node = { ...payload.node, nodeId };
    payload.evidenceSubgraph.nodes[0] = { ...payload.evidenceSubgraph.nodes[0], nodeId };
    payload.evidenceSubgraph.edges[0] = { ...payload.evidenceSubgraph.edges[0], source: nodeId };
    const fetcher = vi.fn().mockResolvedValueOnce(response(payload));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.evidence(GOVERNANCE_RUN_ID, nodeId)).resolves.toMatchObject({ node: { nodeId } });
    const requestedUrl = String(fetcher.mock.calls[0]?.[0]);
    expect(requestedUrl).toBe(`http://api.test/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/nodes/..%2Ftenant%5Caccount/evidence`);
    expect(requestedUrl).not.toContain("/../");
  });

  it("loads every derivation and global case page before filtering by run", async () => {
    const groupPageOne = derivationPage("group");
    const groupPageTwo = derivationPage("group");
    groupPageOne.total = 2; groupPageOne.limit = 10_000;
    groupPageTwo.total = 2; groupPageTwo.offset = 1; groupPageTwo.limit = 10_000;
    groupPageTwo.items[0] = { ...groupPageTwo.items[0], id: "group-2" };
    const otherCase = { ...casePayload(), caseId: `case-${"5".repeat(32)}`, runId: GOVERNANCE_OTHER_RUN_ID };
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(groupPageOne))
      .mockResolvedValueOnce(response(groupPageTwo))
      .mockResolvedValueOnce(response({ schemaVersion: casePayload().schemaVersion, items: [otherCase], total: 2, offset: 0, limit: 100 }))
      .mockResolvedValueOnce(response({ schemaVersion: casePayload().schemaVersion, items: [casePayload()], total: 2, offset: 1, limit: 100 }));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    await expect(client.derivations(GOVERNANCE_RUN_ID, "group")).resolves.toHaveLength(2);
    await expect(client.listCases(GOVERNANCE_RUN_ID)).resolves.toEqual([expect.objectContaining({ runId: GOVERNANCE_RUN_ID })]);
    expect(fetcher.mock.calls.map((call) => String(call[0]))).toEqual([
      `http://api.test/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/groups?offset=0&limit=10000`,
      `http://api.test/api/v2/gfm/governance/runs/${GOVERNANCE_RUN_ID}/groups?offset=1&limit=10000`,
      "http://api.test/api/v2/gfm/governance/cases?offset=0&limit=100",
      "http://api.test/api/v2/gfm/governance/cases?offset=1&limit=100",
    ]);
  });

  it("uses the immutable target label-set, policy, and bounded comparison routes", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(targetLabelSet(), 201))
      .mockResolvedValueOnce(response(targetReviewPolicy(), 201))
      .mockResolvedValueOnce(response(targetReviewPolicy()))
      .mockResolvedValueOnce(response(adaptationComparison()));
    const client = new GovernanceOnlineClient("http://api.test/api/v2/gfm/governance", fetcher as unknown as typeof fetch);
    const request = {
      schemaVersion: "socialgraph-fm.governance-target-label-set/1.1" as const,
      runId: GOVERNANCE_RUN_ID,
      resultHash: "c".repeat(64),
      sidecarReceipt: targetPackageReceipt(),
      sources: targetLabelRecipe().labels.map((row) => ({
        sourceType: "imported_sidecar" as const,
        sourceRecordId: `thailand-authorized:${row.nodeId}`,
        sourceRecordHash: targetLabelSourceRecordHash(row),
        nodeId: row.nodeId,
        cohort: row.label,
        structuralStratum: row.structuralStratum,
        fusedDegree: row.fusedDegree,
        labelsSha256: targetPackageReceipt().labelsSha256,
        receiptHash: targetPackageReceipt().receiptHash,
      })),
    };

    await client.createAdaptationLabelSet(request);
    await client.fitAdaptationPolicy(targetLabelSet().labelSetHash);
    await client.adaptationPolicy(targetReviewPolicy().policyHash);
    await client.adaptationComparison(GOVERNANCE_RUN_ID, targetReviewPolicy().policyHash, 0, 100);

    expect(fetcher.mock.calls.map((call) => [String(call[0]), (call[1] as RequestInit).method ?? "GET"])).toEqual([
      ["http://api.test/api/v2/gfm/governance/adaptations/label-sets", "POST"],
      [`http://api.test/api/v2/gfm/governance/adaptations/label-sets/${targetLabelSet().labelSetHash}/policies`, "POST"],
      [`http://api.test/api/v2/gfm/governance/adaptations/policies/${targetReviewPolicy().policyHash}`, "GET"],
      [`http://api.test/api/v2/gfm/governance/adaptations/runs/${GOVERNANCE_RUN_ID}/policies/${targetReviewPolicy().policyHash}/comparison?offset=0&limit=100`, "GET"],
    ]);
    expect(JSON.parse(String((fetcher.mock.calls[0]?.[1] as RequestInit).body))).toEqual(request);
  });
});
