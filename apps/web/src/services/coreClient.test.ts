import { afterEach, describe, expect, it, vi } from "vitest";

import vectors from "../../../../contracts/core-inference-vectors.json";
import type { CoreRunRequest } from "../types/core";
import { SocialGraphApiError } from "./apiClient";
import { CoreClient } from "./coreClient";
import { sha256Canonical } from "./graphIdentity";

const request = vectors.validRunRequests[0] as CoreRunRequest;
const publicCapabilities = {
  ...vectors.validCapabilities[0],
  schemaVersion: "socialgraph-fm.core-capabilities/2.0",
};
const status = vectors.validStatuses[0];
const result = vectors.validResults[0];

afterEach(() => {
  vi.unstubAllGlobals();
});

function response(payload: unknown, statusCode = 200): Response {
  return new Response(JSON.stringify(payload), {
    status: statusCode,
    headers: { "Content-Type": "application/json" },
  });
}

function rehash<T extends Record<string, unknown>>(value: T, field: string): T {
  const payload = Object.fromEntries(Object.entries(value).filter(([key]) => key !== field));
  return { ...value, [field]: sha256Canonical(payload) };
}

describe("core GFM API client", () => {
  it("binds the default browser fetch receiver instead of causing an illegal invocation", async () => {
    const nativeLikeFetch = vi.fn(function (this: unknown) {
      if (this !== globalThis) throw new TypeError("Illegal invocation");
      return Promise.resolve(response(publicCapabilities));
    });
    vi.stubGlobal("fetch", nativeLikeFetch);
    const client = new CoreClient("http://api.test/api/v1/gfm");

    await expect(client.capabilities()).resolves.toMatchObject({
      schemaVersion: "socialgraph-fm.core-capabilities/2.0",
    });
    expect(nativeLikeFetch).toHaveBeenCalledTimes(1);
  });

  it("uses capabilities as the only task source and never requests the deleted tasks route", async () => {
    const fetcher = vi.fn().mockResolvedValue(response(publicCapabilities));
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);

    await expect(client.capabilities()).resolves.toMatchObject({
      schemaVersion: "socialgraph-fm.core-capabilities/2.0",
      servingReady: false,
      models: [],
      tasks: [],
    });

    expect(fetcher).toHaveBeenCalledTimes(1);
    expect(fetcher.mock.calls[0]?.[0]).toBe("http://api.test/api/v1/gfm/capabilities");
    expect(String(fetcher.mock.calls[0]?.[0])).not.toContain("/tasks");
    expect((client as unknown as Record<string, unknown>).tasks).toBeUndefined();
  });

  it("stops reading and cancels a chunked response once the 8 MiB limit is crossed", async () => {
    let pulls = 0;
    let cancelled = false;
    const chunk = new Uint8Array(1024 * 1024);
    const body = new ReadableStream<Uint8Array>({
      pull(controller) {
        pulls += 1;
        if (pulls > 20) {
          controller.close();
          return;
        }
        controller.enqueue(chunk);
      },
      cancel() {
        cancelled = true;
      },
    });
    const fetcher = vi.fn().mockResolvedValue(new Response(body, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }));

    await expect(new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch).capabilities())
      .rejects.toMatchObject({ code: "GFM_CORE_RESPONSE_TOO_LARGE" });
    expect(cancelled).toBe(true);
    expect(pulls).toBeLessThan(20);
  });

  it("sends the exact public request and keeps public and opaque server hashes distinct", async () => {
    const fetcher = vi.fn().mockResolvedValue(response(status, 202));
    const signal = new AbortController().signal;
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);

    const created = await client.createRun(request, signal);

    expect(fetcher.mock.calls[0]?.[0]).toBe("http://api.test/api/v1/gfm/runs");
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({ method: "POST", signal });
    expect(JSON.parse(String(fetcher.mock.calls[0]?.[1]?.body))).toEqual(request);
    expect(created.binding).toMatchObject({
      runId: status.runId,
      serverRequestHash: status.requestHash,
      taskId: request.taskId,
      graphVersionId: request.graphVersionId,
      modelVersionId: request.modelVersionId,
    });
    expect(created.binding.publicRequestHash).toMatch(/^[0-9a-f]{64}$/u);
    expect(created.binding.publicRequestHash).not.toBe(created.binding.serverRequestHash);
    expect(Object.isFrozen(created)).toBe(true);
    expect(Object.isFrozen(created.binding)).toBe(true);
  });

  it("binds every later status and result to the create receipt and original public identity", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(status, 202))
      .mockResolvedValueOnce(response(status))
      .mockResolvedValueOnce(response(result));
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);
    const created = await client.createRun(request);

    await expect(client.getRun(created.binding.runId, created.binding)).resolves.toMatchObject({
      runId: created.binding.runId,
      requestHash: created.binding.serverRequestHash,
      status: "succeeded",
    });
    await expect(client.getResult(created.binding.runId, created.binding)).resolves.toMatchObject({
      runId: created.binding.runId,
      requestHash: created.binding.serverRequestHash,
      taskId: request.taskId,
      graphVersionId: request.graphVersionId,
      modelVersionId: request.modelVersionId,
    });
    expect(fetcher.mock.calls.map((call) => call[0])).toEqual([
      "http://api.test/api/v1/gfm/runs",
      `http://api.test/api/v1/gfm/runs/${status.runId}`,
      `http://api.test/api/v1/gfm/runs/${status.runId}/result`,
    ]);
  });

  it.each([
    ["status server hash", rehash({ ...status, requestHash: "2".repeat(64) }, "stateHash"), "getRun"],
    ["result graph", rehash({ ...result, graphVersionId: "graph-substituted" }, "resultHash"), "getResult"],
    ["result model", rehash({ ...result, modelVersionId: "socialgraph-fm-core/substituted" }, "resultHash"), "getResult"],
    ["result task", rehash({ ...result, taskId: "core.community_resilience_review" }, "resultHash"), "getResult"],
  ] as const)("rejects a coherent-looking substituted %s", async (_label, payload, operation) => {
    const createFetcher = vi.fn().mockResolvedValue(response(status, 202));
    const first = new CoreClient("http://api.test/api/v1/gfm", createFetcher as unknown as typeof fetch);
    const created = await first.createRun(request);
    const fetcher = vi.fn().mockResolvedValue(response(payload));
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);

    const promise = operation === "getRun"
      ? client.getRun(created.binding.runId, created.binding)
      : client.getResult(created.binding.runId, created.binding);
    await expect(promise).rejects.toThrow("GFM_CORE_RESPONSE_BINDING_INVALID");
  });

  it("forwards AbortSignal on all four calls", async () => {
    const fetcher = vi.fn()
      .mockResolvedValueOnce(response(publicCapabilities))
      .mockResolvedValueOnce(response(status, 202))
      .mockResolvedValueOnce(response(status))
      .mockResolvedValueOnce(response(result));
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);
    const controller = new AbortController();

    await client.capabilities(controller.signal);
    const created = await client.createRun(request, controller.signal);
    await client.getRun(created.binding.runId, created.binding, controller.signal);
    await client.getResult(created.binding.runId, created.binding, controller.signal);

    expect(fetcher.mock.calls.every((call) => call[1]?.signal === controller.signal)).toBe(true);
  });

  it.each([
    [404, "GFM_CORE_GRAPH_VERSION_NOT_FOUND"],
    [409, "GFM_CORE_MODEL_GRAPH_INCOMPATIBLE"],
    [503, "GFM_CORE_MODEL_NOT_INSTALLED"],
  ])("maps HTTP %s to a stable closed error without exposing a raw server message", async (statusCode, code) => {
    const fetcher = vi.fn().mockResolvedValue(response({ detail: { code } }, statusCode));
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);

    const error = await client.createRun(request).catch((candidate: unknown) => candidate);

    expect(error).toBeInstanceOf(SocialGraphApiError);
    expect(error).toMatchObject({ code, status: statusCode });
    expect((error as Error).message).not.toContain("private");
    expect((error as Error).message).not.toContain("checkpoint.pt");
  });

  it("rejects an error envelope with private-message extras without rendering the extra", async () => {
    const fetcher = vi.fn().mockResolvedValue(response({
      detail: { code: "GFM_CORE_MODEL_NOT_INSTALLED", message: "C:\\private\\runtime\\checkpoint.pt" },
    }, 503));
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);

    const error = await client.createRun(request).catch((candidate: unknown) => candidate);

    expect(error).toMatchObject({ code: "GFM_CORE_RESPONSE_INVALID", status: 503 });
    expect((error as Error).message).not.toContain("private");
    expect((error as Error).message).not.toContain("checkpoint.pt");
  });

  it.each(["capabilities", "createRun", "getRun", "getResult"] as const)(
    "maps an unregistered hostile code from %s to GFM_CORE_RESPONSE_INVALID",
    async (operation) => {
      const fetcher = vi.fn().mockResolvedValue(response({
        detail: { code: "CHECKPOINT_C_USERS_PRIVATE_SECRET" },
      }, 503));
      const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);
      const binding = {
        runId: status.runId,
        publicRequestHash: "9".repeat(64),
        serverRequestHash: status.requestHash,
        taskId: request.taskId,
        graphVersionId: request.graphVersionId,
        modelVersionId: request.modelVersionId,
      } as const;
      const promise = operation === "capabilities"
        ? client.capabilities()
        : operation === "createRun"
          ? client.createRun(request)
          : operation === "getRun"
            ? client.getRun(binding.runId, binding)
            : client.getResult(binding.runId, binding);

      const error = await promise.catch((candidate: unknown) => candidate);
      expect(error).toMatchObject({ code: "GFM_CORE_RESPONSE_INVALID", status: 503 });
      expect(JSON.stringify(error)).not.toContain("PRIVATE_SECRET");
    },
  );

  it("rejects an unsafe run ID before constructing a URL", async () => {
    const fetcher = vi.fn();
    const client = new CoreClient("http://api.test/api/v1/gfm", fetcher as unknown as typeof fetch);
    const binding = {
      runId: "../internal",
      publicRequestHash: "1".repeat(64),
      serverRequestHash: "2".repeat(64),
      taskId: request.taskId,
      graphVersionId: request.graphVersionId,
      modelVersionId: request.modelVersionId,
    } as const;

    await expect(client.getRun(binding.runId, binding)).rejects.toThrow("GFM_CORE_RUN_ID_INVALID");
    expect(fetcher).not.toHaveBeenCalled();
  });
});
