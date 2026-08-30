import { afterEach, describe, expect, it, vi } from "vitest";

import vectors from "../../../../contracts/core-inference-vectors.json";
import type {
  CoreCreatedRun,
  CoreRunBinding,
  CoreRunRequest,
  CoreRunResult,
  CoreRunStatus,
} from "../types/core";
import { SocialGraphApiError } from "./apiClient";
import { CoreRunController } from "./coreRunController";

const request = vectors.validRunRequests[0] as CoreRunRequest;
const serverRequestHash = vectors.validStatuses[0].requestHash;
const binding: CoreRunBinding = {
  runId: vectors.validStatuses[0].runId,
  publicRequestHash: "9".repeat(64),
  serverRequestHash,
  taskId: request.taskId,
  graphVersionId: request.graphVersionId,
  modelVersionId: request.modelVersionId,
};
const queued = {
  ...vectors.validStatuses[0],
  status: "queued",
  progress: 0,
  updatedAt: vectors.validStatuses[0].createdAt,
  errorCode: null,
  stateHash: "a".repeat(64),
} as CoreRunStatus;
const running = {
  ...queued,
  status: "running",
  progress: 10,
  updatedAt: "2026-08-14T00:00:00.500000Z",
  stateHash: "b".repeat(64),
} as CoreRunStatus;
const succeeded = vectors.validStatuses[0] as CoreRunStatus;
const result = vectors.validResults[0] as CoreRunResult;

afterEach(() => {
  vi.useRealTimers();
});

describe("GFM run controller", () => {
  it("advances queued -> running -> loading-result -> succeeded and commits only a bound result", async () => {
    vi.useFakeTimers();
    const client = {
      createRun: vi.fn().mockResolvedValue({ status: queued, binding } satisfies CoreCreatedRun),
      getRun: vi.fn()
        .mockResolvedValueOnce(running)
        .mockResolvedValueOnce(succeeded),
      getResult: vi.fn().mockResolvedValue(result),
    };
    const states: string[] = [];
    const controller = new CoreRunController(client, {
      initialPollMs: 100,
      maxPollMs: 200,
      maxPollAttempts: 4,
    });
    controller.subscribe((state) => states.push(state.phase));

    const started = controller.start(request);
    await vi.advanceTimersByTimeAsync(500);
    await started;

    expect(states).toEqual(expect.arrayContaining([
      "submitting",
      "polling",
      "loading-result",
      "succeeded",
    ]));
    expect(controller.getState()).toMatchObject({ phase: "succeeded", result });
  });

  it("stop-following aborts local polling without sending a server cancellation", async () => {
    vi.useFakeTimers();
    const client = {
      createRun: vi.fn().mockResolvedValue({ status: queued, binding } satisfies CoreCreatedRun),
      getRun: vi.fn().mockImplementation((_runId, _binding, signal: AbortSignal) => new Promise((_resolve, reject) => {
        signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), { once: true });
      })),
      getResult: vi.fn(),
    };
    const controller = new CoreRunController(client, { initialPollMs: 1, maxPollMs: 2, maxPollAttempts: 2 });

    void controller.start(request);
    await vi.advanceTimersByTimeAsync(2);
    controller.stopFollowing();
    await vi.runAllTimersAsync();

    expect(controller.getState()).toMatchObject({ phase: "detached", runId: binding.runId });
    expect(client.getResult).not.toHaveBeenCalled();
    expect(Object.keys(client)).toEqual(["createRun", "getRun", "getResult"]);
  });

  it("does not detach a submission before a server run ID is known", async () => {
    let resolveCreated!: (value: CoreCreatedRun) => void;
    const client = {
      createRun: vi.fn().mockReturnValue(new Promise<CoreCreatedRun>((resolve) => { resolveCreated = resolve; })),
      getRun: vi.fn(),
      getResult: vi.fn().mockResolvedValue(result),
    };
    const controller = new CoreRunController(client);
    const started = controller.start(request);

    controller.stopFollowing();
    expect(controller.getState()).toMatchObject({ phase: "submitting" });

    resolveCreated({ status: succeeded, binding });
    await started;
    expect(controller.getState()).toMatchObject({ phase: "succeeded", binding });
  });

  it("ignores a stale result after graph/task/model/target context is reset", async () => {
    let resolveResult!: (value: CoreRunResult) => void;
    const client = {
      createRun: vi.fn().mockResolvedValue({ status: succeeded, binding } satisfies CoreCreatedRun),
      getRun: vi.fn(),
      getResult: vi.fn().mockReturnValue(new Promise<CoreRunResult>((resolve) => { resolveResult = resolve; })),
    };
    const controller = new CoreRunController(client, { initialPollMs: 1, maxPollMs: 2, maxPollAttempts: 2 });
    const started = controller.start(request);
    await Promise.resolve();
    controller.reset("graph-context-changed");
    resolveResult(result);
    await started;

    expect(controller.getState()).toMatchObject({ phase: "idle", reason: "graph-context-changed" });
  });

  it("never publishes an unregistered upstream error code", async () => {
    const client = {
      createRun: vi.fn().mockRejectedValue(new SocialGraphApiError(
        "CHECKPOINT_C_USERS_PRIVATE_SECRET",
        "private",
        503,
      )),
      getRun: vi.fn(),
      getResult: vi.fn(),
    };
    const controller = new CoreRunController(client);

    await controller.start(request);

    expect(controller.getState()).toMatchObject({ phase: "failed", code: "GFM_CORE_RUN_FAILED" });
  });
});
