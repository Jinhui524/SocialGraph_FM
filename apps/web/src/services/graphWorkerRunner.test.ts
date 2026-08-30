import { describe, expect, it, vi } from "vitest";

import { GraphWorkerExecutionError, runGraphTask } from "./graphWorkerRunner";
import { createGraphVersion } from "./graphImport";

class WorkerDouble {
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  readonly postMessage = vi.fn();
  readonly terminate = vi.fn();
}

const request = {
  id: "worker-heavy-1",
  kind: "local_subgraph",
  graph: createGraphVersion("worker.json", [], []),
  focusNodeIds: [],
  depth: 1,
} as const;

describe("graph Worker failure containment", () => {
  it("returns a recoverable timeout instead of replaying the task on the UI thread", async () => {
    const worker = new WorkerDouble();
    const promise = runGraphTask(request, {
      timeoutMs: 1,
      workerFactory: () => worker as unknown as Worker,
    });

    await expect(promise).rejects.toMatchObject({
      code: "GRAPH_WORKER_TIMEOUT",
    });
    expect(worker.postMessage).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });

  it("returns a recoverable Worker failure without direct fallback", async () => {
    const worker = new WorkerDouble();
    const promise = runGraphTask(request, { workerFactory: () => worker as unknown as Worker });
    worker.onerror?.(new Event("error"));

    await expect(promise).rejects.toMatchObject({
      code: "GRAPH_WORKER_FAILED",
    });
    expect(worker.postMessage).toHaveBeenCalledOnce();
    expect(worker.terminate).toHaveBeenCalledOnce();
  });
});
