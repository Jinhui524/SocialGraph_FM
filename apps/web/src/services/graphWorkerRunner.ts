import { runGraphTaskDirect } from "./graphWorkerExecutor";
import type { GraphWorkerRequest, GraphWorkerResponse, GraphWorkerResult } from "./graphWorkerProtocol";

export { runGraphTaskDirect } from "./graphWorkerExecutor";

export interface RunGraphTaskOptions {
  readonly preferWorker?: boolean;
  readonly timeoutMs?: number;
  readonly workerFactory?: () => Worker;
}

export type GraphWorkerFailureCode =
  | "GRAPH_WORKER_UNAVAILABLE"
  | "GRAPH_WORKER_TIMEOUT"
  | "GRAPH_WORKER_FAILED";

export class GraphWorkerExecutionError extends Error {
  constructor(readonly code: GraphWorkerFailureCode) {
    super(code);
    this.name = "GraphWorkerExecutionError";
  }
}

/** Once a heavy task is dispatched to a Worker, failures never replay it on the UI thread. */
export async function runGraphTask(
  request: GraphWorkerRequest,
  options: RunGraphTaskOptions = {},
): Promise<GraphWorkerResult> {
  if (options.preferWorker === false || (typeof Worker === "undefined" && !options.workerFactory)) {
    return runGraphTaskDirect(request);
  }

  let worker: Worker;
  try {
    worker = options.workerFactory?.()
      ?? new Worker(new URL("./graphAnalysis.worker.ts", import.meta.url), { type: "module" });
  } catch {
    throw new GraphWorkerExecutionError("GRAPH_WORKER_UNAVAILABLE");
  }
  try {
    return await new Promise<GraphWorkerResult>((resolve, reject) => {
      const timeout = globalThis.setTimeout(
        () => reject(new GraphWorkerExecutionError("GRAPH_WORKER_TIMEOUT")),
        options.timeoutMs ?? 30_000,
      );
      worker.onmessage = (event: MessageEvent<GraphWorkerResponse>) => {
        if (event.data.id !== request.id) return;
        globalThis.clearTimeout(timeout);
        if (event.data.ok) resolve(event.data.result);
        else reject(new Error(event.data.error));
      };
      worker.onerror = () => {
        globalThis.clearTimeout(timeout);
        reject(new GraphWorkerExecutionError("GRAPH_WORKER_FAILED"));
      };
      worker.postMessage(request);
    });
  } finally {
    worker.terminate();
  }
}
