import { runGraphTaskDirect } from "./graphWorkerExecutor";
import type { GraphWorkerRequest, GraphWorkerResponse } from "./graphWorkerProtocol";

const workerScope = globalThis as unknown as {
  onmessage: ((event: MessageEvent<GraphWorkerRequest>) => void) | null;
  postMessage(message: GraphWorkerResponse): void;
};

workerScope.onmessage = (event) => {
  try {
    workerScope.postMessage({ id: event.data.id, ok: true, result: runGraphTaskDirect(event.data) });
  } catch (error) {
    workerScope.postMessage({
      id: event.data.id,
      ok: false,
      error: error instanceof Error ? error.message : "GRAPH_WORKER_FAILED",
    });
  }
};
