import type { GraphVersion, GraphVersionDiffReport } from "../types/graph";
import { computeGraphVersionDiffCore } from "./graphVersionDiffCore";

interface DiffWorkerRequest {
  readonly id: string;
  readonly from: GraphVersion;
  readonly to: GraphVersion;
  readonly sampleLimit?: number;
}

type DiffWorkerResponse =
  | { readonly id: string; readonly ok: true; readonly report: GraphVersionDiffReport }
  | { readonly id: string; readonly ok: false; readonly error: string };

const workerScope = self as unknown as {
  onmessage: ((event: MessageEvent<DiffWorkerRequest>) => void) | null;
  postMessage(message: DiffWorkerResponse): void;
};

workerScope.onmessage = (event) => {
  const { id, from, to, sampleLimit } = event.data;
  try {
    workerScope.postMessage({
      id,
      ok: true,
      report: computeGraphVersionDiffCore(from, to, { sampleLimit }),
    });
  } catch (error) {
    workerScope.postMessage({
      id,
      ok: false,
      error: error instanceof Error ? error.message : "GRAPH_VERSION_DIFF_FAILED",
    });
  }
};

