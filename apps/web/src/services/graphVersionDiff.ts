import type { GraphRepository, GraphVersion, GraphVersionDiffReport } from "../types/graph";
import {
  computeGraphVersionDiffCore,
  DEFAULT_GRAPH_VERSION_DIFF_SAMPLE_LIMIT,
} from "./graphVersionDiffCore";

export interface GraphVersionDiffOptions {
  readonly sampleLimit?: number;
  readonly workerThreshold?: number;
  readonly timeoutMs?: number;
  readonly forceSynchronous?: boolean;
  readonly workerFactory?: () => Worker;
}

function requestId() {
  return typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `diff-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function factCount(version: GraphVersion) {
  return version.nodes.length + version.edges.length;
}

async function computeInWorker(
  from: GraphVersion,
  to: GraphVersion,
  sampleLimit: number,
  timeoutMs: number,
  workerFactory?: () => Worker,
): Promise<GraphVersionDiffReport> {
  const worker = workerFactory?.()
    ?? new Worker(new URL("./graphVersionDiff.worker.ts", import.meta.url), { type: "module" });
  const id = requestId();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      worker.terminate();
      reject(new Error("GRAPH_VERSION_DIFF_TIMEOUT"));
    }, timeoutMs);
    worker.onmessage = (event: MessageEvent<{
      readonly id: string;
      readonly ok: boolean;
      readonly report?: GraphVersionDiffReport;
      readonly error?: string;
    }>) => {
      if (event.data.id !== id) return;
      window.clearTimeout(timeout);
      worker.terminate();
      if (event.data.ok && event.data.report) resolve(event.data.report);
      else reject(new Error(event.data.error ?? "GRAPH_VERSION_DIFF_FAILED"));
    };
    worker.onerror = () => {
      window.clearTimeout(timeout);
      worker.terminate();
      reject(new Error("GRAPH_VERSION_DIFF_WORKER_FAILED"));
    };
    worker.postMessage({ id, from, to, sampleLimit });
  });
}

/** Uses a module Worker for large versions and a deterministic fallback elsewhere. */
export async function computeGraphVersionDiff(
  from: GraphVersion,
  to: GraphVersion,
  options: GraphVersionDiffOptions = {},
): Promise<GraphVersionDiffReport> {
  const sampleLimit = Math.max(0, Math.floor(options.sampleLimit ?? DEFAULT_GRAPH_VERSION_DIFF_SAMPLE_LIMIT));
  const workerThreshold = Math.max(0, options.workerThreshold ?? 10_000);
  const canUseWorker = !options.forceSynchronous
    && typeof window !== "undefined"
    && (typeof Worker !== "undefined" || Boolean(options.workerFactory))
    && factCount(from) + factCount(to) >= workerThreshold;
  if (!canUseWorker) return computeGraphVersionDiffCore(from, to, { sampleLimit });
  // Once dispatched, a failed large diff is never replayed synchronously on the UI thread.
  return computeInWorker(from, to, sampleLimit, options.timeoutMs ?? 30_000, options.workerFactory);
}

export async function diffGraphVersionsById(
  repository: Pick<GraphRepository, "getGraphVersion">,
  fromVersionId: string,
  toVersionId: string,
  options: GraphVersionDiffOptions = {},
): Promise<GraphVersionDiffReport> {
  const [from, to] = await Promise.all([
    repository.getGraphVersion(fromVersionId),
    repository.getGraphVersion(toVersionId),
  ]);
  if (!from) throw new Error(`GRAPH_VERSION_NOT_FOUND：${fromVersionId}`);
  if (!to) throw new Error(`GRAPH_VERSION_NOT_FOUND：${toVersionId}`);
  return computeGraphVersionDiff(from, to, options);
}
