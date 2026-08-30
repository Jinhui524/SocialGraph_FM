export class BenchmarkStageTimeoutError extends Error {
  constructor(stage, timeoutMs) {
    super(`${stage} timed out after ${timeoutMs}ms`);
    this.name = "BenchmarkStageTimeoutError";
    this.code = "STAGE_TIMEOUT";
    this.stage = stage;
    this.timeoutMs = timeoutMs;
  }
}

/**
 * Applies a wall-clock deadline to one benchmark stage. The action is invoked
 * lazily so synchronous throws and never-settling promises share one contract.
 */
export async function withStageDeadline(stage, action, timeoutMs) {
  if (typeof stage !== "string" || !stage.trim()) {
    throw new TypeError("benchmark stage must be a non-empty string");
  }
  if (typeof action !== "function") {
    throw new TypeError("benchmark stage action must be a function");
  }
  if (!Number.isFinite(timeoutMs) || timeoutMs < 1) {
    throw new RangeError("benchmark stage timeout must be a positive number");
  }

  let timer;
  try {
    return await Promise.race([
      Promise.resolve().then(action),
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new BenchmarkStageTimeoutError(stage, timeoutMs)), timeoutMs);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** A deterministic fault primitive used only by the benchmark-runner tests. */
export function createNeverSettlingStage({ keepProcessAlive = false } = {}) {
  const keepAliveHandle = keepProcessAlive ? setInterval(() => undefined, 1_000) : undefined;
  return {
    promise: new Promise(() => undefined),
    release: () => {
      if (keepAliveHandle) clearInterval(keepAliveHandle);
    },
  };
}
