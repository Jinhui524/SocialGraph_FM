import type {
  CoreCreatedRun,
  CoreRunBinding,
  CoreRunRequest,
  CoreRunResult,
  CoreRunStatus,
} from "../types/core";
import { SocialGraphApiError } from "./apiClient";
import { deepFreeze, isPublicCoreErrorCode } from "./coreContracts";

const LOCAL_FAILURE_CODES: ReadonlySet<string> = new Set([
  "GFM_CORE_RESPONSE_BINDING_INVALID",
  "GFM_CORE_RUN_ID_INVALID",
]);

export interface CoreRunControllerClient {
  createRun(request: CoreRunRequest, signal?: AbortSignal): Promise<CoreCreatedRun>;
  getRun(runId: string, binding: CoreRunBinding, signal?: AbortSignal): Promise<CoreRunStatus>;
  getResult(runId: string, binding: CoreRunBinding, signal?: AbortSignal): Promise<CoreRunResult>;
}

export interface CoreRunControllerOptions {
  readonly initialPollMs?: number;
  readonly maxPollMs?: number;
  readonly maxPollAttempts?: number;
}

export type CoreRunControllerState =
  | { readonly phase: "idle"; readonly reason?: string }
  | { readonly phase: "submitting"; readonly request: CoreRunRequest }
  | {
      readonly phase: "polling";
      readonly binding: CoreRunBinding;
      readonly status: CoreRunStatus;
      readonly attempt: number;
    }
  | {
      readonly phase: "loading-result";
      readonly binding: CoreRunBinding;
      readonly status: CoreRunStatus;
    }
  | {
      readonly phase: "succeeded";
      readonly binding: CoreRunBinding;
      readonly status: CoreRunStatus;
      readonly result: CoreRunResult;
    }
  | {
      readonly phase: "failed";
      readonly code: string;
      readonly runId?: string;
      readonly binding?: CoreRunBinding;
    }
  | {
      readonly phase: "detached";
      readonly runId: string;
      readonly binding: CoreRunBinding;
      readonly serverMayContinue: true;
    };

type Listener = (state: CoreRunControllerState) => void;

function positiveInteger(value: number | undefined, fallback: number): number {
  return Number.isInteger(value) && (value ?? 0) > 0 ? value as number : fallback;
}

function safeFailureCode(error: unknown): string {
  if (error instanceof SocialGraphApiError) {
    return isPublicCoreErrorCode(error.code) ? error.code : "GFM_CORE_RUN_FAILED";
  }
  if (
    error instanceof Error
    && (isPublicCoreErrorCode(error.message) || LOCAL_FAILURE_CODES.has(error.message))
  ) return error.message;
  return "GFM_CORE_RUN_FAILED";
}

export class CoreRunController {
  private readonly initialPollMs: number;
  private readonly maxPollMs: number;
  private readonly maxPollAttempts: number;
  private readonly listeners = new Set<Listener>();
  private state: CoreRunControllerState = Object.freeze({ phase: "idle" });
  private epoch = 0;
  private abortController: AbortController | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private activeBinding: CoreRunBinding | null = null;
  private disposed = false;

  constructor(
    private readonly client: CoreRunControllerClient,
    options: CoreRunControllerOptions = {},
  ) {
    this.initialPollMs = positiveInteger(options.initialPollMs, 500);
    this.maxPollMs = Math.max(this.initialPollMs, positiveInteger(options.maxPollMs, 4_000));
    this.maxPollAttempts = positiveInteger(options.maxPollAttempts, 120);
  }

  getState(): CoreRunControllerState {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  private publish(state: CoreRunControllerState): void {
    if (this.disposed) return;
    this.state = deepFreeze(state);
    for (const listener of this.listeners) listener(this.state);
  }

  private invalidate(): number {
    this.epoch += 1;
    this.abortController?.abort();
    this.abortController = null;
    if (this.timer !== null) clearTimeout(this.timer);
    this.timer = null;
    return this.epoch;
  }

  private current(epoch: number): boolean {
    return !this.disposed && this.epoch === epoch;
  }

  private delay(milliseconds: number, signal: AbortSignal): Promise<void> {
    return new Promise((resolve, reject) => {
      if (signal.aborted) {
        reject(new DOMException("aborted", "AbortError"));
        return;
      }
      const aborted = () => {
        if (this.timer !== null) clearTimeout(this.timer);
        this.timer = null;
        reject(new DOMException("aborted", "AbortError"));
      };
      signal.addEventListener("abort", aborted, { once: true });
      this.timer = setTimeout(() => {
        this.timer = null;
        signal.removeEventListener("abort", aborted);
        resolve();
      }, milliseconds);
    });
  }

  async start(request: CoreRunRequest): Promise<void> {
    const epoch = this.invalidate();
    if (this.disposed) return;
    this.activeBinding = null;
    const abortController = new AbortController();
    this.abortController = abortController;
    this.publish({ phase: "submitting", request });
    try {
      const created = await this.client.createRun(request, abortController.signal);
      if (!this.current(epoch)) return;
      this.activeBinding = created.binding;
      if (created.status.status === "failed") {
        this.publish({
          phase: "failed",
          code: created.status.errorCode ?? "GFM_CORE_RUN_FAILED",
          runId: created.binding.runId,
          binding: created.binding,
        });
        return;
      }
      if (created.status.status === "succeeded") {
        await this.loadResult(created.binding, created.status, epoch, abortController.signal);
        return;
      }

      let status = created.status;
      for (let attempt = 1; attempt <= this.maxPollAttempts; attempt += 1) {
        if (!this.current(epoch)) return;
        this.publish({ phase: "polling", binding: created.binding, status, attempt });
        const milliseconds = Math.min(this.maxPollMs, this.initialPollMs * (2 ** Math.min(8, attempt - 1)));
        await this.delay(milliseconds, abortController.signal);
        if (!this.current(epoch)) return;
        status = await this.client.getRun(created.binding.runId, created.binding, abortController.signal);
        if (!this.current(epoch)) return;
        if (status.status === "failed") {
          this.publish({
            phase: "failed",
            code: status.errorCode ?? "GFM_CORE_RUN_FAILED",
            runId: created.binding.runId,
            binding: created.binding,
          });
          return;
        }
        if (status.status === "succeeded") {
          await this.loadResult(created.binding, status, epoch, abortController.signal);
          return;
        }
      }
      if (this.current(epoch)) {
        this.publish({
          phase: "failed",
          code: "GFM_CORE_POLL_LIMIT_REACHED",
          runId: created.binding.runId,
          binding: created.binding,
        });
      }
    } catch (error) {
      if (!this.current(epoch)) return;
      if (error instanceof DOMException && error.name === "AbortError") return;
      this.publish({
        phase: "failed",
        code: safeFailureCode(error),
        ...(this.activeBinding ? { runId: this.activeBinding.runId, binding: this.activeBinding } : {}),
      });
    } finally {
      if (this.current(epoch)) this.abortController = null;
    }
  }

  private async loadResult(
    binding: CoreRunBinding,
    status: CoreRunStatus,
    epoch: number,
    signal: AbortSignal,
  ): Promise<void> {
    if (!this.current(epoch)) return;
    this.publish({ phase: "loading-result", binding, status });
    const result = await this.client.getResult(binding.runId, binding, signal);
    if (!this.current(epoch)) return;
    this.publish({ phase: "succeeded", binding, status, result });
  }

  stopFollowing(): void {
    const binding = this.activeBinding;
    if (!binding) return;
    this.invalidate();
    this.publish({
      phase: "detached",
      runId: binding.runId,
      binding,
      serverMayContinue: true,
    });
  }

  reset(reason = "context-changed"): void {
    this.invalidate();
    this.activeBinding = null;
    this.publish({ phase: "idle", reason });
  }

  dispose(): void {
    this.invalidate();
    this.disposed = true;
    this.listeners.clear();
    this.activeBinding = null;
  }
}
