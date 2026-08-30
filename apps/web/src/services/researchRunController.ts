import type {
  ResearchCreatedRun,
  ResearchRunBinding,
  ResearchRunRequest,
  ResearchRunResult,
  ResearchRunStatus,
} from "../types/research";
import { SocialGraphApiError } from "./apiClient";
import { deepFreeze } from "./coreContracts";

const PUBLIC_FAILURE_CODES = new Set([
  "GFM_RESEARCH_MODEL_NOT_INSTALLED",
  "GFM_RESEARCH_GRAPH_VERSION_NOT_FOUND",
  "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING",
  "GFM_RESEARCH_GRAPH_INCOMPATIBLE",
  "GFM_RESEARCH_MODEL_MISMATCH",
  "GFM_RESEARCH_GRAPH_IDENTITY_CONFLICT",
  "GFM_RESEARCH_GRAPH_ARTIFACT_MISSING",
  "GFM_RESEARCH_SCENARIO_UNAVAILABLE",
  "GFM_RESEARCH_SCENARIO_NOT_FOUND",
  "GFM_RESEARCH_SCENARIO_MISMATCH",
  "GFM_RESEARCH_RESULT_NOT_READY",
  "GFM_RESEARCH_RUN_NOT_FOUND",
  "GFM_RESEARCH_SERVICE_UNAVAILABLE",
  "GFM_RESEARCH_RESPONSE_INVALID",
  "GFM_RESEARCH_RUN_STATUS_INVALID",
  "GFM_RESEARCH_RUN_RESULT_INVALID",
  "GFM_RESEARCH_RESPONSE_BINDING_INVALID",
  "GFM_RESEARCH_RUN_ID_INVALID",
  "GFM_RESEARCH_POLL_LIMIT_REACHED",
] as const);

export interface ResearchRunControllerClient {
  createRun(request: ResearchRunRequest, signal?: AbortSignal): Promise<ResearchCreatedRun>;
  getRun(
    runId: string,
    binding: ResearchRunBinding,
    signal?: AbortSignal,
  ): Promise<ResearchRunStatus>;
  getResult(
    runId: string,
    binding: ResearchRunBinding,
    signal?: AbortSignal,
  ): Promise<ResearchRunResult>;
}

export type ResearchRunControllerState =
  | { readonly phase: "idle"; readonly reason?: string }
  | { readonly phase: "submitting"; readonly request: ResearchRunRequest }
  | {
      readonly phase: "polling";
      readonly binding: ResearchRunBinding;
      readonly status: ResearchRunStatus;
      readonly attempt: number;
    }
  | {
      readonly phase: "loading-result";
      readonly binding: ResearchRunBinding;
      readonly status: ResearchRunStatus;
    }
  | {
      readonly phase: "succeeded";
      readonly binding: ResearchRunBinding;
      readonly status: ResearchRunStatus;
      readonly result: ResearchRunResult;
    }
  | {
      readonly phase: "failed";
      readonly code: string;
      readonly runId?: string;
      readonly binding?: ResearchRunBinding;
    }
  | {
      readonly phase: "detached";
      readonly runId: string;
      readonly binding: ResearchRunBinding;
      readonly serverMayContinue: true;
    };

interface ResearchRunControllerOptions {
  readonly initialPollMs?: number;
  readonly maxPollMs?: number;
  readonly maxPollAttempts?: number;
  readonly maxRegistrationAttempts?: number;
}

type Listener = (state: ResearchRunControllerState) => void;

function positiveInteger(value: number | undefined, fallback: number): number {
  return Number.isInteger(value) && (value ?? 0) > 0 ? value as number : fallback;
}

function safeFailureCode(error: unknown): string {
  const candidate = error instanceof SocialGraphApiError
    ? error.code
    : error instanceof Error
      ? error.message
      : "";
  return PUBLIC_FAILURE_CODES.has(candidate as never) ? candidate : "GFM_RESEARCH_RUN_FAILED";
}

export class ResearchRunController {
  private readonly initialPollMs: number;
  private readonly maxPollMs: number;
  private readonly maxPollAttempts: number;
  private readonly maxRegistrationAttempts: number;
  private readonly listeners = new Set<Listener>();
  private state: ResearchRunControllerState = Object.freeze({ phase: "idle" });
  private epoch = 0;
  private abortController: AbortController | null = null;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private activeBinding: ResearchRunBinding | null = null;
  private disposed = false;

  constructor(
    private readonly client: ResearchRunControllerClient,
    options: ResearchRunControllerOptions = {},
  ) {
    this.initialPollMs = positiveInteger(options.initialPollMs, 500);
    this.maxPollMs = Math.max(this.initialPollMs, positiveInteger(options.maxPollMs, 4_000));
    this.maxPollAttempts = positiveInteger(options.maxPollAttempts, 120);
    this.maxRegistrationAttempts = positiveInteger(options.maxRegistrationAttempts, 20);
  }

  getState(): ResearchRunControllerState {
    return this.state;
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  private publish(state: ResearchRunControllerState): void {
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

  async start(request: ResearchRunRequest): Promise<void> {
    const epoch = this.invalidate();
    if (this.disposed) return;
    this.activeBinding = null;
    const abortController = new AbortController();
    this.abortController = abortController;
    this.publish({ phase: "submitting", request });
    try {
      let created: ResearchCreatedRun | null = null;
      for (let attempt = 1; attempt <= this.maxRegistrationAttempts; attempt += 1) {
        try {
          created = await this.client.createRun(request, abortController.signal);
          break;
        } catch (error) {
          const registrationPending = error instanceof SocialGraphApiError
            && error.code === "GFM_RESEARCH_GRAPH_REGISTRATION_PENDING";
          if (!registrationPending || attempt === this.maxRegistrationAttempts) throw error;
          await this.delay(
            Math.min(this.maxPollMs, this.initialPollMs * (2 ** Math.min(8, attempt - 1))),
            abortController.signal,
          );
          if (!this.current(epoch)) return;
        }
      }
      if (!created) throw new Error("GFM_RESEARCH_GRAPH_REGISTRATION_PENDING");
      if (!this.current(epoch)) return;
      this.activeBinding = created.binding;
      if (created.status.status === "failed") {
        this.publish({
          phase: "failed",
          code: safeFailureCode(new Error(created.status.errorCode ?? "")),
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
        const waitMs = Math.min(this.maxPollMs, this.initialPollMs * (2 ** Math.min(8, attempt - 1)));
        await this.delay(waitMs, abortController.signal);
        if (!this.current(epoch)) return;
        status = await this.client.getRun(created.binding.runId, created.binding, abortController.signal);
        if (!this.current(epoch)) return;
        if (status.status === "failed") {
          this.publish({
            phase: "failed",
            code: safeFailureCode(new Error(status.errorCode ?? "")),
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
          code: "GFM_RESEARCH_POLL_LIMIT_REACHED",
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
    binding: ResearchRunBinding,
    status: ResearchRunStatus,
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
    this.publish({ phase: "detached", runId: binding.runId, binding, serverMayContinue: true });
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
