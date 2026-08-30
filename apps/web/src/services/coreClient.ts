import type {
  CoreCapabilities,
  CoreClientLike,
  CoreCreatedRun,
  CoreRunBinding,
  CoreRunRequest,
  CoreRunResult,
  CoreRunStatus,
} from "../types/core";
import { SocialGraphApiError, socialGraphApiUrl } from "./apiClient";
import {
  deepFreeze,
  parseCoreCapabilities,
  parseCoreError,
  parseCoreRunRequest,
  parseCoreRunResult,
  parseCoreRunStatus,
} from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";

type Fetcher = typeof fetch;

const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
const RUN_ID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/u;

const SAFE_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  GFM_CORE_GRAPH_VERSION_NOT_FOUND: "当前图版本尚未完成目标域交接。",
  GFM_CORE_MODEL_GRAPH_INCOMPATIBLE: "服务端确认当前模型与图合同不兼容。",
  GFM_CORE_MODEL_NOT_INSTALLED: "当前没有可执行的 SocialGraph-FM Core 模型。",
  GFM_CORE_SERVICE_UNAVAILABLE: "SocialGraph-FM Core 服务暂不可用。",
  GFM_CORE_RUN_NOT_FOUND: "未找到对应的 SocialGraph-FM Core 运行。",
  GFM_CORE_RESULT_NOT_READY: "SocialGraph-FM Core 结果尚未就绪。",
};

function assertRunId(runId: string): void {
  if (!RUN_ID.test(runId)) throw new Error("GFM_CORE_RUN_ID_INVALID");
}

async function readPayload(response: Response): Promise<unknown> {
  const contentLength = response.headers.get("Content-Length");
  if (contentLength && Number(contentLength) > MAX_RESPONSE_BYTES) {
    await response.body?.cancel().catch(() => undefined);
    throw new SocialGraphApiError("GFM_CORE_RESPONSE_TOO_LARGE", "SocialGraph-FM Core 响应超过浏览器安全上限。", 502);
  }
  let text = "";
  if (response.body) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let byteLength = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        byteLength += value.byteLength;
        if (byteLength > MAX_RESPONSE_BYTES) {
          await reader.cancel().catch(() => undefined);
          throw new SocialGraphApiError("GFM_CORE_RESPONSE_TOO_LARGE", "SocialGraph-FM Core 响应超过浏览器安全上限。", 502);
        }
        text += decoder.decode(value, { stream: true });
      }
      text += decoder.decode();
    } finally {
      reader.releaseLock();
    }
  } else {
    text = await response.text();
    if (new TextEncoder().encode(text).byteLength > MAX_RESPONSE_BYTES) {
      throw new SocialGraphApiError("GFM_CORE_RESPONSE_TOO_LARGE", "SocialGraph-FM Core 响应超过浏览器安全上限。", 502);
    }
  }
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new SocialGraphApiError("GFM_CORE_RESPONSE_INVALID", "SocialGraph-FM Core 返回了无法验证的响应。", 502);
  }
  if (!response.ok) {
    let code = "GFM_CORE_SERVICE_UNAVAILABLE";
    try {
      code = parseCoreError(payload).code;
    } catch {
      code = "GFM_CORE_RESPONSE_INVALID";
    }
    throw new SocialGraphApiError(
      code,
      SAFE_ERROR_MESSAGES[code] ?? "SocialGraph-FM Core 请求未完成；服务器细节已隐藏。",
      response.status,
    );
  }
  return payload;
}

function responseInvalid(error: unknown): never {
  if (error instanceof SocialGraphApiError) throw error;
  if (error instanceof DOMException && error.name === "AbortError") throw error;
  throw new SocialGraphApiError("GFM_CORE_RESPONSE_INVALID", "SocialGraph-FM Core 返回了无法验证的响应。", 502);
}

export class CoreClient implements CoreClientLike {
  private readonly baseUrl: string;
  private readonly fetcher: Fetcher;

  constructor(
    baseUrl = socialGraphApiUrl("/api/v1/gfm"),
    fetcher: Fetcher = globalThis.fetch,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/u, "");
    this.fetcher = fetcher.bind(globalThis);
  }

  async capabilities(signal?: AbortSignal): Promise<CoreCapabilities> {
    try {
      const payload = await readPayload(await this.fetcher(`${this.baseUrl}/capabilities`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      }));
      return parseCoreCapabilities(payload);
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async createRun(request: CoreRunRequest, signal?: AbortSignal): Promise<CoreCreatedRun> {
    const validatedRequest = parseCoreRunRequest(request);
    try {
      const payload = await readPayload(await this.fetcher(`${this.baseUrl}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(validatedRequest),
        signal,
      }));
      const status = parseCoreRunStatus(payload);
      assertRunId(status.runId);
      const binding = deepFreeze({
        runId: status.runId,
        publicRequestHash: sha256Canonical(validatedRequest),
        serverRequestHash: status.requestHash,
        taskId: validatedRequest.taskId,
        graphVersionId: validatedRequest.graphVersionId,
        modelVersionId: validatedRequest.modelVersionId,
      } satisfies CoreRunBinding);
      return deepFreeze({ status, binding });
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async getRun(
    runId: string,
    binding: CoreRunBinding,
    signal?: AbortSignal,
  ): Promise<CoreRunStatus> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_CORE_RESPONSE_BINDING_INVALID");
    try {
      const payload = await readPayload(await this.fetcher(`${this.baseUrl}/runs/${runId}`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      }));
      return parseCoreRunStatus(payload, binding);
    } catch (error) {
      if (error instanceof Error && error.message === "GFM_CORE_RESPONSE_BINDING_INVALID") throw error;
      return responseInvalid(error);
    }
  }

  async getResult(
    runId: string,
    binding: CoreRunBinding,
    signal?: AbortSignal,
  ): Promise<CoreRunResult> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_CORE_RESPONSE_BINDING_INVALID");
    try {
      const payload = await readPayload(await this.fetcher(`${this.baseUrl}/runs/${runId}/result`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      }));
      return parseCoreRunResult(payload, binding);
    } catch (error) {
      if (error instanceof Error && error.message === "GFM_CORE_RESPONSE_BINDING_INVALID") throw error;
      return responseInvalid(error);
    }
  }
}
