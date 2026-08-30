import type {
  GlobalModelCapabilities,
  GlobalModelClientLike,
  GlobalModelCreatedRun,
  GlobalModelHealth,
  GlobalModelModelCard,
  GlobalModelNodeEvidence,
  GlobalModelReviewRecord,
  GlobalModelReviewRequest,
  GlobalModelRunBinding,
  GlobalModelRunIdentity,
  GlobalModelRunRequest,
  GlobalModelRunResult,
  GlobalModelRunStatus,
  GlobalModelScenario,
  GlobalModelScenarioPreview,
} from "../types/globalModel";
import { SocialGraphApiError, socialGraphApiUrl } from "./apiClient";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";
import {
  parseGlobalModelCapabilities,
  parseGlobalModelHealth,
  parseGlobalModelModelCard,
  parseGlobalModelNodeEvidence,
  parseGlobalModelReviewRecord,
  parseGlobalModelReviewRequest,
  parseGlobalModelRunRequest,
  parseGlobalModelRunResult,
  parseGlobalModelRunStatus,
  parseGlobalModelScenario,
  parseGlobalModelScenarioPreview,
} from "./globalModelContracts";

type Fetcher = typeof fetch;

const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
const RUN_ID = /^global-model-[0-9a-f]{32}$/u;
const NODE_ID = /^[^/\\]{1,100}$/u;

export const GLOBAL_MODEL_API_ROUTES = Object.freeze({
  root: "/api/v1/gfm/global-model",
  health: "/health",
  capabilities: "/capabilities",
  modelCard: "/model-card",
  scenario: "/scenario",
  scenarioPreview: "/scenario/graph-preview",
  runs: "/runs",
  run: (runId: string) => `/runs/${encodeURIComponent(runId)}`,
  result: (runId: string) => `/runs/${encodeURIComponent(runId)}/result`,
  evidence: (runId: string, nodeId: string) => (
    `/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/evidence`
  ),
  reviews: (runId: string) => `/runs/${encodeURIComponent(runId)}/reviews`,
});

const SAFE_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  GFM_GLOBAL_MODEL_MODEL_NOT_INSTALLED: "SocialGraph-FM Global 模型尚未安装。",
  GFM_GLOBAL_MODEL_SCENARIO_UNAVAILABLE: "Russia 登记场景尚未发布。",
  GFM_GLOBAL_MODEL_SCENARIO_STALE: "Russia 场景与当前模型版本不一致。",
  GFM_GLOBAL_MODEL_MODEL_MISMATCH: "所选协议与当前 SocialGraph-FM Governance 模型版本不一致。",
  GFM_GLOBAL_MODEL_RUN_NOT_FOUND: "未找到对应的 SocialGraph-FM Global 运行。",
  GFM_GLOBAL_MODEL_NODE_NOT_FOUND: "未找到对应的风险账号证据。",
  GFM_GLOBAL_MODEL_SERVICE_UNAVAILABLE: "SocialGraph-FM Global 服务暂不可用。",
  GFM_GLOBAL_MODEL_RESPONSE_INVALID: "SocialGraph-FM Global 返回未通过客户端合同校验。",
};

function assertRunId(runId: string): void {
  if (!RUN_ID.test(runId)) throw new Error("GFM_GLOBAL_MODEL_RUN_ID_INVALID");
}

function assertNodeId(nodeId: string): void {
  if (!NODE_ID.test(nodeId)) throw new Error("GFM_GLOBAL_MODEL_NODE_ID_INVALID");
}

async function readPayload(response: Response): Promise<unknown> {
  const announcedLength = response.headers.get("Content-Length");
  if (announcedLength && Number(announcedLength) > MAX_RESPONSE_BYTES) {
    await response.body?.cancel().catch(() => undefined);
    throw new SocialGraphApiError("GFM_GLOBAL_MODEL_RESPONSE_TOO_LARGE", "SocialGraph-FM Global 响应超过浏览器安全上限。", 502);
  }
  let body = "";
  if (response.body) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let size = 0;
    try {
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        size += value.byteLength;
        if (size > MAX_RESPONSE_BYTES) {
          await reader.cancel().catch(() => undefined);
          throw new SocialGraphApiError("GFM_GLOBAL_MODEL_RESPONSE_TOO_LARGE", "SocialGraph-FM Global 响应超过浏览器安全上限。", 502);
        }
        body += decoder.decode(value, { stream: true });
      }
      body += decoder.decode();
    } finally {
      reader.releaseLock();
    }
  } else {
    body = await response.text();
  }
  let payload: unknown;
  try {
    payload = JSON.parse(body);
  } catch {
    throw new SocialGraphApiError("GFM_GLOBAL_MODEL_RESPONSE_INVALID", "SocialGraph-FM Global 返回了无法验证的响应。", 502);
  }
  if (!response.ok) {
    const candidate = payload && typeof payload === "object" && !Array.isArray(payload)
      ? payload as Record<string, unknown>
      : {};
    const detail = candidate.detail && typeof candidate.detail === "object" && !Array.isArray(candidate.detail)
      ? candidate.detail as Record<string, unknown>
      : {};
    const rawCode = candidate.code ?? detail.code;
    const code = typeof rawCode === "string" && /^[A-Z0-9_]{1,100}$/u.test(rawCode)
      ? rawCode
      : "GFM_GLOBAL_MODEL_RESPONSE_INVALID";
    throw new SocialGraphApiError(
      code,
      SAFE_ERROR_MESSAGES[code] ?? "SocialGraph-FM Global 请求未完成；服务器细节已隐藏。",
      response.status,
    );
  }
  return payload;
}

function responseInvalid(error: unknown): never {
  if (error instanceof SocialGraphApiError) throw error;
  if (error instanceof DOMException && error.name === "AbortError") throw error;
  if (error instanceof Error && (
    error.message === "GFM_GLOBAL_MODEL_RUN_ID_INVALID"
    || error.message === "GFM_GLOBAL_MODEL_NODE_ID_INVALID"
    || error.message === "GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID"
  )) throw error;
  throw new SocialGraphApiError("GFM_GLOBAL_MODEL_RESPONSE_INVALID", "SocialGraph-FM Global 返回了无法验证的响应。", 502);
}

export class GlobalModelClient implements GlobalModelClientLike {
  private readonly baseUrl: string;
  private readonly fetcher: Fetcher;
  private readonly results = new Map<string, GlobalModelRunResult>();

  constructor(
    baseUrl = socialGraphApiUrl(GLOBAL_MODEL_API_ROUTES.root),
    fetcher: Fetcher = globalThis.fetch,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/u, "");
    this.fetcher = fetcher.bind(globalThis);
  }

  private url(path: string): string {
    return `${this.baseUrl}${path}`;
  }

  async capabilities(signal?: AbortSignal): Promise<GlobalModelCapabilities> {
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.capabilities), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelCapabilities(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async health(signal?: AbortSignal): Promise<GlobalModelHealth> {
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.health), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelHealth(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async modelCard(signal?: AbortSignal): Promise<GlobalModelModelCard> {
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.modelCard), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelModelCard(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async scenario(signal?: AbortSignal): Promise<GlobalModelScenario> {
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.scenario), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelScenario(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async scenarioPreview(signal?: AbortSignal): Promise<GlobalModelScenarioPreview> {
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.scenarioPreview), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelScenarioPreview(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async createRun(
    request: GlobalModelRunRequest,
    identity: GlobalModelRunIdentity,
    signal?: AbortSignal,
  ): Promise<GlobalModelCreatedRun> {
    const validated = parseGlobalModelRunRequest(request);
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.runs), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(validated),
        signal,
      });
      const status = parseGlobalModelRunStatus(await readPayload(response));
      const publicRequestHash = sha256Canonical(validated);
      if (status.requestHash !== publicRequestHash) throw new Error("GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID");
      return deepFreeze({
        status,
        binding: {
          runId: status.runId,
          publicRequestHash,
          serverRequestHash: status.requestHash,
          taskId: validated.taskId,
          protocol: validated.protocol,
          datasetVersionId: validated.datasetVersionId,
          graphVersionHash: identity.graphVersionHash,
          modelVersionId: validated.modelVersionId,
          modelVersionHash: identity.modelVersionHash,
        },
      });
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async getRun(
    runId: string,
    binding: GlobalModelRunBinding,
    signal?: AbortSignal,
  ): Promise<GlobalModelRunStatus> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID");
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.run(runId)), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelRunStatus(await readPayload(response), binding);
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async getResult(
    runId: string,
    binding: GlobalModelRunBinding,
    signal?: AbortSignal,
  ): Promise<GlobalModelRunResult> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID");
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.result(runId)), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      const result = parseGlobalModelRunResult(await readPayload(response), binding);
      this.results.set(runId, result);
      return result;
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async nodeEvidence(
    runId: string,
    nodeId: string,
    binding: GlobalModelRunBinding,
    signal?: AbortSignal,
  ): Promise<GlobalModelNodeEvidence> {
    assertRunId(runId);
    assertNodeId(nodeId);
    if (runId !== binding.runId) throw new Error("GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID");
    const result = this.results.get(runId);
    if (!result) throw new Error("GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID");
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.evidence(runId, nodeId)), {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseGlobalModelNodeEvidence(
        await readPayload(response), binding, nodeId, result,
      );
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async submitReview(
    runId: string,
    request: GlobalModelReviewRequest,
    binding: GlobalModelRunBinding,
    signal?: AbortSignal,
  ): Promise<GlobalModelReviewRecord> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_GLOBAL_MODEL_RESPONSE_BINDING_INVALID");
    const validated = parseGlobalModelReviewRequest(request);
    try {
      const response = await this.fetcher(this.url(GLOBAL_MODEL_API_ROUTES.reviews(runId)), {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(validated),
        signal,
      });
      return parseGlobalModelReviewRecord(await readPayload(response), binding, validated);
    } catch (error) {
      return responseInvalid(error);
    }
  }
}
