import type {
  ResearchCapabilities,
  ResearchClientLike,
  ResearchCreatedRun,
  ResearchRunBinding,
  ResearchRunRequest,
  ResearchRunResult,
  ResearchRunStatus,
  ResearchScenarios,
  ResearchScenario,
  ResearchScenarioPreview,
  ResearchSimilarNodesRequest,
  ResearchSimilarNodesResult,
} from "../types/research";
import { SocialGraphApiError, socialGraphApiUrl } from "./apiClient";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";
import {
  parseResearchCapabilities,
  parseResearchRunRequest,
  parseResearchRunResult,
  parseResearchRunStatus,
  parseResearchScenarios,
  parseResearchScenarioPreview,
  parseResearchSimilarNodesRequest,
  parseResearchSimilarNodesResult,
} from "./researchContracts";

type Fetcher = typeof fetch;

const MAX_RESPONSE_BYTES = 8 * 1024 * 1024;
const RUN_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{0,99}$/u;
const SAFE_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  GFM_RESEARCH_MODEL_NOT_INSTALLED: "SocialGraph-FM Research 模型尚未安装。",
  GFM_RESEARCH_GRAPH_VERSION_NOT_FOUND: "当前图版本尚未登记到 SocialGraph-FM Research。",
  GFM_RESEARCH_GRAPH_REGISTRATION_PENDING: "图适配器登记仍在进行。",
  GFM_RESEARCH_GRAPH_INCOMPATIBLE: "当前图或任务与 SocialGraph-FM Research 合同不兼容。",
  GFM_RESEARCH_MODEL_MISMATCH: "模型版本与登记场景或运行请求不一致。",
  GFM_RESEARCH_SCENARIO_MISMATCH: "任务、图版本或模型与登记场景不一致。",
  GFM_RESEARCH_GRAPH_IDENTITY_CONFLICT: "图版本标识与其不可变图哈希冲突。",
  GFM_RESEARCH_GRAPH_ARTIFACT_MISSING: "图制品缺失，无法执行 SocialGraph-FM Research 推理。",
  GFM_RESEARCH_RESPONSE_INVALID: "SocialGraph-FM Research 返回未通过合同校验。",
  GFM_RESEARCH_SCENARIO_UNAVAILABLE: "登记示例场景尚未发布。",
  GFM_RESEARCH_SCENARIO_NOT_FOUND: "未找到登记示例场景。",
  GFM_RESEARCH_RESULT_NOT_READY: "SocialGraph-FM Research 结果尚未就绪。",
  GFM_RESEARCH_RUN_NOT_FOUND: "未找到对应的 SocialGraph-FM Research 运行。",
  GFM_RESEARCH_SERVICE_UNAVAILABLE: "SocialGraph-FM Research 服务暂不可用。",
};

function assertRunId(runId: string): void {
  if (!RUN_ID.test(runId)) throw new Error("GFM_RESEARCH_RUN_ID_INVALID");
}

async function readPayload(response: Response): Promise<unknown> {
  const announcedLength = response.headers.get("Content-Length");
  if (announcedLength && Number(announcedLength) > MAX_RESPONSE_BYTES) {
    await response.body?.cancel().catch(() => undefined);
    throw new SocialGraphApiError("GFM_RESEARCH_RESPONSE_TOO_LARGE", "SocialGraph-FM Research 响应超过浏览器安全上限。", 502);
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
          throw new SocialGraphApiError("GFM_RESEARCH_RESPONSE_TOO_LARGE", "SocialGraph-FM Research 响应超过浏览器安全上限。", 502);
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
    throw new SocialGraphApiError("GFM_RESEARCH_RESPONSE_INVALID", "SocialGraph-FM Research 返回了无法验证的响应。", 502);
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
      : "GFM_RESEARCH_SERVICE_UNAVAILABLE";
    throw new SocialGraphApiError(
      code,
      SAFE_ERROR_MESSAGES[code] ?? "SocialGraph-FM Research 请求未完成；服务器细节已隐藏。",
      response.status,
    );
  }
  return payload;
}

function responseInvalid(error: unknown): never {
  if (error instanceof SocialGraphApiError) throw error;
  if (error instanceof DOMException && error.name === "AbortError") throw error;
  if (error instanceof Error && error.message === "GFM_RESEARCH_RESPONSE_BINDING_INVALID") throw error;
  throw new SocialGraphApiError("GFM_RESEARCH_RESPONSE_INVALID", "SocialGraph-FM Research 返回了无法验证的响应。", 502);
}

export class ResearchClient implements ResearchClientLike {
  private readonly baseUrl: string;
  private readonly fetcher: Fetcher;

  constructor(
    baseUrl = socialGraphApiUrl("/api/v1/gfm/research"),
    fetcher: Fetcher = globalThis.fetch,
  ) {
    this.baseUrl = baseUrl.replace(/\/+$/u, "");
    this.fetcher = fetcher.bind(globalThis);
  }

  async capabilities(signal?: AbortSignal): Promise<ResearchCapabilities> {
    try {
      const response = await this.fetcher(`${this.baseUrl}/capabilities`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseResearchCapabilities(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async scenarios(signal?: AbortSignal): Promise<ResearchScenarios> {
    try {
      const response = await this.fetcher(`${this.baseUrl}/scenarios`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseResearchScenarios(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async scenarioPreview(
    scenarioId: ResearchScenario["scenarioId"],
    signal?: AbortSignal,
  ): Promise<ResearchScenarioPreview> {
    try {
      const response = await this.fetcher(
        `${this.baseUrl}/scenarios/${encodeURIComponent(scenarioId)}/graph-preview`,
        { method: "GET", headers: { Accept: "application/json" }, signal },
      );
      return parseResearchScenarioPreview(await readPayload(response));
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async createRun(request: ResearchRunRequest, signal?: AbortSignal): Promise<ResearchCreatedRun> {
    const validated = parseResearchRunRequest(request);
    try {
      const response = await this.fetcher(`${this.baseUrl}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(validated),
        signal,
      });
      const status = parseResearchRunStatus(await readPayload(response));
      assertRunId(status.runId);
      const binding = deepFreeze({
        runId: status.runId,
        publicRequestHash: sha256Canonical(validated),
        serverRequestHash: status.requestHash,
        graphVersionId: validated.graphVersionId,
        modelVersionId: validated.modelVersionId,
        taskId: validated.taskId,
      } satisfies ResearchRunBinding);
      return deepFreeze({ status, binding });
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async getRun(
    runId: string,
    binding: ResearchRunBinding,
    signal?: AbortSignal,
  ): Promise<ResearchRunStatus> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_RESEARCH_RESPONSE_BINDING_INVALID");
    try {
      const response = await this.fetcher(`${this.baseUrl}/runs/${encodeURIComponent(runId)}`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseResearchRunStatus(await readPayload(response), binding);
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async getResult(
    runId: string,
    binding: ResearchRunBinding,
    signal?: AbortSignal,
  ): Promise<ResearchRunResult> {
    assertRunId(runId);
    if (runId !== binding.runId) throw new Error("GFM_RESEARCH_RESPONSE_BINDING_INVALID");
    try {
      const response = await this.fetcher(`${this.baseUrl}/runs/${encodeURIComponent(runId)}/result`, {
        method: "GET",
        headers: { Accept: "application/json" },
        signal,
      });
      return parseResearchRunResult(await readPayload(response), binding);
    } catch (error) {
      return responseInvalid(error);
    }
  }

  async similarNodes(
    request: ResearchSimilarNodesRequest,
    signal?: AbortSignal,
  ): Promise<ResearchSimilarNodesResult> {
    const validated = parseResearchSimilarNodesRequest(request);
    try {
      const response = await this.fetcher(`${this.baseUrl}/similar-nodes`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(validated),
        signal,
      });
      const result = parseResearchSimilarNodesResult(await readPayload(response));
      if (
        result.graphVersionId !== validated.graphVersionId
        || result.nodeId !== validated.nodeId
        || result.modelVersionId !== validated.modelVersionId
      ) throw new Error("GFM_RESEARCH_RESPONSE_BINDING_INVALID");
      return result;
    } catch (error) {
      return responseInvalid(error);
    }
  }
}
