import { GOVERNANCE_ONLINE_SCHEMA } from "../types/governanceOnline";
import type {
  GovernanceArtifact,
  GovernanceArtifactCompatibility,
  GovernanceCase,
  GovernanceCaseStatus,
  GovernanceDerivation,
  GovernanceFindingPage,
  GovernanceOnlineCapabilities,
  GovernanceOnlineClientLike,
  GovernanceOnlineEvidence,
  GovernanceOnlineHealth,
  GovernanceOnlinePreview,
  GovernanceProjectionRequest,
  GovernanceOnlineRun,
  GovernanceOnlineRunRequest,
  GovernanceOnlineResult,
  GovernanceRunComparison,
  GovernanceReviewDecision,
  GovernanceTargetKind,
  AdaptationComparison,
  AdaptationLabelSet,
  AdaptationLabelSetCreateRequest,
  AdaptationReviewPolicy,
  TargetTaskRegistration,
  RegisteredTargetLabelSetCreateRequest,
  RegisteredTargetLabelSet,
  TargetReviewPolicy,
  TargetAdaptationComparison,
  AdaptationHandoffCreateRequest,
  AdaptationGovernanceHandoff,
  AdaptationOverlayActivation,
  TargetReviewCollectionCreateRequest,
  TargetReviewCollection,
  TargetReviewPolicyFitRequest,
} from "../types/governanceOnline";
import { SocialGraphApiError, socialGraphApiUrl } from "./apiClient";
import {
  parseGovernanceArtifact,
  parseGovernanceArtifactCompatibility,
  parseGovernanceCase,
  parseGovernanceCases,
  parseGovernanceDerivationPage,
  parseCoreFindingPage,
  parseGovernanceOnlineCapabilities,
  parseGovernanceOnlineEvidence,
  parseGovernanceOnlineHealth,
  parseGovernanceOnlinePreview,
  parseGovernanceOnlineResult,
  parseGovernanceOnlineRun,
  parseGovernanceOnlineRunRequest,
  parseGovernanceOnlineRuns,
  parseGovernanceRunComparison,
  parseAdaptationComparison,
  parseTargetLabelSet,
  parseTargetLabelSetCreateRequest,
  parseAdaptationReviewPolicy,
  parseTargetTaskRegistration,
  parseRegisteredTargetLabelSetCreateRequest,
  parseRegisteredTargetLabelSet,
  parseTargetReviewPolicy,
  parseTargetAdaptationComparison,
  parseAdaptationGovernanceHandoff,
  parseAdaptationOverlayActivation,
  parseTargetReviewCollection,
} from "./governanceOnlineContracts";

type Fetcher = typeof fetch;
type Parser<T> = (value: unknown) => T;

const MAX_JSON_BYTES = 16 * 1024 * 1024;
const MAX_REPORT_BYTES = 32 * 1024 * 1024;
export const GOVERNANCE_ONLINE_MAX_UPLOAD_BYTES = 256 * 1024 * 1024;
const DERIVATION_PAGE_SIZE = 10_000;
const CASE_PAGE_SIZE = 100;
const ARTIFACT_ID = /^governance-artifact-[0-9a-f]{32}$/u;
const RUN_ID = /^governance-[0-9a-f]{32}$/u;
const CASE_ID = /^case-[0-9a-f]{32}$/u;
const TARGET_TASK_ID = /^target-task-[0-9a-f]{32}$/u;
const CONTROL_CHARACTER = /[\u0000-\u001f\u007f]/u;

export const GOVERNANCE_ONLINE_ROUTES = Object.freeze({
  root: "/api/v2/gfm/governance",
  health: "/health",
  capabilities: "/capabilities",
  artifacts: "/artifacts",
  artifactCompatibility: "/artifacts/compatibility",
  artifact: (artifactId: string) => `/artifacts/${encodeURIComponent(artifactId)}`,
  materialize: (artifactId: string) => `/artifacts/${encodeURIComponent(artifactId)}/materialize`,
  preview: (artifactId: string) => `/artifacts/${encodeURIComponent(artifactId)}/preview`,
  runs: "/runs",
  compareRuns: "/runs/compare",
  run: (runId: string) => `/runs/${encodeURIComponent(runId)}`,
  cancelRun: (runId: string) => `/runs/${encodeURIComponent(runId)}/cancel`,
  retryRun: (runId: string) => `/runs/${encodeURIComponent(runId)}/retry`,
  result: (runId: string) => `/runs/${encodeURIComponent(runId)}/result`,
  runPreview: (runId: string) => `/runs/${encodeURIComponent(runId)}/graph-preview`,
  findings: (runId: string) => `/runs/${encodeURIComponent(runId)}/nodes`,
  evidence: (runId: string, nodeId: string) => `/runs/${encodeURIComponent(runId)}/nodes/${encodeURIComponent(nodeId)}/evidence`,
  derivations: (runId: string, kind: GovernanceDerivation["kind"]) => {
    const suffix = kind === "group" ? "groups" : kind === "factual_relation" ? "relations" : "potential-links";
    return `/runs/${encodeURIComponent(runId)}/${suffix}`;
  },
  cases: "/cases",
  case: (caseId: string) => `/cases/${encodeURIComponent(caseId)}`,
  transitions: (caseId: string) => `/cases/${encodeURIComponent(caseId)}/transitions`,
  caseItems: (caseId: string) => `/cases/${encodeURIComponent(caseId)}/items`,
  reviews: (caseId: string) => `/cases/${encodeURIComponent(caseId)}/review-events`,
  report: (caseId: string) => `/cases/${encodeURIComponent(caseId)}/report`,
  adaptationLabelSets: "/adaptations/label-sets",
  adaptationPolicies: (labelSetHash: string) => `/adaptations/label-sets/${labelSetHash}/policies`,
  adaptationPolicy: (policyHash: string) => `/adaptations/policies/${policyHash}`,
  adaptationComparison: (runId: string, policyHash: string) => `/adaptations/runs/${encodeURIComponent(runId)}/policies/${policyHash}/comparison`,
  targetTasks: "/target-tasks",
  targetTask: (registrationId: string) => `/target-tasks/${encodeURIComponent(registrationId)}`,
  adaptationHandoffs: "/adaptations/handoffs",
  adaptationReviewCollections: "/adaptations/review-collections",
  adaptationHandoff: (handoffHash: string) => `/adaptations/handoffs/${handoffHash}`,
  activateAdaptationPolicy: (policyHash: string) => `/adaptations/policies/${policyHash}/activate`,
});

const SAFE_MESSAGES: Readonly<Record<string, string>> = {
  GFM_GOVERNANCE_MODEL_UNAVAILABLE: "在线风险模型尚未就绪。",
  GFM_GOVERNANCE_ARTIFACT_INCOMPATIBLE: "推理包未满足 768 维内容特征与五类关系合同。",
  GFM_GOVERNANCE_ARTIFACT_NOT_FOUND: "没有找到这份本地推理制品。",
  GFM_GOVERNANCE_RUN_NOT_FOUND: "没有找到这次在线运行。",
  GFM_GOVERNANCE_RUN_CONFLICT: "这份制品已有运行占用本地推理队列。",
  GFM_GOVERNANCE_RESPONSE_INVALID: "在线推理返回未通过浏览器合同校验。",
};

function exactId(value: string, pattern: RegExp): string {
  if (!pattern.test(value)) throw new Error("GFM_GOVERNANCE_PATH_ID_INVALID");
  return value;
}

function validatedArtifactId(value: string): string { return exactId(value, ARTIFACT_ID); }
function validatedRunId(value: string): string { return exactId(value, RUN_ID); }
function validatedCaseId(value: string): string { return exactId(value, CASE_ID); }
function validatedAdaptationHash(value: string): string { return exactId(value, /^[0-9a-f]{64}$/u); }
function opaqueId(value: string, maximum = 300): string {
  if (!value || value.length > maximum || value.trim() !== value || CONTROL_CHARACTER.test(value)) {
    throw new Error("GFM_GOVERNANCE_PATH_ID_INVALID");
  }
  return value;
}

function validateUpload(file: File): void {
  if (!file.name.toLocaleLowerCase("en-US").endsWith(".zip")) {
    throw new SocialGraphApiError("GFM_GOVERNANCE_UPLOAD_TYPE_INVALID", "请选择 SocialGraph-FM Governance 2.0 ZIP 推理包。", 400);
  }
  if (file.size <= 0 || file.size > GOVERNANCE_ONLINE_MAX_UPLOAD_BYTES) {
    throw new SocialGraphApiError("GFM_GOVERNANCE_UPLOAD_SIZE_INVALID", "推理包为空或超过 256 MiB 本机上限。", 400);
  }
}

function validateTargetTaskUpload(file: File): void {
  if (!file.name.toLocaleLowerCase("en-US").endsWith(".sgtask.zip")) {
    throw new SocialGraphApiError("GFM_GOVERNANCE_UPLOAD_TYPE_INVALID", "请选择一个目标任务包。", 400);
  }
  if (file.size <= 0 || file.size > GOVERNANCE_ONLINE_MAX_UPLOAD_BYTES) {
    throw new SocialGraphApiError("GFM_GOVERNANCE_UPLOAD_SIZE_INVALID", "目标任务包为空或超过本机上限。", 400);
  }
}

function projectionQuery(projection: GovernanceProjectionRequest | undefined): string {
  if (!projection) return "";
  const query = new URLSearchParams({ preset: projection.preset });
  if (projection.nodeBudget !== undefined) query.set("nodeBudget", String(projection.nodeBudget));
  if (projection.edgeBudget !== undefined) query.set("edgeBudget", String(projection.edgeBudget));
  if (projection.relation) query.set("relation", projection.relation);
  if (projection.groupBudget !== undefined) query.set("groupBudget", String(projection.groupBudget));
  for (const nodeId of projection.anchorNodeIds ?? []) query.append("anchorNodeId", opaqueId(nodeId, 128));
  return `?${query.toString()}`;
}

function invalidPage(): never {
  throw new SocialGraphApiError("GFM_GOVERNANCE_RESPONSE_INVALID", SAFE_MESSAGES.GFM_GOVERNANCE_RESPONSE_INVALID, 502);
}

async function readJson(response: Response): Promise<unknown> {
  const announced = Number(response.headers.get("Content-Length") ?? 0);
  if (announced > MAX_JSON_BYTES) {
    await response.body?.cancel().catch(() => undefined);
    throw new SocialGraphApiError("GFM_GOVERNANCE_RESPONSE_TOO_LARGE", "在线推理响应超过浏览器安全上限。", 502);
  }
  const body = await readBoundedText(response, MAX_JSON_BYTES);
  let payload: unknown;
  try {
    payload = JSON.parse(body);
  } catch {
    throw new SocialGraphApiError("GFM_GOVERNANCE_RESPONSE_INVALID", SAFE_MESSAGES.GFM_GOVERNANCE_RESPONSE_INVALID, 502);
  }
  if (response.ok) return payload;
  const candidate = payload && typeof payload === "object" && !Array.isArray(payload) ? payload as Record<string, unknown> : {};
  const detail = candidate.detail && typeof candidate.detail === "object" && !Array.isArray(candidate.detail)
    ? candidate.detail as Record<string, unknown> : {};
  const rawCode = candidate.code ?? detail.code;
  const code = typeof rawCode === "string" && /^[A-Z0-9_]{1,100}$/u.test(rawCode)
    ? rawCode : "GFM_GOVERNANCE_RESPONSE_INVALID";
  throw new SocialGraphApiError(code, SAFE_MESSAGES[code] ?? "在线治理请求未完成；服务器细节已隐藏。", response.status);
}

async function readBoundedText(response: Response, maximum: number): Promise<string> {
  if (!response.body) return response.text();
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let body = "";
  let size = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > maximum) {
        await reader.cancel().catch(() => undefined);
        throw new SocialGraphApiError("GFM_GOVERNANCE_RESPONSE_TOO_LARGE", "在线治理响应超过浏览器安全上限。", 502);
      }
      body += decoder.decode(value, { stream: true });
    }
    return body + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

function normalizeError(error: unknown): never {
  if (error instanceof SocialGraphApiError) throw error;
  if (error instanceof DOMException && error.name === "AbortError") throw error;
  if (error instanceof Error && error.message === "GFM_GOVERNANCE_PATH_ID_INVALID") throw error;
  throw new SocialGraphApiError("GFM_GOVERNANCE_RESPONSE_INVALID", SAFE_MESSAGES.GFM_GOVERNANCE_RESPONSE_INVALID, 502);
}

export class GovernanceOnlineClient implements GovernanceOnlineClientLike {
  private readonly baseUrl: string;
  private readonly fetcher: Fetcher;

  constructor(baseUrl = socialGraphApiUrl(GOVERNANCE_ONLINE_ROUTES.root), fetcher: Fetcher = globalThis.fetch) {
    this.baseUrl = baseUrl.replace(/\/+$/u, "");
    this.fetcher = fetcher.bind(globalThis);
  }

  private url(path: string): string { return `${this.baseUrl}${path}`; }

  private async json<T>(path: string, parser: Parser<T>, init: RequestInit = {}): Promise<T> {
    try {
      const response = await this.fetcher(this.url(path), {
        ...init,
        headers: { Accept: "application/json", ...init.headers },
      });
      return parser(await readJson(response));
    } catch (error) {
      return normalizeError(error);
    }
  }

  health(signal?: AbortSignal): Promise<GovernanceOnlineHealth> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.health, parseGovernanceOnlineHealth, { signal });
  }

  capabilities(signal?: AbortSignal): Promise<GovernanceOnlineCapabilities> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.capabilities, parseGovernanceOnlineCapabilities, { signal });
  }

  async russiaSample(signal?: AbortSignal): Promise<GovernanceArtifact> {
    const capabilities = await this.capabilities(signal);
    if (!capabilities.sampleArtifactId) {
      throw new SocialGraphApiError("GFM_GOVERNANCE_SAMPLE_UNAVAILABLE", "Russia 内置任务尚未登记。", 404);
    }
    return this.json(GOVERNANCE_ONLINE_ROUTES.materialize(validatedArtifactId(capabilities.sampleArtifactId)), parseGovernanceArtifact, { method: "POST", signal });
  }

  inspectArtifact(file: File, signal?: AbortSignal): Promise<GovernanceArtifactCompatibility> {
    try { validateUpload(file); } catch (error) { return Promise.reject(error); }
    const body = new FormData(); body.append("file", file, file.name);
    return this.json(GOVERNANCE_ONLINE_ROUTES.artifactCompatibility, parseGovernanceArtifactCompatibility, { method: "POST", body, signal });
  }

  uploadArtifact(file: File, cleanSelfLoops: boolean, signal?: AbortSignal): Promise<GovernanceArtifact> {
    try { validateUpload(file); } catch (error) { return Promise.reject(error); }
    const body = new FormData();
    body.append("file", file, file.name);
    body.append("cleanSelfLoops", String(cleanSelfLoops));
    return this.json(GOVERNANCE_ONLINE_ROUTES.artifacts, parseGovernanceArtifact, { method: "POST", body, signal });
  }

  registerTargetTask(file: File, signal?: AbortSignal): Promise<TargetTaskRegistration> {
    try { validateTargetTaskUpload(file); } catch (error) { return Promise.reject(error); }
    const body = new FormData(); body.append("file", file, file.name);
    return this.json(GOVERNANCE_ONLINE_ROUTES.targetTasks, parseTargetTaskRegistration, { method: "POST", body, signal });
  }

  targetTask(registrationId: string, signal?: AbortSignal): Promise<TargetTaskRegistration> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.targetTask(exactId(registrationId, TARGET_TASK_ID)), parseTargetTaskRegistration, { signal });
  }

  artifact(artifactId: string, signal?: AbortSignal): Promise<GovernanceArtifact> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.artifact(validatedArtifactId(artifactId)), parseGovernanceArtifact, { signal });
  }

  preview(artifactId: string, signal?: AbortSignal, projection?: GovernanceProjectionRequest): Promise<GovernanceOnlinePreview> {
    return this.json(`${GOVERNANCE_ONLINE_ROUTES.preview(validatedArtifactId(artifactId))}${projectionQuery(projection)}`, parseGovernanceOnlinePreview, { signal });
  }

  runPreview(runId: string, signal?: AbortSignal, projection?: GovernanceProjectionRequest): Promise<GovernanceOnlinePreview> {
    return this.json(`${GOVERNANCE_ONLINE_ROUTES.runPreview(validatedRunId(runId))}${projectionQuery(projection)}`, parseGovernanceOnlinePreview, { signal });
  }

  listRuns(signal?: AbortSignal): Promise<readonly GovernanceOnlineRun[]> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.runs, parseGovernanceOnlineRuns, { signal });
  }

  createRun(request: GovernanceOnlineRunRequest, signal?: AbortSignal): Promise<GovernanceOnlineRun> {
    const body = JSON.stringify(parseGovernanceOnlineRunRequest(request));
    return this.json(GOVERNANCE_ONLINE_ROUTES.runs, parseGovernanceOnlineRun, {
      method: "POST", headers: { "Content-Type": "application/json" }, body, signal,
    });
  }

  run(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineRun> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.run(validatedRunId(runId)), parseGovernanceOnlineRun, { signal });
  }

  cancelRun(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineRun> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.cancelRun(validatedRunId(runId)), parseGovernanceOnlineRun, { method: "POST", signal });
  }

  retryRun(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineRun> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.retryRun(validatedRunId(runId)), parseGovernanceOnlineRun, { method: "POST", signal });
  }

  compareRuns(leftRunId: string, rightRunId: string, limit = 200, signal?: AbortSignal): Promise<GovernanceRunComparison> {
    const query = new URLSearchParams({
      leftRunId: validatedRunId(leftRunId),
      rightRunId: validatedRunId(rightRunId),
      limit: String(Math.max(1, Math.min(10_000, Math.trunc(limit)))),
    });
    return this.json(`${GOVERNANCE_ONLINE_ROUTES.compareRuns}?${query}`, parseGovernanceRunComparison, { signal });
  }

  result(runId: string, signal?: AbortSignal): Promise<GovernanceOnlineResult> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.result(validatedRunId(runId)), parseGovernanceOnlineResult, { signal });
  }

  findings(runId: string, offset: number, limit: number, signal?: AbortSignal): Promise<GovernanceFindingPage> {
    const query = `?offset=${Math.max(0, Math.trunc(offset))}&limit=${Math.max(1, Math.min(10_000, Math.trunc(limit)))}`;
    return this.json(`${GOVERNANCE_ONLINE_ROUTES.findings(validatedRunId(runId))}${query}`, parseCoreFindingPage, { signal });
  }

  evidence(runId: string, nodeId: string, signal?: AbortSignal): Promise<GovernanceOnlineEvidence> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.evidence(validatedRunId(runId), opaqueId(nodeId, 128)), parseGovernanceOnlineEvidence, { signal });
  }

  async derivations(runId: string, kind: GovernanceDerivation["kind"], signal?: AbortSignal): Promise<readonly GovernanceDerivation[]> {
    const expectedRunId = validatedRunId(runId);
    const items: GovernanceDerivation[] = [];
    const seen = new Set<string>();
    let offset = 0;
    let total: number | null = null;
    do {
      const query = new URLSearchParams({ offset: String(offset), limit: String(DERIVATION_PAGE_SIZE) });
      const page = await this.json(`${GOVERNANCE_ONLINE_ROUTES.derivations(expectedRunId, kind)}?${query}`, parseGovernanceDerivationPage, { signal });
      if (page.runId !== expectedRunId || page.offset !== offset || page.limit !== DERIVATION_PAGE_SIZE || total !== null && page.total !== total) invalidPage();
      total ??= page.total;
      if (!page.items.length && offset < total) invalidPage();
      for (const item of page.items) {
        if (item.kind !== kind || seen.has(item.id)) invalidPage();
        seen.add(item.id); items.push(item);
      }
      offset += page.items.length;
    } while (offset < (total ?? 0));
    return Object.freeze(items);
  }

  async listCases(runId: string, signal?: AbortSignal): Promise<readonly GovernanceCase[]> {
    const expectedRunId = validatedRunId(runId);
    const matching: GovernanceCase[] = [];
    const seen = new Set<string>();
    let offset = 0;
    let total: number | null = null;
    do {
      const query = new URLSearchParams({ offset: String(offset), limit: String(CASE_PAGE_SIZE) });
      const page = await this.json(`${GOVERNANCE_ONLINE_ROUTES.cases}?${query}`, parseGovernanceCases, { signal });
      if (page.offset !== offset || page.limit !== CASE_PAGE_SIZE || total !== null && page.total !== total) invalidPage();
      total ??= page.total;
      if (!page.items.length && offset < total) invalidPage();
      for (const item of page.items) {
        if (seen.has(item.caseId)) invalidPage();
        seen.add(item.caseId);
        if (item.runId === expectedRunId) matching.push(item);
      }
      offset += page.items.length;
    } while (offset < (total ?? 0));
    return Object.freeze(matching);
  }

  createCase(runId: string, title: string, description: string, signal?: AbortSignal): Promise<GovernanceCase> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.cases, parseGovernanceCase, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, runId: validatedRunId(runId), title, description }), signal,
    });
  }

  updateCase(caseId: string, state: GovernanceCaseStatus, reason: string, signal?: AbortSignal): Promise<GovernanceCase> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.transitions(validatedCaseId(caseId)), parseGovernanceCase, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, state, reason }), signal,
    });
  }

  addCaseItem(caseId: string, targetType: GovernanceTargetKind, targetId: string, note: string, signal?: AbortSignal): Promise<GovernanceCase> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.caseItems(validatedCaseId(caseId)), parseGovernanceCase, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, targetType, targetId: opaqueId(targetId), note }), signal,
    });
  }

  review(caseId: string, targetType: GovernanceTargetKind, targetId: string, decision: GovernanceReviewDecision, reason: string, signal?: AbortSignal): Promise<GovernanceCase> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.reviews(validatedCaseId(caseId)), parseGovernanceCase, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, targetType, targetId: opaqueId(targetId), decision, reason, actor: "local-analyst" }), signal,
    });
  }

  case(caseId: string, signal?: AbortSignal): Promise<GovernanceCase> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.case(validatedCaseId(caseId)), parseGovernanceCase, { signal });
  }

  async report(caseId: string, format: "json" | "markdown" | "html", signal?: AbortSignal): Promise<Blob> {
    try {
      const response = await this.fetcher(`${this.url(GOVERNANCE_ONLINE_ROUTES.report(validatedCaseId(caseId)))}?format=${format}`, {
        headers: { Accept: format === "html" ? "text/html" : format === "markdown" ? "text/markdown" : "application/json" }, signal,
      });
      if (!response.ok) await readJson(response);
      const length = Number(response.headers.get("Content-Length") ?? 0);
      if (length > MAX_REPORT_BYTES) throw new SocialGraphApiError("GFM_GOVERNANCE_REPORT_TOO_LARGE", "研判报告超过 32MB 上限。", 502);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength > MAX_REPORT_BYTES) throw new SocialGraphApiError("GFM_GOVERNANCE_REPORT_TOO_LARGE", "研判报告超过 32MB 上限。", 502);
      return new Blob([buffer], { type: response.headers.get("Content-Type") ?? "application/octet-stream" });
    } catch (error) {
      return normalizeError(error);
    }
  }

  createAdaptationLabelSet(request: AdaptationLabelSetCreateRequest, signal?: AbortSignal): Promise<AdaptationLabelSet> {
    const body = JSON.stringify(parseTargetLabelSetCreateRequest(request));
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationLabelSets, parseTargetLabelSet, {
      method: "POST", headers: { "Content-Type": "application/json" }, body, signal,
    });
  }

  fitAdaptationPolicy(labelSetHash: string, signal?: AbortSignal): Promise<AdaptationReviewPolicy> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationPolicies(validatedAdaptationHash(labelSetHash)), parseAdaptationReviewPolicy, { method: "POST", signal });
  }

  adaptationPolicy(policyHash: string, signal?: AbortSignal): Promise<AdaptationReviewPolicy> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationPolicy(validatedAdaptationHash(policyHash)), parseAdaptationReviewPolicy, { signal });
  }

  adaptationComparison(runId: string, policyHash: string, offset = 0, limit = 100, signal?: AbortSignal): Promise<AdaptationComparison> {
    const boundedOffset = Math.max(0, Math.trunc(offset));
    const boundedLimit = Math.max(1, Math.min(500, Math.trunc(limit)));
    const path = GOVERNANCE_ONLINE_ROUTES.adaptationComparison(validatedRunId(runId), validatedAdaptationHash(policyHash));
    return this.json(`${path}?offset=${boundedOffset}&limit=${boundedLimit}`, parseAdaptationComparison, { signal });
  }

  createTargetLabelSet(request: RegisteredTargetLabelSetCreateRequest, signal?: AbortSignal): Promise<RegisteredTargetLabelSet> {
    const body = JSON.stringify(parseRegisteredTargetLabelSetCreateRequest(request));
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationLabelSets, parseRegisteredTargetLabelSet, {
      method: "POST", headers: { "Content-Type": "application/json" }, body, signal,
    });
  }

  fitTargetPolicy(labelSetHash: string, request: TargetReviewPolicyFitRequest, signal?: AbortSignal): Promise<TargetReviewPolicy> {
    if (request.schemaVersion !== "socialgraph-fm.governance-target-review-policy-fit-request/1.0") {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_REQUEST_INVALID", "少样本复核顺序请求无效。", 400));
    }
    const body = JSON.stringify({
      schemaVersion: request.schemaVersion,
      targetTaskRegistrationId: exactId(request.targetTaskRegistrationId, TARGET_TASK_ID),
      runId: validatedRunId(request.runId),
      resultHash: validatedAdaptationHash(request.resultHash),
    });
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationPolicies(validatedAdaptationHash(labelSetHash)), parseTargetReviewPolicy, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal,
    });
  }

  targetPolicy(policyHash: string, signal?: AbortSignal): Promise<TargetReviewPolicy> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationPolicy(validatedAdaptationHash(policyHash)), parseTargetReviewPolicy, { signal });
  }

  targetComparison(runId: string, policyHash: string, offset = 0, limit = 500, signal?: AbortSignal): Promise<TargetAdaptationComparison> {
    const boundedOffset = Math.max(0, Math.trunc(offset)); const boundedLimit = Math.max(1, Math.min(500, Math.trunc(limit)));
    const path = GOVERNANCE_ONLINE_ROUTES.adaptationComparison(validatedRunId(runId), validatedAdaptationHash(policyHash));
    return this.json(`${path}?offset=${boundedOffset}&limit=${boundedLimit}`, parseTargetAdaptationComparison, { signal });
  }

  createAdaptationHandoff(request: AdaptationHandoffCreateRequest, signal?: AbortSignal): Promise<AdaptationGovernanceHandoff> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationHandoffs, parseAdaptationGovernanceHandoff, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request), signal,
    });
  }

  adaptationHandoff(handoffHash: string, signal?: AbortSignal): Promise<AdaptationGovernanceHandoff> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationHandoff(validatedAdaptationHash(handoffHash)), parseAdaptationGovernanceHandoff, { signal });
  }

  activateTargetPolicy(policyHash: string, registrationId: string, signal?: AbortSignal): Promise<AdaptationOverlayActivation> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.activateAdaptationPolicy(validatedAdaptationHash(policyHash)), parseAdaptationOverlayActivation, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schemaVersion: "socialgraph-fm.governance-adaptation-overlay/1.0", targetTaskRegistrationId: exactId(registrationId, TARGET_TASK_ID) }), signal,
    });
  }

  createTargetReviewCollection(request: TargetReviewCollectionCreateRequest, signal?: AbortSignal): Promise<TargetReviewCollection> {
    return this.json(GOVERNANCE_ONLINE_ROUTES.adaptationReviewCollections, parseTargetReviewCollection, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request), signal,
    });
  }
}
