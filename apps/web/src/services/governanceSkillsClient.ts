import { SocialGraphApiError, readSocialGraphApiJson, socialGraphApiUrl } from "./apiClient";
import { parseGovernanceOnlineRun } from "./governanceOnlineContracts";
import {
  ASSISTANT_PUBLIC_SKILLS,
  ASSISTANT_SKILLS_SCHEMA,
  ASSISTANT_SKILL_REQUEST_SCHEMA,
  ASSISTANT_SKILL_RESULT_SCHEMA,
  ASSISTANT_SKILL_POLICIES,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
  GOVERNANCE_SKILL_POLICIES,
  type AssistantSkillCatalog,
  type AssistantSkillDescriptor,
  type AssistantSkillName,
  type AssistantSkillResult,
  type GovernanceAssistantSkillTrace,
  type GovernanceConfirmationTicket,
  type GovernanceConfirmationAction,
  type GovernanceKnowledgeItem,
  type GovernanceKnowledgeSearchResponse,
  type GovernanceSimilarCase,
  type GovernanceSimilarCasesQuery,
  type GovernanceSimilarCasesResponse,
  type GovernanceSkillExecutionResponse,
  type GovernanceSkillConfirmationResponse,
  type GovernanceSkillCatalog,
  type GovernanceSkillDescriptor,
  type GovernanceSkillName,
  type GovernanceSkillParams,
  type GovernanceSkillsClientLike,
  type GovernanceSkillsContext,
} from "../types/governanceSkills";

type Fetcher = typeof fetch;
const HASH = /^[0-9a-f]{64}$/u;
const SKILL_SET = new Set<string>(GOVERNANCE_PUBLIC_SKILLS);
const ASSISTANT_SKILL_SET = new Set<string>(ASSISTANT_PUBLIC_SKILLS);

function fail(): never {
  throw new SocialGraphApiError("GFM_GOVERNANCE_SKILLS_RESPONSE_INVALID", "RAG 服务返回未通过浏览器合同校验。", 502);
}

function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return fail();
  return value as Record<string, unknown>;
}

function text(value: unknown, maximum = 2_000): string {
  if (typeof value !== "string" || value.length === 0 || value.length > maximum) return fail();
  return value;
}

function hash(value: unknown): string {
  const result = text(value, 64);
  if (!HASH.test(result)) return fail();
  return result;
}

function bool(value: unknown): boolean {
  if (typeof value !== "boolean") return fail();
  return value;
}

function integer(value: unknown, minimum: number, maximum: number): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < minimum || value > maximum) return fail();
  return value;
}

function list(value: unknown, maximum: number): readonly unknown[] {
  if (!Array.isArray(value) || value.length > maximum) return fail();
  return value;
}

function boundedObject(value: unknown, maximumBytes = 300_000): Readonly<Record<string, unknown>> {
  const item = object(value);
  let serialized: string;
  try {
    serialized = JSON.stringify(item);
  } catch {
    return fail();
  }
  if (serialized.length > maximumBytes) return fail();
  return Object.freeze({ ...item });
}

function finite(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return fail();
  return value;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`)
      .join(",")}}`;
  }
  return JSON.stringify(value) ?? "undefined";
}

function identifier(value: unknown, pattern: RegExp): string {
  const result = text(value, 200);
  if (!pattern.test(result)) return fail();
  return result;
}

function timestamp(value: unknown): string {
  const result = text(value, 100);
  if (!Number.isFinite(Date.parse(result))) return fail();
  return result;
}

function skill(value: unknown): GovernanceSkillName {
  const result = text(value, 100);
  if (!SKILL_SET.has(result)) return fail();
  return result as GovernanceSkillName;
}

function descriptor(value: unknown): GovernanceSkillDescriptor {
  const item = object(value);
  return Object.freeze({
    name: skill(item.name),
    readOnly: bool(item.readOnly),
    confirmationRequired: bool(item.confirmationRequired),
    description: text(item.description, 1_000),
    parameterSchema: Object.freeze({ ...object(item.parameterSchema) }),
  });
}

export function parseGovernanceSkillCatalog(value: unknown): GovernanceSkillCatalog {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_SKILLS_SCHEMA) return fail();
  const items = list(item.items, 8).map(descriptor);
  if (items.length !== GOVERNANCE_PUBLIC_SKILLS.length
    || items.some((entry, index) => {
      const policy = GOVERNANCE_SKILL_POLICIES[index];
      return entry.name !== policy.name
        || entry.readOnly !== policy.readOnly
        || entry.confirmationRequired !== policy.confirmationRequired;
    })) return fail();
  return Object.freeze({
    schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
    items: Object.freeze(items),
    catalogHash: hash(item.catalogHash),
  });
}

function knowledgeItem(value: unknown): GovernanceKnowledgeItem {
  const item = object(value);
  return Object.freeze({
    sourceLabel: text(item.sourceLabel, 200),
    sourceUri: text(item.sourceUri, 500),
    contentHash: hash(item.contentHash),
    chunkHash: hash(item.chunkHash),
    text: text(item.text, 2_000),
    rank: integer(item.rank, 1, 10),
  });
}

export function parseGovernanceKnowledgeSearch(value: unknown): GovernanceKnowledgeSearchResponse {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_SKILLS_SCHEMA) return fail();
  return Object.freeze({
    schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
    items: Object.freeze(list(item.items, 10).map(knowledgeItem)),
    indexHash: hash(item.indexHash),
    auditHash: hash(item.auditHash),
  });
}

function confirmation(value: unknown): GovernanceConfirmationTicket {
  const item = object(value);
  return Object.freeze({
    token: identifier(item.token, /^governance-confirm-[0-9a-f]{64}$/u),
    action: (() => {
      if (item.action !== "run_governance_analysis" && item.action !== "save_draft_report" && item.action !== "submit_review") return fail();
      return item.action as GovernanceConfirmationAction;
    })(),
    requestDigest: hash(item.requestDigest),
    expiresAt: timestamp(item.expiresAt),
  });
}

export function parseGovernanceSkillExecution(value: unknown): GovernanceSkillExecutionResponse {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_SKILLS_SCHEMA) return fail();
  const status = item.status === "completed" || item.status === "confirmation_required" ? item.status : fail();
  const confirmationValue = item.confirmation === null || item.confirmation === undefined
    ? null
    : confirmation(item.confirmation);
  if ((status === "confirmation_required") !== Boolean(confirmationValue)) return fail();
  return Object.freeze({
    schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
    executionId: identifier(item.executionId, /^governance-exec-[0-9a-f]{32}$/u),
    skill: skill(item.skill),
    status,
    result: boundedObject(item.result),
    confirmation: confirmationValue,
    provenance: boundedObject(item.provenance, 20_000),
    auditHash: hash(item.auditHash),
  });
}

export function parseGovernanceSkillConfirmation(value: unknown): GovernanceSkillConfirmationResponse {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_SKILLS_SCHEMA || item.status !== "completed") return fail();
  if (item.action !== "run_governance_analysis" && item.action !== "save_draft_report" && item.action !== "submit_review") return fail();
  if (item.action === "run_governance_analysis") {
    return Object.freeze({
      schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
      action: "run_governance_analysis" as const,
      status: "completed" as const,
      result: parseGovernanceOnlineRun(item.result),
      auditHash: hash(item.auditHash),
    });
  }
  if (item.action === "save_draft_report") return Object.freeze({
    schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
    action: "save_draft_report" as const,
    status: "completed" as const,
    result: boundedObject(item.result),
    auditHash: hash(item.auditHash),
  });
  return Object.freeze({
    schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
    action: "submit_review" as const,
    status: "completed" as const,
    result: boundedObject(item.result),
    auditHash: hash(item.auditHash),
  });
}

function assistantSkillTrace(value: unknown): GovernanceAssistantSkillTrace {
  const trace = object(value);
  const traceSkill = skill(trace.skill);
  if (traceSkill === "run_governance_analysis" || traceSkill === "draft_review_report") return fail();
  return Object.freeze({ skill: traceSkill, requestHash: hash(trace.requestHash), resultHash: hash(trace.resultHash) });
}

function assistantSkill(value: unknown): AssistantSkillName {
  const result = text(value, 100);
  if (!ASSISTANT_SKILL_SET.has(result)) return fail();
  return result as AssistantSkillName;
}

function assistantSkillDescriptor(value: unknown): AssistantSkillDescriptor {
  const item = object(value);
  if (item.readOnly !== true || item.confirmationRequired !== false) return fail();
  const governanceSkills = list(item.governanceSkills, 6).map((entry) => {
    const result = skill(entry);
    if (result === "run_governance_analysis" || result === "draft_review_report") return fail();
    return result;
  });
  return Object.freeze({
    name: assistantSkill(item.name),
    label: text(item.label, 100),
    description: text(item.description, 1_000),
    uiLocation: text(item.uiLocation, 200),
    readOnly: true as const,
    confirmationRequired: false as const,
    governanceSkills: Object.freeze(governanceSkills),
    parameterSchema: Object.freeze({ ...object(item.parameterSchema) }),
  });
}

export function parseAssistantSkillCatalog(value: unknown): AssistantSkillCatalog {
  const item = object(value);
  if (item.schemaVersion !== ASSISTANT_SKILLS_SCHEMA) return fail();
  const items = list(item.items, ASSISTANT_PUBLIC_SKILLS.length).map(assistantSkillDescriptor);
  if (items.length !== ASSISTANT_PUBLIC_SKILLS.length
    || items.some((entry, index) => {
      const policy = ASSISTANT_SKILL_POLICIES[index];
      return entry.name !== policy.name
        || entry.label !== policy.label
        || entry.description !== policy.description
        || entry.uiLocation !== policy.uiLocation
        || entry.readOnly !== policy.readOnly
        || entry.confirmationRequired !== policy.confirmationRequired
        || entry.governanceSkills.length !== policy.governanceSkills.length
        || entry.governanceSkills.some((skillName, skillIndex) => skillName !== policy.governanceSkills[skillIndex])
        || canonicalJson(entry.parameterSchema) !== canonicalJson(policy.parameterSchema);
    })) return fail();
  return Object.freeze({
    schemaVersion: ASSISTANT_SKILLS_SCHEMA,
    items: Object.freeze(items),
    catalogHash: hash(item.catalogHash),
  });
}

function assistantEvidenceRef(value: unknown) {
  const reference = object(value);
  if (reference.sourceKind !== "graph" && reference.sourceKind !== "skill"
    && reference.sourceKind !== "knowledge" && reference.sourceKind !== "case") return fail();
  return Object.freeze({
    label: text(reference.label, 200),
    sourceKind: reference.sourceKind,
    hash: hash(reference.hash),
  });
}

export function parseAssistantSkillResult(value: unknown): AssistantSkillResult {
  const item = object(value);
  if (item.schemaVersion !== ASSISTANT_SKILL_RESULT_SCHEMA) return fail();
  return Object.freeze({
    schemaVersion: ASSISTANT_SKILL_RESULT_SCHEMA,
    executionId: identifier(item.executionId, /^assistant-exec-[0-9a-f]{32}$/u),
    skill: assistantSkill(item.skill),
    answer: text(item.answer, 8_000),
    result: boundedObject(item.result, 100_000),
    skillCalls: Object.freeze(list(item.skillCalls, 5).map(assistantSkillTrace)),
    evidenceRefs: Object.freeze(list(item.evidenceRefs, 50).map(assistantEvidenceRef)),
    citedHashes: Object.freeze(list(item.citedHashes, 50).map(hash)),
    auditHash: hash(item.auditHash),
  });
}

function kindEntry(value: unknown): { readonly kind: "node" | "relation" | "group"; readonly targetIds: readonly string[] } {
  const item = object(value);
  if (item.kind !== "node" && item.kind !== "relation" && item.kind !== "group") return fail();
  const targetIds = list(item.targetIds, 100).map((target) => text(target, 200));
  if (!targetIds.length || [...targetIds].sort().join("\u0000") !== [...targetIds].join("\u0000")) return fail();
  return Object.freeze({ kind: item.kind, targetIds: Object.freeze(targetIds) });
}

function similarCase(value: unknown): GovernanceSimilarCase {
  const item = object(value);
  const components = object(item.components);
  const entries = list(item.kindEntries, 3).map(kindEntry);
  return Object.freeze({
    caseId: identifier(item.caseId, /^case-[0-9a-f]{32}$/u),
    score: finite(item.score),
    components: Object.freeze({ embedding: finite(components.embedding), structure: finite(components.structure), modality: finite(components.modality) }),
    graphVersionHash: hash(item.graphVersionHash),
    modelStateHash: hash(item.modelStateHash),
    kindKey: text(item.kindKey, 80),
    kindEntries: Object.freeze(entries),
    concludedAt: timestamp(item.concludedAt),
    recordHash: hash(item.recordHash),
  });
}

export function parseGovernanceSimilarCases(value: unknown): GovernanceSimilarCasesResponse {
  const item = object(value);
  if (item.schemaVersion !== GOVERNANCE_SKILLS_SCHEMA) return fail();
  const backfill = boundedObject(item.backfill, 2_000);
  if (Object.values(backfill).some((entry) => typeof entry !== "number" || !Number.isInteger(entry) || entry < 0)) return fail();
  return Object.freeze({
    schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
    query: boundedObject(item.query, 10_000),
    items: Object.freeze(list(item.items, 25).map(similarCase)),
    indexHash: hash(item.indexHash),
    backfill: Object.freeze({ ...backfill } as Record<string, number>),
    auditHash: hash(item.auditHash),
  });
}

export class GovernanceSkillsClient implements GovernanceSkillsClientLike {
  private readonly baseUrl: string;
  private readonly fetcher: Fetcher;

  constructor(baseUrl = socialGraphApiUrl("/api/v2/gfm/governance"), fetcher: Fetcher = globalThis.fetch) {
    this.baseUrl = baseUrl.replace(/\/+$/u, "");
    this.fetcher = fetcher.bind(globalThis);
  }

  private async json<T>(path: string, parse: (value: unknown) => T, init: RequestInit = {}): Promise<T> {
    const response = await this.fetcher(`${this.baseUrl}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...init.headers },
    });
    return parse(await readSocialGraphApiJson(response));
  }

  catalog(signal?: AbortSignal): Promise<GovernanceSkillCatalog> {
    return this.json("/skills", parseGovernanceSkillCatalog, { signal });
  }

  executeSkill<Name extends GovernanceSkillName>(
    context: GovernanceSkillsContext,
    skillName: Name,
    params: GovernanceSkillParams<Name>,
    signal?: AbortSignal,
  ): Promise<GovernanceSkillExecutionResponse> {
    return this.json("/skills/execute", parseGovernanceSkillExecution, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        skill: skillName,
        graph: context.graph,
        model: context.model,
        params,
      }),
      signal,
    });
  }

  confirmSkill(token: string, signal?: AbortSignal): Promise<GovernanceSkillConfirmationResponse> {
    if (!/^governance-confirm-[0-9a-f]{64}$/u.test(token)) {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_CONFIRMATION_TOKEN_INVALID", "确认票据无效。", 400));
    }
    return this.json("/skills/confirm", parseGovernanceSkillConfirmation, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ schemaVersion: GOVERNANCE_SKILLS_SCHEMA, token }),
      signal,
    });
  }

  assistantCatalog(signal?: AbortSignal): Promise<AssistantSkillCatalog> {
    return this.json("/assistant/skills", parseAssistantSkillCatalog, { signal });
  }

  executeAssistant(
    context: GovernanceSkillsContext,
    skillName: AssistantSkillName,
    message: string,
    signal?: AbortSignal,
  ): Promise<AssistantSkillResult> {
    const normalized = message.trim();
    if (!normalized || normalized.length > 2_000) {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_ASSISTANT_MESSAGE_INVALID", "问题为空或超过 2,000 字。", 400));
    }
    if (!ASSISTANT_SKILL_SET.has(skillName)) {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_ASSISTANT_SKILL_INVALID", "研判助手 Skill 不受支持。", 400));
    }
    return this.json("/assistant/execute", parseAssistantSkillResult, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schemaVersion: ASSISTANT_SKILL_REQUEST_SCHEMA,
        skill: skillName,
        graph: context.graph,
        model: context.model,
        message: normalized,
        context: {
          ...(context.runId ? { runId: context.runId } : {}),
          ...(context.caseId ? { caseId: context.caseId } : {}),
          ...(context.caseHash ? { caseHash: context.caseHash } : {}),
          ...(context.selectedTarget ? { selectedTarget: { targetType: context.selectedTarget.kind, targetId: context.selectedTarget.targetId } } : {}),
          topK: 100,
        },
      }),
      signal,
    });
  }

  searchKnowledge(
    context: GovernanceSkillsContext,
    query: string,
    limit = 5,
    signal?: AbortSignal,
  ): Promise<GovernanceKnowledgeSearchResponse> {
    const normalized = query.trim();
    if (!normalized || normalized.length > 500) {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_KNOWLEDGE_QUERY_INVALID", "检索问题为空或超过 500 字。", 400));
    }
    return this.json("/knowledge/search", parseGovernanceKnowledgeSearch, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        graph: context.graph,
        model: context.model,
        query: normalized,
        limit: Math.max(1, Math.min(10, Math.trunc(limit))),
      }),
      signal,
    });
  }

  searchSimilarCases(
    context: GovernanceSkillsContext,
    query: GovernanceSimilarCasesQuery,
    signal?: AbortSignal,
  ): Promise<GovernanceSimilarCasesResponse> {
    const explicitObjectQuery = Boolean(query.runId || query.kindEntries?.length);
    if (query.caseId && explicitObjectQuery) {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_SIMILAR_CASE_QUERY_INVALID", "相似案件查询不能同时指定研判单和运行对象。", 400));
    }
    const contextualObjectQuery = !query.caseId && !explicitObjectQuery && Boolean(context.runId && context.selectedTarget);
    const caseId = query.caseId ?? (explicitObjectQuery || contextualObjectQuery ? undefined : context.caseId);
    const runId = caseId ? undefined : query.runId ?? context.runId;
    const kindEntries = caseId ? undefined : query.kindEntries ?? (context.selectedTarget ? [{ kind: context.selectedTarget.kind, targetIds: [context.selectedTarget.targetId] }] : undefined);
    if ((!caseId && (!runId || !kindEntries?.length)) || (caseId && (runId || kindEntries))) {
      return Promise.reject(new SocialGraphApiError("GFM_GOVERNANCE_SIMILAR_CASE_QUERY_INVALID", "相似案件需要研判单或当前运行对象。", 400));
    }
    return this.json("/similar-cases/search", parseGovernanceSimilarCases, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        schemaVersion: GOVERNANCE_SKILLS_SCHEMA,
        graph: context.graph,
        model: context.model,
        ...(caseId ? { caseId } : { runId, kindEntries }),
        limit: Math.max(1, Math.min(25, Math.trunc(query.limit ?? 10))),
      }),
      signal,
    });
  }
}
