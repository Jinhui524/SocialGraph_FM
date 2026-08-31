import type {
  AnalysisIntentResult,
  AnalysisTask,
  GraphContextSummary,
  GraphVersion,
  IntentMeta,
  IntentNormalizationResult,
  IntentNormalizer,
  NormalizeIntentInput,
  ViewCommand,
} from "../types/graph";
import { SOCIALGRAPH_API_BASE_URL } from "./apiClient";

const ANALYSIS_TASKS = new Set<AnalysisTask>([
  "similar_structure",
  "link_prediction",
  "bridge_detection",
  "node_role",
  "community",
  "centrality",
  "overview",
]);
const OVERLAYS = new Set(["degree", "articulation", "components", "community"] as const);
type Fetcher = typeof fetch;

export type IntentServiceStatus =
  | { readonly state: "checking"; readonly label: string }
  | { readonly state: "llm"; readonly label: string; readonly model?: string }
  | { readonly state: "error"; readonly label: string };

export interface HttpIntentNormalizerOptions {
  readonly baseUrl?: string;
  readonly timeoutMs?: number;
  readonly fetcher?: Fetcher;
}

interface JsonObject {
  readonly [key: string]: unknown;
}

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function cleanCategory(value: string | undefined): string | undefined {
  const cleaned = value?.trim().slice(0, 64);
  return cleaned || undefined;
}

/**
 * Converts a complete browser-local graph into the only graph fields permitted
 * to cross the LLM boundary. Node IDs, labels, edges, file names and attributes
 * are deliberately unreachable from the returned object.
 */
export function buildGraphContextSummary(graph: GraphVersion): GraphContextSummary {
  const nodeTypes = unique(graph.nodes.map((node) => cleanCategory(node.type) ?? "")).slice(0, 50);
  const edgeTypes = unique(graph.edges.map((edge) => cleanCategory(edge.type) ?? "")).slice(0, 50);
  const timestamps = graph.edges
    .map((edge) => edge.timestamp?.trim().slice(0, 64))
    .filter((value): value is string => Boolean(value))
    .sort((left, right) => left.localeCompare(right));

  return Object.freeze({
    nodeCount: graph.summary.nodeCount,
    edgeCount: graph.summary.edgeCount,
    density: graph.summary.density,
    connectedComponents: graph.summary.connectedComponents,
    nodeTypes: Object.freeze(nodeTypes),
    edgeTypes: Object.freeze(edgeTypes),
    hasWeight: graph.edges.some((edge) => edge.weight !== undefined),
    hasTimestamp: timestamps.length > 0,
    ...(timestamps.length
      ? { timeRange: Object.freeze({ start: timestamps[0], end: timestamps[timestamps.length - 1] }) }
      : {}),
  });
}

function copyGraphContext(context: GraphContextSummary): GraphContextSummary {
  return {
    nodeCount: context.nodeCount,
    edgeCount: context.edgeCount,
    density: context.density,
    connectedComponents: context.connectedComponents,
    nodeTypes: [...context.nodeTypes].slice(0, 50),
    edgeTypes: [...context.edgeTypes].slice(0, 50),
    hasWeight: context.hasWeight,
    hasTimestamp: context.hasTimestamp,
    ...(context.timeRange
      ? { timeRange: { ...(context.timeRange.start ? { start: context.timeRange.start } : {}), ...(context.timeRange.end ? { end: context.timeRange.end } : {}) } }
      : {}),
  };
}

/** Explicit reconstruction prevents accidental serialization of GraphVersion. */
export function buildNormalizeIntentPayload(input: NormalizeIntentInput): NormalizeIntentInput {
  return {
    text: input.text,
    ...(input.graphContext ? { graphContext: copyGraphContext(input.graphContext) } : {}),
  };
}

function parseMeta(value: unknown): IntentMeta | null {
  if (!isObject(value)) return null;
  if (value.schemaVersion !== "1.0" && value.schemaVersion !== "1.1") return null;
  if (value.source !== "llm") return null;
  if (typeof value.requestId !== "string" || !value.requestId) return null;
  const warnings = Array.isArray(value.warnings)
    ? value.warnings.filter((warning): warning is string => typeof warning === "string").slice(0, 20)
    : [];
  return {
    schemaVersion: value.schemaVersion,
    source: value.source,
    requestId: value.requestId,
    ...(typeof value.model === "string" && value.model ? { model: value.model } : {}),
    warnings,
  };
}

function parseGroundedTerms(value: unknown, originalText: string): string[] {
  if (!Array.isArray(value)) return [];
  return unique(
    value
      .filter((entry): entry is string => typeof entry === "string")
      .map((entry) => entry.trim().slice(0, 80))
      .filter((entry) => originalText.includes(entry)),
  ).slice(0, 20);
}

function parseViewCommand(
  value: unknown,
  originalText: string,
  groundedTargets: readonly string[],
  responseTimeRange?: AnalysisIntentResult["timeRange"],
): ViewCommand | undefined {
  if (!isObject(value)) return undefined;
  const depth = value.depth === 1 || value.depth === 2 || value.depth === 3 ? value.depth : undefined;
  const overlay = typeof value.overlay === "string" && OVERLAYS.has(value.overlay as "degree" | "articulation" | "components" | "community")
    ? value.overlay as "degree" | "articulation" | "components" | "community"
    : undefined;
  const parsedFocusTerms = parseGroundedTerms(value.focusTerms, originalText);
  const focusTerms = parsedFocusTerms.length > 0 ? parsedFocusTerms : [...groundedTargets];
  const nodeTypeTerms = parseGroundedTerms(value.nodeTypeTerms, originalText);
  const edgeTypeTerms = parseGroundedTerms(value.edgeTypeTerms, originalText);
  const timeRange = parseTimeRange(value.timeRange) ?? responseTimeRange;
  const hasCommand = Boolean(
    depth || overlay || focusTerms.length || nodeTypeTerms.length || edgeTypeTerms.length || timeRange,
  );
  if (!hasCommand) return undefined;
  return Object.freeze({
    focusTerms: Object.freeze(focusTerms),
    ...(depth ? { depth } : {}),
    nodeTypeTerms: Object.freeze(nodeTypeTerms),
    edgeTypeTerms: Object.freeze(edgeTypeTerms),
    ...(timeRange ? { timeRange: Object.freeze({ ...timeRange }) } : {}),
    ...(overlay ? { overlay } : {}),
  });
}

function parseTimeRange(value: unknown): AnalysisIntentResult["timeRange"] {
  if (!isObject(value)) return undefined;
  const start = typeof value.start === "string" ? value.start.slice(0, 64) : undefined;
  const end = typeof value.end === "string" ? value.end.slice(0, 64) : undefined;
  return start || end ? { ...(start ? { start } : {}), ...(end ? { end } : {}) } : undefined;
}

function parseFilters(value: unknown): Readonly<Record<string, string | number | boolean>> {
  if (!isObject(value)) return {};
  return Object.freeze(
    Object.fromEntries(
      Object.entries(value)
        .filter(([key, entry]) => /^[a-zA-Z][a-zA-Z0-9_]{0,63}$/u.test(key) && ["string", "number", "boolean"].includes(typeof entry))
        .slice(0, 20),
    ) as Record<string, string | number | boolean>,
  );
}

function parseIntentResponse(value: unknown, originalText: string): IntentNormalizationResult | null {
  if (!isObject(value)) return null;
  const meta = parseMeta(value.meta);
  if (!meta) return null;

  if (value.kind === "chat") {
    if (typeof value.reply !== "string" || !value.reply.trim()) return null;
    return { kind: "chat", reply: value.reply.trim().slice(0, 2_000), meta };
  }

  if (value.kind !== "analysis_request") return null;
  if (typeof value.normalizedText !== "string" || !value.normalizedText.trim()) return null;
  if (typeof value.task !== "string" || !ANALYSIS_TASKS.has(value.task as AnalysisTask)) return null;
  if (typeof value.confidence !== "number" || !Number.isFinite(value.confidence)) return null;

  const warnings = [...meta.warnings];
  const confidence = Math.min(1, Math.max(0, value.confidence));
  const lowConfidence = confidence < 0.5;
  if (lowConfidence) warnings.push("意图置信度较低，已安全回退为网络概览。");
  const targets = Array.isArray(value.targets)
    ? unique(
        value.targets
          .filter((target): target is string => typeof target === "string")
          .map((target) => target.trim().slice(0, 80))
          .filter((target) => originalText.includes(target)),
      ).slice(0, 20)
    : [];
  const timeRange = parseTimeRange(value.timeRange);
  const task = lowConfidence ? "overview" : (value.task as AnalysisTask);
  // A server response is treated as an explicit bounded contract.  In
  // particular, legacy-only `mode` / `layoutPreset` fields must not fall
  // through to the local heuristic and regain control of layout or modes.
  const view = value.view === undefined
    ? undefined
    : parseViewCommand(value.view, originalText, targets, timeRange);

  return {
    kind: "analysis_request",
    normalizedText: value.normalizedText.trim().slice(0, 2_000),
    task,
    targets,
    confidence,
    ...(timeRange ? { timeRange } : {}),
    filters: parseFilters(value.filters),
    ...(view ? { view } : {}),
    meta: { ...meta, warnings },
  };
}

export class HttpIntentNormalizer implements IntentNormalizer {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly fetcher: Fetcher;

  constructor(options: HttpIntentNormalizerOptions = {}) {
    this.baseUrl = (options.baseUrl ?? SOCIALGRAPH_API_BASE_URL).replace(/\/$/u, "");
    this.timeoutMs = options.timeoutMs ?? 15_000;
    this.fetcher = options.fetcher ?? window.fetch.bind(window);
  }

  private async fetchJson(path: string, init?: RequestInit): Promise<unknown> {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await this.fetcher(`${this.baseUrl}${path}`, { ...init, signal: controller.signal });
      if (!response.ok) throw new Error(`HTTP_${response.status}`);
      return await response.json();
    } finally {
      window.clearTimeout(timeout);
    }
  }

  async checkStatus(): Promise<IntentServiceStatus> {
    try {
      const health = await this.fetchJson("/api/v1/health");
      if (!isObject(health) || health.status !== "ok") throw new Error("INVALID_HEALTH_RESPONSE");
      const capabilities = await this.fetchJson("/api/v1/capabilities");
      const normalization = isObject(capabilities) && isObject(capabilities.intentNormalization)
        ? capabilities.intentNormalization
        : null;
      if (!normalization) throw new Error("INVALID_CAPABILITIES_RESPONSE");
      const model = typeof normalization.model === "string" && normalization.model ? normalization.model : undefined;
      const connectionStatus = normalization.connectionStatus;
      if (normalization.configured === true
        && normalization.mode === "llm_required"
        && connectionStatus !== "error"
        && connectionStatus !== "not_configured") {
        const label = connectionStatus === "call_succeeded"
          ? "LLM 本次调用成功"
          : "LLM 已配置 · 等待调用验证";
        return { state: "llm", label, ...(model ? { model } : {}) };
      }
      return { state: "error", label: "LLM 未配置或验证失败" };
    } catch {
      return { state: "error", label: "LLM 服务不可用" };
    }
  }

  async normalizeIntent(input: NormalizeIntentInput): Promise<IntentNormalizationResult> {
    const value = await this.fetchJson("/api/v1/intents/normalize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildNormalizeIntentPayload(input)),
    });
    const parsed = parseIntentResponse(value, input.text);
    if (!parsed) throw new Error("INVALID_INTENT_RESPONSE");
    return parsed;
  }
}
