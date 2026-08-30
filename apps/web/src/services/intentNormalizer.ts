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

const TASK_RULES: ReadonlyArray<{
  task: AnalysisTask;
  keywords: readonly string[];
  normalizedText: string;
}> = [
  {
    task: "similar_structure",
    keywords: ["相似结构", "相似案例", "结构检索", "相似图", "类比网络"],
    normalizedText: "检索与当前网络结构相似的图谱或案例",
  },
  {
    task: "link_prediction",
    keywords: ["链接预测", "关系预测", "潜在关系", "潜在合作", "关系推荐", "合作机会", "推荐关系"],
    normalizedText: "预测当前图谱中可能形成的潜在关系",
  },
  {
    task: "bridge_detection",
    keywords: ["割点", "桥接节点", "关键桥梁", "桥接者", "结构洞", "协作断层", "网络断层", "中介节点"],
    normalizedText: "识别移除后会改变连通结构的桥接节点",
  },
  {
    task: "node_role",
    keywords: ["节点角色", "角色识别", "成员角色", "成员定位", "节点分类", "核心团队识别"],
    normalizedText: "使用图表征识别节点角色与成员定位",
  },
  {
    task: "community",
    keywords: ["社区", "社群", "群落", "圈层", "分区", "团体结构", "社区健康"],
    normalizedText: "分析网络的连通分区与社区结构基线",
  },
  {
    task: "centrality",
    keywords: ["中心性", "影响力", "核心节点", "关键成员", "重要节点", "度数排名", "成员排名"],
    normalizedText: "计算节点度数中心性并生成影响力排名",
  },
  {
    task: "overview",
    keywords: ["概览", "摘要", "统计", "整体", "网络结构", "图谱情况"],
    normalizedText: "生成图谱概览和基础结构指标",
  },
];

const ANALYSIS_TASKS = new Set<AnalysisTask>(TASK_RULES.map((rule) => rule.task));
const OVERLAYS = new Set(["degree", "articulation", "components", "community"] as const);
const CHAT_ONLY_PATTERN = /^(?:你好|您好|嗨|hello|hi|在吗)[!！?？。\s]*$/iu;
const HELP_PATTERN = /^(?:你能做什么|你可以做什么|怎么使用|如何使用|使用帮助|功能介绍)[!！?？。\s]*$/u;
type Fetcher = typeof fetch;

export type IntentServiceStatus =
  | { readonly state: "checking"; readonly label: string }
  | { readonly state: "llm"; readonly label: string; readonly model?: string }
  | { readonly state: "fallback"; readonly label: string; readonly model?: string }
  | { readonly state: "offline"; readonly label: string };

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

function makeRequestId(prefix: string): string {
  const suffix = typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  return `${prefix}-${suffix}`;
}

function unique(values: readonly string[]): string[] {
  return [...new Set(values.map((value) => value.trim()).filter(Boolean))];
}

function extractTargets(text: string): string[] {
  const targets: string[] = [];
  const quoted = /[“”"「」『』]([^“”"「」『』]{1,40})[“”"「」『』]/g;
  for (const match of text.matchAll(quoted)) targets.push(match[1]);

  const mentions = /@([\p{L}\p{N}_-]{1,40})/gu;
  for (const match of text.matchAll(mentions)) targets.push(match[1]);

  const comparison = /(?:比较)\s*([\p{L}\p{N}_-]{1,16}(?:\s*[、,，]\s*[\p{L}\p{N}_-]{1,16})+)/u.exec(text);
  if (comparison) {
    targets.push(
      ...comparison[1]
        .split(/[、,，]/u)
        .map((target) => target.trim().replace(/(?:在|于|的)$/u, "")),
    );
  }

  const path = /(?:显示|查看|寻找|分析)?\s*([\p{L}\p{N}_-]{1,24}?)\s*(?:到|至)\s*([\p{L}\p{N}_-]{1,24}?)\s*(?:之间)?(?:的)?(?:最短)?路径/u.exec(text);
  if (path) targets.push(path[1], path[2]);

  const local = /(?:查看|显示|关注|分析)\s*([\p{L}\p{N}_-]{1,24}?)\s*(?:的)?(?:一|二|两|三|1|2|3)?\s*跳(?:邻居|邻域)/u.exec(text);
  if (local) targets.push(local[1]);

  const possessive = /(?:分析|查看|关注)\s*([\p{L}\p{N}_-]{1,24}?)\s*的(?:中心性|影响力|邻居|邻域|角色)/u.exec(text);
  if (possessive) targets.push(possessive[1]);

  return unique(targets).slice(0, 20);
}

function extractTimeRange(text: string): AnalysisIntentResult["timeRange"] {
  const range = /(19\d{2}|20\d{2})\s*(?:年)?\s*(?:-|—|–|~|～|至|到)\s*(19\d{2}|20\d{2})/u.exec(text);
  if (range) return { start: range[1], end: range[2] };
  const year = /(19\d{2}|20\d{2})\s*年?/u.exec(text);
  return year ? { start: year[1], end: year[1] } : undefined;
}

function fallbackMeta(warnings: readonly string[] = []): IntentMeta {
  return Object.freeze({
    schemaVersion: "1.1" as const,
    source: "deterministic_fallback" as const,
    requestId: makeRequestId("fallback"),
    warnings: Object.freeze([...warnings]),
  });
}

function extractDepth(text: string): 1 | 2 | 3 | undefined {
  const match = /([123一二两三])\s*跳/u.exec(text);
  if (!match) return undefined;
  if (match[1] === "1" || match[1] === "一") return 1;
  if (match[1] === "3" || match[1] === "三") return 3;
  return 2;
}

function extractFilterTerm(text: string, suffix: "关系" | "节点"): string[] {
  const pattern = suffix === "关系"
    ? /(?:只看|筛选|保留)\s*([^，。；;]{1,24}?)(?:关系|边)(?:[，。；;]|$)/u
    : /(?:只看|筛选|保留)\s*([^，。；;]{1,24}?)(?:节点|成员|机构|项目)(?:[，。；;]|$)/u;
  const match = pattern.exec(text);
  if (!match) return [];
  const value = match[1].replace(/^(?:类型为|类型是)/u, "").trim();
  return value ? [value] : [];
}

export function inferLocalViewCommand(
  text: string,
  task: AnalysisTask,
  targets: readonly string[],
  timeRange?: AnalysisIntentResult["timeRange"],
): ViewCommand | undefined {
  const pathRequested = /(?:路径|最短路)/u.test(text);
  const localRequested = /(?:邻居|邻域|局部图|子图|跳网络)/u.test(text);
  const nodeTypeTerms = extractFilterTerm(text, "节点");
  const edgeTypeTerms = extractFilterTerm(text, "关系");
  const layoutPreset = /(?:紧凑|聚拢)/u.test(text)
    ? "compact" as const
    : /(?:展开|分散|拉开)/u.test(text)
      ? "spread" as const
      : /(?:平衡|默认布局)/u.test(text)
        ? "balanced" as const
        : undefined;
  const overlay = /连通分量/u.test(text)
    ? "components" as const
    : task === "centrality"
      ? "degree" as const
      : task === "bridge_detection"
        ? "articulation" as const
        : task === "community"
          ? "community" as const
          : undefined;
  const mode = pathRequested ? "path" as const : localRequested ? "local" as const : undefined;
  const depth = extractDepth(text);
  const hasCommand = Boolean(
    mode || depth || nodeTypeTerms.length || edgeTypeTerms.length || layoutPreset || overlay || timeRange,
  );
  if (!hasCommand) return undefined;
  return Object.freeze({
    ...(mode ? { mode } : {}),
    focusTerms: Object.freeze([...targets]),
    ...(depth ? { depth } : {}),
    nodeTypeTerms: Object.freeze(nodeTypeTerms),
    edgeTypeTerms: Object.freeze(edgeTypeTerms),
    ...(timeRange ? { timeRange: Object.freeze({ ...timeRange }) } : {}),
    ...(layoutPreset ? { layoutPreset } : {}),
    ...(overlay ? { overlay } : {}),
  });
}

export function normalizeIntentLocally(
  input: NormalizeIntentInput,
  warnings: readonly string[] = [],
): IntentNormalizationResult {
  const text = input.text.trim();
  if (CHAT_ONLY_PATTERN.test(text) || HELP_PATTERN.test(text)) {
    return Object.freeze({
      kind: "chat" as const,
      reply: "你好，我可以读取本地 CSV / JSON 关系图，并根据你的研究问题执行网络概览、中心性、桥接节点和连通结构分析。预测类任务会等待真实 GFM API，不会生成模拟结论。",
      meta: fallbackMeta(warnings),
    });
  }

  const matchedRule = TASK_RULES.find((rule) =>
    rule.keywords.some((keyword) => text.toLocaleLowerCase().includes(keyword.toLocaleLowerCase())),
  );
  const timeRange = extractTimeRange(text);
  const task = matchedRule?.task ?? "overview";
  const targets = extractTargets(text);
  const view = inferLocalViewCommand(text, task, targets, timeRange);
  return Object.freeze({
    kind: "analysis_request" as const,
    normalizedText: matchedRule?.normalizedText ?? "生成图谱概览；原始描述未匹配到更具体的分析任务",
    task,
    targets: Object.freeze(targets),
    confidence: text.length === 0 ? 0.2 : matchedRule ? 0.92 : 0.55,
    ...(timeRange ? { timeRange: Object.freeze(timeRange) } : {}),
    filters: Object.freeze({
      ...(timeRange?.start ? { startYear: timeRange.start } : {}),
      ...(timeRange?.end ? { endYear: timeRange.end } : {}),
    }),
    ...(view ? { view } : {}),
    meta: fallbackMeta(warnings),
  });
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
  if (value.source !== "llm" && value.source !== "deterministic_fallback") return null;
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
  fallbackTargets: readonly string[],
  fallbackTimeRange?: AnalysisIntentResult["timeRange"],
): ViewCommand | undefined {
  if (!isObject(value)) return undefined;
  const depth = value.depth === 1 || value.depth === 2 || value.depth === 3 ? value.depth : undefined;
  const overlay = typeof value.overlay === "string" && OVERLAYS.has(value.overlay as "degree" | "articulation" | "components" | "community")
    ? value.overlay as "degree" | "articulation" | "components" | "community"
    : undefined;
  const parsedFocusTerms = parseGroundedTerms(value.focusTerms, originalText);
  const focusTerms = parsedFocusTerms.length > 0 ? parsedFocusTerms : [...fallbackTargets];
  const nodeTypeTerms = parseGroundedTerms(value.nodeTypeTerms, originalText);
  const edgeTypeTerms = parseGroundedTerms(value.edgeTypeTerms, originalText);
  const timeRange = parseTimeRange(value.timeRange) ?? fallbackTimeRange;
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
    ? inferLocalViewCommand(originalText, task, targets, timeRange)
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

export class MockIntentNormalizer implements IntentNormalizer {
  async normalizeIntent(input: NormalizeIntentInput): Promise<IntentNormalizationResult> {
    return normalizeIntentLocally(input);
  }
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
      if (normalization.configured === true && normalization.mode === "llm_with_fallback") {
        const connectionStatus = normalization.connectionStatus;
        const label = connectionStatus === "call_succeeded"
          ? "LLM 本次调用成功"
          : connectionStatus === "fallback"
            ? "LLM 已配置 · 最近调用规则降级"
            : "LLM 已配置 · 等待调用验证";
        return { state: "llm", label, ...(model ? { model } : {}) };
      }
      return { state: "fallback", label: "LLM 未配置 · 规则降级", ...(model ? { model } : {}) };
    } catch {
      return { state: "offline", label: "LLM 离线 · 规则降级" };
    }
  }

  async normalizeIntent(input: NormalizeIntentInput): Promise<IntentNormalizationResult> {
    try {
      const value = await this.fetchJson("/api/v1/intents/normalize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(buildNormalizeIntentPayload(input)),
      });
      const parsed = parseIntentResponse(value, input.text);
      if (!parsed) throw new Error("INVALID_INTENT_RESPONSE");
      return parsed;
    } catch (error) {
      const reason = error instanceof DOMException && error.name === "AbortError"
        ? "LLM 请求超时，已使用本地规则。"
        : "LLM 服务不可用或返回无效结构，已使用本地规则。";
      return normalizeIntentLocally(input, [reason]);
    }
  }
}
