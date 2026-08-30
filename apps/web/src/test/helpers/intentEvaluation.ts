import type { AnalysisTask } from "../../types/graph";
import type { ExpectedIntentKind, IntentEvaluationCase } from "../fixtures/intentEvaluation";

export interface IntentEvaluationOutput {
  readonly kind: ExpectedIntentKind;
  readonly task?: AnalysisTask;
  readonly targets?: readonly string[];
  readonly timeRange?: {
    readonly start?: string;
    readonly end?: string;
  };
}

export interface IntentEvaluationFailure {
  readonly id: string;
  readonly field: "kind" | "task" | "targets" | "timeRange";
  readonly expected: unknown;
  readonly actual: unknown;
}

export interface IntentEvaluationReport {
  readonly total: number;
  readonly kindAccuracy: number;
  readonly taskAccuracy: number;
  readonly targetAccuracy: number;
  readonly timeRangeAccuracy: number;
  readonly failures: readonly IntentEvaluationFailure[];
}

type NormalizeForEvaluation = (
  input: string,
) => IntentEvaluationOutput | Promise<IntentEvaluationOutput>;

function sameStrings(actual: readonly string[] | undefined, expected: readonly string[]): boolean {
  return JSON.stringify(actual ?? []) === JSON.stringify(expected);
}

function sameTimeRange(
  actual: IntentEvaluationOutput["timeRange"],
  expected: NonNullable<IntentEvaluationCase["expectedTimeRange"]>,
): boolean {
  return actual?.start === expected.start && actual?.end === expected.end;
}

export async function scoreIntentEvaluation(
  cases: readonly IntentEvaluationCase[],
  normalize: NormalizeForEvaluation,
): Promise<IntentEvaluationReport> {
  const failures: IntentEvaluationFailure[] = [];
  let correctKinds = 0;
  let correctTasks = 0;
  let taskCases = 0;
  let correctTargets = 0;
  let targetCases = 0;
  let correctTimeRanges = 0;
  let timeRangeCases = 0;

  for (const testCase of cases) {
    const actual = await normalize(testCase.input);
    if (actual.kind === testCase.expectedKind) correctKinds += 1;
    else failures.push({ id: testCase.id, field: "kind", expected: testCase.expectedKind, actual: actual.kind });

    if (testCase.expectedTask) {
      taskCases += 1;
      if (actual.task === testCase.expectedTask) correctTasks += 1;
      else failures.push({ id: testCase.id, field: "task", expected: testCase.expectedTask, actual: actual.task });
    }

    if (testCase.expectedTargets) {
      targetCases += 1;
      if (sameStrings(actual.targets, testCase.expectedTargets)) correctTargets += 1;
      else failures.push({ id: testCase.id, field: "targets", expected: testCase.expectedTargets, actual: actual.targets });
    }

    if (testCase.expectedTimeRange) {
      timeRangeCases += 1;
      if (sameTimeRange(actual.timeRange, testCase.expectedTimeRange)) correctTimeRanges += 1;
      else failures.push({ id: testCase.id, field: "timeRange", expected: testCase.expectedTimeRange, actual: actual.timeRange });
    }
  }

  return Object.freeze({
    total: cases.length,
    kindAccuracy: cases.length === 0 ? 1 : correctKinds / cases.length,
    taskAccuracy: taskCases === 0 ? 1 : correctTasks / taskCases,
    targetAccuracy: targetCases === 0 ? 1 : correctTargets / targetCases,
    timeRangeAccuracy: timeRangeCases === 0 ? 1 : correctTimeRanges / timeRangeCases,
    failures: Object.freeze(failures),
  });
}

const GRAPH_CONTEXT_KEYS = new Set([
  "nodeCount",
  "edgeCount",
  "density",
  "connectedComponents",
  "nodeTypes",
  "edgeTypes",
  "hasWeight",
  "hasTimestamp",
  "timeRange",
]);
const TIME_RANGE_KEYS = new Set(["start", "end"]);

/** Returns paths that violate the graph-context data minimisation contract. */
export function findDisallowedGraphContextPaths(value: unknown): readonly string[] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return ["graphContext"];

  const violations: string[] = [];
  for (const [key, entry] of Object.entries(value)) {
    if (!GRAPH_CONTEXT_KEYS.has(key)) {
      violations.push(`graphContext.${key}`);
      continue;
    }
    if (key !== "timeRange") continue;
    if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
      violations.push("graphContext.timeRange");
      continue;
    }
    for (const nestedKey of Object.keys(entry)) {
      if (!TIME_RANGE_KEYS.has(nestedKey)) violations.push(`graphContext.timeRange.${nestedKey}`);
    }
  }
  return Object.freeze(violations.sort());
}

const FORBIDDEN_REQUEST_KEYS = new Set([
  "nodes",
  "edges",
  "sourcefile",
  "attributes",
  "preview",
  "canonicalgraph",
]);

/** Recursively catches accidental raw graph/file fields in an intent request. */
export function findForbiddenIntentRequestPaths(value: unknown, path = "request"): readonly string[] {
  if (!value || typeof value !== "object") return Object.freeze([]);
  if (Array.isArray(value)) {
    return Object.freeze(value.flatMap((entry, index) => findForbiddenIntentRequestPaths(entry, `${path}[${index}]`)));
  }

  const violations: string[] = [];
  for (const [key, entry] of Object.entries(value)) {
    if (FORBIDDEN_REQUEST_KEYS.has(key.toLocaleLowerCase())) violations.push(`${path}.${key}`);
    violations.push(...findForbiddenIntentRequestPaths(entry, `${path}.${key}`));
  }
  return Object.freeze(violations);
}
