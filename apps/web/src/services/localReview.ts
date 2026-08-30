import type { GraphRepository, SemanticEvent } from "../types/graph";
import { deepFreeze } from "./coreContracts";
import { sha256Canonical } from "./graphIdentity";
import { createSemanticEvent } from "./graphRepository";

const HASH = /^[0-9a-f]{64}$/u;
const DATE_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/u;

export type LocalReviewDecision = "confirmed" | "rejected";

export interface LocalReviewRecord {
  readonly schemaVersion: "socialgraph-fm.local-review/2.0";
  readonly findingHash: string;
  readonly runId: string;
  readonly resultHash: string;
  readonly graphVersionId: string;
  readonly sessionId?: string;
  readonly decision: LocalReviewDecision;
  readonly reviewedAt: string;
  readonly recordHash: string;
}

export interface CreateLocalReviewInput {
  readonly findingHash: string;
  readonly runId: string;
  readonly resultHash: string;
  readonly graphVersionId: string;
  readonly sessionId?: string;
  readonly decision: LocalReviewDecision;
  readonly reviewedAt?: string;
}

function validString(value: unknown, maximum = 1_000): value is string {
  return typeof value === "string" && value.length > 0 && value.length <= maximum;
}

function assertRecord(value: unknown): asserts value is Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("GFM_CORE_LOCAL_REVIEW_INVALID");
  }
}

export function parseLocalReviewRecord(value: unknown): LocalReviewRecord {
  try {
    assertRecord(value);
    const required = [
      "schemaVersion", "findingHash", "runId", "resultHash", "graphVersionId", "decision", "reviewedAt", "recordHash",
    ];
    const allowed = new Set([...required, "sessionId"]);
    if (required.some((key) => !(key in value)) || Object.keys(value).some((key) => !allowed.has(key))) {
      throw new Error("invalid keys");
    }
    if (
      value.schemaVersion !== "socialgraph-fm.local-review/2.0"
      || !validString(value.runId, 100)
      || !validString(value.graphVersionId, 200)
      || !validString(value.findingHash, 64)
      || !validString(value.resultHash, 64)
      || !HASH.test(value.findingHash)
      || !HASH.test(value.resultHash)
      || (value.sessionId !== undefined && !validString(value.sessionId, 200))
      || (value.decision !== "confirmed" && value.decision !== "rejected")
      || !validString(value.reviewedAt, 100)
      || !DATE_TIME.test(value.reviewedAt)
      || !Number.isFinite(Date.parse(value.reviewedAt))
      || !validString(value.recordHash, 64)
      || !HASH.test(value.recordHash)
    ) throw new Error("invalid record");
    const payload = Object.fromEntries(Object.entries(value).filter(([key]) => key !== "recordHash"));
    if (sha256Canonical(payload) !== value.recordHash) throw new Error("hash mismatch");
    return deepFreeze({ ...value } as unknown as LocalReviewRecord);
  } catch {
    throw new Error("GFM_CORE_LOCAL_REVIEW_INVALID");
  }
}

export function createLocalReviewRecord(
  input: CreateLocalReviewInput,
): LocalReviewRecord {
  const payload = {
    schemaVersion: "socialgraph-fm.local-review/2.0" as const,
    findingHash: input.findingHash,
    runId: input.runId,
    resultHash: input.resultHash,
    graphVersionId: input.graphVersionId,
    ...(input.sessionId ? { sessionId: input.sessionId } : {}),
    decision: input.decision,
    reviewedAt: input.reviewedAt ?? new Date().toISOString(),
  };
  return parseLocalReviewRecord({ ...payload, recordHash: sha256Canonical(payload) });
}

export async function appendLocalReview(
  repository: Pick<GraphRepository, "appendEvent">,
  record: LocalReviewRecord,
  options: { readonly eventId?: string } = {},
): Promise<SemanticEvent> {
  const validated = parseLocalReviewRecord(record);
  const event = createSemanticEvent("local_review_recorded", {
    ...(options.eventId ? { id: options.eventId } : {}),
    graphVersionId: validated.graphVersionId,
    ...(validated.sessionId ? { sessionId: validated.sessionId } : {}),
    createdAt: validated.reviewedAt,
    payload: { ...validated },
  });
  await repository.appendEvent(event);
  return event;
}
