import { describe, expect, it } from "vitest";

import { createValidatedCoreFixture } from "../test/fixtures/core";
import { createLocalGraphRepository, createSemanticEvent } from "./graphRepository";
import {
  appendLocalReview,
  createLocalReviewRecord,
  parseLocalReviewRecord,
} from "./localReview";

describe("local governance review memory", () => {
  it("appends a canonical hash-bound local review event and never changes the server finding", async () => {
    const fixture = createValidatedCoreFixture({ graphVersionId: "graph-v1" });
    const findingBefore = JSON.stringify(fixture.finding);
    const repository = createLocalGraphRepository({ forceMemory: true });
    const record = createLocalReviewRecord({
      findingHash: fixture.finding.findingHash,
      runId: fixture.binding.runId,
      resultHash: fixture.result.resultHash,
      graphVersionId: fixture.binding.graphVersionId,
      sessionId: "session-1",
      decision: "confirmed",
      reviewedAt: "2026-08-15T01:02:03.000Z",
    });

    const event = await appendLocalReview(repository, record, { eventId: "review-event-1" });

    expect(event).toMatchObject({
      id: "review-event-1",
      type: "local_review_recorded",
      graphVersionId: "graph-v1",
      sessionId: "session-1",
    });
    expect(event.payload).toMatchObject({
      findingHash: fixture.finding.findingHash,
      decision: "confirmed",
      recordHash: record.recordHash,
    });
    expect(parseLocalReviewRecord(event.payload)).toEqual(record);
    expect((await repository.listEvents("graph-v1"))).toEqual([event]);
    expect(JSON.stringify(fixture.finding)).toBe(findingBefore);
    expect(fixture.finding.reviewStatus).toBe("pending-human-review");
  });

  it("rejects an existing append-only event ID but preserves legacy saveEvent overwrite behavior", async () => {
    const fixture = createValidatedCoreFixture({ graphVersionId: "graph-v1" });
    const repository = createLocalGraphRepository({ forceMemory: true });
    const record = createLocalReviewRecord({
      findingHash: fixture.finding.findingHash,
      runId: fixture.binding.runId,
      resultHash: fixture.result.resultHash,
      graphVersionId: fixture.binding.graphVersionId,
      decision: "rejected",
      reviewedAt: "2026-08-15T02:00:00.000Z",
    });
    await appendLocalReview(repository, record, { eventId: "review-event-1" });

    await expect(appendLocalReview(repository, record, { eventId: "review-event-1" }))
      .rejects.toThrow("SEMANTIC_EVENT_ALREADY_EXISTS");
    expect(await repository.listEvents("graph-v1")).toHaveLength(1);

    await repository.saveEvent(createSemanticEvent("view_saved", {
      id: "legacy-overwrite",
      graphVersionId: "graph-v1",
      payload: { value: "first" },
    }));
    await repository.saveEvent(createSemanticEvent("view_saved", {
      id: "legacy-overwrite",
      graphVersionId: "graph-v1",
      payload: { value: "second" },
    }));
    expect((await repository.listEvents("graph-v1")).find((event) => event.id === "legacy-overwrite")?.payload)
      .toEqual({ value: "second" });
  });

  it("uses a new event and record for a later decision instead of overwriting history", async () => {
    const fixture = createValidatedCoreFixture({ graphVersionId: "graph-v1" });
    const repository = createLocalGraphRepository({ forceMemory: true });
    const base = {
      findingHash: fixture.finding.findingHash,
      runId: fixture.binding.runId,
      resultHash: fixture.result.resultHash,
      graphVersionId: fixture.binding.graphVersionId,
    } as const;
    const first = createLocalReviewRecord({
      ...base,
      decision: "confirmed",
      reviewedAt: "2026-08-15T02:00:00.000Z",
    });
    const second = createLocalReviewRecord({
      ...base,
      decision: "rejected",
      reviewedAt: "2026-08-15T03:00:00.000Z",
    });

    await appendLocalReview(repository, first, { eventId: "review-event-1" });
    await appendLocalReview(repository, second, { eventId: "review-event-2" });

    expect(await repository.listEvents("graph-v1")).toHaveLength(2);
    expect(first.recordHash).not.toBe(second.recordHash);
  });
});
