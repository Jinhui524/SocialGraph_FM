import { describe, expect, it } from "vitest";

import { coreRunContextKey } from "./coreRunContext";

describe("Core run context identity", () => {
  it("cannot collide when node IDs contain the old colon delimiter", () => {
    const shared = {
      graphVersionId: "graph:1",
      taskId: "core.collaboration_completion" as const,
      modelVersionId: "model:1",
      parameters: { topKSimilarCases: 5, candidateLimit: 50 },
    };

    expect(coreRunContextKey({
      ...shared,
      target: { kind: "node-pair", sourceId: "a:b", targetId: "c" },
    })).not.toBe(coreRunContextKey({
      ...shared,
      target: { kind: "node-pair", sourceId: "a", targetId: "b:c" },
    }));
  });
});
