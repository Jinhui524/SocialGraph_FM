import { describe, expect, it } from "vitest";
import {
  isGraphBenchmarkRenderer,
  requestedGraphBenchmarkRenderer,
} from "./graphBenchmarkRenderer";

describe("graph benchmark renderer query", () => {
  it.each(["canvas", "hybrid-webgl", "auto"] as const)(
    "accepts the %s renderer",
    (renderer) => {
      expect(requestedGraphBenchmarkRenderer(`?benchmark=graph&renderer=${renderer}`)).toBe(
        renderer,
      );
      expect(isGraphBenchmarkRenderer(renderer)).toBe(true);
    },
  );

  it("preserves the legacy Canvas default for missing or invalid values", () => {
    expect(requestedGraphBenchmarkRenderer("?benchmark=graph")).toBe("canvas");
    expect(requestedGraphBenchmarkRenderer("?renderer=webgpu")).toBe("canvas");
    expect(isGraphBenchmarkRenderer("webgpu")).toBe(false);
  });
});
