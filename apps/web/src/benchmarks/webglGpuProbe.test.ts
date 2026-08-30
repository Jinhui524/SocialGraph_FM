import { describe, expect, it } from "vitest";
import { requestedGpuProbeMode } from "./webglGpuProbe";

describe("requestedGpuProbeMode", () => {
  it("is opt-in and leaves release runs uninstrumented by default", () => {
    expect(requestedGpuProbeMode("?benchmark=graph")).toBe("off");
    expect(requestedGpuProbeMode("?benchmark=graph&probe=gpu")).toBe("gpu");
    expect(requestedGpuProbeMode("?benchmark=graph&probe=anything")).toBe("off");
  });
});

