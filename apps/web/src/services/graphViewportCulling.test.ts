import { describe, expect, it } from "vitest";
import {
  isSceneOutsideViewport,
  resolveViewportCullSnapshot,
} from "./graphViewportCulling";

describe("graph viewport culling", () => {
  it("rejects empty and previous-scene snapshots", () => {
    expect(resolveViewportCullSnapshot(undefined, "scene-b")).toBeUndefined();
    expect(resolveViewportCullSnapshot({ sceneKey: "scene-b", nodeIds: new Set() }, "scene-b"))
      .toBeUndefined();
    expect(resolveViewportCullSnapshot({ sceneKey: "scene-a", nodeIds: new Set(["a"]) }, "scene-b"))
      .toBeUndefined();
  });

  it("keeps a non-empty snapshot only for its current scene", () => {
    const ids = new Set(["a", "b"]);
    expect(resolveViewportCullSnapshot({ sceneKey: "scene-b", nodeIds: ids }, "scene-b"))
      .toBe(ids);
  });

  it("reports an empty post-fit viewport as offscreen", () => {
    expect(isSceneOutsideViewport(new Set(["a"]), new Set())).toBe(true);
    expect(isSceneOutsideViewport(new Set(["a"]), new Set(["a"]))).toBe(false);
    expect(isSceneOutsideViewport(new Set(), new Set())).toBe(false);
  });
});
