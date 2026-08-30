import { describe, expect, it } from "vitest";

import { graphTypeColour } from "./graphTypePalette";

describe("graph type palette", () => {
  it("keeps colours stable regardless of visible type vocabulary", () => {
    const before = graphTypeColour("organization");
    graphTypeColour("a-new-type-that-sorts-before-organization");
    const after = graphTypeColour("organization");

    expect(after).toBe(before);
    expect(graphTypeColour(" Organization ")).toBe(before);
    expect(graphTypeColour("person")).not.toBe(graphTypeColour("organization"));
    expect(graphTypeColour("institution")).toBe(graphTypeColour("organization"));
    expect(graphTypeColour("organization")).toBe("#7867d9");
    expect(graphTypeColour("person")).toBe("#4d86c6");
    expect(graphTypeColour("account")).toBe("#4d86c6");
    expect(graphTypeColour("governance-account")).toBe(graphTypeColour("account"));
    expect(graphTypeColour("账号")).toBe(graphTypeColour("account"));
    expect(graphTypeColour("project")).toBe("#48a69f");
    expect(graphTypeColour("community")).toBe("#d18a51");
    expect(graphTypeColour("未分类")).toBe("#8790a8");
  });
});
