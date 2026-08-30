import { describe, expect, it } from "vitest";

import { shouldSubmitComposerKey } from "./composerKeyboard";

describe("chat composer keyboard policy", () => {
  it("does not submit Enter while a Chinese IME composition is active", () => {
    expect(shouldSubmitComposerKey({ key: "Enter", shiftKey: false, isComposing: true })).toBe(false);
  });

  it("submits plain Enter but preserves Shift+Enter and other keys", () => {
    expect(shouldSubmitComposerKey({ key: "Enter", shiftKey: false, isComposing: false })).toBe(true);
    expect(shouldSubmitComposerKey({ key: "Enter", shiftKey: true, isComposing: false })).toBe(false);
    expect(shouldSubmitComposerKey({ key: "a", shiftKey: false, isComposing: false })).toBe(false);
  });
});
