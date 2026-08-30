import { describe, expect, it, vi } from "vitest";

import {
  GOVERNANCE_THEME_STORAGE_KEY,
  loadGovernanceTheme,
  saveGovernanceTheme,
} from "./governancePreferences";

describe("governance theme preferences", () => {
  it("defaults invalid and missing values to the selected observatory theme", () => {
    expect(loadGovernanceTheme({ getItem: () => null, setItem: vi.fn() })).toBe("focus-dark");
    expect(loadGovernanceTheme({ getItem: () => "system", setItem: vi.fn() })).toBe("focus-dark");
  });

  it("loads and saves a valid governance theme with its own key", () => {
    const setItem = vi.fn();
    expect(loadGovernanceTheme({ getItem: () => "focus-dark", setItem })).toBe("focus-dark");
    saveGovernanceTheme("brand-light", { getItem: vi.fn(), setItem });
    expect(setItem).toHaveBeenCalledWith(GOVERNANCE_THEME_STORAGE_KEY, "brand-light");
  });

  it("falls back safely when storage access is blocked", () => {
    const blocked = {
      getItem: () => { throw new Error("blocked"); },
      setItem: () => { throw new Error("blocked"); },
    };
    expect(loadGovernanceTheme(blocked)).toBe("focus-dark");
    expect(() => saveGovernanceTheme("focus-dark", blocked)).not.toThrow();
  });
});
