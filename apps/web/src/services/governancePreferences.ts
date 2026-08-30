import type { GraphTheme } from "../types/graph";

export const GOVERNANCE_THEME_STORAGE_KEY = "socialgraph-fm.governance.theme.v1";

type ThemeStorage = Pick<Storage, "getItem" | "setItem">;

function browserStorage(): ThemeStorage | null {
  if (typeof window === "undefined") return null;
  return window.localStorage;
}

export function loadGovernanceTheme(storage: ThemeStorage | null = browserStorage()): GraphTheme {
  try {
    const value = storage?.getItem(GOVERNANCE_THEME_STORAGE_KEY);
    return value === "focus-dark" || value === "brand-light" ? value : "focus-dark";
  } catch {
    return "focus-dark";
  }
}

export function saveGovernanceTheme(theme: GraphTheme, storage: ThemeStorage | null = browserStorage()): void {
  try {
    storage?.setItem(GOVERNANCE_THEME_STORAGE_KEY, theme);
  } catch {
    // Storage may be unavailable in private or embedded browsing contexts.
  }
}
