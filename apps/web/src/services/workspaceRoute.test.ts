import { describe, expect, it } from "vitest";

import {
  hashForWorkspaceRoute,
  isWorkspaceGraphPaneVisible,
  workspaceRouteFromHash,
} from "./workspaceRoute";

describe("workspace URL routing", () => {
  it("maps every public workspace to a stable hash", () => {
    expect(hashForWorkspaceRoute("research")).toBe("#/research");
    expect(hashForWorkspaceRoute("governance")).toBe("#/governance");
    expect(hashForWorkspaceRoute("adaptation")).toBe("#/adaptation");
    expect(hashForWorkspaceRoute("datasets")).toBe("#/datasets");
  });

  it("restores known hashes and fails closed to research for unknown routes", () => {
    expect(workspaceRouteFromHash("#/research")).toBe("research");
    expect(workspaceRouteFromHash("#/governance/")).toBe("governance");
    expect(workspaceRouteFromHash("#/adaptation?focus=russia")).toBe("adaptation");
    expect(workspaceRouteFromHash("#/datasets")).toBe("datasets");
    expect(workspaceRouteFromHash("#/future-workspace")).toBe("research");
    expect(workspaceRouteFromHash("")).toBe("research");
  });

  it("keeps the graph lifecycle visible on desktop while respecting mobile tabs", () => {
    expect(isWorkspaceGraphPaneVisible(1440, "chat")).toBe(true);
    expect(isWorkspaceGraphPaneVisible(1280, "governance")).toBe(true);
    expect(isWorkspaceGraphPaneVisible(390, "chat")).toBe(false);
    expect(isWorkspaceGraphPaneVisible(390, "evidence")).toBe(false);
    expect(isWorkspaceGraphPaneVisible(390, "graph")).toBe(true);
  });
});
