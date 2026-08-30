import { describe, expect, it } from "vitest";

import {
  DEFAULT_WORKSPACE_LAYOUT,
  loadWorkspaceLayout,
  normalizeWorkspaceLayout,
  resolveInternalPaneLayout,
  resolveWorkspaceGraphHeight,
  resolveWorkspaceLayout,
  resizeInternalEvidencePane,
  resizeWorkspacePane,
  saveWorkspaceLayout,
  WORKSPACE_LAYOUT_KEY,
} from "./workspaceLayout";

describe("workspace layout", () => {
  it("clamps restored pane widths and graph ratio", () => {
    expect(normalizeWorkspaceLayout({ leftWidth: 1, rightWidth: 9_000, graphHeightRatio: 2 })).toMatchObject({
      leftWidth: 220,
      rightWidth: 760,
      graphHeightRatio: 0.82,
    });
  });

  it("survives invalid persisted JSON", () => {
    expect(loadWorkspaceLayout({ getItem: () => "{" })).toEqual(DEFAULT_WORKSPACE_LAYOUT);
  });

  it("persists a normalized snapshot", () => {
    const entries = new Map<string, string>();
    saveWorkspaceLayout(
      { ...DEFAULT_WORKSPACE_LAYOUT, leftWidth: 310 },
      { setItem: (key, value) => { entries.set(key, value); } },
    );
    expect(JSON.parse(entries.get(WORKSPACE_LAYOUT_KEY) ?? "{}")).toMatchObject({
      leftWidth: 310,
      layoutVersion: 3,
    });
  });

  it("persists the graph summary state and migrates older snapshots responsively", () => {
    const entries = new Map<string, string>();
    saveWorkspaceLayout(
      { ...DEFAULT_WORKSPACE_LAYOUT, summaryCollapsed: true },
      { setItem: (key, value) => { entries.set(key, value); } },
    );
    expect(JSON.parse(entries.get(WORKSPACE_LAYOUT_KEY) ?? "{}")).toMatchObject({
      summaryCollapsed: true,
    });

    expect(loadWorkspaceLayout({
      getItem: () => JSON.stringify({ ...DEFAULT_WORKSPACE_LAYOUT, summaryCollapsed: undefined }),
    }, 1_280).summaryCollapsed).toBe(true);
    expect(loadWorkspaceLayout({
      getItem: () => JSON.stringify({ ...DEFAULT_WORKSPACE_LAYOUT, summaryCollapsed: false }),
    }, 1_280).summaryCollapsed).toBe(false);
  });

  it("migrates only untouched legacy graph defaults", () => {
    const migrated = loadWorkspaceLayout({
      getItem: () => JSON.stringify({
        ...DEFAULT_WORKSPACE_LAYOUT,
        rightWidth: 460,
        graphHeightRatio: 0.62,
      }),
    });
    expect(migrated).toMatchObject({ rightWidth: 520, graphHeightRatio: 0.68 });

    const customized = loadWorkspaceLayout({
      getItem: () => JSON.stringify({
        ...DEFAULT_WORKSPACE_LAYOUT,
        rightWidth: 500,
        graphHeightRatio: 0.64,
      }),
    });
    expect(customized).toMatchObject({ rightWidth: 500, graphHeightRatio: 0.64 });
  });

  it("resizes both side panes with the expected divider direction", () => {
    expect(resizeWorkspacePane(DEFAULT_WORKSPACE_LAYOUT, "left", 20).leftWidth).toBe(296);
    expect(resizeWorkspacePane(DEFAULT_WORKSPACE_LAYOUT, "right", 20).rightWidth).toBe(500);
  });

  it("uses the stored desktop widths as the rendered widths when the route has room", () => {
    const layout = resolveWorkspaceLayout({
      state: { ...DEFAULT_WORKSPACE_LAYOUT, leftWidth: 360, rightWidth: 760 },
      viewportWidth: 1_920,
      route: "research",
    });

    expect(layout).toMatchObject({
      breakpoint: "desktop",
      leftWidth: 360,
      rightWidth: 760,
      centralMinimum: 620,
      leftResizerVisible: true,
      rightResizerVisible: true,
    });
  });

  it("keeps the governance central minimum while reporting the actual constrained pane pixels", () => {
    const layout = resolveWorkspaceLayout({
      state: { ...DEFAULT_WORKSPACE_LAYOUT, leftWidth: 360, rightWidth: 460 },
      viewportWidth: 1_440,
      route: "governance",
    });

    expect(layout.leftWidth).toBe(356);
    expect(layout.rightWidth).toBe(340);
    expect(layout.centralWidth).toBe(720);
    expect(layout.rightBounds).toEqual({ minimum: 340, maximum: 340 });
  });

  it("uses the compact navigation rail without rendering its non-functional separator", () => {
    const layout = resolveWorkspaceLayout({
      state: DEFAULT_WORKSPACE_LAYOUT,
      viewportWidth: 1_280,
      route: "research",
    });

    expect(layout).toMatchObject({
      breakpoint: "compact",
      leftWidth: 72,
      leftResizerVisible: false,
      rightResizerVisible: true,
    });
  });

  it("removes workspace separators in the mobile tab layout", () => {
    const layout = resolveWorkspaceLayout({
      state: DEFAULT_WORKSPACE_LAYOUT,
      viewportWidth: 1_023,
      route: "adaptation",
    });

    expect(layout).toMatchObject({
      breakpoint: "mobile",
      leftResizerVisible: false,
      rightResizerVisible: false,
      graphResizerVisible: false,
    });
  });

  it("ignores the retired governance evidence width while preserving adaptation evidence bounds", () => {
    const layout = normalizeWorkspaceLayout({
      adaptationEvidenceWidth: 9_000,
      governanceEvidenceWidth: 480,
    });

    expect(layout).toMatchObject({
      adaptationEvidenceWidth: 520,
    });
    expect(layout).not.toHaveProperty("governanceEvidenceWidth");
  });

  it.each(["research", "governance", "adaptation"] as const)("keeps a genuine compact graph-width range for %s", (route) => {
    const layout = resolveWorkspaceLayout({
      state: DEFAULT_WORKSPACE_LAYOUT,
      viewportWidth: 1_024,
      route,
    });

    expect(layout.centralMinimum).toBe(560);
    expect(layout.rightBounds.minimum).toBeLessThan(layout.rightBounds.maximum);
    expect(layout.centralWidth).toBeGreaterThanOrEqual(560);
  });

  it("resizes from the constrained rendered graph width instead of a hidden stored preference", () => {
    const constrained = resolveWorkspaceLayout({
      state: { ...DEFAULT_WORKSPACE_LAYOUT, leftWidth: 360, rightWidth: 460 },
      viewportWidth: 1_440,
      route: "governance",
    });
    const next = resizeWorkspacePane(
      { ...DEFAULT_WORKSPACE_LAYOUT, leftWidth: 360, rightWidth: 460 },
      "right",
      20,
      {
        currentValue: constrained.rightWidth,
        bounds: constrained.rightBounds,
      },
    );

    expect(constrained.rightWidth).toBe(340);
    expect(next.rightWidth).toBe(340);
  });

  it("uses one graph-height pixel resolution for CSS, ARIA bounds, and deltas", () => {
    const graphHeight = resolveWorkspaceGraphHeight(DEFAULT_WORKSPACE_LAYOUT, 900);
    const next = resizeWorkspacePane(DEFAULT_WORKSPACE_LAYOUT, "graph", 40, {
      currentValue: graphHeight.height,
      bounds: graphHeight.bounds,
      viewportHeight: 900,
    });

    expect(graphHeight).toMatchObject({
      height: 594,
      bounds: { minimum: 320, maximum: 717 },
    });
    expect(next.graphHeightRatio).toBeCloseTo(634 / 874, 6);
  });

  it("resolves governance evidence pixels and bounds from the measured result container", () => {
    const layout = resolveInternalPaneLayout({
      containerWidth: 696,
      preferredEvidenceWidth: 520,
      listMinimum: 320,
      dividerWidth: 12,
      evidenceMinimum: 280,
      evidenceMaximum: 520,
    });

    expect(layout).toEqual({
      containerWidth: 696,
      listWidth: 320,
      evidenceWidth: 364,
      evidenceBounds: { minimum: 280, maximum: 364 },
      dividerWidth: 12,
      separatorVisible: true,
    });
  });

  it("starts an internal evidence delta at the resolved pixels and returns the actual clamped value", () => {
    const layout = resolveInternalPaneLayout({
      containerWidth: 606,
      preferredEvidenceWidth: 520,
      listMinimum: 220,
      dividerWidth: 12,
      evidenceMinimum: 280,
      evidenceMaximum: 520,
    });

    expect(layout.evidenceWidth).toBe(374);
    expect(resizeInternalEvidencePane(layout, 10)).toBe(364);
    expect(resizeInternalEvidencePane(layout, -40)).toBe(374);
  });

  it("suppresses an internal separator when the minimum list and evidence pair cannot fit", () => {
    expect(resolveInternalPaneLayout({
      containerWidth: 500,
      preferredEvidenceWidth: 360,
      listMinimum: 220,
      dividerWidth: 12,
      evidenceMinimum: 280,
      evidenceMaximum: 520,
    })).toMatchObject({
      containerWidth: 500,
      listWidth: 500,
      dividerWidth: 0,
      separatorVisible: false,
    });
  });
});
