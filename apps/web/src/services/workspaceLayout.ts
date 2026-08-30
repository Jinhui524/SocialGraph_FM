export interface WorkspaceLayoutState {
  readonly leftWidth: number;
  readonly rightWidth: number;
  readonly adaptationEvidenceWidth: number;
  readonly graphHeightRatio: number;
  readonly leftCollapsed: boolean;
  readonly rightCollapsed: boolean;
  readonly summaryCollapsed: boolean;
}

export const WORKSPACE_LAYOUT_KEY = "socialgraph-fm-workspace-layout-v1";
const WORKSPACE_LAYOUT_VERSION = 3;
const LEGACY_DEFAULT_RIGHT_WIDTH = 460;
const LEGACY_DEFAULT_GRAPH_HEIGHT_RATIO = 0.62;
export const DEFAULT_WORKSPACE_LAYOUT: WorkspaceLayoutState = Object.freeze({
  leftWidth: 276,
  rightWidth: 520,
  adaptationEvidenceWidth: 360,
  graphHeightRatio: 0.68,
  leftCollapsed: false,
  rightCollapsed: false,
  summaryCollapsed: false,
});

export const WORKSPACE_LIMITS = Object.freeze({
  leftMin: 220,
  leftMax: 360,
  rightMin: 340,
  rightMax: 760,
  internalEvidenceMin: 280,
  internalEvidenceMax: 520,
  graphRatioMin: 0.35,
  graphRatioMax: 0.82,
});

export const WORKSPACE_RESIZER_RAIL = 12;
export const COMPACT_NAVIGATION_WIDTH = 72;
export const COMPACT_CENTRAL_MINIMUM = 560;

export type WorkspaceLayoutRoute = "research" | "governance" | "adaptation";
export type WorkspaceLayoutBreakpoint = "desktop" | "compact" | "mobile";

export interface WorkspacePaneBounds {
  readonly minimum: number;
  readonly maximum: number;
}

/**
 * The only layout values that may be rendered by the workspace shell. Stored
 * state remains a user preference; this snapshot accounts for the viewport,
 * route minimum and any collapsed pane before it reaches CSS or ARIA.
 */
export interface ResolvedWorkspaceLayout {
  readonly breakpoint: WorkspaceLayoutBreakpoint;
  readonly leftWidth: number;
  readonly rightWidth: number;
  readonly centralWidth: number;
  readonly centralMinimum: number;
  readonly leftBounds: WorkspacePaneBounds;
  readonly rightBounds: WorkspacePaneBounds;
  readonly leftRailWidth: number;
  readonly rightRailWidth: number;
  readonly leftResizerVisible: boolean;
  readonly rightResizerVisible: boolean;
  readonly graphResizerVisible: boolean;
}

export interface ResolvedWorkspaceGraphHeight {
  readonly height: number;
  readonly bounds: WorkspacePaneBounds;
  readonly usableHeight: number;
}

export interface ResolvedInternalPaneLayout {
  readonly containerWidth: number;
  readonly listWidth: number;
  readonly evidenceWidth: number;
  readonly evidenceBounds: WorkspacePaneBounds;
  readonly dividerWidth: number;
  readonly separatorVisible: boolean;
}

export interface InternalPaneLayoutOptions {
  readonly containerWidth: number;
  readonly preferredEvidenceWidth: number;
  readonly listMinimum: number;
  readonly dividerWidth: number;
  readonly evidenceMinimum: number;
  readonly evidenceMaximum: number;
}

export const INTERNAL_PANE_LIMITS = Object.freeze({
  governanceListMinimum: 320,
  adaptationListMinimum: 220,
  dividerWidth: WORKSPACE_RESIZER_RAIL,
});

export interface WorkspacePaneResizeOptions {
  /** The actual rendered separator value, never an off-screen stored preference. */
  readonly currentValue?: number;
  readonly bounds?: WorkspacePaneBounds;
  readonly viewportHeight?: number;
}

const CENTRAL_MINIMUMS: Readonly<Record<WorkspaceLayoutRoute, number>> = Object.freeze({
  research: 620,
  governance: 720,
  adaptation: 680,
});

function finiteNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}

function bounds(minimum: number, maximum: number): WorkspacePaneBounds {
  return Object.freeze({ minimum, maximum: Math.max(minimum, maximum) });
}

/**
 * Resolves an internal list/evidence pair from the element that owns the grid.
 * Persisted widths are preferences only; every consumer renders and reports
 * this measured, dynamically clamped snapshot.
 */
export function resolveInternalPaneLayout({
  containerWidth,
  preferredEvidenceWidth,
  listMinimum,
  dividerWidth,
  evidenceMinimum,
  evidenceMaximum,
}: InternalPaneLayoutOptions): ResolvedInternalPaneLayout {
  const width = Math.max(0, Math.round(finiteNumber(containerWidth, 0)));
  const listMin = Math.max(0, Math.round(finiteNumber(listMinimum, 0)));
  const divider = Math.max(0, Math.round(finiteNumber(dividerWidth, 0)));
  const evidenceMin = Math.max(0, Math.round(finiteNumber(evidenceMinimum, 0)));
  const evidenceMax = Math.max(evidenceMin, Math.round(finiteNumber(evidenceMaximum, evidenceMin)));
  const preferred = finiteNumber(preferredEvidenceWidth, evidenceMin);
  const separatorVisible = width >= listMin + divider + evidenceMin;

  if (!separatorVisible) {
    const replacementMaximum = Math.min(evidenceMax, width);
    const replacementMinimum = Math.min(evidenceMin, replacementMaximum);
    return Object.freeze({
      containerWidth: width,
      listWidth: width,
      evidenceWidth: clamp(preferred, replacementMinimum, replacementMaximum),
      evidenceBounds: bounds(replacementMinimum, replacementMaximum),
      dividerWidth: 0,
      separatorVisible: false,
    });
  }

  const dynamicMaximum = Math.min(evidenceMax, width - listMin - divider);
  const evidenceWidth = Math.round(clamp(preferred, evidenceMin, dynamicMaximum));
  return Object.freeze({
    containerWidth: width,
    listWidth: width - divider - evidenceWidth,
    evidenceWidth,
    evidenceBounds: bounds(evidenceMin, dynamicMaximum),
    dividerWidth: divider,
    separatorVisible: true,
  });
}

/** Returns the next persisted evidence width from the resolved on-screen value. */
export function resizeInternalEvidencePane(
  layout: ResolvedInternalPaneLayout,
  delta: number,
): number {
  if (!layout.separatorVisible || !Number.isFinite(delta)) return layout.evidenceWidth;
  return Math.round(clamp(
    layout.evidenceWidth - delta,
    layout.evidenceBounds.minimum,
    layout.evidenceBounds.maximum,
  ));
}

/** Resolves the graph/summary split into the same actual pixels used by CSS and ARIA. */
export function resolveWorkspaceGraphHeight(
  state: WorkspaceLayoutState,
  viewportHeight: number,
): ResolvedWorkspaceGraphHeight {
  const normalized = normalizeWorkspaceLayout(state);
  const usableHeight = Math.max(320, Math.round(viewportHeight - 26));
  const graphBounds = bounds(
    320,
    Math.max(320, Math.round(usableHeight * WORKSPACE_LIMITS.graphRatioMax)),
  );
  return Object.freeze({
    height: Math.round(clamp(usableHeight * normalized.graphHeightRatio, graphBounds.minimum, graphBounds.maximum)),
    bounds: graphBounds,
    usableHeight,
  });
}

/** Resolves preference state into actual on-screen pixels for one viewport. */
export function resolveWorkspaceLayout({
  state,
  viewportWidth,
  route,
}: {
  readonly state: WorkspaceLayoutState;
  readonly viewportWidth: number;
  readonly route: WorkspaceLayoutRoute;
}): ResolvedWorkspaceLayout {
  const normalized = normalizeWorkspaceLayout(state);
  const width = Number.isFinite(viewportWidth) ? Math.max(0, viewportWidth) : 0;
  const routeCentralMinimum = CENTRAL_MINIMUMS[route];
  if (width <= 1_023) {
    return Object.freeze({
      breakpoint: "mobile",
      leftWidth: 0,
      rightWidth: 0,
      centralWidth: width,
      centralMinimum: routeCentralMinimum,
      leftBounds: bounds(WORKSPACE_LIMITS.leftMin, WORKSPACE_LIMITS.leftMax),
      rightBounds: bounds(WORKSPACE_LIMITS.rightMin, WORKSPACE_LIMITS.rightMax),
      leftRailWidth: 0,
      rightRailWidth: 0,
      leftResizerVisible: false,
      rightResizerVisible: false,
      graphResizerVisible: false,
    });
  }

  const compact = width < 1_440;
  const centralMinimum = compact ? COMPACT_CENTRAL_MINIMUM : routeCentralMinimum;
  const leftResizerVisible = !compact && !normalized.leftCollapsed;
  const rightResizerVisible = !normalized.rightCollapsed;
  const leftRailWidth = leftResizerVisible ? WORKSPACE_RESIZER_RAIL : 0;
  const rightRailWidth = rightResizerVisible ? WORKSPACE_RESIZER_RAIL : 0;
  const leftMaximum = Math.max(
    WORKSPACE_LIMITS.leftMin,
    Math.min(
      WORKSPACE_LIMITS.leftMax,
      width - centralMinimum - leftRailWidth - rightRailWidth - WORKSPACE_LIMITS.rightMin,
    ),
  );
  const leftWidth = compact
    ? COMPACT_NAVIGATION_WIDTH
    : normalized.leftCollapsed
      ? COMPACT_NAVIGATION_WIDTH
      : clamp(
        normalized.leftWidth,
        WORKSPACE_LIMITS.leftMin,
        leftMaximum,
      );
  const rightAvailable = width - centralMinimum - leftRailWidth - rightRailWidth - leftWidth;
  const rightMinimum = normalized.rightCollapsed
    ? 52
    : Math.min(WORKSPACE_LIMITS.rightMin, Math.max(0, rightAvailable));
  const rightMaximum = normalized.rightCollapsed
    ? 52
    : Math.min(WORKSPACE_LIMITS.rightMax, Math.max(rightMinimum, rightAvailable));
  const rightWidth = normalized.rightCollapsed
    ? 52
    : clamp(normalized.rightWidth, rightMinimum, rightMaximum);
  const breakpoint: WorkspaceLayoutBreakpoint = compact ? "compact" : "desktop";

  return Object.freeze({
    breakpoint,
    leftWidth,
    rightWidth,
    centralWidth: Math.max(0, width - leftWidth - rightWidth - leftRailWidth - rightRailWidth),
    centralMinimum,
    leftBounds: compact || normalized.leftCollapsed
      ? bounds(COMPACT_NAVIGATION_WIDTH, COMPACT_NAVIGATION_WIDTH)
      : bounds(WORKSPACE_LIMITS.leftMin, leftMaximum),
    rightBounds: bounds(rightMinimum, rightMaximum),
    leftRailWidth,
    rightRailWidth,
    leftResizerVisible,
    rightResizerVisible,
    graphResizerVisible: !normalized.rightCollapsed,
  });
}

export function normalizeWorkspaceLayout(value: unknown): WorkspaceLayoutState {
  const record = value && typeof value === "object" ? value as Partial<WorkspaceLayoutState> : {};
  return Object.freeze({
    leftWidth: clamp(
      finiteNumber(record.leftWidth, DEFAULT_WORKSPACE_LAYOUT.leftWidth),
      WORKSPACE_LIMITS.leftMin,
      WORKSPACE_LIMITS.leftMax,
    ),
    rightWidth: clamp(
      finiteNumber(record.rightWidth, DEFAULT_WORKSPACE_LAYOUT.rightWidth),
      WORKSPACE_LIMITS.rightMin,
      WORKSPACE_LIMITS.rightMax,
    ),
    adaptationEvidenceWidth: clamp(
      finiteNumber(record.adaptationEvidenceWidth, DEFAULT_WORKSPACE_LAYOUT.adaptationEvidenceWidth),
      WORKSPACE_LIMITS.internalEvidenceMin,
      WORKSPACE_LIMITS.internalEvidenceMax,
    ),
    graphHeightRatio: clamp(
      finiteNumber(record.graphHeightRatio, DEFAULT_WORKSPACE_LAYOUT.graphHeightRatio),
      WORKSPACE_LIMITS.graphRatioMin,
      WORKSPACE_LIMITS.graphRatioMax,
    ),
    leftCollapsed: record.leftCollapsed === true,
    rightCollapsed: record.rightCollapsed === true,
    summaryCollapsed: record.summaryCollapsed === true,
  });
}

export function loadWorkspaceLayout(
  storage?: Pick<Storage, "getItem">,
  viewportWidth = Number.POSITIVE_INFINITY,
): WorkspaceLayoutState {
  const target = storage ?? (typeof window !== "undefined" ? window.localStorage : undefined);
  const summaryCollapsed = Number.isFinite(viewportWidth) && viewportWidth < 1_360;
  if (!target) {
    return normalizeWorkspaceLayout({
      ...DEFAULT_WORKSPACE_LAYOUT,
      summaryCollapsed,
    });
  }
  try {
    const stored = target.getItem(WORKSPACE_LAYOUT_KEY);
    if (!stored) {
      return normalizeWorkspaceLayout({
        ...DEFAULT_WORKSPACE_LAYOUT,
        summaryCollapsed,
      });
    }
    const parsed = JSON.parse(stored) as Partial<WorkspaceLayoutState> & {
      readonly layoutVersion?: number;
    };
    const legacyDefaults = typeof parsed.layoutVersion !== "number"
      || parsed.layoutVersion < WORKSPACE_LAYOUT_VERSION;
    // Layout v1 snapshots did not persist the summary state. Keep their pane
    // sizes while applying responsive defaults and the enlarged graph defaults
    // exactly once. Values that differ from the old defaults remain user-owned.
    return normalizeWorkspaceLayout({
      ...parsed,
      rightWidth:
        legacyDefaults && parsed.rightWidth === LEGACY_DEFAULT_RIGHT_WIDTH
          ? DEFAULT_WORKSPACE_LAYOUT.rightWidth
          : parsed.rightWidth,
      graphHeightRatio:
        legacyDefaults && parsed.graphHeightRatio === LEGACY_DEFAULT_GRAPH_HEIGHT_RATIO
          ? DEFAULT_WORKSPACE_LAYOUT.graphHeightRatio
          : parsed.graphHeightRatio,
      summaryCollapsed:
        typeof parsed.summaryCollapsed === "boolean"
          ? parsed.summaryCollapsed
          : summaryCollapsed,
    });
  } catch {
    return normalizeWorkspaceLayout({
      ...DEFAULT_WORKSPACE_LAYOUT,
      summaryCollapsed,
    });
  }
}

export function saveWorkspaceLayout(
  state: WorkspaceLayoutState,
  storage?: Pick<Storage, "setItem">,
): void {
  const target = storage ?? (typeof window !== "undefined" ? window.localStorage : undefined);
  if (!target) return;
  try {
    target.setItem(WORKSPACE_LAYOUT_KEY, JSON.stringify({
      ...normalizeWorkspaceLayout(state),
      layoutVersion: WORKSPACE_LAYOUT_VERSION,
    }));
  } catch {
    // Storage can be unavailable in private or embedded contexts. The current
    // page still keeps the in-memory layout state.
  }
}

export function resizeWorkspacePane(
  state: WorkspaceLayoutState,
  pane: "left" | "right" | "graph",
  delta: number,
  viewportHeightOrOptions: number | WorkspacePaneResizeOptions = 1_000,
): WorkspaceLayoutState {
  if (!Number.isFinite(delta)) return state;
  const options = typeof viewportHeightOrOptions === "number"
    ? { viewportHeight: viewportHeightOrOptions }
    : viewportHeightOrOptions;
  if (pane === "left") {
    const paneBounds = options.bounds ?? bounds(WORKSPACE_LIMITS.leftMin, WORKSPACE_LIMITS.leftMax);
    const currentValue = finiteNumber(options.currentValue, state.leftWidth);
    return normalizeWorkspaceLayout({
      ...state,
      leftWidth: clamp(currentValue + delta, paneBounds.minimum, paneBounds.maximum),
      leftCollapsed: false,
    });
  }
  if (pane === "right") {
    const paneBounds = options.bounds ?? bounds(WORKSPACE_LIMITS.rightMin, WORKSPACE_LIMITS.rightMax);
    const currentValue = finiteNumber(options.currentValue, state.rightWidth);
    return normalizeWorkspaceLayout({
      ...state,
      rightWidth: clamp(currentValue - delta, paneBounds.minimum, paneBounds.maximum),
      rightCollapsed: false,
    });
  }
  const graphHeight = resolveWorkspaceGraphHeight(state, options.viewportHeight ?? 1_000);
  const paneBounds = options.bounds ?? graphHeight.bounds;
  const currentValue = finiteNumber(options.currentValue, graphHeight.height);
  const nextHeight = clamp(currentValue + delta, paneBounds.minimum, paneBounds.maximum);
  return normalizeWorkspaceLayout({ ...state, graphHeightRatio: nextHeight / graphHeight.usableHeight });
}
