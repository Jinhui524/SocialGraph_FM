import type { NodeBadgeStyleProps } from "@antv/g6";
import { GRAPH_PREVIEW_POLICY } from "../../services/graphPreviewPolicy";
import { GRAPH_TYPE_COLOURS } from "../../services/graphTypePalette";
import type { AnalysisOverlay, GovernanceFocus, GraphEdge, GraphNode } from "../../types/graph";

export const UNKNOWN_TYPE = "未分类";
export const TYPE_COLOURS = GRAPH_TYPE_COLOURS;
export const EMPTY_GRAPH_NODES: readonly GraphNode[] = Object.freeze([]);
export const EMPTY_GRAPH_EDGES: readonly GraphEdge[] = Object.freeze([]);
export const REFERENCE_POSITIVE_FILL = "#D85C56";
export const REFERENCE_POSITIVE_OUTLINE = "#8E3733";
export const REFERENCE_NEGATIVE_OUTLINE = "#218B7C";

export function graphSemanticBadges(
  referenceLabel?: "positive" | "negative",
  reviewDecision?: "confirmed" | "rejected" | "pending",
): NodeBadgeStyleProps[] {
  const badges: NodeBadgeStyleProps[] = [];
  if (referenceLabel) badges.push({
    text: referenceLabel === "positive" ? "+" : "−",
    placement: "right-top",
    fill: "#ffffff",
    background: true,
    backgroundFill: referenceLabel === "positive" ? "#D85C56" : "#218B7C",
    backgroundRadius: 7,
    fontSize: 10,
    fontWeight: 700,
    padding: [1, 4],
  });
  if (reviewDecision) badges.push({
    text: reviewDecision === "confirmed" ? "✓" : reviewDecision === "rejected" ? "×" : "?",
    placement: "right-bottom",
    fill: "#ffffff",
    background: true,
    backgroundFill: reviewDecision === "confirmed" ? "#D85C56" : reviewDecision === "rejected" ? "#218B7C" : "#C58B2A",
    backgroundRadius: 7,
    fontSize: 9,
    fontWeight: 700,
    padding: [1, 4],
  });
  return badges;
}
export function graphPresentationGhostNodeIds(
  nodes: readonly Pick<GraphNode, "id">[],
  overlay: AnalysisOverlay | null | undefined,
): readonly string[] {
  // Analysis overlays may contain candidate endpoints that are not part of
  // the materialized graph. Rendering those endpoints as synthetic nodes was
  // the source of the ghost columns seen in the governance preview. Keep the
  // exported helper for compatibility, but never add non-fact nodes to G6.
  void nodes;
  void overlay;
  return Object.freeze([]);
}

/** A hidden graph must only refit on the false-to-true visibility transition. */
export function shouldAutoFitVisibleGraphPane(
  previouslyVisible: boolean,
  paneVisible: boolean,
): boolean {
  return !previouslyVisible && paneVisible;
}

/** A click-sized pointer wobble must never enter the drag/force lane. */
export function shouldBeginGraphDrag(
  start: readonly [number, number],
  current: readonly [number, number],
  threshold = GRAPH_PREVIEW_POLICY.dragThresholdPx,
): boolean {
  return Math.hypot(current[0] - start[0], current[1] - start[1]) > threshold;
}

const TYPE_LABELS: Readonly<Record<string, string>> = Object.freeze({
  person: "人员",
  people: "人员",
  user: "人员",
  organization: "组织",
  organisation: "组织",
  institution: "机构",
  project: "项目",
  community: "社区",
  factual_relation: "事实关系",
  potential_link: "潜在线索",
  未分类: "未分类",
});

export function typeLabel(type: string): string {
  return TYPE_LABELS[type.toLocaleLowerCase("zh-CN")] ?? type;
}

export function onlineRiskColour(score: number): string {
  const clamped = Math.max(0, Math.min(1, score));
  const toRgb = (hex: string) => [
    Number.parseInt(hex.slice(1, 3), 16),
    Number.parseInt(hex.slice(3, 5), 16),
    Number.parseInt(hex.slice(5, 7), 16),
  ] as const;
  const left = toRgb(clamped <= 0.5 ? GRAPH_PREVIEW_POLICY.risk.low : GRAPH_PREVIEW_POLICY.risk.review);
  const right = toRgb(clamped <= 0.5 ? GRAPH_PREVIEW_POLICY.risk.review : GRAPH_PREVIEW_POLICY.risk.high);
  const ratio = clamped <= 0.5 ? clamped * 2 : (clamped - 0.5) * 2;
  const channel = (index: number) => Math.round(left[index] + (right[index] - left[index]) * ratio)
    .toString(16).padStart(2, "0");
  return `#${channel(0)}${channel(1)}${channel(2)}`;
}

export function governanceSelectionStates(
  lens: "risk" | "relations" | undefined,
  related: boolean,
): { readonly node: readonly string[]; readonly edge: readonly string[] } {
  if (lens === "risk") {
    return { node: ["governance-selected"], edge: related ? ["governance-focus"] : [] };
  }
  if (lens === "relations") {
    return { node: ["governance-selected"], edge: related ? ["governance-focus"] : [] };
  }
  return { node: ["selected"], edge: related ? ["related"] : [] };
}

export function governanceFocusAppearanceChannels(
  focus: Pick<GovernanceFocus, "nodeIds" | "exactRelationKey"> | null | undefined,
  options: { readonly nodeId?: string; readonly relationKey?: string },
): {
  readonly focused: boolean;
  readonly opacity: number;
  readonly sizeMultiplier: number;
  readonly lineWidthMultiplier: number;
  readonly dualRing: boolean;
  readonly persistentLabel: boolean;
} {
  const focused = options.nodeId !== undefined
    ? Boolean(focus?.nodeIds.includes(options.nodeId))
    : options.relationKey !== undefined
      ? Boolean(focus?.exactRelationKey && focus.exactRelationKey === options.relationKey)
      : false;
  return {
    focused,
    opacity: focus ? focused ? options.relationKey !== undefined ? 0.94 : 1 : options.relationKey !== undefined ? 0.12 : 0.28 : 1,
    sizeMultiplier: focus && focused && options.nodeId !== undefined ? 1.28 : 1,
    lineWidthMultiplier: focus && focused ? 1.6 : 1,
    dualRing: Boolean(focus && focused && options.nodeId !== undefined),
    persistentLabel: Boolean(focus && focused && options.nodeId !== undefined),
  };
}

export function graphAppearanceRequestKey(focus: GovernanceFocus | null | undefined): string {
  return focus
    ? [focus.kind, focus.targetId, focus.nodeIds.join("\u001f"), focus.exactRelationKey ?? "", focus.cameraToken].join("\u0000")
    : "governance-focus:none";
}

export function shouldRelayoutProjection(options: {
  readonly topologyChanged: boolean;
  readonly hasCachedTopology: boolean;
  readonly externalFocusActive: boolean;
}): boolean {
  return options.topologyChanged && !options.hasCachedTopology && !options.externalFocusActive;
}
