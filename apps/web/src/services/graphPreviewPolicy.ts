export const GRAPH_PREVIEW_POLICY = Object.freeze({
  dragThresholdPx: 6,
  labelLimit: 12,
  node: Object.freeze({
    minimumSize: 14,
    degreeScale: 6,
    accountColour: "#4d86c6",
  }),
  edge: Object.freeze({
    factualColour: "#a8b9ce",
    relationshipFactualColour: "#5F7896",
    factualOpacity: 0.32,
    selectedOpacity: 0.75,
    potentialColour: "#7659EF",
    potentialDash: Object.freeze([6, 5] as const),
    focusColour: "#22B8C7",
    riskHighColour: "#E75E58",
    riskReviewColour: "#E5A53B",
  }),
  risk: Object.freeze({
    low: "#3F8F8A",
    review: "#E5A53B",
    high: "#E75E58",
    context: "#5F7896",
    outline: "#5B3231",
    selection: "#7659EF",
  }),
} as const);

export type GraphPreviewPolicy = typeof GRAPH_PREVIEW_POLICY;

export function governanceExactRelationKey(
  source: string,
  target: string,
  modalities: readonly string[],
): string {
  const endpoints = [source, target].sort((left, right) => left.localeCompare(right));
  const relationModalities = [...new Set(modalities)].sort((left, right) => left.localeCompare(right));
  return `${endpoints[0]}\u0000${endpoints[1]}\u0000${relationModalities.join("\u001f")}`;
}

export function isExactGovernanceRelation(
  focusKey: string | undefined,
  source: string,
  target: string,
  modalities: readonly string[],
): boolean {
  return Boolean(focusKey) && focusKey === governanceExactRelationKey(source, target, modalities);
}

export function governanceEdgeStyle(
  lens: "risk" | "relations" | undefined,
  value: string | number | boolean | undefined,
  options: { readonly dark?: boolean; readonly focused?: boolean } = {},
): {
  readonly stroke: string;
  readonly opacity: number;
  readonly widthMultiplier: number;
  readonly arrowFill: string;
} {
  const factual = options.dark ? "#77869a" : GRAPH_PREVIEW_POLICY.edge.factualColour;
  if (lens === "risk") {
    const stroke = value === "evidence-high"
      ? GRAPH_PREVIEW_POLICY.edge.riskHighColour
      : value === "evidence-review" ? GRAPH_PREVIEW_POLICY.edge.riskReviewColour : factual;
    const emphasized = value === "evidence-high" || value === "evidence-review";
    return {
      stroke,
      opacity: emphasized
        ? GRAPH_PREVIEW_POLICY.edge.selectedOpacity
        : options.dark ? 0.42 : GRAPH_PREVIEW_POLICY.edge.factualOpacity,
      widthMultiplier: emphasized ? 1.6 : 1,
      arrowFill: stroke,
    };
  }
  if (lens === "relations") {
    const stroke = options.focused
      ? GRAPH_PREVIEW_POLICY.edge.focusColour
      : GRAPH_PREVIEW_POLICY.edge.relationshipFactualColour;
    return {
      stroke,
      opacity: options.focused ? 0.94 : 0.54,
      widthMultiplier: options.focused ? 1.6 : 1,
      arrowFill: stroke,
    };
  }
  const stroke = value === undefined ? factual : GRAPH_PREVIEW_POLICY.edge.potentialColour;
  return {
    stroke,
    opacity: value === undefined
      ? options.dark ? 0.42 : GRAPH_PREVIEW_POLICY.edge.factualOpacity
      : GRAPH_PREVIEW_POLICY.edge.selectedOpacity,
    widthMultiplier: value === undefined ? 1 : 1.6,
    arrowFill: stroke,
  };
}
