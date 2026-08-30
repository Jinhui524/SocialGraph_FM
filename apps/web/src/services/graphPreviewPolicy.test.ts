import { describe, expect, it } from "vitest";

import { GRAPH_PREVIEW_POLICY } from "./graphPreviewPolicy";
import * as graphPreviewPolicyModule from "./graphPreviewPolicy";
import { graphTypeColour } from "./graphTypePalette";

describe("GRAPH_PREVIEW_POLICY", () => {
  it("keeps chat and governance accounts on one non-risk fill", () => {
    expect(graphTypeColour("account")).toBe(GRAPH_PREVIEW_POLICY.node.accountColour);
    expect(graphTypeColour("governance-account")).toBe(GRAPH_PREVIEW_POLICY.node.accountColour);
    expect(Object.values(GRAPH_PREVIEW_POLICY.risk)).not.toContain(GRAPH_PREVIEW_POLICY.node.accountColour);
  });

  it("locks answer-mode interaction and label budgets", () => {
    expect(GRAPH_PREVIEW_POLICY.dragThresholdPx).toBe(6);
    expect(GRAPH_PREVIEW_POLICY.labelLimit).toBe(12);
    expect(GRAPH_PREVIEW_POLICY.edge.potentialDash).toEqual([6, 5]);
  });

  it("uses the approved governance risk and relationship palette", () => {
    expect(GRAPH_PREVIEW_POLICY.risk).toEqual({
      low: "#3F8F8A",
      review: "#E5A53B",
      high: "#E75E58",
      context: "#5F7896",
      outline: "#5B3231",
      selection: "#7659EF",
    });
    expect(GRAPH_PREVIEW_POLICY.edge).toMatchObject({
      relationshipFactualColour: "#5F7896",
      potentialColour: "#7659EF",
      focusColour: "#22B8C7",
      riskHighColour: "#E75E58",
      riskReviewColour: "#E5A53B",
    });
  });

  it("keeps context relations subdued and matches governance arrowheads to the effective edge colour", () => {
    const resolveStyle = (graphPreviewPolicyModule as unknown as {
      governanceEdgeStyle?: (
        lens: "risk" | "relations" | undefined,
        value: string | number | boolean | undefined,
        options?: { readonly dark?: boolean; readonly focused?: boolean },
      ) => {
        readonly stroke: string;
        readonly opacity: number;
        readonly widthMultiplier: number;
        readonly arrowFill: string;
      };
    }).governanceEdgeStyle;
    expect(resolveStyle).toBeTypeOf("function");
    if (!resolveStyle) return;

    expect(resolveStyle("risk", "evidence-high")).toEqual({
      stroke: "#E75E58", opacity: 0.75, widthMultiplier: 1.6, arrowFill: "#E75E58",
    });
    expect(resolveStyle("risk", "evidence-review")).toEqual({
      stroke: "#E5A53B", opacity: 0.75, widthMultiplier: 1.6, arrowFill: "#E5A53B",
    });
    expect(resolveStyle("risk", "context")).toEqual({
      stroke: "#a8b9ce", opacity: 0.32, widthMultiplier: 1, arrowFill: "#a8b9ce",
    });
    expect(resolveStyle("relations", "factual", { focused: true })).toEqual({
      stroke: "#22B8C7", opacity: 0.94, widthMultiplier: 1.6, arrowFill: "#22B8C7",
    });
  });

  it("focuses only the exact endpoint and modality relation key", () => {
    const exactRelationKey = (graphPreviewPolicyModule as unknown as {
      governanceExactRelationKey?: (source: string, target: string, modalities: readonly string[]) => string;
    }).governanceExactRelationKey;
    const isExactRelation = (graphPreviewPolicyModule as unknown as {
      isExactGovernanceRelation?: (focusKey: string | undefined, source: string, target: string, modalities: readonly string[]) => boolean;
    }).isExactGovernanceRelation;
    expect(exactRelationKey).toBeTypeOf("function");
    expect(isExactRelation).toBeTypeOf("function");
    if (!exactRelationKey || !isExactRelation) return;
    const focusKey = exactRelationKey("n1", "n2", ["tweetSim", "coRT"]);
    expect(isExactRelation(focusKey, "n2", "n1", ["coRT", "tweetSim"])).toBe(true);
    expect(isExactRelation(focusKey, "n1", "n3", ["coRT", "tweetSim"])).toBe(false);
    expect(isExactRelation(focusKey, "n1", "n2", ["coRT"])).toBe(false);
  });
});
