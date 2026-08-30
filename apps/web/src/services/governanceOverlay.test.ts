import { describe, expect, it } from "vitest";

import {
  derivationPage,
  findingPage,
  onlineResult,
  onlineRun,
  onlineRunPreview,
} from "../test/fixtures/governanceOnline";
import { governancePreviewGraph } from "../components/GovernanceOnlineWorkspace";
import type { GovernanceOnlineFinding, GovernanceOnlinePreview } from "../types/governanceOnline";
import * as coreOverlayModule from "./governanceOverlay";

describe("SocialGraph-FM Governance governance overlay presentation", () => {
  const buildOnlineOverlay = (coreOverlayModule as unknown as {
    buildGovernanceOnlineGovernanceOverlay?: (
      graph: ReturnType<typeof governancePreviewGraph>,
      lens: "risk" | "community" | "relations" | "router",
      findings: readonly GovernanceOnlineFinding[],
      relations: ReturnType<typeof derivationPage>["items"],
      links: ReturnType<typeof derivationPage>["items"],
      run: ReturnType<typeof onlineRun>,
      result: ReturnType<typeof onlineResult>,
    ) => import("../types/graph").AnalysisOverlay;
  }).buildGovernanceOnlineGovernanceOverlay;

  it("colours risk bands and only direct derivation-backed evidence relations", () => {
    expect(buildOnlineOverlay).toBeTypeOf("function");
    if (!buildOnlineOverlay) return;
    const baseFindings = findingPage().items as unknown as readonly GovernanceOnlineFinding[];
    const findings = Object.freeze([
      ...baseFindings,
      Object.freeze({ ...baseFindings[2], nodeId: "n4", label: "匿名账号 4", rank: 4 }),
    ]);
    const baseGraph = governancePreviewGraph(onlineRunPreview() as GovernanceOnlinePreview, findings);
    const graph = Object.freeze({
      ...baseGraph,
      nodes: Object.freeze([
        ...baseGraph.nodes,
        Object.freeze({ id: "n4", label: "匿名账号 4", type: "账号", attributes: Object.freeze({}) }),
      ]),
      edges: Object.freeze([
        ...baseGraph.edges,
        Object.freeze({ id: "e2", source: "n2", target: "n3", type: "factual_relation", directed: false, weight: 1, attributes: Object.freeze({}) }),
        Object.freeze({ id: "e3", source: "n3", target: "n4", type: "factual_relation", directed: false, weight: 1, attributes: Object.freeze({}) }),
        Object.freeze({ id: "e4", source: "n1", target: "n4", type: "factual_relation", directed: false, weight: 1, attributes: Object.freeze({}) }),
      ]),
    });
    const relationBase = derivationPage("factual_relation").items[0];
    const relations = [
      relationBase,
      { ...relationBase, id: "factual_relation-review", source: "n2", target: "n3", nodeIds: ["n2", "n3"] },
      { ...relationBase, id: "factual_relation-context", source: "n3", target: "n4", nodeIds: ["n3", "n4"] },
    ] as unknown as ReturnType<typeof derivationPage>["items"];
    const overlay = buildOnlineOverlay(
      graph,
      "risk",
      findings,
      relations,
      derivationPage("potential_link").items,
      onlineRun(),
      onlineResult(),
    );

    expect(overlay.nodeValues).toEqual({
      n1: "risk-high",
      n2: "risk-review",
      n3: "risk-low",
      n4: "risk-low",
    });
    expect(overlay.edgeValues).toEqual({
      e1: "evidence-high",
      e2: "evidence-review",
      e3: "context",
    });
    expect(overlay.edgeValues).not.toHaveProperty("e4");
    expect(overlay.candidateEdges).toBeUndefined();
    expect(overlay.presentation).toEqual({
      governanceLens: "risk",
      riskBands: { n1: "high", n2: "review", n3: "low", n4: "low" },
    });
  });

  it("retains node risk hints while separating solid facts from dashed clues", () => {
    expect(buildOnlineOverlay).toBeTypeOf("function");
    if (!buildOnlineOverlay) return;
    const findings = findingPage().items as unknown as readonly GovernanceOnlineFinding[];
    const graph = governancePreviewGraph(onlineRunPreview() as GovernanceOnlinePreview, findings);
    const overlay = buildOnlineOverlay(
      graph,
      "relations",
      findings,
      derivationPage("factual_relation").items,
      derivationPage("potential_link").items,
      onlineRun(),
      onlineResult(),
    );

    expect(overlay.nodeValues).toEqual({});
    expect(overlay.edgeValues).toEqual({ e1: "factual" });
    expect(overlay.candidateEdges).toEqual([
      {
        id: "__socialgraph_research_candidate__potential_link-1",
        sourceId: "n1",
        targetId: "n2",
        directed: false,
        exactRelationKey: "n1\u0000n2\u0000coRT",
      },
    ]);
    expect(overlay.legend.items.map((item) => item.label)).toEqual(["事实关系", "潜在线索"]);
    expect(overlay.presentation).toEqual({
      governanceLens: "relations",
      riskBands: { n1: "high", n2: "review", n3: "low" },
    });
  });

  it("keeps community colour values while exposing risk as a second channel", () => {
    expect(buildOnlineOverlay).toBeTypeOf("function");
    if (!buildOnlineOverlay) return;
    const findings = findingPage().items as unknown as readonly GovernanceOnlineFinding[];
    const graph = governancePreviewGraph(onlineRunPreview() as GovernanceOnlinePreview, findings);
    const overlay = buildOnlineOverlay(
      graph,
      "community",
      findings,
      derivationPage("factual_relation").items,
      derivationPage("potential_link").items,
      onlineRun(),
      onlineResult(),
    );

    expect(overlay.nodeValues).toEqual({ n1: "group-1", n2: "group-1", n3: "group-1" });
    expect(overlay.presentation).toEqual({
      riskBands: { n1: "high", n2: "review", n3: "low" },
    });
  });
});
