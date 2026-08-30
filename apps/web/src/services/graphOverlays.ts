import type {
  AnalysisOverlay,
  GraphPath,
  GraphVersion,
  OverlayLegend,
} from "../types/graph";
import { computeLouvainCommunities, type CommunityAnalysis } from "./graphCommunities";
import {
  findArticulationPoints,
  getConnectedComponents,
  rankDegreeCentrality,
} from "./graphAlgorithms";

const COMMUNITY_COLORS = [
  "#7367f0",
  "#21b7c5",
  "#3f7ae0",
  "#ef8d4f",
  "#e36a91",
  "#65ae72",
  "#ab72dc",
  "#d3a132",
];

function freezeValues(
  values: Readonly<Record<string, string | number | boolean>>,
): Readonly<Record<string, string | number | boolean>> {
  return Object.freeze({ ...values });
}

function freezeLegend(legend: OverlayLegend): OverlayLegend {
  return Object.freeze({
    ...legend,
    items: Object.freeze(legend.items.map((item) => Object.freeze({ ...item }))),
  });
}

function createOverlay(
  graphVersionId: string,
  kind: AnalysisOverlay["kind"],
  nodeValues: Readonly<Record<string, string | number | boolean>>,
  edgeValues: Readonly<Record<string, string | number | boolean>>,
  legend: OverlayLegend,
  algorithm: string,
  runId?: string,
): AnalysisOverlay {
  return Object.freeze({
    id: `${graphVersionId}:${kind}:${runId ?? algorithm}`,
    graphVersionId,
    kind,
    nodeValues: freezeValues(nodeValues),
    edgeValues: freezeValues(edgeValues),
    legend: freezeLegend(legend),
    provenance: Object.freeze({
      engine: "local_algorithm",
      algorithm,
      ...(runId ? { runId } : {}),
    }),
  });
}

export function buildRawFactsOverlay(graph: GraphVersion): AnalysisOverlay {
  return createOverlay(
    graph.id,
    "raw",
    {},
    {},
    { title: "原始图事实", items: [] },
    "raw-graph-facts",
  );
}

export function buildDegreeOverlay(graph: GraphVersion, runId?: string): AnalysisOverlay {
  const values = Object.fromEntries(
    rankDegreeCentrality(graph).map((entry) => [entry.nodeId, entry.normalizedScore]),
  );
  return createOverlay(
    graph.id,
    "degree",
    values,
    {},
    {
      title: "度中心性",
      items: [
        { value: "low", label: "较低", color: "#c9c7ef" },
        { value: "medium", label: "中等", color: "#8177ef" },
        { value: "high", label: "较高", color: "#4d3fca" },
      ],
    },
    "degree-centrality",
    runId,
  );
}

export function buildArticulationOverlay(graph: GraphVersion, runId?: string): AnalysisOverlay {
  const values = Object.fromEntries(findArticulationPoints(graph).map((nodeId) => [nodeId, true]));
  return createOverlay(
    graph.id,
    "articulation",
    values,
    {},
    { title: "割点", items: [{ value: "true", label: "桥接节点", color: "#ef8d4f" }] },
    "tarjan-articulation-points",
    runId,
  );
}

export function buildComponentsOverlay(graph: GraphVersion, runId?: string): AnalysisOverlay {
  const components = getConnectedComponents(graph);
  const values: Record<string, string> = {};
  components.forEach((members, index) => {
    for (const nodeId of members) values[nodeId] = `component-${index + 1}`;
  });
  return createOverlay(
    graph.id,
    "components",
    values,
    {},
    {
      title: "连通分量",
      items: components.slice(0, 20).map((_members, index) => ({
        value: `component-${index + 1}`,
        label: `分量 ${index + 1}`,
        color: COMMUNITY_COLORS[index % COMMUNITY_COLORS.length],
      })),
    },
    "undirected-connected-components",
    runId,
  );
}

export function buildCommunityOverlay(
  graph: GraphVersion,
  analysis: CommunityAnalysis = computeLouvainCommunities(graph),
  runId?: string,
): AnalysisOverlay {
  return createOverlay(
    graph.id,
    "community",
    analysis.assignments,
    {},
    {
      title: "Louvain 社区",
      items: analysis.communities.slice(0, 20).map((_members, index) => ({
        value: `community-${index + 1}`,
        label: `社区 ${index + 1}`,
        color: COMMUNITY_COLORS[index % COMMUNITY_COLORS.length],
      })),
    },
    `louvain:resolution=1:seed=${analysis.seed}`,
    runId,
  );
}

export function buildPathOverlay(
  graph: GraphVersion,
  path: GraphPath,
  runId?: string,
): AnalysisOverlay {
  return createOverlay(
    graph.id,
    "path",
    Object.fromEntries(path.nodeIds.map((nodeId, index) => [nodeId, index])),
    Object.fromEntries(path.edgeIds.map((edgeId) => [edgeId, true])),
    { title: "最短路径", items: [{ value: "true", label: "路径", color: "#7566f2" }] },
    "unweighted-bfs-shortest-path",
    runId,
  );
}
