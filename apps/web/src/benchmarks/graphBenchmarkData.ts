import { computeGraphSummary } from "../services/graphAlgorithms";
import { buildGraphScene, createDefaultGraphViewState } from "../services/graphScene";
import type { GraphEdge, GraphNode, GraphScene, GraphVersion } from "../types/graph";

export type GraphBenchmarkCaseId = "small" | "medium" | "large";

export interface GraphBenchmarkCase {
  readonly id: GraphBenchmarkCaseId;
  readonly nodeCount: 300 | 1_000 | 3_000;
  readonly edgeCount: 1_000 | 5_000 | 12_000;
  readonly communityCount: 4 | 8 | 12;
  readonly seed: string;
}

export const GRAPH_BENCHMARK_CASES: Readonly<Record<GraphBenchmarkCaseId, GraphBenchmarkCase>> = {
  small: {
    id: "small",
    nodeCount: 300,
    edgeCount: 1_000,
    communityCount: 4,
    seed: "sgfm-300-1000-v1",
  },
  medium: {
    id: "medium",
    nodeCount: 1_000,
    edgeCount: 5_000,
    communityCount: 8,
    seed: "sgfm-1000-5000-v1",
  },
  large: {
    id: "large",
    nodeCount: 3_000,
    edgeCount: 12_000,
    communityCount: 12,
    seed: "sgfm-3000-12000-v1",
  },
};

function seedFromString(value: string): number {
  let seed = 0x811c9dc5;
  for (let index = 0; index < value.length; index += 1) {
    seed ^= value.charCodeAt(index);
    seed = Math.imul(seed, 0x01000193);
  }
  return seed >>> 0 || 0x9e3779b9;
}

function createRandom(seedText: string): () => number {
  let state = seedFromString(seedText);
  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
}

function nodeId(index: number): string {
  return `member-${String(index + 1).padStart(4, "0")}`;
}

function edgeKey(left: number, right: number): string {
  return left < right ? `${left}:${right}` : `${right}:${left}`;
}

export function createBenchmarkGraphVersion(
  benchmarkCase: GraphBenchmarkCase,
): GraphVersion {
  const random = createRandom(benchmarkCase.seed);
  const nodes: GraphNode[] = Array.from({ length: benchmarkCase.nodeCount }, (_, index) => {
    const community = index % benchmarkCase.communityCount;
    return {
      id: nodeId(index),
      label: `成员 ${String(index + 1).padStart(4, "0")}`,
      type: `社区 ${community + 1}`,
      attributes: Object.freeze({ community, benchmark: true }),
    };
  });

  const pairs = new Set<string>();
  const edges: GraphEdge[] = [];
  let intraCommunityEdges = 0;
  let interCommunityEdges = 0;
  const addEdge = (sourceIndex: number, targetIndex: number): boolean => {
    if (sourceIndex === targetIndex) return false;
    const key = edgeKey(sourceIndex, targetIndex);
    if (pairs.has(key)) return false;
    pairs.add(key);
    const sourceCommunity = sourceIndex % benchmarkCase.communityCount;
    const targetCommunity = targetIndex % benchmarkCase.communityCount;
    const isIntraCommunity = sourceCommunity === targetCommunity;
    const index = edges.length;
    edges.push({
      id: `edge-${String(index + 1).padStart(5, "0")}`,
      source: nodeId(sourceIndex),
      target: nodeId(targetIndex),
      type: isIntraCommunity ? "社区协作" : "跨区协调",
      weight: Number((0.5 + random() * 4.5).toFixed(3)),
      timestamp: `${2020 + (index % 6)}-${String((index % 12) + 1).padStart(2, "0")}-01`,
      directed: false,
      attributes: Object.freeze({ benchmark: true }),
    });
    if (isIntraCommunity) intraCommunityEdges += 1;
    else interCommunityEdges += 1;
    return true;
  };

  // Connect every community internally, then connect community representatives.
  // This preserves connectivity without turning the whole scaffold into cross-community edges.
  for (let community = 0; community < benchmarkCase.communityCount; community += 1) {
    const members = nodes
      .map((_, index) => index)
      .filter((index) => index % benchmarkCase.communityCount === community);
    for (let index = 0; index < members.length; index += 1) {
      addEdge(members[index], members[(index + 1) % members.length]);
    }
  }
  for (let community = 0; community < benchmarkCase.communityCount; community += 1) {
    addEdge(community, (community + 1) % benchmarkCase.communityCount);
  }

  const targetIntraCommunityEdges = Math.floor(benchmarkCase.edgeCount * 0.8);
  const targetInterCommunityEdges = benchmarkCase.edgeCount - targetIntraCommunityEdges;
  let attempts = 0;
  const maximumAttempts = benchmarkCase.edgeCount * 100;
  while (edges.length < benchmarkCase.edgeCount && attempts < maximumAttempts) {
    attempts += 1;
    const sourceIndex = Math.floor(random() * benchmarkCase.nodeCount);
    const sourceCommunity = sourceIndex % benchmarkCase.communityCount;
    const intraRemaining = targetIntraCommunityEdges - intraCommunityEdges;
    const interRemaining = targetInterCommunityEdges - interCommunityEdges;
    const createIntraCommunityEdge = interRemaining <= 0 || (
      intraRemaining > 0 && random() < intraRemaining / (intraRemaining + interRemaining)
    );
    let targetIndex: number;
    if (createIntraCommunityEdge) {
      const communityOffset = Math.floor(random() * Math.ceil(benchmarkCase.nodeCount / benchmarkCase.communityCount));
      targetIndex = (sourceCommunity + communityOffset * benchmarkCase.communityCount) % benchmarkCase.nodeCount;
    } else {
      targetIndex = Math.floor(random() * benchmarkCase.nodeCount);
      if (targetIndex % benchmarkCase.communityCount === sourceCommunity) {
        targetIndex = (targetIndex + 1) % benchmarkCase.nodeCount;
      }
    }
    addEdge(sourceIndex, targetIndex);
  }

  if (edges.length !== benchmarkCase.edgeCount) {
    throw new Error(`Unable to create exact benchmark edge count for ${benchmarkCase.id}`);
  }
  if (
    intraCommunityEdges !== targetIntraCommunityEdges ||
    interCommunityEdges !== targetInterCommunityEdges
  ) {
    throw new Error(`Unable to create the 80/20 community mix for ${benchmarkCase.id}`);
  }

  const summary = computeGraphSummary(nodes, edges);
  const graphVersion: GraphVersion = Object.freeze({
    id: `benchmark-${benchmarkCase.id}-${benchmarkCase.seed}`,
    sourceFile: `${benchmarkCase.id}.synthetic.sgfm`,
    createdAt: "2026-08-10T00:00:00.000Z",
    nodes: Object.freeze(nodes),
    edges: Object.freeze(edges),
    summary: Object.freeze(summary),
    issues: Object.freeze([]),
    preview: Object.freeze({
      nodes: Object.freeze(nodes),
      edges: Object.freeze(edges),
      truncated: false,
      originalNodeCount: nodes.length,
      originalEdgeCount: edges.length,
    }),
    truncated: false,
  });
  return graphVersion;
}

export function createBenchmarkScene(graphVersion: GraphVersion): GraphScene {
  return buildGraphScene(graphVersion, {
    viewState: createDefaultGraphViewState(graphVersion.id),
    maxNodes: graphVersion.nodes.length,
    maxEdges: graphVersion.edges.length,
  });
}
