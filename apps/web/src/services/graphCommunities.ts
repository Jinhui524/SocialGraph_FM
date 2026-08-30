import { UndirectedGraph } from "graphology";
import louvain from "graphology-communities-louvain";

import type { GraphVersion } from "../types/graph";

export interface CommunityAnalysis {
  readonly assignments: Readonly<Record<string, string>>;
  readonly communities: readonly (readonly string[])[];
  readonly modularity: number;
  readonly algorithm: "louvain";
  readonly seed: string;
}

function compareIds(left: string, right: string): number {
  return left.localeCompare(right, "zh-CN");
}

function hashSeed(seed: string): number {
  let hash = 2_166_136_261;
  for (let index = 0; index < seed.length; index += 1) {
    hash ^= seed.charCodeAt(index);
    hash = Math.imul(hash, 16_777_619);
  }
  return hash >>> 0;
}

/** Mulberry32 gives Louvain a repeatable random walk for a graph version. */
export function createSeededRandom(seed: string): () => number {
  let state = hashSeed(seed) || 0x6d2b79f5;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function canonicalPair(source: string, target: string): string {
  return compareIds(source, target) <= 0
    ? `${source}\u0000${target}`
    : `${target}\u0000${source}`;
}

function buildLouvainGraph(graph: Pick<GraphVersion, "nodes" | "edges">): UndirectedGraph {
  const result = new UndirectedGraph({ allowSelfLoops: false, multi: false });
  const nodeIds = [...new Set(graph.nodes.map((node) => node.id))].sort(compareIds);
  for (const nodeId of nodeIds) result.addNode(nodeId);

  const weights = new Map<string, number>();
  for (const edge of graph.edges) {
    if (!result.hasNode(edge.source) || !result.hasNode(edge.target)) continue;
    if (edge.source === edge.target) continue;
    const weight = edge.weight === undefined ? 1 : Math.max(0, edge.weight);
    if (!Number.isFinite(weight) || weight <= 0) continue;
    const pair = canonicalPair(edge.source, edge.target);
    weights.set(pair, (weights.get(pair) ?? 0) + weight);
  }

  for (const [pair, weight] of [...weights.entries()].sort(([left], [right]) => compareIds(left, right))) {
    const [source, target] = pair.split("\u0000");
    result.addEdgeWithKey(`edge-${result.size + 1}`, source, target, { weight });
  }
  return result;
}

function canonicalizeCommunities(raw: Readonly<Record<string, number>>): {
  assignments: Readonly<Record<string, string>>;
  communities: readonly (readonly string[])[];
} {
  const grouped = new Map<number, string[]>();
  for (const [nodeId, community] of Object.entries(raw)) {
    const members = grouped.get(community) ?? [];
    members.push(nodeId);
    grouped.set(community, members);
  }

  const communities = [...grouped.values()]
    .map((members) => members.sort(compareIds))
    .sort((left, right) => {
      if (right.length !== left.length) return right.length - left.length;
      return compareIds(left[0] ?? "", right[0] ?? "");
    });
  const assignments: Record<string, string> = {};
  communities.forEach((members, index) => {
    const communityId = `community-${index + 1}`;
    for (const nodeId of members) assignments[nodeId] = communityId;
  });

  return {
    assignments: Object.freeze(assignments),
    communities: Object.freeze(communities.map((members) => Object.freeze([...members]))),
  };
}

/**
 * Runs true modularity-based Louvain on an undirected, weighted simple
 * projection. Parallel relationships are aggregated and the source graph is
 * never mutated.
 */
export function computeLouvainCommunities(
  graph: Pick<GraphVersion, "nodes" | "edges"> & Partial<Pick<GraphVersion, "id">>,
  requestedSeed?: string,
): CommunityAnalysis {
  const seed = requestedSeed ?? graph.id ?? "socialgraph-fm";
  const projected = buildLouvainGraph(graph);

  if (projected.order === 0) {
    return Object.freeze({
      assignments: Object.freeze({}),
      communities: Object.freeze([]),
      modularity: 0,
      algorithm: "louvain" as const,
      seed,
    });
  }

  if (projected.size === 0) {
    const raw = Object.fromEntries(projected.nodes().sort(compareIds).map((nodeId, index) => [nodeId, index]));
    return Object.freeze({
      ...canonicalizeCommunities(raw),
      modularity: 0,
      algorithm: "louvain" as const,
      seed,
    });
  }

  const detailed = louvain.detailed(projected, {
    getEdgeWeight: "weight",
    randomWalk: true,
    rng: createSeededRandom(seed),
    resolution: 1,
  });
  const canonical = canonicalizeCommunities(detailed.communities);
  return Object.freeze({
    ...canonical,
    modularity: Number(detailed.modularity.toFixed(6)),
    algorithm: "louvain" as const,
    seed,
  });
}
