import type { GraphEdge, GraphVersion } from "../types/graph";
import type { CoreRegisteredEdgeIdentity } from "../types/core";
import { deepFreeze } from "./coreContracts";
import { compareUnicodeCodePoints, sha256Canonical } from "./graphIdentity";

export interface RegisteredEdgeIdentityEntry {
  readonly localEdgeId: string;
  readonly identity: CoreRegisteredEdgeIdentity;
}

export interface RegisteredEdgeIdentityIndex {
  readonly entries: readonly RegisteredEdgeIdentityEntry[];
  readonly unprovableEdgeIds: readonly string[];
}

type CachedLocalIdentity =
  | { readonly identity: CoreRegisteredEdgeIdentity }
  | { readonly errorCode: string };

const localIdentityCache = new WeakMap<GraphVersion, Map<string, CachedLocalIdentity>>();
const provenHashCache = new WeakMap<GraphVersion, Map<string, RegisteredEdgeIdentityEntry>>();
const matchedHashCache = new WeakMap<GraphVersion, Map<string, readonly RegisteredEdgeIdentityEntry[]>>();

function cachedFailure(error: unknown): string | null {
  if (!(error instanceof Error)) return null;
  return new Set([
    "GFM_CORE_TARGET_NOT_FOUND",
    "GFM_CORE_EDGE_IDENTITY_UNPROVABLE",
    "GFM_CORE_EDGE_IDENTITY_DUPLICATE",
  ]).has(error.message) ? error.message : null;
}

function graphDirection(graph: GraphVersion): "directed" | "undirected" {
  const directedness = graph.metadata?.directedness;
  if (directedness !== "directed" && directedness !== "undirected") {
    throw new Error("GFM_CORE_EDGE_IDENTITY_UNPROVABLE");
  }
  return directedness;
}

function identityPayload(
  graph: GraphVersion,
  edge: GraphEdge,
): Omit<CoreRegisteredEdgeIdentity, "edgeHash"> {
  const direction = graphDirection(graph);
  if (
    (direction === "directed" && edge.directed !== true)
    || (direction === "undirected" && edge.directed !== false)
    || typeof edge.type !== "string"
    || edge.type.length === 0
    || edge.type.length > 200
    || typeof edge.weight !== "number"
    || !Number.isFinite(edge.weight)
    || edge.source.length === 0
    || edge.source.length > 500
    || edge.target.length === 0
    || edge.target.length > 500
  ) throw new Error("GFM_CORE_EDGE_IDENTITY_UNPROVABLE");

  let sourceId = edge.source;
  let targetId = edge.target;
  if (direction === "undirected" && compareUnicodeCodePoints(sourceId, targetId) > 0) {
    [sourceId, targetId] = [targetId, sourceId];
  }
  return {
    schemaVersion: "socialgraph-fm.core-edge-identity/2.0",
    sourceId,
    targetId,
    edgeType: edge.type,
    weight: edge.weight,
  };
}

function identityForEdge(graph: GraphVersion, edge: GraphEdge): CoreRegisteredEdgeIdentity {
  const payload = identityPayload(graph, edge);
  return deepFreeze({ ...payload, edgeHash: sha256Canonical(payload) });
}

export function buildRegisteredEdgeIdentityIndex(
  graph: GraphVersion,
  candidates: readonly GraphEdge[] = graph.edges,
): RegisteredEdgeIdentityIndex {
  const entries: RegisteredEdgeIdentityEntry[] = [];
  const unprovableEdgeIds: string[] = [];
  const byHash = new Map<string, string>();
  for (const edge of candidates) {
    let identity: CoreRegisteredEdgeIdentity;
    try {
      identity = identityForEdge(graph, edge);
    } catch {
      unprovableEdgeIds.push(edge.id);
      continue;
    }
    const existing = byHash.get(identity.edgeHash);
    if (existing && existing !== edge.id) throw new Error("GFM_CORE_EDGE_IDENTITY_DUPLICATE");
    byHash.set(identity.edgeHash, edge.id);
    entries.push(deepFreeze({ localEdgeId: edge.id, identity }));
  }
  return deepFreeze({ entries, unprovableEdgeIds });
}

export function registeredEdgeIdentityForLocalId(
  graph: GraphVersion,
  localEdgeId: string,
): CoreRegisteredEdgeIdentity {
  let graphCache = localIdentityCache.get(graph);
  if (!graphCache) {
    graphCache = new Map();
    localIdentityCache.set(graph, graphCache);
  }
  const cached = graphCache.get(localEdgeId);
  if (cached) {
    if ("errorCode" in cached) throw new Error(cached.errorCode);
    return cached.identity;
  }
  try {
    const target = graph.edges.find((edge) => edge.id === localEdgeId);
    if (!target) throw new Error("GFM_CORE_TARGET_NOT_FOUND");
    const identity = identityForEdge(graph, target);
    const targetPayload = identityPayload(graph, target);
    for (const edge of graph.edges) {
      if (edge.id === target.id) continue;
      try {
        const candidate = identityPayload(graph, edge);
        if (
          candidate.sourceId === targetPayload.sourceId
          && candidate.targetId === targetPayload.targetId
          && candidate.edgeType === targetPayload.edgeType
          && candidate.weight === targetPayload.weight
        ) {
          throw new Error("GFM_CORE_EDGE_IDENTITY_DUPLICATE");
        }
      } catch (error) {
        if (error instanceof Error && error.message === "GFM_CORE_EDGE_IDENTITY_DUPLICATE") throw error;
      }
    }
    const entry = deepFreeze({ localEdgeId, identity });
    let hashCache = provenHashCache.get(graph);
    if (!hashCache) {
      hashCache = new Map();
      provenHashCache.set(graph, hashCache);
    }
    const existing = hashCache.get(identity.edgeHash);
    if (existing && existing.localEdgeId !== localEdgeId) {
      throw new Error("GFM_CORE_EDGE_IDENTITY_DUPLICATE");
    }
    hashCache.set(identity.edgeHash, entry);
    graphCache.set(localEdgeId, { identity });
    return identity;
  } catch (error) {
    const errorCode = cachedFailure(error);
    if (errorCode) graphCache.set(localEdgeId, { errorCode });
    throw error;
  }
}

export function matchRegisteredEdgeHashes(
  graph: GraphVersion,
  requiredHashes: ReadonlySet<string>,
): readonly RegisteredEdgeIdentityEntry[] {
  const hashes = new Set([...requiredHashes].filter((value) => /^[0-9a-f]{64}$/u.test(value)));
  if (hashes.size === 0) return Object.freeze([]);
  const cacheKey = [...hashes].sort(compareUnicodeCodePoints).join(",");
  let graphCache = matchedHashCache.get(graph);
  if (!graphCache) {
    graphCache = new Map();
    matchedHashCache.set(graph, graphCache);
  }
  const cached = graphCache.get(cacheKey);
  if (cached) return cached;
  const proven = provenHashCache.get(graph);
  const matched = new Map<string, RegisteredEdgeIdentityEntry>();
  for (const hash of hashes) {
    const entry = proven?.get(hash);
    if (entry) matched.set(hash, entry);
  }
  if (matched.size === hashes.size) {
    const result = deepFreeze([...matched.values()]);
    graphCache.set(cacheKey, result);
    return result;
  }
  for (const edge of graph.edges) {
    let identity: CoreRegisteredEdgeIdentity;
    try {
      identity = identityForEdge(graph, edge);
    } catch {
      continue;
    }
    if (!hashes.has(identity.edgeHash)) continue;
    const existing = matched.get(identity.edgeHash);
    if (existing && existing.localEdgeId !== edge.id) {
      throw new Error("GFM_CORE_EDGE_IDENTITY_DUPLICATE");
    }
    matched.set(identity.edgeHash, deepFreeze({ localEdgeId: edge.id, identity }));
    if (matched.size === hashes.size) break;
  }
  const result = deepFreeze([...matched.values()]);
  graphCache.set(cacheKey, result);
  return result;
}
