import type {
  GraphEdge,
  GraphEdgeIdChurnSample,
  GraphEntityDiff,
  GraphFieldDiff,
  GraphNode,
  GraphVersion,
  GraphVersionDiffCount,
  GraphVersionDiffReport,
} from "../types/graph";
import { canonicalJson, compareUnicodeCodePoints, sha256Canonical } from "./graphIdentity";

export const DEFAULT_GRAPH_VERSION_DIFF_SAMPLE_LIMIT = 200;

function canonicalGraph(version: GraphVersion) {
  return {
    nodes: [...version.nodes]
      .sort((left, right) => compareUnicodeCodePoints(left.id, right.id))
      .map((node) => ({
        id: node.id,
        label: node.label,
        type: node.type,
        attributes: node.attributes,
      })),
    edges: [...version.edges]
      .sort((left, right) => compareUnicodeCodePoints(left.id, right.id))
      .map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: edge.type,
        weight: edge.weight,
        timestamp: edge.timestamp,
        directed: edge.directed,
        attributes: edge.attributes,
      })),
  };
}

function fieldDiff(field: string, before: unknown, after: unknown): GraphFieldDiff | null {
  if (before === undefined || after === undefined) {
    if (before === after) return null;
    return {
      field,
      ...(before !== undefined ? { before } : {}),
      ...(after !== undefined ? { after } : {}),
    };
  }
  if (canonicalJson(before) === canonicalJson(after)) return null;
  return {
    field,
    ...(before !== undefined ? { before } : {}),
    ...(after !== undefined ? { after } : {}),
  };
}

function collectObjectDiffs(
  before: unknown,
  after: unknown,
  prefix = "",
): GraphFieldDiff[] {
  const beforeRecord = before && typeof before === "object" && !Array.isArray(before)
    ? before as Record<string, unknown>
    : {};
  const afterRecord = after && typeof after === "object" && !Array.isArray(after)
    ? after as Record<string, unknown>
    : {};
  const keys = [...new Set([...Object.keys(beforeRecord), ...Object.keys(afterRecord)])]
    .sort(compareUnicodeCodePoints);
  const result: GraphFieldDiff[] = [];
  for (const key of keys) {
    const beforeValue = beforeRecord[key];
    const afterValue = afterRecord[key];
    const path = prefix ? `${prefix}.${key}` : key;
    const bothObjects = beforeValue && afterValue
      && typeof beforeValue === "object" && !Array.isArray(beforeValue)
      && typeof afterValue === "object" && !Array.isArray(afterValue);
    if (bothObjects) result.push(...collectObjectDiffs(beforeValue, afterValue, path));
    else {
      const change = fieldDiff(path, beforeValue, afterValue);
      if (change) result.push(change);
    }
  }
  return result;
}

function nodeFieldDiffs(before: GraphNode, after: GraphNode): GraphFieldDiff[] {
  return [
    fieldDiff("label", before.label, after.label),
    fieldDiff("type", before.type, after.type),
    fieldDiff("attributes", before.attributes, after.attributes),
  ].filter((change): change is GraphFieldDiff => Boolean(change));
}

function edgeFactFieldDiffs(before: GraphEdge, after: GraphEdge): GraphFieldDiff[] {
  return [
    fieldDiff("weight", before.weight, after.weight),
    fieldDiff("timestamp", before.timestamp, after.timestamp),
    fieldDiff("attributes", before.attributes, after.attributes),
  ].filter((change): change is GraphFieldDiff => Boolean(change));
}

type EdgeDirectionIdentity = "directed" | "undirected" | "unspecified";

interface CanonicalEdgeStructure {
  readonly source: string;
  readonly target: string;
  readonly type?: string;
  readonly direction: EdgeDirectionIdentity;
}

interface IndexedEdgeFact {
  readonly edge: GraphEdge;
  readonly factSignature: string;
}

function canonicalEdgeStructure(edge: GraphEdge): CanonicalEdgeStructure {
  const direction: EdgeDirectionIdentity = edge.directed === true
    ? "directed"
    : edge.directed === false
      ? "undirected"
      : "unspecified";
  const reverseUndirected = direction === "undirected"
    && compareUnicodeCodePoints(edge.source, edge.target) > 0;
  return {
    source: reverseUndirected ? edge.target : edge.source,
    target: reverseUndirected ? edge.source : edge.target,
    ...(edge.type !== undefined ? { type: edge.type } : {}),
    direction,
  };
}

function edgeStructuralKey(edge: GraphEdge): string {
  return canonicalJson(canonicalEdgeStructure(edge));
}

function edgeFactSignature(edge: GraphEdge): string {
  return canonicalJson({
    ...canonicalEdgeStructure(edge),
    ...(edge.weight !== undefined ? { weight: edge.weight } : {}),
    ...(edge.timestamp !== undefined ? { timestamp: edge.timestamp } : {}),
    attributes: edge.attributes,
  });
}

function emptyCount(): { added: number; removed: number; modified: number } {
  return { added: 0, removed: 0, modified: 0 };
}

function diffEntities<T extends GraphNode | GraphEdge>(
  entity: "node" | "edge",
  beforeEntities: readonly T[],
  afterEntities: readonly T[],
  compare: (before: T, after: T) => readonly GraphFieldDiff[],
): { count: GraphVersionDiffCount; changes: GraphEntityDiff[] } {
  const count = emptyCount();
  const changes: GraphEntityDiff[] = [];
  const beforeById = new Map(beforeEntities.map((item) => [item.id, item]));
  const afterById = new Map(afterEntities.map((item) => [item.id, item]));
  const ids = [...new Set([...beforeById.keys(), ...afterById.keys()])]
    .sort(compareUnicodeCodePoints);

  for (const id of ids) {
    const before = beforeById.get(id);
    const after = afterById.get(id);
    if (!before && after) {
      count.added += 1;
      changes.push({ entity, id, kind: "added", fields: [{ field: "entity", after }] });
      continue;
    }
    if (before && !after) {
      count.removed += 1;
      changes.push({ entity, id, kind: "removed", fields: [{ field: "entity", before }] });
      continue;
    }
    if (!before || !after) continue;
    const fields = compare(before, after);
    if (fields.length) {
      count.modified += 1;
      changes.push({ entity, id, kind: "modified", fields });
    }
  }
  return { count, changes };
}

function groupEdgeFacts(edges: readonly GraphEdge[]): Map<string, IndexedEdgeFact[]> {
  const groups = new Map<string, IndexedEdgeFact[]>();
  for (const edge of edges) {
    const structuralKey = edgeStructuralKey(edge);
    const current = groups.get(structuralKey) ?? [];
    current.push({ edge, factSignature: edgeFactSignature(edge) });
    groups.set(structuralKey, current);
  }
  for (const facts of groups.values()) {
    facts.sort((left, right) =>
      compareUnicodeCodePoints(left.factSignature, right.factSignature)
      || compareUnicodeCodePoints(left.edge.id, right.edge.id));
  }
  return groups;
}

function groupByFactSignature(facts: readonly IndexedEdgeFact[]) {
  const groups = new Map<string, IndexedEdgeFact[]>();
  for (const fact of facts) {
    const current = groups.get(fact.factSignature) ?? [];
    current.push(fact);
    groups.set(fact.factSignature, current);
  }
  for (const group of groups.values()) {
    group.sort((left, right) => compareUnicodeCodePoints(left.edge.id, right.edge.id));
  }
  return groups;
}

function pairSameIds(
  before: readonly IndexedEdgeFact[],
  after: readonly IndexedEdgeFact[],
): {
  readonly pairs: readonly (readonly [IndexedEdgeFact, IndexedEdgeFact])[];
  readonly beforeRemaining: readonly IndexedEdgeFact[];
  readonly afterRemaining: readonly IndexedEdgeFact[];
} {
  const afterById = new Map<string, IndexedEdgeFact[]>();
  for (const fact of after) {
    const current = afterById.get(fact.edge.id) ?? [];
    current.push(fact);
    afterById.set(fact.edge.id, current);
  }
  const pairs: Array<readonly [IndexedEdgeFact, IndexedEdgeFact]> = [];
  const beforeRemaining: IndexedEdgeFact[] = [];
  const matchedAfter = new Set<IndexedEdgeFact>();
  for (const fact of before) {
    const match = afterById.get(fact.edge.id)?.find((candidate) => !matchedAfter.has(candidate));
    if (match) {
      matchedAfter.add(match);
      pairs.push([fact, match]);
    } else {
      beforeRemaining.push(fact);
    }
  }
  return {
    pairs,
    beforeRemaining,
    afterRemaining: after.filter((fact) => !matchedAfter.has(fact)),
  };
}

function addedEdgeChange(edge: GraphEdge): GraphEntityDiff {
  return { entity: "edge", id: edge.id, kind: "added", fields: [{ field: "entity", after: edge }] };
}

function removedEdgeChange(edge: GraphEdge): GraphEntityDiff {
  return { entity: "edge", id: edge.id, kind: "removed", fields: [{ field: "entity", before: edge }] };
}

/**
 * Edge ids are parser-local row identities, not graph facts. Match the edge
 * multiset by canonical structure first and compare fact payloads only inside
 * each structural bucket. This keeps parallel edges exact without conflating a
 * CSV row reorder with an added/removed relationship.
 */
function diffEdges(
  beforeEdges: readonly GraphEdge[],
  afterEdges: readonly GraphEdge[],
): {
  readonly count: GraphVersionDiffCount;
  readonly changes: readonly GraphEntityDiff[];
  readonly idChurn: readonly GraphEdgeIdChurnSample[];
} {
  const count = emptyCount();
  const changes: GraphEntityDiff[] = [];
  const idChurn: GraphEdgeIdChurnSample[] = [];
  const beforeGroups = groupEdgeFacts(beforeEdges);
  const afterGroups = groupEdgeFacts(afterEdges);
  const structuralKeys = [...new Set([...beforeGroups.keys(), ...afterGroups.keys()])]
    .sort(compareUnicodeCodePoints);

  const recordPair = (
    before: IndexedEdgeFact,
    after: IndexedEdgeFact,
    structuralKey: string,
  ) => {
    if (before.edge.id !== after.edge.id) {
      idChurn.push({ beforeId: before.edge.id, afterId: after.edge.id, structuralKey });
    }
  };

  for (const structuralKey of structuralKeys) {
    const beforeFacts = beforeGroups.get(structuralKey) ?? [];
    const afterFacts = afterGroups.get(structuralKey) ?? [];
    if (!beforeFacts.length) {
      count.added += afterFacts.length;
      changes.push(...afterFacts.map(({ edge }) => addedEdgeChange(edge)));
      continue;
    }
    if (!afterFacts.length) {
      count.removed += beforeFacts.length;
      changes.push(...beforeFacts.map(({ edge }) => removedEdgeChange(edge)));
      continue;
    }

    const beforeByFact = groupByFactSignature(beforeFacts);
    const afterByFact = groupByFactSignature(afterFacts);
    const factSignatures = [...new Set([...beforeByFact.keys(), ...afterByFact.keys()])]
      .sort(compareUnicodeCodePoints);
    const beforeRemaining: IndexedEdgeFact[] = [];
    const afterRemaining: IndexedEdgeFact[] = [];

    for (const signature of factSignatures) {
      const beforeExact = beforeByFact.get(signature) ?? [];
      const afterExact = afterByFact.get(signature) ?? [];
      const sameIds = pairSameIds(beforeExact, afterExact);
      for (const [before, after] of sameIds.pairs) recordPair(before, after, structuralKey);
      const exactPairCount = Math.min(
        sameIds.beforeRemaining.length,
        sameIds.afterRemaining.length,
      );
      for (let index = 0; index < exactPairCount; index += 1) {
        recordPair(
          sameIds.beforeRemaining[index]!,
          sameIds.afterRemaining[index]!,
          structuralKey,
        );
      }
      beforeRemaining.push(...sameIds.beforeRemaining.slice(exactPairCount));
      afterRemaining.push(...sameIds.afterRemaining.slice(exactPairCount));
    }

    beforeRemaining.sort((left, right) =>
      compareUnicodeCodePoints(left.factSignature, right.factSignature)
      || compareUnicodeCodePoints(left.edge.id, right.edge.id));
    afterRemaining.sort((left, right) =>
      compareUnicodeCodePoints(left.factSignature, right.factSignature)
      || compareUnicodeCodePoints(left.edge.id, right.edge.id));
    const modifiedCount = Math.min(beforeRemaining.length, afterRemaining.length);
    for (let index = 0; index < modifiedCount; index += 1) {
      const before = beforeRemaining[index]!;
      const after = afterRemaining[index]!;
      recordPair(before, after, structuralKey);
      const fields = edgeFactFieldDiffs(before.edge, after.edge);
      if (fields.length) {
        count.modified += 1;
        changes.push({ entity: "edge", id: after.edge.id, kind: "modified", fields });
      }
    }
    const removed = beforeRemaining.slice(modifiedCount);
    const added = afterRemaining.slice(modifiedCount);
    count.removed += removed.length;
    count.added += added.length;
    changes.push(...removed.map(({ edge }) => removedEdgeChange(edge)));
    changes.push(...added.map(({ edge }) => addedEdgeChange(edge)));
  }

  idChurn.sort((left, right) =>
    compareUnicodeCodePoints(left.structuralKey, right.structuralKey)
    || compareUnicodeCodePoints(left.beforeId, right.beforeId)
    || compareUnicodeCodePoints(left.afterId, right.afterId));
  return { count, changes, idChurn };
}

export interface ComputeGraphVersionDiffOptions {
  readonly sampleLimit?: number;
}

/** Pure deterministic implementation shared by the Worker and no-Worker fallback. */
export function computeGraphVersionDiffCore(
  from: GraphVersion,
  to: GraphVersion,
  options: ComputeGraphVersionDiffOptions = {},
): GraphVersionDiffReport {
  const sampleLimit = Math.max(0, Math.floor(options.sampleLimit ?? DEFAULT_GRAPH_VERSION_DIFF_SAMPLE_LIMIT));
  const fromContentHash = from.contentHash ?? sha256Canonical(canonicalGraph(from));
  const toContentHash = to.contentHash ?? sha256Canonical(canonicalGraph(to));
  const sameContent = fromContentHash === toContentHash;

  const nodeDiff = sameContent
    ? { count: emptyCount(), changes: [] as GraphEntityDiff[] }
    : diffEntities("node", from.nodes, to.nodes, nodeFieldDiffs);
  const edgeDiff = sameContent
    ? { count: emptyCount(), changes: [] as GraphEntityDiff[], idChurn: [] as GraphEdgeIdChurnSample[] }
    : diffEdges(from.edges, to.edges);
  const allChanges = [...nodeDiff.changes, ...edgeDiff.changes]
    .sort((left, right) => compareUnicodeCodePoints(left.entity, right.entity) || compareUnicodeCodePoints(left.id, right.id));

  const versionFields = [
    fieldDiff("sourceFile", from.sourceFile, to.sourceFile),
    fieldDiff("parentVersionId", from.parentVersionId, to.parentVersionId),
    fieldDiff("sourceArtifactIds", from.sourceArtifactIds ?? [], to.sourceArtifactIds ?? []),
    fieldDiff("sourceHash", from.sourceHash, to.sourceHash),
    fieldDiff("buildSpecHash", from.buildSpecHash, to.buildSpecHash),
    fieldDiff("contentHash", fromContentHash, toContentHash),
    fieldDiff("directedness", from.metadata?.directedness, to.metadata?.directedness),
    fieldDiff("summary", from.summary, to.summary),
    fieldDiff("provenance", from.provenance, to.provenance),
    fieldDiff("datasetArtifact", from.datasetArtifact, to.datasetArtifact),
  ].filter((change): change is GraphFieldDiff => Boolean(change));

  return {
    fromVersionId: from.id,
    toVersionId: to.id,
    fromContentHash,
    toContentHash,
    fromHashSource: from.contentHash ? "stored" : "derived",
    toHashSource: to.contentHash ? "stored" : "derived",
    sameContent,
    sameLineage: from.parentVersionId === to.id
      || to.parentVersionId === from.id
      || Boolean(from.sourceHash && from.sourceHash === to.sourceHash),
    summary: {
      nodes: nodeDiff.count,
      edges: edgeDiff.count,
    },
    edgeIdChurn: {
      count: edgeDiff.idChurn.length,
      samples: edgeDiff.idChurn.slice(0, sampleLimit),
      truncated: edgeDiff.idChurn.length > sampleLimit,
    },
    versionFields,
    buildSpecFields: collectObjectDiffs(from.buildSpec, to.buildSpec),
    samples: allChanges.slice(0, sampleLimit),
    sampleLimit,
    truncated: allChanges.length > sampleLimit,
  };
}
