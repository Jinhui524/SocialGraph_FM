import type { GraphNode, GraphVersion, TargetResolution } from "../types/graph";

function compareIds(left: string, right: string): number {
  return left.localeCompare(right, "zh-CN");
}

export function normalizeNodeTerm(value: string): string {
  return value
    .normalize("NFKC")
    .trim()
    .toLocaleLowerCase("zh-CN")
    .replace(/[\s·•_.-]+/gu, "");
}

function candidatesForExact(nodes: readonly GraphNode[], normalized: string): string[] {
  return nodes
    .filter(
      (node) => normalizeNodeTerm(node.id) === normalized || normalizeNodeTerm(node.label) === normalized,
    )
    .map((node) => node.id)
    .sort(compareIds);
}

function candidatesForSubstring(nodes: readonly GraphNode[], normalized: string): string[] {
  if (!normalized) return [];
  return nodes
    .filter((node) => {
      const normalizedId = normalizeNodeTerm(node.id);
      const normalizedLabel = normalizeNodeTerm(node.label);
      return normalizedId.includes(normalized) || normalizedLabel.includes(normalized);
    })
    .map((node) => node.id)
    .sort(compareIds);
}

export function resolveViewTarget(
  graph: Pick<GraphVersion, "nodes">,
  requestedTerm: string,
): TargetResolution {
  const term = requestedTerm.trim();
  const idMatch = graph.nodes.find((node) => node.id === term);
  if (idMatch) return Object.freeze({ status: "resolved", term, nodeId: idMatch.id, match: "id_exact" });

  const normalized = normalizeNodeTerm(term);
  if (!normalized) return Object.freeze({ status: "not_found", term });

  const exact = [...new Set(candidatesForExact(graph.nodes, normalized))];
  if (exact.length === 1) {
    return Object.freeze({
      status: "resolved",
      term,
      nodeId: exact[0],
      match: "normalized_exact",
    });
  }
  if (exact.length > 1) {
    return Object.freeze({
      status: "ambiguous",
      term,
      candidateNodeIds: Object.freeze(exact),
    });
  }

  const substring = [...new Set(candidatesForSubstring(graph.nodes, normalized))];
  if (substring.length === 1) {
    return Object.freeze({
      status: "resolved",
      term,
      nodeId: substring[0],
      match: "unique_substring",
    });
  }
  if (substring.length > 1) {
    return Object.freeze({
      status: "ambiguous",
      term,
      candidateNodeIds: Object.freeze(substring),
    });
  }
  return Object.freeze({ status: "not_found", term });
}

export function resolveViewTargets(
  graph: Pick<GraphVersion, "nodes">,
  terms: readonly string[],
): readonly TargetResolution[] {
  return Object.freeze(terms.map((term) => resolveViewTarget(graph, term)));
}
