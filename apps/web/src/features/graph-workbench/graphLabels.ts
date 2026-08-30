import type { GraphNode } from "../../types/graph";
import { UNKNOWN_TYPE } from "./graphPresentation";

export function nodeType(node: GraphNode) {
  return node.type?.trim() || UNKNOWN_TYPE;
}
export function compactGraphLabel(label: string, limit = 10) {
  const characters = Array.from(label.trim());
  return characters.length > limit
    ? `${characters.slice(0, limit).join("")}…`
    : characters.join("");
}

export function graphLabelIdsForZoom(
  nodes: readonly Pick<GraphNode, "id">[],
  degreeById: ReadonlyMap<string, number>,
  options: {
    readonly zoom: number;
    readonly threshold: number;
    readonly labelLimit?: number;
    readonly selectedNodeId?: string | null;
    readonly focusNodeIds: readonly string[];
    readonly pathNodeIds: ReadonlySet<string>;
  },
) {
  const threshold = Math.max(0, Math.min(100, options.threshold));
  const labelLimit = options.labelLimit === undefined || !Number.isFinite(options.labelLimit)
    ? undefined
    : Math.max(0, Math.floor(options.labelLimit));
  const sorted = [...nodes].sort(
    (left, right) =>
      (degreeById.get(right.id) ?? 0) - (degreeById.get(left.id) ?? 0),
  );
  if (sorted.length <= 50 && labelLimit === undefined) {
    return new Set(sorted.map((node) => node.id));
  }

  let count = Math.max(4, Math.ceil(sorted.length * (1 - threshold / 100)));
  const sizeCap = sorted.length > 1_000 ? 80 : sorted.length > 300 ? 50 : sorted.length;
  count = Math.min(count, sizeCap);
  if (options.zoom < 0.55) count = Math.min(count, 8);
  else if (options.zoom < 0.85) count = Math.min(count, 18);
  else if (options.zoom > 1.45 && sorted.length <= 220) count = sorted.length;
  else if (options.zoom > 1.9) count = Math.min(sorted.length, Math.max(count, 120));
  count = Math.min(count, sizeCap);
  if (labelLimit !== undefined) count = Math.min(count, labelLimit);

  const result = new Set(sorted.slice(0, count).map((node) => node.id));
  if (options.selectedNodeId) result.add(options.selectedNodeId);
  for (const id of options.focusNodeIds) result.add(id);
  for (const id of options.pathNodeIds) result.add(id);
  return result;
}

export function joinClassNames(...names: Array<string | false | null | undefined>) {
  return names.filter(Boolean).join(" ");
}

export function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)").matches
  );
}

export function graphAnimation(): false | { duration: number; easing: string } {
  return prefersReducedMotion()
    ? false
    : {
        duration: 260,
        easing: "ease-out",
      };
}
