import { Graph, type LayoutOptions } from "@antv/g6";
import type { LayoutPreset } from "../../types/graph";
import { graphCanvasPerformanceProfile } from "./graphPerformancePolicy";
import type { ForceLayoutRuntime, ForceSettings } from "./graphRendererPolicy";

export function hashText(value: string) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function seededRandom(seedText: string) {
  let value = hashText(seedText) || 1;
  return () => {
    value += 0x6d2b79f5;
    let next = value;
    next = Math.imul(next ^ (next >>> 15), next | 1);
    next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
    return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
  };
}

export function setForceFixedPosition(
  graph: Graph,
  nodeId: string,
  position: [number, number] | null,
) {
  const runtime = graph as unknown as {
    context?: {
      layout?: { getLayoutInstance?: () => ForceLayoutRuntime[] };
    };
  };
  const layouts = runtime.context?.layout?.getLayoutInstance?.() ?? [];
  for (const wrapper of layouts) {
    const layout = wrapper.instance ?? wrapper;
    if (layout.id && layout.id !== "d3-force") continue;
    layout.setFixedPosition?.(nodeId, position);
  }
}

export function automaticLayoutPreset(nodeCount: number): LayoutPreset {
  if (nodeCount <= 50) return "spread";
  if (nodeCount <= 300) return "balanced";
  return "compact";
}

export function forceLayoutOptions(
  settings: ForceSettings,
  nodeCount: number,
  nodeScale: number,
  seed: string,
): LayoutOptions {
  const profile = graphCanvasPerformanceProfile(nodeCount);
  return {
    type: "d3-force",
    animation: false,
    // G6 executes exactly `iterations` ticks for non-animated layouts. Without
    // this bound it silently falls back to 300 synchronous ticks.
    iterations: profile.initialIterations,
    alphaMin: profile.alphaMin,
    alphaDecay: profile.alphaDecay,
    velocityDecay: profile.velocityDecay,
    randomSource: seededRandom(seed),
    center: { strength: settings.centerStrength },
    link: {
      distance: settings.linkDistance,
      strength: settings.linkStrength,
      iterations: nodeCount > 900 ? 1 : 2,
    },
    manyBody: {
      strength: settings.repulsion,
      theta: nodeCount > 900 ? 0.98 : 0.82,
    },
    collide: profile.collision
      ? {
          radius: (nodeCount > 900 ? 9 : nodeCount > 100 ? 15 : 24) * nodeScale,
          strength: 0.84,
          iterations: nodeCount > 180 ? 1 : 2,
        }
      : false,
  };
}
