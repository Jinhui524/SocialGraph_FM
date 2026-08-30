import type {
  GraphRendererPreference,
  GraphRendererStatus,
  LayoutPreset,
} from "../../types/graph";

export interface ForceSettings {
  centerStrength: number;
  repulsion: number;
  linkStrength: number;
  linkDistance: number;
}

export interface DisplaySettings {
  nodeScale: number;
  edgeScale: number;
  arrows: boolean;
  labelThreshold: number;
}

export function graphRendererFallbackWarning(
  status: Pick<GraphRendererStatus, "fallbackReason">,
): { readonly label: string; readonly title: string } | null {
  if (!status.fallbackReason) return null;
  return {
    label: "已切换兼容渲染",
    title: "系统已自动切换到稳定渲染模式，图谱功能不受影响。",
  };
}

export function rendererPreferenceForRuntime(
  requested: GraphRendererPreference | undefined,
): GraphRendererPreference {
  if (typeof window === "undefined") return "auto";
  const benchmark = new URLSearchParams(window.location.search).get("benchmark") === "graph";
  return benchmark ? requested ?? "auto" : "auto";
}

export const FORCE_PRESETS: Readonly<Record<LayoutPreset, ForceSettings>> = {
  balanced: {
    centerStrength: 0.14,
    repulsion: -115,
    linkStrength: 0.32,
    linkDistance: 76,
  },
  compact: {
    centerStrength: 0.24,
    repulsion: -72,
    linkStrength: 0.56,
    linkDistance: 48,
  },
  spread: {
    centerStrength: 0.08,
    repulsion: -190,
    linkStrength: 0.22,
    linkDistance: 112,
  },
};

export const DEFAULT_DISPLAY: DisplaySettings = {
  nodeScale: 1,
  edgeScale: 1,
  arrows: true,
  labelThreshold: 58,
};

export type GraphElementStates = Record<string, string[]>;

export type ForceLayoutRuntime = {
  id?: string;
  setFixedPosition?: (id: string, position: [number, number] | null) => void;
  instance?: ForceLayoutRuntime;
};
