export interface GraphCanvasPerformanceProfile {
  readonly initialIterations: number;
  readonly alphaMin: number;
  readonly alphaDecay: number;
  readonly velocityDecay: number;
  readonly collision: boolean;
  readonly hover: boolean;
  readonly minimap: boolean;
  readonly minimapDelay: number;
  readonly directDrag: boolean;
  readonly dragAlphaTarget: number;
  readonly releaseAlpha: number;
}

export interface GraphCanvasInteractionLodConfig {
  readonly enabled: boolean;
  readonly debounce: number;
  readonly shapes: {
    readonly node: readonly string[];
    readonly edge: readonly string[];
  };
}

/**
 * G6's viewport optimizer toggles child DisplayObject visibility directly.
 * That keeps the Canvas gesture path cheap: node key shapes stay visible,
 * while ordinary labels and edges are restored once after the gesture.
 */
export function graphCanvasInteractionLodConfig(
  nodeCount: number,
): GraphCanvasInteractionLodConfig {
  return Object.freeze({
    enabled: nodeCount > 300,
    debounce: nodeCount > 300 ? 250 : 160,
    shapes: Object.freeze({
      node: Object.freeze(["key"]),
      edge: Object.freeze([]),
    }),
  });
}

/**
 * Bounded force budgets keep Canvas interactive instead of paying G6's
 * default 300 synchronous d3-force ticks for every initial render.
 */
export function graphCanvasPerformanceProfile(
  nodeCount: number,
): GraphCanvasPerformanceProfile {
  if (nodeCount <= 50) {
    return {
      initialIterations: 96,
      alphaMin: 0.003,
      alphaDecay: 0.075,
      velocityDecay: 0.44,
      collision: true,
      hover: true,
      minimap: true,
      minimapDelay: 1_100,
      directDrag: false,
      dragAlphaTarget: 0.065,
      releaseAlpha: 0.12,
    };
  }
  if (nodeCount <= 180) {
    return {
      initialIterations: 96,
      alphaMin: 0.003,
      alphaDecay: 0.075,
      velocityDecay: 0.46,
      collision: true,
      hover: true,
      minimap: true,
      minimapDelay: 1_400,
      directDrag: false,
      dragAlphaTarget: 0.065,
      releaseAlpha: 0.12,
    };
  }
  if (nodeCount <= 300) {
    return {
      initialIterations: 80,
      alphaMin: 0.003,
      alphaDecay: 0.085,
      velocityDecay: 0.48,
      collision: true,
      hover: false,
      minimap: nodeCount <= 300,
      minimapDelay: 1_800,
      directDrag: false,
      dragAlphaTarget: 0.045,
      releaseAlpha: 0.1,
    };
  }
  if (nodeCount <= 1_000) {
    return {
      initialIterations: 64,
      alphaMin: 0.0035,
      alphaDecay: 0.095,
      velocityDecay: 0.5,
      collision: true,
      hover: false,
      minimap: false,
      minimapDelay: 2_200,
      directDrag: true,
      dragAlphaTarget: 0.025,
      releaseAlpha: 0.08,
    };
  }
  return {
    initialIterations: 48,
    alphaMin: 0.004,
    alphaDecay: 0.11,
    velocityDecay: 0.54,
    collision: false,
    hover: false,
    minimap: false,
    minimapDelay: 2_600,
    directDrag: true,
    dragAlphaTarget: 0.015,
    releaseAlpha: 0.055,
  };
}
