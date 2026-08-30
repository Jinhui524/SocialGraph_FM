import type { Graph, ViewportAnimationEffectTiming } from "@antv/g6";

export interface GraphCameraFitResult {
  readonly fitted: boolean;
  readonly zoom: number;
  readonly retries: number;
  readonly clipped: boolean;
}

export interface GraphCameraControllerOptions {
  readonly padding?: number;
  readonly minZoom?: number;
  readonly maxZoom?: number;
  readonly maxValidationRetries?: number;
  readonly waitForStableFrames?: (count: number) => Promise<void>;
  readonly getViewportSize?: () => readonly [number, number];
}

export interface GraphCameraFocusOptions {
  /** Node that must land at the visual centre of the CSS viewport. */
  readonly anchorElementId: string;
  /** Keep focused neighbourhoods readable without jumping to an extreme zoom. */
  readonly minZoom?: number;
  readonly maxZoom?: number;
  readonly animation?: false | ViewportAnimationEffectTiming;
}

type GraphCameraApi = Pick<
  Graph,
  | "destroyed"
  | "getElementPosition"
  | "getElementRenderBounds"
  | "getSize"
  | "getViewportByCanvas"
  | "translateBy"
  | "zoomTo"
>;

interface CombinedBounds {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

const NODE_WORLD_MARGIN = 32;

function waitForAnimationFrames(count: number): Promise<void> {
  return new Promise((resolve) => {
    const next = (remaining: number) => {
      if (remaining <= 0) {
        resolve();
        return;
      }
      requestAnimationFrame(() => next(remaining - 1));
    };
    next(count);
  });
}

export function combineElementBounds(
  graph: GraphCameraApi,
  elementIds: readonly string[],
): CombinedBounds | null {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const id of elementIds) {
    try {
      const bounds = graph.getElementRenderBounds(id);
      minX = Math.min(minX, Number(bounds.min[0]));
      minY = Math.min(minY, Number(bounds.min[1]));
      maxX = Math.max(maxX, Number(bounds.max[0]));
      maxY = Math.max(maxY, Number(bounds.max[1]));
    } catch {
      // Elements can disappear between a scene update and the stable frame.
    }
  }
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) return null;
  return { minX, minY, maxX, maxY };
}

/**
 * G6's shape render bounds may be reported in a renderer-layer coordinate
 * space while a multi-layer canvas is settling. Node model positions are
 * stable world coordinates, so use them as the authoritative fit extent and
 * reserve enough world-space margin for the largest supported node glyph.
 */
export function combineElementPositionBounds(
  graph: GraphCameraApi,
  elementIds: readonly string[],
): CombinedBounds | null {
  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const id of elementIds) {
    try {
      const [x, y] = graph.getElementPosition(id);
      if (!Number.isFinite(Number(x)) || !Number.isFinite(Number(y))) continue;
      minX = Math.min(minX, Number(x) - NODE_WORLD_MARGIN);
      minY = Math.min(minY, Number(y) - NODE_WORLD_MARGIN);
      maxX = Math.max(maxX, Number(x) + NODE_WORLD_MARGIN);
      maxY = Math.max(maxY, Number(y) + NODE_WORLD_MARGIN);
    } catch {
      // Elements can disappear between a scene update and the stable frame.
    }
  }
  if (![minX, minY, maxX, maxY].every(Number.isFinite)) return null;
  return { minX, minY, maxX, maxY };
}

export class GraphCameraController {
  private readonly padding: number;
  private readonly minZoom: number;
  private readonly maxZoom: number;
  private readonly maxValidationRetries: number;
  private readonly waitForStableFrames: (count: number) => Promise<void>;
  private readonly getViewportSize: () => readonly [number, number];

  constructor(
    private readonly graph: GraphCameraApi,
    options: GraphCameraControllerOptions = {},
  ) {
    this.padding = options.padding ?? 48;
    // Large deterministic layouts can be much taller than the compact right
    // pane. A 0.12 floor made a complete 2,708-node Cora view physically
    // impossible to fit inside a 280px canvas. Keep a conservative lower
    // bound so the explicit recover action can always honour its safe area.
    this.minZoom = options.minZoom ?? 0.01;
    this.maxZoom = options.maxZoom ?? 4;
    this.maxValidationRetries = options.maxValidationRetries ?? 3;
    this.waitForStableFrames = options.waitForStableFrames ?? waitForAnimationFrames;
    this.getViewportSize = options.getViewportSize ?? (() => this.graph.getSize());
  }

  private calculateZoom(bounds: CombinedBounds, scale = 1): number {
    const [width, height] = this.getViewportSize();
    const contentWidth = Math.max(1, bounds.maxX - bounds.minX);
    const contentHeight = Math.max(1, bounds.maxY - bounds.minY);
    const availableWidth = Math.max(1, width - this.padding * 2);
    const availableHeight = Math.max(1, height - this.padding * 2);
    return Math.max(
      this.minZoom,
      Math.min(
        this.maxZoom,
        Math.min(availableWidth / contentWidth, availableHeight / contentHeight) * scale,
      ),
    );
  }

  private async applyAbsoluteTransform(
    bounds: CombinedBounds,
    zoom: number,
    animation: false | ViewportAnimationEffectTiming,
  ): Promise<void> {
    const [width, height] = this.getViewportSize();
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    // G6's backing canvas can be scaled by devicePixelRatio while viewport
    // transforms use CSS pixels. `Graph.getSize()`/`focusElement()` therefore
    // centred the graph on the backing-store midpoint (the visual bottom-right
    // at DPR=2). Zoom around the CSS viewport centre, then apply a relative
    // viewport-pixel delta derived from G6's public coordinate conversion.
    await this.graph.zoomTo(zoom, animation, [width / 2, height / 2]);
    if (this.graph.destroyed) return;
    const [renderedCenterX, renderedCenterY] = this.graph.getViewportByCanvas([
      centerX,
      centerY,
    ]);
    await this.graph.translateBy(
      [width / 2 - renderedCenterX, height / 2 - renderedCenterY],
      animation,
    );
  }

  private isClipped(bounds: CombinedBounds): boolean {
    const [width, height] = this.getViewportSize();
    const [minX, minY] = this.graph.getViewportByCanvas([bounds.minX, bounds.minY]);
    const [maxX, maxY] = this.graph.getViewportByCanvas([bounds.maxX, bounds.maxY]);
    const tolerance = 1;
    return (
      Math.min(minX, maxX) < this.padding - tolerance ||
      Math.min(minY, maxY) < this.padding - tolerance ||
      Math.max(minX, maxX) > width - this.padding + tolerance ||
      Math.max(minY, maxY) > height - this.padding + tolerance
    );
  }

  async fit(
    elementIds: readonly string[],
    animation: false | ViewportAnimationEffectTiming = false,
  ): Promise<GraphCameraFitResult> {
    if (this.graph.destroyed || elementIds.length === 0) {
      return { fitted: false, zoom: 1, retries: 0, clipped: false };
    }
    await this.waitForStableFrames(2);
    if (this.graph.destroyed) {
      return { fitted: false, zoom: 1, retries: 0, clipped: false };
    }
    const bounds =
      combineElementPositionBounds(this.graph, elementIds) ??
      combineElementBounds(this.graph, elementIds);
    if (!bounds) return { fitted: false, zoom: 1, retries: 0, clipped: false };

    let zoom = this.calculateZoom(bounds);
    let retries = 0;
    await this.applyAbsoluteTransform(bounds, zoom, animation);
    let clipped = this.isClipped(bounds);
    while (clipped && retries < this.maxValidationRetries && !this.graph.destroyed) {
      retries += 1;
      zoom = Math.max(this.minZoom, zoom * 0.9);
      await this.applyAbsoluteTransform(bounds, zoom, false);
      clipped = this.isClipped(bounds);
    }
    return { fitted: true, zoom, retries, clipped };
  }

  /**
   * Centres one anchor while deriving a comfortable zoom from its visible
   * neighbourhood. This intentionally avoids G6 `focusElement()`: that API
   * may centre against a DPR-scaled backing store instead of the CSS viewport.
   */
  async focus(
    elementIds: readonly string[],
    options: GraphCameraFocusOptions,
  ): Promise<GraphCameraFitResult> {
    if (this.graph.destroyed || elementIds.length === 0) {
      return { fitted: false, zoom: 1, retries: 0, clipped: false };
    }
    await this.waitForStableFrames(1);
    if (this.graph.destroyed) {
      return { fitted: false, zoom: 1, retries: 0, clipped: false };
    }
    const neighbourhoodBounds =
      combineElementPositionBounds(this.graph, elementIds) ??
      combineElementBounds(this.graph, elementIds);
    const anchorBounds =
      combineElementPositionBounds(this.graph, [options.anchorElementId]) ??
      combineElementBounds(this.graph, [options.anchorElementId]);
    if (!neighbourhoodBounds || !anchorBounds) {
      return { fitted: false, zoom: 1, retries: 0, clipped: false };
    }

    const focusMinZoom = Math.max(this.minZoom, options.minZoom ?? 0.72);
    const focusMaxZoom = Math.min(this.maxZoom, options.maxZoom ?? 1.35);
    const zoom = Math.max(
      focusMinZoom,
      Math.min(focusMaxZoom, this.calculateZoom(neighbourhoodBounds, 0.9)),
    );
    await this.applyAbsoluteTransform(anchorBounds, zoom, options.animation ?? false);
    return {
      fitted: true,
      zoom,
      retries: 0,
      clipped: this.isClipped(neighbourhoodBounds),
    };
  }

  /** True when at least one model position intersects the CSS viewport. */
  hasVisibleElement(elementIds: readonly string[], margin = 24): boolean {
    if (this.graph.destroyed) return false;
    const [width, height] = this.getViewportSize();
    for (const id of elementIds) {
      try {
        const position = this.graph.getElementPosition(id);
        const point = this.graph.getViewportByCanvas(position);
        const x = Number(point[0]);
        const y = Number(point[1]);
        if (
          Number.isFinite(x) &&
          Number.isFinite(y) &&
          x >= -margin &&
          y >= -margin &&
          x <= width + margin &&
          y <= height + margin
        ) {
          return true;
        }
      } catch {
        // Scene changes can remove an element between frames.
      }
    }
    return false;
  }
}
