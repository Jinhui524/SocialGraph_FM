import type { Graph, GraphData, LayoutOptions, Point } from "@antv/g6";

export interface GraphCameraSnapshot {
  readonly schemaVersion: "socialgraph-fm.graph-camera/2";
  readonly sceneIdentity: string;
  readonly position: Point;
  readonly zoom: number;
  /** Canvas/world point that occupied the CSS viewport centre. */
  readonly worldCenter: Point;
  readonly viewportSize: readonly [number, number];
}

type LegacyGraphCameraSnapshot = Pick<GraphCameraSnapshot, "position" | "zoom">
  & Partial<Pick<GraphCameraSnapshot, "schemaVersion" | "sceneIdentity" | "worldCenter" | "viewportSize">>;

interface GraphCommandTokenStamp {
  readonly token: number;
  readonly activation: number;
  readonly sceneGeneration: number;
}

export interface GraphCameraSnapshotCacheKey {
  readonly workspace: string;
  readonly graphIdentity: string;
  readonly lens: string;
}

export class GraphCameraSnapshotCache {
  private readonly snapshots = new Map<string, GraphCameraSnapshot>();

  private key(value: GraphCameraSnapshotCacheKey): string {
    return `${value.workspace}\u0000${value.graphIdentity}\u0000${value.lens}`;
  }

  set(key: GraphCameraSnapshotCacheKey, snapshot: GraphCameraSnapshot): void {
    if (
      snapshot.schemaVersion !== "socialgraph-fm.graph-camera/2"
      || !snapshot.sceneIdentity
      || !Number.isFinite(snapshot.zoom)
      || snapshot.zoom <= 0
      || [...snapshot.position, ...snapshot.worldCenter, ...snapshot.viewportSize].some((value) => !Number.isFinite(Number(value)))
      || snapshot.viewportSize.some((value) => Number(value) <= 0)
    ) return;
    this.snapshots.set(this.key(key), Object.freeze({
      ...snapshot,
      position: Object.freeze([...snapshot.position]) as Point,
      worldCenter: Object.freeze([...snapshot.worldCenter]) as Point,
      viewportSize: Object.freeze([...snapshot.viewportSize]) as [number, number],
    }));
  }

  get(key: GraphCameraSnapshotCacheKey): GraphCameraSnapshot | undefined {
    return this.snapshots.get(this.key(key));
  }
}

export interface GraphSceneTransition {
  /** Preserve is the default for filters, view modes, and depth changes. */
  readonly layout: "preserve" | "local-relax" | "full";
  /** Camera fitting is deliberately opt-in. */
  readonly camera: "preserve" | "focus-anchor" | "fit-if-offscreen";
  readonly anchorNodeIds?: readonly string[];
}

export interface GraphSceneLease {
  readonly sceneIdentity: string;
  readonly generation: number;
}

export const PRESERVE_GRAPH_SCENE: GraphSceneTransition = Object.freeze({
  layout: "preserve",
  camera: "preserve",
});

export interface GraphMutationDiagnostics {
  readonly mutationInFlight: number;
  readonly mutationInFlightMax: number;
  readonly queuedMutations: number;
  readonly positionFramePending: boolean;
}

interface PendingGraphResize {
  readonly size: readonly [number, number];
  readonly worldCenter?: Point;
}

/**
 * Small compatibility boundary around the public G6 graph API.
 *
 * Keeping camera preservation and layout updates here prevents React visual
 * changes from accidentally recreating the renderer or resetting the view.
 */
export class GraphEngineAdapter {
  private viewportSize: readonly [number, number];
  private rendererMutationInFlight = 0;
  private rendererMutationInFlightMax = 0;
  private readonly rendererMutationQueue: Array<{
    readonly action: () => unknown | Promise<unknown>;
    readonly resolve: (value: unknown) => void;
    readonly reject: (error: unknown) => void;
  }> = [];
  private resizeApplyInFlight = false;
  private pendingResize: PendingGraphResize | null = null;
  private pendingResizeWaiters: Array<{
    resolve: () => void;
    reject: (error: unknown) => void;
  }> = [];
  private positionApplyInFlight = false;
  private pendingPositions:
    | Readonly<Record<string, { readonly x: number; readonly y: number }>>
    | null = null;
  private pendingPositionWaiters: Array<{
    resolve: () => void;
    reject: (error: unknown) => void;
  }> = [];
  private appearanceGeneration = 0;
  private visibilityGeneration = 0;
  private cameraGeneration = 0;
  private cameraGestureCancellationGeneration = 0;
  private readonly latestCameraRequestTokenByScene = new Map<string, GraphCommandTokenStamp>();
  private readonly latestExternalRestoreTokenByScene = new Map<string, GraphCommandTokenStamp>();
  private readonly latestExternalVisibilityTokenByScene = new Map<string, GraphCommandTokenStamp>();
  private sceneGeneration = 0;
  private sceneDataKey: string | null;

  constructor(
    private readonly graph: Graph,
    private readonly getViewportSize: () => readonly [number, number] = () => graph.getSize(),
    initialSceneDataKey: string | null = null,
  ) {
    this.viewportSize = [...getViewportSize()] as [number, number];
    this.sceneDataKey = initialSceneDataKey;
  }

  /**
   * Serializes the Canvas mutations used by direct dragging and interaction
   * LOD. Keeping this boundary below React guarantees that a pointer frame,
   * a Worker frame, and the one-shot LOD transition cannot mutate G6 at the
   * same time.
   */
  private enqueueRendererMutation<T>(action: () => T | Promise<T>): Promise<T> {
    if (this.graph.destroyed) return Promise.resolve(undefined as T);
    return new Promise<T>((resolve, reject) => {
      this.rendererMutationQueue.push({
        action,
        resolve: (value) => resolve(value as T),
        reject,
      });
      void this.drainRendererMutationQueue();
    });
  }

  private async drainRendererMutationQueue(): Promise<void> {
    if (this.rendererMutationInFlight > 0) return;
    const mutation = this.rendererMutationQueue.shift();
    if (!mutation) return;
    this.rendererMutationInFlight = 1;
    this.rendererMutationInFlightMax = Math.max(
      this.rendererMutationInFlightMax,
      this.rendererMutationInFlight,
    );
    try {
      mutation.resolve(await mutation.action());
    } catch (error) {
      mutation.reject(error);
    } finally {
      this.rendererMutationInFlight = 0;
      if (this.rendererMutationQueue.length > 0) {
        void this.drainRendererMutationQueue();
      }
    }
  }

  mutationDiagnostics(): GraphMutationDiagnostics {
    return Object.freeze({
      mutationInFlight: this.rendererMutationInFlight,
      mutationInFlightMax: this.rendererMutationInFlightMax,
      queuedMutations: this.rendererMutationQueue.length,
      positionFramePending: Boolean(this.pendingPositions),
    });
  }

  private currentSceneIdentity(): string {
    return this.sceneDataKey ?? "unbound-scene";
  }

  private acceptMonotonicToken(
    tokens: Map<string, GraphCommandTokenStamp>,
    sceneIdentity: string,
    requestToken: number | undefined,
    tokenScope: string,
    activation = 0,
  ): boolean {
    if (requestToken === undefined) return true;
    if (!Number.isFinite(requestToken) || !Number.isFinite(activation)) return false;
    const key = `${tokenScope}\u0000${sceneIdentity}`;
    const latest = tokens.get(key);
    if (latest) {
      if (requestToken < latest.token) return false;
      if (
        requestToken === latest.token
        && activation <= latest.activation
        && this.sceneGeneration <= latest.sceneGeneration
      ) return false;
    }
    tokens.set(key, {
      token: requestToken,
      activation,
      sceneGeneration: this.sceneGeneration,
    });
    return true;
  }

  captureCamera(): GraphCameraSnapshot {
    const viewportSize = this.getViewportSize();
    return {
      schemaVersion: "socialgraph-fm.graph-camera/2",
      sceneIdentity: this.currentSceneIdentity(),
      position: [...this.graph.getPosition()] as Point,
      zoom: this.graph.getZoom(),
      worldCenter: [
        ...this.graph.getCanvasByViewport([
          viewportSize[0] / 2,
          viewportSize[1] / 2,
        ]),
      ] as Point,
      viewportSize: [...viewportSize] as [number, number],
    };
  }

  private async applyCameraSnapshot(snapshot: LegacyGraphCameraSnapshot): Promise<void> {
    if (this.graph.destroyed) return;
    // A V2 world centre is the stable camera anchor. G6's relative translation
    // moves the camera by `-offset / zoom`, so convert the canvas-space centre
    // delta back to a viewport-space offset after restoring zoom.
    await this.graph.zoomTo(snapshot.zoom, false, [0, 0]);
    if (this.graph.destroyed) return;
    if (!snapshot.worldCenter) {
      await this.graph.translateTo(snapshot.position, false);
      return;
    }
    const viewportSize = this.getViewportSize();
    const currentWorldCenter = this.graph.getCanvasByViewport([
      viewportSize[0] / 2,
      viewportSize[1] / 2,
    ]);
    await this.graph.translateBy([
      (Number(currentWorldCenter[0]) - Number(snapshot.worldCenter[0])) * snapshot.zoom,
      (Number(currentWorldCenter[1]) - Number(snapshot.worldCenter[1])) * snapshot.zoom,
    ], false);
  }

  async restoreCamera(
    snapshot: LegacyGraphCameraSnapshot,
    expectedSceneIdentity = snapshot.sceneIdentity ?? this.currentSceneIdentity(),
    requestToken?: number,
    tokenScope = "external-restore",
    activation = 0,
  ): Promise<boolean> {
    if (
      this.graph.destroyed
      || expectedSceneIdentity !== this.currentSceneIdentity()
      || !this.acceptMonotonicToken(
        this.latestExternalRestoreTokenByScene,
        expectedSceneIdentity,
        requestToken,
        tokenScope,
        activation,
      )
    ) return false;
    const generation = ++this.cameraGeneration;
    return this.enqueueRendererMutation(async () => {
      if (
        this.graph.destroyed
        || generation !== this.cameraGeneration
        || (snapshot.sceneIdentity !== undefined && snapshot.sceneIdentity !== expectedSceneIdentity)
        || expectedSceneIdentity !== this.currentSceneIdentity()
      ) return false;
      const measuredViewport = this.getViewportSize();
      await this.performResizePreservingWorldCenter([
        Math.max(1, Math.round(Number(measuredViewport[0]))),
        Math.max(1, Math.round(Number(measuredViewport[1]))),
      ]);
      if (
        this.graph.destroyed
        || generation !== this.cameraGeneration
        || expectedSceneIdentity !== this.currentSceneIdentity()
      ) return false;
      await this.applyCameraSnapshot(snapshot);
      return !this.graph.destroyed
        && generation === this.cameraGeneration
        && expectedSceneIdentity === this.currentSceneIdentity();
    });
  }

  async replaceScene(
    data: GraphData,
    transition: GraphSceneTransition = PRESERVE_GRAPH_SCENE,
    sceneDataKey?: string,
  ): Promise<GraphSceneLease | null> {
    if (this.graph.destroyed) return null;
    const generation = ++this.sceneGeneration;
    // Scene replacement is the generation boundary for every command that
    // targets renderer elements or camera coordinates from the prior scene.
    this.cameraGeneration += 1;
    this.visibilityGeneration += 1;
    return this.enqueueRendererMutation(async () => {
      if (this.graph.destroyed || generation !== this.sceneGeneration) return null;
      const camera = this.captureCamera();
      const liveNodeIds = new Set(this.graph.getNodeData().map((node) => String(node.id)));
      const positionedNodes = (data.nodes ?? []).map((node) => {
        if (!liveNodeIds.has(String(node.id))) return node;
        try {
          const position = this.graph.getElementPosition(String(node.id));
          const x = Number(position[0]);
          const y = Number(position[1]);
          if (!Number.isFinite(x) || !Number.isFinite(y)) return node;
          return {
            ...node,
            style: { ...node.style, x, y },
          };
        } catch {
          return node;
        }
      });
      this.graph.setData({ ...data, nodes: positionedNodes });
      await this.graph.draw();
      if (transition.layout === "full" && !this.graph.destroyed) {
        await this.graph.layout();
      }
      await this.applyCameraSnapshot(camera);
      if (sceneDataKey !== undefined && !this.graph.destroyed && generation === this.sceneGeneration) {
        this.sceneDataKey = sceneDataKey;
      }
      if (this.graph.destroyed || generation !== this.sceneGeneration) return null;
      return Object.freeze({
        sceneIdentity: this.currentSceneIdentity(),
        generation,
      });
    });
  }

  isSceneLeaseCurrent(lease: GraphSceneLease): boolean {
    return !this.graph.destroyed
      && lease.generation === this.sceneGeneration
      && lease.sceneIdentity === this.currentSceneIdentity();
  }

  captureSceneLease(): GraphSceneLease | null {
    if (this.graph.destroyed) return null;
    return Object.freeze({
      sceneIdentity: this.currentSceneIdentity(),
      generation: this.sceneGeneration,
    });
  }

  async runSceneTransaction(
    lease: GraphSceneLease,
    action: (isCurrent: () => boolean) => Promise<void>,
  ): Promise<boolean> {
    if (!this.isSceneLeaseCurrent(lease)) return false;
    return this.enqueueRendererMutation(async () => {
      const isCurrent = () => this.isSceneLeaseCurrent(lease);
      if (!isCurrent()) return false;
      await action(isCurrent);
      return isCurrent();
    });
  }

  applyPositions(
    positions: Readonly<Record<string, { readonly x: number; readonly y: number }>>,
  ): Promise<void> {
    if (this.graph.destroyed || Object.keys(positions).length === 0) {
      return Promise.resolve();
    }
    return new Promise<void>((resolve, reject) => {
      // Pointer and Worker updates share one renderer mutation lane. While a
      // Canvas mutation is in flight only the newest not-yet-applied frame is
      // retained; every superseded caller settles with that newest frame.
      this.pendingPositions = {
        ...(this.pendingPositions ?? {}),
        ...positions,
      };
      this.pendingPositionWaiters.push({ resolve, reject });
      void this.drainPositionQueue();
    });
  }

  private async drainPositionQueue(): Promise<void> {
    if (this.positionApplyInFlight || this.graph.destroyed || !this.pendingPositions) return;
    let positions = this.pendingPositions;
    const waiters = this.pendingPositionWaiters;
    this.pendingPositions = null;
    this.pendingPositionWaiters = [];
    this.positionApplyInFlight = true;
    try {
      // G6's translate stage updates only the requested element transforms and
      // their incident geometry. updateNodeData + draw repaints the whole scene
      // and exceeded the medium/large frame budget during worker relaxation.
      await this.enqueueRendererMutation(() => {
        // A renderer state transition may have occupied the mutation lane after
        // this frame was queued. Resolve that back-pressure with the newest
        // pointer/Worker frame instead of replaying stale coordinates first.
        if (this.pendingPositions) {
          positions = { ...positions, ...this.pendingPositions };
          this.pendingPositions = null;
          waiters.push(...this.pendingPositionWaiters);
          this.pendingPositionWaiters = [];
        }
        const translated = Object.fromEntries(
          Object.entries(positions).map(([id, point]) => [id, [point.x, point.y] as Point]),
        );
        return this.graph.translateElementTo(translated, false);
      });
      for (const waiter of waiters) waiter.resolve();
    } catch (error) {
      for (const waiter of waiters) waiter.reject(error);
    } finally {
      this.positionApplyInFlight = false;
      if (this.graph.destroyed) {
        const error = new Error("GRAPH_ENGINE_DESTROYED");
        for (const waiter of this.pendingPositionWaiters) waiter.reject(error);
        this.pendingPositionWaiters = [];
        this.pendingPositions = null;
      } else if (this.pendingPositions) {
        void this.drainPositionQueue();
      }
    }
  }

  /** Applies a single batched state transition through the drag mutation lane. */
  async applyElementStates(
    states: Readonly<Record<string, readonly string[]>>,
  ): Promise<void> {
    if (this.graph.destroyed || Object.keys(states).length === 0) return;
    const mutableStates = Object.fromEntries(
      Object.entries(states).map(([id, values]) => [id, [...values]]),
    );
    await this.enqueueRendererMutation(() => {
      if (this.graph.destroyed) return undefined;
      return this.graph.setElementState(mutableStates, false);
    });
  }

  async applyVisibility(
    changes: Readonly<Record<string, "visible" | "hidden">>,
    expectedSceneIdentity = this.currentSceneIdentity(),
  ): Promise<boolean> {
    if (this.graph.destroyed || Object.keys(changes).length === 0) return false;
    const generation = this.sceneGeneration;
    return this.enqueueRendererMutation(async () => {
      if (
        this.graph.destroyed
        || generation !== this.sceneGeneration
        || expectedSceneIdentity !== this.currentSceneIdentity()
      ) return false;
      const liveIds = new Set([
        ...this.graph.getNodeData().map((node) => String(node.id)),
        ...this.graph.getEdgeData().map((edge) => String(edge.id)),
      ]);
      const liveChanges = Object.fromEntries(
        Object.entries(changes).filter(([id]) => liveIds.has(id)),
      );
      if (Object.keys(liveChanges).length === 0) return false;
      await this.graph.setElementVisibility(liveChanges, false);
      return !this.graph.destroyed
        && generation === this.sceneGeneration
        && expectedSceneIdentity === this.currentSceneIdentity();
    });
  }

  async ensureVisible(
    nodeIds: readonly string[],
    expectedSceneDataKey?: string,
    requestToken?: number,
    tokenScope = "internal",
    activation = 0,
  ): Promise<boolean> {
    const sceneIdentity = expectedSceneDataKey ?? this.currentSceneIdentity();
    if (
      this.graph.destroyed
      || nodeIds.length === 0
      || !this.acceptMonotonicToken(
        this.latestExternalVisibilityTokenByScene,
        sceneIdentity,
        requestToken,
        tokenScope,
        activation,
      )
    ) return false;
    const generation = ++this.visibilityGeneration;
    return this.enqueueRendererMutation(async () => {
      if (
        this.graph.destroyed
        || generation !== this.visibilityGeneration
        || (expectedSceneDataKey !== undefined && expectedSceneDataKey !== this.sceneDataKey)
      ) return false;
      const liveIds = new Set(this.graph.getNodeData().map((node) => String(node.id)));
      if (nodeIds.some((id) => !liveIds.has(id))) return false;
      for (const id of nodeIds) {
        const position = this.graph.getElementPosition(id);
        if (!Number.isFinite(Number(position[0])) || !Number.isFinite(Number(position[1]))) {
          return false;
        }
      }
      if (
        generation !== this.visibilityGeneration
        || (expectedSceneDataKey !== undefined && expectedSceneDataKey !== this.sceneDataKey)
      ) return false;
      await this.graph.setElementVisibility(
        Object.fromEntries(nodeIds.map((id) => [id, "visible" as const])),
        false,
      );
      return generation === this.visibilityGeneration
        && (expectedSceneDataKey === undefined || expectedSceneDataKey === this.sceneDataKey);
    });
  }

  async runCameraForScene(
    expectedSceneDataKey: string,
    requestToken: number,
    action: () => Promise<void>,
    tokenScope = "internal",
    activation = 0,
  ): Promise<boolean> {
    if (this.graph.destroyed) return false;
    if (!this.acceptMonotonicToken(
      this.latestCameraRequestTokenByScene,
      expectedSceneDataKey,
      requestToken,
      tokenScope,
      activation,
    )) return false;
    const generation = ++this.cameraGeneration;
    const gestureCancellationGeneration = this.cameraGestureCancellationGeneration;
    return this.enqueueRendererMutation(async () => {
      if (
        this.graph.destroyed
        || generation !== this.cameraGeneration
        || expectedSceneDataKey !== this.sceneDataKey
      ) return false;
      const camera = this.captureCamera();
      let actionError: unknown;
      try {
        await action();
      } catch (error) {
        actionError = error;
      }
      const current = !this.graph.destroyed
        && generation === this.cameraGeneration
        && expectedSceneDataKey === this.sceneDataKey;
      if (!current) {
        if (
          !this.graph.destroyed
          && gestureCancellationGeneration === this.cameraGestureCancellationGeneration
        ) await this.applyCameraSnapshot(camera);
        return false;
      }
      if (actionError !== undefined) throw actionError;
      return true;
    });
  }

  /**
   * Stops an animated camera command when a direct canvas gesture begins.
   * The zero-distance public transform synchronously cancels G6's landmark
   * animation; the generation boundary prevents the stale command from
   * restoring its pre-focus camera over the user's subsequent pan.
   */
  cancelCameraForGesture(expectedSceneIdentity = this.currentSceneIdentity()): boolean {
    if (
      this.graph.destroyed
      || expectedSceneIdentity !== this.currentSceneIdentity()
    ) return false;
    this.cameraGeneration += 1;
    this.visibilityGeneration += 1;
    this.cameraGestureCancellationGeneration += 1;
    try {
      void this.graph.translateBy([0, 0], false).catch(() => undefined);
    } catch {
      // A graph destroyed between the guard and transform is already cancelled.
    }
    return true;
  }

  async setAppearance(
    apply: (isCurrent: () => boolean) => Promise<void>,
    expectedSceneIdentity = this.currentSceneIdentity(),
  ): Promise<boolean> {
    if (this.graph.destroyed) return false;
    const generation = ++this.appearanceGeneration;
    const sceneGeneration = this.sceneGeneration;
    return this.enqueueRendererMutation(async () => {
      const isCurrent = () => !this.graph.destroyed
        && generation === this.appearanceGeneration
        && sceneGeneration === this.sceneGeneration
        && expectedSceneIdentity === this.currentSceneIdentity();
      if (!isCurrent()) return false;
      const camera = this.captureCamera();
      await apply(isCurrent);
      if (!isCurrent()) return false;
      await this.applyCameraSnapshot(camera);
      return isCurrent();
    });
  }

  async setForces(layout: LayoutOptions): Promise<void> {
    if (this.graph.destroyed) return;
    await this.enqueueRendererMutation(async () => {
      if (this.graph.destroyed) return;
      const camera = this.captureCamera();
      this.graph.stopLayout();
      this.graph.setLayout(layout);
      await this.graph.layout();
      await this.applyCameraSnapshot(camera);
    });
  }

  private async performResizePreservingWorldCenter(
    nextSize: readonly [number, number],
    preservedWorldCenter?: Point,
  ): Promise<void> {
    if (this.graph.destroyed) return;
    // The DOM may settle between adapter construction and the first observer
    // delivery. G6's live canvas size is the authoritative pre-resize
    // coordinate system; the cached size is only a defensive fallback.
    const graphSize = this.graph.getSize();
    const previousSize: readonly [number, number] = (
      Number.isFinite(Number(graphSize[0]))
      && Number.isFinite(Number(graphSize[1]))
      && Number(graphSize[0]) > 0
      && Number(graphSize[1]) > 0
    )
      ? [Math.round(Number(graphSize[0])), Math.round(Number(graphSize[1]))]
      : this.viewportSize;
    const sizeChanged = !(
      previousSize[0] === nextSize[0]
      && previousSize[1] === nextSize[1]
    );
    if (!sizeChanged && !preservedWorldCenter) return;
    const previousWorldCenter = preservedWorldCenter
      ?? this.graph.getCanvasByViewport([
        previousSize[0] / 2,
        previousSize[1] / 2,
      ]);
    if (sizeChanged) {
      this.graph.resize(nextSize[0], nextSize[1]);
      if (this.graph.destroyed) return;
    }
    this.viewportSize = [...nextSize] as [number, number];
    const renderedCenter = this.graph.getViewportByCanvas(previousWorldCenter);
    await this.graph.translateBy(
      [
        nextSize[0] / 2 - Number(renderedCenter[0]),
        nextSize[1] / 2 - Number(renderedCenter[1]),
      ],
      false,
    );
  }

  resizePreservingWorldCenter(
    width?: number,
    height?: number,
    preservedWorldCenter?: Point,
  ): Promise<void> {
    if (this.graph.destroyed) return Promise.resolve();
    const measured = this.getViewportSize();
    const nextWidth = Number.isFinite(width) ? Math.max(1, Math.round(width!)) : measured[0];
    const nextHeight = Number.isFinite(height) ? Math.max(1, Math.round(height!)) : measured[1];
    return new Promise<void>((resolve, reject) => {
      // ResizeObserver and fullscreen changes can produce several dimensions
      // in one frame. Retain only the newest pending dimensions and keep one
      // G6 mutation in flight; all callers settle after that newest size lands.
      this.pendingResize = {
        size: [nextWidth, nextHeight],
        ...(preservedWorldCenter
          ? { worldCenter: [...preservedWorldCenter] as Point }
          : {}),
      };
      this.pendingResizeWaiters.push({ resolve, reject });
      void this.drainResizeQueue();
    });
  }

  private async drainResizeQueue(): Promise<void> {
    if (this.resizeApplyInFlight || !this.pendingResize) return;
    if (this.graph.destroyed) {
      const waiters = this.pendingResizeWaiters;
      this.pendingResizeWaiters = [];
      this.pendingResize = null;
      for (const waiter of waiters) waiter.resolve();
      return;
    }
    const nextResize = this.pendingResize;
    this.pendingResize = null;
    this.resizeApplyInFlight = true;
    try {
      await this.enqueueRendererMutation(
        () => this.performResizePreservingWorldCenter(
          nextResize.size,
          nextResize.worldCenter,
        ),
      );
    } catch (error) {
      const waiters = this.pendingResizeWaiters;
      this.pendingResizeWaiters = [];
      this.pendingResize = null;
      for (const waiter of waiters) waiter.reject(error);
      return;
    } finally {
      this.resizeApplyInFlight = false;
    }
    if (this.pendingResize) {
      void this.drainResizeQueue();
      return;
    }
    const waiters = this.pendingResizeWaiters;
    this.pendingResizeWaiters = [];
    for (const waiter of waiters) waiter.resolve();
  }

  /** Backwards-compatible alias for callers outside the workbench. */
  async resize(): Promise<void> {
    await this.resizePreservingWorldCenter();
  }
}
