import type { GraphEdge, GraphNode } from "../types/graph";

export interface LocalForceFrame {
  readonly epoch: string;
  readonly sequence: number;
  readonly nodeIndices: Uint32Array;
  readonly positions: Float32Array;
  readonly alpha: number;
  readonly computeMs: number;
  readonly activeCount: number;
  readonly roundTripMs: number;
  readonly targetNodeIndex?: number;
}

export interface LocalForcePolicy {
  readonly depth: number;
  readonly limit: number;
}

export interface LocalForceDragEnd {
  readonly x: number;
  readonly y: number;
  readonly pinned: boolean;
}

export function localForcePolicy(nodeCount: number): LocalForcePolicy {
  if (nodeCount <= 300) return Object.freeze({ depth: 2, limit: 64 });
  if (nodeCount <= 1_000) return Object.freeze({ depth: 2, limit: 48 });
  return Object.freeze({ depth: 1, limit: 24 });
}

interface WorkerLike {
  onmessage: ((event: MessageEvent) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
  terminate(): void;
}

export interface LocalForceControllerOptions {
  readonly onFrame: (frame: LocalForceFrame) => void;
  readonly workerFactory?: () => WorkerLike;
  readonly now?: () => number;
  readonly wallClock?: () => number;
  /** Frames older than this cannot represent the current pointer position. */
  readonly maxFrameAgeMs?: number;
}

export class LocalForceController {
  private readonly worker: WorkerLike;
  private readonly now: () => number;
  private readonly wallClock: () => number;
  private readonly maxFrameAgeMs: number;
  private readonly idToIndex = new Map<string, number>();
  private indexToId: readonly string[] = [];
  private epoch = "";
  private sequence = 0;
  private lastMoveAt = -Infinity;

  constructor(private readonly options: LocalForceControllerOptions) {
    this.now = options.now ?? (() => performance.now());
    this.wallClock = options.wallClock ?? (() => Date.now());
    this.maxFrameAgeMs = options.maxFrameAgeMs ?? 120;
    this.worker =
      options.workerFactory?.() ??
      new Worker(new URL("./graphLayout.worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent) => {
      const message = event.data as LocalForceFrame & {
        readonly type: string;
        readonly emittedAtEpochMs?: number;
      };
      if (message.type !== "frame" || message.epoch !== this.epoch) return;
      if (message.sequence < this.sequence) return;
      const frameAgeMs = message.emittedAtEpochMs === undefined
        ? 0
        : Math.max(0, this.wallClock() - message.emittedAtEpochMs);
      // Canvas work can temporarily delay the window's Worker message queue.
      // Applying an old relaxation frame after the pointer has moved causes a
      // visible snap-back and queues more renderer work; retain only fresh
      // frames and let the Worker's next tick supply the latest coordinates.
      if (message.emittedAtEpochMs !== undefined && frameAgeMs > this.maxFrameAgeMs) return;
      this.options.onFrame({
        ...message,
        // Cooling emits several frames after the final pointer message. Using
        // the frame's own emit time avoids reporting that intentional cooling
        // window as transport latency.
        roundTripMs:
          message.computeMs +
          frameAgeMs,
      });
    };
  }

  initialize(
    epoch: string,
    nodes: readonly GraphNode[],
    edges: readonly GraphEdge[],
    positionsById: ReadonlyMap<string, { readonly x: number; readonly y: number }>,
    pinnedNodeIds: ReadonlySet<string> = new Set(),
  ): void {
    this.epoch = epoch;
    // Keep a monotonic command barrier when the same graph identity receives
    // a replacement topology. Frames already queued by the previous init can
    // then be rejected by sequence instead of briefly snapping old positions
    // into the new scene.
    this.sequence += 1;
    this.lastMoveAt = -Infinity;
    this.idToIndex.clear();
    this.indexToId = nodes.map((node) => node.id);
    const positions = new Float32Array(nodes.length * 2);
    nodes.forEach((node, index) => {
      this.idToIndex.set(node.id, index);
      const point = positionsById.get(node.id) ?? { x: 0, y: 0 };
      positions[index * 2] = point.x;
      positions[index * 2 + 1] = point.y;
    });
    const pairValues: number[] = [];
    for (const edge of edges) {
      const source = this.idToIndex.get(edge.source);
      const target = this.idToIndex.get(edge.target);
      if (source === undefined || target === undefined || source === target) continue;
      pairValues.push(source, target);
    }
    const edgePairs = new Uint32Array(pairValues);
    const pinnedNodeIndices = new Uint32Array(
      [...pinnedNodeIds]
        .map((id) => this.idToIndex.get(id))
        .filter((index): index is number => index !== undefined),
    );
    this.worker.postMessage(
      {
        type: "init",
        epoch,
        nodeIds: nodes.map((node) => node.id),
        positions,
        edgePairs,
        pinnedNodeIndices,
      },
      [positions.buffer, edgePairs.buffer, pinnedNodeIndices.buffer],
    );
  }

  nodeId(index: number): string | undefined {
    return this.indexToId[index];
  }

  dragStart(nodeId: string, x: number, y: number, nodeCount: number): boolean {
    const nodeIndex = this.idToIndex.get(nodeId);
    if (nodeIndex === undefined) return false;
    this.sequence += 1;
    const policy = localForcePolicy(nodeCount);
    this.worker.postMessage({
      type: "drag-start",
      epoch: this.epoch,
      sequence: this.sequence,
      nodeIndex,
      x,
      y,
      depth: policy.depth,
      limit: policy.limit,
    });
    return true;
  }

  dragMove(x: number, y: number): boolean {
    const at = this.now();
    if (at - this.lastMoveAt < 1000 / 15) return false;
    this.lastMoveAt = at;
    this.sequence += 1;
    this.worker.postMessage({
      type: "drag-move",
      epoch: this.epoch,
      sequence: this.sequence,
      x,
      y,
    });
    return true;
  }

  dragEnd(finalPosition: LocalForceDragEnd): void {
    this.sequence += 1;
    this.worker.postMessage({
      type: "drag-end",
      epoch: this.epoch,
      sequence: this.sequence,
      x: finalPosition.x,
      y: finalPosition.y,
      pinned: finalPosition.pinned,
    });
  }

  destroy(): void {
    this.worker.postMessage({ type: "dispose", epoch: this.epoch });
    this.worker.terminate();
  }
}
