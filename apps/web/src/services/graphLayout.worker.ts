/// <reference lib="webworker" />

type InitMessage = {
  type: "init";
  epoch: string;
  nodeIds: string[];
  positions: Float32Array;
  edgePairs: Uint32Array;
  pinnedNodeIndices: Uint32Array;
};
type DragStartMessage = {
  type: "drag-start";
  epoch: string;
  sequence: number;
  nodeIndex: number;
  x: number;
  y: number;
  depth: number;
  limit: number;
};
type DragMoveMessage = {
  type: "drag-move";
  epoch: string;
  sequence: number;
  x: number;
  y: number;
};
type DragEndMessage = {
  type: "drag-end";
  epoch: string;
  sequence: number;
  x: number;
  y: number;
  pinned: boolean;
};
type DisposeMessage = { type: "dispose"; epoch: string };
type Incoming = InitMessage | DragStartMessage | DragMoveMessage | DragEndMessage | DisposeMessage;

let epoch = "";
let positions: Float32Array<ArrayBufferLike> = new Float32Array();
let velocities: Float32Array<ArrayBufferLike> = new Float32Array();
let edges: Uint32Array<ArrayBufferLike> = new Uint32Array();
let neighbours: number[][] = [];
let active: number[] = [];
let targetIndex = -1;
let targetX = 0;
let targetY = 0;
let sequence = 0;
let alpha = 0;
let releaseDeadline = 0;
let targetPinned = false;
let pinned = new Set<number>();
let timer: ReturnType<typeof setTimeout> | undefined;

const workerScope = self as unknown as DedicatedWorkerGlobalScope;

function schedule(): void {
  if (timer !== undefined) return;
  timer = setTimeout(tick, 16);
}

function buildNeighbours(nodeCount: number): void {
  neighbours = Array.from({ length: nodeCount }, () => [] as number[]);
  for (let index = 0; index < edges.length; index += 2) {
    const source = edges[index];
    const target = edges[index + 1];
    if (source >= nodeCount || target >= nodeCount || source === target) continue;
    neighbours[source].push(target);
    neighbours[target].push(source);
  }
}

function localNodes(start: number, depth: number, limit: number): number[] {
  const result = [start];
  const seen = new Set(result);
  let frontier = [start];
  for (let step = 0; step < depth && frontier.length && result.length < limit; step += 1) {
    const next: number[] = [];
    for (const node of frontier) {
      for (const neighbour of neighbours[node] ?? []) {
        if (seen.has(neighbour)) continue;
        seen.add(neighbour);
        result.push(neighbour);
        next.push(neighbour);
        if (result.length >= limit) return result;
      }
    }
    frontier = next;
  }
  return result;
}

function emit(computeMs: number): void {
  const indices = new Uint32Array(active);
  const values = new Float32Array(active.length * 2);
  active.forEach((nodeIndex, index) => {
    values[index * 2] = positions[nodeIndex * 2];
    values[index * 2 + 1] = positions[nodeIndex * 2 + 1];
  });
  workerScope.postMessage(
    {
      type: "frame",
      epoch,
      sequence,
      nodeIndices: indices,
      positions: values,
      alpha,
      computeMs,
      activeCount: active.length,
      targetNodeIndex: targetIndex,
      // Dedicated workers and the window do not reliably share a
      // performance.timeOrigin. Epoch time is comparable across both realms.
      emittedAtEpochMs: Date.now(),
    },
    [indices.buffer, values.buffer],
  );
}

function tick(): void {
  timer = undefined;
  if (!active.length || targetIndex < 0) return;
  const startedAt = performance.now();
  const repulsion = 46 * alpha;
  const spring = 0.055 * alpha;
  const damping = 0.74;

  // The dragged node is authoritative and follows the pointer exactly.
  positions[targetIndex * 2] = targetX;
  positions[targetIndex * 2 + 1] = targetY;
  velocities[targetIndex * 2] = 0;
  velocities[targetIndex * 2 + 1] = 0;

  for (const node of active) {
    if (node === targetIndex || pinned.has(node)) continue;
    let forceX = 0;
    let forceY = 0;
    const x = positions[node * 2];
    const y = positions[node * 2 + 1];
    for (const neighbour of neighbours[node] ?? []) {
      const dx = positions[neighbour * 2] - x;
      const dy = positions[neighbour * 2 + 1] - y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const delta = distance - 72;
      const factor = (delta * spring) / distance;
      forceX += dx * factor;
      forceY += dy * factor;
    }
    for (const other of active) {
      if (other === node) continue;
      const dx = x - positions[other * 2];
      const dy = y - positions[other * 2 + 1];
      const distance2 = Math.max(144, dx * dx + dy * dy);
      forceX += (dx / Math.sqrt(distance2)) * (repulsion / distance2) * 64;
      forceY += (dy / Math.sqrt(distance2)) * (repulsion / distance2) * 64;
    }
    velocities[node * 2] = (velocities[node * 2] + forceX) * damping;
    velocities[node * 2 + 1] = (velocities[node * 2 + 1] + forceY) * damping;
  }

  for (const node of active) {
    if (node === targetIndex || pinned.has(node)) continue;
    positions[node * 2] += velocities[node * 2];
    positions[node * 2 + 1] += velocities[node * 2 + 1];
  }
  emit(performance.now() - startedAt);

  const releasing = releaseDeadline > 0;
  alpha = Math.max(0, alpha * (releasing ? 0.82 : 0.94));
  if ((!releasing && alpha >= 0.025) || (releasing && performance.now() < releaseDeadline && alpha >= 0.02)) {
    schedule();
  } else if (releasing) {
    releaseDeadline = 0;
    if (!targetPinned) targetIndex = -1;
    active = [];
  }
}

workerScope.onmessage = (event: MessageEvent<Incoming>) => {
  const message = event.data;
  if (message.type === "init") {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    epoch = message.epoch;
    positions = message.positions;
    velocities = new Float32Array(positions.length);
    edges = message.edgePairs;
    pinned = new Set(message.pinnedNodeIndices);
    active = [];
    targetIndex = -1;
    targetPinned = false;
    releaseDeadline = 0;
    alpha = 0;
    buildNeighbours(positions.length / 2);
    workerScope.postMessage({ type: "ready", epoch });
    return;
  }
  if (message.epoch !== epoch) return;
  if (message.type === "dispose") {
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    active = [];
    return;
  }
  sequence = message.sequence;
  if (message.type === "drag-start") {
    targetIndex = message.nodeIndex;
    targetX = message.x;
    targetY = message.y;
    targetPinned = false;
    releaseDeadline = 0;
    active = localNodes(targetIndex, message.depth, message.limit);
    alpha = 0.72;
    schedule();
  } else if (message.type === "drag-move") {
    targetX = message.x;
    targetY = message.y;
    alpha = Math.max(alpha, 0.46);
    schedule();
  } else if (message.type === "drag-end") {
    targetX = message.x;
    targetY = message.y;
    targetPinned = message.pinned;
    releaseDeadline = performance.now() + 600;
    alpha = Math.max(alpha, 0.22);
    schedule();
  }
};

export {};
