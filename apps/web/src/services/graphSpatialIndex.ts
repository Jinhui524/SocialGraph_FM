import type { GraphEdge } from "../types/graph";

export interface IndexedPoint {
  readonly id: string;
  readonly x: number;
  readonly y: number;
}

export interface GraphBounds {
  readonly minX: number;
  readonly minY: number;
  readonly maxX: number;
  readonly maxY: number;
}

export interface SpatialPickResult {
  readonly id?: string;
  readonly candidateCount: number;
  readonly durationMs: number;
}

export interface SpatialPickOracleReport {
  readonly checked: number;
  readonly mismatches: number;
  readonly durationP95Ms: number;
  readonly candidateP95: number;
}

function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

/**
 * A small deterministic uniform-grid index. Node picking and viewport queries
 * operate on nearby cells instead of scanning every rendered element.
 */
export class GraphSpatialIndex {
  private readonly cells = new Map<string, Set<string>>();
  private readonly points = new Map<string, { x: number; y: number; cell: string }>();

  constructor(readonly cellSize = 64) {
    if (!Number.isFinite(cellSize) || cellSize <= 0) {
      throw new Error("GraphSpatialIndex cellSize must be positive");
    }
  }

  private cellCoordinate(value: number): number {
    return Math.floor(value / this.cellSize);
  }

  private cellKey(x: number, y: number): string {
    return `${this.cellCoordinate(x)}:${this.cellCoordinate(y)}`;
  }

  private removeFromCell(id: string, cell: string): void {
    const members = this.cells.get(cell);
    members?.delete(id);
    if (members?.size === 0) this.cells.delete(cell);
  }

  clear(): void {
    this.cells.clear();
    this.points.clear();
  }

  rebuild(points: readonly IndexedPoint[]): void {
    this.clear();
    this.update(points);
  }

  update(points: readonly IndexedPoint[]): void {
    for (const point of points) {
      if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
      const nextCell = this.cellKey(point.x, point.y);
      const previous = this.points.get(point.id);
      if (previous && previous.cell !== nextCell) {
        this.removeFromCell(point.id, previous.cell);
      }
      const members = this.cells.get(nextCell) ?? new Set<string>();
      members.add(point.id);
      this.cells.set(nextCell, members);
      this.points.set(point.id, { x: point.x, y: point.y, cell: nextCell });
    }
  }

  remove(ids: readonly string[]): void {
    for (const id of ids) {
      const point = this.points.get(id);
      if (!point) continue;
      this.removeFromCell(id, point.cell);
      this.points.delete(id);
    }
  }

  nearest(x: number, y: number, radius: number): SpatialPickResult {
    const startedAt = now();
    if (!Number.isFinite(x) || !Number.isFinite(y) || radius < 0) {
      return { candidateCount: 0, durationMs: now() - startedAt };
    }
    const candidates = this.queryRect({
      minX: x - radius,
      minY: y - radius,
      maxX: x + radius,
      maxY: y + radius,
    });
    let id: string | undefined;
    let bestDistance = radius * radius;
    for (const candidateId of candidates) {
      const point = this.points.get(candidateId);
      if (!point) continue;
      const dx = point.x - x;
      const dy = point.y - y;
      const distance = dx * dx + dy * dy;
      if (distance <= bestDistance) {
        bestDistance = distance;
        id = candidateId;
      }
    }
    return {
      id,
      candidateCount: candidates.size,
      durationMs: now() - startedAt,
    };
  }

  queryRect(bounds: GraphBounds): ReadonlySet<string> {
    const result = new Set<string>();
    const minCellX = this.cellCoordinate(Math.min(bounds.minX, bounds.maxX));
    const maxCellX = this.cellCoordinate(Math.max(bounds.minX, bounds.maxX));
    const minCellY = this.cellCoordinate(Math.min(bounds.minY, bounds.maxY));
    const maxCellY = this.cellCoordinate(Math.max(bounds.minY, bounds.maxY));
    for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
        const members = this.cells.get(`${cellX}:${cellY}`);
        if (!members) continue;
        for (const id of members) {
          const point = this.points.get(id);
          if (
            point &&
            point.x >= bounds.minX &&
            point.x <= bounds.maxX &&
            point.y >= bounds.minY &&
            point.y <= bounds.maxY
          ) {
            result.add(id);
          }
        }
      }
    }
    return result;
  }

  point(id: string): Readonly<{ x: number; y: number }> | undefined {
    const point = this.points.get(id);
    return point ? { x: point.x, y: point.y } : undefined;
  }

  /** Benchmark-only correctness check against an O(N) nearest-node oracle. */
  diagnosePicking(sampleCount = 200, radius = 1): SpatialPickOracleReport {
    const points = [...this.points.entries()];
    if (points.length === 0 || sampleCount <= 0) {
      return { checked: 0, mismatches: 0, durationP95Ms: 0, candidateP95: 0 };
    }
    const knownCount = Math.min(points.length, Math.ceil(sampleCount / 2));
    const queries: Array<{ x: number; y: number }> = [];
    for (let index = 0; index < knownCount; index += 1) {
      const sourceIndex = Math.floor((index / Math.max(1, knownCount)) * points.length);
      const point = points[Math.min(points.length - 1, sourceIndex)]?.[1];
      if (point) queries.push({ x: point.x, y: point.y });
    }
    const xs = points.map(([, point]) => point.x);
    const ys = points.map(([, point]) => point.y);
    const outsideX = Math.max(...xs) + Math.max(this.cellSize * 4, radius * 4);
    const outsideY = Math.max(...ys) + Math.max(this.cellSize * 4, radius * 4);
    while (queries.length < sampleCount) {
      const offset = queries.length - knownCount;
      queries.push({ x: outsideX + offset * this.cellSize, y: outsideY + offset * this.cellSize });
    }
    const durations: number[] = [];
    const candidates: number[] = [];
    let mismatches = 0;
    for (const query of queries) {
      const indexed = this.nearest(query.x, query.y, radius);
      let oracleId: string | undefined;
      let bestDistance = radius * radius;
      for (const [id, point] of points) {
        const dx = point.x - query.x;
        const dy = point.y - query.y;
        const distance = dx * dx + dy * dy;
        if (distance <= bestDistance) {
          bestDistance = distance;
          oracleId = id;
        }
      }
      if (indexed.id !== oracleId) mismatches += 1;
      durations.push(indexed.durationMs);
      candidates.push(indexed.candidateCount);
    }
    const p95 = (values: number[]) => {
      const ordered = [...values].sort((left, right) => left - right);
      return ordered[Math.max(0, Math.ceil(ordered.length * 0.95) - 1)] ?? 0;
    };
    return {
      checked: queries.length,
      mismatches,
      durationP95Ms: p95(durations),
      candidateP95: p95(candidates),
    };
  }
}

/** Stable adjacency lookups used by highlighting and local force relaxation. */
export class GraphAdjacencyIndex {
  private readonly neighboursByNode = new Map<string, Set<string>>();
  private readonly edgeIdsByNode = new Map<string, Set<string>>();

  constructor(edges: readonly Pick<GraphEdge, "id" | "source" | "target">[]) {
    for (const edge of edges) {
      this.add(this.neighboursByNode, edge.source, edge.target);
      this.add(this.neighboursByNode, edge.target, edge.source);
      this.add(this.edgeIdsByNode, edge.source, edge.id);
      this.add(this.edgeIdsByNode, edge.target, edge.id);
    }
  }

  private add(map: Map<string, Set<string>>, key: string, value: string): void {
    const values = map.get(key) ?? new Set<string>();
    values.add(value);
    map.set(key, values);
  }

  neighbours(id: string): ReadonlySet<string> {
    return this.neighboursByNode.get(id) ?? new Set<string>();
  }

  edgeIds(id: string): ReadonlySet<string> {
    return this.edgeIdsByNode.get(id) ?? new Set<string>();
  }

  localNodeIds(seedId: string, depth: number, limit: number): readonly string[] {
    if (!seedId || limit <= 0) return [];
    const result = [seedId];
    const seen = new Set(result);
    let frontier = [seedId];
    for (let step = 0; step < depth && frontier.length && result.length < limit; step += 1) {
      const next: string[] = [];
      for (const id of frontier) {
        for (const neighbour of this.neighbours(id)) {
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
}
