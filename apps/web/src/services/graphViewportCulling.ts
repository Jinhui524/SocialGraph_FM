export interface ViewportCullSnapshot {
  readonly sceneKey: string;
  readonly nodeIds: ReadonlySet<string>;
}

/** Empty or previous-scene sets must never hide a newly rendered graph. */
export function resolveViewportCullSnapshot(
  snapshot: ViewportCullSnapshot | undefined,
  sceneKey: string,
): ReadonlySet<string> | undefined {
  if (!snapshot || snapshot.sceneKey !== sceneKey || snapshot.nodeIds.size === 0) {
    return undefined;
  }
  return snapshot.nodeIds;
}

export function isSceneOutsideViewport(
  sceneNodeIds: ReadonlySet<string>,
  viewportNodeIds: ReadonlySet<string>,
): boolean {
  if (sceneNodeIds.size === 0) return false;
  for (const id of sceneNodeIds) {
    if (viewportNodeIds.has(id)) return false;
  }
  return true;
}
