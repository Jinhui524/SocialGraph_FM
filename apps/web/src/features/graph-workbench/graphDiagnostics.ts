export function publishGraphCameraDataset(
  root: HTMLElement,
  camera: { readonly x: number; readonly y: number; readonly zoom: number },
  worldCenter: { readonly x: number; readonly y: number },
): void {
  root.dataset.cameraX = String(camera.x);
  root.dataset.cameraY = String(camera.y);
  root.dataset.cameraZoom = String(camera.zoom);
  root.dataset.worldCenterX = String(worldCenter.x);
  root.dataset.worldCenterY = String(worldCenter.y);
}

export function publishGraphNodeCoordinateDataset(
  root: HTMLElement,
  node: { readonly id: string; readonly x: number; readonly y: number },
): void {
  root.dataset.coordinateNodeId = node.id;
  root.dataset.coordinateNodeX = String(node.x);
  root.dataset.coordinateNodeY = String(node.y);
}
