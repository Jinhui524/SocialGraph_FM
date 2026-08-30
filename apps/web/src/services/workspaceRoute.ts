export const WORKSPACE_ROUTES = ["research", "governance", "adaptation", "datasets"] as const;

export type WorkspaceRoute = typeof WORKSPACE_ROUTES[number];

const ROUTE_SET = new Set<string>(WORKSPACE_ROUTES);

export function hashForWorkspaceRoute(route: WorkspaceRoute): string {
  return `#/${route}`;
}

export function workspaceRouteFromHash(hash: string): WorkspaceRoute {
  const path = hash.replace(/^#\/?/, "").split(/[/?#]/, 1)[0]?.trim() ?? "";
  return ROUTE_SET.has(path) ? path as WorkspaceRoute : "research";
}

export function isWorkspaceGraphPaneVisible(viewportWidth: number, mobilePanel: string): boolean {
  return viewportWidth >= 1_024 || mobilePanel === "graph";
}
