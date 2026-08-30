import type { SidebarWorkspace } from "../../components/Sidebar";
import type {
  GraphCameraSnapshot,
  GraphCameraSnapshotCacheKey,
} from "../../services/graphEngineAdapter";
import { GraphCameraSnapshotCache } from "../../services/graphEngineAdapter";
import type { WorkspaceRoute } from "../../services/workspaceRoute";

export function resolveWorkspaceCameraSnapshot(
  cache: GraphCameraSnapshotCache,
  key: GraphCameraSnapshotCacheKey | null,
): GraphCameraSnapshot | undefined {
  return key ? cache.get(key) : undefined;
}

export function sidebarWorkspaceFromRoute(route: WorkspaceRoute): SidebarWorkspace {
  if (route === "governance" || route === "adaptation") return route;
  return "chat";
}

export function routeForSidebarWorkspace(workspace: SidebarWorkspace): WorkspaceRoute {
  return workspace === "chat" ? "research" : workspace;
}
