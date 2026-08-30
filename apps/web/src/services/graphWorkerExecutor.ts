import { computeLouvainCommunities } from "./graphCommunities";
import { buildGraphScene } from "./graphScene";
import { extractLocalSubgraph, findShortestPath } from "./graphTraversal";
import type { GraphWorkerRequest, GraphWorkerResult } from "./graphWorkerProtocol";

export function runGraphTaskDirect(request: GraphWorkerRequest): GraphWorkerResult {
  switch (request.kind) {
    case "community":
      return computeLouvainCommunities(request.graph, request.seed);
    case "local_subgraph":
      return extractLocalSubgraph(request.graph, request.focusNodeIds, request.depth);
    case "shortest_path":
      return findShortestPath(request.graph, request.sourceId, request.targetId);
    case "build_scene":
      return buildGraphScene(request.graph, request.options);
  }
}
