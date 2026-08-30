import type { GraphPath, GraphScene, GraphSlice, GraphVersion } from "../types/graph";
import type { CommunityAnalysis } from "./graphCommunities";
import type { BuildGraphSceneOptions } from "./graphScene";

export type GraphWorkerRequest =
  | {
      readonly id: string;
      readonly kind: "community";
      readonly graph: GraphVersion;
      readonly seed?: string;
    }
  | {
      readonly id: string;
      readonly kind: "local_subgraph";
      readonly graph: GraphVersion;
      readonly focusNodeIds: readonly string[];
      readonly depth: 1 | 2 | 3;
    }
  | {
      readonly id: string;
      readonly kind: "shortest_path";
      readonly graph: GraphVersion;
      readonly sourceId: string;
      readonly targetId: string;
    }
  | {
      readonly id: string;
      readonly kind: "build_scene";
      readonly graph: GraphVersion;
      readonly options?: BuildGraphSceneOptions;
    };

export type GraphWorkerResult = CommunityAnalysis | GraphSlice | GraphPath | GraphScene | null;

export type GraphWorkerResponse =
  | { readonly id: string; readonly ok: true; readonly result: GraphWorkerResult }
  | { readonly id: string; readonly ok: false; readonly error: string };
