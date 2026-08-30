import type { CoreTaskId } from "../types/core";
import { sha256Canonical } from "./graphIdentity";

export type CoreRunContextTarget =
  | { readonly kind: "community"; readonly communityId: string }
  | { readonly kind: "node"; readonly nodeId: string }
  | { readonly kind: "edge"; readonly edgeId: string }
  | { readonly kind: "node-pair"; readonly sourceId: string; readonly targetId: string };

export interface CoreRunContext {
  readonly graphVersionId: string;
  readonly taskId: CoreTaskId;
  readonly modelVersionId: string;
  readonly target: CoreRunContextTarget;
  readonly parameters: {
    readonly topKSimilarCases: number;
    readonly candidateLimit: number;
  };
}

export function coreRunContextKey(context: CoreRunContext): string {
  return sha256Canonical(context);
}
