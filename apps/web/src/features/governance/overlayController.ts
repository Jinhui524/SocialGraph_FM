import { buildRawFactsOverlay } from "../../services/graphOverlays";
import type { GraphViewAction } from "../../services/graphViewState";
import type { AnalysisOverlay, GraphVersion } from "../../types/graph";

export function resolveGraphVersionOverlay(
  graphVersionId: string,
  explicitOverlay: AnalysisOverlay | null | undefined,
  defaultOverlay: AnalysisOverlay | null | undefined,
): AnalysisOverlay | null {
  if (explicitOverlay?.graphVersionId === graphVersionId) return explicitOverlay;
  if (defaultOverlay?.graphVersionId === graphVersionId) return defaultOverlay;
  return null;
}
export interface GovernanceCandidateFocusInput {
  readonly messageRunId?: string;
  readonly currentRunId?: string;
  readonly runGraphVersionHash?: string;
  readonly currentGraphVersionHash?: string;
  readonly previewNodes: readonly {
    readonly id: string;
    readonly score: number | null;
  }[];
  readonly graphNodeIds: readonly string[];
}

export type GovernanceCandidateFocus = Readonly<{
  status: "ready" | "stale" | "empty";
  nodeIds: readonly string[];
}>;

export function resolveGovernanceCandidateFocus(
  input: GovernanceCandidateFocusInput,
): GovernanceCandidateFocus {
  if (
    !input.messageRunId
    || input.messageRunId !== input.currentRunId
    || !input.runGraphVersionHash
    || input.runGraphVersionHash !== input.currentGraphVersionHash
  ) {
    return Object.freeze({ status: "stale", nodeIds: Object.freeze([]) });
  }
  const graphNodeIds = new Set(input.graphNodeIds);
  const seen = new Set<string>();
  const nodeIds = input.previewNodes
    .map((node, index) => ({ node, index }))
    .sort((left, right) => {
      const leftScore = typeof left.node.score === "number" ? left.node.score : Number.NEGATIVE_INFINITY;
      const rightScore = typeof right.node.score === "number" ? right.node.score : Number.NEGATIVE_INFINITY;
      return rightScore - leftScore || left.index - right.index;
    })
    .map(({ node }) => node.id)
    .filter((nodeId) => {
      if (!graphNodeIds.has(nodeId) || seen.has(nodeId)) return false;
      seen.add(nodeId);
      return true;
    })
    .slice(0, 8);
  return nodeIds.length
    ? Object.freeze({ status: "ready", nodeIds: Object.freeze(nodeIds) })
    : Object.freeze({ status: "empty", nodeIds: Object.freeze([]) });
}

export interface GraphVersionOverlayControllerOptions {
  readonly computeDefaultOverlay: (
    version: GraphVersion,
    activation: number,
  ) => Promise<AnalysisOverlay | null>;
  readonly onOverlayChange: (overlay: AnalysisOverlay | null) => void;
  readonly onError: (error: unknown) => void;
}

export class GraphVersionOverlayController {
  private activation = 0;
  private activeGraphVersionId: string | null = null;
  private explicitOverlay: AnalysisOverlay | null = null;
  private rawOverlay: AnalysisOverlay | null = null;
  private defaultOverlayVisible = false;
  private readonly defaultOverlays = new Map<string, AnalysisOverlay>();
  private readonly inFlightGlobalResults = new Map<string, Promise<AnalysisOverlay | null>>();
  private readonly successfulGlobalResults = new Set<string>();
  private readonly acceptedGlobalResultByGraph = new Map<string, string>();
  private readonly knownGlobalResultsByGraph = new Map<string, Set<string>>();

  constructor(private readonly options: GraphVersionOverlayControllerOptions) {}

  activate(version: GraphVersion, explicitOverlay: AnalysisOverlay | null = null): void {
    ++this.activation;
    this.activeGraphVersionId = version.id;
    this.rawOverlay = buildRawFactsOverlay(version);
    const acceptedKey = this.acceptedGlobalResultByGraph.get(version.id);
    this.defaultOverlayVisible = Boolean(acceptedKey && this.defaultOverlays.has(acceptedKey));
    this.explicitOverlay = explicitOverlay?.graphVersionId === version.id ? explicitOverlay : null;
    this.publish();
  }

  acceptGlobalResult(
    version: GraphVersion,
    result: {
      readonly protocol: "global";
      readonly status: "succeeded";
      readonly graphVersionHash: string;
      readonly runId: string;
      readonly resultHash: string;
    },
  ): boolean {
    const graphHash = version.datasetArtifact?.canonicalGraphHash ?? version.contentHash;
    if (
      this.activeGraphVersionId !== version.id
      || result.protocol !== "global"
      || result.status !== "succeeded"
      || result.graphVersionHash !== graphHash
      || version.nodes.length === 0
    ) return false;
    const resultKey = [
      version.id,
      result.graphVersionHash,
      result.runId,
      result.resultHash,
    ].join("\u0000");
    const knownResults = this.knownGlobalResultsByGraph.get(version.id) ?? new Set<string>();
    const acceptedResult = this.acceptedGlobalResultByGraph.get(version.id);
    if (acceptedResult && acceptedResult !== resultKey && knownResults.has(resultKey)) return false;
    knownResults.add(resultKey);
    this.knownGlobalResultsByGraph.set(version.id, knownResults);
    this.acceptedGlobalResultByGraph.set(version.id, resultKey);
    if (this.successfulGlobalResults.has(resultKey)) {
      if (this.defaultOverlays.has(resultKey)) {
        this.defaultOverlayVisible = true;
        this.publish();
      }
      return false;
    }
    if (this.inFlightGlobalResults.has(resultKey)) return false;
    const activation = this.activation;
    const pending = this.options.computeDefaultOverlay(version, activation);
    this.inFlightGlobalResults.set(resultKey, pending);
    void pending.then((defaultOverlay) => {
      this.inFlightGlobalResults.delete(resultKey);
      if (defaultOverlay?.graphVersionId !== version.id) return;
      this.defaultOverlays.set(resultKey, defaultOverlay);
      this.successfulGlobalResults.add(resultKey);
      if (
        this.activeGraphVersionId !== version.id
        || this.acceptedGlobalResultByGraph.get(version.id) !== resultKey
      ) return;
      this.defaultOverlayVisible = true;
      this.publish();
    }).catch((error) => {
      this.inFlightGlobalResults.delete(resultKey);
      if (this.activeGraphVersionId === version.id
        && this.acceptedGlobalResultByGraph.get(version.id) === resultKey) {
        this.options.onError(error);
      }
    });
    return true;
  }

  setExplicit(graphVersionId: string, overlay: AnalysisOverlay): void {
    if (
      this.activeGraphVersionId !== graphVersionId
      || overlay.graphVersionId !== graphVersionId
    ) return;
    this.explicitOverlay = overlay;
    this.publish();
  }

  clearExplicit(graphVersionId: string, expectedKind?: AnalysisOverlay["kind"]): void {
    if (this.activeGraphVersionId !== graphVersionId) return;
    if (expectedKind && this.explicitOverlay?.kind !== expectedKind) return;
    this.explicitOverlay = null;
    this.publish();
  }

  deactivate(): void {
    this.activation += 1;
    this.activeGraphVersionId = null;
    this.explicitOverlay = null;
    this.rawOverlay = null;
    this.defaultOverlayVisible = false;
    this.options.onOverlayChange(null);
  }

  private publish(): void {
    if (!this.activeGraphVersionId) {
      this.options.onOverlayChange(null);
      return;
    }
    this.options.onOverlayChange(resolveGraphVersionOverlay(
      this.activeGraphVersionId,
      this.explicitOverlay,
      this.defaultOverlayVisible
        ? this.defaultOverlays.get(this.acceptedGlobalResultByGraph.get(this.activeGraphVersionId) ?? "") ?? this.rawOverlay
        : this.rawOverlay,
    ));
  }
}

export interface GovernanceCandidateLocateEffects {
  readonly applyGraphAction: (action: GraphViewAction) => void;
  readonly expandGraph: () => void;
  readonly switchMobilePanel: (panel: "graph") => void;
  readonly notify: (message: string) => void;
  readonly saveOverview?: () => void;
}

export function locateGovernanceCandidates(
  input: GovernanceCandidateFocusInput,
  effects: GovernanceCandidateLocateEffects,
): GovernanceCandidateFocus {
  const focus = resolveGovernanceCandidateFocus(input);
  if (focus.status === "stale") {
    effects.notify("该候选报告不属于当前图谱，请重新运行分析");
    return focus;
  }
  if (focus.status === "empty") {
    effects.notify("当前图谱预览中没有可定位的重点候选");
    return focus;
  }
  effects.saveOverview?.();
  effects.applyGraphAction({ type: "set_focus", nodeIds: focus.nodeIds });
  effects.expandGraph();
  effects.switchMobilePanel("graph");
  effects.notify(`已定位 ${focus.nodeIds.length} 个重点候选`);
  return focus;
}
