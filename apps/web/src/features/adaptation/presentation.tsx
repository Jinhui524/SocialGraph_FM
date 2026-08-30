import type { GraphPreviewProps } from "../../components/GraphPreview";
import type {
  AdaptationGovernanceTarget,
  AdaptationLane,
} from "../../components/GovernanceWorkbench";
import type { GovernanceTaskEntry } from "../../components/GovernanceTaskSelector";
import type { AnalysisOverlay, GovernanceFocus, GraphVersion } from "../../types/graph";

export interface AdaptationLanePresentation {
  readonly graph: GraphVersion | null;
  readonly overlay: AnalysisOverlay | null;
  readonly focus: GovernanceFocus | undefined;
  readonly camera: GraphPreviewProps["cameraFocusCommand"] | undefined;
  readonly abortEpoch: number;
}

export interface AdaptationLanePresentationState {
  readonly activeLane: AdaptationLane;
  readonly lanes: Readonly<Record<AdaptationLane, AdaptationLanePresentation>>;
}

const EMPTY_ADAPTATION_PRESENTATION: AdaptationLanePresentation = Object.freeze({
  graph: null,
  overlay: null,
  focus: undefined,
  camera: undefined,
  abortEpoch: 0,
});

export function createAdaptationLanePresentationState(): AdaptationLanePresentationState {
  return Object.freeze({
    activeLane: "zero_shot",
    lanes: Object.freeze({
      zero_shot: EMPTY_ADAPTATION_PRESENTATION,
      few_shot: EMPTY_ADAPTATION_PRESENTATION,
    }),
  });
}

export function updateAdaptationLanePresentation(
  state: AdaptationLanePresentationState,
  lane: AdaptationLane,
  patch: Partial<AdaptationLanePresentation>,
): AdaptationLanePresentationState {
  const lanes = Object.freeze({
    ...state.lanes,
    [lane]: Object.freeze({ ...state.lanes[lane], ...patch }),
  });
  const otherLane: AdaptationLane = state.activeLane === "zero_shot" ? "few_shot" : "zero_shot";
  const activeLane = lanes[state.activeLane].graph || !lanes[otherLane].graph
    ? state.activeLane
    : otherLane;
  return Object.freeze({ activeLane, lanes });
}

export function activateAdaptationLanePresentation(
  state: AdaptationLanePresentationState,
  lane: AdaptationLane,
): AdaptationLanePresentationState {
  if (state.activeLane === lane) return state;
  if (!state.lanes[lane].graph && state.lanes[state.activeLane].graph) return state;
  return Object.freeze({ ...state, activeLane: lane });
}

export function adaptationCameraLens(lane: AdaptationLane): string {
  return `adaptation:${lane}`;
}

export function AdaptationGraphSwitcher({
  state,
  onSelect,
}: {
  readonly state: AdaptationLanePresentationState;
  readonly onSelect: (lane: AdaptationLane) => void;
}) {
  if (!state.lanes.zero_shot.graph || !state.lanes.few_shot.graph) return null;
  return (
    <div className="adaptation-graph-switcher" role="group" aria-label="切换适配网络">
      <button
        type="button"
        aria-pressed={state.activeLane === "zero_shot"}
        onClick={() => onSelect("zero_shot")}
      >
        零样本网络
      </button>
      <button
        type="button"
        aria-pressed={state.activeLane === "few_shot"}
        onClick={() => onSelect("few_shot")}
      >
        少样本网络
      </button>
    </div>
  );
}

export function governanceTaskEntryFromAdaptationTarget(
  target: AdaptationGovernanceTarget,
): GovernanceTaskEntry {
  const adaptation = target.lane === "zero_shot"
    ? Object.freeze({ lane: target.lane, registration: target.registration })
    : target.handoff && target.policy && target.comparison && target.adaptedOverlay
      ? Object.freeze({
          lane: target.lane,
          registration: target.registration,
          handoff: target.handoff,
          policy: target.policy,
          comparison: target.comparison,
          adaptedOverlay: target.adaptedOverlay,
        })
      : null;
  if (!adaptation) throw new Error("ADAPTATION_GOVERNANCE_HANDOFF_INCOMPLETE");
  return Object.freeze({
    id: target.registration.registrationId,
    label: target.registration.task.displayName,
    kind: "target" as const,
    snapshot: target.snapshot,
    graph: target.graph,
    adaptation,
    validationToken: 0,
  });
}
