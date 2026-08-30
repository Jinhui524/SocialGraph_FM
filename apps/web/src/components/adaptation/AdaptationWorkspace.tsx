import { CheckCircle, CircleNotch, FileArrowUp, ShieldCheck, WarningCircle, X } from "@phosphor-icons/react";
import { useEffect, useMemo, useRef, useState, type RefObject } from "react";

import { graphTopologyKey } from "../../services/graphDeterministicLayout";
import { sha256Canonical } from "../../services/graphIdentity";
import { graphTypeColour } from "../../services/graphTypePalette";
import { buildAdaptedReviewPriorityOverlay } from "../../services/governanceAdaptation";
import { governancePreviewGraph } from "../GovernanceOnlineWorkspace";
import { GOVERNANCE_ONLINE_SCHEMA, GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA, type TargetAdaptationComparison, type AdaptationGovernanceHandoff, type GovernanceOnlineClientLike, type GovernanceOnlineEvidence, type GovernanceOnlinePreview, type GovernanceOnlineResult, type GovernanceOnlineRun, type TargetReviewPolicy, type TargetTaskRegistration } from "../../types/governanceOnline";
import type { AnalysisOverlay, GovernanceFocus, GraphVersion } from "../../types/graph";
import { GOVERNANCE_WORKSPACE_SCHEMA, type GovernanceWorkspaceSnapshot } from "../../services/governanceWorkspaceStore";
import { AdaptationTransferEvidenceDialog } from "./AdaptationTransferEvidenceDialog";
import { buildAdaptationTransferEvidence, type AdaptationModelCardState } from "./AdaptationTransferEvidence";

export type { AdaptationModelCardState } from "./AdaptationTransferEvidence";

export type AdaptationLane = "zero_shot" | "few_shot";

export interface AdaptationGovernanceTarget {
  readonly lane: AdaptationLane;
  readonly registration: TargetTaskRegistration;
  readonly snapshot: GovernanceWorkspaceSnapshot;
  readonly graph: GraphVersion;
  readonly handoff?: AdaptationGovernanceHandoff;
  readonly policy?: TargetReviewPolicy;
  readonly comparison?: TargetAdaptationComparison;
  readonly adaptedOverlay?: AnalysisOverlay;
}

export interface AdaptationLanePresentationPatch {
  readonly graph?: GraphVersion | null;
  readonly overlay?: AnalysisOverlay | null;
  readonly focus?: GovernanceFocus | undefined;
  readonly camera?: { readonly nodeIds: readonly string[]; readonly anchorNodeId?: string; readonly token: number; readonly projectionIdentity?: string } | undefined;
  readonly abortEpoch?: number;
}

export interface GovernanceWorkbenchProps {
  readonly client: GovernanceOnlineClientLike;
  readonly onGraphChange?: (lane: AdaptationLane, graph: GraphVersion | null) => void;
  readonly onOverlayChange: (overlay: AnalysisOverlay | null) => void;
  readonly onFocusChange?: (focus: GovernanceFocus | undefined) => void;
  readonly onCameraFocusChange?: (command: { readonly nodeIds: readonly string[]; readonly anchorNodeId?: string; readonly token: number; readonly projectionIdentity?: string } | undefined) => void;
  readonly onGovernanceHandoff?: (target: AdaptationGovernanceTarget) => void;
  readonly onLanePresentationChange?: (lane: AdaptationLane, patch: AdaptationLanePresentationPatch) => void;
  readonly onActiveLaneChange?: (lane: AdaptationLane) => void;
  readonly modelCardState?: AdaptationModelCardState;
  readonly onClose: () => void;
  /* Kept optional during the App migration; target lanes never read shared governance state. */
  readonly graph?: GraphVersion | null;
  readonly snapshot?: GovernanceWorkspaceSnapshot | null;
  readonly onOverviewLoad?: (preview: GovernanceOnlinePreview, result: GovernanceOnlineResult) => void;
  readonly onOverviewClear?: () => void;
  readonly onOpenGlobal?: () => void;
  readonly selectedNodeId?: string | null;
  readonly onSelectedNodeIdChange?: (nodeId: string | null) => void;
}

interface LaneEpochs {
  readonly artifact: number;
  readonly run: number;
  readonly policy: number;
  readonly graph: number;
  readonly camera: number;
  readonly focus: number;
  readonly abort: number;
}

type LanePhase = "empty" | "registering" | "raw" | "confirm_run" | "running" | "ready" | "fitting" | "calibration_error" | "insufficient" | "compared" | "handoff" | "error";

interface LaneState {
  readonly phase: LanePhase;
  readonly epochs: LaneEpochs;
  readonly fileName: string | null;
  readonly registration: TargetTaskRegistration | null;
  readonly preview: GovernanceOnlinePreview | null;
  readonly graph: GraphVersion | null;
  readonly run: GovernanceOnlineRun | null;
  readonly result: GovernanceOnlineResult | null;
  readonly policy: TargetReviewPolicy | null;
  readonly comparison: TargetAdaptationComparison | null;
  readonly selectedNodeId: string | null;
  readonly evidence: GovernanceOnlineEvidence | null;
  readonly page: number;
  readonly overlay: "labels" | "community";
  readonly message: string | null;
}

const ZERO_EPOCHS: LaneEpochs = Object.freeze({ artifact: 0, run: 0, policy: 0, graph: 0, camera: 0, focus: 0, abort: 0 });
const EMPTY_STATE: LaneState = Object.freeze({ phase: "empty", epochs: ZERO_EPOCHS, fileName: null, registration: null, preview: null, graph: null, run: null, result: null, policy: null, comparison: null, selectedNodeId: null, evidence: null, page: 0, overlay: "community", message: null });
const PAGE_SIZE = 25;

function changedPackageEpochs(value: LaneEpochs): LaneEpochs {
  return { artifact: value.artifact + 1, run: value.run + 1, policy: value.policy + 1, graph: value.graph + 1, camera: value.camera + 1, focus: value.focus + 1, abort: value.abort + 1 };
}

function abortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError" || error instanceof Error && error.name === "AbortError";
}

function exactPreview(preview: GovernanceOnlinePreview, registration: TargetTaskRegistration, scored: boolean): boolean {
  const nodeIds = new Set(preview.nodes.map((node) => node.id));
  return preview.artifactId === registration.artifact.artifactId
    && preview.datasetContentHash === registration.artifact.datasetContentHash
    && preview.graphVersionHash === registration.artifact.graphVersionHash
    && preview.nodeCount === registration.task.nodeCount
    && preview.nodes.length === registration.task.nodeCount
    && nodeIds.size === registration.task.nodeCount
    && preview.edgeCount === registration.task.fusedEdgeCount
    && preview.edges.length === registration.task.fusedEdgeCount
    && !preview.partialPreview
    && preview.edges.every((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
    && (scored ? Boolean(preview.runId && preview.resultHash) : preview.runId === null && preview.resultHash === null);
}

function exactResult(result: GovernanceOnlineResult, run: GovernanceOnlineRun, registration: TargetTaskRegistration, preview: GovernanceOnlinePreview): boolean {
  const previewNodeIds = new Set(preview.nodes.map((node) => node.id));
  const findingNodeIds = new Set(result.findings.map((finding) => finding.nodeId));
  return result.runId === run.runId && result.requestHash === run.requestHash
    && result.artifactId === registration.artifact.artifactId
    && result.datasetContentHash === registration.artifact.datasetContentHash
    && result.graphVersionHash === registration.artifact.graphVersionHash
    && result.modelVersionHash === run.modelVersionHash && result.modelStateHash === run.modelStateHash
    && result.totalFindings === registration.task.nodeCount && result.distribution.total === registration.task.nodeCount
    && result.findings.length === registration.task.nodeCount
    && findingNodeIds.size === previewNodeIds.size
    && [...previewNodeIds].every((nodeId) => findingNodeIds.has(nodeId));
}

function fewShotLabelsReady(registration: TargetTaskRegistration): boolean {
  const labels = registration.labels; const receipt = registration.labelReceipt;
  if (registration.task.mode !== "few_shot" || !labels || !receipt
    || labels.labels.length !== 16 || labels.positiveCount !== 8 || labels.negativeCount !== 8
    || labels.labels.filter((row) => row.label === "positive").length !== 8
    || labels.labels.filter((row) => row.label === "negative").length !== 8
    || new Set(labels.labels.map((row) => row.nodeId)).size !== 16
    || labels.taskId !== registration.task.taskId || labels.taskId !== registration.targetReceipt.taskId
    || labels.inferenceSha256 !== registration.targetReceipt.inferenceSha256
    || labels.inferenceSha256 !== registration.artifact.bundleSha256) return false;
  const eligible = new Set(receipt.eligibleNodeIds);
  return labels.labels.every((row) => eligible.has(row.nodeId));
}

export function buildFewShotLabelOverlay(
  graph: GraphVersion,
  registration: TargetTaskRegistration,
): AnalysisOverlay | null {
  if (!fewShotLabelsReady(registration) || !registration.labels) return null;
  const graphNodeIds = new Set(graph.nodes.map((node) => node.id));
  if (registration.labels.labels.some((row) => !graphNodeIds.has(row.nodeId))) return null;
  const referenceLabels = Object.freeze(Object.fromEntries(registration.labels.labels.map((row) => [row.nodeId, row.label])));
  const nodeValues = Object.freeze(Object.fromEntries(registration.labels.labels.map((row) => [
    row.nodeId,
    row.label === "positive" ? "reference-positive" : "reference-negative",
  ])));
  return Object.freeze({
    id: `${graph.id}:imported-few-shot-labels:${registration.labels.labelSetHash}`,
    graphVersionId: graph.id,
    kind: "governance",
    nodeValues,
    edgeValues: Object.freeze({}),
    presentation: Object.freeze({ governanceLens: "risk", referenceLabels }),
    legend: Object.freeze({
      title: "已知标签",
      items: Object.freeze([
        Object.freeze({ value: "reference-positive", label: "+ 已标注风险", color: "#D85C56" }),
        Object.freeze({ value: "reference-negative", label: "− 已标注对照", color: "#218B7C" }),
      ]),
    }),
    provenance: Object.freeze({
      engine: "imported-label-set",
      algorithm: "few-shot-label-view-v1",
      scopeHash: registration.labels.labelSetHash,
      taskId: registration.task.taskId,
      graphVersionHash: registration.artifact.graphVersionHash,
    }),
  });
}

export function decorateWithFewShotReferenceLabels(
  overlay: AnalysisOverlay,
  registration: TargetTaskRegistration,
): AnalysisOverlay {
  if (!fewShotLabelsReady(registration) || !registration.labels) return overlay;
  const referenceLabels = Object.freeze(Object.fromEntries(
    registration.labels.labels.map((row) => [row.nodeId, row.label]),
  ));
  const existingLegendValues = new Set(overlay.legend.items.map((item) => item.value));
  return Object.freeze({
    ...overlay,
    presentation: Object.freeze({
      ...overlay.presentation,
      referenceLabels,
    }),
    legend: Object.freeze({
      title: overlay.legend.title,
      items: Object.freeze([
        ...(!existingLegendValues.has("reference-positive") ? [{ value: "reference-positive", label: "+ 已标注风险", color: "#D85C56" }] : []),
        ...(!existingLegendValues.has("reference-negative") ? [{ value: "reference-negative", label: "− 已标注对照", color: "#218B7C" }] : []),
        ...overlay.legend.items,
      ]),
    }),
  });
}

function sameBinding(left: TargetReviewPolicy["binding"], right: TargetReviewPolicy["binding"]): boolean {
  return sha256Canonical(left) === sha256Canonical(right);
}

function bindingMatchesExecution(binding: TargetReviewPolicy["binding"], registration: TargetTaskRegistration, run: GovernanceOnlineRun, result: GovernanceOnlineResult): boolean {
  return binding.artifactId === registration.artifact.artifactId
    && binding.datasetContentHash === registration.artifact.datasetContentHash
    && binding.graphVersionHash === registration.artifact.graphVersionHash
    && binding.runId === run.runId && binding.requestHash === run.requestHash && binding.resultHash === result.resultHash
    && binding.modelVersionId === result.modelVersionId && binding.modelVersionHash === result.modelVersionHash
    && binding.modelStateHash === result.modelStateHash;
}

function comparisonIsExact(comparison: TargetAdaptationComparison, graph: GraphVersion, policy: TargetReviewPolicy): boolean {
  const graphIds = new Set(graph.nodes.map((node) => node.id)); const rowIds = new Set(comparison.rows.map((row) => row.nodeId));
  const permutation = (ranks: readonly number[]) => [...ranks].sort((left, right) => left - right).every((rank, index) => rank === index + 1);
  return comparison.policyHash === policy.policyHash && sameBinding(comparison.binding, policy.binding)
    && comparison.total === graphIds.size && comparison.rows.length === graphIds.size && rowIds.size === graphIds.size
    && [...graphIds].every((nodeId) => rowIds.has(nodeId))
    && permutation(comparison.rows.map((row) => row.baseRank)) && permutation(comparison.rows.map((row) => row.adaptedRank));
}

export function buildFindingCommunityOverlay(
  graph: GraphVersion,
  result: GovernanceOnlineResult,
  registration?: TargetTaskRegistration,
): AnalysisOverlay {
  const graphNodeIds = new Set(graph.nodes.map((node) => node.id));
  const findingNodeIds = new Set(result.findings.map((finding) => finding.nodeId));
  if (graphNodeIds.size !== findingNodeIds.size || [...graphNodeIds].some((nodeId) => !findingNodeIds.has(nodeId))) {
    throw new Error("COMMUNITY_FINDINGS_INCOMPLETE");
  }
  const unassigned = "community-unassigned";
  const counts = new Map<string, number>();
  const nodeValues = Object.freeze(Object.fromEntries(result.findings.map((finding) => {
    const communityId = finding.communityId?.trim() || unassigned;
    counts.set(communityId, (counts.get(communityId) ?? 0) + 1);
    return [finding.nodeId, communityId];
  })));
  const communities = [...counts.entries()].sort(([leftId, leftCount], [rightId, rightCount]) => rightCount - leftCount || leftId.localeCompare(rightId, "zh-CN"));
  const overlay: AnalysisOverlay = Object.freeze({
    id: `${graph.id}:finding-communities:${result.resultHash}`,
    graphVersionId: graph.id,
    kind: "community",
    nodeValues,
    edgeValues: Object.freeze({}),
    legend: Object.freeze({
      title: "协同组群",
      items: Object.freeze(communities.slice(0, 20).map(([communityId], index) => Object.freeze({
        value: communityId,
        label: communityId === unassigned ? "未归组" : `群组 ${index + 1}`,
        color: graphTypeColour(communityId),
      }))),
    }),
    provenance: Object.freeze({
      engine: "socialgraph-governance",
      algorithm: "finding-community-assignment",
      runId: result.runId,
      resultHash: result.resultHash,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
    }),
  });
  return registration ? decorateWithFewShotReferenceLabels(overlay, registration) : overlay;
}

function executionIsExact(state: LaneState): state is LaneState & {
  readonly registration: TargetTaskRegistration;
  readonly preview: GovernanceOnlinePreview;
  readonly graph: GraphVersion;
  readonly run: GovernanceOnlineRun;
  readonly result: GovernanceOnlineResult;
} {
  return Boolean(
    state.registration
    && state.preview
    && state.graph
    && state.run
    && state.result
    && exactPreview(state.preview, state.registration, true)
    && exactResult(state.result, state.run, state.registration, state.preview),
  );
}

function handoffIsReady(state: LaneState, lane: AdaptationLane): boolean {
  if (!executionIsExact(state)) return false;
  if (lane === "zero_shot") return true;
  return Boolean(
    state.policy
    && state.policy.status === "ready"
    && state.registration.labels
    && state.policy.labelSetHash === state.registration.labels.labelSetHash
    && bindingMatchesExecution(state.policy.binding, state.registration, state.run, state.result)
    && state.comparison
    && comparisonIsExact(state.comparison, state.graph, state.policy),
  );
}

function targetSnapshot(state: LaneState, activeCaseId?: string): GovernanceWorkspaceSnapshot {
  if (!state.registration || !state.preview || !state.run || !state.result) throw new Error("TARGET_HANDOFF_NOT_READY");
  return Object.freeze({ schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA, sessionId: state.registration.registrationId, sourceFileName: `${state.registration.task.displayName}.sgtask.zip`, artifact: state.registration.artifact, preview: state.preview, run: state.run, result: state.result, ...(activeCaseId ? { activeCaseId } : {}), updatedAt: new Date().toISOString() });
}

function AdaptationGuide({ zeroLaneRef, fewLaneRef, onClose }: {
  readonly zeroLaneRef: RefObject<HTMLElement | null>;
  readonly fewLaneRef: RefObject<HTMLElement | null>;
  readonly onClose: () => void;
}) {
  const focusLane = (laneRef: RefObject<HTMLElement | null>) => {
    const lane = laneRef.current;
    if (!lane) return;
    lane.scrollIntoView?.({ behavior: "smooth", block: "start" });
    lane.focus({ preventScroll: true });
  };
  return <section className="adaptation-guide" aria-labelledby="adaptation-guide-title">
    <header>
      <div><span>适配能力</span><h2 id="adaptation-guide-title">面向新网络的风险迁移</h2><p>在不更新基础模型的前提下，将既有跨域表征用于标注稀缺或全新网络，形成可移交的网络表征、协同组群与核验对象。</p></div>
      <button type="button" onClick={onClose} aria-label="关闭适配工作台"><X size={18} /></button>
    </header>
    <nav aria-label="选择适配路径">
      <button type="button" onClick={() => focusLane(zeroLaneRef)}><span><strong>跨域新活动 · 零样本</strong><small>暂无可靠标签时完成全网初筛</small></span><ShieldCheck size={20} /></button>
      <button type="button" onClick={() => focusLane(fewLaneRef)}><span><strong>稀缺标注 · 少样本</strong><small>利用少量已核对对象适配目标网络</small></span><ShieldCheck size={20} /></button>
    </nav>
    <p className="adaptation-guide__boundary">两条路径均可将候选、关系和证据移交治理应用，最终结论由人工确认。</p>
  </section>;
}

function Lane({ lane, laneRef, client, modelCardState, nextCameraToken, onGraphChange, onOverlayChange, onFocusChange, onCameraFocusChange, onGovernanceHandoff, onLanePresentationChange, onActiveLaneChange }: {
  readonly lane: AdaptationLane;
  readonly laneRef: RefObject<HTMLElement | null>;
  readonly client: GovernanceOnlineClientLike;
  readonly modelCardState?: AdaptationModelCardState;
  readonly nextCameraToken: () => number;
  readonly onGraphChange?: GovernanceWorkbenchProps["onGraphChange"];
  readonly onOverlayChange: GovernanceWorkbenchProps["onOverlayChange"];
  readonly onFocusChange?: GovernanceWorkbenchProps["onFocusChange"];
  readonly onCameraFocusChange?: GovernanceWorkbenchProps["onCameraFocusChange"];
  readonly onGovernanceHandoff?: GovernanceWorkbenchProps["onGovernanceHandoff"];
  readonly onLanePresentationChange?: GovernanceWorkbenchProps["onLanePresentationChange"];
  readonly onActiveLaneChange?: GovernanceWorkbenchProps["onActiveLaneChange"];
}) {
  const [state, setState] = useState<LaneState>(EMPTY_STATE);
  const [transferEvidenceOpen, setTransferEvidenceOpen] = useState(false);
  const stateRef = useRef(state); stateRef.current = state;
  const generationRef = useRef(0); const operationRef = useRef(0); const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const begin = (epoch: keyof Pick<LaneEpochs, "run" | "policy" | "graph" | "camera" | "focus">) => {
    abortRef.current?.abort(); const controller = new AbortController(); abortRef.current = controller; const operation = ++operationRef.current; const generation = generationRef.current;
    onActiveLaneChange?.(lane); onLanePresentationChange?.(lane, { abortEpoch: stateRef.current.epochs.abort + 1 });
    setState((current) => ({ ...current, epochs: { ...current.epochs, [epoch]: current.epochs[epoch] + 1, abort: current.epochs.abort + 1 } }));
    return { controller, operation, generation };
  };
  const current = (capture: { controller: AbortController; operation: number; generation: number }) => !capture.controller.signal.aborted && capture.operation === operationRef.current && capture.generation === generationRef.current;

  const publishCamera = (graph: GraphVersion, nodeIds: readonly string[], anchorNodeId?: string, token = nextCameraToken()) => {
    setState((value) => ({ ...value, epochs: { ...value.epochs, camera: value.epochs.camera + 1 } }));
    const command = { nodeIds, ...(anchorNodeId ? { anchorNodeId } : {}), token, projectionIdentity: graphTopologyKey(graph.nodes, graph.edges) };
    onCameraFocusChange?.(command); onLanePresentationChange?.(lane, { camera: command });
  };

  const upload = async (file: File | undefined) => {
    if (!file) return;
    setTransferEvidenceOpen(false);
    abortRef.current?.abort(); const controller = new AbortController(); abortRef.current = controller;
    const generation = ++generationRef.current; const operation = ++operationRef.current;
    setState((value) => ({ ...EMPTY_STATE, phase: "registering", fileName: file.name, epochs: changedPackageEpochs(value.epochs) }));
    onGraphChange?.(lane, null); onOverlayChange(null); onFocusChange?.(undefined); onCameraFocusChange?.(undefined);
    onLanePresentationChange?.(lane, { graph: null, overlay: null, focus: undefined, camera: undefined, abortEpoch: stateRef.current.epochs.abort + 1 });
    try {
      const registration = await client.registerTargetTask(file, controller.signal);
      if (!current({ controller, generation, operation })) return;
      if (registration.task.mode !== lane) throw new Error("TARGET_LANE_MISMATCH");
      const preview = await client.preview(registration.artifact.artifactId, controller.signal, { preset: "overview", nodeBudget: registration.task.nodeCount, edgeBudget: registration.task.fusedEdgeCount });
      if (!current({ controller, generation, operation })) return;
      if (!exactPreview(preview, registration, false)) throw new Error("TARGET_GRAPH_INCOMPLETE");
      const graph = governancePreviewGraph(preview, [], registration.task.displayName);
      const labelOverlay = lane === "few_shot" ? buildFewShotLabelOverlay(graph, registration) : null;
      setState((value) => ({ ...value, phase: "raw", registration, preview, graph, overlay: labelOverlay ? "labels" : "community", message: null, epochs: { ...value.epochs, graph: value.epochs.graph + 1 } }));
      onGraphChange?.(lane, graph);
      onOverlayChange(labelOverlay);
      onLanePresentationChange?.(lane, { graph, overlay: labelOverlay });
      onActiveLaneChange?.(lane);
      publishCamera(graph, graph.nodes.map((node) => node.id));
    } catch (error) {
      if (abortError(error) || !current({ controller, generation, operation })) return;
      setState((value) => ({ ...value, phase: "error", message: "目标任务包未通过完整性或路径校验。" }));
    }
  };

  const organizeFewShot = async (
    captured: {
      readonly registration: TargetTaskRegistration;
      readonly run: GovernanceOnlineRun;
      readonly result: GovernanceOnlineResult;
      readonly graph: GraphVersion;
    },
    operation: { readonly controller: AbortController; readonly operation: number; readonly generation: number },
  ) => {
    if (!captured.registration.labels || !fewShotLabelsReady(captured.registration)) throw new Error("TARGET_LABELS_NOT_READY");
    const labelSet = await client.createTargetLabelSet({ schemaVersion: GOVERNANCE_REGISTERED_TARGET_LABEL_SET_SCHEMA, sourceType: "imported_sidecar", targetTaskRegistrationId: captured.registration.registrationId, runId: captured.run.runId, resultHash: captured.result.resultHash }, operation.controller.signal);
    if (!current(operation) || sha256Canonical(labelSet) !== sha256Canonical(captured.registration.labels)) throw new Error("LABEL_BINDING_MISMATCH");
    const fitted = await client.fitTargetPolicy(labelSet.labelSetHash, {
      schemaVersion: "socialgraph-fm.governance-target-review-policy-fit-request/1.0",
      targetTaskRegistrationId: captured.registration.registrationId,
      runId: captured.run.runId,
      resultHash: captured.result.resultHash,
    }, operation.controller.signal);
    if (!current(operation) || fitted.labelSetHash !== labelSet.labelSetHash || !bindingMatchesExecution(fitted.binding, captured.registration, captured.run, captured.result)
      || fitted.eligibleLabelCount !== 16 || fitted.positiveCount !== 8 || fitted.negativeCount !== 8) throw new Error("POLICY_NOT_READY");
    if (fitted.status === "insufficient_signal") {
      if (fitted.selectedLambda !== 0) throw new Error("POLICY_SIGNAL_STATE_INVALID");
      setState((value) => ({
        ...value,
        phase: "insufficient",
        policy: fitted,
        comparison: null,
        overlay: "community",
          message: "标签信号不足；当前少样本网络暂不能适配。请更换标签或目标任务包后重新登记，基础结果保持不变。",
      }));
      return;
    }
    if (fitted.status !== "ready") throw new Error("POLICY_NOT_READY");
    const policy = await client.targetPolicy(fitted.policyHash, operation.controller.signal);
    if (!current(operation) || sha256Canonical(policy) !== sha256Canonical(fitted)) throw new Error("POLICY_BINDING_MISMATCH");
    const comparison = await client.targetComparison(captured.run.runId, policy.policyHash, 0, 500, operation.controller.signal);
    if (!current(operation) || !comparisonIsExact(comparison, captured.graph, policy)) throw new Error("COMPARISON_INCOMPLETE");
    setState((value) => ({ ...value, phase: "compared", policy, comparison, overlay: "community", page: 0, message: null }));
  };

  const run = async () => {
    const captured = stateRef.current; if (!captured.registration) return;
    if (captured.graph) onGraphChange?.(lane, captured.graph);
    const operation = begin("run");
    const pendingOverlay = lane === "few_shot" && captured.graph ? buildFewShotLabelOverlay(captured.graph, captured.registration) : null;
    setState((value) => ({ ...value, phase: "running", message: null, result: null, policy: null, comparison: null, evidence: null, selectedNodeId: null, page: 0, overlay: pendingOverlay ? "labels" : "community" }));
    onOverlayChange(pendingOverlay); onFocusChange?.(undefined); onLanePresentationChange?.(lane, { overlay: pendingOverlay, focus: undefined });
    try {
      const capabilities = await client.capabilities(operation.controller.signal);
      if (!current(operation) || !capabilities.onlineForwardReady || !capabilities.modelVersionId || !capabilities.modelStateHash) throw new Error("GLOBAL_UNAVAILABLE");
      let status = await client.createRun({ schemaVersion: GOVERNANCE_ONLINE_SCHEMA, protocol: "global", artifactId: captured.registration.artifact.artifactId, datasetContentHash: captured.registration.artifact.datasetContentHash, graphVersionHash: captured.registration.artifact.graphVersionHash, modelVersionId: capabilities.modelVersionId, modelStateHash: capabilities.modelStateHash, topK: captured.registration.task.nodeCount }, operation.controller.signal);
      for (let attempt = 0; current(operation) && (status.status === "queued" || status.status === "running") && attempt < 600; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 100)); status = await client.run(status.runId, operation.controller.signal);
      }
      if (!current(operation) || status.status !== "succeeded") throw new Error("GLOBAL_RUN_FAILED");
      const result = await client.result(status.runId, operation.controller.signal);
      if (!current(operation) || !captured.preview || !exactResult(result, status, captured.registration, captured.preview)) throw new Error("GLOBAL_RESULT_INCOMPLETE");
      const preview = await client.runPreview(status.runId, operation.controller.signal, { preset: "overview", nodeBudget: captured.registration.task.nodeCount, edgeBudget: captured.registration.task.fusedEdgeCount });
      if (!current(operation) || !exactPreview(preview, captured.registration, true) || preview.resultHash !== result.resultHash || preview.runId !== status.runId) throw new Error("GLOBAL_GRAPH_INCOMPLETE");
      const graph = governancePreviewGraph(preview, result.findings, captured.registration.task.displayName);
      const communityOverlay = buildFindingCommunityOverlay(graph, result, lane === "few_shot" ? captured.registration : undefined);
      setState((value) => ({ ...value, phase: lane === "few_shot" ? "fitting" : "ready", run: status, result, preview, graph, overlay: "community", message: null, epochs: { ...value.epochs, graph: value.epochs.graph + 1 } }));
      onGraphChange?.(lane, graph); onOverlayChange(communityOverlay); onLanePresentationChange?.(lane, { graph, overlay: communityOverlay }); publishCamera(graph, graph.nodes.map((node) => node.id));
      if (lane === "few_shot") {
        try {
          await organizeFewShot({ registration: captured.registration, run: status, result, graph }, operation);
        } catch (error) {
          if (abortError(error) || !current(operation)) return;
          setState((value) => ({ ...value, phase: "calibration_error", policy: null, comparison: null, overlay: "community", message: "少样本网络适配未完成；基础结果保持不变，可重试适配。" }));
        }
      }
    } catch (error) {
      if (abortError(error) || !current(operation)) return;
      setState((value) => ({ ...value, phase: "error", message: "目标网络分析未完成；没有发布协同组群。" }));
    }
  };

  const retryOrganization = async () => {
    const captured = stateRef.current;
    if (lane !== "few_shot" || !executionIsExact(captured)) return;
    onGraphChange?.(lane, captured.graph);
    const operation = begin("policy");
    setState((value) => ({ ...value, phase: "fitting", policy: null, comparison: null, overlay: "community", message: null }));
    try {
      await organizeFewShot(captured, operation);
    } catch (error) {
      if (abortError(error) || !current(operation)) return;
      setState((value) => ({ ...value, phase: "calibration_error", policy: null, comparison: null, overlay: "community", message: "少样本网络适配未完成；基础结果保持不变，可重试适配。" }));
    }
  };

  const select = async (nodeId: string) => {
    const captured = stateRef.current; if (!captured.graph || !captured.run || !captured.graph.nodes.some((node) => node.id === nodeId)) return;
    onGraphChange?.(lane, captured.graph);
    const operation = begin("focus");
    setState((value) => ({ ...value, selectedNodeId: nodeId, evidence: null }));
    const cameraToken = nextCameraToken();
    const focus = { kind: "node" as const, targetId: nodeId, nodeIds: [nodeId], cameraToken };
    onFocusChange?.(focus); onLanePresentationChange?.(lane, { focus }); publishCamera(captured.graph, [nodeId], nodeId, cameraToken);
    try { const evidence = await client.evidence(captured.run.runId, nodeId, operation.controller.signal); if (current(operation) && evidence.node.nodeId === nodeId) setState((value) => ({ ...value, evidence })); } catch (error) { if (!abortError(error) && current(operation)) setState((value) => ({ ...value, message: "直接关系证据暂时不可用。" })); }
  };

  const handoff = async () => {
    const captured = stateRef.current; if (!handoffIsReady(captured, lane) || !captured.registration || !captured.run || !captured.result) return;
    if (captured.graph) onGraphChange?.(lane, captured.graph);
    const operation = begin("policy"); setState((value) => ({ ...value, phase: "handoff", message: null }));
    try {
      if (lane === "zero_shot") {
        const items = captured.result.findings.slice(0, 25).map((finding) => ({ targetType: "node" as const, targetId: finding.nodeId, note: `风险排序 #${finding.rank}` }));
        const request = { schemaVersion: "socialgraph-fm.governance-review-collection/1.0" as const, idempotencyKey: `${captured.registration.registrationId}:${captured.result.resultHash.slice(0, 16)}`, targetTaskRegistrationId: captured.registration.registrationId, runId: captured.run.runId, resultHash: captured.result.resultHash, title: `${captured.registration.task.displayName} 风险候选复核`, description: "由零样本风险排序显式移交。", items };
        const collection = await client.createTargetReviewCollection(request, operation.controller.signal);
        if (!current(operation)) return;
        const caseItems = new Map(collection.case.items.map((item) => [`${item.targetType}:${item.targetId}`, item.note]));
        if (collection.idempotencyKey !== request.idempotencyKey || collection.targetTaskRegistrationId !== captured.registration.registrationId
          || collection.requestHash !== sha256Canonical(request) || collection.resultHash !== captured.result.resultHash
          || collection.case.runId !== captured.run.runId || collection.case.state !== "active"
          || collection.case.title !== request.title || collection.case.description !== request.description
          || collection.case.items.length !== items.length || new Set(collection.case.items.map((item) => item.itemId)).size !== items.length
          || caseItems.size !== items.length || items.some((item) => caseItems.get(`${item.targetType}:${item.targetId}`) !== item.note)) throw new Error("REVIEW_COLLECTION_IDENTITY_MISMATCH");
        if (!captured.graph) throw new Error("TARGET_GRAPH_NOT_READY");
        onGovernanceHandoff?.({ lane, registration: captured.registration, snapshot: targetSnapshot(captured, collection.case.caseId), graph: captured.graph });
      } else {
        if (!captured.policy || !captured.comparison) throw new Error("POLICY_HANDOFF_NOT_READY");
        const next = await client.createAdaptationHandoff({ schemaVersion: "socialgraph-fm.governance-adaptation-handoff/1.0", targetTaskRegistrationId: captured.registration.registrationId, policyHash: captured.policy.policyHash, decision: "pending_governance_review" }, operation.controller.signal);
        if (!current(operation) || next.baseModelMutation || next.targetTaskRegistrationId !== captured.registration.registrationId
          || next.targetReceiptHash !== captured.registration.targetReceipt.receiptHash || next.labelSetHash !== captured.policy.labelSetHash
          || next.policyHash !== captured.policy.policyHash || next.comparisonHash !== captured.comparison.comparisonHash
          || !sameBinding(next.binding, captured.policy.binding)) throw new Error("HANDOFF_IDENTITY_MISMATCH");
        if (!captured.graph) throw new Error("TARGET_GRAPH_NOT_READY");
        onGovernanceHandoff?.({
          lane,
          registration: captured.registration,
          snapshot: targetSnapshot(captured),
          graph: captured.graph,
          handoff: next,
          policy: captured.policy,
          comparison: captured.comparison,
          adaptedOverlay: decorateWithFewShotReferenceLabels(
            buildAdaptedReviewPriorityOverlay(captured.graph, captured.result, captured.comparison),
            captured.registration,
          ),
        });
      }
      if (current(operation)) setState((value) => ({ ...value, phase: captured.comparison ? "compared" : "ready", message: "已创建独立治理任务。" }));
    } catch (error) { if (!abortError(error) && current(operation)) setState((value) => ({ ...value, phase: captured.comparison ? "compared" : "ready", message: "治理移交未完成；当前结果仍保留。" })); }
  };

  const labels = state.registration?.labels;
  const labelsReady = state.registration ? fewShotLabelsReady(state.registration) : false;
  const graphLabelById = new Map(state.graph?.nodes.map((node) => [node.id, node.label]) ?? []);
  const candidates = state.result?.findings.slice(state.page * PAGE_SIZE, (state.page + 1) * PAGE_SIZE) ?? [];
  const pages = Math.ceil((state.result?.totalFindings ?? 0) / PAGE_SIZE);
  const readyForHandoff = handoffIsReady(state, lane);
  const handoffReason = state.phase === "handoff"
    ? "正在创建治理任务"
    : readyForHandoff
      ? "候选、关系和证据已具备治理复核条件"
      : state.phase === "insufficient"
        ? "当前标签信号不足，暂不能移交"
        : state.phase === "fitting"
          ? "正在完成少样本网络适配"
          : lane === "zero_shot"
          ? "完成网络分析后可进入治理应用"
          : "完成网络适配后可进入治理应用";
  const transferEvidence = useMemo(() => state.result && state.registration
    ? buildAdaptationTransferEvidence({
      result: state.result,
      registration: state.registration,
      modelCardState,
      selectedNodeId: state.selectedNodeId,
      selectedEvidence: state.evidence,
      policy: state.policy,
      comparison: state.comparison,
    })
    : null, [modelCardState, state.comparison, state.evidence, state.policy, state.registration, state.result, state.selectedNodeId]);
  const transferEvidenceReady = transferEvidence?.status === "ready";
  const transferEvidenceTitle = transferEvidenceReady
    ? "查看迁移依据"
    : transferEvidence?.status === "unavailable" && transferEvidence.reason === "model_card_loading"
      ? "正在核对模型凭证"
      : "迁移来源凭证暂不可用";
  const messageIsWarning = state.phase === "insufficient";
  const messageIsError = state.phase === "error" || state.phase === "calibration_error" || Boolean(state.message?.includes("未完成") || state.message?.includes("不可用"));
  return <section ref={laneRef} tabIndex={-1} className={`adaptation-lane is-${lane === "zero_shot" ? "zero" : "few"}`} role="region" aria-label={lane === "zero_shot" ? "零样本路径" : "少样本路径"} data-phase={state.phase} data-registration-id={state.registration?.registrationId} data-task-id={state.registration?.task.taskId} data-artifact-id={state.registration?.artifact.artifactId} data-inference-hash={state.registration?.artifact.bundleSha256} data-graph-version-hash={state.registration?.artifact.graphVersionHash} data-target-receipt-hash={state.registration?.targetReceipt.receiptHash} data-registration-hash={state.registration?.registrationHash} data-outer-bundle-hash={state.registration?.outerBundleSha256} data-node-set-hash={state.registration?.targetReceipt.nodeSetSha256} data-run-id={state.run?.runId} data-result-hash={state.result?.resultHash} data-selected-node-id={state.selectedNodeId ?? undefined} data-overlay={state.overlay} data-artifact-epoch={state.epochs.artifact} data-run-epoch={state.epochs.run} data-policy-epoch={state.epochs.policy} data-graph-epoch={state.epochs.graph} data-camera-epoch={state.epochs.camera} data-focus-epoch={state.epochs.focus} data-abort-epoch={state.epochs.abort}>
    <header><span>{lane === "zero_shot" ? "01" : "02"}</span><div><small>{lane === "zero_shot" ? "目标网络分析" : "少量标注适配"}</small><h3>{lane === "zero_shot" ? "零样本网络分析" : "少样本网络适配"}</h3></div><ShieldCheck size={24} /></header>
    <p>{lane === "zero_shot" ? "在无可靠标签条件下识别协同组群与待治理核验账号。" : "使用少量已核对标签适配目标网络，基础模型保持不变。"}</p>
    <div className="adaptation-lane__handoff"><button type="button" className="is-primary" disabled={!readyForHandoff || state.phase === "handoff"} aria-describedby={`${lane}-handoff-reason`} onClick={() => void handoff()}>{state.phase === "handoff" ? <CircleNotch className="spin" size={17} /> : <ShieldCheck size={17} />}进入治理应用</button><small id={`${lane}-handoff-reason`}>{handoffReason}</small></div>
    <label className="adaptation-file-action"><input type="file" accept=".sgtask.zip,application/zip" aria-label={lane === "zero_shot" ? "零样本目标任务包" : "少样本目标任务包"} onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ""; void upload(file); }} /><FileArrowUp size={16} />选择目标任务包</label>
    {state.phase === "registering" ? <p role="status"><CircleNotch className="spin" size={16} />正在登记目标任务</p> : null}
    {state.registration ? <div className="adaptation-task-identity"><strong>{state.registration.task.displayName}</strong><span>{state.registration.task.nodeCount} 个对象 · {state.registration.task.fusedEdgeCount} 条关系</span></div> : null}
    {lane === "few_shot" && labels && !labelsReady ? <p className="adaptation-workspace__notice is-error" role="alert"><WarningCircle size={16} />少样本标签未通过完整性校验，请核对目标任务包。</p> : null}
    {state.message ? <p role={messageIsError || messageIsWarning ? "alert" : "status"} className={messageIsError ? "adaptation-workspace__notice is-error" : messageIsWarning ? "adaptation-workspace__notice" : undefined}>{messageIsError || messageIsWarning ? <WarningCircle size={16} /> : <CheckCircle size={16} />}{state.message}</p> : null}
    {state.phase === "raw" ? <button type="button" className="is-primary" onClick={() => setState((value) => ({ ...value, phase: "confirm_run" }))}>开始分析</button> : null}
    {state.phase === "confirm_run" ? <div className="adaptation-confirm" role="group" aria-label="网络分析确认"><span>{lane === "zero_shot" ? "将分析当前目标网络并识别协同组群。" : "将分析当前目标网络并完成少样本适配。"}</span><button type="button" onClick={() => void run()}>确认分析</button><button type="button" onClick={() => setState((value) => ({ ...value, phase: "raw" }))}>取消</button></div> : null}
    {state.phase === "running" ? <p role="status"><CircleNotch className="spin" size={16} />正在分析目标网络</p> : null}
    {state.result ? <p className="adaptation-analysis-status"><CheckCircle size={16} />协同组群已就绪 · {state.result.totalFindings} 个账号</p> : null}
    {transferEvidence ? <button type="button" className="adaptation-transfer-trigger" disabled={!transferEvidenceReady} title={transferEvidenceTitle} aria-label={transferEvidenceReady ? "查看迁移依据" : `查看迁移依据：${transferEvidenceTitle}`} onClick={() => setTransferEvidenceOpen(true)}><ShieldCheck size={17} />查看迁移依据</button> : null}
    {state.phase === "fitting" ? <p role="status"><CircleNotch className="spin" size={16} />正在完成少样本网络适配</p> : null}
    {lane === "few_shot" && state.phase === "calibration_error" ? <button type="button" className="adaptation-retry" onClick={() => void retryOrganization()}>重试适配</button> : null}
    {state.result && state.phase !== "fitting" ? <section className="adaptation-candidates" aria-label="重点账号"><header><strong>重点账号</strong><span>{state.result.totalFindings} 个</span></header><div>{candidates.map((finding) => { const label = graphLabelById.get(finding.nodeId) ?? "匿名账号"; return <button type="button" key={finding.nodeId} className={state.selectedNodeId === finding.nodeId ? "is-selected" : undefined} aria-label={`${label}，待治理核验`} aria-pressed={state.selectedNodeId === finding.nodeId} onClick={() => void select(finding.nodeId)}><strong>{label}</strong><small>待治理核验</small></button>; })}</div></section> : null}
    {state.result && pages > 1 ? <nav className="adaptation-pagination" aria-label="账号分页"><button type="button" disabled={state.page === 0} onClick={() => setState((value) => ({ ...value, page: value.page - 1 }))}>上一页</button><span>{state.page + 1} / {pages}</span><button type="button" disabled={state.page + 1 >= pages} onClick={() => setState((value) => ({ ...value, page: value.page + 1 }))}>下一页</button></nav> : null}
    {state.evidence ? <section className="adaptation-evidence" aria-label="直接关系证据"><h4>直接关系证据</h4><p>{state.evidence.neighbors.length} 个直接关联对象</p></section> : null}
    <AdaptationTransferEvidenceDialog open={transferEvidenceOpen && transferEvidence?.status === "ready"} lane={lane} evidence={transferEvidence?.status === "ready" ? transferEvidence.value : null} nodeCount={state.registration?.task.nodeCount ?? 0} relationCount={state.registration?.task.fusedEdgeCount ?? 0} onClose={() => setTransferEvidenceOpen(false)} />
  </section>;
}

export function GovernanceWorkbench({ client, modelCardState, onGraphChange, onOverlayChange, onFocusChange, onCameraFocusChange, onGovernanceHandoff, onLanePresentationChange, onActiveLaneChange, onClose }: GovernanceWorkbenchProps) {
  const cameraTokenRef = useRef(0);
  const zeroLaneRef = useRef<HTMLElement>(null);
  const fewLaneRef = useRef<HTMLElement>(null);
  const nextCameraToken = () => ++cameraTokenRef.current;
  return <section className="adaptation-workspace" aria-label="适配能力工作区">
    <AdaptationGuide zeroLaneRef={zeroLaneRef} fewLaneRef={fewLaneRef} onClose={onClose} />
    <div className="adaptation-lanes">
      <Lane lane="zero_shot" laneRef={zeroLaneRef} client={client} modelCardState={modelCardState} nextCameraToken={nextCameraToken} onGraphChange={onGraphChange} onOverlayChange={onOverlayChange} onFocusChange={onFocusChange} onCameraFocusChange={onCameraFocusChange} onGovernanceHandoff={onGovernanceHandoff} onLanePresentationChange={onLanePresentationChange} onActiveLaneChange={onActiveLaneChange} />
      <Lane lane="few_shot" laneRef={fewLaneRef} client={client} modelCardState={modelCardState} nextCameraToken={nextCameraToken} onGraphChange={onGraphChange} onOverlayChange={onOverlayChange} onFocusChange={onFocusChange} onCameraFocusChange={onCameraFocusChange} onGovernanceHandoff={onGovernanceHandoff} onLanePresentationChange={onLanePresentationChange} onActiveLaneChange={onActiveLaneChange} />
    </div>
  </section>;
}

export { GovernanceWorkbench as AdaptationWorkspace };
