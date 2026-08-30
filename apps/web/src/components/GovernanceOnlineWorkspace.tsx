import {
  Archive,
  ArrowCounterClockwise,
  Brain,
  CheckCircle,
  CircleNotch,
  ClockCounterClockwise,
  DownloadSimple,
  FileText,
  Graph,
  Link,
  MagnifyingGlass,
  Plus,
  ShieldCheck,
  TreeStructure,
  UsersThree,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

import { SocialGraphApiError } from "../services/apiClient";
import { createGraphVersion } from "../services/graphImport";
import { governanceAccountLabel, governanceLimitationLabel, governanceModalityLabel } from "../services/governancePresentation";
import { buildGovernanceOnlineGovernanceOverlay } from "../services/governanceOverlay";
import {
  buildAdaptedReviewPriorityOverlay,
  revalidateGovernanceAdaptation,
  sortDerivationsByAdaptedRank,
  sortFindingsByAdaptedRank,
  withAdaptedRankPresentation,
  type GovernanceAdaptationState,
  type ValidatedGovernanceAdaptationState,
} from "../services/governanceAdaptation";
import { governanceExactRelationKey } from "../services/graphPreviewPolicy";
import {
  governanceProjectionSpec,
  projectGovernanceGraph,
  type GovernanceGraphLens,
  type GovernanceProjectionSpec,
} from "../services/governanceReadableProjection";
import type { AnalysisOverlay, GovernanceFocus, GraphNode, GraphVersion } from "../types/graph";
import type { GovernanceSkillsClientLike, GovernanceSkillsContext } from "../types/governanceSkills";
import {
  type GovernanceArtifact,
  type GovernanceCase,
  type GovernanceCaseStatus,
  type GovernanceDerivation,
  type GovernanceOnlineCapabilities,
  type GovernanceOnlineClientLike,
  type GovernanceOnlineEvidence,
  type GovernanceOnlineFinding,
  type GovernanceOnlinePreview,
  type GovernanceProjectionRequest,
  type GovernanceOnlineResult,
  type GovernanceOnlineRun,
  type GovernanceReviewDecision,
  type GovernanceTargetKind,
} from "../types/governanceOnline";
import {
  GOVERNANCE_WORKSPACE_SCHEMA,
  type GovernanceWorkspaceSnapshot,
} from "../services/governanceWorkspaceStore";
import { EvidenceDossier } from "./EvidenceDossier";

type AsyncState<T> =
  | { readonly state: "idle" | "loading" }
  | { readonly state: "ready"; readonly value: T }
  | { readonly state: "error"; readonly message: string };
type LeftView = "findings" | "groups" | "relations" | "cases";
type RelationView = "factual" | "clues";
type WorkspaceMode = "candidates" | "relations" | "cases";
type CandidateReviewView = "pending" | "resolved";
type Lens = GovernanceGraphLens;
type SelectedTarget = { readonly kind: GovernanceTargetKind; readonly id: string; readonly nodeIds: readonly string[] };
type SelectedSourceFile = { readonly name: string; readonly artifactId: string | null };

const STAGE_META = [
  ["validating", "输入检查"],
  ["preprocessing", "准备图谱"],
  ["inferencing", "风险推理"],
  ["deriving", "生成研判"],
  ["freezing", "保存结果"],
] as const;
const CASE_NEXT: Readonly<Partial<Record<GovernanceCaseStatus, GovernanceCaseStatus>>> = {
  draft: "active", active: "concluded", concluded: "archived", archived: "active",
};
const CASE_ACTION_LABEL: Readonly<Record<GovernanceCaseStatus, string>> = {
  draft: "开始研判", active: "形成结论", concluded: "归档", archived: "重新打开",
};
const CASE_STATUS_LABEL: Readonly<Record<GovernanceCaseStatus, string>> = {
  draft: "草稿", active: "研判中", concluded: "已形成结论", archived: "已归档",
};
const REVIEW_DECISION_LABEL: Readonly<Record<GovernanceReviewDecision, string>> = {
  confirmed: "确认", rejected: "驳回", pending: "待定",
};
const DEFAULT_PROJECTION_REQUEST: GovernanceProjectionRequest = Object.freeze({
  preset: "overview",
  nodeBudget: 120,
  edgeBudget: 240,
});
const EMPTY_NODE_IDS: readonly string[] = Object.freeze([]);
const GOVERNANCE_GRAPH_PROJECTION_VERSION = "governance-preview/2.1";

function describeError(error: unknown): string {
  if (error instanceof SocialGraphApiError) return error.message;
  if (error instanceof Error && error.message === "GFM_GOVERNANCE_RESPONSE_INVALID") return "服务响应未通过一致性校验。";
  return "本机在线治理请求未完成；现有图与复核记录未被修改。";
}

function stageIndex(stage: GovernanceOnlineRun["stage"] | undefined): number {
  if (!stage || stage === "queued") return -1;
  if (stage === "completed") return STAGE_META.length;
  return STAGE_META.findIndex(([id]) => id === stage);
}
function riskLabel(band: GovernanceOnlineFinding["riskBand"]): string {
  return band === "high" ? "高风险候选" : band === "review" ? "建议复核" : "低风险参照";
}
function runStatusLabel(status: GovernanceOnlineRun["status"]): string {
  if (status === "succeeded") return "已完成";
  if (status === "queued") return "排队中";
  if (status === "running") return "进行中";
  if (status === "cancelled") return "已取消";
  if (status === "interrupted") return "已中断";
  return "失败";
}
function caseItemDecision(item: GovernanceCase["items"][number], caseItem: GovernanceCase): GovernanceReviewDecision | undefined {
  return caseItem.currentDecisions[`${item.targetType}:${item.targetId}`] ?? caseItem.currentDecisions[item.targetId];
}
function caseNextStep(caseItem: GovernanceCase, reviewedCount: number): string {
  if (caseItem.state === "concluded") return "结论已形成，可以导出报告或继续归档。";
  if (caseItem.state === "archived") return "研判单已归档；需要继续核查时可重新打开。";
  if (!caseItem.items.length) return "先选择节点、群组或关系，并将其加入当前研判单。";
  if (caseItem.state === "draft") return "点击“开始研判”，再对单内对象提交人工结论。";
  const remaining = caseItem.items.length - reviewedCount;
  return remaining > 0 ? `继续复核 ${remaining} 个尚未形成结论的对象。` : "全部对象已复核，可以点击“形成结论”。";
}


function buildGovernanceGraphVersion(
  preview: GovernanceOnlinePreview,
  findings: readonly GovernanceOnlineFinding[],
  sourceName: string,
): GraphVersion {
  const findingById = new Map(findings.map((finding) => [finding.nodeId, finding]));
  const projection = createGraphVersion(
    sourceName,
    preview.nodes.map((node) => {
      const finding = findingById.get(node.id);
      return {
        // Use the same semantic type as the conversation graph. Risk is an
        // attention layer, not a replacement for the graph's entity palette.
        id: node.id, label: governanceAccountLabel(node.label, node.id), type: "账号",
        attributes: Object.freeze({
          degree: node.degree, structureMissing: node.structureMissing, score: node.score ?? finding?.score ?? null,
          rank: finding?.rank ?? null,
          riskBand: node.riskBand ?? finding?.riskBand ?? null, groupId: node.groupId ?? finding?.communityId ?? null,
          primaryExpert: finding?.routes.find((route) => route.expert !== "shared")?.expert ?? null,
        }),
      };
    }),
    preview.edges.map((edge) => ({
      id: edge.id, source: edge.source, target: edge.target,
      type: "factual_relation", directed: false, weight: 1,
      attributes: Object.freeze({ modalities: Object.freeze([...edge.modalities]), factual: edge.factual }),
    })),
    [{
      code: "governance_projection", severity: "info",
      message: preview.partialPreview
        ? "画布为有界投影；在线推理与治理派生均基于完整后端图。"
        : "画布投影与在线推理绑定同一不可变图身份。",
      details: { nodeCount: preview.nodeCount, edgeCount: preview.edgeCount },
    }],
    { provenance: { origin: "research_dataset", pipeline: "dataset-artifact", pipelineVersion: GOVERNANCE_GRAPH_PROJECTION_VERSION, sourceHashScheme: "dataset-content-hash-v2" } },
  );
  return Object.freeze({
    ...projection,
    preview: Object.freeze({ ...projection.preview, truncated: preview.partialPreview, originalNodeCount: preview.nodeCount, originalEdgeCount: preview.edgeCount }),
    truncated: preview.partialPreview,
    datasetArtifact: Object.freeze({
      id: preview.artifactId, datasetName: sourceName, checksum: preview.previewHash,
      canonicalGraphHash: preview.graphVersionHash, contentHash: preview.datasetContentHash, scope: preview.partialPreview ? "projection" : "complete",
    }),
  });
}

/** A user upload is a new auditable browser revision, even when its factual
 * content matches a prior upload. SocialGraph-FM Governance content identity remains bound in
 * datasetArtifact and must not replace the GraphVersion UUID. */
export function governanceArtifactDisplayName(
  artifact: Pick<GovernanceArtifact, "datasetId" | "displayName"> | null | undefined,
  sourceFileName: string | null | undefined,
): string {
  const registered = artifact?.displayName?.trim() || artifact?.datasetId?.trim();
  const internalSource = /(?:\bgovernance-artifact-[0-9a-f]{32}\b|\banswer[\s_-]*pack\b|\b(?:russia|cuba|uae|venezuela|iran|china)(?:[\s_.-]|$))/iu;
  if (registered && !internalSource.test(registered)) return registered;
  const fileName = sourceFileName?.trim() ?? "";
  const stem = fileName.replace(/\.[^.]+$/u, "");
  if (!stem || internalSource.test(stem)) return "当前会话治理图";
  return stem.replace(/[-_]+/gu, " ").trim();
}

export function governanceImportedGraphVersion(preview: GovernanceOnlinePreview, sourceName = "治理数据"): GraphVersion {
  return buildGovernanceGraphVersion(preview, [], sourceName);
}

/** Stable presentation identity for the non-persisted governance canvas. */
export function governancePreviewGraph(preview: GovernanceOnlinePreview, findings: readonly GovernanceOnlineFinding[], sourceName = "治理数据"): GraphVersion {
  const projection = buildGovernanceGraphVersion(preview, findings, sourceName);
  return Object.freeze({
    ...projection,
    id: `governance-view:${preview.datasetContentHash}:${GOVERNANCE_GRAPH_PROJECTION_VERSION}`,
  });
}

export interface GovernanceOnlineWorkspaceProps {
  readonly client: GovernanceOnlineClientLike;
  readonly onGraphPresentationChange: (presentation: GovernanceGraphPresentation | null) => void;
  readonly ragOpen: boolean;
  readonly onRagOpenChange: (open: boolean) => void;
  /** Mounted inside the stable candidate region so opening the assistant never
   * inserts a new row above the workbench. */
  readonly assistantPanel?: ReactNode;
  /** Optional LLM organizer for node evidence. Structured evidence remains
   * available when this service is absent or unavailable. */
  readonly evidenceSummaryClient?: GovernanceSkillsClientLike;
  /** App-level binding shared with the conversation workspace. */
  readonly sessionId?: string;
  readonly sharedSnapshot?: GovernanceWorkspaceSnapshot | null;
  /** A conversation upload is still being verified and will bind here when ready. */
  readonly sharedUploadPendingFileName?: string | null;
  /** Visible recovery feedback from the App-level session binding. */
  readonly sharedRestoreMessage?: string | null;
  readonly onSharedSnapshotChange?: (snapshot: GovernanceWorkspaceSnapshot) => void;
  readonly onGlobalResultAccepted?: (run: GovernanceOnlineRun, result: GovernanceOnlineResult) => void;
  /** Retained so the shell can identify its compact route without creating a
   * second evidence surface. */
  /** Immutable handoff coordinates. The workspace reloads every live record
   * before allowing an adapted priority to affect governance. */
  readonly adaptation?: GovernanceAdaptationState;
  readonly adaptationValidationToken?: number;
}

export interface GovernanceGraphPresentation {
  readonly graph: GraphVersion | null;
  readonly selectedNodeId: string | null;
  readonly focusNodeIds: readonly string[];
  readonly activeOverlay: AnalysisOverlay | null;
  readonly lens: GovernanceGraphLens;
  readonly projectionSpec: GovernanceProjectionSpec;
  readonly focus?: GovernanceFocus;
  readonly skillsContext: GovernanceSkillsContext | null;
  readonly onSelectNode: (node: GraphNode | null) => void;
}

export function GovernanceOnlineWorkspace({
  client,
  onGraphPresentationChange,
  ragOpen,
  onRagOpenChange,
  assistantPanel,
  evidenceSummaryClient,
  sessionId = "",
  sharedSnapshot = null,
  sharedUploadPendingFileName = null,
  sharedRestoreMessage = null,
  onSharedSnapshotChange,
  onGlobalResultAccepted,
  adaptation,
  adaptationValidationToken = 0,
}: GovernanceOnlineWorkspaceProps) {
  const [capabilities, setCapabilities] = useState<AsyncState<GovernanceOnlineCapabilities>>({ state: "loading" });
  const [artifact, setArtifact] = useState<AsyncState<GovernanceArtifact>>({ state: "idle" });
  const [preview, setPreview] = useState<AsyncState<GovernanceOnlinePreview>>({ state: "idle" });
  const [run, setRun] = useState<GovernanceOnlineRun | null>(null);
  const [result, setResult] = useState<GovernanceOnlineResult | null>(null);
  const [findings, setFindings] = useState<readonly GovernanceOnlineFinding[]>([]);
  const [groups, setGroups] = useState<readonly GovernanceDerivation[]>([]);
  const [relations, setRelations] = useState<readonly GovernanceDerivation[]>([]);
  const [links, setLinks] = useState<readonly GovernanceDerivation[]>([]);
  const [history, setHistory] = useState<readonly GovernanceOnlineRun[]>([]);
  const [historyFeedback, setHistoryFeedback] = useState<string | null>(null);
  const [cases, setCases] = useState<readonly GovernanceCase[]>([]);
  const [activeCaseId, setActiveCaseId] = useState<string | null>(null);
  const [selected, setSelected] = useState<SelectedTarget | null>(null);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [focus, setFocus] = useState<GovernanceFocus | undefined>(undefined);
  const focusRef = useRef<GovernanceFocus | undefined>(focus);
  focusRef.current = focus;
  const cameraTokenRef = useRef(0);
  const [evidence, setEvidence] = useState<AsyncState<GovernanceOnlineEvidence>>({ state: "idle" });
  const [leftView, setLeftView] = useState<LeftView>("findings");
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("candidates");
  const [relationView, setRelationView] = useState<RelationView>("factual");
  const [candidateReviewView, setCandidateReviewView] = useState<CandidateReviewView>("pending");
  const [visibleCandidateCount, setVisibleCandidateCount] = useState(50);
  const [lens, setLens] = useState<Lens>("risk");
  const [selectedSourceFile, setSelectedSourceFile] = useState<SelectedSourceFile | null>(null);
  const [reviewReason, setReviewReason] = useState("");
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [validatedAdaptation, setValidatedAdaptation] = useState<AsyncState<ValidatedGovernanceAdaptationState>>({ state: "idle" });
  const validatedAdaptationKeyRef = useRef<string | null>(null);
  const adaptationValidationEpochRef = useRef(0);
  const sourceEpochRef = useRef(0);
  const restoredSnapshotKeyRef = useRef<string | null>(null);
  const restoredArtifactIdentityRef = useRef<string | null>(null);
  const emittedSnapshotKeyRef = useRef<string | null>(null);
  const previousSessionIdRef = useRef(sessionId);
  const [boundSessionId, setBoundSessionId] = useState(sessionId);
  const bodyRef = useRef<HTMLElement | null>(null);
  const adaptationValidationKey = adaptation && sharedSnapshot
    ? [
      adaptation.lane,
      adaptation.registration.registrationId,
      adaptation.registration.registrationHash,
      adaptation.handoff?.handoffHash ?? "",
      adaptation.policy?.policyHash ?? "",
      adaptation.comparison?.comparisonHash ?? "",
      sharedSnapshot.artifact.artifactHash,
      sharedSnapshot.run?.statusHash ?? "",
      sharedSnapshot.result?.resultHash ?? "",
      adaptationValidationToken,
    ].join(":")
    : null;

  const readyCapabilities = capabilities.state === "ready" ? capabilities.value : null;
  const readyArtifact = artifact.state === "ready" ? artifact.value : null;
  const readyPreview = preview.state === "ready" ? preview.value : null;
  const activeCase = cases.find((item) => item.caseId === activeCaseId) ?? cases[0] ?? null;
  const reviewedCaseItemCount = activeCase?.items.filter((item) => Boolean(caseItemDecision(item, activeCase))).length ?? 0;
  const currentFinding = selected?.kind === "node" ? findings.find((item) => item.nodeId === selected.id) ?? null : null;
  const currentDerivation = selected?.kind === "group"
    ? groups.find((item) => item.id === selected.id) ?? null
    : selected?.kind === "relation"
      ? relations.find((item) => item.id === selected.id) ?? links.find((item) => item.id === selected.id) ?? null
      : null;
  const skillsContext = useMemo<GovernanceSkillsContext | null>(() => readyArtifact
    && readyCapabilities?.modelVersionId
    && readyCapabilities.modelStateHash
    ? Object.freeze({
      graph: Object.freeze({
        artifactId: readyArtifact.artifactId,
        datasetContentHash: readyArtifact.datasetContentHash,
        graphVersionHash: readyArtifact.graphVersionHash,
      }),
      model: Object.freeze({
        modelVersionId: readyCapabilities.modelVersionId,
        modelStateHash: readyCapabilities.modelStateHash,
      }),
      ...(run?.runId ? { runId: run.runId } : {}),
      ...(activeCase?.caseId ? { caseId: activeCase.caseId } : {}),
      ...(activeCase?.caseHash ? { caseHash: activeCase.caseHash } : {}),
      caseItemCount: activeCase?.items.length ?? 0,
      ...(selected?.nodeIds.length ? { selectedNodeIds: Object.freeze([...new Set(selected.nodeIds)].sort()) } : {}),
      ...(selected ? { selectedTarget: Object.freeze({ kind: selected.kind, targetId: selected.id }) } : {}),
    })
    : null, [activeCase?.caseHash, activeCase?.caseId, activeCase?.items.length, readyArtifact, readyCapabilities?.modelStateHash, readyCapabilities?.modelVersionId, run?.runId, selected]);

  useEffect(() => {
    const controller = new AbortController();
    client.capabilities(controller.signal).then((value) => setCapabilities({ state: "ready", value })).catch((error) => { if (!controller.signal.aborted) setCapabilities({ state: "error", message: describeError(error) }); });
    client.listRuns(controller.signal).then(setHistory).catch(() => undefined);
    return () => controller.abort();
  }, [client]);

  const clearBoundRun = useCallback(() => {
    setRun(null); setResult(null); setFindings([]); setGroups([]); setRelations([]); setLinks([]); setCases([]); setActiveCaseId(null); setHistoryFeedback(null); setSelected(null); setEvidenceOpen(false); setEvidence({ state: "idle" }); setReviewReason(""); setWorkspaceMode("candidates"); setCandidateReviewView("pending");
  }, []);

  useEffect(() => {
    if (previousSessionIdRef.current === sessionId) return;
    previousSessionIdRef.current = sessionId;
    sourceEpochRef.current += 1;
    restoredSnapshotKeyRef.current = null;
    restoredArtifactIdentityRef.current = null;
    emittedSnapshotKeyRef.current = null;
    clearBoundRun();
    setArtifact({ state: "idle" });
    setPreview({ state: "idle" });
    setSelectedSourceFile(null);
    setNotice(null);
    setBoundSessionId(sessionId);
  }, [clearBoundRun, sessionId]);

  const evidenceNodeId = selected?.kind === "node" ? selected.id : null;
  const evidenceRequestKey = evidenceOpen && run?.status === "succeeded" && result && evidenceNodeId
    ? `${run.runId}:${result.resultHash}:${evidenceNodeId}`
    : null;
  useEffect(() => {
    if (!evidenceRequestKey || !run || !result || !evidenceNodeId) { setEvidence({ state: "idle" }); return; }
    const expected = {
      runId: run.runId,
      resultHash: result.resultHash,
      artifactId: result.artifactId,
      datasetContentHash: result.datasetContentHash,
      graphVersionHash: result.graphVersionHash,
      modelVersionId: result.modelVersionId,
      modelVersionHash: result.modelVersionHash,
      modelStateHash: result.modelStateHash,
      threshold: result.threshold,
      nodeId: evidenceNodeId,
    };
    const controller = new AbortController(); setEvidence({ state: "loading" });
    client.evidence(expected.runId, expected.nodeId, controller.signal).then((value) => {
      if (value.runId !== expected.runId || value.resultHash !== expected.resultHash || value.artifactId !== expected.artifactId || value.datasetContentHash !== expected.datasetContentHash || value.graphVersionHash !== expected.graphVersionHash || value.modelVersionId !== expected.modelVersionId || value.modelVersionHash !== expected.modelVersionHash || value.modelStateHash !== expected.modelStateHash || value.threshold !== expected.threshold || value.node.nodeId !== expected.nodeId) throw new Error("GFM_GOVERNANCE_RESPONSE_INVALID");
      setEvidence({ state: "ready", value });
    }).catch((error) => { if (!controller.signal.aborted) setEvidence({ state: "error", message: describeError(error) }); });
    return () => controller.abort();
  }, [client, evidenceRequestKey]);

  const sourceDisplayName = governanceArtifactDisplayName(readyArtifact, selectedSourceFile?.name);
  const fullGraph = useMemo(() => readyPreview ? governancePreviewGraph(readyPreview, findings, sourceDisplayName) : null, [findings, readyPreview, sourceDisplayName]);
  const selectedNodeId = selected?.kind === "node" ? selected.id : null;
  const focusNodeIds = focus?.nodeIds ?? EMPTY_NODE_IDS;
  const projectionSpec = useMemo(() => governanceProjectionSpec(lens, false), [lens]);
  const graph = useMemo(
    () => fullGraph ? projectGovernanceGraph(fullGraph, projectionSpec, EMPTY_NODE_IDS) : null,
    [fullGraph, projectionSpec],
  );
  const adaptationComparison = validatedAdaptation.state === "ready"
    && validatedAdaptationKeyRef.current === adaptationValidationKey
    && validatedAdaptation.value.lane === "few_shot"
    ? validatedAdaptation.value.comparison ?? null
    : null;
  const adaptationRows = useMemo(
    () => new Map(adaptationComparison?.rows.map((row) => [row.nodeId, row]) ?? []),
    [adaptationComparison],
  );
  const referenceLabels = useMemo(() => validatedAdaptation.state === "ready"
    && validatedAdaptationKeyRef.current === adaptationValidationKey
    && validatedAdaptation.value.lane === "few_shot"
    && validatedAdaptation.value.registration.labels
    ? Object.freeze(Object.fromEntries(validatedAdaptation.value.registration.labels.labels.map((row) => [row.nodeId, row.label])))
    : Object.freeze({}), [adaptationValidationKey, validatedAdaptation]);
  const graphReviewDecisions = useMemo(() => {
    if (!activeCase) return Object.freeze({});
    const values: Record<string, GovernanceReviewDecision> = {};
    for (const [key, decision] of Object.entries(activeCase.currentDecisions)) {
      const [kind, targetId] = key.includes(":") ? key.split(":", 2) : ["node", key];
      if (kind === "node" && targetId) values[targetId] = decision;
    }
    return Object.freeze(values);
  }, [activeCase]);
  const overlay = useMemo(() => {
    if (!graph || !run || !result) return null;
    const base = buildGovernanceOnlineGovernanceOverlay(graph, lens, findings, relations, links, run, result);
    const ranked = !adaptationComparison ? base : lens === "risk" || lens === "relations"
      ? buildAdaptedReviewPriorityOverlay(graph, result, adaptationComparison, lens, base)
      : withAdaptedRankPresentation(base, adaptationComparison);
    return Object.freeze({
      ...ranked,
      presentation: Object.freeze({
        ...ranked.presentation,
        ...(Object.keys(referenceLabels).length ? { referenceLabels } : {}),
        ...(Object.keys(graphReviewDecisions).length ? { reviewDecisions: graphReviewDecisions } : {}),
      }),
    });
  }, [adaptationComparison, findings, graph, graphReviewDecisions, lens, links, referenceLabels, relations, result, run]);
  const projectionRequest = useMemo<GovernanceProjectionRequest | undefined>(() => {
    if (projectionSpec.preset === "groups") {
      return { preset: "groups", groupBudget: projectionSpec.groupBudget ?? 12 };
    }
    if (projectionSpec.preset === "relation") {
      return {
        preset: "relation",
        nodeBudget: projectionSpec.nodeBudget,
        edgeBudget: projectionSpec.edgeBudget,
      };
    }
    return DEFAULT_PROJECTION_REQUEST;
  }, [projectionSpec]);
  const projectionRequestKey = JSON.stringify(projectionRequest ?? null);
  const previewRunId = run?.status === "succeeded" && result ? run.runId : null;
  const analysisComplete = run?.status === "succeeded" && Boolean(result);
  const acceptedGlobalResultRef = useRef<string | null>(null);

  useEffect(() => {
    if (!run || run.status !== "succeeded" || !result || !readyArtifact) return;
    if (run.runId !== result.runId
      || run.graphVersionHash !== result.graphVersionHash
      || result.graphVersionHash !== readyArtifact.graphVersionHash) return;
    const key = `${result.runId}\u0000${result.resultHash}`;
    if (acceptedGlobalResultRef.current === key) return;
    acceptedGlobalResultRef.current = key;
    onGlobalResultAccepted?.(run, result);
  }, [onGlobalResultAccepted, readyArtifact, result, run]);

  useEffect(() => {
    if (!readyArtifact) return;
    const controller = new AbortController();
    const request = previewRunId
      ? client.runPreview(previewRunId, controller.signal, projectionRequest)
      : client.preview(readyArtifact.artifactId, controller.signal, projectionRequest);
    request.then((nextPreview) => {
      if (nextPreview.artifactId !== readyArtifact.artifactId
        || nextPreview.datasetContentHash !== readyArtifact.datasetContentHash
        || nextPreview.graphVersionHash !== readyArtifact.graphVersionHash
        || projectionRequest && nextPreview.preset !== undefined && nextPreview.preset !== projectionRequest.preset
        || previewRunId && (nextPreview.runId !== previewRunId || nextPreview.resultHash !== result?.resultHash)) {
        throw new Error("GFM_GOVERNANCE_RESPONSE_INVALID");
      }
      setPreview({ state: "ready", value: nextPreview });
    }).catch((error) => {
      if (controller.signal.aborted) return;
      const message = describeError(error);
      setPreview((current) => current.state === "ready" ? current : { state: "error", message });
    });
    return () => controller.abort();
  }, [client, previewRunId, projectionRequestKey, readyArtifact, result?.resultHash]);
  const selectTarget = useCallback((next: SelectedTarget) => {
    // Selection drives appearance only. It never opens evidence, changes the
    // graph projection, or requests a camera movement.
    setEvidence({ state: "idle" });
    setSelected(next);
    setEvidenceOpen(false);
    const nodeIds = [...new Set(next.nodeIds)];
    const relation = next.kind === "relation"
      ? relations.find((candidate) => candidate.id === next.id) ?? links.find((candidate) => candidate.id === next.id)
      : null;
    const source = relation?.source ?? nodeIds[0];
    const target = relation?.target ?? nodeIds[1];
    const exactRelationKey = source && target && relation
      ? governanceExactRelationKey(source, target, relation.modalities)
      : undefined;
    const nextFocus = Object.freeze({
      kind: next.kind,
      targetId: next.id,
      nodeIds: Object.freeze(nodeIds),
      ...(exactRelationKey ? { exactRelationKey } : {}),
      cameraToken: ++cameraTokenRef.current,
    });
    focusRef.current = nextFocus;
    setFocus(nextFocus);
  }, [links, relations]);
  const resolveCaseTarget = useCallback((item: GovernanceCase["items"][number]): SelectedTarget => {
    if (item.targetType === "node") return { kind: "node", id: item.targetId, nodeIds: [item.targetId] };
    const derivation = item.targetType === "group"
      ? groups.find((candidate) => candidate.id === item.targetId)
      : relations.find((candidate) => candidate.id === item.targetId) ?? links.find((candidate) => candidate.id === item.targetId);
    return { kind: item.targetType, id: item.targetId, nodeIds: derivation?.nodeIds ?? [] };
  }, [groups, links, relations]);
  const handleCaseSelect = useCallback((nextCase: GovernanceCase) => {
    setActiveCaseId(nextCase.caseId);
    if (activeCaseId !== nextCase.caseId) setReviewReason("");
    const selectedBelongsToCase = Boolean(selected && nextCase.items.some((item) => item.targetType === selected.kind && item.targetId === selected.id));
    if (!selectedBelongsToCase && nextCase.items[0]) selectTarget(resolveCaseTarget(nextCase.items[0]));
    else if (!selectedBelongsToCase) { setSelected(null); setEvidenceOpen(false); setEvidence({ state: "idle" }); }
  }, [activeCaseId, resolveCaseTarget, selectTarget, selected]);
  const clearTransientFocus = useCallback(() => {
    if (!focusRef.current) return;
    focusRef.current = undefined;
    setFocus(undefined);
  }, []);
  const clearSelectedTarget = useCallback(() => {
    clearTransientFocus();
    setSelected(null);
    setEvidenceOpen(false);
    setEvidence({ state: "idle" });
  }, [clearTransientFocus]);
  const closeEvidencePanel = useCallback(() => {
    setEvidenceOpen(false);
  }, []);
  const handleLensChange = useCallback((nextLens: Lens) => {
    setSelected(null);
    setEvidenceOpen(false);
    setEvidence({ state: "idle" });
    setReviewReason("");
    setLens(nextLens);
    onRagOpenChange(false);
    if (nextLens === "risk") setLeftView("findings");
    else if (nextLens === "community") setLeftView("groups");
    else if (nextLens === "relations") {
      setLeftView("relations");
      setRelationView("factual");
    }
  }, [onRagOpenChange]);
  const handleGraphNodeSelect = useCallback((node: GraphNode | null) => {
    if (!node) {
      clearTransientFocus();
      setSelected(null);
      setEvidenceOpen(false);
      setEvidence({ state: "idle" });
      return;
    }
    selectTarget({ kind: "node", id: node.id, nodeIds: [node.id] });
  }, [clearTransientFocus, selectTarget]);

  useEffect(() => {
    if (selected) return;
    setFocus(undefined);
  }, [selected]);

  useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !event.defaultPrevented && !document.querySelector('[role="dialog"][aria-modal="true"]') && !evidenceOpen) {
        clearTransientFocus();
        setSelected(null);
        setEvidence({ state: "idle" });
      }
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [clearTransientFocus, evidenceOpen]);

  useEffect(() => { setVisibleCandidateCount(50); }, [candidateReviewView, leftView, relationView]);

  useEffect(() => {
    onGraphPresentationChange({
      graph,
      selectedNodeId,
      focusNodeIds,
      activeOverlay: overlay,
      lens,
      projectionSpec,
      ...(focus ? { focus } : {}),
      skillsContext,
      onSelectNode: handleGraphNodeSelect,
    });
  }, [
    focusNodeIds,
    focus,
    graph,
    handleGraphNodeSelect,
    analysisComplete,
    lens,
    onGraphPresentationChange,
    overlay,
    projectionSpec,
    selected,
    selectedNodeId,
    skillsContext,
  ]);

  useEffect(() => () => onGraphPresentationChange(null), [onGraphPresentationChange]);

  const hydrateSucceededRun = useCallback(async (
    status: GovernanceOnlineRun,
    signal: AbortSignal,
    expectedSourceEpoch = sourceEpochRef.current,
  ) => {
    const nextResult = await client.result(status.runId, signal);
    if (nextResult.runId !== status.runId || nextResult.requestHash !== status.requestHash || nextResult.artifactId !== status.artifactId || nextResult.datasetContentHash !== status.datasetContentHash || nextResult.graphVersionHash !== status.graphVersionHash || nextResult.modelVersionId !== status.modelVersionId || nextResult.modelVersionHash !== status.modelVersionHash || nextResult.modelStateHash !== status.modelStateHash) {
      throw new Error("GFM_GOVERNANCE_RESPONSE_INVALID");
    }
    const [runArtifact, allFindings, scoredPreview, groupItems, relationItems, linkItems, caseItems] = await Promise.all([
      client.artifact(status.artifactId, signal),
      client.findings(status.runId, 0, Math.min(10_000, nextResult.totalFindings), signal), client.runPreview(status.runId, signal, DEFAULT_PROJECTION_REQUEST),
      client.derivations(status.runId, "group", signal), client.derivations(status.runId, "factual_relation", signal),
      client.derivations(status.runId, "potential_link", signal), client.listCases(status.runId, signal),
    ]);
    if (runArtifact.artifactId !== status.artifactId || runArtifact.datasetContentHash !== status.datasetContentHash || runArtifact.graphVersionHash !== status.graphVersionHash || allFindings.runId !== status.runId || scoredPreview.runId !== status.runId || scoredPreview.resultHash !== nextResult.resultHash || scoredPreview.artifactId !== status.artifactId || scoredPreview.datasetContentHash !== status.datasetContentHash || scoredPreview.graphVersionHash !== status.graphVersionHash) {
      throw new Error("GFM_GOVERNANCE_RESPONSE_INVALID");
    }
    if (signal.aborted || expectedSourceEpoch !== sourceEpochRef.current) return;
    setArtifact({ state: "ready", value: runArtifact });
    setSelectedSourceFile((current) => current?.artifactId === runArtifact.artifactId ? current : null);
    setPreview({ state: "ready", value: scoredPreview });
    setRun(status); setResult(nextResult); setFindings(allFindings.items); setGroups(groupItems); setRelations(relationItems); setLinks(linkItems); setCases(caseItems); setActiveCaseId(caseItems[0]?.caseId ?? null);
    setSelected(null); setEvidenceOpen(false); setEvidence({ state: "idle" }); setWorkspaceMode("candidates");
    setHistory((items) => [status, ...items.filter((item) => item.runId !== status.runId)]);
  }, [client]);

  const sharedSnapshotRestoreKey = sharedSnapshot && sharedSnapshot.sessionId === sessionId
    ? [
      sessionId,
      sharedSnapshot.artifact.artifactHash,
      sharedSnapshot.run?.statusHash ?? "",
      sharedSnapshot.result?.resultHash ?? "",
    ].join(":")
    : null;

  useEffect(() => {
    if (!sharedSnapshot || sharedSnapshotRestoreKey === null) {
      restoredSnapshotKeyRef.current = null;
      return;
    }
    const key = sharedSnapshotRestoreKey;
    if (restoredSnapshotKeyRef.current === key) return;
    restoredSnapshotKeyRef.current = key;
    const artifactIdentity = [
      sharedSnapshot.artifact.artifactId,
      sharedSnapshot.artifact.datasetContentHash,
      sharedSnapshot.artifact.graphVersionHash,
    ].join(":");
    if (restoredArtifactIdentityRef.current !== artifactIdentity) {
      restoredArtifactIdentityRef.current = artifactIdentity;
      sourceEpochRef.current += 1;
    }
    const expectedSourceEpoch = sourceEpochRef.current;
    setBoundSessionId(sessionId);
    setSelectedSourceFile({ name: sharedSnapshot.sourceFileName, artifactId: sharedSnapshot.artifact.artifactId });
    setArtifact({ state: "ready", value: sharedSnapshot.artifact });
    setPreview({ state: "ready", value: sharedSnapshot.preview });
    setRun(sharedSnapshot.run ?? null);
    setResult(sharedSnapshot.result ?? null);
    setActiveCaseId(sharedSnapshot.activeCaseId ?? null);
    if (sharedSnapshot.run?.status === "succeeded") {
      const controller = new AbortController();
      void hydrateSucceededRun(sharedSnapshot.run, controller.signal, expectedSourceEpoch).catch((error) => {
        if (!controller.signal.aborted) setNotice(describeError(error));
      });
      return () => controller.abort();
    }
  }, [hydrateSucceededRun, sessionId, sharedSnapshotRestoreKey]);

  useEffect(() => {
    const validationEpoch = ++adaptationValidationEpochRef.current;
    validatedAdaptationKeyRef.current = null;
    if (!adaptation) {
      setValidatedAdaptation({ state: "idle" });
      return;
    }
    if (!sharedSnapshot || sharedSnapshot.sessionId !== sessionId || adaptationValidationKey === null) {
      setValidatedAdaptation({ state: "error", message: "适配交接已失效；已移除少样本复核顺序，仅保留基础风险排序。" });
      return;
    }
    const controller = new AbortController();
    const expectedKey = adaptationValidationKey;
    setValidatedAdaptation({ state: "loading" });
    void revalidateGovernanceAdaptation(client, adaptation, sharedSnapshot, controller.signal).then((value) => {
      if (controller.signal.aborted || validationEpoch !== adaptationValidationEpochRef.current) return;
      validatedAdaptationKeyRef.current = expectedKey;
      setValidatedAdaptation({ state: "ready", value });
    }).catch((error) => {
      if (controller.signal.aborted || validationEpoch !== adaptationValidationEpochRef.current || error instanceof DOMException && error.name === "AbortError") return;
      validatedAdaptationKeyRef.current = null;
      const stale = error instanceof SocialGraphApiError && error.status === 409;
      setValidatedAdaptation({
        state: "error",
        message: stale
          ? "适配交接已失效；已移除少样本复核顺序，仅保留基础风险排序。请返回适配能力重新生成交接。"
          : "适配交接未通过实时身份校验；已移除少样本复核顺序，仅保留基础风险排序。",
      });
    });
    return () => {
      controller.abort();
      if (validationEpoch === adaptationValidationEpochRef.current) adaptationValidationEpochRef.current += 1;
    };
  }, [adaptationValidationKey, client]);

  useEffect(() => {
    if (!onSharedSnapshotChange || !sessionId || boundSessionId !== sessionId || !readyArtifact || !readyPreview) return;
    const emissionKey = [
      sessionId,
      readyArtifact.artifactHash,
      readyPreview.previewHash,
      run?.statusHash ?? "",
      result?.resultHash ?? "",
      activeCaseId ?? "",
      selectedSourceFile?.name ?? sharedSnapshot?.sourceFileName ?? "",
    ].join(":");
    if (emittedSnapshotKeyRef.current === emissionKey) return;
    emittedSnapshotKeyRef.current = emissionKey;
    const next = Object.freeze({
      schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
      sessionId,
      sourceFileName: selectedSourceFile?.name ?? sharedSnapshot?.sourceFileName ?? "当前会话",
      artifact: readyArtifact,
      preview: readyPreview,
      ...(run ? { run } : {}),
      ...(result ? { result } : {}),
      ...(activeCaseId ? { activeCaseId } : {}),
      updatedAt: new Date().toISOString(),
    } satisfies GovernanceWorkspaceSnapshot);
    // Mark the semantic identity before the parent stores and echoes this
    // snapshot. The current component already owns the hydrated candidate
    // state, so its own echo must not start a second restore that can clear a
    // user's in-progress selection or case review.
    restoredSnapshotKeyRef.current = [
      sessionId,
      readyArtifact.artifactHash,
      run?.statusHash ?? "",
      result?.resultHash ?? "",
    ].join(":");
    onSharedSnapshotChange(next);
  }, [activeCaseId, boundSessionId, onSharedSnapshotChange, readyArtifact, readyPreview, result, run, selectedSourceFile?.name, sessionId, sharedSnapshot?.sourceFileName]);

  const reopenRun = async (item: GovernanceOnlineRun) => {
    setNotice(null);
    if (item.status !== "succeeded") {
      setHistoryFeedback("未完成记录仅供查阅，不能在治理应用中重试或重新分析。");
      return;
    }
    if (run?.runId === item.runId && result?.runId === item.runId) {
      setHistoryFeedback("该运行已载入当前工作台，可以继续查看候选、证据和研判单。");
      return;
    }
    const controller = new AbortController();
    setBusyAction(`history:${item.runId}`);
    setHistoryFeedback("正在载入已保存的运行结果…");
    try {
      await hydrateSucceededRun(item, controller.signal);
      setHistoryFeedback("运行已载入当前工作台，可以继续查看候选、证据和研判单。");
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) {
        setHistoryFeedback("运行记录载入失败，当前工作台内容未被修改。");
        setNotice(describeError(error));
      }
    } finally { setBusyAction(null); }
  };

  const createCase = async () => {
    if (!run || run.status !== "succeeded") return; setBusyAction("case");
    try { const next = await client.createCase(run.runId, `风险研判 ${new Date().toLocaleDateString("zh-CN")}`, "当前风险分析形成的人工研判单。", undefined); setCases((items) => [next, ...items]); setActiveCaseId(next.caseId); setWorkspaceMode("candidates"); setLeftView("cases"); }
    catch (error) { setNotice(describeError(error)); } finally { setBusyAction(null); }
  };
  const prepareSelectedForReview = async () => {
    if (!run || run.status !== "succeeded" || !selected || busyAction === "prepare-review") return;
    setBusyAction("prepare-review");
    try {
      let nextCase = activeCase && (activeCase.state === "draft" || activeCase.state === "active")
        ? activeCase
        : await client.createCase(
          run.runId,
          `风险研判 ${new Date().toLocaleDateString("zh-CN")}`,
          "当前风险分析形成的人工研判单。",
          undefined,
        );
      if (!nextCase.items.some((item) => item.targetType === selected.kind && item.targetId === selected.id)) {
        nextCase = await client.addCaseItem(nextCase.caseId, selected.kind, selected.id, "由治理工作台加入研判范围。");
      }
      if (nextCase.state === "draft") {
        nextCase = await client.updateCase(nextCase.caseId, "active", "加入当前对象并开始人工复核");
      }
      setCases((items) => items.some((item) => item.caseId === nextCase.caseId)
        ? items.map((item) => item.caseId === nextCase.caseId ? nextCase : item)
        : [nextCase, ...items]);
      setActiveCaseId(nextCase.caseId);
    } catch (error) {
      setNotice(describeError(error));
    } finally {
      setBusyAction(null);
    }
  };
  const submitReview = async (decision: GovernanceReviewDecision) => {
    if (!activeCase || activeCase.state !== "active" || !selected || !reviewReason.trim() || !activeCase.items.some((item) => item.targetType === selected.kind && item.targetId === selected.id)) return; setBusyAction("review");
    try { const next = await client.review(activeCase.caseId, selected.kind, selected.id, decision, reviewReason.trim()); setCases((items) => items.map((item) => item.caseId === next.caseId ? next : item)); setReviewReason(""); }
    catch (error) { setNotice(describeError(error)); } finally { setBusyAction(null); }
  };
  const transitionCase = async () => {
    if (!activeCase) return; const nextState = CASE_NEXT[activeCase.state]; if (!nextState) return; setBusyAction("transition");
    try { const next = await client.updateCase(activeCase.caseId, nextState, "治理工作台阶段流转"); setCases((items) => items.map((item) => item.caseId === next.caseId ? next : item)); }
    catch (error) { setNotice(describeError(error)); } finally { setBusyAction(null); }
  };
  const exportReport = async (format: "json" | "markdown" | "html") => {
    if (!activeCase) return; setBusyAction(`report:${format}`);
    try { const blob = await client.report(activeCase.caseId, format); const url = URL.createObjectURL(blob); const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${activeCase.caseId}.${format === "markdown" ? "md" : format}`; anchor.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000); }
    catch (error) { setNotice(describeError(error)); } finally { setBusyAction(null); }
  };
  const governanceFindings = useMemo(
    () => sortFindingsByAdaptedRank(findings, adaptationComparison),
    [adaptationComparison, findings],
  );
  const governanceGroups = useMemo(
    () => sortDerivationsByAdaptedRank(groups, adaptationComparison),
    [adaptationComparison, groups],
  );
  const governanceRelations = useMemo(
    () => sortDerivationsByAdaptedRank(relations, adaptationComparison),
    [adaptationComparison, relations],
  );
  const governanceLinks = useMemo(
    () => sortDerivationsByAdaptedRank(links, adaptationComparison),
    [adaptationComparison, links],
  );
  const riskFindings = governanceFindings.filter((finding) => finding.riskBand === "high" || finding.riskBand === "review");
  const decisionFor = (kind: GovernanceTargetKind, targetId: string): GovernanceReviewDecision | undefined => activeCase
    ? activeCase.currentDecisions[`${kind}:${targetId}`] ?? activeCase.currentDecisions[targetId]
    : undefined;
  const pendingRiskFindings = [
    ...riskFindings.filter((finding) => decisionFor("node", finding.nodeId) === "pending"),
    ...riskFindings.filter((finding) => decisionFor("node", finding.nodeId) === undefined),
  ];
  const resolvedRiskFindings = riskFindings.filter((finding) => {
    const decision = decisionFor("node", finding.nodeId);
    return decision === "confirmed" || decision === "rejected";
  });
  const displayedRiskFindings = candidateReviewView === "pending" ? pendingRiskFindings : resolvedRiskFindings;
  const riskFindingById = new Map(findings.map((finding) => [finding.nodeId, finding]));
  const riskGroups = governanceGroups.filter((group) => group.nodeIds.some((nodeId) => {
    const band = riskFindingById.get(nodeId)?.riskBand;
    return band === "high" || band === "review";
  }));
  const listItems = leftView === "groups" ? riskGroups : relationView === "factual" ? governanceRelations : governanceLinks;
  const sessionHistory = readyArtifact
    ? history.filter((item) => item.artifactId === readyArtifact.artifactId
      && item.datasetContentHash === readyArtifact.datasetContentHash
      && item.graphVersionHash === readyArtifact.graphVersionHash)
    : [];
  const visibleFindings = displayedRiskFindings.slice(0, visibleCandidateCount);
  const visibleListItems = listItems.slice(0, visibleCandidateCount);
  const visibleCases = cases.slice(0, visibleCandidateCount);
  const visibleHistory = sessionHistory.slice(0, visibleCandidateCount);
  const candidateTotal = leftView === "findings" ? displayedRiskFindings.length
    : leftView === "groups" || leftView === "relations" ? listItems.length
      : leftView === "cases" ? cases.length : 0;
  const canLoadMore = candidateTotal > visibleCandidateCount;
  const candidateLabel = useCallback((nodeId: string): string => governanceAccountLabel(findings.find((item) => item.nodeId === nodeId)?.label, nodeId), [findings]);
  const reviewRank = useCallback((finding: GovernanceOnlineFinding): number => adaptationRows.get(finding.nodeId)?.adaptedRank ?? finding.rank, [adaptationRows]);
  const reviewRankDelta = useCallback((finding: GovernanceOnlineFinding): number | null => adaptationRows.get(finding.nodeId)?.rankDelta ?? null, [adaptationRows]);
  const reviewPriorityLabel = useCallback((finding: GovernanceOnlineFinding): string => {
    const adapted = adaptationRows.get(finding.nodeId);
    return adapted
      ? `适配后复核优先级 #${adapted.adaptedRank}${adapted.rankDelta === 0 ? "" : adapted.rankDelta < 0 ? ` · 上升 ${Math.abs(adapted.rankDelta)}` : ` · 下降 ${adapted.rankDelta}`}`
      : `风险排序 #${finding.rank}`;
  }, [adaptationRows]);
  const assistantMode = ragOpen;
  const activeWorkspaceMode = assistantMode ? "assistant" : workspaceMode;
  const completedStage = stageIndex(run?.stage);
  const currentStageLabel = run
    ? run.stage === "completed" ? "分析完成" : STAGE_META.find(([id]) => id === run.stage)?.[1] ?? "等待处理"
    : readyArtifact ? "数据已就绪" : "等待数据";
  const stageProgress = run ? Math.max(1, Math.min(STAGE_META.length, completedStage + 1)) : 0;
  const selectedInActiveCase = Boolean(activeCase && selected && activeCase.items.some((item) => item.targetType === selected.kind && item.targetId === selected.id));
  const selectedReadyForReview = Boolean(activeCase?.state === "active" && selectedInActiveCase);
  const adaptationReady = validatedAdaptation.state === "ready" && validatedAdaptationKeyRef.current === adaptationValidationKey;
  const adaptationWarning = validatedAdaptation.state === "error" ? validatedAdaptation.message : null;
  const currentDecision = activeCase && selected ? activeCase.currentDecisions[`${selected.kind}:${selected.id}`] ?? activeCase.currentDecisions[selected.id] : undefined;
  const openWorkspaceMode = (mode: WorkspaceMode | "assistant") => {
    if (mode === "assistant") {
      onRagOpenChange(true);
      setEvidenceOpen(false);
      return;
    }
    onRagOpenChange(false);
    if (mode !== workspaceMode) clearSelectedTarget();
    setWorkspaceMode(mode);
    setEvidenceOpen(false);
    if (mode === "candidates") {
      setLeftView("findings");
      setCandidateReviewView("pending");
      setLens("risk");
    } else if (mode === "relations") {
      setLeftView("groups");
      setLens("community");
    } else {
      setLeftView("cases");
    }
  };
  const selectedTitle = currentFinding
    ? candidateLabel(currentFinding.nodeId)
    : currentDerivation?.kind === "group"
      ? `风险群组 · ${currentDerivation.memberCount ?? currentDerivation.nodeIds.length} 个账号`
      : currentDerivation
        ? `${candidateLabel(currentDerivation.source ?? currentDerivation.nodeIds[0])} — ${candidateLabel(currentDerivation.target ?? currentDerivation.nodeIds[1])}`
        : selected?.id ?? "治理对象";
  const currentDerivationRank = currentDerivation
    ? Math.max(0, (currentDerivation.kind === "group"
      ? riskGroups
      : currentDerivation.factual
        ? governanceRelations
        : governanceLinks).findIndex((item) => item.id === currentDerivation.id) + 1) || null
    : null;
  const endpointRiskLabel = (nodeId: string): string => {
    const finding = riskFindingById.get(nodeId);
    return finding ? riskLabel(finding.riskBand) : "普通关联账号";
  };
  const reviewContent = selected && result ? <div className="governance-dialog-review-form">
    <header><div><strong>人工结论</strong><span>结论与模型排序分开记录</span></div>{activeCase ? <em>{CASE_STATUS_LABEL[activeCase.state]}</em> : null}</header>
    {activeCase ? <label><span>当前研判单</span><select value={activeCase.caseId} onChange={(event) => { const nextCase = cases.find((item) => item.caseId === event.target.value); if (nextCase) handleCaseSelect(nextCase); }}>{cases.map((item) => <option key={item.caseId} value={item.caseId}>{item.title}</option>)}</select></label> : null}
    <div className="governance-dialog-review-form__actions">
      {selectedReadyForReview
        ? <p className="governance-review-ready" role="status"><CheckCircle />已加入研判单，可提交人工结论</p>
        : <button type="button" className="is-primary" disabled={busyAction === "prepare-review"} onClick={() => void prepareSelectedForReview()}>{busyAction === "prepare-review" ? <CircleNotch className="spin" /> : <Plus />}{busyAction === "prepare-review" ? "正在准备复核…" : "加入并开始复核"}</button>}
    </div>
    {currentDecision ? <p className="governance-current-decision"><CheckCircle />当前人工结论：{REVIEW_DECISION_LABEL[currentDecision]}</p> : null}
    <label><span>复核理由</span><textarea rows={4} maxLength={2000} value={reviewReason} onChange={(event) => setReviewReason(event.target.value)} placeholder="记录证据来源、判断依据与仍待核验的事项" /></label>
    <p className="governance-review-hint">{!selectedReadyForReview ? "先将当前对象加入研判单并开始复核。" : !reviewReason.trim() ? "填写理由后可提交。" : "选择人工结论并记录到当前研判单。"}</p>
    <div className="governance-review-actions"><button type="button" disabled={!selectedReadyForReview || !reviewReason.trim() || busyAction === "review"} onClick={() => void submitReview("confirmed")}>确认</button><button type="button" disabled={!selectedReadyForReview || !reviewReason.trim() || busyAction === "review"} onClick={() => void submitReview("rejected")}>驳回</button><button type="button" disabled={!selectedReadyForReview || !reviewReason.trim() || busyAction === "review"} onClick={() => void submitReview("pending")}>待定</button></div>
    {activeCase?.reviewEvents.length ? <ol className="governance-timeline">{activeCase.reviewEvents.slice().reverse().slice(0, 6).map((event) => <li key={event.eventId}><i /><span><strong>{REVIEW_DECISION_LABEL[event.decision]} · {event.targetId}</strong><small>{event.reason}</small><time>{new Date(event.createdAt).toLocaleString("zh-CN")}</time></span></li>)}</ol> : <p className="governance-empty">尚无人工复核记录。</p>}
  </div> : <p>当前对象尚未绑定可复核结果。</p>;

  return (
    <section className={`governance-workspace ${adaptation ? "has-adaptation-status" : ""}`} aria-label="当前会话治理" data-testid="governance-workspace" data-workspace-mode={activeWorkspaceMode}>
      <nav className="governance-mode-nav" aria-label="治理工作模式">
        <div className="governance-mode-nav__identity"><ShieldCheck weight="fill" /><span><strong>当前会话治理</strong><small>风险排序与人工复核</small></span></div>
        <button type="button" aria-label="风险节点" title={`风险节点：${riskFindings.length.toLocaleString()} 个对象`} aria-pressed={activeWorkspaceMode === "candidates"} onClick={() => openWorkspaceMode("candidates")}><MagnifyingGlass /><span><strong>风险节点</strong><small>{pendingRiskFindings.length.toLocaleString()} 个待复核</small></span></button>
        <button type="button" aria-label="群组与关系" title="群组与关系" aria-pressed={activeWorkspaceMode === "relations"} onClick={() => openWorkspaceMode("relations")}><UsersThree /><span><strong>群组与关系</strong><small>风险群组与重点关系</small></span></button>
        <button type="button" aria-label="研判单" title="研判单" aria-pressed={activeWorkspaceMode === "cases"} onClick={() => openWorkspaceMode("cases")}><FileText /><span><strong>研判单</strong><small>{cases.length.toLocaleString()} 份记录</small></span></button>
        <button type="button" aria-label="研判助手" title="研判助手" aria-pressed={activeWorkspaceMode === "assistant"} onClick={() => openWorkspaceMode("assistant")}><Brain /><span><strong>研判助手</strong><small>生成可审计报告</small></span></button>
        <details className="governance-history"><summary aria-label="运行记录" title="运行记录"><ClockCounterClockwise /></summary><div><header><strong>运行记录</strong><small>{sessionHistory.length} 条</small></header>{visibleHistory.length ? visibleHistory.map((item) => <button type="button" key={item.runId} disabled={item.status !== "succeeded"} title={item.status === "succeeded" ? "打开已保存结果" : "未完成记录仅供查阅"} onClick={() => void reopenRun(item)}><span><strong>{new Date(item.createdAt).toLocaleString("zh-CN")}</strong><small>{runStatusLabel(item.status)}</small></span>{busyAction === `history:${item.runId}` ? <CircleNotch className="spin" /> : item.status === "succeeded" ? <ArrowCounterClockwise /> : <X />}</button>) : <p>暂无运行记录。</p>}{historyFeedback ? <p role="status">{historyFeedback}</p> : null}</div></details>
      </nav>

      {adaptation ? <div className={`governance-adaptation-status ${adaptationReady ? "is-ready" : adaptationWarning ? "is-invalid" : "is-checking"}`} role={adaptationWarning ? "alert" : "status"}>
        <ShieldCheck />
        <span>{adaptationReady
          ? adaptation.lane === "few_shot" ? "少样本复核顺序已重新校验，候选按适配后位次排列" : "基础风险排序身份已重新校验"
          : adaptationWarning ?? "正在重新校验适配交接身份…"}</span>
      </div> : null}

      {notice ? <div className="governance-notice" role="status"><WarningCircle />{notice}<button type="button" aria-label="关闭提示" onClick={() => setNotice(null)}><X /></button></div> : null}

      {(run?.status === "queued" || run?.status === "running") ? <div className="governance-running" role="status"><CircleNotch className="spin" /><span>{currentStageLabel}</span><progress max={STAGE_META.length} value={stageProgress} /></div> : null}

      <main ref={bodyRef} className={`governance-main ${assistantMode ? "is-assistant" : ""}`}>
        {assistantMode ? <div className="governance-assistant-slot">{assistantPanel ?? <div className="governance-assistant-empty"><Brain /><strong>研判助手</strong><span>当前界面正在连接研判报告服务。</span></div>}</div> : !readyArtifact || !analysisComplete ? <section className="governance-session-empty">
          {sharedUploadPendingFileName || artifact.state === "loading" || preview.state === "loading" ? <><CircleNotch className="spin" /><h2>正在同步当前会话</h2><p>治理结果准备完成后会自动出现在这里。</p></> : <><Graph /><h2>当前会话暂无治理结果</h2><p>{sharedRestoreMessage ?? "请在对话研究中提交图谱并完成分析。"}</p><button type="button" onClick={() => { window.location.hash = "#/research"; }}>返回对话研究</button></>}
        </section> : <section className="governance-work-surface">
          {workspaceMode === "candidates" ? <div className="governance-candidate-review-tabs" role="tablist" aria-label="风险节点研判状态">
            <button type="button" role="tab" aria-selected={candidateReviewView === "pending"} onClick={() => { clearSelectedTarget(); setCandidateReviewView("pending"); }}>待复核 <span>{pendingRiskFindings.length}</span></button>
            <button type="button" role="tab" aria-selected={candidateReviewView === "resolved"} onClick={() => { clearSelectedTarget(); setCandidateReviewView("resolved"); }}>已研判 <span>{resolvedRiskFindings.length}</span></button>
          </div> : null}
          {workspaceMode === "relations" ? <div className="governance-relation-tabs" role="tablist" aria-label="群组与关系类型">
            <button type="button" role="tab" aria-selected={leftView === "groups"} onClick={() => { clearSelectedTarget(); setLeftView("groups"); setLens("community"); }}>风险群组 <span>{riskGroups.length}</span></button>
            <button type="button" role="tab" aria-selected={leftView === "relations" && relationView === "factual"} onClick={() => { clearSelectedTarget(); setLeftView("relations"); setRelationView("factual"); setLens("relations"); }}>事实关系 <span>{relations.length}</span></button>
            <button type="button" role="tab" aria-selected={leftView === "relations" && relationView === "clues"} onClick={() => { clearSelectedTarget(); setLeftView("relations"); setRelationView("clues"); setLens("relations"); }}>潜在线索 <span>{links.length}</span></button>
          </div> : null}

          {selected && workspaceMode !== "cases" ? <div className="governance-selection-banner" role="status"><span><i className={currentFinding?.riskBand === "high" ? "is-high" : currentFinding?.riskBand === "review" ? "is-review" : ""} /><span><strong>{selectedTitle}</strong><small>已在图谱中突出关联节点与关系，视角位置保持不变</small></span></span><button type="button" onClick={() => setEvidenceOpen(true)}><ShieldCheck />查看证据</button></div> : null}

          <div className="governance-result-list" aria-label={workspaceMode === "candidates" ? "风险节点" : workspaceMode === "relations" ? "群组与关系" : "研判单"}>
            {workspaceMode === "candidates" ? visibleFindings.map((finding) => {
              const reviewDecision = decisionFor("node", finding.nodeId);
              return <article key={finding.nodeId} className={selected?.kind === "node" && selected.id === finding.nodeId ? "is-selected" : ""}>
              <button type="button" className="governance-result-list__select" onClick={() => selectTarget({ kind: "node", id: finding.nodeId, nodeIds: [finding.nodeId] })}><b>#{reviewRank(finding)}</b><span><strong>{candidateLabel(finding.nodeId)}</strong>{reviewDecision ? <span className={`governance-review-status is-${reviewDecision}`}>人工{REVIEW_DECISION_LABEL[reviewDecision]}</span> : null}<small>{riskLabel(finding.riskBand)}{finding.structureMissing ? " · 结构信息待补" : ""}{reviewRankDelta(finding) !== null ? ` · 基础排序 #${finding.rank}` : ""}</small></span><em>{reviewPriorityLabel(finding)}</em></button>
              <button type="button" className="governance-result-list__evidence" aria-label={`查看 ${candidateLabel(finding.nodeId)} 的证据`} onClick={() => { selectTarget({ kind: "node", id: finding.nodeId, nodeIds: [finding.nodeId] }); setEvidenceOpen(true); }}><ShieldCheck />查看证据</button>
            </article>;
            }) : null}

            {workspaceMode === "relations" ? visibleListItems.map((item, index) => {
              const groupHigh = item.nodeIds.filter((nodeId) => riskFindingById.get(nodeId)?.riskBand === "high").length;
              const groupReview = item.nodeIds.filter((nodeId) => riskFindingById.get(nodeId)?.riskBand === "review").length;
              const source = item.source ?? item.nodeIds[0];
              const target = item.target ?? item.nodeIds[1];
              const targetKind = item.kind === "group" ? "group" : "relation";
              const reviewDecision = decisionFor(targetKind, item.id);
              return <article key={item.id} className={selected?.id === item.id ? "is-selected" : ""}>
                <button type="button" className="governance-result-list__select" onClick={() => selectTarget({ kind: targetKind, id: item.id, nodeIds: item.nodeIds })}><b>{item.kind === "group" ? <UsersThree /> : item.kind === "factual_relation" ? <Link /> : <TreeStructure />}</b><span><strong>{item.kind === "group" ? `${item.memberCount ?? item.nodeIds.length} 个账号的风险群组` : `${candidateLabel(source)} — ${candidateLabel(target)}`}</strong>{reviewDecision ? <span className={`governance-review-status is-${reviewDecision}`}>人工{REVIEW_DECISION_LABEL[reviewDecision]}</span> : null}<small>{item.kind === "group" ? `高风险 ${groupHigh} · 建议复核 ${groupReview} · ${item.modalities.map(governanceModalityLabel).join("、") || "关系类型待核"}` : `${item.factual ? "事实关系" : "潜在线索（非事实边）"} · ${endpointRiskLabel(source)} / ${endpointRiskLabel(target)} · ${item.modalities.map(governanceModalityLabel).join("、") || "关系属性待核"}`}</small></span><em>复核顺序 #{index + 1}</em></button>
                <button type="button" className="governance-result-list__evidence" aria-label={`查看 ${item.kind === "group" ? "风险群组" : "关系"}证据`} onClick={() => { selectTarget({ kind: item.kind === "group" ? "group" : "relation", id: item.id, nodeIds: item.nodeIds }); setEvidenceOpen(true); }}><ShieldCheck />查看证据</button>
              </article>;
            }) : null}

            {workspaceMode === "cases" ? <div className="governance-cases-workspace">
              <header><div><h2>研判单</h2><p>保存人工复核结论、理由与导出记录。</p></div><button type="button" disabled={!result || busyAction === "case"} onClick={() => void createCase()}><Plus />新建研判单</button></header>
              <div className="governance-case-grid">{visibleCases.map((item) => <button type="button" key={item.caseId} aria-pressed={activeCase?.caseId === item.caseId} onClick={() => handleCaseSelect(item)}><FileText /><span><strong>{item.title}</strong><small>{CASE_STATUS_LABEL[item.state]} · {item.items.length} 个对象 · {item.reviewEvents.length} 次复核</small></span></button>)}</div>
              {activeCase ? <section className="governance-case-detail"><header><div><h3>{activeCase.title}</h3><p>{activeCase.description || "未填写研判说明。"}</p></div><span>{CASE_STATUS_LABEL[activeCase.state]}</span></header><dl><div><dt>治理对象</dt><dd>{activeCase.items.length}</dd></div><div><dt>已复核</dt><dd>{reviewedCaseItemCount}</dd></div><div><dt>待复核</dt><dd>{Math.max(0, activeCase.items.length - reviewedCaseItemCount)}</dd></div></dl><p>{caseNextStep(activeCase, reviewedCaseItemCount)}</p><div className="governance-case-actions"><button type="button" onClick={() => void transitionCase()}>{activeCase.state === "archived" ? <ArrowCounterClockwise /> : <Archive />}{CASE_ACTION_LABEL[activeCase.state]}</button><button type="button" onClick={() => void exportReport("html")}><DownloadSimple />HTML</button><button type="button" onClick={() => void exportReport("markdown")}><DownloadSimple />Markdown</button></div></section> : <p className="governance-empty">暂无研判单。选择风险节点后可从证据档案建立研判单。</p>}
            </div> : null}

            {workspaceMode !== "cases" && !visibleFindings.length && leftView === "findings" ? <p className="governance-empty">{candidateReviewView === "pending" ? "当前没有待复核的风险节点。" : "当前还没有已确认或已驳回的对象。"}</p> : null}
            {workspaceMode === "relations" && !visibleListItems.length ? <p className="governance-empty">当前视图没有需要展示的群组或关系。</p> : null}
            {canLoadMore ? <button type="button" className="governance-load-more" onClick={() => setVisibleCandidateCount((count) => count + 50)}>加载更多</button> : null}
          </div>
        </section>}
      </main>

      <EvidenceDossier
        open={evidenceOpen && Boolean(selected) && analysisComplete}
        target={selected}
        title={selectedTitle}
        finding={currentFinding}
        derivation={currentDerivation}
        derivationRank={currentDerivationRank}
        evidence={evidence}
        skillsContext={skillsContext}
        summaryClient={evidenceSummaryClient}
        reviewContent={reviewContent}
        candidateLabel={candidateLabel}
        reviewRank={reviewRank}
        onSelectNeighbor={(nodeId) => selectTarget({ kind: "node", id: nodeId, nodeIds: [nodeId] })}
        onClose={closeEvidencePanel}
      />
    </section>
  );
}
