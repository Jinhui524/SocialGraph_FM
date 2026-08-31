import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type SetStateAction,
} from "react";
import { SocialGraphWorkspaceView } from "./SocialGraphWorkspaceView";
import {
  CaretLeft,
  CaretRight,
  CheckCircle,
  CircleNotch,
  CloudArrowUp,
  FolderOpen,
  Paperclip,
  PaperPlaneTilt,
  ShieldCheck,
  SidebarSimple,
  Sparkle,
  WarningCircle,
} from "@phosphor-icons/react";
import { Sidebar, type SidebarWorkspace } from "../../components/Sidebar";
import { CoreUsageGuide } from "../../components/CoreUsageGuide";
import type { AssistantActivityKind } from "../../components/AssistantActivity";
import { ResearchDatasetPanel } from "../../components/ResearchDatasetPanel";
import { AdaptationWorkspace, type AdaptationGovernanceTarget, type AdaptationLane, type AdaptationModelCardState } from "../../components/GovernanceWorkbench";
import { GovernanceTaskSelector, governanceWorkspaceMountKey, resolveGovernanceTask, type GovernanceTaskEntry } from "../../components/GovernanceTaskSelector";
import { GovernanceRagPanel } from "../../components/GovernanceRagPanel";
import {
  GovernanceOnlineWorkspace,
  governanceArtifactDisplayName,
  governanceImportedGraphVersion,
  governancePreviewGraph,
  type GovernanceGraphPresentation,
} from "../../components/GovernanceOnlineWorkspace";
import { useGovernanceWorkspace } from "../../components/GovernanceWorkspaceProvider";
import { VersionLifecyclePanel } from "../../components/VersionLifecyclePanel";
import { WorkspaceDrawer } from "../../components/WorkspaceDrawer";
import { WorkspaceResizeHandle } from "../../components/WorkspaceResizeHandle";
import {
  GraphImportMappingCard,
  GraphTableRoleCard,
  type GraphImportMappingValue,
} from "../../components/GraphImportMappingCard";
import type {
  GraphPreviewExportHandlers,
  GraphPreviewViewSnapshot,
} from "../../components/GraphPreview";
import { buildDemoGraphVersion, createGraphVersion, LocalGraphImportAdapter } from "../../services/graphImport";
import { HttpGraphBuildIntentNormalizer } from "../../services/graphBuildIntent";
import {
  buildGraphScene,
  buildSemanticGraphSlice,
  createDefaultGraphViewState,
  normalizeGraphViewState,
} from "../../services/graphScene";
import {
  createGraphWorkbenchViewState,
  reduceGraphView,
  type GraphViewAction,
} from "../../services/graphViewState";
import {
  buildArticulationOverlay,
  buildCommunityOverlay,
  buildComponentsOverlay,
  buildDegreeOverlay,
  buildRawFactsOverlay,
} from "../../services/graphOverlays";
import {
  createLocalGraphRepository,
  createResearchSession,
  createSemanticEvent,
} from "../../services/graphRepository";
import { applyViewCommand } from "../../services/viewCommand";
import {
  buildGraphContextSummary,
  HttpIntentNormalizer,
  type IntentServiceStatus,
} from "../../services/intentNormalizer";
import { LocalAnalysisExecutor } from "../../services/localAnalysisExecutor";
import { describeUnavailableAnalysis } from "../../services/analysisUnavailable";
import { createScopedGraphSlice } from "../../services/graphAlgorithms";
import { applyPreparedAnalysisFilters, prepareAnalysisFilters } from "../../services/analysisFilters";
import {
  createSourceArtifact,
  requestPersistentGraphStorage,
  type StorageCapacity,
} from "../../services/sourceArtifact";
import { GraphWorkerExecutionError, runGraphTask } from "../../services/graphWorkerRunner";
import {
  GraphCameraSnapshotCache,
  type GraphCameraSnapshot,
  type GraphCameraSnapshotCacheKey,
} from "../../services/graphEngineAdapter";
import { SocialGraphApiError, socialGraphApiUrl } from "../../services/apiClient";
import { CoreClient } from "../../services/coreClient";
import { ResearchClient } from "../../services/researchClient";
import { GovernanceOnlineClient } from "../../services/governanceOnlineClient";
import { GlobalModelClient } from "../../services/globalModelClient";
import { GOVERNANCE_WORKSPACE_SCHEMA, type GovernanceWorkspaceSnapshot } from "../../services/governanceWorkspaceStore";
import { GovernanceSkillsClient } from "../../services/governanceSkillsClient";
import { governanceAccountLabel, governanceModalityLabel } from "../../services/governancePresentation";
import { shouldSubmitComposerKey } from "../../services/composerKeyboard";
import type { GovernanceSkillsContext } from "../../types/governanceSkills";
import type {
  GovernanceOnlineRun,
} from "../../types/governanceOnline";
import { loadGovernanceTheme, saveGovernanceTheme } from "../../services/governancePreferences";
import {
  hashForWorkspaceRoute,
  isWorkspaceGraphPaneVisible,
  workspaceRouteFromHash,
  type WorkspaceRoute,
} from "../../services/workspaceRoute";
import {
  graphVersionFromDatasetArtifact,
  ResearchDatasetClient,
  type DatasetArtifact,
} from "../../services/researchDatasetClient";
import {
  DEFAULT_WORKSPACE_LAYOUT,
  loadWorkspaceLayout,
  resolveWorkspaceGraphHeight,
  resolveWorkspaceLayout,
  resizeWorkspacePane,
  saveWorkspaceLayout,
  type WorkspaceLayoutRoute,
  type WorkspaceLayoutState,
} from "../../services/workspaceLayout";
import type {
  AnalysisRun,
  AnalysisOverlay,
  ConversationMessage,
  GraphScene,
  GraphFilters,
  GraphTheme,
  GraphNode,
  GraphVersion,
  GovernanceFocus,
  GraphVersionProvenance,
  GraphViewState,
  GraphWorkbenchViewState,
  GraphRepository,
  ImportRun,
  FileProfile,
  GraphBuildSpec,
  SourceArtifact,
  NormalizedIntent,
  ResearchSession,
  TargetResolution,
  ViewCommand,
} from "../../types/graph";
import type { CoreWorkbenchServiceState } from "../../types/core";
import type { ResearchScenario, ResearchServiceState } from "../../types/research";
import {
  AdaptationGraphSwitcher,
  activateAdaptationLanePresentation,
  adaptationCameraLens,
  createAdaptationLanePresentationState,
  governanceTaskEntryFromAdaptationTarget,
  updateAdaptationLanePresentation,
  type AdaptationLanePresentationState,
} from "../adaptation/presentation";
export {
  AdaptationGraphSwitcher,
  activateAdaptationLanePresentation,
  adaptationCameraLens,
  createAdaptationLanePresentationState,
  governanceTaskEntryFromAdaptationTarget,
  updateAdaptationLanePresentation,
} from "../adaptation/presentation";
export type {
  AdaptationLanePresentation,
  AdaptationLanePresentationState,
} from "../adaptation/presentation";

import {
  GOVERNANCE_ANALYSIS_STAGES,
  assistantActivityForEntry,
  assistantGuidanceStateForEntry,
  canOpenGovernanceReview,
  completeConfirmedPlanningMessage,
  governanceProgressStageIndex,
  governanceRunIdForPersistence,
  governanceRunIdFromStoredMessage,
  invalidateChatConfirmations,
  presentGovernanceRunProgress,
  updateGovernancePlanningProgress,
  type ChatEntry,
  type GovernanceProgressStage,
} from "../assistant/chatModel";
export {
  GOVERNANCE_ANALYSIS_STAGES,
  assistantActivityForEntry,
  assistantGuidanceStateForEntry,
  canOpenGovernanceReview,
  completeConfirmedPlanningMessage,
  governanceRunIdForPersistence,
  governanceRunIdFromStoredMessage,
  invalidateChatConfirmations,
  presentGovernanceRunProgress,
  updateGovernancePlanningProgress,
} from "../assistant/chatModel";
export type { ChatEntry, GovernanceProgressStage } from "../assistant/chatModel";

import {
  browserImportProvenance,
  pendingProfileForRole,
  type ImportViewState,
  type PendingImportDraft,
  type PendingTargetResolution,
} from "../graph-workbench/importModel";

import {
  resolveWorkspaceCameraSnapshot,
  routeForSidebarWorkspace,
  sidebarWorkspaceFromRoute,
} from "./navigation";
export { resolveWorkspaceCameraSnapshot } from "./navigation";

import { ORDINARY_PRESENTATION_COPY } from "./presentationCopy";
export { ORDINARY_PRESENTATION_COPY } from "./presentationCopy";

type WorkspacePanel = "sessions" | "guide" | "diagnostics" | "datasets" | "rename" | null;

interface LlmDiagnosticResult {
  readonly state: "idle" | "running" | "success" | "error";
  readonly latencyMs?: number;
  readonly schemaVersion?: string;
  readonly requestId?: string;
  readonly model?: string;
  readonly source?: "llm";
  readonly task?: string;
  readonly warnings?: readonly string[];
  readonly message?: string;
}

interface DatasetDiagnosticResult {
  readonly state: "idle" | "running" | "accepted" | "rejected" | "error";
  readonly detectedFormat?: string;
  readonly inspectionId?: string;
  readonly nodeCount?: number;
  readonly edgeCount?: number;
  readonly message?: string;
}

const SAMPLE_FILES = [
  {
    id: "governance-collaboration",
    name: "治理协作网络",
    description: "两个协作社区、桥接成员、关系权重与时间字段。",
    path: "/samples/governance-collaboration.csv",
  },
  {
    id: "heterogeneous-community",
    name: "异构社区网络",
    description: "人员、组织、项目和多种关系类型。",
    path: "/samples/heterogeneous-community.json",
  },
  {
    id: "manual-mapping",
    name: "字段映射 TSV",
    description: "非标准列名，用于验证字段映射流程。",
    path: "/samples/manual-mapping.tsv",
  },
  {
    id: "community-bridge-graphml",
    name: "GraphML 桥接结构",
    description: "验证浏览器安全 XML 解析、属性键与桥接结构。",
    path: "/samples/community-bridge.graphml",
  },
  {
    id: "collaboration-gexf",
    name: "GEXF 合作关系",
    description: "验证 GEXF 节点标签、关系类型和端点校验。",
    path: "/samples/collaboration-network.gexf",
  },
  {
    id: "invalid-dangling-edge",
    name: "异常校验 · 悬空关系",
    description: "应排除引用不存在节点的关系，并保留非阻断质量提示。",
    path: "/samples/invalid-dangling-edge.json",
  },
  {
    id: "invalid-duplicate-node",
    name: "异常校验 · 重复节点",
    description: "应明确拒绝重复节点 ID，不生成错误图。",
    path: "/samples/invalid-duplicate-node.json",
  },
] as const;

const GEOM_GCN_SAMPLE_PATH = "/samples/geom-gcn-toy.zip";

import {
  WelcomeAtlas,
  researchPromptSkillRequest,
  researchPromptForText,
  researchPrompts,
  welcomePromptAction,
  type ResearchPrompt,
} from "./welcome";
export {
  WelcomeAtlas,
  researchPromptSkillRequest,
  researchPromptForText,
  researchPrompts,
  welcomePromptAction,
} from "./welcome";
export type {
  ResearchPrompt,
  ResearchPromptContextScope,
} from "./welcome";

const GraphPreview = lazy(() => import("../../components/GraphPreview"));

const timeNow = () =>
  new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false }).format(new Date());

const makeId = (prefix: string) => `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

export function buildGovernanceUploadConversationEntries(
  file: { readonly name: string; readonly size: number },
  timestamp: string,
  identity: { readonly artifactId: string; readonly datasetContentHash: string },
): readonly ChatEntry[] {
  return Object.freeze([
    {
      id: makeId("user-governance-upload"),
      role: "user" as const,
      text: "上传推理包并准备分析。",
      timestamp,
      file: Object.freeze({ name: file.name, size: file.size }),
    },
    {
      id: `assistant-governance-guidance-${identity.artifactId}-${identity.datasetContentHash}`,
      role: "assistant" as const,
      text: ORDINARY_PRESENTATION_COPY.governanceNextStep,
      timestamp,
      state: "success" as const,
    },
  ]);
}

export function mergeGovernanceUploadConversationEntries(
  current: readonly ChatEntry[],
  additions: readonly ChatEntry[],
): readonly ChatEntry[] {
  const existingIds = new Set(current.map((entry) => entry.id));
  return Object.freeze([
    ...current,
    ...additions.filter((entry) => !existingIds.has(entry.id)),
  ]);
}

const taskNames: Record<NormalizedIntent["task"], string> = {
  overview: "网络概览",
  centrality: "中心性分析",
  bridge_detection: "桥接节点识别",
  community: "Louvain 社区发现",
  link_prediction: "潜在关系预测",
  node_role: "节点角色识别",
  similar_structure: "相似结构检索",
};

const initialDemoGraph = buildDemoGraphVersion();
const DEMO_SESSION_ID = "demo-research";
const SEEDED_SESSIONS: readonly ResearchSession[] = [
  createResearchSession("高校科研团队协作网络分析", {
    id: DEMO_SESSION_ID,
    graphVersionId: initialDemoGraph.id,
    updatedAt: initialDemoGraph.createdAt,
  }),
  createResearchSession("社区志愿者关系图谱构建", { id: "volunteer-network" }),
  createResearchSession("科技企业合作伙伴识别", { id: "industry-partners" }),
];

const initialMessages: ChatEntry[] = [
  {
    id: "demo-user",
    role: "user",
    text: "请帮我分析我们实验室与其他实验室的合作网络，找出关键合作枢纽和潜在合作对象。",
    timestamp: "10:24",
    file: { name: initialDemoGraph.sourceFile, size: 2_400_000 },
  },
];

function delay(milliseconds: number) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function isGraphConstructionRevision(text: string): boolean {
  return /(?:重新构图|构图规则|字段(?:映射|含义)|起点列|终点列|有向图|无向图|时间格式|重复边|自环)/u.test(text);
}

function isGovernanceAnalysisRequest(text: string): boolean {
  return /^(?:请)?(?:开始|运行|执行)(?:当前|一次|本次)?(?:治理)?分析[。！!\s]*$/u.test(text.trim());
}

function analysisViewKey(state: GraphViewState): string {
  return JSON.stringify({
    graphVersionId: state.graphVersionId,
    mode: state.mode,
    focusNodeIds: state.focusNodeIds,
    pathEndpointIds: state.pathEndpointIds,
    depth: state.depth,
    filters: state.filters,
  });
}

function sessionTime(updatedAt: string): string {
  const value = new Date(updatedAt);
  if (Number.isNaN(value.getTime())) return "本地";
  return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit" }).format(value);
}

function sessionTitleFromFile(fileName: string): string {
  const name = fileName.replace(/\.[^.]+$/u, "").trim();
  return name || "新建关系图";
}

function messageTime(createdAt: string): string {
  const value = new Date(createdAt);
  if (Number.isNaN(value.getTime())) return timeNow();
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(value);
}

function messageStatusFromEntry(entry: ChatEntry): ConversationMessage["status"] {
  if (entry.role === "user") return "completed";
  if (entry.state === "working") return "pending";
  if (entry.state === "warning") return "warning";
  if (entry.state === "error") return "failed";
  return "completed";
}

function buildRequestedOverlay(
  graph: GraphVersion,
  kind: Exclude<AnalysisOverlay["kind"], "raw" | "path" | "governance">,
  runId?: string,
): AnalysisOverlay {
  if (kind === "degree") return buildDegreeOverlay(graph, runId);
  if (kind === "articulation") return buildArticulationOverlay(graph, runId);
  if (kind === "community") return buildCommunityOverlay(graph, undefined, runId);
  return buildComponentsOverlay(graph, runId);
}

function defaultOverlayForRun(graph: GraphVersion, run: AnalysisRun): AnalysisOverlay | null {
  if (run.status !== "succeeded" || !run.result) return null;
  if (run.result.kind === "centrality") return buildDegreeOverlay(graph, run.id);
  if (run.result.kind === "bridge_detection") return buildArticulationOverlay(graph, run.id);
  if (run.result.kind === "community") return buildCommunityOverlay(graph, undefined, run.id);
  if (run.result.kind === "connected_components") return buildComponentsOverlay(graph, run.id);
  return null;
}

function withScopeProvenance(overlay: AnalysisOverlay, scopeHash: string): AnalysisOverlay {
  return Object.freeze({
    ...overlay,
    provenance: Object.freeze({ ...overlay.provenance, scopeHash }),
  });
}

import {
  GraphVersionOverlayController,
  locateGovernanceCandidates,
  resolveGovernanceCandidateFocus,
  resolveGraphVersionOverlay,
} from "../governance/overlayController";
export {
  GraphVersionOverlayController,
  locateGovernanceCandidates,
  resolveGovernanceCandidateFocus,
  resolveGraphVersionOverlay,
} from "../governance/overlayController";
export type {
  GovernanceCandidateFocus,
  GovernanceCandidateFocusInput,
  GovernanceCandidateLocateEffects,
  GraphVersionOverlayControllerOptions,
} from "../governance/overlayController";

import {
  buildAnalysisResultMarkdown,
  ensureHumanReviewGuidance,
  resultDescription,
} from "../governance/reports";
export {
  buildAnalysisResultMarkdown,
  ensureHumanReviewGuidance,
  resultDescription,
} from "../governance/reports";

function sameGovernanceArtifactIdentity(
  left: GovernanceWorkspaceSnapshot["artifact"],
  right: GovernanceWorkspaceSnapshot["artifact"],
): boolean {
  return left.artifactId === right.artifactId
    && left.datasetContentHash === right.datasetContentHash
    && left.graphVersionHash === right.graphVersionHash;
}

import {
  AssistantEntry,
  FileBadge,
  GraphBuildReviewCard,
  ImportTimeline,
  TargetResolutionCard,
  UserEntry,
  publicAssistantCopy,
} from "../graph-workbench/conversationPanels";
export {
  AssistantEntry,
  FileBadge,
  ImportTimeline,
  UserEntry,
  publicAssistantCopy,
} from "../graph-workbench/conversationPanels";

import {
  RightSummary,
  publicGraphSourceLabel,
} from "../graph-workbench/RightSummary";
export {
  RightSummary,
  publicGraphSourceLabel,
} from "../graph-workbench/RightSummary";

export function SocialGraphApp() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const composerInputRef = useRef<HTMLTextAreaElement>(null);
  const researchScrollRef = useRef<HTMLDivElement>(null);
  const importAdapter = useMemo(() => new LocalGraphImportAdapter(), []);
  const intentNormalizer = useMemo(() => new HttpIntentNormalizer(), []);
  const graphBuildIntentNormalizer = useMemo(() => new HttpGraphBuildIntentNormalizer(), []);
  const analysisExecutor = useMemo(() => new LocalAnalysisExecutor([initialDemoGraph]), []);
  const graphRepository = useMemo(() => createLocalGraphRepository(), []);
  const researchDatasetClient = useMemo(() => new ResearchDatasetClient(), []);
  const coreClient = useMemo(() => new CoreClient(), []);
  const researchClient = useMemo(() => new ResearchClient(), []);
  const governanceOnlineClient = useMemo(() => new GovernanceOnlineClient(), []);
  const globalModelClient = useMemo(() => new GlobalModelClient(), []);
  const governanceSkillsClient = useMemo(() => new GovernanceSkillsClient(), []);
  const governanceWorkspace = useGovernanceWorkspace();
  const governanceWorkspaceSnapshotRef = useRef(governanceWorkspace.snapshot);
  governanceWorkspaceSnapshotRef.current = governanceWorkspace.snapshot;
  const [initialWorkspaceRoute] = useState<WorkspaceRoute>(() => workspaceRouteFromHash(window.location.hash));

  const [activeSessionId, setActiveSessionId] = useState(() =>
    window.localStorage.getItem("socialgraph-fm-active-session") ?? "",
  );
  const [sessions, setSessions] = useState<readonly ResearchSession[]>([]);
  const [trashedSessions, setTrashedSessions] = useState<readonly ResearchSession[]>([]);
  const [graphVersion, setGraphVersion] = useState<GraphVersion | null>(null);
  const [graphWorkbenchView, setGraphWorkbenchView] = useState<GraphWorkbenchViewState>(() =>
    createGraphWorkbenchViewState(createDefaultGraphViewState("empty")),
  );
  const viewState = graphWorkbenchView.viewState;
  const graphInteraction = graphWorkbenchView.interaction;
  const setViewState = useCallback((update: SetStateAction<GraphViewState>) => {
    setGraphWorkbenchView((current) => ({
      viewState: typeof update === "function" ? update(current.viewState) : update,
      interaction: current.interaction,
    }));
  }, []);
  const replaceViewState = useCallback((nextViewState: GraphViewState) => {
    setGraphWorkbenchView((current) => reduceGraphView(current, {
      type: "replace_view",
      viewState: nextViewState,
    }));
  }, []);
  const dispatchGraphView = useCallback((action: GraphViewAction) => {
    setGraphWorkbenchView((current) => reduceGraphView(current, action));
  }, []);
  const [activeOverlay, setActiveOverlay] = useState<AnalysisOverlay | null>(null);
  const [governanceLegacyOverlay, setGovernanceLegacyOverlay] = useState<AnalysisOverlay | null>(null);
  const handleGovernanceOverlayChange = useCallback((overlay: AnalysisOverlay | null) => {
    setGovernanceLegacyOverlay(overlay);
  }, []);
  const [analysisSceneOverride, setAnalysisSceneOverride] = useState<{
    readonly viewKey: string;
    readonly scene: GraphScene;
  } | null>(null);
  const [pendingTargetResolution, setPendingTargetResolution] = useState<PendingTargetResolution | null>(null);
  const [graphExportHandlers, setGraphExportHandlers] = useState<GraphPreviewExportHandlers | null>(null);
  const [messages, setMessages] = useState<ChatEntry[]>([]);
  const [importState, setImportState] = useState<ImportViewState>({ kind: "idle" });
  const selectedNode = useMemo(
    () => graphVersion?.nodes.find((node) => node.id === graphInteraction.selectedNodeId) ?? null,
    [graphInteraction.selectedNodeId, graphVersion],
  );
  const [draft, setDraft] = useState("");
  const [pendingImport, setPendingImport] = useState<PendingImportDraft | null>(null);
  const [isSending, setIsSending] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [activeWorkspace, setActiveWorkspace] = useState<SidebarWorkspace>(() => sidebarWorkspaceFromRoute(initialWorkspaceRoute));
  const [governanceMounted, setGovernanceMounted] = useState(initialWorkspaceRoute === "governance" || initialWorkspaceRoute === "adaptation");
  const [adaptationMounted, setAdaptationMounted] = useState(initialWorkspaceRoute === "adaptation");
  const [adaptationModelCardState, setAdaptationModelCardState] = useState<AdaptationModelCardState>({ status: "loading", card: null });
  const [mobilePanel, setMobilePanel] = useState<"chat" | "governance" | "adaptation" | "graph">(() => (
    initialWorkspaceRoute === "governance"
      ? "governance"
      : initialWorkspaceRoute === "adaptation"
        ? "adaptation"
        : "chat"
  ));
  const selectMobileWorkspacePanel = useCallback((panel: "chat" | "governance" | "adaptation" | "graph") => {
    const scrollContainer = researchScrollRef.current;
    const shouldResetScroll = panel !== mobilePanel && panel !== "graph" && scrollContainer;
    setMobilePanel(panel);
    if (shouldResetScroll) window.requestAnimationFrame(() => {
      if (typeof scrollContainer.scrollTo === "function") {
        scrollContainer.scrollTo({ top: 0, left: 0, behavior: "auto" });
      } else {
        scrollContainer.scrollTop = 0;
        scrollContainer.scrollLeft = 0;
      }
    });
  }, [mobilePanel]);
  const [toast, setToast] = useState<string | null>(null);
  const showToast = useCallback((message: string) => {
    setToast(message);
    window.setTimeout(() => setToast(null), 2600);
  }, []);
  const [workspaceLayout, setWorkspaceLayout] = useState<WorkspaceLayoutState>(() =>
    loadWorkspaceLayout(undefined, window.innerWidth));
  const [workspacePanel, setWorkspacePanel] = useState<WorkspacePanel>(() => initialWorkspaceRoute === "datasets" ? "datasets" : null);
  const [viewportWidth, setViewportWidth] = useState(() => window.innerWidth);
  const [viewportHeight, setViewportHeight] = useState(() => window.innerHeight);
  const [renameSessionId, setRenameSessionId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [heroManuallyOpen, setHeroManuallyOpen] = useState(false);
  const [storageCapacity, setStorageCapacity] = useState<StorageCapacity>({ persisted: false });
  const [targetDomainBusy, setTargetDomainBusy] = useState(false);
  const [datasetPanelEpoch, setDatasetPanelEpoch] = useState(0);
  const [llmDiagnostic, setLlmDiagnostic] = useState<LlmDiagnosticResult>({ state: "idle" });
  const [datasetDiagnostic, setDatasetDiagnostic] = useState<DatasetDiagnosticResult>({ state: "idle" });
  const [sessionReady, setSessionReady] = useState(false);
  const [hydratedSessionId, setHydratedSessionId] = useState("");
  const [intentServiceStatus, setIntentServiceStatus] = useState<IntentServiceStatus>({
    state: "checking",
    label: "检测 LLM 服务",
  });
  const [coreService, setCoreService] = useState<CoreWorkbenchServiceState>({ state: "checking" });
  const [researchService, setResearchService] = useState<ResearchServiceState>({ state: "checking" });
  const [governanceChannel, setGovernanceChannel] = useState<"online" | "governance" | "research">("online");
  const [governanceGraphPresentation, setGovernanceGraphPresentation] = useState<GovernanceGraphPresentation | null>(null);
  const [governanceFilters, setGovernanceFilters] = useState<GraphFilters>({ nodeTypes: [], edgeTypes: [] });
  const [governanceTheme, setGovernanceTheme] = useState<GraphTheme>(() => loadGovernanceTheme());
  const [adaptationTheme, setAdaptationTheme] = useState<GraphTheme>("focus-dark");
  const governanceViewSnapshotRef = useRef<GraphPreviewViewSnapshot | null>(null);
  const chatViewSnapshotRef = useRef<GraphPreviewViewSnapshot | null>(null);
  const adaptationViewSnapshotRef = useRef<GraphPreviewViewSnapshot | null>(null);
  const graphCameraSnapshotCacheRef = useRef(new GraphCameraSnapshotCache());
  const [chatLocateOverview, setChatLocateOverview] = useState<GraphPreviewViewSnapshot | null>(null);
  const chatLocateOverviewRef = useRef<GraphPreviewViewSnapshot | null>(null);
  const [workspaceCameraRestore, setWorkspaceCameraRestore] = useState<(
    | (GraphCameraSnapshot & {
      readonly workspace: SidebarWorkspace;
      readonly x: number;
      readonly y: number;
      readonly token: number;
    })
    | {
      readonly workspace: SidebarWorkspace;
      readonly x: number;
      readonly y: number;
      readonly zoom: number;
      readonly token: number;
    }
  ) | null>(null);
  const workspaceCameraRestoreTokenRef = useRef(0);
  const [adaptationLanePresentation, setAdaptationLanePresentation] = useState<AdaptationLanePresentationState>(() => createAdaptationLanePresentationState());
  const adaptationLanePresentationRef = useRef(adaptationLanePresentation);
  adaptationLanePresentationRef.current = adaptationLanePresentation;
  const activeAdaptationPresentation = adaptationLanePresentation.lanes[adaptationLanePresentation.activeLane];

  useEffect(() => {
    if (activeWorkspace === "adaptation") setAdaptationMounted(true);
  }, [activeWorkspace]);
  useEffect(() => {
    if (!adaptationMounted) return;
    const controller = new AbortController();
    setAdaptationModelCardState({ status: "loading", card: null });
    globalModelClient.modelCard(controller.signal).then((card) => {
      if (!controller.signal.aborted) setAdaptationModelCardState({ status: "ready", card });
    }).catch(() => {
      if (!controller.signal.aborted) setAdaptationModelCardState({ status: "error", card: null });
    });
    return () => controller.abort();
  }, [adaptationMounted, globalModelClient]);
  const governanceAdaptationCameraTokenRef = useRef(0);
  const [ragPanelOpen, setRagPanelOpen] = useState(false);
  const [adaptationGovernanceTargets, setAdaptationGovernanceTargets] = useState<readonly GovernanceTaskEntry[]>([]);
  const [activeGovernanceTaskId, setActiveGovernanceTaskId] = useState("session");
  useEffect(() => {
    if (activeWorkspace !== "governance" || activeGovernanceTaskId === "session") return;
    setAdaptationGovernanceTargets((items) => items.map((item) => item.id === activeGovernanceTaskId
      ? { ...item, validationToken: (item.validationToken ?? 0) + 1 }
      : item));
  }, [activeGovernanceTaskId, activeWorkspace]);
  const [governanceServiceState, setGovernanceServiceState] = useState<
    | { readonly state: "checking" }
    | { readonly state: "ready"; readonly device: "cpu"; readonly modelVersionId: string; readonly modelStateHash: string }
    | { readonly state: "model_unavailable" }
    | { readonly state: "unavailable" }
  >({ state: "checking" });
  const researchScenarioEpochRef = useRef(0);
  const currentGraphVersionRef = useRef<GraphVersion | null>(graphVersion);
  const currentViewStateRef = useRef(viewState);
  const currentImportStateRef = useRef(importState);
  const currentOverlayRef = useRef<AnalysisOverlay | null>(activeOverlay);
  const researchUserContextRef = useRef<{
    readonly graph: GraphVersion;
    readonly viewState: GraphViewState;
    readonly importState: ImportViewState;
    readonly overlay: AnalysisOverlay | null;
  } | null>(null);
  currentGraphVersionRef.current = graphVersion;
  currentViewStateRef.current = viewState;
  currentImportStateRef.current = importState;
  currentOverlayRef.current = activeOverlay;
  const governanceGraphPresentationRef = useRef<GovernanceGraphPresentation | null>(governanceGraphPresentation);
  governanceGraphPresentationRef.current = governanceGraphPresentation;
  const graphVersionOverlayControllerRef = useRef<GraphVersionOverlayController | null>(null);
  if (!graphVersionOverlayControllerRef.current) {
    graphVersionOverlayControllerRef.current = new GraphVersionOverlayController({
      computeDefaultOverlay: async (version, activation) => {
        const result = await runGraphTask({
          id: `community-${version.id}-${activation}`,
          kind: "community",
          graph: version,
          seed: version.id,
        });
        if (!result || typeof result !== "object" || !("assignments" in result)) return null;
        return buildCommunityOverlay(version, result);
      },
      onOverlayChange: setActiveOverlay,
      onError: (error) => {
        showToast(error instanceof GraphWorkerExecutionError
          ? "后台图分析未完成，已停止且未在主线程重跑；请缩小图范围后重试。"
          : "后台图分析失败，请稍后重试。");
      },
    });
  }
  const graphVersionOverlayController = graphVersionOverlayControllerRef.current;
  const importRequestEpochRef = useRef(0);
  const intentRequestEpochRef = useRef(0);
  const governanceChatRequestEpochRef = useRef(0);
  const governanceChatRequestAbortRef = useRef<AbortController | null>(null);
  const confirmingChatMessageIdsRef = useRef(new Set<string>());
  const completedCoreReportsRef = useRef(new Set<string>());
  const activeSessionIdRef = useRef(activeSessionId);
  activeSessionIdRef.current = activeSessionId;

  useEffect(() => {
    for (const entry of messages) {
      if (entry.role === "assistant" && entry.governanceRunId) {
        completedCoreReportsRef.current.add(entry.governanceRunId);
      }
    }
  }, [messages]);

  const cacheWorkspaceCamera = useCallback((
    workspace: SidebarWorkspace,
    graphIdentity: string,
    lens: string,
    snapshot: GraphCameraSnapshot,
  ) => {
    const key = Object.freeze({ workspace, graphIdentity, lens });
    graphCameraSnapshotCacheRef.current.set(key, snapshot);
  }, []);

  const currentWorkspaceCameraKey = useCallback((workspace: SidebarWorkspace): GraphCameraSnapshotCacheKey | null => {
    if (workspace === "chat") {
      const graphIdentity = currentGraphVersionRef.current?.id;
      return graphIdentity ? { workspace, graphIdentity, lens: "overview" } : null;
    }
    if (workspace === "governance") {
      const presentation = governanceGraphPresentationRef.current;
      const graphIdentity = presentation?.graph?.id ?? governanceViewSnapshotRef.current?.graphVersionId;
      return graphIdentity ? { workspace, graphIdentity, lens: presentation?.lens ?? "risk" } : null;
    }
    const current = adaptationLanePresentationRef.current;
    const graphIdentity = current.lanes[current.activeLane].graph?.id;
    return graphIdentity ? {
      workspace,
      graphIdentity,
      lens: adaptationCameraLens(current.activeLane),
    } : null;
  }, []);

  const requestWorkspaceCameraRestore = useCallback((key: GraphCameraSnapshotCacheKey | null) => {
    const camera = resolveWorkspaceCameraSnapshot(graphCameraSnapshotCacheRef.current, key);
    if (!camera || !key) {
      setWorkspaceCameraRestore(null);
      return;
    }
    workspaceCameraRestoreTokenRef.current += 1;
    setWorkspaceCameraRestore({
      ...camera,
      workspace: key.workspace as SidebarWorkspace,
      x: Number(camera.position[0]),
      y: Number(camera.position[1]),
      token: workspaceCameraRestoreTokenRef.current,
    });
  }, []);

  const cacheActiveAdaptationCamera = useCallback(() => {
    const current = adaptationLanePresentationRef.current;
    const graph = current.lanes[current.activeLane].graph;
    const view = adaptationViewSnapshotRef.current;
    const root = document.querySelector<HTMLElement>('[aria-label="适配任务关系图"]');
    const viewport = root?.querySelector<HTMLElement>(".graph-preview__viewport");
    if (!graph || !view || view.graphVersionId !== graph.id || !root || !viewport) return;
    const x = Number(root.dataset.cameraX);
    const y = Number(root.dataset.cameraY);
    const zoom = Number(root.dataset.cameraZoom);
    const worldX = Number(root.dataset.worldCenterX);
    const worldY = Number(root.dataset.worldCenterY);
    if (![x, y, zoom, worldX, worldY].every(Number.isFinite)
      || viewport.clientWidth <= 0 || viewport.clientHeight <= 0) return;
    cacheWorkspaceCamera(
      "adaptation",
      graph.id,
      adaptationCameraLens(current.activeLane),
      {
        ...view.camera,
        position: [x, y],
        zoom,
        worldCenter: [worldX, worldY],
        viewportSize: [viewport.clientWidth, viewport.clientHeight],
      },
    );
  }, [cacheWorkspaceCamera]);

  useEffect(() => {
    const updateViewport = () => {
      setViewportWidth(window.innerWidth);
      setViewportHeight(window.innerHeight);
    };
    window.addEventListener("resize", updateViewport);
    return () => window.removeEventListener("resize", updateViewport);
  }, []);

  useEffect(() => {
    const restoreWorkspaceRoute = () => {
      const route = workspaceRouteFromHash(window.location.hash);
      if (route === "datasets") {
        setWorkspacePanel("datasets");
        return;
      }
      setWorkspacePanel((current) => current === "datasets" ? null : current);
      const workspace = sidebarWorkspaceFromRoute(route);
      if (workspace === "governance" || workspace === "adaptation") setGovernanceMounted(true);
      requestWorkspaceCameraRestore(currentWorkspaceCameraKey(workspace));
      setActiveWorkspace(workspace);
      setMobilePanel(workspace === "governance" ? "governance" : workspace === "adaptation" ? "adaptation" : "chat");
    };
    window.addEventListener("popstate", restoreWorkspaceRoute);
    window.addEventListener("hashchange", restoreWorkspaceRoute);
    return () => {
      window.removeEventListener("popstate", restoreWorkspaceRoute);
      window.removeEventListener("hashchange", restoreWorkspaceRoute);
    };
  }, [currentWorkspaceCameraKey, requestWorkspaceCameraRestore]);

  useEffect(() => {
    const route = workspacePanel === "datasets" ? "datasets" : routeForSidebarWorkspace(activeWorkspace);
    const nextHash = hashForWorkspaceRoute(route);
    if (window.location.hash !== nextHash) window.history.replaceState(null, "", nextHash);
  }, [activeWorkspace, workspacePanel]);

  const bindGovernanceWorkspaceSnapshot = useCallback(async (next: GovernanceWorkspaceSnapshot) => {
    const previous = governanceWorkspaceSnapshotRef.current;
    if (previous?.sessionId === next.sessionId
      && !sameGovernanceArtifactIdentity(previous.artifact, next.artifact)) {
      governanceChatRequestEpochRef.current += 1;
      governanceChatRequestAbortRef.current?.abort();
      governanceChatRequestAbortRef.current = null;
      setIsSending(false);
      setMessages(invalidateChatConfirmations);
    }
    governanceWorkspaceSnapshotRef.current = next;
    await governanceWorkspace.bindSnapshot(next);
  }, [governanceWorkspace.bindSnapshot]);
  const currentGraphVersionIdRef = useRef<string | null>(null);
  useEffect(() => {
    currentGraphVersionIdRef.current = graphVersion?.id ?? null;
  }, [graphVersion]);
  useEffect(() => {
    saveGovernanceTheme(governanceTheme);
  }, [governanceTheme]);
  useEffect(() => {
    void governanceWorkspace.activateSession(activeSessionId);
  }, [activeSessionId, governanceWorkspace.activateSession]);
  useEffect(() => {
    governanceChatRequestEpochRef.current += 1;
    governanceChatRequestAbortRef.current?.abort();
    governanceChatRequestAbortRef.current = null;
    setIsSending(false);
    setMessages(invalidateChatConfirmations);
  }, [activeSessionId]);
  useEffect(() => {
    const controller = new AbortController();
    let active = true;
    setGovernanceServiceState({ state: "checking" });
    void Promise.all([
      governanceOnlineClient.health(controller.signal),
      governanceOnlineClient.capabilities(controller.signal),
    ]).then(([health, capabilities]) => {
      if (!active) return;
      const identityMatches = health.modelVersionId === capabilities.modelVersionId
        && health.modelVersionHash === capabilities.modelVersionHash
        && health.modelStateHash === capabilities.modelStateHash;
      if (health.servingReady && health.onlineForwardReady
        && capabilities.servingReady && capabilities.onlineForwardReady && identityMatches) {
        setGovernanceServiceState({
          state: "ready",
          device: health.device,
          modelVersionId: capabilities.modelVersionId!,
          modelStateHash: capabilities.modelStateHash!,
        });
      } else {
        setGovernanceServiceState({ state: "model_unavailable" });
      }
    }).catch(() => { if (active) setGovernanceServiceState({ state: "unavailable" }); });
    return () => { active = false; controller.abort(); };
  }, [governanceOnlineClient]);
  const activateGraphVersionOverlay = useCallback((
    version: GraphVersion,
    explicitOverlay: AnalysisOverlay | null | undefined = currentOverlayRef.current,
  ) => {
    graphVersionOverlayController.activate(version, explicitOverlay ?? null);
  }, [graphVersionOverlayController]);
  const graphScene = useMemo(
    () => {
      if (!graphVersion) return null;
      const effectiveView = viewState.graphVersionId === graphVersion.id
        ? viewState
        : createDefaultGraphViewState(graphVersion.id);
      if (
        analysisSceneOverride?.scene.graphVersionId === graphVersion.id
        && analysisSceneOverride.viewKey === analysisViewKey(effectiveView)
      ) {
        return Object.freeze({
          ...analysisSceneOverride.scene,
          ...(activeOverlay?.graphVersionId === graphVersion.id ? { overlay: activeOverlay } : {}),
        });
      }
      return buildGraphScene(graphVersion, {
        viewState: effectiveView,
        ...(activeOverlay?.graphVersionId === graphVersion.id ? { overlay: activeOverlay } : {}),
      });
    },
    [
      activeOverlay,
      analysisSceneOverride,
      graphVersion,
      viewState.depth,
      viewState.filters,
      viewState.focusNodeIds,
      viewState.graphVersionId,
      viewState.mode,
      viewState.pathEndpointIds,
    ],
  );

  useEffect(() => {
    const cancelGraphTool = (event: KeyboardEvent) => {
      if (event.key === "Escape") dispatchGraphView({ type: "cancel_tool" });
    };
    window.addEventListener("keydown", cancelGraphTool);
    return () => window.removeEventListener("keydown", cancelGraphTool);
  }, [dispatchGraphView]);

  useEffect(() => {
    let active = true;
    void intentNormalizer.checkStatus().then((status) => {
      if (active) setIntentServiceStatus(status);
    });
    return () => { active = false; };
  }, [intentNormalizer]);

  useEffect(() => {
    let active = true;
    const abortController = new AbortController();
    void coreClient.capabilities(abortController.signal)
      .then((capabilities) => {
        if (active) setCoreService({ state: "connected", capabilities });
      })
      .catch((error) => {
        if (!active) return;
        setCoreService({
          state: "unavailable",
          code: error instanceof SocialGraphApiError ? error.code : "GFM_CORE_CAPABILITIES_UNAVAILABLE",
        });
      });
    return () => {
      active = false;
      abortController.abort();
    };
  }, [coreClient]);

  useEffect(() => {
    let active = true;
    const abortController = new AbortController();
    void researchClient.capabilities(abortController.signal)
      .then((capabilities) => {
        if (active) setResearchService({ state: "connected", capabilities });
      })
      .catch((error) => {
        if (!active) return;
        setResearchService({
          state: "unavailable",
          code: error instanceof SocialGraphApiError ? error.code : "GFM_RESEARCH_CAPABILITIES_UNAVAILABLE",
        });
      });
    return () => {
      active = false;
      abortController.abort();
    };
  }, [researchClient]);

  useEffect(() => {
    let active = true;
    void requestPersistentGraphStorage().then((capacity) => {
      if (!active) return;
      setStorageCapacity(capacity);
    });
    return () => { active = false; };
  }, []);

  const refreshSessionLists = useCallback(async () => {
    const [active, trashed] = await Promise.all([
      graphRepository.listSessions("active"),
      graphRepository.listSessions("trashed"),
    ]);
    // The first release no longer advertises built-in fixtures. Keep their
    // records intact so existing local work can still be recovered explicitly.
    const hiddenBuiltIns = new Set([DEMO_SESSION_ID, "volunteer-network", "industry-partners"]);
    const visibleActive = active.filter((session) => !hiddenBuiltIns.has(session.id));
    setSessions(visibleActive);
    setTrashedSessions(trashed);
    return visibleActive;
  }, [graphRepository]);

  useEffect(() => {
    let active = true;
    const initializeWorkspace = async () => {
      const initialization = await graphRepository.getInitializationMetadata();
      if (!initialization) {
        const now = new Date().toISOString();
        await graphRepository.saveInitializationMetadata({
          initializedAt: now,
          seededDemoVersion: 0,
          updatedAt: now,
        });
      }
      let storedSessions = await refreshSessionLists();
      if (!storedSessions.length) {
        const blank = createResearchSession(`新建研究会话 · ${timeNow()}`);
        await graphRepository.saveSession(blank);
        storedSessions = await refreshSessionLists();
      }
      if (!active) return;
      const rememberedId = window.localStorage.getItem("socialgraph-fm-active-session");
      const selectedSession = storedSessions.find((session) => session.id === rememberedId)
        ?? storedSessions[0];
      if (!selectedSession) return;
      setActiveSessionId(selectedSession.id);
      window.localStorage.setItem("socialgraph-fm-active-session", selectedSession.id);
      setSessionReady(true);
    };
    void initializeWorkspace();
    return () => { active = false; };
  }, [graphRepository, refreshSessionLists]);

  useEffect(() => {
    if (!sessionReady) return;
    let active = true;
    const hydrateSession = async () => {
      importRequestEpochRef.current += 1;
      intentRequestEpochRef.current += 1;
      setPendingImport(null);
      const session = await graphRepository.getSession(activeSessionId);
      if (!session || session.lifecycle !== "active") return;
      const storedMessages = await graphRepository.listMessages(session.id);
      let previousUserText = "";
      const restoredMessages: ChatEntry[] = [];
      for (const stored of storedMessages) {
        const attachment = (stored as ConversationMessage & {
          readonly attachment?: { readonly name: string; readonly size: number };
        }).attachment;
        const attachments = stored.attachments?.map((item) => ({ name: item.name, size: item.size }))
          ?? (attachment ? [{ name: attachment.name, size: attachment.size }] : []);
        if (stored.role === "user") {
          previousUserText = stored.text;
          restoredMessages.push({
            id: stored.id,
            role: "user",
            text: stored.text,
            timestamp: messageTime(stored.createdAt),
            ...(attachments[0] ? { file: attachments[0] } : {}),
            ...(attachments.length ? { files: attachments } : {}),
          });
          continue;
        }
        let status = stored.status;
        if (status === "pending") {
          status = "interrupted";
          await graphRepository.saveMessage({ ...stored, status });
        }
        const run = stored.analysisRunId
          ? await graphRepository.getAnalysisRun(stored.analysisRunId)
          : undefined;
        const intent = stored.intent?.kind === "analysis_request" ? stored.intent : undefined;
        const governanceRunId = governanceRunIdFromStoredMessage(stored);
        restoredMessages.push({
          id: stored.id,
          role: "assistant",
          text: status === "interrupted" ? "上次请求在页面关闭前没有完成，可以安全重试。" : stored.text,
          timestamp: messageTime(stored.createdAt),
          state: status === "completed" ? "success"
            : status === "warning" ? "warning"
              : "error",
          ...(intent ? { intent, intentMeta: intent.meta } : {}),
          ...(run ? { run } : {}),
          ...(governanceRunId ? { governanceRunId } : {}),
          ...(status === "interrupted" && previousUserText ? { retryText: previousUserText } : {}),
          ...(session.id === DEMO_SESSION_ID ? { demo: true } : {}),
        });
      }
      if (!active) return;
      setMessages(restoredMessages);
      setHeroManuallyOpen(false);
      graphVersionOverlayController.deactivate();
      setPendingTargetResolution(null);
      if (!session.graphVersionId) {
        setGraphVersion(null);
        setAnalysisSceneOverride(null);
        replaceViewState(createDefaultGraphViewState("empty"));
        setImportState({ kind: "idle" });
        setHydratedSessionId(session.id);
        return;
      }
      const storedGraph = await graphRepository.getGraphVersion(session.graphVersionId);
      if (!active || !storedGraph) return;
      analysisExecutor.registerGraphVersion(storedGraph);
      setGraphVersion(storedGraph);
      setAnalysisSceneOverride(null);
      activateGraphVersionOverlay(storedGraph, null);
      const storedView = await graphRepository.getViewState(storedGraph.id);
      replaceViewState(normalizeGraphViewState(storedGraph.id, storedView));
      setImportState({ kind: "success", fileName: storedGraph.sourceFile, version: storedGraph });
      setHydratedSessionId(session.id);
    };
    void hydrateSession();
    return () => { active = false; };
  }, [activeSessionId, activateGraphVersionOverlay, analysisExecutor, graphRepository, graphVersionOverlayController, replaceViewState, sessionReady]);

  useEffect(() => {
    saveWorkspaceLayout(workspaceLayout);
  }, [workspaceLayout]);

  useEffect(() => {
    if (!graphVersion || viewState.graphVersionId !== graphVersion.id) return;
    const timeout = window.setTimeout(() => {
      void graphRepository.saveViewState(viewState);
    }, 220);
    return () => window.clearTimeout(timeout);
  }, [graphRepository, graphVersion, viewState]);

  const persistChatEntry = useCallback(
    async (entry: ChatEntry, intentResult?: ConversationMessage["intent"]) => {
      const governanceRunId = governanceRunIdForPersistence(entry);
      const message: ConversationMessage = {
        id: entry.id,
        sessionId: activeSessionId,
        role: entry.role,
        text: entry.text,
        status: messageStatusFromEntry(entry),
        createdAt: new Date().toISOString(),
        ...(entry.role === "user" && (entry.files?.length || entry.file)
          ? {
              attachment: {
                name: (entry.files?.[0] ?? entry.file!).name,
                size: (entry.files?.[0] ?? entry.file!).size,
                kind: "file" as const,
              },
              attachments: (entry.files ?? [entry.file!]).map((file) => ({
                name: file.name,
                size: file.size,
                kind: "file" as const,
              })),
            }
          : {}),
        ...(intentResult ? { intent: intentResult } : {}),
        ...(entry.role === "assistant" && entry.run ? { analysisRunId: entry.run.id } : {}),
        ...(governanceRunId ? { governanceRunId } : {}),
      };
      await graphRepository.saveMessage(message);
    },
    [activeSessionId, graphRepository],
  );

  const handleGraphExport = useCallback(
    ({ format }: { format: "png" | "json" }) => {
      showToast(format === "png" ? "图谱 PNG 已导出" : "视图配置 JSON 已导出");
    },
    [showToast],
  );

  const registerImportedGraph = useCallback(
    async (
      version: GraphVersion,
      file: File,
      artifacts: readonly SourceArtifact[] = [],
      sourceMessageId?: string,
    ) => {
      const persistedVersion: GraphVersion = version.provenance
        ? version
        : Object.freeze({
            ...version,
            provenance: browserImportProvenance(
              version.parentVersionId ? "construction_revision" : undefined,
            ),
          });
      const nextView = createDefaultGraphViewState(persistedVersion.id);
      const updatedSession = createResearchSession(sessionTitleFromFile(file.name), {
        id: activeSessionId,
        graphVersionId: persistedVersion.id,
      });
      const importEvent = createSemanticEvent("graph_imported", {
        graphVersionId: persistedVersion.id,
        sessionId: activeSessionId,
        payload: {
          nodeCount: persistedVersion.summary.nodeCount,
          edgeCount: persistedVersion.summary.edgeCount,
          contentHash: persistedVersion.contentHash ?? "legacy",
          parentVersionId: persistedVersion.parentVersionId ?? "none",
        },
      });
      await graphRepository.saveImportBundle({
        sourceArtifacts: artifacts,
        graphVersion: persistedVersion,
        viewState: nextView,
        session: updatedSession,
        event: importEvent,
        ...(sourceMessageId ? { sourceMessageId } : {}),
      });
      analysisExecutor.registerGraphVersion(persistedVersion);
      setGraphVersion(persistedVersion);
      setAnalysisSceneOverride(null);
      replaceViewState(nextView);
      activateGraphVersionOverlay(persistedVersion);
      setPendingTargetResolution(null);
      setImportState({ kind: "success", fileName: file.name, version: persistedVersion });
      setHeroManuallyOpen(false);
      setMobilePanel("graph");
      showToast(persistedVersion.parentVersionId ? "图谱修订已保存" : "图谱已生成");
      setSessions(await graphRepository.listSessions());
    },
    [activeSessionId, activateGraphVersionOverlay, analysisExecutor, graphRepository, replaceViewState, showToast],
  );

  const registerDatasetArtifact = useCallback(
    async (artifact: DatasetArtifact, graphVersionIdOverride?: string) => {
      const artifactVersion = graphVersionFromDatasetArtifact(artifact);
      const version: GraphVersion = graphVersionIdOverride
        ? Object.freeze({ ...artifactVersion, id: graphVersionIdOverride })
        : artifactVersion;
      const projection = version.datasetArtifact?.scope === "projection";
      const nextView = createDefaultGraphViewState(version.id);
      analysisExecutor.registerGraphVersion(version);
      setGraphVersion(version);
      setAnalysisSceneOverride(null);
      replaceViewState(nextView);
      // Compute presentation-only communities away from the main thread. The
      // epoch guard prevents a late result from colouring a newer graph.
      activateGraphVersionOverlay(version);
      setPendingTargetResolution(null);
      setImportState({ kind: "success", fileName: version.sourceFile, version });
      setHeroManuallyOpen(false);
      setMobilePanel("graph");
      setWorkspacePanel(null);
      const updatedSession = createResearchSession(artifact.datasetName ?? "科研数据集", {
        id: activeSessionId,
        graphVersionId: version.id,
      });
      await graphRepository.saveGraphVersion(version);
      await graphRepository.saveViewState(nextView);
      await graphRepository.saveSession(updatedSession);
      await graphRepository.saveEvent(createSemanticEvent("graph_imported", {
        graphVersionId: version.id,
        sessionId: activeSessionId,
        payload: {
          nodeCount: version.summary.nodeCount,
          edgeCount: version.summary.edgeCount,
          datasetArtifactId: artifact.id,
          scope: version.datasetArtifact?.scope ?? "complete",
        },
      }));
      setSessions(await graphRepository.listSessions());
      showToast(projection ? "已打开科研数据投影" : "科研数据 Artifact 已载入");
    },
    [activeSessionId, activateGraphVersionOverlay, analysisExecutor, graphRepository, replaceViewState, showToast],
  );

  const displayResearchScenario = useCallback((version: GraphVersion): void => {
    const current = currentGraphVersionRef.current;
    if (
      !researchUserContextRef.current
      && current
      && !current.datasetArtifact?.id.startsWith("research-preview:")
      && !current.datasetArtifact?.id.startsWith("governance-preview:")
    ) {
      researchUserContextRef.current = {
        graph: current,
        viewState: currentViewStateRef.current,
        importState: currentImportStateRef.current,
        overlay: currentOverlayRef.current,
      };
    }
    const nextView = createDefaultGraphViewState(version.id);
    analysisExecutor.registerGraphVersion(version);
    setGraphVersion(version);
    setAnalysisSceneOverride(null);
    replaceViewState(nextView);
    activateGraphVersionOverlay(version, null);
    setPendingTargetResolution(null);
    setImportState({ kind: "success", fileName: version.sourceFile, version });
    setHeroManuallyOpen(false);
    setWorkspacePanel(null);
  }, [activateGraphVersionOverlay, analysisExecutor, replaceViewState]);

  const restoreResearchUserGraph = useCallback((): void => {
    researchScenarioEpochRef.current += 1;
    const context = researchUserContextRef.current;
    if (!context) return;
    researchUserContextRef.current = null;
    analysisExecutor.registerGraphVersion(context.graph);
    setGraphVersion(context.graph);
    setAnalysisSceneOverride(null);
    replaceViewState(context.viewState);
    activateGraphVersionOverlay(context.graph, context.overlay);
    setPendingTargetResolution(null);
    setImportState(context.importState);
  }, [activateGraphVersionOverlay, analysisExecutor, replaceViewState]);

  const handleResearchSourceModeChange = useCallback((mode: "examples" | "my-graph"): void => {
    if (mode === "my-graph") restoreResearchUserGraph();
  }, [restoreResearchUserGraph]);

  const handleResearchScenarioSelect = useCallback(async (scenario: ResearchScenario): Promise<boolean> => {
    if (!scenario.graphVersionHash) return false;
    const epoch = ++researchScenarioEpochRef.current;
    const current = currentGraphVersionRef.current;
    if (
      !researchUserContextRef.current
      && current
      && !current.datasetArtifact?.id.startsWith("research-preview:")
      && !current.datasetArtifact?.id.startsWith("governance-preview:")
    ) {
      researchUserContextRef.current = {
        graph: current,
        viewState: currentViewStateRef.current,
        importState: currentImportStateRef.current,
        overlay: currentOverlayRef.current,
      };
    }
    try {
      const artifacts = await researchDatasetClient.listArtifacts();
      if (epoch !== researchScenarioEpochRef.current) return false;
      const matching = artifacts.find((artifact) => (
        artifact.scope !== "projection"
        && artifact.canonicalGraphHash === scenario.graphVersionHash
      ));
      if (matching) {
        const artifact = await researchDatasetClient.getArtifact(matching.id);
        if (epoch !== researchScenarioEpochRef.current || artifact.canonicalGraphHash !== scenario.graphVersionHash) return false;
        const artifactVersion = graphVersionFromDatasetArtifact(artifact);
        displayResearchScenario(Object.freeze({ ...artifactVersion, id: scenario.graphVersionId }));
        return true;
      }
    } catch {
      // A registered scenario remains usable when the optional DatasetStore listing is unavailable.
    }

    const preview = await researchClient.scenarioPreview(scenario.scenarioId);
    if (epoch !== researchScenarioEpochRef.current) return false;
    if (
      preview.graphVersionId !== scenario.graphVersionId
      || preview.graphVersionHash !== scenario.graphVersionHash
      || preview.modelVersionId !== scenario.modelVersionId
    ) return false;
    const projection = createGraphVersion(
      `SocialGraph-FM Research · ${scenario.datasetId}`,
      preview.nodes.map((node) => ({
        id: node.id,
        label: governanceAccountLabel(node.label, node.id),
        type: "research-node",
        attributes: Object.freeze({ researchScenarioId: scenario.scenarioId }),
      })),
      preview.edges.map((edge) => ({
        id: edge.id,
        source: edge.source,
        target: edge.target,
        type: "research-relation",
        directed: edge.directed,
        attributes: Object.freeze({ researchScenarioId: scenario.scenarioId }),
      })),
      [{
        code: "research_scenario_projection",
        severity: "info",
        message: "当前画布是后端登记图的只读投影；推理绑定完整图哈希。",
        details: { nodeCount: preview.nodeCount, edgeCount: preview.edgeCount },
      }],
      {
        provenance: {
          origin: "research_dataset",
          pipeline: "dataset-artifact",
          pipelineVersion: "research-preview/1.0",
          sourceHashScheme: "dataset-content-hash-v2",
        },
      },
    );
    const version: GraphVersion = Object.freeze({
      ...projection,
      id: preview.graphVersionId,
      preview: Object.freeze({
        ...projection.preview,
        truncated: preview.partialPreview || projection.preview.truncated,
        originalNodeCount: preview.nodeCount,
        originalEdgeCount: preview.edgeCount,
      }),
      truncated: preview.partialPreview || projection.preview.truncated,
      datasetArtifact: Object.freeze({
        id: `research-preview:${scenario.scenarioId}`,
        datasetName: scenario.datasetId,
        checksum: preview.previewHash,
        canonicalGraphHash: preview.graphVersionHash,
        contentHash: preview.previewHash,
        scope: "projection",
      }),
    });
    displayResearchScenario(version);
    return true;
  }, [
    displayResearchScenario,
    researchDatasetClient,
    researchClient,
  ]);

  const prepareResearchGraph = useCallback(async (version: GraphVersion) => {
    const prepared = await researchDatasetClient.prepareGraphVersionTargetDomain(version, "gfm_research");
    return {
      graphVersionHash: prepared.artifact.canonicalGraphHash,
      compatibility: prepared.researchCompatibility ?? null,
    };
  }, [researchDatasetClient]);

  const adaptationOverviewGraph = activeAdaptationPresentation.graph;

  const handleAdaptationGraphNodeSelect = useCallback((node: GraphNode | null) => {
    const targetGraph = adaptationOverviewGraph;
    if (!targetGraph) {
      setAdaptationLanePresentation((state) => updateAdaptationLanePresentation(state, state.activeLane, { focus: undefined, camera: undefined }));
      return;
    }
    if (node) {
      const token = ++governanceAdaptationCameraTokenRef.current;
      setAdaptationLanePresentation((state) => updateAdaptationLanePresentation(state, state.activeLane, {
        focus: { kind: "node", targetId: node.id, nodeIds: [node.id], cameraToken: token },
        camera: undefined,
      }));
      return;
    }
    setAdaptationLanePresentation((state) => updateAdaptationLanePresentation(state, state.activeLane, {
      focus: undefined,
      camera: undefined,
    }));
  }, [adaptationOverviewGraph]);

  const restoreAdaptationOverview = useCallback(() => {
    handleAdaptationGraphNodeSelect(null);
    requestWorkspaceCameraRestore(currentWorkspaceCameraKey("adaptation"));
  }, [currentWorkspaceCameraKey, handleAdaptationGraphNodeSelect, requestWorkspaceCameraRestore]);

  const switchAdaptationGraph = useCallback((lane: AdaptationLane) => {
    const current = adaptationLanePresentationRef.current;
    if (
      lane === current.activeLane
      || !current.lanes.zero_shot.graph
      || !current.lanes.few_shot.graph
    ) return;
    const currentGraph = current.lanes[current.activeLane].graph;
    const currentSnapshot = adaptationViewSnapshotRef.current;
    if (
      currentGraph
      && currentSnapshot
      && !current.lanes[current.activeLane].focus
      && (!currentSnapshot.graphVersionId || currentSnapshot.graphVersionId === currentGraph.id)
    ) {
      cacheWorkspaceCamera(
        "adaptation",
        currentGraph.id,
        adaptationCameraLens(current.activeLane),
        currentSnapshot.camera,
      );
    }
    const targetGraph = current.lanes[lane].graph;
    setAdaptationLanePresentation((state) => activateAdaptationLanePresentation(state, lane));
    const targetKey: GraphCameraSnapshotCacheKey | null = targetGraph ? {
      workspace: "adaptation",
      graphIdentity: targetGraph.id,
      lens: adaptationCameraLens(lane),
    } : null;
    const targetCamera = resolveWorkspaceCameraSnapshot(
      graphCameraSnapshotCacheRef.current,
      targetKey,
    );
    if (!targetCamera) {
      setWorkspaceCameraRestore(null);
      return;
    }
    workspaceCameraRestoreTokenRef.current += 1;
    setWorkspaceCameraRestore({
      workspace: "adaptation",
      x: Number(targetCamera.position[0]),
      y: Number(targetCamera.position[1]),
      zoom: targetCamera.zoom,
      token: workspaceCameraRestoreTokenRef.current,
    });
  }, [cacheWorkspaceCamera]);

  const preparePendingImport = useCallback(
    async (
      files: readonly File[],
      profiles: readonly FileProfile[],
      edgeIndex: number | undefined,
      baseGraphVersionId: string | undefined,
    ) => {
      const requestEpoch = ++importRequestEpochRef.current;
      setImportState({ kind: "parsing", fileName: files.map((file) => file.name).join(" + "), stage: "inspect" });
      const roles: SourceArtifact["role"][] = files.length === 1
        ? ["single"]
        : files.map((_file, index) => index === edgeIndex ? "edges" : "nodes");
      let artifacts: readonly SourceArtifact[];
      try {
        artifacts = await Promise.all(files.map((file, index) => createSourceArtifact(file, roles[index])));
      } catch (error) {
        if (requestEpoch === importRequestEpochRef.current) {
          setImportState({
            kind: "error",
            fileName: files.map((file) => file.name).join(" + "),
            message: error instanceof Error ? error.message : "文件准备失败。",
            issues: [],
          });
        }
        return;
      }
      if (requestEpoch !== importRequestEpochRef.current) return;
      if (currentGraphVersionIdRef.current !== (baseGraphVersionId ?? null)) {
        setImportState({
          kind: "error",
          fileName: files.map((file) => file.name).join(" + "),
          message: "当前图谱已变化；请重新附加文件以建立明确的版本关系。",
          issues: [],
        });
        return;
      }
      setPendingImport({
        files: Object.freeze([...files]),
        profiles: Object.freeze([...profiles]),
        artifacts: Object.freeze(artifacts),
        requestToken: makeId("import-request"),
        ...(baseGraphVersionId ? { baseGraphVersionId } : {}),
      });
      setImportState({ kind: "idle" });
      setHeroManuallyOpen(false);
      setMobilePanel("chat");
      showToast(files.length === 2 ? "nodes + edges 角色已确认，请补充说明后提交" : "文件已附加，请确认或补充数据说明");
    },
    [showToast],
  );

  const handleGovernanceZip = useCallback(async (file: File) => {
    const requestEpoch = ++importRequestEpochRef.current;
    const submittedSessionId = activeSessionId;
    const requestIsCurrent = () => requestEpoch === importRequestEpochRef.current
      && activeSessionIdRef.current === submittedSessionId;
    // Replacing the package in the same session invalidates any older
    // confirmation/polling flow before it can write its run back to the new
    // binding. The server-side run may finish, but it is no longer attached to
    // the newly selected input.
    governanceChatRequestEpochRef.current += 1;
    governanceChatRequestAbortRef.current?.abort();
    governanceChatRequestAbortRef.current = null;
    setIsSending(false);
    setMessages(invalidateChatConfirmations);
    setPendingImport(null);
    setImportState({ kind: "parsing", fileName: file.name, stage: "inspect" });
    try {
      const compatibility = await governanceOnlineClient.inspectArtifact(file);
      if (!requestIsCurrent()) return;
      if (!compatibility.compatible) {
        throw new Error(compatibility.issues.join("；") || ORDINARY_PRESENTATION_COPY.governanceCompatibilityError);
      }
      const cleanSelfLoops = compatibility.requiresSelfLoopCleaning
        ? window.confirm(`检测到 ${compatibility.selfLoopsDetected} 条自环。是否创建记录清洗配方的兼容版本？原文件不会被修改。`)
        : false;
      if (compatibility.requiresSelfLoopCleaning && !cleanSelfLoops) {
        setImportState({ kind: "idle" });
        showToast("已取消登记；推理包和当前图均未修改");
        return;
      }
      const artifact = await governanceOnlineClient.uploadArtifact(file, cleanSelfLoops);
      if (!requestIsCurrent()) return;
      const preview = await governanceOnlineClient.preview(artifact.artifactId, undefined, {
        preset: "overview", nodeBudget: 120, edgeBudget: 240,
      });
      if (!requestIsCurrent()) return;
      if (artifact.artifactId !== preview.artifactId
        || artifact.datasetContentHash !== preview.datasetContentHash
        || artifact.graphVersionHash !== preview.graphVersionHash) {
        throw new Error("推理包登记身份与图谱预览不一致。");
      }
      const version = governanceImportedGraphVersion(preview, governanceArtifactDisplayName(artifact, file.name));
      const nextView = createDefaultGraphViewState(version.id);
      if (!requestIsCurrent()) return;
      const currentSession = await graphRepository.getSession(submittedSessionId);
      if (!requestIsCurrent()) return;
      if (!currentSession) throw new Error("当前研究会话已不存在，请重新选择推理包。");
      const updatedSession = createResearchSession(currentSession.title, {
        id: currentSession.id,
        graphVersionId: version.id,
        lifecycle: currentSession.lifecycle,
        ...(currentSession.deletedAt ? { deletedAt: currentSession.deletedAt } : {}),
      });
      await graphRepository.saveImportBundle({
        sourceArtifacts: [],
        graphVersion: version,
        viewState: nextView,
        session: updatedSession,
        event: createSemanticEvent("graph_imported", {
          graphVersionId: version.id,
          sessionId: submittedSessionId,
          payload: {
            nodeCount: version.summary.nodeCount,
            edgeCount: version.summary.edgeCount,
            datasetArtifactId: artifact.artifactId,
            scope: version.datasetArtifact?.scope ?? "complete",
          },
        }),
      }, requestIsCurrent);
      if (!requestIsCurrent()) return;
      const snapshot = Object.freeze({
        schemaVersion: GOVERNANCE_WORKSPACE_SCHEMA,
        sessionId: submittedSessionId,
        sourceFileName: file.name,
        artifact,
        preview,
        updatedAt: new Date().toISOString(),
      } satisfies GovernanceWorkspaceSnapshot);
      await bindGovernanceWorkspaceSnapshot(snapshot);
      if (!requestIsCurrent()) return;
      const uploadEntries = buildGovernanceUploadConversationEntries(file, timeNow(), artifact);
      for (const entry of uploadEntries) await persistChatEntry(entry);
      if (!requestIsCurrent()) return;
      setGraphVersion(version);
      replaceViewState(nextView);
      activateGraphVersionOverlay(version);
      setImportState({ kind: "success", fileName: file.name, version });
      setMessages((current) => [...mergeGovernanceUploadConversationEntries(current, uploadEntries)]);
      setHeroManuallyOpen(false);
      setMobilePanel("chat");
      showToast("推理包已登记；可在对话中开始分析，治理应用会复用同一份数据");
      await refreshSessionLists();
    } catch (error) {
      if (!requestIsCurrent()) return;
      setImportState({
        kind: "error",
        fileName: file.name,
        message: error instanceof Error ? error.message : "推理包登记失败。",
        issues: [],
      });
    }
  }, [activeSessionId, activateGraphVersionOverlay, bindGovernanceWorkspaceSnapshot, graphRepository, governanceOnlineClient, persistChatEntry, refreshSessionLists, replaceViewState, showToast]);

  const handleFiles = useCallback(
    async (inputFiles: FileList | File[]) => {
      const files = Array.from(inputFiles);
      if (!files.length) return;
      if (files.length === 1 && files[0].name.toLocaleLowerCase("en-US").endsWith(".zip")) {
        await handleGovernanceZip(files[0]);
        return;
      }
      if (files.length > 2) {
        setImportState({
          kind: "error",
          fileName: `${files.length} 个文件`,
          message: "一次最多提交一个标准图文件，或 nodes + edges 两个表。",
          issues: [],
        });
        return;
      }
      const requestEpoch = ++importRequestEpochRef.current;
      setPendingImport(null);
      setImportState({ kind: "parsing", fileName: files.map((file) => file.name).join(" + "), stage: "inspect" });
      try {
        const profiles = await Promise.all(files.map((file) => importAdapter.inspect(file)));
        if (requestEpoch !== importRequestEpochRef.current) return;
        const unsupported = profiles.find((profile) => !profile.supported);
        if (unsupported) {
          setImportState({
            kind: "error",
            fileName: unsupported.name,
            message: unsupported.issues[0]?.message ?? "文件无法读取。",
            issues: unsupported.issues,
          });
          return;
        }
        const baseGraphVersionId = graphVersion?.id;
        if (files.length === 2) {
          if (profiles.some((profile) => !["csv", "tsv"].includes(profile.format))) {
            setImportState({
              kind: "error",
              fileName: files.map((file) => file.name).join(" + "),
              message: "nodes + edges 双表当前只接受 CSV / TSV；JSON、GraphML 与 GEXF 请单文件上传。",
              issues: [],
            });
            return;
          }
          const edgeCandidates = profiles
            .map((profile, index) => profile.suggestedMapping?.source && profile.suggestedMapping?.target ? index : -1)
            .filter((index) => index >= 0);
          const namedEdge = files.findIndex((file) => /(?:^|[_\-.])(edges?|links?|relations?)(?:[_\-.]|$)/iu.test(file.name));
          const initialEdgeIndex = edgeCandidates.length === 1 ? edgeCandidates[0] : namedEdge >= 0 ? namedEdge : 1;
          setImportState({
            kind: "roles",
            files: Object.freeze(files),
            profiles: Object.freeze(profiles),
            initialEdgeIndex,
            ...(baseGraphVersionId ? { baseGraphVersionId } : {}),
          });
          return;
        }
        await preparePendingImport(files, profiles, undefined, baseGraphVersionId);
      } catch (error) {
        if (requestEpoch !== importRequestEpochRef.current) return;
        setImportState({
          kind: "error",
          fileName: files.map((file) => file.name).join(" + "),
          message: error instanceof Error ? error.message : "文件准备失败。",
          issues: [],
        });
      }
    },
    [graphVersion, handleGovernanceZip, importAdapter, preparePendingImport],
  );

  const submitImportDraft = useCallback(async (
    pending: PendingImportDraft,
    description: string,
    parentVersionId?: string,
    reconstructionReason: GraphVersionProvenance["reconstructionReason"] = "construction_revision",
  ): Promise<{ readonly run: ImportRun; readonly source: "llm"; readonly warnings: readonly string[] }> => {
    const requestEpoch = ++importRequestEpochRef.current;
    const expectedBaseGraphVersionId = pending.baseGraphVersionId ?? null;
    setImportState({
      kind: "parsing",
      fileName: pending.files.map((file) => file.name).join(" + "),
      stage: "parse",
    });
    const normalized = await graphBuildIntentNormalizer.normalizeGraphBuildIntent({
      description: description.trim() || "请按文件字段建立规范关系图。",
      requestToken: pending.requestToken,
      ...(expectedBaseGraphVersionId ? { baseGraphVersionId: expectedBaseGraphVersionId } : {}),
      files: pending.artifacts.map((artifact, index) => ({
        artifactId: artifact.id,
        role: artifact.role,
        format: artifact.format,
        columns: pending.profiles[index]?.columns ?? [],
      })),
      allowedPolicies: {
        direction: ["file", "directed", "undirected"],
        duplicateEdges: ["preserve", "merge_sum", "reject"],
        selfLoops: ["preserve", "reject"],
        danglingEndpoints: ["derive_nodes", "reject"],
        timeFormats: ["none", "auto", "iso8601", "year", "unix_seconds", "unix_milliseconds"],
      },
    });
    if (requestEpoch !== importRequestEpochRef.current || normalized.requestToken !== pending.requestToken) {
      throw new Error("较早的构图请求已失效，未修改当前图版本。");
    }
    if (currentGraphVersionIdRef.current !== expectedBaseGraphVersionId) {
      throw new Error("当前图谱已变化；旧构图响应已安全丢弃。");
    }
    const run = await importAdapter.parseFiles(pending.files, normalized.spec, {
      buildSpec: normalized.spec,
      sourceArtifacts: pending.artifacts,
      ...(parentVersionId ? { parentVersionId } : {}),
      provenance: browserImportProvenance(
        parentVersionId ? reconstructionReason : undefined,
      ),
    });
    if (requestEpoch !== importRequestEpochRef.current || currentGraphVersionIdRef.current !== expectedBaseGraphVersionId) {
      throw new Error("当前图谱已变化；解析结果未提交。");
    }
    if (run.status === "needs_mapping") {
      setImportState({
        kind: "mapping",
        pending,
        issues: run.issues,
        spec: normalized.spec,
        source: normalized.source,
        normalizationWarnings: normalized.warnings,
        ...(parentVersionId ? { parentVersionId } : {}),
        ...(parentVersionId ? { reconstructionReason } : {}),
      });
    } else if (run.status === "failed" || !run.graphVersion) {
      setImportState({
        kind: "error",
        fileName: pending.files.map((file) => file.name).join(" + "),
        message: run.error ?? "构图规则或文件质量未通过验证。",
        issues: run.issues,
      });
    } else {
      setImportState({
        kind: "review",
        pending,
        spec: normalized.spec,
        run: run as ImportRun & { readonly graphVersion: GraphVersion },
        source: normalized.source,
        warnings: normalized.warnings,
        ...(parentVersionId ? { parentVersionId } : {}),
        ...(parentVersionId ? { reconstructionReason } : {}),
      });
    }
    return { run, source: normalized.source, warnings: normalized.warnings };
  }, [graphBuildIntentNormalizer, importAdapter]);

  const reparseImportMapping = useCallback(async (
    state: Extract<ImportViewState, { kind: "mapping" }>,
    value: GraphImportMappingValue,
  ) => {
    const requestEpoch = ++importRequestEpochRef.current;
    const expectedBaseGraphVersionId = state.pending.baseGraphVersionId ?? null;
    const spec: GraphBuildSpec = Object.freeze({
      ...state.spec,
      ...(value.nodeMapping ? { nodeMapping: Object.freeze({ ...value.nodeMapping }) } : {}),
      edgeMapping: Object.freeze({ ...value.edgeMapping }),
      timeFormat: value.timeFormat,
    });
    setImportState({
      kind: "parsing",
      fileName: state.pending.files.map((file) => file.name).join(" + "),
      stage: "parse",
    });
    const run = await importAdapter.parseFiles(state.pending.files, spec, {
      buildSpec: spec,
      sourceArtifacts: state.pending.artifacts,
      ...(state.parentVersionId ? { parentVersionId: state.parentVersionId } : {}),
      provenance: browserImportProvenance(
        state.parentVersionId
          ? state.reconstructionReason ?? "construction_revision"
          : undefined,
      ),
    });
    if (
      requestEpoch !== importRequestEpochRef.current ||
      currentGraphVersionIdRef.current !== expectedBaseGraphVersionId
    ) {
      throw new Error("当前图谱已变化；字段映射结果未提交。");
    }
    if (run.status !== "ready" || !run.graphVersion) {
      setImportState({
        ...state,
        spec,
        issues: run.issues,
      });
      showToast(run.error ?? "字段映射仍有阻断性问题，请根据质量报告修改。");
      return;
    }
    setImportState({
      kind: "review",
      pending: state.pending,
      spec,
      run: run as ImportRun & { readonly graphVersion: GraphVersion },
      source: state.source,
      warnings: state.normalizationWarnings,
      ...(state.parentVersionId ? { parentVersionId: state.parentVersionId } : {}),
      ...(state.reconstructionReason ? { reconstructionReason: state.reconstructionReason } : {}),
    });
  }, [importAdapter, showToast]);

  const rebuildPendingFromVersion = useCallback(async (version: GraphVersion): Promise<PendingImportDraft | null> => {
    const sourceArtifactIds = [...new Set([
      ...(version.sourceArtifactIds ?? []),
      ...(version.buildSpec?.sourceArtifactIds ?? []),
    ])];
    if (!sourceArtifactIds.length) return null;
    const artifacts = (await Promise.all(
      sourceArtifactIds.map((artifactId) => graphRepository.getSourceArtifact(artifactId)),
    )).filter((artifact): artifact is SourceArtifact => Boolean(artifact));
    if (artifacts.length !== sourceArtifactIds.length) return null;
    const files = artifacts.map((artifact) => new File([artifact.blob], artifact.name, { type: artifact.mimeType }));
    const verifiedArtifacts = await Promise.all(
      files.map((file, index) => createSourceArtifact(file, artifacts[index]?.role ?? "single")),
    );
    if (verifiedArtifacts.some((candidate, index) => candidate.sha256 !== artifacts[index]?.sha256)) {
      throw new Error("SOURCE_ARTIFACT_HASH_MISMATCH：源文件内容与保存的 SHA-256 不一致，已拒绝重建。");
    }
    const profiles = await Promise.all(files.map((file) => importAdapter.inspect(file)));
    if (profiles.some((profile) => !profile.supported)) return null;
    return {
      files: Object.freeze(files),
      profiles: Object.freeze(profiles),
      artifacts: Object.freeze(artifacts),
      requestToken: makeId("import-revision"),
      baseGraphVersionId: version.id,
    };
  }, [graphRepository, importAdapter]);

  const applyNormalizedView = useCallback(
    (command: ViewCommand, chosenNodeIds: readonly string[] = []): readonly string[] => {
      if (!graphVersion) return ["当前没有可应用视图命令的图谱。"];
      const currentView = viewState.graphVersionId === graphVersion.id
        ? viewState
        : createDefaultGraphViewState(graphVersion.id);
      const outcome = applyViewCommand(graphVersion, currentView, command, chosenNodeIds);
      setAnalysisSceneOverride(null);
      replaceViewState(outcome.nextState);
      if (outcome.requestedOverlay) {
        graphVersionOverlayController.setExplicit(
          graphVersion.id,
          buildRequestedOverlay(graphVersion, outcome.requestedOverlay),
        );
      }
      const ambiguous = outcome.targetResolutions.filter(
        (resolution): resolution is Extract<TargetResolution, { status: "ambiguous" }> => resolution.status === "ambiguous",
      );
      if (ambiguous.length && chosenNodeIds.length === 0) {
        setPendingTargetResolution({ command, resolutions: ambiguous });
      } else {
        setPendingTargetResolution(null);
      }
      setMobilePanel("graph");
      void graphRepository.saveEvent(createSemanticEvent("intent_applied", {
        graphVersionId: graphVersion.id,
        sessionId: activeSessionId,
        payload: {
          mode: outcome.nextState.mode,
          depth: outcome.nextState.depth,
          overlay: outcome.requestedOverlay ?? "none",
        },
      }));
      return outcome.warnings;
    },
    [activeSessionId, graphRepository, graphVersion, graphVersionOverlayController, replaceViewState, viewState],
  );

  const executeNormalizedAnalysis = useCallback(async (
    intent: NormalizedIntent,
    chosenNodeIds: readonly string[] = [],
  ): Promise<
    | { readonly kind: "ambiguous"; readonly warnings: readonly string[] }
    | { readonly kind: "completed"; readonly run: AnalysisRun; readonly warnings: readonly string[] }
  > => {
    const graph = graphVersion;
    if (!graph) throw new Error("当前没有可分析的图谱。");
    const requestEpoch = ++intentRequestEpochRef.current;
    const baseGraphVersionId = graph.id;
    const currentView = viewState.graphVersionId === graph.id
      ? viewState
      : createDefaultGraphViewState(graph.id);
    const prepared = prepareAnalysisFilters(graph, intent);
    const outcome = prepared.command
      ? applyViewCommand(graph, currentView, prepared.command, chosenNodeIds)
      : {
          nextState: currentView,
          targetResolutions: [] as readonly TargetResolution[],
          warnings: [] as readonly string[],
          requestedOverlay: undefined,
        };
    const ambiguous = outcome.targetResolutions.filter(
      (resolution): resolution is Extract<TargetResolution, { status: "ambiguous" }> => resolution.status === "ambiguous",
    );
    if (ambiguous.length && chosenNodeIds.length === 0 && prepared.command) {
      setPendingTargetResolution({ command: prepared.command, resolutions: ambiguous, intent });
      return { kind: "ambiguous", warnings: [...outcome.warnings, ...prepared.warnings] };
    }
    setPendingTargetResolution(null);
    const executableViewState: GraphViewState = Object.freeze({
      ...outcome.nextState,
      filters: applyPreparedAnalysisFilters(outcome.nextState.filters, prepared),
    });
    replaceViewState(executableViewState);
    setMobilePanel("graph");
    const semanticSlice = buildSemanticGraphSlice(graph, { viewState: executableViewState });
    const scopedGraph = createScopedGraphSlice(
      graph.id,
      semanticSlice.slice.nodes,
      semanticSlice.slice.edges,
      semanticSlice.filters,
      false,
    );
    const scopedScene = buildGraphScene(graph, { viewState: executableViewState });
    setAnalysisSceneOverride({ viewKey: analysisViewKey(executableViewState), scene: scopedScene });
    let run = await analysisExecutor.createAnalysis({
      graphVersionId: graph.id,
      graphVersion: graph,
      intent,
      scopedGraph,
    });
    if (run.status === "queued" || run.status === "running") run = await analysisExecutor.getAnalysis(run.id);
    if (requestEpoch !== intentRequestEpochRef.current || currentGraphVersionIdRef.current !== baseGraphVersionId) {
      throw new Error("当前图谱已变化；旧分析结果已安全丢弃。");
    }
    const unavailable = run.engine === "unavailable" || run.result?.kind === "unavailable";
    if (!unavailable) {
      const overlayKind = outcome.requestedOverlay;
      const scopedVersion: GraphVersion = Object.freeze({
        ...graph,
        nodes: scopedGraph.slice.nodes,
        edges: scopedGraph.slice.edges,
        summary: run.result?.kind === "overview" ? run.result.summary : graph.summary,
      });
      const overlay = overlayKind
        ? buildRequestedOverlay(scopedVersion, overlayKind, run.id)
        : defaultOverlayForRun(scopedVersion, run);
      if (overlay) graphVersionOverlayController.setExplicit(
        graph.id,
        withScopeProvenance(overlay, scopedGraph.scope.scopeHash),
      );
    }
    await graphRepository.saveAnalysisRun(run);
    await graphRepository.saveEvent(createSemanticEvent("analysis_completed", {
      graphVersionId: graph.id,
      sessionId: activeSessionId,
      payload: {
        runId: run.id,
        task: intent.task,
        status: run.status,
        scopeHash: scopedGraph.scope.scopeHash,
        nodeCount: scopedGraph.scope.nodeCount,
        edgeCount: scopedGraph.scope.edgeCount,
      },
    }));
    return { kind: "completed", run, warnings: Object.freeze([...outcome.warnings, ...prepared.warnings]) };
  }, [activeSessionId, analysisExecutor, graphRepository, graphVersion, graphVersionOverlayController, replaceViewState, viewState]);

  const handleGraphViewSnapshot = useCallback(
    (snapshot: GraphPreviewViewSnapshot) => {
      if (activeWorkspace !== "chat") return;
      if (!graphVersion) return;
      if (snapshot.graphVersionId && snapshot.graphVersionId !== graphVersion.id) return;
      chatViewSnapshotRef.current = snapshot;
      if (snapshot.mode === "global") {
        cacheWorkspaceCamera("chat", graphVersion.id, "overview", snapshot.camera);
      }
      setViewState((current) => {
        const base = current.graphVersionId === graphVersion.id
          ? current
          : createDefaultGraphViewState(graphVersion.id);
        const nextFocusNodeIds = [...snapshot.focusNodeIds];
        const nextPathEndpointIds = [...snapshot.pathEndpointIds];
        const sameFocus = base.focusNodeIds.length === nextFocusNodeIds.length
          && base.focusNodeIds.every((nodeId, index) => nodeId === nextFocusNodeIds[index]);
        const samePathEndpoints = base.pathEndpointIds.length === nextPathEndpointIds.length
          && base.pathEndpointIds.every((nodeId, index) => nodeId === nextPathEndpointIds[index]);
        const samePinned = JSON.stringify(base.pinnedNodes) === JSON.stringify(snapshot.pinnedNodes);
        const cameraX = Number(snapshot.camera.position[0]);
        const cameraY = Number(snapshot.camera.position[1]);
        const sameCamera = Math.abs(base.camera.x - cameraX) < 0.01
          && Math.abs(base.camera.y - cameraY) < 0.01
          && Math.abs(base.camera.zoom - snapshot.camera.zoom) < 0.001;
        if (
          base.mode === snapshot.mode
          && base.depth === snapshot.depth
          && base.theme === snapshot.theme
          && base.layoutPreset === snapshot.layoutPreset
          && base.rendererPreference === snapshot.rendererPreference
          && sameFocus
          && samePathEndpoints
          && samePinned
          && sameCamera
        ) return current;
        return {
          ...base,
          mode: snapshot.mode,
          depth: snapshot.depth,
          theme: snapshot.theme,
          layoutPreset: snapshot.layoutPreset,
          rendererPreference: snapshot.rendererPreference,
          focusNodeIds: sameFocus ? base.focusNodeIds : nextFocusNodeIds,
          pathEndpointIds: samePathEndpoints ? base.pathEndpointIds : nextPathEndpointIds,
          camera: { x: cameraX, y: cameraY, zoom: snapshot.camera.zoom },
          pinnedNodes: samePinned ? base.pinnedNodes : { ...snapshot.pinnedNodes },
        };
      });
    },
    [activeWorkspace, cacheWorkspaceCamera, graphVersion],
  );

  const handleGovernanceViewSnapshot = useCallback((snapshot: GraphPreviewViewSnapshot) => {
    if (activeWorkspace === "governance") {
      governanceViewSnapshotRef.current = snapshot;
      cacheWorkspaceCamera(
        "governance",
        snapshot.graphVersionId ?? "governance-empty",
        governanceGraphPresentation?.lens ?? "risk",
        snapshot.camera,
      );
      return;
    }
    if (activeWorkspace === "adaptation") {
      adaptationViewSnapshotRef.current = snapshot;
      const current = adaptationLanePresentationRef.current;
      if (!current.lanes[current.activeLane].focus) {
        cacheWorkspaceCamera(
          "adaptation",
          snapshot.graphVersionId ?? "adaptation-empty",
          adaptationCameraLens(current.activeLane),
          snapshot.camera,
        );
      }
    }
  }, [activeWorkspace, cacheWorkspaceCamera, governanceGraphPresentation?.lens]);

  const patchGraphView = useCallback(
    (patch: Partial<GraphViewState>) => {
      if (!graphVersion) return;
      if (patch.depth !== undefined || patch.filters !== undefined || patch.mode !== undefined
        || patch.focusNodeIds !== undefined || patch.pathEndpointIds !== undefined) {
        setAnalysisSceneOverride(null);
      }
      setViewState((current) => ({
        ...(current.graphVersionId === graphVersion.id
          ? current
          : createDefaultGraphViewState(graphVersion.id)),
        ...patch,
        graphVersionId: graphVersion.id,
      }));
    },
    [graphVersion],
  );

  const handleGraphFiltersChange = useCallback(
    (filters: GraphFilters) => patchGraphView({ filters }),
    [patchGraphView],
  );

  const handleGraphViewModeChange = useCallback((mode: GraphViewState["mode"]) => {
    setAnalysisSceneOverride(null);
    dispatchGraphView({ type: "activate_mode", mode });
    if (mode === "global") {
      setPendingTargetResolution(null);
      if (graphVersion) graphVersionOverlayController.clearExplicit(graphVersion.id, "path");
    }
  }, [dispatchGraphView, graphVersion, graphVersionOverlayController]);

  const handleGraphNodeSelect = useCallback((node: GraphNode | null) => {
    dispatchGraphView({ type: "select_node", nodeId: node?.id ?? null });
  }, [dispatchGraphView]);

  const handleFocusNodeIdsChange = useCallback((nodeIds: readonly string[]) => {
    setAnalysisSceneOverride(null);
    dispatchGraphView({ type: "set_focus", nodeIds });
  }, [dispatchGraphView]);

  const restoreChatLocateOverview = useCallback(() => {
    const overview = chatLocateOverviewRef.current;
    if (!overview) return;
    chatLocateOverviewRef.current = null;
    setChatLocateOverview(null);
    setAnalysisSceneOverride(null);
    dispatchGraphView({ type: "activate_mode", mode: "global" });
    workspaceCameraRestoreTokenRef.current += 1;
    setWorkspaceCameraRestore({
      ...overview.camera,
      workspace: "chat",
      x: Number(overview.camera.position[0]),
      y: Number(overview.camera.position[1]),
      token: workspaceCameraRestoreTokenRef.current,
    });
  }, [dispatchGraphView]);

  const handleLocateGovernanceCandidates = useCallback((
    entry: Extract<ChatEntry, { role: "assistant" }>,
  ) => {
    const snapshot = governanceWorkspaceSnapshotRef.current;
    locateGovernanceCandidates({
      messageRunId: entry.governanceRunId,
      currentRunId: snapshot?.run?.runId,
      runGraphVersionHash: snapshot?.run?.graphVersionHash,
      currentGraphVersionHash: graphVersion?.datasetArtifact?.canonicalGraphHash,
      previewNodes: snapshot?.preview.nodes ?? [],
      graphNodeIds: graphVersion?.preview.nodes.map((node) => node.id) ?? [],
    }, {
      saveOverview: () => {
        const overview = chatViewSnapshotRef.current;
        if (!overview || overview.graphVersionId !== graphVersion?.id) return;
        chatLocateOverviewRef.current = overview;
        setChatLocateOverview(overview);
      },
      applyGraphAction: (action) => {
        setAnalysisSceneOverride(null);
        dispatchGraphView(action);
      },
      expandGraph: () => setWorkspaceLayout((current) => ({ ...current, rightCollapsed: false })),
      switchMobilePanel: setMobilePanel,
      notify: showToast,
    });
  }, [dispatchGraphView, graphVersion, showToast]);

  const handlePathEndpointIdsChange = useCallback((nodeIds: readonly string[]) => {
    setAnalysisSceneOverride(null);
    dispatchGraphView({ type: "set_path_endpoints", nodeIds });
  }, [dispatchGraphView]);

  const chatGovernanceContext = useMemo<GovernanceSkillsContext | null>(() => {
    const snapshot = governanceWorkspace.snapshot;
    if (!snapshot || snapshot.sessionId !== activeSessionId || governanceServiceState.state !== "ready") return null;
    const selectedContext = governanceGraphPresentation?.skillsContext;
    const selectionMatchesSnapshot = selectedContext?.graph.artifactId === snapshot.artifact.artifactId
      && selectedContext.graph.datasetContentHash === snapshot.artifact.datasetContentHash
      && selectedContext.graph.graphVersionHash === snapshot.artifact.graphVersionHash
      && selectedContext.model.modelVersionId === governanceServiceState.modelVersionId
      && selectedContext.model.modelStateHash === governanceServiceState.modelStateHash;
    return Object.freeze({
      graph: Object.freeze({
        artifactId: snapshot.artifact.artifactId,
        datasetContentHash: snapshot.artifact.datasetContentHash,
        graphVersionHash: snapshot.artifact.graphVersionHash,
      }),
      model: Object.freeze({
        modelVersionId: governanceServiceState.modelVersionId,
        modelStateHash: governanceServiceState.modelStateHash,
      }),
      ...(snapshot.run?.runId ? { runId: snapshot.run.runId } : {}),
      ...(snapshot.activeCaseId ? { caseId: snapshot.activeCaseId } : {}),
      ...(selectionMatchesSnapshot && selectedContext?.selectedNodeIds
        ? { selectedNodeIds: selectedContext.selectedNodeIds }
        : {}),
      ...(selectionMatchesSnapshot && selectedContext?.selectedTarget
        ? { selectedTarget: selectedContext.selectedTarget }
        : {}),
    });
  }, [activeSessionId, governanceGraphPresentation?.skillsContext, governanceServiceState, governanceWorkspace.snapshot]);

  const sendMessage = useCallback(
    async (textOverride?: string | ResearchPrompt) => {
      const submittedSessionId = activeSessionId;
      const promptOverride = typeof textOverride === "object" ? textOverride : undefined;
      const text = (promptOverride?.text ?? (typeof textOverride === "string" ? textOverride : draft)).trim();
      const researchPrompt = promptOverride ?? researchPromptForText(text);
      const importDraft = textOverride === undefined ? pendingImport : null;
      if ((!text && !importDraft) || isSending) return;
      const submittedText = text || "请按文件字段建立规范关系图。";
      const userEntry: ChatEntry = {
        id: makeId("user-chat"),
        role: "user",
        text: submittedText,
        timestamp: timeNow(),
        ...(importDraft
          ? {
              file: { name: importDraft.files[0].name, size: importDraft.files[0].size },
              files: importDraft.files.map((file) => ({ name: file.name, size: file.size })),
            }
          : {}),
      };
      const submittedImportDraft = importDraft
        ? Object.freeze({ ...importDraft, sourceMessageId: userEntry.id })
        : null;
      const assistantId = makeId("assistant-request");
      const pendingActivityKind: AssistantActivityKind = importDraft
        ? "graph_import"
        : chatGovernanceContext
          ? "governance"
          : "graph_analysis";
      const pendingEntry: ChatEntry = {
        id: assistantId,
        role: "assistant",
        text: importDraft
          ? "正在检查数据结构、关系方向与质量边界，完成后将生成可确认的图谱草稿…"
          : chatGovernanceContext
            ? "正在核对当前推理包与治理目标，准备风险候选、群组和关系的分析路径…"
            : "正在核对当前图谱与研究目标，并选择合适的结构分析方法…",
        timestamp: timeNow(),
        state: "working",
        activity: { kind: pendingActivityKind, state: "working" },
      };
      const replacePending = (entry: Extract<ChatEntry, { role: "assistant" }>) => {
        const completed = Object.freeze({
          ...entry,
          activity: Object.freeze({ kind: pendingActivityKind, state: "completed" as const }),
        });
        setMessages((current) => current.map((item) => item.id === assistantId ? completed : item));
      };
      setDraft("");
      setIsSending(true);
      setHeroManuallyOpen(false);
      setMessages((current) => [...current, userEntry, pendingEntry]);
      const startedAt = performance.now();
      try {
        // The source message must exist before a later atomic import transaction
        // can bind its SourceArtifacts to that exact submission.
        await Promise.all([persistChatEntry(userEntry), persistChatEntry(pendingEntry)]);
        if (submittedImportDraft) {
          const submitted = await submitImportDraft(submittedImportDraft, submittedText);
          const ready = submitted.run.status === "ready" && Boolean(submitted.run.graphVersion);
          const mapping = submitted.run.status === "needs_mapping";
          const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
            id: assistantId,
            role: "assistant",
            text: ready
              ? `已生成可验证的构图草稿。请检查方向、字段、关系策略与质量报告；确认后将创建新的图谱版本，再继续提出结构或治理分析目标。${submitted.warnings.join(" ")}`
              : mapping
                ? "文件结构已识别，但起点与终点字段仍有歧义。请先在映射卡确认关系两端；确认前不会生成图谱版本，也不会改变当前图谱。"
                : submitted.run.error ?? "构图规则或数据质量未通过校验，未修改当前图版本。",
            timestamp: timeNow(),
            state: ready ? "warning" : mapping ? "warning" : "error",
          };
          replacePending(finalEntry);
          await persistChatEntry(finalEntry);
          return;
        }

        if (chatGovernanceContext) {
          const requestEpoch = ++governanceChatRequestEpochRef.current;
          governanceChatRequestAbortRef.current?.abort();
          const controller = new AbortController();
          governanceChatRequestAbortRef.current = controller;
          if (isGovernanceAnalysisRequest(submittedText)) {
            const prepared = await governanceSkillsClient.executeSkill(
              chatGovernanceContext,
              "run_governance_analysis",
              { protocol: "global", topK: 100 },
              controller.signal,
            );
            if (controller.signal.aborted
              || requestEpoch !== governanceChatRequestEpochRef.current
              || activeSessionIdRef.current !== submittedSessionId) return;
            const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
              id: assistantId,
              role: "assistant",
              text: prepared.status === "confirmation_required"
                ? "治理分析计划已准备完成。请确认后运行 Global 图基础模型；模型结果仅用于安排人工复核顺序。"
                : "治理分析请求已完成。",
              timestamp: timeNow(),
              state: prepared.status === "confirmation_required" ? "warning" : "success",
              ...(prepared.confirmation ? { confirmation: prepared.confirmation } : {}),
            };
            replacePending(finalEntry);
            await persistChatEntry(finalEntry);
            return;
          }
          const promptRequest = researchPrompt
            ? researchPromptSkillRequest(chatGovernanceContext, researchPrompt)
            : null;
          const dispatched = await governanceSkillsClient.executeAssistant(
            promptRequest?.context ?? chatGovernanceContext,
            promptRequest?.skill ?? "answer_governance_question",
            submittedText,
            controller.signal,
          );
          if (controller.signal.aborted
            || requestEpoch !== governanceChatRequestEpochRef.current
            || activeSessionIdRef.current !== submittedSessionId) return;
          const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
            id: assistantId,
            role: "assistant",
            text: dispatched.answer,
            timestamp: timeNow(),
            state: "success",
          };
          replacePending(finalEntry);
          await persistChatEntry(finalEntry);
          return;
        }

        if (graphVersion && isGraphConstructionRevision(submittedText)) {
          const revisionDraft = await rebuildPendingFromVersion(graphVersion);
          if (revisionDraft) {
            const submitted = await submitImportDraft(
              Object.freeze({ ...revisionDraft, sourceMessageId: userEntry.id }),
              submittedText,
              graphVersion.id,
            );
            const ready = submitted.run.status === "ready" && Boolean(submitted.run.graphVersion);
            const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
              id: assistantId,
              role: "assistant",
              text: ready
                ? "已从同一份原始数据生成构图修订草稿；确认后才会创建子图谱版本，旧版本始终保持不变。"
                : submitted.run.status === "needs_mapping"
                  ? "构图修订仍需确认字段映射；确认前旧版本保持不变。"
                  : submitted.run.error ?? "构图修订未通过验证，旧版本保持不变。",
              timestamp: timeNow(),
              state: ready ? "warning" : submitted.run.status === "needs_mapping" ? "warning" : "error",
            };
            replacePending(finalEntry);
            await persistChatEntry(finalEntry);
            return;
          }
        }

        const intentEpoch = ++intentRequestEpochRef.current;
        const baseGraphVersionId = graphVersion?.id ?? null;
        const normalized = await intentNormalizer.normalizeIntent({
          text: submittedText,
          ...(graphVersion ? { graphContext: buildGraphContextSummary(graphVersion) } : {}),
        });
        if (intentEpoch !== intentRequestEpochRef.current || currentGraphVersionIdRef.current !== baseGraphVersionId) {
          throw new Error("当前图谱已变化；旧意图响应已安全丢弃。");
        }
        setLlmDiagnostic({
          state: "success",
          latencyMs: Math.round(performance.now() - startedAt),
          schemaVersion: normalized.meta.schemaVersion,
          requestId: normalized.meta.requestId,
          ...(normalized.meta.model ? { model: normalized.meta.model } : {}),
          source: normalized.meta.source,
          ...(normalized.kind === "analysis_request" ? { task: normalized.task } : {}),
          warnings: normalized.meta.warnings,
        });
        setIntentServiceStatus({
          state: "llm",
          label: "LLM 本次调用成功",
          ...(normalized.meta.model ? { model: normalized.meta.model } : {}),
        });
        if (normalized.kind === "chat") {
          const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
            id: assistantId,
            role: "assistant",
            text: normalized.reply,
            timestamp: timeNow(),
            state: "success",
            intentMeta: normalized.meta,
          };
          replacePending(finalEntry);
          await persistChatEntry(finalEntry, normalized);
          return;
        }
        const intent = normalized;
        if (!graphVersion) {
          const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
            id: assistantId,
            role: "assistant",
            text: `已识别你的目标为“${taskNames[intent.task]}”。当前还没有可分析的关系图，请先上传数据；系统将先核对字段与关系方向，再运行分析并把重点节点或群组同步到右侧图谱。你的文字不会被当作图事实。`,
            timestamp: timeNow(),
            state: "warning",
            intent,
            intentMeta: intent.meta,
          };
          replacePending(finalEntry);
          await persistChatEntry(finalEntry, intent);
          return;
        }
        const execution = await executeNormalizedAnalysis(intent);
        if (execution.kind === "ambiguous") {
          const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
            id: assistantId,
            role: "assistant",
            text: "分析目标存在同名或相似节点。请先在本地候选卡确认；确认前不会运行算法或提交部分筛选。",
            timestamp: timeNow(),
            state: "warning",
            intent,
            intentMeta: intent.meta,
          };
          replacePending(finalEntry);
          await persistChatEntry(finalEntry, intent);
          return;
        }
        const { run, warnings: viewWarnings } = execution;
        const unavailable = run.engine === "unavailable" || run.result?.kind === "unavailable";
        const unavailableReason = describeUnavailableAnalysis(
          run.result?.kind === "unavailable" ? run.result : undefined,
          coreService,
        );
        const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
          id: assistantId,
          role: "assistant",
          text: unavailable
            ? `已识别分析对象与目标，但当前无法执行该方法。${unavailableReason} 当前图谱与已有结果保持不变；可调整目标后重试。`
            : viewWarnings.length
              ? `已完成当前图谱的结构分析，并应用可确定的视图操作。${viewWarnings.join(" ")} 结果展示在下方并同步到右侧图谱；可继续定位重点节点、调整范围或追问。`
              : intent.view
                ? "已完成当前图谱的结构分析，并将聚焦、筛选或高亮要求应用到右侧交互图。你可以检查重点对象、调整范围或继续追问。"
                : "已根据当前图谱完成结构分析，结果完全来自已登记的关系事实。请查看下方结论，并在右侧图谱定位重点对象或继续提出问题。",
          timestamp: timeNow(),
          state: unavailable ? "warning" : "success",
          intent,
          intentMeta: intent.meta,
          run,
        };
        replacePending(finalEntry);
        await persistChatEntry(finalEntry, intent);
      } catch (error) {
        if (activeSessionIdRef.current !== submittedSessionId
          || error instanceof DOMException && error.name === "AbortError") return;
        const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
          id: assistantId,
          role: "assistant",
          text: `${error instanceof Error ? error.message : "分析请求失败，请重试。"} 当前图谱与已有结果未被修改，可修正目标后重新发送。`,
          timestamp: timeNow(),
          state: "error",
          retryText: submittedText,
        };
        replacePending(finalEntry);
        await persistChatEntry(finalEntry);
        setLlmDiagnostic({
          state: "error",
          latencyMs: Math.round(performance.now() - startedAt),
          message: finalEntry.text,
        });
      } finally {
        if (activeSessionIdRef.current === submittedSessionId) {
          setIsSending(false);
          const currentSession = await graphRepository.getSession(activeSessionId);
          if (currentSession) {
            const title = currentSession.title.startsWith("新建研究会话")
              ? submittedText.slice(0, 28)
              : currentSession.title;
            await graphRepository.saveSession(createResearchSession(title, {
              id: currentSession.id,
              ...(currentSession.graphVersionId ? { graphVersionId: currentSession.graphVersionId } : {}),
            }));
            await refreshSessionLists();
          }
        }
      }
    },
    [
      activeSessionId,
      chatGovernanceContext,
      draft,
      executeNormalizedAnalysis,
      graphRepository,
      graphVersion,
      coreService,
      pendingImport,
      rebuildPendingFromVersion,
      intentNormalizer,
      isSending,
      governanceSkillsClient,
      persistChatEntry,
      refreshSessionLists,
      submitImportDraft,
    ],
  );

  const confirmChatAction = useCallback(async (entry: Extract<ChatEntry, { role: "assistant" }>) => {
    if (!entry.confirmation || !chatGovernanceContext || !governanceWorkspaceSnapshotRef.current) return;
    if (confirmingChatMessageIdsRef.current.has(entry.id)) return;
    confirmingChatMessageIdsRef.current.add(entry.id);
    const submittedSessionId = activeSessionId;
    const actionEpoch = ++governanceChatRequestEpochRef.current;
    governanceChatRequestAbortRef.current?.abort();
    const controller = new AbortController();
    governanceChatRequestAbortRef.current = controller;
    const ensureCurrent = () => {
      if (controller.signal.aborted
        || actionEpoch !== governanceChatRequestEpochRef.current
        || activeSessionIdRef.current !== submittedSessionId) {
        throw new DOMException("Stale SocialGraph-FM Governance chat action", "AbortError");
      }
    };
    setIsSending(true);
    let latestRun: GovernanceOnlineRun | null = null;
    let backendRunSucceeded = false;
    let analysisArtifactsReady = false;
    try {
      const confirmed = await governanceSkillsClient.confirmSkill(entry.confirmation.token, controller.signal);
      ensureCurrent();
      if (confirmed.action !== "run_governance_analysis") {
        setMessages((current) => current.map((item) => item.id === entry.id && item.role === "assistant"
          ? { ...item, confirmation: undefined, text: `${item.text}\n\n> 操作已确认并写入本地审计记录。` }
          : item));
        showToast(confirmed.action === "submit_review" ? "人工复核已追加" : "研判草稿已保存");
        return;
      }
      let status = confirmed.result;
      latestRun = status;
      const initialStatus = presentGovernanceRunProgress(status);
      let presentedProgress = `${initialStatus.stage}:${initialStatus.progress}`;
      const runningPlanningEntry = updateGovernancePlanningProgress([entry], entry.id, initialStatus)[0];
      setMessages((current) => updateGovernancePlanningProgress(current, entry.id, initialStatus));
      await persistChatEntry(runningPlanningEntry);
      ensureCurrent();
      const baseSnapshot = governanceWorkspaceSnapshotRef.current;
      if (status.artifactId !== baseSnapshot.artifact.artifactId
        || status.datasetContentHash !== baseSnapshot.artifact.datasetContentHash
        || status.graphVersionHash !== baseSnapshot.artifact.graphVersionHash) {
        throw new Error("确认运行与当前推理包身份不一致。");
      }
      await bindGovernanceWorkspaceSnapshot(Object.freeze({ ...baseSnapshot, run: status, updatedAt: new Date().toISOString() }));
      ensureCurrent();
      for (let attempt = 0; (status.status === "queued" || status.status === "running") && attempt < 900; attempt += 1) {
        await new Promise((resolve) => window.setTimeout(resolve, 350));
        ensureCurrent();
        status = await governanceOnlineClient.run(status.runId, controller.signal);
        latestRun = status;
        const visibleStatus = presentGovernanceRunProgress(status);
        const nextProgress = `${visibleStatus.stage}:${visibleStatus.progress}`;
        if (nextProgress !== presentedProgress) {
          presentedProgress = nextProgress;
          const progressStatus = visibleStatus;
          setMessages((current) => updateGovernancePlanningProgress(current, entry.id, progressStatus));
        }
      }
      ensureCurrent();
      if (status.status !== "succeeded") throw new Error(status.errorCode ?? `分析未完成：${status.status}`);
      backendRunSucceeded = true;
      const reportingStatus = { stage: "reporting" as const, progress: 95 };
      const reportingPlanningEntry = updateGovernancePlanningProgress([entry], entry.id, reportingStatus)[0];
      setMessages((current) => updateGovernancePlanningProgress(current, entry.id, reportingStatus));
      await persistChatEntry(reportingPlanningEntry);
      ensureCurrent();
      const [result, preview] = await Promise.all([
        governanceOnlineClient.result(status.runId, controller.signal),
        governanceOnlineClient.runPreview(status.runId, controller.signal, { preset: "overview", nodeBudget: 120, edgeBudget: 240 }),
      ]);
      ensureCurrent();
      const currentSnapshot = governanceWorkspaceSnapshotRef.current;
      if (!currentSnapshot
        || currentSnapshot.sessionId !== baseSnapshot.sessionId
        || !sameGovernanceArtifactIdentity(currentSnapshot.artifact, baseSnapshot.artifact)) {
        throw new DOMException("Stale SocialGraph-FM Governance package", "AbortError");
      }
      await bindGovernanceWorkspaceSnapshot(Object.freeze({ ...currentSnapshot, run: status, result, preview, updatedAt: new Date().toISOString() }));
      analysisArtifactsReady = true;
      const acceptedGraph = currentGraphVersionRef.current;
      if (acceptedGraph) graphVersionOverlayController.acceptGlobalResult(acceptedGraph, {
        protocol: "global",
        status: "succeeded",
        graphVersionHash: result.graphVersionHash,
        runId: status.runId,
        resultHash: result.resultHash,
      });
      const reportContext: GovernanceSkillsContext = Object.freeze({ ...chatGovernanceContext, runId: status.runId });
      const report = await governanceSkillsClient.executeAssistant(
        reportContext,
        "generate_global_situation_report",
        "请生成本次治理分析摘要，列出高关注账号、风险群组、事实关系、潜在线索和人工复核建议。",
        controller.signal,
      );
      const reportText = ensureHumanReviewGuidance(report.answer);
      ensureCurrent();
      const reportEntry: Extract<ChatEntry, { role: "assistant" }> = {
        id: makeId("assistant-governance-report"), role: "assistant", text: reportText,
        timestamp: timeNow(), state: "success", governanceRunId: status.runId,
      };
      const completedPlanningEntry = completeConfirmedPlanningMessage([entry], entry.id, reportEntry)
        .find((item) => item.id === entry.id)!;
      await persistChatEntry(completedPlanningEntry);
      if (!completedCoreReportsRef.current.has(status.runId)) {
        completedCoreReportsRef.current.add(status.runId);
        setMessages((current) => completeConfirmedPlanningMessage(current, entry.id, reportEntry));
        await persistChatEntry(reportEntry);
      }
      showToast("治理分析完成，治理报告已生成");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      if (latestRun) {
        const failureEntry: Extract<ChatEntry, { role: "assistant" }> = {
          ...entry,
          confirmation: undefined,
          state: backendRunSucceeded ? "warning" : "error",
          text: analysisArtifactsReady
            ? "当前治理图谱的结果与预览已经就绪，但研判结论未能完成整理。风险候选、群组和重点关系仍可在治理应用中复核；图事实未被改写。"
            : backendRunSucceeded
              ? "模型推理已经完成，但结果与图谱预览尚未完整同步。当前进度保留在结论整理阶段，请重试以恢复可复核结果；图事实未被改写。"
            : `${error instanceof Error ? error.message : "治理分析未完成。"} 当前图事实与既有结果未被改写，请在治理应用核对服务状态后重新分析。`,
          activity: undefined,
          governanceProgress: backendRunSucceeded
            ? { stage: "reporting", progress: 95 }
            : presentGovernanceRunProgress(latestRun),
        };
        setMessages((current) => current.map((item) => item.id === entry.id ? failureEntry : item));
        await persistChatEntry(failureEntry);
      }
      showToast(error instanceof Error ? error.message : "确认操作未完成");
    } finally {
      confirmingChatMessageIdsRef.current.delete(entry.id);
      if (actionEpoch === governanceChatRequestEpochRef.current
        && activeSessionIdRef.current === submittedSessionId) setIsSending(false);
    }
  }, [activeSessionId, bindGovernanceWorkspaceSnapshot, chatGovernanceContext, graphVersionOverlayController, governanceOnlineClient, governanceSkillsClient, persistChatEntry, showToast]);

  const handleWelcomePrompt = useCallback((prompt: ResearchPrompt) => {
    if (welcomePromptAction(Boolean(graphVersion)) === "send") {
      void sendMessage(prompt);
      return;
    }
    setDraft(prompt.text);
    showToast("研究目标已填入；请先上传关系数据，再发送分析");
    window.requestAnimationFrame(() => composerInputRef.current?.focus());
  }, [graphVersion, sendMessage, showToast]);

  const beginSessionTransition = (nextSessionId: string) => {
    importRequestEpochRef.current += 1;
    intentRequestEpochRef.current += 1;
    governanceChatRequestEpochRef.current += 1;
    governanceChatRequestAbortRef.current?.abort();
    governanceChatRequestAbortRef.current = null;
    activeSessionIdRef.current = nextSessionId;
    setIsSending(false);
    setMessages(invalidateChatConfirmations);
  };

  const startNewSession = async () => {
    const newSession = createResearchSession(`新建研究会话 · ${timeNow()}`);
    beginSessionTransition(newSession.id);
    await graphRepository.saveSession(newSession);
    await refreshSessionLists();
    setActiveSessionId(newSession.id);
    window.localStorage.setItem("socialgraph-fm-active-session", newSession.id);
    setSidebarOpen(false);
    setMobilePanel("chat");
    showToast("已创建空白研究会话");
  };

  const restoreGraphVersion = async (version: GraphVersion) => {
    importRequestEpochRef.current += 1;
    intentRequestEpochRef.current += 1;
    currentGraphVersionIdRef.current = version.id;
    analysisExecutor.registerGraphVersion(version);
    const storedView = await graphRepository.getViewState(version.id);
    const nextView = normalizeGraphViewState(version.id, storedView);
    setGraphVersion(version);
    setAnalysisSceneOverride(null);
    replaceViewState(nextView);
    activateGraphVersionOverlay(version);
    setPendingTargetResolution(null);
    setPendingImport(null);
    setImportState({ kind: "success", fileName: version.sourceFile, version });
    const session = await graphRepository.getSession(activeSessionId);
    if (session) await graphRepository.saveSession(createResearchSession(session.title, {
      id: session.id,
      graphVersionId: version.id,
    }));
    await refreshSessionLists();
    showToast("已切换到所选图谱");
  };

  const requestGraphVersionRebuild = async (version: GraphVersion) => {
    await restoreGraphVersion(version);
    const pending = await rebuildPendingFromVersion(version);
    if (!pending) {
      throw new Error("该版本缺少完整、可验证的原始数据；请重新选择原始文件。 ");
    }
    setWorkspacePanel(null);
    setPendingImport(null);
    setMobilePanel("chat");
    await submitImportDraft(
      pending,
      "使用当前确定性构图管线重建同一份源数据，并保留真实字段语义。",
      version.id,
      "pipeline_upgrade",
    );
  };

  const prepareCurrentGraphAsTargetDomain = async () => {
    if (!graphVersion || targetDomainBusy) return;
    setTargetDomainBusy(true);
    try {
      const prepared = await researchDatasetClient.prepareGraphVersionTargetDomain(graphVersion);
      const readiness = await researchDatasetClient.getReadiness(prepared.artifact.id);
      setDatasetPanelEpoch((current) => current + 1);
      showToast(
        readiness.status === "ready"
          ? "目标域 DatasetArtifact 已建立并通过就绪校验"
          : `目标域 DatasetArtifact 已建立；训练就绪状态为 ${readiness.status}`,
      );
    } catch (error) {
      showToast(error instanceof Error ? error.message : "目标域数据交接失败");
    } finally {
      setTargetDomainBusy(false);
    }
  };

  const switchSession = (sessionId: string) => {
    beginSessionTransition(sessionId);
    setActiveSessionId(sessionId);
    window.localStorage.setItem("socialgraph-fm-active-session", sessionId);
    setSidebarOpen(false);
    showToast("正在恢复本地研究会话");
  };

  const beginRenameSession = (sessionId: string) => {
    const target = [...sessions, ...trashedSessions].find((session) => session.id === sessionId);
    if (!target) return;
    setRenameSessionId(sessionId);
    setRenameDraft(target.title);
    setWorkspacePanel("rename");
  };

  const submitRenameSession = async () => {
    if (!renameSessionId || !renameDraft.trim()) return;
    const target = await graphRepository.getSession(renameSessionId);
    if (!target) return;
    await graphRepository.saveSession(createResearchSession(renameDraft, {
      id: target.id,
      ...(target.graphVersionId ? { graphVersionId: target.graphVersionId } : {}),
      lifecycle: target.lifecycle,
      ...(target.deletedAt ? { deletedAt: target.deletedAt } : {}),
    }));
    await refreshSessionLists();
    setWorkspacePanel(null);
    showToast("会话名称已更新");
  };

  const duplicateSession = async (sessionId: string) => {
    const source = await graphRepository.getSession(sessionId);
    if (!source) return;
    const duplicate = createResearchSession(`${source.title} · 副本`, {
      ...(source.graphVersionId ? { graphVersionId: source.graphVersionId } : {}),
    });
    await graphRepository.saveSession(duplicate);
    const sourceMessages = await graphRepository.listMessages(source.id);
    for (const [index, message] of sourceMessages.entries()) {
      await graphRepository.saveMessage({
        ...message,
        id: makeId("message-copy"),
        sessionId: duplicate.id,
        createdAt: new Date(Date.now() + index).toISOString(),
      });
    }
    await refreshSessionLists();
    switchSession(duplicate.id);
    showToast("已复制会话及其消息");
  };

  const trashSession = async (sessionId: string) => {
    await graphRepository.trashSession(sessionId);
    let remaining = await refreshSessionLists();
    if (sessionId === activeSessionId) {
      if (!remaining.length) {
        const blank = createResearchSession(`新建研究会话 · ${timeNow()}`);
        await graphRepository.saveSession(blank);
        remaining = await refreshSessionLists();
      }
      if (remaining[0]) switchSession(remaining[0].id);
    }
    showToast("会话已移入回收站，可随时恢复");
  };

  const restoreSession = async (sessionId: string) => {
    await graphRepository.restoreSession(sessionId);
    await refreshSessionLists();
    showToast("会话已恢复");
  };

  const purgeSession = async (sessionId: string) => {
    const target = trashedSessions.find((session) => session.id === sessionId);
    if (!target || !window.confirm(`永久删除“${target.title}”及其本地消息？此操作不可撤销。`)) return;
    await graphRepository.purgeSession(sessionId);
    await refreshSessionLists();
    showToast("会话已永久删除");
  };

  const fetchSampleFile = useCallback(async (path: string) => {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`示例文件加载失败（HTTP ${response.status}）`);
    const blob = await response.blob();
    const name = decodeURIComponent(path.split("/").pop() ?? "sample.csv");
    return new File([blob], name, { type: blob.type || "application/octet-stream" });
  }, []);

  const loadSample = useCallback(async (path: string) => {
    try {
      const file = await fetchSampleFile(path);
      await handleFiles([file]);
      setWorkspacePanel(null);
      showToast("示例已附加；可补充数据说明后提交");
    } catch (error) {
      showToast(error instanceof Error ? error.message : "示例加载失败");
    }
  }, [fetchSampleFile, handleFiles, showToast]);

  const runLlmDiagnostic = useCallback(async () => {
    setLlmDiagnostic({ state: "running" });
    const startedAt = performance.now();
    try {
      const normalized = await intentNormalizer.normalizeIntent({
        text: "识别这张协作网络中的桥接节点并高亮，使用两跳邻域查看关键成员。",
        ...(graphVersion ? { graphContext: buildGraphContextSummary(graphVersion) } : {}),
      });
      setLlmDiagnostic({
        state: "success",
        latencyMs: Math.round(performance.now() - startedAt),
        schemaVersion: normalized.meta.schemaVersion,
        requestId: normalized.meta.requestId,
        ...(normalized.meta.model ? { model: normalized.meta.model } : {}),
        source: normalized.meta.source,
        ...(normalized.kind === "analysis_request" ? { task: normalized.task } : {}),
        warnings: normalized.meta.warnings,
      });
      setIntentServiceStatus({
        state: "llm",
        label: "LLM 本次调用成功",
        ...(normalized.meta.model ? { model: normalized.meta.model } : {}),
      });
      setWorkspacePanel("diagnostics");
    } catch (error) {
      setLlmDiagnostic({
        state: "error",
        latencyMs: Math.round(performance.now() - startedAt),
        message: error instanceof Error ? error.message : "LLM 诊断请求失败",
      });
    }
  }, [graphVersion, intentNormalizer]);

  const inspectResearchSample = useCallback(async () => {
    setDatasetDiagnostic({ state: "running" });
    try {
      const file = await fetchSampleFile(GEOM_GCN_SAMPLE_PATH);
      const form = new FormData();
      form.append("files", file, file.name);
      const response = await fetch(socialGraphApiUrl("/api/v1/dataset-imports/inspect"), {
        method: "POST",
        body: form,
      });
      const payload = await response.json() as {
        id?: string;
        inspectionId?: string;
        detectedFormat?: string;
        status?: string;
        profile?: { nodeCount?: number; edgeCount?: number };
        issues?: readonly { message?: string }[];
      };
      if (!response.ok || payload.status !== "accepted") {
        setDatasetDiagnostic({
          state: "rejected",
          ...(payload.detectedFormat ? { detectedFormat: payload.detectedFormat } : {}),
          message: payload.issues?.[0]?.message ?? `研究数据检查失败（HTTP ${response.status}）`,
        });
        return;
      }
      setDatasetDiagnostic({
        state: "accepted",
        ...(payload.detectedFormat ? { detectedFormat: payload.detectedFormat } : {}),
        ...(payload.inspectionId ?? payload.id ? { inspectionId: payload.inspectionId ?? payload.id } : {}),
        ...(payload.profile?.nodeCount !== undefined ? { nodeCount: payload.profile.nodeCount } : {}),
        ...(payload.profile?.edgeCount !== undefined ? { edgeCount: payload.profile.edgeCount } : {}),
      });
    } catch (error) {
      setDatasetDiagnostic({
        state: "error",
        message: error instanceof Error ? error.message : "研究数据检查服务不可用",
      });
    }
  }, [fetchSampleFile]);

  const runFullChainDiagnostic = useCallback(async () => {
    setLlmDiagnostic({ state: "running" });
    const startedAt = performance.now();
    try {
      const file = await fetchSampleFile("/samples/governance-collaboration.csv");
      const artifact = await createSourceArtifact(file, "single");
      const imported = await importAdapter.parse(file, undefined, {
        sourceArtifacts: [artifact],
        provenance: browserImportProvenance(),
      });
      if (imported.status !== "ready" || !imported.graphVersion) {
        throw new Error(imported.error ?? "示例未能生成图谱版本");
      }
      const version = imported.graphVersion;
      const fileEntry: ChatEntry = {
        id: makeId("diagnostic-file"),
        role: "user",
        text: "运行完整链路诊断：加载合成治理协作网络并识别桥接节点。",
        timestamp: timeNow(),
        file: { name: file.name, size: file.size },
      };
      setMessages((current) => [...current, fileEntry]);
      await persistChatEntry(fileEntry);
      await registerImportedGraph(version, file, [artifact], fileEntry.id);

      const normalized = await intentNormalizer.normalizeIntent({
        text: "找出桥接节点并高亮，查看跨区协调员的两跳邻居。",
        graphContext: buildGraphContextSummary(version),
      });
      if (normalized.kind !== "analysis_request") throw new Error("诊断问题未被识别为图分析请求");
      const command: ViewCommand = normalized.view ?? {
        mode: "local",
        focusTerms: ["跨区协调员"],
        depth: 2,
        nodeTypeTerms: [],
        edgeTypeTerms: [],
        overlay: "articulation",
      };
      const outcome = applyViewCommand(version, createDefaultGraphViewState(version.id), command);
      replaceViewState(outcome.nextState);
      const requestedOverlay = outcome.requestedOverlay ?? "articulation";
      const diagnosticSemanticSlice = buildSemanticGraphSlice(version, { viewState: outcome.nextState });
      const scopedGraph = createScopedGraphSlice(
        version.id,
        diagnosticSemanticSlice.slice.nodes,
        diagnosticSemanticSlice.slice.edges,
        diagnosticSemanticSlice.filters,
        false,
      );
      const diagnosticScene = buildGraphScene(version, { viewState: outcome.nextState });
      graphVersionOverlayController.setExplicit(
        version.id,
        withScopeProvenance(buildRequestedOverlay(version, requestedOverlay), scopedGraph.scope.scopeHash),
      );
      let run = await analysisExecutor.createAnalysis({
        graphVersionId: version.id,
        graphVersion: version,
        intent: normalized,
        scopedGraph,
      });
      if (run.status === "queued" || run.status === "running") run = await analysisExecutor.getAnalysis(run.id);
      await graphRepository.saveAnalysisRun(run);
      const finalEntry: Extract<ChatEntry, { role: "assistant" }> = {
        id: makeId("diagnostic-result"),
        role: "assistant",
        text: "完整链路已通过：示例解析、图谱版本、意图理解、视图命令、本地图算法和高亮覆盖层均已完成。",
        timestamp: timeNow(),
        state: "success",
        intent: normalized,
        intentMeta: normalized.meta,
        run,
      };
      setMessages((current) => [...current, finalEntry]);
      await persistChatEntry(finalEntry, normalized);
      setLlmDiagnostic({
        state: "success",
        latencyMs: Math.round(performance.now() - startedAt),
        schemaVersion: normalized.meta.schemaVersion,
        requestId: normalized.meta.requestId,
        ...(normalized.meta.model ? { model: normalized.meta.model } : {}),
        source: normalized.meta.source,
        task: normalized.task,
        warnings: [...normalized.meta.warnings, ...outcome.warnings],
      });
      setWorkspacePanel("diagnostics");
      setMobilePanel("graph");
      showToast("完整链路诊断已完成");
    } catch (error) {
      setLlmDiagnostic({
        state: "error",
        latencyMs: Math.round(performance.now() - startedAt),
        message: error instanceof Error ? error.message : "完整链路诊断失败",
      });
      setWorkspacePanel("diagnostics");
    }
  }, [
    analysisExecutor,
    fetchSampleFile,
    graphVersionOverlayController,
    graphRepository,
    importAdapter,
    intentNormalizer,
    persistChatEntry,
    registerImportedGraph,
    replaceViewState,
    showToast,
  ]);

  const governanceTaskEntries: readonly GovernanceTaskEntry[] = [
    { id: "session", label: "当前会话治理", kind: "session", snapshot: governanceWorkspace.snapshot, graph: null },
    ...adaptationGovernanceTargets,
  ];
  const activeGovernanceTask = resolveGovernanceTask(governanceTaskEntries, activeGovernanceTaskId);
  const activeGovernanceSnapshot = activeGovernanceTask?.snapshot ?? null;
  const activeGovernanceSessionId = activeGovernanceTask?.kind === "target" ? activeGovernanceTask.id : activeSessionId;
  const governanceGraphVersion = governanceGraphPresentation?.graph ?? (activeGovernanceTask?.kind === "target" ? activeGovernanceTask.graph : null);
  const operationalGraphVersion = activeWorkspace === "adaptation"
    ? adaptationOverviewGraph
    : governanceGraphVersion;
  const displayedGraphVersion = activeWorkspace === "chat" ? graphVersion : operationalGraphVersion;
  const graphReady = Boolean(displayedGraphVersion);
  const graphPaneVisible = isWorkspaceGraphPaneVisible(viewportWidth, mobilePanel);
  const workspaceTitle = activeWorkspace === "governance"
    ? "治理应用"
    : activeWorkspace === "adaptation"
      ? "适配能力"
      : "对话研究";
  const primaryMobilePanel = activeWorkspace === "governance"
    ? "governance"
    : activeWorkspace === "adaptation"
      ? "adaptation"
      : "chat";
  const activeGovernanceRunId = governanceWorkspace.snapshot?.run?.status === "succeeded"
    ? governanceWorkspace.snapshot.run.runId
    : undefined;
  const governanceUploadPendingFileName = importState.kind === "parsing"
    && importState.fileName.toLocaleLowerCase("en-US").endsWith(".zip")
    ? importState.fileName
    : null;
  const chatComposerUnavailable = !sessionReady
    || hydratedSessionId !== activeSessionId
    || governanceWorkspace.restoreState === "restoring"
    || governanceUploadPendingFileName !== null;
  const showHero = (messages.length === 0 && importState.kind === "idle" && !graphVersion) || heroManuallyOpen;
  const sidebarSessions = sessions
    .filter((session) => session.id !== DEMO_SESSION_ID && session.id !== "volunteer-network" && session.id !== "industry-partners")
    .map((session) => ({
    id: session.id,
    title: session.title,
    time: sessionTime(session.updatedAt),
  }));
  const workspaceLayoutRoute: WorkspaceLayoutRoute = activeWorkspace === "chat" ? "research" : activeWorkspace;
  const resolvedWorkspaceLayout = useMemo(
    () => resolveWorkspaceLayout({ state: workspaceLayout, viewportWidth, route: workspaceLayoutRoute }),
    [viewportWidth, workspaceLayout, workspaceLayoutRoute],
  );
  const resolvedGraphHeight = useMemo(
    () => resolveWorkspaceGraphHeight(workspaceLayout, viewportHeight),
    [viewportHeight, workspaceLayout],
  );
  const workspaceStyle = {
    "--workspace-left-width": `${resolvedWorkspaceLayout.leftWidth}px`,
    "--workspace-right-width": `${resolvedWorkspaceLayout.rightWidth}px`,
    "--workspace-right-expanded-width": `${resolvedWorkspaceLayout.rightWidth}px`,
    "--workspace-left-rail": `${resolvedWorkspaceLayout.leftRailWidth}px`,
    "--workspace-right-rail": `${resolvedWorkspaceLayout.rightRailWidth}px`,
    "--workspace-central-minimum": `${resolvedWorkspaceLayout.centralMinimum}px`,
    "--workspace-graph-height": `${resolvedGraphHeight.height}px`,
  } as CSSProperties;
  const llmStatusDescription = llmDiagnostic.source === "llm"
    ? "本次 LLM 调用成功"
    : intentServiceStatus.state === "llm"
      ? "LLM 已配置 · 等待本次调用验证"
      : intentServiceStatus.label;
  const governanceStatusDescription = governanceServiceState.state === "checking"
    ? "正在加载在线风险模型"
    : governanceServiceState.state === "ready"
      ? `SocialGraph-FM Governance 在线，可使用 ${governanceServiceState.device.toUpperCase()} 推理`
      : governanceServiceState.state === "model_unavailable"
        ? "SocialGraph-FM Governance 服务已连接，在线模型尚未就绪"
        : "SocialGraph-FM Governance 服务不可用；普通结构分析仍可使用";

  return <SocialGraphWorkspaceView model={{
    GraphPreview,
    activeAdaptationPresentation,
    activeGovernanceRunId,
    activeGovernanceSessionId,
    activeGovernanceSnapshot,
    activeGovernanceTask,
    activeOverlay,
    activeSessionId,
    activeWorkspace,
    adaptationLanePresentation,
    adaptationModelCardState,
    adaptationMounted,
    adaptationTheme,
    applyNormalizedView,
    beginRenameSession,
    bindGovernanceWorkspaceSnapshot,
    cacheActiveAdaptationCamera,
    chatComposerUnavailable,
    chatLocateOverview,
    composerInputRef,
    confirmChatAction,
    currentGraphVersionIdRef,
    currentGraphVersionRef,
    currentWorkspaceCameraKey,
    datasetPanelEpoch,
    draft,
    duplicateSession,
    executeNormalizedAnalysis,
    fileInputRef,
    coreService,
    governanceStatusDescription,
    governanceFilters,
    governanceGraphPresentation,
    governanceMounted,
    governanceTaskEntries,
    governanceTheme,
    graphExportHandlers,
    graphPaneVisible,
    graphReady,
    graphRepository,
    graphScene,
    graphVersion,
    graphVersionOverlayController,
    handleAdaptationGraphNodeSelect,
    handleFiles,
    handleFocusNodeIdsChange,
    handleGovernanceViewSnapshot,
    handleGraphExport,
    handleGraphFiltersChange,
    handleGraphNodeSelect,
    handleGraphViewModeChange,
    handleGraphViewSnapshot,
    handleLocateGovernanceCandidates,
    handlePathEndpointIdsChange,
    handleWelcomePrompt,
    heroManuallyOpen,
    hydratedSessionId,
    importRequestEpochRef,
    importState,
    intentServiceStatus,
    governanceOnlineClient,
    governanceServiceState,
    governanceSkillsClient,
    governanceUploadPendingFileName,
    governanceWorkspace,
    isDragging,
    isSending,
    llmDiagnostic,
    makeId,
    messages,
    mobilePanel,
    operationalGraphVersion,
    patchGraphView,
    pendingImport,
    pendingTargetResolution,
    persistChatEntry,
    prepareCurrentGraphAsTargetDomain,
    preparePendingImport,
    primaryMobilePanel,
    purgeSession,
    ragPanelOpen,
    registerDatasetArtifact,
    registerImportedGraph,
    renameDraft,
    reparseImportMapping,
    requestGraphVersionRebuild,
    requestWorkspaceCameraRestore,
    researchDatasetClient,
    researchScrollRef,
    resolvedGraphHeight,
    resolvedWorkspaceLayout,
    restoreAdaptationOverview,
    restoreChatLocateOverview,
    restoreGraphVersion,
    restoreResearchUserGraph,
    restoreSession,
    runLlmDiagnostic,
    selectMobileWorkspacePanel,
    selectedNode,
    sendMessage,
    sessionReady,
    sessionTime,
    sessions,
    setActiveGovernanceTaskId,
    setActiveWorkspace,
    setAdaptationGovernanceTargets,
    setAdaptationLanePresentation,
    setAdaptationTheme,
    setDraft,
    setGovernanceFilters,
    setGovernanceGraphPresentation,
    setGovernanceMounted,
    setGovernanceTheme,
    setGraphExportHandlers,
    setHeroManuallyOpen,
    setImportState,
    setIsDragging,
    setMessages,
    setMobilePanel,
    setPendingImport,
    setPendingTargetResolution,
    setRagPanelOpen,
    setRenameDraft,
    setSidebarOpen,
    setWorkspaceLayout,
    setWorkspacePanel,
    showHero,
    showToast,
    sidebarOpen,
    sidebarSessions,
    startNewSession,
    storageCapacity,
    submitRenameSession,
    switchAdaptationGraph,
    switchSession,
    targetDomainBusy,
    timeNow,
    toast,
    trashSession,
    trashedSessions,
    viewState,
    viewportHeight,
    viewportWidth,
    workspaceCameraRestore,
    workspaceLayout,
    workspaceLayoutRoute,
    workspacePanel,
    workspaceStyle,
    workspaceTitle,
  }} />;
}
