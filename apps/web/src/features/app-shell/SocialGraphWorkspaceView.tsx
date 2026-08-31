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
import { GovernanceWorkspaceProvider, useGovernanceWorkspace } from "../../components/GovernanceWorkspaceProvider";
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
  GraphPreviewProps,
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
} from "../app-shell/navigation";
import { ORDINARY_PRESENTATION_COPY } from "../app-shell/presentationCopy";
import {
  WelcomeAtlas,
  researchPromptSkillRequest,
  researchPromptForText,
  researchPrompts,
  welcomePromptAction,
  type ResearchPrompt,
} from "../app-shell/welcome";
import {
  GraphVersionOverlayController,
  locateGovernanceCandidates,
  resolveGovernanceCandidateFocus,
  resolveGraphVersionOverlay,
} from "../governance/overlayController";
import {
  buildAnalysisResultMarkdown,
  ensureHumanReviewGuidance,
  resultDescription,
} from "../governance/reports";
import {
  AssistantEntry,
  FileBadge,
  GraphBuildReviewCard,
  ImportTimeline,
  TargetResolutionCard,
  UserEntry,
  publicAssistantCopy,
} from "../graph-workbench/conversationPanels";
import {
  RightSummary,
  publicGraphSourceLabel,
} from "../graph-workbench/RightSummary";

type AnalysisExecution =
  | { readonly kind: "ambiguous"; readonly warnings: readonly string[] }
  | { readonly kind: "completed"; readonly run: AnalysisRun; readonly warnings: readonly string[] };

export function SocialGraphWorkspaceView({ model }: { readonly model: Record<string, any> }) {
  const {
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
  } = model;
  const resizePanePreservingCamera = (pane: "left" | "right", delta: number) => {
    requestWorkspaceCameraRestore(currentWorkspaceCameraKey(activeWorkspace));
    setWorkspaceLayout((current: WorkspaceLayoutState) => {
      const resolved = resolveWorkspaceLayout({
        state: current,
        viewportWidth,
        route: workspaceLayoutRoute,
      });
      const side = pane === "left" ? resolved.leftWidth : resolved.rightWidth;
      const bounds = pane === "left" ? resolved.leftBounds : resolved.rightBounds;
      return resizeWorkspacePane(current, pane, delta, { currentValue: side, bounds });
    });
  };
  return <div
      className={`app-shell ${workspaceLayout.leftCollapsed ? "is-left-collapsed" : ""} ${workspaceLayout.rightCollapsed ? "is-right-collapsed" : ""}`}
      style={workspaceStyle}
    >
      <div className={`sidebar-host ${sidebarOpen ? "is-open" : ""}`}>
        <Sidebar
          activeWorkspace={activeWorkspace}
          onWorkspaceChange={(workspace) => {
            if (workspace === "chat") restoreResearchUserGraph();
            if (workspace === "governance" || workspace === "adaptation") {
              setGovernanceMounted(true);
            }
            window.history.pushState(null, "", hashForWorkspaceRoute(routeForSidebarWorkspace(workspace)));
            setWorkspacePanel(null);
            requestWorkspaceCameraRestore(currentWorkspaceCameraKey(workspace));
            setActiveWorkspace(workspace);
            setMobilePanel(workspace === "governance" ? "governance" : workspace === "adaptation" ? "adaptation" : "chat");
            setSidebarOpen(false);
          }}
          activeSession={activeSessionId}
          sessions={sidebarSessions}
          onSessionChange={(sessionId) => {
            restoreResearchUserGraph();
            window.history.pushState(null, "", hashForWorkspaceRoute("research"));
            setActiveWorkspace("chat");
            setMobilePanel("chat");
            switchSession(sessionId);
          }}
          onNewSession={() => {
            restoreResearchUserGraph();
            window.history.pushState(null, "", hashForWorkspaceRoute("research"));
            setActiveWorkspace("chat");
            setMobilePanel("chat");
            void startNewSession();
          }}
          onClose={() => setSidebarOpen(false)}
          collapsed={workspaceLayout.leftCollapsed}
          onToggleCollapsed={() => setWorkspaceLayout((current: WorkspaceLayoutState) => ({ ...current, leftCollapsed: !current.leftCollapsed }))}
          onShowSessions={() => setWorkspacePanel("sessions")}
          onRenameSession={beginRenameSession}
          onDuplicateSession={(sessionId) => void duplicateSession(sessionId)}
          onTrashSession={(sessionId) => void trashSession(sessionId)}
          datasetsOpen={workspacePanel === "datasets"}
          onOpenDatasets={() => {
            window.history.pushState(null, "", hashForWorkspaceRoute("datasets"));
            setWorkspacePanel("datasets");
          }}
          onSupportAction={() => setWorkspacePanel("guide")}
        />
      </div>
      {sidebarOpen ? <button className="drawer-scrim" type="button" aria-label="关闭导航" onClick={() => setSidebarOpen(false)} /> : null}

      {resolvedWorkspaceLayout.leftResizerVisible ? <WorkspaceResizeHandle
        className="workspace-resizer--left"
        axis="vertical"
        label="调整项目导航宽度"
        value={resolvedWorkspaceLayout.leftWidth}
        minimum={resolvedWorkspaceLayout.leftBounds.minimum}
        maximum={resolvedWorkspaceLayout.leftBounds.maximum}
        onDelta={(delta) => resizePanePreservingCamera("left", delta)}
        onReset={() => setWorkspaceLayout((current: WorkspaceLayoutState) => ({
          ...current,
          leftWidth: DEFAULT_WORKSPACE_LAYOUT.leftWidth,
          leftCollapsed: false,
        }))}
      /> : null}

      <main className={`research-column is-${activeWorkspace}-workspace`}>
        <header className="mobile-header">
          <button className="icon-button" type="button" aria-label="打开导航" onClick={() => setSidebarOpen(true)}>
            <SidebarSimple size={21} weight="light" />
          </button>
          <div><img src="/assets/brand-mark.png" alt="" /><strong>SocialGraph-FM</strong></div>
          <span className="local-status"><span />本地</span>
        </header>

        <div className="research-header">
          <div>
            <span className="section-symbol"><Sparkle size={16} weight="fill" /></span>
            <h1>{workspaceTitle}</h1>
            {activeWorkspace === "chat" && !showHero ? <button className="hero-reopen" type="button" onClick={() => setHeroManuallyOpen(true)}>查看开始页</button> : null}
          </div>
          <div className="connection-statuses" aria-label="服务连接状态">
            {governanceServiceState.state === "model_unavailable" || governanceServiceState.state === "unavailable" ? <button
              type="button"
              className="status-pill diagnostics-trigger is-system is-partial is-icon-only"
              title={governanceStatusDescription}
              aria-label={`${governanceStatusDescription}。点击查看系统状态`}
              aria-live="polite"
              onClick={() => setWorkspacePanel("diagnostics")}
            >
              <WarningCircle size={16} aria-hidden="true" />
            </button> : null}
            <button
              className="icon-button pane-collapse-button"
              type="button"
              aria-label={workspaceLayout.rightCollapsed ? "展开图谱栏" : "折叠图谱栏"}
              onClick={() => setWorkspaceLayout((current: WorkspaceLayoutState) => ({ ...current, rightCollapsed: !current.rightCollapsed }))}
            >
              {workspaceLayout.rightCollapsed ? <CaretLeft size={17} /> : <CaretRight size={17} />}
            </button>
          </div>
        </div>

        <div className={`mobile-tabs mobile-tabs--workspace ${activeWorkspace !== "chat" ? "is-governance" : ""}`} role="tablist" aria-label={activeWorkspace === "chat" ? "研究视图" : `${workspaceTitle}视图`}>
          <button
            type="button"
            role="tab"
            aria-selected={mobilePanel === primaryMobilePanel}
            onClick={() => selectMobileWorkspacePanel(primaryMobilePanel)}
          >{activeWorkspace === "chat" ? "对话" : "任务"}</button>
          <button type="button" role="tab" aria-selected={mobilePanel === "graph"} onClick={() => selectMobileWorkspacePanel("graph")}>图谱{graphReady ? <span /> : null}</button>
        </div>

        <div ref={researchScrollRef} className={`research-scroll ${mobilePanel === "graph" ? "is-mobile-hidden" : ""}`}>
          {governanceMounted ? (
            <section
              className="governance-page"
              aria-label="治理应用工作台"
              aria-hidden={activeWorkspace !== "governance"}
              hidden={activeWorkspace !== "governance"}
            >
              <GovernanceTaskSelector entries={governanceTaskEntries} activeId={activeGovernanceTask?.id ?? "session"} onSelect={(taskId) => {
                if (taskId === activeGovernanceTask?.id && activeGovernanceTask.kind === "target") {
                  setAdaptationGovernanceTargets((items: readonly GovernanceTaskEntry[]) => items.map((item) => item.id === taskId
                    ? { ...item, validationToken: (item.validationToken ?? 0) + 1 }
                    : item));
                }
                setActiveGovernanceTaskId(taskId);
                setGovernanceGraphPresentation(null);
                setGovernanceFilters({ nodeTypes: [], edgeTypes: [] });
                setRagPanelOpen(false);
              }} />
              <GovernanceOnlineWorkspace
                key={governanceWorkspaceMountKey(activeGovernanceTask)}
                client={governanceOnlineClient}
                sessionId={activeGovernanceSessionId}
                sharedSnapshot={activeGovernanceSnapshot}
                sharedUploadPendingFileName={activeGovernanceTask?.kind === "target" ? null : governanceUploadPendingFileName}
                sharedRestoreMessage={activeGovernanceTask?.kind === "target" ? null : governanceWorkspace.restoreMessage}
                adaptation={activeGovernanceTask?.kind === "target" ? activeGovernanceTask.adaptation : undefined}
                adaptationValidationToken={activeGovernanceTask?.kind === "target" ? activeGovernanceTask.validationToken : undefined}
                onSharedSnapshotChange={(snapshot) => {
                  if (activeGovernanceTask?.kind === "target") {
                    setAdaptationGovernanceTargets((items: readonly GovernanceTaskEntry[]) => items.map((item) => item.id === activeGovernanceTask.id ? { ...item, snapshot } : item));
                    return;
                  }
                  void bindGovernanceWorkspaceSnapshot(snapshot);
                }}
                onGlobalResultAccepted={(run, result) => {
                  if (activeGovernanceTask?.kind === "target") return;
                  const acceptedGraph = currentGraphVersionRef.current;
                  if (!acceptedGraph) return;
                  graphVersionOverlayController.acceptGlobalResult(acceptedGraph, {
                    protocol: "global",
                    status: "succeeded",
                    graphVersionHash: result.graphVersionHash,
                    runId: run.runId,
                    resultHash: result.resultHash,
                  });
                }}
                onGraphPresentationChange={setGovernanceGraphPresentation}
                ragOpen={ragPanelOpen}
                onRagOpenChange={setRagPanelOpen}
                assistantPanel={<GovernanceRagPanel
                  client={governanceSkillsClient}
                  context={governanceGraphPresentation?.skillsContext ?? null}
                  embedded
                  onClose={() => setRagPanelOpen(false)}
                />}
                evidenceSummaryClient={governanceSkillsClient}
              />
            </section>
          ) : null}
          {adaptationMounted ? (
            <section className="adaptation-page" aria-label="适配能力工作台" hidden={activeWorkspace !== "adaptation"} aria-hidden={activeWorkspace !== "adaptation"} inert={activeWorkspace !== "adaptation"}>
              <AdaptationWorkspace
                client={governanceOnlineClient}
                modelCardState={adaptationModelCardState}
                onOverlayChange={() => undefined}
                onLanePresentationChange={(lane, patch) => setAdaptationLanePresentation((state: AdaptationLanePresentationState) => updateAdaptationLanePresentation(state, lane, patch))}
                onActiveLaneChange={(lane) => setAdaptationLanePresentation((state: AdaptationLanePresentationState) => activateAdaptationLanePresentation(state, lane))}
                onGovernanceHandoff={(target: AdaptationGovernanceTarget) => {
                  cacheActiveAdaptationCamera();
                  const entry = governanceTaskEntryFromAdaptationTarget(target);
                  setAdaptationGovernanceTargets((items: readonly GovernanceTaskEntry[]) => [...items.filter((item) => item.id !== entry.id), entry]);
                  setActiveGovernanceTaskId(entry.id);
                  setGovernanceMounted(true);
                  setGovernanceGraphPresentation(null);
                  window.history.pushState(null, "", hashForWorkspaceRoute("governance"));
                  setActiveWorkspace("governance");
                  setMobilePanel("governance");
                }}
                onClose={() => {
                  window.history.pushState(null, "", hashForWorkspaceRoute("research"));
                  setActiveWorkspace("chat");
                  setMobilePanel("chat");
                }}
              />
            </section>
          ) : null}
          {activeWorkspace === "chat" ? (
            <>
          {showHero ? (
            <>
              {heroManuallyOpen ? <button className="hero-reopen" type="button" onClick={() => setHeroManuallyOpen(false)}>收起开始页</button> : null}
              <WelcomeAtlas onPrompt={handleWelcomePrompt} onUpload={() => fileInputRef.current?.click()} />
              <div className="conversation-divider"><span>当前研究</span></div>
            </>
          ) : null}

          <section className="conversation" aria-label="研究对话">
            {messages.length === 0 && !showHero ? (
              <div className="conversation-empty">
                <CloudArrowUp size={30} weight="light" />
                <strong>先从真实数据开始</strong>
                <p>上传 CSV / TSV、JSON、GraphML 或 GEXF。训练张量由安全研究数据适配服务检查。</p>
                <button className="secondary-button" type="button" onClick={() => fileInputRef.current?.click()}><FolderOpen size={17} />选择文件</button>
              </div>
            ) : null}
            {messages.map((entry: ChatEntry) =>
              entry.role === "user" ? <UserEntry entry={entry} key={entry.id} /> : <AssistantEntry
                entry={entry}
                key={entry.id}
                onRetry={(text) => void sendMessage(text)}
                onConfirm={(item) => void confirmChatAction(item)}
                activeGovernanceRunId={activeGovernanceRunId}
                onOpenReview={() => {
                  window.history.pushState(null, "", hashForWorkspaceRoute("governance"));
                  setGovernanceMounted(true);
                  setActiveWorkspace("governance");
                  setMobilePanel("governance");
                }}
                onLocateGraph={handleLocateGovernanceCandidates}
              />,
            )}
            <ImportTimeline state={importState} />
            {importState.kind === "roles" ? (
              <GraphTableRoleCard
                files={importState.files}
                profiles={importState.profiles}
                initialEdgeIndex={importState.initialEdgeIndex}
                onApply={(edgeIndex) => {
                  const expectedBase = importState.baseGraphVersionId ?? null;
                  if (currentGraphVersionIdRef.current !== expectedBase) {
                    showToast("当前图谱已变化；请重新附加两个文件");
                    return;
                  }
                  void preparePendingImport(
                    importState.files,
                    importState.profiles,
                    edgeIndex,
                    importState.baseGraphVersionId,
                  ).catch((error: unknown) => {
                    setImportState({
                      kind: "error",
                      fileName: importState.files.map((file: File) => file.name).join(" + "),
                      message: error instanceof Error ? error.message : "文件角色确认失败。",
                      issues: [],
                    });
                  });
                }}
                onCancel={() => {
                  importRequestEpochRef.current += 1;
                  setPendingImport(null);
                  setImportState({ kind: "idle" });
                }}
              />
            ) : null}
            {importState.kind === "mapping" ? (
              <GraphImportMappingCard
                key={importState.pending.requestToken}
                nodeProfile={pendingProfileForRole(importState.pending, "nodes")}
                edgeProfile={pendingProfileForRole(importState.pending, "edges") ?? pendingProfileForRole(importState.pending, "single") ?? importState.pending.profiles[0]}
                initialNodeMapping={importState.spec.nodeMapping}
                initialEdgeMapping={importState.spec.edgeMapping}
                initialTimeFormat={importState.spec.timeFormat}
                issues={importState.issues}
                onApply={(value) => void reparseImportMapping(importState, value).catch((error: unknown) => {
                  showToast(error instanceof Error ? error.message : "字段映射解析失败");
                })}
                onCancel={() => {
                  importRequestEpochRef.current += 1;
                  setImportState({ kind: "idle" });
                  setPendingImport(null);
                }}
              />
            ) : null}
            {importState.kind === "review" ? (
              <GraphBuildReviewCard
                state={importState}
                onEdit={() => {
                  setImportState({
                    kind: "mapping",
                    pending: importState.pending,
                    spec: importState.spec,
                    issues: importState.run.issues,
                    source: importState.source,
                    normalizationWarnings: importState.warnings,
                    ...(importState.parentVersionId ? { parentVersionId: importState.parentVersionId } : {}),
                    ...(importState.reconstructionReason
                      ? { reconstructionReason: importState.reconstructionReason }
                      : {}),
                  });
                }}
                onCancel={() => {
                  importRequestEpochRef.current += 1;
                  setImportState({ kind: "idle" });
                  setPendingImport(null);
                  showToast("已取消构图草稿，当前图版本未变化");
                }}
                onConfirm={() => {
                  const review = importState;
                  const expectedBase = review.pending.baseGraphVersionId ?? null;
                  if (currentGraphVersionIdRef.current !== expectedBase) {
                    showToast("当前图谱已变化；该草稿不能提交");
                    return;
                  }
                  if (JSON.stringify(review.run.graphVersion.buildSpec) !== JSON.stringify(review.spec)) {
                    showToast("构图字段已变化；请重新验证草稿后再保存");
                    return;
                  }
                  void registerImportedGraph(
                    review.run.graphVersion,
                    review.pending.files[0],
                    review.pending.artifacts,
                    review.pending.sourceMessageId,
                  ).then(() => {
                    setPendingImport(null);
                  }).catch((error: unknown) => showToast(error instanceof Error ? error.message : "版本保存失败"));
                }}
              />
            ) : null}
            {pendingTargetResolution && graphVersion ? (
              <TargetResolutionCard
                key={`${graphVersion.id}:${pendingTargetResolution.resolutions.map((item: TargetResolution) => item.term).join("|")}`}
                graph={graphVersion}
                pending={pendingTargetResolution}
                onApply={(nodeIds) => {
                  const pending = pendingTargetResolution;
                  if (!pending.intent) {
                    const warnings = applyNormalizedView(pending.command, nodeIds);
                    showToast(warnings.length ? "已应用已解析节点，部分提示保留" : "自然语言视图已应用");
                    return;
                  }
                  void executeNormalizedAnalysis(pending.intent, nodeIds).then(async (execution: AnalysisExecution) => {
                    if (execution.kind !== "completed") return;
                    const unavailable = execution.run.engine === "unavailable" || execution.run.result?.kind === "unavailable";
                    const unavailableReason = describeUnavailableAnalysis(
                      execution.run.result?.kind === "unavailable"
                        ? execution.run.result
                        : undefined,
                      coreService,
                    );
                    const entry: Extract<ChatEntry, { role: "assistant" }> = {
                      id: makeId("assistant-resolution"),
                      role: "assistant",
                      text: unavailable
                        ? `已确认分析目标。${unavailableReason}`
                        : `已确认目标并在同一可见范围运行本地图算法（${execution.run.scope?.nodeCount ?? 0} 节点 / ${execution.run.scope?.edgeCount ?? 0} 关系）。`,
                      timestamp: timeNow(),
                      state: unavailable ? "warning" : "success",
                      intent: pending.intent!,
                      intentMeta: pending.intent!.meta,
                      run: execution.run,
                    };
                    setMessages((current: ChatEntry[]) => [...current, entry]);
                    await persistChatEntry(entry, pending.intent);
                    showToast("已确认目标并完成分析");
                  }).catch((error: unknown) => showToast(error instanceof Error ? error.message : "分析失败"));
                }}
                onCancel={() => setPendingTargetResolution(null)}
              />
            ) : null}
          </section>
            </>
          ) : null}
        </div>

        {activeWorkspace === "chat" ? <div className={`composer-wrap ${mobilePanel !== "chat" ? "is-mobile-hidden" : ""}`}>
          {pendingImport ? (
            <div className="user-message-body" role="group" aria-label="待提交的数据附件">
              {pendingImport.files.map((file: File) => <FileBadge file={file} key={`${file.name}:${file.size}`} />)}
              <button
                type="button"
                className="secondary-button"
                onClick={() => { importRequestEpochRef.current += 1; setPendingImport(null); setImportState({ kind: "idle" }); }}
              >
                移除附件
              </button>
            </div>
          ) : null}
          <div
            className={`composer ${isDragging ? "is-dragging" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => { if (event.currentTarget === event.target) setIsDragging(false); }}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              if (!chatComposerUnavailable) {
                handleFiles(event.dataTransfer.files);
              }
            }}
          >
            <button
              className="attach-button"
              type="button"
              onClick={() => fileInputRef.current?.click()}
              aria-label="上传关系图文件"
              disabled={chatComposerUnavailable}
            >
              <Paperclip size={23} weight="light" />
            </button>
            <textarea
              ref={composerInputRef}
              value={draft}
              rows={2}
              maxLength={2000}
              disabled={chatComposerUnavailable}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (shouldSubmitComposerKey({
                  key: event.key,
                  shiftKey: event.shiftKey,
                  isComposing: event.nativeEvent.isComposing,
                })) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder={!sessionReady || hydratedSessionId !== activeSessionId
                ? "正在恢复研究会话…"
                : governanceUploadPendingFileName
                  ? `正在检查 ${governanceUploadPendingFileName}，完成后即可开始分析…`
                : isDragging
                ? "松开以上传关系图文件"
                : pendingImport
                  ? "可选：说明起点/终点、方向、权重或时间字段，然后一起提交…"
                  : "继续说明研究问题，或上传 CSV / TSV / JSON / GraphML / GEXF…"}
              aria-label="研究问题"
            />
            <button className="send-button" type="button" onClick={() => void sendMessage()} disabled={chatComposerUnavailable || (!draft.trim() && !pendingImport) || isSending} aria-label={pendingImport ? "提交附件与数据说明" : "发送研究问题"}>
              {isSending ? <CircleNotch size={22} className="spin" /> : <PaperPlaneTilt size={22} weight="fill" />}
            </button>
            <div className={`composer-meta ${!graphVersion && draft.trim() ? "is-guidance" : ""}`}>
              <small>{!graphVersion && draft.trim()
                ? "请先上传关系数据，再发送这个研究目标"
                : "Enter 发送 · Shift + Enter 换行"}</small>
            </div>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            disabled={chatComposerUnavailable}
            accept=".zip,.csv,.tsv,.json,.graphml,.gexf,.xml,application/zip,text/csv,text/tab-separated-values,application/json,application/xml,text/xml"
            hidden
            onChange={(event) => {
              if (event.target.files) handleFiles(event.target.files);
              event.target.value = "";
            }}
          />
        </div> : null}
      </main>

      {resolvedWorkspaceLayout.rightResizerVisible ? <WorkspaceResizeHandle
        className="workspace-resizer--right"
        axis="vertical"
        label="调整图谱栏宽度"
        value={resolvedWorkspaceLayout.rightWidth}
        minimum={resolvedWorkspaceLayout.rightBounds.minimum}
        maximum={resolvedWorkspaceLayout.rightBounds.maximum}
        onDelta={(delta) => resizePanePreservingCamera("right", delta)}
        onReset={() => setWorkspaceLayout((current: WorkspaceLayoutState) => ({
          ...current,
          rightWidth: DEFAULT_WORKSPACE_LAYOUT.rightWidth,
          rightCollapsed: false,
        }))}
      /> : null}

      <aside
        className={`graph-column ${workspaceLayout.rightCollapsed ? "is-collapsed" : ""} ${workspaceLayout.summaryCollapsed || activeWorkspace !== "chat" ? "is-summary-collapsed" : ""} ${mobilePanel !== "graph" ? "is-mobile-hidden" : ""}`}
        aria-label="实时图谱与摘要"
      >
        {workspaceLayout.rightCollapsed ? (
          <div className="graph-column__collapsed-rail">
            <button type="button" onClick={() => setWorkspaceLayout((current: WorkspaceLayoutState) => ({ ...current, rightCollapsed: false }))} aria-label="展开图谱栏">
              <CaretLeft size={17} />
            </button>
            <span>{activeWorkspace === "governance" ? "治理图谱" : activeWorkspace === "adaptation" ? "适配图谱" : "实时图谱"}</span>
          </div>
        ) : null}
        <div className="graph-column__content" aria-hidden={workspaceLayout.rightCollapsed}>
          <div className="graph-pane">
            <Suspense fallback={<div className="graph-preview-loading"><CircleNotch size={22} className="spin" /><span>正在加载图谱引擎…</span></div>}>
              {governanceMounted || activeWorkspace === "adaptation" ? <div
                className="graph-workspace-surface"
                aria-hidden={activeWorkspace === "chat"}
                hidden={activeWorkspace === "chat"}
                inert={activeWorkspace === "chat"}
              >
              <GraphPreview
                graphVersion={operationalGraphVersion}
                selectedNodeId={activeWorkspace === "adaptation" ? activeAdaptationPresentation.focus?.kind === "node" ? activeAdaptationPresentation.focus.targetId : null : governanceGraphPresentation?.selectedNodeId ?? null}
                onSelectNode={activeWorkspace === "adaptation"
                  ? handleAdaptationGraphNodeSelect
                  : governanceGraphPresentation?.onSelectNode}
                theme={activeWorkspace === "adaptation" ? adaptationTheme : governanceTheme}
                focusNodeIds={activeWorkspace === "adaptation" ? [] : governanceGraphPresentation?.focusNodeIds ?? []}
                governanceFocus={activeWorkspace === "adaptation" ? activeAdaptationPresentation.focus : governanceGraphPresentation?.focus}
                activeOverlay={activeWorkspace === "adaptation" ? activeAdaptationPresentation.overlay : governanceGraphPresentation?.activeOverlay ?? null}
                filters={governanceFilters}
                onThemeChange={activeWorkspace === "adaptation" ? setAdaptationTheme : setGovernanceTheme}
                onFiltersChange={setGovernanceFilters}
                onViewStateChange={handleGovernanceViewSnapshot}
                cameraFocusCommand={activeWorkspace === "adaptation" && activeAdaptationPresentation.camera
                  ? {
                    ...activeAdaptationPresentation.camera,
                    commandScope: adaptationCameraLens(adaptationLanePresentation.activeLane),
                  }
                  : undefined}
                cameraRestoreCommand={workspaceCameraRestore?.workspace === activeWorkspace
                  ? workspaceCameraRestore
                  : undefined}
                title={activeWorkspace === "adaptation" ? "适配图谱" : "治理图谱"}
                headerAccessory={activeWorkspace === "adaptation" ? <AdaptationGraphSwitcher
                  state={adaptationLanePresentation}
                  onSelect={switchAdaptationGraph}
                /> : undefined}
                ariaLabel={activeWorkspace === "adaptation" ? "适配任务关系图" : "治理关系图"}
                emptyState={activeWorkspace === "adaptation" ? {
                  title: "等待目标域图谱",
                  description: "登记目标域任务包后，将在这里显示零样本与适配后图谱。",
                } : undefined}
                returnToOverviewAction={activeWorkspace === "adaptation" && activeAdaptationPresentation.focus ? {
                  label: "返回适配全图",
                  onReturn: restoreAdaptationOverview,
                } : undefined}
                labelLimit={12}
                showNodeRanks={activeWorkspace !== "adaptation"}
                summaryCollapsed
                isPaneVisible={activeWorkspace !== "chat" && graphPaneVisible}
              />
              </div> : null}
              <div
                className="graph-workspace-surface"
                aria-hidden={activeWorkspace !== "chat"}
                hidden={activeWorkspace !== "chat"}
                inert={activeWorkspace !== "chat"}
              >
              <GraphPreview
                graphVersion={graphVersion}
                scene={graphScene}
                selectedNodeId={selectedNode?.id ?? null}
                onSelectNode={handleGraphNodeSelect}
                viewMode={viewState.mode}
                depth={viewState.depth}
                theme={viewState.theme}
                layoutPreset={viewState.layoutPreset}
                rendererPreference={viewState.rendererPreference}
                focusNodeIds={viewState.focusNodeIds}
                pathEndpointIds={viewState.pathEndpointIds}
                pinnedNodes={viewState.pinnedNodes}
                activeOverlay={activeOverlay}
                filters={viewState.filters}
                onViewModeChange={handleGraphViewModeChange}
                onDepthChange={(depth: GraphViewState["depth"]) => patchGraphView({ depth })}
                onThemeChange={(theme: GraphViewState["theme"]) => patchGraphView({ theme })}
                onLayoutPresetChange={(layoutPreset: GraphViewState["layoutPreset"]) => patchGraphView({ layoutPreset })}
                onRendererPreferenceChange={(rendererPreference: GraphViewState["rendererPreference"]) => patchGraphView({ rendererPreference })}
                onFocusNodeIdsChange={handleFocusNodeIdsChange}
                onPathEndpointIdsChange={handlePathEndpointIdsChange}
                onPinnedNodesChange={(pinnedNodes: GraphViewState["pinnedNodes"]) => {
                  patchGraphView({ pinnedNodes });
                  if (graphVersion) {
                    void graphRepository.saveEvent(createSemanticEvent("node_pinned", {
                      graphVersionId: graphVersion.id,
                      sessionId: activeSessionId,
                      payload: { pinnedCount: Object.keys(pinnedNodes).length },
                    }));
                  }
                }}
                onFiltersChange={handleGraphFiltersChange}
                onViewStateChange={handleGraphViewSnapshot}
                cameraRestoreCommand={workspaceCameraRestore?.workspace === "chat"
                  ? workspaceCameraRestore
                  : undefined}
                returnToOverviewAction={chatLocateOverview ? {
                  label: "返回完整图",
                  onReturn: restoreChatLocateOverview,
                } : undefined}
                onExportReady={setGraphExportHandlers}
                onExport={handleGraphExport}
                enableMinimap={
                  import.meta.env.DEV
                  && new URLSearchParams(window.location.search).get("minimap") === "1"
                }
                summaryCollapsed={workspaceLayout.summaryCollapsed}
                summaryControlsId="graph-summary-panel"
                onSummaryCollapsedChange={(summaryCollapsed: NonNullable<GraphPreviewProps["summaryCollapsed"]>) =>
                  setWorkspaceLayout((current: WorkspaceLayoutState) => ({ ...current, summaryCollapsed }))}
                isPaneVisible={activeWorkspace === "chat" && graphPaneVisible}
              />
              </div>
            </Suspense>
          </div>
          {activeWorkspace === "chat" && !workspaceLayout.summaryCollapsed && resolvedWorkspaceLayout.graphResizerVisible ? (
            <>
              <WorkspaceResizeHandle
                className="workspace-resizer--graph"
                axis="horizontal"
                label="调整图谱画布高度"
                value={resolvedGraphHeight.height}
                minimum={resolvedGraphHeight.bounds.minimum}
                maximum={resolvedGraphHeight.bounds.maximum}
                onDelta={(delta) => setWorkspaceLayout((current: WorkspaceLayoutState) => {
                  const resolved = resolveWorkspaceGraphHeight(current, viewportHeight);
                  return resizeWorkspacePane(current, "graph", delta, {
                    currentValue: resolved.height,
                    bounds: resolved.bounds,
                    viewportHeight,
                  });
                })}
                onReset={() => setWorkspaceLayout((current: WorkspaceLayoutState) => ({ ...current, graphHeightRatio: DEFAULT_WORKSPACE_LAYOUT.graphHeightRatio }))}
              />
              <div id="graph-summary-panel" className="graph-summary-scroll">
                <RightSummary
                  graph={graphVersion}
                  selectedNode={selectedNode}
                  viewState={graphVersion ? viewState : null}
                  scene={graphScene}
                  onExport={graphExportHandlers ? () => void graphExportHandlers.exportPng() : undefined}
                />
              </div>
            </>
          ) : null}
        </div>
      </aside>

      {workspacePanel === "sessions" ? (
        <WorkspaceDrawer title="会话管理" description="活动会话与本地回收站都会持久化保存。" onClose={() => setWorkspacePanel(null)} wide>
          <section className="panel-section">
            <div className="panel-section__title"><strong>活动会话</strong><span>{sessions.length}</span></div>
            <div className="session-manager-list">
              {sessions.map((session: ResearchSession) => (
                <div className="session-manager-row" key={session.id}>
                  <span><strong>{session.title}</strong><small>更新于 {sessionTime(session.updatedAt)}</small></span>
                  <div className="panel-actions">
                    <button type="button" onClick={() => { switchSession(session.id); setWorkspacePanel(null); }}>打开</button>
                    <button type="button" onClick={() => beginRenameSession(session.id)}>重命名</button>
                    <button type="button" className="is-danger" onClick={() => void trashSession(session.id)}>回收</button>
                  </div>
                </div>
              ))}
            </div>
          </section>
          <section className="panel-section">
            <div className="panel-section__title"><strong>回收站</strong><span>{trashedSessions.length}</span></div>
            {trashedSessions.length ? (
              <div className="session-manager-list">
                {trashedSessions.map((session: ResearchSession) => (
                  <div className="session-manager-row" key={session.id}>
                    <span><strong>{session.title}</strong><small>移入时间 {session.deletedAt ? sessionTime(session.deletedAt) : "本地"}</small></span>
                    <div className="panel-actions">
                      <button type="button" onClick={() => void restoreSession(session.id)}>恢复</button>
                      <button type="button" className="is-danger" onClick={() => void purgeSession(session.id)}>永久删除</button>
                    </div>
                  </div>
                ))}
              </div>
            ) : <p>回收站为空。永久删除会话只清理会话、消息和会话事件；图版本与源文件需在“数据集管理”中独立回收。</p>}
          </section>
        </WorkspaceDrawer>
      ) : null}

      {workspacePanel === "datasets" ? (
        <WorkspaceDrawer
          title="数据管理"
          description="管理图数据、推理包与可恢复记录。"
          onClose={() => {
            window.history.replaceState(null, "", hashForWorkspaceRoute(routeForSidebarWorkspace(activeWorkspace)));
            setWorkspacePanel(null);
          }}
          wide
        >
          <ResearchDatasetPanel
            key={datasetPanelEpoch}
            client={researchDatasetClient}
            onOpenArtifact={registerDatasetArtifact}
            onOpenInferencePackages={() => {
              window.history.replaceState(null, "", hashForWorkspaceRoute("governance"));
              setWorkspacePanel(null);
              setGovernanceMounted(true);
              setActiveWorkspace("governance");
              setMobilePanel("governance");
            }}
            onNotify={showToast}
          />
          <details className="dataset-outer-admin">
            <summary>版本与数据交接</summary>
            <section className="panel-section target-domain-handoff" aria-labelledby="target-domain-handoff-title">
              <div className="panel-section__title">
                <strong id="target-domain-handoff-title">当前图数据交接</strong>
                <span>仅显式执行</span>
              </div>
              <p>用户图只作为目标域事实数据交接，不会自动上传、成为预训练语料或更新模型权重。</p>
              <div className="diagnostic-grid">
                <div className="diagnostic-row"><strong>当前版本</strong><code>{graphVersion?.id ?? "尚未选择"}</code></div>
                <div className="diagnostic-row"><strong>浏览器存储</strong><span>
                  {storageCapacity.persisted ? "已授予持久存储" : "可能受浏览器空间回收策略影响"}
                  {storageCapacity.usage !== undefined && storageCapacity.quota
                    ? ` · ${Math.round(storageCapacity.usage / 1_048_576)} / ${Math.round(storageCapacity.quota / 1_048_576)} MB`
                    : ""}
                </span></div>
              </div>
              <div className="panel-actions">
                <button
                  type="button"
                  disabled={!graphVersion || targetDomainBusy || Boolean(graphVersion.datasetArtifact)}
                  onClick={() => void prepareCurrentGraphAsTargetDomain()}
                >
                  {targetDomainBusy ? <CircleNotch size={15} className="spin" /> : <ShieldCheck size={15} />}
                  建立目标域数据
                </button>
              </div>
            </section>
            <VersionLifecyclePanel
              repository={graphRepository}
              currentGraphVersionId={graphVersion?.id}
              onActivateVersion={restoreGraphVersion}
              onRequestRebuild={requestGraphVersionRebuild}
              onNotify={showToast}
            />
          </details>
        </WorkspaceDrawer>
      ) : null}

      {workspacePanel === "rename" ? (
        <WorkspaceDrawer title="重命名会话" onClose={() => setWorkspacePanel(null)}>
          <form className="rename-form" onSubmit={(event) => { event.preventDefault(); void submitRenameSession(); }}>
            <label htmlFor="session-title">会话名称</label>
            <input id="session-title" value={renameDraft} maxLength={80} autoFocus onChange={(event) => setRenameDraft(event.target.value)} />
            <div className="panel-actions">
              <button type="button" onClick={() => setWorkspacePanel(null)}>取消</button>
              <button type="submit" disabled={!renameDraft.trim()}>保存</button>
            </div>
          </form>
        </WorkspaceDrawer>
      ) : null}

      {workspacePanel === "guide" ? (
        <WorkspaceDrawer title="使用指南" description="上传、分析、复核，三步完成图谱研究。" onClose={() => setWorkspacePanel(null)}>
          <CoreUsageGuide />
        </WorkspaceDrawer>
      ) : null}

      {workspacePanel === "diagnostics" ? (
        <WorkspaceDrawer title="系统诊断" description="这里只展示脱敏元数据，不显示 API Key 或完整图明细。" onClose={() => setWorkspacePanel(null)}>
          <section className="panel-section">
            <div className="panel-section__title"><strong>服务状态</strong></div>
            <div className="diagnostic-grid">
              <div className="diagnostic-row"><strong>LLM 配置</strong><span>{intentServiceStatus.label}</span></div>
              <div className="diagnostic-row"><strong>本次调用</strong><span className={`diagnostic-status is-${llmDiagnostic.state}`}>{llmDiagnostic.state === "success" ? "LLM 调用成功" : llmDiagnostic.state === "running" ? "测试中" : llmDiagnostic.state === "error" ? "调用失败" : "尚未测试"}</span></div>
              <div className="diagnostic-row"><strong>SocialGraph-FM Governance</strong><span>{governanceStatusDescription}</span></div>
            </div>
          </section>
          <section className="panel-section">
            <div className="panel-section__title"><strong>最近一次元数据</strong></div>
            <div className="diagnostic-grid">
              <div className="diagnostic-row"><strong>Schema</strong><code>{llmDiagnostic.schemaVersion ?? "—"}</code></div>
              <div className="diagnostic-row"><strong>requestId</strong><code>{llmDiagnostic.requestId ?? "—"}</code></div>
              <div className="diagnostic-row"><strong>model</strong><code>{llmDiagnostic.model ?? "未返回"}</code></div>
              <div className="diagnostic-row"><strong>source / task</strong><code>{llmDiagnostic.source ? `${llmDiagnostic.source}${llmDiagnostic.task ? ` / ${llmDiagnostic.task}` : ""}` : "—"}</code></div>
              <div className="diagnostic-row"><strong>耗时</strong><code>{llmDiagnostic.latencyMs === undefined ? "—" : `${llmDiagnostic.latencyMs} ms`}</code></div>
              <div className="diagnostic-row"><strong>警告</strong><code>{llmDiagnostic.warnings?.length ? llmDiagnostic.warnings.join(", ") : llmDiagnostic.message ?? "无"}</code></div>
            </div>
            <div className="panel-actions">
              <button type="button" onClick={() => void runLlmDiagnostic()} disabled={llmDiagnostic.state === "running"}>重新测试 LLM</button>
            </div>
          </section>
          <section className="panel-section">
            <div className="panel-section__title"><strong>脱敏请求边界</strong></div>
            <p>普通问答仅发送问题文本与图级聚合指标。只有用户主动生成证据摘要或研判报告时，才会发送受控只读分析中的伪名账号端点、关系模态、权重与绑定指纹；不会发送原帖、发布时间、采集来源、源文件、完整图、向量或未授权属性。</p>
          </section>
          <section className="panel-section">
            <div className="panel-section__title"><strong>版本与源文件</strong></div>
            <p>图谱版本对比、兼容重建、引用预演、回收站与永久删除已迁移到“数据管理”，避免从诊断页绕过引用检查。</p>
            <div className="panel-actions">
              <button type="button" onClick={() => setWorkspacePanel("datasets")}>打开数据集管理</button>
            </div>
          </section>
        </WorkspaceDrawer>
      ) : null}

      {toast ? <div className="toast" role="status"><CheckCircle size={17} weight="fill" />{toast}</div> : null}
    </div>;
}
