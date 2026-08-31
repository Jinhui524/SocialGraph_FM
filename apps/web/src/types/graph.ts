export const MAX_IMPORT_BYTES = 20 * 1024 * 1024;
export const MAX_PREVIEW_NODES = 300;
export const MAX_PREVIEW_EDGES = 1_000;
/** Workbench scene limits. The immutable graph can be substantially larger. */
export const MAX_VISIBLE_NODES = 3_000;
export const MAX_VISIBLE_EDGES = 12_000;

export type JsonPrimitive = string | number | boolean | null;
export type GraphAttributeValue = JsonPrimitive | readonly JsonPrimitive[];
export type GraphAttributes = Readonly<Record<string, GraphAttributeValue>>;

export interface GraphNode {
  readonly id: string;
  readonly label: string;
  readonly type?: string;
  readonly attributes: GraphAttributes;
}

export interface GraphEdge {
  readonly id: string;
  readonly source: string;
  readonly target: string;
  readonly type?: string;
  readonly weight?: number;
  readonly timestamp?: string;
  readonly directed?: boolean;
  readonly attributes: GraphAttributes;
}

export type ValidationSeverity = "info" | "warning" | "error";

export interface ValidationIssue {
  readonly code: string;
  readonly severity: ValidationSeverity;
  readonly message: string;
  readonly row?: number;
  readonly entityId?: string;
  readonly details?: Readonly<Record<string, string | number | boolean>>;
}

export interface GraphSummary {
  readonly nodeCount: number;
  readonly edgeCount: number;
  /** Undirected density 2m / n(n-1); multigraph data can therefore exceed 1. */
  readonly density: number;
  /** Mean incident-edge degree; parallel relationships are counted. */
  readonly averageDegree: number;
  readonly connectedComponents: number;
  readonly isolatedNodes: number;
}

export interface GraphPreview {
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
  readonly truncated: boolean;
  readonly originalNodeCount: number;
  readonly originalEdgeCount: number;
}

/** Browser-local immutable copy of an uploaded source file. */
export interface SourceArtifact {
  readonly id: string;
  readonly sha256: string;
  readonly name: string;
  readonly size: number;
  readonly mimeType: string;
  readonly format: ImportFormat;
  readonly role: "single" | "nodes" | "edges";
  readonly createdAt: string;
  readonly blob: Blob;
}

export type GraphInputShape = "standard_graph" | "edge_table" | "node_edge_tables";
export type EdgeDirectionPolicy = "file" | "directed" | "undirected";
export type DuplicateEdgePolicy = "preserve" | "merge_sum" | "reject";
export type SelfLoopPolicy = "preserve" | "reject";
export type DanglingEndpointPolicy = "derive_nodes" | "reject";
export type GraphTimeFormat =
  | "none"
  | "auto"
  | "iso8601"
  | "year"
  | "unix_seconds"
  | "unix_milliseconds";

export interface NodeColumnMapping {
  readonly id: string;
  readonly label?: string;
  readonly type?: string;
}

/** A validated, deterministic recipe. LLM output is never executed directly. */
export interface GraphBuildSpec {
  readonly schemaVersion: "1.0";
  readonly inputShape: GraphInputShape;
  readonly sourceArtifactIds: readonly string[];
  readonly nodeMapping?: NodeColumnMapping;
  readonly edgeMapping?: ColumnMapping;
  readonly directionPolicy: EdgeDirectionPolicy;
  readonly duplicateEdgePolicy: DuplicateEdgePolicy;
  readonly selfLoopPolicy: SelfLoopPolicy;
  readonly danglingEndpointPolicy: DanglingEndpointPolicy;
  readonly timeFormat: GraphTimeFormat;
  readonly description?: string;
}

/** Auditable producer identity. Missing provenance means a read-only legacy version. */
export interface GraphVersionProvenance {
  readonly origin: "browser_import" | "research_dataset" | "seed_demo";
  readonly pipeline: "browser-import" | "dataset-artifact" | "demo";
  readonly pipelineVersion: string;
  readonly buildSpecSchemaVersion?: GraphBuildSpec["schemaVersion"];
  readonly sourceHashScheme?: "artifact-sha256-list-v1" | "dataset-content-hash-v2";
  readonly reconstructionReason?: "pipeline_upgrade" | "construction_revision";
}

export interface GraphVersion {
  readonly id: string;
  readonly sourceFile: string;
  readonly createdAt: string;
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
  readonly summary: GraphSummary;
  readonly issues: readonly ValidationIssue[];
  readonly preview: GraphPreview;
  /** Convenience flag mirrored from preview.truncated for UI status badges. */
  readonly truncated: boolean;
  /** Present on versions created by the v2 deterministic import pipeline. */
  readonly sourceArtifactIds?: readonly string[];
  readonly sourceHash?: string;
  readonly buildSpecHash?: string;
  readonly contentHash?: string;
  readonly parentVersionId?: string;
  readonly buildSpec?: GraphBuildSpec;
  readonly provenance?: GraphVersionProvenance;
  readonly metadata?: {
    readonly directedness: "directed" | "undirected" | "mixed" | "unspecified";
  };
  /** Present when the visible graph is a projection of a persisted research artifact. */
  readonly datasetArtifact?: {
    readonly id: string;
    readonly datasetName: string;
    readonly checksum: string;
    readonly canonicalGraphHash: string;
    readonly contentHash?: string;
    readonly trainingRefHash?: string;
    readonly datasetRole?: "benchmark" | "target_domain" | "pretraining_candidate";
    readonly scope: "complete" | "projection";
  };
}

/** Lightweight immutable index record; history UIs must not load full graph facts. */
export interface GraphVersionManifest {
  readonly id: string;
  readonly sourceFile: string;
  readonly createdAt: string;
  readonly parentVersionId?: string;
  readonly sourceArtifactIds: readonly string[];
  readonly sourceHash?: string;
  readonly buildSpecHash?: string;
  readonly contentHash?: string;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly directedness: "directed" | "undirected" | "mixed" | "unspecified";
  readonly datasetArtifactId?: string;
  readonly provenance?: GraphVersionProvenance;
}

export type ManagedResourceKind = "graph_version" | "source_artifact";
export type ResourceLifecycleState = "active" | "trashed";
export type ResourceListState = ResourceLifecycleState | "all";

/** Mutable lifecycle is deliberately stored outside immutable facts and blobs. */
export interface ResourceLifecycle {
  readonly key: string;
  readonly kind: ManagedResourceKind;
  readonly targetId: string;
  readonly state: ResourceLifecycleState;
  readonly updatedAt: string;
  readonly trashedAt?: string;
}

export type GraphEntityChangeKind = "added" | "removed" | "modified";

export interface GraphFieldDiff {
  readonly field: string;
  readonly before?: unknown;
  readonly after?: unknown;
}

export interface GraphEntityDiff {
  readonly entity: "node" | "edge";
  readonly id: string;
  readonly kind: GraphEntityChangeKind;
  readonly fields: readonly GraphFieldDiff[];
}

export interface GraphVersionDiffCount {
  readonly added: number;
  readonly removed: number;
  readonly modified: number;
}

export interface GraphEdgeIdChurnSample {
  readonly beforeId: string;
  readonly afterId: string;
  readonly structuralKey: string;
}

export interface GraphEdgeIdChurn {
  readonly count: number;
  readonly samples: readonly GraphEdgeIdChurnSample[];
  readonly truncated: boolean;
}

export interface GraphVersionDiffReport {
  readonly fromVersionId: string;
  readonly toVersionId: string;
  readonly fromContentHash: string;
  readonly toContentHash: string;
  readonly fromHashSource: "stored" | "derived";
  readonly toHashSource: "stored" | "derived";
  readonly sameContent: boolean;
  readonly sameLineage: boolean;
  readonly summary: {
    readonly nodes: GraphVersionDiffCount;
    readonly edges: GraphVersionDiffCount;
  };
  /** Internal edge ids changed while the structural fact multiset stayed aligned. */
  readonly edgeIdChurn: GraphEdgeIdChurn;
  readonly versionFields: readonly GraphFieldDiff[];
  readonly buildSpecFields: readonly GraphFieldDiff[];
  /** Deterministically sorted, bounded examples. Exact totals live in summary. */
  readonly samples: readonly GraphEntityDiff[];
  readonly sampleLimit: number;
  readonly truncated: boolean;
}

export type GraphViewMode = "global" | "local" | "path";
export type GraphInteractionTool =
  | "browse"
  | "pick_local_focus"
  | "pick_path_start"
  | "pick_path_end";
export type GraphTheme = "brand-light" | "focus-dark";
export type LayoutPreset = "balanced" | "compact" | "spread";
export type GraphRendererPreference = "auto" | "canvas" | "hybrid-webgl";
export type GraphRendererKind = "canvas" | "hybrid-webgl";

export interface GraphRendererStatus {
  readonly requested: GraphRendererPreference;
  readonly resolved: GraphRendererKind;
  readonly webglSupported: boolean;
  readonly fallbackReason?: string;
  readonly lazyLoadMs?: number;
  readonly contextLossCount: number;
}
export type AnalysisOverlayKind =
  | "raw"
  | "degree"
  | "articulation"
  | "components"
  | "community"
  | "path"
  | "governance";

export interface GraphTimeRange {
  readonly start?: string;
  readonly end?: string;
}

export interface GraphFilters {
  readonly nodeTypes: readonly string[];
  readonly edgeTypes: readonly string[];
  readonly timeRange?: GraphTimeRange;
  readonly minWeight?: number;
  readonly maxWeight?: number;
  readonly directed?: boolean;
  /** Internal fail-closed marker; never sent to an LLM. */
  readonly emptyReason?:
    | "direction_mismatch"
    | "direction_unknown"
    | "invalid_weight_range"
    | "invalid_time_range"
    | "manual_empty_selection";
}

/** A constrained, non-mutating instruction produced from natural language. */
export interface ViewCommand {
  readonly mode?: GraphViewMode;
  readonly focusTerms: readonly string[];
  readonly depth?: 1 | 2 | 3;
  readonly nodeTypeTerms: readonly string[];
  readonly edgeTypeTerms: readonly string[];
  readonly timeRange?: GraphTimeRange;
  readonly layoutPreset?: LayoutPreset;
  readonly overlay?: Exclude<AnalysisOverlayKind, "raw" | "path" | "governance">;
}

export type TargetMatchKind = "id_exact" | "normalized_exact" | "unique_substring";

export type TargetResolution =
  | {
      readonly status: "resolved";
      readonly term: string;
      readonly nodeId: string;
      readonly match: TargetMatchKind;
    }
  | {
      readonly status: "ambiguous";
      readonly term: string;
      readonly candidateNodeIds: readonly string[];
    }
  | {
      readonly status: "not_found";
      readonly term: string;
    };

export interface CameraState {
  readonly x: number;
  readonly y: number;
  readonly zoom: number;
}

export interface GraphViewState {
  readonly graphVersionId: string;
  readonly mode: GraphViewMode;
  readonly focusNodeIds: readonly string[];
  /** Path endpoints are independent from local-view focus nodes. */
  readonly pathEndpointIds: readonly string[];
  readonly depth: 1 | 2 | 3;
  readonly filters: GraphFilters;
  readonly theme: GraphTheme;
  readonly layoutPreset: LayoutPreset;
  readonly rendererPreference: GraphRendererPreference;
  readonly camera: CameraState;
  readonly pinnedNodes: Readonly<Record<string, { readonly x: number; readonly y: number }>>;
}

export interface GraphInteractionState {
  readonly tool: GraphInteractionTool;
  readonly selectedNodeId: string | null;
  readonly pendingPathStartId?: string;
}

export interface GraphWorkbenchViewState {
  readonly viewState: GraphViewState;
  readonly interaction: GraphInteractionState;
}

export type GraphSceneStatus =
  | { readonly kind: "ready" }
  | { readonly kind: "awaiting_focus" }
  | { readonly kind: "awaiting_path_end"; readonly sourceId: string }
  | { readonly kind: "no_path"; readonly sourceId: string; readonly targetId: string };

export interface ApplyViewCommandResult {
  readonly nextState: GraphViewState;
  readonly targetResolutions: readonly TargetResolution[];
  readonly warnings: readonly string[];
  readonly requestedOverlay?: Exclude<AnalysisOverlayKind, "raw" | "path" | "governance">;
}

export interface OverlayLegendItem {
  readonly value: string;
  readonly label: string;
  readonly color?: string;
}

export interface OverlayLegend {
  readonly title: string;
  readonly items: readonly OverlayLegendItem[];
}

export interface AnalysisOverlay {
  readonly id: string;
  readonly graphVersionId: string;
  readonly kind: AnalysisOverlayKind;
  readonly nodeValues: Readonly<Record<string, string | number | boolean>>;
  readonly edgeValues: Readonly<Record<string, string | number | boolean>>;
  /** Presentation-only candidate relations. They are never GraphVersion facts. */
  readonly candidateEdges?: readonly {
    readonly id: string;
    readonly sourceId: string;
    readonly targetId: string;
    readonly directed: boolean;
    readonly exactRelationKey?: string;
  }[];
  /** Optional renderer hints. Older overlays remain valid without them. */
  readonly presentation?: {
    readonly governanceLens?: "risk" | "relations";
    readonly riskBands?: Readonly<Record<string, "high" | "review" | "low">>;
    /** Immutable server comparison markers; negative values moved toward the
     * front of the adapted review queue. */
    readonly rankDeltas?: Readonly<Record<string, number>>;
    /** Exact immutable server rank for an adapted comparison view. */
    readonly adaptedRanks?: Readonly<Record<string, number>>;
    /** Imported target-domain labels. These are input evidence, not model risk bands. */
    readonly referenceLabels?: Readonly<Record<string, "positive" | "negative">>;
    /** Current-case human decisions. These never mutate model findings or graph facts. */
    readonly reviewDecisions?: Readonly<Record<string, "confirmed" | "rejected" | "pending">>;
  };
  readonly legend: OverlayLegend;
  readonly provenance: {
    readonly engine: string;
    readonly algorithm: string;
    readonly runId?: string;
    readonly scopeHash?: string;
    readonly resultHash?: string;
    readonly findingHash?: string;
    readonly publicRequestHash?: string;
    readonly serverRequestHash?: string;
    readonly taskId?: string;
    readonly graphVersionHash?: string;
    readonly modelVersionId?: string;
    readonly modelVersionHash?: string;
  };
}

/** One atomic, presentation-only governance selection. The selected evidence
 * object is stored separately so clearing this focus never discards review
 * context. */
export interface GovernanceFocus {
  readonly kind: "node" | "group" | "relation";
  readonly targetId: string;
  readonly nodeIds: readonly string[];
  readonly exactRelationKey?: string;
  readonly cameraToken: number;
}

export interface GraphSlice {
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
  readonly nodeIds: readonly string[];
  readonly edgeIds?: readonly string[];
}

export interface AnalysisScope {
  readonly graphVersionId: string;
  readonly nodeIds: readonly string[];
  readonly edgeIds: readonly string[];
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly scopeHash: string;
  readonly truncated: boolean;
  readonly filters: GraphFilters;
}

export interface ScopedGraphSlice {
  readonly scope: AnalysisScope;
  readonly slice: GraphSlice;
}

export interface GraphPath extends GraphSlice {
  readonly edgeIds: readonly string[];
  readonly sourceId: string;
  readonly targetId: string;
}

export interface GraphScene {
  readonly graphVersionId: string;
  readonly nodes: readonly GraphNode[];
  readonly edges: readonly GraphEdge[];
  readonly focusNodeIds: readonly string[];
  readonly pathEndpointIds: readonly string[];
  readonly pathNodeIds: readonly string[];
  readonly pathEdgeIds: readonly string[];
  readonly status: GraphSceneStatus;
  readonly overlay?: AnalysisOverlay;
  readonly truncated: boolean;
  readonly originalNodeCount: number;
  readonly originalEdgeCount: number;
  readonly visibleNodeCount: number;
  readonly visibleEdgeCount: number;
  /** Identity of the bounded render projection, separate from an analysis scope. */
  readonly projectionHash: string;
}

export interface ColumnMapping {
  readonly source: string;
  readonly target: string;
  readonly sourceLabel?: string;
  readonly targetLabel?: string;
  readonly sourceType?: string;
  readonly targetType?: string;
  readonly edgeType?: string;
  readonly weight?: string;
  readonly timestamp?: string;
}

export type ImportFormat = "csv" | "tsv" | "json" | "graphml" | "gexf" | "npz" | "unsupported";

export interface FileProfile {
  readonly name: string;
  readonly size: number;
  readonly format: ImportFormat;
  readonly supported: boolean;
  readonly headers: readonly string[];
  readonly columns?: readonly FileColumnProfile[];
  readonly suggestedMapping?: Partial<ColumnMapping>;
  readonly needsMapping: boolean;
  readonly issues: readonly ValidationIssue[];
}

export type InferredColumnType = "empty" | "boolean" | "integer" | "number" | "datetime" | "string";

/** Privacy-safe schema profile: contains no source values or sample rows. */
export interface FileColumnProfile {
  readonly name: string;
  readonly inferredType: InferredColumnType;
  readonly missingRate: number;
  readonly cardinality: number;
  readonly nonNullCount: number;
  readonly nullCount: number;
}

export type ImportStatus = "ready" | "needs_mapping" | "failed";

export type ImportMappingField =
  | "node.id"
  | "edge.source"
  | "edge.target";

/** A file-scoped mapping request. It never contains source rows or values. */
export interface ImportMappingRequest {
  readonly nodeTable?: {
    readonly artifactId?: string;
    readonly headers: readonly string[];
    readonly suggestedMapping: Partial<NodeColumnMapping>;
  };
  readonly edgeTable: {
    readonly artifactId?: string;
    readonly headers: readonly string[];
    readonly suggestedMapping: Partial<ColumnMapping>;
  };
  readonly missingFields: readonly ImportMappingField[];
}

export interface ImportRun {
  readonly status: ImportStatus;
  readonly headers?: readonly string[];
  readonly suggestedMapping?: Partial<ColumnMapping>;
  readonly mappingRequest?: ImportMappingRequest;
  readonly graphVersion?: GraphVersion;
  readonly issues: readonly ValidationIssue[];
  readonly error?: string;
}

export interface GraphImportParseOptions {
  readonly buildSpec?: GraphBuildSpec;
  readonly sourceArtifacts?: readonly SourceArtifact[];
  readonly parentVersionId?: string;
  readonly provenance?: GraphVersionProvenance;
}

export interface GraphImportAdapter {
  inspect(file: File): Promise<FileProfile>;
  parse(file: File, mapping?: ColumnMapping, options?: GraphImportParseOptions): Promise<ImportRun>;
  parseFiles(
    files: readonly File[],
    buildSpec: GraphBuildSpec,
    options?: GraphImportParseOptions,
  ): Promise<ImportRun>;
}

export type AnalysisTask =
  | "overview"
  | "centrality"
  | "bridge_detection"
  | "community"
  | "link_prediction"
  | "node_role"
  | "similar_structure";

export interface GraphContextSummary {
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly density: number;
  readonly connectedComponents: number;
  readonly nodeTypes: readonly string[];
  readonly edgeTypes: readonly string[];
  readonly hasWeight: boolean;
  readonly hasTimestamp: boolean;
  readonly timeRange?: {
    readonly start?: string;
    readonly end?: string;
  };
}

export interface NormalizeIntentInput {
  readonly text: string;
  readonly graphContext?: GraphContextSummary;
}

export interface GraphBuildIntentInput {
  readonly description: string;
  readonly requestToken: string;
  readonly baseGraphVersionId?: string;
  readonly files: readonly {
    readonly artifactId: string;
    readonly role: SourceArtifact["role"];
    readonly format: ImportFormat;
    readonly columns: readonly FileColumnProfile[];
  }[];
  readonly allowedPolicies: {
    readonly direction: readonly EdgeDirectionPolicy[];
    readonly duplicateEdges: readonly DuplicateEdgePolicy[];
    readonly selfLoops: readonly SelfLoopPolicy[];
    readonly danglingEndpoints: readonly DanglingEndpointPolicy[];
    readonly timeFormats: readonly GraphTimeFormat[];
  };
}

export interface GraphBuildIntentResult {
  readonly kind: "construction_revision";
  readonly requestToken: string;
  readonly baseGraphVersionId?: string;
  readonly spec: GraphBuildSpec;
  readonly source: "llm";
  readonly warnings: readonly string[];
}

export interface GraphBuildIntentNormalizer {
  normalizeGraphBuildIntent(input: GraphBuildIntentInput): Promise<GraphBuildIntentResult>;
}

/** Kept as an alias while integrations migrate to NormalizeIntentInput. */
export type ChatIntentInput = NormalizeIntentInput;

export interface IntentMeta {
  readonly schemaVersion: "1.0" | "1.1";
  readonly source: "llm";
  readonly requestId: string;
  readonly model?: string;
  readonly warnings: readonly string[];
}

export interface ChatIntentResult {
  readonly kind: "chat";
  readonly reply: string;
  readonly meta: IntentMeta;
}

export interface AnalysisIntentResult {
  readonly kind: "analysis_request";
  readonly normalizedText: string;
  readonly task: AnalysisTask;
  readonly targets: readonly string[];
  readonly confidence: number;
  readonly timeRange?: {
    readonly start?: string;
    readonly end?: string;
  };
  readonly filters: Readonly<Record<string, string | number | boolean>>;
  readonly view?: ViewCommand;
  readonly meta: IntentMeta;
}

export type IntentNormalizationResult = ChatIntentResult | AnalysisIntentResult;

/** Analysis runs only accept an analysis intent, never a conversational reply. */
export type NormalizedIntent = AnalysisIntentResult;

export interface DegreeRankEntry {
  readonly nodeId: string;
  readonly label: string;
  readonly degree: number;
  readonly normalizedScore: number;
}

export type AnalysisResult =
  | {
      readonly kind: "overview";
      readonly summary: GraphSummary;
      readonly topDegree: readonly DegreeRankEntry[];
      readonly articulationPoints: readonly string[];
    }
  | {
      readonly kind: "centrality";
      readonly ranking: readonly DegreeRankEntry[];
    }
  | {
      readonly kind: "connected_components";
      readonly components: readonly (readonly string[])[];
    }
  | {
      readonly kind: "community";
      readonly communities: readonly (readonly string[])[];
      readonly assignments: Readonly<Record<string, string>>;
      readonly modularity: number;
      readonly message: string;
    }
  | {
      readonly kind: "bridge_detection";
      readonly articulationPoints: readonly string[];
    }
  | {
      readonly kind: "unavailable";
      readonly code: "GFM_CORE_NOT_CONNECTED" | "DATASET_PROJECTION_REQUIRES_SERVER_ANALYSIS";
      readonly message: string;
      readonly requestedTask: AnalysisTask;
    };

export interface CreateAnalysisInput {
  readonly graphVersionId: string;
  readonly intent: NormalizedIntent;
  readonly scopedGraph?: ScopedGraphSlice;
  /** Lets the local mock register a freshly imported graph without a backend store. */
  readonly graphVersion?: GraphVersion;
}

export interface AnalysisRun {
  readonly id: string;
  readonly graphVersionId: string;
  readonly intent: NormalizedIntent;
  readonly engine: "local_algorithm" | "gfm" | "unavailable";
  readonly status: "queued" | "running" | "succeeded" | "failed";
  readonly createdAt: string;
  readonly completedAt?: string;
  readonly result?: AnalysisResult;
  readonly scope?: AnalysisScope;
  readonly error?: string;
}

export interface IntentNormalizer {
  normalizeIntent(input: NormalizeIntentInput): Promise<IntentNormalizationResult>;
}

export interface AnalysisExecutor {
  registerGraphVersion(version: GraphVersion): void;
  createAnalysis(input: CreateAnalysisInput): Promise<AnalysisRun>;
  getAnalysis(runId: string): Promise<AnalysisRun>;
}

/** @deprecated Prefer separate IntentNormalizer and AnalysisExecutor dependencies. */
export interface SocialGraphApi extends IntentNormalizer, AnalysisExecutor {}

export interface ResearchSession {
  readonly id: string;
  readonly title: string;
  readonly graphVersionId?: string;
  readonly lifecycle: "active" | "trashed";
  readonly deletedAt?: string;
  readonly updatedAt: string;
}

export interface ConversationMessage {
  readonly id: string;
  readonly sessionId: string;
  readonly role: "user" | "assistant";
  readonly text: string;
  readonly status: "pending" | "completed" | "warning" | "failed" | "interrupted";
  readonly intent?: IntentNormalizationResult;
  readonly analysisRunId?: string;
  /** Links a completed assistant report to its governance review run. */
  readonly governanceRunId?: string;
  /** Indexed copy of attachment references for deletion-impact checks. */
  readonly sourceArtifactIds?: readonly string[];
  readonly attachment?: {
    readonly name: string;
    readonly size: number;
    readonly kind: "file";
  };
  readonly attachments?: readonly {
    readonly name: string;
    readonly size: number;
    readonly kind: "file";
    readonly sourceArtifactId?: string;
  }[];
  readonly createdAt: string;
}

/** Tracks one-time repository initialization without coupling it to demo sessions. */
export interface RepositoryInitializationMetadata {
  readonly initializedAt: string;
  readonly seededDemoVersion: number;
  readonly updatedAt: string;
}

export interface GraphImportPersistenceBundle {
  readonly sourceArtifacts: readonly SourceArtifact[];
  readonly graphVersion: GraphVersion;
  readonly viewState: GraphViewState;
  readonly session: ResearchSession;
  readonly event: SemanticEvent;
  /** When present, attachment references are linked in the same import transaction. */
  readonly sourceMessageId?: string;
}

export type SessionListState = "active" | "trashed" | "all";

export type SemanticEventType =
  | "graph_imported"
  | "intent_applied"
  | "view_changed"
  | "analysis_completed"
  | "local_review_recorded"
  | "node_pinned"
  | "view_saved";

export interface SemanticEvent {
  readonly id: string;
  readonly type: SemanticEventType;
  readonly createdAt: string;
  readonly graphVersionId?: string;
  readonly sessionId?: string;
  readonly payload: Readonly<Record<string, JsonPrimitive>>;
}

export type DeletionReferenceKind =
  | "active_session"
  | "trashed_session"
  | "active_child_version"
  | "trashed_child_version"
  | "analysis_message"
  | "graph_version"
  | "source_message"
  | "training_dataset_ref";

export interface DeletionReference {
  readonly kind: DeletionReferenceKind;
  readonly id: string;
  readonly label: string;
  readonly blocksTrash: boolean;
  readonly blocksPurge: boolean;
}

export type DeletionDependentKind =
  | "view_state"
  | "analysis_run"
  | "semantic_event"
  | "manifest"
  | "lifecycle";

export interface DeletionDependentGroup {
  readonly kind: DeletionDependentKind;
  readonly count: number;
  readonly ids: readonly string[];
}

export interface RetainedDependency {
  readonly kind: "source_artifact" | "dataset_artifact";
  readonly id: string;
  readonly label: string;
}

export interface DeletionImpact {
  readonly targetKind: ManagedResourceKind;
  readonly targetId: string;
  readonly targetLabel: string;
  readonly state: ResourceLifecycleState;
  readonly references: readonly DeletionReference[];
  readonly dependents: readonly DeletionDependentGroup[];
  readonly retainedDependencies: readonly RetainedDependency[];
  readonly canTrash: boolean;
  readonly canPurge: boolean;
  /** Optimistic concurrency token over the complete sorted reference set. */
  readonly impactHash: string;
}

export type RepositoryChangeKind =
  | "graph_saved"
  | "graph_trashed"
  | "graph_restored"
  | "graph_purged"
  | "source_saved"
  | "source_trashed"
  | "source_restored"
  | "source_purged"
  | "semantic_event_appended"
  | "session_changed"
  | "message_changed";

export interface RepositoryChange {
  readonly kind: RepositoryChangeKind;
  readonly ids: readonly string[];
  readonly createdAt: string;
  readonly originId: string;
}

export interface GraphRepository {
  readonly storageMode: "indexeddb" | "memory";
  saveSourceArtifact(artifact: SourceArtifact): Promise<void>;
  getSourceArtifact(id: string): Promise<SourceArtifact | undefined>;
  listSourceArtifacts(state?: ResourceListState): Promise<readonly SourceArtifact[]>;
  deleteSourceArtifact(id: string): Promise<void>;
  inspectSourceArtifactDeletion(id: string): Promise<DeletionImpact>;
  trashSourceArtifact(id: string, expectedImpactHash?: string): Promise<void>;
  restoreSourceArtifact(id: string): Promise<void>;
  purgeSourceArtifact(id: string, expectedImpactHash?: string): Promise<void>;
  saveImportBundle(bundle: GraphImportPersistenceBundle, guard?: () => boolean): Promise<void>;
  saveGraphVersion(version: GraphVersion): Promise<void>;
  getGraphVersion(id: string): Promise<GraphVersion | undefined>;
  listGraphVersions(state?: ResourceListState): Promise<readonly GraphVersion[]>;
  getGraphVersionManifest(id: string): Promise<GraphVersionManifest | undefined>;
  listGraphVersionManifests(state?: ResourceListState): Promise<readonly GraphVersionManifest[]>;
  inspectGraphVersionDeletion(id: string): Promise<DeletionImpact>;
  trashGraphVersion(id: string, expectedImpactHash?: string): Promise<void>;
  restoreGraphVersion(id: string): Promise<void>;
  purgeGraphVersion(id: string, expectedImpactHash?: string): Promise<void>;
  getResourceLifecycle(kind: ManagedResourceKind, id: string): Promise<ResourceLifecycle>;
  saveViewState(state: GraphViewState): Promise<void>;
  getViewState(graphVersionId: string): Promise<GraphViewState | undefined>;
  saveAnalysisRun(run: AnalysisRun): Promise<void>;
  getAnalysisRun(id: string): Promise<AnalysisRun | undefined>;
  listAnalysisRuns(graphVersionId?: string): Promise<readonly AnalysisRun[]>;
  saveEvent(event: SemanticEvent): Promise<void>;
  appendEvent(event: SemanticEvent): Promise<void>;
  listEvents(graphVersionId?: string): Promise<readonly SemanticEvent[]>;
  saveSession(session: ResearchSession): Promise<void>;
  getSession(id: string): Promise<ResearchSession | undefined>;
  listSessions(state?: SessionListState): Promise<readonly ResearchSession[]>;
  saveMessage(message: ConversationMessage): Promise<void>;
  listMessages(sessionId: string): Promise<readonly ConversationMessage[]>;
  deleteMessage(messageId: string): Promise<void>;
  associateMessageSourceArtifacts(messageId: string, sourceArtifactIds: readonly string[]): Promise<void>;
  trashSession(id: string): Promise<void>;
  restoreSession(id: string): Promise<void>;
  purgeSession(id: string): Promise<void>;
  getInitializationMetadata(): Promise<RepositoryInitializationMetadata | undefined>;
  saveInitializationMetadata(metadata: RepositoryInitializationMetadata): Promise<void>;
  subscribe(listener: (change: RepositoryChange) => void): () => void;
  dispose(): void;
}
