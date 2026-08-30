import type {
  GraphAttributes,
  GraphEdge,
  GraphNode,
  GraphVersion,
  ValidationIssue,
} from "../types/graph";
import { inferGraphDirectedness } from "./graphAlgorithms";
import {
  canonicalJson,
  compareUnicodeCodePoints,
  sha256Text,
} from "./graphIdentity";
import { SocialGraphApiError, socialGraphApiUrl } from "./apiClient";

export interface DatasetProfile {
  readonly nodeCount?: number;
  readonly edgeCount?: number;
  readonly featureDimension?: number;
  readonly labelCount?: number;
  readonly splitNames: readonly string[];
  readonly directed: boolean;
}

export interface DatasetIssue {
  readonly severity: "warning" | "error";
  readonly code: string;
  readonly message: string;
  readonly file?: string;
}

export interface TrustedDiscoveredDataset {
  readonly name: string;
  readonly detectedFormat: string;
  readonly fileCount: number;
}

export interface TrustedConversionJob {
  readonly id: string;
  readonly sourcePath: string;
  readonly trustedRoot: string;
  readonly status: "awaiting_authorization" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
  readonly progress: number;
  readonly fileCount: number;
  readonly totalBytes: number;
  readonly datasets: readonly TrustedDiscoveredDataset[];
  readonly artifactIds: readonly string[];
  readonly issues: readonly DatasetIssue[];
  readonly converterPython: string;
  readonly createdAt: string;
  readonly updatedAt: string;
  readonly authorizationToken?: string;
}

export interface DatasetArtifactRef {
  readonly schemaVersion?: "1.0" | "2.0" | "2.1" | "2.2";
  readonly id: string;
  readonly datasetName?: string;
  readonly checksum: string;
  readonly canonicalGraphHash: string;
  readonly contentHash?: string;
  readonly manifestHash?: string;
  readonly datasetRole?: "benchmark" | "target_domain" | "pretraining_candidate";
  readonly readinessStatus?: "legacy" | "unchecked";
  readonly scope: "complete" | "projection";
  readonly profile: DatasetProfile;
  readonly createdAt: string;
  readonly lifecycle?: "active" | "trashed";
}

export interface ArtifactReference {
  readonly kind: "graph_dataset_binding" | "embedded_training_ref";
  readonly id: string;
  readonly blocking: boolean;
  readonly detail: Readonly<Record<string, unknown>>;
}

export interface DatasetArtifactDeletionImpact {
  readonly artifactId: string;
  readonly lifecycle: "active" | "trashed";
  readonly blockers: readonly ArtifactReference[];
  readonly dependents: readonly ArtifactReference[];
  readonly preserved: readonly string[];
  readonly impactHash: string;
}

export interface ResourceLifecycle {
  readonly artifactId: string;
  readonly status: "active" | "trashed";
  readonly updatedAt: string;
  readonly trashedAt?: string | null;
}

export interface DatasetArtifactLifecycleResponse {
  readonly lifecycle: ResourceLifecycle;
  readonly impact: DatasetArtifactDeletionImpact;
}

export interface DatasetArtifactPurgeResponse {
  readonly artifactId: string;
  readonly purged: true;
  readonly cleanupPending: boolean;
}

export interface DatasetArtifact extends DatasetArtifactRef {
  readonly inspectionId: string;
  readonly sourceFormat: string;
  readonly sourceFiles: readonly string[];
  readonly rawManifest: Readonly<Record<string, unknown>>;
  readonly derivedManifest: Readonly<Record<string, unknown>>;
  readonly sourceFileDigests?: readonly {
    readonly path: string;
    readonly role: string;
    readonly size: number;
    readonly sha256: string;
  }[];
  readonly nodeIdentity?: {
    readonly id: string;
    readonly arrayName: string;
    readonly kind: "source" | "row_index";
    readonly count: number;
    readonly unique: boolean;
  } | null;
  readonly featureSchemas?: readonly { readonly id: string; readonly arrayName: string; readonly shape: readonly number[] }[];
  readonly labelSchemas?: readonly { readonly id: string; readonly arrayName: string; readonly classCount?: number }[];
  readonly graphVariants?: readonly { readonly id: string }[];
  readonly featureRecipes?: readonly { readonly id: string; readonly graphVariant: string; readonly fitScope: string }[];
  readonly splitSets?: readonly { readonly id: string; readonly kind: string; readonly foldCount: number }[];
  readonly taskSpecs?: readonly { readonly id: string; readonly kind: string; readonly evaluationProtocol: string }[];
  readonly licensePolicy?: {
    readonly status: "verified" | "user_attested" | "restricted" | "unknown";
    readonly identifier: string;
    readonly allowedUses: readonly ("evaluation" | "adaptation" | "inference" | "pretraining")[];
  } | null;
  readonly licenseEvidence?: readonly {
    readonly id: string;
    readonly kind: "official_metadata" | "official_license" | "user_attestation";
    readonly sourceUrl?: string | null;
    readonly sha256?: string | null;
    readonly recordedAt: string;
    readonly recordedBy: string;
  }[];
  readonly dataGovernance?: DataGovernancePolicy | null;
  readonly preparationSpec?: DatasetPreparationSpec | null;
  readonly trainingRef?: TrainingDatasetRef | null;
  readonly trainingRefs?: readonly TrainingDatasetRef[];
  readonly graphView: {
    readonly id: string;
    readonly nodes: readonly {
      id: string;
      label: string;
      nodeType?: string | null;
      attributes?: GraphAttributes;
    }[];
    readonly edges: readonly {
      id: string;
      source: string;
      target: string;
      edgeType?: string | null;
      weight?: number | null;
      timestamp?: string | null;
      directed?: boolean | null;
      attributes?: GraphAttributes;
    }[];
    readonly summary: {
      readonly nodeCount: number;
      readonly edgeCount: number;
      readonly density: number;
      readonly connectedComponents: number;
      readonly visibleNodeCount: number;
      readonly visibleEdgeCount: number;
      readonly partialPreview: boolean;
    };
  };
}

export interface TrainingDatasetRef {
  readonly schemaVersion: "1.0" | "1.1";
  readonly artifactId: string;
  readonly contentHash: string;
  readonly manifestHash?: string;
  readonly graphVariant: string;
  readonly splitSetId?: string;
  readonly splitFold?: number;
  readonly featureRecipeId: string;
  readonly taskSpecId?: string;
  readonly datasetRole: "benchmark" | "target_domain" | "pretraining_candidate";
  readonly intendedUse: "evaluation" | "adaptation" | "inference" | "pretraining";
  readonly refHash: string;
}

export interface DatasetReadinessIssue {
  readonly code: string;
  readonly message: string;
  readonly severity: "blocker" | "warning";
}

export interface DatasetReadiness {
  readonly artifactId: string;
  readonly status: "ready" | "blocked" | "legacy" | "corrupt";
  readonly contentHash: string;
  readonly manifestHash: string;
  readonly trainingRef?: TrainingDatasetRef | null;
  readonly blockers: readonly DatasetReadinessIssue[];
  readonly warnings: readonly DatasetReadinessIssue[];
  readonly checkedAt: string;
}

export interface MaterializedDatasetContract {
  readonly artifactId: string;
  readonly trainingRefHash: string;
  readonly nodeCount: number;
  readonly edgeCount: number;
  readonly featureShape: readonly number[];
  readonly labelShape?: readonly number[] | null;
  readonly splitSizes: Readonly<Record<string, number>>;
  readonly taskKind: "node_classification" | "link_prediction";
}

export interface DatasetInspection {
  readonly id: string;
  readonly detectedFormat: string;
  readonly status: "accepted" | "mapping_required" | "conversion_required" | "rejected";
  readonly profile?: DatasetProfile;
  readonly issues: readonly DatasetIssue[];
  readonly datasetCandidates: readonly string[];
  readonly serverGraphFactHash?: string | null;
}

export interface RuntimeCapability {
  readonly buildId: string;
  readonly apiContract: "socialgraph-fm-api/1.1";
  readonly storageSchema: "dataset-store/2";
  readonly datasetArtifactSchemas: readonly string[];
  readonly trainingRefSchemas: readonly string[];
  readonly graphHandoffSchemas: readonly string[];
  readonly graphFactHash: "graph-fact-hash/1";
  readonly converterEnvironmentFingerprint: string;
}

export interface ResearchDatasetCapabilities {
  readonly persistentArtifacts: boolean;
  readonly trustedLocalEnabled: boolean;
  readonly loopbackOnly: boolean;
  readonly safeUploadFormats: readonly string[];
  readonly runtime: RuntimeCapability;
}

export interface DataGovernancePolicy {
  readonly containsPersonalData: boolean;
  readonly deidentified: boolean;
  readonly attributeAllowlist: readonly string[];
  readonly excludedAttributes: readonly string[];
  readonly retention: "session" | "project" | "research_archive";
  readonly userDataTrainingOptIn: false;
}

export interface DatasetPreparationSpec {
  readonly schemaVersion: "1.0";
  readonly graphVersionId: string;
  readonly featureAttributes: readonly string[];
  readonly labelAttribute: string | null;
  readonly taskKind: "none" | "node_classification" | "link_prediction";
  readonly splitStrategy: "none" | "provided" | "temporal";
  readonly excludedAttributes: readonly string[];
  readonly deidentify: boolean;
  readonly governance: DataGovernancePolicy;
}

export interface GraphDatasetBinding {
  readonly id: string;
  readonly graphVersionId: string;
  readonly graphFactHash: string;
  readonly artifactId: string;
  readonly preparationHash: string;
  readonly createdAt: string;
}

export interface GraphVersionTargetDomainEnvelope {
  readonly schemaVersion: "socialgraph-fm-graph/1.1";
  readonly graphVersionId: string;
  readonly contentHash: string;
  readonly buildSpecHash: string;
  readonly sourceFile: string;
  readonly graphFactHash: string;
  readonly directedness: "directed" | "undirected" | "mixed" | "unspecified";
  readonly nodes: readonly {
    readonly id: string;
    readonly label: string;
    readonly type: string | null;
    readonly attributes: GraphAttributes;
  }[];
  readonly edges: readonly {
    readonly id: string;
    readonly source: string;
    readonly target: string;
    readonly type: string | null;
    readonly weight: number | null;
    readonly timestamp: string | null;
    readonly directed: boolean | null;
    readonly attributes: GraphAttributes;
  }[];
}

export interface PreparedGraphVersionTargetDomain {
  readonly file: File;
  readonly inspection: DatasetInspection;
  readonly artifact: DatasetArtifact;
  readonly binding: GraphDatasetBinding;
  readonly reused: boolean;
  readonly researchCompatibility?: ResearchGraphCompatibility | null;
}

export interface ResearchGraphCompatibility {
  readonly intendedUse: "gfm_research";
  readonly status: "compatible" | "blocked";
  readonly compatibleTaskIds: readonly ["core.collaboration_completion"] | readonly [];
  readonly auxiliaryCapabilities: readonly ["similar-nodes"] | readonly [];
  readonly blockers: readonly { readonly code: string; readonly message: string }[];
  readonly adapterStatus: "pending_registration" | "ready";
}

const SHA256_PATTERN = /^[a-f0-9]{64}$/u;

const compareUnicode = compareUnicodeCodePoints;

export function computeGraphFactHash(
  envelope: Pick<GraphVersionTargetDomainEnvelope, "directedness" | "nodes" | "edges">,
): string {
  const facts = {
    directedness: envelope.directedness,
    nodes: [...envelope.nodes].sort((left, right) => compareUnicode(left.id, right.id)),
    edges: [...envelope.edges].sort((left, right) => (
      compareUnicode(left.id, right.id)
      || compareUnicode(left.source, right.source)
      || compareUnicode(left.target, right.target)
    )),
  };
  return sha256Text(`socialgraph-fm-graph-fact-v1\0${canonicalJson(facts)}`);
}

/** Build the only public text handoff accepted for browser GraphVersions. */
export function graphVersionTargetDomainEnvelope(version: GraphVersion): GraphVersionTargetDomainEnvelope {
  if (version.datasetArtifact?.scope === "projection") {
    throw new Error("研究数据投影不是完整 GraphVersion，不能再次导出为目标域数据。");
  }
  if (!version.contentHash || !SHA256_PATTERN.test(version.contentHash)) {
    throw new Error("GraphVersion 缺少可验证的 contentHash。");
  }
  if (!version.buildSpecHash || !SHA256_PATTERN.test(version.buildSpecHash)) {
    throw new Error("GraphVersion 缺少可验证的 buildSpecHash；请从确定性导入链路重新构图。");
  }
  if (version.summary.nodeCount !== version.nodes.length || version.summary.edgeCount !== version.edges.length) {
    throw new Error("GraphVersion 完整事实数量与摘要不一致，禁止导出不完整投影。");
  }
  const nodeIds = new Set(version.nodes.map((node) => node.id));
  if (nodeIds.size !== version.nodes.length) throw new Error("GraphVersion 节点 ID 不唯一。");
  const edgeIds = new Set(version.edges.map((edge) => edge.id));
  if (edgeIds.size !== version.edges.length) throw new Error("GraphVersion 边 ID 不唯一。");
  const dangling = version.edges.find((edge) => !nodeIds.has(edge.source) || !nodeIds.has(edge.target));
  if (dangling) throw new Error(`边 ${dangling.id} 引用了不存在的节点。`);

  const inferredDirectedness = inferGraphDirectedness(version);
  if (version.metadata?.directedness && version.metadata.directedness !== inferredDirectedness) {
    throw new Error("GraphVersion 图级 directedness 与逐边事实不一致。");
  }
  const nodes = [...version.nodes]
    .sort((left, right) => compareUnicode(left.id, right.id))
    .map((node) => ({
      id: node.id,
      label: node.label,
      type: node.type ?? null,
      attributes: node.attributes,
    }));
  const edges = [...version.edges]
    .sort((left, right) => compareUnicode(left.id, right.id))
    .map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: edge.type ?? null,
      weight: edge.weight ?? null,
      timestamp: edge.timestamp ?? null,
      directed: edge.directed ?? null,
      attributes: edge.attributes,
    }));
  const graphFactHash = computeGraphFactHash({ directedness: inferredDirectedness, nodes, edges });
  return Object.freeze({
    schemaVersion: "socialgraph-fm-graph/1.1",
    graphVersionId: version.id,
    contentHash: version.contentHash,
    buildSpecHash: version.buildSpecHash,
    sourceFile: version.sourceFile,
    graphFactHash,
    directedness: inferredDirectedness,
    nodes: Object.freeze(nodes),
    edges: Object.freeze(edges),
  });
}

export function graphVersionTargetDomainFile(version: GraphVersion): File {
  const payload = graphVersionTargetDomainEnvelope(version);
  const safeId = version.id.replace(/[^a-zA-Z0-9._-]+/gu, "-").slice(0, 120) || "graph-version";
  return new File(
    [JSON.stringify(payload)],
    `${safeId}.sgfm-graph.json`,
    { type: "application/vnd.socialgraph-fm.graph+json" },
  );
}

async function readJson<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({})) as T & { code?: unknown; detail?: unknown };
  if (!response.ok) {
    const detailRecord = payload.detail && typeof payload.detail === "object" && !Array.isArray(payload.detail)
      ? payload.detail as { code?: unknown; msg?: unknown; message?: unknown }
      : null;
    const rawCode = payload.code ?? detailRecord?.code;
    const code = typeof rawCode === "string" && /^[A-Z0-9_]{1,100}$/u.test(rawCode)
      ? rawCode
      : "RESEARCH_DATASET_SERVICE_ERROR";
    const detail = typeof payload.detail === "string"
      ? payload.detail
      : Array.isArray(payload.detail)
        ? payload.detail
          .map((entry) => {
            if (!entry || typeof entry !== "object") return null;
            const candidate = entry as { msg?: unknown; message?: unknown };
            return typeof candidate.msg === "string"
              ? candidate.msg
              : typeof candidate.message === "string"
                ? candidate.message
                : null;
          })
          .filter((message): message is string => Boolean(message))
          .join("；") || undefined
        : detailRecord
          ? typeof detailRecord.message === "string"
            ? detailRecord.message
            : typeof detailRecord.msg === "string"
              ? detailRecord.msg
              : undefined
          : undefined;
    throw new SocialGraphApiError(
      code,
      detail ?? `研究数据服务请求未完成（${code}）。`,
      response.status,
    );
  }
  return payload;
}

export class ResearchDatasetClient {
  constructor(private readonly baseUrl = socialGraphApiUrl("/api/v1")) {}

  async capabilities(): Promise<ResearchDatasetCapabilities> {
    const payload = await readJson<{
      researchDatasets?: Omit<ResearchDatasetCapabilities, "runtime">;
      runtime?: RuntimeCapability;
    }>(
      await fetch(`${this.baseUrl}/capabilities`),
    );
    const runtime = payload.runtime;
    const compatible = runtime?.apiContract === "socialgraph-fm-api/1.1"
      && runtime.storageSchema === "dataset-store/2"
      && runtime.graphFactHash === "graph-fact-hash/1"
      && runtime.datasetArtifactSchemas.includes("2.2")
      && runtime.trainingRefSchemas.includes("1.1")
      && runtime.graphHandoffSchemas.includes("socialgraph-fm-graph/1.1")
      && SHA256_PATTERN.test(runtime.converterEnvironmentFingerprint);
    if (!compatible) {
      throw new Error("研究数据后端版本过旧或协议不兼容，请重启最新 API 后再试。");
    }
    return { ...(payload.researchDatasets ?? {
      persistentArtifacts: false,
      trustedLocalEnabled: false,
      loopbackOnly: true,
      safeUploadFormats: [],
    }), runtime };
  }

  async inspectLocal(sourcePath: string): Promise<TrustedConversionJob> {
    return readJson(await fetch(`${this.baseUrl}/dataset-imports/inspect-local`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sourcePath }),
    }));
  }

  async authorize(jobId: string, authorizationToken: string): Promise<TrustedConversionJob> {
    return readJson(await fetch(`${this.baseUrl}/dataset-imports/local-jobs/${encodeURIComponent(jobId)}/authorize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ authorizationToken, confirmTrusted: true }),
    }));
  }

  async getJob(jobId: string): Promise<TrustedConversionJob> {
    return readJson(await fetch(`${this.baseUrl}/dataset-imports/local-jobs/${encodeURIComponent(jobId)}`));
  }

  async cancel(jobId: string): Promise<TrustedConversionJob> {
    return readJson(await fetch(`${this.baseUrl}/dataset-imports/local-jobs/${encodeURIComponent(jobId)}/cancel`, {
      method: "POST",
    }));
  }

  async listArtifacts(includeTrashed = false): Promise<readonly DatasetArtifactRef[]> {
    const query = includeTrashed ? "?includeTrashed=true" : "";
    return readJson(await fetch(`${this.baseUrl}/dataset-artifacts${query}`));
  }

  async getArtifact(artifactId: string): Promise<DatasetArtifact> {
    return readJson(await fetch(`${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}`));
  }

  async getDeletionImpact(artifactId: string): Promise<DatasetArtifactDeletionImpact> {
    return readJson(await fetch(
      `${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}/deletion-impact`,
    ));
  }

  async trashArtifact(artifactId: string): Promise<DatasetArtifactLifecycleResponse> {
    return readJson(await fetch(
      `${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}/trash`,
      { method: "POST" },
    ));
  }

  async restoreArtifact(artifactId: string): Promise<DatasetArtifactLifecycleResponse> {
    return readJson(await fetch(
      `${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}/restore`,
      { method: "POST" },
    ));
  }

  async purgeArtifact(
    artifactId: string,
    impactHash: string,
    confirmation: string,
  ): Promise<DatasetArtifactPurgeResponse> {
    return readJson(await fetch(
      `${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}/purge`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ impactHash, confirmation }),
      },
    ));
  }

  async getReadiness(artifactId: string, trainingRefHash?: string): Promise<DatasetReadiness> {
    const query = trainingRefHash
      ? `?trainingRefHash=${encodeURIComponent(trainingRefHash)}`
      : "";
    return readJson(await fetch(
      `${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}/readiness${query}`,
    ));
  }

  async resolveTrainingRef(reference: Pick<TrainingDatasetRef,
    "artifactId" | "contentHash" | "graphVariant" | "splitSetId" | "splitFold" | "featureRecipeId" | "taskSpecId" | "intendedUse"
  >): Promise<{ readonly reference: TrainingDatasetRef; readonly readiness: DatasetReadiness }> {
    return readJson(await fetch(`${this.baseUrl}/training-dataset-refs/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        artifactId: reference.artifactId,
        contentHash: reference.contentHash,
        graphVariant: reference.graphVariant,
        splitSetId: reference.splitSetId,
        splitFold: reference.splitFold ?? 0,
        featureRecipeId: reference.featureRecipeId,
        taskSpecId: reference.taskSpecId,
        intendedUse: reference.intendedUse,
      }),
    }));
  }

  async getMaterializedContract(
    artifactId: string,
    trainingRefHash: string,
  ): Promise<MaterializedDatasetContract> {
    return readJson(await fetch(
      `${this.baseUrl}/dataset-artifacts/${encodeURIComponent(artifactId)}/materialized-contract?trainingRefHash=${encodeURIComponent(trainingRefHash)}`,
    ));
  }

  async inspectPackage(file: File, dataset?: string): Promise<DatasetInspection> {
    const form = new FormData();
    form.append("files", file, file.name);
    if (dataset?.trim()) form.append("dataset", dataset.trim());
    const inspection = await readJson<
      Omit<DatasetInspection, "datasetCandidates"> & { readonly datasetCandidates?: readonly string[] }
    >(await fetch(`${this.baseUrl}/dataset-imports/inspect`, { method: "POST", body: form }));
    return { ...inspection, datasetCandidates: inspection.datasetCandidates ?? [] };
  }

  async commitInspection(inspectionId: string): Promise<DatasetArtifact> {
    return readJson(await fetch(`${this.baseUrl}/dataset-imports/${encodeURIComponent(inspectionId)}/commit`, {
      method: "POST",
    }));
  }

  async reserveGraphHandoff(graphVersionId: string, graphFactHash: string): Promise<{
    readonly token: string;
    readonly graphVersionId: string;
    readonly graphFactHash: string;
    readonly expiresAt: string;
  }> {
    return readJson(await fetch(`${this.baseUrl}/graph-dataset-handoffs/reserve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ graphVersionId, graphFactHash }),
    }));
  }

  async cancelGraphHandoff(token: string): Promise<void> {
    await readJson(await fetch(`${this.baseUrl}/graph-dataset-handoffs/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    }));
  }

  async commitGraphHandoff(
    token: string,
    envelope: GraphVersionTargetDomainEnvelope,
    preparation: DatasetPreparationSpec,
    intendedUse: "dataset" | "gfm_research" = "dataset",
  ): Promise<{
    readonly binding: GraphDatasetBinding;
    readonly artifact: DatasetArtifact;
    readonly reused: boolean;
    readonly researchCompatibility?: ResearchGraphCompatibility | null;
  }> {
    return readJson(await fetch(`${this.baseUrl}/graph-dataset-handoffs/commit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token, envelope, preparation, intendedUse }),
    }));
  }

  /** Explicit user-triggered handoff only; callers decide when to invoke it. */
  async prepareGraphVersionTargetDomain(
    version: GraphVersion,
    intendedUse: "dataset" | "gfm_research" = "dataset",
  ): Promise<PreparedGraphVersionTargetDomain> {
    // Fail closed before sending graph facts to a stale or incompatible API.
    await this.capabilities();
    const envelope = graphVersionTargetDomainEnvelope(version);
    const file = graphVersionTargetDomainFile(version);
    const inspection = await this.inspectPackage(file);
    if (inspection.status !== "accepted") {
      const detail = inspection.issues.map((issue) => issue.message).join("；");
      throw new Error(detail || "GraphVersion 目标域交接未通过检查。");
    }
    if (inspection.serverGraphFactHash !== envelope.graphFactHash) {
      throw new Error("浏览器与后端计算的 GraphFactHash 不一致，交接已拒绝。");
    }
    const allAttributes = [...new Set([
      ...version.nodes.flatMap((node) => Object.keys(node.attributes)),
      ...version.edges.flatMap((edge) => Object.keys(edge.attributes)),
    ])].sort(compareUnicode);
    const preparation: DatasetPreparationSpec = {
      schemaVersion: "1.0",
      graphVersionId: version.id,
      featureAttributes: [],
      labelAttribute: null,
      taskKind: "none",
      splitStrategy: "none",
      excludedAttributes: allAttributes,
      deidentify: false,
      governance: {
        containsPersonalData: true,
        deidentified: false,
        attributeAllowlist: [],
        excludedAttributes: allAttributes,
        retention: "project",
        userDataTrainingOptIn: false,
      },
    };
    const reservation = await this.reserveGraphHandoff(version.id, envelope.graphFactHash);
    let committed: Awaited<ReturnType<ResearchDatasetClient["commitGraphHandoff"]>>;
    try {
      committed = await this.commitGraphHandoff(reservation.token, envelope, preparation, intendedUse);
    } catch (error) {
      await this.cancelGraphHandoff(reservation.token).catch(() => undefined);
      throw error;
    }
    const { artifact } = committed;
    if (artifact.datasetRole !== "target_domain") {
      throw new Error("服务端返回了非 target_domain Artifact，交接已拒绝。");
    }
    return {
      file,
      inspection,
      artifact,
      binding: committed.binding,
      reused: committed.reused,
      ...(intendedUse === "gfm_research"
        ? { researchCompatibility: committed.researchCompatibility ?? null }
        : {}),
    };
  }
}

export function graphVersionFromDatasetArtifact(artifact: DatasetArtifact): GraphVersion {
  const nodes: readonly GraphNode[] = Object.freeze(artifact.graphView.nodes.map((node) => Object.freeze({
    id: node.id,
    label: node.label,
    ...(node.nodeType ? { type: node.nodeType } : {}),
    attributes: Object.freeze({ ...(node.attributes ?? {}), datasetArtifactId: artifact.id }),
  })));
  const edges: readonly GraphEdge[] = Object.freeze(artifact.graphView.edges.map((edge) => Object.freeze({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    ...(edge.edgeType ? { type: edge.edgeType } : {}),
    ...(edge.weight != null ? { weight: edge.weight } : {}),
    ...(edge.timestamp ? { timestamp: edge.timestamp } : {}),
    directed: edge.directed ?? artifact.profile.directed,
    attributes: Object.freeze({ ...(edge.attributes ?? {}), datasetArtifactId: artifact.id }),
  })));
  const projection = artifact.scope === "projection" || artifact.graphView.summary.partialPreview;
  const issues: readonly ValidationIssue[] = projection
    ? Object.freeze([Object.freeze({
        code: "DATASET_ARTIFACT_PROJECTION",
        severity: "info" as const,
        message: `当前展示 ${nodes.length}/${artifact.graphView.summary.nodeCount} 个节点的科研数据投影；全图分析必须在服务端 Artifact 上执行。`,
      })])
    : Object.freeze([]);
  return Object.freeze({
    id: `artifact-${artifact.id}-${artifact.canonicalGraphHash.slice(0, 10)}`,
    sourceFile: `${artifact.datasetName ?? artifact.id}.sgfm-artifact`,
    createdAt: artifact.createdAt,
    nodes,
    edges,
    summary: Object.freeze({
      nodeCount: artifact.graphView.summary.nodeCount,
      edgeCount: artifact.graphView.summary.edgeCount,
      density: artifact.graphView.summary.density,
      averageDegree: artifact.graphView.summary.nodeCount > 0
        ? (2 * artifact.graphView.summary.edgeCount) / artifact.graphView.summary.nodeCount
        : 0,
      connectedComponents: artifact.graphView.summary.connectedComponents,
      isolatedNodes: 0,
    }),
    issues,
    preview: Object.freeze({
      nodes,
      edges,
      truncated: projection,
      originalNodeCount: artifact.graphView.summary.nodeCount,
      originalEdgeCount: artifact.graphView.summary.edgeCount,
    }),
    truncated: projection,
    metadata: Object.freeze({ directedness: artifact.profile.directed ? "directed" : "undirected" }),
    provenance: Object.freeze({
      origin: "research_dataset" as const,
      pipeline: "dataset-artifact" as const,
      pipelineVersion: artifact.schemaVersion ?? "1.0",
      sourceHashScheme: "dataset-content-hash-v2" as const,
    }),
    datasetArtifact: Object.freeze({
      id: artifact.id,
      datasetName: artifact.datasetName ?? artifact.id,
      checksum: artifact.checksum,
      canonicalGraphHash: artifact.canonicalGraphHash,
      ...(artifact.contentHash ? { contentHash: artifact.contentHash } : {}),
      ...(artifact.trainingRef?.refHash ? { trainingRefHash: artifact.trainingRef.refHash } : {}),
      ...(artifact.datasetRole ? { datasetRole: artifact.datasetRole } : {}),
      scope: projection ? "projection" : "complete",
    }),
  });
}
