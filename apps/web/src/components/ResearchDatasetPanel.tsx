import {
  ArrowClockwise,
  CheckCircle,
  CircleNotch,
  Database,
  FileZip,
  FolderOpen,
  Play,
  ArrowCounterClockwise,
  ShieldCheck,
  Stop,
  Trash,
  WarningCircle,
} from "@phosphor-icons/react";
import { type KeyboardEvent, useCallback, useEffect, useRef, useState } from "react";

import {
  ResearchDatasetClient,
  type DatasetArtifact,
  type DatasetArtifactRef,
  type DatasetInspection,
  type DataGovernancePolicy,
  type DatasetArtifactDeletionImpact,
  type DatasetPreparationSpec,
  type DatasetReadiness,
  type MaterializedDatasetContract,
  type ResearchDatasetCapabilities,
  type TrustedConversionJob,
} from "../services/researchDatasetClient";

const TRUSTED_PATH_KEY = "socialgraph-fm-trusted-dataset-path";

interface ResearchDatasetPanelProps {
  readonly client: ResearchDatasetClient;
  readonly onOpenArtifact: (artifact: DatasetArtifact) => Promise<void> | void;
  readonly onOpenInferencePackages?: () => void;
  readonly onNotify?: (message: string) => void;
}

interface LicenseEvidenceView {
  readonly id: string;
  readonly kind: "official_metadata" | "official_license" | "user_attestation";
  readonly sourceUrl?: string | null;
  readonly recordedAt: string;
  readonly recordedBy: string;
}

type ArtifactContractView = DatasetArtifact & {
  readonly licenseEvidence?: readonly LicenseEvidenceView[];
  readonly dataGovernance?: DataGovernancePolicy | null;
  readonly preparationSpec?: DatasetPreparationSpec | null;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function jobStatusLabel(status: TrustedConversionJob["status"]): string {
  switch (status) {
    case "awaiting_authorization": return "等待一次性授权";
    case "queued": return "等待独立转换进程";
    case "running": return "正在转换";
    case "succeeded": return "转换完成";
    case "failed": return "转换失败";
    case "cancelled": return "已取消";
  }
}

function readinessLabel(status: DatasetReadiness["status"] | "unchecked"): string {
  switch (status) {
    case "ready": return "训练合同就绪";
    case "blocked": return "训练前置阻断";
    case "legacy": return "旧版需重导";
    case "corrupt": return "完整性异常";
    case "unchecked": return "尚未检查";
  }
}

function artifactStatusLabel(status: DatasetReadiness["status"] | "unchecked"): string {
  switch (status) {
    case "ready": return "可用";
    case "blocked": return "需要处理";
    case "legacy": return "旧版需重导";
    case "corrupt": return "完整性异常";
    case "unchecked": return "尚未检查";
  }
}

function datasetRoleLabel(role: DatasetArtifactRef["datasetRole"]): string {
  switch (role) {
    case "benchmark": return "评测基准";
    case "pretraining_candidate": return "预训练候选";
    case "target_domain":
    default: return "目标域数据";
  }
}

function artifactUpdatedLabel(createdAt: string): string {
  const date = new Date(createdAt);
  if (Number.isNaN(date.getTime())) return "更新时间未知";
  return `更新于 ${new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "numeric",
    day: "numeric",
  }).format(date)}`;
}

function licenseStatusLabel(status: NonNullable<DatasetArtifact["licensePolicy"]>["status"] | undefined): string {
  switch (status) {
    case "verified": return "证据已核验";
    case "user_attested": return "用户声明";
    case "restricted": return "受限使用";
    case "unknown":
    default: return "用途未确认";
  }
}

function evidenceKindLabel(kind: LicenseEvidenceView["kind"]): string {
  switch (kind) {
    case "official_license": return "官方许可证";
    case "official_metadata": return "官方元数据";
    case "user_attestation": return "用户声明";
  }
}

function taskKindLabel(kind: DatasetPreparationSpec["taskKind"]): string {
  switch (kind) {
    case "node_classification": return "节点分类";
    case "link_prediction": return "链路预测";
    case "none": return "未指定训练任务";
  }
}

function retentionLabel(retention: DataGovernancePolicy["retention"]): string {
  switch (retention) {
    case "session": return "仅会话";
    case "project": return "项目期";
    case "research_archive": return "科研归档";
  }
}

function isLegacyArtifact(artifact: DatasetArtifactRef): boolean {
  return artifact.schemaVersion !== "2.2" || artifact.readinessStatus === "legacy";
}

function contractItems(items: readonly { readonly id: string }[] | undefined): string {
  if (!items?.length) return "无";
  const preview = items.slice(0, 6).map((item) => item.id).join("、");
  return items.length > 6 ? `${preview} 等 ${items.length} 项` : preview;
}

function isProtocolError(message: string): boolean {
  return /协议不兼容|版本过旧|重启最新 API/u.test(message);
}

function publicDatasetError(message: string): string {
  return message
    .replace(/TrainingDatasetRef|TrainingRef/gu, "数据准备凭据")
    .replace(/GraphDatasetBinding|目标域 Binding/gu, "图数据交接记录")
    .replace(/训练前置合同|独立物化合同|训练合同/gu, "数据准备校验");
}

function artifactReferenceKindLabel(
  kind: DatasetArtifactDeletionImpact["dependents"][number]["kind"],
  technical: boolean,
): string {
  switch (kind) {
    case "graph_dataset_binding": return technical ? "GraphDatasetBinding" : "图谱交接记录";
    case "embedded_training_ref": return technical ? "内嵌 TrainingDatasetRef" : "数据准备引用";
  }
}

export function ResearchDatasetPanel({ client, onOpenArtifact, onOpenInferencePackages, onNotify }: ResearchDatasetPanelProps) {
  const packageInputRef = useRef<HTMLInputElement>(null);
  const trustedPathInputRef = useRef<HTMLInputElement>(null);
  const adminToolsRef = useRef<HTMLDetailsElement>(null);
  const activeArtifactTabRef = useRef<HTMLButtonElement>(null);
  const trashedArtifactTabRef = useRef<HTMLButtonElement>(null);
  const [capabilities, setCapabilities] = useState<ResearchDatasetCapabilities | null>(null);
  const [artifacts, setArtifacts] = useState<readonly DatasetArtifactRef[]>([]);
  const [sourcePath, setSourcePath] = useState(() =>
    window.localStorage.getItem(TRUSTED_PATH_KEY) ?? "",
  );
  const [job, setJob] = useState<TrustedConversionJob | null>(null);
  const [authorizationToken, setAuthorizationToken] = useState<string | null>(null);
  const [trustedConfirmed, setTrustedConfirmed] = useState(false);
  const [packageInspection, setPackageInspection] = useState<DatasetInspection | null>(null);
  const [packageFile, setPackageFile] = useState<File | null>(null);
  const [selectedPackageDataset, setSelectedPackageDataset] = useState<string>("");
  const [showUnnamedArtifacts, setShowUnnamedArtifacts] = useState(false);
  const [artifactView, setArtifactView] = useState<"active" | "trashed">("active");
  const [deletionImpacts, setDeletionImpacts] = useState<Readonly<Record<string, DatasetArtifactDeletionImpact>>>({});
  const [purgeConfirmations, setPurgeConfirmations] = useState<Readonly<Record<string, string>>>({});
  const [adminToolsOpen, setAdminToolsOpen] = useState(false);
  const [artifactContracts, setArtifactContracts] = useState<Readonly<Record<string, {
    readonly artifact: DatasetArtifact;
    readonly readiness: DatasetReadiness;
    readonly resolvedRefHash?: string;
    readonly materializedContract?: MaterializedDatasetContract;
  }>>>({});
  const [busy, setBusy] = useState<"loading" | "inspect" | "authorize" | "package" | "open" | "readiness" | "resolve" | "lifecycle" | "purge" | null>("loading");
  const [error, setError] = useState<string | null>(null);

  const refreshArtifacts = useCallback(async () => {
    try {
      const next = await client.listArtifacts(true);
      setArtifacts(next);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法刷新 Artifact 列表");
    }
  }, [client]);

  useEffect(() => {
    let active = true;
    void Promise.all([client.capabilities(), client.listArtifacts(true)])
      .then(([nextCapabilities, nextArtifacts]) => {
        if (!active) return;
        setCapabilities(nextCapabilities);
        setArtifacts(nextArtifacts);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "研究数据服务暂时不可用");
      })
      .finally(() => {
        if (active) setBusy(null);
      });
    return () => { active = false; };
  }, [client]);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    let active = true;
    const poll = window.setInterval(() => {
      void client.getJob(job.id).then((next) => {
        if (!active) return;
        setJob(next);
        if (next.status === "succeeded") {
          window.clearInterval(poll);
          void refreshArtifacts();
          onNotify?.("科研数据集已转换并持久化");
        } else if (next.status === "failed" || next.status === "cancelled") {
          window.clearInterval(poll);
        }
      }).catch((reason: unknown) => {
        if (active) setError(reason instanceof Error ? reason.message : "无法读取转换进度");
      });
    }, 900);
    return () => {
      active = false;
      window.clearInterval(poll);
    };
  }, [client, job, onNotify, refreshArtifacts]);

  const inspectLocal = async () => {
    if (!sourcePath.trim()) return;
    setBusy("inspect");
    setError(null);
    setJob(null);
    setAuthorizationToken(null);
    setTrustedConfirmed(false);
    try {
      const inspected = await client.inspectLocal(sourcePath.trim());
      window.localStorage.setItem(TRUSTED_PATH_KEY, sourcePath.trim());
      setJob(inspected);
      setAuthorizationToken(inspected.authorizationToken ?? null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "目录检查失败");
    } finally {
      setBusy(null);
    }
  };

  const authorizeLocal = async () => {
    if (!job || !authorizationToken || !trustedConfirmed) return;
    setBusy("authorize");
    setError(null);
    try {
      const authorized = await client.authorize(job.id, authorizationToken);
      setAuthorizationToken(null);
      setJob(authorized);
      onNotify?.("已启动隔离的本地转换进程");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "授权转换失败");
    } finally {
      setBusy(null);
    }
  };

  const cancelLocal = async () => {
    if (!job) return;
    setError(null);
    try {
      setJob(await client.cancel(job.id));
      onNotify?.("已取消科研数据转换");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "取消转换失败");
    }
  };

  const inspectPackage = async (file: File, dataset?: string) => {
    setBusy("package");
    setError(null);
    setPackageInspection(null);
    setPackageFile(file);
    try {
      const inspection = await client.inspectPackage(file, dataset);
      setPackageInspection(inspection);
      const candidates = inspection.datasetCandidates ?? [];
      setSelectedPackageDataset((current) => {
        if (dataset && candidates.includes(dataset)) return dataset;
        if (candidates.includes(current)) return current;
        return candidates[0] ?? "";
      });
      if (inspection.status === "accepted") onNotify?.("研究数据包检查通过，可提交为 Artifact");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "研究数据包检查失败");
    } finally {
      setBusy(null);
    }
  };

  const commitPackage = async () => {
    if (!packageInspection || packageInspection.status !== "accepted") return;
    setBusy("package");
    setError(null);
    try {
      const artifact = await client.commitInspection(packageInspection.id);
      await refreshArtifacts();
      setPackageInspection(null);
      setPackageFile(null);
      setSelectedPackageDataset("");
      onNotify?.("研究数据 Artifact 已创建");
      await onOpenArtifact(artifact);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Artifact 创建失败");
    } finally {
      setBusy(null);
    }
  };

  const openArtifact = async (artifactId: string) => {
    setBusy("open");
    setError(null);
    try {
      await onOpenArtifact(await client.getArtifact(artifactId));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法打开科研数据投影");
    } finally {
      setBusy(null);
    }
  };

  const inspectArtifactReadiness = async (artifactId: string) => {
    setBusy("readiness");
    setError(null);
    try {
      const artifact = await client.getArtifact(artifactId);
      const readiness = await client.getReadiness(artifactId, artifact.trainingRef?.refHash);
      setArtifactContracts((current) => ({
        ...current,
        [artifactId]: { artifact, readiness },
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法检查训练前置合同");
    } finally {
      setBusy(null);
    }
  };

  const resolveArtifactTrainingRef = async (artifactId: string) => {
    const contract = artifactContracts[artifactId];
    const reference = contract?.readiness.trainingRef ?? contract?.artifact.trainingRef;
    if (!contract || !reference?.splitSetId || !reference.taskSpecId) return;
    setBusy("resolve");
    setError(null);
    try {
      const resolved = await client.resolveTrainingRef({
        artifactId: reference.artifactId,
        contentHash: reference.contentHash,
        graphVariant: reference.graphVariant,
        splitSetId: reference.splitSetId,
        splitFold: reference.splitFold ?? 0,
        featureRecipeId: reference.featureRecipeId,
        taskSpecId: reference.taskSpecId,
        intendedUse: reference.intendedUse,
      });
      setArtifactContracts((current) => ({
        ...current,
        [artifactId]: {
          artifact: contract.artifact,
          readiness: resolved.readiness,
          resolvedRefHash: resolved.reference.refHash,
        },
      }));
      if (resolved.readiness.status === "ready") {
        try {
          const materializedContract = await client.getMaterializedContract(
            artifactId,
            resolved.reference.refHash,
          );
          setArtifactContracts((current) => ({
            ...current,
            [artifactId]: {
              ...current[artifactId],
              artifact: contract.artifact,
              readiness: resolved.readiness,
              resolvedRefHash: resolved.reference.refHash,
              materializedContract,
            },
          }));
        } catch (reason) {
          setError(reason instanceof Error
            ? `训练引用已解析，但独立物化合同读取失败：${reason.message}`
            : "训练引用已解析，但独立物化合同读取失败");
        }
      }
      onNotify?.("数据准备凭据已生成并通过审计；尚未启动训练");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法生成训练引用");
    } finally {
      setBusy(null);
    }
  };

  const previewArtifactDeletion = async (artifactId: string) => {
    setBusy("lifecycle");
    setError(null);
    try {
      const impact = await client.getDeletionImpact(artifactId);
      setDeletionImpacts((current) => ({ ...current, [artifactId]: impact }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法预演 Artifact 删除影响");
    } finally {
      setBusy(null);
    }
  };

  const trashArtifact = async (artifactId: string) => {
    if (!deletionImpacts[artifactId]) return;
    setBusy("lifecycle");
    setError(null);
    try {
      await client.trashArtifact(artifactId);
      await refreshArtifacts();
      setDeletionImpacts((current) => {
        const next = { ...current };
        delete next[artifactId];
        return next;
      });
      onNotify?.("DatasetArtifact 已移入后端回收站，可随时恢复");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法将 Artifact 移入回收站");
    } finally {
      setBusy(null);
    }
  };

  const restoreArtifact = async (artifactId: string) => {
    setBusy("lifecycle");
    setError(null);
    try {
      await client.restoreArtifact(artifactId);
      await refreshArtifacts();
      setDeletionImpacts((current) => {
        const next = { ...current };
        delete next[artifactId];
        return next;
      });
      onNotify?.("DatasetArtifact 已从后端回收站恢复");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法恢复 Artifact");
    } finally {
      setBusy(null);
    }
  };

  const purgeArtifact = async (artifactId: string) => {
    const impact = deletionImpacts[artifactId];
    const confirmation = purgeConfirmations[artifactId] ?? "";
    if (!impact || impact.blockers.length || confirmation !== artifactId.slice(-8)) return;
    setBusy("purge");
    setError(null);
    try {
      const result = await client.purgeArtifact(artifactId, impact.impactHash, confirmation);
      await refreshArtifacts();
      setDeletionImpacts((current) => {
        const next = { ...current };
        delete next[artifactId];
        return next;
      });
      setPurgeConfirmations((current) => {
        const next = { ...current };
        delete next[artifactId];
        return next;
      });
      onNotify?.(result.cleanupPending
        ? "Artifact 记录已永久删除；文件清理待恢复队列继续处理"
        : "DatasetArtifact 已永久删除");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "无法永久删除 Artifact");
    } finally {
      setBusy(null);
    }
  };

  const localEnabled = capabilities?.trustedLocalEnabled === true;
  const runtimeCompatible = capabilities?.runtime !== undefined;
  const conversionActive = job?.status === "queued" || job?.status === "running";
  const activeArtifacts = artifacts.filter((artifact) => artifact.lifecycle !== "trashed");
  const trashedArtifacts = artifacts.filter((artifact) => artifact.lifecycle === "trashed");
  const viewArtifacts = artifactView === "active" ? activeArtifacts : trashedArtifacts;
  const namedArtifacts = viewArtifacts.filter((artifact) => Boolean(artifact.datasetName));
  const unnamedArtifacts = viewArtifacts.filter((artifact) => !artifact.datasetName);
  const visibleArtifacts = showUnnamedArtifacts ? viewArtifacts : namedArtifacts;

  const activateArtifactView = (view: "active" | "trashed") => {
    setArtifactView(view);
  };

  const handleArtifactTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const nextView = artifactView === "active" ? "trashed" : "active";
    activateArtifactView(nextView);
    (nextView === "active" ? activeArtifactTabRef : trashedArtifactTabRef).current?.focus();
  };

  const focusTrustedPath = () => {
    setAdminToolsOpen(true);
    if (adminToolsRef.current) adminToolsRef.current.open = true;
    queueMicrotask(() => trustedPathInputRef.current?.focus());
  };

  const visibleError = error ? publicDatasetError(error) : null;

  return (
    <div className="research-datasets">
      <section className="dataset-hero" aria-labelledby="dataset-hero-title">
        <span className="dataset-hero__icon"><Database size={22} weight="duotone" /></span>
        <div>
          <h3 id="dataset-hero-title">数据管理</h3>
          <p>管理图数据、推理包与可恢复记录。</p>
        </div>
        <span
          className={`dataset-service-dot ${runtimeCompatible ? "is-ready" : ""}`}
          title={runtimeCompatible ? "数据服务正常" : "数据服务需要更新"}
        />
      </section>

      {visibleError ? (
        <div className="dataset-alert is-error" role="alert">
          <WarningCircle size={18} weight="fill" />
          <span>
            <strong>{isProtocolError(error ?? "") ? "服务协议不兼容" : "研究数据操作失败"}</strong>
            {visibleError}
            {isProtocolError(error ?? "") ? " 当前页已禁用导入与数据准备操作，请重启最新 API 后刷新。" : ""}
          </span>
        </div>
      ) : null}

      <details
        className="dataset-advanced-tools"
        ref={adminToolsRef}
        onToggle={(event) => setAdminToolsOpen(event.currentTarget.open)}
      >
        <summary>管理员工具</summary>
      {adminToolsOpen ? (
      <div className="dataset-advanced-tools__body">
      <section className="panel-section dataset-section">
        <div className="panel-section__title"><strong>本地受信目录</strong><span>仅 127.0.0.1</span></div>
        <label className="dataset-field">
          <span>数据目录</span>
          <div className="dataset-path-row">
            <input
              ref={trustedPathInputRef}
              id="trusted-dataset-path"
              value={sourcePath}
              onChange={(event) => setSourcePath(event.target.value)}
              disabled={!localEnabled || conversionActive}
              spellCheck={false}
            />
            <button type="button" onClick={() => void inspectLocal()} disabled={!localEnabled || !sourcePath.trim() || busy !== null || conversionActive}>
              {busy === "inspect" ? <CircleNotch size={16} className="spin" /> : <FolderOpen size={16} />}
              检查目录
            </button>
          </div>
        </label>
        {!localEnabled ? (
          <p className="dataset-inline-note">后端尚未启用受信目录。配置 <code>ENABLE_TRUSTED_LOCAL_CONVERSION=true</code> 与允许根目录后重启 API。</p>
        ) : null}

        {job ? (
          <div className="dataset-job" aria-live="polite">
            <div className="dataset-job__head">
              <span><strong>{jobStatusLabel(job.status)}</strong><small>{job.fileCount} 个文件 · {formatBytes(job.totalBytes)}</small></span>
              <strong>{job.progress}%</strong>
            </div>
            <progress max={100} value={job.progress}>{job.progress}%</progress>
            {job.datasets.length ? (
              <div className="dataset-discoveries">
                {job.datasets.map((dataset) => (
                  <span key={`${dataset.name}-${dataset.detectedFormat}`}>
                    <Database size={14} />{dataset.name}<small>{dataset.detectedFormat} · {dataset.fileCount} 文件</small>
                  </span>
                ))}
              </div>
            ) : null}
            {job.issues.length ? (
              <ul className="dataset-issues">
                {job.issues.map((issue, index) => <li key={`${issue.code}-${index}`}>{issue.message}</li>)}
              </ul>
            ) : null}
            {job.status === "awaiting_authorization" ? (
              <div className="dataset-authorization">
                <label>
                  <input type="checkbox" checked={trustedConfirmed} onChange={(event) => setTrustedConfirmed(event.target.checked)} />
                  我确认这是本人受信的科研目录，并理解 .pt / pickle 不是恶意文件安全沙箱。
                </label>
                <button type="button" onClick={() => void authorizeLocal()} disabled={!trustedConfirmed || !authorizationToken || busy !== null}>
                  {busy === "authorize" ? <CircleNotch size={16} className="spin" /> : <Play size={16} />}
                  授权并转换
                </button>
              </div>
            ) : null}
            {conversionActive ? (
              <button className="dataset-cancel" type="button" onClick={() => void cancelLocal()}><Stop size={15} />取消任务</button>
            ) : null}
          </div>
        ) : null}
      </section>

      <section className="panel-section dataset-section">
        <div className="panel-section__title"><strong>可移植研究包</strong><span>.sgfm.zip</span></div>
        <p>安全文本与数值 NPZ 可直接检查；未知 .pt / pickle 仍只检测，不在公开上传接口中加载。</p>
        <input
          ref={packageInputRef}
          type="file"
          accept=".zip,.sgfm.zip,application/zip"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) void inspectPackage(file);
            event.target.value = "";
          }}
        />
        <div className="panel-actions">
          <button type="button" onClick={() => packageInputRef.current?.click()} disabled={busy !== null || !runtimeCompatible}>
            <FileZip size={16} />选择研究包
          </button>
          {packageInspection?.status === "accepted" ? (
            <button type="button" onClick={() => void commitPackage()} disabled={busy !== null}>
              <ShieldCheck size={16} />提交 Artifact
            </button>
          ) : null}
        </div>
        {packageInspection?.status === "mapping_required" && packageInspection.datasetCandidates.length ? (
          <div className="dataset-selection">
            <label>
              <span>选择包内数据集</span>
              <select
                value={selectedPackageDataset}
                onChange={(event) => setSelectedPackageDataset(event.target.value)}
                disabled={busy !== null}
              >
                {packageInspection.datasetCandidates.map((candidate) => (
                  <option value={candidate} key={candidate}>{candidate}</option>
                ))}
              </select>
            </label>
            <button
              type="button"
              onClick={() => {
                if (packageFile && selectedPackageDataset) {
                  void inspectPackage(packageFile, selectedPackageDataset);
                }
              }}
              disabled={!packageFile || !selectedPackageDataset || busy !== null}
            >
              <ShieldCheck size={16} />检查所选数据集
            </button>
          </div>
        ) : null}
        {packageInspection ? (
          <div className={`dataset-alert ${packageInspection.status === "accepted" ? "is-success" : "is-warning"}`} role="status">
            {packageInspection.status === "accepted" ? <CheckCircle size={18} weight="fill" /> : <WarningCircle size={18} weight="fill" />}
            <span><strong>{packageInspection.detectedFormat}</strong>{packageInspection.issues[0]?.message ?? "检查完成"}</span>
          </div>
        ) : null}
      </section>

      </div>
      ) : null}
      </details>

      <section className="panel-section dataset-section">
        <div className="panel-section__title">
          <strong>我的数据</strong>
          <div className="dataset-section-actions">
            {onOpenInferencePackages ? <button type="button" className="dataset-package-manager" onClick={onOpenInferencePackages}>管理推理包</button> : null}
            <button className="dataset-refresh" type="button" onClick={() => void refreshArtifacts()} aria-label="刷新数据列表"><ArrowClockwise size={16} /></button>
          </div>
        </div>
        <div className="dataset-lifecycle-tabs" role="tablist" aria-label="数据分类" aria-orientation="horizontal">
          <button
            ref={activeArtifactTabRef}
            id="dataset-tab-active"
            type="button"
            role="tab"
            aria-selected={artifactView === "active"}
            aria-controls="dataset-artifacts-active"
            tabIndex={artifactView === "active" ? 0 : -1}
            onClick={() => activateArtifactView("active")}
            onKeyDown={handleArtifactTabKeyDown}
          >
            我的数据 <span>{activeArtifacts.length}</span>
          </button>
          <button
            ref={trashedArtifactTabRef}
            id="dataset-tab-trashed"
            type="button"
            role="tab"
            aria-selected={artifactView === "trashed"}
            aria-controls="dataset-artifacts-trashed"
            tabIndex={artifactView === "trashed" ? 0 : -1}
            onClick={() => activateArtifactView("trashed")}
            onKeyDown={handleArtifactTabKeyDown}
          >
            回收站 <span>{trashedArtifacts.length}</span>
          </button>
        </div>
        {(() => {
          const artifactListContent = (
            <>
          {visibleArtifacts.length ? visibleArtifacts.map((artifact) => {
            const legacy = isLegacyArtifact(artifact);
            const deletionImpact = deletionImpacts[artifact.id];
            const confirmationSuffix = artifact.id.slice(-8);
            const purgeConfirmation = purgeConfirmations[artifact.id] ?? "";
            const displayedStatus = artifactContracts[artifact.id]?.readiness.status
              ?? artifact.readinessStatus
              ?? (legacy ? "legacy" : "unchecked");
            return (
              <article className={`dataset-artifact ${legacy ? "is-legacy" : ""}`} key={artifact.id}>
                <span className="dataset-artifact__mark"><Database size={18} weight="duotone" /></span>
                <div className="dataset-artifact__summary">
                  <strong>{artifact.datasetName ?? artifact.id}</strong>
                  <span>图数据 · {datasetRoleLabel(artifact.datasetRole ?? "target_domain")}</span>
                  <span>{artifact.profile.nodeCount ?? 0} 个节点 · {artifact.profile.edgeCount ?? 0} 条关系</span>
                  <span>{artifactUpdatedLabel(artifact.createdAt)}</span>
                  <span className={`dataset-readiness is-${displayedStatus}`}>{artifactStatusLabel(displayedStatus)}</span>
                </div>
                <div className="panel-actions dataset-artifact__actions">
                  {artifactView === "active" ? (
                    <button type="button" onClick={() => void openArtifact(artifact.id)} disabled={busy === "open"}>打开</button>
                  ) : null}
                </div>
                <details className="dataset-artifact-menu">
                  <summary>更多操作</summary>
                  <div>
                    {artifactView === "active" ? (
                      <>
                        <button type="button" onClick={() => void inspectArtifactReadiness(artifact.id)} disabled={busy !== null || !runtimeCompatible}>检查状态</button>
                        <button type="button" onClick={() => void previewArtifactDeletion(artifact.id)} disabled={busy !== null}><Trash size={15} />删除</button>
                        {legacy ? <button type="button" onClick={focusTrustedPath}>定位重新导入入口</button> : null}
                      </>
                    ) : (
                      <>
                        <button type="button" onClick={() => void restoreArtifact(artifact.id)} disabled={busy !== null}><ArrowCounterClockwise size={15} />恢复</button>
                        <button type="button" onClick={() => void previewArtifactDeletion(artifact.id)} disabled={busy !== null}>查看依赖</button>
                      </>
                    )}
                  </div>
                {deletionImpact ? (
                  <section className="dataset-deletion-impact" aria-label={`${artifact.datasetName ?? artifact.id} 删除影响`}>
                    <div className="dataset-deletion-impact__head">
                      <span>
                        <strong>后端删除影响预演</strong>
                        <small>{adminToolsOpen ? `Impact ${deletionImpact.impactHash.slice(0, 16)} · ` : ""}执行时后端会重新扫描引用</small>
                      </span>
                      <button type="button" onClick={() => void previewArtifactDeletion(artifact.id)} disabled={busy !== null}>
                        刷新预演
                      </button>
                    </div>
                    <div className="dataset-deletion-impact__counts">
                      <span>阻断引用 <strong>{deletionImpact.blockers.length}</strong></span>
                      <span>依赖项 <strong>{deletionImpact.dependents.length}</strong></span>
                      <span>保留项 <strong>{deletionImpact.preserved.length}</strong></span>
                    </div>
                    {deletionImpact.blockers.length ? (
                      <div className="dataset-impact-references is-blocking">
                        <strong>永久删除阻断项</strong>
                        <ul>{deletionImpact.blockers.map((reference) => (
                          <li key={`${reference.kind}-${reference.id}`}>
                            {artifactReferenceKindLabel(reference.kind, adminToolsOpen)}{adminToolsOpen ? <> · <code>{reference.id}</code></> : null}
                          </li>
                        ))}</ul>
                      </div>
                    ) : null}
                    {deletionImpact.dependents.length ? (
                      <div className="dataset-impact-references">
                        <strong>已发现依赖</strong>
                        <ul>{deletionImpact.dependents.slice(0, 8).map((reference) => (
                          <li key={`${reference.kind}-${reference.id}`}>
                            {artifactReferenceKindLabel(reference.kind, adminToolsOpen)}{adminToolsOpen ? <> · <code>{reference.id}</code></> : null}{reference.blocking ? " · 阻断" : " · 随数据记录清理"}
                          </li>
                        ))}</ul>
                        {deletionImpact.dependents.length > 8 ? (
                          <small>其余 {deletionImpact.dependents.length - 8} 项已计入 ImpactHash；执行时后端会按完整清单复核。</small>
                        ) : null}
                      </div>
                    ) : null}
                    {deletionImpact.preserved.length ? (
                      <p>
                        保留：{deletionImpact.preserved.slice(0, 6).join("、")}
                        {deletionImpact.preserved.length > 6 ? ` 等 ${deletionImpact.preserved.length} 项` : ""}
                      </p>
                    ) : null}
                    {artifactView === "active" ? (
                      <div className="dataset-impact-actions">
                        <p>移入回收站是可恢复操作；引用不会被静默删除。</p>
                        <button type="button" onClick={() => void trashArtifact(artifact.id)} disabled={busy !== null}>
                          <Trash size={15} />移入回收站
                        </button>
                      </div>
                    ) : (
                      <div className="dataset-purge-confirmation">
                        <label>
                          <span>输入资源 ID 后 8 位 <code>{confirmationSuffix}</code></span>
                          <input
                            aria-label={`输入 ${artifact.datasetName ?? artifact.id} 资源 ID 后 8 位`}
                            value={purgeConfirmation}
                            onChange={(event) => setPurgeConfirmations((current) => ({
                              ...current,
                              [artifact.id]: event.target.value,
                            }))}
                            autoComplete="off"
                            spellCheck={false}
                          />
                        </label>
                        <button
                          className="is-danger"
                          type="button"
                          onClick={() => void purgeArtifact(artifact.id)}
                          disabled={busy !== null || deletionImpact.blockers.length > 0 || purgeConfirmation !== confirmationSuffix}
                        >
                          永久删除
                        </button>
                        {deletionImpact.blockers.length ? <small>存在阻断引用，永久删除不可用。</small> : null}
                      </div>
                    )}
                  </section>
                ) : null}
                </details>
                {adminToolsOpen && legacy && artifactView === "active" ? (
                  <div className="dataset-legacy-note" role="note">
                    <WarningCircle size={17} weight="fill" />
                    <span><strong>旧版记录保持只读</strong>请从原始可信目录或研究包重新导入生成新版数据记录；不会覆盖当前记录。</span>
                  </div>
                ) : null}
                {adminToolsOpen && artifactView === "active" ? <details className="dataset-artifact-technical">
                  <summary>审计信息</summary>
                  {(() => {
                  const contract = artifactContracts[artifact.id];
                  const details = (contract?.artifact ?? artifact) as ArtifactContractView;
                  const auditIdentity = (
                    <div>
                      <dt>审计身份</dt>
                      <dd>内容哈希（contentHash） <code>{details.contentHash ?? "未记录"}</code></dd>
                      <dd>图事实哈希（canonicalGraphHash） <code>{details.canonicalGraphHash}</code></dd>
                      <dd>清单哈希（manifestHash） <code>{details.manifestHash ?? "未记录"}</code></dd>
                      <dd>源校验和 <code>{details.checksum}</code></dd>
                    </div>
                  );
                  if (!contract) {
                    return (
                      <div className="dataset-contract-card dataset-audit-details">
                        <dl className="dataset-contract-grid">{auditIdentity}</dl>
                        <p className="dataset-inline-note">检查状态后可查看数据使用条件与训练准备详情。</p>
                      </div>
                    );
                  }
                  const readiness = contract.readiness;
                  const reference = readiness.trainingRef ?? details.trainingRef;
                  const canResolve = readiness.status === "ready" && Boolean(reference?.splitSetId && reference.taskSpecId);
                  const governance = details.dataGovernance ?? details.preparationSpec?.governance;
                  const evidence = details.licenseEvidence ?? [];
                  return (
                    <div className={`dataset-contract-card is-${readiness.status}`} role="status" aria-label={`${details.datasetName ?? details.id} 训练合同`}>
                      <div className="dataset-contract-card__status">
                        {readiness.status === "ready" ? <CheckCircle size={19} weight="fill" /> : <WarningCircle size={19} weight="fill" />}
                        <span>
                          <strong>{readinessLabel(readiness.status)}</strong>
                          <small>Artifact {details.schemaVersion ?? "1.0"} · TrainingRef {reference?.schemaVersion ?? "未生成"}</small>
                        </span>
                      </div>

                      {details.datasetRole === "target_domain" ? (
                        <p className="dataset-binding-boundary">
                          目标域 Binding 只证明图事实已交接；不代表训练就绪，也不会自动进入基础模型预训练。
                        </p>
                      ) : null}

                      <dl className="dataset-contract-grid">
                        {auditIdentity}
                        <div>
                          <dt>许可证</dt>
                          <dd>{details.licensePolicy?.identifier ?? "未记录"} · {licenseStatusLabel(details.licensePolicy?.status)}</dd>
                          <dd>允许用途：{details.licensePolicy?.allowedUses.length ? details.licensePolicy.allowedUses.join("、") : "无"}</dd>
                          <dd>证据：{evidence.length ? evidence.map((item) => evidenceKindLabel(item.kind)).join("、") : "无可核验证据"}</dd>
                        </div>
                        <div>
                          <dt>数据治理</dt>
                          <dd>{governance?.containsPersonalData ? "含个人数据" : "未标记个人数据"} · {governance?.deidentified ? "已去标识" : "未去标识"}</dd>
                          <dd>保留：{governance ? retentionLabel(governance.retention) : "未声明"}</dd>
                          <dd>{governance?.userDataTrainingOptIn === false ? "基础权重训练未授权" : "训练授权状态未知"}</dd>
                        </div>
                        <div>
                          <dt>准备规则</dt>
                          <dd>{details.preparationSpec ? taskKindLabel(details.preparationSpec.taskKind) : "未提供 PreparationSpec"}</dd>
                          <dd>特征字段 {details.preparationSpec?.featureAttributes.length ?? 0} · 标签 {details.preparationSpec?.labelAttribute ?? "无"}</dd>
                          <dd>Split {details.preparationSpec?.splitStrategy ?? "未指定"} · 排除字段 {details.preparationSpec?.excludedAttributes.length ?? 0}</dd>
                        </div>
                        <div>
                          <dt>训练资产</dt>
                          <dd>特征：{contractItems(details.featureSchemas)}</dd>
                          <dd>标签：{contractItems(details.labelSchemas)}</dd>
                          <dd>Variant：{contractItems(details.graphVariants)}</dd>
                        </div>
                        <div>
                          <dt>评测合同</dt>
                          <dd>SplitSet：{contractItems(details.splitSets)}</dd>
                          <dd>TaskSpec：{contractItems(details.taskSpecs)}</dd>
                          <dd>FeatureRecipe：{contractItems(details.featureRecipes)}</dd>
                        </div>
                        <div>
                          <dt>引用身份</dt>
                          <dd>{reference ? `${reference.graphVariant} · ${reference.intendedUse}` : "尚无候选训练引用"}</dd>
                          <dd>Fold：{reference ? reference.splitFold ?? 0 : "未指定"}</dd>
                          <dd>{contract.resolvedRefHash ? `Ref ${contract.resolvedRefHash.slice(0, 16)}` : "尚未解析 RefHash"}</dd>
                        </div>
                      </dl>

                      {contract.materializedContract ? (
                        <section className="dataset-materialized-contract" aria-label="独立物化训练合同">
                          <div>
                            <strong>只读物化 Bundle 已验证</strong>
                            <span>这是独立训练输入校验，不等同于目标域 Binding，也不代表训练已经启动。</span>
                          </div>
                          <dl>
                            <div><dt>图规模</dt><dd>{contract.materializedContract.nodeCount} 节点 · {contract.materializedContract.edgeCount} 关系</dd></div>
                            <div><dt>特征 shape</dt><dd>{contract.materializedContract.featureShape.join(" × ") || "无"}</dd></div>
                            <div><dt>标签 shape</dt><dd>{contract.materializedContract.labelShape?.join(" × ") || "无"}</dd></div>
                            <div><dt>任务</dt><dd>{taskKindLabel(contract.materializedContract.taskKind)}</dd></div>
                            <div className="is-wide">
                              <dt>Split sizes</dt>
                              <dd>{Object.entries(contract.materializedContract.splitSizes)
                                .map(([name, size]) => `${name} ${size}`)
                                .join(" · ") || "无"}</dd>
                            </div>
                          </dl>
                        </section>
                      ) : null}

                      {readiness.blockers.length ? (
                        <div className="dataset-contract-issues is-blocker">
                          <strong>阻断项（{readiness.blockers.length}）</strong>
                          <ul>{readiness.blockers.map((issue) => <li key={issue.code}><code>{issue.code}</code>{issue.message}</li>)}</ul>
                        </div>
                      ) : null}
                      {readiness.warnings.length ? (
                        <div className="dataset-contract-issues">
                          <strong>警告（{readiness.warnings.length}）</strong>
                          <ul>{readiness.warnings.map((issue) => <li key={issue.code}><code>{issue.code}</code>{issue.message}</li>)}</ul>
                        </div>
                      ) : null}

                      {canResolve ? (
                        <button type="button" onClick={() => void resolveArtifactTrainingRef(artifact.id)} disabled={busy !== null}>
                          生成 TrainingDatasetRef {reference?.schemaVersion ?? "1.1"}
                        </button>
                      ) : readiness.status === "ready" ? (
                        <p className="dataset-inline-note">Readiness 已通过，但未提供完整 SplitSet/TaskSpec 候选，暂不能生成训练引用。</p>
                      ) : null}
                    </div>
                  );
                  })()}
                </details> : null}
              </article>
            );
          }) : (
            <div className="dataset-empty"><Database size={22} weight="light" /><span>{busy === "loading" ? "正在读取数据…" : artifactView === "trashed" ? "回收站为空" : "还没有已转换的科研数据"}</span></div>
          )}
          {unnamedArtifacts.length ? (
            <button
              className="dataset-unnamed-toggle"
              type="button"
              onClick={() => setShowUnnamedArtifacts((current) => !current)}
            >
              {showUnnamedArtifacts ? "隐藏未命名验证产物" : `显示 ${unnamedArtifacts.length} 个未命名验证产物`}
            </button>
          ) : null}
            </>
          );
          return (
            <>
              <div
                className="dataset-artifact-list"
                id="dataset-artifacts-active"
                role="tabpanel"
                aria-labelledby="dataset-tab-active"
                hidden={artifactView !== "active"}
              >
                {artifactView === "active" ? artifactListContent : null}
              </div>
              <div
                className="dataset-artifact-list"
                id="dataset-artifacts-trashed"
                role="tabpanel"
                aria-labelledby="dataset-tab-trashed"
                hidden={artifactView !== "trashed"}
              >
                {artifactView === "trashed" ? artifactListContent : null}
              </div>
            </>
          );
        })()}
      </section>
    </div>
  );
}
