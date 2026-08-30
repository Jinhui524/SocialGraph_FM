import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  CURRENT_BROWSER_IMPORT_PIPELINE_VERSION,
  inspectGraphVersionCompatibility,
  type GraphVersionCompatibility,
} from "../services/graphVersionCompatibility";
import { diffGraphVersionsById } from "../services/graphVersionDiff";
import { downloadSourceArtifact } from "../services/sourceArtifact";
import type {
  DeletionImpact,
  GraphRepository,
  GraphVersion,
  GraphVersionDiffReport,
  GraphVersionManifest,
  ManagedResourceKind,
  ResourceLifecycleState,
  SourceArtifact,
} from "../types/graph";
import "../version-lifecycle-panel.css";

export interface VersionLifecyclePanelProps {
  readonly repository: GraphRepository;
  readonly currentGraphVersionId?: string;
  readonly onActivateVersion: (version: GraphVersion) => Promise<void> | void;
  readonly onRequestRebuild?: (
    version: GraphVersion,
    compatibility: GraphVersionCompatibility,
  ) => Promise<void> | void;
  readonly onNotify?: (message: string) => void;
}

type LifecycleTab = ResourceLifecycleState;

interface FlatVersionTreeItem {
  readonly manifest: GraphVersionManifest;
  readonly depth: number;
}

interface LifecycleTarget {
  readonly kind: ManagedResourceKind;
  readonly id: string;
  readonly label: string;
}

interface CompatibilityDetail {
  readonly version: GraphVersion;
  readonly result: GraphVersionCompatibility;
}

function formatBytes(bytes: number): string {
  if (bytes < 1_024) return `${bytes} B`;
  if (bytes < 1_048_576) return `${(bytes / 1_024).toFixed(1)} KB`;
  return `${(bytes / 1_048_576).toFixed(1)} MB`;
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function shortHash(value?: string): string {
  return value ? `${value.slice(0, 12)}…` : "legacy";
}

function confirmationSuffix(id: string): string {
  return id.slice(-8);
}

function versionCompatibilityHint(manifest: GraphVersionManifest): {
  readonly tone: "pipeline" | "legacy" | "research";
  readonly label: string;
} {
  if (manifest.datasetArtifactId || manifest.provenance?.origin === "research_dataset") {
    return { tone: "research", label: "研究数据投影" };
  }
  if (
    manifest.provenance?.pipeline === "browser-import"
    && manifest.provenance.pipelineVersion === CURRENT_BROWSER_IMPORT_PIPELINE_VERSION
  ) {
    return { tone: "pipeline", label: "当前解析管线" };
  }
  return {
    tone: "legacy",
    label: manifest.provenance ? "旧解析管线" : "兼容版本",
  };
}

function compatibilityStatusLabel(status: GraphVersionCompatibility["status"]): string {
  switch (status) {
    case "current": return "当前解析管线";
    case "research_dataset": return "研究数据 Artifact 投影";
    case "upgrade_available": return "可从原始文件创建重建子版本";
    case "source_missing": return "重建所需源文件不完整";
    case "legacy_read_only": return "兼容只读版本";
  }
}

function buildVersionTree(manifests: readonly GraphVersionManifest[]): readonly FlatVersionTreeItem[] {
  const byId = new Map(manifests.map((manifest) => [manifest.id, manifest]));
  const children = new Map<string, GraphVersionManifest[]>();
  for (const manifest of manifests) {
    if (!manifest.parentVersionId || !byId.has(manifest.parentVersionId)) continue;
    const siblings = children.get(manifest.parentVersionId) ?? [];
    siblings.push(manifest);
    children.set(manifest.parentVersionId, siblings);
  }
  const compare = (left: GraphVersionManifest, right: GraphVersionManifest) =>
    right.createdAt.localeCompare(left.createdAt) || left.id.localeCompare(right.id);
  for (const siblings of children.values()) siblings.sort(compare);
  const roots = manifests
    .filter((manifest) => !manifest.parentVersionId || !byId.has(manifest.parentVersionId))
    .sort(compare);
  const result: FlatVersionTreeItem[] = [];
  const visited = new Set<string>();
  const visit = (manifest: GraphVersionManifest, depth: number) => {
    if (visited.has(manifest.id)) return;
    visited.add(manifest.id);
    result.push({ manifest, depth });
    for (const child of children.get(manifest.id) ?? []) visit(child, depth + 1);
  };
  for (const root of roots) visit(root, 0);
  // A malformed legacy cycle must remain visible instead of disappearing.
  for (const manifest of [...manifests].sort(compare)) visit(manifest, 0);
  return result;
}

function defaultDiffPair(
  manifest: GraphVersionManifest,
  manifests: readonly GraphVersionManifest[],
  currentGraphVersionId?: string,
): readonly [string, string] | null {
  if (manifest.parentVersionId) return [manifest.parentVersionId, manifest.id];
  if (currentGraphVersionId && currentGraphVersionId !== manifest.id) {
    return [manifest.id, currentGraphVersionId];
  }
  const child = manifests
    .filter((candidate) => candidate.parentVersionId === manifest.id)
    .sort((left, right) => left.createdAt.localeCompare(right.createdAt) || left.id.localeCompare(right.id))[0];
  return child ? [manifest.id, child.id] : null;
}

function exportJson(report: GraphVersionDiffReport): void {
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `graph-diff-${report.fromVersionId.slice(-8)}-${report.toVersionId.slice(-8)}.json`;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function errorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}

export function VersionLifecyclePanel({
  repository,
  currentGraphVersionId,
  onActivateVersion,
  onRequestRebuild,
  onNotify,
}: VersionLifecyclePanelProps) {
  const refreshSequence = useRef(0);
  const [tab, setTab] = useState<LifecycleTab>("active");
  const [manifests, setManifests] = useState<readonly GraphVersionManifest[]>([]);
  const [artifacts, setArtifacts] = useState<readonly SourceArtifact[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [diffReport, setDiffReport] = useState<GraphVersionDiffReport | null>(null);
  const [compatibility, setCompatibility] = useState<CompatibilityDetail | null>(null);
  const [impactTarget, setImpactTarget] = useState<LifecycleTarget | null>(null);
  const [impact, setImpact] = useState<DeletionImpact | null>(null);
  const [purgeConfirmation, setPurgeConfirmation] = useState("");

  const refresh = useCallback(async () => {
    const sequence = ++refreshSequence.current;
    setLoading(true);
    try {
      const [nextManifests, nextArtifacts] = await Promise.all([
        repository.listGraphVersionManifests(tab),
        repository.listSourceArtifacts(tab),
      ]);
      if (sequence !== refreshSequence.current) return;
      setManifests(nextManifests);
      setArtifacts(nextArtifacts);
      setError(null);
    } catch (reason) {
      if (sequence === refreshSequence.current) {
        setError(errorMessage(reason, "无法读取版本生命周期数据"));
      }
    } finally {
      if (sequence === refreshSequence.current) setLoading(false);
    }
  }, [repository, tab]);

  useEffect(() => {
    void refresh();
    const unsubscribe = repository.subscribe(() => void refresh());
    return () => {
      refreshSequence.current += 1;
      unsubscribe();
    };
  }, [refresh, repository]);

  const tree = useMemo(() => buildVersionTree(manifests), [manifests]);
  const manifestById = useMemo(
    () => new Map(manifests.map((manifest) => [manifest.id, manifest])),
    [manifests],
  );

  const notify = (message: string) => {
    setStatus(message);
    onNotify?.(message);
  };

  const activateVersion = async (manifest: GraphVersionManifest) => {
    setBusy(`activate:${manifest.id}`);
    setError(null);
    try {
      const version = await repository.getGraphVersion(manifest.id);
      if (!version) throw new Error(`GRAPH_VERSION_NOT_FOUND：${manifest.id}`);
      await onActivateVersion(version);
      notify(`已将 ${manifest.sourceFile} 设为当前图版本`);
    } catch (reason) {
      setError(errorMessage(reason, "无法切换当前图版本"));
    } finally {
      setBusy(null);
    }
  };

  const inspectCompatibility = async (manifest: GraphVersionManifest) => {
    setBusy(`compatibility:${manifest.id}`);
    setError(null);
    try {
      const version = await repository.getGraphVersion(manifest.id);
      if (!version) throw new Error(`GRAPH_VERSION_NOT_FOUND：${manifest.id}`);
      setCompatibility({
        version,
        result: await inspectGraphVersionCompatibility(repository, version),
      });
    } catch (reason) {
      setError(errorMessage(reason, "无法检查版本兼容性"));
    } finally {
      setBusy(null);
    }
  };

  const requestRebuild = async () => {
    if (!compatibility || !onRequestRebuild) return;
    setBusy(`rebuild:${compatibility.version.id}`);
    setError(null);
    try {
      await onRequestRebuild(compatibility.version, compatibility.result);
      notify("已提交重建子版本请求；旧版本保持不变");
      setCompatibility(null);
    } catch (reason) {
      setError(errorMessage(reason, "无法创建重建子版本"));
    } finally {
      setBusy(null);
    }
  };

  const openDiff = async (manifest: GraphVersionManifest) => {
    const pair = defaultDiffPair(manifest, manifests, currentGraphVersionId);
    if (!pair) return;
    setBusy(`diff:${manifest.id}`);
    setError(null);
    try {
      const report = await diffGraphVersionsById(repository, pair[0], pair[1]);
      if (
        !report.sameLineage
        && !window.confirm("这两个 GraphVersion 不属于同一父子链或同一 sourceHash。仍要执行跨来源比较吗？")
      ) return;
      setDiffReport(report);
    } catch (reason) {
      setError(errorMessage(reason, "无法比较图版本"));
    } finally {
      setBusy(null);
    }
  };

  const openImpact = async (target: LifecycleTarget) => {
    setImpactTarget(target);
    setImpact(null);
    setPurgeConfirmation("");
    setBusy(`impact:${target.kind}:${target.id}`);
    setError(null);
    try {
      const nextImpact = target.kind === "graph_version"
        ? await repository.inspectGraphVersionDeletion(target.id)
        : await repository.inspectSourceArtifactDeletion(target.id);
      setImpact(nextImpact);
    } catch (reason) {
      setImpactTarget(null);
      setError(errorMessage(reason, "无法预演删除引用"));
    } finally {
      setBusy(null);
    }
  };

  const closeImpact = () => {
    setImpactTarget(null);
    setImpact(null);
    setPurgeConfirmation("");
  };

  const trashTarget = async () => {
    if (!impactTarget || !impact?.canTrash) return;
    setBusy(`trash:${impactTarget.kind}:${impactTarget.id}`);
    setError(null);
    try {
      if (impactTarget.kind === "graph_version") {
        await repository.trashGraphVersion(impactTarget.id, impact.impactHash);
      } else {
        await repository.trashSourceArtifact(impactTarget.id, impact.impactHash);
      }
      notify(`${impactTarget.label} 已移入回收站`);
      closeImpact();
      await refresh();
    } catch (reason) {
      setError(errorMessage(reason, "移入回收站失败"));
    } finally {
      setBusy(null);
    }
  };

  const restoreTarget = async () => {
    if (!impactTarget || impact?.state !== "trashed") return;
    setBusy(`restore:${impactTarget.kind}:${impactTarget.id}`);
    setError(null);
    try {
      if (impactTarget.kind === "graph_version") {
        await repository.restoreGraphVersion(impactTarget.id);
      } else {
        await repository.restoreSourceArtifact(impactTarget.id);
      }
      notify(`${impactTarget.label} 已从回收站恢复`);
      closeImpact();
      await refresh();
    } catch (reason) {
      setError(errorMessage(reason, "恢复失败"));
    } finally {
      setBusy(null);
    }
  };

  const purgeTarget = async () => {
    if (!impactTarget || !impact?.canPurge) return;
    if (purgeConfirmation !== confirmationSuffix(impactTarget.id)) return;
    setBusy(`purge:${impactTarget.kind}:${impactTarget.id}`);
    setError(null);
    try {
      if (impactTarget.kind === "graph_version") {
        await repository.purgeGraphVersion(impactTarget.id, impact.impactHash);
      } else {
        await repository.purgeSourceArtifact(impactTarget.id, impact.impactHash);
      }
      notify(`${impactTarget.label} 已永久删除，无法恢复`);
      closeImpact();
      await refresh();
    } catch (reason) {
      setError(errorMessage(reason, "永久删除失败"));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className="version-lifecycle-panel" aria-labelledby="version-lifecycle-title" aria-busy={loading}>
      <header className="version-lifecycle-panel__header">
        <div>
          <h2 id="version-lifecycle-title">版本与源文件</h2>
          <p>GraphVersion 保持不可变；回收、恢复与永久删除只通过生命周期侧写执行。</p>
        </div>
        <span className={`version-lifecycle-panel__storage is-${repository.storageMode}`}>
          {repository.storageMode === "indexeddb" ? "IndexedDB 持久化" : "会话内存存储"}
        </span>
      </header>

      <div className="version-lifecycle-tabs" role="tablist" aria-label="生命周期状态">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "active"}
          className={tab === "active" ? "is-active" : ""}
          onClick={() => setTab("active")}
        >有效资源</button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "trashed"}
          className={tab === "trashed" ? "is-active" : ""}
          onClick={() => setTab("trashed")}
        >回收站</button>
        <button type="button" className="version-lifecycle-refresh" onClick={() => void refresh()} disabled={loading}>
          刷新
        </button>
      </div>

      {error ? <div className="version-lifecycle-alert is-error" role="alert">{error}</div> : null}
      {status ? <div className="version-lifecycle-alert is-success" role="status">{status}</div> : null}

      <section className="version-lifecycle-section" aria-labelledby="graph-version-history-title">
        <div className="version-lifecycle-section__title">
          <div>
            <h3 id="graph-version-history-title">GraphVersion 历史</h3>
            <p>子版本缩进展示；比较默认使用父版本作为基线。</p>
          </div>
          <span>{manifests.length}</span>
        </div>
        <div className="version-tree" role="list" aria-label={`${tab === "active" ? "有效" : "已回收"}图版本`}>
          {tree.length ? tree.map(({ manifest, depth }) => {
            const hint = versionCompatibilityHint(manifest);
            const diffPair = defaultDiffPair(manifest, manifests, currentGraphVersionId);
            const parent = manifest.parentVersionId ? manifestById.get(manifest.parentVersionId) : undefined;
            const isCurrent = manifest.id === currentGraphVersionId;
            return (
              <article
                className={`version-tree-item ${isCurrent ? "is-current" : ""}`}
                style={{ "--version-indent": `${Math.min(depth, 8) * 22}px` } as React.CSSProperties}
                key={manifest.id}
                role="listitem"
              >
                <span className="version-tree-item__rail" aria-hidden="true" />
                <div className="version-tree-item__body">
                  <div className="version-tree-item__heading">
                    <strong>{manifest.sourceFile}</strong>
                    {isCurrent ? <span className="version-badge is-current">当前</span> : null}
                    <span className={`version-badge is-${hint.tone}`}>{hint.label}</span>
                  </div>
                  <span className="version-tree-item__meta">
                    {manifest.nodeCount} 节点 · {manifest.edgeCount} 关系 · {manifest.directedness} · {formatDate(manifest.createdAt)}
                  </span>
                  <code title={manifest.id}>{manifest.id}</code>
                  <span className="version-tree-item__lineage">
                    {manifest.parentVersionId
                      ? `父版本：${parent?.sourceFile ?? manifest.parentVersionId}`
                      : "根版本"}
                    {` · content ${shortHash(manifest.contentHash)}`}
                  </span>
                </div>
                <div className="version-tree-item__actions">
                  {tab === "active" ? (
                    <button
                      type="button"
                      disabled={isCurrent || busy !== null}
                      onClick={() => void activateVersion(manifest)}
                      aria-label={`设为当前 ${manifest.sourceFile}`}
                    >{isCurrent ? "当前版本" : "设为当前"}</button>
                  ) : null}
                  <button
                    type="button"
                    disabled={busy !== null}
                    onClick={() => void inspectCompatibility(manifest)}
                    aria-label={`检查兼容性 ${manifest.sourceFile}`}
                  >兼容性</button>
                  <button
                    type="button"
                    disabled={!diffPair || busy !== null}
                    onClick={() => void openDiff(manifest)}
                    aria-label={`比较版本 ${manifest.sourceFile}`}
                  >比较</button>
                  <button
                    type="button"
                    disabled={busy !== null}
                    onClick={() => void openImpact({ kind: "graph_version", id: manifest.id, label: manifest.sourceFile })}
                    aria-label={`查看引用与删除 ${manifest.sourceFile}`}
                  >引用与删除</button>
                </div>
              </article>
            );
          }) : (
            <div className="version-lifecycle-empty">{loading ? "正在读取版本索引…" : "此状态下没有 GraphVersion"}</div>
          )}
        </div>
      </section>

      <section className="version-lifecycle-section" aria-labelledby="source-artifact-history-title">
        <div className="version-lifecycle-section__title">
          <div>
            <h3 id="source-artifact-history-title">SourceArtifact</h3>
            <p>原始 Blob 与 SHA-256 独立管理；被版本或消息引用时不会静默删除。</p>
          </div>
          <span>{artifacts.length}</span>
        </div>
        <div className="source-artifact-list" role="list" aria-label={`${tab === "active" ? "有效" : "已回收"}源文件`}>
          {artifacts.length ? artifacts.map((artifact) => (
            <article className="source-artifact-item" key={artifact.id} role="listitem">
              <div>
                <strong>{artifact.name}</strong>
                <span>{artifact.role} · {artifact.format.toUpperCase()} · {formatBytes(artifact.size)} · {formatDate(artifact.createdAt)}</span>
                <code title={artifact.sha256}>SHA-256 {artifact.sha256}</code>
                <small title={artifact.id}>{artifact.id}</small>
              </div>
              <div className="source-artifact-item__actions">
                <button type="button" onClick={() => downloadSourceArtifact(artifact)}>导出原文件</button>
                <button
                  type="button"
                  disabled={busy !== null}
                  onClick={() => void openImpact({ kind: "source_artifact", id: artifact.id, label: artifact.name })}
                  aria-label={`查看引用与删除 ${artifact.name}`}
                >引用与删除</button>
              </div>
            </article>
          )) : (
            <div className="version-lifecycle-empty">{loading ? "正在读取源文件索引…" : "此状态下没有 SourceArtifact"}</div>
          )}
        </div>
      </section>

      {diffReport ? (
        <section className="version-lifecycle-dialog" role="dialog" aria-modal="true" aria-labelledby="version-diff-title">
          <div className="version-lifecycle-dialog__surface is-wide">
            <header>
              <div>
                <h3 id="version-diff-title">版本差异</h3>
                <code>{diffReport.fromVersionId} → {diffReport.toVersionId}</code>
              </div>
              <button type="button" onClick={() => setDiffReport(null)} aria-label="关闭版本差异">关闭</button>
            </header>
            <div className="version-diff-summary">
              <div><span>节点</span><strong>+{diffReport.summary.nodes.added} / −{diffReport.summary.nodes.removed} / Δ{diffReport.summary.nodes.modified}</strong></div>
              <div><span>关系事实</span><strong>+{diffReport.summary.edges.added} / −{diffReport.summary.edges.removed} / Δ{diffReport.summary.edges.modified}</strong></div>
              <div><span>edge ID churn</span><strong>{diffReport.edgeIdChurn.count}</strong></div>
              <div><span>内容哈希</span><strong>{diffReport.sameContent ? "完全一致" : "不同"}</strong></div>
            </div>
            {!diffReport.sameLineage ? (
              <p className="compatibility-warning">这是经确认执行的跨来源比较；差异可能同时包含数据来源和构图规则变化。</p>
            ) : null}
            {diffReport.edgeIdChurn.count ? (
              <p className="version-diff-note">关系事实保持匹配，但有 {diffReport.edgeIdChurn.count} 个内部 edge ID 发生变化；常见原因是 CSV 行重排。</p>
            ) : null}
            <div className="version-diff-fields">
              <span>版本字段 {diffReport.versionFields.length}</span>
              <span>构图规则字段 {diffReport.buildSpecFields.length}</span>
              <span>事实差异项 {diffReport.samples.length}{diffReport.truncated ? "+" : ""}</span>
            </div>
            {diffReport.samples.length ? (
              <ol className="version-diff-samples">
                {diffReport.samples.slice(0, 20).map((sample, index) => (
                  <li key={`${sample.entity}-${sample.id}-${sample.kind}-${index}`}>
                    <code>{sample.entity}:{sample.id}</code>
                    <span>{sample.kind} · {sample.fields.map((field) => field.field).join("、")}</span>
                  </li>
                ))}
              </ol>
            ) : <p className="version-diff-note">没有节点或关系事实变化。</p>}
            <footer>
              <button type="button" onClick={() => exportJson(diffReport)}>导出差异 JSON</button>
            </footer>
          </div>
        </section>
      ) : null}

      {compatibility ? (
        <section className="version-lifecycle-dialog" role="dialog" aria-modal="true" aria-labelledby="version-compatibility-title">
          <div className="version-lifecycle-dialog__surface">
            <header>
              <div>
                <h3 id="version-compatibility-title">版本兼容性</h3>
                <strong>{compatibility.version.sourceFile}</strong>
              </div>
              <button type="button" onClick={() => setCompatibility(null)} aria-label="关闭版本兼容性">关闭</button>
            </header>
            <div className={`compatibility-status is-${compatibility.result.status}`}>
              <strong>{compatibilityStatusLabel(compatibility.result.status)}</strong>
              {compatibility.result.message ? <p>{compatibility.result.message}</p> : null}
            </div>
            {compatibility.result.allNodesUntyped ? (
              <p className="compatibility-warning">此版本所有节点均未保存类型，因此节点会统一显示为“未分类”颜色。这是旧解析结果，不是渲染器丢失配色。</p>
            ) : null}
            {compatibility.result.missingSourceArtifactIds.length ? (
              <details>
                <summary>缺失的 SourceArtifact（{compatibility.result.missingSourceArtifactIds.length}）</summary>
                <ul>{compatibility.result.missingSourceArtifactIds.map((id) => <li key={id}><code>{id}</code></li>)}</ul>
              </details>
            ) : null}
            {compatibility.result.canDeterministicallyRebuild ? (
              <p className="compatibility-guidance">重建会读取同一 SourceArtifact，以当前字段语义创建带 parentVersionId 的新版本；不会覆盖此版本。</p>
            ) : null}
            <footer>
              {compatibility.result.canDeterministicallyRebuild && onRequestRebuild ? (
                <button type="button" onClick={() => void requestRebuild()} disabled={busy !== null}>用当前规则创建重建子版本</button>
              ) : null}
            </footer>
          </div>
        </section>
      ) : null}

      {impactTarget ? (
        <section className="version-lifecycle-dialog" role="dialog" aria-modal="true" aria-labelledby="deletion-impact-title">
          <div className="version-lifecycle-dialog__surface">
            <header>
              <div>
                <h3 id="deletion-impact-title">引用预演</h3>
                <strong>{impactTarget.label}</strong>
                <code>{impactTarget.id}</code>
              </div>
              <button type="button" onClick={closeImpact} aria-label="关闭引用预演">关闭</button>
            </header>
            {!impact ? <p className="version-diff-note">正在计算引用集合与 impactHash…</p> : (
              <>
                <div className="deletion-impact-summary">
                  <span>状态：{impact.state === "active" ? "有效" : "回收站"}</span>
                  <span>引用：{impact.references.length}</span>
                  <span>随目标清理：{impact.dependents.reduce((sum, group) => sum + group.count, 0)}</span>
                </div>
                {impact.references.length ? (
                  <div className="deletion-impact-block">
                    <h4>引用关系</h4>
                    <ul>{impact.references.map((reference) => (
                      <li key={`${reference.kind}-${reference.id}`}>
                        <span><strong>{reference.label}</strong><code>{reference.kind} · {reference.id}</code></span>
                        <em>{reference.blocksPurge ? "阻止永久删除" : "保留引用"}</em>
                      </li>
                    ))}</ul>
                  </div>
                ) : <p className="version-diff-note">没有外部引用。</p>}
                {impact.dependents.length ? (
                  <details>
                    <summary>永久删除时一并清理的派生记录</summary>
                    <ul>{impact.dependents.map((dependent) => (
                      <li key={dependent.kind}>{dependent.kind}：{dependent.count}</li>
                    ))}</ul>
                  </details>
                ) : null}
                {impact.retainedDependencies.length ? (
                  <details>
                    <summary>永久删除后仍保留的资源</summary>
                    <ul>{impact.retainedDependencies.map((dependency) => (
                      <li key={`${dependency.kind}-${dependency.id}`}>{dependency.kind}：{dependency.label}</li>
                    ))}</ul>
                  </details>
                ) : null}
                <code className="impact-hash" title={impact.impactHash}>impactHash {impact.impactHash}</code>
                {impact.state === "active" ? (
                  <footer>
                    <button type="button" className="is-danger" disabled={!impact.canTrash || busy !== null} onClick={() => void trashTarget()}>
                      移入回收站
                    </button>
                    {!impact.canTrash ? <span>请先解除阻断引用；系统不会级联删除事实资源。</span> : null}
                  </footer>
                ) : (
                  <div className="permanent-delete-zone">
                    <button type="button" onClick={() => void restoreTarget()} disabled={busy !== null}>从回收站恢复</button>
                    <label>
                      <span>输入 ID 后 8 位 <code>{confirmationSuffix(impactTarget.id)}</code> 确认永久删除</span>
                      <input
                        value={purgeConfirmation}
                        onChange={(event) => setPurgeConfirmation(event.target.value)}
                        aria-label="输入 ID 后 8 位确认永久删除"
                        autoComplete="off"
                        spellCheck={false}
                      />
                    </label>
                    <button
                      type="button"
                      className="is-danger"
                      disabled={
                        !impact.canPurge
                        || purgeConfirmation !== confirmationSuffix(impactTarget.id)
                        || busy !== null
                      }
                      onClick={() => void purgeTarget()}
                    >永久删除</button>
                    {!impact.canPurge ? <p>仍有引用或子版本；请按引用关系从叶子版本开始清理。</p> : null}
                  </div>
                )}
              </>
            )}
          </div>
        </section>
      ) : null}
    </section>
  );
}
