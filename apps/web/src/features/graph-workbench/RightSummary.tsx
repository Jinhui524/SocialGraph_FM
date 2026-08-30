import { Graph, Sparkle, WarningCircle } from "@phosphor-icons/react";
import type { GraphNode, GraphScene, GraphVersion, GraphViewState } from "../../types/graph";

const INTERNAL_GOVERNANCE_SOURCE = /(?:\bgovernance-artifact-[0-9a-f]{32}\b|\banswer[\s_-]*pack\b|\b(?:russia|cuba|uae|venezuela|iran|china)(?:[\s_.-]|$))/iu;

export function publicGraphSourceLabel(graph: {
  readonly sourceFile: string;
  readonly datasetArtifact?: { readonly datasetName: string };
}): string {
  const candidates = [graph.datasetArtifact?.datasetName, graph.sourceFile]
    .map((value) => value?.trim() ?? "")
    .filter(Boolean);
  return candidates.find((value) => !INTERNAL_GOVERNANCE_SOURCE.test(value))
    ?? "当前会话治理图";
}

export function RightSummary({
  graph,
  selectedNode,
  viewState,
  scene,
  onExport,
}: {
  graph: GraphVersion | null;
  selectedNode: GraphNode | null;
  viewState: GraphViewState | null;
  scene: GraphScene | null;
  onExport?: () => void;
}) {
  if (!graph) {
    return (
      <section className="right-empty">
        <span className="right-empty__icon"><Graph size={28} weight="light" /></span>
        <strong>等待图数据</strong>
        <p>上传 CSV / TSV、JSON、GraphML 或 GEXF 后，这里会显示真实图谱、指标与质量信息。</p>
      </section>
    );
  }
  const summary = graph.summary;
  const warnings = graph.issues.filter((issue) => issue.severity !== "info");
  const hasActiveScopeFilters = Boolean(
    viewState
      && (
        viewState.filters.nodeTypes.length > 0
        || viewState.filters.edgeTypes.length > 0
        || Boolean(viewState.filters.timeRange)
        || viewState.filters.minWeight !== undefined
        || viewState.filters.maxWeight !== undefined
        || viewState.filters.directed !== undefined
        || Boolean(viewState.filters.emptyReason)
      ),
  );
  const showCurrentView = Boolean(
    viewState
      && scene
      && (
        viewState.theme !== "brand-light"
        || hasActiveScopeFilters
        || Object.keys(viewState.pinnedNodes).length > 0
        || scene.truncated
      ),
  );
  const publicSourceLabel = publicGraphSourceLabel(graph);
  return (
    <>
      <section className="summary-card" aria-labelledby="graph-summary-title">
        <div className="card-title-row">
          <div><strong id="graph-summary-title">图谱摘要</strong><span className="live-dot">真实计算</span></div>
          <button className="ghost-button" type="button" onClick={onExport} disabled={!onExport}>导出图像</button>
        </div>
        <div className="metric-grid">
          <div><small>实体总数</small><strong>{summary.nodeCount.toLocaleString()}</strong></div>
          <div><small>关系总数</small><strong>{summary.edgeCount.toLocaleString()}</strong></div>
          <div><small>密度</small><strong>{summary.density.toFixed(3)}</strong></div>
          <div><small>平均度</small><strong>{summary.averageDegree.toFixed(2)}</strong></div>
        </div>
        <div className="finding-block">
          <div className="finding-block__heading">
            <Sparkle size={17} weight="fill" />
            <strong>结构观察</strong>
            <span className="source-chip">本地算法</span>
          </div>
          <p>
            网络包含 {summary.connectedComponents} 个连通分量
            {summary.isolatedNodes ? `，其中 ${summary.isolatedNodes} 个孤立节点` : "，没有孤立节点"}。
          </p>
        </div>
      </section>

      {showCurrentView && viewState && scene ? (
        <section className="view-state-card" aria-label="当前图谱视图">
          <div className="card-title-row">
            <strong>当前视图</strong>
            <span>{viewState.theme === "focus-dark" ? "沉浸深色" : "品牌浅色"}</span>
          </div>
          <div className="view-state-card__stats">
            <span>
              <small>分析范围</small>
              <strong>{scene.originalNodeCount.toLocaleString()} / {scene.originalEdgeCount.toLocaleString()}</strong>
            </span>
            <span>
              <small>画布展示</small>
              <strong>{scene.visibleNodeCount.toLocaleString()} / {scene.visibleEdgeCount.toLocaleString()}</strong>
            </span>
            <span><small>关系保留</small><strong>{summary.edgeCount > 0 ? `${Math.round((scene.visibleEdgeCount / summary.edgeCount) * 100)}%` : "—"}</strong></span>
          </div>
          {viewState.filters.minWeight !== undefined || viewState.filters.maxWeight !== undefined ? (
            <p className="view-state-card__notice">
              关系权重：{viewState.filters.minWeight ?? "不限"} – {viewState.filters.maxWeight ?? "不限"}
            </p>
          ) : null}
          {viewState.filters.directed !== undefined ? (
            <p className="view-state-card__notice">
              关系方向：{viewState.filters.directed ? "仅有向关系" : "仅无向关系"}
            </p>
          ) : null}
          {viewState.filters.emptyReason ? (
            <p className="view-state-card__notice"><WarningCircle size={14} weight="fill" />筛选条件与当前图的可验证元数据冲突，本次分析范围为空。</p>
          ) : null}
          {scene.truncated ? <p className="view-state-card__notice"><WarningCircle size={14} weight="fill" />当前为确定性局部预览，完整图指标保持不变。</p> : null}
        </section>
      ) : null}

      {selectedNode ? (
        <section className="node-detail-card" aria-live="polite">
          <div className="card-title-row"><strong>已选节点</strong><span className="entity-chip">{selectedNode.type || "未分类"}</span></div>
          <div className="selected-node-row">
            <span className="selected-node-avatar">{selectedNode.label.slice(0, 1).toUpperCase()}</span>
            <div><strong>{selectedNode.label}</strong><small>ID · {selectedNode.id}</small></div>
          </div>
          {Object.keys(selectedNode.attributes).length ? (
            <dl className="node-attributes">
              {Object.entries(selectedNode.attributes).slice(0, 4).map(([key, value]) => (
                <div key={key}><dt>{key}</dt><dd>{Array.isArray(value) ? value.join("、") : String(value)}</dd></div>
              ))}
            </dl>
          ) : <p className="muted-note">文件未提供额外节点属性。</p>}
        </section>
      ) : null}

      <section className="version-card">
        <div className="card-title-row">
          <strong>数据源</strong>
        </div>
        <div className="version-card__row">
          <span className="version-icon"><Graph size={18} weight="light" /></span>
          <span>
            <strong>{publicSourceLabel}</strong>
            <small>已就绪</small>
          </span>
        </div>
        {warnings.length ? <div className="quality-line has-warning">
          <WarningCircle size={16} weight="fill" />
          <span>{warnings.length} 条非阻断质量提示</span>
        </div> : null}
      </section>
    </>
  );
}
