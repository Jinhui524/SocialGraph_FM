import { ArrowRight, Graph, Shuffle } from "@phosphor-icons/react";
import { useMemo, useState } from "react";

import type {
  ColumnMapping,
  FileProfile,
  GraphTimeFormat,
  NodeColumnMapping,
  ValidationIssue,
} from "../types/graph";

interface ColumnSelectProps {
  readonly label: string;
  readonly required?: boolean;
  readonly value: string;
  readonly profile: FileProfile;
  readonly placeholder: string;
  readonly onChange: (value: string) => void;
}

function ColumnSelect({
  label,
  required = false,
  value,
  profile,
  placeholder,
  onChange,
}: ColumnSelectProps) {
  const profileByName = useMemo(
    () => new Map((profile.columns ?? []).map((column) => [column.name, column])),
    [profile.columns],
  );
  return (
    <label>
      <span>{label} {required ? <em>必选</em> : <small>可选</small>}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{placeholder}</option>
        {profile.headers.map((header) => {
          const column = profileByName.get(header);
          const detail = column
            ? ` · ${column.inferredType} · 缺失 ${(column.missingRate * 100).toFixed(1)}% · ${column.cardinality} 值`
            : "";
          return <option value={header} key={header}>{header}{detail}</option>;
        })}
      </select>
    </label>
  );
}

export interface GraphImportMappingValue {
  readonly nodeMapping?: NodeColumnMapping;
  readonly edgeMapping: ColumnMapping;
  readonly timeFormat: GraphTimeFormat;
}

export function GraphImportMappingCard({
  nodeProfile,
  edgeProfile,
  initialNodeMapping,
  initialEdgeMapping,
  initialTimeFormat,
  issues,
  onApply,
  onCancel,
}: {
  readonly nodeProfile?: FileProfile;
  readonly edgeProfile: FileProfile;
  readonly initialNodeMapping?: Partial<NodeColumnMapping>;
  readonly initialEdgeMapping?: Partial<ColumnMapping>;
  readonly initialTimeFormat: GraphTimeFormat;
  readonly issues: readonly ValidationIssue[];
  readonly onApply: (value: GraphImportMappingValue) => void;
  readonly onCancel: () => void;
}) {
  const [nodeId, setNodeId] = useState(initialNodeMapping?.id ?? "");
  const [nodeLabel, setNodeLabel] = useState(initialNodeMapping?.label ?? "");
  const [nodeType, setNodeType] = useState(initialNodeMapping?.type ?? "");
  const [source, setSource] = useState(initialEdgeMapping?.source ?? "");
  const [target, setTarget] = useState(initialEdgeMapping?.target ?? "");
  const [sourceLabel, setSourceLabel] = useState(nodeProfile ? "" : initialEdgeMapping?.sourceLabel ?? "");
  const [targetLabel, setTargetLabel] = useState(nodeProfile ? "" : initialEdgeMapping?.targetLabel ?? "");
  const [sourceType, setSourceType] = useState(nodeProfile ? "" : initialEdgeMapping?.sourceType ?? "");
  const [targetType, setTargetType] = useState(nodeProfile ? "" : initialEdgeMapping?.targetType ?? "");
  const [edgeType, setEdgeType] = useState(initialEdgeMapping?.edgeType ?? "");
  const [weight, setWeight] = useState(initialEdgeMapping?.weight ?? "");
  const [timestamp, setTimestamp] = useState(initialEdgeMapping?.timestamp ?? "");
  const [timeFormat, setTimeFormat] = useState<Exclude<GraphTimeFormat, "none">>(
    initialTimeFormat === "none" ? "auto" : initialTimeFormat,
  );
  const canSubmit = source.length > 0 && target.length > 0 && source !== target && (!nodeProfile || nodeId.length > 0);
  const allUnclassified = nodeProfile ? !nodeType : !sourceType && !targetType;
  const mappedNodeColumns = new Set([nodeId, nodeLabel, nodeType].filter(Boolean));
  const mappedEdgeColumns = new Set([
    source,
    target,
    sourceLabel,
    targetLabel,
    sourceType,
    targetType,
    edgeType,
    weight,
    timestamp,
  ].filter(Boolean));

  return (
    <section className="mapping-card graph-import-mapping" aria-labelledby="graph-import-mapping-title">
      <div className="mapping-card__heading">
        <span className="mapping-icon"><Graph size={20} weight="light" /></span>
        <div>
          <strong id="graph-import-mapping-title">确认构图字段</strong>
          <p>所有选择都严格引用文件真实列；确认前不会持久化 GraphVersion。</p>
        </div>
      </div>

      {nodeProfile ? (
        <fieldset className="mapping-fieldset">
          <legend>节点表 · {nodeProfile.name}</legend>
          <div className="mapping-grid">
            <ColumnSelect label="唯一 ID" required value={nodeId} profile={nodeProfile} placeholder="选择节点 ID 列" onChange={setNodeId} />
            <ColumnSelect label="显示名称" value={nodeLabel} profile={nodeProfile} placeholder="默认使用节点 ID" onChange={setNodeLabel} />
            <ColumnSelect label="实体类型" value={nodeType} profile={nodeProfile} placeholder="不指定节点类型" onChange={setNodeType} />
          </div>
          <p className="mapping-attribute-note">显示名称不是训练标签；其余 {Math.max(0, nodeProfile.headers.length - mappedNodeColumns.size)} 列将原样保留为节点属性。</p>
        </fieldset>
      ) : null}

      <fieldset className="mapping-fieldset">
        <legend>关系表 · {edgeProfile.name}</legend>
        <div className="mapping-grid">
          <ColumnSelect label="起点 ID" required value={source} profile={edgeProfile} placeholder="选择起点列" onChange={setSource} />
          <ColumnSelect label="终点 ID" required value={target} profile={edgeProfile} placeholder="选择终点列" onChange={setTarget} />
          <ColumnSelect label="关系类型" value={edgeType} profile={edgeProfile} placeholder="不指定关系类型" onChange={setEdgeType} />
          {!nodeProfile ? <ColumnSelect label="起点显示名称" value={sourceLabel} profile={edgeProfile} placeholder="默认使用起点 ID" onChange={setSourceLabel} /> : null}
          {!nodeProfile ? <ColumnSelect label="终点显示名称" value={targetLabel} profile={edgeProfile} placeholder="默认使用终点 ID" onChange={setTargetLabel} /> : null}
          {!nodeProfile ? <ColumnSelect label="起点实体类型" value={sourceType} profile={edgeProfile} placeholder="不指定" onChange={setSourceType} /> : null}
          {!nodeProfile ? <ColumnSelect label="终点实体类型" value={targetType} profile={edgeProfile} placeholder="不指定" onChange={setTargetType} /> : null}
          <ColumnSelect label="权重" value={weight} profile={edgeProfile} placeholder="不指定权重" onChange={setWeight} />
          <ColumnSelect label="时间" value={timestamp} profile={edgeProfile} placeholder="不指定时间" onChange={setTimestamp} />
          {timestamp ? (
            <label>
              <span>时间格式 <em>必选</em></span>
              <select value={timeFormat} onChange={(event) => setTimeFormat(event.target.value as Exclude<GraphTimeFormat, "none">)}>
                <option value="auto">自动检测</option>
                <option value="iso8601">ISO 8601</option>
                <option value="year">年份</option>
                <option value="unix_seconds">Unix 秒</option>
                <option value="unix_milliseconds">Unix 毫秒</option>
              </select>
            </label>
          ) : null}
        </div>
        <p className="mapping-attribute-note">其余 {Math.max(0, edgeProfile.headers.length - mappedEdgeColumns.size)} 列将原样保留为关系属性。</p>
      </fieldset>

      {source && source === target ? <p className="field-error">起点和终点不能使用同一列。</p> : null}
      {allUnclassified ? <p className="mapping-warning">尚未映射实体类型；所有节点可能以“未分类”显示，异构图任务前建议确认类型字段。</p> : null}
      {issues.length ? (
        <details className="mapping-issues" open={issues.some((issue) => issue.severity === "error")}>
          <summary>质量与映射提示（{issues.length}）</summary>
          <ul>{issues.slice(0, 20).map((issue, index) => <li key={`${issue.code}:${issue.row ?? "graph"}:${index}`}>[{issue.severity}] {issue.message}</li>)}</ul>
        </details>
      ) : null}
      <div className="mapping-card__actions">
        <button className="secondary-button" type="button" onClick={onCancel}>取消导入</button>
        <button
          className="primary-button primary-button--small"
          type="button"
          disabled={!canSubmit}
          onClick={() => onApply({
            ...(nodeProfile ? {
              nodeMapping: {
                id: nodeId,
                ...(nodeLabel ? { label: nodeLabel } : {}),
                ...(nodeType ? { type: nodeType } : {}),
              },
            } : {}),
            edgeMapping: {
              source,
              target,
              ...(!nodeProfile && sourceLabel ? { sourceLabel } : {}),
              ...(!nodeProfile && targetLabel ? { targetLabel } : {}),
              ...(!nodeProfile && sourceType ? { sourceType } : {}),
              ...(!nodeProfile && targetType ? { targetType } : {}),
              ...(edgeType ? { edgeType } : {}),
              ...(weight ? { weight } : {}),
              ...(timestamp ? { timestamp } : {}),
            },
            timeFormat: timestamp ? timeFormat : "none",
          })}
        >
          验证映射并生成草稿 <ArrowRight size={16} />
        </button>
      </div>
    </section>
  );
}

export function GraphTableRoleCard({
  files,
  profiles,
  initialEdgeIndex,
  onApply,
  onCancel,
}: {
  readonly files: readonly File[];
  readonly profiles: readonly FileProfile[];
  readonly initialEdgeIndex: number;
  readonly onApply: (edgeIndex: number) => void;
  readonly onCancel: () => void;
}) {
  const [edgeIndex, setEdgeIndex] = useState(initialEdgeIndex);
  return (
    <section className="mapping-card graph-table-role-card" aria-labelledby="graph-table-role-title">
      <div className="mapping-card__heading">
        <span className="mapping-icon"><Shuffle size={20} weight="light" /></span>
        <div>
          <strong id="graph-table-role-title">确认 nodes + edges 文件角色</strong>
          <p>请选择关系表；另一个文件将作为节点表。系统不会根据原始值猜测角色。</p>
        </div>
      </div>
      <fieldset className="table-role-options">
        <legend>哪个文件保存关系？</legend>
        {files.map((file, index) => (
          <label key={`${file.name}:${file.size}:${index}`}>
            <input type="radio" name="edge-table" value={index} checked={edgeIndex === index} onChange={() => setEdgeIndex(index)} />
            <span><strong>{file.name}</strong><small>{profiles[index]?.headers.slice(0, 8).join("、") || "未读取到列"}</small></span>
          </label>
        ))}
      </fieldset>
      <div className="mapping-card__actions">
        <button className="secondary-button" type="button" onClick={onCancel}>取消导入</button>
        <button className="primary-button primary-button--small" type="button" onClick={() => onApply(edgeIndex)}>
          确认文件角色 <ArrowRight size={16} />
        </button>
      </div>
    </section>
  );
}
