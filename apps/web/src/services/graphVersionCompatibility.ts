import type { GraphRepository, GraphVersion } from "../types/graph";

export const CURRENT_BROWSER_IMPORT_PIPELINE_VERSION = "2.0.0";

export type GraphVersionCompatibilityStatus =
  | "current"
  | "research_dataset"
  | "upgrade_available"
  | "source_missing"
  | "legacy_read_only";

export interface GraphVersionCompatibility {
  readonly status: GraphVersionCompatibilityStatus;
  readonly allNodesUntyped: boolean;
  readonly canDeterministicallyRebuild: boolean;
  readonly sourceArtifactIds: readonly string[];
  readonly missingSourceArtifactIds: readonly string[];
  readonly message?: string;
}

/** Derives compatibility without modifying or silently migrating immutable graph bytes. */
export async function inspectGraphVersionCompatibility(
  repository: Pick<GraphRepository, "getSourceArtifact">,
  version: GraphVersion,
): Promise<GraphVersionCompatibility> {
  const sourceArtifactIds = [...new Set([
    ...(version.sourceArtifactIds ?? []),
    ...(version.buildSpec?.sourceArtifactIds ?? []),
  ])].sort((left, right) => left.localeCompare(right));
  const allNodesUntyped = version.nodes.length > 0 && version.nodes.every((node) => !node.type?.trim());

  if (version.datasetArtifact || version.provenance?.origin === "research_dataset") {
    return {
      status: "research_dataset",
      allNodesUntyped,
      canDeterministicallyRebuild: false,
      sourceArtifactIds,
      missingSourceArtifactIds: [],
    };
  }

  const artifacts = await Promise.all(sourceArtifactIds.map((id) => repository.getSourceArtifact(id)));
  const missingSourceArtifactIds = sourceArtifactIds.filter((_id, index) => !artifacts[index]);
  const pipelineCurrent = version.provenance?.pipeline === "browser-import"
    && version.provenance.pipelineVersion === CURRENT_BROWSER_IMPORT_PIPELINE_VERSION;

  if (pipelineCurrent) {
    return {
      status: "current",
      allNodesUntyped,
      canDeterministicallyRebuild: sourceArtifactIds.length > 0 && missingSourceArtifactIds.length === 0,
      sourceArtifactIds,
      missingSourceArtifactIds,
      ...(allNodesUntyped ? { message: "当前版本没有已识别的节点类型，画面会使用“未分类”样式。" } : {}),
    };
  }
  if (sourceArtifactIds.length > 0 && missingSourceArtifactIds.length === 0) {
    return {
      status: "upgrade_available",
      allNodesUntyped,
      canDeterministicallyRebuild: true,
      sourceArtifactIds,
      missingSourceArtifactIds,
      message: "该兼容版本可从同一 SourceArtifact 用当前确定性管线重建；旧版本不会被改写。",
    };
  }
  if (sourceArtifactIds.length > 0) {
    return {
      status: "source_missing",
      allNodesUntyped,
      canDeterministicallyRebuild: false,
      sourceArtifactIds,
      missingSourceArtifactIds,
      message: "部分原始 SourceArtifact 已缺失，需要重新选择原文件后才能创建重建子版本。",
    };
  }
  return {
    status: "legacy_read_only",
    allNodesUntyped,
    canDeterministicallyRebuild: false,
    sourceArtifactIds,
    missingSourceArtifactIds,
    message: allNodesUntyped
      ? "该兼容版本未保存节点类型和可验证源文件，因此统一显示为“未分类”；请重新选择原文件创建子版本。"
      : "该兼容版本没有可验证的 SourceArtifact；旧版本保持只读。",
  };
}

