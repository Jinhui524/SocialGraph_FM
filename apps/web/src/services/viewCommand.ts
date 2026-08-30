import type {
  ApplyViewCommandResult,
  GraphViewState,
  GraphVersion,
  TargetResolution,
  ViewCommand,
} from "../types/graph";
import { normalizeNodeTerm, resolveViewTargets } from "./targetResolver";
import { normalizeGraphViewState } from "./graphScene";

function unique(values: readonly string[]): string[] {
  return [...new Set(values)];
}

function resolveTypeTerms(
  requestedTerms: readonly string[],
  availableValues: readonly string[],
  label: string,
): { values: string[]; warnings: string[] } {
  const available = unique(availableValues.filter(Boolean)).sort((left, right) => left.localeCompare(right, "zh-CN"));
  const values: string[] = [];
  const warnings: string[] = [];
  for (const term of requestedTerms) {
    const normalized = normalizeNodeTerm(term);
    if (!normalized) continue;
    let matches = available.filter((value) => normalizeNodeTerm(value) === normalized);
    if (matches.length === 0) {
      matches = available.filter((value) => normalizeNodeTerm(value).includes(normalized));
    }
    if (matches.length === 1) values.push(matches[0]);
    else if (matches.length > 1) warnings.push(`${label}“${term}”匹配到多个候选，请手动选择。`);
    else warnings.push(`当前图中不存在${label}“${term}”，未应用该筛选。`);
  }
  return { values: unique(values), warnings };
}

/**
 * Applies an LLM-produced command only after resolving every graph-specific
 * term against local facts. It never mutates GraphVersion or currentState.
 */
export function applyViewCommand(
  graph: GraphVersion,
  currentState: GraphViewState,
  command: ViewCommand,
  chosenNodeIds: readonly string[] = [],
): ApplyViewCommandResult {
  currentState = normalizeGraphViewState(graph.id, currentState);
  const targetResolutions = resolveViewTargets(graph, command.focusTerms);
  const existingIds = new Set(graph.nodes.map((node) => node.id));
  const warnings: string[] = [];
  const resolvedIds = targetResolutions
    .filter((resolution): resolution is Extract<TargetResolution, { status: "resolved" }> => resolution.status === "resolved")
    .map((resolution) => resolution.nodeId);

  for (const resolution of targetResolutions) {
    if (resolution.status === "ambiguous") {
      warnings.push(`“${resolution.term}”匹配到多个节点，请从候选节点中选择。`);
    } else if (resolution.status === "not_found") {
      warnings.push(`当前图中没有找到“${resolution.term}”。`);
    }
  }

  const validChosenIds = unique(chosenNodeIds).filter((nodeId) => {
    if (existingIds.has(nodeId)) return true;
    warnings.push(`节点“${nodeId}”不属于当前 GraphVersion，已忽略。`);
    return false;
  });
  const requestedFocus = unique([...resolvedIds, ...validChosenIds]);
  const hasRequestedTargets = command.focusTerms.length > 0 || chosenNodeIds.length > 0;
  const currentFocusNodeIds = [...currentState.focusNodeIds].filter((nodeId) => existingIds.has(nodeId));
  const currentPathEndpointIds = [...currentState.pathEndpointIds].filter((nodeId) => existingIds.has(nodeId));
  let focusNodeIds = currentFocusNodeIds;
  let pathEndpointIds = currentPathEndpointIds;

  let mode = command.mode ?? currentState.mode;
  if (mode === "local") {
    const requestedLocalFocus = hasRequestedTargets ? requestedFocus : currentFocusNodeIds;
    if (requestedLocalFocus.length === 0) {
      warnings.push("局部图需要一个焦点节点，当前视图模式保持不变。");
      mode = currentState.mode;
    } else {
      focusNodeIds = requestedLocalFocus;
      pathEndpointIds = [];
    }
  } else if (mode === "path") {
    const requestedPathEndpoints = hasRequestedTargets ? requestedFocus : currentPathEndpointIds;
    if (requestedPathEndpoints.length < 2) {
      warnings.push("路径图需要两个已解析节点，当前视图模式保持不变。");
      mode = currentState.mode;
    } else {
      pathEndpointIds = requestedPathEndpoints.slice(0, 2);
      if (currentState.mode !== "path") focusNodeIds = [];
    }
  } else if (command.mode === "global") {
    focusNodeIds = [];
    pathEndpointIds = [];
  }

  if (mode !== (command.mode ?? currentState.mode)) {
    focusNodeIds = currentFocusNodeIds;
    pathEndpointIds = currentPathEndpointIds;
  }

  const nodeTypeResult = resolveTypeTerms(
    command.nodeTypeTerms,
    graph.nodes.map((node) => node.type ?? ""),
    "节点类型",
  );
  const edgeTypeResult = resolveTypeTerms(
    command.edgeTypeTerms,
    graph.edges.map((edge) => edge.type ?? ""),
    "关系类型",
  );
  warnings.push(...nodeTypeResult.warnings, ...edgeTypeResult.warnings);

  const nextState: GraphViewState = Object.freeze({
    graphVersionId: graph.id,
    mode,
    focusNodeIds: Object.freeze(focusNodeIds),
    pathEndpointIds: Object.freeze(pathEndpointIds),
    depth: command.depth ?? currentState.depth,
    filters: Object.freeze({
      nodeTypes: Object.freeze(
        command.nodeTypeTerms.length > 0 ? nodeTypeResult.values : [...currentState.filters.nodeTypes],
      ),
      edgeTypes: Object.freeze(
        command.edgeTypeTerms.length > 0 ? edgeTypeResult.values : [...currentState.filters.edgeTypes],
      ),
      ...(command.timeRange
        ? { timeRange: Object.freeze({ ...command.timeRange }) }
        : currentState.filters.timeRange
          ? { timeRange: Object.freeze({ ...currentState.filters.timeRange }) }
          : {}),
      ...(currentState.filters.minWeight !== undefined ? { minWeight: currentState.filters.minWeight } : {}),
      ...(currentState.filters.maxWeight !== undefined ? { maxWeight: currentState.filters.maxWeight } : {}),
      ...(currentState.filters.directed !== undefined ? { directed: currentState.filters.directed } : {}),
      ...(currentState.filters.emptyReason ? { emptyReason: currentState.filters.emptyReason } : {}),
    }),
    theme: currentState.theme,
    layoutPreset: command.layoutPreset ?? currentState.layoutPreset,
    rendererPreference: currentState.rendererPreference,
    camera: Object.freeze({ ...currentState.camera }),
    pinnedNodes: Object.freeze(
      Object.fromEntries(
        Object.entries(currentState.pinnedNodes).map(([nodeId, position]) => [
          nodeId,
          Object.freeze({ ...position }),
        ]),
      ),
    ),
  });

  return Object.freeze({
    nextState,
    targetResolutions: Object.freeze([...targetResolutions]),
    warnings: Object.freeze(warnings),
    ...(command.overlay ? { requestedOverlay: command.overlay } : {}),
  });
}
