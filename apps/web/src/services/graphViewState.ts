import type {
  GraphInteractionState,
  GraphViewMode,
  GraphViewState,
  GraphWorkbenchViewState,
} from "../types/graph";

export type GraphViewAction =
  | { readonly type: "activate_mode"; readonly mode: GraphViewMode }
  | { readonly type: "select_node"; readonly nodeId: string | null }
  | { readonly type: "set_focus"; readonly nodeIds: readonly string[] }
  | { readonly type: "set_path_endpoints"; readonly nodeIds: readonly string[] }
  | { readonly type: "cancel_tool" }
  | { readonly type: "clear_selection" }
  | { readonly type: "update_view"; readonly patch: Partial<GraphViewState> }
  | {
      readonly type: "replace_view";
      readonly viewState: GraphViewState;
      readonly resetInteraction?: boolean;
    };

export function createDefaultGraphInteractionState(): GraphInteractionState {
  return Object.freeze({ tool: "browse" as const, selectedNodeId: null });
}

export function createGraphWorkbenchViewState(viewState: GraphViewState): GraphWorkbenchViewState {
  return Object.freeze({
    viewState,
    interaction: createDefaultGraphInteractionState(),
  });
}

function uniqueNodeIds(nodeIds: readonly string[], limit?: number): readonly string[] {
  const result = [...new Set(nodeIds.filter(Boolean))];
  return Object.freeze(limit === undefined ? result : result.slice(0, limit));
}

function withView(
  state: GraphWorkbenchViewState,
  patch: Partial<GraphViewState>,
  interaction: GraphInteractionState = state.interaction,
): GraphWorkbenchViewState {
  return Object.freeze({
    viewState: Object.freeze({ ...state.viewState, ...patch }),
    interaction: Object.freeze(interaction),
  });
}

/**
 * The single transition function for graph-mode tools and node selection.
 * Entering local/path mode is safe before or after selecting nodes: an
 * incomplete tool keeps its mode while GraphScene renders the filtered global
 * graph until the required node(s) are available.
 */
export function reduceGraphView(
  state: GraphWorkbenchViewState,
  action: GraphViewAction,
): GraphWorkbenchViewState {
  switch (action.type) {
    case "activate_mode": {
      if (action.mode === "global") {
        return withView(
          state,
          { mode: "global", focusNodeIds: Object.freeze([]), pathEndpointIds: Object.freeze([]) },
          createDefaultGraphInteractionState(),
        );
      }
      if (action.mode === "local") {
        const selectedNodeId = state.interaction.selectedNodeId;
        return withView(
          state,
          {
            mode: "local",
            focusNodeIds: selectedNodeId ? Object.freeze([selectedNodeId]) : Object.freeze([]),
            pathEndpointIds: Object.freeze([]),
          },
          selectedNodeId
            ? { tool: "browse", selectedNodeId }
            : { tool: "pick_local_focus", selectedNodeId: null },
        );
      }
      const selectedNodeId = state.interaction.selectedNodeId;
      const pathEndpointIds = selectedNodeId ? Object.freeze([selectedNodeId]) : Object.freeze([]);
      return withView(
        state,
        { mode: "path", pathEndpointIds },
        selectedNodeId
          ? { tool: "pick_path_end", selectedNodeId, pendingPathStartId: selectedNodeId }
          : { tool: "pick_path_start", selectedNodeId: null },
      );
    }

    case "select_node": {
      if (!action.nodeId) {
        return Object.freeze({
          viewState: state.viewState,
          interaction: Object.freeze({ ...state.interaction, selectedNodeId: null }),
        });
      }
      if (state.interaction.tool === "pick_local_focus") {
        return withView(
          state,
          { mode: "local", focusNodeIds: Object.freeze([action.nodeId]) },
          { tool: "browse", selectedNodeId: action.nodeId },
        );
      }
      if (state.interaction.tool === "pick_path_start") {
        return withView(
          state,
          { mode: "path", pathEndpointIds: Object.freeze([action.nodeId]) },
          { tool: "pick_path_end", selectedNodeId: action.nodeId, pendingPathStartId: action.nodeId },
        );
      }
      if (state.interaction.tool === "pick_path_end") {
        const sourceId = state.interaction.pendingPathStartId ?? state.viewState.pathEndpointIds[0];
        if (!sourceId || sourceId === action.nodeId) {
          return Object.freeze({
            viewState: state.viewState,
            interaction: Object.freeze({
              tool: "pick_path_end" as const,
              selectedNodeId: action.nodeId,
              ...(sourceId ? { pendingPathStartId: sourceId } : {}),
            }),
          });
        }
        return withView(
          state,
          { mode: "path", pathEndpointIds: Object.freeze([sourceId, action.nodeId]) },
          { tool: "browse", selectedNodeId: action.nodeId },
        );
      }
      return Object.freeze({
        viewState: state.viewState,
        interaction: Object.freeze({ tool: "browse" as const, selectedNodeId: action.nodeId }),
      });
    }

    case "set_focus": {
      const focusNodeIds = uniqueNodeIds(action.nodeIds);
      return withView(
        state,
        { mode: "local", focusNodeIds, pathEndpointIds: Object.freeze([]) },
        focusNodeIds.length > 0
          ? { tool: "browse", selectedNodeId: focusNodeIds[0] }
          : { tool: "pick_local_focus", selectedNodeId: null },
      );
    }

    case "set_path_endpoints": {
      const pathEndpointIds = uniqueNodeIds(action.nodeIds, 2);
      if (pathEndpointIds.length === 0) {
        return withView(
          state,
          { mode: "path", pathEndpointIds },
          { tool: "pick_path_start", selectedNodeId: null },
        );
      }
      if (pathEndpointIds.length === 1) {
        return withView(
          state,
          { mode: "path", pathEndpointIds },
          {
            tool: "pick_path_end",
            selectedNodeId: pathEndpointIds[0],
            pendingPathStartId: pathEndpointIds[0],
          },
        );
      }
      return withView(
        state,
        { mode: "path", pathEndpointIds },
        { tool: "browse", selectedNodeId: pathEndpointIds[1] },
      );
    }

    case "cancel_tool":
      return Object.freeze({
        viewState: state.viewState,
        interaction: Object.freeze({ tool: "browse" as const, selectedNodeId: state.interaction.selectedNodeId }),
      });

    case "clear_selection":
      return Object.freeze({
        viewState: state.viewState,
        interaction: Object.freeze({ ...state.interaction, selectedNodeId: null }),
      });

    case "update_view":
      return withView(state, action.patch);

    case "replace_view":
      return Object.freeze({
        viewState: action.viewState,
        interaction: action.resetInteraction === false
          ? state.interaction
          : createDefaultGraphInteractionState(),
      });
  }
}
