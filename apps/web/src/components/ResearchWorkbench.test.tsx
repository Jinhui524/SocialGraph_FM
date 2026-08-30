import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { buildDemoGraphVersion } from "../services/graphImport";
import { createLocalGraphRepository } from "../services/graphRepository";
import type {
  ResearchCapabilities,
  ResearchClientLike,
  ResearchRunBinding,
  ResearchRunResult,
  ResearchRunStatus,
  ResearchScenarios,
  ResearchTargetScope,
  ResearchTaskId,
} from "../types/research";
import { ResearchWorkbench } from "./ResearchWorkbench";

afterEach(cleanup);

const HASHES = {
  graph: "2".repeat(64),
  model: "3".repeat(64),
  artifact: "4".repeat(64),
  request: "5".repeat(64),
  result: "6".repeat(64),
  state: "7".repeat(64),
  capability: "8".repeat(64),
  scenarios: "9".repeat(64),
};

const TASKS = [
  "research.content_policy_review",
  "research.account_risk_review",
  "research.signed_relation_review",
  "core.collaboration_completion",
] as const satisfies readonly ResearchTaskId[];

function capability(): ResearchCapabilities {
  return {
    schemaVersion: "socialgraph-fm.research/1.0",
    channel: "research",
    releaseLabel: "SocialGraph-FM Research",
    seed: 1729,
    preliminary: true,
    researchServingReady: true,
    unavailableReason: null,
    model: {
      modelVersionId: "socialgraph-fm-research/model",
      modelVersionHash: HASHES.model,
      artifactHash: HASHES.artifact,
      taskIds: TASKS,
      graphSchemaVersion: "socialgraph-fm.core-graph-bundle/2.0",
      maxNodes: 50_000,
      maxEdges: 1_500_000,
      claimStatus: "not_demonstrated",
    },
    taskIds: TASKS,
    upload: {
      compatibleTaskIds: ["core.collaboration_completion"],
      auxiliaryCapabilities: ["similar-nodes"],
      minNodes: 5,
      maxNodes: 50_000,
      maxEdges: 1_500_000,
    },
    capabilityHash: HASHES.capability,
  };
}

function scenarios(nodeId: string): ResearchScenarios {
  const rows: readonly [
    "twitch-content-policy" | "tolokers-account-risk" | "wiki-rfa-signed-relation" | "email-eu-collaboration",
    string,
    ResearchTaskId,
    ResearchTargetScope,
  ][] = [
    ["twitch-content-policy", "twitch-language", TASKS[0], { kind: "nodes", nodeIds: [nodeId] }],
    ["tolokers-account-risk", "tolokers", TASKS[1], { kind: "nodes", nodeIds: [nodeId] }],
    ["wiki-rfa-signed-relation", "wiki-rfa", TASKS[2], { kind: "directed-node-pairs", pairs: [[nodeId, "member-li"]] }],
    ["email-eu-collaboration", "email-eu-core", TASKS[3], { kind: "collaboration-candidates", anchorNodeId: nodeId, topK: 20 }],
  ];
  return {
    schemaVersion: "socialgraph-fm.research/1.0",
    releaseLabel: "SocialGraph-FM Research",
    seed: 1729,
    preliminary: true,
    scenarios: rows.map(([scenarioId, datasetId, taskId, defaultTargetScope]) => ({
      scenarioId,
      datasetId,
      title: `${datasetId} 登记场景`,
      taskId,
      graphVersionId: "research:twitch-language",
      graphVersionHash: HASHES.graph,
      modelVersionId: "socialgraph-fm-research/model",
      enabled: true,
      unavailableReason: null,
      defaultTargetScope,
      primaryMetric: { name: "Macro-F1", value: 0.71 },
      scratchDelta: 0.03,
    })),
    scenariosHash: HASHES.scenarios,
  };
}

function successfulClient(nodeId: string): ResearchClientLike {
  const binding: ResearchRunBinding = {
    runId: "research-run-1",
    publicRequestHash: "a".repeat(64),
    serverRequestHash: HASHES.request,
    graphVersionId: "research:twitch-language",
    modelVersionId: "socialgraph-fm-research/model",
    taskId: TASKS[0],
  };
  const status: ResearchRunStatus = {
    schemaVersion: "socialgraph-fm.research/1.0",
    runId: binding.runId,
    requestHash: binding.serverRequestHash,
    status: "succeeded",
    progress: 100,
    createdAt: "2026-08-16T00:00:00.000000Z",
    updatedAt: "2026-08-16T00:00:01.000000Z",
    errorCode: null,
    stateHash: HASHES.state,
  };
  const result: ResearchRunResult = {
    schemaVersion: "socialgraph-fm.research/1.0",
    runId: binding.runId,
    requestHash: binding.serverRequestHash,
    taskId: TASKS[0],
    graphVersionId: binding.graphVersionId,
    graphVersionHash: HASHES.graph,
    modelVersionId: binding.modelVersionId,
    modelVersionHash: HASHES.model,
    seed: 1729,
    preliminary: true,
    calibrationStatus: "ranking_only",
    findings: [{
      id: "content-finding-1",
      rank: 1,
      entityType: "node",
      entityIds: [nodeId],
      score: 0.72,
      scoreKind: "ranking-score",
      calibrated: false,
      reasonCodes: ["structure.shared-encoder"],
      limitations: ["Single-seed preliminary result."],
      reviewRequired: true,
    }],
    completedAt: "2026-08-16T00:00:01.000000Z",
    resultHash: HASHES.result,
  };
  return {
    capabilities: vi.fn().mockResolvedValue(capability()),
    scenarios: vi.fn().mockResolvedValue(scenarios(nodeId)),
    scenarioPreview: vi.fn(),
    createRun: vi.fn().mockResolvedValue({ status, binding }),
    getRun: vi.fn(),
    getResult: vi.fn().mockResolvedValue(result),
    similarNodes: vi.fn(),
  };
}

describe("SocialGraph-FM Research governance workbench", () => {
  it("runs a registered scenario, labels preliminary evidence, and emits a bound overlay", async () => {
    const base = buildDemoGraphVersion();
    const nodeId = base.nodes[0]!.id;
    const graph = Object.freeze({ ...base, id: "research:twitch-language", contentHash: HASHES.graph });
    const client = successfulClient(nodeId);
    const onOverlayChange = vi.fn();

    render(<ResearchWorkbench
      graph={graph}
      selectedNodeId={nodeId}
      pathEndpointIds={[]}
      service={{ state: "connected", capabilities: capability() }}
      client={client}
      sessionId="session-1"
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={onOverlayChange}
      onClose={vi.fn()}
    />);

    expect(await screen.findByText("twitch-language 登记场景")).toBeInTheDocument();
    expect(screen.getByText("SocialGraph-FM Research")).toBeInTheDocument();
    expect(screen.getByText("单次实验初步结果")).toBeInTheDocument();
    expect(screen.getByText("尚未证明优于单域基线")).toBeInTheDocument();
    const runButton = screen.getByRole("button", { name: "运行治理任务" });
    await waitFor(() => expect(runButton).toBeEnabled());
    fireEvent.click(runButton);

    await waitFor(() => expect(client.createRun).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(client.getResult).toHaveBeenCalledTimes(1));
    expect((await screen.findAllByText("0.7200"))).toHaveLength(2);
    expect(screen.getByText(/仅排序分数/)).toBeInTheDocument();
    expect(screen.getByText("structure.shared-encoder")).toBeInTheDocument();
    expect(screen.getByText(/单随机种子初步结果/)).toBeInTheDocument();
    await waitFor(() => expect(onOverlayChange).toHaveBeenCalledWith(expect.objectContaining({
      kind: "governance",
      graphVersionId: graph.id,
      nodeValues: expect.objectContaining({ [nodeId]: "subject" }),
    })));
    expect(client.createRun).toHaveBeenCalledWith(expect.objectContaining({
      taskId: "research.content_policy_review",
      scenarioId: "twitch-content-policy",
    }), expect.any(AbortSignal));
  });

  it("keeps entity results unavailable when SocialGraph-FM Research is not ready", async () => {
    const graph = buildDemoGraphVersion();
    const client = successfulClient(graph.nodes[0]!.id);
    render(<ResearchWorkbench
      graph={graph}
      selectedNodeId={graph.nodes[0]!.id}
      pathEndpointIds={[]}
      service={{ state: "unavailable", code: "GFM_RESEARCH_SERVICE_UNAVAILABLE" }}
      client={client}
      sessionId="session-1"
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
      onClose={vi.fn()}
    />);

    expect(await screen.findByText(/SocialGraph-FM Research 服务暂不可用/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "运行治理任务" })).toBeDisabled();
    expect(screen.queryByText("候选排序")).not.toBeInTheDocument();
    expect(client.createRun).not.toHaveBeenCalled();
  });

  it("shows domain-task blockers for My Graph instead of training a label head in the browser", async () => {
    const graph = buildDemoGraphVersion();
    const client = successfulClient(graph.nodes[0]!.id);
    const onSourceModeChange = vi.fn();
    render(<ResearchWorkbench
      graph={graph}
      selectedNodeId={graph.nodes[0]!.id}
      pathEndpointIds={[]}
      service={{ state: "connected", capabilities: capability() }}
      client={client}
      sessionId="session-1"
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
      onClose={vi.fn()}
      onSourceModeChange={onSourceModeChange}
    />);
    await screen.findByText("twitch-language 登记场景");
    fireEvent.click(screen.getByRole("button", { name: /我的图谱/ }));

    expect(onSourceModeChange).toHaveBeenCalledWith("my-graph");
    expect(screen.getByText(/三个标签专用任务仅运行登记示例/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "准备并运行 SocialGraph-FM Research 分析" })).toBeDisabled();
  });

  it("labels registered full-graph counts separately from a bounded scenario projection", async () => {
    const base = buildDemoGraphVersion();
    const nodeId = base.nodes[0]!.id;
    const graph = Object.freeze({
      ...base,
      id: "research:twitch-language",
      contentHash: HASHES.graph,
      truncated: true,
      preview: Object.freeze({
        ...base.preview,
        truncated: true,
        originalNodeCount: 1_000,
        originalEdgeCount: 2_000,
      }),
      datasetArtifact: Object.freeze({
        id: "research-preview:twitch-content-policy",
        datasetName: "twitch-language",
        checksum: HASHES.graph,
        canonicalGraphHash: HASHES.graph,
        scope: "projection" as const,
      }),
    });
    const client = successfulClient(nodeId);
    render(<ResearchWorkbench
      graph={graph}
      selectedNodeId={nodeId}
      pathEndpointIds={[]}
      service={{ state: "connected", capabilities: capability() }}
      client={client}
      sessionId="session-1"
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
      onClose={vi.fn()}
    />);

    const runButton = await screen.findByRole("button", { name: "运行治理任务" });
    await waitFor(() => expect(runButton).toBeEnabled());
    fireEvent.click(runButton);

    expect(await screen.findByText("完整图节点")).toBeInTheDocument();
    expect(screen.getByText("1,000")).toBeInTheDocument();
    expect(screen.getByText("完整图关系")).toBeInTheDocument();
    expect(screen.getByText("2,000")).toBeInTheDocument();
    expect(screen.getByText("投影可见")).toBeInTheDocument();
  });
});
