import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type {
  CoreCapabilities,
  CoreEntityType,
  CoreFindingType,
  CoreRunStatus,
  CoreTaskId,
} from "../types/core";
import { SocialGraphApiError } from "../services/apiClient";
import { registeredEdgeIdentityForLocalId } from "../services/coreEdgeIdentity";
import { createGraphVersion } from "../services/graphImport";
import { sha256Canonical } from "../services/graphIdentity";
import { createLocalGraphRepository } from "../services/graphRepository";
import { createValidatedCoreFixture } from "../test/fixtures/core";
import { CoreWorkbenchPanel } from "./CoreWorkbenchPanel";

afterEach(cleanup);

const graph = createGraphVersion("core-social.json", [
  { id: "community-a", label: "协作社区", type: "community", attributes: {} },
  { id: "actor-a", label: "成员 A", type: "person", attributes: {} },
  { id: "actor-b", label: "成员 B", type: "person", attributes: {} },
], [{
  id: "membership-a",
  source: "actor-a",
  target: "community-a",
  type: "member_of",
  weight: 1,
  directed: true,
  attributes: {},
}]);

function capabilities(state: "accepted" | "servingReady" = "servingReady"): CoreCapabilities {
  const tasks = [
    "core.community_resilience_review",
    "core.risk_and_trust_review",
    "core.collaboration_completion",
  ] as const;
  const taskBindings = [
    [tasks[0], "community", "regression-interval", "validation-residual-interval", "community"],
    [tasks[1], "node", "binary-calibration", "sigmoid", "risk-node"],
    [tasks[1], "edge", "binary-calibration", "sigmoid", "risk-edge"],
    [tasks[2], "node-pair", "binary-calibration", "sigmoid", "collaboration"],
  ].map(([taskId, entityType, confidenceKind, method, adapterDomain], index) => ({
    taskId,
    entityType,
    confidenceKind,
    calibrationVersion: `calibration/${index + 1}`,
    method,
    calibrationArtifactHash: (index + 2).toString(16).repeat(64),
    calibrationProtocolHash: (index + 6).toString(16).repeat(64),
    adapterDomain,
    adapterSchemaHash: "a".repeat(64),
    adapterStateHash: "b".repeat(64),
    featureContractHash: (index + 1).toString(16).repeat(64),
  })) as CoreCapabilities["models"][number]["taskBindings"];
  const graphFeatureContractHash = sha256Canonical(taskBindings.map((binding) => ({
    taskId: binding.taskId,
    entityType: binding.entityType,
    featureContractHash: binding.featureContractHash,
  })));
  return {
    schemaVersion: "socialgraph-fm.core-capabilities/2.0",
    registryHash: "1".repeat(64),
    registryGeneration: 1,
    servingReady: state === "servingReady",
    models: [{
      modelVersionId: "socialgraph-fm-core/review",
      modelVersionHash: "5".repeat(64),
      state,
      tasks,
      graphSchemaVersions: ["socialgraph-fm.core-graph-bundle/2.0"],
      graphFeatureContractHash,
      taskBindings,
      maxNodes: 10_000,
      maxEdges: 100_000,
    }],
    tasks,
    readiness: {
      modelValidated: true,
      coreServingReady: state === "servingReady",
    },
  };
}

function successfulClient(options: {
  readonly taskId?: CoreTaskId;
  readonly findingType?: CoreFindingType;
  readonly entityType?: CoreEntityType;
  readonly subjectIds?: readonly string[];
} = {}) {
  const membershipHash = registeredEdgeIdentityForLocalId(graph, "membership-a").edgeHash;
  const fixture = createValidatedCoreFixture({
    graphVersionId: graph.id,
    ...(options.taskId ? { taskId: options.taskId } : {}),
    subjectIds: options.subjectIds ?? ["actor-a"],
    ...(options.findingType ? { findingType: options.findingType } : {}),
    ...(options.entityType ? { entityType: options.entityType } : {}),
    pathNodeIds: ["actor-a", "community-a"],
    pathEdgeIds: [membershipHash],
  });
  const status: CoreRunStatus = {
    schemaVersion: "socialgraph-fm.core-run-status/2.0",
    runId: fixture.binding.runId,
    requestHash: fixture.binding.serverRequestHash,
    status: "succeeded",
    progress: 100,
    createdAt: "2026-08-15T00:00:00.000Z",
    updatedAt: "2026-08-15T00:00:01.000Z",
    errorCode: null,
    stateHash: "a".repeat(64),
  };
  return {
    fixture,
    client: {
      capabilities: vi.fn().mockResolvedValue(capabilities()),
      createRun: vi.fn().mockResolvedValue({ status, binding: fixture.binding }),
      getRun: vi.fn(),
      getResult: vi.fn().mockResolvedValue(fixture.result),
    },
  };
}

describe("core workbench panel", () => {
  it("keeps local graph work available and exposes no entity finding when the API is unavailable", () => {
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "unavailable", code: "GFM_CORE_SERVICE_UNAVAILABLE" }}
      client={{
        capabilities: vi.fn(),
        createRun: vi.fn(),
        getRun: vi.fn(),
        getResult: vi.fn(),
      }}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      sessionId="session-1"
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    expect(screen.getByRole("heading", { name: "静态图治理任务" })).toBeInTheDocument();
    expect(screen.getByText(/SocialGraph-FM Core 服务暂不可用/)).toBeInTheDocument();
    expect(screen.getByText(/本地图导入与确定性分析仍可使用/)).toBeInTheDocument();
    expect(screen.queryByText(/模型发现 #/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Penn94|gender|性别预测/iu)).not.toBeInTheDocument();
  });

  it("never renders an unregistered service code supplied by an upstream boundary", () => {
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "unavailable", code: "CHECKPOINT_C_USERS_PRIVATE_SECRET" }}
      client={{ capabilities: vi.fn(), createRun: vi.fn(), getRun: vi.fn(), getResult: vi.fn() }}
      selectedNodeId={null}
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    expect(screen.getByText(/GFM_CORE_SERVICE_UNAVAILABLE/)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("PRIVATE_SECRET");
  });

  it("shows the three Core task labels without temporal, future, or enforcement wording", () => {
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{
        state: "connected",
        capabilities: {
          schemaVersion: "socialgraph-fm.core-capabilities/2.0",
          registryHash: "1".repeat(64),
          registryGeneration: 0,
          servingReady: false,
          models: [],
          tasks: [],
          readiness: { modelValidated: false, coreServingReady: false },
        },
      }}
      client={{ capabilities: vi.fn(), createRun: vi.fn(), getRun: vi.fn(), getResult: vi.fn() }}
      selectedNodeId={null}
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    expect(screen.getByText("社区韧性复核")).toBeInTheDocument();
    expect(screen.getByText("风险与信任复核")).toBeInTheDocument();
    expect(screen.getByText("协作关系补全")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "治理任务" })).toBeInTheDocument();
    expect(screen.getByText(/GFM 服务已连接；正式模型未就绪/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/Time2Vec|未来预测|自动处罚|自动执法/u);
  });

  it("runs a serving model, overlays only existing evidence, and keeps local review separate from server truth", async () => {
    const { client, fixture } = successfulClient();
    const repository = createLocalGraphRepository({ forceMemory: true });
    const onOverlayChange = vi.fn();
    const onReportExport = vi.fn();

    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      sessionId="session-1"
      repository={repository}
      onOverlayChange={onOverlayChange}
      onReportExport={onReportExport}
    />);

    fireEvent.click(screen.getByRole("button", { name: "风险与信任复核" }));
    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));

    expect(await screen.findByText("模型发现 #1")).toBeInTheDocument();
    const resultRegion = screen.getByRole("region", { name: "治理复核结果" });
    expect(resultRegion).toHaveAttribute("aria-live", "polite");
    await waitFor(() => expect(resultRegion).toHaveFocus());
    expect(screen.getByText("registered_model.score-reference")).toBeInTheDocument();
    expect(screen.getByText(/校准置信度不是违规、风险或事实为真的概率/)).toBeInTheDocument();
    expect(screen.getByText(/服务器状态：待人工复核/)).toBeInTheDocument();
    expect(screen.getByText(/publicRequestHash.*浏览器可重算/)).toBeInTheDocument();
    expect(screen.getByText(/serverRequestHash.*隐藏 envelope/)).toBeInTheDocument();
    expect(fixture.binding.publicRequestHash).not.toBe(fixture.binding.serverRequestHash);
    expect(onOverlayChange).toHaveBeenLastCalledWith(expect.objectContaining({
      kind: "governance",
      graphVersionId: graph.id,
      nodeValues: expect.objectContaining({ "actor-a": "subject" }),
      edgeValues: expect.objectContaining({ "membership-a": "evidence" }),
    }));

    fireEvent.click(screen.getByRole("button", { name: "本地确认" }));
    await waitFor(async () => {
      const events = await repository.listEvents(graph.id);
      expect(events).toHaveLength(1);
      expect(events[0]).toMatchObject({
        type: "local_review_recorded",
        payload: expect.objectContaining({ decision: "confirmed" }),
      });
    });
    expect(screen.getByText(/本地人工复核：已确认/)).toBeInTheDocument();
    expect(screen.getByText(/不会改写服务器 finding/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "导出 JSON 报告" }));
    fireEvent.click(screen.getByRole("button", { name: "导出 Markdown 报告" }));
    expect(onReportExport).toHaveBeenCalledTimes(2);
    expect(onReportExport.mock.calls.map((call) => call[1])).toEqual(["json", "markdown"]);
    expect(onReportExport.mock.calls[0]![0].reportHash)
      .toBe(onReportExport.mock.calls[1]![0].reportHash);
    expect(onReportExport.mock.calls[0]![0].generatedAt)
      .toBe(onReportExport.mock.calls[1]![0].generatedAt);
  });

  it("renders community regression uncertainty as residual coverage, never as a probability", async () => {
    const { client } = successfulClient({
      taskId: "core.community_resilience_review",
      findingType: "community-resilience-candidate",
      entityType: "community",
      subjectIds: ["community-a"],
    });
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="community-a"
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    fireEvent.click(screen.getByRole("button", { name: "社区韧性复核" }));
    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));

    expect(await screen.findByText("模型发现 #1")).toBeInTheDocument();
    expect(screen.getByText("回归区间（非概率）")).toBeInTheDocument();
    expect(screen.getByText("0.1000 – 0.4000")).toBeInTheDocument();
    expect(screen.getByText(/验证残差覆盖率 90.00%/)).toBeInTheDocument();
    expect(screen.getByText(/不是概率.*待人工复核/)).toBeInTheDocument();
    expect(screen.queryByText("校准置信度")).not.toBeInTheDocument();
  });

  it("labels a signed edge finding as relation review rather than a risk candidate", async () => {
    const { client } = successfulClient({
      findingType: "signed-relation-review",
      entityType: "edge",
      subjectIds: ["actor-a", "community-a"],
    });
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));

    expect(await screen.findByText("有符号关系复核（待人工复核）")).toBeInTheDocument();
    expect(screen.queryByText("风险候选（待人工复核）")).not.toBeInTheDocument();
  });

  it("blocks an already-recorded directed collaboration pair and requires a missing pair", () => {
    const { client } = successfulClient();
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={["actor-a", "community-a"]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    fireEvent.click(screen.getByRole("button", { name: "协作关系补全" }));

    expect(screen.getByText(/已有同向记录关系.*请选择尚未记录的节点对/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "运行治理复核" })).toBeDisabled();
    expect(client.createRun).not.toHaveBeenCalled();
  });

  it("keeps Penn94-scale target DOM bounded while accepting an exact off-suggestion node ID", () => {
    const nodes = Array.from({ length: 2_000 }, (_value, index) => ({
      id: `node-${index.toString().padStart(4, "0")}`,
      label: `Node ${index}`,
      type: "person",
      attributes: {},
    }));
    const edges = Array.from({ length: 1_999 }, (_value, index) => ({
      id: `edge-${index.toString().padStart(4, "0")}`,
      source: nodes[index]!.id,
      target: nodes[index + 1]!.id,
      type: "collaborates",
      weight: 1,
      directed: true,
      attributes: {},
    }));
    const largeGraph = createGraphVersion("penn94-scale.json", nodes, edges);
    let fullEdgeMapCalls = 0;
    const instrumentedEdges = new Proxy(largeGraph.edges, {
      get(target, property, receiver) {
        if (property === "map") {
          return (...args: Parameters<typeof target.map>) => {
            fullEdgeMapCalls += 1;
            return target.map(...args);
          };
        }
        return Reflect.get(target, property, receiver);
      },
    });
    const instrumentedGraph = { ...largeGraph, edges: instrumentedEdges };
    render(<CoreWorkbenchPanel
      graph={instrumentedGraph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={successfulClient().client}
      selectedNodeId={null}
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    expect(document.querySelectorAll("option").length).toBeLessThanOrEqual(301);
    fireEvent.change(screen.getByLabelText("节点 ID"), { target: { value: "node-1999" } });
    expect(screen.getByRole("button", { name: "运行治理复核" })).toBeEnabled();

    fireEvent.change(screen.getByLabelText("复核对象类型"), { target: { value: "edge" } });
    const mapCallsAfterGraphIndex = fullEdgeMapCalls;
    fireEvent.change(screen.getByLabelText("关系本地 ID"), { target: { value: "not-an-edge" } });
    expect(fullEdgeMapCalls).toBe(mapCallsAfterGraphIndex);
  });

  it("keeps accepted-only models visible but never runnable", () => {
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities("accepted") }}
      client={successfulClient().client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    expect(screen.getByText(/已登记验收，但尚未 servingReady/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "运行治理复核" })).toBeDisabled();
  });

  it.each([
    ["GFM_CORE_GRAPH_VERSION_NOT_FOUND", /精确交接当前不可变 GraphVersion/],
    ["GFM_CORE_MODEL_GRAPH_INCOMPATIBLE", /服务端权威合同判定不兼容/],
    ["GFM_CORE_MODEL_NOT_INSTALLED", /模型尚未安装或不可服务/],
  ])("turns authoritative %s failures into bounded remediation without showing raw server text", async (code, guidance) => {
    const { client } = successfulClient();
    client.createRun.mockRejectedValueOnce(new SocialGraphApiError(code, "PRIVATE raw upstream path E:\\secret", 409));
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    fireEvent.click(screen.getByRole("button", { name: "风险与信任复核" }));
    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));

    expect(await screen.findByText(guidance)).toBeInTheDocument();
    expect(document.body.textContent).not.toContain("E:\\secret");
  });

  it("detaches from a stale result when task context changes and never fabricates cancellation", async () => {
    const { client, fixture } = successfulClient();
    let resolveResult!: (value: typeof fixture.result) => void;
    client.getResult.mockReturnValueOnce(new Promise((resolve) => { resolveResult = resolve; }));
    const onOverlayChange = vi.fn();
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={onOverlayChange}
    />);

    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));
    await waitFor(() => expect(client.getResult).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "社区韧性复核" }));
    await act(async () => { resolveResult(fixture.result); });

    expect(screen.queryByText(/模型发现 #/)).not.toBeInTheDocument();
    expect(onOverlayChange).not.toHaveBeenCalledWith(expect.objectContaining({ kind: "governance" }));
    expect(Object.keys(client)).toEqual(["capabilities", "createRun", "getRun", "getResult"]);
    expect(document.body.textContent).not.toMatch(/取消推理/u);
  });

  it("does not offer stop-following before the server returns a run ID", () => {
    const { client } = successfulClient();
    client.createRun.mockReturnValueOnce(new Promise(() => {}));
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      repository={createLocalGraphRepository({ forceMemory: true })}
      onOverlayChange={vi.fn()}
    />);

    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));

    expect(screen.getByText("正在提交静态图复核…")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "停止跟踪" })).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/取消|已停止/u);
  });

  it("disables report export while a local review append is in flight", async () => {
    const { client } = successfulClient();
    const baseRepository = createLocalGraphRepository({ forceMemory: true });
    let resolveAppend!: () => void;
    const repository = {
      listEvents: baseRepository.listEvents.bind(baseRepository),
      appendEvent: vi.fn().mockReturnValue(new Promise<void>((resolve) => { resolveAppend = resolve; })),
    };
    render(<CoreWorkbenchPanel
      graph={graph}
      service={{ state: "connected", capabilities: capabilities() }}
      client={client}
      selectedNodeId="actor-a"
      pathEndpointIds={[]}
      repository={repository}
      onOverlayChange={vi.fn()}
      onReportExport={vi.fn()}
    />);
    fireEvent.click(screen.getByRole("button", { name: "运行治理复核" }));
    await screen.findByText("模型发现 #1");

    fireEvent.click(screen.getByRole("button", { name: "本地确认" }));

    expect(screen.getByRole("button", { name: "导出 JSON 报告" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "导出 Markdown 报告" })).toBeDisabled();
    resolveAppend();
    await waitFor(() => expect(screen.getByRole("button", { name: "导出 JSON 报告" })).toBeEnabled());
  });
});
