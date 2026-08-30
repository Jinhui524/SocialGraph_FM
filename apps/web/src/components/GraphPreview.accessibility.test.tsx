import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalysisOverlay, GraphVersion } from "../types/graph";

vi.mock("../services/graphRenderer", () => ({
  detectWebGLSupport: () => false,
  resolveGraphRendererKind: () => "canvas",
  // Keep the renderer pending: these tests exercise React controls and never
  // need to construct a G6 canvas in JSDOM.
  loadGraphRenderer: vi.fn(() => new Promise(() => undefined)),
}));

import GraphPreview, { graphRendererFallbackWarning } from "./GraphPreview";

const graphVersion: GraphVersion = {
  id: "graph-a11y",
  sourceFile: "a11y.csv",
  createdAt: "2026-08-10T00:00:00.000Z",
  nodes: [
    { id: "u-1", label: "研究者甲", type: "人员", attributes: {} },
    { id: "o-1", label: "实验室", type: "组织", attributes: {} },
  ],
  edges: [
    {
      id: "e-1",
      source: "u-1",
      target: "o-1",
      type: "隶属",
      attributes: {},
    },
  ],
  summary: {
    nodeCount: 2,
    edgeCount: 1,
    density: 1,
    averageDegree: 1,
    connectedComponents: 1,
    isolatedNodes: 0,
  },
  issues: [],
  preview: {
    nodes: [
      { id: "u-1", label: "研究者甲", type: "人员", attributes: {} },
      { id: "o-1", label: "实验室", type: "组织", attributes: {} },
    ],
    edges: [
      {
        id: "e-1",
        source: "u-1",
        target: "o-1",
        type: "隶属",
        attributes: {},
      },
    ],
    truncated: false,
    originalNodeCount: 2,
    originalEdgeCount: 1,
  },
  truncated: false,
};

describe("GraphPreview accessibility controls", () => {
  beforeEach(() => {
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    vi.stubGlobal("cancelAnimationFrame", vi.fn());
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("describes automatic renderer fallback without exposing technical controls", () => {
    expect(
      graphRendererFallbackWarning({ fallbackReason: "WEBGL_UNSUPPORTED" }),
    ).toEqual({
      label: "已切换兼容渲染",
      title: "系统已自动切换到稳定渲染模式，图谱功能不受影响。",
    });
    expect(graphRendererFallbackWarning({})).toBeNull();
  });

  it("supports a contextual empty state without changing the ordinary import guidance", () => {
    const contextual = render(<GraphPreview graphVersion={null} emptyState={{
      title: "等待目标域图谱",
      description: "登记目标域任务包后，将在这里显示零样本与适配后图谱。",
    }} />);
    expect(screen.getByRole("status")).toHaveTextContent("等待目标域图谱");
    expect(screen.getByRole("status")).toHaveTextContent("登记目标域任务包后，将在这里显示零样本与适配后图谱。");
    expect(screen.queryByText("上传 CSV 或 JSON 后将在这里生成真实预览")).not.toBeInTheDocument();
    contextual.unmount();

    render(<GraphPreview graphVersion={null} />);
    expect(screen.getByRole("status")).toHaveTextContent("等待图谱数据");
    expect(screen.getByRole("status")).toHaveTextContent("上传 CSV 或 JSON 后将在这里生成真实预览");
  });

  it("renders a product switcher in a full-width second header row", () => {
    const onZero = vi.fn();
    const { container } = render(<GraphPreview
      graphVersion={graphVersion}
      headerAccessory={<div className="adaptation-graph-switcher" role="group" aria-label="适配任务图谱">
        <button type="button" aria-pressed="true" data-lane="zero_shot" onClick={onZero}>零样本图谱</button>
        <button type="button" aria-pressed="false" data-lane="few_shot">少样本图谱</button>
      </div>}
    />);

    const header = container.querySelector(".graph-preview__header");
    const accessory = container.querySelector(".graph-preview__header-accessory");
    expect(header).toHaveClass("has-accessory");
    expect(accessory?.parentElement).toBe(header);
    expect(screen.getByRole("group", { name: "适配任务图谱" })).toBe(accessory?.firstElementChild);
    expect(screen.getByRole("button", { name: "零样本图谱" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "少样本图谱" })).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(screen.getByRole("button", { name: "零样本图谱" }));
    expect(onZero).toHaveBeenCalledOnce();
  });

  it("exposes the controlled summary disclosure in the graph header without a production minimap", () => {
    const onSummaryCollapsedChange = vi.fn();
    const { container, rerender } = render(
      <GraphPreview
        graphVersion={graphVersion}
        summaryCollapsed
        summaryControlsId="summary-details"
        onSummaryCollapsedChange={onSummaryCollapsedChange}
      />,
    );

    const expand = screen.getByRole("button", { name: "展开图谱摘要" });
    expect(expand).toHaveAttribute("aria-expanded", "false");
    expect(expand).toHaveAttribute("aria-controls", "summary-details");
    expect(expand).toHaveTextContent("");
    expect(expand.querySelector("svg")).not.toBeNull();
    fireEvent.click(expand);
    expect(onSummaryCollapsedChange).toHaveBeenCalledWith(false);
    expect(container.querySelector(".graph-preview__minimap")).not.toBeInTheDocument();

    rerender(
      <GraphPreview
        graphVersion={graphVersion}
        summaryCollapsed={false}
        summaryControlsId="summary-details"
        onSummaryCollapsedChange={onSummaryCollapsedChange}
      />,
    );
    const collapse = screen.getByRole("button", { name: "收起图谱摘要" });
    expect(collapse).toHaveAttribute("aria-expanded", "true");
    expect(collapse).toHaveTextContent("");
    expect(collapse).toHaveAttribute("title", "收起图谱摘要");
  });

  it("moves focus into the filter portal and restores the trigger on Escape", async () => {
    render(<GraphPreview graphVersion={graphVersion} />);
    const trigger = screen.getByRole("button", { name: /筛选节点类型/ });

    fireEvent.click(trigger);

    const firstCheckbox = screen.getByRole("checkbox", { name: /人员/ });
    await waitFor(() => expect(firstCheckbox).toHaveFocus());

    fireEvent.keyDown(document, { key: "Escape" });

    expect(screen.queryByRole("dialog", { name: "筛选节点类型" })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("closes on an outside pointer and restores the filter trigger", async () => {
    render(<GraphPreview graphVersion={graphVersion} />);
    const trigger = screen.getByRole("button", { name: /筛选节点类型/ });

    fireEvent.click(trigger);
    await waitFor(() => expect(screen.getByRole("checkbox", { name: /人员/ })).toHaveFocus());

    fireEvent.pointerDown(document.body);

    expect(screen.queryByRole("dialog", { name: "筛选节点类型" })).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });

  it("uses an Enter/candidate search flow without a separate locate button", () => {
    const onSelectNode = vi.fn();
    render(<GraphPreview graphVersion={graphVersion} onSelectNode={onSelectNode} />);
    expect(screen.queryByRole("button", { name: "定位节点" })).not.toBeInTheDocument();

    const input = screen.getByRole("textbox", { name: "按名称或 ID 搜索节点" });
    fireEvent.change(input, {
      target: { value: "研究者" },
    });

    const option = screen.getByRole("option", { name: /研究者甲/ });
    expect(option).toHaveAttribute("aria-selected", "true");
    expect(input).toHaveAttribute("aria-activedescendant", option.id);
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onSelectNode).toHaveBeenCalledWith(graphVersion.nodes[0]);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onSelectNode).toHaveBeenLastCalledWith(null);
    expect(screen.queryByRole("button", { name: "全局" })).not.toBeInTheDocument();
    expect(screen.queryByText("设置")).not.toBeInTheDocument();
  });

  it("edits a filter draft and applies it once without immediate mutation", async () => {
    const onFiltersChange = vi.fn();
    render(<GraphPreview graphVersion={graphVersion} onFiltersChange={onFiltersChange} />);
    fireEvent.click(screen.getByRole("button", { name: /筛选节点类型/ }));
    const people = await screen.findByRole("checkbox", { name: /人员/ });

    fireEvent.click(people);
    expect(onFiltersChange).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "应用筛选" }));
    expect(onFiltersChange).toHaveBeenCalledTimes(1);
    expect(onFiltersChange).toHaveBeenCalledWith({
      nodeTypes: ["组织"],
      edgeTypes: [],
    });
  });

  it("reports real constraints and never lets the last relationship type bounce back to all", async () => {
    const onFiltersChange = vi.fn();
    render(
      <GraphPreview
        graphVersion={graphVersion}
        filters={{ nodeTypes: [], edgeTypes: [], minWeight: 2, directed: true }}
        onFiltersChange={onFiltersChange}
      />,
    );
    const trigger = screen.getByRole("button", { name: /当前有 2 项约束/ });
    expect(trigger).toHaveTextContent("2");
    expect(screen.getByRole("img", { name: /关系图，当前显示 0 个节点和 0 条关系/ }))
      .toBeInTheDocument();
    fireEvent.click(trigger);
    const relationship = await screen.findByRole("checkbox", { name: /隶属/ });

    fireEvent.click(relationship);

    expect(relationship).toBeChecked();
    expect(screen.getByText(/至少保留一种关系类型/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "显示全部" }));
    fireEvent.click(screen.getByRole("button", { name: "应用筛选" }));
    expect(onFiltersChange).toHaveBeenCalledWith({ nodeTypes: [], edgeTypes: [] });
  });

  it("deduplicates relationship facts and exposes solid versus dashed semantics once", () => {
    const relationshipGraph: GraphVersion = {
      ...graphVersion,
      edges: [{ ...graphVersion.edges[0], type: "factual_relation" }],
      preview: {
        ...graphVersion.preview,
        edges: [{ ...graphVersion.preview.edges[0], type: "factual_relation" }],
      },
    };
    const overlay: AnalysisOverlay = {
      id: "relationship-overlay",
      graphVersionId: relationshipGraph.id,
      kind: "governance",
      nodeValues: {},
      edgeValues: { "e-1": "factual" },
      presentation: { governanceLens: "relations" },
      legend: {
        title: "事实关系 / 潜在线索",
        items: [
          { value: "factual", label: "事实关系", color: "#5F7896" },
          { value: "candidate", label: "潜在线索", color: "#7659EF" },
        ],
      },
      provenance: { engine: "test", algorithm: "relations" },
    };

    const { container } = render(<GraphPreview graphVersion={relationshipGraph} activeOverlay={overlay} />);
    const legend = container.querySelector(".graph-preview__legend");
    expect(legend).not.toBeNull();
    expect(legend).toHaveTextContent("事实关系");
    expect(legend).toHaveTextContent("潜在线索");
    expect(legend?.querySelectorAll('[data-relation-kind="factual"]')).toHaveLength(1);
    expect(legend?.querySelectorAll('[data-relation-kind="potential"]')).toHaveLength(1);
    expect(legend?.querySelector('[data-relation-kind="potential"]')).toHaveClass("is-dashed");
  });

  it("reacts to GovernanceFocus changes on one graph identity without recreating or relaying out", () => {
    const { container, rerender } = render(
      <GraphPreview graphVersion={graphVersion} />,
    );
    const root = container.querySelector<HTMLElement>(".graph-preview")!;
    const baseline = {
      engine: root.dataset.engineCreateCount,
      layout: root.dataset.layoutCount,
      appearance: root.dataset.appearanceRequestKey,
    };
    rerender(
      <GraphPreview
        graphVersion={graphVersion}
        governanceFocus={{ kind: "node", targetId: "u-1", nodeIds: ["u-1"], cameraToken: 1 }}
      />,
    );
    expect(root.dataset.appearanceRequestKey).not.toBe(baseline.appearance);
    const nodeAppearance = root.dataset.appearanceRequestKey;
    rerender(
      <GraphPreview
        graphVersion={graphVersion}
        governanceFocus={{
          kind: "relation",
          targetId: "relation-1",
          nodeIds: ["u-1", "o-1"],
          exactRelationKey: "o-1\u0000u-1\u0000coRT",
          cameraToken: 2,
        }}
      />,
    );
    expect(root.dataset.appearanceRequestKey).not.toBe(nodeAppearance);
    expect(root.dataset.appearanceRequestKey).toContain("relation-1");
    expect(root.dataset.engineCreateCount).toBe(baseline.engine);
    expect(root.dataset.layoutCount).toBe(baseline.layout);
  });
});
