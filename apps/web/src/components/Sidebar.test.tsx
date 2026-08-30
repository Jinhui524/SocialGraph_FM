import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Sidebar, type SidebarSessionItem } from "./Sidebar";

const sessions: readonly SidebarSessionItem[] = [
  { id: "research", title: "高校科研团队协作网络分析", time: "10:24" },
  { id: "volunteer", title: "社区志愿者关系图谱构建", time: "昨天" },
  { id: "industry", title: "科技企业合作伙伴识别", time: "08-06" },
];

afterEach(cleanup);

function renderSidebar(overrides: { collapsed?: boolean } = {}) {
  const callbacks = {
    onWorkspaceChange: vi.fn(),
    onSessionChange: vi.fn(),
    onNewSession: vi.fn(),
    onToggleCollapsed: vi.fn(),
    onShowSessions: vi.fn(),
    onRenameSession: vi.fn(),
    onDuplicateSession: vi.fn(),
    onTrashSession: vi.fn(),
    onOpenDatasets: vi.fn(),
    onSupportAction: vi.fn(),
  };

  const rendered = render(
    <Sidebar
      activeWorkspace="chat"
      activeSession="research"
      sessions={sessions}
      collapsed={overrides.collapsed}
      {...callbacks}
    />,
  );
  return { ...callbacks, container: rendered.container };
}

describe("Sidebar recent-session search", () => {
  it("opens a real title search and filters sessions", () => {
    renderSidebar();

    fireEvent.click(screen.getByRole("button", { name: "搜索最近会话" }));
    const input = screen.getByRole("searchbox", { name: "搜索最近会话" });
    fireEvent.change(input, { target: { value: "志愿" } });

    expect(screen.getByText("社区志愿者关系图谱构建")).toBeInTheDocument();
    expect(screen.queryByText("高校科研团队协作网络分析")).not.toBeInTheDocument();
    expect(screen.queryByText("科技企业合作伙伴识别")).not.toBeInTheDocument();
  });

  it("shows an explicit empty state and clears the query", () => {
    renderSidebar();
    fireEvent.click(screen.getByRole("button", { name: "搜索最近会话" }));
    fireEvent.change(screen.getByRole("searchbox", { name: "搜索最近会话" }), {
      target: { value: "不存在的会话" },
    });

    expect(screen.getByText("没有匹配的会话")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "清空会话搜索" }));
    expect(screen.getByText("高校科研团队协作网络分析")).toBeInTheDocument();
  });

  it("supports Ctrl+K and Escape with focus restoration", async () => {
    renderSidebar();

    fireEvent.keyDown(window, { key: "k", ctrlKey: true });
    await waitFor(() => expect(screen.getByRole("searchbox", { name: "搜索最近会话" })).toHaveFocus());

    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("searchbox", { name: "搜索最近会话" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "搜索最近会话" })).toHaveFocus();
  });

  it("uses an accessible floating search when the sidebar is collapsed", () => {
    const callbacks = renderSidebar({ collapsed: true });

    fireEvent.click(screen.getByRole("button", { name: "搜索最近会话" }));

    expect(callbacks.onToggleCollapsed).toHaveBeenCalledTimes(1);
    const dialog = screen.getByRole("dialog", { name: "搜索最近会话" });
    expect(dialog).toBeInTheDocument();
    fireEvent.change(within(dialog).getByRole("searchbox", { name: "搜索最近会话" }), {
      target: { value: "企业" },
    });
    fireEvent.click(within(dialog).getByRole("button", { name: /科技企业合作伙伴识别/ }));
    expect(callbacks.onSessionChange).toHaveBeenCalledWith("industry");
  });

  it("opens the real dataset workspace entry", () => {
    const callbacks = renderSidebar();
    fireEvent.click(screen.getByRole("button", { name: "数据管理" }));
    expect(callbacks.onOpenDatasets).toHaveBeenCalledTimes(1);
  });

  it("offers only released destinations and opens the adaptation workspace", () => {
    const callbacks = renderSidebar();
    fireEvent.click(screen.getByRole("button", { name: "治理应用" }));
    expect(callbacks.onWorkspaceChange).toHaveBeenCalledWith("governance");

    fireEvent.click(screen.getByRole("button", { name: "适配能力" }));
    expect(callbacks.onWorkspaceChange).toHaveBeenCalledWith("adaptation");

    expect(screen.queryByRole("button", { name: "图谱库" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "分析工具" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "实验记录" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "模板市场" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "帮助中心" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "意见反馈" })).not.toBeInTheDocument();
  });

  it("makes the underlying session list inert while compact search is open", () => {
    const { container } = renderSidebar({ collapsed: true });
    fireEvent.click(screen.getByRole("button", { name: "搜索最近会话" }));
    expect(container.querySelector(".session-list")).toHaveAttribute("inert");
  });
});
