import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { type ReactNode, useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WorkspaceDrawer } from "./WorkspaceDrawer";

function DrawerHarness({ children, onClose = vi.fn() }: { readonly children: ReactNode; readonly onClose?: () => void }) {
  const [open, setOpen] = useState(false);
  const close = () => {
    setOpen(false);
    onClose();
  };

  return (
    <>
      <button type="button" onClick={() => setOpen(true)}>打开抽屉</button>
      {open ? (
        <WorkspaceDrawer title="测试抽屉" onClose={close}>
          {children}
        </WorkspaceDrawer>
      ) : null}
    </>
  );
}

afterEach(cleanup);

describe("WorkspaceDrawer", () => {
  it("traps keyboard focus, closes on Escape, and restores the opener focus", () => {
    const onClose = vi.fn();
    render(<DrawerHarness onClose={onClose}><button type="button">抽屉操作</button></DrawerHarness>);

    const opener = screen.getByRole("button", { name: "打开抽屉" });
    opener.focus();
    fireEvent.click(opener);

    const closeButton = screen.getByRole("button", { name: "关闭测试抽屉" });
    const drawerAction = screen.getByRole("button", { name: "抽屉操作" });
    expect(closeButton).toHaveFocus();

    fireEvent.keyDown(closeButton, { key: "Tab", shiftKey: true });
    expect(drawerAction).toHaveFocus();
    fireEvent.keyDown(drawerAction, { key: "Tab" });
    expect(closeButton).toHaveFocus();

    fireEvent.keyDown(closeButton, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(opener).toHaveFocus();
  });

  it("cycles between the first and last visible controls without entering a closed details section", () => {
    render(
      <DrawerHarness>
        <details>
          <summary>更多操作</summary>
          <button type="button">关闭时不可聚焦的操作</button>
        </details>
        <button type="button">第一个可见操作</button>
        <button type="button">最后一个可见操作</button>
      </DrawerHarness>,
    );

    fireEvent.click(screen.getByRole("button", { name: "打开抽屉" }));
    const closeButton = screen.getByRole("button", { name: "关闭测试抽屉" });
    const lastAction = screen.getByRole("button", { name: "最后一个可见操作" });

    fireEvent.keyDown(closeButton, { key: "Tab", shiftKey: true });
    expect(lastAction).toHaveFocus();
    fireEvent.keyDown(lastAction, { key: "Tab" });
    expect(closeButton).toHaveFocus();
  });

  it("treats a visible summary as the focus boundary when its details section is closed", () => {
    render(
      <DrawerHarness>
        <details>
          <summary>更多操作</summary>
          <button type="button">关闭时不可聚焦的操作</button>
        </details>
      </DrawerHarness>,
    );

    fireEvent.click(screen.getByRole("button", { name: "打开抽屉" }));
    const closeButton = screen.getByRole("button", { name: "关闭测试抽屉" });
    const summary = screen.getByText("更多操作");
    summary.focus();
    fireEvent.keyDown(summary, { key: "Tab" });
    expect(closeButton).toHaveFocus();
    fireEvent.keyDown(closeButton, { key: "Tab", shiftKey: true });
    expect(summary).toHaveFocus();
  });
});
