import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WorkspaceResizeHandle } from "./WorkspaceResizeHandle";

afterEach(() => document.body.replaceChildren());

describe("WorkspaceResizeHandle", () => {
  it("releases pointer capture when a resize is cancelled", () => {
    const onDelta = vi.fn();
    render(
      <WorkspaceResizeHandle
        axis="vertical"
        label="调整图谱栏宽度"
        value={460}
        minimum={340}
        maximum={760}
        onDelta={onDelta}
        onReset={vi.fn()}
      />,
    );
    const separator = screen.getByRole("separator", { name: "调整图谱栏宽度" });
    const setPointerCapture = vi.fn();
    const releasePointerCapture = vi.fn();
    Object.assign(separator, {
      setPointerCapture,
      releasePointerCapture,
      hasPointerCapture: () => true,
    });

    fireEvent.pointerDown(separator, { pointerId: 9, clientX: 100 });
    fireEvent.pointerCancel(separator, { pointerId: 9 });

    expect(setPointerCapture).toHaveBeenCalledWith(9);
    expect(releasePointerCapture).toHaveBeenCalledWith(9);
    expect(onDelta).not.toHaveBeenCalled();
    expect(separator).not.toHaveClass("is-active");
  });

  it("uses 10px and shift-40px keyboard increments and reports the supplied rendered pixels", () => {
    const onDelta = vi.fn();
    const onReset = vi.fn();
    render(
      <WorkspaceResizeHandle
        axis="vertical"
        label="调整图谱栏宽度"
        value={356}
        minimum={220}
        maximum={356}
        onDelta={onDelta}
        onReset={onReset}
      />,
    );
    const separator = screen.getByRole("separator", { name: "调整图谱栏宽度" });

    expect(separator).toHaveAttribute("aria-valuenow", "356");
    fireEvent.keyDown(separator, { key: "ArrowLeft" });
    fireEvent.keyDown(separator, { key: "ArrowRight", shiftKey: true });
    fireEvent.doubleClick(separator);

    expect(onDelta).toHaveBeenNthCalledWith(1, -10);
    expect(onDelta).toHaveBeenNthCalledWith(2, 40);
    expect(onReset).toHaveBeenCalledTimes(1);
  });
});
