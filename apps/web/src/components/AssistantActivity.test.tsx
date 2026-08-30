import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { AssistantActivityView } from "./AssistantActivity";

afterEach(cleanup);

describe("AssistantActivityView", () => {
  it("shows one compact live thinking status while work is running", () => {
    render(<AssistantActivityView activity={{ kind: "governance", state: "working" }} />);

    expect(screen.getByRole("status")).toHaveTextContent("思考中");
    expect(screen.getByRole("status")).toHaveTextContent("正在核对治理上下文");
  });

  it("collapses completed work into auditable high-level phases without internals", () => {
    render(<AssistantActivityView activity={{ kind: "governance", state: "completed" }} />);

    expect(screen.getByText("思考已完成")).toBeInTheDocument();
    expect(screen.getByText("理解问题")).toBeInTheDocument();
    expect(screen.getByText("核对图谱与证据")).toBeInTheDocument();
    expect(screen.getByText("整理复核建议")).toBeInTheDocument();
    expect(screen.queryByText(/模型|工具调用|GFM|hash/i)).not.toBeInTheDocument();
  });
});
