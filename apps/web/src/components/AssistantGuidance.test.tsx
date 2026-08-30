import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { AssistantGuidance, type AssistantGuidanceState } from "./AssistantGuidance";

afterEach(cleanup);

describe("AssistantGuidance", () => {
  const states: ReadonlyArray<readonly [AssistantGuidanceState, string]> = [
    ["upload_ready", "图谱已就绪"],
    ["awaiting_confirmation", "等待你的确认"],
    ["running", "正在生成治理研判"],
    ["completed", "分析结果已固化"],
    ["evidence_followup", "进入证据复核"],
    ["failed", "本次处理未完成"],
  ];

  it.each(states)("renders %s with a consistent three-part handoff", (state, title) => {
    render(<AssistantGuidance state={state} />);

    const guidance = screen.getByRole("region", { name: "下一步指引" });
    expect(guidance).toHaveTextContent(title);
    expect(guidance).toHaveTextContent("正在做什么");
    expect(guidance).toHaveTextContent("将得到什么");
    expect(guidance).toHaveTextContent("下一步");
  });
});
