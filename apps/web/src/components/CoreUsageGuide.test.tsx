import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { CoreUsageGuide } from "./CoreUsageGuide";

describe("CoreUsageGuide", () => {
  it("keeps the public guide focused on the three production steps", () => {
    render(<CoreUsageGuide />);
    expect(screen.getByText("上传关系数据")).toBeInTheDocument();
    expect(screen.getByText("用自然语言研究")).toBeInTheDocument();
    expect(screen.getByText("进入治理应用复核")).toBeInTheDocument();
    expect(screen.queryByText(/legacy|ViewCommand|smoke/i)).not.toBeInTheDocument();
  });
});
