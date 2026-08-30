import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SafeMarkdown } from "./SafeMarkdown";

describe("SafeMarkdown", () => {
  it("renders concise GFM markdown without raw HTML or images", () => {
    const { container } = render(<SafeMarkdown text={'## 治理摘要\n\n- 候选 A\n\n<img src="https://example.com/x.png" onerror="alert(1)">\n\n|节点|分数|\n|---|---|\n|A|0.9|'} />);
    expect(screen.getByRole("heading", { name: "治理摘要" })).toBeInTheDocument();
    expect(screen.getByText("候选 A")).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(container.querySelector("img")).toBeNull();
  });

  it("removes unsafe and non-https external links", () => {
    render(<SafeMarkdown text={'[危险](javascript:alert(1)) [数据](data:text/html;base64,PHNjcmlwdD4=) [邮件](mailto:user@example.com) [明文](http://example.com) [绕过](//evil.example/path) [安全](https://example.com)'} />);
    expect(screen.getByText("危险").closest("a")).toBeNull();
    expect(screen.getByText("数据").closest("a")).toBeNull();
    expect(screen.getByText("邮件").closest("a")).toBeNull();
    expect(screen.getByText("明文").closest("a")).toBeNull();
    expect(screen.getByText("绕过").closest("a")).toBeNull();
    expect(screen.getByRole("link", { name: "安全" })).toHaveAttribute("href", "https://example.com/");
  });

  it("allows local navigation while suppressing markdown images and raw executable HTML", () => {
    const { container } = render(<SafeMarkdown text={'[站内](/governance) [锚点](#evidence) ![远端像素](https://example.com/pixel.png) <script>alert(1)</script>'} />);
    expect(screen.getByRole("link", { name: "站内" })).toHaveAttribute("href", "/governance");
    expect(screen.getByRole("link", { name: "锚点" })).toHaveAttribute("href", "#evidence");
    expect(container.querySelector("img")).toBeNull();
    expect(container.querySelector("script")).toBeNull();
  });
});
