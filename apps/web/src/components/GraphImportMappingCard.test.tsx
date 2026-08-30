import "@testing-library/jest-dom/vitest";

import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { FileProfile } from "../types/graph";
import { GraphImportMappingCard, GraphTableRoleCard } from "./GraphImportMappingCard";

afterEach(cleanup);

function profile(name: string, headers: readonly string[]): FileProfile {
  return {
    name,
    size: 128,
    format: "csv",
    supported: true,
    headers,
    columns: headers.map((header) => ({
      name: header,
      inferredType: header.includes("weight") ? "number" : "string",
      missingRate: 0,
      cardinality: 2,
      nonNullCount: 2,
      nullCount: 0,
    })),
    needsMapping: false,
    issues: [],
  };
}

describe("GraphImportMappingCard", () => {
  it("preserves every optional single-edge-table mapping on apply", () => {
    const onApply = vi.fn();
    const edgeProfile = profile("relations.csv", [
      "from",
      "to",
      "from_label",
      "to_label",
      "from_type",
      "to_type",
      "relation",
      "weight",
      "year",
    ]);
    render(
      <GraphImportMappingCard
        edgeProfile={edgeProfile}
        initialEdgeMapping={{
          source: "from",
          target: "to",
          sourceLabel: "from_label",
          targetLabel: "to_label",
          sourceType: "from_type",
          targetType: "to_type",
          edgeType: "relation",
          weight: "weight",
          timestamp: "year",
        }}
        initialTimeFormat="year"
        issues={[]}
        onApply={onApply}
        onCancel={() => undefined}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /验证映射并生成草稿/ }));
    expect(onApply).toHaveBeenCalledWith({
      edgeMapping: {
        source: "from",
        target: "to",
        sourceLabel: "from_label",
        targetLabel: "to_label",
        sourceType: "from_type",
        targetType: "to_type",
        edgeType: "relation",
        weight: "weight",
        timestamp: "year",
      },
      timeFormat: "year",
    });
  });

  it("collects non-standard node id, display and entity type fields", () => {
    const onApply = vi.fn();
    render(
      <GraphImportMappingCard
        nodeProfile={profile("entities.csv", ["entity_key", "display_name", "entity_kind", "department"])}
        edgeProfile={profile("links.csv", ["actor", "object", "relation"])}
        initialEdgeMapping={{ source: "actor", target: "object", edgeType: "relation" }}
        initialTimeFormat="none"
        issues={[]}
        onApply={onApply}
        onCancel={() => undefined}
      />,
    );

    fireEvent.change(screen.getByLabelText(/唯一 ID/), { target: { value: "entity_key" } });
    fireEvent.change(screen.getByLabelText(/显示名称/), { target: { value: "display_name" } });
    fireEvent.change(screen.getByLabelText(/实体类型/), { target: { value: "entity_kind" } });
    fireEvent.click(screen.getByRole("button", { name: /验证映射并生成草稿/ }));

    expect(onApply).toHaveBeenCalledWith(expect.objectContaining({
      nodeMapping: { id: "entity_key", label: "display_name", type: "entity_kind" },
      edgeMapping: { source: "actor", target: "object", edgeType: "relation" },
      timeFormat: "none",
    }));
  });
});

describe("GraphTableRoleCard", () => {
  it("lets the user explicitly swap which file is the edge table", () => {
    const onApply = vi.fn();
    const files = [
      new File(["key\n1\n"], "first.csv", { type: "text/csv" }),
      new File(["a,b\n1,2\n"], "second.csv", { type: "text/csv" }),
    ];
    render(
      <GraphTableRoleCard
        files={files}
        profiles={[profile("first.csv", ["key"]), profile("second.csv", ["a", "b"])]}
        initialEdgeIndex={1}
        onApply={onApply}
        onCancel={() => undefined}
      />,
    );

    fireEvent.click(screen.getByLabelText(/first.csv/));
    fireEvent.click(screen.getByRole("button", { name: /确认文件角色/ }));
    expect(onApply).toHaveBeenCalledWith(0);
  });
});
