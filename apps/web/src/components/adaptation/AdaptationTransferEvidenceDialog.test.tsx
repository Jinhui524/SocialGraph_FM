import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { targetComparison, targetEvidence, targetPolicy, targetResult, targetTaskRegistration } from "../../test/fixtures/governanceTargetTask";
import { globalModelModelCard } from "../../test/fixtures/globalModel";
import type { GlobalModelModelCard } from "../../types/globalModel";
import type { TargetAdaptationComparison, GovernanceOnlineEvidence, GovernanceOnlineResult, TargetReviewPolicy, TargetTaskRegistration } from "../../types/governanceOnline";
import { buildAdaptationTransferEvidence } from "./AdaptationTransferEvidence";
import { AdaptationTransferEvidenceDialog } from "./AdaptationTransferEvidenceDialog";

afterEach(cleanup);

function transferEvidence() {
  const result = targetResult() as unknown as GovernanceOnlineResult;
  const card = globalModelModelCard();
  const state = buildAdaptationTransferEvidence({
    result,
    registration: targetTaskRegistration() as unknown as TargetTaskRegistration,
    modelCardState: {
      status: "ready",
      card: {
        ...card,
        modelVersionId: result.modelVersionId,
        modelVersionHash: result.modelVersionHash,
        protocols: { ...card.protocols, global: { ...card.protocols.global, modelVersionId: result.modelVersionId, modelVersionHash: result.modelVersionHash, modelStateHash: result.modelStateHash } },
      } as GlobalModelModelCard,
    },
    selectedNodeId: "target-node-001",
    selectedEvidence: targetEvidence("target-node-001") as unknown as GovernanceOnlineEvidence,
    policy: targetPolicy() as unknown as TargetReviewPolicy,
    comparison: targetComparison() as unknown as TargetAdaptationComparison,
  });
  if (state.status !== "ready") throw new Error(state.reason);
  return state.value;
}

function Harness() {
  const [open, setOpen] = useState(false);
  return <><button type="button" onClick={() => setOpen(true)}>打开迁移依据</button><AdaptationTransferEvidenceDialog open={open} lane="few_shot" evidence={transferEvidence()} nodeCount={108} relationCount={220} onClose={() => setOpen(false)} /></>;
}

describe("AdaptationTransferEvidenceDialog", () => {
  it("presents anonymous routing, selected-object evidence, and few-shot calibration without geographic copy", () => {
    render(<AdaptationTransferEvidenceDialog open lane="few_shot" evidence={transferEvidence()} nodeCount={108} relationCount={220} onClose={() => undefined} />);
    const dialog = screen.getByRole("dialog", { name: "少样本源域路由与校正" });

    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(within(dialog).getByRole("table", { name: "匿名专家路由明细" })).toBeVisible();
    expect(within(dialog).getAllByRole("row")).toHaveLength(8);
    expect(within(dialog).getAllByText("源域专家 04")).toHaveLength(2);
    expect(within(dialog).getByRole("region", { name: "当前选中账号的迁移依据" })).toHaveTextContent("对象 1");
    expect(within(dialog).getByText("λ 0.50")).toBeVisible();
    expect(within(dialog).getByText(/校正只改变人工复核顺序/u)).toBeVisible();
    expect(dialog.textContent).not.toMatch(/china|cuba|iran|russia|UAE|venezuela/u);
    expect(dialog).toHaveTextContent("不是风险概率、因果贡献或人工结论");
  });

  it("closes with Escape and restores focus to the trigger", async () => {
    render(<Harness />);
    const trigger = screen.getByRole("button", { name: "打开迁移依据" });
    trigger.focus();
    fireEvent.click(trigger);
    const dialog = screen.getByRole("dialog", { name: "少样本源域路由与校正" });
    await waitFor(() => expect(within(dialog).getByRole("button", { name: "关闭迁移依据" })).toHaveFocus());
    fireEvent.keyDown(dialog, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
  });
});
