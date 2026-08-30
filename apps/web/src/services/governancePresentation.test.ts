import { describe, expect, it } from "vitest";

import { governanceAccountLabel, governanceLimitationLabel, governanceModalityLabel } from "./governancePresentation";

describe("SocialGraph-FM Governance business presentation", () => {
  it("translates internal relation modalities into readable graph labels", () => {
    expect(governanceModalityLabel("coRT")).toBe("协同转发");
    expect(governanceModalityLabel("coURL")).toBe("共链传播");
    expect(governanceModalityLabel("hashSeq")).toBe("话题序列");
    expect(governanceModalityLabel("fastRT")).toBe("快速转发");
    expect(governanceModalityLabel("tweetSim")).toBe("内容相似");
    expect(governanceModalityLabel("fused")).toBe("综合关系");
  });

  it("turns serving limitations into concise analyst-facing Chinese guidance", () => {
    expect(governanceLimitationLabel("Scores are analyst-facing risk candidates and never automatic enforcement decisions."))
      .toBe("风险分数仅用于安排人工复核顺序，不作为自动处置结论。");
    expect(governanceLimitationLabel("The two-hop view is capped at 300 nodes and 1000 factual edges; relation weights are explanation-only and do not prove coordination or intent."))
      .toBe("两跳证据视图最多展示 300 个节点和 1,000 条事实关系；关系权重仅用于说明已登记连接，不能证明协同行为或主观意图。");
    expect(governanceLimitationLabel("Louvain community priority is derived from member risk, not proof of coordination."))
      .toBe("群组复核顺序由成员风险信号派生，不构成成员协同行为的直接证明。");
    expect(governanceLimitationLabel("Derived analyst priority over a factual input relation; it is not proof of coordination."))
      .toBe("该优先级从已登记事实关系派生，仅用于安排人工核验，不构成协同行为证明。");
    expect(governanceLimitationLabel("Bounded same-community similarity lead; this is not a factual or future edge."))
      .toBe("该线索来自同群组内的有界相似度计算，既不是已登记事实关系，也不代表未来一定形成关系。");
    expect(governanceLimitationLabel("风险候选必须由人工复核。"))
      .toBe("风险候选必须由人工复核。");
    expect(governanceLimitationLabel(undefined))
      .toBe("证据用于支持人工复核，不构成自动处置结论。");
  });

  it("localizes anonymous account labels without changing stable identifiers", () => {
    expect(governanceAccountLabel("Anonymous account 458", "russia:458")).toBe("匿名账号 458");
    expect(governanceAccountLabel("Account 28", "russia:28")).toBe("账号 28");
    expect(governanceAccountLabel("重点账号", "russia:458")).toBe("重点账号");
    expect(governanceAccountLabel(undefined, "russia:458")).toBe("russia:458");
  });
});
