import { describe, expect, it } from "vitest";

import { createValidatedCoreFixture } from "../test/fixtures/core";
import { createLocalReviewRecord } from "./localReview";
import {
  buildCoreReport,
  coreFindingLabel,
  serializeCoreReportJson,
  serializeCoreReportMarkdown,
} from "./coreReport";

describe("deterministic Core report", () => {
  it("exports exact provenance, evidence, limitations, local reviews, and safe Core wording", () => {
    const fixture = createValidatedCoreFixture({
      graphVersionId: "graph-v1",
      pathNodeIds: ["a"],
      pathEdgeIds: ["6".repeat(64)],
      includeSimilarCase: true,
    });
    const review = createLocalReviewRecord({
      findingHash: fixture.finding.findingHash,
      runId: fixture.binding.runId,
      resultHash: fixture.result.resultHash,
      graphVersionId: fixture.binding.graphVersionId,
      sessionId: "session-1",
      decision: "confirmed",
      reviewedAt: "2026-08-15T02:00:00.000Z",
    });
    const report = buildCoreReport({
      binding: fixture.binding,
      result: fixture.result,
      finding: fixture.finding,
      reviews: [review],
    });

    expect(report).toMatchObject({
      schemaVersion: "socialgraph-fm.core-report/2.0",
      taskLabel: "风险候选（待人工复核）",
      runId: fixture.binding.runId,
      publicRequestHash: fixture.binding.publicRequestHash,
      serverRequestHash: fixture.binding.serverRequestHash,
      resultHash: fixture.result.resultHash,
      findingHash: fixture.finding.findingHash,
      reviewStatus: "pending-human-review",
      localReviews: [review],
      generatedAt: review.reviewedAt,
    });
    const json = serializeCoreReportJson(report);
    const markdown = serializeCoreReportMarkdown(report);
    expect(buildCoreReport({
      binding: fixture.binding,
      result: fixture.result,
      finding: fixture.finding,
      reviews: [review],
    })).toEqual(report);
    expect(serializeCoreReportJson(report)).toBe(json);
    expect(markdown).toContain("风险候选（待人工复核）");
    expect(markdown).toContain("不授权自动处罚或执法");
    expect(markdown).toContain("非因果");
    expect(markdown).toContain("不预测未来事件");
    expect(markdown).toContain(fixture.finding.evidence[0]!.evidenceHash);
    expect(markdown).toContain(fixture.finding.evidence[0]!.modelScoreHash!);
    expect(markdown).toContain(fixture.finding.evidence[1]!.algorithmConfigHash!);
    if (fixture.finding.calibratedConfidence.schemaVersion !== "socialgraph-fm.core-calibrated-confidence/2.0") {
      throw new Error("expected calibrated probability fixture");
    }
    expect(markdown).toContain(fixture.finding.calibratedConfidence.calibrationArtifactHash);
    expect(markdown).toContain(fixture.finding.calibratedConfidence.calibrationProtocolHash);
    expect(markdown).toContain(fixture.finding.similarCases[0]!.structuralRecordHash);
    expect(markdown).toContain(fixture.finding.similarCases[0]!.sourceGraphVersionHash);
    expect(markdown).toContain(fixture.finding.similarCases[0]!.queryHash);
    expect(markdown).toContain(fixture.binding.graphVersionId);
    expect(markdown).toContain(review.sessionId!);
    expect(markdown).toContain(review.recordHash);
    expect(markdown).not.toMatch(/Penn94|gender|性别预测/iu);
    expect(report.evidence[0]).not.toHaveProperty("value");
  });

  it("uses the required closed finding-specific labels without calling a signed edge a risk candidate", () => {
    expect(coreFindingLabel("core-collaboration-completion"))
      .toBe("静态关系补全建议");
    expect(coreFindingLabel("node-risk-candidate"))
      .toBe("风险候选（待人工复核）");
    expect(coreFindingLabel("signed-relation-review"))
      .toBe("有符号关系复核（待人工复核）");
    expect(coreFindingLabel("community-resilience-candidate"))
      .toBe("社区韧性结构复核");
  });

  it("exports community residual intervals with explicit non-probability coverage semantics", () => {
    const fixture = createValidatedCoreFixture({
      graphVersionId: "graph-v1",
      taskId: "core.community_resilience_review",
      findingType: "community-resilience-candidate",
      entityType: "community",
      subjectIds: ["community-a"],
    });
    const report = buildCoreReport({
      binding: fixture.binding,
      result: fixture.result,
      finding: fixture.finding,
      reviews: [],
    });
    const markdown = serializeCoreReportMarkdown(report);

    expect(report.calibratedConfidence).toMatchObject({
      schemaVersion: "socialgraph-fm.core-regression-confidence-interval/1.0",
      lowerBound: 0.1,
      upperBound: 0.4,
      coverage: 0.9,
      validationCount: 32,
    });
    expect(report.safetyStatements.join(" ")).toMatch(/验证残差覆盖.*不是概率.*人工复核/u);
    expect(markdown).toContain("回归区间与验证残差覆盖（非概率）");
    expect(markdown).toContain("confidenceArtifactHash");
    expect(markdown).not.toContain("校准置信度不是违规、风险或事实为真的概率");
  });

  it("keeps hostile graph identifiers inside canonical JSON code context", () => {
    const hostile = "node`\n# forged heading\n<script>alert(1)</script>";
    const fixture = createValidatedCoreFixture({
      graphVersionId: "graph-v1",
      subjectIds: [hostile],
    });
    const report = buildCoreReport({
      binding: fixture.binding,
      result: fixture.result,
      finding: fixture.finding,
      reviews: [],
    });
    const markdown = serializeCoreReportMarkdown(report);

    expect(markdown).toContain("\\n# forged heading\\n");
    expect(markdown.split("\n")).not.toContain("# forged heading");
    expect(markdown.split("\n").filter((line) => line.includes("<script>")))
      .toEqual(expect.arrayContaining([expect.stringMatching(/^ {4}/u)]));
  });
});
