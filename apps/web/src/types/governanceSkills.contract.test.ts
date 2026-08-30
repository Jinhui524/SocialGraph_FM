import { describe, expect, it } from "vitest";

import {
  GOVERNANCE_CONFIRMATION_GATED_SKILLS,
  GOVERNANCE_PRODUCT_SKILL_NAMESPACE,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_READ_ONLY_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
  GOVERNANCE_SKILL_POLICIES,
  type GovernanceSkillParameters,
} from "./governanceSkills";

describe("generated SocialGraph-FM Governance product-skill contract", () => {
  it("keeps a distinct namespace and the stable wire version", () => {
    expect(GOVERNANCE_PRODUCT_SKILL_NAMESPACE).toBe(
      "socialgraph-fm.product-skills.governance",
    );
    expect(GOVERNANCE_SKILLS_SCHEMA).toBe("socialgraph-fm.governance-skills/1.0");
  });

  it("partitions all eight skills into six read-only and two gated operations", () => {
    expect(GOVERNANCE_PUBLIC_SKILLS).toHaveLength(8);
    expect(GOVERNANCE_READ_ONLY_SKILLS).toHaveLength(6);
    expect(GOVERNANCE_CONFIRMATION_GATED_SKILLS).toEqual([
      "run_governance_analysis",
      "draft_review_report",
    ]);
    expect(GOVERNANCE_SKILL_POLICIES.map((item) => item.name)).toEqual(
      GOVERNANCE_PUBLIC_SKILLS,
    );
    expect(
      GOVERNANCE_SKILL_POLICIES.filter((item) => item.readOnly).map((item) => item.name),
    ).toEqual(GOVERNANCE_READ_ONLY_SKILLS);
    expect(
      GOVERNANCE_SKILL_POLICIES.filter((item) => item.confirmationRequired).map(
        (item) => item.name,
      ),
    ).toEqual(GOVERNANCE_CONFIRMATION_GATED_SKILLS);
  });

  it("generates parameter types from each referenced public JSON Schema", () => {
    const run: GovernanceSkillParameters["run_governance_analysis"] = {
      protocol: "global",
      topK: 100,
    };
    const relation: GovernanceSkillParameters["rank_coordination_relations"] = {
      runId: `governance-${"1".repeat(32)}`,
      relationKind: "factual",
      modalities: ["coRT"],
    };
    expect(run.protocol).toBe("global");
    expect(relation.modalities).toEqual(["coRT"]);
  });
});
