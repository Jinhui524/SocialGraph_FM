import type { GovernanceOnlineRun } from "./governanceOnline";
import {
  ASSISTANT_PRODUCT_SKILL_NAMESPACE,
  ASSISTANT_PUBLIC_SKILLS,
  ASSISTANT_SKILLS_SCHEMA,
  ASSISTANT_SKILL_REQUEST_SCHEMA,
  ASSISTANT_SKILL_RESULT_SCHEMA,
  ASSISTANT_SKILL_POLICIES,
  GOVERNANCE_CONFIRMATION_GATED_SKILLS,
  GOVERNANCE_PRODUCT_SKILL_NAMESPACE,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_READ_ONLY_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
  GOVERNANCE_SKILL_POLICIES,
  type GeneratedGovernanceReadOnlySkillName,
  type GeneratedGovernanceSkillName,
  type GeneratedGovernanceSkillParameters,
  type GeneratedGovernanceSkillParams,
} from "../generated/governanceSkillsContract";

export {
  ASSISTANT_PRODUCT_SKILL_NAMESPACE,
  ASSISTANT_PUBLIC_SKILLS,
  ASSISTANT_SKILLS_SCHEMA,
  ASSISTANT_SKILL_REQUEST_SCHEMA,
  ASSISTANT_SKILL_RESULT_SCHEMA,
  ASSISTANT_SKILL_POLICIES,
  GOVERNANCE_CONFIRMATION_GATED_SKILLS,
  GOVERNANCE_PRODUCT_SKILL_NAMESPACE,
  GOVERNANCE_PUBLIC_SKILLS,
  GOVERNANCE_READ_ONLY_SKILLS,
  GOVERNANCE_SKILLS_SCHEMA,
  GOVERNANCE_SKILL_POLICIES,
};

export type GovernanceSkillName = GeneratedGovernanceSkillName;
export type GovernanceReadOnlySkillName = GeneratedGovernanceReadOnlySkillName;
export type GovernanceSkillParameters = GeneratedGovernanceSkillParameters;
export type GovernanceSkillParams<Name extends GovernanceSkillName> =
  GeneratedGovernanceSkillParams<Name>;
export type GovernanceConfirmationAction = "run_governance_analysis" | "save_draft_report" | "submit_review";
export type GovernanceTargetKind = "node" | "relation" | "group";
export type GovernanceAssistantEvidenceSource = "graph" | "skill" | "knowledge" | "case";

export type AssistantSkillName = typeof ASSISTANT_PUBLIC_SKILLS[number];

export interface GovernanceSkillsContext {
  readonly graph: {
    readonly artifactId: string;
    readonly datasetContentHash: string;
    readonly graphVersionHash: string;
  };
  readonly model: {
    readonly modelVersionId: string;
    readonly modelStateHash: string;
  };
  readonly runId?: string;
  readonly caseId?: string;
  /** Immutable revision of the selected case. Reports fail closed when stale. */
  readonly caseHash?: string;
  /** Frontend-only prerequisite signal; it is never serialized to the API. */
  readonly caseItemCount?: number;
  readonly selectedNodeIds?: readonly string[];
  readonly selectedTarget?: {
    readonly kind: GovernanceTargetKind;
    readonly targetId: string;
  };
}

export interface GovernanceSkillDescriptor {
  readonly name: GovernanceSkillName;
  readonly readOnly: boolean;
  readonly confirmationRequired: boolean;
  readonly description: string;
  readonly parameterSchema: Readonly<Record<string, unknown>>;
}

export interface GovernanceSkillCatalog {
  readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
  readonly items: readonly GovernanceSkillDescriptor[];
  readonly catalogHash: string;
}

export interface GovernanceKnowledgeItem {
  readonly sourceLabel: string;
  readonly sourceUri: string;
  readonly contentHash: string;
  readonly chunkHash: string;
  readonly text: string;
  readonly rank: number;
}

export interface GovernanceKnowledgeSearchResponse {
  readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
  readonly items: readonly GovernanceKnowledgeItem[];
  readonly indexHash: string;
  readonly auditHash: string;
}

export interface GovernanceConfirmationTicket {
  readonly token: string;
  readonly action: GovernanceConfirmationAction;
  readonly requestDigest: string;
  readonly expiresAt: string;
}

export interface GovernanceSkillExecutionResponse {
  readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
  readonly executionId: string;
  readonly skill: GovernanceSkillName;
  readonly status: "completed" | "confirmation_required";
  readonly result: Readonly<Record<string, unknown>>;
  readonly confirmation: GovernanceConfirmationTicket | null;
  readonly provenance: Readonly<Record<string, unknown>>;
  readonly auditHash: string;
}

export type GovernanceSkillConfirmationResponse =
  | {
      readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
      readonly action: "run_governance_analysis";
      readonly status: "completed";
      readonly result: GovernanceOnlineRun;
      readonly auditHash: string;
    }
  | {
      readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
      readonly action: "save_draft_report";
      readonly status: "completed";
      readonly result: Readonly<Record<string, unknown>>;
      readonly auditHash: string;
    }
  | {
      readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
      readonly action: "submit_review";
      readonly status: "completed";
      readonly result: Readonly<Record<string, unknown>>;
      readonly auditHash: string;
    };

export interface AssistantSkillDescriptor {
  readonly name: AssistantSkillName;
  readonly label: string;
  readonly description: string;
  readonly uiLocation: string;
  readonly readOnly: true;
  readonly confirmationRequired: false;
  readonly governanceSkills: readonly GovernanceReadOnlySkillName[];
  readonly parameterSchema: Readonly<Record<string, unknown>>;
}

export interface AssistantSkillCatalog {
  readonly schemaVersion: typeof ASSISTANT_SKILLS_SCHEMA;
  readonly items: readonly AssistantSkillDescriptor[];
  readonly catalogHash: string;
}

export interface AssistantSkillResult {
  readonly schemaVersion: typeof ASSISTANT_SKILL_RESULT_SCHEMA;
  readonly executionId: string;
  readonly skill: AssistantSkillName;
  readonly answer: string;
  readonly result: Readonly<Record<string, unknown>>;
  readonly skillCalls: readonly GovernanceAssistantSkillTrace[];
  readonly evidenceRefs: readonly GovernanceAssistantEvidenceRef[];
  readonly citedHashes: readonly string[];
  readonly auditHash: string;
}

export interface GovernanceAssistantEvidenceRef {
  readonly label: string;
  readonly sourceKind: GovernanceAssistantEvidenceSource;
  readonly hash: string;
}

export interface GovernanceAssistantSkillTrace {
  readonly skill: GovernanceReadOnlySkillName;
  readonly requestHash: string;
  readonly resultHash: string;
}

export interface GovernanceCaseKindEntry {
  readonly kind: GovernanceTargetKind;
  readonly targetIds: readonly string[];
}

export interface GovernanceSimilarCasesQuery {
  readonly caseId?: string;
  readonly runId?: string;
  readonly kindEntries?: readonly GovernanceCaseKindEntry[];
  readonly limit?: number;
}

export interface GovernanceSimilarCase {
  readonly caseId: string;
  readonly score: number;
  readonly components: {
    readonly embedding: number;
    readonly structure: number;
    readonly modality: number;
  };
  readonly graphVersionHash: string;
  readonly modelStateHash: string;
  readonly kindKey: string;
  readonly kindEntries: readonly GovernanceCaseKindEntry[];
  readonly concludedAt: string;
  readonly recordHash: string;
}

export interface GovernanceSimilarCasesResponse {
  readonly schemaVersion: typeof GOVERNANCE_SKILLS_SCHEMA;
  readonly query: Readonly<Record<string, unknown>>;
  readonly items: readonly GovernanceSimilarCase[];
  readonly indexHash: string;
  readonly backfill: Readonly<Record<string, number>>;
  readonly auditHash: string;
}

export interface GovernanceSkillsClientLike {
  catalog(signal?: AbortSignal): Promise<GovernanceSkillCatalog>;
  executeSkill<Name extends GovernanceSkillName>(
    context: GovernanceSkillsContext,
    skill: Name,
    params: GovernanceSkillParams<Name>,
    signal?: AbortSignal,
  ): Promise<GovernanceSkillExecutionResponse>;
  confirmSkill(token: string, signal?: AbortSignal): Promise<GovernanceSkillConfirmationResponse>;
  assistantCatalog(signal?: AbortSignal): Promise<AssistantSkillCatalog>;
  executeAssistant(
    context: GovernanceSkillsContext,
    skill: AssistantSkillName,
    message: string,
    signal?: AbortSignal,
  ): Promise<AssistantSkillResult>;
  searchKnowledge(
    context: GovernanceSkillsContext,
    query: string,
    limit?: number,
    signal?: AbortSignal,
  ): Promise<GovernanceKnowledgeSearchResponse>;
  searchSimilarCases(
    context: GovernanceSkillsContext,
    query: GovernanceSimilarCasesQuery,
    signal?: AbortSignal,
  ): Promise<GovernanceSimilarCasesResponse>;
}
