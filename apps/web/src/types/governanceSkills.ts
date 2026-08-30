import type { GovernanceOnlineRun } from "./governanceOnline";
import {
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

export const GOVERNANCE_ASSISTANT_SCHEMA = "socialgraph-fm.governance-assistant/1.0" as const;
export const GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA = "socialgraph-fm.governance-assistant-dispatch/1.0" as const;

export {
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
export type GovernanceAssistantDispatchIntent = "answer" | "start_analysis" | "open_review" | "submit_review" | "draft_report";
export type GovernanceAnswerMode = "overview" | "analysis_summary" | "coordination_summary" | "evidence_requirements" | "review_guidance" | "method_scope" | "knowledge" | "case_draft";
export type GovernanceTargetKind = "node" | "relation" | "group";
export type GovernanceAssistantGenerationMode = "llm_assisted" | "deterministic_report";
export type GovernanceAssistantFallbackPhase = "intent" | "planning" | "skill_execution" | "narration";
export type GovernanceAssistantEvidenceSource = "graph" | "skill" | "knowledge" | "case";

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

export interface GovernanceAssistantDispatchContext {
  readonly intent?: "answer";
  readonly answerMode?: GovernanceAnswerMode;
  readonly narrationMode?: "auto" | "deterministic_only";
  readonly topK?: number;
  readonly reviewDecision?: "confirmed" | "rejected" | "pending";
  readonly reviewReason?: string;
}

export interface GovernanceAssistantDispatchResponse {
  readonly schemaVersion: typeof GOVERNANCE_ASSISTANT_DISPATCH_SCHEMA;
  readonly dispatchId: string;
  readonly intent: GovernanceAssistantDispatchIntent;
  readonly answerMode: GovernanceAnswerMode | null;
  readonly status: "completed" | "confirmation_required" | "blocked";
  readonly answer: string;
  readonly result: Readonly<Record<string, unknown>>;
  readonly deterministicFallback: boolean;
  readonly generationMode?: GovernanceAssistantGenerationMode;
  readonly fallbackPhase?: GovernanceAssistantFallbackPhase | null;
  readonly reasonCode?: string | null;
  readonly evidenceRefs?: readonly GovernanceAssistantEvidenceRef[];
  readonly confirmation: GovernanceConfirmationTicket | null;
  readonly navigation: {
    readonly view: "governance_review";
    readonly runId: string;
    readonly caseId?: string;
    readonly target?: { readonly targetType: GovernanceTargetKind; readonly targetId: string };
  } | null;
  readonly skillCalls: readonly GovernanceAssistantSkillTrace[];
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

export interface GovernanceAssistantTurnResponse {
  readonly schemaVersion: typeof GOVERNANCE_ASSISTANT_SCHEMA;
  readonly turnId: string;
  readonly answer: string;
  readonly deterministicFallback: boolean;
  readonly skillCalls: readonly GovernanceAssistantSkillTrace[];
  readonly citedHashes: readonly string[];
  readonly auditHash: string;
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
  assistantTurn(
    context: GovernanceSkillsContext,
    message: string,
    signal?: AbortSignal,
  ): Promise<GovernanceAssistantTurnResponse>;
  dispatchAssistant(
    context: GovernanceSkillsContext,
    message: string,
    options?: GovernanceAssistantDispatchContext,
    signal?: AbortSignal,
  ): Promise<GovernanceAssistantDispatchResponse>;
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
