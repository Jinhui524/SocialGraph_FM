import type {
  FileProfile,
  GraphBuildSpec,
  GraphVersion,
  GraphVersionProvenance,
  ImportRun,
  NormalizedIntent,
  SourceArtifact,
  TargetResolution,
  ValidationIssue,
  ViewCommand,
} from "../../types/graph";

export interface PendingTargetResolution {
  readonly command: ViewCommand;
  readonly resolutions: readonly Extract<TargetResolution, { status: "ambiguous" }>[];
  readonly intent?: NormalizedIntent;
}

export interface PendingImportDraft {
  readonly files: readonly File[];
  readonly profiles: readonly FileProfile[];
  readonly artifacts: readonly SourceArtifact[];
  readonly requestToken: string;
  readonly baseGraphVersionId?: string;
  /** User message that submitted the attachment or construction revision. */
  readonly sourceMessageId?: string;
}

export type ImportViewState =
  | { kind: "idle" }
  | { kind: "parsing"; fileName: string; stage: "inspect" | "parse" | "version" }
  | {
      kind: "roles";
      files: readonly File[];
      profiles: readonly FileProfile[];
      initialEdgeIndex: number;
      baseGraphVersionId?: string;
    }
  | {
      kind: "mapping";
      pending: PendingImportDraft;
      issues: readonly ValidationIssue[];
      spec: GraphBuildSpec;
      source: "llm" | "deterministic_fallback";
      normalizationWarnings: readonly string[];
      parentVersionId?: string;
      reconstructionReason?: GraphVersionProvenance["reconstructionReason"];
    }
  | {
      kind: "review";
      pending: PendingImportDraft;
      spec: GraphBuildSpec;
      run: ImportRun & { readonly graphVersion: GraphVersion };
      source: "llm" | "deterministic_fallback";
      warnings: readonly string[];
      parentVersionId?: string;
      reconstructionReason?: GraphVersionProvenance["reconstructionReason"];
    }
  | { kind: "success"; fileName: string; version: GraphVersion }
  | { kind: "error"; fileName: string; message: string; issues: readonly ValidationIssue[] };

export function browserImportProvenance(
  reconstructionReason?: GraphVersionProvenance["reconstructionReason"],
): GraphVersionProvenance {
  return Object.freeze({
    origin: "browser_import",
    pipeline: "browser-import",
    pipelineVersion: "2.0.0",
    buildSpecSchemaVersion: "1.0",
    sourceHashScheme: "artifact-sha256-list-v1",
    ...(reconstructionReason ? { reconstructionReason } : {}),
  });
}

export function pendingProfileForRole(
  pending: PendingImportDraft,
  role: SourceArtifact["role"],
): FileProfile | undefined {
  const index = pending.artifacts.findIndex((artifact) => artifact.role === role);
  return index >= 0 ? pending.profiles[index] : undefined;
}
