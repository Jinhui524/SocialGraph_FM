import type {
  AnalysisRun,
  CreateAnalysisInput,
  GraphVersion,
  IntentNormalizationResult,
  NormalizeIntentInput,
  SocialGraphApi,
} from "../types/graph";
import { MockIntentNormalizer } from "./intentNormalizer";
import { LocalAnalysisExecutor } from "./localAnalysisExecutor";

export { buildDemoGraphVersion } from "./graphImport";
export { normalizeIntentLocally } from "./intentNormalizer";

/**
 * Compatibility facade for older imports. New code should inject an
 * IntentNormalizer and AnalysisExecutor separately.
 */
export class MockSocialGraphApi implements SocialGraphApi {
  private readonly intentNormalizer = new MockIntentNormalizer();
  private readonly analysisExecutor: LocalAnalysisExecutor;

  constructor(initialGraphVersions: readonly GraphVersion[] = []) {
    this.analysisExecutor = new LocalAnalysisExecutor(initialGraphVersions);
  }

  registerGraphVersion(version: GraphVersion): void {
    this.analysisExecutor.registerGraphVersion(version);
  }

  normalizeIntent(input: NormalizeIntentInput): Promise<IntentNormalizationResult> {
    return this.intentNormalizer.normalizeIntent(input);
  }

  createAnalysis(input: CreateAnalysisInput): Promise<AnalysisRun> {
    return this.analysisExecutor.createAnalysis(input);
  }

  getAnalysis(runId: string): Promise<AnalysisRun> {
    return this.analysisExecutor.getAnalysis(runId);
  }
}
