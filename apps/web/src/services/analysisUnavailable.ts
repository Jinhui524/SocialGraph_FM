import type { AnalysisResult, AnalysisRun } from "../types/graph";
import type { CoreWorkbenchServiceState } from "../types/core";

type UnavailableAnalysisResult = Extract<AnalysisResult, { readonly kind: "unavailable" }>;

export function analysisEngineLabel(run: AnalysisRun | undefined): string {
  if (!run) return "等待图数据";
  if (run.engine === "local_algorithm") return "本地图算法";
  if (run.engine === "gfm") return "GFM 模型";
  if (run.result?.kind === "unavailable" && run.result.code === "GFM_CORE_NOT_CONNECTED") {
    return "GFM 模型未就绪";
  }
  return "分析不可用";
}

export function describeUnavailableAnalysis(
  result: UnavailableAnalysisResult | undefined,
  service: CoreWorkbenchServiceState,
): string {
  if (!result) return "分析服务当前不可用。";
  if (result.code !== "GFM_CORE_NOT_CONNECTED") return result.message;
  if (service.state === "unavailable") return "GFM 服务当前不可用。";
  if (service.state === "checking") return "GFM 服务状态仍在检查。";
  if (!service.capabilities.servingReady) {
    return "GFM 服务已连接，但 registry 中尚无正式晋级的 servingReady 模型。";
  }
  return "当前没有与该任务和图版本兼容的 GFM 模型。";
}
