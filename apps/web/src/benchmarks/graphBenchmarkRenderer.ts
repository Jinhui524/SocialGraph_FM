export const GRAPH_BENCHMARK_RENDERERS = [
  "canvas",
  "hybrid-webgl",
  "auto",
] as const;

export type GraphBenchmarkRenderer = (typeof GRAPH_BENCHMARK_RENDERERS)[number];
export type GraphBenchmarkResolvedRenderer = Exclude<GraphBenchmarkRenderer, "auto">;

export function isGraphBenchmarkRenderer(
  value: string | null | undefined,
): value is GraphBenchmarkRenderer {
  return GRAPH_BENCHMARK_RENDERERS.includes(value as GraphBenchmarkRenderer);
}

/**
 * The benchmark remains Canvas-compatible by default. Renderer comparison is
 * opt-in through `?renderer=...` or the benchmark runner's CLI/env options.
 */
export function requestedGraphBenchmarkRenderer(
  search: string,
): GraphBenchmarkRenderer {
  const value = new URLSearchParams(search).get("renderer");
  return isGraphBenchmarkRenderer(value) ? value : "canvas";
}

