import { createRoot } from "react-dom/client";
import "./styles.css";
import "./workspace-enhancements.css";
import "./research-dataset.css";
import "./core-workbench.css";
import "./research.css";
import "./governance.css";
import "./governance-online.css";

const isGraphBenchmark = new URLSearchParams(window.location.search).get("benchmark") === "graph";
const root = createRoot(document.getElementById("root")!);

async function renderApplication() {
  if (isGraphBenchmark) {
    const { installWebglGpuProbe, requestedGpuProbeMode } = await import("./benchmarks/webglGpuProbe");
    if (requestedGpuProbeMode(window.location.search) === "gpu") installWebglGpuProbe();
    const { GraphBenchmarkPage } = await import("./benchmarks/GraphBenchmarkPage");
    root.render(<GraphBenchmarkPage />);
    return;
  }
  const { default: App } = await import("./App");
  root.render(<App />);
}

void renderApplication();
