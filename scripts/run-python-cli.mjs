import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const cli = fileURLToPath(new URL("./socialgraph.py", import.meta.url));
const configured = process.env.SOCIALGRAPH_PYTHON?.trim();
const candidates = configured
  ? [[configured, []]]
  : process.platform === "win32"
    ? [["py", ["-3.12"]], ["python", []], ["python3", []]]
    : [["python3", []], ["python", []]];

let selected;
for (const [command, prefix] of candidates) {
  const probe = spawnSync(
    command,
    [...prefix, "-c", "import sys; raise SystemExit(0 if (3, 12) <= sys.version_info < (3, 13) else 1)"],
    { stdio: "ignore", shell: false, windowsHide: true },
  );
  if (probe.status === 0) {
    selected = [command, prefix];
    break;
  }
}

if (!selected) {
  console.error(
    "Python 3.12+ was not found. Install Python 3.12 or set SOCIALGRAPH_PYTHON to its executable.",
  );
  process.exit(127);
}

const [command, prefix] = selected;
const completed = spawnSync(command, [...prefix, cli, ...process.argv.slice(2)], {
  stdio: "inherit",
  shell: false,
  windowsHide: false,
});
if (completed.error) {
  console.error(`Could not start SocialGraph-FM: ${completed.error.message}`);
  process.exit(1);
}
process.exit(completed.status ?? 1);
