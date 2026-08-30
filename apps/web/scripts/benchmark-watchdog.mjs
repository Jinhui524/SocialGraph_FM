import { spawn } from "node:child_process";

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

export async function terminateProcessTree(child, graceMs = 1_000) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  child.kill("SIGTERM");
  await delay(graceMs);
  if (child.exitCode !== null || child.signalCode !== null) return;
  if (process.platform === "win32") {
    await new Promise((resolve) => {
      const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], {
        windowsHide: true,
        stdio: "ignore",
      });
      killer.once("exit", resolve);
      killer.once("error", resolve);
    });
  } else {
    child.kill("SIGKILL");
  }
}

export async function waitForChildWithDeadline(child, timeoutMs) {
  let timer;
  const outcome = await Promise.race([
    new Promise((resolve) => {
      child.once("exit", (code, signal) => resolve({ code, signal, timedOut: false }));
      child.once("error", (error) => resolve({ code: null, signal: null, error, timedOut: false }));
    }),
    new Promise((resolve) => {
      timer = setTimeout(() => resolve({
        code: null,
        signal: "PROCESS_TIMEOUT",
        timedOut: true,
      }), timeoutMs);
    }),
  ]);
  if (timer) clearTimeout(timer);
  if (outcome.timedOut) await terminateProcessTree(child);
  return outcome;
}
