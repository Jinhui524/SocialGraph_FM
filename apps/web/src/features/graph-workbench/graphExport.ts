export function safeFileStem(value: string | undefined) {
  const stem = (value || "socialgraph")
    .replace(/\.[^.]+$/, "")
    .replace(/[\\/:*?"<>|\s]+/g, "-")
    .replace(/^-+|-+$/g, "");
  return stem || "socialgraph";
}

export function downloadDataUrl(dataUrl: string, fileName: string) {
  const anchor = document.createElement("a");
  anchor.href = dataUrl;
  anchor.download = fileName;
  anchor.rel = "noopener";
  anchor.click();
}

export function downloadJson(value: unknown, fileName: string) {
  const blob = new Blob([JSON.stringify(value, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  downloadDataUrl(url, fileName);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
