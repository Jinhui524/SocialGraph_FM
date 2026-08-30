import type { ImportFormat, SourceArtifact } from "../types/graph";
import { createOpaqueId, sha256Hex } from "./graphIdentity";

export interface StorageCapacity {
  readonly persisted: boolean;
  readonly usage?: number;
  readonly quota?: number;
}

export function inferImportFormat(file: Pick<File, "name" | "type">): ImportFormat {
  const extension = file.name.split(".").pop()?.toLocaleLowerCase();
  if (extension === "csv" || file.type === "text/csv") return "csv";
  if (extension === "tsv" || file.type === "text/tab-separated-values") return "tsv";
  if (extension === "json" || file.type === "application/json") return "json";
  if (extension === "graphml" || file.type === "application/graphml+xml") return "graphml";
  if (extension === "gexf" || file.type === "application/gexf+xml") return "gexf";
  if (extension === "npz" || file.type === "application/x-npz") return "npz";
  return "unsupported";
}

export async function createSourceArtifact(
  file: File,
  role: SourceArtifact["role"] = "single",
): Promise<SourceArtifact> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  const sha256 = sha256Hex(bytes);
  return Object.freeze({
    id: createOpaqueId("source"),
    sha256,
    name: file.name,
    size: file.size,
    mimeType: file.type || "application/octet-stream",
    format: inferImportFormat(file),
    role,
    createdAt: new Date().toISOString(),
    blob: file.slice(0, file.size, file.type || "application/octet-stream"),
  });
}

export async function requestPersistentGraphStorage(): Promise<StorageCapacity> {
  const storage = typeof navigator !== "undefined" ? navigator.storage : undefined;
  if (!storage) return { persisted: false };
  let persisted = false;
  try {
    persisted = typeof storage.persisted === "function" && await storage.persisted();
    if (!persisted && typeof storage.persist === "function") persisted = await storage.persist();
  } catch {
    persisted = false;
  }
  try {
    const estimate = await storage.estimate();
    return {
      persisted,
      ...(typeof estimate.usage === "number" ? { usage: estimate.usage } : {}),
      ...(typeof estimate.quota === "number" ? { quota: estimate.quota } : {}),
    };
  } catch {
    return { persisted };
  }
}

export function downloadSourceArtifact(artifact: SourceArtifact): void {
  const url = URL.createObjectURL(artifact.blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = artifact.name;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
