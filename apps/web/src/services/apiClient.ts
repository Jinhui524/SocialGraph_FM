const configuredBaseUrl = (import.meta.env.VITE_SOCIALGRAPH_API_BASE_URL ?? "").trim();

/** The single browser-side origin for every SocialGraph API request. */
export const SOCIALGRAPH_API_BASE_URL = configuredBaseUrl.replace(/\/+$/u, "");

export function socialGraphApiUrl(path: string, baseUrl = SOCIALGRAPH_API_BASE_URL): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${baseUrl.replace(/\/+$/u, "")}${normalizedPath}`;
}

export class SocialGraphApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "SocialGraphApiError";
  }
}

export async function readSocialGraphApiJson<T>(response: Response): Promise<T> {
  let payload: unknown;
  try {
    payload = await response.json();
  } catch {
    payload = undefined;
  }
  if (!response.ok) {
    const body = payload && typeof payload === "object"
      ? payload as { code?: unknown; message?: unknown; detail?: unknown }
      : undefined;
    const detail = body?.detail && typeof body.detail === "object"
      ? body.detail as { code?: unknown; message?: unknown }
      : undefined;
    const code = typeof body?.code === "string"
      ? body.code
      : typeof detail?.code === "string"
        ? detail.code
        : `HTTP_${response.status}`;
    const message = typeof body?.message === "string"
      ? body.message
      : typeof detail?.message === "string"
        ? detail.message
        : typeof body?.detail === "string"
          ? body.detail
          : `SocialGraph API request failed (HTTP ${response.status}).`;
    throw new SocialGraphApiError(code, message, response.status);
  }
  return payload as T;
}
