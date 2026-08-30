import { describe, expect, it } from "vitest";

import {
  readSocialGraphApiJson,
  SocialGraphApiError,
  socialGraphApiUrl,
} from "./apiClient";

describe("shared SocialGraph API client", () => {
  it("joins same-origin and explicit API bases consistently", () => {
    expect(socialGraphApiUrl("api/v1/gfm/capabilities", "")).toBe("/api/v1/gfm/capabilities");
    expect(socialGraphApiUrl("/api/v1/gfm/capabilities", "http://127.0.0.1:8000/"))
      .toBe("http://127.0.0.1:8000/api/v1/gfm/capabilities");
  });

  it("normalizes FastAPI detail errors into a typed fail-closed error", async () => {
    const response = new Response(JSON.stringify({
      detail: { code: "GFM_CORE_MODEL_NOT_INSTALLED", message: "No validated model is installed." },
    }), { status: 503, headers: { "Content-Type": "application/json" } });

    const error = await readSocialGraphApiJson(response).catch((candidate) => candidate);

    expect(error).toBeInstanceOf(SocialGraphApiError);
    expect(error).toMatchObject({ code: "GFM_CORE_MODEL_NOT_INSTALLED", status: 503 });
  });
});
