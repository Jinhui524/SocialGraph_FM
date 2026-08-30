import { describe, expect, it } from "vitest";
import canonicalVectors from "../../../../packages/gfm/tests/golden/canonical-vectors.json";

import {
  canonicalJson,
  compareUnicodeCodePoints,
  sha256Canonical,
  sha256Text,
} from "./graphIdentity";

describe("graph identity", () => {
  it("implements SHA-256 and stable object-key canonicalisation", () => {
    expect(sha256Text("abc")).toBe("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    expect(canonicalJson({ b: 2, a: { d: 4, c: 3 } })).toBe('{"a":{"c":3,"d":4},"b":2}');
    expect(sha256Canonical({ a: 1, b: 2 })).toBe(sha256Canonical({ b: 2, a: 1 }));
  });

  it("uses Unicode code-point ordering and a cross-runtime golden digest", () => {
    expect(["😀", "\uE000", "中"].sort(compareUnicodeCodePoints)).toEqual(["中", "\uE000", "😀"]);
    for (const vector of canonicalVectors.vectors) {
      expect(canonicalJson(vector.value), vector.name).toBe(vector.canonical);
      expect(sha256Canonical(vector.value), vector.name).toBe(vector.sha256);
    }
    expect(canonicalJson({ "2": "two", "10": "ten", "01": "one" }))
      .toBe('{"01":"one","10":"ten","2":"two"}');
  });

  it.each([Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY])(
    "rejects non-finite values instead of silently hashing %s as null",
    (value) => {
      expect(() => canonicalJson({ value })).toThrow("CANONICAL_JSON_NON_FINITE_NUMBER");
    },
  );
});
