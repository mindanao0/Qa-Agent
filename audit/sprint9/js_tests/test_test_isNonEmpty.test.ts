function isNonEmpty(value: unknown): boolean {
  if (value === null || value === undefined) return false;
  if (typeof value === 'string') return value.trim().length > 0;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === 'object') return Object.keys(value as object).length > 0;
  return true;
}

import { describe, it, expect } from "vitest";
describe("isNonEmpty", () => {
  it("returns false for null", () => { expect(isNonEmpty(null)).toBe(false); });
  it("returns false for undefined", () => { expect(isNonEmpty(undefined)).toBe(false); });
  it("returns true for non-empty string", () => { expect(isNonEmpty("hello")).toBe(true); });
});