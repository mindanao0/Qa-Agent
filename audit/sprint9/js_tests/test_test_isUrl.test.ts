function isUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return url.protocol === 'http:' || url.protocol === 'https:';
  } catch {
    return false;
  }
}

import { describe, it, expect } from "vitest";
describe("isUrl", () => {
  it("returns true for valid http URL", () => { expect(isUrl('http://example.com')).toBe(true); });
  it("returns false for invalid URL", () => { expect(isUrl('ftp://example.com')).toBe(false); });
});