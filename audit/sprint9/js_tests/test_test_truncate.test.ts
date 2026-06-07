function truncate(str: string, maxLen: number, suffix: string = '...'): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen) + suffix;
}
import { describe, it, expect } from "vitest";
describe("truncate", () => {
  it("truncates string to maxLen", () => { expect(truncate('hello', 3)).toBe('hel...'); });
  it("does not truncate if length is less than or equal to maxLen", () => { expect(truncate('hi', 2)).toBe('hi'); });
  it("appends suffix if truncated", () => { expect(truncate('hello world', 5, '->')).toBe('hell->'); });
});