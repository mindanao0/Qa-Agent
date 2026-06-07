function capitalize(str: string): string {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}
import { describe, it, expect } from "vitest";
describe("capitalize", () => {
  it("capitalizes the first character of a string", () => { expect(capitalize('hello')).toBe('Hello'); });
  it("returns an empty string if input is empty", () => { expect(capitalize('')).toBe(''); });
  it("does not modify already capitalized strings", () => { expect(capitalize('Hello')).toBe('Hello'); });
});