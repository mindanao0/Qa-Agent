function isEmail(value: string): boolean {
  return /^[^\ @]+@[^\ @]+\.[^\u0020@]+$/.test(value);
}
import { describe, it, expect } from "vitest";
describe("isEmail", () => {
  it("valid email", () => { expect(isEmail("example@example.com")).toBe(true); });
  it("invalid email with space", () => { expect(isEmail("example example@example.com")).toBe(false); });
});