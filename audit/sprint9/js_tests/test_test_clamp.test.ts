function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}
import { describe, it, expect } from "vitest";
describe("clamp", () => {
  it("clamps a number within the range", () => { expect(clamp(5, 1, 10)).toBe(5); });
  it("clamps a number below the minimum", () => { expect(clamp(-1, 1, 10)).toBe(1); });
  it("clamps a number above the maximum", () => { expect(clamp(12, 1, 10)).toBe(10); });
});