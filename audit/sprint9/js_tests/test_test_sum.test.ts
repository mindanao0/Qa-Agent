function sum(arr: number[]): number {
  let total = 0;
  for (const n of arr) { total += n; }
  return total;
}
import { describe, it, expect } from "vitest";
describe("sum", () => {
  it("returns 0 for an empty array", () => { expect(sum([])).toBe(0); });
  it("returns the sum of positive numbers", () => { expect(sum([1, 2, 3])).toBe(6); });
  it("returns the sum of negative numbers", () => { expect(sum([-1, -2, -3])).toBe(-6); });
});