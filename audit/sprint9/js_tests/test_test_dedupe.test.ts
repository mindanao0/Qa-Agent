function dedupe<T>(arr: T[]): T[] {
  const seen = new Set<T>();
  const result: T[] = [];
  for (const item of arr) {
    if (!seen.has(item)) { seen.add(item); result.push(item); }
  }
  return result;
}
import { describe, it, expect } from "vitest";
describe("dedupe", () => {
  it("removes duplicates", () => { expect(dedupe([1, 2, 3, 2, 4])).toEqual([1, 2, 3, 4]); });
  it("handles empty array", () => { expect(dedupe([])).toEqual([]); });
  it("preserves order", () => { expect(dedupe([5, 4, 3, 2, 1])).toEqual([5, 4, 3, 2, 1]); });
});