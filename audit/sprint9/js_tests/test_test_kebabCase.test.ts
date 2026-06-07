function kebabCase(str: string): string {
  return str
    .replace(/([A-Z])/g, '-$1')
    .replace(/[ _]+/g, '-')
    .toLowerCase()
    .replace(/^-|-$/g, '');
}
import { describe, it, expect } from "vitest";
describe("kebabCase", () => {
  it("converts camelCase to kebab-case", () => { expect(kebabCase('helloWorld')).toBe('helloworld'); });
  it("handles spaces and underscores", () => { expect(kebabCase('hello world')).toBe('helloworld'); });
});