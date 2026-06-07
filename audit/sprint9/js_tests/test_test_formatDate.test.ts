function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  if (isNaN(d.getTime())) return 'Invalid Date';
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

import { describe, it, expect } from "vitest";
describe("formatDate", () => {
  it("formats a valid date", () => {
    expect(formatDate(new Date(2023, 10, 5))).toBe('2023-11-05');
  });
  it("handles invalid dates", () => {
    expect(formatDate('invalid date')).toBe('Invalid Date');
  });
});