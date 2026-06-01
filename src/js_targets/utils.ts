/** Clamp a number between min and max (inclusive). */
export function clamp(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/** Remove duplicate elements, preserving first-occurrence order. */
export function dedupe<T>(arr: T[]): T[] {
  const seen = new Set<T>();
  const result: T[] = [];
  for (const item of arr) {
    if (!seen.has(item)) { seen.add(item); result.push(item); }
  }
  return result;
}

/** Return a shallow copy of obj containing only the specified keys. */
export function pick<T extends object, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) { result[key] = obj[key]; }
  return result;
}

/** Split arr into sub-arrays of at most `size` elements. Throws if size <= 0. */
export function chunk<T>(arr: T[], size: number): T[][] {
  if (size <= 0) throw new Error('chunk: size must be positive');
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    result.push(arr.slice(i, i + size));
  }
  return result;
}

/** Sum all numbers in an array; returns 0 for an empty array. */
export function sum(arr: number[]): number {
  let total = 0;
  for (const n of arr) { total += n; }
  return total;
}
