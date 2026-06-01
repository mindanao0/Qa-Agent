/** Uppercase the first character of a string. */
export function capitalize(str: string): string {
  if (!str) return str;
  return str.charAt(0).toUpperCase() + str.slice(1);
}

/** Truncate str to maxLen chars, appending suffix if truncated (result may exceed maxLen). */
export function truncate(str: string, maxLen: number, suffix: string = '...'): string {
  if (str.length <= maxLen) return str;
  return str.slice(0, maxLen) + suffix;
}

/** Format a Date or ISO string as YYYY-MM-DD. Returns 'Invalid Date' on bad input. */
export function formatDate(date: Date | string): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  if (isNaN(d.getTime())) return 'Invalid Date';
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

/** Convert a camelCase or snake_case string to kebab-case. Trims leading and trailing dashes. */
export function kebabCase(str: string): string {
  return str
    .replace(/([A-Z])/g, '-$1')
    .replace(/[ _]+/g, '-')
    .toLowerCase()
    .replace(/^-|-$/g, '');
}
