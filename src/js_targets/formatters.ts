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

/** Format a Date or ISO string for the given locale (default en-US). Returns 'Invalid Date' on bad input. */
export function formatDate(date: Date | string, locale: string = 'en-US'): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  if (isNaN(d.getTime())) return 'Invalid Date';
  return d.toLocaleDateString(locale);
}

/** Convert a camelCase or snake_case string to kebab-case. Trims leading and trailing dashes. */
export function kebabCase(str: string): string {
  return str
    .replace(/([A-Z])/g, '-$1')
    .replace(/[\s_]+/g, '-')
    .toLowerCase()
    .replace(/^-|-$/g, '');
}
