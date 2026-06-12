/**
 * Tiny path-based router: each mode lives at a named path under the app base
 * (e.g. /neuroforge/learn, /neuroforge/versus) instead of an html file. Pure
 * string functions so they're trivially unit-testable; the App wires history.
 */

export type Mode = 'learn' | 'challenge' | 'sandbox' | 'versus';

const MODE_TO_SLUG: Record<Mode, string> = {
  learn: 'learn',
  challenge: 'challenges',
  sandbox: 'sandbox',
  versus: 'versus',
};

const SLUG_TO_MODE: Record<string, Mode> = Object.fromEntries(
  (Object.entries(MODE_TO_SLUG) as Array<[Mode, string]>).map(([m, s]) => [s, m]),
) as Record<string, Mode>;

/** The canonical URL path for a mode. `base` always ends with '/'. */
export function pathForMode(mode: Mode, base: string): string {
  return base + MODE_TO_SLUG[mode];
}

/**
 * Resolve a location pathname to a mode. Strips the base, a stray `index.html`,
 * and trailing slashes. Returns null for the bare base (home) or unknown paths
 * — the caller decides the default.
 */
export function modeFromPath(pathname: string, base: string): Mode | null {
  let rest = pathname.startsWith(base) ? pathname.slice(base.length) : pathname.replace(/^\//, '');
  rest = rest.replace(/index\.html$/, '').replace(/\/+$/, '').replace(/^\/+/, '');
  if (rest === '') return null;
  return SLUG_TO_MODE[rest] ?? null;
}

/** True when the current pathname is not already the canonical path for `mode`. */
export function needsUrlUpdate(pathname: string, mode: Mode, base: string): boolean {
  return pathname !== pathForMode(mode, base);
}
