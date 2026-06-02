import { emptyProgress, PROGRESS_VERSION, type Progress } from '../game/progress';

const STORAGE_KEY = 'neuroforge.progress';

/**
 * Load progress from localStorage, returning a fresh empty progress on anything
 * unexpected (missing key, unavailable storage, malformed JSON, or a version we no
 * longer understand). Persistence must never throw — a corrupt save just resets.
 */
export function loadProgress(): Progress {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return emptyProgress();
    const parsed: unknown = JSON.parse(raw);
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      (parsed as Progress).version !== PROGRESS_VERSION ||
      typeof (parsed as Progress).completed !== 'object' ||
      (parsed as Progress).completed === null
    ) {
      return emptyProgress();
    }
    return parsed as Progress;
  } catch {
    return emptyProgress();
  }
}

/** Persist progress. Silently no-ops if storage is unavailable (e.g. private mode). */
export function saveProgress(progress: Progress): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(progress));
  } catch {
    /* storage unavailable or quota exceeded — ignore */
  }
}

/** Remove the stored progress (used by a "reset progress" action). */
export function clearProgress(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}
