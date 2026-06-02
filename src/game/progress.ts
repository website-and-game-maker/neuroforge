import type { Challenge } from './challenges';

/** Persisted player progress: best score recorded per completed challenge. */
export interface Progress {
  version: number;
  completed: Record<string, { bestScore: number }>;
}

export const PROGRESS_VERSION = 1;

export function emptyProgress(): Progress {
  return { version: PROGRESS_VERSION, completed: {} };
}

export function isCompleted(p: Progress, id: string): boolean {
  return Object.prototype.hasOwnProperty.call(p.completed, id);
}

/** Record a completion, keeping the best (highest) score seen. Returns a new Progress. */
export function markComplete(p: Progress, id: string, score: number): Progress {
  const prev = p.completed[id]?.bestScore ?? -Infinity;
  const bestScore = Math.max(prev, score);
  return {
    ...p,
    completed: { ...p.completed, [id]: { bestScore } },
  };
}

/**
 * Challenges unlock sequentially: the first is always available, and each subsequent
 * one unlocks once the previous challenge is completed.
 */
export function isUnlocked(p: Progress, challenges: Challenge[], index: number): boolean {
  if (index <= 0) return true;
  const prev = challenges[index - 1];
  return prev ? isCompleted(p, prev.id) : true;
}
