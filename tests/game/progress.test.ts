import { describe, it, expect } from 'vitest';
import {
  emptyProgress,
  markComplete,
  isCompleted,
  isUnlocked,
  PROGRESS_VERSION,
} from '../../src/game/progress';
import type { Challenge } from '../../src/game/challenges';

const fakeChallenges = [
  { id: 'a' },
  { id: 'b' },
  { id: 'c' },
] as unknown as Challenge[];

describe('progress', () => {
  it('emptyProgress carries the version and an empty map', () => {
    const p = emptyProgress();
    expect(p.version).toBe(PROGRESS_VERSION);
    expect(p.completed).toEqual({});
  });

  it('markComplete records a score and keeps the best one', () => {
    let p = emptyProgress();
    p = markComplete(p, 'a', 0.96);
    expect(isCompleted(p, 'a')).toBe(true);
    expect(p.completed['a']!.bestScore).toBeCloseTo(0.96, 12);

    p = markComplete(p, 'a', 0.99);
    expect(p.completed['a']!.bestScore).toBeCloseTo(0.99, 12);

    p = markComplete(p, 'a', 0.5); // lower score should not lower the best
    expect(p.completed['a']!.bestScore).toBeCloseTo(0.99, 12);
  });

  it('markComplete does not mutate the input', () => {
    const p = emptyProgress();
    const q = markComplete(p, 'a', 1);
    expect(p.completed).toEqual({});
    expect(q).not.toBe(p);
  });

  it('unlocks sequentially', () => {
    let p = emptyProgress();
    expect(isUnlocked(p, fakeChallenges, 0)).toBe(true);
    expect(isUnlocked(p, fakeChallenges, 1)).toBe(false);

    p = markComplete(p, 'a', 1);
    expect(isUnlocked(p, fakeChallenges, 1)).toBe(true);
    expect(isUnlocked(p, fakeChallenges, 2)).toBe(false);

    p = markComplete(p, 'b', 1);
    expect(isUnlocked(p, fakeChallenges, 2)).toBe(true);
  });
});
