// @vitest-environment jsdom
import { describe, it, expect, beforeEach } from 'vitest';
import { loadProgress, saveProgress, clearProgress } from '../../src/state/persistence';
import { emptyProgress, markComplete, PROGRESS_VERSION } from '../../src/game/progress';

const KEY = 'neuroforge.progress';

describe('persistence', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('round-trips a progress object', () => {
    const p = markComplete(emptyProgress(), 'xor', 0.98);
    saveProgress(p);
    expect(loadProgress()).toEqual(p);
  });

  it('returns empty progress when nothing is stored', () => {
    expect(loadProgress()).toEqual(emptyProgress());
  });

  it('returns empty progress on corrupt JSON (no throw)', () => {
    localStorage.setItem(KEY, '{not valid json');
    expect(loadProgress()).toEqual(emptyProgress());
  });

  it('returns empty progress on a mismatched version', () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({ version: PROGRESS_VERSION + 999, completed: { xor: { bestScore: 1 } } }),
    );
    expect(loadProgress()).toEqual(emptyProgress());
  });

  it('returns empty progress when completed is malformed', () => {
    localStorage.setItem(KEY, JSON.stringify({ version: PROGRESS_VERSION, completed: null }));
    expect(loadProgress()).toEqual(emptyProgress());
  });

  it('clearProgress removes the stored value', () => {
    saveProgress(markComplete(emptyProgress(), 'blobs', 1));
    clearProgress();
    expect(loadProgress()).toEqual(emptyProgress());
  });
});
