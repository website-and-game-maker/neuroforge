import { describe, it, expect } from 'vitest';
import { modeFromPath, pathForMode } from '../../src/ui/router';

const BASE = '/neuroforge/';

describe('pathForMode', () => {
  it('maps every mode to a named path under the base', () => {
    expect(pathForMode('learn', BASE)).toBe('/neuroforge/learn');
    expect(pathForMode('challenge', BASE)).toBe('/neuroforge/challenges');
    expect(pathForMode('sandbox', BASE)).toBe('/neuroforge/sandbox');
    expect(pathForMode('versus', BASE)).toBe('/neuroforge/versus');
  });

  it('works at a root base too', () => {
    expect(pathForMode('learn', '/')).toBe('/learn');
  });
});

describe('modeFromPath', () => {
  it('resolves named paths to modes', () => {
    expect(modeFromPath('/neuroforge/learn', BASE)).toBe('learn');
    expect(modeFromPath('/neuroforge/challenges', BASE)).toBe('challenge');
    expect(modeFromPath('/neuroforge/sandbox', BASE)).toBe('sandbox');
    expect(modeFromPath('/neuroforge/versus', BASE)).toBe('versus');
  });

  it('treats the bare base, index.html, and trailing slashes as home (null)', () => {
    expect(modeFromPath('/neuroforge/', BASE)).toBeNull();
    expect(modeFromPath('/neuroforge/index.html', BASE)).toBeNull();
    expect(modeFromPath('/neuroforge/learn/', BASE)).toBe('learn');
  });

  it('returns null for unknown paths, resolves base-less slugs (dev servers)', () => {
    expect(modeFromPath('/neuroforge/nonsense', BASE)).toBeNull();
    expect(modeFromPath('/other/learn', BASE)).toBeNull();
    expect(modeFromPath('/learn', BASE)).toBe('learn'); // base-less fallback
  });
});
