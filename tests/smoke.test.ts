import { describe, it, expect } from 'vitest';

describe('toolchain smoke test', () => {
  it('runs TypeScript tests', () => {
    const sum = (xs: number[]): number => xs.reduce((a, b) => a + b, 0);
    expect(sum([1, 2, 3, 4])).toBe(10);
  });
});
