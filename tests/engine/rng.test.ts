import { describe, it, expect } from 'vitest';
import { Rng } from '../../src/engine/rng';

describe('Rng', () => {
  it('is deterministic for a given seed', () => {
    const a = new Rng(42);
    const b = new Rng(42);
    const seqA = Array.from({ length: 8 }, () => a.next());
    const seqB = Array.from({ length: 8 }, () => b.next());
    expect(seqA).toEqual(seqB);
  });

  it('produces different sequences for different seeds', () => {
    const a = new Rng(1);
    const b = new Rng(2);
    expect(a.next()).not.toBe(b.next());
  });

  it('next() stays within [0, 1) and is roughly uniform', () => {
    const r = new Rng(7);
    let sum = 0;
    const n = 10000;
    for (let i = 0; i < n; i++) {
      const x = r.next();
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(1);
      sum += x;
    }
    expect(sum / n).toBeCloseTo(0.5, 1); // within 0.05
  });

  it('gaussian() has ~0 mean and ~1 std', () => {
    const r = new Rng(123);
    const n = 20000;
    let mean = 0;
    const xs: number[] = [];
    for (let i = 0; i < n; i++) {
      const x = r.gaussian();
      xs.push(x);
      mean += x;
    }
    mean /= n;
    let varSum = 0;
    for (const x of xs) varSum += (x - mean) * (x - mean);
    const std = Math.sqrt(varSum / n);
    expect(Math.abs(mean)).toBeLessThan(0.05);
    expect(Math.abs(std - 1)).toBeLessThan(0.05);
  });

  it('range(lo, hi) stays within bounds', () => {
    const r = new Rng(9);
    for (let i = 0; i < 1000; i++) {
      const x = r.range(-3, 5);
      expect(x).toBeGreaterThanOrEqual(-3);
      expect(x).toBeLessThan(5);
    }
  });

  it('int(maxExclusive) returns integers in [0, max)', () => {
    const r = new Rng(11);
    const counts = new Array(5).fill(0);
    for (let i = 0; i < 5000; i++) {
      const x = r.int(5);
      expect(Number.isInteger(x)).toBe(true);
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThan(5);
      counts[x]++;
    }
    // every bucket should be hit
    expect(counts.every((c) => c > 0)).toBe(true);
  });

  it('shuffleInPlace is a deterministic permutation', () => {
    const base = () => [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const a = base();
    const b = base();
    new Rng(5).shuffleInPlace(a);
    new Rng(5).shuffleInPlace(b);
    expect(a).toEqual(b); // deterministic
    expect([...a].sort((x, y) => x - y)).toEqual(base()); // same multiset
    expect(a).not.toEqual(base()); // actually shuffled
  });
});
