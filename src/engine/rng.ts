/**
 * Seeded pseudo-random number generator.
 *
 * Determinism matters here: identical seeds must reproduce identical datasets and
 * weight initialisations, both for tests and for reproducible play. Uses mulberry32
 * (a small, well-distributed 32-bit generator) with Box–Muller for normals.
 */
export class Rng {
  private state: number;
  private spareGaussian: number | null = null;

  constructor(seed: number) {
    // Force into a non-zero 32-bit integer state.
    this.state = (seed | 0) || 0x9e3779b9;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    // mulberry32
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Standard normal via Box–Muller (caches the second sample). */
  gaussian(): number {
    if (this.spareGaussian !== null) {
      const v = this.spareGaussian;
      this.spareGaussian = null;
      return v;
    }
    // Avoid log(0) by drawing u1 in (0, 1].
    const u1 = 1 - this.next();
    const u2 = this.next();
    const mag = Math.sqrt(-2 * Math.log(u1));
    this.spareGaussian = mag * Math.sin(2 * Math.PI * u2);
    return mag * Math.cos(2 * Math.PI * u2);
  }

  /** Uniform float in [lo, hi). */
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** Uniform integer in [0, maxExclusive). */
  int(maxExclusive: number): number {
    return Math.floor(this.next() * maxExclusive);
  }

  /** Fisher–Yates in-place shuffle. */
  shuffleInPlace<T>(arr: T[]): void {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const tmp = arr[i]!;
      arr[i] = arr[j]!;
      arr[j] = tmp;
    }
  }
}
