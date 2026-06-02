import { describe, it, expect } from 'vitest';
import {
  CLASS_A,
  CLASS_B,
  rgba,
  mix,
  probColor,
  classColor,
  weightColor,
} from '../../src/viz/colors';

describe('colors', () => {
  it('rgba formats a colour string', () => {
    expect(rgba({ r: 10, g: 20, b: 30, a: 0.5 })).toBe('rgba(10, 20, 30, 0.5)');
  });

  it('mix returns endpoints at t=0 and t=1 and the midpoint at t=0.5', () => {
    expect(mix(CLASS_A, CLASS_B, 0)).toEqual(CLASS_A);
    expect(mix(CLASS_A, CLASS_B, 1)).toEqual(CLASS_B);
    const m = mix(CLASS_A, CLASS_B, 0.5);
    expect(m.r).toBeCloseTo((CLASS_A.r + CLASS_B.r) / 2, 6);
  });

  it('mix clamps t outside [0,1]', () => {
    expect(mix(CLASS_A, CLASS_B, -5)).toEqual(CLASS_A);
    expect(mix(CLASS_A, CLASS_B, 5)).toEqual(CLASS_B);
  });

  it('probColor is blue-dominant near 0 and orange-dominant near 1', () => {
    expect(probColor(0).b).toBeGreaterThan(probColor(0).r);
    expect(probColor(1).r).toBeGreaterThan(probColor(1).b);
  });

  it('classColor maps labels to solid class colours', () => {
    expect(classColor(0)).toEqual(CLASS_A);
    expect(classColor(1)).toEqual(CLASS_B);
  });

  it('weightColor: sign chooses hue, magnitude scales opacity', () => {
    expect(weightColor(2, 2).r).toBeGreaterThan(weightColor(2, 2).b); // positive → warm
    expect(weightColor(-2, 2).b).toBeGreaterThan(weightColor(-2, 2).r); // negative → cool
    expect(weightColor(2, 2).a).toBeGreaterThan(weightColor(0.2, 2).a); // bigger |w| → more opaque
  });
});
