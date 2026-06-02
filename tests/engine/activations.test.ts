import { describe, it, expect } from 'vitest';
import {
  ReLU,
  Tanh,
  Sigmoid,
  Identity,
  activationByName,
} from '../../src/engine/activations';
import type { Activation } from '../../src/engine/activations';

describe('activations: known values', () => {
  it('ReLU', () => {
    expect(ReLU.f(-2)).toBe(0);
    expect(ReLU.f(0)).toBe(0);
    expect(ReLU.f(3)).toBe(3);
    expect(ReLU.df(-2)).toBe(0);
    expect(ReLU.df(3)).toBe(1);
  });

  it('Tanh', () => {
    expect(Tanh.f(0)).toBeCloseTo(0, 12);
    expect(Tanh.df(0)).toBeCloseTo(1, 12);
  });

  it('Sigmoid', () => {
    expect(Sigmoid.f(0)).toBeCloseTo(0.5, 12);
    expect(Sigmoid.df(0)).toBeCloseTo(0.25, 12);
    // saturates
    expect(Sigmoid.f(50)).toBeCloseTo(1, 6);
    expect(Sigmoid.f(-50)).toBeCloseTo(0, 6);
  });

  it('Identity', () => {
    expect(Identity.f(3.5)).toBe(3.5);
    expect(Identity.df(3.5)).toBe(1);
  });
});

describe('activations: analytic derivative matches numeric', () => {
  const all: Activation[] = [ReLU, Tanh, Sigmoid, Identity];
  const points = [-2.3, -0.7, 0.4, 1.1, 2.8];
  const eps = 1e-6;

  for (const act of all) {
    it(`${act.name} df ≈ centered finite difference`, () => {
      for (const x of points) {
        if (act === ReLU && Math.abs(x) < 1e-3) continue; // skip kink
        const numeric = (act.f(x + eps) - act.f(x - eps)) / (2 * eps);
        expect(act.df(x)).toBeCloseTo(numeric, 5);
      }
    });
  }
});

describe('activationByName', () => {
  it('resolves known names case-insensitively', () => {
    expect(activationByName('relu')).toBe(ReLU);
    expect(activationByName('Tanh')).toBe(Tanh);
    expect(activationByName('sigmoid')).toBe(Sigmoid);
    expect(activationByName('identity')).toBe(Identity);
  });

  it('throws on unknown name', () => {
    expect(() => activationByName('gelu')).toThrow();
  });
});
