import { describe, it, expect } from 'vitest';
import { Matrix } from '../../src/engine/matrix';
import { SGD } from '../../src/engine/optimizer';
import type { Param } from '../../src/engine/layers';

function param(values: number[], grads: number[]): Param {
  return {
    value: Matrix.fromRows([values]),
    grad: Matrix.fromRows([grads]),
  };
}

describe('SGD', () => {
  it('plain step moves params by -lr * grad', () => {
    const p = param([1.0, -2.0], [0.5, 4.0]);
    new SGD(0.1, 0, 0).step([p]);
    expect(p.value.toRows()[0]![0]).toBeCloseTo(1.0 - 0.1 * 0.5, 12);
    expect(p.value.toRows()[0]![1]).toBeCloseTo(-2.0 - 0.1 * 4.0, 12);
  });

  it('momentum accumulates velocity across steps', () => {
    const p = param([1.0], [0.5]);
    const opt = new SGD(0.1, 0.9, 0);
    opt.step([p]); // v=-0.05, w=0.95
    expect(p.value.get(0, 0)).toBeCloseTo(0.95, 12);
    opt.step([p]); // v=0.9*-0.05 - 0.05 = -0.095, w=0.855
    expect(p.value.get(0, 0)).toBeCloseTo(0.855, 12);
  });

  it('L2 adds a weight-decay term', () => {
    const p = param([2.0], [0.0]);
    new SGD(0.1, 0, 0.5).step([p]); // grad = 0 + 0.5*2 = 1; w = 2 - 0.1 = 1.9
    expect(p.value.get(0, 0)).toBeCloseTo(1.9, 12);
  });

  it('keys velocity on the stable value identity (separate optimizers do not share)', () => {
    const p = param([1.0], [1.0]);
    new SGD(0.1, 0.9, 0).step([p]); // optimizer A, v=-0.1, w=0.9
    expect(p.value.get(0, 0)).toBeCloseTo(0.9, 12);
    // A fresh optimizer has no stored velocity, so it behaves like step 1 again.
    p.grad.set(0, 0, 1.0);
    new SGD(0.1, 0.9, 0).step([p]); // v=-0.1, w=0.8
    expect(p.value.get(0, 0)).toBeCloseTo(0.8, 12);
  });
});
