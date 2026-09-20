import { describe, it, expect } from 'vitest';
import { Matrix } from '../../src/engine/matrix';
import {
  SGD,
  Adam,
  createOptimizer,
  isOptimizerName,
  OPTIMIZER_NAMES,
} from '../../src/engine/optimizer';
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

describe('Adam', () => {
  it('first step moves each weight by ~lr against the gradient sign, whatever its scale', () => {
    // With m and v both bias-corrected, step 1 reduces to  -lr * g / (|g| + eps),
    // i.e. a step of size lr in the direction -sign(g) — independent of |g|. That
    // scale-invariance is the whole point of the method. (eps still costs a relative
    // eps/|g|, which is why the tiny gradient is only matched to 5 places.)
    const p = param([0, 0, 0], [0.0001, -5, 1000]);
    new Adam(0.01).step([p]);
    const w = p.value.toRows()[0]!;
    expect(w[0]!).toBeCloseTo(-0.01, 5);
    expect(w[1]!).toBeCloseTo(+0.01, 9);
    expect(w[2]!).toBeCloseTo(-0.01, 9);
  });

  it('matches the reference update over several steps', () => {
    const grads = [0.5, 0.25, -0.75, 0.1];
    const lr = 0.05;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;

    let w = 1.0;
    let m = 0;
    let v = 0;
    for (let t = 1; t <= grads.length; t++) {
      const g = grads[t - 1]!;
      m = b1 * m + (1 - b1) * g;
      v = b2 * v + (1 - b2) * g * g;
      w -= (lr * (m / (1 - b1 ** t))) / (Math.sqrt(v / (1 - b2 ** t)) + eps);
    }

    const p = param([1.0], [grads[0]!]);
    const opt = new Adam(lr);
    for (const g of grads) {
      p.grad.set(0, 0, g);
      opt.step([p]);
    }
    expect(p.value.get(0, 0)).toBeCloseTo(w, 12);
  });

  it('shares the step counter across params, so bias correction is consistent', () => {
    const a = param([0], [1]);
    const b = param([0], [1]);
    const opt = new Adam(0.01);
    opt.step([a, b]); // both see t = 1
    expect(a.value.get(0, 0)).toBeCloseTo(b.value.get(0, 0), 12);
  });

  it('L2 adds a weight-decay term to the gradient', () => {
    // grad = 0 + l2*w = 0.5*2 = 1 > 0, so the (scale-free) first step is -lr.
    const p = param([2.0], [0.0]);
    new Adam(0.1, 0.5).step([p]);
    expect(p.value.get(0, 0)).toBeCloseTo(1.9, 6);
  });

  it('a zero gradient leaves the weight alone', () => {
    const p = param([3.0], [0.0]);
    new Adam(0.1).step([p]);
    expect(p.value.get(0, 0)).toBeCloseTo(3.0, 12);
  });
});

describe('createOptimizer', () => {
  it('builds each registered optimizer', () => {
    expect(createOptimizer('sgd', { lr: 0.1 })).toBeInstanceOf(SGD);
    expect(createOptimizer('adam', { lr: 0.1 })).toBeInstanceOf(Adam);
    expect(createOptimizer('ADAM', { lr: 0.1 })).toBeInstanceOf(Adam);
  });

  it('passes momentum and l2 through to SGD', () => {
    const p = param([1.0], [0.5]);
    createOptimizer('sgd', { lr: 0.1, momentum: 0.9, l2: 0 }).step([p]);
    expect(p.value.get(0, 0)).toBeCloseTo(0.95, 12);
  });

  it('throws on an unknown name', () => {
    expect(() => createOptimizer('adagrad', { lr: 0.1 })).toThrow(/Unknown optimizer/);
  });

  it('every listed name is constructible, and isOptimizerName agrees', () => {
    for (const name of OPTIMIZER_NAMES) {
      expect(isOptimizerName(name)).toBe(true);
      expect(() => createOptimizer(name, { lr: 0.01 })).not.toThrow();
    }
    expect(isOptimizerName('adagrad')).toBe(false);
  });
});
