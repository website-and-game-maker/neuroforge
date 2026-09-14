import { describe, it, expect } from 'vitest';
import { Matrix } from '../../src/engine/matrix';
import { BCE, MSE } from '../../src/engine/losses';
import { numericalGradient, maxRelError } from '../helpers/gradcheck';

describe('MSE', () => {
  it('forward computes mean squared error over all elements', () => {
    const out = Matrix.fromRows([[0.5], [-1.0], [2.0]]);
    const tgt = Matrix.fromRows([[1.0], [0.0], [1.5]]);
    // diffs: -0.5, -1.0, 0.5 -> squares 0.25, 1, 0.25 -> mean = 0.5
    expect(MSE.forward(out, tgt)).toBeCloseTo(0.5, 12);
  });

  it('backward matches finite differences', () => {
    const out = Matrix.fromRows([[0.5], [-1.0], [2.0], [0.2]]);
    const tgt = Matrix.fromRows([[1.0], [0.0], [1.5], [-0.3]]);
    const analytic = MSE.backward(out, tgt);
    const numeric = numericalGradient(() => MSE.forward(out, tgt), out.data);
    expect(maxRelError(analytic.data, numeric)).toBeLessThan(1e-5);
  });
});

describe('BCE', () => {
  it('forward computes mean binary cross-entropy', () => {
    const out = Matrix.fromRows([[0.3], [0.7]]);
    const tgt = Matrix.fromRows([[0], [1]]);
    // -[log(0.7) + log(0.7)] / 2 = -log(0.7) = 0.356675
    expect(BCE.forward(out, tgt)).toBeCloseTo(0.3566749, 6);
  });

  it('backward matches finite differences (non-saturated)', () => {
    const out = Matrix.fromRows([[0.3], [0.7], [0.6], [0.45]]);
    const tgt = Matrix.fromRows([[0], [1], [1], [0]]);
    const analytic = BCE.backward(out, tgt);
    const numeric = numericalGradient(() => BCE.forward(out, tgt), out.data, 1e-6);
    expect(maxRelError(analytic.data, numeric)).toBeLessThan(1e-4);
  });

  it('is finite even at saturated predictions (clamping)', () => {
    const out = Matrix.fromRows([[0], [1]]);
    const tgt = Matrix.fromRows([[1], [0]]); // worst case
    expect(Number.isFinite(BCE.forward(out, tgt))).toBe(true);
  });

  it('backward matches finite differences at saturated predictions (0/1)', () => {
    // forward() clamps p into [EPS, 1-EPS], which is locally flat right at the
    // saturation points — the true gradient there is 0, not the huge value you'd
    // get from plugging the clamped p into the unclamped derivative formula.
    const out = Matrix.fromRows([[0], [1], [0], [1]]);
    const tgt = Matrix.fromRows([[1], [0], [0], [1]]);
    const analytic = BCE.backward(out, tgt);
    // eps small enough that both perturbed points still clamp to the same value.
    const numeric = numericalGradient(() => BCE.forward(out, tgt), out.data, 1e-9);
    expect(maxRelError(analytic.data, numeric)).toBeLessThan(1e-4);
  });
});
