import { describe, it, expect } from 'vitest';
import { Matrix, matmul, addRowVector } from '../../src/engine/matrix';
import { Dense, ActivationLayer } from '../../src/engine/layers';
import { ReLU, Tanh, Sigmoid, Identity, type Activation } from '../../src/engine/activations';
import { Rng } from '../../src/engine/rng';
import { numericalGradient, maxRelError, dot } from '../helpers/gradcheck';

describe('Dense forward', () => {
  it('computes x·W + b', () => {
    const dense = new Dense(2, 2, new Rng(1));
    // Overwrite weights with known values for a deterministic check.
    dense.W.data.set([1, 2, 3, 4]); // [[1,2],[3,4]]
    dense.b.data.set([10, 20]);
    const x = Matrix.fromRows([[1, 1]]);
    const expected = addRowVector(matmul(x, dense.W), dense.b);
    expect(dense.forward(x).toRows()).toEqual(expected.toRows());
    expect(dense.forward(x).toRows()).toEqual([[1 * 1 + 1 * 3 + 10, 1 * 2 + 1 * 4 + 20]]);
  });
});

describe('Dense backward — gradient check', () => {
  it('dW, db, and dX match finite differences', () => {
    const rng = new Rng(7);
    const dense = new Dense(3, 2, rng);
    const X = Matrix.fromRows([
      [0.5, -1.0, 2.0],
      [1.0, 0.3, -0.7],
    ]);
    // Fixed upstream gradient G, so loss = sum(forward(X) ⊙ G), dL/dY = G.
    const G = Matrix.fromRows([
      [0.7, -0.2],
      [0.1, 0.9],
    ]);
    const lossAt = (): number => dot(dense.forward(X).data, G.data);

    // Analytic gradients.
    dense.forward(X);
    const dX = dense.backward(G);
    const [pW, pB] = dense.params();

    // Numeric gradients.
    const numW = numericalGradient(lossAt, pW!.value.data);
    const numB = numericalGradient(lossAt, pB!.value.data);
    const numX = numericalGradient(lossAt, X.data);

    expect(maxRelError(pW!.grad.data, numW)).toBeLessThan(1e-5);
    expect(maxRelError(pB!.grad.data, numB)).toBeLessThan(1e-5);
    expect(maxRelError(dX.data, numX)).toBeLessThan(1e-5);
  });
});

describe('ActivationLayer backward — gradient check', () => {
  // Points kept away from 0 so ReLU's kink (non-differentiable at x=0) never lands
  // inside a finite-difference step.
  const X = Matrix.fromRows([[0.5, -1.0, 2.0, 0.1]]);
  const G = Matrix.fromRows([[0.3, 0.7, -0.2, 1.0]]);

  const activations: Activation[] = [ReLU, Tanh, Sigmoid, Identity];
  for (const activation of activations) {
    it(`${activation.name} dX matches finite differences`, () => {
      const act = new ActivationLayer(activation);
      const lossAt = (): number => dot(act.forward(X).data, G.data);

      act.forward(X);
      const dX = act.backward(G);
      const numX = numericalGradient(lossAt, X.data);

      expect(maxRelError(dX.data, numX)).toBeLessThan(1e-5);
      expect(act.params()).toEqual([]);
    });
  }
});

describe('Dense params', () => {
  it('value identities are stable; grads refresh after backward', () => {
    const dense = new Dense(2, 3, new Rng(2));
    const p1 = dense.params();
    const p2 = dense.params();
    expect(p1[0]!.value).toBe(p2[0]!.value); // stable identity (optimizer keys on this)
    expect(p1[1]!.value).toBe(p2[1]!.value);

    const X = Matrix.fromRows([[1, 2]]);
    dense.forward(X);
    dense.backward(Matrix.fromRows([[1, 1, 1]]));
    // db should now be the column-sum of the upstream gradient = [1,1,1].
    expect(dense.params()[1]!.grad.toRows()).toEqual([[1, 1, 1]]);
  });
});
