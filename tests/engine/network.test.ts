import { describe, it, expect } from 'vitest';
import { Matrix } from '../../src/engine/matrix';
import { Network, mlp } from '../../src/engine/network';
import { Tanh, Sigmoid, Identity } from '../../src/engine/activations';
import { BCE } from '../../src/engine/losses';
import { Rng } from '../../src/engine/rng';
import { numericalGradient, maxRelError } from '../helpers/gradcheck';

describe('mlp construction', () => {
  it('produces output of shape [n×outDim] in (0,1) with a sigmoid head', () => {
    const net = mlp(2, [4], 1, Tanh, Sigmoid, new Rng(1));
    const X = Matrix.fromRows([
      [0.2, -0.5],
      [1.0, 0.3],
      [-0.7, 0.8],
    ]);
    const y = net.forward(X);
    expect(y.rows).toBe(3);
    expect(y.cols).toBe(1);
    for (const v of y.data) {
      expect(v).toBeGreaterThan(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('params() count equals the sum of Dense parameters', () => {
    const net = mlp(2, [3, 5], 1, Tanh, Identity, new Rng(2));
    // Dense layers: (2→3),(3→5),(5→1) => 3 Dense × (W,b) = 6 params
    expect(net.params().length).toBe(6);
  });
});

describe('Network end-to-end gradient check', () => {
  it('every parameter gradient matches finite differences', () => {
    const rng = new Rng(7);
    const net = mlp(2, [4, 3], 1, Tanh, Sigmoid, rng);
    const X = Matrix.fromRows([
      [0.5, -1.0],
      [-0.3, 0.7],
      [1.2, 0.1],
      [-0.8, -0.6],
    ]);
    const Y = Matrix.fromRows([[1], [0], [1], [0]]);

    const lossAt = (): number => BCE.forward(net.forward(X), Y);

    // Analytic gradients via one forward + backward.
    const out = net.forward(X);
    net.backward(BCE.backward(out, Y));
    const params = net.params();

    for (const p of params) {
      const numeric = numericalGradient(lossAt, p.value.data, 1e-5);
      expect(maxRelError(p.grad.data, numeric)).toBeLessThan(1e-5);
    }
  });
});

describe('Network forward/backward composition', () => {
  it('backward returns dX with the input shape', () => {
    const net = new Network(mlp(3, [4], 2, Tanh, Identity, new Rng(3)).layers);
    const X = Matrix.fromRows([[0.1, 0.2, 0.3]]);
    const out = net.forward(X);
    const dX = net.backward(Matrix.zeros(out.rows, out.cols));
    expect(dX.rows).toBe(1);
    expect(dX.cols).toBe(3);
  });
});
