import { describe, it, expect } from 'vitest';
import { mlp } from '../../src/engine/network';
import { Tanh, Sigmoid } from '../../src/engine/activations';
import { Matrix } from '../../src/engine/matrix';
import { Rng } from '../../src/engine/rng';

describe('Network.forwardVerbose', () => {
  const net = mlp(2, [4, 3], 1, Tanh, Sigmoid, new Rng(7));
  const X = Matrix.fromRows([
    [0.1, 0.2],
    [-0.3, 0.4],
    [0.5, -0.6],
  ]);

  it('returns one column per neuron layer: input, hiddens, output', () => {
    const { columns } = net.forwardVerbose(X);
    expect(columns.length).toBe(4); // input + 2 hidden + output
    expect(columns.map((c) => c.cols)).toEqual([2, 4, 3, 1]);
    expect(columns.every((c) => c.rows === X.rows)).toBe(true);
  });

  it('column[0] is the input and the last column equals forward()', () => {
    const { columns } = net.forwardVerbose(X);
    expect(columns[0]!.data).toEqual(X.data);
    const out = net.forward(X);
    const last = columns[columns.length - 1]!;
    for (let i = 0; i < out.data.length; i++) {
      expect(last.data[i]!).toBeCloseTo(out.data[i]!, 12);
    }
  });

  it('handles a no-hidden-layer network', () => {
    const lin = mlp(2, [], 1, Tanh, Sigmoid, new Rng(1));
    const { columns } = lin.forwardVerbose(X);
    expect(columns.map((c) => c.cols)).toEqual([2, 1]);
  });
});
