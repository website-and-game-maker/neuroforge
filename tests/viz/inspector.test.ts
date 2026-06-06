import { describe, it, expect } from 'vitest';
import {
  computeInspectorLayout,
  hitTestNeuron,
  neuronField,
} from '../../src/viz/inspector';
import { mlp } from '../../src/engine/network';
import { Tanh, Sigmoid } from '../../src/engine/activations';
import { Matrix } from '../../src/engine/matrix';
import { Rng } from '../../src/engine/rng';

const net = mlp(2, [4, 3], 1, Tanh, Sigmoid, new Rng(3));

describe('computeInspectorLayout', () => {
  it('lays out one column per layer with correct sizes', () => {
    const layout = computeInspectorLayout(net, 320, 220, 'classification');
    expect(layout.cols.map((c) => c.size)).toEqual([2, 4, 3, 1]);
    expect(layout.cols[0]!.header).toBe('INPUT');
    expect(layout.cols[layout.cols.length - 1]!.header).toBe('OUTPUT');
    expect(layout.cols[0]!.nodes.length).toBe(2);
  });

  it('samples very wide layers down to a display cap', () => {
    const wide = mlp(2, [40], 1, Tanh, Sigmoid, new Rng(1));
    const layout = computeInspectorLayout(wide, 320, 220, 'classification');
    const hidden = layout.cols[1]!;
    expect(hidden.size).toBe(40);
    expect(hidden.sampled).toBe(true);
    expect(hidden.nodes.length).toBeLessThanOrEqual(16);
  });

  it('hit-tests a neuron at its rendered position', () => {
    const layout = computeInspectorLayout(net, 320, 220, 'classification');
    const node = layout.cols[1]!.nodes[0]!;
    const hit = hitTestNeuron(layout, layout.cols[1]!.x, node.y);
    expect(hit).toEqual({ col: 1, idx: node.idx });
    expect(hitTestNeuron(layout, -50, -50)).toBeNull();
  });
});

describe('neuronField', () => {
  it('produces a single squashed activation channel in [0,1]', () => {
    const X = Matrix.fromRows([
      [0.2, 0.1],
      [-0.5, 0.3],
    ]);
    const out = neuronField(net, { col: 1, idx: 0 })(X);
    expect(out.rows).toBe(2);
    expect(out.cols).toBe(1);
    for (const v of out.data) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
