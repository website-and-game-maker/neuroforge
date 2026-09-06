import { describe, it, expect } from 'vitest';
import {
  computeInspectorLayout,
  drawInspector,
  hitTestNeuron,
  neuronField,
} from '../../src/viz/inspector';
import { mlp } from '../../src/engine/network';
import { Tanh, Sigmoid, Identity } from '../../src/engine/activations';
import { Matrix } from '../../src/engine/matrix';
import { Rng } from '../../src/engine/rng';

/** A minimal CanvasRenderingContext2D stub that just records fillStyle at fill() time. */
function makeCtxRecordingFills(): { ctx: CanvasRenderingContext2D; fills: string[] } {
  const fills: string[] = [];
  const ctx = {
    clearRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {},
    arc: () => {},
    fillText: () => {},
    fill: () => {
      fills.push((ctx as unknown as { fillStyle: string }).fillStyle);
    },
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 0,
    font: '',
    textAlign: 'left',
  } as unknown as CanvasRenderingContext2D;
  return { ctx, fills };
}

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

describe('drawInspector output node color (regression)', () => {
  it('gives a negative regression prediction a different color than a positive one', () => {
    // 1-in, 1-out regression net with an unbounded (Identity) output — a raw [0,1]
    // clamp would flatten every negative prediction to the same solid color.
    const regNet = mlp(1, [2], 1, Tanh, Identity, new Rng(1));
    const layout = computeInspectorLayout(regNet, 200, 140, 'regression');

    const negative = makeCtxRecordingFills();
    drawInspector(negative.ctx, layout, {
      task: 'regression',
      probe: [new Matrix(1, 1), new Matrix(1, 2), new Matrix(1, 1, new Float64Array([-5]))],
    });

    const positive = makeCtxRecordingFills();
    drawInspector(positive.ctx, layout, {
      task: 'regression',
      probe: [new Matrix(1, 1), new Matrix(1, 2), new Matrix(1, 1, new Float64Array([5]))],
    });

    // Last fill() call in each run paints the output node (nodes are drawn in column order).
    expect(negative.fills.at(-1)).not.toBe(positive.fills.at(-1));
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
