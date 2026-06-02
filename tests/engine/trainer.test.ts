import { describe, it, expect } from 'vitest';
import { Matrix } from '../../src/engine/matrix';
import { mlp } from '../../src/engine/network';
import { Tanh, Sigmoid, Identity } from '../../src/engine/activations';
import { BCE, MSE } from '../../src/engine/losses';
import { Rng } from '../../src/engine/rng';
import { Trainer, accuracy, mseMetric } from '../../src/engine/trainer';

describe('metrics', () => {
  it('accuracy thresholds at 0.5', () => {
    const pred = Matrix.fromRows([[0.9], [0.2], [0.6], [0.4]]);
    const tgt = Matrix.fromRows([[1], [0], [0], [1]]);
    expect(accuracy(pred, tgt)).toBeCloseTo(0.5, 12);
  });

  it('mseMetric is mean squared error', () => {
    const pred = Matrix.fromRows([[1], [2]]);
    const tgt = Matrix.fromRows([[1], [0]]);
    expect(mseMetric(pred, tgt)).toBeCloseTo(2.0, 12);
  });
});

// Clustered XOR: four gaussian blobs at (±0.5, ±0.5); same-sign → 0, opposite → 1.
function makeXor(n: number, seed: number): { X: Matrix; Y: Matrix } {
  const rng = new Rng(seed);
  const centers = [
    [0.5, 0.5, 0],
    [-0.5, -0.5, 0],
    [0.5, -0.5, 1],
    [-0.5, 0.5, 1],
  ];
  const xs: number[][] = [];
  const ys: number[][] = [];
  for (let i = 0; i < n; i++) {
    const c = centers[i % 4]!;
    xs.push([c[0]! + rng.gaussian() * 0.1, c[1]! + rng.gaussian() * 0.1]);
    ys.push([c[2]!]);
  }
  return { X: Matrix.fromRows(xs), Y: Matrix.fromRows(ys) };
}

describe('Trainer mechanics', () => {
  it('step(k) advances stepCount and appends k loss entries; deterministic per seed', () => {
    const { X, Y } = makeXor(64, 1);
    const cfg = { lr: 0.1, momentum: 0.9, l2: 0, batchSize: 16 };
    const a = new Trainer(mlp(2, [8], 1, Tanh, Sigmoid, new Rng(3)), BCE, X, Y, cfg, new Rng(99));
    const b = new Trainer(mlp(2, [8], 1, Tanh, Sigmoid, new Rng(3)), BCE, X, Y, cfg, new Rng(99));
    a.step(5);
    b.step(5);
    expect(a.stepCount).toBe(5);
    expect(a.lossHistory.length).toBe(5);
    expect(a.lossHistory).toEqual(b.lossHistory); // identical seeds -> identical training
  });
});

describe('Trainer learns', () => {
  it('learns clustered XOR to >= 95% train accuracy', () => {
    const { X, Y } = makeXor(200, 1);
    const net = mlp(2, [8, 8], 1, Tanh, Sigmoid, new Rng(7));
    const trainer = new Trainer(
      net,
      BCE,
      X,
      Y,
      { lr: 0.3, momentum: 0.9, l2: 0, batchSize: 16 },
      new Rng(123),
    );
    trainer.step(2000);
    const acc = accuracy(trainer.predict(X), Y);
    expect(acc).toBeGreaterThanOrEqual(0.95);
  });

  it('fits a sine curve below an MSE threshold', () => {
    const rng = new Rng(2);
    const xs: number[][] = [];
    const ys: number[][] = [];
    for (let i = 0; i < 200; i++) {
      const x = rng.range(-1, 1);
      xs.push([x]);
      ys.push([Math.sin(Math.PI * x) * 0.8]);
    }
    const X = Matrix.fromRows(xs);
    const Y = Matrix.fromRows(ys);
    const net = mlp(1, [16, 16], 1, Tanh, Identity, new Rng(5));
    const trainer = new Trainer(
      net,
      MSE,
      X,
      Y,
      { lr: 0.05, momentum: 0.9, l2: 0, batchSize: 16 },
      new Rng(321),
    );
    trainer.step(3000);
    expect(mseMetric(trainer.predict(X), Y)).toBeLessThan(0.02);
  });
});
