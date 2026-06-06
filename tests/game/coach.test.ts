import { describe, it, expect } from 'vitest';
import { coach, type CoachInput } from '../../src/game/coach';
import type { StudioConfig } from '../../src/ui/studio';

const cfg = (hidden: number[], lr = 0.1): StudioConfig => ({
  hidden,
  activation: 'tanh',
  lr,
  l2: 0,
  batchSize: 16,
  momentum: 0.9,
});

const flat = (v: number, n = 60): number[] => Array.from({ length: n }, () => v);
const rising = (n = 60): number[] => Array.from({ length: n }, (_, i) => 0.2 + i * 0.01);

const base: CoachInput = {
  task: 'classification',
  steps: 500,
  trainScore: 0.6,
  testScore: 0.58,
  lossHistory: flat(0.6),
  config: cfg([8]),
  pointCount: 200,
};

describe('coach', () => {
  it('is idle before any training', () => {
    expect(coach({ ...base, steps: 0 }).tone).toBe('idle');
  });

  it('warns about a high learning rate when loss is rising', () => {
    const tip = coach({ ...base, lossHistory: rising() });
    expect(tip.tone).toBe('warn');
    expect(tip.title.toLowerCase()).toContain('learning rate');
  });

  it('flags a missing hidden layer when stuck and depth 0', () => {
    const tip = coach({ ...base, config: cfg([]), trainScore: 0.55 });
    expect(tip.title.toLowerCase()).toContain('no hidden layer');
  });

  it('celebrates a clean solve with a small train/test gap', () => {
    const tip = coach({
      ...base,
      trainScore: 0.99,
      testScore: 0.97,
      target: { kind: 'accuracy', min: 0.95 },
    });
    expect(tip.tone).toBe('success');
  });

  it('warns about overfitting when train >> test', () => {
    const tip = coach({
      ...base,
      trainScore: 0.99,
      testScore: 0.82,
      target: { kind: 'accuracy', min: 0.95 },
    });
    expect(tip.tone).toBe('good');
    expect(tip.title.toLowerCase()).toContain('overfit');
  });

  it('handles regression: a line cannot bend', () => {
    const tip = coach({
      ...base,
      task: 'regression',
      config: cfg([]),
      trainScore: 0.2,
      testScore: 0.2,
      target: { kind: 'mse', max: 0.02 },
    });
    expect(tip.body.toLowerCase()).toContain('line');
  });
});
