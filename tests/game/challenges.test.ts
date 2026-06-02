import { describe, it, expect } from 'vitest';
import { CHALLENGES } from '../../src/game/challenges';
import { activationByName } from '../../src/engine/activations';

describe('CHALLENGES', () => {
  it('is non-empty with unique ids', () => {
    expect(CHALLENGES.length).toBeGreaterThan(0);
    const ids = CHALLENGES.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('orders classification challenges before regression', () => {
    const firstRegression = CHALLENGES.findIndex((c) => c.task === 'regression');
    const lastClassification = CHALLENGES.map((c) => c.task).lastIndexOf('classification');
    expect(firstRegression).toBeGreaterThan(lastClassification);
  });

  it('each challenge has a valid starter and target', () => {
    for (const c of CHALLENGES) {
      expect(Array.isArray(c.starter.hidden)).toBe(true);
      expect(() => activationByName(c.starter.activation)).not.toThrow();
      expect(c.starter.lr).toBeGreaterThan(0);
      expect(c.starter.batchSize).toBeGreaterThanOrEqual(1);
      expect(c.starter.momentum).toBeGreaterThanOrEqual(0);
      expect(c.starter.l2).toBeGreaterThanOrEqual(0);

      if (c.target.kind === 'accuracy') {
        expect(c.target.min).toBeGreaterThan(0);
        expect(c.target.min).toBeLessThanOrEqual(1);
      } else {
        expect(c.target.max).toBeGreaterThan(0);
      }
      expect(c.why.length).toBeGreaterThan(20);
    }
  });

  it('each makeData matches its declared task', () => {
    for (const c of CHALLENGES) {
      const train = c.makeData(c.trainSeed);
      const test = c.makeData(c.testSeed);
      expect(train.kind).toBe(c.task);
      expect(test.kind).toBe(c.task);
      // train/test splits must differ
      expect(train.X.toRows()).not.toEqual(test.X.toRows());
    }
  });
});
