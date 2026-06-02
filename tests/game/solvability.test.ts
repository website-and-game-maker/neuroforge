import { describe, it, expect } from 'vitest';
import { CHALLENGES } from '../../src/game/challenges';
import { mlp } from '../../src/engine/network';
import { activationByName, Sigmoid, Identity } from '../../src/engine/activations';
import { BCE, MSE } from '../../src/engine/losses';
import { Rng } from '../../src/engine/rng';
import { Trainer, accuracy, mseMetric } from '../../src/engine/trainer';

/**
 * Every challenge must be solvable with its own suggested starter config within a
 * realistic training budget, judged on a *held-out* test split. If this fails, the
 * game has an unwinnable level — a design bug, not just a flaky test.
 */
describe('every challenge is solvable with its starter config', () => {
  for (const ch of CHALLENGES) {
    it(`${ch.id} reaches its target on the test split`, () => {
      const train = ch.makeData(ch.trainSeed);
      const test = ch.makeData(ch.testSeed);
      const isClass = ch.task === 'classification';
      const net = mlp(
        train.inDim,
        ch.starter.hidden,
        1,
        activationByName(ch.starter.activation),
        isClass ? Sigmoid : Identity,
        new Rng(1000),
      );
      const trainer = new Trainer(
        net,
        isClass ? BCE : MSE,
        train.X,
        train.Y,
        {
          lr: ch.starter.lr,
          momentum: ch.starter.momentum,
          l2: ch.starter.l2,
          batchSize: ch.starter.batchSize,
        },
        new Rng(2000),
      );
      trainer.step(5000);

      const pred = trainer.predict(test.X);
      if (ch.target.kind === 'accuracy') {
        expect(accuracy(pred, test.Y)).toBeGreaterThanOrEqual(ch.target.min);
      } else {
        expect(mseMetric(pred, test.Y)).toBeLessThanOrEqual(ch.target.max);
      }
    });
  }
});
