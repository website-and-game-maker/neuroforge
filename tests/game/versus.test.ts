import { describe, it, expect } from 'vitest';
import {
  LiveMatch,
  LIVE_RULES,
  pickSabotagePoint,
  liveTrainerPlan,
  liveTrainerHint,
  neuronsUsed,
  type Difficulty,
  type PlacedPoint,
} from '../../src/game/versus';
import { Studio, defaultConfig } from '../../src/ui/studio';
import { Matrix } from '../../src/engine/matrix';
import { Rng } from '../../src/engine/rng';

function seededPoints(n: number, rngSeed = 1): PlacedPoint[] {
  const rng = new Rng(rngSeed);
  return Array.from({ length: n }, (_, i) => ({
    x: rng.range(-0.9, 0.9),
    y: rng.range(-0.9, 0.9),
    label: i % 2,
    byAi: true,
    atMs: i * 100,
  }));
}

function trainedStudio(points: PlacedPoint[]): Studio {
  const s = new Studio({ ...defaultConfig(), hidden: [8], activation: 'tanh', lr: 0.2 });
  s.task = 'classification';
  const m = new LiveMatch(LIVE_RULES);
  for (const p of points) m.addPoint(p.x, p.y, p.label, p.byAi);
  s.setData(m.dataset(), m.dataset());
  s.rebuild(true);
  if (points.length > 0) s.step(300);
  return s;
}

describe('LiveMatch', () => {
  it('advances the clock with clamped deltas and finishes exactly at the duration', () => {
    const m = new LiveMatch({ ...LIVE_RULES, durationMs: 5000 });
    expect(m.done).toBe(false);
    m.advance(2000); // clamped to 1000
    expect(m.elapsedMs).toBe(1000);
    m.advance(-50); // ignored
    expect(m.elapsedMs).toBe(1000);
    for (let i = 0; i < 10; i++) m.advance(1000);
    expect(m.elapsedMs).toBe(5000);
    expect(m.done).toBe(true);
    expect(m.remainingMs).toBe(0);
  });

  it('enforces the point budget and refuses points after the bell', () => {
    const m = new LiveMatch({ ...LIVE_RULES, durationMs: 1000, pointBudget: 2 });
    expect(m.addPoint(0, 0, 0, false)).toBe(true);
    expect(m.addPoint(0.5, 0.5, 1, false)).toBe(true);
    expect(m.addPoint(0.2, 0.2, 0, false)).toBe(false); // budget exhausted
    expect(m.budgetLeft).toBe(0);
    const m2 = new LiveMatch({ ...LIVE_RULES, durationMs: 1000 });
    m2.advance(1000);
    expect(m2.addPoint(0, 0, 0, false)).toBe(false); // match over
  });

  it('builds a dataset from placed points (empty-safe)', () => {
    const m = new LiveMatch(LIVE_RULES);
    expect(m.dataset().X.rows).toBe(0);
    m.addPoint(0.1, -0.2, 1, true);
    const ds = m.dataset();
    expect(ds.X.rows).toBe(1);
    expect(ds.X.get(0, 0)).toBeCloseTo(0.1);
    expect(ds.Y.data[0]).toBe(1);
  });

  it('declares no winner before the bell, forfeits to the trainer with few points', () => {
    const m = new LiveMatch({ ...LIVE_RULES, durationMs: 1000, minPoints: 5 });
    const predict = trainedStudio(seededPoints(4)).forward;
    m.addPoint(0, 0, 0, false);
    expect(m.winner(predict)).toBeNull();
    m.advance(1000);
    expect(m.winner(predict)).toBe('trainer'); // 1 < minPoints
  });

  it('scores accuracy over all placed points and picks the winner by threshold', () => {
    const rules = { ...LIVE_RULES, durationMs: 1000, minPoints: 4 };
    const m = new LiveMatch(rules);
    // Linearly separable points: left = 0, right = 1 — easy to fit.
    for (let i = 0; i < 10; i++) m.addPoint(i < 5 ? -0.6 : 0.6, (i % 5) * 0.2 - 0.4, i < 5 ? 0 : 1, false);
    const s = new Studio({ ...defaultConfig(), hidden: [6], activation: 'tanh', lr: 0.3 });
    s.task = 'classification';
    s.setData(m.dataset(), m.dataset());
    s.rebuild(true);
    s.step(600);
    m.advance(1000);
    const acc = m.score(s.forward);
    expect(acc).toBeGreaterThan(rules.winThreshold);
    expect(m.winner(s.forward)).toBe('trainer');
  });
});

describe('pickSabotagePoint', () => {
  it('seeds points on an empty board (no model yet) for every difficulty', () => {
    for (const skill of ['easy', 'medium', 'hard'] as Difficulty[]) {
      const p = pickSabotagePoint(null, [], skill, new Rng(7));
      expect(p).not.toBeNull();
      expect(Math.abs(p!.x)).toBeLessThan(1);
      expect(Math.abs(p!.y)).toBeLessThan(1);
      expect([0, 1]).toContain(p!.label);
    }
  });

  it('exploits a trained model: places the opposite of a confident prediction', () => {
    const points = seededPoints(20, 3);
    const studio = trainedStudio(points);
    const p = pickSabotagePoint(studio.forward, points, 'medium', new Rng(9));
    expect(p).not.toBeNull();
    // The chosen label must contradict the model's current guess at that spot.
    const prob = studio.forward(new Matrix(1, 2, new Float64Array([p!.x, p!.y]))).data[0]!;
    expect(p!.label).toBe(prob >= 0.5 ? 0 : 1);
  });

  it('never stacks a point onto an existing one', () => {
    const points = seededPoints(30, 5);
    const studio = trainedStudio(points);
    for (let i = 0; i < 5; i++) {
      const p = pickSabotagePoint(studio.forward, points, 'hard', new Rng(100 + i));
      if (!p) continue;
      for (const q of points) {
        expect(Math.hypot(p.x - q.x, p.y - q.y)).toBeGreaterThan(0.05);
      }
    }
  });
});

describe('liveTrainerPlan', () => {
  it('respects the neuron budget at every difficulty and board size', () => {
    for (const skill of ['easy', 'medium', 'hard'] as Difficulty[]) {
      for (const pts of [0, 20, 60]) {
        const plan = liveTrainerPlan(pts, skill, LIVE_RULES.neuronBudget);
        expect(neuronsUsed(plan)).toBeLessThanOrEqual(LIVE_RULES.neuronBudget);
        expect(plan.hidden.every((w) => w >= 1)).toBe(true);
        expect(plan.lr).toBeGreaterThan(0);
      }
    }
  });

  it('spends more capacity as points accumulate (hard)', () => {
    const small = neuronsUsed(liveTrainerPlan(0, 'hard', LIVE_RULES.neuronBudget));
    const big = neuronsUsed(liveTrainerPlan(60, 'hard', LIVE_RULES.neuronBudget));
    expect(big).toBeGreaterThanOrEqual(small);
  });
});

describe('liveTrainerHint', () => {
  it('is empty at level 0 and non-empty otherwise, mentioning time left', () => {
    const m = new LiveMatch(LIVE_RULES);
    m.advance(1000);
    expect(liveTrainerHint(m, defaultConfig(), 0.7, 0)).toBe('');
    const h = liveTrainerHint(m, defaultConfig(), 0.7, 3);
    expect(h.length).toBeGreaterThan(0);
    expect(h).toContain('s left');
  });

  it('points a struggling trainer at unused capacity', () => {
    const m = new LiveMatch(LIVE_RULES);
    for (const p of seededPoints(10)) m.addPoint(p.x, p.y, p.label, true);
    const h = liveTrainerHint(m, { ...defaultConfig(), hidden: [4] }, 0.6, 1);
    expect(h.toLowerCase()).toContain('neuron');
  });
});
