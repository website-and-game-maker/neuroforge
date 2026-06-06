import { describe, it, expect } from 'vitest';
import {
  VersusMatch,
  VERSUS_RULES,
  saboteurMove,
  trainerPlan,
  trainerHint,
  neuronsUsed,
  type Difficulty,
} from '../../src/game/versus';
import { moonsData } from '../../src/data/datasets';
import { Studio, defaultConfig } from '../../src/ui/studio';
import { Rng } from '../../src/engine/rng';

function trainedStudio(match: VersusMatch): Studio {
  const s = new Studio({ ...defaultConfig(), hidden: [8], activation: 'tanh', lr: 0.2 });
  s.task = 'classification';
  s.setData(match.dataset(), match.dataset());
  s.rebuild(true);
  s.step(400);
  return s;
}

describe('VersusMatch', () => {
  it('combines base + sabotage into one dataset', () => {
    const base = moonsData(40, 1);
    const m = new VersusMatch(VERSUS_RULES, base);
    expect(m.dataset().X.rows).toBe(40);
    m.addSabotage([{ x: 0.1, y: 0.2, label: 1, byAi: true }]);
    const ds = m.dataset();
    expect(ds.X.rows).toBe(41);
    expect(ds.Y.data[40]).toBe(1);
    expect(ds.X.get(40, 0)).toBeCloseTo(0.1);
  });

  it('alternates phases with the trainer getting the last word', () => {
    const m = new VersusMatch({ ...VERSUS_RULES, rounds: 2 }, moonsData(20, 1));
    expect(m.phase).toBe('trainer');
    m.endTrainerTurn();
    expect(m.phase).toBe('saboteur');
    m.endSaboteurTurn(); // attack 1 done
    expect(m.phase).toBe('trainer');
    m.endTrainerTurn();
    expect(m.phase).toBe('saboteur');
    m.endSaboteurTurn(); // attack 2 done (== rounds)
    expect(m.phase).toBe('trainer'); // final defense
    m.endTrainerTurn();
    expect(m.phase).toBe('done');
  });

  it('scores accuracy over all points and decides a winner only when done', () => {
    const m = new VersusMatch(VERSUS_RULES, moonsData(60, 2));
    const s = trainedStudio(m);
    const acc = m.score(s.forward);
    expect(acc).toBeGreaterThan(0);
    expect(acc).toBeLessThanOrEqual(1);
    expect(m.winner(s.forward)).toBeNull();
  });
});

describe('saboteurMove', () => {
  it('places ≤ budget opposite-class points within the arena', () => {
    const m = new VersusMatch(VERSUS_RULES, moonsData(80, 3));
    const s = trainedStudio(m);
    const pts = saboteurMove(s.forward, m, 'medium', new Rng(5));
    expect(pts.length).toBeGreaterThan(0);
    expect(pts.length).toBeLessThanOrEqual(VERSUS_RULES.pointsPerRound);
    for (const p of pts) {
      expect(p.byAi).toBe(true);
      expect([0, 1]).toContain(p.label);
      expect(Math.abs(p.x)).toBeLessThan(1.2);
      expect(Math.abs(p.y)).toBeLessThan(1.2);
    }
  });

  it('never stacks points on the same spot', () => {
    const m = new VersusMatch(VERSUS_RULES, moonsData(80, 4));
    const s = trainedStudio(m);
    const pts = saboteurMove(s.forward, m, 'hard', new Rng(9));
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot(pts[i]!.x - pts[j]!.x, pts[i]!.y - pts[j]!.y);
        expect(d).toBeGreaterThan(0.02);
      }
    }
  });

  it('fills its full quota via the relaxed fallback even on easy', () => {
    const m = new VersusMatch(VERSUS_RULES, moonsData(80, 5));
    const s = trainedStudio(m);
    const pts = saboteurMove(s.forward, m, 'easy', new Rng(3));
    expect(pts.length).toBe(VERSUS_RULES.pointsPerRound);
  });
});

describe('trainerPlan', () => {
  it('never exceeds the neuron budget at any difficulty', () => {
    const m = new VersusMatch(VERSUS_RULES, moonsData(80, 6));
    for (const skill of ['easy', 'medium', 'hard'] as Difficulty[]) {
      const plan = trainerPlan(m, skill);
      expect(neuronsUsed(plan.config)).toBeLessThanOrEqual(VERSUS_RULES.neuronBudget);
      expect(plan.steps).toBeGreaterThan(0);
    }
  });
});

describe('trainerHint', () => {
  it('is empty when hints are off and non-empty otherwise', () => {
    const m = new VersusMatch(VERSUS_RULES, moonsData(40, 1));
    expect(trainerHint(m, defaultConfig(), 0.7, 0)).toBe('');
    expect(trainerHint(m, defaultConfig(), 0.7, 1).length).toBeGreaterThan(0);
  });
});
