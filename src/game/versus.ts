import { Matrix } from '../engine/matrix';
import { Rng } from '../engine/rng';
import { accuracy } from '../engine/trainer';
import type { Dataset } from '../data/datasets';
import type { StudioConfig } from '../ui/studio';

export type Role = 'trainer' | 'saboteur';
export type Difficulty = 'easy' | 'medium' | 'hard';
export type Phase = 'trainer' | 'saboteur' | 'done';

/** A point the saboteur drops onto the arena. `byAi` marks AI-placed points for the UI. */
export interface SabotagePoint {
  x: number;
  y: number;
  label: number;
  byAi: boolean;
}

/** Match rules — fixed per match for fairness; difficulty only tunes the AI's skill. */
export interface VersusRules {
  /** How many saboteur attacks; the trainer always gets the last word. */
  rounds: number;
  pointsPerRound: number;
  /** Max total hidden neurons the trainer may field. */
  neuronBudget: number;
  /** Accuracy (over all points) at or above which the Trainer wins. */
  winThreshold: number;
  /** Step budget for an automated (AI) trainer turn. */
  aiTrainSteps: number;
}

export const VERSUS_RULES: VersusRules = {
  rounds: 5,
  pointsPerRound: 9,
  neuronBudget: 18,
  winThreshold: 0.88,
  aiTrainSteps: 1600,
};

/** Base-pattern size for a match (kept modest so sabotage is a meaningful fraction). */
export const VERSUS_BASE_POINTS = 100;

/** Predict P(class B / orange) for a set of [x,y] rows. */
export type ProbFn = (X: Matrix) => Matrix;

/* ===========================================================================
   Match state
   ========================================================================= */

export class VersusMatch {
  phase: Phase = 'trainer';
  /** Number of saboteur attacks completed. */
  attacksDone = 0;
  readonly sabotage: SabotagePoint[] = [];

  constructor(
    readonly rules: VersusRules,
    /** Base "true pattern" both sides build on. */
    readonly base: Dataset,
  ) {}

  /** Combined dataset: the base pattern plus every sabotage point placed so far. */
  dataset(): Dataset {
    const baseRows = this.base.X.rows;
    const n = baseRows + this.sabotage.length;
    const X = new Float64Array(n * 2);
    const Y = new Float64Array(n);
    X.set(this.base.X.data, 0);
    Y.set(this.base.Y.data, 0);
    for (let i = 0; i < this.sabotage.length; i++) {
      const s = this.sabotage[i]!;
      X[(baseRows + i) * 2] = s.x;
      X[(baseRows + i) * 2 + 1] = s.y;
      Y[baseRows + i] = s.label;
    }
    return {
      kind: 'classification',
      name: 'Versus',
      inDim: 2,
      X: new Matrix(n, 2, X),
      Y: new Matrix(n, 1, Y),
    };
  }

  addSabotage(points: SabotagePoint[]): void {
    this.sabotage.push(...points);
  }

  /** Saboteur points still available this attack. */
  budgetLeft(placedThisTurn: number): number {
    return Math.max(0, this.rules.pointsPerRound - placedThisTurn);
  }

  endTrainerTurn(): void {
    this.phase = this.attacksDone >= this.rules.rounds ? 'done' : 'saboteur';
  }

  endSaboteurTurn(): void {
    this.attacksDone++;
    this.phase = 'trainer';
  }

  /** 1-based label for the attack currently being prepared. */
  get attackNumber(): number {
    return Math.min(this.attacksDone + 1, this.rules.rounds);
  }

  /** Accuracy over every point currently on the arena. */
  score(predict: ProbFn): number {
    const ds = this.dataset();
    if (ds.X.rows === 0) return 0;
    return accuracy(predict(ds.X), ds.Y);
  }

  /** Final verdict once the match is done. */
  winner(predict: ProbFn): Role | null {
    if (this.phase !== 'done') return null;
    return this.score(predict) >= this.rules.winThreshold ? 'trainer' : 'saboteur';
  }
}

/* ===========================================================================
   Saboteur AI
   ========================================================================= */

function nearestDist(x: number, y: number, pts: Array<{ x: number; y: number }>): number {
  let best = Infinity;
  for (const p of pts) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < best) best = d;
  }
  return best;
}

/**
 * Choose where the saboteur drops points. The core trick: find places the model is
 * *confidently wrong-able* — drop a blue point deep in a region the model calls orange
 * (and vice-versa), forcing the trainer to either misclassify it or contort the boundary.
 *
 * Difficulty scales cleverness: `easy` is nearly random; `medium` greedily targets the
 * most confident regions; `hard` also prefers isolated spots far from same-class support
 * (harder for a capacity-limited network to carve out).
 */
export function saboteurMove(
  predict: ProbFn,
  match: VersusMatch,
  skill: Difficulty,
  rng: Rng,
  count = match.rules.pointsPerRound,
): SabotagePoint[] {
  const existing = [
    ...match.base.X.toRows().map((r, i) => ({ x: r[0]!, y: r[1]!, label: match.base.Y.data[i]! })),
    ...match.sabotage.map((s) => ({ x: s.x, y: s.y, label: s.label })),
  ];

  // Candidate grid over the data region, with jitter so repeated turns differ.
  const candidates: Array<{ x: number; y: number }> = [];
  const G = 22;
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const jx = (rng.next() - 0.5) * (2 / G);
      const jy = (rng.next() - 0.5) * (2 / G);
      const x = -1 + (2 * (i + 0.5)) / G + jx;
      const y = -1 + (2 * (j + 0.5)) / G + jy;
      if (x > -1.05 && x < 1.05 && y > -1.05 && y < 1.05) candidates.push({ x, y });
    }
  }

  const probs = predict(Matrix.fromRows(candidates.map((c) => [c.x, c.y])));

  const minSepOpposite = 0.1; // don't sit right on an opposite-class point (no contradictions)
  const spread = 0.16; // keep our own drops apart

  const scored = candidates.map((c, k) => {
    const p = probs.data[k]!; // P(orange)
    const modelGuess = p >= 0.5 ? 1 : 0;
    const label = 1 - modelGuess; // place the opposite class
    const confidence = Math.abs(p - 0.5) * 2; // [0,1]: how sure the model is here
    const sameClass = existing.filter((e) => e.label === label);
    const oppClass = existing.filter((e) => e.label !== label);
    const sameDist = nearestDist(c.x, c.y, sameClass);
    const oppDist = nearestDist(c.x, c.y, oppClass);
    let s: number;
    if (skill === 'easy') s = rng.next(); // basically random
    else if (skill === 'medium') s = confidence;
    else s = confidence * (0.6 + Math.min(1, sameDist)); // hard: prefer isolated, deep-in-wrong spots
    return { c, label, score: s, oppDist, confidence };
  });

  const chosen: SabotagePoint[] = [];
  const take = (
    list: typeof scored,
    spreadMin: number,
  ): void => {
    for (const cand of list) {
      if (chosen.length >= count) break;
      if (cand.oppDist < minSepOpposite) continue; // never contradict a nearby real point
      if (nearestDist(cand.c.x, cand.c.y, chosen) < spreadMin) continue;
      chosen.push({ x: cand.c.x, y: cand.c.y, label: cand.label, byAi: true });
    }
  };

  // Primary pass: the skill's own ordering at full spread.
  take([...scored].sort((a, b) => b.score - a.score), spread);

  // Fallback: if the quota isn't met, relax the spread and rank by confidence so an
  // attack always lands meaningful points (the minSepOpposite contradiction guard stays).
  if (chosen.length < count) {
    const byConfidence = [...scored].sort((a, b) => b.confidence - a.confidence);
    for (const relaxed of [spread * 0.6, spread * 0.3, 0]) {
      if (chosen.length >= count) break;
      take(byConfidence, relaxed);
    }
  }
  return chosen;
}

/* ===========================================================================
   Trainer AI (auto-config within the neuron budget)
   ========================================================================= */

export interface TrainerPlan {
  config: StudioConfig;
  steps: number;
}

/**
 * Pick an architecture + hyperparameters for an automated trainer turn, respecting the
 * neuron budget. Skill scales how well the budget is used: `easy` under-builds, `hard`
 * spends the budget on a capable two-layer ReLU net.
 */
export function trainerPlan(match: VersusMatch, skill: Difficulty): TrainerPlan {
  const budget = match.rules.neuronBudget;
  const pts = match.dataset().X.rows;
  // More points on the board → lean toward more capacity (still capped by budget).
  const want = Math.min(budget, Math.round(8 + pts / 12));

  let hidden: number[];
  let activation: string;
  let lr: number;
  let steps = match.rules.aiTrainSteps;

  if (skill === 'easy') {
    hidden = [Math.max(2, Math.min(budget, 6))];
    activation = 'tanh';
    lr = 0.08;
    steps = Math.round(steps * 0.7);
  } else if (skill === 'medium') {
    const w = Math.max(4, Math.min(budget, want));
    hidden = [w];
    activation = 'tanh';
    lr = 0.12;
  } else {
    // hard: split the budget across two ReLU layers for sharp, capacity-efficient creases.
    const total = Math.min(budget, Math.max(12, want));
    const a = Math.min(budget - 2, Math.round(total * 0.6));
    const b = Math.max(2, Math.min(budget - a, total - a));
    hidden = [a, b];
    activation = 'relu';
    lr = 0.06;
    steps = Math.round(steps * 1.15);
  }

  return {
    config: { hidden, activation, lr, l2: 0, batchSize: 16, momentum: 0.9 },
    steps,
  };
}

/* ===========================================================================
   Hints for a human Trainer — leveled from hand-holding to nudges.
   ========================================================================= */

export type HintLevel = 0 | 1 | 2 | 3; // 0 = off, 1 = most help, 3 = least

export function neuronsUsed(config: StudioConfig): number {
  return config.hidden.reduce((s, w) => s + w, 0);
}

/** Where did the last attack land, on average? Helps point a player at the trouble. */
function lastAttackCentroid(match: VersusMatch): { x: number; y: number; n: number } {
  const last = match.sabotage.filter((s) => s.byAi).slice(-match.rules.pointsPerRound);
  if (last.length === 0) return { x: 0, y: 0, n: 0 };
  const x = last.reduce((s, p) => s + p.x, 0) / last.length;
  const y = last.reduce((s, p) => s + p.y, 0) / last.length;
  return { x, y, n: last.length };
}

function region(x: number, y: number): string {
  const v = y > 0.33 ? 'top' : y < -0.33 ? 'bottom' : 'middle';
  const h = x > 0.33 ? 'right' : x < -0.33 ? 'left' : 'centre';
  if (v === 'middle' && h === 'centre') return 'the centre';
  if (v === 'middle') return `the ${h}`;
  if (h === 'centre') return `the ${v}`;
  return `the ${v}-${h}`;
}

export function trainerHint(
  match: VersusMatch,
  config: StudioConfig,
  acc: number,
  level: HintLevel,
): string {
  if (level === 0) return '';
  const used = neuronsUsed(config);
  const budget = match.rules.neuronBudget;
  const free = budget - used;
  const { x, y, n } = lastAttackCentroid(match);
  const where = n > 0 ? region(x, y) : 'the new points';

  if (level >= 3) {
    // Minimal — just the situation.
    return `Accuracy ${(acc * 100).toFixed(0)}%, target ${(match.rules.winThreshold * 100).toFixed(0)}%. You’ve spent ${used}/${budget} neurons.`;
  }
  if (level === 2) {
    if (acc >= match.rules.winThreshold)
      return `You’re above target (${(acc * 100).toFixed(0)}%). Train a bit more to lock it in before the next attack.`;
    if (free >= 6) return `Below target near ${where}. You have ${free} unused neurons — spend some.`;
    return `Below target and nearly out of budget. Make your neurons count: retrain longer, or switch activation to ReLU for sharper boundaries.`;
  }
  // level 1 — most explicit.
  if (acc >= match.rules.winThreshold) {
    return `Holding at ${(acc * 100).toFixed(0)}%. Press Train for a few seconds so the boundary firms up around ${where}, then end your turn.`;
  }
  if (config.hidden.length === 0) {
    return `Your network has no hidden layer — it can only draw one straight line, and the saboteur’s points in ${where} need a bent boundary. Add a hidden layer of ~12, then Train.`;
  }
  if (free >= 8) {
    return `The saboteur clustered points in ${where} and your boundary can’t reach them. You have ${free} unused neurons (of ${budget}). Add a layer of ${Math.min(free, 12)} or widen an existing one, Reset, and Train.`;
  }
  if (free >= 2) {
    return `Close. The trouble is around ${where}. Add your last ${free} neurons, switch to ReLU for crisper creases, and train longer.`;
  }
  return `You’re at the neuron budget (${budget}). No more capacity to add — squeeze it out: ReLU activation, a higher learning rate, and more training steps to fit ${where}.`;
}
