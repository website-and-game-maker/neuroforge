import { Matrix } from '../engine/matrix';
import { Rng } from '../engine/rng';
import { accuracy } from '../engine/trainer';
import type { Dataset } from '../data/datasets';
import type { StudioConfig } from '../ui/studio';

export type Role = 'trainer' | 'saboteur';
export type Difficulty = 'easy' | 'medium' | 'hard';
export type HintLevel = 0 | 1 | 2 | 3; // 0 = off, 1 = most help, 3 = least

/** Predict P(class B / orange) for a set of [x,y] rows. */
export type ProbFn = (X: Matrix) => Matrix;

/** A point placed on the arena during a match. `atMs` is match-time of placement. */
export interface PlacedPoint {
  x: number;
  y: number;
  label: number;
  byAi: boolean;
  atMs: number;
}

/**
 * Live-match rules. There are no base patterns and no turns: the saboteur paints the
 * dataset (any class, anywhere, anytime) from a fixed point budget while the trainer
 * trains continuously under a neuron budget. When the clock runs out, the trainer wins
 * iff accuracy over every placed point is at or above the threshold.
 */
export interface LiveRules {
  durationMs: number;
  pointBudget: number;
  neuronBudget: number;
  winThreshold: number;
  /** Fewer than this many points at the end is a forfeit — trainer wins. */
  minPoints: number;
}

export const LIVE_RULES: LiveRules = {
  durationMs: 120_000,
  pointBudget: 60,
  neuronBudget: 18,
  winThreshold: 0.85,
  minPoints: 12,
};

/** AI pacing (ms between actions) per difficulty. */
export const AI_DROP_INTERVAL: Record<Difficulty, number> = {
  easy: 3000,
  medium: 2200,
  hard: 1500,
};
export const AI_TUNE_INTERVAL = 6000;

/* ===========================================================================
   Match state — a clock, a point budget, and a verdict. Time is injected via
   advance(dt) so the game logic stays deterministic and testable.
   ========================================================================= */

export class LiveMatch {
  elapsedMs = 0;
  readonly points: PlacedPoint[] = [];

  constructor(readonly rules: LiveRules) {}

  get done(): boolean {
    return this.elapsedMs >= this.rules.durationMs;
  }

  get remainingMs(): number {
    return Math.max(0, this.rules.durationMs - this.elapsedMs);
  }

  get budgetLeft(): number {
    return Math.max(0, this.rules.pointBudget - this.points.length);
  }

  /** Advance the match clock. dt is clamped so a background tab can't teleport time. */
  advance(dtMs: number): void {
    if (this.done) return;
    const dt = Math.min(1000, Math.max(0, dtMs));
    this.elapsedMs = Math.min(this.rules.durationMs, this.elapsedMs + dt);
  }

  /** Place a point if the match is live and budget remains. Returns success. */
  addPoint(x: number, y: number, label: number, byAi: boolean): boolean {
    if (this.done || this.budgetLeft <= 0) return false;
    this.points.push({ x, y, label, byAi, atMs: this.elapsedMs });
    return true;
  }

  /** All placed points as a Dataset (may be empty — callers handle 0 rows). */
  dataset(): Dataset {
    const n = this.points.length;
    const X = new Float64Array(n * 2);
    const Y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = this.points[i]!;
      X[i * 2] = p.x;
      X[i * 2 + 1] = p.y;
      Y[i] = p.label;
    }
    return {
      kind: 'classification',
      name: 'Versus',
      inDim: 2,
      X: new Matrix(n, 2, X),
      Y: new Matrix(n, 1, Y),
    };
  }

  /** Accuracy over every placed point (1 when nothing has been placed yet). */
  score(predict: ProbFn): number {
    const ds = this.dataset();
    if (ds.X.rows === 0) return 1;
    return accuracy(predict(ds.X), ds.Y);
  }

  /** Final verdict, only once the clock has run out. */
  winner(predict: ProbFn): Role | null {
    if (!this.done) return null;
    if (this.points.length < this.rules.minPoints) return 'trainer';
    return this.score(predict) >= this.rules.winThreshold ? 'trainer' : 'saboteur';
  }
}

/* ===========================================================================
   AI Saboteur — one point at a time, live.

   Two regimes: while the board is sparse it SEEDS a structurally hard pattern
   (the saboteur is the data generator now — there are no base patterns); once
   the trainer's model has shape, it EXPLOITS confidently-wrong regions.
   ========================================================================= */

const SEED_UNTIL = 14;

function nearestDist(x: number, y: number, pts: ReadonlyArray<{ x: number; y: number }>): number {
  let best = Infinity;
  for (const p of pts) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < best) best = d;
  }
  return best;
}

/** Label a position per the difficulty's seed pattern. */
function seedLabel(x: number, y: number, skill: Difficulty, rng: Rng): number {
  if (skill === 'easy') return rng.next() < 0.5 ? 0 : 1; // noise — easy to fit or ignore
  if (skill === 'medium') {
    // Alternating rings: closed-curve boundaries, needs real capacity.
    const r = Math.hypot(x, y);
    return Math.floor(r / 0.34) % 2;
  }
  // hard: fine checkerboard — many disjoint regions, brutal for a small budget.
  const cell = 0.5;
  return (Math.floor((x + 1) / cell) + Math.floor((y + 1) / cell)) % 2;
}

/**
 * Choose the saboteur's next point. `predict` may be null while the trainer has no
 * trained model yet (then we always seed). Returns null only if no position is viable.
 */
export function pickSabotagePoint(
  predict: ProbFn | null,
  points: ReadonlyArray<PlacedPoint>,
  skill: Difficulty,
  rng: Rng,
): { x: number; y: number; label: number } | null {
  const seeding = !predict || points.length < SEED_UNTIL;
  // Hard keeps reinforcing its pattern part of the time even late — structure is what
  // starves an 18-neuron budget, not isolated potshots.
  const reinforce = skill === 'hard' && rng.next() < 0.35;

  if (seeding || reinforce) {
    for (let tries = 0; tries < 24; tries++) {
      const x = rng.range(-0.95, 0.95);
      const y = rng.range(-0.95, 0.95);
      if (nearestDist(x, y, points) < 0.06) continue; // don't stack on an existing point
      return { x, y, label: seedLabel(x, y, skill, rng) };
    }
    return null;
  }

  // Exploit: probe a jittered grid, hit where the model is confidently wrong-able.
  const G = 18;
  const candidates: Array<{ x: number; y: number }> = [];
  for (let i = 0; i < G; i++) {
    for (let j = 0; j < G; j++) {
      const x = -1 + (2 * (i + 0.5)) / G + (rng.next() - 0.5) * (2 / G);
      const y = -1 + (2 * (j + 0.5)) / G + (rng.next() - 0.5) * (2 / G);
      candidates.push({ x, y });
    }
  }
  const probs = predict(Matrix.fromRows(candidates.map((c) => [c.x, c.y])));

  let best: { x: number; y: number; label: number } | null = null;
  let bestScore = -Infinity;
  for (let k = 0; k < candidates.length; k++) {
    const c = candidates[k]!;
    if (nearestDist(c.x, c.y, points) < 0.06) continue;
    const p = probs.data[k]!;
    const label = p >= 0.5 ? 0 : 1; // opposite of the model's current guess
    const confidence = Math.abs(p - 0.5) * 2;
    const sameDist = nearestDist(c.x, c.y, points.filter((q) => q.label === label));
    let s: number;
    if (skill === 'easy') s = rng.next() * (confidence > 0.15 ? 1 : 0.1);
    else if (skill === 'medium') s = confidence;
    else s = confidence * (0.5 + Math.min(1, sameDist)); // isolated deep strikes
    if (s > bestScore) {
      bestScore = s;
      best = { x: c.x, y: c.y, label };
    }
  }
  return best;
}

/* ===========================================================================
   AI Trainer — periodic live retuning within the neuron budget.
   ========================================================================= */

export function neuronsUsed(config: StudioConfig): number {
  return config.hidden.reduce((s, w) => s + w, 0);
}

/**
 * Pick the architecture an automated trainer should be running right now, given how
 * much data is on the board. Skill scales how well the budget is spent. The caller
 * applies it only when it differs (warm-keeping trained weights otherwise).
 */
export function liveTrainerPlan(
  pointCount: number,
  skill: Difficulty,
  budget: number,
): StudioConfig {
  if (skill === 'easy') {
    return { hidden: [Math.min(budget, 6)], activation: 'tanh', lr: 0.08, l2: 0, batchSize: 16, momentum: 0.9 };
  }
  if (skill === 'medium') {
    const w = Math.min(budget, 8 + Math.floor(pointCount / 12));
    return { hidden: [w], activation: 'tanh', lr: 0.12, l2: 0, batchSize: 16, momentum: 0.9 };
  }
  // hard: spend the budget across two ReLU layers, growing with the data.
  const total = Math.min(budget, 12 + Math.floor(pointCount / 10));
  const a = Math.max(2, Math.round(total * 0.6));
  const b = Math.max(2, total - a);
  return { hidden: [a, b], activation: 'relu', lr: 0.06, l2: 0, batchSize: 16, momentum: 0.9 };
}

/* ===========================================================================
   Live hints for a human Trainer — leveled from hand-holding to bare stats.
   ========================================================================= */

function region(x: number, y: number): string {
  const v = y > 0.33 ? 'top' : y < -0.33 ? 'bottom' : 'middle';
  const h = x > 0.33 ? 'right' : x < -0.33 ? 'left' : 'centre';
  if (v === 'middle' && h === 'centre') return 'the centre';
  if (v === 'middle') return `the ${h}`;
  if (h === 'centre') return `the ${v}`;
  return `the ${v}-${h}`;
}

/** Where the saboteur has been hitting lately (centroid of recent drops). */
function recentAttackRegion(match: LiveMatch): string | null {
  const recent = match.points.slice(-8);
  if (recent.length < 3) return null;
  const x = recent.reduce((s, p) => s + p.x, 0) / recent.length;
  const y = recent.reduce((s, p) => s + p.y, 0) / recent.length;
  return region(x, y);
}

export function liveTrainerHint(
  match: LiveMatch,
  config: StudioConfig,
  acc: number,
  level: HintLevel,
): string {
  if (level === 0) return '';
  const used = neuronsUsed(config);
  const budget = match.rules.neuronBudget;
  const free = budget - used;
  const secs = Math.ceil(match.remainingMs / 1000);
  const target = match.rules.winThreshold;
  const where = recentAttackRegion(match);

  if (level >= 3) {
    return `${secs}s left · accuracy ${(acc * 100).toFixed(0)}% (need ${(target * 100).toFixed(0)}%) · ${used}/${budget} neurons.`;
  }
  if (level === 2) {
    if (acc >= target) return `Holding above target with ${secs}s left. Keep training — late drops${where ? ` near ${where}` : ''} can still flip it.`;
    if (free >= 4) return `Below target. You still have ${free} unused neurons — spend them${where ? `; the pressure is in ${where}` : ''}.`;
    return `Below target and out of spare neurons. Squeeze the ones you have: try ReLU, nudge the learning rate, keep training.`;
  }
  // level 1 — most explicit
  if (acc >= target) {
    return `You're winning (${(acc * 100).toFixed(0)}% ≥ ${(target * 100).toFixed(0)}%) with ${secs}s on the clock. Leave training running so the boundary keeps absorbing new points${where ? ` around ${where}` : ''}.`;
  }
  if (config.hidden.length === 0) {
    return `No hidden layer = one straight line, and the saboteur is painting shapes a line can't split. Add a hidden layer (try ${Math.min(12, Math.max(4, free))}) and keep Train running.`;
  }
  if (free >= 4) {
    return `The saboteur is beating you${where ? ` around ${where}` : ''}. You've only spent ${used} of ${budget} neurons — widen a layer or add one (${free} free), then let it retrain.`;
  }
  return `You're at the ${budget}-neuron cap and still under target. Switch to ReLU for sharper creases, raise the learning rate a touch, and don't stop training — ${secs}s left.`;
}
