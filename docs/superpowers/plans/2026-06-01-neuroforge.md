# NeuroForge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A browser game/playground where you build and train a from-scratch neural network to beat classification and regression challenges, deployed to GitHub Pages.

**Architecture:** A dependency-free ML engine (matrix → layers → network → optimizer → trainer) with hand-derived backprop verified by numerical gradient checking; seeded dataset generators; a challenge/progress layer; Canvas visualizations; a vanilla-DOM 3-zone UI. Training is *steppable* and driven by `requestAnimationFrame` so the UI never blocks.

**Tech Stack:** TypeScript, Vite, Vitest. Vanilla DOM + Canvas. No runtime deps.

---

## Shared Contracts (types used across tasks — keep names consistent)

```ts
// engine/matrix.ts
class Matrix {
  readonly rows: number; readonly cols: number; readonly data: Float64Array; // row-major
  constructor(rows: number, cols: number, data?: Float64Array);
  static zeros(rows: number, cols: number): Matrix;
  static fromRows(rows: number[][]): Matrix;
  get(r: number, c: number): number;
  set(r: number, c: number, v: number): void;
  toRows(): number[][];
  clone(): Matrix;
}
function matmul(a: Matrix, b: Matrix): Matrix;        // [m×k]·[k×n] -> [m×n]
function transpose(a: Matrix): Matrix;
function addRowVector(a: Matrix, row: Matrix): Matrix; // row is [1×cols]
function sumRows(a: Matrix): Matrix;                   // -> [1×cols]
function mapMatrix(a: Matrix, fn: (x: number) => number): Matrix;
function hadamard(a: Matrix, b: Matrix): Matrix;
function scale(a: Matrix, s: number): Matrix;

// engine/rng.ts
class Rng {
  constructor(seed: number);
  next(): number;          // uniform [0,1)
  gaussian(): number;      // standard normal (Box–Muller)
  range(lo: number, hi: number): number;
  int(maxExclusive: number): number;
  shuffleInPlace<T>(arr: T[]): void;
}

// engine/activations.ts
interface Activation { name: string; f(x: number): number; df(x: number): number; }
const ReLU: Activation; const Tanh: Activation; const Sigmoid: Activation; const Identity: Activation;
function activationByName(name: string): Activation;

// engine/layers.ts
interface Param { value: Matrix; grad: Matrix; }   // value identity is STABLE across steps
interface Layer { forward(x: Matrix): Matrix; backward(dY: Matrix): Matrix; params(): Param[]; }
class Dense implements Layer { constructor(inDim: number, outDim: number, rng: Rng); }
class ActivationLayer implements Layer { constructor(act: Activation); }

// engine/losses.ts
interface Loss { forward(output: Matrix, target: Matrix): number; backward(output: Matrix, target: Matrix): Matrix; }
const BCE: Loss;   // output in (0,1), target in {0,1}
const MSE: Loss;

// engine/network.ts
class Network {
  constructor(layers: Layer[]);
  forward(x: Matrix): Matrix;
  backward(dOut: Matrix): Matrix;
  params(): Param[];
}
function mlp(inDim: number, hidden: number[], outDim: number,
             act: Activation, outAct: Activation, rng: Rng): Network;

// engine/optimizer.ts
class SGD { constructor(lr: number, momentum: number, l2: number); step(params: Param[]): void; }

// engine/trainer.ts
interface TrainConfig { lr: number; momentum: number; l2: number; batchSize: number; }
class Trainer {
  constructor(net: Network, loss: Loss, X: Matrix, Y: Matrix, cfg: TrainConfig, rng: Rng);
  step(numBatches?: number): void;       // runs minibatch SGD updates; appends to history
  predict(X: Matrix): Matrix;
  readonly lossHistory: number[];
  readonly stepCount: number;
}
function accuracy(pred: Matrix, target: Matrix): number;   // binary, threshold 0.5
function mseMetric(pred: Matrix, target: Matrix): number;

// data/datasets.ts
type TaskKind = 'classification' | 'regression';
interface Dataset { kind: TaskKind; name: string; inDim: number; X: Matrix; Y: Matrix; }
// classification (inDim=2, Y=[n×1] in {0,1}):
function gaussianBlobs(n: number, seed: number): Dataset;
function xorData(n: number, seed: number): Dataset;
function circlesData(n: number, seed: number): Dataset;
function moonsData(n: number, seed: number): Dataset;
function spiralsData(n: number, seed: number): Dataset;
// regression (inDim=1, Y=[n×1]):
function regLinear(n: number, seed: number): Dataset;
function regSine(n: number, seed: number): Dataset;

// game/challenges.ts
interface Target { kind: 'accuracy'; min: number } | { kind: 'mse'; max: number }
interface Starter { hidden: number[]; activation: string; lr: number; l2: number; batchSize: number; momentum: number }
interface Challenge {
  id: string; title: string; task: TaskKind;
  makeData: (seed: number) => Dataset; trainSeed: number; testSeed: number;
  target: Target; why: string; starter: Starter; maxNeurons?: number;
}
const CHALLENGES: Challenge[];

// game/progress.ts
interface Progress { version: number; completed: Record<string, { bestScore: number }> }
function emptyProgress(): Progress;
function markComplete(p: Progress, id: string, score: number): Progress;
function isUnlocked(p: Progress, index: number): boolean;   // sequential unlock

// state/persistence.ts
function loadProgress(): Progress;   // safe against corrupt/missing localStorage
function saveProgress(p: Progress): void;
```

**Gradient-check helper** (in `tests/helpers/gradcheck.ts`): centered finite differences with relative-error assertion `< 1e-5`.

---

## Phase 1 — Engine (reasoning-critical; strict TDD + gradient checks)

### Task 1: Matrix
**Files:** Create `src/engine/matrix.ts`; Test `tests/engine/matrix.test.ts`
- [ ] Test: `Matrix.fromRows([[1,2],[3,4]])` has rows=2, cols=2, `get(1,0)===3`; `toRows()` round-trips.
- [ ] Test: `matmul([[1,2,3]],[[4],[5],[6]])` → `[[32]]`; shape `[2×3]·[3×2]`→`[2×2]` correct values.
- [ ] Test: `transpose` of `[[1,2,3],[4,5,6]]` → `[[1,4],[2,5],[3,6]]`.
- [ ] Test: `addRowVector([[1,2],[3,4]], [[10,20]])` → `[[11,22],[13,24]]`.
- [ ] Test: `sumRows([[1,2],[3,4]])` → `[[4,6]]`.
- [ ] Test: `hadamard`, `scale`, `mapMatrix` elementwise correctness.
- [ ] Test: `matmul` throws on shape mismatch.
- [ ] Implement; run tests to green. Commit.

### Task 2: Rng
**Files:** Create `src/engine/rng.ts`; Test `tests/engine/rng.test.ts`
- [ ] Test: same seed → identical `next()` sequence; different seed → different.
- [ ] Test: `next()` always in [0,1); over 10k samples mean ≈ 0.5 (±0.03).
- [ ] Test: `gaussian()` over 20k samples mean ≈ 0 (±0.05), std ≈ 1 (±0.05).
- [ ] Test: `shuffleInPlace` is a permutation (same multiset) and deterministic per seed.
- [ ] Implement (mulberry32 + Box–Muller). Commit.

### Task 3: Activations
**Files:** Create `src/engine/activations.ts`; Test `tests/engine/activations.test.ts`
- [ ] Test: ReLU f/df at -2,0,3; Tanh f(0)=0 df(0)=1; Sigmoid f(0)=0.5 df(0)=0.25; Identity.
- [ ] Test: numeric derivative of each `f` matches `df` (rel err < 1e-6) at several points.
- [ ] Test: `activationByName('relu')===ReLU`, throws on unknown.
- [ ] Implement. Commit.

### Task 4: Dense & ActivationLayer (+ gradient check)
**Files:** Create `src/engine/layers.ts`; Test `tests/engine/layers.test.ts`, `tests/helpers/gradcheck.ts`
- [ ] Write `gradcheck` helper: given a function θ→scalar and analytic grad, assert centered-FD rel err < 1e-5.
- [ ] Test: `Dense.forward` matches manual `x·W + b` on a fixed small W,b.
- [ ] Test: Dense `backward` — gradient-check `dW`, `db`, and `dX` against FD through a scalar loss `sum(forward(x))` and `sum(forward(x)*randMask)`.
- [ ] Test: ActivationLayer (Tanh) backward gradient-checked.
- [ ] Test: `params()` returns stable `value` identities across calls; grads update after backward.
- [ ] Implement. Commit.

### Task 5: Losses (+ gradient check)
**Files:** Create `src/engine/losses.ts`; Test `tests/engine/losses.test.ts`
- [ ] Test: MSE forward value on a known pair; backward gradient-checked vs FD.
- [ ] Test: BCE forward value on a known pair (with clamping); backward gradient-checked vs FD at non-saturated points.
- [ ] Implement (clamp prob to [1e-7, 1-1e-7]). Commit.

### Task 6: Network (+ end-to-end gradient check)
**Files:** Create `src/engine/network.ts`; Test `tests/engine/network.test.ts`
- [ ] Test: `mlp(2,[4],1,Tanh,Sigmoid,rng)` forward output shape `[n×1]`, values in (0,1).
- [ ] Test: end-to-end gradient check — for a fixed tiny net + BCE, every parameter's analytic grad matches FD (rel err < 1e-5).
- [ ] Test: `params()` count equals sum of Dense params.
- [ ] Implement. Commit.

### Task 7: Optimizer
**Files:** Create `src/engine/optimizer.ts`; Test `tests/engine/optimizer.test.ts`
- [ ] Test: one SGD step with momentum=0,l2=0 moves params by `-lr*grad` exactly.
- [ ] Test: momentum accumulates velocity across two steps as expected.
- [ ] Test: l2 adds `lr*l2*w` decay term.
- [ ] Test: velocity is keyed to stable `value` identity (two steps on same params reuse velocity).
- [ ] Implement. Commit.

### Task 8: Trainer + metrics + learning tests
**Files:** Create `src/engine/trainer.ts`; Test `tests/engine/trainer.test.ts`
- [ ] Test: `accuracy` and `mseMetric` on hand-built predictions.
- [ ] Test: `step(k)` increments `stepCount` and appends `k` (or #batches) loss entries; deterministic per seed.
- [ ] Test (learning, classification): Trainer with `mlp(2,[8,8],1,Tanh,Sigmoid)` + BCE learns XOR (seed-fixed) to ≥ 95% train accuracy within a step budget.
- [ ] Test (learning, regression): Trainer with `mlp(1,[16],1,Tanh,Identity)` + MSE fits `regSine` below MSE threshold within a budget.
- [ ] Implement. Commit.

## Phase 2 — Data

### Task 9: Dataset generators
**Files:** Create `src/data/datasets.ts`; Test `tests/data/datasets.test.ts`
- [ ] Test: every generator returns correct `kind`, `inDim`, matrix shapes `X:[n×inDim]`, `Y:[n×1]`.
- [ ] Test: determinism — same seed → identical X,Y; classification labels ⊆ {0,1}.
- [ ] Test: `xorData` labels equal `(x>0) XOR (y>0)` for clean (low-noise) points.
- [ ] Test: all classification X within plotting range (|v| ≤ ~1.5).
- [ ] Implement blobs, xor, circles, moons, spirals, regLinear, regSine. Commit.

## Phase 3 — Game logic

### Task 10: Challenges + progress
**Files:** Create `src/game/challenges.ts`, `src/game/progress.ts`; Test `tests/game/progress.test.ts`, `tests/game/challenges.test.ts`
- [ ] Test (progress): `emptyProgress` empty; `markComplete` records max bestScore (doesn't lower it); `isUnlocked(p,0)` true, `isUnlocked(p,i>0)` true iff previous completed.
- [ ] Test (challenges): `CHALLENGES` non-empty, unique ids, ordered easy→hard; each has a valid `starter` and `target`; each `makeData(seed)` matches its declared `task`.
- [ ] Implement an ordered challenge list (blobs→xor→circles→moons→spirals→regLinear→regSine) with reasoning `why` text. Commit.

### Task 11: Persistence
**Files:** Create `src/state/persistence.ts`; Test `tests/state/persistence.test.ts` (`// @vitest-environment jsdom`)
- [ ] Test: `saveProgress`→`loadProgress` round-trips.
- [ ] Test: missing key → `emptyProgress`; corrupt JSON → `emptyProgress` (no throw); wrong version → `emptyProgress`.
- [ ] Implement (try/catch, version guard). Commit.

## Phase 4 — Visualization (pure transforms unit-tested; rendering verified in browser)

### Task 12: Viz transforms + renderers
**Files:** Create `src/viz/coords.ts`, `src/viz/colors.ts`, `src/viz/points.ts`, `src/viz/boundary.ts`, `src/viz/regression.ts`, `src/viz/chart.ts`, `src/viz/netdiagram.ts`; Test `tests/viz/coords.test.ts`, `tests/viz/colors.test.ts`
- [ ] Test (coords): `worldToScreen`/`screenToWorld` round-trip; domain [-1,1] maps to [pad, size-pad].
- [ ] Test (colors): probability 0→class-A color, 1→class-B color, 0.5→neutral; weight sign→diverging color; outputs valid rgb strings.
- [ ] Implement renderers as `(ctx, ...data, layout)` functions using the tested transforms (boundary samples a grid of model probabilities; regression draws predicted curve; chart draws normalized loss sparkline; netdiagram draws nodes+edges with thickness∝|weight|). Commit.

## Phase 5 — UI (3-zone layout, frontend-design, spacious)

### Task 13: Styles + app shell
**Files:** Create `src/styles/theme.css`, `src/styles/app.css`; rewrite `src/main.ts`; Create `src/ui/app.ts`, `src/ui/dom.ts` (tiny `el()` helper); Test `tests/ui/dom.test.ts` (`// @vitest-environment jsdom`)
- [ ] Test (dom helper): `el('div',{class:'x'},[el('span',{},['hi'])])` builds the right tree/text.
- [ ] Implement theme (spacing scale, color tokens, typography), 3-zone responsive grid, app shell mounting controls/arena/metrics. Commit.

### Task 14: Controls + wiring + run loop
**Files:** Create `src/ui/controls.ts`, `src/ui/arena.ts`, `src/ui/metrics.ts`; modify `src/ui/app.ts`
- [ ] Implement controls (hidden layers, activation, lr, l2, batch, momentum; Train/Pause, Step, Reset, New data; challenge selector + sandbox/draw mode).
- [ ] Implement the rAF run loop: each frame `trainer.step(k)` then redraw arena + metrics; pause/resume; live decision boundary / regression curve.
- [ ] Implement challenge pass detection (eval on test split) → mark complete + persist + unlock next; show the `why` hint.
- [ ] Implement draw-your-own dataset (click canvas to add class-A/B points) in sandbox.
- [ ] Manual: `npm run dev`, verify train/visualize/responsiveness. Commit.

## Phase 6 — Verify & ship

### Task 15: Full verification
- [ ] `npm run typecheck` clean; `npm run test:run` all green; `npm run build` succeeds.
- [ ] Multi-dimensional code review (Workflow): correctness, simplification, design/spacing. Apply fixes.
- [ ] Commit + push; wait for Pages deploy.
- [ ] In-browser (Chrome) test of the LIVE GitHub Pages URL: load, play a classification challenge to completion, a regression challenge, sandbox draw mode; check spacing on desktop + narrow viewport; check console for errors.
- [ ] Fix anything found; redeploy; re-verify. Report final URL.
```
