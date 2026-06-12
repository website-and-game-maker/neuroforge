# NeuroForge — Design Spec

**Date:** 2026-06-01
**Status:** Approved (build it + regression mode)

## One-liner

A browser game/playground where you build, train, and reason about a small neural
network **implemented from scratch in TypeScript** (no ML libraries). You watch it
learn in real time, and beat escalating "capacity-reasoning" challenges.

## Why this design

The user wanted something fun, well-designed, and where the hard parts are *logical
and reasoning-based*. The reasoning core here is the **machine-learning engine**:
hand-derived forward/backward passes (backpropagation), verified for correctness with
**numerical gradient checking**. The gameplay is itself a reasoning exercise — each
challenge is a puzzle about *network capacity* (why XOR needs a hidden layer, why
two spirals need depth/width and nonlinearity, how learning rate / regularization
change the outcome).

## Goals / Success criteria

1. A neural network engine built from scratch, **provably correct** via gradient checks
   and a learning test (network learns XOR to ~100% train accuracy).
2. Live, smooth, non-blocking training visualization (decision boundary + curves).
3. Classification **and** regression challenges, plus a free sandbox.
4. Deployed and working on GitHub Pages (verified in a real browser).
5. Deliberate, spacious, polished UI. No cramped or awkward regions.
6. Comprehensive automated tests; zero known bugs.

## Non-goals (YAGNI)

Images/MNIST, convolutional/recurrent nets, WebGL/GPU, accounts/backend, multiplayer.

## Architecture

Zero runtime dependencies. TypeScript + Vite + Vitest. Vanilla DOM + Canvas.

```
src/
  engine/
    matrix.ts       Matrix over Float64Array: zeros, fromRows, matmul, addRowVec,
                    transpose, map, axpy, scale, ... (pure, fully unit-tested)
    rng.ts          Seeded PRNG (mulberry32) for deterministic data + init
    activations.ts  ReLU / Tanh / Sigmoid: value + derivative
    layers.ts       Dense layer (W, b, grads) forward/backward; Activation layer
    losses.ts       BCE (binary) + MSE (regression): loss + dL/dOutput
    network.ts      Sequential MLP: forward, backward, paramAndGrad views
    optimizer.ts    SGD with momentum + L2 weight decay
    trainer.ts      Steppable trainer: holds data, runs N minibatch steps per call,
                    tracks loss/accuracy history. (Easily movable to a Web Worker.)
  data/
    datasets.ts     Seeded generators: blobs, xor, circles, moons, spirals (classif.)
                    + regression generators: sine, step, noisy-linear (1-D -> 1-D)
    types.ts        Dataset, Sample, TaskKind ('classification' | 'regression')
  game/
    challenges.ts   Challenge definitions (dataset, target metric, constraints, hints)
    progress.ts     Scoring + which challenges are unlocked/completed
  state/
    persistence.ts  localStorage load/save of progress + best scores (safe + versioned)
  viz/
    boundary.ts     Decision-boundary heatmap renderer (classification)
    regression.ts   Regression curve renderer (predicted function vs samples)
    points.ts       Scatter renderer for samples
    chart.ts        Loss/accuracy sparkline renderer
    netdiagram.ts   Live network diagram: nodes per layer, edge thickness ~ |weight|
  ui/
    app.ts          Top-level app shell + mode routing (challenge / sandbox)
    controls.ts     Architecture + hyperparameter controls (sliders, selects, buttons)
    layout: 3 zones — Controls (left) · Arena (center) · Metrics (right)
  styles/
    theme.css, app.css  Dark "lab instrument" theme; spacing scale; responsive grid
  main.ts           Bootstraps the app
```

### Engine details (the reasoning-critical part)

- **Matrix**: row-major `Float64Array` with `rows`, `cols`. Operations return new
  matrices (or write into a provided out-buffer where it matters for speed). Kept tiny
  and total-function; this is the most heavily unit-tested module.
- **Dense layer** `y = x·W + b` where `x` is `[batch × in]`, `W` is `[in × out]`,
  `b` is `[out]`. Backward computes `dW = xᵀ·dY`, `db = sum_rows(dY)`,
  `dX = dY·Wᵀ`. Each of these is checked numerically.
- **Activation layers** apply elementwise `f` forward and `f'(pre) ⊙ dY` backward.
- **Losses**: BCE expects sigmoid output in (0,1); MSE for regression. Each returns
  scalar loss and `dL/dOutput` for the backward pass.
- **Network**: a list of layers; `forward(x)` caches activations, `backward(dY)`
  walks layers in reverse, accumulating param grads. `params()` exposes
  `{value, grad}` matrices for the optimizer.
- **Optimizer**: `theta -= lr * (grad + l2 * theta)` with momentum buffer.
- **Trainer**: shuffles indices with the seeded RNG, iterates minibatches, calls
  network forward/backward + optimizer step. `step(batches)` returns updated metrics
  so the UI can drive it incrementally from `requestAnimationFrame`.

### Correctness strategy

- **Gradient checking**: for each layer and the whole network, compare analytic grads
  to centered finite differences `(L(θ+ε) − L(θ−ε)) / 2ε`; assert relative error < 1e-5.
- **Learning tests**: trainer learns XOR to ~100% train accuracy within a step budget;
  regression fits a line/sine below an MSE threshold. Deterministic via fixed seeds.
- **Data determinism**: same seed → identical datasets.

## Gameplay

- **Challenge mode**: ordered list, increasing difficulty. Each challenge defines a
  dataset (seeded), a target metric (test accuracy for classification, max MSE for
  regression), optional constraints (e.g. neuron budget), and a written "why" hint.
  Reaching the target marks it complete and unlocks the next; best score persists.
- **Sandbox mode**: pick any dataset (including *draw-your-own* points for
  classification), full control of architecture + hyperparameters, train freely.
- **Controls**: hidden layers & widths, activation, learning rate, L2, batch size,
  momentum; buttons: Train/Pause, Step, Reset weights, New data.

## Visualization

- **Classification arena**: decision-boundary heatmap (model probability over the
  plane) with the dataset scattered on top; updates live during training.
- **Regression arena**: the model's predicted curve drawn over the sample scatter.
- **Metrics**: loss curve, accuracy/MSE readout, epoch/step counter, train vs test.
- **Network diagram**: layers of neurons; edge thickness/color encodes weight sign &
  magnitude; optional neuron-activation glow for the hovered/sampled input.

## Design language

Dark, calm, instrument-like. A defined spacing scale (8px base) with generous gaps;
3-zone responsive grid that collapses to stacked panels on narrow screens; large,
comfortable controls; smooth, non-janky animation; clear visual hierarchy and legible
typography. Explicit attention to avoiding cramped or awkward layouts.

## Testing

- Unit: matrix, rng, activations, layers (grad-checked), losses (grad-checked),
  network (grad-checked end-to-end), optimizer, datasets (determinism + shape),
  challenges/progress (scoring + unlock logic), persistence (round-trip + corruption
  safety).
- Integration: trainer learns XOR and a regression target to threshold.
- Manual: load the deployed GitHub Pages URL in Chrome, play a challenge, confirm
  live training, responsiveness, and spacing on desktop + narrow viewport.

## Deployment

Vite build to `dist`, published to GitHub Pages via GitHub Actions
(`configure-pages` → `upload-pages-artifact` → `deploy-pages`). CI runs typecheck +
tests before building, so only green builds deploy. `base` = `/neuroforge/`.
Final URL: `https://website-and-game-maker.github.io/neuroforge/`.
