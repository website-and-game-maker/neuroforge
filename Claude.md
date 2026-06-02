# NeuroForge — Project Notes for Claude

## What this is

A browser game/playground for building and training a small neural network
**implemented from scratch** (no ML libraries). Pure client-side, deployed to GitHub
Pages. The reasoning-critical core is the ML engine, whose gradients are verified with
numerical gradient checking.

## Stack

- TypeScript, Vite (build/dev), Vitest (tests). Zero runtime dependencies.
- Vanilla DOM + Canvas for UI/visualization.
- Deployed to GitHub Pages via `.github/workflows/deploy.yml`.

## Layout

See `docs/superpowers/specs/2026-06-01-neuroforge-design.md` for the full design.
- `src/engine/` — matrix, rng, activations, layers, losses, network, optimizer, trainer
- `src/data/` — seeded dataset generators (classification + regression)
- `src/viz/` — Canvas renderers (boundary, regression curve, points, charts, net diagram)
- `src/game/` — challenge definitions + progress/scoring
- `src/state/` — localStorage persistence
- `src/ui/` — app shell + controls (3-zone layout)

## Commands

```bash
npm install
npm run dev
npm run test:run
npm run typecheck
npm run build
```

## Conventions / constraints

- **No ML or math libraries** — the engine is hand-written on purpose.
- Engine functions are pure and total where possible; the engine has no DOM imports.
- Every layer/loss gradient must have a numerical gradient-check test.
- Training is *steppable* (driven by requestAnimationFrame) so the UI never blocks.
- Datasets and weight init are seeded for determinism (tests + reproducible play).
- `base` in `vite.config.ts` is `/neuroforge/` to match the Pages repo path.

## How to continue

1. `npm install`
2. `npm run test:run` — confirm green.
3. `npm run dev` — open the printed URL.
4. Implement per the spec; keep tests green; commit in logical chunks.
5. Push to `main` to deploy; verify https://pycoder42.github.io/neuroforge/ in a browser.
