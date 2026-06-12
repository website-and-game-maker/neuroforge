# NeuroForge

**Train a neural network in your browser — from scratch.**

NeuroForge is a small game/playground for building and training neural networks. The
entire ML engine (matrices, dense layers, activations, losses, backpropagation, and an
SGD-with-momentum optimizer) is implemented by hand in TypeScript — **no ML
libraries** — and its gradients are verified with numerical gradient checking.

Watch a network learn in real time: the decision boundary morphs as it trains, loss
drops, and a live network diagram shows weights strengthening. Beat escalating
challenges that are really puzzles about *network capacity* (why does XOR need a hidden
layer? why do two spirals need depth?), or play freely in the sandbox.

🔗 **Live site:** https://website-and-game-maker.github.io/neuroforge/

## Develop

```bash
npm install
npm run dev        # local dev server
npm test           # watch-mode tests
npm run test:run   # one-shot tests
npm run typecheck  # tsc --noEmit
npm run build      # typecheck + production build to dist/
```

## Deploy

Pushing to `main` triggers `.github/workflows/deploy.yml`, which typechecks, tests,
builds, and publishes `dist/` to GitHub Pages.

## License

MIT
