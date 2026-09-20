import type { Matrix } from './matrix';
import type { Param } from './layers';

/**
 * An update rule. Called once per minibatch with every trainable parameter; reads
 * `p.grad` (refreshed by the last backward pass) and mutates `p.value.data` in place.
 *
 * Per-parameter state (momentum velocity, Adam moments) is keyed on the parameter's
 * stable `value` Matrix identity — its data buffer is updated in place, so the
 * reference never changes for the life of the layer.
 */
export interface Optimizer {
  step(params: Param[]): void;
}

export interface OptimizerConfig {
  lr: number;
  /** Only used by SGD. */
  momentum?: number;
  /** L2 weight decay, folded into the gradient (`g ← g + l2·w`) by every optimizer. */
  l2?: number;
}

/**
 * Stochastic gradient descent with (heavy-ball) momentum and L2 weight decay.
 *
 *   v ← momentum · v − lr · (grad + l2 · w)
 *   w ← w + v
 */
export class SGD implements Optimizer {
  private readonly velocity = new Map<Matrix, Float64Array>();

  constructor(
    private readonly lr: number,
    private readonly momentum: number = 0,
    private readonly l2: number = 0,
  ) {}

  step(params: Param[]): void {
    for (const p of params) {
      const w = p.value.data;
      const g = p.grad.data;
      let v = this.velocity.get(p.value);
      if (!v) {
        v = new Float64Array(w.length);
        this.velocity.set(p.value, v);
      }
      for (let i = 0; i < w.length; i++) {
        const grad = g[i]! + this.l2 * w[i]!;
        v[i] = this.momentum * v[i]! - this.lr * grad;
        w[i]! += v[i]!;
      }
    }
  }
}

/** Adam's standard exponential-decay rates and denominator guard. */
export const ADAM_BETA1 = 0.9;
export const ADAM_BETA2 = 0.999;
export const ADAM_EPS = 1e-8;

/**
 * Adam — SGD with a *per-weight* step size.
 *
 * It keeps two running averages of each weight's gradient: `m`, the mean (the same
 * "keep going the way you were going" idea as momentum), and `v`, the mean *square*
 * (how big this weight's gradients have been lately). The update divides by √v, so a
 * weight with consistently huge gradients takes small steps and a weight with tiny
 * gradients still makes progress — one learning rate suits every layer at once.
 *
 *   m ← β₁·m + (1−β₁)·g
 *   v ← β₂·v + (1−β₂)·g²
 *   m̂ = m / (1−β₁ᵗ),  v̂ = v / (1−β₂ᵗ)      ← bias correction (see below)
 *   w ← w − lr · m̂ / (√v̂ + ε)
 *
 * Both averages start at zero, which biases them toward zero for the first few steps
 * — hence dividing by `1−βᵗ`, which is small early (undoing the shrinkage) and → 1
 * later. `t` counts optimizer steps, so it is shared across all parameters.
 *
 * Because the update is normalised by the gradient's own scale, its magnitude is
 * roughly `lr` regardless of the loss surface — which is why Adam wants a much
 * smaller learning rate than SGD (~0.01–0.05 here, not ~0.2).
 */
export class Adam implements Optimizer {
  private readonly m = new Map<Matrix, Float64Array>();
  private readonly v = new Map<Matrix, Float64Array>();
  private t = 0;

  constructor(
    private readonly lr: number,
    private readonly l2: number = 0,
    private readonly beta1: number = ADAM_BETA1,
    private readonly beta2: number = ADAM_BETA2,
    private readonly eps: number = ADAM_EPS,
  ) {}

  step(params: Param[]): void {
    this.t++;
    // Bias-correction factors depend only on t, so compute them once per step.
    const corr1 = 1 - Math.pow(this.beta1, this.t);
    const corr2 = 1 - Math.pow(this.beta2, this.t);
    for (const p of params) {
      const w = p.value.data;
      const g = p.grad.data;
      let m = this.m.get(p.value);
      let v = this.v.get(p.value);
      if (!m || !v) {
        m = new Float64Array(w.length);
        v = new Float64Array(w.length);
        this.m.set(p.value, m);
        this.v.set(p.value, v);
      }
      for (let i = 0; i < w.length; i++) {
        const grad = g[i]! + this.l2 * w[i]!;
        m[i] = this.beta1 * m[i]! + (1 - this.beta1) * grad;
        v[i] = this.beta2 * v[i]! + (1 - this.beta2) * grad * grad;
        const mHat = m[i]! / corr1;
        const vHat = v[i]! / corr2;
        w[i]! -= (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}

export const OPTIMIZER_NAMES = ['sgd', 'adam'] as const;
export type OptimizerName = (typeof OPTIMIZER_NAMES)[number];

export function isOptimizerName(name: string): name is OptimizerName {
  return (OPTIMIZER_NAMES as readonly string[]).includes(name);
}

/** Build an optimizer by name. Throws on an unknown name (same contract as activations). */
export function createOptimizer(name: string, cfg: OptimizerConfig): Optimizer {
  switch (name.toLowerCase()) {
    case 'sgd':
      return new SGD(cfg.lr, cfg.momentum ?? 0, cfg.l2 ?? 0);
    case 'adam':
      return new Adam(cfg.lr, cfg.l2 ?? 0);
    default:
      throw new Error(`Unknown optimizer: ${name}`);
  }
}
