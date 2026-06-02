import type { Matrix } from './matrix';
import type { Param } from './layers';

/**
 * Stochastic gradient descent with (heavy-ball) momentum and L2 weight decay.
 *
 *   v ← momentum · v − lr · (grad + l2 · w)
 *   w ← w + v
 *
 * Velocity is stored per parameter, keyed on the parameter's stable `value` Matrix
 * identity (its data buffer is updated in place, so the reference never changes).
 */
export class SGD {
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
