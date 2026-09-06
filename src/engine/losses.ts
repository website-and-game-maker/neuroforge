import { Matrix } from './matrix';

/**
 * A loss couples a scalar objective with its gradient w.r.t. the network output.
 * Both are averaged over all output elements so the gradient magnitude is independent
 * of batch size.
 */
export interface Loss {
  forward(output: Matrix, target: Matrix): number;
  backward(output: Matrix, target: Matrix): Matrix;
}

function requireSameShape(a: Matrix, b: Matrix): void {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(
      `loss: output/target shape mismatch ${a.rows}x${a.cols} vs ${b.rows}x${b.cols}`,
    );
  }
}

/** Mean squared error. */
export const MSE: Loss = {
  forward(output, target) {
    requireSameShape(output, target);
    const n = output.data.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const d = output.data[i]! - target.data[i]!;
      sum += d * d;
    }
    return sum / n;
  },
  backward(output, target) {
    requireSameShape(output, target);
    const n = output.data.length;
    const grad = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      grad[i] = (2 * (output.data[i]! - target.data[i]!)) / n;
    }
    return new Matrix(output.rows, output.cols, grad);
  },
};

const EPS = 1e-7;

/** Binary cross-entropy. Expects probabilities in (0,1) and targets in {0,1}. */
export const BCE: Loss = {
  forward(output, target) {
    requireSameShape(output, target);
    const n = output.data.length;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const p = Math.min(1 - EPS, Math.max(EPS, output.data[i]!));
      const t = target.data[i]!;
      sum += -(t * Math.log(p) + (1 - t) * Math.log(1 - p));
    }
    return sum / n;
  },
  backward(output, target) {
    requireSameShape(output, target);
    const n = output.data.length;
    const grad = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const raw = output.data[i]!;
      // forward() clamps p into [EPS, 1-EPS], so it is locally flat (derivative 0)
      // outside that window — match that here instead of blowing up at raw=0/1.
      if (raw <= EPS || raw >= 1 - EPS) continue;
      const t = target.data[i]!;
      grad[i] = (raw - t) / (raw * (1 - raw)) / n;
    }
    return new Matrix(output.rows, output.cols, grad);
  },
};
