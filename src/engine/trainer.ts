import { Matrix } from './matrix';
import type { Network } from './network';
import type { Loss } from './losses';
import { SGD } from './optimizer';
import type { Rng } from './rng';

export interface TrainConfig {
  lr: number;
  momentum: number;
  l2: number;
  batchSize: number;
}

/** Gather a subset of rows from `m` into a new matrix (used to build minibatches). */
function gatherRows(m: Matrix, idx: number[]): Matrix {
  const out = new Float64Array(idx.length * m.cols);
  for (let r = 0; r < idx.length; r++) {
    const src = idx[r]! * m.cols;
    out.set(m.data.subarray(src, src + m.cols), r * m.cols);
  }
  return new Matrix(idx.length, m.cols, out);
}

/**
 * Drives minibatch SGD over a fixed dataset. Training is *steppable*: the UI calls
 * `step(k)` once per animation frame so the page never blocks. An epoch's worth of
 * shuffled indices is consumed batch by batch, reshuffling when exhausted.
 */
/** Cap the retained loss history so a long training run can't grow memory without bound
 *  (the chart only ever needs a recent window). Trimming is deterministic, so identical
 *  seeds still produce identical histories. */
const MAX_LOSS_HISTORY = 5000;
const LOSS_HISTORY_KEEP = 4000;

export class Trainer {
  readonly lossHistory: number[] = [];
  stepCount = 0;

  private readonly opt: SGD;
  private readonly batchSize: number;
  private readonly indices: number[];
  private cursor = 0;

  constructor(
    private readonly net: Network,
    private readonly loss: Loss,
    private readonly X: Matrix,
    private readonly Y: Matrix,
    cfg: TrainConfig,
    private readonly rng: Rng,
  ) {
    this.opt = new SGD(cfg.lr, cfg.momentum, cfg.l2);
    this.batchSize = Math.max(1, Math.min(cfg.batchSize, X.rows));
    this.indices = Array.from({ length: X.rows }, (_, i) => i);
    this.rng.shuffleInPlace(this.indices);
  }

  step(numBatches = 1): void {
    for (let b = 0; b < numBatches; b++) {
      const idx = this.nextBatchIndices();
      const bx = gatherRows(this.X, idx);
      const by = gatherRows(this.Y, idx);
      const out = this.net.forward(bx);
      const l = this.loss.forward(out, by);
      this.net.backward(this.loss.backward(out, by));
      this.opt.step(this.net.params());
      this.lossHistory.push(l);
      if (this.lossHistory.length > MAX_LOSS_HISTORY) {
        this.lossHistory.splice(0, this.lossHistory.length - LOSS_HISTORY_KEEP);
      }
      this.stepCount++;
    }
  }

  predict(X: Matrix): Matrix {
    return this.net.forward(X);
  }

  private nextBatchIndices(): number[] {
    if (this.cursor + this.batchSize > this.indices.length) {
      this.rng.shuffleInPlace(this.indices);
      this.cursor = 0;
    }
    const slice = this.indices.slice(this.cursor, this.cursor + this.batchSize);
    this.cursor += this.batchSize;
    return slice;
  }
}

function requireSameShape(a: Matrix, b: Matrix, op: string): void {
  if (a.rows !== b.rows || a.cols !== b.cols) {
    throw new Error(`${op}: pred/target shape mismatch ${a.rows}x${a.cols} vs ${b.rows}x${b.cols}`);
  }
}

/** Binary classification accuracy with a 0.5 decision threshold. */
export function accuracy(pred: Matrix, target: Matrix): number {
  requireSameShape(pred, target, 'accuracy');
  const n = pred.data.length;
  if (n === 0) return 0;
  let correct = 0;
  for (let i = 0; i < n; i++) {
    const label = pred.data[i]! >= 0.5 ? 1 : 0;
    if (label === target.data[i]) correct++;
  }
  return correct / n;
}

/** Mean squared error metric (for regression readouts). */
export function mseMetric(pred: Matrix, target: Matrix): number {
  requireSameShape(pred, target, 'mseMetric');
  const n = pred.data.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const d = pred.data[i]! - target.data[i]!;
    sum += d * d;
  }
  return sum / n;
}
