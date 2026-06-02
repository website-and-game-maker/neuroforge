import {
  Matrix,
  matmul,
  transpose,
  addRowVector,
  sumRows,
  hadamard,
  mapMatrix,
} from './matrix';
import type { Activation } from './activations';
import type { Rng } from './rng';

/**
 * A trainable parameter. The `value` Matrix identity is STABLE for the life of the
 * layer (its data buffer is updated in place by the optimizer), so optimizers may key
 * per-parameter state (e.g. momentum velocity) on it. `grad` is refreshed each
 * backward pass and read by the optimizer immediately after.
 */
export interface Param {
  value: Matrix;
  grad: Matrix;
}

export interface Layer {
  /** Forward pass; caches whatever backward needs. */
  forward(x: Matrix): Matrix;
  /** Backward pass: given dL/dY, return dL/dX and stash parameter grads. */
  backward(dY: Matrix): Matrix;
  /** Trainable parameters (empty for parameter-free layers). */
  params(): Param[];
}

/**
 * Fully-connected layer: y = x·W + b, with x = [batch×in], W = [in×out], b = [1×out].
 * Backward: dW = xᵀ·dY, db = colsum(dY), dX = dY·Wᵀ.
 */
export class Dense implements Layer {
  readonly W: Matrix; // [in×out]
  readonly b: Matrix; // [1×out]
  private lastX: Matrix | null = null;
  private readonly paramList: Param[];

  constructor(inDim: number, outDim: number, rng: Rng) {
    // Xavier/Glorot-style initialisation keeps activation variance stable across
    // layers, which matters for tanh/sigmoid nets.
    const std = Math.sqrt(2 / (inDim + outDim));
    const wData = new Float64Array(inDim * outDim);
    for (let i = 0; i < wData.length; i++) wData[i] = rng.gaussian() * std;
    this.W = new Matrix(inDim, outDim, wData);
    this.b = Matrix.zeros(1, outDim);
    this.paramList = [
      { value: this.W, grad: Matrix.zeros(inDim, outDim) },
      { value: this.b, grad: Matrix.zeros(1, outDim) },
    ];
  }

  forward(x: Matrix): Matrix {
    this.lastX = x;
    return addRowVector(matmul(x, this.W), this.b);
  }

  backward(dY: Matrix): Matrix {
    const x = this.lastX;
    if (!x) throw new Error('Dense.backward called before forward');
    this.paramList[0]!.grad = matmul(transpose(x), dY); // dW = xᵀ·dY  [in×out]
    this.paramList[1]!.grad = sumRows(dY); // db = colsum(dY)        [1×out]
    return matmul(dY, transpose(this.W)); // dX = dY·Wᵀ              [batch×in]
  }

  params(): Param[] {
    return this.paramList;
  }
}

/** Elementwise activation layer: y = f(x), dX = f'(x) ⊙ dY. */
export class ActivationLayer implements Layer {
  private lastX: Matrix | null = null;

  constructor(private readonly act: Activation) {}

  forward(x: Matrix): Matrix {
    this.lastX = x;
    return mapMatrix(x, this.act.f);
  }

  backward(dY: Matrix): Matrix {
    const x = this.lastX;
    if (!x) throw new Error('ActivationLayer.backward called before forward');
    const dfx = mapMatrix(x, this.act.df);
    return hadamard(dfx, dY);
  }

  params(): Param[] {
    return [];
  }
}
