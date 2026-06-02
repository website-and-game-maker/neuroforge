import type { Matrix } from '../engine/matrix';

/** A function that maps a batch of inputs [n×inDim] to model outputs [n×1]. */
export type Predictor = (X: Matrix) => Matrix;
